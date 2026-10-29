#!/usr/bin/env python3
"""Held-out midtraining loss slices (Appendix D.1): 24 components x ~150k GPT-2 tokens, 7,428 documents,
3,608,122 scored tokens in the paper's copy (receipts/heldout_slices_manifest.json).

stage: continue each component's backward walk over the Dolmino file list past the files the fresh 240M
       corpus staged (file-level disjoint from both corpora); receipts/heldout_slices_stage_manifest.json
       lists the paper's files (--files-from).
build: keep documents whose SHA-256 prefix is in neither the midtraining corpus oracle nor the fresh
       corpus, >= 32 tokens, truncated to 2048 tokens, until each component has >= 150k tokens; write
       <component>.npz (int32 tokens + int64 offsets, one document per entry) and manifest.json, the
       input of `python -m unified_training.evals holdout-loss --slices`.
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dolmino_fresh import doc_hash_pair  # noqa: E402
from stage_dolmino_fresh import RECEIPTS, REPO, REVISION, comp_root  # noqa: E402

TOKENS_PER_COMPONENT = 150_000
MAX_DOC_TOKENS = 2048
MIN_DOC_TOKENS = 32
BYTES_PER_TOKEN, MARGIN = 4.5, 4.0


def stage(a) -> None:
    from huggingface_hub import HfApi, hf_hub_download

    if a.files_from:
        plan = {c: f for c, f in json.loads(a.files_from.read_text())["files"].items()}
    else:
        weights = json.loads(a.mixture.read_text())["token_weights"]
        taken = {f for files in json.loads(a.fresh_stage_manifest.read_text())["files"].values() for f in files}
        info = HfApi().dataset_info(REPO, revision=REVISION, files_metadata=True)
        sizes = {s.rfilename: s.size or 0 for s in info.siblings
                 if s.rfilename.startswith("data/") and s.rfilename.endswith(".zst")}
        by_component = collections.defaultdict(list)
        for f in sizes:
            by_component[comp_root(f)].append(f)
        budget = int(TOKENS_PER_COMPONENT * BYTES_PER_TOKEN * MARGIN)
        plan = {}
        for component in sorted(weights):
            got, chosen = 0, []
            for f in sorted(by_component[component], reverse=True):  # same backward walk, skip fresh files
                if f in taken:
                    continue
                if got >= budget:
                    break
                chosen.append(f)
                got += sizes[f]
            plan[component] = chosen
    a.dest.mkdir(parents=True, exist_ok=True)
    for component, files in plan.items():
        for f in files:
            hf_hub_download(REPO, f, repo_type="dataset", revision=REVISION, local_dir=a.dest)
    (a.dest / "stage_manifest.json").write_text(json.dumps(
        {"repo_id": REPO, "revision": REVISION, "tokens_per_component": TOKENS_PER_COMPONENT,
         "selection": "tail continuation past the fresh-corpus staging (files it took are skipped)",
         "files": plan}, indent=2, sort_keys=True))


def build(a) -> None:
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    oracle = np.load(a.oracle)
    fresh = np.array([doc_hash_pair(t) for shard in sorted(a.fresh_corpus.glob("data/train-*.parquet"))
                      for t in pq.read_table(shard, columns=["text"])["text"].to_pylist()], dtype=oracle.dtype)
    fresh.sort()

    def consumed(pair) -> bool:
        key = np.array([pair], dtype=oracle.dtype)[0]
        for ref in (oracle, fresh):
            pos = np.searchsorted(ref, key)
            if pos < len(ref) and ref[pos] == key:
                return True
        return False

    tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    files_by_component = json.loads((a.stage / "stage_manifest.json").read_text())["files"]
    a.out.mkdir(parents=True, exist_ok=True)
    stats = {}
    for component, files in sorted(files_by_component.items()):
        tokens, offsets = [], [0]
        scanned = rejected = truncated = 0
        for rel in files:
            path = a.stage / rel
            if not path.exists():
                continue
            proc = subprocess.Popen(["zstd", "-dc", str(path)], stdout=subprocess.PIPE)
            for line in io.TextIOWrapper(proc.stdout, encoding="utf-8", errors="replace"):
                if offsets[-1] >= TOKENS_PER_COMPONENT:
                    break
                try:
                    text = json.loads(line).get("text") or ""
                except json.JSONDecodeError:
                    continue
                if not text.strip():
                    continue
                scanned += 1
                if consumed(doc_hash_pair(text)):
                    rejected += 1
                    continue
                ids = tok(text, add_special_tokens=False).input_ids
                if len(ids) < MIN_DOC_TOKENS:
                    continue
                if len(ids) > MAX_DOC_TOKENS:
                    ids, truncated = ids[:MAX_DOC_TOKENS], truncated + 1
                tokens.append(np.asarray(ids, dtype=np.int32))
                offsets.append(offsets[-1] + len(ids))
            proc.stdout.close()
            proc.wait()
            if offsets[-1] >= TOKENS_PER_COMPONENT:
                break
        stream = np.concatenate(tokens) if tokens else np.zeros(0, dtype=np.int32)
        np.savez(a.out / f"{component}.npz", tokens=stream, offsets=np.asarray(offsets, dtype=np.int64))
        stats[component] = {"docs": len(tokens), "tokens": int(offsets[-1]), "scanned": scanned,
                            "rejected_consumed_or_anchor": rejected, "truncated": truncated}
        print(f"[slices] {component}: {len(tokens)} docs, {offsets[-1]:,} tokens", flush=True)
    (a.out / "manifest.json").write_text(json.dumps({
        "construction": "upstream tail past the fresh-corpus staging; documents absent from the midtraining "
                        "corpus and the fresh corpus",
        "target_tokens_per_component": TOKENS_PER_COMPONENT, "max_doc_tokens": MAX_DOC_TOKENS,
        "mixture_token_weights": json.loads(a.mixture.read_text())["token_weights"], "components": stats,
    }, indent=2, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage")
    s.add_argument("--dest", type=Path, required=True)
    s.add_argument("--files-from", type=Path)
    s.add_argument("--fresh-stage-manifest", type=Path, default=RECEIPTS / "dolmino_fresh_stage_manifest.json")
    s.add_argument("--mixture", type=Path, default=RECEIPTS / "dolmino_midtrain_mixture.json")
    b = sub.add_parser("build")
    b.add_argument("--stage", type=Path, required=True)
    b.add_argument("--oracle", type=Path, required=True)
    b.add_argument("--fresh-corpus", type=Path, required=True, help="output dir of build_dolmino_fresh.py")
    b.add_argument("--tokenizer", required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--mixture", type=Path, default=RECEIPTS / "dolmino_midtrain_mixture.json")
    a = ap.parse_args()
    stage(a) if a.cmd == "stage" else build(a)


if __name__ == "__main__":
    main()
