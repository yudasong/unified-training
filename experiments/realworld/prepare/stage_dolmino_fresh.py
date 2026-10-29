#!/usr/bin/env python3
"""Download Dolmino shards the base model's midtraining corpus never used (the fresh 240M corpus source).

The midtraining corpus walked each component's sorted upstream file list from the front, so this walks
it from the back (file-level disjointness; the document-hash oracle verifies it later). Per-component byte
budgets follow the midtraining mixture. ``--files-from receipts/dolmino_fresh_stage_manifest.json``
downloads exactly the files the paper's corpus was built from. Resumable.

  python stage_dolmino_fresh.py --dest $DATA_DIR/dolmino-fresh/upstream \
      [--mixture receipts/dolmino_midtrain_mixture.json] [--files-from receipts/dolmino_fresh_stage_manifest.json]
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

REPO = "allenai/dolma3_dolmino_mix-100B-1025"
REVISION = "f23942ae8a8114af6e992efe8188ce8c531acd16"
RECEIPTS = Path(__file__).resolve().parent / "receipts"
TARGET_TOKENS = 360_000_000
BYTES_PER_TOKEN = 4.5  # ~bytes of text per GPT-2 token
MARGIN = 2.5           # hash rejects, zstd framing, length variance


def comp_root(path: str) -> str:
    d = path.split("/")[1]
    d = re.sub(r"_(19|20)_.*$", "", d)
    d = re.sub(r"-high_quality-.*$", "", d)
    d = re.sub(r"-fim.*$", "-fim", d) if d.startswith("stack_edu") else d
    return re.sub(r"-part\d+$", "", d)


def plan(weights: dict[str, float], sizes: dict[str, int], target_tokens: int = TARGET_TOKENS) -> dict:
    by_component: dict[str, list[str]] = collections.defaultdict(list)
    for f in sizes:
        by_component[comp_root(f)].append(f)
    missing = sorted(set(weights) - set(by_component))
    if missing:
        raise SystemExit(f"components with no upstream folder: {missing}")
    rows = {}
    for component, weight in sorted(weights.items(), key=lambda kv: -kv[1]):
        budget = int(weight * target_tokens * BYTES_PER_TOKEN * MARGIN)
        got, chosen = 0, []
        for f in sorted(by_component[component], reverse=True):  # backward walk = the untouched tail
            if got >= budget:
                break
            chosen.append(f)
            got += sizes[f]
        rows[component] = {"files": chosen, "bytes": got, "budget": budget}
    return rows


def main() -> None:
    from huggingface_hub import HfApi, hf_hub_download

    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--mixture", type=Path, default=RECEIPTS / "dolmino_midtrain_mixture.json")
    ap.add_argument("--files-from", type=Path, help="stage manifest whose file list to download verbatim")
    ap.add_argument("--target-tokens", type=int, default=TARGET_TOKENS)
    a = ap.parse_args()

    if a.files_from:
        rows = {c: {"files": f, "bytes": None, "budget": None}
                for c, f in json.loads(a.files_from.read_text())["files"].items()}
    else:
        weights = json.loads(a.mixture.read_text())["token_weights"]
        info = HfApi().dataset_info(REPO, revision=REVISION, files_metadata=True)  # one request for all sizes
        sizes = {s.rfilename: s.size or 0 for s in info.siblings
                 if s.rfilename.startswith("data/") and s.rfilename.endswith(".zst")}
        rows = plan(weights, sizes, a.target_tokens)
    a.dest.mkdir(parents=True, exist_ok=True)
    for component, row in rows.items():
        print(f"[stage] {component}: {len(row['files'])} files", flush=True)
        for f in row["files"]:
            hf_hub_download(REPO, f, repo_type="dataset", revision=REVISION, local_dir=a.dest)
    manifest = {
        "repo_id": REPO, "revision": REVISION,
        "selection": "sorted-file-list tail per component (front consumed by the midtraining corpus)",
        "target_tokens": a.target_tokens,
        "components": {k: {"n_files": len(v["files"]), "bytes": v["bytes"]} for k, v in rows.items()},
        "files": {k: v["files"] for k, v in rows.items()},
    }
    (a.dest / "stage_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
