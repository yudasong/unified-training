#!/usr/bin/env python3
"""Fresh, disjoint midtraining-mixture source corpus from the staged Dolmino shards.

Drops every document whose SHA-256 prefix is in the hash oracle of the base model's midtraining corpus
(dolmino_inventory.py), counts GPT-2 tokens of the survivors, and fills each component up to its share of
360M tokens (midtraining mixture), writing train-*.parquet shards of (component, text) -- the input of the
document-anchor builder -- plus source_manifest.json (paper's copy: receipts/dolmino_fresh_source_manifest.json).
Needs the `zstd` binary.

  python build_dolmino_fresh.py --stage <stage dir> --oracle <corpus_hashes.npy> --tokenizer <hf dir> --out <dir>
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
from pathlib import Path

import numpy as np

RECEIPTS = Path(__file__).resolve().parent / "receipts"
SHARD_ROWS = 100_000


def doc_hash_pair(text: str) -> tuple[int, int]:
    import hashlib

    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little"), int.from_bytes(digest[8:16], "little")


def in_oracle(oracle: np.ndarray, text: str) -> bool:
    pair = np.array([doc_hash_pair(text)], dtype=oracle.dtype)[0]
    pos = np.searchsorted(oracle, pair)
    return bool(pos < len(oracle) and oracle[pos] == pair)


def main() -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=Path, required=True)
    ap.add_argument("--oracle", type=Path, required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mixture", type=Path, default=RECEIPTS / "dolmino_midtrain_mixture.json")
    ap.add_argument("--target-tokens", type=int, default=360_000_000)
    a = ap.parse_args()

    oracle = np.load(a.oracle)  # sorted structured (<u8, <u8)
    weights = json.loads(a.mixture.read_text())["token_weights"]
    manifest = json.loads((a.stage / "stage_manifest.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    (a.out / "data").mkdir(parents=True, exist_ok=True)
    quotas = {c: int(w * a.target_tokens) for c, w in weights.items()}
    stats: dict[str, dict[str, int]] = {}
    rows_component: list[str] = []
    rows_text: list[str] = []
    shard_index = 0

    def flush(force: bool = False) -> None:
        nonlocal shard_index, rows_component, rows_text
        while len(rows_text) >= SHARD_ROWS or (force and rows_text):
            take = min(SHARD_ROWS, len(rows_text))
            table = pa.table({"component": rows_component[:take], "text": rows_text[:take]})
            pq.write_table(table, a.out / "data" / f"train-{shard_index:05d}.parquet")
            rows_component, rows_text = rows_component[take:], rows_text[take:]
            shard_index += 1

    for component, files in sorted(manifest["files"].items()):
        quota = quotas.get(component)
        if quota is None:
            continue
        got = kept = 0
        st = stats[component] = {"kept_docs": 0, "kept_tokens": 0, "excluded_consumed": 0, "scanned": 0}
        batch: list[str] = []

        def take_batch() -> None:
            nonlocal got, kept
            survivors = []
            for text in batch:
                st["scanned"] += 1
                if in_oracle(oracle, text):
                    st["excluded_consumed"] += 1
                else:
                    survivors.append(text)
            batch.clear()
            if not survivors:
                return
            encoded = tokenizer(survivors, add_special_tokens=False)["input_ids"]
            for text, ids in zip(survivors, encoded):
                if got >= quota:
                    break
                rows_component.append(component)
                rows_text.append(text)
                got += len(ids)
                kept += 1
            flush()

        for rel in files:
            if got >= quota:
                break
            with subprocess.Popen(["zstd", "-dc", str(a.stage / rel)], stdout=subprocess.PIPE) as proc:
                for line in io.TextIOWrapper(proc.stdout, encoding="utf-8", errors="strict"):
                    if got >= quota:
                        break
                    try:
                        text = json.loads(line).get("text") or ""
                    except json.JSONDecodeError:
                        continue
                    if not text.strip():
                        continue
                    batch.append(text)
                    if len(batch) >= 512:
                        take_batch()
            take_batch()
        st["kept_docs"], st["kept_tokens"] = kept, got
        print(f"[fresh] {component}: kept {kept} docs / {got / 1e6:.1f}M tokens (quota {quota / 1e6:.1f}M), "
              f"excluded {st['excluded_consumed']} consumed", flush=True)
        if got < quota * 0.9:
            raise SystemExit(f"component under-filled: {component} {got} < 0.9x{quota}; stage more files")
    flush(force=True)

    total = sum(s["kept_tokens"] for s in stats.values())
    source_manifest = {
        "repo_id": manifest["repo_id"], "revision": manifest["revision"],
        "derived": "fresh-disjoint sample; every document hash-verified absent from the midtraining corpus",
        "total_kept_tokens": total,
        "realized_weights": {c: s["kept_tokens"] / total for c, s in sorted(stats.items())},
        "stats": stats,
    }
    (a.out / "source_manifest.json").write_text(json.dumps(source_manifest, indent=2, sort_keys=True))
    print(f"[fresh] {total / 1e6:.1f}M tokens, {shard_index} shards", flush=True)


if __name__ == "__main__":
    main()
