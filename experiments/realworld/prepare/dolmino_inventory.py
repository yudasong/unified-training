#!/usr/bin/env python3
"""Inventory the base model's Dolmino midtraining corpus: document hashes + component mixture.

Input: the train-*.parquet shards (columns component, text) the base model was midtrained on.
Writes <out>/corpus_hashes.npy (sorted, deduplicated uint64 pairs = first 16 bytes of each document's
SHA-256: the disjointness oracle for the fresh corpus) and <out>/mixture.json (per-component GPT-2-token
weights; the paper's copy is receipts/dolmino_midtrain_mixture.json). Per-shard outputs are durable, so an
interrupted run resumes.

  CORPUS_SRC=<parquet dir> CORPUS_OUT=<dir> CORPUS_TOKENIZER=<hf dir> python dolmino_inventory.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HASH_DTYPE = np.dtype("<u8, <u8")


def doc_hash_pair(text: str) -> tuple[int, int]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little"), int.from_bytes(digest[8:16], "little")


def process_shard(args: tuple[str, str]) -> str:
    shard_path, out_dir = args
    shard, out = Path(shard_path), Path(out_dir)
    hash_file = out / f"{shard.stem}.hashes.npy"
    stats_file = out / f"{shard.stem}.stats.json"
    if hash_file.is_file() and stats_file.is_file():
        return f"skip {shard.name}"

    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(os.environ["CORPUS_TOKENIZER"], local_files_only=True)
    table = pq.read_table(shard, columns=["component", "text"])
    components = table.column("component").to_pylist()
    texts = table.column("text").to_pylist()

    hashes = np.empty(len(texts), dtype=HASH_DTYPE)
    stats: dict[str, dict[str, int]] = {}
    for i, (component, text) in enumerate(zip(components, texts)):
        hashes[i] = doc_hash_pair(text)
        entry = stats.setdefault(component, {"docs": 0, "bytes": 0, "gpt2_tokens": 0})
        entry["docs"] += 1
        entry["bytes"] += len(text.encode("utf-8"))
    for start in range(0, len(texts), 256):  # fast tokenizer, no special tokens, exact weights
        encoded = tokenizer(texts[start:start + 256], add_special_tokens=False)["input_ids"]
        for ids, component in zip(encoded, components[start:start + 256]):
            stats[component]["gpt2_tokens"] += len(ids)

    hashes.sort()
    tmp = hash_file.with_name(hash_file.name + ".tmp.npy")
    with open(tmp, "wb") as handle:
        np.save(handle, hashes)
    tmp.rename(hash_file)
    stats_file.write_text(json.dumps(stats, sort_keys=True))
    return f"done {shard.name}: {len(texts)} docs"


def merge(shards: list[Path], out: Path) -> dict:
    merged = np.concatenate([np.load(out / "shards" / f"{s.stem}.hashes.npy") for s in shards])
    merged.sort()
    before = len(merged)
    merged = np.unique(merged)
    np.save(out / "corpus_hashes.npy", merged)
    totals: dict[str, dict[str, int]] = {}
    for s in shards:
        for component, entry in json.loads((out / "shards" / f"{s.stem}.stats.json").read_text()).items():
            t = totals.setdefault(component, {"docs": 0, "bytes": 0, "gpt2_tokens": 0})
            for key, value in entry.items():
                t[key] += value
    grand = sum(v["gpt2_tokens"] for v in totals.values())
    mixture = {
        "components": totals,
        "total_gpt2_tokens": grand,
        "total_docs": before,
        "duplicate_texts_within_corpus": before - len(merged),
        "token_weights": {k: v["gpt2_tokens"] / grand for k, v in sorted(totals.items())},
    }
    (out / "mixture.json").write_text(json.dumps(mixture, indent=2, sort_keys=True))
    return mixture


def main() -> None:
    source, out = Path(os.environ["CORPUS_SRC"]), Path(os.environ["CORPUS_OUT"])
    workers = int(os.environ.get("CORPUS_WORKERS", "32"))
    (out / "shards").mkdir(parents=True, exist_ok=True)
    shards = sorted(source.glob("train-*.parquet"))
    if not shards:
        raise SystemExit(f"no shards under {source}")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for message in pool.map(process_shard, [(str(s), str(out / "shards")) for s in shards]):
            print(f"[inventory] {message}", flush=True)
    mixture = merge(shards, out)
    print(f"[inventory] {mixture['total_docs']} docs, {mixture['total_gpt2_tokens']:,} gpt2 tokens", flush=True)


if __name__ == "__main__":
    sys.exit(main())
