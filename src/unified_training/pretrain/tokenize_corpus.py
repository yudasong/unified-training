"""Tokenize a pretraining corpus into the paper's GPT-2 Megatron shards, following a shard plan.

    python -m unified_training.pretrain.tokenize_corpus PLAN.json.gz SOURCE_ROOT OUT_DIR [--shards 0,25,23]
    python -m unified_training.pretrain.tokenize_corpus PLAN.json.gz --list-files [--shards ...]   # files to download

A plan (scripts/pretrain/plans/) lists, for every output shard, the source files (relative to SOURCE_ROOT,
a local copy of the plan's ``source_repo`` Hugging Face dataset) in processing order, and the document
count, token count and xxh64 checksums of the shard as it was built for the paper. Each document is
GPT-2 ``text`` tokens + EOS (50256), stored as uint16; rows with a null ``text`` are skipped. Writes
OUT_DIR/shard_NNNNNN.{bin,idx} and a receipt; with ``xxhash`` installed the receipt says whether the
shard is byte-identical to the paper's. Shards are independent: run several processes on disjoint
``--shards`` lists. CPU only.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from unified_training.pretrain.indexed import IndexedDatasetBuilder

TOKENIZE_BATCH = 1000


def load_plan(path: str | Path) -> dict:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as stream:
        return json.load(stream)


def iter_texts(path: Path, text_field: str) -> Iterator[str]:
    """Texts of one source file in stored order (parquet, .jsonl, .jsonl.zst)."""
    name = path.name
    if name.endswith(".parquet"):
        import pyarrow.parquet as pq

        for batch in pq.ParquetFile(path).iter_batches(columns=[text_field], batch_size=10000):
            for text in batch.column(text_field).to_pylist():
                if text is not None:
                    yield str(text)
        return
    if name.endswith(".zst"):
        import zstandard

        raw = zstandard.ZstdDecompressor().stream_reader(open(path, "rb"))
        lines = io.TextIOWrapper(raw, encoding="utf-8")
    elif name.endswith((".jsonl", ".json")):
        lines = open(path, encoding="utf-8")
    else:
        raise ValueError(f"unsupported source file: {path}")
    with lines:
        for line in lines:
            if line.strip():
                text = json.loads(line).get(text_field)
                if text is not None:
                    yield str(text)


def make_tokenize(model: str, revision: str | None, add_eos: bool = True):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    eos = tokenizer.eos_token_id

    def tokenize(texts: list[str]) -> list[list[int]]:
        encoded = tokenizer(texts, add_special_tokens=False, return_attention_mask=False,
                            return_token_type_ids=False)["input_ids"]
        return [list(ids) + ([eos] if add_eos else []) for ids in encoded]

    return tokenize


def tokenize_shard(plan: dict, index: int, source_root: Path, out_dir: Path, tokenize) -> dict:
    spec = plan["shards"][index]
    prefix = out_dir / f"shard_{index:06d}"
    receipt_path = Path(str(prefix) + ".receipt.json")
    if receipt_path.exists():
        return json.loads(receipt_path.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    partial_bin, partial_idx = Path(str(prefix) + ".bin.partial"), Path(str(prefix) + ".idx.partial")
    with open(partial_bin, "wb") as stream:
        builder = IndexedDatasetBuilder(stream, dtype=np.dtype(plan["dtype"]))
        for relative, _size in spec["files"]:
            batch: list[str] = []
            for text in iter_texts(source_root / relative, plan["text_field"]):
                batch.append(text)
                if len(batch) >= TOKENIZE_BATCH:
                    for ids in tokenize(batch):
                        builder.add_document(ids)
                    batch = []
            for ids in tokenize(batch) if batch else []:
                builder.add_document(ids)
        builder.finalize()
    with open(partial_idx, "wb") as stream:
        builder.write_index(stream)
    partial_bin.rename(str(prefix) + ".bin")
    partial_idx.rename(str(prefix) + ".idx")
    receipt = {"shard": index, "documents": len(builder.sequence_lengths), "tokens": builder.total_tokens,
               "expected_documents": spec["documents"], "expected_tokens": spec["tokens"]}
    try:
        import xxhash

        for suffix in ("bin", "idx"):
            digest = xxhash.xxh64()
            with open(str(prefix) + "." + suffix, "rb") as stream:
                for block in iter(lambda: stream.read(1 << 24), b""):
                    digest.update(block)
            receipt[f"{suffix}_xxh64"] = f"xxh64:{digest.hexdigest()}"
        receipt["identical_to_paper"] = (receipt["bin_xxh64"] == spec["bin_xxh64"]
                                         and receipt["idx_xxh64"] == spec["idx_xxh64"])
    except ImportError:
        receipt["identical_to_paper"] = None
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("plan")
    parser.add_argument("source_root", type=Path, nargs="?")
    parser.add_argument("out_dir", type=Path, nargs="?")
    parser.add_argument("--list-files", action="store_true", help="print the source files the plan reads and exit")
    parser.add_argument("--shards", default=None, help="comma list of shard indices (default: all)")
    parser.add_argument("--tokenizer", default=None, help="tokenizer dir or hub id (default: the plan's)")
    args = parser.parse_args(argv)
    plan = load_plan(args.plan)
    indices = range(plan["num_shards"]) if args.shards is None else [int(i) for i in args.shards.split(",")]
    if args.list_files:
        print("\n".join(relative for i in indices for relative, _size in plan["shards"][i]["files"]))
        return
    if args.source_root is None or args.out_dir is None:
        parser.error("SOURCE_ROOT and OUT_DIR are required")
    tok = plan["tokenizer"]
    tokenize = make_tokenize(args.tokenizer or tok["model"], None if args.tokenizer else tok["revision"], tok["add_eos"])
    for index in indices:
        receipt = tokenize_shard(plan, index, args.source_root, args.out_dir, tokenize)
        print(json.dumps(receipt), flush=True)
        if receipt["documents"] != receipt["expected_documents"] or receipt["tokens"] != receipt["expected_tokens"]:
            print(f"WARNING: shard {index} differs from the paper's counts", flush=True)


if __name__ == "__main__":
    main()
