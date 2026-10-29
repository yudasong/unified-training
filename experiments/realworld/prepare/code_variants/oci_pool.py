#!/usr/bin/env python3
"""OpenCodeInstruct coding pool: the candidate supply of the code-replacement corpus (App. D.4).

Phases (each refuses to overwrite its output):
  stage     download and pin nvidia/OpenCodeInstruct shards 0-9 of 50 at the paper's revision
  build     average_test_score=1 rows rendered as a contract prompt (question + up to three of its own
            assert-style tests calling a function the AST-parseable solution defines), simple ChatML,
            prompt weight 0 / answer + <|im_end|> weight 1, complete conversations of <= 2049 stored tokens,
            exact-conversation and prompt exact/near (Jaccard >= 0.92) deduplication
  filter    drop the rows the paper's pre-filter benchmark scan matched (pinned receipt) and write the
            decontaminated pool
  finalize  integrity checks plus the post-filter scans (exact and rarest-16-token), then READY

The paper's pool also held three other sources processed ahead of OpenCodeInstruct; none of their keys
deduplicated an OpenCodeInstruct row, so this single-source build retains the same 388,940 rows in the same
order (episode ids are the paper's minus 26,131).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from common import load_receipt, sha
from unified_training.episodes import INDEX_DTYPE
from oci_records import (
    SUPERVISED_TOKEN_COUNT_BASIS,
    V6_MAX_STORED_TOKENS,
    VERIFIED_OPENCODE_COMPONENT,
    PromptDeduplicator,
    conversation_sha256,
    normalized_prompt_sha256,
    opencodeinstruct_contract_messages,
    prompt_sha256,
    simple_chatml_chunks,
)

REPO_ID = "nvidia/OpenCodeInstruct"
REVISION = "8f3ba5bafe4d6e8db46082cf7ae6741bc370604d"
FILES = tuple(f"data/train-{shard:05d}-of-00050.parquet" for shard in range(10))
SOURCE_NAME = "opencodeinstruct"
EXPECTED_ROWS = 1_000_000
COLUMNS = ("input", "output", "unit_tests", "average_test_score")
LICENSE = {"id": "CC-BY-4.0", "usage_policy": "research",
           "note": "Declared by the nvidia/OpenCodeInstruct dataset card."}
POOL_KIND = "verified_reasoning_v6_source_pool"


# ------------------------------------------------------------------ stage

def stage(args) -> None:
    from huggingface_hub import HfApi, hf_hub_download

    out = args.sources
    if (out / "READY").exists():
        verify_staging(out)
        print(f"OCI_SOURCES_ALREADY_READY: {out}")
        return
    info = HfApi().dataset_info(REPO_ID, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise ValueError(f"{REPO_ID}@{REVISION} resolved to {info.sha}")
    siblings = {s.rfilename: s for s in info.siblings}
    files = []
    for name in FILES:
        s = siblings[name]
        path = Path(hf_hub_download(repo_id=REPO_ID, repo_type="dataset", revision=REVISION, filename=name,
                                    local_dir=out / SOURCE_NAME))
        if path.stat().st_size != s.size or (s.lfs is not None and sha(path) != s.lfs.sha256):
            raise IOError(f"download does not match the pinned object: {name}")
        files.append({"path": name, "bytes": int(s.size), "git_blob_sha1": s.blob_id,
                      "lfs_sha256": s.lfs.sha256 if s.lfs else None, "sha256": sha(path)})
    rows = sum(pq.ParquetFile(out / SOURCE_NAME / f["path"]).metadata.num_rows for f in files)
    if rows != EXPECTED_ROWS:
        raise ValueError(f"{rows} rows staged, expected {EXPECTED_ROWS}")
    manifest = {"schema_version": 1, "kind": "verified_reasoning_sft_sources", "status": "ready", "recipe": "v6",
                "sources": [{"name": SOURCE_NAME, "repo_id": REPO_ID, "revision": REVISION, "rows": rows,
                             "license": LICENSE, "files": files}]}
    (out / "source_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (out / "READY").touch()
    print(f"OCI_SOURCES_READY: {out}")


def verify_staging(root: Path) -> dict:
    manifest = json.loads((root / "source_manifest.json").read_text())
    (record,) = manifest["sources"]
    assert record["repo_id"] == REPO_ID and record["revision"] == REVISION
    assert [f["path"] for f in record["files"]] == list(FILES)
    for f in record["files"]:
        if sha(root / SOURCE_NAME / f["path"]) != f["sha256"]:
            raise ValueError(f"local source content changed: {f['path']}")
    return manifest


# ------------------------------------------------------------------ build

def iter_records(root: Path, batch_rows: int):
    """Source rows in pinned shard order and physical Parquet row order; the row index is the identity."""
    row_index = 0
    for name in FILES:
        path = root / SOURCE_NAME / name
        for batch in pq.ParquetFile(path).iter_batches(columns=list(COLUMNS), batch_size=batch_rows,
                                                       use_threads=False):
            for record in batch.to_pylist():
                current = row_index
                row_index += 1
                yield current, f"{SOURCE_NAME}:{current}", SOURCE_NAME, f"{SOURCE_NAME}/{name}", record


def build(args) -> None:
    from transformers import AutoTokenizer

    if args.raw.exists():
        raise FileExistsError(args.raw)
    verify_staging(args.sources)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    tokenizer.chat_template = args.chat_template.read_text()
    probe = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]
    if tokenizer.apply_chat_template(probe, tokenize=False, add_generation_prompt=False) != "".join(
            simple_chatml_chunks(probe)):
        raise ValueError("hard-coded simple-ChatML chunks differ from the template")
    args.raw.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.raw.name}.staging-", dir=args.raw.parent))
    stats, seen = Counter(), {}
    dedup = PromptDeduplicator(near_jaccard_threshold=0.92)
    index_rows, offset, episode, supervised = [], 0, 0, 0
    schema = pa.schema([("episode", pa.int64()), ("component", pa.string()), ("source_subset", pa.string()),
                        ("source_file", pa.string()), ("source_row_index", pa.int64()), ("source_id", pa.string()),
                        ("verification_kind", pa.string()), ("conversation_sha256", pa.string()),
                        ("prompt_sha256", pa.string()), ("normalized_prompt_sha256", pa.string()),
                        ("assistant_response_sha256", pa.string()), ("total_tokens", pa.int32()),
                        ("assistant_loss_tokens", pa.int32()), ("truncated", pa.bool_()),
                        ("original_total_tokens", pa.int32())])
    writer = pq.ParquetWriter(staging / "provenance.parquet", schema, compression="zstd")
    pending = []
    try:
        with (staging / "tokens.bin").open("xb") as tf, (staging / "weights.bin").open("xb") as wf, \
                (staging / "duplicates.jsonl").open("x") as df:
            def flush():
                nonlocal offset, episode, supervised
                if not pending:
                    return
                rendered = [simple_chatml_chunks(item[4]) for item in pending]
                enc = lambda texts: tokenizer(texts, add_special_tokens=False, padding=False,  # noqa: E731
                                              truncation=False)["input_ids"]
                rows = []
                for item, p_ids, r_ids in zip(pending, enc([r[0] for r in rendered]), enc([r[1] for r in rendered]),
                                              strict=True):
                    source_row, source_id, subset, source_file, messages, conv_hash, p_hash = item
                    total = len(p_ids) + len(r_ids)
                    if not r_ids:
                        stats["empty_response"] += 1
                        continue
                    if total > V6_MAX_STORED_TOKENS:  # code is never truncated
                        stats["overlong"] += 1
                        continue
                    owner = seen.get(conv_hash)
                    if owner is not None:
                        kind, owner_parts = "exact_conversation", owner
                    else:
                        found = dedup.find_duplicate(messages[0]["content"])
                        kind, owner_parts = (None, None) if found is None else (found[0], tuple(found[1:]))
                    if kind is not None:
                        stats["duplicate"] += 1
                        stats[f"duplicate_{kind}"] += 1
                        df.write(json.dumps({"kind": kind, "conversation_sha256": conv_hash,
                                             "normalized_prompt_sha256": normalized_prompt_sha256(messages[0]["content"]),
                                             "duplicate_component": VERIFIED_OPENCODE_COMPONENT,
                                             "duplicate_source_id": source_id, "owner_component": owner_parts[0],
                                             "owner_source_id": owner_parts[1]}, sort_keys=True) + "\n")
                        continue
                    tokens = np.asarray([*p_ids, *r_ids], dtype="<i4")
                    weights = np.concatenate((np.zeros(len(p_ids), dtype="<f4"), np.ones(len(r_ids), dtype="<f4")))
                    tokens.tofile(tf)
                    weights.tofile(wf)
                    index_rows.append((offset, len(p_ids), total, 0.0, 1.0, -1, 0))
                    rows.append({"episode": episode, "component": VERIFIED_OPENCODE_COMPONENT, "source_subset": subset,
                                 "source_file": source_file, "source_row_index": source_row, "source_id": source_id,
                                 "verification_kind": "opencodeinstruct_unit_tests_and_python_ast",
                                 "conversation_sha256": conv_hash, "prompt_sha256": p_hash,
                                 "normalized_prompt_sha256": normalized_prompt_sha256(messages[0]["content"]),
                                 "assistant_response_sha256": hashlib.sha256(
                                     messages[1]["content"].encode("utf-8")).hexdigest(),
                                 "total_tokens": total, "assistant_loss_tokens": len(r_ids),
                                 "truncated": False, "original_total_tokens": total})
                    seen[conv_hash] = (VERIFIED_OPENCODE_COMPONENT, source_id)
                    dedup.add(messages[0]["content"], component=VERIFIED_OPENCODE_COMPONENT, source_id=source_id)
                    offset += total
                    episode += 1
                    supervised += len(r_ids)
                    stats["retained"] += 1
                    stats["retained_complete"] += 1
                if rows:
                    writer.write_table(pa.Table.from_pylist(rows, schema=schema))
                pending.clear()

            for source_row, source_id, subset, source_file, record in iter_records(args.sources, args.batch_rows):
                stats["source_rows"] += 1
                try:
                    messages = opencodeinstruct_contract_messages(record)
                    ast.parse(messages[1]["content"])
                except SyntaxError:
                    stats["invalid_python_ast"] += 1
                    continue
                except (TypeError, ValueError):
                    stats["verification_rejected"] += 1
                    continue
                stats["verified_candidates"] += 1
                pending.append((source_row, source_id, subset, source_file, messages, conversation_sha256(messages),
                                prompt_sha256(messages)))
                if len(pending) >= args.tokenize_batch:
                    flush()
            flush()
        writer.close()
        if stats["source_rows"] != EXPECTED_ROWS:
            raise ValueError(f"scanned {stats['source_rows']} rows, expected {EXPECTED_ROWS}")
        index = np.asarray(index_rows, dtype=INDEX_DTYPE)
        np.save(staging / "index.npy", index)
        manifest = {
            "schema_version": 1, "kind": POOL_KIND, "recipe": "v6", "stage": "length_filtered_deduplicated_raw",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_manifest_sha256": sha(args.sources / "source_manifest.json"),
            "source_revisions": {REPO_ID: REVISION}, "source_licenses": {REPO_ID: LICENSE},
            "tokenizer_sha256": sha(Path(args.tokenizer) / "tokenizer.json"),
            "chat_template_sha256": sha(args.chat_template), "window": V6_MAX_STORED_TOKENS,
            "max_causal_positions": V6_MAX_STORED_TOKENS - 1, "overlong_policy": "drop_complete_conversation",
            "mask_convention": "raw token-target mask; prompt=0, assistant+im_end=1",
            "supervised_token_count_basis": SUPERVISED_TOKEN_COUNT_BASIS,
            "deduplication": {"method": "global_exact_conversation_plus_prompt_exact_near",
                              "prompt_near_jaccard_threshold": 0.92, "duplicates_file": "duplicates.jsonl",
                              "duplicates_sha256": sha(staging / "duplicates.jsonl")},
            "decontamination": {"benchmark": None, "post_filter_zero_matches": False},
            "component_group_ids": {VERIFIED_OPENCODE_COMPONENT: 0},
            "component_mix_weights": {VERIFIED_OPENCODE_COMPONENT: 1.0},
            "component_build_stats": {VERIFIED_OPENCODE_COMPONENT: dict(stats)},
            "component_realized_episodes": {VERIFIED_OPENCODE_COMPONENT: len(index)},
            "component_realized_tokens": {VERIFIED_OPENCODE_COMPONENT: int(offset)},
            "component_realized_supervised_tokens": {VERIFIED_OPENCODE_COMPONENT: int(supervised)},
            "component_realized_weighted_tokens": {VERIFIED_OPENCODE_COMPONENT: int(supervised)},
            "n_episodes": len(index), "n_tokens": int(offset), "n_supervised_tokens": int(supervised),
            "n_weighted_tokens": int(supervised),
            "max_token_id": int(np.memmap(staging / "tokens.bin", dtype="<i4", mode="r").max()),
            "artifacts": {k: {"file": f, "sha256": sha(staging / f)} for k, f in
                          (("tokens", "tokens.bin"), ("weights", "weights.bin"), ("index", "index.npy"))},
            "provenance_file": "provenance.parquet", "provenance_sha256": sha(staging / "provenance.parquet"),
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        (staging / "DONE").touch()
        os.rename(staging, args.raw)
    except BaseException:
        writer.close()
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"OCI_POOL_RAW_READY: {dict(stats)} supervised_tokens={supervised:,} -> {args.raw}", flush=True)


# ------------------------------------------------------------------ filter

def filter_pool(args) -> None:
    """Drop the pinned decontamination rows; the provenance episode column is renumbered."""
    if args.pool.exists():
        raise FileExistsError(args.pool)
    manifest = json.loads((args.raw / "manifest.json").read_text())
    assert manifest["stage"] == "length_filtered_deduplicated_raw" and (args.raw / "DONE").is_file()
    index = np.load(args.raw / "index.npy", mmap_mode="r")
    tokens = np.memmap(args.raw / "tokens.bin", dtype="<i4", mode="r")
    weights = np.memmap(args.raw / "weights.bin", dtype="<f4", mode="r")
    for rec in manifest["artifacts"].values():
        assert sha(args.raw / rec["file"]) == rec["sha256"]
    prov = pq.read_table(args.raw / "provenance.parquet", use_threads=False)
    pinned = {r["source_row_index"]: r["conversation_sha256"] for r in load_receipt("oci_decontamination_drops.json")["rows"]}
    rows = prov.column("source_row_index").to_pylist()
    hashes = prov.column("conversation_sha256").to_pylist()
    keep = np.ones(len(index), dtype=bool)
    for episode, (row, digest) in enumerate(zip(rows, hashes)):
        if row in pinned:
            if pinned[row] != digest:
                raise ValueError(f"pinned drop {row} has a different conversation than recorded")
            keep[episode] = False
    if int((~keep).sum()) != len(pinned):
        raise ValueError(f"found {int((~keep).sum())} of {len(pinned)} pinned decontamination rows")
    old_to_new = np.full(len(index), -1, dtype=np.int64)
    old_to_new[keep] = np.arange(int(keep.sum()))
    args.pool.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.pool.name}.staging-", dir=args.pool.parent))
    out_index = np.empty(int(keep.sum()), dtype=INDEX_DTYPE)
    off = tok_total = sup_total = 0
    try:
        with (staging / "tokens.bin").open("wb") as tf, (staging / "weights.bin").open("wb") as wf:
            j = 0
            for i, row in enumerate(index):
                if not keep[i]:
                    continue
                lo, n = int(row["offset"]), int(row["total_len"])
                t = np.asarray(tokens[lo:lo + n], dtype="<i4")
                w = np.asarray(weights[lo:lo + n], dtype="<f4")
                t.tofile(tf)
                w.tofile(wf)
                out_index[j] = (off, int(row["prompt_len"]), n, float(row["reward"]), float(row["weight"]),
                                int(row["policy_version"]), int(row["group_id"]))
                off += n
                tok_total += n
                sup_total += int(np.count_nonzero(w > 0))
                j += 1
        np.save(staging / "index.npy", out_index)
        table = prov.filter(pa.array(keep))
        table = table.set_column(table.schema.get_field_index("episode"), "episode",
                                 pa.array(old_to_new[np.asarray(table["episode"])], type=pa.int64()))
        pq.write_table(table, staging / "provenance.parquet", compression="zstd")
        shutil.copy2(args.raw / "duplicates.jsonl", staging / "duplicates.jsonl")
        manifest.update(stage="decontaminated_unverified", n_episodes=len(out_index), n_tokens=off,
                        n_supervised_tokens=sup_total, n_weighted_tokens=sup_total,
                        component_realized_episodes={VERIFIED_OPENCODE_COMPONENT: len(out_index)},
                        component_realized_tokens={VERIFIED_OPENCODE_COMPONENT: tok_total},
                        component_realized_supervised_tokens={VERIFIED_OPENCODE_COMPONENT: sup_total},
                        component_realized_weighted_tokens={VERIFIED_OPENCODE_COMPONENT: sup_total},
                        provenance_sha256=sha(staging / "provenance.parquet"))
        manifest["artifacts"] = {k: {"file": f, "sha256": sha(staging / f)} for k, f in
                                 (("tokens", "tokens.bin"), ("weights", "weights.bin"), ("index", "index.npy"))}
        manifest["decontamination"] = {"benchmark": "broad16-gsm8k", "method": "drop_complete_matching_episode",
                                       "drops": "pinned receipt of the paper's pre-filter scan",
                                       "n_dropped_episodes": len(pinned), "post_filter_zero_matches": False}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        (staging / "DONE").touch()
        os.rename(staging, args.pool)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"OCI_POOL_FILTERED: dropped={len(pinned)} retained={len(out_index):,} -> {args.pool}", flush=True)


# ------------------------------------------------------------------ finalize

def finalize(args) -> None:
    from scans import audit_pair

    manifest = json.loads((args.pool / "manifest.json").read_text())
    assert manifest["stage"] == "decontaminated_unverified" and (args.pool / "DONE").is_file()
    index = np.load(args.pool / "index.npy", mmap_mode="r")
    weights = np.memmap(args.pool / "weights.bin", dtype="<f4", mode="r")
    for rec in manifest["artifacts"].values():
        assert sha(args.pool / rec["file"]) == rec["sha256"]
    assert np.array_equal(index["offset"], np.r_[0, np.cumsum(index["total_len"][:-1])])
    assert int(index["total_len"].max()) <= manifest["window"]
    for row in index:
        lo, n, p = int(row["offset"]), int(row["total_len"]), int(row["prompt_len"])
        w = np.asarray(weights[lo:lo + n])
        if not (0 < p < n) or np.any(w[:p] != 0) or not np.all(w[p:] == 1):
            raise ValueError("malformed assistant-only mask")
    prov = pq.read_table(args.pool / "provenance.parquet", columns=["episode", "conversation_sha256",
                                                                   "normalized_prompt_sha256"], use_threads=False)
    assert prov.column("episode").to_pylist() == list(range(len(index)))
    for column in ("conversation_sha256", "normalized_prompt_sha256"):
        values = prov.column(column).to_pylist()
        assert len(set(values)) == len(values), f"duplicate {column} after filtering"
    reports = audit_pair(args.pool, args.tokenizer, args.samples, args.pool / "decontamination_reports", "clean-")
    post = []
    for path in reports:
        r = json.loads(path.read_text())
        post.append({"phase": "post_filter", "file": f"decontamination_reports/{path.name}", "sha256": sha(path),
                     "n_examples": r["n_examples"], "n_matching_examples": r["n_matching_examples"],
                     "n_matching_occurrences": r["n_matching_occurrences"], "max_match_tokens": r["max_match_tokens"]})
    clean = all(p["n_matching_occurrences"] == 0 for p in post)
    if not clean:
        # The paper's pool was scanned with token frequencies of a four-source corpus; a different rarest window
        # here is reported, not acted on, so the pool stays the paper's. The final payload is rescanned anyway.
        print(f"WARNING: post-filter scans report matches: {post}", flush=True)
    manifest["decontamination"].update(post_filter_reports=post, post_filter_zero_matches=clean)
    manifest["stage"] = "ready"
    (args.pool / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (args.pool / "READY").touch()
    print(f"OCI_POOL_READY: episodes={len(index):,} supervised={manifest['n_supervised_tokens']:,} -> {args.pool}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["stage", "build", "filter", "finalize"])
    p.add_argument("--sources", type=Path, help="staged OpenCodeInstruct shards")
    p.add_argument("--raw", type=Path, help="length-filtered, deduplicated pool")
    p.add_argument("--pool", type=Path, help="decontaminated pool")
    p.add_argument("--tokenizer", type=Path, help="HF directory of the base model's tokenizer")
    p.add_argument("--chat-template", type=Path, default=Path(__file__).with_name("simple_chatml.jinja"))
    p.add_argument("--samples", type=Path, help="benchmark bundle JSONL (../benchmark_samples.py)")
    p.add_argument("--batch-rows", type=int, default=2048)
    p.add_argument("--tokenize-batch", type=int, default=1024)
    args = p.parse_args()
    {"stage": stage, "build": build, "filter": filter_pool, "finalize": finalize}[args.phase](args)


if __name__ == "__main__":
    main()
