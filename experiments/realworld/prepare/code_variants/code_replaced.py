#!/usr/bin/env python3
"""Code-replaced corpus (App. D.4): the 240M corpus with its three code sources replaced by tested Python.

Phases:
  verify    run every eligible OpenCodeInstruct episode's supplied assert-style tests (at most three, shown
            in its prompt) in a sandbox: fresh child per candidate, no network/process syscalls, 2 s CPU,
            3 s wall, 1 GiB; keep the passes
  assemble  copy the corpus in physical order without the trainer-holdout episodes; each removed code
            window is replaced, in place, by whole passing coding episodes (seeded permutation, seed 1234)
            until the removed supervised-token budget (59,451,795) is met; prompts weight 0, answers 1
  validate  byte-compare every row with its parent, run the final benchmark scans (must be clean), check
            trainer rows at sequence length 2048, then write DONE/READY

The paper's v2 artifact excluded 11 coding episodes whose answers matched the HumanEval/26 signature
fingerprint in a first assembly's final scan (receipts/oci_fingerprint_exclusions.json, applied by default).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from common import REMOVED, excluded_original_ids, load_pool, load_receipt, sha, write_json
from unified_training.episodes import INDEX_DTYPE
from oci_records import (
    VERIFIED_OPENCODE_COMPONENT,
    conversation_sha256,
    opencodeinstruct_contract_messages,
)

HERE = Path(__file__).resolve().parent
OCI_REVISION = "8f3ba5bafe4d6e8db46082cf7ae6741bc370604d"


def verify(args) -> None:
    args.work.mkdir(parents=True, exist_ok=False)
    args.scratch.mkdir(parents=True, exist_ok=True)
    manifest, index, tokens, weights = load_pool(args.pool)
    assert (args.pool / "READY").exists()
    for entry in manifest["artifacts"].values():
        assert sha(args.pool / entry["file"]) == entry["sha256"]
    assert sha(args.pool / "provenance.parquet") == manifest["provenance_sha256"]
    staged = json.loads((args.sources / "source_manifest.json").read_text())
    assert sha(args.sources / "source_manifest.json") == manifest["source_manifest_sha256"]
    source = next(x for x in staged["sources"] if x["name"] == "opencodeinstruct")
    assert source["revision"] == OCI_REVISION and len(source["files"]) == 10
    for entry in source["files"]:
        assert sha(args.sources / "opencodeinstruct" / entry["path"]) == entry["sha256"]
    provenance = pq.read_table(args.pool / "provenance.parquet", use_threads=False,
                               filters=[("component", "=", VERIFIED_OPENCODE_COMPONENT)]).to_pylist()
    by_source = {int(x["source_row_index"]): x for x in provenance}
    assert len(by_source) == len(provenance)
    write_json(args.work / "input_identities.json", {
        "pool": str(args.pool.resolve()), "pool_manifest_sha256": sha(args.pool / "manifest.json"),
        "source_manifest_sha256": sha(args.sources / "source_manifest.json"), "source": source,
        "eligible_oci_episodes": len(provenance), "verified_pool_artifacts": manifest["artifacts"],
        "builder_sha256": sha(__file__), "sandbox_python_sha256": sha(HERE / "coding_anchor_sandbox.py"),
        "sandbox_shell_sha256": sha(HERE / "coding_anchor_sandbox.sh")})
    print(f"verified inputs; {len(provenance):,} complete clean OpenCodeInstruct candidates", flush=True)
    local, lock, processes = threading.local(), threading.Lock(), []

    def check(item):
        global_index, record = item
        prov = by_source[global_index]
        messages = opencodeinstruct_contract_messages(record)
        assert conversation_sha256(messages) == prov["conversation_sha256"]
        assert not prov["truncated"]
        tests = json.loads(record["unit_tests"])
        if not isinstance(tests, list) or not all(isinstance(t, str) for t in tests):
            raise ValueError("malformed supplied suite")
        if not hasattr(local, "process"):
            with lock:
                n = len(processes)
                root = args.scratch / f"sandbox-{n:02d}"
                root.mkdir(exist_ok=False)
                error = (args.work / f"sandbox-{n:02d}.stderr").open("w")
                local.process = subprocess.Popen(
                    ["/bin/bash", str(HERE / "coding_anchor_sandbox.sh"), str(root),
                     str(HERE / "coding_anchor_sandbox.py"), args.sandbox_python],
                    env={"PATH": "/usr/bin:/bin"}, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=error, text=True, bufsize=1)
                processes.append((local.process, error))
        process = local.process
        process.stdin.write(json.dumps({"code": messages[1]["content"], "tests": tests}) + "\n")
        process.stdin.flush()
        line = process.stdout.readline()
        if not line:
            raise RuntimeError("test sandbox exited unexpectedly")
        result = json.loads(line)
        result.update(pool_episode=int(prov["episode"]), source_row_index=global_index,
                      supervised_tokens=int(prov["assistant_loss_tokens"]),
                      conversation_sha256=prov["conversation_sha256"],
                      supplied_tests_sha256=hashlib.sha256(record["unit_tests"].encode()).hexdigest())
        return result

    counts, passed_tokens, passed, scanned, counter = Counter(), 0, [], 0, 0
    start = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor, \
                (args.work / "local_test_receipts.jsonl").open("x") as receipt:
            for entry in source["files"]:
                path = args.sources / "opencodeinstruct" / entry["path"]
                for batch in pq.ParquetFile(path).iter_batches(
                        batch_size=2048, columns=["input", "output", "unit_tests", "average_test_score"]):
                    jobs = []
                    for record in batch.to_pylist():
                        if scanned in by_source:
                            jobs.append((scanned, record))
                        scanned += 1
                    for result in executor.map(check, jobs):
                        receipt.write(json.dumps(result, sort_keys=True) + "\n")
                        counts[result["status"]] += 1
                        counter += 1
                        if result["status"] == "pass":
                            passed.append(result["pool_episode"])
                            passed_tokens += result["supervised_tokens"]
                    receipt.flush()
                    print(json.dumps({"checked": counter, "counts": counts, "passed_supervised_tokens": passed_tokens,
                                      "seconds": round(time.monotonic() - start, 1)}), flush=True)
    finally:
        for process, error in processes:
            if process.stdin:
                process.stdin.close()
            process.wait(timeout=10)
            error.close()
    assert counter == len(provenance)
    np.save(args.work / "passing_pool_episode_ids.npy", np.asarray(passed, dtype="<i8"))
    write_json(args.work / "verification_summary.json", {
        "counts": counts, "checked": counter, "passed_supervised_tokens": passed_tokens,
        "seconds": time.monotonic() - start, "workers": args.workers,
        "python": "system Python 3.11, standard library", "cpu_limit_seconds": 2, "wall_limit_seconds": 3,
        "memory_limit_bytes": 1024**3,
        "test_policy": "all supplied strings must parse with a top-level assert; all executed in fresh solution namespaces",
        "scope": "passes supplied tests, not a proof of semantic correctness"})
    (args.work / "VERIFIED").touch()


def fingerprint_exclusion_ids(pool: Path) -> list[int]:
    """Pool episode ids of the HumanEval/26 fingerprint exclusions (receipt keyed by OpenCodeInstruct row)."""
    receipt = load_receipt("oci_fingerprint_exclusions.json")
    wanted = {r["source_row_index"]: r["conversation_sha256"] for r in receipt["rows"]}
    prov = pq.read_table(pool / "provenance.parquet", use_threads=False,
                         columns=["episode", "source_row_index", "conversation_sha256"]).to_pylist()
    ids = [int(r["episode"]) for r in prov if r["source_row_index"] in wanted
           and wanted[r["source_row_index"]] == r["conversation_sha256"]]
    if len(ids) != len(wanted):
        raise ValueError(f"resolved {len(ids)} of {len(wanted)} fingerprint exclusions in {pool}")
    return ids


def assemble(args) -> None:
    assert (args.work / "VERIFIED").exists()
    am, ai, at, aw = load_pool(args.anchor)
    pm, pi, pt, pw = load_pool(args.pool)
    assert am["tokenizer_sha256"] == pm["tokenizer_sha256"]
    excluded = excluded_original_ids(args.anchor, ai, at, args.holdout)
    groups = {v: k for k, v in am["component_group_ids"].items()}
    code_groups = {am["component_group_ids"][k] for k in REMOVED}
    target = sum(int(r["total_len"] - r["prompt_len"]) for i, r in enumerate(ai)
                 if i not in excluded and int(r["group_id"]) in code_groups)
    passing = np.load(args.work / "passing_pool_episode_ids.npy")
    rng = np.random.default_rng(args.seed)
    passing = passing[rng.permutation(len(passing))]
    coding_exclusions = [] if args.no_fingerprint_exclusions else fingerprint_exclusion_ids(args.pool)
    if coding_exclusions:
        passing = passing[~np.isin(passing, coding_exclusions)]
    mass = pi["total_len"][passing] - pi["prompt_len"][passing]
    assert int(mass.sum()) >= target, "insufficient locally verified unique coding supply"
    count = int(np.searchsorted(np.cumsum(mass, dtype=np.int64), target)) + 1
    selected = passing[:count]
    assert len(set(map(int, selected))) == len(selected)
    args.output.mkdir(parents=True, exist_ok=False)
    output_rows, provenance = [], []
    component_tokens, component_rows = Counter(), Counter()
    total_tokens, code_emitted, code_due, cursor, max_id = 0, 0, 0, 0, 0
    coding_group = max(groups) + 1
    with (args.output / "tokens.bin").open("xb", buffering=8 * 1024**2) as tf, \
            (args.output / "weights.bin").open("xb", buffering=8 * 1024**2) as wf:
        def emit(pool_name, source_i):
            nonlocal total_tokens, max_id
            ix, tok, wei = (ai, at, aw) if pool_name == "backbone" else (pi, pt, pw)
            r = ix[source_i]
            lo = int(r["offset"])
            hi = lo + int(r["total_len"])
            t, w = tok[lo:hi], wei[lo:hi]
            plen = int(r["prompt_len"])
            assert np.all(w[:plen] == 0) and np.all(w[plen:] == 1)
            assert 0 < plen < len(t) <= (1025 if pool_name == "backbone" else 2049)
            group = int(r["group_id"]) if pool_name == "backbone" else coding_group
            component = groups[group] if pool_name == "backbone" else VERIFIED_OPENCODE_COMPONENT
            tf.write(t.tobytes())
            wf.write(w.tobytes())
            output_rows.append((total_tokens, plen, len(t), float(r["reward"]), 1., -1, group))
            provenance.append((len(output_rows) - 1, pool_name, int(source_i), component))
            total_tokens += len(t)
            component_tokens[component] += len(t) - plen
            component_rows[component] += 1
            max_id = max(max_id, int(t.max()))

        for i, row in enumerate(ai):
            if i in excluded:
                continue
            if int(row["group_id"]) in code_groups:
                code_due += int(row["total_len"] - row["prompt_len"])
                while code_emitted < code_due:
                    j = int(selected[cursor])
                    cursor += 1
                    emit("coding_pool", j)
                    code_emitted += int(pi[j]["total_len"] - pi[j]["prompt_len"])
            else:
                emit("backbone", i)
    assert cursor == len(selected) and 0 <= code_emitted - target < 2049
    np.save(args.output / "index.npy", np.array(output_rows, dtype=INDEX_DTYPE))
    pq.write_table(pa.table({"episode": [x[0] for x in provenance], "source_pool": [x[1] for x in provenance],
                             "source_episode": [x[2] for x in provenance], "component": [x[3] for x in provenance]}),
                   args.output / "provenance.parquet")
    np.save(args.output / "excluded_original_episode_ids.npy", np.array(sorted(excluded), dtype="<i8"))
    np.save(args.output / "selected_coding_pool_episode_ids.npy", selected)
    coding_provenance = pq.read_table(args.pool / "provenance.parquet", use_threads=False)
    assert np.array_equal(coding_provenance.column("episode").to_numpy(), np.arange(len(pi)))
    pq.write_table(coding_provenance.take(pa.array(selected)), args.output / "coding_provenance.parquet")
    manifest = {
        "kind": "anchor", "format": "document_aware_raw_plus_assistant_only_python",
        "variant": "fresh_code_replacement_oci_locally_tested_v1", "anchor_weight": 1.,
        "tokenizer": am["tokenizer"], "tokenizer_sha256": am["tokenizer_sha256"],
        "window": 2049, "raw_window": 1025, "reasoning_window": 2049, "max_causal_positions": 2048,
        "max_token_id": max_id, "n_episodes": len(output_rows), "n_tokens": total_tokens,
        "n_supervised_tokens": sum(component_tokens.values()), "n_weighted_tokens": sum(component_tokens.values()),
        "supervised_token_count_basis": "count(token_weight > 0)",
        "mean_response_len": sum(component_tokens.values()) / len(output_rows), "mean_weight": 1., "mean_reward": 0.,
        "component_group_ids": {**{k: v for k, v in am["component_group_ids"].items() if k not in REMOVED},
                                VERIFIED_OPENCODE_COMPONENT: coding_group},
        "component_realized_supervised_tokens": dict(component_tokens),
        "component_realized_episodes": dict(component_rows),
        "replacement": {
            "removed_components": sorted(REMOVED), "target_supervised_tokens": target,
            "actual_supervised_tokens": code_emitted, "whole_episode_overshoot": code_emitted - target,
            "coding_pool": str(args.pool.resolve()), "selection_seed": args.seed,
            "unique_coding_episodes": len(selected), "repeated_coding_episodes": 0,
            "additional_coding_exclusion": {"receipt": "oci_fingerprint_exclusions.json",
                                            "pool_episode_ids": coding_exclusions} if coding_exclusions else None,
            "verification_directory": str(args.work.resolve()),
            "verification_summary": json.loads((args.work / "verification_summary.json").read_text())},
        "backbone": {"source_anchor": str(args.anchor.resolve()),
                     "source_manifest_sha256": sha(args.anchor / "manifest.json"),
                     "source_artifacts_sha256": {f: sha(args.anchor / f) for f in ("index.npy", "tokens.bin", "weights.bin")},
                     "preservation": "exact source tokens/masks and non-code physical relative order, except holdout exclusions"},
        "holdout_exclusion": {"source_episode_ids": 512, "excluded_with_exact_duplicates": len(excluded),
                              "strategy": "physically excluded before writing; independent of training seed"},
        "decontamination": {"benchmark": "broad16-gsm8k", "method": "unchanged subsets of already cleaned parents; final scan pending",
                            "post_filter_zero_matches": False},
        "builder_sha256": sha(__file__), "stage": "awaiting_final_validation",
    }
    manifest["artifacts"] = {f: {"file": f, "sha256": sha(args.output / f)} for f in (
        "tokens.bin", "weights.bin", "index.npy", "provenance.parquet", "coding_provenance.parquet",
        "selected_coding_pool_episode_ids.npy", "excluded_original_episode_ids.npy")}
    write_json(args.output / "manifest.json", manifest)
    write_json(args.work / "assembly_summary.json", {"target": target, "coding_tokens": code_emitted,
               "total_supervised_tokens": sum(component_tokens.values()), "episodes": len(output_rows),
               "holdout_exclusions": len(excluded)})
    print(json.dumps({k: v for k, v in manifest["replacement"].items() if k != "verification_summary"}, indent=2),
          flush=True)


def validate(args) -> None:
    from transformers import AutoTokenizer

    from scans import audit_pair
    from unified_training.dataset import _episode_row
    from unified_training.episodes import Round

    out = args.output
    assert not (out / "READY").exists()
    m, ix, tokens, weights = load_pool(out)
    anchor, pool = Path(m["backbone"]["source_anchor"]), Path(m["replacement"]["coding_pool"])
    am, ai, at, aw = load_pool(anchor)
    pm, pi, pt, pw = load_pool(pool)
    assert sha(anchor / "manifest.json") == m["backbone"]["source_manifest_sha256"]
    for entry in m["artifacts"].values():
        assert sha(out / entry["file"]) == entry["sha256"]
    provenance = pq.read_table(out / "provenance.parquet").to_pydict()
    assert len(provenance["episode"]) == len(ix) == m["n_episodes"]
    assert np.array_equal(ix["offset"], np.concatenate(([0], np.cumsum(ix["total_len"][:-1]))))
    assert len(tokens) == len(weights) == int(ix[-1]["offset"] + ix[-1]["total_len"]) == m["n_tokens"]
    excluded = set(map(int, np.load(out / "excluded_original_episode_ids.npy")))
    selected = list(map(int, np.load(out / "selected_coding_pool_episode_ids.npy")))
    passing = set(map(int, np.load(Path(m["replacement"]["verification_directory"]) / "passing_pool_episode_ids.npy")))
    assert set(selected) <= passing and len(selected) == len(set(selected))
    assert set(REMOVED).isdisjoint(m["component_group_ids"])
    keep_groups = {v for k, v in am["component_group_ids"].items() if k not in REMOVED}
    expected_backbone = [i for i, r in enumerate(ai) if i not in excluded and int(r["group_id"]) in keep_groups]
    actual_backbone, actual_code, counts, code_positions = [], [], Counter(), []
    longest_code = (0, 0)
    for i, row in enumerate(ix):
        assert provenance["episode"][i] == i
        name, j = provenance["source_pool"][i], provenance["source_episode"][i]
        is_code = name == "coding_pool"
        si, st, sw = (pi, pt, pw) if is_code else (ai, at, aw)
        sr = si[j]
        lo, n, so = int(row["offset"]), int(row["total_len"]), int(sr["offset"])
        assert int(row["prompt_len"]) == int(sr["prompt_len"])
        assert n == int(sr["total_len"]) <= (2049 if is_code else 1025)
        assert np.array_equal(tokens[lo:lo + n], st[so:so + n])
        assert np.array_equal(weights[lo:lo + n], sw[so:so + n])
        plen = int(row["prompt_len"])
        assert 0 < plen < n
        assert np.all(weights[lo:lo + plen] == 0) and np.all(weights[lo + plen:lo + n] == 1)
        counts[provenance["component"][i]] += n - plen
        if is_code:
            actual_code.append(j)
            code_positions.append(i)
            longest_code = max(longest_code, (n, i))
            assert provenance["component"][i] == VERIFIED_OPENCODE_COMPONENT
        else:
            actual_backbone.append(j)
    assert actual_backbone == expected_backbone and actual_code == selected
    assert dict(counts) == m["component_realized_supervised_tokens"]
    assert sum(counts.values()) == m["n_supervised_tokens"]
    coding_prov = pq.read_table(out / "coding_provenance.parquet").to_pydict()
    for column in ("source_id", "conversation_sha256", "normalized_prompt_sha256"):
        assert len(set(coding_prov[column])) == len(selected)
    assert not any(coding_prov["truncated"]) and coding_prov["episode"] == selected
    report_paths = audit_pair(out, args.tokenizer, args.samples, out / "decontamination_reports", "final-broad16-")
    reports = []
    for path in report_paths:
        report = json.loads(path.read_text())
        assert Path(report["tokens"]).resolve() == (out / "tokens.bin").resolve()
        assert report["n_matching_examples"] == report["n_matching_occurrences"] == 0, f"benchmark overlap: {path}"
        assert not report["examples_at_match_limit"]
        reports.append({"file": str(path.relative_to(out)), "sha256": sha(path), "n_examples": report["n_examples"],
                        "n_matching_occurrences": 0})
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    sample_positions = list(np.asarray(code_positions)[np.linspace(0, len(code_positions) - 1, 32, dtype=int)])
    sample_positions.append(longest_code[1])
    round_ = Round(out, tokens, weights, ix, m)
    samples = []
    for i in sample_positions:
        i = int(i)
        r = ix[i]
        lo, n, p = int(r["offset"]), int(r["total_len"]), int(r["prompt_len"])
        row = _episode_row([round_], (0, i), seq_length=2048, pad_token_id=0)
        assert int(row["loss_mask"].sum()) == int(row["denom_mask"].sum()) == n - p
        assert row["tokens"].numel() == 2048
        response = tokenizer.decode(tokens[lo + p:lo + n].tolist())
        assert response.endswith("<|im_end|>\n")
        samples.append({"episode": i, "source_pool_episode": provenance["source_episode"][i],
                        "prompt": tokenizer.decode(tokens[lo:lo + p].tolist()), "response": response,
                        "supervised_tokens": n - p, "total_tokens": n})
    with (out / "coding_examples.jsonl").open("w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    receipt = {"all_payload_rows_verified_against_parents": len(ix),
               "preserved_backbone_episodes": len(actual_backbone), "unique_coding_episodes": len(selected),
               "heldout_source_episode_intersection": len(excluded.intersection(actual_backbone)),
               "supervised_tokens": sum(counts.values()), "longest_code_stored_tokens": longest_code[0],
               "consumer_checks": len(samples), "consumer_sequence_length": 2048,
               "consumer_denominator_and_loss_token_counts_match": True,
               "selected_source_files": dict(Counter(coding_prov["source_file"])),
               "decontamination_reports": reports, "validator_sha256": sha(__file__)}
    assert receipt["heldout_source_episode_intersection"] == 0
    write_json(out / "validation.json", receipt)
    m["decontamination"].update(method="final episode-aware full-question and rarest-16-token scans",
                                post_filter_zero_matches=True, reports=reports)
    m["stage"] = "ready"
    for filename in ("validation.json", "coding_examples.jsonl"):
        m["artifacts"][filename] = {"file": filename, "sha256": sha(out / filename)}
    write_json(out / "manifest.json", m)
    (out / "DONE").touch()
    (out / "READY").touch()
    print(json.dumps(receipt, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["verify", "assemble", "validate"])
    p.add_argument("--anchor", type=Path, help="the 240M midtraining corpus")
    p.add_argument("--pool", type=Path, help="OpenCodeInstruct pool (oci_pool.py)")
    p.add_argument("--sources", type=Path, help="staged OpenCodeInstruct shards")
    p.add_argument("--holdout", type=Path, help="trainer validation-holdout dir of the 240M corpus (optional; "
                   "default: pinned receipt of the excluded episode ids)")
    p.add_argument("--work", type=Path, required=True, help="verification receipts directory")
    p.add_argument("--output", type=Path, help="output corpus directory")
    p.add_argument("--scratch", type=Path, help="empty node-local directory for sandbox roots")
    p.add_argument("--tokenizer", type=Path, help="HF directory of the base model's tokenizer (validate)")
    p.add_argument("--samples", type=Path, help="benchmark bundle (validate)")
    p.add_argument("--sandbox-python", default="/usr/bin/python3.11")
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--no-fingerprint-exclusions", action="store_true",
                   help="skip the 11 HumanEval/26 exclusions (reproduces the superseded v1 selection)")
    args = p.parse_args()
    {"verify": verify, "assemble": assemble, "validate": validate}[args.phase](args)


if __name__ == "__main__":
    main()
