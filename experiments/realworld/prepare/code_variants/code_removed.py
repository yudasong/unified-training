#!/usr/bin/env python3
"""Code-removed corpus (App. D.4): the 240M corpus without its three designated code sources.

Deletion only (the variant the paper trained on): rows of cranecode, stack_edu-fim and code-meta-reasoning
and the trainer-holdout episodes (plus 2 exact duplicates) are dropped; the remaining 647,327 windows keep
their tokens, masks and physical order (180,391,606 supervised tokens). "No explicit code" means these
source components are absent, not that the remaining text contains no code.

Phases: assemble, then validate (benchmark scans must be clean; trainer rows checked at length 2048).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from common import REMOVED, excluded_original_ids, load_pool, sha, write_json
from unified_training.episodes import INDEX_DTYPE

EXPECTED_SUPERVISED_TOKENS = 180_391_606


class Writer:
    def __init__(self, path: Path):
        self.path = path
        path.mkdir(parents=True, exist_ok=False)
        self.tf = (path / "tokens.bin").open("xb", buffering=8 * 1024**2)
        self.wf = (path / "weights.bin").open("xb", buffering=8 * 1024**2)
        self.rows, self.provenance = [], []
        self.tokens, self.max_id = 0, 0
        self.counts, self.episodes = Counter(), Counter()

    def add(self, tokens, weights, group, component, source_pool, source_episode):
        n = len(tokens)
        assert 2 <= n <= 1025 and weights[0] == 0 and np.all(weights[1:] == 1)
        self.tf.write(np.asarray(tokens, dtype="<i4").tobytes())
        self.wf.write(np.asarray(weights, dtype="<f4").tobytes())
        self.rows.append((self.tokens, 1, n, 0., 1., -1, group))
        self.provenance.append((len(self.rows) - 1, component, source_pool, int(source_episode), "", -1, -1))
        self.tokens += n
        self.max_id = max(self.max_id, int(np.max(tokens)))
        self.counts[component] += n - 1
        self.episodes[component] += 1

    def finish(self, metadata: dict) -> dict:
        self.tf.close()
        self.wf.close()
        np.save(self.path / "index.npy", np.array(self.rows, dtype=INDEX_DTYPE))
        names = ("episode", "component", "source_pool", "source_episode", "source_file", "source_row", "window_start")
        pq.write_table(pa.table({name: [r[i] for r in self.provenance] for i, name in enumerate(names)}),
                       self.path / "provenance.parquet")
        supervised = sum(self.counts.values())
        metadata = dict(metadata, n_episodes=len(self.rows), n_tokens=self.tokens, n_supervised_tokens=supervised,
                        n_weighted_tokens=supervised, supervised_token_count_basis="count(token_weight > 0)",
                        max_token_id=self.max_id, mean_response_len=supervised / len(self.rows), mean_reward=0.,
                        mean_weight=1., component_realized_supervised_tokens=dict(self.counts),
                        component_realized_episodes=dict(self.episodes), stage="awaiting_final_validation")
        metadata["artifacts"] = {f: {"file": f, "sha256": sha(self.path / f)}
                                 for f in ("tokens.bin", "weights.bin", "index.npy", "provenance.parquet")}
        write_json(self.path / "manifest.json", metadata)
        return metadata


def setup(args):
    am, ai, at, aw = load_pool(args.anchor)
    excluded = excluded_original_ids(args.anchor, ai, at, args.holdout)
    groups = {v: k for k, v in am["component_group_ids"].items()}
    base = [i for i, r in enumerate(ai) if i not in excluded and groups[int(r["group_id"])] not in REMOVED]
    supervised = sum(int(ai[i]["total_len"]) - 1 for i in base)
    assert supervised == EXPECTED_SUPERVISED_TOKENS, supervised
    return am, ai, at, aw, excluded, groups, base


def assemble(args) -> None:
    am, ai, at, aw, excluded, groups, base = setup(args)
    w = Writer(args.output)
    for i in base:
        r = ai[i]
        lo = int(r["offset"])
        hi = lo + int(r["total_len"])
        group = int(r["group_id"])
        w.add(at[lo:hi], aw[lo:hi], group, groups[group], "original_anchor", i)
    w.finish({"kind": "anchor", "format": "document_aware_raw", "anchor_weight": 1., "window": 1025,
              "raw_window": 1025, "tokenizer": am["tokenizer"], "tokenizer_sha256": am["tokenizer_sha256"],
              "removed_components": sorted(REMOVED), "content_scope": "source-component exclusion only",
              "component_group_ids": {k: v for k, v in am["component_group_ids"].items() if k not in REMOVED},
              "source_anchor": str(args.anchor.resolve()),
              "source_anchor_manifest_sha256": sha(args.anchor / "manifest.json"),
              "excluded_source_episode_ids": sorted(excluded), "builder_sha256": sha(__file__),
              "decontamination": {"benchmark": "broad16-gsm8k", "post_filter_zero_matches": False},
              "variant": "no_explicit_code_deletion_v1", "construction": "deletion_only"})
    print(f"DELETION_ASSEMBLED {args.output}", flush=True)


def validate(args) -> None:
    from transformers import AutoTokenizer

    from scans import audit_pair
    from unified_training.dataset import _episode_row
    from unified_training.episodes import Round

    output = args.output
    m, ix, t, w = load_pool(output)
    am, ai, at, aw, excluded, groups, base = setup(args)
    prov = pq.read_table(output / "provenance.parquet").to_pydict()
    assert m["construction"] == "deletion_only"
    assert not set(REMOVED).intersection(m["component_group_ids"])
    assert len(ix) == m["n_episodes"] and len(t) == len(w) == m["n_tokens"]
    assert np.array_equal(ix["offset"], np.r_[0, np.cumsum(ix["total_len"][:-1])])
    for entry in m["artifacts"].values():
        assert sha(output / entry["file"]) == entry["sha256"]
    actual_base, component_tokens = [], Counter()
    for i, r in enumerate(ix):
        n, lo, j, comp = int(r["total_len"]), int(r["offset"]), int(prov["source_episode"][i]), prov["component"][i]
        assert comp not in REMOVED and int(r["prompt_len"]) == 1 and 2 <= n <= 1025
        assert prov["episode"][i] == i and groups[int(r["group_id"])] == comp
        assert prov["source_pool"][i] == "original_anchor"
        actual_base.append(j)
        so = int(ai[j]["offset"])
        assert n == int(ai[j]["total_len"])
        assert np.array_equal(t[lo:lo + n], at[so:so + n])
        assert np.array_equal(w[lo:lo + n].view("<u4"), aw[so:so + n].view("<u4"))
        assert w[lo] == 0 and np.all(w[lo + 1:lo + n] == 1)
        component_tokens[comp] += n - 1
    assert actual_base == base and not excluded.intersection(actual_base)
    assert dict(component_tokens) == m["component_realized_supervised_tokens"]
    assert sum(component_tokens.values()) == m["n_supervised_tokens"]
    reports = []
    for path in audit_pair(output, args.tokenizer, args.samples, output / "decontamination_reports"):
        r = json.loads(path.read_text())
        assert r["n_matching_occurrences"] == r["n_matching_examples"] == 0, f"benchmark overlap: {path}"
        assert not r["examples_at_match_limit"]
        reports.append({"file": str(path.relative_to(output)), "sha256": sha(path), "n_examples": r["n_examples"],
                        "matches": 0})
    round_ = Round(output, t, w, ix, m)
    sample_ids = np.unique(np.r_[np.linspace(0, len(ix) - 1, 24, dtype=int), np.argmax(ix["total_len"])]).tolist()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    with (output / "examples.jsonl").open("w") as f:
        for i in sample_ids:
            row = _episode_row([round_], (0, i), seq_length=2048, pad_token_id=0)
            n, lo = int(ix[i]["total_len"]), int(ix[i]["offset"])
            assert int(row["loss_mask"].sum()) == int(row["denom_mask"].sum()) == n - 1
            f.write(json.dumps({"episode": i, "component": prov["component"][i], "source_pool": prov["source_pool"][i],
                                "text": tokenizer.decode(t[lo:lo + n].tolist())}) + "\n")
    receipt = {"payload_rows_checked": len(ix), "preserved_original_rows": len(actual_base),
               "new_unique_donor_windows": 0, "heldout_source_rows_present": 0, "removed_component_rows_present": 0,
               "supervised_tokens": sum(component_tokens.values()), "trainer_checks": len(sample_ids),
               "trainer_sequence_length": 2048, "benchmark_reports": reports, "validator_sha256": sha(__file__)}
    write_json(output / "validation.json", receipt)
    m["decontamination"].update(post_filter_zero_matches=True, reports=reports)
    m["stage"] = "ready"
    for name in ("examples.jsonl", "validation.json"):
        m["artifacts"][name] = {"file": name, "sha256": sha(output / name)}
    write_json(output / "manifest.json", m)
    (output / "DONE").touch()
    (output / "READY").touch()
    print(json.dumps(receipt, indent=2), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["assemble", "validate"])
    p.add_argument("--anchor", type=Path, required=True, help="the 240M midtraining corpus")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--holdout", type=Path, help="trainer validation-holdout dir of the 240M corpus (optional; "
                   "default: pinned receipt of the excluded episode ids)")
    p.add_argument("--tokenizer", type=Path, help="HF directory of the base model's tokenizer (validate)")
    p.add_argument("--samples", type=Path, help="benchmark bundle (validate)")
    args = p.parse_args()
    {"assemble": assemble, "validate": validate}[args.phase](args)


if __name__ == "__main__":
    main()
