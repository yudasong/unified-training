#!/usr/bin/env python3
"""Evaluation prompts used to decontaminate the midtraining corpora (5,074 rows): ARC-Challenge, GSM8K, SciQ,
MATH-500 and IFEval test prompts plus HumanEval+ / MBPP+ prompts, one JSONL row {id, benchmark, question}.
Input of `python -m unified_training.tools.audit_benchmark_overlap --samples`.

  python benchmark_samples.py --out $DATA_DIR/decontamination/broad16-gsm8k-test.jsonl \
      [--evalplus-dir <dir with HumanEvalPlus-v0.1.10.jsonl, MbppPlus-v0.2.0.jsonl>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

# benchmark, (repo, config, revision, split), question field, id field
HF_SOURCES = [
    ("arc_challenge-test", ("allenai/ai2_arc", "ARC-Challenge", "210d026faf9955653af8916fad021475a3f00453", "test"), "question", "id"),
    ("gsm8k-test", ("openai/gsm8k", "main", "e53f048856ff4f594e959d75785d2c2d37b678ee", "test"), "question", None),
    ("sciq-test", ("allenai/sciq", None, "2c94ad3e1aafab77146f384e23536f97a4849815", "test"), "question", None),
    ("math500-test", ("HuggingFaceH4/MATH-500", None, "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be", "test"), "problem", "unique_id"),
    ("ifeval", ("google/IFEval", None, "966cd89545d6b6acfd7638bc708b98261ca58e84", "train"), "prompt", "key"),
]
EVALPLUS = [("humaneval-plus", "HumanEvalPlus-v0.1.10.jsonl"), ("mbpp-plus", "MbppPlus-v0.2.0.jsonl")]
RECEIPT = Path(__file__).resolve().parent / "receipts" / "broad16_gsm8k_test_manifest.json"


def hf_rows(repo: str, config: str | None, revision: str, split: str) -> list[dict]:
    from datasets import load_dataset

    return list(load_dataset(repo, config, revision=revision, split=split))


def evalplus_rows(name: str, local_dir: Path | None) -> list[dict]:
    if local_dir is not None:
        return [json.loads(line) for line in (local_dir / name).read_text().splitlines() if line.strip()]
    from evalplus.data import get_human_eval_plus, get_mbpp_plus

    problems = get_human_eval_plus() if name.startswith("HumanEval") else get_mbpp_plus()
    return list(problems.values())


def rows_to_records(benchmark: str, rows: list[dict], question_key: str, id_key: str | None) -> list[dict]:
    out = []
    for index, row in enumerate(rows):
        local_id = str(row[id_key]) if id_key and row.get(id_key) is not None else str(index)
        out.append({"id": f"{benchmark}:{local_id}", "benchmark": benchmark, "question": str(row[question_key])})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--evalplus-dir", type=Path)
    a = ap.parse_args()
    records = []
    for benchmark, (repo, config, revision, split), qkey, idkey in HF_SOURCES:
        records += rows_to_records(benchmark, hf_rows(repo, config, revision, split), qkey, idkey)
    for benchmark, name in EVALPLUS:
        records += rows_to_records(benchmark, evalplus_rows(name, a.evalplus_dir), "prompt", "task_id")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as stream:
        for r in records:
            stream.write(json.dumps(r) + "\n")
    digest = hashlib.sha256(a.out.read_bytes()).hexdigest()
    counts = Counter(r["benchmark"] for r in records)
    print(f"[bench] {len(records)} prompts {dict(counts)} sha256 {digest}")
    if RECEIPT.exists():
        paper = json.loads(RECEIPT.read_text())
        same = digest == paper["output_sha256"]
        print(f"[bench] {'matches' if same else 'DIFFERS from'} the paper's bundle ({paper['output_sha256'][:12]}...)")


if __name__ == "__main__":
    main()
