#!/usr/bin/env python3
"""KodCode coding-RL set of Appendix D.7: 10,000 medium + 10,000 hard Python problems, 512 held out.

The problem selection (GPT-4o difficulty labels, metadata filters, a pass@16 pilot of the staged math
model, stratified selection) is fixed by receipts/kodcode_hard_v2_ids.tsv (ordered: training rows, then
held-out rows). This script rebuilds each row from KodCode-V1: problem = question + missing function
signature(s) + coding instruction, answer = the pytest module. Outputs (NeMo-RL format):
  kodcode20k_hard_v2_train.parquet   (problem, answer)   -> KODCODE_TRAIN_FILE
  kodcode20k_hard_v2_holdout.jsonl   (problem, answer)   -> KODCODE_EVAL_FILE

  python kodcode.py --out $DATA_DIR/kodcode [--snapshot <local KodCode-V1 dir>] [--verify-against <dir>]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

REPO, REVISION = "KodCode/KodCode-V1", "97d81f0695ddd7c5c02bfe16436db9a17a4a21e5"
IDS = Path(__file__).resolve().parent / "receipts" / "kodcode_hard_v2_ids.tsv"
INSTRUCTION = "Write the solution in Python. Put the complete code in one ```python block and do not include tests."


def declarations(test_info) -> list[str]:
    out = []
    for entry in test_info or []:
        decl = (entry.get("function_declaration") or "").strip()
        if decl and decl not in out:
            out.append(decl)
    return out


def build_prompt(question: str, decls: list[str]) -> str:
    q = question.strip()
    missing = [d for d in decls if d.rstrip(":") not in q]
    parts = [q]
    if missing:
        names = ", ".join(f"`{d}`" for d in missing)
        parts.append(f"Implement {'this function' if len(missing) == 1 else 'these functions'}: {names}")
    parts.append(INSTRUCTION)
    return "\n\n".join(parts)


def read_ids(path: Path = IDS) -> list[tuple[str, str]]:
    lines = path.read_text().splitlines()[1:]
    return [tuple(line.split("\t")[:2]) for line in lines if line]


def load_kodcode(snapshot: str | None) -> dict[str, dict]:
    import pandas as pd

    if snapshot is None:
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(REPO, repo_type="dataset", revision=REVISION, allow_patterns=["data/*.parquet"])
    files = sorted(glob.glob(os.path.join(snapshot, "**", "*.parquet"), recursive=True))
    cols = ["question_id", "question", "test", "test_info"]
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    rows = {}
    for r in df.itertuples(index=False):
        info = r.test_info.tolist() if hasattr(r.test_info, "tolist") else r.test_info
        rows.setdefault(str(r.question_id), {"question": str(r.question), "test": str(r.test), "test_info": info})
    return rows


def build(rows_by_id: dict[str, dict], ids: list[tuple[str, str]]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {"train": [], "holdout": []}
    for split, kod_id in ids:
        r = rows_by_id[kod_id]
        out[split].append({"problem": build_prompt(r["question"], declarations(r["test_info"])), "answer": r["test"]})
    return out


def main() -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--snapshot", help="local KodCode-V1 snapshot (default: download the pinned revision)")
    ap.add_argument("--verify-against", type=Path, help="dir with the original train parquet / holdout jsonl")
    a = ap.parse_args()
    data = build(load_kodcode(a.snapshot), read_ids())
    a.out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(data["train"]), a.out / "kodcode20k_hard_v2_train.parquet")
    with open(a.out / "kodcode20k_hard_v2_holdout.jsonl", "w") as f:
        for row in data["holdout"]:
            f.write(json.dumps(row) + "\n")
    print(f"[kodcode] train {len(data['train'])}, holdout {len(data['holdout'])} -> {a.out}")
    if a.verify_against:
        ref_train = pq.read_table(a.verify_against / "kodcode20k_hard_v2_train.parquet").to_pylist()
        ref_hold = [json.loads(line) for line in open(a.verify_against / "kodcode20k_hard_v2_holdout.jsonl")]
        for name, mine, ref in (("train", data["train"], ref_train), ("holdout", data["holdout"], ref_hold)):
            bad = [i for i, (m, r) in enumerate(zip(mine, ref)) if (m["problem"], m["answer"]) != (r["problem"], r["answer"])]
            print(f"[kodcode] verify {name}: {len(mine)} vs {len(ref)} rows, {len(bad)} mismatches {bad[:5]}")


if __name__ == "__main__":
    main()
