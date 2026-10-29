#!/usr/bin/env python3
"""Math12K RL prompts (11,977 after length filtering) and MATH-500 validation file.

Writes, under --out:
  math12k_train.parquet               hiyouga/math12k train (12,000 rows: problem, answer)    -> NeMo-RL (MaxRL from the base model)
  math500_test.jsonl                  HuggingFaceH4/MATH-500 test (500 rows)                  -> NeMo-RL validation
  math12k_verlbox_unified_train.json  unified-trainer prompts: problem + verl boxed instruction, gold answer;
                                      empty rows and prompts of >= 1024 chat tokens dropped  -> UNIFIED_PROMPTS_JSON
The unified prompt file of the paper has sha256 bbd227edcabb8f11bb3dea8cba19c7349ef4935d88bd3c30213ccef620aafdcb
(checked at the end; --tokenizer must be the base model's).

  python math12k.py --out $DATA_DIR/math12k --tokenizer $CKPT_DIR/realworld/base
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

MATH12K = ("hiyouga/math12k", "9da556f65a8ca76a4bb0f4c0d609d662773f8707")
MATH500 = ("HuggingFaceH4/MATH-500", "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be")
PAPER_SHA256 = "bbd227edcabb8f11bb3dea8cba19c7349ef4935d88bd3c30213ccef620aafdcb"
VERL_SUFFIX = " Let's think step by step and output the final answer within \\boxed{}."
CHAT_TEMPLATE = "<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_rows(rows) -> tuple[list[dict], list[int]]:
    out, dropped = [], []
    for source_index, row in enumerate(rows):
        question, gold = str(row.get("problem") or ""), str(row.get("answer") or "")
        if not question.strip() or not gold.strip():
            dropped.append(source_index)
            continue
        out.append({"question": question, "gold": gold, "source_index": source_index})
    return out, dropped


def filter_overlong(rows: list[dict], count_tokens, max_model_len: int = 1024) -> tuple[list[dict], list[dict]]:
    kept, dropped = [], []
    for row in rows:
        n = count_tokens(CHAT_TEMPLATE.format(question=row["question"]))
        if n >= max_model_len:
            dropped.append({"source_index": row["source_index"], "prompt_tokens": n})
        else:
            kept.append(row)
    return kept, dropped


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp-{os.getpid()}"
    with tmp.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
    os.replace(tmp, path)


def unified_prompts(rows: list[dict], count_tokens) -> tuple[list[dict], dict]:
    canon, dropped_empty = canonical_rows(rows)
    for row in canon:
        row["question"] = row["question"] + VERL_SUFFIX
    kept, dropped_long = filter_overlong(canon, count_tokens)
    return kept, {"dropped_empty_question_or_gold": dropped_empty, "dropped_overlong_prompts": dropped_long}


def main() -> None:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tokenizer", required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    train = load_dataset(MATH12K[0], revision=MATH12K[1])["train"].select_columns(["problem", "answer"])
    assert len(train) == 12000, len(train)
    train.to_parquet(str(a.out / "math12k_train.parquet"))
    test = load_dataset(MATH500[0], revision=MATH500[1])["test"].select_columns(["problem", "answer"])
    assert len(test) == 500, len(test)
    test.to_json(str(a.out / "math500_test.jsonl"), orient="records", lines=True)

    tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    rows, dropped = unified_prompts(train.to_list(), lambda s: len(tok.encode(s, add_special_tokens=False)))
    dest = a.out / "math12k_verlbox_unified_train.json"
    atomic_json(dest, rows)
    digest = sha256(dest)
    atomic_json(a.out / "math12k_verlbox_unified_manifest.json", {
        "source": {"repo": MATH12K[0], "revision": MATH12K[1]}, "output_rows": len(rows), "output_sha256": digest,
        **{k: v for k, v in dropped.items()}, "verl_suffix": VERL_SUFFIX, "max_model_len": 1024})
    print(f"[math12k] {len(rows)} prompts -> {dest} (sha256 {digest[:12]}..., paper {PAPER_SHA256[:12]}...)")
    if len(rows) != 11977 or digest != PAPER_SHA256:
        print("[math12k] WARNING: differs from the paper's prompt file (tokenizer or dataset revision?)")


if __name__ == "__main__":
    main()
