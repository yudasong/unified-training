"""Many-sample pass@k (default n=256) on MATH-500 / GSM8K.

Same prompts, chat template, sampling (T=1, top-k 50, top-p 0.95, 1024 new tokens, stop <|im_end|>) and
math-verify scorer as the 16-sample lm-eval tasks; only the sample count differs. One call scores one shard
of the problems on one GPU and writes per-problem correct counts; ``--merge`` combines the shards.
"""

from __future__ import annotations

import json
from pathlib import Path

from unified_training.evals import math_answers
from unified_training.evals.passk import MANY_KS, aggregate_counts

TASKS = {
    "gsm8k": dict(path="openai/gsm8k", name="main", revision="e53f048856ff4f594e959d75785d2c2d37b678ee",
                  split="test", prompt=math_answers.doc_to_gsm8k, gold=math_answers.gsm8k_gold_answer),
    "math500": dict(path="HuggingFaceH4/MATH-500", name=None, revision="6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
                    split="test", prompt=math_answers.doc_to_math, gold=lambda d: d["answer"]),
}


def load_docs(task: str) -> list[dict]:
    from datasets import load_dataset

    spec = TASKS[task]
    return list(load_dataset(spec["path"], spec["name"], revision=spec["revision"], split=spec["split"]))


def run_shard(task: str, model: str, n: int, shard: str, output: Path, seed: int = 1234, limit: int | None = None) -> Path:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    spec = TASKS[task]
    docs = load_docs(task)
    if limit:
        docs = docs[:limit]
    idx, world = (int(x) for x in shard.split("/"))
    mine = [(i, d) for i, d in enumerate(docs) if i % world == idx]
    tokenizer = AutoTokenizer.from_pretrained(model)
    prompts = [tokenizer.apply_chat_template([{"role": "user", "content": spec["prompt"](d)}],
                                             tokenize=False, add_generation_prompt=True) for _, d in mine]
    llm = LLM(model=model, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.85,
              enable_prefix_caching=True, enforce_eager=True, seed=seed)
    params = SamplingParams(n=n, temperature=1.0, top_k=50, top_p=0.95, max_tokens=1024,
                            stop=["<|im_end|>"], seed=seed + idx)
    outputs = llm.generate(prompts, params)
    rows = []
    for (i, d), out in zip(mine, outputs):
        texts = [o.text for o in out.outputs]
        if len(texts) != n:
            raise SystemExit(f"doc {i}: expected {n} samples, got {len(texts)}")
        gold = spec["gold"](d)
        rows.append({"doc": i, "n": n, "correct": sum(math_answers.math_correct(gold, t) for t in texts)})
    out_path = Path(output) / f"{task}.shard{idx}of{world}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"task": task, "model": model, "n": n, "seed": seed, "shard": shard,
                                    "rows": rows}, indent=1))
    return out_path


def merge(task: str, n: int, output: Path) -> dict:
    shards = sorted(Path(output).glob(f"{task}.shard*of*.json"))
    if not shards:
        raise SystemExit(f"no shards for {task} under {output}")
    rows, worlds = [], set()
    for s in shards:
        d = json.loads(s.read_text())
        worlds.add(int(d["shard"].split("/")[1]))
        rows += d["rows"]
    if len(worlds) != 1 or len(shards) != next(iter(worlds)):
        raise SystemExit(f"incomplete shard set for {task}: {[s.name for s in shards]}")
    rows.sort(key=lambda r: r["doc"])
    if not all(r["n"] == n for r in rows):
        raise SystemExit(f"{task}: shards were sampled with a different n than {n}")
    summary = {"task": task, "n": n, "docs": len(rows), **aggregate_counts([r["correct"] for r in rows], n, MANY_KS)}
    Path(output, f"{task}.passk{n}.json").write_text(json.dumps(summary, indent=2))
    return summary
