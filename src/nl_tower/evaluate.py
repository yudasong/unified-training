"""Accuracy and coverage of a checkpoint on an nl_tower test set (Appendix C.4).

    python -m nl_tower.evaluate --spec nltw_m --model CKPT --test TEST.jsonl --out OUT.json [--n 64 --max-tokens 1280]

Every question is answered with ``--n`` sampled responses (temperature 1, top-p 1, no top-k, vLLM seed 0, 2,048-token
context, at most 1,280 new tokens) plus one greedy response. A response is correct when its last boxed answer gives
the complete gold assignment (answers.py). Output JSON (one per checkpoint and test):

  sample: acc (mean correctness of all samples = pass@1), pass@k for k = 1, 2, 4, ..., n, coverage (any correct),
          box_rate, cap_rate (share stopped by the token limit), mean_tokens, per_q = [[qid, n, correct], ...],
          examples (first three questions, truncated)
  greedy: acc, box_rate, cap_rate, mean_tokens, examples
  spec, model, n, temp, n_test

The paper reports sample pass@1 (accuracy) and pass@16 (coverage), averaged over the two RL seeds of a configuration
after scoring each model separately (nl_tower.results).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from .answers import eval_correct, last_boxed_answer
from .passk import pass_at_k
from .specs import make_dataset

KS = (1, 2, 4, 8, 16, 32, 64)


def score_outputs(
    ds, test: list[dict], outputs: list[list[tuple[str, int, bool]]], sampled: bool = True, n_head: int = 256
) -> dict:
    """Score ``outputs[i]`` = [(text, n_tokens, hit_token_limit), ...] of test row i. ``sampled`` adds pass@k,
    coverage and the per-question counts (the ``sample`` block)."""
    per_q, flat, toks, boxed, capped, examples = [], [], [], [], [], []
    for t, outs in zip(test, outputs, strict=True):
        entry = {"question": t["question"], "answer": t["answer"], "metadata": t["metadata"]}
        c = 0
        for text, n_tok, hit in outs:
            ans = last_boxed_answer(text)
            ok = eval_correct(ds, ans, entry)
            boxed.append(int(ans is not None))
            c += ok
            flat.append(ok)
            toks.append(n_tok)
            capped.append(hit)
        per_q.append((len(outs), c))
        if len(examples) < 3:
            examples.append({"q": t["question"][:300], "gold": t["answer"], "out": outs[0][0][:600]})
    r = {"acc": statistics.fmean(flat), "box_rate": statistics.fmean(boxed), "mean_tokens": statistics.fmean(toks),
         "cap_rate": statistics.fmean(capped), "examples": examples}  # fmt: skip
    if sampled:
        n = min(len(o) for o in outputs)
        for k in KS:
            if k <= n:
                r[f"pass@{k}"] = statistics.fmean(pass_at_k(nn, c, k) for nn, c in per_q)
        r["coverage"] = statistics.fmean(c > 0 for _, c in per_q)
        r["per_q"] = [[t["qid"], nn, c] for t, (nn, c) in zip(test, per_q)]
        if len(test) > n_head:  # the first 256 questions, comparable with earlier, shorter test sets
            for k in (1, 16, 64):
                if k <= n:
                    r[f"pass@{k}_head{n_head}"] = statistics.fmean(pass_at_k(nn, c, k) for nn, c in per_q[:n_head])
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True, help="rung of the test set, e.g. nltw_m, nltw_hb, nltw_hb12, nltw_m0")
    ap.add_argument("--model", required=True, help="HF checkpoint directory (with tokenizer)")
    ap.add_argument("--test", required=True, help="test JSONL (data/<spec>/corpus_tw1/test.jsonl)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=1280)
    ap.add_argument("--gpu-mem", type=float, default=0.4)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams

    ds = make_dataset(a.spec, size=1, seed=1)  # only its score_answer is used
    test = [json.loads(line) for line in open(a.test)]
    llm = LLM(a.model, dtype="bfloat16", gpu_memory_utilization=a.gpu_mem, max_model_len=2048, seed=0, tokenizer=a.model)
    prompts = [t["prompt"] for t in test]
    res = {}
    for name, sp in (
        ("sample", SamplingParams(n=a.n, temperature=a.temp, top_p=1.0, max_tokens=a.max_tokens, seed=0)),
        ("greedy", SamplingParams(n=1, temperature=0.0, max_tokens=a.max_tokens)),
    ):
        outs = llm.generate(prompts, sp)
        texts = [[(s.text, len(s.token_ids), s.finish_reason == "length") for s in o.outputs] for o in outs]
        res[name] = score_outputs(ds, test, texts, sampled=name == "sample")
    res.update({"spec": a.spec, "model": a.model, "n": a.n, "temp": a.temp, "n_test": len(test)})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    s = res["sample"]
    print("EVAL", a.spec, a.model, json.dumps({k: round(v, 4) for k, v in s.items() if isinstance(v, float)}),
          "greedy", round(res["greedy"]["acc"], 4), flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
