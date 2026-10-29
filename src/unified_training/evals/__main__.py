"""python -m unified_training.evals <command> ...  (see the package docstring for the protocol)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _csv(value: str) -> list[str]:
    return [v for v in value.split(",") if v]


def _kv(value: str) -> dict[str, str]:
    return dict(item.split("=", 1) for item in _csv(value))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m unified_training.evals")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("sampled", help="lm-eval generation tasks (16-sample math, greedy MCQ)")
    p.add_argument("--model", required=True)
    p.add_argument("--tasks", type=_csv, default="math500_16,gsm8k_16,sciq_greedy,arc_challenge_greedy")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int)

    p = sub.add_parser("math-passk", help="many-sample pass@k on MATH-500 / GSM8K (one shard, or --merge)")
    p.add_argument("--task", required=True, choices=("math500", "gsm8k"))
    p.add_argument("--model")
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--shard", default="0/1")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--limit", type=int)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--merge", action="store_true")

    p = sub.add_parser("code", help="EvalPlus HumanEval+ / MBPP+ (generate, execute, pass@k)")
    p.add_argument("--model")
    p.add_argument("--dataset", required=True, choices=("humaneval", "mbpp"))
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--phase", choices=("all", "generate", "execute"), default="all")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--parallel", type=int, default=16)
    p.add_argument("--task-limit", type=int)
    p.add_argument("--deadline-s", type=float, help="score samples unfinished after this many seconds as "
                   "timeouts (paper: none for n=16, 3600 for n=32)")

    p = sub.add_parser("retention", help="lm-eval log-likelihood tasks, 0-shot, HF backend, no chat template")
    p.add_argument("--model", required=True)
    p.add_argument("--tasks", type=_csv, default="mmlu")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int)

    p = sub.add_parser("holdout-loss", help="held-out loss (slices dir or Megatron shards)")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--slices", type=Path)
    g.add_argument("--megatron", type=_kv, help="name=PREFIX[,name=PREFIX]")
    p.add_argument("--weights", type=_kv, default={}, help="name=w[,name=w] (megatron)")
    p.add_argument("--tokens-per-corpus", type=int, default=400_000)
    p.add_argument("--seed", type=int, default=20260905)

    p = sub.add_parser("battery", help="all suites for one checkpoint (resumable)")
    p.add_argument("--model", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--suite", type=_csv, default="eval16,passk,retention")
    p.add_argument("--slices", type=Path, help="held-out midtraining slices (retention suite)")
    p.add_argument("--gpus", type=_csv, help="GPU ids for the many-sample shards (default: first visible)")
    p.add_argument("--math-n", type=int, default=256)
    p.add_argument("--code-n", type=int, default=32)
    p.add_argument("--retention-tasks", type=_csv, default="mmlu")
    p.add_argument("--base-holdout", type=float)
    p.add_argument("--limit", type=int, help="smoke test: first N problems per benchmark")

    p = sub.add_parser("table", help="metrics of every EVAL_ROOT/<config>/step_<N> (+ trainer validation) -> CSV")
    p.add_argument("eval_root", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--runs", type=Path, help="training runs root (adds the in-training validation losses)")
    p.add_argument("--base-holdout", type=float, help="default: EVAL_ROOT/base/step_*/retention/holdout_loss.json")

    p = sub.add_parser("summarize", help="flatten eval outputs of one checkpoint to paper metrics")
    p.add_argument("roots", type=Path, nargs="+")
    p.add_argument("--output", type=Path)
    p.add_argument("--base-holdout", type=float)

    a = ap.parse_args(argv)
    if a.cmd == "sampled":
        from unified_training.evals import lm_eval_runner

        tasks = a.tasks if isinstance(a.tasks, list) else _csv(a.tasks)
        lm_eval_runner.run_generation(a.model, tasks, a.output, a.limit)
    elif a.cmd == "math-passk":
        from unified_training.evals import math_passk

        if a.merge:
            print(json.dumps(math_passk.merge(a.task, a.n, a.output)))
        else:
            if not a.model:
                ap.error("--model is required for a shard run")
            math_passk.run_shard(a.task, a.model, a.n, a.shard, a.output, a.seed, a.limit)
    elif a.cmd == "code":
        from unified_training.evals import code_eval

        out = code_eval.run(a.model, a.dataset, a.n, a.output, a.seed, a.parallel, a.task_limit, a.deadline_s, a.phase)
        if out is not None:
            print(json.dumps(out))
    elif a.cmd == "retention":
        from unified_training.evals import lm_eval_runner

        tasks = a.tasks if isinstance(a.tasks, list) else _csv(a.tasks)
        lm_eval_runner.run_retention(a.model, tasks, a.output, a.limit)
    elif a.cmd == "holdout-loss":
        from unified_training.evals import holdout_loss

        if a.slices:
            holdout_loss.eval_slices(a.model, a.slices, a.output)
        else:
            weights = {k: float(v) for k, v in a.weights.items()}
            corpora = {k: Path(v) for k, v in a.megatron.items()}
            holdout_loss.eval_megatron(a.model, corpora, weights, a.output, a.tokens_per_corpus, a.seed)
    elif a.cmd == "battery":
        from unified_training.evals import battery

        suites = a.suite if isinstance(a.suite, list) else _csv(a.suite)
        rt = a.retention_tasks if isinstance(a.retention_tasks, list) else _csv(a.retention_tasks)
        print(json.dumps(battery.run(a.model, a.output, suites, a.slices, a.gpus, a.math_n, a.code_n, rt,
                                     a.limit, a.base_holdout), indent=2))
    elif a.cmd == "table":
        from unified_training.evals import summarize

        rows = summarize.table(a.eval_root, a.out, a.runs, a.base_holdout)
        print(f"{len(rows)} rows -> {a.out}")
    elif a.cmd == "summarize":
        from unified_training.evals import summarize

        print(json.dumps(summarize.summarize(a.roots, a.output, a.base_holdout), indent=2))


if __name__ == "__main__":
    main()
