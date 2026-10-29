"""All evaluations of one checkpoint, resumable (finished parts are skipped).

Suites (the paper's evaluation schedule in parentheses):
* ``eval16``: MATH-500 / GSM8K 16-sample pass@k, SciQ / ARC-C greedy, HumanEval+ / MBPP+ 16-sample
  (every 1000 updates).
* ``passk``: MATH-500 / GSM8K with 256 samples, HumanEval+ / MBPP+ with 32 samples (final checkpoints).
* ``retention``: held-out midtraining loss and MMLU (every 2000 updates).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from unified_training.evals import code_eval, lm_eval_runner, math_passk, summarize


def _passk_parallel(model: str, task: str, n: int, out: Path, gpus: list[str], limit: int | None) -> None:
    """One shard per GPU, run as subprocesses, then merge."""
    if (out / f"{task}.passk{n}.json").exists():
        return
    procs = []
    for i, gpu in enumerate(gpus):
        if (out / f"{task}.shard{i}of{len(gpus)}.json").exists():
            continue
        cmd = [sys.executable, "-m", "unified_training.evals", "math-passk", "--model", model, "--task", task,
               "--n", str(n), "--shard", f"{i}/{len(gpus)}", "--output", str(out)]
        if limit:
            cmd += ["--limit", str(limit)]
        procs.append(subprocess.Popen(cmd, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu}))
    if any(p.wait() for p in procs):
        raise RuntimeError(f"math-passk {task} shard failed")
    math_passk.merge(task, n, out)


def run(model: str, output: Path, suites: list[str], slices: Path | None = None, gpus: list[str] | None = None,
        math_n: int = 256, code_n: int = 32, retention_tasks: list[str] | None = None,
        limit: int | None = None, base_holdout: float | None = None, code_parallel: int = 16) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    gpus = gpus or [os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]]
    if "eval16" in suites:
        lm_eval_runner.run_generation(model, list(lm_eval_runner.GENERATION_TASKS), output, limit)
        for ds in ("humaneval", "mbpp"):
            code_eval.run(model, ds, 16, output, parallel=code_parallel, task_limit=limit)
    if "passk" in suites:
        for task in ("gsm8k", "math500"):
            _passk_parallel(model, task, math_n, output / "passk", gpus, limit)
        for ds in ("humaneval", "mbpp"):
            code_eval.run(model, ds, code_n, output, parallel=code_parallel, task_limit=limit, deadline_s=3600)
    if "retention" in suites:
        if slices is None:
            print("battery: retention without --slices: the held-out midtraining loss is skipped (MMLU only)",
                  file=sys.stderr, flush=True)
        elif not (output / "retention" / "holdout_loss.json").exists():
            from unified_training.evals import holdout_loss

            holdout_loss.eval_slices(Path(model), slices, output / "retention" / "holdout_loss.json")
        lm_eval_runner.run_retention(model, retention_tasks or list(lm_eval_runner.RETENTION_TASKS), output, limit)
    return summarize.summarize([output], output / "summary.json", base_holdout)
