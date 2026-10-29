"""lm-eval invocations: sampled/greedy generation tasks (vLLM) and log-likelihood retention tasks (HF)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

TASKS_DIR = Path(__file__).resolve().parent / "tasks"
GENERATION_TASKS = ("math500_16", "gsm8k_16", "sciq_greedy", "arc_challenge_greedy")
RETENTION_TASKS = ("mmlu",)  # the controlled setting adds arc_easy, hellaswag, truthfulqa_mc2, winogrande
SEED = 1234


def generation_command(model: str, task: str, output: Path, limit: int | None = None) -> list[str]:
    """One task per call, chat template applied, as in the paper's evaluation."""
    cmd = [sys.executable, "-m", "lm_eval", "--model", "vllm",
           "--model_args", (f"pretrained={model},dtype=bfloat16,max_model_len=2048,gpu_memory_utilization=0.85,"
                            f"enable_prefix_caching=True,enforce_eager=True,seed={SEED}"),
           "--tasks", task, "--include_path", str(TASKS_DIR), "--apply_chat_template",
           "--batch_size", "auto", "--seed", str(SEED), "--output_path", str(output), "--log_samples"]
    if limit:
        cmd += ["--limit", str(limit)]
    return cmd


def retention_command(model: str, tasks: list[str], output: Path, limit: int | None = None) -> list[str]:
    """0-shot log-likelihood scoring, HF backend, no chat template (plain continuation scoring)."""
    cmd = [sys.executable, "-m", "lm_eval", "--model", "hf",
           "--model_args", f"pretrained={model},dtype=bfloat16,trust_remote_code=True",
           "--tasks", ",".join(tasks), "--num_fewshot", "0", "--batch_size", "16", "--seed", str(SEED),
           "--log_samples", "--output_path", str(output)]
    if limit:
        cmd += ["--limit", str(limit)]
    return cmd


def run_generation(model: str, tasks: list[str], output_dir: Path, limit: int | None = None) -> None:
    for task in tasks:
        task_root = Path(output_dir) / "noncode" / task
        done = task_root / "GENERATION_DONE"
        if done.exists():
            continue
        task_root.mkdir(parents=True, exist_ok=True)
        subprocess.run(generation_command(model, task, task_root / "results", limit), check=True)
        done.write_text("done\n")


def run_retention(model: str, tasks: list[str], output_dir: Path, limit: int | None = None) -> None:
    root = Path(output_dir) / "retention"
    done = root / f"LL_DONE.{'+'.join(tasks)}"
    if done.exists():
        return
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(retention_command(model, tasks, root / "ll_battery", limit), check=True)
    done.write_text("done\n")
