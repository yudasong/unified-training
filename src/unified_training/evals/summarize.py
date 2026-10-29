"""Flatten one checkpoint's evaluation outputs into the metrics reported in the paper (percent; loss in nats).

Reads the layout written by ``battery`` (``noncode/``, ``code/``, ``passk/``, ``retention/``) and also the
layout of the original evaluation outputs (separate roots for the 16-sample, many-sample and retention
evaluations: pass them all), so figures can be rebuilt from either.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from statistics import mean

GEN_KEYS = {
    "math500": ("math500_16", "pass_at_1,all_16"), "gsm8k": ("gsm8k_16", "pass_at_1,all_16"),
    "sciq": ("sciq_greedy", "greedy_accuracy,greedy"), "arc_c": ("arc_challenge_greedy", "greedy_accuracy,greedy"),
}
# log-likelihood tasks: "<name>" = acc, "<name>_norm" = acc_norm when the task reports it
LL_TASKS = {"mmlu": "mmlu", "arc_e": "arc_easy", "hellaswag": "hellaswag", "truthfulqa": "truthfulqa_mc2",
            "winogrande": "winogrande"}


def _newest(paths):
    paths = sorted(paths, key=lambda p: p.name)
    return paths[-1] if paths else None


def _lm_eval_results(root: Path, task: str) -> dict | None:
    for base in (root / "noncode" / task, root / "retention" / "ll_battery", root / "ll_battery"):
        f = _newest([p for p in base.rglob("results_*.json")] if base.is_dir() else [])
        if f is not None:
            res = json.loads(f.read_text())["results"]
            if task in res:
                return res[task]
    ll = root / "ll_battery.json"
    if ll.exists():
        res = json.loads(ll.read_text())["results"]
        return res.get(task)
    return None


def _first_json(candidates):
    for c in candidates:
        if c.exists():
            return json.loads(c.read_text())
    return None


def collect(roots: list[Path], base_holdout: float | None = None) -> dict:
    roots = [Path(r) for r in roots]
    m: dict[str, float] = {}
    for root in roots:
        for name, (task, key) in GEN_KEYS.items():
            r = _lm_eval_results(root, task)
            if r and key in r:
                m[f"{name}_pass1" if name in ("math500", "gsm8k") else name] = 100 * r[key]
                if name in ("math500", "gsm8k") and "pass_at_16,all_16" in r:
                    m[f"{name}_pass16"] = 100 * r["pass_at_16,all_16"]
        for name, task in LL_TASKS.items():
            r = _lm_eval_results(root, task)
            for key, suffix in (("acc,none", ""), ("acc_norm,none", "_norm")):
                if r and key in r:
                    m[name + suffix] = 100 * r[key]
        for ds, short in (("humaneval", "humaneval_plus"), ("mbpp", "mbpp_plus")):
            s16 = _first_json([root / "code" / f"{ds}.passk16.json"])
            if s16:
                m[f"{short}_pass1"] = 100 * s16["plus"]["pass_at_1"]
                m[f"{short}_pass16"] = 100 * s16["plus"]["pass_at_16"]
            for f in sorted((root / "code").glob(f"{ds}*.passk*.json")) if (root / "code").is_dir() else []:
                n = int(f.name.rsplit(".passk", 1)[1].split(".")[0])
                if n > 16:
                    s = json.loads(f.read_text())
                    for k in (32, 64, 128, 256):
                        if f"pass_at_{k}" in s["plus"]:
                            m[f"{short}_pass{k}"] = 100 * s["plus"][f"pass_at_{k}"]
        for task in ("math500", "gsm8k"):
            for f in [*root.glob(f"{task}.passk*.json"), *(root / "passk").glob(f"{task}.passk*.json")]:
                s = json.loads(f.read_text())
                for k in (32, 64, 128, 256):
                    if f"pass_at_{k}" in s:
                        m[f"{task}_pass{k}"] = 100 * s[f"pass_at_{k}"]
        hl = _first_json([root / "retention" / "holdout_loss.json", root / "holdout_loss.json"])
        if hl:
            m["holdout_loss"] = hl["macro_avg_nats"]
    for k in (1, 16, 32, 64, 256):
        for pair, name in ((("math500", "gsm8k"), "math"), (("humaneval_plus", "mbpp_plus"), "coding")):
            keys = [f"{p}_pass{k}" for p in pair]
            if all(key in m for key in keys):
                m[f"{name}_pass{k}"] = mean(m[key] for key in keys)
    if base_holdout is not None and "holdout_loss" in m:
        m["holdout_delta"] = m["holdout_loss"] - base_holdout
    return dict(sorted(m.items()))


def summarize(roots: list[Path], output: Path | None = None, base_holdout: float | None = None) -> dict:
    metrics = collect(roots, base_holdout)
    if output is not None:
        Path(output).write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


VALIDATION_RE = re.compile(r"validation @ (\d+): lm loss ([0-9.eE+-]+)(?: \| midtrain lm ([0-9.eE+-]+))?")


def trainer_validation(run_dir: Path) -> dict[int, dict[str, float]]:
    """In-training validation losses (every UNIFIED_EVAL_INTERVAL updates) parsed from a run's trainer logs."""
    out: dict[int, dict[str, float]] = {}
    for log in sorted(Path(run_dir).glob("logs/trainer*.log")):
        for m in VALIDATION_RE.finditer(log.read_text(errors="replace")):
            row = {"val_lm": float(m.group(2))}
            if m.group(3):
                row["val_midtrain_lm"] = float(m.group(3))
            out[int(m.group(1))] = row
    return out


def table(eval_root: Path, out_csv: Path, runs_root: Path | None = None, base_holdout: float | None = None) -> list[dict]:
    """One CSV row per (config, step) from EVAL_ROOT/<config>/step_<N>/ (and RUNS_ROOT/<config>/logs)."""
    eval_root = Path(eval_root)
    if base_holdout is None:  # the paper subtracts the base model's measured loss
        base = sorted(eval_root.glob("base/step_*/retention/holdout_loss.json"))
        base_holdout = json.loads(base[0].read_text())["macro_avg_nats"] if base else None
    rows: dict[tuple[str, int], dict] = {}
    for step_dir in sorted(eval_root.glob("*/step_*")):
        key = (step_dir.parent.name, int(step_dir.name.split("_")[1]))
        metrics = collect([step_dir], base_holdout)
        rows[key] = {"config": key[0], "step": key[1], **metrics}
    if runs_root is not None:
        for run in sorted(p for p in Path(runs_root).iterdir() if p.is_dir()):
            for step, vals in trainer_validation(run).items():
                rows.setdefault((run.name, step), {"config": run.name, "step": step}).update(vals)
    ordered = [rows[k] for k in sorted(rows)]
    columns = ["config", "step"] + sorted({c for r in ordered for c in r} - {"config", "step"})
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(ordered)
    return ordered
