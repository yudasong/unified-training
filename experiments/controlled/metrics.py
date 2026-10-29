"""Accuracy, coverage and compute of every evaluated checkpoint, and two-seed means (Appendices A.2, C.4, C.10).

    python metrics.py [--eval-dir DIR | --results evals.parquet] [--runs-dir DIR] [--out DIR]   (defaults: $EVAL_DIR, $RUNS_DIR)

Writes three CSV files to --out:
  checkpoints.csv  one row per evaluated model and test: run id, its settings, update (0 = initialization), pass@1,
                   pass@16 and, when the run's trainer logs are in --runs-dir, RL prompt/response tokens, midtraining
                   tokens during RL and model FLOPs (total and beyond the shared pretraining)
  means.csv        two-seed means per configuration, test and update, only where both seeds were evaluated; scores are
                   computed per model first, never from pooled responses
  retention.csv    held-out pretraining loss and the five benchmarks of each initialization (eval_init.sh)
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path

from nl_tower import compute, results

KS = (1, 16)
BENCH = (("mmlu", "acc,none"), ("arc_easy", "acc_norm,none"), ("hellaswag", "acc_norm,none"),
         ("truthfulqa_mc2", "acc,none"), ("winogrande", "acc,none"))  # fmt: skip


def tokens_and_flops(row, step, runs_dir):
    """Token counts and FLOPs of a run through ``step`` from its trainer logs ({} without logs)."""
    logs = sorted(glob.glob(str(Path(runs_dir) / row["id"] / "logs" / "trainer*.log"))) if runs_dir else []
    loss_tokens = compute.logged_loss_tokens(logs)
    if not loss_tokens or max(loss_tokens) < step:
        return {}
    m = int(row["m"])
    resp = compute.rl_response_tokens(loss_tokens, m, step)
    return {
        "rl_prompt_tokens": step * compute.RL_ROWS * compute.PROMPT_TOKENS[row["setting"]],
        "rl_response_tokens": resp,
        "midtraining_tokens_rl": step * m * compute.MID_EXAMPLE_TOKENS,
        "flops_total": compute.flops(row["init"], row["setting"], m, step, resp),
        "flops_beyond_pretraining": compute.flops(row["init"], row["setting"], m, step, resp, additional=True),
    }


def retention(eval_dir, init):
    d = Path(eval_dir) / "retention" / init
    out = {}
    loss = d / "pretrain_holdout_loss.json"
    if loss.exists():
        j = json.loads(loss.read_text())
        out.update(climbmix_loss=j["per_corpus"]["climbmix"]["nats_per_token"],
                   dolmino_loss=j["per_corpus"]["dolmino"]["nats_per_token"], mean_loss=j["weighted_nats"])  # fmt: skip
    found = sorted(glob.glob(str(d / "retention" / "ll_battery" / "**" / "results_*.json"), recursive=True))
    if found:
        res = json.loads(Path(found[-1]).read_text())["results"]
        for task, metric in BENCH:
            out[task] = 100 * res[task][metric]
        out["mean_score"] = sum(out[t] for t, _ in BENCH) / len(BENCH)
    return out


def write(path, rows):
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print("wrote", path, len(rows), "rows")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=str(Path(__file__).with_name("runs.tsv")))
    ap.add_argument("--eval-dir", default=os.environ.get("EVAL_DIR", "outputs/controlled/eval"),
                    help="score JSONs of eval_run.sh / eval_init.sh (default: $EVAL_DIR, else outputs/controlled/eval)")
    ap.add_argument("--results", default=None, help="public results table (evals.parquet) instead of --eval-dir")
    ap.add_argument("--runs-dir", default=os.environ.get("RUNS_DIR"),
                    help="RL run directories (trainer logs, for tokens and FLOPs; default: $RUNS_DIR)")
    ap.add_argument("--out", default="outputs/controlled")
    a = ap.parse_args()
    runs = results.read_runs(a.runs)
    by_id = {r["id"]: r for r in runs}
    recs = results.load_eval_dir(a.eval_dir) if Path(a.eval_dir).is_dir() else []
    if a.results:  # the paper's own evaluations; initializations still come from --eval-dir
        recs = [r for r in recs if r["run"].startswith("init_")] + results.load_results_parquet(a.results, runs)
    if not recs:
        raise SystemExit(f"no score files in {a.eval_dir} (set --eval-dir or EVAL_DIR, or --results for the public table)")
    per_model = {k: results.scores(recs, k) for k in KS}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rows, flops = [], {}
    for key in sorted(per_model[1]):
        run, test, step = key
        row = by_id.get(run, {})
        r = {"run": run, "setting": row.get("setting", ""), "recipe": row.get("recipe", ""),
             "init": row.get("init", run[len("init_"):] if run.startswith("init_") else ""), "m": row.get("m", ""),
             "seed": row.get("seed", ""), "test": test, "step": step}  # fmt: skip
        r.update({f"pass@{k}": per_model[k][key] for k in KS})
        if row:
            tf = tokens_and_flops(row, step, a.runs_dir) if step else {}
            r.update(tf)
            flops[(run, step)] = tf.get("flops_total")
        elif run.startswith("init_"):
            init = r["init"]
            r["flops_total"] = compute.flops(init, "base", 0, 0, 0.0)
            r["flops_beyond_pretraining"] = compute.flops(init, "base", 0, 0, 0.0, additional=True)
        rows.append(r)
    write(out / "checkpoints.csv", rows)

    means = {k: results.seed_means(per_model[k]) for k in KS}
    fl = defaultdict(list)
    for (run, step), f in flops.items():
        if f is not None:
            fl[(results.config_of(run), step)].append(f)
    mrows = []
    for key in sorted(means[1]):
        config, test, step = key
        r = {"config": config, "test": test, "step": step, "seeds": "+".join(results.SEEDS)}
        r.update({f"pass@{k}": means[k][key] for k in KS})
        f = fl.get((config, step), [])
        if len(f) == len(results.SEEDS):
            r["flops_total"] = sum(f) / len(f)
            r["flops_beyond_pretraining"] = r["flops_total"] - 6 * compute.N_PARAMS * compute.T_PRE
        mrows.append(r)
    write(out / "means.csv", mrows)

    inits = sorted({r["init"] for r in runs} | {"pretrain-only"})
    rrows = [{"init": i, **retention(a.eval_dir, i)} for i in inits]
    write(out / "retention.csv", [r for r in rrows if len(r) > 1] or [{"init": ""}])


if __name__ == "__main__":
    main()
