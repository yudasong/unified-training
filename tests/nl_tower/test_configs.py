"""experiments/controlled: every run of runs.tsv resolves to a complete, explicit trainer configuration (CPU only)."""

from __future__ import annotations

import csv
import os
import subprocess
from pathlib import Path

import pytest

X = Path(__file__).resolve().parents[2] / "experiments" / "controlled"
RUNS = list(csv.DictReader(open(X / "runs.tsv"), delimiter="\t"))


def resolve(row):
    files = [X / "configs" / f for f in ("common.env", f"{row['setting']}.env", f"init/{row['init']}.env",
                                          f"mix/m{row['m']}.env", f"recipe/{row['recipe']}.env", f"seed/s{row['seed']}.env",
                                          "layout/colocated4.env")]  # fmt: skip
    for f in files:
        assert f.is_file(), f
    script = "set -a; " + "; ".join(f"source {f}" for f in files) + "; env"
    env = dict(os.environ, DATA_DIR="/D", CKPT_DIR="/C")
    env.pop("NL_TOWER_WORKDIR", None)
    out = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, check=True).stdout
    return dict(line.split("=", 1) for line in out.splitlines() if line.startswith("UNIFIED_"))


def test_table():
    assert len(RUNS) == 360
    assert len({r["id"] for r in RUNS}) == len(RUNS)
    for r in RUNS:
        assert r["id"] == f"{r['setting']}_{r['recipe']}_{r['init']}_m{r['m']}_s{r['seed']}"
        assert r["steps"] == {"base": "6000", "extended": "12000"}[r["setting"]]
        assert (r["recipe"] == "rl") == (r["m"] == "0")
        assert r["recipe"] != "rl" or r["init"].startswith("staged") or r["setting"] == "extended"
        assert r["init"] in ("mixed-133M", "mixed-267M", "mixed-534M", "mixed-1.07B", "mixed-2.14B") or (
            r["init"].startswith("staged-") and r["recipe"] == "rl")


@pytest.mark.parametrize("row", RUNS, ids=[r["id"] for r in RUNS])
def test_resolved_env(row):
    env = resolve(row)
    m = int(row["m"])
    assert env["UNIFIED_ANCHOR_MODE"] == ("replace" if m == 0 else "add")  # set explicitly for every run
    assert env["UNIFIED_ANCHOR_LOSS"] == ("sft" if row["recipe"].startswith("separate") else "pooled")
    assert float(env["UNIFIED_ANCHOR_TOKEN_FRACTION"]) == m / 512
    assert ("UNIFIED_ANCHOR_DIRS" in env) == (m > 0)
    assert ("UNIFIED_SEP_SFT_ALPHA" in env) == row["recipe"].startswith("separate")
    if row["recipe"].startswith("separate"):
        assert env["UNIFIED_SEP_SFT_ALPHA"] == row["recipe"].split("-a")[1]
    assert ("UNIFIED_PRETRAIN_DIRS" in env) == (row["recipe"] == "pre-mid-rl")
    if row["recipe"] == "pre-mid-rl":
        assert env["UNIFIED_PRETRAIN_DIRS"].endswith(f"pretrain_tw1X_p10_m{m}") and env["UNIFIED_PRETRAIN_ROWS_PER_STEP"] == "4"
    assert env["UNIFIED_TRAIN_ITERS"] == row["steps"] and env["UNIFIED_SPLIT_SEED"] == row["seed"]
    assert env["UNIFIED_PROMPTS_JSON"].startswith("/D/nl_tower/unified_data/")
    assert env["UNIFIED_POLICY_HF"] == f"/C/controlled/{row['init']}"
    assert (env["UNIFIED_GENERATOR_GPU_IDS"], env["UNIFIED_TRAINER_GPU_IDS"], env["UNIFIED_GPU_MEM_UTIL"]) == (
        "0,1,2,3", "0,1,2,3", "0.5")  # the paper's layout: generator and trainer share one 4-GPU node
    # Table 7: 32 prompts x 16 generations, lr 1e-5 constant, AdamW (0.9, 0.999), wd 0.01, 1,280 response tokens
    assert (env["UNIFIED_GBS"], env["UNIFIED_G"], env["UNIFIED_LR"], env["UNIFIED_ADAM_BETA2"], env["UNIFIED_WD"],
            env["UNIFIED_MAX_TOKENS"], env["UNIFIED_WARMUP_ITERS"], env["UNIFIED_MAXRL_BASELINE"]) == (
            "512", "16", "1e-5", "0.999", "0.01", "1280", "0", "0")  # fmt: skip
