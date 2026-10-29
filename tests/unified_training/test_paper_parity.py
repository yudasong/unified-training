# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""End-to-end CPU parity with the trainer revisions the paper's runs executed.

Each case runs the release trainer's main() for 6 updates on 2 CPU ranks (gloo) over a tiny fixed
fixture (cpu_trainer.py) and compares every logged update (loss, grad norm, token counts, anchor /
pretraining rows, KL, validation losses) with golden values recorded by running the paper's own trainer
revision on the same fixture with the same CPU replacements:
  controlled runs (Sec. 5.1, App. C): snapshot 8e8a4b4 (+ the App. C.8 pre+mid+RL patch);
  real-world runs (Sec. 5.2, App. D, E): main at 3717d96.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "src"

COMMON = {
    "UNIFIED_TRAIN_ITERS": "6", "UNIFIED_GBS": "16", "UNIFIED_MBS": "2", "UNIFIED_SEQ": "32", "UNIFIED_LR": "1e-3",
    "UNIFIED_PAD_TOKEN_ID": "0", "UNIFIED_SAVE_INTERVAL": "1000", "UNIFIED_EVAL_MILESTONE_INTERVAL": "3",
    "UNIFIED_LOG_INTERVAL": "1", "UNIFIED_GEN_WORLD": "2", "UNIFIED_STREAMING": "1", "UNIFIED_RESCAN_EVERY": "512",
    "UNIFIED_LOSS_NORM": "response_tokens", "UNIFIED_GRAD_CLIP": "1.0", "UNIFIED_WARMUP_ITERS": "0",
    "UNIFIED_KEEP_POLICIES": "24", "UNIFIED_NCCL_TIMEOUT_SECONDS": "600", "UNIFIED_TEMP": "1.0",
}
CONTROLLED = {
    **COMMON, "UNIFIED_TOP_K": "-1", "UNIFIED_TOP_P": "1.0", "UNIFIED_WD": "0.01", "UNIFIED_ADAM_BETA2": "0.999",
    "UNIFIED_WITHOUT_REPLACEMENT": "1", "UNIFIED_RECENCY_WINDOW": "4", "UNIFIED_PUBLISH_INTERVAL": "1",
    "UNIFIED_MAX_POLICY_AGE_STEPS": "1", "UNIFIED_STRICT_POLICY_AGE": "1", "UNIFIED_ROLLOUT_WAIT_TIMEOUT_SECONDS": "60",
    "UNIFIED_ROLLOUT_POLL_SECONDS": "1", "UNIFIED_EVAL_INTERVAL": "0", "UNIFIED_LOG_LOSS_TOKENS": "1",
    "UNIFIED_ANCHOR_DIRS": "{fx}/anchor_demo", "UNIFIED_ANCHOR_MODE": "add", "UNIFIED_ANCHOR_ADD_BASIS": "rows",
    "UNIFIED_ANCHOR_TOKEN_FRACTION": "0.125", "UNIFIED_MIX_RECALC": "launch", "UNIFIED_MIX_BASIS": "response_tokens",
    "UNIFIED_REQUIRE_NO_ANCHOR_REPLAY": "0", "UNIFIED_MAX_ANCHOR_ROW_FRACTION": "1.0", "_rounds": "rounds_maxrl",
}
REAL_WORLD = {
    **COMMON, "UNIFIED_TOP_K": "50", "UNIFIED_TOP_P": "0.95", "UNIFIED_WD": "0", "UNIFIED_ADAM_BETA2": "0.999",
    "UNIFIED_WITHOUT_REPLACEMENT": "0", "UNIFIED_RECENCY_WINDOW": "8", "UNIFIED_PUBLISH_INTERVAL": "4",
    "UNIFIED_MAX_POLICY_AGE_STEPS": "4", "UNIFIED_EVAL_INTERVAL": "3", "UNIFIED_EVAL_ITERS": "1",
    "UNIFIED_VALID_HOLDOUT": "16", "UNIFIED_MIDTRAIN_VAL_DIR": "{fx}/midtrain_holdout", "_rounds": "rounds_maxrl",
}
SCHEDULE = '[{"dirs":["{fx}/anchor_docs"],"token_fraction":{"start":0.1,"end":0.0,"end_progress":0.5}}]'
CASES = {
    # unified (pooled) loss, m rows of demonstrations per update, pretraining rows, strict policy age
    "controlled_pre_mid_rl": {**CONTROLLED, "UNIFIED_ANCHOR_LOSS": "pooled", "UNIFIED_PRETRAIN_DIRS": "{fx}/pretrain",
                                  "UNIFIED_PRETRAIN_ROWS_PER_STEP": "3"},
    # App. C.9 separately normalized loss
    "controlled_separate_alpha": {**CONTROLLED, "UNIFIED_ANCHOR_LOSS": "sft", "UNIFIED_SEP_SFT_ALPHA": "0.5"},
    # Sec. 5.2 half-anneal schedule, additive, REINFORCE rows, top-k/top-p reconstruction, midtraining holdout
    "realworld_schedule_reinforce": {**REAL_WORLD, "UNIFIED_ANCHOR_DIRS": "{fx}/anchor_docs",
                                     "UNIFIED_STREAMS_JSON": SCHEDULE, "UNIFIED_ANCHOR_MODE": "add",
                                     "UNIFIED_ANCHOR_ADD_BASIS": "tokens", "UNIFIED_ANCHOR_TOKEN_FRACTION": "0",
                                     "UNIFIED_MIX_RECALC": "launch", "UNIFIED_MIX_BASIS": "response_tokens",
                                     "UNIFIED_MAX_ANCHOR_ROW_FRACTION": "0.5", "UNIFIED_REQUIRE_NO_ANCHOR_REPLAY": "0",
                                     "UNIFIED_ANCHOR_LOSS": "pooled", "_rounds": "rounds_reinforce"},
    # App. E reference-KL penalty, no midtraining data
    "realworld_kl": {**REAL_WORLD, "UNIFIED_KL_COEF": "0.1", "UNIFIED_KL_REF_HF": "{fx}/init"},
}

# Recorded from the paper's trainer revisions (see the module docstring); timing fields removed.
GOLDEN: dict[str, list[str]] = {
    'controlled_separate_alpha': [
        "[unified-hf] iteration        1/6 | consumed samples: 16 | lm loss: 5.218451E+00 | grad norm: 0.097 | lr: 1.000E-03 | anchor rows added: 2 | loss tokens: 213 | sep rl term: 3.138307E+00 | sep demo nll: 4.160288E+00 | T_rl: 191 | T_demo: 22",
        "[unified-hf] iteration        2/6 | consumed samples: 32 | lm loss: 6.107991E+00 | grad norm: 0.093 | lr: 1.000E-03 | anchor rows added: 2 | loss tokens: 209 | sep rl term: 4.028137E+00 | sep demo nll: 4.159708E+00 | T_rl: 181 | T_demo: 28",
        "[unified-hf] iteration        3/6 | consumed samples: 48 | lm loss: 5.482525E+00 | grad norm: 0.118 | lr: 1.000E-03 | anchor rows added: 2 | loss tokens: 221 | sep rl term: 3.405076E+00 | sep demo nll: 4.154897E+00 | T_rl: 202 | T_demo: 19",
        "[unified-hf] iteration        4/6 | consumed samples: 64 | lm loss: 5.790646E+00 | grad norm: 0.116 | lr: 1.000E-03 | anchor rows added: 2 | loss tokens: 215 | sep rl term: 3.712307E+00 | sep demo nll: 4.156678E+00 | T_rl: 189 | T_demo: 26",
        "[unified-hf] iteration        5/6 | consumed samples: 80 | lm loss: 4.205158E+00 | grad norm: 0.090 | lr: 1.000E-03 | anchor rows added: 2 | loss tokens: 184 | sep rl term: 2.126155E+00 | sep demo nll: 4.158006E+00 | T_rl: 164 | T_demo: 20",
        "[unified-hf] iteration        6/6 | consumed samples: 96 | lm loss: 7.129349E+00 | grad norm: 0.114 | lr: 1.000E-03 | anchor rows added: 2 | loss tokens: 188 | sep rl term: 5.047636E+00 | sep demo nll: 4.163426E+00 | T_rl: 163 | T_demo: 25",
    ],
    'controlled_pre_mid_rl': [
        "[unified-hf] iteration        1/6 | consumed samples: 16 | lm loss: 3.336609E+00 | grad norm: 0.064 | lr: 1.000E-03 | anchor rows added: 2 | loss tokens: 237 | interval tokens rl/demo/pretrain: 191/22/24 | demo nll: 4.160288E+00 | pretrain nll: 4.159725E+00 | pretrain rows added: 3",
        "[unified-hf] iteration        2/6 | consumed samples: 32 | lm loss: 4.057107E+00 | grad norm: 0.059 | lr: 1.000E-03 | anchor rows added: 2 | loss tokens: 233 | interval tokens rl/demo/pretrain: 181/28/24 | demo nll: 4.159745E+00 | pretrain nll: 4.156497E+00 | pretrain rows added: 3",
        "[unified-hf] iteration        3/6 | consumed samples: 48 | lm loss: 3.537367E+00 | grad norm: 0.066 | lr: 1.000E-03 | anchor rows added: 2 | loss tokens: 245 | interval tokens rl/demo/pretrain: 202/19/24 | demo nll: 4.154805E+00 | pretrain nll: 4.162163E+00 | pretrain rows added: 3",
        "[unified-hf] iteration        4/6 | consumed samples: 64 | lm loss: 3.805508E+00 | grad norm: 0.066 | lr: 1.000E-03 | anchor rows added: 2 | loss tokens: 239 | interval tokens rl/demo/pretrain: 189/26/24 | demo nll: 4.156316E+00 | pretrain nll: 4.160921E+00 | pretrain rows added: 3",
        "[unified-hf] iteration        5/6 | consumed samples: 80 | lm loss: 2.555837E+00 | grad norm: 0.052 | lr: 1.000E-03 | anchor rows added: 2 | loss tokens: 208 | interval tokens rl/demo/pretrain: 164/20/24 | demo nll: 4.158105E+00 | pretrain nll: 4.156679E+00 | pretrain rows added: 3",
        "[unified-hf] iteration        6/6 | consumed samples: 96 | lm loss: 4.842272E+00 | grad norm: 0.082 | lr: 1.000E-03 | anchor rows added: 2 | loss tokens: 212 | interval tokens rl/demo/pretrain: 163/25/24 | demo nll: 4.162944E+00 | pretrain nll: 4.157810E+00 | pretrain rows added: 3",
    ],
    'realworld_kl': [
        "[unified-hf] validation @ 0: lm loss 2.372033 | midtrain lm 4.161889",
        "[unified-hf] iteration        1/6 | consumed samples: 16 | lm loss: 3.222994E+00 | grad norm: 0.084 | lr: 1.000E-03 | kl: 0.000000E+00",
        "[unified-hf] iteration        2/6 | consumed samples: 32 | lm loss: 3.872770E+00 | grad norm: 0.083 | lr: 1.000E-03 | kl: 6.748685E-06",
        "[unified-hf] iteration        3/6 | consumed samples: 48 | lm loss: 5.164733E+00 | grad norm: 0.111 | lr: 1.000E-03 | kl: 1.612890E-05",
        "[unified-hf] validation @ 3: lm loss 2.348302 | midtrain lm 4.160967",
        "[unified-hf] iteration        4/6 | consumed samples: 64 | lm loss: 1.757559E+00 | grad norm: 0.062 | lr: 1.000E-03 | kl: 2.826651E-05",
        "[unified-hf] iteration        5/6 | consumed samples: 80 | lm loss: 4.820745E+00 | grad norm: 0.096 | lr: 1.000E-03 | kl: 4.541511E-05",
        "[unified-hf] iteration        6/6 | consumed samples: 96 | lm loss: 2.152060E+00 | grad norm: 0.045 | lr: 1.000E-03 | kl: 6.423171E-05",
        "[unified-hf] validation @ 6: lm loss 2.241906 | midtrain lm 4.159530",
    ],
    'realworld_schedule_reinforce': [
        "[unified-hf] validation @ 0: lm loss 1.011745 | midtrain lm 4.161889",
        "[unified-hf] iteration        1/6 | consumed samples: 16 | lm loss: 2.038313E+00 | grad norm: 0.043 | lr: 1.000E-03 | anchor rows added: 1",
        "[unified-hf] iteration        2/6 | consumed samples: 32 | lm loss: 1.405706E+00 | grad norm: 0.033 | lr: 1.000E-03 | anchor rows added: 1",
        "[unified-hf] iteration        3/6 | consumed samples: 48 | lm loss: 2.107265E+00 | grad norm: 0.039 | lr: 1.000E-03 | anchor rows added: 0",
        "[unified-hf] validation @ 3: lm loss 0.975566 | midtrain lm 4.161206",
        "[unified-hf] iteration        4/6 | consumed samples: 64 | lm loss: 9.057520E-01 | grad norm: 0.028 | lr: 1.000E-03 | anchor rows added: 0",
        "[unified-hf] iteration        5/6 | consumed samples: 80 | lm loss: 1.592315E+00 | grad norm: 0.033 | lr: 1.000E-03 | anchor rows added: 0",
        "[unified-hf] iteration        6/6 | consumed samples: 96 | lm loss: 1.317666E+00 | grad norm: 0.030 | lr: 1.000E-03 | anchor rows added: 0",
        "[unified-hf] validation @ 6: lm loss 0.974638 | midtrain lm 4.159830",
    ],
}


def run_case(case: dict, fx: Path, run_dir: Path, src: str, package: str) -> str:
    policy = run_dir / "policies" / "v0000000"
    policy.mkdir(parents=True)
    for f in (fx / "init").iterdir():
        (policy / f.name).symlink_to(f)
    (policy / "DONE").touch()
    env = {k: v for k, v in os.environ.items() if not k.startswith("UNIFIED_")}
    env.update({k: v.replace("{fx}", str(fx)) for k, v in case.items() if not k.startswith("_")})
    env.update(UNIFIED_RUN_DIR=str(run_dir), UNIFIED_POLICY_HF=str(fx / "init"),
               UNIFIED_EPISODES_DIRS=str(fx / case["_rounds"]), CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
               PYTHONPATH=src, PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false")
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2",
           str(HERE / "cpu_trainer.py"), "run", src, package]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-4000:]
    return proc.stdout


def logged_updates(stdout: str) -> list[str]:
    lines = re.findall(r"\[unified-hf\] (?:iteration|validation @) .*", stdout)
    return [re.sub(r"ms per iteration: [0-9.]+ \| ", "", line) for line in lines]


def numbers_match(expected: str, actual: str) -> bool:
    pattern = r"-?\d+(?:\.\d+)?(?:E[+-]\d+)?"
    if re.sub(pattern, "#", expected) != re.sub(pattern, "#", actual):
        return False
    return all(math.isclose(float(a), float(b), rel_tol=1e-4, abs_tol=2e-3)
               for a, b in zip(re.findall(pattern, expected), re.findall(pattern, actual)))


@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory):
    fx = tmp_path_factory.mktemp("parity") / "fixture"
    proc = subprocess.run([sys.executable, str(HERE / "cpu_trainer.py"), "fixture", str(fx)],
                          env={**os.environ, "PYTHONPATH": str(SRC)}, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-4000:]
    return fx


@pytest.mark.parametrize("name", sorted(CASES))
def test_release_trainer_matches_the_paper_revision(name, fixture_dir, tmp_path):
    lines = logged_updates(run_case(CASES[name], fixture_dir, tmp_path / name, str(SRC), "unified_training"))
    golden = GOLDEN[name]
    assert len(lines) == len(golden), (lines, golden)
    for expected, actual in zip(golden, lines):
        assert numbers_match(expected, actual), (expected, actual)


if __name__ == "__main__":   # record golden values from a trainer revision: SRC_DIR PACKAGE OUT_JSON CASE...
    import tempfile

    src_dir, package, out_json, *names = sys.argv[1:]
    root = Path(tempfile.mkdtemp(dir=os.environ.get("TMPDIR")))
    subprocess.run([sys.executable, str(HERE / "cpu_trainer.py"), "fixture", str(root / "fixture")], check=True,
                   env={**os.environ, "PYTHONPATH": str(SRC)})
    recorded = {n: logged_updates(run_case(CASES[n], root / "fixture", root / n, src_dir, package)) for n in names}
    Path(out_json).write_text(json.dumps(recorded, indent=1))
