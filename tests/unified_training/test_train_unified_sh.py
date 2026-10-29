# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""scripts/train_unified.sh with a fake interpreter standing in for the generator and torchrun."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "train_unified.sh"
FAKE_PYTHON = """#!/bin/bash
if [[ "$1 $2" == "-m unified_training.gen_service" ]]; then
  echo "gen rank=$UNIFIED_GEN_RANK world=$UNIFIED_GEN_WORLD tag=$UNIFIED_GEN_TAG gpu=$CUDA_VISIBLE_DEVICES"
  for _ in $(seq 300); do [[ -e "$UNIFIED_RUN_DIR/STOP_GENERATOR" ]] && exit 0; sleep 0.1; done
  exit 1
fi
if [[ "$1 $2" == "-m torch.distributed.run" ]]; then
  echo "trainer gpu=$CUDA_VISIBLE_DEVICES lr=$UNIFIED_LR iters=$UNIFIED_TRAIN_ITERS args=$*"
  sleep 0.5
  exit "${FAKE_TRAINER_RC:-0}"
fi
exit 9
"""

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


@pytest.fixture
def setup(tmp_path):
    fake = tmp_path / "fake_python"
    fake.write_text(FAKE_PYTHON)
    fake.chmod(0o755)
    init = tmp_path / "ckpt" / "init"
    init.mkdir(parents=True)
    (init / "config.json").write_text("{}")
    (init / "model.safetensors").write_bytes(b"x")
    data = tmp_path / "data"
    data.mkdir()
    (data / "prompts.json").write_text('[{"question": "q", "gold": "1"}]')
    cfg = tmp_path / "run.env"
    cfg.write_text("# a config\nUNIFIED_POLICY_HF=${CKPT_DIR}/init\nUNIFIED_PROMPTS_JSON=${DATA_DIR}/prompts.json\n"
                   "UNIFIED_LR=1e-6\nUNIFIED_TRAIN_ITERS=100\nUNIFIED_GBS=16\nUNIFIED_MBS=2\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith("UNIFIED_")}
    env.update(PYTHON=str(fake), DATA_DIR=str(data), CKPT_DIR=str(tmp_path / "ckpt"), CUDA_VISIBLE_DEVICES="0,1,2,3",
               NODE_TMP=str(tmp_path / "node"), GEN_STOP_GRACE="30")
    return tmp_path, cfg, env


def launch(env, *args, run_dir):
    return subprocess.run(["bash", str(SCRIPT), *args], env={**env, "RUN_DIR": str(run_dir)}, capture_output=True,
                          text=True, timeout=120)


def test_single_node_launch(setup):
    tmp, cfg, env = setup
    run = tmp / "run"
    proc = launch(env, str(cfg), "UNIFIED_TRAIN_ITERS=5", run_dir=run)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    receipt = (run / "launch.env").read_text()
    for line in ("UNIFIED_LR=1e-6", "UNIFIED_TRAIN_ITERS=5", f"UNIFIED_RUN_DIR={run}", "UNIFIED_GEN_WORLD=2",
                 "UNIFIED_GENERATOR_GPU_IDS=0,1", "UNIFIED_TRAINER_GPU_IDS=2,3", f"UNIFIED_EPISODES_DIRS={run}/rounds"):
        assert line in receipt.splitlines(), line
    seed = run / "policies" / "v0000000"
    assert (seed / "DONE").exists() and (seed / "config.json").resolve() == (tmp / "ckpt/init/config.json").resolve()
    assert (run / "STOP_GENERATOR").read_text().strip() == "0"
    assert "gen rank=0 world=2 tag=r00 gpu=0" in (run / "logs/generator-rank0.log").read_text()
    assert "gen rank=1 world=2 tag=r01 gpu=1" in (run / "logs/generator-rank1.log").read_text()
    trainer = (run / "logs/trainer0.log").read_text()
    assert "trainer gpu=2,3 lr=1e-6 iters=5" in trainer and "--nproc-per-node=2" in trainer
    assert "-m unified_training.hf_trainer" in trainer


def test_colocated_layout_trainer_exit_code_and_clobber_guard(setup):
    tmp, cfg, env = setup
    run = tmp / "run"
    env = {**env, "UNIFIED_GENERATOR_GPU_IDS": "0,1,2,3", "UNIFIED_TRAINER_GPU_IDS": "0,1,2,3", "FAKE_TRAINER_RC": "3"}
    proc = launch(env, str(cfg), run_dir=run)
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "co-located" in proc.stdout
    assert (run / "STOP_GENERATOR").read_text().strip() == "3"
    assert "trainer gpu=0,1,2,3" in (run / "logs/trainer0.log").read_text()
    (run / "rounds" / "gr00-b0000000-q000000000").mkdir()
    proc = launch(env, str(cfg), run_dir=run)
    assert proc.returncode == 2 and "RESUME=1" in proc.stderr
    proc = launch({**env, "RESUME": "1", "FAKE_TRAINER_RC": "0"}, str(cfg), run_dir=run)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_rejects_bad_arguments_and_batch_shape(setup):
    tmp, cfg, env = setup
    assert launch(env, "not-a-file-or-assignment", run_dir=tmp / "a").returncode == 2
    proc = launch(env, str(cfg), "UNIFIED_GBS=10", run_dir=tmp / "b")
    assert proc.returncode == 2 and "not divisible" in proc.stderr


FAKE_REMOTE_SHELL = """#!/bin/bash
host=$1
shift
echo "on $host: $*"
exec bash -c "$*"
"""


def test_hostfile_launch_starts_generator_and_trainer_nodes(setup):
    tmp, cfg, env = setup
    remote = tmp / "fake_remote"
    remote.write_text(FAKE_REMOTE_SHELL)
    remote.chmod(0o755)
    hosts = tmp / "hosts.txt"
    hosts.write_text("# generator node, then trainer nodes\ngenhost0\ntrainhost0 slots=4\ntrainhost1\n")
    env = {**env, "HOSTFILE": str(hosts), "REMOTE_SHELL": str(remote), "UNIFIED_GENERATOR_NNODES": "1",
           "UNIFIED_TRAINER_NNODES": "2", "CUDA_VISIBLE_DEVICES": "0,1"}
    run = tmp / "run"
    proc = launch(env, str(cfg), "UNIFIED_STREAMS_JSON=[{\"dirs\":[\"a b\"],\"token_fraction\":0.1}]", run_dir=run)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    resolved = (run / "launch-resolved.env").read_text()
    assert "UNIFIED_TRAINER_NNODES=2" in resolved and "UNIFIED_LR=1e-6" in resolved
    assert "gen rank=0 world=2 tag=r00 gpu=0" in (run / "logs/generator-rank0.log").read_text()
    assert "gen rank=1 world=2 tag=r01 gpu=1" in (run / "logs/generator-rank1.log").read_text()
    for j in (0, 1):
        trainer = (run / f"logs/trainer{j}.log").read_text()
        assert f"--node-rank={j}" in trainer and "--master-addr=trainhost0" in trainer and "--nnodes=2" in trainer
    assert (run / "STOP_GENERATOR").read_text().strip() == "0"
    assert "on genhost0:" in (run / "logs/node-gen0-genhost0.log").read_text()
    assert "on trainhost1:" in (run / "logs/node-train1-trainhost1.log").read_text()
    # the JSON value survives the round trip through the resolved env file
    import shlex
    line = next(l for l in resolved.splitlines() if l.startswith("UNIFIED_STREAMS_JSON="))
    assert shlex.split(line.split("=", 1)[1])[0] == '[{"dirs":["a b"],"token_fraction":0.1}]'
