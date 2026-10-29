# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import json
import os

import pytest

from unified_training.checkpoint_milestones import (
    milestone_dir_name,
    milestones_crossed,
    preserve_checkpoint_subset,
)


def test_milestones_crossed_supports_exact_and_nearest_boundaries():
    assert milestones_crossed(1984, 2016, 2000) == [2000]
    assert milestones_crossed(3968, 4000, 2000) == [4000]
    assert milestones_crossed(0, 4500, 2000) == [2000, 4000]
    assert milestones_crossed(0, 4500, 0) == []


@pytest.mark.parametrize("previous,current,interval", [(-1, 0, 1), (2, 1, 1), (0, 1, -1)])
def test_milestones_crossed_rejects_invalid_steps(previous, current, interval):
    with pytest.raises(ValueError):
        milestones_crossed(previous, current, interval)


def test_preserve_checkpoint_subset_hardlinks_only_selected_entries(tmp_path):
    source = tmp_path / "step_63"
    weights = source / "policy" / "weights"
    optimizer = source / "policy" / "optimizer"
    weights.mkdir(parents=True)
    optimizer.mkdir(parents=True)
    (weights / "rank0.distcp").write_bytes(b"weights")
    (optimizer / "rank0.distcp").write_bytes(b"optimizer")
    (source / "training_info.json").write_text("{}\n")

    destination = tmp_path / "eval" / milestone_dir_name(2000, 2016)
    manifest = {
        "target_optimizer_step": 2000,
        "actual_optimizer_step": 2016,
        "weights_only": True,
    }
    preserve_checkpoint_subset(
        source,
        destination,
        include=("policy/weights", "policy/tokenizer", "training_info.json"),
        manifest=manifest,
    )

    preserved = destination / "policy" / "weights" / "rank0.distcp"
    assert preserved.read_bytes() == b"weights"
    assert os.stat(preserved).st_ino == os.stat(weights / "rank0.distcp").st_ino
    assert not (destination / "policy" / "optimizer").exists()
    assert json.loads((destination / "milestone.json").read_text())["weights_only"] is True

    # Exact retries are idempotent; conflicting metadata is rejected.
    preserve_checkpoint_subset(
        source,
        destination,
        include=("policy/weights",),
        manifest=manifest,
    )
    with pytest.raises(FileExistsError):
        preserve_checkpoint_subset(
            source,
            destination,
            include=("policy/weights",),
            manifest={**manifest, "actual_optimizer_step": 2048},
        )


@pytest.mark.parametrize(
    "relative_weight",
    [
        "model.safetensors",
        "model-00001-of-00002.safetensors",
        "policy/weights/__0_0.distcp",
        "policy/weights/iter_0000000/mp_rank_00/model_optim_rng.pt",
    ],
)
def test_preserve_checkpoint_subset_accepts_hf_dtensor_and_megatron_layouts(
    tmp_path, relative_weight
):
    source = tmp_path / "source"
    weight = source / relative_weight
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"model weights")
    destination = tmp_path / "eval" / "target_0002000-actual_0002000"

    preserve_checkpoint_subset(
        source,
        destination,
        include=(".",),
        manifest={"target_optimizer_step": 2000, "actual_optimizer_step": 2000},
    )

    assert (destination / relative_weight).read_bytes() == b"model weights"


@pytest.mark.parametrize("weight_contents", [None, b""])
def test_preserve_checkpoint_subset_rejects_metadata_only_or_empty_weights(
    tmp_path, weight_contents
):
    source = tmp_path / "step_63"
    weights = source / "policy" / "weights"
    weights.mkdir(parents=True)
    (weights / ".metadata").write_bytes(b"checkpoint metadata")
    megatron_iteration = weights / "iter_0000000"
    megatron_iteration.mkdir()
    (megatron_iteration / "common.pt").write_bytes(b"common checkpoint state")
    if weight_contents is not None:
        (weights / "__0_0.distcp").write_bytes(weight_contents)
    (source / "training_info.json").write_text("{}\n")
    destination = tmp_path / "eval" / "target_0002000-actual_0002016"

    with pytest.raises(FileNotFoundError, match="no non-empty model weight files"):
        preserve_checkpoint_subset(
            source,
            destination,
            include=("policy/weights", "training_info.json"),
            manifest={"target_optimizer_step": 2000, "actual_optimizer_step": 2016},
        )

    assert not destination.exists()
    assert not list(destination.parent.glob(f".{destination.name}.tmp-*"))


def test_preserve_checkpoint_subset_rejects_include_that_omits_existing_weights(tmp_path):
    source = tmp_path / "step_63"
    weights = source / "policy" / "weights"
    weights.mkdir(parents=True)
    (weights / "__0_0.distcp").write_bytes(b"model weights")
    (source / "training_info.json").write_text("{}\n")
    destination = tmp_path / "eval" / "target_0002000-actual_0002016"

    with pytest.raises(FileNotFoundError, match="no non-empty model weight files"):
        preserve_checkpoint_subset(
            source,
            destination,
            include=("training_info.json",),
            manifest={"target_optimizer_step": 2000, "actual_optimizer_step": 2016},
        )

    assert not destination.exists()
    assert not list(destination.parent.glob(f".{destination.name}.tmp-*"))


def test_preserve_checkpoint_subset_revalidates_idempotent_destination(tmp_path):
    source = tmp_path / "step_63"
    source.mkdir()
    (source / "training_info.json").write_text("{}\n")
    destination = tmp_path / "eval" / "target_0002000-actual_0002016"
    destination.mkdir(parents=True)
    manifest = {"target_optimizer_step": 2000, "actual_optimizer_step": 2016}
    payload = {**manifest, "source": str(source.resolve()), "format_version": 1}
    (destination / "milestone.json").write_text(json.dumps(payload))

    with pytest.raises(FileNotFoundError, match="existing milestone"):
        preserve_checkpoint_subset(
            source,
            destination,
            include=("training_info.json",),
            manifest=manifest,
        )
