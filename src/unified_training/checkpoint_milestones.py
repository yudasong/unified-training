# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Weights-only checkpoint retention for longitudinal evaluation."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path


# A preserved subset is only useful when it contains an actual model payload.
# These patterns cover the formats written by the current callers:
#
# - DTensor and Megatron torch-distributed checkpoints: ``*.distcp``
# - Hugging Face and PEFT checkpoints: safetensors or PyTorch ``.bin`` files
# - legacy Megatron rank checkpoints: ``model[_optim]_rng.pt``
#
# Metadata-only files such as ``.metadata``, ``common.pt``, and weight-index JSON
# deliberately do not qualify: none can restore a model without a shard file.
_MODEL_WEIGHT_PATTERNS = (
    "**/*.distcp",
    "**/*.safetensors",
    "**/pytorch_model.bin",
    "**/pytorch_model-*.bin",
    "**/adapter_model.bin",
    "**/model_optim_rng.pt",
    "**/model_rng.pt",
)


def milestones_crossed(previous_step: int, current_step: int, interval: int) -> list[int]:
    """Return interval boundaries crossed by a monotonic optimizer-step update."""
    if previous_step < 0 or current_step < previous_step:
        raise ValueError(
            f"expected 0 <= previous_step <= current_step, got {previous_step}, {current_step}"
        )
    if interval < 0:
        raise ValueError(f"milestone interval must be non-negative, got {interval}")
    if interval == 0 or current_step == previous_step:
        return []
    first = (previous_step // interval + 1) * interval
    return list(range(first, current_step + 1, interval))


def milestone_dir_name(target_step: int, actual_step: int) -> str:
    if target_step <= 0 or actual_step < target_step:
        raise ValueError(
            f"expected 0 < target_step <= actual_step, got {target_step}, {actual_step}"
        )
    return f"target_{target_step:07d}-actual_{actual_step:07d}"


def preserve_checkpoint_subset(
    source: Path,
    destination: Path,
    *,
    include: Iterable[str],
    manifest: Mapping[str, object],
) -> Path:
    """Atomically hard-link selected checkpoint entries into a protected directory.

    Hard links keep storage overhead near zero while making the selected files
    independent of rolling-retention directory deletion. Cross-device paths fall
    back to ordinary copies. The selected subset must contain at least one non-empty
    model-weight file in a supported HF, DTensor, or Megatron layout; metadata alone
    is never published as a milestone.
    """
    source = source.resolve()
    destination = destination.resolve()
    payload = dict(manifest)
    payload["source"] = str(source)
    payload["format_version"] = 1

    manifest_path = destination / "milestone.json"
    if destination.exists():
        if manifest_path.is_file() and json.loads(manifest_path.read_text()) == payload:
            _require_model_weights(destination, description="existing milestone")
            return destination
        raise FileExistsError(f"milestone destination already exists with different metadata: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir()

    try:
        copied = 0
        for relative in include:
            src = source / relative
            if not src.exists():
                continue
            dst = stage / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(
                    src,
                    dst,
                    symlinks=True,
                    copy_function=_link_or_copy,
                    dirs_exist_ok=True,
                )
            else:
                _link_or_copy(src, dst)
            copied += 1
        if copied == 0:
            raise FileNotFoundError(f"none of the requested checkpoint entries exist under {source}")
        _require_model_weights(stage, description=f"checkpoint subset selected from {source}")
        (stage / "milestone.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.rename(stage, destination)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return destination


def _require_model_weights(root: Path, *, description: str) -> None:
    for pattern in _MODEL_WEIGHT_PATTERNS:
        for candidate in root.glob(pattern):
            if candidate.is_file() and candidate.stat().st_size > 0:
                return
    expected = ", ".join(_MODEL_WEIGHT_PATTERNS)
    raise FileNotFoundError(
        f"{description} contains no non-empty model weight files under {root}; "
        f"expected at least one file matching: {expected}"
    )


def _link_or_copy(source: str | Path, destination: str | Path) -> str:
    try:
        os.link(source, destination, follow_symlinks=False)
    except OSError as error:
        if error.errno != getattr(os, "EXDEV", 18):
            raise
        shutil.copy2(source, destination, follow_symlinks=False)
    return str(destination)
