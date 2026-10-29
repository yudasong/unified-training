# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Held-out slice of the midtraining anchor for UNIFIED_MIDTRAIN_VAL_DIR (val/midtrain_lm_loss).

The trainer samples its anchor through ``WeightedEpisodeDataset``; in the first epoch
that is source episode ``permutation[anchor_draw_index]`` with
``default_rng((UNIFIED_SPLIT_SEED + 1, 0))``. ``prepare`` copies windows from an unused
tail of that permutation into a small immutable round; the manifest records the
training-row bound that proves these positions cannot be consumed by the run.
``row-fraction-bound`` gives the largest UNIFIED_MAX_ANCHOR_ROW_FRACTION that keeps a
holdout tail of a given size outside one anchor pass.

    python -m unified_training.tools.build_midtrain_holdout prepare --anchor-dir ANCHOR \
        --output-dir HOLDOUT --expected-manifest-sha256 SHA --expected-anchor-windows N \
        --expected-window 1025 --max-training-rows ROWS --max-anchor-row-fraction F \
        [--permutation-seed 1235 --eval-windows 512]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from unified_training.episodes import (
    INDEX_FILE,
    MANIFEST_FILE,
    TOKENS_FILE,
    WEIGHTS_FILE,
    Round,
    RoundWriter,
    load_round,
)


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()



def select_holdout_indices(
    *,
    source_windows: int,
    max_training_rows: int,
    max_anchor_row_fraction: float,
    permutation_seed: int,
    eval_windows: int,
) -> tuple[np.ndarray, int]:
    """Return source indices after the conservative first-epoch training prefix."""
    if source_windows <= 0 or max_training_rows < 0 or eval_windows <= 0:
        raise ValueError("source_windows/eval_windows must be positive and max_training_rows non-negative")
    if not 0.0 <= max_anchor_row_fraction <= 1.0:
        raise ValueError("max_anchor_row_fraction must be in [0, 1]")
    if permutation_seed < 0:
        raise ValueError("permutation_seed must be non-negative")

    # ceil is intentionally conservative. MixedEpisodeDataset consumes
    # floor(training_rows * actual_row_fraction), and the actual fraction is
    # required to be no larger than this bound.
    max_train_anchor_draws = math.ceil(max_training_rows * max_anchor_row_fraction)
    if max_train_anchor_draws + eval_windows > source_windows:
        raise ValueError(
            f"holdout does not fit in first permutation epoch: training bound "
            f"{max_train_anchor_draws} + eval {eval_windows} > {source_windows}"
        )
    permutation = np.random.default_rng((permutation_seed, 0)).permutation(source_windows)
    selected = np.asarray(
        permutation[max_train_anchor_draws : max_train_anchor_draws + eval_windows],
        dtype=np.int64,
    )
    return selected, max_train_anchor_draws


def derive_anchor_row_fraction_bound(
    *,
    source_windows: int,
    max_training_rows: int,
    holdout_windows: int,
) -> tuple[float, int]:
    """Return the largest safe row-fraction ceiling with a held-out tail.

    The trainer may consume at most ``ceil(max_training_rows * bound)`` anchor
    rows under the conservative holdout proof. The returned bound reserves
    ``holdout_windows`` distinct rows at the end of the same first-epoch
    permutation and therefore also stays within one anchor pass.
    """
    if source_windows <= 0 or max_training_rows <= 0 or holdout_windows <= 0:
        raise ValueError(
            "source_windows, max_training_rows, and holdout_windows must be positive"
        )
    available_training_windows = source_windows - holdout_windows
    if available_training_windows <= 0:
        raise ValueError(
            "anchor has no training capacity after reserving the holdout: "
            f"source={source_windows}, holdout={holdout_windows}"
        )
    max_train_anchor_draws = min(
        available_training_windows,
        max_training_rows,
    )
    bound = max_train_anchor_draws / max_training_rows
    # Guard against a platform rounding the product just above the integer
    # numerator and making ceil reserve one extra row.
    while math.ceil(max_training_rows * bound) > max_train_anchor_draws:
        bound = float(np.nextafter(bound, 0.0))
    if not 0.0 < bound <= 1.0:
        raise AssertionError(f"invalid derived anchor row-fraction bound: {bound}")
    proven_draws = math.ceil(max_training_rows * bound)
    if proven_draws + holdout_windows > source_windows:
        raise AssertionError(
            "derived anchor row-fraction bound does not preserve the holdout"
        )
    return bound, proven_draws


def _holdout_file_hashes(directory: Path) -> dict[str, str]:
    return {
        "tokens_sha256": sha256_file(directory / TOKENS_FILE),
        "weights_sha256": sha256_file(directory / WEIGHTS_FILE),
        "index_sha256": sha256_file(directory / INDEX_FILE),
    }


def validate_holdout(directory: Path) -> tuple[Round, dict[str, Any], str]:
    round_data = load_round(directory)
    manifest = round_data.manifest
    if manifest.get("kind") != "midtrain_schedule_holdout":
        raise ValueError(f"{directory} is not a midtrain_schedule_holdout")
    expected_windows = int(manifest.get("eval_windows", -1))
    if expected_windows <= 0 or expected_windows != len(round_data.index):
        raise ValueError(
            f"holdout window count mismatch: manifest={expected_windows}, index={len(round_data.index)}"
        )
    for field, actual in _holdout_file_hashes(directory).items():
        expected = manifest.get(field)
        if expected != actual:
            raise ValueError(f"holdout {field} mismatch: expected={expected!r}, actual={actual!r}")
    return round_data, manifest, sha256_file(directory / MANIFEST_FILE)


def prepare_holdout(
    *,
    anchor_dir: Path,
    output_dir: Path,
    expected_manifest_sha256: str,
    expected_anchor_windows: int,
    expected_window: int,
    max_training_rows: int,
    max_anchor_row_fraction: float,
    permutation_seed: int,
    eval_windows: int,
) -> dict[str, Any]:
    anchor_dir = anchor_dir.resolve()
    output_dir = output_dir.resolve()
    if anchor_dir == output_dir:
        raise ValueError("output_dir must differ from anchor_dir")
    actual_manifest_sha256 = sha256_file(anchor_dir / MANIFEST_FILE)
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            "anchor manifest SHA-256 mismatch: "
            f"expected={expected_manifest_sha256}, actual={actual_manifest_sha256}"
        )
    source = load_round(anchor_dir)
    if len(source.index) != expected_anchor_windows:
        raise ValueError(
            f"anchor window count mismatch: expected={expected_anchor_windows}, actual={len(source.index)}"
        )
    if int(source.manifest.get("n_episodes", -1)) != expected_anchor_windows:
        raise ValueError("anchor manifest n_episodes does not match the pinned window count")
    if int(source.manifest.get("window", -1)) != expected_window:
        raise ValueError(
            f"anchor window mismatch: expected={expected_window}, manifest={source.manifest.get('window')}"
        )

    selected, max_train_anchor_draws = select_holdout_indices(
        source_windows=expected_anchor_windows,
        max_training_rows=max_training_rows,
        max_anchor_row_fraction=max_anchor_row_fraction,
        permutation_seed=permutation_seed,
        eval_windows=eval_windows,
    )
    expected_provenance = {
        "source_anchor": str(anchor_dir),
        "source_manifest_sha256": actual_manifest_sha256,
        "source_n_episodes": expected_anchor_windows,
        "source_window": expected_window,
        "max_training_rows": max_training_rows,
        "max_anchor_row_fraction": max_anchor_row_fraction,
        "max_train_anchor_draws": max_train_anchor_draws,
        "permutation_seed": permutation_seed,
        "permutation_epoch": 0,
        "permutation_tail_start": max_train_anchor_draws,
        "eval_windows": eval_windows,
        "source_episode_indices_sha256": hashlib.sha256(selected.astype("<i8").tobytes()).hexdigest(),
    }

    if output_dir.exists():
        _, existing, _ = validate_holdout(output_dir)
        mismatches = {
            key: (existing.get(key), value)
            for key, value in expected_provenance.items()
            if existing.get(key) != value
        }
        if mismatches:
            raise FileExistsError(f"existing holdout has different provenance: {mismatches}")
        return existing

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        writer = RoundWriter(staged)
        for source_index in selected:
            tokens, weights, row = source.episode(int(source_index))
            if not 0 < len(tokens) <= expected_window:
                raise ValueError(
                    f"source episode {int(source_index)} has {len(tokens)} tokens, "
                    f"expected 1..{expected_window}"
                )
            writer.add(
                tokens,
                weights,
                prompt_len=int(row["prompt_len"]),
                reward=float(row["reward"]),
                weight=float(row["weight"]),
                policy_version=int(row["policy_version"]),
                group_id=int(row["group_id"]),
            )
        manifest = writer.finalize(
            {
                "format_version": 1,
                "kind": "midtrain_schedule_holdout",
                "format": "raw",
                "split_method": "unused_epoch0_permutation_tail",
                "disjointness_scope": "active_training_schedule_indices",
                **expected_provenance,
                "source_episode_indices": [int(index) for index in selected],
                "numpy_version": np.__version__,
            }
        )
        manifest.update(_holdout_file_hashes(staged))
        (staged / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(staged, output_dir)
    except BaseException:
        shutil.rmtree(staged, ignore_errors=True)
        raise
    validate_holdout(output_dir)
    return manifest


def _prepare_command(args: argparse.Namespace) -> int:
    manifest = prepare_holdout(
        anchor_dir=args.anchor_dir,
        output_dir=args.output_dir,
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_anchor_windows=args.expected_anchor_windows,
        expected_window=args.expected_window,
        max_training_rows=args.max_training_rows,
        max_anchor_row_fraction=args.max_anchor_row_fraction,
        permutation_seed=args.permutation_seed,
        eval_windows=args.eval_windows,
    )
    print(
        f"[midtrain-holdout] ready: {args.output_dir} windows={manifest['eval_windows']} "
        f"tail_start={manifest['permutation_tail_start']}"
    )
    return 0


def _row_fraction_bound_command(args: argparse.Namespace) -> int:
    bound, max_train_anchor_draws = derive_anchor_row_fraction_bound(
        source_windows=args.source_windows,
        max_training_rows=args.max_training_rows,
        holdout_windows=args.holdout_windows,
    )
    if args.json:
        print(
            json.dumps(
                {
                    "max_anchor_row_fraction": bound,
                    "max_train_anchor_draws": max_train_anchor_draws,
                    "max_training_rows": args.max_training_rows,
                    "source_windows": args.source_windows,
                    "holdout_windows": args.holdout_windows,
                },
                sort_keys=True,
            )
        )
    else:
        print(format(bound, ".17g"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="materialize a proven-unused anchor permutation tail")
    prepare.add_argument("--anchor-dir", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--expected-manifest-sha256", required=True)
    prepare.add_argument("--expected-anchor-windows", type=int, required=True)
    prepare.add_argument("--expected-window", type=int, default=1_025)
    prepare.add_argument("--max-training-rows", type=int, required=True)
    prepare.add_argument("--max-anchor-row-fraction", type=float, required=True)
    prepare.add_argument("--permutation-seed", type=int, default=1_235)
    prepare.add_argument("--eval-windows", type=int, default=512)
    prepare.set_defaults(handler=_prepare_command)

    bound = subparsers.add_parser(
        "row-fraction-bound",
        help="derive a one-pass anchor row ceiling while reserving a holdout tail",
    )
    bound.add_argument("--source-windows", type=int, required=True)
    bound.add_argument("--max-training-rows", type=int, required=True)
    bound.add_argument("--holdout-windows", type=int, required=True)
    bound.add_argument("--json", action="store_true")
    bound.set_defaults(handler=_row_fraction_bound_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
