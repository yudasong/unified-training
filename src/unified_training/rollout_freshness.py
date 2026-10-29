# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Fail-closed progress tracking for asynchronous unified rollout streams."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


class RolloutFreshnessError(RuntimeError):
    """Committed rollout metadata is malformed or incompatible with this run."""


@dataclass(frozen=True)
class RolloutFreshnessSnapshot:
    """Rank-balanced view of all visible, DONE-gated generator rounds."""

    expected_world: int
    round_counts: tuple[int, ...]
    latest_policy_versions: tuple[int | None, ...]
    total_rounds: int

    @property
    def missing_ranks(self) -> tuple[int, ...]:
        return tuple(
            rank
            for rank, count in enumerate(self.round_counts)
            if count == 0
        )

    @property
    def complete_cycles(self) -> int:
        return min(self.round_counts, default=0)

    @property
    def complete_policy_version(self) -> int | None:
        if any(version is None for version in self.latest_policy_versions):
            return None
        return min(
            int(version)
            for version in self.latest_policy_versions
            if version is not None
        )

    def summary(self) -> str:
        policies = ",".join(
            "none" if version is None else str(version)
            for version in self.latest_policy_versions
        )
        counts = ",".join(str(count) for count in self.round_counts)
        return (
            f"world={self.expected_world} counts=[{counts}] "
            f"latest_policy=[{policies}] complete_cycles={self.complete_cycles}"
        )


@dataclass(frozen=True)
class RolloutFreshnessObservation:
    snapshot: RolloutFreshnessSnapshot
    stale_steps: int
    policy_lag_steps: int | None
    blocked_reason: str | None


@dataclass(frozen=True)
class _RoundRecord:
    generator_rank: int
    policy_version: int


class RolloutFreshnessGuard:
    """Track complete producer cycles and behavior-policy lag.

    A complete cycle advances only when every configured generator rank has
    committed another visible round. This prevents a healthy rank from hiding
    a stalled peer. The guard caches validated manifests; rescans only parse new
    round directories.
    """

    def __init__(
        self,
        parent_dirs: list[str | Path],
        *,
        expected_world: int,
        max_stale_steps: int,
        max_policy_lag_steps: int,
        initial_step: int,
    ) -> None:
        if expected_world <= 0:
            raise ValueError(
                f"expected_world must be positive, got {expected_world}"
            )
        if max_stale_steps < 0:
            raise ValueError(
                f"max_stale_steps must be non-negative, got {max_stale_steps}"
            )
        if max_policy_lag_steps < 0:
            raise ValueError(
                "max_policy_lag_steps must be non-negative, got "
                f"{max_policy_lag_steps}"
            )
        if max_stale_steps == 0 and max_policy_lag_steps == 0:
            raise ValueError("at least one rollout freshness limit must be enabled")
        if initial_step < 0:
            raise ValueError(f"initial_step must be non-negative, got {initial_step}")

        self._parents = tuple(Path(parent).resolve() for parent in parent_dirs)
        if not self._parents:
            raise ValueError("rollout freshness guard needs at least one parent dir")
        self._expected_world = expected_world
        self._max_stale_steps = max_stale_steps
        self._max_policy_lag_steps = max_policy_lag_steps
        self._records: dict[Path, _RoundRecord] = {}
        self._progress_baseline_counts: tuple[int, ...] | None = None
        self._last_progress_step = initial_step

    def _visible_round_dirs(self) -> list[Path]:
        round_dirs: list[Path] = []
        for parent in self._parents:
            if parent.name.startswith("."):
                continue
            if (parent / "DONE").is_file():
                round_dirs.append(parent)
                continue
            try:
                children = parent.iterdir()
            except FileNotFoundError:
                continue
            round_dirs.extend(
                child
                for child in children
                if child.is_dir()
                and not child.name.startswith(".")
                and (child / "DONE").is_file()
            )
        return sorted(set(round_dirs))

    def _read_record(self, round_dir: Path) -> _RoundRecord:
        manifest_path = round_dir / "manifest.json"
        try:
            with manifest_path.open() as stream:
                manifest = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise RolloutFreshnessError(
                f"cannot read finalized rollout manifest {manifest_path}: {exc}"
            ) from exc
        try:
            generator_rank = int(manifest["generator_rank"])
            generator_world = int(manifest["generator_world"])
            policy_version = int(manifest["policy_version"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RolloutFreshnessError(
                f"invalid generator metadata in {manifest_path}"
            ) from exc
        if generator_world != self._expected_world:
            raise RolloutFreshnessError(
                f"{manifest_path} has generator_world={generator_world}, "
                f"expected {self._expected_world}"
            )
        if not 0 <= generator_rank < generator_world:
            raise RolloutFreshnessError(
                f"{manifest_path} has invalid generator_rank={generator_rank} "
                f"for world={generator_world}"
            )
        if policy_version < 0:
            raise RolloutFreshnessError(
                f"{manifest_path} has negative policy_version={policy_version}"
            )
        return _RoundRecord(
            generator_rank=generator_rank,
            policy_version=policy_version,
        )

    def snapshot(self) -> RolloutFreshnessSnapshot:
        visible = self._visible_round_dirs()
        visible_set = set(visible)
        vanished = set(self._records) - visible_set
        if vanished:
            example = min(vanished)
            raise RolloutFreshnessError(
                "a previously committed rollout disappeared while training: "
                f"{example}"
            )
        for round_dir in visible:
            if round_dir not in self._records:
                self._records[round_dir] = self._read_record(round_dir)

        counts = [0] * self._expected_world
        latest: list[int | None] = [None] * self._expected_world
        for record in self._records.values():
            rank = record.generator_rank
            counts[rank] += 1
            previous = latest[rank]
            if previous is None or record.policy_version > previous:
                latest[rank] = record.policy_version
        return RolloutFreshnessSnapshot(
            expected_world=self._expected_world,
            round_counts=tuple(counts),
            latest_policy_versions=tuple(latest),
            total_rounds=len(self._records),
        )

    def observe(self, trainer_step: int) -> RolloutFreshnessObservation:
        if trainer_step < 0:
            raise ValueError(f"trainer_step must be non-negative, got {trainer_step}")
        snapshot = self.snapshot()

        if snapshot.missing_ranks:
            missing = ",".join(str(rank) for rank in snapshot.missing_ranks)
            return RolloutFreshnessObservation(
                snapshot=snapshot,
                stale_steps=trainer_step - self._last_progress_step,
                policy_lag_steps=None,
                blocked_reason=f"missing committed rollout rank(s): {missing}",
            )

        if self._progress_baseline_counts is None:
            self._progress_baseline_counts = snapshot.round_counts
            self._last_progress_step = trainer_step
        else:
            deltas = tuple(
                current - baseline
                for current, baseline in zip(
                    snapshot.round_counts,
                    self._progress_baseline_counts,
                )
            )
            if any(delta < 0 for delta in deltas):
                raise RolloutFreshnessError(
                    "committed rollout rank count moved backwards relative to "
                    f"baseline {self._progress_baseline_counts}: "
                    f"{snapshot.round_counts}"
                )
            # Reset only when every rank has advanced since the previous reset.
            # Updating every baseline to "now" deliberately discards a fast
            # rank's surplus credit: that rank must publish again before a later
            # slow-rank round can count as another healthy complete cycle.
            if min(deltas) > 0:
                self._progress_baseline_counts = snapshot.round_counts
                self._last_progress_step = trainer_step

        stale_steps = trainer_step - self._last_progress_step
        complete_policy = snapshot.complete_policy_version
        policy_lag = (
            None
            if complete_policy is None
            else max(trainer_step - complete_policy, 0)
        )
        blocked_reasons: list[str] = []
        if (
            self._max_stale_steps > 0
            and stale_steps >= self._max_stale_steps
        ):
            blocked_reasons.append(
                "no complete rank-balanced rollout cycle for "
                f"{stale_steps} trainer steps "
                f"(limit {self._max_stale_steps})"
            )
        if (
            self._max_policy_lag_steps > 0
            and policy_lag is not None
            and policy_lag > self._max_policy_lag_steps
        ):
            blocked_reasons.append(
                f"behavior-policy lag is {policy_lag} trainer steps "
                f"(limit {self._max_policy_lag_steps})"
            )
        return RolloutFreshnessObservation(
            snapshot=snapshot,
            stale_steps=stale_steps,
            policy_lag_steps=policy_lag,
            blocked_reason="; ".join(blocked_reasons) or None,
        )
