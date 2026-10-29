# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Crash-safe publication helpers for unified HF trainer checkpoints."""

from __future__ import annotations

import os
import shutil
import threading
from collections.abc import Callable
from pathlib import Path


class BackgroundCheckpointWriter:
    """Serialize background writes and surface worker failures to the trainer."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._description = ""

    def wait(self) -> None:
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._error is not None:
            raise RuntimeError(
                f"background checkpoint write failed ({self._description}): {self._error}"
            ) from self._error

    def start(self, write: Callable[[], None], *, description: str) -> None:
        self.wait()
        self._description = description

        def run() -> None:
            try:
                write()
            except BaseException as exc:
                self._error = exc

        self._thread = threading.Thread(target=run, name=description, daemon=True)
        self._thread.start()


def _fsync_dir(path: Path) -> None:
    """Best-effort directory fsync for small checkpoint commit records."""
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        # Some distributed filesystems reject directory fsync. Atomic rename is
        # still the important property; do not make an otherwise valid save fail.
        pass


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically replace a short text file on the same filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}"
    with temporary.open("w") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_dir(path.parent)


def validate_resume_pair(
    policies: Path,
    state_dir: Path,
    latest_file: Path,
) -> int:
    """Return the latest committed step and repair its generator marker.

    ``latest_step.txt`` is the transaction commit record. Once it names a step,
    both the policy directory and optimizer file must exist. ``DONE`` is
    intentionally published after that commit; if a process dies in the tiny
    interval between the two operations, a restart safely recreates it.
    """
    if not latest_file.exists():
        return 0
    try:
        step = int(latest_file.read_text().strip())
    except ValueError as exc:
        raise RuntimeError(f"invalid checkpoint commit record: {latest_file}") from exc
    if step <= 0:
        raise RuntimeError(f"checkpoint commit step must be positive, got {step}")
    policy = policies / f"v{step:07d}"
    optimizer = state_dir / f"opt_{step:07d}.pt"
    missing = [str(path) for path in (policy, optimizer) if not path.exists()]
    if missing:
        raise RuntimeError(
            f"checkpoint commit {latest_file} names an incomplete pair; missing {missing}"
        )
    done = policy / "DONE"
    if not done.exists():
        done.touch()
        _fsync_dir(policy)
    return step


def commit_checkpoint_pair(
    *,
    staged_policy: Path,
    staged_optimizer: Path,
    final_policy: Path,
    final_optimizer: Path,
    latest_file: Path,
    step: int,
) -> None:
    """Publish a staged HF policy and optimizer as one resumable transaction.

    The commit record is replaced only after both final paths exist. The policy
    ``DONE`` marker is last, so generators never observe a checkpoint that the
    trainer cannot resume. Unpublished leftovers from a killed earlier attempt
    at the same step are safe to replace.
    """
    if step <= 0:
        raise ValueError(f"checkpoint step must be positive, got {step}")
    if not staged_policy.is_dir() or not staged_optimizer.is_file():
        raise RuntimeError(
            f"checkpoint staging is incomplete: {staged_policy}, {staged_optimizer}"
        )
    if (staged_policy / "DONE").exists():
        raise RuntimeError("staged policy must not contain DONE before transaction commit")

    final_policy.parent.mkdir(parents=True, exist_ok=True)
    final_optimizer.parent.mkdir(parents=True, exist_ok=True)
    if final_policy.exists():
        if (final_policy / "DONE").exists():
            raise FileExistsError(f"refusing to replace published policy {final_policy}")
        shutil.rmtree(final_policy)

    os.replace(staged_policy, final_policy)
    os.replace(staged_optimizer, final_optimizer)
    _fsync_dir(final_policy.parent)
    _fsync_dir(final_optimizer.parent)

    # This is the durable policy+optimizer commit point used for trainer resume.
    atomic_write_text(latest_file, str(step))

    # Generator visibility is deliberately after trainer resumability.
    (final_policy / "DONE").touch()
    _fsync_dir(final_policy)


def publish_policy_snapshot(*, staged_policy: Path, final_policy: Path) -> None:
    """Commit a weights-only policy snapshot for the generators: rename, then DONE.

    Unlike :func:`commit_checkpoint_pair` this writes no optimizer state and does
    not touch the commit record, so the snapshot is NOT a resume point.  A
    resumed trainer removes such snapshots newer than its resume step
    (:func:`prune_unpaired_policies`) before it republishes them.
    """
    if not staged_policy.is_dir():
        raise RuntimeError(f"policy staging is incomplete: {staged_policy}")
    if (staged_policy / "DONE").exists():
        raise RuntimeError("staged policy must not contain DONE before publish")
    final_policy.parent.mkdir(parents=True, exist_ok=True)
    if final_policy.exists():
        if (final_policy / "DONE").exists():
            raise FileExistsError(f"refusing to replace published policy {final_policy}")
        shutil.rmtree(final_policy)
    os.replace(staged_policy, final_policy)
    _fsync_dir(final_policy.parent)
    (final_policy / "DONE").touch()
    _fsync_dir(final_policy)


def prune_unpaired_policies(policies: Path, state_dir: Path, latest_step: int) -> list[str]:
    """Remove weights-only snapshots newer than the last resumable checkpoint.

    After a resume the trainer re-runs those steps and publishes them again; a
    leftover future snapshot would be picked by the generators (newest DONE
    wins) and would collide with the republish.  Paired checkpoints (an
    ``opt_<step>.pt`` exists) are never touched.
    """
    removed: list[str] = []
    if not policies.is_dir():
        return removed
    for d in sorted(policies.glob("v*")):
        digits = d.name[1:]
        if not digits.isdigit():
            continue
        step = int(digits)
        if step > latest_step and not (state_dir / f"opt_{step:07d}.pt").exists():
            shutil.rmtree(d)
            removed.append(d.name)
    return removed
