"""Collective, opt-in freshness admission before any model forward pass."""
from __future__ import annotations

import time
from collections.abc import Callable

import torch
import torch.distributed as dist

from unified_training.dataset import NoFreshEpisodes


def _global_max(value: int, device) -> int:
    flag = torch.tensor(value, dtype=torch.int32, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return int(flag.item())


def _transient_load_error(exc: BaseException) -> bool:
    """A generator round whose files are not yet fully visible on this node.

    Strict mode rescans the pool before every update, so a trainer rank can
    pick up a round whose index has landed on shared storage while its ``tokens.bin``
    is still shorter than the length the index declares (numpy: "mmap length
    is greater than file size"), or whose payload files are not visible yet at
    all.  The next poll sees the complete files, so this is a wait, not an
    error.
    """
    if isinstance(exc, FileNotFoundError):
        return True
    return isinstance(exc, ValueError) and "mmap length is greater than file size" in str(exc)


def _batch_status(micro_rows, step: int, max_age: int) -> int:
    """0 admissible; 1 stale; unknown/future provenance is an error."""
    stale = False
    for rows in micro_rows:
        for row in rows:
            if not bool(row.get("sampling_filter", True)):
                continue
            version = int(row.get("source_policy_version", -1))
            if version < 0 or version > step:
                raise ValueError(f"invalid consumed RL policy version {version} at trainer step {step}")
            stale |= step - version > max_age
    return int(stale)


def fetch_fresh_batch(
    stream, fetch: Callable, *, step: int, max_age: int, device,
    timeout_seconds: float, poll_seconds: float,
    stop_requested: Callable[[], bool] = lambda: False,
    service_pause: Callable[[], None] = lambda: None,
    on_wait: Callable[[], None] = lambda: None,
):
    """Return (micro_rows, blocked_seconds), or (None, blocked_seconds) on stop.

    All DP ranks must call together, with the same collective callbacks. Retries
    keep the optimizer step and caller's sample indices fixed. No stale partial
    batch is returned. A rank-local load/metadata error is raised on every rank
    before any can enter a model forward or backward collective.
    """
    if max_age <= 0 or timeout_seconds <= 0 or poll_seconds <= 0:
        raise ValueError("strict age, timeout and polling interval must be positive")
    deadline = time.monotonic() + timeout_seconds
    blocked_at = None
    while True:
        service_pause()
        if stop_requested():
            return None, 0.0 if blocked_at is None else time.monotonic() - blocked_at
        status, error, rows = 0, None, None
        try:
            stream.set_trainer_step(step)
            stream.rescan()
            if not stream._eligible_refs:
                raise NoFreshEpisodes("primary stream has no complete fresh generator pool")
            rows = fetch()
            status = _batch_status(rows, step, max_age)
        except NoFreshEpisodes:
            status = 1
        except Exception as exc:
            if _transient_load_error(exc):
                status = 1      # round still landing on the filesystem: wait and retry
            else:
                status, error = 2, exc
        status = _global_max(status, device)
        if status == 2:
            raise RuntimeError(f"strict freshness batch failed at step {step}: "
                               f"{error or 'another trainer rank failed'}") from error
        if status == 0:
            return rows, 0.0 if blocked_at is None else time.monotonic() - blocked_at
        if blocked_at is None:
            blocked_at = time.monotonic()
            callback_error = None
            try:
                on_wait()
            except Exception as exc:
                callback_error = exc
            if _global_max(int(callback_error is not None), device):
                raise RuntimeError("strict freshness wait callback failed") from callback_error
        if _global_max(int(time.monotonic() >= deadline), device):
            raise TimeoutError(f"strict freshness: no admissible global batch within "
                               f"{timeout_seconds:g}s at trainer step {step}")
        time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))
