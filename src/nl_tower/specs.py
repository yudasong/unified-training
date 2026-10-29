"""Puzzle distributions ("rungs") and deterministic item streams.

``make_dataset(spec, size, seed)`` builds the dataset exactly as ``reasoning_gym.create_dataset("nl_tower", size=,
seed=, **params)`` would, so ``ds[i]`` is a pure function of ``(seed, i)``.

Paper names (Appendix B, C.1): ``nltw_e`` / ``nltw_m0`` = midtraining questions with 6-7 / 8-9 people (demonstrated
patterns only; ``nltw_m0`` is also the demonstrated-pattern test of Table 8); ``nltw_m`` = Base (8-9 people, 2-4
omitted steps); ``nltw_hb`` = Extended (10-11 people); ``nltw_hb12`` = Extended generalization (12 people, test only).
``nltw_h`` and ``nltw_h9`` are exclusion sets: the published sets were drawn excluding their questions (contamination
lists), so they are generated to keep the other sets byte-identical.

Step-pattern sets (weights of the chain-step patterns):
  G0  ("covered", demonstrated in mid-training): friend, friend2, order, rankn, dfriend_occ
  ALL adds the held-out compositions dfriend_rank, dfriend_above, nested (RL prompts and tests only)
  HARD = ALL with the held-out compositions weighted up
"""

import hashlib
import os

from .generator import NLTowerConfig, NLTowerDataset

_TW_G0 = {
    "friend": 3,
    "friend2": 1,
    "order": 2,
    "rankn": 1,
    "dfriend_occ": 2,
    "dfriend_rank": 0,
    "dfriend_above": 0,
    "nested": 0,
}
_TW_ALL = {
    "friend": 2,
    "friend2": 1,
    "order": 1,
    "rankn": 1,
    "dfriend_occ": 2,
    "dfriend_rank": 2,
    "dfriend_above": 2,
    "nested": 1,
}
_TW_HARD = {
    "friend": 1,
    "friend2": 1,
    "order": 1,
    "rankn": 1,
    "dfriend_occ": 1,
    "dfriend_rank": 2,
    "dfriend_above": 2,
    "nested": 2,
}
HELD_OUT = ("dfriend_rank", "dfriend_above", "nested")

# spec -> (task, generator parameters). "deferred_min/max" bound the number of HELD-OUT chain steps.
SPECS = {
    # e / m0: midtraining questions (G0 patterns only); m / h: RL prompts and tests (omitted compositions).
    "nltw_e": (
        "nl_tower",
        dict(
            n_people_min=6,
            n_people_max=7,
            max_value=40,
            step_patterns=_TW_G0,
            deferred_min=0,
            deferred_max=0,
            n_noise=2,
        ),
    ),
    "nltw_m0": (
        "nl_tower",
        dict(
            n_people_min=8,
            n_people_max=9,
            max_value=40,
            step_patterns=_TW_G0,
            deferred_min=0,
            deferred_max=0,
            n_noise=2,
        ),
    ),
    "nltw_m": (
        "nl_tower",
        dict(
            n_people_min=8,
            n_people_max=9,
            max_value=40,
            step_patterns=_TW_ALL,
            deferred_min=2,
            deferred_max=4,
            n_noise=2,
        ),
    ),
    "nltw_h": (
        "nl_tower",
        dict(
            n_people_min=9,
            n_people_max=10,
            max_value=40,
            step_patterns=_TW_HARD,
            deferred_min=4,
            deferred_max=6,
            n_noise=2,
        ),
    ),
    # the hard rung restricted to 9 people (generated only for the contamination lists)
    "nltw_h9": (
        "nl_tower",
        dict(
            n_people_min=9,
            n_people_max=9,
            max_value=40,
            step_patterns=_TW_HARD,
            deferred_min=4,
            deferred_max=6,
            n_noise=2,
        ),
    ),
    # Extended: Base patterns and 2-4 omitted steps, 10-11 people
    "nltw_hb": (
        "nl_tower",
        dict(
            n_people_min=10,
            n_people_max=11,
            max_value=40,
            step_patterns=_TW_ALL,
            deferred_min=2,
            deferred_max=4,
            n_noise=2,
        ),
    ),
    # Extended generalization: 12 people, never trained on (test only)
    "nltw_hb12": (
        "nl_tower",
        dict(
            n_people_min=12,
            n_people_max=12,
            max_value=40,
            step_patterns=_TW_ALL,
            deferred_min=2,
            deferred_max=4,
            n_noise=2,
        ),
    ),
}


def qhash(q: str) -> str:
    """Question id: first 16 hex digits of sha256(question text)."""
    return hashlib.sha256(q.encode("utf-8")).hexdigest()[:16]


def make_dataset(spec: str, size: int, seed: int) -> NLTowerDataset:
    task, params = SPECS[spec]
    assert task == "nl_tower", task
    cfg = NLTowerConfig(size=size, seed=seed, **params)
    cfg.validate()
    return NLTowerDataset(config=cfg)


# Parallel item generation. ds[i] is a pure function of (seed, i), so a fork-pool over indices reproduces the
# sequential order bit for bit.
_PDS = None


def _pget(i):
    return _PDS[i]


def rg_workers() -> int:
    """RG_WORKERS, default: the CPUs this process may run on, at most 32 (1 = sequential)."""
    if os.environ.get("RG_WORKERS"):
        return int(os.environ["RG_WORKERS"])
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    return min(cpus, 32)


def iter_items(ds, size: int):
    """Yield ds[0..size) in index order from rg_workers() forked workers.
    Consumers may break early; closing the generator terminates the pool."""
    global _PDS
    import multiprocessing as mp

    w = rg_workers()
    if w <= 1 or size < 64:
        for i in range(size):
            yield ds[i]
        return
    _PDS = ds
    pool = mp.get_context("fork").Pool(w)
    try:
        yield from pool.imap(_pget, range(size), chunksize=8)
    finally:
        pool.terminate()
        pool.join()
