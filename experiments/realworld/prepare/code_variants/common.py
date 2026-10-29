"""Shared helpers for the App. D.4 corpus variants (code removed / code replaced)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RECEIPTS = HERE / "receipts"
# The three designated code sources of the 240M midtraining corpus (24.8% of its tokens).
REMOVED = {"cranecode", "stack_edu-fim", "code-meta-reasoning"}


def sha(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def load_receipt(name: str) -> dict:
    return json.loads((RECEIPTS / name).read_text())


def load_pool(path: Path):
    """(manifest, index, tokens, weights) of an episode-format directory."""
    path = Path(path)
    return (json.loads((path / "manifest.json").read_text()),
            np.load(path / "index.npy", mmap_mode="r"),
            np.memmap(path / "tokens.bin", dtype="<i4", mode="r"),
            np.memmap(path / "weights.bin", dtype="<f4", mode="r"))


def excluded_original_ids(anchor: Path, ai, at, holdout: Path | None) -> set[int]:
    """Trainer-holdout episodes of the 240M corpus plus exact duplicates of their windows.

    With ``holdout`` (the trainer validation-holdout directory built from this corpus) the set is recomputed
    as in the original build and checked against the pinned receipt; without it the receipt is used.
    """
    receipt = load_receipt("excluded_source_episode_ids.json")
    if receipt["source_anchor_manifest_sha256"] != sha(Path(anchor) / "manifest.json"):
        raise ValueError("the excluded-id receipt belongs to a different source corpus")
    pinned = set(map(int, receipt["excluded_source_episode_ids"]))
    if holdout is None:
        return pinned
    manifest = json.loads((Path(holdout) / "manifest.json").read_text())
    assert manifest["source_n_episodes"] == len(ai)
    assert manifest["source_manifest_sha256"] == sha(Path(anchor) / "manifest.json")
    ids = np.asarray(manifest["source_episode_indices"], dtype="<i8")
    assert hashlib.sha256(ids.tobytes()).hexdigest() == manifest["source_episode_indices_sha256"]
    held_ids = set(map(int, ids))

    def episode_hash(i):
        r = ai[i]
        lo = int(r["offset"])
        return hashlib.sha256(at[lo:lo + int(r["total_len"])].tobytes()).digest()

    held_hashes = {episode_hash(i) for i in held_ids}
    excluded = {i for i in range(len(ai)) if i in held_ids or episode_hash(i) in held_hashes}
    if excluded != pinned:
        raise ValueError("recomputed holdout exclusions differ from the pinned receipt")
    return excluded
