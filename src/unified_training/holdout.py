"""Validate anchor holdout identity without importing training dependencies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import struct


def load_anchor_holdouts(directories: list[str]) -> list[dict]:
    """Read committed holdout metadata; preserve legacy positional records.

    Source identity is recorded here, not matched to a training pool. The
    dataset builder must perform that match before applying any exclusion.
    Legacy records without episode IDs require the original epoch-zero
    permutation seed and prefix guard at the point of sampling.
    """
    records = []
    for directory in directories:
        root = Path(directory).resolve()
        path = root / "manifest.json"

        def invalid(message: str) -> ValueError:
            return ValueError(f"anchor holdout {path}: {message}")

        if not (root / "DONE").is_file():
            raise invalid("missing DONE marker")
        try:
            raw = path.read_bytes()
            manifest = json.loads(raw)
        except (OSError, ValueError) as error:
            raise invalid(f"cannot read manifest: {error}") from error
        if not isinstance(manifest, dict):
            raise invalid("manifest must be a JSON object")

        def integer(name: str, default=None) -> int:
            value = manifest.get(name, default)
            if type(value) is not int:
                raise invalid(f"{name} must be an integer")
            return value

        def digest(name: str, *, optional: bool = False):
            value = manifest.get(name)
            if value is None and optional:
                return None
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
                raise invalid(f"{name} must be a SHA-256 hex digest")
            return value.lower()

        source = manifest.get("source_anchor")
        if not isinstance(source, str) or not source.strip():
            raise invalid("source_anchor must be a nonempty path")
        source_path = Path(source)
        if not source_path.is_absolute():
            source_path = root / source_path
        source_count = integer("source_n_episodes")
        if source_count <= 0:
            raise invalid("source_n_episodes must be positive")
        seed = integer("permutation_seed")
        if seed < 0:
            raise invalid("permutation_seed must be nonnegative")
        epoch = integer("permutation_epoch", 0)
        if epoch != 0:
            raise invalid("only permutation_epoch=0 is supported")
        tail = integer("permutation_tail_start")
        if not 0 <= tail < source_count:
            raise invalid("permutation_tail_start is outside the source")
        count = integer("eval_windows", manifest.get("n_episodes"))
        if not 0 < count <= source_count - tail:
            raise invalid("eval_windows must fit within the source permutation tail")
        if "n_episodes" in manifest and integer("n_episodes") != count:
            raise invalid("n_episodes does not match eval_windows")

        indices = manifest.get("source_episode_indices")
        indices_digest = digest("source_episode_indices_sha256", optional=indices is None)
        if indices is not None:
            if not isinstance(indices, list) or len(indices) != count:
                raise invalid("source_episode_indices length must equal eval_windows")
            if any(type(index) is not int or not 0 <= index < source_count or index >= 2**63 for index in indices):
                raise invalid("source_episode_indices must contain in-range signed 64-bit integers")
            if len(set(indices)) != len(indices):
                raise invalid("source_episode_indices must be unique")
            actual = hashlib.sha256(b"".join(struct.pack("<q", index) for index in indices)).hexdigest()
            if actual != indices_digest:
                raise invalid("source_episode_indices_sha256 does not match the packed episode IDs")

        records.append({
            "manifest_path": str(path),
            "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "source_anchor": str(source_path.resolve()),
            "source_manifest_sha256": digest("source_manifest_sha256"),
            "source_n_episodes": source_count,
            "permutation_seed": seed,
            "permutation_epoch": epoch,
            "permutation_tail_start": tail,
            "source_episode_indices": indices,
            "source_episode_indices_sha256": indices_digest,
            "eval_windows": count,
        })
    return records


def anchor_holdout_contract(directories: list[str], split_seed: int) -> dict:
    """Compact provenance with identity hashes and the supported guard strategy."""
    if type(split_seed) is not int or split_seed < 0:
        raise ValueError("split_seed must be a nonnegative integer")
    holdouts = []
    for record in load_anchor_holdouts(directories):
        indices = record["source_episode_indices"]
        holdouts.append({
            **{key: value for key, value in record.items() if key != "source_episode_indices"},
            "source_episode_indices_count": None if indices is None else len(indices),
            "strategy": "legacy_epoch0_prefix" if indices is None else "source_episode_ids",
        })
    return {
        "holdout_algorithm": "holdout_source_ids_v1",
        "split_seed": split_seed,
        "midtrain_holdouts": holdouts,
    }
