"""Anchor holdout metadata identity and legacy-format validation."""

import hashlib
import json
import struct

import pytest

from unified_training.holdout import anchor_holdout_contract, load_anchor_holdouts


@pytest.fixture
def holdout_metadata(tmp_path):
    directory = tmp_path / "holdout"
    directory.mkdir()
    (directory / "DONE").touch()
    indices = [7, 2]
    manifest = {
        "source_anchor": "../anchor",
        "source_manifest_sha256": "a" * 64,
        "source_n_episodes": 10,
        "permutation_seed": 1235,
        "permutation_epoch": 0,
        "permutation_tail_start": 8,
        "eval_windows": 2,
        "n_episodes": 2,
        "source_episode_indices": indices,
        "source_episode_indices_sha256": hashlib.sha256(struct.pack("<qq", *indices)).hexdigest(),
    }
    return directory, manifest


def _write_holdout(directory, manifest):
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest))
    return [str(directory)]


def test_load_anchor_holdout_normalizes_identity(holdout_metadata):
    directory, manifest = holdout_metadata
    record, = load_anchor_holdouts(_write_holdout(directory, manifest))
    assert record["source_anchor"] == str(directory.parent / "anchor")
    assert record["manifest_path"] == str(directory / "manifest.json")
    assert record["manifest_sha256"] == hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()
    assert record["source_episode_indices"] == [7, 2]
    assert record["source_episode_indices_sha256"] == manifest["source_episode_indices_sha256"]
    assert json.loads(json.dumps(record)) == record


def test_load_anchor_holdout_preserves_legacy_identity(holdout_metadata):
    directory, manifest = holdout_metadata
    del manifest["source_episode_indices"]
    del manifest["permutation_epoch"]
    del manifest["eval_windows"]
    record, = load_anchor_holdouts(_write_holdout(directory, manifest))
    assert record["source_episode_indices"] is None
    assert record["permutation_epoch"] == 0
    assert record["eval_windows"] == 2
    assert record["permutation_seed"] == 1235
    assert record["source_episode_indices_sha256"] == manifest["source_episode_indices_sha256"]


@pytest.mark.parametrize("field,value,match", [
    ("source_anchor", "", "source_anchor"),
    ("source_manifest_sha256", "bad", "SHA-256"),
    ("source_n_episodes", 0, "positive"),
    ("source_n_episodes", True, "integer"),
    ("permutation_seed", -1, "nonnegative"),
    ("permutation_seed", "1235", "integer"),
    ("permutation_epoch", 1, "epoch=0"),
    ("permutation_tail_start", -1, "outside"),
    ("permutation_tail_start", 10, "outside"),
    ("eval_windows", 3, "fit"),
    ("eval_windows", 0, "fit"),
    ("n_episodes", 1, "does not match"),
    ("source_episode_indices", [7], "length"),
    ("source_episode_indices", [7, 7], "unique"),
    ("source_episode_indices", [7, 10], "in-range"),
    ("source_episode_indices", [7, -1], "in-range"),
    ("source_episode_indices", [7, True], "in-range"),
    ("source_episode_indices", [7, 3], "does not match"),
    ("source_episode_indices_sha256", "f" * 64, "does not match"),
    ("source_episode_indices_sha256", None, "SHA-256"),
])
def test_load_anchor_holdout_rejects_invalid_metadata(holdout_metadata, field, value, match):
    directory, manifest = holdout_metadata
    manifest[field] = value
    with pytest.raises(ValueError, match=match):
        load_anchor_holdouts(_write_holdout(directory, manifest))


def test_load_anchor_holdout_requires_commit_and_manifest(holdout_metadata):
    directory, manifest = holdout_metadata
    with pytest.raises(ValueError, match="cannot read manifest"):
        load_anchor_holdouts([str(directory)])
    _write_holdout(directory, manifest)
    (directory / "DONE").unlink()
    with pytest.raises(ValueError, match="DONE"):
        load_anchor_holdouts([str(directory)])


@pytest.mark.parametrize("raw", ["{", "[]"])
def test_load_anchor_holdout_rejects_malformed_manifest(holdout_metadata, raw):
    directory, _ = holdout_metadata
    (directory / "manifest.json").write_text(raw)
    with pytest.raises(ValueError, match="manifest"):
        load_anchor_holdouts([str(directory)])


@pytest.mark.parametrize("legacy", [False, True])
def test_anchor_holdout_contract_records_compact_strategy(holdout_metadata, legacy):
    directory, manifest = holdout_metadata
    if legacy:
        del manifest["source_episode_indices"]
    contract = anchor_holdout_contract(_write_holdout(directory, manifest), 1236)
    assert contract["split_seed"] == 1236
    assert contract["holdout_algorithm"] == "holdout_source_ids_v1"
    record, = contract["midtrain_holdouts"]
    assert record["strategy"] == ("legacy_epoch0_prefix" if legacy else "source_episode_ids")
    assert record["source_episode_indices_count"] == (None if legacy else 2)
    assert "source_episode_indices" not in record
    assert record["source_manifest_sha256"] == manifest["source_manifest_sha256"]


def test_empty_anchor_holdouts():
    assert load_anchor_holdouts([]) == []
    assert anchor_holdout_contract([], 1234)["midtrain_holdouts"] == []
