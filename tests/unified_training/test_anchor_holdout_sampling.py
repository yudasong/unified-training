"""Identity exclusion must preserve valid historical prefixes and survive seeds."""

import hashlib
import json

import numpy as np
import pytest

from unified_training.dataset import build_unified_datasets
from unified_training.episodes import RoundWriter


def pools(tmp_path, *, legacy=False):
    primary = tmp_path / "rl"
    source = tmp_path / "anchor"
    for path, count, offset in ((primary / "round", 32, 10000), (source, 24, 100)):
        writer = RoundWriter(path)
        for i in range(count):
            length = 3 + i % 3
            writer.add(
                np.full(length, offset + i, dtype=np.int32),
                np.array([0] + [1] * (length - 1), dtype=np.float32),
                prompt_len=1, reward=1, weight=1, policy_version=0, group_id=i,
            )
        writer.finalize()
    holdout = tmp_path / "holdout"
    holdout.mkdir()
    (holdout / "DONE").touch()
    indices = np.random.default_rng((17, 0)).permutation(24)[20:]
    manifest = dict(
        source_anchor=str(source.resolve()),
        source_manifest_sha256=hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest(),
        source_n_episodes=24, permutation_seed=17, permutation_epoch=0,
        permutation_tail_start=20, eval_windows=4, n_episodes=4,
    )
    if not legacy:
        manifest.update(
            source_episode_indices=indices.tolist(),
            source_episode_indices_sha256=hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest(),
        )
    (holdout / "manifest.json").write_text(json.dumps(manifest))
    return primary, source, holdout, set(indices.tolist())


def build(primary, source, holdout, *, seed=16, mode="floor", protected=True):
    kwargs = dict(
        episodes_dirs=[str(primary)], anchor_dirs=[str(source)],
        anchor_token_fraction=0.2, train_samples=80, valid_samples=0,
        sequence_length=8, split_seed=seed, mix_recalc="launch",
        streaming=mode == "online",
        midtrain_holdout_dirs=[str(holdout)] if protected else [],
        reserved_anchor_tail=0 if protected else 4,
    )
    if mode == "online":
        kwargs["mix_recalc"] = "online"
    elif mode == "schedule":
        kwargs["anchor_token_fraction"] = 0
        kwargs["streams"] = [{"dirs": [str(source)], "token_fraction": {"start": 0.2, "end": 0.2}}]
    return build_unified_datasets(**kwargs)[0]


@pytest.mark.parametrize("mode", ["floor", "online", "schedule"])
def test_matching_seed_preserves_previous_rows_and_dose(tmp_path, mode):
    primary, source, holdout, _ = pools(tmp_path)
    old = build(primary, source, holdout, mode=mode, protected=False)
    guarded = build(primary, source, holdout, mode=mode)
    # Global-index mixers cycle the original safe prefix even across many wraps.
    count = 80 if mode == "floor" else 400
    for i in range(count):
        before, after = old[i], guarded[i]
        assert before.keys() == after.keys()
        for key in before:
            assert before[key].equal(after[key]), (mode, i, key)
    if mode != "schedule":
        assert old.anchor_fraction == guarded.anchor_fraction
        for step in range(10):
            assert old.additive_plan(step, 4) == guarded.additive_plan(step, 4)
        for i in range(20):
            assert old.anchor_row(i)["tokens"].equal(guarded.anchor_row(i)["tokens"])


@pytest.mark.parametrize("mode", ["floor", "online", "schedule"])
@pytest.mark.parametrize("seed", [16, 17, 18])
def test_source_ids_are_excluded_for_every_seed_and_epoch(tmp_path, mode, seed):
    primary, source, holdout, ids = pools(tmp_path)
    guarded = build(primary, source, holdout, seed=seed, mode=mode)
    anchor = guarded._streams[1] if mode == "schedule" else guarded._anchor
    assert anchor.n_episodes == 20
    assert {ep for _, ep in anchor._refs}.isdisjoint(ids)
    for i in range(3 * anchor.n_episodes):
        assert int(anchor[i]["tokens"][0]) - 100 not in ids
    for i in range(400):
        row = guarded[i]
        if not bool(row["sampling_filter"]):
            assert int(row["tokens"][0]) - 100 not in ids
    if mode != "schedule":
        # Includes additive slots that would cross the old reserved tail.
        for i in range(80):
            assert int(guarded.anchor_row(i)["tokens"][0]) - 100 not in ids


def test_legacy_positional_guard_requires_original_seed(tmp_path):
    primary, source, holdout, ids = pools(tmp_path, legacy=True)
    with pytest.raises(ValueError, match="matching anchor seed"):
        build(primary, source, holdout, seed=17)
    old = build(primary, source, holdout, protected=False)
    guarded = build(primary, source, holdout)
    for i in range(80):
        assert old[i]["tokens"].equal(guarded[i]["tokens"])
    for i in range(60):
        assert int(guarded.anchor_row(i)["tokens"][0]) - 100 not in ids


def test_source_alias_matches_but_other_source_is_unchanged(tmp_path):
    primary, source, holdout, _ = pools(tmp_path)
    alias = tmp_path / "alias"
    alias.symlink_to(source, target_is_directory=True)
    assert build(primary, alias, holdout)._anchor.n_episodes == 20
    manifest_path = holdout / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_anchor"] = str(tmp_path / "unrelated")
    manifest_path.write_text(json.dumps(manifest))
    assert build(primary, source, holdout)._anchor.n_episodes == 24


def test_matching_path_with_changed_source_refuses(tmp_path):
    primary, source, holdout, _ = pools(tmp_path)
    path = source / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["extra"] = "changed"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="source identity changed"):
        build(primary, source, holdout)


def test_duplicate_source_alias_is_rejected(tmp_path):
    primary, source, holdout, _ = pools(tmp_path)
    alias = tmp_path / "alias"
    alias.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="duplicate holdout source"):
        build_unified_datasets(
            episodes_dirs=[str(primary)], train_samples=32, valid_samples=0,
            anchor_dirs=[str(source), str(alias)], anchor_token_fraction=0.2,
            midtrain_holdout_dirs=[str(holdout)], mix_recalc="launch",
        )


def test_combined_anchor_pool_excludes_only_matching_source_ids(tmp_path):
    primary, source, holdout, ids = pools(tmp_path)
    guarded, _, _ = build_unified_datasets(
        episodes_dirs=[str(primary)], train_samples=32, valid_samples=0,
        anchor_dirs=[str(source), str(primary / "round")], anchor_token_fraction=0.2,
        midtrain_holdout_dirs=[str(holdout)], mix_recalc="launch", split_seed=19,
    )
    refs = set(guarded._anchor._refs)
    assert all((0, ep) not in refs and (1, ep) in refs for ep in ids)
    assert guarded._anchor.n_episodes == 24 + 32 - 4
