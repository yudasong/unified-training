"""Strict streaming freshness: actual episode versions and complete producer ranks."""
from types import SimpleNamespace

import numpy as np
import pytest

from unified_training.dataset import (
    NoFreshEpisodes, StopRequested, StreamingEpisodeDataset, build_unified_datasets, streaming_primary,
)
from unified_training.episodes import INDEX_FILE, RoundWriter


def _round(parent, name, versions, *, rank=0, world=1, manifest_version=999, marked=True):
    path = parent / name
    writer = RoundWriter(path)
    for episode, version in enumerate(versions):
        writer.add(np.array([1, 2, 3], dtype=np.int32),
                   np.array([0., 1., 1.], dtype=np.float32),
                   prompt_len=1, reward=1., weight=1.,
                   policy_version=version, group_id=episode)
    manifest = {}
    if marked:
        manifest.update(generator_rank=rank, generator_world=world)
    if manifest_version is not None:
        manifest['policy_version'] = manifest_version
    writer.finalize(manifest)
    return path


def _dataset(parent, *, strict=True, step=10, window=0, **kwargs):
    return StreamingEpisodeDataset(
        [str(parent)], seq_length=8, total_samples=100,
        strict_policy_age=strict, initial_trainer_step=step,
        max_policy_age_steps=4, recency_window=window, **kwargs,
    )


def _versions(ds):
    return [int(ds._rounds[r].index[e]['policy_version']) for r, e in ds._eligible_refs]


@pytest.mark.parametrize('limit', [0, -1])
def test_strict_requires_positive_age_limit(tmp_path, limit):
    with pytest.raises(ValueError, match='positive max_policy_age_steps'):
        StreamingEpisodeDataset([str(tmp_path)], seq_length=8, total_samples=1,
                                strict_policy_age=True, max_policy_age_steps=limit)


def test_strict_rejects_negative_initial_step(tmp_path):
    with pytest.raises(ValueError, match='initial_trainer_step'):
        _dataset(tmp_path, step=-1)


@pytest.mark.parametrize('window', [0, 1])
def test_strict_removes_newest_floor_and_empty_fallback(tmp_path, window):
    _round(tmp_path, 'r000', [0], manifest_version=0)
    legacy = _dataset(tmp_path, strict=False, window=window)
    legacy.set_trainer_step(10)
    strict = _dataset(tmp_path, window=window)
    assert _versions(legacy) == [0]
    assert strict._eligible_refs == []
    with pytest.raises(NoFreshEpisodes, match='trainer step 10'):
        strict[0]


@pytest.mark.parametrize('manifest_version', [None, 0, 999])
def test_actual_index_versions_override_manifest_and_include_cutoff(tmp_path, manifest_version):
    _round(tmp_path, 'r000', [-1, 5, 6, 9, 10, 11], manifest_version=manifest_version)
    ds = _dataset(tmp_path, without_replacement=True)
    assert _versions(ds) == [6, 9, 10]
    assert {int(ds[i]['source_policy_version']) for i in range(3)} == {6, 9, 10}
    ds.set_trainer_step(11)
    assert _versions(ds) == [9, 10, 11]  # same round survives, different actual rows


def test_unchanged_pool_preserves_shuffle_revision_and_cache(tmp_path):
    _round(tmp_path, 'r000', [9, 10])
    ds = _dataset(tmp_path, without_replacement=True)
    first = ds[0]['source_policy_version'].item()
    revision, cache = ds._eligible_revision, ds._perm_cache
    ds.set_trainer_step(10)
    ds.rescan()
    ds.set_trainer_step(11)
    assert ds._eligible_revision == revision
    assert ds._perm_cache is cache
    assert ds[0]['source_policy_version'].item() == first
    ds.set_trainer_step(14)
    assert _versions(ds) == [10]
    assert ds._eligible_revision > revision


@pytest.mark.parametrize('window', [0, 2])
@pytest.mark.parametrize('second_rank', ['missing', 'stale', 'unknown', 'future'])
def test_incomplete_fresh_rank_pool_stays_empty(tmp_path, window, second_rank):
    _round(tmp_path, 'g0-r000', [10], rank=0, world=2)
    if second_rank != 'missing':
        version = {'stale': 0, 'unknown': -1, 'future': 11}[second_rank]
        _round(tmp_path, 'g1-r000', [version], rank=1, world=2)
    ds = _dataset(tmp_path, window=window)
    assert not ds._eligible_refs
    with pytest.raises(NoFreshEpisodes):
        ds[0]


def test_rescan_recovers_all_ranks_balances_fresh_rows_and_expires(tmp_path):
    _round(tmp_path, 'g0-r000', [10, 10, 0], rank=0, world=2)
    _round(tmp_path, 'g1-r000', [0], rank=1, world=2)
    ds = _dataset(tmp_path, window=2)
    assert not ds._eligible_refs
    _round(tmp_path, 'g1-r001', [10, 0, 10, 10], rank=1, world=2)
    assert ds.rescan() == 4
    assert _versions(ds) == [10] * 4
    ranks = [ds._rounds[r].manifest['generator_rank'] for r, _ in ds._eligible_refs]
    assert ranks == [0, 0, 1, 1]
    assert int(ds[0]['source_policy_version']) == 10
    ds.set_trainer_step(15)
    with pytest.raises(NoFreshEpisodes):
        ds[0]
    _round(tmp_path, 'g0-r002', [15], rank=0, world=2)
    _round(tmp_path, 'g1-r002', [15], rank=1, world=2)
    ds.rescan()
    assert _versions(ds) == [15, 15]


def test_strict_recency_selects_latest_fresh_rounds_only(tmp_path):
    _round(tmp_path, 'r000', [9])
    _round(tmp_path, 'r001', [10])
    _round(tmp_path, 'r002', [0])
    ds = _dataset(tmp_path, window=1)
    assert _versions(ds) == [10]


@pytest.mark.parametrize('window', [0, 1])
def test_generic_producers_fail_closed_only_in_strict_mode(tmp_path, window):
    _round(tmp_path, 'r000', [10], marked=False, manifest_version=10)
    strict = _dataset(tmp_path, window=window)
    legacy = _dataset(tmp_path, strict=False, window=window)
    legacy.set_trainer_step(10)
    assert not strict._eligible_refs
    assert _versions(legacy) == [10]


def test_missing_episode_version_fails_closed(tmp_path):
    path = _round(tmp_path, 'r000', [10], manifest_version=10)
    index = np.load(path / INDEX_FILE)
    fields = [name for name in index.dtype.names if name != 'policy_version']
    reduced = np.empty(len(index), dtype=[(name, index.dtype.fields[name][0]) for name in fields])
    for name in fields:
        reduced[name] = index[name]
    np.save(path / INDEX_FILE, reduced)
    ds = _dataset(tmp_path)
    with pytest.raises(NoFreshEpisodes):
        ds[0]


def test_fully_excluded_rank_does_not_count_as_available(tmp_path):
    _round(tmp_path, 'g0-r000', [10], rank=0, world=2)
    other = _round(tmp_path, 'g1-r000', [10], rank=1, world=2)
    ds = _dataset(tmp_path, exclude=frozenset({(str(other.resolve()), 0)}))
    assert not ds._eligible_refs


def test_empty_strict_constructor_can_recover(tmp_path):
    ds = _dataset(tmp_path)
    with pytest.raises(NoFreshEpisodes):
        ds[0]
    _round(tmp_path, 'r000', [10])
    ds.rescan()
    assert _versions(ds) == [10]


def test_default_initial_step_accepts_policy_zero(tmp_path):
    _round(tmp_path, 'r000', [0, 1])
    ds = StreamingEpisodeDataset([str(tmp_path)], seq_length=8, total_samples=1,
                                 strict_policy_age=True, max_policy_age_steps=4)
    assert _versions(ds) == [0]


def test_builder_passes_strict_and_resume_step(tmp_path):
    _round(tmp_path, 'r000', [6, 10, 11])
    ds, _, _ = build_unified_datasets(
        episodes_dirs=[str(tmp_path)], train_samples=10, valid_samples=0,
        valid_holdout=0, sequence_length=8, streaming=True,
        strict_policy_age=True, max_policy_age_steps=4, initial_trainer_step=10,
    )
    assert streaming_primary(ds) is ds
    assert _versions(ds) == [6, 10]


def test_strict_builder_rejects_fixed_pool(tmp_path):
    with pytest.raises(ValueError, match='requires streaming'):
        build_unified_datasets(episodes_dirs=[str(tmp_path)], train_samples=1,
                               valid_samples=0, strict_policy_age=True,
                               max_policy_age_steps=4)


def test_builder_stops_while_waiting_for_first_rounds(tmp_path):
    rounds = tmp_path / 'rounds'
    rounds.mkdir()
    stop = tmp_path / 'STOP_TRAINER'
    stop.touch()
    with pytest.raises(StopRequested):
        build_unified_datasets(episodes_dirs=[str(rounds)], train_samples=1, valid_samples=0,
                               valid_holdout=0, sequence_length=8, streaming=True, stop_file=stop)


def test_streaming_primary_unwraps_supported_mixers_and_handles_cycles(tmp_path):
    _round(tmp_path, 'r000', [10])
    ds = _dataset(tmp_path)
    wrapped = SimpleNamespace(_primary=SimpleNamespace(_streams=[ds, object()]))
    assert streaming_primary(wrapped) is ds
    assert streaming_primary(object()) is None
    cycle = SimpleNamespace()
    cycle._primary = cycle
    assert streaming_primary(cycle) is None


def test_strict_window_keeps_equal_recent_round_counts(tmp_path):
    for rank, versions in [(0, [6, 8, 10]), (1, [7, 9])]:
        for number, version in enumerate(versions):
            _round(tmp_path, f'g{rank}-r{number:03}', [version], rank=rank, world=2)
    ds = _dataset(tmp_path, window=4)
    assert _versions(ds) == [8, 10, 7, 9]


@pytest.mark.parametrize('window', [1, 3])
def test_invalid_world_sized_window_never_uses_generic_fallback(tmp_path, window):
    for rank in range(2):
        _round(tmp_path, f'g{rank}-r000', [10], rank=rank, world=2)
    ds = _dataset(tmp_path, window=window)
    assert not ds._eligible_refs
    with pytest.raises(NoFreshEpisodes):
        ds[0]


def test_inconsistent_world_metadata_fails_closed(tmp_path):
    _round(tmp_path, 'g0-r000', [10], rank=0, world=1)
    _round(tmp_path, 'g1-r000', [10], rank=1, world=2)
    ds = _dataset(tmp_path)
    assert not ds._eligible_refs
