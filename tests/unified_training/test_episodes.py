# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for the unified episode shard format and the weighted dataset.

Shard-format tests are numpy-only and run in plain CI. Dataset tests need
torch and are skipped when it isn't importable.
"""

from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest

from unified_training.episodes import (
    RoundWriter,
    anchor_row_fraction,
    is_round_dir,
    load_round,
    load_rounds,
    normalize_weights,
    pack_token_docs,
    rank_balanced_recent_round_indices,
    row_fractions,
    validate_no_anchor_replay,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None


def _episode(prompt_len: int, resp_len: int, weight: float, base: int = 100):
    tokens = np.arange(base, base + prompt_len + resp_len, dtype=np.int32)
    weights = np.zeros(prompt_len + resp_len, dtype=np.float32)
    weights[prompt_len:] = weight
    return tokens, weights


def _write_round(tmp_path, episodes, name="r000"):
    w = RoundWriter(tmp_path / name)
    for i, (prompt_len, resp_len, weight, reward) in enumerate(episodes):
        tokens, weights = _episode(prompt_len, resp_len, weight, base=100 * (i + 1))
        w.add(
            tokens, weights, prompt_len=prompt_len, reward=reward, weight=weight,
            policy_version=3, group_id=i,
        )
    w.finalize({"source": "test"})
    return tmp_path / name


def test_roundtrip(tmp_path):
    d = _write_round(tmp_path, [(4, 6, 2.0, 1.0), (3, 2, 0.5, 1.0), (5, 8, 1.0, 0.0)])
    assert is_round_dir(d)
    r = load_round(d)
    assert len(r.index) == 3
    assert r.manifest["source"] == "test"
    assert r.manifest["n_episodes"] == 3
    assert r.manifest["n_tokens"] == 10 + 5 + 13

    tokens, weights, row = r.episode(1)
    exp_tokens, exp_weights = _episode(3, 2, 0.5, base=200)
    np.testing.assert_array_equal(tokens, exp_tokens)
    np.testing.assert_array_equal(weights, exp_weights)
    assert int(row["prompt_len"]) == 3
    assert float(row["weight"]) == 0.5
    assert int(row["policy_version"]) == 3


def test_writer_validation(tmp_path):
    w = RoundWriter(tmp_path / "bad")
    tokens, weights = _episode(4, 4, 1.0)
    weights[0] = 1.0  # nonzero prompt weight must be rejected
    with pytest.raises(ValueError, match="prompt tokens"):
        w.add(tokens, weights, prompt_len=4, reward=1.0, weight=1.0, policy_version=0, group_id=0)
    with pytest.raises(ValueError, match="empty round"):
        w.finalize()
    assert not is_round_dir(tmp_path / "bad")


def test_unfinalized_round_is_invisible(tmp_path):
    d = _write_round(tmp_path, [(2, 2, 1.0, 1.0)], name="r000")
    (tmp_path / "r001").mkdir()  # in-progress round: no DONE marker
    rounds = load_rounds([tmp_path / "r000"])
    assert len(rounds) == 1
    with pytest.raises(FileNotFoundError):
        load_round(tmp_path / "r001")


def test_load_rounds_parent_dir(tmp_path):
    _write_round(tmp_path, [(2, 2, 1.0, 1.0)], name="r000")
    _write_round(tmp_path, [(2, 3, 2.0, 1.0), (2, 2, 1.0, 0.0)], name="r001")
    rounds = load_rounds([tmp_path])
    assert [len(r.index) for r in rounds] == [1, 2]


def test_round_payload_is_lazy_and_reopens_after_release(tmp_path):
    d = _write_round(tmp_path, [(4, 6, 2.0, 1.0), (3, 2, 0.5, 1.0)])
    r = load_round(d)
    # No payload memmaps (hence no file descriptors) until an episode is read.
    assert r.tokens is None and r.weights is None
    tokens, weights, _row = r.episode(1)
    exp_tokens, exp_weights = _episode(3, 2, 0.5, base=200)
    np.testing.assert_array_equal(tokens, exp_tokens)
    np.testing.assert_array_equal(weights, exp_weights)
    assert r.tokens is not None and r.weights is not None
    r.release()
    assert r.tokens is None and r.weights is None
    tokens_again, weights_again, _row = r.episode(1)  # transparently reopens
    np.testing.assert_array_equal(tokens_again, exp_tokens)
    np.testing.assert_array_equal(weights_again, exp_weights)


def test_load_round_rejects_truncated_payload(tmp_path):
    from unified_training.episodes import WEIGHTS_FILE

    d = _write_round(tmp_path, [(4, 6, 1.0, 1.0)])
    payload = d / WEIGHTS_FILE
    payload.write_bytes(payload.read_bytes()[:-4])
    with pytest.raises(ValueError, match="index expectation"):
        load_round(d)


def test_load_rounds_interleaves_generator_ranks_by_global_chunk(tmp_path):
    names = [
        "gmath12k-g00-b0000001-q000000008",
        "gmath12k-g07-b0000000-q000000007",
        "gmath12k-g00-b0000000-q000000000",
        "gmath12k-g03-b0000000-q000000003",
        "gmath12k-g01-b0000000-q000000001",
    ]
    for name in names:
        _write_round(tmp_path, [(2, 2, 1.0, 1.0)], name=name)

    rounds = load_rounds([tmp_path])
    assert [round_.round_dir.name for round_ in rounds] == [
        "gmath12k-g00-b0000000-q000000000",
        "gmath12k-g01-b0000000-q000000001",
        "gmath12k-g03-b0000000-q000000003",
        "gmath12k-g07-b0000000-q000000007",
        "gmath12k-g00-b0000001-q000000008",
    ]


def test_rank_balanced_recency_handles_partial_fast_rank_cycles(tmp_path):
    world = 8
    produced = [(batch, rank) for batch in range(2) for rank in range(world)]
    produced += [(batch, rank) for batch in range(2, 5) for rank in (0, 1)]
    for batch, rank in reversed(produced):
        global_chunk = batch * world + rank
        name = f"gmath12k-g{rank:02d}-b{batch:07d}-q{global_chunk:09d}"
        writer = RoundWriter(tmp_path / name)
        tokens, weights = _episode(2, 2, 1.0, base=1000 + global_chunk * 10)
        writer.add(
            tokens,
            weights,
            prompt_len=2,
            reward=1.0,
            weight=1.0,
            policy_version=batch,
            group_id=rank,
        )
        writer.finalize(
            {
                "generator_rank": rank,
                "generator_world": world,
                "global_chunk": global_chunk,
            }
        )

    rounds = load_rounds([tmp_path])
    selected = rank_balanced_recent_round_indices(rounds, window=world)
    assert len(selected) == world
    manifests = [rounds[index].manifest for index in selected]
    assert {manifest["generator_rank"] for manifest in manifests} == set(range(world))
    assert {
        (manifest["generator_rank"], manifest["global_chunk"])
        for manifest in manifests
    } == {(0, 32), (1, 33)} | {(rank, 8 + rank) for rank in range(2, world)}

    rounds[selected[0]].manifest.pop("generator_world")
    assert rank_balanced_recent_round_indices(rounds, window=world) == []


def test_normalize_weights():
    # 2 episodes: weights 4.0 over 10 tokens, 0.5 over 30 tokens.
    scale = normalize_weights(np.array([4.0, 0.5]), np.array([10, 30]))
    # After scaling, token-weighted mean is exactly 1.
    assert np.isclose((scale * 4.0 * 10 + scale * 0.5 * 30) / 40, 1.0)
    with pytest.raises(ValueError):
        normalize_weights(np.array([0.0]), np.array([5]))


def test_max_token_id_stamped(tmp_path):
    d = _write_round(tmp_path, [(2, 3, 1.0, 1.0)], name="r000")
    # _episode uses base 100 -> tokens 100..104; max is 104.
    assert load_round(d).manifest["max_token_id"] == 104


def test_row_fractions():
    # Equal episode lengths -> row fractions equal the token fractions.
    np.testing.assert_allclose(row_fractions([0.5, 0.5], [100, 100]), [0.5, 0.5])
    # Short RL episodes vs long anchor windows: to give the anchor 25% of TOKENS it must
    # be a much smaller fraction of ROWS.
    rf = row_fractions([0.75, 0.25], [77, 1024])
    assert rf[1] < 0.03 and abs(sum(rf) - 1.0) < 1e-9
    # token-mass check: rf weighted by tokens recovers the token fractions.
    mass = [rf[i] * t for i, t in enumerate([77, 1024])]
    assert abs(mass[1] / sum(mass) - 0.25) < 1e-9
    with pytest.raises(ValueError):
        row_fractions([0.5, 0.5], [0, 100])  # nonpositive tokens/ep


def test_midtrain40_row_fraction_respects_proven_holdout_bound():
    assert anchor_row_fraction(
        0.40, 340, 1024, max_row_fraction=0.20
    ) == pytest.approx(0.181236673774)
    with pytest.raises(ValueError, match="exceeds the proven holdout bound"):
        anchor_row_fraction(0.40, 500, 1024, max_row_fraction=0.20)


def test_no_anchor_replay_capacity_accepts_exact_single_pass():
    assert validate_no_anchor_replay(4, 0.5, 2) == 2


def test_no_anchor_replay_capacity_fails_before_second_pass():
    with pytest.raises(
        ValueError,
        match=r"replay the fixed anchor: draws=3 available_episodes=2",
    ):
        validate_no_anchor_replay(6, 0.5, 2)


def test_pack_token_docs():
    docs = [[1, 2, 3], [4, 5], [6, 7, 8, 9, 10, 11]]
    windows = list(pack_token_docs(iter(docs), window_len=5, eos_id=0))
    # Stream with EOS: 1 2 3 0 4 5 0 6 7 8 9 10 11 0  (14 tokens -> 2 full windows)
    assert [w.tolist() for w in windows] == [[1, 2, 3, 0, 4], [5, 0, 6, 7, 8]]
    assert all(w.dtype == np.int32 for w in windows)


def test_mixed_fraction_math():
    """The floor-fraction interleave hits the fraction exactly and indexes both streams densely."""
    f, n = 0.3, 1000
    picks = [int((i + 1) * f) > int(i * f) for i in range(n)]
    assert sum(picks) == int(n * f)
    anchor_idx = [int(i * f) for i in range(n) if picks[i]]
    primary_idx = [i - int(i * f) for i in range(n) if not picks[i]]
    # Each stream sees consecutive local indices 0..k-1 — no gaps, no repeats.
    assert anchor_idx == list(range(len(anchor_idx)))
    assert primary_idx == list(range(len(primary_idx)))


@pytest.mark.skipif(not HAS_TORCH, reason="torch unavailable")
class TestWeightedEpisodeDataset:
    def _dataset(self, tmp_path, seq_length=16, total_samples=64, episodes=None):
        from unified_training.dataset import WeightedEpisodeDataset

        d = _write_round(tmp_path, episodes or [(4, 6, 2.0, 1.0), (3, 2, 0.5, 1.0)])
        rounds = load_rounds([d])
        refs = [(0, e) for e in range(len(rounds[0].index))]
        return WeightedEpisodeDataset(
            rounds, refs, seq_length=seq_length, total_samples=total_samples, seed=7, pad_token_id=0
        )

    def test_shift_alignment(self, tmp_path):
        """labels must be tokens shifted by one and loss_mask aligned with labels."""
        ds = self._dataset(tmp_path, episodes=[(4, 6, 2.0, 1.0)])
        s = ds[0]
        assert s["tokens"].shape == s["labels"].shape == s["loss_mask"].shape == (16,)
        # Within the episode span, labels[t] == tokens[t+1].
        np.testing.assert_array_equal(s["tokens"][1:10].numpy(), s["labels"][:9].numpy())
        # The weight on labels[t] is the weight of the *predicted* token t+1:
        # prompt_len=4 -> label positions 0..2 are prompt predictions (w=0),
        # positions 3..8 predict the 6 response tokens (w=2.0), then pads (w=0).
        w = s["loss_mask"].numpy()
        np.testing.assert_array_equal(w[:3], np.zeros(3))
        np.testing.assert_array_equal(w[3:9], np.full(6, 2.0))
        np.testing.assert_array_equal(w[9:], np.zeros(7))
        assert s["position_ids"].tolist() == list(range(16))

    def test_exact_fit_no_truncation(self, tmp_path):
        # total_len == seq_length + 1 fits exactly: all response predictions keep weight.
        ds = self._dataset(tmp_path, seq_length=9, episodes=[(4, 6, 1.5, 1.0)])
        w = ds[0]["loss_mask"].numpy()
        assert np.count_nonzero(w) == 6

    def test_overlong_is_zero_weighted(self, tmp_path):
        ds = self._dataset(tmp_path, seq_length=8, episodes=[(4, 6, 2.0, 1.0)])
        s = ds[0]
        assert s["loss_mask"].sum().item() == 0.0
        assert s["tokens"].shape == (8,)

    def test_modulo_epochs_cover_pool(self, tmp_path):
        ds = self._dataset(tmp_path, total_samples=8)
        # 2 episodes -> 4 epochs in 8 samples; each epoch is a permutation of the pool.
        first_tokens = [int(ds[i]["tokens"][0]) for i in range(8)]
        for epoch in range(4):
            assert sorted(first_tokens[2 * epoch : 2 * epoch + 2]) == [100, 200]

    def test_deterministic(self, tmp_path):
        ds = self._dataset(tmp_path)
        a, b = ds[5], ds[5]
        assert torch_equal(a, b)


@pytest.mark.skipif(not HAS_TORCH, reason="torch unavailable")
class TestStreamingEpisodeDataset:
    def _stream(self, tmp_path, **kw):
        from unified_training.dataset import StreamingEpisodeDataset

        _write_round(tmp_path, [(4, 6, 1.0, 1.0), (3, 2, 1.0, 1.0)], name="g0-00000")
        return StreamingEpisodeDataset(
            [str(tmp_path)], seq_length=16, total_samples=1000, seed=3, **kw
        )

    def test_policy_age_filter_tracks_the_trainer_step(self, tmp_path):
        from unified_training.dataset import StreamingEpisodeDataset

        world = 2
        for batch, version in ((0, 0), (1, 16), (2, 32)):
            for rank in range(world):
                chunk = batch * world + rank
                w = RoundWriter(tmp_path / f"gmath12k-g{rank:02d}-b{batch:07d}-q{chunk:09d}")
                tokens, weights = _episode(2, 2, 1.0, base=1000 + chunk * 10)
                w.add(tokens, weights, prompt_len=2, reward=1.0, weight=1.0, policy_version=version, group_id=rank)
                w.finalize({"generator_rank": rank, "generator_world": world, "global_chunk": chunk, "policy_version": version})
        ds = StreamingEpisodeDataset([str(tmp_path)], seq_length=16, total_samples=1000, seed=3, max_policy_age_steps=16)
        assert ds.n_episodes == 6 and len(ds._eligible_refs) == 6   # no step yet: no filtering
        ds.set_trainer_step(40)          # cutoff 24 -> only the v32 rounds
        assert ds.policy_age_stats()["usable_rounds"] == 2 and len(ds._eligible_refs) == 2
        ds.set_trainer_step(30)          # cutoff 14 -> v16 and v32 rounds
        assert ds.policy_age_stats()["usable_rounds"] == 4 and len(ds._eligible_refs) == 4
        ds.set_trainer_step(100)         # everything too old: the newest round per rank stays usable
        assert ds.policy_age_stats()["usable_rounds"] == 2 and len(ds._eligible_refs) == 2
        off = StreamingEpisodeDataset([str(tmp_path)], seq_length=16, total_samples=1000, seed=3)
        off.set_trainer_step(100)
        assert len(off._eligible_refs) == 6

    def test_rescan_folds_in_new_rounds(self, tmp_path):
        ds = self._stream(tmp_path, rescan_every=4)
        assert ds.n_episodes == 2
        _ = [ds[i] for i in range(3)]
        _write_round(tmp_path, [(2, 5, 1.0, 1.0)], name="g0-00001")
        assert ds.n_episodes == 2  # not yet rescanned
        _ = [ds[i] for i in range(3, 6)]  # crosses the rescan_every=4 boundary
        assert ds.n_episodes == 3

    def test_unfinalized_rounds_invisible(self, tmp_path):
        ds = self._stream(tmp_path, rescan_every=1)
        (tmp_path / "g0-00001").mkdir()  # in-progress: no DONE
        _ = ds[0]
        assert ds.n_episodes == 2

    def test_recency_retirement_releases_round_payloads(self, tmp_path):
        # Retired rounds must not keep their payload memmaps (two fds each)
        # open, or a long streaming run exhausts the process fd limit.
        ds = self._stream(tmp_path, rescan_every=1, recency_window=1)
        _ = ds[0]
        for r in range(1, 4):
            _write_round(tmp_path, [(2, 4, 1.0, 1.0)], name=f"g0-{r:05d}")
            _ = ds[r]  # every fetch rescans; older rounds retire
        assert len(ds._rounds) == 4
        eligible = {round_index for round_index, _ in ds._eligible_refs}
        for round_index, rnd in enumerate(ds._rounds):
            if round_index not in eligible:
                assert rnd.tokens is None and rnd.weights is None
        _ = ds[100]  # sampling from the surviving window still works

    def test_finalized_hidden_staging_rounds_are_invisible(self, tmp_path):
        ds = self._stream(tmp_path, rescan_every=1)
        _write_round(
            tmp_path,
            [(2, 5, 1.0, 1.0)],
            name=".g0-00001.tmp-123",
        )

        assert ds.rescan() == 0
        assert ds.n_episodes == 2

    def test_rescan_retries_round_when_done_disappears_during_load(
        self, tmp_path, monkeypatch
    ):
        import unified_training.dataset as dataset_module

        ds = self._stream(tmp_path, rescan_every=1)
        round_dir = _write_round(
            tmp_path,
            [(2, 5, 1.0, 1.0)],
            name="g0-00001",
        )
        real_load_round = dataset_module.load_round

        def race_load_round(path):
            if str(path) == str(round_dir):
                (round_dir / "DONE").unlink()
            return real_load_round(path)

        monkeypatch.setattr(dataset_module, "load_round", race_load_round)
        assert ds.rescan() == 0
        assert ds.n_episodes == 2

        monkeypatch.setattr(dataset_module, "load_round", real_load_round)
        (round_dir / "DONE").touch()
        assert ds.rescan() == 1
        assert ds.n_episodes == 3

    def test_initial_stream_scan_ignores_finalized_hidden_staging_rounds(
        self, tmp_path
    ):
        from unified_training.dataset import build_unified_datasets

        _write_round(
            tmp_path,
            [(2, 2, 1.0, 1.0)],
            name="g0-00000",
        )
        _write_round(
            tmp_path,
            [(2, 2, 1.0, 1.0)],
            name=".g0-00001.tmp-123",
        )

        train, _valid, banner = build_unified_datasets(
            episodes_dirs=[str(tmp_path)],
            train_samples=8,
            valid_samples=0,
            sequence_length=16,
            streaming=True,
        )

        assert train.n_episodes == 1
        assert "1 round(s)" in banner

    def test_launch_measure_survives_a_round_arriving_mid_build(
        self, tmp_path, monkeypatch
    ):
        """Regression: a round finalized during the dataset build.

        ``build_unified_datasets`` scans the episode dirs once, then the
        streaming dataset rescans them in its own constructor.  A generator that
        finalizes a round between those two scans leaves the dataset holding
        MORE rounds than the caller -- and its recency-eligible refs then index
        a round the caller's list does not contain.  Measuring the launch RL
        tokens against the caller's list raised
        ``IndexError: list index out of range``.  The measurement must use the rounds the
        refs actually index.
        """

        from unified_training import episodes as episodes_module
        from unified_training.dataset import build_unified_datasets

        _write_round(tmp_path, [(2, 2, 1.0, 1.0)], name="g0-00000")

        # build_unified_datasets imports load_rounds from episodes inside the
        # function body, so the patch must land on the source module.
        real_scan = episodes_module.load_rounds
        state = {"calls": 0}

        def scan_then_publish_a_new_round(dirs, *args, **kwargs):
            """Caller scans first; a new round lands before the dataset scans."""
            state["calls"] += 1
            result = real_scan(dirs, *args, **kwargs)
            if state["calls"] == 1:
                _write_round(tmp_path, [(3, 3, 1.0, 1.0)], name="g0-00001")
            return result

        monkeypatch.setattr(episodes_module, "load_rounds", scan_then_publish_a_new_round)

        train, _valid, _banner = build_unified_datasets(
            episodes_dirs=[str(tmp_path)],
            train_samples=8,
            valid_samples=0,
            sequence_length=16,
            streaming=True,
            recency_window=1,
            anchor_token_fraction=0.0,
        )
        # The dataset saw the later round; the build must not have raised.
        assert train.n_episodes >= 1

    def test_exclude_by_full_path(self, tmp_path):
        import os

        key = (os.path.abspath(str(tmp_path / "g0-00000")), 0)
        ds = self._stream(tmp_path, exclude=frozenset([key]))
        assert ds.n_episodes == 1  # episode 0 held out

    def test_exclude_matches_through_symlink_parent(self, tmp_path):
        import os

        from unified_training.dataset import StreamingEpisodeDataset

        real = tmp_path / "real"
        real.mkdir()
        _write_round(
            real,
            [(4, 6, 1.0, 1.0), (3, 2, 1.0, 1.0)],
            name="g0-00000",
        )
        linked = tmp_path / "linked"
        linked.symlink_to(real, target_is_directory=True)
        key = (os.path.realpath(real / "g0-00000"), 0)
        ds = StreamingEpisodeDataset(
            [str(linked)],
            seq_length=16,
            total_samples=100,
            seed=3,
            exclude=frozenset([key]),
        )
        assert ds.n_episodes == 1

    def test_with_replacement_deterministic_and_valid(self, tmp_path):
        ds = self._stream(tmp_path)
        assert torch_equal(ds[123], ds[123])
        # any index is valid regardless of pool size
        _ = ds[999]

    def test_recency_window_restricts_to_recent_rounds(self, tmp_path):
        from unified_training.dataset import StreamingEpisodeDataset
        from unified_training.episodes import RoundWriter

        # Three rounds with DISTINCT token bases so we can tell which round a draw came from.
        for r in range(3):
            w = RoundWriter(tmp_path / f"g0-{r:05d}")
            toks = np.arange(1000 * (r + 1), 1000 * (r + 1) + 5, dtype=np.int32)
            wt = np.zeros(5, dtype=np.float32); wt[2:] = 1.0
            w.add(toks, wt, prompt_len=2, reward=1.0, weight=1.0, policy_version=r, group_id=0)
            w.finalize()
        ds = StreamingEpisodeDataset(
            [str(tmp_path)], seq_length=16, total_samples=1000, seed=1, recency_window=1
        )
        # window=1 -> only the last round (base 3000) ever drawn.
        firsts = {int(ds[i]["tokens"][0]) for i in range(200)}
        assert firsts == {3000}

    def test_recency_cycle_contains_every_generator_rank(self, tmp_path):
        from unified_training.dataset import StreamingEpisodeDataset
        from unified_training.episodes import RoundWriter

        world = 8
        for batch in range(2):
            for rank in range(world):
                global_chunk = batch * world + rank
                name = (
                    f"gmath12k-g{rank:02d}-b{batch:07d}-q{global_chunk:09d}"
                )
                writer = RoundWriter(tmp_path / name)
                base = 1000 * (batch + 1) + rank * 10
                tokens = np.arange(base, base + 5, dtype=np.int32)
                weights = np.zeros(5, dtype=np.float32)
                weights[2:] = 1.0
                writer.add(
                    tokens,
                    weights,
                    prompt_len=2,
                    reward=1.0,
                    weight=1.0,
                    policy_version=batch,
                    group_id=rank,
                )
                writer.finalize(
                    {
                        "generator_rank": rank,
                        "generator_world": world,
                        "global_chunk": global_chunk,
                    }
                )

        # Unequal producer speeds create three partial extra cycles from ranks
        # 0 and 1. A plain global-q tail would contain those ranks repeatedly;
        # rank-aware recency must still choose exactly one latest round/rank.
        for batch in range(2, 5):
            for rank in (0, 1):
                global_chunk = batch * world + rank
                name = (
                    f"gmath12k-g{rank:02d}-b{batch:07d}-q{global_chunk:09d}"
                )
                writer = RoundWriter(tmp_path / name)
                base = 1000 * (batch + 1) + rank * 10
                tokens = np.arange(base, base + 5, dtype=np.int32)
                weights = np.zeros(5, dtype=np.float32)
                weights[2:] = 1.0
                writer.add(
                    tokens,
                    weights,
                    prompt_len=2,
                    reward=1.0,
                    weight=1.0,
                    policy_version=batch,
                    group_id=rank,
                )
                writer.finalize(
                    {
                        "generator_rank": rank,
                        "generator_world": world,
                        "global_chunk": global_chunk,
                    }
                )

        ds = StreamingEpisodeDataset(
            [str(tmp_path)],
            seq_length=16,
            total_samples=100,
            seed=1,
            recency_window=world,
            without_replacement=True,
        )
        firsts = {int(ds[i]["tokens"][0]) for i in range(world)}
        assert firsts == {5000, 5010} | {2000 + rank * 10 for rank in range(2, world)}

    def test_recency_empty_tail_falls_back(self, tmp_path):
        """A trailing round fully absorbed into the holdout must not crash recency (rng.integers(0))."""
        from unified_training.dataset import StreamingEpisodeDataset
        from unified_training.episodes import RoundWriter

        # round 0: 4 episodes; round 1 (latest): 1 episode -> put it entirely in `exclude`.
        w0 = RoundWriter(tmp_path / "g0-00000")
        for k in range(4):
            t = np.arange(100 + 10 * k, 105 + 10 * k, dtype=np.int32)
            wt = np.zeros(5, dtype=np.float32); wt[2:] = 1.0
            w0.add(t, wt, prompt_len=2, reward=1.0, weight=1.0, policy_version=0, group_id=k)
        w0.finalize()
        w1 = RoundWriter(tmp_path / "g0-00001")
        t = np.arange(3000, 3005, dtype=np.int32)
        wt = np.zeros(5, dtype=np.float32); wt[2:] = 1.0
        w1.add(t, wt, prompt_len=2, reward=1.0, weight=1.0, policy_version=1, group_id=0)
        w1.finalize()
        import os
        exclude = frozenset([(os.path.abspath(str(tmp_path / "g0-00001")), 0)])
        ds = StreamingEpisodeDataset(
            [str(tmp_path)], seq_length=16, total_samples=100, seed=1,
            recency_window=1, exclude=exclude,
        )
        # latest round fully excluded -> recency tail empty -> must fall back to full pool, no crash.
        got = {int(ds[i]["tokens"][0]) for i in range(100)}
        assert got and got.isdisjoint({3000})  # only the non-excluded round-0 episodes

    def test_persisted_validation_refs_survive_stream_growth(self, tmp_path):
        from unified_training.dataset import build_unified_datasets

        _write_round(
            tmp_path,
            [(2, 2, 1.0, 1.0), (2, 2, 1.0, 0.0), (2, 2, 1.0, 1.0)],
            name="g0-00000",
        )
        _write_round(
            tmp_path,
            [(2, 2, 1.0, 1.0), (2, 2, 1.0, 0.0), (2, 2, 1.0, 1.0)],
            name="g0-00001",
        )
        manifest = tmp_path / "valid-holdout.json"
        common = dict(
            episodes_dirs=[str(tmp_path)],
            train_samples=32,
            valid_samples=8,
            sequence_length=16,
            valid_holdout=2,
            split_seed=7,
            streaming=True,
            valid_refs_manifest=manifest,
        )
        train_before, valid_before, _ = build_unified_datasets(
            **common, persist_valid_refs=True
        )
        payload = json.loads(manifest.read_text())
        expected = [
            (record["round"], record["episode_idx"])
            for record in payload["ordered_valid_refs"]
        ]
        assert len(expected) == 2

        _write_round(
            tmp_path,
            [(2, 2, 1.0, 1.0), (2, 2, 1.0, 1.0)],
            name="g0-00002",
        )
        train_after, valid_after, _ = build_unified_datasets(
            **common, persist_valid_refs=False
        )

        def identities(dataset):
            return [
                (dataset._rounds[round_index].round_dir.name, episode_index)
                for round_index, episode_index in dataset._refs
            ]

        assert identities(valid_before) == identities(valid_after) == expected
        for round_name, episode_index in expected:
            absolute = str((tmp_path / round_name).resolve())
            assert (absolute, episode_index) in train_before._exclude
            assert (absolute, episode_index) in train_after._exclude

    def test_persisted_validation_refs_reject_changed_seed(self, tmp_path):
        from unified_training.dataset import build_unified_datasets

        _write_round(
            tmp_path,
            [(2, 2, 1.0, 1.0), (2, 2, 1.0, 0.0)],
            name="g0-00000",
        )
        manifest = tmp_path / "valid-holdout.json"
        common = dict(
            episodes_dirs=[str(tmp_path)],
            train_samples=8,
            valid_samples=4,
            sequence_length=16,
            valid_holdout=1,
            streaming=True,
            valid_refs_manifest=manifest,
        )
        build_unified_datasets(**common, split_seed=7, persist_valid_refs=True)
        with pytest.raises(ValueError, match="seed changed"):
            build_unified_datasets(**common, split_seed=8, persist_valid_refs=False)

    def test_persisted_validation_refs_record_small_pool_effective_size(self, tmp_path):
        from unified_training.dataset import build_unified_datasets

        _write_round(
            tmp_path,
            [(2, 2, 1.0, 1.0), (2, 2, 1.0, 0.0)],
            name="g0-00000",
        )
        manifest = tmp_path / "valid-holdout.json"
        common = dict(
            episodes_dirs=[str(tmp_path)],
            train_samples=8,
            valid_samples=4,
            sequence_length=16,
            valid_holdout=10,
            split_seed=7,
            streaming=True,
            valid_refs_manifest=manifest,
        )
        build_unified_datasets(**common, persist_valid_refs=True)
        payload = json.loads(manifest.read_text())
        assert payload["requested_valid_holdout"] == 10
        assert payload["valid_holdout"] == 1
        assert len(payload["ordered_valid_refs"]) == 1
        build_unified_datasets(**common, persist_valid_refs=False)


@pytest.mark.skipif(not HAS_TORCH, reason="torch unavailable")
class TestMultiStream:
    class _Stub:
        def __init__(self, sid, n_episodes=None):
            self.sid = sid
            if n_episodes is not None:
                self.n_episodes = n_episodes

        def __getitem__(self, idx):
            import torch
            return {"stream": torch.tensor(self.sid)}

    def test_realized_fraction_and_determinism(self):
        from unified_training.dataset import MultiStreamEpisodeDataset

        ds = MultiStreamEpisodeDataset(
            [self._Stub(0), self._Stub(1)],
            token_fraction_fns=[lambda p: 0.0, lambda p: 0.5],  # equal tokens -> 50/50 rows
            tokens_per_ep=[10.0, 10.0], total_samples=4000, seed=5,
        )
        rows = [ds[i] for i in range(4000)]
        picks = [int(row["stream"]) for row in rows]
        frac = sum(picks) / len(picks)
        assert abs(frac - 0.5) < 0.03  # statistical, ~50%
        assert all(bool(row["sampling_filter"]) == (pick == 0) for row, pick in zip(rows, picks))
        assert int(ds[7]["stream"]) == int(ds[7]["stream"])  # deterministic

    def test_two_way_mixer_marks_policy_and_anchor_rows(self):
        from unified_training.dataset import MixedEpisodeDataset

        ds = MixedEpisodeDataset(
            self._Stub(0), self._Stub(1), anchor_fraction=0.5, total_samples=4
        )
        assert ds.anchor_fraction == 0.5
        assert int(ds[0]["stream"]) == 0 and bool(ds[0]["sampling_filter"])
        assert int(ds[1]["stream"]) == 1 and not bool(ds[1]["sampling_filter"])

    def test_no_anchor_replay_guard_rejects_training_shape_before_sampling(self):
        from unified_training.dataset import MixedEpisodeDataset

        with pytest.raises(
            ValueError,
            match=r"replay the fixed anchor: draws=3 available_episodes=2",
        ):
            MixedEpisodeDataset(
                self._Stub(0),
                self._Stub(1, n_episodes=2),
                anchor_fraction=0.5,
                total_samples=6,
                require_no_anchor_replay=True,
            )

    def test_additive_plan_is_exact_sequential_and_replay_checked(self):
        from unified_training.dataset import MixedEpisodeDataset

        ds = MixedEpisodeDataset(
            self._Stub(0), self._Stub(1, n_episodes=1000), anchor_fraction=0.2,
            total_samples=4000, token_fraction=0.05,
        )
        n = ds.additive_rows_per_step(8, "tokens")
        assert n == pytest.approx(8 * 0.2 / 0.8)  # == 2 rows on top of 8
        assert ds.additive_rows_per_step(8, "rows") == pytest.approx(0.4)
        plans = [ds.additive_plan(s, 8, "rows") for s in range(10)]
        assert sum(k for k, _ in plans) == 4  # floor(10 * 0.4)
        assert [b for _, b in plans] == [int(s * 0.4) for s in range(10)]  # sequential anchor indices
        assert ds.validate_additive(8, "tokens") == 1000  # ceil(4000/8)=500 steps * 2
        strict = MixedEpisodeDataset(
            self._Stub(0), self._Stub(1, n_episodes=999), anchor_fraction=0.2,
            total_samples=4000, require_no_anchor_replay=False,
        )
        strict._require_no_replay = True
        with pytest.raises(ValueError, match="replay the fixed anchor"):
            strict.validate_additive(8, "tokens")
        assert bool(ds.primary_row(3)["sampling_filter"]) and not bool(ds.anchor_row(3)["sampling_filter"])

    def test_no_anchor_replay_guard_accepts_exact_single_pass(self):
        from unified_training.dataset import MixedEpisodeDataset

        ds = MixedEpisodeDataset(
            self._Stub(0),
            self._Stub(1, n_episodes=2),
            anchor_fraction=0.5,
            total_samples=4,
            require_no_anchor_replay=True,
        )
        assert ds.anchor_draws == ds.anchor_episodes == 2
        assert [int(ds[index]["stream"]) for index in range(4)] == [0, 1, 0, 1]

    def test_schedule_anneals(self):
        from unified_training.dataset import MultiStreamEpisodeDataset

        # anchor token fraction anneals 0.5 -> 0.0 over the run; late picks are ~all primary.
        ds = MultiStreamEpisodeDataset(
            [self._Stub(0), self._Stub(1)],
            token_fraction_fns=[lambda p: 0.0, lambda p: 0.5 * (1 - p)],
            tokens_per_ep=[10.0, 10.0], total_samples=2000, seed=9,
        )
        early = sum(int(ds[i]["stream"]) for i in range(200)) / 200
        late = sum(int(ds[i]["stream"]) for i in range(1800, 2000)) / 200
        assert early > late and late < 0.1

    def test_scheduled_additive_plan(self):
        from unified_training.dataset import MixedEpisodeDataset, MultiStreamEpisodeDataset

        def mixer(fn, n_streams=2):
            return MultiStreamEpisodeDataset(
                [self._Stub(i) for i in range(n_streams)],
                token_fraction_fns=[lambda p: 0.0] + [fn] * (n_streams - 1),
                tokens_per_ep=[10.0] * n_streams, total_samples=4000, seed=9,
            )

        # a constant schedule reproduces the constant-dose additive rule exactly
        const = mixer(lambda p: 0.2)
        ref = MixedEpisodeDataset(self._Stub(0), self._Stub(1), anchor_fraction=0.2, total_samples=4000)
        assert [const.additive_plan(s, 8, "tokens") for s in range(500)] == [
            ref.additive_plan(s, 8, "tokens") for s in range(500)
        ]
        assert const.validate_additive(8, "tokens") == ref.validate_additive(8, "tokens") == 1000

        # half anneal 0.2 -> 0: sequential anchor indices, nothing after the knee
        half = mixer(lambda p: 0.2 * (1.0 - min(p / 0.5, 1.0)))
        plans = [half.additive_plan(s, 8, "rows") for s in range(520)]
        assert sum(k for k, _ in plans[:500]) == half.validate_additive(8, "rows")
        assert all(b == sum(k for k, _ in plans[:s]) for s, (_, b) in enumerate(plans))
        assert plans[0][0] == 1 and all(k == 0 for k, _ in plans[251:])
        assert half.validate_additive(8, "rows") == 200  # 8 rows/step * mean share 0.05 over 500 steps
        assert mixer(lambda p: 0.2 * (1.0 - min(p / 0.5, 1.0))).additive_plan(137, 8, "rows") == plans[137]
        assert half.additive_rows_per_step(8, "rows") == pytest.approx(0.4, rel=0.01)
        assert bool(half.primary_row(3)["sampling_filter"]) and not bool(half.anchor_row(3)["sampling_filter"])
        with pytest.raises(ValueError, match="exactly one"):
            mixer(lambda p: 0.1, n_streams=3).additive_plan(0, 8, "tokens")

    def test_match_rl_anneal_expected_rows_matches_the_sampler(self):
        """The closed-form RL-row count the trainer uses at a resume tracks the
        sampler's realized RL rows under the solved horizon, and the anchor share
        is zero from the horizon on."""
        import numpy as np
        from unified_training.dataset import MultiStreamEpisodeDataset, as_schedule, match_rl_anneal_horizon
        from unified_training.episodes import row_fractions
        gbs, rl_steps, x0, tpe = 64, 400, 0.05, [300.0, 2048.0]
        r0 = row_fractions([1.0 - x0, x0], tpe)[1]
        T = match_rl_anneal_horizon(rl_steps, r0)
        n = (T + 20) * gbs
        ds = MultiStreamEpisodeDataset([self._Stub(0), self._Stub(1)], token_fraction_fns=[lambda p: 0.0, as_schedule({"start": x0, "end": 0.0, "end_progress": 1.0})],
                                       tokens_per_ep=tpe, total_samples=T * gbs, seed=7)
        ds.match_rl_horizon, ds.launch_row_fraction, ds.rows_per_step = T, r0, gbs
        realized = np.cumsum([int(ds[i]["sampling_filter"]) for i in range(n)])
        for step in (T // 4, T // 2, T, T + 10):
            got, want = int(realized[step * gbs - 1]), ds.expected_rl_rows(step)
            assert abs(got - want) <= 0.02 * step * gbs + 2 * gbs, (step, got, want)
        assert all(int(ds[i]["sampling_filter"]) == 1 for i in range(T * gbs, n))
        assert ds.expected_rl_rows(T) >= rl_steps * gbs

    def test_schedule_end_progress_pins_end_value_mid_run(self):
        from unified_training.dataset import as_schedule

        fn = as_schedule({"start": 0.02, "end": 0.0, "end_progress": 0.5})
        assert fn(0.0) == pytest.approx(0.02)
        assert fn(0.25) == pytest.approx(0.01)
        assert fn(0.5) == pytest.approx(0.0)
        assert fn(0.75) == pytest.approx(0.0)
        assert fn(1.0) == pytest.approx(0.0)
        # without end_progress the historical whole-run ramp is unchanged
        ramp = as_schedule({"start": 0.02, "end": 0.0})
        assert ramp(0.5) == pytest.approx(0.01)
        assert ramp(1.0) == pytest.approx(0.0)
        with pytest.raises(ValueError, match="end_progress"):
            as_schedule({"start": 0.1, "end": 0.0, "end_progress": 0.0})
        with pytest.raises(ValueError, match="end_progress"):
            as_schedule({"start": 0.1, "end": 0.0, "end_progress": 1.5})


@pytest.mark.skipif(not HAS_TORCH, reason="torch unavailable")
def test_vocab_guard(tmp_path):
    from unified_training.dataset import _check_vocab

    d = _write_round(tmp_path, [(2, 3, 1.0, 1.0)], name="r000")  # max token id 104
    rounds = load_rounds([d])
    _check_vocab(rounds, expected_vocab_size=200, label="t")  # ok
    _check_vocab(rounds, expected_vocab_size=0, label="t")  # disabled -> ok
    with pytest.raises(ValueError, match="wrong tokenizer"):
        _check_vocab(rounds, expected_vocab_size=50, label="t")


def torch_equal(a, b):
    import torch

    return all(torch.equal(a[k], b[k]) for k in a)


def test_mean_episode_mass_bases(tmp_path):
    from unified_training.episodes import mean_episode_mass, mean_response_tokens

    # (prompt, resp, weight): a maxRL winner (w=2), a zero-weight wrong answer,
    # and a plain anchor-style row (w=1).
    d = _write_round(tmp_path, [(4, 6, 2.0, 1.0), (3, 2, 0.0, 0.0), (5, 8, 1.0, 1.0)])
    r = load_round(d)
    refs = [(0, 0), (0, 1), (0, 2)]
    # supervised tokens: 6 + 0 + 8; coefficient magnitudes must not matter.
    sup = mean_episode_mass([r], refs, basis="supervised_tokens")
    assert sup == pytest.approx((6 + 0 + 8) / 3)
    assert sup == pytest.approx(mean_response_tokens([r], refs))
    # weight mass: 2*6 + 0 + 1*8 — the loss-numerator share.
    assert mean_episode_mass([r], refs, basis="weight") == pytest.approx(20 / 3)
    assert mean_episode_mass([r], [(0, 0)], basis="weight") == pytest.approx(12.0)
    with pytest.raises(ValueError, match="basis"):
        mean_episode_mass([r], refs, basis="rows")
    with pytest.raises(ValueError, match="empty"):
        mean_episode_mass([r], [], basis="weight")


@pytest.mark.skipif(not HAS_TORCH, reason="torch unavailable")
class TestAdaptiveMixed:
    class _AnchorStub:
        n_episodes = 1000

        def __getitem__(self, idx):
            import torch

            return {"stream": torch.tensor(1)}

    class _StreamStub:
        """Minimal stand-in exposing StreamingEpisodeDataset's adaptive-mix surface."""

        def __init__(self, rounds, refs):
            self._rounds = rounds
            self._eligible_refs = refs
            self._eligible_revision = 1

        def __getitem__(self, idx):
            import torch

            return {"stream": torch.tensor(0)}

    def _primary(self, tmp_path, *, resp_len, weight=1.0, name="g0-00000", n=8):
        d = _write_round(tmp_path, [(2, resp_len, weight, 1.0)] * n, name=name)
        r = load_round(d)
        return self._StreamStub([r], [(0, e) for e in range(n)]), r

    def _mixer(self, primary, **kw):
        from unified_training.dataset import AdaptiveMixedEpisodeDataset

        args = dict(
            token_fraction=0.25, anchor_mass_per_ep=64.0, total_samples=20000, seed=7
        )
        args.update(kw)
        return AdaptiveMixedEpisodeDataset(primary, self._AnchorStub(), **args)

    def test_row_fraction_tracks_pool_mass(self, tmp_path):
        primary, _ = self._primary(tmp_path, resp_len=6)  # 6 supervised tok/ep
        ds = self._mixer(primary)
        r0 = 0.25 * 6 / (64.0 * 0.75 + 0.25 * 6)
        assert ds.anchor_fraction == pytest.approx(r0)
        # The pool grows with longer (better-solved) episodes -> mass rises -> so must r.
        d2 = _write_round(tmp_path, [(2, 18, 1.0, 1.0)] * 8, name="g0-00001")
        primary._rounds.append(load_round(d2))
        primary._eligible_refs = primary._eligible_refs + [(1, e) for e in range(8)]
        primary._eligible_revision += 1
        r1 = 0.25 * 12 / (64.0 * 0.75 + 0.25 * 12)
        assert ds.anchor_fraction == pytest.approx(r1)
        picks = [int(ds[i]["stream"]) for i in range(20000)]
        realized = sum(picks) / len(picks)
        assert abs(realized - r1) < 0.01  # statistical Bernoulli mixing
        assert torch_equal(ds[5], ds[5])  # per-index deterministic at fixed fraction

    def test_additive_rows_follow_the_replace_mode_row_fraction(self, tmp_path):
        primary, _ = self._primary(tmp_path, resp_len=6)
        ds = self._mixer(primary)
        r = ds.anchor_fraction
        # tokens basis: same token share as replace mode, RL rows held at gbs
        assert ds.additive_rows_per_step(512, "tokens") == pytest.approx(512 * r / (1 - r))
        # rows basis: the fraction is a share of the RL row count
        assert ds.additive_rows_per_step(512, "rows") == pytest.approx(0.25 * 512)
        assert ds.additive_rows_cap(512, "rows") == 128
        capped = self._mixer(primary, max_row_fraction=0.01)
        assert capped.additive_rows_cap(512, "tokens") == 6  # ceil(512 * 0.01 / 0.99)
        with pytest.raises(ValueError):
            ds.additive_rows_per_step(512, "episodes")
        counts = [ds.additive_plan(s, 512, "tokens")[0] for s in range(2000)]
        assert abs(sum(counts) / 2000 - 512 * r / (1 - r)) < 0.1
        assert ds.additive_plan(7, 512, "tokens")[1] == 7 * ds.additive_rows_cap(512, "tokens")
        assert bool(ds.primary_row(3)["sampling_filter"]) is True
        assert bool(ds.anchor_row(3)["sampling_filter"]) is False

    def test_response_tokens_basis_counts_wrong_answers_too(self, tmp_path):
        from unified_training.episodes import mean_episode_mass

        # eight episodes of 10 response tokens: four correct (weight 1), four wrong (weight 0)
        d = _write_round(tmp_path, [(2, 10, 1.0, 1.0)] * 4 + [(2, 10, 0.0, 0.0)] * 4, name="g0-00007")
        r = load_round(d); refs = [(0, e) for e in range(8)]
        assert mean_episode_mass([r], refs, basis="supervised_tokens") == pytest.approx(5.0)  # only the correct half
        assert mean_episode_mass([r], refs, basis="response_tokens") == pytest.approx(10.0)  # every response token
        primary = self._StreamStub([r], refs)
        sup = self._mixer(primary, basis="supervised_tokens")
        allt = self._mixer(primary, basis="response_tokens")
        assert sup.anchor_fraction == pytest.approx(0.25 * 5 / (48.0 + 0.25 * 5))
        assert allt.anchor_fraction == pytest.approx(0.25 * 10 / (48.0 + 0.25 * 10))
        assert allt.anchor_fraction > sup.anchor_fraction

    def test_weight_basis_targets_gradient_mass(self, tmp_path):
        primary, _ = self._primary(tmp_path, resp_len=6, weight=2.0)
        sup = self._mixer(primary, basis="supervised_tokens")
        wgt = self._mixer(primary, basis="weight")
        assert sup.anchor_fraction == pytest.approx(0.25 * 6 / (48.0 + 0.25 * 6))
        assert wgt.anchor_fraction == pytest.approx(0.25 * 12 / (48.0 + 0.25 * 12))
        assert wgt.anchor_fraction > sup.anchor_fraction

    def test_clamped_at_row_cap(self, tmp_path):
        primary, _ = self._primary(tmp_path, resp_len=6)
        ds = self._mixer(primary, max_row_fraction=0.01)
        assert ds.anchor_fraction == pytest.approx(0.01)

    def test_zero_mass_window_keeps_previous_fraction(self, tmp_path):
        primary, _ = self._primary(tmp_path, resp_len=6)
        ds = self._mixer(primary)
        before = ds.anchor_fraction
        assert before > 0
        d2 = _write_round(tmp_path, [(2, 6, 0.0, 0.0)] * 8, name="g0-00001")
        primary._rounds = [load_round(d2)]
        primary._eligible_refs = [(0, e) for e in range(8)]
        primary._eligible_revision += 1
        assert ds.anchor_fraction == pytest.approx(before)

    def test_requires_streaming_surface(self):
        with pytest.raises(ValueError, match="streaming primary"):
            self._mixer(object())

    def test_rejects_bad_basis_and_bounds(self, tmp_path):
        primary, _ = self._primary(tmp_path, resp_len=6)
        with pytest.raises(ValueError, match="basis"):
            self._mixer(primary, basis="rows")
        with pytest.raises(ValueError, match="token_fraction"):
            self._mixer(primary, token_fraction=1.5)
        with pytest.raises(ValueError, match="anchor mass"):
            self._mixer(primary, anchor_mass_per_ep=0.0)


@pytest.mark.skipif(not HAS_TORCH, reason="torch unavailable")
class TestOnlineMixBuild:
    def _dirs(self, tmp_path):
        rl = tmp_path / "rl"
        rl.mkdir()
        _write_round(rl, [(2, 4, 1.0, 1.0)] * 4, name="g0-00000")
        anchor = tmp_path / "anchor"
        anchor.mkdir()
        _write_round(anchor, [(1, 8, 1.0, 1.0)] * 4, name="a-00000")
        return rl, anchor

    def _build(self, rl, anchor, **kw):
        from unified_training.dataset import build_unified_datasets

        args = dict(
            episodes_dirs=[str(rl)],
            train_samples=64,
            valid_samples=0,
            sequence_length=16,
            anchor_dirs=[str(anchor)],
            anchor_token_fraction=0.25,
            streaming=True,
        )
        args.update(kw)
        return build_unified_datasets(**args)

    def test_online_build_uses_adaptive_mixer(self, tmp_path):
        from unified_training.dataset import AdaptiveMixedEpisodeDataset

        rl, anchor = self._dirs(tmp_path)
        train, _valid, banner = self._build(rl, anchor, mix_recalc="online")
        assert isinstance(train, AdaptiveMixedEpisodeDataset)
        assert "ONLINE recalc" in banner
        # rl 4 supervised tok/ep vs anchor 8 -> r = .25*4/(8*.75+.25*4)
        assert train.anchor_fraction == pytest.approx(0.25 * 4 / (6.0 + 1.0))

    def test_online_is_the_default(self, tmp_path):
        from unified_training.dataset import AdaptiveMixedEpisodeDataset

        rl, anchor = self._dirs(tmp_path)
        train, _valid, banner = self._build(rl, anchor)  # no mix_recalc given
        assert isinstance(train, AdaptiveMixedEpisodeDataset)
        assert "ONLINE recalc" in banner

    def test_weight_basis_launch_banner(self, tmp_path):
        rl, anchor = self._dirs(tmp_path)
        _train, _valid, banner = self._build(
            rl, anchor, mix_basis="weight", mix_recalc="launch"
        )
        assert "basis=weight" in banner

    def test_online_rejects_fixed_pool(self, tmp_path):
        rl, anchor = self._dirs(tmp_path)
        with pytest.raises(ValueError, match="requires streaming"):
            self._build(rl, anchor, mix_recalc="online", streaming=False)

    def test_online_rejects_no_replay(self, tmp_path):
        rl, anchor = self._dirs(tmp_path)
        with pytest.raises(ValueError, match="no-replay|no_anchor_replay"):
            self._build(rl, anchor, mix_recalc="online", require_no_anchor_replay=True)

    def test_online_rejects_schedules_and_legacy_row_fraction(self, tmp_path):
        rl, anchor = self._dirs(tmp_path)
        with pytest.raises(ValueError, match="one constant"):
            self._build(
                rl, anchor, mix_recalc="online", anchor_token_fraction=0.0,
                streams=[{"dirs": [str(anchor)], "token_fraction": {"start": 0.5, "end": 0.0}}],
            )
        with pytest.raises(ValueError, match="one constant"):
            self._build(
                rl, anchor, mix_recalc="online",
                anchor_token_fraction=0.0, anchor_fraction=0.05,
            )

    def test_invalid_flag_values_rejected(self, tmp_path):
        rl, anchor = self._dirs(tmp_path)
        with pytest.raises(ValueError, match="mix_basis"):
            self._build(rl, anchor, mix_basis="rows")
        with pytest.raises(ValueError, match="mix_recalc"):
            self._build(rl, anchor, mix_recalc="sometimes")


@pytest.mark.skipif(not HAS_TORCH, reason="torch unavailable")
class TestHoldoutReservation:
    """Online mixing must not draw the rows the mid-training holdout owns.

    The mixer addresses the anchor by the GLOBAL training index, which wraps the
    anchor permutation many times over a long run, while the holdout proof
    reserves the tail of the epoch-0 permutation and assumes training consumes
    only the prefix.  Without a clamp the "held-out" LM loss is measured on rows
    the model trained on, and the contamination grows with the anchor dose.
    """

    class _CountingAnchor:
        n_episodes = 1000

        def __init__(self):
            self.seen = []

        def __len__(self):
            # Deliberately NOT the episode count: the real anchor's __len__ is
            # its training sample count, and clamping against it would not
            # exclude the reserved tail.
            return 999_999

        def __getitem__(self, idx):
            import torch

            self.seen.append(int(idx))
            return {"stream": torch.tensor(1)}

    def _mixer(self, tmp_path, anchor, reserved):
        from unified_training.dataset import AdaptiveMixedEpisodeDataset

        helper = TestAdaptiveMixed()
        primary, _ = helper._primary(tmp_path, resp_len=6)
        return AdaptiveMixedEpisodeDataset(
            primary,
            anchor,
            token_fraction=0.5,
            anchor_mass_per_ep=8.0,
            total_samples=20000,
            seed=7,
            reserved_anchor_tail=reserved,
        )

    def test_reserved_tail_is_never_addressed(self, tmp_path):
        anchor = self._CountingAnchor()
        ds = self._mixer(tmp_path, anchor, reserved=512)
        for idx in range(6000):  # wraps the 1000-episode permutation many times
            ds[idx]
        assert anchor.seen, "no anchor rows were drawn; test proves nothing"
        assert max(anchor.seen) < 1000 - 512
        assert min(anchor.seen) >= 0

    def test_without_a_holdout_behaviour_is_unchanged(self, tmp_path):
        anchor = self._CountingAnchor()
        ds = self._mixer(tmp_path, anchor, reserved=0)
        for idx in range(3000):
            ds[idx]
        # Unclamped, the raw global index reaches past the reserved tail.
        assert max(anchor.seen) >= 1000 - 512

    def test_reservation_cannot_exceed_the_anchor(self, tmp_path):
        with pytest.raises(ValueError, match="nothing would remain"):
            self._mixer(tmp_path, self._CountingAnchor(), reserved=1000)

    def test_negative_reservation_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="non-negative"):
            self._mixer(tmp_path, self._CountingAnchor(), reserved=-1)


@pytest.mark.skipif(not HAS_TORCH, reason="torch unavailable")
class TestNwayHoldoutReservation:
    """The schedule mixer must not draw the rows a holdout reserved either.

    ``MultiStreamEpisodeDataset`` addresses a stream by the GLOBAL training
    index, exactly like the adaptive mixer did, so an annealed-dose run would
    train over its own mid-training holdout without this clamp.
    """

    class _Counting:
        n_episodes = 1000

        def __init__(self):
            self.seen = []

        def __len__(self):
            return 999_999  # not the episode count, on purpose

        def __getitem__(self, idx):
            import torch

            self.seen.append(int(idx))
            return {"stream": torch.tensor(1)}

    def _mixer(self, aux, reserved):
        from unified_training.dataset import MultiStreamEpisodeDataset

        primary = self._Counting()
        return MultiStreamEpisodeDataset(
            [primary, aux],
            token_fraction_fns=[lambda p: 0.0, lambda p: 0.5],
            tokens_per_ep=[100.0, 100.0],
            total_samples=20000,
            seed=7,
            reserved_aux_tails=[0, reserved],
        )

    def test_reserved_tail_is_never_addressed(self):
        aux = self._Counting()
        ds = self._mixer(aux, reserved=512)
        for idx in range(6000):
            ds[idx]
        assert aux.seen, "no aux rows drawn; test proves nothing"
        assert max(aux.seen) < 1000 - 512

    def test_without_a_holdout_behaviour_is_unchanged(self):
        aux = self._Counting()
        ds = self._mixer(aux, reserved=0)
        for idx in range(3000):
            ds[idx]
        assert max(aux.seen) >= 1000 - 512

    def test_length_mismatch_is_rejected(self):
        from unified_training.dataset import MultiStreamEpisodeDataset

        with pytest.raises(ValueError, match="length must match"):
            MultiStreamEpisodeDataset(
                [self._Counting(), self._Counting()],
                token_fraction_fns=[lambda p: 0.0, lambda p: 0.5],
                tokens_per_ep=[100.0, 100.0],
                total_samples=100,
                reserved_aux_tails=[0],
            )


@pytest.mark.skipif(not HAS_TORCH, reason="torch unavailable")
class TestScheduleValidation:
    """A schedule that cannot be honoured must fail, not saturate.

    __getitem__ clamps to keep probabilities well formed; without validation a
    typo (negative share, or aux shares summing past 1) would run to completion
    at a dose nobody asked for and report the configured one.
    """

    class _Stub:
        n_episodes = 100

        def __len__(self):
            return 100

        def __getitem__(self, idx):
            import torch

            return {"stream": torch.tensor(0)}

    def _build(self, fn):
        from unified_training.dataset import MultiStreamEpisodeDataset

        return MultiStreamEpisodeDataset(
            [self._Stub(), self._Stub()],
            token_fraction_fns=[lambda p: 0.0, fn],
            tokens_per_ep=[100.0, 200.0],
            total_samples=1000,
        )

    def test_negative_share_is_rejected(self):
        with pytest.raises(ValueError, match="negative"):
            self._build(lambda p: -0.01)

    def test_share_exceeding_one_is_rejected(self):
        with pytest.raises(ValueError, match="no mass for the primary"):
            self._build(lambda p: 1.5)

    def test_invalid_only_mid_schedule_is_still_caught(self):
        # Valid at both endpoints, impossible in between: endpoint-only checking
        # would pass this.
        self._ok = True
        with pytest.raises(ValueError, match="no mass for the primary"):
            self._build(lambda p: 0.05 if p in (0.0, 1.0) else 2.0)

    def test_valid_anneal_is_accepted(self):
        ds = self._build(lambda p: 0.05 + (0.01 - 0.05) * p)
        assert len(ds) == 1000
