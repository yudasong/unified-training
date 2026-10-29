# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for the shared rollout logic and the anchor token-fraction math (numpy-only)."""

from __future__ import annotations

import numpy as np
import pytest

from unified_training.episodes import anchor_row_fraction, load_round
from unified_training.rollout_common import (
    extract_answer,
    response_token_budget,
    select_prompt_shard,
    select_prompt_slice,
    weight_groups,
    write_episode_round,
)


def test_extract_answer():
    assert extract_answer("blah blah #### 42") == "42"
    assert extract_answer("#### 1,234") == "1234"
    assert extract_answer("the result is 7.") == "7"
    assert extract_answer("first 3 then 9") == "9"
    assert extract_answer("no numbers here") == ""


class TestSelectPromptSlice:
    def test_rotation_covers_each_pass(self):
        seen = []
        offset = 0
        while offset < 20:  # two passes over 10 prompts in chunks of 4
            p, idxs = select_prompt_slice(10, offset, 4)
            seen.append((p, idxs))
            offset += len(idxs)
        pass0 = [i for p, idxs in seen if p == 0 for i in idxs]
        pass1 = [i for p, idxs in seen if p == 1 for i in idxs]
        assert sorted(pass0) == list(range(10)) and sorted(pass1) == list(range(10))
        assert pass0 != pass1  # different shuffle per pass

    def test_no_wrap_inside_batch(self):
        # offset 8 of 10 with chunk 4 -> truncated to 2, next batch starts pass 1
        _, idxs = select_prompt_slice(10, 8, 4)
        assert len(idxs) == 2
        p, idxs2 = select_prompt_slice(10, 10, 4)
        assert p == 1 and len(idxs2) == 4

    def test_deterministic(self):
        assert select_prompt_slice(100, 37, 8) == select_prompt_slice(100, 37, 8)


class TestSelectPromptShard:
    def test_math12k_two_passes_are_exact_and_disjoint(self):
        n_prompts, limit, world = 12_000, 1_024, 8
        chunks_by_pass = {0: [], 1: []}

        # 12 chunks per pass, so three local batches across eight ranks cover
        # exactly two passes (24 global chunks).
        for local_batch in range(3):
            for rank in range(world):
                pass_num, indices = select_prompt_shard(
                    n_prompts, local_batch, limit, rank, world
                )
                chunks_by_pass[pass_num].append(indices)

        for chunks in chunks_by_pass.values():
            assert sorted(len(chunk) for chunk in chunks) == [736] + [1_024] * 11
            for i, left in enumerate(chunks):
                for right in chunks[i + 1 :]:
                    assert set(left).isdisjoint(right)
            assert sorted(index for chunk in chunks for index in chunk) == list(range(n_prompts))

        pass0 = [index for chunk in chunks_by_pass[0] for index in chunk]
        pass1 = [index for chunk in chunks_by_pass[1] for index in chunk]
        assert pass0 != pass1

    def test_effective_math12k_shape_covers_two_passes(self):
        n_prompts, limit, world = 11_981, 128, 8
        chunks_by_pass = {0: [], 1: []}
        for local_batch in range(24):
            for rank in range(world):
                pass_num, indices = select_prompt_shard(
                    n_prompts, local_batch, limit, rank, world
                )
                if pass_num < 2:
                    chunks_by_pass[pass_num].append(indices)

        for chunks in chunks_by_pass.values():
            assert len(chunks) == 94
            assert sorted(map(len, chunks)) == [77] + [128] * 93
            flattened = [index for chunk in chunks for index in chunk]
            assert len(flattened) == len(set(flattened)) == n_prompts
            assert sorted(flattened) == list(range(n_prompts))

    def test_restart_is_deterministic(self):
        args = (12_000, 1, 1_024, 3, 8)
        first = select_prompt_shard(*args)
        assert first == select_prompt_shard(*args)
        assert first[0] == 0 and len(first[1]) == 736

    def test_small_odd_input(self):
        chunks_by_pass = {0: [], 1: []}
        for local_batch in range(2):
            for rank in range(4):
                pass_num, indices = select_prompt_shard(7, local_batch, 3, rank, 4)
                if pass_num < 2:
                    chunks_by_pass[pass_num].append(indices)

        for chunks in chunks_by_pass.values():
            assert sorted(map(len, chunks)) == [1, 3, 3]
            assert sorted(index for chunk in chunks for index in chunk) == list(range(7))

    @pytest.mark.parametrize(
        ("args", "message"),
        [
            ((0, 0, 1, 0, 1), "n_prompts"),
            ((1, -1, 1, 0, 1), "local_batch"),
            ((1, 0, 0, 0, 1), "limit"),
            ((1, 0, 1, 0, 0), "world"),
            ((1, 0, 1, -1, 2), "rank"),
            ((1, 0, 1, 2, 2), "rank"),
        ],
    )
    def test_validation(self, args, message):
        with pytest.raises(ValueError, match=message):
            select_prompt_shard(*args)


@pytest.mark.parametrize(
    ("prompt_len", "expected"),
    [(64, 320), (704, 320), (858, 166), (1_023, 1), (1_024, 0), (1_948, 0)],
)
def test_response_token_budget(prompt_len, expected):
    assert response_token_budget(prompt_len, 1_024, 320) == expected


@pytest.mark.parametrize(
    "args", [(-1, 1_024, 320), (1, 0, 320), (1, 1_024, 0)]
)
def test_response_token_budget_rejects_invalid_dimensions(args):
    with pytest.raises(ValueError):
        response_token_budget(*args)


def _comps(*specs):
    """spec: (correct, finished) -> completion tuple with dummy tokens/text."""
    return [([10, 11], f"t{i} #### 5", fin, cor) for i, (cor, fin) in enumerate(specs)]


class TestWeightGroups:
    def test_all_wrong_dropped(self):
        eps, st = weight_groups(
            [(0, [1, 2], _comps((False, True), (False, True)))],
            wmax=8.0, keep_allcorrect=True, hard_filter=False,
        )
        assert eps == [] and st.groups_all_wrong == 1 and st.solved == 0

    def test_truncated_never_trainable(self):
        eps, st = weight_groups(
            [(0, [1, 2], _comps((True, False), (True, True)))],
            wmax=8.0, keep_allcorrect=True, hard_filter=False,
        )
        # truncated correct one dropped; remaining group is all-correct p=1 -> w=1
        assert st.n_truncated == 1 and len(eps) == 1 and eps[0].weight == 1.0

    def test_weight_is_inverse_phat_clamped(self):
        eps, st = weight_groups(
            [(0, [1, 2], _comps((True, True), *[(False, True)] * 7))],
            wmax=8.0, keep_allcorrect=True, hard_filter=False,
        )
        assert len(eps) == 1 and eps[0].weight == pytest.approx(8.0)  # 1/(1/8) hits wmax exactly
        eps, _ = weight_groups(
            [(0, [1, 2], _comps((True, True), (True, True), (False, True), (False, True)))],
            wmax=8.0, keep_allcorrect=True, hard_filter=False,
        )
        assert len(eps) == 2 and all(e.weight == pytest.approx(2.0) for e in eps)

    def test_hard_filter_and_allcorrect_knob(self):
        groups = [(0, [1], _comps((True, True), (False, True)))]
        eps, _ = weight_groups(groups, wmax=8.0, keep_allcorrect=True, hard_filter=True)
        assert eps[0].weight == 1.0
        eps, st = weight_groups(
            [(0, [1], _comps((True, True), (True, True)))],
            wmax=8.0, keep_allcorrect=False, hard_filter=False,
        )
        assert eps == [] and st.groups_allcorrect_dropped == 1

    def test_reinforce_weights_have_no_inverse_phat(self):
        # UNIFIED_HARD_FILTER=1 is the REINFORCE objective: reward 1 for a correct sample,
        # 0 for a wrong one, no 1/p_hat factor and nothing for wmax to clip.
        groups = [(0, [1], _comps((True, True), (False, True), (False, True), (False, True))),
                  (1, [1], _comps((True, True), (True, True), (True, True), (False, True)))]
        eps, st = weight_groups(groups, wmax=8.0, keep_allcorrect=True, hard_filter=True,
                                emit_zero_weight=True)
        assert [e.weight for e in eps] == [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]
        assert [e.reward for e in eps] == [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]
        assert st.n_wmax_clipped_groups == 0
        maxrl, _ = weight_groups(groups, wmax=8.0, keep_allcorrect=True, hard_filter=False,
                                 emit_zero_weight=True)
        assert [e.weight for e in maxrl][:4] == [4.0, 0.0, 0.0, 0.0]

    def test_episode_assembly(self):
        eps, _ = weight_groups(
            [(7, [1, 2, 3], [([20, 21], "x #### 5", True, True)])],
            wmax=8.0, keep_allcorrect=True, hard_filter=False,
        )
        assert eps[0].tokens == [1, 2, 3, 20, 21]
        assert eps[0].prompt_len == 3 and eps[0].group_id == 7

    def test_emit_zero_weight_keeps_wrong_denominator_rows(self):
        eps, st = weight_groups(
            [(0, [1, 2], _comps((True, True), (False, True), (False, True), (False, True)))],
            wmax=8.0,
            keep_allcorrect=True,
            hard_filter=False,
            emit_zero_weight=True,
        )
        assert st.solved == 1 and st.groups_all_wrong == 0
        assert [e.reward for e in eps] == [1.0, 0.0, 0.0, 0.0]
        assert [e.weight for e in eps] == [pytest.approx(4.0), 0.0, 0.0, 0.0]

    def test_emit_zero_weight_keeps_all_wrong_rows(self):
        eps, st = weight_groups(
            [(0, [1, 2], _comps((False, True), (False, True)))],
            wmax=8.0,
            keep_allcorrect=True,
            hard_filter=False,
            emit_zero_weight=True,
        )
        assert st.groups_all_wrong == 1 and st.solved == 0
        assert len(eps) == 2
        assert all(e.weight == 0.0 and e.reward == 0.0 for e in eps)

    def test_drop_all_wrong_overrides_zero_weight_emission(self):
        groups = [(0, [1, 2], _comps((False, True), (False, True))),
                  (1, [1, 2], _comps((True, True), (False, True)))]
        eps, st = weight_groups(groups, wmax=8.0, keep_allcorrect=True, hard_filter=False,
                                emit_zero_weight=True, drop_all_wrong=True)
        assert st.groups_all_wrong == 1 and st.solved == 1
        # the all-wrong group is gone; the mixed group keeps its zero-weight wrong row
        assert [e.group_id for e in eps] == [1, 1]
        assert [e.weight for e in eps] == [pytest.approx(2.0), 0.0]

    def test_truncated_scored_normally_when_kept_for_parity(self):
        eps, st = weight_groups(
            [(0, [1, 2], _comps((True, False), (False, True)))],
            wmax=8.0,
            keep_allcorrect=True,
            hard_filter=False,
            emit_zero_weight=True,
            drop_truncated=False,
        )
        assert st.n_truncated == 1
        assert [e.reward for e in eps] == [1.0, 0.0]
        assert [e.weight for e in eps] == [pytest.approx(2.0), 0.0]

    def test_truncated_can_be_forced_incorrect(self):
        eps, st = weight_groups(
            [(0, [1, 2], _comps((True, False), (False, True)))],
            wmax=8.0,
            keep_allcorrect=True,
            hard_filter=False,
            emit_zero_weight=True,
            drop_truncated=False,
            force_truncated_incorrect=True,
        )
        assert st.n_truncated == 1 and st.groups_all_wrong == 1
        assert [e.reward for e in eps] == [0.0, 0.0]
        assert [e.weight for e in eps] == [0.0, 0.0]


def test_write_episode_round_normalizes(tmp_path):
    from unified_training.rollout_common import Episode

    eps = [
        Episode(tokens=[1, 2, 3, 4], prompt_len=2, weight=4.0, group_id=0),
        Episode(tokens=[5, 6, 7, 8, 9, 10], prompt_len=2, weight=1.0, group_id=1),
    ]
    m = write_episode_round(
        tmp_path / "r", eps, policy_version=3, hard_filter=False, manifest_extra={"k": "v"}
    )
    r = load_round(tmp_path / "r")
    r.ensure_loaded()  # payload memmaps are lazy
    # token-weighted mean of response weights must be exactly 1 after scaling
    w = np.asarray(r.weights)
    assert float(w[w > 0].mean()) == pytest.approx(1.0)
    assert m["k"] == "v" and m["n_episodes"] == 2
    assert int(r.index["policy_version"][0]) == 3


def test_write_episode_round_can_keep_raw_maxrl_weights(tmp_path):
    from unified_training.rollout_common import Episode

    eps = [
        Episode(tokens=[1, 2, 3, 4], prompt_len=2, weight=4.0, group_id=0, reward=1.0),
        Episode(tokens=[5, 6, 7, 8, 9, 10], prompt_len=2, weight=0.0, group_id=1, reward=0.0),
    ]
    m = write_episode_round(
        tmp_path / "r",
        eps,
        policy_version=3,
        hard_filter=False,
        normalize_episode_weights=False,
        manifest_extra={"k": "v"},
    )
    r = load_round(tmp_path / "r")
    r.ensure_loaded()  # payload memmaps are lazy
    np.testing.assert_array_equal(np.asarray(r.weights[:4]), np.array([0.0, 0.0, 4.0, 4.0], dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(r.weights[4:]), np.zeros(6, dtype=np.float32))
    assert m["weight_scale"] == pytest.approx(1.0)
    assert m["normalize_episode_weights"] is False
    assert float(r.index["reward"][0]) == pytest.approx(1.0)
    assert float(r.index["reward"][1]) == pytest.approx(0.0)


def test_anchor_row_fraction():
    # The E2 numbers: f_tok=0.25 with 77-token episodes vs 1024-token windows -> ~2.4% rows.
    r = anchor_row_fraction(0.25, 77.0, 1024.0)
    assert r == pytest.approx(0.02445, abs=1e-4)
    # Recover the token fraction from the row fraction.
    f = r * 1024 / (r * 1024 + (1 - r) * 77)
    assert f == pytest.approx(0.25, abs=1e-6)
    # Equal lengths -> row fraction == token fraction.
    assert anchor_row_fraction(0.3, 100.0, 100.0) == pytest.approx(0.3)
    with pytest.raises(ValueError):
        anchor_row_fraction(1.5, 77.0, 1024.0)


class TestEasyGroupFilter:
    """max_group_acc drops (mostly-)solved groups; easy_keep_frac retains a hash-stable slice."""

    def test_disabled_by_default(self):
        eps, st = weight_groups(
            [(0, [1], _comps((True, True), (True, True)))],
            wmax=8.0, keep_allcorrect=True, hard_filter=False,
        )
        assert len(eps) == 2 and st.groups_easy_dropped == 0 and st.groups_easy_kept == 0

    def test_threshold_one_drops_fully_solved_only(self):
        groups = [
            (0, [1], _comps((True, True), (True, True))),            # p=1.0 -> dropped
            (1, [1], _comps((True, True), (False, True))),           # p=0.5 -> kept
        ]
        eps, st = weight_groups(
            groups, wmax=8.0, keep_allcorrect=True, hard_filter=False,
            max_group_acc=1.0, easy_keep_frac=0.0,
        )
        assert st.groups_easy_dropped == 1
        assert {e.group_id for e in eps} == {1}

    def test_threshold_mostly_solved(self):
        # p = 0.875 (7/8) meets a 0.875 threshold; p = 0.75 does not
        g_hi = (0, [1], _comps(*([(True, True)] * 7 + [(False, True)])))
        g_lo = (1, [1], _comps(*([(True, True)] * 6 + [(False, True)] * 2)))
        eps, st = weight_groups(
            [g_hi, g_lo], wmax=8.0, keep_allcorrect=True, hard_filter=False,
            max_group_acc=0.875, easy_keep_frac=0.0,
        )
        assert st.groups_easy_dropped == 1
        assert {e.group_id for e in eps} == {1}

    def test_keep_frac_is_deterministic_and_proportional(self):
        groups = [
            (gid, [1], _comps((True, True), (True, True))) for gid in range(2000)
        ]
        eps1, st1 = weight_groups(
            list(groups), wmax=8.0, keep_allcorrect=True, hard_filter=False,
            max_group_acc=1.0, easy_keep_frac=0.25,
        )
        eps2, st2 = weight_groups(
            list(reversed(groups)), wmax=8.0, keep_allcorrect=True, hard_filter=False,
            max_group_acc=1.0, easy_keep_frac=0.25,
        )
        # same groups survive regardless of order
        assert {e.group_id for e in eps1} == {e.group_id for e in eps2}
        assert st1.groups_easy_kept == st2.groups_easy_kept
        # keep fraction lands near 25%
        assert 0.20 < st1.groups_easy_kept / 2000 < 0.30

    def test_kept_easy_still_respects_keep_allcorrect(self):
        eps, st = weight_groups(
            [(0, [1], _comps((True, True), (True, True)))],
            wmax=8.0, keep_allcorrect=False, hard_filter=False,
            max_group_acc=1.0, easy_keep_frac=1.0,
        )
        assert eps == [] and st.groups_easy_kept == 1 and st.groups_allcorrect_dropped == 1

    def test_invalid_params_raise(self):
        with pytest.raises(ValueError):
            weight_groups([], wmax=8.0, keep_allcorrect=True, hard_filter=False, easy_keep_frac=1.5)
        with pytest.raises(ValueError):
            weight_groups([], wmax=8.0, keep_allcorrect=True, hard_filter=False, max_group_acc=0.0)
