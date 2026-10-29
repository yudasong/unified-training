# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Focused tests for the HF policy-loss semantics."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from unified_training.hf_trainer import (
    _micro_rows,
    _stack,
    additive_anchor_count,
    group_rows_by_length,
    loss_denominator,
    row_length,
    trim_length,
    match_rl_projected_steps,
    rank_slots,
    rl_rows_after,
    selected_token_logprobs,
    validation_consumed_samples_at_resume,
    validation_evaluation_count,
    weighted_ce_sums,
)


def test_unfiltered_weighted_loss_matches_cross_entropy():
    logits = torch.tensor([[[2.0, 0.5, -1.0], [0.0, 1.0, 2.0]]], requires_grad=True)
    labels = torch.tensor([[0, 2]])
    weights = torch.tensor([[1.0, 2.0]])

    num, den = weighted_ce_sums(logits, labels, weights, norm="weights", top_p=1.0)
    expected = (F.cross_entropy(logits.view(-1, 3), labels.view(-1), reduction="none") * weights.view(-1)).sum()

    torch.testing.assert_close(num, expected)
    assert den.item() == pytest.approx(3.0)


def test_top_k_then_top_p_matches_truncated_distribution():
    # top_k=3 retains logits [4,3,2]. Nucleus p=.8 then drops logit 2,
    # leaving the distribution softmax([4,3]).
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0, 0.0]])
    labels = torch.tensor([0])
    actual = selected_token_logprobs(logits, labels, top_k=3, top_p=0.8)
    expected = torch.log_softmax(torch.tensor([4.0, 3.0]), dim=0)[0]
    torch.testing.assert_close(actual[0], expected)


def test_temperature_scales_policy_distribution():
    logits = torch.tensor([[2.0, 0.0]])
    labels = torch.tensor([0])

    actual = selected_token_logprobs(logits, labels, top_p=1.0, temperature=2.0)
    expected = torch.log_softmax(torch.tensor([1.0, 0.0]), dim=0)[0]

    torch.testing.assert_close(actual[0], expected)


def test_temperature_changes_top_p_support_before_normalization():
    logits = torch.tensor([[4.0, 3.0, 2.0]])
    labels = torch.tensor([2])

    cold = selected_token_logprobs(logits, labels, top_p=0.8, temperature=1.0)
    warm = selected_token_logprobs(logits, labels, top_p=0.8, temperature=4.0)

    assert cold.item() == 0.0
    assert warm.item() < 0.0


def test_filtered_loss_has_zero_gradient_outside_sampling_support():
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0, 0.0]], requires_grad=True)
    lp = selected_token_logprobs(logits, torch.tensor([0]), top_k=3, top_p=0.8)
    (-lp.sum()).backward()

    assert torch.count_nonzero(logits.grad[0, :2]).item() == 2
    torch.testing.assert_close(logits.grad[0, 2:], torch.zeros(3))


def test_target_outside_recomputed_support_contributes_zero():
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0, 0.0]], requires_grad=True)
    lp, support_mask = selected_token_logprobs(
        logits,
        torch.tensor([2]),
        top_k=3,
        top_p=0.8,
        return_support_mask=True,
    )

    assert lp.item() == 0.0
    assert support_mask.tolist() == [False]
    (-lp.sum()).backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


@pytest.mark.parametrize("norm", ["weights", "tokens", "response_tokens"])
def test_target_outside_recomputed_support_is_removed_from_denominator(norm):
    logits = torch.tensor(
        [[[4.0, 3.0, 0.0], [4.0, 3.0, 0.0]]], requires_grad=True
    )
    labels = torch.tensor([[0, 2]])
    weights = torch.ones(1, 2)
    denom_mask = torch.ones(1, 2)

    num, den = weighted_ce_sums(
        logits,
        labels,
        weights,
        denom_mask,
        norm=norm,
        top_k=2,
        top_p=1.0,
    )

    expected = -torch.log_softmax(logits[0, 0, :2], dim=-1)[0]
    torch.testing.assert_close(num, expected)
    assert den.item() == pytest.approx(1.0)

    (num / den).backward()
    assert torch.count_nonzero(logits.grad[0, 0]).item() == 2
    torch.testing.assert_close(logits.grad[0, 1], torch.zeros(3))


def test_all_zero_weights_keep_response_denominator_and_grad_path():
    logits = torch.randn(1, 2, 5, requires_grad=True)
    labels = torch.tensor([[1, 2]])
    weights = torch.zeros(1, 2)
    denom_mask = torch.ones(1, 2)

    num, den = weighted_ce_sums(
        logits,
        labels,
        weights,
        denom_mask,
        "response_tokens",
        top_k=-1,
        top_p=1.0,
    )
    assert num.item() == 0.0 and den.item() == 2.0
    num.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


def test_zero_weight_out_of_support_response_is_removed_from_denominator():
    logits = torch.tensor([[[4.0, 3.0, 0.0], [4.0, 3.0, 0.0]]], requires_grad=True)
    labels = torch.tensor([[0, 2]])
    weights = torch.zeros(1, 2)
    denom_mask = torch.ones(1, 2)

    num, den = weighted_ce_sums(
        logits,
        labels,
        weights,
        denom_mask,
        "response_tokens",
        top_k=2,
        top_p=1.0,
    )

    assert num.item() == 0.0
    assert den.item() == 1.0
    num.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


def test_sampling_filter_keeps_midtraining_rows_on_full_vocab_ce():
    logits = torch.tensor([[[4.0, 2.0, 0.0]], [[4.0, 2.0, 0.0]]], requires_grad=True)
    labels = torch.tensor([[0], [2]])
    weights = torch.ones(2, 1)

    num, den = weighted_ce_sums(
        logits,
        labels,
        weights,
        top_k=1,
        top_p=1.0,
        sampling_filter=torch.tensor([True, False]),
    )
    expected_midtraining = F.cross_entropy(logits[1], labels[1])
    torch.testing.assert_close(num, expected_midtraining)
    assert den.item() == 2.0

    num.backward()
    torch.testing.assert_close(logits.grad[0], torch.zeros_like(logits.grad[0]))
    assert torch.count_nonzero(logits.grad[1]).item() == 3


def test_chunk_activation_checkpoint_matches_direct_values_and_gradients():
    base = torch.tensor(
        [
            [4.0, 3.0, 2.0, 1.0, 0.0],
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [2.0, 4.0, 1.0, 3.0, 0.0],
        ]
    )
    labels = torch.tensor([0, 4, 3])
    direct_logits = base.clone().requires_grad_(True)
    checkpointed_logits = base.clone().requires_grad_(True)

    direct = selected_token_logprobs(
        direct_logits,
        labels,
        top_k=4,
        top_p=0.9,
        temperature=1.0,
        chunk_size=2,
        activation_checkpoint_chunks=False,
    )
    checkpointed = selected_token_logprobs(
        checkpointed_logits,
        labels,
        top_k=4,
        top_p=0.9,
        temperature=1.0,
        chunk_size=2,
        activation_checkpoint_chunks=True,
    )
    torch.testing.assert_close(checkpointed, direct)

    (-direct.sum()).backward()
    (-checkpointed.sum()).backward()
    torch.testing.assert_close(checkpointed_logits.grad, direct_logits.grad)


@pytest.mark.parametrize(
    ("train_iters", "interval", "expected"),
    [(8000, 128, 64), (481, 1, 482), (481, 128, 5), (8000, 0, 0)],
)
def test_validation_evaluation_count_includes_baseline_and_final(
    train_iters, interval, expected
):
    assert validation_evaluation_count(train_iters, interval) == expected


def test_validation_resume_cursor_skips_completed_validation_slices():
    assert validation_consumed_samples_at_resume(
        480, eval_interval=128, eval_iters=1, gbs=512
    ) == 2048
    assert validation_consumed_samples_at_resume(
        512, eval_interval=128, eval_iters=1, gbs=512
    ) == 2560
    assert validation_consumed_samples_at_resume(
        0, eval_interval=128, eval_iters=1, gbs=512
    ) == 0


@pytest.mark.parametrize("top_p", [0.0, -0.5, 1.1])
def test_invalid_top_p_rejected(top_p):
    with pytest.raises(ValueError, match="top_p"):
        selected_token_logprobs(torch.zeros(1, 4), torch.zeros(1, dtype=torch.long), top_p=top_p)


@pytest.mark.parametrize("temperature", [0.0, -0.5])
def test_invalid_temperature_rejected(temperature):
    with pytest.raises(ValueError, match="temperature"):
        selected_token_logprobs(
            torch.zeros(1, 4),
            torch.zeros(1, dtype=torch.long),
            temperature=temperature,
        )


def test_additive_anchor_count_matches_expectation_and_is_rank_agnostic():
    # floor plus a Bernoulli on the fractional part: the mean over steps is the expectation
    expected = 0.8
    draws = [additive_anchor_count(expected, seed=11, step=s) for s in range(4000)]
    assert set(draws) <= {0, 1}
    assert abs(sum(draws) / len(draws) - expected) < 0.03
    assert [additive_anchor_count(2.3, seed=5, step=s) for s in range(50)] == [
        additive_anchor_count(2.3, seed=5, step=s) for s in range(50)
    ]  # every rank draws the same count for the same (seed, step)
    assert additive_anchor_count(0.0, seed=1, step=3) == 0
    assert additive_anchor_count(3.0, seed=1, step=3) == 3


def test_rank_slots_partition_the_step_with_at_most_one_row_of_imbalance():
    n, world = 11, 4
    slots = [rank_slots(n, world=world, rank=r) for r in range(world)]
    assert sorted(sum(slots, [])) == list(range(n))
    sizes = [len(s) for s in slots]
    assert max(sizes) - min(sizes) <= 1
    assert rank_slots(0, world=4, rank=2) == []


def test_match_rl_anneal_horizon_consumes_the_rl_budget():
    """Linear anneal r(s)=r0(1-s/T): RL rows after T steps equal rl_steps x GBS (within one step)."""
    from unified_training.dataset import match_rl_anneal_horizon
    for r0 in (0.0006, 0.002, 0.01, 0.05):
        T = match_rl_anneal_horizon(12000, r0)
        rows = 512 * (T - r0 * T / 2.0)
        assert 12000 * 512 <= rows < 12000 * 512 + 512 * 1.0001
        assert T > 12000
    assert match_rl_anneal_horizon(12000, 0.0) == 12000
    assert match_rl_anneal_horizon(0, 0.01) == 0
    # general schedules: ramp-up 0 -> r has the same mean as r -> 0; a half-way anneal halves it;
    # a constant schedule reproduces the constant-fraction horizon
    assert match_rl_anneal_horizon(12000, 0.0, 0.02, 1.0) == match_rl_anneal_horizon(12000, 0.02)
    assert match_rl_anneal_horizon(12000, 0.02, 0.0, 0.5) == match_rl_anneal_horizon(12000, 0.01)
    import math
    assert match_rl_anneal_horizon(12000, 0.02, 0.02, 1.0) == math.ceil(12000 / 0.98)


def test_match_rl_projected_steps_reaches_the_rl_row_target_exactly():
    gbs, n = 512, 12000
    for f in (0.0, 5.2e-5, 0.009, 0.09, 0.169):
        s = match_rl_projected_steps(n, gbs, f)
        assert rl_rows_after(s, gbs, f) >= n * gbs
        assert s == n or rl_rows_after(s - 1, gbs, f) < n * gbs  # smallest such step
    assert match_rl_projected_steps(n, gbs, 0.0) == n
    assert match_rl_projected_steps(n, gbs, 0.09) == 13187  # ~ n / (1 - r)
    assert match_rl_projected_steps(0, gbs, 0.1) == 0


def test_publish_and_prune_weights_only_snapshots(tmp_path):
    from unified_training.hf_checkpoint import prune_unpaired_policies, publish_policy_snapshot

    policies, state = tmp_path / "policies", tmp_path / "trainer_hf"
    state.mkdir()
    # a paired checkpoint at 16 and weights-only snapshots at 20 and 24
    for step in (16, 20, 24):
        staged = policies / f".v{step:07d}.pub-1"
        staged.mkdir(parents=True); (staged / "model.safetensors").write_text("w")
        publish_policy_snapshot(staged_policy=staged, final_policy=policies / f"v{step:07d}")
        assert (policies / f"v{step:07d}" / "DONE").exists() and not staged.exists()
    (state / "opt_0000016.pt").write_text("o")
    with pytest.raises(FileExistsError):
        again = policies / ".v0000020.pub-2"; again.mkdir(); (again / "x").write_text("w")
        publish_policy_snapshot(staged_policy=again, final_policy=policies / "v0000020")
    # a resume at 16 drops the future weights-only snapshots but keeps the pair
    assert prune_unpaired_policies(policies, state, 16) == ["v0000020", "v0000024"]
    assert (policies / "v0000016" / "DONE").exists()
    assert prune_unpaired_policies(policies, state, 16) == []


def _row(seq, prompt_len, resp_len, weight=1.0):
    tokens = torch.full((seq,), 7, dtype=torch.int64); tokens[: prompt_len + resp_len] = torch.arange(prompt_len + resp_len) + 10
    loss_mask = torch.zeros(seq); loss_mask[prompt_len - 1 : prompt_len + resp_len - 1] = weight
    denom = torch.zeros(seq); denom[prompt_len - 1 : prompt_len + resp_len - 1] = 1.0
    return {"tokens": tokens, "labels": torch.roll(tokens, -1), "loss_mask": loss_mask, "denom_mask": denom,
            "position_ids": torch.arange(seq), "sampling_filter": torch.tensor(True)}


def test_trimmed_microbatch_keeps_every_supervised_position_and_the_same_loss_sums():
    torch.manual_seed(0)
    rows = [_row(256, 8, 20), _row(256, 30, 100)]
    assert row_length(rows[0]) == 27 and row_length(rows[1]) == 129
    assert trim_length(rows) == 192   # 129 rounded up to a multiple of 64
    full = _stack(rows, "cpu", trim=False); trimmed = _stack(rows, "cpu", trim=True)
    assert trimmed["tokens"].shape == (2, 192) and full["tokens"].shape == (2, 256)
    assert trimmed["sampling_filter"].shape == (2,)
    vocab = 50
    logits_full = torch.randn(2, 256, vocab)
    n_full, d_full = weighted_ce_sums(logits_full, full["labels"] % vocab, full["loss_mask"], full["denom_mask"], norm="weights")
    n_trim, d_trim = weighted_ce_sums(logits_full[:, :192].contiguous(), trimmed["labels"] % vocab, trimmed["loss_mask"], trimmed["denom_mask"], norm="weights")
    assert torch.allclose(n_full, n_trim) and torch.allclose(d_full, d_trim)


def test_trimming_keeps_zero_weight_response_tokens_in_the_response_token_denominator():
    """Production emits wrong completions with weight 0 but counts their response
    tokens in the loss denominator (UNIFIED_LOSS_NORM=response_tokens). A zero-weight
    row must therefore be trimmed to the end of its response, not to its last
    weighted position (which does not exist); trimming to the last weighted
    position would shrink the denominator and inflate the loss."""
    torch.manual_seed(1)
    rows = [_row(512, 100, 150, weight=0.0), _row(512, 8, 20, weight=1.0)]   # wrong row longer than the correct one
    assert row_length(rows[0]) == 249        # end of the wrong response, not 1
    assert row_length(rows[1]) == 27
    assert trim_length(rows) == 256
    full = _stack(rows, "cpu", trim=False); trimmed = _stack(rows, "cpu", trim=True)
    vocab = 50
    logits_full = torch.randn(2, 512, vocab)
    n_full, d_full = weighted_ce_sums(logits_full, full["labels"] % vocab, full["loss_mask"], full["denom_mask"], norm="response_tokens")
    n_trim, d_trim = weighted_ce_sums(logits_full[:, :256].contiguous(), trimmed["labels"] % vocab, trimmed["loss_mask"], trimmed["denom_mask"], norm="response_tokens")
    assert torch.allclose(n_full, n_trim) and torch.allclose(d_full, d_trim)
    assert float(d_full) == 170.0            # 150 wrong + 20 correct response tokens, all in the denominator
    # length grouping still clusters by the new length
    groups = group_rows_by_length(rows + [_row(512, 4, 5, weight=0.0)], 2)
    assert [[row_length(r) for r in g] for g in groups] == [[8, 27], [249]]


def test_group_rows_by_length_is_a_permutation_into_similar_lengths():
    rows = [_row(128, 4, n) for n in (60, 5, 30, 90, 10, 45)]
    groups = group_rows_by_length(rows, 2)
    assert sorted(id(r) for g in groups for r in g) == sorted(id(r) for r in rows)
    lengths = [[row_length(r) for r in g] for g in groups]
    assert lengths == [[8, 13], [33, 48], [63, 93]]


@pytest.mark.parametrize("denominator", ["absent", "empty"])
def test_trimming_keeps_signed_weights_without_response_denominator(denominator):
    row = _row(256, 96, 100, weight=-0.5)
    if denominator == "absent":
        row.pop("denom_mask")
    else:
        row["denom_mask"].zero_()
    assert row_length(row) == 195
    batch = _stack([row], "cpu", trim=True)
    assert batch["tokens"].shape == (1, 256)
    torch.testing.assert_close(batch["loss_mask"].sum(), torch.tensor(-50.0))


def _mixed_trim_rows(case):
    rows = [
        _row(256, 8, 20, 1.0),
        _row(256, 96, 100, 0.0),
        _row(256, 6, 90, 1.0),
        _row(256, 96, 100, -0.25),
        _row(256, 4, 5, 0.0),
        _row(256, 64, 160, 0.5),
        _row(256, 32, 100, -0.25),
        _row(256, 96, 128, 0.0),
    ]
    rows[2]["sampling_filter"] = torch.tensor(False)  # full-vocabulary anchor CE
    rows[3]["denom_mask"].zero_()  # signed advantages remain live independently
    rows[4]["denom_mask"].zero_()  # completely inactive row
    if case == "outside_support":
        for row in rows:
            row["sampling_filter"] = torch.tensor(True)
            row["labels"].fill_(4)
    elif case != "mixed":
        for row in rows:
            row["loss_mask"].zero_()
            if case == "empty":
                row["denom_mask"].zero_()
    return rows


@pytest.mark.parametrize("norm", ["weights", "tokens", "response_tokens"])
@pytest.mark.parametrize("filtered", [False, True])
@pytest.mark.parametrize("case", ["mixed", "zero_weight", "empty", "outside_support"])
def test_grouped_trimmed_ddp_accumulation_matches_full_batch_gradient(norm, filtered, case):
    """Exercise the trainer's MBS=2/world=2 split and provisional/exact rescale.

    Every rank has four primary rows; grouping changes each rank's microbatch
    membership. A shared trainable logits table makes gradients from different
    rows accumulate on the same parameters, as in an actual model.
    """
    rows = _mixed_trim_rows(case)
    world, mbs, vocab = 2, 2, 5
    initial = torch.tensor([3.0, 2.0, 1.0, 0.0, -1.0]).repeat(vocab, 1)
    top_k, top_p = (3, 0.8) if filtered else (-1, 1.0)

    def sums(parameter, micro_rows, trim):
        batch = _stack(micro_rows, "cpu", trim=trim)
        return weighted_ce_sums(
            parameter[batch["tokens"] % vocab],
            batch["labels"] % vocab,
            batch["loss_mask"],
            batch["denom_mask"],
            norm=norm, top_k=top_k, top_p=top_p,
            sampling_filter=batch["sampling_filter"],
        )

    reference = initial.clone().requires_grad_()
    expected_num, expected_den = sums(reference, rows, False)
    (expected_num / expected_den.clamp(min=1)).backward()
    provisional = torch.stack([
        loss_denominator(row["loss_mask"], norm, row["denom_mask"])
        for row in rows
    ]).sum().clamp(min=1)

    for trim in (False, True):
        total_num, total_den = torch.tensor(0.0), torch.tensor(0.0)
        rank_gradients = []
        for rank in range(world):
            parameter = initial.clone().requires_grad_()
            micros = [_micro_rows(rows, base=0, micro=micro, world=world, rank=rank, mbs=mbs)
                      for micro in range(2)]
            if trim:
                micros = group_rows_by_length([row for micro in micros for row in micro], mbs)
            for micro in micros:
                num, den = sums(parameter, micro, trim)
                (num * world / provisional).backward()
                total_num += num.detach()
                total_den += den.detach()
            rank_gradients.append(parameter.grad)
        gradient = torch.stack(rank_gradients).mean(0)
        gradient *= provisional / total_den.clamp(min=1)
        torch.testing.assert_close(total_num, expected_num.detach(), atol=2e-4, rtol=2e-6)
        torch.testing.assert_close(total_den, expected_den.detach())
        torch.testing.assert_close(gradient, reference.grad, atol=2e-6, rtol=2e-5)
        assert torch.isfinite(gradient).all()
    if filtered and case == "mixed":
        assert expected_den < provisional  # the support correction was exercised
    if case in ("zero_weight", "empty") or (filtered and case == "outside_support"):
        torch.testing.assert_close(reference.grad, torch.zeros_like(reference.grad))
    if filtered and case == "outside_support":
        assert expected_den == 0 and provisional > 1
