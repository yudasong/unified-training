# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Reasoning-puzzle knobs in rollout_common: prompt templates, the maxRL baseline, the REQUIRE_EOS rule."""

from __future__ import annotations

import pytest

from unified_training.rollout_common import (
    CHAT_TEMPLATE,
    prompt_template,
    truncated_without_eos,
    weight_groups,
)

EOS = 50256
QUESTION = (
    "Mia, Noah and Ella live on different floors. The occupied floors are {5, 9, 10}.\n"
    'Answer with a comma-separated list like "Mia=NUMBER, ...".\n\nAnswer: '
)
BASELINE = dict(wmax=8.0, keep_allcorrect=True, hard_filter=False, emit_zero_weight=True, baseline=True)


def test_raw_template_is_the_question_verbatim_with_an_eos_only_stop():
    fmt, stop = prompt_template("raw")
    assert fmt.format(question=QUESTION) == QUESTION
    assert stop == []


def test_chatml_template_is_unchanged():
    fmt, stop = prompt_template("chatml")
    assert fmt == CHAT_TEMPLATE == "<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    assert fmt.format(question="What is 2+2?") == "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
    assert stop == ["<|im_end|>"]


def test_prompt_template_returns_a_fresh_stop_list_and_rejects_unknown_names():
    prompt_template("chatml")[1].append("<|endoftext|>")
    assert prompt_template("chatml")[1] == ["<|im_end|>"]
    with pytest.raises(ValueError, match="unknown prompt template"):
        prompt_template("llama3")


def _group(group_id, outcomes, finished=None):
    finished = finished or [True] * len(outcomes)
    completions = [([10 + i, 11], f"c{i}", fin, cor) for i, (cor, fin) in enumerate(zip(outcomes, finished))]
    return group_id, [1, 2], completions


def _closed_form(k, n, eps=1e-6):
    p = k / n
    return (1.0 - p) / (p + eps), -p / (p + eps)


@pytest.mark.parametrize(("k", "n"), [(1, 4), (2, 4), (3, 4), (1, 16), (5, 16), (15, 16)])
def test_baseline_weights_follow_the_closed_form(k, n):
    eps, stats = weight_groups([_group(0, [True] * k + [False] * (n - k))], **BASELINE)
    correct, wrong = _closed_form(k, n)
    assert [e.weight for e in eps] == pytest.approx([correct] * k + [wrong] * (n - k), rel=1e-6)
    assert [e.reward for e in eps] == [1.0] * k + [0.0] * (n - k)
    # correct -> (1-p)/p and wrong -> -1 up to eps; the baseline makes each group sum to zero
    assert correct == pytest.approx((n - k) / k, rel=1e-4) and wrong == pytest.approx(-1.0, rel=1e-4)
    assert sum(e.weight for e in eps) == pytest.approx(0.0, abs=1e-5)
    assert stats.n_wmax_clipped_groups == 0  # 1/16 -> weight 15 is not clipped at wmax 8


def test_baseline_all_correct_and_all_wrong_groups_get_zero_weight():
    eps, stats = weight_groups([_group(0, [True] * 4), _group(1, [False] * 4)], **BASELINE)
    assert [e.weight for e in eps] == [0.0] * 8
    assert [e.reward for e in eps] == [1.0] * 4 + [0.0] * 4
    assert stats.solved == 1 and stats.groups_all_wrong == 1


def test_baseline_eps_enters_the_denominator():
    eps, _ = weight_groups([_group(0, [True, False, False, False])], **{**BASELINE, "baseline_eps": 0.5})
    assert [e.weight for e in eps] == pytest.approx([0.75 / 0.75, -0.25 / 0.75, -0.25 / 0.75, -0.25 / 0.75])
    for bad in (0.0, -1e-6):
        with pytest.raises(ValueError, match="baseline_eps must be positive"):
            weight_groups([], **{**BASELINE, "baseline_eps": bad})


@pytest.mark.parametrize("override", [{"hard_filter": True}, {"emit_zero_weight": False}, {"weight_eps": 0.1}])
def test_baseline_rejects_incompatible_knobs(override):
    with pytest.raises(ValueError, match="baseline maxRL needs"):
        weight_groups([], **{**BASELINE, **override})


def test_baseline_off_keeps_the_inverse_phat_weights():
    groups = [
        _group(0, [True, False, False, False]),
        _group(1, [True, True, False, False]),
        _group(2, [True, True]),
        _group(3, [False, False]),
    ]
    kwargs = dict(wmax=8.0, keep_allcorrect=True, hard_filter=False, emit_zero_weight=True)
    default, default_stats = weight_groups(groups, **kwargs)
    explicit, explicit_stats = weight_groups(groups, **kwargs, baseline=False, baseline_eps=0.5)
    assert default == explicit and default_stats == explicit_stats
    assert [e.weight for e in default] == [4.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]


@pytest.mark.parametrize(
    ("token_ids", "budget", "eos_id", "expected"),
    [
        ([5, 6, 7], 3, EOS, True),  # filled the budget without EOS
        ([5, 6, EOS], 3, EOS, False),  # EOS as the budget's last token is a finished answer
        ([5, EOS], 3, EOS, False),  # stopped before the budget
        ([5, 6], 3, EOS, False),  # stopped before the budget on a stop string
        ([EOS], 1, EOS, False),
        ([5, 6, 7, 8], 3, EOS, True),
        ([5, 6, EOS], 3, None, True),  # no EOS id: a full budget is always truncated
    ],
)
def test_truncated_without_eos(token_ids, budget, eos_id, expected):
    assert truncated_without_eos(token_ids, budget, eos_id) is expected


def test_require_eos_truncation_is_a_wrong_member_of_its_group():
    # UNIFIED_REQUIRE_EOS=1: the generator marks the completion unfinished and runs with
    # DROP_TRUNCATED=0 and FORCE_TRUNCATED_INCORRECT=1, so it stays in the group as a wrong answer.
    outcomes = [True, True, False, False]
    finished = [not truncated_without_eos(ids, 3, EOS) for ids in ([5, EOS], [5, 6, 7], [5, 6, EOS], [5])]
    assert finished == [True, False, True, True]
    eps, stats = weight_groups(
        [_group(0, outcomes, finished)], **BASELINE, drop_truncated=False, force_truncated_incorrect=True
    )
    correct, wrong = _closed_form(1, 4)
    assert stats.n_truncated == 1 and len(eps) == 4
    assert [e.reward for e in eps] == [1.0, 0.0, 0.0, 0.0]
    assert [e.weight for e in eps] == pytest.approx([correct, wrong, wrong, wrong], rel=1e-6)
