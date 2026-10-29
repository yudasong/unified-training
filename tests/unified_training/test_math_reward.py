# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import pytest

from unified_training import math_reward


def test_last_number_mode_preserves_existing_gsm8k_behavior():
    assert math_reward.answer_is_correct(
        "Reasoning... therefore #### 1,024", "1024", mode="last_number"
    )
    assert not math_reward.answer_is_correct(
        "Reasoning... therefore 1023", "1024", mode="last_number"
    )


def test_hf_math_mode_boxes_gold_and_uses_metric(monkeypatch):
    calls = []

    def metric(golds, responses):
        calls.append((golds, responses))
        return 1.0, object()

    class FakeTimeout(BaseException):
        pass

    monkeypatch.setattr(math_reward, "_hf_math_verifier", lambda: (metric, FakeTimeout))
    assert math_reward.answer_is_correct(
        r"The result is \\boxed{\\frac{7}{20}}.",
        r"\\dfrac{7}{20}",
        mode="hf_math_verify",
    )
    assert calls == [
        ([r"\boxed{\\dfrac{7}{20}}"], [r"The result is \\boxed{\\frac{7}{20}}."])
    ]


@pytest.mark.parametrize("raised", [ValueError("bad parse")])
def test_hf_math_mode_converts_verifier_errors_to_incorrect(monkeypatch, raised):
    def metric(_golds, _responses):
        raise raised

    class FakeTimeout(BaseException):
        pass

    monkeypatch.setattr(math_reward, "_hf_math_verifier", lambda: (metric, FakeTimeout))
    assert not math_reward.answer_is_correct("answer", "gold", mode="hf_math_verify")


def test_hf_math_mode_catches_baseexception_timeout(monkeypatch):
    class FakeTimeout(BaseException):
        pass

    def metric(_golds, _responses):
        raise FakeTimeout()

    monkeypatch.setattr(math_reward, "_hf_math_verifier", lambda: (metric, FakeTimeout))
    assert not math_reward.answer_is_correct("answer", "gold", mode="hf_math_verify")


@pytest.mark.parametrize("response,gold", [("", "1"), ("answer", ""), (" ", " ")])
def test_blank_response_or_gold_is_always_incorrect(response, gold):
    assert not math_reward.answer_is_correct(response, gold, mode="hf_math_verify")


def test_unknown_reward_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown reward mode"):
        math_reward.answer_is_correct("1", "1", mode="symbolic-ish")


def test_verl_box_mode_requires_the_boxed_format():
    # Correct value without a \boxed{} must NOT score: the format is rewarded.
    assert not math_reward.answer_is_correct(
        "Step by step... the answer is 42.", "42", mode="verl_box"
    )
    assert math_reward.answer_is_correct(
        "Step by step... the answer is \\boxed{42}.", "42", mode="verl_box"
    )
    assert not math_reward.answer_is_correct(
        "\\boxed{41}", "42", mode="verl_box"
    )


def test_verl_box_mode_uses_minerva_normalization_not_sympy():
    # dfrac/frac and a/b -> \frac{a}{b} normalize to equal strings...
    assert math_reward.answer_is_correct(
        "so \\boxed{\\frac{7}{20}}", "\\dfrac{7}{20}", mode="verl_box"
    )
    assert math_reward.answer_is_correct(
        "so \\boxed{7/20}", "\\dfrac{7}{20}", mode="verl_box"
    )
    # ...but arithmetic the grader would have to EVALUATE does not score
    # (the sympy-equivalence exploit is structurally closed).
    assert not math_reward.answer_is_correct(
        "so \\boxed{0.5 - 0.15}", "0.35", mode="verl_box"
    )


def test_verl_box_mode_scores_only_the_last_box():
    assert not math_reward.answer_is_correct(
        "\\boxed{42} wait, actually \\boxed{41}", "42", mode="verl_box"
    )
