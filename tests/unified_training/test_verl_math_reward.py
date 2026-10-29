from __future__ import annotations

import pytest

from unified_training.verl_math_reward import compute_score


@pytest.mark.parametrize(
    ("response", "gold"),
    [
        ("the answer is \\boxed{\\frac{7}{20}}", "\\dfrac{7}{20}"),
        ("so \\boxed{7/20}", "\\dfrac{7}{20}"),
        ("thus \\boxed{\\frac{2\\sqrt{149}}{3}}", "\\frac{2 \\sqrt{149}}{3}"),
        ("\\boxed{42}", "42"),
        ("\\boxed{x^3-10x^2+37x-52}", "x^3-10x^2+37x-52"),
    ],
)
def test_normalized_equivalence_accepts_equivalent_forms(response: str, gold: str) -> None:
    assert compute_score(response, gold) == 1.0


@pytest.mark.parametrize(
    ("response", "gold"),
    [
        # Correct value but unboxed: verl's MATH scorer enforces the format.
        ("Answer: \\frac{7}{20}", "\\dfrac{7}{20}"),
        # The sympy-equivalence exploit: normalized STRING comparison rejects
        # an unevaluated expression even though it equals the gold.
        ("\\boxed{\\frac{3(-2)-2}{-2-2}+1}", "\\frac{14}{3}"),
        # Plain wrong answer.
        ("\\boxed{\\frac{7}{20}}", "\\frac{14}{3}"),
    ],
)
def test_rejects_unboxed_wrong_and_exploit(response: str, gold: str) -> None:
    assert compute_score(response, gold) == 0.0


def test_last_box_wins() -> None:
    response = "first try \\boxed{5} but actually \\boxed{42}"
    assert compute_score(response, "42") == 1.0
    assert compute_score(response, "5") == 0.0
