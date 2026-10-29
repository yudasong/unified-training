from __future__ import annotations

import pytest

from unified_training.math_reward import (
    HF_MATH_VERIFY_STRICT,
    VALID_REWARD_MODES,
    validate_reward_mode,
)
from unified_training.strict_math_gate import (
    apply_strict_gate,
    extract_final_expression,
    final_answer_is_unevaluated,
    is_unevaluated_numeric_expression,
)

FX_EXPLOIT = (
    "So, the final answer is $\\frac{3(-2)-2}{-2-2} + \\frac{3(-1)-2}{-1-2} + 1$."
    "<|im_end|>"
)
FX_EXPLOIT_TRUNCATED_LOOP = (
    "$f(-2) + f(-1) + f(0) = 2 + 1 + 1 = 4$\n"
    + "Therefore, the final answer is $\\frac{3(-2)-2}{-2-2} + \\frac{3(-1)-2}{-1-2} + 1$.\n"
    * 12
)
ECHO_HACK = "$99^2+99+1$"


@pytest.mark.parametrize(
    "response",
    [
        FX_EXPLOIT,
        FX_EXPLOIT_TRUNCATED_LOOP,
        ECHO_HACK,
        "The answer is \\boxed{2 + \\frac{5}{3} + 1}<|im_end|>",
        "$\\frac{-8}{-4}+\\frac{-5}{-3}+\\frac{-2}{-2}$<|im_end|>",
        "The value is $3^3$.<|im_end|>",
        "distance $= \\sqrt{(2-(-4))^2+(-6-3)^2}$<|im_end|>",
        "$5^5 \\div 5^4 - 5^3 + 5^2 \\cdot 5^1$",
    ],
)
def test_gate_fires_on_unevaluated_final_answers(response: str) -> None:
    assert final_answer_is_unevaluated(response)
    assert apply_strict_gate(1.0, response) == 0.0


@pytest.mark.parametrize(
    "response",
    [
        "The answer is \\boxed{\\frac{14}{3}}<|im_end|>",
        "The answer is 42.<|im_end|>",
        "$0.5$<|im_end|>",
        "$1/2$<|im_end|>",
        "The point is $(3, \\frac{\\pi}{2})$.<|im_end|>",
        "$3\\sqrt{2}$<|im_end|>",
        "$2\\pi$<|im_end|>",
        "The angle is $74^\\circ$.<|im_end|>",
        "So $p - q$ is the result.<|im_end|>",
        "\\boxed{x^2 + 1}<|im_end|>",
        "we get $f(0) = \\frac{-2}{-2} = 1$<|im_end|>",
        "ratio is $3:4$<|im_end|>",
        "negative: \\boxed{-\\frac{2}{3}}<|im_end|>",
    ],
)
def test_gate_passes_stated_and_symbolic_answers(response: str) -> None:
    assert not final_answer_is_unevaluated(response)
    assert apply_strict_gate(1.0, response) == 1.0


def test_gate_never_promotes_zero_scores() -> None:
    assert apply_strict_gate(0.0, FX_EXPLOIT) == 0.0
    assert apply_strict_gate(0.0, "The answer is 42.<|im_end|>") == 0.0


def test_equation_chain_keeps_terminal_value() -> None:
    response = "$f(-2)+f(-1)+f(0) = 2 + \\frac{5}{3} + 1 = \\frac{14}{3}$<|im_end|>"
    assert extract_final_expression(response) == "\\frac{14}{3}"
    assert not final_answer_is_unevaluated(response)


def test_expression_outside_tail_is_ignored() -> None:
    filler = "step " * 400
    response = f"early work $1+1$ then {filler} and the answer is 5.<|im_end|>"
    assert not final_answer_is_unevaluated(response)


def test_expression_classifier_directly() -> None:
    assert is_unevaluated_numeric_expression("2 + \\frac{5}{3} + 1")
    assert is_unevaluated_numeric_expression("\\frac{3(-2)-2}{-2-2}")
    assert not is_unevaluated_numeric_expression("\\frac{14}{3}")
    assert not is_unevaluated_numeric_expression("p - q")
    assert not is_unevaluated_numeric_expression("106")


def test_strict_mode_is_registered() -> None:
    assert HF_MATH_VERIFY_STRICT in VALID_REWARD_MODES
    assert validate_reward_mode(HF_MATH_VERIFY_STRICT) == HF_MATH_VERIFY_STRICT
