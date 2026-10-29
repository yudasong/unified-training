# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Answer verification shared by unified rollout and evaluation paths.

The default keeps the historical GSM8K final-number behavior. Math runs opt in
to the same Hugging Face ``math-verify`` metric/configuration used by NeMo RL's
``HFVerifyWorker``, without starting Ray actors inside every vLLM producer.
"""

from __future__ import annotations

from functools import lru_cache

from unified_training.rollout_common import extract_answer
from unified_training.strict_math_gate import final_answer_is_unevaluated

LAST_NUMBER = "last_number"
HF_MATH_VERIFY = "hf_math_verify"
# hf_math_verify plus the strict-form gate: a purely numeric final answer must
# be a stated value, not arithmetic the grader would evaluate (see
# strict_math_gate.py).
HF_MATH_VERIFY_STRICT = "hf_math_verify_strict"
# verl's default MATH scorer: the answer must appear in the response's LAST
# \boxed{} (the format itself is rewarded) and is compared to the gold under
# Minerva-style string normalization — no sympy evaluation, so the
# expression-equivalence exploit cannot score. Pair with prompts that ask for
# the boxed format.
VERL_BOX = "verl_box"
# Unit-test reward for code prompts: ``gold`` is a pytest module importing the
# candidate from ``solution`` (KodCode style); see code_reward.py.
CODE_TESTS = "code_tests"
# nl_tower reasoning puzzles (controlled experiments): the LAST brace-matched
# \boxed{...} is parsed as Name=integer pairs and compared, order-free and
# name-casefolded, to the gold pairs.  ``gold`` is the answer string AFTER
# "nl_tower|" in the task's ground-truth string.  No box, or an empty gold dict, is wrong.
NL_TOWER = "nl_tower"
VALID_REWARD_MODES = (LAST_NUMBER, HF_MATH_VERIFY, HF_MATH_VERIFY_STRICT, VERL_BOX, CODE_TESTS, NL_TOWER)


def _is_escaped(text: str, i: int) -> bool:
    """True when text[i] is escaped (odd number of preceding backslashes)."""
    n = 0
    i -= 1
    while i >= 0 and text[i] == "\\":
        n += 1
        i -= 1
    return n % 2 == 1


def last_boxed(text: str):
    """Content of the LAST ``\\boxed{`` whose
    braces balance (escaped braces ignored), stripped; None when absent/unclosed."""
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None
    depth = 1
    for pos in range(start + len(marker), len(text)):
        c = text[pos]
        if c == "{" and not _is_escaped(text, pos):
            depth += 1
        elif c == "}" and not _is_escaped(text, pos):
            depth -= 1
            if depth == 0:
                return text[start + len(marker):pos].strip()
    return None


def nl_tower_correct(response: str, gold: str) -> bool:
    """nl_tower reward as a boolean: boxed Name=integer pairs equal the gold pairs."""
    import re

    ans = last_boxed(response or "")
    if ans is None:
        return False
    ans = re.sub(r"\\text\{([^{}]*)\}", r"\1", ans).strip().strip("$").strip()
    def pairs(s: str) -> dict[str, int]:
        return {m.group(1).casefold(): int(m.group(2)) for m in re.finditer(r"(\w+)\s*=\s*(-?\d+)", s)}

    try:
        g = pairs(gold)
        return bool(g) and pairs(ans) == g
    except Exception:
        return False


@lru_cache(maxsize=1)
def _hf_math_verifier():
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

    metric = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(
            ExprExtractionConfig(),
            LatexExtractionConfig(),
        ),
    )
    return metric, TimeoutException


def validate_reward_mode(mode: str) -> str:
    if mode not in VALID_REWARD_MODES:
        raise ValueError(
            f"unknown reward mode {mode!r}; expected one of {VALID_REWARD_MODES}"
        )
    return mode


def answer_is_correct(response: str, gold: str, *, mode: str = LAST_NUMBER) -> bool:
    """Return correctness under the selected benchmark verifier."""
    validate_reward_mode(mode)
    response = str(response).strip()
    gold = str(gold).strip()
    if not response or not gold:
        return False
    if mode == LAST_NUMBER:
        return extract_answer(response) == gold
    if mode == VERL_BOX:
        from unified_training.verl_math_reward import compute_score

        return compute_score(response, gold) > 0.0
    if mode == CODE_TESTS:
        from unified_training.code_reward import tests_pass

        return tests_pass(response, gold)
    if mode == NL_TOWER:
        return nl_tower_correct(response, gold)

    metric, timeout_exception = _hf_math_verifier()
    try:
        score, _ = metric([f"\\boxed{{{gold}}}"], [response])
        correct = float(score) > 0.0
    # math-verify's TimeoutException inherits BaseException in supported
    # versions, so it must be named separately from Exception.
    except (Exception, timeout_exception):
        return False
    if correct and mode == HF_MATH_VERIFY_STRICT:
        return not final_answer_is_unevaluated(response)
    return correct


def answers_are_correct(responses, golds, *, mode: str = LAST_NUMBER) -> list[bool]:
    """``answer_is_correct`` over aligned sequences.

    Code tests run in sandboxed subprocesses, so that mode scores the whole
    batch concurrently; the math verifiers are cheap and run in order.
    """
    validate_reward_mode(mode)
    responses = [str(r).strip() for r in responses]
    golds = [str(g).strip() for g in golds]
    if len(responses) != len(golds):
        raise ValueError(f"{len(responses)} responses for {len(golds)} golds")
    if mode == CODE_TESTS:
        from unified_training.code_reward import batch_tests_pass

        keep = [bool(r) and bool(g) for r, g in zip(responses, golds)]
        scored = iter(batch_tests_pass(
            [r for r, k in zip(responses, keep) if k],
            [g for g, k in zip(golds, keep) if k],
        ))
        return [next(scored) if k else False for k in keep]
    return [answer_is_correct(r, g, mode=mode) for r, g in zip(responses, golds)]
