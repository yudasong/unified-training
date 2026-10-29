"""MATH-500 / GSM8K prompts, gold answers and math-verify scoring (16-sample protocol).

The prompt strings, gold extraction and scorer are those of the evaluator that produced the paper's
numbers; the lm-eval task YAMLs in ``tasks/`` call into this module.
"""

from __future__ import annotations

from functools import lru_cache

from unified_training.evals.passk import EXPECTED_SAMPLES, score_samples


def doc_to_math(doc) -> str:
    return (
        "Solve the following mathematics problem. Show your reasoning and put only the final "
        f"answer inside \\boxed{{}}.\n\n{doc['problem'].strip()}"
    )


def doc_to_gsm8k(doc) -> str:
    return (
        "Solve the following grade-school mathematics problem. Show your reasoning and put "
        f"only the final answer inside \\boxed{{}}.\n\n{doc['question'].strip()}"
    )


def gsm8k_gold_answer(doc) -> str:
    marker = "####"
    answer = str(doc["answer"])
    if marker not in answer:
        raise ValueError("GSM8K answer is missing its final-answer marker")
    return answer.rsplit(marker, 1)[1].strip().replace(",", "")


@lru_cache(maxsize=1)
def _math_metric():
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

    metric = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    return metric, TimeoutException


def math_correct(gold: str, response: str) -> bool:
    """math-verify equivalence of the response's final answer and ``\\boxed{gold}``."""
    metric, timeout_exception = _math_metric()
    try:
        score, _ = metric([f"\\boxed{{{gold}}}"], [response])
        return float(score) > 0.0
    # math-verify 0.8.0's TimeoutException inherits BaseException, so it is named explicitly.
    except (Exception, timeout_exception):
        return False


def all_responses(results, n: int = EXPECTED_SAMPLES) -> list[str]:
    if len(results) != 1 or not isinstance(results[0], list):
        raise ValueError(f"expected one repeated generation request, got {results!r}")
    responses = results[0]
    if len(responses) != n or not all(isinstance(item, str) for item in responses):
        raise ValueError(f"expected {n} string responses, got {responses!r}")
    return responses


def process_math_results(doc, results) -> dict[str, float]:
    return score_samples([math_correct(doc["answer"], r) for r in all_responses(results)])


def process_gsm8k_results(doc, results) -> dict[str, float]:
    gold = gsm8k_gold_answer(doc)
    return score_samples([math_correct(gold, r) for r in all_responses(results)])
