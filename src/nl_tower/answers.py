"""Scoring of nl_tower completions by the evaluator (Appendix C.4).

1. The answer is the content of the last ``\\boxed{...}`` whose braces balance (``last_boxed_answer``; an escaped brace
   does not count). An empty box counts as no box.
2. The answer is correct when ``NLTowerDataset.score_answer`` gives 1.0 for the raw answer or for its normalized form
   (``eval_correct``): every ``Name=floor`` pair of the gold assignment and no other pair, no partial credit.
   ``normalize`` strips surrounding ``$`` and ``\\(...\\)``, unwraps ``\\text{...}`` and drops the LaTeX spaces ``\\,``
   and ``\\ ``.

The RL reward (the trainer's ``nl_tower`` reward mode) reads the same last box but case-folds names and compares floor
numbers as integers; the evaluator is case-sensitive and compares numbers as strings ("Ava=07" is not "Ava=7").
"""

from __future__ import annotations

import re
from typing import Any

_BOX = "\\boxed{"


def _escaped(text: str, pos: int) -> bool:
    """True when ``text[pos]`` is preceded by an odd number of backslashes."""
    n = 0
    pos -= 1
    while pos >= 0 and text[pos] == "\\":
        n += 1
        pos -= 1
    return n % 2 == 1


def last_boxed_answer(text: str | None) -> str | None:
    """Content of the last brace-balanced ``\\boxed{...}``, stripped; None if it is missing, unclosed or empty."""
    text = text or ""
    start = text.rfind(_BOX)
    if start < 0:
        return None
    depth = 1
    for pos in range(start + len(_BOX), len(text)):
        if text[pos] == "{" and not _escaped(text, pos):
            depth += 1
        elif text[pos] == "}" and not _escaped(text, pos):
            depth -= 1
            if depth == 0:
                return text[start + len(_BOX) : pos].strip() or None
    return None


def normalize(answer: str) -> str:
    """The evaluator's normalization of a boxed answer before scoring."""
    s = answer.strip().strip("$").strip()
    s = re.sub(r"^\\\(|\\\)$", "", s).strip()
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    s = s.replace(r"\,", "").replace(r"\ ", " ")
    return s.strip()


def eval_score(ds: Any, answer: str, entry: dict[str, Any]) -> float:
    """max(score_answer(answer), score_answer(normalize(answer))): normalization can only add correct answers."""
    best = 0.0
    for candidate in (answer, normalize(answer)):
        try:
            best = max(best, float(ds.score_answer(candidate, entry)))
        except Exception:
            pass
    return best


def eval_correct(ds: Any, answer: str | None, entry: dict[str, Any]) -> bool:
    """The evaluator's correctness of a parsed answer (None = no box) for dataset item ``entry``."""
    return answer is not None and eval_score(ds, answer, entry) >= 1.0
