# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Strict-form gate for math answer verification.

``math-verify`` scores by sympy equivalence, so a response that ends with the
UNEVALUATED setup of a computation (e.g. ``$\\frac{3(-2)-2}{-2-2}+1$`` instead
of ``14/3``) is credited even when the model never did the arithmetic — the
reward-hacking channel. This module rejects exactly that channel: a **purely numeric** final
answer must be a stated value, not an arithmetic combination of literals.

Deliberately preserved: every evaluated form (``0.5``, ``1/2``,
``\\frac{14}{3}``, ``3\\sqrt{2}``, ``2\\pi``, ``(3, \\frac{\\pi}{2})``) and
every symbolic answer (``p - q``, ``x^2+1``) — symbolic answers legitimately
contain operations, so the gate never fires on expressions with free
variables.

The detection rules are a hand-validated classifier. Dependency-free on
purpose (no sympy).
"""

from __future__ import annotations

import re

# Tail window in which the final expression must end for it to count as the
# response's answer.
_TAIL_CHARS = 400
_TAIL_FRACTION = 0.25

_SPACING_MACROS = re.compile(r"\\[,;!:]|\\quad|\\qquad|\\ ")
_TEXT_MACRO = re.compile(r"\\text\{[^{}]*\}")
_LEFT_RIGHT = re.compile(r"\\left|\\right")

# Grammar of "stated value" answers: signed literals, simple fractions,
# radicals and pi multiples, ratios, and tuples thereof.
_NUMBER = r"-?\d+(?:\.\d+)?"
_SIMPLE_ATOMS = (
    rf"{_NUMBER}",
    rf"{_NUMBER}/\d+",
    rf"-?\\frac\{{-?\d+\}}\{{-?\d+\}}",
    rf"-?\\frac\{{\\pi\}}\{{\d+\}}",
    rf"-?\\frac\{{\d*\\sqrt\{{\d+\}}\}}\{{\d+\}}",
    rf"-?\d*\\sqrt\{{\d+\}}",
    rf"-?\d*\\pi",
    rf"{_NUMBER}:{_NUMBER}",
)
_SIMPLE_ATOM_RE = re.compile(r"^(?:" + "|".join(_SIMPLE_ATOMS) + r")$")

# Letters other than these LaTeX commands mean the answer is symbolic and the
# gate must never fire (symbolic answers legitimately contain operations).
_ALLOWED_COMMANDS = ("frac", "sqrt", "cdot", "times", "div", "pi", "boxed")

_OPERATION_PATTERNS = (
    re.compile(r"\+"),                     # any addition
    re.compile(r"[\d)}]\s*-"),             # binary minus after a value
    re.compile(r"\\cdot|\\times|\\div|\*"),
    re.compile(r"\d\s*\("),                # implicit multiplication 3(...)
    re.compile(r"\)\s*\("),
    re.compile(r"\)\s*\d"),
    re.compile(r"\)\s*/|/\s*\("),          # division on parenthesized operand
    re.compile(r"\d\s*\^"),                # numeric power
)


def _balanced_span(text: str, open_index: int) -> int | None:
    """Return the index one past the brace-balanced group opened at open_index."""
    depth = 0
    for index in range(open_index, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return index + 1
    return None


def _last_boxed(text: str) -> tuple[int, str] | None:
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    end = _balanced_span(text, start + len("\\boxed"))
    if end is None:
        return None
    return end, text[start + len("\\boxed{") : end - 1]


def _last_math_span(text: str) -> tuple[int, str] | None:
    """Last $...$ / $$...$$ / \\[...\\] span containing math-ish content."""
    best: tuple[int, str] | None = None
    display = None
    for match in re.finditer(r"\\\[(.+?)\\\]", text, re.S):
        display = (match.end(), match.group(1))
    if display is not None:
        best = display
    # Dollar spans under normal pairing and under a one-off parity shift, so a
    # single unbalanced $ earlier in the text does not desynchronize us.
    positions = [match.start() for match in re.finditer(r"\$+", text)]
    for phase in (0, 1):
        for index in range(phase, len(positions) - 1, 2):
            start = positions[index]
            end = positions[index + 1]
            inner = text[start:end].lstrip("$")
            if not inner or not re.search(r"[\d\\]", inner):
                continue
            span_end = end + len(re.match(r"\$+", text[end:]).group(0) or "$")
            if best is None or span_end >= best[0]:
                best = (span_end, inner)
    return best


def extract_final_expression(response: str) -> str | None:
    """The expression that acts as the response's final answer, or None."""
    text = str(response)
    for stop in ("<|im_end|>", "<|endoftext|>"):
        if text.endswith(stop):
            text = text[: -len(stop)]
    text = text.rstrip()
    if not text:
        return None

    candidates: list[tuple[int, str]] = []
    boxed = _last_boxed(text)
    if boxed is not None:
        candidates.append(boxed)
    span = _last_math_span(text)
    if span is not None:
        candidates.append(span)
    if not candidates:
        match = None
        for match in re.finditer(
            rf"(?:answer|result)\s*(?:is|:)?\s*({_NUMBER})", text, re.I
        ):
            pass
        if match is not None:
            candidates.append((match.end(), match.group(1)))
    if not candidates:
        return None

    end, expression = max(candidates, key=lambda item: item[0])
    tail_start = max(len(text) - _TAIL_CHARS, int(len(text) * (1 - _TAIL_FRACTION)))
    if end < tail_start:
        return None

    inner_boxed = _last_boxed(expression)
    if inner_boxed is not None:
        expression = inner_boxed[1]
    # Equation chain: keep only the terminal value.
    if "=" in expression:
        expression = expression.rsplit("=", 1)[1]
    return expression.strip()


def _normalize(expression: str) -> str:
    expression = _LEFT_RIGHT.sub("", expression)
    expression = _SPACING_MACROS.sub("", expression)
    expression = _TEXT_MACRO.sub("", expression)
    expression = expression.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    expression = expression.replace("^\\circ", "").replace("^{\\circ}", "")
    expression = expression.replace("\\%", "").replace("%", "")
    expression = expression.replace("$", "")
    return expression.strip().strip(".")


def _is_purely_numeric(expression: str) -> bool:
    stripped = expression
    for command in _ALLOWED_COMMANDS:
        stripped = stripped.replace("\\" + command, "")
    return not re.search(r"[A-Za-z]", stripped)


def _is_stated_value(expression: str) -> bool:
    expression = expression.strip().strip("()[]").strip()
    parts = [part.strip() for part in expression.split(",")] if "," in expression else [expression]
    return all(
        _SIMPLE_ATOM_RE.fullmatch(part.replace(" ", "")) is not None for part in parts
    )


def _contains_literal_arithmetic(expression: str) -> bool:
    if any(pattern.search(expression) for pattern in _OPERATION_PATTERNS):
        return True
    # A \frac or \sqrt whose argument is itself composite.
    for match in re.finditer(r"\\(?:frac|sqrt)\{", expression):
        end = _balanced_span(expression, match.end() - 1)
        if end is None:
            continue
        inner = expression[match.end() : end - 1]
        if re.search(r"[+*]|\\cdot|\\times|\d\s*\(|\)\s*\d|[\d)}]\s*-", inner):
            return True
    return False


def is_unevaluated_numeric_expression(expression: str) -> bool:
    """True when a purely numeric expression is computed-by-grader, not stated."""
    normalized = _normalize(expression)
    if not normalized:
        return False
    if not _is_purely_numeric(normalized):
        return False
    if _is_stated_value(normalized):
        return False
    return _contains_literal_arithmetic(normalized)


def final_answer_is_unevaluated(response: str) -> bool:
    """The strict gate: does this response answer with unevaluated arithmetic?"""
    expression = extract_final_expression(response)
    if expression is None:
        return False
    return is_unevaluated_numeric_expression(expression)


def apply_strict_gate(score: float, response: str) -> float:
    """Zero a positive verifier score when the final answer was never computed."""
    if score > 0.0 and final_answer_is_unevaluated(response):
        return 0.0
    return score
