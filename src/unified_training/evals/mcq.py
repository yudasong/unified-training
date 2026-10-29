"""SciQ / ARC-Challenge: choice-visible greedy prompt and strict leading-label scoring (8 new tokens)."""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Sequence


_CHOICE_LABELS = tuple(chr(ord("A") + index) for index in range(26))
_STRICT_CHOICE = re.compile(r"[A-Z]")
_LEADING_CHOICE = re.compile(
    # A bare label needs a token boundary so that prose such as ``Cats`` is
    # not read as choice C.  Closing punctuation is itself an unambiguous
    # boundary, however, and must not require following whitespace: vLLM can
    # render a generated stop-token fragment immediately after ``C.`` (for
    # example ``C.<|im_end``).
    r"^\s*(?:(?:final\s+)?answer\s*:\s*)?"
    r"(?:\(([A-Z])\)(?:\.)?|([A-Z])(?:[.)]|(?=\s|$)))",
    re.IGNORECASE,
)


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _converted_row(doc: dict, choices: Sequence[str], answer_index: int) -> dict:
    if not 2 <= len(choices) <= len(_CHOICE_LABELS):
        raise ValueError(f"expected between 2 and 26 choices, got {len(choices)}")
    normalized_choices = [
        _text(choice, field=f"choice {index}") for index, choice in enumerate(choices)
    ]
    if not 0 <= answer_index < len(normalized_choices):
        raise ValueError(f"answer index {answer_index} is outside the choice list")
    return {
        "eval_choices": normalized_choices,
        "eval_answer": _CHOICE_LABELS[answer_index],
    }


def process_arc_docs(dataset):
    """Canonicalize ARC labels while retaining the dataset's published order."""

    def convert(doc):
        choices = doc.get("choices")
        if not isinstance(choices, dict):
            raise ValueError("ARC row is missing its choices mapping")
        labels = [str(label).strip().upper() for label in choices.get("label", [])]
        texts = list(choices.get("text", []))
        if len(labels) != len(texts) or len(set(labels)) != len(labels):
            raise ValueError("ARC choice labels and texts must be unique and aligned")
        answer_key = str(doc.get("answerKey", "")).strip().upper()
        if answer_key not in labels:
            raise ValueError(f"ARC answer key {answer_key!r} is not among {labels!r}")
        return _converted_row(doc, texts, labels.index(answer_key))

    return dataset.map(convert)


def process_sciq_docs(dataset):
    """Shuffle SciQ choices deterministically without position-label leakage."""

    def convert(doc):
        question = _text(doc.get("question"), field="SciQ question")
        support = str(doc.get("support", "")).strip()
        tagged_choices = [
            (_text(doc.get("distractor1"), field="SciQ distractor1"), False),
            (_text(doc.get("distractor2"), field="SciQ distractor2"), False),
            (_text(doc.get("distractor3"), field="SciQ distractor3"), False),
            (_text(doc.get("correct_answer"), field="SciQ correct_answer"), True),
        ]
        seed_material = f"{question}\n{support}".encode()
        seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
        random.Random(seed).shuffle(tagged_choices)
        answer_index = next(
            index for index, (_, is_correct) in enumerate(tagged_choices) if is_correct
        )
        return _converted_row(
            doc,
            [choice for choice, _ in tagged_choices],
            answer_index,
        )

    return dataset.map(convert)


def doc_to_greedy_mcq(doc: dict) -> str:
    """Render a zero-shot, choice-visible prompt with an exact output contract."""
    choices = doc.get("eval_choices")
    if not isinstance(choices, list) or not 2 <= len(choices) <= len(_CHOICE_LABELS):
        raise ValueError("eval_choices must contain between 2 and 26 choices")
    labels = _CHOICE_LABELS[: len(choices)]

    sections = ["Select the best answer to the multiple-choice question below."]
    support = str(doc.get("support", "")).strip()
    if support:
        sections.append(f"Context:\n{support}")
    sections.append(f"Question:\n{_text(doc.get('question'), field='question')}")
    sections.append(
        "Choices:\n"
        + "\n".join(
            f"{label}. {_text(choice, field=f'choice {index}')}"
            for index, (label, choice) in enumerate(zip(labels, choices))
        )
    )
    sections.append(
        "Respond with exactly one uppercase letter from the listed choices "
        f"({_human_join(labels)}), and nothing else."
    )
    return "\n\n".join(sections)


def doc_to_mcq_target(doc: dict) -> str:
    target = doc.get("eval_answer")
    choices = doc.get("eval_choices")
    if not isinstance(choices, list) or target not in _CHOICE_LABELS[: len(choices)]:
        raise ValueError(f"invalid multiple-choice target: {target!r}")
    return target


def _human_join(labels: Sequence[str]) -> str:
    if len(labels) == 2:
        return f"{labels[0]} or {labels[1]}"
    return ", ".join(labels[:-1]) + f", or {labels[-1]}"


def extract_greedy_choice(response: str, num_choices: int) -> str | None:
    """Extract only an answer anchored at the start of a greedy response.

    Bare labels, parenthesized labels, label punctuation, and an explicit
    leading ``Answer:`` prefix are accepted. The parser never scans later text
    and never infers a label from answer text.
    """
    if not isinstance(response, str):
        raise TypeError(f"greedy response must be a string, got {type(response).__name__}")
    if not 2 <= num_choices <= len(_CHOICE_LABELS):
        raise ValueError(f"num_choices must be between 2 and 26, got {num_choices}")
    allowed = _CHOICE_LABELS[:num_choices]
    match = _LEADING_CHOICE.match(response)
    if match is None:
        return None
    candidate = (match.group(1) or match.group(2)).upper()
    return candidate if candidate in allowed else None


def is_format_valid_response(response: str, num_choices: int) -> bool:
    """Whether a response follows the requested exact uppercase-letter format."""
    if not isinstance(response, str):
        raise TypeError(f"greedy response must be a string, got {type(response).__name__}")
    if not 2 <= num_choices <= len(_CHOICE_LABELS):
        raise ValueError(f"num_choices must be between 2 and 26, got {num_choices}")
    stripped = response.strip()
    return bool(
        _STRICT_CHOICE.fullmatch(stripped)
        and stripped in _CHOICE_LABELS[:num_choices]
    )


def unpack_single_response(results: object) -> str:
    """Unpack supported lm-eval singleton shapes and reject every other shape."""
    if not isinstance(results, (list, tuple)) or len(results) != 1:
        raise ValueError(f"expected exactly one generation result, got {results!r}")
    response = results[0]
    # A take_first filter supplies ["A"]. A logger/filter variant can retain the
    # repeated-response dimension and supply [["A"]]. Both encode one request
    # and one response; deeper or plural nesting is ambiguous and must fail.
    if isinstance(response, (list, tuple)):
        if len(response) != 1:
            raise ValueError(f"expected one repeated response, got {response!r}")
        response = response[0]
    if not isinstance(response, str):
        raise ValueError(f"expected a string response, got {response!r}")
    return response


def process_greedy_mcq_results(doc: dict, results: object) -> dict[str, float]:
    response = unpack_single_response(results)
    num_choices = len(doc.get("eval_choices", []))
    prediction = extract_greedy_choice(response, num_choices)
    target = doc_to_mcq_target(doc)
    return {
        "greedy_accuracy": float(prediction == target),
        "format_valid_rate": float(is_format_valid_response(response, num_choices)),
    }
