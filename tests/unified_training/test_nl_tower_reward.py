# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Reward mode nl_tower: the reasoning-puzzle scorer (last boxed Name=integer pairs)."""

from __future__ import annotations

import pytest

from unified_training import math_reward
from unified_training.math_reward import (
    NL_TOWER,
    VALID_REWARD_MODES,
    _is_escaped,
    answer_is_correct,
    answers_are_correct,
    nl_tower_correct,
    last_boxed,
    validate_reward_mode,
)

GOLD = "Felix=6, Zoe=18, Ruby=39, Ava=13, Owen=33, Noah=31, Hugo=36, Liam=37, Lucas=27"
REASONING = (
    "Since Ava must be on an odd floor, Ava is one of {13, 27, 31, 33, 37, 39}.\n"
    "The clue Liam - Lucas = 10 leaves these pairs for (Liam,Lucas): {(37,27)}.\n"
    "That pins Noah to 31, so 31 is taken.\n"
)


def _boxed(answer: str) -> str:
    return REASONING + "\\boxed{" + answer + "}"


def _texted(answer: str) -> str:
    return ", ".join(f"\\text{{{name}}}={value}" for name, value in (p.split("=") for p in answer.split(", ")))


def test_realistic_demo_is_correct_in_any_order():
    assert answer_is_correct(_boxed(GOLD), GOLD, mode=NL_TOWER)
    assert answer_is_correct(_boxed(", ".join(reversed(GOLD.split(", ")))), GOLD, mode=NL_TOWER)
    assert answers_are_correct([_boxed(GOLD), _boxed("Felix=6")], [GOLD, GOLD], mode=NL_TOWER) == [True, False]


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        # a wrong value, a missing person, an extra person, a sign flip
        (_boxed(GOLD.replace("Zoe=18", "Zoe=19")), False),
        (_boxed(GOLD.replace(", Lucas=27", "")), False),
        (_boxed(GOLD + ", Mia=5"), False),
        (_boxed(GOLD.replace("Zoe=18", "Zoe=-18")), False),
        # no box: bare answer, missing backslash, missing brace, empty box
        (REASONING + GOLD, False),
        (REASONING + "boxed{" + GOLD + "}", False),
        (REASONING + "\\boxed " + GOLD, False),
        (_boxed(""), False),
        # the LAST box is graded
        (_boxed("Felix=1") + "\nOn reflection:\n\\boxed{" + GOLD + "}", True),
        (_boxed(GOLD) + "\nOn reflection:\n\\boxed{Felix=1}", False),
        # an unclosed later box hides an earlier complete one
        (_boxed(GOLD) + "\nOn reflection:\n\\boxed{Felix=6, Zoe=1", False),
        # \text{} wrappers, $ delimiters, casing, spacing and separators are normalised
        (_boxed(_texted(GOLD)), True),
        (_boxed("$" + GOLD + "$"), True),
        (_boxed(GOLD.upper()), True),
        (_boxed(GOLD.lower()), True),
        (_boxed(GOLD.replace("=", " = ")), True),
        (_boxed(GOLD.replace(", ", "; ")), True),
        (_boxed(GOLD.replace(", ", "\n")), True),
        # escaped braces do not close (or open) the box; an unescaped one closes it early
        (_boxed(GOLD.replace(", Ruby", " \\} Ruby")), True),
        (_boxed(GOLD.replace(", Ruby", " \\{ Ruby")), True),
        (_boxed(GOLD.replace(", Ruby", "} Ruby")), False),
        # Unicode digits parse to their integer value (re's \d and int() are Unicode-aware)
        (_boxed(GOLD.replace("Zoe=18", "Zoe=１８")), True),
        (_boxed(GOLD.replace("Zoe=18", "Zoe=١٨")), True),
        (_boxed(GOLD.replace("Zoe=18", "Zoe=１９")), False),
    ],
)
def test_nl_tower_edge_cases(response, expected):
    assert nl_tower_correct(response, GOLD) is expected
    assert answer_is_correct(response, GOLD, mode=NL_TOWER) is expected


def test_every_truncation_of_a_correct_answer_is_wrong():
    full = _boxed(GOLD)
    assert nl_tower_correct(full, GOLD)
    assert not any(nl_tower_correct(full[:cut], GOLD) for cut in range(len(full)))


@pytest.mark.parametrize("gold", ["", "   ", "unknown", "Felix is on 6"])
def test_gold_without_pairs_is_never_correct(gold):
    assert not nl_tower_correct(_boxed(GOLD), gold)
    assert not nl_tower_correct(_boxed(gold), gold)
    assert not answer_is_correct(_boxed(gold), gold, mode=NL_TOWER)


def test_missing_response_is_wrong():
    assert not nl_tower_correct(None, GOLD)
    assert not nl_tower_correct("", GOLD)


def test_last_boxed_matches_braces_and_honours_escapes():
    assert last_boxed("a \\boxed{ \\text{A}=1, B=2 } b") == "\\text{A}=1, B=2"
    assert last_boxed("\\boxed{A=1} then \\boxed{A=2}") == "A=2"
    assert last_boxed("\\boxed{A=1 \\} B=2}") == "A=1 \\} B=2"
    assert last_boxed("\\boxed{A=1 \\\\} B=2}") == "A=1 \\\\"
    assert last_boxed("\\boxed{A=1") is None
    assert last_boxed("no box") is None
    assert _is_escaped("a\\}", 2)
    assert not _is_escaped("a\\\\}", 3)
    assert not _is_escaped("}", 0)


def test_mode_is_registered_and_other_modes_do_not_use_it(monkeypatch):
    assert NL_TOWER == "nl_tower" and NL_TOWER in VALID_REWARD_MODES
    assert validate_reward_mode(NL_TOWER) == NL_TOWER

    def fail(*_args, **_kwargs):
        raise AssertionError("nl_tower scorer called outside mode nl_tower")

    monkeypatch.setattr(math_reward, "nl_tower_correct", fail)
    assert answer_is_correct("so the answer is #### 12", "12", mode="last_number")
    assert not answer_is_correct(_boxed(GOLD), GOLD, mode="last_number")
    assert not answer_is_correct(_boxed(GOLD), GOLD)
