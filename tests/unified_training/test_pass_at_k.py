# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Unbiased pass@k of the generator's rollout statistics."""

from __future__ import annotations

import pytest

from unified_training.rollout_common import mean_pass_at_ks, pass_at_k


def test_unbiased_pass_at_k_known_values() -> None:
    assert pass_at_k(16, 0, 8) == 0.0
    assert pass_at_k(16, 16, 1) == 1.0
    assert pass_at_k(16, 1, 1) == pytest.approx(1 / 16)
    assert pass_at_k(16, 1, 16) == 1.0


def test_mean_pass_at_ks_uses_all_sixteen_samples() -> None:
    groups = [
        [True] + [False] * 15,
        [False] * 16,
        [True] * 4 + [False] * 12,
    ]
    scores = mean_pass_at_ks(groups)
    assert tuple(scores) == (1, 2, 4, 8, 16)
    assert scores[1] == pytest.approx((1 / 16 + 0 + 4 / 16) / 3)
    assert scores[16] == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    "args",
    [
        (0, 0, 1),
        (16, -1, 1),
        (16, 17, 1),
        (16, 1, 0),
        (16, 1, 17),
    ],
)
def test_pass_at_k_rejects_invalid_arguments(args: tuple[int, int, int]) -> None:
    with pytest.raises(ValueError):
        pass_at_k(*args)

