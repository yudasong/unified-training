import hashlib
import json
from collections import Counter

import numpy as np
import pytest

from unified_training.corpus_mix import (
    OLMO3_COMPONENT_WEIGHTS,
    allocate_weighted_counts,
    interleave_weighted_counts,
    normalize_component_weights,
    stable_component_seed,
)


def test_olmo3_profile_has_all_components_and_normalizes() -> None:
    normalized = normalize_component_weights(OLMO3_COMPONENT_WEIGHTS)

    assert len(normalized) == 24
    assert sum(normalized.values()) == pytest.approx(1.0)
    assert normalized["common_crawl-high-quality"] == pytest.approx(0.225 / 1.0004)


def test_allocate_weighted_counts_is_exact_and_keeps_every_component() -> None:
    weights = {"web": 0.6, "math": 0.3, "code": 0.1}

    counts = allocate_weighted_counts(101, weights)

    assert counts == {"code": 10, "math": 30, "web": 61}
    assert sum(counts.values()) == 101
    assert all(count > 0 for count in counts.values())


def test_interleave_weighted_counts_preserves_counts_without_component_blocks() -> None:
    counts = {"web": 6, "math": 3, "code": 1}

    schedule = list(interleave_weighted_counts(counts))

    assert Counter(schedule) == Counter(counts)
    assert schedule != ["code", "math", "math", "math", *("web" for _ in range(6))]
    assert max(
        len(run)
        for component in counts
        for run in "".join("1" if item == component else "0" for item in schedule).split("0")
    ) <= 2


def test_invalid_or_unrepresentable_mixtures_fail() -> None:
    with pytest.raises(ValueError, match="invalid weight"):
        normalize_component_weights({"web": 1.0, "math": 0.0})
    with pytest.raises(ValueError, match="cannot represent all"):
        allocate_weighted_counts(1, {"web": 0.5, "math": 0.5})


def test_component_seed_is_stable_and_component_specific() -> None:
    assert stable_component_seed(1234, "math") == stable_component_seed(1234, "math")
    assert stable_component_seed(1234, "math") != stable_component_seed(1234, "code")
