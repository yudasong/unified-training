"""Unbiased pass@k (Chen et al., 2021) and per-problem aggregation."""

from __future__ import annotations

from math import comb
from typing import Iterable, Sequence

PASS_KS = (1, 2, 4, 8, 16)
MANY_KS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
EXPECTED_SAMPLES = 16


def pass_at_k(n: int, correct: int, k: int) -> float:
    """Unbiased pass@k estimator from n samples with `correct` successes."""
    if not 0 <= correct <= n:
        raise ValueError(f"expected 0 <= correct <= n, got correct={correct}, n={n}")
    if not 1 <= k <= n:
        raise ValueError(f"expected 1 <= k <= n, got k={k}, n={n}")
    if n - correct < k:
        return 1.0
    return 1.0 - comb(n - correct, k) / comb(n, k)


def score_samples(correct: Sequence[bool], *, prefix: str = "", n: int = EXPECTED_SAMPLES) -> dict[str, float]:
    """sample_mean and pass@{1,2,4,8,16} for one problem's `n` graded responses."""
    if len(correct) != n:
        raise ValueError(f"expected {n} responses, got {len(correct)}")
    count = sum(bool(c) for c in correct)
    key_prefix = f"{prefix}_" if prefix else ""
    scores = {f"{key_prefix}sample_mean": count / len(correct)}
    scores.update({f"{key_prefix}pass_at_{k}": pass_at_k(len(correct), count, k) for k in PASS_KS if k <= n})
    return scores


def aggregate_counts(counts: Iterable[int], n: int, ks: Iterable[int] = MANY_KS) -> dict[str, float]:
    """Mean over problems of pass@k, given each problem's number of correct samples out of n."""
    counts = list(counts)
    if not counts:
        raise ValueError("no problems to aggregate")
    out = {"sample_mean": sum(counts) / (len(counts) * n)}
    for k in ks:
        if k <= n:
            out[f"pass_at_{k}"] = sum(pass_at_k(n, c, k) for c in counts) / len(counts)
    return out
