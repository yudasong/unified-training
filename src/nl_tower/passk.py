"""Unbiased pass@k (Chen et al., 2021), Appendix C.4.

For one question with n samples of which c are correct: pass@k = 1 - C(n-c, k) / C(n, k) (1 when n - c < k). A test
set's pass@k is the mean over questions. Evaluation outputs store ``[qid, n, c]`` per question (``sample.per_q``), so any
slice or k can be recomputed without new samples.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def pass_at_k(n: int, c: int, k: int) -> float:
    if not 1 <= k <= n:
        raise ValueError(f"need 1 <= k <= n, got k={k}, n={n}")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def mean_pass_at_k(per_q: Iterable[Sequence[int]], k: int) -> float:
    """Mean pass@k over questions; ``per_q`` holds (n, c) or (qid, n, c)."""
    vals = [pass_at_k(int(x[-2]), int(x[-1]), k) for x in per_q]
    return sum(vals) / len(vals) if vals else float("nan")
