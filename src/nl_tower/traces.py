"""Chain-of-thought traces for nl_tower questions, written by the deterministic solver (CPU only, no model).

For each question the solver (oracle.py) is run with up to 2K of the 8 variants in VARIANTS (2 narration styles x clue
orders x mixed / unary-first / binary-first propagation) and the first K distinct traces are kept. Rows are
``{qid, prompt, target, answer}``; ``target`` is the solver's narration followed by ``\\boxed{answer}``. The question
order is shuffled with ``seed`` before ``questions`` truncates it, and the output rows are shuffled again by the same
generator. The midtraining corpus (796,175 rows) was written with K = 4 and seed 7 (``prepare_data.py corpus``)."""

from __future__ import annotations

import random
from collections.abc import Iterable
from typing import Any

from .oracle import oracle_trace

VARIANTS = [
    (0, 0, "mixed"),
    (1, 1, "unary_first"),
    (0, 2, "binary_first"),
    (1, 3, "mixed"),
    (0, 4, "unary_first"),
    (1, 5, "binary_first"),
    (0, 6, "mixed"),
    (1, 7, "unary_first"),
]


def question_traces(row: dict[str, Any], k: int = 4) -> list[str]:
    """Up to ``k`` distinct solver traces for one pool row (``metadata`` and ``answer``), in VARIANTS order."""
    seen, made = set(), []
    for style, order_seed, unary in VARIANTS[: max(2 * k, k)]:
        _, trace = oracle_trace(row["metadata"], row["answer"], style=style, order_seed=order_seed, unary=unary)
        if trace is None or trace in seen:
            continue
        seen.add(trace)
        made.append(trace)
        if len(made) >= k:
            break
    return made


def cot_rows(
    pool_rows: Iterable[dict[str, Any]], k: int = 4, questions: int = 0, seed: int = 7
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Trace rows for the questions of a pool -> (rows, summary). ``questions`` > 0 keeps that many questions."""
    rows = list(pool_rows)
    rng = random.Random(seed)
    rng.shuffle(rows)
    if questions:
        rows = rows[:questions]
    out, nfail, ntr = [], 0, 0
    for r in rows:
        made = question_traces(r, k)
        if not made:
            nfail += 1
            continue
        ntr += len(made)
        for t in made:
            out.append({"qid": r["qid"], "prompt": r["prompt"], "target": t, "answer": r["answer"]})
    rng.shuffle(out)
    summary = {
        "cmd": "cot",
        "questions": len(rows),
        "failed": nfail,
        "rows": len(out),
        "traces_per_q": round(ntr / max(1, len(rows) - nfail), 2),
    }
    return out, summary
