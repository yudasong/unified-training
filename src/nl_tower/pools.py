"""Question pools, tests, RL prompt pools, the trace corpus mix and its contamination guard (CPU only).

* ``pool_rows``: a test set is the first N unique questions of a seeded generator stream, after skipping excluded
  qids; the pool is the questions after it. Rows ``{qid, prompt, question, answer, metadata}`` with
  ``prompt = question + "\\n\\nAnswer: "``.
* ``rl_pool_rows``: a fresh RL prompt pool, the first N unique questions of ``make_dataset(spec, size=3N, seed)``. Rows
  ``{spec, task, qid, prompt, question, answer, metadata}`` with ``prompt = question.strip() + "\\n\\nAnswer: "``.
* ``mix_rows``: shuffled concatenation (``random.Random(seed).shuffle``).
* ``exclusion_rows`` / ``audit``: every question that must not appear in a mid-training corpus, and the overlap check
  of a corpus against it (exact question text and qid).

The exclusion list takes the RL prompt tables of earlier prompt files from ``mixes.verl_prompt_tables`` (built in
memory, same rows in the same order); the audit report stores the corpus path relative to the work directory.
"""

from __future__ import annotations

import collections
import json
import os
import random
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .specs import SPECS, iter_items, make_dataset, qhash

SUFFIX = "\n\nAnswer: "


def jsonable(o: Any) -> Any:
    """Generator metadata as plain JSON types (tuples -> lists, keys -> str)."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, list | tuple | set):
        return [jsonable(v) for v in o]
    if isinstance(o, str | int | float | bool) or o is None:
        return o
    return str(o)


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """One ``json.dumps(row)`` per line, written to ``path.tmp`` and renamed."""
    path = str(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def read_qids(paths: Iterable[str | Path]) -> set[str]:
    """The ``qid`` fields of JSONL files (the format of ``exclude`` lists and of pool files)."""
    qids = set()
    for p in paths:
        with open(p) as f:
            for line in f:
                qids.add(json.loads(line)["qid"])
    return qids


def pool_rows(
    spec: str,
    n_train: int,
    n_test: int,
    seed: int,
    exclude: set[str] | frozenset[str] = frozenset(),
    log: Callable[[str], None] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """-> (test_rows, pool_rows, n_excluded). The first ``n_test`` unique questions are the test split."""
    want = n_train + n_test
    size = max(2 * want, want + 256)
    ds = make_dataset(spec, size=size, seed=seed)
    seen, uniq, t0, n_excl = set(), [], time.time(), 0
    it = iter_items(ds, size)
    for e in it:
        h = qhash(e["question"])
        if h in exclude:
            n_excl += 1
            continue
        if h in seen:
            continue
        seen.add(h)
        uniq.append(e)
        if log and len(uniq) % 1000 == 0:
            log(f"{spec}: {len(uniq)}/{want} unique questions after {time.time() - t0:.0f}s")
        if len(uniq) >= want:
            break
    it.close()

    def row(e):
        return {
            "qid": qhash(e["question"]),
            "prompt": e["question"] + SUFFIX,
            "question": e["question"],
            "answer": str(e["answer"]),
            "metadata": jsonable(e.get("metadata", {})),
        }

    test, train = uniq[:n_test], uniq[n_test : n_test + n_train]
    return [row(e) for e in test], [row(e) for e in train], n_excl


def rl_pool_rows(spec: str, n: int, seed: int = 777) -> list[dict[str, Any]]:
    """The first ``n`` unique questions of ``make_dataset(spec, size=3n, seed)`` as RL prompt-pool rows."""
    ds = make_dataset(spec, size=n * 3, seed=seed)
    rows, seen = [], set()
    it = iter_items(ds, n * 3)
    for e in it:
        h = qhash(e["question"])
        if h in seen:
            continue
        seen.add(h)
        rows.append(
            {
                "spec": spec,
                "task": SPECS[spec][0],
                "qid": h,
                "prompt": f"{e['question'].strip()}{SUFFIX}",
                "question": e["question"],
                "answer": str(e["answer"]),
                "metadata": jsonable(e.get("metadata", {})),
            }
        )
        if len(rows) >= n:
            break
    it.close()
    return rows


def mix_rows(inputs: Sequence[Sequence[dict[str, Any]]], seed: int) -> list[dict[str, Any]]:
    """Concatenate row lists and shuffle with ``random.Random(seed)``."""
    rows = [r for rows in inputs for r in rows]
    random.Random(seed).shuffle(rows)
    return rows


def question_of(row: Mapping[str, Any]) -> str:
    """The question text of a pool, corpus or prompt row (its prompt without the answer suffix)."""
    if "question" in row:
        return row["question"].strip()
    p = row["prompt"]
    if isinstance(p, list):
        p = p[0]["content"]
    return p[: -len(SUFFIX)].strip() if p.endswith(SUFFIX) else p.strip()


# Sources of the exclusion list, in its recorded order: the four 09-21 rungs' tests and pools, the three RL prompt
# pools, then the six earlier RL prompt tables (mixes.VERL_TABLES).
EXCLUSION_RUNGS = ("nltw_e", "nltw_m0", "nltw_m", "nltw_h")
EXCLUSION_RL_POOLS = ("nltw_m0", "nltw_m", "nltw_h")


def exclusion_rows(root: str | Path, verl_tables: Mapping[str, Sequence[str]]) -> list[dict[str, str]]:
    """Every question that must not appear in a mid-training corpus, as rows ``{qid, question, src}``.

    ``root`` is the work directory; ``verl_tables`` maps each earlier table name to its prompts in table order
    (``mixes.verl_prompt_tables``). The published list has 139,404 rows / 66,096 unique qids."""
    from . import paths

    data, rl_prompts = paths.get("data", root), paths.get("rl_prompts", root)
    rows = []

    def add(src, q):
        rows.append({"qid": qhash(q), "question": q, "src": src})

    for sp in EXCLUSION_RUNGS:
        for f in ("test.jsonl", "pool.jsonl"):
            for r in read_jsonl(data / sp / "corpus_tw1" / f):
                add(f"{sp}/corpus_tw1/{f}", r["question"])
    for sp in EXCLUSION_RL_POOLS:
        for r in read_jsonl(rl_prompts / f"{sp}.jsonl"):
            add(f"rl_prompts/{sp}", r["question"])
    for name, prompts in verl_tables.items():
        for p in prompts:
            add(f"rl/data/{name}", question_of({"prompt": p}))
    return rows


def audit(corpus: str | Path, exclude: Sequence[Mapping[str, str]], root: str | Path) -> dict[str, Any]:
    """Exact-text and qid overlap of a corpus (rows with ``question`` or ``prompt``) with every source of ``exclude``.

    ``total_overlap`` of the ``new`` block must be 0. The corpus path in the report is relative to ``root`` when it
    lies under it."""
    by_src, by_src_h = collections.defaultdict(set), collections.defaultdict(set)
    for r in exclude:
        by_src[r["src"]].add(r["question"])
        by_src_h[r["src"]].add(r["qid"])

    def rel(p):
        try:
            return str(Path(p).resolve().relative_to(Path(root).resolve()))
        except ValueError:
            return str(p)

    def one(path):
        qs, hs, n = set(), set(), 0
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                q = question_of(r)
                qs.add(q)
                hs.add(r.get("qid") or qhash(q))
                n += 1
        rep = {"rows": n, "unique_questions": len(qs), "overlap_text": {}, "overlap_hash": {}}
        for s in by_src:
            ot, oh = len(qs & by_src[s]), len(hs & by_src_h[s])
            if ot or oh:
                rep["overlap_text"][s] = ot
                rep["overlap_hash"][s] = oh
        rep["total_overlap"] = sum(rep["overlap_text"].values()) + sum(rep["overlap_hash"].values())
        return rep

    return {"corpus": rel(corpus), "new": one(corpus)}
