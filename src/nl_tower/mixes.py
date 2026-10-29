"""RL prompt mixes of the two settings, as trainer prompt JSONs (CPU only; numpy, no pandas).

Each mix is a JSON list of ``{"question", "gold", "qid"}``: ``question`` is the raw completion prompt (question +
``"\n\nAnswer: "``), ``gold`` the answer string scored by the trainer's ``nl_tower`` reward.

* Base (Table 5): 10,000 demonstrated-pattern (``nltw_m0``) + 18,570 omitted-composition (``nltw_m``) prompts. Each
  pool is shuffled with ``random.Random(0)`` and cut; the union is shuffled with ``random.Random(1)``. Test:
  ``nltw_m``.
* Extended: 10,000 ``nltw_m0`` + 11,666 ``nltw_m`` rows drawn from the Base rows + the 10,000 ``nltw_hb`` (10-11
  people) prompts, shuffled with ``random.Random(1012)``. Test: ``nltw_hb``; generalization test: ``nltw_hb12``.
  The 21,666 Base-size rows and their order come from an intermediate mix (``order_mix``): the Base rows are drawn by
  pandas ``DataFrame.sample(n, random_state=1)``, joined with a 10,000-prompt ``nltw_h`` block and shuffled by
  ``DataFrame.sample(frac=1, random_state=2)``. That mix only fixes the published order.

``DataFrame.sample(n, random_state=s)`` on a frame of length L takes rows
``numpy.random.RandomState(s).choice(L, n, replace=False)`` in that order (``frac=1`` means n = L); ``sample_rows``
does exactly that, so pandas is not needed.

qid caveat, kept as published: the Base JSONs use the question hash (``qhash(question)``), the Extended JSONs the hash
of the full prompt (``qhash(question + "\n\nAnswer: ")``).
"""

from __future__ import annotations

import json
import os
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .pools import SUFFIX, read_jsonl
from .specs import SPECS, qhash

# Earlier prompt tables (same rows as the mixes), in the order the contamination exclusion list
# (pools.exclusion_rows) reads them.
VERL_TABLES = ("nltw_m_train", "nltw_m_test", "nltw_hmix_train", "nltw_hmix_test", "nltw_h_train", "nltw_h_test")


def records(rows: Sequence[Mapping[str, Any]], split: str, n: int | None = None, seed: int = 0) -> list[dict]:
    """Prompt records ``{prompt, gold, qid, spec}`` of pool or test rows; with ``n``: shuffle with Random(seed), cut
    to ``n``."""
    rows = list(rows)
    if n:
        random.Random(seed).shuffle(rows)
        rows = rows[:n]
    return [
        {"prompt": r["prompt"], "gold": str(r["answer"]), "qid": r["qid"], "spec": r.get("spec", split)} for r in rows
    ]


def sample_rows(rows: Sequence[Any], n: int, random_state: int) -> list[Any]:
    """``pandas.DataFrame(rows).sample(n=n, random_state=random_state)``, positionally, without pandas."""
    idx = np.random.RandomState(random_state).choice(len(rows), n, replace=False)
    return [rows[i] for i in idx]


def base_mix(m0_pool, m_pool, m_test, n_m0: int, n_m: int, select_seed: int = 0, shuffle_seed: int = 1):
    """-> (train records, test records) of the Base mix."""
    train = records(m0_pool, "rl_train", n_m0, select_seed) + records(m_pool, "rl_train", n_m, select_seed)
    random.Random(shuffle_seed).shuffle(train)
    return train, records(m_test, "test")


def order_mix(base_train, h_pool, h_test, n_m0: int, n_m: int, n_h: int, select_state: int = 1, shuffle_state: int = 2):
    """-> (train records, test records) of the intermediate mix that orders the Extended rows (from the Base rows)."""
    m0 = sample_rows([r for r in base_train if r["spec"] == "nltw_m0"], n_m0, select_state)
    m = sample_rows([r for r in base_train if r["spec"] == "nltw_m"], n_m, select_state)
    h = records(h_pool, "rl_train")
    if len(h) != n_h:
        raise ValueError(f"nltw_h RL pool has {len(h)} rows, expected {n_h}")
    mix = m0 + m + h
    return sample_rows(mix, len(mix), shuffle_state), records(h_test, "test")


def json_rows(recs: Sequence[Mapping[str, Any]], qid: str = "prompt") -> list[dict[str, str]]:
    """Prompt-JSON rows; ``qid="prompt"``: hash of the full prompt (Extended), ``"source"``: the row's qid (Base)."""
    if qid not in ("prompt", "source"):
        raise ValueError(qid)
    return [
        {"question": r["prompt"], "gold": r["gold"], "qid": qhash(r["prompt"]) if qid == "prompt" else r["qid"]}
        for r in recs
    ]


def third_block(pool: Sequence[Mapping[str, Any]], spec: str) -> list[dict[str, str]]:
    """Prompt-JSON rows of a third-block pool; every row must be a raw prompt of rung ``spec``."""
    params = SPECS[spec][1]
    for r in pool:
        m = r["metadata"]
        ok = params["n_people_min"] <= len(m["names"]) <= params["n_people_max"]
        ok = ok and params["deferred_min"] <= m["n_held"] <= params["deferred_max"]
        if not (r["prompt"].endswith(SUFFIX) and ok):
            raise ValueError(f"{r['qid']} is not a {spec} prompt")
    return json_rows(records(pool, "rl_train"))


def extended_mix(order_train, third: Sequence[Mapping[str, str]], test_qids: set[str], shuffle_seed: int):
    """The 8-9-person rows of the intermediate mix (in its order) + ``third``, shuffled with Random(shuffle_seed)."""
    rows = json_rows([r for r in order_train if r["spec"] in ("nltw_m0", "nltw_m")]) + list(third)
    qs = [e["qid"] for e in rows]
    if len(set(qs)) != len(qs):
        raise ValueError("duplicate prompts in the mix")
    if set(qs) & test_qids:
        raise ValueError("test questions in the mix")
    random.Random(shuffle_seed).shuffle(rows)
    return rows


def prompt_test_qids(tests: Sequence[Sequence[Mapping[str, Any]]]) -> set[str]:
    """Prompt-hash qids of test rows (the qid scheme of the Extended files)."""
    return {qhash(r["prompt"]) for rows in tests for r in rows}


def write_json(path: str | Path, rows: Any, ensure_ascii: bool = True) -> None:
    """``json.dumps(rows)`` (no newline), written atomically; refuses to overwrite."""
    path = str(path)
    if os.path.exists(path):
        raise FileExistsError(f"refusing to overwrite {path}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        f.write(json.dumps(rows, ensure_ascii=ensure_ascii))
    os.replace(path + ".tmp", path)


def build_levels(root: str | Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    """The Base and Extended prompt mixes in memory from the pools and tests under ``root``.

    ``spec`` is the ``data`` section of data_spec.json. Returns the prompt-JSON rows of every file (key: its path
    relative to ``root``) under ``"files"`` and the earlier prompt tables under ``"verl_tables"``."""
    root = Path(root)
    pools = {p["spec"]: read_jsonl(root / p["file"]) for p in spec["rl_pools"]}
    mx = spec["mixes"]
    base, order, ext = mx["base"], mx["order"], mx["extended"]
    m_test, h_test = read_jsonl(root / base["test"]), read_jsonl(root / order["test"])
    base_train, base_test = base_mix(
        pools["nltw_m0"], pools["nltw_m"], m_test, base["m0"], base["m"], base["select_seed"], base["shuffle_seed"]
    )
    order_train, order_test = order_mix(
        base_train, pools["nltw_h"], h_test, order["m0"], order["m"], order["h"], order["select_random_state"],
        order["shuffle_random_state"],
    )  # fmt: skip
    test_qids = prompt_test_qids([read_jsonl(root / t) for t in mx["contamination_tests"]])
    hb_train = extended_mix(
        order_train, third_block(read_jsonl(root / ext["third_pool"]), ext["third_spec"]), test_qids, ext["shuffle_seed"]
    )
    hb_test = json_rows(records(read_jsonl(root / ext["test"]), "test"))
    if {e["qid"] for e in hb_test} & {e["qid"] for e in hb_train}:
        raise ValueError("the Extended test overlaps its prompt mix")

    files = {
        base["train_json"]: json_rows(base_train, qid="source"),
        base["test_json"]: json_rows(base_test, qid="source"),
        ext["train_json"]: hb_train,
        ext["test_json"]: hb_test,
    }
    h_train = records(pools["nltw_h"], "rl_train")
    tables = {"nltw_m_train": base_train, "nltw_m_test": base_test, "nltw_hmix_train": order_train,
              "nltw_hmix_test": order_test, "nltw_h_train": h_train, "nltw_h_test": order_test}  # fmt: skip
    return {
        "files": files,
        # json.dumps(ensure_ascii=...) per file: the Base files were written with ensure_ascii=False (the text is ASCII)
        "ensure_ascii": {name: name not in (base["train_json"], base["test_json"]) for name in files},
        "verl_tables": {name: [r["prompt"] for r in tables[name]] for name in VERL_TABLES},
        "sizes": {"base": len(base_train), "extended": len(hb_train)},
    }


def verl_prompt_tables(root: str | Path, spec: Mapping[str, Any]) -> dict[str, list[str]]:
    """The prompts of the six earlier tables, in table order (input of ``pools.exclusion_rows``)."""
    return build_levels(root, spec)["verl_tables"]
