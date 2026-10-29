"""Evaluation outputs -> per-model scores -> two-seed means (Appendix C.4).

Scores are always computed per trained model from its per-question counts and then averaged over the two RL seeds of
a configuration; responses of different models are never pooled. A curve point (or final value) exists only when both
seeds have an evaluation at that update.

Sources:
* ``load_eval_dir(EVAL_DIR)``: the files written by ``nl_tower.evaluate``, named ``EVAL_DIR/<spec>/<name>_s<step>.json``
  for RL checkpoints and ``EVAL_DIR/<spec>/init_<init>.json`` for initializations (step 0);
* ``load_results_parquet(FILE, runs)``: the public results table (``data/evals.parquet`` of the results dataset), whose
  ``run`` column is the run id of runs.tsv.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from .passk import mean_pass_at_k

SEEDS = ("1234", "4321")
_EVAL = re.compile(r"^(?P<name>.+)_s(?P<step>\d+)\.json$")


def read_runs(path: str | Path) -> list[dict[str, str]]:
    """Rows of runs.tsv (id, setting, recipe, init, m, seed, steps, group, note)."""
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def config_of(run_id: str) -> str:
    """The configuration of a run id: the id without its ``_s<seed>`` suffix."""
    return re.sub(r"_s\d+$", "", run_id)


def load_eval_dir(eval_dir: str | Path, tests: Iterable[str] | None = None) -> list[dict]:
    """Records {run, test, step, per_q} of every evaluation file under ``eval_dir``."""
    out = []
    for spec_dir in sorted(Path(eval_dir).iterdir()):
        if not spec_dir.is_dir() or (tests is not None and spec_dir.name not in tests):
            continue
        for p in sorted(spec_dir.glob("*.json")):
            if p.name.startswith("init_"):
                name, step = p.stem, 0
            else:
                m = _EVAL.match(p.name)
                if not m:
                    continue
                name, step = m["name"], int(m["step"])
            d = json.loads(p.read_text())
            out.append({"run": name, "test": spec_dir.name, "step": step, "per_q": d["sample"]["per_q"]})
    return out


def load_results_parquet(path: str | Path, runs: list[dict[str, str]]) -> list[dict]:
    """Records of the public results table for the runs of ``runs`` (rows of other runs are dropped)."""
    import pyarrow.parquet as pq

    ids = {r["id"] for r in runs}
    t = pq.read_table(path, columns=["run", "test", "step", "per_q"]).to_pylist()
    return [
        {"run": r["run"], "test": r["test"], "step": int(r["step"]),
         "per_q": [[q["qid"], q["n"], q["correct"]] for q in r["per_q"]]}
        for r in t if r["run"] in ids
    ]  # fmt: skip


def scores(records: Iterable[dict], k: int) -> dict[tuple[str, str, int], float]:
    """{(run, test, step): pass@k} of each evaluated model."""
    return {(r["run"], r["test"], r["step"]): mean_pass_at_k(r["per_q"], k) for r in records}


def seed_means(per_model: dict[tuple[str, str, int], float], seeds: tuple[str, ...] = SEEDS) -> dict:
    """{(config, test, step): mean over seeds}, only where every seed of ``seeds`` has a score."""
    groups = defaultdict(dict)
    for (run, test, step), v in per_model.items():
        m = re.search(r"_s(\d+)$", run)
        if m:
            groups[(config_of(run), test, step)][m[1]] = v
    return {key: sum(vals[s] for s in seeds) / len(seeds) for key, vals in groups.items() if all(s in vals for s in seeds)}
