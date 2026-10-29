"""nl_tower environment: rungs, solver, traces and the evaluator's scoring rule.

CPU only, no data files. Every rung generates unique puzzles that the solver finishes without guessing, with a scorer
that accepts exactly the gold answer; generation fails loudly without the solver; the package needs neither
reasoning-gym nor torch.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from nl_tower import answers
from nl_tower.generator import NLTowerConfig, NLTowerDataset
from nl_tower.oracle import oracle_trace, solve_verbal
from nl_tower.specs import SPECS, iter_items, make_dataset, qhash
from nl_tower.traces import VARIANTS, cot_rows, question_traces

SRC = Path(__file__).resolve().parents[2] / "src"
RUNGS = ["nltw_e", "nltw_m0", "nltw_m", "nltw_h", "nltw_h9", "nltw_hb", "nltw_hb12"]


def _tup(x):
    return tuple(_tup(y) for y in x) if isinstance(x, list) else x


@pytest.mark.parametrize("spec", RUNGS)
def test_rung(spec):
    n = 3 if spec in ("nltw_hb12", "nltw_hb") else 10
    ds = make_dataset(spec, size=n, seed=4242)
    params = SPECS[spec][1]
    seen = set()
    for i in range(n):
        e = ds[i]
        m = e["metadata"]
        h = qhash(e["question"])
        assert h not in seen
        seen.add(h)
        assert params["n_people_min"] <= len(m["names"]) <= params["n_people_max"]
        assert params["deferred_min"] <= m["n_held"] <= params["deferred_max"]
        res = solve_verbal(m["names"], m["domain"], m["friends"], [tuple(_tup(c)) for c in m["clues"]])
        assert res["solution"] == m["solution"]
        assert ds.score_answer(e["answer"], e) == 1.0
        wrong = e["answer"].replace(f"={m['solution'][0]}", f"={m['solution'][0] + 100}", 1)
        assert ds.score_answer(wrong, e) == 0.0
        assert ds.score_answer(None, e) == 0.0
        _, trace = oracle_trace(m, e["answer"], style=i % 2, order_seed=i)
        assert trace is not None and trace.endswith("\\boxed{" + e["answer"] + "}")


def test_generation_fails_without_oracle(monkeypatch):
    monkeypatch.setitem(sys.modules, "nl_tower.oracle", None)  # `from .oracle import` now raises
    ds = make_dataset("nltw_m", size=1, seed=1)
    with pytest.raises(ImportError, match="refusing to generate"):
        ds[0]


def test_dataset_protocol():
    """The local base class keeps reasoning-gym's semantics: validation, len, iteration, seed default."""
    ds = make_dataset("nltw_e", size=3, seed=5)
    assert len(ds) == 3 and ds.seed == 5
    assert [e["question"] for e in ds] == [ds[i]["question"] for i in range(3)]
    assert ds[1]["metadata"]["source_index"] == 1
    unseeded = NLTowerDataset(NLTowerConfig(**SPECS["nltw_e"][1], size=2))
    assert isinstance(unseeded.seed, int)
    with pytest.raises(AssertionError):
        NLTowerDataset(NLTowerConfig(n_people_min=13, n_people_max=13))


def test_parallel_items_equal_sequential(monkeypatch):
    ds = make_dataset("nltw_e", size=64, seed=11)
    monkeypatch.setenv("RG_WORKERS", "1")
    sequential = [e["question"] for e in iter_items(ds, 64)]
    monkeypatch.setenv("RG_WORKERS", "2")
    assert [e["question"] for e in iter_items(ds, 64)] == sequential


def test_imports_without_reasoning_gym_or_torch():
    code = textwrap.dedent(
        """
        import sys

        class Block:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in ("reasoning_gym", "torch", "vllm", "pandas"):
                    raise ImportError("blocked " + name)
                return None

        sys.meta_path.insert(0, Block())
        from nl_tower import answers, generator, oracle, specs, traces  # noqa: F401
        print(specs.qhash(specs.make_dataset("nltw_e", size=1, seed=20260921)[0]["question"]))
        """
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(SRC), os.environ.get("PYTHONPATH", "")]))
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "409f62954b4b51bb"  # first question of the published easy test set


def test_registration_with_reasoning_gym():
    reasoning_gym = pytest.importorskip("reasoning_gym")
    from reasoning_gym.dataset import ProceduralDataset as RGProceduralDataset

    from nl_tower.generator import register_with_reasoning_gym

    register_with_reasoning_gym()
    register_with_reasoning_gym()  # idempotent
    ds = reasoning_gym.create_dataset("nl_tower", size=1, seed=20260921, **SPECS["nltw_e"][1])
    assert isinstance(ds, NLTowerDataset) and isinstance(ds, RGProceduralDataset)
    assert ds[0]["question"] == make_dataset("nltw_e", size=1, seed=20260921)[0]["question"]


def _pool_rows(spec, n, seed):
    ds = make_dataset(spec, size=n, seed=seed)
    rows = []
    for e in ds:
        rows.append(
            json.loads(
                json.dumps(
                    {
                        "qid": qhash(e["question"]),
                        "prompt": e["question"] + "\n\nAnswer: ",
                        "answer": e["answer"],
                        "metadata": e["metadata"],
                    }
                )
            )
        )
    return rows


def test_cot_rows():
    pool = _pool_rows("nltw_e", 4, 3)
    out, summary = cot_rows(pool, k=4, seed=7)
    again, _ = cot_rows(pool, k=4, seed=7)
    assert out == again and summary["failed"] == 0 and summary["rows"] == len(out)
    by_q = {}
    for r in out:
        assert set(r) == {"qid", "prompt", "target", "answer"}
        assert r["target"].endswith("\\boxed{" + r["answer"] + "}")
        by_q.setdefault(r["qid"], []).append(r["target"])
    assert set(by_q) == {r["qid"] for r in pool}
    for traces in by_q.values():
        assert 1 <= len(traces) <= 4 and len(set(traces)) == len(traces)
    assert len(question_traces(pool[0], k=2)) <= 2 and len(VARIANTS) == 8
    first_two, summary = cot_rows(pool, k=4, questions=2, seed=7)
    assert summary["questions"] == 2 and len({r["qid"] for r in first_two}) == 2


# The box parser of the evaluator that scored the paper's runs, written out independently.
def _old_parse(text):
    def escaped(t, pos):
        n = 0
        pos -= 1
        while pos >= 0 and t[pos] == "\\":
            n += 1
            pos -= 1
        return n % 2 == 1

    start = text.rfind(r"\boxed{")
    if start < 0:
        return None
    depth = 1
    for pos in range(start + len(r"\boxed{"), len(text)):
        if text[pos] == "{" and not escaped(text, pos):
            depth += 1
        elif text[pos] == "}" and not escaped(text, pos):
            depth -= 1
            if depth == 0:
                return text[start + len(r"\boxed{") : pos].strip() or None
    return None


@pytest.mark.parametrize(
    "text",
    [
        "So Ava = 12.\n\\boxed{Ava=12, Noah=9}",
        "\\boxed{Ava=1} then \\boxed{ Ava=2 }",
        "\\boxed{\\text{Ava}=3}",
        "\\boxed{a{b}c}",
        "\\boxed{a\\{b}",
        "\\boxed{a\\\\{b}}",
        "\\boxed{}",
        "\\boxed{   }",
        "\\boxed{Ava=1",
        "\\boxed{Ava=1} \\boxed{Noah=",
        "no box at all",
        "",
    ],
)
def test_last_boxed_answer_matches_old_parser(text):
    assert answers.last_boxed_answer(text) == _old_parse(text)


def test_normalize():
    assert answers.normalize(" $Ava=1, Noah=2$ ") == "Ava=1, Noah=2"
    assert answers.normalize("\\(Ava=1\\)") == "Ava=1"
    assert answers.normalize("\\text{Ava}=1,\\,\\text{Noah}=2") == "Ava=1,Noah=2"
    assert answers.normalize("Ava=1,\\ Noah=2") == "Ava=1, Noah=2"


def test_eval_correct_and_differences_from_the_rl_reward():
    ds = make_dataset("nltw_e", size=1, seed=1)
    entry = {"answer": "Ava=12, Noah=9"}
    assert answers.eval_correct(ds, "Noah=9, Ava=12", entry)
    assert answers.eval_correct(ds, "$Ava=12, Noah=9$", entry)
    assert answers.eval_correct(ds, "\\text{Ava}=12, \\text{Noah}=9", entry)
    assert not answers.eval_correct(ds, None, entry)
    assert not answers.eval_correct(ds, "Ava=12", entry)
    assert not answers.eval_correct(ds, "Ava=12, Noah=9, Mia=3", entry)
    # Documented differences: the evaluator is case-sensitive and compares numbers as strings; the RL reward is not.
    for answer in ("ava=12, noah=9", "Ava=012, Noah=9"):
        assert not answers.eval_correct(ds, answer, entry)
    math_reward = pytest.importorskip("unified_training.math_reward")
    for answer in ("ava=12, noah=9", "Ava=012, Noah=9"):
        assert math_reward.nl_tower_correct("\\boxed{" + answer + "}", entry["answer"])
