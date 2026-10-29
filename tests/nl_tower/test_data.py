"""nl_tower data builders: pools, the pandas-free mix rule, the two RL mixes, the contamination audit, the artifact
record and the prepare_data stages on a tiny spec. CPU only."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

from nl_tower import demos, manifest, mixes, paths, pools, prepare_data
from nl_tower.specs import qhash

SPEC = Path(prepare_data.SPEC)
RECORD = Path(prepare_data.RECORD)
SUFFIX = "\n\nAnswer: "


def _data_spec():
    return json.loads(SPEC.read_text())["data"]


def test_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("NL_TOWER_WORKDIR", str(tmp_path / "w"))
    assert paths.workdir() == tmp_path / "w"
    assert paths.get("rl_prompts") == tmp_path / "w" / "results" / "rl_prompts"
    monkeypatch.delenv("NL_TOWER_WORKDIR")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    assert paths.workdir() == tmp_path / "nl_tower"
    assert paths.get("unified_data", "/x") == Path("/x/unified_data")


def test_pool_rows_split_and_exclusion():
    test, pool, n_excl = pools.pool_rows("nltw_e", 3, 2, seed=5)
    assert (len(test), len(pool), n_excl) == (2, 3, 0)
    for r in test + pool:
        assert list(r) == ["qid", "prompt", "question", "answer", "metadata"]
        assert r["qid"] == qhash(r["question"]) and r["prompt"] == r["question"] + SUFFIX
        assert isinstance(r["metadata"]["clues"][0], list)  # tuples became JSON lists
    assert len({r["qid"] for r in test + pool}) == 5
    test2, pool2, n_excl2 = pools.pool_rows("nltw_e", 3, 2, seed=5, exclude={test[0]["qid"]})
    assert n_excl2 == 1 and test2[0] == test[1] and test2[1] == pool[0]


def test_rl_pool_rows():
    rows = pools.rl_pool_rows("nltw_e", 3, seed=777)
    assert len(rows) == 3
    for r in rows:
        assert list(r) == ["spec", "task", "qid", "prompt", "question", "answer", "metadata"]
        assert (r["spec"], r["task"]) == ("nltw_e", "nl_tower")
        assert r["prompt"] == r["question"].strip() + SUFFIX and r["qid"] == qhash(r["question"])


def test_write_read_jsonl_and_mix(tmp_path):
    rows = [{"qid": str(i), "x": [i, (i, i)]} for i in range(5)]
    out = tmp_path / "a" / "rows.jsonl"
    pools.write_jsonl(out, rows)
    assert out.read_text() == "".join(json.dumps(r) + "\n" for r in rows)
    assert not (tmp_path / "a" / "rows.jsonl.tmp").exists()
    mixed = pools.mix_rows([rows[:2], rows[2:]], seed=3)
    want = list(rows)
    random.Random(3).shuffle(want)
    assert mixed == want


@pytest.mark.parametrize("length,n,state", [(50, 50, 2), (37, 11, 1), (1000, 333, 7), (31666, 31666, 2)])
def test_sample_rows_equals_pandas(length, n, state):
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"i": np.arange(length)})
    want = frame.sample(n=n, random_state=state)["i"].tolist()
    assert mixes.sample_rows(list(range(length)), n, state) == want
    if n == length:
        assert frame.sample(frac=1, random_state=state)["i"].tolist() == want


# ---------------------------------------------------------------- synthetic work directory for the level mixes
def _row(spec, i, people=9, held=4, answer=None):
    q = f"{spec} question {i}"
    answer = answer or ", ".join(f"P{j}={j + 1}" for j in range(people))
    return {"spec": spec, "task": "nl_tower", "qid": qhash(q), "prompt": q + SUFFIX, "question": q, "answer": answer,
            "metadata": {"names": [f"P{j}" for j in range(people)], "n_held": held}}  # fmt: skip


def _write_synthetic(root):
    def w(rel, rows):
        pools.write_jsonl(root / rel, rows)

    for sp in ("nltw_e", "nltw_m0", "nltw_m"):
        w(f"data/{sp}/corpus_tw1/test.jsonl", [_row(f"{sp}-test", i) for i in range(4)])
        w(f"data/{sp}/corpus_tw1/pool.jsonl", [_row(f"{sp}-pool", i) for i in range(5)])
    w("data/nltw_h/corpus_tw1/test.jsonl", [_row("nltw_h-test", i, people=9 + i % 2) for i in range(6)])
    w("data/nltw_h/corpus_tw1/pool.jsonl", [_row("nltw_h-pool", i) for i in range(5)])
    w("data/nltw_h9/corpus_tw1/pool.jsonl", [_row("nltw_h9", i, people=9, held=4 + i % 3) for i in range(4)])
    w("data/nltw_hb/corpus_tw1/pool.jsonl", [_row("nltw_hb", i, people=10 + i % 2, held=2 + i % 3) for i in range(4)])
    w("data/nltw_hb/corpus_tw1/test.jsonl", [_row("nltw_hb-test", i, people=10) for i in range(3)])
    for sp, n in (("nltw_m0", 8), ("nltw_m", 10), ("nltw_h", 5)):
        w(f"results/rl_prompts/{sp}.jsonl", [dict(_row(sp, i), spec=sp) for i in range(n)])
    spec = _data_spec()
    spec["mixes"]["base"].update(m0=6, m=7)
    spec["mixes"]["order"].update(m0=4, m=5, h=5)
    return spec


def test_levels_on_synthetic_pools(tmp_path):
    spec = _write_synthetic(tmp_path)
    levels = mixes.build_levels(tmp_path, spec)
    assert levels == mixes.build_levels(tmp_path, spec)  # deterministic
    f = levels["files"]
    l2, ext = f["unified_data/nltw_m_train.json"], f["unified_data/nltw_hbmix_train.json"]
    assert sorted(f) == sorted(["unified_data/nltw_m_train.json", "unified_data/nltw_m_test.json",
                                "unified_data/nltw_hbmix_train.json", "unified_data/nltw_hbmix_test.json"])
    assert levels["sizes"] == {"base": 13, "extended": 13}
    for name, rows in f.items():
        assert all(list(r) == ["question", "gold", "qid"] and r["question"].endswith(SUFFIX) for r in rows), name
        assert len({r["qid"] for r in rows}) == len(rows), name
    assert all(r["qid"] == qhash(r["question"][: -len(SUFFIX)]) for r in l2)  # Base: question hash
    assert all(r["qid"] == qhash(r["question"]) for r in ext)  # Extended: prompt hash
    tests = {qhash(json.loads(line)["prompt"]) for sp in ("nltw_e", "nltw_m0", "nltw_m", "nltw_h")
             for line in (tmp_path / f"data/{sp}/corpus_tw1/test.jsonl").read_text().splitlines()}  # fmt: skip
    assert not ({r["qid"] for r in ext} & tests)
    assert sum(r["question"].startswith("nltw_hb ") for r in ext) == 4  # the 10-11-person block
    assert not any(r["question"].startswith("nltw_h ") for r in ext)  # the intermediate block is not in the Extended mix
    assert len(f["unified_data/nltw_hbmix_test.json"]) == 3
    assert list(levels["verl_tables"]) == list(mixes.VERL_TABLES)
    assert [len(levels["verl_tables"][t]) for t in mixes.VERL_TABLES] == [13, 4, 14, 6, 5, 6]
    assert levels["ensure_ascii"]["unified_data/nltw_hbmix_train.json"]
    assert not levels["ensure_ascii"]["unified_data/nltw_m_train.json"]


def test_third_block_rejects_wrong_rung(tmp_path):
    with pytest.raises(ValueError, match="not a nltw_hb prompt"):
        mixes.third_block([_row("nltw_hb", 0, people=9)], "nltw_hb")


def test_exclusion_rows_and_audit(tmp_path):
    spec = _write_synthetic(tmp_path)
    rows = pools.exclusion_rows(tmp_path, mixes.verl_prompt_tables(tmp_path, spec))
    assert [r["src"] for r in rows[:5]] == ["nltw_e/corpus_tw1/test.jsonl"] * 4 + ["nltw_e/corpus_tw1/pool.jsonl"]
    assert rows[-1]["src"] == "rl/data/nltw_h_test"
    assert len(rows) == 4 * (4 + 5) + 2 + 23 + (13 + 4 + 14 + 6 + 5 + 6)
    assert all(r["qid"] == qhash(r["question"]) and not r["question"].endswith(SUFFIX) for r in rows)
    clean = [{"qid": qhash("fresh"), "prompt": "fresh" + SUFFIX, "target": "t", "answer": "a"}]
    dirty = clean + [{"qid": rows[0]["qid"], "prompt": rows[0]["question"] + SUFFIX, "target": "t", "answer": "a"}]
    for name, corpus, overlap in (("clean", clean, 0), ("dirty", dirty, 2)):
        pools.write_jsonl(tmp_path / f"data/{name}.jsonl", corpus)
        report = pools.audit(tmp_path / f"data/{name}.jsonl", rows, tmp_path)
        assert report["corpus"] == f"data/{name}.jsonl" and report["new"]["total_overlap"] == overlap


def test_demo_jsonl(tmp_path):
    cot = [{"qid": "q1", "prompt": "P1" + SUFFIX, "target": "line\n\\boxed{A=1}", "answer": "A=1"}]
    pools.write_jsonl(tmp_path / "cot.jsonl", cot)
    assert demos.write_demo_jsonl(tmp_path / "cot.jsonl", tmp_path / "demos.jsonl") == 1
    assert (tmp_path / "demos.jsonl").read_text() == json.dumps(
        {"prompt": "P1" + SUFFIX, "response": cot[0]["target"]}
    ) + "\n"
    with pytest.raises(FileExistsError):
        demos.write_demo_jsonl(tmp_path / "cot.jsonl", tmp_path / "demos.jsonl")


def test_manifest_check(tmp_path):
    (tmp_path / "a.jsonl").write_text('{"qid": "x"}\n')
    (tmp_path / "b.json").write_text('[{"qid": "y"}]')
    (tmp_path / "audit.json").write_text(json.dumps({"corpus": "c", "new": {"rows": 2, "total_overlap": 0}}))
    (tmp_path / "anchor.json").write_text(json.dumps({"n_episodes": 3, "tokenizer": "/somewhere"}))
    a = manifest.describe(tmp_path / "a.jsonl")
    assert (a["rows"], a["first_qids"], a["bytes"]) == (1, ["x"], 13)
    assert manifest.describe(tmp_path / "b.json")["first_qids"] == ["y"]
    record = {"groups": {"g": [
        {"path": "a.jsonl", "md5": a["md5"]},
        {"path": "b.json", "md5": "0" * 32},
        {"path": "missing.jsonl", "md5": "0" * 32},
        {"path": "audit.json", "check": "audit_result", "md5": "0" * 32,
         "audit_result": {"rows": 2, "total_overlap": 0}},
        {"path": "anchor.json", "check": "anchor_manifest", "md5": "0" * 32, "expected": {"n_episodes": 3}},
    ], "other": [{"path": "a.jsonl", "md5": "0" * 32}]}}  # fmt: skip
    got = [s for s, *_ in manifest.check(record, tmp_path, ["g"])]
    assert got == ["SAME", "DIFF", "MISSING", "SAME", "SAME"]


def test_spec_and_record_agree():
    spec, record = _data_spec(), json.loads(RECORD.read_text())
    recorded = {it["path"]: (g, it) for g, items in record["groups"].items() for it in items}
    built = [
        f"{q['dir']}/{f}"
        for q in spec["question_sets"]
        for f, n in (("test.jsonl", q["n_test"]), ("pool.jsonl", q["n_pool"]))
        if n
    ]
    built += [p["file"] for p in spec["rl_pools"]]
    built += [
        lv[k]
        for name, lv in spec["mixes"].items()
        if name != "contamination_tests"
        for k in ("train_json", "test_json")
        if k in lv
    ]
    c = spec["corpus"]
    built += [c["exclusion_list"], c["audit_exclusion_list"], c["corpus"], c["audit"]]
    built += [f"data/{r}/corpus_{p}/pool.jsonl" for r in c["rungs"] for p in ("tw1L", "tw1Xnew", c["tag"])]
    built += [f"data/{r}/corpus_{c['tag']}/cot_k{c['traces']['k']}.jsonl" for r in c["rungs"]]  # fmt: skip
    assert len(built) == 31 and len(set(built)) == 31  # 6 tests, 6 pools, 3 RL pools, 12 corpus files, 4 mixes
    for p in built:
        group, it = recorded[p]
        assert group in ("tests", "pools", "rl_pools", "corpus", "mixes"), p
        assert it.get("check", "md5") in ("md5", "audit_result"), p
    rebuilt = {p for p, (g, it) in recorded.items() if g in ("tests", "pools", "rl_pools", "corpus", "mixes")}
    assert rebuilt == set(built)
    for q in spec["question_sets"]:
        if q["n_test"]:
            assert recorded[f"{q['dir']}/test.jsonl"][1]["rows"] == q["n_test"]


def _tiny_spec(path):
    spec = json.loads(SPEC.read_text())
    d = spec["data"]
    sizes = {"nltw_e": (2, 3), "nltw_m0": (2, 3), "nltw_m": (2, 3), "nltw_h": (2, 3), "nltw_h9": (0, 2),
             "nltw_hb": (2, 2), "nltw_hb12": (1, 0)}  # fmt: skip
    for qs in d["question_sets"]:
        qs["n_test"], qs["n_pool"] = sizes[qs["spec"]]
    for rp, n in zip(d["rl_pools"], (4, 5, 3)):
        rp["n"] = n
    d["mixes"]["base"].update(m0=4, m=4)
    d["mixes"]["order"].update(m0=2, m=3, h=3)
    for qp in d["corpus"]["question_pools"]:
        qp["n"] = 2
    path.write_text(json.dumps(spec))
    return d


def _run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["prepare_data", *map(str, args)])
    prepare_data.main()


@pytest.mark.integration
def test_prepare_script_end_to_end(tmp_path, monkeypatch):
    """All stages on a tiny spec (about 1-2 minutes: the hard rungs' puzzles take seconds each)."""
    monkeypatch.setenv("RG_WORKERS", "4")
    spec = _tiny_spec(tmp_path / "spec.json")
    work = tmp_path / "work"
    _run(
        monkeypatch,
        "--workdir",
        work,
        "--spec",
        tmp_path / "spec.json",
        "tests",
        "rl-pools",
        "mixes",
        "corpus",
    )
    corpus = pools.read_jsonl(work / spec["corpus"]["corpus"])
    cot = [pools.read_jsonl(work / f"data/{r}/corpus_tw1X/cot_k4.jsonl") for r in spec["corpus"]["rungs"]]
    assert len({r["qid"] for r in corpus}) == 8 and len(corpus) == sum(map(len, cot)) <= 4 * 2 * 4
    audit = json.loads((work / spec["corpus"]["audit"]).read_text())
    assert audit["corpus"] == spec["corpus"]["corpus"] and audit["new"]["total_overlap"] == 0
    built = sorted(p for p in work.rglob("*") if p.is_file())
    assert len(built) == 31 and not [p for p in built if p.name.endswith(".tmp")]
    record = {"groups": {"all": [{"path": str(p.relative_to(work)), "md5": manifest.md5(p)} for p in built]}}
    (tmp_path / "record.json").write_text(json.dumps(record))
    before = {p: p.stat().st_mtime_ns for p in built}
    _run(
        monkeypatch,
        "--workdir",
        work,
        "--spec",
        tmp_path / "spec.json",
        "tests",
        "rl-pools",
        "mixes",
        "corpus",
    )
    assert {p: p.stat().st_mtime_ns for p in built} == before  # existing outputs are kept
    _run(monkeypatch, "--workdir", work, "--record", tmp_path / "record.json", "verify", "--groups", "all")
    (work / spec["mixes"]["base"]["train_json"]).write_text("[]")
    with pytest.raises(SystemExit) as e:
        _run(monkeypatch, "--workdir", work, "--record", tmp_path / "record.json", "verify", "--groups", "all")
    assert e.value.code == 1


WORDS = "<unk> <eos> Mia lives on floor 5 . Answer : So Mia=5 \\boxed{Mia=5}".split()


def test_demos_stage(tmp_path, monkeypatch):
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    episodes = pytest.importorskip("unified_training.episodes")  # the anchor builder belongs to the trainer package
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="<unk>"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>")
    tok.save_pretrained(tmp_path / "tok")
    work = tmp_path / "work"
    cot = [{"qid": "q", "prompt": "Mia lives on floor 5 . Answer :", "target": "So Mia=5 \\boxed{Mia=5}",
            "answer": "Mia=5"}] * 3
    pools.write_jsonl(work / "data/nltw_m/corpus_tw1X/cot.jsonl", cot)
    _run(monkeypatch, "--workdir", work, "--tokenizer", tmp_path / "tok", "demos")
    anchor = episodes.load_round(work / "unified_data/anchors/demos_tw1X_resp")
    assert anchor.manifest["n_episodes"] == 3 and anchor.manifest["mask"] == "response"
    tokens, weights, row = anchor.episode(2)
    assert int(row["prompt_len"]) == 8  # 8 prompt words; 3 response words + EOS carry the loss
    assert weights[:8].tolist() == [0.0] * 8 and weights[8:].tolist() == [1.0] * 4
    assert tokens.tolist()[-1] == WORDS.index("<eos>")
    _run(monkeypatch, "--workdir", work, "--tokenizer", tmp_path / "tok", "demos")  # kept, no error
