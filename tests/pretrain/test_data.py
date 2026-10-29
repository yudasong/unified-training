"""Corpus tokenization and the shipped shard plans."""

from pathlib import Path

import pytest

from unified_training.pretrain.indexed import read_document, read_index
from unified_training.pretrain.tokenize_corpus import iter_texts, load_plan, main, tokenize_shard

REPO = Path(__file__).resolve().parents[2]
PLANS = REPO / "scripts" / "pretrain" / "plans"


def fake_tokenize(texts):
    return [[ord(c) for c in text] + [50256] for text in texts]


def test_tokenize_shard_follows_plan_order_and_skips_null_text(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    source = tmp_path / "src"
    (source / "data" / "b").mkdir(parents=True)
    pq.write_table(pa.table({"text": ["ab", None, "c"], "other": [1, 2, 3]}), source / "a.parquet")
    (source / "data" / "b" / "x.jsonl").write_text('{"text": "d"}\n\n{"text": null}\n{"text": "ef"}\n')
    files = [["data/b/x.jsonl", 1], ["a.parquet", 1]]  # plan order, not path order
    expected = [[100, 50256], [101, 102, 50256], [97, 98, 50256], [99, 50256]]
    plan = {"text_field": "text", "dtype": "uint16", "num_shards": 1,
            "shards": [{"files": files, "documents": 4, "tokens": 10, "bin_xxh64": "", "idx_xxh64": ""}]}
    receipt = tokenize_shard(plan, 0, source, tmp_path / "out", fake_tokenize)
    prefix = str(tmp_path / "out" / "shard_000000")
    meta = read_index(prefix)
    assert [read_document(prefix, i, meta) for i in range(len(meta["lengths"]))] == expected
    assert (receipt["documents"], receipt["tokens"]) == (4, 10)
    assert receipt["identical_to_paper"] in (False, None)
    assert tokenize_shard(plan, 0, source, tmp_path / "out", None) == receipt  # receipt = done


def test_zstd_jsonl_reader(tmp_path):
    zstandard = pytest.importorskip("zstandard")
    path = tmp_path / "part.jsonl.zst"
    path.write_bytes(zstandard.ZstdCompressor().compress(b'{"text": "x"}\n{"text": "y z"}\n'))
    assert list(iter_texts(path, "text")) == ["x", "y z"]


@pytest.mark.parametrize("name,files,tokens", [("climbmix-100b", 1900, 103_784_525_443),
                                               ("dolmino-100b", 71090, 123_689_370_236),
                                               ("climbmix-400b", 6543, 357_369_866_766)])
def test_shipped_plans(name, files, tokens):
    plan = load_plan(PLANS / f"{name}.json.gz")
    assert plan["num_shards"] == len(plan["shards"]) == (256 if name == "climbmix-400b" else 64)
    assert sum(len(s["files"]) for s in plan["shards"]) == files
    assert sum(s["tokens"] for s in plan["shards"]) == tokens
    assert plan["tokenizer"] == {"model": "gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
                                 "add_eos": True, "add_bos": False}
    held_out = [plan["splits"]["valid"][0], plan["splits"]["test"][0]]
    assert held_out == ([9, 234] if name == "climbmix-400b" else [25, 23])
    assert sorted(plan["splits"]["train"] + held_out) == list(range(plan["num_shards"]))
    if name != "climbmix-400b":  # the 124M run drew 241,935,657 / 80,645,219 tokens per train shard: no cycling
        per_shard = 241_935_657 if name.startswith("climbmix") else 80_645_219
        assert min(plan["shards"][i]["tokens"] for i in plan["splits"]["train"]) > per_shard


def test_list_files_for_selected_shards(capsys):
    plan = load_plan(PLANS / "climbmix-400b.json.gz")
    main([str(PLANS / "climbmix-400b.json.gz"), "--list-files", "--shards", "103,60,5"])
    listed = capsys.readouterr().out.split()
    assert listed == [f for i in (103, 60, 5) for f, _size in plan["shards"][i]["files"]]
    assert all(name.endswith(".parquet") for name in listed)
