# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Midtraining-corpus pipeline of Sec. 5.2: build_document_anchor -> audit -> sanitize -> holdout.

On a tiny 24-component corpus the release tools must reproduce the payload hashes that the original
scripts (the builder whose sha256 is recorded in the paper corpus's manifest, and its sanitize /
holdout companions) produce on the same input, and the manifest must carry the paper's settings.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from unified_training.corpus_mix import OLMO3_COMPONENT_WEIGHTS  # noqa: E402
from unified_training.episodes import load_round  # noqa: E402
from unified_training.tools import (  # noqa: E402
    audit_benchmark_overlap,
    build_document_anchor,
    build_midtrain_holdout,
    sanitize_anchor,
)

QUESTIONS = [
    "which planet in the solar system has the largest number of known moons today according to the most recent survey",
    "how many grams of salt are needed to make one liter of a saline solution used in hospitals for patients",
    "what is the boiling point of water at sea level measured in degrees celsius under standard atmospheric pressure",
]
SETTINGS = dict(ANCHOR_TOKENS="6000", ANCHOR_WINDOW="33", ANCHOR_SOURCE_TOKENS="4000", ANCHOR_SAMPLE_MARGIN="1.4",
                ANCHOR_SEED="1234", ANCHOR_BATCH_ROWS="64", ANCHOR_GLOB="train-*.parquet")
# sha256 of (index.npy, tokens.bin, weights.bin) written by the original scripts on make_corpus().
GOLDEN = {
    'raw': [
        'aea18948b8096afd1f48baa288c2c93c7af7b1a79e7c3c0758aa09152452f078',
        'a4e7a323e8325394fda08d3022e3edfa028454140db7e915d80815657f8884be',
        'ea2d1d7c0065dddd9e3043ba661322ffe25b443e16e68135140a07b9505ea58a',
    ],
    'clean': [
        '35084faa761276f5de30a940fd1c827ac80244edb628291cc4f1e1197b032aca',
        '4d42cebc54671e7f9e6181b4e887b8dec816fe98778a784e44ae80da0e041317',
        'e5838b792bee67163ac767f02ba120542cef9ceb5ba0d339f4c30c54eac7d370',
    ],
    'holdout': [
        'a3148aa42b22a38ae907e8acbd3de8f3061b1070c3115ca05d2639ab22007509',
        'abdfb3d5d3815a89c8368231ac1e83b4e1afc818dc4676d183723f717f7610ac',
        '4faa35c97003cf4a00666189f3e19384d15a5e1d68234e4030e4d821e3c3302c',
    ],
}


def make_corpus(out: Path) -> None:
    import transformers

    words = ["<pad>", "<eos>", "<unk>"] + [f"w{i}" for i in range(200)]
    for w in " ".join(QUESTIONS).split():
        if w not in words:
            words.append(w)
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    transformers.PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>",
                                         pad_token="<pad>").save_pretrained(out / "tok")
    rng = np.random.default_rng(0)
    (out / "src").mkdir()
    components = sorted(OLMO3_COMPONENT_WEIGHTS) + ["unused-extra-component"]
    for shard in range(3):
        comp_col, text_col = [], []
        for c in components:
            for _ in range(int(rng.integers(30, 60))):
                text = " ".join(f"w{int(x)}" for x in rng.integers(0, 200, size=int(rng.integers(1, 60))))
                if rng.random() < 0.03:
                    prefix = " ".join(f"w{int(x)}" for x in rng.integers(0, 200, size=int(rng.integers(1, 6))))
                    text = prefix + " " + QUESTIONS[int(rng.integers(len(QUESTIONS)))] + " w1"
                if rng.random() < 0.03:
                    text = "  "
                comp_col.append(c)
                text_col.append(text)
        order = rng.permutation(len(comp_col))
        pq.write_table(pa.table({"component": [comp_col[i] for i in order], "text": [text_col[i] for i in order]}),
                       out / "src" / f"train-{shard:05d}.parquet", row_group_size=97)
    with (out / "bench.jsonl").open("w") as f:
        for i, q in enumerate(QUESTIONS + ["a question that never appears anywhere in the corpus at all w3 w4"]):
            f.write(json.dumps({"doc": {"id": f"q{i}", "question": q}}) + "\n")


def payload(d: Path) -> list[str]:
    return [hashlib.sha256((d / f).read_bytes()).hexdigest() for f in ("index.npy", "tokens.bin", "weights.bin")]


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    out = tmp_path_factory.mktemp("anchor")
    make_corpus(out)
    env = dict(SETTINGS, ANCHOR_SRC=str(out / "src"), ANCHOR_OUT=str(out / "raw"), ANCHOR_TOKENIZER=str(out / "tok"))
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        build_document_anchor.main()
    finally:
        for k, v in saved.items():
            os.environ.pop(k) if v is None else os.environ.__setitem__(k, v)
    reports = []
    for mode, extra in (("exact", []), ("fp16", ["--min-match-tokens", "16", "--max-match-tokens", "16"])):
        audit_benchmark_overlap.main(["--tokens", str(out / "raw/tokens.bin"), "--index", str(out / "raw/index.npy"),
                                      "--tokenizer", str(out / "tok"), "--samples", str(out / "bench.jsonl"),
                                      "--output", str(out / f"{mode}.json"), "--max-matches", "1000",
                                      "--fail-on-match-limit", *extra])
        reports += ["--report", str(out / f"{mode}.json")]
    sanitize_anchor.main(["--source", str(out / "raw"), "--output", str(out / "clean"), *reports,
                          "--suite", "broad16-gsm8k", "--split", "test", "--seed", "1234"])
    clean = json.loads((out / "clean/manifest.json").read_text())
    build_midtrain_holdout.main(["prepare", "--anchor-dir", str(out / "clean"), "--output-dir", str(out / "holdout"),
                                 "--expected-manifest-sha256",
                                 hashlib.sha256((out / "clean/manifest.json").read_bytes()).hexdigest(),
                                 "--expected-anchor-windows", str(clean["n_episodes"]), "--expected-window", "33",
                                 "--max-training-rows", "100", "--max-anchor-row-fraction", "0.5",
                                 "--permutation-seed", "1235", "--eval-windows", "16"])
    return out


def test_boundary_anchor_has_the_paper_settings(pipeline):
    m = json.loads((pipeline / "raw/manifest.json").read_text())
    assert (m["kind"], m["format"], m["variant"]) == ("anchor", "document_aware_raw", "boundary")
    assert m["component_mix_profile"] == "olmo3-commoncrawl5-docaware-boundary"
    assert m["mix_mode"] == "component_supervised_prediction_tokens"
    assert m["document_boundary_policy"] == "one_document_per_episode; split_only_within_document"
    assert (m["window"], m["component_seed"], m["component_sample_margin"]) == (33, 1234, 1.4)
    assert m["chat_replacement"] is None and m["reasoning_sft_replacement"] is None
    assert m["component_weights"] == build_document_anchor.component_profile()
    assert len(m["component_weights"]) == 24 and "unused-extra-component" not in m["component_weights"]
    for c, target in m["component_target_supervised_tokens"].items():
        assert m["component_realized_supervised_tokens"][c] >= target
    rnd = load_round(pipeline / "raw")
    for i in range(len(rnd.index)):
        tokens, weights, row = rnd.episode(i)
        assert int(row["prompt_len"]) == 1 and len(tokens) <= 33
        assert weights[0] == 0.0 and np.all(weights[1:] == 1.0)


def test_audits_find_the_planted_questions_and_sanitize_replaces_them(pipeline):
    for mode in ("exact", "fp16"):
        report = json.loads((pipeline / f"{mode}.json").read_text())
        assert report["n_matching_examples"] == 3 and report["n_matching_occurrences"] > 0
    clean = json.loads((pipeline / "clean/manifest.json").read_text())
    d = clean["decontamination"]
    assert d["method"] == "episode_aware_same_component_replacement" and d["n_replaced_episodes"] > 0
    assert clean["n_episodes"] == json.loads((pipeline / "raw/manifest.json").read_text())["n_episodes"]
    holdout = json.loads((pipeline / "holdout/manifest.json").read_text())
    assert holdout["kind"] == "midtrain_schedule_holdout" and holdout["eval_windows"] == 16


@pytest.mark.parametrize("stage", ["raw", "clean", "holdout"])
def test_payloads_match_the_original_scripts(pipeline, stage):
    assert payload(pipeline / stage) == GOLDEN[stage]
