# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""unified_training.tools.build_pair_anchor: JSONL {prompt, response} pairs -> one anchor round."""

from __future__ import annotations

import hashlib
import json
import runpy

import numpy as np
import pytest

from unified_training.episodes import load_round

WORDS = "<unk> <eos> Mia Noah lives on floor 5 9 . Answer : Since Mia=5 , Noah=9 so".split()
PAIRS = [
    {"prompt": "Mia lives on floor 5 . Answer :", "response": "Since Mia lives on floor 5 , Mia=5"},
    {"prompt": "Noah lives on floor 9 . Answer :", "response": "so Noah=9"},
]


@pytest.fixture
def tokenizer_dir(tmp_path):
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="<unk>"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>")
    tok.save_pretrained(tmp_path / "tok")
    return tmp_path / "tok"


def _build(monkeypatch, tmp_path, tokenizer_dir, **env):
    source = tmp_path / "pairs.jsonl"
    source.write_text("".join(json.dumps(p) + "\n" for p in PAIRS))
    out = tmp_path / f"anchor_{env.get('ANCHOR_MASK', 'default')}"
    monkeypatch.setenv("ANCHOR_JSONL", str(source))
    monkeypatch.setenv("ANCHOR_OUT", str(out))
    monkeypatch.setenv("ANCHOR_TOKENIZER", str(tokenizer_dir))
    for key in ("ANCHOR_MASK", "ANCHOR_MAX_PROMPT", "ANCHOR_MAX_RESPONSE", "ANCHOR_WEIGHT"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    runpy.run_module("unified_training.tools.build_pair_anchor", run_name="__main__", alter_sys=False)
    return load_round(out)


def _ids(text: str) -> list[int]:
    return [WORDS.index(word) for word in text.split()]


def _check_round(anchor, *, mask: str, weight: float) -> None:
    eos = WORDS.index("<eos>")
    all_tokens = []
    for i, pair in enumerate(PAIRS):
        tokens, weights, row = anchor.episode(i)
        prompt_ids, response_ids = _ids(pair["prompt"]), _ids(pair["response"])
        assert tokens.tolist() == prompt_ids + response_ids + [eos]
        prompt_len = len(prompt_ids) if mask == "response" else 1
        assert int(row["prompt_len"]) == prompt_len and int(row["total_len"]) == len(tokens)
        assert weights[:prompt_len].tolist() == [0.0] * prompt_len
        assert weights[prompt_len:].tolist() == [weight] * (len(tokens) - prompt_len)
        assert float(row["weight"]) == weight and float(row["reward"]) == 0.0
        assert int(row["policy_version"]) == -1 and int(row["group_id"]) == i
        all_tokens.append(tokens.astype(np.int32))

    manifest = anchor.manifest
    n_prompt = sum(len(_ids(p["prompt"])) for p in PAIRS)
    n_response = sum(len(_ids(p["response"])) + 1 for p in PAIRS)
    assert manifest["kind"] == "anchor" and manifest["format"] == "pairs" and manifest["mask"] == mask
    assert manifest["eos_token_id"] == eos and manifest["anchor_weight"] == weight
    assert manifest["max_prompt_tokens"] == 640 and manifest["max_response_tokens"] == 1280
    assert manifest["n_prompt_tokens"] == n_prompt and manifest["n_response_tokens_incl_eos"] == n_response
    assert manifest["n_episodes"] == len(PAIRS) and manifest["n_tokens"] == n_prompt + n_response
    supervised = n_response if mask == "response" else n_prompt + n_response - len(PAIRS)
    assert manifest["n_supervised_tokens"] == supervised
    assert manifest["tokens_sha256"] == hashlib.sha256(np.concatenate(all_tokens).tobytes()).hexdigest()


def test_default_mask_supervises_the_response_and_eos_only(monkeypatch, tmp_path, tokenizer_dir):
    anchor = _build(monkeypatch, tmp_path, tokenizer_dir, ANCHOR_WEIGHT="0.5")
    _check_round(anchor, mask="response", weight=0.5)


def test_all_mask_supervises_every_predictable_token(monkeypatch, tmp_path, tokenizer_dir):
    anchor = _build(monkeypatch, tmp_path, tokenizer_dir, ANCHOR_MASK="all")
    _check_round(anchor, mask="all", weight=1.0)


def test_rejects_overlong_pairs_and_unknown_masks(monkeypatch, tmp_path, tokenizer_dir):
    with pytest.raises(SystemExit, match="exceed prompt 4"):
        _build(monkeypatch, tmp_path, tokenizer_dir, ANCHOR_MAX_PROMPT="4")
    with pytest.raises(SystemExit, match="ANCHOR_MASK must be"):
        _build(monkeypatch, tmp_path, tokenizer_dir, ANCHOR_MASK="prompt")
