# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Pretraining rows of pre+mid+RL (App. C.8): row builder, per-step row selection, and the shared loss."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from unified_training.dataset import build_eval_dataset
from unified_training.episodes import load_round
from unified_training.hf_trainer import _stack, rank_slots, pretrain_micro_batches, weighted_ce_sums
from unified_training.tools.build_pretrain_rows import main as build_pretrain_rows

VOCAB = 50


def _shards(tmp_path, name, sizes, seed):
    rng = np.random.default_rng(seed)
    files = []
    for i, size in enumerate(sizes):
        path = tmp_path / f"{name}_{i}.bin"
        rng.integers(0, VOCAB, size=size).astype(np.uint16).tofile(path)
        files.append(str(path))
    return files


def _reference_rows(sources, n, chunk, seed):
    """The frozen C.8 builder's sampling, copied verbatim (two sources, 75/25 by rows)."""
    rng = np.random.default_rng(seed)
    rows = []
    for files, count in ((sources[0], round(0.75 * n)), (sources[1], n - round(0.75 * n))):
        mm = [np.memmap(f, dtype=np.uint16, mode="r") for f in files]
        for _ in range(count):
            k = int(rng.integers(len(mm)))
            off = int(rng.integers(0, len(mm[k]) - chunk - 2))
            rows.append(np.asarray(mm[k][off:off + chunk + 1], dtype=np.int32))
    perm = rng.permutation(len(rows))
    return [rows[i] for i in perm]


@pytest.fixture
def pretrain_round(tmp_path):
    web = _shards(tmp_path, "web", [5000, 3000, 4000], seed=1)
    other = _shards(tmp_path, "other", [2500, 3500], seed=2)
    demo = tmp_path / "demo_manifest.json"
    demo.write_text(json.dumps({"mean_response_len": 382.35174617208537}))
    out = tmp_path / "pretrain"
    build_pretrain_rows([
        str(out), "--rows", "40", "--seed", "2008", "--demo-manifest", str(demo),
        "--demo-rows-per-step", "8", "--pretrain-rows-per-step", "4", "--ratio", "0.10",
        "--source", "web=0.75:" + ",".join(web), "--source", "other=0.25:" + ",".join(other),
    ])
    return out, (web, other)


def test_chunk_rule_reproduces_the_c8_pools():
    # 10% of the midtraining tokens of a step, spread over 4 pretraining rows (App. A.1).
    mean = 382.35174617208537
    assert round(0.10 * 8 * mean / 4) == 76
    assert round(0.10 * 32 * mean / 4) == 306


def test_builder_matches_the_frozen_sampling_and_format(pretrain_round):
    out, sources = pretrain_round
    rnd = load_round(out)
    expected = _reference_rows(sources, 40, 76, 2008)
    assert len(rnd.index) == 40
    for i, ref in enumerate(expected):
        tokens, weights, row = rnd.episode(i)
        assert tokens.tolist() == ref.tolist()
        assert int(row["prompt_len"]) == 1 and float(row["weight"]) == 1.0
        assert weights[0] == 0.0 and np.all(weights[1:] == 1.0) and len(weights) == 77
    m = rnd.manifest
    assert m["chunk_supervised_tokens"] == 76 and m["pretrain_tokens_per_step"] == 304
    assert m["rows_by_source"] == {"web": 30, "other": 10}
    assert m["n_supervised_tokens"] == 40 * 76
    assert abs(m["realized_ratio_expected"] - 304 / (8 * 382.35174617208537)) < 1e-12


def test_builder_refuses_to_overwrite(pretrain_round):
    out, sources = pretrain_round
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        build_pretrain_rows([str(out), "--rows", "4", "--seed", "1", "--chunk", "8",
                           "--source", "web=1.0:" + sources[0][0]])


def _pool(out, *, steps, rows, seq=96):
    return build_eval_dataset([str(out)], sequence_length=seq, pad_token_id=0,
                              total_samples=steps * rows, seed=1234 + 101)


@pytest.mark.parametrize("world", [1, 2, 3])
def test_pretrain_rows_partition_each_step_across_ranks(pretrain_round, world):
    out, _ = pretrain_round
    ds = _pool(out, steps=5, rows=4)
    for step in range(5):
        got = {}
        for rank in range(world):
            micro = pretrain_micro_batches(ds, step=step, rows_per_step=4, world=world, rank=rank, mbs=1)
            slots = rank_slots(4, world=world, rank=rank)
            assert [len(m) for m in micro] == [1] * len(slots)
            for j, m in zip(slots, micro):
                assert not bool(m[0]["sampling_filter"])
                got[j] = m[0]["tokens"]
        assert sorted(got) == [0, 1, 2, 3]
        for j in range(4):   # stateless: draw step * rows + j, identical after a restart
            assert torch.equal(got[j], ds[step * 4 + j]["tokens"])


def test_pretrain_rows_are_unit_weight_full_vocab_ce_in_the_shared_token_mean(pretrain_round):
    out, _ = pretrain_round
    ds = _pool(out, steps=1, rows=4)
    rows = pretrain_micro_batches(ds, step=0, rows_per_step=4, world=1, rank=0, mbs=4)[0]
    b = _stack(rows, "cpu", trim=False)
    torch.manual_seed(0)
    logits = torch.randn(*b["tokens"].shape, VOCAB)
    # top-k/top-p apply to RL rows only; pretraining rows keep full-vocabulary CE.
    num, den = weighted_ce_sums(logits, b["labels"], b["loss_mask"], b["denom_mask"], norm="response_tokens",
                                top_k=3, top_p=0.5, sampling_filter=b["sampling_filter"])
    mask = b["loss_mask"] > 0
    ref = -torch.log_softmax(logits, -1).gather(-1, b["labels"].unsqueeze(-1)).squeeze(-1)[mask].sum()
    assert torch.allclose(num, ref)
    assert int(den) == 4 * 76 == int(mask.sum())
    assert torch.all(b["loss_mask"][mask] == 1.0)

    # Shared token mean: summing per-population numerators/denominators (anchor, pretraining, RL
    # micro-batches) gives the loss of the concatenated batch (Eq. 2).
    rl_logits = torch.randn(*b["tokens"].shape, VOCAB)
    rl_weights = b["loss_mask"] * 2.5
    rl_num, rl_den = weighted_ce_sums(rl_logits, b["labels"], rl_weights, b["denom_mask"], norm="response_tokens")
    cat_num, cat_den = weighted_ce_sums(
        torch.cat([logits, rl_logits]), torch.cat([b["labels"], b["labels"]]),
        torch.cat([b["loss_mask"], rl_weights]), torch.cat([b["denom_mask"], b["denom_mask"]]),
        norm="response_tokens", sampling_filter=torch.cat([b["sampling_filter"], torch.ones(4, dtype=torch.bool)]),
    )
    assert torch.allclose((num + rl_num) / (den + rl_den), cat_num / cat_den)
