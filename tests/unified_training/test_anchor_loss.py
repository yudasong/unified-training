# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""UNIFIED_ANCHOR_LOSS: how anchor (demo) rows enter the loss, and the separated sft step."""

from __future__ import annotations

import copy
import types

import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from unified_training.hf_trainer import resolve_anchor_loss, selected_token_logprobs, separated_anchor_step

ADD = {"anchor_mode": "add", "has_anchors": True}
REPLACE = {"anchor_mode": "replace", "has_anchors": True}
NO_ANCHORS = {"anchor_mode": "add", "has_anchors": False}


@pytest.mark.parametrize(
    ("env", "context", "expected"),
    [
        # Default: the pooled shared token mean of Eq. (2), in every anchor mode.
        ({}, ADD, ("pooled", None)),
        ({"UNIFIED_ANCHOR_LOSS": ""}, ADD, ("pooled", None)),
        ({}, REPLACE, ("pooled", None)),
        ({}, NO_ANCHORS, ("pooled", None)),
        ({"UNIFIED_SEP_SFT_ALPHA": "0.5"}, ADD, ("sft", 0.5)),
        ({"UNIFIED_SEP_SFT_ALPHA": "0"}, ADD, ("sft", 0.0)),
        ({"UNIFIED_ANCHOR_LOSS": "sft"}, ADD, ("sft", 1.0)),
        ({"UNIFIED_ANCHOR_LOSS": " SFT "}, ADD, ("sft", 1.0)),
        ({"UNIFIED_ANCHOR_LOSS": "sft", "UNIFIED_SEP_SFT_ALPHA": "0.1"}, ADD, ("sft", 0.1)),
        ({"UNIFIED_ANCHOR_LOSS": "pooled"}, ADD, ("pooled", None)),
        ({"UNIFIED_ANCHOR_LOSS": "pooled"}, REPLACE, ("pooled", None)),
        ({"UNIFIED_ANCHOR_LOSS": "pooled"}, NO_ANCHORS, ("pooled", None)),
    ],
)
def test_resolve_anchor_loss(env, context, expected):
    assert resolve_anchor_loss(env, **context) == expected


@pytest.mark.parametrize(
    ("env", "context", "match"),
    [
        ({"UNIFIED_ANCHOR_LOSS": "pooled", "UNIFIED_SEP_SFT_ALPHA": "0.5"}, ADD, "only applies to"),
        ({"UNIFIED_ANCHOR_LOSS": "sft"}, REPLACE, "needs UNIFIED_ANCHOR_MODE=add"),
        ({"UNIFIED_ANCHOR_LOSS": "sft"}, NO_ANCHORS, "needs anchor rows"),
        ({"UNIFIED_SEP_SFT_ALPHA": "0.5"}, REPLACE, "needs UNIFIED_ANCHOR_MODE=add"),
        ({"UNIFIED_SEP_SFT_ALPHA": "0.5"}, NO_ANCHORS, "needs anchor rows"),
        ({"UNIFIED_ANCHOR_LOSS": "shared"}, ADD, "must be one of"),
        ({"UNIFIED_SEP_SFT_ALPHA": "-0.5"}, ADD, "non-negative"),
        ({"UNIFIED_SEP_SFT_ALPHA": "half"}, ADD, "must be a number"),
    ],
)
def test_resolve_anchor_loss_rejects(env, context, match):
    with pytest.raises(ValueError, match=match):
        resolve_anchor_loss(env, **context)


VOCAB, SEQ = 23, 24


class _CausalToyLM(torch.nn.Module):
    """Embedding, causal running mean, linear head; returns ``.logits`` like a HF causal LM."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB, 16)
        self.proj = torch.nn.Linear(16, 16)
        self.head = torch.nn.Linear(16, VOCAB)

    def forward(self, input_ids):
        hidden = torch.tanh(self.proj(self.embed(input_ids)))
        steps = torch.arange(1, hidden.shape[1] + 1, dtype=hidden.dtype).view(1, -1, 1)
        return types.SimpleNamespace(logits=self.head(hidden.cumsum(1) / steps))


def _row(generator, *, rl: bool, weight: float) -> dict:
    total = int(torch.randint(8, SEQ, (1,), generator=generator))
    prompt = int(torch.randint(2, 6, (1,), generator=generator))
    ids = torch.zeros(SEQ + 1, dtype=torch.long)
    ids[:total] = torch.randint(1, VOCAB, (total,), generator=generator)
    weights, denom = torch.zeros(SEQ + 1), torch.zeros(SEQ + 1)
    weights[prompt:total] = weight
    denom[prompt:total] = 1.0
    return {
        "tokens": ids[:-1],
        "labels": ids[1:],
        "loss_mask": weights[1:],
        "denom_mask": denom[1:],
        "position_ids": torch.arange(SEQ),
        "sampling_filter": torch.tensor(rl),
    }


def _reference(model, rl_rows, demo_rows, *, alpha: float, top_p: float):
    """L = sum_RL(-w log pi)/T_rl + alpha * sum_demo(-w log pi)/T_demo, one row at a time."""
    reference = copy.deepcopy(model)
    rl_num, demo_num = torch.zeros(()), torch.zeros(())
    t_rl = t_demo = 0.0
    for row in rl_rows:
        live = row["denom_mask"].bool()
        logits = reference(input_ids=row["tokens"][None]).logits[0]
        logprobs, support = selected_token_logprobs(
            logits[live], row["labels"][live], top_p=top_p, return_support_mask=True
        )
        rl_num = rl_num - (row["loss_mask"][live] * logprobs).sum()
        t_rl += float(support.sum())
    for row in demo_rows:
        logits = reference(input_ids=row["tokens"][None]).logits[0]
        logprobs = torch.log_softmax(logits, -1).gather(-1, row["labels"][:, None])[:, 0]
        demo_num = demo_num - (row["loss_mask"] * logprobs).sum()
        t_demo += float(row["denom_mask"].sum())
    (rl_num / t_rl + alpha * demo_num / t_demo).backward()
    grad = torch.cat([p.grad.reshape(-1) for p in reference.parameters()])
    return grad, rl_num.detach(), demo_num.detach(), t_rl, t_demo


@pytest.fixture
def gloo_world_of_one(tmp_path):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("torch.distributed with the gloo backend is unavailable")
    if dist.is_initialized():
        pytest.skip("a default process group is already initialized")
    try:
        dist.init_process_group("gloo", store=dist.FileStore(str(tmp_path / "gloo_store"), 1), rank=0, world_size=1)
    except RuntimeError as error:
        pytest.skip(f"cannot start a gloo process group: {error}")
    yield
    dist.destroy_process_group()


@pytest.mark.filterwarnings("ignore:CUDA is not available")  # the step's bf16 autocast is CUDA-only
@pytest.mark.filterwarnings("ignore:User provided device_type of 'cuda'")
@pytest.mark.parametrize(("alpha", "top_p"), [(0.3, 1.0), (0.3, 0.9), (1.0, 1.0)])
def test_separated_anchor_step_matches_the_reference_gradient(gloo_world_of_one, alpha, top_p):
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(1)
    rl_rows = [_row(generator, rl=True, weight=w) for w in (2.0, 0.0, -0.5, 0.0, 1.25, -1.0)]
    demo_rows = [_row(generator, rl=False, weight=1.0) for _ in range(4)]
    model = _CausalToyLM()
    grad_ref, rl_num, demo_num, t_rl, t_demo = _reference(model, rl_rows, demo_rows, alpha=alpha, top_p=top_p)

    ddp = DistributedDataParallel(model)
    sums = separated_anchor_step(
        [demo_rows[:2], demo_rows[2:]],  # two anchor micro-batches
        [rl_rows[i : i + 2] for i in range(0, len(rl_rows), 2)],
        ddp=ddp,
        model=model,
        opt=torch.optim.SGD(model.parameters(), lr=0.0),
        device=torch.device("cpu"),
        alpha=alpha,
        loss_norm="response_tokens",
        top_k=-1,
        top_p=top_p,
        temperature=1.0,
        trim_rows=True,
        logprob_chunk_size=5,
    )
    grad = torch.cat([p.grad.reshape(-1) for p in model.parameters()])

    torch.testing.assert_close(grad, grad_ref, rtol=1e-5, atol=1e-6 * float(grad_ref.abs().max()))
    assert float(sums[1]) == t_rl and float(sums[3]) == t_demo
    torch.testing.assert_close(sums[0], rl_num, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(sums[2], demo_num, rtol=1e-5, atol=1e-5)
    assert t_demo == sum(float(r["denom_mask"].sum()) for r in demo_rows)
    all_rl_tokens = sum(float(r["denom_mask"].sum()) for r in rl_rows)
    if top_p < 1.0:
        assert t_rl < all_rl_tokens  # the support rescale of the RL term was exercised
    else:
        assert t_rl == all_rl_tokens  # zero-weight RL rows count in T_rl
