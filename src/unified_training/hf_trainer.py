# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unified trainer: one weighted token-likelihood loss over static data and RL generations.

Plain transformers + DDP, consuming the data layer in ``dataset.py`` (episode
shards written by the generator, static anchor/replay rows, streaming). Every row
is a token sequence with per-token weights (prompt 0; static data 1; RL responses
the weight A from the generator). The loss is Eq. (2) of the paper:

    L = -sum_rows sum_tokens w * log pi(token) / T_B

with T_B the global count of non-prompt tokens in the optimizer batch (all-reduced;
DDP's gradient averaging is cancelled by scaling). RL rows use selected-token
log-probabilities under the rollout sampling distribution (temperature, then top-k,
then top-p); static rows use full-vocabulary CE.

Checkpoints are saved directly as HF policies (``policies/v{step:07d}/`` + DONE),
which the generators poll and load; weights-only snapshots are published every
UNIFIED_PUBLISH_INTERVAL updates. Sample indexing is contiguous (micro-batch
slices per rank), so ``consumed = step * GBS`` and resume is index-exact.
Launched with torchrun (``python -m torch.distributed.run -m unified_training.hf_trainer``);
every knob is a UNIFIED_* environment variable (docs/configuration.md).
"""

from __future__ import annotations

import json
import math
import numpy as np
import datetime
import os
import signal
import shutil
import time
from collections.abc import Callable, Mapping
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from unified_training.checkpoint_milestones import (
    milestone_dir_name,
    preserve_checkpoint_subset,
)
from unified_training.dataset import StopRequested, build_eval_dataset, build_unified_datasets, streaming_primary
from unified_training.freshness_batch import fetch_fresh_batch
from unified_training.hf_checkpoint import (
    BackgroundCheckpointWriter,
    commit_checkpoint_pair,
    validate_resume_pair,
)
from unified_training.rollout_freshness import (
    RolloutFreshnessError,
    RolloutFreshnessGuard,
)


def _env(name: str, default):
    v = os.environ.get(name)
    return type(default)(v) if v is not None else default


def validation_evaluation_count(train_iters: int, eval_interval: int) -> int:
    """Number of baseline/scheduled/final validation calls in a full leg."""
    if eval_interval <= 0:
        return 0
    return 1 + train_iters // eval_interval + int(train_iters % eval_interval != 0)


def validation_consumed_samples_at_resume(
    step: int,
    *,
    eval_interval: int,
    eval_iters: int,
    gbs: int,
) -> int:
    """Cursor after validation calls durably implied by an intermediate step."""
    if step <= 0 or eval_interval <= 0:
        return 0
    completed_calls = 1 + step // eval_interval  # baseline @0 plus scheduled points
    return completed_calls * eval_iters * gbs


def loss_denominator(
    weights: torch.Tensor,
    norm: str,
    denom_mask: torch.Tensor | None = None,
    support_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-batch normalizer: Σw ('weights', matches Megatron) or #weighted positions
    ('tokens', stays positive so NEGATIVE advantages don't blow up the denominator).
    'response_tokens' uses the explicit response-token mask, so zero-advantage
    wrong/all-wrong responses still count in the denominator like NeMo-RL's
    token-level maxRL loss."""
    w = weights.float().view(-1)
    support = None
    if support_mask is not None:
        try:
            support = torch.broadcast_to(
                support_mask.to(device=weights.device, dtype=torch.bool), weights.shape
            ).reshape(-1)
        except RuntimeError as exc:
            raise ValueError(
                f"support_mask shape {support_mask.shape} cannot broadcast to weights {weights.shape}"
            ) from exc
        w = torch.where(support, w, torch.zeros_like(w))
    if norm == "response_tokens":
        if denom_mask is None:
            raise ValueError("loss_norm=response_tokens requires denom_mask")
        d = denom_mask.float().view(-1)
        if support is not None:
            d = torch.where(support, d, torch.zeros_like(d))
        return d.sum()
    return w.ne(0).float().sum() if norm == "tokens" else w.sum()


def _selected_token_logprobs_chunk(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    top_k: int,
    top_p: float,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate one bounded row chunk without retaining a full FP32 copy."""
    scaled = logits.float()
    # The production sampler uses temperature 1.0. Dividing by one used to
    # materialize another full [selected_tokens, vocab] tensor and caused the
    # step-491 OOM; preserve the exact identity without that allocation.
    if temperature != 1.0:
        scaled = scaled / temperature

    filtering = top_k > 0 or top_p < 1.0
    if filtering:
        with torch.no_grad():
            keep = torch.ones_like(scaled, dtype=torch.bool)
            if top_k > 0:
                threshold = torch.topk(scaled, top_k, dim=-1).values[..., -1:]
                keep &= scaled >= threshold
            if top_p < 1.0:
                ranked, order = scaled.masked_fill(~keep, -float("inf")).sort()
                cumulative = ranked.softmax(dim=-1).cumsum(dim=-1)
                ranked_keep = cumulative > 1.0 - top_p
                ranked_keep[..., -1] = True
                nucleus_keep = torch.zeros_like(ranked_keep).scatter(
                    -1, order, ranked_keep
                )
                keep &= nucleus_keep
        scaled = scaled.masked_fill(~keep, -float("inf"))

    token_lp = torch.log_softmax(scaled, dim=-1).gather(
        -1, labels.unsqueeze(-1)
    ).squeeze(-1)
    support_valid = torch.isfinite(token_lp)
    token_lp = torch.where(support_valid, token_lp, torch.zeros_like(token_lp))
    return token_lp, support_valid


def selected_token_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    top_k: int = -1,
    top_p: float = 1.0,
    temperature: float = 1.0,
    chunk_size: int = 64,
    return_support_mask: bool = False,
    activation_checkpoint_chunks: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Selected-token log-probs under the generation sampling distribution.

    NeMo-RL applies temperature scaling, top-k, then top-p to policy logits
    before evaluating the actor log-prob. The support decision is
    non-differentiable: gradients pass through retained logits and are zero
    outside the retained support.
    """
    if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] != labels.shape[0]:
        raise ValueError(f"expected logits [N,V] and labels [N], got {logits.shape} and {labels.shape}")
    if not 0.0 < float(top_p) <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    if float(temperature) <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if int(top_k) > logits.shape[-1]:
        raise ValueError(f"top_k {top_k} exceeds vocabulary size {logits.shape[-1]}")

    selected: list[torch.Tensor] = []
    support_masks: list[torch.Tensor] = []
    chunk_size = max(int(chunk_size), 1)
    for start in range(0, logits.shape[0], chunk_size):
        chunk = logits[start : start + chunk_size]
        chunk_labels = labels[start : start + chunk_size]
        operation = lambda chunk_, labels_: _selected_token_logprobs_chunk(
            chunk_,
            labels_,
            top_k=int(top_k),
            top_p=float(top_p),
            temperature=float(temperature),
        )
        if activation_checkpoint_chunks and torch.is_grad_enabled() and chunk.requires_grad:
            # Non-reentrant checkpointing recomputes each [chunk, vocab]
            # log-softmax during backward instead of retaining every FP32
            # activation until the micro-batch backward pass.
            token_lp, support_valid = activation_checkpoint(
                operation,
                chunk,
                chunk_labels,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            token_lp, support_valid = operation(chunk, chunk_labels)
        selected.append(token_lp)
        support_masks.append(support_valid)
    token_lp = torch.cat(selected) if selected else logits.new_empty((0,))
    if return_support_mask:
        support_valid = (
            torch.cat(support_masks)
            if support_masks
            else torch.empty(0, dtype=torch.bool, device=logits.device)
        )
        return token_lp, support_valid
    return token_lp


def weighted_ce_sums(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    denom_mask: torch.Tensor | None = None,
    norm: str = "weights",
    top_k: int = -1,
    top_p: float = 1.0,
    temperature: float = 1.0,
    sampling_filter: torch.Tensor | None = None,
    logprob_chunk_size: int = 64,
):
    """Return (-Σ w*log pi(token), denominator) for one micro-batch.

    ``sampling_filter`` marks policy rows/positions that use the reconstructed
    rollout distribution. False positions are mid-training data and retain
    ordinary full-vocabulary CE. If omitted, every weighted position is a
    policy token.
    """
    w = weights.float().view(-1)
    active = w.ne(0)

    if sampling_filter is None:
        sf = torch.ones_like(active, dtype=torch.bool)
        policy = active
        midtraining = torch.zeros_like(active)
    else:
        sf = sampling_filter.to(device=weights.device, dtype=torch.bool)
        if sf.ndim == 1 and weights.ndim > 1 and sf.shape[0] == weights.shape[0]:
            sf = sf.reshape((-1,) + (1,) * (weights.ndim - 1))
        try:
            sf = torch.broadcast_to(sf, weights.shape).reshape(-1)
        except RuntimeError as exc:
            raise ValueError(
                f"sampling_filter shape {sampling_filter.shape} cannot broadcast to weights {weights.shape}"
            ) from exc
        policy = active & sf
        midtraining = active & ~sf

    flat_logits = logits.view(-1, logits.size(-1))
    flat_labels = labels.view(-1)
    support_valid = torch.ones_like(w, dtype=torch.bool)
    # Keep a graph-connected zero for batches whose advantages are all zero.
    num = logits.sum() * 0.0

    # ``response_tokens`` deliberately normalizes by response positions even when
    # their advantage is zero. Those denominator-only policy positions must still
    # have their filtered support checked; otherwise an out-of-support token is
    # removed from the numerator but silently retained in the denominator.
    if norm == "response_tokens":
        if denom_mask is None:
            raise ValueError("loss_norm=response_tokens requires denom_mask")
        denominator_positions = denom_mask.to(device=weights.device).reshape(-1).ne(0)
    else:
        denominator_positions = active
    policy_eval = policy | (denominator_positions & sf)

    if torch.any(policy_eval):
        token_lp, policy_support_valid = selected_token_logprobs(
            flat_logits[policy_eval],
            flat_labels[policy_eval],
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            chunk_size=logprob_chunk_size,
            return_support_mask=True,
        )
        support_valid[policy_eval] = policy_support_valid
        active_in_policy_eval = policy[policy_eval]
        if torch.any(active_in_policy_eval):
            num = num - (
                token_lp[active_in_policy_eval] * w[policy]
            ).sum()
    if torch.any(midtraining):
        token_lp = selected_token_logprobs(
            flat_logits[midtraining],
            flat_labels[midtraining],
            top_k=-1,
            top_p=1.0,
            temperature=1.0,
            chunk_size=logprob_chunk_size,
        )
        num = num - (token_lp * w[midtraining]).sum()
    den = loss_denominator(
        weights,
        norm,
        denom_mask,
        support_mask=support_valid.view_as(weights),
    )
    return num, den


def kl_penalty_sums(
    logits: torch.Tensor,
    ref_logits: torch.Tensor,
    labels: torch.Tensor,
    denom_mask: torch.Tensor,
    sampling_filter: torch.Tensor | None = None,
    logprob_chunk_size: int = 64,
    clamp: float = 20.0,
) -> tuple[torch.Tensor, int]:
    """Reference-policy KL penalty for one micro-batch (App. E): returns
    (Σ k3, number of positions) over every RL response position, where

        k3 = exp(r) - r - 1,   r = log π_ref(t) - log π(t)   at the sampled token t.

    k3 is the Schulman (2020) estimator of KL(π ‖ π_ref) on on-policy samples: non-negative,
    zero iff the two token probabilities agree, gradient (1 - exp(r)) · ∇log π(t). It is the
    estimator NeMo-RL uses for its reference-policy penalty (calculate_kl_penalty_joschu2020).
    Both log-probs are full-vocabulary log-softmax values (no top-k/top-p reconstruction: the
    penalty is on the policy itself, not on the generation-time sampling distribution); the
    reference values carry no gradient. Positions are the response tokens of every RL row in
    the micro-batch (``denom_mask``, so zero-weight responses are penalized like the others,
    as NeMo-RL penalizes every sample); mid-training anchor rows (sampling_filter False) are
    excluded. ``r`` is clamped to ±clamp only to keep exp() finite. The caller adds
    coef · Σ k3 to the loss numerator, so under UNIFIED_LOSS_NORM=response_tokens the penalty
    is the per-response-token mean KL over the global batch, NeMo-RL's normalization."""
    pos = denom_mask.to(device=logits.device).reshape(-1).ne(0)
    if sampling_filter is not None:
        sf = sampling_filter.to(device=logits.device, dtype=torch.bool)
        if sf.ndim == 1 and denom_mask.ndim > 1 and sf.shape[0] == denom_mask.shape[0]:
            sf = sf.reshape((-1,) + (1,) * (denom_mask.ndim - 1))
        pos = pos & torch.broadcast_to(sf, denom_mask.shape).reshape(-1)
    n = int(pos.sum())
    if n == 0:
        return logits.sum() * 0.0, 0
    flat_labels = labels.view(-1)[pos]
    lp = selected_token_logprobs(
        logits.view(-1, logits.size(-1))[pos], flat_labels,
        top_k=-1, top_p=1.0, temperature=1.0, chunk_size=logprob_chunk_size,
    )
    with torch.no_grad():
        lp_ref = selected_token_logprobs(
            ref_logits.view(-1, ref_logits.size(-1))[pos], flat_labels,
            top_k=-1, top_p=1.0, temperature=1.0, chunk_size=logprob_chunk_size,
        )
    r = (lp_ref.detach() - lp).clamp(-clamp, clamp)
    return (torch.exp(r) - r - 1.0).sum(), n


def row_length(row: dict) -> int:
    """Tokens needed to compute every position that enters the loss: up to the
    last position with a loss weight OR a denominator mask entry (right-padding
    beyond it never affects earlier logits under causal attention).

    The denominator mask matters under ``UNIFIED_LOSS_NORM=response_tokens``:
    a zero-weight (wrong) response carries no numerator but every one of its
    response tokens counts in the denominator, so trimming it to its last
    weighted position (none) would drop those tokens and inflate the loss by the
    ratio of all response tokens to correct-response tokens."""
    # Signed advantages are active in weighted_ce_sums too. They may have no
    # denominator mask under the weights/tokens normalizers.
    live = row["loss_mask"].ne(0)
    dm = row.get("denom_mask")
    if dm is not None:
        live = live | dm.ne(0)
    nz = torch.nonzero(live)
    return int(nz[-1]) + 1 if len(nz) else 1


def trim_length(rows: list[dict], multiple: int = 64) -> int:
    """Common trimmed length for a micro-batch, rounded up to `multiple`."""
    seq = int(rows[0]["tokens"].shape[0])
    need = max(row_length(r) for r in rows)
    return min(seq, ((need + multiple - 1) // multiple) * multiple)


def group_rows_by_length(rows: list[dict], mbs: int) -> list[list[dict]]:
    """Regroup a rank's rows for one optimizer step into micro-batches of similar
    length. The step's loss is a sum over rows with one global denominator, so the
    grouping does not change the gradient; it only reduces padding."""
    order = sorted(range(len(rows)), key=lambda i: row_length(rows[i]))
    return [[rows[i] for i in order[j:j + mbs]] for j in range(0, len(order), mbs)]


def _stack(rows: list[dict], device, trim: bool = True) -> dict[str, torch.Tensor]:
    """Stack a micro-batch; with `trim`, drop the padded tail beyond the longest row."""
    seq = int(rows[0]["tokens"].shape[0])
    L = trim_length(rows) if trim else seq
    out = {}
    for k in rows[0]:
        if k == "source_policy_version":
            continue  # CPU provenance, not a model input or loss tensor.
        v = rows[0][k]
        if v.dim() >= 1 and v.shape[0] == seq:
            out[k] = torch.stack([r[k][:L] for r in rows]).to(device, non_blocking=True)
        else:
            out[k] = torch.stack([r[k] for r in rows]).to(device, non_blocking=True)
    return out


def consumed_policy_age_metrics(micro_rows: list[list[dict]], *, trainer_step: int,
                                device, max_policy_age_steps: int = 0) -> dict:
    """Measure one committed batch, including zero-weight RL rows, across DP ranks.

    trainer_step is the pre-update model version used for this batch. All ranks
    must pass their equal-sized primary/mixed micro-batches; appended anchor
    micro-batches are omitted. Replacement anchors are excluded by sampling_filter.
    Unknown provenance is counted, never treated as policy version zero.
    """
    versions = [
        max(int(row.get("source_policy_version", -1)), -1)
        if bool(row.get("sampling_filter", True)) else -2
        for rows in micro_rows for row in rows
    ]
    local = torch.tensor(versions, dtype=torch.int64, device=device)
    if dist.is_available() and dist.is_initialized():
        gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, local)
        values = torch.cat(gathered).cpu().tolist()
    else:
        values = local.cpu().tolist()
    known = [v for v in values if v >= 0]
    ages = np.asarray([trainer_step - v for v in known], dtype=np.int64)
    result = {
        "optimizer_step": trainer_step + 1,
        "policy_age_reference_step": trainer_step,
        "rl_rows": sum(v != -2 for v in values),
        "anchor_rows_excluded": values.count(-2),
        "known_rows": len(known),
        "unknown_rows": values.count(-1),
    }
    if known:
        result.update({
            "mean_steps": float(ages.mean()),
            "p50_steps": float(np.quantile(ages, .5)),
            "p95_steps": float(np.quantile(ages, .95)),
            "min_steps": int(ages.min()),
            "max_steps": int(ages.max()),
            "fraction_gt_4": float((ages > 4).mean()),
            "fraction_gt_16": float((ages > 16).mean()),
            "future_policy_rows": int((ages < 0).sum()),
        })
        if max_policy_age_steps > 0:
            result["configured_limit_steps"] = max_policy_age_steps
            result["fraction_gt_configured_limit"] = float((ages > max_policy_age_steps).mean())
    return result


def _micro_rows(ds, *, base: int, micro: int, world: int, rank: int, mbs: int) -> list[dict]:
    """Megatron 'single'-sampler order: contiguous mbs*world block per micro, rank slice."""
    start = base + micro * world * mbs + rank * mbs
    return [ds[i] for i in range(start, start + mbs)]


# ---- additive anchor mode (UNIFIED_ANCHOR_MODE=add) -------------------------
# The RL batch stays at GBS rows; anchor rows are appended on top, so a step is
# GBS + n rows with n drawn per step.  Ranks take slots j == rank (mod world),
# so their extra micro-batch counts differ by at most one; those run under
# DDP.no_sync() and the final RL micro-batch (present on every rank) carries
# the synchronized all-reduce.  Grads accumulated inside no_sync are included.
ANCHOR_MODES = ("replace", "add")


from unified_training.dataset import additive_anchor_count  # noqa: E402  (re-exported for tests)


def rank_slots(n: int, *, world: int, rank: int) -> list[int]:
    """This rank's share of n global slots: j in [0, n) with j % world == rank."""
    return list(range(rank, n, world))


def pretrain_micro_batches(pretrain_ds, *, step: int, rows_per_step: int, world: int, rank: int,
                           mbs: int) -> list[list[dict]]:
    """This rank's pretraining-row micro-batches (pre+mid+RL) for one optimizer step (App. C.8).

    Row j of step s is draw ``s * rows_per_step + j`` of the epoch-shuffled pretraining-row pool
    (stateless, so a resumed run continues the same sequence); ranks take slots
    j % world == rank. Pretraining rows are static data: ``sampling_filter`` False keeps them
    on full-vocabulary CE, and they join the shared token-mean loss like anchor rows.
    """
    rows = []
    for j in rank_slots(rows_per_step, world=world, rank=rank):
        row = dict(pretrain_ds[step * rows_per_step + j])
        row["sampling_filter"] = torch.tensor(False)
        rows.append(row)
    return [rows[i:i + mbs] for i in range(0, len(rows), mbs)]


# ---- anchor loss style (UNIFIED_ANCHOR_LOSS=sft|pooled) ---------------------
ANCHOR_LOSSES = ("sft", "pooled")


def resolve_anchor_loss(env: Mapping[str, str], *, anchor_mode: str, has_anchors: bool) -> tuple[str, float | None]:
    """How anchor (demo) rows enter the loss; returns ``(style, alpha)``.

    ``UNIFIED_ANCHOR_LOSS`` selects the style:

    * ``"pooled"`` (default): one token mean over RL and anchor tokens together,
      anchor tokens weighted by their shard weights -- the unified objective, Eq. (2)
      of the paper. ``alpha`` is None and ``UNIFIED_SEP_SFT_ALPHA`` must be unset.
    * ``"sft"``: the separately normalized objective of App. C.9; the anchor tokens
      form their own term, normalised by their own global token count (computed by
      :func:`separated_anchor_step`)::

          L = sum_RL(-w log pi) / T_rl + alpha * sum_anchor(-w log pi) / T_anchor

      with ``alpha = UNIFIED_SEP_SFT_ALPHA`` (1.0 when unset). Anchor tokens keep
      their shard weights ``w``. Needs ``anchor_mode == "add"`` (anchor rows in their
      own micro-batches) and anchors.
    * unset or empty: ``"sft"`` when ``UNIFIED_SEP_SFT_ALPHA`` is set, else ``"pooled"``.

    ``has_anchors``: the run configures anchor rows (UNIFIED_ANCHOR_DIRS or
    UNIFIED_STREAMS_JSON). Raises ValueError for an unknown style, a non-numeric or
    negative alpha, alpha together with ``"pooled"``, and ``"sft"`` (explicit, or
    implied by alpha) outside add mode or without anchors. main() additionally
    requires UNIFIED_LOSS_NORM=response_tokens and UNIFIED_KL_COEF=0 for ``"sft"``.
    """
    style = env.get("UNIFIED_ANCHOR_LOSS", "").strip().lower()
    alpha_raw = env.get("UNIFIED_SEP_SFT_ALPHA", "").strip()
    if style not in ("", *ANCHOR_LOSSES):
        raise ValueError(f"UNIFIED_ANCHOR_LOSS must be one of {ANCHOR_LOSSES}, got {style!r}")
    alpha = None
    if alpha_raw:
        try:
            alpha = float(alpha_raw)
        except ValueError as exc:
            raise ValueError(f"UNIFIED_SEP_SFT_ALPHA must be a number, got {alpha_raw!r}") from exc
        if not alpha >= 0.0:
            raise ValueError(f"UNIFIED_SEP_SFT_ALPHA must be non-negative, got {alpha}")
    if style == "pooled":
        if alpha is not None:
            raise ValueError("UNIFIED_SEP_SFT_ALPHA only applies to UNIFIED_ANCHOR_LOSS=sft; unset it for pooled")
        return "pooled", None
    if not style and alpha is None:
        return "pooled", None
    source = "UNIFIED_ANCHOR_LOSS=sft" if style else "UNIFIED_SEP_SFT_ALPHA (implies UNIFIED_ANCHOR_LOSS=sft)"
    if anchor_mode != "add":
        raise ValueError(
            f"{source} needs UNIFIED_ANCHOR_MODE=add (anchor rows in their own micro-batches); "
            "replace mode shares one loss, use UNIFIED_ANCHOR_LOSS=pooled"
        )
    if not has_anchors:
        raise ValueError(
            f"{source} needs anchor rows (UNIFIED_ANCHOR_DIRS or UNIFIED_STREAMS_JSON); "
            "without anchors there is nothing to separate"
        )
    return "sft", 1.0 if alpha is None else alpha


def separated_anchor_step(
    add_micro: list[list[dict]],
    rl_micro: list[list[dict]],
    *,
    ddp,
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    device,
    alpha: float,
    loss_norm: str,
    top_k: int,
    top_p: float,
    temperature: float,
    trim_rows: bool,
    logprob_chunk_size: int,
    ce_sums: Callable = weighted_ce_sums,
) -> torch.Tensor:
    """One optimizer step's gradient under UNIFIED_ANCHOR_LOSS=sft (App. C.9):

        L = alpha * sum_{demo tokens} -log pi / T_demo  +  sum_{RL tokens} -w log pi / T_rl

    T_demo = anchor-row denominator tokens (denom_mask; the response + EOS of a
    response-masked demo shard) and T_rl = ALL RL response tokens, zero-weight rows
    included; both are global over this optimizer step (all-reduced). The demo term
    is full-vocabulary CE, the RL term uses the reconstructed sampler distribution,
    exactly as in the shared loss. RL micro-batches run first so that their exact,
    support-checked denominator rescales only the RL gradient; the anchor
    micro-batches follow. Ranks can hold different numbers of anchor micro-batches,
    so every pass runs under no_sync and the summed gradient is all-reduced once
    (SUM: each rank's local gradient is already its share of the global objective,
    hence no world-size factor). ``ddp`` wraps ``model``; ``ce_sums`` is the
    per-micro-batch loss (weighted_ce_sums). Returns the local
    [rl_num, rl_den, demo_num, demo_den].
    """

    def local_den_sum(micro: list[list[dict]]) -> torch.Tensor:
        dens = [loss_denominator(r["loss_mask"], loss_norm, r.get("denom_mask")) for rows in micro for r in rows]
        return torch.stack(dens).sum().to(device) if dens else torch.zeros((), device=device)

    provisional = torch.stack([local_den_sum(rl_micro), local_den_sum(add_micro)])
    dist.all_reduce(provisional)
    provisional_rl = torch.clamp(provisional[0], min=1.0)
    demo_den = torch.clamp(provisional[1], min=1.0)  # exact: anchors have no support filter
    sums = torch.zeros(4, device=device)
    opt.zero_grad(set_to_none=True)

    def run(micro: list[list[dict]], denominator: torch.Tensor, coef: float, offset: int) -> None:
        for rows in micro:
            b = _stack(rows, device, trim=trim_rows)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = ddp(input_ids=b["tokens"]).logits
            num, micro_den = ce_sums(
                logits,
                b["labels"],
                b["loss_mask"],
                b.get("denom_mask"),
                norm=loss_norm,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                sampling_filter=b.get("sampling_filter"),
                logprob_chunk_size=logprob_chunk_size,
            )
            (coef * num / denominator).backward()
            sums[offset] += num.detach()
            sums[offset + 1] += micro_den.detach()

    with ddp.no_sync():
        run(rl_micro, provisional_rl, 1.0, 0)
        exact_rl = sums[1].clone()
        dist.all_reduce(exact_rl)
        rl_rescale = provisional_rl / torch.clamp(exact_rl, min=1.0)
        if float(rl_rescale) != 1.0:
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(rl_rescale)
        run(add_micro, demo_den, alpha, 2)
    grads = []
    for parameter in model.parameters():
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        grads.append(parameter.grad)
    flat = torch._utils._flatten_dense_tensors(grads)
    dist.all_reduce(flat)
    for grad, synced in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
        grad.copy_(synced)
    return sums


# ---- match_rl (UNIFIED_MATCH_RL_STEPS=N): run until the RL rows consumed equal
# what staged RL consumes in N steps (N*GBS).  In replace mode each step holds
# GBS*(1-r) RL rows, so the run overshoots N steps by about 1/(1-r).
def rl_rows_after(steps: int, gbs: int, row_fraction: float) -> int:
    """RL rows consumed after `steps` steps under the constant floor mixer."""
    idx = steps * gbs
    return idx - int(idx * row_fraction)


def match_rl_projected_steps(rl_steps: int, gbs: int, row_fraction: float) -> int:
    """Smallest step count whose RL-row consumption reaches rl_steps*GBS (floor mixer: exact)."""
    if rl_steps <= 0:
        return 0
    if row_fraction <= 0:
        return rl_steps
    target = rl_steps * gbs
    s = max(rl_steps, int(math.floor(rl_steps / (1.0 - row_fraction))) - 1)
    while rl_rows_after(s, gbs, row_fraction) < target:
        s += 1
    return s


class _PrimaryRows:
    """Index view over the mixer's RL pool for the Megatron-order sampler."""

    def __init__(self, mixer) -> None:
        self._mixer = mixer

    def __getitem__(self, idx: int) -> dict:
        return self._mixer.primary_row(idx)


def main() -> None:
    # Resuming a long run folds every finalized round into the stream before the
    # first collective; with tens of thousands of rounds on a parallel filesystem
    # this can exceed torch's 10-minute default NCCL timeout, so the default is one hour.
    nccl_timeout_s = int(os.environ.get("UNIFIED_NCCL_TIMEOUT_SECONDS", "3600"))
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=nccl_timeout_s))
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device("cuda")
    is_rank0 = rank == 0

    run_dir = Path(os.environ["UNIFIED_RUN_DIR"])
    policies = run_dir / "policies"
    state_dir = run_dir / "trainer_hf"
    stop_file = Path(os.environ.get("UNIFIED_TRAIN_STOP_FILE", run_dir / "STOP_TRAINER"))
    deadline_epoch = _env("UNIFIED_DEADLINE_EPOCH", 0.0)
    bootstrap = os.environ.get("UNIFIED_POLICY_HF", "")
    train_iters = _env("UNIFIED_TRAIN_ITERS", 8000)
    gbs = _env("UNIFIED_GBS", 256)
    mbs = _env("UNIFIED_MBS", 8)
    seq = _env("UNIFIED_SEQ", 1024)
    lr = _env("UNIFIED_LR", 2e-5)
    # Project-wide default: constant learning rate from optimizer step zero.
    # Warmup remains available only when a specific run explicitly opts in.
    warmup = _env("UNIFIED_WARMUP_ITERS", 0)
    save_interval = _env("UNIFIED_SAVE_INTERVAL", 200)
    eval_milestone_interval = _env("UNIFIED_EVAL_MILESTONE_INTERVAL", 1000)
    eval_interval = _env("UNIFIED_EVAL_INTERVAL", 200)
    eval_iters = _env("UNIFIED_EVAL_ITERS", 8)
    keep_policies = _env("UNIFIED_KEEP_POLICIES", 24)
    log_interval = _env("UNIFIED_LOG_INTERVAL", 10)
    consumed_age_interval = _env("UNIFIED_CONSUMED_AGE_LOG_INTERVAL", 0)
    if consumed_age_interval < 0:
        raise ValueError("UNIFIED_CONSUMED_AGE_LOG_INTERVAL must be non-negative")
    runtime_proof_step = _env("UNIFIED_RUNTIME_PROOF_STEP", 0)
    runtime_proof_dir = Path(
        os.environ.get("UNIFIED_RUNTIME_PROOF_DIR", run_dir / "runtime-proof")
    )
    grad_clip = _env("UNIFIED_GRAD_CLIP", 1.0)
    weight_decay = _env("UNIFIED_WD", 0.1)          # regular maxRL uses 0.0
    adam_beta2 = _env("UNIFIED_ADAM_BETA2", 0.95)   # regular maxRL uses 0.999
    # Reference-policy KL penalty (App. E): coef · KL(π ‖ π_ref) on RL response tokens,
    # k3 estimator, added to the loss numerator (see kl_penalty_sums). 0 = off.
    # The reference defaults to the run's init.
    kl_coef = _env("UNIFIED_KL_COEF", 0.0)
    assert kl_coef >= 0, kl_coef
    kl_ref_hf = os.environ.get("UNIFIED_KL_REF_HF", "")
    train_top_k = _env("UNIFIED_TOP_K", -1)
    train_top_p = _env("UNIFIED_TOP_P", 0.95)
    train_temperature = _env("UNIFIED_TEMP", 1.0)
    logprob_chunk_size = _env("UNIFIED_LOGPROB_CHUNK_SIZE", 64)
    streaming_enabled = os.environ.get("UNIFIED_STREAMING", "0") == "1"
    rollout_max_stale_steps = _env("UNIFIED_ROLLOUT_MAX_STALE_STEPS", 0)
    rollout_max_policy_lag_steps = _env(
        "UNIFIED_ROLLOUT_MAX_POLICY_LAG_STEPS",
        0,
    )
    rollout_wait_timeout_seconds = _env(
        "UNIFIED_ROLLOUT_WAIT_TIMEOUT_SECONDS",
        600.0,
    )
    rollout_poll_seconds = _env("UNIFIED_ROLLOUT_POLL_SECONDS", 10.0)
    rollout_check_interval_steps = _env(
        "UNIFIED_ROLLOUT_CHECK_INTERVAL_STEPS",
        4,
    )
    generator_world = _env("UNIFIED_GEN_WORLD", 1)
    rollout_freshness_enabled = bool(
        rollout_max_stale_steps or rollout_max_policy_lag_steps
    )
    # weights: denom = Σw (matches Megatron; positive weights only).
    # tokens:  denom = #weighted positions (supports NEGATIVE advantages — Σw can be ~0).
    # response_tokens: denom = assistant/response tokens, including zero-advantage
    # wrong samples; this matches NeMo-RL's token-level maxRL denominator.
    loss_norm = os.environ.get("UNIFIED_LOSS_NORM", "weights")
    assert loss_norm in ("weights", "tokens", "response_tokens"), loss_norm
    anchor_mode = os.environ.get("UNIFIED_ANCHOR_MODE", "replace")
    anchor_add_basis = os.environ.get("UNIFIED_ANCHOR_ADD_BASIS", "tokens")
    assert anchor_mode in ANCHOR_MODES, anchor_mode
    assert anchor_add_basis in ("tokens", "rows"), anchor_add_basis
    match_rl_steps = _env("UNIFIED_MATCH_RL_STEPS", 0)
    match_rl_anneal = os.environ.get("UNIFIED_MATCH_RL_ANNEAL", "0") == "1"
    if match_rl_anneal and not match_rl_steps:
        raise ValueError("UNIFIED_MATCH_RL_ANNEAL=1 needs UNIFIED_MATCH_RL_STEPS>0")
    assert match_rl_steps >= 0, match_rl_steps
    # Weights-only policy snapshots for the generators between full checkpoints
    # (0 = off: generators see a new policy only at every save_interval).
    publish_interval = _env("UNIFIED_PUBLISH_INTERVAL", 0)
    assert publish_interval >= 0, publish_interval
    # Rounds generated by a policy more than this many steps behind the trainer
    # are not sampled (the newest round per generator rank is always kept). 0 = off.
    max_policy_age_steps = _env("UNIFIED_MAX_POLICY_AGE_STEPS", 0)
    assert max_policy_age_steps >= 0, max_policy_age_steps
    strict_policy_age = os.environ.get("UNIFIED_STRICT_POLICY_AGE", "0") == "1"
    refresh_rounds_every_step = os.environ.get("UNIFIED_REFRESH_ROUNDS_EVERY_STEP", "0") == "1"
    if strict_policy_age:
        if max_policy_age_steps <= 0:
            raise ValueError("strict policy age requires UNIFIED_MAX_POLICY_AGE_STEPS>0")
        effective_publish_interval = publish_interval or save_interval
        if not 0 < effective_publish_interval <= max_policy_age_steps:
            raise ValueError("strict policy age requires a publication/save interval no larger than the age limit")
        if rollout_wait_timeout_seconds <= 0 or rollout_poll_seconds <= 0:
            raise ValueError("strict policy age requires positive rollout timeout and polling interval")
    # Micro-batch trimming: rows are stored padded to seq_length but
    # episodes average a few hundred tokens; trim each micro-batch to its longest
    # row and group a rank's rows by length. Mathematically identical loss.
    trim_rows = os.environ.get("UNIFIED_TRIM_ROWS", "1") == "1"
    if match_rl_steps and anchor_mode != "replace":
        raise ValueError("UNIFIED_MATCH_RL_STEPS applies to replace mode (add mode already holds GBS RL rows per step)")
    # Anchor (demo) loss style, UNIFIED_ANCHOR_LOSS=pooled|sft (see resolve_anchor_loss):
    # pooled (default) = the shared token mean below (Eq. 2); sft = the separately
    # normalized objective of App. C.9 (separated_anchor_step).
    anchor_dirs = [d for d in os.environ.get("UNIFIED_ANCHOR_DIRS", "").split(",") if d]
    streams_spec = json.loads(os.environ["UNIFIED_STREAMS_JSON"]) if os.environ.get("UNIFIED_STREAMS_JSON") else None
    anchor_loss, anchor_alpha = resolve_anchor_loss(
        os.environ, anchor_mode=anchor_mode, has_anchors=bool(anchor_dirs or streams_spec)
    )
    sep_sft = anchor_loss == "sft"
    sep_alpha = anchor_alpha if sep_sft else 0.0
    if sep_sft:
        if loss_norm != "response_tokens":
            raise ValueError(
                "UNIFIED_ANCHOR_LOSS=sft needs UNIFIED_LOSS_NORM=response_tokens (per-population token counts); "
                "set UNIFIED_ANCHOR_LOSS=pooled for the shared loss"
            )
        if kl_coef > 0:
            raise ValueError(
                "UNIFIED_ANCHOR_LOSS=sft is not implemented together with UNIFIED_KL_COEF > 0; "
                "set UNIFIED_ANCHOR_LOSS=pooled for the shared loss"
            )
    # Append the step's global loss-denominator token count to the iteration log line (off by default).
    log_loss_tokens = os.environ.get("UNIFIED_LOG_LOSS_TOKENS", "0") == "1"
    # Per-stream token counts and NLLs on the log line (always on with pretraining rows).
    log_stream_tokens = os.environ.get("UNIFIED_LOG_STREAM_TOKENS", "0") == "1"
    assert train_top_k == -1 or train_top_k > 0, train_top_k
    assert 0.0 < train_top_p <= 1.0, train_top_p
    assert train_temperature > 0.0, train_temperature
    assert warmup >= 0, warmup
    assert logprob_chunk_size > 0, logprob_chunk_size
    assert eval_milestone_interval >= 0, eval_milestone_interval
    assert runtime_proof_step >= 0, runtime_proof_step
    assert rollout_max_stale_steps >= 0, rollout_max_stale_steps
    assert rollout_max_policy_lag_steps >= 0, rollout_max_policy_lag_steps
    assert rollout_wait_timeout_seconds > 0, rollout_wait_timeout_seconds
    assert rollout_poll_seconds > 0, rollout_poll_seconds
    assert rollout_check_interval_steps > 0, rollout_check_interval_steps
    assert generator_world > 0, generator_world
    if rollout_freshness_enabled and not streaming_enabled:
        raise ValueError(
            "rollout freshness limits require UNIFIED_STREAMING=1"
        )
    assert gbs % (mbs * world) == 0, f"GBS {gbs} must divide by mbs*world {mbs}*{world}"
    accum = gbs // (mbs * world)

    # ---- resume state: latest_step.txt names the last COMPLETE (policy, opt) pair.
    latest_file = state_dir / "latest_step.txt"
    step = validate_resume_pair(policies, state_dir, latest_file)
    if is_rank0:
        from unified_training.hf_checkpoint import prune_unpaired_policies

        pruned = prune_unpaired_policies(policies, state_dir, step)
        if pruned:
            print(f"[unified-hf] removed {len(pruned)} weights-only snapshot(s) newer than the resume step {step}: {pruned[0]}..{pruned[-1]}", flush=True)
    dist.barrier()
    model_src = str(policies / f"v{step:07d}") if step > 0 else bootstrap
    if is_rank0:
        print(f"[unified-hf] world={world} gbs={gbs} mbs={mbs} accum={accum} "
              f"resume_step={step} model={model_src} train_top_k={train_top_k} "
              f"train_top_p={train_top_p} train_temperature={train_temperature} "
              f"logprob_chunk={logprob_chunk_size} trim_rows={int(trim_rows)}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_src, torch_dtype=torch.float32, attn_implementation="sdpa"
    ).to(device)
    model.train()
    ddp = DDP(model)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, adam_beta2), weight_decay=weight_decay)
    if step > 0:
        opt.load_state_dict(
            torch.load(state_dir / f"opt_{step:07d}.pt", map_location="cpu", weights_only=False)
        )
    kl_model = None
    if kl_coef > 0:
        kl_src = kl_ref_hf or bootstrap
        if not kl_src:
            raise ValueError("UNIFIED_KL_COEF > 0 needs UNIFIED_KL_REF_HF or UNIFIED_POLICY_HF")
        kl_model = AutoModelForCausalLM.from_pretrained(
            kl_src, torch_dtype=torch.float32, attn_implementation="sdpa"
        ).to(device)
        kl_model.eval()
        kl_model.requires_grad_(False)
        if is_rank0:
            print(f"[unified-hf] kl penalty: coef={kl_coef:g} estimator=k3 ref={kl_src}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(bootstrap or model_src)

    valid_samples = validation_evaluation_count(train_iters, eval_interval) * eval_iters * gbs
    valid_refs_manifest = Path(
        os.environ.get("UNIFIED_VALID_REFS_MANIFEST", state_dir / "valid_holdout.json")
    )
    if (
        step > 0
        and streaming_enabled
        and valid_samples > 0
        and not valid_refs_manifest.is_file()
    ):
        raise RuntimeError(
            "refusing a streaming resume without the original validation holdout: "
            f"{valid_refs_manifest}"
        )
    dataset_args = dict(
        episodes_dirs=os.environ["UNIFIED_EPISODES_DIRS"].split(","),
        train_samples=train_iters * gbs,
        valid_samples=valid_samples,
        sequence_length=seq,
        valid_holdout=_env("UNIFIED_VALID_HOLDOUT", 128),
        split_seed=_env("UNIFIED_SPLIT_SEED", 1234),
        pad_token_id=_env("UNIFIED_PAD_TOKEN_ID", 50256),
        anchor_dirs=anchor_dirs,
        anchor_fraction=_env("UNIFIED_ANCHOR_FRACTION", 0.0),
        anchor_token_fraction=_env("UNIFIED_ANCHOR_TOKEN_FRACTION", 0.0),
        max_anchor_row_fraction=_env("UNIFIED_MAX_ANCHOR_ROW_FRACTION", 1.0),
        mix_basis=os.environ.get("UNIFIED_MIX_BASIS", "supervised_tokens"),
        mix_recalc=os.environ.get("UNIFIED_MIX_RECALC", "online"),
        require_no_anchor_replay=(
            os.environ.get("UNIFIED_REQUIRE_NO_ANCHOR_REPLAY", "0") == "1"
        ),
        streams=streams_spec,
        streaming=streaming_enabled,
        rescan_every=_env("UNIFIED_RESCAN_EVERY", 256),
        recency_window=_env("UNIFIED_RECENCY_WINDOW", 0),
        max_policy_age_steps=max_policy_age_steps,
        strict_policy_age=strict_policy_age,
        initial_trainer_step=step,
        stop_file=stop_file,
        without_replacement=os.environ.get("UNIFIED_WITHOUT_REPLACEMENT", "0") == "1",
        # Wrong-tokenizer guard, on by default (model vocab).
        expected_vocab_size=_env("UNIFIED_EXPECTED_VOCAB", int(getattr(model.config, "vocab_size", 0))),
        valid_refs_manifest=valid_refs_manifest,
        midtrain_holdout_dirs=[
            d for d in os.environ.get("UNIFIED_MIDTRAIN_VAL_DIR", "").split(",") if d
        ],
        # Frozen beside the validation-holdout manifest so every rank and every
        # resume reuses rank zero's measurement instead of re-measuring a pool
        # that has grown in the meantime.
        mix_state_manifest=(Path(run_dir) / "launch-mix-state.json") if run_dir else None,
        match_rl_steps=match_rl_steps,
        match_rl_anneal=match_rl_anneal,
        global_batch_size=gbs,
    )
    # Rank zero freezes the initial streaming holdout before the other ranks
    # scan the growing round directory. Every rank then loads the same manifest.
    # A stop requested while rank zero still waits for the first rounds (e.g. the generators failed to start)
    # ends every rank before the first update; nothing in RUN_DIR changes.
    stop_before_start = torch.zeros(1, device=device, dtype=torch.int32)
    if is_rank0:
        try:
            train_ds, valid_ds, banner = build_unified_datasets(
                **dataset_args,
                persist_valid_refs=(step == 0 and valid_samples > 0),
                persist_mix_state=(step == 0),
            )
        except StopRequested as exc:
            print(f"[unified-hf] stop requested: {exc}; exiting before the first update", flush=True)
            stop_before_start.fill_(1)
    dist.broadcast(stop_before_start, 0)
    if stop_before_start.item():
        dist.destroy_process_group()
        return
    if not is_rank0:
        train_ds, valid_ds, banner = build_unified_datasets(
            **dataset_args,
            persist_valid_refs=False,
        )
    if is_rank0:
        print(banner, flush=True)
    primary_stream = streaming_primary(train_ds)
    if (strict_policy_age or refresh_rounds_every_step) and primary_stream is None:
        raise ValueError("strict freshness/per-step refresh requires a streaming primary RL dataset")
    if max_policy_age_steps and not hasattr(train_ds, "set_trainer_step"):
        if is_rank0:
            print("[unified-hf] WARNING: UNIFIED_MAX_POLICY_AGE_STEPS set but this dataset has no streaming RL pool; policy-age filter disabled", flush=True)
        max_policy_age_steps = 0
    if max_policy_age_steps:
        train_ds.set_trainer_step(step)
        if is_rank0:
            print(f"[unified-hf] max_policy_age_steps={max_policy_age_steps}: rounds from a policy more than "
                  f"{max_policy_age_steps} steps behind the trainer are excluded "
                  + ("(strict: no stale fallback)" if strict_policy_age else "(newest round per generator kept)"), flush=True)

    rl_rows = train_ds
    if consumed_age_interval and is_rank0:
        print(f"[unified-hf] consumed-policy age: one actual global batch every {consumed_age_interval} "
              "optimizer steps; age measured against the pre-update step", flush=True)
    if anchor_mode == "add":
        if not hasattr(train_ds, "additive_plan"):
            raise RuntimeError(
                "UNIFIED_ANCHOR_MODE=add needs a two-way anchor mixer "
                "(UNIFIED_ANCHOR_TOKEN_FRACTION>0, one anchor stream)"
            )
        rl_rows = _PrimaryRows(train_ds)
        total_add = train_ds.validate_additive(gbs, anchor_add_basis)
        if is_rank0:
            print(
                f"[unified-hf] anchor_mode=add basis={anchor_add_basis}: "
                f"{gbs} RL rows per step plus "
                f"{train_ds.additive_rows_per_step(gbs, anchor_add_basis):.3f} expected "
                f"anchor rows ({total_add:,} over the run); replace-mode row_f would be "
                f"{getattr(train_ds, 'anchor_fraction', float('nan')):.4f}",
                flush=True,
            )
    if is_rank0:
        if sep_sft:
            print(
                f"[unified-hf] anchor_loss=sft alpha={sep_alpha:g}: L = sum(-w log pi over RL rows)/T_rl + "
                f"{sep_alpha:g} * sum(anchor NLL)/T_demo, both token counts global per step",
                flush=True,
            )
        else:
            print(
                "[unified-hf] anchor_loss=pooled alpha=None: one token mean over RL and anchor tokens "
                f"(anchor_mode={anchor_mode}, anchors={'yes' if anchor_dirs or streams_spec else 'no'})",
                flush=True,
            )
    # pre+mid+RL pretraining rows (App. C.8; off unless UNIFIED_PRETRAIN_DIRS is set): a second additive stream of
    # exactly UNIFIED_PRETRAIN_ROWS_PER_STEP pretraining rows appended to every step after the anchor rows, in
    # their own micro-batches. The rows (prompt-less windows: first token context, weight 1 on the rest) join
    # the same token-mean objective as the RL and anchor rows. Row j of step s is draw s * rows + j of an
    # epoch-shuffled pool (seed UNIFIED_SPLIT_SEED + 101): stateless and resume-safe.
    pretrain_dirs = [d for d in os.environ.get("UNIFIED_PRETRAIN_DIRS", "").split(",") if d]
    pretrain_rows = _env("UNIFIED_PRETRAIN_ROWS_PER_STEP", 0)
    pretrain_ds = None
    if pretrain_dirs or pretrain_rows:
        if not pretrain_dirs or pretrain_rows <= 0:
            raise ValueError("UNIFIED_PRETRAIN_DIRS and UNIFIED_PRETRAIN_ROWS_PER_STEP>0 go together")
        if anchor_mode != "add" or sep_sft:
            raise ValueError("pre+mid+RL rows are implemented for the shared loss with UNIFIED_ANCHOR_MODE=add only")
        pretrain_ds = build_eval_dataset(
            pretrain_dirs,
            sequence_length=seq,
            pad_token_id=dataset_args["pad_token_id"],
            total_samples=train_iters * pretrain_rows,
            seed=dataset_args["split_seed"] + 101,
            expected_vocab_size=dataset_args["expected_vocab_size"],
        )
        if is_rank0:
            print(
                f"[unified-hf] pre+mid+RL pretraining rows: {pretrain_rows} rows per step from {','.join(pretrain_dirs)} "
                f"({pretrain_ds.n_episodes:,} rows in the pool, {train_iters * pretrain_rows:,} draws over the run), "
                "shared token-mean loss",
                flush=True,
            )
    # Per-stream sums since the last log line: [rl_num, rl_den, demo_num, demo_den, pretrain_num, pretrain_den].
    stream_interval = torch.zeros(6, device=device)

    # match_rl bookkeeping: exact RL-row counter (persisted beside the optimizer
    # state), projected stop step published for the leg and the dispatcher.
    match_rl_target_rows = match_rl_steps * gbs
    match_rl_file = state_dir / "match_rl.json"
    target_file = state_dir / "match_rl_target.txt"
    rl_rows_consumed = 0
    if match_rl_steps:
        if match_rl_target_rows > train_iters * gbs:
            raise ValueError(
                f"UNIFIED_MATCH_RL_STEPS={match_rl_steps} cannot be reached within "
                f"UNIFIED_TRAIN_ITERS={train_iters}; raise the iteration bound"
            )
        anneal_horizon = getattr(train_ds, "match_rl_horizon", None)
        if match_rl_anneal and anneal_horizon is None:
            raise RuntimeError("UNIFIED_MATCH_RL_ANNEAL=1 but the dataset builder solved no anneal horizon (needs one scheduled anchor stream)")
        if anneal_horizon is not None and anneal_horizon > train_iters:
            raise ValueError(
                f"match_rl anneal horizon {anneal_horizon} exceeds UNIFIED_TRAIN_ITERS={train_iters}; raise the iteration bound"
            )
        if step > 0:
            rec = json.load(open(match_rl_file)) if match_rl_file.exists() else {}
            if rec.get("step") == step:
                rl_rows_consumed = int(rec["rl_rows_consumed"])
            elif anneal_horizon is not None:
                rl_rows_consumed = train_ds.expected_rl_rows(step)
                if is_rank0:
                    print("[unified-hf] match_rl: no persisted RL-row count at resume; using the anneal's expected count", flush=True)
            elif type(train_ds).__name__ == "MixedEpisodeDataset":
                rl_rows_consumed = rl_rows_after(step, gbs, train_ds.anchor_fraction)
            else:
                rl_rows_consumed = int(step * gbs * (1.0 - getattr(train_ds, "anchor_fraction", 0.0)))
                if is_rank0:
                    print("[unified-hf] match_rl: no persisted RL-row count at resume; using the expected count", flush=True)
        r0 = float(getattr(train_ds, "anchor_fraction", 0.0))
        projected = anneal_horizon if anneal_horizon is not None else match_rl_projected_steps(match_rl_steps, gbs, r0)
        if anneal_horizon is not None:
            r0 = float(train_ds.launch_row_fraction)   # printed as row_f below; the schedule's start
        if is_rank0:
            target_file.parent.mkdir(parents=True, exist_ok=True)
            target_file.write_text(f"{projected}\n")
            print(
                f"[unified-hf] match_rl: stop when RL rows reach {match_rl_target_rows:,} "
                f"({match_rl_steps} staged-RL steps x {gbs}); row_f={r0:.6f} -> projected stop "
                f"step {projected} (bound {train_iters}); consumed so far {rl_rows_consumed:,}",
                flush=True,
            )

    freshness_initial_step = step
    freshness_guard = (
        RolloutFreshnessGuard(
            dataset_args["episodes_dirs"],
            expected_world=generator_world,
            max_stale_steps=rollout_max_stale_steps,
            max_policy_lag_steps=rollout_max_policy_lag_steps,
            initial_step=step,
        )
        if is_rank0 and rollout_freshness_enabled
        else None
    )
    if is_rank0 and rollout_freshness_enabled:
        print(
            "[unified-hf] rollout freshness guard: "
            f"generator_world={generator_world} "
            f"max_stale_steps={rollout_max_stale_steps} "
            f"max_policy_lag_steps={rollout_max_policy_lag_steps} "
            f"check_every={rollout_check_interval_steps} "
            f"wait_timeout={rollout_wait_timeout_seconds:.0f}s "
            f"poll={rollout_poll_seconds:.0f}s",
            flush=True,
        )

    def publish_runtime_proof(at_step: int) -> None:
        runtime_proof_dir.mkdir(parents=True, exist_ok=True)
        final = runtime_proof_dir / f"rank{rank:03d}.json"
        temporary = runtime_proof_dir / f".rank{rank:03d}.tmp-{os.getpid()}"
        payload = {
            "rank": rank,
            "world": world,
            "step": at_step,
            "gbs": gbs,
            "mbs": mbs,
            "accum": accum,
            "dataset_banner": banner,
            "anchor_row_fraction": getattr(train_ds, "anchor_fraction", None),
            "anchor_mode": anchor_mode,
            "anchor_add_basis": anchor_add_basis if anchor_mode == "add" else None,
            "anchor_rows_added_expected": (
                train_ds.additive_rows_per_step(gbs, anchor_add_basis) if anchor_mode == "add" else 0.0
            ),
            "match_rl_steps": match_rl_steps,
            "anchor_loss": anchor_loss,
            "sep_sft_alpha": sep_alpha if sep_sft else None,
            "policy_publish_interval": publish_interval,
            "max_policy_age_steps": max_policy_age_steps,
            "strict_policy_age": strict_policy_age,
            "refresh_rounds_every_step": refresh_rounds_every_step,
            "match_rl_rl_rows_consumed": rl_rows_consumed,
            "cuda_max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "cuda_max_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "cuda_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        }
        with temporary.open("w") as stream:
            json.dump(payload, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, final)
        # This explicit post-optimizer barrier proves every DDP rank reached the
        # same completed step after publishing its own dataset-view sentinel.
        dist.barrier()
        if is_rank0:
            print(
                f"[unified-hf] runtime proof: {world}/{world} ranks completed step "
                f"{at_step} -> {runtime_proof_dir}",
                flush=True,
            )

    # Optional mid-training retention metric: a held-out slice of mid-training windows
    # (built from a corpus shard the training data never touches). Scored each eval as
    # val/midtrain_lm_loss. OFF unless UNIFIED_MIDTRAIN_VAL_DIR is set.
    midtrain_val_ds = None
    _mv_dir = os.environ.get("UNIFIED_MIDTRAIN_VAL_DIR", "")
    if _mv_dir:
        midtrain_val_ds = build_eval_dataset(
            [d for d in _mv_dir.split(",") if d],
            sequence_length=seq, pad_token_id=_env("UNIFIED_PAD_TOKEN_ID", 50256),
            total_samples=max(eval_iters, 1) * gbs,
            expected_vocab_size=_env("UNIFIED_EXPECTED_VOCAB", int(getattr(model.config, "vocab_size", 0))),
        )
        if is_rank0:
            print(f"[unified-hf] mid-training val: {midtrain_val_ds.n_episodes} windows from {_mv_dir}", flush=True)

    def global_loss_sums(num: torch.Tensor, den: torch.Tensor) -> tuple[float, float]:
        t = torch.stack([num.detach(), den.detach()])
        dist.all_reduce(t)
        return float(t[0]), float(t[1])

    def _eval_loss(ds, base: int, *, policy_distribution: bool = True) -> float:
        num_t, den_t = torch.zeros((), device=device), torch.zeros((), device=device)
        with torch.no_grad():
            for it in range(eval_iters):
                for micro in range(accum):
                    rows = _micro_rows(ds, base=base + it * gbs,
                                       micro=micro, world=world, rank=rank, mbs=mbs)
                    b = _stack(rows, device, trim=trim_rows)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = ddp.module(input_ids=b["tokens"]).logits
                    n, d = weighted_ce_sums(
                        logits,
                        b["labels"],
                        b["loss_mask"],
                        b.get("denom_mask"),
                        norm=loss_norm,
                        top_k=train_top_k if policy_distribution else -1,
                        top_p=train_top_p if policy_distribution else 1.0,
                        temperature=(
                            train_temperature if policy_distribution else 1.0
                        ),
                        sampling_filter=b.get("sampling_filter"),
                        logprob_chunk_size=logprob_chunk_size,
                    )
                    num_t += n
                    den_t += d
        n, d = global_loss_sums(num_t, den_t)
        return n / max(d, 1.0)

    def evaluate(at_step: int, consumed_valid: int) -> tuple[float, int]:
        ddp.eval()
        rl_loss = _eval_loss(valid_ds, consumed_valid)
        logs = {"step": at_step, "val/lm_loss": rl_loss}
        if midtrain_val_ds is not None:
            # base=0 -> the same held-out slice every eval, so the trend is comparable
            logs["val/midtrain_lm_loss"] = _eval_loss(
                midtrain_val_ds, 0, policy_distribution=False
            )
        ddp.train()
        if is_rank0:
            extra = (f" | midtrain lm {logs['val/midtrain_lm_loss']:.6f}"
                     if midtrain_val_ds is not None else "")
            print(f"[unified-hf] validation @ {at_step}: lm loss {rl_loss:.6f}{extra}", flush=True)
        return rl_loss, consumed_valid + eval_iters * gbs

    last_saved_step = step

    def preserve_eval_milestone(at_step: int, policy_dir: Path) -> None:
        milestone_root = Path(
            os.environ.get(
                "UNIFIED_EVAL_MILESTONE_DIR", run_dir / "eval-checkpoints"
            )
        )
        target = milestone_root / milestone_dir_name(at_step, at_step)
        preserve_checkpoint_subset(
            policy_dir,
            target,
            include=(".",),
            manifest={
                "backend": "unified-hf",
                "target_optimizer_step": at_step,
                "actual_optimizer_step": at_step,
                "weights_only": True,
            },
        )

    publish_writer = BackgroundCheckpointWriter()

    def wait_publish() -> None:
        publish_writer.wait()

    def publish_policy(at_step: int) -> None:
        """Weights-only snapshot for the generators, written by a background thread
        from a CPU copy taken synchronously (one copy of the model, ~2 s)."""
        if not is_rank0:
            return
        from unified_training.hf_checkpoint import publish_policy_snapshot

        wait_publish()
        if (policies / f"v{at_step:07d}" / "DONE").exists():
            return
        cpu_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        staged = policies / f".v{at_step:07d}.pub-{os.getpid()}"
        final = policies / f"v{at_step:07d}"

        def _write() -> None:
            if staged.exists():
                shutil.rmtree(staged)
            model.save_pretrained(staged, state_dict=cpu_state, safe_serialization=True)
            tokenizer.save_pretrained(staged)
            publish_policy_snapshot(staged_policy=staged, final_policy=final)
            print(f"[unified-hf] published policy v{at_step:07d} (weights only)", flush=True)

        publish_writer.start(_write, description=f"publish-v{at_step:07d} to {final}")

    def save(at_step: int, *, preserve_for_eval: bool = False) -> None:
        nonlocal last_saved_step
        if is_rank0:
            wait_publish()
            pol = policies / f"v{at_step:07d}"
            policies.mkdir(parents=True, exist_ok=True)
            if pol.exists() and not (state_dir / f"opt_{at_step:07d}.pt").exists():
                shutil.rmtree(pol)   # a weights-only snapshot of this step; the full pair replaces it
            state_dir.mkdir(parents=True, exist_ok=True)
            staged_pol = policies / f".v{at_step:07d}.tmp-{os.getpid()}"
            staged_opt = state_dir / f".opt_{at_step:07d}.pt.tmp-{os.getpid()}"
            if staged_pol.exists():
                shutil.rmtree(staged_pol)
            staged_opt.unlink(missing_ok=True)
            model.save_pretrained(staged_pol, safe_serialization=True)
            tokenizer.save_pretrained(staged_pol)
            torch.save(opt.state_dict(), staged_opt)
            commit_checkpoint_pair(
                staged_policy=staged_pol,
                staged_optimizer=staged_opt,
                final_policy=pol,
                final_optimizer=state_dir / f"opt_{at_step:07d}.pt",
                latest_file=latest_file,
                step=at_step,
            )
            if preserve_for_eval:
                preserve_eval_milestone(at_step, pol)
            for old in sorted(state_dir.glob("opt_*.pt"))[:-2]:
                old.unlink()
            kept = sorted(d for d in policies.glob("v*") if (d / "DONE").exists())
            for old in kept[:-keep_policies]:
                if old.name == "v0000000":
                    continue
                if (state_dir / f"opt_{old.name[1:]}.pt").exists():
                    continue   # a resumable pair (its optimizer file still exists): keep it
                shutil.rmtree(old)
            suffix = " and preserved evaluation milestone" if preserve_for_eval else ""
            print(f"[unified-hf] saved policy v{at_step:07d}{suffix}", flush=True)
        last_saved_step = at_step
        dist.barrier()

    # Repair the small crash window between committing a resumable policy and
    # publishing its evaluation milestone. This also pins an exact resume
    # boundary before rolling retention can remove it during the continuation.
    if step > 0 and eval_milestone_interval and step % eval_milestone_interval == 0:
        if is_rank0:
            preserve_eval_milestone(step, policies / f"v{step:07d}")
            print(
                f"[unified-hf] verified evaluation milestone v{step:07d} at resume",
                flush=True,
            )
        dist.barrier()

    signal_state: dict[str, int | None] = {"signal": None}

    def _request_stop(signum, _frame) -> None:
        signal_state["signal"] = int(signum)

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _request_stop)

    def collective_stop_requested() -> bool:
        local_stop = (
            signal_state["signal"] is not None
            or stop_file.exists()
            or (deadline_epoch > 0 and time.time() >= deadline_epoch)
        )
        flag = torch.tensor(int(local_stop), device=device, dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    def enforce_rollout_freshness(at_step: int) -> None:
        """Collectively pause before an update until rollout progress recovers."""
        if (
            not rollout_freshness_enabled
            or (at_step - freshness_initial_step)
            % rollout_check_interval_steps
            != 0
        ):
            return

        # 0 = recovered, 1 = still blocked, 2 = fatal. Every poll is collective
        # so no rank sits in one NCCL operation for the entire wait timeout.
        state = torch.zeros((), device=device, dtype=torch.int32)
        wait_deadline = time.monotonic() + rollout_wait_timeout_seconds
        announced_wait = False
        while True:
            state.zero_()
            if is_rank0:
                assert freshness_guard is not None
                try:
                    observation = freshness_guard.observe(at_step)
                except RolloutFreshnessError as exc:
                    print(
                        "[unified-hf] ERROR: invalid committed rollout stream: "
                        f"{exc}",
                        flush=True,
                    )
                    state.fill_(2)
                except BaseException as exc:
                    # Rank zero must always broadcast a terminal state; otherwise
                    # peers can remain stuck in the collective after an
                    # unexpected filesystem/runtime error.
                    print(
                        "[unified-hf] ERROR: rollout freshness check failed "
                        f"unexpectedly: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    state.fill_(2)
                else:
                    if observation.blocked_reason is None:
                        if announced_wait:
                            print(
                                "[unified-hf] rollout freshness recovered at "
                                f"trainer step {at_step}: "
                                f"{observation.snapshot.summary()}",
                                flush=True,
                            )
                    else:
                        state.fill_(1)
                        if not announced_wait:
                            print(
                                "[unified-hf] pausing before optimizer update "
                                f"{at_step + 1}: "
                                f"{observation.blocked_reason}; "
                                f"{observation.snapshot.summary()}",
                                flush=True,
                            )
                            announced_wait = True
                        if time.monotonic() >= wait_deadline:
                            print(
                                "[unified-hf] ERROR: rollout freshness did not "
                                f"recover within "
                                f"{rollout_wait_timeout_seconds:.0f}s at "
                                f"trainer step {at_step}: "
                                f"{observation.blocked_reason}; "
                                f"{observation.snapshot.summary()}",
                                flush=True,
                            )
                            state.fill_(2)

            dist.broadcast(state, src=0)
            state_value = int(state.item())
            if state_value == 0:
                return
            if state_value == 2:
                raise RuntimeError(
                    f"rollout freshness guard failed at trainer step {at_step}"
                )
            # Service stop/deadline signals while paused. The caller re-checks
            # the collective stop condition before it can fetch another row.
            if collective_stop_requested():
                return
            remaining = wait_deadline - time.monotonic()
            time.sleep(max(0.0, min(rollout_poll_seconds, remaining)))

    valid_cursor_override = os.environ.get("UNIFIED_VALID_CONSUMED_SAMPLES")
    if valid_cursor_override is not None:
        try:
            consumed_valid = int(valid_cursor_override)
        except ValueError as exc:
            raise ValueError(
                f"invalid UNIFIED_VALID_CONSUMED_SAMPLES={valid_cursor_override!r}"
            ) from exc
        if consumed_valid < 0:
            raise ValueError("UNIFIED_VALID_CONSUMED_SAMPLES must be non-negative")
    else:
        consumed_valid = validation_consumed_samples_at_resume(
            step,
            eval_interval=eval_interval,
            eval_iters=eval_iters,
            gbs=gbs,
        )
    if is_rank0 and step > 0:
        print(
            f"[unified-hf] validation resume cursor={consumed_valid} samples",
            flush=True,
        )
    # Establish both the RL holdout and optional mid-training-retention
    # baselines before the first optimizer update.  A resumed allocation keeps
    # the original step-zero record in the persisted log/W&B stream and must
    # not emit a second baseline from a later policy.
    if valid_ds is not None and eval_interval and step == 0:
        _, consumed_valid = evaluate(0, consumed_valid)
    t_log = time.monotonic()
    paused = False
    while step < train_iters:
        if collective_stop_requested():
            # A stop can arrive after the prior iteration's post-update check
            # but before this loop-entry check. Preserve that committed update
            # instead of rolling back as many as save_interval-1 steps.
            if step != last_saved_step:
                save(step)
            paused = True
            break
        enforce_rollout_freshness(step)
        if collective_stop_requested():
            if step != last_saved_step:
                save(step)
            paused = True
            break
        for g in opt.param_groups:
            g["lr"] = lr * min((step + 1) / max(warmup, 1), 1.0)
        base = step * gbs
        if max_policy_age_steps and not strict_policy_age:
            train_ds.set_trainer_step(step)
        rl_micro = None
        freshness_wait_seconds = 0.0
        if strict_policy_age:
            def announce_freshness_wait():
                # Do not join the asynchronous publisher here: an I/O stall
                # must remain subject to the collective freshness timeout/stop.
                if is_rank0:
                    print(f"[unified-hf] strict freshness waiting before update {step + 1}", flush=True)

            rl_micro, freshness_wait_seconds = fetch_fresh_batch(
                primary_stream,
                lambda: [_micro_rows(rl_rows, base=base, micro=m, world=world, rank=rank, mbs=mbs)
                         for m in range(accum)],
                step=step, max_age=max_policy_age_steps, device=device,
                timeout_seconds=rollout_wait_timeout_seconds, poll_seconds=rollout_poll_seconds,
                stop_requested=collective_stop_requested,
                on_wait=announce_freshness_wait,
            )
            if rl_micro is None:
                if step != last_saved_step:
                    save(step)
                paused = True
                break
            if freshness_wait_seconds and is_rank0:
                print(f"[unified-hf] strict freshness recovered before update {step + 1}: "
                      f"blocked_seconds={freshness_wait_seconds:.3f}", flush=True)
        elif refresh_rounds_every_step:
            primary_stream.rescan()
            rl_micro = [_micro_rows(rl_rows, base=base, micro=m, world=world, rank=rank, mbs=mbs)
                        for m in range(accum)]

        # Megatron-style global normalization. Start with the data-only denominator
        # so micro-batch gradients stay well scaled. Recomputed top-k/top-p support is
        # only known after the forward pass; after all micro-batches, rescale the
        # accumulated gradient from this provisional denominator to the exact one.
        n_add = 0
        add_micro: list[list[dict]] = []
        if anchor_mode == "add":
            n_add, add_base = train_ds.additive_plan(step, gbs, anchor_add_basis)
            add_rows = [
                train_ds.anchor_row(add_base + j)
                for j in rank_slots(n_add, world=world, rank=rank)
            ]
            add_micro = [add_rows[i:i + mbs] for i in range(0, len(add_rows), mbs)]
        pretrain_micro: list[list[dict]] = []
        if pretrain_ds is not None:
            pretrain_micro = pretrain_micro_batches(pretrain_ds, step=step, rows_per_step=pretrain_rows,
                                                    world=world, rank=rank, mbs=mbs)
        # Anchor micro-batches first (a variable count per rank, all under no_sync);
        # the RL micro-batches follow and the last one, present on every rank,
        # is the synchronized pass.
        if rl_micro is None:
            rl_micro = [
                _micro_rows(rl_rows, base=base, micro=m, world=world, rank=rank, mbs=mbs)
                for m in range(accum)
            ]
        if trim_rows:
            rl_micro = group_rows_by_length([r for rows in rl_micro for r in rows], mbs)
        micro_rows = add_micro + pretrain_micro + rl_micro
        n_add_micro, n_pretrain_micro = len(add_micro), len(pretrain_micro)
        if sep_sft:
            sep_sums = separated_anchor_step(
                add_micro, rl_micro, ddp=ddp, model=model, opt=opt, device=device, alpha=sep_alpha,
                loss_norm=loss_norm, top_k=train_top_k, top_p=train_top_p, temperature=train_temperature,
                trim_rows=trim_rows, logprob_chunk_size=logprob_chunk_size,
            )
            num_acc = sep_sums[0] + sep_sums[2]
            local_den = sep_sums[1] + sep_sums[3]
            kl_acc = torch.zeros((), device=device)
            kl_n_acc = torch.zeros((), device=device)
        else:
            provisional_local_den = torch.stack(
                [
                    loss_denominator(r["loss_mask"], loss_norm, r.get("denom_mask"))
                    for rows in micro_rows
                    for r in rows
                ]
            ).sum().to(device)
            provisional_global_den = provisional_local_den.clone()
            dist.all_reduce(provisional_global_den)
            provisional_denom = torch.clamp(provisional_global_den, min=1.0)

            num_acc = torch.zeros((), device=device)
            local_den = torch.zeros((), device=device)
            kl_acc = torch.zeros((), device=device)
            kl_n_acc = torch.zeros((), device=device)
            opt.zero_grad(set_to_none=True)
            n_micro = len(micro_rows)
            for m in range(n_micro):
                b = _stack(micro_rows[m], device, trim=trim_rows)
                ctx = ddp.no_sync() if m < n_micro - 1 else torch.enable_grad()
                with ctx:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = ddp(input_ids=b["tokens"]).logits
                    num, micro_den = weighted_ce_sums(
                        logits,
                        b["labels"],
                        b["loss_mask"],
                        b.get("denom_mask"),
                        norm=loss_norm,
                        top_k=train_top_k,
                        top_p=train_top_p,
                        temperature=train_temperature,
                        sampling_filter=b.get("sampling_filter"),
                        logprob_chunk_size=logprob_chunk_size,
                    )
                    total = num
                    if kl_model is not None:
                        if b.get("denom_mask") is None:
                            raise ValueError("UNIFIED_KL_COEF needs response-token rows (denom_mask); set UNIFIED_LOSS_NORM=response_tokens")
                        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                            ref_logits = kl_model(input_ids=b["tokens"]).logits
                        kl_sum, kl_n = kl_penalty_sums(
                            logits, ref_logits, b["labels"], b["denom_mask"],
                            sampling_filter=b.get("sampling_filter"),
                            logprob_chunk_size=logprob_chunk_size,
                        )
                        del ref_logits
                        total = num + kl_coef * kl_sum
                        kl_acc += kl_sum.detach()
                        kl_n_acc += float(kl_n)
                    (total * world / provisional_denom).backward()
                num_acc += num.detach()
                local_den += micro_den.detach()
                stream_slot = 2 if m < n_add_micro else (4 if m < n_add_micro + n_pretrain_micro else 0)
                stream_interval[stream_slot] += num.detach()
                stream_interval[stream_slot + 1] += micro_den.detach()

            global_den = local_den.clone()
            dist.all_reduce(global_den)
            exact_denom = torch.clamp(global_den, min=1.0)
            grad_rescale = provisional_denom / exact_denom
            if float(grad_rescale) != 1.0:
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(grad_rescale)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        step += 1
        if consumed_age_interval and step % consumed_age_interval == 0:
            age_metrics = consumed_policy_age_metrics(
                rl_micro, trainer_step=step - 1, device=device,
                max_policy_age_steps=max_policy_age_steps,
            )
            age_metrics["strict_policy_age"] = int(strict_policy_age)
            age_metrics["freshness_wait_seconds"] = freshness_wait_seconds
            if is_rank0:
                print("[unified-hf] consumed_policy_age " + json.dumps(age_metrics, sort_keys=True), flush=True)
        matched_final = False
        if match_rl_steps:
            local_rl = torch.tensor(
                sum(int(bool(r["sampling_filter"])) for rows in micro_rows for r in rows),
                device=device, dtype=torch.float64,
            )
            dist.all_reduce(local_rl)
            rl_rows_consumed += int(local_rl.item())
            if rl_rows_consumed >= match_rl_target_rows and step < train_iters:
                train_iters = step   # this step is the matched endpoint: final save, eval, milestone
                matched_final = True
            elif (type(train_ds).__name__ != "MixedEpisodeDataset" and getattr(train_ds, "match_rl_horizon", None) is None
                  and step % log_interval == 0 and is_rank0):
                r_now = float(getattr(train_ds, "anchor_fraction", 0.0))
                target_file.write_text(f"{match_rl_projected_steps(match_rl_steps, gbs, r_now)}\n")
            if matched_final and is_rank0:
                target_file.write_text(f"{step}\n")
                print(f"[unified-hf] match_rl: RL rows {rl_rows_consumed:,} reached the target at step {step}; finishing here", flush=True)

        if runtime_proof_step and step == runtime_proof_step:
            publish_runtime_proof(step)

        if step % log_interval == 0:
            n, d = global_loss_sums(num_acc, local_den)
            kl_mean = None
            if kl_model is not None:
                ks, kn = global_loss_sums(kl_acc, kl_n_acc)
                kl_mean = ks / max(kn, 1.0)
            lm_loss = n / max(d, 1.0)
            extra_note = f" | loss tokens: {int(d)}" if log_loss_tokens else ""
            if sep_sft:
                # The separated objective (what was optimized) replaces the shared token mean.
                sep_global = sep_sums.clone()
                dist.all_reduce(sep_global)
                sep_rl_term = float(sep_global[0]) / max(float(sep_global[1]), 1.0)
                sep_demo_nll = float(sep_global[2]) / max(float(sep_global[3]), 1.0)
                lm_loss = sep_rl_term + sep_alpha * sep_demo_nll
                extra_note += (f" | sep rl term: {sep_rl_term:.6E} | sep demo nll: {sep_demo_nll:.6E}"
                               f" | T_rl: {int(sep_global[1])} | T_demo: {int(sep_global[3])}")
            if pretrain_ds is not None or log_stream_tokens:
                # Loss-token counts per stream summed over the steps since the previous log line (the sum of
                # these over a run is the realized token exposure), and the demo / pretraining NLL over them.
                st = stream_interval.clone()
                dist.all_reduce(st)
                stream_interval.zero_()
                extra_note += (f" | interval tokens rl/demo/pretrain: {int(st[1])}/{int(st[3])}/{int(st[5])}"
                               f" | demo nll: {float(st[2]) / max(float(st[3]), 1.0):.6E}"
                               f" | pretrain nll: {float(st[4]) / max(float(st[5]), 1.0):.6E}"
                               f" | pretrain rows added: {pretrain_rows}")
            if is_rank0:
                dt = (time.monotonic() - t_log) / log_interval
                t_log = time.monotonic()
                print(
                    f"[unified-hf] iteration {step:>8}/{train_iters} | consumed samples: {step * gbs} | "
                    f"ms per iteration: {dt * 1e3:.1f} | lm loss: {lm_loss:.6E} | "
                    f"grad norm: {float(grad_norm):.3f} | lr: {opt.param_groups[0]['lr']:.3E}"
                    + (f" | kl: {kl_mean:.6E}" if kl_mean is not None else "")
                    + (f" | anchor rows added: {n_add}" if anchor_mode == "add" else "")
                    + (f" | rl rows: {rl_rows_consumed}/{match_rl_target_rows}" if match_rl_steps else "")
                    + extra_note,
                    flush=True,
                )
        if (
            valid_ds is not None
            and eval_interval
            and (step % eval_interval == 0 or step == train_iters)
        ):
            _, consumed_valid = evaluate(step, consumed_valid)
        # A deadline/signal can arrive immediately after the optimizer update.
        # Evaluate a just-committed scheduled/final step before honoring that
        # stop, otherwise a resumed allocation starts after this step and its
        # deterministic RL/mid-training validation point is lost forever.
        preserve_for_eval = bool(
            eval_milestone_interval and step % eval_milestone_interval == 0
        ) or matched_final   # the matched endpoint is a milestone too
        if match_rl_steps and is_rank0 and (step % save_interval == 0 or step == train_iters or preserve_for_eval):
            tmp = match_rl_file.with_name(f".{match_rl_file.name}.tmp-{os.getpid()}")
            tmp.write_text(json.dumps({"step": step, "rl_rows_consumed": rl_rows_consumed, "target_rows": match_rl_target_rows}) + "\n")
            os.replace(tmp, match_rl_file)
        if collective_stop_requested():
            if step != last_saved_step:
                save(step, preserve_for_eval=preserve_for_eval)
            paused = True
            break
        if step % save_interval == 0 or step == train_iters or preserve_for_eval:
            save(step, preserve_for_eval=preserve_for_eval)
        elif publish_interval and step % publish_interval == 0:
            publish_policy(step)

    wait_publish()
    if is_rank0:
        state = "paused" if paused else "training complete"
        print(f"[unified-hf] {state} at step {step}/{train_iters}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
