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

"""Pure rollout logic of the generator (prompt selection, group weights, round writing).

Everything here is vLLM-free and torch-free (numpy only) so it can be unit
tested in plain CI. The vLLM-coupled callers feed in plain python structures:
per-prompt groups of (token_ids, text, finished) completions.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from unified_training.episodes import RoundWriter, normalize_weights

CHAT_TEMPLATE = "<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"

# Prompt templates selectable with UNIFIED_PROMPT_TEMPLATE.
# "chatml" (default) is the historical CHAT_TEMPLATE with its "<|im_end|>" stop string.
# "raw" uses the question text verbatim (a plain completion prompt, e.g. one ending in
# "\n\nAnswer: ") and stops on the tokenizer's EOS token only (the reasoning-puzzle
# setting: a GPT-2-tokenizer model whose chat template is the identity).
PROMPT_TEMPLATES = {"chatml": CHAT_TEMPLATE, "raw": "{question}"}
PROMPT_STOP_STRINGS = {"chatml": ["<|im_end|>"], "raw": []}


def prompt_template(name: str) -> tuple[str, list[str]]:
    """(format string, stop strings) of a named prompt template."""
    if name not in PROMPT_TEMPLATES:
        raise ValueError(f"unknown prompt template {name!r}; expected one of {sorted(PROMPT_TEMPLATES)}")
    return PROMPT_TEMPLATES[name], list(PROMPT_STOP_STRINGS[name])


STANDARD_PASS_KS = (1, 2, 4, 8, 16)


def pass_at_k(n: int, correct: int, k: int) -> float:
    """Return the unbiased pass@k estimator from ``n`` sampled responses."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not 0 <= correct <= n:
        raise ValueError(
            f"correct must satisfy 0 <= correct <= n, got correct={correct}, n={n}"
        )
    if not 1 <= k <= n:
        raise ValueError(f"k must satisfy 1 <= k <= n, got k={k}, n={n}")
    if n - correct < k:
        return 1.0
    return 1.0 - math.comb(n - correct, k) / math.comb(n, k)


def mean_pass_at_ks(
    correctness_groups: Sequence[Sequence[bool]],
    *,
    ks: Iterable[int] = STANDARD_PASS_KS,
) -> dict[int, float]:
    """Average unbiased pass@k estimates across equally sampled prompts."""
    if not correctness_groups:
        raise ValueError("correctness_groups must not be empty")
    sample_count = len(correctness_groups[0])
    if sample_count <= 0:
        raise ValueError("each correctness group must contain at least one sample")
    normalized_ks = tuple(int(k) for k in ks)
    if not normalized_ks or len(set(normalized_ks)) != len(normalized_ks):
        raise ValueError(f"ks must be non-empty and unique, got {normalized_ks}")
    for k in normalized_ks:
        if not 1 <= k <= sample_count:
            raise ValueError(
                f"k must satisfy 1 <= k <= samples_per_prompt={sample_count}, got {k}"
            )

    totals = {k: 0.0 for k in normalized_ks}
    for index, group in enumerate(correctness_groups):
        if len(group) != sample_count:
            raise ValueError(
                f"correctness group {index} has {len(group)} samples, expected {sample_count}"
            )
        if any(not isinstance(value, bool) for value in group):
            raise TypeError(f"correctness group {index} contains a non-bool value")
        correct = sum(group)
        for k in normalized_ks:
            totals[k] += pass_at_k(sample_count, correct, k)
    return {k: totals[k] / len(correctness_groups) for k in normalized_ks}


def extract_answer(text: str) -> str:
    """Final-number extraction — must stay identical to eval_hf*.py."""
    m = re.findall(r"####\s*(-?[\d,]+)", text) or re.findall(r"(-?[\d][\d,]*\.?\d*)", text)
    return (m[-1].replace(",", "").rstrip(".") if m else "")


def select_prompt_slice(n_prompts: int, offset: int, limit: int) -> tuple[int, list[int]]:
    """Rotating, pass-shuffled prompt selection for a continuously running producer.

    ``offset`` is a monotonically growing global counter (batch * chunk). Each full
    pass over the prompt set gets its own deterministic shuffle, so successive
    batches walk the whole set in a different order every pass. Returns
    (pass_number, indices); the slice is truncated at a pass boundary (no wrap
    inside a batch) so a batch never mixes two passes.
    """
    if n_prompts <= 0:
        return 0, []
    pass_num, off = divmod(int(offset), n_prompts)
    perm = np.random.default_rng(pass_num).permutation(n_prompts)
    take = min(int(limit), n_prompts - off) if limit else n_prompts - off
    return pass_num, [int(i) for i in perm[off : off + take]]


def select_prompt_shard(
    n_prompts: int,
    local_batch: int,
    limit: int,
    rank: int,
    world: int,
) -> tuple[int, list[int]]:
    """Select one deterministic, disjoint prompt chunk for a producer rank.

    Every shuffled pass is divided into ``ceil(n_prompts / limit)`` chunks. The
    pair ``(local_batch, rank)`` maps to global chunk ``local_batch * world +
    rank``, so all producer ranks cover each pass exactly once without needing
    to run at the same speed or coordinate through shared mutable state.

    Returns ``(pass_number, indices)``. A chunk never crosses a pass boundary;
    the last chunk of a pass can therefore contain fewer than ``limit`` rows.
    Callers must persist ``local_batch`` across process restarts.
    """
    if n_prompts <= 0:
        raise ValueError(f"n_prompts must be positive (got {n_prompts})")
    if local_batch < 0:
        raise ValueError(f"local_batch must be non-negative (got {local_batch})")
    if limit <= 0:
        raise ValueError(f"limit must be positive (got {limit})")
    if world <= 0:
        raise ValueError(f"world must be positive (got {world})")
    if not 0 <= rank < world:
        raise ValueError(f"rank must satisfy 0 <= rank < world (got rank={rank}, world={world})")

    chunks_per_pass = (n_prompts + limit - 1) // limit
    global_chunk = local_batch * world + rank
    pass_num, chunk_in_pass = divmod(global_chunk, chunks_per_pass)
    start = chunk_in_pass * limit
    perm = np.random.default_rng(pass_num).permutation(n_prompts)
    return pass_num, [int(i) for i in perm[start : start + limit]]


def response_token_budget(prompt_len: int, max_model_len: int, max_tokens: int) -> int:
    """Maximum safe response length for a fixed-length policy/trainer context.

    Returns zero when the prompt itself cannot fit and should be skipped.
    """
    if prompt_len < 0:
        raise ValueError(f"prompt_len must be non-negative, got {prompt_len}")
    if max_model_len <= 0:
        raise ValueError(f"max_model_len must be positive, got {max_model_len}")
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}")
    return max(0, min(max_tokens, max_model_len - prompt_len))


def truncated_without_eos(token_ids, budget: int, eos_id: int | None) -> bool:
    """The REQUIRE_EOS truncation rule (UNIFIED_REQUIRE_EOS=1).

    A completion is truncated iff it filled its whole response budget AND its last
    token is not EOS. An EOS emitted as the budget's final token is a finished answer
    (vLLM reports finish_reason "length" for it); a shorter completion is finished
    whatever its last token.
    """
    return len(token_ids) >= budget and (not token_ids or token_ids[-1] != eos_id)


@dataclass
class GroupStats:
    groups_all_wrong: int = 0
    groups_allcorrect_dropped: int = 0
    groups_easy_dropped: int = 0
    groups_easy_kept: int = 0
    n_truncated: int = 0
    n_wmax_clipped_groups: int = 0
    solved: int = 0
    n_groups: int = 0


@dataclass
class Episode:
    tokens: list[int]
    prompt_len: int
    weight: float  # raw group weight, pre-normalization
    group_id: int
    reward: float = 1.0
    text: str = ""


def _easy_group_kept(group_id: int, keep_frac: float) -> bool:
    """Deterministic per-group keep decision for above-threshold (easy) groups.

    Knuth multiplicative hash of the group id -> [0, 1); stable across ranks,
    restarts, and batch order so a group's fate never flip-flops between waves.
    """
    if keep_frac <= 0.0:
        return False
    if keep_frac >= 1.0:
        return True
    return ((group_id * 2654435761) % (1 << 32)) / float(1 << 32) < keep_frac


def weight_groups(
    groups,
    *,
    wmax: float,
    weight_eps: float = 0.0,
    keep_allcorrect: bool,
    hard_filter: bool,
    emit_zero_weight: bool = False,
    drop_truncated: bool = True,
    force_truncated_incorrect: bool = False,
    max_group_acc: float = 1.1,
    easy_keep_frac: float = 0.0,
    drop_all_wrong: bool = False,
    stats: GroupStats | None = None,
    baseline: bool = False,
    baseline_eps: float = 1e-6,
) -> tuple[list[Episode], GroupStats]:
    """Off-policy maxRL group weighting (A = r/mean) as a data filter.

    ``groups`` yields (group_id, prompt_token_ids, completions) where each
    completion is (token_ids, text, finished, correct). By default this is a
    compact positive-only data filter: length-truncated completions are dropped,
    all-wrong groups are dropped, and wrong completions in partially-correct
    groups are omitted (A = 0 -> no numerator contribution).

    Set ``emit_zero_weight=True`` and ``drop_truncated=False`` to match NeMo-RL's
    token-level maxRL denominator more closely: zero-advantage wrong/all-wrong
    responses are written with response weights 0, so a trainer using a separate
    response-token denominator can count them even though they add no numerator.
    NeMo-RL still scores length-truncated responses unless overlong filtering is
    enabled; ``force_truncated_incorrect=True`` is an extra safety knob, not the
    reference-parity behavior.

    ``weight_eps`` smooths the weight denominator: w = 1/(p_hat + weight_eps),
    matching the estimator-side maxrl_eps hyperparameter. At the default 0.0 the
    historical w = 1/p_hat is unchanged. A positive value softly bounds the
    lone-success amplification at 1/(1/G + eps); wmax still applies on top as a
    hard ceiling.

    ``max_group_acc`` drops groups the policy already (mostly) solves: a group
    whose empirical pass rate p_hat is >= the threshold is discarded before
    episode emission, except for a deterministic ``easy_keep_frac`` fraction
    kept (by group-id hash) as anti-regression coverage. The default 1.1 keeps
    the filter off (p_hat cannot exceed 1.0). This trades oversampled easy
    groups for batch slots on partially-solved problems; it runs before the
    ``keep_allcorrect`` check, which still applies to the kept-easy remainder.

    ``drop_all_wrong`` discards groups with no correct completion even when
    ``emit_zero_weight`` is set: those groups add nothing to the
    loss numerator but otherwise occupy batch rows, count in a response-token
    loss denominator and in the ``response_tokens`` anchor-dose basis.

    ``baseline=True`` (UNIFIED_MAXRL_BASELINE=1) emits the on-policy maxRL
    advantage WITH the group-mean baseline (the staged-RL references):
    ``A = (r - mean) / (mean + baseline_eps)`` for EVERY completion of the group,
    in float32 (as verl's maxrl advantage estimator).  Correct -> (1-p)/(p+eps),
    wrong -> -p/(p+eps), all-wrong and all-correct groups -> 0.  Signed weights need
    the response-token denominator and zero-weight emission, so this mode requires
    ``emit_zero_weight=True`` and rejects ``hard_filter``/``weight_eps``; ``wmax`` is
    not applied.
    """
    if baseline and (hard_filter or not emit_zero_weight or weight_eps != 0.0):
        raise ValueError(
            "baseline maxRL needs emit_zero_weight=True, hard_filter=False and weight_eps=0"
        )
    if baseline and baseline_eps <= 0:
        raise ValueError(f"baseline_eps must be positive, got {baseline_eps}")
    if weight_eps < 0:
        raise ValueError(f"weight_eps must be non-negative, got {weight_eps}")
    if not 0.0 <= easy_keep_frac <= 1.0:
        raise ValueError(f"easy_keep_frac must be in [0, 1], got {easy_keep_frac}")
    if max_group_acc <= 0.0:
        raise ValueError(f"max_group_acc must be positive, got {max_group_acc}")
    stats = stats or GroupStats()
    episodes: list[Episode] = []
    for group_id, prompt_ids, completions in groups:
        comps = []
        for token_ids, text, finished, correct in completions:
            if not finished:
                stats.n_truncated += 1
                if drop_truncated:
                    continue
                if force_truncated_incorrect:
                    # Safety mode: an unfinished completion hit max_tokens without the turn stop.
                    # This is stricter than NeMo-RL's overlong_filtering=false behavior.
                    correct = False
            comps.append((token_ids, text, correct))
        if not comps:
            continue
        stats.n_groups += 1
        p_hat = sum(1.0 for *_x, c in comps if c) / len(comps)
        if p_hat > 0:
            stats.solved += 1
        if p_hat == 0.0:
            stats.groups_all_wrong += 1
            if not emit_zero_weight or drop_all_wrong:
                continue
        if p_hat >= max_group_acc:
            if _easy_group_kept(group_id, easy_keep_frac):
                stats.groups_easy_kept += 1
            else:
                stats.groups_easy_dropped += 1
                continue
        if p_hat == 1.0 and not keep_allcorrect:
            stats.groups_allcorrect_dropped += 1
            continue
        w = 0.0 if p_hat == 0.0 else (1.0 if hard_filter else min(1.0 / (p_hat + weight_eps), wmax))
        if p_hat > 0.0 and not hard_filter and not baseline and 1.0 / (p_hat + weight_eps) > wmax:
            stats.n_wmax_clipped_groups += 1
        if baseline:
            # float32, as verl: scores are a float32 tensor, torch.mean(stack(...)).
            mean32 = np.float32(sum(1.0 for *_x, c in comps if c)) / np.float32(len(comps))
            adv_correct = float((np.float32(1.0) - mean32) / (mean32 + np.float32(baseline_eps)))
            adv_wrong = float((np.float32(0.0) - mean32) / (mean32 + np.float32(baseline_eps)))
        for token_ids, text, correct in comps:
            if not correct and not emit_zero_weight:
                continue
            if baseline:
                ep_weight = adv_correct if correct else adv_wrong
            else:
                ep_weight = w if correct else 0.0
            episodes.append(
                Episode(
                    tokens=list(prompt_ids) + list(token_ids),
                    prompt_len=len(prompt_ids),
                    weight=ep_weight,
                    group_id=group_id,
                    reward=1.0 if correct else 0.0,
                    text=text,
                )
            )
    return episodes, stats


def write_episode_round(
    out_dir,
    episodes: list[Episode],
    *,
    policy_version: int,
    hard_filter: bool,
    normalize_episode_weights: bool = True,
    manifest_extra: dict,
) -> dict:
    """Write a round, optionally normalizing positive weights to token-mean 1.

    ``normalize_episode_weights=False`` is needed for NeMo-RL maxRL parity when
    zero-advantage rows are emitted and the trainer normalizes by all response
    tokens. In that mode the raw ``1 / p_hat`` weights are already the desired
    advantages; rescaling them changes the objective.
    """
    if not episodes:
        raise ValueError("no episodes to write")
    resp_lens = np.array([len(e.tokens) - e.prompt_len for e in episodes], dtype=np.float64)
    raw_w = np.array([e.weight for e in episodes], dtype=np.float64)
    weighted_tokens = float(np.dot(raw_w, resp_lens))
    scale = (
        1.0
        if hard_filter or not normalize_episode_weights or weighted_tokens <= 0.0
        else normalize_weights(raw_w, resp_lens)
    )

    writer = RoundWriter(out_dir)
    for ep, sw in zip(episodes, raw_w * scale):
        weights = np.zeros(len(ep.tokens), dtype=np.float32)
        weights[ep.prompt_len :] = sw
        writer.add(
            np.array(ep.tokens, dtype=np.int32), weights,
            prompt_len=ep.prompt_len, reward=ep.reward, weight=float(sw),
            policy_version=policy_version, group_id=ep.group_id,
        )
    return writer.finalize({
        **manifest_extra,
        "weight_scale": float(scale),
        "hard_filter": hard_filter,
        "normalize_episode_weights": bool(normalize_episode_weights),
    })
