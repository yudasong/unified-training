"""Training exposure and model-FLOP estimates of the controlled experiments (Appendix A.2).

    C_total      = 6 N (T_pre + T_mid) + 8 N T_rollout + 6 N T_mix
    C_additional = C_total - 6 N T_pre

T_pre: the shared 20,000,014,336-token pretraining stream. T_mid: midtraining tokens before RL (8147 x k x 2048 for the
unified init with k sequences per update; E x 509,942,713 for E staged epochs). T_rollout: RL prompt + response tokens
(prompts: 512 per update x the mean prompt length of the RL mix; responses: the trainer's logged loss-token count every
ten updates minus m x the mean midtraining response length). T_mix: m complete midtraining examples per update.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

N_PARAMS = 123_587_328
T_PRE = 20_000_014_336  # 38147 updates x 256 x 2048
CPT_UPDATES, SEQ = 8147, 2048
CORPUS_TOKENS, CORPUS_EXAMPLES = 509_942_713, 796_175
CORPUS_RESPONSE_TOKENS = 304_424_398  # response + EOS tokens of the corpus (the supervised tokens during RL)
MID_EXAMPLE_TOKENS = CORPUS_TOKENS / CORPUS_EXAMPLES  # 640.49
MID_RESPONSE_TOKENS = CORPUS_RESPONSE_TOKENS / CORPUS_EXAMPLES  # 382.36
RL_ROWS = 512
# mean GPT-2 tokens per RL prompt of the two prompt mixes
PROMPT_TOKENS = {"base": 297.4218, "extended": 315.4656}
# init ids (the checkpoint names): unified inits by midtraining tokens mixed into pretraining (k sequences per update)
MIXED_K = {"mixed-133M": 8, "mixed-267M": 16, "mixed-534M": 32, "mixed-1.07B": 64, "mixed-2.14B": 128}
STAGED_EPOCHS = {"staged-1ep-lr2e-4": 1, "staged-1ep-lr5e-5": 1, "staged-2ep-lr2e-4": 2, "staged-2ep-lr5e-5": 2,
                 "staged-4ep-lr2e-4": 4, "staged-4ep-lr5e-5": 4}  # fmt: skip
_LOG = re.compile(r"iteration\s+(\d+)/\d+ \|.*\| loss tokens: (\d+)")


def midtraining_tokens_before_rl(init: str) -> float:
    """T_mid of an init id (mixed-<tokens>, staged-<E>ep-lr<LR>, or pretrain-only)."""
    if init in MIXED_K:
        return CPT_UPDATES * MIXED_K[init] * SEQ
    if init in STAGED_EPOCHS:
        return STAGED_EPOCHS[init] * CORPUS_TOKENS
    if init == "pretrain-only":
        return 0
    raise ValueError(init)


def corpus_passes(updates: int, m: int) -> float:
    """Midtraining epochs during RL: U m / 796175 (the horizontal axis of Figure 4)."""
    return updates * m / CORPUS_EXAMPLES


def logged_loss_tokens(log_paths: Iterable[str | Path]) -> dict[int, int]:
    """{update: loss tokens of that update} from trainer logs (one line every UNIFIED_LOG_INTERVAL updates)."""
    out = {}
    for p in log_paths:
        for line in Path(p).read_text(errors="replace").splitlines():
            m = _LOG.search(line)
            if m:
                out[int(m[1])] = int(m[2])
    return out


def rl_response_tokens(loss_tokens: dict[int, int], m: int, upto: int, interval: int = 10) -> float:
    """Cumulative RL response tokens through update ``upto``: each logged update stands for ``interval`` updates."""
    return sum(interval * (t - m * MID_RESPONSE_TOKENS) for s, t in loss_tokens.items() if s <= upto)


def flops(init: str, setting: str, m: int, updates: int, rl_response: float, additional: bool = False) -> float:
    """Model FLOPs through ``updates`` RL updates (0 = the initialization)."""
    t_rollout = updates * RL_ROWS * PROMPT_TOKENS[setting] + rl_response
    c = 6 * N_PARAMS * (T_PRE + midtraining_tokens_before_rl(init)) + 8 * N_PARAMS * t_rollout
    c += 6 * N_PARAMS * updates * m * MID_EXAMPLE_TOKENS
    return c - 6 * N_PARAMS * T_PRE if additional else c
