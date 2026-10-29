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

"""Weighted episode shard format — the unified rollout->trainer data contract.

One round directory holds one round of rollouts:

    <round_dir>/
      tokens.bin     flat little-endian int32 token ids, all episodes back to back
      weights.bin    flat little-endian float32 per-token loss weights (1:1 with tokens;
                     0 on prompt tokens, the episode weight on response tokens)
      index.npy      structured array, one row per episode (INDEX_DTYPE)
      manifest.json  round metadata (counts, reward stats, policy_version, ...)
      DONE           marker written last; readers must ignore dirs without it

numpy-only on purpose: the generator and data builders import it without torch.

Weights are *absolute* multipliers on the per-token CE loss. The trainer's loss
is self-normalized (sum(w*ce)/sum(w)), so producers should scale weights to a
mean of ~1 over weighted tokens (see ``normalize_weights``) to keep token
accounting and gradient scale comparable across rounds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

INDEX_DTYPE = np.dtype(
    [
        ("offset", "<i8"),  # start of the episode in tokens.bin/weights.bin (in elements)
        ("prompt_len", "<i4"),
        ("total_len", "<i4"),  # prompt + response (+ appended eos), in tokens
        ("reward", "<f4"),
        ("weight", "<f4"),  # scalar episode weight (what response tokens carry)
        ("policy_version", "<i4"),
        ("group_id", "<i4"),  # prompt/group index within the round
    ]
)

TOKENS_FILE = "tokens.bin"
WEIGHTS_FILE = "weights.bin"
INDEX_FILE = "index.npy"
MANIFEST_FILE = "manifest.json"
DONE_FILE = "DONE"


def round_order_key(path: str | Path) -> tuple[int, int, str]:
    """Order generator rounds by global chunk, with lexical fallback.

    Unified generators publish names ending in ``-q<global_chunk>``. Sorting
    those names lexically groups every batch from generator 0 before generator
    1, which makes a round-count recency window collapse onto one generator.
    The global chunk is already interleaved as ``batch * world + rank`` and is
    therefore the correct deterministic order for a multi-generator stream.
    """
    name = Path(path).name
    marker = name.rfind("-q")
    if marker >= 0:
        global_chunk = name[marker + 2 :]
        if global_chunk.isdigit():
            return (0, int(global_chunk), name)
    return (1, 0, name)


class RoundWriter:
    """Accumulate episodes in memory and write one round directory atomically.

    Rounds are small (tens of MB), so buffering in RAM and writing once keeps
    the format trivially crash-safe: the DONE marker is written last and
    readers skip rounds without it.
    """

    def __init__(self, round_dir: str | Path) -> None:
        self.round_dir = Path(round_dir)
        self._tokens: list[np.ndarray] = []
        self._weights: list[np.ndarray] = []
        self._index: list[tuple] = []
        self._offset = 0

    def add(
        self,
        tokens: np.ndarray,
        weights: np.ndarray,
        *,
        prompt_len: int,
        reward: float,
        weight: float,
        policy_version: int,
        group_id: int,
    ) -> None:
        tokens = np.asarray(tokens, dtype=np.int32)
        weights = np.asarray(weights, dtype=np.float32)
        if tokens.ndim != 1 or tokens.shape != weights.shape:
            raise ValueError(f"tokens/weights must be 1-D and aligned, got {tokens.shape} vs {weights.shape}")
        if not 0 < prompt_len <= len(tokens):
            raise ValueError(f"prompt_len {prompt_len} out of range for episode of {len(tokens)} tokens")
        if np.any(weights[:prompt_len] != 0.0):
            raise ValueError("prompt tokens must carry zero weight")
        self._tokens.append(tokens)
        self._weights.append(weights)
        self._index.append((self._offset, prompt_len, len(tokens), reward, weight, policy_version, group_id))
        self._offset += len(tokens)

    def __len__(self) -> int:
        return len(self._index)

    def finalize(self, manifest: dict | None = None) -> dict:
        if not self._index:
            raise ValueError("no episodes added; refusing to write an empty round")
        self.round_dir.mkdir(parents=True, exist_ok=True)
        tokens = np.concatenate(self._tokens)
        weights = np.concatenate(self._weights)
        index = np.array(self._index, dtype=INDEX_DTYPE)

        tokens.astype("<i4").tofile(self.round_dir / TOKENS_FILE)
        weights.astype("<f4").tofile(self.round_dir / WEIGHTS_FILE)
        np.save(self.round_dir / INDEX_FILE, index)

        out = dict(manifest or {})
        supervised_token_count = int(np.count_nonzero(weights > 0))
        out.update(
            n_episodes=int(len(index)),
            n_tokens=int(tokens.size),
            # ``n_weighted_tokens`` is retained for format compatibility.  It
            # is a binary count of trained-on positions, never a sum of loss
            # coefficient magnitudes.
            n_weighted_tokens=supervised_token_count,
            n_supervised_tokens=supervised_token_count,
            supervised_token_count_basis="count(token_weight > 0)",
            max_token_id=int(tokens.max()) if tokens.size else -1,
            mean_response_len=float(np.mean(index["total_len"] - index["prompt_len"])),
            mean_reward=float(np.mean(index["reward"])),
            mean_weight=float(np.mean(index["weight"])),
        )
        with (self.round_dir / MANIFEST_FILE).open("w") as f:
            json.dump(out, f, indent=2)
        (self.round_dir / DONE_FILE).touch()
        return out


@dataclass
class Round:
    """A finalized round opened for reading.

    Token/weight arrays are memmapped lazily: ``load_round`` leaves them
    ``None`` and ``episode()`` opens them on first access, because each open
    round pins two file descriptors and a long run folds thousands of rounds.
    ``release()`` drops them again once a round can no longer be sampled.
    """

    round_dir: Path
    tokens: np.ndarray | None
    weights: np.ndarray | None
    index: np.ndarray
    manifest: dict = field(default_factory=dict)

    def ensure_loaded(self) -> None:
        """Open the payload memmaps if this round is unloaded or released."""
        if self.tokens is None:
            self.tokens = np.memmap(self.round_dir / TOKENS_FILE, dtype="<i4", mode="r")
            self.weights = np.memmap(self.round_dir / WEIGHTS_FILE, dtype="<f4", mode="r")

    def release(self) -> None:
        """Drop the payload memmaps so their file descriptors close.

        Bookkeeping (index/manifest) stays, so the round remains addressable
        and ``episode()`` transparently reopens if it is ever needed again.
        """
        self.tokens = None
        self.weights = None

    def episode(self, i: int) -> tuple[np.ndarray, np.ndarray, np.void]:
        """Return (tokens, weights, index row) for episode ``i`` as plain arrays."""
        self.ensure_loaded()
        row = self.index[i]
        lo, hi = int(row["offset"]), int(row["offset"]) + int(row["total_len"])
        return np.asarray(self.tokens[lo:hi]), np.asarray(self.weights[lo:hi]), row


def rank_balanced_recent_round_indices(
    rounds: list[Round],
    *,
    window: int,
    eligible_round_indices: set[int] | None = None,
) -> list[int]:
    """Select equally many newest rounds from every generator rank.

    Returns an empty list when the manifests are not one complete, consistent
    generator world or when ``window`` is not a positive world-size multiple.
    Callers can then use their generic recency fallback.
    """
    if window <= 0:
        return []
    allowed = (
        set(range(len(rounds)))
        if eligible_round_indices is None
        else eligible_round_indices
    )
    worlds: set[int] = set()
    rank_rounds: dict[int, list[int]] = {}
    unmarked_rounds = 0
    for round_index, rnd in enumerate(rounds):
        if round_index not in allowed:
            continue
        has_rank = "generator_rank" in rnd.manifest
        has_world = "generator_world" in rnd.manifest
        if not has_rank and not has_world:
            unmarked_rounds += 1
            continue
        if not (has_rank and has_world):
            return []
        try:
            rank = int(rnd.manifest["generator_rank"])
            world = int(rnd.manifest["generator_world"])
        except (KeyError, TypeError, ValueError):
            return []
        if world <= 0 or rank < 0 or rank >= world:
            return []
        worlds.add(world)
        rank_rounds.setdefault(rank, []).append(round_index)

    if rank_rounds and unmarked_rounds:
        return []
    if len(worlds) != 1:
        return []
    world = next(iter(worlds))
    if window < world or window % world != 0 or not all(
        rank_rounds.get(rank) for rank in range(world)
    ):
        return []

    requested_per_rank = window // world
    available_per_rank = min(len(rank_rounds[rank]) for rank in range(world))
    take_per_rank = min(requested_per_rank, available_per_rank)
    selected: list[int] = []
    for rank in range(world):
        ordered = sorted(
            rank_rounds[rank], key=lambda index: round_order_key(rounds[index].round_dir)
        )
        selected.extend(ordered[-take_per_rank:])
    return sorted(
        selected, key=lambda index: round_order_key(rounds[index].round_dir)
    )


def is_round_dir(path: str | Path) -> bool:
    return (Path(path) / DONE_FILE).exists()


def load_round(round_dir: str | Path) -> Round:
    d = Path(round_dir)
    if not is_round_dir(d):
        raise FileNotFoundError(f"{d} is not a finalized round (missing {DONE_FILE})")
    index = np.load(d / INDEX_FILE)
    expected = int(index["offset"][-1] + index["total_len"][-1]) if len(index) else 0
    # Validate payload sizes from stat() rather than by opening memmaps: both
    # payloads are 4-byte scalars, and keeping two descriptors open per round
    # exhausts the process fd limit once a run has folded thousands of rounds.
    for name in (TOKENS_FILE, WEIGHTS_FILE):
        actual = (d / name).stat().st_size
        if actual != expected * 4:
            raise ValueError(f"{d}: {name} size {actual} bytes != index expectation {expected * 4}")
    manifest = json.loads((d / MANIFEST_FILE).read_text()) if (d / MANIFEST_FILE).exists() else {}
    return Round(round_dir=d, tokens=None, weights=None, index=index, manifest=manifest)


def load_rounds(dirs: list[str | Path]) -> list[Round]:
    """Load several rounds; each entry may be a round dir or a parent of round dirs."""
    rounds: list[Round] = []
    for entry in dirs:
        p = Path(entry)
        if is_round_dir(p):
            rounds.append(load_round(p))
            continue
        subdirs = (
            sorted(
                (d for d in p.iterdir() if d.is_dir() and is_round_dir(d)),
                key=round_order_key,
            )
            if p.is_dir()
            else []
        )
        if not subdirs:
            raise FileNotFoundError(f"{p} is neither a finalized round nor a parent of finalized rounds")
        rounds.extend(load_round(d) for d in subdirs)
    return rounds


def pack_token_docs(docs, window_len: int, eos_id: int):
    """Concat-and-chop documents into fixed ``window_len`` token windows (midtrain-style).

    ``docs`` yields token-id lists; an EOS is appended to each. Yields full int32
    windows only (the final partial window is dropped). Used to turn corpus data into
    anchor "episodes" so RL rollouts and plain training data share one shard format.
    """
    buf: list[int] = []
    for doc in docs:
        buf.extend(doc)
        buf.append(eos_id)
        while len(buf) >= window_len:
            yield np.array(buf[:window_len], dtype=np.int32)
            buf = buf[window_len:]


def anchor_row_fraction(
    token_fraction: float,
    rl_tokens_per_ep: float,
    anchor_tokens_per_ep: float,
    *,
    max_row_fraction: float = 1.0,
) -> float:
    """Row fraction r such that anchor rows carry ``token_fraction`` of trained tokens.

    Every position with ``token_weight > 0`` counts once, independently of the
    magnitude of its loss coefficient.  Anchor windows are ~full seq_length
    while RL episodes are short responses, so a naive row fraction
    over-anchors the gradient. Solves
    r*T_a / (r*T_a + (1-r)*T_r) = f  =>  r = f*T_r / (T_a*(1-f) + f*T_r).
    """
    if not 0.0 < token_fraction < 1.0:
        raise ValueError(f"token_fraction must be in (0, 1), got {token_fraction}")
    if rl_tokens_per_ep <= 0 or anchor_tokens_per_ep <= 0:
        raise ValueError("tokens-per-episode must be positive")
    if not 0.0 <= max_row_fraction <= 1.0:
        raise ValueError("max_row_fraction must be between 0 and 1")
    f = token_fraction
    row_fraction = f * rl_tokens_per_ep / (
        anchor_tokens_per_ep * (1.0 - f) + f * rl_tokens_per_ep
    )
    if row_fraction > max_row_fraction:
        raise ValueError(
            "derived anchor row fraction exceeds the proven holdout bound: "
            f"{row_fraction:.6f} > {max_row_fraction:.6f}"
        )
    return row_fraction


def validate_no_anchor_replay(
    total_samples: int,
    anchor_row_fraction: float,
    available_episodes: int,
) -> int:
    """Return required anchor draws, rejecting a shape that exceeds one pool pass.

    The two-way mixer emits exactly ``floor(total_samples * row_fraction)``
    anchor rows.  A fixed anchor starts a new shuffled epoch after
    ``available_episodes`` draws, so any larger request necessarily replays data.
    """
    draws = int(max(int(total_samples), 1) * float(anchor_row_fraction))
    available = int(available_episodes)
    if available < 0:
        raise ValueError("available anchor episodes must be non-negative")
    if draws > available:
        raise ValueError(
            "the requested training shape would replay the fixed anchor: "
            f"draws={draws:,} available_episodes={available:,}"
        )
    return draws


def row_fractions(token_fractions: list[float], tokens_per_ep: list[float]) -> list[float]:
    """Per-stream ROW fractions that realize the given per-stream TOKEN fractions.

    N-way generalization of :func:`anchor_row_fraction`. The trainer loss is
    token-normalized, so to give stream s a target share ``f_s`` of weighted
    *tokens* when its episodes carry ``T_s`` weighted tokens each, sample rows in
    proportion to ``f_s / T_s`` (a short episode must appear more often than a
    long window to contribute the same token mass). Returns row fractions summing
    to 1. ``token_fractions`` need not sum to 1 (they are renormalized).
    """
    if len(token_fractions) != len(tokens_per_ep):
        raise ValueError("token_fractions and tokens_per_ep length mismatch")
    if any(t <= 0 for t in tokens_per_ep):
        raise ValueError("tokens-per-episode must be positive")
    if any(f < 0 for f in token_fractions):
        raise ValueError("token fractions must be non-negative")
    w = [f / t for f, t in zip(token_fractions, tokens_per_ep)]
    s = sum(w)
    if s <= 0:
        raise ValueError("all token fractions are zero")
    return [x / s for x in w]


def mean_response_tokens(rounds: list["Round"], refs: list[tuple[int, int]]) -> float:
    """Mean trained-on tokens per episode over a ref pool.

    Each position with ``token_weight > 0`` contributes one token; coefficient
    magnitudes do not affect the count. ``total_len - prompt_len`` is only
    equivalent when every response token is supervised. Document-aware anchors
    can have variable-length episodes and chat anchors intentionally mask
    user/system tokens, so derive the blend denominator from the binary mask.
    """
    if not refs:
        raise ValueError("empty ref pool")
    refs_by_round: dict[int, list[int]] = {}
    for round_index, episode_index in refs:
        refs_by_round.setdefault(round_index, []).append(episode_index)
    total = 0
    for round_index, episode_indices in refs_by_round.items():
        round_ = rounds[round_index]
        if (
            len(episode_indices) == len(round_.index)
            and set(episode_indices) == set(range(len(round_.index)))
            and "n_weighted_tokens" in round_.manifest
        ):
            total += int(round_.manifest["n_weighted_tokens"])
            continue
        round_.ensure_loaded()
        for episode_index in episode_indices:
            row = round_.index[episode_index]
            start = int(row["offset"])
            stop = start + int(row["total_len"])
            total += int(np.count_nonzero(round_.weights[start:stop] > 0))
    return total / len(refs)


MIX_BASES = ("supervised_tokens", "weight", "response_tokens")


def mean_episode_mass(
    rounds: list["Round"],
    refs: list[tuple[int, int]],
    *,
    basis: str = "supervised_tokens",
) -> float:
    """Mean per-episode blend mass over a ref pool, under a chosen mixing basis.

    ``supervised_tokens`` counts positions with ``token_weight > 0`` (each once,
    coefficient magnitudes ignored) — data exposure. ``response_tokens`` counts
    every response position whether or not it carries a loss, so an RL episode
    contributes its full response length even when its answer was wrong and its
    weight is zero: the anchor share is then a share of ALL RL tokens generated,
    independent of the pass rate. ``weight`` sums the loss
    coefficients — the episode's share of the loss numerator, so an anchor
    fraction under this basis targets gradient mass rather than token counts
    (maxRL advantages are typically ≫ 1, so the same target fraction yields
    more anchor rows than the token basis).

    Both are computed from the index alone (no ``weights.bin`` reads), so this
    is cheap enough to re-evaluate at every streaming rescan: per episode,
    mass = ``|weight| * (total_len - prompt_len)`` (zero when ``weight == 0``).
    This is exact under the format invariant that every supervised position in
    an episode carries the scalar ``weight`` recorded in its index row — true
    of every current producer (RL rollouts, raw windows, assistant-masked SFT).
    For the supervised basis on full rounds this equals the manifest's
    ``n_weighted_tokens`` path in :func:`mean_response_tokens`.
    """
    if basis not in MIX_BASES:
        raise ValueError(f"mix basis must be one of {MIX_BASES}, got {basis!r}")
    if not refs:
        raise ValueError("empty ref pool")
    refs_by_round: dict[int, list[int]] = {}
    for round_index, episode_index in refs:
        refs_by_round.setdefault(round_index, []).append(episode_index)
    total = 0.0
    for round_index, episode_indices in refs_by_round.items():
        index = rounds[round_index].index
        rows = index[np.asarray(episode_indices, dtype=np.int64)]
        # APPROXIMATION, and the reason it is acceptable here: this is the
        # index-based fast path, so "supervised" is the whole response span
        # rather than a count of individually non-zero token weights.  Under the
        # scalar-episode-weight invariant these coincide -- every response token
        # of a non-zero-weight episode carries that episode's weight -- which is
        # what the maxRL producer emits.  If a producer ever masks tokens WITHIN
        # a response (partial-credit shaping, intra-response filtering), this
        # OVERCOUNTS that episode's mass and the realized anchor dose comes out
        # below target.  `mean_response_tokens()` reads weights.bin and is exact;
        # it is what the supervised_tokens basis uses.  Keep it that way unless
        # you have re-verified this invariant for the producer in question.
        supervised = (rows["total_len"] - rows["prompt_len"]).astype(np.float64)
        if basis == "weight":
            total += float(np.sum(np.abs(rows["weight"]) * supervised))
        elif basis == "response_tokens":
            total += float(np.sum(supervised))
        else:
            total += float(np.sum(supervised[rows["weight"] != 0]))
    return total / len(refs)


def normalize_weights(weight_by_episode: np.ndarray, response_len_by_episode: np.ndarray) -> float:
    """Scale factor that makes the token-weighted mean equal 1 over response tokens.

    Producers multiply every episode weight by this before writing, so the
    trainer's self-normalized loss sees comparable token accounting across
    rounds regardless of the raw advantage scale.
    """
    total_w = float(np.dot(weight_by_episode, response_len_by_episode))
    total_tokens = float(response_len_by_episode.sum())
    if total_w <= 0:
        raise ValueError("cannot normalize: total weight is zero")
    return total_tokens / total_w
