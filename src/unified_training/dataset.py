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

"""Torch datasets over weighted episode rounds: the trainer's data buffer.

Each sample is ONE padded episode (no cross-episode packing), so the model's
default causal mask is already correct: pads sit after the response and carry
zero weight. A sample is a dict tokens/labels/loss_mask/denom_mask/...;
``loss_mask`` carries the float per-token weights of Eq. (2) and ``denom_mask``
marks the non-prompt tokens counted in its normalizer.

Consumption modes:
  * ``WeightedEpisodeDataset`` -- fixed pool, epoch-shuffled modulo indexing
    (static data: anchors, pretraining rows, held-out slices).
  * ``StreamingEpisodeDataset`` -- GROWING pool: rescans its parent dirs for
    newly finalized generator rounds while training runs (the RL buffer).
  * ``MixedEpisodeDataset`` / ``AdaptiveMixedEpisodeDataset`` /
    ``MultiStreamEpisodeDataset`` -- the RL stream mixed with static data by a
    target token fraction (constant or scheduled), see ``build_unified_datasets``.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from unified_training.episodes import (
    Round,
    is_round_dir,
    load_round,
    rank_balanced_recent_round_indices,
    round_order_key,
    row_fractions,
    validate_no_anchor_replay,
)


def _episode_row(
    rounds: list[Round], ref: tuple[int, int], *, seq_length: int, pad_token_id: int
) -> dict[str, torch.Tensor]:
    """Build one trainer row (tokens/labels/loss_mask/position_ids) from an episode.

    Overlong episodes keep their row (shape stability) but are trained with
    all-zero weight; the producer should cap generation length instead.
    """
    round_i, ep_i = ref
    tokens, weights, _row = rounds[round_i].episode(ep_i)
    s1 = seq_length + 1
    buf = np.full(s1, pad_token_id, dtype=np.int64)
    wbuf = np.zeros(s1, dtype=np.float32)
    dbuf = np.zeros(s1, dtype=np.float32)
    if len(tokens) <= s1:
        buf[: len(tokens)] = tokens
        wbuf[: len(weights)] = weights
        dbuf[int(_row["prompt_len"]) : len(tokens)] = 1.0
    else:
        buf[:] = tokens[:s1]
    return {
        "tokens": torch.from_numpy(buf[:-1].copy()),
        "labels": torch.from_numpy(buf[1:].copy()),
        "loss_mask": torch.from_numpy(wbuf[1:].copy()),
        "denom_mask": torch.from_numpy(dbuf[1:].copy()),
        "position_ids": torch.arange(seq_length, dtype=torch.int64),
        # Preserve the episode-level origin for consumed-row diagnostics.
        "source_policy_version": torch.tensor(int(_row["policy_version"]), dtype=torch.int64),
    }


class WeightedEpisodeDataset(Dataset):
    """Fixed-pool episode dataset: index ``i`` is episode ``perm_{i // n}[i % n]``.

    The dataset has whatever length the trainer asks for (train_iters * GBS) and
    ``consumed_train_samples`` resume just works.
    """

    def __init__(
        self,
        rounds: list[Round],
        episode_refs: list[tuple[int, int]],
        *,
        seq_length: int,
        total_samples: int,
        seed: int = 1234,
        pad_token_id: int = 0,
        excluded_refs: frozenset[tuple[int, int]] = frozenset(),
    ) -> None:
        """``episode_refs`` is the (round_idx, episode_idx) pool this split draws from."""
        if not episode_refs:
            raise ValueError("empty episode pool")
        self._rounds = rounds
        # Filter identities AFTER the original-size shuffle. Shuffling a shorter
        # list would change every sample, including a valid historical prefix.
        self._source_to_retained: np.ndarray | None = None
        if excluded_refs:
            retained = [ref for ref in episode_refs if ref not in excluded_refs]
            if not retained:
                raise ValueError("holdout exclusion leaves an empty anchor pool")
            mapping = np.full(len(episode_refs), -1, dtype=np.int64)
            kept = [i for i, ref in enumerate(episode_refs) if ref not in excluded_refs]
            mapping[kept] = np.arange(len(kept), dtype=np.int64)
            self._source_to_retained = mapping
            episode_refs = retained
        self._refs = episode_refs
        # Online/scheduled mixers historically cycle the safe epoch-0 prefix.
        # The builder sets this after validating a holdout's source identity.
        self.holdout_epoch0_prefix: int | None = None
        self._seq_length = seq_length
        self._total_samples = max(int(total_samples), 1)
        self._seed = seed
        self._pad = pad_token_id
        self._epoch_perm_cache: tuple[int, np.ndarray] | None = None

        n_over = sum(
            1 for r, e in episode_refs if int(rounds[r].index["total_len"][e]) > seq_length + 1
        )
        self._n_overlong = n_over
        if n_over:
            print(f"[unified] WARNING: {n_over}/{len(episode_refs)} episodes exceed seq_length+1; zero-weighted")

    def __len__(self) -> int:
        return self._total_samples

    @property
    def n_episodes(self) -> int:
        return len(self._refs)

    def _perm(self, epoch: int) -> np.ndarray:
        if self._epoch_perm_cache is None or self._epoch_perm_cache[0] != epoch:
            rng = np.random.default_rng((self._seed, epoch))
            if self._source_to_retained is None:
                permutation = rng.permutation(len(self._refs))
            else:
                permutation = self._source_to_retained[rng.permutation(len(self._source_to_retained))]
                permutation = permutation[permutation >= 0]
            self._epoch_perm_cache = (epoch, permutation)
        return self._epoch_perm_cache[1]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        n = len(self._refs)
        epoch, pos = divmod(int(idx), n)
        ref = self._refs[self._perm(epoch)[pos]]
        return _episode_row(self._rounds, ref, seq_length=self._seq_length, pad_token_id=self._pad)


class NoFreshEpisodes(RuntimeError):
    """The strict streaming pool has no complete set of fresh generator ranks."""


class StopRequested(RuntimeError):
    """The stop file appeared while the streaming builder waited for the first rounds."""


class StreamingEpisodeDataset(Dataset):
    """Growing-pool episode dataset — the continuous trainer's data source.

    Watches ``parent_dirs`` and folds newly finalized round dirs into the pool
    every ``rescan_every`` fetches (per dataloader worker; each worker rescans
    independently). Index -> episode uses per-index seeded sampling *with
    replacement* — there are no epochs in a stream, and it keeps every index
    valid under any pool size. Consequences, accepted by design:

      * DP ranks/workers may briefly see different pool sizes; every fetch is
        still valid data, batches never block or mismatch.
      * Restart replay is not sample-exact (the pool a given index sees depends
        on arrival time). Fine for RL data; the anchor stream (via
        ``MixedEpisodeDataset``) and validation stay exactly reproducible.

    ``exclude`` pins down a holdout (by ``(absolute_round_dir, episode_idx)``) so a
    validation set carved from the initial scan is never trained on. Keys use the
    absolute dir path — basenames would collide across parents.

    ``recency_window`` (0 = off) restricts sampling to episodes from the most recent
    W rounds — a freshness dial for the off-policy pool. Multi-generator rounds carry
    ``generator_rank``/``generator_world`` metadata; when W is a world-size multiple,
    the window selects the same number of newest nonempty rounds from every rank.
    This stays rank-balanced across partial cycles and unequal producer speeds.
    Opt-in ``strict_policy_age`` removes the availability floor and all stale-pool
    fallbacks, and checks actual episode versions in [step - max_age, step].
    Missing version or generator topology metadata is ineligible. If any rank
    lacks fresh rows, fetching raises ``NoFreshEpisodes`` until fresh data arrives.
    """

    def __init__(
        self,
        parent_dirs: list[str],
        *,
        seq_length: int,
        total_samples: int,
        seed: int = 1234,
        pad_token_id: int = 0,
        rescan_every: int = 256,
        exclude: frozenset[tuple[str, int]] = frozenset(),
        recency_window: int = 0,
        max_policy_age_steps: int = 0,
        strict_policy_age: bool = False,
        initial_trainer_step: int = 0,
        expected_vocab_size: int = 0,
        without_replacement: bool = False,
    ) -> None:
        # Canonicalize once so exclusion keys built from Round.round_dir.resolve()
        # still match when the supplied parent traverses a symlink.
        self._parents = [str(Path(parent).resolve()) for parent in parent_dirs]
        self._seq_length = seq_length
        self._total_samples = max(int(total_samples), 1)
        self._seed = seed
        self._pad = pad_token_id
        self._rescan_every = max(int(rescan_every), 1)
        self._exclude = exclude
        self._recency_window = max(int(recency_window), 0)
        # Policy-age eligibility (UNIFIED_MAX_POLICY_AGE_STEPS): a round is usable
        # only while the policy it was generated from is at most this many steps
        # behind the trainer; the newest round of every generator rank is always
        # kept so the pool never empties right after a snapshot.  0 = off.
        self._max_policy_age = max(int(max_policy_age_steps), 0)
        self._strict_policy_age = bool(strict_policy_age)
        if self._strict_policy_age and self._max_policy_age <= 0:
            raise ValueError("strict_policy_age requires positive max_policy_age_steps")
        if self._strict_policy_age and int(initial_trainer_step) < 0:
            raise ValueError("initial_trainer_step must be nonnegative")
        self._trainer_step: int | None = (
            int(initial_trainer_step) if self._strict_policy_age else None
        )
        # Finalized indexes are immutable; inspect their distinct known versions once.
        self._round_versions: dict[int, np.ndarray] = {}
        self._age_allowed: frozenset[int] | None = None
        self._expected_vocab = int(expected_vocab_size)
        self._without_replacement = bool(without_replacement)
        self._perm_cache: tuple[tuple[int, int], np.ndarray] | None = None
        self._rounds: list[Round] = []
        self._refs: list[tuple[int, int]] = []
        self._round_first_ref: list[int] = []  # _refs index where each round's refs begin
        self._eligible_refs: list[tuple[int, int]] = []
        self._eligible_revision = 0
        self._seen: set[str] = set()
        self._fetches = 0
        self.rescan()
        if not self._refs and not self._strict_policy_age:
            raise ValueError(f"no finalized rounds under {self._parents} (and none excluded-adjacent)")

    def rescan(self) -> int:
        """Fold any newly finalized rounds into the pool; returns #episodes added."""
        import os

        added = 0
        for parent in self._parents:
            try:
                entries = sorted(os.listdir(parent), key=round_order_key)
            except FileNotFoundError:
                continue
            for name in entries:
                # Generators finalize a hidden staging directory (including DONE)
                # before atomically renaming it to its visible round name.  A staging
                # directory is therefore not a published round even when DONE exists.
                if name.startswith("."):
                    continue
                path = os.path.abspath(os.path.join(parent, name))
                if path in self._seen or not is_round_dir(path):
                    continue
                try:
                    rnd = load_round(path)
                except FileNotFoundError:
                    # The directory/marker can disappear between is_round_dir() and
                    # load_round() (the staging -> published rename is one example).
                    # Retry it on a later rescan, but still fail on a finalized round
                    # whose required payload files are genuinely missing.
                    if not is_round_dir(path):
                        continue
                    raise
                _check_vocab([rnd], self._expected_vocab, "stream-round")  # validate rounds folded in mid-run too
                ri = len(self._rounds)
                self._rounds.append(rnd)
                self._round_first_ref.append(len(self._refs))
                for e in range(len(rnd.index)):
                    if (path, e) not in self._exclude:
                        self._refs.append((ri, e))
                        added += 1
                self._seen.add(path)
        if added:
            self._refresh_eligible_refs()
            print(f"[unified] stream: +{added} episodes ({len(self._refs)} total from {len(self._rounds)} rounds)")
        return added

    def __len__(self) -> int:
        return self._total_samples

    @property
    def n_episodes(self) -> int:
        return len(self._refs)

    @property
    def recency_summary(self) -> str:
        round_indices = sorted({round_index for round_index, _ in self._eligible_refs})
        ranks: set[int] = set()
        worlds: set[int] = set()
        episodes_by_rank: dict[int, int] = {}
        for round_index in round_indices:
            manifest = self._rounds[round_index].manifest
            try:
                rank = int(manifest["generator_rank"])
                world = int(manifest["generator_world"])
            except (KeyError, TypeError, ValueError):
                continue
            if world > 0 and 0 <= rank < world:
                ranks.add(rank)
                worlds.add(world)
        for round_index, _ in self._eligible_refs:
            manifest = self._rounds[round_index].manifest
            try:
                rank = int(manifest["generator_rank"])
            except (KeyError, TypeError, ValueError):
                continue
            episodes_by_rank[rank] = episodes_by_rank.get(rank, 0) + 1
        summary = f"recency_rounds={len(round_indices)}"
        if len(worlds) == 1:
            world = next(iter(worlds))
            summary += f" rank_balanced={len(ranks)}/{world}"
            episode_balanced = (
                len(ranks) == world
                and len(episodes_by_rank) == world
                and len(set(episodes_by_rank.values())) == 1
            )
            summary += f" rank_episode_balanced={int(episode_balanced)}"
        return summary

    def _round_ref_bounds(self, round_index: int) -> tuple[int, int]:
        start = self._round_first_ref[round_index]
        end = (
            self._round_first_ref[round_index + 1]
            if round_index + 1 < len(self._round_first_ref)
            else len(self._refs)
        )
        return start, end

    def _round_refs(self, round_index: int) -> list[tuple[int, int]]:
        start, end = self._round_ref_bounds(round_index)
        return self._refs[start:end]

    def _round_has_refs(self, round_index: int) -> bool:
        start, end = self._round_ref_bounds(round_index)
        return start < end

    def set_trainer_step(self, step: int) -> None:
        """Advance the cutoff; strict mode also expires rows within mixed rounds.

        An unchanged pool retains its shuffle revision and cached permutation.
        """
        if self._strict_policy_age and self._trainer_step == int(step):
            return
        self._trainer_step = int(step)
        if self._max_policy_age <= 0:
            return
        allowed = self._age_allowed_rounds()
        if self._strict_policy_age or allowed != self._age_allowed:
            self._age_allowed = allowed
            self._refresh_eligible_refs()

    def _age_allowed_rounds(self) -> frozenset[int] | None:
        if self._max_policy_age <= 0 or self._trainer_step is None:
            return None
        cutoff = self._trainer_step - self._max_policy_age
        if self._strict_policy_age:
            allowed = set()
            for index in range(len(self._rounds)):
                versions = self._known_round_versions(index)
                pos = int(np.searchsorted(versions, cutoff))
                if pos < len(versions) and versions[pos] <= self._trainer_step:
                    allowed.add(index)
            return frozenset(allowed)
        allowed: set[int] = set()
        newest_by_rank: dict[int, tuple[int, int]] = {}
        for index, rnd in enumerate(self._rounds):
            if not self._round_has_refs(index):
                continue
            version = int(rnd.manifest.get("policy_version", -1))
            if version >= cutoff:
                allowed.add(index)
            rank = int(rnd.manifest.get("generator_rank", 0))
            if rank not in newest_by_rank or (version, index) > newest_by_rank[rank]:
                newest_by_rank[rank] = (version, index)
        for _version, index in newest_by_rank.values():
            allowed.add(index)   # floor: every rank's newest round stays usable
        return frozenset(allowed)

    def _known_round_versions(self, round_index: int) -> np.ndarray:
        """Actual, nonnegative episode versions, independent of round metadata."""
        if round_index not in self._round_versions:
            index = self._rounds[round_index].index
            versions = np.empty(0, dtype=np.int64)
            if "policy_version" in (index.dtype.names or ()):
                values = index["policy_version"]
                if np.issubdtype(values.dtype, np.integer):
                    episode_indices = [episode for _, episode in self._round_refs(round_index)]
                    values = values[episode_indices]
                    versions = np.unique(values[values >= 0])
            self._round_versions[round_index] = versions
        return self._round_versions[round_index]

    def _strict_eligible_refs(self, allowed: frozenset[int]) -> list[tuple[int, int]]:
        # Establish the generator topology from all folded rounds, including stale
        # ones. Otherwise a missing fresh rank could silently shrink the world.
        metadata: dict[int, tuple[int, int]] = {}
        worlds: set[int] = set()
        for round_index, rnd in enumerate(self._rounds):
            try:
                rank = int(rnd.manifest["generator_rank"])
                world = int(rnd.manifest["generator_world"])
            except (KeyError, TypeError, ValueError):
                continue
            if world > 0 and 0 <= rank < world:
                metadata[round_index] = (rank, world)
                worlds.add(world)
        if len(worlds) != 1:
            return []
        world = next(iter(worlds))
        candidates = set(allowed).intersection(metadata)
        if {metadata[index][0] for index in candidates} != set(range(world)):
            return []
        # With recency disabled retain all available balanced rounds. A configured
        # window keeps its existing meaning: W rounds, with equal counts per rank.
        window = self._recency_window or world * len(self._rounds)
        selected = rank_balanced_recent_round_indices(
            self._rounds, window=window, eligible_round_indices=candidates,
        )
        if not selected:
            return []
        assert self._trainer_step is not None
        cutoff = max(0, self._trainer_step - self._max_policy_age)
        refs_by_rank: dict[int, list[tuple[int, int]]] = {}
        for round_index in selected:
            rank = metadata[round_index][0]
            versions = self._rounds[round_index].index["policy_version"]
            refs_by_rank.setdefault(rank, []).extend(
                ref for ref in self._round_refs(round_index)
                if cutoff <= int(versions[ref[1]]) <= self._trainer_step
            )
        if len(refs_by_rank) != world or not all(refs_by_rank.values()):
            return []
        episodes_per_rank = min(map(len, refs_by_rank.values()))
        return [
            ref for rank in sorted(refs_by_rank)
            for ref in refs_by_rank[rank][:episodes_per_rank]
        ]

    def policy_age_stats(self) -> dict:
        """Current cutoff and the number of usable rounds (for logs/tests)."""
        allowed = self._age_allowed
        return {
            "trainer_step": self._trainer_step,
            "max_policy_age_steps": self._max_policy_age,
            "usable_rounds": None if allowed is None else len(allowed),
            "rounds": len(self._rounds),
        }

    def _refresh_eligible_refs(self) -> None:
        """Rebuild the cached recency pool after finalized rounds are folded in."""
        eligible: list[tuple[int, int]] = []
        window = self._recency_window
        allowed = self._age_allowed_rounds() if self._max_policy_age > 0 else None
        self._age_allowed = allowed
        if self._strict_policy_age:
            eligible = self._strict_eligible_refs(allowed or frozenset())
        elif window <= 0:
            if allowed is None:
                eligible = list(self._refs)
            else:
                for round_index in sorted(allowed):
                    eligible.extend(self._round_refs(round_index))
        else:
            # Prefer an exactly rank-balanced window for unified generator shards.
            # Select by per-rank logical round order, not arrival/lexical order.
            selected = rank_balanced_recent_round_indices(
                self._rounds,
                window=window,
                eligible_round_indices={
                    index
                    for index in range(len(self._rounds))
                    if self._round_has_refs(index) and (allowed is None or index in allowed)
                },
            )
            if selected:
                refs_by_rank: dict[int, list[tuple[int, int]]] = {}
                for round_index in selected:
                    rank = int(self._rounds[round_index].manifest["generator_rank"])
                    refs_by_rank.setdefault(rank, []).extend(
                        self._round_refs(round_index)
                    )
                # Equalize sampling mass as well as round counts. End-of-pass
                # shards and holdout exclusions can make round sizes differ.
                episodes_per_rank = min(map(len, refs_by_rank.values()))
                for rank in sorted(refs_by_rank):
                    eligible.extend(refs_by_rank[rank][:episodes_per_rank])

            if not eligible:
                # Generic producer fallback: a contiguous tail of W nonempty rounds.
                selected = [
                    round_index
                    for round_index in range(len(self._rounds) - 1, -1, -1)
                    if self._round_has_refs(round_index)
                ][:window]
                for round_index in reversed(selected):
                    eligible.extend(self._round_refs(round_index))

        # A fully excluded recent tail must never make rng.integers(0) crash.
        next_refs = eligible if self._strict_policy_age else eligible or list(self._refs)
        if not self._strict_policy_age or next_refs != self._eligible_refs:
            self._eligible_revision += 1
            self._perm_cache = None
        self._eligible_refs = next_refs
        # Drop ineligible rounds' payload memmaps: each open round pins two file
        # descriptors, and a run that streams rounds for hours exhausts the
        # process fd limit if every round it ever folded stays open. Payloads
        # remap lazily if a previously future round becomes eligible later.
        keep = {round_index for round_index, _ in self._eligible_refs}
        for round_index, rnd in enumerate(self._rounds):
            if round_index not in keep and rnd.tokens is not None:
                rnd.release()

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        self._fetches += 1
        if self._fetches % self._rescan_every == 0:
            self.rescan()
        refs = self._eligible_refs
        n = len(refs)
        if self._strict_policy_age and not n:
            raise NoFreshEpisodes(
                f"no rank-balanced fresh episodes at trainer step {self._trainer_step} "
                f"within max_policy_age_steps={self._max_policy_age}"
            )
        if self._without_replacement and n > 0:
            # WITHOUT replacement: cycle a per-window shuffle so consecutive indices draw
            # DISTINCT episodes (each once per pass over the eligible window), matching
            # NeMo-RL's exactly-once consumption rather than sampling with replacement
            # (only ~1-e^-r distinct coverage at reuse r; ~46% at r=0.62). Cached per
            # (eligible-pool revision, epoch).
            epoch = int(idx) // n
            key = (self._eligible_revision, epoch)
            if self._perm_cache is None or self._perm_cache[0] != key:
                self._perm_cache = (
                    key,
                    np.random.default_rng(
                        (self._seed, self._eligible_revision, epoch)
                    ).permutation(n),
                )
            ref = refs[int(self._perm_cache[1][int(idx) % n])]
        else:
            rng = np.random.default_rng((self._seed, int(idx)))
            ref = refs[int(rng.integers(n))]
        return _episode_row(self._rounds, ref, seq_length=self._seq_length, pad_token_id=self._pad)


def streaming_primary(dataset) -> StreamingEpisodeDataset | None:
    """Find the primary streaming pool through the supported mixer wrappers."""
    seen: set[int] = set()
    while dataset is not None and id(dataset) not in seen:
        if isinstance(dataset, StreamingEpisodeDataset):
            return dataset
        seen.add(id(dataset))
        if hasattr(dataset, "_primary"):
            dataset = dataset._primary
        else:
            streams = getattr(dataset, "_streams", None)
            dataset = streams[0] if streams else None
    return None


def additive_anchor_count(expected: float, *, seed: int, step: int) -> int:
    """Integer anchor rows for one additive-mode step: floor(expected) plus a
    Bernoulli draw on the fractional part, seeded by (seed, step) so every
    data-parallel rank agrees."""
    import math

    if expected <= 0:
        return 0
    whole = int(math.floor(expected))
    frac = expected - whole
    return whole + int(np.random.default_rng((seed, step)).random() < frac)


class MixedEpisodeDataset(Dataset):
    """Deterministic blend of an RL-episode stream and a corpus "anchor" stream.

    This is what makes the trainer *weighted midtraining* rather than async RL:
    anchor windows (plain training data, uniform weight) and RL episodes (advantage
    weights) flow through one loss. Sample ``idx`` is an anchor draw iff
    ``floor((idx+1)*f) > floor(idx*f)`` — exactly fraction ``f`` of draws in the long
    run, stateless, and stable under ``consumed_train_samples`` resume. Stream-local
    indices (``floor(idx*f)`` anchors / ``idx - floor(idx*f)`` primaries before ``idx``)
    feed each stream's own indexing, so both pools cycle independently. Works over
    both fixed and streaming primaries.
    """

    def __init__(
        self,
        primary: Dataset,
        anchor: Dataset,
        *,
        anchor_fraction: float,
        total_samples: int,
        require_no_anchor_replay: bool = False,
        token_fraction: float | None = None,
    ) -> None:
        if not 0.0 < anchor_fraction < 1.0:
            raise ValueError(f"anchor_fraction must be in (0, 1), got {anchor_fraction}")
        self._primary = primary
        self._anchor = anchor
        self._f = anchor_fraction
        self._tf = None if token_fraction is None else float(token_fraction)
        self._require_no_replay = bool(require_no_anchor_replay)
        self._total_samples = max(int(total_samples), 1)
        anchor_draws = int(self._total_samples * self._f)
        anchor_episodes = getattr(anchor, "n_episodes", None)
        if require_no_anchor_replay:
            if anchor_episodes is None:
                raise ValueError(
                    "require_no_anchor_replay needs an anchor dataset with n_episodes"
                )
            anchor_draws = validate_no_anchor_replay(
                self._total_samples, self._f, int(anchor_episodes)
            )
        self.anchor_draws = anchor_draws
        self.anchor_episodes = anchor_episodes

    @property
    def anchor_fraction(self) -> float:
        """Return the exact row fraction used by the deterministic mixer."""

        return self._f

    def set_trainer_step(self, step: int) -> None:
        f = getattr(self._primary, "set_trainer_step", None)
        if f is not None:
            f(step)

    def __len__(self) -> int:
        return self._total_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        idx = int(idx)
        anchors_before = int(idx * self._f)
        if int((idx + 1) * self._f) > anchors_before:
            row = dict(self._anchor[anchors_before])
            row["sampling_filter"] = torch.tensor(False)
            return row
        row = dict(self._primary[idx - anchors_before])
        row["sampling_filter"] = torch.tensor(True)
        return row

    # ---- additive mode (UNIFIED_ANCHOR_MODE=add): anchor rows on top of the RL batch.
    # The expected count per step is constant here, so the per-step count is the
    # same floor idiom as the row mixer above: step s appends
    # floor((s+1)*n) - floor(s*n) rows, addressed sequentially from floor(s*n) --
    # stateless, resume-safe, exactly n per step in the long run, and the total
    # over the run (floor(steps*n)) is what the no-replay guard checks.
    ADD_BASES = ("tokens", "rows")

    def additive_rows_per_step(self, gbs: int, basis: str = "tokens") -> float:
        if basis not in self.ADD_BASES:
            raise ValueError(f"additive basis must be one of {self.ADD_BASES}, got {basis!r}")
        if basis == "rows":
            if self._tf is None:
                raise ValueError("additive basis 'rows' needs the mixer's token_fraction")
            return self._tf * gbs
        return gbs * self._f / (1.0 - self._f)

    def additive_rows_cap(self, gbs: int, basis: str = "tokens") -> int:
        import math

        return max(1, math.ceil(self.additive_rows_per_step(gbs, basis)))

    def validate_additive(self, gbs: int, basis: str = "tokens") -> int:
        """Total additive anchor draws over the run; rejects anchor replay when required."""
        import math

        steps = math.ceil(self._total_samples / max(int(gbs), 1))
        total = int(math.floor(steps * self.additive_rows_per_step(gbs, basis)))
        if self._require_no_replay and self.anchor_episodes is not None and total > int(self.anchor_episodes):
            raise ValueError(
                "additive anchor mode would replay the fixed anchor: "
                f"draws={total:,} available_episodes={int(self.anchor_episodes):,}"
            )
        return total

    def additive_plan(self, step: int, gbs: int, basis: str = "tokens") -> tuple[int, int]:
        """(rows to append this step, first anchor index for them)."""
        n = self.additive_rows_per_step(gbs, basis)
        base = int(step * n)
        return int((step + 1) * n) - base, base

    def primary_row(self, idx: int) -> dict:
        row = dict(self._primary[idx])
        row["sampling_filter"] = torch.tensor(True)
        return row

    def anchor_row(self, k: int) -> dict:
        row = dict(self._anchor[k])
        row["sampling_filter"] = torch.tensor(False)
        return row


class AdaptiveMixedEpisodeDataset(Dataset):
    """2-way blend whose row fraction re-tracks the primary pool's live statistics.

    The launch-mode floor mixer freezes the anchor row fraction from the rounds
    present at startup, but the quantity that fraction targets — the RL stream's
    mean per-episode mass — is policy-dependent: supervised tokens/episode grew
    several-fold over long math RL runs as pass rates rose, so a frozen fraction
    under-delivers the anchor share exactly when training succeeds. This mixer
    re-derives the row fraction from the primary's CURRENT eligible pool every
    time that pool changes (streaming rescan revision), so the realized share
    tracks the target throughout the run.

    Per-index decisions use a seeded RNG at the current row fraction (the
    :class:`MultiStreamEpisodeDataset` idiom): realized fractions match the
    target statistically, and a restart replays decisions from whatever fraction
    it re-measures — consistent with the streaming primary, which is itself not
    sample-exact across restarts. The derived fraction is CLAMPED (not errored)
    at ``max_row_fraction``, bounding worst-case anchor draws by
    ``max_row_fraction * total_samples``; exact no-replay accounting needs the
    constant floor mixer, so ``require_no_anchor_replay`` pairs only with
    launch-mode mixing.
    """

    def __init__(
        self,
        primary: Dataset,
        anchor: Dataset,
        *,
        token_fraction: float,
        anchor_mass_per_ep: float,
        basis: str = "supervised_tokens",
        max_row_fraction: float = 1.0,
        total_samples: int,
        seed: int = 1234,
        reserved_anchor_tail: int = 0,
    ) -> None:
        from unified_training.episodes import MIX_BASES

        if not 0.0 < token_fraction < 1.0:
            raise ValueError(f"token_fraction must be in (0, 1), got {token_fraction}")
        if reserved_anchor_tail < 0:
            raise ValueError("reserved_anchor_tail must be non-negative")
        if anchor_mass_per_ep <= 0:
            raise ValueError("anchor mass per episode must be positive")
        if basis not in MIX_BASES:
            raise ValueError(f"mix basis must be one of {MIX_BASES}, got {basis!r}")
        if not 0.0 <= max_row_fraction <= 1.0:
            raise ValueError("max_row_fraction must be between 0 and 1")
        for attr in ("_rounds", "_eligible_refs", "_eligible_revision"):
            if not hasattr(primary, attr):
                raise ValueError(
                    "online mix recalculation needs a streaming primary "
                    f"(missing {attr}); use mix_recalc='launch' for fixed pools"
                )
        self._primary = primary
        self._anchor = anchor
        self._tf = float(token_fraction)
        self._anchor_mass = float(anchor_mass_per_ep)
        self._basis = basis
        self._max_row_f = float(max_row_fraction)
        self._total_samples = max(int(total_samples), 1)
        self._seed = seed
        self._reserved_tail = int(reserved_anchor_tail)
        self._anchor_episodes = int(
            getattr(anchor, "n_episodes", 0) or len(anchor)
        )
        if self._reserved_tail >= self._anchor_episodes:
            raise ValueError(
                f"mid-training holdout reserves {self._reserved_tail} of only "
                f"{self._anchor_episodes} anchor episodes; nothing would remain"
            )
        self._cached_revision: int | None = None
        self._row_f = 0.0
        self._rl_mass = 0.0
        # Per-process realized counters (approximate under data-parallel sharding:
        # each rank only sees its own indices).
        self.fetches = 0
        self.anchor_fetches = 0
        self._refresh()

    def _refresh(self) -> None:
        from unified_training.episodes import mean_episode_mass

        revision = self._primary._eligible_revision
        if revision == self._cached_revision:
            return
        rl_mass = mean_episode_mass(
            self._primary._rounds, self._primary._eligible_refs, basis=self._basis
        )
        self._cached_revision = revision
        if rl_mass <= 0:
            # An all-zero-weight window (e.g. every recent group all-wrong) has no
            # RL mass to balance against; keep the previous fraction rather than
            # silently dropping the anchor to zero.
            if self._row_f > 0:
                print(
                    "[unified] adaptive mix: eligible pool has zero "
                    f"{self._basis} mass; keeping row_f={self._row_f:.4f}"
                )
                return
            rl_mass = 1.0  # bootstrap fallback: minimal positive mass
        f = self._tf
        row_f = f * rl_mass / (self._anchor_mass * (1.0 - f) + f * rl_mass)
        clamped = min(row_f, self._max_row_f)
        if self._row_f == 0.0 or abs(clamped - self._row_f) > 0.02 * self._row_f:
            note = f" (clamped from {row_f:.4f})" if clamped < row_f else ""
            print(
                f"[unified] adaptive mix: rl {rl_mass:.1f} vs anchor "
                f"{self._anchor_mass:.1f} {self._basis}/ep -> row_f={clamped:.4f}{note}"
            )
        self._rl_mass = rl_mass
        self._row_f = clamped

    def set_trainer_step(self, step: int) -> None:
        f = getattr(self._primary, "set_trainer_step", None)
        if f is not None:
            f(step)

    @property
    def anchor_fraction(self) -> float:
        """Current derived row fraction (recomputed when the primary pool changes)."""

        self._refresh()
        return self._row_f

    def mix_stats(self) -> dict:
        """Per-process realized counters plus the current derived fraction."""

        return {
            "row_fraction": self._row_f,
            "rl_mass_per_ep": self._rl_mass,
            "fetches": self.fetches,
            "anchor_fetches": self.anchor_fetches,
        }

    def __len__(self) -> int:
        return self._total_samples

    # ---- additive mode (UNIFIED_ANCHOR_MODE=add): anchor rows on top of the RL batch
    ADD_BASES = ("tokens", "rows")

    def additive_rows_per_step(self, gbs: int, basis: str = "tokens") -> float:
        """Expected anchor rows to ADD on top of ``gbs`` RL rows in one step.

        ``tokens``: the anchor carries this mixer's token fraction of the step's
        trained positions -- the same target replace mode realizes by swapping
        rows out -- but with the RL rows held at ``gbs``.  With the current
        (clamped) replace-mode row fraction r that is n = gbs * r / (1 - r).
        ``rows``: the fraction is read as a share of the RL row count,
        n = token_fraction * gbs (a much heavier token dose: one anchor window
        is a full sequence, an RL episode only its response).
        """
        if basis not in self.ADD_BASES:
            raise ValueError(f"additive basis must be one of {self.ADD_BASES}, got {basis!r}")
        if basis == "rows":
            return self._tf * gbs
        r = self.anchor_fraction
        return gbs * r / (1.0 - r)

    def additive_rows_cap(self, gbs: int, basis: str = "tokens") -> int:
        """Upper bound on additive rows per step (the row cap, or the row share)."""
        import math

        if basis == "rows":
            return max(1, math.ceil(self._tf * gbs))
        return max(1, math.ceil(gbs * self._max_row_f / max(1.0 - self._max_row_f, 1e-9)))

    def validate_additive(self, gbs: int, basis: str = "tokens") -> int:
        """Additive draws are bounded by the row cap; replay accounting is not exact here."""
        import math

        return int(math.ceil(self._total_samples / max(int(gbs), 1)) * self.additive_rows_cap(gbs, basis))

    def additive_plan(self, step: int, gbs: int, basis: str = "tokens") -> tuple[int, int]:
        """(rows to append this step, first slot id): the count is a seeded Bernoulli
        round of the current expectation (it changes with the RL pool), the slots
        are step-strided by the cap and mapped through the holdout-safe index."""
        n = additive_anchor_count(self.additive_rows_per_step(gbs, basis), seed=self._seed, step=step)
        return n, step * self.additive_rows_cap(gbs, basis)

    def primary_row(self, idx: int) -> dict:
        """An RL row by training index (no anchor swap), tagged for the sampling filter."""
        self.fetches += 1
        row = dict(self._primary[idx])
        row["sampling_filter"] = torch.tensor(True)
        return row

    def anchor_row(self, slot: int) -> dict:
        """The anchor row for an additive slot id (holdout tail excluded)."""
        self.fetches += 1
        self.anchor_fetches += 1
        row = dict(self._anchor[self._anchor_index(slot)])
        row["sampling_filter"] = torch.tensor(False)
        return row

    def _anchor_index(self, idx: int) -> int:
        """Map a training index onto an anchor row the holdout does not own.

        The anchor is addressed by the GLOBAL training index, which wraps the
        anchor permutation many times over a long run (4.1M rows against a 618K
        anchor wraps ~6.6x).  The mid-training holdout proof, by contrast,
        reserves a TAIL of that permutation and bounds training to the prefix
        before it -- a bound that only holds while consumption is sequential.
        Under this mixer it is not, so without the clamp below every reserved
        window is eventually a draw candidate and the "held-out" LM loss is
        measured on rows the model trained on, with the contamination growing
        with the anchor dose: worst exactly where the retention metric matters.

        Clamping the addressable range to the unreserved prefix restores
        disjointness by construction rather than by argument, and keeps the
        decision a pure function of ``idx`` so restarts still replay it.

        CONSEQUENCE, stated because it is a real behaviour change: the anchor
        indexes as ``perm_{i // n}[i % n]`` with a per-epoch permutation, and
        the holdout only reserves the tail of the EPOCH-0 permutation (its
        manifest records ``split_method=unused_epoch0_permutation_tail``).  The
        clamp therefore keeps every draw inside epoch 0, so an anchor that is
        replayed repeats in a fixed order instead of reshuffling per epoch.
        That is immaterial for a dose drawing under one pass -- the case this
        was built for -- but a heavily replayed anchor loses per-epoch
        reshuffling, which is the honest cost of a constructive guarantee.
        Runs that do not declare a holdout keep the previous behaviour.
        """

        guarded_prefix = getattr(self._anchor, "holdout_epoch0_prefix", None)
        if guarded_prefix is not None:
            return idx % guarded_prefix
        if self._reserved_tail <= 0:
            return idx
        # n_episodes, NOT len(): __len__ is the training sample count, while the
        # permutation period -- the thing the holdout tail is defined against --
        # is the episode count.  Clamping against the wrong one would silently
        # fail to exclude the reserved windows.
        usable = self._anchor_episodes - self._reserved_tail
        if usable <= 0:
            raise ValueError(
                f"anchor of {self._anchor_episodes} episodes cannot reserve "
                f"{self._reserved_tail} for the mid-training holdout"
            )
        return idx % usable

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        idx = int(idx)
        self.fetches += 1
        self._refresh()
        if np.random.default_rng((self._seed, idx)).random() < self._row_f:
            self.anchor_fetches += 1
            row = dict(self._anchor[self._anchor_index(idx)])
            row["sampling_filter"] = torch.tensor(False)
            return row
        row = dict(self._primary[idx])
        row["sampling_filter"] = torch.tensor(True)
        return row


def match_rl_anneal_horizon(rl_steps: int, launch_row_fraction: float,
                            end_row_fraction: float = 0.0, end_progress: float = 1.0) -> int:
    """Step at which a piecewise-linear anchor row-fraction schedule consumes
    exactly ``rl_steps`` x GBS RL rows (match_rl + schedules).

    The row fraction runs linearly from ``launch_row_fraction`` (ra) to
    ``end_row_fraction`` (rb) over the first ``end_progress`` (q) of the horizon
    T and holds rb after.  Its mean over [0, T] is q (ra + rb)/2 + (1 - q) rb,
    so the RL rows after T steps are GBS T (1 - mean) and T = rl_steps / (1 - mean).
    The anneal-to-zero case is rb = 0, q = 1: T = rl_steps / (1 - ra/2).  The
    token->row map is not exactly linear in the token fraction, but at the doses
    used (<= 2%) the error is far below one step; the trainer's RL-row counter,
    not this projection, decides the stop.
    """
    if rl_steps <= 0:
        return 0
    for name, v in (("launch_row_fraction", launch_row_fraction), ("end_row_fraction", end_row_fraction)):
        if not 0.0 <= v < 1.0:
            raise ValueError(f"{name} must be in [0, 1), got {v}")
    if not 0.0 < end_progress <= 1.0:
        raise ValueError(f"end_progress must be in (0, 1], got {end_progress}")
    mean = end_progress * (launch_row_fraction + end_row_fraction) / 2.0 + (1.0 - end_progress) * end_row_fraction
    return int(math.ceil(rl_steps / (1.0 - mean)))


def as_schedule(spec):
    """Coerce a fraction spec into a callable ``progress(0..1) -> fraction``.

    ``spec`` is a constant float, or ``{"start": a, "end": b}`` for a linear
    anneal in training progress (enables midtrain(1)<->RL(0) continuum schedules).
    An optional ``"end_progress": q`` (0 < q <= 1, default 1) reaches ``b`` at
    progress ``q`` and holds it for the rest of the run, so a dose can hit zero
    mid-run instead of only at the final step.
    """
    if isinstance(spec, dict):
        a, b = float(spec["start"]), float(spec["end"])
        q = float(spec.get("end_progress", 1.0))
        if not 0.0 < q <= 1.0:
            raise ValueError(f"end_progress must be in (0, 1], got {q}")
        return lambda p: a + (b - a) * min(p / q, 1.0)
    c = float(spec)
    return lambda p: c


class MultiStreamEpisodeDataset(Dataset):
    """N-way (>=2) blend of token streams at per-stream TOKEN fractions, with schedules.

    Generalizes :class:`MixedEpisodeDataset`. ``streams[0]`` is the primary (RL)
    stream; its token fraction is the remainder ``1 - sum(others)``. For each global
    ``idx`` the stream is chosen by a deterministic per-index RNG with probabilities
    equal to the ROW fractions implied by the current TOKEN fractions and the measured
    per-stream tokens/episode (``row_fractions``) — resume-exact (pure function of
    idx) and schedule-aware (fractions evaluated at ``progress = idx/total``). Unlike
    the exact 2-way floor rule, fractions are hit statistically; at 1e5+ draws the
    realized share is within ~1e-3 of target. The chosen substream gets the global
    ``idx`` (it decorrelates via its own seed), so each pool cycles independently.
    """

    def __init__(
        self,
        streams: list[Dataset],
        *,
        token_fraction_fns: list,  # one callable progress->token_fraction per stream
        tokens_per_ep: list[float],
        total_samples: int,
        seed: int = 1234,
        reserved_aux_tails: list[int] | None = None,
    ) -> None:
        if len(streams) < 2:
            raise ValueError("MultiStreamEpisodeDataset needs >=2 streams")
        if not (len(streams) == len(token_fraction_fns) == len(tokens_per_ep)):
            raise ValueError("streams / token_fraction_fns / tokens_per_ep length mismatch")
        self._streams = streams
        self._tf_fns = token_fraction_fns
        self._tpe = tokens_per_ep
        self._total_samples = max(int(total_samples), 1)
        self._seed = seed
        # Reserved tail per stream, mirroring AdaptiveMixedEpisodeDataset: this
        # mixer also addresses a stream by the GLOBAL training index, which wraps
        # that stream's epoch-0 permutation many times over a long run, so
        # without a clamp the windows a mid-training holdout reserved are drawn
        # like any others and its "held-out" loss is measured on trained rows.
        # Entry 0 is the RL primary and never has a holdout.
        self._validate_schedules(token_fraction_fns)
        self._reserved = list(reserved_aux_tails or [0] * len(streams))
        if len(self._reserved) != len(streams):
            raise ValueError("reserved_aux_tails length must match streams")
        if any(value < 0 for value in self._reserved):
            raise ValueError("reserved_aux_tails must be non-negative")

    @staticmethod
    def _validate_schedules(token_fraction_fns: list) -> None:
        """Reject a schedule that is impossible, rather than saturating it.

        ``__getitem__`` clamps to keep probabilities well formed, which would
        otherwise turn a typo -- a negative share, or aux shares summing past
        1.0 so the primary stream vanishes -- into a silently different
        experiment that still runs to completion and reports a dose it never
        used.  Sampled across the whole progress range because a schedule can
        be valid at both endpoints and invalid between them.
        """

        for step_index in range(101):
            progress = step_index / 100.0
            values = [float(fn(progress)) for fn in token_fraction_fns[1:]]
            for stream_index, value in enumerate(values, start=1):
                if value < 0.0:
                    raise ValueError(
                        f"stream {stream_index} token fraction is negative at "
                        f"progress {progress:.2f}: {value}"
                    )
            total = sum(values)
            if total > 1.0 + 1e-9:
                raise ValueError(
                    f"auxiliary token fractions sum to {total:.6f} at progress "
                    f"{progress:.2f}, leaving no mass for the primary stream"
                )

    def __len__(self) -> int:
        return self._total_samples

    def _stream_index(self, stream: int, idx: int) -> int:
        guarded_prefix = getattr(self._streams[stream], "holdout_epoch0_prefix", None)
        if guarded_prefix is not None:
            return idx % guarded_prefix
        reserved = self._reserved[stream]
        if reserved <= 0:
            return idx
        episodes = int(
            getattr(self._streams[stream], "n_episodes", 0) or len(self._streams[stream])
        )
        usable = episodes - reserved
        if usable <= 0:
            raise ValueError(
                f"stream {stream} has {episodes} episodes and cannot reserve {reserved}"
            )
        return idx % usable

    def set_trainer_step(self, step: int) -> None:
        f = getattr(self._streams[0], "set_trainer_step", None)   # the primary RL stream
        if f is not None:
            f(step)

    # match_rl + anneal: set by the builder when the schedule's
    # horizon was solved so the anchor share reaches zero where the RL-row
    # count matches the staged-RL budget.  ``expected_rl_rows`` is the closed
    # form the trainer uses only when no persisted counter exists at a resume.
    match_rl_horizon: int | None = None
    launch_row_fraction: float | None = None
    end_row_fraction: float = 0.0
    end_progress: float = 1.0
    rows_per_step: int | None = None

    def expected_rl_rows(self, step: int) -> int:
        """Closed-form RL rows after ``step`` steps under the schedule's row fraction
        r(u): linear from ra to rb over [0, qT], rb on [qT, T], and 0 (pure RL) past T."""
        if self.match_rl_horizon is None or self.launch_row_fraction is None or not self.rows_per_step:
            raise RuntimeError("expected_rl_rows needs a match_rl schedule horizon (builder sets it)")
        T, ra, rb, q, g = self.match_rl_horizon, self.launch_row_fraction, self.end_row_fraction, self.end_progress, self.rows_per_step
        s = min(int(step), T); knee = q * T
        if s <= knee:
            anchor = ra * s + (rb - ra) * s * s / (2.0 * knee)
        else:
            anchor = ra * knee + (rb - ra) * knee / 2.0 + rb * (s - knee)
        rows = g * (s - anchor)
        if step > T:
            rows += g * (step - T)
        return int(rows)

    # ---- additive mode (UNIFIED_ANCHOR_MODE=add) with a scheduled anchor.
    # Every step keeps all GBS RL rows (the primary stream, addressed sequentially)
    # and appends anchor rows whose expected count follows the schedule: at step s,
    # progress u = s / steps and the anchor share x(u) gives, on basis "tokens", the
    # replace-mode row fraction r(u) and gbs * r / (1 - r) rows -- the rule
    # MixedEpisodeDataset uses at a constant share -- or x(u) * gbs on basis "rows".
    # Step s appends floor(C(s+1)) - floor(C(s)) rows for the cumulative expectation
    # C, addressed from floor(C(s)): stateless, resume-exact, and the total over the
    # run is floor(C(steps)). Anchor rows go through the holdout-safe stream index.
    # One anchor stream only.
    ADD_BASES = ("tokens", "rows")

    def _additive_rate(self, u: float, gbs: int, basis: str) -> float:
        x = max(0.0, float(self._tf_fns[1](min(max(u, 0.0), 1.0))))
        if basis == "rows":
            return x * gbs
        if x <= 0.0:
            return 0.0
        r = row_fractions([max(0.0, 1.0 - x), x], self._tpe)[1]
        return gbs * r / (1.0 - r)

    def _additive_curve(self, gbs: int, basis: str) -> tuple[np.ndarray, int, float]:
        if basis not in self.ADD_BASES:
            raise ValueError(f"additive basis must be one of {self.ADD_BASES}, got {basis!r}")
        if len(self._streams) != 2:
            raise ValueError("additive anchor mode supports exactly one scheduled anchor stream")
        key = (int(gbs), basis)
        cached = getattr(self, "_additive_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        steps = max(1, math.ceil(self._total_samples / max(int(gbs), 1)))
        rates = np.array([self._additive_rate(s / steps, gbs, basis) for s in range(steps)])
        cumulative = np.concatenate([[0.0], np.cumsum(rates)])
        curve = (cumulative, steps, self._additive_rate(1.0, gbs, basis))
        self._additive_cache = (key, curve)
        return curve

    def _additive_cumulative(self, step: int, gbs: int, basis: str) -> float:
        cumulative, steps, tail = self._additive_curve(gbs, basis)
        if step <= steps:
            return float(cumulative[step])
        return float(cumulative[-1]) + tail * (step - steps)

    def additive_rows_per_step(self, gbs: int, basis: str = "tokens") -> float:
        """Mean anchor rows per step over the run (the schedule sets the per-step count)."""
        cumulative, steps, _ = self._additive_curve(gbs, basis)
        return float(cumulative[-1]) / steps

    def additive_rows_cap(self, gbs: int, basis: str = "tokens") -> int:
        cumulative, _, tail = self._additive_curve(gbs, basis)
        peak = max(float(np.diff(cumulative).max(initial=0.0)), tail)
        return max(1, math.ceil(peak) + 1)

    def validate_additive(self, gbs: int, basis: str = "tokens") -> int:
        """Total additive anchor draws over the run."""
        cumulative, _, _ = self._additive_curve(gbs, basis)
        return int(math.floor(float(cumulative[-1]) + 1e-9))

    def additive_plan(self, step: int, gbs: int, basis: str = "tokens") -> tuple[int, int]:
        """(rows to append this step, first anchor index for them)."""
        step = int(step)
        # 1e-9 keeps cumsum round-off from moving a row across an integer boundary
        lo = math.floor(self._additive_cumulative(step, gbs, basis) + 1e-9)
        hi = math.floor(self._additive_cumulative(step + 1, gbs, basis) + 1e-9)
        return hi - lo, lo

    def primary_row(self, idx: int) -> dict:
        row = dict(self._streams[0][self._stream_index(0, int(idx))])
        row["sampling_filter"] = torch.tensor(True)
        return row

    def anchor_row(self, k: int) -> dict:
        row = dict(self._streams[1][self._stream_index(1, int(k))])
        row["sampling_filter"] = torch.tensor(False)
        return row

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        idx = int(idx)
        p = min(idx / self._total_samples, 1.0)
        # token fractions: entries 1.. are explicit; entry 0 (primary) is the remainder.
        # These clamps exist so arithmetic noise cannot produce a negative
        # probability, NOT to paper over a bad schedule: a spec that asks for a
        # negative share, or for aux shares summing past 1, is a configuration
        # error and is rejected at construction (see _validate_schedules) rather
        # than silently rewritten into something the caller did not ask for.
        others = [max(0.0, fn(p)) for fn in self._tf_fns[1:]]
        primary = max(0.0, 1.0 - sum(others))
        tf = [primary] + others
        rf = row_fractions(tf, self._tpe)
        s = int(np.random.default_rng((self._seed, idx)).choice(len(self._streams), p=rf))
        row = dict(self._streams[s][self._stream_index(s, idx)])
        row["sampling_filter"] = torch.tensor(s == 0)
        return row


def split_episode_refs(
    rounds: list[Round], *, valid_holdout: int, seed: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Deterministically split the pooled episodes into (train, valid) ref lists.

    The holdout is taken per shuffled pool, capped to half the pool so a tiny
    round can never starve training.
    """
    refs = [(r, e) for r, rnd in enumerate(rounds) for e in range(len(rnd.index))]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(refs))
    n_valid = min(max(valid_holdout, 0), len(refs) // 2)
    valid = [refs[i] for i in order[:n_valid]]
    train = [refs[i] for i in order[n_valid:]]
    return train, valid


def _valid_ref_manifest_payload(
    rounds: list[Round],
    valid_refs: list[tuple[int, int]],
    *,
    valid_holdout: int,
    seed: int,
) -> dict:
    """Return stable validation identities for an exact streaming restart.

    Dataset-local round indices are not durable because the streaming pool grows.
    Round basenames are stable for the unified generator and are rejected when
    ambiguous, so a persisted holdout can be mapped back into any later scan.
    """
    names = [round_.round_dir.name for round_ in rounds]
    if len(names) != len(set(names)):
        raise ValueError("validation holdout persistence requires unique round basenames")
    return {
        "schema_version": 1,
        "split_seed": int(seed),
        "valid_holdout": len(valid_refs),
        "requested_valid_holdout": int(valid_holdout),
        "episode_count": int(sum(len(round_.index) for round_ in rounds)),
        "initial_rounds": [
            {"round": round_.round_dir.name, "episodes": int(len(round_.index))}
            for round_ in rounds
        ],
        "ordered_valid_refs": [
            {"round": rounds[round_index].round_dir.name, "episode_idx": int(episode_index)}
            for round_index, episode_index in valid_refs
        ],
    }


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _load_valid_refs_manifest(
    rounds: list[Round],
    path: Path,
    *,
    valid_holdout: int,
    seed: int,
) -> list[tuple[int, int]]:
    """Load and validate a stable ordered validation holdout manifest."""
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"invalid validation holdout manifest schema: {path}")
    if payload.get("split_seed") != int(seed):
        raise ValueError(
            f"validation holdout seed changed: manifest={payload.get('split_seed')} current={seed}"
        )
    manifest_requested = payload.get(
        "requested_valid_holdout", payload.get("valid_holdout")
    )
    if manifest_requested != int(valid_holdout):
        raise ValueError(
            "validation holdout size changed: "
            f"manifest={manifest_requested} current={valid_holdout}"
        )

    episode_count = payload.get("episode_count")
    if not isinstance(episode_count, int) or episode_count < 0:
        raise ValueError(f"invalid validation episode count in {path}")
    expected_ref_count = min(max(int(valid_holdout), 0), episode_count // 2)
    if payload.get("valid_holdout") != expected_ref_count:
        raise ValueError(
            "validation manifest effective holdout is inconsistent: "
            f"manifest={payload.get('valid_holdout')} expected={expected_ref_count}"
        )

    by_name: dict[str, tuple[int, Round]] = {}
    for round_index, round_ in enumerate(rounds):
        name = round_.round_dir.name
        if name in by_name:
            raise ValueError(f"ambiguous round basename {name!r} while loading {path}")
        by_name[name] = (round_index, round_)

    raw_refs = payload.get("ordered_valid_refs")
    if not isinstance(raw_refs, list) or len(raw_refs) != expected_ref_count:
        raise ValueError(
            f"validation manifest must contain exactly {expected_ref_count} ordered refs: {path}"
        )
    valid_refs: list[tuple[int, int]] = []
    seen: set[tuple[str, int]] = set()
    for raw in raw_refs:
        if not isinstance(raw, dict) or not isinstance(raw.get("round"), str):
            raise ValueError(f"invalid validation ref in {path}: {raw!r}")
        name = raw["round"]
        try:
            episode_index = int(raw["episode_idx"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid validation episode index in {path}: {raw!r}") from exc
        if name not in by_name:
            raise ValueError(f"validation round {name!r} is missing from the current pool")
        round_index, round_ = by_name[name]
        if episode_index < 0 or episode_index >= len(round_.index):
            raise ValueError(
                f"validation ref {name}:{episode_index} exceeds {len(round_.index)} episodes"
            )
        identity = (name, episode_index)
        if identity in seen:
            raise ValueError(f"duplicate validation ref in {path}: {name}:{episode_index}")
        seen.add(identity)
        valid_refs.append((round_index, episode_index))
    return valid_refs


def _check_vocab(rounds: list[Round], expected_vocab_size: int, label: str) -> None:
    """Guard against shards tokenized with the wrong tokenizer training garbage silently."""
    if expected_vocab_size <= 0:
        return
    for rnd in rounds:
        mx = int(rnd.manifest.get("max_token_id", -1))
        if mx < 0:  # legacy shard without the stamp: scan the memmap once
            was_loaded = rnd.tokens is not None
            rnd.ensure_loaded()
            mx = int(np.asarray(rnd.tokens).max()) if rnd.tokens.size else -1
            if not was_loaded:  # keep the fold path fd-neutral
                rnd.release()
        if mx >= expected_vocab_size:
            raise ValueError(
                f"{label} round {rnd.round_dir} has token id {mx} >= expected vocab "
                f"{expected_vocab_size} — wrong tokenizer? Refusing to train."
            )


def _anchor_holdout_exclusions(rounds, refs, holdouts, *, seed):
    """Bind holdout source IDs to this stream; retain valid old prefix order."""
    import hashlib

    excluded = set()
    matched = []
    by_path = {}
    protected_paths = {h["source_anchor"] for h in holdouts}
    for i, rnd in enumerate(rounds):
        path = str(rnd.round_dir.resolve())
        if path in by_path and path in protected_paths:
            raise ValueError(f"duplicate holdout source in anchor stream: {path}")
        by_path[path] = (i, rnd)
    for holdout in holdouts:
        match = by_path.get(holdout["source_anchor"])
        if match is None:
            continue  # E.g. an SFT anchor evaluated on a different corpus.
        ri, rnd = match
        digest = hashlib.sha256((rnd.round_dir / "manifest.json").read_bytes()).hexdigest()
        if digest != holdout["source_manifest_sha256"] or len(rnd.index) != holdout["source_n_episodes"]:
            raise ValueError(f"holdout source identity changed: {rnd.round_dir} ({holdout['manifest_path']})")
        ids = holdout["source_episode_indices"]
        if ids is None:
            if len(rounds) != 1 or seed != holdout["permutation_seed"]:
                raise ValueError(
                    "legacy positional holdout reservation requires a single source and "
                    f"matching anchor seed: stream={seed}, holdout={holdout['permutation_seed']}; "
                    "provide source_episode_indices for seed-independent exclusion"
                )
            order = np.random.default_rng((seed, 0)).permutation(len(refs))
            excluded.update(refs[int(i)] for i in order[holdout["permutation_tail_start"]:])
        else:
            excluded.update((ri, int(i)) for i in ids)
        matched.append(holdout)
    if not matched:
        return frozenset(), None
    prefix = len(refs) - len(excluded)
    if len(rounds) == 1 and all(h["permutation_seed"] == seed for h in matched):
        # Keep the old conservative prefix/cycle, including any unused tail
        # beyond the explicitly held-out windows. Verify the positional claim
        # before promising sample-exact compatibility.
        prefix = min(h["permutation_tail_start"] for h in matched)
        order = np.random.default_rng((seed, 0)).permutation(len(refs))
        if any(refs[int(i)] in excluded for i in order[:prefix]):
            raise ValueError("holdout source IDs contradict the claimed safe permutation prefix")
    if prefix <= 0:
        raise ValueError("holdout exclusion leaves no usable anchor prefix")
    return frozenset(excluded), prefix


def build_unified_datasets(
    *,
    episodes_dirs: list[str],
    train_samples: int,
    valid_samples: int,
    sequence_length: int = 1024,
    valid_holdout: int = 128,
    split_seed: int = 1234,
    pad_token_id: int = 0,
    anchor_dirs: list[str] | None = None,
    anchor_fraction: float = 0.0,
    anchor_token_fraction: float = 0.0,
    max_anchor_row_fraction: float = 1.0,
    mix_basis: str = "supervised_tokens",
    mix_recalc: str = "online",
    require_no_anchor_replay: bool = False,
    reserved_anchor_tail: int = 0,
    midtrain_holdout_dirs: list[str] | None = None,
    streams: list[dict] | None = None,
    streaming: bool = False,
    rescan_every: int = 256,
    recency_window: int = 0,
    max_policy_age_steps: int = 0,
    strict_policy_age: bool = False,
    initial_trainer_step: int = 0,
    without_replacement: bool = False,
    stream_wait_minutes: int = 60,
    stop_file: str | Path | None = None,
    expected_vocab_size: int = 0,
    valid_refs_manifest: str | Path | None = None,
    persist_valid_refs: bool = False,
    mix_state_manifest: str | Path | None = None,
    persist_mix_state: bool = False,
    match_rl_steps: int = 0,
    match_rl_anneal: bool = False,
    global_batch_size: int = 0,
):
    """Unified dataset builder (called by hf_trainer.py). Semantics: fixed or streaming RL pool,
    validation holdout from the initial scan only (streaming: excluded forever).

    Auxiliary data is blended in by TOKEN fraction (row fractions derived from
    measured episode lengths). Two ways to specify it:
      * ``streams`` (general): list of ``{"dirs": [...], "token_fraction": x}`` where
        ``x`` is a float or ``{"start","end"[,"end_progress"]}`` schedule. The RL
        stream takes the remainder ``1 - sum(others)``. >1 stream or any
        schedule -> N-way mixer.
      * ``anchor_dirs`` + ``anchor_token_fraction``/``anchor_fraction`` (shorthand for
        a single constant stream). The single-constant case uses the exact 2-way
        floor mixer (validated); everything else uses the per-index RNG N-way mixer.

    ``mix_basis`` selects what an anchor fraction is a share of:
    ``supervised_tokens`` (positions with weight > 0 — data exposure, default),
    ``response_tokens`` (every RL response token, right or wrong — the share is
    then independent of the pass rate) or
    ``weight`` (sum of loss coefficients — gradient mass; maxRL advantages are
    typically ≫ 1, so the same target yields more anchor rows). ``mix_recalc``
    selects when the RL stream's per-episode mass is measured: ``online``
    (the default: re-measured whenever the streaming pool changes, so the
    realized share tracks the target as the policy improves; requires streaming,
    a single constant anchor, and is incompatible with
    ``require_no_anchor_replay``) or ``launch`` (once, from the rounds present
    at startup — the historical behavior, which under-delivers the anchor share
    as pass rates rise; required for schedules, N-way streams, fixed pools, and
    exact no-replay accounting).

    ``midtrain_holdout_dirs`` binds committed holdout source episode IDs to
    matching auxiliary directories, excluding them for every shuffle seed.
    Legacy manifests without IDs require their original permutation seed.
    Shuffle order and full-source mass calibration preserve historical valid
    prefixes. ``reserved_anchor_tail`` is the legacy caller-supplied fallback
    when no holdout manifests are provided.

    ``expected_vocab_size`` (>0) rejects any shard whose max token id is out of range
    (wrong-tokenizer guard). ``recency_window`` biases the streaming RL pool toward
    its most recent W rounds. Returns (train_ds, valid_ds, banner_string).
    """
    import time

    from unified_training.episodes import (
        MIX_BASES,
        load_rounds,
        mean_episode_mass,
        mean_response_tokens,
    )
    from unified_training.holdout import load_anchor_holdouts

    anchor_holdouts = load_anchor_holdouts(midtrain_holdout_dirs or [])

    if strict_policy_age and (not streaming or int(max_policy_age_steps) <= 0):
        raise ValueError("strict_policy_age requires streaming and positive max_policy_age_steps")
    if not episodes_dirs:
        raise ValueError("episodes_dirs is empty — pass at least one round dir")
    if not 0.0 <= max_anchor_row_fraction <= 1.0:
        raise ValueError("max_anchor_row_fraction must be between 0 and 1")
    if mix_basis not in MIX_BASES:
        raise ValueError(f"mix_basis must be one of {MIX_BASES}, got {mix_basis!r}")
    if mix_recalc not in ("launch", "online"):
        raise ValueError(f"mix_recalc must be 'launch' or 'online', got {mix_recalc!r}")

    # Read the frozen measurement before scanning/measuring the live pool. Every
    # later rank and resume must use it; reading weights only to discard their
    # measurement adds shared-filesystem traffic without affecting the mix.
    mix_state_path = Path(mix_state_manifest) if mix_state_manifest else None
    frozen_rl_tokens = None
    if mix_state_path is not None and mix_state_path.is_file():
        stored = json.loads(mix_state_path.read_text())
        if stored.get("mix_basis") != mix_basis:
            raise ValueError(
                f"frozen mix state was measured on basis {stored.get('mix_basis')!r}, "
                f"but this run asks for {mix_basis!r}: {mix_state_path}"
            )
        frozen_rl_tokens = float(stored["rl_tokens_per_ep"])

    def _measure(rounds_, refs_):
        # 'supervised_tokens'/launch keeps the historical exact path (manifest fast
        # path + weights.bin fallback); 'weight' uses the index-based mass.
        if mix_basis == "supervised_tokens":
            return mean_response_tokens(rounds_, refs_)
        return mean_episode_mass(rounds_, refs_, basis=mix_basis)

    _eligible: list[tuple[int, int]] = []  # recency-eligible refs (streaming only)
    deadline = time.monotonic() + stream_wait_minutes * 60
    while True:
        try:
            initial_dirs = list(episodes_dirs)
            if streaming:
                # Match StreamingEpisodeDataset's publication boundary during the
                # initial holdout scan too: hidden generator staging dirs may already
                # contain DONE but are not visible/committed rounds yet.
                initial_dirs = []
                for entry in episodes_dirs:
                    path = Path(entry)
                    if is_round_dir(path):
                        if not path.name.startswith("."):
                            initial_dirs.append(str(path))
                    elif path.is_dir():
                        initial_dirs.extend(
                            str(child)
                            for child in sorted(path.iterdir(), key=round_order_key)
                            if child.is_dir()
                            and not child.name.startswith(".")
                            and is_round_dir(child)
                        )
                if not initial_dirs:
                    raise FileNotFoundError(
                        f"no visible finalized rounds under {episodes_dirs}"
                    )
            rounds = load_rounds(initial_dirs)
            break
        except FileNotFoundError:
            if not streaming or time.monotonic() > deadline:
                raise
            if stop_file is not None and Path(stop_file).exists():
                raise StopRequested(f"{stop_file} present while waiting for the first rounds") from None
            print(f"[unified] stream: no finalized rounds under {episodes_dirs} yet; waiting 30s")
            time.sleep(30)
    _check_vocab(rounds, expected_vocab_size, "RL")

    effective_holdout = valid_holdout if valid_samples > 0 else 0
    valid_manifest_path = Path(valid_refs_manifest) if valid_refs_manifest else None
    if (
        effective_holdout > 0
        and valid_manifest_path is not None
        and valid_manifest_path.is_file()
    ):
        valid_refs = _load_valid_refs_manifest(
            rounds,
            valid_manifest_path,
            valid_holdout=effective_holdout,
            seed=split_seed,
        )
        valid_keys = {
            (rounds[round_index].round_dir.name, episode_index)
            for round_index, episode_index in valid_refs
        }
        train_refs = [
            (round_index, episode_index)
            for round_index, round_ in enumerate(rounds)
            for episode_index in range(len(round_.index))
            if (round_.round_dir.name, episode_index) not in valid_keys
        ]
    else:
        train_refs, valid_refs = split_episode_refs(
            rounds, valid_holdout=effective_holdout, seed=split_seed
        )
        if persist_valid_refs and valid_manifest_path is not None:
            _atomic_json(
                valid_manifest_path,
                _valid_ref_manifest_payload(
                    rounds,
                    valid_refs,
                    valid_holdout=effective_holdout,
                    seed=split_seed,
                ),
            )
    common = dict(seq_length=sequence_length, seed=split_seed, pad_token_id=pad_token_id)

    stream_recency_note = ""
    if streaming:
        exclude = frozenset((str(rounds[r].round_dir.resolve()), e) for r, e in valid_refs)
        train_ds = StreamingEpisodeDataset(
            [str(d) for d in episodes_dirs],
            total_samples=train_samples, rescan_every=rescan_every, exclude=exclude,
            recency_window=recency_window, max_policy_age_steps=max_policy_age_steps,
            strict_policy_age=strict_policy_age, initial_trainer_step=initial_trainer_step,
            expected_vocab_size=expected_vocab_size,
            without_replacement=without_replacement, **common,
        )
        stream_recency_note = train_ds.recency_summary
        # Measure the pool the sampler will actually draw from.  With a recency
        # window the stream samples only the newest rounds, and those episodes
        # are systematically LONGER than the initial scan's average because the
        # policy improves; measuring the whole scan therefore understates the RL
        # mass and over-doses the anchor.  Online recalculation already reads
        # `_eligible_refs`; launch mode has to do the same at startup or the two
        # modes disagree about what the same target dose means.
        # Refs must be measured against the rounds list they INDEX.  The
        # streaming dataset rescans `episodes_dirs` in its own constructor, so
        # its `_rounds` can be longer than the caller's earlier scan whenever a
        # generator finalized a round in between -- and then an eligible ref
        # addresses a round the caller never saw.  Pairing the dataset's refs
        # with the caller's `rounds` raised IndexError.  The online path has always paired
        # `_primary._rounds` with `_primary._eligible_refs`; launch mode now
        # does the same.
        _eligible = list(getattr(train_ds, "_eligible_refs", []) or [])
        _eligible_rounds = list(getattr(train_ds, "_rounds", []) or [])
        if frozen_rl_tokens is not None:
            rl_tokens = frozen_rl_tokens
        elif _eligible and _eligible_rounds:
            rl_tokens = _measure(_eligible_rounds, _eligible)
        else:
            rl_tokens = _measure(rounds, train_refs + valid_refs)
    else:
        train_ds = WeightedEpisodeDataset(rounds, train_refs, total_samples=train_samples, **common)
        rl_tokens = frozen_rl_tokens if frozen_rl_tokens is not None else _measure(rounds, train_refs)

    # Normalize aux-stream specs. Each: dirs, and exactly one of:
    #   const_token_f (float) | sched (callable progress->token_f) | row_fraction (legacy).
    if sum(bool(x) for x in (streams, anchor_token_fraction > 0, anchor_fraction > 0)) > 1:
        raise ValueError(
            "set only ONE of streams / anchor_token_fraction / anchor_fraction (ambiguous blend)"
        )
    aux: list[dict] = []
    if streams:
        for sp in streams:
            tfspec = sp["token_fraction"]
            if isinstance(tfspec, dict):
                aux.append({"dirs": sp["dirs"], "sched": as_schedule(tfspec), "spec": dict(tfspec)})
            else:
                aux.append({"dirs": sp["dirs"], "const_token_f": float(tfspec)})
    elif anchor_token_fraction > 0:
        aux.append({"dirs": anchor_dirs or [], "const_token_f": anchor_token_fraction})
    elif anchor_fraction > 0:  # legacy explicit ROW fraction
        aux.append({"dirs": anchor_dirs or [], "row_fraction": anchor_fraction})

    # P0: the launch-mode row fraction is solved from rl_tokens, which is
    # measured off a GROWING pool.  Rank zero builds before the barrier and the
    # other ranks after it, so each rank would otherwise measure a different
    # pool and freeze a different mix; a resume would measure a pool many rounds
    # larger than step 0 and silently change the dose mid-experiment.  Freeze it
    # once, persist it beside the validation-holdout manifest, and have every
    # later reader take the stored value.
    if frozen_rl_tokens is None and mix_state_path is not None and persist_mix_state:
        _atomic_json(
            mix_state_path,
            {
                "schema_version": 1,
                "kind": "unified_launch_mix_state",
                "mix_basis": mix_basis,
                "mix_recalc": mix_recalc,
                "rl_tokens_per_ep": float(rl_tokens),
                "measured_over_episodes": len(_eligible) if streaming else None,
            },
        )

    anchor_note = ""
    basis_note = "" if mix_basis == "supervised_tokens" else f" basis={mix_basis}"
    if mix_recalc == "online" and aux:
        # Online recalculation re-derives the row fraction from the live pool;
        # every shape it cannot adapt is rejected up front rather than ignored.
        if not streaming:
            raise ValueError(
                "mix_recalc='online' requires streaming=True (a growing pool); "
                "set mix_recalc='launch' for fixed pools"
            )
        if len(aux) != 1 or "const_token_f" not in aux[0]:
            raise ValueError(
                "mix_recalc='online' supports exactly one constant-token-fraction "
                "anchor (no schedules, no N-way streams, no legacy row fraction); "
                "set mix_recalc='launch' for those shapes"
            )
        if require_no_anchor_replay:
            raise ValueError(
                "require_no_anchor_replay needs the constant floor mixer's exact draw "
                "count; use mix_recalc='launch' or drop the no-replay requirement"
            )
    if aux:
        if any(not a["dirs"] for a in aux):
            raise ValueError("a blend stream has empty dirs")
        for i, a in enumerate(aux):
            a_rounds = load_rounds(list(a["dirs"]))
            _check_vocab(a_rounds, expected_vocab_size, f"stream{i + 1}")
            a_refs = [(r, e) for r, rnd in enumerate(a_rounds) for e in range(len(rnd.index))]
            excluded, prefix = _anchor_holdout_exclusions(
                a_rounds, a_refs, anchor_holdouts, seed=split_seed + 1 + i,
            )
            a["ds"] = WeightedEpisodeDataset(
                a_rounds, a_refs, total_samples=train_samples,
                seq_length=sequence_length, seed=split_seed + 1 + i, pad_token_id=pad_token_id,
                excluded_refs=excluded,
            )
            a["ds"].holdout_epoch0_prefix = prefix
            # Preserve launch calibration from the full source pool. Exclusion
            # changes eligible IDs, not the recorded historical dose estimator.
            a["tpe"] = _measure(a_rounds, a_refs)
            a["n"] = a["ds"].n_episodes
            a["reserved"] = reserved_anchor_tail if not anchor_holdouts else 0

        # Exact 2-way floor mixer when there's a single CONSTANT (or legacy-row) aux
        # stream — the validated path. Anything else (N>1 or a schedule) -> N-way mixer.
        single_constant = len(aux) == 1 and "sched" not in aux[0]
        if single_constant and mix_recalc == "online":
            a = aux[0]
            train_ds = AdaptiveMixedEpisodeDataset(
                train_ds,
                a["ds"],
                token_fraction=a["const_token_f"],
                anchor_mass_per_ep=a["tpe"],
                basis=mix_basis,
                max_row_fraction=max_anchor_row_fraction,
                total_samples=train_samples,
                seed=split_seed,
                reserved_anchor_tail=a["reserved"],
            )
            anchor_note = (
                f", anchor: {a['n']} windows @ token_f={a['const_token_f']}"
                f" -> row_f={train_ds.anchor_fraction:.4f} (ONLINE recalc{basis_note},"
                f" rl {rl_tokens:.0f} vs anchor {a['tpe']:.0f} mass/ep,"
                f" row cap {max_anchor_row_fraction}"
                + (
                    f", holdout tail {a['reserved']} reserved"
                    if a["reserved"]
                    else ("" if anchor_holdouts else ", no holdout reserved")
                )
                + ")"
            )
        elif single_constant:
            from unified_training.episodes import anchor_row_fraction

            a = aux[0]
            if "row_fraction" in a:
                row_f = a["row_fraction"]
                anchor_note = f", anchor: {a['n']} windows @ row_f={row_f} (legacy row fraction)"
            else:
                row_f = anchor_row_fraction(
                    a["const_token_f"],
                    rl_tokens,
                    a["tpe"],
                    max_row_fraction=max_anchor_row_fraction,
                )
                anchor_note = (f", anchor: {a['n']} windows @ token_f={a['const_token_f']}"
                               f" -> row_f={row_f:.17g} (rl {rl_tokens:.0f} vs anchor"
                               f" {a['tpe']:.0f} tok/ep{basis_note})")
            if row_f > max_anchor_row_fraction:
                raise ValueError(
                    "derived anchor row fraction exceeds the proven holdout bound: "
                    f"{row_f:.6f} > {max_anchor_row_fraction:.6f}"
                )
            # The floor mixer consumes the anchor as a sequential prefix, which
            # is what the holdout proof assumes -- but only the externally set
            # cap keeps that prefix short of the reserved tail.  Safety that
            # depends on a caller passing the right number is not safety, so
            # check it here against the reservation the holdout itself declares.
            if a["reserved"] > 0:
                anchor_episodes = int(
                    getattr(a["ds"], "n_episodes", 0) or len(a["ds"])
                )
                projected_draws = int(train_samples * row_f)
                usable = anchor_episodes - a["reserved"]
                if projected_draws > usable:
                    raise ValueError(
                        f"floor mixer would draw {projected_draws:,} anchor rows, past the "
                        f"{usable:,} unreserved episodes ({anchor_episodes:,} total less "
                        f"{a['reserved']:,} held out): the mid-training holdout would "
                        "be trained on"
                    )
            train_ds = MixedEpisodeDataset(
                train_ds,
                a["ds"],
                anchor_fraction=row_f,
                token_fraction=a.get("const_token_f"),
                total_samples=train_samples,
                require_no_anchor_replay=require_no_anchor_replay,
            )
            if require_no_anchor_replay:
                anchor_note += (
                    f", no-replay={train_ds.anchor_draws}/{train_ds.anchor_episodes} episodes"
                )
        else:
            if require_no_anchor_replay:
                raise ValueError(
                    "require_no_anchor_replay currently supports only one constant anchor stream"
                )
            streams_ds = [train_ds] + [a["ds"] for a in aux]
            tf_fns = [lambda p: 0.0] + [a.get("sched") or as_schedule(a["const_token_f"]) for a in aux]
            tpe = [rl_tokens] + [a["tpe"] for a in aux]
            # match_rl + anneal: one linear anneal to zero whose
            # horizon is solved from the launch row fraction so the dose hits
            # zero where the RL-row count matches match_rl_steps x GBS.  The
            # schedule is re-based to end at progress 1.0 of that horizon; past
            # it the mixer is pure RL until the trainer's counter stops the run.
            anneal_horizon = None
            if match_rl_anneal:
                if match_rl_steps <= 0 or global_batch_size <= 0:
                    raise ValueError("match_rl_anneal needs match_rl_steps > 0 and global_batch_size > 0")
                if len(aux) != 1 or "spec" not in aux[0]:
                    raise ValueError("match_rl_anneal supports exactly one scheduled anchor stream")
                spec = aux[0]["spec"]
                x0, x1 = float(spec["start"]), float(spec["end"])
                q = float(spec.get("end_progress", 1.0))
                if x0 <= 0.0 and x1 <= 0.0:
                    raise ValueError(f"match_rl_anneal needs a schedule with a positive start or end, got {spec}")
                r0 = row_fractions([1.0 - x0, x0], tpe)[1] if x0 > 0 else 0.0
                r1 = row_fractions([1.0 - x1, x1], tpe)[1] if x1 > 0 else 0.0
                anneal_horizon = match_rl_anneal_horizon(match_rl_steps, r0, r1, q)
                tf_fns[1] = as_schedule({"start": x0, "end": x1, "end_progress": q})
                train_samples = anneal_horizon * global_batch_size
            # The floor path raises above this cap and the adaptive path clamps
            # to it; the scheduled path used to ignore it entirely, so a bad
            # schedule or a bad tokens/episode estimate had nothing to stop it.
            # Check the whole schedule, not just its endpoints: a caller may
            # anneal upward.
            if max_anchor_row_fraction < 1.0:
                for aux_index, fn in enumerate(tf_fns[1:], start=1):
                    for step_index in range(101):
                        progress = step_index / 100.0
                        probe = [0.0] * len(tf_fns)
                        probe[aux_index] = max(0.0, fn(progress))
                        probe[0] = max(0.0, 1.0 - sum(probe[1:]))
                        realized = row_fractions(probe, tpe)[aux_index]
                        if realized > max_anchor_row_fraction + 1e-12:
                            raise ValueError(
                                "scheduled anchor row fraction exceeds the cap at "
                                f"progress {progress:.2f}: {realized:.6f} > "
                                f"{max_anchor_row_fraction:.6f}"
                            )
            train_ds = MultiStreamEpisodeDataset(
                streams_ds, token_fraction_fns=tf_fns, tokens_per_ep=tpe,
                total_samples=train_samples, seed=split_seed,
                reserved_aux_tails=[0] + [a["reserved"] for a in aux],
            )
            descr = ", ".join(f"{a['n']}eps@{a['tpe']:.0f}tok" for a in aux)
            anchor_note = f", {len(aux)} aux stream(s) [{descr}], rl {rl_tokens:.0f}tok/ep (N-way token mix)"
            if anneal_horizon is not None:
                train_ds.match_rl_horizon = anneal_horizon
                train_ds.launch_row_fraction = r0
                train_ds.end_row_fraction = r1
                train_ds.end_progress = q
                train_ds.rows_per_step = global_batch_size
                anchor_note += (
                    f", match_rl schedule: row_f {r0:.6f} -> {r1:.6f} by progress {q:g}, horizon step {anneal_horizon}"
                    f" ({match_rl_steps} staged-RL steps of RL rows)"
                )

    if anchor_holdouts and aux:
        protected = sum(a["ds"]._source_to_retained is not None for a in aux)
        anchor_note += f", identity-bound holdout guard: {protected}/{len(aux)} source-matched streams"

    valid_ds = (
        WeightedEpisodeDataset(rounds, valid_refs, total_samples=valid_samples, **common)
        if valid_refs and valid_samples > 0
        else None
    )
    mode = "streaming" if streaming else "fixed"
    rec = (
        f", recency_window={recency_window} {stream_recency_note}"
        + (f", max_policy_age={max_policy_age_steps} steps" if max_policy_age_steps else "")
        if recency_window
        else ""
    )
    banner = (
        f"[unified] dataset[{mode}]: {len(rounds)} round(s), {len(train_refs)} train episodes"
        f" ({len(valid_refs)} valid holdout), seq_length={sequence_length}{rec}{anchor_note}"
    )
    return train_ds, valid_ds, banner


def build_eval_dataset(
    dirs: list[str],
    *,
    sequence_length: int,
    pad_token_id: int,
    total_samples: int,
    seed: int = 1234,
    expected_vocab_size: int = 0,
) -> WeightedEpisodeDataset:
    """Fixed eval dataset over ALL windows/episodes in ``dirs`` — e.g. a held-out
    mid-training slice for a retention metric. No train/val split: every window is
    eligible, so reading from a fixed base each eval scores the same set and gives a
    comparable loss trend. Kept separate from the training pool by the caller (build it
    from a corpus slice the training data never touches).
    """
    from unified_training.episodes import load_rounds

    rounds = load_rounds(list(dirs))
    _check_vocab(rounds, expected_vocab_size, "eval")
    refs = [(r, e) for r, rnd in enumerate(rounds) for e in range(len(rnd.index))]
    return WeightedEpisodeDataset(
        rounds, refs, seq_length=sequence_length, total_samples=total_samples,
        seed=seed, pad_token_id=pad_token_id,
    )
