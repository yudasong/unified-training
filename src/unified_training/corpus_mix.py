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

"""Deterministic helpers for building token-proportional corpus mixtures."""

from __future__ import annotations

import hashlib
import heapq
import math
from collections.abc import Iterator, Mapping


# OLMo-3 Dolma 3 Dolmino 100B mixture from the dataset's per-source table.
OLMO3_COMPONENT_WEIGHTS: dict[str, float] = {
    "common_crawl-high-quality": 0.225,
    "dolmino-math": 0.107,
    "stack_edu-fim": 0.100,
    "cranecode": 0.100,
    "reddit_to_flashcards": 0.059,
    "cranemath": 0.0563,
    "stem-heavy-crawl": 0.050,
    "olmocr_science_pdfs": 0.050,
    "dolmino_1-flan": 0.050,
    "nemotron-synth-qa": 0.050,
    "wiki_to_rcqa": 0.030,
    "general_reasoning_mix": 0.0187,
    "qwq-reasoning-traces": 0.0187,
    "megamatt": 0.0173,
    "llama_nemotron-reasoning-traces": 0.0125,
    "openthoughts2-reasoning-traces": 0.0125,
    "tulu-3-sft": 0.011,
    "tinymath-mind": 0.009,
    "omr-rewrite-fullthoughts": 0.0085,
    "code-meta-reasoning": 0.0046,
    "math-meta-reasoning": 0.0038,
    "gemini-reasoning-traces": 0.0025,
    "tinymath-pot": 0.0024,
    "program_verifiable": 0.0016,
}


def normalize_component_weights(weights: Mapping[str, float]) -> dict[str, float]:
    """Validate and normalize positive component weights in stable name order."""
    if not weights:
        raise ValueError("component weights must not be empty")
    clean: dict[str, float] = {}
    for component, value in weights.items():
        weight = float(value)
        if not component:
            raise ValueError("component names must not be empty")
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"component {component!r} has invalid weight {value!r}")
        clean[str(component)] = weight
    total = sum(clean.values())
    return {component: clean[component] / total for component in sorted(clean)}


def allocate_weighted_counts(
    total: int,
    weights: Mapping[str, float],
    *,
    require_all: bool = True,
) -> dict[str, int]:
    """Allocate exactly ``total`` integer items with the largest-remainder method."""
    if total <= 0:
        raise ValueError(f"total must be positive, got {total}")
    normalized = normalize_component_weights(weights)
    if require_all and total < len(normalized):
        raise ValueError(
            f"cannot represent all {len(normalized)} components with only {total} items"
        )

    raw = {component: total * weight for component, weight in normalized.items()}
    counts = {component: math.floor(value) for component, value in raw.items()}
    unallocated = total - sum(counts.values())
    order = sorted(normalized, key=lambda component: (-(raw[component] % 1), component))
    for component in order[:unallocated]:
        counts[component] += 1
    if require_all:
        for component in (name for name, count in counts.items() if count == 0):
            donors = [name for name, count in counts.items() if count > 1]
            if not donors:
                raise ValueError("could not reserve one item for every component")
            donor = max(
                donors,
                key=lambda name: (counts[name] - raw[name], counts[name], name),
            )
            counts[donor] -= 1
            counts[component] = 1
    return counts


def interleave_weighted_counts(counts: Mapping[str, int]) -> Iterator[str]:
    """Yield each component exactly its count, spread evenly through the sequence."""
    heap: list[tuple[float, str, int, int]] = []
    for component, value in sorted(counts.items()):
        count = int(value)
        if count < 0:
            raise ValueError(f"component {component!r} has negative count {count}")
        if count:
            heapq.heappush(heap, (0.5 / count, component, 0, count))

    while heap:
        _, component, index, count = heapq.heappop(heap)
        yield component
        next_index = index + 1
        if next_index < count:
            position = (next_index + 0.5) / count
            heapq.heappush(heap, (position, component, next_index, count))


def stable_component_seed(seed: int, component: str) -> int:
    """Derive a process-independent NumPy seed for one component."""
    digest = hashlib.blake2b(f"{seed}:{component}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little")
