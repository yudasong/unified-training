# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Build the document-aware midtraining anchor (the 240M-token corpus of Sec. 5.2 / App. D).

Documents are never concatenated: each document (plus EOS) becomes one episode, and a
document longer than ANCHOR_WINDOW tokens is split only within itself. Every token but
the first of an episode carries weight 1 (document continuation, unit weights). Component
quotas follow the OLMo-3 Dolmino 24-component mixture with common crawl reduced to 5%,
counted over supervised prediction tokens.

Per component: the source rows of that component are Bernoulli-subsampled at
``min(1, ANCHOR_SAMPLE_MARGIN * quota / estimated_supply)`` (shard order and rows
shuffled with a component-specific seed), tokenized into a candidate spool until
``ceil(margin * quota)`` supervised tokens, and a seeded permutation of the candidates
is taken until the quota is met. The output interleaves components evenly.

Required: ANCHOR_SRC (dir of parquet shards with ``component`` and ``text`` columns),
ANCHOR_OUT, ANCHOR_TOKENIZER (HF tokenizer dir). Optional (paper values):
ANCHOR_TOKENS=240000000 ANCHOR_WINDOW=1025 ANCHOR_SAMPLE_MARGIN=1.4 ANCHOR_SEED=1234
ANCHOR_BATCH_ROWS=2048 ANCHOR_SOURCE_TOKENS=10000000000 ANCHOR_GLOB=train-*.parquet
ANCHOR_TEXT_COL=text ANCHOR_COMPONENT_COL=component; provenance strings recorded in the
manifest: ANCHOR_SOURCE_REPO_ID, ANCHOR_SOURCE_REVISION, ANCHOR_SOURCE_MANIFEST_SHA256,
ANCHOR_TOKENIZER_SHA256.

    python -m unified_training.tools.build_document_anchor

Then decontaminate with ``tools.audit_benchmark_overlap`` and ``tools.sanitize_anchor``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from unified_training.corpus_mix import (
    OLMO3_COMPONENT_WEIGHTS,
    allocate_weighted_counts,
    interleave_weighted_counts,
    normalize_component_weights,
    stable_component_seed,
)
from unified_training.episodes import INDEX_DTYPE

SUPERVISED_TOKEN_COUNT_BASIS = "count(token_weight > 0)"
VARIANT = "boundary"
CANDIDATE_DTYPE = np.dtype(
    [
        ("offset", "<i8"),
        ("total_len", "<i4"),
        ("weighted_tokens", "<i4"),
        ("prompt_len", "<i4"),
        ("truncated", "u1"),
        ("original_total_len", "<i4"),
    ]
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def component_profile() -> dict[str, float]:
    """The OLMo-3 mixture with common crawl reduced from 22.5% to 5%, normalized."""
    weights = dict(OLMO3_COMPONENT_WEIGHTS)
    weights["common_crawl-high-quality"] = 0.05
    return normalize_component_weights(weights)


def inventory_components(
    shards: list[Path], component_col: str, text_col: str
) -> tuple[dict[str, int], dict[str, list[Path]]]:
    counts: Counter[str] = Counter()
    component_shards: dict[str, list[Path]] = defaultdict(list)
    for shard in shards:
        parquet = pq.ParquetFile(shard)
        columns = set(parquet.schema_arrow.names)
        missing = {component_col, text_col} - columns
        if missing:
            raise ValueError(f"{shard} is missing required columns {sorted(missing)}")
        local: Counter[str] = Counter()
        for batch in parquet.iter_batches(columns=[component_col], batch_size=1_000_000):
            local.update(str(value) for value in batch.column(0).to_pylist() if value)
        counts.update(local)
        for component in local:
            component_shards[component].append(shard)
    return dict(sorted(counts.items())), {
        component: sorted(paths) for component, paths in sorted(component_shards.items())
    }


def iter_sampled_raw_documents(
    *,
    component: str,
    shards: list[Path],
    component_col: str,
    text_col: str,
    sample_fraction: float,
    seed: int,
    batch_rows: int,
    stats: dict[str, int | float],
):
    rng = np.random.default_rng(stable_component_seed(seed, component))
    shard_order = list(shards)
    rng.shuffle(shard_order)
    for shard in shard_order:
        stats["shards_visited"] += 1
        parquet = pq.ParquetFile(shard)
        for batch in parquet.iter_batches(
            columns=[component_col, text_col], batch_size=batch_rows
        ):
            components = batch.column(0).to_pylist()
            indices = np.fromiter(
                (index for index, value in enumerate(components) if value == component),
                dtype=np.int64,
            )
            stats["rows_scanned"] += len(indices)
            if not len(indices):
                continue
            if sample_fraction < 1.0:
                indices = indices[rng.random(len(indices)) < sample_fraction]
            rng.shuffle(indices)
            texts = batch.column(1).take(pa.array(indices)).to_pylist()
            for text in texts:
                if not text or not text.strip():
                    stats["empty_rows"] += 1
                    continue
                stats["sampled_docs"] += 1
                yield text


def iter_raw_episodes(texts, tokenizer, eos_id: int, window: int, stats: dict):
    for text in texts:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            stats["empty_rows"] += 1
            continue
        token_ids.append(eos_id)
        stats["sampled_document_tokens"] += len(token_ids)
        n_segments = math.ceil(len(token_ids) / window)
        if n_segments > 1:
            stats["split_documents"] += 1
        for start in range(0, len(token_ids), window):
            segment = np.asarray(token_ids[start : start + window], dtype=np.int32)
            if len(segment) < 2:
                stats["dropped_single_token_segments"] += 1
                continue
            mask = np.ones(len(segment), dtype=np.uint8)
            mask[0] = 0
            yield segment, mask


def write_candidate_spool(
    episodes,
    *,
    directory: Path,
    component_id: int,
    stats: dict,
    stop_after_weighted_tokens: int,
) -> tuple[Path, Path, np.ndarray]:
    tokens_path = directory / f"{component_id:02d}.tokens.bin"
    masks_path = directory / f"{component_id:02d}.masks.bin"
    records: list[tuple[int, int, int, int, int, int]] = []
    offset = 0
    with tokens_path.open("wb") as token_stream, masks_path.open("wb") as mask_stream:
        for tokens, mask in episodes:
            if len(tokens) != len(mask) or mask[0] != 0:
                raise ValueError("candidate tokens/mask are misaligned")
            weighted = int(np.count_nonzero(mask))
            if weighted <= 0:
                continue
            prompt_len = int(np.flatnonzero(mask)[0])
            np.asarray(tokens, dtype="<i4").tofile(token_stream)
            np.asarray(mask, dtype=np.uint8).tofile(mask_stream)
            records.append((offset, len(tokens), weighted, prompt_len, 0, len(tokens)))
            offset += len(tokens)
            stats["candidate_episodes"] += 1
            stats["candidate_tokens"] += len(tokens)
            stats["candidate_weighted_tokens"] += weighted
            stats["candidate_supervised_tokens"] += weighted
            if stats["candidate_weighted_tokens"] >= stop_after_weighted_tokens:
                break
    metadata = np.asarray(records, dtype=CANDIDATE_DTYPE)
    if not len(metadata):
        raise RuntimeError("component produced no candidate episodes")
    return tokens_path, masks_path, metadata


def select_episodes(
    metadata: np.ndarray, *, target_weighted_tokens: int, seed: int, component: str
) -> tuple[np.ndarray, int, int]:
    """Select candidates in a seeded random order, without replacement, until the quota is met."""
    rng = np.random.default_rng(stable_component_seed(seed, f"{component}:selection"))
    order = rng.permutation(len(metadata))
    selected: list[int] = []
    weighted = 0
    tokens = 0
    for index in order:
        selected.append(int(index))
        weighted += int(metadata[index]["weighted_tokens"])
        tokens += int(metadata[index]["total_len"])
        if weighted >= target_weighted_tokens:
            break
    if weighted < target_weighted_tokens:
        raise RuntimeError(
            f"component {component!r} supplied {weighted:,} supervised candidate tokens "
            f"for a {target_weighted_tokens:,} target; increase ANCHOR_SAMPLE_MARGIN"
        )
    return np.asarray(selected, dtype=np.int64), weighted, tokens


def build_output(
    *,
    output: Path,
    component_weights: dict[str, float],
    component_ids: dict[str, int],
    spools: dict[str, tuple[Path, Path, np.ndarray]],
    selections: dict[str, np.ndarray],
    manifest: dict,
) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{os.getpid()}"
    if temporary.exists() or output.exists():
        raise FileExistsError(f"refusing to overwrite {temporary if temporary.exists() else output}")
    temporary.mkdir()
    counts = {component: len(selection) for component, selection in selections.items()}
    n_episodes = sum(counts.values())
    index = np.empty(n_episodes, dtype=INDEX_DTYPE)
    token_maps = {component: np.memmap(paths[0], dtype="<i4", mode="r") for component, paths in spools.items()}
    mask_maps = {component: np.memmap(paths[1], dtype=np.uint8, mode="r") for component, paths in spools.items()}
    cursors = {component: 0 for component in component_weights}
    component_tokens = Counter()
    component_weighted = Counter()
    output_offset = 0
    try:
        with (temporary / "tokens.bin").open("wb") as token_stream, (
            temporary / "weights.bin"
        ).open("wb") as weight_stream:
            for output_index, component in enumerate(interleave_weighted_counts(counts)):
                selected_index = int(selections[component][cursors[component]])
                cursors[component] += 1
                metadata = spools[component][2][selected_index]
                start = int(metadata["offset"])
                stop = start + int(metadata["total_len"])
                tokens = np.asarray(token_maps[component][start:stop], dtype="<i4")
                mask = np.asarray(mask_maps[component][start:stop], dtype=np.uint8)
                tokens.tofile(token_stream)
                np.asarray(mask, dtype="<f4").tofile(weight_stream)
                weighted = int(np.count_nonzero(mask))
                index[output_index] = (
                    output_offset, int(metadata["prompt_len"]), len(tokens), 0.0, 1.0, -1,
                    component_ids[component],
                )
                output_offset += len(tokens)
                component_tokens[component] += len(tokens)
                component_weighted[component] += weighted
                if (output_index + 1) % 50_000 == 0:
                    print(f"  [write] {output_index + 1:,}/{n_episodes:,} episodes, "
                          f"{output_offset / 1e6:.1f}M tokens", flush=True)
        np.save(temporary / "index.npy", index)
        n_weighted = sum(component_weighted.values())
        manifest.update(
            n_episodes=n_episodes,
            n_tokens=output_offset,
            n_weighted_tokens=n_weighted,
            n_supervised_tokens=n_weighted,
            supervised_token_count_basis=SUPERVISED_TOKEN_COUNT_BASIS,
            max_token_id=max(int(token_map.max()) for token_map in token_maps.values()),
            mean_response_len=float(np.mean(index["total_len"] - index["prompt_len"])),
            mean_weighted_tokens_per_episode=n_weighted / n_episodes,
            mean_reward=0.0,
            mean_weight=1.0,
            component_realized_tokens=dict(component_tokens),
            component_realized_weighted_tokens=dict(component_weighted),
            component_realized_supervised_tokens=dict(component_weighted),
            component_realized_weighted_token_fractions={
                component: component_weighted[component] / n_weighted for component in component_weights
            },
        )
        manifest["component_realized_episodes"] = {component: counts[component] for component in component_weights}
        manifest["payload_sha256"] = {
            name: file_sha256(temporary / name) for name in ("index.npy", "tokens.bin", "weights.bin")
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (temporary / "DONE").touch()
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def main() -> None:
    from transformers import AutoTokenizer

    source = Path(os.environ["ANCHOR_SRC"])
    output = Path(os.environ["ANCHOR_OUT"])
    tokenizer_path = Path(os.environ["ANCHOR_TOKENIZER"])
    variant = os.environ.get("ANCHOR_VARIANT", VARIANT)
    if variant != VARIANT:
        raise ValueError(f"only ANCHOR_VARIANT={VARIANT} (the paper's corpus) is supported, got {variant!r}")
    target_tokens = int(os.environ.get("ANCHOR_TOKENS", "240000000"))
    raw_window = int(os.environ.get("ANCHOR_WINDOW", "1025"))
    source_tokens = int(os.environ.get("ANCHOR_SOURCE_TOKENS", "10000000000"))
    sample_margin = float(os.environ.get("ANCHOR_SAMPLE_MARGIN", "1.4"))
    seed = int(os.environ.get("ANCHOR_SEED", "1234"))
    batch_rows = int(os.environ.get("ANCHOR_BATCH_ROWS", "2048"))
    glob_pattern = os.environ.get("ANCHOR_GLOB", "train-*.parquet")
    text_col = os.environ.get("ANCHOR_TEXT_COL", "text")
    component_col = os.environ.get("ANCHOR_COMPONENT_COL", "component")
    if target_tokens <= 0 or raw_window <= 1 or source_tokens <= 0 or sample_margin <= 1:
        raise ValueError("invalid token/window/source/margin settings")
    if (output / "DONE").exists():
        raise FileExistsError(f"refusing to overwrite finalized anchor {output}")

    shards = sorted(source.glob(glob_pattern))
    if not shards:
        raise FileNotFoundError(f"no source parquet under {source} matching {glob_pattern}")
    source_rows, component_shards = inventory_components(shards, component_col, text_col)
    component_weights = component_profile()
    missing = sorted(set(component_weights) - set(source_rows))
    if missing:
        raise ValueError(f"source is missing required components: {missing}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("tokenizer has no EOS token id")
    target_by_component = allocate_weighted_counts(target_tokens, component_weights)
    original_weights = normalize_component_weights(OLMO3_COMPONENT_WEIGHTS)
    component_ids = {component: index for index, component in enumerate(component_weights)}

    print(f"[document-anchor] variant={variant} target={target_tokens:,} supervised tokens "
          f"raw_window={raw_window} reasoning_window={raw_window} "
          f"margin={sample_margin} source={source} -> {output}", flush=True)
    for component, weight in component_weights.items():
        print(f"  {component:36} target={100 * weight:6.3f}% "
              f"supervised_tokens={target_by_component[component]:,}", flush=True)

    build_stats: dict[str, dict] = {}
    spool_summaries: dict[str, dict] = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-candidates-", dir=output.parent) as tmp:
        tmp_path = Path(tmp)
        spools = {}
        selections = {}
        for component in component_weights:
            target = target_by_component[component]
            estimated_source = max(1, int(source_tokens * original_weights[component]))
            sample_fraction = min(1.0, sample_margin * target / estimated_source)
            stats = {
                "source_rows": source_rows.get(component, 0),
                "source_shards": len(component_shards.get(component, [])),
                "shards_visited": 0,
                "rows_scanned": 0,
                "source_sequences": 0,
                "sample_fraction": sample_fraction,
                "sampled_docs": 0,
                "sampled_document_tokens": 0,
                "split_documents": 0,
                "empty_rows": 0,
                "dropped_single_token_segments": 0,
                "dropped_overlong_or_short_sequences": 0,
                "dropped_zero_loss_sequences": 0,
                "candidate_episodes": 0,
                "candidate_tokens": 0,
                "candidate_weighted_tokens": 0,
                "candidate_supervised_tokens": 0,
                "target_weighted_tokens": target,
                "target_supervised_tokens": target,
            }
            print(f"[document-anchor] sampling {component} at {sample_fraction:.5f}", flush=True)
            texts = iter_sampled_raw_documents(
                component=component, shards=component_shards[component], component_col=component_col,
                text_col=text_col, sample_fraction=sample_fraction, seed=seed, batch_rows=batch_rows,
                stats=stats,
            )
            episodes = iter_raw_episodes(texts, tokenizer, eos_id, raw_window, stats)
            spool = write_candidate_spool(
                episodes, directory=tmp_path, component_id=component_ids[component], stats=stats,
                stop_after_weighted_tokens=math.ceil(sample_margin * target),
            )
            selection, selected_weighted, selected_tokens = select_episodes(
                spool[2], target_weighted_tokens=target, seed=seed, component=component
            )
            spools[component] = spool
            selections[component] = selection
            stats["selected_episodes"] = len(selection)
            stats["selected_tokens"] = selected_tokens
            stats["selected_weighted_tokens"] = selected_weighted
            stats["selected_supervised_tokens"] = selected_weighted
            build_stats[component] = stats
            spool_summaries[component] = {
                "candidate_episodes": len(spool[2]),
                "selected_episodes": len(selection),
                "selected_tokens": selected_tokens,
                "selected_weighted_tokens": selected_weighted,
                "selected_supervised_tokens": selected_weighted,
            }
            print(f"  selected {len(selection):,}/{len(spool[2]):,} episodes, "
                  f"{selected_weighted:,} supervised tokens", flush=True)

        # Manifest keys match the builder that produced the paper's corpus; the keys of
        # the unused (chat / reasoning) variants are kept as null for compatibility.
        manifest = build_output(
            output=output,
            component_weights=component_weights,
            component_ids=component_ids,
            spools=spools,
            selections=selections,
            manifest={
                "kind": "anchor",
                "format": "document_aware_raw",
                "variant": variant,
                "source": str(source),
                "source_glob": glob_pattern,
                "source_repo_id": os.environ.get("ANCHOR_SOURCE_REPO_ID"),
                "source_revision": os.environ.get("ANCHOR_SOURCE_REVISION"),
                "source_manifest_sha256": os.environ.get("ANCHOR_SOURCE_MANIFEST_SHA256"),
                "tokenizer": str(tokenizer_path),
                "tokenizer_sha256": os.environ.get("ANCHOR_TOKENIZER_SHA256"),
                "builder_sha256": file_sha256(Path(__file__)),
                "requested_weighted_tokens": target_tokens,
                "requested_supervised_tokens": target_tokens,
                "supervised_token_count_basis": SUPERVISED_TOKEN_COUNT_BASIS,
                "window": raw_window,
                "raw_window": raw_window,
                "reasoning_window": None,
                "component_windows": {component: raw_window for component in component_weights},
                "anchor_weight": 1.0,
                "mix_mode": "component_supervised_prediction_tokens",
                "component_mix_profile": f"olmo3-commoncrawl5-docaware-{variant}",
                "section_fraction": None,
                "fim_filter": None,
                "reasoning_truncated_selection": None,
                "component_weights": component_weights,
                "component_target_weighted_tokens": target_by_component,
                "component_target_supervised_tokens": target_by_component,
                "component_group_ids": component_ids,
                "component_seed": seed,
                "component_sample_margin": sample_margin,
                "component_batch_rows": batch_rows,
                "source_token_estimate": source_tokens,
                "text_column": text_col,
                "component_column": component_col,
                "document_boundary_policy": "one_document_per_episode; split_only_within_document",
                "chat_replacement": None,
                "reasoning_sft_replacement": None,
                "source_component_rows": source_rows,
                "component_build_stats": build_stats,
                "component_spool_summaries": spool_summaries,
            },
        )

    print(f"ANCHOR_DONE: {manifest['n_episodes']:,} episodes, "
          f"{manifest['n_tokens'] / 1e9:.3f}B stored tokens, "
          f"{manifest['n_weighted_tokens'] / 1e9:.3f}B supervised tokens -> {output}", flush=True)


if __name__ == "__main__":
    main()
