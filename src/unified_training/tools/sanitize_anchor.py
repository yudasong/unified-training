# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Decontaminate a document-aware anchor: replace every episode an overlap report flags.

Each flagged episode is replaced by a clean episode of the same component (a seeded
shuffle of that component's clean episodes), so the anchor keeps its shape and mixture.
Reports come from ``tools.audit_benchmark_overlap`` (the paper ran two scans: exact full
questions and the rarest 16-token window, over the broad16 + GSM8K evaluation bundle).

    python -m unified_training.tools.sanitize_anchor --source RAW --output CLEAN \\
        --report exact.json --report fingerprint16.json --suite broad16-gsm8k --split test --seed 1234
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from unified_training.corpus_mix import stable_component_seed
from unified_training.episodes import INDEX_DTYPE


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--suite", default="broad16")
    parser.add_argument("--split", default="test")
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contaminated_episodes(reports: list[Path], n_episodes: int) -> tuple[set[int], list[dict]]:
    contaminated: set[int] = set()
    records: list[dict] = []
    for report_path in reports:
        report = json.loads(report_path.read_text())
        for example in report.get("examples", []):
            for match in example.get("matches", []):
                episode = int(match["episode"])
                if not 0 <= episode < n_episodes:
                    raise ValueError(f"report episode {episode} is outside the source pool")
                contaminated.add(episode)
                records.append(
                    {
                        "report": report_path.name,
                        "doc_id": example.get("doc_id"),
                        "episode": episode,
                        "token_offset": int(match["token_offset"]),
                        "match_tokens": int(match["match_tokens"]),
                    }
                )
    return contaminated, records


def choose_donors(
    index: np.ndarray, contaminated: set[int], *, seed: int, group_names: dict[int, str]
) -> dict[int, int]:
    bad_by_group: dict[int, list[int]] = defaultdict(list)
    for episode in sorted(contaminated):
        bad_by_group[int(index[episode]["group_id"])].append(episode)
    all_indices = np.arange(len(index), dtype=np.int64)
    donors: dict[int, int] = {}
    bad_array = np.asarray(sorted(contaminated), dtype=np.int64)
    for group_id, bad_episodes in sorted(bad_by_group.items()):
        candidates = all_indices[index["group_id"] == group_id]
        candidates = candidates[~np.isin(candidates, bad_array)]
        if len(candidates) < len(bad_episodes):
            raise ValueError(f"component group {group_id} has too few clean donor episodes")
        rng = np.random.default_rng(stable_component_seed(seed, f"decontam:{group_names[group_id]}"))
        rng.shuffle(candidates)
        donors.update(
            {bad: int(donor) for bad, donor in zip(bad_episodes, candidates[: len(bad_episodes)], strict=True)}
        )
    return donors


def sanitize(source: Path, output: Path, reports: list[Path], *, seed: int, suite: str, split: str) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if not (source / "DONE").exists():
        raise ValueError(f"source is not finalized: {source}")
    manifest = json.loads((source / "manifest.json").read_text())
    index = np.load(source / "index.npy", mmap_mode="r")
    tokens = np.memmap(source / "tokens.bin", dtype="<i4", mode="r")
    weights = np.memmap(source / "weights.bin", dtype="<f4", mode="r")
    if len(tokens) != len(weights) or len(index) != int(manifest["n_episodes"]):
        raise ValueError("source files disagree with the manifest")
    group_ids = manifest["component_group_ids"]
    group_names = {int(group_id): component for component, group_id in group_ids.items()}
    contaminated, match_records = contaminated_episodes(reports, len(index))
    donors = choose_donors(index, contaminated, seed=seed, group_names=group_names)
    output_source_episodes = [donors.get(episode, episode) for episode in range(len(index))]

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{os.getpid()}"
    if temporary.exists():
        raise FileExistsError(f"temporary path already exists: {temporary}")
    temporary.mkdir()
    output_index = np.empty(len(output_source_episodes), dtype=INDEX_DTYPE)
    output_offset = 0
    n_weighted = 0
    max_token_id = -1
    component_tokens = Counter()
    component_weighted = Counter()
    component_episodes = Counter()
    try:
        with (temporary / "tokens.bin").open("wb") as token_stream, (
            temporary / "weights.bin"
        ).open("wb") as weight_stream:
            for output_episode, source_episode in enumerate(output_source_episodes):
                row = index[source_episode]
                start = int(row["offset"])
                stop = start + int(row["total_len"])
                episode_tokens = np.asarray(tokens[start:stop], dtype="<i4")
                episode_weights = np.asarray(weights[start:stop], dtype="<f4")
                episode_tokens.tofile(token_stream)
                episode_weights.tofile(weight_stream)
                output_index[output_episode] = (
                    output_offset, int(row["prompt_len"]), len(episode_tokens), float(row["reward"]),
                    float(row["weight"]), int(row["policy_version"]), int(row["group_id"]),
                )
                weighted = int(np.count_nonzero(episode_weights > 0))
                component = group_names[int(row["group_id"])]
                component_episodes[component] += 1
                component_tokens[component] += len(episode_tokens)
                component_weighted[component] += weighted
                output_offset += len(episode_tokens)
                n_weighted += weighted
                max_token_id = max(max_token_id, int(episode_tokens.max(initial=-1)))
                if (output_episode + 1) % 50_000 == 0:
                    print(f"  [sanitize] {output_episode + 1:,}/{len(output_source_episodes):,} episodes", flush=True)
        if not len(output_index):
            raise ValueError("sanitization removed every source episode")
        np.save(temporary / "index.npy", output_index)
        report_dir = temporary / "decontamination_reports"
        report_dir.mkdir()
        report_metadata = []
        for report in reports:
            shutil.copy2(report, report_dir / report.name)
            report_metadata.append({"file": str(Path("decontamination_reports") / report.name), "sha256": sha256(report)})
        components = manifest["component_weights"]
        manifest.update(
            parent_anchor_pool=str(source),
            n_episodes=len(output_index),
            n_tokens=output_offset,
            n_weighted_tokens=n_weighted,
            n_supervised_tokens=n_weighted,
            supervised_token_count_basis="count(token_weight > 0)",
            max_token_id=max_token_id,
            mean_response_len=float(np.mean(output_index["total_len"] - output_index["prompt_len"])),
            mean_weighted_tokens_per_episode=n_weighted / len(output_index),
            mean_supervised_tokens_per_episode=n_weighted / len(output_index),
            mean_reward=float(np.mean(output_index["reward"])),
            mean_weight=float(np.mean(output_index["weight"])),
            component_realized_episodes={c: component_episodes[c] for c in components},
            component_realized_tokens={c: component_tokens[c] for c in components},
            component_realized_weighted_tokens={c: component_weighted[c] for c in components},
            component_realized_supervised_tokens={c: component_weighted[c] for c in components},
            component_realized_weighted_token_fractions={c: component_weighted[c] / n_weighted for c in components},
            component_realized_supervised_token_fractions={c: component_weighted[c] / n_weighted for c in components},
        )
        reuse_count = len(output_source_episodes) - len(set(output_source_episodes))
        manifest["decontamination"] = {
            "benchmark": suite,
            "splits": [split],
            "method": "episode_aware_same_component_replacement",
            "strategy": "replace",
            "seed": seed,
            "n_reported_matches": len(match_records),
            "n_replaced_episodes": len(donors),
            "source_episode_reuse_count": reuse_count,
            "retained_source_episode_ids_unique": reuse_count == 0,
            "reports": report_metadata,
            "matches": match_records,
            "replacements": [
                {"episode": episode, "donor_episode": donor, "component": group_names[int(index[episode]["group_id"])]}
                for episode, donor in sorted(donors.items())
            ],
        }
        manifest["payload_sha256"] = {name: sha256(temporary / name) for name in ("index.npy", "tokens.bin", "weights.bin")}
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (temporary / "DONE").touch()
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"ANCHOR_SANITIZED: replaced {len(donors)} episodes for {suite}:{split} -> {output}", flush=True)
    return manifest


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    sanitize(args.source, args.output, args.report, seed=args.seed, suite=args.suite, split=args.split)


if __name__ == "__main__":
    main()
