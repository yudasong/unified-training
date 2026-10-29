# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Find benchmark questions inside an anchor round (input of ``tools.sanitize_anchor``).

Each benchmark question is tokenized (dropping its boundary-sensitive first token) and
searched in ``tokens.bin`` through its rarest token; only matches wholly inside one
episode count (episodes are independent training examples). With
``--min-match-tokens 16 --max-match-tokens 16`` the search uses each question's rarest
16-token window (the fingerprint scan) instead of the full question (the exact scan).

``--samples`` are JSONL (or Arrow stream) files of evaluation examples with a
``question`` field (optionally nested under ``doc``, with ``id`` / ``choices`` /
``answerKey``), e.g. lm-eval sample dumps of the evaluation suites.

    python -m unified_training.tools.audit_benchmark_overlap --tokens RAW/tokens.bin \\
        --index RAW/index.npy --tokenizer TOK --samples bench/*.jsonl --output exact.json \\
        --max-matches 1000 --fail-on-match-limit [--min-match-tokens 16 --max-match-tokens 16]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Example:
    doc_id: str
    question: str
    choices: tuple[str, ...]
    labels: tuple[str, ...]
    answer_key: str


@dataclass(frozen=True)
class Pattern:
    example_index: int
    token_ids: np.ndarray
    anchor_offset: int

    @property
    def anchor_id(self) -> int:
        return int(self.token_ids[self.anchor_offset])


def load_examples(paths: list[Path]) -> list[Example]:
    examples: list[Example] = []
    seen: set[str] = set()
    for path in paths:
        if path.suffix == ".arrow":
            import pyarrow as pa

            with pa.memory_map(str(path), "r") as source:
                records = pa.ipc.open_stream(source).read_all().to_pylist()
        else:
            with path.open() as stream:
                records = [json.loads(line) for line in stream]
        for record in records:
            doc = record["doc"] if "doc" in record else record
            doc_id = str(doc.get("id", record.get("doc_id", len(examples))))
            if doc_id in seen:
                continue
            seen.add(doc_id)
            choices = doc.get("choices", {})
            examples.append(
                Example(
                    doc_id=doc_id,
                    question=doc["question"],
                    choices=tuple(choices.get("text", ())),
                    labels=tuple(choices.get("label", ())),
                    answer_key=str(doc.get("answerKey", record.get("target", ""))),
                )
            )
    return examples


def token_frequencies(tokens: np.ndarray, vocab_size: int, chunk_tokens: int) -> np.ndarray:
    counts = np.zeros(vocab_size, dtype=np.int64)
    for start in range(0, len(tokens), chunk_tokens):
        chunk = np.asarray(tokens[start : start + chunk_tokens])
        maximum = int(chunk.max(initial=0))
        if maximum >= len(counts):
            counts.resize(maximum + 1, refcheck=False)
        counts[: maximum + 1] += np.bincount(chunk, minlength=maximum + 1)
    return counts


def make_patterns(
    examples: list[Example], tokenizer, counts: np.ndarray, min_match_tokens: int, max_match_tokens: int | None
) -> tuple[list[Pattern], list[int]]:
    patterns: list[Pattern] = []
    impossible: list[int] = []
    for example_index, example in enumerate(examples):
        ids = tokenizer.encode(example.question, add_special_tokens=False)
        # GPT-style tokenizers encode the first token differently after whitespace.
        # Dropping it makes the match independent of the source document boundary.
        if len(ids) - 1 >= min_match_tokens:
            ids = ids[1:]
        if len(ids) < min_match_tokens:
            impossible.append(example_index)
            continue
        token_ids = np.asarray(ids, dtype=np.int32)
        frequencies = counts[token_ids]
        if max_match_tokens and len(token_ids) > max_match_tokens:
            width = max_match_tokens
            present = (frequencies > 0).astype(np.int32)
            present_per_window = np.convolve(present, np.ones(width, dtype=np.int32), mode="valid")
            scores = np.convolve(np.log1p(frequencies), np.ones(width), mode="valid")
            scores[present_per_window != width] = np.inf
            if not np.isfinite(scores).any():
                impossible.append(example_index)
                continue
            window_start = int(np.argmin(scores))
            token_ids = token_ids[window_start : window_start + width]
            frequencies = frequencies[window_start : window_start + width]
        if np.any(frequencies == 0):
            impossible.append(example_index)
            continue
        patterns.append(Pattern(example_index, token_ids, int(np.argmin(frequencies))))
    return patterns, impossible


def scan_patterns(tokens: np.ndarray, patterns: list[Pattern], chunk_tokens: int, max_matches: int) -> dict[int, list[int]]:
    if not patterns:
        return {}
    by_anchor: dict[int, list[Pattern]] = defaultdict(list)
    for pattern in patterns:
        by_anchor[pattern.anchor_id].append(pattern)
    lookup = np.zeros(max(by_anchor) + 1, dtype=np.bool_)
    lookup[list(by_anchor)] = True
    matches: dict[int, list[int]] = defaultdict(list)
    for base in range(0, len(tokens), chunk_tokens):
        chunk = np.asarray(tokens[base : base + chunk_tokens])
        in_vocab = chunk < len(lookup)
        local_positions = np.flatnonzero(in_vocab & lookup[np.minimum(chunk, len(lookup) - 1)])
        if not len(local_positions):
            continue
        candidate_ids = chunk[local_positions]
        for anchor_id in np.unique(candidate_ids):
            positions = base + local_positions[candidate_ids == anchor_id]
            for pattern in by_anchor[int(anchor_id)]:
                if len(matches[pattern.example_index]) >= max_matches:
                    continue
                starts = positions.astype(np.int64) - pattern.anchor_offset
                starts = starts[(starts >= 0) & (starts + len(pattern.token_ids) <= len(tokens))]
                if not len(starts):
                    continue
                width = len(pattern.token_ids)
                for offset in sorted({0, 1, width // 2, width - 2, width - 1}):
                    starts = starts[tokens[starts + offset] == pattern.token_ids[offset]]
                    if not len(starts):
                        break
                for start in starts:
                    start_int = int(start)
                    if np.array_equal(tokens[start_int : start_int + width], pattern.token_ids):
                        matches[pattern.example_index].append(start_int)
                        if len(matches[pattern.example_index]) >= max_matches:
                            break
    return matches


def episode_matches(raw_matches: dict[int, list[int]], patterns: list[Pattern], index: np.ndarray):
    """Keep matches wholly inside one episode; returns ({example: [(offset, episode)]}, n_cross)."""
    patterns_by_example = {pattern.example_index: pattern for pattern in patterns}
    episode_offsets = np.asarray(index["offset"])
    matches: dict[int, list[tuple[int, int]]] = {}
    cross_episode = 0
    for example_index, offsets in raw_matches.items():
        width = len(patterns_by_example[example_index].token_ids)
        valid: list[tuple[int, int]] = []
        for offset in offsets:
            episode = int(np.searchsorted(episode_offsets, offset, side="right") - 1)
            if episode < 0:
                cross_episode += 1
                continue
            episode_start = int(index[episode]["offset"])
            episode_stop = episode_start + int(index[episode]["total_len"])
            if offset < episode_start or offset + width > episode_stop:
                cross_episode += 1
                continue
            valid.append((int(offset), episode))
        if valid:
            matches[example_index] = valid
    return matches, cross_episode


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--samples", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-tokens", type=int, default=20_000_000)
    parser.add_argument("--min-match-tokens", type=int, default=8)
    parser.add_argument("--max-match-tokens", type=int)
    parser.add_argument("--context-tokens", type=int, default=256)
    parser.add_argument("--max-matches", type=int, default=1000)
    parser.add_argument("--fail-on-match", action="store_true")
    parser.add_argument("--fail-on-match-limit", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    from transformers import AutoTokenizer

    args = parse_args(argv)
    if args.max_matches <= 0:
        raise ValueError("--max-matches must be positive")
    examples = load_examples(args.samples)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    tokens = np.memmap(args.tokens, dtype="<i4", mode="r")
    index = np.load(args.index, mmap_mode="r")
    if not len(index) or int(index[-1]["offset"] + index[-1]["total_len"]) != len(tokens):
        raise ValueError("index does not cover the token file")

    counts = token_frequencies(tokens, len(tokenizer), args.chunk_tokens)
    patterns, impossible = make_patterns(examples, tokenizer, counts, args.min_match_tokens, args.max_match_tokens)
    raw_matches = scan_patterns(tokens, patterns, args.chunk_tokens, args.max_matches)
    patterns_by_example = {pattern.example_index: pattern for pattern in patterns}
    matches, cross_episode = episode_matches(raw_matches, patterns, index)

    rows = []
    for example_index, example in enumerate(examples):
        width = len(patterns_by_example[example_index].token_ids) if example_index in patterns_by_example else 0
        described = []
        for offset, episode in matches.get(example_index, ()):
            episode_start = int(index[episode]["offset"])
            episode_stop = episode_start + int(index[episode]["total_len"])
            context_start = max(episode_start, offset - args.context_tokens)
            context_stop = min(episode_stop, offset + args.context_tokens)
            described.append({
                "token_offset": offset,
                "match_tokens": width,
                "episode": episode,
                "context": tokenizer.decode(tokens[context_start:context_stop].tolist()),
            })
        rows.append({
            "doc_id": example.doc_id,
            "question": example.question,
            "choices": example.choices,
            "labels": example.labels,
            "answer_key": example.answer_key,
            "matches": described,
        })

    matched = [row for row in rows if row["matches"]]
    at_limit = [row for row in matched if len(row["matches"]) == args.max_matches]
    report = {
        "tokens": str(args.tokens),
        "index": str(args.index),
        "samples": [str(path) for path in args.samples],
        "n_corpus_tokens": len(tokens),
        "max_match_tokens": args.max_match_tokens,
        "n_examples": len(examples),
        "n_searchable_examples": len(patterns),
        "n_absent_or_too_short": len(impossible),
        "n_matching_examples": len(matched),
        "n_matching_occurrences": sum(len(row["matches"]) for row in matched),
        "n_cross_episode_false_matches": cross_episode,
        "max_matches_per_example": args.max_matches,
        "examples_at_match_limit": [row["doc_id"] for row in at_limit],
        "examples": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"EPISODE_AUDIT: {len(matched)}/{len(examples)} matching examples, "
          f"{report['n_matching_occurrences']} occurrences, "
          f"{cross_episode} cross-episode candidates ignored -> {args.output}", flush=True)
    if args.fail_on_match_limit and at_limit:
        raise SystemExit(4)
    if args.fail_on_match and matched:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
