# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Pretraining rows (pre+mid+RL) for UNIFIED_PRETRAIN_DIRS (App. C.8), as one anchor round.

Each row is a contiguous window of CHUNK + 1 tokens at a random offset of a tokenized
pretraining shard (flat uint16 token ids, e.g. Megatron ``.bin`` files): the first
token is context (prompt_len 1, weight 0) and the next CHUNK tokens carry weight 1,
so every supervised token is a next-token prediction on pretraining text.

CHUNK is chosen so that the pretraining tokens of a step are ``--ratio`` times the
midtraining (anchor response) tokens of that step:

    CHUNK = round(ratio * demo_rows_per_step * mean_demo_response_tokens / pretrain_rows_per_step)

with the mean read from the anchor round's manifest (``mean_response_len``), or give
``--chunk`` directly. Sources are sampled by rows: ``--source NAME=FRACTION:file1,file2``;
every source but the last gets round(FRACTION * rows) rows, the last the remainder.
Rows are drawn per source in order (shard uniformly at random, then offset uniformly at
random) and written in one random permutation, all from ``numpy.random.default_rng(seed)``.

  python -m unified_training.tools.build_pretrain_rows OUT_DIR --rows 50000 --seed 2008 \\
      --demo-manifest ANCHOR_DIR/manifest.json --demo-rows-per-step 8 --pretrain-rows-per-step 4 \\
      --ratio 0.10 --source climbmix=0.75:a.bin,b.bin,c.bin --source dolmino=0.25:d.bin,e.bin,f.bin
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from unified_training.episodes import RoundWriter


def parse_source(spec: str) -> tuple[str, float, list[str]]:
    name, sep, rest = spec.partition("=")
    fraction, sep2, files = rest.partition(":")
    if not (sep and sep2 and name and files):
        raise ValueError(f"--source must be NAME=FRACTION:file1,file2 (got {spec!r})")
    return name, float(fraction), [f for f in files.split(",") if f]


def source_counts(fractions: list[float], rows: int) -> list[int]:
    counts = [round(f * rows) for f in fractions[:-1]]
    return counts + [rows - sum(counts)]


def sample_windows(shards: list[np.ndarray], count: int, chunk: int, rng) -> list[tuple[np.ndarray, int, int]]:
    out = []
    for _ in range(count):
        k = int(rng.integers(len(shards)))
        off = int(rng.integers(0, len(shards[k]) - chunk - 2))
        out.append((np.asarray(shards[k][off:off + chunk + 1], dtype=np.int32), k, off))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--rows", type=int, required=True, help="rows in the pool")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--source", action="append", required=True, help="NAME=FRACTION:file1,file2 (uint16 tokens)")
    ap.add_argument("--chunk", type=int, help="supervised tokens per row (overrides the ratio rule)")
    ap.add_argument("--demo-manifest", type=Path, help="manifest.json of the anchor (midtraining) round")
    ap.add_argument("--demo-rows-per-step", type=int, help="anchor rows per update (m)")
    ap.add_argument("--pretrain-rows-per-step", type=int, default=4)
    ap.add_argument("--ratio", type=float, default=0.10, help="pretraining tokens / midtraining tokens per step")
    args = ap.parse_args(argv)

    if args.out.exists():
        sys.exit(f"ERROR: refusing to overwrite {args.out}")
    meta: dict = {}
    if args.chunk is None:
        if args.demo_manifest is None or args.demo_rows_per_step is None:
            sys.exit("ERROR: give --chunk, or --demo-manifest with --demo-rows-per-step")
        mean_resp = float(json.loads(args.demo_manifest.read_text())["mean_response_len"])
        chunk = round(args.ratio * args.demo_rows_per_step * mean_resp / args.pretrain_rows_per_step)
        meta = {
            "demo_rows_per_step": args.demo_rows_per_step, "demo_mean_response_tokens": mean_resp,
            "ratio_target": args.ratio,
            "demo_response_tokens_per_step_expected": args.demo_rows_per_step * mean_resp,
            "realized_ratio_expected": args.pretrain_rows_per_step * chunk / (args.demo_rows_per_step * mean_resp),
        }
    else:
        chunk = args.chunk
    if chunk <= 0:
        sys.exit(f"ERROR: chunk must be positive, got {chunk}")

    sources = [parse_source(s) for s in args.source]
    counts = source_counts([f for _, f, _ in sources], args.rows)
    rng = np.random.default_rng(args.seed)
    rows: list[tuple[np.ndarray, str, int, int]] = []
    by_source = {}
    for (name, _fraction, files), count in zip(sources, counts):
        shards = [np.memmap(f, dtype=np.uint16, mode="r") for f in files]
        rows.extend((w, name, k, off) for w, k, off in sample_windows(shards, count, chunk, rng))
        by_source[name] = count
    perm = rng.permutation(len(rows))
    weights = np.ones(chunk + 1, dtype=np.float32)
    weights[0] = 0.0
    writer = RoundWriter(args.out)
    for i in perm:
        writer.add(rows[i][0], weights, prompt_len=1, reward=0.0, weight=1.0, policy_version=0, group_id=0)
    info = writer.finalize({
        "kind": "anchor", "format": "pretraining_windows", "mask": "all_but_first",
        "pretrain_rows_per_step": args.pretrain_rows_per_step, "chunk_supervised_tokens": chunk,
        "pretrain_tokens_per_step": args.pretrain_rows_per_step * chunk, "seed": args.seed,
        "rows_by_source": by_source, "sources": {n: [Path(f).name for f in fs] for n, _, fs in sources},
        "anchor_weight": 1.0, **meta,
    })
    (args.out / "provenance.json").write_text(json.dumps(
        [{"row": j, "source": rows[i][1], "shard": sources[[n for n, _, _ in sources].index(rows[i][1])][2][rows[i][2]],
          "offset": rows[i][3]} for j, i in enumerate(perm)], indent=0))
    print(args.out, json.dumps({k: info[k] for k in ("n_episodes", "n_tokens", "n_supervised_tokens",
                                                       "chunk_supervised_tokens", "rows_by_source")}))


if __name__ == "__main__":
    main()
