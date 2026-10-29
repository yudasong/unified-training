#!/usr/bin/env python3
"""Compare a rebuilt corpus with the paper's artifact: tokens.bin, weights.bin and index.npy must match."""
import json
import sys
from pathlib import Path

from common import load_receipt, sha


def main() -> None:
    name, out = sys.argv[1], Path(sys.argv[2])
    expected = load_receipt("expected_artifacts.json")[name]
    manifest = json.loads((out / "manifest.json").read_text())
    bad = {f: sha(out / f) for f in ("tokens.bin", "weights.bin", "index.npy") if sha(out / f) != expected[f]}
    for key in ("n_episodes", "n_supervised_tokens"):
        if manifest[key] != expected[key]:
            bad[key] = manifest[key]
    if bad:
        raise SystemExit(f"{name}: differs from the paper's artifact: {bad}")
    print(f"{name}: identical to the paper's artifact ({expected['name']})")


if __name__ == "__main__":
    main()
