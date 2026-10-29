"""Work-directory layout of the nl_tower data.

Every path derives from one root, ``NL_TOWER_WORKDIR`` (default ``$DATA_DIR/nl_tower``). The layout below is the one
the published artifacts (artifacts.json) were recorded in, so a rebuilt tree can be checked file by file.
"""

from __future__ import annotations

import os
from pathlib import Path

LAYOUT = {
    "data": "data",  # question pools, tests and the trace corpus: data/<spec>/corpus_<tag>/
    "rl_prompts": "results/rl_prompts",  # RL prompt pools: <spec>.jsonl
    "unified_data": "unified_data",  # trainer prompt JSONs, demo JSONL, anchors/<round>/ (demos and pretraining rows)
}


def workdir() -> Path:
    """``NL_TOWER_WORKDIR``, else ``$DATA_DIR/nl_tower``."""
    root = os.environ.get("NL_TOWER_WORKDIR")
    if root:
        return Path(root)
    base = os.environ.get("DATA_DIR")
    if base:
        return Path(base) / "nl_tower"
    raise SystemExit("set NL_TOWER_WORKDIR or DATA_DIR")


def get(name: str, root: str | Path | None = None) -> Path:
    """Directory ``name`` of LAYOUT under ``root`` (default: ``workdir()``)."""
    return Path(root if root is not None else workdir()) / LAYOUT[name]
