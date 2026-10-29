"""Demonstration rows for RL with midtraining data, from the trace corpus (CPU only).

``demos_<tag>.jsonl`` holds one ``{"prompt", "response"}`` object per corpus row, in corpus order: the raw prompt and
the solver trace (no EOS text). ``prepare_data.py demos`` turns it into the anchor round that ``UNIFIED_ANCHOR_DIRS``
points at, with the trainer's anchor builder and mask ``response`` (prompt tokens carry no loss; response tokens and
EOS carry weight 1). Rows longer than 640 prompt or 1,280 response tokens are an error (the corpus has none).
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def demo_row(row: dict) -> dict[str, str]:
    """A trace-corpus row ``{qid, prompt, target, answer}`` -> ``{prompt, response}``."""
    return {"prompt": row["prompt"], "response": row["target"]}


def write_demo_jsonl(corpus: str | Path, out: str | Path) -> int:
    """Stream ``corpus`` (cot.jsonl) into ``out`` (demos JSONL), atomically; refuses to overwrite. -> rows written."""
    out = str(out)
    if os.path.exists(out):
        raise FileExistsError(f"refusing to overwrite {out}")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    n = 0
    with open(corpus) as src, open(out + ".tmp", "w") as dst:
        for line in src:
            dst.write(json.dumps(demo_row(json.loads(line))) + "\n")
            n += 1
    os.replace(out + ".tmp", out)
    return n


# The trainer's builder of a {prompt, response} anchor round (env ANCHOR_JSONL, ANCHOR_OUT, ANCHOR_TOKENIZER,
# ANCHOR_MASK, ANCHOR_MAX_PROMPT, ANCHOR_MAX_RESPONSE, ANCHOR_WEIGHT), run as ``python -m``.
ANCHOR_BUILDER = os.environ.get("NL_TOWER_ANCHOR_BUILDER", "unified_training.tools.build_pair_anchor")
