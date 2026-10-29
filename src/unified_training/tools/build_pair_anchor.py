# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Build one anchor round from a JSONL of {"prompt", "response"} pairs (static SFT-style data).

Writes ONE finalized round directory that UNIFIED_ANCHOR_DIRS can point at, one
episode per pair:

    tokens = tok(prompt, add_special_tokens=False) + tok(response, add_special_tokens=False) + [eos]

Prompt and response are tokenized separately and the response ends with EOS. No
chat template is applied: the prompt text is used verbatim.

ANCHOR_MASK selects which positions carry cross-entropy weight ANCHOR_WEIGHT:
  response  (default) the response tokens + EOS only (prompt_len = len(prompt ids)),
            the midtraining-during-RL mask of the controlled experiments (App. A.1);
  all       every predictable token (prompt_len=1: position 0 is never a label).

Rows whose prompt/response exceed ANCHOR_MAX_PROMPT / ANCHOR_MAX_RESPONSE tokens
(response counted with its EOS) are a hard error. CPU only; needs transformers + numpy.

  ANCHOR_JSONL=demos.jsonl ANCHOR_OUT=<dir> ANCHOR_TOKENIZER=<hf dir> \
  [ANCHOR_MASK=response|all ANCHOR_MAX_PROMPT=640 ANCHOR_MAX_RESPONSE=1280 ANCHOR_WEIGHT=1.0] \
  python -m unified_training.tools.build_pair_anchor
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from unified_training.episodes import RoundWriter

SRC = Path(os.environ["ANCHOR_JSONL"])
OUT = Path(os.environ["ANCHOR_OUT"])
TOKENIZER = os.environ["ANCHOR_TOKENIZER"]
MASK = os.environ.get("ANCHOR_MASK", "response")
MAX_PROMPT = int(os.environ.get("ANCHOR_MAX_PROMPT", "640"))
MAX_RESPONSE = int(os.environ.get("ANCHOR_MAX_RESPONSE", "1280"))
WEIGHT = float(os.environ.get("ANCHOR_WEIGHT", "1.0"))
BATCH = 4096

if MASK not in ("all", "response"):
    sys.exit(f"ERROR: ANCHOR_MASK must be 'all' or 'response', got {MASK!r}")
if OUT.exists():
    sys.exit(f"ERROR: refusing to overwrite existing {OUT}")

tok = AutoTokenizer.from_pretrained(TOKENIZER)
eos_id = tok.eos_token_id
if eos_id is None:
    sys.exit("ERROR: tokenizer has no eos_token_id")

pairs = []
with SRC.open() as stream:
    for line_no, line in enumerate(stream, 1):
        row = json.loads(line)
        if not isinstance(row.get("prompt"), str) or not isinstance(row.get("response"), str):
            sys.exit(f"ERROR: {SRC}:{line_no} needs string 'prompt' and 'response'")
        pairs.append((row["prompt"], row["response"]))
if not pairs:
    sys.exit(f"ERROR: {SRC} is empty")
print(f"[pair-anchor] {len(pairs)} pairs from {SRC} mask={MASK} tokenizer={TOKENIZER} eos={eos_id}", flush=True)

writer = RoundWriter(OUT)
too_long = 0
n_prompt_tokens = n_response_tokens = 0
digest = hashlib.sha256()
for start in range(0, len(pairs), BATCH):
    chunk = pairs[start:start + BATCH]
    prompt_ids = tok([p for p, _ in chunk], add_special_tokens=False)["input_ids"]
    response_ids = tok([r for _, r in chunk], add_special_tokens=False)["input_ids"]
    for offset, (p_ids, r_ids) in enumerate(zip(prompt_ids, response_ids)):
        r_ids = list(r_ids) + [eos_id]
        if len(p_ids) > MAX_PROMPT or len(r_ids) > MAX_RESPONSE or not p_ids:
            too_long += 1
            continue
        tokens = np.asarray(list(p_ids) + r_ids, dtype=np.int32)
        weights = np.zeros(len(tokens), dtype=np.float32)
        prompt_len = len(p_ids) if MASK == "response" else 1
        weights[prompt_len:] = WEIGHT
        writer.add(tokens, weights, prompt_len=prompt_len, reward=0.0, weight=WEIGHT,
                   policy_version=-1, group_id=start + offset)
        digest.update(tokens.tobytes())
        n_prompt_tokens += len(p_ids)
        n_response_tokens += len(r_ids)
    print(f"  {min(start + BATCH, len(pairs))}/{len(pairs)} pairs", flush=True)
if too_long:
    sys.exit(
        f"ERROR: {too_long} pairs exceed prompt {MAX_PROMPT} / response {MAX_RESPONSE} tokens "
        "(or have an empty prompt)"
    )

manifest = writer.finalize({
    "kind": "anchor", "format": "pairs", "mask": MASK, "source_jsonl": str(SRC),
    "tokenizer": TOKENIZER, "eos_token_id": int(eos_id), "anchor_weight": WEIGHT,
    "max_prompt_tokens": MAX_PROMPT, "max_response_tokens": MAX_RESPONSE,
    "tokenization": "tok(prompt,add_special_tokens=False) + tok(response,add_special_tokens=False) + [eos]",
    "n_prompt_tokens": n_prompt_tokens, "n_response_tokens_incl_eos": n_response_tokens,
    "tokens_sha256": digest.hexdigest(),
})
print(f"ANCHOR_DONE: {manifest['n_episodes']} episodes, {manifest['n_tokens']} tokens "
      f"({manifest['n_supervised_tokens']} supervised) -> {OUT}", flush=True)
