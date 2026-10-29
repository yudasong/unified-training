#!/usr/bin/env bash
# Corpus variant "code removed": the 240M midtraining corpus without cranecode, stack_edu-fim and
# code-meta-reasoning (647,327 windows, 180.4M supervised tokens). CPU only.
#   ANCHOR_DIR  the 240M midtraining corpus (python -m unified_training.tools ... document anchor builder)
#   OUT_DIR     where the new corpus is written
#   TOKENIZER   HF directory of the base model (tokenizer.json sha256 66e4a08a...)
#   DATA_DIR    working directory
#   BENCH_SAMPLES  benchmark bundle JSONL (default: built under DATA_DIR by ../benchmark_samples.py)
#   HOLDOUT_DIR optional: trainer validation-holdout dir of ANCHOR_DIR (else the pinned excluded-id receipt)
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../../../.." && pwd)"
: "${ANCHOR_DIR:?}" "${OUT_DIR:?}" "${TOKENIZER:?}" "${DATA_DIR:?}"
py="${PYTHON:-python}"
export PYTHONPATH="$here:$repo/src${PYTHONPATH:+:$PYTHONPATH}"
holdout=(); [[ -n "${HOLDOUT_DIR:-}" ]] && holdout=(--holdout "$HOLDOUT_DIR")

samples="${BENCH_SAMPLES:-$DATA_DIR/decontamination/broad16-gsm8k-test.jsonl}"
[[ -s "$samples" ]] || "$py" "$repo/experiments/realworld/prepare/benchmark_samples.py" --out "$samples"
out="$OUT_DIR/midtrain-code-removed"
[[ -f "$out/manifest.json" ]] || "$py" "$here/code_removed.py" assemble --anchor "$ANCHOR_DIR" --output "$out" "${holdout[@]}"
[[ -f "$out/READY" ]] || "$py" "$here/code_removed.py" validate --anchor "$ANCHOR_DIR" --output "$out" \
  --tokenizer "$TOKENIZER" --samples "$samples" "${holdout[@]}"
"$py" "$here/check_expected.py" code_removed "$out"
