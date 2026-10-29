#!/usr/bin/env bash
# Corpus variant "code replaced": the 240M midtraining corpus with its three code sources replaced by
# execution-tested Python answers from nvidia/OpenCodeInstruct (838,213 episodes, 239.8M supervised tokens).
# CPU only; the test sandbox needs unprivileged user namespaces (unshare) and a system python3.11 with
# libseccomp. Same variables as code_removed.sh, plus
#   SCRATCH         empty node-local directory for the sandbox roots
#   SANDBOX_PYTHON  system interpreter inside the sandbox (default /usr/bin/python3.11)
#   WORKERS         parallel test sandboxes (default 24)
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../../../.." && pwd)"
: "${ANCHOR_DIR:?}" "${OUT_DIR:?}" "${TOKENIZER:?}" "${DATA_DIR:?}" "${SCRATCH:?}"
py="${PYTHON:-python}"
export PYTHONPATH="$here:$repo/src${PYTHONPATH:+:$PYTHONPATH}"
holdout=(); [[ -n "${HOLDOUT_DIR:-}" ]] && holdout=(--holdout "$HOLDOUT_DIR")

samples="${BENCH_SAMPLES:-$DATA_DIR/decontamination/broad16-gsm8k-test.jsonl}"
[[ -s "$samples" ]] || "$py" "$repo/experiments/realworld/prepare/benchmark_samples.py" --out "$samples"

# 1. OpenCodeInstruct coding pool (nvidia/OpenCodeInstruct @ 8f3ba5ba, shards 0-9 of 50).
sources="$DATA_DIR/opencodeinstruct/sources"
raw="$DATA_DIR/opencodeinstruct/pool-raw"
pool="$DATA_DIR/opencodeinstruct/pool-clean"
"$py" "$here/oci_pool.py" stage --sources "$sources"
[[ -f "$raw/DONE" ]] || "$py" "$here/oci_pool.py" build --sources "$sources" --raw "$raw" --tokenizer "$TOKENIZER"
[[ -f "$pool/DONE" ]] || "$py" "$here/oci_pool.py" filter --raw "$raw" --pool "$pool"
[[ -f "$pool/READY" ]] || "$py" "$here/oci_pool.py" finalize --pool "$pool" --tokenizer "$TOKENIZER" --samples "$samples"

# 2. Sandboxed unit tests, assembly (seed 1234), final scans and validation.
work="$DATA_DIR/opencodeinstruct/verification"
out="$OUT_DIR/midtrain-code-replaced"
[[ -f "$work/VERIFIED" ]] || "$py" "$here/code_replaced.py" verify --pool "$pool" --sources "$sources" \
  --work "$work" --scratch "$SCRATCH" --workers "${WORKERS:-24}" --sandbox-python "${SANDBOX_PYTHON:-/usr/bin/python3.11}"
[[ -f "$out/manifest.json" ]] || "$py" "$here/code_replaced.py" assemble --anchor "$ANCHOR_DIR" --pool "$pool" \
  --work "$work" --output "$out" --seed 1234 "${holdout[@]}"
[[ -f "$out/READY" ]] || "$py" "$here/code_replaced.py" validate --work "$work" --output "$out" \
  --tokenizer "$TOKENIZER" --samples "$samples"
"$py" "$here/check_expected.py" code_replaced "$out"
