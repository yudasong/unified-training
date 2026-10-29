#!/usr/bin/env bash
# Data for the real-world experiments (CPU only).
#   bash prepare.sh [math12k] [kodcode] [corpora]        (default: all three)
#   corpora downloads the published midtraining corpora and held-out sets (HF dataset sfanm/unified-training,
#   realworld/*) into $DATA_DIR/realworld/; the optional stages rebuild_midtrain and rebuild_code_variants rebuild them.
# Env: DATA_DIR (outputs), CKPT_DIR (base model at $CKPT_DIR/realworld/base, for its tokenizer; downloaded if missing),
#   PYTHON (default: python of the active environment, the benchmark environment of the README),
#   MIDTRAIN_SRC   rebuild_midtrain only: train-*.parquet (component, text) of the base model's midtraining corpus; only its
#                  document hashes and token mixture are used, to keep the 240M corpus disjoint from it. That corpus is
#                  not part of this release, so the rebuild is for reference; the runs use the downloaded corpus
#   SCRATCH        rebuild_code_variants only: empty node-local directory for the unit-test sandboxes
# Outputs under DATA_DIR: math12k/, kodcode/, realworld/{midtrain-240m (the midtraining corpus), midtrain-val-holdout
# (trainer validation windows), heldout-loss-slices (24-component held-out loss), midtrain-code-{removed,replaced}}.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); repo=$(cd "$here/../.." && pwd); p=$here/prepare
py=${PYTHON:-python}; D=${DATA_DIR:?}; base=${CKPT_DIR:?}/realworld/base
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
hf_get() { [[ -f $CKPT_DIR/$1/config.json ]] || hf download sfanm/unified-training --include "$1/*" --local-dir "$CKPT_DIR"; }
stages=("$@"); [[ ${#stages[@]} -gt 0 ]] || stages=(math12k kodcode corpora)

midtrain() {
  hf_get realworld/base
  local inv=$D/dolmino-midtrain-inventory fresh=$D/dolmino-fresh raw=$D/midtrain-240m-raw out=$D/realworld/midtrain-240m
  [[ -f $inv/corpus_hashes.npy ]] || CORPUS_SRC=${MIDTRAIN_SRC:?} CORPUS_OUT=$inv CORPUS_TOKENIZER=$base \
    "$py" "$p/dolmino_inventory.py"
  "$py" "$p/stage_dolmino_fresh.py" --dest "$fresh/upstream" --files-from "$p/receipts/dolmino_fresh_stage_manifest.json"
  [[ -f $fresh/raw/source_manifest.json ]] || "$py" "$p/build_dolmino_fresh.py" --stage "$fresh/upstream" \
    --oracle "$inv/corpus_hashes.npy" --tokenizer "$base" --out "$fresh/raw"
  [[ -f $raw/manifest.json ]] || ANCHOR_SRC=$fresh/raw/data ANCHOR_OUT=$raw ANCHOR_TOKENIZER=$base \
    ANCHOR_TOKENS=240000000 ANCHOR_WINDOW=1025 ANCHOR_SOURCE_TOKENS=500000000 ANCHOR_SAMPLE_MARGIN=1.4 \
    ANCHOR_SEED=1234 ANCHOR_BATCH_ROWS=2048 ANCHOR_SOURCE_REPO_ID=allenai/dolma3_dolmino_mix-100B-1025 \
    ANCHOR_SOURCE_REVISION=f23942ae8a8114af6e992efe8188ce8c531acd16 "$py" -m unified_training.tools.build_document_anchor
  local bench=$D/decontamination/broad16-gsm8k-test.jsonl
  [[ -s $bench ]] || "$py" "$p/benchmark_samples.py" --out "$bench"
  if [[ ! -f $out/manifest.json ]]; then
    "$py" -m unified_training.tools.audit_benchmark_overlap --tokens "$raw/tokens.bin" --index "$raw/index.npy" \
      --tokenizer "$base" --samples "$bench" --output "$raw/broad16-exact.json" --max-matches 1000
    "$py" -m unified_training.tools.audit_benchmark_overlap --tokens "$raw/tokens.bin" --index "$raw/index.npy" \
      --tokenizer "$base" --samples "$bench" --output "$raw/broad16-16token.json" --max-matches 1000 \
      --min-match-tokens 16 --max-match-tokens 16
    "$py" -m unified_training.tools.sanitize_anchor --source "$raw" --output "$out" --report "$raw/broad16-exact.json" \
      --report "$raw/broad16-16token.json" --suite broad16-gsm8k --split test --seed 1234
  fi
  local sha windows
  sha=$(sha256sum "$out/manifest.json" | cut -d' ' -f1)
  windows=$("$py" -c "import json,sys; print(json.load(open(sys.argv[1]))['n_episodes'])" "$out/manifest.json")
  echo "[prepare] midtrain-240m: $windows windows, manifest sha256 $sha (paper: 733393, 12d89d86cc4689c8...)"
  [[ -f $D/realworld/midtrain-val-holdout/manifest.json ]] || "$py" -m unified_training.tools.build_midtrain_holdout prepare \
    --anchor-dir "$out" --output-dir "$D/realworld/midtrain-val-holdout" --expected-manifest-sha256 "$sha" \
    --expected-anchor-windows "$windows" --expected-window 1025 --max-training-rows 4096000 \
    --max-anchor-row-fraction 0.178926025390625 --permutation-seed 1235 --eval-windows 512
  "$py" "$p/heldout_loss_slices.py" stage --dest "$D/heldout-loss-upstream" \
    --files-from "$p/receipts/heldout_slices_stage_manifest.json"
  [[ -f $D/realworld/heldout-loss-slices/manifest.json ]] || "$py" "$p/heldout_loss_slices.py" build \
    --stage "$D/heldout-loss-upstream" --oracle "$inv/corpus_hashes.npy" --fresh-corpus "$fresh/raw" \
    --tokenizer "$base" --out "$D/realworld/heldout-loss-slices"
}

for stage in "${stages[@]}"; do
  case $stage in
    math12k) hf_get realworld/base; "$py" "$p/math12k.py" --out "$D/math12k" --tokenizer "$base" ;;
    kodcode) "$py" "$p/kodcode.py" --out "$D/kodcode" ;;
    corpora)
      for f in midtrain-240m midtrain-code-removed midtrain-code-replaced midtrain-val-holdout heldout-loss-slices; do
        [[ -f $D/realworld/$f/manifest.json ]] || hf download sfanm/unified-training --repo-type dataset \
          --include "realworld/$f/*" --local-dir "$D"
      done ;;
    rebuild_midtrain) midtrain ;;
    rebuild_code_variants)
      hf_get realworld/base
      export ANCHOR_DIR=$D/realworld/midtrain-240m OUT_DIR=$D/realworld TOKENIZER=$base HOLDOUT_DIR=$D/realworld/midtrain-val-holdout
      bash "$p/code_variants/code_removed.sh"
      bash "$p/code_variants/code_replaced.sh" ;;
    *) echo "unknown stage $stage" >&2; exit 2 ;;
  esac
done
