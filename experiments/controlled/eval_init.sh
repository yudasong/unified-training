#!/usr/bin/env bash
# Evaluations of the initializations before RL (one GPU):   eval_init.sh [INIT ...]   (default: all twelve)
#   puzzles: demonstrated patterns (nltw_m0), Base (nltw_m), Extended (nltw_hb) and
#     Extended generalization (nltw_hb12), 64 samples per question -> EVAL_DIR/<spec>/init_<init>.json
#   retention, with the shared harness unified_training.evals: zero-shot MMLU, ARC-Easy,
#     HellaSwag, TruthfulQA MC2, WinoGrande, and held-out pretraining loss on ~400k tokens of the validation shard 25
#     of each corpus (HOLDOUT_CLIMBMIX / HOLDOUT_DOLMINO: Megatron prefixes from scripts/pretrain/prepare_data.sh
#     climbmix-100b 25 and dolmino-100b 25) weighted 0.75 / 0.25
#     -> EVAL_DIR/retention/<init>/
# PART=puzzles|retention runs one part only: puzzles in the training environment, retention in the benchmark
# environment (README, Install), e.g. PART=retention PYTHON=<benchmark env>/bin/python eval_init.sh
set -euo pipefail
source "$(dirname "$0")/common.sh"
: "${CKPT_DIR:?set CKPT_DIR}"
HOLDOUT_CLIMBMIX=${HOLDOUT_CLIMBMIX:-$DATA_DIR/pretrain/climbmix-100b/shard_000025}
HOLDOUT_DOLMINO=${HOLDOUT_DOLMINO:-$DATA_DIR/pretrain/dolmino-100b/shard_000025}
for init in ${@:-$INITS}; do
  ensure_init "$init"
  model=$(init_dir "$init")
  if [[ ${PART:-all} != retention ]]; then
    for spec in nltw_m0 nltw_m nltw_hb nltw_hb12; do
      out=$EVAL_DIR/$spec/init_$init.json
      [[ -f $out ]] || "$PYTHON" -m nl_tower.evaluate --spec "$spec" --model "$model" \
        --test "$NL_TOWER_WORKDIR/data/$spec/corpus_tw1/test.jsonl" --out "$out" --n 64 --max-tokens 1280
    done
  fi
  if [[ ${PART:-all} != puzzles ]]; then
    d=$EVAL_DIR/retention/$init
    "$PYTHON" -m unified_training.evals retention --model "$model" \
      --tasks mmlu,arc_easy,hellaswag,truthfulqa_mc2,winogrande --output "$d"
    [[ -f $d/pretrain_holdout_loss.json ]] || "$PYTHON" -m unified_training.evals holdout-loss --model "$model" \
      --megatron "climbmix=$HOLDOUT_CLIMBMIX,dolmino=$HOLDOUT_DOLMINO" --weights climbmix=0.75,dolmino=0.25 \
      --tokens-per-corpus 400000 --seed 20260905 --output "$d/pretrain_holdout_loss.json"
  fi
done
