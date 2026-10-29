#!/usr/bin/env bash
# Train one unified math-RL run on one node:   bash run_rl.sh ID [UNIFIED_X=value ...]
#   ID: a config in configs/ (const0p5, fullanneal1, ...; README). The run uses every GPU of the node (or those in
#   CUDA_VISIBLE_DEVICES): the first half generate, the rest train. 2 GPUs work; the number of trainer GPUs must
#   divide 256 (512 rows per update, micro-batch 2).
#   The recency window is set to the number of generator GPUs (configs/common.env: the paper's 8 = the newest round
#   of each of its 8 generator GPUs); UNIFIED_RECENCY_WINDOW=N overrides it.
# Env: DATA_DIR (prepare.sh outputs), CKPT_DIR (base model, downloaded on first use), RUNS_DIR, PYTHON.
# The run goes to RUNS_DIR/ID; an interrupted run continues where it stopped. Several nodes: scripts/train_unified.sh.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); repo=$(cd "$here/../.." && pwd)
id=${1:?usage: run_rl.sh ID [UNIFIED_X=value ...]}; shift
[[ -f $here/configs/$id.env && $id != common ]] || { echo "unknown config $id (see configs/)" >&2; exit 2; }
: "${DATA_DIR:?}" "${CKPT_DIR:?}" "${RUNS_DIR:?}"
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
[[ -f $CKPT_DIR/realworld/base/config.json ]] ||
  hf download sfanm/unified-training --include "realworld/base/*" --local-dir "$CKPT_DIR"

if [[ -n ${UNIFIED_GENERATOR_GPU_IDS:-} ]]; then
  IFS=, read -r -a gen <<<"$UNIFIED_GENERATOR_GPU_IDS"; n_gen=${#gen[@]}
else   # scripts/train_unified.sh gives the first half of the GPUs to the generators
  if [[ -n ${CUDA_VISIBLE_DEVICES:-} ]]; then IFS=, read -r -a gpus <<<"$CUDA_VISIBLE_DEVICES"
  else mapfile -t gpus < <(nvidia-smi --query-gpu=index --format=csv,noheader); fi
  n_gen=$(( ${#gpus[@]} / 2 ))
fi
(( n_gen >= 1 )) || { echo "need at least 2 GPUs" >&2; exit 2; }

run=$RUNS_DIR/$id
[[ -f $run/trainer_hf/latest_step.txt ]] && export RESUME=1
RUN_DIR=$run bash "$repo/scripts/train_unified.sh" "$here/configs/common.env" "$here/configs/$id.env" \
  "UNIFIED_RECENCY_WINDOW=$n_gen" "$@"
