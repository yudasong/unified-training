#!/usr/bin/env bash
# Train one RL run of runs.tsv on one 4-GPU node (LAYOUT=colocated1 / colocated2: 1 or 2 GPUs; LAYOUT=split8: 8 GPUs):
#   run_rl.sh RUN_ID [UNIFIED_X=value ...]
# Configs: common + setting (base/extended) + init + midtraining mix (m) + loss recipe + seed + GPU layout.
# PRINT=1 prints the config files instead of training. An interrupted run continues where it stopped.
set -euo pipefail
source "$(dirname "$0")/common.sh"
id=${1:?usage: run_rl.sh RUN_ID [UNIFIED_X=value ...]}; shift
row=$(awk -F'\t' -v id="$id" 'NR > 1 && $1 == id' "$X/runs.tsv")
[[ -n $row ]] || { echo "unknown run id $id (see runs.tsv)" >&2; exit 2; }
IFS=$'\t' read -r _ setting recipe init m seed steps _ <<< "$row"
C=$X/configs
cfgs=("$C/common.env" "$C/$setting.env" "$C/init/$init.env" "$C/mix/m$m.env" "$C/recipe/$recipe.env" "$C/seed/s$seed.env"
      "$C/layout/${LAYOUT:-colocated4}.env")
if [[ ${PRINT:-0} == 1 ]]; then printf '%s\n' "${cfgs[@]}"; exit 0; fi
run=$RUNS_DIR/$id
final=$(printf '%s/eval-checkpoints/target_%07d-actual_%07d' "$run" "$steps" "$steps")
[[ -d $final ]] && { echo "done: $id"; exit 0; }
[[ -f $run/trainer_hf/latest_step.txt ]] && export RESUME=1
ensure_init "$init"
RUN_DIR=$run bash "$REPO/scripts/train_unified.sh" "${cfgs[@]}" "$@"
