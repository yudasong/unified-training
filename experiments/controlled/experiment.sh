#!/usr/bin/env bash
# The RL runs of one group of runs.tsv:   experiment.sh GROUP [base|extended] list|train|eval
#   GROUP  unified          unified starting model, midtraining data in RL, one shared loss
#          staged           staged starting model, RL only
#          no-mix           unified starting model, RL only
#          separate-loss    as unified, with separately normalized RL and midtraining losses
#          pre-mid-rl       as unified, plus pretraining text in every RL update (pre+mid+RL)
#          all
#   list   print the run ids (one RL run per 4-GPU node, e.g. as a job array over this list)
#   train  train them here one after another (run_rl.sh; finished runs are skipped, interrupted ones resume)
#   eval   score every milestone checkpoint (eval_run.sh curve)
# Then: python metrics.py --eval-dir "$EVAL_DIR" --runs-dir "$RUNS_DIR" --out OUT
set -euo pipefail
source "$(dirname "$0")/common.sh"
usage="usage: experiment.sh GROUP [base|extended] list|train|eval"
group=${1:?$usage}; shift
setting=""; if [[ ${1:-} == base || ${1:-} == extended ]]; then setting=$1; shift; fi
what=${1:?$usage}
ids=$(ids_for "$group" "$setting")
[[ -n $ids ]] || { echo "no runs in group $group ${setting:+(setting $setting)}" >&2; exit 2; }
case $what in
  list) echo "$ids" ;;
  train) for id in $ids; do bash "$X/run_rl.sh" "$id" || echo "run $id failed or stopped" >&2; done ;;
  eval) for id in $ids; do bash "$X/eval_run.sh" "$id" curve; done ;;
  *) echo "$usage" >&2; exit 2 ;;
esac
