# Sourced by the scripts in experiments/controlled. Paths come from the environment:
#   DATA_DIR   data root (nl_tower data in $DATA_DIR/nl_tower, tokenized pretraining shards in $DATA_DIR/pretrain)
#   CKPT_DIR   models before RL in CKPT_DIR/controlled/<init>; missing ones are downloaded from the public HF repo
#              (HF_REPO, default sfanm/unified-training; subfolders and sha256 in the repo card)
#   RUNS_DIR   RL run directories (default runs/controlled)      EVAL_DIR  evaluation JSONs (default outputs/controlled/eval)
#   PYTHON     interpreter (default python)
X=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$X/../.." && pwd)
export NL_TOWER_WORKDIR=${NL_TOWER_WORKDIR:-${DATA_DIR:-$REPO/data}/nl_tower}
export RUNS_DIR=${RUNS_DIR:-$REPO/runs/controlled}
export EVAL_DIR=${EVAL_DIR:-$REPO/outputs/controlled/eval}
export PYTHONPATH=$REPO/src${PYTHONPATH:+:$PYTHONPATH}
export PYTHON=${PYTHON:-python}
# initializations: pretraining only; unified (midtraining tokens mixed into pretraining); staged (epochs, peak lr)
INITS="pretrain-only mixed-133M mixed-267M mixed-534M mixed-1.07B mixed-2.14B staged-1ep-lr2e-4 staged-1ep-lr5e-5
staged-2ep-lr2e-4 staged-2ep-lr5e-5 staged-4ep-lr2e-4 staged-4ep-lr5e-5"

# runs.tsv ids of group $1 (unified, staged, no-mix, separate-loss, pre-mid-rl or all), optionally only
# setting $2 (base or extended)
ids_for() {
  awk -F'\t' -v g="$1" -v s="${2:-}" 'NR == 1 {for (i = 1; i <= NF; i++) col[$i] = i; next}
    (g == "all" || $col["group"] == g) && (s == "" || $col["setting"] == s) {print $1}' "$X/runs.tsv"
}

# HF directory of an init id; downloads it from the public repo when missing
init_dir() { echo "${CKPT_DIR:?set CKPT_DIR}/controlled/$1"; }
ensure_init() {
  local d; d=$(init_dir "$1")
  [[ -f $d/config.json ]] && return 0
  echo "downloading controlled/$1 from ${HF_REPO:-sfanm/unified-training} into $CKPT_DIR"
  hf download "${HF_REPO:-sfanm/unified-training}" --include "controlled/$1/*" --local-dir "$CKPT_DIR"
}
