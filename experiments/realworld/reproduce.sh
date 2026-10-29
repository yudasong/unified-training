#!/usr/bin/env bash
# Train, evaluate and tabulate one group of real-world runs (README).
#   bash reproduce.sh GROUP [train|eval|table|all]
#   GROUP: base constant schedules code-data reinforce kl (unified trainer) | staged coding (NeMo-RL)
# Env: DATA_DIR (prepare.sh outputs), CKPT_DIR, RUNS_DIR, EVAL_DIR, PYTHON, GPUS (evaluation GPUs, default 0,1,2,3).
# Unified runs: run_rl.sh <id> (one node, any number of GPUs >= 2). Staged math RL and coding RL: run_nemo_rl.sh.
# coding starts from the five math-RL models in HF sfanm/unified-training (realworld/math-rl/*); with
# MATH_RL_FROM=runs it uses this machine's runs instead (staged step 12000, const0p5/1/2/5 final checkpoints).
# Evaluation: benchmarks and pass@k every 1,000 updates, MMLU and held-out loss every 2,000.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); repo=$(cd "$here/../.." && pwd)
item=${1:?group}; phase=${2:-all}; py=${PYTHON:-python}
: "${DATA_DIR:?}" "${CKPT_DIR:?}" "${RUNS_DIR:?}" "${EVAL_DIR:?}"; GPUS=${GPUS:-0,1,2,3}
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
base=$CKPT_DIR/realworld/base

case $item in
  base|staged|coding) unified=() ;;
  constant) unified=(const0 const0p1 const0p2 const0p5 const1 const2 const5) ;;
  schedules) unified=(fullanneal0p2 fullanneal0p5 fullanneal1 halfanneal1 halfanneal2 halfanneal5 q3anneal1 q3anneal2
                      rampup1) ;;
  code-data) unified=(coderemoved_const0p5 coderemoved_fullanneal0p5 codereplaced_const0p5 codereplaced_fullanneal0p5) ;;
  reinforce) unified=(reinforce_const0p5 reinforce_const1 reinforce_const2 reinforce_const5 reinforce_halfanneal1
                      reinforce_halfanneal2) ;;
  kl) unified=(kl0p001 kl0p01 kl0p1) ;;
  *) echo "unknown group $item" >&2; exit 2 ;;
esac
nemo_math=$([[ $item == staged ]] && echo 1 || echo 0)
coding_starts=(staged unified-0.5pct unified-1pct unified-2pct unified-5pct)

hf_get() { [[ -f $CKPT_DIR/$1/config.json ]] || hf download sfanm/unified-training --include "$1/*" --local-dir "$CKPT_DIR"; }
last_milestone() { ls -d "$RUNS_DIR/$1"/eval-checkpoints/target_* | sort | tail -1; }
nemo_hf() {  # NeMo-RL checkpoint step N of run $1 as an HF dir
  local out=$RUNS_DIR/$1/hf/step_$(printf %07d "$2")
  [[ -f $out/CONVERSION_DONE ]] || "$py" "$here/convert_nemo_rl_checkpoint.py" \
    --source-checkpoint "$RUNS_DIR/$1/checkpoints/step_$2" --optimizer-step "$2" --output "$out"
  echo "$out"
}
coding_start() {  # math-RL model a coding run starts from
  if [[ ${MATH_RL_FROM:-hf} == runs ]]; then
    local p=${1#unified-}; p=${p%pct}; p=${p/./p}
    case $1 in staged) nemo_hf maxrl_math12k 12000 ;; *) last_milestone "const$p" ;; esac
  else hf_get "realworld/math-rl/$1"; echo "$CKPT_DIR/realworld/math-rl/$1"; fi
}
battery() {  # config step model suites
  "$py" -m unified_training.evals battery --model "$3" --output "$EVAL_DIR/$1/step_$(printf %07d "$2")" --suite "$4" \
    --slices "$DATA_DIR/realworld/heldout-loss-slices" --gpus "$GPUS"
}
schedule() {  # config step model: the App D.1 schedule
  local s=eval16,passk; (( $2 % 2000 == 0 )) && s=$s,retention; battery "$1" "$2" "$3" "$s"
}

train() {
  hf_get realworld/base
  for id in ${unified[@]+"${unified[@]}"}; do bash "$here/run_rl.sh" "$id"; done
  if (( nemo_math )); then bash "$here/run_nemo_rl.sh" math12k "$RUNS_DIR/maxrl_math12k" "$base"; fi
  if [[ $item == coding ]]; then
    for s in "${coding_starts[@]}"; do bash "$here/run_nemo_rl.sh" kodcode "$RUNS_DIR/coding_from_$s" "$(coding_start "$s")"; done
  fi
}

evaluate() {
  hf_get realworld/base
  [[ $item == base ]] && battery base 0 "$base" eval16,passk,retention
  if [[ $item == coding ]]; then  # step 0 = the math-RL start; then the coding checkpoints
    for s in "${coding_starts[@]}"; do
      for step in 0 1000 2000 3000 4000; do
        local m; if (( step == 0 )); then m=$(coding_start "$s"); else m=$(nemo_hf "coding_from_$s" "$step"); fi
        battery "coding_from_$s" "$step" "$m" eval16,retention
      done
    done
  fi
  for id in ${unified[@]+"${unified[@]}"}; do
    [[ -d $RUNS_DIR/$id/eval-checkpoints ]] || { echo "not trained yet, skipped: $id"; continue; }
    for m in "$RUNS_DIR/$id"/eval-checkpoints/target_*; do
      if [[ -d $m ]]; then schedule "$id" "$((10#${m##*actual_}))" "$m"; fi
    done
  done
  if (( nemo_math )); then
    for step in $(seq 1000 1000 12000); do schedule maxrl_math12k "$step" "$(nemo_hf maxrl_math12k "$step")"; done
  fi
}

table() { "$py" -m unified_training.evals table "$EVAL_DIR" --runs "$RUNS_DIR" --out "$EVAL_DIR/metrics.csv"; }
case $phase in
  train) train ;; eval) evaluate ;; table) table ;; all) train; evaluate; table ;;
  *) echo "phase: train|eval|table|all" >&2; exit 2 ;;
esac
