#!/usr/bin/env bash
# Unified training: vLLM generator processes + a torchrun trainer, sharing RUN_DIR (one node or several).
#
#   RUN_DIR=runs/my_run [DATA_DIR=...] [CKPT_DIR=...] \
#     scripts/train_unified.sh [CONFIG.env ...] [UNIFIED_X=value ...]
#
# CONFIG.env files are bash-sourced KEY=VALUE lines (later files override earlier ones; they may use
# ${DATA_DIR}, ${CKPT_DIR}, ${RUN_DIR}); trailing KEY=VALUE arguments are applied last. Every knob is a
# UNIFIED_* variable (docs/configuration.md). Required: RUN_DIR, UNIFIED_POLICY_HF, UNIFIED_PROMPTS_JSON.
#
# Launcher variables:
#   PYTHON                      interpreter with torch, vllm and unified_training (default: python)
#   RESUME=1                    re-enter an existing RUN_DIR (trainer resumes from trainer_hf/latest_step.txt)
#   UNIFIED_GENERATOR_GPU_IDS   generator GPUs of a node, one vLLM process each (one node: first half of the
#                               GPUs; ROLE=gen: all GPUs)
#   UNIFIED_TRAINER_GPU_IDS     trainer GPUs of a node, one DDP rank each (one node: second half; ROLE=train: all).
#                               The lists may overlap (co-located roles; then lower UNIFIED_GPU_MEM_UTIL, e.g. 0.5)
#   NODE_TMP                    node-local scratch for vLLM/Triton caches (default: ${TMPDIR:-/tmp}/unified-$USER-<run>)
#   GEN_STOP_GRACE              seconds generators get to finish their batch after training (default: 600)
#   ROLE=all|gen|train          all (default) runs both roles here; gen/train run one role of a multi-node run
#
# Stop safely: touch "$RUN_DIR/STOP_TRAINER" (or send SIGTERM to this script). The trainer checkpoints at the
# next step boundary and exits, then generators stop after their current batch. Exit code = trainer's.
#
# Outputs (RUN_DIR): launch.env (resolved config), policies/ (published weights), rounds/ (generations),
# trainer_hf/ (resumable state), eval-checkpoints/target_*/ (HF milestones), logs/.
#
# Multi-node (RUN_DIR on a filesystem every node mounts; separate generator and trainer nodes):
#   HOSTFILE=hosts.txt UNIFIED_GENERATOR_NNODES=G UNIFIED_TRAINER_NNODES=T scripts/train_unified.sh cfg.env
#     The first G hosts of HOSTFILE (one per line, first column) run generators, the next T run trainers (master =
#     the first trainer host); each role is started with REMOTE_SHELL (default "ssh -o BatchMode=yes", called as
#     REMOTE_SHELL HOST COMMAND) and re-runs this script with ROLE=gen / ROLE=train on that node. Roles get only
#     the variables saved in RUN_DIR/launch-resolved.env (UNIFIED_*, RUN_DIR, DATA_DIR, CKPT_DIR, PYTHON, ...); for
#     site settings (compiler, caches, proxy, modules) make REMOTE_SHELL a wrapper script that loads them, e.g.
#       host=$1; shift; exec ssh -o BatchMode=yes "$host" "source /path/to/site_env.sh; $*"
#   Or start the roles yourself (e.g. from a scheduler), once per node:
#     generator node i:  ROLE=gen  GEN_NODE_INDEX=i UNIFIED_GENERATOR_NNODES=G   scripts/train_unified.sh cfg.env
#     trainer node j:    ROLE=train NODE_RANK=j MASTER_ADDR=<trainer node 0> UNIFIED_TRAINER_NNODES=T \
#                        [UNIFIED_TRAINER_MASTER_PORT=29500] scripts/train_unified.sh cfg.env
#   GBS must be divisible by UNIFIED_MBS x (trainer nodes x trainer GPUs per node); set UNIFIED_GEN_WORLD (total
#   generator processes) for the trainer's optional rollout-freshness guard.
set -euo pipefail

die() { echo "train_unified.sh: ERROR: $*" >&2; exit 2; }
log() { echo "[train_unified] $(date -u +%FT%TZ) $*"; }
SCRIPT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")

# ---- configuration: env files, then KEY=VALUE overrides ---------------------------------------------------
overrides=()
for arg in "$@"; do
  if [[ -f "$arg" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$arg"
    set +a
  elif [[ "$arg" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
    overrides+=("$arg")
  else
    die "argument is neither a readable env file nor KEY=VALUE: $arg"
  fi
done
for kv in "${overrides[@]+"${overrides[@]}"}"; do export "${kv?}"; done

: "${RUN_DIR:?set RUN_DIR (output directory of this run)}"
ROLE=${ROLE:-all}
PYTHON=${PYTHON:-python}
RESUME=${RESUME:-0}
GEN_STOP_GRACE=${GEN_STOP_GRACE:-600}
HOSTFILE=${HOSTFILE:-}
LAUNCH_CHILD=${LAUNCH_CHILD:-0}   # set on the roles a HOSTFILE launch starts
NODE_TMP_GIVEN=${NODE_TMP:-}
case $ROLE in all|gen|train) ;; *) die "ROLE must be all, gen or train (got $ROLE)";; esac
CLUSTER=0
[[ $ROLE == all && -n "$HOSTFILE" ]] && CLUSTER=1
: "${UNIFIED_POLICY_HF:?set UNIFIED_POLICY_HF (initial HF checkpoint dir)}"
: "${UNIFIED_PROMPTS_JSON:?set UNIFIED_PROMPTS_JSON (RL prompts: JSON list of {question, gold})}"
[[ -r "$UNIFIED_POLICY_HF/config.json" ]] || die "UNIFIED_POLICY_HF has no config.json: $UNIFIED_POLICY_HF"
[[ -r "$UNIFIED_PROMPTS_JSON" ]] || die "UNIFIED_PROMPTS_JSON is not readable: $UNIFIED_PROMPTS_JSON"

mkdir -p "$RUN_DIR"
RUN_DIR=$(cd "$RUN_DIR" && pwd)
export RUN_DIR UNIFIED_RUN_DIR="$RUN_DIR"
export UNIFIED_EPISODES_DIRS=${UNIFIED_EPISODES_DIRS:-$RUN_DIR/rounds}
export UNIFIED_STREAMING=${UNIFIED_STREAMING:-1}
export UNIFIED_TRAIN_STOP_FILE=${UNIFIED_TRAIN_STOP_FILE:-$RUN_DIR/STOP_TRAINER}
export UNIFIED_GENERATOR_NNODES=${UNIFIED_GENERATOR_NNODES:-1}
export UNIFIED_TRAINER_NNODES=${UNIFIED_TRAINER_NNODES:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1} TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
export WANDB_MODE=${WANDB_MODE:-disabled}
GEN_NODE_INDEX=${GEN_NODE_INDEX:-0}
NODE_RANK=${NODE_RANK:-0}

# ---- GPU layout of this node (the HOSTFILE head runs no role itself) ---------------------------------------
join() { local IFS=,; echo "$*"; }
if ((CLUSTER == 0)); then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=, read -r -a visible <<<"$CUDA_VISIBLE_DEVICES"
  else
    mapfile -t visible < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || true)
  fi
  n_visible=${#visible[@]}
  if [[ $ROLE == all && ( -z "${UNIFIED_GENERATOR_GPU_IDS:-}" || -z "${UNIFIED_TRAINER_GPU_IDS:-}" ) ]]; then
    ((n_visible >= 2)) || die "need at least 2 GPUs (found $n_visible); set UNIFIED_GENERATOR_GPU_IDS and UNIFIED_TRAINER_GPU_IDS"
    half=$((n_visible / 2))
    UNIFIED_GENERATOR_GPU_IDS=${UNIFIED_GENERATOR_GPU_IDS:-$(join "${visible[@]:0:half}")}
    UNIFIED_TRAINER_GPU_IDS=${UNIFIED_TRAINER_GPU_IDS:-$(join "${visible[@]:half}")}
  fi
  if [[ $ROLE == gen && -z "${UNIFIED_GENERATOR_GPU_IDS:-}" ]]; then UNIFIED_GENERATOR_GPU_IDS=$(join "${visible[@]}"); fi
  if [[ $ROLE == train && -z "${UNIFIED_TRAINER_GPU_IDS:-}" ]]; then UNIFIED_TRAINER_GPU_IDS=$(join "${visible[@]}"); fi
  [[ -z "${UNIFIED_GENERATOR_GPU_IDS:-}" ]] || export UNIFIED_GENERATOR_GPU_IDS
  [[ -z "${UNIFIED_TRAINER_GPU_IDS:-}" ]] || export UNIFIED_TRAINER_GPU_IDS
  [[ $ROLE == train ]] || : "${UNIFIED_GENERATOR_GPU_IDS:?no generator GPUs found; set UNIFIED_GENERATOR_GPU_IDS}"
  [[ $ROLE == gen ]] || : "${UNIFIED_TRAINER_GPU_IDS:?no trainer GPUs found; set UNIFIED_TRAINER_GPU_IDS}"
  IFS=, read -r -a gen_gpus <<<"${UNIFIED_GENERATOR_GPU_IDS:-}"
  IFS=, read -r -a train_gpus <<<"${UNIFIED_TRAINER_GPU_IDS:-}"
  # every GPU a role uses must be on this node (checked when the visible GPUs are numeric ids, not UUIDs)
  if ((n_visible > 0)) && [[ ",$(join "${visible[@]}")," =~ ^(,[0-9]+)+,$ ]]; then
    used=()
    [[ $ROLE == train ]] || used+=(${gen_gpus[@]+"${gen_gpus[@]}"})
    [[ $ROLE == gen ]] || used+=(${train_gpus[@]+"${train_gpus[@]}"})
    for g in ${used[@]+"${used[@]}"}; do
      [[ ",$(join "${visible[@]}")," == *",$g,"* ]] || die "GPU $g (UNIFIED_GENERATOR_GPU_IDS / UNIFIED_TRAINER_GPU_IDS)" \
        "is not on this node, whose GPUs are $(join "${visible[@]}") (controlled runs on 1-2 GPUs: LAYOUT=colocated1 / colocated2)"
    done
  fi
  if [[ $ROLE != train ]]; then
    export UNIFIED_GEN_WORLD=$((UNIFIED_GENERATOR_NNODES * ${#gen_gpus[@]}))
  fi
  if [[ $ROLE != gen ]]; then
    export UNIFIED_TRAINER_NPROC_PER_NODE=${#train_gpus[@]}
    trainer_world=$((UNIFIED_TRAINER_NNODES * UNIFIED_TRAINER_NPROC_PER_NODE))
    gbs=${UNIFIED_GBS:-256} mbs=${UNIFIED_MBS:-8}
    ((gbs % (mbs * trainer_world) == 0)) || die "UNIFIED_GBS=$gbs is not divisible by UNIFIED_MBS=$mbs x $trainer_world trainer ranks"
  fi
  if [[ $ROLE == all ]]; then
    # Overlapping lists co-locate both roles on the same GPUs (the reasoning-puzzle runs: both on GPUs 0-3);
    # UNIFIED_GPU_MEM_UTIL (vLLM's share of each GPU) must then leave room for the trainer.
    for g in "${gen_gpus[@]}"; do
      if [[ ",${UNIFIED_TRAINER_GPU_IDS}," == *",$g,"* ]]; then
        log "GPU $g is shared by a generator and a trainer rank (co-located; UNIFIED_GPU_MEM_UTIL=${UNIFIED_GPU_MEM_UTIL:-0.6})"
      fi
    done
  fi
  NODE_TMP=${NODE_TMP:-${TMPDIR:-/tmp}/unified-${USER:-user}-$(basename "$RUN_DIR")}
  mkdir -p "$NODE_TMP"
fi
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/rounds" "$RUN_DIR/policies"

# ---- run directory: refuse to clobber, seed the cold-start policy, clear stale stop markers ----------------
OWNER=0   # the process that owns the launch: a one-node run, a HOSTFILE head, or trainer node 0 started by hand
if [[ $LAUNCH_CHILD != 1 && ( $ROLE == all || ( $ROLE == train && $NODE_RANK == 0 ) ) ]]; then OWNER=1; fi
if [[ $OWNER == 1 && $RESUME != 1 ]]; then
  if [[ -e "$RUN_DIR/trainer_hf/latest_step.txt" ]] || compgen -G "$RUN_DIR/rounds/*" >/dev/null; then
    die "$RUN_DIR already holds a run; set RESUME=1 to continue it or choose a new RUN_DIR"
  fi
fi
if [[ ! -f "$RUN_DIR/policies/v0000000/DONE" ]]; then
  seed="$RUN_DIR/policies/.v0000000.tmp-$$-$(hostname)"
  mkdir -p "$seed"
  for f in "$UNIFIED_POLICY_HF"/*; do ln -s "$(cd "$(dirname "$f")" && pwd)/$(basename "$f")" "$seed/"; done
  touch "$seed/DONE"
  mv -T "$seed" "$RUN_DIR/policies/v0000000" 2>/dev/null || rm -rf "$seed"   # another node won the race
  log "seeded policies/v0000000 from $UNIFIED_POLICY_HF"
fi
if [[ $OWNER == 1 ]]; then
  rm -f "$RUN_DIR/STOP_GENERATOR" "$UNIFIED_TRAIN_STOP_FILE"
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  {
    echo "# train_unified.sh role=$ROLE resume=$RESUME hostfile=${HOSTFILE:-none} $stamp"
    env | grep -E '^(UNIFIED_[A-Z0-9_]*|DATA_DIR|CKPT_DIR|RUN_DIR)=' | sort
  } >"$RUN_DIR/launch-$stamp.env"
  cp "$RUN_DIR/launch-$stamp.env" "$RUN_DIR/launch.env"
fi

first_writer() {   # first_writer FILE VALUE: an existing marker wins
  local tmp="$1.tmp-$$"
  printf '%s\n' "$2" >"$tmp"
  ln "$tmp" "$1" 2>/dev/null || true
  rm -f "$tmp"
}

# ---- roles --------------------------------------------------------------------------------------------------
gen_pids=()
start_generators() {
  local i gpu rank
  for i in "${!gen_gpus[@]}"; do
    gpu=${gen_gpus[i]}
    rank=$((GEN_NODE_INDEX * ${#gen_gpus[@]} + i))
    mkdir -p "$NODE_TMP/vllm-r$rank" "$NODE_TMP/triton-r$rank"
    CUDA_VISIBLE_DEVICES=$gpu UNIFIED_GEN_RANK=$rank UNIFIED_GEN_TAG=$(printf 'r%02d' "$rank") \
      VLLM_CACHE_ROOT="$NODE_TMP/vllm-r$rank" TRITON_CACHE_DIR="$NODE_TMP/triton-r$rank" \
      VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_SKIP_P2P_CHECK=1 VLLM_NO_USAGE_STATS=1 \
      OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      "$PYTHON" -m unified_training.gen_service >>"$RUN_DIR/logs/generator-rank$rank.log" 2>&1 &
    gen_pids+=($!)
    log "generator rank $rank on GPU $gpu (pid $!) -> logs/generator-rank$rank.log"
  done
}

trainer_pid=
start_trainer() {
  local rdzv=(--standalone)
  if ((UNIFIED_TRAINER_NNODES > 1)); then
    : "${MASTER_ADDR:?ROLE=train with UNIFIED_TRAINER_NNODES>1 needs MASTER_ADDR}"
    rdzv=(--node-rank="$NODE_RANK" --master-addr="$MASTER_ADDR" --master-port="${UNIFIED_TRAINER_MASTER_PORT:-29500}")
  fi
  CUDA_VISIBLE_DEVICES=$UNIFIED_TRAINER_GPU_IDS OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    TRITON_CACHE_DIR="$NODE_TMP/triton-trainer" \
    "$PYTHON" -m torch.distributed.run --nnodes="$UNIFIED_TRAINER_NNODES" \
    --nproc-per-node="$UNIFIED_TRAINER_NPROC_PER_NODE" "${rdzv[@]}" \
    -m unified_training.hf_trainer >>"$RUN_DIR/logs/trainer$NODE_RANK.log" 2>&1 &
  trainer_pid=$!
  log "trainer node $NODE_RANK on GPUs $UNIFIED_TRAINER_GPU_IDS (pid $trainer_pid) -> logs/trainer$NODE_RANK.log"
}

wait_or_kill() {   # wait_or_kill PID...: give processes GEN_STOP_GRACE seconds, then terminate leftovers
  local deadline=$((SECONDS + GEN_STOP_GRACE)) pid alive
  while ((SECONDS < deadline)); do
    alive=0
    for pid in "$@"; do kill -0 "$pid" 2>/dev/null && alive=1; done
    ((alive == 0)) && break
    sleep 2
  done
  for pid in "$@"; do kill -TERM "$pid" 2>/dev/null || true; done
  for pid in "$@"; do wait "$pid" 2>/dev/null || true; done
}

on_signal() {   # checkpoint-safe: the trainer polls the stop file every step, generators between batches
  log "signal received: requesting a checkpoint-safe stop"
  if [[ $ROLE == gen ]]; then first_writer "$RUN_DIR/STOP_GENERATOR" 143; else first_writer "$UNIFIED_TRAIN_STOP_FILE" stop; fi
}
trap on_signal TERM INT

# watch_roles TRAINER_PID GEN_PIDS...: while the trainer runs, a generator that exits requests a checkpoint-safe stop
watch_roles() {
  local trainer=$1 i grc
  shift
  local gens=("$@")
  while kill -0 "$trainer" 2>/dev/null; do
    for i in "${!gens[@]}"; do
      if ! kill -0 "${gens[i]}" 2>/dev/null; then
        grc=0; wait "${gens[i]}" || grc=$?
        unset 'gens[i]'
        log "generator $i exited (rc=$grc) during training; requesting a checkpoint-safe trainer stop"
        first_writer "$UNIFIED_TRAIN_STOP_FILE" "generator-exit-$grc"
      fi
    done
    sleep 2
  done
}

rc=0
if ((CLUSTER == 1)); then
  # ---- HOSTFILE head: start every role on its node and wait for trainer node 0 -----------------------------
  mapfile -t hosts < <(grep -v '^[[:space:]]*#' "$HOSTFILE" | awk 'NF {print $1}')
  G=$UNIFIED_GENERATOR_NNODES T=$UNIFIED_TRAINER_NNODES
  ((${#hosts[@]} >= G + T)) || die "HOSTFILE lists ${#hosts[@]} hosts; need $G generator + $T trainer hosts"
  read -r -a remote <<<"${REMOTE_SHELL:-ssh -o BatchMode=yes}"
  resolved="$RUN_DIR/launch-resolved.env"
  {
    echo "# resolved by train_unified.sh for the roles of a HOSTFILE launch"
    for k in $(compgen -e | grep -E '^(UNIFIED_[A-Z0-9_]*|RUN_DIR|DATA_DIR|CKPT_DIR|PYTHON|GEN_STOP_GRACE|HF_HUB_OFFLINE|TRANSFORMERS_OFFLINE|HF_DATASETS_OFFLINE|PYTORCH_ALLOC_CONF|WANDB_MODE)$'); do
      printf '%s=%q\n' "$k" "${!k}"
    done
    [[ -z "$NODE_TMP_GIVEN" ]] || printf 'NODE_TMP=%q\n' "$NODE_TMP_GIVEN"
    printf 'PYTHON=%q\nGEN_STOP_GRACE=%q\n' "$PYTHON" "$GEN_STOP_GRACE"
  } >"$resolved"
  start_role() {   # start_role HOST LOGNAME VAR=VALUE...
    local host=$1 name=$2 cmd
    shift 2
    cmd=$(printf '%q ' env "$@" LAUNCH_CHILD=1 bash "$SCRIPT" "$resolved")
    "${remote[@]}" "$host" "$cmd" >>"$RUN_DIR/logs/$name-$host.log" 2>&1 &
    log "started $* on $host (pid $!) -> logs/$name-$host.log"
  }
  gen_sessions=() train_sessions=()
  for ((i = 0; i < G; i++)); do
    start_role "${hosts[i]}" "node-gen$i" ROLE=gen GEN_NODE_INDEX="$i"
    gen_sessions+=($!)
  done
  master=${hosts[G]}
  for ((j = 0; j < T; j++)); do
    start_role "${hosts[G + j]}" "node-train$j" ROLE=train NODE_RANK="$j" MASTER_ADDR="$master"
    train_sessions+=($!)
  done
  watch_roles "${train_sessions[0]}" "${gen_sessions[@]}"
  wait "${train_sessions[0]}" || rc=$?
  for pid in "${train_sessions[@]:1}"; do wait "$pid" || true; done
  first_writer "$RUN_DIR/STOP_GENERATOR" "$rc"
  log "trainer node 0 exited rc=$rc; stopping generators"
  wait_or_kill "${gen_sessions[@]}"
else
  case $ROLE in
    gen)
      start_generators
      for pid in "${gen_pids[@]}"; do wait "$pid" || rc=$?; done
      ;;
    train)
      start_trainer
      while kill -0 "$trainer_pid" 2>/dev/null; do sleep 2; done
      wait "$trainer_pid" || rc=$?
      if [[ $NODE_RANK == 0 ]]; then first_writer "$RUN_DIR/STOP_GENERATOR" "$rc"; fi
      ;;
    all)
      start_generators
      start_trainer
      watch_roles "$trainer_pid" "${gen_pids[@]}"
      wait "$trainer_pid" || rc=$?
      first_writer "$RUN_DIR/STOP_GENERATOR" "$rc"
      log "trainer exited rc=$rc; stopping generators"
      wait_or_kill "${gen_pids[@]}"
      ;;
  esac
fi
log "done role=$ROLE rc=$rc run=$RUN_DIR"
exit "$rc"
