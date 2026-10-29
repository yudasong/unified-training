#!/usr/bin/env bash
# MaxRL (with baseline subtraction) runs that use NeMo-RL:
#   bash run_nemo_rl.sh math12k OUT [INIT]   Math12K from the base model (staged math RL; its step-12000
#                                            checkpoint starts a coding run), 12,000 updates
#   bash run_nemo_rl.sh kodcode OUT INIT     coding RL on KodCode from a math-RL model, 4,000 updates
# Needs: PYTHON = the environment of scripts/setup_nemo_rl_env.sh (which also fetches and patches NeMo-RL into
# NEMO_RL_DIR), CKPT_DIR (base model), DATA_DIR (prepare.sh outputs). Paper layout: NODES=4, GPUS_PER_NODE=4.
# Start Ray first, as the paper's runs did, also on one node (without a cluster NeMo-RL starts a local Ray with one
# worker per CPU, which can exceed a job's process limit). On each node, with the ray of PYTHON's environment:
#   VLLM_HOST_IP=<node address> ray start --node-ip-address=<node address> --num-cpus=28 --num-gpus=$GPUS_PER_NODE \
#     --disable-usage-stats  --head --port=6379 --include-dashboard=false   (other nodes: --address=<head address>:6379)
# (VLLM_HOST_IP keeps vLLM off 0.0.0.0 on nodes without outbound internet; set it only for ray start, since NeMo-RL
# copies this script's environment to every worker.) Then on the head node: RAY_ADDRESS=<head address>:6379 bash
# run_nemo_rl.sh ... Extra Hydra overrides may follow INIT. DRY_RUN=1 prints the command only.
# Checkpoints: OUT/checkpoints/step_N; convert_nemo_rl_checkpoint.py turns one into an HF directory.
set -euo pipefail
recipe=${1:?math12k|kodcode}; out=${2:?output dir}; init=${3:-${CKPT_DIR:?}/realworld/base}
shift $(( $# < 3 ? $# : 3 ))
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); repo=$(cd "$here/../.." && pwd); cfg=$here/nemo_rl
NEMO_RL_DIR=${NEMO_RL_DIR:-$repo/third_party/nemo-rl/src}
NODES=${NODES:-4}; GPUS_PER_NODE=${GPUS_PER_NODE:-4}

common=(grpo.adv_estimator.name=maxrl loss_fn.reference_policy_kl_penalty=0.0 loss_fn.disable_ppo_ratio=false
        grpo.max_num_epochs=100 "cluster.gpus_per_node=$GPUS_PER_NODE" "cluster.num_nodes=$NODES"
        "policy.generation.colocated.resources.gpus_per_node=$GPUS_PER_NODE"
        "policy.generation.colocated.resources.num_nodes=$NODES"
        checkpointing.keep_top_k=null checkpointing.metric_name=null grpo.val_period=100
        policy.sequence_packing.enabled=false)
case $recipe in
  math12k)
    config=$cfg/maxrl_math12k.yaml
    export RL_TRAIN_FILE=${RL_TRAIN_FILE:-${DATA_DIR:?}/math12k/math12k_train.parquet}
    export RL_VAL_FILE=${RL_VAL_FILE:-$DATA_DIR/math12k/math500_test.jsonl}
    extra=(grpo.max_num_steps=12000 checkpointing.save_period=1000 grpo.max_val_samples=500 grpo.val_batch_size=100
           "data.default.prompt_file=$cfg/math_verl_boxed.txt" env.math.math_verify_impl=verl_math_reward) ;;
  kodcode)
    config=$cfg/maxrl_kodcode.yaml
    export RL_TRAIN_FILE=${RL_TRAIN_FILE:-${DATA_DIR:?}/kodcode/kodcode20k_hard_v2_train.parquet}
    export RL_VAL_FILE=${RL_VAL_FILE:-$DATA_DIR/kodcode/kodcode20k_hard_v2_holdout.jsonl}
    # sandboxed pytest reward (unified_training.code_reward): a directory with pytest importable by the sandbox
    export UNIFIED_CODE_TEST_PYTHONPATH=${UNIFIED_CODE_TEST_PYTHONPATH:?dir containing pytest for the sandbox}
    export UNIFIED_CODE_TEST_WORKERS=${UNIFIED_CODE_TEST_WORKERS:-8} UNIFIED_CODE_TEST_TIMEOUT=${UNIFIED_CODE_TEST_TIMEOUT:-10}
    export UNIFIED_CODE_TEST_TMPDIR=${UNIFIED_CODE_TEST_TMPDIR:-/tmp}
    extra=(grpo.max_num_steps=4000 checkpointing.save_period=1000 grpo.max_val_samples=512 grpo.val_batch_size=64) ;;
  *) echo "unknown recipe $recipe" >&2; exit 2 ;;
esac
export RL_POLICY_MODEL=$init RL_OUTPUT_DIR=$out
cmd=("${PYTHON:-python}" "$NEMO_RL_DIR/examples/run_grpo.py" --config "$config" "${common[@]}" "${extra[@]}" "$@")
if [[ ${DRY_RUN:-0} == 1 ]]; then printf '%s\n' "${cmd[@]}"; exit 0; fi

mkdir -p "$out/logs"
export PYTHONPATH="$NEMO_RL_DIR:$repo/src${PYTHONPATH:+:$PYTHONPATH}"
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1 RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
"${cmd[@]}"
