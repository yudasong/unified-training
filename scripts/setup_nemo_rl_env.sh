#!/usr/bin/env bash
# NeMo-RL environment (Python 3.13, torch 2.10.0, vLLM 0.17.1, ray 2.57.0) for the staged math RL and the coding RL of
# the real-world setting:   bash scripts/setup_nemo_rl_env.sh [VENV_DIR]   (default: .venv-nemo-rl in the repository)
# Installs requirements-nemo-rl.txt, fetches NeMo-RL at the pinned commit and applies our patch
# (third_party/nemo-rl/setup.sh; NEMO_RL_DIR moves the checkout). Then: PYTHON=VENV_DIR/bin/python bash run_nemo_rl.sh.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
venv=${1:-$repo/.venv-nemo-rl}
nemo=${NEMO_RL_DIR:-$repo/third_party/nemo-rl/src}

uv venv --allow-existing --python 3.13 "$venv"
VIRTUAL_ENV=$venv uv pip install -r "$repo/requirements-nemo-rl.txt"
NEMO_RL_DIR=$nemo bash "$repo/third_party/nemo-rl/setup.sh"
PYTHONPATH="$nemo:$repo/src" "$venv/bin/python" -c "
import nemo_rl.algorithms.grpo, nemo_rl.environments.math_environment, nemo_rl.models.generation.vllm.vllm_worker
import nemo_rl.models.policy.workers.dtensor_policy_worker, unified_training.code_reward"
echo "NeMo-RL environment ready: $venv (NeMo-RL checkout: $nemo)"
