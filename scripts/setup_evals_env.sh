#!/usr/bin/env bash
# Benchmark-harness environment (Python 3.13, vLLM 0.17.1, lm-eval 0.4.9) for unified_training.evals and the real-world
# data preparation:   bash scripts/setup_evals_env.sh [VENV_DIR]   (default: .venv-evals in the repository)
# lm-eval 0.4.9, the version behind the paper's numbers, predates vLLM 0.17.1; the bundled patch adapts its vLLM backend.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
venv=${1:-$repo/.venv-evals}
patch=src/unified_training/evals/patches/lm_eval_vllm.patch

uv venv --allow-existing --python 3.13 "$venv"
VIRTUAL_ENV=$venv uv pip install -r "$repo/requirements-evals.txt"
VIRTUAL_ENV=$venv uv pip install --no-deps -e "$repo"
site=$("$venv/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
if git -C "$repo" apply -p1 --unsafe-paths --directory="$site" --reverse --check "$patch" 2>/dev/null; then
  echo "lm-eval is already patched"
else
  git -C "$repo" apply -p1 --unsafe-paths --directory="$site" "$patch"
fi
"$venv/bin/python" -c "import lm_eval.models.vllm_causallms"
echo "benchmark environment ready: $venv"
