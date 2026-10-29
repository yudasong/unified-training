#!/usr/bin/env bash
# Fetch NeMo-RL at the pinned commit and apply our patch (idempotent).
#   bash third_party/nemo-rl/setup.sh            -> checkout in third_party/nemo-rl/src
#   NEMO_RL_DIR=/path bash third_party/nemo-rl/setup.sh
# scripts/setup_nemo_rl_env.sh runs this and builds the environment (requirements-nemo-rl.txt). Afterwards put the
# checkout on PYTHONPATH (export PYTHONPATH=$NEMO_RL_DIR:$PYTHONPATH) and run with NEMO_RL_PY_EXECUTABLES_SYSTEM=1
# (all Ray actors use that one interpreter, as in the paper's runs); experiments/realworld/run_nemo_rl.sh does both.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
commit="$(awk '/^Commit:/ {print $2; exit}' "$here/UPSTREAM")"
url="$(awk '/^Upstream:/ {print $2; exit}' "$here/UPSTREAM")"
dir="${NEMO_RL_DIR:-$here/src}"

if [[ ! -d "$dir/.git" ]]; then
  mkdir -p "$dir"
  git -C "$dir" init -q
  git -C "$dir" fetch -q --depth 1 "$url" "$commit"
  git -C "$dir" checkout -q FETCH_HEAD
fi
[[ "$(git -C "$dir" rev-parse HEAD)" == "$commit" ]] || { echo "ERROR: $dir is not at $commit" >&2; exit 1; }

if git -C "$dir" apply --reverse --check "$here/nemo_rl.patch" 2>/dev/null; then
  echo "[nemo-rl] patch already applied in $dir"
else
  git -C "$dir" apply "$here/nemo_rl.patch"
  echo "[nemo-rl] patched $dir at ${commit:0:8}"
fi
echo "export PYTHONPATH=$dir:\${PYTHONPATH:-}"
