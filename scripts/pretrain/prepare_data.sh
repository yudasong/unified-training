#!/bin/bash
# prepare_data.sh CORPUS [SHARDS]          CORPUS = a plan in plans/ (climbmix-100b, dolmino-100b, climbmix-400b)
# Tokenize a pretraining corpus into the paper's GPT-2 Megatron shards (64 or 256; CPU only; one process per call, so
# split SHARDS, e.g. "0,1,2", over several calls/nodes). Each receipt says whether the shard is byte-identical
# to the one used in the paper (needs the xxhash package).
# Source data (Hugging Face datasets; licences in NOTICE): climbmix-* = karpathy/climbmix-400b-shuffle (CC BY-NC 4.0),
# dolmino-* = allenai/dolma3_dolmino_mix-100B-1025 (ODC-By). Download the files the plan names:
#   python -m unified_training.pretrain.tokenize_corpus plans/CORPUS.json.gz --list-files [--shards SHARDS] > files.txt
#   hf download --repo-type dataset <repo> --local-dir $SOURCE_ROOT $(cat files.txt)
# Env: SOURCE_ROOT (local copy of the source dataset), OUT_DIR (default $DATA_DIR/pretrain/CORPUS).
# Used by experiments/controlled: the pre+mid+RL runs (App C.8) reads climbmix-400b shards 103,60,5 and
# dolmino-100b shards 0,1,2; the held-out pretraining loss (App C.4) reads climbmix-100b and dolmino-100b shard 25.
set -euo pipefail
TOP=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$(cd "$TOP/../.." && pwd)/src:${PYTHONPATH:-}"
CORPUS=${1:?usage: prepare_data.sh CORPUS [SHARDS]}
OUT=${OUT_DIR:-${DATA_DIR:?set DATA_DIR or OUT_DIR}/pretrain/$CORPUS}
args=("$TOP/plans/$CORPUS.json.gz" "${SOURCE_ROOT:?set SOURCE_ROOT}" "$OUT")
[[ -n ${2:-} ]] && args+=(--shards "$2")
nice -n 19 "${PYTHON:-python}" -m unified_training.pretrain.tokenize_corpus "${args[@]}"
