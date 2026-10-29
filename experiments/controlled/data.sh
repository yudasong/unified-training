#!/usr/bin/env bash
# Data of the controlled experiments (CPU only):   data.sh [STAGE ...]   (default: all, in order; pretrain-rows is
# skipped when its pretraining shards are missing and no stage was named)
#   tests rl-pools mixes corpus   puzzle test sets, RL prompt pools, Base/Extended prompt mixes, the 796,175-example
#                                 midtraining corpus (nl_tower.prepare_data, ~4 GB; RG_WORKERS forked workers, default
#                                 the available CPUs up to 32: about 45 min with 24-40 workers)
#   demos                         the corpus as the anchor round mixed into RL (needs the GPT-2 tokenizer of the 124M
#                                 models: NL_TOWER_TOKENIZER, default the mixed-133M checkpoint)
#   pretrain-rows                 pretraining-text rounds (group pre-mid-rl) for 8 and 32 examples per update, from the tokenized
#                                 shards built by scripts/pretrain/prepare_data.sh climbmix-400b 103,60,5 and
#                                 prepare_data.sh dolmino-100b 0,1,2 (PRETRAIN_CLIMBMIX_DIR, PRETRAIN_DOLMINO_DIR)
#   verify                        md5 of every rebuilt file against the published artifacts
# The same data is on Hugging Face for browsing (sfanm/nltower-env: tests, pools; sfanm/nltower-rl-mixes: prompt
# mixes; sfanm/nltower-traces-tw1X: corpus); the scripts read the files rebuilt here, which verify checks.
set -euo pipefail
source "$(dirname "$0")/common.sh"
stages=${*:-tests rl-pools mixes corpus demos pretrain-rows verify}
named=${*:+1}
for s in $stages; do
  case $s in
    tests|rl-pools|mixes|corpus|verify) "$PYTHON" -m nl_tower.prepare_data "$s" ;;
    demos) [[ -n ${NL_TOWER_TOKENIZER:-} ]] || ensure_init mixed-133M  # any 124M checkpoint has the tokenizer
           "$PYTHON" -m nl_tower.prepare_data demos --tokenizer "${NL_TOWER_TOKENIZER:-$(init_dir mixed-133M)}" ;;
    pretrain-rows)
      CM=${PRETRAIN_CLIMBMIX_DIR:-$DATA_DIR/pretrain/climbmix-400b}; DO=${PRETRAIN_DOLMINO_DIR:-$DATA_DIR/pretrain/dolmino-100b}
      if [[ ! -f $CM/shard_000103.bin || ! -f $DO/shard_000000.bin ]]; then
        msg="no pretraining shards under $CM and $DO (scripts/pretrain/prepare_data.sh builds them; see the header of data.sh)"
        [[ -n $named ]] && { echo "pretrain-rows: $msg" >&2; exit 1; }
        echo "skipped pretrain-rows (only the pre-mid-rl runs need it): $msg"; continue
      fi
      for m in 8 32; do
        out=$NL_TOWER_WORKDIR/unified_data/anchors/pretrain_tw1X_p10_m$m
        [[ -d $out ]] && { echo "exists, kept: $out"; continue; }
        "$PYTHON" -m unified_training.tools.build_pretrain_rows "$out" --rows 50000 --seed $((2000 + m)) \
          --demo-manifest "$NL_TOWER_WORKDIR/unified_data/anchors/demos_tw1X_resp/manifest.json" --demo-rows-per-step $m \
          --pretrain-rows-per-step 4 --ratio 0.10 \
          --source "climbmix=0.75:$CM/shard_000103.bin,$CM/shard_000060.bin,$CM/shard_000005.bin" \
          --source "dolmino=0.25:$DO/shard_000000.bin,$DO/shard_000001.bin,$DO/shard_000002.bin"
      done ;;
    *) echo "unknown stage $s" >&2; exit 2 ;;
  esac
done
