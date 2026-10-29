#!/usr/bin/env bash
# Score the milestone checkpoints of one RL run (1,024 questions x 64 samples, one GPU):
#   eval_run.sh RUN_ID [final|curve]
# final: the last update only. curve: every 500 updates on the primary test (Base: nltw_m; Extended: nltw_hb) and
# every 1000 on Extended generalization (nltw_hb12), as in the paper's curves. Existing results are kept.
set -euo pipefail
source "$(dirname "$0")/common.sh"
id=${1:?usage: eval_run.sh RUN_ID [final|curve]}; mode=${2:-final}
row=$(awk -F'\t' -v id="$id" 'NR > 1 && $1 == id' "$X/runs.tsv")
[[ -n $row ]] || { echo "unknown run id $id (see runs.tsv)" >&2; exit 2; }
IFS=$'\t' read -r _ setting _ _ _ _ steps _ <<< "$row"
if [[ $setting == base ]]; then tests="nltw_m:500"; else tests="nltw_hb:500 nltw_hb12:1000"; fi
for t in $tests; do
  spec=${t%%:*}; every=${t##*:}
  if [[ $mode == final ]]; then todo=$steps; else todo=$(seq "$every" "$every" "$steps"); fi
  for s in $todo; do
    ck=$(printf '%s/%s/eval-checkpoints/target_%07d-actual_%07d' "$RUNS_DIR" "$id" "$s" "$s")
    out=$EVAL_DIR/$spec/${id}_s$s.json
    [[ -f $out ]] && continue
    if [[ ! -d $ck ]]; then
      [[ $mode == final ]] && echo "no checkpoint at update $s yet: $id (eval_run.sh $id curve scores the earlier ones)" >&2
      continue
    fi
    "$PYTHON" -m nl_tower.evaluate --spec "$spec" --model "$ck" --test "$NL_TOWER_WORKDIR/data/$spec/corpus_tw1/test.jsonl" \
      --out "$out" --n 64 --max-tokens 1280
  done
done
