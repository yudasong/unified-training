# nl_tower

The reasoning-puzzle environment of the controlled experiments (Section 5.1, Appendices A-C). In each puzzle, n people
live on distinct floors of a tower, clues relate their floors through references such as "the best friend of the person
on floor 19", and the answer is the full `Name=floor` assignment. The package generates the puzzles, writes the
midtraining traces with a deterministic solver, builds the data and scores models with pass@k.

Puzzle sets (`specs.py`): `nltw_e` (6-7 people) and `nltw_m0` (8-9) use only the clue compositions demonstrated in
midtraining and supply the corpus questions; `nltw_m0` is also the demonstrated-pattern test. `nltw_m` (8-9 people,
Base), `nltw_hb` (10-11, Extended) and `nltw_hb12` (12, Extended generalization, never trained on) add held-out
compositions. Test sets have 1,024 questions. The trainer ([`unified_training`](../unified_training/README.md)) reads
the prompt mixes (`UNIFIED_PROMPTS_JSON`) and demonstration rows (`UNIFIED_ANCHOR_DIRS`) built here; with
`UNIFIED_REWARD_MODE=nl_tower` its generator scores answers with `nl_tower_correct` in
[`math_reward.py`](../unified_training/math_reward.py).

## Modules

| Module | Role |
|---|---|
| `generator.py` | Puzzle generator (`NLTowerConfig`, `NLTowerDataset`): referential clues, minimal clue sets; a puzzle is kept only if the solver finishes it without branching. reasoning-gym is optional (`register_with_reasoning_gym()`). |
| `specs.py` | Puzzle distributions (`SPECS`) and deterministic item streams (`make_dataset(spec, size, seed)`, `iter_items`). |
| `oracle.py` | Step-by-step solver (`solve_verbal`, `oracle_trace`): candidate elimination with reference resolution. It writes every midtraining trace; no language model does. |
| `traces.py` | Midtraining corpus rows: up to K distinct solver traces per question, in different narration styles and clue orders. |
| `pools.py` | Test sets, question pools and RL prompt pools from seeded streams; the corpus mix and its contamination audit. |
| `mixes.py` | The Base (`nltw_m0` + `nltw_m`) and Extended (adds `nltw_hb`) RL prompt mixes as trainer prompt JSONs. |
| `demos.py` | Demonstration rows `{prompt, response}` from the corpus, for a round with loss on response tokens only. |
| `prepare_data.py`, `manifest.py`, `paths.py` | Data CLI (stages below), the check against `artifacts.json`, and the layout under `NL_TOWER_WORKDIR`. |
| `answers.py`, `passk.py`, `evaluate.py` | Answer check (the last brace-balanced `\boxed{...}` must give exactly the gold `Name=floor` pairs), unbiased pass@k, and the evaluator CLI (vLLM). |
| `results.py`, `compute.py` | Per-model scores and two-seed means; token and FLOP estimates (Appendix A.2). |

## Data

```bash
bash experiments/controlled/data.sh          # tests rl-pools mixes corpus demos pretrain-rows verify (CPU)
bash experiments/controlled/data.sh verify   # every rebuilt file against the published artifacts (artifacts.json)
```

[`data.sh`](../../experiments/controlled/data.sh) runs `python -m nl_tower.prepare_data STAGE` with the seeds, sizes and
exclusions of `data_spec.json`; the builders reproduce the published files byte for byte. `demos` needs the GPT-2
tokenizer of the 124M models (`NL_TOWER_TOKENIZER`); `pretrain-rows` is needed only for pre+mid+RL runs. Outputs go
under `NL_TOWER_WORKDIR` (default `$DATA_DIR/nl_tower`). Published copies are the Hugging Face datasets
`sfanm/nltower-env` (tests, RL pools), `sfanm/nltower-rl-mixes` (prompt mixes, configs `base` and `extended`) and
`sfanm/nltower-traces-tw1X` (midtraining traces).

## Evaluation

```bash
python -m nl_tower.evaluate --spec nltw_m --model CKPT \
  --test $NL_TOWER_WORKDIR/data/nltw_m/corpus_tw1/test.jsonl --out OUT.json --n 64 --max-tokens 1280
```

Each question gets `--n` samples at temperature 1 plus one greedy response; the output JSON holds pass@k for
k = 1, 2, 4, ..., 64, coverage and per-question counts `[qid, n, correct]`. `experiments/controlled/eval_run.sh RUN_ID
[final|curve]` runs it on a run's checkpoints (Base: `nltw_m`; Extended: `nltw_hb`, `nltw_hb12`) and `eval_init.sh` on
the starting models, writing `EVAL_DIR/<spec>/<run>_s<step>.json` and `EVAL_DIR/<spec>/init_<init>.json`. `results.py`
computes pass@k per model from its own counts, then averages over the two RL seeds where both have a score; responses of
different models are never pooled. `python experiments/controlled/metrics.py --eval-dir $EVAL_DIR --runs-dir $RUNS_DIR --out OUT` writes
these scores and the two-seed means as CSV. The paper reports pass@1 (accuracy) and pass@16 (coverage); the full
workflow is in [experiments/controlled/README.md](../../experiments/controlled/README.md).
