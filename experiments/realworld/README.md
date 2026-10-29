# Real-world midtraining data

A 0.76B base model learns math with RL on Math12K while a 240M-token midtraining corpus (web text, math, code,
reasoning traces) is mixed in. A follow-up stage then trains coding with RL.

## Setup

```bash
export DATA_DIR=... CKPT_DIR=... RUNS_DIR=... EVAL_DIR=...
source ../../.venv-evals/bin/activate   # data preparation uses the benchmark environment (README, Install)
bash prepare.sh   # base model; Math12K and KodCode from pinned sources; midtraining corpora and held-out sets
```

`prepare.sh` downloads the base model (`realworld/base` in `sfanm/unified-training`) and the published corpora;
`prepare.sh rebuild_midtrain rebuild_code_variants` rebuilds the corpora instead (for reference: `rebuild_midtrain` needs
the base model's own midtraining corpus, `MIDTRAIN_SRC`, which is not part of this release).

## Run

Everything runs from this folder:

```bash
bash run_rl.sh ID                    # one math-RL run of the unified trainer; ID = a file in configs/ (const0p5, ...)
bash reproduce.sh GROUP train        # every run of a group (table below)
bash reproduce.sh GROUP eval         # evaluate its checkpoints (benchmark environment)
bash reproduce.sh GROUP table        # one CSV of all evaluated checkpoints
```

Train in the training environment (`../../.venv`; `../../.venv-nemo-rl` for `staged` and `coding`) and evaluate in the benchmark
environment (`../../.venv-evals`), e.g. `PYTHON=../../.venv-evals/bin/python bash reproduce.sh constant eval`.
To continue an interrupted run, run the same command again; a trailing `UNIFIED_DEADLINE_EPOCH=<Unix time>` on
`run_rl.sh` makes the trainer save and stop before a job's time limit.

| group | runs (`configs/<id>.env`) | what varies |
|---|---|---|
| `base` | none | evaluates the base model |
| `constant` | `const0`, `const0p1`, `const0p2`, `const0p5`, `const1`, `const2`, `const5` | share of midtraining tokens, constant (`0p5` = 0.5%; `const0` = RL only) |
| `schedules` | `fullanneal*`, `halfanneal*`, `q3anneal*`, `rampup1` | share decays linearly to 0 at the end / half / three quarters of training, or grows from 0 |
| `code-data` | `coderemoved_*`, `codereplaced_*` | the corpus without its code sources, or with them replaced by tested code |
| `reinforce` | `reinforce_*` | REINFORCE weights instead of MaxRL |
| `kl` | `kl0p001`, `kl0p01`, `kl0p1` | no midtraining data; KL penalty toward the base model instead |
| `staged` | NeMo-RL | MaxRL on Math12K from the base model, 12,000 updates (`run_nemo_rl.sh math12k`) |
| `coding` | NeMo-RL | coding RL on KodCode, 4,000 updates, from each math-RL model in `realworld/math-rl/*` (staged and constant 0.5/1/2/5%; `run_nemo_rl.sh kodcode`) |

Unified runs train 12,000 updates with `configs/common.env` plus their own file. NeMo-RL runs (`staged`, `coding`) use
their own environment: run `bash ../../scripts/setup_nemo_rl_env.sh` once (it also fetches and patches NeMo-RL), set
`PYTHON=../../.venv-nemo-rl/bin/python`, and start Ray first as the header of `run_nemo_rl.sh` shows.

## GPUs

- **Unified runs** use one node: the first half of its GPUs (or of `CUDA_VISIBLE_DEVICES`) generate, the rest train.
  2 GPUs work; the number of trainer GPUs must divide 256. The loss is normalized over the whole 512-row update, so
  the trainer GPU count does not change it, and `run_rl.sh` sets the recency window to the number of generator GPUs
  (paper: 8). Speed changes: about 23 s per update on 2 A100s (12,000 updates in about 3 days), 4.7 s on 12.
  The paper used 8 generator and 16 trainer GPUs over several nodes, 3.4 s per update (`HOSTFILE` in
  `scripts/train_unified.sh`, `docs/configuration.md`). How old the policy behind each row is (at most 4 updates,
  `UNIFIED_MAX_POLICY_AGE_STEPS`) depends on how fast generation keeps up with training, so it can differ from the
  paper's layout; we have not compared full runs on fewer GPUs with the paper's.
- **NeMo-RL runs**: `NODES` x `GPUS_PER_NODE` (paper: 4 x 4). One node of 4 A100s took about 33 s per update at the
  start, against 29 s on the paper's 16 GPUs. Start Ray before `run_nemo_rl.sh`, also on one node (its header).
- **Evaluation**: `GPUS=0,1` (default `0,1,2,3`).

## Evaluation

- `python -m unified_training.evals battery --model CKPT --output $EVAL_DIR/<id>/step_<N> --slices $DATA_DIR/realworld/heldout-loss-slices`: math (MATH-500, GSM8K) and coding (HumanEval+, MBPP+) pass@k, and retention (held-out midtraining loss, MMLU, SciQ, ARC-Challenge). Without `--slices` the held-out loss is skipped. `--suite` picks `eval16`, `passk` (256 samples per problem, the slow one) and `retention` (default: all three).
- `python -m unified_training.evals table $EVAL_DIR --runs $RUNS_DIR --out metrics.csv`: every evaluated checkpoint in one CSV.

`reproduce.sh GROUP eval` runs the paper's schedule: benchmarks and pass@k every 1,000 updates, retention every 2,000.
