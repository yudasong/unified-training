# Reasoning puzzles (controlled setting)

A 124M model learns the reasoning puzzles of `src/nl_tower` in two stages, each done either unified or staged:

1. **Pretraining + midtraining** gives the starting models. They are on Hugging Face and download on first use.
2. **Midtraining + RL** (mid+RL), or **pre+mid+RL** with pretraining data added, is what the scripts here train.

## Setup

```bash
export DATA_DIR=... CKPT_DIR=... RUNS_DIR=... EVAL_DIR=...
bash data.sh   # puzzle sets, RL prompts, the 796,175-example midtraining corpus (CPU); `data.sh verify` checks md5
               # uses up to 32 CPUs (RG_WORKERS sets the number); about 45 min with 24-40 workers
```

Pretraining text (for `pre-mid-rl` and the held-out loss) comes from `scripts/pretrain/prepare_data.sh`.

## 1. Pretraining + midtraining: starting models

| model in `sfanm/unified-training` | training |
|---|---|
| `controlled/pretrain-only` | pretraining only |
| `controlled/mixed-{133M,267M,534M,1.07B,2.14B}` | unified: that many midtraining tokens mixed into the last 8,147 pretraining updates |
| `controlled/staged-{1,2,4}ep-lr{2e-4,5e-5}` | staged: midtraining after pretraining, 1/2/4 epochs at peak lr 2e-4 or 5e-5 |

## 2. Midtraining + RL

`run_rl.sh ID` trains one run of `runs.tsv` on one 4-GPU node. On A100s a Base run (6,000 updates) took 16-21 h and
an Extended run (12,000) 35-43 h. Other GPU counts: `LAYOUT=colocated2` (2 GPUs, about 10 s per update against 9 s on
4), `LAYOUT=colocated1` (1 GPU, about 15 s), `LAYOUT=split8` (8 GPUs). In every layout an update trains on the 512
newest generated rows. To continue an interrupted run, run the same command again; a trailing `UNIFIED_DEADLINE_EPOCH=<Unix time>`
makes the trainer save and stop before a job's time limit. A run takes one config per factor:

| factor | configs |
|---|---|
| puzzles | `base` (8-9 people, 6,000 updates), `extended` (adds 10-11 people, 12,000 updates) |
| starting model | `init/*` (above) |
| midtraining data | `mix/m{0,4,8,16,32,64}`: midtraining examples added to the 512 RL examples of each update |
| loss | `recipe/unified` (one shared loss), `recipe/separate-a{0.1,0.5,1.0}` (separately normalized RL and midtraining losses, midtraining weighted α), `recipe/pre-mid-rl` (pre+mid+RL: unified + pretraining text, ~10% of the midtraining tokens), `recipe/rl` (RL only) |
| seed | `seed/s1234`, `seed/s4321` |

`experiment.sh GROUP [base|extended] list|train|eval` runs a whole group:

| group | runs |
|---|---|
| `unified` | unified starting models, m = 4-64, shared loss |
| `staged` | staged starting models, RL only |
| `no-mix` | unified starting models, RL only (extended setting only) |
| `separate-loss` | as `unified`, with separately normalized losses |
| `pre-mid-rl` | pre+mid+RL: as `unified` (534M and 2.14B starting models, m = 8 and 32), plus pretraining text in every update |

## 3. Evaluation

- `eval_run.sh ID [final|curve]` scores a run on its puzzles (1,024 questions × 64 samples): `final` (default) the last update, `curve` every 500 updates and, for `extended`, the 12-person puzzles every 1,000 (use `curve` for a run that has not finished).
- `eval_init.sh` scores the starting models on all puzzle sets and on retention: held-out pretraining loss and MMLU, ARC-Easy, HellaSwag, TruthfulQA, WinoGrande. Run the puzzles with the training environment (`PART=puzzles`) and retention with the benchmark environment (`PART=retention PYTHON=<benchmark env>/bin/python`).
- `python metrics.py --eval-dir $EVAL_DIR --runs-dir $RUNS_DIR --out OUT` writes CSVs: pass@1 / pass@16, tokens and FLOPs per checkpoint, two-seed means (per model, never pooled) and retention.

The public results [`sfanm/nltower-results`](https://huggingface.co/datasets/sfanm/nltower-results) use the run ids of
`runs.tsv` (`metrics.py --results` reads them).
