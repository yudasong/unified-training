# unified_training

The trainer of the paper. A vLLM generator and a trainer share one run directory: the generator writes weighted RL
generations, and the trainer mixes them with static data (pretraining, midtraining or SFT rows) under one weighted
token-likelihood loss (Eq. 2), publishing new weights back to the generator.

```
prompts --> gen_service (vLLM)  --writes-->  RUN_DIR/rounds/        weighted RL generations
                 ^                                  |
                 | reloads                          v
          RUN_DIR/policies/  <--publishes--  hf_trainer (torchrun)  <--  static rounds (UNIFIED_ANCHOR_DIRS,
                                                                         UNIFIED_PRETRAIN_DIRS)
```

All data, generated or static, uses one format: a *round* is a directory of token ids with per-token loss weights
(`episodes.py`). Static data are rounds whose response tokens have weight 1; RL rounds carry the weight the generator
assigns to each completion (MaxRL: 1/p̂ on correct completions).

## Modules

| Module | Role |
|---|---|
| `gen_service.py`, `rollout_common.py` | Generator: samples completions per prompt, verifies them, weights each group, writes rounds; reloads published weights between batches (`vllm_ext.py`, `policy_weights.py`). |
| `episodes.py` | The round format: the generator-to-trainer contract and the format of all static data. |
| `dataset.py` | The trainer's data buffer: static pools, the growing RL stream, and their mixture by token share. |
| `hf_trainer.py` | Trainer: the Eq. (2) loss over mixed rows with DDP; saves and publishes checkpoints (`hf_checkpoint.py`, `checkpoint_milestones.py`). |
| `rollout_freshness.py`, `freshness_batch.py` | Policy-age guard: train only on rounds from recent enough policies. |
| `math_reward.py`, `strict_math_gate.py`, `verl_math_reward.py` | Answer verifiers, chosen by `UNIFIED_REWARD_MODE` (`last_number`, `hf_math_verify[_strict]`, `verl_box`, `nl_tower`). |
| `code_reward.py`, `code_sandbox/` | Unit-test reward for code prompts (`code_tests`), run in a sandbox. |
| `corpus_mix.py`, `holdout.py` | Token-proportional corpus mixtures; identity checks of held-out slices. |
| `tools/` | Data builders that write rounds: SFT pairs, the document-aware midtraining corpus, decontamination, held-out slices, pretraining rows for pre+mid+RL. |
| `pretrain/` | Tokenizes pretraining corpora into GPT-2 Megatron shards. |
| `evals/` | Benchmark harness (`python -m unified_training.evals`): math and code pass@k, SciQ, ARC-Challenge, held-out loss and retention benchmarks. Runs in the separate benchmark environment (README, Install). |

`episodes` and `rollout_common` need only numpy; `dataset` adds torch, `hf_trainer` transformers, `gen_service` vLLM.

## Running

```bash
RUN_DIR=runs/my-run bash scripts/train_unified.sh common.env my-run.env
```

[`scripts/train_unified.sh`](../../scripts/train_unified.sh) starts the generators and the trainer on one node, or on
several from a `HOSTFILE`. Every setting is a `UNIFIED_*` variable; the required ones are `UNIFIED_POLICY_HF` (starting
model) and `UNIFIED_PROMPTS_JSON` (RL prompts). The full list is in
[docs/configuration.md](../../docs/configuration.md); the experiment configs under `experiments/` are complete examples.
