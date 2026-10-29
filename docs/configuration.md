# Configuration

Every knob of the unified trainer (T), generator (G) and `scripts/train_unified.sh` (L) is a `UNIFIED_*` environment variable; defaults are the code defaults, which are not always the paper values, so experiment `.env` files set every knob their setting depends on.
Paths come from `RUN_DIR`, `DATA_DIR` and `CKPT_DIR` (see `scripts/train_unified.sh`).

| name | default | meaning |
|---|---|---|
| `UNIFIED_RUN_DIR` | `$RUN_DIR` (L) | Run directory shared by all roles (policies/, rounds/, trainer_hf/, eval-checkpoints/, logs/). |
| `UNIFIED_POLICY_HF` | required | Initial HF checkpoint (T loads it at step 0; L seeds `policies/v0000000` from it). |
| `UNIFIED_PROMPTS_JSON` | required (G) | RL prompts: JSON list of `{"question", "gold"[, "source_index"]}`. |
| `UNIFIED_EPISODES_DIRS` | `$RUN_DIR/rounds` (L) | Comma list of RL round parents the trainer reads. |
| `UNIFIED_STREAMING` | `1` (L), `0` (T) | Streaming RL buffer: rescan the round dirs for new rounds while training. |
| `UNIFIED_TRAIN_ITERS` | 8000 | Optimizer updates. |
| `UNIFIED_GBS` | 256 | RL rows per update (anchor rows are added on top in add mode). |
| `UNIFIED_MBS` | 8 | Rows per micro-batch per trainer rank; GBS must divide by MBS x ranks. |
| `UNIFIED_SEQ` | 1024 | Trainer sequence length (rows are padded/truncated to SEQ + 1 tokens). |
| `UNIFIED_LR` | 2e-5 | AdamW learning rate (constant after warmup). |
| `UNIFIED_WARMUP_ITERS` | 0 | Linear warmup updates. |
| `UNIFIED_WD` | 0.1 | AdamW weight decay. |
| `UNIFIED_ADAM_BETA2` | 0.95 | AdamW beta2 (beta1 0.9, eps 1e-8 fixed). |
| `UNIFIED_GRAD_CLIP` | 1.0 | Gradient-norm clip. |
| `UNIFIED_LOSS_NORM` | `weights` | Denominator of the token mean: `response_tokens` (all non-prompt tokens incl. zero-weight responses; Eq. 2, every paper run), `tokens` (weighted positions), `weights` (sum of weights). |
| `UNIFIED_TEMP` | 1.0 | Sampling temperature (G) and the temperature of the reconstructed sampler for RL log-probs (T). |
| `UNIFIED_TOP_K` | -1 | Top-k for sampling (G) and for the reconstructed RL log-probs (T); -1 = off. |
| `UNIFIED_TOP_P` | 0.95 | Top-p for sampling (G) and reconstructed RL log-probs (T); out-of-support tokens leave numerator and denominator. |
| `UNIFIED_LOGPROB_CHUNK_SIZE` | 64 | Positions per checkpointed log-softmax chunk (memory only). |
| `UNIFIED_TRIM_ROWS` | 1 | Trim each micro-batch to its longest row and group rows by length (same loss). |
| `UNIFIED_PAD_TOKEN_ID` | 50256 | Pad id for trainer rows. |
| `UNIFIED_EXPECTED_VOCAB` | model vocab | Rejects rounds with token ids outside the vocabulary. |
| `UNIFIED_SPLIT_SEED` | 1234 | Seed of the validation split; anchor order uses seed+1, pretraining-row order seed+101. |
| `UNIFIED_KL_COEF` | 0 | Reference-policy KL penalty (k3 estimator on RL response tokens, App. E); 0 = off. |
| `UNIFIED_KL_REF_HF` | `UNIFIED_POLICY_HF` | Reference policy of the KL penalty. |
| `UNIFIED_ANCHOR_DIRS` | empty | Comma list of static-data (anchor) rounds mixed with the RL rows. |
| `UNIFIED_ANCHOR_MODE` | `replace` | `replace`: anchor rows take a share of the GBS rows; `add`: GBS RL rows plus anchor rows appended in their own micro-batches. |
| `UNIFIED_ANCHOR_ADD_BASIS` | `tokens` | Add mode: `tokens` turns TOKEN_FRACTION into rows per step via measured lengths (real-world doses); `rows` adds exactly TOKEN_FRACTION x GBS rows (controlled m = 4..64). |
| `UNIFIED_ANCHOR_TOKEN_FRACTION` | 0 | Target share of anchor tokens in the loss tokens (constant schedule). |
| `UNIFIED_ANCHOR_FRACTION` | 0 | Replace mode: anchor share of rows given directly (instead of a token fraction). |
| `UNIFIED_STREAMS_JSON` | unset | Scheduled mixing: `[{"dirs": [...], "token_fraction": x or {"start", "end", "end_progress"}}]` (linear in training progress, held after end_progress). |
| `UNIFIED_MIX_BASIS` | `supervised_tokens` | RL-token measure for fraction-to-row conversion: `response_tokens` (all RL response tokens; paper), `supervised_tokens`, `weight`. |
| `UNIFIED_MIX_RECALC` | `online` | `launch`: freeze the row fraction measured at launch (`launch-mix-state.json`; paper); `online`: re-derive as the RL pool changes. |
| `UNIFIED_MAX_ANCHOR_ROW_FRACTION` | 1.0 | Upper bound on the derived anchor row fraction (keeps the midtraining holdout unused). |
| `UNIFIED_REQUIRE_NO_ANCHOR_REPLAY` | 0 | 1 = refuse a run that would draw more anchor rows than one pass. |
| `UNIFIED_ANCHOR_LOSS` | `pooled` | `pooled`: one token mean over RL and anchor tokens (Eq. 2); `sft`: separately normalized anchor term (App. C.9). |
| `UNIFIED_SEP_SFT_ALPHA` | unset (1.0 under `sft`) | Weight alpha of the anchor term under `sft`; setting it selects `sft`. Needs add mode and `response_tokens`. |
| `UNIFIED_PRETRAIN_DIRS` | empty | Pretraining rows of the pre+mid+RL setting (App. C.8), appended every update after the anchor rows, unit weight, shared token mean; add mode + pooled only. |
| `UNIFIED_PRETRAIN_ROWS_PER_STEP` | 0 | Pretraining rows per update (row j of step s = draw s x rows + j). |
| `UNIFIED_MATCH_RL_STEPS` | 0 | Replace mode: stop when the RL rows consumed equal N x GBS. |
| `UNIFIED_MATCH_RL_ANNEAL` | 0 | With MATCH_RL_STEPS and one scheduled stream: anneal the anchor to zero at that endpoint. |
| `UNIFIED_RECENCY_WINDOW` | 0 | RL rows are drawn from the newest N rounds (rank-balanced across generators); 0 = all rounds. |
| `UNIFIED_WITHOUT_REPLACEMENT` | 0 | Draw RL rows without replacement within the window. |
| `UNIFIED_RESCAN_EVERY` | 256 | Rescan the round dirs every N row fetches. |
| `UNIFIED_REFRESH_ROUNDS_EVERY_STEP` | 0 | Rescan before every update. |
| `UNIFIED_PUBLISH_INTERVAL` | 0 | Publish weights for the generators every k updates (0 = only at saves). |
| `UNIFIED_MAX_POLICY_AGE_STEPS` | 0 | Exclude rounds generated by a policy more than N updates old (the newest round per generator is kept unless strict). |
| `UNIFIED_STRICT_POLICY_AGE` | 0 | 1 = never train on rows older than MAX_POLICY_AGE_STEPS; wait for fresh rounds instead. |
| `UNIFIED_ROLLOUT_WAIT_TIMEOUT_SECONDS` | 600 | Max wait for fresh rows (strict age, freshness guard). |
| `UNIFIED_ROLLOUT_POLL_SECONDS` | 10 | Poll interval while waiting. |
| `UNIFIED_ROLLOUT_MAX_STALE_STEPS` | 0 | Freshness guard: pause updates when the generators fall this many steps behind; 0 = off. |
| `UNIFIED_ROLLOUT_MAX_POLICY_LAG_STEPS` | 0 | Freshness guard: max policy lag of the newest rounds; 0 = off. |
| `UNIFIED_ROLLOUT_CHECK_INTERVAL_STEPS` | 4 | Freshness guard check interval. |
| `UNIFIED_CONSUMED_AGE_LOG_INTERVAL` | 0 | Log the policy age of a consumed batch every N updates. |
| `UNIFIED_VALID_HOLDOUT` | 128 | RL episodes of the first rounds held out (never trained on) for the validation loss. |
| `UNIFIED_EVAL_INTERVAL` | 200 | Validation loss every N updates (and at the end); 0 = off. |
| `UNIFIED_EVAL_ITERS` | 8 | Global batches per validation. |
| `UNIFIED_MIDTRAIN_VAL_DIR` | empty | Held-out midtraining slice scored at each validation (`val/midtrain_lm_loss`, App. E Fig. 30). |
| `UNIFIED_VALID_REFS_MANIFEST` | `trainer_hf/valid_holdout.json` | Frozen validation split (resume check). |
| `UNIFIED_VALID_CONSUMED_SAMPLES` | derived | Override of the validation cursor on resume. |
| `UNIFIED_SAVE_INTERVAL` | 200 | Full resumable save (policy + optimizer) every N updates. |
| `UNIFIED_KEEP_POLICIES` | 24 | Saved policies kept (plus the last two optimizer states). |
| `UNIFIED_EVAL_MILESTONE_INTERVAL` | 1000 | HF export to `eval-checkpoints/target_*` every N updates (what evaluations read). |
| `UNIFIED_EVAL_MILESTONE_DIR` | `$RUN_DIR/eval-checkpoints` | Milestone directory. |
| `UNIFIED_LOG_INTERVAL` | 10 | Log line every N updates. |
| `UNIFIED_LOG_LOSS_TOKENS` | 0 | Append the global loss-token count to the log line. |
| `UNIFIED_LOG_STREAM_TOKENS` | 0 | Append per-stream loss tokens (RL / midtraining / pretraining) and NLLs since the last log line; always on with pretraining rows. |
| `UNIFIED_TRAIN_STOP_FILE` | `$RUN_DIR/STOP_TRAINER` | Its presence makes the trainer save and exit at the next update boundary (before the first rounds arrive: exit without saving). |
| `UNIFIED_DEADLINE_EPOCH` | 0 | Same stop at this Unix time; 0 = none. |
| `UNIFIED_NCCL_TIMEOUT_SECONDS` | 3600 | Process-group timeout (resume folds every round before the first collective). |
| `UNIFIED_RUNTIME_PROOF_STEP` | 0 | Write per-rank diagnostics at this step; 0 = off. |
| `UNIFIED_RUNTIME_PROOF_DIR` | `$RUN_DIR/runtime-proof` | Directory of those diagnostics. |
| `UNIFIED_CHUNK` | 1024 | Prompts per generator round (per generator process). |
| `UNIFIED_G` | 16 | Completions per prompt. |
| `UNIFIED_MAX_TOKENS` | 1024 | Response cap (also bounded by MAX_MODEL_LEN - prompt length). |
| `UNIFIED_MAX_MODEL_LEN` | 1024 | vLLM context length; longer prompts are skipped. |
| `UNIFIED_GPU_MEM_UTIL` | 0.6 | vLLM GPU memory share (lower it when generator and trainer share GPUs). |
| `UNIFIED_REPEAT_PROMPTS` | 1 | Submit each prompt G times with n=1 (instead of one request with n=G). |
| `UNIFIED_PROMPT_TEMPLATE` | `chatml` | `chatml` (user/assistant tags, stop at `<\|im_end\|>`) or `raw` (question verbatim, stop at EOS; reasoning puzzles). |
| `UNIFIED_REWARD_MODE` | `last_number` | `verl_box` (MATH, real world), `nl_tower` (puzzles), `hf_math_verify`, `hf_math_verify_strict`, `code_tests` (KodCode-style pytest), `last_number`. |
| `UNIFIED_REQUIRE_EOS` | 0 | 1 = a completion that fills its budget without EOS is wrong (kept in its group); needs DROP_TRUNCATED=0. |
| `UNIFIED_EMIT_ZERO_WEIGHT` | 0 | 1 = write wrong completions with weight 0 so they count in the token mean (every paper run). |
| `UNIFIED_HARD_FILTER` | 0 | 1 = REINFORCE: weight 1 on correct completions instead of MaxRL's 1/p_hat (App. D.5). |
| `UNIFIED_MAXRL_BASELINE` | 0 | 1 = MaxRL with group-mean baseline, A = (r - p_hat)/(p_hat + eps) on every completion. |
| `UNIFIED_MAXRL_EPS` | 1e-6 | eps of the baseline form. |
| `UNIFIED_WMAX` | 8.0 | Clip of the 1/p_hat weight (paper runs: 1e9, i.e. none). |
| `UNIFIED_NORMALIZE_WEIGHTS` | 1 | Rescale each round's weights to token-mean 1 (paper runs: 0). |
| `UNIFIED_KEEP_ALLCORRECT` | 1 | Keep groups whose completions are all correct. |
| `UNIFIED_DROP_ALL_WRONG` | 0 | Drop groups with no correct completion. |
| `UNIFIED_DROP_TRUNCATED` | 1 | Drop length-truncated completions (paper runs: 0). |
| `UNIFIED_FORCE_TRUNCATED_INCORRECT` | 0 | Score length-truncated completions as wrong. |
| `UNIFIED_MAX_GROUP_ACC` | 1.1 | Drop groups with pass rate >= this (1.1 = off). |
| `UNIFIED_EASY_KEEP_FRAC` | 0.1 | Fraction of those easy groups kept (deterministic by group id). |
| `UNIFIED_POLICY_RELOAD_READ_MODE` | `mmap` | How generators read published weights: `mmap` or `read` (eager). |
| `UNIFIED_MAX_BATCHES` | 0 | Generator stops after N rounds; 0 = until STOP_GENERATOR. |
| `UNIFIED_MAX_SECONDS` | 0 | Generator wall-clock limit; 0 = none. |
| `UNIFIED_FORCE_RELOAD` | 0 | Test knob: reload weights every round. |
| `UNIFIED_STOP_AFTER_RELOADS` | 0 | Test knob: stop after N reloads. |
| `UNIFIED_GEN_RANK` | set by L | Generator process index (prompt shard). |
| `UNIFIED_GEN_WORLD` | set by L | Number of generator processes (all nodes). |
| `UNIFIED_GEN_TAG` | set by L | Stable generator name (round names, resume cursor). |
| `UNIFIED_CODE_TEST_PYTHON` | current python | Interpreter for `code_tests` sandboxes. |
| `UNIFIED_CODE_TEST_PYTHONPATH` | empty | Extra PYTHONPATH inside the sandbox (where pytest lives). |
| `UNIFIED_CODE_TEST_WORKERS` | CPU count (max 32) | Parallel sandboxes. |
| `UNIFIED_CODE_TEST_TIMEOUT` | 10 | Seconds per test run. |
| `UNIFIED_CODE_TEST_MEMORY_MB` | 4096 | Memory limit per sandbox. |
| `UNIFIED_CODE_TEST_TMPDIR` | system tmp | Sandbox scratch directory. |
| `UNIFIED_CODE_TEST_FAST` | 1 | In-process fast runner for simple test modules (falls back to pytest). |
| `UNIFIED_GENERATOR_GPU_IDS` | first half of GPUs (L) | Generator GPUs of this node, one process each. |
| `UNIFIED_TRAINER_GPU_IDS` | second half (L) | Trainer GPUs of this node, one DDP rank each; may overlap the generator list. |
| `UNIFIED_GENERATOR_NNODES` | 1 (L) | Generator nodes (multi-node). |
| `UNIFIED_TRAINER_NNODES` | 1 (L) | Trainer nodes (multi-node). |
| `UNIFIED_TRAINER_NPROC_PER_NODE` | derived (L) | Trainer ranks per node (= number of trainer GPU ids). |
| `UNIFIED_TRAINER_MASTER_PORT` | 29500 (L) | Rendezvous port of multi-node trainers. |
