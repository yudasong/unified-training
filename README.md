# Training Foundation Models with a Single Algorithm

Code for the paper. One trainer fits static data (pretraining, midtraining, SFT) and RL generations with a single
weighted likelihood loss; a generator appends RL rounds to a shared buffer while the trainer samples the data mixture.

## Install

Needs [uv](https://docs.astral.sh/uv/) (pip has no `--override`) and an NVIDIA driver >= 580: the pinned
torch 2.11.0 wheel is built for CUDA 13.0.

```bash
# trainer + vLLM generator (Python 3.12); overrides.txt keeps the paper's numpy 2.5.2, which numba's metadata excludes
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[generator,controlled,test]" --override overrides.txt   # + pretrain to tokenize pretraining data

# benchmark harness and real-world data preparation: separate environment (Python 3.13, vLLM 0.17.1, lm-eval 0.4.9)
bash scripts/setup_evals_env.sh && source .venv-evals/bin/activate

# NeMo-RL runs (staged math RL and coding RL of the real-world setting): a third environment (Python 3.13,
# torch 2.10.0, vLLM 0.17.1, ray 2.57.0) with NeMo-RL fetched at the pinned commit and patched
bash scripts/setup_nemo_rl_env.sh
```

`uv pip check` lists numba and mistral-common (numpy pins) in the training environment and the repository's own
torch/numpy/pyarrow pins in the benchmark environment (installed with `--no-deps`), and decord in the NeMo-RL
environment (its only wheel is tagged for CPython 3.6; it imports fine); none of these affects the code here.

On clusters, keep caches (`HF_HOME`, `TRITON_CACHE_DIR`, `VLLM_CACHE_ROOT`, `TMPDIR`) on node-local disk. Triton needs
GCC (`CC=gcc CXX=g++` where `cc` is another compiler, e.g. Cray). If the puzzle evaluator fails to build vLLM's
FlashInfer sampler, set `VLLM_USE_FLASHINFER_SAMPLER=0`. Use whole GPUs: vLLM has hung on A100 MIG slices.

## Checkpoints and data

Starting models are on Hugging Face in [`sfanm/unified-training`](https://huggingface.co/sfanm/unified-training)
(`controlled/*` for Section 5.1, `realworld/*` for Section 5.2); midtraining corpora and held-out sets are in the dataset
[`sfanm/unified-training`](https://huggingface.co/datasets/sfanm/unified-training). The experiment scripts download what they need.

## Training

```bash
RUN_DIR=runs/my-run bash scripts/train_unified.sh common.env my-run.env
```

Runs are configured with `UNIFIED_*` environment variables ([docs/configuration.md](docs/configuration.md)). The launcher
runs on one node (generator and trainer on separate or shared GPUs) or on several nodes from a `HOSTFILE`.

## Reproducing the paper

| Setting | Folder |
|---|---|
| Section 5.1, Appendices A-C: reasoning puzzles, 124M model | [experiments/controlled](experiments/controlled) |
| Section 5.2, Appendices D-E: real-world midtraining data, 0.76B model | [experiments/realworld](experiments/realworld) |

Each folder's README describes its training stages, run configs and evaluation. The staged math RL and the
follow-up coding RL of Section 5.2 use NeMo-RL (`scripts/setup_nemo_rl_env.sh` builds its environment and applies
our patch to a pinned commit).

## Layout

```
src/unified_training/   trainer, generator, rewards, data tools (tools/), benchmark harness (evals/)
src/nl_tower/           reasoning-puzzle environment, data builders, pass@k evaluator
experiments/            configs and scripts per paper setting
third_party/nemo-rl/    pinned NeMo-RL + patch
tests/                  CPU tests (pytest tests)
```

## Citation

```bibtex
@article{unifiedtraining2026,
  title  = {Training Foundation Models with a Single Algorithm},
  note   = {Citation to be added},
  year   = {2026}
}
```

## License

Code: Apache-2.0; third-party code and data licences are listed in [NOTICE](NOTICE).
