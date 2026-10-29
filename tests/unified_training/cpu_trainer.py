# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""CPU harness for the trainer: a tiny deterministic fixture and a gloo runner of hf_trainer.main().

``python cpu_trainer.py fixture OUT`` builds the fixture; under ``torch.distributed.run``,
``python cpu_trainer.py run SRC_DIR PACKAGE`` runs ``PACKAGE.hf_trainer.main()`` from SRC_DIR with the
CUDA-only calls replaced (nccl -> gloo, cuda -> cpu, bf16 autocast off). The same replacements apply to
any revision of the trainer, which is how the golden values of test_paper_parity.py were recorded.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

VOCAB, SEQ = 64, 32
CPU_PATCHES = (
    ('dist.init_process_group("nccl", timeout=', 'dist.init_process_group("gloo", timeout='),
    ("torch.cuda.set_device(rank % torch.cuda.device_count())", "pass"),
    ('device = torch.device("cuda")', 'device = torch.device("cpu")'),
    ('torch.autocast("cuda", dtype=torch.bfloat16)', 'torch.autocast("cpu", dtype=torch.bfloat16, enabled=False)'),
)


def build_fixture(out: Path) -> None:
    import tokenizers
    import torch
    import transformers

    from unified_training.episodes import RoundWriter
    from unified_training.rollout_common import weight_groups, write_episode_round
    from unified_training.tools.build_midtrain_holdout import prepare_holdout, sha256_file

    out.mkdir(parents=True, exist_ok=False)
    words = ["<pad>", "<eos>"] + [f"t{i}" for i in range(2, VOCAB)]
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="t2"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="t2", eos_token="<eos>",
                                               pad_token="<pad>")
    cfg = transformers.GPT2Config(vocab_size=VOCAB, n_positions=64, n_embd=32, n_layer=2, n_head=2, bos_token_id=1,
                                  eos_token_id=1, pad_token_id=0, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    model = transformers.GPT2LMHeadModel(cfg)
    rng = np.random.default_rng(0)
    with torch.no_grad():   # numpy-drawn weights: independent of torch's initializers
        for _, p in sorted(model.named_parameters()):
            p.copy_(torch.from_numpy((0.05 * rng.standard_normal(tuple(p.shape))).astype(np.float32)))
    model.save_pretrained(out / "init", safe_serialization=True)
    tok.save_pretrained(out / "init")

    for name, hard_filter in (("rounds_maxrl", False), ("rounds_reinforce", True)):
        rng = np.random.default_rng(10)
        batch = {0: 0, 1: 0}
        for version in range(6):   # 2 generator ranks x policy versions 0..5, 4 prompts x 4 completions
            for rank in (0, 1):
                groups = []
                for p in range(4):
                    prompt = rng.integers(2, VOCAB, size=int(rng.integers(3, 8))).tolist()
                    comps = [(rng.integers(2, VOCAB, size=int(rng.integers(2, 20))).tolist() + [1], "", True,
                              bool(rng.random() < 0.4)) for _ in range(4)]
                    groups.append((1000 * version + 10 * rank + p, prompt, comps))
                episodes, _ = weight_groups(groups, wmax=1e9, keep_allcorrect=True, hard_filter=hard_filter,
                                            emit_zero_weight=True, drop_truncated=False)
                d = out / name / f"gr{rank:02d}-b{batch[rank]:07d}-q{batch[rank] * 2 + rank:09d}"
                write_episode_round(d, episodes, policy_version=version, hard_filter=hard_filter,
                                    normalize_episode_weights=False,
                                    manifest_extra={"generator_rank": rank, "generator_world": 2,
                                                    "policy_version": version})
                batch[rank] += 1

    def windows(path: Path, n: int, lo: int, hi: int, seed: int, groups: int = 1, prompt_lo: int = 0) -> None:
        r = np.random.default_rng(seed)
        w = RoundWriter(path)
        for i in range(n):
            prompt = r.integers(2, VOCAB, size=int(r.integers(prompt_lo, prompt_lo + 5))) if prompt_lo else []
            body = np.concatenate([r.integers(2, VOCAB, size=int(r.integers(lo, hi))), [1]])
            tokens = np.concatenate([prompt, body]).astype(np.int32)
            weights = np.ones(len(tokens), np.float32)
            prompt_len = max(len(prompt), 1)
            weights[:prompt_len] = 0.0
            w.add(tokens, weights, prompt_len=prompt_len, reward=0.0, weight=1.0, policy_version=-1,
                  group_id=i % groups)
        w.finalize({"kind": "anchor", "window": SEQ + 1})

    windows(out / "anchor_demo", 200, 3, 18, seed=1, prompt_lo=3)    # response-masked demonstrations
    windows(out / "anchor_docs", 400, 3, 32, seed=2, groups=3)      # document windows, first token context
    windows(out / "pretrain", 120, 8, 9, seed=3)                       # pretraining-row windows
    prepare_holdout(anchor_dir=out / "anchor_docs", output_dir=out / "midtrain_holdout",
                    expected_manifest_sha256=sha256_file(out / "anchor_docs" / "manifest.json"),
                    expected_anchor_windows=400, expected_window=SEQ + 1, max_training_rows=6 * 16,
                    max_anchor_row_fraction=0.5, permutation_seed=1235, eval_windows=8)


def run(src_dir: str, package: str) -> None:
    sys.path.insert(0, src_dir)
    path = Path(src_dir) / package.replace(".", "/") / "hf_trainer.py"
    text = path.read_text()
    for old, new in CPU_PATCHES:
        if old not in text:
            raise SystemExit(f"CPU patch target missing in {path}: {old!r}")
        text = text.replace(old, new)
    module = types.ModuleType(f"{package}.hf_trainer_cpu")
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(text, str(path), "exec"), module.__dict__)
    module.main()


if __name__ == "__main__":
    if sys.argv[1] == "fixture":
        build_fixture(Path(sys.argv[2]))
    else:
        run(sys.argv[2], sys.argv[3])
