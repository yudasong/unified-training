"""Held-out language-modelling loss (nats/token) of one HF checkpoint.

Every document is scored on its own, in consecutive windows of at most 1024 predicted tokens (window i
predicts tokens [i*1024+1, (i+1)*1024] given the preceding tokens of that window only).

Two sources:
* ``slices``: a directory with ``manifest.json`` (``components``, ``mixture_token_weights``) and one
  ``<component>.npz`` per component (``tokens``, ``offsets``). Used for the 24-component held-out
  midtraining loss of the real-world experiments; ``macro_avg_nats`` (equal component weights) is the
  number reported in the paper.
* ``megatron``: Megatron indexed datasets (``<prefix>.idx/.bin``, uint16 tokens); a fixed seeded sample of
  documents (<= 2048 tokens each, documents shorter than 32 tokens skipped) up to a token budget per
  corpus. Used for the held-out pretraining loss of the controlled (124M) experiments.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

WINDOW = 1024
MAX_DOC_TOKENS = 2048
MIN_DOC_TOKENS = 32
EOD = 50256
_IDX_HEADER = 9 + 8 + 1 + 8 + 8  # magic, version, dtype code, sequence count, document count


def score_documents(model, docs, window: int = WINDOW, device: str = "cuda") -> tuple[float, int]:
    """Sum of next-token NLL and number of scored tokens over `docs` (iterables of token ids)."""
    import torch

    nll, count = 0.0, 0
    with torch.no_grad():
        for doc in docs:
            for s in range(0, len(doc) - 1, window):
                chunk = torch.as_tensor(np.asarray(doc[s:s + window + 1]), dtype=torch.long, device=device)
                if len(chunk) < 2:
                    continue
                logits = model(chunk[:-1].unsqueeze(0)).logits.float()
                nll += torch.nn.functional.cross_entropy(logits[0], chunk[1:], reduction="sum").item()
                count += len(chunk) - 1
    return nll, count


def window_token_count(doc_lengths, window: int = WINDOW) -> int:
    """Number of tokens score_documents scores for documents of the given lengths."""
    return sum(max(int(n) - 1, 0) for n in doc_lengths)


def load_model(path: str | Path, trust_remote_code: bool = True):
    import torch
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        str(path), torch_dtype=torch.bfloat16, trust_remote_code=trust_remote_code).to("cuda").eval()


def slice_documents(slices: Path, component: str) -> list[np.ndarray]:
    z = np.load(Path(slices) / f"{component}.npz")
    tokens, offsets = z["tokens"], z["offsets"]
    return [tokens[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]


def summarize_components(per: dict[str, dict], weights: dict[str, float]) -> dict[str, float]:
    macro = float(np.mean([v["nats_per_token"] for v in per.values()]))
    wsum = sum(weights.get(c, 0.0) for c in per)
    weighted = float(sum(per[c]["nats_per_token"] * weights.get(c, 0.0) for c in per) / wsum) if wsum else float("nan")
    return {"macro_avg_nats": macro, "mixture_weighted_nats": weighted}


def eval_slices(model_path: Path, slices: Path, output: Path) -> dict:
    manifest = json.loads((Path(slices) / "manifest.json").read_text())
    model = load_model(model_path)
    per = {}
    for comp in sorted(manifest["components"]):
        nll, count = score_documents(model, slice_documents(slices, comp))
        per[comp] = {"nats_per_token": nll / max(count, 1), "tokens": count}
        print(f"[loss] {comp}: {per[comp]['nats_per_token']:.4f} ({count:,} tok)", flush=True)
    out = {"model": str(model_path), "per_component": per,
           **summarize_components(per, manifest.get("mixture_token_weights", {})),
           "slices_manifest": str(Path(slices) / "manifest.json")}
    _write(output, out)
    print(f"[loss] macro {out['macro_avg_nats']:.4f} | mixture-weighted {out['mixture_weighted_nats']:.4f}", flush=True)
    return out


def read_megatron_index(prefix: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = np.fromfile(f"{prefix}.idx", dtype=np.uint8)
    if bytes(raw[:9]) != b"MMIDIDX\x00\x00":
        raise ValueError(f"not a Megatron index: {prefix}.idx")
    if int(raw[17]) != 8:
        raise ValueError(f"expected a uint16 token shard (dtype code 8), got {int(raw[17])}")
    n_seq = int(raw[18:26].view(np.int64)[0])
    sizes = raw[_IDX_HEADER:_IDX_HEADER + 4 * n_seq].view(np.int32)
    pointers = raw[_IDX_HEADER + 4 * n_seq:_IDX_HEADER + 12 * n_seq].view(np.int64)
    return sizes, pointers


def sample_megatron_docs(prefix: Path, token_budget: int, seed: int) -> list[np.ndarray]:
    sizes, pointers = read_megatron_index(prefix)
    tokens = np.memmap(f"{prefix}.bin", dtype=np.uint16, mode="r")
    docs, total = [], 0
    for i in np.random.default_rng(seed).permutation(len(sizes)):
        size = int(sizes[i])
        if size < MIN_DOC_TOKENS:
            continue
        start = int(pointers[i]) // 2
        doc = np.array(tokens[start:start + min(size, MAX_DOC_TOKENS)], dtype=np.int64)
        docs.append(doc)
        total += len(doc) - 1
        if total >= token_budget:
            break
    return docs


def eval_megatron(model_path: Path, corpora: dict[str, Path], weights: dict[str, float], output: Path,
                  tokens_per_corpus: int = 400_000, seed: int = 20260905) -> dict:
    model = load_model(model_path, trust_remote_code=False)
    vocab = model.get_input_embeddings().num_embeddings
    per = {}
    for name, prefix in corpora.items():
        docs = sample_megatron_docs(prefix, tokens_per_corpus, seed)
        for doc in docs:
            if doc.max() >= vocab:
                raise ValueError(f"{name}: token id {int(doc.max())} outside vocab {vocab}")
        nll, count = score_documents(model, docs)
        per[name] = {"nats_per_token": nll / max(count, 1), "tokens": count, "documents": len(docs),
                     "documents_ending_in_eod": int(sum(int(d[-1] == EOD) for d in docs)), "shard": str(prefix)}
        print(f"[loss] {name}: {per[name]['nats_per_token']:.4f} ({count:,} tok, {len(docs)} docs)", flush=True)
    wsum = sum(weights.get(n, 0.0) for n in per)
    weighted = sum(per[n]["nats_per_token"] * weights.get(n, 0.0) for n in per) / wsum if wsum else float("nan")
    out = {"model": str(model_path), "per_corpus": per, "weights": weights, "weighted_nats": weighted,
           "window": WINDOW, "seed": seed, "tokens_per_corpus_target": tokens_per_corpus}
    if weights == {"climbmix": 0.75, "dolmino": 0.25}:
        out["mix_75_25_nats"] = weighted  # key used by the original controlled-setting outputs
    _write(output, out)
    return out


def _write(output: Path, obj: dict) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(obj, indent=2, sort_keys=True))
