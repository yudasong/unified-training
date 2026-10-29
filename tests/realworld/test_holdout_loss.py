"""Windowed held-out loss: token accounting, aggregation and the Megatron index reader (CPU only)."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from unified_training.evals import holdout_loss

torch = pytest.importorskip("torch")


class _UniformLM(torch.nn.Module):
    """Uniform next-token distribution over V tokens: loss is log(V) per token."""

    def __init__(self, vocab=11):
        super().__init__()
        self.vocab = vocab
        self.calls = []

    def forward(self, ids):
        self.calls.append(ids.shape[1])
        return SimpleNamespace(logits=torch.zeros(ids.shape[0], ids.shape[1], self.vocab))


def test_windows_score_every_token_once():
    docs = [np.arange(1, 2500) % 11, np.arange(5) % 11, np.array([3])]
    model = _UniformLM()
    nll, count = holdout_loss.score_documents(model, docs, device="cpu")
    assert count == holdout_loss.window_token_count([len(d) for d in docs]) == 2498 + 4 + 0
    assert nll / count == pytest.approx(np.log(11), rel=1e-6)
    assert max(model.calls) == holdout_loss.WINDOW  # no window sees more than 1024 inputs


def test_component_summary_equal_and_mixture_weights():
    per = {"a": {"nats_per_token": 1.0}, "b": {"nats_per_token": 2.0}}
    s = holdout_loss.summarize_components(per, {"a": 3.0, "b": 1.0})
    assert s["macro_avg_nats"] == pytest.approx(1.5)
    assert s["mixture_weighted_nats"] == pytest.approx(1.25)


def test_slice_documents_roundtrip(tmp_path):
    tokens = np.array([5, 6, 7, 8, 9, 10], dtype=np.uint16)
    np.savez(tmp_path / "comp.npz", tokens=tokens, offsets=np.array([0, 2, 6]))
    (tmp_path / "manifest.json").write_text(json.dumps({"components": {"comp": {}}}))
    docs = holdout_loss.slice_documents(tmp_path, "comp")
    assert [d.tolist() for d in docs] == [[5, 6], [7, 8, 9, 10]]


def _write_megatron(prefix, docs):
    tokens = np.concatenate([np.asarray(d, dtype=np.uint16) for d in docs])
    tokens.tofile(f"{prefix}.bin")
    sizes = np.array([len(d) for d in docs], dtype=np.int32)
    pointers = np.concatenate([[0], np.cumsum(sizes[:-1]) * 2]).astype(np.int64)
    header = b"MMIDIDX\x00\x00" + np.int64(1).tobytes() + bytes([8]) + np.int64(len(docs)).tobytes() + \
        np.int64(len(docs) + 1).tobytes()
    with open(f"{prefix}.idx", "wb") as f:
        f.write(header + sizes.tobytes() + pointers.tobytes() + np.arange(len(docs) + 1, dtype=np.int64).tobytes())


def test_megatron_reader_and_sampling(tmp_path):
    docs = [np.arange(40) + i for i in range(5)] + [np.arange(10)] + [np.arange(3000)]
    prefix = tmp_path / "shard"
    _write_megatron(prefix, docs)
    sizes, pointers = holdout_loss.read_megatron_index(prefix)
    assert sizes.tolist() == [len(d) for d in docs]
    sampled = holdout_loss.sample_megatron_docs(prefix, token_budget=10**9, seed=0)
    assert len(sampled) == 6  # the 10-token document is skipped
    assert max(len(d) for d in sampled) == holdout_loss.MAX_DOC_TOKENS
    again = holdout_loss.sample_megatron_docs(prefix, token_budget=10**9, seed=0)
    assert all((a == b).all() for a, b in zip(sampled, again))
    small = holdout_loss.sample_megatron_docs(prefix, token_budget=50, seed=0)
    assert sum(len(d) - 1 for d in small) >= 50 and len(small) < 6
