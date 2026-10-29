"""Evaluator scoring, pass@k, two-seed averaging and the compute arithmetic of Appendix A (CPU only)."""

from __future__ import annotations

import json
import math

import pytest

from nl_tower import compute, results
from nl_tower.evaluate import score_outputs
from nl_tower.passk import mean_pass_at_k, pass_at_k
from nl_tower.specs import make_dataset


def test_pass_at_k():
    assert pass_at_k(64, 0, 16) == 0.0 and pass_at_k(64, 64, 1) == 1.0
    assert pass_at_k(64, 1, 1) == pytest.approx(1 / 64)
    assert pass_at_k(4, 2, 2) == pytest.approx(1 - 1 / 6)  # 1 - C(2,2)/C(4,2)
    assert pass_at_k(64, 50, 16) == 1.0  # n - c < k
    assert mean_pass_at_k([("a", 4, 0), ("b", 4, 4)], 1) == 0.5
    with pytest.raises(ValueError):
        pass_at_k(4, 1, 5)


def test_score_outputs():
    ds = make_dataset("nltw_e", size=1, seed=20260921)
    e = ds[0]
    test = [{"qid": "q0", "question": e["question"], "answer": e["answer"], "metadata": e["metadata"]}]
    good, bad = f"so \\boxed{{{e['answer']}}}", "\\boxed{Ava=1}"
    outs = [[(good, 10, False), (bad, 10, False), ("no box", 1280, True), (good, 20, False)]]
    s = score_outputs(ds, test, outs)
    assert s["acc"] == 0.5 and s["pass@1"] == 0.5 and s["per_q"] == [["q0", 4, 2]]
    assert s["pass@2"] == pytest.approx(1 - 1 / 6) and s["pass@4"] == 1.0 and "pass@8" not in s
    assert s["box_rate"] == 0.75 and s["cap_rate"] == 0.25 and s["coverage"] == 1.0
    g = score_outputs(ds, test, [[(good, 5, False)]], sampled=False)
    assert g["acc"] == 1.0 and "per_q" not in g


def test_seed_means_never_pool(tmp_path):
    # seed 1234: 1 of 2 questions always right; seed 4321: half of the samples right on both questions
    per_q = {"1234": [["a", 4, 4], ["b", 4, 0]], "4321": [["a", 4, 2], ["b", 4, 2]]}
    for seed, pq in per_q.items():
        p = tmp_path / "nltw_m" / f"base_unified_mixed-133M_m4_s{seed}_s6000.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"sample": {"per_q": pq}}))
    (tmp_path / "nltw_m" / "init_mixed-133M.json").write_text(json.dumps({"sample": {"per_q": [["a", 4, 1], ["b", 4, 0]]}}))
    recs = results.load_eval_dir(tmp_path)
    per_model = results.scores(recs, 2)
    assert per_model[("init_mixed-133M", "nltw_m", 0)] == pytest.approx(0.25)
    means = results.seed_means(per_model)
    # per-model pass@2: 0.5 and 1 - C(2,2)/C(4,2) = 5/6; pooled samples would give a different number
    assert means[("base_unified_mixed-133M_m4", "nltw_m", 6000)] == pytest.approx((0.5 + 5 / 6) / 2)
    del per_model[("base_unified_mixed-133M_m4_s4321", "nltw_m", 6000)]
    assert results.seed_means(per_model) == {}  # one seed only: no point


def test_paper_exposure_numbers():
    # Table 3 / App. C.2: 133M, 267M, 534M, 1.07B, 2.14B midtraining tokens; Fig. 5 legend FLOPs before RL
    toks = {i: compute.midtraining_tokens_before_rl(i) for i in compute.MIXED_K}
    assert [round(t / 1e9, 3) for t in toks.values()] == [0.133, 0.267, 0.534, 1.068, 2.136]
    assert list(compute.MIXED_K.values()) == [8, 16, 32, 64, 128]
    flops = [6 * compute.N_PARAMS * t for t in toks.values()]
    assert [f"{f:.1e}" for f in flops] == ["9.9e+16", "2.0e+17", "4.0e+17", "7.9e+17", "1.6e+18"]
    assert compute.T_PRE == 38147 * 256 * 2048
    assert compute.midtraining_tokens_before_rl("staged-2ep-lr2e-4") == 2 * 509_942_713
    assert compute.midtraining_tokens_before_rl("pretrain-only") == 0
    # App. A.2: m in {4..64} gives 0.03-0.48 passes in Base and 0.06-0.96 in Extended
    assert [round(compute.corpus_passes(6000, m), 2) for m in (4, 8, 16, 32, 64)] == [0.03, 0.06, 0.12, 0.24, 0.48]
    assert [round(compute.corpus_passes(12000, m), 2) for m in (4, 8, 16, 32, 64)] == [0.06, 0.12, 0.24, 0.48, 0.96]
    assert compute.MID_RESPONSE_TOKENS == pytest.approx(382.358, abs=1e-3)


def test_flops(tmp_path):
    log = tmp_path / "trainer.log"
    log.write_text("".join(f"[unified-hf] iteration {s:>8}/6000 | consumed samples: 0 | lm loss: 1.0 | anchor rows added: 8 "
                           f"| loss tokens: {200000 + 8 * 382}\n" for s in range(10, 6001, 10)))  # fmt: skip
    lt = compute.logged_loss_tokens([log])
    assert len(lt) == 600 and lt[10] == 200000 + 8 * 382
    resp = compute.rl_response_tokens(lt, 8, 6000)
    assert resp == pytest.approx(6000 * (200000 + 8 * 382 - 8 * compute.MID_RESPONSE_TOKENS))
    c = compute.flops("mixed-534M", "base", 8, 6000, resp)
    n = compute.N_PARAMS
    want = 6 * n * (compute.T_PRE + 8147 * 32 * 2048) + 8 * n * (6000 * 512 * compute.PROMPT_TOKENS["base"] + resp)
    want += 6 * n * 6000 * 8 * compute.MID_EXAMPLE_TOKENS
    assert c == pytest.approx(want)
    assert compute.flops("mixed-534M", "base", 8, 6000, resp, additional=True) == pytest.approx(want - 6 * n * compute.T_PRE)
    assert math.isclose(compute.MID_EXAMPLE_TOKENS, 640.49, abs_tol=0.01)
