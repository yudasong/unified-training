"""Answer extraction and metric aggregation of the evaluation harness (CPU only)."""

import importlib.util
import json
from math import comb
from pathlib import Path

import pytest

from unified_training.evals import math_answers, mcq
from unified_training.evals.passk import aggregate_counts, pass_at_k, score_samples

TASKS = Path(__file__).resolve().parents[2] / "src" / "unified_training" / "evals" / "tasks"


def test_pass_at_k_matches_closed_form():
    for n, c, k in [(16, 0, 1), (16, 3, 1), (16, 3, 8), (256, 10, 32), (32, 31, 2)]:
        expected = 1.0 if n - c < k else 1 - comb(n - c, k) / comb(n, k)
        assert pass_at_k(n, c, k) == pytest.approx(expected)
    assert pass_at_k(16, 5, 1) == pytest.approx(5 / 16)
    with pytest.raises(ValueError):
        pass_at_k(16, 17, 1)
    with pytest.raises(ValueError):
        pass_at_k(16, 2, 17)


def test_score_samples_and_aggregate():
    s = score_samples([True] * 4 + [False] * 12)
    assert s["sample_mean"] == s["pass_at_1"] == 0.25
    assert set(s) == {"sample_mean", "pass_at_1", "pass_at_2", "pass_at_4", "pass_at_8", "pass_at_16"}
    with pytest.raises(ValueError):
        score_samples([True] * 15)
    agg = aggregate_counts([0, 256, 128], 256)
    assert agg["pass_at_1"] == pytest.approx(0.5)
    assert agg["pass_at_256"] == pytest.approx(2 / 3)
    assert "pass_at_32" in aggregate_counts([1, 2], 32) and "pass_at_64" not in aggregate_counts([1, 2], 32)


def test_math_prompts_are_the_paper_prompts():
    assert math_answers.doc_to_math({"problem": " What is 1+1? "}) == (
        "Solve the following mathematics problem. Show your reasoning and put only the final answer inside "
        "\\boxed{}.\n\nWhat is 1+1?")
    assert math_answers.doc_to_gsm8k({"question": "How many?\n"}).startswith(
        "Solve the following grade-school mathematics problem. Show your reasoning and put only the final "
        "answer inside \\boxed{}.\n\nHow many?")


def test_gsm8k_gold_answer():
    assert math_answers.gsm8k_gold_answer({"answer": "6 * 7 = 42\n#### 1,042"}) == "1042"
    with pytest.raises(ValueError):
        math_answers.gsm8k_gold_answer({"answer": "no marker"})


def test_all_responses_shape_check():
    assert len(math_answers.all_responses([["a"] * 16])) == 16
    with pytest.raises(ValueError):
        math_answers.all_responses([["a"] * 15])
    with pytest.raises(ValueError):
        math_answers.all_responses(["a"])


@pytest.mark.skipif(importlib.util.find_spec("math_verify") is None, reason="math-verify not installed")
def test_math_verify_scoring():
    assert math_answers.math_correct("\\frac{1}{2}", "so the answer is \\boxed{0.5}")
    assert math_answers.math_correct("42", "... \\boxed{42}")
    assert not math_answers.math_correct("42", "... \\boxed{41}")
    results = [["\\boxed{3}"] * 4 + ["\\boxed{4}"] * 12]
    assert math_answers.process_gsm8k_results({"answer": "#### 3"}, results)["pass_at_1"] == 0.25


@pytest.mark.parametrize("response,expected", [
    ("B", "B"), ("(C)", "C"), ("C.<|im_end", "C"), ("Answer: D", "D"), ("  a) yes", "A"),
    ("Cats are mammals", None), ("The answer is B", None), ("E", None), ("", None),
])
def test_strict_leading_choice(response, expected):
    assert mcq.extract_greedy_choice(response, 4) == expected


def test_mcq_format_and_scoring():
    assert mcq.is_format_valid_response(" B ", 4) and not mcq.is_format_valid_response("B.", 4)
    doc = {"question": "Q?", "eval_choices": ["w", "x", "y", "z"], "eval_answer": "C"}
    assert mcq.process_greedy_mcq_results(doc, ["C"]) == {"greedy_accuracy": 1.0, "format_valid_rate": 1.0}
    assert mcq.process_greedy_mcq_results(doc, [["(c)."]])["greedy_accuracy"] == 1.0
    with pytest.raises(ValueError):
        mcq.process_greedy_mcq_results(doc, ["A", "B"])
    prompt = mcq.doc_to_greedy_mcq({**doc, "support": "ctx"})
    assert prompt.endswith("Respond with exactly one uppercase letter from the listed choices (A, B, C, or D), "
                           "and nothing else.")
    assert "Context:\nctx" in prompt and "Choices:\nA. w\nB. x\nC. y\nD. z" in prompt


class _FakeDataset(list):
    def map(self, fn):
        return _FakeDataset({**d, **fn(d)} for d in self)


def test_sciq_shuffle_is_deterministic_and_tracks_answer():
    row = {"question": "Which?", "support": "s", "distractor1": "a", "distractor2": "b", "distractor3": "c",
           "correct_answer": "d"}
    first = mcq.process_sciq_docs(_FakeDataset([row]))[0]
    second = mcq.process_sciq_docs(_FakeDataset([row]))[0]
    assert first["eval_choices"] == second["eval_choices"]
    assert first["eval_choices"][ord(first["eval_answer"]) - ord("A")] == "d"


def test_arc_labels_canonicalised():
    row = {"choices": {"label": ["1", "2", "3", "4"], "text": ["w", "x", "y", "z"]}, "answerKey": "3"}
    out = mcq.process_arc_docs(_FakeDataset([row]))[0]
    assert out["eval_answer"] == "C" and out["eval_choices"] == ["w", "x", "y", "z"]


def test_task_yaml_functions_resolve():
    """Every `!function module.name` in the lm-eval task YAMLs resolves in the shim next to it."""
    for yml in TASKS.glob("*.yaml"):
        for line in yml.read_text().splitlines():
            if "!function" not in line:
                continue
            module, name = line.split("!function", 1)[1].strip().split(".")
            spec = importlib.util.spec_from_file_location(module, TASKS / f"{module}.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            assert callable(getattr(mod, name)), (yml.name, name)


def test_task_yaml_sampling_protocol():
    text = {p.stem: p.read_text() for p in TASKS.glob("*.yaml")}
    for task in ("math500_16", "gsm8k_16"):
        for needle in ("repeats: 16", "temperature: 1.0", "top_k: 50", "top_p: 0.95", "max_gen_toks: 1024"):
            assert needle in text[task], (task, needle)
    for task in ("sciq_greedy", "arc_challenge_greedy"):
        assert "do_sample: false" in text[task] and "max_gen_toks: 8" in text[task]


def _write_lm_eval(root: Path, task: str, payload: dict) -> None:
    d = root / "noncode" / task / "results" / "model"
    d.mkdir(parents=True)
    (d / "results_2026-01-01T00-00-00.json").write_text(json.dumps({"results": {task: payload}}))


def test_summarize_battery_layout(tmp_path):
    from unified_training.evals.summarize import collect

    _write_lm_eval(tmp_path, "math500_16", {"pass_at_1,all_16": 0.2, "pass_at_16,all_16": 0.6})
    _write_lm_eval(tmp_path, "gsm8k_16", {"pass_at_1,all_16": 0.3, "pass_at_16,all_16": 0.7})
    _write_lm_eval(tmp_path, "sciq_greedy", {"greedy_accuracy,greedy": 0.62})
    _write_lm_eval(tmp_path, "arc_challenge_greedy", {"greedy_accuracy,greedy": 0.3234})
    (tmp_path / "code").mkdir()
    for ds, p1 in (("humaneval", 0.12), ("mbpp", 0.21)):
        (tmp_path / "code" / f"{ds}.passk16.json").write_text(json.dumps(
            {"plus": {"pass_at_1": p1, "pass_at_16": 0.5}, "base": {}}))
        (tmp_path / "code" / f"{ds}.n32.passk32.json").write_text(json.dumps(
            {"plus": {"pass_at_1": p1, "pass_at_32": 0.45}, "base": {}}))
    (tmp_path / "passk").mkdir()
    for task, p32 in (("math500", 0.6), ("gsm8k", 0.8)):
        (tmp_path / "passk" / f"{task}.passk256.json").write_text(json.dumps(
            {"pass_at_32": p32, "pass_at_64": p32 + 0.05, "pass_at_256": p32 + 0.1}))
    ret = tmp_path / "retention"
    (ret / "ll_battery" / "model").mkdir(parents=True)
    (ret / "ll_battery" / "model" / "results_x.json").write_text(json.dumps(
        {"results": {"mmlu": {"acc,none": 0.3999}, "hellaswag": {"acc,none": 0.4, "acc_norm,none": 0.5}}}))
    (ret / "holdout_loss.json").write_text(json.dumps({"macro_avg_nats": 1.3481}))

    m = collect([tmp_path], base_holdout=1.4352)
    assert m["math_pass1"] == pytest.approx(25.0)
    assert m["coding_pass1"] == pytest.approx(16.5)
    assert m["math_pass32"] == pytest.approx(70.0)
    assert m["coding_pass32"] == pytest.approx(45.0)
    assert m["math500_pass256"] == pytest.approx(70.0)
    assert m["sciq"] == pytest.approx(62.0) and m["arc_c"] == pytest.approx(32.34)
    assert m["mmlu"] == pytest.approx(39.99) and m["hellaswag_norm"] == pytest.approx(50.0)
    assert m["holdout_delta"] == pytest.approx(-0.0871)


def test_summarize_original_layout_split_roots(tmp_path):
    """The original outputs keep many-sample and retention results in separate roots."""
    from unified_training.evals.summarize import collect

    eval16, passk, ret = tmp_path / "eval16", tmp_path / "passk", tmp_path / "ret"
    _write_lm_eval(eval16, "math500_16", {"pass_at_1,all_16": 0.1683})
    (passk / "code").mkdir(parents=True)
    (passk / "math500.passk256.json").write_text(json.dumps({"pass_at_64": 0.656, "pass_at_256": 0.772}))
    (passk / "code" / "mbpp.passk32.json").write_text(json.dumps({"plus": {"pass_at_32": 0.5}}))
    ret.mkdir()
    (ret / "holdout_loss.json").write_text(json.dumps({"macro_avg_nats": 1.5}))
    (ret / "ll_battery.json").write_text(json.dumps({"results": {"mmlu": {"acc,none": 0.4}}}))
    m = collect([eval16, passk, ret])
    assert m["math500_pass1"] == pytest.approx(16.83) and m["math500_pass64"] == pytest.approx(65.6)
    assert m["mbpp_plus_pass32"] == pytest.approx(50.0)
    assert m["mmlu"] == pytest.approx(40.0) and m["holdout_loss"] == 1.5 and "holdout_delta" not in m


def test_table_rows_and_trainer_validation(tmp_path):
    import csv

    from unified_training.evals.summarize import table, trainer_validation

    ev, runs = tmp_path / "eval", tmp_path / "runs"
    for cfg, step, loss in (("base", 0, 1.5), ("const0p5", 1000, 1.4)):
        d = ev / cfg / f"step_{step:07d}" / "retention"
        d.mkdir(parents=True)
        (d / "holdout_loss.json").write_text(json.dumps({"macro_avg_nats": loss}))
    (runs / "const0p5" / "logs").mkdir(parents=True)
    (runs / "const0p5" / "logs" / "trainer0.log").write_text(
        "[unified-hf] validation @ 200: lm loss 0.500000 | midtrain lm 1.420000\n"
        "[unified-hf] validation @ 1000: lm loss 0.400000 | midtrain lm 1.410000\n")
    assert trainer_validation(runs / "const0p5")[200] == {"val_lm": 0.5, "val_midtrain_lm": 1.42}
    rows = table(ev, tmp_path / "m.csv", runs)
    by = {(r["config"], r["step"]): r for r in rows}
    assert by[("const0p5", 1000)]["holdout_delta"] == pytest.approx(-0.1)
    assert by[("const0p5", 1000)]["val_midtrain_lm"] == 1.41 and ("const0p5", 200) in by
    with open(tmp_path / "m.csv") as f:
        assert next(csv.reader(f))[:2] == ["config", "step"]
