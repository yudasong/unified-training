"""HumanEval+ / MBPP+ via EvalPlus: sampled code generation, sandboxed execution, pass@k.

Decoding follows the paper: chat-formatted EvalPlus prompt, T=1, top-k 50, top-p 0.95, 768 new tokens,
context 2048, vLLM seed ``seed ^ sha256(prompt)[:4]`` per prompt. ``n=16`` (pass@1 in the figures) goes
through ``evalplus.codegen`` one task at a time with whole-task resume; other ``n`` (pass@32 coverage)
generate all tasks in one batched vLLM call with the same per-prompt seeds. Execution uses EvalPlus with an
overall deadline: samples still running ``deadline_s`` after scoring starts count as timeouts.

Generated programs are executed: run ``--phase execute`` in a sandbox (e.g. ``unshare -r -n`` with CPU/file limits).
"""

from __future__ import annotations

import collections
import hashlib
import importlib
import json
from pathlib import Path

from unified_training.evals.passk import EXPECTED_SAMPLES, MANY_KS, PASS_KS, aggregate_counts

INSTRUCTION_PREFIX = ("Please provide a self-contained Python script that solves the following problem "
                      "in a markdown code block:")
RESPONSE_PREFIX = ("Below is a Python script with a self-contained function that solves the problem "
                   "and passes corresponding tests:")


def problems_for(dataset: str, task_limit: int | None = None) -> dict:
    from evalplus.data import get_human_eval_plus, get_mbpp_plus

    problems = get_human_eval_plus() if dataset == "humaneval" else get_mbpp_plus()
    if task_limit:
        problems = dict(list(problems.items())[:task_limit])
    return problems


def make_decoder(model: str, dataset: str, seed: int):
    """The checkpoint decoder used for every paper code evaluation (an evalplus DecoderBase)."""
    from evalplus.provider.base import DecoderBase
    from evalplus.provider.utility import EOS, extra_eos_for_direct_completion, make_raw_chat_prompt
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    class CheckpointDecoder(DecoderBase):
        def __init__(self) -> None:
            super().__init__(model, batch_size=16, temperature=1.0, max_new_tokens=768, dtype="bfloat16",
                             instruction_prefix=INSTRUCTION_PREFIX, response_prefix=RESPONSE_PREFIX)
            self.dataset, self.seed = dataset, seed
            self.tokenizer = AutoTokenizer.from_pretrained(model)
            self.eos = [*EOS, "<|im_end|>"]
            if self.is_direct_completion():
                self.eos.extend(extra_eos_for_direct_completion(dataset))
            else:
                self.eos.append("\n```\n")
            self.llm = LLM(model=model, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.85,
                           enable_prefix_caching=True, enforce_eager=True, seed=seed)

        def is_direct_completion(self) -> bool:
            return self.tokenizer.chat_template is None

        def render(self, prompt: str) -> str:
            if self.is_direct_completion():
                return prompt
            return make_raw_chat_prompt(prompt, self.instruction_prefix, self.response_prefix, self.tokenizer)

        def sampling_params(self, rendered: str, n: int, do_sample: bool = True):
            prompt_seed = int.from_bytes(hashlib.sha256(rendered.encode()).digest()[:4], "big")
            return SamplingParams(n=n, temperature=self.temperature if do_sample else 0.0,
                                  top_k=50 if do_sample else -1, top_p=0.95 if do_sample else 1.0,
                                  max_tokens=self.max_new_tokens, stop=self.eos, seed=self.seed ^ prompt_seed)

        def codegen(self, prompt: str, do_sample: bool = True, num_samples: int = 16) -> list[str]:
            rendered = self.render(prompt)
            outputs = self.llm.generate([rendered], self.sampling_params(rendered, num_samples, do_sample),
                                        use_tqdm=False)
            return [c.text.replace("\t", "    ") for c in outputs[0].outputs]

    return CheckpointDecoder()


# ---------------------------------------------------------------- generation

def _read_jsonl_for_resume(path: Path) -> tuple[list[dict], bool]:
    if not path.exists():
        return [], False
    lines = path.read_text().splitlines()
    rows, truncated = [], False
    for index, line in enumerate(lines):
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
            truncated = True  # only a malformed trailing record (killed mid-append) may be dropped
    return rows, truncated


def reconcile_resume_files(path: Path, n: int = EXPECTED_SAMPLES) -> None:
    """Keep only whole tasks (n sanitized + n raw rows) so a resumed run never re-samples a task prefix."""
    raw_path = path.with_name(path.stem + ".raw.jsonl")
    sanitized, s_trunc = _read_jsonl_for_resume(path)
    raw, r_trunc = _read_jsonl_for_resume(raw_path)
    if not (sanitized or raw or s_trunc or r_trunc):
        return
    s_counts = collections.Counter(r["task_id"] for r in sanitized)
    r_counts = collections.Counter(r["task_id"] for r in raw)
    keep = {t for t in s_counts.keys() | r_counts.keys() if s_counts[t] == n and r_counts[t] == n}
    for p, rows, trunc in ((path, sanitized, s_trunc), (raw_path, raw, r_trunc)):
        kept = [r for r in rows if r["task_id"] in keep]
        if trunc or kept != rows:
            stage = p.with_suffix(p.suffix + ".reconcile.tmp")
            stage.write_text("".join(json.dumps(r) + "\n" for r in kept))
            stage.replace(p)


def codegen16(model: str, dataset: str, output: Path, seed: int = 1234, task_limit: int | None = None) -> None:
    """16 samples per task through evalplus.codegen (resumable at whole-task boundaries)."""
    from evalplus.codegen import codegen

    problems = problems_for(dataset, task_limit)
    output.parent.mkdir(parents=True, exist_ok=True)
    reconcile_resume_files(output)
    decoder = make_decoder(model, dataset, seed)
    codegen(target_path=str(output), model=decoder, dataset=problems, greedy=False, n_samples=16, resume=True)
    _write_metadata(output, model, dataset, len(problems), 16, seed, decoder.eos, "evalplus.codegen, per task")


def codegen_batched(model: str, dataset: str, output: Path, n: int, seed: int = 1234,
                    task_limit: int | None = None) -> None:
    """n samples per task in one batched vLLM call, same prompts/seeds/stops/sanitizer as codegen16."""
    from evalplus.sanitize import sanitize

    problems = problems_for(dataset, task_limit)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output.with_suffix(".raw.jsonl")
    for stale in (output, raw_path):  # no partial resume: regenerate the whole set
        if stale.exists():
            stale.unlink()
    decoder = make_decoder(model, dataset, seed)
    direct = decoder.is_direct_completion()
    task_ids, prompts, rendered, params = [], [], [], []
    for task_id, task in problems.items():
        prompt = task["prompt"].strip() + "\n"
        text = decoder.render(prompt)
        task_ids.append(task_id)
        prompts.append(prompt)
        rendered.append(text)
        params.append(decoder.sampling_params(text, n))
    outputs = decoder.llm.generate(rendered, params, use_tqdm=True)
    with open(output, "a") as f_san, open(raw_path, "a") as f_raw:
        for task_id, prompt, out in zip(task_ids, prompts, outputs):
            if len(out.outputs) != n:
                raise SystemExit(f"{task_id}: expected {n} samples, got {len(out.outputs)}")
            for cand in out.outputs:
                impl = cand.text.replace("\t", "    ")
                solution = prompt + impl if direct else impl
                entry = problems[task_id]["entry_point"]
                f_san.write(json.dumps({"task_id": task_id, "solution": sanitize(solution, entrypoint=entry)}) + "\n")
                f_raw.write(json.dumps({"task_id": task_id, "solution": solution}) + "\n")
    _write_metadata(output, model, dataset, len(problems), n, seed, decoder.eos, "batched single vLLM call")


def _write_metadata(output: Path, model, dataset, tasks, n, seed, stop, generation) -> None:
    meta = {"model": model, "dataset": dataset, "tasks": tasks, "samples_per_task": n, "temperature": 1.0,
            "top_k": 50, "top_p": 0.95, "max_new_tokens": 768, "max_model_len": 2048, "stop": stop, "seed": seed,
            "generation": generation}
    output.with_suffix(".metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


# ---------------------------------------------------------------- execution

def aggregate(results: dict, n: int) -> dict:
    """pass@k for EvalPlus base tests and base+plus tests (the paper reports ``plus``)."""
    from evalplus.eval import PASS

    tasks = results["eval"]
    if not tasks:
        raise ValueError("EvalPlus result contains no tasks")
    bad = {t: len(s) for t, s in tasks.items() if len(s) != n}
    if bad:
        raise ValueError(f"expected {n} samples for every task, got {bad}")
    ks = PASS_KS if n == EXPECTED_SAMPLES else MANY_KS
    out = {"tasks": len(tasks), "samples_per_task": n}
    for plus in (False, True):
        counts = [sum(s["base_status"] == PASS and (not plus or s["plus_status"] == PASS) for s in samples)
                  for samples in tasks.values()]
        out["plus" if plus else "base"] = aggregate_counts(counts, n, ks)
    return out


def _add_execution_deadline(evaluate_module, deadline_s: float) -> None:
    """Bound EvalPlus's wait on its worker futures: unfinished samples become timeouts."""
    import concurrent.futures as cf
    import time
    from warnings import warn

    submitted = {}

    class BoundedPool(cf.ProcessPoolExecutor):
        stuck = False

        def submit(self, fn, *fargs, **kwargs):
            future = super().submit(fn, *fargs, **kwargs)
            submitted[future] = fargs
            return future

        def shutdown(self, wait=True, *, cancel_futures=False):
            if BoundedPool.stuck:
                for proc in list((self._processes or {}).values()):
                    try:
                        proc.kill()
                    except Exception:  # noqa: BLE001
                        pass
                return super().shutdown(wait=wait, cancel_futures=True)
            return super().shutdown(wait=wait, cancel_futures=cancel_futures)

    class _Done:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    def bounded_as_completed(futures):
        start = time.time()
        pending = set(futures)
        try:
            for future in cf.as_completed(futures, timeout=deadline_s):
                pending.discard(future)
                yield future
        except cf.TimeoutError:
            BoundedPool.stuck = True
            warn(f"{len(pending)} sample(s) unfinished after {time.time() - start:.0f}s; scoring them as timeouts")
            for future in pending:
                _, completion_id, problem, solution, _, base_only, _, identifier = submitted[future][:8]
                result = {"completion_id": completion_id, "task_id": problem["task_id"], "_identifier": identifier,
                          "solution": solution, "base": ("timeout", [])}
                if not base_only:
                    result["plus"] = ("timeout", [])
                yield _Done(result)

    evaluate_module.ProcessPoolExecutor = BoundedPool
    evaluate_module.as_completed = bounded_as_completed


def execute(dataset: str, samples: Path, n: int, parallel: int = 16, task_limit: int | None = None,
            deadline_s: float | None = None) -> dict:
    evaluate_module = importlib.import_module("evalplus.evaluate")
    if deadline_s:
        _add_execution_deadline(evaluate_module, deadline_s)
    if task_limit:
        problems = problems_for(dataset, task_limit)
        if dataset == "humaneval":
            evaluate_module.get_human_eval_plus = lambda *a, **k: problems
        else:
            evaluate_module.get_mbpp_plus = lambda *a, **k: problems
    result = samples.with_suffix(".eval_results.json")
    evaluate_module.evaluate(dataset=dataset, samples=str(samples), output_file=str(result), parallel=parallel,
                             mini=False)
    summary = aggregate(json.loads(result.read_text()), n)
    samples.with_suffix(f".passk{n}.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def samples_path(output_dir: Path, dataset: str, n: int) -> Path:
    return Path(output_dir) / "code" / (f"{dataset}.jsonl" if n == EXPECTED_SAMPLES else f"{dataset}.n{n}.jsonl")


def run(model: str | None, dataset: str, n: int, output_dir: Path, seed: int = 1234, parallel: int = 16,
        task_limit: int | None = None, deadline_s: float | None = None, phase: str = "all") -> dict | None:
    """Generate and/or execute one benchmark (phase: all | generate | execute); finished parts are skipped."""
    samples = samples_path(output_dir, dataset, n)
    summary = samples.with_suffix(f".passk{n}.json")
    if summary.exists():
        return json.loads(summary.read_text())
    done = samples.with_suffix(".GENERATION_DONE")
    if phase in ("all", "generate") and not done.exists():
        if not model:
            raise ValueError("--model is required to generate")
        if n == EXPECTED_SAMPLES:
            codegen16(model, dataset, samples, seed, task_limit)
        else:
            codegen_batched(model, dataset, samples, n, seed, task_limit)
        done.write_text("done\n")
    if phase == "generate":
        return None
    if not done.exists():
        raise FileNotFoundError(f"no finished generation at {samples}")
    return execute(dataset, samples, n, parallel, task_limit, deadline_s)

