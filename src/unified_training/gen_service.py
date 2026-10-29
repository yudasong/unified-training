"""Generator: a long-running vLLM producer of weighted RL generations for the trainer.

One persistent vLLM engine per GPU; between batches it polls <run>/policies/ for a
newer snapshot (v{step:07d}/DONE) and, on change, reloads the weights into the live
engine (unified_training.vllm_ext.UnifiedWorkerExtension) and invalidates the
prefix/KV cache. Generation is offline-batch, so reloads happen with no requests in
flight.

Each batch: take the next pass-shuffled chunk of UNIFIED_CHUNK prompts, sample G
completions per prompt, verify them (UNIFIED_REWARD_MODE), weight each group
(MaxRL: A = 1/p_hat on correct completions, 0 on wrong ones; REINFORCE with
UNIFIED_HARD_FILTER=1: A = 1 on correct ones) and write one DONE-gated round
directory rounds/g{TAG}-b{batch}-q{global_chunk}. The trainer folds new rounds into
its buffer. The generator never waits for the trainer.

    UNIFIED_RUN_DIR=<run> UNIFIED_PROMPTS_JSON=<prompts.json> UNIFIED_GEN_RANK=<r> \\
    UNIFIED_GEN_WORLD=<n> UNIFIED_GEN_TAG=<unique> python -m unified_training.gen_service

UNIFIED_PROMPTS_JSON is a JSON list of {"question": ..., "gold": ...} records.
"""
import gc
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

# Keep the V1 engine core in-process so worker_extension_cls resolves through this
# package, and collective_rpc is a same-process call (NeMo-RL does the same).
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import numpy as np
from vllm import LLM, SamplingParams

from unified_training.math_reward import (  # noqa: E402
    LAST_NUMBER,
    answers_are_correct,
    validate_reward_mode,
)
from unified_training.rollout_common import (  # noqa: E402
    STANDARD_PASS_KS,
    mean_pass_at_ks,
    prompt_template,
    response_token_budget,
    select_prompt_shard,
    truncated_without_eos,
    weight_groups,
    write_episode_round,
)
from unified_training.policy_weights import POLICY_READ_MODES  # noqa: E402

POLICY_READ_MODE = os.environ.get("UNIFIED_POLICY_RELOAD_READ_MODE", "mmap")
if POLICY_READ_MODE not in POLICY_READ_MODES:
    raise ValueError(f"UNIFIED_POLICY_RELOAD_READ_MODE must be one of {POLICY_READ_MODES}")

RUN_DIR = Path(os.environ["UNIFIED_RUN_DIR"])
PROMPTS = os.environ["UNIFIED_PROMPTS_JSON"]
CHUNK = int(os.environ.get("UNIFIED_CHUNK", "1024"))
G = int(os.environ.get("UNIFIED_G", "16"))
TEMP = float(os.environ.get("UNIFIED_TEMP", "1.0"))
MAX_TOKENS = int(os.environ.get("UNIFIED_MAX_TOKENS", "1024"))
MAX_MODEL_LEN = int(os.environ.get("UNIFIED_MAX_MODEL_LEN", "1024"))
TOP_K = int(os.environ.get("UNIFIED_TOP_K", "-1"))  # -1 = off (vLLM default); regular maxRL uses 50
TOP_P = float(os.environ.get("UNIFIED_TOP_P", "0.95"))
GPU_MEM_UTIL = float(os.environ.get("UNIFIED_GPU_MEM_UTIL", "0.6"))
REPEAT_PROMPTS = os.environ.get("UNIFIED_REPEAT_PROMPTS", "1") == "1"
WMAX = float(os.environ.get("UNIFIED_WMAX", "8.0"))
KEEP_ALLCORRECT = os.environ.get("UNIFIED_KEEP_ALLCORRECT", "1") == "1"
# Easy-group filter: drop groups whose empirical pass rate >= MAX_GROUP_ACC before
# emission (default 1.1 = off), keeping a deterministic EASY_KEEP_FRAC fraction
# by group-id hash as anti-regression coverage. See rollout_common.weight_groups.
MAX_GROUP_ACC = float(os.environ.get("UNIFIED_MAX_GROUP_ACC", "1.1"))
EASY_KEEP_FRAC = float(os.environ.get("UNIFIED_EASY_KEEP_FRAC", "0.1"))
HARD_FILTER = os.environ.get("UNIFIED_HARD_FILTER", "0") == "1"
EMIT_ZERO_WEIGHT = os.environ.get("UNIFIED_EMIT_ZERO_WEIGHT", "0") == "1"
# Drop groups with no correct completion even under EMIT_ZERO_WEIGHT:
# they carry no learning signal but take batch rows and denominator tokens.
DROP_ALL_WRONG = os.environ.get("UNIFIED_DROP_ALL_WRONG", "0") == "1"
DROP_TRUNCATED = os.environ.get("UNIFIED_DROP_TRUNCATED", "1") == "1"
FORCE_TRUNCATED_INCORRECT = os.environ.get("UNIFIED_FORCE_TRUNCATED_INCORRECT", "0") == "1"
NORMALIZE_WEIGHTS = os.environ.get("UNIFIED_NORMALIZE_WEIGHTS", "1") == "1"
MAX_BATCHES = int(os.environ.get("UNIFIED_MAX_BATCHES", "0"))  # 0 = until MAX_SECONDS/wall
MAX_SECONDS = float(os.environ.get("UNIFIED_MAX_SECONDS", "0"))  # 0 = unlimited
FORCE_RELOAD = os.environ.get("UNIFIED_FORCE_RELOAD", "0") == "1"  # test knob: reload every batch
STOP_AFTER_RELOADS = int(os.environ.get("UNIFIED_STOP_AFTER_RELOADS", "0"))  # test knob: 0 = never
TAG = os.environ.get("UNIFIED_GEN_TAG") or str(os.getpid())
GEN_RANK = int(os.environ.get("UNIFIED_GEN_RANK", "0"))
GEN_WORLD = int(os.environ.get("UNIFIED_GEN_WORLD", "1"))
REWARD_MODE = validate_reward_mode(os.environ.get("UNIFIED_REWARD_MODE", LAST_NUMBER))
# UNIFIED_PROMPT_TEMPLATE: "chatml" (default, CHAT_TEMPLATE + "<|im_end|>" stop) or
#   "raw" (question verbatim, EOS-only stop), see rollout_common.PROMPT_TEMPLATES.
PROMPT_TEMPLATE_NAME = os.environ.get("UNIFIED_PROMPT_TEMPLATE", "chatml")
PROMPT_FORMAT, STOP_STRINGS = prompt_template(PROMPT_TEMPLATE_NAME)
# UNIFIED_REQUIRE_EOS=1 (reasoning-puzzle runs): a completion is
#   truncated iff it filled its whole response budget AND its last token is not EOS
#   (an EOS emitted as the budget's final token is a finished answer), and a
#   truncated completion is scored wrong without grading. It stays in the group
#   (counts in p_hat and in the response-token denominator), so DROP_TRUNCATED
#   must be 0.
REQUIRE_EOS = os.environ.get("UNIFIED_REQUIRE_EOS", "0") == "1"
if REQUIRE_EOS:
    if DROP_TRUNCATED:
        raise ValueError(
            "UNIFIED_REQUIRE_EOS=1 keeps truncated completions as wrong answers; set UNIFIED_DROP_TRUNCATED=0"
        )
    FORCE_TRUNCATED_INCORRECT = True
# UNIFIED_MAXRL_BASELINE=1: maxRL WITH the group-mean baseline,
#   A = (r - mean)/(mean + UNIFIED_MAXRL_EPS) on every completion
#   (signed weights; rollout_common.weight_groups). Needs zero-weight emission and raw
#   (unnormalized) weights.
MAXRL_BASELINE = os.environ.get("UNIFIED_MAXRL_BASELINE", "0") == "1"
MAXRL_EPS = float(os.environ.get("UNIFIED_MAXRL_EPS", "1e-6"))
if MAXRL_BASELINE and (NORMALIZE_WEIGHTS or not EMIT_ZERO_WEIGHT or HARD_FILTER):
    raise ValueError(
        "UNIFIED_MAXRL_BASELINE=1 needs UNIFIED_NORMALIZE_WEIGHTS=0, UNIFIED_EMIT_ZERO_WEIGHT=1 "
        "and UNIFIED_HARD_FILTER=0"
    )

if not 0 <= GEN_RANK < GEN_WORLD:
    raise ValueError(f"UNIFIED_GEN_RANK={GEN_RANK} must be in [0, {GEN_WORLD})")
if MAX_MODEL_LEN <= 1:
    raise ValueError(f"UNIFIED_MAX_MODEL_LEN must exceed 1, got {MAX_MODEL_LEN}")
if MAX_TOKENS <= 0:
    raise ValueError(f"UNIFIED_MAX_TOKENS must be positive, got {MAX_TOKENS}")

POLICIES = RUN_DIR / "policies"
ROUNDS = RUN_DIR / "rounds"
PROGRESS = RUN_DIR / "generator_progress" / TAG
STOP_FILE = RUN_DIR / "STOP_GENERATOR"
ROUNDS.mkdir(parents=True, exist_ok=True)
PROGRESS.mkdir(parents=True, exist_ok=True)
with open(PROMPTS) as _stream:
    rows = json.load(_stream)
if not isinstance(rows, list) or not rows:
    raise ValueError(f"UNIFIED_PROMPTS_JSON must contain a non-empty JSON list: {PROMPTS}")
for _index, _row in enumerate(rows):
    if not isinstance(_row, dict) or "question" not in _row or "gold" not in _row:
        raise ValueError(f"prompt row {_index} must contain question and gold")


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


PROMPTS_SHA256 = _sha256(PROMPTS)
t0 = time.monotonic()


def newest_policy() -> tuple[Path, int]:
    best, best_ver = None, -1
    for d in POLICIES.iterdir() if POLICIES.is_dir() else []:
        if not (d / "DONE").exists():
            continue
        digits = "".join(c for c in d.name if c.isdigit())
        ver = int(digits) if digits else 0
        if ver > best_ver:
            best, best_ver = d, ver
    if best is None:
        raise FileNotFoundError(f"no converted policy with DONE under {POLICIES}")
    return best, best_ver


policy_dir, policy_ver = newest_policy()
print(f"[gen] starting engine on {policy_dir} (v{policy_ver})", flush=True)
llm = LLM(
    model=str(policy_dir),
    dtype="bfloat16",
    gpu_memory_utilization=GPU_MEM_UTIL,
    max_model_len=MAX_MODEL_LEN,
    enforce_eager=True,
    enable_prefix_caching=True,
    trust_remote_code=True,
    logprobs_mode="processed_logprobs",
    worker_extension_cls="unified_training.vllm_ext.UnifiedWorkerExtension",
)
tok = llm.get_tokenizer()
STOP_TOKEN_IDS = [tok.eos_token_id] if tok.eos_token_id is not None else None


def _prompt_ids(row) -> list[int]:
    return tok.encode(PROMPT_FORMAT.format(question=row["question"]), add_special_tokens=False)


def _sampling_params(*, n: int, temperature: float, top_p: float, top_k: int | None,
                     max_tokens: int = MAX_TOKENS) -> SamplingParams:
    kwargs = {} if top_k is None else {"top_k": top_k}
    return SamplingParams(
        n=n,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        logprobs=0,
        stop=list(STOP_STRINGS),
        stop_token_ids=STOP_TOKEN_IDS,
        include_stop_str_in_output=True,
        **kwargs,
    )


def _response_budget(prompt_ids: list[int]) -> int:
    budget = response_token_budget(len(prompt_ids), MAX_MODEL_LEN, MAX_TOKENS)
    if budget <= 0:
        raise ValueError(
            f"prompt length {len(prompt_ids)} does not fit max_model_len={MAX_MODEL_LEN}"
        )
    return budget


def _generate_groups(
    prompt_ids: list[list[int]],
    *,
    n: int,
    temperature: float,
    top_p: float,
    top_k: int | None,
    repeat_prompts: bool,
):
    """Generate with an explicit per-request sequence-length budget.

    ``repeat_prompts`` (UNIFIED_REPEAT_PROMPTS, default on) submits each prompt n times
    with n=1 instead of one request with n samples.
    """
    if repeat_prompts:
        prompts = [{"prompt_token_ids": ids} for ids in prompt_ids for _ in range(n)]
        params = [
            _sampling_params(n=1, temperature=temperature, top_p=top_p, top_k=top_k,
                             max_tokens=_response_budget(ids))
            for ids in prompt_ids
            for _ in range(n)
        ]
    else:
        prompts = [{"prompt_token_ids": ids} for ids in prompt_ids]
        params = [
            _sampling_params(n=n, temperature=temperature, top_p=top_p, top_k=top_k,
                             max_tokens=_response_budget(ids))
            for ids in prompt_ids
        ]
    return llm.generate(prompts, params)

def reload_policy(new_dir: Path, new_ver: int) -> None:
    """In-process weight swap + full cache invalidation (NeMo-RL post-refit sequence)."""
    import torch

    t = time.monotonic()
    llm.collective_rpc("update_weights_from_disk", args=(str(new_dir), POLICY_READ_MODE))
    llm.llm_engine.reset_prefix_cache()
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[gen] reloaded weights v{new_ver} <- {new_dir} ({time.monotonic()-t:.1f}s) read_mode={POLICY_READ_MODE}", flush=True)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}"
    with temporary.open("w") as stream:
        json.dump(payload, stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


_progress_identity = {
    "schema_version": 1,
    "tag": TAG,
    "generator_rank": GEN_RANK,
    "generator_world": GEN_WORLD,
    "chunk": CHUNK,
    "n_prompts": len(rows),
    "prompts_sha256": PROMPTS_SHA256,
}
_cursor_file = PROGRESS / "cursor.json"


def _global_chunk(local_batch: int) -> int:
    return local_batch * GEN_WORLD + GEN_RANK


def _round_dir(local_batch: int) -> Path:
    return ROUNDS / f"g{TAG}-b{local_batch:07d}-q{_global_chunk(local_batch):09d}"


def _skip_file(local_batch: int) -> Path:
    return PROGRESS / f"skip-b{local_batch:07d}-q{_global_chunk(local_batch):09d}.json"


def _batch_is_committed(local_batch: int) -> bool:
    return (_round_dir(local_batch) / "DONE").exists() or _skip_file(local_batch).exists()


def _commit_cursor(next_local_batch: int) -> None:
    _atomic_json(
        _cursor_file,
        {**_progress_identity, "next_local_batch": next_local_batch},
    )


batch = 0
if _cursor_file.exists():
    cursor = json.loads(_cursor_file.read_text())
    mismatches = {
        key: (cursor.get(key), expected)
        for key, expected in _progress_identity.items()
        if cursor.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(
            f"generator progress identity changed for stable tag {TAG!r}: {mismatches}"
        )
    batch = int(cursor.get("next_local_batch", 0))

# Recover the tiny crash windows between publishing a round/skip marker and
# advancing the cursor. A killed generation with neither marker is replayed.
recovered_from = batch
while _batch_is_committed(batch):
    batch += 1
if batch != recovered_from or not _cursor_file.exists():
    _commit_cursor(batch)

print(
    f"[gen] tag={TAG} shard={GEN_RANK}/{GEN_WORLD} prompts={len(rows)} "
    f"chunk={CHUNK} resume_local_batch={batch} reward={REWARD_MODE} "
    f"template={PROMPT_TEMPLATE_NAME} stop={STOP_STRINGS} stop_token_ids={STOP_TOKEN_IDS} "
    f"require_eos={int(REQUIRE_EOS)} maxrl_baseline={int(MAXRL_BASELINE)} "
    f"G={G} max_tokens={MAX_TOKENS} max_model_len={MAX_MODEL_LEN} temp={TEMP} top_k={TOP_K} top_p={TOP_P}",
    flush=True,
)
produced = 0
reloads = 0
while True:
    if STOP_FILE.exists():
        print(f"[gen] stop requested through {STOP_FILE}", flush=True)
        break
    if STOP_AFTER_RELOADS and reloads >= STOP_AFTER_RELOADS:
        print(f"[gen] reached UNIFIED_STOP_AFTER_RELOADS={STOP_AFTER_RELOADS}", flush=True)
        break
    if MAX_BATCHES and produced >= MAX_BATCHES:
        print(f"[gen] reached UNIFIED_MAX_BATCHES={MAX_BATCHES}", flush=True)
        break
    if MAX_SECONDS and time.monotonic() - t0 > MAX_SECONDS:
        print(f"[gen] reached UNIFIED_MAX_SECONDS={MAX_SECONDS:.0f}", flush=True)
        break

    new_dir, new_ver = newest_policy()
    if new_ver > policy_ver or (FORCE_RELOAD and produced > 0):
        reload_policy(new_dir, new_ver)
        policy_dir, policy_ver = new_dir, new_ver
        reloads += 1

    pass_num, selected_indices = select_prompt_shard(
        len(rows), batch, CHUNK, GEN_RANK, GEN_WORLD
    )
    global_chunk = _global_chunk(batch)
    chunks_per_pass = (len(rows) + CHUNK - 1) // CHUNK
    chunk_in_pass = global_chunk % chunks_per_pass
    encoded = [(index, rows[index], _prompt_ids(rows[index])) for index in selected_indices]
    overlong = [
        int(row.get("source_index", index))
        for index, row, ids in encoded
        if len(ids) >= MAX_MODEL_LEN
    ]
    eligible = [(index, row, ids) for index, row, ids in encoded if len(ids) < MAX_MODEL_LEN]
    chunk_indices = [index for index, _row, _ids in eligible]
    chunk_rows = [row for _index, row, _ids in eligible]
    prompt_ids = [ids for _index, _row, ids in eligible]
    budgets = [_response_budget(ids) for ids in prompt_ids]
    generation_start = time.monotonic()
    outs = (
        _generate_groups(
            prompt_ids,
            n=G,
            temperature=TEMP,
            top_p=TOP_P,
            top_k=TOP_K,
            repeat_prompts=REPEAT_PROMPTS,
        )
        if prompt_ids
        else []
    )
    generation_seconds = time.monotonic() - generation_start

    groups = []
    verification_start = time.monotonic()
    for gid, r in enumerate(chunk_rows):
        comps = []
        outputs = (
            [o.outputs[0] for o in outs[gid * G : (gid + 1) * G]]
            if REPEAT_PROMPTS
            else outs[gid].outputs
        )
        for c in outputs:
            # include_stop_str_in_output=True -> c.token_ids already ends with the model's
            # OWN <|im_end|> tokens (fused or not); store verbatim, matching NeMo-RL. The old
            # re-append guard compared raw token-ids and DOUBLED <|im_end|> on BPE-fused
            # boundaries (e.g. "...jar." -> fused ".<" token) in ~7.5% of episodes, training an
            # anti-stop signal (<|im_end|> followed by <|im_end|>).
            ids = list(c.token_ids)
            text = c.text.strip()
            if REQUIRE_EOS:
                # Truncated iff the budget is full and the last token is
                # not EOS (vLLM reports "length" even when the final budget token is EOS).
                finished = not truncated_without_eos(ids, budgets[gid], tok.eos_token_id)
            else:
                finished = c.finish_reason == "stop"
            comps.append((ids, text, finished))
        group_id = int(r.get("source_index", chunk_indices[gid]))
        groups.append((group_id, prompt_ids[gid], comps))
    # Score the whole chunk at once: the code-test reward runs sandboxed
    # subprocesses concurrently, the math verifiers are unchanged in order.
    scored = iter(answers_are_correct(
        [text for _gid, _pids, comps in groups for _ids, text, _fin in comps],
        [r["gold"] for r, (_gid, _pids, comps) in zip(chunk_rows, groups) for _c in comps],
        mode=REWARD_MODE,
    ))
    groups = [
        # REQUIRE_EOS: a truncated completion is wrong (next(scored) is still consumed first).
        (group_id, pids, [(ids, text, finished, next(scored) and (finished or not REQUIRE_EOS))
                          for ids, text, finished in comps])
        for group_id, pids, comps in groups
    ]
    verification_seconds = time.monotonic() - verification_start
    rollout_passk = (
        mean_pass_at_ks(
            [[bool(correct) for _ids, _text, _finished, correct in comps]
             for _group_id, _prompt_ids, comps in groups],
            ks=(k for k in STANDARD_PASS_KS if k <= G),
        )
        if groups
        else {}
    )
    episodes, stats = weight_groups(
        groups, wmax=WMAX, keep_allcorrect=KEEP_ALLCORRECT, hard_filter=HARD_FILTER,
        emit_zero_weight=EMIT_ZERO_WEIGHT, drop_truncated=DROP_TRUNCATED,
        force_truncated_incorrect=FORCE_TRUNCATED_INCORRECT,
        max_group_acc=MAX_GROUP_ACC, easy_keep_frac=EASY_KEEP_FRAC,
        drop_all_wrong=DROP_ALL_WRONG,
        baseline=MAXRL_BASELINE, baseline_eps=MAXRL_EPS,
    )

    out_dir = _round_dir(batch)
    if not episodes:
        _atomic_json(
            _skip_file(batch),
            {
                **_progress_identity,
                "local_batch": batch,
                "global_chunk": global_chunk,
                "pass_num": pass_num,
                "chunk_in_pass": chunk_in_pass,
                "n_selected": len(selected_indices),
                "n_eligible": len(eligible),
                "overlong_source_indices": overlong,
                "reason": "no episodes survived filtering",
            },
        )
        _commit_cursor(batch + 1)
        print(
            f"[gen] batch {batch} q{global_chunk}: nothing survived filtering "
            f"(pass {pass_num}; selected={len(selected_indices)} eligible={len(eligible)} "
            f"overlong={len(overlong)}), committed skip",
            flush=True,
        )
        batch += 1
        produced += 1
        continue
    staging_dir = ROUNDS / f".{out_dir.name}.tmp-{os.getpid()}"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)
    with (staging_dir / "episodes.jsonl").open("w") as f:
        for ep in episodes:
            f.write(json.dumps({"group": ep.group_id, "weight": ep.weight, "text": ep.text}) + "\n")
    manifest = write_episode_round(
        staging_dir, episodes, policy_version=policy_ver, hard_filter=HARD_FILTER,
        normalize_episode_weights=NORMALIZE_WEIGHTS,
        manifest_extra={
            "model": str(policy_dir), "prompts": PROMPTS,
            "prompts_sha256": PROMPTS_SHA256,
            "n_prompts": len(chunk_rows), "n_selected_prompts": len(selected_indices),
            "pass_num": pass_num, "chunk_in_pass": chunk_in_pass,
            "local_batch": batch, "global_chunk": global_chunk,
            "generator_rank": GEN_RANK, "generator_world": GEN_WORLD,
            "G": G, "temp": TEMP, "max_tokens": MAX_TOKENS,
            "max_model_len": MAX_MODEL_LEN,
            "min_effective_max_tokens": min(budgets),
            "max_effective_max_tokens": max(budgets),
            "overlong_source_indices": overlong,
            "top_k": TOP_K, "top_p": TOP_P,
            "reward_mode": REWARD_MODE,
            "prompt_template": PROMPT_TEMPLATE_NAME, "stop_strings": STOP_STRINGS,
            "require_eos": REQUIRE_EOS, "maxrl_baseline": MAXRL_BASELINE,
            "maxrl_eps": MAXRL_EPS if MAXRL_BASELINE else None,
            "policy_version": policy_ver, "keep_allcorrect": KEEP_ALLCORRECT, "wmax": WMAX,
            "max_group_acc": MAX_GROUP_ACC, "easy_keep_frac": EASY_KEEP_FRAC,
            "drop_all_wrong": DROP_ALL_WRONG,
            "normalize_episode_weights": NORMALIZE_WEIGHTS,
            "repeat_prompts": REPEAT_PROMPTS, "stop_token_ids": STOP_TOKEN_IDS,
            "gpu_memory_utilization": GPU_MEM_UTIL,
            "force_truncated_incorrect": FORCE_TRUNCATED_INCORRECT,
            "groups_all_wrong": stats.groups_all_wrong,
            "groups_allcorrect_dropped": stats.groups_allcorrect_dropped,
            "groups_easy_dropped": stats.groups_easy_dropped,
            "groups_easy_kept": stats.groups_easy_kept,
            "n_truncated": stats.n_truncated, "n_wmax_clipped_groups": stats.n_wmax_clipped_groups,
            "pass_at_g": stats.solved / max(stats.n_groups, 1),
            **{f"pass_at_{k}": value for k, value in rollout_passk.items()},
            "generation_seconds": generation_seconds,
            "verification_seconds": verification_seconds,
        },
    )
    if out_dir.exists():
        raise FileExistsError(f"refusing to replace existing round {out_dir}")
    os.replace(staging_dir, out_dir)
    _commit_cursor(batch + 1)
    print(
        f"[gen] batch {batch} q{global_chunk} (pass {pass_num}, v{policy_ver}): "
        f"{len(chunk_rows)}/{len(selected_indices)} eligible prompts | "
        f"pass@{G} {100*stats.solved/max(stats.n_groups,1):.1f}% | {manifest['n_episodes']} eps | "
        f"resp_len {manifest['mean_response_len']:.1f} | gen {generation_seconds:.1f}s "
        f"verify {verification_seconds:.1f}s -> {out_dir}",
        flush=True,
    )
    batch += 1
    produced += 1

print(
    f"[gen] rollout production stopped batches={produced} "
    f"wall={time.monotonic()-t0:.0f}s",
    flush=True,
)
print(
    f"[gen] UNIFIED_GEN_DONE batches={produced} wall={time.monotonic()-t0:.0f}s",
    flush=True,
)
