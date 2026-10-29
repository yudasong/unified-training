"""OpenCodeInstruct record contract, hashing and prompt deduplication of the coding pool (App. D.4).

Copied verbatim from the reasoning-SFT module that built the paper's pool (only the parts on the
OpenCodeInstruct path).
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import unicodedata
from typing import Any

VERIFIED_OPENCODE_COMPONENT = "verified-reasoning-sft-opencodeinstruct"

VERIFIED_MAX_CAUSAL_POSITIONS = 2048

V6_MAX_CAUSAL_POSITIONS = VERIFIED_MAX_CAUSAL_POSITIONS

V6_MAX_STORED_TOKENS = V6_MAX_CAUSAL_POSITIONS + 1

V6_CONTRACT_TESTS_SHOWN = 3

V6_CONTRACT_INSTRUCTION = "Write a Python solution that passes these tests:"

SUPERVISED_TOKEN_COUNT_BASIS = "count(token_weight > 0)"

_SPACE_RE = re.compile(r"\s+")

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)

_OUTER_CODE_FENCE_RE = re.compile(
    r"\A\s*```(?:python|py)?\s*\n?(.*?)\n?```\s*\Z",
    flags=re.IGNORECASE | re.DOTALL,
)


def canonical_conversation_bytes(messages: list[dict[str, str]]) -> bytes:
    """Byte-exact canonical JSON used only for exact conversation deduplication."""

    return json.dumps(
        messages,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def conversation_sha256(messages: list[dict[str, str]]) -> str:
    return hashlib.sha256(canonical_conversation_bytes(messages)).hexdigest()


def prompt_sha256(messages: list[dict[str, str]]) -> str:
    return hashlib.sha256(messages[0]["content"].encode("utf-8")).hexdigest()


def normalize_prompt(text: str) -> str:
    """Conservative normalization used for prompt-level exact deduplication."""

    return _SPACE_RE.sub(" ", unicodedata.normalize("NFKC", text).casefold()).strip()


def normalized_prompt_sha256(text: str) -> str:
    return hashlib.sha256(normalize_prompt(text).encode("utf-8")).hexdigest()


def prompt_shingle_hashes(text: str, *, width: int = 5) -> frozenset[int]:
    """Return stable word/punctuation shingles for conservative near dedup."""

    if width <= 0:
        raise ValueError("shingle width must be positive")
    tokens = _TOKEN_RE.findall(normalize_prompt(text))
    if not tokens:
        return frozenset()
    if len(tokens) < width:
        payloads = ["\x1f".join(tokens)]
    else:
        payloads = [
            "\x1f".join(tokens[start : start + width])
            for start in range(len(tokens) - width + 1)
        ]
    return frozenset(
        int.from_bytes(
            hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest(), "big"
        )
        for payload in payloads
    )


def prompt_simhash64(shingles: frozenset[int]) -> int:
    """Compute a deterministic 64-bit SimHash over hashed prompt shingles."""

    if not shingles:
        return 0
    accum = [0] * 64
    for value in shingles:
        for bit in range(64):
            accum[bit] += 1 if value & (1 << bit) else -1
    result = 0
    for bit, score in enumerate(accum):
        if score >= 0:
            result |= 1 << bit
    return result


class PromptDeduplicator:
    """Exact-normalized plus high-overlap lexical prompt deduplicator.

    SimHash bands cheaply identify candidates; the final decision uses exact
    Jaccard overlap over five-token shingles. Candidate recall is probabilistic:
    prompts with no identical 16-bit SimHash band are not compared, even when
    their Jaccard overlap exceeds the threshold. The conservative default is
    designed to remove formatting/paraphrase variants without merging merely
    topically related math or code problems.
    """

    def __init__(self, *, near_jaccard_threshold: float = 0.92) -> None:
        if not 0.0 < near_jaccard_threshold <= 1.0:
            raise ValueError("near_jaccard_threshold must be in (0, 1]")
        self.near_jaccard_threshold = near_jaccard_threshold
        self._normalized: dict[str, tuple[str, str]] = {}
        self._entries: list[tuple[str, str, frozenset[int]]] = []
        self._bands: dict[tuple[int, int], list[int]] = {}

    def find_duplicate(
        self, text: str
    ) -> tuple[str, str, str] | None:
        """Return ``(kind, owner_component, owner_id)`` for an existing prompt."""

        digest = normalized_prompt_sha256(text)
        exact_owner = self._normalized.get(digest)
        if exact_owner is not None:
            return ("normalized_exact_prompt", *exact_owner)

        shingles = prompt_shingle_hashes(text)
        fingerprint = prompt_simhash64(shingles)
        candidates: set[int] = set()
        for band in range(4):
            candidates.update(
                self._bands.get((band, (fingerprint >> (16 * band)) & 0xFFFF), ())
            )
        if len(shingles) >= 8:
            for index in sorted(candidates):
                owner_component, owner_id, owner_shingles = self._entries[index]
                union = len(shingles | owner_shingles)
                similarity = len(shingles & owner_shingles) / union if union else 1.0
                if similarity >= self.near_jaccard_threshold:
                    return ("near_lexical_prompt", owner_component, owner_id)
        return None

    def add(self, text: str, *, component: str, source_id: str) -> None:
        digest = normalized_prompt_sha256(text)
        if digest in self._normalized:
            raise ValueError("cannot register a duplicate normalized prompt")
        shingles = prompt_shingle_hashes(text)
        fingerprint = prompt_simhash64(shingles)
        index = len(self._entries)
        self._normalized[digest] = (component, source_id)
        self._entries.append((component, source_id, shingles))
        for band in range(4):
            key = (band, (fingerprint >> (16 * band)) & 0xFFFF)
            self._bands.setdefault(key, []).append(index)


def strip_outer_code_fence(text: str) -> str:
    """Project one fenced Python answer to direct executable source code."""

    value = text.strip()
    match = _OUTER_CODE_FENCE_RE.fullmatch(value)
    return (match.group(1) if match else value).strip()


def _defined_function_names(tree: ast.AST) -> set[str]:
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _assert_called_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


def opencodeinstruct_contract_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    """Project one fully-passing OpenCodeInstruct row to contract-shaped SFT.

    The eval contract the code benchmarks grade is "define the named function
    that satisfies these tests", so the rendered prompt shows the problem PLUS
    up to ``V6_CONTRACT_TESTS_SHOWN`` of the row's own assert-style unit tests,
    and the response is the direct (unfenced) solution code.  A row is accepted
    only when every executed test passed upstream (``average_test_score == 1``),
    the solution parses and defines at least one function, and the shown tests
    parse and actually call a function the solution defines.
    """

    try:
        score = float(record.get("average_test_score"))
    except (TypeError, ValueError):
        score = float("nan")
    if score != 1.0:
        raise ValueError("record's unit tests did not all pass upstream")
    problem = record.get("input")
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError("record has no problem statement")
    code = strip_outer_code_fence(record.get("output") or "")
    if not code:
        raise ValueError("record has no solution code")
    try:
        solution_tree = ast.parse(code)
    except SyntaxError as error:
        raise ValueError("solution code does not parse") from error
    defined = _defined_function_names(solution_tree)
    if not defined:
        raise ValueError("solution defines no function")
    tests = record.get("unit_tests")
    if isinstance(tests, str):
        try:
            tests = json.loads(tests)
        except json.JSONDecodeError as error:
            raise ValueError("unit_tests field is not valid JSON") from error
    if not isinstance(tests, list) or not tests:
        raise ValueError("record has no unit tests")
    valid_tests: list[str] = []
    for test in tests:
        if not isinstance(test, str) or not test.strip():
            continue
        try:
            test_tree = ast.parse(test.strip())
        except SyntaxError:
            continue
        if not any(isinstance(node, ast.Assert) for node in ast.walk(test_tree)):
            continue
        if not (_assert_called_names(test_tree) & defined):
            continue
        valid_tests.append(test.strip())
        if len(valid_tests) == V6_CONTRACT_TESTS_SHOWN:
            break
    if not valid_tests:
        raise ValueError("no unit test parses and calls a defined function")
    prompt = (
        f"{problem.strip()}\n\n{V6_CONTRACT_INSTRUCTION}\n"
        "```python\n" + "\n".join(valid_tests) + "\n```"
    )
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": code},
    ]


def simple_chatml_chunks(messages: list[dict[str, str]]) -> tuple[str, str]:
    """Render the exact chunks used by assistant-only simple-ChatML training.

    Chunking is intentional: the source packer tokenizes role-labelled chunks
    independently before concatenation, so encoding one combined string could
    change a boundary token for a BPE tokenizer.
    """

    user = messages[0]["content"]
    assistant = messages[1]["content"]
    prompt = (
        f"<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    response = f"{assistant}<|im_end|>\n"
    return prompt, response

