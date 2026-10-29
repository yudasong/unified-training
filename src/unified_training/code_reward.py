# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Unit-test reward for code prompts (KodCode-style pytest suites).

The prompt row's ``gold`` is a pytest module that imports the candidate from
``solution`` (``from solution import f``).  The model's last ```python block is
written to ``solution.py`` next to the tests in a private temporary directory
and pytest is run in a subprocess with a wall-clock timeout, an address-space
limit and no network or home directory.  Reward is 1 when pytest exits 0.

Most KodCode test modules are plain ``assert`` functions and importing pytest
costs more than running them, so those run under a small in-tree runner
(code_sandbox/fast_runner.py with the code_sandbox/pytest_shim.py ``pytest``
providing ``raises``, ``approx`` and ``mark.parametrize``) started with
``python -I -S``; modules that use fixtures, capture, other marks, skips or
unittest fall back to real pytest.  ``UNIFIED_CODE_TEST_FAST=0`` forces pytest.

Both paths install code_sandbox/guard.py (an audit hook that blocks deletion and
writes outside the sandbox, subprocesses, signals, network and ctypes) before the
test module is imported, and a run counts as passed only when the process exits 0
AND prints a per-run nonce line that only the runner/conftest can produce after
every test passed, so ``sys.exit(0)`` or a printed fake summary earns nothing.

Tunables (environment):
  UNIFIED_CODE_TEST_TIMEOUT     seconds per test run (default 10)
  UNIFIED_CODE_TEST_WORKERS     concurrent test runs per process (default min(32, cpus))
  UNIFIED_CODE_TEST_PYTHON      interpreter for the sandbox (default: this interpreter)
  UNIFIED_CODE_TEST_PYTHONPATH  extra PYTHONPATH for the sandbox (pytest location)
  UNIFIED_CODE_TEST_MEMORY_MB   RLIMIT_AS for the sandbox (default 4096)
  UNIFIED_CODE_TEST_TMPDIR      parent for the temporary directories (default TMPDIR)
"""

from __future__ import annotations

import os
import re
import resource
import shutil
import signal
import subprocess
import sys
import secrets
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

_FENCE = re.compile(r"```(?:python|py|Python)?[ \t]*\n(.*?)```", re.DOTALL)
_FENCE_OPEN = re.compile(r"```(?:python|py|Python)?[ \t]*\n(.*)\Z", re.DOTALL)


def extract_python(response: str) -> str:
    """The last fenced python block; an unterminated final block; else the raw text."""
    text = str(response)
    blocks = _FENCE.findall(text)
    if blocks:
        return blocks[-1].strip("\n")
    tail = _FENCE_OPEN.search(text)
    if tail:
        return tail.group(1).strip("\n")
    return text.strip()


def _timeout_seconds() -> float:
    return float(os.environ.get("UNIFIED_CODE_TEST_TIMEOUT", "10"))


def _workers() -> int:
    env = os.environ.get("UNIFIED_CODE_TEST_WORKERS")
    if env:
        return max(1, int(env))
    return max(1, min(32, os.cpu_count() or 1))


def _sandbox_python() -> str:
    return os.environ.get("UNIFIED_CODE_TEST_PYTHON") or sys.executable


def _sandbox_env() -> dict[str, str]:
    path = os.environ.get("PATH", "/usr/bin:/bin")
    extra = os.environ.get("UNIFIED_CODE_TEST_PYTHONPATH", "")
    env = {
        "PATH": path,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONIOENCODING": "utf-8",
        "LC_ALL": "C.UTF-8",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        # no proxies, no HF/W&B credentials, no network helpers
    }
    if extra:
        env["PYTHONPATH"] = extra
    return env


def _limit_resources() -> None:  # runs in the child before exec
    mem = int(os.environ.get("UNIFIED_CODE_TEST_MEMORY_MB", "4096")) * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
    except (ValueError, OSError):
        pass
    try:
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024 * 1024, 64 * 1024 * 1024))
    except (ValueError, OSError):
        pass
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ValueError, OSError):
        pass


_SANDBOX_ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "code_sandbox")

# Constructs the in-tree runner cannot reproduce: fixtures (also any test function
# taking arguments), capture/monkeypatch/tmp paths, marks other than parametrize,
# skips and expected failures, unittest, conftest, hypothesis.
_NEEDS_PYTEST = re.compile(
    r"@pytest\.fixture|\bfixture\b|\b(capsys|capfd|monkeypatch|tmp_path|tmpdir)\b"
    r"|pytest\.mark\.(?!parametrize\b)|pytest\.(skip|xfail|importorskip|fail|warns|deprecated_call)\b"
    r"|\bunittest\b|\bconftest\b|\bhypothesis\b|def test_\w*\([^)]*[A-Za-z]",
)


def needs_pytest(test: str) -> bool:
    """True when the test module uses pytest features the fast runner lacks."""
    return bool(_NEEDS_PYTEST.search(test))


def _fast_enabled() -> bool:
    return os.environ.get("UNIFIED_CODE_TEST_FAST", "1") != "0"


def _run_sandboxed(cmd: list[str], workdir: str, timeout: float, nonce: str) -> tuple[bool, str]:
    """Run ``cmd`` in ``workdir``; pass only on exit 0 plus the nonce pass line."""
    env = _sandbox_env()
    env["HOME"] = workdir
    env["TMPDIR"] = workdir
    try:
        proc = subprocess.Popen(
            cmd, cwd=workdir, env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            start_new_session=True, preexec_fn=_limit_resources,
        )
    except OSError as exc:
        return False, f"spawn-failed: {exc}"
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.communicate()
        return False, "timeout"
    text = out.decode("utf-8", errors="replace")
    marker = f"__CODE_REWARD_PASS__ {nonce}"
    if proc.returncode == 0 and any(line.rstrip().endswith(marker) for line in text.splitlines()):
        return True, "passed"
    if proc.returncode == 0:
        return False, "exit 0 without the pass line"
    tail = text.strip().splitlines()
    return False, (tail[-1] if tail else f"rc={proc.returncode}")[:200]


def run_pytest(solution: str, test: str, *, timeout: float | None = None,
               fast: bool | None = None) -> tuple[bool, str]:
    """Run ``test`` (a pytest module importing ``solution``) against ``solution``.

    Returns (passed, short_reason).  Never raises on a bad candidate: syntax
    errors, exceptions, failing asserts, timeouts and memory kills all count as
    a failed run.  Plain assert modules run under the in-tree runner (see the
    module docstring) unless ``fast`` is False or the module needs real pytest.
    """
    timeout = _timeout_seconds() if timeout is None else float(timeout)
    if not solution.strip() or not test.strip():
        return False, "empty"
    if fast is None:
        fast = _fast_enabled()
    use_fast = fast and not needs_pytest(test)
    parent = os.environ.get("UNIFIED_CODE_TEST_TMPDIR") or None
    workdir = tempfile.mkdtemp(prefix="codetest-", dir=parent)
    try:
        with open(os.path.join(workdir, "solution.py"), "w", encoding="utf-8") as stream:
            stream.write(solution.rstrip("\n") + "\n")
        with open(os.path.join(workdir, "test_solution.py"), "w", encoding="utf-8") as stream:
            stream.write(test.rstrip("\n") + "\n")
        nonce = secrets.token_hex(16)
        with open(os.path.join(workdir, ".nonce"), "w", encoding="utf-8") as stream:
            stream.write(nonce + "\n")
        shutil.copyfile(os.path.join(_SANDBOX_ASSETS, "guard.py"), os.path.join(workdir, "_guard.py"))
        if use_fast:
            shutil.copyfile(os.path.join(_SANDBOX_ASSETS, "pytest_shim.py"), os.path.join(workdir, "pytest.py"))
            shutil.copyfile(os.path.join(_SANDBOX_ASSETS, "fast_runner.py"), os.path.join(workdir, "_runner.py"))
            cmd = [_sandbox_python(), "-I", "-S", "_runner.py"]
        else:
            shutil.copyfile(os.path.join(_SANDBOX_ASSETS, "conftest_guard.py"), os.path.join(workdir, "conftest.py"))
            cmd = [
                _sandbox_python(), "-m", "pytest", "-q", "-x", "--no-header", "-s",
                "-p", "no:cacheprovider", "--rootdir", workdir, "test_solution.py",
            ]
        return _run_sandboxed(cmd, workdir, timeout, nonce)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def tests_pass(response: str, test: str, *, timeout: float | None = None) -> bool:
    return run_pytest(extract_python(response), str(test), timeout=timeout)[0]


def batch_tests_pass(responses: Iterable[str], tests: Iterable[str], *,
                     workers: int | None = None, timeout: float | None = None) -> list[bool]:
    """``tests_pass`` over aligned sequences, running the sandboxes concurrently."""
    pairs = list(zip(responses, tests))
    if not pairs:
        return []
    workers = _workers() if workers is None else max(1, int(workers))
    with ThreadPoolExecutor(max_workers=min(workers, len(pairs))) as pool:
        return list(pool.map(lambda p: tests_pass(p[0], p[1], timeout=timeout), pairs))
