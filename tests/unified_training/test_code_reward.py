# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import os
import sys

import pytest

from unified_training import code_reward, math_reward

TEST = """from solution import add

def test_add_small():
    assert add(1, 2) == 3

def test_add_negative():
    assert add(-1, 1) == 0
"""


@pytest.fixture(autouse=True)
def _sandbox_can_import_pytest(monkeypatch):
    # The sandbox interpreter must find pytest wherever this test session found it.
    monkeypatch.setenv("UNIFIED_CODE_TEST_PYTHONPATH", os.pathsep.join(p for p in sys.path if p))
    monkeypatch.setenv("UNIFIED_CODE_TEST_PYTHON", sys.executable)


def test_extract_python_takes_the_last_fenced_block():
    text = "Plan:\n```python\nx = 1\n```\nFixed:\n```python\ndef add(a, b):\n    return a + b\n```\n"
    assert code_reward.extract_python(text) == "def add(a, b):\n    return a + b"


def test_extract_python_accepts_unterminated_block_and_raw_text():
    assert code_reward.extract_python("```python\ndef f():\n    pass") == "def f():\n    pass"
    assert code_reward.extract_python("def f():\n    pass\n") == "def f():\n    pass"


def test_passing_solution_scores_one():
    response = "Here it is:\n```python\ndef add(a, b):\n    return a + b\n```"
    assert code_reward.tests_pass(response, TEST)
    assert math_reward.answer_is_correct(response, TEST, mode="code_tests")


def test_wrong_solution_syntax_error_and_missing_function_score_zero():
    assert not code_reward.tests_pass("```python\ndef add(a, b):\n    return a - b\n```", TEST)
    assert not code_reward.tests_pass("```python\ndef add(a, b)\n    return a + b\n```", TEST)
    assert not code_reward.tests_pass("```python\ndef plus(a, b):\n    return a + b\n```", TEST)
    assert not code_reward.tests_pass("", TEST)


def test_infinite_loop_is_killed_by_the_timeout():
    passed, reason = code_reward.run_pytest(
        "def add(a, b):\n    while True:\n        pass\n", TEST, timeout=3
    )
    assert not passed
    assert reason == "timeout"


def test_batch_scoring_keeps_order_and_matches_single_calls():
    good = "```python\ndef add(a, b):\n    return a + b\n```"
    bad = "```python\ndef add(a, b):\n    return 0\n```"
    responses = [good, bad, good, "", bad]
    out = math_reward.answers_are_correct(responses, [TEST] * 5, mode="code_tests")
    assert out == [True, False, True, False, False]


def test_batch_scoring_for_math_modes_is_unchanged():
    out = math_reward.answers_are_correct(
        ["so #### 1,024", "so 1023"], ["1024", "1024"], mode="last_number"
    )
    assert out == [True, False]


SHIM_TEST = """import pytest
from solution import div

def test_div_ok():
    assert div(6, 3) == 2

def test_div_zero():
    with pytest.raises(ZeroDivisionError):
        div(1, 0)

def test_div_approx():
    assert div(1, 3) == pytest.approx(0.3333333, rel=1e-5)
"""
PARAM_TEST = """import pytest
from solution import div

@pytest.mark.parametrize("a,b,expected", [(8, 2, 4), (9, 3, 3)])
def test_div_param(a, b, expected):
    assert div(a, b) == expected
"""
DIV_OK = "```python\ndef div(a, b):\n    return a / b\n```"
DIV_BAD = "```python\ndef div(a, b):\n    return a // b if b else 0\n```"


def test_needs_pytest_detection():
    assert not code_reward.needs_pytest(TEST)
    assert code_reward.needs_pytest("import pytest\n@pytest.fixture\ndef x():\n    return 1\n\ndef test_a(x):\n    assert x == 1\n")
    assert code_reward.needs_pytest("def test_out(capsys):\n    print('hi')\n    assert capsys.readouterr().out == 'hi\\n'\n")
    assert code_reward.needs_pytest("import pytest\n\ndef test_s():\n    pytest.skip('no')\n")
    assert code_reward.needs_pytest("import unittest\n\nclass T(unittest.TestCase):\n    def test_a(self):\n        pass\n")


def test_fast_runner_agrees_with_pytest_on_plain_asserts():
    good = "def add(a, b):\n    return a + b\n"
    bad = "def add(a, b):\n    return a - b\n"
    for solution, expected in ((good, True), (bad, False)):
        assert code_reward.run_pytest(solution, TEST, fast=True)[0] is expected
        assert code_reward.run_pytest(solution, TEST, fast=False)[0] is expected


def test_fast_runner_reports_syntax_error_and_no_tests_as_failures():
    assert not code_reward.run_pytest("def add(a, b)\n    return a + b\n", TEST, fast=True)[0]
    assert not code_reward.run_pytest("def add(a, b):\n    return a + b\n", "from solution import add\n", fast=True)[0]


def test_shim_handles_raises_and_approx_like_pytest():
    assert not code_reward.needs_pytest(SHIM_TEST)
    assert code_reward.tests_pass(DIV_OK, SHIM_TEST)
    assert not code_reward.tests_pass(DIV_BAD, SHIM_TEST)  # no ZeroDivisionError raised
    assert code_reward.run_pytest(code_reward.extract_python(DIV_OK), SHIM_TEST, fast=False)[0]
    assert not code_reward.run_pytest(code_reward.extract_python(DIV_BAD), SHIM_TEST, fast=False)[0]


def test_parametrized_tests_route_to_real_pytest_and_pass():
    # test functions with parameters are conservatively sent to pytest (fixtures look the same)
    assert code_reward.needs_pytest(PARAM_TEST)
    assert code_reward.tests_pass(DIV_OK, PARAM_TEST)
    assert not code_reward.tests_pass("```python\ndef div(a, b):\n    return 0\n```", PARAM_TEST)


def test_fixture_module_falls_back_to_real_pytest():
    fixture_test = "import pytest\nfrom solution import add\n\n@pytest.fixture\ndef nums():\n    return (2, 3)\n\ndef test_add(nums):\n    assert add(*nums) == 5\n"
    assert code_reward.needs_pytest(fixture_test)
    assert code_reward.tests_pass("```python\ndef add(a, b):\n    return a + b\n```", fixture_test)


def test_fast_flag_off_forces_pytest(monkeypatch):
    monkeypatch.setenv("UNIFIED_CODE_TEST_FAST", "0")
    assert code_reward.tests_pass("```python\ndef add(a, b):\n    return a + b\n```", TEST)


def _write_probe_target(tmp_path):
    target = tmp_path / "victim.txt"
    target.write_text("keep me\n")
    return target


@pytest.mark.parametrize("fast", [True, False])
def test_exit_zero_and_printed_summary_do_not_count_as_passing(fast):
    exit_sol = "import sys\nsys.exit(0)\n"
    fake_sol = "print('2 passed in 0.01s')\nprint('__CODE_REWARD_PASS__ deadbeef')\nimport os\nos._exit(0)\n"
    assert not code_reward.run_pytest(exit_sol, TEST, fast=fast)[0]
    assert not code_reward.run_pytest(fake_sol, TEST, fast=fast)[0]


@pytest.mark.parametrize("fast", [True, False])
def test_guard_blocks_deletion_and_writes_outside_the_sandbox(tmp_path, fast):
    victim = _write_probe_target(tmp_path)
    delete_sol = f"import os\nos.remove({str(victim)!r})\n\ndef add(a, b):\n    return a + b\n"
    rmtree_sol = f"import shutil\nshutil.rmtree({str(tmp_path)!r}, ignore_errors=True)\n\ndef add(a, b):\n    return a + b\n"
    write_sol = f"open({str(tmp_path / 'new.txt')!r}, 'w').write('x')\n\ndef add(a, b):\n    return a + b\n"
    for sol in (delete_sol, rmtree_sol, write_sol):
        passed, reason = code_reward.run_pytest(sol, TEST, fast=fast)
        assert not passed, reason
    assert victim.read_text() == "keep me\n"
    assert not (tmp_path / "new.txt").exists()


@pytest.mark.parametrize("fast", [True, False])
def test_guard_blocks_subprocess_and_sockets(fast):
    sub_sol = "import subprocess\nsubprocess.run(['true'])\n\ndef add(a, b):\n    return a + b\n"
    sock_sol = "import socket\nsocket.socket()\n\ndef add(a, b):\n    return a + b\n"
    for sol in (sub_sol, sock_sol):
        assert not code_reward.run_pytest(sol, TEST, fast=fast)[0]


@pytest.mark.parametrize("fast", [True, False])
def test_guard_allows_files_inside_the_sandbox(fast):
    sol = "open('scratch.txt', 'w').write('x')\nimport os\nos.remove('scratch.txt')\n\ndef add(a, b):\n    return a + b\n"
    assert code_reward.run_pytest(sol, TEST, fast=fast)[0]
