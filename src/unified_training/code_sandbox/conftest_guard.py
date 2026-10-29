"""conftest.py for the real-pytest sandbox path (copied into the sandbox directory).

Loaded by pytest before the test module: reads and removes the run nonce,
installs the audit-hook guard, and prints the nonce pass line at session end
only when the session collected tests and every one of them passed.
"""
import os
import sys

sys.path.insert(0, os.getcwd())
import _guard  # noqa: E402

_NONCE = _guard.read_nonce(os.getcwd())
_guard.install(os.getcwd())


def pytest_sessionfinish(session, exitstatus):
    if int(exitstatus) == 0 and session.testscollected > 0 and session.testsfailed == 0:
        print("\n" + _guard.pass_line(_NONCE), flush=True)  # own line even after the progress dots
