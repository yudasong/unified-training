"""Run a plain assert-style test module against ``solution.py`` without pytest.

Copied into the sandbox directory as ``_runner.py`` and started with
``python -I -S`` (no site, no user paths).  Reads and removes the run nonce,
installs the audit-hook guard, collects module-level ``test_*`` functions and
``Test*`` classes, expands ``mark.parametrize`` from the shim, runs every case
and prints the nonce pass line only when at least one test ran and none raised.
"""
import importlib.util
import inspect
import os
import sys

sys.path.insert(0, ".")
import _guard  # noqa: E402  (copied next to this file)

_NONCE = _guard.read_nonce(os.getcwd())
_guard.install(os.getcwd())

spec = importlib.util.spec_from_file_location("test_solution", "test_solution.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def cases(fn):
    params = getattr(fn, "_parametrize", None)
    if not params:
        return [{}]
    out = [{}]
    for names, values in params:
        new = []
        for base in out:
            for v in values:
                v = v if len(names) > 1 else (v,)
                d = dict(base)
                d.update(zip(names, v))
                new.append(d)
        out = new
    return out


ran = 0
for name, obj in list(vars(mod).items()):
    if name.startswith("test_") and callable(obj) and not inspect.isclass(obj):
        for kw in cases(obj):
            obj(**kw)
            ran += 1
    elif name.startswith("Test") and inspect.isclass(obj):
        inst = obj()
        for mname in dir(obj):
            if mname.startswith("test_") and callable(getattr(inst, mname)):
                for kw in cases(getattr(obj, mname)):
                    getattr(inst, mname)(**kw)
                    ran += 1
if ran == 0:
    print("no tests collected")
    sys.exit(5)
print("%d passed" % ran)
print(_guard.pass_line(_NONCE))
