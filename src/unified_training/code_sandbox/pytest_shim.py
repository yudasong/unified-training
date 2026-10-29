"""Minimal ``pytest`` stand-in for plain assert-style test modules.

Written into the sandbox directory as ``pytest.py`` by code_reward's fast path so
``import pytest`` resolves without importing the real package.  Supports
``raises`` (context-manager and callable forms, ``match``), ``approx`` (numbers,
sequences, dicts, ``rel``/``abs``/``nan_ok``) and ``mark.parametrize``; other
marks are identity decorators.  Modules needing fixtures, capture, skips or
unittest are routed to real pytest by code_reward.needs_pytest().
"""
import math as _math
import re as _re


class _Raises:
    def __init__(self, exc, match=None):
        self.exc, self.match, self.value = exc, match, None

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is None:
            raise AssertionError("DID NOT RAISE %r" % (self.exc,))
        if not issubclass(et, self.exc):
            return False
        if self.match is not None and not _re.search(self.match, str(ev)):
            raise AssertionError("exception message %r does not match %r" % (str(ev), self.match))
        self.value = ev
        return True


def raises(exc, *args, match=None, **kwargs):
    if args:  # pytest.raises(Exc, func, *a, **kw) form
        func, fargs = args[0], args[1:]
        try:
            func(*fargs, **kwargs)
        except exc:
            return None
        raise AssertionError("DID NOT RAISE %r" % (exc,))
    return _Raises(exc, match)


class approx:
    def __init__(self, expected, rel=None, abs=None, nan_ok=False):
        self.expected, self.rel, self.abs, self.nan_ok = expected, rel, abs, nan_ok

    def _close(self, a, b):
        if isinstance(a, bool) or isinstance(b, bool):
            return a == b
        if isinstance(b, (list, tuple)) and isinstance(a, (list, tuple)):
            return len(a) == len(b) and all(self._close(x, y) for x, y in zip(a, b))
        if isinstance(b, dict) and isinstance(a, dict):
            return a.keys() == b.keys() and all(self._close(a[k], b[k]) for k in b)
        try:
            a, b = float(a), float(b)
        except (TypeError, ValueError):
            return a == b
        if _math.isnan(a) or _math.isnan(b):
            return self.nan_ok and _math.isnan(a) and _math.isnan(b)
        rel = 1e-6 if self.rel is None else self.rel
        tol = max(rel * abs(b), 1e-12 if self.abs is None else self.abs)
        return abs(a - b) <= tol

    def __eq__(self, other):
        return self._close(other, self.expected)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "approx(%r)" % (self.expected,)


class _Mark:
    def parametrize(self, argnames, argvalues, **kwargs):
        names = [n.strip() for n in argnames.split(",")] if isinstance(argnames, str) else list(argnames)

        def deco(fn):
            fn._parametrize = getattr(fn, "_parametrize", []) + [(names, list(argvalues))]
            return fn
        return deco

    def __getattr__(self, name):
        def deco(*a, **k):
            if len(a) == 1 and callable(a[0]) and not k:
                return a[0]
            return lambda fn: fn
        return deco


mark = _Mark()


def main(*args, **kwargs):
    """Test modules that end with ``pytest.main()`` at top level: a no-op here; the
    runner collects and runs the functions itself."""
    return 0
