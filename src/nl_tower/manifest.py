"""Check rebuilt nl_tower data against the recorded artifacts (artifacts.json next to this module).

The record lists every published data artifact: path relative to the work directory, bytes, md5, row count and first
qids. Each entry has a ``check``:

* ``md5``: the rebuilt file must be byte-identical;
* ``audit_result``: a contamination audit; its ``new`` block must equal the recorded ``audit_result``;
* ``anchor_manifest``: an anchor round manifest; the recorded ``expected`` keys must be equal (provenance keys such as
  ``source_jsonl`` and ``tokenizer`` differ by construction).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def md5(path: str | Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def describe(path: str | Path) -> dict[str, Any]:
    """bytes, md5 and, for JSONL / JSON lists, rows and the first five qids."""
    p = Path(path)
    d: dict[str, Any] = {"bytes": p.stat().st_size, "md5": md5(p)}
    if p.name.endswith(".jsonl"):
        n, first = 0, []
        with open(p) as f:
            for line in f:
                if len(first) < 5:
                    first.append(json.loads(line).get("qid"))
                n += 1
        d["rows"], d["first_qids"] = n, first
    elif p.name.endswith(".json"):
        with open(p) as f:
            obj = json.load(f)
        if isinstance(obj, list):
            d["rows"], d["first_qids"] = len(obj), [e.get("qid") for e in obj[:5]]
    return d


def check_entry(entry: dict[str, Any], path: Path) -> tuple[str, str]:
    """-> (status, detail); status is SAME, DIFF or MISSING."""
    mode = entry.get("check", "md5")
    if not path.exists():
        return "MISSING", str(path)
    if mode == "md5":
        got = md5(path)
        return ("SAME" if got == entry["md5"] else "DIFF"), f"md5 {got} vs {entry['md5']}"
    with open(path) as f:
        got = json.load(f)
    if mode == "audit_result":
        same = got.get("new") == entry["audit_result"]
        return ("SAME" if same else "DIFF"), f"audit result {got.get('new')}"
    if mode == "anchor_manifest":
        diff = {k: (got.get(k), v) for k, v in entry["expected"].items() if got.get(k) != v}
        return ("SAME" if not diff else "DIFF"), f"differing keys {diff}" if diff else "expected keys equal"
    raise ValueError(f"unknown check {mode!r}")


def check(record: dict[str, Any], root: str | Path, groups: Iterable[str] | None = None) -> list[tuple[str, str, str, str]]:
    """-> [(status, group, path, detail)] for every recorded entry of ``groups`` (default: all)."""
    want = set(groups) if groups else None
    out = []
    for group, items in record["groups"].items():
        if want is not None and group not in want:
            continue
        base = Path(root)
        for it in items:
            status, detail = check_entry(it, base / it["path"])
            out.append((status, group, it["path"], detail))
    return out
