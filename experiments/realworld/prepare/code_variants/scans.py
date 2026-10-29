"""The two benchmark-overlap scans every D.4 corpus carries (scanner: unified_training.tools.audit_benchmark_overlap).

``exact``: each question's full token sequence (minus its first token); ``16token``: each question's rarest
16-token window, rarity measured on the scanned corpus. Only matches wholly inside one episode count.
"""
from __future__ import annotations

import json
from pathlib import Path


def audit_pair(pool: Path, tokenizer: Path, samples: Path, out_dir: Path, prefix: str = "") -> list[Path]:
    from unified_training.tools.audit_benchmark_overlap import main as audit

    paths = []
    for name, extra in (("exact", []), ("16token", ["--min-match-tokens", "16", "--max-match-tokens", "16"])):
        path = Path(out_dir) / f"{prefix}{name}.json"
        audit(["--tokens", str(Path(pool) / "tokens.bin"), "--index", str(Path(pool) / "index.npy"),
               "--tokenizer", str(tokenizer), "--samples", str(samples), "--output", str(path),
               "--max-matches", "1000", "--fail-on-match-limit", *extra])
        if json.loads(path.read_text())["examples_at_match_limit"]:
            raise SystemExit(4)
        paths.append(path)
    return paths
