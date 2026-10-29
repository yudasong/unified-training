"""experiments/controlled/metrics.py on a synthetic evaluation directory (CPU only)."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from nl_tower import compute

ROOT = Path(__file__).resolve().parents[2]


def test_metrics(tmp_path):
    ev, runs = tmp_path / "eval", tmp_path / "runs"
    per_q = {"1234": [["a", 16, 16], ["b", 16, 0]], "4321": [["a", 16, 8], ["b", 16, 8]]}
    for seed, pq in per_q.items():
        run = f"base_unified_mixed-534M_m8_s{seed}"
        f = ev / "nltw_m" / f"{run}_s6000.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({"sample": {"per_q": pq}}))
        (runs / run / "logs").mkdir(parents=True)
        (runs / run / "logs" / "trainer0.log").write_text("".join(
            f"[unified-hf] iteration {s:>8}/6000 | lm loss: 1.0 | loss tokens: 200000\n" for s in range(10, 6001, 10)))
    (ev / "nltw_m" / "init_mixed-534M.json").write_text(json.dumps({"sample": {"per_q": [["a", 16, 4], ["b", 16, 0]]}}))
    out = tmp_path / "out"
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    subprocess.run([sys.executable, str(ROOT / "experiments/controlled/metrics.py"), "--eval-dir", str(ev),
                    "--runs-dir", str(runs), "--out", str(out)], env=env, check=True)  # fmt: skip
    ck = {(r["run"], r["step"]): r for r in csv.DictReader(open(out / "checkpoints.csv"))}
    assert float(ck[("init_mixed-534M", "0")]["pass@1"]) == pytest.approx(0.125)
    assert float(ck[("init_mixed-534M", "0")]["flops_beyond_pretraining"]) == pytest.approx(
        6 * compute.N_PARAMS * 8147 * 32 * 2048)
    r = ck[("base_unified_mixed-534M_m8_s1234", "6000")]
    assert (r["init"], r["m"], r["seed"], float(r["pass@1"])) == ("mixed-534M", "8", "1234", 0.5)
    resp = 6000 * (200000 - 8 * compute.MID_RESPONSE_TOKENS)
    assert float(r["flops_total"]) == pytest.approx(compute.flops("mixed-534M", "base", 8, 6000, resp))
    means = list(csv.DictReader(open(out / "means.csv")))
    assert len(means) == 1 and means[0]["config"] == "base_unified_mixed-534M_m8"
    assert float(means[0]["pass@1"]) == pytest.approx(0.5)  # (0.5 + 0.5) / 2
    assert float(means[0]["pass@16"]) == pytest.approx((0.5 + 1.0) / 2)  # per model, not pooled
    assert float(means[0]["flops_total"]) == pytest.approx(float(r["flops_total"]))
