"""Consumed-row provenance and global age accounting; CPU/Gloo is sufficient."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from unified_training.dataset import WeightedEpisodeDataset
from unified_training.episodes import RoundWriter, load_round
from unified_training.hf_trainer import (
    _stack, consumed_policy_age_metrics, group_rows_by_length,
)


def _row(version, *, rl=True):
    result = {
        "tokens": torch.tensor([1, 2, 0, 0]),
        "labels": torch.tensor([2, 3, 0, 0]),
        "loss_mask": torch.zeros(4),  # zero-weight responses must still count
        "denom_mask": torch.tensor([1., 1., 0., 0.]),
        "sampling_filter": torch.tensor(rl),
    }
    if version is not None:
        result["source_policy_version"] = torch.tensor(version)
    return result


def test_episode_version_survives_grouping_without_entering_model_batch(tmp_path):
    writer = RoundWriter(tmp_path / "round")
    for version, weight, length in [(80, 0., 4), (96, 1., 6)]:
        writer.add(np.arange(length, dtype=np.int32),
                   np.array([0.] + [weight] * (length - 1), dtype=np.float32),
                   prompt_len=1, reward=weight, weight=weight,
                   policy_version=version, group_id=version)
    # Episode-level origin, not this deliberately different round-level value.
    writer.finalize({"policy_version": 999})
    ds = WeightedEpisodeDataset([load_round(tmp_path / "round")], [(0, 0), (0, 1)],
                                seq_length=8, total_samples=2)
    rows = [ds[0], ds[1]]
    assert sorted(int(r["source_policy_version"]) for r in rows) == [80, 96]
    grouped = group_rows_by_length(rows, 2)
    actual = _stack(grouped[0], "cpu", trim=True)
    plain = [{k: v for k, v in r.items() if k != "source_policy_version"} for r in grouped[0]]
    expected = _stack(plain, "cpu", trim=True)
    assert actual.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key])
    result = consumed_policy_age_metrics(grouped, trainer_step=100, device="cpu",
                                       max_policy_age_steps=4)
    assert result["optimizer_step"] == 101
    assert result["mean_steps"] == 12
    assert result["fraction_gt_configured_limit"] == .5
    assert result["known_rows"] == 2


def test_unknown_anchor_and_future_versions_are_not_reported_as_fresh():
    result = consumed_policy_age_metrics(
        [[_row(None), _row(-3), _row(100, rl=False), _row(101)]],
        trainer_step=100, device="cpu")
    assert result["rl_rows"] == 3
    assert result["unknown_rows"] == 2
    assert result["anchor_rows_excluded"] == 1
    assert result["future_policy_rows"] == 1
    assert result["min_steps"] == -1
    empty = consumed_policy_age_metrics([[_row(None), _row(0, rl=False)]],
                                       trainer_step=100, device="cpu")
    assert empty["known_rows"] == 0
    assert "mean_steps" not in empty


def _global_worker(rank, rendezvous, output):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        rows = [_row(80), _row(96)] if rank == 0 else [_row(1, rl=False), _row(None)]
        result = consumed_policy_age_metrics([rows], trainer_step=100, device="cpu",
                                           max_policy_age_steps=4)
        Path(output + str(rank)).write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


def test_age_summary_uses_both_data_parallel_ranks(tmp_path):
    mp.spawn(_global_worker, args=((tmp_path / "rdzv").as_uri(), str(tmp_path / "rank")),
             nprocs=2, join=True)
    a = json.loads((tmp_path / "rank0").read_text())
    b = json.loads((tmp_path / "rank1").read_text())
    assert a == b
    assert a["rl_rows"] == 3 and a["known_rows"] == 2
    assert a["unknown_rows"] == 1 and a["anchor_rows_excluded"] == 1
    assert a["mean_steps"] == 12 and a["p50_steps"] == 12
    assert a["p95_steps"] == pytest.approx(19.2)
    assert a["fraction_gt_4"] == .5 and a["fraction_gt_16"] == .5
