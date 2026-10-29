"""Strict admission waits and fails collectively before returning model inputs."""
from datetime import timedelta
import json
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from unified_training.dataset import NoFreshEpisodes
from unified_training.freshness_batch import fetch_fresh_batch, _batch_status, _global_max


class Stream:
    def __init__(self):
        self.scans = 0
        self._eligible_refs = [(0, 0)]
    def set_trainer_step(self, step):
        self.step = step
    def rescan(self):
        self.scans += 1


def row(version, rl=True):
    return {"source_policy_version": torch.tensor(version), "sampling_filter": torch.tensor(rl)}


def test_actual_rows_enforce_inclusive_cutoff_and_exclude_anchors():
    assert _batch_status([[row(6), row(10), row(0, False)]], 10, 4) == 0
    assert _batch_status([[row(5)]], 10, 4) == 1
    for version in (-1, 11):
        with pytest.raises(ValueError, match="invalid consumed"):
            _batch_status([[row(version)]], 10, 4)


def _worker(rank, rendezvous, output):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=45))
    results = {}
    try:
        stream, waits = Stream(), []
        def fetch():
            if rank == 1 and stream.scans < 3:
                raise NoFreshEpisodes("producer has not committed a fresh round")
            return [[row(10 if rank == 0 else 6)]]
        batch, blocked = fetch_fresh_batch(stream, fetch, step=10, max_age=4,
            device="cpu", timeout_seconds=10, poll_seconds=.01,
            on_wait=lambda: waits.append(1))
        results["recovered"] = dict(scans=stream.scans, waits=len(waits),
                                    version=int(batch[0][0]["source_policy_version"]), blocked=blocked)
        for case in ("timeout", "bad_version", "stop"):
            stream = Stream()
            def fetch_bad():
                if case == "bad_version" and rank == 1:
                    return [[row(-1)]]
                raise NoFreshEpisodes("empty")
            try:
                batch, blocked = fetch_fresh_batch(stream, fetch_bad, step=10, max_age=4,
                    device="cpu", timeout_seconds=.1, poll_seconds=.01,
                    stop_requested=lambda: bool(_global_max(int(case == "stop" and rank == 1 and stream.scans >= 1), "cpu")))
                assert case == "stop" and batch is None
                results[case] = "stopped"
            except TimeoutError:
                results[case] = "timeout"
            except RuntimeError:
                results[case] = "error"
        Path(output + str(rank)).write_text(json.dumps(results))
    finally:
        dist.destroy_process_group()


def test_one_rank_waits_then_all_recover_timeout_fail_and_stop(tmp_path):
    mp.spawn(_worker, args=((tmp_path / "rdzv").as_uri(), str(tmp_path / "rank")), nprocs=2, join=True)
    for rank in range(2):
        result = json.loads((tmp_path / f"rank{rank}").read_text())
        assert result["recovered"]["scans"] == 3
        assert result["recovered"]["waits"] == 1
        assert result["recovered"]["blocked"] > 0
        assert result["timeout"] == "timeout"
        assert result["bad_version"] == "error"
        assert result["stop"] == "stopped"


def test_empty_primary_blocks_even_an_anchor_only_mixer_batch():
    stream = Stream()
    stream._eligible_refs = []
    fetched = []
    def fetch():
        fetched.append(True)
        return [[row(0, False)]]
    batch, blocked = fetch_fresh_batch(stream, fetch, step=10, max_age=4,
        device="cpu", timeout_seconds=1, poll_seconds=.001,
        stop_requested=lambda: stream.scans >= 1)
    assert batch is None and not fetched
    assert blocked > 0


def test_round_files_not_yet_visible_are_a_wait_not_an_error():
    """A short tokens.bin (index landed before the payload) is retried, not fatal."""
    stream = Stream()
    calls = []
    def fetch():
        calls.append(True)
        if len(calls) == 1:
            raise ValueError("mmap length is greater than file size")
        if len(calls) == 2:
            raise FileNotFoundError("rounds/r0001234/tokens.bin")
        return [[row(10)]]
    batch, blocked = fetch_fresh_batch(stream, fetch, step=10, max_age=4,
        device="cpu", timeout_seconds=2, poll_seconds=.001)
    assert int(batch[0][0]["source_policy_version"]) == 10
    assert stream.scans == 3 and blocked > 0
    with pytest.raises(RuntimeError, match="unrelated"):
        fetch_fresh_batch(Stream(), lambda: (_ for _ in ()).throw(ValueError("unrelated")),
                          step=10, max_age=4, device="cpu", timeout_seconds=1, poll_seconds=.001)
