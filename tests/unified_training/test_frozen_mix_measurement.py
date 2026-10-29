"""Frozen launch measurements must avoid redundant live-pool payload reads."""

import json

import numpy as np
import pytest

from unified_training.dataset import build_unified_datasets
from unified_training.episodes import RoundWriter


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("basis", ["supervised_tokens", "weight", "response_tokens"])
def test_frozen_mix_skips_primary_measurement(tmp_path, monkeypatch, streaming, basis):
    from unified_training import episodes

    def write_round(path, weight):
        writer = RoundWriter(path)
        for i in range(4):
            writer.add(
                np.array([1, 2, 3, 4], dtype=np.int32),
                np.array([0, weight, weight, weight], dtype=np.float32),
                prompt_len=1, reward=1, weight=weight, policy_version=1, group_id=i,
            )
        writer.finalize()

    primary = tmp_path / "primary"
    anchor = tmp_path / "anchor"
    write_round(primary / "round0", 2)
    write_round(anchor, 1)
    state = tmp_path / "mix.json"
    kwargs = dict(
        episodes_dirs=[str(primary)], train_samples=32, valid_samples=0,
        sequence_length=8, streaming=streaming, mix_recalc="launch",
        mix_basis=basis, anchor_dirs=[str(anchor)], anchor_token_fraction=0.1,
        mix_state_manifest=state,
    )
    original, _, _ = build_unified_datasets(**kwargs, persist_mix_state=True)
    saved = json.loads(state.read_text())
    original_measure = episodes.mean_response_tokens if basis == "supervised_tokens" else episodes.mean_episode_mass
    measured = []

    def measure(rounds, refs, **options):
        measured.extend(round_.round_dir for round_ in rounds)
        assert all(round_.round_dir == anchor for round_ in rounds)
        return original_measure(rounds, refs, **options)

    name = "mean_response_tokens" if basis == "supervised_tokens" else "mean_episode_mass"
    monkeypatch.setattr(episodes, name, measure)
    resumed, _, _ = build_unified_datasets(**kwargs)
    assert measured == [anchor]
    assert resumed.anchor_fraction == original.anchor_fraction
    assert json.loads(state.read_text()) == saved
    for index in range(32):
        left, right = original[index], resumed[index]
        for key in left:
            assert left[key].equal(right[key])


def test_frozen_mix_still_rejects_basis_change(tmp_path):
    state = tmp_path / "mix.json"
    state.write_text(json.dumps({"mix_basis": "weight", "rl_tokens_per_ep": 3.0}))
    with pytest.raises(ValueError, match="frozen mix state was measured on basis"):
        build_unified_datasets(
            episodes_dirs=[str(tmp_path / "missing")], train_samples=32,
            valid_samples=0, mix_basis="response_tokens", mix_state_manifest=state,
        )
