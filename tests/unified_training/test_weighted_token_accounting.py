from pathlib import Path

import numpy as np

from unified_training.episodes import RoundWriter, load_round, mean_response_tokens


def test_token_mix_counts_each_trained_position_once(tmp_path: Path) -> None:
    writer = RoundWriter(tmp_path / "anchor")
    writer.add(
        np.array([10, 11, 12, 13, 14], dtype=np.int32),
        np.array([0, 0, 0, 0.125, 9.0], dtype=np.float32),
        prompt_len=1,
        reward=0.0,
        weight=1.0,
        policy_version=-1,
        group_id=0,
    )
    writer.add(
        np.array([20, 21, 22], dtype=np.int32),
        np.array([0, 200.0, 0], dtype=np.float32),
        prompt_len=1,
        reward=0.0,
        weight=1.0,
        policy_version=-1,
        group_id=0,
    )
    manifest = writer.finalize({"kind": "anchor"})

    round_ = load_round(tmp_path / "anchor")
    assert mean_response_tokens([round_], [(0, 0), (0, 1)]) == 1.5
    assert manifest["n_weighted_tokens"] == 3
    assert manifest["n_supervised_tokens"] == 3
    assert manifest["supervised_token_count_basis"] == "count(token_weight > 0)"
