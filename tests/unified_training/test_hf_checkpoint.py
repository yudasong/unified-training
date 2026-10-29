from __future__ import annotations

import pytest

from unified_training.hf_checkpoint import (
    BackgroundCheckpointWriter,
    commit_checkpoint_pair,
    validate_resume_pair,
)


def test_background_checkpoint_writer_waits_before_next_write():
    writer = BackgroundCheckpointWriter()
    events = []
    writer.start(lambda: events.append("first"), description="first snapshot")
    writer.start(lambda: events.append("second"), description="second snapshot")
    writer.wait()
    writer.wait()
    assert events == ["first", "second"]


@pytest.mark.parametrize("next_operation", ["wait", "start"])
def test_background_checkpoint_writer_propagates_failure(next_operation):
    writer = BackgroundCheckpointWriter()
    failure = OSError("quota exceeded")

    def fail():
        raise failure

    writer.start(fail, description="publish-v0000016 to /policies/v0000016")
    with pytest.raises(RuntimeError, match="publish-v0000016.*quota exceeded") as caught:
        if next_operation == "wait":
            writer.wait()
        else:
            writer.start(lambda: pytest.fail("must not publish after failure"), description="next")
    assert caught.value.__cause__ is failure
    # A handled failure cannot silently disappear on a later checkpoint save.
    with pytest.raises(RuntimeError, match="quota exceeded"):
        writer.wait()


def _staged_pair(tmp_path, step: int):
    staged_policy = tmp_path / f".v{step:07d}.tmp"
    staged_policy.mkdir()
    (staged_policy / "config.json").write_text("{}")
    staged_optimizer = tmp_path / f".opt_{step:07d}.pt.tmp"
    staged_optimizer.write_bytes(b"optimizer")
    return staged_policy, staged_optimizer


def test_checkpoint_pair_commit_is_resumable_and_generator_visible(tmp_path):
    policies = tmp_path / "policies"
    state = tmp_path / "trainer_hf"
    policies.mkdir()
    state.mkdir()
    staged_policy, staged_optimizer = _staged_pair(tmp_path, 7)
    final_policy = policies / "v0000007"
    final_optimizer = state / "opt_0000007.pt"
    latest = state / "latest_step.txt"

    commit_checkpoint_pair(
        staged_policy=staged_policy,
        staged_optimizer=staged_optimizer,
        final_policy=final_policy,
        final_optimizer=final_optimizer,
        latest_file=latest,
        step=7,
    )

    assert latest.read_text() == "7"
    assert final_optimizer.read_bytes() == b"optimizer"
    assert (final_policy / "config.json").is_file()
    assert (final_policy / "DONE").is_file()
    assert validate_resume_pair(policies, state, latest) == 7


def test_resume_repairs_done_only_after_complete_pair(tmp_path):
    policies = tmp_path / "policies"
    state = tmp_path / "trainer_hf"
    policy = policies / "v0000011"
    policy.mkdir(parents=True)
    state.mkdir()
    (policy / "config.json").write_text("{}")
    (state / "opt_0000011.pt").write_bytes(b"optimizer")
    latest = state / "latest_step.txt"
    latest.write_text("11")

    assert not (policy / "DONE").exists()
    assert validate_resume_pair(policies, state, latest) == 11
    assert (policy / "DONE").is_file()


@pytest.mark.parametrize("missing", ["policy", "optimizer"])
def test_resume_rejects_incomplete_committed_pair(tmp_path, missing):
    policies = tmp_path / "policies"
    state = tmp_path / "trainer_hf"
    policies.mkdir()
    state.mkdir()
    if missing != "policy":
        (policies / "v0000005").mkdir()
    if missing != "optimizer":
        (state / "opt_0000005.pt").write_bytes(b"optimizer")
    latest = state / "latest_step.txt"
    latest.write_text("5")

    with pytest.raises(RuntimeError, match="incomplete pair"):
        validate_resume_pair(policies, state, latest)


def test_checkpoint_commit_refuses_to_replace_published_policy(tmp_path):
    policies = tmp_path / "policies"
    state = tmp_path / "trainer_hf"
    final_policy = policies / "v0000003"
    final_policy.mkdir(parents=True)
    (final_policy / "DONE").touch()
    state.mkdir()
    staged_policy, staged_optimizer = _staged_pair(tmp_path, 3)

    with pytest.raises(FileExistsError, match="published policy"):
        commit_checkpoint_pair(
            staged_policy=staged_policy,
            staged_optimizer=staged_optimizer,
            final_policy=final_policy,
            final_optimizer=state / "opt_0000003.pt",
            latest_file=state / "latest_step.txt",
            step=3,
        )
