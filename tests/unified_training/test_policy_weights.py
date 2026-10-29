"""Eager policy reads must preserve exact serialized tensors and failures."""
import pytest
import torch
from safetensors.torch import save_file
from unified_training.policy_weights import load_policy_shard


def test_read_and_mmap_are_bitwise_identical(tmp_path):
    path = tmp_path / 'weights.safetensors'
    original = {
        'float32': torch.tensor([[-3.0, 0.0, 0.1], [float('inf'), -0.0, 7.25]]),
        'bf16': torch.arange(18).reshape(3, 6).to(torch.bfloat16),
        'integer': torch.tensor([0, -1, 2**40], dtype=torch.int64),
        'scalar': torch.tensor(2.5),
        'empty': torch.empty((0, 4)),
    }
    save_file(original, path)
    before = path.read_bytes()
    mapped, eager = load_policy_shard(path), load_policy_shard(path, 'read')
    assert mapped.keys() == eager.keys() == original.keys()
    for key in original:
        assert mapped[key].dtype == eager[key].dtype == original[key].dtype
        assert mapped[key].shape == eager[key].shape == original[key].shape
        assert mapped[key].device.type == eager[key].device.type == 'cpu'
        assert torch.equal(mapped[key], original[key])
        assert mapped[key].reshape(-1).view(torch.uint8).tolist() == eager[key].reshape(-1).view(torch.uint8).tolist()
    assert path.read_bytes() == before


@pytest.mark.parametrize('mode', ['mmap', 'read'])
def test_bad_shard_fails_in_both_modes(tmp_path, mode):
    path = tmp_path / 'bad.safetensors'
    path.write_bytes(b'not a safetensors file')
    with pytest.raises(Exception):
        load_policy_shard(path, mode)


def test_invalid_mode_fails_before_reading(tmp_path):
    with pytest.raises(ValueError, match='policy read mode'):
        load_policy_shard(tmp_path / 'missing', 'unknown')
