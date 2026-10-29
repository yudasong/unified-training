"""CPU weight loading options for immutable, published policy snapshots."""
from pathlib import Path

POLICY_READ_MODES = ("mmap", "read")


def load_policy_shard(path, read_mode="mmap"):
    """Load identical CPU tensors via mmap or an eager sequential file read.

    Eager reads avoid demand-paging shared storage during GPU refits. They use
    additional CPU memory (file bytes plus decoded tensors), bounded to one
    shard per worker. No snapshot or cache file is modified. The caller must
    only use finalized immutable snapshots and release each shard after refit.
    """
    if read_mode not in POLICY_READ_MODES:
        raise ValueError(f"policy read mode must be one of {POLICY_READ_MODES}, got {read_mode!r}")
    from safetensors.torch import load, load_file

    if read_mode == "mmap":
        return load_file(path)
    return load(Path(path).read_bytes())
