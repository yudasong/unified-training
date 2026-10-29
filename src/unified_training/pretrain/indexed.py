# Writer adapted from NVIDIA Nemotron (src/nemotron/data_prep/formats/indexed_dataset.py), Apache-2.0.
"""Megatron ``.bin``/``.idx`` indexed datasets (format version 1): a streaming writer and a reader."""

from __future__ import annotations

import struct
from typing import BinaryIO

import numpy as np

_INDEX_HEADER = b"MMIDIDX\x00\x00"
DTYPE_CODES = {np.dtype(np.int32): 4, np.dtype(np.int64): 5, np.dtype(np.uint16): 8}
CODE_DTYPES = {code: dtype for dtype, code in DTYPE_CODES.items()}


class IndexedDatasetBuilder:
    """Append documents to a ``.bin`` stream; ``write_index`` writes the matching ``.idx``."""

    def __init__(self, bin_file: BinaryIO, dtype: np.dtype = np.uint16, buffer_size: int = 64 * 1024):
        self.bin_file = bin_file
        self.dtype = np.dtype(dtype)
        self.sequence_lengths: list[int] = []
        self.total_tokens = 0
        self._buffer = bytearray()
        self._buffer_size = buffer_size

    def add_document(self, tokens: list[int]) -> None:
        if not tokens:
            return
        data = np.asarray(tokens, dtype=self.dtype).tobytes(order="C")
        self._buffer.extend(data)
        self.sequence_lengths.append(len(tokens))
        self.total_tokens += len(tokens)
        if len(self._buffer) >= self._buffer_size:
            self.flush()

    def flush(self) -> None:
        if self._buffer:
            self.bin_file.write(self._buffer)
            self._buffer.clear()

    def finalize(self) -> None:
        self.flush()

    def write_index(self, idx_file: BinaryIO) -> None:
        count = len(self.sequence_lengths)
        lengths = np.array(self.sequence_lengths, dtype=np.int64)
        pointers = np.zeros(count, dtype=np.int64)
        if count > 1:
            pointers[1:] = np.cumsum(lengths[:-1] * self.dtype.itemsize)
        documents = np.arange(count + 1, dtype=np.int64)  # one sequence per document
        idx_file.write(_INDEX_HEADER)
        idx_file.write(struct.pack("<Q", 1))
        idx_file.write(struct.pack("<B", DTYPE_CODES[self.dtype]))
        idx_file.write(struct.pack("<Q", count))
        idx_file.write(struct.pack("<Q", count + 1))
        idx_file.write(lengths.astype(np.int32).tobytes())
        idx_file.write(pointers.tobytes())
        idx_file.write(documents.tobytes())


def read_index(prefix: str) -> dict:
    """Header, lengths, pointers and document indices of ``prefix.idx``."""
    with open(prefix + ".idx", "rb") as stream:
        if stream.read(9) != _INDEX_HEADER:
            raise ValueError(f"not a Megatron index: {prefix}.idx")
        version, code, count, document_count = struct.unpack("<QBQQ", stream.read(25))
        lengths = np.frombuffer(stream.read(count * 4), dtype="<i4")
        pointers = np.frombuffer(stream.read(count * 8), dtype="<i8")
        documents = np.frombuffer(stream.read(document_count * 8), dtype="<i8")
    return {"version": version, "dtype": CODE_DTYPES[code], "lengths": lengths, "pointers": pointers,
            "documents": documents}


def read_document(prefix: str, index: int, meta: dict | None = None) -> list[int]:
    meta = meta or read_index(prefix)
    dtype = meta["dtype"]
    data = np.memmap(prefix + ".bin", dtype=dtype, mode="r", offset=int(meta["pointers"][index]),
                     shape=(int(meta["lengths"][index]),))
    return data.tolist()
