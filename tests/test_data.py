from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tests.adapters import run_get_batch, run_load_token_array


@pytest.mark.level1
def test_load_token_array_is_read_only_little_endian_memmap(tmp_path: Path) -> None:
    path = tmp_path / "tokens.bin"
    np.asarray([0, 17, 8191], dtype=np.dtype("<u2")).tofile(path)
    tokens = run_load_token_array(path)
    assert isinstance(tokens, np.memmap)
    assert tokens.ndim == 1
    assert tokens.dtype == np.dtype("<u2")
    assert tokens.mode == "r"
    np.testing.assert_array_equal(tokens, [0, 17, 8191])


@pytest.mark.level3
def test_load_token_array_rejects_missing_and_odd_byte_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_load_token_array(tmp_path / "missing.bin")
    malformed = tmp_path / "malformed.bin"
    malformed.write_bytes(b"\x00")
    with pytest.raises(Exception):
        run_load_token_array(malformed)


@pytest.mark.level1
def test_get_batch_matches_exact_generator_starts_and_shift() -> None:
    tokens = np.arange(257, dtype=np.dtype("<u2"))
    actual_generator = torch.Generator().manual_seed(2025)
    reference_generator = torch.Generator().manual_seed(2025)
    starts = torch.randint(0, len(tokens) - 11, (6,), generator=reference_generator)

    x, y = run_get_batch(tokens, 6, 11, "cpu", actual_generator)
    expected = torch.stack(
        [
            torch.from_numpy(tokens[start : start + 12].astype(np.int64))
            for start in starts.tolist()
        ]
    )
    assert x.shape == y.shape == (6, 11)
    assert x.dtype == y.dtype == torch.long
    assert x.device.type == y.device.type == "cpu"
    torch.testing.assert_close(x, expected[:, :-1])
    torch.testing.assert_close(y, expected[:, 1:])


@pytest.mark.level2
def test_get_batch_generator_is_reproducible_and_stateful() -> None:
    tokens = np.arange(1000, dtype=np.dtype("<u2"))
    first = torch.Generator().manual_seed(13)
    second = torch.Generator().manual_seed(13)
    x1, y1 = run_get_batch(tokens, 7, 19, "cpu", first)
    x2, y2 = run_get_batch(tokens, 7, 19, torch.device("cpu"), second)
    torch.testing.assert_close(x1, x2)
    torch.testing.assert_close(y1, y2)
    x1_next, _ = run_get_batch(tokens, 7, 19, "cpu", first)
    assert not torch.equal(x1, x1_next)


@pytest.mark.level3
def test_get_batch_reaches_first_and_final_valid_start() -> None:
    tokens = np.arange(5, dtype=np.dtype("<u2"))
    x, _ = run_get_batch(tokens, 128, 3, "cpu", torch.Generator().manual_seed(123))
    assert set(x[:, 0].tolist()) == {0, 1}


@pytest.mark.level2
def test_get_batch_reads_only_sampled_slices() -> None:
    class SliceOnlyTokenStream:
        def __init__(self) -> None:
            self.values = np.arange(128, dtype=np.dtype("<u2"))

        def __len__(self) -> int:
            return len(self.values)

        def __getitem__(self, index):
            if not isinstance(index, slice):
                raise AssertionError("batching must read sampled slices")
            return self.values[index]

        def __array__(self, *args, **kwargs):
            raise AssertionError("the complete corpus must not be converted")

    x, y = run_get_batch(
        SliceOnlyTokenStream(), 3, 9, "cpu", torch.Generator().manual_seed(1)
    )
    assert x.shape == y.shape == (3, 9)


@pytest.mark.level3
@pytest.mark.parametrize("batch_size,sequence_length", [(0, 8), (2, 0), (-1, 8)])
def test_get_batch_rejects_invalid_sizes(
    batch_size: int, sequence_length: int
) -> None:
    with pytest.raises(Exception):
        run_get_batch(
            np.arange(20, dtype=np.uint16),
            batch_size,
            sequence_length,
            "cpu",
            torch.Generator(),
        )


@pytest.mark.level3
def test_get_batch_accepts_minimum_stream_and_rejects_shorter() -> None:
    minimum = np.arange(9, dtype=np.uint16)
    x, y = run_get_batch(minimum, 2, 8, "cpu", torch.Generator().manual_seed(0))
    assert x.shape == y.shape == (2, 8)
    with pytest.raises(Exception):
        run_get_batch(
            np.arange(8, dtype=np.uint16), 2, 8, "cpu", torch.Generator()
        )
