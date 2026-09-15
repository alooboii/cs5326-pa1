from __future__ import annotations

import pytest
import torch

from tests.adapters import run_rope


def _explicit_adjacent_pair_rope(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    head_dim = x.shape[-1]
    pair_ids = torch.arange(0, head_dim, 2, dtype=torch.float64, device=x.device)
    inverse_frequencies = theta ** (-pair_ids / head_dim)
    angles = positions.to(torch.float64)[..., None] * inverse_frequencies
    batch_dims = x.ndim - 2
    position_batch_dims = positions.ndim - 1
    angles = angles.reshape(
        *positions.shape[:-1],
        *((1,) * (batch_dims - position_batch_dims)),
        positions.shape[-1],
        head_dim // 2,
    )
    even = x.double()[..., 0::2]
    odd = x.double()[..., 1::2]
    return torch.stack(
        (even * angles.cos() - odd * angles.sin(), even * angles.sin() + odd * angles.cos()),
        dim=-1,
    ).flatten(-2).to(x.dtype)


def test_rope_matches_explicit_adjacent_pair_rotation() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 3, 5, 8, dtype=torch.float64)
    positions = torch.tensor([0, 2, 5, 7, 11])
    actual = run_rope(8, 10_000.0, 16, x, positions)
    expected = _explicit_adjacent_pair_rope(x, positions, 10_000.0)
    # The cache is intentionally stored in float32, even for float64 inputs.
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)


def test_rope_dot_product_depends_only_on_relative_position() -> None:
    q = torch.randn(1, 1, 8)
    k = torch.randn(1, 1, 8)
    score_a = (
        run_rope(8, 10_000.0, 64, q, torch.tensor([7]))
        * run_rope(8, 10_000.0, 64, k, torch.tensor([19]))
    ).sum()
    score_b = (
        run_rope(8, 10_000.0, 64, q, torch.tensor([13]))
        * run_rope(8, 10_000.0, 64, k, torch.tensor([25]))
    ).sum()
    torch.testing.assert_close(score_a, score_b, rtol=2e-5, atol=2e-5)


def test_rope_validates_dimensions_position_dtype_and_bounds() -> None:
    with pytest.raises(ValueError, match="even"):
        run_rope(7, 10_000.0, 16, torch.randn(2, 4, 7), torch.arange(4))
    with pytest.raises(TypeError, match="integer"):
        run_rope(
            8,
            10_000.0,
            16,
            torch.randn(2, 4, 8),
            torch.arange(4, dtype=torch.float32),
        )
    with pytest.raises(ValueError, match="outside"):
        run_rope(8, 10_000.0, 16, torch.randn(2, 4, 8), torch.tensor([0, 1, 2, 16]))
    with pytest.raises(ValueError, match="head_dim"):
        run_rope(8, 10_000.0, 16, torch.randn(2, 4, 6), torch.arange(4))
    with pytest.raises(Exception):
        run_rope(8, 10_000.0, 16, torch.randn(2, 4, 8), torch.arange(3))
