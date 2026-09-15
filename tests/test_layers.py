from __future__ import annotations

import torch

from tests.adapters import run_embedding, run_linear, run_rmsnorm, run_silu, run_swiglu


def test_linear_matches_explicit_matrix_multiply() -> None:
    leading_shape = (2, 3)
    torch.manual_seed(1)
    weights = torch.randn(7, 4, dtype=torch.float64)
    x = torch.randn(*leading_shape, 4, dtype=torch.float64, requires_grad=True)
    actual = run_linear(4, 7, weights, x)
    expected = x @ weights.T
    assert actual.shape == (*leading_shape, 7)
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_embedding_lookup_and_repeated_indices() -> None:
    weights = torch.randn(7, 3, dtype=torch.float64)
    token_ids = torch.tensor([[1, 4], [4, 2]])
    output = run_embedding(7, 3, weights, token_ids)
    assert output.shape == (2, 2, 3)
    torch.testing.assert_close(output, weights[token_ids])
    torch.testing.assert_close(output[0, 1], output[1, 0])


def test_embedding_accepts_arbitrary_integer_id_shapes() -> None:
    weights = torch.arange(55, dtype=torch.float32).reshape(11, 5)
    scalar = run_embedding(11, 5, weights, torch.tensor(3))
    cube_ids = torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
    cube = run_embedding(11, 5, weights, cube_ids)
    assert scalar.shape == (5,)
    assert cube.shape == (2, 2, 2, 5)
    torch.testing.assert_close(cube, weights[cube_ids])


def test_rmsnorm_matches_reference_and_preserves_low_precision_dtype() -> None:
    weights = torch.randn(6, dtype=torch.float16)
    x = torch.randn(2, 3, 4, 6, dtype=torch.float16)
    actual = run_rmsnorm(6, 1e-5, weights, x)
    reference = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5)
    reference = reference * weights.float()
    assert actual.shape == x.shape
    assert actual.dtype == x.dtype
    torch.testing.assert_close(actual.float(), reference, rtol=2e-3, atol=2e-3)


def test_rmsnorm_preserves_float64_precision() -> None:
    torch.manual_seed(3)
    weights = torch.randn(9, dtype=torch.float64)
    x = torch.randn(2, 5, 9, dtype=torch.float64)
    actual = run_rmsnorm(9, 1e-12, weights, x)
    expected = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-12) * weights
    assert actual.dtype == torch.float64
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_rmsnorm_large_half_values_remain_finite() -> None:
    weights = torch.ones(4, dtype=torch.float16)
    large = torch.full((2, 3, 4), 30_000.0, dtype=torch.float16)
    output = run_rmsnorm(4, 1e-5, weights, large)
    assert output.dtype == large.dtype
    assert torch.isfinite(output).all()


def test_silu_matches_definition_and_has_finite_gradient() -> None:
    x = torch.linspace(-20, 20, 41, dtype=torch.float64, requires_grad=True)
    actual = run_silu(x)
    torch.testing.assert_close(actual, x * torch.sigmoid(x))
    actual.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_swiglu_matches_gate_up_down_formula() -> None:
    torch.manual_seed(4)
    x = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    gate_weight = torch.randn(6, 4, dtype=torch.float64)
    up_weight = torch.randn(6, 4, dtype=torch.float64)
    down_weight = torch.randn(4, 6, dtype=torch.float64)
    gate = x @ gate_weight.T
    expected = ((gate * torch.sigmoid(gate)) * (x @ up_weight.T)) @ down_weight.T
    actual = run_swiglu(4, 6, gate_weight, down_weight, up_weight, x)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_swiglu_supports_arbitrary_leading_dimensions() -> None:
    x = torch.randn(2, 1, 3, 8, dtype=torch.float32)
    gate = torch.randn(12, 8)
    up = torch.randn(12, 8)
    down = torch.randn(8, 12)
    actual = run_swiglu(8, 12, gate, down, up, x)
    expected = ((x @ gate.T) * torch.sigmoid(x @ gate.T) * (x @ up.T)) @ down.T
    assert actual.shape == x.shape
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
