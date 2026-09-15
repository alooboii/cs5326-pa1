from __future__ import annotations

import math

import pytest
import torch

from tests.adapters import (
    run_grouped_query_self_attention,
    run_scaled_dot_product_attention,
    run_softmax,
)


@pytest.mark.level1
def test_softmax_matches_reference_across_dimensions_and_large_offsets() -> None:
    torch.manual_seed(10)
    x = (torch.randn(2, 3, 5, dtype=torch.float64) * 100).requires_grad_()
    for dim in (0, 1, -1):
        actual = run_softmax(x + 10_000, dim)
        expected = torch.softmax(x, dim=dim)
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(actual.sum(dim), torch.ones_like(actual.sum(dim)))
    run_softmax(x, -1).square().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.level2
def test_scaled_attention_supports_leading_dims_masks_and_distinct_value_width() -> None:
    torch.manual_seed(11)
    q = torch.randn(2, 3, 4, 8, dtype=torch.float64, requires_grad=True)
    k = torch.randn(2, 3, 6, 8, dtype=torch.float64, requires_grad=True)
    v = torch.randn(2, 3, 6, 5, dtype=torch.float64, requires_grad=True)
    mask = torch.ones(1, 1, 4, 6, dtype=torch.bool).tril()
    scores = torch.einsum("...qd,...kd->...qk", q, k) / math.sqrt(8)
    expected = torch.softmax(scores.masked_fill(~mask, -torch.inf), dim=-1) @ v
    actual = run_scaled_dot_product_attention(q, k, v, mask)
    assert actual.shape == (2, 3, 4, 5)
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-9)
    actual.square().mean().backward()
    assert all(tensor.grad is not None and torch.isfinite(tensor.grad).all() for tensor in (q, k, v))


@pytest.mark.level3
def test_scaled_attention_validates_core_shapes_and_boolean_mask() -> None:
    q = torch.randn(2, 4, 8)
    k = torch.randn(2, 6, 7)
    v = torch.randn(2, 6, 5)
    with pytest.raises(ValueError, match="feature"):
        run_scaled_dot_product_attention(q, k, v)
    with pytest.raises(ValueError, match="sequence"):
        run_scaled_dot_product_attention(q, torch.randn(2, 6, 8), torch.randn(2, 5, 5))
    with pytest.raises(TypeError, match="boolean"):
        run_scaled_dot_product_attention(q, torch.randn(2, 6, 8), v, torch.ones(4, 6))


@pytest.mark.level3
def test_scaled_attention_rejects_an_all_masked_query() -> None:
    q = torch.randn(2, 3, 4)
    k = torch.randn(2, 5, 4)
    v = torch.randn(2, 5, 7)
    mask = torch.ones(1, 3, 5, dtype=torch.bool)
    mask[:, 1] = False
    with pytest.raises(Exception):
        run_scaled_dot_product_attention(q, k, v, mask)


@pytest.mark.level1
def test_scaled_attention_without_mask_matches_reference() -> None:
    q = torch.randn(4, 6, dtype=torch.float64, requires_grad=True)
    k = torch.randn(7, 6, dtype=torch.float64, requires_grad=True)
    v = torch.randn(7, 3, dtype=torch.float64, requires_grad=True)
    expected = torch.softmax(q @ k.T / math.sqrt(6), dim=-1) @ v
    actual = run_scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert all(t.grad is not None for t in (q, k, v))


def _attention_weights(d_model: int, n_q_heads: int, n_kv_heads: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(100 + n_kv_heads)
    head_dim = d_model // n_q_heads
    return {
        "q_proj.weight": torch.randn(n_q_heads * head_dim, d_model, generator=generator),
        "k_proj.weight": torch.randn(n_kv_heads * head_dim, d_model, generator=generator),
        "v_proj.weight": torch.randn(n_kv_heads * head_dim, d_model, generator=generator),
        "out_proj.weight": torch.randn(d_model, n_q_heads * head_dim, generator=generator),
    }


def _rope_reference(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    head_dim = x.shape[-1]
    frequencies = theta ** (-torch.arange(0, head_dim, 2, device=x.device, dtype=x.dtype) / head_dim)
    angles = positions.to(x.dtype)[..., None] * frequencies
    if positions.ndim == 1:
        angles = angles[None, None]
    else:
        angles = angles[:, None]
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack(
        (even * angles.cos() - odd * angles.sin(), even * angles.sin() + odd * angles.cos()), dim=-1
    ).flatten(-2)


def _explicit_repeat_gqa(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    *,
    n_q_heads: int,
    n_kv_heads: int,
    positions: torch.Tensor,
    theta: float,
) -> torch.Tensor:
    batch, sequence, d_model = x.shape
    head_dim = d_model // n_q_heads
    queries_per_kv = n_q_heads // n_kv_heads
    q = (x @ weights["q_proj.weight"].T).view(batch, sequence, n_q_heads, head_dim).transpose(1, 2)
    k = (x @ weights["k_proj.weight"].T).view(batch, sequence, n_kv_heads, head_dim).transpose(1, 2)
    v = (x @ weights["v_proj.weight"].T).view(batch, sequence, n_kv_heads, head_dim).transpose(1, 2)
    q = _rope_reference(q, positions, theta)
    k = _rope_reference(k, positions, theta)
    k = k.repeat_interleave(queries_per_kv, dim=1)
    v = v.repeat_interleave(queries_per_kv, dim=1)
    scores = q @ k.transpose(-1, -2) / math.sqrt(head_dim)
    causal = torch.ones(sequence, sequence, dtype=torch.bool, device=x.device).tril()
    probabilities = torch.softmax(scores.masked_fill(~causal, -torch.inf), dim=-1)
    output = (probabilities @ v).transpose(1, 2).reshape(batch, sequence, d_model)
    return output @ weights["out_proj.weight"].T


@pytest.mark.parametrize("n_kv_heads", [1, 2, 4])
@pytest.mark.level2
def test_gqa_mqa_and_mha_match_independent_explicit_repeat(n_kv_heads: int) -> None:
    torch.manual_seed(12)
    d_model, n_q_heads, sequence = 16, 4, 6
    x = torch.randn(2, sequence, d_model)
    positions = torch.tensor([[0, 1, 2, 4, 7, 8], [3, 4, 5, 6, 7, 9]])
    weights = _attention_weights(d_model, n_q_heads, n_kv_heads)
    actual = run_grouped_query_self_attention(
        d_model,
        n_q_heads,
        n_kv_heads,
        16,
        10_000.0,
        weights["q_proj.weight"],
        weights["k_proj.weight"],
        weights["v_proj.weight"],
        weights["out_proj.weight"],
        x,
        positions,
    )
    expected = _explicit_repeat_gqa(
        x,
        weights,
        n_q_heads=n_q_heads,
        n_kv_heads=n_kv_heads,
        positions=positions,
        theta=10_000.0,
    )
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


@torch.no_grad()
@pytest.mark.level1
def test_gqa_is_strictly_causal_and_prefix_invariant() -> None:
    d_model, n_q_heads, n_kv_heads = 16, 4, 2
    weights = _attention_weights(d_model, n_q_heads, n_kv_heads)
    x = torch.randn(1, 8, d_model)
    changed = x.clone()
    changed[:, 5:] = torch.randn_like(changed[:, 5:]) * 20
    arguments = (
        d_model,
        n_q_heads,
        n_kv_heads,
        16,
        10_000.0,
        weights["q_proj.weight"],
        weights["k_proj.weight"],
        weights["v_proj.weight"],
        weights["out_proj.weight"],
    )
    original = run_grouped_query_self_attention(*arguments, x)
    modified = run_grouped_query_self_attention(*arguments, changed)
    torch.testing.assert_close(original[:, :5], modified[:, :5], rtol=1e-5, atol=1e-5)


@pytest.mark.level3
def test_gqa_validates_head_ratios_and_context_length() -> None:
    x = torch.randn(1, 4, 18)
    weights = _attention_weights(18, 3, 1)
    with pytest.raises(Exception):
        run_grouped_query_self_attention(
            18,
            4,
            1,
            8,
            10_000.0,
            torch.randn(18, 18),
            torch.randn(4, 18),
            torch.randn(4, 18),
            torch.randn(18, 18),
            x,
        )
    with pytest.raises(Exception):
        run_grouped_query_self_attention(
            18,
            3,
            1,
            3,
            10_000.0,
            weights["q_proj.weight"],
            weights["k_proj.weight"],
            weights["v_proj.weight"],
            weights["out_proj.weight"],
            x,
        )
