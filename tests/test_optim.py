from __future__ import annotations

import math
import pytest
import torch

from tests.adapters import (
    get_adamw_cls,
    run_cross_entropy,
    run_get_lr_cosine_schedule,
    run_gradient_clipping,
)


def test_cross_entropy_matches_reference_for_arbitrary_leading_dimensions() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 4, 11, dtype=torch.float64, requires_grad=True)
    targets = torch.randint(0, 11, (2, 3, 4))
    actual = run_cross_entropy(logits, targets)
    expected = torch.nn.functional.cross_entropy(logits.reshape(-1, 11), targets.reshape(-1))
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def _adamw_reference(
    parameter: torch.Tensor,
    gradients: list[torch.Tensor],
    *,
    lr: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
) -> torch.Tensor:
    result = parameter.clone()
    first_moment = torch.zeros_like(result)
    second_moment = torch.zeros_like(result)
    beta1, beta2 = betas
    for step, gradient in enumerate(gradients, start=1):
        first_moment = beta1 * first_moment + (1 - beta1) * gradient
        second_moment = beta2 * second_moment + (1 - beta2) * gradient.square()
        result = result * (1 - lr * weight_decay)
        first_unbiased = first_moment / (1 - beta1**step)
        second_unbiased = second_moment / (1 - beta2**step)
        result = result - lr * first_unbiased / (second_unbiased.sqrt() + eps)
    return result


def test_adamw_matches_assignment_equations_for_multiple_steps() -> None:
    initial = torch.tensor([1.0, -2.0], dtype=torch.float64)
    gradients = [
        torch.tensor([0.25, -0.5], dtype=torch.float64),
        torch.tensor([-0.1, 0.3], dtype=torch.float64),
        torch.tensor([0.7, 0.2], dtype=torch.float64),
    ]
    arguments = dict(lr=0.1, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.2)
    parameter = torch.nn.Parameter(initial.clone())
    optimizer = get_adamw_cls()([parameter], **arguments)
    for gradient in gradients:
        parameter.grad = gradient.clone()
        optimizer.step()
    expected = _adamw_reference(initial, gradients, **arguments)
    torch.testing.assert_close(parameter, expected)


def test_adamw_honors_parameter_groups_and_skips_missing_gradients() -> None:
    first = torch.nn.Parameter(torch.tensor([1.0]))
    second = torch.nn.Parameter(torch.tensor([2.0]))
    untouched = torch.nn.Parameter(torch.tensor([3.0]))
    optimizer = get_adamw_cls()(
        [
            {"params": [first], "lr": 0.1, "weight_decay": 0.0},
            {"params": [second, untouched], "lr": 0.01, "weight_decay": 0.5},
        ],
        betas=(0.0, 0.0),
        eps=0.0,
    )
    first.grad = torch.ones_like(first)
    second.grad = torch.ones_like(second)
    optimizer.step()
    torch.testing.assert_close(first, torch.tensor([0.9]))
    torch.testing.assert_close(second, torch.tensor([1.98]))
    torch.testing.assert_close(untouched, torch.tensor([3.0]))
    assert untouched not in optimizer.state


def test_cosine_schedule_warmup_boundaries_floor_and_zero_warmup() -> None:
    arguments = dict(
        learning_rate_max=1.0,
        learning_rate_min=0.1,
        warmup_steps=2,
        cosine_steps=10,
    )
    assert run_get_lr_cosine_schedule(0, **arguments) == 0.0
    assert run_get_lr_cosine_schedule(1, **arguments) == 0.5
    assert run_get_lr_cosine_schedule(2, **arguments) == 1.0
    assert run_get_lr_cosine_schedule(10, **arguments) == pytest.approx(0.1)
    assert run_get_lr_cosine_schedule(20, **arguments) == 0.1
    arguments["warmup_steps"] = 0
    assert run_get_lr_cosine_schedule(0, **arguments) == 1.0


def test_global_gradient_clipping_uses_one_scale_factor_and_returns_preclip_norm() -> None:
    first = torch.nn.Parameter(torch.zeros(1))
    second = torch.nn.Parameter(torch.zeros(1))
    ignored = torch.nn.Parameter(torch.zeros(1))
    first.grad = torch.tensor([3.0])
    second.grad = torch.tensor([4.0])
    norm = run_gradient_clipping([first, second, ignored], 2.5)
    assert norm == 5.0
    torch.testing.assert_close(first.grad, torch.tensor([1.5]), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(second.grad, torch.tensor([2.0]), rtol=1e-5, atol=1e-5)
