from __future__ import annotations

import math
import copy

import pytest
import torch

from tests.adapters import (
    get_adamw_cls,
    run_cross_entropy,
    run_get_lr_cosine_schedule,
    run_gradient_clipping,
)


@pytest.mark.level1
def test_cross_entropy_matches_reference_for_arbitrary_leading_dimensions() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 4, 11, dtype=torch.float64, requires_grad=True)
    targets = torch.randint(0, 11, (2, 3, 4))
    actual = run_cross_entropy(logits, targets)
    expected = torch.nn.functional.cross_entropy(logits.reshape(-1, 11), targets.reshape(-1))
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


@pytest.mark.level3
def test_cross_entropy_is_stable_and_shift_invariant() -> None:
    logits = torch.tensor(
        [[1000.0, -1000.0, 999.0], [-1000.0, 1000.0, 998.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    targets = torch.tensor([0, 2])
    loss = run_cross_entropy(logits, targets)
    shifted = run_cross_entropy(logits + 50_000, targets)
    torch.testing.assert_close(loss, shifted)
    assert torch.isfinite(loss)
    loss.backward()
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


@pytest.mark.level1
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


@pytest.mark.level2
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


@pytest.mark.level2
def test_adamw_state_dict_round_trip_continues_exactly() -> None:
    first = torch.nn.Parameter(torch.tensor([1.0, -1.0], dtype=torch.float64))
    optimizer_a = get_adamw_cls()([first], lr=0.02, betas=(0.8, 0.9), weight_decay=0.1)
    first.grad = torch.tensor([0.2, -0.4], dtype=torch.float64)
    optimizer_a.step()

    second = torch.nn.Parameter(first.detach().clone())
    optimizer_b = get_adamw_cls()([second], lr=0.02, betas=(0.8, 0.9), weight_decay=0.1)
    optimizer_b.load_state_dict(copy.deepcopy(optimizer_a.state_dict()))
    gradient = torch.tensor([-0.3, 0.7], dtype=torch.float64)
    first.grad = gradient.clone()
    second.grad = gradient.clone()
    optimizer_a.step()
    optimizer_b.step()
    torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.level3
def test_adamw_rejects_sparse_gradients() -> None:
    parameter = torch.nn.Parameter(torch.zeros(4, 3))
    parameter.grad = torch.sparse_coo_tensor(
        torch.tensor([[0, 2]]),
        torch.ones(2, 3),
        parameter.shape,
        check_invariants=False,
    )
    optimizer = get_adamw_cls()([parameter])
    with pytest.raises(RuntimeError, match="sparse"):
        optimizer.step()


@pytest.mark.level1
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


@pytest.mark.parametrize(
    "step,warmup,cycle",
    [(-1, 0, 10), (0, -1, 10), (0, 10, 10), (0, 11, 10)],
)
@pytest.mark.level3
def test_cosine_schedule_rejects_invalid_arguments(step: int, warmup: int, cycle: int) -> None:
    with pytest.raises(ValueError):
        run_get_lr_cosine_schedule(step, 1.0, 0.1, warmup, cycle)


@pytest.mark.level1
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


@pytest.mark.level2
def test_gradient_clipping_noop_and_empty_gradients() -> None:
    parameter = torch.nn.Parameter(torch.zeros(2))
    parameter.grad = torch.tensor([0.3, 0.4])
    before = parameter.grad.clone()
    assert run_gradient_clipping([parameter], 1.0) == pytest.approx(0.5)
    torch.testing.assert_close(parameter.grad, before)
    parameter.grad = None
    assert run_gradient_clipping([parameter], 1.0) == 0.0
    with pytest.raises(ValueError):
        run_gradient_clipping([parameter], 0.0)


@pytest.mark.level2
def test_adamw_closure_runs_with_gradients_enabled_and_returns_loss() -> None:
    parameter = torch.nn.Parameter(torch.tensor([2.0]))
    optimizer = get_adamw_cls()([parameter], lr=0.1)
    calls = 0

    def closure() -> torch.Tensor:
        nonlocal calls
        calls += 1
        optimizer.zero_grad()
        loss = parameter.square().sum()
        loss.backward()
        return loss

    loss = optimizer.step(closure)
    assert calls == 1
    assert loss is not None and float(loss.detach()) == pytest.approx(4.0)
    assert parameter.item() < 2.0


@pytest.mark.level3
def test_adamw_uses_per_parameter_steps_after_missing_gradients() -> None:
    first = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    delayed = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    optimizer = get_adamw_cls()([first, delayed], lr=0.01, betas=(0.8, 0.9))
    first.grad = torch.tensor([0.5], dtype=torch.float64)
    delayed.grad = None
    optimizer.step()
    first.grad = torch.tensor([0.25], dtype=torch.float64)
    delayed.grad = torch.tensor([0.25], dtype=torch.float64)
    optimizer.step()
    assert optimizer.state[first]["step"] == 2
    assert optimizer.state[delayed]["step"] == 1


@pytest.mark.level3
def test_adamw_validates_effective_group_values_at_step_time() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = get_adamw_cls()([parameter])
    parameter.grad = torch.ones_like(parameter)
    optimizer.param_groups[0]["lr"] = -1.0
    with pytest.raises(Exception):
        optimizer.step()


@pytest.mark.level3
def test_cosine_schedule_rejects_invalid_rates_and_noninteger_steps() -> None:
    with pytest.raises(Exception):
        run_get_lr_cosine_schedule(0, 0.1, 0.2, 0, 10)
    with pytest.raises(Exception):
        run_get_lr_cosine_schedule(0.5, 1.0, 0.1, 0, 10)  # type: ignore[arg-type]
