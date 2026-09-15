from __future__ import annotations

import io
from pathlib import Path

import pytest
import torch

from tests.adapters import get_adamw_cls, run_load_checkpoint, run_save_checkpoint


def _objects(seed: int = 0):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 5),
        torch.nn.Tanh(),
        torch.nn.Linear(5, 2),
    )
    optimizer = get_adamw_cls()(model.parameters(), lr=0.01, betas=(0.8, 0.9))
    train_generator = torch.Generator().manual_seed(111)
    val_generator = torch.Generator().manual_seed(222)
    return model, optimizer, train_generator, val_generator


def _update(model, optimizer, generator: torch.Generator) -> float:
    optimizer.zero_grad()
    x = torch.randn(4, 3, generator=generator)
    target = torch.randn(4, 2, generator=generator)
    loss = (model(x) - target).square().mean()
    loss.backward()
    optimizer.step()
    return float(loss.detach())


@pytest.mark.level1
def test_checkpointing_round_trip_restores_all_required_state(tmp_path: Path) -> None:
    model, optimizer, train_generator, val_generator = _objects()
    _update(model, optimizer, train_generator)
    torch.rand(7, generator=val_generator)
    expected_model = {key: value.clone() for key, value in model.state_dict().items()}
    expected_optimizer = optimizer.state_dict()
    expected_train_state = train_generator.get_state().clone()
    expected_val_state = val_generator.get_state().clone()

    path = tmp_path / "checkpoint.pt"
    run_save_checkpoint(model, optimizer, 17, train_generator, val_generator, path)

    for parameter in model.parameters():
        parameter.data.zero_()
    optimizer.state.clear()
    train_generator.manual_seed(1)
    val_generator.manual_seed(2)

    next_step = run_load_checkpoint(
        path, model, optimizer, train_generator, val_generator
    )
    assert next_step == 17
    for key, expected in expected_model.items():
        torch.testing.assert_close(model.state_dict()[key], expected, rtol=0, atol=0)
    assert optimizer.state_dict()["param_groups"] == expected_optimizer["param_groups"]
    assert torch.equal(train_generator.get_state(), expected_train_state)
    assert torch.equal(val_generator.get_state(), expected_val_state)


@pytest.mark.level2
def test_checkpointing_supports_binary_file_objects() -> None:
    model, optimizer, train_generator, val_generator = _objects()
    buffer = io.BytesIO()
    run_save_checkpoint(model, optimizer, 9, train_generator, val_generator, buffer)
    buffer.seek(0)

    restored_model, restored_optimizer, restored_train, restored_val = _objects(99)
    next_step = run_load_checkpoint(
        buffer,
        restored_model,
        restored_optimizer,
        restored_train,
        restored_val,
    )
    assert next_step == 9
    for actual, expected in zip(restored_model.parameters(), model.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.level3
def test_checkpointing_resume_matches_the_next_uninterrupted_update() -> None:
    model_a, optimizer_a, train_a, val_a = _objects()
    _update(model_a, optimizer_a, train_a)
    torch.rand(5, generator=val_a)

    buffer = io.BytesIO()
    run_save_checkpoint(model_a, optimizer_a, 1, train_a, val_a, buffer)
    buffer.seek(0)

    model_b, optimizer_b, train_b, val_b = _objects(999)
    assert run_load_checkpoint(buffer, model_b, optimizer_b, train_b, val_b) == 1

    next_val_a = torch.rand(8, generator=val_a)
    next_val_b = torch.rand(8, generator=val_b)
    torch.testing.assert_close(next_val_a, next_val_b, rtol=0, atol=0)
    loss_a = _update(model_a, optimizer_a, train_a)
    loss_b = _update(model_b, optimizer_b, train_b)
    assert loss_a == pytest.approx(loss_b, rel=0, abs=0)
    for actual, expected in zip(model_a.parameters(), model_b.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
