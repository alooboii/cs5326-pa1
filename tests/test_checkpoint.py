from __future__ import annotations

from pathlib import Path

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
