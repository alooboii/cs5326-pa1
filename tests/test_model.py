from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tests.adapters import get_transformer_lm, run_transformer_block, run_transformer_lm


SNAPSHOTS = Path(__file__).parent / "_snapshots"


def _load_fixture(name: str) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    with np.load(SNAPSHOTS / name) as archive:
        inputs = {
            key.removeprefix("input::"): torch.from_numpy(archive[key].copy())
            for key in archive.files
            if key.startswith("input::")
        }
        weights = {
            key.removeprefix("weight::"): torch.from_numpy(archive[key].copy())
            for key in archive.files
            if key.startswith("weight::")
        }
    return inputs, weights


def _block_arguments() -> tuple[int, int, int, int, int, float]:
    return (16, 4, 2, 24, 12, 10_000.0)


def _lm_arguments() -> tuple[int, int, int, int, int, int, int, float]:
    return (23, 12, 16, 2, 4, 2, 24, 10_000.0)


def test_transformer_block_matches_frozen_pre_norm_reference() -> None:
    inputs, weights = _load_fixture("transformer_block.npz")
    actual = run_transformer_block(
        *_block_arguments(),
        weights,
        inputs["x"],
        token_positions=inputs["positions"],
    )
    torch.testing.assert_close(actual, inputs["expected"], rtol=1e-5, atol=1e-5)


def test_transformer_lm_matches_frozen_full_and_truncated_references() -> None:
    inputs, weights = _load_fixture("transformer_lm.npz")
    full = run_transformer_lm(*_lm_arguments(), weights, inputs["token_ids"])
    truncated = run_transformer_lm(*_lm_arguments(), weights, inputs["token_ids"][:, :4])
    torch.testing.assert_close(full, inputs["expected"], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(truncated, inputs["truncated_expected"], rtol=1e-5, atol=1e-5)


def test_model_forward_backward_reaches_every_parameter() -> None:
    model = get_transformer_lm(*_lm_arguments())
    tokens = torch.randint(0, 23, (3, 7))
    logits = model(tokens)
    assert logits.shape == (3, 7, 23)
    logits.square().mean().backward()
    parameters = list(model.parameters())
    assert parameters
    assert all(parameter.grad is not None for parameter in parameters)
    assert all(torch.isfinite(parameter.grad).all() for parameter in parameters if parameter.grad is not None)


def test_model_rejects_empty_and_overlength_sequences() -> None:
    model = get_transformer_lm(31, 8, 16, 1, 4, 2, 24, 10_000.0)
    with pytest.raises(Exception):
        model(torch.empty(2, 0, dtype=torch.long))
    with pytest.raises(Exception):
        model(torch.zeros(2, 9, dtype=torch.long))
