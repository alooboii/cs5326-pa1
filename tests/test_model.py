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


@pytest.mark.level1
def test_transformer_block_matches_frozen_pre_norm_reference() -> None:
    inputs, weights = _load_fixture("transformer_block.npz")
    actual = run_transformer_block(
        *_block_arguments(),
        weights,
        inputs["x"],
        token_positions=inputs["positions"],
    )
    torch.testing.assert_close(actual, inputs["expected"], rtol=1e-5, atol=1e-5)


@pytest.mark.level2
def test_attention_residual_path_matches_diagnostic_fixture() -> None:
    inputs, weights = _load_fixture("transformer_block_attention_residual.npz")
    actual = run_transformer_block(
        *_block_arguments(), weights, inputs["x"], token_positions=inputs["positions"]
    )
    torch.testing.assert_close(actual, inputs["expected"], rtol=1e-5, atol=1e-5)
    assert not torch.equal(actual, inputs["x"])


@pytest.mark.level2
def test_ffn_residual_path_matches_diagnostic_fixture() -> None:
    inputs, weights = _load_fixture("transformer_block_ffn_residual.npz")
    actual = run_transformer_block(
        *_block_arguments(), weights, inputs["x"], token_positions=inputs["positions"]
    )
    torch.testing.assert_close(actual, inputs["expected"], rtol=1e-5, atol=1e-5)
    assert not torch.equal(actual, inputs["x"])


@pytest.mark.level1
def test_transformer_lm_matches_frozen_full_and_truncated_references() -> None:
    inputs, weights = _load_fixture("transformer_lm.npz")
    full = run_transformer_lm(*_lm_arguments(), weights, inputs["token_ids"])
    truncated = run_transformer_lm(*_lm_arguments(), weights, inputs["token_ids"][:, :4])
    torch.testing.assert_close(full, inputs["expected"], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(truncated, inputs["truncated_expected"], rtol=1e-5, atol=1e-5)


@torch.no_grad()
@pytest.mark.level2
def test_transformer_lm_is_prefix_invariant() -> None:
    inputs, weights = _load_fixture("transformer_lm.npz")
    original = inputs["token_ids"]
    changed = original.clone()
    changed[:, 4:] = torch.tensor([[11, 12, 13], [14, 15, 16]])
    original_logits = run_transformer_lm(*_lm_arguments(), weights, original)
    changed_logits = run_transformer_lm(*_lm_arguments(), weights, changed)
    torch.testing.assert_close(original_logits[:, :4], changed_logits[:, :4], rtol=1e-5, atol=1e-5)


@pytest.mark.level2
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


@pytest.mark.level1
def test_recommended_parameter_count_and_untied_embeddings() -> None:
    vocab_size, d_model = 8192, 512
    model = get_transformer_lm(
        vocab_size, 256, d_model, 4, 16, 4, 1344, 10_000.0
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == 19_272_192
    vocabulary_matrices = [
        parameter
        for parameter in model.parameters()
        if tuple(parameter.shape) == (vocab_size, d_model)
    ]
    assert len(vocabulary_matrices) == 2
    assert vocabulary_matrices[0].data_ptr() != vocabulary_matrices[1].data_ptr()


@pytest.mark.level2
def test_model_initialization_and_no_dropout() -> None:
    torch.manual_seed(2026)
    vocab_size, d_model = 257, 64
    model = get_transformer_lm(vocab_size, 16, d_model, 2, 8, 2, 192, 10_000.0)
    assert not any(isinstance(module, torch.nn.Dropout) for module in model.modules())

    gains = [p for p in model.parameters() if p.ndim == 1]
    assert gains
    for gain in gains:
        torch.testing.assert_close(gain, torch.ones_like(gain))

    vocabulary_matrices = [
        parameter
        for parameter in model.parameters()
        if tuple(parameter.shape) == (vocab_size, d_model)
    ]
    assert len(vocabulary_matrices) == 2
    standard_deviations = sorted(
        float(matrix.detach().std()) for matrix in vocabulary_matrices
    )
    linear_std = (2.0 / (d_model + vocab_size)) ** 0.5
    assert standard_deviations[0] == pytest.approx(linear_std, rel=0.15)
    assert standard_deviations[1] == pytest.approx(1.0, rel=0.15)
    assert max(
        float(matrix.detach().abs().max()) for matrix in vocabulary_matrices
    ) <= 3.0
    embedding_matrix = max(
        vocabulary_matrices, key=lambda matrix: float(matrix.detach().std())
    )
    for matrix in (p for p in model.parameters() if p.ndim == 2):
        if matrix is embedding_matrix:
            continue
        expected_std = (2.0 / (matrix.shape[0] + matrix.shape[1])) ** 0.5
        assert float(matrix.detach().std()) == pytest.approx(expected_std, rel=0.18)
        assert float(matrix.detach().abs().max()) <= 3.01 * expected_std


@pytest.mark.level3
def test_model_rejects_empty_and_overlength_sequences() -> None:
    model = get_transformer_lm(31, 8, 16, 1, 4, 2, 24, 10_000.0)
    with pytest.raises(Exception):
        model(torch.empty(2, 0, dtype=torch.long))
    with pytest.raises(Exception):
        model(torch.zeros(2, 9, dtype=torch.long))


@pytest.mark.level3
def test_transformer_lm_uses_explicit_positions() -> None:
    inputs, weights = _load_fixture("transformer_lm.npz")
    token_ids = inputs["token_ids"][:, :4]
    ordinary = run_transformer_lm(*_lm_arguments(), weights, token_ids)
    positions = torch.tensor([[0, 1, 3, 6], [1, 2, 5, 7]])
    shifted = run_transformer_lm(
        *_lm_arguments(), weights, token_ids, token_positions=positions
    )
    assert shifted.shape == ordinary.shape
    assert not torch.allclose(shifted, ordinary)
