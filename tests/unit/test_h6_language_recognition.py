from __future__ import annotations

import importlib
import math

import pytest
import torch

from vfe4.types.h6 import FrozenTensorSnapshot


def _recognition_api() -> tuple[type[object], type[object], type[object]]:
    module = importlib.import_module("vfe4.recognition")
    required = (
        "RecognitionConditioning",
        "StructuredLanguageRecognition",
        "FactorizedLanguageRecognition",
    )
    missing = tuple(name for name in required if not hasattr(module, name))
    assert not missing, f"missing H6 language recognition API: {missing}"
    return tuple(getattr(module, name) for name in required)  # type: ignore[return-value]


def test_filtering_and_smoothing_expose_only_the_declared_observations() -> None:
    RecognitionConditioning, _, _ = _recognition_api()
    source = torch.tensor([11, 12, 13], dtype=torch.int64)
    filtering = RecognitionConditioning.create(
        mode="filtering", horizon=3, observed_tokens=source
    )
    smoothing = RecognitionConditioning.create(
        mode="smoothing", horizon=3, observed_tokens=source
    )

    assert type(filtering.observed_tokens) is FrozenTensorSnapshot
    assert filtering.visible_tokens(0).numel() == 0
    assert filtering.visible_tokens(1).tolist() == [11]
    assert filtering.visible_tokens(2).tolist() == [11, 12]
    assert filtering.visible_tokens(3).tolist() == [11, 12, 13]
    assert smoothing.visible_tokens(0).tolist() == [11, 12, 13]
    assert smoothing.visible_tokens(2).tolist() == [11, 12, 13]

    source.add_(100)
    returned = filtering.visible_tokens(2)
    returned.zero_()
    assert filtering.visible_tokens(2).tolist() == [11, 12]
    with pytest.raises(ValueError, match="receiver_t"):
        filtering.visible_tokens(4)

    private = getattr(
        filtering.observed_tokens, "_FrozenTensorSnapshot__owned"
    )
    private.add_(1)
    with pytest.raises(ValueError, match="integrity"):
        filtering.visible_tokens(1)


def test_structured_and_population_factorized_gaussians_are_typed_and_live() -> None:
    RecognitionConditioning, Structured, Factorized = _recognition_api()
    conditioning = RecognitionConditioning.create(
        mode="smoothing",
        horizon=2,
        observed_tokens=torch.tensor([3, 4], dtype=torch.int64),
    )
    mean_leaf = torch.tensor([0.2, -0.1], dtype=torch.float64, requires_grad=True)
    raw_diagonal = torch.tensor(
        [0.1, -0.2], dtype=torch.float64, requires_grad=True
    )
    off_diagonal = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    cholesky = torch.stack(
        (
            torch.stack((torch.exp(raw_diagonal[0]), torch.zeros_like(off_diagonal))),
            torch.stack((off_diagonal, torch.exp(raw_diagonal[1]))),
        )
    )

    structured = Structured.create(
        conditioning=conditioning,
        mean=mean_leaf,
        precision_cholesky=cholesky,
    )
    assert structured.family == "structured_full_spd"
    assert type(structured.mean) is FrozenTensorSnapshot
    assert type(structured.precision_cholesky) is FrozenTensorSnapshot
    assert structured.precision().shape == (2, 2)
    entropy = structured.entropy()
    expected = 0.5 * 2 * (1.0 + math.log(2.0 * math.pi)) - raw_diagonal.sum()
    assert torch.allclose(entropy, expected)
    entropy.backward()
    assert raw_diagonal.grad is not None
    assert torch.equal(
        raw_diagonal.grad, torch.tensor([-1.0, -1.0], dtype=torch.float64)
    )

    factorized_cholesky = torch.diag(torch.exp(raw_diagonal.detach()))
    factorized = Factorized.create(
        conditioning=conditioning,
        mean=mean_leaf.detach(),
        precision_cholesky=factorized_cholesky,
        block_sizes=(1, 1),
    )
    assert factorized.family == "population_factorized_block_spd"
    assert factorized.block_sizes == (1, 1)
    assert torch.equal(factorized.precision_cholesky_value(), factorized_cholesky)

    with pytest.raises(ValueError, match="block diagonal"):
        Factorized.create(
            conditioning=conditioning,
            mean=mean_leaf.detach(),
            precision_cholesky=cholesky.detach(),
            block_sizes=(1, 1),
        )


def test_language_gaussian_reparameterized_sample_preserves_autograd() -> None:
    RecognitionConditioning, Structured, _ = _recognition_api()
    conditioning = RecognitionConditioning.create(
        mode="filtering",
        horizon=1,
        observed_tokens=torch.tensor([7], dtype=torch.int64),
    )
    mean = torch.tensor([0.5, -0.5], dtype=torch.float64, requires_grad=True)
    cholesky = torch.diag(
        torch.tensor([2.0, 4.0], dtype=torch.float64, requires_grad=True)
    )
    law = Structured.create(
        conditioning=conditioning,
        mean=mean,
        precision_cholesky=cholesky,
    )

    sample = law.rsample(torch.tensor([1.0, -2.0], dtype=torch.float64))
    assert torch.allclose(sample, torch.tensor([1.0, -1.0], dtype=torch.float64))
    sample.sum().backward()
    assert mean.grad is not None
    assert torch.equal(mean.grad, torch.ones_like(mean))
