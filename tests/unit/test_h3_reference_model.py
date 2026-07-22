from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from vfe4.generative import (
    H3CanonicalJoint,
    H3GenerativeModel,
    H3ScalarGaussianFactor,
)
from vfe4.validation.h3_fixture import (
    H3_COUPLED_FIXTURE_PATH,
    H3_ZERO_CONTROL_FIXTURE_PATH,
    parse_h3_fixture_bytes,
)


FIXTURE_CASES = (
    (H3_COUPLED_FIXTURE_PATH, "h3-coupled-v1"),
    (H3_ZERO_CONTROL_FIXTURE_PATH, "h3-zero-control-v1"),
)


def _fixture(path, fixture_id):
    return parse_h3_fixture_bytes(
        path.read_bytes(), expected_fixture_id=fixture_id
    )


@pytest.mark.parametrize(("path", "fixture_id"), FIXTURE_CASES)
def test_normalized_scalar_factors_reconstruct_log_joint_and_canonical_law(
    path, fixture_id
) -> None:
    fixture = _fixture(path, fixture_id)
    model = H3GenerativeModel.from_fixture(fixture)
    y = torch.tensor([0.3, -0.4, 0.7, -0.2], dtype=torch.float64)

    factor_values = tuple(factor.log_prob(y) for factor in model.factors)
    expected_precision = torch.stack(
        tuple(
            torch.outer(factor.row, factor.row) / factor.variance
            for factor in model.factors
        )
    ).sum(dim=0)
    expected_natural = torch.stack(
        tuple(
            factor.target * factor.row / factor.variance
            for factor in model.factors
        )
    ).sum(dim=0)
    expected_constant = torch.stack(
        tuple(
            -0.5
            * (
                factor.target.square() / factor.variance
                + torch.log(2.0 * torch.pi * factor.variance)
            )
            for factor in model.factors
        )
    ).sum()
    canonical = model.canonical_joint()

    assert len(model.factors) == 6
    assert torch.equal(model.log_joint(y), torch.stack(factor_values).sum())
    assert torch.equal(canonical.precision, expected_precision)
    assert torch.equal(canonical.natural, expected_natural)
    assert torch.equal(canonical.log_constant, expected_constant)
    canonical_value = (
        -0.5 * y @ canonical.precision @ y
        + canonical.natural @ y
        + canonical.log_constant
    )
    comparison_allowance = (
        4096.0
        * torch.finfo(torch.float64).eps
        * max(1.0, abs(model.log_joint(y).item()), abs(canonical_value.item()))
    )
    assert abs(model.log_joint(y).item() - canonical_value.item()) <= comparison_allowance


def test_reference_model_uses_only_factors_and_keeps_y_autograd_live() -> None:
    fixture = _fixture(H3_COUPLED_FIXTURE_PATH, "h3-coupled-v1")
    altered_reference = replace(
        fixture,
        reference_posterior_precision=(
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        reference_posterior_natural=(9.0, 8.0, 7.0, 6.0),
        reference_log_evidence=123.0,
        reference_analytic_factorized_reverse_kl=45.0,
    )
    original = H3GenerativeModel.from_fixture(fixture)
    changed = H3GenerativeModel.from_fixture(altered_reference)
    y = torch.tensor(
        [0.1, -0.2, 0.4, 0.8], dtype=torch.float64, requires_grad=True
    )

    value = changed.log_joint(y)
    gradient = torch.autograd.grad(value, y)[0]

    assert value.requires_grad
    assert bool(torch.isfinite(gradient).all())
    assert torch.equal(changed.canonical_joint().precision, original.canonical_joint().precision)
    assert torch.equal(changed.canonical_joint().natural, original.canonical_joint().natural)
    assert torch.equal(
        changed.canonical_joint().log_constant,
        original.canonical_joint().log_constant,
    )


def test_factor_and_canonical_records_own_their_tensor_storage() -> None:
    row = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    target = torch.tensor(0.0, dtype=torch.float64)
    variance = torch.tensor(1.0, dtype=torch.float64)
    factor = H3ScalarGaussianFactor(row, target, variance)
    row[0] = 7.0
    target.fill_(8.0)
    variance.fill_(9.0)

    assert torch.equal(
        factor.row, torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    )
    assert factor.target.item() == 0.0
    assert factor.variance.item() == 1.0
    exposed = factor.row
    exposed[0] = 11.0
    assert factor.row[0].item() == 1.0

    model = H3GenerativeModel.from_fixture(
        _fixture(H3_COUPLED_FIXTURE_PATH, "h3-coupled-v1")
    )
    canonical = model.canonical_joint()
    assert isinstance(canonical, H3CanonicalJoint)
    exposed_precision = canonical.precision
    exposed_precision[0, 0] = 999.0
    assert canonical.precision[0, 0].item() != 999.0


@pytest.mark.parametrize(
    ("row", "target", "variance", "message"),
    (
        (
            torch.zeros(3, dtype=torch.float64),
            torch.zeros((), dtype=torch.float64),
            torch.ones((), dtype=torch.float64),
            "row",
        ),
        (
            torch.zeros(4, dtype=torch.float32),
            torch.zeros((), dtype=torch.float64),
            torch.ones((), dtype=torch.float64),
            "float64",
        ),
        (
            torch.zeros(4, dtype=torch.float64),
            torch.zeros(1, dtype=torch.float64),
            torch.ones((), dtype=torch.float64),
            "target",
        ),
        (
            torch.zeros(4, dtype=torch.float64),
            torch.zeros((), dtype=torch.float64),
            torch.zeros((), dtype=torch.float64),
            "positive",
        ),
        (
            torch.tensor([0.0, 0.0, float("nan"), 0.0], dtype=torch.float64),
            torch.zeros((), dtype=torch.float64),
            torch.ones((), dtype=torch.float64),
            "finite",
        ),
    ),
)
def test_scalar_factor_rejects_malformed_tensor_contracts(
    row: torch.Tensor,
    target: torch.Tensor,
    variance: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        H3ScalarGaussianFactor(row, target, variance)


def test_model_rejects_malformed_values_without_repair() -> None:
    model = H3GenerativeModel.from_fixture(
        _fixture(H3_COUPLED_FIXTURE_PATH, "h3-coupled-v1")
    )
    with pytest.raises(ValueError, match="shape"):
        model.log_joint(torch.zeros(3, dtype=torch.float64))
    with pytest.raises(ValueError, match="float64"):
        model.log_joint(torch.zeros(4, dtype=torch.float32))
    with pytest.raises(ValueError, match="finite"):
        model.log_joint(
            torch.tensor([0.0, 0.0, 0.0, float("inf")], dtype=torch.float64)
        )
    with pytest.raises(ValueError, match="H3Fixture"):
        H3GenerativeModel.from_fixture(object())
