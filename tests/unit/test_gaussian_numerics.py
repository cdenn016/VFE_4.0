from __future__ import annotations

import sys

import pytest
import torch

from vfe4.numerics import gaussian_log_prob, require_spd


def test_require_spd_returns_owned_float64_matrix() -> None:
    matrix = torch.tensor([[2.0, 0.25], [0.25, 1.0]], dtype=torch.float64)

    checked = require_spd(matrix, name="covariance")
    matrix[0, 0] = -1.0

    assert checked.dtype is torch.float64
    assert checked[0, 0].item() == pytest.approx(2.0)


@pytest.mark.parametrize(
    "matrix",
    [
        torch.tensor([[1.0, 2.0], [2.0, 4.0]], dtype=torch.float64),
        torch.tensor([[1.0, 2.0], [0.0, 1.0]], dtype=torch.float64),
        torch.ones((2, 3), dtype=torch.float64),
        torch.eye(2, dtype=torch.float32),
    ],
)
def test_require_spd_rejects_non_spd_or_wrong_precision(matrix: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="covariance"):
        require_spd(matrix, name="covariance")


def test_gaussian_log_prob_matches_torch_distribution_normalizer() -> None:
    value = torch.tensor([0.25, -0.5], dtype=torch.float64)
    mean = torch.tensor([0.0, 0.5], dtype=torch.float64)
    covariance = torch.tensor([[2.0, 0.25], [0.25, 1.0]], dtype=torch.float64)

    result = gaussian_log_prob(value, mean, covariance)
    expected = torch.distributions.MultivariateNormal(mean, covariance).log_prob(value)

    assert result.item() == pytest.approx(expected.item())


def test_gaussian_log_prob_rejects_mismatched_shape() -> None:
    with pytest.raises(ValueError, match="value"):
        gaussian_log_prob(
            torch.ones(2, dtype=torch.float64),
            torch.ones(1, dtype=torch.float64),
            torch.eye(2, dtype=torch.float64),
        )


def test_gaussian_log_prob_rejects_a_nonfinite_derived_density() -> None:
    with pytest.raises(ValueError, match="gaussian_log_prob"):
        gaussian_log_prob(
            torch.tensor([sys.float_info.max], dtype=torch.float64),
            torch.zeros(1, dtype=torch.float64),
            torch.eye(1, dtype=torch.float64),
        )
