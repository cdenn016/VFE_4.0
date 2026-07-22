from __future__ import annotations

import math
import sys

import pytest
import torch

from vfe4.numerics import DenseCholeskyPrecision, InformationGaussian
from vfe4.types import MatrixBlock, PrecisionFactor


def _correlated_precision() -> torch.Tensor:
    return torch.tensor(
        [[4.0, 1.0, 0.5], [1.0, 3.0, -0.25], [0.5, -0.25, 2.0]],
        dtype=torch.float64,
    )


def test_dense_precision_validates_and_owns_a_float64_spd_matrix() -> None:
    matrix = _correlated_precision()
    factor = DenseCholeskyPrecision(matrix)
    matrix[0, 0] = -1.0

    assert isinstance(factor, PrecisionFactor)
    assert factor.dimension == 3
    assert factor.diagnostics.dimension == 3
    assert factor.diagnostics.min_cholesky_pivot > 0.0
    assert factor.diagnostics.lambda_min > 0.0
    assert factor.diagnostics.lambda_max >= factor.diagnostics.lambda_min
    assert factor.diagnostics.kappa_2 == pytest.approx(
        factor.diagnostics.lambda_max / factor.diagnostics.lambda_min
    )
    assert torch.equal(
        factor.solve(torch.tensor([4.0, 1.0, 0.5], dtype=torch.float64)),
        torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
    )


@pytest.mark.parametrize(
    "matrix",
    [
        torch.eye(2, dtype=torch.float32),
        torch.tensor([[1.0, float("nan")], [float("nan"), 1.0]], dtype=torch.float64),
        torch.tensor([[2.0, 1.0], [0.0, 2.0]], dtype=torch.float64),
        torch.tensor([[1.0, 2.0], [2.0, 1.0]], dtype=torch.float64),
        torch.ones((2, 3), dtype=torch.float64),
    ],
)
def test_dense_precision_rejects_invalid_matrices(matrix: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="matrix"):
        DenseCholeskyPrecision(matrix)


def test_dense_precision_solves_vectors_and_matrices_exactly() -> None:
    factor = DenseCholeskyPrecision(torch.diag(torch.tensor([2.0, 4.0, 8.0], dtype=torch.float64)))

    torch.testing.assert_close(
        factor.solve(torch.tensor([2.0, 8.0, 24.0], dtype=torch.float64)),
        torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
        rtol=0.0,
        atol=1e-15,
    )
    torch.testing.assert_close(
        factor.solve(
            torch.tensor([[2.0, 4.0], [8.0, 12.0], [24.0, 32.0]], dtype=torch.float64)
        ),
        torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], dtype=torch.float64),
        rtol=0.0,
        atol=1e-15,
    )


@pytest.mark.parametrize(
    "rhs",
    [
        torch.ones(2, dtype=torch.float64),
        torch.ones(3, dtype=torch.float32),
        torch.tensor([1.0, float("inf"), 2.0], dtype=torch.float64),
        torch.ones((3, 1, 1), dtype=torch.float64),
    ],
)
def test_dense_precision_rejects_invalid_solve_rhs(rhs: torch.Tensor) -> None:
    factor = DenseCholeskyPrecision(_correlated_precision())

    with pytest.raises(ValueError, match="rhs"):
        factor.solve(rhs)


def test_dense_precision_logdet_is_twice_the_log_cholesky_diagonal() -> None:
    matrix = _correlated_precision()
    factor = DenseCholeskyPrecision(matrix)
    expected = 2.0 * torch.log(torch.diagonal(torch.linalg.cholesky(matrix))).sum()

    assert factor.logdet().item() == pytest.approx(expected.item())


def test_selected_inverse_returns_rectangular_blocks_without_aliasing() -> None:
    matrix = _correlated_precision()
    factor = DenseCholeskyPrecision(matrix)
    block = MatrixBlock(rows=(2, 0), columns=(1,))

    selected = factor.selected_inverse((block,))
    expected = torch.linalg.solve(matrix, torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64))[[2, 0]].unsqueeze(1)

    assert selected[block].shape == (2, 1)
    assert torch.allclose(selected[block], expected, rtol=0.0, atol=1e-15)
    selected[block][0, 0] = 99.0
    assert torch.allclose(
        factor.selected_inverse((block,))[block], expected, rtol=0.0, atol=1e-15
    )


@pytest.mark.parametrize(
    "block",
    [
        MatrixBlock(rows=(0,), columns=(0, 1, 2)),
        MatrixBlock(rows=(3,), columns=(0,)),
        MatrixBlock(rows=(0,), columns=(3,)),
    ],
)
def test_selected_inverse_rejects_full_or_out_of_range_requests(block: MatrixBlock) -> None:
    factor = DenseCholeskyPrecision(_correlated_precision())

    with pytest.raises(ValueError, match="block|column"):
        factor.selected_inverse((block,))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rows": (0, 0), "columns": (1,)},
        {"rows": (0,), "columns": (1, 1)},
        {"rows": (), "columns": (1,)},
        {"rows": (0,), "columns": ()},
    ],
)
def test_matrix_block_rejects_duplicate_or_empty_indices(kwargs: dict[str, tuple[int, ...]]) -> None:
    with pytest.raises(ValueError, match="rows|columns"):
        MatrixBlock(**kwargs)


def test_dense_precision_sampling_is_deterministic_for_supplied_noise() -> None:
    factor = DenseCholeskyPrecision(torch.diag(torch.tensor([4.0, 9.0], dtype=torch.float64)))
    noise = torch.tensor([2.0, -3.0], dtype=torch.float64)

    first = factor.sample(noise)
    noise[0] = 20.0

    assert torch.equal(first, torch.tensor([1.0, -1.0], dtype=torch.float64))
    assert torch.equal(
        factor.sample(torch.tensor([2.0, -3.0], dtype=torch.float64)), first
    )


def test_quadratic_and_trace_inverse_product_match_hand_calculation() -> None:
    factor = DenseCholeskyPrecision(torch.diag(torch.tensor([2.0, 4.0], dtype=torch.float64)))
    left = DenseCholeskyPrecision(torch.diag(torch.tensor([3.0, 8.0], dtype=torch.float64)))

    assert factor.quadratic(torch.tensor([2.0, -1.0], dtype=torch.float64)).item() == pytest.approx(12.0)
    assert factor.trace_inverse_product(left).item() == pytest.approx(3.5)


def test_trace_inverse_product_rejects_wrong_factor_or_dimension() -> None:
    factor = DenseCholeskyPrecision(torch.eye(2, dtype=torch.float64))

    with pytest.raises(ValueError, match="implementation"):
        factor.trace_inverse_product(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="dimension"):
        factor.trace_inverse_product(DenseCholeskyPrecision(torch.eye(3, dtype=torch.float64)))


def test_information_gaussian_matches_closed_form_density_normalizer_and_entropy() -> None:
    h = torch.tensor([2.0, -4.0], dtype=torch.float64)
    precision = torch.diag(torch.tensor([2.0, 4.0], dtype=torch.float64))
    gaussian = InformationGaussian.from_information(h, precision)
    value = torch.tensor([2.0, 0.5], dtype=torch.float64)
    mean = torch.tensor([1.0, -1.0], dtype=torch.float64)
    covariance = torch.diag(torch.tensor([0.5, 0.25], dtype=torch.float64))
    expected_distribution = torch.distributions.MultivariateNormal(mean, covariance)

    torch.testing.assert_close(gaussian.mean(), mean, rtol=0.0, atol=1e-15)
    assert gaussian.log_prob(value).item() == pytest.approx(
        expected_distribution.log_prob(value).item()
    )
    expected_log_normalizer = 0.5 * (
        torch.dot(h, mean).item() - torch.logdet(precision).item() + 2 * math.log(2.0 * math.pi)
    )
    assert gaussian.log_normalizer().item() == pytest.approx(expected_log_normalizer)
    assert gaussian.entropy().item() == pytest.approx(expected_distribution.entropy().item())


def test_information_gaussian_owns_inputs_and_exposes_only_cloned_information() -> None:
    h = torch.tensor([0.4, -0.2], dtype=torch.float64)
    precision = torch.tensor([[2.0, 0.3], [0.3, 1.5]], dtype=torch.float64)
    gaussian = InformationGaussian.from_information(h, precision)
    expected_mean = gaussian.mean()
    h[0] = 100.0
    precision[0, 0] = 100.0
    returned_h = gaussian.h
    returned_j = gaussian.J
    returned_h[0] = -100.0
    returned_j[0, 0] = -100.0

    assert torch.equal(gaussian.mean(), expected_mean)
    assert gaussian.h[0].item() == pytest.approx(0.4)
    assert gaussian.J[0, 0].item() == pytest.approx(2.0)
    assert not hasattr(gaussian, "covariance")
    assert not hasattr(gaussian, "inverse")
    assert not hasattr(gaussian, "moment_matrix")


@pytest.mark.parametrize(
    ("h", "precision"),
    [
        (torch.ones(2, dtype=torch.float32), torch.eye(2, dtype=torch.float64)),
        (torch.tensor([0.0, float("nan")], dtype=torch.float64), torch.eye(2, dtype=torch.float64)),
        (torch.ones((2, 1), dtype=torch.float64), torch.eye(2, dtype=torch.float64)),
        (torch.ones(3, dtype=torch.float64), torch.eye(2, dtype=torch.float64)),
    ],
)
def test_information_gaussian_rejects_invalid_information_inputs(
    h: torch.Tensor, precision: torch.Tensor
) -> None:
    with pytest.raises(ValueError, match="h|dimension"):
        InformationGaussian.from_information(h, precision)


def test_information_gaussian_rejects_invalid_values_and_nonfinite_derivations() -> None:
    gaussian = InformationGaussian.from_information(
        torch.zeros(1, dtype=torch.float64), torch.eye(1, dtype=torch.float64)
    )

    with pytest.raises(ValueError, match="value"):
        gaussian.log_prob(torch.ones(2, dtype=torch.float64))
    with pytest.raises(ValueError, match="value"):
        gaussian.log_prob(torch.ones(1, dtype=torch.float32))
    with pytest.raises(ValueError, match="log_prob"):
        gaussian.log_prob(torch.tensor([sys.float_info.max], dtype=torch.float64))


def test_information_gaussian_oriented_kl_matches_torch_kl_q_to_p() -> None:
    q = InformationGaussian.from_information(
        torch.tensor([0.4, -0.2], dtype=torch.float64),
        torch.tensor([[2.0, 0.3], [0.3, 1.5]], dtype=torch.float64),
    )
    p = InformationGaussian.from_information(
        torch.tensor([-0.1, 0.5], dtype=torch.float64),
        torch.tensor([[1.4, -0.2], [-0.2, 2.2]], dtype=torch.float64),
    )
    q_reference = torch.distributions.MultivariateNormal(q.mean(), precision_matrix=q.J)
    p_reference = torch.distributions.MultivariateNormal(p.mean(), precision_matrix=p.J)
    expected = torch.distributions.kl_divergence(q_reference, p_reference)

    result = q.oriented_kl(p)

    assert result.item() >= 0.0
    assert result.item() == pytest.approx(expected.item())
    assert not hasattr(q, "covariance")
    assert torch.equal(q.factor.solve(q.h), q.mean())


def test_information_gaussian_oriented_kl_rejects_wrong_dimension() -> None:
    q = InformationGaussian.from_information(
        torch.zeros(2, dtype=torch.float64), torch.eye(2, dtype=torch.float64)
    )
    p = InformationGaussian.from_information(
        torch.zeros(3, dtype=torch.float64), torch.eye(3, dtype=torch.float64)
    )

    with pytest.raises(ValueError, match="dimension"):
        q.oriented_kl(p)


def test_selected_moment_blocks_adds_the_mean_outer_product() -> None:
    gaussian = InformationGaussian.from_information(
        torch.tensor([2.0, 6.0, -4.0], dtype=torch.float64),
        torch.diag(torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)),
    )
    block = MatrixBlock(rows=(2, 0), columns=(1,))

    selected = gaussian.selected_moment_blocks((block,))

    torch.testing.assert_close(
        selected[block],
        torch.tensor([[-2.0], [2.0]], dtype=torch.float64),
        rtol=0.0,
        atol=1e-15,
    )
