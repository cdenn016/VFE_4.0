from __future__ import annotations

import pytest
import torch

from vfe4.numerics.block_layout import BlockChainLayout, BlockId
from vfe4.numerics.block_tridiagonal import BlockTridiagonalCholesky
from vfe4.types.h8 import BlockTridiagonalPrecision


def _problem() -> tuple[BlockChainLayout, BlockTridiagonalPrecision, torch.Tensor]:
    layout = BlockChainLayout(horizon=2, d_z=1, d_m=1)
    diag = torch.stack(
        (
            torch.tensor([[4.0, 0.2], [0.2, 3.5]], dtype=torch.float64),
            torch.tensor([[4.5, -0.1], [-0.1, 4.0]], dtype=torch.float64),
            torch.tensor([[3.8, 0.15], [0.15, 4.2]], dtype=torch.float64),
        )
    )
    lower = torch.stack(
        (
            torch.tensor([[0.25, 0.05], [-0.1, 0.2]], dtype=torch.float64),
            torch.tensor([[0.2, -0.05], [0.08, 0.15]], dtype=torch.float64),
        )
    )
    dense = torch.zeros(
        (layout.dimension, layout.dimension),
        dtype=torch.float64,
    )
    for population in range(layout.population_size):
        block_slice = layout.block_slice(population)
        dense[block_slice, block_slice] = diag[population]
        if population:
            previous = layout.block_slice(population - 1)
            dense[block_slice, previous] = lower[population - 1]
            dense[previous, block_slice] = lower[population - 1].T
    return layout, BlockTridiagonalPrecision(layout, diag, lower), dense


def test_block_factor_solve_and_logdet_match_the_small_dense_oracle() -> None:
    layout, precision, dense = _problem()
    factor = BlockTridiagonalCholesky.factorize(precision)
    rhs = torch.arange(
        1,
        layout.dimension + 1,
        dtype=torch.float64,
    ).reshape(layout.population_size, layout.block_size)

    observed = factor.solve(rhs)
    expected = torch.linalg.solve(dense, rhs.reshape(-1)).reshape_as(rhs)

    assert torch.max(torch.abs(observed - expected)).item() < 2e-14
    assert torch.abs(factor.logdet() - torch.linalg.slogdet(dense).logabsdet) < 2e-14
    assert factor.fill.matches_expected_fill
    assert factor.pattern.precision_offsets == (-1, 0, 1)


def test_selected_blocks_sample_quadratic_trace_and_condition_are_local() -> None:
    layout, precision, dense = _problem()
    factor = BlockTridiagonalCholesky.factorize(precision)
    selected = factor.selected_inverse(layout.stored_block_ids)
    selected_diag, selected_lower = selected._block_refs()
    dense_inverse = torch.linalg.inv(dense)
    for population in range(layout.population_size):
        block_slice = layout.block_slice(population)
        assert torch.max(
            torch.abs(
                selected_diag[population]
                - dense_inverse[block_slice, block_slice]
            )
        ).item() < 3e-14
        if population:
            previous = layout.block_slice(population - 1)
            assert torch.max(
                torch.abs(
                    selected_lower[population - 1]
                    - dense_inverse[block_slice, previous]
                )
            ).item() < 3e-14

    value = torch.linspace(
        -0.5,
        0.5,
        layout.dimension,
        dtype=torch.float64,
    ).reshape(layout.population_size, layout.block_size)
    noise = torch.linspace(
        0.1,
        0.6,
        layout.dimension,
        dtype=torch.float64,
    ).reshape_as(value)
    dense_factor = torch.linalg.cholesky(dense)
    expected_sample = torch.linalg.solve_triangular(
        dense_factor.T,
        noise.reshape(-1, 1),
        upper=True,
    ).reshape_as(noise)
    assert torch.max(torch.abs(factor.sample(noise) - expected_sample)).item() < 2e-14
    assert torch.abs(
        factor.quadratic(value)
        - value.reshape(-1) @ dense @ value.reshape(-1)
    ) < 2e-14
    assert torch.abs(
        factor.trace_inverse_product(precision)
        - torch.trace(dense @ dense_inverse)
    ) < 3e-13
    assert factor.diagnostics.kappa_1_estimate > 0.0
    assert factor.counters.selected_coverage_complete
    assert factor.counters.maximum_sample_rhs_width == 1


def test_backend_rejects_global_width_and_off_pattern_selection_without_aliasing() -> None:
    layout, precision, _ = _problem()
    source_diag, _ = precision._block_refs()
    with pytest.raises(TypeError, match="factory-only"):
        BlockTridiagonalCholesky()
    factor = BlockTridiagonalCholesky.factorize(precision)
    frozen_factor = factor.diagonal_factor
    source_diag.add_(100.0)
    assert torch.equal(factor.diagonal_factor, frozen_factor)

    with pytest.raises(ValueError, match="width"):
        factor.solve(
            torch.zeros(
                (
                    layout.population_size,
                    layout.block_size,
                    layout.block_size + 1,
                ),
                dtype=torch.float64,
            )
        )
    with pytest.raises(ValueError, match="canonical"):
        factor.selected_inverse((BlockId.diagonal(0),))
    with pytest.raises(ValueError, match="explicit"):
        factor.solve(
            torch.zeros(
                (layout.dimension, layout.dimension),
                dtype=torch.float64,
            )
        )
    with pytest.raises(ValueError, match="exactly"):
        factor.sample(
            torch.zeros(
                (
                    layout.population_size,
                    layout.block_size,
                    2,
                ),
                dtype=torch.float64,
            )
        )
    assert factor.counters.attempted_forbidden_rhs_widths == (
        layout.block_size + 1,
        layout.dimension,
        2,
    )
    assert factor.counters.attempted_forbidden_selected_blocks == 1
    assert factor.counters.maximum_sample_rhs_width == 2


def test_condition_diagnostic_never_flattens_a_population_vector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, precision, _ = _problem()
    factor = BlockTridiagonalCholesky.factorize(precision)

    def forbidden_flatten(*args: object, **kwargs: object) -> torch.Tensor:
        del args, kwargs
        raise AssertionError("condition diagnostics cannot flatten or reshape")

    monkeypatch.setattr(torch.Tensor, "flatten", forbidden_flatten)
    monkeypatch.setattr(torch.Tensor, "reshape", forbidden_flatten)
    diagnostics = factor.diagnostics
    assert diagnostics.index_sha256
    assert diagnostics.sign_sha256
