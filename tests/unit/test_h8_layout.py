from __future__ import annotations

import dataclasses

import pytest
import torch

from vfe4.numerics.block_layout import (
    BlockChainLayout,
    BlockId,
    H8_MAX_STORAGE_SCALARS,
    H8_REFERENCE_CHANNEL_DIMENSION,
    H8_REFERENCE_HORIZON,
)
from vfe4.types.h8 import (
    H8_CORRECTNESS_CASES,
    H8_NEGATIVE_CONTROL_IDS,
    H8_PRODUCTION_SEEDS,
    BackendCounterSnapshot,
    BlockFillRecord,
    BlockPatternRecord,
    BlockStorageExpectation,
    BlockStorageRecord,
    BlockTridiagonalPrecision,
    BlockWorkspaceRecord,
    H8ChildRequest,
    H8ResourceRecord,
)


def test_h8_reference_layout_has_exact_population_major_arithmetic() -> None:
    layout = BlockChainLayout(horizon=128, d_z=20, d_m=20)

    assert layout.population_size == 129
    assert layout.block_size == 40
    assert layout.dimension == 5_160
    assert layout.diagonal_scalar_count == 206_400
    assert layout.lower_scalar_count == 204_800
    assert layout.band_storage_scalar_count == H8_MAX_STORAGE_SCALARS
    assert layout.information_scalar_count == 5_160
    assert layout.dense_scalar_count == 26_625_600
    assert layout.state_slice(0) == slice(0, 20)
    assert layout.model_slice(0) == slice(20, 40)
    assert layout.state_slice(1) == slice(40, 60)
    assert layout.coordinate_id(59) == (1, "z", 19)
    assert layout.coordinate_id(60) == (1, "m", 0)
    assert layout.flatten_coordinate(128, "m", 19) == 5_159


def test_h8_layout_owns_one_canonical_lower_band_without_upper_storage() -> None:
    layout = BlockChainLayout(horizon=2, d_z=1, d_m=1)

    expected = (
        BlockId.diagonal(0),
        BlockId.diagonal(1),
        BlockId.diagonal(2),
        BlockId.lower(1),
        BlockId.lower(2),
    )
    assert layout.stored_block_ids == expected
    assert layout.require_complete_stored_blocks(expected) == expected


def test_h8_layout_rejects_invalid_dimensions_and_storage_before_allocation() -> None:
    invalid = (
        {"horizon": True, "d_z": 1, "d_m": 1},
        {"horizon": 0, "d_z": 1, "d_m": 1},
        {"horizon": 1, "d_z": 0, "d_m": 1},
        {"horizon": 1, "d_z": 1, "d_m": -1},
        {"horizon": 128, "d_z": 21, "d_m": 20},
    )
    for values in invalid:
        with pytest.raises(ValueError):
            BlockChainLayout(**values)


def test_h8_block_ids_and_complete_inventory_fail_closed() -> None:
    layout = BlockChainLayout(horizon=2, d_z=1, d_m=1)

    for invalid in (
        (BlockId.diagonal(0),),
        layout.stored_block_ids + (BlockId.diagonal(0),),
        (
            BlockId.diagonal(0),
            BlockId.diagonal(1),
            BlockId.diagonal(2),
            BlockId(kind="lower_adjacent", row=2, column=1),
            BlockId(kind="lower_adjacent", row=1, column=0),
        ),
    ):
        with pytest.raises(ValueError):
            layout.require_complete_stored_blocks(invalid)

    with pytest.raises(ValueError):
        BlockId(kind="diagonal", row=1, column=0)
    with pytest.raises(ValueError):
        BlockId(kind="lower_adjacent", row=2, column=0)
    with pytest.raises(ValueError):
        layout.require_complete_stored_blocks(list(layout.stored_block_ids))


def test_h8_layout_records_are_immutable() -> None:
    layout = BlockChainLayout(horizon=1, d_z=1, d_m=1)
    block = BlockId.diagonal(0)

    with pytest.raises(dataclasses.FrozenInstanceError):
        layout.horizon = 2  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        block.row = 1  # type: ignore[misc]


def test_h8_layout_shape_guards_bind_native_axes_and_widths() -> None:
    layout = BlockChainLayout(horizon=2, d_z=1, d_m=1)

    assert layout.require_block_vector_shape((3, 2)) == (3, 2)
    assert layout.require_block_matrix_shape((3, 2, 2), adjacent=False) == (
        3,
        2,
        2,
    )
    assert layout.require_block_matrix_shape((2, 2, 2), adjacent=True) == (
        2,
        2,
        2,
    )
    assert layout.require_rhs_shape((3, 2, 2)) == (3, 2, 2)
    assert layout.require_sample_shape((3, 2)) == (3, 2)

    for invalid in ((3, 2, 3), (6,), (3, True), (3, 2, 0)):
        with pytest.raises(ValueError):
            layout.require_rhs_shape(invalid)
    with pytest.raises(ValueError):
        layout.require_block_matrix_shape((3, 2, 2), adjacent=True)
    with pytest.raises(ValueError):
        layout.require_bounded_storage_scalar_count(
            H8_MAX_STORAGE_SCALARS + 1,
            name="precision",
        )


def test_h8_precision_owns_exact_float64_cpu_block_storage() -> None:
    layout = BlockChainLayout(horizon=1, d_z=1, d_m=1)
    diag = torch.eye(2, dtype=torch.float64).repeat(2, 1, 1)
    lower = torch.zeros((1, 2, 2), dtype=torch.float64)

    precision = BlockTridiagonalPrecision(layout, diag, lower)
    diag[0, 0, 0] = 9.0

    assert precision.diag[0, 0, 0].item() == 1.0
    assert BlockPatternRecord().precision_offsets == (-1, 0, 1)
    expected = BlockStorageExpectation.for_layout(layout)
    assert expected.upper_block_scalar_count == 0

    observed = BlockStorageRecord(
        layout=layout,
        precision_scalar_count=expected.precision_scalar_count + 1,
        factor_scalar_count=expected.factor_scalar_count,
        selected_inverse_scalar_count=expected.selected_inverse_scalar_count,
        information_scalar_count=expected.information_scalar_count,
        upper_block_scalar_count=4,
    )
    assert not observed.matches_expectation
    assert observed.over_cap_categories == ()

    fill = BlockFillRecord(
        layout=layout,
        stored_block_ids=(BlockId.diagonal(0),),
        observed_offband_blocks=1,
        duplicated_upper_blocks=2,
    )
    assert not fill.matches_expected_fill


def test_h8_observation_records_preserve_violations_for_gate_decisions() -> None:
    layout = BlockChainLayout(horizon=1, d_z=1, d_m=1)
    expected = BlockStorageExpectation.for_layout(layout)
    observed = BlockStorageRecord(
        layout=layout,
        precision_scalar_count=H8_MAX_STORAGE_SCALARS + 1,
        factor_scalar_count=expected.factor_scalar_count,
        selected_inverse_scalar_count=expected.selected_inverse_scalar_count,
        information_scalar_count=expected.information_scalar_count,
        upper_block_scalar_count=4,
    )
    workspace = BlockWorkspaceRecord(
        maximum_shape=(2, 2, 3),
        maximum_scalar_count=12,
        maximum_rhs_width=3,
        attempted_forbidden_rhs_widths=(3,),
    )
    counters = BackendCounterSnapshot(
        layout=layout,
        factorization_calls=1,
        forward_substitution_calls=1,
        backward_substitution_calls=1,
        solve_calls=1,
        logdet_calls=1,
        selected_inverse_calls=1,
        sample_calls=1,
        quadratic_calls=1,
        trace_calls=1,
        sparse_matvec_calls=1,
        maximum_rhs_width=3,
        maximum_sample_rhs_width=2,
        selected_block_ids=(BlockId.diagonal(0),),
        selected_block_count=1,
        attempted_forbidden_selected_blocks=1,
        attempted_forbidden_rhs_widths=(3,),
    )

    assert observed.over_cap_categories == ("precision",)
    assert observed.upper_block_scalar_count == 4
    assert workspace.maximum_rhs_width == 3
    assert not counters.selected_coverage_complete
    assert counters.maximum_sample_rhs_width == 2


def test_h8_frozen_protocol_literals_and_child_request_are_exact() -> None:
    digest = "a" * 64
    request = H8ChildRequest(
        mode="production",
        seed=H8_PRODUCTION_SEEDS[0],
        repetition=0,
        config_sha256=digest,
        protocol_sha256=digest,
        control_id=None,
    )
    resource = H8ResourceRecord(
        adapter="test-adapter",
        adapter_sha256=digest,
        pre_current_rss_bytes=100,
        pre_lifetime_peak_bytes=120,
        pre_private_bytes=90,
        post_current_rss_bytes=130,
        post_lifetime_peak_bytes=150,
        post_private_bytes=110,
        conservative_incremental_hwm_bytes=50,
        peak_to_peak_diagnostic_bytes=30,
        parent_elapsed_ns=10,
        child_elapsed_ns=8,
    )

    assert H8_REFERENCE_HORIZON == 128
    assert H8_REFERENCE_CHANNEL_DIMENSION == 20
    assert len(H8_CORRECTNESS_CASES) == 12
    assert len(H8_NEGATIVE_CONTROL_IDS) == 12
    assert request.repetition == 0
    assert resource.conservative_incremental_hwm_bytes == 50
    with pytest.raises(dataclasses.FrozenInstanceError):
        request.repetition = 1  # type: ignore[misc]
    with pytest.raises(ValueError):
        H8ChildRequest(
            mode="production",
            seed=H8_PRODUCTION_SEEDS[0],
            repetition=5,
            config_sha256=digest,
            protocol_sha256=digest,
            control_id=None,
        )
