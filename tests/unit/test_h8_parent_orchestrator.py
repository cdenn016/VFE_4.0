from __future__ import annotations

import pytest

from vfe4.types.h8 import (
    H8_NEGATIVE_CONTROL_IDS,
    H8_PRODUCTION_SEEDS,
    H8ChildRequest,
)


def test_parent_preallocates_the_frozen_thirty_request_sequence() -> None:
    from verification.h8_orchestrator import build_h8_child_request_plan

    config_sha256 = "a" * 64
    protocol_sha256 = "b" * 64

    plan = build_h8_child_request_plan(
        config_sha256=config_sha256,
        protocol_sha256=protocol_sha256,
    )

    expected_identities = (
        *(
            ("production", seed, repetition, None)
            for seed in (20260721, 20260722, 20260723)
            for repetition in range(5)
        ),
        *(
            ("profiler", seed, None, None)
            for seed in (20260721, 20260722, 20260723)
        ),
        *(
            ("negative_control", 20260721, None, control_id)
            for control_id in (
                "torch_matrix_d_d",
                "torch_flat_d2",
                "torch_near_d2",
                "torch_length_d",
                "torch_block_pair_slab",
                "torch_triangular_pair_storage",
                "torch_pair_stack",
                "torch_eye_full_rhs",
                "torch_dense_eigvalsh",
                "numpy_matrix_d_d",
                "numpy_outer_d_d",
                "numpy_matmul_d_d",
            )
        ),
    )
    observed_identities = tuple(
        (
            request.mode,
            request.seed,
            request.repetition,
            request.control_id,
        )
        for request in plan
    )

    assert type(plan) is tuple
    assert len(plan) == 30
    assert all(type(request) is H8ChildRequest for request in plan)
    assert observed_identities == expected_identities
    assert tuple(
        request.seed
        for request in plan
        if request.mode == "production" and request.repetition == 0
    ) == H8_PRODUCTION_SEEDS
    assert tuple(
        request.control_id
        for request in plan
        if request.mode == "negative_control"
    ) == H8_NEGATIVE_CONTROL_IDS
    assert all(
        request.config_sha256 == config_sha256
        and request.protocol_sha256 == protocol_sha256
        for request in plan
    )
    with pytest.raises(AttributeError):
        plan[0].seed = 1  # type: ignore[misc]
