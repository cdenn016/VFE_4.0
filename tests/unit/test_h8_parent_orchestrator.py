from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from verification.h8_budget import (
    H8ChildProcessRecord,
    make_h8_identity_record,
)
from verification.h8_gate import H8PrerequisiteArtifactValidation
from vfe4.config.schema import H8ValidationConfig
from vfe4.types.h8 import (
    H8_NEGATIVE_CONTROL_IDS,
    H8_PRODUCTION_SEEDS,
    H8ChildRequest,
)
from vfe4.types.results import GateStatus


def _parent_identities() -> dict[str, object]:
    return {
        name: make_h8_identity_record(name, payload)
        for name, payload in (
            (
                "hardware",
                {
                    "platform": "test",
                    "system": "test",
                    "machine": "test",
                    "processor": "test",
                    "cpu_count": 1,
                    "python": "test",
                    "implementation": "test",
                },
            ),
            ("affinity", {"adapter": "test", "cpus": [0]}),
            (
                "thread",
                {
                    "environment": {
                        "OMP_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1",
                        "NUMEXPR_NUM_THREADS": "1",
                        "VECLIB_MAXIMUM_THREADS": "1",
                    },
                    "torch_num_threads": 1,
                    "torch_num_interop_threads": 1,
                },
            ),
            (
                "blas",
                {
                    "torch_version": "test",
                    "numpy_version": "test",
                    "torch_config": "test",
                    "numpy_config": "test",
                },
            ),
        )
    }


def _start_authorization(
    *,
    registry_sha256: str = "d" * 64,
    validated_registry_sha256: str | None = None,
):
    from verification.h8_orchestrator import derive_h8_child_start_authorization

    prerequisite_validation = H8PrerequisiteArtifactValidation.create(
        registry_sha256=(
            registry_sha256
            if validated_registry_sha256 is None
            else validated_registry_sha256
        ),
        revalidated_reference_names=(
            "h1_h5",
            "h1_prefix_prior",
            "h6_prefix",
            "h7",
            "h6_prediction",
        ),
        obligations=(),
    )
    return derive_h8_child_start_authorization(
        config=H8ValidationConfig.create(),
        current_registry_sha256=registry_sha256,
        prerequisite_validation=prerequisite_validation,
        correctness_statuses=tuple(
            (cell_id, GateStatus.PASS) for cell_id in range(1, 13)
        ),
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


def test_parent_records_fake_child_timeout_and_stops_at_the_issued_prefix(
    tmp_path: Path,
) -> None:
    from verification.h8_orchestrator import run_h8_parent_attempt

    authorization = _start_authorization()
    observed_invocations = []

    def timed_out_child(invocation):
        observed_invocations.append(invocation)
        return H8ChildProcessRecord(
            timed_out=True,
            exit_code=None,
            stdout=b"",
            stderr=b"fake timeout",
            parent_elapsed_ns=60_000_000_001,
        )

    parent_run = run_h8_parent_attempt(
        authorization=authorization,
        repository_root=tmp_path,
        identities=_parent_identities(),
        base_environment={"PATH": "preserved"},
        child_runner=timed_out_child,
    )

    assert authorization.valid_start is True
    assert len(observed_invocations) == 1
    assert len(parent_run.request_plan) == 30
    assert len(parent_run.issued) == 1
    assert len(parent_run.attempts) == 1
    issued = parent_run.issued[0]
    assert issued.request == parent_run.request_plan[0]
    assert issued.invocation is observed_invocations[0]
    assert issued.process_record.timed_out is True
    assert issued.attempt == parent_run.attempts[0]
    assert issued.attempt.status is GateStatus.FAIL
    assert issued.attempt.reasons == ("child_timeout",)
    assert issued.repository_root == tmp_path.resolve()
    assert len(issued.launch_contract_sha256) == 64
    assert tuple(item.request for item in parent_run.issued) == (
        parent_run.request_plan[0],
    )


def test_parent_cross_binds_fake_child_records_without_launching_a_real_child(
    tmp_path: Path,
) -> None:
    from verification.h8_orchestrator import (
        H8ChildStartAuthorization,
        H8IssuedLaunchRecord,
        run_h8_parent_attempt,
    )

    authorization = _start_authorization()
    outcomes = [
        H8ChildProcessRecord(
            timed_out=False,
            exit_code=0,
            stdout=b"not canonical child JSON\n",
            stderr=b"fake malformed child",
            parent_elapsed_ns=17,
        ),
        OSError(5, "fake spawn denied"),
    ]
    observed_invocations = []

    def malformed_then_spawn_error(invocation):
        observed_invocations.append(invocation)
        outcome = outcomes.pop(0)
        if isinstance(outcome, OSError):
            raise outcome
        return outcome

    parent_run = run_h8_parent_attempt(
        authorization=authorization,
        repository_root=tmp_path,
        identities=_parent_identities(),
        base_environment={},
        child_runner=malformed_then_spawn_error,
    )

    assert len(observed_invocations) == 2
    assert len(parent_run.issued) == 2
    assert parent_run.attempts[0].status is GateStatus.INCONCLUSIVE
    assert parent_run.attempts[1].status is GateStatus.FAIL
    assert parent_run.issued[0].process_record.stdout == (
        b"not canonical child JSON\n"
    )
    assert parent_run.issued[1].process_record.exit_code == -5
    assert parent_run.issued[1].process_record.stderr == b"[Errno 5] fake spawn denied"
    assert parent_run.issued[0].launch_contract_sha256 != (
        parent_run.issued[1].launch_contract_sha256
    )
    assert tuple(item.request for item in parent_run.issued) == (
        parent_run.request_plan[0],
        parent_run.request_plan[1],
    )
    assert tuple(item.attempt for item in parent_run.issued) == parent_run.attempts

    first_issued = parent_run.issued[0]
    forged_attempt = dataclasses.replace(
        first_issued.attempt,
        identities_sha256="f" * 64,
    )
    with pytest.raises(ValueError, match="fresh child classification"):
        H8IssuedLaunchRecord(
            request=first_issued.request,
            invocation=first_issued.invocation,
            process_record=first_issued.process_record,
            repository_root=first_issued.repository_root,
            launch_contract_sha256=first_issued.launch_contract_sha256,
            attempt=forged_attempt,
        )

    assert not hasattr(H8ChildStartAuthorization, "_derive")
    forged_authorization = object.__new__(H8ChildStartAuthorization)
    for field in dataclasses.fields(authorization):
        object.__setattr__(
            forged_authorization,
            field.name,
            getattr(authorization, field.name),
        )
    forged_calls = []

    def forged_child(invocation):
        forged_calls.append(invocation)
        raise AssertionError("forged authorization launched a child")

    with pytest.raises(ValueError, match="factory-issued"):
        run_h8_parent_attempt(
            authorization=forged_authorization,
            repository_root=tmp_path,
            identities=_parent_identities(),
            child_runner=forged_child,
        )
    assert forged_calls == []

    mutated_authorization = _start_authorization()
    object.__setattr__(
        mutated_authorization,
        "authorization_sha256",
        "f" * 64,
    )
    with pytest.raises(ValueError, match="authorization SHA-256 is stale"):
        run_h8_parent_attempt(
            authorization=mutated_authorization,
            repository_root=tmp_path,
            identities=_parent_identities(),
            child_runner=forged_child,
        )
    assert forged_calls == []

    unauthorized = _start_authorization(
        validated_registry_sha256="e" * 64,
    )
    unauthorized_calls = []

    def forbidden_child(invocation):
        unauthorized_calls.append(invocation)
        raise AssertionError("unauthorized parent launched a child")

    blocked_run = run_h8_parent_attempt(
        authorization=unauthorized,
        repository_root=tmp_path,
        identities=_parent_identities(),
        child_runner=forbidden_child,
    )
    assert unauthorized.valid_start is False
    assert unauthorized.obligations == (
        "h8_prerequisite_validation_registry_mismatch",
    )
    assert unauthorized_calls == []
    assert blocked_run.request_plan == ()
    assert blocked_run.issued == ()
    assert blocked_run.attempts == ()
