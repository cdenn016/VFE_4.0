from __future__ import annotations

import dataclasses
import hashlib
import io
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

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
    H8ChildAttemptRecord,
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
                    "release": "test-release",
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
                        "torch_version": "2.9.1",
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
    from verification.h8_orchestrator import (
        _run_h8_parent_attempt_for_test,
    )

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

    parent_run = _run_h8_parent_attempt_for_test(
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
        _run_h8_parent_attempt_for_test,
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

    parent_run = _run_h8_parent_attempt_for_test(
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
    forged_attempt = object.__new__(H8ChildAttemptRecord)
    for field in dataclasses.fields(first_issued.attempt):
        object.__setattr__(
            forged_attempt,
            field.name,
            (
                "f" * 64
                if field.name == "identities_sha256"
                else getattr(first_issued.attempt, field.name)
            ),
        )
    with pytest.raises(ValueError, match="factory-issued|validated process bundle"):
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
        _run_h8_parent_attempt_for_test(
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
        _run_h8_parent_attempt_for_test(
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

    blocked_run = _run_h8_parent_attempt_for_test(
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


def test_parent_authority_is_nonforgeable_and_bound_to_exact_source_run(
    tmp_path: Path,
) -> None:
    from verification.h8_orchestrator import (
        H8ParentAttemptRun,
        _mint_h8_parent_attempt_authority,
        _run_h8_parent_attempt_for_test,
    )
    from verification.h8_parent_authority import (
        H8ParentAttemptAuthority,
        require_h8_parent_attempt_authority,
    )

    authorization = _start_authorization()

    def timed_out_child(_invocation):
        return H8ChildProcessRecord(
            timed_out=True,
            exit_code=None,
            stdout=b"",
            stderr=b"fake timeout",
            parent_elapsed_ns=60_000_000_001,
        )

    parent_run = _run_h8_parent_attempt_for_test(
        authorization=authorization,
        repository_root=tmp_path,
        identities=_parent_identities(),
        child_runner=timed_out_child,
    )
    authority = _mint_h8_parent_attempt_authority(parent_run)

    assert type(authority) is H8ParentAttemptAuthority
    assert authority.child_config_sha256 == H8ValidationConfig.create().config_sha256
    assert authority.protocol_sha256 == authorization.protocol_sha256
    assert authority.authorization_sha256 == authorization.authorization_sha256
    assert authority.attempts is parent_run.attempts
    assert require_h8_parent_attempt_authority(authority) is authority
    assert (
        require_h8_parent_attempt_authority(authority, source_run=parent_run)
        is authority
    )
    assert all(
        len(getattr(authority, name)) == 64
        for name in (
            "request_plan_sha256",
            "issued_prefix_sha256",
            "authority_sha256",
        )
    )

    with pytest.raises(TypeError, match="factory-only"):
        H8ParentAttemptAuthority()

    copied_authority = object.__new__(H8ParentAttemptAuthority)
    for field in dataclasses.fields(authority):
        object.__setattr__(
            copied_authority,
            field.name,
            getattr(authority, field.name),
        )
    with pytest.raises(ValueError, match="factory-issued"):
        require_h8_parent_attempt_authority(copied_authority)

    equal_source_run = H8ParentAttemptRun(
        authorization=parent_run.authorization,
        request_plan=parent_run.request_plan,
        issued=parent_run.issued,
        attempts=parent_run.attempts,
    )
    assert equal_source_run == parent_run
    assert equal_source_run is not parent_run
    with pytest.raises(ValueError, match="source run"):
        require_h8_parent_attempt_authority(
            authority,
            source_run=equal_source_run,
        )

    object.__setattr__(authority, "authority_sha256", "f" * 64)
    with pytest.raises(ValueError, match="authority SHA-256"):
        require_h8_parent_attempt_authority(authority)


def test_parent_authority_rejects_post_mint_source_authorization_replacement(
    tmp_path: Path,
) -> None:
    from verification.h8_orchestrator import (
        _mint_h8_parent_attempt_authority,
        _run_h8_parent_attempt_for_test,
    )
    from verification.h8_parent_authority import (
        require_h8_parent_attempt_authority,
    )

    authorization = _start_authorization()

    def timed_out_child(_invocation):
        return H8ChildProcessRecord(
            timed_out=True,
            exit_code=None,
            stdout=b"",
            stderr=b"fake timeout",
            parent_elapsed_ns=60_000_000_001,
        )

    parent_run = _run_h8_parent_attempt_for_test(
        authorization=authorization,
        repository_root=tmp_path,
        identities=_parent_identities(),
        child_runner=timed_out_child,
    )
    authority = _mint_h8_parent_attempt_authority(parent_run)
    replacement = _start_authorization()
    assert replacement == authorization
    assert replacement is not authorization

    object.__setattr__(parent_run, "authorization", replacement)

    with pytest.raises(ValueError, match="source authorization"):
        require_h8_parent_attempt_authority(authority)


def test_public_parent_identity_collection_uses_neutral_patchable_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from verification import h8_child, h8_orchestrator

    expected = _parent_identities()
    calls = []

    def fake_collector():
        calls.append("called")
        return expected

    def forbidden_child_private(*_args, **_kwargs):
        raise AssertionError("parent reached into child-private identity collection")

    monkeypatch.setattr(
        h8_orchestrator,
        "_collect_h8_runtime_identities",
        fake_collector,
    )
    monkeypatch.setattr(
        h8_child,
        "_collect_identities",
        forbidden_child_private,
    )

    observed = h8_orchestrator.collect_h8_parent_identities()

    assert observed is expected
    assert calls == ["called"]


def test_h8_v2_config_and_protocol_contract_are_complete_and_shared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from verification import (
        h8_child,
        h8_orchestrator,
        h8_preflight,
        h8_protocol,
        h8_wire,
    )
    from verification.h8_budget import (
        H8_CHILD_ENVELOPE_KEYS,
        H8_CHILD_IDENTITY_KEYS,
        H8_CHILD_REQUEST_KEYS,
        H8_CHILD_RESULT_KEYS,
        H8_CHILD_SCHEMA_VERSION,
        H8_OPERATION_SCOPES,
        H8_REQUIRED_NUMPY_PRODUCERS,
        H8_REQUIRED_PASS_DECISIONS,
        H8_REQUIRED_RESIDUALS,
        H8_SCALE_RESIDUAL_SPECS,
        H8_SETUP_SCOPES,
        parse_h8_child_stdout,
    )
    from verification.h8_preflight import _validate_target_h8
    from vfe4.config.resolve import resolve_h8_validation_config
    from vfe4.types.h8 import (
        H8_CORRECTNESS_CASES,
        H8_CORRECTNESS_CONTROL_IDS,
        H8_CORRECTNESS_ORDERED_SOURCE_PAIRS,
        H8_CORRECTNESS_SOURCES,
        H8_NEGATIVE_CONTROL_IDS,
        H8_PROFILER_API_CONTRACT_SHA256,
        H8_PROFILER_MEMORY_SOURCE_SHA256,
        H8_PROFILER_SOURCE_SHA256,
        H8_PRODUCTION_SAMPLE_SEED_PAIRS,
        H8_REQUIRED_OPERATIONS,
    )
    from vfe4.numerics.block_tridiagonal import (
        H8_HAGER_HIGHAM_1NORM_POLICY,
    )

    config = H8ValidationConfig.create()
    expected_schemas = {
        "factor_schema": "h8-block-tridiagonal-cholesky-v1",
        "selected_inverse_schema": (
            "h8-block-takahashi-selected-inverse-v1"
        ),
        "condition_estimator_schema": "HagerHigham1NormEstimate-v1",
        "allocation_schema": "h8-allocation-observability-v1",
        "profiler_raw_event_schema": "h8-torch-profiler-raw-event-v1",
        "child_schema": "h8-child-v2",
    }
    assert config.schema_version == "h8-validation-config-v2"
    assert (
        config.config_sha256
        == "e2bfda7f74fb04688515594df9ed3bf5ab8bfab6cfca54e64170ef47cbab11a5"
    )
    assert {
        name: getattr(config, name) for name in expected_schemas
    } == expected_schemas
    assert (
        h8_preflight.H8_FROZEN_SECTION_SHA256
        == config.config_sha256
    )

    raw = json.loads(config.canonical_json)
    assert resolve_h8_validation_config(raw) == config
    assert _validate_target_h8({"h8": raw}) == raw
    for name, replacement in (
        ("schema_version", "h8-validation-config-v1"),
        ("child_schema", "h8-child-v1"),
        ("factor_schema", "forged-factor-schema"),
    ):
        drifted = dict(raw)
        drifted[name] = replacement
        with pytest.raises(ValueError):
            resolve_h8_validation_config(drifted)
        with pytest.raises(ValueError):
            _validate_target_h8({"h8": drifted})
    with pytest.raises(ValueError, match="SHA-256"):
        dataclasses.replace(config, config_sha256="0" * 64)

    assert hasattr(h8_orchestrator, "_h8_protocol_preimage")
    _h8_protocol_preimage = h8_orchestrator._h8_protocol_preimage
    build_h8_protocol_sha256 = h8_orchestrator.build_h8_protocol_sha256
    preimage = _h8_protocol_preimage(config)
    assert preimage["domain"] == "vfe4.h8.parent-child-protocol.v2"
    assert preimage["validation_config"] == {
        "schema_version": config.schema_version,
        "config_sha256": config.config_sha256,
        **expected_schemas,
    }
    child_contract = preimage["child_contract"]
    assert child_contract == {
        "module": "verification.h8_child",
        "schema_version": "h8-child-v2",
        "request_fields": H8_CHILD_REQUEST_KEYS,
        "envelope_fields": H8_CHILD_ENVELOPE_KEYS,
        "result_fields": H8_CHILD_RESULT_KEYS,
        "identity_fields": H8_CHILD_IDENTITY_KEYS,
    }
    inventories = preimage["execution_inventories"]
    assert inventories["production_sample_seed_pairs"] == (
        H8_PRODUCTION_SAMPLE_SEED_PAIRS
    )
    assert inventories["correctness_seed_table"] == H8_CORRECTNESS_CASES
    assert inventories["correctness_sources"] == H8_CORRECTNESS_SOURCES
    assert inventories["correctness_ordered_source_pairs"] == (
        H8_CORRECTNESS_ORDERED_SOURCE_PAIRS
    )
    assert inventories["correctness_control_ids"] == (
        H8_CORRECTNESS_CONTROL_IDS
    )
    assert inventories["required_operations"] == H8_REQUIRED_OPERATIONS
    assert inventories["operation_scopes"] == tuple(
        (name, H8_OPERATION_SCOPES[name]) for name in H8_REQUIRED_OPERATIONS
    )
    assert inventories["required_residuals"] == H8_REQUIRED_RESIDUALS
    assert inventories["setup_scopes"] == tuple(sorted(H8_SETUP_SCOPES))
    assert inventories["scale_residual_specs"] == tuple(
        (residual_id, H8_SCALE_RESIDUAL_SPECS[residual_id])
        for residual_id in H8_REQUIRED_RESIDUALS
    )
    assert inventories["required_pass_decisions"] == (
        H8_REQUIRED_PASS_DECISIONS
    )
    assert inventories["required_numpy_producers"] == tuple(
        sorted(H8_REQUIRED_NUMPY_PRODUCERS)
    )
    assert tuple(
        item["control_id"] for item in inventories["negative_controls"]
    ) == H8_NEGATIVE_CONTROL_IDS
    assert inventories["production_order"] == tuple(
        (seed, repetition)
        for seed in (20260721, 20260722, 20260723)
        for repetition in range(5)
    )
    assert inventories["profiler_seed_order"] == (
        20260721,
        20260722,
        20260723,
    )

    numerical = preimage["numerical_contract"]
    assert numerical == {
        "eps": float.fromhex("0x1.0000000000000p-52"),
        "rounding_multiplier": 4096.0,
        "solver_relative_budget": 1e-9,
        "max_allowance_fraction": 1e-4,
        "minimum_cholesky_pivot": 1e-8,
        "condition_estimator": {
            "schema": "HagerHigham1NormEstimate-v1",
            "norm": "matrix_1_norm",
            "maximum_iterations": 8,
            "policy": H8_HAGER_HIGHAM_1NORM_POLICY,
            "estimate_is_diagnostic_not_exact_spectrum": True,
        },
        "residual_allowance_policy": {
            "allowance_sum": "math.fsum",
            "component_order": (
                "left_rounding",
                "left_solver",
                "left_quadrature",
                "right_rounding",
                "right_solver",
                "right_quadrature",
                "pair_reduction",
            ),
            "gamma": "n_times_eps_over_1_minus_n_times_eps",
            "operand_rounding": (
                "rounding_multiplier_times_gamma_local_operation_count"
                "_times_max_1_absolute_sum_bound"
            ),
            "operand_solver": (
                "solver_relative_budget_times_max_1_infinity_norm"
                "_iff_solver_produced_else_zero"
            ),
            "pair_reduction": (
                "rounding_multiplier_times_gamma_compared_scalar_count"
                "_plus_1_times_max_1_left_inf_right_inf"
            ),
            "scale": "max(1,left_infinity_norm,right_infinity_norm)",
            "decisive_operator": "<",
            "decisive_fraction": 1e-4,
            "decisive_equality_status": "inconclusive",
            "residual_pass_operator": "<=",
            "residual_equality_status": "pass",
            "condition_estimate_in_allowance": False,
        },
    }
    boundary = preimage["boundary_contract"]
    assert boundary["limits_are_inclusive"] is True
    assert boundary["max_seconds"] == 60.0
    assert boundary["max_process_incremental_bytes"] == 128 * 1024 * 1024
    assert boundary["max_torch_population_bytes"] == 64 * 1024 * 1024
    assert boundary["max_storage_scalars_per_category"] == 411_200
    assert boundary["max_rhs_width"] == 40
    assert boundary["sample_width"] == 1
    assert boundary["offband_fill_limit"] == 0
    assert boundary["forbidden_attempt_limit"] == 0
    runtime = preimage["runtime_contract"]
    assert runtime["device"] == "cpu"
    assert runtime["dtype"] == "float64"
    assert runtime["grad_enabled"] is False
    assert runtime["scale_layout"] == {
        "horizon": 128,
        "d_z": 20,
        "d_m": 20,
    }
    assert runtime["thread_environment"] == tuple(
        (name, "1")
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        )
    )
    assert runtime["torch_num_threads"] == 1
    assert runtime["torch_num_interop_threads"] == 1
    assert runtime["fresh_process_per_request"] is True
    assert runtime["launch"] == {
        "argv_tail": ("-m", "verification.h8_child"),
        "canonical_stdin_one_line": True,
        "capture_stdout": True,
        "capture_stderr": True,
        "timeout_seconds": 60.0,
    }
    assert runtime["conservative_hwm_formulas"] == {
        "primary": "max(0,post_lifetime_peak-pre_current_rss)",
        "supplementary": "max(0,post_lifetime_peak-pre_lifetime_peak)",
    }
    assert preimage["source_identities"]["profiler_api"] == {
        "torch_version": "2.9.1",
        "memory_profile_source_sha256": H8_PROFILER_MEMORY_SOURCE_SHA256,
        "profiler_source_sha256": H8_PROFILER_SOURCE_SHA256,
        "api_contract_sha256": H8_PROFILER_API_CONTRACT_SHA256,
        "invocation": {
            "activities": ("CPU",),
            "profile_memory": True,
            "record_shapes": True,
            "with_stack": True,
        },
    }

    digest = build_h8_protocol_sha256(config)
    assert digest == hashlib.sha256(
        json.dumps(
            preimage,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert digest == build_h8_protocol_sha256(H8ValidationConfig.create())
    assert digest == h8_child._runtime_protocol_sha256(config)
    assert (
        config.child_schema
        == H8_CHILD_SCHEMA_VERSION
        == h8_child._SCHEMA_VERSION
    )
    assert h8_child._REQUEST_KEYS == H8_CHILD_REQUEST_KEYS
    assert h8_child._ENVELOPE_KEYS == H8_CHILD_ENVELOPE_KEYS
    assert h8_child._RESULT_KEYS == H8_CHILD_RESULT_KEYS
    assert h8_child._IDENTITY_KEYS == H8_CHILD_IDENTITY_KEYS
    assert (
        h8_child._PROFILER_INVOCATION_ITEMS
        == h8_wire.H8_PROFILER_INVOCATION_ITEMS
    )
    with monkeypatch.context() as context:
        context.setattr(
            h8_protocol,
            "build_h8_protocol_sha256",
            lambda _config: (_ for _ in ()).throw(
                TypeError("synthetic protocol signature drift")
            ),
        )
        with pytest.raises(
            h8_child._ChildObservabilityError,
            match="protocol preimage is unavailable",
        ):
            h8_child._runtime_protocol_sha256(config)
    assert "child_runner" not in inspect.signature(
        h8_orchestrator.run_h8_parent_attempt
    ).parameters
    assert "child_runner" in inspect.signature(
        h8_orchestrator._run_h8_parent_attempt_for_test
    ).parameters

    request = {
        "mode": "production",
        "seed": 20260721,
        "repetition": 0,
        "config_sha256": config.config_sha256,
        "protocol_sha256": digest,
        "control_id": None,
    }
    request_line = h8_wire.canonical_json_line(request)
    request_sha256 = hashlib.sha256(request_line[:-1]).hexdigest()
    for name, drifted in (
        ("_SCHEMA_VERSION", "h8-child-local-drift"),
        ("_REQUEST_KEYS", ("mode",)),
        ("_ENVELOPE_KEYS", ("schema_version",)),
        ("_RESULT_KEYS", ("input_sha256",)),
        ("_IDENTITY_KEYS", ("hardware",)),
        ("_PRODUCTION_SAMPLE_SEEDS", {20260721: 1}),
        ("_MAX_SECONDS", 59.0),
        ("_PROFILER_INVOCATION_ITEMS", (("activities", ("CUDA",)),)),
    ):
        stdin_buffer = io.BytesIO(request_line)
        stdout_buffer = io.BytesIO()
        with monkeypatch.context() as context:
            context.setattr(h8_child, name, drifted)
            context.setattr(
                h8_child.sys,
                "stdin",
                SimpleNamespace(buffer=stdin_buffer),
            )
            context.setattr(
                h8_child.sys,
                "stdout",
                SimpleNamespace(buffer=stdout_buffer),
            )
            assert h8_child.main() == 0
        recovered = parse_h8_child_stdout(stdout_buffer.getvalue())
        assert recovered["schema_version"] == "h8-child-v2"
        assert recovered["request_sha256"] == request_sha256
        assert recovered["config_sha256"] == config.config_sha256
        assert recovered["protocol_sha256"] == digest
        assert recovered["status"] == "inconclusive"
        assert recovered["obligations"] == ["child_local_contract_drift"]
        error = recovered["error"]
        assert isinstance(error, dict)
        assert error["kind"] == "child_local_contract_drift"
        assert error["witnessed_violation"] is False
