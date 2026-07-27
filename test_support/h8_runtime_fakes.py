"""Typed, workload-free H8 runtime evidence builders for focused tests.

These builders are extracted from the Task 7C2 runtime-section tests so an
integration test can exercise the real parent decoder, gate, publisher, reopen,
and pointer paths without importing the large allocation test module.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import zlib
from collections.abc import Mapping

from vfe4.inference.h8_allocation import (
    H8AllocationPolicy,
    H8DispatchEvent,
    H8NegativeControlSpec,
    H8NumpyGuardEvent,
    H8StorageSpan,
    h8_negative_control_specs,
    make_h8_control_result,
)
from vfe4.numerics.block_layout import BlockChainLayout
from vfe4.types.h8 import (
    H8_CORRECTNESS_CASES,
    H8_CORRECTNESS_CONTROL_IDS,
    H8_CORRECTNESS_ORDERED_SOURCE_PAIRS,
    H8_CORRECTNESS_SOURCES,
    H8_PROFILER_API_CONTRACT_SHA256,
    H8_PROFILER_MEMORY_SOURCE_SHA256,
    H8_PROFILER_SOURCE_SHA256,
    H8CorrectnessCell,
    H8CorrectnessControlResult,
    H8CorrectnessEndpointRecord,
    H8CorrectnessSourceResult,
    H8InvariantRecord,
    H8ProfilerEventRecord,
    H8TensorKey,
    h8_correctness_endpoint_ids,
)
from vfe4.types.results import GateStatus
from verification.h8_budget import (
    H8_OPERATION_SCOPES,
    H8_SCALE_RESIDUAL_SPECS,
    H8ChildInvocation,
    H8ChildProcessRecord,
    compare_operands,
    make_h8_identity_record,
    make_operand_record,
)
from verification.h8_wire import H8_CHILD_IDENTITY_ENV


def fixture_jsonable(value: object) -> object:
    """Return the JSON-shaped value used by the real child decoder."""

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: fixture_jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, GateStatus):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): fixture_jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return [fixture_jsonable(item) for item in value]
    if isinstance(value, list):
        return [fixture_jsonable(item) for item in value]
    return value


def make_test_parent_identities() -> dict[str, object]:
    """Return one exact lightweight identity inventory for a fake parent."""

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
                        "MKL_THREADING_LAYER": "SEQUENTIAL",
                    },
                    "forbidden_environment_present": False,
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


def _production_policy() -> H8AllocationPolicy:
    return H8AllocationPolicy(
        BlockChainLayout(horizon=128, d_z=20, d_m=20)
    )


def _objective_fixture() -> dict[str, object]:
    def term(role: str, receiver_t: int | None = None) -> dict[str, object]:
        factor_id = (
            "initial_joint"
            if receiver_t is None
            else f"{role}:{receiver_t:04d}"
        )
        return {
            "factor_id": factor_id,
            "role": role,
            "receiver_t": receiver_t,
            "value": 1.0 if receiver_t is None else 0.0,
            "absolute_sum_bound": 1.0 if receiver_t is None else 0.0,
        }

    return {
        "horizon": 128,
        "initial_joint": term("initial_joint"),
        "model_transitions": [
            term("model_transition", receiver_t)
            for receiver_t in range(1, 129)
        ],
        "state_transitions": [
            term("state_transition", receiver_t)
            for receiver_t in range(1, 129)
        ],
        "emissions_order21": [
            term("emission_order21", receiver_t)
            for receiver_t in range(1, 129)
        ],
        "emissions_order17": [
            term("emission_order17", receiver_t)
            for receiver_t in range(1, 129)
        ],
        "recognition_entropy": 0.0,
        "log_normalizer": 0.0,
        "model_source_kl": 0.0,
        "state_source_kl": 0.0,
        "source_entropy": 0.0,
        "quadrature_absolute_difference": 0.0,
        "complete_order21": 1.0,
        "absolute_term_sum": 1.0,
    }


def _residual_allowance_fixture() -> dict[str, object]:
    endpoints: dict[tuple[int, ...], dict[str, object]] = {}

    def zero_endpoint(shape: tuple[int, ...]) -> dict[str, object]:
        endpoint = endpoints.get(shape)
        if endpoint is not None:
            return dict(endpoint)
        scalar_count = 1
        for dimension in shape:
            scalar_count *= dimension
        raw = b"\0" * (scalar_count * 8)
        compressed = zlib.compress(raw, level=9)
        endpoint = {
            "encoding": "float64-le-zlib-base64-v1",
            "shape": list(shape),
            "scalar_count": scalar_count,
            "raw_nbytes": len(raw),
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "compressed_nbytes": len(compressed),
            "payload_b64": base64.b64encode(compressed).decode("ascii"),
        }
        endpoints[shape] = endpoint
        return dict(endpoint)

    groups: dict[str, object] = {}
    for residual_id, specs in H8_SCALE_RESIDUAL_SPECS.items():
        comparisons = []
        for (
            comparison_id,
            left_id,
            right_id,
            shape,
            left_solver,
            right_solver,
        ) in specs:
            left = make_operand_record(
                operand_id=left_id,
                shape=shape,
                infinity_norm=0.0,
                absolute_sum_bound=0.0,
                local_operation_count=shape[-1],
                source="block",
                solver_produced=left_solver,
            )
            right = make_operand_record(
                operand_id=right_id,
                shape=shape,
                infinity_norm=0.0,
                absolute_sum_bound=0.0,
                local_operation_count=shape[-1],
                source="block",
                solver_produced=right_solver,
            )
            comparisons.append(
                {
                    "allowance": fixture_jsonable(
                        compare_operands(
                            comparison_id=comparison_id,
                            left=left,
                            right=right,
                            residual=0.0,
                        )
                    ),
                    "left_endpoint": zero_endpoint(shape),
                    "right_endpoint": zero_endpoint(shape),
                }
            )
        groups[residual_id] = {
            "residual_id": residual_id,
            "aggregation": "max_residual_all_comparisons_must_pass",
            "residual": 0.0,
            "comparisons": comparisons,
            "decisive": True,
            "passed": True,
        }
    return groups


def _problem_evidence_fixture() -> dict[str, object]:
    horizon = 128
    pivots = [1.0] * horizon
    norms = [0.1] * horizon
    return {
        "generative_sha256": "9" * 64,
        "recognition_sha256": "a" * 64,
        "local_spd_diagnostics": {
            "schema_version": "h8-local-spd-diagnostics-v1",
            "horizon": horizon,
            "generative_initial_min_pivot": 1.0,
            "model_transition_min_pivots": pivots,
            "state_transition_min_pivots": pivots,
            "recognition_initial_min_pivot": 1.0,
            "recognition_transition_min_pivots": pivots,
            "global_min_pivot": 1.0,
        },
        "transition_norms": {
            "schema_version": "h8-transition-norms-v1",
            "horizon": horizon,
            "norm": "operator_2",
            "model_transition_norms": norms,
            "state_transition_norms": norms,
            "state_model_coupling_norms": norms,
            "recognition_transition_norms": norms,
            "max_model_transition_norm": 0.1,
            "max_state_transition_norm": 0.1,
            "max_state_model_coupling_norm": 0.1,
            "max_recognition_transition_norm": 0.1,
        },
        "observation_sha256": "b" * 64,
    }


def make_child_envelope(**updates: object) -> dict[str, object]:
    """Build one fresh complete production child envelope."""

    digest = "a" * 64
    layout = {"horizon": 128, "d_z": 20, "d_m": 20}
    stored_blocks = [
        *(
            {"kind": "diagonal", "row": index, "column": index}
            for index in range(129)
        ),
        *(
            {
                "kind": "lower_adjacent",
                "row": index,
                "column": index - 1,
            }
            for index in range(1, 129)
        ),
    ]
    required_operations = (
        "factorization",
        "forward_substitution",
        "backward_substitution",
        "mean_solve",
        "logdet",
        "selected_inverse",
        "sample_width_one",
        "quadratic",
        "sparse_trace",
        "condition_estimate",
        "entropy",
        "log_normalizer",
        "complete_objective",
    )
    allowance_groups = _residual_allowance_fixture()
    decision_flags = {
        "time_pass": True,
        "process_memory_pass": True,
        "torch_memory_pass": True,
        "rhs_width_pass": True,
        "sample_width_pass": True,
        "storage_pass": True,
        "offband_fill_pass": True,
        "forbidden_attempts_zero": True,
        "pivot_margin_pass": True,
        "finite_pass": True,
        "residual_allowances_pass": True,
        "dispatch_observed": True,
        "dispatch_backend_cross_check_pass": True,
    }
    required_numpy_operators = (
        "numpy.random.Generator.standard_normal",
        "numpy.asarray",
        "numpy.ascontiguousarray",
        "numpy.multiply",
        "numpy.add",
        "numpy.all",
        "numpy.divide",
        "numpy.eye",
        "numpy.isfinite",
        "numpy.matmul",
        "numpy.sqrt",
        "numpy.transpose",
        "numpy.linalg.cholesky",
        "numpy.linalg.norm",
    )
    numpy_inventory = [
        {
            "site": "local",
            "shape": [1],
            "dtype": "float64",
            "nbytes": 8,
            "sha256": "8" * 64,
        }
    ]
    numpy_inventory_sha256 = hashlib.sha256(
        json.dumps(
            numpy_inventory,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    dispatch_storage_key = "cpu:1:8:8"
    dispatch_event = H8DispatchEvent(
        sequence=0,
        operator="aten::empty",
        semantic_site="local",
        control_id=None,
        input_shapes=(),
        output_shapes=((1,),),
        physical_output_shapes=((1,),),
        stack_member_shapes=(),
        stack_member_count=0,
        dtype="torch.float64",
        device="cpu",
        float64_equivalent_scalars=1,
        classifications=("local",),
        storage_spans=(
            H8StorageSpan(
                storage_key=dispatch_storage_key,
                device="cpu",
                pointer=8,
                span_start=8,
                span_end=16,
                nbytes=8,
            ),
        ),
        alias_storage_keys=(),
        new_storage_keys=(dispatch_storage_key,),
        allocated_float64_equivalent_scalars=1,
        live_float64_equivalent_scalars_by_site=(("local", 1),),
        stack=("test.py:1",),
        executed=True,
        forbidden_reason=None,
        live_storage_bytes_after=8,
        population_live_storage_bytes_after=0,
    )
    dispatch_trace_sha256 = hashlib.sha256(
        json.dumps(
            [dispatch_event],
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    dispatch_scope_witnesses = [
        {
            "semantic_scope": scope,
            "event_start": 0,
            "event_end": 1,
            "event_count": 1,
            "callback_completed": True,
        }
        for scope in sorted(
            {
                *H8_OPERATION_SCOPES.values(),
                "production.problem_build",
                "production.assembly",
            }
        )
    ]
    result = {
        "input_sha256": "1" * 64,
        "sample_noise_sha256": "2" * 64,
        "problem_evidence": _problem_evidence_fixture(),
        "objective": _objective_fixture(),
        "storage": {
            "layout": layout,
            "precision_scalar_count": 411_200,
            "factor_scalar_count": 411_200,
            "selected_inverse_scalar_count": 411_200,
            "information_scalar_count": 5_160,
            "upper_block_scalar_count": 0,
        },
        "fill": {
            "layout": layout,
            "stored_block_ids": stored_blocks,
            "observed_offband_blocks": 0,
            "duplicated_upper_blocks": 0,
        },
        "workspace": {
            "maximum_shape": [40, 40],
            "maximum_scalar_count": 1_600,
            "maximum_rhs_width": 40,
            "attempted_forbidden_rhs_widths": [],
        },
        "counters": {
            "layout": layout,
            "factorization_calls": 1,
            "forward_substitution_calls": 4,
            "backward_substitution_calls": 5,
            "solve_calls": 3,
            "logdet_calls": 1,
            "selected_inverse_calls": 2,
            "sample_calls": 1,
            "quadratic_calls": 1,
            "trace_calls": 1,
            "sparse_matvec_calls": 1,
            "maximum_rhs_width": 40,
            "maximum_sample_rhs_width": 1,
            "selected_block_ids": stored_blocks,
            "selected_block_count": 257,
            "attempted_forbidden_selected_blocks": 0,
            "attempted_forbidden_rhs_widths": [],
        },
        "allocation": {
            "dispatch_trace_sha256": dispatch_trace_sha256,
            "dispatch_event_count": 1,
            "dispatch_events": [fixture_jsonable(dispatch_event)],
            "dispatch_scope_witnesses": dispatch_scope_witnesses,
            "dispatch_cross_check": {
                "complete": True,
                "obligations": [],
                "backend_forbidden_attempt_count": 0,
                "dispatch_forbidden_attempt_count": 0,
                "reconciled_operation_counts": [["factorize", 1, 1]],
            },
            "dispatch_forbidden_attempt_count": 0,
            "dispatch_live_peak_bytes": 8,
            "torch_population_peak_bytes": 0,
            "profiler_trace_sha256": None,
            "profiler_events": [],
            "profiler_lossy_rows": [],
            "preexisting_storage_count": None,
            "preexisting_bytes": None,
            "baseline_live_bytes": None,
            "profiler_reconstructed_live_peak_bytes": None,
            "profiler_all_joined_and_liveness_reconciled": None,
            "numpy_guard_event_count": len(required_numpy_operators),
            "numpy_guard_events": [
                {
                    "sequence": sequence,
                    "operator": operator,
                    "semantic_site": "local",
                    "control_id": None,
                    "input_shapes": [],
                    "output_shapes": [[1]],
                    "dtype": "float64",
                    "float64_equivalent_scalars": 1,
                    "executed": True,
                    "forbidden_reason": None,
                }
                for sequence, operator in enumerate(required_numpy_operators)
            ],
            "numpy_inventory": numpy_inventory,
            "numpy_inventory_sha256": numpy_inventory_sha256,
            "backend_forbidden_attempt_count": 0,
            "observed_channels": [
                "dispatch",
                "numpy_guard",
                "backend",
                "os_hwm",
            ],
        },
        "resources": {
            "adapter": "test",
            "adapter_sha256": "3" * 64,
            "pre_current_rss_bytes": 100,
            "pre_lifetime_peak_bytes": 120,
            "pre_private_bytes": 90,
            "post_current_rss_bytes": 110,
            "post_lifetime_peak_bytes": 130,
            "post_private_bytes": 95,
            "conservative_incremental_hwm_bytes": 30,
            "peak_to_peak_diagnostic_bytes": 10,
            "parent_elapsed_ns": 0,
            "child_elapsed_ns": 1,
        },
        "diagnostics": {
            "estimator": "HagerHigham1NormEstimate-v1",
            "kappa_1_estimate": 1.0,
            "iterations": 1,
            "convergence_reason": "test",
            "index_sha256": "4" * 64,
            "sign_sha256": "5" * 64,
            "per_block_min_pivots": [1.0] * 129,
            "global_min_pivot": 1.0,
            "per_block_pivot_margins": [1.0 - 1e-8] * 129,
            "global_pivot_margin": 1.0 - 1e-8,
        },
        "operation_reachability": {
            name: True for name in required_operations
        },
        "residuals": {
            name: 0.0 for name in H8_SCALE_RESIDUAL_SPECS
        },
        "resource_decisions": {
            **decision_flags,
            "residual_allowances": allowance_groups,
            "dispatch_backend_cross_check_obligations": [],
            "conservative_incremental_hwm_bytes": 30,
            "torch_population_peak_bytes": 0,
        },
        "invariants": [
            {
                "invariant_id": name,
                "status": "pass",
                "value": 1,
                "limit": 1,
                "detail": f"{name}=True",
                "obligations": [],
            }
            for name in decision_flags
        ],
    }
    record: dict[str, object] = {
        "schema_version": "h8-child-v2",
        "mode": "production",
        "seed": 20260721,
        "repetition": 0,
        "control_id": None,
        "request_sha256": digest,
        "config_sha256": "b" * 64,
        "protocol_sha256": "c" * 64,
        "status": "pass",
        "obligations": [],
        "identities": make_test_parent_identities(),
        "result": result,
        "control": None,
        "error": None,
    }
    record.update(updates)
    return record


def make_profiler_envelope(**updates: object) -> dict[str, object]:
    """Build one fresh complete profiler child envelope."""

    envelope = make_child_envelope(mode="profiler", repetition=None)
    result = envelope["result"]
    assert isinstance(result, dict)
    allocation = result["allocation"]
    decisions = result["resource_decisions"]
    invariants = result["invariants"]
    assert isinstance(allocation, dict)
    assert isinstance(decisions, dict)
    assert isinstance(invariants, list)

    baseline_key = H8TensorKey(
        tensor_id=1,
        storage_ptr=8,
        allocation_id=1,
        device="cpu",
    )
    transient_key = H8TensorKey(
        tensor_id=2,
        storage_ptr=16,
        allocation_id=2,
        device="cpu",
    )
    events = (
        H8ProfilerEventRecord(
            source_row_index=0,
            timestamp_ns=-1,
            action="PREEXISTING",
            tensor_key=baseline_key,
            version=0,
            nbytes=8,
            dtype="torch.float64",
            device="cpu",
            operator="aten::empty",
            stack=("test.py:1",),
            logical_shape=(1,),
            classification="local",
            matched_event_node_indices=(0,),
            join_witness_sha256="c" * 64,
            live_bytes_after=8,
        ),
        H8ProfilerEventRecord(
            source_row_index=1,
            timestamp_ns=0,
            action="CREATE",
            tensor_key=transient_key,
            version=0,
            nbytes=8,
            dtype="torch.float64",
            device="cpu",
            operator="aten::empty",
            stack=("test.py:2",),
            logical_shape=(1,),
            classification="local",
            matched_event_node_indices=(1,),
            join_witness_sha256="d" * 64,
            live_bytes_after=16,
        ),
        H8ProfilerEventRecord(
            source_row_index=2,
            timestamp_ns=1,
            action="INCREMENT_VERSION",
            tensor_key=transient_key,
            version=1,
            nbytes=0,
            dtype="torch.float64",
            device="cpu",
            operator="aten::add_",
            stack=("test.py:3",),
            logical_shape=(1,),
            classification="local",
            matched_event_node_indices=(2,),
            join_witness_sha256="e" * 64,
            live_bytes_after=16,
        ),
        H8ProfilerEventRecord(
            source_row_index=3,
            timestamp_ns=2,
            action="DESTROY",
            tensor_key=transient_key,
            version=1,
            nbytes=-8,
            dtype="torch.float64",
            device="cpu",
            operator="aten::empty",
            stack=("test.py:4",),
            logical_shape=(1,),
            classification="local",
            matched_event_node_indices=(3,),
            join_witness_sha256="f" * 64,
            live_bytes_after=8,
        ),
    )
    allocation.update(
        {
            "profiler_trace_sha256": hashlib.sha256(
                json.dumps(
                    events,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
            ).hexdigest(),
            "profiler_events": [
                fixture_jsonable(event) for event in events
            ],
            "preexisting_storage_count": 1,
            "preexisting_bytes": 8,
            "baseline_live_bytes": 8,
            "profiler_reconstructed_live_peak_bytes": 16,
            "profiler_all_joined_and_liveness_reconciled": True,
            "observed_channels": [
                "dispatch",
                "profiler",
                "numpy_guard",
                "backend",
                "os_hwm",
            ],
            "profiler_api": {
                "torch_version": "2.9.1",
                "memory_profile_source_sha256": (
                    H8_PROFILER_MEMORY_SOURCE_SHA256
                ),
                "profiler_source_sha256": H8_PROFILER_SOURCE_SHA256,
                "api_contract_sha256": H8_PROFILER_API_CONTRACT_SHA256,
            },
        }
    )
    dispatch_pass = decisions.pop("dispatch_backend_cross_check_pass")
    dispatch_obligations = decisions.pop(
        "dispatch_backend_cross_check_obligations"
    )
    decisions.update(
        {
            "profiler_join_pass": True,
            "profiler_reconstructed_live_peak_bytes": 16,
            "dispatch_backend_cross_check_pass": dispatch_pass,
            "dispatch_backend_cross_check_obligations": dispatch_obligations,
        }
    )
    dispatch_invariant = invariants.pop()
    assert (
        dispatch_invariant["invariant_id"]
        == "dispatch_backend_cross_check_pass"
    )
    invariants.extend(
        (
            {
                "invariant_id": "profiler_join_pass",
                "status": "pass",
                "value": 1,
                "limit": 1,
                "detail": "profiler_join_pass=True",
                "obligations": [],
            },
            dispatch_invariant,
        )
    )
    envelope.update(updates)
    return envelope


def make_control_envelope(
    spec: H8NegativeControlSpec | None = None,
) -> dict[str, object]:
    """Build one fresh complete negative-control child envelope."""

    if spec is None:
        spec = next(
            item
            for item in h8_negative_control_specs(
                _production_policy().layout
            )
            if item.control_id == "torch_eye_full_rhs"
        )
    if spec.control_id.startswith("numpy_"):
        event: H8DispatchEvent | H8NumpyGuardEvent = H8NumpyGuardEvent(
            sequence=0,
            operator=spec.requested_operation,
            semantic_site=None,
            control_id=spec.control_id,
            input_shapes=spec.logical_shapes[:-1],
            output_shapes=(spec.logical_shapes[-1],),
            dtype="float64",
            float64_equivalent_scalars=0,
            executed=False,
            forbidden_reason=spec.expected_reason,
        )
        observed_channels = ("numpy_guard",)
        evidence = {
            "events": (event,),
            "operation_returned": False,
            "caught_forbidden": True,
            "pre_execution_detected": True,
            "executed_past_detector": False,
        }
    else:
        event = H8DispatchEvent(
            sequence=0,
            operator=spec.requested_operation,
            semantic_site=None,
            control_id=spec.control_id,
            input_shapes=spec.logical_shapes[:-1],
            output_shapes=(spec.logical_shapes[-1],),
            physical_output_shapes=(),
            stack_member_shapes=(),
            stack_member_count=0,
            dtype="torch.float64",
            device="cpu",
            float64_equivalent_scalars=0,
            classifications=(),
            storage_spans=(),
            alias_storage_keys=(),
            new_storage_keys=(),
            allocated_float64_equivalent_scalars=0,
            live_float64_equivalent_scalars_by_site=(),
            stack=("test.py:1",),
            executed=False,
            forbidden_reason=spec.expected_reason,
            live_storage_bytes_after=0,
            population_live_storage_bytes_after=0,
        )
        backend = (
            {
                "before": (),
                "after": (_production_policy().layout.dimension,),
                "detected": True,
                "executed_past_detector": False,
                "unexpected_exception": None,
            }
            if spec.control_id == "torch_eye_full_rhs"
            else None
        )
        observed_channels = (
            ("dispatch", "backend")
            if backend is not None
            else ("dispatch",)
        )
        evidence = {
            "dispatch": (event,),
            "backend": backend,
            "operation_returned": False,
            "caught_forbidden": True,
            "pre_execution_detected": True,
            "executed_past_detector": False,
        }
    result = make_h8_control_result(
        spec,
        observed_channels=observed_channels,
        detected=True,
        event_payload=evidence,
    )
    return make_child_envelope(
        mode="negative_control",
        repetition=None,
        control_id=spec.control_id,
        result=None,
        control={
            "summary": fixture_jsonable(result),
            "evidence": fixture_jsonable(evidence),
        },
    )


def make_fake_h8_process_record(
    invocation: H8ChildInvocation,
) -> H8ChildProcessRecord:
    """Bind one fresh typed fake envelope to the exact issued request."""

    if type(invocation) is not H8ChildInvocation:
        raise ValueError("invocation must be an exact H8ChildInvocation")
    invocation.__post_init__()
    request = json.loads(invocation.stdin[:-1])
    if type(request) is not dict:
        raise ValueError("fake invocation request must be one object")
    mode = request.get("mode")
    if mode == "production":
        envelope = make_child_envelope()
    elif mode == "profiler":
        envelope = make_profiler_envelope()
    elif mode == "negative_control":
        control_id = request.get("control_id")
        spec = next(
            (
                item
                for item in h8_negative_control_specs(
                    _production_policy().layout
                )
                if item.control_id == control_id
            ),
            None,
        )
        if spec is None:
            raise ValueError("fake invocation has an unknown control")
        envelope = make_control_envelope(spec)
    else:
        raise ValueError("fake invocation has an unknown mode")
    identity_json = invocation.environment.get(H8_CHILD_IDENTITY_ENV)
    if type(identity_json) is not str:
        raise ValueError("fake invocation lacks parent identities")
    identities = json.loads(identity_json)
    if type(identities) is not dict:
        raise ValueError("fake parent identities must be one object")
    envelope.update(
        {
            "mode": request["mode"],
            "seed": request["seed"],
            "repetition": request["repetition"],
            "control_id": request["control_id"],
            "config_sha256": request["config_sha256"],
            "protocol_sha256": request["protocol_sha256"],
            "request_sha256": hashlib.sha256(
                invocation.stdin[:-1]
            ).hexdigest(),
            "identities": identities,
        }
    )
    return H8ChildProcessRecord.from_payload(envelope)


def make_pass_correctness_cells() -> tuple[H8CorrectnessCell, ...]:
    """Return the exact 12-cell typed PASS grid without numerical work."""

    cells: list[H8CorrectnessCell] = []
    for cell_id, (horizon, k, problem_seed, noise_seed) in enumerate(
        H8_CORRECTNESS_CASES,
        start=1,
    ):
        layout = BlockChainLayout(horizon=horizon, d_z=k, d_m=k)
        endpoint_ids = h8_correctness_endpoint_ids(horizon)
        operands: dict[tuple[str, str], object] = {}
        source_results: list[H8CorrectnessSourceResult] = []
        for source in H8_CORRECTNESS_SOURCES:
            endpoints: list[H8CorrectnessEndpointRecord] = []
            for endpoint_id in endpoint_ids:
                operand = make_operand_record(
                    operand_id=f"{source}:{endpoint_id}",
                    shape=(1,),
                    infinity_norm=0.0,
                    absolute_sum_bound=0.0,
                    local_operation_count=1,
                    source=source,
                    solver_produced=False,
                )
                operands[(source, endpoint_id)] = operand
                endpoints.append(
                    H8CorrectnessEndpointRecord(
                        endpoint_id=endpoint_id,
                        raw_values=(0.0,),
                        operand=operand,
                    )
                )
            source_results.append(
                H8CorrectnessSourceResult(
                    source=source,
                    endpoints=tuple(endpoints),
                )
            )
        comparisons = tuple(
            compare_operands(
                comparison_id=f"{endpoint_id}:{left}->{right}",
                left=operands[(left, endpoint_id)],  # type: ignore[arg-type]
                right=operands[(right, endpoint_id)],  # type: ignore[arg-type]
                residual=0.0,
            )
            for endpoint_id in endpoint_ids
            for left, right in H8_CORRECTNESS_ORDERED_SOURCE_PAIRS
        )
        cells.append(
            H8CorrectnessCell(
                cell_id=cell_id,
                layout=layout,
                problem_seed=problem_seed,
                sample_noise_seed=noise_seed,
                problem_sha256=f"{cell_id:x}" * 64,
                sample_noise_sha256=f"{cell_id + 1:x}" * 64,
                source_results=tuple(source_results),
                pair_comparisons=comparisons,
                wrong_path_controls=tuple(
                    H8CorrectnessControlResult(
                        control_id=control_id,
                        residual=1.0,
                        allowance=0.0,
                        decisive=True,
                        status=GateStatus.PASS,
                        obligations=(),
                    )
                    for control_id in H8_CORRECTNESS_CONTROL_IDS
                ),
                invariants=(
                    H8InvariantRecord(
                        invariant_id="fake_cell_complete",
                        status=GateStatus.PASS,
                        value=1,
                        limit=1,
                        detail="test-only typed evidence",
                        obligations=(),
                    ),
                ),
                status=GateStatus.PASS,
                obligations=(),
            )
        )
    return tuple(cells)


__all__ = [
    "fixture_jsonable",
    "make_child_envelope",
    "make_control_envelope",
    "make_fake_h8_process_record",
    "make_pass_correctness_cells",
    "make_profiler_envelope",
    "make_test_parent_identities",
]
