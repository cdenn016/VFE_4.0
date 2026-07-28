from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import inspect
import io
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest
import torch
from test_support.h8_runtime_fakes import (
    fixture_jsonable as _fixture_jsonable,
    make_child_envelope as _child_envelope,
    make_control_envelope as _control_child_envelope,
    make_pass_correctness_cells,
    make_profiler_envelope as _profiler_child_envelope,
)

from vfe4.inference.h8_allocation import (
    H8AllocationPolicy,
    H8DispatchEvent,
    H8DispatchTrace,
    H8ForbiddenAllocation,
    H8NumpyAllocationGuard,
    H8NumpyGuardEvent,
    H8ProfilerEnrichment,
    H8ProfilerObservabilityGap,
    H8RawProfilerEvent,
    classify_h8_operator,
    cross_check_h8_backend_dispatch,
    h8_control_detected_pre_execution,
    h8_negative_control_specs,
    h8_tensor_storage_span,
    make_h8_control_result,
    parse_h8_lossy_profiler_rows,
    parse_h8_profiler_events,
)
from vfe4.generative.reference_h8 import make_h8_problem
from vfe4.numerics.block_layout import BlockChainLayout
from vfe4.types.h8 import (
    H8AllocationRecord,
    BackendCounterSnapshot,
    BlockStorageRecord,
    BlockWorkspaceRecord,
    H8ChildAttemptRecord,
    H8ChildRequest,
    H8ChildResult,
    H8ControlResult,
    H8TensorKey,
)
from vfe4.types.results import GateStatus
from verification.h8_child import (
    _ProfilerAllocationFact,
    _ProfilerSourceIndexes,
    _ProfilerTensorFact,
    _ProfilerVersionFact,
    _join_profiler_source_facts,
    _run_negative_control,
    _windows_memory_snapshot,
)
from verification.h8_budget import (
    H8_CHILD_IDENTITY_ENV,
    H8ChildDecision,
    H8ChildInvocation,
    H8ChildProcessRecord,
    build_h8_child_invocation,
    classify_h8_child_outcome,
    conservative_hwm_endpoints,
    decode_h8_child_result,
    decode_h8_control_result,
    make_h8_child_attempt_record,
    make_h8_identity_record,
    parse_h8_child_stdout,
    windows_process_memory_layout,
)
from verification.h8_wire import (
    H8_FORBIDDEN_ENVIRONMENT,
    H8_THREAD_ENVIRONMENT_ITEMS,
    canonical_json_bytes,
    prepare_h8_script_environment,
    require_h8_startup_environment,
)


def _production_policy() -> H8AllocationPolicy:
    return H8AllocationPolicy(BlockChainLayout(horizon=128, d_z=20, d_m=20))


def test_semantic_site_whitelist_uses_logical_production_dimensions() -> None:
    policy = _production_policy()
    policy.register_generator_objective_site(
        "generator.transition",
        (128, 20, 20),
    )

    assert (
        policy.classify_allocation(
            site="factor.diagonal",
            logical_shape=(129, 40, 40),
            operator="aten.empty",
        ).classification
        == "block_diagonal"
    )
    assert (
        policy.classify_allocation(
            site="rhs",
            logical_shape=(129, 40, 17),
            operator="aten.empty",
        ).classification
        == "block_rhs"
    )
    assert (
        policy.classify_allocation(
            site="generator.transition",
            logical_shape=(128, 20, 20),
            operator="aten.empty",
        ).classification
        == "generator_objective"
    )

    rejected = {
        "dense_matrix": (5160, 5160),
        "flat_dense": (5160 * 5160,),
        "near_dense": (5159, 5159),
        "global_axis": (5160,),
        "pair_slab": (129, 129, 40, 40),
        "triangular_pairs": (129 * 130 // 2, 40, 40),
        "combined_pairs": (129 * 129, 40, 40),
    }
    for name, shape in rejected.items():
        with pytest.raises(H8ForbiddenAllocation, match=name.split("_")[0]):
            policy.classify_allocation(
                site="rhs",
                logical_shape=shape,
                operator="aten.empty",
            )

    with pytest.raises(H8ForbiddenAllocation, match="unregistered"):
        policy.classify_allocation(
            site="factor.diagonal",
            logical_shape=(129, 40, 39),
            operator="aten.reshape",
        )


def test_dispatch_preflight_classifies_dense_operations_before_execution() -> None:
    policy = _production_policy()
    local = classify_h8_operator(
        policy,
        operator="aten.linalg_eigvalsh",
        operand_shapes=((40, 40),),
        output_shape=(40,),
    )
    assert local == "local_eigensolver"

    forbidden = (
        ("aten.linalg_eigvalsh", ((5160, 5160),), (5160,)),
        ("aten.linalg_cholesky", ((5160, 5160),), (5160, 5160)),
        ("aten.eye", (), (5160, 5160)),
        ("aten.linalg_solve", ((40, 40), (129, 40, 5160)), None),
        ("aten.one_hot", ((5160,),), (5160, 5160)),
    )
    for operator, operands, output in forbidden:
        with pytest.raises(H8ForbiddenAllocation):
            classify_h8_operator(
                policy,
                operator=operator,
                operand_shapes=operands,
                output_shape=output,
            )


def test_dispatch_counts_one_live_storage_for_a_registered_view_alias() -> None:
    policy = _production_policy()
    trace = H8DispatchTrace(policy)
    base = torch.empty((2, 2), dtype=torch.float64)
    span = h8_tensor_storage_span(base)
    trace.register_preexisting(
        base,
        site="factor.diagonal",
        logical_shape=(129, 40, 40),
        storage_span=span,
        nbytes=span.nbytes,
    )

    with trace:
        with trace.semantic_site(
            "factor.diagonal",
            logical_output_shapes=((129, 40, 40),),
        ):
            alias = base.view(2, 2)

    assert trace.live_storage_bytes == base.untyped_storage().nbytes()
    assert trace.live_peak_bytes == base.untyped_storage().nbytes()
    assert trace.population_live_storage_bytes == base.untyped_storage().nbytes()
    assert trace.population_live_peak_bytes == base.untyped_storage().nbytes()
    assert trace.baseline_live_bytes == base.untyped_storage().nbytes()
    assert trace.events[0].operator == "PREEXISTING"
    assert trace.events[0].new_storage_keys == (span.storage_key,)
    assert trace.events[-1].alias_storage_keys
    assert trace.events[-1].forbidden_reason is None
    del alias


def test_dispatch_composite_scope_is_strict_non_nested_and_restored() -> None:
    policy = _production_policy()
    trace = H8DispatchTrace(policy)
    assert policy.registered_composite_sites == (
        "production.problem_build",
        "production.assembly",
        "production.factorization",
        "production.mean_solve",
        "production.forward_substitution",
        "production.backward_substitution",
        "production.logdet",
        "production.selected_inverse",
        "production.sample_width_one",
        "production.quadratic",
        "production.sparse_trace",
        "production.condition_estimate",
        "production.entropy",
        "production.log_normalizer",
        "production.complete_objective",
    )

    with pytest.raises(ValueError, match="registered semantic scope"):
        with trace.semantic_site("production.unknown"):
            pass

    with pytest.raises(ValueError, match="derive output shapes"):
        with trace.semantic_site(
            "production.assembly",
            logical_output_shapes=((policy.layout.block_size,),),
        ):
            pass

    with trace.semantic_site("production.assembly"):
        with pytest.raises(RuntimeError, match="cannot be nested"):
            with trace.semantic_site("local"):
                pass

    with pytest.raises(RuntimeError, match="scope body"):
        with trace.semantic_site("production.assembly"):
            raise RuntimeError("scope body")
    with trace.semantic_site("production.factorization"):
        pass

    with pytest.raises(H8ForbiddenAllocation, match="semantic allocation site"):
        with trace:
            torch.empty((policy.layout.block_size, policy.layout.block_size))


def test_dispatch_composite_scope_maps_mixed_shapes_to_exact_leaf_sites() -> None:
    policy = _production_policy()
    trace = H8DispatchTrace(policy)
    n = policy.layout.population_size
    b = policy.layout.block_size
    with trace:
        with trace.semantic_site("production.assembly"):
            torch.empty((b, b), dtype=torch.float64)
            torch.empty((n, b), dtype=torch.float64)
            torch.empty((n, b, b), dtype=torch.float64)
            torch.empty((n - 1, b, b), dtype=torch.float64)

    assert tuple(event.semantic_site for event in trace.events) == (
        "local",
        "information",
        "precision.diagonal",
        "precision.lower",
    )
    assert all(event.forbidden_reason is None for event in trace.events)


def test_dispatch_composite_scope_rejects_unregistered_shape_and_operator() -> None:
    policy = _production_policy()
    trace = H8DispatchTrace(policy)
    b = policy.layout.block_size
    with trace:
        with trace.semantic_site("production.assembly"):
            local = torch.empty((b, b), dtype=torch.float64)
            with pytest.raises(H8ForbiddenAllocation, match="output shape"):
                torch.empty((b, b, b), dtype=torch.float64)
            with pytest.raises(H8ForbiddenAllocation, match="operator"):
                torch.sin(local)

    assert trace.events[-2].executed is False
    assert trace.events[-1].executed is False


def test_dispatch_rejects_unregistered_inputs_and_has_no_caller_counter() -> None:
    trace = H8DispatchTrace(_production_policy())
    unregistered = torch.empty((2, 2), dtype=torch.float64)
    with pytest.raises(H8ForbiddenAllocation, match="unregistered non-control"):
        with trace:
            with trace.semantic_site(
                "local",
                logical_output_shapes=((2, 2),),
            ):
                unregistered.clone()

    assert trace.events[-1].executed is False
    assert trace.events[-1].input_shapes == ((2, 2),)
    assert not hasattr(trace, "record_backend_operation")
    with pytest.raises(AttributeError):
        trace.events.append(trace.events[-1])  # type: ignore[attr-defined]


def test_cross_check_requires_exact_nonzero_dispatch_coverage() -> None:
    policy = _production_policy()
    layout = policy.layout
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
        maximum_rhs_width=1,
        maximum_sample_rhs_width=1,
        selected_block_ids=layout.stored_block_ids,
        selected_block_count=len(layout.stored_block_ids),
        attempted_forbidden_selected_blocks=0,
    )
    storage = BlockStorageRecord(
        layout=layout,
        precision_scalar_count=layout.band_storage_scalar_count,
        factor_scalar_count=layout.band_storage_scalar_count,
        selected_inverse_scalar_count=layout.band_storage_scalar_count,
        information_scalar_count=layout.information_scalar_count,
        upper_block_scalar_count=0,
    )
    workspace = BlockWorkspaceRecord(
        maximum_shape=(layout.block_size, 1),
        maximum_scalar_count=layout.block_size,
        maximum_rhs_width=1,
    )
    result = cross_check_h8_backend_dispatch(
        layout=layout,
        counters=counters,
        storage=storage,
        workspace=workspace,
        dispatch=H8DispatchTrace(policy),
    )

    assert not result.complete
    assert "dispatch_event_stream_empty" in result.obligations
    assert all(
        backend > 0 and observed == 0
        for _operation, backend, observed in result.reconciled_operation_counts
    )


def test_numpy_guard_rejects_before_call_and_restores_every_callable() -> None:
    policy = _production_policy()
    original_empty = np.empty
    original_eigvalsh = np.linalg.eigvalsh
    guard = H8NumpyAllocationGuard(policy)

    with pytest.raises(H8ForbiddenAllocation, match="dense"):
        with guard:
            np.empty((policy.layout.dimension, policy.layout.dimension))

    assert np.empty is original_empty
    assert np.linalg.eigvalsh is original_eigvalsh
    assert guard.events[-1].operator == "numpy.empty"
    assert guard.events[-1].executed is False


def test_numpy_guard_preflights_keyword_shapes_axes_and_linalg_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _production_policy()
    called: list[str] = []

    def forbidden_original(*_args: object, **_kwargs: object) -> None:
        called.append("called")
        raise AssertionError("guard called the original")

    monkeypatch.setattr(np, "empty", forbidden_original)
    with pytest.raises(H8ForbiddenAllocation, match="dense"):
        with H8NumpyAllocationGuard(policy):
            np.empty(
                shape=(policy.layout.dimension, policy.layout.dimension),
            )
    assert not called

    monkeypatch.setattr(np, "full", forbidden_original)
    with pytest.raises(H8ForbiddenAllocation, match="dense"):
        with H8NumpyAllocationGuard(policy):
            np.full(
                fill_value=0.0,
                shape=(policy.layout.dimension, policy.layout.dimension),
            )
    assert not called

    base = np.ones((2, 2), dtype=np.float64)
    monkeypatch.setattr(np, "reshape", forbidden_original)
    unregistered_guard = H8NumpyAllocationGuard(policy)
    with pytest.raises(H8ForbiddenAllocation, match="unregistered non-control"):
        with unregistered_guard:
            with unregistered_guard.semantic_site("local"):
                np.reshape(a=base, shape=(2, 2))
    assert unregistered_guard.events[-1].executed is False

    reshape_guard = H8NumpyAllocationGuard(policy)
    reshape_guard.register_preexisting(base, site="local", logical_shape=(2, 2))
    with pytest.raises(H8ForbiddenAllocation, match="dense"):
        with reshape_guard:
            np.reshape(
                a=base,
                shape=(policy.layout.dimension, policy.layout.dimension),
            )
    assert not called

    monkeypatch.setattr(np, "resize", forbidden_original)
    resize_guard = H8NumpyAllocationGuard(policy)
    resize_guard.register_preexisting(base, site="local", logical_shape=(2, 2))
    with pytest.raises(H8ForbiddenAllocation, match="dense"):
        with resize_guard:
            np.resize(
                a=base,
                new_shape=(policy.layout.dimension, policy.layout.dimension),
            )
    assert not called

    block = np.ones((2, 2), dtype=np.float64)
    monkeypatch.setattr(np, "stack", forbidden_original)
    stack_guard = H8NumpyAllocationGuard(policy)
    stack_guard.register_preexisting(
        block,
        site="factor.diagonal",
        logical_shape=(129, 40, 40),
    )
    with pytest.raises(H8ForbiddenAllocation, match="pair slab"):
        with stack_guard:
            np.stack([block] * 129, axis=0)
    assert not called

    monkeypatch.setattr(np, "concatenate", forbidden_original)
    concatenate_guard = H8NumpyAllocationGuard(policy)
    concatenate_guard.register_preexisting(
        block,
        site="factor.diagonal",
        logical_shape=(129, 40, 40),
    )
    with pytest.raises(H8ForbiddenAllocation, match="combined pair"):
        with concatenate_guard:
            np.concatenate([block] * 129, 0)
    assert not called

    eigenvalues = np.ones((2,), dtype=np.float64)
    eigenvectors = np.ones((2, 2), dtype=np.float64)

    def fake_eig(_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return eigenvalues, eigenvectors

    monkeypatch.setattr(np.linalg, "eig", fake_eig)
    eig_guard = H8NumpyAllocationGuard(policy)
    eig_guard.register_preexisting(base, site="local", logical_shape=(2, 2))
    with eig_guard:
        with eig_guard.semantic_site("local"):
            np.linalg.eig(base)
    assert eig_guard.events[-1].output_shapes == ((2,), (2, 2))
    assert eig_guard.events[-1].executed

    monkeypatch.setattr(np, "empty", forbidden_original)
    with pytest.raises(H8ForbiddenAllocation, match="storage cap"):
        with H8NumpyAllocationGuard(policy):
            np.empty((205_601,), np.complex128)
    assert not called

    matrix = np.ones((3, 2), dtype=np.float64)
    rhs = np.ones((3,), dtype=np.float64)
    solution = np.ones((2,), dtype=np.float64)
    residuals = np.ones((1,), dtype=np.float64)
    rank = np.int64(2)
    singular_values = np.ones((2,), dtype=np.float64)

    def fake_lstsq(
        _matrix: np.ndarray,
        _rhs: np.ndarray,
        *,
        rcond: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.int64, np.ndarray]:
        del rcond
        return solution, residuals, rank, singular_values

    monkeypatch.setattr(np.linalg, "lstsq", fake_lstsq)
    lstsq_guard = H8NumpyAllocationGuard(policy)
    lstsq_guard.register_preexisting(matrix, site="local", logical_shape=(3, 2))
    lstsq_guard.register_preexisting(rhs, site="local", logical_shape=(3,))
    with lstsq_guard:
        with lstsq_guard.semantic_site("local"):
            np.linalg.lstsq(matrix, rhs, rcond=None)
    assert lstsq_guard.events[-1].output_shapes == ((2,), (1,), (), (2,))
    assert lstsq_guard.events[-1].executed

    empty_residuals = np.ones((0,), dtype=np.float64)
    deficient_rank = np.int64(1)

    def fake_rank_deficient_lstsq(
        _matrix: np.ndarray,
        _rhs: np.ndarray,
        *,
        rcond: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.int64, np.ndarray]:
        del rcond
        return solution, empty_residuals, deficient_rank, singular_values

    monkeypatch.setattr(np.linalg, "lstsq", fake_rank_deficient_lstsq)
    deficient_guard = H8NumpyAllocationGuard(policy)
    deficient_guard.register_preexisting(matrix, site="local", logical_shape=(3, 2))
    deficient_guard.register_preexisting(rhs, site="local", logical_shape=(3,))
    with deficient_guard:
        with deficient_guard.semantic_site("local"):
            np.linalg.lstsq(matrix, rhs, rcond=None)
    assert deficient_guard.events[-1].output_shapes == ((2,), (0,), (), (2,))
    assert deficient_guard.events[-1].executed

    impossible_residuals = np.ones((2,), dtype=np.float64)

    def fake_malformed_lstsq(
        _matrix: np.ndarray,
        _rhs: np.ndarray,
        *,
        rcond: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.int64, np.ndarray]:
        del rcond
        return solution, impossible_residuals, rank, singular_values

    monkeypatch.setattr(np.linalg, "lstsq", fake_malformed_lstsq)
    malformed_guard = H8NumpyAllocationGuard(policy)
    malformed_guard.register_preexisting(matrix, site="local", logical_shape=(3, 2))
    malformed_guard.register_preexisting(rhs, site="local", logical_shape=(3,))
    with pytest.raises(H8ForbiddenAllocation, match="every preflighted"):
        with malformed_guard:
            with malformed_guard.semantic_site("local"):
                np.linalg.lstsq(matrix, rhs, rcond=None)
    assert malformed_guard.events[-1].executed

    incompatible_rhs = np.ones((2,), dtype=np.float64)
    monkeypatch.setattr(np.linalg, "lstsq", forbidden_original)
    forbidden_lstsq_guard = H8NumpyAllocationGuard(policy)
    forbidden_lstsq_guard.register_preexisting(
        matrix,
        site="local",
        logical_shape=(3, 2),
    )
    forbidden_lstsq_guard.register_preexisting(
        incompatible_rhs,
        site="local",
        logical_shape=(2,),
    )
    with pytest.raises(H8ForbiddenAllocation, match="right-hand side"):
        with forbidden_lstsq_guard:
            with forbidden_lstsq_guard.semantic_site("local"):
                np.linalg.lstsq(matrix, incompatible_rhs, rcond=None)
    assert forbidden_lstsq_guard.events[-1].executed is False
    assert not called


def _joined_rows() -> tuple[
    tuple[H8RawProfilerEvent, ...],
    tuple[H8ProfilerEnrichment, ...],
]:
    baseline = H8TensorKey(1, 100, 10, "cpu")
    created = H8TensorKey(2, 200, 20, "cpu")
    rows = (
        H8RawProfilerEvent(0, 10, "PREEXISTING", baseline, 0, 8),
        H8RawProfilerEvent(1, 20, "CREATE", created, 0, 16),
        H8RawProfilerEvent(2, 30, "INCREMENT_VERSION", created, 1, 0),
        H8RawProfilerEvent(3, 40, "DESTROY", created, 1, -16),
        H8RawProfilerEvent(4, 50, "DESTROY", baseline, 0, -8),
    )
    enrichments = tuple(
        H8ProfilerEnrichment(
            source_row_index=row.source_row_index,
            tensor_key=row.tensor_key,
            version=row.version,
            dtype="torch.float64",
            operator="aten.empty",
            stack=("h8_child.py:1",),
            logical_shape=((1,) if row.tensor_key == baseline else (2,)),
            classification="local",
            matched_event_node_indices=(row.source_row_index,),
            storage_span_start=row.tensor_key.storage_ptr,
            storage_span_end=(
                row.tensor_key.storage_ptr + (8 if row.tensor_key == baseline else 16)
            ),
            storage_nbytes=8 if row.tensor_key == baseline else 16,
        )
        for row in rows
    )
    return rows, enrichments


def _profiler_enrichment(
    row: H8RawProfilerEvent,
    *,
    logical_shape: tuple[int, ...],
    storage_nbytes: int,
    alias_of: H8TensorKey | None = None,
) -> H8ProfilerEnrichment:
    return H8ProfilerEnrichment(
        source_row_index=row.source_row_index,
        tensor_key=row.tensor_key,
        version=row.version,
        dtype="torch.float64",
        operator="aten.empty",
        stack=("h8_child.py:1",),
        logical_shape=logical_shape,
        classification="local",
        matched_event_node_indices=(row.source_row_index,),
        storage_span_start=row.tensor_key.storage_ptr,
        storage_span_end=row.tensor_key.storage_ptr + storage_nbytes,
        storage_nbytes=storage_nbytes,
        alias_of=alias_of,
    )


def _joined_alias_rows(
    *,
    alias_pointer: int = 1_000,
    alias_allocation_id: int = 70,
) -> tuple[
    tuple[H8RawProfilerEvent, ...],
    tuple[H8ProfilerEnrichment, ...],
]:
    base = H8TensorKey(10, 1_000, 70, "cpu")
    alias = H8TensorKey(11, alias_pointer, alias_allocation_id, "cpu")
    rows = (
        H8RawProfilerEvent(0, 10, "PREEXISTING", base, 0, 16),
        H8RawProfilerEvent(1, 20, "CREATE", alias, 0, 8),
        H8RawProfilerEvent(2, 30, "INCREMENT_VERSION", alias, 1, 0),
        H8RawProfilerEvent(3, 40, "DESTROY", alias, 1, -8),
        H8RawProfilerEvent(4, 50, "DESTROY", base, 0, -16),
    )
    enrichments = tuple(
        _profiler_enrichment(
            row,
            logical_shape=(1,) if row.tensor_key == alias else (2,),
            storage_nbytes=8 if row.tensor_key == alias else 16,
            alias_of=base if row.tensor_key == alias else None,
        )
        for row in rows
    )
    return rows, enrichments


def test_profiler_parser_joins_deduplicates_and_reconstructs_liveness() -> None:
    rows, enrichments = _joined_rows()
    trace = parse_h8_profiler_events(
        rows + (rows[2],),
        enrichments + (enrichments[2],),
        policy=_production_policy(),
    )

    assert tuple(event.action for event in trace.events) == (
        "PREEXISTING",
        "CREATE",
        "INCREMENT_VERSION",
        "DESTROY",
        "DESTROY",
    )
    assert trace.preexisting_storage_count == 1
    assert trace.preexisting_bytes == 8
    assert trace.baseline_live_bytes == 8
    assert trace.live_peak_bytes == 24
    assert trace.events[2].live_bytes_after == 24
    assert trace.all_joined_and_liveness_reconciled
    with pytest.raises(dataclasses.FrozenInstanceError):
        trace.events[0].nbytes = 9  # type: ignore[misc]

    alias_rows, alias_enrichments = _joined_alias_rows()
    assert alias_enrichments[0].storage_nbytes == 16
    assert alias_enrichments[1].storage_nbytes == 8
    alias_trace = parse_h8_profiler_events(
        alias_rows,
        alias_enrichments,
        policy=_production_policy(),
    )
    assert alias_trace.live_peak_bytes == 16
    assert alias_trace.events[1].live_bytes_after == 16


def test_profiler_parser_fails_closed_on_join_and_liveness_gaps() -> None:
    rows, enrichments = _joined_rows()
    with pytest.raises(H8ProfilerObservabilityGap, match="empty"):
        parse_h8_profiler_events((), (), policy=_production_policy())

    with pytest.raises(H8ProfilerObservabilityGap, match="partial"):
        parse_h8_profiler_events(
            rows[:2] + rows[3:],
            enrichments[:2] + enrichments[3:],
            policy=_production_policy(),
        )

    with pytest.raises(H8ProfilerObservabilityGap, match="join"):
        parse_h8_profiler_events(
            rows,
            enrichments[:-1],
            policy=_production_policy(),
        )

    duplicate_create = H8RawProfilerEvent(
        5,
        21,
        "CREATE",
        rows[1].tensor_key,
        0,
        16,
    )
    duplicate_enrichment = dataclasses.replace(
        enrichments[1],
        source_row_index=5,
        matched_event_node_indices=(5,),
    )
    with pytest.raises(H8ForbiddenAllocation, match="duplicate CREATE"):
        parse_h8_profiler_events(
            rows + (duplicate_create,),
            enrichments + (duplicate_enrichment,),
            policy=_production_policy(),
        )

    lossy = parse_h8_lossy_profiler_rows(
        ((10, "CREATE", 8, "CPU"),),
    )
    assert lossy[0].nbytes == 8
    with pytest.raises(H8ProfilerObservabilityGap, match="lossy"):
        parse_h8_profiler_events(lossy, (), policy=_production_policy())


def test_profiler_alias_requires_one_already_live_exact_containing_target() -> None:
    rows, enrichments = _joined_alias_rows()
    nonexistent = dataclasses.replace(
        enrichments[1],
        alias_of=H8TensorKey(99, 1_000, 70, "cpu"),
    )
    with pytest.raises(H8ProfilerObservabilityGap, match="nonexistent"):
        parse_h8_profiler_events(
            rows,
            enrichments[:1] + (nonexistent,) + enrichments[2:],
            policy=_production_policy(),
        )

    baseline = H8TensorKey(20, 300, 30, "cpu")
    future_target = H8TensorKey(21, 400, 40, "cpu")
    future_alias = H8TensorKey(22, 400, 40, "cpu")
    future_rows = (
        H8RawProfilerEvent(0, 10, "PREEXISTING", baseline, 0, 8),
        H8RawProfilerEvent(1, 20, "CREATE", future_alias, 0, 8),
        H8RawProfilerEvent(2, 30, "CREATE", future_target, 0, 16),
        H8RawProfilerEvent(3, 40, "INCREMENT_VERSION", future_target, 1, 0),
        H8RawProfilerEvent(4, 50, "DESTROY", future_target, 1, -16),
        H8RawProfilerEvent(5, 60, "DESTROY", future_alias, 0, -8),
        H8RawProfilerEvent(6, 70, "DESTROY", baseline, 0, -8),
    )
    future_enrichments = tuple(
        _profiler_enrichment(
            row,
            logical_shape=(
                (1,) if row.tensor_key in (baseline, future_alias) else (2,)
            ),
            storage_nbytes=(8 if row.tensor_key in (baseline, future_alias) else 16),
            alias_of=future_target if row.tensor_key == future_alias else None,
        )
        for row in future_rows
    )
    with pytest.raises(H8ProfilerObservabilityGap, match="future"):
        parse_h8_profiler_events(
            future_rows,
            future_enrichments,
            policy=_production_policy(),
        )

    destroyed_target = H8TensorKey(30, 500, 50, "cpu")
    destroyed_alias = H8TensorKey(31, 500, 50, "cpu")
    destroyed_rows = (
        H8RawProfilerEvent(0, 10, "PREEXISTING", destroyed_target, 0, 16),
        H8RawProfilerEvent(1, 20, "DESTROY", destroyed_target, 0, -16),
        H8RawProfilerEvent(2, 30, "CREATE", destroyed_alias, 0, 8),
        H8RawProfilerEvent(3, 40, "INCREMENT_VERSION", destroyed_alias, 1, 0),
        H8RawProfilerEvent(4, 50, "DESTROY", destroyed_alias, 1, -8),
    )
    destroyed_enrichments = tuple(
        _profiler_enrichment(
            row,
            logical_shape=(1,) if row.tensor_key == destroyed_alias else (2,),
            storage_nbytes=8 if row.tensor_key == destroyed_alias else 16,
            alias_of=(destroyed_target if row.tensor_key == destroyed_alias else None),
        )
        for row in destroyed_rows
    )
    with pytest.raises(H8ProfilerObservabilityGap, match="destroyed"):
        parse_h8_profiler_events(
            destroyed_rows,
            destroyed_enrichments,
            policy=_production_policy(),
        )

    first_target = H8TensorKey(40, 600, 60, "cpu")
    second_target = H8TensorKey(40, 700, 61, "cpu")
    ambiguous_alias = H8TensorKey(41, 600, 60, "cpu")
    ambiguous_rows = (
        H8RawProfilerEvent(0, 10, "PREEXISTING", first_target, 0, 16),
        H8RawProfilerEvent(1, 11, "PREEXISTING", second_target, 0, 16),
        H8RawProfilerEvent(2, 20, "CREATE", ambiguous_alias, 0, 8),
        H8RawProfilerEvent(3, 30, "INCREMENT_VERSION", ambiguous_alias, 1, 0),
        H8RawProfilerEvent(4, 40, "DESTROY", ambiguous_alias, 1, -8),
        H8RawProfilerEvent(5, 50, "DESTROY", first_target, 0, -16),
        H8RawProfilerEvent(6, 51, "DESTROY", second_target, 0, -16),
    )
    ambiguous_enrichments = tuple(
        _profiler_enrichment(
            row,
            logical_shape=(1,) if row.tensor_key == ambiguous_alias else (2,),
            storage_nbytes=8 if row.tensor_key == ambiguous_alias else 16,
            alias_of=first_target if row.tensor_key == ambiguous_alias else None,
        )
        for row in ambiguous_rows
    )
    with pytest.raises(H8ProfilerObservabilityGap, match="ambiguous"):
        parse_h8_profiler_events(
            ambiguous_rows,
            ambiguous_enrichments,
            policy=_production_policy(),
        )

    outside_rows, outside_enrichments = _joined_alias_rows(alias_pointer=1_009)
    with pytest.raises(H8ForbiddenAllocation, match="containing storage span"):
        parse_h8_profiler_events(
            outside_rows,
            outside_enrichments,
            policy=_production_policy(),
        )

    incompatible_rows, incompatible_enrichments = _joined_alias_rows(
        alias_allocation_id=71,
    )
    with pytest.raises(H8ForbiddenAllocation, match="incompatible storage"):
        parse_h8_profiler_events(
            incompatible_rows,
            incompatible_enrichments,
            policy=_production_policy(),
        )


def test_profiler_parser_rejects_bad_versions_leaks_and_unclassifiable_rows() -> None:
    rows, enrichments = _joined_rows()
    bad_increment = dataclasses.replace(rows[2], version=3)
    bad_increment_join = dataclasses.replace(enrichments[2], version=3)
    with pytest.raises(H8ForbiddenAllocation, match="nonmonotone"):
        parse_h8_profiler_events(
            rows[:2] + (bad_increment,) + rows[3:],
            enrichments[:2] + (bad_increment_join,) + enrichments[3:],
            policy=_production_policy(),
        )

    with pytest.raises(H8ForbiddenAllocation, match="leaked"):
        parse_h8_profiler_events(
            rows[:3] + (rows[4],),
            enrichments[:3] + (enrichments[4],),
            policy=_production_policy(),
        )

    forbidden_shape = dataclasses.replace(
        enrichments[1],
        logical_shape=(5160,),
    )
    with pytest.raises(H8ForbiddenAllocation, match="global axis"):
        parse_h8_profiler_events(
            rows,
            enrichments[:1] + (forbidden_shape,) + enrichments[2:],
            policy=_production_policy(),
        )

    unknown_dtype = dataclasses.replace(enrichments[1], dtype="torch.unknown")
    with pytest.raises(H8ProfilerObservabilityGap, match="dtype"):
        parse_h8_profiler_events(
            rows,
            enrichments[:1] + (unknown_dtype,) + enrichments[2:],
            policy=_production_policy(),
        )

    ambiguous_join = dataclasses.replace(
        enrichments[1],
        operator="aten.zeros",
    )
    with pytest.raises(H8ProfilerObservabilityGap, match="nonunique"):
        parse_h8_profiler_events(
            rows,
            enrichments + (ambiguous_join,),
            policy=_production_policy(),
        )

    byte_drift = dataclasses.replace(
        enrichments[1],
        storage_span_end=224,
        storage_nbytes=24,
    )
    with pytest.raises(H8ForbiddenAllocation, match="bytes and storage span"):
        parse_h8_profiler_events(
            rows,
            enrichments[:1] + (byte_drift,) + enrichments[2:],
            policy=_production_policy(),
        )

    over_cap_bytes = (411_200 * 8) + 8
    over_cap_create = dataclasses.replace(rows[1], nbytes=over_cap_bytes)
    over_cap_join = dataclasses.replace(
        enrichments[1],
        storage_span_end=enrichments[1].storage_span_start + over_cap_bytes,
        storage_nbytes=over_cap_bytes,
    )
    with pytest.raises(H8ForbiddenAllocation, match="profiler storage"):
        parse_h8_profiler_events(
            rows[:1] + (over_cap_create,) + rows[2:],
            enrichments[:1] + (over_cap_join,) + enrichments[2:],
            policy=_production_policy(),
        )


def test_negative_control_definitions_are_exact_and_preflight_rejected() -> None:
    policy = _production_policy()
    specs = h8_negative_control_specs(policy.layout)
    assert tuple(spec.control_id for spec in specs) == (
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
    assert specs[7].assigned_channels == ("backend", "dispatch")
    assert specs[-1].assigned_channels == ("numpy_guard",)

    for spec in specs:
        with pytest.raises(H8ForbiddenAllocation):
            policy.preflight_control(spec)


def test_control_detection_requires_pre_execution_assigned_reason() -> None:
    spec = h8_negative_control_specs(_production_policy().layout)[9]
    event = H8NumpyGuardEvent(
        sequence=0,
        operator=spec.requested_operation,
        semantic_site=None,
        control_id=spec.control_id,
        input_shapes=(),
        output_shapes=spec.logical_shapes,
        dtype=None,
        float64_equivalent_scalars=0,
        executed=False,
        forbidden_reason=spec.expected_reason,
    )
    assert h8_control_detected_pre_execution(
        spec,
        (event,),
        operation_returned=False,
        caught_forbidden=True,
    )
    assert not h8_control_detected_pre_execution(
        spec,
        (dataclasses.replace(event, executed=True),),
        operation_returned=False,
        caught_forbidden=True,
    )
    assert not h8_control_detected_pre_execution(
        spec,
        (event,),
        operation_returned=True,
        caught_forbidden=True,
    )

    result = make_h8_control_result(
        spec,
        observed_channels=("numpy_guard",),
        detected=False,
        event_payload={"event": dataclasses.replace(event, executed=True)},
    )
    assert result.status is GateStatus.FAIL
    assert result.obligations == ("negative_control_executed_past_detector",)


def _attempt_from_fake_envelope(
    tmp_path: Path,
    envelope: dict[str, object],
) -> tuple[H8ChildAttemptRecord, Mapping[str, object]]:
    request = H8ChildRequest(
        mode=envelope["mode"],  # type: ignore[arg-type]
        seed=envelope["seed"],  # type: ignore[arg-type]
        repetition=envelope["repetition"],  # type: ignore[arg-type]
        config_sha256=envelope["config_sha256"],  # type: ignore[arg-type]
        protocol_sha256=envelope["protocol_sha256"],  # type: ignore[arg-type]
        control_id=envelope["control_id"],  # type: ignore[arg-type]
    )
    identities = envelope["identities"]
    assert isinstance(identities, dict)
    invocation = build_h8_child_invocation(
        dataclasses.asdict(request),
        repository_root=tmp_path,
        identities=identities,
        base_environment={},
    )
    envelope["request_sha256"] = hashlib.sha256(
        invocation.stdin[:-1]
    ).hexdigest()
    process_record = H8ChildProcessRecord.from_payload(envelope)
    assert parse_h8_child_stdout(process_record.stdout) == envelope
    decision = classify_h8_child_outcome(
        process_record,
        valid_start=True,
        invocation=invocation,
    )
    attempt = make_h8_child_attempt_record(
        request,
        invocation,
        process_record,
        decision,
    )
    assert isinstance(decision.payload, Mapping)
    return attempt, decision.payload


def test_child_invocation_is_exact_and_freezes_thread_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = {
        "mode": "production",
        "seed": 20260721,
        "repetition": 0,
        "config_sha256": "b" * 64,
        "protocol_sha256": "c" * 64,
        "control_id": None,
    }
    identities = _child_envelope()["identities"]
    base_environment = {
        "PATH": "preserved",
        "mkl_threading_layer": "hostile-alias",
        "vfe4_h8_child_identities_json": "hostile-alias",
    }
    invocation = build_h8_child_invocation(
        request,
        repository_root=tmp_path,
        identities=identities,
        base_environment=base_environment,
    )
    base_environment["PATH"] = "mutated after invocation construction"

    assert invocation.argv[1:] == ("-m", "verification.h8_child")
    assert invocation.cwd == tmp_path.resolve()
    assert invocation.stdin == (
        b'{"config_sha256":"'
        + (b"b" * 64)
        + b'","control_id":null,"mode":"production","protocol_sha256":"'
        + (b"c" * 64)
        + b'","repetition":0,"seed":20260721}\n'
    )
    assert invocation.timeout_seconds == 60.0
    assert invocation.capture_stdout
    assert invocation.capture_stderr
    assert invocation.environment["PATH"] == "preserved"
    assert {
        name: invocation.environment[name]
        for name, _value in H8_THREAD_ENVIRONMENT_ITEMS
    } == dict(H8_THREAD_ENVIRONMENT_ITEMS)
    assert "mkl_threading_layer" not in invocation.environment
    assert "vfe4_h8_child_identities_json" not in invocation.environment
    assert tuple(
        key
        for key in invocation.environment
        if key.casefold() == H8_CHILD_IDENTITY_ENV.casefold()
    ) == (H8_CHILD_IDENTITY_ENV,)
    aliased_identity_environment = dict(invocation.environment)
    identity_payload = aliased_identity_environment.pop(H8_CHILD_IDENTITY_ENV)
    aliased_identity_environment[
        H8_CHILD_IDENTITY_ENV.casefold()
    ] = identity_payload
    with pytest.raises(ValueError, match="canonical key"):
        dataclasses.replace(
            invocation,
            environment=aliased_identity_environment,
        )
    assert not any(
        key.casefold()
        in {name.casefold() for name in H8_FORBIDDEN_ENVIRONMENT}
        for key in invocation.environment
    )
    assert require_h8_startup_environment(invocation.environment) == dict(
        H8_THREAD_ENVIRONMENT_ITEMS
    )

    script_environment = {
        "Path": "preserved",
        "mkl_threading_layer": "hostile-alias",
        "omp_num_threads": "99",
    }
    prepare_h8_script_environment(script_environment)
    assert script_environment["Path"] == "preserved"
    assert {
        name: script_environment[name]
        for name, _value in H8_THREAD_ENVIRONMENT_ITEMS
    } == dict(H8_THREAD_ENVIRONMENT_ITEMS)
    assert "mkl_threading_layer" not in script_environment
    assert "omp_num_threads" not in script_environment

    with pytest.raises(ValueError, match="KMP_DUPLICATE_LIB_OK"):
        prepare_h8_script_environment(
            {"kmp_duplicate_lib_ok": "TRUE"},
        )
    with pytest.raises(TypeError):
        invocation.environment["PATH"] = "mutated"  # type: ignore[index]

    import verification.h8_child as h8_child
    from vfe4.config import H8ValidationConfig
    from verification.h8_protocol import build_h8_protocol_sha256

    runtime_config = H8ValidationConfig.create()
    control_id = str(_control_child_envelope()["control_id"])
    child_request = {
        "mode": "negative_control",
        "seed": 20260721,
        "repetition": None,
        "config_sha256": runtime_config.config_sha256,
        "protocol_sha256": build_h8_protocol_sha256(runtime_config),
        "control_id": control_id,
    }
    child_identities = _child_envelope()["identities"]
    assert isinstance(child_identities, Mapping)
    monkeypatch.setenv(
        H8_CHILD_IDENTITY_ENV,
        canonical_json_bytes(child_identities).decode("ascii"),
    )
    fake_torch = SimpleNamespace(
        no_grad=lambda: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        h8_child,
        "_load_runtime",
        lambda: (fake_torch, object()),
    )
    monkeypatch.setattr(
        h8_child,
        "_set_and_verify_torch_threads",
        lambda _torch: None,
    )
    monkeypatch.setattr(
        h8_child,
        "_collect_identities",
        lambda **_kwargs: dict(child_identities),
    )

    def drifting_control(*_args: object, **_kwargs: object) -> object:
        monkeypatch.setenv("kMp_DuPlIcAtE_LiB_oK", "TRUE")
        return _control_child_envelope()["control"]

    monkeypatch.setattr(
        h8_child,
        "_run_negative_control",
        drifting_control,
    )
    stdin_bytes = io.BytesIO(canonical_json_bytes(child_request) + b"\n")
    stdout_bytes = io.BytesIO()
    stdin_text = io.TextIOWrapper(stdin_bytes, encoding="utf-8")
    stdout_text = io.TextIOWrapper(stdout_bytes, encoding="utf-8")
    monkeypatch.setattr(sys, "stdin", stdin_text)
    monkeypatch.setattr(sys, "stdout", stdout_text)

    assert h8_child.main() == 0
    stdout_text.flush()
    drift_payload = json.loads(stdout_bytes.getvalue())
    assert drift_payload["status"] == "inconclusive"
    assert drift_payload["error"]["kind"] == "environment_observability_gap"
    assert "KMP_DUPLICATE_LIB_OK" in drift_payload["error"]["message"]


@pytest.mark.parametrize(
    "changes",
    (
        {"argv": ("python", "-m", "not_the_h8_child")},
        {"timeout_seconds": 59.0},
        {
            "environment": {
                H8_CHILD_IDENTITY_ENV: "{}",
                "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
                "MKL_THREADING_LAYER": "SEQUENTIAL",
            },
        },
        {
            "environment": {
                H8_CHILD_IDENTITY_ENV: "{}",
                **dict(H8_THREAD_ENVIRONMENT_ITEMS),
                "kMp_DuPlIcAtE_LiB_oK": "TRUE",
            },
        },
    ),
)
def test_child_invocation_rejects_noncanonical_launch_contract(
    tmp_path: Path,
    changes: dict[str, object],
) -> None:
    request = {
        "mode": "production",
        "seed": 20260721,
        "repetition": 0,
        "config_sha256": "b" * 64,
        "protocol_sha256": "c" * 64,
        "control_id": None,
    }
    invocation = build_h8_child_invocation(
        request,
        repository_root=tmp_path,
        identities=_child_envelope()["identities"],  # type: ignore[arg-type]
        base_environment={},
    )

    with pytest.raises(ValueError, match="exact|thread|environment|KMP"):
        dataclasses.replace(invocation, **changes)


def test_child_stdout_parser_requires_one_canonical_json_line() -> None:
    envelope = _child_envelope()
    canonical = (
        json.dumps(
            envelope,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    parsed = parse_h8_child_stdout(canonical)
    assert parsed == envelope

    malformed = b'{"schema_version":"h8-child-v2"}\n{"schema_version":"h8-child-v2"}\n'
    with pytest.raises(ValueError, match="one canonical JSON line"):
        parse_h8_child_stdout(malformed)
    with pytest.raises(ValueError, match="canonical"):
        parse_h8_child_stdout(b'{ "schema_version": "h8-child-v2" }\n')
    with pytest.raises(ValueError, match="finite"):
        parse_h8_child_stdout(
            b'{"value":NaN}\n',
        )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        parse_h8_child_stdout(
            b'{"status":"pass","status":"fail"}\n',
        )


def test_public_child_result_decoder_returns_the_closed_typed_record() -> None:
    envelope = _child_envelope()

    decoded = decode_h8_child_result(envelope)

    assert type(decoded) is H8ChildResult
    assert decoded.mode == "production"
    assert decoded.seed == 20260721
    assert decoded.repetition == 0
    assert decoded.input_sha256 == "1" * 64
    assert type(decoded.allocation) is H8AllocationRecord
    assert decoded.allocation.observed_channels == (
        "dispatch",
        "numpy_guard",
        "backend",
        "os_hwm",
    )
    with pytest.raises(ValueError, match="control"):
        decode_h8_control_result(envelope)


def test_child_failure_classification_preserves_witnessed_failure_dominance() -> None:
    incomplete = b'{"schema_version":"h8-child-v2"'
    witnessed_nonzero = H8ChildProcessRecord(
        timed_out=False,
        exit_code=3,
        stdout=incomplete,
        stderr=b"later observability unavailable",
        parent_elapsed_ns=7,
    )
    decision = classify_h8_child_outcome(
        witnessed_nonzero,
        valid_start=True,
    )
    assert decision.status is GateStatus.FAIL
    assert "nonzero_child_exit" in decision.reasons
    assert "invalid_child_stdout" in decision.reasons
    assert decision.stdout_sha256
    assert decision.stderr_sha256
    assert decision.exit_code == 3

    timeout = classify_h8_child_outcome(
        H8ChildProcessRecord(
            timed_out=True,
            exit_code=None,
            stdout=b"",
            stderr=b"",
            parent_elapsed_ns=60_000_000_001,
        ),
        valid_start=True,
    )
    assert timeout.status is GateStatus.FAIL
    assert timeout.reasons == ("child_timeout",)

    missing_join = _child_envelope(
        status="inconclusive",
        obligations=["profiler_event_tree_join_missing_or_nonunique"],
        result=None,
        error={
            "kind": "profiler_observability_gap",
            "message": "join missing",
            "witnessed_violation": False,
        },
    )
    decision = classify_h8_child_outcome(
        H8ChildProcessRecord.from_payload(missing_join),
        valid_start=True,
    )
    assert decision.status is GateStatus.INCONCLUSIVE

    resource_failure = _child_envelope(
        status="fail",
        obligations=[],
        result={
            "input_sha256": "1" * 64,
            "sample_noise_sha256": "2" * 64,
            "problem_evidence": {},
            "objective": {},
            "storage": {},
            "fill": {},
            "workspace": {},
            "counters": {},
            "allocation": {},
            "resources": {},
            "diagnostics": {},
            "operation_reachability": {},
            "residuals": {},
            "resource_decisions": {
                "process_memory_pass": False,
                "conservative_incremental_hwm_bytes": 134_217_729,
            },
            "invariants": [],
        },
    )
    decision = classify_h8_child_outcome(
        H8ChildProcessRecord.from_payload(resource_failure),
        valid_start=True,
    )
    assert decision.status is GateStatus.FAIL
    assert "child_reported_witnessed_failure" in decision.reasons


def test_child_identity_or_hwm_unavailability_is_not_silently_zero() -> None:
    missing_thread = _child_envelope()
    identities = dict(missing_thread["identities"])  # type: ignore[arg-type]
    identities.pop("thread")
    missing_thread["identities"] = identities
    decision = classify_h8_child_outcome(
        H8ChildProcessRecord.from_payload(missing_thread),
        valid_start=True,
    )
    assert decision.status is GateStatus.INCONCLUSIVE
    assert "invalid_child_stdout" in decision.reasons

    conservative, peak_delta = conservative_hwm_endpoints(
        pre_current_rss_bytes=100,
        pre_lifetime_peak_bytes=150,
        post_lifetime_peak_bytes=180,
    )
    assert conservative == 80
    assert peak_delta == 30
    with pytest.raises(ValueError, match="nonnegative"):
        conservative_hwm_endpoints(
            pre_current_rss_bytes=-1,
            pre_lifetime_peak_bytes=0,
            post_lifetime_peak_bytes=0,
        )

    fields64, expected64 = windows_process_memory_layout(pointer_size=8)
    assert fields64 == (
        "cb",
        "PageFaultCount",
        "PeakWorkingSetSize",
        "WorkingSetSize",
        "QuotaPeakPagedPoolUsage",
        "QuotaPagedPoolUsage",
        "QuotaPeakNonPagedPoolUsage",
        "QuotaNonPagedPoolUsage",
        "PagefileUsage",
        "PeakPagefileUsage",
        "PrivateUsage",
    )
    assert expected64 == 80
    assert windows_process_memory_layout(pointer_size=4)[1] == 44


def test_child_result_must_match_parent_request_and_environment(
    tmp_path: Path,
) -> None:
    request = {
        "mode": "production",
        "seed": 20260721,
        "repetition": 0,
        "config_sha256": "b" * 64,
        "protocol_sha256": "c" * 64,
        "control_id": None,
    }
    identities = _child_envelope()["identities"]
    invocation = build_h8_child_invocation(
        request,
        repository_root=tmp_path,
        identities=identities,
        base_environment={},
    )
    request_sha256 = hashlib.sha256(invocation.stdin[:-1]).hexdigest()
    matching = _child_envelope(
        request_sha256=request_sha256,
        identities=identities,
    )
    decision = classify_h8_child_outcome(
        H8ChildProcessRecord.from_payload(matching),
        valid_start=True,
        invocation=invocation,
    )
    assert decision.status is GateStatus.PASS

    mismatched = dict(matching)
    mismatched["request_sha256"] = "9" * 64
    decision = classify_h8_child_outcome(
        H8ChildProcessRecord.from_payload(mismatched),
        valid_start=True,
        invocation=invocation,
    )
    assert decision.status is GateStatus.FAIL
    assert "child_request_or_environment_identity_mismatch" in decision.reasons


def _verified_h8_production_attempt_inputs(
    tmp_path: Path,
    *,
    parent_elapsed_ns: int = 37,
) -> tuple[
    H8ChildRequest,
    H8ChildInvocation,
    H8ChildProcessRecord,
    H8ChildDecision,
    dict[str, object],
]:
    request = H8ChildRequest(
        mode="production",
        seed=20260721,
        repetition=0,
        config_sha256="b" * 64,
        protocol_sha256="c" * 64,
        control_id=None,
    )
    identities = _child_envelope()["identities"]
    assert isinstance(identities, dict)
    invocation = build_h8_child_invocation(
        dataclasses.asdict(request),
        repository_root=tmp_path,
        identities=identities,
        base_environment={},
    )
    payload = _child_envelope(
        request_sha256=hashlib.sha256(invocation.stdin[:-1]).hexdigest(),
        identities=identities,
    )
    process_record = H8ChildProcessRecord.from_payload(
        payload,
        parent_elapsed_ns=parent_elapsed_ns,
    )
    decision = classify_h8_child_outcome(
        process_record,
        valid_start=True,
        invocation=invocation,
    )
    return request, invocation, process_record, decision, payload


def test_h8_profiler_pins_reject_full_torch_version_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject any profiler runtime that is not the exact pinned install."""

    import verification.h8_child as h8_child
    from verification import h8_wire

    memory_source = b"fixture memory profiler source\n"
    profiler_source = b"fixture profiler source\n"
    expected_memory_hash = hashlib.sha256(memory_source).hexdigest()
    expected_profiler_hash = hashlib.sha256(profiler_source).hexdigest()
    assert h8_child._PROFILER_TORCH_VERSION == h8_wire.H8_PROFILER_TORCH_VERSION
    assert (
        h8_child._PROFILER_MEMORY_SOURCE_SHA256
        == h8_wire.H8_PROFILER_MEMORY_SOURCE_SHA256
    )
    assert h8_child._PROFILER_SOURCE_SHA256 == h8_wire.H8_PROFILER_SOURCE_SHA256
    assert (
        h8_child._PROFILER_API_CONTRACT_SHA256
        == h8_wire.H8_PROFILER_API_CONTRACT_SHA256
    )
    monkeypatch.setattr(
        h8_child,
        "_PROFILER_MEMORY_SOURCE_SHA256",
        expected_memory_hash,
    )
    monkeypatch.setattr(
        h8_child,
        "_PROFILER_SOURCE_SHA256",
        expected_profiler_hash,
    )

    torch_root = tmp_path / "torch"
    profiler_root = torch_root / "profiler"
    profiler_root.mkdir(parents=True)
    torch_init = torch_root / "__init__.py"
    memory_path = profiler_root / "_memory_profiler.py"
    profiler_path = profiler_root / "profiler.py"
    torch_init.write_bytes(b"fixture torch package\n")
    memory_path.write_bytes(memory_source)
    profiler_path.write_bytes(profiler_source)
    fake_torch = SimpleNamespace(
        __file__=str(torch_init),
        __version__=h8_wire.H8_PROFILER_TORCH_VERSION,
    )

    assert h8_child._verify_profiler_pins(fake_torch) == {
        "torch_version": h8_wire.H8_PROFILER_TORCH_VERSION,
        "memory_profile_source_sha256": expected_memory_hash,
        "profiler_source_sha256": expected_profiler_hash,
        "api_contract_sha256": h8_wire.H8_PROFILER_API_CONTRACT_SHA256,
    }

    for version in (
        h8_wire.H8_PROFILER_TORCH_VERSION.split("+", maxsplit=1)[0],
        "9.9.9+forged",
    ):
        fake_torch.__version__ = version
        with pytest.raises(h8_child._ProfilerUnavailable, match="exactly"):
            h8_child._verify_profiler_pins(fake_torch)
    fake_torch.__version__ = h8_wire.H8_PROFILER_TORCH_VERSION

    for path in (memory_path, profiler_path):
        original = path.read_bytes()
        path.write_bytes(original + b"drift\n")
        with pytest.raises(h8_child._ProfilerUnavailable, match="hash mismatch"):
            h8_child._verify_profiler_pins(fake_torch)
        path.write_bytes(original)


def test_h8_profiler_schema_inspector_is_bounded_with_fakes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inspect only the pinned tiny workload and reject private-schema drift."""

    from enum import Enum

    import verification.h8_child as h8_child
    from verification import h8_wire

    class _Action(Enum):
        PREEXISTING = 1
        CREATE = 2
        INCREMENT_VERSION = 3
        DESTROY = 4

    class _EventType(Enum):
        Allocation = 1
        TorchOp = 2

    class _ExtraFields_Allocation:
        pass

    class _ExtraFields_TorchOp:
        pass

    class _FakeTensor:
        def __init__(self, calls: list[object]) -> None:
            self._calls = calls

        def clone(self) -> _FakeTensor:
            self._calls.append("clone")
            return _FakeTensor(self._calls)

        def add_(self, value: float) -> _FakeTensor:
            self._calls.append(("add_", value))
            return self

    class _FakeContext:
        def __init__(
            self,
            calls: list[object],
            name: str,
            value: object | None = None,
        ) -> None:
            self._calls = calls
            self._name = name
            self._value = self if value is None else value

        def __enter__(self) -> object:
            self._calls.append((self._name, "enter"))
            return self._value

        def __exit__(self, *error: object) -> None:
            self._calls.append((self._name, "exit", error[0]))

    def make_fake_torch(
        *,
        action_type: type[Enum] = _Action,
        malformed_row: bool = False,
        complete_key: bool = True,
        include_torch_op: bool = True,
    ) -> tuple[SimpleNamespace, list[object]]:
        calls: list[object] = []
        storage = SimpleNamespace(
            ptr=101,
            allocation_id=202 if complete_key else None,
        )
        private_key = SimpleNamespace(
            id=303,
            storage=storage,
            device="cpu",
            storage_data_ptr=101,
            sizes=(2,),
            dtype="torch.float64",
        )
        timeline: tuple[object, ...] = (
            (
                11,
                action_type.CREATE,
                (private_key, 0),
                16,
            ),
            (
                12,
                action_type.INCREMENT_VERSION,
                (private_key, 0),
                16,
            ),
        )
        if malformed_row:
            timeline = (timeline[0][:-1], timeline[1])  # type: ignore[index]

        allocation_fields = _ExtraFields_Allocation()
        allocation_fields.id = private_key.id
        allocation_fields.storage = storage
        allocation_fields.device = private_key.device
        allocation_fields.alloc_size = 16
        allocation_node = SimpleNamespace(
            typed=(_EventType.Allocation, allocation_fields),
            children=(),
        )
        torch_op_fields = _ExtraFields_TorchOp()
        torch_op_fields.name = "aten::add_"
        torch_op_fields.inputs = (private_key,)
        torch_op_node = SimpleNamespace(
            typed=(_EventType.TorchOp, torch_op_fields),
            children=(),
        )
        roots = (
            (allocation_node, torch_op_node)
            if include_torch_op
            else (allocation_node,)
        )
        memory_profile = SimpleNamespace(timeline=timeline)
        kineto_results = SimpleNamespace(
            experimental_event_tree=lambda: (
                calls.append("experimental_event_tree") or roots
            )
        )
        profile_value = SimpleNamespace(
            _memory_profile=lambda: (
                calls.append("_memory_profile") or memory_profile
            ),
            profiler=SimpleNamespace(kineto_results=kineto_results),
        )

        def profile(**kwargs: object) -> _FakeContext:
            calls.append(("profile", kwargs))
            return _FakeContext(calls, "profile", profile_value)

        def no_grad() -> _FakeContext:
            calls.append("no_grad")
            return _FakeContext(calls, "no_grad")

        def empty(
            shape: tuple[int, ...],
            *,
            dtype: object,
            device: str,
        ) -> _FakeTensor:
            calls.append(("empty", shape, dtype, device))
            return _FakeTensor(calls)

        fake_torch = SimpleNamespace(
            __file__=str(torch_init),
            __version__=h8_wire.H8_PROFILER_TORCH_VERSION,
            float64="fake-float64",
            no_grad=no_grad,
            empty=empty,
            profiler=SimpleNamespace(
                ProfilerActivity=SimpleNamespace(CPU="fake-cpu-activity"),
                profile=profile,
            ),
        )
        return fake_torch, calls

    memory_source = b"fixture memory profiler source\n"
    profiler_source = b"fixture profiler source\n"
    torch_root = tmp_path / "torch"
    profiler_root = torch_root / "profiler"
    profiler_root.mkdir(parents=True)
    torch_init = torch_root / "__init__.py"
    torch_init.write_bytes(b"fixture torch package\n")
    (profiler_root / "_memory_profiler.py").write_bytes(memory_source)
    (profiler_root / "profiler.py").write_bytes(profiler_source)
    monkeypatch.setattr(
        h8_child,
        "_PROFILER_MEMORY_SOURCE_SHA256",
        hashlib.sha256(memory_source).hexdigest(),
    )
    monkeypatch.setattr(
        h8_child,
        "_PROFILER_SOURCE_SHA256",
        hashlib.sha256(profiler_source).hexdigest(),
    )
    monkeypatch.setattr(
        h8_child,
        "_operation_graph",
        lambda *_args, **_kwargs: pytest.fail(
            "schema inspection reached the H8 production operation graph"
        ),
    )

    fake_torch, calls = make_fake_torch()
    inspection = h8_child.inspect_installed_h8_profiler_schema(fake_torch)

    assert dataclasses.asdict(inspection) == {
        "schema_version": "h8-installed-profiler-schema-inspection-v1",
        "torch_version": h8_wire.H8_PROFILER_TORCH_VERSION,
        "memory_profile_source_sha256": hashlib.sha256(memory_source).hexdigest(),
        "profiler_source_sha256": hashlib.sha256(profiler_source).hexdigest(),
        "api_contract_sha256": h8_wire.H8_PROFILER_API_CONTRACT_SHA256,
        "profiler_flags": (
            ("activities", ("CPU",)),
            ("profile_memory", True),
            ("record_shapes", True),
            ("with_stack", True),
        ),
        "action_enum": (
            "PREEXISTING",
            "CREATE",
            "INCREMENT_VERSION",
            "DESTROY",
        ),
        "observed_actions": ("CREATE", "INCREMENT_VERSION"),
        "required_event_types": ("Allocation", "TorchOp"),
        "timeline_row_count": 2,
        "event_node_count": 2,
        "decoded_tensor_key_count": 3,
    }
    with pytest.raises(dataclasses.FrozenInstanceError):
        inspection.timeline_row_count = 0
    assert not hasattr(inspection, "status")
    assert calls == [
        "no_grad",
        ("no_grad", "enter"),
        (
            "profile",
            {
                "activities": ["fake-cpu-activity"],
                "profile_memory": True,
                "record_shapes": True,
                "with_stack": True,
            },
        ),
        ("profile", "enter"),
        ("empty", (2,), "fake-float64", "cpu"),
        "clone",
        ("add_", 1.0),
        ("profile", "exit", None),
        ("no_grad", "exit", None),
        "_memory_profile",
        "experimental_event_tree",
    ]

    incomplete_action = Enum(
        "_Action",
        ("PREEXISTING", "CREATE", "INCREMENT_VERSION"),
    )
    rejection_cases = (
        ({"malformed_row": True}, "exact four-tuple"),
        ({"action_type": incomplete_action}, "complete four-action enum"),
        ({"complete_key": False}, "TensorKey"),
        ({"include_torch_op": False}, "Allocation/TorchOp"),
    )
    for overrides, message in rejection_cases:
        drifted_torch, _calls = make_fake_torch(**overrides)
        with pytest.raises(h8_child._ProfilerUnavailable, match=message):
            h8_child.inspect_installed_h8_profiler_schema(drifted_torch)


def test_h8_child_v2_retains_lossless_private_evidence_without_public_schema_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vfe4.types as public_types
    import vfe4.types.h8 as h8_types
    from verification.h8_gate import _attempt_payload, _child_payload
    from vfe4.types.h8 import (
        H8LocalSPDDiagnostics,
        H8ProductionProblemEvidence,
        H8TransitionNorms,
        SparseConditionDiagnostics,
    )

    result_keys = (
        "input_sha256",
        "sample_noise_sha256",
        "problem_evidence",
        "objective",
        "storage",
        "fill",
        "workspace",
        "counters",
        "allocation",
        "resources",
        "diagnostics",
        "operation_reachability",
        "residuals",
        "resource_decisions",
        "invariants",
    )
    production_envelope = _child_envelope()
    profiler_envelope = _profiler_child_envelope()
    for envelope in (production_envelope, profiler_envelope):
        raw_result = envelope["result"]
        assert isinstance(raw_result, dict)
        assert tuple(raw_result) == result_keys

    production_attempt, trusted_production = _attempt_from_fake_envelope(
        tmp_path,
        production_envelope,
    )
    profiler_attempt, _trusted_profiler = _attempt_from_fake_envelope(
        tmp_path,
        profiler_envelope,
    )
    control_attempt, _trusted_control = _attempt_from_fake_envelope(
        tmp_path,
        _control_child_envelope(),
    )
    nonpass_attempts = tuple(
        _attempt_from_fake_envelope(
            tmp_path,
            _child_envelope(
                status=status,
                obligations=(
                    ["fake_observability_gap"]
                    if status == "inconclusive"
                    else []
                ),
                result=None,
                error={
                    "kind": "fake_observability_gap",
                    "message": "fake evidence only",
                    "witnessed_violation": status == "fail",
                },
            ),
        )[0]
        for status in ("inconclusive", "fail")
    )

    assert hasattr(h8_types, "H8DecodedPassEvidence")
    private_type = h8_types.H8DecodedPassEvidence
    assert tuple(
        field.name for field in dataclasses.fields(private_type)
    ) == (
        "sample_noise_sha256",
        "problem_evidence",
        "condition_diagnostics",
        "allocation",
        "child_identities",
    )
    assert "H8DecodedPassEvidence" not in h8_types.__all__
    assert not hasattr(public_types, "H8DecodedPassEvidence")
    for attempt in (production_attempt, profiler_attempt):
        evidence = attempt.pass_evidence
        assert type(evidence) is private_type
        assert evidence.sample_noise_sha256 == "2" * 64
        assert type(evidence.problem_evidence) is H8ProductionProblemEvidence
        assert (
            type(evidence.problem_evidence.local_spd_diagnostics)
            is H8LocalSPDDiagnostics
        )
        assert (
            type(evidence.problem_evidence.transition_norms)
            is H8TransitionNorms
        )
        assert type(evidence.condition_diagnostics) is (
            SparseConditionDiagnostics
        )
        assert isinstance(evidence.allocation, MappingProxyType)
        assert isinstance(evidence.child_identities, MappingProxyType)
        assert tuple(evidence.child_identities) == (
            "hardware",
            "affinity",
            "thread",
            "blas",
        )
        with pytest.raises(TypeError, match="factory-only"):
            dataclasses.replace(attempt, pass_evidence=None)
    assert control_attempt.pass_evidence is None
    assert all(attempt.pass_evidence is None for attempt in nonpass_attempts)
    with pytest.raises(TypeError, match="factory-only"):
        dataclasses.replace(
            control_attempt,
            pass_evidence=production_attempt.pass_evidence,
        )

    retained = production_attempt.pass_evidence
    assert retained is not None
    raw_result = trusted_production["result"]
    raw_identities = trusted_production["identities"]
    assert isinstance(raw_result, dict)
    raw_allocation = raw_result["allocation"]
    assert isinstance(raw_allocation, dict)
    raw_channels = raw_allocation["observed_channels"]
    assert isinstance(raw_channels, list)
    raw_channels.append("forged")
    assert isinstance(raw_identities, dict)
    raw_hardware = raw_identities["hardware"]
    assert isinstance(raw_hardware, dict)
    raw_hardware["platform"] = "forged"
    assert retained.allocation["observed_channels"] == (
        "dispatch",
        "numpy_guard",
        "backend",
        "os_hwm",
    )
    frozen_hardware = retained.child_identities["hardware"]
    assert isinstance(frozen_hardware, Mapping)
    assert frozen_hardware["platform"] == "test"
    with pytest.raises(TypeError):
        retained.allocation["forged"] = True  # type: ignore[index]
    with pytest.raises(TypeError):
        frozen_hardware["platform"] = "forged"  # type: ignore[index]

    assert tuple(
        field.name for field in dataclasses.fields(H8ChildResult)
    ) == (
        "mode",
        "seed",
        "repetition",
        "input_sha256",
        "objective",
        "storage",
        "fill",
        "workspace",
        "counters",
        "allocation",
        "resources",
        "invariants",
    )
    attempt_keys = (
        "request",
        "status",
        "reasons",
        "result_kind",
        "result_identity",
        "nonpass_envelope",
        "timed_out",
        "exit_code",
        "parent_elapsed_ns",
        "request_sha256",
        "identities_sha256",
        "stdout_sha256",
        "stderr_sha256",
        "operation_reachability",
        "residuals",
        "resource_decisions",
    )
    assert tuple(_attempt_payload(production_attempt)) == attempt_keys
    assert "pass_evidence" not in _attempt_payload(production_attempt)
    published_keys = (
        "mode",
        "seed",
        "repetition",
        "input_sha256",
        "objective",
        "storage",
        "fill",
        "workspace",
        "counters",
        "allocation",
        "resources",
        "invariants",
        "parent_elapsed_ns",
        "child_elapsed_ns",
        "exit_code",
        "stdout_sha256",
        "stderr_sha256",
        "operation_reachability",
        "residuals",
        "resource_decisions",
    )
    for attempt in (production_attempt, profiler_attempt):
        assert type(attempt.result) is H8ChildResult
        assert tuple(_child_payload(attempt.result, attempt)) == published_keys

    with pytest.raises(TypeError, match="factory-only"):
        private_type(
            sample_noise_sha256=retained.sample_noise_sha256,
            problem_evidence=retained.problem_evidence,
            condition_diagnostics=retained.condition_diagnostics,
            allocation=retained.allocation,
            child_identities=retained.child_identities,
        )
    with pytest.raises(TypeError, match="factory-only"):
        H8ChildAttemptRecord(
            **{
                field.name: getattr(production_attempt, field.name)
                for field in dataclasses.fields(production_attempt)
            }
        )
    forged_attempt = object.__new__(H8ChildAttemptRecord)
    for field in dataclasses.fields(production_attempt):
        object.__setattr__(
            forged_attempt,
            field.name,
            getattr(production_attempt, field.name),
        )
    with pytest.raises(ValueError, match="factory-issued"):
        h8_types._require_h8_child_attempt_record(forged_attempt)

    assert type(production_attempt.result) is H8ChildResult
    retained_input_sha256 = production_attempt.result.input_sha256
    object.__setattr__(
        production_attempt.result,
        "input_sha256",
        "0" * 64,
    )
    with pytest.raises(ValueError, match="factory-issued"):
        h8_types._require_h8_child_attempt_record(production_attempt)
    object.__setattr__(
        production_attempt.result,
        "input_sha256",
        retained_input_sha256,
    )
    assert h8_types._require_h8_child_attempt_record(
        production_attempt
    ) is production_attempt

    retained_generative_sha256 = retained.problem_evidence.generative_sha256
    object.__setattr__(
        retained.problem_evidence,
        "generative_sha256",
        "0" * 64,
    )
    with pytest.raises(ValueError, match="factory-issued"):
        h8_types._require_h8_child_attempt_record(production_attempt)
    object.__setattr__(
        retained.problem_evidence,
        "generative_sha256",
        retained_generative_sha256,
    )
    assert h8_types._require_h8_decoded_pass_evidence(retained) is retained

    for exit_code, parent_elapsed_ns, expected_reason in (
        (9, 37, "nonzero_child_exit"),
        (0, 60_000_000_001, "parent_elapsed_budget_breach"),
    ):
        request, invocation, _, _, payload = (
            _verified_h8_production_attempt_inputs(tmp_path)
        )
        process_record = H8ChildProcessRecord.from_payload(
            payload,
            exit_code=exit_code,
            parent_elapsed_ns=parent_elapsed_ns,
        )
        decision = classify_h8_child_outcome(
            process_record,
            valid_start=True,
            invocation=invocation,
        )
        attempt = make_h8_child_attempt_record(
            request,
            invocation,
            process_record,
            decision,
        )
        assert attempt.status is GateStatus.FAIL
        assert expected_reason in attempt.reasons
        assert attempt.result is None
        assert attempt.pass_evidence is None
        assert attempt.operation_reachability is None
        assert attempt.residuals is None
        assert attempt.resource_decisions is None
        assert attempt.nonpass_envelope is not None
        assert attempt.nonpass_envelope["status"] == "pass"

    from vfe4.types.h8 import (
        H8_PROFILER_API_CONTRACT_SHA256,
        H8_PROFILER_MEMORY_SOURCE_SHA256,
        H8_PROFILER_SOURCE_SHA256,
        H8_PROFILER_TORCH_VERSION,
    )

    profiler_pins = {
        "torch_version": H8_PROFILER_TORCH_VERSION,
        "memory_profile_source_sha256": H8_PROFILER_MEMORY_SOURCE_SHA256,
        "profiler_source_sha256": H8_PROFILER_SOURCE_SHA256,
        "api_contract_sha256": H8_PROFILER_API_CONTRACT_SHA256,
    }
    for name, expected in profiler_pins.items():
        drifted = _profiler_child_envelope()
        drifted_result = drifted["result"]
        assert isinstance(drifted_result, dict)
        drifted_allocation = drifted_result["allocation"]
        assert isinstance(drifted_allocation, dict)
        drifted_api = drifted_allocation["profiler_api"]
        assert isinstance(drifted_api, dict)
        assert drifted_api[name] == expected
        drifted_api[name] = (
            "forged-torch-version"
            if name == "torch_version"
            else "0" * 64
        )
        with pytest.raises(ValueError, match="profiler"):
            decode_h8_child_result(drifted)

    import verification.h8_budget as h8_budget
    from verification.h8_orchestrator import (
        H8IssuedLaunchRecord,
        build_h8_launch_contract_sha256,
    )

    type_only_attempt = h8_types._issue_h8_child_attempt_record(
        **{
            field.name: getattr(production_attempt, field.name)
            for field in dataclasses.fields(production_attempt)
        }
    )
    with pytest.raises(ValueError, match="validated process bundle"):
        h8_budget._require_h8_budget_issued_attempt(type_only_attempt)
    assert h8_budget._require_h8_budget_issued_attempt(
        production_attempt
    ) is production_attempt

    validation_counts = {
        "allocation": 0,
        "relationships": 0,
        "profiler_events": 0,
    }
    original_allocation = h8_budget._validate_allocation_evidence
    original_relationships = h8_budget._validate_pass_relationships
    original_profiler_events = h8_budget._decode_profiler_events

    def counted_allocation(*args, **kwargs):
        validation_counts["allocation"] += 1
        return original_allocation(*args, **kwargs)

    def counted_relationships(*args, **kwargs):
        validation_counts["relationships"] += 1
        return original_relationships(*args, **kwargs)

    def counted_profiler_events(*args, **kwargs):
        validation_counts["profiler_events"] += 1
        return original_profiler_events(*args, **kwargs)

    monkeypatch.setattr(
        h8_budget,
        "_validate_allocation_evidence",
        counted_allocation,
    )
    monkeypatch.setattr(
        h8_budget,
        "_validate_pass_relationships",
        counted_relationships,
    )
    monkeypatch.setattr(
        h8_budget,
        "_decode_profiler_events",
        counted_profiler_events,
    )
    single_pass_envelope = _profiler_child_envelope()
    single_pass_request = H8ChildRequest(
        mode="profiler",
        seed=single_pass_envelope["seed"],  # type: ignore[arg-type]
        repetition=None,
        config_sha256=single_pass_envelope["config_sha256"],  # type: ignore[arg-type]
        protocol_sha256=single_pass_envelope["protocol_sha256"],  # type: ignore[arg-type]
        control_id=None,
    )
    single_pass_identities = single_pass_envelope["identities"]
    assert isinstance(single_pass_identities, dict)
    single_pass_invocation = build_h8_child_invocation(
        dataclasses.asdict(single_pass_request),
        repository_root=tmp_path,
        identities=single_pass_identities,
        base_environment={},
    )
    single_pass_envelope["request_sha256"] = hashlib.sha256(
        single_pass_invocation.stdin[:-1]
    ).hexdigest()
    single_pass_process = H8ChildProcessRecord.from_payload(
        single_pass_envelope
    )
    single_pass_decision = classify_h8_child_outcome(
        single_pass_process,
        valid_start=True,
        invocation=single_pass_invocation,
    )
    single_pass_attempt = make_h8_child_attempt_record(
        single_pass_request,
        single_pass_invocation,
        single_pass_process,
        single_pass_decision,
    )
    issued = H8IssuedLaunchRecord(
        request=single_pass_request,
        invocation=single_pass_invocation,
        process_record=single_pass_process,
        repository_root=tmp_path.resolve(),
        launch_contract_sha256=build_h8_launch_contract_sha256(
            single_pass_invocation,
            repository_root=tmp_path,
        ),
        attempt=single_pass_attempt,
    )
    issued.__post_init__()
    assert h8_types._require_h8_child_attempt_record(
        single_pass_attempt
    ) is single_pass_attempt
    assert validation_counts == {
        "allocation": 1,
        "relationships": 1,
        "profiler_events": 1,
    }


def test_child_attempt_factory_decodes_and_binds_verified_pass(
    tmp_path: Path,
) -> None:
    (
        request,
        invocation,
        process_record,
        decision,
        payload,
    ) = _verified_h8_production_attempt_inputs(tmp_path)

    attempt = make_h8_child_attempt_record(
        request,
        invocation,
        process_record,
        decision,
    )

    assert type(attempt) is H8ChildAttemptRecord
    assert attempt.request is request
    assert attempt.status is GateStatus.PASS
    assert attempt.reasons == ()
    assert type(attempt.result) is H8ChildResult
    assert attempt.timed_out is False
    assert attempt.exit_code == 0
    assert attempt.parent_elapsed_ns == 37
    assert attempt.request_sha256 == hashlib.sha256(
        invocation.stdin[:-1]
    ).hexdigest()
    identity_json = invocation.environment[H8_CHILD_IDENTITY_ENV]
    assert attempt.identities_sha256 == hashlib.sha256(
        identity_json.encode("ascii")
    ).hexdigest()
    assert attempt.stdout_sha256 == hashlib.sha256(
        process_record.stdout
    ).hexdigest()
    assert attempt.stderr_sha256 == hashlib.sha256(
        process_record.stderr
    ).hexdigest()
    result_payload = payload["result"]
    assert isinstance(result_payload, dict)
    assert dict(attempt.operation_reachability or {}) == result_payload[
        "operation_reachability"
    ]
    assert dict(attempt.residuals or {}) == result_payload["residuals"]
    result_decisions = result_payload["resource_decisions"]
    assert isinstance(result_decisions, dict)
    assert attempt.resource_decisions is not None
    assert set(attempt.resource_decisions) == set(result_decisions)
    assert attempt.resource_decisions["time_pass"] is True
    assert attempt.resource_decisions[
        "conservative_incremental_hwm_bytes"
    ] == 30
    assert attempt.result.resources.parent_elapsed_ns == 0


def test_child_attempt_factory_retains_owned_immutable_result_metadata(
    tmp_path: Path,
) -> None:
    (
        request,
        invocation,
        process_record,
        decision,
        _,
    ) = _verified_h8_production_attempt_inputs(tmp_path)
    attempt = make_h8_child_attempt_record(
        request,
        invocation,
        process_record,
        decision,
    )
    decision_payload = decision.payload
    assert isinstance(decision_payload, dict)
    result_payload = decision_payload["result"]
    assert isinstance(result_payload, dict)
    reachability = result_payload["operation_reachability"]
    residuals = result_payload["residuals"]
    decisions = result_payload["resource_decisions"]
    assert isinstance(reachability, dict)
    assert isinstance(residuals, dict)
    assert isinstance(decisions, dict)

    reachability["factorization"] = False
    residuals["solve"] = 1.0
    decisions["time_pass"] = False
    allowances = decisions["residual_allowances"]
    assert isinstance(allowances, dict)
    solve_allowance = allowances["solve"]
    assert isinstance(solve_allowance, dict)
    solve_allowance["passed"] = False

    assert attempt.operation_reachability is not None
    assert attempt.operation_reachability["factorization"] is True
    assert attempt.residuals is not None
    assert attempt.residuals["solve"] == 0.0
    assert attempt.resource_decisions is not None
    assert attempt.resource_decisions["time_pass"] is True
    frozen_allowances = attempt.resource_decisions["residual_allowances"]
    assert isinstance(frozen_allowances, Mapping)
    frozen_solve_allowance = frozen_allowances["solve"]
    assert isinstance(frozen_solve_allowance, Mapping)
    assert frozen_solve_allowance["passed"] is True
    with pytest.raises(TypeError):
        attempt.operation_reachability["factorization"] = False  # type: ignore[index]
    with pytest.raises(TypeError):
        attempt.residuals["solve"] = 1.0  # type: ignore[index]
    with pytest.raises(TypeError):
        attempt.resource_decisions["time_pass"] = False  # type: ignore[index]
    with pytest.raises(TypeError):
        frozen_solve_allowance["passed"] = False  # type: ignore[index]


def test_child_attempt_factory_preserves_timeout_without_result(
    tmp_path: Path,
) -> None:
    request, invocation, _, _, _ = _verified_h8_production_attempt_inputs(
        tmp_path
    )
    process_record = H8ChildProcessRecord(
        timed_out=True,
        exit_code=None,
        stdout=b"",
        stderr=b"deadline exceeded",
        parent_elapsed_ns=60_000_000_001,
    )
    decision = classify_h8_child_outcome(
        process_record,
        valid_start=True,
        invocation=invocation,
    )

    attempt = make_h8_child_attempt_record(
        request,
        invocation,
        process_record,
        decision,
    )

    assert attempt.status is GateStatus.FAIL
    assert attempt.reasons == ("child_timeout",)
    assert attempt.timed_out is True
    assert attempt.exit_code is None
    assert attempt.result is None
    assert attempt.operation_reachability is None
    assert attempt.residuals is None
    assert attempt.resource_decisions is None


def test_child_attempt_factory_preserves_nonzero_exit_without_result(
    tmp_path: Path,
) -> None:
    request, invocation, _, _, _ = _verified_h8_production_attempt_inputs(
        tmp_path
    )
    process_record = H8ChildProcessRecord(
        timed_out=False,
        exit_code=9,
        stdout=b"",
        stderr=b"child failed before emitting an envelope",
        parent_elapsed_ns=11,
    )
    decision = classify_h8_child_outcome(
        process_record,
        valid_start=True,
        invocation=invocation,
    )

    attempt = make_h8_child_attempt_record(
        request,
        invocation,
        process_record,
        decision,
    )

    assert attempt.status is GateStatus.FAIL
    assert attempt.reasons == (
        "nonzero_child_exit",
        "invalid_child_stdout",
    )
    assert attempt.timed_out is False
    assert attempt.exit_code == 9
    assert attempt.result is None
    assert attempt.operation_reachability is None
    assert attempt.residuals is None
    assert attempt.resource_decisions is None


def test_child_attempt_factory_rejects_request_or_identity_drift(
    tmp_path: Path,
) -> None:
    (
        request,
        invocation,
        process_record,
        decision,
        _,
    ) = _verified_h8_production_attempt_inputs(tmp_path)

    with pytest.raises(ValueError, match="request.*invocation"):
        make_h8_child_attempt_record(
            dataclasses.replace(request, seed=request.seed + 1),
            invocation,
            process_record,
            decision,
        )

    identity_json = invocation.environment[H8_CHILD_IDENTITY_ENV]
    noncanonical_environment = dict(invocation.environment)
    noncanonical_environment[H8_CHILD_IDENTITY_ENV] = json.dumps(
        json.loads(identity_json),
        indent=2,
    )
    noncanonical_invocation = dataclasses.replace(
        invocation,
        environment=noncanonical_environment,
    )
    with pytest.raises(ValueError, match="identities.*canonical"):
        make_h8_child_attempt_record(
            request,
            noncanonical_invocation,
            process_record,
            decision,
        )


def test_child_attempt_factory_retains_witnessed_identity_mismatch(
    tmp_path: Path,
) -> None:
    request, invocation, _, _, payload = (
        _verified_h8_production_attempt_inputs(tmp_path)
    )
    payload["config_sha256"] = "d" * 64
    process_record = H8ChildProcessRecord.from_payload(payload)
    decision = classify_h8_child_outcome(
        process_record,
        valid_start=True,
        invocation=invocation,
    )

    attempt = make_h8_child_attempt_record(
        request,
        invocation,
        process_record,
        decision,
    )

    assert attempt.status is GateStatus.FAIL
    assert "child_request_or_environment_identity_mismatch" in attempt.reasons
    assert attempt.result is None
    assert attempt.nonpass_envelope is not None
    assert attempt.nonpass_envelope["config_sha256"] == "d" * 64
    assert attempt.operation_reachability is None
    assert attempt.residuals is None
    assert attempt.resource_decisions is None


def test_child_attempt_factory_keeps_identity_observability_inconclusive(
    tmp_path: Path,
) -> None:
    request, invocation, _, _, payload = (
        _verified_h8_production_attempt_inputs(tmp_path)
    )
    payload["status"] = "inconclusive"
    payload["obligations"] = ["environment_observability_gap"]
    payload["result"] = None
    payload["error"] = {
        "kind": "environment_observability_gap",
        "message": "runtime identity unavailable",
        "witnessed_violation": False,
    }
    payload["identities"] = {
        name: make_h8_identity_record(
            name,
            {"observability_error": "runtime identity unavailable"},
        )
        for name in ("hardware", "affinity", "thread", "blas")
    }
    process_record = H8ChildProcessRecord.from_payload(payload)
    decision = classify_h8_child_outcome(
        process_record,
        valid_start=True,
        invocation=invocation,
    )

    attempt = make_h8_child_attempt_record(
        request,
        invocation,
        process_record,
        decision,
    )

    assert decision.status is GateStatus.INCONCLUSIVE
    assert "child_request_or_environment_identity_mismatch" not in (
        decision.reasons
    )
    assert attempt.status is GateStatus.INCONCLUSIVE
    assert attempt.result is None
    assert attempt.nonpass_envelope is not None
    assert attempt.operation_reachability is None
    assert attempt.residuals is None
    assert attempt.resource_decisions is None


def test_child_attempt_factory_retains_timeout_envelope_without_trusting_it(
    tmp_path: Path,
) -> None:
    request, invocation, _, _, payload = (
        _verified_h8_production_attempt_inputs(tmp_path)
    )
    stdout = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    process_record = H8ChildProcessRecord(
        timed_out=True,
        exit_code=None,
        stdout=stdout,
        stderr=b"deadline exceeded",
        parent_elapsed_ns=60_000_000_001,
    )
    decision = classify_h8_child_outcome(
        process_record,
        valid_start=True,
        invocation=invocation,
    )

    attempt = make_h8_child_attempt_record(
        request,
        invocation,
        process_record,
        decision,
    )

    assert attempt.status is GateStatus.FAIL
    assert "child_timeout" in attempt.reasons
    assert attempt.result is None
    assert attempt.nonpass_envelope is not None
    assert attempt.nonpass_envelope["status"] == "pass"
    assert attempt.operation_reachability is None
    assert attempt.residuals is None
    assert attempt.resource_decisions is None


@pytest.mark.parametrize(
    "witness_kind",
    ("operation_reachability", "resource_decision"),
)
def test_child_attempt_factory_promotes_retained_witness_to_fail(
    tmp_path: Path,
    witness_kind: str,
) -> None:
    request, invocation, _, _, payload = (
        _verified_h8_production_attempt_inputs(tmp_path)
    )
    payload["status"] = "inconclusive"
    payload["obligations"] = ["partial_child_evidence"]
    result_payload = payload["result"]
    assert isinstance(result_payload, dict)
    if witness_kind == "operation_reachability":
        reachability = result_payload["operation_reachability"]
        assert isinstance(reachability, dict)
        reachability["factorization"] = False
    else:
        decisions = result_payload["resource_decisions"]
        assert isinstance(decisions, dict)
        decisions["time_pass"] = False
    process_record = H8ChildProcessRecord.from_payload(payload)
    decision = classify_h8_child_outcome(
        process_record,
        valid_start=True,
        invocation=invocation,
    )

    attempt = make_h8_child_attempt_record(
        request,
        invocation,
        process_record,
        decision,
    )

    assert decision.status is GateStatus.FAIL
    assert attempt.status is GateStatus.FAIL
    assert attempt.nonpass_envelope is not None


@pytest.mark.parametrize(
    "witness_kind",
    ("child_elapsed", "residual_decision"),
)
def test_child_attempt_factory_recovers_raw_witness_from_invalid_pass_envelope(
    tmp_path: Path,
    witness_kind: str,
) -> None:
    request, invocation, _, _, payload = (
        _verified_h8_production_attempt_inputs(tmp_path)
    )
    result_payload = payload["result"]
    assert isinstance(result_payload, dict)
    if witness_kind == "child_elapsed":
        resources = result_payload["resources"]
        assert isinstance(resources, dict)
        resources["child_elapsed_ns"] = 60_000_000_001
    else:
        decisions = result_payload["resource_decisions"]
        assert isinstance(decisions, dict)
        allowance_groups = decisions["residual_allowances"]
        assert isinstance(allowance_groups, dict)
        solve_group = allowance_groups["solve"]
        assert isinstance(solve_group, dict)
        comparisons = solve_group["comparisons"]
        assert isinstance(comparisons, list)
        comparison = comparisons[0]
        assert isinstance(comparison, dict)
        allowance = comparison["allowance"]
        assert isinstance(allowance, dict)
        threshold = allowance["allowance"]
        assert type(threshold) is float
        allowance["residual"] = threshold + 1.0

    process_record = H8ChildProcessRecord.from_payload(payload)
    decision = classify_h8_child_outcome(
        process_record,
        valid_start=True,
        invocation=invocation,
    )

    assert decision.status is GateStatus.FAIL
    assert (
        "invalid_stdout_retains_witnessed_resource_failure"
        in decision.reasons
    )
    assert decision.payload is None

    attempt = make_h8_child_attempt_record(
        request,
        invocation,
        process_record,
        decision,
    )

    assert attempt.status is GateStatus.FAIL
    assert attempt.result is None
    assert attempt.nonpass_envelope is None
    assert attempt.stdout_sha256 == hashlib.sha256(
        process_record.stdout
    ).hexdigest()


@pytest.mark.parametrize(
    ("child_status", "obligations", "expected_status"),
    [
        ("fail", [], GateStatus.FAIL),
        (
            "inconclusive",
            ["partial_child_evidence"],
            GateStatus.INCONCLUSIVE,
        ),
    ],
)
def test_child_attempt_factory_retains_parseable_nonpass_partial_envelope(
    tmp_path: Path,
    child_status: str,
    obligations: list[str],
    expected_status: GateStatus,
) -> None:
    request, invocation, _, _, payload = (
        _verified_h8_production_attempt_inputs(tmp_path)
    )
    payload["status"] = child_status
    payload["obligations"] = obligations
    result_payload = payload["result"]
    assert isinstance(result_payload, dict)
    result_payload["operation_reachability"] = None
    result_payload["residuals"] = None
    result_payload["resource_decisions"] = {
        "partial_endpoint_retained": True,
    }
    process_record = H8ChildProcessRecord.from_payload(payload)
    decision = classify_h8_child_outcome(
        process_record,
        valid_start=True,
        invocation=invocation,
    )

    attempt = make_h8_child_attempt_record(
        request,
        invocation,
        process_record,
        decision,
    )

    assert attempt.status is expected_status
    assert attempt.result is None
    assert attempt.nonpass_envelope is not None
    assert attempt.nonpass_envelope["status"] == child_status
    retained_result = attempt.nonpass_envelope["result"]
    assert isinstance(retained_result, Mapping)
    assert retained_result["operation_reachability"] is None
    assert retained_result["residuals"] is None
    with pytest.raises(TypeError):
        retained_result["residuals"] = {}  # type: ignore[index]
    assert attempt.operation_reachability is None
    assert attempt.residuals is None
    assert attempt.resource_decisions == {
        "partial_endpoint_retained": True,
    }


def test_child_attempt_factory_rejects_decision_process_drift(
    tmp_path: Path,
) -> None:
    (
        request,
        invocation,
        process_record,
        decision,
        _,
    ) = _verified_h8_production_attempt_inputs(tmp_path)
    mismatched_decisions = (
        dataclasses.replace(decision, timed_out=True),
        dataclasses.replace(decision, exit_code=7),
        dataclasses.replace(decision, parent_elapsed_ns=38),
        dataclasses.replace(decision, stdout_sha256="0" * 64),
        dataclasses.replace(decision, stderr_sha256="0" * 64),
        dataclasses.replace(
            decision,
            status=GateStatus.INCONCLUSIVE,
            reasons=("later evidence unavailable",),
        ),
    )

    for mismatched in mismatched_decisions:
        with pytest.raises(ValueError, match="decision"):
            make_h8_child_attempt_record(
                request,
                invocation,
                process_record,
                mismatched,
            )


def test_child_pass_rejects_hollow_nested_evidence() -> None:
    hollow = _child_envelope()
    hollow["result"] = {
        name: {} if name != "invariants" else []
        for name in (
            "input_sha256",
            "sample_noise_sha256",
            "problem_evidence",
            "objective",
            "storage",
            "fill",
            "workspace",
            "counters",
            "allocation",
            "resources",
            "diagnostics",
            "operation_reachability",
            "residuals",
            "resource_decisions",
            "invariants",
        )
    }
    with pytest.raises(ValueError, match="complete nested evidence"):
        parse_h8_child_stdout(
            json.dumps(
                hollow,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )


def test_child_pass_rejects_skeletal_typed_sections_and_control_evidence() -> None:
    skeletal = _child_envelope()
    result = skeletal["result"]
    assert isinstance(result, dict)
    result["objective"] = {"complete_order21": 1.0}
    with pytest.raises(ValueError, match="objective"):
        parse_h8_child_stdout(
            json.dumps(
                skeletal,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )

    spec = next(
        item
        for item in h8_negative_control_specs(_production_policy().layout)
        if item.control_id == "torch_eye_full_rhs"
    )
    evidence = {
        "dispatch": (
            H8DispatchEvent(
                sequence=0,
                operator=spec.requested_operation,
                semantic_site=None,
                control_id=spec.control_id,
                input_shapes=(),
                output_shapes=(spec.logical_shapes[0],),
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
            ),
        ),
        "backend": {
            "before": (),
            "after": (5160,),
            "detected": True,
            "executed_past_detector": False,
            "unexpected_exception": None,
        },
        "operation_returned": False,
        "caught_forbidden": True,
        "pre_execution_detected": True,
        "executed_past_detector": False,
    }
    control = make_h8_control_result(
        spec,
        observed_channels=spec.assigned_channels,
        detected=True,
        event_payload=evidence,
    )
    negative = _child_envelope(
        mode="negative_control",
        repetition=None,
        control_id=spec.control_id,
        result=None,
        control={
            "summary": _fixture_jsonable(control),
            "evidence": _fixture_jsonable(evidence),
        },
    )
    parse_h8_child_stdout(
        json.dumps(
            negative,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    decoded_control = decode_h8_control_result(negative)
    assert type(decoded_control) is H8ControlResult
    assert decoded_control == control
    with pytest.raises(ValueError, match="child result"):
        decode_h8_child_result(negative)
    control_wrapper = negative["control"]
    assert isinstance(control_wrapper, dict)
    control_payload = control_wrapper["summary"]
    assert isinstance(control_payload, dict)
    control_payload["requested_operation"] = "torch.empty"
    with pytest.raises(ValueError, match="control"):
        parse_h8_child_stdout(
            json.dumps(
                negative,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )


def test_completed_production_over_parent_timeout_is_fail(
    tmp_path: Path,
) -> None:
    request = {
        "mode": "production",
        "seed": 20260721,
        "repetition": 0,
        "config_sha256": "b" * 64,
        "protocol_sha256": "c" * 64,
        "control_id": None,
    }
    identities = _child_envelope()["identities"]
    invocation = build_h8_child_invocation(
        request,
        repository_root=tmp_path,
        identities=identities,
        base_environment={},
    )
    payload = _child_envelope(
        request_sha256=hashlib.sha256(invocation.stdin[:-1]).hexdigest(),
        identities=identities,
    )
    record = H8ChildProcessRecord.from_payload(
        payload,
        parent_elapsed_ns=60_000_000_001,
    )
    decision = classify_h8_child_outcome(
        record,
        valid_start=True,
        invocation=invocation,
    )
    assert decision.status is GateStatus.FAIL
    assert "parent_elapsed_budget_breach" in decision.reasons
    assert (
        json.dumps(
            decision.payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
        == record.stdout
    )


def test_finite_residual_above_frozen_allowance_is_fail() -> None:
    false_pass = _child_envelope()
    result = false_pass["result"]
    assert isinstance(result, dict)
    decisions = result["resource_decisions"]
    assert isinstance(decisions, dict)
    allowances = decisions["residual_allowances"]
    assert isinstance(allowances, dict)
    group = allowances["solve"]
    assert isinstance(group, dict)
    comparisons = group["comparisons"]
    assert isinstance(comparisons, list)
    comparison = comparisons[0]
    assert isinstance(comparison, dict)
    allowance = comparison["allowance"]
    assert isinstance(allowance, dict)
    allowance["left_rounding_component"] = 1.0
    with pytest.raises(ValueError, match="allowance"):
        parse_h8_child_stdout(
            json.dumps(
                false_pass,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )


def test_profiler_join_uses_separate_full_source_indexes() -> None:
    key = H8TensorKey(
        tensor_id=7,
        storage_ptr=11,
        allocation_id=13,
        device="cpu",
    )
    wrong_key = dataclasses.replace(key, allocation_id=14)
    indexes = _ProfilerSourceIndexes(
        allocations=(
            _ProfilerAllocationFact(
                node_index=0,
                timestamp_ns=17,
                tensor_key=key,
                nbytes=8,
                operator="aten::empty",
                stack=("source.py:1",),
                action="CREATE",
            ),
            _ProfilerAllocationFact(
                node_index=1,
                timestamp_ns=17,
                tensor_key=wrong_key,
                nbytes=8,
                operator="aten::empty",
                stack=("source.py:1",),
                action="CREATE",
            ),
        ),
        tensors=(
            _ProfilerTensorFact(
                node_index=2,
                tensor_key=key,
                dtype="torch.float64",
                logical_shape=(1,),
                operator="aten::add",
                stack=("source.py:2",),
            ),
        ),
        versions=(
            _ProfilerVersionFact(
                node_index=3,
                timestamp_ns=17,
                action="CREATE",
                tensor_key=key,
                version=0,
                tensor_node_indices=(2,),
            ),
        ),
        storage_sizes=((key, 8),),
    )
    joined = _join_profiler_source_facts(
        indexes,
        action="CREATE",
        timestamp_ns=17,
        tensor_key=key,
        version=0,
        nbytes=8,
    )
    assert joined.tensor_key == key
    assert joined.matched_event_node_indices == (0, 2, 3)


def test_profiler_timestamp_minus_one_is_preexisting_only() -> None:
    key = H8TensorKey(
        tensor_id=7,
        storage_ptr=11,
        allocation_id=13,
        device="cpu",
    )
    row = H8RawProfilerEvent(
        source_row_index=0,
        timestamp_ns=-1,
        action="PREEXISTING",
        tensor_key=key,
        version=0,
        nbytes=8,
    )
    assert row.timestamp_ns == -1
    with pytest.raises(ValueError, match="timestamp"):
        dataclasses.replace(row, action="CREATE")


def test_h8_generator_uses_preflighted_numpy_producers() -> None:
    source = inspect.getsource(make_h8_problem)
    assert "allocation_guard.standard_normal" in source
    for direct_operator in (" * sn(", " / np.sqrt(", " @ "):
        assert direct_operator not in source


def test_windows_hwm_uses_exact_wintypes_and_last_error_contract() -> None:
    source = inspect.getsource(_windows_memory_snapshot)
    assert '("cb", wintypes.DWORD)' in source
    assert "get_process.argtypes = []" in source
    assert "get_process.restype = wintypes.HANDLE" in source
    assert "get_memory.restype = wintypes.BOOL" in source
    assert source.count("ctypes.set_last_error(0)") == 2


def test_torch_eye_full_rhs_child_control_witnesses_both_channels() -> None:
    result = _run_negative_control(
        torch,
        np,
        control_id="torch_eye_full_rhs",
    )
    summary = result["summary"]
    assert summary["status"] == "pass"
    assert summary["observed_channels"] == ["dispatch", "backend"]
    assert result["evidence"]["backend"]["detected"] is True


def test_post_h8_protocol_is_inventory_derived_and_click_run() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    plan = (
        repository_root
        / "docs"
        / "superpowers"
        / "plans"
        / "2026-07-21-vfe4-post-h8-wikitext103-training.md"
    ).read_text(encoding="utf-8")
    amendment = (
        repository_root
        / "docs"
        / "preregistrations"
        / "2026-07-25-post-h8-arm-gate-amendment.md"
    ).read_text(encoding="utf-8")

    arm_ids = (
        "WT103-A0-AR-v1",
        "WT103-A5-PARENT-SPECIFIC-PREFIX-COMPLETE-v1",
        "WT103-A5-FIXED-COMPLETE-v1",
        "WT103-A5-PARENT-SPECIFIC-PREFIX-EMISSION-v1",
        "WT103-A5-NOLATENT-v1",
    )
    for document in (plan, amendment):
        offsets = tuple(document.index(arm_id) for arm_id in arm_ids)
        assert offsets == tuple(sorted(offsets))
        assert "EndpointInventory" in document
        assert "`source_lock|readiness|train|resume`" in document
        assert 'operation="idle"' in document
        assert "generate_vfe4_figures.py" in document

    for stale_literal in (
        "all 16 checkpoints",
        "exact 16 expected endpoint",
        "Freeze all 16 terminal",
        "exactly 16 post-pass",
        "exactly 2,048 A5",
        "source_lock|train|resume|figures",
    ):
        assert stale_literal not in plan

    role_rows = (
        "| `WT103-A0-AR-v1` | `build_wt103_a0@wt103-arm-v1` "
        "| `cross_entropy` | `absent` | false / false "
        "| `exact_autoregressive` | `PRIMARY_REFERENCE` |",
        "| `WT103-A5-PARENT-SPECIFIC-PREFIX-COMPLETE-v1` "
        "| `build_wt103_a5_parent_specific@wt103-arm-v1` "
        "| `complete_elbo` | `parent_specific_pooled_prefix` "
        "| true / true | `weighted_smc` | `PRIMARY_ENDPOINT` |",
        "| `WT103-A5-FIXED-COMPLETE-v1` "
        "| `build_wt103_a5_fixed@wt103-arm-v1` | `complete_elbo` "
        "| `fixed` | true / true | `weighted_smc` | `PRIOR_CONTROL` |",
        "| `WT103-A5-PARENT-SPECIFIC-PREFIX-EMISSION-v1` "
        "| `build_wt103_a5_parent_specific@wt103-arm-v1` "
        "| `emission_only_ablation_non_elbo` "
        "| `parent_specific_pooled_prefix` | true / true "
        "| `weighted_smc` | `OBJECTIVE_GATE` |",
        "| `WT103-A5-NOLATENT-v1` "
        "| `build_wt103_a5_nolatent@wt103-arm-v1` | `cross_entropy` "
        "| `absent` | false / false | `exact_autoregressive` "
        "| `LATENT_PATH_CONTROL` |",
    )
    for row in role_rows:
        assert row in plan
        assert row in amendment

    gate_order = "\n".join(
        (
            "SOURCE_LOCK",
            "H8_EXACT_REVISION",
            "POST_H8_READINESS",
            "OBJECTIVE",
            "PRIMARY",
            "PRIOR_CONTROL",
            "LATENT_PATH_CONTROL",
        )
    )
    assert gate_order in amendment
    assert "`OBJECTIVE` must precede and be a prerequisite of `PRIMARY`" in plan

    for required_inventory_field in (
        "tuning_attempt_keys:",
        "terminal_checkpoint_keys:",
        "validation_endpoint_keys:",
        "test_endpoint_keys:",
        "raw_score_record_keys:",
        "result_row_keys:",
        "figure_panel_keys:",
        "figure_series_keys:",
        "endpoint_inventory_sha256:",
    ):
        assert required_inventory_field in plan
    inventory_schema = plan.split("class EndpointInventory:", 1)[1].split(
        "@dataclass", 1
    )[0]
    for forbidden_count_field in (
        "arm_count:",
        "tuning_attempt_count:",
        "terminal_checkpoint_count:",
        "raw_score_record_count:",
        "result_row_count:",
        "figure_panel_count:",
        "figure_series_count:",
    ):
        assert forbidden_count_field not in inventory_schema
    assert "counts are read-only `len(...)` properties" in plan
    assert "No consumer may accept a separately entered arm count" in amendment

    for geometry_literal in (
        "training population dimension D = L*b = 5,120",
        "H8 synthetic population dimension = 5,160",
        "A5 direct source lookback W = 20",
        "A0 direct attention reach = full causal 128",
    ):
        assert geometry_literal in amendment
    assert "No WikiText-103 loader, download, cache creation" in amendment
    assert "until an exact H8 PASS exists for the same implementation revision" in (
        amendment
    )
    assert "No implementation task, data acquisition, source-lock operation" in plan
    assert "Figure generation uses a separate editable dictionary" in amendment


def test_h8_lossless_runtime_views_require_six_way_consensus_and_exact_run_order(
    tmp_path: Path,
) -> None:
    from verification.h8_runtime import (
        build_h8_lossless_runtime_evidence_views,
    )
    from verification.h8_wire import (
        H8_COLD_REPETITIONS,
        H8_PRODUCTION_SEEDS,
    )

    problem_keys = (
        "problem_seed",
        "sample_noise_seed",
        "input_sha256",
        "sample_noise_sha256",
        "generative_sha256",
        "recognition_sha256",
        "local_spd_diagnostics",
        "transition_norms",
        "observation_sha256",
    )
    factor_keys = (
        "mode",
        "seed",
        "repetition",
        "input_sha256",
        "fill",
        "workspace",
        "condition_diagnostics",
        "counters",
        "reconstruction_invariants",
    )
    allocation_keys = (
        "mode",
        "seed",
        "repetition",
        "input_sha256",
        "allocation",
        "resources",
    )

    def exact_result(attempt: H8ChildAttemptRecord) -> H8ChildResult:
        assert type(attempt.result) is H8ChildResult
        return attempt.result

    production_attempts: list[H8ChildAttemptRecord] = []
    for seed in H8_PRODUCTION_SEEDS:
        for repetition in range(H8_COLD_REPETITIONS):
            envelope = _child_envelope(seed=seed, repetition=repetition)
            if seed == H8_PRODUCTION_SEEDS[0] and repetition == 4:
                result = envelope["result"]
                assert isinstance(result, dict)
                problem = result["problem_evidence"]
                assert isinstance(problem, dict)
                problem["generative_sha256"] = "8" * 64
            attempt, _payload = _attempt_from_fake_envelope(tmp_path, envelope)
            production_attempts.append(attempt)

    profiler_attempts = [
        _attempt_from_fake_envelope(
            tmp_path,
            _profiler_child_envelope(seed=seed),
        )[0]
        for seed in H8_PRODUCTION_SEEDS
    ]
    drifted_attempts = (*production_attempts, *profiler_attempts)
    drifted_production_runs = tuple(
        exact_result(attempt) for attempt in production_attempts
    )
    profiler_runs = tuple(
        exact_result(attempt) for attempt in profiler_attempts
    )
    with pytest.raises(ValueError, match="consensus"):
        build_h8_lossless_runtime_evidence_views(
            child_attempts=drifted_attempts,
            production_runs=drifted_production_runs,
            profiler_runs=profiler_runs,
        )

    replacement, _payload = _attempt_from_fake_envelope(
        tmp_path,
        _child_envelope(
            seed=H8_PRODUCTION_SEEDS[0],
            repetition=4,
        ),
    )
    production_attempts[4] = replacement
    child_attempts = (*production_attempts, *profiler_attempts)
    production_runs = tuple(
        exact_result(attempt) for attempt in production_attempts
    )
    views = build_h8_lossless_runtime_evidence_views(
        child_attempts=child_attempts,
        production_runs=production_runs,
        profiler_runs=profiler_runs,
    )

    expected_run_order = (
        *(
            ("production", seed, repetition)
            for seed in H8_PRODUCTION_SEEDS
            for repetition in range(H8_COLD_REPETITIONS)
        ),
        *(("profiler", seed, None) for seed in H8_PRODUCTION_SEEDS),
    )
    assert tuple(problem["problem_seed"] for problem in views.problems) == (
        H8_PRODUCTION_SEEDS
    )
    assert len(views.problems) == 3
    assert len(views.factor_runs) == 18
    assert len(views.allocation_runs) == 18
    assert all(tuple(problem) == problem_keys for problem in views.problems)
    assert all(tuple(run) == factor_keys for run in views.factor_runs)
    assert all(tuple(run) == allocation_keys for run in views.allocation_runs)
    assert tuple(
        (run["mode"], run["seed"], run["repetition"])
        for run in views.factor_runs
    ) == expected_run_order
    assert tuple(
        (run["mode"], run["seed"], run["repetition"])
        for run in views.allocation_runs
    ) == expected_run_order

    for factor_run, allocation_run, attempt in zip(
        views.factor_runs,
        views.allocation_runs,
        child_attempts,
        strict=True,
    ):
        result = exact_result(attempt)
        evidence = attempt.pass_evidence
        decisions = attempt.resource_decisions
        assert evidence is not None
        assert decisions is not None
        assert _fixture_jsonable(factor_run["fill"]) == _fixture_jsonable(
            result.fill
        )
        assert _fixture_jsonable(factor_run["workspace"]) == _fixture_jsonable(
            result.workspace
        )
        assert _fixture_jsonable(
            factor_run["condition_diagnostics"]
        ) == _fixture_jsonable(evidence.condition_diagnostics)
        assert _fixture_jsonable(factor_run["counters"]) == _fixture_jsonable(
            result.counters
        )
        reconstruction = factor_run["reconstruction_invariants"]
        assert isinstance(reconstruction, Mapping)
        assert tuple(reconstruction) == (
            "residuals",
            "residual_allowances",
        )
        assert reconstruction["residuals"] == attempt.residuals
        assert (
            reconstruction["residual_allowances"]
            == decisions["residual_allowances"]
        )
        assert isinstance(allocation_run["allocation"], Mapping)
        assert tuple(allocation_run["allocation"]) == tuple(evidence.allocation)
        assert _fixture_jsonable(
            allocation_run["allocation"]
        ) == _fixture_jsonable(evidence.allocation)
        assert _fixture_jsonable(
            allocation_run["resources"]
        ) == _fixture_jsonable(result.resources)
        resources = allocation_run["resources"]
        assert isinstance(resources, Mapping)
        assert resources["parent_elapsed_ns"] == 0

    with pytest.raises(TypeError):
        views.problems[0]["problem_seed"] = 0  # type: ignore[index]
    with pytest.raises(TypeError):
        views.allocation_runs[0]["allocation"][
            "dispatch_event_count"
        ] = 0  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        views.factor_runs = ()  # type: ignore[misc]

    swapped_attempts = (
        child_attempts[1],
        child_attempts[0],
        *child_attempts[2:],
    )
    with pytest.raises(ValueError, match="authoritative"):
        build_h8_lossless_runtime_evidence_views(
            child_attempts=swapped_attempts,
            production_runs=production_runs,
            profiler_runs=profiler_runs,
        )


def _h8_v4_fake_evidence(
    tmp_path: Path,
) -> tuple[
    object,
    str,
    tuple[H8ChildAttemptRecord, ...],
    tuple[H8ChildResult, ...],
    tuple[H8ChildResult, ...],
    tuple[H8ControlResult, ...],
    tuple[object, ...],
    tuple[str, ...],
]:
    from vfe4.config.schema import H8ValidationConfig
    from verification.h8_protocol import build_h8_protocol_sha256
    from verification.h8_wire import (
        H8_COLD_REPETITIONS,
        H8_PRODUCTION_SEEDS,
    )

    config = H8ValidationConfig.create()
    protocol_sha256 = build_h8_protocol_sha256(config)

    production_attempts = tuple(
        _attempt_from_fake_envelope(
            tmp_path,
            _child_envelope(
                seed=seed,
                repetition=repetition,
                config_sha256=config.config_sha256,
                protocol_sha256=protocol_sha256,
            ),
        )[0]
        for seed in H8_PRODUCTION_SEEDS
        for repetition in range(H8_COLD_REPETITIONS)
    )
    profiler_attempts = tuple(
        _attempt_from_fake_envelope(
            tmp_path,
            _profiler_child_envelope(
                seed=seed,
                config_sha256=config.config_sha256,
                protocol_sha256=protocol_sha256,
            ),
        )[0]
        for seed in H8_PRODUCTION_SEEDS
    )
    control_attempts = tuple(
        _attempt_from_fake_envelope(
            tmp_path,
            _control_child_envelope(spec)
            | {
                "config_sha256": config.config_sha256,
                "protocol_sha256": protocol_sha256,
            },
        )[0]
        for spec in h8_negative_control_specs(_production_policy().layout)
    )
    child_attempts = (
        *production_attempts,
        *profiler_attempts,
        *control_attempts,
    )
    production_runs = tuple(
        attempt.result
        for attempt in production_attempts
        if type(attempt.result) is H8ChildResult
    )
    profiler_runs = tuple(
        attempt.result
        for attempt in profiler_attempts
        if type(attempt.result) is H8ChildResult
    )
    controls = tuple(
        attempt.result
        for attempt in control_attempts
        if type(attempt.result) is H8ControlResult
    )

    correctness = make_pass_correctness_cells()
    open_locks = (
        "h8_runtime_sections_not_bound",
        "h8_parent_orchestrator_not_implemented",
    )

    return (
        config,
        protocol_sha256,
        child_attempts,
        production_runs,
        profiler_runs,
        controls,
        correctness,
        open_locks,
    )


def test_h8_v4_runtime_sections_are_owned_lossless_and_evidence_derived(
    tmp_path: Path,
) -> None:
    from verification.h8_gate import assemble_h8_gate_evaluation
    from verification.h8_runtime import build_h8_v4_runtime_sections

    (
        config,
        protocol_sha256,
        child_attempts,
        production_runs,
        profiler_runs,
        controls,
        correctness,
        open_locks,
    ) = _h8_v4_fake_evidence(tmp_path)

    sections = build_h8_v4_runtime_sections(
        config=config,
        candidate_head="1" * 40,
        candidate_dirty_digest="a" * 64,
        candidate_junit_sha256="b" * 64,
        current_refs_registry_sha256="c" * 64,
        dependency_closure_sha256="d" * 64,
        preregistration_sha256="e" * 64,
        prerequisites_current_and_pass=True,
        correctness=correctness,
        child_attempts=child_attempts,
        production_runs=production_runs,
        profiler_runs=profiler_runs,
        controls=controls,
        result_status=GateStatus.INCONCLUSIVE,
        result_obligations=open_locks,
    )

    assert tuple(sections) == (
        "revision",
        "config",
        "interpretation",
        "protocol",
        "environment",
        "problems",
        "storage",
        "factor",
        "allocation",
        "budgets",
        "invariants",
        "artifacts",
    )
    assert sections["revision"]["manuscript_sha256"] == (
        "d733880d3613d32a97b7a12c93ff6c037d0abdfd9ce4810e411769997dbad03c"
    )
    assert sections["config"]["protocol_sha256"] == protocol_sha256
    assert sections["environment"]["platform_release"] == "test-release"
    assert len(sections["problems"]) == 3
    assert tuple(sections["factor"]) == (
        "schema_version",
        "algorithm",
        "pattern",
        "runs",
    )
    assert len(sections["factor"]["runs"]) == 18
    assert tuple(sections["allocation"]) == (
        "schema_version",
        "whitelist",
        "runs",
        "tracemalloc_supplementary",
        "all_observable",
        "no_forbidden_attempts",
    )
    assert len(sections["allocation"]["runs"]) == 18
    assert sections["allocation"]["tracemalloc_supplementary"] is None
    assert sections["storage"] == {
        "h_scalars": 5_160,
        "input_precision_scalars": 411_200,
        "factor_scalars": 411_200,
        "selected_inverse_scalars": 411_200,
        "category_cap_scalars": 411_200,
        "dense_forbidden_scalars": 26_625_600,
        "input_within_cap": True,
        "factor_within_cap": True,
        "selected_within_cap": True,
    }
    assert sections["invariants"]["all_pass"] is True
    assert "runtime_sections" not in inspect.signature(
        assemble_h8_gate_evaluation
    ).parameters

    with pytest.raises(TypeError):
        sections["environment"]["platform_release"] = "forged"  # type: ignore[index]
    with pytest.raises(ValueError, match="exact result objects"):
        build_h8_v4_runtime_sections(
            config=config,
            candidate_head="1" * 40,
            candidate_dirty_digest="a" * 64,
            candidate_junit_sha256="b" * 64,
            current_refs_registry_sha256="c" * 64,
            dependency_closure_sha256="d" * 64,
            preregistration_sha256="e" * 64,
            prerequisites_current_and_pass=True,
            correctness=correctness,
            child_attempts=child_attempts,
            production_runs=(
                dataclasses.replace(production_runs[0]),
                *production_runs[1:],
            ),
            profiler_runs=profiler_runs,
            controls=controls,
            result_status=GateStatus.INCONCLUSIVE,
            result_obligations=open_locks,
        )


def test_h8_v4_runtime_sections_retain_complete_runs_for_control_fail_prefixes(
    tmp_path: Path,
) -> None:
    from verification.h8_gate import (
        _attempts_cross_bound,
        _attempt_ids,
        _exact_attempt_prefix,
        _inventory_complete,
        H8_EXPECTED_CHILD_ATTEMPT_IDS,
    )
    from verification.h8_runtime import build_h8_v4_runtime_sections

    (
        config,
        protocol_sha256,
        child_attempts,
        production_runs,
        profiler_runs,
        controls,
        correctness,
        _open_locks,
    ) = _h8_v4_fake_evidence(tmp_path)
    run_attempts = child_attempts[:18]
    control_attempts = child_attempts[18:]
    specs = h8_negative_control_specs(_production_policy().layout)

    for all_pass_length in (18, 19, 29):
        with pytest.raises(ValueError, match="first witnessed FAIL"):
            build_h8_v4_runtime_sections(
                config=config,
                candidate_head="1" * 40,
                candidate_dirty_digest="a" * 64,
                candidate_junit_sha256="b" * 64,
                current_refs_registry_sha256="c" * 64,
                dependency_closure_sha256="d" * 64,
                preregistration_sha256="e" * 64,
                prerequisites_current_and_pass=True,
                correctness=correctness,
                child_attempts=child_attempts[:all_pass_length],
                production_runs=production_runs,
                profiler_runs=profiler_runs,
                controls=controls[: all_pass_length - len(run_attempts)],
                result_status=GateStatus.INCONCLUSIVE,
                result_obligations=(
                    "h8_runtime_sections_not_bound",
                    "h8_parent_orchestrator_not_implemented",
                ),
            )

    for failure_index in (0, len(controls) // 2, len(controls) - 1):
        spec = specs[failure_index]
        failed_envelope = _control_child_envelope(spec)
        failed_envelope.update(
            {
                "config_sha256": config.config_sha256,
                "protocol_sha256": protocol_sha256,
                "status": "fail",
                "obligations": [],
                "control": None,
                "error": {
                    "kind": "negative_control_executed_past_detector",
                    "message": "test-only witnessed negative-control failure",
                    "witnessed_violation": True,
                },
            }
        )
        failed_attempt, _payload = _attempt_from_fake_envelope(
            tmp_path,
            failed_envelope,
        )
        prefix_attempts = (
            *run_attempts,
            *control_attempts[:failure_index],
            failed_attempt,
        )
        prefix_controls = controls[:failure_index]

        assert _attempt_ids(prefix_attempts) == H8_EXPECTED_CHILD_ATTEMPT_IDS[
            : len(prefix_attempts)
        ]
        assert [attempt.status for attempt in prefix_attempts[:-1]] == [
            GateStatus.PASS
        ] * (len(prefix_attempts) - 1)
        assert prefix_attempts[-1].status is GateStatus.FAIL
        assert _exact_attempt_prefix(prefix_attempts)
        assert _attempts_cross_bound(
            prefix_attempts,
            production_runs,
            profiler_runs,
            prefix_controls,
        )
        assert _inventory_complete(
            correctness,
            prefix_attempts,
            production_runs,
            profiler_runs,
            prefix_controls,
        )
        sections = build_h8_v4_runtime_sections(
            config=config,
            candidate_head="1" * 40,
            candidate_dirty_digest="a" * 64,
            candidate_junit_sha256="b" * 64,
            current_refs_registry_sha256="c" * 64,
            dependency_closure_sha256="d" * 64,
            preregistration_sha256="e" * 64,
            prerequisites_current_and_pass=True,
            correctness=correctness,
            child_attempts=prefix_attempts,
            production_runs=production_runs,
            profiler_runs=profiler_runs,
            controls=prefix_controls,
            result_status=GateStatus.FAIL,
            result_obligations=(),
        )

        assert len(sections["problems"]) == 3
        assert len(sections["factor"]["runs"]) == 18
        assert len(sections["allocation"]["runs"]) == 18
        assert len(prefix_attempts) == 19 + failure_index
        assert sections["invariants"]["controls_complete"] is False
        assert sections["invariants"]["all_pass"] is False
        assert (
            sections["invariants"]["witnessed_failure_dominance_applied"]
            is True
        )
