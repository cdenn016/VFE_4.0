from __future__ import annotations

import base64
import dataclasses
import hashlib
import inspect
import json
import zlib
from pathlib import Path

import numpy as np
import pytest
import torch

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
    H8StorageSpan,
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
    H8_OPERATION_SCOPES,
    H8_SCALE_RESIDUAL_SPECS,
    H8ChildProcessRecord,
    build_h8_child_invocation,
    classify_h8_child_outcome,
    compare_operands,
    conservative_hwm_endpoints,
    decode_h8_child_result,
    decode_h8_control_result,
    make_operand_record,
    make_h8_identity_record,
    parse_h8_child_stdout,
    windows_process_memory_layout,
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


def _fixture_jsonable(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _fixture_jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, GateStatus):
        return value.value
    if isinstance(value, tuple):
        return [_fixture_jsonable(item) for item in value]
    return value


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
                    "allowance": _fixture_jsonable(
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


def _child_envelope(**updates: object) -> dict[str, object]:
    digest = "a" * 64
    layout = {"horizon": 128, "d_z": 20, "d_m": 20}
    stored_blocks = [
        *(
            {"kind": "diagonal", "row": index, "column": index}
            for index in range(129)
        ),
        *(
            {"kind": "lower_adjacent", "row": index, "column": index - 1}
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
            "dispatch_events": [_fixture_jsonable(dispatch_event)],
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
        "residuals": {name: 0.0 for name in H8_SCALE_RESIDUAL_SPECS},
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
        "schema_version": "h8-child-v1",
        "mode": "production",
        "seed": 20260721,
        "repetition": 0,
        "control_id": None,
        "request_sha256": digest,
        "config_sha256": "b" * 64,
        "protocol_sha256": "c" * 64,
        "status": "pass",
        "obligations": [],
        "identities": {
            name: make_h8_identity_record(
                name,
                payload,
            )
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
        },
        "result": result,
        "control": None,
        "error": None,
    }
    record.update(updates)
    return record


def test_child_invocation_is_exact_and_freezes_thread_environment(
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
        base_environment={"PATH": "preserved"},
    )

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
        invocation.environment[name]
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        )
    } == {"1"}


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

    malformed = b'{"schema_version":"h8-child-v1"}\n{"schema_version":"h8-child-v1"}\n'
    with pytest.raises(ValueError, match="one canonical JSON line"):
        parse_h8_child_stdout(malformed)
    with pytest.raises(ValueError, match="canonical"):
        parse_h8_child_stdout(b'{ "schema_version": "h8-child-v1" }\n')
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
    incomplete = b'{"schema_version":"h8-child-v1"'
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


def test_child_pass_rejects_hollow_nested_evidence() -> None:
    hollow = _child_envelope()
    hollow["result"] = {
        name: {} if name != "invariants" else []
        for name in (
            "input_sha256",
            "sample_noise_sha256",
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
