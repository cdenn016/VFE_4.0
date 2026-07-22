from __future__ import annotations

import gc
import hashlib
import inspect
import json
import weakref
from dataclasses import fields, replace

import numpy as np
import pytest

from vfe4.config import H4ConditionEnvelopeConfig
from vfe4.generative.reference_h4 import h4_anchor_from_h3, make_h4_problem
from vfe4.types.h4 import (
    H4_ALLOWANCE_ELEMENT_COUNTS,
    H4AllowanceOperationCount,
    H4NativeInformationState,
    H4NativeMomentState,
    H4SelectedMoment,
    H4SolverResult,
    H4TerminalLaw,
    canonical_h4_problem_bytes,
)
from vfe4.validation.h3_fixture import (
    H3_COUPLED_FIXTURE_PATH,
    H3_ZERO_CONTROL_FIXTURE_PATH,
    parse_h3_fixture_bytes,
)
from verification import h4_budget as h4_budget_module
from verification.h4_budget import (
    _H4AllowanceGroupInput,
    _H4AllowanceOperandGroup,
    H4AllowanceResultSource,
    H4AnchorAllowanceSource,
    H4ResultAllowanceGroupBundle,
    H4SixInvariantAllowanceAccumulator,
    aggregate_allowance_groups,
    allowance_group_header,
    build_h4_anchor_identity_allowance,
    build_h4_complete_objective_allowance,
    build_h4_exact_posterior_gap_allowance,
    build_h4_selected_moment_allowance,
    build_h4_terminal_h_allowance,
    build_h4_terminal_j_allowance,
    cholesky_operation_count,
    dot_operation_count,
    gamma_n,
    innovation_condition_record,
    h4_anchor_identity_groups,
    h4_result_allowance_group_bundle,
    matrix_multiply_operation_count,
    operand_allowance,
    pair_element_allowance,
    posterior_condition_record,
    triangular_solve_operation_count,
    new_h4_six_invariant_allowance_accumulator,
)
from verification.numpy_oracles.h4_gaussian import (
    evaluate_h4_oracle,
    reverse_kl_to_h4_oracle,
)


def test_h4_budget_scalar_arithmetic_and_operation_counts() -> None:
    assert gamma_n(0) == 0.0
    assert dot_operation_count(0) == 0 and dot_operation_count(4) == 7
    assert matrix_multiply_operation_count(2, 3, 4) == 40
    assert triangular_solve_operation_count(4, 2) == 32
    assert cholesky_operation_count(4) == 22
    scale = 2.0
    left = operand_allowance(
        label="left", value=2.0, value_norm=2.0,
        absolute_summand_accumulation=3.0, condition_numbers=(4.0,),
        operation_counts=(H4AllowanceOperationCount("dot", 7),),
        solver_produced=True, invariant_scale=scale,
    )
    right = operand_allowance(
        label="right", value=2.0, value_norm=2.0,
        absolute_summand_accumulation=2.0, condition_numbers=(1.0,),
        operation_counts=(), solver_produced=False, invariant_scale=scale,
    )
    element = pair_element_allowance(
        stream_index=0, invariant="terminal_h_equivalence",
        problem_id="h4-coupled-T7-dz4-dm4-seed104729-v1",
        comparison_source="solver_to_oracle", repetition_index=0,
        arm="information", path="h[0]", shape=(64,), flat_index=0,
        left=left, right=right,
    )
    assert element.passed and element.decisive
    assert element.left.solver_allowance == 2.0e-9


def test_h4_allowance_wrappers_have_frozen_keyword_only_signatures() -> None:
    wrappers = (
        build_h4_anchor_identity_allowance,
        build_h4_exact_posterior_gap_allowance,
        build_h4_terminal_h_allowance,
        build_h4_terminal_j_allowance,
        build_h4_selected_moment_allowance,
        build_h4_complete_objective_allowance,
    )
    for wrapper in wrappers:
        parameters = inspect.signature(wrapper).parameters
        assert tuple(parameters) == ("expected_group_headers", "groups")
        assert all(item.kind is inspect.Parameter.KEYWORD_ONLY for item in parameters.values())


def _result_source(
    *, payload: bytes, arm: str, repetition_index: int | None,
) -> H4AllowanceResultSource:
    oracle = evaluate_h4_oracle(payload)
    def symmetric(matrix: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
        values = np.asarray(matrix, dtype=np.float64)
        values = 0.5 * (values + values.T)
        return tuple(tuple(float(value) for value in row) for row in values)

    selected = tuple(
        H4SelectedMoment(item.name, item.mean, symmetric(item.covariance))
        for item in oracle.selected_moments
    )
    if arm == "information":
        native_information = H4NativeInformationState(
            oracle.natural, oracle.precision, oracle.mean,
            oracle.canonical_log_normalizer,
        )
        native_moment = None
    else:
        native_information = None
        native_moment = H4NativeMomentState(
            oracle.mean, symmetric(oracle.covariance), oracle.canonical_log_normalizer,
        )
    result = H4SolverResult(
        oracle.problem_id, oracle.problem_sha256, arm, "h4-single-pass-v1",
        len(oracle.factor_ids), native_information, native_moment,
    )
    terminal = H4TerminalLaw(
        arm, oracle.natural, oracle.precision, oracle.mean, selected,
        oracle.canonical_log_normalizer, 0.0,
    )
    kl = reverse_kl_to_h4_oracle(
        oracle, mean=terminal.mean, precision=terminal.J,
    )
    return H4AllowanceResultSource(
        payload, repetition_index, oracle, result, terminal, kl,
    )


def test_h4_task3_allowance_source_and_bundle_public_contract() -> None:
    assert tuple(field.name for field in fields(H4AllowanceResultSource)) == (
        "problem_payload", "repetition_index", "oracle", "result", "terminal",
        "kl_to_oracle",
    )
    assert tuple(field.name for field in fields(H4AnchorAllowanceSource)) == (
        "h3_fixture_bytes", "information", "moment",
    )
    assert tuple(field.name for field in fields(H4ResultAllowanceGroupBundle)) == (
        "kl_to_zero", "terminal_h", "terminal_J",
        "selected_mean_and_covariance", "complete_objective",
    )
    assert tuple(inspect.signature(h4_result_allowance_group_bundle).parameters) == ("source",)
    assert inspect.signature(h4_result_allowance_group_bundle).parameters["source"].kind is inspect.Parameter.KEYWORD_ONLY
    assert type(new_h4_six_invariant_allowance_accumulator()) is H4SixInvariantAllowanceAccumulator
    assert tuple(
        inspect.signature(
            H4SixInvariantAllowanceAccumulator.anchor_identity_record,
        ).parameters
    ) == ("self",)
    assert {
        "H4AllowanceResultSource", "H4AnchorAllowanceSource",
        "H4ResultAllowanceGroupBundle", "H4SixInvariantAllowanceAccumulator",
        "h4_anchor_identity_groups", "h4_result_allowance_group_bundle",
        "new_h4_six_invariant_allowance_accumulator",
    } <= set(h4_budget_module.__all__)
    assert "_H4AllowanceGroupInput" not in h4_budget_module.__all__
    assert "_H4AllowanceOperandGroup" not in h4_budget_module.__all__


def test_h4_result_bundle_has_exact_operand_local_tables_and_numeric_headers() -> None:
    payload = canonical_h4_problem_bytes(
        make_h4_problem(seed=104729, kind="coupled", horizon=7),
    )
    source = _result_source(payload=payload, arm="information", repetition_index=0)
    with pytest.raises(ValueError, match="scaled.*repetition"):
        replace(source, repetition_index=None)
    bundle = h4_result_allowance_group_bundle(source=source)
    assert tuple(item.label for item in bundle.terminal_h.left.operation_counts) == ()
    assert tuple(item.label for item in bundle.terminal_J.left.operation_counts) == (
        "terminal_information_precision_proof_cholesky",
    )
    assert tuple(item.label for item in bundle.terminal_J.right.operation_counts) == (
        "factor_covariance_cholesky", "factor_precision_solves_A",
        "factor_J_assembly_matmuls", "factor_J_sum_reduction",
        "posterior_precision_symmetrization",
    )
    objective_labels = tuple(
        item.label for item in bundle.complete_objective.right.operation_counts
    )
    assert objective_labels[-7:] == (
        "factor_c_sum_reduction", "posterior_precision_symmetrization",
        "posterior_precision_cholesky", "posterior_natural_solve",
        "posterior_quadratic", "posterior_logdet_reduction",
        "route_sum_reduction",
    )
    baseline = allowance_group_header("terminal_h_equivalence", bundle.terminal_h)
    header = json.loads(baseline)
    expected_vector_digest = hashlib.sha256(
        b"vfe4.h4.allowance-group-vector.v1\x00"
        + b"left_value\x00"
        + bundle.terminal_h.left.values.size.to_bytes(8, "big")
        + np.asarray(bundle.terminal_h.left.values, dtype="<f8").tobytes(order="C")
    ).hexdigest()
    assert header["left_values_sha256"] == expected_vector_digest
    changed_values = bundle.terminal_h.left.values.copy()
    changed_values[0] += 1.0
    changed_values.setflags(write=False)
    changed_left = replace(bundle.terminal_h.left, values=changed_values)
    changed = replace(bundle.terminal_h, left=changed_left)
    assert baseline != allowance_group_header("terminal_h_equivalence", changed)
    with pytest.raises(ValueError, match="header"):
        aggregate_allowance_groups(
            "terminal_h_equivalence", expected_element_count=394_240,
            expected_group_headers=(baseline,), groups=(changed,),
        )


def test_h4_result_bundle_rejects_mixed_identity_shape_and_selected_sequence() -> None:
    payload = canonical_h4_problem_bytes(
        make_h4_problem(seed=104729, kind="coupled", horizon=7),
    )
    source = _result_source(payload=payload, arm="information", repetition_index=0)
    bundle = h4_result_allowance_group_bundle(source=source)
    malformed_groups = (
        replace(bundle.terminal_h, repetition_index=1),
        replace(bundle.terminal_h, arm="moment"),
        replace(bundle.terminal_h, problem_sha256="b" * 64),
        replace(bundle.terminal_h, shape=(8, 8)),
    )
    for malformed in malformed_groups:
        with pytest.raises(ValueError, match="bundle|identity|shape|repetition|arm"):
            replace(bundle, terminal_h=malformed)
    with pytest.raises(ValueError, match="selected|sequence"):
        replace(
            bundle,
            selected_mean_and_covariance=(
                bundle.selected_mean_and_covariance[1],
                bundle.selected_mean_and_covariance[0],
                *bundle.selected_mean_and_covariance[2:],
            ),
        )


def test_h4_anchor_repetition_and_accumulator_order_are_one_shot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_bytes = H3_COUPLED_FIXTURE_PATH.read_bytes()
    fixture = parse_h3_fixture_bytes(
        fixture_bytes, expected_fixture_id="h3-coupled-v1",
    )
    payload = canonical_h4_problem_bytes(h4_anchor_from_h3(fixture))
    information = _result_source(payload=payload, arm="information", repetition_index=None)
    moment = _result_source(payload=payload, arm="moment", repetition_index=None)
    with pytest.raises(ValueError, match="anchor.*repetition"):
        H4AnchorAllowanceSource(
            fixture_bytes, replace(information, repetition_index=0), moment,
        )
    anchor = H4AnchorAllowanceSource(fixture_bytes, information, moment)
    groups = h4_anchor_identity_groups(source=anchor)
    assert sum(np.prod(group.shape) for group in groups[:10]) == 40
    assert sum(np.prod(group.shape) for group in groups[10:20]) == 40
    assert sum(np.prod(group.shape) for group in groups[20:]) == 22
    adapter = {group.path_prefix: group.left for group in groups[20:]}
    assert tuple(item.label for item in adapter["adapter.c"].operation_counts) == (
        "factor_covariance_cholesky", "factor_precision_solves_b",
        "factor_c_quadratics", "factor_c_logdet_reductions",
        "factor_c_scalar_combinations", "factor_c_sum_reduction",
    )
    full = {item.label: item.count for item in adapter["adapter.logZ"].operation_counts}
    J = {item.label: item.count for item in adapter["adapter.J"].operation_counts}
    h = {item.label: item.count for item in adapter["adapter.h"].operation_counts}
    c = {item.label: item.count for item in adapter["adapter.c"].operation_counts}
    assert full["factor_covariance_cholesky"] == J["factor_covariance_cholesky"]
    assert full["factor_triangular_solves"] == (
        J["factor_precision_solves_A"] + h["factor_precision_solves_b"]
    )
    assert full["factor_assembly_matmuls"] == (
        J["factor_J_assembly_matmuls"] + h["factor_h_assembly_matmuls"]
    )
    assert full["factor_quadratics"] == c["factor_c_quadratics"]
    assert full["factor_logdet_reductions"] == c["factor_c_logdet_reductions"]
    assert full["factor_J_sum_reduction"] == J["factor_J_sum_reduction"]
    assert full["factor_h_sum_reduction"] == h["factor_h_sum_reduction"]
    assert full["factor_c_scalar_combinations"] == c["factor_c_scalar_combinations"]
    assert full["factor_c_sum_reduction"] == c["factor_c_sum_reduction"]
    assert full["posterior_precision_symmetrization"] == J["posterior_precision_symmetrization"]
    result_ref = weakref.ref(information.result)
    accumulator = new_h4_six_invariant_allowance_accumulator()
    array_refs: list[weakref.ReferenceType[np.ndarray]] = []
    real_anchor_groups = h4_budget_module.h4_anchor_identity_groups

    def tracked_anchor_groups(
        *, source: H4AnchorAllowanceSource,
    ) -> tuple[_H4AllowanceGroupInput, ...]:
        tracked = real_anchor_groups(source=source)
        array_refs.extend(
            weakref.ref(array)
            for group in tracked
            for array in (
                group.left.values,
                group.left.absolute_summand_accumulations,
                group.right.values,
                group.right.absolute_summand_accumulations,
            )
        )
        return tracked

    monkeypatch.setattr(
        h4_budget_module, "h4_anchor_identity_groups", tracked_anchor_groups,
    )
    accumulator.consume(anchor)
    gc.collect()
    assert array_refs and all(reference() is None for reference in array_refs)
    with pytest.raises(ValueError, match="order|duplicate"):
        accumulator.consume(anchor)
    del anchor, information, moment
    gc.collect()
    assert result_ref() is None
    assert not hasattr(accumulator, "sources") and not hasattr(accumulator, "_sources")


def _anchor_source(kind: str) -> H4AnchorAllowanceSource:
    if kind == "coupled":
        path, fixture_id = H3_COUPLED_FIXTURE_PATH, "h3-coupled-v1"
    else:
        path, fixture_id = H3_ZERO_CONTROL_FIXTURE_PATH, "h3-zero-control-v1"
    fixture_bytes = path.read_bytes()
    fixture = parse_h3_fixture_bytes(
        fixture_bytes, expected_fixture_id=fixture_id,
    )
    payload = canonical_h4_problem_bytes(h4_anchor_from_h3(fixture))
    return H4AnchorAllowanceSource(
        fixture_bytes,
        _result_source(payload=payload, arm="information", repetition_index=None),
        _result_source(payload=payload, arm="moment", repetition_index=None),
    )


def test_h4_early_anchor_identity_snapshot_is_cached_decisive_and_allows_scaled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accumulator = new_h4_six_invariant_allowance_accumulator()
    accumulator.consume(_anchor_source("coupled"))
    accumulator.consume(_anchor_source("zero_control"))

    def forbidden_replay(*args: object, **kwargs: object) -> object:
        raise AssertionError("anchor source was replayed")

    monkeypatch.setattr(
        h4_budget_module, "h4_anchor_identity_groups", forbidden_replay,
    )
    record = accumulator.anchor_identity_record()
    assert record.invariant == "h3_anchor_identity"
    assert record.expected_element_count == record.observed_element_count == 184
    assert record.decisive
    assert accumulator.anchor_identity_record() is record

    scaled_payload = canonical_h4_problem_bytes(
        make_h4_problem(seed=104729, kind="coupled", horizon=7),
    )
    scaled = _result_source(
        payload=scaled_payload, arm="information", repetition_index=0,
    )
    accumulator.consume(scaled)
    with pytest.raises(ValueError, match="anchor|boundary|scaled|closed"):
        accumulator.anchor_identity_record()
    with pytest.raises(ValueError, match="failed|closed"):
        accumulator.consume(scaled)
    with pytest.raises(ValueError, match="failed|closed"):
        accumulator.finalize()


def test_h4_early_anchor_snapshot_is_required_and_invalid_calls_fail_closed() -> None:
    coupled = _anchor_source("coupled")
    zero = _anchor_source("zero_control")

    premature = new_h4_six_invariant_allowance_accumulator()
    premature.consume(coupled)
    with pytest.raises(ValueError, match="anchor|boundary|premature"):
        premature.anchor_identity_record()
    with pytest.raises(ValueError, match="failed|closed"):
        premature.consume(zero)
    with pytest.raises(ValueError, match="failed|closed"):
        premature.anchor_identity_record()
    with pytest.raises(ValueError, match="failed|closed"):
        premature.finalize()

    reordered = new_h4_six_invariant_allowance_accumulator()
    with pytest.raises(ValueError, match="anchor.*order"):
        reordered.consume(zero)
    with pytest.raises(ValueError, match="failed|closed"):
        reordered.consume(coupled)
    with pytest.raises(ValueError, match="failed|closed"):
        reordered.anchor_identity_record()
    with pytest.raises(ValueError, match="failed|closed"):
        reordered.finalize()


def test_h4_incomplete_finalize_permanently_fail_closes_accumulator() -> None:
    accumulator = new_h4_six_invariant_allowance_accumulator()
    coupled = _anchor_source("coupled")
    with pytest.raises(ValueError, match="incomplete"):
        accumulator.finalize()
    with pytest.raises(ValueError, match="failed|closed"):
        accumulator.consume(coupled)
    with pytest.raises(ValueError, match="failed|closed"):
        accumulator.anchor_identity_record()
    with pytest.raises(ValueError, match="failed|closed"):
        accumulator.finalize()


def test_h4_full_finalize_reuses_cached_anchor_and_post_finalize_fails_closed() -> None:
    accumulator = new_h4_six_invariant_allowance_accumulator()
    coupled = _anchor_source("coupled")
    accumulator.consume(coupled)
    accumulator.consume(_anchor_source("zero_control"))
    anchor_record = accumulator.anchor_identity_record()

    class FinalState:
        def finalize(self) -> object:
            return anchor_record

    class ForbiddenAnchorState:
        def finalize(self) -> object:
            raise AssertionError("cached anchor state was finalized twice")

    accumulator._states = {
        name: (
            ForbiddenAnchorState()
            if name == "h3_anchor_identity"
            else FinalState()
        )
        for name, _ in H4_ALLOWANCE_ELEMENT_COUNTS
    }
    accumulator._position = 2 + 120 * 11 * 2
    records = accumulator.finalize()
    assert len(records) == 6 and records[0] is anchor_record
    with pytest.raises(ValueError, match="final|closed"):
        accumulator.anchor_identity_record()
    with pytest.raises(ValueError, match="final|closed"):
        accumulator.consume(coupled)
    with pytest.raises(ValueError, match="final|closed"):
        accumulator.finalize()


def test_h4_vectorized_allowance_aggregation_uses_read_only_groups() -> None:
    count = 2_640
    values = np.linspace(-1.0, 1.0, count, dtype=np.float64)
    values.setflags(write=False)
    sums = np.abs(values).copy()
    sums.setflags(write=False)
    left = _H4AllowanceOperandGroup("solver", values, 1.0, sums, (2.0,), (), True)
    right_values = np.zeros(count, dtype=np.float64)
    right_sums = np.zeros(count, dtype=np.float64)
    right_values.setflags(write=False)
    right_sums.setflags(write=False)
    right = _H4AllowanceOperandGroup("zero", right_values, 0.0, right_sums, (1.0,), (), False)
    group = _H4AllowanceGroupInput(
        "h4-coupled-T7-dz4-dm4-seed104729-v1", "a" * 64,
        "solver_to_oracle", 0, "information", "kl", (count,), left, right,
    )
    header = allowance_group_header("exact_posterior_gap_equivalence", group)
    record = build_h4_exact_posterior_gap_allowance(
        expected_group_headers=(header,), groups=(group,),
    )
    assert record.observed_element_count == count
    assert record.first_failed_element is not None


def test_h4_condition_classifiers_apply_only_their_declared_limits() -> None:
    envelope = H4ConditionEnvelopeConfig(
        1.0e-6, 1.0e6, 1.0e8, 1.0e-3, 16.0,
        1.0e-6, 1.0e6, 1.0e8, True,
    )
    posterior = posterior_condition_record(
        problem_id="p", problem_sha256="a" * 64, source="numpy_oracle",
        repetition_index=None, dimension=4, minimum_eigenvalue=1.0e-6,
        maximum_eigenvalue=1.0e6, condition_number=1.0e8,
        minimum_cholesky_pivot=1.0e-3, mean_infinity_norm=16.0,
        envelope=envelope,
    )
    innovation = innovation_condition_record(
        problem_id="p", problem_sha256="a" * 64, source="numpy_oracle",
        repetition_index=None, factor_id="observation[1]", time_index=1,
        parent_coordinate_indices=(0, 1), innovation_dimension=2,
        minimum_eigenvalue=1.0e-6, maximum_eigenvalue=1.0e6,
        condition_number=1.0e8, envelope=envelope,
    )
    assert posterior.eligible and innovation.eligible


def _large_terminal_h_group(late_value: float) -> _H4AllowanceGroupInput:
    count = 394_240
    values = np.zeros(count, dtype=np.float64)
    values[4096] = late_value
    sums = np.abs(values)
    zeros = np.zeros(count, dtype=np.float64)
    for array in (values, sums, zeros):
        array.setflags(write=False)
    left = _H4AllowanceOperandGroup(
        "solver", values, 2.0, sums, (1.0,), (), True,
    )
    right = _H4AllowanceOperandGroup(
        "oracle", zeros, 0.0, zeros, (1.0,), (), False,
    )
    return _H4AllowanceGroupInput(
        "h4-coupled-T31-dz4-dm4-seed104729-v1", "a" * 64,
        "solver_to_oracle", 0, "information", "h", (count,), left, right,
    )


def test_h4_allowance_stream_chunks_bound_memory_and_hashes_late_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_group = _large_terminal_h_group(1.0)
    changed_group = _large_terminal_h_group(2.0)
    header = allowance_group_header("terminal_h_equivalence", baseline_group)
    changed_header = allowance_group_header("terminal_h_equivalence", changed_group)
    assert header != changed_header

    scratch_rows: list[int] = []
    real_empty = h4_budget_module.np.empty

    def tracked_empty(shape: int, *args: object, **kwargs: object) -> np.ndarray:
        dtype = kwargs.get("dtype")
        if dtype is not None and np.dtype(dtype) == h4_budget_module._PACKED_ROW_DTYPE:
            scratch_rows.append(shape)
        return real_empty(shape, *args, **kwargs)

    monkeypatch.setattr(h4_budget_module.np, "empty", tracked_empty)
    baseline = aggregate_allowance_groups(
        "terminal_h_equivalence", expected_element_count=394_240,
        expected_group_headers=(header,), groups=(baseline_group,),
    )
    assert scratch_rows and max(scratch_rows) == 4096
    assert baseline.first_failed_element is not None
    assert baseline.first_failed_element.stream_index == 4096
    assert baseline.first_failed_element.flat_index == 4096
    assert baseline.first_failed_element.path == "h[4096]"
    assert not hasattr(baseline, "elements")

    monkeypatch.setattr(h4_budget_module, "H4_MAXIMUM_CHUNK_ROWS", 2048)
    rechunked = aggregate_allowance_groups(
        "terminal_h_equivalence", expected_element_count=394_240,
        expected_group_headers=(header,), groups=(baseline_group,),
    )
    assert rechunked.element_stream_sha256 == baseline.element_stream_sha256

    monkeypatch.setattr(h4_budget_module, "H4_MAXIMUM_CHUNK_ROWS", 4096)
    changed = aggregate_allowance_groups(
        "terminal_h_equivalence", expected_element_count=394_240,
        expected_group_headers=(changed_header,), groups=(changed_group,),
    )
    assert changed.element_stream_sha256 != baseline.element_stream_sha256
