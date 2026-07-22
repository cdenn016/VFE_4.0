from __future__ import annotations

import ast
import copy
import inspect
import json
import math
import struct
from dataclasses import fields, replace
from pathlib import Path

import pytest

import verification.numpy_oracles.h5_updates as oracle_module
from verification.numpy_oracles.h5_updates import (
    H5OracleOperandEvidence,
    H5OracleTermEvidence,
    H5OracleUpdate,
    oracle_complete_delta,
    oracle_exact_e_block,
    oracle_exact_m_block,
    oracle_exact_source_row,
)
from vfe4.inference.h5_updates import (
    exact_conjugate_gaussian_e_update,
    exact_gaussian_m_update,
    exact_source_row_update,
)
from vfe4.objective.h5_complete import evaluate_h5_complete_elbo
from vfe4.types.updates import (
    H5_RULE_CONTRACTS,
    H5UpdateRule,
    UpdateRequest,
    canonical_h5_semantic_state_bytes,
    initial_live,
)
from vfe4.validation.h5_update_spec import build_h5_reference_state


ROOT = Path(__file__).parents[2]
H1_BYTES = (ROOT / "vfe4/validation/fixtures/h1_v1.json").read_bytes()
H5_BYTES = (
    ROOT / "vfe4/validation/fixtures/h5_conditional_update_v1.json"
).read_bytes()


def _request(rule: H5UpdateRule) -> UpdateRequest:
    label, variables, parameters, schedule = H5_RULE_CONTRACTS[rule]
    return UpdateRequest(
        "h5-update-request-v1",
        f"oracle-{rule.value}",
        rule,
        label,
        variables,
        parameters,
        schedule,
    )


def _decode_float(value):
    if isinstance(value, str) and value.startswith(("0x", "-0x")):
        return float.fromhex(value)
    if isinstance(value, list):
        return [_decode_float(item) for item in value]
    if isinstance(value, dict):
        return {key: _decode_float(item) for key, item in value.items()}
    return value


def _oracle_active_values(result: H5OracleUpdate, rule: H5UpdateRule):
    recognition = _decode_float(json.loads(result.candidate_recognition_json))
    model = _decode_float(json.loads(result.candidate_model_json))
    if rule is H5UpdateRule.EXACT_Z0:
        row = next(item for item in recognition["gaussians"] if item[0] == "q[z0]")
        return (row[1]["values"][0], row[2]["values"][0])
    if rule is H5UpdateRule.EXACT_SOURCE_ROW_A2:
        row = next(
            item
            for item in recognition["categoricals"]
            if item[0] == "q[source_row_a2]"
        )
        return tuple(row[3]["values"])
    block = next(
        item
        for item in model["parameter_blocks"]
        if item[0] == "theta[state_transition_2]"
    )
    return tuple(value["values"][0] for _, value in block[1])


def _production_active_values(candidate, rule: H5UpdateRule):
    if rule is H5UpdateRule.EXACT_Z0:
        row = next(x for x in candidate.recognition.gaussians if x.coordinate_id == "q[z0]")
        return (row.mean.values[0], row.variance.values[0])
    if rule is H5UpdateRule.EXACT_SOURCE_ROW_A2:
        row = next(
            x
            for x in candidate.recognition.categoricals
            if x.coordinate_id == "q[source_row_a2]"
        )
        return row.probabilities.values
    block = next(
        x
        for x in candidate.model.parameter_blocks
        if x.block_id == "theta[state_transition_2]"
    )
    return tuple(value.values[0] for _, value in block.values)


def test_oracle_record_schema_and_module_are_independent() -> None:
    assert tuple(field.name for field in fields(H5OracleTermEvidence)) == (
        "schema_version",
        "term_id",
        "objective_sign",
        "value_order_21",
        "value_order_17",
        "signed_reported_value",
        "absolute_summands_order_21",
        "absolute_summands_order_17",
        "condition_numbers_order_21",
        "condition_numbers_order_17",
        "operation_count_order_21",
        "operation_count_order_17",
        "convergence_estimate",
        "rounding_order_21",
        "rounding_order_17",
        "comparison_rounding",
        "total",
    )
    assert tuple(field.name for field in fields(H5OracleOperandEvidence)) == (
        "schema_version",
        "operand",
        "complete_term_trace",
        "value",
        "operation_count",
        "condition_numbers",
        "absolute_summands",
        "convergence",
        "rounding",
        "allowance",
    )
    assert tuple(field.name for field in fields(H5OracleUpdate)) == (
        "schema_version",
        "rule",
        "candidate_recognition_json",
        "candidate_model_json",
        "candidate_condition_numbers",
        "semantic_state_sha256",
        "before",
        "after",
        "delta",
    )
    assert oracle_module.__all__ == [
        "H5OracleTermEvidence",
        "H5OracleOperandEvidence",
        "H5OracleUpdate",
        "oracle_exact_e_block",
        "oracle_exact_source_row",
        "oracle_exact_m_block",
        "oracle_complete_delta",
    ]
    complete_signature = inspect.signature(
        H5OracleOperandEvidence.from_complete_terms
    )
    assert tuple(complete_signature.parameters) == (
        "operand",
        "complete_term_trace",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in complete_signature.parameters.values()
    )
    delta_signature = inspect.signature(H5OracleOperandEvidence.from_delta)
    assert tuple(delta_signature.parameters) == ("before", "after")
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in delta_signature.parameters.values()
    )
    source = (ROOT / "verification/numpy_oracles/h5_updates.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    imports = tuple(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ) + tuple(
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )
    assert "torch" not in imports
    assert not any(name == "vfe4" or name.startswith("vfe4.") for name in imports)


@pytest.mark.parametrize(
    ("rule", "production_function", "oracle_function"),
    (
        (
            H5UpdateRule.EXACT_Z0,
            exact_conjugate_gaussian_e_update,
            oracle_exact_e_block,
        ),
        (
            H5UpdateRule.EXACT_SOURCE_ROW_A2,
            exact_source_row_update,
            oracle_exact_source_row,
        ),
        (
            H5UpdateRule.EXACT_STATE_TRANSITION_2_M,
            exact_gaussian_m_update,
            oracle_exact_m_block,
        ),
    ),
)
def test_byte_only_oracle_matches_exact_candidate_fields(
    rule, production_function, oracle_function
) -> None:
    reference = build_h5_reference_state(H1_BYTES, H5_BYTES)
    live = initial_live(reference)
    production = production_function(reference, live, _request(rule))
    result = oracle_function(
        H1_BYTES,
        H5_BYTES,
        canonical_h5_semantic_state_bytes(live.recognition, live.model),
    )
    assert result.delta.value == pytest.approx(
        result.after.value - result.before.value, rel=0.0, abs=2e-14
    )
    assert _oracle_active_values(result, rule) == pytest.approx(
        _production_active_values(production, rule), rel=0.0, abs=2e-13
    )
    if rule is H5UpdateRule.EXACT_STATE_TRANSITION_2_M:
        assert result.candidate_condition_numbers[0][0] == "G_condition_number"
        assert result.candidate_condition_numbers[0][1] >= 1.0
    else:
        assert result.candidate_condition_numbers == ()


def test_oracle_complete_delta_recomputes_both_states_from_bytes() -> None:
    reference = build_h5_reference_state(H1_BYTES, H5_BYTES)
    live = initial_live(reference)
    candidate = exact_conjugate_gaussian_e_update(
        reference, live, _request(H5UpdateRule.EXACT_Z0)
    )
    result = oracle_complete_delta(
        H1_BYTES,
        H5_BYTES,
        canonical_h5_semantic_state_bytes(live.recognition, live.model),
        canonical_h5_semantic_state_bytes(candidate.recognition, candidate.model),
        rule=H5UpdateRule.EXACT_Z0.value,
    )
    assert result.after.value > result.before.value
    assert result.delta.value == pytest.approx(
        result.after.value - result.before.value, rel=0.0, abs=2e-14
    )


_TERM_IDS = (
    "expected_log_emission[1]",
    "expected_log_emission[2]",
    "initial_model_kl",
    "initial_state_kl",
    "model_source_kl[1]",
    "model_source_kl[2]",
    "model_transition_kl[1]",
    "model_transition_kl[2]",
    "state_source_kl[1]",
    "state_source_kl[2]",
    "state_transition_kl[1]",
    "state_transition_kl[2]",
)
_TERM_SIGNS = (1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)
_ANALYTIC_COUNTS = {
    "initial_model_kl": 192,
    "initial_state_kl": 192,
    "model_source_kl[1]": 32,
    "model_source_kl[2]": 64,
    "model_transition_kl[1]": 192,
    "model_transition_kl[2]": 320,
    "state_source_kl[1]": 32,
    "state_source_kl[2]": 96,
    "state_transition_kl[1]": 256,
    "state_transition_kl[2]": 448,
}


def _gamma_n(count: int) -> float:
    epsilon = float.fromhex("0x1.0000000000000p-52")
    return (count * epsilon) / (1.0 - count * epsilon)


def _emission_count(order: int) -> int:
    return 32 * order * order + 8 * order + 32


def _expected_rounding(term, *, order: int) -> float:
    value = getattr(term, f"value_order_{order}")
    summands = getattr(term, f"absolute_summands_order_{order}")
    conditions = getattr(term, f"condition_numbers_order_{order}")
    count = getattr(term, f"operation_count_order_{order}")
    return float(
        4096.0
        * _gamma_n(count)
        * max(1.0, *conditions)
        * max(1.0, abs(value), math.fsum(summands))
    )


def _assert_complete_operand_recomputes(operand, role: str) -> None:
    assert operand.operand == role
    trace = operand.complete_term_trace
    assert tuple(item.term_id for item in trace) == _TERM_IDS
    assert tuple(item.objective_sign for item in trace) == _TERM_SIGNS
    assert operand.value.hex() == math.fsum(
        item.signed_reported_value for item in trace
    ).hex()
    assert operand.operation_count == 13 + sum(
        item.operation_count_order_21 + item.operation_count_order_17
        for item in trace
    )
    assert operand.condition_numbers == tuple(
        value
        for item in trace
        for values in (
            item.condition_numbers_order_21,
            item.condition_numbers_order_17,
        )
        for value in values
    )
    assert operand.absolute_summands == (
        tuple(
            value
            for item in trace
            for values in (
                item.absolute_summands_order_21,
                item.absolute_summands_order_17,
            )
            for value in values
        )
        + tuple(abs(item.signed_reported_value) for item in trace)
    )
    expected_convergence = math.fsum(item.convergence_estimate for item in trace)
    reduction = float(
        4096.0
        * _gamma_n(13)
        * max(1.0, math.fsum(abs(item.signed_reported_value) for item in trace))
    )
    expected_rounding = math.fsum(
        tuple(
            value
            for item in trace
            for value in (
                item.rounding_order_21,
                item.rounding_order_17,
                item.comparison_rounding,
            )
        )
        + (reduction,)
    )
    assert operand.convergence.hex() == expected_convergence.hex()
    assert operand.rounding.hex() == expected_rounding.hex()
    assert operand.allowance.hex() == math.fsum(
        (expected_convergence, expected_rounding)
    ).hex()


def test_oracle_retains_self_validating_both_order_term_and_operand_evidence() -> None:
    reference = build_h5_reference_state(H1_BYTES, H5_BYTES)
    live = initial_live(reference)
    result = oracle_exact_e_block(
        H1_BYTES,
        H5_BYTES,
        canonical_h5_semantic_state_bytes(live.recognition, live.model),
    )
    _assert_complete_operand_recomputes(result.before, "before")
    _assert_complete_operand_recomputes(result.after, "after")
    for operand in (result.before, result.after):
        for term in operand.complete_term_trace:
            expected_counts = (
                (_emission_count(21), _emission_count(17))
                if term.term_id.startswith("expected_log_emission")
                else (_ANALYTIC_COUNTS[term.term_id],) * 2
            )
            assert (
                term.operation_count_order_21,
                term.operation_count_order_17,
            ) == expected_counts
            assert term.signed_reported_value.hex() == float(
                term.objective_sign * term.value_order_21
            ).hex()
            assert term.convergence_estimate.hex() == abs(
                term.value_order_21 - term.value_order_17
            ).hex()
            assert term.rounding_order_21.hex() == _expected_rounding(
                term, order=21
            ).hex()
            assert term.rounding_order_17.hex() == _expected_rounding(
                term, order=17
            ).hex()
            comparison = float(
                4096.0
                * _gamma_n(3)
                * max(
                    1.0,
                    abs(term.value_order_21),
                    abs(term.value_order_17),
                    abs(term.value_order_21) + abs(term.value_order_17),
                )
            )
            assert term.comparison_rounding.hex() == comparison.hex()
            assert term.total.hex() == math.fsum(
                (
                    term.convergence_estimate,
                    term.rounding_order_21,
                    term.rounding_order_17,
                    term.comparison_rounding,
                )
            ).hex()
            if not term.term_id.startswith("expected_log_emission"):
                assert term.value_order_21.hex() == term.value_order_17.hex()
                assert term.convergence_estimate.hex() == 0.0.hex()
    assert result.delta.operand == "delta"
    assert result.delta.complete_term_trace == ()
    assert result.delta.operation_count == 3
    assert result.delta.condition_numbers == (1.0,)
    assert result.delta.absolute_summands == (
        abs(result.before.value),
        abs(result.after.value),
    )
    subtraction = float(
        4096.0
        * _gamma_n(3)
        * max(
            1.0,
            abs(result.before.value),
            abs(result.after.value),
            abs(result.delta.value),
            abs(result.before.value) + abs(result.after.value),
        )
    )
    assert result.delta.convergence.hex() == math.fsum(
        (result.before.convergence, result.after.convergence)
    ).hex()
    assert result.delta.rounding.hex() == subtraction.hex()
    assert result.delta.allowance.hex() == math.fsum(
        (result.before.allowance, result.after.allowance, result.delta.rounding)
    ).hex()


@pytest.mark.parametrize(
    "field_name",
    (
        "signed_reported_value",
        "convergence_estimate",
        "rounding_order_21",
        "rounding_order_17",
        "comparison_rounding",
        "total",
    ),
)
def test_oracle_term_rejects_one_field_derived_evidence_mutation(
    field_name: str,
) -> None:
    reference = build_h5_reference_state(H1_BYTES, H5_BYTES)
    live = initial_live(reference)
    result = oracle_exact_e_block(
        H1_BYTES,
        H5_BYTES,
        canonical_h5_semantic_state_bytes(live.recognition, live.model),
    )
    term = result.before.complete_term_trace[0]
    with pytest.raises(ValueError):
        replace(
            term,
            **{field_name: math.nextafter(getattr(term, field_name), math.inf)},
        )
    with pytest.raises(ValueError):
        replace(
            term,
            value_order_21=math.nextafter(term.value_order_21, math.inf),
        )
    with pytest.raises(ValueError, match="operation count"):
        replace(
            term,
            operation_count_order_21=term.operation_count_order_21 + 1,
        )


def test_oracle_operand_rejects_direct_construction_order_and_aggregate_mutation() -> None:
    reference = build_h5_reference_state(H1_BYTES, H5_BYTES)
    live = initial_live(reference)
    result = oracle_exact_e_block(
        H1_BYTES,
        H5_BYTES,
        canonical_h5_semantic_state_bytes(live.recognition, live.model),
    )
    with pytest.raises(TypeError):
        H5OracleOperandEvidence(
            "h5-oracle-operand-evidence-v1",
            "before",
            result.before.complete_term_trace,
            result.before.value,
            result.before.operation_count,
            result.before.condition_numbers,
            result.before.absolute_summands,
            result.before.convergence,
            result.before.rounding,
            result.before.allowance,
        )
    swapped = (
        result.before.complete_term_trace[1],
        result.before.complete_term_trace[0],
    ) + result.before.complete_term_trace[2:]
    with pytest.raises(ValueError, match="term.*order|signed.*order"):
        H5OracleOperandEvidence.from_complete_terms(
            operand="before", complete_term_trace=swapped
        )
    tampered_before = copy.copy(result.before)
    object.__setattr__(
        tampered_before,
        "rounding",
        math.nextafter(tampered_before.rounding, math.inf),
    )
    with pytest.raises(ValueError):
        replace(result, before=tampered_before)
    tampered_delta = copy.copy(result.delta)
    object.__setattr__(
        tampered_delta,
        "allowance",
        math.nextafter(tampered_delta.allowance, math.inf),
    )
    with pytest.raises(ValueError):
        replace(result, delta=tampered_delta)


def test_production_allowance_perturbation_cannot_change_oracle_evidence() -> None:
    reference = build_h5_reference_state(H1_BYTES, H5_BYTES)
    live = initial_live(reference)
    semantic_bytes = canonical_h5_semantic_state_bytes(
        live.recognition, live.model
    )
    before = oracle_exact_e_block(H1_BYTES, H5_BYTES, semantic_bytes)
    production = evaluate_h5_complete_elbo(
        reference,
        live,
        frozen_complement_sha256="0" * 64,
    )
    tampered_allowance = copy.copy(production.complete_allowance)
    object.__setattr__(
        tampered_allowance,
        "total",
        math.nextafter(tampered_allowance.total, math.inf),
    )
    assert tampered_allowance.total != production.complete_allowance.total
    after = oracle_exact_e_block(H1_BYTES, H5_BYTES, semantic_bytes)
    assert after == before


def test_oracle_rejects_semantically_equal_but_raw_different_fixture_bytes() -> None:
    reference = build_h5_reference_state(H1_BYTES, H5_BYTES)
    live = initial_live(reference)
    live_bytes = canonical_h5_semantic_state_bytes(live.recognition, live.model)
    with pytest.raises(ValueError, match="raw SHA-256"):
        oracle_exact_e_block(H1_BYTES, H5_BYTES + b" ", live_bytes)


def _mutated_semantic_state(kind: str) -> bytes:
    reference = build_h5_reference_state(H1_BYTES, H5_BYTES)
    live = initial_live(reference)
    data = canonical_h5_semantic_state_bytes(live.recognition, live.model)
    domain = b"vfe4.h5.semantic-state.v1\x00"
    cursor = len(domain)
    recognition_length = struct.unpack(">Q", data[cursor : cursor + 8])[0]
    cursor += 8
    recognition = json.loads(data[cursor : cursor + recognition_length])
    cursor += recognition_length
    model_length = struct.unpack(">Q", data[cursor : cursor + 8])[0]
    cursor += 8
    model = json.loads(data[cursor : cursor + model_length])

    if kind == "categorical_support":
        row = next(
            item
            for item in recognition["categoricals"]
            if item[0] == "q[source_row_a2]"
        )
        row[1] = [1, 0]
    elif kind == "categorical_condition":
        row = next(
            item
            for item in recognition["categoricals"]
            if item[0] == "q[source_row_a2]"
        )
        row[2] = [["b2", 1]]
    elif kind == "objective_schema":
        model["objective_schema_sha256"] = "00" * 32
    elif kind == "reconstruction_records":
        model["reconstruction_records"][0][1][0] = "changed.binding"
    elif kind == "shared_groups":
        model["shared_groups"][0][2][0] = "changed.consumer"
    else:
        raise AssertionError(f"unknown mutation kind {kind}")

    recognition_bytes = json.dumps(
        recognition,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    model_bytes = json.dumps(
        model,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return (
        domain
        + struct.pack(">Q", len(recognition_bytes))
        + recognition_bytes
        + struct.pack(">Q", len(model_bytes))
        + model_bytes
    )


@pytest.mark.parametrize(
    "kind",
    (
        "categorical_support",
        "categorical_condition",
        "objective_schema",
        "reconstruction_records",
        "shared_groups",
    ),
)
def test_oracle_rejects_rehashed_semantic_metadata_mutation(kind: str) -> None:
    with pytest.raises(ValueError):
        oracle_exact_e_block(H1_BYTES, H5_BYTES, _mutated_semantic_state(kind))
