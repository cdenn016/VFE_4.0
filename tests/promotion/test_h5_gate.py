from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path
from types import MappingProxyType

import pytest

import verification.h5_gate as gate
from verification.h5_gate import (
    H5_CONTROL_BASE_RULE_BY_ID,
    H5_CONTROL_DETECTION_BY_ID,
    H5_POSITIVE_RULE_BY_ID,
    H5_PREFLIGHT_ERROR_KINDS_BY_PHASE,
    H5CandidateComparison,
    H5CandidateScalarComparison,
    H5ControlDetection,
    H5ControlId,
    H5ControlResult,
    H5DeltaAgreement,
    H5DeltaImplementationEvidence,
    H5DeltaOperandEvidence,
    H5GateEvaluation,
    H5GateResult,
    H5PositiveCaseId,
    H5PositiveCaseResult,
    H5PreflightError,
    H5PreflightErrorKind,
    H5PreflightPhase,
    H5PreflightRecord,
    H5UnavailableField,
    H5ValidationPayloadRecord,
    evaluate_h5,
    h5_validation_payload,
)
from vfe4.config import ResolvedConfig, resolve_config
from vfe4.inference.h5_updates import (
    AttemptFailureReason,
    AttemptPhase,
    CompletedUpdateAttempt,
    FailedUpdateAttempt,
)
from vfe4.types.h5_schema import (
    H5_C,
    H5_CANDIDATE_COMPARISON_OPERATION_COUNTS,
    H5_FACTOR_INPUT_SCHEMA_SHA256,
    H5_FACTOR_INPUT_SCHEMA_VERSION,
    H5_FACTOR_UNIVERSE,
    H5_MODEL_BLOCK_UNIVERSE,
    H5_NONCLAIM_IDS,
    H5_OBJECTIVE_SCHEMA_SHA256,
    H5_PARAMETER_DEPENDENCY_ROWS,
    H5_RECOGNITION_COORDINATE_UNIVERSE,
    H5_VALIDATION_PAYLOAD_DOMAIN,
    H5_VARIABLE_DEPENDENCY_ROWS,
    gamma_n,
)
from vfe4.types.results import GateStatus, InvariantResult
from vfe4.types.updates import H5UpdateRule, UpdateLabel


ROOT = Path(__file__).parents[2]
H1_BYTES = (ROOT / "vfe4/validation/fixtures/h1_v1.json").read_bytes()
H5_BYTES = (
    ROOT / "vfe4/validation/fixtures/h5_conditional_update_v1.json"
).read_bytes()


def _raw_config(run_root: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "objective_schema_version": "vfe4-state-elbo-v1",
        "run": {
            "mode": "verify",
            "seed": 20260721,
            "device": "cpu",
            "dtype": "float64",
            "deterministic": True,
        },
        "data": {"kind": "frozen_fixture", "identity": "h1-v1"},
        "model": {
            "horizon": 2,
            "d_z": 1,
            "d_m": 1,
            "vocabulary_size": 3,
            "state_parent_sets": [[0], [0, 1]],
            "model_parent_sets": [[0], [0, 1]],
            "state_source_support": [[0], [0, 1]],
            "model_source_support": [[0], [0, 1]],
            "geometry": "fixed_population_frames",
        },
        "recognition": {
            "conditioning": "smoothing",
            "family": "structured_linear_gaussian_mixture",
            "source_treatment": "exact_enumeration",
        },
        "inference": {
            "operation": "evaluate_only",
            "estimator": "deterministic_quadrature",
        },
        "optimization": {
            "e_like_update": "none",
            "m_like_update": "none",
            "expected_autograd_scope": "none",
        },
        "validation": {
            "gates": ["H1"],
            "fixture_id": "h1-v1",
            "quadrature_order": 21,
            "convergence_check_order": 17,
            "maximum_convergence_estimate": 1.0e-9,
        },
        "artifacts": {"run_root": str(run_root)},
    }


@pytest.fixture(scope="session")
def h5_config(tmp_path_factory: pytest.TempPathFactory) -> ResolvedConfig:
    return resolve_config(
        _raw_config(tmp_path_factory.mktemp("h5-gate") / "runs"),
        repo_root=ROOT,
    )


@pytest.fixture(scope="session")
def h5_evaluation(h5_config: ResolvedConfig) -> H5GateEvaluation:
    return evaluate_h5(
        h5_config,
        h1_fixture_bytes=H1_BYTES,
        h5_update_spec_bytes=H5_BYTES,
    )


RECORD_FIELDS = {
    H5PreflightError: ("phase", "kind", "detail"),
    H5PreflightRecord: (
        "schema_version",
        "phase",
        "h1_fixture_raw_sha256",
        "update_spec_raw_sha256",
        "errors",
        "unavailable_fields",
        "obligation",
    ),
    H5DeltaOperandEvidence: (
        "schema_version",
        "operand",
        "value",
        "operation_count",
        "condition_numbers",
        "absolute_summands",
        "rounding",
        "allowance",
    ),
    H5DeltaImplementationEvidence: (
        "schema_version",
        "implementation",
        "before",
        "after",
        "delta",
        "operand_shaped",
    ),
    H5DeltaAgreement: (
        "schema_version",
        "rule",
        "production",
        "oracle",
        "comparison_rounding",
        "allowance",
        "absolute_error",
        "passed",
    ),
    H5CandidateScalarComparison: (
        "field_id",
        "production_value",
        "oracle_value",
        "operation_count",
        "production_condition_number",
        "oracle_condition_number",
        "production_rounding",
        "oracle_rounding",
        "comparison_rounding",
        "allowance",
        "absolute_error",
        "passed",
    ),
    H5CandidateComparison: (
        "rule",
        "scalar_comparisons",
        "max_absolute_error",
        "max_allowance",
        "passed",
    ),
    H5PositiveCaseResult: (
        "schema_version",
        "case_id",
        "outcome",
        "production_semantic_state_sha256",
        "oracle_semantic_state_sha256",
        "candidate_comparison",
        "delta_agreement",
        "passed",
        "detail",
    ),
    H5ControlResult: (
        "schema_version",
        "control_id",
        "expected_detection",
        "observed_detection",
        "outcome",
        "passed",
        "detail",
    ),
    H5GateResult: (
        "schema_version",
        "gate",
        "status",
        "preflight",
        "h1_fixture_raw_sha256",
        "update_spec_raw_sha256",
        "update_spec_canonical_sha256",
        "objective_schema_sha256",
        "factor_input_schema_version",
        "factor_input_schema_sha256",
        "reference_sha256",
        "positive_cases",
        "controls",
        "invariants",
        "obligations",
    ),
    H5ValidationPayloadRecord: (
        "schema_version",
        "result",
        "reference_sha256",
        "factor_universe",
        "recognition_coordinate_universe",
        "model_block_universe",
        "variable_dependency_rows",
        "parameter_dependency_rows",
        "positive_attempts",
        "controls",
        "oracle_results",
        "nonclaims",
        "canonical_bytes",
        "payload_sha256",
    ),
    H5GateEvaluation: (
        "schema_version",
        "result",
        "reference",
        "positive_attempts",
        "controls",
        "oracle_results",
        "validation_payload",
    ),
}


def test_h5_gate_records_are_exact_frozen_slotted_and_maps_are_closed(
    h5_evaluation: H5GateEvaluation,
) -> None:
    for record_type, expected_fields in RECORD_FIELDS.items():
        assert tuple(field.name for field in fields(record_type)) == expected_fields
        assert hasattr(record_type, "__slots__")

    with pytest.raises(FrozenInstanceError):
        h5_evaluation.result.status = GateStatus.FAIL  # type: ignore[misc]

    assert isinstance(H5_CONTROL_DETECTION_BY_ID, MappingProxyType)
    assert isinstance(H5_POSITIVE_RULE_BY_ID, MappingProxyType)
    assert isinstance(H5_CONTROL_BASE_RULE_BY_ID, MappingProxyType)
    assert tuple(H5_CONTROL_DETECTION_BY_ID) == tuple(H5ControlId)
    assert tuple(H5_POSITIVE_RULE_BY_ID) == tuple(H5PositiveCaseId)
    assert tuple(H5_CONTROL_BASE_RULE_BY_ID) == tuple(H5ControlId)
    assert H5_CONTROL_BASE_RULE_BY_ID[H5ControlId.CHANGED_INPUT_EQUAL_VALUE] is H5UpdateRule.EXACT_STATE_TRANSITION_2_M
    assert H5_CONTROL_BASE_RULE_BY_ID[H5ControlId.CHANGED_VALUE_SAME_INPUT] is H5UpdateRule.GENERALIZED_EM_EMISSION_1

    assert isinstance(H5_PREFLIGHT_ERROR_KINDS_BY_PHASE, MappingProxyType)
    assert tuple(H5_PREFLIGHT_ERROR_KINDS_BY_PHASE) == tuple(H5PreflightPhase)
    owned = tuple(
        kind
        for phase in H5PreflightPhase
        for kind in H5_PREFLIGHT_ERROR_KINDS_BY_PHASE[phase]
    )
    assert len(owned) == len(set(owned)) == len(H5PreflightErrorKind)
    assert H5_PREFLIGHT_ERROR_KINDS_BY_PHASE[H5PreflightPhase.READY] == ()
    with pytest.raises(ValueError, match="phase"):
        H5PreflightError(
            H5PreflightPhase.H1_FIXTURE_VALIDATION,
            H5PreflightErrorKind.INVALID_UPDATE_SPEC_SCHEMA,
            "wrong owner",
        )


def test_h5_gate_consumes_captured_bytes_and_requires_all_cases_and_controls(
    h5_evaluation: H5GateEvaluation,
) -> None:
    evaluation = h5_evaluation
    assert evaluation.result.status is GateStatus.PASS
    assert evaluation.result.preflight.phase is H5PreflightPhase.READY
    assert evaluation.result.preflight.errors == ()
    assert evaluation.result.preflight.unavailable_fields == ()
    assert evaluation.result.obligations == ()
    assert tuple(case.case_id for case in evaluation.result.positive_cases or ()) == tuple(H5PositiveCaseId)
    assert tuple(control.control_id for control in evaluation.result.controls or ()) == tuple(H5ControlId)
    assert tuple(item.name for item in evaluation.result.invariants) == gate.H5_INVARIANT_NAMES
    assert all(item.passed for item in evaluation.result.invariants)
    assert "valid_mm" not in " ".join(evaluation.result.obligations)
    assert "read_bytes" not in inspect.getsource(evaluate_h5)
    assert "read_text" not in inspect.getsource(evaluate_h5)

    assert evaluation.result.h1_fixture_raw_sha256 == hashlib.sha256(H1_BYTES).hexdigest()
    assert evaluation.result.update_spec_raw_sha256 == hashlib.sha256(H5_BYTES).hexdigest()
    assert evaluation.result.update_spec_canonical_sha256 == evaluation.reference.specification.canonical_sha256
    assert evaluation.result.objective_schema_sha256 == H5_OBJECTIVE_SCHEMA_SHA256
    assert evaluation.result.factor_input_schema_version == H5_FACTOR_INPUT_SCHEMA_VERSION
    assert evaluation.result.factor_input_schema_sha256 == H5_FACTOR_INPUT_SCHEMA_SHA256
    assert evaluation.result.reference_sha256 == evaluation.reference.reference_sha256


def test_five_positive_cases_reconstruct_candidates_and_independent_deltas(
    h5_evaluation: H5GateEvaluation,
) -> None:
    cases = h5_evaluation.result.positive_cases
    attempts = h5_evaluation.positive_attempts
    oracle_results = h5_evaluation.oracle_results
    assert cases is not None and attempts is not None and oracle_results is not None
    assert tuple(case.outcome for case in cases) == attempts

    expected_fields = {
        H5UpdateRule.EXACT_Z0: ("exact_z0.mean", "exact_z0.variance"),
        H5UpdateRule.EXACT_SOURCE_ROW_A2: (
            "exact_source_row_a2.probability[0]",
            "exact_source_row_a2.probability[1]",
        ),
        H5UpdateRule.EXACT_STATE_TRANSITION_2_M: (
            "exact_state_transition_2_m.alpha_0",
            "exact_state_transition_2_m.alpha_1",
            "exact_state_transition_2_m.B_base",
            "exact_state_transition_2_m.c",
            "exact_state_transition_2_m.R",
        ),
    }
    for case, oracle in zip(cases, oracle_results, strict=True):
        outcome = case.outcome
        assert isinstance(outcome, CompletedUpdateAttempt)
        assert outcome.request.request_id == case.case_id.value
        assert outcome.request.rule is H5_POSITIVE_RULE_BY_ID[case.case_id]
        assert case.passed and case.delta_agreement.passed
        assert case.delta_agreement.rule is outcome.request.rule
        assert case.production_semantic_state_sha256 == outcome.after.evaluated_state_sha256
        assert case.production_semantic_state_sha256 != outcome.hashes.candidate_sha256
        assert case.oracle_semantic_state_sha256 == oracle.semantic_state_sha256
        if outcome.request.rule in expected_fields:
            comparison = case.candidate_comparison
            assert comparison is not None and comparison.passed
            assert tuple(item.field_id for item in comparison.scalar_comparisons) == expected_fields[outcome.request.rule]
            for scalar in comparison.scalar_comparisons:
                assert scalar.operation_count == H5_CANDIDATE_COMPARISON_OPERATION_COUNTS[scalar.field_id]
                expected_production = H5_C * gamma_n(scalar.operation_count) * max(1.0, scalar.production_condition_number) * max(1.0, abs(scalar.production_value))
                expected_oracle = H5_C * gamma_n(scalar.operation_count) * max(1.0, scalar.oracle_condition_number) * max(1.0, abs(scalar.oracle_value))
                expected_comparison = H5_C * gamma_n(3) * max(1.0, abs(scalar.production_value), abs(scalar.oracle_value), abs(scalar.production_value) + abs(scalar.oracle_value))
                assert scalar.production_rounding.hex() == float(expected_production).hex()
                assert scalar.oracle_rounding.hex() == float(expected_oracle).hex()
                assert scalar.comparison_rounding.hex() == float(expected_comparison).hex()
                assert scalar.allowance.hex() == math.fsum((expected_production, expected_oracle, expected_comparison)).hex()
                assert scalar.absolute_error == abs(scalar.production_value - scalar.oracle_value)
        else:
            assert case.candidate_comparison is None

        agreement = case.delta_agreement
        assert agreement.production is not agreement.oracle
        assert agreement.production.before is not agreement.oracle.before
        assert agreement.production.after is not agreement.oracle.after
        assert agreement.production.delta is not agreement.oracle.delta
        assert agreement.production.operand_shaped is True
        assert agreement.oracle.operand_shaped is True
        assert agreement.production.before.value == outcome.before.terms.complete_elbo
        assert agreement.production.after.value == outcome.after.terms.complete_elbo
        assert agreement.production.delta.value == outcome.delta_elbo
        assert agreement.production.delta.allowance.hex() == outcome.allowance.epsilon_delta.hex()
        assert agreement.oracle.before.value == oracle.before.value
        assert agreement.oracle.after.value == oracle.after.value
        assert agreement.oracle.delta.value == oracle.delta.value
        assert agreement.oracle.delta.allowance.hex() == oracle.delta.allowance.hex()
        expected_rounding = H5_C * gamma_n(3) * max(
            1.0,
            abs(agreement.production.delta.value),
            abs(agreement.oracle.delta.value),
            abs(agreement.production.delta.value) + abs(agreement.oracle.delta.value),
        )
        assert agreement.comparison_rounding.hex() == float(expected_rounding).hex()
        assert agreement.allowance.hex() == math.fsum(
            (
                agreement.production.delta.allowance,
                agreement.oracle.delta.allowance,
                agreement.comparison_rounding,
            )
        ).hex()
        assert agreement.passed is (agreement.absolute_error <= agreement.allowance)

    natural = cases[-1].outcome
    assert isinstance(natural, CompletedUpdateAttempt) and not natural.accepted
    assert natural.hashes.final_live_sha256 == natural.hashes.before_live_sha256
    assert natural.hashes.final_recognition_sha256 == natural.hashes.before_recognition_sha256
    assert natural.hashes.final_model_sha256 == natural.hashes.before_model_sha256
    assert natural.hashes.final_optimizer_sha256 == natural.hashes.before_optimizer_sha256
    assert natural.hashes.final_rng_sha256 == natural.hashes.before_rng_sha256

    first = cases[0]
    changed_oracle_hash = "f" * 64 if first.oracle_semantic_state_sha256 != "f" * 64 else "e" * 64
    assert replace(first, oracle_semantic_state_sha256=changed_oracle_hash).passed
    with pytest.raises(ValueError, match="semantic"):
        replace(
            first,
            production_semantic_state_sha256=first.outcome.hashes.candidate_sha256,
        )


def test_seven_controls_have_fresh_requests_and_exact_typed_detections(
    h5_evaluation: H5GateEvaluation,
) -> None:
    controls = h5_evaluation.controls
    assert controls is not None
    expected_failures = {
        H5ControlId.OMIT_CHILD: (
            AttemptPhase.AFTER_EVALUATION,
            AttemptFailureReason.FACTOR_COVERAGE_MISMATCH,
        ),
        H5ControlId.OMIT_EMISSION: (
            AttemptPhase.AFTER_EVALUATION,
            AttemptFailureReason.FACTOR_COVERAGE_MISMATCH,
        ),
        H5ControlId.FORCE_UNRESOLVED_GEM: (
            AttemptPhase.DECISION,
            AttemptFailureReason.DECISION_POLICY_VIOLATION,
        ),
        H5ControlId.MISLABEL_NATURAL: (
            AttemptPhase.FREEZE,
            AttemptFailureReason.LABEL_PROVENANCE_MISMATCH,
        ),
        H5ControlId.MUTATE_REJECTION: (
            AttemptPhase.COMMIT_OR_ROLLBACK,
            AttemptFailureReason.ROLLBACK_HASH_MISMATCH,
        ),
        H5ControlId.CHANGED_VALUE_SAME_INPUT: (
            AttemptPhase.AFTER_EVALUATION,
            AttemptFailureReason.DETERMINISTIC_REEVALUATION_MISMATCH,
        ),
    }
    positive_hashes = {
        item.outcome.request.request_sha256
        for item in h5_evaluation.result.positive_cases or ()
    }
    for control in controls:
        assert control.outcome.request.request_id == control.control_id.value
        assert control.outcome.request.rule is H5_CONTROL_BASE_RULE_BY_ID[control.control_id]
        assert control.outcome.request.request_sha256 not in positive_hashes
        assert control.expected_detection is H5_CONTROL_DETECTION_BY_ID[control.control_id]
        assert control.observed_detection is control.expected_detection
        assert control.passed
        if control.control_id in expected_failures:
            assert isinstance(control.outcome, FailedUpdateAttempt)
            assert (control.outcome.phase, control.outcome.reason) == expected_failures[control.control_id]

    omitted_child = controls[0].outcome
    omitted_emission = controls[1].outcome
    assert isinstance(omitted_child, FailedUpdateAttempt)
    assert isinstance(omitted_emission, FailedUpdateAttempt)
    assert omitted_child.missing_factor_ids == ("state_transition[2]",)
    assert omitted_emission.missing_factor_ids == ("emission[1]",)
    assert len(omitted_child.partial_after.observed_records) == 11
    assert len(omitted_emission.partial_after.observed_records) == 11

    mislabel = controls[3].outcome
    assert isinstance(mislabel, FailedUpdateAttempt)
    assert mislabel.hashes.candidate_draft_sha256 is not None
    assert mislabel.hashes.candidate_recognition_sha256 is not None
    assert mislabel.hashes.candidate_model_sha256 is not None
    assert mislabel.hashes.candidate_sha256 is None
    assert mislabel.hashes.predecision_live_sha256 is None
    assert mislabel.hashes.predecision_optimizer_sha256 is None
    assert mislabel.hashes.predecision_rng_sha256 is None

    changed_input = controls[5].outcome
    assert isinstance(changed_input, CompletedUpdateAttempt)
    assert "state_transition[2]" in changed_input.observed_affected_factor_ids
    assert "state_transition[2]" not in changed_input.value_changed_factor_ids

    changed_value = controls[6].outcome
    assert isinstance(changed_value, FailedUpdateAttempt)
    assert "state_transition[2]" not in changed_value.observed_affected_factor_ids
    assert "state_transition[2]" in changed_value.value_changed_factor_ids
    assert changed_value.deterministic_reevaluation is not None
    assert changed_value.deterministic_reevaluation.matched is False


def test_acceptance_boundaries_are_closed_for_exact_and_strict_for_gem() -> None:
    epsilon = 0.125
    assert gate._acceptance_for_label(UpdateLabel.EXACT_COORDINATE, -epsilon, epsilon)
    assert gate._acceptance_for_label(UpdateLabel.EXACT_COORDINATE, epsilon, epsilon)
    assert not gate._acceptance_for_label(
        UpdateLabel.EXACT_COORDINATE,
        math.nextafter(-epsilon, -math.inf),
        epsilon,
    )
    assert not gate._acceptance_for_label(UpdateLabel.GENERALIZED_EM, -epsilon, epsilon)
    assert not gate._acceptance_for_label(UpdateLabel.GENERALIZED_EM, epsilon, epsilon)
    assert gate._acceptance_for_label(
        UpdateLabel.GENERALIZED_EM,
        math.nextafter(epsilon, math.inf),
        epsilon,
    )


def test_emission_gem_closed_allowance_band_is_inconclusive_before_failure() -> None:
    epsilon = 0.125
    for delta in (-epsilon, -0.0, 0.0, epsilon):
        assert gate._emission_touching_gem_delta_is_indecisive(delta, epsilon)
    assert not gate._emission_touching_gem_delta_is_indecisive(
        math.nextafter(-epsilon, -math.inf), epsilon
    )
    assert not gate._emission_touching_gem_delta_is_indecisive(
        math.nextafter(epsilon, math.inf), epsilon
    )

    for simultaneous_decisive_failure in (False, True):
        status, obligations = gate._classify_completed_h5_evidence(
            gem_delta_indecisive=True,
            simultaneous_decisive_failure=simultaneous_decisive_failure,
        )
        assert status is GateStatus.INCONCLUSIVE
        assert obligations == (
            "resolve emission-touching GEM delta outside complete allowance",
        )

    assert gate._classify_completed_h5_evidence(
        gem_delta_indecisive=False,
        simultaneous_decisive_failure=True,
    ) == (GateStatus.FAIL, ())
    assert gate._classify_completed_h5_evidence(
        gem_delta_indecisive=False,
        simultaneous_decisive_failure=False,
    ) == (GateStatus.PASS, ())


def test_delta_nested_mutations_cannot_preserve_positive_gate_or_payload_pass(
    h5_evaluation: H5GateEvaluation,
) -> None:
    result = h5_evaluation.result
    positives = result.positive_cases or ()
    positive = positives[0]
    agreement = positive.delta_agreement
    production = agreement.production

    with pytest.raises(ValueError, match="operand_shaped"):
        replace(production, operand_shaped=False)

    for field_name in ("rounding", "allowance"):
        bad_before = replace(
            production.before,
            **{
                field_name: math.nextafter(
                    getattr(production.before, field_name), math.inf
                )
            },
        )
        bad_production = replace(production, before=bad_before)
        bad_agreement = replace(agreement, production=bad_production)
        with pytest.raises(ValueError, match="production"):
            replace(positive, delta_agreement=bad_agreement)

    corrupted_positive = copy.deepcopy(positive)
    object.__setattr__(
        corrupted_positive.delta_agreement.production,
        "operand_shaped",
        False,
    )
    with pytest.raises(ValueError, match="positive|operand_shaped"):
        replace(result, positive_cases=(corrupted_positive, *positives[1:]))

    exact_positive = copy.deepcopy(positive)
    scalar = exact_positive.candidate_comparison.scalar_comparisons[0]
    object.__setattr__(
        scalar,
        "allowance",
        math.nextafter(scalar.allowance, math.inf),
    )
    with pytest.raises(ValueError, match="positive|allowance"):
        replace(result, positive_cases=(exact_positive, *positives[1:]))

    drifted_invariant = replace(
        result.invariants[0], detail="fabricated invariant detail"
    )
    with pytest.raises(ValueError, match="invariant"):
        replace(result, invariants=(drifted_invariant, *result.invariants[1:]))

    oracle_corruption = copy.deepcopy(positive)
    oracle_before = oracle_corruption.delta_agreement.oracle.before
    object.__setattr__(
        oracle_before,
        "allowance",
        math.nextafter(oracle_before.allowance, math.inf),
    )
    corrupted_result = replace(
        result,
        positive_cases=(oracle_corruption, *positives[1:]),
    )
    with pytest.raises(ValueError, match="oracle|delta|payload"):
        replace(h5_evaluation.validation_payload, result=corrupted_result)


def test_production_complete_allowance_rejects_regrouped_and_ulp_mutations(
    h5_evaluation: H5GateEvaluation,
) -> None:
    positive = (h5_evaluation.result.positive_cases or ())[0]
    baseline = positive.outcome.before
    rebuilt = gate._production_complete_operand("before", baseline)
    assert rebuilt.allowance.hex() == baseline.complete_allowance.total.hex()

    for field_name in ("rounding_order_21", "total"):
        mutated = copy.deepcopy(baseline)
        term = mutated.term_allowances[0]
        object.__setattr__(
            term,
            field_name,
            math.nextafter(getattr(term, field_name), math.inf),
        )
        with pytest.raises(ValueError, match="allowance|rounding|recomput"):
            gate._production_complete_operand("before", mutated)

    mutated = copy.deepcopy(baseline)
    object.__setattr__(
        mutated.complete_allowance,
        "reduction_rounding",
        math.nextafter(mutated.complete_allowance.reduction_rounding, math.inf),
    )
    with pytest.raises(ValueError, match="allowance|rounding|recomput"):
        gate._production_complete_operand("before", mutated)

    convergence = math.fsum(
        item.convergence_estimate for item in baseline.term_allowances
    )
    rounding = math.fsum(
        tuple(
            number
            for item in baseline.term_allowances
            for number in (
                item.rounding_order_21,
                item.rounding_order_17,
                item.comparison_rounding,
            )
        )
        + (baseline.complete_allowance.reduction_rounding,)
    )
    regrouped = math.fsum((convergence, rounding))
    assert regrouped.hex() != baseline.complete_allowance.total.hex()
    mutated = copy.deepcopy(baseline)
    object.__setattr__(mutated.complete_allowance, "total", regrouped)
    with pytest.raises(ValueError, match="allowance|total|recomput"):
        gate._production_complete_operand("before", mutated)

    for direction in (-math.inf, math.inf):
        mutated = copy.deepcopy(baseline)
        object.__setattr__(
            mutated.complete_allowance,
            "total",
            math.nextafter(mutated.complete_allowance.total, direction),
        )
        with pytest.raises(ValueError, match="allowance|total|recomput"):
            gate._production_complete_operand("before", mutated)


def test_gate_status_precedence_and_record_contradictions_fail_closed(
    h5_evaluation: H5GateEvaluation,
) -> None:
    result = h5_evaluation.result
    finite_failure = InvariantResult(
        result.invariants[0].name, False, 0.0, 1.0, "finite failure"
    )
    unavailable = InvariantResult(
        result.invariants[1].name, False, None, None, "unavailable"
    )
    mixed = (finite_failure, unavailable, *result.invariants[2:])
    with pytest.raises(ValueError, match="invariant"):
        replace(
            result,
            status=GateStatus.INCONCLUSIVE,
            invariants=mixed,
            obligations=("resolve unavailable invariant",),
        )
    with pytest.raises(ValueError, match="invariant"):
        replace(
            result,
            status=GateStatus.FAIL,
            invariants=(finite_failure, *result.invariants[1:]),
        )
    with pytest.raises(ValueError):
        replace(result, status=GateStatus.FAIL)
    with pytest.raises(ValueError):
        replace(result, status=GateStatus.INCONCLUSIVE, obligations=("invented",))
    with pytest.raises(ValueError):
        replace(
            result,
            status=GateStatus.PASS,
            invariants=(finite_failure, *result.invariants[1:]),
        )

    first = (result.positive_cases or ())[0]
    with pytest.raises(ValueError, match="case"):
        replace(first, case_id=H5PositiveCaseId.ACCEPTED_GEM)
    first_control = (result.controls or ())[0]
    with pytest.raises(ValueError, match="detection"):
        replace(
            first_control,
            observed_detection=H5ControlDetection.EMISSION_FACTOR_COVERAGE_FAILURE,
            passed=True,
        )


def _assert_inconclusive_preflight(
    evaluation: H5GateEvaluation,
    *,
    phase: H5PreflightPhase,
    kind: H5PreflightErrorKind,
    h1_bytes: bytes,
    h5_bytes: bytes,
) -> None:
    result = evaluation.result
    assert result.status is GateStatus.INCONCLUSIVE
    assert result.preflight.phase is phase
    assert len(result.preflight.errors) == 1
    assert result.preflight.errors[0].kind is kind
    assert result.preflight.errors[0].phase is phase
    assert result.preflight.h1_fixture_raw_sha256 == hashlib.sha256(h1_bytes).hexdigest()
    assert result.preflight.update_spec_raw_sha256 == hashlib.sha256(h5_bytes).hexdigest()
    assert result.preflight.unavailable_fields == tuple(H5UnavailableField)
    assert result.preflight.obligation
    assert result.obligations == (result.preflight.obligation,)
    assert result.invariants == ()
    assert result.update_spec_canonical_sha256 is None
    assert result.objective_schema_sha256 is None
    assert result.factor_input_schema_version is None
    assert result.factor_input_schema_sha256 is None
    assert result.reference_sha256 is None
    assert result.positive_cases is None
    assert result.controls is None
    assert evaluation.reference is None
    assert evaluation.positive_attempts is None
    assert evaluation.controls is None
    assert evaluation.oracle_results is None
    assert evaluation.validation_payload.nonclaims == H5_NONCLAIM_IDS

    payload = h5_validation_payload(evaluation)
    assert payload["reference_sha256"] is None
    assert payload["factor_universe"] is None
    assert payload["recognition_coordinate_universe"] is None
    assert payload["model_block_universe"] is None
    assert payload["variable_dependency_rows"] is None
    assert payload["parameter_dependency_rows"] is None
    assert payload["positive_attempts"] is None
    assert payload["controls"] is None
    assert payload["oracle_results"] is None
    assert tuple(payload["nonclaims"]) == H5_NONCLAIM_IDS
    assert payload["result"]["update_spec_canonical_sha256"] is None
    assert payload["result"]["positive_cases"] is None


def test_preflight_raw_digest_failures_are_typed_without_fabrication(
    h5_config: ResolvedConfig,
) -> None:
    invalid_h1 = b"invalid-h1"
    evaluation = evaluate_h5(
        h5_config,
        h1_fixture_bytes=invalid_h1,
        h5_update_spec_bytes=H5_BYTES,
    )
    _assert_inconclusive_preflight(
        evaluation,
        phase=H5PreflightPhase.H1_FIXTURE_VALIDATION,
        kind=H5PreflightErrorKind.INVALID_H1_FIXTURE,
        h1_bytes=invalid_h1,
        h5_bytes=H5_BYTES,
    )

    invalid_h5 = b"invalid-h5"
    evaluation = evaluate_h5(
        h5_config,
        h1_fixture_bytes=H1_BYTES,
        h5_update_spec_bytes=invalid_h5,
    )
    _assert_inconclusive_preflight(
        evaluation,
        phase=H5PreflightPhase.UPDATE_SPEC_VALIDATION,
        kind=H5PreflightErrorKind.UPDATE_SPEC_RAW_DIGEST_MISMATCH,
        h1_bytes=H1_BYTES,
        h5_bytes=invalid_h5,
    )


def test_preflight_schema_and_reference_failures_are_phase_typed(
    h5_config: ResolvedConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = (
        (
            "parse_h5_update_spec_bytes",
            lambda _value: (_ for _ in ()).throw(ValueError("malformed schema")),
            H5PreflightErrorKind.INVALID_UPDATE_SPEC_SCHEMA,
        ),
        (
            "build_h5_reference_state",
            lambda *_values: (_ for _ in ()).throw(ValueError("reference failed")),
            H5PreflightErrorKind.REFERENCE_CONSTRUCTION_FAILED,
        ),
    )
    for name, replacement, kind in cases:
        with monkeypatch.context() as scoped:
            scoped.setattr(gate, name, replacement)
            evaluation = evaluate_h5(
                h5_config,
                h1_fixture_bytes=H1_BYTES,
                h5_update_spec_bytes=H5_BYTES,
            )
        phase = (
            H5PreflightPhase.UPDATE_SPEC_VALIDATION
            if kind is H5PreflightErrorKind.INVALID_UPDATE_SPEC_SCHEMA
            else H5PreflightPhase.REFERENCE_CONSTRUCTION
        )
        _assert_inconclusive_preflight(
            evaluation,
            phase=phase,
            kind=kind,
            h1_bytes=H1_BYTES,
            h5_bytes=H5_BYTES,
        )

    for name, kind in (
        ("H5_OBJECTIVE_SCHEMA_SHA256", H5PreflightErrorKind.OBJECTIVE_SCHEMA_IDENTITY_FAILED),
        ("H5_FACTOR_INPUT_SCHEMA_SHA256", H5PreflightErrorKind.FACTOR_INPUT_SCHEMA_IDENTITY_FAILED),
    ):
        with monkeypatch.context() as scoped:
            scoped.setattr(gate, name, "0" * 64)
            evaluation = evaluate_h5(
                h5_config,
                h1_fixture_bytes=H1_BYTES,
                h5_update_spec_bytes=H5_BYTES,
            )
        _assert_inconclusive_preflight(
            evaluation,
            phase=H5PreflightPhase.REFERENCE_CONSTRUCTION,
            kind=kind,
            h1_bytes=H1_BYTES,
            h5_bytes=H5_BYTES,
        )


def test_post_preflight_candidate_reconstruction_mismatch_is_ready_inconclusive(
    h5_config: ResolvedConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with monkeypatch.context() as scoped:
        scoped.setattr(gate, "h5_semantic_state_sha256", lambda *_values: "0" * 64)
        evaluation = evaluate_h5(
            h5_config,
            h1_fixture_bytes=H1_BYTES,
            h5_update_spec_bytes=H5_BYTES,
        )

    result = evaluation.result
    assert result.status is GateStatus.INCONCLUSIVE
    assert result.preflight.phase is H5PreflightPhase.READY
    assert result.preflight.errors == ()
    assert result.preflight.unavailable_fields == ()
    assert result.preflight.obligation is None
    assert result.update_spec_canonical_sha256 is not None
    assert result.objective_schema_sha256 == H5_OBJECTIVE_SCHEMA_SHA256
    assert result.factor_input_schema_version == H5_FACTOR_INPUT_SCHEMA_VERSION
    assert result.factor_input_schema_sha256 == H5_FACTOR_INPUT_SCHEMA_SHA256
    assert result.reference_sha256 is not None
    assert result.positive_cases is None
    assert result.controls is None
    assert result.invariants == ()
    assert result.obligations == (
        "resolve deterministic H5 candidate reconstruction same-domain hash agreement",
    )

    assert evaluation.reference is not None
    assert evaluation.reference.reference_sha256 == result.reference_sha256
    assert evaluation.positive_attempts is None
    assert evaluation.controls is None
    assert evaluation.oracle_results is None
    payload = evaluation.validation_payload
    assert payload.reference_sha256 == result.reference_sha256
    assert payload.factor_universe == H5_FACTOR_UNIVERSE
    assert payload.recognition_coordinate_universe == H5_RECOGNITION_COORDINATE_UNIVERSE
    assert payload.model_block_universe == H5_MODEL_BLOCK_UNIVERSE
    assert payload.variable_dependency_rows == H5_VARIABLE_DEPENDENCY_ROWS
    assert payload.parameter_dependency_rows == H5_PARAMETER_DEPENDENCY_ROWS
    assert payload.positive_attempts is None
    assert payload.controls is None
    assert payload.oracle_results is None
    assert payload.nonclaims == H5_NONCLAIM_IDS


def test_payload_is_byte_bound_complete_and_rejects_shortened_nonclaims(
    h5_evaluation: H5GateEvaluation,
) -> None:
    record = h5_evaluation.validation_payload
    assert record.reference_sha256 == h5_evaluation.reference.reference_sha256
    assert record.factor_universe == H5_FACTOR_UNIVERSE
    assert record.recognition_coordinate_universe == H5_RECOGNITION_COORDINATE_UNIVERSE
    assert record.model_block_universe == H5_MODEL_BLOCK_UNIVERSE
    assert record.variable_dependency_rows == H5_VARIABLE_DEPENDENCY_ROWS
    assert record.parameter_dependency_rows == H5_PARAMETER_DEPENDENCY_ROWS
    assert record.nonclaims == H5_NONCLAIM_IDS
    assert record.payload_sha256 == hashlib.sha256(
        H5_VALIDATION_PAYLOAD_DOMAIN + record.canonical_bytes
    ).hexdigest()
    decoded = json.loads(record.canonical_bytes)
    assert decoded["reference_sha256"] == record.reference_sha256
    payload = h5_validation_payload(h5_evaluation)
    assert payload["payload_sha256"] == record.payload_sha256
    assert "canonical_bytes" not in payload
    assert tuple(payload) == tuple(
        field.name
        for field in fields(H5ValidationPayloadRecord)
        if field.name != "canonical_bytes"
    )
    with pytest.raises(ValueError, match="nonclaims"):
        replace(record, nonclaims=H5_NONCLAIM_IDS[:-1])
