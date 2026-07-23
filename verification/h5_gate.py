"""Fail-closed H5 update-coherence promotion gate."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum
from fractions import Fraction
from types import MappingProxyType
from typing import Literal

from verification.numpy_oracles.h5_updates import (
    H5OracleOperandEvidence,
    H5OracleUpdate,
    oracle_complete_delta,
    oracle_exact_e_block,
    oracle_exact_m_block,
    oracle_exact_source_row,
)
from vfe4.config.schema import ResolvedConfig
from vfe4.inference.h5_updates import (
    AttemptFailureReason,
    AttemptPhase,
    CompletedUpdateAttempt,
    FailedUpdateAttempt,
    H5AttemptOutcome,
    H5FaultInjection,
    H5FaultKind,
    execute_update,
    exact_conjugate_gaussian_e_update,
    exact_gaussian_m_update,
    exact_source_row_update,
    propose_generalized_em,
    propose_natural_gradient,
)
from vfe4.numerics.h5_budget import (
    DEFAULT_H5_BUDGET_CONFIG,
    H5CompleteAllowance,
    H5DeltaAllowance,
    H5TermAllowance,
)
from vfe4.objective.dependency_graph import (
    FactorDependencyGraph,
    build_h5_reference_dependency_graph,
)
from vfe4.objective.h5_complete import (
    CompleteElboEvaluation,
    evaluate_h5_complete_elbo,
)
from vfe4.types.h5_schema import (
    H5_C,
    H5_CANDIDATE_COMPARISON_OPERATION_COUNTS,
    H5_FACTOR_INPUT_SCHEMA_DOMAIN,
    H5_FACTOR_INPUT_SCHEMA_SHA256,
    H5_FACTOR_INPUT_SCHEMA_VERSION,
    H5_FACTOR_UNIVERSE,
    H5_H1_FIXTURE_RAW_SHA256,
    H5_MODEL_SNAPSHOT_DOMAIN,
    H5_MODEL_BLOCK_UNIVERSE,
    H5_NONCLAIM_IDS,
    H5_OBJECTIVE_SCHEMA_DOMAIN,
    H5_OBJECTIVE_SCHEMA_SHA256,
    H5_PARAMETER_DEPENDENCY_ROWS,
    H5_RECOGNITION_SNAPSHOT_DOMAIN,
    H5_RECOGNITION_COORDINATE_UNIVERSE,
    H5_REFERENCE_STATE_DOMAIN,
    H5_UPDATE_SPEC_DOMAIN,
    H5_VALIDATION_PAYLOAD_DOMAIN,
    H5_VARIABLE_DEPENDENCY_ROWS,
    canonical_h5_factor_input_schema_core_bytes,
    canonical_h5_objective_schema_core_bytes,
    gamma_n,
)
from vfe4.types.results import GateStatus, InvariantResult
from vfe4.types.updates import (
    H5_RULE_CONTRACTS,
    H5CandidateSnapshot,
    H5ReferenceState,
    H5UpdateRule,
    UpdateLabel,
    UpdateRequest,
    canonical_h5_model_snapshot_bytes,
    canonical_h5_recognition_snapshot_bytes,
    canonical_h5_reference_state_bytes,
    canonical_h5_semantic_state_bytes,
    h5_semantic_state_sha256,
    initial_live,
)
from vfe4.validation.h5_update_spec import (
    EXPECTED_H5_UPDATE_SPEC_RAW_SHA256,
    build_h5_reference_state,
    parse_h5_update_spec_bytes,
)


_LOWER_HEX = frozenset("0123456789abcdef")


class H5PositiveCaseId(str, Enum):
    EXACT_GAUSSIAN_E = "exact_gaussian_e_coordinate"
    EXACT_SOURCE_ROW = "exact_categorical_source_coordinate"
    EXACT_GAUSSIAN_M = "exact_gaussian_m_coordinate_fixed_recognition"
    ACCEPTED_GEM = "accepted_resolved_generalized_em"
    REJECTED_NATURAL = "rejected_proposal_rollback"


class H5ControlId(str, Enum):
    OMIT_CHILD = "child_factor_omission_detected"
    OMIT_EMISSION = "emission_factor_omission_detected"
    FORCE_UNRESOLVED_GEM = "unresolved_gem_acceptance_detected"
    MISLABEL_NATURAL = "natural_gradient_mislabel_detected"
    MUTATE_REJECTION = "rejection_mutation_detected"
    CHANGED_INPUT_EQUAL_VALUE = "changed_input_equal_value_detected"
    CHANGED_VALUE_SAME_INPUT = "changed_value_unchanged_input_not_affected"


class H5ControlDetection(str, Enum):
    CHILD_FACTOR_COVERAGE_FAILURE = "child_factor_coverage_failure"
    EMISSION_FACTOR_COVERAGE_FAILURE = "emission_factor_coverage_failure"
    UNRESOLVED_GEM_POLICY_FAILURE = "unresolved_gem_policy_failure"
    NATURAL_LABEL_PROVENANCE_FAILURE = "natural_label_provenance_failure"
    REJECTION_ROLLBACK_HASH_FAILURE = "rejection_rollback_hash_failure"
    INPUT_HASH_CHANGE_WITH_EQUAL_VALUE = "input_hash_change_with_equal_value"
    VALUE_CHANGE_WITH_SAME_INPUT = "value_change_with_same_input"


H5_CONTROL_DETECTION_BY_ID = MappingProxyType(
    dict(zip(H5ControlId, H5ControlDetection, strict=True))
)

H5_POSITIVE_RULE_BY_ID = MappingProxyType(
    {
        H5PositiveCaseId.EXACT_GAUSSIAN_E: H5UpdateRule.EXACT_Z0,
        H5PositiveCaseId.EXACT_SOURCE_ROW: H5UpdateRule.EXACT_SOURCE_ROW_A2,
        H5PositiveCaseId.EXACT_GAUSSIAN_M: H5UpdateRule.EXACT_STATE_TRANSITION_2_M,
        H5PositiveCaseId.ACCEPTED_GEM: H5UpdateRule.GENERALIZED_EM_EMISSION_1,
        H5PositiveCaseId.REJECTED_NATURAL: H5UpdateRule.NATURAL_GRADIENT_Z1,
    }
)

H5_CONTROL_BASE_RULE_BY_ID = MappingProxyType(
    {
        H5ControlId.OMIT_CHILD: H5UpdateRule.EXACT_Z0,
        H5ControlId.OMIT_EMISSION: H5UpdateRule.GENERALIZED_EM_EMISSION_1,
        H5ControlId.FORCE_UNRESOLVED_GEM: H5UpdateRule.GENERALIZED_EM_EMISSION_1,
        H5ControlId.MISLABEL_NATURAL: H5UpdateRule.NATURAL_GRADIENT_Z1,
        H5ControlId.MUTATE_REJECTION: H5UpdateRule.NATURAL_GRADIENT_Z1,
        H5ControlId.CHANGED_INPUT_EQUAL_VALUE: H5UpdateRule.EXACT_STATE_TRANSITION_2_M,
        H5ControlId.CHANGED_VALUE_SAME_INPUT: H5UpdateRule.GENERALIZED_EM_EMISSION_1,
    }
)


class H5PreflightPhase(str, Enum):
    H1_FIXTURE_VALIDATION = "h1_fixture_validation"
    UPDATE_SPEC_VALIDATION = "update_spec_validation"
    REFERENCE_CONSTRUCTION = "reference_construction"
    READY = "ready"


class H5PreflightErrorKind(str, Enum):
    INVALID_H1_FIXTURE = "invalid_h1_fixture"
    UPDATE_SPEC_RAW_DIGEST_MISMATCH = "update_spec_raw_digest_mismatch"
    INVALID_UPDATE_SPEC_SCHEMA = "invalid_update_spec_schema"
    REFERENCE_CONSTRUCTION_FAILED = "reference_construction_failed"
    OBJECTIVE_SCHEMA_IDENTITY_FAILED = "objective_schema_identity_failed"
    FACTOR_INPUT_SCHEMA_IDENTITY_FAILED = "factor_input_schema_identity_failed"


H5_PREFLIGHT_ERROR_KINDS_BY_PHASE = MappingProxyType(
    {
        H5PreflightPhase.H1_FIXTURE_VALIDATION: (
            H5PreflightErrorKind.INVALID_H1_FIXTURE,
        ),
        H5PreflightPhase.UPDATE_SPEC_VALIDATION: (
            H5PreflightErrorKind.UPDATE_SPEC_RAW_DIGEST_MISMATCH,
            H5PreflightErrorKind.INVALID_UPDATE_SPEC_SCHEMA,
        ),
        H5PreflightPhase.REFERENCE_CONSTRUCTION: (
            H5PreflightErrorKind.REFERENCE_CONSTRUCTION_FAILED,
            H5PreflightErrorKind.OBJECTIVE_SCHEMA_IDENTITY_FAILED,
            H5PreflightErrorKind.FACTOR_INPUT_SCHEMA_IDENTITY_FAILED,
        ),
        H5PreflightPhase.READY: (),
    }
)


class H5UnavailableField(str, Enum):
    UPDATE_SPEC_CANONICAL_SHA256 = "update_spec_canonical_sha256"
    OBJECTIVE_SCHEMA_SHA256 = "objective_schema_sha256"
    FACTOR_INPUT_SCHEMA_VERSION = "factor_input_schema_version"
    FACTOR_INPUT_SCHEMA_SHA256 = "factor_input_schema_sha256"
    REFERENCE = "reference"
    REFERENCE_SHA256 = "reference_sha256"
    FACTOR_UNIVERSE = "factor_universe"
    RECOGNITION_COORDINATE_UNIVERSE = "recognition_coordinate_universe"
    MODEL_BLOCK_UNIVERSE = "model_block_universe"
    VARIABLE_DEPENDENCY_ROWS = "variable_dependency_rows"
    PARAMETER_DEPENDENCY_ROWS = "parameter_dependency_rows"
    POSITIVE_CASES = "positive_cases"
    POSITIVE_ATTEMPTS = "positive_attempts"
    CONTROLS = "controls"
    ORACLE_RESULTS = "oracle_results"


H5_INVARIANT_NAMES = (
    "fixture_and_objective_schema_identity",
    "closed_update_taxonomy",
    "dependency_graph_complete",
    "exact_gaussian_e_coordinate",
    "exact_categorical_source_coordinate",
    "exact_gaussian_m_coordinate_fixed_recognition",
    "accepted_resolved_generalized_em",
    "rejected_proposal_rollback",
    "child_factor_omission_detected",
    "emission_factor_omission_detected",
    "unresolved_gem_acceptance_detected",
    "natural_gradient_mislabel_detected",
    "rejection_mutation_detected",
    "changed_input_equal_value_detected",
    "changed_value_unchanged_input_not_affected",
    "all_delta_allowances_operand_shaped",
)


def _sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase 64-hex SHA-256")
    return value


def _finite(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite binary64 float")
    return value


def _nonnegative(value: object, name: str) -> float:
    checked = _finite(value, name)
    if checked < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return checked


def _finite_tuple(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
) -> tuple[float, ...]:
    if type(value) is not tuple:
        raise ValueError(f"{name} must be a tuple")
    checked = tuple(_finite(item, f"{name}[{index}]") for index, item in enumerate(value))
    if minimum is not None and any(item < minimum for item in checked):
        raise ValueError(f"{name} entries must be at least {minimum}")
    return checked


def _canonicalize(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if type(value) is float:
        return _finite(value, "canonical float").hex()
    if type(value) in (str, int, bool) or value is None:
        return value
    if type(value) is bytes:
        return {"length": len(value), "hex": value.hex()}
    if type(value) is tuple:
        return [_canonicalize(item) for item in value]
    if isinstance(value, Mapping):
        if not all(type(key) is str and key for key in value):
            raise ValueError("canonical mappings require nonempty string keys")
        return {key: _canonicalize(value[key]) for key in sorted(value)}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _canonicalize(getattr(value, item.name))
            for item in fields(value)
        }
    raise ValueError(f"unsupported canonical H5 value: {type(value).__name__}")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        _canonicalize(value),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class H5PreflightError:
    phase: H5PreflightPhase
    kind: H5PreflightErrorKind
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.phase, H5PreflightPhase):
            raise ValueError("preflight error phase is invalid")
        if (
            not isinstance(self.kind, H5PreflightErrorKind)
            or self.kind not in H5_PREFLIGHT_ERROR_KINDS_BY_PHASE[self.phase]
        ):
            raise ValueError("preflight error kind does not belong to its phase")
        if type(self.detail) is not str or not self.detail:
            raise ValueError("preflight error detail must be nonempty")


@dataclass(frozen=True, slots=True)
class H5PreflightRecord:
    schema_version: Literal["h5-preflight-record-v1"]
    phase: H5PreflightPhase
    h1_fixture_raw_sha256: str
    update_spec_raw_sha256: str
    errors: tuple[H5PreflightError, ...]
    unavailable_fields: tuple[H5UnavailableField, ...]
    obligation: str | None

    def __post_init__(self) -> None:
        if self.schema_version != "h5-preflight-record-v1":
            raise ValueError("unsupported H5 preflight schema")
        if not isinstance(self.phase, H5PreflightPhase):
            raise ValueError("preflight phase is invalid")
        _sha256(self.h1_fixture_raw_sha256, "h1_fixture_raw_sha256")
        _sha256(self.update_spec_raw_sha256, "update_spec_raw_sha256")
        if type(self.errors) is not tuple or not all(
            isinstance(item, H5PreflightError) for item in self.errors
        ):
            raise ValueError("preflight errors must be a typed tuple")
        if type(self.unavailable_fields) is not tuple or not all(
            isinstance(item, H5UnavailableField) for item in self.unavailable_fields
        ):
            raise ValueError("unavailable_fields must be a typed tuple")
        if self.phase is H5PreflightPhase.READY:
            if self.errors or self.unavailable_fields or self.obligation is not None:
                raise ValueError("READY preflight cannot retain errors or unavailable fields")
        elif (
            len(self.errors) != 1
            or self.errors[0].phase is not self.phase
            or self.unavailable_fields != tuple(H5UnavailableField)
            or type(self.obligation) is not str
            or not self.obligation
        ):
            raise ValueError("failed preflight requires one typed error and full unavailability")


@dataclass(frozen=True, slots=True)
class H5DeltaOperandEvidence:
    schema_version: Literal["h5-delta-operand-evidence-v1"]
    operand: Literal["before", "after", "delta"]
    value: float
    operation_count: int
    condition_numbers: tuple[float, ...]
    absolute_summands: tuple[float, ...]
    rounding: float
    allowance: float

    def __post_init__(self) -> None:
        if self.schema_version != "h5-delta-operand-evidence-v1":
            raise ValueError("unsupported H5 delta-operand schema")
        if self.operand not in ("before", "after", "delta"):
            raise ValueError("delta operand role is invalid")
        _finite(self.value, "value")
        if type(self.operation_count) is not int or self.operation_count <= 0:
            raise ValueError("operation_count must be a positive integer")
        conditions = _finite_tuple(
            self.condition_numbers, "condition_numbers", minimum=1.0
        )
        summands = _finite_tuple(
            self.absolute_summands, "absolute_summands", minimum=0.0
        )
        rounding = _nonnegative(self.rounding, "rounding")
        allowance = _nonnegative(self.allowance, "allowance")
        if not conditions or not summands:
            raise ValueError("operand-shaped evidence requires local conditions and summands")
        if self.operand == "delta":
            expected_rounding = H5_C * gamma_n(3) * max(
                1.0,
                *summands,
                abs(self.value),
                summands[0] + summands[1],
            )
            if (
                self.operation_count != 3
                or conditions != (1.0,)
                or len(summands) != 2
                or rounding.hex() != float(expected_rounding).hex()
                or allowance < rounding
            ):
                raise ValueError("delta operand evidence does not match the frozen formula")
        object.__setattr__(self, "condition_numbers", conditions)
        object.__setattr__(self, "absolute_summands", summands)


@dataclass(frozen=True, slots=True)
class H5DeltaImplementationEvidence:
    schema_version: Literal["h5-delta-implementation-evidence-v1"]
    implementation: Literal["production", "oracle"]
    before: H5DeltaOperandEvidence
    after: H5DeltaOperandEvidence
    delta: H5DeltaOperandEvidence
    operand_shaped: bool

    def __post_init__(self) -> None:
        if self.schema_version != "h5-delta-implementation-evidence-v1":
            raise ValueError("unsupported H5 delta-implementation schema")
        if self.implementation not in ("production", "oracle"):
            raise ValueError("implementation must be production or oracle")
        if not all(
            isinstance(item, H5DeltaOperandEvidence)
            for item in (self.before, self.after, self.delta)
        ):
            raise ValueError("implementation evidence requires three typed operands")
        before, after, delta = tuple(
            replace(item) for item in (self.before, self.after, self.delta)
        )
        if (before.operand, after.operand, delta.operand) != (
            "before",
            "after",
            "delta",
        ):
            raise ValueError("implementation operand roles are not ordered")
        if self.operand_shaped is not True:
            raise ValueError("operand_shaped must be exactly true")
        expected_delta = float(after.value - before.value)
        expected_summands = (abs(before.value), abs(after.value))
        if (
            delta.value.hex() != expected_delta.hex()
            or delta.absolute_summands != expected_summands
        ):
            raise ValueError("delta operand must be derived from its local operands")
        object.__setattr__(self, "before", before)
        object.__setattr__(self, "after", after)
        object.__setattr__(self, "delta", delta)


@dataclass(frozen=True, slots=True)
class H5DeltaAgreement:
    schema_version: Literal["h5-delta-agreement-v1"]
    rule: H5UpdateRule
    production: H5DeltaImplementationEvidence
    oracle: H5DeltaImplementationEvidence
    comparison_rounding: float
    allowance: float
    absolute_error: float
    passed: bool

    def __post_init__(self) -> None:
        if self.schema_version != "h5-delta-agreement-v1":
            raise ValueError("unsupported H5 delta-agreement schema")
        if not isinstance(self.rule, H5UpdateRule):
            raise ValueError("rule must be an H5UpdateRule")
        if (
            not isinstance(self.production, H5DeltaImplementationEvidence)
            or not isinstance(self.oracle, H5DeltaImplementationEvidence)
            or self.production is self.oracle
        ):
            raise ValueError("delta agreement requires independent implementation records")
        if any(
            left is right
            for left, right in zip(
                (
                    self.production.before,
                    self.production.after,
                    self.production.delta,
                ),
                (self.oracle.before, self.oracle.after, self.oracle.delta),
                strict=True,
            )
        ):
            raise ValueError("production and oracle operands cannot share identity")
        production = replace(self.production)
        oracle = replace(self.oracle)
        if (
            production.implementation != "production"
            or oracle.implementation != "oracle"
        ):
            raise ValueError("delta agreement implementation roles are invalid")
        expected_comparison = H5_C * gamma_n(3) * max(
            1.0,
            abs(production.delta.value),
            abs(oracle.delta.value),
            abs(production.delta.value) + abs(oracle.delta.value),
        )
        expected_allowance = math.fsum(
            (
                production.delta.allowance,
                oracle.delta.allowance,
                expected_comparison,
            )
        )
        expected_error = abs(production.delta.value - oracle.delta.value)
        if _nonnegative(self.comparison_rounding, "comparison_rounding").hex() != float(
            expected_comparison
        ).hex():
            raise ValueError("delta comparison rounding is not locally shaped")
        if _nonnegative(self.allowance, "allowance").hex() != expected_allowance.hex():
            raise ValueError("delta agreement allowance is not recomputed")
        if _nonnegative(self.absolute_error, "absolute_error").hex() != expected_error.hex():
            raise ValueError("delta absolute_error is not recomputed")
        if type(self.passed) is not bool or self.passed is not (
            expected_error <= expected_allowance
        ):
            raise ValueError("delta agreement passed flag is inconsistent")
        object.__setattr__(self, "production", production)
        object.__setattr__(self, "oracle", oracle)


@dataclass(frozen=True, slots=True)
class H5CandidateScalarComparison:
    field_id: str
    production_value: float
    oracle_value: float
    operation_count: int
    production_condition_number: float
    oracle_condition_number: float
    production_rounding: float
    oracle_rounding: float
    comparison_rounding: float
    allowance: float
    absolute_error: float
    passed: bool

    def __post_init__(self) -> None:
        if self.field_id not in H5_CANDIDATE_COMPARISON_OPERATION_COUNTS:
            raise ValueError("candidate comparison field_id is outside H5")
        production = _finite(self.production_value, "production_value")
        oracle = _finite(self.oracle_value, "oracle_value")
        expected_count = H5_CANDIDATE_COMPARISON_OPERATION_COUNTS[self.field_id]
        if self.operation_count != expected_count:
            raise ValueError("candidate comparison operation count changed")
        production_condition = _finite(
            self.production_condition_number, "production_condition_number"
        )
        oracle_condition = _finite(
            self.oracle_condition_number, "oracle_condition_number"
        )
        if production_condition < 1.0 or oracle_condition < 1.0:
            raise ValueError("candidate conditions must be at least one")
        expected_production = (
            H5_C
            * gamma_n(expected_count)
            * max(1.0, production_condition)
            * max(1.0, abs(production))
        )
        expected_oracle = (
            H5_C
            * gamma_n(expected_count)
            * max(1.0, oracle_condition)
            * max(1.0, abs(oracle))
        )
        expected_comparison = H5_C * gamma_n(3) * max(
            1.0, abs(production), abs(oracle), abs(production) + abs(oracle)
        )
        expected_allowance = math.fsum(
            (expected_production, expected_oracle, expected_comparison)
        )
        expected_error = abs(production - oracle)
        for name, actual, expected in (
            ("production_rounding", self.production_rounding, expected_production),
            ("oracle_rounding", self.oracle_rounding, expected_oracle),
            ("comparison_rounding", self.comparison_rounding, expected_comparison),
            ("allowance", self.allowance, expected_allowance),
            ("absolute_error", self.absolute_error, expected_error),
        ):
            if _nonnegative(actual, name).hex() != float(expected).hex():
                raise ValueError(f"{name} is not recomputed from its operands")
        if type(self.passed) is not bool or self.passed is not (
            expected_error <= expected_allowance
        ):
            raise ValueError("candidate scalar passed flag is inconsistent")


_CANDIDATE_FIELDS_BY_RULE = MappingProxyType(
    {
        H5UpdateRule.EXACT_Z0: (
            "exact_z0.mean",
            "exact_z0.variance",
        ),
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
)


@dataclass(frozen=True, slots=True)
class H5CandidateComparison:
    rule: H5UpdateRule
    scalar_comparisons: tuple[H5CandidateScalarComparison, ...]
    max_absolute_error: float
    max_allowance: float
    passed: bool

    def __post_init__(self) -> None:
        if self.rule not in _CANDIDATE_FIELDS_BY_RULE:
            raise ValueError("only exact H5 rules have candidate comparisons")
        if type(self.scalar_comparisons) is not tuple or not all(
            isinstance(item, H5CandidateScalarComparison)
            for item in self.scalar_comparisons
        ):
            raise ValueError("scalar_comparisons must be a typed tuple")
        comparisons = tuple(replace(item) for item in self.scalar_comparisons)
        expected_fields = _CANDIDATE_FIELDS_BY_RULE[self.rule]
        if tuple(item.field_id for item in comparisons) != expected_fields:
            raise ValueError("candidate scalar field order changed")
        max_error = max(item.absolute_error for item in comparisons)
        max_allowance = max(item.allowance for item in comparisons)
        if _nonnegative(self.max_absolute_error, "max_absolute_error").hex() != max_error.hex():
            raise ValueError("max_absolute_error is not recomputed")
        if _nonnegative(self.max_allowance, "max_allowance").hex() != max_allowance.hex():
            raise ValueError("max_allowance is not recomputed")
        if type(self.passed) is not bool or self.passed is not all(
            item.passed for item in comparisons
        ):
            raise ValueError("candidate comparison passed flag is inconsistent")
        object.__setattr__(self, "scalar_comparisons", comparisons)


def _acceptance_for_label(label: UpdateLabel, delta: float, epsilon: float) -> bool:
    if not isinstance(label, UpdateLabel):
        raise ValueError("label must be an UpdateLabel")
    checked_delta = _finite(delta, "delta")
    checked_epsilon = _nonnegative(epsilon, "epsilon")
    if label is UpdateLabel.EXACT_COORDINATE:
        return checked_delta >= -checked_epsilon
    if label in (
        UpdateLabel.GENERALIZED_EM,
        UpdateLabel.NATURAL_GRADIENT_PROPOSAL,
    ):
        return checked_delta > checked_epsilon
    raise ValueError("label has no H5 v1 acceptance producer")


def _positive_outcome_passes(case_id: H5PositiveCaseId, outcome: H5AttemptOutcome) -> bool:
    if not isinstance(outcome, CompletedUpdateAttempt):
        return False
    if outcome.request.rule is not H5_POSITIVE_RULE_BY_ID[case_id]:
        return False
    expected_acceptance = _acceptance_for_label(
        outcome.request.requested_label,
        outcome.delta_elbo,
        outcome.allowance.epsilon_delta,
    )
    if outcome.accepted is not expected_acceptance:
        return False
    if case_id is H5PositiveCaseId.ACCEPTED_GEM:
        return outcome.accepted and outcome.delta_elbo > outcome.allowance.epsilon_delta
    if case_id is H5PositiveCaseId.REJECTED_NATURAL:
        hashes = outcome.hashes
        return (
            not outcome.accepted
            and outcome.delta_elbo < -outcome.allowance.epsilon_delta
            and hashes.final_live_sha256 == hashes.before_live_sha256
            and hashes.final_recognition_sha256 == hashes.before_recognition_sha256
            and hashes.final_model_sha256 == hashes.before_model_sha256
            and hashes.final_optimizer_sha256 == hashes.before_optimizer_sha256
            and hashes.final_rng_sha256 == hashes.before_rng_sha256
        )
    return outcome.accepted


@dataclass(frozen=True, slots=True)
class H5PositiveCaseResult:
    schema_version: Literal["h5-positive-case-result-v1"]
    case_id: H5PositiveCaseId
    outcome: H5AttemptOutcome
    production_semantic_state_sha256: str
    oracle_semantic_state_sha256: str
    candidate_comparison: H5CandidateComparison | None
    delta_agreement: H5DeltaAgreement
    passed: bool
    detail: str

    def __post_init__(self) -> None:
        if self.schema_version != "h5-positive-case-result-v1":
            raise ValueError("unsupported H5 positive-case schema")
        if not isinstance(self.case_id, H5PositiveCaseId):
            raise ValueError("case_id must be an H5PositiveCaseId")
        if not isinstance(self.outcome, (CompletedUpdateAttempt, FailedUpdateAttempt)):
            raise ValueError("positive case outcome is not typed")
        if self.outcome.request.request_id != self.case_id.value:
            raise ValueError("positive case request_id does not equal its case ID")
        if self.outcome.request.rule is not H5_POSITIVE_RULE_BY_ID[self.case_id]:
            raise ValueError("positive case rule does not match its case ID")
        production_semantic = _sha256(
            self.production_semantic_state_sha256,
            "production_semantic_state_sha256",
        )
        _sha256(self.oracle_semantic_state_sha256, "oracle_semantic_state_sha256")
        if not isinstance(self.outcome, CompletedUpdateAttempt):
            raise ValueError("positive cases require a completed typed outcome")
        if production_semantic != self.outcome.after.evaluated_state_sha256:
            raise ValueError("production semantic hash is not the evaluated candidate hash")
        exact = self.outcome.request.rule in _CANDIDATE_FIELDS_BY_RULE
        if exact:
            if (
                not isinstance(self.candidate_comparison, H5CandidateComparison)
                or self.candidate_comparison.rule is not self.outcome.request.rule
            ):
                raise ValueError("exact positive case requires its candidate comparison")
            candidate_comparison = replace(self.candidate_comparison)
        elif self.candidate_comparison is not None:
            raise ValueError("proposal positive cases cannot fabricate candidate comparisons")
        else:
            candidate_comparison = None
        if (
            not isinstance(self.delta_agreement, H5DeltaAgreement)
            or self.delta_agreement.rule is not self.outcome.request.rule
        ):
            raise ValueError("positive case delta agreement has the wrong rule")
        delta_agreement = replace(self.delta_agreement)
        expected_production = _production_delta_evidence(self.outcome)
        if delta_agreement.production != expected_production:
            raise ValueError(
                "positive production delta evidence failed independent reconstruction"
            )
        expected_pass = (
            _positive_outcome_passes(self.case_id, self.outcome)
            and delta_agreement.passed
            and (
                candidate_comparison is None
                or candidate_comparison.passed
            )
        )
        if type(self.passed) is not bool or self.passed is not expected_pass:
            raise ValueError("positive case passed flag is inconsistent")
        if type(self.detail) is not str or not self.detail:
            raise ValueError("positive case detail must be nonempty")
        object.__setattr__(self, "candidate_comparison", candidate_comparison)
        object.__setattr__(self, "delta_agreement", delta_agreement)


def _control_outcome_detection(
    control_id: H5ControlId,
    outcome: H5AttemptOutcome,
) -> H5ControlDetection | None:
    if control_id is H5ControlId.OMIT_CHILD:
        if (
            isinstance(outcome, FailedUpdateAttempt)
            and outcome.phase is AttemptPhase.AFTER_EVALUATION
            and outcome.reason is AttemptFailureReason.FACTOR_COVERAGE_MISMATCH
            and outcome.missing_factor_ids == ("state_transition[2]",)
            and outcome.partial_after is not None
            and len(outcome.partial_after.observed_records) == len(H5_FACTOR_UNIVERSE) - 1
        ):
            return H5ControlDetection.CHILD_FACTOR_COVERAGE_FAILURE
    elif control_id is H5ControlId.OMIT_EMISSION:
        if (
            isinstance(outcome, FailedUpdateAttempt)
            and outcome.phase is AttemptPhase.AFTER_EVALUATION
            and outcome.reason is AttemptFailureReason.FACTOR_COVERAGE_MISMATCH
            and outcome.missing_factor_ids == ("emission[1]",)
            and outcome.partial_after is not None
            and len(outcome.partial_after.observed_records) == len(H5_FACTOR_UNIVERSE) - 1
        ):
            return H5ControlDetection.EMISSION_FACTOR_COVERAGE_FAILURE
    elif control_id is H5ControlId.FORCE_UNRESOLVED_GEM:
        if (
            isinstance(outcome, FailedUpdateAttempt)
            and outcome.phase is AttemptPhase.DECISION
            and outcome.reason is AttemptFailureReason.DECISION_POLICY_VIOLATION
            and outcome.attempted_accept is True
            and outcome.decision_delta_elbo is not None
            and outcome.decision_epsilon_delta is not None
            and abs(outcome.decision_delta_elbo) <= outcome.decision_epsilon_delta
        ):
            return H5ControlDetection.UNRESOLVED_GEM_POLICY_FAILURE
    elif control_id is H5ControlId.MISLABEL_NATURAL:
        hashes = outcome.hashes
        if (
            isinstance(outcome, FailedUpdateAttempt)
            and outcome.phase is AttemptPhase.FREEZE
            and outcome.reason is AttemptFailureReason.LABEL_PROVENANCE_MISMATCH
            and hashes.candidate_draft_sha256 is not None
            and hashes.candidate_recognition_sha256 is not None
            and hashes.candidate_model_sha256 is not None
            and hashes.candidate_sha256 is None
            and hashes.predecision_live_sha256 is None
            and hashes.predecision_optimizer_sha256 is None
            and hashes.predecision_rng_sha256 is None
            and hashes.final_live_sha256 == hashes.before_live_sha256
            and hashes.final_recognition_sha256 == hashes.before_recognition_sha256
            and hashes.final_model_sha256 == hashes.before_model_sha256
            and hashes.final_optimizer_sha256 == hashes.before_optimizer_sha256
            and hashes.final_rng_sha256 == hashes.before_rng_sha256
        ):
            return H5ControlDetection.NATURAL_LABEL_PROVENANCE_FAILURE
    elif control_id is H5ControlId.MUTATE_REJECTION:
        hashes = outcome.hashes
        if (
            isinstance(outcome, FailedUpdateAttempt)
            and outcome.phase is AttemptPhase.COMMIT_OR_ROLLBACK
            and outcome.reason is AttemptFailureReason.ROLLBACK_HASH_MISMATCH
            and hashes.final_live_sha256 != hashes.before_live_sha256
            and hashes.final_recognition_sha256 != hashes.before_recognition_sha256
            and hashes.final_rng_sha256 != hashes.before_rng_sha256
            and hashes.final_model_sha256 == hashes.before_model_sha256
            and hashes.final_optimizer_sha256 == hashes.before_optimizer_sha256
        ):
            return H5ControlDetection.REJECTION_ROLLBACK_HASH_FAILURE
    elif control_id is H5ControlId.CHANGED_INPUT_EQUAL_VALUE:
        if isinstance(outcome, CompletedUpdateAttempt):
            before = next(
                item
                for item in outcome.before.factor_records
                if item.factor_id == "state_transition[2]"
            )
            after = next(
                item
                for item in outcome.after.factor_records
                if item.factor_id == "state_transition[2]"
            )
            if (
                before.input_hash.input_sha256 != after.input_hash.input_sha256
                and before.value_order_21.hex() == after.value_order_21.hex()
                and before.value_order_17.hex() == after.value_order_17.hex()
                and "state_transition[2]" in outcome.observed_affected_factor_ids
                and "state_transition[2]" not in outcome.value_changed_factor_ids
            ):
                return H5ControlDetection.INPUT_HASH_CHANGE_WITH_EQUAL_VALUE
    elif control_id is H5ControlId.CHANGED_VALUE_SAME_INPUT:
        if (
            isinstance(outcome, FailedUpdateAttempt)
            and outcome.phase is AttemptPhase.AFTER_EVALUATION
            and outcome.reason is AttemptFailureReason.DETERMINISTIC_REEVALUATION_MISMATCH
            and outcome.deterministic_reevaluation is not None
            and not outcome.deterministic_reevaluation.matched
            and "state_transition[2]" not in outcome.observed_affected_factor_ids
            and "state_transition[2]" in outcome.value_changed_factor_ids
        ):
            before = next(
                item
                for item in outcome.before.factor_records
                if item.factor_id == "state_transition[2]"
            )
            reported = next(
                item
                for item in outcome.partial_after.observed_records
                if item.factor_id == "state_transition[2]"
            )
            recheck = outcome.deterministic_reevaluation
            if (
                reported.input_hash.input_sha256
                == before.input_hash.input_sha256
                == recheck.input_sha256
                and reported.value_order_21.hex()
                == float(before.value_order_21 + 1.0e-6).hex()
                and reported.value_order_17.hex()
                == float(before.value_order_17 + 1.0e-6).hex()
                and recheck.recomputed_value_order_21.hex()
                == before.value_order_21.hex()
                and recheck.recomputed_value_order_17.hex()
                == before.value_order_17.hex()
            ):
                return H5ControlDetection.VALUE_CHANGE_WITH_SAME_INPUT
    return None


@dataclass(frozen=True, slots=True)
class H5ControlResult:
    schema_version: Literal["h5-control-result-v1"]
    control_id: H5ControlId
    expected_detection: H5ControlDetection
    observed_detection: H5ControlDetection | None
    outcome: H5AttemptOutcome
    passed: bool
    detail: str

    def __post_init__(self) -> None:
        if self.schema_version != "h5-control-result-v1":
            raise ValueError("unsupported H5 control-result schema")
        if not isinstance(self.control_id, H5ControlId):
            raise ValueError("control_id must be an H5ControlId")
        if self.expected_detection is not H5_CONTROL_DETECTION_BY_ID[self.control_id]:
            raise ValueError("expected control detection does not match its ID")
        if self.observed_detection is not None and not isinstance(
            self.observed_detection, H5ControlDetection
        ):
            raise ValueError("observed_detection must be typed or None")
        if not isinstance(self.outcome, (CompletedUpdateAttempt, FailedUpdateAttempt)):
            raise ValueError("control outcome is not typed")
        if self.outcome.request.request_id != self.control_id.value:
            raise ValueError("control request_id does not equal its control ID")
        if self.outcome.request.rule is not H5_CONTROL_BASE_RULE_BY_ID[self.control_id]:
            raise ValueError("control request rule does not match its base rule")
        actual = _control_outcome_detection(self.control_id, self.outcome)
        if self.observed_detection is not actual:
            raise ValueError("observed detection does not match the typed control outcome")
        expected_pass = actual is self.expected_detection
        if type(self.passed) is not bool or self.passed is not expected_pass:
            raise ValueError("control passed flag is inconsistent with detection")
        if type(self.detail) is not str or not self.detail:
            raise ValueError("control detail must be nonempty")


H5_GEM_ALLOWANCE_OBLIGATION = (
    "resolve emission-touching GEM delta outside complete allowance"
)


def _emission_touching_gem_delta_is_indecisive(
    delta: float,
    epsilon: float,
) -> bool:
    checked_delta = _finite(delta, "delta")
    checked_epsilon = _nonnegative(epsilon, "epsilon")
    return -checked_epsilon <= checked_delta <= checked_epsilon


def _classify_completed_h5_evidence(
    *,
    gem_delta_indecisive: bool,
    simultaneous_decisive_failure: bool,
) -> tuple[GateStatus, tuple[str, ...]]:
    if type(gem_delta_indecisive) is not bool:
        raise ValueError("gem_delta_indecisive must be a bool")
    if type(simultaneous_decisive_failure) is not bool:
        raise ValueError("simultaneous_decisive_failure must be a bool")
    if gem_delta_indecisive:
        return GateStatus.INCONCLUSIVE, (H5_GEM_ALLOWANCE_OBLIGATION,)
    if simultaneous_decisive_failure:
        return GateStatus.FAIL, ()
    return GateStatus.PASS, ()


def _gem_positive_is_indecisive(
    positive_cases: tuple[H5PositiveCaseResult, ...],
) -> bool:
    gem = next(
        item
        for item in positive_cases
        if item.case_id is H5PositiveCaseId.ACCEPTED_GEM
    )
    return _emission_touching_gem_delta_is_indecisive(
        gem.outcome.delta_elbo,
        gem.outcome.allowance.epsilon_delta,
    )


def _completed_h5_invariants(
    positive_cases: tuple[H5PositiveCaseResult, ...],
    controls: tuple[H5ControlResult, ...],
) -> tuple[InvariantResult, ...]:
    taxonomy_pass = (
        tuple(H5_POSITIVE_RULE_BY_ID) == tuple(H5PositiveCaseId)
        and tuple(H5_CONTROL_BASE_RULE_BY_ID) == tuple(H5ControlId)
        and set(H5_POSITIVE_RULE_BY_ID.values()) == set(H5UpdateRule)
    )
    delta_pass = all(
        item.delta_agreement.passed
        and item.delta_agreement.production.operand_shaped
        and item.delta_agreement.oracle.operand_shaped
        for item in positive_cases
    )
    flags = (
        True,
        taxonomy_pass,
        True,
        *(item.passed for item in positive_cases),
        *(item.passed for item in controls),
        delta_pass,
    )
    return tuple(
        InvariantResult(
            name,
            passed,
            1.0 if passed else 0.0,
            1.0,
            "available finite typed H5 evidence",
        )
        for name, passed in zip(H5_INVARIANT_NAMES, flags, strict=True)
    )


@dataclass(frozen=True, slots=True)
class H5GateResult:
    schema_version: Literal["h5-gate-result-v1"]
    gate: Literal["H5"]
    status: GateStatus
    preflight: H5PreflightRecord
    h1_fixture_raw_sha256: str
    update_spec_raw_sha256: str
    update_spec_canonical_sha256: str | None
    objective_schema_sha256: str | None
    factor_input_schema_version: Literal["h5-factor-input-v1"] | None
    factor_input_schema_sha256: str | None
    reference_sha256: str | None
    positive_cases: tuple[H5PositiveCaseResult, ...] | None
    controls: tuple[H5ControlResult, ...] | None
    invariants: tuple[InvariantResult, ...]
    obligations: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != "h5-gate-result-v1" or self.gate != "H5":
            raise ValueError("H5 gate-result identity is frozen")
        if not isinstance(self.status, GateStatus) or not isinstance(
            self.preflight, H5PreflightRecord
        ):
            raise ValueError("H5 result status/preflight is invalid")
        if (
            _sha256(self.h1_fixture_raw_sha256, "h1_fixture_raw_sha256")
            != self.preflight.h1_fixture_raw_sha256
            or _sha256(self.update_spec_raw_sha256, "update_spec_raw_sha256")
            != self.preflight.update_spec_raw_sha256
        ):
            raise ValueError("gate raw digests must equal preflight digests")
        if type(self.invariants) is not tuple or not all(
            isinstance(item, InvariantResult) for item in self.invariants
        ):
            raise ValueError("invariants must be a typed tuple")
        if type(self.obligations) is not tuple or not all(
            type(item) is str and item for item in self.obligations
        ):
            raise ValueError("obligations must be nonempty strings")

        if self.preflight.phase is not H5PreflightPhase.READY:
            if (
                self.status is not GateStatus.INCONCLUSIVE
                or any(
                    item is not None
                    for item in (
                        self.update_spec_canonical_sha256,
                        self.objective_schema_sha256,
                        self.factor_input_schema_version,
                        self.factor_input_schema_sha256,
                        self.reference_sha256,
                        self.positive_cases,
                        self.controls,
                    )
                )
                or self.invariants != ()
                or self.obligations != (self.preflight.obligation,)
            ):
                raise ValueError("preflight failure must remain typed and unfabricated")
            return

        for name in (
            "update_spec_canonical_sha256",
            "objective_schema_sha256",
            "factor_input_schema_sha256",
            "reference_sha256",
        ):
            _sha256(getattr(self, name), name)
        if (
            self.objective_schema_sha256 != H5_OBJECTIVE_SCHEMA_SHA256
            or self.factor_input_schema_version != H5_FACTOR_INPUT_SCHEMA_VERSION
            or self.factor_input_schema_sha256 != H5_FACTOR_INPUT_SCHEMA_SHA256
        ):
            raise ValueError("ready H5 result schema identities are not frozen")
        if self.positive_cases is None or self.controls is None:
            if (
                self.positive_cases is not None
                or self.controls is not None
                or self.invariants != ()
                or self.status is not GateStatus.INCONCLUSIVE
                or not self.obligations
            ):
                raise ValueError(
                    "unavailable ready evidence requires one coherent INCONCLUSIVE branch"
                )
            return
        if (
            type(self.positive_cases) is not tuple
            or tuple(item.case_id for item in self.positive_cases)
            != tuple(H5PositiveCaseId)
            or not all(
                isinstance(item, H5PositiveCaseResult)
                for item in self.positive_cases
            )
        ):
            raise ValueError("ready positive cases must equal the typed H5 inventory")
        if (
            type(self.controls) is not tuple
            or tuple(item.control_id for item in self.controls) != tuple(H5ControlId)
            or not all(isinstance(item, H5ControlResult) for item in self.controls)
        ):
            raise ValueError("ready controls must equal the typed H5 inventory")
        positive_cases = tuple(replace(item) for item in self.positive_cases)
        controls = tuple(replace(item) for item in self.controls)
        expected_invariants = _completed_h5_invariants(positive_cases, controls)
        if self.invariants != expected_invariants:
            raise ValueError("H5 invariants must be reconstructed from nested evidence")
        gem_indecisive = _gem_positive_is_indecisive(positive_cases)
        expected_status, expected_obligations = _classify_completed_h5_evidence(
            gem_delta_indecisive=gem_indecisive,
            simultaneous_decisive_failure=any(
                not item.passed for item in expected_invariants
            ),
        )
        if (
            self.status is not expected_status
            or self.obligations != expected_obligations
        ):
            raise ValueError(
                "H5 status/obligations do not follow INCONCLUSIVE-before-FAIL precedence"
            )
        object.__setattr__(self, "positive_cases", positive_cases)
        object.__setattr__(self, "controls", controls)
        object.__setattr__(self, "invariants", expected_invariants)


def _payload_record_core(record: H5ValidationPayloadRecord) -> dict[str, object]:
    return {
        item.name: getattr(record, item.name)
        for item in fields(H5ValidationPayloadRecord)
        if item.name not in ("canonical_bytes", "payload_sha256")
    }


@dataclass(frozen=True, slots=True)
class H5ValidationPayloadRecord:
    schema_version: Literal[1]
    result: H5GateResult
    reference_sha256: str | None
    factor_universe: tuple[str, ...] | None
    recognition_coordinate_universe: tuple[str, ...] | None
    model_block_universe: tuple[str, ...] | None
    variable_dependency_rows: tuple[tuple[str, tuple[str, ...]], ...] | None
    parameter_dependency_rows: tuple[tuple[str, tuple[str, ...]], ...] | None
    positive_attempts: tuple[H5AttemptOutcome, ...] | None
    controls: tuple[H5ControlResult, ...] | None
    oracle_results: tuple[H5OracleUpdate, ...] | None
    nonclaims: tuple[str, ...]
    canonical_bytes: bytes = field(init=False, repr=False)
    payload_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != 1 or not isinstance(self.result, H5GateResult):
            raise ValueError("H5 validation payload identity is frozen")
        result = replace(self.result)
        object.__setattr__(self, "result", result)
        if self.nonclaims != H5_NONCLAIM_IDS:
            raise ValueError("H5 payload nonclaims must equal the frozen tuple")
        ready = result.preflight.phase is H5PreflightPhase.READY
        optionals = (
            self.reference_sha256,
            self.factor_universe,
            self.recognition_coordinate_universe,
            self.model_block_universe,
            self.variable_dependency_rows,
            self.parameter_dependency_rows,
            self.positive_attempts,
            self.controls,
            self.oracle_results,
        )
        if not ready:
            if any(item is not None for item in optionals):
                raise ValueError("preflight-inconclusive payload must use explicit nulls")
        else:
            _sha256(self.reference_sha256, "reference_sha256")
            if self.reference_sha256 != result.reference_sha256:
                raise ValueError("payload reference hash differs from its result")
            if (
                self.factor_universe != H5_FACTOR_UNIVERSE
                or self.recognition_coordinate_universe
                != H5_RECOGNITION_COORDINATE_UNIVERSE
                or self.model_block_universe != H5_MODEL_BLOCK_UNIVERSE
                or self.variable_dependency_rows != H5_VARIABLE_DEPENDENCY_ROWS
                or self.parameter_dependency_rows != H5_PARAMETER_DEPENDENCY_ROWS
            ):
                raise ValueError("payload universes/dependency rows changed")
            evidence_fields = (
                self.positive_attempts,
                self.controls,
                self.oracle_results,
            )
            if result.positive_cases is None or result.controls is None:
                if result.status is not GateStatus.INCONCLUSIVE or any(
                    item is not None for item in evidence_fields
                ):
                    raise ValueError(
                        "unavailable ready evidence must remain explicit nulls"
                    )
            elif (
                self.positive_attempts is None
                or self.controls is None
                or self.oracle_results is None
                or self.positive_attempts
                != tuple(item.outcome for item in result.positive_cases)
                or self.controls != result.controls
                or len(self.oracle_results) != len(result.positive_cases)
                or not all(isinstance(item, H5OracleUpdate) for item in self.oracle_results)
            ):
                raise ValueError("payload attempts/controls/oracles do not match the result")
            if result.positive_cases is not None:
                for positive, attempt, oracle in zip(
                    result.positive_cases,
                    self.positive_attempts,
                    self.oracle_results,
                    strict=True,
                ):
                    rebuilt_agreement = compare_h5_complete_delta(attempt, oracle)
                    if positive.delta_agreement != rebuilt_agreement:
                        raise ValueError(
                            "payload oracle delta evidence failed independent reconstruction"
                        )
            object.__setattr__(self, "factor_universe", tuple(self.factor_universe))
            object.__setattr__(
                self,
                "recognition_coordinate_universe",
                tuple(self.recognition_coordinate_universe),
            )
            object.__setattr__(self, "model_block_universe", tuple(self.model_block_universe))
            object.__setattr__(
                self,
                "variable_dependency_rows",
                tuple((name, tuple(values)) for name, values in self.variable_dependency_rows),
            )
            object.__setattr__(
                self,
                "parameter_dependency_rows",
                tuple((name, tuple(values)) for name, values in self.parameter_dependency_rows),
            )
            if self.positive_attempts is not None:
                object.__setattr__(self, "positive_attempts", tuple(self.positive_attempts))
            if self.controls is not None:
                object.__setattr__(self, "controls", tuple(self.controls))
            if self.oracle_results is not None:
                object.__setattr__(self, "oracle_results", tuple(self.oracle_results))
        object.__setattr__(self, "nonclaims", tuple(self.nonclaims))
        canonical = _canonical_json_bytes(_payload_record_core(self))
        object.__setattr__(self, "canonical_bytes", canonical)
        object.__setattr__(
            self,
            "payload_sha256",
            hashlib.sha256(H5_VALIDATION_PAYLOAD_DOMAIN + canonical).hexdigest(),
        )


@dataclass(frozen=True, slots=True)
class H5GateEvaluation:
    schema_version: Literal["h5-gate-evaluation-v1"]
    result: H5GateResult
    reference: H5ReferenceState | None
    positive_attempts: tuple[H5AttemptOutcome, ...] | None
    controls: tuple[H5ControlResult, ...] | None
    oracle_results: tuple[H5OracleUpdate, ...] | None
    validation_payload: H5ValidationPayloadRecord

    def __post_init__(self) -> None:
        if self.schema_version != "h5-gate-evaluation-v1":
            raise ValueError("unsupported H5 gate-evaluation schema")
        if not isinstance(self.result, H5GateResult) or not isinstance(
            self.validation_payload, H5ValidationPayloadRecord
        ):
            raise ValueError("evaluation result/payload is not typed")
        if self.validation_payload.result != self.result:
            raise ValueError("evaluation payload does not embed its result")
        ready = self.result.preflight.phase is H5PreflightPhase.READY
        if not ready:
            if any(
                item is not None
                for item in (
                    self.reference,
                    self.positive_attempts,
                    self.controls,
                    self.oracle_results,
                )
            ):
                raise ValueError("preflight failure cannot expose derived evaluation fields")
        else:
            if (
                not isinstance(self.reference, H5ReferenceState)
                or self.result.reference_sha256 != self.reference.reference_sha256
                or self.validation_payload.reference_sha256
                != self.reference.reference_sha256
            ):
                raise ValueError("ready evaluation reference is inconsistent")
            evidence_fields = (
                self.positive_attempts,
                self.controls,
                self.oracle_results,
            )
            if self.result.positive_cases is None or self.result.controls is None:
                if self.result.status is not GateStatus.INCONCLUSIVE or any(
                    item is not None for item in evidence_fields
                ):
                    raise ValueError(
                        "unavailable ready evaluation evidence must remain explicit nulls"
                    )
            elif (
                self.positive_attempts is None
                or self.controls is None
                or self.oracle_results is None
                or self.positive_attempts
                != tuple(item.outcome for item in self.result.positive_cases)
                or self.controls != self.result.controls
                or self.validation_payload.positive_attempts != self.positive_attempts
                or self.validation_payload.controls != self.controls
                or self.validation_payload.oracle_results != self.oracle_results
            ):
                raise ValueError("ready evaluation fields are inconsistent")


def _decode_float_hex(value: object) -> object:
    if type(value) is str and value.startswith(("0x", "-0x")):
        return float.fromhex(value)
    if type(value) is list:
        return [_decode_float_hex(item) for item in value]
    if type(value) is dict:
        return {key: _decode_float_hex(item) for key, item in value.items()}
    return value


def _candidate_active_values(
    candidate: H5CandidateSnapshot,
) -> tuple[tuple[str, float], ...]:
    if candidate.rule is H5UpdateRule.EXACT_Z0:
        coordinate = next(
            item
            for item in candidate.recognition.gaussians
            if item.coordinate_id == "q[z0]"
        )
        return (
            ("exact_z0.mean", coordinate.mean.values[0]),
            ("exact_z0.variance", coordinate.variance.values[0]),
        )
    if candidate.rule is H5UpdateRule.EXACT_SOURCE_ROW_A2:
        coordinate = next(
            item
            for item in candidate.recognition.categoricals
            if item.coordinate_id == "q[source_row_a2]"
        )
        return tuple(
            (f"exact_source_row_a2.probability[{index}]", value)
            for index, value in enumerate(coordinate.probabilities.values)
        )
    if candidate.rule is H5UpdateRule.EXACT_STATE_TRANSITION_2_M:
        block = next(
            item
            for item in candidate.model.parameter_blocks
            if item.block_id == "theta[state_transition_2]"
        )
        return tuple(
            (f"exact_state_transition_2_m.{name}", value.values[0])
            for name, value in block.values
        )
    raise ValueError("candidate rule has no independent exact-field comparison")


def _oracle_active_values(oracle: H5OracleUpdate) -> tuple[tuple[str, float], ...]:
    recognition = _decode_float_hex(json.loads(oracle.candidate_recognition_json))
    model = _decode_float_hex(json.loads(oracle.candidate_model_json))
    if oracle.rule == H5UpdateRule.EXACT_Z0.value:
        row = next(item for item in recognition["gaussians"] if item[0] == "q[z0]")
        return (
            ("exact_z0.mean", row[1]["values"][0]),
            ("exact_z0.variance", row[2]["values"][0]),
        )
    if oracle.rule == H5UpdateRule.EXACT_SOURCE_ROW_A2.value:
        row = next(
            item
            for item in recognition["categoricals"]
            if item[0] == "q[source_row_a2]"
        )
        return tuple(
            (f"exact_source_row_a2.probability[{index}]", value)
            for index, value in enumerate(row[3]["values"])
        )
    if oracle.rule == H5UpdateRule.EXACT_STATE_TRANSITION_2_M.value:
        block = next(
            item
            for item in model["parameter_blocks"]
            if item[0] == "theta[state_transition_2]"
        )
        return tuple(
            (f"exact_state_transition_2_m.{name}", value["values"][0])
            for name, value in block[1]
        )
    raise ValueError("oracle rule has no independent exact-field comparison")


def compare_h5_exact_candidate(
    production: H5CandidateSnapshot,
    oracle: H5OracleUpdate,
) -> H5CandidateComparison:
    if not isinstance(production, H5CandidateSnapshot) or not isinstance(
        oracle, H5OracleUpdate
    ):
        raise ValueError("candidate comparison requires typed production and oracle records")
    if production.rule.value != oracle.rule or production.rule not in _CANDIDATE_FIELDS_BY_RULE:
        raise ValueError("candidate comparison rules do not match an exact H5 rule")
    production_values = _candidate_active_values(production)
    oracle_values = _oracle_active_values(oracle)
    if tuple(name for name, _ in production_values) != tuple(
        name for name, _ in oracle_values
    ):
        raise ValueError("candidate comparison fields do not align")
    production_condition = (
        production.numerical_diagnostics[0][1]
        if production.rule is H5UpdateRule.EXACT_STATE_TRANSITION_2_M
        else 1.0
    )
    oracle_condition = (
        oracle.candidate_condition_numbers[0][1]
        if production.rule is H5UpdateRule.EXACT_STATE_TRANSITION_2_M
        else 1.0
    )
    comparisons: list[H5CandidateScalarComparison] = []
    for (field_id, production_value), (_, oracle_value) in zip(
        production_values, oracle_values, strict=True
    ):
        count = H5_CANDIDATE_COMPARISON_OPERATION_COUNTS[field_id]
        production_rounding = (
            H5_C
            * gamma_n(count)
            * max(1.0, production_condition)
            * max(1.0, abs(production_value))
        )
        oracle_rounding = (
            H5_C
            * gamma_n(count)
            * max(1.0, oracle_condition)
            * max(1.0, abs(oracle_value))
        )
        comparison_rounding = H5_C * gamma_n(3) * max(
            1.0,
            abs(production_value),
            abs(oracle_value),
            abs(production_value) + abs(oracle_value),
        )
        allowance = math.fsum(
            (production_rounding, oracle_rounding, comparison_rounding)
        )
        error = abs(production_value - oracle_value)
        comparisons.append(
            H5CandidateScalarComparison(
                field_id,
                float(production_value),
                float(oracle_value),
                count,
                float(production_condition),
                float(oracle_condition),
                float(production_rounding),
                float(oracle_rounding),
                float(comparison_rounding),
                float(allowance),
                float(error),
                error <= allowance,
            )
        )
    result = tuple(comparisons)
    return H5CandidateComparison(
        production.rule,
        result,
        max(item.absolute_error for item in result),
        max(item.allowance for item in result),
        all(item.passed for item in result),
    )


def _rebuild_h5_term_allowance(item: H5TermAllowance) -> H5TermAllowance:
    if not isinstance(item, H5TermAllowance):
        raise ValueError("production term allowance is not typed")
    return H5TermAllowance(
        **{record.name: getattr(item, record.name) for record in fields(H5TermAllowance)}
    )


def _fraction_sum(values: tuple[float, ...]) -> Fraction:
    total = Fraction(0, 1)
    for index, value in enumerate(values):
        total += Fraction.from_float(_finite(value, f"fraction_values[{index}]"))
    return total


def _fsum_rounding_error(values: tuple[float, ...], result: float) -> Fraction:
    checked_result = _finite(result, "fsum result")
    return abs(Fraction.from_float(checked_result) - _fraction_sum(values))


def _production_complete_operand(
    role: Literal["before", "after"],
    evaluation: CompleteElboEvaluation,
) -> H5DeltaOperandEvidence:
    if not isinstance(evaluation, CompleteElboEvaluation):
        raise ValueError("production complete operand requires a typed evaluation")
    if type(evaluation.term_allowances) is not tuple:
        raise ValueError("production term allowances must be a tuple")
    trace = tuple(
        _rebuild_h5_term_allowance(item) for item in evaluation.term_allowances
    )
    complete = evaluation.complete_allowance
    if not isinstance(complete, H5CompleteAllowance):
        raise ValueError("production complete allowance is not typed")
    rebuilt_complete = H5CompleteAllowance(
        trace,
        complete.reduction_rounding,
        complete.total,
        complete.stochastic_contribution,
    )
    if (
        complete.term_allowances != evaluation.term_allowances
        or rebuilt_complete != complete
    ):
        raise ValueError(
            "production complete allowance failed exact nested reconstruction"
        )

    signed = tuple(item.signed_reported_value for item in trace)
    operation_count = 13 + sum(
        item.operation_count_order_21 + item.operation_count_order_17
        for item in trace
    )
    conditions = tuple(
        number
        for item in trace
        for local in (
            item.condition_numbers_order_21,
            item.condition_numbers_order_17,
        )
        for number in local
    )
    summands = tuple(
        number
        for item in trace
        for local in (
            item.absolute_summands_order_21,
            item.absolute_summands_order_17,
        )
        for number in local
    ) + tuple(abs(item) for item in signed)

    convergence_values = tuple(item.convergence_estimate for item in trace)
    rounding_values = tuple(
        number
        for item in trace
        for number in (
            item.rounding_order_21,
            item.rounding_order_17,
            item.comparison_rounding,
        )
    ) + (rebuilt_complete.reduction_rounding,)
    convergence = math.fsum(convergence_values)
    rounding = math.fsum(rounding_values)
    trace_total = math.fsum((convergence, rounding))

    term_component_groups = tuple(
        (
            item.convergence_estimate,
            item.rounding_order_21,
            item.rounding_order_17,
            item.comparison_rounding,
        )
        for item in trace
    )
    term_totals = tuple(item.total for item in trace)
    term_total_sum = math.fsum(term_totals)
    authoritative_total = math.fsum(
        (term_total_sum, rebuilt_complete.reduction_rounding)
    )
    if authoritative_total.hex() != rebuilt_complete.total.hex():
        raise ValueError("production complete allowance total was not reconstructed")

    # Every binary64/fsum stage is bounded against the exact rational sum of
    # its binary64 operands. This covers each term total, the separate
    # convergence/rounding reductions, their trace reduction, the term-total
    # reduction, and the authoritative final reduction without a half-ULP
    # assumption or an associativity assumption.
    trace_path_bound = (
        _fsum_rounding_error(convergence_values, convergence)
        + _fsum_rounding_error(rounding_values, rounding)
        + _fsum_rounding_error((convergence, rounding), trace_total)
    )
    authoritative_path_bound = (
        sum(
            (
                _fsum_rounding_error(components, item.total)
                for components, item in zip(
                    term_component_groups, trace, strict=True
                )
            ),
            Fraction(0, 1),
        )
        + _fsum_rounding_error(term_totals, term_total_sum)
        + _fsum_rounding_error(
            (term_total_sum, rebuilt_complete.reduction_rounding),
            authoritative_total,
        )
    )
    exact_difference = abs(
        Fraction.from_float(trace_total)
        - Fraction.from_float(authoritative_total)
    )
    if exact_difference > trace_path_bound + authoritative_path_bound:
        raise ValueError(
            "production complete allowance exceeded the exact regrouping bound"
        )
    allowance = rebuilt_complete.total
    return H5DeltaOperandEvidence(
        "h5-delta-operand-evidence-v1",
        role,
        float(evaluation.terms.complete_elbo),
        operation_count,
        conditions,
        summands,
        float(rounding),
        float(allowance),
    )


def _production_delta_evidence(
    outcome: CompletedUpdateAttempt,
) -> H5DeltaImplementationEvidence:
    if not isinstance(outcome, CompletedUpdateAttempt):
        raise ValueError("production delta evidence requires a completed attempt")
    allowance = H5DeltaAllowance(
        outcome.allowance.before_total,
        outcome.allowance.after_total,
        outcome.allowance.subtraction_rounding,
        outcome.allowance.stochastic_contribution,
        outcome.allowance.epsilon_delta,
    )
    if allowance != outcome.allowance:
        raise ValueError("transaction delta allowance failed exact reconstruction")
    before = _production_complete_operand("before", outcome.before)
    after = _production_complete_operand("after", outcome.after)
    delta = H5DeltaOperandEvidence(
        "h5-delta-operand-evidence-v1",
        "delta",
        float(outcome.delta_elbo),
        3,
        (1.0,),
        (abs(before.value), abs(after.value)),
        float(allowance.subtraction_rounding),
        float(allowance.epsilon_delta),
    )
    if (
        before.allowance.hex() != allowance.before_total.hex()
        or after.allowance.hex() != allowance.after_total.hex()
    ):
        raise ValueError("transaction delta allowance does not bind its complete operands")
    return H5DeltaImplementationEvidence(
        "h5-delta-implementation-evidence-v1",
        "production",
        before,
        after,
        delta,
        True,
    )


def _oracle_operand_copy(
    role: Literal["before", "after"],
    operand: H5OracleOperandEvidence,
) -> H5DeltaOperandEvidence:
    rebuilt = H5OracleOperandEvidence.from_complete_terms(
        operand=role,
        complete_term_trace=operand.complete_term_trace,
    )
    if rebuilt != operand:
        raise ValueError("oracle operand failed independent term-trace revalidation")
    return H5DeltaOperandEvidence(
        "h5-delta-operand-evidence-v1",
        role,
        float(rebuilt.value),
        rebuilt.operation_count,
        tuple(rebuilt.condition_numbers),
        tuple(rebuilt.absolute_summands),
        float(rebuilt.rounding),
        float(rebuilt.allowance),
    )


def _oracle_delta_evidence(oracle: H5OracleUpdate) -> H5DeltaImplementationEvidence:
    before = _oracle_operand_copy("before", oracle.before)
    after = _oracle_operand_copy("after", oracle.after)
    rebuilt_before = H5OracleOperandEvidence.from_complete_terms(
        operand="before", complete_term_trace=oracle.before.complete_term_trace
    )
    rebuilt_after = H5OracleOperandEvidence.from_complete_terms(
        operand="after", complete_term_trace=oracle.after.complete_term_trace
    )
    rebuilt_delta = H5OracleOperandEvidence.from_delta(
        before=rebuilt_before, after=rebuilt_after
    )
    if rebuilt_delta != oracle.delta:
        raise ValueError("oracle delta failed independent local reconstruction")
    delta = H5DeltaOperandEvidence(
        "h5-delta-operand-evidence-v1",
        "delta",
        float(rebuilt_delta.value),
        rebuilt_delta.operation_count,
        tuple(rebuilt_delta.condition_numbers),
        tuple(rebuilt_delta.absolute_summands),
        float(rebuilt_delta.rounding),
        float(rebuilt_delta.allowance),
    )
    return H5DeltaImplementationEvidence(
        "h5-delta-implementation-evidence-v1",
        "oracle",
        before,
        after,
        delta,
        True,
    )


def compare_h5_complete_delta(
    outcome: CompletedUpdateAttempt,
    oracle: H5OracleUpdate,
) -> H5DeltaAgreement:
    if not isinstance(outcome, CompletedUpdateAttempt) or not isinstance(
        oracle, H5OracleUpdate
    ):
        raise ValueError("delta comparison requires completed production and oracle records")
    if outcome.request.rule.value != oracle.rule:
        raise ValueError("production and oracle delta rules differ")
    production = _production_delta_evidence(outcome)
    independent = _oracle_delta_evidence(oracle)
    comparison = H5_C * gamma_n(3) * max(
        1.0,
        abs(production.delta.value),
        abs(independent.delta.value),
        abs(production.delta.value) + abs(independent.delta.value),
    )
    allowance = math.fsum(
        (production.delta.allowance, independent.delta.allowance, comparison)
    )
    error = abs(production.delta.value - independent.delta.value)
    return H5DeltaAgreement(
        "h5-delta-agreement-v1",
        outcome.request.rule,
        production,
        independent,
        float(comparison),
        float(allowance),
        float(error),
        error <= allowance,
    )


class _ProductionEvaluator:
    def __init__(self, reference: H5ReferenceState) -> None:
        self.reference = reference

    def evaluate(
        self,
        state: object,
        *,
        frozen_complement_sha256: str,
        cache: object = None,
    ) -> CompleteElboEvaluation:
        return evaluate_h5_complete_elbo(
            self.reference,
            state,
            frozen_complement_sha256=frozen_complement_sha256,
            cache=cache,
        )


def _request(identifier: str, rule: H5UpdateRule) -> UpdateRequest:
    label, variables, parameters, schedule = H5_RULE_CONTRACTS[rule]
    return UpdateRequest(
        "h5-update-request-v1",
        identifier,
        rule,
        label,
        variables,
        parameters,
        schedule,
    )


class _H5RequiredEvidenceUnavailable(RuntimeError):
    def __init__(self, obligation: str, detail: str) -> None:
        super().__init__(detail)
        self.obligation = obligation


def _reconstruct_candidate(
    reference: H5ReferenceState,
    request: UpdateRequest,
    outcome: CompletedUpdateAttempt,
) -> H5CandidateSnapshot:
    live = initial_live(reference)
    if request.rule is H5UpdateRule.EXACT_Z0:
        candidate = exact_conjugate_gaussian_e_update(reference, live, request)
    elif request.rule is H5UpdateRule.EXACT_SOURCE_ROW_A2:
        candidate = exact_source_row_update(reference, live, request)
    elif request.rule is H5UpdateRule.EXACT_STATE_TRANSITION_2_M:
        candidate = exact_gaussian_m_update(reference, live, request)
    elif request.rule is H5UpdateRule.GENERALIZED_EM_EMISSION_1:
        candidate = propose_generalized_em(
            reference, live, request, outcome.damping
        )
    elif request.rule is H5UpdateRule.NATURAL_GRADIENT_Z1:
        candidate = propose_natural_gradient(
            reference, live, request, outcome.damping
        )
    else:
        raise ValueError("H5 positive request rule is outside the closed taxonomy")
    hashes = outcome.hashes
    semantic = h5_semantic_state_sha256(candidate.recognition, candidate.model)
    if (
        candidate.candidate_sha256 != hashes.candidate_sha256
        or candidate.recognition.state_sha256 != hashes.candidate_recognition_sha256
        or candidate.model.state_sha256 != hashes.candidate_model_sha256
        or semantic != outcome.after.evaluated_state_sha256
    ):
        raise _H5RequiredEvidenceUnavailable(
            "resolve deterministic H5 candidate reconstruction same-domain hash agreement",
            "deterministic candidate reconstruction hashes disagree",
        )
    return candidate


def _oracle_for_candidate(
    rule: H5UpdateRule,
    *,
    h1_bytes: bytes,
    h5_bytes: bytes,
    before_state_bytes: bytes,
    after_state_bytes: bytes,
) -> H5OracleUpdate:
    if rule is H5UpdateRule.EXACT_Z0:
        return oracle_exact_e_block(h1_bytes, h5_bytes, before_state_bytes)
    if rule is H5UpdateRule.EXACT_SOURCE_ROW_A2:
        return oracle_exact_source_row(h1_bytes, h5_bytes, before_state_bytes)
    if rule is H5UpdateRule.EXACT_STATE_TRANSITION_2_M:
        return oracle_exact_m_block(h1_bytes, h5_bytes, before_state_bytes)
    return oracle_complete_delta(
        h1_bytes,
        h5_bytes,
        before_state_bytes,
        after_state_bytes,
        rule=rule.value,
    )


_FAULT_BY_CONTROL = MappingProxyType(
    {
        H5ControlId.OMIT_CHILD: H5FaultInjection(
            H5FaultKind.OMIT_CHILD, "state_transition[2]", None
        ),
        H5ControlId.OMIT_EMISSION: H5FaultInjection(
            H5FaultKind.OMIT_EMISSION, "emission[1]", None
        ),
        H5ControlId.FORCE_UNRESOLVED_GEM: H5FaultInjection(
            H5FaultKind.FORCE_UNRESOLVED_GEM_ACCEPT, None, None
        ),
        H5ControlId.MISLABEL_NATURAL: H5FaultInjection(
            H5FaultKind.MISLABEL_NATURAL_AS_EXACT, None, None
        ),
        H5ControlId.MUTATE_REJECTION: H5FaultInjection(
            H5FaultKind.MUTATE_REJECTED_LIVE_AND_RNG, None, None
        ),
        H5ControlId.CHANGED_INPUT_EQUAL_VALUE: H5FaultInjection(
            H5FaultKind.CHANGE_INPUT_KEEP_VALUE, "state_transition[2]", None
        ),
        H5ControlId.CHANGED_VALUE_SAME_INPUT: H5FaultInjection(
            H5FaultKind.CHANGE_VALUE_KEEP_INPUT, "state_transition[2]", 1.0e-6
        ),
    }
)


def _run_positive_cases(
    reference: H5ReferenceState,
    h1_bytes: bytes,
    h5_bytes: bytes,
) -> tuple[
    tuple[H5PositiveCaseResult, ...],
    tuple[H5AttemptOutcome, ...],
    tuple[H5OracleUpdate, ...],
]:
    results: list[H5PositiveCaseResult] = []
    attempts: list[H5AttemptOutcome] = []
    oracles: list[H5OracleUpdate] = []
    for case_id in H5PositiveCaseId:
        rule = H5_POSITIVE_RULE_BY_ID[case_id]
        request = _request(case_id.value, rule)
        live = initial_live(reference)
        transaction = execute_update(
            reference,
            live,
            request,
            _ProductionEvaluator(reference),
            DEFAULT_H5_BUDGET_CONFIG,
        )
        outcome = transaction.outcome
        attempts.append(outcome)
        if not isinstance(outcome, CompletedUpdateAttempt):
            raise ValueError(
                f"positive case {case_id.value} did not produce complete evidence"
            )
        candidate = _reconstruct_candidate(reference, request, outcome)
        before_bytes = canonical_h5_semantic_state_bytes(
            live.recognition, live.model
        )
        after_bytes = canonical_h5_semantic_state_bytes(
            candidate.recognition, candidate.model
        )
        oracle = _oracle_for_candidate(
            rule,
            h1_bytes=h1_bytes,
            h5_bytes=h5_bytes,
            before_state_bytes=before_bytes,
            after_state_bytes=after_bytes,
        )
        oracles.append(oracle)
        candidate_comparison = (
            compare_h5_exact_candidate(candidate, oracle)
            if rule in _CANDIDATE_FIELDS_BY_RULE
            else None
        )
        delta_agreement = compare_h5_complete_delta(outcome, oracle)
        production_semantic = h5_semantic_state_sha256(
            candidate.recognition, candidate.model
        )
        passed = (
            _positive_outcome_passes(case_id, outcome)
            and delta_agreement.passed
            and (candidate_comparison is None or candidate_comparison.passed)
        )
        results.append(
            H5PositiveCaseResult(
                "h5-positive-case-result-v1",
                case_id,
                outcome,
                production_semantic,
                oracle.semantic_state_sha256,
                candidate_comparison,
                delta_agreement,
                passed,
                "complete typed outcome, deterministic reconstruction, and independent delta checked",
            )
        )
    return tuple(results), tuple(attempts), tuple(oracles)


def _run_controls(reference: H5ReferenceState) -> tuple[H5ControlResult, ...]:
    results: list[H5ControlResult] = []
    for control_id in H5ControlId:
        rule = H5_CONTROL_BASE_RULE_BY_ID[control_id]
        request = _request(control_id.value, rule)
        transaction = execute_update(
            reference,
            initial_live(reference),
            request,
            _ProductionEvaluator(reference),
            DEFAULT_H5_BUDGET_CONFIG,
            fault_injection=_FAULT_BY_CONTROL[control_id],
        )
        outcome = transaction.outcome
        observed = _control_outcome_detection(control_id, outcome)
        expected = H5_CONTROL_DETECTION_BY_ID[control_id]
        results.append(
            H5ControlResult(
                "h5-control-result-v1",
                control_id,
                expected,
                observed,
                outcome,
                observed is expected,
                "typed injected fault detection checked at its first valid phase",
            )
        )
    return tuple(results)


def _invariant(name: str, passed: bool, detail: str) -> InvariantResult:
    return InvariantResult(name, passed, 1.0 if passed else 0.0, 1.0, detail)


def _preflight_failure(
    *,
    h1_raw_sha256: str,
    h5_raw_sha256: str,
    phase: H5PreflightPhase,
    kind: H5PreflightErrorKind,
    detail: str,
) -> H5GateEvaluation:
    checked_detail = detail if detail else kind.value
    obligation = f"repair H5 {phase.value}: {kind.value}"
    error = H5PreflightError(phase, kind, checked_detail)
    preflight = H5PreflightRecord(
        "h5-preflight-record-v1",
        phase,
        h1_raw_sha256,
        h5_raw_sha256,
        (error,),
        tuple(H5UnavailableField),
        obligation,
    )
    result = H5GateResult(
        "h5-gate-result-v1",
        "H5",
        GateStatus.INCONCLUSIVE,
        preflight,
        h1_raw_sha256,
        h5_raw_sha256,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        (),
        (obligation,),
    )
    payload = H5ValidationPayloadRecord(
        1,
        result,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        H5_NONCLAIM_IDS,
    )
    return H5GateEvaluation(
        "h5-gate-evaluation-v1",
        result,
        None,
        None,
        None,
        None,
        payload,
    )


def _ready_inconclusive(
    *,
    h1_raw_sha256: str,
    h5_raw_sha256: str,
    reference: H5ReferenceState,
    graph: FactorDependencyGraph,
    obligation: str,
) -> H5GateEvaluation:
    if type(obligation) is not str or not obligation:
        raise ValueError("ready INCONCLUSIVE obligation must be nonempty")
    preflight = H5PreflightRecord(
        "h5-preflight-record-v1",
        H5PreflightPhase.READY,
        h1_raw_sha256,
        h5_raw_sha256,
        (),
        (),
        None,
    )
    result = H5GateResult(
        "h5-gate-result-v1",
        "H5",
        GateStatus.INCONCLUSIVE,
        preflight,
        h1_raw_sha256,
        h5_raw_sha256,
        reference.specification.canonical_sha256,
        reference.objective_schema_sha256,
        H5_FACTOR_INPUT_SCHEMA_VERSION,
        reference.factor_input_schema_sha256,
        reference.reference_sha256,
        None,
        None,
        (),
        (obligation,),
    )
    payload = H5ValidationPayloadRecord(
        1,
        result,
        reference.reference_sha256,
        graph.factor_universe,
        graph.recognition_coordinate_universe,
        graph.model_block_universe,
        graph.variable_dependencies,
        graph.parameter_dependencies,
        None,
        None,
        None,
        H5_NONCLAIM_IDS,
    )
    return H5GateEvaluation(
        "h5-gate-evaluation-v1",
        result,
        reference,
        None,
        None,
        None,
        payload,
    )


def evaluate_h5(
    config: ResolvedConfig,
    *,
    h1_fixture_bytes: bytes,
    h5_update_spec_bytes: bytes,
) -> H5GateEvaluation:
    if not isinstance(config, ResolvedConfig):
        raise ValueError("config must be a ResolvedConfig")
    if (
        config.run.device != "cpu"
        or config.run.dtype != "float64"
        or config.run.deterministic is not True
    ):
        raise ValueError("H5 v1 requires deterministic CPU binary64 configuration")
    if type(h1_fixture_bytes) is not bytes or type(h5_update_spec_bytes) is not bytes:
        raise ValueError("H5 gate consumes captured immutable bytes")
    h1_digest = hashlib.sha256(h1_fixture_bytes).hexdigest()
    h5_digest = hashlib.sha256(h5_update_spec_bytes).hexdigest()
    if h1_digest != H5_H1_FIXTURE_RAW_SHA256:
        return _preflight_failure(
            h1_raw_sha256=h1_digest,
            h5_raw_sha256=h5_digest,
            phase=H5PreflightPhase.H1_FIXTURE_VALIDATION,
            kind=H5PreflightErrorKind.INVALID_H1_FIXTURE,
            detail=(
                f"expected H1 raw SHA-256 {H5_H1_FIXTURE_RAW_SHA256}, got {h1_digest}"
            ),
        )
    if h5_digest != EXPECTED_H5_UPDATE_SPEC_RAW_SHA256:
        return _preflight_failure(
            h1_raw_sha256=h1_digest,
            h5_raw_sha256=h5_digest,
            phase=H5PreflightPhase.UPDATE_SPEC_VALIDATION,
            kind=H5PreflightErrorKind.UPDATE_SPEC_RAW_DIGEST_MISMATCH,
            detail=(
                "expected H5 update-spec raw SHA-256 "
                f"{EXPECTED_H5_UPDATE_SPEC_RAW_SHA256}, got {h5_digest}"
            ),
        )
    try:
        parsed_specification = parse_h5_update_spec_bytes(h5_update_spec_bytes)
    except (RuntimeError, ValueError) as exc:
        return _preflight_failure(
            h1_raw_sha256=h1_digest,
            h5_raw_sha256=h5_digest,
            phase=H5PreflightPhase.UPDATE_SPEC_VALIDATION,
            kind=H5PreflightErrorKind.INVALID_UPDATE_SPEC_SCHEMA,
            detail=str(exc),
        )
    try:
        reference = build_h5_reference_state(
            h1_fixture_bytes, h5_update_spec_bytes
        )
    except (RuntimeError, ValueError) as exc:
        return _preflight_failure(
            h1_raw_sha256=h1_digest,
            h5_raw_sha256=h5_digest,
            phase=H5PreflightPhase.REFERENCE_CONSTRUCTION,
            kind=H5PreflightErrorKind.REFERENCE_CONSTRUCTION_FAILED,
            detail=str(exc),
        )
    objective_recomputed = hashlib.sha256(
        H5_OBJECTIVE_SCHEMA_DOMAIN + canonical_h5_objective_schema_core_bytes()
    ).hexdigest()
    if (
        reference.objective_schema_sha256 != H5_OBJECTIVE_SCHEMA_SHA256
        or objective_recomputed != H5_OBJECTIVE_SCHEMA_SHA256
    ):
        return _preflight_failure(
            h1_raw_sha256=h1_digest,
            h5_raw_sha256=h5_digest,
            phase=H5PreflightPhase.REFERENCE_CONSTRUCTION,
            kind=H5PreflightErrorKind.OBJECTIVE_SCHEMA_IDENTITY_FAILED,
            detail="objective schema hash failed exact recomputation",
        )
    factor_recomputed = hashlib.sha256(
        H5_FACTOR_INPUT_SCHEMA_DOMAIN
        + canonical_h5_factor_input_schema_core_bytes()
    ).hexdigest()
    if (
        reference.factor_input_schema_sha256 != H5_FACTOR_INPUT_SCHEMA_SHA256
        or factor_recomputed != H5_FACTOR_INPUT_SCHEMA_SHA256
    ):
        return _preflight_failure(
            h1_raw_sha256=h1_digest,
            h5_raw_sha256=h5_digest,
            phase=H5PreflightPhase.REFERENCE_CONSTRUCTION,
            kind=H5PreflightErrorKind.FACTOR_INPUT_SCHEMA_IDENTITY_FAILED,
            detail="factor-input schema hash failed exact recomputation",
        )
    if (
        parsed_specification.canonical_sha256
        != reference.specification.canonical_sha256
    ):
        return _preflight_failure(
            h1_raw_sha256=h1_digest,
            h5_raw_sha256=h5_digest,
            phase=H5PreflightPhase.REFERENCE_CONSTRUCTION,
            kind=H5PreflightErrorKind.REFERENCE_CONSTRUCTION_FAILED,
            detail="parsed and reference update-spec canonical hashes differ",
        )
    try:
        graph = build_h5_reference_dependency_graph(reference.specification)
    except (RuntimeError, ValueError) as exc:
        return _preflight_failure(
            h1_raw_sha256=h1_digest,
            h5_raw_sha256=h5_digest,
            phase=H5PreflightPhase.REFERENCE_CONSTRUCTION,
            kind=H5PreflightErrorKind.REFERENCE_CONSTRUCTION_FAILED,
            detail=f"H5 dependency-graph construction failed: {exc}",
        )
    try:
        positive_cases, positive_attempts, oracle_results = _run_positive_cases(
            reference, h1_fixture_bytes, h5_update_spec_bytes
        )
        controls = _run_controls(reference)
    except _H5RequiredEvidenceUnavailable as exc:
        return _ready_inconclusive(
            h1_raw_sha256=h1_digest,
            h5_raw_sha256=h5_digest,
            reference=reference,
            graph=graph,
            obligation=exc.obligation,
        )
    except (RuntimeError, ValueError):
        # The reference is valid and retained; missing later evidence is not
        # mislabeled as a preflight failure or converted into a finite claim.
        return _ready_inconclusive(
            h1_raw_sha256=h1_digest,
            h5_raw_sha256=h5_digest,
            reference=reference,
            graph=graph,
            obligation="complete unavailable H5 positive/control/oracle evidence",
        )

    preflight = H5PreflightRecord(
        "h5-preflight-record-v1",
        H5PreflightPhase.READY,
        h1_digest,
        h5_digest,
        (),
        (),
        None,
    )
    invariants = _completed_h5_invariants(positive_cases, controls)
    status, obligations = _classify_completed_h5_evidence(
        gem_delta_indecisive=_gem_positive_is_indecisive(positive_cases),
        simultaneous_decisive_failure=any(
            not item.passed for item in invariants
        ),
    )
    result = H5GateResult(
        "h5-gate-result-v1",
        "H5",
        status,
        preflight,
        h1_digest,
        h5_digest,
        reference.specification.canonical_sha256,
        reference.objective_schema_sha256,
        H5_FACTOR_INPUT_SCHEMA_VERSION,
        reference.factor_input_schema_sha256,
        reference.reference_sha256,
        positive_cases,
        controls,
        invariants,
        obligations,
    )
    payload = H5ValidationPayloadRecord(
        1,
        result,
        reference.reference_sha256,
        H5_FACTOR_UNIVERSE,
        H5_RECOGNITION_COORDINATE_UNIVERSE,
        H5_MODEL_BLOCK_UNIVERSE,
        H5_VARIABLE_DEPENDENCY_ROWS,
        H5_PARAMETER_DEPENDENCY_ROWS,
        positive_attempts,
        controls,
        oracle_results,
        H5_NONCLAIM_IDS,
    )
    return H5GateEvaluation(
        "h5-gate-evaluation-v1",
        result,
        reference,
        positive_attempts,
        controls,
        oracle_results,
        payload,
    )


def h5_validation_payload(evaluation: H5GateEvaluation) -> dict[str, object]:
    if type(evaluation) is not H5GateEvaluation:
        raise ValueError("H5 validation payload requires the exact evaluation type")
    record = evaluation.validation_payload
    result = {
        name: _canonicalize(value)
        for name, value in _payload_record_core(record).items()
    }
    result["payload_sha256"] = record.payload_sha256
    expected = tuple(
        item.name
        for item in fields(H5ValidationPayloadRecord)
        if item.name != "canonical_bytes"
    )
    if tuple(result) != expected:
        raise RuntimeError("H5 validation payload field order drifted")
    return result


def h5_update_binding_preimages(
    evaluation: H5GateEvaluation,
) -> dict[str, object]:
    """Encode the eight intrinsic H5 binding preimages for provenance."""

    if type(evaluation) is not H5GateEvaluation:
        raise ValueError("H5 update-binding preimages require the exact evaluation type")
    if (
        evaluation.result.preflight.phase is not H5PreflightPhase.READY
        or evaluation.reference is None
    ):
        raise ValueError("H5 update-binding preimages require a READY evaluation")
    reference = evaluation.reference
    preimages = {
        "update_spec_raw_sha256": reference.raw_update_spec_bytes,
        "update_spec_canonical_sha256": (
            H5_UPDATE_SPEC_DOMAIN + reference.specification.canonical_bytes
        ),
        "objective_schema_sha256": (
            H5_OBJECTIVE_SCHEMA_DOMAIN
            + canonical_h5_objective_schema_core_bytes()
        ),
        "factor_input_schema_sha256": (
            H5_FACTOR_INPUT_SCHEMA_DOMAIN
            + canonical_h5_factor_input_schema_core_bytes()
        ),
        "reference_sha256": (
            H5_REFERENCE_STATE_DOMAIN
            + canonical_h5_reference_state_bytes(reference)
        ),
        "recognition_state_sha256": (
            H5_RECOGNITION_SNAPSHOT_DOMAIN
            + canonical_h5_recognition_snapshot_bytes(
                reference.initial_recognition
            )
        ),
        "model_state_sha256": (
            H5_MODEL_SNAPSHOT_DOMAIN
            + canonical_h5_model_snapshot_bytes(reference.initial_model)
        ),
        "validation_payload_sha256": (
            H5_VALIDATION_PAYLOAD_DOMAIN
            + evaluation.validation_payload.canonical_bytes
        ),
    }
    expected_digests = {
        "update_spec_raw_sha256": evaluation.result.update_spec_raw_sha256,
        "update_spec_canonical_sha256": (
            evaluation.result.update_spec_canonical_sha256
        ),
        "objective_schema_sha256": evaluation.result.objective_schema_sha256,
        "factor_input_schema_sha256": (
            evaluation.result.factor_input_schema_sha256
        ),
        "reference_sha256": evaluation.result.reference_sha256,
        "recognition_state_sha256": (
            reference.initial_recognition.state_sha256
        ),
        "model_state_sha256": reference.initial_model.state_sha256,
        "validation_payload_sha256": (
            evaluation.validation_payload.payload_sha256
        ),
    }
    observed_digests = {
        name: hashlib.sha256(preimage).hexdigest()
        for name, preimage in preimages.items()
    }
    if observed_digests != expected_digests:
        mismatches = tuple(
            name
            for name in preimages
            if observed_digests[name] != expected_digests[name]
        )
        raise RuntimeError(
            "H5 update-binding preimages differ from digest summaries: "
            + ", ".join(mismatches)
        )
    return {
        "schema_version": "h5-update-binding-preimages-v1",
        "encoding": "hex",
        "preimages": {
            name: preimage.hex()
            for name, preimage in preimages.items()
        },
    }


__all__ = [
    "H5PositiveCaseId",
    "H5ControlId",
    "H5ControlDetection",
    "H5_CONTROL_DETECTION_BY_ID",
    "H5_POSITIVE_RULE_BY_ID",
    "H5_CONTROL_BASE_RULE_BY_ID",
    "H5PreflightPhase",
    "H5PreflightErrorKind",
    "H5_PREFLIGHT_ERROR_KINDS_BY_PHASE",
    "H5UnavailableField",
    "H5_INVARIANT_NAMES",
    "H5PreflightError",
    "H5PreflightRecord",
    "H5DeltaOperandEvidence",
    "H5DeltaImplementationEvidence",
    "H5DeltaAgreement",
    "H5CandidateScalarComparison",
    "H5CandidateComparison",
    "H5PositiveCaseResult",
    "H5ControlResult",
    "H5GateResult",
    "H5ValidationPayloadRecord",
    "H5GateEvaluation",
    "compare_h5_exact_candidate",
    "compare_h5_complete_delta",
    "evaluate_h5",
    "h5_update_binding_preimages",
    "h5_validation_payload",
]
