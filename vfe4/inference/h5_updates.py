"""Closed H5 coordinate updates and freeze-before-evaluate transactions."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

import torch

from vfe4.numerics.h5_budget import (
    H5BudgetConfig,
    H5DeltaAllowance,
    epsilon_delta,
)
from vfe4.numerics.quadrature import probabilists_gauss_hermite
from vfe4.objective.dependency_graph import (
    build_h5_reference_dependency_graph,
    expected_affected_factors,
)
from vfe4.objective.h5_complete import (
    CacheDisposition,
    CompleteElboEvaluation,
    CompleteElboEvaluator,
    FactorCacheEntry,
    FactorCacheKey,
    FactorEvaluationRecord,
    StaleFactorCacheError,
    evaluate_h5_complete_elbo,
)
from vfe4.types.h5_schema import (
    H5_FACTOR_UNIVERSE,
    H5_FROZEN_COMPLEMENT_DOMAIN,
    H5_MODEL_BLOCK_UNIVERSE,
    H5_RECOGNITION_COORDINATE_UNIVERSE,
)
from vfe4.types.updates import (
    CategoricalRecognitionCoordinate,
    FrozenByteState,
    FrozenTensorValue,
    GaussianRecognitionCoordinate,
    H5_RULE_CONTRACTS,
    H5CandidateSnapshot,
    H5LiveState,
    H5ModelSnapshot,
    H5ReferenceState,
    H5UpdateRule,
    ModelParameterBlock,
    RecognitionSnapshot,
    UpdateLabel,
    UpdateRequest,
)


_LOWER_HEX = frozenset("0123456789abcdef")
_GAUSSIAN_IDS = H5_RECOGNITION_COORDINATE_UNIVERSE[:6]
_CATEGORICAL_IDS = H5_RECOGNITION_COORDINATE_UNIVERSE[6:]
_MODEL_FIELDS = {
    "theta[state_transition_2]": (
        "alpha_0",
        "alpha_1",
        "B_base",
        "c",
        "R",
    ),
    "theta[emission_1]": ("w_z", "w_m", "bias"),
    "theta[shared_decoder_transition]": ("s",),
}
_LOG_2_PI = math.log(2.0 * math.pi)


class AttemptPhase(str, Enum):
    REQUEST = "request"
    BEFORE_EVALUATION = "before_evaluation"
    PROPOSAL = "proposal"
    FREEZE = "freeze"
    AFTER_EVALUATION = "after_evaluation"
    DEPENDENCY_VALIDATION = "dependency_validation"
    DECISION = "decision"
    COMMIT_OR_ROLLBACK = "commit_or_rollback"


class AttemptFailureReason(str, Enum):
    LABEL_PROVENANCE_MISMATCH = "label_provenance_mismatch"
    FACTOR_COVERAGE_MISMATCH = "factor_coverage_mismatch"
    AFFECTED_FACTOR_MISMATCH = "affected_factor_mismatch"
    STALE_CACHE = "stale_cache"
    NONFINITE_OR_INVALID_CANDIDATE = "nonfinite_or_invalid_candidate"
    DECISION_POLICY_VIOLATION = "decision_policy_violation"
    ROLLBACK_HASH_MISMATCH = "rollback_hash_mismatch"
    DETERMINISTIC_REEVALUATION_MISMATCH = "deterministic_reevaluation_mismatch"


class DecisionReason(str, Enum):
    EXACT_WITHIN_ALLOWANCE = "exact_within_allowance"
    RESOLVED_POSITIVE = "resolved_positive"
    RESOLVED_DECREASE_REJECTED = "resolved_decrease_rejected"
    UNRESOLVED_DELTA_REJECTED = "unresolved_delta_rejected"


class H5FaultKind(str, Enum):
    OMIT_CHILD = "omit_child"
    OMIT_EMISSION = "omit_emission"
    FORCE_UNRESOLVED_GEM_ACCEPT = "force_unresolved_gem_accept"
    MISLABEL_NATURAL_AS_EXACT = "mislabel_natural_as_exact"
    MUTATE_REJECTED_LIVE_AND_RNG = "mutate_rejected_live_and_rng"
    CHANGE_INPUT_KEEP_VALUE = "change_input_keep_value"
    CHANGE_VALUE_KEEP_INPUT = "change_value_keep_input"


@dataclass(frozen=True)
class H5FaultInjection:
    kind: H5FaultKind
    target_factor_id: str | None
    scalar_delta: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, H5FaultKind):
            raise ValueError("kind must be an H5FaultKind")
        if self.target_factor_id is not None and self.target_factor_id not in H5_FACTOR_UNIVERSE:
            raise ValueError("target_factor_id is outside the H5 factor universe")
        if self.scalar_delta is not None and (
            type(self.scalar_delta) is not float or not math.isfinite(self.scalar_delta)
        ):
            raise ValueError("scalar_delta must be a finite binary64 float or None")


H5_CANDIDATE_DRAFT_DOMAIN: Final[bytes] = b"vfe4.h5.candidate-draft.v1\x00"


def _require_sha256(value: object, name: str) -> str:
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


def _canonicalize(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("canonical H5 floats must be finite")
        return value.hex()
    if type(value) in (str, int, bool) or value is None:
        return value
    if type(value) is bytes:
        return {"hex": value.hex(), "length": len(value)}
    if type(value) is tuple:
        return [_canonicalize(item) for item in value]
    if isinstance(value, Mapping):
        if not all(type(key) is str and key for key in value):
            raise ValueError("canonical mapping keys must be nonempty strings")
        return {key: _canonicalize(value[key]) for key in sorted(value)}
    raise ValueError(f"unsupported canonical value: {type(value).__name__}")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        _canonicalize(value),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def _tensor_core(value: FrozenTensorValue) -> object:
    return {"dtype": value.dtype, "shape": value.shape, "values": value.values}


def _recognition_core(value: RecognitionSnapshot) -> object:
    return {
        "schema_version": value.schema_version,
        "gaussians": tuple(
            (item.coordinate_id, _tensor_core(item.mean), _tensor_core(item.variance))
            for item in value.gaussians
        ),
        "categoricals": tuple(
            (
                item.coordinate_id,
                item.support,
                item.conditioned_on,
                _tensor_core(item.probabilities),
            )
            for item in value.categoricals
        ),
    }


def _model_core(value: H5ModelSnapshot) -> object:
    return {
        "schema_version": value.schema_version,
        "objective_schema_sha256": value.objective_schema_sha256,
        "parameter_blocks": tuple(
            (
                block.block_id,
                tuple((name, _tensor_core(item)) for name, item in block.values),
            )
            for block in value.parameter_blocks
        ),
        "reconstruction_records": tuple(
            (item.factor_id, item.bindings) for item in value.reconstruction_records
        ),
        "shared_groups": tuple(
            (item.group_id, item.source, item.consumers) for item in value.shared_groups
        ),
    }


@dataclass(frozen=True)
class H5CandidateDraft:
    schema_version: Literal["h5-candidate-draft-v1"]
    rule: H5UpdateRule
    request_sha256: str
    producer_label: UpdateLabel
    variables: tuple[str, ...]
    parameters: tuple[str, ...]
    damping: float
    numerical_diagnostics: tuple[tuple[str, float], ...]
    recognition: RecognitionSnapshot
    model: H5ModelSnapshot
    candidate_draft_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != "h5-candidate-draft-v1":
            raise ValueError("unsupported H5 candidate-draft schema")
        if not isinstance(self.rule, H5UpdateRule):
            raise ValueError("rule must be an H5UpdateRule")
        _require_sha256(self.request_sha256, "request_sha256")
        if not isinstance(self.producer_label, UpdateLabel):
            raise ValueError("producer_label must be an UpdateLabel")
        if type(self.variables) is not tuple or type(self.parameters) is not tuple:
            raise ValueError("variables and parameters must be tuples")
        if any(type(item) is not str or not item for item in self.variables + self.parameters):
            raise ValueError("active identifiers must be nonempty strings")
        damping = _finite(self.damping, "damping")
        if type(self.numerical_diagnostics) is not tuple:
            raise ValueError("numerical_diagnostics must be a tuple")
        diagnostics: list[tuple[str, float]] = []
        for index, item in enumerate(self.numerical_diagnostics):
            if type(item) is not tuple or len(item) != 2 or type(item[0]) is not str:
                raise ValueError(f"numerical_diagnostics[{index}] is invalid")
            diagnostics.append((item[0], _finite(item[1], f"diagnostic[{index}]")))
        recognition = RecognitionSnapshot(
            self.recognition.schema_version,
            self.recognition.gaussians,
            self.recognition.categoricals,
        )
        model = H5ModelSnapshot(
            self.model.schema_version,
            self.model.parameter_blocks,
            self.model.reconstruction_records,
            self.model.shared_groups,
        )
        object.__setattr__(self, "variables", tuple(self.variables))
        object.__setattr__(self, "parameters", tuple(self.parameters))
        object.__setattr__(self, "damping", damping)
        object.__setattr__(self, "numerical_diagnostics", tuple(diagnostics))
        object.__setattr__(self, "recognition", recognition)
        object.__setattr__(self, "model", model)
        object.__setattr__(
            self,
            "candidate_draft_sha256",
            hashlib.sha256(
                H5_CANDIDATE_DRAFT_DOMAIN + canonical_h5_candidate_draft_bytes(self)
            ).hexdigest(),
        )


def canonical_h5_candidate_draft_bytes(draft: H5CandidateDraft) -> bytes:
    if not isinstance(draft, H5CandidateDraft):
        raise ValueError("draft must be an H5CandidateDraft")
    return _canonical_json_bytes(
        {
            "schema_version": draft.schema_version,
            "rule": draft.rule,
            "request_sha256": draft.request_sha256,
            "producer_label": draft.producer_label,
            "variables": draft.variables,
            "parameters": draft.parameters,
            "damping": draft.damping,
            "numerical_diagnostics": draft.numerical_diagnostics,
            "recognition": _recognition_core(draft.recognition),
            "model": _model_core(draft.model),
        }
    )


@dataclass(frozen=True)
class UpdateHashRecord:
    schema_version: Literal["h5-update-hash-record-v1"]
    request_sha256: str
    before_live_sha256: str
    before_recognition_sha256: str
    before_model_sha256: str
    before_optimizer_sha256: str
    before_rng_sha256: str
    predecision_live_sha256: str | None
    predecision_optimizer_sha256: str | None
    predecision_rng_sha256: str | None
    candidate_draft_sha256: str | None
    candidate_sha256: str | None
    candidate_recognition_sha256: str | None
    candidate_model_sha256: str | None
    frozen_complement_sha256: str
    final_live_sha256: str
    final_recognition_sha256: str
    final_model_sha256: str
    final_optimizer_sha256: str
    final_rng_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != "h5-update-hash-record-v1":
            raise ValueError("unsupported H5 update-hash schema")
        for name in (
            "request_sha256",
            "before_live_sha256",
            "before_recognition_sha256",
            "before_model_sha256",
            "before_optimizer_sha256",
            "before_rng_sha256",
            "frozen_complement_sha256",
            "final_live_sha256",
            "final_recognition_sha256",
            "final_model_sha256",
            "final_optimizer_sha256",
            "final_rng_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        for name in (
            "predecision_live_sha256",
            "predecision_optimizer_sha256",
            "predecision_rng_sha256",
            "candidate_draft_sha256",
            "candidate_sha256",
            "candidate_recognition_sha256",
            "candidate_model_sha256",
        ):
            value = getattr(self, name)
            if value is not None:
                _require_sha256(value, name)
        candidate_group = (
            self.candidate_draft_sha256,
            self.candidate_recognition_sha256,
            self.candidate_model_sha256,
        )
        if any(item is None for item in candidate_group) and any(
            item is not None for item in candidate_group
        ):
            raise ValueError("draft, recognition, and model hashes appear together")
        predecision = (
            self.predecision_live_sha256,
            self.predecision_optimizer_sha256,
            self.predecision_rng_sha256,
        )
        if any(item is None for item in predecision) and any(
            item is not None for item in predecision
        ):
            raise ValueError("predecision hashes appear together")
        if self.candidate_sha256 is not None and any(item is None for item in candidate_group):
            raise ValueError("a final candidate requires draft and state hashes")
        if any(item is not None for item in predecision) and self.candidate_sha256 is None:
            raise ValueError("predecision hashes require a final candidate")


@dataclass(frozen=True)
class PartialFactorEvaluation:
    observed_records: tuple[FactorEvaluationRecord, ...]
    expected_factor_ids: tuple[str, ...]
    missing_factor_ids: tuple[str, ...]
    extra_factor_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.observed_records) is not tuple or not all(
            isinstance(item, FactorEvaluationRecord) for item in self.observed_records
        ):
            raise ValueError("observed_records must be factor records")
        if self.expected_factor_ids != H5_FACTOR_UNIVERSE:
            raise ValueError("expected_factor_ids must equal the H5 factor universe")
        observed = tuple(item.factor_id for item in self.observed_records)
        missing = tuple(item for item in H5_FACTOR_UNIVERSE if item not in observed)
        extra = tuple(item for item in observed if item not in H5_FACTOR_UNIVERSE)
        if self.missing_factor_ids != missing or self.extra_factor_ids != extra:
            raise ValueError("partial factor coverage diagnostics are inconsistent")


@dataclass(frozen=True)
class DeterministicReevaluationRecord:
    factor_id: str
    input_sha256: str
    reported_value_order_21: float
    reported_value_order_17: float
    recomputed_value_order_21: float
    recomputed_value_order_17: float
    matched: bool

    def __post_init__(self) -> None:
        if self.factor_id not in H5_FACTOR_UNIVERSE:
            raise ValueError("factor_id is outside H5")
        _require_sha256(self.input_sha256, "input_sha256")
        values = tuple(
            _finite(getattr(self, name), name)
            for name in (
                "reported_value_order_21",
                "reported_value_order_17",
                "recomputed_value_order_21",
                "recomputed_value_order_17",
            )
        )
        expected = values[0].hex() == values[2].hex() and values[1].hex() == values[3].hex()
        if type(self.matched) is not bool or self.matched is not expected:
            raise ValueError("matched must equal the exact pair comparison")


@dataclass(frozen=True)
class CompletedUpdateAttempt:
    schema_version: Literal["h5-completed-attempt-v1"]
    request: UpdateRequest
    producer_label: UpdateLabel
    variables: tuple[str, ...]
    parameters: tuple[str, ...]
    expected_factor_ids: tuple[str, ...]
    expected_affected_factor_ids: tuple[str, ...]
    reevaluated_factor_ids: tuple[str, ...]
    reused_factor_ids: tuple[str, ...]
    observed_affected_factor_ids: tuple[str, ...]
    value_changed_factor_ids: tuple[str, ...]
    missing_factor_ids: tuple[str, ...]
    extra_factor_ids: tuple[str, ...]
    before: CompleteElboEvaluation
    after: CompleteElboEvaluation
    delta_elbo: float
    allowance: H5DeltaAllowance
    accepted: bool
    decision_reason: DecisionReason
    line_search_step: int | None
    damping: float
    autograd_scope: tuple[str, ...]
    hashes: UpdateHashRecord

    def __post_init__(self) -> None:
        if self.schema_version != "h5-completed-attempt-v1":
            raise ValueError("unsupported completed-attempt schema")
        if not isinstance(self.request, UpdateRequest):
            raise ValueError("request must be an UpdateRequest")
        if self.producer_label is not self.request.requested_label:
            raise ValueError("producer label must equal the requested label")
        if self.variables != self.request.variables or self.parameters != self.request.parameters:
            raise ValueError("attempt active blocks must equal the request")
        if self.expected_factor_ids != H5_FACTOR_UNIVERSE:
            raise ValueError("expected factors must equal the H5 universe")
        if self.missing_factor_ids or self.extra_factor_ids:
            raise ValueError("a completed attempt cannot have coverage gaps")
        if self.expected_affected_factor_ids != self.observed_affected_factor_ids:
            raise ValueError("observed affected factors must equal the dependency graph")
        for name, factor_ids in (
            ("reevaluated", self.reevaluated_factor_ids),
            ("reused", self.reused_factor_ids),
        ):
            if type(factor_ids) is not tuple or tuple(
                item for item in H5_FACTOR_UNIVERSE if item in factor_ids
            ) != factor_ids:
                raise ValueError(f"{name} factors must be unique and use universe order")
        reevaluated_set = set(self.reevaluated_factor_ids)
        reused_set = set(self.reused_factor_ids)
        if reevaluated_set & reused_set:
            raise ValueError("reevaluated and reused factors must be disjoint")
        if reevaluated_set | reused_set != set(H5_FACTOR_UNIVERSE):
            raise ValueError("reevaluated/reused factors must exactly partition the universe")
        delta = _finite(self.delta_elbo, "delta_elbo")
        if delta.hex() != float(self.after.terms.complete_elbo - self.before.terms.complete_elbo).hex():
            raise ValueError("delta_elbo must be recomputed from complete objectives")
        if not isinstance(self.allowance, H5DeltaAllowance):
            raise ValueError("allowance must be an H5DeltaAllowance")
        if type(self.accepted) is not bool or not isinstance(self.decision_reason, DecisionReason):
            raise ValueError("decision fields are invalid")
        exact = self.request.requested_label is UpdateLabel.EXACT_COORDINATE
        expected_accepted = (
            delta >= -self.allowance.epsilon_delta
            if exact
            else delta > self.allowance.epsilon_delta
        )
        if self.accepted is not expected_accepted:
            raise ValueError("accepted does not match the closed H5 decision policy")
        expected_reason = (
            DecisionReason.RESOLVED_DECREASE_REJECTED
            if delta < -self.allowance.epsilon_delta
            else DecisionReason.EXACT_WITHIN_ALLOWANCE
            if exact and expected_accepted
            else DecisionReason.RESOLVED_POSITIVE
            if expected_accepted
            else DecisionReason.UNRESOLVED_DELTA_REJECTED
        )
        if self.decision_reason is not expected_reason:
            raise ValueError("decision_reason does not match the closed H5 policy")
        damping = _finite(self.damping, "damping")
        if self.request.rule is H5UpdateRule.GENERALIZED_EM_EMISSION_1:
            if (
                type(self.line_search_step) is not int
                or self.line_search_step < 0
                or self.line_search_step >= len(self.request.damping_schedule)
            ):
                raise ValueError("generalized EM requires an in-range line-search step")
            expected_damping = self.request.damping_schedule[self.line_search_step]
            if damping.hex() != expected_damping.hex():
                raise ValueError("line-search step and damping must identify the same schedule entry")
        elif self.line_search_step is not None:
            raise ValueError("only generalized EM may record a line-search step")
        elif damping not in self.request.damping_schedule:
            raise ValueError("damping must belong to the request schedule")
        expected_scope = (
            (
                "theta[emission_1].w_z",
                "theta[emission_1].w_m",
                "theta[emission_1].bias",
            )
            if self.request.rule is H5UpdateRule.GENERALIZED_EM_EMISSION_1
            else ("q[z1].mean",)
            if self.request.rule is H5UpdateRule.NATURAL_GRADIENT_Z1
            else ()
        )
        if self.autograd_scope != expected_scope:
            raise ValueError("autograd_scope does not match the update rule")
        if self.hashes.request_sha256 != self.request.request_sha256:
            raise ValueError("hash record does not match the request")
        if self.hashes.candidate_sha256 is None or self.hashes.candidate_draft_sha256 is None:
            raise ValueError("completed attempts require draft and final candidate hashes")
        if any(
            item is None
            for item in (
                self.hashes.predecision_live_sha256,
                self.hashes.predecision_optimizer_sha256,
                self.hashes.predecision_rng_sha256,
            )
        ):
            raise ValueError("completed attempts require predecision hashes")


@dataclass(frozen=True)
class FailedUpdateAttempt:
    schema_version: Literal["h5-failed-attempt-v1"]
    request: UpdateRequest
    producer_label: UpdateLabel | None
    phase: AttemptPhase
    reason: AttemptFailureReason
    before: CompleteElboEvaluation | None
    partial_after: PartialFactorEvaluation | None
    expected_factor_ids: tuple[str, ...]
    expected_affected_factor_ids: tuple[str, ...]
    observed_factor_ids: tuple[str, ...]
    observed_affected_factor_ids: tuple[str, ...]
    value_changed_factor_ids: tuple[str, ...]
    missing_factor_ids: tuple[str, ...]
    extra_factor_ids: tuple[str, ...]
    decision_delta_elbo: float | None
    decision_epsilon_delta: float | None
    attempted_accept: bool | None
    deterministic_reevaluation: DeterministicReevaluationRecord | None
    hashes: UpdateHashRecord
    obligations: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != "h5-failed-attempt-v1":
            raise ValueError("unsupported failed-attempt schema")
        if not isinstance(self.request, UpdateRequest):
            raise ValueError("request must be an UpdateRequest")
        if self.producer_label is not None and not isinstance(self.producer_label, UpdateLabel):
            raise ValueError("producer_label must be an UpdateLabel or None")
        if not isinstance(self.phase, AttemptPhase) or not isinstance(self.reason, AttemptFailureReason):
            raise ValueError("failure phase/reason are invalid")
        if self.expected_factor_ids != H5_FACTOR_UNIVERSE:
            raise ValueError("expected factor universe changed")
        if self.hashes.request_sha256 != self.request.request_sha256:
            raise ValueError("hash record does not match the request")
        if type(self.obligations) is not tuple or not self.obligations:
            raise ValueError("failed attempts require an explicit obligation")
        for name in ("decision_delta_elbo", "decision_epsilon_delta"):
            value = getattr(self, name)
            if value is not None:
                _finite(value, name)
        if self.attempted_accept is not None and type(self.attempted_accept) is not bool:
            raise ValueError("attempted_accept must be bool or None")
        has_final_candidate = self.hashes.candidate_sha256 is not None
        has_predecision = self.hashes.predecision_live_sha256 is not None
        if self.phase in (AttemptPhase.REQUEST, AttemptPhase.BEFORE_EVALUATION):
            if self.before is not None or self.producer_label is not None or has_final_candidate or has_predecision:
                raise ValueError("early failure exposes unavailable proposal/evaluation evidence")
        elif self.before is None:
            raise ValueError("post-before-evaluation failures require the before evaluation")
        if self.phase is AttemptPhase.FREEZE:
            has_draft = self.hashes.candidate_draft_sha256 is not None
            if (
                has_final_candidate
                or has_predecision
                or self.partial_after is not None
                or self.decision_delta_elbo is not None
                or self.attempted_accept is not None
            ):
                raise ValueError("freeze failure hash/evaluation fields are phase-invalid")
            if self.reason is AttemptFailureReason.LABEL_PROVENANCE_MISMATCH:
                if not has_draft:
                    raise ValueError("label-provenance freeze failure requires a rejected draft")
            elif has_draft:
                raise ValueError("generic freeze failure cannot fabricate a candidate draft")
        if self.phase in (
            AttemptPhase.AFTER_EVALUATION,
            AttemptPhase.DEPENDENCY_VALIDATION,
            AttemptPhase.DECISION,
            AttemptPhase.COMMIT_OR_ROLLBACK,
        ) and (not has_final_candidate or not has_predecision):
            raise ValueError("post-freeze failures require final candidate and predecision hashes")
        if self.phase is AttemptPhase.DECISION and (
            self.decision_delta_elbo is None
            or self.decision_epsilon_delta is None
            or self.attempted_accept is None
        ):
            raise ValueError("decision failure requires all decision diagnostics")
        if self.phase is AttemptPhase.COMMIT_OR_ROLLBACK and (
            self.decision_delta_elbo is None
            or self.decision_epsilon_delta is None
            or self.attempted_accept is None
        ):
            raise ValueError("commit/rollback failure requires all decision diagnostics")
        if self.reason is AttemptFailureReason.DETERMINISTIC_REEVALUATION_MISMATCH:
            if self.deterministic_reevaluation is None or self.partial_after is None:
                raise ValueError("deterministic mismatch requires partial and recheck evidence")
        elif self.deterministic_reevaluation is not None:
            raise ValueError("only deterministic mismatch may carry a recheck")


H5AttemptOutcome: TypeAlias = CompletedUpdateAttempt | FailedUpdateAttempt


@dataclass(frozen=True)
class H5TransactionResult:
    schema_version: Literal["h5-transaction-result-v1"]
    live: H5LiveState
    outcome: H5AttemptOutcome

    def __post_init__(self) -> None:
        if self.schema_version != "h5-transaction-result-v1":
            raise ValueError("unsupported transaction-result schema")
        if not isinstance(self.live, H5LiveState):
            raise ValueError("live must be an H5LiveState")
        if not isinstance(self.outcome, (CompletedUpdateAttempt, FailedUpdateAttempt)):
            raise ValueError("outcome must be a typed H5 attempt")


def _checked_tensor(value: object, name: str, *, ndim: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if value.device.type != "cpu" or value.dtype is not torch.float64 or value.ndim != ndim:
        raise ValueError(f"{name} must be a CPU float64 tensor of rank {ndim}")
    if not value.requires_grad or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite and require gradients")
    return value


def _check_no_alias(tensors: tuple[torch.Tensor, ...]) -> None:
    pointers = tuple(item.untyped_storage().data_ptr() for item in tensors)
    if len(set(pointers)) != len(pointers):
        raise ValueError("differentiable leaves must not share storage")


@dataclass
class DifferentiableRecognitionState:
    active_coordinate_ids: tuple[str, ...]
    mean_leaves: Mapping[str, torch.Tensor]
    log_variance_leaves: Mapping[str, torch.Tensor]
    categorical_logit_leaves: Mapping[str, torch.Tensor]

    def __post_init__(self) -> None:
        if type(self.active_coordinate_ids) is not tuple:
            raise ValueError("active_coordinate_ids must be a tuple")
        if any(item not in H5_RECOGNITION_COORDINATE_UNIVERSE for item in self.active_coordinate_ids):
            raise ValueError("active coordinate is outside H5")
        expected_order = tuple(
            item for item in H5_RECOGNITION_COORDINATE_UNIVERSE if item in self.active_coordinate_ids
        )
        if self.active_coordinate_ids != expected_order or len(set(self.active_coordinate_ids)) != len(self.active_coordinate_ids):
            raise ValueError("active coordinates must be unique and use universe order")
        mappings = (
            ("mean_leaves", self.mean_leaves, _GAUSSIAN_IDS, 0),
            ("log_variance_leaves", self.log_variance_leaves, _GAUSSIAN_IDS, 0),
            ("categorical_logit_leaves", self.categorical_logit_leaves, _CATEGORICAL_IDS, 1),
        )
        all_keys: set[str] = set()
        leaves: list[torch.Tensor] = []
        for name, mapping, universe, ndim in mappings:
            if not isinstance(mapping, Mapping):
                raise ValueError(f"{name} must be a mapping")
            copied = dict(mapping)
            if any(key not in universe or key not in self.active_coordinate_ids for key in copied):
                raise ValueError(f"{name} contains an inactive coordinate")
            for key, value in copied.items():
                leaves.append(_checked_tensor(value, f"{name}[{key}]", ndim=ndim))
            all_keys.update(copied)
            setattr(self, name, MappingProxyType(copied))
        if all_keys != set(self.active_coordinate_ids):
            raise ValueError("every active recognition coordinate needs at least one leaf")
        if set(self.categorical_logit_leaves) & (
            set(self.mean_leaves) | set(self.log_variance_leaves)
        ):
            raise ValueError("categorical and Gaussian leaf domains cannot overlap")
        _check_no_alias(tuple(leaves))


@dataclass
class DifferentiableModelState:
    active_block_ids: tuple[str, ...]
    unconstrained_leaves: Mapping[str, torch.Tensor]

    def __post_init__(self) -> None:
        if type(self.active_block_ids) is not tuple:
            raise ValueError("active_block_ids must be a tuple")
        expected_order = tuple(
            item for item in H5_MODEL_BLOCK_UNIVERSE if item in self.active_block_ids
        )
        if self.active_block_ids != expected_order or len(set(self.active_block_ids)) != len(self.active_block_ids):
            raise ValueError("active model blocks must be unique and use universe order")
        if not isinstance(self.unconstrained_leaves, Mapping):
            raise ValueError("unconstrained_leaves must be a mapping")
        copied = dict(self.unconstrained_leaves)
        expected_keys = tuple(
            f"{block}.{name}" for block in self.active_block_ids for name in _MODEL_FIELDS[block]
        )
        if tuple(copied) != expected_keys:
            raise ValueError("model leaves must contain every active field in schema order")
        checked: list[torch.Tensor] = []
        for key, value in copied.items():
            ndim = 1 if any(key.endswith(f".{name}") for name in ("w_z", "w_m", "bias")) else 0
            tensor = _checked_tensor(value, f"unconstrained_leaves[{key}]", ndim=ndim)
            if ndim == 1 and tuple(tensor.shape) != (3,):
                raise ValueError("emission leaves must have shape (3,)")
            checked.append(tensor)
        _check_no_alias(tuple(checked))
        self.unconstrained_leaves = MappingProxyType(copied)


class _DraftRejected(ValueError):
    def __init__(self, draft: H5CandidateDraft, cause: ValueError) -> None:
        self.draft = draft
        self.cause = cause
        super().__init__(str(cause))


def _request_copy(request: UpdateRequest) -> UpdateRequest:
    return UpdateRequest(
        request.schema_version,
        request.request_id,
        request.rule,
        request.requested_label,
        request.variables,
        request.parameters,
        request.damping_schedule,
    )


def canonical_frozen_complement_bytes(
    reference: H5ReferenceState, live: H5LiveState, request: UpdateRequest
) -> bytes:
    if not isinstance(reference, H5ReferenceState) or not isinstance(live, H5LiveState):
        raise ValueError("reference and live must be typed H5 states")
    rebuilt = _request_copy(request)
    inactive_gaussians = tuple(
        (item.coordinate_id, _tensor_core(item.mean), _tensor_core(item.variance))
        for item in live.recognition.gaussians
        if item.coordinate_id not in rebuilt.variables
    )
    inactive_categoricals = tuple(
        (
            item.coordinate_id,
            item.support,
            item.conditioned_on,
            _tensor_core(item.probabilities),
        )
        for item in live.recognition.categoricals
        if item.coordinate_id not in rebuilt.variables
    )
    inactive_blocks = tuple(
        (
            block.block_id,
            tuple((name, _tensor_core(value)) for name, value in block.values),
        )
        for block in live.model.parameter_blocks
        if block.block_id not in rebuilt.parameters
    )
    return _canonical_json_bytes(
        {
            "schema_version": "h5-frozen-complement-v1",
            "reference_sha256": reference.reference_sha256,
            "request_sha256": rebuilt.request_sha256,
            "inactive_gaussians": inactive_gaussians,
            "inactive_categoricals": inactive_categoricals,
            "inactive_model_blocks": inactive_blocks,
            "model_reconstruction_records": tuple(
                (item.factor_id, item.bindings) for item in live.model.reconstruction_records
            ),
            "model_shared_groups": tuple(
                (item.group_id, item.source, item.consumers) for item in live.model.shared_groups
            ),
            "optimizer_sha256": live.optimizer_state.state_sha256,
            "rng_sha256": live.rng_state.state_sha256,
        }
    )


def _complement_sha256(reference: H5ReferenceState, live: H5LiveState, request: UpdateRequest) -> str:
    return hashlib.sha256(
        H5_FROZEN_COMPLEMENT_DOMAIN
        + canonical_frozen_complement_bytes(reference, live, request)
    ).hexdigest()


def _m_solution(live: H5LiveState) -> tuple[torch.Tensor, float, float]:
    gaussians = {
        item.coordinate_id: (item.mean.values[0], item.variance.values[0])
        for item in live.recognition.gaussians
    }
    categorical = {
        item.coordinate_id: item.probabilities.values
        for item in live.recognition.categoricals
    }
    gamma = categorical["q[model_source_b2]"]
    rows = (
        categorical["q[source_row_a2]"],
        categorical["q[state_source_a2_b1]"],
    )
    G = torch.zeros((4, 4), dtype=torch.float64)
    g = torch.zeros(4, dtype=torch.float64)
    z2_mean, z2_variance = gaussians["q[z2]"]
    for b in (0, 1):
        for a in (0, 1):
            weight = gamma[b] * rows[b][a]
            z_mean, z_variance = gaussians[f"q[z{a}]"]
            m_mean, m_variance = gaussians["q[m2]"]
            means = torch.tensor(
                (z_mean if a == 0 else 0.0, z_mean if a == 1 else 0.0, m_mean, 1.0),
                dtype=torch.float64,
            )
            moment = torch.outer(means, means)
            moment[a, a] += z_variance
            moment[2, 2] += m_variance
            G += weight * moment
            g += weight * means * z2_mean
    G = 0.5 * (G + G.T)
    chol = torch.linalg.cholesky(G)
    theta = torch.cholesky_solve(g[:, None], chol)[:, 0]
    residual = z2_variance + z2_mean * z2_mean - 2.0 * torch.dot(theta, g) + torch.dot(theta, G @ theta)
    condition = float(torch.linalg.cond(G).item())
    R = float(residual.item())
    if not bool(torch.isfinite(theta).all()) or not math.isfinite(R) or R <= 0.0:
        raise ValueError("exact H5 M solve produced an invalid candidate")
    return theta, R, condition


def _freeze_with_draft(
    reference: H5ReferenceState,
    live: H5LiveState,
    recognition_working: DifferentiableRecognitionState,
    model_working: DifferentiableModelState,
    *,
    request: UpdateRequest,
    producer_label: UpdateLabel,
    damping: float,
    expected_frozen_complement_sha256: str,
) -> tuple[H5CandidateSnapshot, H5CandidateDraft]:
    rebuilt = _request_copy(request)
    expected = _require_sha256(
        expected_frozen_complement_sha256, "expected_frozen_complement_sha256"
    )
    observed = _complement_sha256(reference, live, rebuilt)
    if observed != expected:
        raise ValueError("frozen complement SHA-256 mismatch")
    if recognition_working.active_coordinate_ids != rebuilt.variables:
        raise ValueError("recognition working state does not match the request")
    if model_working.active_block_ids != rebuilt.parameters:
        raise ValueError("model working state does not match the request")

    gaussians: list[GaussianRecognitionCoordinate] = []
    for item in live.recognition.gaussians:
        coordinate_id = item.coordinate_id
        mean = (
            FrozenTensorValue.from_tensor(
                recognition_working.mean_leaves[coordinate_id].detach()
            )
            if coordinate_id in recognition_working.mean_leaves
            else item.mean
        )
        variance = (
            FrozenTensorValue.from_tensor(
                torch.exp(
                    recognition_working.log_variance_leaves[coordinate_id].detach()
                )
            )
            if coordinate_id in recognition_working.log_variance_leaves
            else item.variance
        )
        gaussians.append(GaussianRecognitionCoordinate(coordinate_id, mean, variance))
    categoricals: list[CategoricalRecognitionCoordinate] = []
    for item in live.recognition.categoricals:
        probabilities = (
            FrozenTensorValue.from_tensor(
                torch.softmax(
                    recognition_working.categorical_logit_leaves[item.coordinate_id].detach(),
                    dim=0,
                )
            )
            if item.coordinate_id in recognition_working.categorical_logit_leaves
            else item.probabilities
        )
        categoricals.append(
            CategoricalRecognitionCoordinate(
                item.coordinate_id,
                item.support,
                item.conditioned_on,
                probabilities,
            )
        )
    recognition = RecognitionSnapshot(
        "h5-recognition-snapshot-v1", tuple(gaussians), tuple(categoricals)
    )

    blocks: list[ModelParameterBlock] = []
    for block in live.model.parameter_blocks:
        values = []
        for name, value in block.values:
            key = f"{block.block_id}.{name}"
            replacement = (
                FrozenTensorValue.from_tensor(model_working.unconstrained_leaves[key].detach())
                if key in model_working.unconstrained_leaves
                else value
            )
            values.append((name, replacement))
        blocks.append(ModelParameterBlock(block.block_id, tuple(values)))
    model = H5ModelSnapshot(
        "h5-model-snapshot-v1",
        tuple(blocks),
        live.model.reconstruction_records,
        live.model.shared_groups,
    )
    diagnostics: tuple[tuple[str, float], ...] = ()
    if rebuilt.rule is H5UpdateRule.EXACT_STATE_TRANSITION_2_M:
        _, _, condition = _m_solution(live)
        diagnostics = (("G_condition_number", condition),)
    draft = H5CandidateDraft(
        "h5-candidate-draft-v1",
        rebuilt.rule,
        rebuilt.request_sha256,
        producer_label,
        rebuilt.variables,
        rebuilt.parameters,
        _finite(damping, "damping"),
        diagnostics,
        recognition,
        model,
    )
    if (
        draft.rule is not rebuilt.rule
        or draft.request_sha256 != rebuilt.request_sha256
        or draft.variables != rebuilt.variables
        or draft.parameters != rebuilt.parameters
    ):
        raise ValueError("candidate draft does not reproduce the request")
    try:
        candidate = H5CandidateSnapshot(
            "h5-candidate-v1",
            draft.rule,
            draft.request_sha256,
            draft.producer_label,
            draft.variables,
            draft.parameters,
            draft.damping,
            draft.numerical_diagnostics,
            draft.recognition,
            draft.model,
        )
    except ValueError as exc:
        raise _DraftRejected(draft, exc) from exc
    return candidate, draft


def freeze_candidate(
    reference: H5ReferenceState,
    live: H5LiveState,
    recognition_working: DifferentiableRecognitionState,
    model_working: DifferentiableModelState,
    *,
    request: UpdateRequest,
    producer_label: UpdateLabel,
    damping: float,
    expected_frozen_complement_sha256: str,
) -> H5CandidateSnapshot:
    candidate, _ = _freeze_with_draft(
        reference,
        live,
        recognition_working,
        model_working,
        request=request,
        producer_label=producer_label,
        damping=damping,
        expected_frozen_complement_sha256=expected_frozen_complement_sha256,
    )
    return candidate


def _empty_recognition() -> DifferentiableRecognitionState:
    return DifferentiableRecognitionState((), {}, {}, {})


def _empty_model() -> DifferentiableModelState:
    return DifferentiableModelState((), {})


def _working_gaussian(
    coordinate_id: str, mean: float, variance: float, *, include_variance: bool
) -> DifferentiableRecognitionState:
    means = {
        coordinate_id: torch.tensor(mean, dtype=torch.float64, requires_grad=True)
    }
    log_variances = (
        {
            coordinate_id: torch.tensor(
                math.log(variance), dtype=torch.float64, requires_grad=True
            )
        }
        if include_variance
        else {}
    )
    return DifferentiableRecognitionState(
        (coordinate_id,), means, log_variances, {}
    )


def _require_rule(request: UpdateRequest, rule: H5UpdateRule) -> None:
    rebuilt = _request_copy(request)
    if rebuilt.rule is not rule:
        raise ValueError(f"request rule must equal {rule.value}")


def _exact_z0_working(
    reference: H5ReferenceState, live: H5LiveState
) -> DifferentiableRecognitionState:
    h1 = json.loads(reference.raw_h1_fixture_bytes)
    covariance = torch.tensor(h1["initial_joint"]["covariance"], dtype=torch.float64)
    mean = torch.tensor(h1["initial_joint"]["mean"], dtype=torch.float64)
    precision = torch.linalg.inv(covariance)
    information = precision @ mean
    qg = {
        item.coordinate_id: (item.mean.values[0], item.variance.values[0])
        for item in live.recognition.gaussians
    }
    qc = {
        item.coordinate_id: item.probabilities.values
        for item in live.recognition.categoricals
    }
    blocks = {
        block.block_id: {name: value.values for name, value in block.values}
        for block in live.model.parameter_blocks
    }
    transition = blocks["theta[state_transition_2]"]
    shared = blocks["theta[shared_decoder_transition]"]["s"][0]
    gamma = qc["q[model_source_b2]"]
    rows = (qc["q[source_row_a2]"], qc["q[state_source_a2_b1]"])
    w20 = math.fsum(gamma[b] * rows[b][0] for b in (0, 1))
    alpha10 = h1["frames"][1] / h1["frames"][0]
    B1 = h1["state_model_slopes"][0]
    c1 = h1["state_offsets"][0]
    R1 = h1["state_variances"][0]
    alpha20 = transition["alpha_0"][0]
    B2 = transition["B_base"][0] + shared
    c2 = transition["c"][0]
    R2 = transition["R"][0]
    J = float(precision[0, 0]) + alpha10 * alpha10 / R1 + w20 * alpha20 * alpha20 / R2
    h = (
        float(information[0])
        - float(precision[0, 1]) * qg["q[m0]"][0]
        + alpha10 * (qg["q[z1]"][0] - B1 * qg["q[m1]"][0] - c1) / R1
        + w20 * alpha20 * (qg["q[z2]"][0] - B2 * qg["q[m2]"][0] - c2) / R2
    )
    if not math.isfinite(J) or J <= 0.0:
        raise ValueError("exact z0 information precision must be positive")
    return _working_gaussian("q[z0]", h / J, 1.0 / J, include_variance=True)


def exact_conjugate_gaussian_e_update(
    reference: H5ReferenceState, live: H5LiveState, request: UpdateRequest
) -> H5CandidateSnapshot:
    _require_rule(request, H5UpdateRule.EXACT_Z0)
    working = _exact_z0_working(reference, live)
    return freeze_candidate(
        reference,
        live,
        working,
        _empty_model(),
        request=request,
        producer_label=request.requested_label,
        damping=1.0,
        expected_frozen_complement_sha256=_complement_sha256(reference, live, request),
    )


def _exact_source_row_working(
    reference: H5ReferenceState, live: H5LiveState
) -> DifferentiableRecognitionState:
    h1 = json.loads(reference.raw_h1_fixture_bytes)
    qg = {
        item.coordinate_id: (item.mean.values[0], item.variance.values[0])
        for item in live.recognition.gaussians
    }
    blocks = {
        block.block_id: {name: value.values for name, value in block.values}
        for block in live.model.parameter_blocks
    }
    transition = blocks["theta[state_transition_2]"]
    shared = blocks["theta[shared_decoder_transition]"]["s"][0]
    B = transition["B_base"][0] + shared
    c = transition["c"][0]
    R = transition["R"][0]
    logits: list[float] = []
    for a, alpha in enumerate((transition["alpha_0"][0], transition["alpha_1"][0])):
        parent_mean, parent_variance = qg[f"q[z{a}]"]
        target_mean, target_variance = qg["q[z2]"]
        model_mean, model_variance = qg["q[m2]"]
        residual = target_mean - alpha * parent_mean - B * model_mean - c
        ell = -0.5 * (
            _LOG_2_PI
            + math.log(R)
            + (
                target_variance
                + alpha * alpha * parent_variance
                + B * B * model_variance
                + residual * residual
            )
            / R
        )
        logits.append(math.log(h1["state_source_priors"][1][a]) + ell)
    working_logits = torch.tensor(logits, dtype=torch.float64, requires_grad=True)
    return DifferentiableRecognitionState(
        ("q[source_row_a2]",),
        {},
        {},
        {"q[source_row_a2]": working_logits},
    )


def exact_source_row_update(
    reference: H5ReferenceState, live: H5LiveState, request: UpdateRequest
) -> H5CandidateSnapshot:
    _require_rule(request, H5UpdateRule.EXACT_SOURCE_ROW_A2)
    working = _exact_source_row_working(reference, live)
    return freeze_candidate(
        reference,
        live,
        working,
        _empty_model(),
        request=request,
        producer_label=request.requested_label,
        damping=1.0,
        expected_frozen_complement_sha256=_complement_sha256(reference, live, request),
    )


def _model_working_from_values(
    block_id: str, values: Mapping[str, torch.Tensor | float | tuple[float, ...]]
) -> DifferentiableModelState:
    leaves: dict[str, torch.Tensor] = {}
    for name in _MODEL_FIELDS[block_id]:
        value = values[name]
        tensor = (
            value.detach().clone().requires_grad_(True)
            if isinstance(value, torch.Tensor)
            else torch.tensor(value, dtype=torch.float64, requires_grad=True)
        )
        leaves[f"{block_id}.{name}"] = tensor
    return DifferentiableModelState((block_id,), leaves)


def _exact_m_working(live: H5LiveState) -> DifferentiableModelState:
    theta, R, _ = _m_solution(live)
    shared = next(
        value.values[0]
        for block in live.model.parameter_blocks
        if block.block_id == "theta[shared_decoder_transition]"
        for name, value in block.values
        if name == "s"
    )
    return _model_working_from_values(
        "theta[state_transition_2]",
        {
            "alpha_0": float(theta[0].item()),
            "alpha_1": float(theta[1].item()),
            "B_base": float(theta[2].item()) - shared,
            "c": float(theta[3].item()),
            "R": R,
        },
    )


def exact_gaussian_m_update(
    reference: H5ReferenceState, live: H5LiveState, request: UpdateRequest
) -> H5CandidateSnapshot:
    _require_rule(request, H5UpdateRule.EXACT_STATE_TRANSITION_2_M)
    working = _exact_m_working(live)
    return freeze_candidate(
        reference,
        live,
        _empty_recognition(),
        working,
        request=request,
        producer_label=request.requested_label,
        damping=1.0,
        expected_frozen_complement_sha256=_complement_sha256(reference, live, request),
    )


def _resolved_tensors(
    live: H5LiveState,
    recognition_working: DifferentiableRecognitionState,
    model_working: DifferentiableModelState,
) -> tuple[
    dict[str, tuple[torch.Tensor, torch.Tensor]],
    dict[str, torch.Tensor],
    dict[str, dict[str, torch.Tensor]],
]:
    gaussians: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for item in live.recognition.gaussians:
        mean = recognition_working.mean_leaves.get(
            item.coordinate_id,
            torch.tensor(item.mean.values[0], dtype=torch.float64),
        )
        variance = (
            torch.exp(recognition_working.log_variance_leaves[item.coordinate_id])
            if item.coordinate_id in recognition_working.log_variance_leaves
            else torch.tensor(item.variance.values[0], dtype=torch.float64)
        )
        gaussians[item.coordinate_id] = (mean, variance)
    categoricals: dict[str, torch.Tensor] = {}
    for item in live.recognition.categoricals:
        categoricals[item.coordinate_id] = (
            torch.softmax(
                recognition_working.categorical_logit_leaves[item.coordinate_id], dim=0
            )
            if item.coordinate_id in recognition_working.categorical_logit_leaves
            else torch.tensor(item.probabilities.values, dtype=torch.float64)
        )
    blocks: dict[str, dict[str, torch.Tensor]] = {}
    for block in live.model.parameter_blocks:
        blocks[block.block_id] = {
            name: model_working.unconstrained_leaves.get(
                f"{block.block_id}.{name}", value.to_tensor()
            )
            for name, value in block.values
        }
    return gaussians, categoricals, blocks


def _torch_log_normal(
    target: tuple[torch.Tensor, torch.Tensor],
    parents: tuple[tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]], ...],
    offset: torch.Tensor,
    variance: torch.Tensor,
) -> torch.Tensor:
    residual_mean = target[0] - offset - sum(
        coefficient * parent[0] for coefficient, parent in parents
    )
    residual_variance = target[1] + sum(
        coefficient * coefficient * parent[1] for coefficient, parent in parents
    )
    return -0.5 * (
        _LOG_2_PI
        + torch.log(variance)
        + (residual_variance + residual_mean * residual_mean) / variance
    )


def differentiable_h5_complete_elbo_order_21(
    reference: H5ReferenceState,
    live: H5LiveState,
    recognition_working: DifferentiableRecognitionState,
    model_working: DifferentiableModelState,
) -> torch.Tensor:
    if not isinstance(reference, H5ReferenceState) or not isinstance(live, H5LiveState):
        raise ValueError("reference/live must be typed H5 states")
    if not isinstance(recognition_working, DifferentiableRecognitionState) or not isinstance(model_working, DifferentiableModelState):
        raise ValueError("working states must be typed")
    h1 = json.loads(reference.raw_h1_fixture_bytes)
    qg, qc, blocks = _resolved_tensors(live, recognition_working, model_working)
    dtype = torch.float64
    total = torch.zeros((), dtype=dtype)

    p_mean = torch.tensor(h1["initial_joint"]["mean"], dtype=dtype)
    p_covariance = torch.tensor(h1["initial_joint"]["covariance"], dtype=dtype)
    precision = torch.linalg.inv(p_covariance)
    q_mean = torch.stack((qg["q[z0]"][0], qg["q[m0]"][0]))
    q_covariance = torch.diag(torch.stack((qg["q[z0]"][1], qg["q[m0]"][1])))
    displacement = q_mean - p_mean
    total = total - _LOG_2_PI - 0.5 * torch.logdet(p_covariance)
    total = total - 0.5 * (
        torch.trace(precision @ q_covariance) + displacement @ precision @ displacement
    )

    shared = blocks["theta[shared_decoder_transition]"]["s"]
    for time in (1, 2):
        gamma = qc[f"q[model_source_b{time}]"]
        prior = torch.tensor(h1["model_source_priors"][time - 1], dtype=dtype)
        total = total + torch.sum(gamma * torch.log(prior))
        rows = (
            (qc["q[state_source_a1_b0]"],)
            if time == 1
            else (qc["q[source_row_a2]"], qc["q[state_source_a2_b1]"])
        )
        state_prior = torch.tensor(h1["state_source_priors"][time - 1], dtype=dtype)
        total = total + sum(
            gamma[b] * torch.sum(row * torch.log(state_prior))
            for b, row in enumerate(rows)
        )
        if time == 1:
            alpha_m = (torch.tensor(h1["frames"][1] / h1["frames"][0], dtype=dtype),)
            c_m = torch.tensor(h1["model_offsets"][0], dtype=dtype)
            R_m = torch.tensor(h1["model_variances"][0], dtype=dtype)
            alpha_z = alpha_m
            B = torch.tensor(h1["state_model_slopes"][0], dtype=dtype)
            c_z = torch.tensor(h1["state_offsets"][0], dtype=dtype)
            R_z = torch.tensor(h1["state_variances"][0], dtype=dtype)
        else:
            alpha_m = tuple(
                torch.tensor(value, dtype=dtype)
                for value in (
                    h1["frames"][2] / h1["frames"][0],
                    h1["frames"][2] / h1["frames"][1],
                )
            )
            c_m = torch.tensor(h1["model_offsets"][1], dtype=dtype)
            R_m = torch.tensor(h1["model_variances"][1], dtype=dtype)
            transition = blocks["theta[state_transition_2]"]
            alpha_z = (transition["alpha_0"], transition["alpha_1"])
            B = transition["B_base"] + shared
            c_z = transition["c"]
            R_z = transition["R"]
        total = total + sum(
            gamma[b]
            * _torch_log_normal(
                qg[f"q[m{time}]"],
                ((alpha_m[b], qg[f"q[m{b}]"]),),
                c_m,
                R_m,
            )
            for b in range(len(alpha_m))
        )
        total = total + sum(
            gamma[b]
            * row[a]
            * _torch_log_normal(
                qg[f"q[z{time}]"],
                ((alpha_z[a], qg[f"q[z{a}]"]), (B, qg[f"q[m{time}]"])),
                c_z,
                R_z,
            )
            for b, row in enumerate(rows)
            for a in range(len(row))
        )

        if time == 1:
            emission = blocks["theta[emission_1]"]
            w_z = emission["w_z"].clone()
            w_z = torch.cat(((w_z[0] + shared).reshape(1), w_z[1:]))
            w_m = emission["w_m"]
            bias = emission["bias"]
        else:
            decoder = h1["decoder"][1]
            w_z_base = torch.tensor(decoder["w_z"], dtype=dtype)
            w_z = torch.cat(((w_z_base[0] + shared).reshape(1), w_z_base[1:]))
            w_m = torch.tensor(decoder["w_m"], dtype=dtype)
            bias = torch.tensor(decoder["bias"], dtype=dtype)
        nodes, weights = probabilists_gauss_hermite(21, dtype=dtype)
        z_points = qg[f"q[z{time}]"][0] + torch.sqrt(qg[f"q[z{time}]"][1]) * nodes
        m_points = qg[f"q[m{time}]"][0] + torch.sqrt(qg[f"q[m{time}]"][1]) * nodes
        z_grid = z_points[:, None, None]
        m_grid = m_points[None, :, None]
        logits = z_grid * w_z + m_grid * w_m + bias
        selected = torch.log_softmax(logits, dim=-1)[
            :, :, h1["observation_labels"][time - 1] - 1
        ]
        total = total + torch.sum(weights[:, None] * weights[None, :] * selected)

    entropy = sum(
        0.5 * torch.log(2.0 * math.pi * math.e * variance)
        for _, variance in qg.values()
    )
    for time in (1, 2):
        gamma = qc[f"q[model_source_b{time}]"]
        entropy = entropy - torch.sum(gamma * torch.log(gamma))
        rows = (
            (qc["q[state_source_a1_b0]"],)
            if time == 1
            else (qc["q[source_row_a2]"], qc["q[state_source_a2_b1]"])
        )
        entropy = entropy + sum(
            gamma[b] * (-torch.sum(row * torch.log(row)))
            for b, row in enumerate(rows)
        )
    total = total + entropy
    if not bool(torch.isfinite(total)):
        raise ValueError("differentiable H5 complete objective is nonfinite")
    return total


def _live_block_values(live: H5LiveState, block_id: str) -> dict[str, torch.Tensor]:
    block = next(item for item in live.model.parameter_blocks if item.block_id == block_id)
    return {name: value.to_tensor() for name, value in block.values}


def propose_generalized_em(
    reference: H5ReferenceState,
    live: H5LiveState,
    request: UpdateRequest,
    damping: float,
) -> H5CandidateSnapshot:
    _require_rule(request, H5UpdateRule.GENERALIZED_EM_EMISSION_1)
    if type(damping) is not float or damping not in request.damping_schedule:
        raise ValueError("damping is outside the frozen GEM schedule")
    current, gradients = _generalized_em_direction(reference, live, request)
    return _generalized_em_candidate(
        reference, live, request, damping, current, gradients
    )


def _require_differentiable_objective_concordance(
    reference: H5ReferenceState,
    live: H5LiveState,
    request: UpdateRequest,
    differentiable_objective: torch.Tensor,
) -> None:
    complement = _complement_sha256(reference, live, request)
    task6 = evaluate_h5_complete_elbo(
        reference,
        live,
        frozen_complement_sha256=complement,
    )
    detached = float(differentiable_objective.detach().item())
    rounding_allowance = math.fsum(
        tuple(item.rounding_order_21 for item in task6.term_allowances)
        + (task6.complete_allowance.reduction_rounding,)
    )
    residual = abs(detached - task6.terms.complete_elbo)
    if not math.isfinite(residual) or residual > rounding_allowance:
        raise ValueError(
            "differentiable complete objective disagrees with the Task 6 order-21 objective"
        )


def _generalized_em_direction(
    reference: H5ReferenceState, live: H5LiveState, request: UpdateRequest
) -> tuple[dict[str, torch.Tensor], tuple[torch.Tensor, ...]]:
    current = _live_block_values(live, "theta[emission_1]")
    working = _model_working_from_values("theta[emission_1]", current)
    objective = differentiable_h5_complete_elbo_order_21(
        reference, live, _empty_recognition(), working
    )
    _require_differentiable_objective_concordance(
        reference, live, request, objective
    )
    leaves = tuple(working.unconstrained_leaves.values())
    gradients = torch.autograd.grad(objective, leaves)
    return current, tuple(gradient.detach().clone() for gradient in gradients)


def _generalized_em_candidate(
    reference: H5ReferenceState,
    live: H5LiveState,
    request: UpdateRequest,
    damping: float,
    current: Mapping[str, torch.Tensor],
    gradients: tuple[torch.Tensor, ...],
) -> H5CandidateSnapshot:
    proposed_working = _generalized_em_working(current, gradients, damping)
    return freeze_candidate(
        reference,
        live,
        _empty_recognition(),
        proposed_working,
        request=request,
        producer_label=request.requested_label,
        damping=damping,
        expected_frozen_complement_sha256=_complement_sha256(reference, live, request),
    )


def _generalized_em_working(
    current: Mapping[str, torch.Tensor],
    gradients: tuple[torch.Tensor, ...],
    damping: float,
) -> DifferentiableModelState:
    proposed = {
        name: current[name] + damping * gradient
        for name, gradient in zip(_MODEL_FIELDS["theta[emission_1]"], gradients, strict=True)
    }
    return _model_working_from_values("theta[emission_1]", proposed)


def propose_natural_gradient(
    reference: H5ReferenceState,
    live: H5LiveState,
    request: UpdateRequest,
    step_size: float,
) -> H5CandidateSnapshot:
    _require_rule(request, H5UpdateRule.NATURAL_GRADIENT_Z1)
    if type(step_size) is not float or step_size not in request.damping_schedule:
        raise ValueError("step_size is outside the frozen natural-gradient schedule")
    proposed = _natural_gradient_working(reference, live, request, step_size)
    return freeze_candidate(
        reference,
        live,
        proposed,
        _empty_model(),
        request=request,
        producer_label=request.requested_label,
        damping=step_size,
        expected_frozen_complement_sha256=_complement_sha256(reference, live, request),
    )


def _natural_gradient_working(
    reference: H5ReferenceState,
    live: H5LiveState,
    request: UpdateRequest,
    step_size: float,
) -> DifferentiableRecognitionState:
    coordinate = next(
        item for item in live.recognition.gaussians if item.coordinate_id == "q[z1]"
    )
    mean = torch.tensor(
        coordinate.mean.values[0], dtype=torch.float64, requires_grad=True
    )
    working = DifferentiableRecognitionState(("q[z1]",), {"q[z1]": mean}, {}, {})
    objective = differentiable_h5_complete_elbo_order_21(
        reference, live, working, _empty_model()
    )
    _require_differentiable_objective_concordance(
        reference, live, request, objective
    )
    (gradient,) = torch.autograd.grad(objective, (mean,))
    proposed_mean = mean.detach() + step_size * coordinate.variance.values[0] * gradient.detach()
    return DifferentiableRecognitionState(
        ("q[z1]",),
        {"q[z1]": proposed_mean.clone().requires_grad_(True)},
        {},
        {},
    )


def _draft_from_candidate(candidate: H5CandidateSnapshot) -> H5CandidateDraft:
    return H5CandidateDraft(
        "h5-candidate-draft-v1",
        candidate.rule,
        candidate.request_sha256,
        candidate.producer_label,
        candidate.variables,
        candidate.parameters,
        candidate.damping,
        candidate.numerical_diagnostics,
        candidate.recognition,
        candidate.model,
    )


def _cache(evaluation: CompleteElboEvaluation) -> dict[FactorCacheKey, FactorCacheEntry]:
    return {
        FactorCacheKey(
            record.factor_id,
            record.input_hash,
            (21, 17),
            record.frozen_complement_sha256,
        ): FactorCacheEntry(
            FactorCacheKey(
                record.factor_id,
                record.input_hash,
                (21, 17),
                record.frozen_complement_sha256,
            ),
            record,
        )
        for record in evaluation.factor_records
    }


def _evaluate(
    evaluator: CompleteElboEvaluator,
    state: H5LiveState | H5CandidateSnapshot,
    complement: str,
    cache: Mapping[FactorCacheKey, FactorCacheEntry] | None,
) -> CompleteElboEvaluation:
    result = evaluator.evaluate(
        state, frozen_complement_sha256=complement, cache=cache
    )
    if not isinstance(result, CompleteElboEvaluation):
        raise ValueError("H5 evaluator returned an invalid result")
    return result


def _live_hashes(live: H5LiveState) -> tuple[str, str, str, str, str]:
    return (
        live.state_sha256,
        live.recognition.state_sha256,
        live.model.state_sha256,
        live.optimizer_state.state_sha256,
        live.rng_state.state_sha256,
    )


def _hash_record(
    request: UpdateRequest,
    before_live: H5LiveState,
    complement: str,
    final_live: H5LiveState,
    *,
    draft: H5CandidateDraft | None,
    candidate: H5CandidateSnapshot | None,
    predecision: bool,
) -> UpdateHashRecord:
    before = _live_hashes(before_live)
    final = _live_hashes(final_live)
    return UpdateHashRecord(
        "h5-update-hash-record-v1",
        request.request_sha256,
        *before,
        before[0] if predecision else None,
        before[3] if predecision else None,
        before[4] if predecision else None,
        draft.candidate_draft_sha256 if draft is not None else None,
        candidate.candidate_sha256 if candidate is not None else None,
        draft.recognition.state_sha256 if draft is not None else None,
        draft.model.state_sha256 if draft is not None else None,
        complement,
        *final,
    )


def _failed(
    request: UpdateRequest,
    live: H5LiveState,
    complement: str,
    *,
    producer: UpdateLabel | None,
    phase: AttemptPhase,
    reason: AttemptFailureReason,
    before: CompleteElboEvaluation | None,
    draft: H5CandidateDraft | None = None,
    candidate: H5CandidateSnapshot | None = None,
    predecision: bool = False,
    partial: PartialFactorEvaluation | None = None,
    expected_affected: tuple[str, ...] = (),
    observed_affected: tuple[str, ...] = (),
    value_changed: tuple[str, ...] = (),
    delta: float | None = None,
    epsilon: float | None = None,
    attempted_accept: bool | None = None,
    deterministic: DeterministicReevaluationRecord | None = None,
    final_for_hash: H5LiveState | None = None,
    obligation: str,
) -> H5TransactionResult:
    final_hash_state = final_for_hash if final_for_hash is not None else live
    observed_ids = (
        tuple(item.factor_id for item in partial.observed_records) if partial else ()
    )
    missing = partial.missing_factor_ids if partial else ()
    extra = partial.extra_factor_ids if partial else ()
    hashes = _hash_record(
        request,
        live,
        complement,
        final_hash_state,
        draft=draft,
        candidate=candidate,
        predecision=predecision,
    )
    failure = FailedUpdateAttempt(
        "h5-failed-attempt-v1",
        request,
        producer,
        phase,
        reason,
        before,
        partial,
        H5_FACTOR_UNIVERSE,
        expected_affected,
        observed_ids,
        observed_affected,
        value_changed,
        missing,
        extra,
        delta,
        epsilon,
        attempted_accept,
        deterministic,
        hashes,
        (obligation,),
    )
    return H5TransactionResult("h5-transaction-result-v1", live, failure)


def _candidate_as_live(live: H5LiveState, candidate: H5CandidateSnapshot) -> H5LiveState:
    return H5LiveState(
        "h5-live-state-v1",
        candidate.recognition,
        candidate.model,
        live.optimizer_state,
        live.rng_state,
    )


def _reflected_input_working(live: H5LiveState) -> DifferentiableModelState:
    values = _live_block_values(live, "theta[state_transition_2]")
    qg = {
        item.coordinate_id: (item.mean.values[0], item.variance.values[0])
        for item in live.recognition.gaussians
    }
    shared = _live_block_values(live, "theta[shared_decoder_transition]")["s"].item()
    B = values["B_base"].item() + shared
    numerator = qg["q[z0]"][0] * (
        qg["q[z2]"][0] - B * qg["q[m2]"][0] - values["c"].item()
    )
    denominator = qg["q[z0]"][1] + qg["q[z0]"][0] ** 2
    alpha_hat = numerator / denominator
    values["alpha_0"] = torch.tensor(
        2.0 * alpha_hat - values["alpha_0"].item(), dtype=torch.float64
    )
    return _model_working_from_values("theta[state_transition_2]", values)


def _identity_m_working(live: H5LiveState) -> DifferentiableModelState:
    return _model_working_from_values(
        "theta[state_transition_2]",
        _live_block_values(live, "theta[state_transition_2]"),
    )


def execute_update(
    reference: H5ReferenceState,
    live: H5LiveState,
    request: UpdateRequest,
    evaluator: CompleteElboEvaluator,
    budget: H5BudgetConfig,
    *,
    fault_injection: H5FaultInjection | None = None,
) -> H5TransactionResult:
    if not isinstance(reference, H5ReferenceState) or not isinstance(live, H5LiveState):
        raise ValueError("reference and live must be typed H5 states")
    rebuilt = _request_copy(request)
    if not isinstance(budget, H5BudgetConfig):
        raise ValueError("budget must be an H5BudgetConfig")
    if fault_injection is not None and not isinstance(fault_injection, H5FaultInjection):
        raise ValueError("fault_injection must be typed or None")
    complement = _complement_sha256(reference, live, rebuilt)
    captured_before_hashes = _live_hashes(live)
    graph = build_h5_reference_dependency_graph(reference.specification)
    expected_affected = expected_affected_factors(
        graph, variables=rebuilt.variables, parameters=rebuilt.parameters
    )
    try:
        before = _evaluate(evaluator, live, complement, None)
    except StaleFactorCacheError:
        return _failed(
            rebuilt,
            live,
            complement,
            producer=None,
            phase=AttemptPhase.BEFORE_EVALUATION,
            reason=AttemptFailureReason.STALE_CACHE,
            before=None,
            expected_affected=expected_affected,
            obligation="before evaluation cache was stale",
        )

    candidate: H5CandidateSnapshot | None = None
    draft: H5CandidateDraft | None = None
    chosen_after: CompleteElboEvaluation | None = None
    line_search_step: int | None = None
    autograd_scope: tuple[str, ...] = ()
    recognition_working = _empty_recognition()
    model_working = _empty_model()
    damping = rebuilt.damping_schedule[0]
    producer_label = rebuilt.requested_label
    gem_current: dict[str, torch.Tensor] | None = None
    gem_direction: tuple[torch.Tensor, ...] | None = None
    force_unresolved_gem = (
        fault_injection is not None
        and fault_injection.kind is H5FaultKind.FORCE_UNRESOLVED_GEM_ACCEPT
    )
    try:
        if rebuilt.rule is H5UpdateRule.EXACT_Z0:
            recognition_working = _exact_z0_working(reference, live)
        elif rebuilt.rule is H5UpdateRule.EXACT_SOURCE_ROW_A2:
            recognition_working = _exact_source_row_working(reference, live)
        elif rebuilt.rule is H5UpdateRule.EXACT_STATE_TRANSITION_2_M:
            model_working = (
                _reflected_input_working(live)
                if fault_injection is not None
                and fault_injection.kind is H5FaultKind.CHANGE_INPUT_KEEP_VALUE
                else _identity_m_working(live)
                if fault_injection is not None
                and fault_injection.kind is H5FaultKind.CHANGE_VALUE_KEEP_INPUT
                else _exact_m_working(live)
            )
        elif rebuilt.rule is H5UpdateRule.GENERALIZED_EM_EMISSION_1:
            autograd_scope = (
                "theta[emission_1].w_z",
                "theta[emission_1].w_m",
                "theta[emission_1].bias",
            )
            if force_unresolved_gem:
                current = _live_block_values(live, "theta[emission_1]")
                current["bias"] = current["bias"].clone()
                current["bias"][0] = math.nextafter(current["bias"][0].item(), math.inf)
                model_working = _model_working_from_values(
                    "theta[emission_1]", current
                )
            else:
                gem_current, gem_direction = _generalized_em_direction(
                    reference, live, rebuilt
                )
        else:
            autograd_scope = ("q[z1].mean",)
            recognition_working = _natural_gradient_working(
                reference, live, rebuilt, damping
            )
            if (
                fault_injection is not None
                and fault_injection.kind is H5FaultKind.MISLABEL_NATURAL_AS_EXACT
            ):
                producer_label = UpdateLabel.EXACT_COORDINATE
    except (RuntimeError, ValueError) as exc:
        return _failed(
            rebuilt,
            live,
            complement,
            producer=rebuilt.requested_label,
            phase=AttemptPhase.PROPOSAL,
            reason=AttemptFailureReason.NONFINITE_OR_INVALID_CANDIDATE,
            before=before,
            expected_affected=expected_affected,
            obligation=f"proposal failed: {exc}",
        )

    if rebuilt.rule is not H5UpdateRule.GENERALIZED_EM_EMISSION_1 or force_unresolved_gem:
        try:
            candidate, draft = _freeze_with_draft(
                reference,
                live,
                recognition_working,
                model_working,
                request=rebuilt,
                producer_label=producer_label,
                damping=damping,
                expected_frozen_complement_sha256=complement,
            )
        except _DraftRejected as exc:
            return _failed(
                rebuilt,
                live,
                complement,
                producer=exc.draft.producer_label,
                phase=AttemptPhase.FREEZE,
                reason=AttemptFailureReason.LABEL_PROVENANCE_MISMATCH,
                before=before,
                draft=exc.draft,
                expected_affected=expected_affected,
                obligation="candidate label provenance was rejected before final freeze",
            )
        except (RuntimeError, ValueError) as exc:
            return _failed(
                rebuilt,
                live,
                complement,
                producer=producer_label,
                phase=AttemptPhase.FREEZE,
                reason=AttemptFailureReason.NONFINITE_OR_INVALID_CANDIDATE,
                before=before,
                expected_affected=expected_affected,
                obligation=f"candidate freeze failed: {exc}",
            )

    if rebuilt.rule is H5UpdateRule.GENERALIZED_EM_EMISSION_1 and not force_unresolved_gem:
        if gem_current is None or gem_direction is None:
            return _failed(
                rebuilt,
                live,
                complement,
                producer=producer_label,
                phase=AttemptPhase.PROPOSAL,
                reason=AttemptFailureReason.NONFINITE_OR_INVALID_CANDIDATE,
                before=before,
                expected_affected=expected_affected,
                obligation="generalized-EM proposal did not produce a direction",
            )
        for index, damping in enumerate(rebuilt.damping_schedule):
            model_working = _generalized_em_working(
                gem_current, gem_direction, damping
            )
            try:
                proposed, proposed_draft = _freeze_with_draft(
                    reference,
                    live,
                    _empty_recognition(),
                    model_working,
                    request=rebuilt,
                    producer_label=producer_label,
                    damping=damping,
                    expected_frozen_complement_sha256=complement,
                )
            except _DraftRejected as exc:
                return _failed(
                    rebuilt,
                    live,
                    complement,
                    producer=exc.draft.producer_label,
                    phase=AttemptPhase.FREEZE,
                    reason=AttemptFailureReason.LABEL_PROVENANCE_MISMATCH,
                    before=before,
                    draft=exc.draft,
                    expected_affected=expected_affected,
                    obligation="candidate label provenance was rejected before final freeze",
                )
            except (RuntimeError, ValueError) as exc:
                return _failed(
                    rebuilt,
                    live,
                    complement,
                    producer=producer_label,
                    phase=AttemptPhase.FREEZE,
                    reason=AttemptFailureReason.NONFINITE_OR_INVALID_CANDIDATE,
                    before=before,
                    expected_affected=expected_affected,
                    obligation=f"candidate freeze failed: {exc}",
                )
            candidate = proposed
            draft = proposed_draft
            line_search_step = index
            if _live_hashes(live) != captured_before_hashes:
                return _failed(
                    rebuilt,
                    live,
                    complement,
                    producer=producer_label,
                    phase=AttemptPhase.COMMIT_OR_ROLLBACK,
                    reason=AttemptFailureReason.ROLLBACK_HASH_MISMATCH,
                    before=before,
                    draft=draft,
                    candidate=candidate,
                    predecision=True,
                    expected_affected=expected_affected,
                    obligation="proposal mutated caller-owned live state before evaluation",
                )
            try:
                proposed_after = _evaluate(
                    evaluator, proposed, complement, _cache(before)
                )
            except StaleFactorCacheError:
                return _failed(
                    rebuilt,
                    live,
                    complement,
                    producer=producer_label,
                    phase=AttemptPhase.AFTER_EVALUATION,
                    reason=AttemptFailureReason.STALE_CACHE,
                    before=before,
                    draft=draft,
                    candidate=candidate,
                    predecision=True,
                    expected_affected=expected_affected,
                    obligation="after evaluation cache was stale",
                )
            except (RuntimeError, ValueError) as exc:
                return _failed(
                    rebuilt,
                    live,
                    complement,
                    producer=producer_label,
                    phase=AttemptPhase.AFTER_EVALUATION,
                    reason=AttemptFailureReason.NONFINITE_OR_INVALID_CANDIDATE,
                    before=before,
                    draft=draft,
                    candidate=candidate,
                    predecision=True,
                    expected_affected=expected_affected,
                    obligation=f"after evaluation failed: {exc}",
                )
            chosen_after = proposed_after
            allowance = epsilon_delta(
                before.complete_allowance,
                proposed_after.complete_allowance,
                before_elbo=before.terms.complete_elbo,
                after_elbo=proposed_after.terms.complete_elbo,
            )
            delta = proposed_after.terms.complete_elbo - before.terms.complete_elbo
            if delta > allowance.epsilon_delta:
                break

    if candidate is None or draft is None:
        return _failed(
            rebuilt,
            live,
            complement,
            producer=producer_label,
            phase=AttemptPhase.PROPOSAL,
            reason=AttemptFailureReason.NONFINITE_OR_INVALID_CANDIDATE,
            before=before,
            expected_affected=expected_affected,
            obligation="proposal did not produce a frozen candidate",
        )

    if _live_hashes(live) != captured_before_hashes:
        return _failed(
            rebuilt,
            live,
            complement,
            producer=rebuilt.requested_label,
            phase=AttemptPhase.COMMIT_OR_ROLLBACK,
            reason=AttemptFailureReason.ROLLBACK_HASH_MISMATCH,
            before=before,
            draft=draft,
            candidate=candidate,
            predecision=True,
            expected_affected=expected_affected,
            obligation="proposal mutated caller-owned live state before evaluation",
        )
    try:
        after = chosen_after or _evaluate(
            evaluator, candidate, complement, _cache(before)
        )
    except StaleFactorCacheError:
        return _failed(
            rebuilt,
            live,
            complement,
            producer=rebuilt.requested_label,
            phase=AttemptPhase.AFTER_EVALUATION,
            reason=AttemptFailureReason.STALE_CACHE,
            before=before,
            draft=draft,
            candidate=candidate,
            predecision=True,
            expected_affected=expected_affected,
            obligation="after evaluation cache was stale",
        )
    except (RuntimeError, ValueError) as exc:
        return _failed(
            rebuilt,
            live,
            complement,
            producer=producer_label,
            phase=AttemptPhase.AFTER_EVALUATION,
            reason=AttemptFailureReason.NONFINITE_OR_INVALID_CANDIDATE,
            before=before,
            draft=draft,
            candidate=candidate,
            predecision=True,
            expected_affected=expected_affected,
            obligation=f"after evaluation failed: {exc}",
        )

    if fault_injection is not None and fault_injection.kind in (
        H5FaultKind.OMIT_CHILD,
        H5FaultKind.OMIT_EMISSION,
    ):
        omitted = (
            fault_injection.target_factor_id
            or (
                "state_transition[2]"
                if fault_injection.kind is H5FaultKind.OMIT_CHILD
                else "emission[1]"
            )
        )
        records = tuple(item for item in after.factor_records if item.factor_id != omitted)
        partial = PartialFactorEvaluation(
            records, H5_FACTOR_UNIVERSE, (omitted,), ()
        )
        return _failed(
            rebuilt,
            live,
            complement,
            producer=rebuilt.requested_label,
            phase=AttemptPhase.AFTER_EVALUATION,
            reason=AttemptFailureReason.FACTOR_COVERAGE_MISMATCH,
            before=before,
            draft=draft,
            candidate=candidate,
            predecision=True,
            partial=partial,
            expected_affected=expected_affected,
            obligation=f"after evaluation omitted {omitted}",
        )

    before_by_id = {item.factor_id: item for item in before.factor_records}
    after_by_id = {item.factor_id: item for item in after.factor_records}
    observed_affected = tuple(
        factor_id
        for factor_id in H5_FACTOR_UNIVERSE
        if before_by_id[factor_id].input_hash.input_sha256
        != after_by_id[factor_id].input_hash.input_sha256
    )
    value_changed = tuple(
        factor_id
        for factor_id in H5_FACTOR_UNIVERSE
        if (
            before_by_id[factor_id].value_order_21.hex(),
            before_by_id[factor_id].value_order_17.hex(),
        )
        != (
            after_by_id[factor_id].value_order_21.hex(),
            after_by_id[factor_id].value_order_17.hex(),
        )
    )

    if fault_injection is not None and fault_injection.kind is H5FaultKind.CHANGE_VALUE_KEEP_INPUT:
        factor_id = fault_injection.target_factor_id or "state_transition[2]"
        original = after_by_id[factor_id]
        delta_value = fault_injection.scalar_delta if fault_injection.scalar_delta is not None else 1.0e-6
        corrupted = replace(
            original,
            input_hash=before_by_id[factor_id].input_hash,
            value_order_21=original.value_order_21 + delta_value,
            value_order_17=original.value_order_17 + delta_value,
        )
        records = tuple(corrupted if item.factor_id == factor_id else item for item in after.factor_records)
        partial = PartialFactorEvaluation(records, H5_FACTOR_UNIVERSE, (), ())
        deterministic = DeterministicReevaluationRecord(
            factor_id,
            corrupted.input_hash.input_sha256,
            corrupted.value_order_21,
            corrupted.value_order_17,
            before_by_id[factor_id].value_order_21,
            before_by_id[factor_id].value_order_17,
            False,
        )
        diagnostic_values = tuple(
            item
            for item in H5_FACTOR_UNIVERSE
            if item == factor_id or item in value_changed
        )
        diagnostic_affected = tuple(item for item in observed_affected if item != factor_id)
        return _failed(
            rebuilt,
            live,
            complement,
            producer=rebuilt.requested_label,
            phase=AttemptPhase.AFTER_EVALUATION,
            reason=AttemptFailureReason.DETERMINISTIC_REEVALUATION_MISMATCH,
            before=before,
            draft=draft,
            candidate=candidate,
            predecision=True,
            partial=partial,
            expected_affected=expected_affected,
            observed_affected=diagnostic_affected,
            value_changed=diagnostic_values,
            deterministic=deterministic,
            obligation=f"reported {factor_id} scalar failed deterministic reevaluation",
        )

    if observed_affected != expected_affected:
        partial = PartialFactorEvaluation(after.factor_records, H5_FACTOR_UNIVERSE, (), ())
        return _failed(
            rebuilt,
            live,
            complement,
            producer=rebuilt.requested_label,
            phase=AttemptPhase.DEPENDENCY_VALIDATION,
            reason=AttemptFailureReason.AFFECTED_FACTOR_MISMATCH,
            before=before,
            draft=draft,
            candidate=candidate,
            predecision=True,
            partial=partial,
            expected_affected=expected_affected,
            observed_affected=observed_affected,
            value_changed=value_changed,
            obligation="observed input-hash changes did not equal the dependency graph",
        )

    delta = float(after.terms.complete_elbo - before.terms.complete_elbo)
    allowance = epsilon_delta(
        before.complete_allowance,
        after.complete_allowance,
        before_elbo=before.terms.complete_elbo,
        after_elbo=after.terms.complete_elbo,
    )
    exact = rebuilt.requested_label is UpdateLabel.EXACT_COORDINATE
    accepted = delta >= -allowance.epsilon_delta if exact else delta > allowance.epsilon_delta
    if delta < -allowance.epsilon_delta:
        reason = DecisionReason.RESOLVED_DECREASE_REJECTED
    elif accepted and exact:
        reason = DecisionReason.EXACT_WITHIN_ALLOWANCE
    elif accepted:
        reason = DecisionReason.RESOLVED_POSITIVE
    else:
        reason = DecisionReason.UNRESOLVED_DELTA_REJECTED

    if fault_injection is not None and fault_injection.kind is H5FaultKind.FORCE_UNRESOLVED_GEM_ACCEPT:
        return _failed(
            rebuilt,
            live,
            complement,
            producer=rebuilt.requested_label,
            phase=AttemptPhase.DECISION,
            reason=AttemptFailureReason.DECISION_POLICY_VIOLATION,
            before=before,
            draft=draft,
            candidate=candidate,
            predecision=True,
            expected_affected=expected_affected,
            observed_affected=observed_affected,
            value_changed=value_changed,
            delta=delta,
            epsilon=allowance.epsilon_delta,
            attempted_accept=True,
            obligation="unresolved generalized-EM delta was forced to accept",
        )

    final_live = _candidate_as_live(live, candidate) if accepted else live
    if (
        fault_injection is not None
        and fault_injection.kind is H5FaultKind.MUTATE_REJECTED_LIVE_AND_RNG
        and not accepted
    ):
        coordinate = next(
            item for item in live.recognition.gaussians if item.coordinate_id == "q[z1]"
        )
        mutated_gaussians = tuple(
            GaussianRecognitionCoordinate(
                item.coordinate_id,
                FrozenTensorValue(
                    "float64",
                    (),
                    (
                        math.nextafter(item.mean.values[0], math.inf),
                    ),
                ),
                item.variance,
            )
            if item.coordinate_id == "q[z1]"
            else item
            for item in live.recognition.gaussians
        )
        mutated_recognition = RecognitionSnapshot(
            "h5-recognition-snapshot-v1",
            mutated_gaussians,
            live.recognition.categoricals,
        )
        mutated_rng = FrozenByteState(
            "h5-deterministic-rng-v1",
            b'{"algorithm":"none","counter":1}',
        )
        corrupted = H5LiveState(
            "h5-live-state-v1",
            mutated_recognition,
            live.model,
            live.optimizer_state,
            mutated_rng,
        )
        return _failed(
            rebuilt,
            live,
            complement,
            producer=rebuilt.requested_label,
            phase=AttemptPhase.COMMIT_OR_ROLLBACK,
            reason=AttemptFailureReason.ROLLBACK_HASH_MISMATCH,
            before=before,
            draft=draft,
            candidate=candidate,
            predecision=True,
            expected_affected=expected_affected,
            observed_affected=observed_affected,
            value_changed=value_changed,
            delta=delta,
            epsilon=allowance.epsilon_delta,
            attempted_accept=False,
            final_for_hash=corrupted,
            obligation="rejected update mutated live recognition and RNG state",
        )

    reevaluated = tuple(
        item.factor_id
        for item in after.factor_records
        if item.cache_disposition is CacheDisposition.REEVALUATED
    )
    reused = tuple(
        item.factor_id
        for item in after.factor_records
        if item.cache_disposition is CacheDisposition.REUSED
    )
    hashes = _hash_record(
        rebuilt,
        live,
        complement,
        final_live,
        draft=draft,
        candidate=candidate,
        predecision=True,
    )
    completed = CompletedUpdateAttempt(
        "h5-completed-attempt-v1",
        rebuilt,
        candidate.producer_label,
        candidate.variables,
        candidate.parameters,
        H5_FACTOR_UNIVERSE,
        expected_affected,
        reevaluated,
        reused,
        observed_affected,
        value_changed,
        (),
        (),
        before,
        after,
        delta,
        allowance,
        accepted,
        reason,
        line_search_step,
        candidate.damping,
        autograd_scope,
        hashes,
    )
    return H5TransactionResult("h5-transaction-result-v1", final_live, completed)


__all__ = [
    "H5_CANDIDATE_DRAFT_DOMAIN",
    "AttemptPhase",
    "AttemptFailureReason",
    "DecisionReason",
    "H5FaultKind",
    "H5FaultInjection",
    "H5CandidateDraft",
    "UpdateHashRecord",
    "PartialFactorEvaluation",
    "DeterministicReevaluationRecord",
    "CompletedUpdateAttempt",
    "FailedUpdateAttempt",
    "H5AttemptOutcome",
    "H5TransactionResult",
    "DifferentiableRecognitionState",
    "DifferentiableModelState",
    "canonical_h5_candidate_draft_bytes",
    "exact_conjugate_gaussian_e_update",
    "exact_source_row_update",
    "exact_gaussian_m_update",
    "differentiable_h5_complete_elbo_order_21",
    "propose_generalized_em",
    "propose_natural_gradient",
    "freeze_candidate",
    "canonical_frozen_complement_bytes",
    "execute_update",
]
