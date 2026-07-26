"""Immutable result records with fail-closed validation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, Literal

if TYPE_CHECKING:
    from .h7 import (
        H7ControlResult,
        H7GateOutcome,
        H7PredecessorReference,
        H7TrialResult,
    )
    from .h8 import (
        H8ChildAttemptRecord,
        H8ChildResult,
        H8ControlResult,
        H8CorrectnessCell,
    )


@dataclass(frozen=True)
class NumericalAllowance:
    convergence_estimate: float
    rounding_allowance: float

    def __post_init__(self) -> None:
        _require_nonnegative_finite(self.convergence_estimate, "convergence_estimate")
        _require_nonnegative_finite(self.rounding_allowance, "rounding_allowance")
        if not math.isfinite(self.total):
            raise ValueError("total allowance must be finite")

    @property
    def total(self) -> float:
        return self.convergence_estimate + self.rounding_allowance


@dataclass(frozen=True)
class ElboTermAllowances:
    expected_log_emission: tuple[NumericalAllowance, NumericalAllowance]
    initial_model_kl: NumericalAllowance
    initial_state_kl: NumericalAllowance
    model_source_kl: tuple[NumericalAllowance, NumericalAllowance]
    model_transition_kl: tuple[NumericalAllowance, NumericalAllowance]
    state_source_kl: tuple[NumericalAllowance, NumericalAllowance]
    state_transition_kl: tuple[NumericalAllowance, NumericalAllowance]
    joint_recognition_entropy: NumericalAllowance
    complete_elbo: NumericalAllowance

    def __post_init__(self) -> None:
        for name, value in (
            ("expected_log_emission", self.expected_log_emission),
            ("model_source_kl", self.model_source_kl),
            ("model_transition_kl", self.model_transition_kl),
            ("state_source_kl", self.state_source_kl),
            ("state_transition_kl", self.state_transition_kl),
        ):
            _require_allowance_pair(value, name)
        for name, value in (
            ("initial_model_kl", self.initial_model_kl),
            ("initial_state_kl", self.initial_state_kl),
            ("joint_recognition_entropy", self.joint_recognition_entropy),
            ("complete_elbo", self.complete_elbo),
        ):
            if not isinstance(value, NumericalAllowance):
                raise ValueError(f"{name} must be a NumericalAllowance")


@dataclass(frozen=True)
class ElboTerms:
    expected_log_emission: tuple[float, float]
    initial_model_kl: float
    initial_state_kl: float
    model_source_kl: tuple[float, float]
    model_transition_kl: tuple[float, float]
    state_source_kl: tuple[float, float]
    state_transition_kl: tuple[float, float]
    joint_recognition_entropy: float
    allowances: ElboTermAllowances
    complete_elbo: float

    def __post_init__(self) -> None:
        for name, value in (
            ("expected_log_emission", self.expected_log_emission),
            ("model_source_kl", self.model_source_kl),
            ("model_transition_kl", self.model_transition_kl),
            ("state_source_kl", self.state_source_kl),
            ("state_transition_kl", self.state_transition_kl),
        ):
            _require_finite_pair(value, name)
        for name, value in (
            ("initial_model_kl", self.initial_model_kl),
            ("initial_state_kl", self.initial_state_kl),
            ("joint_recognition_entropy", self.joint_recognition_entropy),
            ("complete_elbo", self.complete_elbo),
        ):
            _require_finite(value, name)
        if not isinstance(self.allowances, ElboTermAllowances):
            raise ValueError("allowances must be ElboTermAllowances")

        objective_terms = (
            *self.expected_log_emission,
            self.initial_model_kl,
            self.initial_state_kl,
            *self.model_source_kl,
            *self.model_transition_kl,
            *self.state_source_kl,
            *self.state_transition_kl,
        )
        expected = (
            sum(self.expected_log_emission)
            - self.initial_model_kl
            - self.initial_state_kl
            - sum(self.model_source_kl)
            - sum(self.model_transition_kl)
            - sum(self.state_source_kl)
            - sum(self.state_transition_kl)
        )
        if not math.isfinite(expected):
            raise ValueError("expected objective must be finite")
        absolute_term_sum = sum(abs(term) for term in objective_terms)
        if not math.isfinite(absolute_term_sum):
            raise ValueError("absolute-term sum must be finite")
        arithmetic_allowance = 256.0 * math.ulp(1.0) * max(1.0, absolute_term_sum)
        if not math.isfinite(arithmetic_allowance):
            raise ValueError("arithmetic allowance must be finite")
        residual = abs(self.complete_elbo - expected)
        if not math.isfinite(residual):
            raise ValueError("complete_elbo residual must be finite")
        if residual > arithmetic_allowance:
            raise ValueError("complete_elbo does not match its partitioned terms")


@dataclass(frozen=True)
class InvariantResult:
    name: str
    passed: bool
    value: float | None
    limit: float | None
    detail: str

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise ValueError("name must be a nonempty string")
        if type(self.passed) is not bool:
            raise ValueError("passed must be a bool")
        _require_optional_finite(self.value, "value")
        _require_optional_finite(self.limit, "limit")
        if type(self.detail) is not str:
            raise ValueError("detail must be a string")


class GateStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


_H1_PREFIX_PRIOR_V2_TERM_NAMES = (
    "expected_log_emission[0]",
    "expected_log_emission[1]",
    "initial_model_kl",
    "initial_state_kl",
    "model_source_kl[0]",
    "model_source_kl[1]",
    "model_transition_kl[0]",
    "model_transition_kl[1]",
    "state_source_kl[0]",
    "state_source_kl[1]",
    "state_transition_kl[0]",
    "state_transition_kl[1]",
    "joint_recognition_entropy",
    "complete_elbo",
)
H1_PREFIX_PRIOR_V2_INVARIANT_NAMES = (
    *(
        name
        for history_id in ("active", "swapped")
        for name in (
            *(
                item
                for bank in ("state", "model")
                for item in (
                    f"source_prior.{bank}.production_vs_oracle.{history_id}",
                    f"source_prior.{bank}.normalized.{history_id}",
                    f"source_prior.{bank}.support.{history_id}",
                )
            ),
            f"objective.{history_id}.monolithic_vs_local",
            f"objective.{history_id}.monolithic_vs_identity",
            f"objective.{history_id}.local_vs_independent_local",
            f"objective.{history_id}.independent_local_vs_identity",
            *(
                f"objective.{history_id}.term.{term_name}"
                for term_name in _H1_PREFIX_PRIOR_V2_TERM_NAMES
            ),
        )
    ),
    "source_prior.state.parent_assignment_swaps",
    "source_prior.model.parent_assignment_swaps",
    "joint.current_target_and_suffix_blind",
    "joint.parent_swap_changes_complete_objective",
    "schema.parent_specific_scorer_v2",
)


@dataclass(frozen=True)
class GateResult:
    gate: Literal["H1", "H2"]
    status: GateStatus
    fixture_id: Literal["h1-v1"]
    residual: float | None
    calibrated_allowance: float | None
    measurements: Mapping[str, float | None]
    invariants: tuple[InvariantResult, ...]
    obligations: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.gate not in ("H1", "H2"):
            raise ValueError("gate must be H1 or H2")
        if self.fixture_id != "h1-v1":
            raise ValueError("fixture_id must be h1-v1")
        if not isinstance(self.status, GateStatus):
            raise ValueError("status must be a GateStatus")
        _require_optional_finite(self.residual, "residual")
        _require_optional_finite(self.calibrated_allowance, "calibrated_allowance")
        if not isinstance(self.measurements, Mapping):
            raise ValueError("measurements must be a mapping")
        copied_measurements = dict(self.measurements)
        for name, value in copied_measurements.items():
            if type(name) is not str or not name:
                raise ValueError("measurement names must be nonempty strings")
            _require_optional_finite(value, f"measurements[{name!r}]")
        if type(self.invariants) is not tuple or not all(
            isinstance(item, InvariantResult) for item in self.invariants
        ):
            raise ValueError("invariants must be a tuple of InvariantResult")
        invariant_names = tuple(item.name for item in self.invariants)
        if len(set(invariant_names)) != len(invariant_names):
            raise ValueError("invariant names must be unique")
        if type(self.obligations) is not tuple or not all(
            type(item) is str and item for item in self.obligations
        ):
            raise ValueError("obligations must be a tuple of nonempty strings")

        if self.status is GateStatus.INCONCLUSIVE:
            if not self.obligations:
                raise ValueError("inconclusive results require an obligation")
        else:
            if not copied_measurements:
                raise ValueError("pass/fail measurements must be nonempty")
            if not self.invariants:
                raise ValueError("pass/fail invariants must be nonempty")
            if self.obligations:
                raise ValueError("pass/fail results cannot retain obligations")
            _require_finite(self.residual, "residual")
            _require_finite(self.calibrated_allowance, "calibrated_allowance")
            if any(value is None for value in copied_measurements.values()):
                raise ValueError("pass/fail results require finite measurements")
            if self.status is GateStatus.PASS and not all(
                item.passed for item in self.invariants
            ):
                raise ValueError("pass requires every invariant to pass")
            if self.status is GateStatus.FAIL and not any(
                not item.passed and item.value is not None and item.limit is not None
                for item in self.invariants
            ):
                raise ValueError("fail requires a finite failed invariant")
        object.__setattr__(self, "measurements", MappingProxyType(copied_measurements))


@dataclass(frozen=True)
class H1PrefixPriorGateResult:
    """Result of the separate H1 prefix-conditioned-prior prerequisite."""

    gate: Literal["H1-Prefix-Prior"]
    status: GateStatus
    fixture_id: Literal["h1-prefix-prior-v1"]
    residual: float | None
    calibrated_allowance: float | None
    measurements: Mapping[str, float | None]
    invariants: tuple[InvariantResult, ...]
    obligations: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.gate != "H1-Prefix-Prior":
            raise ValueError("gate must be H1-Prefix-Prior")
        if self.fixture_id != "h1-prefix-prior-v1":
            raise ValueError("fixture_id must be h1-prefix-prior-v1")
        if not isinstance(self.status, GateStatus):
            raise ValueError("status must be a GateStatus")
        _require_optional_finite(self.residual, "residual")
        _require_optional_finite(
            self.calibrated_allowance, "calibrated_allowance"
        )
        if not isinstance(self.measurements, Mapping):
            raise ValueError("measurements must be a mapping")
        copied_measurements = dict(self.measurements)
        if tuple(copied_measurements) != (
            "monolithic_elbo",
            "local_elbo",
            "evidence_minus_posterior_kl",
        ):
            raise ValueError("H1 prefix-prior measurement inventory is incomplete")
        for name, value in copied_measurements.items():
            _require_optional_finite(value, f"measurements[{name!r}]")
        if type(self.invariants) is not tuple or not all(
            isinstance(item, InvariantResult) for item in self.invariants
        ):
            raise ValueError("invariants must be a tuple of InvariantResult")
        invariant_names = tuple(item.name for item in self.invariants)
        if not invariant_names or len(set(invariant_names)) != len(invariant_names):
            raise ValueError("H1 prefix-prior invariant names must be nonempty and unique")
        _require_obligations(self.obligations)

        if self.status is GateStatus.INCONCLUSIVE:
            if not self.obligations:
                raise ValueError("inconclusive H1 prefix-prior requires an obligation")
        else:
            if self.obligations:
                raise ValueError("conclusive H1 prefix-prior cannot retain obligations")
            _require_finite(self.residual, "residual")
            _require_finite(self.calibrated_allowance, "calibrated_allowance")
            if any(value is None for value in copied_measurements.values()):
                raise ValueError(
                    "conclusive H1 prefix-prior requires finite measurements"
                )
            if self.status is GateStatus.PASS and not all(
                item.passed for item in self.invariants
            ):
                raise ValueError("H1 prefix-prior PASS requires every invariant")
            if self.status is GateStatus.FAIL and not any(
                not item.passed and item.value is not None and item.limit is not None
                for item in self.invariants
            ):
                raise ValueError(
                    "H1 prefix-prior FAIL requires a finite failed invariant"
                )
        object.__setattr__(
            self, "measurements", MappingProxyType(copied_measurements)
        )


@dataclass(frozen=True)
class H1PrefixPriorV2GateResult:
    """Result of the parent-specific scorer-v2 H1 prerequisite."""

    gate: Literal["H1-Prefix-Prior"]
    status: GateStatus
    fixture_id: Literal["h1-prefix-prior-scorer-v2"]
    scorer_schema: Literal["parent-specific-pooled-prefix-bilinear-v1"]
    fixture_sha256: str
    generative_factor_schema_sha256: str
    invariants: tuple[InvariantResult, ...]
    obligations: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            self.gate != "H1-Prefix-Prior"
            or self.fixture_id != "h1-prefix-prior-scorer-v2"
            or self.scorer_schema
            != "parent-specific-pooled-prefix-bilinear-v1"
        ):
            raise ValueError("H1 scorer-v2 identity is not exact")
        if not isinstance(self.status, GateStatus):
            raise ValueError("status must be a GateStatus")
        _require_sha256(self.fixture_sha256, "fixture_sha256")
        _require_sha256(
            self.generative_factor_schema_sha256,
            "generative_factor_schema_sha256",
        )
        if (
            type(self.invariants) is not tuple
            or not self.invariants
            or not all(
                isinstance(item, InvariantResult)
                for item in self.invariants
            )
            or len({item.name for item in self.invariants})
            != len(self.invariants)
        ):
            raise ValueError("H1 scorer-v2 invariants must be unique and nonempty")
        _require_obligations(self.obligations)
        invariant_names = tuple(item.name for item in self.invariants)
        if (
            self.status is not GateStatus.INCONCLUSIVE
            and invariant_names != H1_PREFIX_PRIOR_V2_INVARIANT_NAMES
        ):
            raise ValueError(
                "conclusive H1 scorer-v2 requires the exact invariant inventory"
            )
        if self.status is GateStatus.PASS:
            if self.obligations or not all(
                item.passed for item in self.invariants
            ):
                raise ValueError("H1 scorer-v2 PASS requires every invariant")
        elif self.status is GateStatus.FAIL:
            if self.obligations or not any(
                not item.passed for item in self.invariants
            ):
                raise ValueError("H1 scorer-v2 FAIL requires a failed invariant")
        elif not self.obligations:
            raise ValueError("H1 scorer-v2 INCONCLUSIVE requires an obligation")


@dataclass(frozen=True, init=False)
class H6PrefixGateResult:
    """Result of the independent H6 Prefix safety gate."""

    gate: Literal["H6-Prefix"]
    status: GateStatus
    validation_payload_sha256: str
    prefix_certificate_set_sha256: str
    obligations: tuple[str, ...]
    _certificates: tuple[object, ...] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.gate != "H6-Prefix":
            raise ValueError("gate must be H6-Prefix")
        if not isinstance(self.status, GateStatus):
            raise ValueError("status must be a GateStatus")
        _require_sha256(self.validation_payload_sha256, "validation_payload_sha256")
        _require_sha256(
            self.prefix_certificate_set_sha256,
            "prefix_certificate_set_sha256",
        )
        _require_obligations(self.obligations)
        if self.status is GateStatus.INCONCLUSIVE:
            if not self.obligations:
                raise ValueError("inconclusive H6-Prefix requires an obligation")
        elif self.obligations:
            raise ValueError("conclusive H6-Prefix cannot retain obligations")
        from .h6 import (
            EvidenceStatus,
            PrefixCertificate,
            _owned_hash,
            _prefix_certificate_set_sha256,
        )

        if not self._certificates or any(
            type(item) is not PrefixCertificate for item in self._certificates
        ):
            raise ValueError("H6-Prefix result requires typed certificates")
        certificates = tuple(self._certificates)
        for certificate in certificates:
            certificate.__post_init__()
        if any(item.status is EvidenceStatus.FAIL for item in certificates):
            expected_status = GateStatus.FAIL
            expected_obligations: tuple[str, ...] = ()
        elif any(item.status is EvidenceStatus.INCONCLUSIVE for item in certificates):
            expected_status = GateStatus.INCONCLUSIVE
            expected_obligations = tuple(
                sorted({value for item in certificates for value in item.obligations})
            )
        else:
            expected_status = GateStatus.PASS
            expected_obligations = ()
        expected_payload = _owned_hash(
            "vfe4.h6.prefix-validation-payload-set.v1",
            tuple(sorted(item.validation_payload_sha256 for item in certificates)),
        )
        if (
            self.status is not expected_status
            or self.obligations != expected_obligations
            or self.validation_payload_sha256 != expected_payload
            or self.prefix_certificate_set_sha256
            != _prefix_certificate_set_sha256(certificates)
        ):
            raise ValueError("H6-Prefix result does not match its certificate set")

    @classmethod
    def from_certificates(cls, certificates: Mapping[object, object]) -> "H6PrefixGateResult":
        from .h6 import EvidenceStatus, PrefixCertificate, _owned_hash, _prefix_certificate_set_sha256

        frozen = tuple(
            item
            for _, item in sorted(
                certificates.items(), key=lambda pair: repr(pair[0])
            )
        )
        if not frozen or any(type(item) is not PrefixCertificate for item in frozen):
            raise ValueError("typed Prefix certificates are required")
        if any(item.status is EvidenceStatus.FAIL for item in frozen):
            status = GateStatus.FAIL
            obligations: tuple[str, ...] = ()
        elif any(item.status is EvidenceStatus.INCONCLUSIVE for item in frozen):
            status = GateStatus.INCONCLUSIVE
            obligations = tuple(sorted({value for item in frozen for value in item.obligations}))
        else:
            status = GateStatus.PASS
            obligations = ()
        instance = object.__new__(cls)
        values = {
            "gate": "H6-Prefix",
            "status": status,
            "validation_payload_sha256": _owned_hash(
                "vfe4.h6.prefix-validation-payload-set.v1",
                tuple(sorted(item.validation_payload_sha256 for item in frozen)),
            ),
            "prefix_certificate_set_sha256": _prefix_certificate_set_sha256(frozen),
            "obligations": obligations,
            "_certificates": frozen,
        }
        for name, value in values.items():
            object.__setattr__(instance, name, value)
        instance.__post_init__()
        return instance


@dataclass(frozen=True, slots=True, init=False)
class H6BoundedPrefixGateResult:
    """Exact result derived from one ordered bounded Prefix certificate set."""

    gate: Literal["H6-Prefix"]
    status: GateStatus
    config_sha256: str
    workload_plan_sha256: str
    validation_payload_sha256: str
    prefix_certificate_set_sha256: str
    obligations: tuple[str, ...]
    _certificate_set: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self) is not H6BoundedPrefixGateResult:
            raise TypeError(
                "record requires the exact H6BoundedPrefixGateResult type"
            )
        if self.gate != "H6-Prefix":
            raise ValueError("bounded Prefix gate must be H6-Prefix")
        if type(self.status) is not GateStatus:
            raise ValueError("bounded Prefix status must be exact")
        for name in (
            "config_sha256",
            "workload_plan_sha256",
            "validation_payload_sha256",
            "prefix_certificate_set_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        _require_obligations(self.obligations)
        from .h6 import (
            BoundedPrefixCertificateSet,
            EvidenceStatus,
        )

        if type(self._certificate_set) is not BoundedPrefixCertificateSet:
            raise ValueError(
                "bounded Prefix result requires its exact certificate set"
            )
        certificate_set = self._certificate_set
        certificate_set.__post_init__()
        certificates = certificate_set.certificates
        if any(
            certificate.status is EvidenceStatus.FAIL
            for certificate in certificates
        ):
            expected_status = GateStatus.FAIL
            expected_obligations: tuple[str, ...] = ()
        elif any(
            certificate.status is EvidenceStatus.INCONCLUSIVE
            for certificate in certificates
        ):
            expected_status = GateStatus.INCONCLUSIVE
            expected_obligations = tuple(
                sorted(
                    {
                        obligation
                        for certificate in certificates
                        for obligation in certificate.obligations
                    }
                )
            )
        else:
            expected_status = GateStatus.PASS
            expected_obligations = ()
        if (
            self.status is not expected_status
            or self.obligations != expected_obligations
            or self.config_sha256 != certificate_set.config_sha256
            or self.workload_plan_sha256
            != certificate_set.workload_plan_sha256
            or self.validation_payload_sha256
            != certificate_set.validation_payload_sha256
            or self.prefix_certificate_set_sha256
            != certificate_set.prefix_certificate_set_sha256
        ):
            raise ValueError(
                "bounded Prefix result does not match its certificate set"
            )

    @classmethod
    def from_certificate_set(
        cls,
        certificate_set: object,
    ) -> "H6BoundedPrefixGateResult":
        from .h6 import (
            BoundedPrefixCertificateSet,
            EvidenceStatus,
        )

        if cls is not H6BoundedPrefixGateResult:
            raise TypeError(
                "factory requires the exact H6BoundedPrefixGateResult type"
            )
        if type(certificate_set) is not BoundedPrefixCertificateSet:
            raise ValueError(
                "factory requires an exact bounded Prefix certificate set"
            )
        certificate_set.__post_init__()
        certificates = certificate_set.certificates
        if any(
            certificate.status is EvidenceStatus.FAIL
            for certificate in certificates
        ):
            status = GateStatus.FAIL
            obligations: tuple[str, ...] = ()
        elif any(
            certificate.status is EvidenceStatus.INCONCLUSIVE
            for certificate in certificates
        ):
            status = GateStatus.INCONCLUSIVE
            obligations = tuple(
                sorted(
                    {
                        obligation
                        for certificate in certificates
                        for obligation in certificate.obligations
                    }
                )
            )
        else:
            status = GateStatus.PASS
            obligations = ()
        instance = object.__new__(cls)
        values = {
            "gate": "H6-Prefix",
            "status": status,
            "config_sha256": certificate_set.config_sha256,
            "workload_plan_sha256": (
                certificate_set.workload_plan_sha256
            ),
            "validation_payload_sha256": (
                certificate_set.validation_payload_sha256
            ),
            "prefix_certificate_set_sha256": (
                certificate_set.prefix_certificate_set_sha256
            ),
            "obligations": obligations,
            "_certificate_set": certificate_set,
        }
        for name, value in values.items():
            object.__setattr__(instance, name, value)
        instance.__post_init__()
        return instance


@dataclass(frozen=True, init=False)
class H6PredictionResult:
    """Result of the separately authorized H6 Prediction evidence stage."""

    gate: Literal["H6-Prediction"]
    status: GateStatus
    readiness_sha256: str
    smc_bias_semantics_sha256: str | None
    metrics_sha256: str | None
    obligations: tuple[str, ...]
    _readiness: object = field(repr=False, compare=False)
    _decision: object = field(repr=False, compare=False)
    _metrics_bytes: bytes | None = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.gate != "H6-Prediction":
            raise ValueError("gate must be H6-Prediction")
        if not isinstance(self.status, GateStatus):
            raise ValueError("status must be a GateStatus")
        _require_sha256(self.readiness_sha256, "readiness_sha256")
        if self.smc_bias_semantics_sha256 is not None:
            _require_sha256(
                self.smc_bias_semantics_sha256,
                "smc_bias_semantics_sha256",
            )
        if self.metrics_sha256 is not None:
            _require_sha256(self.metrics_sha256, "metrics_sha256")
        _require_obligations(self.obligations)
        if self.status is GateStatus.INCONCLUSIVE:
            if not self.obligations:
                raise ValueError("inconclusive H6-Prediction requires an obligation")
        else:
            if self.obligations:
                raise ValueError("conclusive H6-Prediction cannot retain obligations")
            if self.metrics_sha256 is None:
                raise ValueError("conclusive H6-Prediction requires metrics")
        from .h6 import (
            EvidenceStatus,
            H6PredictionReadinessToken,
            OrderedPredictionDecision,
            PredictionDecision,
        )

        if type(self._readiness) is not H6PredictionReadinessToken:
            raise ValueError("typed H6 Prediction readiness is required")
        if type(self._decision) not in (
            PredictionDecision,
            OrderedPredictionDecision,
        ):
            raise ValueError("typed H6 prediction decision is required")
        if type(self._decision) is OrderedPredictionDecision:
            if (
                self._readiness.readiness_schema
                != "h6-prediction-readiness-v2"
                or self.smc_bias_semantics_sha256
                != self._readiness.smc_bias_semantics_sha256
                or self._readiness.objective_gate_spec_sha256
                != self._decision.objective_gate_spec_sha256
            ):
                raise ValueError(
                    "amended metrics require readiness bound to the same OBJECTIVE spec"
                )
        elif (
            self._readiness.readiness_schema
            != "h6-prediction-readiness-v1"
            or self.smc_bias_semantics_sha256 is not None
        ):
            raise ValueError("legacy metrics cannot close amended readiness")
        self._readiness.__post_init__()
        self._decision.__post_init__()
        if self.readiness_sha256 != self._readiness.readiness_sha256:
            raise ValueError("readiness_sha256 does not match readiness token")
        expected_status = GateStatus[self._decision.status.name]
        if self.status is not expected_status or self.obligations != self._decision.obligations:
            raise ValueError("prediction result does not match its decision")
        if self._metrics_bytes is None:
            if self.metrics_sha256 is not None:
                raise ValueError("metrics_sha256 cannot exist without metrics bytes")
        elif (
            type(self._metrics_bytes) is not bytes
            or self.metrics_sha256 != hashlib.sha256(self._metrics_bytes).hexdigest()
        ):
            raise ValueError("metrics_sha256 does not match metrics bytes")
        if self._decision.status in (EvidenceStatus.PASS, EvidenceStatus.FAIL) and self._metrics_bytes is None:
            raise ValueError("conclusive prediction decisions require metrics bytes")
        if self._metrics_bytes is None:
            raise ValueError("H6 Prediction result requires a metrics artifact")
        derived_decision = _prediction_decision_from_metrics(self._metrics_bytes)
        if derived_decision != self._decision:
            raise ValueError("PredictionDecision does not match metrics bytes")
        metrics_payload = json.loads(self._metrics_bytes)
        metrics_semantics_sha256 = (
            metrics_payload["smc_bias_semantics_sha256"]
            if metrics_payload["schema"] == "h6-prediction-metrics-v2"
            else None
        )
        if metrics_semantics_sha256 != self.smc_bias_semantics_sha256:
            raise ValueError(
                "H6 Prediction result does not bind the metrics SMC bias semantics"
            )

    @classmethod
    def from_metrics(
        cls,
        *,
        readiness: object,
        metrics_bytes: bytes,
    ) -> "H6PredictionResult":
        from .h6 import H6PredictionReadinessToken

        if type(readiness) is not H6PredictionReadinessToken:
            raise ValueError("typed H6 Prediction readiness is required")
        if type(metrics_bytes) is not bytes:
            raise ValueError("metrics_bytes must be immutable bytes")
        decision = _prediction_decision_from_metrics(metrics_bytes)
        metrics_payload = json.loads(metrics_bytes)
        smc_bias_semantics_sha256 = (
            metrics_payload["smc_bias_semantics_sha256"]
            if metrics_payload["schema"] == "h6-prediction-metrics-v2"
            else None
        )
        instance = object.__new__(cls)
        values = {
            "gate": "H6-Prediction",
            "status": GateStatus[decision.status.name],
            "readiness_sha256": readiness.readiness_sha256,
            "smc_bias_semantics_sha256": smc_bias_semantics_sha256,
            "metrics_sha256": hashlib.sha256(metrics_bytes).hexdigest(),
            "obligations": decision.obligations,
            "_readiness": readiness,
            "_decision": decision,
            "_metrics_bytes": bytes(metrics_bytes),
        }
        for name, value in values.items():
            object.__setattr__(instance, name, value)
        instance.__post_init__()
        return instance


_H7_FIXTURE_HASH_KEYS = (
    "h1_fixture_raw_sha256",
    "h7_fixture_raw_sha256",
    "density_probe_table_raw_sha256",
    "density_probe_set_sha256",
    "scalar_probe_table_raw_sha256",
    "scalar_probe_set_sha256",
    "precision_operand_table_raw_sha256",
    "precision_operand_set_sha256",
    "oracle_inventory_sha256",
)
_H7_PREDECESSOR_KEYS = ("h1_h5", "h1_prefix_prior", "h6_prefix")
_H7ExpectedNegativeState = Literal[
    "success",
    "false_acceptance",
    "inconclusive",
]


def _h7_expected_negative_state(
    trial: "H7TrialResult",
) -> _H7ExpectedNegativeState:
    """Classify the outside-stabilizer result from its complete owned evidence."""

    from .h7 import H7ResidualRecord, H7TrialResult

    if type(trial) is not H7TrialResult:
        raise ValueError("expected-negative evidence must be an exact H7TrialResult")
    trial.__post_init__()
    if trial.spec.role != "expected_negative":
        raise ValueError("expected-negative evidence has the wrong trial role")
    if not trial.envelope.passed:
        return "inconclusive"
    if trial.predicate_satisfied:
        return "success"

    objective_residuals = tuple(
        residual
        for objective in trial.objective_by_recognition_family.values()
        for residual in (
            objective.initial_joint_kl.residual,
            *(item.residual for item in objective.local_terms),
            *(
                observation.residual
                for evaluation in objective.density_probe_evaluations
                for observation in evaluation.observations
            ),
            *objective.scorer_residuals,
            objective.complete_local,
            objective.complete_monolithic,
            objective.p_density_shift,
            objective.q_density_shift,
            objective.log_ratio,
            objective.entropy_shift,
            *(
                item
                for item in (objective.evidence, objective.posterior_kl)
                if type(item) is H7ResidualRecord
            ),
        )
    )
    covariance_accepted = (
        trial.r_abs.passed
        and trial.r_rel.passed
        and trial.r_back_max.passed
        and all(item.passed for item in trial.backward_by_operand)
        and all(item.passed for item in trial.residuals)
        and all(item.passed for item in objective_residuals)
    )
    return "false_acceptance" if covariance_accepted else "inconclusive"


@dataclass(frozen=True)
class H7GateResult:
    """Sole fail-closed result record for the H7 covariance gate."""

    fixture_hash_keys: ClassVar[tuple[str, ...]] = _H7_FIXTURE_HASH_KEYS
    predecessor_keys: ClassVar[tuple[str, ...]] = _H7_PREDECESSOR_KEYS

    gate: Literal["H7"]
    status: GateStatus
    fixture_hashes: Mapping[str, str]
    predecessor_references: Mapping[str, "H7PredecessorReference"]
    trials: tuple["H7TrialResult", ...]
    controls: tuple["H7ControlResult", ...]
    outcome: "H7GateOutcome"
    obligations: tuple[str, ...]
    result_sha256: str

    @classmethod
    def create(
        cls,
        *,
        gate: Literal["H7"],
        status: GateStatus,
        fixture_hashes: Mapping[str, str],
        predecessor_references: Mapping[str, "H7PredecessorReference"],
        trials: tuple["H7TrialResult", ...],
        controls: tuple["H7ControlResult", ...],
        outcome: "H7GateOutcome",
        obligations: tuple[str, ...],
    ) -> "H7GateResult":
        """Defensively own mappings and bind all semantic fields."""

        from .h7 import h7_owned_sha256

        owned_fixtures = _freeze_h7_digest_mapping(
            fixture_hashes,
            allowed_keys=_H7_FIXTURE_HASH_KEYS,
            name="fixture_hashes",
        )
        owned_predecessors = _freeze_h7_predecessors(predecessor_references)
        semantic = {
            "gate": gate,
            "status": status,
            "fixture_hashes": owned_fixtures,
            "predecessor_references": owned_predecessors,
            "trials": tuple(trials),
            "controls": tuple(controls),
            "outcome": outcome,
            "obligations": tuple(obligations),
        }
        return cls(
            **semantic,
            result_sha256=h7_owned_sha256(
                "vfe4.h7.gate-result.v1",
                semantic,
            ),
        )

    def __post_init__(self) -> None:
        from .h7 import (
            H7_CONTROL_IDS,
            H7_REQUIRED_TRIAL_IDS,
            H7_SCALAR_TRIAL_IDS,
            H7ControlResult,
            H7FailOutcome,
            H7InconclusiveOutcome,
            H7PassOutcome,
            H7PredecessorReference,
            H7TrialResult,
            h7_owned_sha256,
        )

        if self.gate != "H7":
            raise ValueError("H7 gate result must declare gate H7")
        if not isinstance(self.status, GateStatus):
            raise ValueError("H7 status must be a GateStatus")
        _require_obligations(self.obligations)
        if len(set(self.obligations)) != len(self.obligations):
            raise ValueError("H7 obligations must be unique")

        fixtures = _freeze_h7_digest_mapping(
            self.fixture_hashes,
            allowed_keys=_H7_FIXTURE_HASH_KEYS,
            name="fixture_hashes",
        )
        predecessors = _freeze_h7_predecessors(self.predecessor_references)
        object.__setattr__(self, "fixture_hashes", fixtures)
        object.__setattr__(self, "predecessor_references", predecessors)

        if type(self.trials) is not tuple or any(
            type(item) is not H7TrialResult for item in self.trials
        ):
            raise ValueError("H7 trials must be an exact tuple of H7TrialResult")
        if type(self.controls) is not tuple or any(
            type(item) is not H7ControlResult for item in self.controls
        ):
            raise ValueError("H7 controls must be an exact tuple of H7ControlResult")
        for predecessor in predecessors.values():
            if type(predecessor) is not H7PredecessorReference:
                raise ValueError(
                    "H7 predecessor mappings require exact predecessor records"
                )
            predecessor.__post_init__()
        for trial in self.trials:
            trial.__post_init__()
        for control in self.controls:
            control.__post_init__()
        if type(self.outcome) not in (
            H7PassOutcome,
            H7FailOutcome,
            H7InconclusiveOutcome,
        ):
            raise ValueError("H7 outcome must use the closed outcome union")
        self.outcome.__post_init__()

        trial_ids = tuple(item.spec.trial_id for item in self.trials)
        control_ids = tuple(item.control_id for item in self.controls)
        if (
            len(set(trial_ids)) != len(trial_ids)
            or any(item not in H7_REQUIRED_TRIAL_IDS for item in trial_ids)
        ):
            raise ValueError(
                "H7 trial IDs must be unique members of the closed inventory"
            )
        if (
            len(set(control_ids)) != len(control_ids)
            or any(item not in H7_CONTROL_IDS for item in control_ids)
        ):
            raise ValueError(
                "H7 control IDs must be unique members of the closed inventory"
            )

        complete_inventory = (
            tuple(fixtures) == _H7_FIXTURE_HASH_KEYS
            and tuple(predecessors) == _H7_PREDECESSOR_KEYS
            and trial_ids == H7_REQUIRED_TRIAL_IDS
            and control_ids == H7_CONTROL_IDS
        )
        scalar_or_positive = tuple(
            item
            for item in self.trials
            if item.spec.role in ("scalar_regression", "positive_covariance")
        )
        expected_negative = tuple(
            item for item in self.trials if item.spec.role == "expected_negative"
        )
        expected_negative_state: _H7ExpectedNegativeState = (
            _h7_expected_negative_state(expected_negative[0])
            if len(expected_negative) == 1
            else "inconclusive"
        )
        incomplete_runtime_evidence = (
            not complete_inventory
            or any(not item.envelope.passed for item in self.trials)
            or len(expected_negative) != 1
            or expected_negative_state == "inconclusive"
            or any(not item.detected for item in self.controls)
        )
        finite_refutation_ids = tuple(
            f"{item.spec.trial_id}:{item.spec.expected_predicate}"
            for item in scalar_or_positive
            if item.envelope.passed and not item.predicate_satisfied
        )
        finite_refutation = bool(finite_refutation_ids)

        if self.status is GateStatus.INCONCLUSIVE:
            if (
                type(self.outcome) is not H7InconclusiveOutcome
                or not self.obligations
                or self.outcome.obligations != self.obligations
            ):
                raise ValueError(
                    "INCONCLUSIVE H7 requires one matching nonempty obligation set"
                )
        elif self.status is GateStatus.FAIL:
            if (
                type(self.outcome) is not H7FailOutcome
                or self.obligations
                or incomplete_runtime_evidence
                or self.outcome.failed_invariant_ids != finite_refutation_ids
                or self.outcome.expected_negative_false_acceptance
                != (expected_negative_state == "false_acceptance")
                or not (
                    finite_refutation
                    or expected_negative_state == "false_acceptance"
                )
            ):
                raise ValueError(
                    "FAIL H7 requires complete evidence and a finite refutation"
                )
        elif (
            type(self.outcome) is not H7PassOutcome
            or self.obligations
            or incomplete_runtime_evidence
            or finite_refutation
            or tuple(item.spec.trial_id for item in scalar_or_positive)
            != (*H7_SCALAR_TRIAL_IDS, *H7_REQUIRED_TRIAL_IDS[2:-1])
            or not all(item.predicate_satisfied for item in self.trials)
            or expected_negative_state != "success"
        ):
            raise ValueError("PASS H7 requires the exact completely closed inventory")

        semantic = {
            "gate": self.gate,
            "status": self.status,
            "fixture_hashes": self.fixture_hashes,
            "predecessor_references": self.predecessor_references,
            "trials": self.trials,
            "controls": self.controls,
            "outcome": self.outcome,
            "obligations": self.obligations,
        }
        _require_sha256(self.result_sha256, "result_sha256")
        if self.result_sha256 != h7_owned_sha256(
            "vfe4.h7.gate-result.v1",
            semantic,
        ):
            raise ValueError("result_sha256 does not match the complete H7 result")


@dataclass(frozen=True)
class H8GateResult:
    """Fail-closed result for the bounded H8 sparse-scale systems gate."""

    gate: Literal["H8"]
    status: GateStatus
    config_sha256: str
    candidate_junit_sha256: str | None
    current_refs_registry_sha256: str | None
    h7_manifest_sha256: str | None
    h6_prediction_manifest_sha256: str | None
    correctness: tuple["H8CorrectnessCell", ...]
    child_attempts: tuple["H8ChildAttemptRecord", ...]
    production_runs: tuple["H8ChildResult", ...]
    profiler_runs: tuple["H8ChildResult", ...]
    controls: tuple["H8ControlResult", ...]
    obligations: tuple[str, ...]

    def __post_init__(self) -> None:
        from .h8 import (
            H8_CORRECTNESS_CASES,
            H8_NEGATIVE_CONTROL_IDS,
            H8_PRODUCTION_SEEDS,
            H8ChildAttemptRecord,
            H8ChildResult,
            H8ControlResult,
            H8CorrectnessCell,
        )

        if self.gate != "H8":
            raise ValueError("H8 gate result must declare gate H8")
        if not isinstance(self.status, GateStatus):
            raise ValueError("H8 status must be a GateStatus")
        if self.status is GateStatus.PASS:
            raise ValueError(
                "PASS H8 remains unavailable until parent orchestration "
                "and runtime revalidation are implemented"
            )
        _require_sha256(self.config_sha256, "config_sha256")
        prerequisite_hash_names = (
            "candidate_junit_sha256",
            "current_refs_registry_sha256",
            "h7_manifest_sha256",
            "h6_prediction_manifest_sha256",
        )
        for name in prerequisite_hash_names:
            value = getattr(self, name)
            if value is not None:
                _require_sha256(value, name)
        missing_prerequisite_hashes = tuple(
            name for name in prerequisite_hash_names if getattr(self, name) is None
        )
        _require_obligations(self.obligations)
        if len(set(self.obligations)) != len(self.obligations):
            raise ValueError("H8 obligations must be unique")
        if self.status is GateStatus.INCONCLUSIVE:
            if not self.obligations:
                raise ValueError("INCONCLUSIVE H8 requires an open obligation")
        else:
            if self.obligations:
                raise ValueError("conclusive H8 cannot retain obligations")
            if missing_prerequisite_hashes:
                raise ValueError(
                    "conclusive H8 requires every canonical prerequisite hash"
                )

        typed_inventories = (
            ("correctness", self.correctness, H8CorrectnessCell),
            ("child_attempts", self.child_attempts, H8ChildAttemptRecord),
            ("production_runs", self.production_runs, H8ChildResult),
            ("profiler_runs", self.profiler_runs, H8ChildResult),
            ("controls", self.controls, H8ControlResult),
        )
        for name, values, expected_type in typed_inventories:
            if type(values) is not tuple or any(
                type(item) is not expected_type for item in values
            ):
                raise ValueError(f"{name} must retain exact typed records")
            for item in values:
                item.__post_init__()

        correctness_ids = tuple(item.cell_id for item in self.correctness)
        expected_correctness_ids = tuple(
            range(1, len(H8_CORRECTNESS_CASES) + 1)
        )
        _require_h8_inventory_prefix(
            correctness_ids,
            expected_correctness_ids,
            "correctness cells",
        )
        attempt_ids = tuple(
            (
                item.request.mode,
                item.request.seed,
                item.request.repetition,
                item.request.control_id,
            )
            for item in self.child_attempts
        )
        expected_attempt_ids = (
            *(
                ("production", seed, repetition, None)
                for seed in H8_PRODUCTION_SEEDS
                for repetition in range(5)
            ),
            *(
                ("profiler", seed, None, None)
                for seed in H8_PRODUCTION_SEEDS
            ),
            *(
                (
                    "negative_control",
                    H8_PRODUCTION_SEEDS[0],
                    None,
                    control_id,
                )
                for control_id in H8_NEGATIVE_CONTROL_IDS
            ),
        )
        _require_h8_inventory_prefix(
            attempt_ids,
            expected_attempt_ids,
            "child attempts",
        )
        if any(
            attempt.request.config_sha256 != self.config_sha256
            for attempt in self.child_attempts
        ):
            raise ValueError("child attempts must bind the H8 config")

        attempt_production_runs = tuple(
            item.result
            for item in self.child_attempts
            if item.request.mode == "production"
            and type(item.result) is H8ChildResult
        )
        attempt_profiler_runs = tuple(
            item.result
            for item in self.child_attempts
            if item.request.mode == "profiler"
            and type(item.result) is H8ChildResult
        )
        attempt_controls = tuple(
            item.result
            for item in self.child_attempts
            if item.request.mode == "negative_control"
            and type(item.result) is H8ControlResult
        )
        if (
            self.production_runs != attempt_production_runs
            or self.profiler_runs != attempt_profiler_runs
            or self.controls != attempt_controls
        ):
            raise ValueError(
                "run/control inventories must equal result-bearing attempts"
            )

        production_ids = tuple(
            (item.seed, item.repetition) for item in self.production_runs
        )
        expected_production_ids = tuple(
            (seed, repetition)
            for seed in H8_PRODUCTION_SEEDS
            for repetition in range(5)
        )
        if any(item.mode != "production" for item in self.production_runs):
            raise ValueError("production_runs may contain only production results")
        profiler_ids = tuple(item.seed for item in self.profiler_runs)
        if any(
            item.mode != "profiler" or item.repetition is not None
            for item in self.profiler_runs
        ):
            raise ValueError("profiler_runs may contain only profiler results")
        control_ids = tuple(item.control_id for item in self.controls)

        retained_statuses = (
            *(item.status for item in self.correctness),
            *(item.status for item in self.child_attempts),
            *(item.status for item in self.controls),
            *(
                invariant.status
                for child in (*self.production_runs, *self.profiler_runs)
                for invariant in child.invariants
            ),
        )
        witnessed_fail = GateStatus.FAIL in retained_statuses
        complete_pass = (
            correctness_ids == expected_correctness_ids
            and attempt_ids == expected_attempt_ids
            and all(
                item.status is GateStatus.PASS
                for item in self.child_attempts
            )
            and production_ids == expected_production_ids
            and profiler_ids == H8_PRODUCTION_SEEDS
            and control_ids == H8_NEGATIVE_CONTROL_IDS
            and retained_statuses
            and all(status is GateStatus.PASS for status in retained_statuses)
        )
        if self.status is GateStatus.PASS and not complete_pass:
            raise ValueError("PASS H8 requires the exact completely passing inventory")
        if self.status is GateStatus.FAIL and not witnessed_fail:
            raise ValueError("FAIL H8 requires retained witnessed-failure evidence")
        if self.status is GateStatus.INCONCLUSIVE and witnessed_fail:
            raise ValueError("a witnessed H8 failure cannot be masked as INCONCLUSIVE")


def _require_h8_inventory_prefix(
    observed: tuple[object, ...],
    expected: tuple[object, ...],
    name: str,
) -> None:
    if len(set(observed)) != len(observed):
        raise ValueError(f"{name} cannot contain duplicate identities")
    if observed != expected[: len(observed)]:
        raise ValueError(f"{name} must retain frozen order without gaps")


def _freeze_h7_digest_mapping(
    value: Mapping[str, str],
    *,
    allowed_keys: tuple[str, ...],
    name: str,
) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or any(
        type(key) is not str for key in value
    ):
        raise ValueError(f"{name} must be a string-keyed mapping")
    unexpected = set(value).difference(allowed_keys)
    if unexpected:
        raise ValueError(f"{name} contains keys outside the closed H7 inventory")
    owned: dict[str, str] = {}
    for key in allowed_keys:
        if key in value:
            digest = value[key]
            _require_sha256(digest, f"{name}[{key!r}]")
            owned[key] = digest
    return MappingProxyType(owned)


def _freeze_h7_predecessors(
    value: Mapping[str, "H7PredecessorReference"],
) -> Mapping[str, "H7PredecessorReference"]:
    if not isinstance(value, Mapping) or any(
        type(key) is not str for key in value
    ):
        raise ValueError("predecessor_references must be a string-keyed mapping")
    unexpected = set(value).difference(_H7_PREDECESSOR_KEYS)
    if unexpected:
        raise ValueError(
            "predecessor_references contains keys outside the H7 registry"
        )
    return MappingProxyType(
        {key: value[key] for key in _H7_PREDECESSOR_KEYS if key in value}
    )


def _prediction_decision_from_metrics(metrics_bytes: bytes) -> object:
    from .h6 import (
        ObjectiveGateDecision,
        ObjectiveGateSpec,
        OrderedPredictionDecision,
        PredictionDecision,
    )

    try:
        payload = json.loads(metrics_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("metrics bytes must be canonical JSON") from exc
    if type(payload) is not dict:
        raise ValueError("metrics bytes must encode an object")
    try:
        canonical = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("metrics bytes contain unsupported values") from exc
    if canonical != metrics_bytes:
        raise ValueError("metrics bytes do not match the canonical H6 schema")
    schema = payload.get("schema")
    if schema == "h6-prediction-metrics-v2":
        expected_keys = {
            "schema",
            "objective_gate_spec_sha256",
            "smc_bias_semantics_sha256",
            "opening_policy",
            "evaluation_order",
            "opening_count",
            "test_opening_sha256",
            "raw_endpoint_inventory_sha256",
            "objective_estimator_complete",
            "objective_interval_eligible",
            "objective_interval",
            "objective_status",
            "primary_estimator_complete",
            "primary_interval_eligible",
            "primary_interval",
            "primary_disposition",
        }
        if set(payload) != expected_keys:
            raise ValueError("v2 metrics field inventory is not exact")
        spec = ObjectiveGateSpec.create()
        from vfe4.evaluation.smc_uncertainty import SMC_BIAS_SEMANTICS

        if (
            payload["objective_gate_spec_sha256"] != spec.spec_sha256
            or payload["smc_bias_semantics_sha256"]
            != SMC_BIAS_SEMANTICS.semantics_sha256
            or payload["opening_policy"] != spec.opening_policy
            or payload["evaluation_order"] != spec.evaluation_order
            or payload["opening_count"] != 1
        ):
            raise ValueError(
                "v2 metrics do not bind the frozen SMC semantics and "
                "one-opening OBJECTIVE spec"
            )
        raw_inventory = payload["raw_endpoint_inventory_sha256"]
        test_opening_sha256 = payload["test_opening_sha256"]
        _require_sha256(test_opening_sha256, "test_opening_sha256")
        _require_sha256(raw_inventory, "raw_endpoint_inventory_sha256")
        objective_complete = payload["objective_estimator_complete"]
        objective_eligible = payload["objective_interval_eligible"]
        primary_complete = payload["primary_estimator_complete"]
        primary_eligible = payload["primary_interval_eligible"]
        if (
            type(objective_complete) is not bool
            or type(objective_eligible) is not bool
            or type(primary_complete) is not bool
            or type(primary_eligible) is not bool
        ):
            raise ValueError("v2 metrics estimator flags must be boolean")
        objective_interval = _prediction_metrics_interval(
            payload["objective_interval"],
            name="objective_interval",
            required=True,
        )
        primary_interval = _prediction_metrics_interval(
            payload["primary_interval"],
            name="primary_interval",
            required=primary_complete,
        )
        objective = ObjectiveGateDecision.classify(
            objective_interval=objective_interval,
            estimator_complete=objective_complete,
            interval_eligible=objective_eligible,
        )
        decision = OrderedPredictionDecision.classify(
            objective_gate_spec=spec,
            objective=objective,
            primary_interval=primary_interval,
            primary_estimator_complete=primary_complete,
            primary_interval_eligible=primary_eligible,
            opening_count=1,
            test_opening_sha256=test_opening_sha256,
            raw_endpoint_inventory_sha256=raw_inventory,
        )
        if (
            payload["objective_status"] != objective.status.value
            or payload["primary_disposition"] != decision.primary_disposition
        ):
            raise ValueError("v2 metrics dispositions are not derived from intervals")
        return decision
    if schema != "h6-prediction-metrics-v1":
        raise ValueError("metrics bytes do not match a supported H6 schema")
    estimator_complete = payload.get("estimator_complete")
    if type(estimator_complete) is not bool:
        raise ValueError("metrics estimator_complete must be boolean")
    interval_payload = payload.get("primary_interval")
    if interval_payload is None:
        interval = None
    elif (
        type(interval_payload) is not dict
        or set(interval_payload) != {"lower", "upper"}
        or any(type(interval_payload[name]) is not float for name in ("lower", "upper"))
    ):
        raise ValueError("metrics primary_interval must be an exact finite float pair")
    else:
        interval = (interval_payload["lower"], interval_payload["upper"])
    return PredictionDecision.classify(
        primary_interval=interval,
        estimator_complete=estimator_complete,
    )


def _prediction_metrics_interval(
    payload: object,
    *,
    name: str,
    required: bool,
) -> tuple[float, float] | None:
    if payload is None:
        if required:
            raise ValueError(f"metrics {name} is required")
        return None
    if (
        type(payload) is not dict
        or set(payload) != {"lower", "upper"}
        or any(type(payload[key]) is not float for key in ("lower", "upper"))
    ):
        raise ValueError(f"metrics {name} must be an exact finite float pair")
    interval = (payload["lower"], payload["upper"])
    if (
        not all(math.isfinite(value) for value in interval)
        or interval[0] > interval[1]
    ):
        raise ValueError(f"metrics {name} must be an ordered finite pair")
    return interval


def _require_allowance_pair(value: object, name: str) -> None:
    if type(value) is not tuple or len(value) != 2 or not all(
        isinstance(item, NumericalAllowance) for item in value
    ):
        raise ValueError(f"{name} must be a pair of NumericalAllowance")


def _require_sha256(value: object, name: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase 64-hex SHA-256")


def _require_obligations(value: object) -> None:
    if type(value) is not tuple or any(type(item) is not str or not item for item in value):
        raise ValueError("obligations must be a tuple of nonempty strings")


def _require_finite_pair(value: object, name: str) -> None:
    if type(value) is not tuple or len(value) != 2:
        raise ValueError(f"{name} must be a pair")
    for index, item in enumerate(value):
        _require_finite(item, f"{name}[{index}]")


def _require_nonnegative_finite(value: object, name: str) -> None:
    _require_finite(value, name)
    if float(value) < 0.0:
        raise ValueError(f"{name} must be nonnegative")


def _require_optional_finite(value: object, name: str) -> None:
    if value is not None:
        _require_finite(value, name)


def _require_finite(value: object, name: str) -> None:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
