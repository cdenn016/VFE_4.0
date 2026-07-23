"""Immutable result records with fail-closed validation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Literal


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


@dataclass(frozen=True, init=False)
class H6PredictionResult:
    """Result of the separately authorized H6 Prediction evidence stage."""

    gate: Literal["H6-Prediction"]
    status: GateStatus
    readiness_sha256: str
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
        from .h6 import EvidenceStatus, H6PredictionReadinessToken, PredictionDecision

        if type(self._readiness) is not H6PredictionReadinessToken:
            raise ValueError("typed H6 Prediction readiness is required")
        if type(self._decision) is not PredictionDecision:
            raise ValueError("typed PredictionDecision is required")
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
        instance = object.__new__(cls)
        values = {
            "gate": "H6-Prediction",
            "status": GateStatus[decision.status.name],
            "readiness_sha256": readiness.readiness_sha256,
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


def _prediction_decision_from_metrics(metrics_bytes: bytes) -> object:
    from .h6 import PredictionDecision

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
    if canonical != metrics_bytes or payload.get("schema") != "h6-prediction-metrics-v1":
        raise ValueError("metrics bytes do not match the canonical H6 schema")
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
