"""Immutable result records with fail-closed validation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
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
        arithmetic_allowance = 256.0 * math.ulp(1.0) * max(
            1.0, sum(abs(term) for term in objective_terms)
        )
        if abs(self.complete_elbo - expected) > arithmetic_allowance:
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
    gate: Literal["H1"]
    status: GateStatus
    fixture_id: Literal["h1-v1"]
    residual: float | None
    calibrated_allowance: float | None
    measurements: Mapping[str, float | None]
    invariants: tuple[InvariantResult, ...]
    obligations: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.gate != "H1":
            raise ValueError("gate must be H1")
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
            if type(name) is not str:
                raise ValueError("measurement names must be strings")
            _require_optional_finite(value, f"measurements[{name!r}]")
        if type(self.invariants) is not tuple or not all(
            isinstance(item, InvariantResult) for item in self.invariants
        ):
            raise ValueError("invariants must be a tuple of InvariantResult")
        if type(self.obligations) is not tuple or not all(
            type(item) is str and item for item in self.obligations
        ):
            raise ValueError("obligations must be a tuple of nonempty strings")

        if self.status is GateStatus.INCONCLUSIVE:
            if not self.obligations:
                raise ValueError("inconclusive results require an obligation")
        else:
            _require_finite(self.residual, "residual")
            _require_finite(self.calibrated_allowance, "calibrated_allowance")
            if any(value is None for value in copied_measurements.values()):
                raise ValueError("pass/fail results require finite measurements")
        object.__setattr__(self, "measurements", MappingProxyType(copied_measurements))


def _require_allowance_pair(value: object, name: str) -> None:
    if type(value) is not tuple or len(value) != 2 or not all(
        isinstance(item, NumericalAllowance) for item in value
    ):
        raise ValueError(f"{name} must be a pair of NumericalAllowance")


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
