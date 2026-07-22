"""Immutable H3-only fixture, configuration, arm, and gate records."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, TypeAlias

from .results import GateStatus, InvariantResult


H3FixtureId: TypeAlias = Literal["h3-coupled-v1", "h3-zero-control-v1"]
H3FixtureKind: TypeAlias = Literal["coupled", "zero_control"]
H3RecognitionFamily: TypeAlias = Literal[
    "structured_full_spd", "fine_factorized_diagonal"
]
H3Vector4: TypeAlias = tuple[float, float, float, float]
H3Matrix4: TypeAlias = tuple[H3Vector4, H3Vector4, H3Vector4, H3Vector4]

_ZERO_MEAN: H3Vector4 = (0.0, 0.0, 0.0, 0.0)
_IDENTITY_PRECISION: H3Matrix4 = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")

# These invariants are availability or eligibility checks. They deliberately do
# not receive fabricated rounding allowances. Every other H3 invariant is a
# numerical comparison and must own one exact-name operand-local record.
H3_NON_ALLOWANCE_INVARIANTS = frozenset(
    {
        "fixture_hashes_match",
        "independent_control_contract",
        "posterior_condition_envelope",
        "all_arms_converged",
        "all_invariant_allowances_decisive",
    }
)


@dataclass(frozen=True)
class H3ScalarFactorRecord:
    factor_id: str
    row: H3Vector4
    target: float
    variance: float

    def __post_init__(self) -> None:
        if type(self.factor_id) is not str or not self.factor_id:
            raise ValueError("factor_id must be a nonempty string")
        object.__setattr__(self, "row", _vector4(self.row, "row"))
        _finite(self.target, "target")
        _positive_finite(self.variance, "variance")


@dataclass(frozen=True)
class H3Fixture:
    fixture_schema_version: int
    fixture_id: H3FixtureId
    kind: H3FixtureKind
    horizon: int
    d_z: int
    d_m: int
    dimension: int
    continuous_order: tuple[str, str, str, str]
    initial_factors: tuple[H3ScalarFactorRecord, H3ScalarFactorRecord]
    transition_factors: tuple[H3ScalarFactorRecord, H3ScalarFactorRecord]
    observation_map: tuple[H3Vector4, H3Vector4]
    observation_values: tuple[float, float]
    observation_covariance: tuple[tuple[float, float], tuple[float, float]]
    observation_factors: tuple[H3ScalarFactorRecord, H3ScalarFactorRecord]
    reference_posterior_precision: H3Matrix4
    reference_posterior_natural: H3Vector4 | None
    reference_log_evidence: float | None
    reference_analytic_factorized_reverse_kl: float | None

    def __post_init__(self) -> None:
        if type(self.fixture_schema_version) is not int or self.fixture_schema_version != 1:
            raise ValueError("fixture_schema_version must equal integer 1")
        if self.fixture_id not in ("h3-coupled-v1", "h3-zero-control-v1"):
            raise ValueError("fixture_id must be an H3 fixture ID")
        expected_kind = (
            "coupled" if self.fixture_id == "h3-coupled-v1" else "zero_control"
        )
        if self.kind != expected_kind:
            raise ValueError("kind must match fixture_id")
        if type(self.horizon) is not int or self.horizon != 1:
            raise ValueError("horizon must equal integer 1")
        if (type(self.d_z), type(self.d_m), type(self.dimension)) != (int, int, int):
            raise ValueError("dimensions must be integers")
        if (self.d_z, self.d_m, self.dimension) != (1, 1, 4):
            raise ValueError("H3 dimensions must equal d_z=1, d_m=1, D=4")
        if self.continuous_order != ("z0", "m0", "z1", "m1"):
            raise ValueError("continuous_order must equal (z0,m0,z1,m1)")
        for name, value in (
            ("initial_factors", self.initial_factors),
            ("transition_factors", self.transition_factors),
            ("observation_factors", self.observation_factors),
        ):
            if type(value) is not tuple or len(value) != 2 or not all(
                isinstance(item, H3ScalarFactorRecord) for item in value
            ):
                raise ValueError(f"{name} must contain exactly two H3 scalar factors")
        factor_ids = tuple(factor.factor_id for factor in self.factors)
        if factor_ids != (
            "z0_prior",
            "m0_prior",
            "m1_transition",
            "z1_transition",
            "z1_observation",
            "m1_observation",
        ):
            raise ValueError("H3 factor IDs and order must match the frozen contract")
        if len(set(factor_ids)) != 6:
            raise ValueError("H3 factor IDs must be unique")
        object.__setattr__(
            self,
            "observation_map",
            _matrix_rows(self.observation_map, 2, "observation_map"),
        )
        object.__setattr__(
            self,
            "observation_values",
            _vector(self.observation_values, 2, "observation_values"),
        )
        object.__setattr__(
            self,
            "observation_covariance",
            _matrix(self.observation_covariance, 2, 2, "observation_covariance"),
        )
        object.__setattr__(
            self,
            "reference_posterior_precision",
            _matrix4(self.reference_posterior_precision, "reference_posterior_precision"),
        )
        if self.reference_posterior_natural is not None:
            object.__setattr__(
                self,
                "reference_posterior_natural",
                _vector4(
                    self.reference_posterior_natural,
                    "reference_posterior_natural",
                ),
            )
        _optional_finite(self.reference_log_evidence, "reference_log_evidence")
        _optional_nonnegative_finite(
            self.reference_analytic_factorized_reverse_kl,
            "reference_analytic_factorized_reverse_kl",
        )

    @property
    def factors(self) -> tuple[H3ScalarFactorRecord, ...]:
        return (*self.initial_factors, *self.transition_factors, *self.observation_factors)


@dataclass(frozen=True)
class H3InitializationConfig:
    mean: H3Vector4 = _ZERO_MEAN
    precision: H3Matrix4 = _IDENTITY_PRECISION

    def __post_init__(self) -> None:
        object.__setattr__(self, "mean", _vector4(self.mean, "mean"))
        object.__setattr__(self, "precision", _matrix4(self.precision, "precision"))
        if self.mean != _ZERO_MEAN or self.precision != _IDENTITY_PRECISION:
            raise ValueError("H3 initialization must be zero mean and identity precision")


@dataclass(frozen=True)
class H3OptimizationConfig:
    learning_rate: float = 1.0
    maximum_iterations_per_step: int = 1
    maximum_evaluations_per_step: int = 25
    tolerance_gradient: float = 1.0e-12
    tolerance_change: float = 1.0e-18
    history_size: int = 20
    line_search: str = "strong_wolfe"
    maximum_accepted_iterations: int = 200
    maximum_closure_evaluations: int = 5_000
    terminal_gradient_infinity_norm: float = 1.0e-8
    terminal_objective_change: float = 1.0e-12
    required_consecutive_accepted_iterations: int = 3

    def __post_init__(self) -> None:
        expected = (
            self.learning_rate == 1.0,
            type(self.maximum_iterations_per_step) is int
            and self.maximum_iterations_per_step == 1,
            type(self.maximum_evaluations_per_step) is int
            and self.maximum_evaluations_per_step == 25,
            self.tolerance_gradient == 1.0e-12,
            self.tolerance_change == 1.0e-18,
            type(self.history_size) is int and self.history_size == 20,
            self.line_search == "strong_wolfe",
            type(self.maximum_accepted_iterations) is int
            and self.maximum_accepted_iterations == 200,
            type(self.maximum_closure_evaluations) is int
            and self.maximum_closure_evaluations == 5_000,
            self.terminal_gradient_infinity_norm == 1.0e-8,
            self.terminal_objective_change == 1.0e-12,
            type(self.required_consecutive_accepted_iterations) is int
            and self.required_consecutive_accepted_iterations == 3,
        )
        for name, value in (
            ("learning_rate", self.learning_rate),
            ("tolerance_gradient", self.tolerance_gradient),
            ("tolerance_change", self.tolerance_change),
            ("terminal_gradient_infinity_norm", self.terminal_gradient_infinity_norm),
            ("terminal_objective_change", self.terminal_objective_change),
        ):
            _positive_finite(value, name)
        if not all(expected):
            raise ValueError("H3 optimizer settings must equal the frozen contract")


@dataclass(frozen=True)
class H3DecisionConfig:
    dimension: int = 4
    minimum_precision_eigenvalue: float = 1.0e-4
    maximum_precision_eigenvalue: float = 1.0e4
    maximum_precision_condition_number: float = 1.0e6
    maximum_mean_infinity_norm: float = 4.0
    minimum_coupled_gap_nats: float = 0.50
    maximum_structured_gap_fraction: float = 0.01
    maximum_allowance_fraction: float = 0.01

    def __post_init__(self) -> None:
        for name, value in (
            ("minimum_precision_eigenvalue", self.minimum_precision_eigenvalue),
            ("maximum_precision_eigenvalue", self.maximum_precision_eigenvalue),
            (
                "maximum_precision_condition_number",
                self.maximum_precision_condition_number,
            ),
            ("maximum_mean_infinity_norm", self.maximum_mean_infinity_norm),
            ("minimum_coupled_gap_nats", self.minimum_coupled_gap_nats),
            ("maximum_structured_gap_fraction", self.maximum_structured_gap_fraction),
            ("maximum_allowance_fraction", self.maximum_allowance_fraction),
        ):
            _positive_finite(value, name)
        if (
            type(self.dimension) is not int
            or self.dimension != 4
            or self.minimum_precision_eigenvalue != 1.0e-4
            or self.maximum_precision_eigenvalue != 1.0e4
            or self.maximum_precision_condition_number != 1.0e6
            or self.maximum_mean_infinity_norm != 4.0
            or self.minimum_coupled_gap_nats != 0.50
            or self.maximum_structured_gap_fraction != 0.01
            or self.maximum_allowance_fraction != 0.01
        ):
            raise ValueError("H3 decision settings must equal the frozen contract")


@dataclass(frozen=True)
class H3ArmResult:
    family: H3RecognitionFamily
    converged: bool
    failure_reason: str | None
    accepted_iterations: int
    closure_evaluations: int
    terminal_elbo: float | None
    terminal_gradient_infinity_norm: float | None
    terminal_objective_change: float | None
    terminal_mean: H3Vector4 | None
    terminal_precision_cholesky: H3Matrix4 | None
    terminal_precision: H3Matrix4 | None
    accepted_elbos: tuple[float, ...]
    canonical_trace_sha256: str

    def __post_init__(self) -> None:
        if self.family not in (
            "structured_full_spd",
            "fine_factorized_diagonal",
        ):
            raise ValueError("family must be an H3 recognition family")
        if type(self.converged) is not bool:
            raise ValueError("converged must be a bool")
        if type(self.accepted_iterations) is not int or self.accepted_iterations < 0:
            raise ValueError("accepted_iterations must be a nonnegative integer")
        if type(self.closure_evaluations) is not int or self.closure_evaluations < 0:
            raise ValueError("closure_evaluations must be a nonnegative integer")
        if type(self.accepted_elbos) is not tuple:
            raise ValueError("accepted_elbos must be a tuple")
        if self.accepted_iterations != len(self.accepted_elbos):
            raise ValueError(
                "accepted_iterations must equal the number of accepted_elbos"
            )
        for index, value in enumerate(self.accepted_elbos):
            _finite(value, f"accepted_elbos[{index}]")
        for name, value in (
            ("terminal_elbo", self.terminal_elbo),
            (
                "terminal_gradient_infinity_norm",
                self.terminal_gradient_infinity_norm,
            ),
            ("terminal_objective_change", self.terminal_objective_change),
        ):
            _optional_finite(value, name)
        if self.terminal_mean is not None:
            object.__setattr__(self, "terminal_mean", _vector4(self.terminal_mean, "terminal_mean"))
        if self.terminal_precision_cholesky is not None:
            cholesky = _matrix4(
                self.terminal_precision_cholesky,
                "terminal_precision_cholesky",
            )
            if any(cholesky[row][column] != 0.0 for row in range(4) for column in range(row + 1, 4)):
                raise ValueError("terminal_precision_cholesky must be lower triangular")
            if any(cholesky[index][index] <= 0.0 for index in range(4)):
                raise ValueError("terminal_precision_cholesky must have positive diagonal")
            object.__setattr__(self, "terminal_precision_cholesky", cholesky)
        if self.terminal_precision is not None:
            object.__setattr__(
                self,
                "terminal_precision",
                _matrix4(self.terminal_precision, "terminal_precision"),
            )
        if self.canonical_trace_sha256 is None:
            raise ValueError("canonical_trace_sha256 must be present for every H3 arm")
        _sha256(self.canonical_trace_sha256, "canonical_trace_sha256")
        if self.converged:
            if self.failure_reason is not None:
                raise ValueError("converged H3 arms cannot retain a failure reason")
            required = (
                self.terminal_elbo,
                self.terminal_gradient_infinity_norm,
                self.terminal_objective_change,
                self.terminal_mean,
                self.terminal_precision_cholesky,
                self.terminal_precision,
                self.canonical_trace_sha256,
            )
            if any(value is None for value in required) or not self.accepted_elbos:
                raise ValueError("converged H3 arms require complete terminal evidence")
        elif type(self.failure_reason) is not str or not self.failure_reason:
            raise ValueError("nonconverged H3 arms require a failure reason")


@dataclass(frozen=True)
class H3FixtureHashes:
    coupled_expected_sha256: str
    coupled_observed_sha256: str
    zero_control_expected_sha256: str
    zero_control_observed_sha256: str

    def __post_init__(self) -> None:
        for name, value in (
            ("coupled_expected_sha256", self.coupled_expected_sha256),
            ("coupled_observed_sha256", self.coupled_observed_sha256),
            ("zero_control_expected_sha256", self.zero_control_expected_sha256),
            ("zero_control_observed_sha256", self.zero_control_observed_sha256),
        ):
            _sha256(value, name)
        if self.coupled_expected_sha256 == self.zero_control_expected_sha256:
            raise ValueError("the two expected raw fixture hashes must differ")
        if self.coupled_observed_sha256 == self.zero_control_observed_sha256:
            raise ValueError("the two observed raw fixture hashes must differ")

    @property
    def coupled_matches(self) -> bool:
        return self.coupled_expected_sha256 == self.coupled_observed_sha256

    @property
    def zero_control_matches(self) -> bool:
        return self.zero_control_expected_sha256 == self.zero_control_observed_sha256


@dataclass(frozen=True)
class H3GateResult:
    gate: Literal["H3"]
    coupled_fixture_id: Literal["h3-coupled-v1"]
    zero_control_fixture_id: Literal["h3-zero-control-v1"]
    status: GateStatus
    measurements: Mapping[str, float | None]
    invariants: tuple[InvariantResult, ...]
    allowances_by_invariant: Mapping[str, object] = field(default_factory=dict)
    obligations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.gate != "H3":
            raise ValueError("gate must be H3")
        if self.coupled_fixture_id != "h3-coupled-v1":
            raise ValueError("coupled_fixture_id must be h3-coupled-v1")
        if self.zero_control_fixture_id != "h3-zero-control-v1":
            raise ValueError("zero_control_fixture_id must be h3-zero-control-v1")
        if not isinstance(self.status, GateStatus):
            raise ValueError("status must be a GateStatus")
        if not isinstance(self.measurements, Mapping):
            raise ValueError("measurements must be a mapping")
        copied_measurements = dict(self.measurements)
        for name, value in copied_measurements.items():
            if type(name) is not str or not name:
                raise ValueError("measurement names must be nonempty strings")
            _optional_finite(value, f"measurements[{name!r}]")
        if type(self.invariants) is not tuple or not all(
            isinstance(item, InvariantResult) for item in self.invariants
        ):
            raise ValueError("invariants must be a tuple of InvariantResult")
        invariant_names = tuple(item.name for item in self.invariants)
        if len(set(invariant_names)) != len(invariant_names):
            raise ValueError("invariant names must be unique")
        if not isinstance(self.allowances_by_invariant, Mapping):
            raise ValueError("allowances_by_invariant must be a mapping")
        allowance_mapping = dict(self.allowances_by_invariant)
        if any(type(name) is not str or not name for name in allowance_mapping):
            raise ValueError("allowance invariant names must be nonempty strings")
        expected_allowances = {
            name for name in invariant_names if name not in H3_NON_ALLOWANCE_INVARIANTS
        }
        eligibility_invariants = tuple(
            item
            for item in self.invariants
            if item.name in H3_NON_ALLOWANCE_INVARIANTS
        )
        decision_invariants = tuple(
            item
            for item in self.invariants
            if item.name not in H3_NON_ALLOWANCE_INVARIANTS
        )
        if set(allowance_mapping) != expected_allowances:
            raise ValueError(
                "allowances_by_invariant names must exactly match allowance-bearing invariants"
            )
        frozen_allowances: dict[str, object] = {}
        for name, record in allowance_mapping.items():
            if not isinstance(record, Mapping) or not record:
                raise ValueError(f"allowance record {name!r} must be a nonempty mapping")
            frozen_allowances[name] = _freeze_json_like(record, f"allowance[{name!r}]")
        if type(self.obligations) is not tuple or not all(
            type(item) is str and item for item in self.obligations
        ):
            raise ValueError("obligations must be a tuple of nonempty strings")

        if self.status is GateStatus.INCONCLUSIVE:
            if not self.obligations:
                raise ValueError("inconclusive H3 results require an obligation")
        else:
            if not copied_measurements:
                raise ValueError("pass/fail H3 measurements must be nonempty")
            if not self.invariants:
                raise ValueError("pass/fail H3 invariants must be nonempty")
            if self.obligations:
                raise ValueError("pass/fail H3 results cannot retain obligations")
            if any(value is None for value in copied_measurements.values()):
                raise ValueError("pass/fail H3 results require finite measurements")
            if any(not item.passed for item in eligibility_invariants):
                raise ValueError(
                    "pass/fail H3 results require every eligibility invariant to pass"
                )
            if any(
                item.value is None or item.limit is None
                for item in decision_invariants
            ):
                raise ValueError(
                    "pass/fail H3 decision invariants require finite values and limits"
                )
            if self.status is GateStatus.PASS and not all(
                item.passed for item in self.invariants
            ):
                raise ValueError("H3 PASS requires every invariant to pass")
            if self.status is GateStatus.FAIL and not any(
                not item.passed for item in decision_invariants
            ):
                raise ValueError(
                    "H3 FAIL requires a finite failed allowance-bearing invariant"
                )
        object.__setattr__(
            self,
            "measurements",
            MappingProxyType(copied_measurements),
        )
        object.__setattr__(
            self,
            "allowances_by_invariant",
            MappingProxyType(frozen_allowances),
        )


def _freeze_json_like(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        copied: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str or not key:
                raise ValueError(f"{name} keys must be nonempty strings")
            copied[key] = _freeze_json_like(item, f"{name}.{key}")
        return MappingProxyType(copied)
    if type(value) in (tuple, list):
        return tuple(
            _freeze_json_like(item, f"{name}[{index}]")
            for index, item in enumerate(value)
        )
    if value is None or type(value) in (str, bool):
        return value
    if type(value) in (int, float):
        _finite(value, name)
        return value
    raise ValueError(f"{name} must contain immutable JSON-compatible values")


def _matrix_rows(value: object, rows: int, name: str) -> tuple[H3Vector4, ...]:
    if type(value) is not tuple or len(value) != rows:
        raise ValueError(f"{name} must be a tuple with {rows} rows")
    return tuple(_vector4(row, f"{name}[{index}]") for index, row in enumerate(value))


def _matrix4(value: object, name: str) -> H3Matrix4:
    rows = _matrix_rows(value, 4, name)
    return (rows[0], rows[1], rows[2], rows[3])


def _matrix(
    value: object, rows: int, columns: int, name: str
) -> tuple[tuple[float, ...], ...]:
    if type(value) is not tuple or len(value) != rows:
        raise ValueError(f"{name} must be a tuple with {rows} rows")
    return tuple(
        _vector(row, columns, f"{name}[{index}]")
        for index, row in enumerate(value)
    )


def _vector4(value: object, name: str) -> H3Vector4:
    result = _vector(value, 4, name)
    return (result[0], result[1], result[2], result[3])


def _vector(value: object, size: int, name: str) -> tuple[float, ...]:
    if type(value) is not tuple or len(value) != size:
        raise ValueError(f"{name} must be a numeric tuple of length {size}")
    result = tuple(float(item) for item in value)
    for index, item in enumerate(value):
        _finite(item, f"{name}[{index}]")
    return result


def _sha256(value: object, name: str) -> None:
    if type(value) is not str or _HEX_SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 64-hex SHA-256 digest")


def _optional_nonnegative_finite(value: object, name: str) -> None:
    if value is not None:
        _finite(value, name)
        if float(value) < 0.0:
            raise ValueError(f"{name} must be nonnegative")


def _optional_finite(value: object, name: str) -> None:
    if value is not None:
        _finite(value, name)


def _positive_finite(value: object, name: str) -> None:
    _finite(value, name)
    if float(value) <= 0.0:
        raise ValueError(f"{name} must be positive")


def _finite(value: object, name: str) -> None:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite numeric data")


__all__ = [
    "H3ArmResult",
    "H3DecisionConfig",
    "H3Fixture",
    "H3FixtureHashes",
    "H3FixtureId",
    "H3FixtureKind",
    "H3GateResult",
    "H3InitializationConfig",
    "H3Matrix4",
    "H3OptimizationConfig",
    "H3RecognitionFamily",
    "H3ScalarFactorRecord",
    "H3Vector4",
]
