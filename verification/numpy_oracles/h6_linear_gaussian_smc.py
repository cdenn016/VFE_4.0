"""Independent NumPy-only linear-Gaussian sensitivity control for H6 SMC.

This module does not import Torch, the production particle recursion, or
``vfe4.numerics.linear_gaussian``.  It supplies an exact Kalman innovation
recursion and an immutable report for externally produced raw SMC estimates.
The report is deliberately a sensitivity control; it cannot transfer a bias
sign or bound to a trained language-model endpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


_FIXTURE_SCHEMA = "h6-linear-gaussian-fixture-v1"
_REPORT_SCHEMA = "h6-linear-gaussian-smc-sensitivity-v1"
_ALLOWED_FIXTURE_PROFILES = ((3, 2), (32, 16))
PINNED_T32_D16_RELATIVE_PATH = (
    "verification/fixtures/h6_linear_gaussian_t32_d16_v1.json"
)
PINNED_T32_D16_SHA256 = (
    "95cfeeb7ff9abc4935937d97db6a8906343d6ad5e1bf74726600d9d81aa8bfcd"
)


def _finite_float(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite binary64 float")
    return value


def _sha256(domain: bytes, payload: object) -> str:
    canonical = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(domain + b"\x00" + canonical).hexdigest()


def _canonical_fixture_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


def _require_vector(
    values: object,
    *,
    length: int,
    name: str,
) -> tuple[float, ...]:
    if type(values) is not tuple or len(values) != length:
        raise ValueError(f"{name} must have shape ({length},)")
    for value in values:
        _finite_float(value, name)
    return values


def _require_matrix(
    values: object,
    *,
    rows: int,
    columns: int,
    name: str,
) -> tuple[tuple[float, ...], ...]:
    if type(values) is not tuple or len(values) != rows:
        raise ValueError(f"{name} must have shape ({rows}, {columns})")
    for row in values:
        _require_vector(row, length=columns, name=name)
    return values


@dataclass(frozen=True, slots=True)
class LinearGaussianFixture:
    """Owned deterministic state-space fixture for an innovation recursion."""

    schema_version: Literal["h6-linear-gaussian-fixture-v1"]
    time_steps: int
    state_dimension: int
    observation_dimension: int
    initial_mean: tuple[float, ...]
    initial_covariance: tuple[tuple[float, ...], ...]
    transition_matrix: tuple[tuple[float, ...], ...]
    process_covariance: tuple[tuple[float, ...], ...]
    observation_matrix: tuple[tuple[float, ...], ...]
    observation_covariance: tuple[tuple[float, ...], ...]
    observations: tuple[tuple[float, ...], ...]
    fixture_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != _FIXTURE_SCHEMA:
            raise ValueError("linear-Gaussian fixture schema is frozen")
        if (
            type(self.time_steps) is not int
            or type(self.state_dimension) is not int
            or (self.time_steps, self.state_dimension)
            not in _ALLOWED_FIXTURE_PROFILES
            or self.observation_dimension != self.state_dimension
        ):
            raise ValueError(
                "fixture must be the (T=3,d=2) unit profile or "
                "(T=32,d=16) evidence profile"
            )
        state = self.state_dimension
        observation = self.observation_dimension
        _require_vector(
            self.initial_mean,
            length=state,
            name="initial_mean",
        )
        for values, rows, columns, name in (
            (
                self.initial_covariance,
                state,
                state,
                "initial_covariance",
            ),
            (self.transition_matrix, state, state, "transition_matrix"),
            (self.process_covariance, state, state, "process_covariance"),
            (
                self.observation_matrix,
                observation,
                state,
                "observation_matrix",
            ),
            (
                self.observation_covariance,
                observation,
                observation,
                "observation_covariance",
            ),
            (
                self.observations,
                self.time_steps,
                observation,
                "observations",
            ),
        ):
            _require_matrix(
                values,
                rows=rows,
                columns=columns,
                name=name,
            )
        for covariance, name in (
            (self.initial_covariance, "initial_covariance"),
            (self.process_covariance, "process_covariance"),
            (self.observation_covariance, "observation_covariance"),
        ):
            array = np.asarray(covariance, dtype=np.float64)
            if not np.array_equal(array, array.T):
                raise ValueError(f"{name} must be exactly symmetric")
            try:
                np.linalg.cholesky(array)
            except np.linalg.LinAlgError as exc:
                raise ValueError(f"{name} must be positive definite") from exc
        expected = hashlib.sha256(
            _canonical_fixture_bytes(_fixture_payload(self))
        ).hexdigest()
        if self.fixture_sha256 != expected:
            raise ValueError("fixture_sha256 is stale")


def _fixture_payload(fixture: LinearGaussianFixture) -> dict[str, object]:
    return {
        "schema_version": fixture.schema_version,
        "time_steps": fixture.time_steps,
        "state_dimension": fixture.state_dimension,
        "observation_dimension": fixture.observation_dimension,
        "initial_mean": fixture.initial_mean,
        "initial_covariance": fixture.initial_covariance,
        "transition_matrix": fixture.transition_matrix,
        "process_covariance": fixture.process_covariance,
        "observation_matrix": fixture.observation_matrix,
        "observation_covariance": fixture.observation_covariance,
        "observations": fixture.observations,
    }


def deterministic_linear_gaussian_fixture(
    *,
    time_steps: Literal[3] = 3,
    dimension: Literal[2] = 2,
) -> LinearGaussianFixture:
    """Construct only the tiny generated unit-test fixture."""

    if (time_steps, dimension) != (3, 2):
        raise ValueError(
            "the generated fixture is only the (T=3,d=2) unit profile; "
            "load the pinned JSON for (T=32,d=16)"
        )
    identity = tuple(
        tuple(float(row == column) for column in range(dimension))
        for row in range(dimension)
    )
    transition = tuple(
        tuple(
            0.75
            if row == column
            else 0.05
            if column == row + 1
            else 0.0
            for column in range(dimension)
        )
        for row in range(dimension)
    )
    process_covariance = tuple(
        tuple(0.2 if row == column else 0.0 for column in range(dimension))
        for row in range(dimension)
    )
    observation_covariance = tuple(
        tuple(0.3 if row == column else 0.0 for column in range(dimension))
        for row in range(dimension)
    )
    observations = tuple(
        tuple(
            float(
                math.sin((time_index + 1) * (component + 1) / 17.0)
                + 0.1
                * math.cos(
                    (time_index + 1 + 2 * component) / 11.0
                )
            )
            for component in range(dimension)
        )
        for time_index in range(time_steps)
    )
    values: dict[str, object] = {
        "schema_version": _FIXTURE_SCHEMA,
        "time_steps": time_steps,
        "state_dimension": dimension,
        "observation_dimension": dimension,
        "initial_mean": (0.0,) * dimension,
        "initial_covariance": identity,
        "transition_matrix": transition,
        "process_covariance": process_covariance,
        "observation_matrix": identity,
        "observation_covariance": observation_covariance,
        "observations": observations,
    }
    provisional = object.__new__(LinearGaussianFixture)
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    fixture_sha256 = hashlib.sha256(
        _canonical_fixture_bytes(_fixture_payload(provisional))
    ).hexdigest()
    return LinearGaussianFixture(
        **values,
        fixture_sha256=fixture_sha256,
    )


def fixture_canonical_bytes(fixture: LinearGaussianFixture) -> bytes:
    """Return stable canonical JSON bytes for the exact owned fixture."""

    if type(fixture) is not LinearGaussianFixture:
        raise ValueError("canonicalization requires LinearGaussianFixture")
    fixture.__post_init__()
    return _canonical_fixture_bytes(_fixture_payload(fixture))


def _tuple_vector(value: object, name: str) -> tuple[float, ...]:
    if type(value) is not list:
        raise ValueError(f"{name} must be a JSON array")
    return tuple(float(item) for item in value)


def _tuple_matrix(
    value: object,
    name: str,
) -> tuple[tuple[float, ...], ...]:
    if type(value) is not list:
        raise ValueError(f"{name} must be a JSON matrix")
    return tuple(_tuple_vector(row, name) for row in value)


def load_pinned_linear_gaussian_fixture(
    fixture_path: Path,
) -> LinearGaussianFixture:
    """Load and byte-verify the checked-in T=32, d=16 decimal fixture."""

    if not isinstance(fixture_path, Path):
        raise ValueError("fixture_path must be pathlib.Path")
    raw = fixture_path.read_bytes()
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    if raw_sha256 != PINNED_T32_D16_SHA256:
        raise ValueError("pinned T32,d16 fixture bytes do not match digest")
    try:
        root = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("pinned fixture is not canonical JSON") from exc
    expected_keys = {
        "schema_version",
        "time_steps",
        "state_dimension",
        "observation_dimension",
        "initial_mean",
        "initial_covariance",
        "transition_matrix",
        "process_covariance",
        "observation_matrix",
        "observation_covariance",
        "observations",
    }
    if type(root) is not dict or set(root) != expected_keys:
        raise ValueError("pinned fixture JSON keys are stale")
    if raw != _canonical_fixture_bytes(root):
        raise ValueError("pinned fixture JSON bytes are not canonical")
    fixture = LinearGaussianFixture(
        schema_version=root["schema_version"],
        time_steps=root["time_steps"],
        state_dimension=root["state_dimension"],
        observation_dimension=root["observation_dimension"],
        initial_mean=_tuple_vector(root["initial_mean"], "initial_mean"),
        initial_covariance=_tuple_matrix(
            root["initial_covariance"],
            "initial_covariance",
        ),
        transition_matrix=_tuple_matrix(
            root["transition_matrix"],
            "transition_matrix",
        ),
        process_covariance=_tuple_matrix(
            root["process_covariance"],
            "process_covariance",
        ),
        observation_matrix=_tuple_matrix(
            root["observation_matrix"],
            "observation_matrix",
        ),
        observation_covariance=_tuple_matrix(
            root["observation_covariance"],
            "observation_covariance",
        ),
        observations=_tuple_matrix(root["observations"], "observations"),
        fixture_sha256=raw_sha256,
    )
    if (
        fixture.time_steps,
        fixture.state_dimension,
        fixture.observation_dimension,
    ) != (32, 16, 16):
        raise ValueError("pinned fixture profile is not T32,d16")
    return fixture


def kalman_log_likelihood(fixture: LinearGaussianFixture) -> float:
    """Evaluate ``log p(y_0:T)`` by an independent innovation recursion."""

    if type(fixture) is not LinearGaussianFixture:
        raise ValueError("Kalman oracle requires LinearGaussianFixture")
    fixture.__post_init__()
    mean = np.asarray(fixture.initial_mean, dtype=np.float64)
    covariance = np.asarray(
        fixture.initial_covariance,
        dtype=np.float64,
    )
    transition = np.asarray(fixture.transition_matrix, dtype=np.float64)
    process = np.asarray(fixture.process_covariance, dtype=np.float64)
    observation_matrix = np.asarray(
        fixture.observation_matrix,
        dtype=np.float64,
    )
    observation_noise = np.asarray(
        fixture.observation_covariance,
        dtype=np.float64,
    )
    observations = np.asarray(fixture.observations, dtype=np.float64)
    dimension = fixture.observation_dimension
    normalizer = dimension * math.log(2.0 * math.pi)
    log_likelihood_terms: list[float] = []

    for time_index, observation in enumerate(observations):
        innovation = observation - observation_matrix @ mean
        projected_covariance = observation_matrix @ covariance
        innovation_covariance = (
            projected_covariance @ observation_matrix.T
            + observation_noise
        )
        sign, log_determinant = np.linalg.slogdet(innovation_covariance)
        if sign <= 0.0 or not math.isfinite(float(log_determinant)):
            raise ValueError("innovation covariance is not positive definite")
        solved_innovation = np.linalg.solve(
            innovation_covariance,
            innovation,
        )
        quadratic = float(innovation @ solved_innovation)
        log_likelihood_terms.append(
            -0.5 * (normalizer + float(log_determinant) + quadratic)
        )
        gain = np.linalg.solve(
            innovation_covariance,
            projected_covariance,
        ).T
        mean = mean + gain @ innovation
        covariance = covariance - gain @ projected_covariance
        covariance = 0.5 * (covariance + covariance.T)
        if time_index + 1 < fixture.time_steps:
            mean = transition @ mean
            covariance = transition @ covariance @ transition.T + process
            covariance = 0.5 * (covariance + covariance.T)

    result = float(math.fsum(log_likelihood_terms))
    if not math.isfinite(result):
        raise ValueError("Kalman log likelihood is nonfinite")
    return result


@dataclass(frozen=True, slots=True)
class LinearGaussianSmcRawRecord:
    stream_id: int
    particle_count: int
    raw_log_likelihood: float

    def __post_init__(self) -> None:
        if type(self.stream_id) is not int or self.stream_id < 0:
            raise ValueError("stream_id must be a nonnegative integer")
        if type(self.particle_count) is not int or self.particle_count <= 0:
            raise ValueError("particle_count must be a positive integer")
        _finite_float(self.raw_log_likelihood, "raw_log_likelihood")


@dataclass(frozen=True, slots=True)
class LinearGaussianQ2Record:
    stream_id: int
    lower_particle_count: int
    upper_particle_count: int
    lower_raw_log_likelihood: float
    upper_raw_log_likelihood: float
    q2_log_likelihood: float

    def __post_init__(self) -> None:
        if type(self.stream_id) is not int or self.stream_id < 0:
            raise ValueError("stream_id must be a nonnegative integer")
        if (
            type(self.lower_particle_count) is not int
            or self.lower_particle_count <= 0
            or self.upper_particle_count != 2 * self.lower_particle_count
        ):
            raise ValueError("Q2 requires an exact N-to-2N particle pair")
        for name in (
            "lower_raw_log_likelihood",
            "upper_raw_log_likelihood",
            "q2_log_likelihood",
        ):
            _finite_float(getattr(self, name), name)
        expected = (
            2.0 * self.upper_raw_log_likelihood
            - self.lower_raw_log_likelihood
        )
        if self.q2_log_likelihood != expected:
            raise ValueError("Q2 record arithmetic is stale")


def _derive_q2_records(
    raw_records: tuple[LinearGaussianSmcRawRecord, ...],
) -> tuple[LinearGaussianQ2Record, ...]:
    if (
        type(raw_records) is not tuple
        or len(raw_records) < 4
        or any(
            type(record) is not LinearGaussianSmcRawRecord
            for record in raw_records
        )
    ):
        raise ValueError("continuous report requires paired raw records")
    for record in raw_records:
        record.__post_init__()
    ordered = tuple(
        sorted(
            raw_records,
            key=lambda record: (record.stream_id, record.particle_count),
        )
    )
    if ordered != raw_records:
        raise ValueError("raw_records must be ordered by stream and particles")
    by_key: dict[tuple[int, int], float] = {}
    for record in raw_records:
        key = (record.stream_id, record.particle_count)
        if key in by_key:
            raise ValueError("raw_records contain a duplicate stream/particle")
        by_key[key] = record.raw_log_likelihood
    particle_counts = tuple(
        sorted({record.particle_count for record in raw_records})
    )
    if (
        len(particle_counts) != 2
        or particle_counts[1] != 2 * particle_counts[0]
    ):
        raise ValueError("continuous Q2 report requires exactly N and 2N")
    streams = tuple(sorted({record.stream_id for record in raw_records}))
    expected_keys = {
        (stream_id, particle_count)
        for stream_id in streams
        for particle_count in particle_counts
    }
    if len(streams) < 2 or set(by_key) != expected_keys:
        raise ValueError(
            "continuous report requires complete N/2N pairs for at least "
            "two streams"
        )
    lower_particle_count, upper_particle_count = particle_counts
    return tuple(
        LinearGaussianQ2Record(
            stream_id=stream_id,
            lower_particle_count=lower_particle_count,
            upper_particle_count=upper_particle_count,
            lower_raw_log_likelihood=by_key[
                (stream_id, lower_particle_count)
            ],
            upper_raw_log_likelihood=by_key[
                (stream_id, upper_particle_count)
            ],
            q2_log_likelihood=(
                2.0 * by_key[(stream_id, upper_particle_count)]
                - by_key[(stream_id, lower_particle_count)]
            ),
        )
        for stream_id in streams
    )


@dataclass(frozen=True, slots=True)
class LinearGaussianSmcCalibrationReport:
    """Two-sided continuous sensitivity record with an explicit nontransfer."""

    schema_version: Literal["h6-linear-gaussian-smc-sensitivity-v1"]
    fixture_sha256: str
    exact_log_likelihood: float
    raw_records: tuple[LinearGaussianSmcRawRecord, ...]
    q2_records: tuple[LinearGaussianQ2Record, ...]
    q2_mean: float
    q2_two_sided_radius: float
    q2_interval_lower: float
    q2_interval_upper: float
    q2_interval_contains_exact: bool
    q2_remainder_direction: Literal[
        "unknown_without_signed_expansion"
    ]
    q2_bound_kind: Literal["two_sided_sensitivity_envelope"]
    transfer_claim: Literal[
        "sensitivity_control_not_trained_endpoint_bound"
    ]
    report_sha256: str

    @property
    def q2_estimates(self) -> tuple[float, ...]:
        return tuple(record.q2_log_likelihood for record in self.q2_records)

    def __post_init__(self) -> None:
        if self.schema_version != _REPORT_SCHEMA:
            raise ValueError("continuous SMC report schema is frozen")
        if (
            type(self.fixture_sha256) is not str
            or len(self.fixture_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.fixture_sha256
            )
        ):
            raise ValueError("fixture_sha256 must be lowercase SHA-256")
        for name in (
            "exact_log_likelihood",
            "q2_mean",
            "q2_two_sided_radius",
            "q2_interval_lower",
            "q2_interval_upper",
        ):
            _finite_float(getattr(self, name), name)
        if self.q2_two_sided_radius < 0.0:
            raise ValueError("Q2 sensitivity radius must be nonnegative")
        if (
            type(self.q2_records) is not tuple
            or len(self.q2_records) < 2
            or any(
                type(record) is not LinearGaussianQ2Record
                for record in self.q2_records
            )
        ):
            raise ValueError("continuous report requires at least two Q2 streams")
        for record in self.q2_records:
            record.__post_init__()
        expected_q2_records = _derive_q2_records(self.raw_records)
        if self.q2_records != expected_q2_records:
            raise ValueError(
                "q2_records must be exactly derived from raw_records"
            )
        expected_mean = math.fsum(self.q2_estimates) / len(self.q2_records)
        expected_lower = expected_mean - self.q2_two_sided_radius
        expected_upper = expected_mean + self.q2_two_sided_radius
        expected_contains = (
            expected_lower
            <= self.exact_log_likelihood
            <= expected_upper
        )
        if (
            self.q2_mean != expected_mean
            or self.q2_interval_lower != expected_lower
            or self.q2_interval_upper != expected_upper
            or self.q2_interval_contains_exact is not expected_contains
        ):
            raise ValueError("Q2 sensitivity interval fields are inconsistent")
        if (
            self.q2_remainder_direction
            != "unknown_without_signed_expansion"
        ):
            raise ValueError("continuous control cannot sign the Q2 remainder")
        if self.q2_bound_kind != "two_sided_sensitivity_envelope":
            raise ValueError("continuous Q2 sensitivity bound is two-sided")
        if (
            self.transfer_claim
            != "sensitivity_control_not_trained_endpoint_bound"
        ):
            raise ValueError("continuous calibration cannot claim transfer")
        expected_sha256 = _report_sha256(self)
        if self.report_sha256 != expected_sha256:
            raise ValueError("continuous SMC report hash is stale")


def _report_payload(
    report: LinearGaussianSmcCalibrationReport,
) -> dict[str, object]:
    return {
        "schema_version": report.schema_version,
        "fixture_sha256": report.fixture_sha256,
        "exact_log_likelihood": report.exact_log_likelihood.hex(),
        "raw_records": tuple(
            (
                record.stream_id,
                record.particle_count,
                record.raw_log_likelihood.hex(),
            )
            for record in report.raw_records
        ),
        "q2_records": tuple(
            (
                record.stream_id,
                record.lower_particle_count,
                record.upper_particle_count,
                record.lower_raw_log_likelihood.hex(),
                record.upper_raw_log_likelihood.hex(),
                record.q2_log_likelihood.hex(),
            )
            for record in report.q2_records
        ),
        "q2_mean": report.q2_mean.hex(),
        "q2_two_sided_radius": report.q2_two_sided_radius.hex(),
        "q2_interval_lower": report.q2_interval_lower.hex(),
        "q2_interval_upper": report.q2_interval_upper.hex(),
        "q2_interval_contains_exact": report.q2_interval_contains_exact,
        "q2_remainder_direction": report.q2_remainder_direction,
        "q2_bound_kind": report.q2_bound_kind,
        "transfer_claim": report.transfer_claim,
    }


def _report_sha256(report: LinearGaussianSmcCalibrationReport) -> str:
    return _sha256(
        b"vfe4.h6.linear-gaussian-smc-sensitivity.v1",
        _report_payload(report),
    )


def build_continuous_sensitivity_report(
    *,
    fixture: LinearGaussianFixture,
    raw_records: tuple[LinearGaussianSmcRawRecord, ...],
    lower_particle_count: int,
    upper_particle_count: int,
    q2_two_sided_radius: float,
) -> LinearGaussianSmcCalibrationReport:
    """Bind externally generated paired SMC estimates to the exact oracle."""

    if type(fixture) is not LinearGaussianFixture:
        raise ValueError("continuous report requires LinearGaussianFixture")
    fixture.__post_init__()
    if (
        type(lower_particle_count) is not int
        or lower_particle_count <= 0
        or upper_particle_count != 2 * lower_particle_count
    ):
        raise ValueError("continuous Q2 report requires N and 2N")
    _finite_float(q2_two_sided_radius, "q2_two_sided_radius")
    if q2_two_sided_radius < 0.0:
        raise ValueError("q2_two_sided_radius must be nonnegative")
    q2_records = _derive_q2_records(raw_records)
    if (
        q2_records[0].lower_particle_count != lower_particle_count
        or q2_records[0].upper_particle_count != upper_particle_count
    ):
        raise ValueError("declared particle pair does not match raw_records")
    q2_mean = math.fsum(
        record.q2_log_likelihood for record in q2_records
    ) / len(q2_records)
    exact = kalman_log_likelihood(fixture)
    lower = q2_mean - q2_two_sided_radius
    upper = q2_mean + q2_two_sided_radius
    values: dict[str, object] = {
        "schema_version": _REPORT_SCHEMA,
        "fixture_sha256": fixture.fixture_sha256,
        "exact_log_likelihood": exact,
        "raw_records": raw_records,
        "q2_records": q2_records,
        "q2_mean": q2_mean,
        "q2_two_sided_radius": q2_two_sided_radius,
        "q2_interval_lower": lower,
        "q2_interval_upper": upper,
        "q2_interval_contains_exact": lower <= exact <= upper,
        "q2_remainder_direction": "unknown_without_signed_expansion",
        "q2_bound_kind": "two_sided_sensitivity_envelope",
        "transfer_claim": (
            "sensitivity_control_not_trained_endpoint_bound"
        ),
    }
    provisional = object.__new__(LinearGaussianSmcCalibrationReport)
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    report_sha256 = _report_sha256(provisional)
    return LinearGaussianSmcCalibrationReport(
        **values,
        report_sha256=report_sha256,
    )


__all__ = [
    "PINNED_T32_D16_RELATIVE_PATH",
    "PINNED_T32_D16_SHA256",
    "LinearGaussianFixture",
    "LinearGaussianQ2Record",
    "LinearGaussianSmcCalibrationReport",
    "LinearGaussianSmcRawRecord",
    "build_continuous_sensitivity_report",
    "deterministic_linear_gaussian_fixture",
    "fixture_canonical_bytes",
    "kalman_log_likelihood",
    "load_pinned_linear_gaussian_fixture",
]
