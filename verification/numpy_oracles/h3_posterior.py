"""Independent NumPy posterior oracle for the two frozen H3 fixtures."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


_DIMENSION = 4
_FACTOR_COUNT = 6
_EPS = float(np.finfo(np.float64).eps)
_C = 4096.0
_ROOT_FIELDS = {
    "fixture_schema_version",
    "fixture_id",
    "kind",
    "horizon",
    "dimensions",
    "continuous_order",
    "initial",
    "transitions",
    "observation",
    "reference",
}
_DIMENSION_FIELDS = {"d_z", "d_m", "joint_dimension"}
_SECTION_FIELDS = {"factors"}
_OBSERVATION_FIELDS = {"map", "values", "covariance", "factors"}
_FACTOR_FIELDS = {"factor_id", "row", "target", "variance"}
_COUPLED_REFERENCE_FIELDS = {
    "posterior_precision",
    "posterior_natural",
    "log_evidence",
    "analytic_factorized_reverse_kl",
}
_ZERO_REFERENCE_FIELDS = {"posterior_precision"}
_ORDER = ("z0", "m0", "z1", "m1")
_FACTOR_IDS = (
    "z0_prior",
    "m0_prior",
    "m1_transition",
    "z1_transition",
    "z1_observation",
    "m1_observation",
)
_COMMON_INITIAL = (
    ("z0_prior", (1.0, 0.0, 0.0, 0.0), 0.0, 1.0),
    ("m0_prior", (0.0, 1.0, 0.0, 0.0), 0.0, 1.0),
)
_COUPLED_TRANSITIONS = (
    ("m1_transition", (0.0, -0.8, 0.0, 1.0), 0.0, 0.36),
    ("z1_transition", (-0.7, 0.0, 1.0, -0.6), 0.0, 0.25),
)
_ZERO_TRANSITIONS = (
    ("m1_transition", (0.0, 0.0, 0.0, 1.0), 0.0, 0.36),
    ("z1_transition", (0.0, 0.0, 1.0, 0.0), 0.0, 0.25),
)
_OBSERVATION_MAP = (
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)
_OBSERVATION_COVARIANCE = ((0.64, 0.0), (0.0, 0.64))
_COUPLED_OBSERVATIONS = (1.1, 0.2)
_ZERO_OBSERVATIONS = (0.4, -0.7)
_COUPLED_REFERENCE_PRECISION = (
    (2.96, 0.0, -2.8, 1.68),
    (0.0, 2.77777777777778, 0.0, -2.22222222222222),
    (-2.8, 0.0, 5.5625, -2.4),
    (1.68, -2.22222222222222, -2.4, 5.78027777777778),
)
_COUPLED_REFERENCE_NATURAL = (0.0, 0.0, 1.71875, 0.3125)
_COUPLED_REFERENCE_EVIDENCE = -2.6536596233553
_COUPLED_REFERENCE_GAP = 0.6815463199745935
_ZERO_REFERENCE_PRECISION = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 5.5625, 0.0),
    (0.0, 0.0, 0.0, 4.34027777777778),
)


@dataclass(frozen=True)
class H3PosteriorOracleEvaluation:
    fixture_id: str
    precision: np.ndarray = field(compare=False)
    natural: np.ndarray = field(compare=False)
    mean: np.ndarray = field(compare=False)
    covariance: np.ndarray = field(compare=False)
    log_evidence: float
    analytic_factorized_precision: np.ndarray = field(compare=False)
    analytic_factorized_mean: np.ndarray = field(compare=False)
    analytic_factorized_reverse_kl: float
    diagnostics: Mapping[str, object] = field(compare=False)


@dataclass(frozen=True)
class _Factor:
    factor_id: str
    row: tuple[float, float, float, float]
    target: float
    variance: float


@dataclass(frozen=True)
class _ParsedFixture:
    fixture_id: str
    factors: tuple[_Factor, ...]
    reference_precision: tuple[tuple[float, ...], ...]
    reference_natural: tuple[float, ...] | None
    reference_evidence: float | None
    reference_gap: float | None


def evaluate_h3_posterior_oracle(
    data: bytes, *, expected_fixture_id: str
) -> H3PosteriorOracleEvaluation:
    """Parse raw fixture bytes and derive its exact Gaussian posterior."""

    parsed = _parse_fixture(data, expected_fixture_id)
    precision_parts: list[np.ndarray] = []
    natural_parts: list[np.ndarray] = []
    constant_parts: list[float] = []
    for factor in parsed.factors:
        row = np.asarray(factor.row, dtype=np.float64)
        precision_parts.append(np.outer(row, row) / factor.variance)
        natural_parts.append(factor.target * row / factor.variance)
        constant_parts.append(
            -0.5
            * (
                factor.target**2 / factor.variance
                + math.log(2.0 * math.pi * factor.variance)
            )
        )

    precision_stack = np.stack(precision_parts)
    natural_stack = np.stack(natural_parts)
    precision = np.sum(precision_stack, axis=0, dtype=np.float64)
    natural = np.sum(natural_stack, axis=0, dtype=np.float64)
    log_constant = math.fsum(constant_parts)
    if not np.array_equal(precision, precision.T):
        raise ValueError("assembled posterior precision must be exactly symmetric")
    precision_cholesky = _cholesky(precision, "assembled posterior precision")
    mean = np.linalg.solve(precision, natural)
    covariance_raw = np.linalg.solve(
        precision, np.eye(_DIMENSION, dtype=np.float64)
    )
    _require_finite_array(mean, "posterior mean")
    _require_finite_array(covariance_raw, "posterior covariance solve")
    covariance = 0.5 * (covariance_raw + covariance_raw.T)

    evidence_terms = (
        log_constant,
        0.5 * float(natural @ mean),
        0.5 * _DIMENSION * math.log(2.0 * math.pi),
        -float(np.log(np.diag(precision_cholesky)).sum()),
    )
    log_evidence = _finite(math.fsum(evidence_terms), "log evidence")
    precision_diagonal_logs = tuple(
        math.log(float(value)) for value in np.diag(precision)
    )
    cholesky_log_terms = tuple(
        -2.0 * math.log(float(value))
        for value in np.diag(precision_cholesky)
    )
    gap_terms = (*precision_diagonal_logs, *cholesky_log_terms)
    analytic_gap = _finite(
        0.5 * math.fsum(gap_terms), "analytic factorized reverse KL"
    )
    analytic_precision = np.diag(np.diag(precision)).astype(
        np.float64, copy=False
    )

    eigenvalues = np.linalg.eigvalsh(precision)
    _require_finite_array(eigenvalues, "posterior precision eigenvalues")
    if float(eigenvalues[0]) <= 0.0:
        raise ValueError("posterior precision eigenvalues must be positive")
    kappa = float(eigenvalues[-1] / eigenvalues[0])
    precision_absolute = np.sum(np.abs(precision_stack), axis=0)
    natural_absolute = np.sum(np.abs(natural_stack), axis=0)
    log_constant_absolute = math.fsum(abs(value) for value in constant_parts)
    evidence_absolute = math.fsum(
        (
            log_constant_absolute,
            abs(evidence_terms[1]),
            abs(evidence_terms[2]),
            abs(evidence_terms[3]),
        )
    )
    gap_absolute = 0.5 * math.fsum(abs(value) for value in gap_terms)
    diagnostics = MappingProxyType(
        {
            "dimension": _DIMENSION,
            "factor_count": _FACTOR_COUNT,
            "minimum_cholesky_pivot": float(
                np.min(np.diag(precision_cholesky))
            ),
            "lambda_min": float(eigenvalues[0]),
            "lambda_max": float(eigenvalues[-1]),
            "kappa_2": kappa,
            "precision_absolute_summand_accumulation": tuple(
                tuple(float(value) for value in row)
                for row in precision_absolute
            ),
            "natural_absolute_summand_accumulation": tuple(
                float(value) for value in natural_absolute
            ),
            "log_constant": log_constant,
            "log_constant_absolute_summand_accumulation": log_constant_absolute,
            "log_evidence_absolute_summand_accumulation": evidence_absolute,
            "analytic_factorized_reverse_kl_absolute_summand_accumulation": gap_absolute,
            "posterior_solve_residual_infinity_norm": float(
                np.max(
                    np.abs(
                        precision @ covariance_raw
                        - np.eye(_DIMENSION, dtype=np.float64)
                    )
                )
            ),
        }
    )
    _require_reference_agreement(
        parsed,
        precision=precision,
        natural=natural,
        log_evidence=log_evidence,
        analytic_gap=analytic_gap,
        precision_absolute=precision_absolute,
        natural_absolute=natural_absolute,
        evidence_absolute=evidence_absolute,
        gap_absolute=gap_absolute,
        kappa=kappa,
    )

    return H3PosteriorOracleEvaluation(
        fixture_id=parsed.fixture_id,
        precision=_readonly(precision),
        natural=_readonly(natural),
        mean=_readonly(mean),
        covariance=_readonly(covariance),
        log_evidence=log_evidence,
        analytic_factorized_precision=_readonly(analytic_precision),
        analytic_factorized_mean=_readonly(mean),
        analytic_factorized_reverse_kl=analytic_gap,
        diagnostics=diagnostics,
    )


def reverse_kl_to_oracle(
    oracle: H3PosteriorOracleEvaluation,
    *,
    mean: np.ndarray,
    precision: np.ndarray,
) -> float:
    """Evaluate the oriented dense Gaussian ``KL(q || oracle)``."""

    if not isinstance(oracle, H3PosteriorOracleEvaluation):
        raise ValueError("oracle must be an H3PosteriorOracleEvaluation")
    checked_mean = _array(mean, (_DIMENSION,), "mean")
    checked_precision = _array(
        precision, (_DIMENSION, _DIMENSION), "precision"
    )
    if not np.array_equal(checked_precision, checked_precision.T):
        raise ValueError("precision must be exactly symmetric")
    q_cholesky = _cholesky(checked_precision, "precision")
    identity = np.eye(_DIMENSION, dtype=np.float64)
    q_covariance = np.linalg.solve(
        q_cholesky.T, np.linalg.solve(q_cholesky, identity)
    )
    _require_finite_array(q_covariance, "candidate covariance solve")
    q_sign, q_logdet = np.linalg.slogdet(checked_precision)
    p_sign, p_logdet = np.linalg.slogdet(oracle.precision)
    if q_sign != 1.0 or p_sign != 1.0:
        raise ValueError("precision slog-determinants must have positive sign")
    delta = oracle.mean - checked_mean
    parts = (
        float(np.trace(oracle.precision @ q_covariance)),
        float(delta @ oracle.precision @ delta),
        -float(_DIMENSION),
        float(q_logdet),
        -float(p_logdet),
    )
    return _finite(0.5 * math.fsum(parts), "reverse KL")


def _require_reference_agreement(
    parsed: _ParsedFixture,
    *,
    precision: np.ndarray,
    natural: np.ndarray,
    log_evidence: float,
    analytic_gap: float,
    precision_absolute: np.ndarray,
    natural_absolute: np.ndarray,
    evidence_absolute: float,
    gap_absolute: float,
    kappa: float,
) -> None:
    expected_precision = np.asarray(
        parsed.reference_precision, dtype=np.float64
    )
    for row in range(_DIMENSION):
        for column in range(_DIMENSION):
            _require_close(
                float(precision[row, column]),
                float(expected_precision[row, column]),
                float(precision_absolute[row, column]),
                abs(float(expected_precision[row, column])),
                kappa,
                "posterior precision reference",
            )
    if parsed.reference_natural is not None:
        expected_natural = np.asarray(
            parsed.reference_natural, dtype=np.float64
        )
        for index in range(_DIMENSION):
            _require_close(
                float(natural[index]),
                float(expected_natural[index]),
                float(natural_absolute[index]),
                abs(float(expected_natural[index])),
                kappa,
                "posterior natural reference",
            )
    if parsed.reference_evidence is not None:
        _require_close(
            log_evidence,
            parsed.reference_evidence,
            evidence_absolute,
            abs(parsed.reference_evidence),
            kappa,
            "log evidence reference",
        )
    if parsed.reference_gap is not None:
        _require_close(
            analytic_gap,
            parsed.reference_gap,
            gap_absolute,
            abs(parsed.reference_gap),
            kappa,
            "analytic gap reference",
        )
    else:
        _require_close(
            analytic_gap,
            0.0,
            gap_absolute,
            0.0,
            kappa,
            "zero-control analytic gap",
        )


def _require_close(
    actual: float,
    expected: float,
    actual_absolute: float,
    expected_absolute: float,
    kappa: float,
    name: str,
) -> None:
    operation_gamma = _gamma(16 * _DIMENSION + 64)
    actual_allowance = (
        _C
        * operation_gamma
        * max(1.0, kappa)
        * max(1.0, abs(actual), actual_absolute)
    )
    expected_allowance = (
        _C
        * operation_gamma
        * max(1.0, kappa)
        * max(1.0, abs(expected), expected_absolute)
    )
    comparison = _C * _gamma(_DIMENSION + 2) * max(
        1.0, abs(actual), abs(expected), abs(actual) + abs(expected)
    )
    if abs(actual - expected) > actual_allowance + expected_allowance + comparison:
        raise ValueError(f"{name} disagreement")


def _parse_fixture(data: bytes, expected_fixture_id: str) -> _ParsedFixture:
    if type(data) is not bytes:
        raise ValueError("data must be raw bytes")
    if expected_fixture_id not in (
        "h3-coupled-v1",
        "h3-zero-control-v1",
    ):
        raise ValueError("expected_fixture_id must name one frozen H3 fixture")
    try:
        raw = json.loads(
            data.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"fixture JSON could not be parsed: {error}") from error
    root = _fields(raw, _ROOT_FIELDS, "fixture")
    if _integer(root["fixture_schema_version"], "fixture_schema_version") != 1:
        raise ValueError("fixture_schema_version must equal integer one")
    if root["fixture_id"] != expected_fixture_id:
        raise ValueError("fixture_id does not match expected_fixture_id")
    expected_kind = (
        "coupled" if expected_fixture_id == "h3-coupled-v1" else "zero_control"
    )
    if root["kind"] != expected_kind:
        raise ValueError("kind does not match expected fixture")
    if _integer(root["horizon"], "horizon") != 1:
        raise ValueError("horizon must equal integer one")
    dimensions = _fields(root["dimensions"], _DIMENSION_FIELDS, "dimensions")
    if (
        _integer(dimensions["d_z"], "dimensions.d_z"),
        _integer(dimensions["d_m"], "dimensions.d_m"),
        _integer(dimensions["joint_dimension"], "dimensions.joint_dimension"),
    ) != (1, 1, _DIMENSION):
        raise ValueError("fixture dimensions do not match H3")
    order = tuple(
        _string(item, f"continuous_order[{index}]")
        for index, item in enumerate(
            _sequence(root["continuous_order"], _DIMENSION, "continuous_order")
        )
    )
    if order != _ORDER:
        raise ValueError("continuous_order does not match H3")

    initial = _parse_factor_section(root["initial"], "initial", _FACTOR_IDS[:2])
    transitions = _parse_factor_section(
        root["transitions"], "transitions", _FACTOR_IDS[2:4]
    )
    observation = _fields(
        root["observation"], _OBSERVATION_FIELDS, "observation"
    )
    observation_map = _matrix(
        observation["map"], 2, _DIMENSION, "observation.map"
    )
    observation_values = _vector(
        observation["values"], 2, "observation.values"
    )
    observation_covariance = _matrix(
        observation["covariance"], 2, 2, "observation.covariance"
    )
    observation_factors = _parse_factors(
        observation["factors"], "observation.factors", _FACTOR_IDS[4:]
    )
    expected_transitions = (
        _COUPLED_TRANSITIONS
        if expected_fixture_id == "h3-coupled-v1"
        else _ZERO_TRANSITIONS
    )
    expected_observations = (
        _COUPLED_OBSERVATIONS
        if expected_fixture_id == "h3-coupled-v1"
        else _ZERO_OBSERVATIONS
    )
    _require_factor_values(initial, _COMMON_INITIAL, "initial")
    _require_factor_values(transitions, expected_transitions, "transitions")
    if observation_map != _OBSERVATION_MAP:
        raise ValueError("observation map does not match the frozen fixture")
    if observation_covariance != _OBSERVATION_COVARIANCE:
        raise ValueError("observation covariance does not match the frozen fixture")
    if observation_values != expected_observations:
        raise ValueError("observation values do not match the frozen fixture")
    expected_observation_factors = tuple(
        (
            _FACTOR_IDS[index + 4],
            _OBSERVATION_MAP[index],
            expected_observations[index],
            _OBSERVATION_COVARIANCE[index][index],
        )
        for index in range(2)
    )
    _require_factor_values(
        observation_factors,
        expected_observation_factors,
        "observation factors",
    )
    factors = (*initial, *transitions, *observation_factors)
    if len(factors) != _FACTOR_COUNT or len({item.factor_id for item in factors}) != 6:
        raise ValueError("fixture must contain six unique ordered factors")

    reference_fields = (
        _COUPLED_REFERENCE_FIELDS
        if expected_fixture_id == "h3-coupled-v1"
        else _ZERO_REFERENCE_FIELDS
    )
    reference = _fields(root["reference"], reference_fields, "reference")
    reference_precision = _matrix(
        reference["posterior_precision"],
        _DIMENSION,
        _DIMENSION,
        "reference.posterior_precision",
    )
    if expected_fixture_id == "h3-coupled-v1":
        reference_natural = _vector(
            reference["posterior_natural"],
            _DIMENSION,
            "reference.posterior_natural",
        )
        reference_evidence = _number(
            reference["log_evidence"], "reference.log_evidence"
        )
        reference_gap = _number(
            reference["analytic_factorized_reverse_kl"],
            "reference.analytic_factorized_reverse_kl",
        )
        if (
            reference_precision != _COUPLED_REFERENCE_PRECISION
            or reference_natural != _COUPLED_REFERENCE_NATURAL
            or reference_evidence != _COUPLED_REFERENCE_EVIDENCE
            or reference_gap != _COUPLED_REFERENCE_GAP
        ):
            raise ValueError("coupled reference does not match the frozen values")
    else:
        reference_natural = None
        reference_evidence = None
        reference_gap = None
        if reference_precision != _ZERO_REFERENCE_PRECISION:
            raise ValueError("zero-control reference does not match the frozen values")
    return _ParsedFixture(
        fixture_id=expected_fixture_id,
        factors=factors,
        reference_precision=reference_precision,
        reference_natural=reference_natural,
        reference_evidence=reference_evidence,
        reference_gap=reference_gap,
    )


def _parse_factor_section(
    value: object, name: str, expected_ids: tuple[str, ...]
) -> tuple[_Factor, _Factor]:
    section = _fields(value, _SECTION_FIELDS, name)
    return _parse_factors(section["factors"], f"{name}.factors", expected_ids)


def _parse_factors(
    value: object, name: str, expected_ids: tuple[str, ...]
) -> tuple[_Factor, _Factor]:
    records: list[_Factor] = []
    for index, raw in enumerate(_sequence(value, 2, name)):
        factor = _fields(raw, _FACTOR_FIELDS, f"{name}[{index}]")
        factor_id = _string(factor["factor_id"], f"{name}[{index}].factor_id")
        if factor_id != expected_ids[index]:
            raise ValueError(f"{name} factor ID/order mismatch")
        row = _vector(factor["row"], _DIMENSION, f"{name}[{index}].row")
        target = _number(factor["target"], f"{name}[{index}].target")
        variance = _number(factor["variance"], f"{name}[{index}].variance")
        if variance <= 0.0:
            raise ValueError(f"{name}[{index}].variance must be positive")
        records.append(
            _Factor(factor_id, row, target, variance)  # type: ignore[arg-type]
        )
    return records[0], records[1]


def _require_factor_values(
    actual: tuple[_Factor, _Factor],
    expected: tuple[tuple[str, tuple[float, ...], float, float], ...],
    name: str,
) -> None:
    observed = tuple(
        (item.factor_id, item.row, item.target, item.variance) for item in actual
    )
    if observed != expected:
        raise ValueError(f"{name} do not match the frozen factors")


def _array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.float64:
        raise ValueError(f"{name} must be a float64 NumPy array")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    _require_finite_array(value, name)
    return value


def _cholesky(value: np.ndarray, name: str) -> np.ndarray:
    try:
        result = np.linalg.cholesky(value)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{name} must be positive definite") from error
    _require_finite_array(result, f"{name} Cholesky")
    return result


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _require_finite_array(value: np.ndarray, name: str) -> None:
    if not bool(np.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


def _gamma(n: int) -> float:
    numerator = n * _EPS
    if n <= 0 or numerator >= 1.0:
        raise ValueError("rounding operation count is invalid")
    return numerator / (1.0 - numerator)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant {value!r} is forbidden")


def _fields(value: object, expected: set[str], name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"{name} fields must equal {sorted(expected)}")
    return value


def _sequence(value: object, size: int, name: str) -> list[Any]:
    if type(value) is not list or len(value) != size:
        raise ValueError(f"{name} must be a list of length {size}")
    return value


def _vector(value: object, size: int, name: str) -> tuple[float, ...]:
    return tuple(
        _number(item, f"{name}[{index}]")
        for index, item in enumerate(_sequence(value, size, name))
    )


def _matrix(
    value: object, rows: int, columns: int, name: str
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        _vector(row, columns, f"{name}[{index}]")
        for index, row in enumerate(_sequence(value, rows, name))
    )


def _number(value: object, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite numeric data")
    return float(value)


def _integer(value: object, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def _string(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _finite(value: object, name: str) -> float:
    if type(value) is bool or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be finite numeric data")
    checked = float(value)
    if not math.isfinite(checked):
        raise ValueError(f"{name} must be finite numeric data")
    return checked


__all__ = [
    "H3PosteriorOracleEvaluation",
    "evaluate_h3_posterior_oracle",
    "reverse_kl_to_oracle",
]
