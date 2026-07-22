"""Strict raw-byte parsing for the two independently frozen H3 fixtures."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from vfe4.types.h3 import H3Fixture, H3ScalarFactorRecord


H3_COUPLED_FIXTURE_PATH = (
    Path(__file__).with_name("fixtures") / "h3_coupled_v1.json"
)
H3_ZERO_CONTROL_FIXTURE_PATH = (
    Path(__file__).with_name("fixtures") / "h3_zero_control_v1.json"
)

# Measured directly from the frozen raw files; no JSON reserialization is used.
H3_COUPLED_SHA256 = "6779f5b0a2e27aa5e203764bcc4d84c1b1daedb9423fcefdf28dce3cf7e40e03"
H3_ZERO_CONTROL_SHA256 = "ba600e09e0ae7e2b7576fbf4446a8e5b38a605c7621eb0cd5586689dccb89acf"

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
_CONTINUOUS_ORDER = ("z0", "m0", "z1", "m1")
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
_COUPLED_OBSERVATION_VALUES = (1.1, 0.2)
_ZERO_OBSERVATION_VALUES = (0.4, -0.7)
_COUPLED_REFERENCE_PRECISION = (
    (2.96, 0.0, -2.8, 1.68),
    (0.0, 2.77777777777778, 0.0, -2.22222222222222),
    (-2.8, 0.0, 5.5625, -2.4),
    (1.68, -2.22222222222222, -2.4, 5.78027777777778),
)
_COUPLED_REFERENCE_NATURAL = (0.0, 0.0, 1.71875, 0.3125)
_COUPLED_LOG_EVIDENCE = -2.6536596233553
_COUPLED_ANALYTIC_GAP = 0.6815463199745935
_ZERO_REFERENCE_PRECISION = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 5.5625, 0.0),
    (0.0, 0.0, 0.0, 4.34027777777778),
)


def parse_h3_fixture_bytes(
    data: bytes,
    *,
    expected_fixture_id: Literal["h3-coupled-v1", "h3-zero-control-v1"],
) -> H3Fixture:
    """Parse one H3 fixture into owned immutable Python numeric tuples."""
    if type(data) is not bytes:
        raise ValueError("data must be raw bytes")
    if expected_fixture_id not in ("h3-coupled-v1", "h3-zero-control-v1"):
        raise ValueError("expected_fixture_id must name one frozen H3 fixture")
    try:
        text = data.decode("utf-8", errors="strict")
        raw = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_object_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"H3 fixture JSON could not be parsed: {exc}") from exc
    root = _fields(raw, _ROOT_FIELDS, "fixture")
    schema_version = _integer(root["fixture_schema_version"], "fixture_schema_version")
    if schema_version != 1:
        raise ValueError("fixture_schema_version must equal integer 1")
    if root["fixture_id"] != expected_fixture_id:
        raise ValueError("fixture_id does not match expected_fixture_id")
    expected_kind = "coupled" if expected_fixture_id == "h3-coupled-v1" else "zero_control"
    if root["kind"] != expected_kind:
        raise ValueError("kind does not match fixture_id")
    horizon = _integer(root["horizon"], "horizon")
    if horizon != 1:
        raise ValueError("horizon must equal integer 1")

    dimensions = _fields(root["dimensions"], _DIMENSION_FIELDS, "dimensions")
    d_z = _integer(dimensions["d_z"], "dimensions.d_z")
    d_m = _integer(dimensions["d_m"], "dimensions.d_m")
    dimension = _integer(dimensions["joint_dimension"], "dimensions.joint_dimension")
    if (d_z, d_m, dimension) != (1, 1, 4):
        raise ValueError("dimensions must equal d_z=1, d_m=1, joint_dimension=4")
    continuous_order = tuple(
        _string(item, f"continuous_order[{index}]")
        for index, item in enumerate(
            _sequence(root["continuous_order"], 4, "continuous_order")
        )
    )
    if continuous_order != _CONTINUOUS_ORDER:
        raise ValueError("continuous_order must equal [z0,m0,z1,m1]")

    initial = _parse_factor_section(root["initial"], "initial", _FACTOR_IDS[:2])
    transitions = _parse_factor_section(
        root["transitions"], "transitions", _FACTOR_IDS[2:4]
    )
    observation = _fields(root["observation"], _OBSERVATION_FIELDS, "observation")
    observation_map = _matrix(observation["map"], 2, 4, "observation.map")
    observation_values = _vector(observation["values"], 2, "observation.values")
    observation_covariance = _matrix(
        observation["covariance"], 2, 2, "observation.covariance"
    )
    observation_factors = _parse_factors(
        observation["factors"], "observation.factors", _FACTOR_IDS[4:]
    )
    factors = (*initial, *transitions, *observation_factors)
    if len({factor.factor_id for factor in factors}) != 6:
        raise ValueError("all six H3 factor IDs must be unique")

    _require_factor_values(initial, _COMMON_INITIAL, "initial")
    expected_transitions = (
        _COUPLED_TRANSITIONS
        if expected_fixture_id == "h3-coupled-v1"
        else _ZERO_TRANSITIONS
    )
    _require_factor_values(transitions, expected_transitions, "transitions")
    if observation_map != _OBSERVATION_MAP:
        raise ValueError("observation.map must equal the frozen identity selection")
    if observation_covariance != _OBSERVATION_COVARIANCE:
        raise ValueError("observation.covariance must equal 0.64 I2")
    expected_values = (
        _COUPLED_OBSERVATION_VALUES
        if expected_fixture_id == "h3-coupled-v1"
        else _ZERO_OBSERVATION_VALUES
    )
    if observation_values != expected_values:
        raise ValueError("observation.values do not match the frozen fixture")
    expected_observation_factors = tuple(
        (
            _FACTOR_IDS[4 + index],
            _OBSERVATION_MAP[index],
            expected_values[index],
            _OBSERVATION_COVARIANCE[index][index],
        )
        for index in range(2)
    )
    _require_factor_values(
        observation_factors,
        expected_observation_factors,
        "observation.factors",
    )

    reference = _fields(
        root["reference"],
        _COUPLED_REFERENCE_FIELDS
        if expected_fixture_id == "h3-coupled-v1"
        else _ZERO_REFERENCE_FIELDS,
        "reference",
    )
    reference_precision = _matrix(
        reference["posterior_precision"], 4, 4, "reference.posterior_precision"
    )
    if expected_fixture_id == "h3-coupled-v1":
        reference_natural = _vector(
            reference["posterior_natural"], 4, "reference.posterior_natural"
        )
        reference_log_evidence = _number(
            reference["log_evidence"], "reference.log_evidence"
        )
        reference_gap = _number(
            reference["analytic_factorized_reverse_kl"],
            "reference.analytic_factorized_reverse_kl",
        )
        if (
            reference_precision != _COUPLED_REFERENCE_PRECISION
            or reference_natural != _COUPLED_REFERENCE_NATURAL
            or reference_log_evidence != _COUPLED_LOG_EVIDENCE
            or reference_gap != _COUPLED_ANALYTIC_GAP
        ):
            raise ValueError("coupled reference values do not match the frozen contract")
    else:
        reference_natural = None
        reference_log_evidence = None
        reference_gap = None
        if reference_precision != _ZERO_REFERENCE_PRECISION:
            raise ValueError("zero-control posterior precision must equal the frozen diagonal")
        _require_diagonal(reference_precision, "zero-control posterior precision")

    return H3Fixture(
        fixture_schema_version=schema_version,
        fixture_id=expected_fixture_id,
        kind=expected_kind,
        horizon=horizon,
        d_z=d_z,
        d_m=d_m,
        dimension=dimension,
        continuous_order=continuous_order,  # type: ignore[arg-type]
        initial_factors=initial,
        transition_factors=transitions,
        observation_map=observation_map,  # type: ignore[arg-type]
        observation_values=observation_values,  # type: ignore[arg-type]
        observation_covariance=observation_covariance,  # type: ignore[arg-type]
        observation_factors=observation_factors,
        reference_posterior_precision=reference_precision,  # type: ignore[arg-type]
        reference_posterior_natural=reference_natural,  # type: ignore[arg-type]
        reference_log_evidence=reference_log_evidence,
        reference_analytic_factorized_reverse_kl=reference_gap,
    )


def validate_independent_control(coupled: H3Fixture, zero: H3Fixture) -> None:
    """Fail closed unless ``zero`` is the separately authored matched control."""
    if not isinstance(coupled, H3Fixture) or not isinstance(zero, H3Fixture):
        raise ValueError("coupled and zero must be H3Fixture records")
    if (coupled.fixture_id, coupled.kind) != ("h3-coupled-v1", "coupled"):
        raise ValueError("coupled must be the coupled H3 fixture")
    if (zero.fixture_id, zero.kind) != ("h3-zero-control-v1", "zero_control"):
        raise ValueError("zero must be the zero-control H3 fixture")

    common_fields = (
        "fixture_schema_version",
        "horizon",
        "d_z",
        "d_m",
        "dimension",
        "continuous_order",
        "initial_factors",
        "observation_map",
        "observation_covariance",
    )
    for name in common_fields:
        if getattr(coupled, name) != getattr(zero, name):
            raise ValueError(f"control differs in forbidden common field {name}")

    if tuple(factor.factor_id for factor in coupled.factors) != _FACTOR_IDS:
        raise ValueError("coupled factor order does not match the frozen contract")
    if tuple(factor.factor_id for factor in zero.factors) != _FACTOR_IDS:
        raise ValueError("zero-control factor order does not match the frozen contract")
    _require_factor_values(coupled.initial_factors, _COMMON_INITIAL, "coupled.initial")
    _require_factor_values(zero.initial_factors, _COMMON_INITIAL, "zero.initial")
    _require_factor_values(
        coupled.transition_factors,
        _COUPLED_TRANSITIONS,
        "coupled.transitions",
    )
    _require_factor_values(
        zero.transition_factors,
        _ZERO_TRANSITIONS,
        "zero.transitions",
    )
    for coupled_factor, zero_factor in zip(
        coupled.transition_factors, zero.transition_factors, strict=True
    ):
        if (
            coupled_factor.factor_id != zero_factor.factor_id
            or coupled_factor.target != zero_factor.target
            or coupled_factor.variance != zero_factor.variance
        ):
            raise ValueError("control transitions may differ only in coordinate rows")

    if coupled.observation_values != _COUPLED_OBSERVATION_VALUES:
        raise ValueError("coupled observation values do not match the frozen contract")
    if zero.observation_values != _ZERO_OBSERVATION_VALUES:
        raise ValueError("zero observation values do not match the frozen contract")
    for index, (coupled_factor, zero_factor) in enumerate(
        zip(coupled.observation_factors, zero.observation_factors, strict=True)
    ):
        if (
            coupled_factor.factor_id != zero_factor.factor_id
            or coupled_factor.row != zero_factor.row
            or coupled_factor.variance != zero_factor.variance
        ):
            raise ValueError(
                "control observation factors may differ only in observed targets"
            )
        if coupled_factor.target != coupled.observation_values[index]:
            raise ValueError("coupled observation target/value mismatch")
        if zero_factor.target != zero.observation_values[index]:
            raise ValueError("zero observation target/value mismatch")

    # A valid zero control has no initial-coordinate coefficient in either
    # transition and no cross-coordinate coefficient in its frozen posterior.
    if zero.transition_factors[0].row != (0.0, 0.0, 0.0, 1.0):
        raise ValueError("zero m1 transition must have no parent coefficient")
    if zero.transition_factors[1].row != (0.0, 0.0, 1.0, 0.0):
        raise ValueError("zero z1 transition must have no parent coefficient")
    _require_diagonal(
        zero.reference_posterior_precision,
        "zero-control posterior precision",
    )
    if zero.reference_posterior_precision != _ZERO_REFERENCE_PRECISION:
        raise ValueError("zero-control posterior precision does not match the frozen value")


def _parse_factor_section(
    value: object, name: str, expected_ids: tuple[str, ...]
) -> tuple[H3ScalarFactorRecord, H3ScalarFactorRecord]:
    section = _fields(value, _SECTION_FIELDS, name)
    return _parse_factors(section["factors"], f"{name}.factors", expected_ids)


def _parse_factors(
    value: object, name: str, expected_ids: tuple[str, ...]
) -> tuple[H3ScalarFactorRecord, H3ScalarFactorRecord]:
    records: list[H3ScalarFactorRecord] = []
    for index, raw_factor in enumerate(_sequence(value, 2, name)):
        factor = _fields(raw_factor, _FACTOR_FIELDS, f"{name}[{index}]")
        factor_id = _string(factor["factor_id"], f"{name}[{index}].factor_id")
        if factor_id != expected_ids[index]:
            raise ValueError(f"{name} factor IDs/order do not match the frozen contract")
        row = _vector(factor["row"], 4, f"{name}[{index}].row")
        target = _number(factor["target"], f"{name}[{index}].target")
        variance = _number(factor["variance"], f"{name}[{index}].variance")
        if variance <= 0.0:
            raise ValueError(f"{name}[{index}].variance must be positive")
        records.append(
            H3ScalarFactorRecord(
                factor_id=factor_id,
                row=row,  # type: ignore[arg-type]
                target=target,
                variance=variance,
            )
        )
    return (records[0], records[1])


def _require_factor_values(
    actual: tuple[H3ScalarFactorRecord, H3ScalarFactorRecord],
    expected: tuple[tuple[str, tuple[float, ...], float, float], ...],
    name: str,
) -> None:
    observed = tuple(
        (factor.factor_id, factor.row, factor.target, factor.variance)
        for factor in actual
    )
    if observed != expected:
        raise ValueError(f"{name} factors do not match the frozen contract")


def _require_diagonal(matrix: tuple[tuple[float, ...], ...], name: str) -> None:
    for row in range(len(matrix)):
        for column in range(len(matrix[row])):
            if row != column and matrix[row][column] != 0.0:
                raise ValueError(f"{name} must be exactly diagonal")


def _reject_duplicate_object_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant {value!r} is forbidden")


def _fields(value: object, expected: set[str], name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"{name} fields must equal {sorted(expected)}")
    return value


def _sequence(value: object, size: int, name: str) -> list[Any]:
    if type(value) is not list or len(value) != size:
        raise ValueError(f"{name} must be a list of length {size}")
    return value


def _matrix(
    value: object, rows: int, columns: int, name: str
) -> tuple[tuple[float, ...], ...]:
    outer = _sequence(value, rows, name)
    return tuple(
        _vector(row, columns, f"{name}[{index}]")
        for index, row in enumerate(outer)
    )


def _vector(value: object, size: int, name: str) -> tuple[float, ...]:
    return tuple(
        _number(item, f"{name}[{index}]")
        for index, item in enumerate(_sequence(value, size, name))
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


H3_EXPECTED_SHA256_BY_FIXTURE_ID = MappingProxyType(
    {
        "h3-coupled-v1": H3_COUPLED_SHA256,
        "h3-zero-control-v1": H3_ZERO_CONTROL_SHA256,
    }
)

__all__ = [
    "H3_COUPLED_FIXTURE_PATH",
    "H3_COUPLED_SHA256",
    "H3_EXPECTED_SHA256_BY_FIXTURE_ID",
    "H3_ZERO_CONTROL_FIXTURE_PATH",
    "H3_ZERO_CONTROL_SHA256",
    "parse_h3_fixture_bytes",
    "validate_independent_control",
]
