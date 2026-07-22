"""Independent NumPy-only dual-route oracle for canonical H4 problem bytes."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

H4OracleRouteOperationLabel: TypeAlias = Literal[
    "factor_covariance_cholesky", "factor_triangular_solves",
    "factor_assembly_matmuls", "factor_quadratics", "factor_logdet_reductions",
    "factor_J_sum_reduction", "factor_h_sum_reduction",
    "factor_c_scalar_combinations", "factor_c_sum_reduction",
    "posterior_precision_symmetrization",
    "posterior_precision_cholesky", "posterior_natural_solve",
    "posterior_quadratic", "posterior_logdet_reduction",
    "affine_propagation_matmuls", "innovation_assembly", "innovation_cholesky",
    "innovation_triangular_solves", "innovation_quadratics",
    "innovation_logdet_reductions", "kalman_gain_solves", "mean_updates",
    "covariance_updates", "route_sum_reduction",
]

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_EPSILON = 2.220446049250313e-16
_ROUNDING_CONSTANT = 4096
_MAXIMUM_ALLOWANCE_SCALE_FRACTION = 1.0e-4
_CANONICAL_LABELS = (
    "factor_covariance_cholesky", "factor_triangular_solves",
    "factor_assembly_matmuls", "factor_quadratics", "factor_logdet_reductions",
    "factor_J_sum_reduction", "factor_h_sum_reduction",
    "factor_c_scalar_combinations", "factor_c_sum_reduction",
    "posterior_precision_symmetrization",
    "posterior_precision_cholesky", "posterior_natural_solve",
    "posterior_quadratic", "posterior_logdet_reduction", "route_sum_reduction",
)
_PREDICTIVE_LABELS = (
    "affine_propagation_matmuls", "innovation_assembly", "innovation_cholesky",
    "innovation_triangular_solves", "innovation_quadratics",
    "innovation_logdet_reductions", "kalman_gain_solves", "mean_updates",
    "covariance_updates", "route_sum_reduction",
)


def _finite(value: object, name: str) -> None:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be an exact finite float")


def _gamma(n: int) -> float:
    if type(n) is not int or n < 0 or n * _EPSILON >= 1.0:
        raise ValueError("route operation count is outside the gamma domain")
    return (n * _EPSILON) / (1.0 - n * _EPSILON)


def _dot(k: int) -> int:
    return max(0, 2 * k - 1)


def _matmul(m: int, k: int, n: int) -> int:
    return m * n * _dot(k)


def _triangular(n: int, rhs: int) -> int:
    return rhs * n * n


def _cholesky(n: int) -> int:
    return math.ceil(n * n * n / 3)


@dataclass(frozen=True, slots=True)
class H4OracleSelectedMoment:
    name: str
    coordinate_indices: tuple[int, ...]
    mean: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name or type(self.coordinate_indices) is not tuple or not self.coordinate_indices or any(type(item) is not int or item < 0 for item in self.coordinate_indices) or any(left >= right for left, right in zip(self.coordinate_indices, self.coordinate_indices[1:], strict=False)):
            raise ValueError("oracle selected-moment identity is invalid")
        size = len(self.coordinate_indices)
        if type(self.mean) is not tuple or len(self.mean) != size or type(self.covariance) is not tuple or len(self.covariance) != size or any(type(row) is not tuple or len(row) != size for row in self.covariance):
            raise ValueError("oracle selected-moment shapes are invalid")
        for value in (*self.mean, *(item for row in self.covariance for item in row)):
            _finite(value, "selected moment value")


@dataclass(frozen=True, slots=True)
class H4OracleOperandEvidence:
    path: str
    value: float
    value_norm: float
    absolute_summand_accumulation: float
    condition_numbers: tuple[float, ...]
    operation_counts: tuple[tuple[H4OracleRouteOperationLabel, int], ...]

    def __post_init__(self) -> None:
        if type(self.path) is not str or not self.path:
            raise ValueError("oracle operand path must be nonempty")
        for name in ("value", "value_norm", "absolute_summand_accumulation"):
            _finite(getattr(self, name), name)
        if self.value_norm != abs(self.value) or self.absolute_summand_accumulation < 0.0:
            raise ValueError("oracle route scalar norm/accumulation is inconsistent")
        if type(self.condition_numbers) is not tuple or not self.condition_numbers or any(type(item) is not float or not math.isfinite(item) or item <= 0.0 for item in self.condition_numbers):
            raise ValueError("oracle route condition tuple is invalid")
        if type(self.operation_counts) is not tuple or any(type(item) is not tuple or len(item) != 2 or type(item[0]) is not str or type(item[1]) is not int or item[1] < 0 for item in self.operation_counts):
            raise ValueError("oracle route operation table is invalid")
        labels = tuple(item[0] for item in self.operation_counts)
        expected = _CANONICAL_LABELS if self.path == "canonical_log_normalizer" else _PREDICTIVE_LABELS if self.path == "predictive_log_normalizer" else None
        if expected is None or labels != expected:
            raise ValueError("oracle route operation labels/order are frozen")


@dataclass(frozen=True, slots=True)
class H4OraclePosteriorDiagnostic:
    dimension: int
    minimum_eigenvalue: float
    maximum_eigenvalue: float
    condition_number: float
    minimum_cholesky_pivot: float
    mean_infinity_norm: float

    def __post_init__(self) -> None:
        if type(self.dimension) is not int or self.dimension <= 0:
            raise ValueError("oracle posterior dimension must be positive")
        for name in tuple(self.__dataclass_fields__)[1:]:
            _finite(getattr(self, name), name)
        if self.minimum_eigenvalue <= 0.0 or self.maximum_eigenvalue < self.minimum_eigenvalue or self.condition_number < 1.0 or self.minimum_cholesky_pivot <= 0.0 or self.mean_infinity_norm < 0.0:
            raise ValueError("oracle posterior diagnostic is not finite SPD")


@dataclass(frozen=True, slots=True)
class H4OracleInnovationDiagnostic:
    factor_id: str
    time_index: int
    parent_coordinate_indices: tuple[int, ...]
    innovation_dimension: int
    minimum_eigenvalue: float
    maximum_eigenvalue: float
    condition_number: float

    def __post_init__(self) -> None:
        if type(self.factor_id) is not str or not self.factor_id or type(self.time_index) is not int or self.time_index <= 0 or type(self.parent_coordinate_indices) is not tuple or not self.parent_coordinate_indices or any(type(item) is not int or item < 0 for item in self.parent_coordinate_indices) or any(left >= right for left, right in zip(self.parent_coordinate_indices, self.parent_coordinate_indices[1:], strict=False)) or type(self.innovation_dimension) is not int or self.innovation_dimension <= 0:
            raise ValueError("oracle innovation identity is invalid")
        for name in ("minimum_eigenvalue", "maximum_eigenvalue", "condition_number"):
            _finite(getattr(self, name), name)
        if self.minimum_eigenvalue <= 0.0 or self.maximum_eigenvalue < self.minimum_eigenvalue or self.condition_number < 1.0:
            raise ValueError("oracle innovation is not finite SPD")


@dataclass(frozen=True, slots=True)
class H4OracleRouteAgreement:
    problem_id: str
    problem_sha256: str
    canonical_operand: H4OracleOperandEvidence
    predictive_operand: H4OracleOperandEvidence
    float64_epsilon: float
    rounding_constant: Literal[4096]
    solver_allowance: Literal[0.0]
    maximum_allowance_scale_fraction: float
    invariant_scale: float
    canonical_rounding_allowance: float
    predictive_rounding_allowance: float
    comparison_reduction_allowance: float
    residual: float
    normalized_residual: float
    final_allowance: float
    allowance_scale_ratio: float
    decisiveness_rule: Literal["allowance_scale_ratio_strictly_less_than_1e-4"]
    pass_rule: Literal["residual_less_than_or_equal_to_final_allowance"]
    decisive: bool
    passed: bool
    eligible: bool

    def __post_init__(self) -> None:
        if type(self.problem_id) is not str or not self.problem_id or type(self.problem_sha256) is not str or _SHA256.fullmatch(self.problem_sha256) is None:
            raise ValueError("oracle route agreement identity is invalid")
        if type(self.canonical_operand) is not H4OracleOperandEvidence or type(self.predictive_operand) is not H4OracleOperandEvidence or self.canonical_operand.path != "canonical_log_normalizer" or self.predictive_operand.path != "predictive_log_normalizer":
            raise ValueError("oracle route agreement must own the two ordered operands")
        if (self.float64_epsilon, self.rounding_constant, self.solver_allowance, self.maximum_allowance_scale_fraction) != (_EPSILON, 4096, 0.0, 1.0e-4):
            raise ValueError("oracle route allowance policy is frozen")
        canonical_count = sum(item[1] for item in self.canonical_operand.operation_counts)
        predictive_count = sum(item[1] for item in self.predictive_operand.operation_counts)
        invariant_scale = max(1.0, abs(self.canonical_operand.value), abs(self.predictive_operand.value))
        canonical_rounding = 4096.0 * _gamma(canonical_count) * max((1.0, *self.canonical_operand.condition_numbers)) * max(1.0, abs(self.canonical_operand.value), self.canonical_operand.absolute_summand_accumulation)
        predictive_rounding = 4096.0 * _gamma(predictive_count) * max((1.0, *self.predictive_operand.condition_numbers)) * max(1.0, abs(self.predictive_operand.value), self.predictive_operand.absolute_summand_accumulation)
        comparison = 4096.0 * _gamma(3) * max(1.0, abs(self.canonical_operand.value), abs(self.predictive_operand.value), abs(self.canonical_operand.value) + abs(self.predictive_operand.value))
        residual = abs(self.canonical_operand.value - self.predictive_operand.value)
        final = canonical_rounding + predictive_rounding + comparison
        ratio = final / invariant_scale
        normalized = residual / final
        decisive = ratio < 1.0e-4
        passed = residual <= final
        expected = (invariant_scale, canonical_rounding, predictive_rounding, comparison, residual, normalized, final, ratio, decisive, passed, decisive and passed)
        actual = (self.invariant_scale, self.canonical_rounding_allowance, self.predictive_rounding_allowance, self.comparison_reduction_allowance, self.residual, self.normalized_residual, self.final_allowance, self.allowance_scale_ratio, self.decisive, self.passed, self.eligible)
        if actual != expected or self.decisiveness_rule != "allowance_scale_ratio_strictly_less_than_1e-4" or self.pass_rule != "residual_less_than_or_equal_to_final_allowance":
            raise ValueError("oracle route agreement arithmetic is inconsistent")


@dataclass(frozen=True, slots=True)
class H4OracleEvaluation:
    schema_version: Literal["h4-numpy-oracle-v1"]
    problem_id: str
    problem_sha256: str
    source_kind: Literal["scaled_pcg64", "h3_anchor"]
    seed: int
    kind: Literal["coupled", "zero_control"]
    horizon: int
    d_z: int
    d_m: int
    dimension: int
    coordinate_order: tuple[str, ...]
    factor_ids: tuple[str, ...]
    precision: tuple[tuple[float, ...], ...]
    natural: tuple[float, ...]
    constant: float
    mean: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    canonical_log_normalizer: float
    predictive_log_normalizer: float
    route_agreement: H4OracleRouteAgreement
    selected_moments: tuple[H4OracleSelectedMoment, ...]
    posterior_diagnostic: H4OraclePosteriorDiagnostic
    innovation_diagnostics: tuple[H4OracleInnovationDiagnostic, ...]
    operand_evidence: tuple[H4OracleOperandEvidence, ...]

    def __post_init__(self) -> None:
        if self.schema_version != "h4-numpy-oracle-v1" or self.source_kind not in ("scaled_pcg64", "h3_anchor") or self.kind not in ("coupled", "zero_control"):
            raise ValueError("oracle evaluation schema/source is invalid")
        if type(self.route_agreement) is not H4OracleRouteAgreement or self.route_agreement.problem_id != self.problem_id or self.route_agreement.problem_sha256 != self.problem_sha256:
            raise ValueError("oracle route agreement provenance mismatch")
        if self.canonical_log_normalizer != self.route_agreement.canonical_operand.value or self.predictive_log_normalizer != self.route_agreement.predictive_operand.value or self.operand_evidence != (self.route_agreement.canonical_operand, self.route_agreement.predictive_operand):
            raise ValueError("oracle operands must be owned exactly once in route order")
        if type(self.selected_moments) is not tuple or not all(type(item) is H4OracleSelectedMoment for item in self.selected_moments) or type(self.posterior_diagnostic) is not H4OraclePosteriorDiagnostic or type(self.innovation_diagnostics) is not tuple or not all(type(item) is H4OracleInnovationDiagnostic for item in self.innovation_diagnostics):
            raise ValueError("oracle diagnostic records must be exact immutable tuples")


@dataclass(frozen=True, slots=True)
class H4OracleKLEvaluation:
    value: float
    trace_term: float
    quadratic_mean_term: float
    minus_dimension_term: float
    candidate_logdet_precision_term: float
    minus_oracle_logdet_precision_term: float
    absolute_summand_accumulation: float
    candidate_condition_number: float
    oracle_condition_number: float
    operation_counts: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        for name in tuple(self.__dataclass_fields__)[:-1]:
            _finite(getattr(self, name), name)
        if self.absolute_summand_accumulation < 0.0 or self.candidate_condition_number < 1.0 or self.oracle_condition_number < 1.0 or type(self.operation_counts) is not tuple or any(type(item) is not tuple or len(item) != 2 or type(item[0]) is not str or type(item[1]) is not int or item[1] < 0 for item in self.operation_counts):
            raise ValueError("oracle KL evidence is malformed")


@dataclass(frozen=True, slots=True)
class _RawDraw:
    draw_index: int
    name: str
    shape: tuple[int, ...]
    values: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class _Factor:
    factor_id: str
    role: str
    time_index: int
    normalized: tuple[int, ...]
    parents: tuple[int, ...]
    matrix: NDArray[np.float64]
    target: NDArray[np.float64]
    covariance: NDArray[np.float64]
    raw_draws: tuple[_RawDraw, ...]


def _duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse(payload: bytes) -> tuple[dict[str, object], tuple[_Factor, ...], str]:
    if type(payload) is not bytes:
        raise ValueError("H4 oracle consumes exact canonical bytes")
    try:
        envelope = json.loads(payload.decode("utf-8"), object_pairs_hook=_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid H4 neutral-problem JSON") from error
    if type(envelope) is not dict or set(envelope) != {"schema_version", "canonical_sha256", "problem"} or envelope.get("schema_version") != "h4-neutral-problem-v1":
        raise ValueError("wrong H4 neutral-problem envelope schema")
    digest = envelope["canonical_sha256"]
    core = envelope["problem"]
    if type(digest) is not str or _SHA256.fullmatch(digest) is None or type(core) is not dict:
        raise ValueError("invalid H4 core digest/record")
    canonical_core = json.dumps(core, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    expected_digest = hashlib.sha256(b"vfe4.h4.neutral-problem.v1\x00" + canonical_core).hexdigest()
    if digest != expected_digest:
        raise ValueError("H4 core digest mismatch")
    core_keys = {"problem_id", "source_kind", "seed", "kind", "horizon", "d_z", "d_m", "dimension", "coordinate_order", "factor_schedule"}
    if set(core) != core_keys:
        raise ValueError("H4 problem core has unknown or missing keys")
    for name in ("problem_id", "source_kind", "kind"):
        if type(core[name]) is not str:
            raise ValueError(f"H4 core {name} has wrong type")
    for name in ("seed", "horizon", "d_z", "d_m", "dimension"):
        if type(core[name]) is not int:
            raise ValueError(f"H4 core {name} has wrong type")
    coordinate_order = core["coordinate_order"]
    factor_schedule = core["factor_schedule"]
    if type(coordinate_order) is not list or len(coordinate_order) != core["dimension"] or not all(type(item) is str and item for item in coordinate_order) or type(factor_schedule) is not list or not factor_schedule:
        raise ValueError("H4 coordinate/factor schedule is malformed")
    factors = tuple(_parse_factor(item, int(core["dimension"])) for item in factor_schedule)
    _validate_source(core, factors)
    return core, factors, digest


def _parse_factor(raw: object, dimension: int) -> _Factor:
    keys = {"factor_id", "role", "time_index", "normalized_coordinate_indices", "parent_coordinate_indices", "matrix", "target", "covariance", "raw_draws"}
    if type(raw) is not dict or set(raw) != keys:
        raise ValueError("H4 factor has unknown or missing keys")
    factor_id, role, time_index = raw["factor_id"], raw["role"], raw["time_index"]
    if type(factor_id) is not str or not factor_id or role not in ("initial", "transition", "observation") or type(time_index) is not int or time_index < 0:
        raise ValueError("H4 factor identity is malformed")
    normalized = _indices(raw["normalized_coordinate_indices"], dimension)
    parents = _indices(raw["parent_coordinate_indices"], dimension)
    if set(normalized) & set(parents):
        raise ValueError("H4 factor normalized/parent metadata overlap")
    matrix = _array2(raw["matrix"], "factor matrix")
    target = _array1(raw["target"], "factor target")
    covariance = _array2(raw["covariance"], "factor covariance")
    if matrix.shape != (target.size, dimension) or covariance.shape != (target.size, target.size) or not np.array_equal(covariance, covariance.T):
        raise ValueError("H4 factor shapes/symmetry are invalid")
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError("H4 factor covariance is not SPD") from error
    if role in ("initial", "transition"):
        if len(normalized) != target.size or not np.array_equal(matrix[:, normalized], np.eye(target.size, dtype=np.float64)):
            raise ValueError("H4 normalized factor columns must be identity")
    elif normalized:
        raise ValueError("H4 observation cannot normalize coordinates")
    allowed = set(normalized) | set(parents)
    if any(matrix[row, column] != 0.0 for row in range(matrix.shape[0]) for column in range(dimension) if column not in allowed):
        raise ValueError("H4 factor has noncausal matrix support")
    raw_draw_records = raw["raw_draws"]
    if type(raw_draw_records) is not list:
        raise ValueError("H4 raw-draw provenance must be a list")
    raw_draws = tuple(_parse_raw_draw(draw) for draw in raw_draw_records)
    draw_indices = tuple(draw.draw_index for draw in raw_draws)
    if any(
        left >= right
        for left, right in zip(draw_indices, draw_indices[1:], strict=False)
    ):
        raise ValueError("H4 raw draws must be strictly increasing by draw index")
    if len({draw.name for draw in raw_draws}) != len(raw_draws):
        raise ValueError("H4 raw draw names must be unique within a factor")
    return _Factor(
        factor_id, role, time_index, normalized, parents, matrix, target,
        covariance, raw_draws,
    )


def _parse_raw_draw(raw: object) -> _RawDraw:
    keys = {"draw_index", "name", "shape", "values"}
    if type(raw) is not dict or set(raw) != keys:
        raise ValueError("H4 raw draw has unknown or missing keys")
    draw_index, name, shape, values = (
        raw["draw_index"], raw["name"], raw["shape"], raw["values"],
    )
    if type(draw_index) is not int or draw_index < 0:
        raise ValueError("H4 raw draw index must be a nonnegative integer")
    if type(name) is not str or not name:
        raise ValueError("H4 raw draw name must be nonempty")
    if type(shape) is not list or any(
        type(item) is not int or item < 0 for item in shape
    ):
        raise ValueError("H4 raw draw shape must contain nonnegative integers")
    expected_values = math.prod(shape) if shape else 1
    if (
        type(values) is not list
        or len(values) != expected_values
        or any(
            type(item) not in (int, float) or not math.isfinite(float(item))
            for item in values
        )
    ):
        raise ValueError("H4 raw draw values must be finite row-major float64 data")
    return _RawDraw(
        draw_index, name, tuple(shape), tuple(float(item) for item in values),
    )


def _indices(raw: object, dimension: int) -> tuple[int, ...]:
    if type(raw) is not list or any(type(item) is not int or item < 0 or item >= dimension for item in raw) or any(left >= right for left, right in zip(raw, raw[1:], strict=False)):
        raise ValueError("H4 factor indices must be strictly ascending and in range")
    return tuple(raw)


def _array1(raw: object, name: str) -> NDArray[np.float64]:
    if type(raw) is not list or not raw or any(type(item) not in (int, float) or not math.isfinite(float(item)) for item in raw):
        raise ValueError(f"{name} must be a nonempty finite vector")
    return np.asarray(raw, dtype=np.float64)


def _array2(raw: object, name: str) -> NDArray[np.float64]:
    if type(raw) is not list or not raw or not all(type(row) is list and row for row in raw) or len({len(row) for row in raw}) != 1 or any(type(item) not in (int, float) or not math.isfinite(float(item)) for row in raw for item in row):
        raise ValueError(f"{name} must be a nonempty finite matrix")
    return np.asarray(raw, dtype=np.float64)


def _validate_source(core: dict[str, object], factors: tuple[_Factor, ...]) -> None:
    source, kind = core["source_kind"], core["kind"]
    horizon, dimension = int(core["horizon"]), int(core["dimension"])
    ids = tuple(item.factor_id for item in factors)
    if len(ids) != len(set(ids)):
        raise ValueError("H4 factor IDs must be unique")
    if source == "scaled_pcg64":
        if type(core["seed"]) is not int or int(core["seed"]) <= 0 or kind not in ("coupled", "zero_control") or horizon not in (7, 15, 31) or (core["d_z"], core["d_m"]) != (4, 4) or dimension != (horizon + 1) * 8:
            raise ValueError("unsupported scaled H4 identity")
        expected_id = f"h4-{kind}-T{horizon}-dz4-dm4-seed{core['seed']}-v1"
        expected_ids = ("initial_joint", *(item for time in range(1, horizon + 1) for item in (f"m_transition[{time}]", f"z_transition[{time}]", f"observation[{time}]")))
        expected_coordinates = tuple(
            f"{prefix}[{time},{index}]"
            for time in range(horizon + 1)
            for prefix in ("z", "m")
            for index in range(4)
        )
        if (
            core["problem_id"] != expected_id
            or tuple(core["coordinate_order"]) != expected_coordinates
            or ids != expected_ids
        ):
            raise ValueError("scaled H4 problem ID/schedule is frozen")
        _validate_scaled_source(
            seed=int(core["seed"]), kind=str(kind), horizon=horizon,
            dimension=dimension, factors=factors,
        )
    elif source == "h3_anchor":
        if core["seed"] != 0 or horizon != 1 or (core["d_z"], core["d_m"], dimension) != (1, 1, 4) or kind not in ("coupled", "zero_control") or core["problem_id"] != f"h4-anchor-h3-{kind if kind == 'coupled' else 'zero-control'}-v1":
            raise ValueError("unsupported H3 anchor identity")
        if tuple(core["coordinate_order"]) != ("z0", "m0", "z1", "m1"):
            raise ValueError("H3 anchor coordinate order is frozen")
        _validate_h3_source(str(kind), factors)
    else:
        raise ValueError("unsupported H4 source_kind")


def _raw_draw(index: int, name: str, value: NDArray[np.float64]) -> _RawDraw:
    return _RawDraw(
        index, name, tuple(value.shape),
        tuple(float(item) for item in value.reshape(-1)),
    )


def _source_block(time: int, component: Literal["z", "m"] = "z") -> tuple[int, ...]:
    start = time * 8 + (4 if component == "m" else 0)
    return tuple(range(start, start + 4))


def _source_clip(value: NDArray[np.float64]) -> NDArray[np.float64]:
    return value * min(1.0, 0.65 / float(np.linalg.norm(value, 2)))


def _require_exact_factor(
    factor: _Factor,
    *,
    role: str,
    time_index: int,
    normalized: tuple[int, ...],
    parents: tuple[int, ...],
    matrix: NDArray[np.float64],
    target: NDArray[np.float64],
    covariance: NDArray[np.float64],
    raw_draws: tuple[_RawDraw, ...],
    source_label: str,
) -> None:
    if (
        (factor.role, factor.time_index, factor.normalized, factor.parents)
        != (role, time_index, normalized, parents)
        or factor.raw_draws != raw_draws
        or not np.array_equal(factor.matrix, matrix)
        or not np.array_equal(factor.target, target)
        or not np.array_equal(factor.covariance, covariance)
    ):
        raise ValueError(f"{source_label} factor/source identity is frozen")


def _validate_scaled_source(
    *,
    seed: int,
    kind: str,
    horizon: int,
    dimension: int,
    factors: tuple[_Factor, ...],
) -> None:
    initial_matrix = np.zeros((8, dimension), dtype=np.float64)
    initial_matrix[:, :8] = np.eye(8, dtype=np.float64)
    _require_exact_factor(
        factors[0], role="initial", time_index=0, normalized=tuple(range(8)),
        parents=(), matrix=initial_matrix, target=np.zeros(8, dtype=np.float64),
        covariance=np.eye(8, dtype=np.float64), raw_draws=(),
        source_label="scaled initial",
    )

    rng = np.random.Generator(np.random.PCG64(seed))
    for time in range(1, horizon + 1):
        raw_a_m = rng.standard_normal((4, 4))
        raw_a_z = rng.standard_normal((4, 4))
        raw_b = rng.standard_normal((4, 4))
        c_m = rng.uniform(-0.25, 0.25, size=4)
        c_z = rng.uniform(-0.25, 0.25, size=4)
        r_m = rng.uniform(0.5, 1.5, size=4)
        r_z = rng.uniform(0.5, 1.5, size=4)
        raw_g = rng.standard_normal((8, 8))
        offset = rng.uniform(-0.25, 0.25, size=8)
        observation_noise = rng.uniform(0.75, 1.25, size=8)
        observed_target = rng.uniform(-1.0, 1.0, size=8)

        active_a_m = _source_clip(raw_a_m)
        joined = _source_clip(np.concatenate((raw_a_z, raw_b), axis=1))
        active_a_z, active_b = joined[:, :4], joined[:, 4:]
        if kind == "zero_control":
            active_a_m = np.zeros_like(active_a_m)
            active_a_z = np.zeros_like(active_a_z)
            active_b = np.zeros_like(active_b)

        z_previous = _source_block(time - 1)
        m_previous = _source_block(time - 1, "m")
        z_current = _source_block(time)
        m_current = _source_block(time, "m")
        m_matrix = np.zeros((4, dimension), dtype=np.float64)
        m_matrix[:, m_current] = np.eye(4, dtype=np.float64)
        m_matrix[:, m_previous] = -active_a_m
        z_matrix = np.zeros((4, dimension), dtype=np.float64)
        z_matrix[:, z_current] = np.eye(4, dtype=np.float64)
        z_matrix[:, z_previous] = -active_a_z
        z_matrix[:, m_current] = -active_b
        observation_local = (
            np.eye(8, dtype=np.float64)
            + 0.05 * raw_g / max(1.0, float(np.linalg.norm(raw_g, 2)))
        )
        observation_matrix = np.zeros((8, dimension), dtype=np.float64)
        observation_matrix[:, (*z_current, *m_current)] = observation_local
        base = 11 * (time - 1)
        m_draws = (
            _raw_draw(base, f"A_m[{time}]", raw_a_m),
            _raw_draw(base + 3, f"c_m[{time}]", c_m),
            _raw_draw(base + 5, f"R_m[{time}]", r_m),
        )
        z_draws = (
            _raw_draw(base + 1, f"A_z[{time}]", raw_a_z),
            _raw_draw(base + 2, f"B[{time}]", raw_b),
            _raw_draw(base + 4, f"c_z[{time}]", c_z),
            _raw_draw(base + 6, f"R_z[{time}]", r_z),
        )
        observation_draws = (
            _raw_draw(base + 7, f"G[{time}]", raw_g),
            _raw_draw(base + 8, f"observation_offset[{time}]", offset),
            _raw_draw(
                base + 9, f"observation_noise[{time}]", observation_noise,
            ),
            _raw_draw(base + 10, f"observed_target[{time}]", observed_target),
        )
        m_factor, z_factor, observation_factor = factors[
            1 + 3 * (time - 1):1 + 3 * time
        ]
        _require_exact_factor(
            m_factor, role="transition", time_index=time, normalized=m_current,
            parents=m_previous, matrix=m_matrix, target=c_m,
            covariance=np.diag(r_m), raw_draws=m_draws,
            source_label=f"scaled m_transition[{time}]",
        )
        _require_exact_factor(
            z_factor, role="transition", time_index=time, normalized=z_current,
            parents=(*z_previous, *m_current), matrix=z_matrix, target=c_z,
            covariance=np.diag(r_z), raw_draws=z_draws,
            source_label=f"scaled z_transition[{time}]",
        )
        _require_exact_factor(
            observation_factor, role="observation", time_index=time,
            normalized=(), parents=(*z_current, *m_current),
            matrix=observation_matrix, target=observed_target - offset,
            covariance=np.diag(observation_noise), raw_draws=observation_draws,
            source_label=f"scaled observation[{time}]",
        )


def _validate_h3_source(kind: str, factors: tuple[_Factor, ...]) -> None:
    expected_ids = (
        "z0_prior", "m0_prior", "m1_transition", "z1_transition",
        "z1_observation", "m1_observation",
    )
    if tuple(factor.factor_id for factor in factors) != expected_ids:
        raise ValueError("H3 anchor factor IDs/order are frozen")
    transition_rows = (
        ((0.0, -0.8, 0.0, 1.0), (-0.7, 0.0, 1.0, -0.6))
        if kind == "coupled"
        else ((0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0, 0.0))
    )
    observation_targets = (1.1, 0.2) if kind == "coupled" else (0.4, -0.7)
    expected = (
        ("initial", 0, (0,), (), (1.0, 0.0, 0.0, 0.0), 0.0, 1.0),
        ("initial", 0, (1,), (), (0.0, 1.0, 0.0, 0.0), 0.0, 1.0),
        ("transition", 1, (3,), (1,), transition_rows[0], 0.0, 0.36),
        ("transition", 1, (2,), (0, 3), transition_rows[1], 0.0, 0.25),
        ("observation", 1, (), (2,), (0.0, 0.0, 1.0, 0.0), observation_targets[0], 0.64),
        ("observation", 1, (), (3,), (0.0, 0.0, 0.0, 1.0), observation_targets[1], 0.64),
    )
    for factor, item in zip(factors, expected, strict=True):
        role, time_index, normalized, parents, row, target, variance = item
        _require_exact_factor(
            factor, role=role, time_index=time_index, normalized=normalized,
            parents=parents, matrix=np.asarray((row,), dtype=np.float64),
            target=np.asarray((target,), dtype=np.float64),
            covariance=np.asarray(((variance,),), dtype=np.float64), raw_draws=(),
            source_label=f"H3 {factor.factor_id}",
        )


def _solve_spd(matrix: NDArray[np.float64], rhs: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    lower = np.linalg.cholesky(matrix)
    return np.linalg.solve(lower.T, np.linalg.solve(lower, rhs)), lower


def _canonical_route(factors: tuple[_Factor, ...], dimension: int) -> tuple[NDArray[np.float64], NDArray[np.float64], float, NDArray[np.float64], NDArray[np.float64], float, H4OracleOperandEvidence]:
    precision = np.zeros((dimension, dimension), dtype=np.float64)
    natural = np.zeros(dimension, dtype=np.float64)
    constant = 0.0
    conditions: list[float] = []
    counts = {label: 0 for label in _CANONICAL_LABELS}
    for factor in factors:
        rows = factor.target.size
        solved_matrix, lower = _solve_spd(factor.covariance, factor.matrix)
        solved_target = np.linalg.solve(lower.T, np.linalg.solve(lower, factor.target))
        precision += factor.matrix.T @ solved_matrix
        natural += factor.matrix.T @ solved_target
        quadratic = float(factor.target @ solved_target)
        logdet = 2.0 * float(np.sum(np.log(np.diag(lower)), dtype=np.float64))
        constant += -0.5 * (quadratic + rows * math.log(2.0 * math.pi) + logdet)
        conditions.append(float(np.linalg.cond(factor.covariance)))
        counts["factor_covariance_cholesky"] += _cholesky(rows)
        counts["factor_triangular_solves"] += 2 * _triangular(rows, dimension + 1)
        counts["factor_assembly_matmuls"] += _matmul(dimension, rows, dimension) + _matmul(dimension, rows, 1)
        counts["factor_quadratics"] += _dot(rows)
        counts["factor_logdet_reductions"] += rows + max(0, rows - 1)
    factor_count = len(factors)
    counts["factor_J_sum_reduction"] = factor_count * dimension * dimension
    counts["factor_h_sum_reduction"] = factor_count * dimension
    counts["factor_c_scalar_combinations"] = 4 * factor_count
    counts["factor_c_sum_reduction"] = factor_count
    counts["posterior_precision_symmetrization"] = 2 * dimension * dimension
    precision = 0.5 * (precision + precision.T)
    mean, lower_precision = _solve_spd(precision, natural)
    covariance = np.linalg.solve(lower_precision.T, np.linalg.solve(lower_precision, np.eye(dimension, dtype=np.float64)))
    quadratic_component = 0.5 * float(natural @ mean)
    logdet_component = -float(np.sum(np.log(np.diag(lower_precision)), dtype=np.float64))
    dimension_component = 0.5 * dimension * math.log(2.0 * math.pi)
    log_normalizer = constant + quadratic_component + logdet_component + dimension_component
    conditions.append(float(np.linalg.cond(precision)))
    counts["posterior_precision_cholesky"] = _cholesky(dimension)
    counts["posterior_natural_solve"] = 2 * _triangular(dimension, 1)
    counts["posterior_quadratic"] = _dot(dimension)
    counts["posterior_logdet_reduction"] = dimension + max(0, dimension - 1)
    counts["route_sum_reduction"] = 3
    operand = H4OracleOperandEvidence(
        "canonical_log_normalizer", log_normalizer, abs(log_normalizer),
        abs(constant) + abs(quadratic_component) + abs(logdet_component) + abs(dimension_component),
        tuple(conditions), tuple((label, counts[label]) for label in _CANONICAL_LABELS),
    )
    return precision, natural, constant, mean, covariance, log_normalizer, operand


def _predictive_route(factors: tuple[_Factor, ...], dimension: int) -> tuple[NDArray[np.float64], NDArray[np.float64], float, tuple[H4OracleInnovationDiagnostic, ...], H4OracleOperandEvidence]:
    initial = tuple(item for item in factors if item.role == "initial")
    initial_indices = tuple(sorted({index for factor in initial for index in factor.normalized}))
    local = {global_index: local_index for local_index, global_index in enumerate(initial_indices)}
    initial_precision = np.zeros((len(initial_indices), len(initial_indices)), dtype=np.float64)
    initial_natural = np.zeros(len(initial_indices), dtype=np.float64)
    for factor in initial:
        matrix = factor.matrix[:, initial_indices]
        solved_matrix, lower = _solve_spd(factor.covariance, matrix)
        solved_target = np.linalg.solve(lower.T, np.linalg.solve(lower, factor.target))
        initial_precision += matrix.T @ solved_matrix
        initial_natural += matrix.T @ solved_target
    mean, lower = _solve_spd(initial_precision, initial_natural)
    covariance = np.linalg.solve(lower.T, np.linalg.solve(lower, np.eye(len(initial_indices), dtype=np.float64)))
    active = list(initial_indices)
    log_normalizer = 0.0
    absolute_increment_sum = 0.0
    conditions: list[float] = []
    innovations: list[H4OracleInnovationDiagnostic] = []
    counts = {label: 0 for label in _PREDICTIVE_LABELS}
    observation_count = 0
    for factor in factors[len(initial):]:
        if factor.role == "transition":
            active_map = {global_index: local_index for local_index, global_index in enumerate(active)}
            if any(index not in active_map for index in factor.parents):
                raise ValueError("transition references an unavailable parent")
            affine = -factor.matrix[:, active]
            child_mean = factor.target + affine @ mean
            cross = affine @ covariance
            child_covariance = factor.covariance + cross @ affine.T
            enlarged = np.zeros((covariance.shape[0] + len(factor.normalized), covariance.shape[1] + len(factor.normalized)), dtype=np.float64)
            enlarged[: covariance.shape[0], : covariance.shape[1]] = covariance
            enlarged[covariance.shape[0]:, : covariance.shape[1]] = cross
            enlarged[: covariance.shape[0], covariance.shape[1]:] = cross.T
            enlarged[covariance.shape[0]:, covariance.shape[1]:] = child_covariance
            mean = np.concatenate((mean, child_mean))
            covariance = 0.5 * (enlarged + enlarged.T)
            active.extend(factor.normalized)
            conditions.append(float(np.linalg.cond(covariance)))
            child = len(factor.normalized)
            old = covariance.shape[0] - child
            counts["affine_propagation_matmuls"] += _matmul(child, old, 1) + _matmul(child, old, old) + _matmul(child, old, child)
        elif factor.role == "observation":
            if any(index not in active for index in factor.parents):
                raise ValueError("observation references an unavailable parent")
            matrix = factor.matrix[:, active]
            residual = factor.target - matrix @ mean
            cross = covariance @ matrix.T
            innovation = factor.covariance + matrix @ cross
            lower = np.linalg.cholesky(innovation)
            whitened = np.linalg.solve(lower, residual)
            increment = -0.5 * (
                float(np.sum(whitened * whitened, dtype=np.float64))
                + factor.target.size * math.log(2.0 * math.pi)
                + 2.0 * float(np.sum(np.log(np.diag(lower)), dtype=np.float64))
            )
            gain_rhs = np.linalg.solve(lower.T, np.linalg.solve(lower, cross.T))
            gain = gain_rhs.T
            mean = mean + gain @ residual
            covariance_correction = gain @ innovation @ gain.T
            updated_covariance = covariance - covariance_correction
            covariance = 0.5 * (updated_covariance + updated_covariance.T)
            log_normalizer += increment
            absolute_increment_sum += abs(increment)
            eigenvalues = np.linalg.eigvalsh(innovation)
            innovations.append(H4OracleInnovationDiagnostic(
                factor.factor_id, factor.time_index, factor.parents, factor.target.size,
                float(eigenvalues[0]), float(eigenvalues[-1]), float(np.linalg.cond(innovation)),
            ))
            conditions.append(float(np.linalg.cond(innovation)))
            rows, active_size = factor.target.size, len(active)
            counts["innovation_assembly"] += _matmul(rows, active_size, rows) + rows * rows
            counts["innovation_cholesky"] += _cholesky(rows)
            counts["innovation_triangular_solves"] += _triangular(rows, 1)
            counts["innovation_quadratics"] += rows
            counts["innovation_logdet_reductions"] += rows + max(0, rows - 1)
            counts["kalman_gain_solves"] += 2 * _triangular(rows, active_size)
            counts["mean_updates"] += _matmul(active_size, rows, 1) + active_size
            counts["covariance_updates"] += _matmul(active_size, rows, rows) + _matmul(active_size, rows, active_size) + active_size * active_size
            observation_count += 1
        else:
            raise ValueError("initial factors must be the leading predictive block")
    if sorted(active) != list(range(dimension)):
        raise ValueError("predictive route did not construct every global coordinate")
    permutation = np.asarray([active.index(index) for index in range(dimension)], dtype=np.int64)
    global_mean = mean[permutation]
    global_covariance = covariance[np.ix_(permutation, permutation)]
    counts["route_sum_reduction"] = max(0, observation_count - 1)
    operand = H4OracleOperandEvidence(
        "predictive_log_normalizer", log_normalizer, abs(log_normalizer),
        absolute_increment_sum, tuple(conditions or (1.0,)),
        tuple((label, counts[label]) for label in _PREDICTIVE_LABELS),
    )
    return global_mean, global_covariance, log_normalizer, tuple(innovations), operand


def _route_agreement(problem_id: str, digest: str, canonical: H4OracleOperandEvidence, predictive: H4OracleOperandEvidence) -> H4OracleRouteAgreement:
    n_canonical = sum(item[1] for item in canonical.operation_counts)
    n_predictive = sum(item[1] for item in predictive.operation_counts)
    scale = max(1.0, abs(canonical.value), abs(predictive.value))
    canonical_rounding = 4096.0 * _gamma(n_canonical) * max((1.0, *canonical.condition_numbers)) * max(1.0, abs(canonical.value), canonical.absolute_summand_accumulation)
    predictive_rounding = 4096.0 * _gamma(n_predictive) * max((1.0, *predictive.condition_numbers)) * max(1.0, abs(predictive.value), predictive.absolute_summand_accumulation)
    comparison = 4096.0 * _gamma(3) * max(1.0, abs(canonical.value), abs(predictive.value), abs(canonical.value) + abs(predictive.value))
    residual = abs(canonical.value - predictive.value)
    final = canonical_rounding + predictive_rounding + comparison
    normalized = residual / final
    ratio = final / scale
    decisive = ratio < 1.0e-4
    passed = residual <= final
    return H4OracleRouteAgreement(
        problem_id, digest, canonical, predictive, _EPSILON, 4096, 0.0, 1.0e-4,
        scale, canonical_rounding, predictive_rounding, comparison, residual,
        normalized, final, ratio, "allowance_scale_ratio_strictly_less_than_1e-4",
        "residual_less_than_or_equal_to_final_allowance", decisive, passed,
        decisive and passed,
    )


def evaluate_h4_oracle(problem_payload: bytes) -> H4OracleEvaluation:
    core, factors, digest = _parse(problem_payload)
    dimension = int(core["dimension"])
    precision, natural, constant, canonical_mean, canonical_covariance, canonical_log_z, canonical_operand = _canonical_route(factors, dimension)
    predictive_mean, predictive_covariance, predictive_log_z, innovations, predictive_operand = _predictive_route(factors, dimension)
    agreement = _route_agreement(str(core["problem_id"]), digest, canonical_operand, predictive_operand)
    if not np.allclose(canonical_mean, predictive_mean, rtol=0.0, atol=agreement.final_allowance) or not np.allclose(canonical_covariance, predictive_covariance, rtol=0.0, atol=max(agreement.final_allowance, 1.0e-12)):
        raise ValueError("independent H4 posterior routes disagree outside route allowance")
    selected_indices = _selected_indices(factors, int(core["horizon"]))
    selected = tuple(
        H4OracleSelectedMoment(
            name, indices, tuple(float(value) for value in canonical_mean[list(indices)]),
            tuple(tuple(float(value) for value in row) for row in canonical_covariance[np.ix_(indices, indices)]),
        )
        for name, indices in selected_indices
    )
    eigenvalues = np.linalg.eigvalsh(precision)
    lower = np.linalg.cholesky(precision)
    posterior = H4OraclePosteriorDiagnostic(
        dimension, float(eigenvalues[0]), float(eigenvalues[-1]),
        float(np.linalg.cond(precision)), float(np.min(np.diag(lower))),
        float(np.max(np.abs(canonical_mean))),
    )
    return H4OracleEvaluation(
        "h4-numpy-oracle-v1", str(core["problem_id"]), digest,
        core["source_kind"], int(core["seed"]), core["kind"], int(core["horizon"]),
        int(core["d_z"]), int(core["d_m"]), dimension,
        tuple(core["coordinate_order"]), tuple(item.factor_id for item in factors),
        tuple(tuple(float(value) for value in row) for row in precision),
        tuple(float(value) for value in natural), float(constant),
        tuple(float(value) for value in canonical_mean),
        tuple(tuple(float(value) for value in row) for row in canonical_covariance),
        float(canonical_log_z), float(predictive_log_z), agreement, selected,
        posterior, innovations, (canonical_operand, predictive_operand),
    )


def _selected_indices(factors: tuple[_Factor, ...], horizon: int) -> tuple[tuple[str, tuple[int, ...]], ...]:
    initial = tuple(sorted({index for item in factors if item.role == "initial" for index in item.normalized}))
    terminal = tuple(sorted({index for item in factors if item.role == "transition" and item.time_index == horizon for index in item.normalized}))
    observations = tuple(
        (f"observation[{time}]", tuple(sorted({index for item in factors if item.role == "observation" and item.time_index == time for index in item.parents})))
        for time in range(1, horizon + 1)
    )
    rows = (("initial", initial), ("terminal", terminal), *observations)
    if any(not indices for _, indices in rows):
        raise ValueError("H4 selected coordinate block is missing")
    return rows


def reverse_kl_to_h4_oracle(
    oracle: H4OracleEvaluation,
    *,
    mean: tuple[float, ...],
    precision: tuple[tuple[float, ...], ...],
) -> H4OracleKLEvaluation:
    if type(oracle) is not H4OracleEvaluation or type(mean) is not tuple or len(mean) != oracle.dimension or type(precision) is not tuple or len(precision) != oracle.dimension or any(type(row) is not tuple or len(row) != oracle.dimension for row in precision):
        raise ValueError("candidate Gaussian shape does not match the H4 oracle")
    if any(type(value) is not float or not math.isfinite(value) for value in (*mean, *(item for row in precision for item in row))):
        raise ValueError("candidate Gaussian must contain exact finite floats")
    candidate_mean = np.asarray(mean, dtype=np.float64)
    candidate_precision = np.asarray(precision, dtype=np.float64)
    oracle_precision = np.asarray(oracle.precision, dtype=np.float64)
    if not np.array_equal(candidate_precision, candidate_precision.T):
        raise ValueError("candidate precision must be symmetric")
    try:
        lower_candidate = np.linalg.cholesky(candidate_precision)
    except np.linalg.LinAlgError as error:
        raise ValueError("candidate precision must be SPD") from error
    identity = np.eye(oracle.dimension, dtype=np.float64)
    candidate_covariance = np.linalg.solve(lower_candidate.T, np.linalg.solve(lower_candidate, identity))
    trace_term = float(np.trace(oracle_precision @ candidate_covariance))
    delta = np.asarray(oracle.mean, dtype=np.float64) - candidate_mean
    quadratic = float(delta @ oracle_precision @ delta)
    minus_dimension = float(-oracle.dimension)
    candidate_logdet = 2.0 * float(np.sum(np.log(np.diag(lower_candidate)), dtype=np.float64))
    lower_oracle = np.linalg.cholesky(oracle_precision)
    minus_oracle_logdet = -2.0 * float(np.sum(np.log(np.diag(lower_oracle)), dtype=np.float64))
    value = 0.5 * (trace_term + quadratic + minus_dimension + candidate_logdet + minus_oracle_logdet)
    absolute = 0.5 * sum(abs(item) for item in (trace_term, quadratic, minus_dimension, candidate_logdet, minus_oracle_logdet))
    counts = (
        ("candidate_precision_cholesky", _cholesky(oracle.dimension)),
        ("candidate_covariance_solve", 2 * _triangular(oracle.dimension, oracle.dimension)),
        ("trace_matmul", _matmul(oracle.dimension, oracle.dimension, oracle.dimension)),
        ("trace_reduction", oracle.dimension + max(0, oracle.dimension - 1)),
        ("mean_quadratic", _matmul(1, oracle.dimension, oracle.dimension) + _dot(oracle.dimension)),
        ("candidate_logdet", oracle.dimension + max(0, oracle.dimension - 1)),
        ("oracle_precision_cholesky", _cholesky(oracle.dimension)),
        ("oracle_logdet", oracle.dimension + max(0, oracle.dimension - 1)),
        ("kl_sum_reduction", 4),
    )
    return H4OracleKLEvaluation(
        value, trace_term, quadratic, minus_dimension, candidate_logdet,
        minus_oracle_logdet, absolute, float(np.linalg.cond(candidate_precision)),
        float(np.linalg.cond(oracle_precision)), counts,
    )


__all__ = [
    "H4OracleEvaluation", "H4OracleInnovationDiagnostic", "H4OracleKLEvaluation",
    "H4OracleOperandEvidence", "H4OraclePosteriorDiagnostic",
    "H4OracleRouteAgreement", "H4OracleRouteOperationLabel",
    "H4OracleSelectedMoment", "evaluate_h4_oracle", "reverse_kl_to_h4_oracle",
]
