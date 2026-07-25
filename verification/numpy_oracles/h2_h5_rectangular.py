"""Independent NumPy oracle for the asymmetric H2/H5 Gaussian sibling."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

import numpy as np


Vector: TypeAlias = tuple[float, ...]
Matrix: TypeAlias = tuple[Vector, ...]
_REPORT_DOMAIN = b"vfe4.h2-h5-rectangular-oracle-report.v1\x00"
_FROZEN_RAW_SHA256 = (
    "6925ffe08e4d8acbc7790b6318f3e26a0509a8208ebf062f62f721332d194aa5"
)
_FROZEN_CANONICAL_SHA256 = (
    "02add1038f70cedd2cb5b0adad0c3b23696960f9fe2a0c4942df7eea77e3f58c"
)


def _dense_matmul(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    if left.ndim != 2 or right.ndim not in (1, 2):
        raise ValueError("rectangular oracle matmul supports matrix-vector/matrix")
    if left.shape[1] != right.shape[0]:
        raise ValueError("rectangular oracle matmul dimensions do not align")
    if right.ndim == 1:
        result = np.empty(left.shape[0], dtype=np.float64)
        for row in range(left.shape[0]):
            result[row] = math.fsum(
                float(left[row, index]) * float(right[index])
                for index in range(left.shape[1])
            )
        return result
    result = np.empty(
        (left.shape[0], right.shape[1]), dtype=np.float64
    )
    for row in range(left.shape[0]):
        for column in range(right.shape[1]):
            result[row, column] = math.fsum(
                float(left[row, index]) * float(right[index, column])
                for index in range(left.shape[1])
            )
    return result


class _DenseArray(np.ndarray):
    """Small-array NumPy value whose ``@`` path is independent of BLAS."""

    __array_priority__ = 1000

    def __matmul__(self, other: object) -> _DenseArray:
        return _dense_matmul(
            np.asarray(self, dtype=np.float64),
            np.asarray(other, dtype=np.float64),
        ).view(_DenseArray)

    def __rmatmul__(self, other: object) -> _DenseArray:
        return _dense_matmul(
            np.asarray(other, dtype=np.float64),
            np.asarray(self, dtype=np.float64),
        ).view(_DenseArray)


@dataclass(frozen=True, slots=True)
class RectangularUpdateOracleReport:
    """Closed-form information assembly and both CAVI natural updates."""

    schema_version: Literal["rectangular-update-oracle-report-v1"]
    fixture_raw_sha256: str
    fixture_canonical_sha256: str
    time_index: int
    model_recoil_residual: Vector
    model_precision_pullback: Matrix
    model_recoil_natural: Vector
    state_precision: Matrix
    state_natural: Vector
    state_probe: Vector
    state_solution: Vector
    state_probe_objective: float
    state_solved_objective: float
    state_completion_square_gap: float
    state_solution_gradient_max_abs: float
    model_precision: Matrix
    model_natural: Vector
    model_probe: Vector
    model_solution: Vector
    model_probe_objective: float
    model_solved_objective: float
    model_completion_square_gap: float
    model_solution_gradient_max_abs: float
    report_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != "rectangular-update-oracle-report-v1":
            raise ValueError("unsupported rectangular oracle-report schema")
        for name in ("fixture_raw_sha256", "fixture_canonical_sha256"):
            value = getattr(self, name)
            if (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        if type(self.time_index) is not int or self.time_index < 1:
            raise ValueError("time_index must be a positive integer")
        _array(self.model_recoil_residual, (3,), "model_recoil_residual")
        _array(self.model_precision_pullback, (3, 3), "model_precision_pullback")
        _array(self.model_recoil_natural, (3,), "model_recoil_natural")
        _array(self.state_precision, (2, 2), "state_precision")
        _array(self.state_natural, (2,), "state_natural")
        _array(self.state_probe, (2,), "state_probe")
        _array(self.state_solution, (2,), "state_solution")
        _array(self.model_precision, (3, 3), "model_precision")
        _array(self.model_natural, (3,), "model_natural")
        _array(self.model_probe, (3,), "model_probe")
        _array(self.model_solution, (3,), "model_solution")
        for name in (
            "state_probe_objective",
            "state_solved_objective",
            "state_completion_square_gap",
            "state_solution_gradient_max_abs",
            "model_probe_objective",
            "model_solved_objective",
            "model_completion_square_gap",
            "model_solution_gradient_max_abs",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite float")
        for channel in ("state", "model"):
            gap = getattr(self, f"{channel}_completion_square_gap")
            gradient = getattr(self, f"{channel}_solution_gradient_max_abs")
            if gap <= 0.0:
                raise ValueError(
                    f"{channel} probe must be distinct from the solved minimum"
                )
            if gradient < 0.0:
                raise ValueError(
                    f"{channel} solution gradient norm must be nonnegative"
                )
            observed_gap = getattr(
                self, f"{channel}_probe_objective"
            ) - getattr(self, f"{channel}_solved_objective")
            if not math.isclose(
                observed_gap,
                gap,
                rel_tol=2.0e-13,
                abs_tol=2.0e-13,
            ):
                raise ValueError(
                    f"{channel} objective gap must equal its completion square"
                )
        object.__setattr__(
            self,
            "report_sha256",
            hashlib.sha256(
                _REPORT_DOMAIN + _canonical_report_bytes(self)
            ).hexdigest(),
        )


def _array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).view(_DenseArray)
    if array.shape != shape or not bool(np.isfinite(array).all()):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array


def _fixture_array(
    fixture: object,
    name: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    try:
        value = getattr(fixture, name)
    except AttributeError as exc:
        raise ValueError(f"fixture is missing {name}") from exc
    return _array(value, shape, f"fixture.{name}")


def _vector(value: np.ndarray) -> Vector:
    return tuple(float(item) for item in np.asarray(value, dtype=np.float64))


def _matrix(value: np.ndarray) -> Matrix:
    return tuple(
        tuple(float(item) for item in row)
        for row in np.asarray(value, dtype=np.float64)
    )


def _cholesky_solve_spd(
    precision: np.ndarray,
    natural: np.ndarray,
    name: str,
) -> np.ndarray:
    """Solve one tiny SPD system by scalar Cholesky, independent of LAPACK."""

    dimension = precision.shape[0]
    lower = np.zeros((dimension, dimension), dtype=np.float64).view(_DenseArray)
    for row in range(dimension):
        for column in range(row + 1):
            residual = float(precision[row, column]) - math.fsum(
                float(lower[row, index]) * float(lower[column, index])
                for index in range(column)
            )
            if row == column:
                if residual <= 0.0 or not math.isfinite(residual):
                    raise ValueError(f"{name} precision must be positive definite")
                lower[row, column] = math.sqrt(residual)
            else:
                lower[row, column] = residual / lower[column, column]
    intermediate = np.empty(dimension, dtype=np.float64).view(_DenseArray)
    for row in range(dimension):
        intermediate[row] = (
            float(natural[row])
            - math.fsum(
                float(lower[row, index]) * float(intermediate[index])
                for index in range(row)
            )
        ) / float(lower[row, row])
    solution = np.empty(dimension, dtype=np.float64).view(_DenseArray)
    for row in range(dimension - 1, -1, -1):
        solution[row] = (
            float(intermediate[row])
            - math.fsum(
                float(lower[index, row]) * float(solution[index])
                for index in range(row + 1, dimension)
            )
        ) / float(lower[row, row])
    return solution


def _coordinate_objective(
    precision: np.ndarray,
    natural: np.ndarray,
    coordinate: np.ndarray,
) -> float:
    quadratic = _dense_matmul(precision, coordinate)
    return 0.5 * math.fsum(
        float(coordinate[index]) * float(quadratic[index])
        for index in range(coordinate.shape[0])
    ) - math.fsum(
        float(natural[index]) * float(coordinate[index])
        for index in range(coordinate.shape[0])
    )


def _objective_witness(
    precision: np.ndarray,
    natural: np.ndarray,
    probe: np.ndarray,
    name: str,
) -> tuple[np.ndarray, float, float, float, float]:
    solution = _cholesky_solve_spd(precision, natural, name)
    probe_objective = _coordinate_objective(precision, natural, probe)
    solved_objective = _coordinate_objective(precision, natural, solution)
    displacement = probe - solution
    displaced = _dense_matmul(precision, displacement)
    completion_gap = 0.5 * math.fsum(
        float(displacement[index]) * float(displaced[index])
        for index in range(displacement.shape[0])
    )
    gradient = _dense_matmul(precision, solution) - natural
    gradient_max_abs = max(
        (abs(float(item)) for item in gradient),
        default=0.0,
    )
    return (
        solution,
        probe_objective,
        solved_objective,
        completion_gap,
        gradient_max_abs,
    )


def _weighted_parent_mean(
    weights: np.ndarray,
    transports: tuple[object, ...],
    means: np.ndarray,
    parents: tuple[int, ...],
    dimension: int,
    name: str,
) -> np.ndarray:
    if weights.shape != (len(parents),):
        raise ValueError(f"{name} weights do not match dense parents")
    if len(transports) != len(parents):
        raise ValueError(f"{name} transports do not match dense parents")
    result = np.zeros(dimension, dtype=np.float64).view(_DenseArray)
    for row, parent in enumerate(parents):
        transport = _array(
            transports[row],
            (dimension, dimension),
            f"{name}.transport[{row}]",
        )
        result = result + weights[row] * (transport @ means[parent])
    return result


def evaluate_rectangular_update_oracle(
    fixture: object,
    *,
    time_index: int = 2,
) -> RectangularUpdateOracleReport:
    """Evaluate equations (6.746) and (6.775) without production imports."""

    expected_identity = (
        "h2-h5-rectangular-v1",
        "peer-review-c5-pcg64-31337",
        "numpy.random.PCG64",
        31337,
        3,
        2,
        3,
        2,
    )
    observed_identity = tuple(
        getattr(fixture, name, None)
        for name in (
            "schema_version",
            "fixture_id",
            "generator",
            "seed",
            "horizon",
            "d_z",
            "d_m",
            "observation_dimension",
        )
    )
    if observed_identity != expected_identity:
        raise ValueError("fixture does not match the frozen rectangular C5 identity")
    if (
        getattr(fixture, "raw_sha256", None) != _FROZEN_RAW_SHA256
        or getattr(fixture, "canonical_sha256", None)
        != _FROZEN_CANONICAL_SHA256
    ):
        raise ValueError("fixture hashes do not match the frozen rectangular C5")
    if type(time_index) is not int or time_index < 1 or time_index > 3:
        raise ValueError("time_index must be in [1, 3]")
    parents_by_time = getattr(fixture, "dense_parents", None)
    expected_parents = ((0,), (0, 1), (0, 1, 2))
    if parents_by_time != expected_parents:
        raise ValueError("fixture must use dense causal parents range(t)")

    state_means = _fixture_array(fixture, "state_means", (4, 2))
    model_means = _fixture_array(fixture, "model_means", (4, 3))
    state_precisions = _fixture_array(fixture, "state_precisions", (3, 2, 2))
    model_precisions = _fixture_array(fixture, "model_precisions", (3, 3, 3))
    state_model_maps = _fixture_array(fixture, "state_model_maps", (3, 2, 3))
    state_offsets = _fixture_array(fixture, "state_offsets", (3, 2))
    model_offsets = _fixture_array(fixture, "model_offsets", (3, 3))
    observation_precisions = _fixture_array(
        fixture, "observation_precisions", (3, 2, 2)
    )
    observation_state_maps = _fixture_array(
        fixture, "observation_state_maps", (3, 2, 2)
    )
    observation_model_maps = _fixture_array(
        fixture, "observation_model_maps", (3, 2, 3)
    )
    observations = _fixture_array(fixture, "observations", (3, 2))
    observation_offsets = _fixture_array(
        fixture, "observation_offsets", (3, 2)
    )

    state_weights = getattr(fixture, "state_parent_weights", None)
    model_weights = getattr(fixture, "model_parent_weights", None)
    state_transports = getattr(fixture, "state_transports", None)
    model_transports = getattr(fixture, "model_transports", None)
    if not all(
        type(value) is tuple and len(value) == 3
        for value in (
            state_weights,
            model_weights,
            state_transports,
            model_transports,
        )
    ):
        raise ValueError("fixture parent arrays must be immutable three-row tuples")

    s = time_index
    row = s - 1
    parents = expected_parents[row]
    p_z = state_precisions[row]
    p_m = model_precisions[row]
    b_s = state_model_maps[row]
    z_parent_mean = _weighted_parent_mean(
        _array(state_weights[row], (s,), "state_parent_weights"),
        state_transports[row],
        state_means,
        parents,
        2,
        "state_parents",
    )
    m_parent_mean = _weighted_parent_mean(
        _array(model_weights[row], (s,), "model_parent_weights"),
        model_transports[row],
        model_means,
        parents,
        3,
        "model_parents",
    )

    state_precision = p_z.copy()
    state_natural = p_z @ (
        z_parent_mean + b_s @ model_means[s] + state_offsets[row]
    )
    for child in range(s + 1, 4):
        child_row = child - 1
        child_parents = expected_parents[child_row]
        parent_row = child_parents.index(s)
        weight = float(state_weights[child_row][parent_row])
        transport = _array(
            state_transports[child_row][parent_row],
            (2, 2),
            "state_child_transport",
        )
        child_precision = state_precisions[child_row]
        child_residual = (
            state_means[child]
            - state_model_maps[child_row] @ model_means[child]
            - state_offsets[child_row]
        )
        state_precision = (
            state_precision
            + weight * (transport.T @ child_precision @ transport)
        )
        state_natural = (
            state_natural
            + weight * (transport.T @ child_precision @ child_residual)
        )
    p_x = observation_precisions[row]
    c_z = observation_state_maps[row]
    c_m = observation_model_maps[row]
    state_precision = state_precision + c_z.T @ p_x @ c_z
    state_natural = state_natural + c_z.T @ p_x @ (
        observations[row]
        - c_m @ model_means[s]
        - observation_offsets[row]
    )

    model_recoil_residual = state_means[s] - z_parent_mean - state_offsets[row]
    model_precision_pullback = b_s.T @ p_z @ b_s
    model_recoil_natural = b_s.T @ p_z @ model_recoil_residual
    model_precision = p_m + model_precision_pullback
    model_natural = (
        p_m @ (m_parent_mean + model_offsets[row]) + model_recoil_natural
    )
    for child in range(s + 1, 4):
        child_row = child - 1
        child_parents = expected_parents[child_row]
        parent_row = child_parents.index(s)
        weight = float(model_weights[child_row][parent_row])
        transport = _array(
            model_transports[child_row][parent_row],
            (3, 3),
            "model_child_transport",
        )
        child_precision = model_precisions[child_row]
        child_residual = model_means[child] - model_offsets[child_row]
        model_precision = (
            model_precision
            + weight * (transport.T @ child_precision @ transport)
        )
        model_natural = (
            model_natural
            + weight * (transport.T @ child_precision @ child_residual)
        )
    model_precision = model_precision + c_m.T @ p_x @ c_m
    model_natural = model_natural + c_m.T @ p_x @ (
        observations[row]
        - c_z @ state_means[s]
        - observation_offsets[row]
    )
    (
        state_solution,
        state_probe_objective,
        state_solved_objective,
        state_completion_square_gap,
        state_solution_gradient_max_abs,
    ) = _objective_witness(
        state_precision,
        state_natural,
        state_means[s],
        "state",
    )
    (
        model_solution,
        model_probe_objective,
        model_solved_objective,
        model_completion_square_gap,
        model_solution_gradient_max_abs,
    ) = _objective_witness(
        model_precision,
        model_natural,
        model_means[s],
        "model",
    )

    return RectangularUpdateOracleReport(
        schema_version="rectangular-update-oracle-report-v1",
        fixture_raw_sha256=getattr(fixture, "raw_sha256", ""),
        fixture_canonical_sha256=getattr(fixture, "canonical_sha256", ""),
        time_index=s,
        model_recoil_residual=_vector(model_recoil_residual),
        model_precision_pullback=_matrix(model_precision_pullback),
        model_recoil_natural=_vector(model_recoil_natural),
        state_precision=_matrix(state_precision),
        state_natural=_vector(state_natural),
        state_probe=_vector(state_means[s]),
        state_solution=_vector(state_solution),
        state_probe_objective=float(state_probe_objective),
        state_solved_objective=float(state_solved_objective),
        state_completion_square_gap=float(state_completion_square_gap),
        state_solution_gradient_max_abs=float(
            state_solution_gradient_max_abs
        ),
        model_precision=_matrix(model_precision),
        model_natural=_vector(model_natural),
        model_probe=_vector(model_means[s]),
        model_solution=_vector(model_solution),
        model_probe_objective=float(model_probe_objective),
        model_solved_objective=float(model_solved_objective),
        model_completion_square_gap=float(model_completion_square_gap),
        model_solution_gradient_max_abs=float(
            model_solution_gradient_max_abs
        ),
    )


def _canonical(value: object) -> object:
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("oracle report values must be finite")
        return value.hex()
    if type(value) in (str, int):
        return value
    if type(value) is tuple:
        return [_canonical(item) for item in value]
    raise ValueError(f"unsupported oracle report value {type(value).__name__}")


def _canonical_report_bytes(report: RectangularUpdateOracleReport) -> bytes:
    core = {
        name: _canonical(getattr(report, name))
        for name in (
            "schema_version",
            "fixture_raw_sha256",
            "fixture_canonical_sha256",
            "time_index",
            "model_recoil_residual",
            "model_precision_pullback",
            "model_recoil_natural",
            "state_precision",
            "state_natural",
            "state_probe",
            "state_solution",
            "state_probe_objective",
            "state_solved_objective",
            "state_completion_square_gap",
            "state_solution_gradient_max_abs",
            "model_precision",
            "model_natural",
            "model_probe",
            "model_solution",
            "model_probe_objective",
            "model_solved_objective",
            "model_completion_square_gap",
            "model_solution_gradient_max_abs",
        )
    }
    return json.dumps(
        core,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")


__all__ = [
    "RectangularUpdateOracleReport",
    "evaluate_rectangular_update_oracle",
]
