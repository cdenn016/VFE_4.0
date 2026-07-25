"""Production-versus-NumPy gate for the rectangular H2/H5 sibling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
import torch

from verification.numpy_oracles.h2_h5_rectangular import (
    evaluate_rectangular_update_oracle,
)
from vfe4.inference.h5_updates import (
    rectangular_gaussian_coordinate_objective,
    rectangular_model_natural_update,
    rectangular_state_natural_update,
)
from vfe4.numerics.linear_gaussian import assemble_rectangular_information
from vfe4.validation.h2_h5_rectangular_fixture import (
    H2H5RectangularFixture,
    H2_H5_RECTANGULAR_CANONICAL_SHA256,
    H2_H5_RECTANGULAR_RAW_SHA256,
    load_h2_h5_rectangular_fixture,
)


Vector: TypeAlias = tuple[float, ...]
Matrix: TypeAlias = tuple[Vector, ...]


@dataclass(frozen=True, slots=True)
class H2H5RectangularGateResult:
    schema_version: Literal["h2-h5-rectangular-gate-result-v1"]
    fixture_raw_sha256: str
    fixture_canonical_sha256: str
    oracle_report_sha256: str
    time_index: int
    production_model_precision_pullback: Matrix
    production_model_recoil_natural: Vector
    production_state_precision: Matrix
    production_state_natural: Vector
    production_state_solution: Vector
    production_state_probe_objective: float
    production_state_solved_objective: float
    production_state_completion_square_gap: float
    production_state_solution_gradient_max_abs: float
    production_model_precision: Matrix
    production_model_natural: Vector
    production_model_solution: Vector
    production_model_probe_objective: float
    production_model_solved_objective: float
    production_model_completion_square_gap: float
    production_model_solution_gradient_max_abs: float
    information_transpose_rejected: bool
    state_update_transpose_rejected: bool
    model_update_transpose_rejected: bool
    state_minimum_witness_passed: bool
    model_minimum_witness_passed: bool
    maximum_absolute_error: float
    passed: bool

    def __post_init__(self) -> None:
        if self.schema_version != "h2-h5-rectangular-gate-result-v1":
            raise ValueError("unsupported rectangular gate-result schema")
        for name in (
            "fixture_raw_sha256",
            "fixture_canonical_sha256",
            "oracle_report_sha256",
        ):
            value = getattr(self, name)
            if (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        if type(self.time_index) is not int or self.time_index < 1:
            raise ValueError("time_index must be a positive integer")
        for name, shape in (
            ("production_model_precision_pullback", (3, 3)),
            ("production_model_recoil_natural", (3,)),
            ("production_state_precision", (2, 2)),
            ("production_state_natural", (2,)),
            ("production_state_solution", (2,)),
            ("production_model_precision", (3, 3)),
            ("production_model_natural", (3,)),
            ("production_model_solution", (3,)),
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != shape or not bool(np.isfinite(value).all()):
                raise ValueError(f"{name} must be finite with shape {shape}")
        for name in (
            "information_transpose_rejected",
            "state_update_transpose_rejected",
            "model_update_transpose_rejected",
            "state_minimum_witness_passed",
            "model_minimum_witness_passed",
            "passed",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be Boolean")
        for name in (
            "production_state_probe_objective",
            "production_state_solved_objective",
            "production_state_completion_square_gap",
            "production_state_solution_gradient_max_abs",
            "production_model_probe_objective",
            "production_model_solved_objective",
            "production_model_completion_square_gap",
            "production_model_solution_gradient_max_abs",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite float")
        if (
            type(self.maximum_absolute_error) is not float
            or not math.isfinite(self.maximum_absolute_error)
            or self.maximum_absolute_error < 0.0
        ):
            raise ValueError("maximum_absolute_error must be finite and nonnegative")


def _tensor(value: object) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.float64, device="cpu")


def _vector(value: torch.Tensor) -> Vector:
    return tuple(float(item) for item in value.detach().cpu())


def _matrix(value: torch.Tensor) -> Matrix:
    return tuple(
        tuple(float(item) for item in row)
        for row in value.detach().cpu()
    )


def _maximum_error(
    actual: tuple[object, ...],
    expected: tuple[object, ...],
) -> float:
    errors = []
    for left, right in zip(actual, expected, strict=True):
        difference = np.abs(
            np.asarray(left, dtype=np.float64)
            - np.asarray(right, dtype=np.float64)
        )
        errors.append(float(np.max(difference)))
    return max(errors, default=0.0)


def _transpose_rejected(function: object, kwargs: dict[str, object]) -> bool:
    try:
        function(**kwargs)  # type: ignore[operator]
    except ValueError as exc:
        return "transposed" in str(exc)
    return False


def _production_objective_witness(
    precision: torch.Tensor,
    natural: torch.Tensor,
    probe: torch.Tensor,
) -> tuple[torch.Tensor, float, float, float, float, bool]:
    solution = torch.linalg.solve(precision, natural)
    probe_objective = float(
        rectangular_gaussian_coordinate_objective(
            precision=precision,
            natural=natural,
            coordinate=probe,
        ).item()
    )
    solved_objective = float(
        rectangular_gaussian_coordinate_objective(
            precision=precision,
            natural=natural,
            coordinate=solution,
        ).item()
    )
    displacement = probe - solution
    completion_gap = float(
        (0.5 * torch.dot(displacement, precision @ displacement)).item()
    )
    gradient_max_abs = float(
        torch.max(torch.abs(precision @ solution - natural)).item()
    )
    observed_gap = probe_objective - solved_objective
    tolerance = 5.0e-13 * max(1.0, abs(observed_gap), abs(completion_gap))
    passed = (
        completion_gap > 0.0
        and observed_gap > 0.0
        and abs(observed_gap - completion_gap) <= tolerance
        and gradient_max_abs <= 5.0e-13
    )
    return (
        solution,
        probe_objective,
        solved_objective,
        completion_gap,
        gradient_max_abs,
        passed,
    )


def evaluate_h2_h5_rectangular_gate(
    fixture: H2H5RectangularFixture | None = None,
    *,
    time_index: int = 2,
) -> H2H5RectangularGateResult:
    """Compare the production sibling paths with the independent dense oracle."""

    checked_fixture = (
        load_h2_h5_rectangular_fixture() if fixture is None else fixture
    )
    if type(checked_fixture) is not H2H5RectangularFixture:
        raise ValueError("fixture must be H2H5RectangularFixture")
    if (
        checked_fixture.raw_sha256 != H2_H5_RECTANGULAR_RAW_SHA256
        or checked_fixture.canonical_sha256
        != H2_H5_RECTANGULAR_CANONICAL_SHA256
    ):
        raise ValueError("fixture hashes do not match the frozen rectangular C5")
    oracle = evaluate_rectangular_update_oracle(
        checked_fixture, time_index=time_index
    )
    s = time_index
    row = s - 1
    parents = checked_fixture.dense_parents[row]
    children = tuple(range(s + 1, checked_fixture.horizon + 1))

    state_precision = _tensor(checked_fixture.state_precisions[row])
    model_precision = _tensor(checked_fixture.model_precisions[row])
    state_model_map = _tensor(checked_fixture.state_model_maps[row])
    recoil_residual = _tensor(oracle.model_recoil_residual)
    information = assemble_rectangular_information(
        state_precision=state_precision,
        state_model_map=state_model_map,
        model_recoil_residual=recoil_residual,
    )

    state_kwargs: dict[str, object] = {
        "state_precision": state_precision,
        "parent_weights": _tensor(
            checked_fixture.state_parent_weights[row]
        ),
        "parent_transports": tuple(
            _tensor(value) for value in checked_fixture.state_transports[row]
        ),
        "parent_state_means": tuple(
            _tensor(checked_fixture.state_means[parent])
            for parent in parents
        ),
        "state_model_map": state_model_map,
        "model_mean": _tensor(checked_fixture.model_means[s]),
        "state_offset": _tensor(checked_fixture.state_offsets[row]),
        "child_weights": _tensor(
            tuple(
                checked_fixture.state_parent_weights[child - 1][
                    checked_fixture.dense_parents[child - 1].index(s)
                ]
                for child in children
            )
        ),
        "child_transports": tuple(
            _tensor(
                checked_fixture.state_transports[child - 1][
                    checked_fixture.dense_parents[child - 1].index(s)
                ]
            )
            for child in children
        ),
        "child_precisions": tuple(
            _tensor(checked_fixture.state_precisions[child - 1])
            for child in children
        ),
        "child_state_means": tuple(
            _tensor(checked_fixture.state_means[child])
            for child in children
        ),
        "child_state_model_maps": tuple(
            _tensor(checked_fixture.state_model_maps[child - 1])
            for child in children
        ),
        "child_model_means": tuple(
            _tensor(checked_fixture.model_means[child])
            for child in children
        ),
        "child_state_offsets": tuple(
            _tensor(checked_fixture.state_offsets[child - 1])
            for child in children
        ),
        "observation_precision": _tensor(
            checked_fixture.observation_precisions[row]
        ),
        "observation_state_map": _tensor(
            checked_fixture.observation_state_maps[row]
        ),
        "observation_model_map": _tensor(
            checked_fixture.observation_model_maps[row]
        ),
        "observation": _tensor(checked_fixture.observations[row]),
        "observation_offset": _tensor(
            checked_fixture.observation_offsets[row]
        ),
    }
    state_update = rectangular_state_natural_update(**state_kwargs)

    model_kwargs: dict[str, object] = {
        "model_precision": model_precision,
        "state_precision": state_precision,
        "state_model_map": state_model_map,
        "model_parent_weights": _tensor(
            checked_fixture.model_parent_weights[row]
        ),
        "model_parent_transports": tuple(
            _tensor(value) for value in checked_fixture.model_transports[row]
        ),
        "model_parent_means": tuple(
            _tensor(checked_fixture.model_means[parent])
            for parent in parents
        ),
        "model_offset": _tensor(checked_fixture.model_offsets[row]),
        "state_mean": _tensor(checked_fixture.state_means[s]),
        "state_parent_weights": _tensor(
            checked_fixture.state_parent_weights[row]
        ),
        "state_parent_transports": tuple(
            _tensor(value) for value in checked_fixture.state_transports[row]
        ),
        "state_parent_means": tuple(
            _tensor(checked_fixture.state_means[parent])
            for parent in parents
        ),
        "state_offset": _tensor(checked_fixture.state_offsets[row]),
        "child_weights": _tensor(
            tuple(
                checked_fixture.model_parent_weights[child - 1][
                    checked_fixture.dense_parents[child - 1].index(s)
                ]
                for child in children
            )
        ),
        "child_transports": tuple(
            _tensor(
                checked_fixture.model_transports[child - 1][
                    checked_fixture.dense_parents[child - 1].index(s)
                ]
            )
            for child in children
        ),
        "child_precisions": tuple(
            _tensor(checked_fixture.model_precisions[child - 1])
            for child in children
        ),
        "child_model_means": tuple(
            _tensor(checked_fixture.model_means[child])
            for child in children
        ),
        "child_model_offsets": tuple(
            _tensor(checked_fixture.model_offsets[child - 1])
            for child in children
        ),
        "observation_precision": _tensor(
            checked_fixture.observation_precisions[row]
        ),
        "observation_state_map": _tensor(
            checked_fixture.observation_state_maps[row]
        ),
        "observation_model_map": _tensor(
            checked_fixture.observation_model_maps[row]
        ),
        "observation": _tensor(checked_fixture.observations[row]),
        "observation_offset": _tensor(
            checked_fixture.observation_offsets[row]
        ),
    }
    model_update = rectangular_model_natural_update(**model_kwargs)
    (
        production_state_solution,
        production_state_probe_objective,
        production_state_solved_objective,
        production_state_completion_square_gap,
        production_state_solution_gradient_max_abs,
        state_minimum_witness_passed,
    ) = _production_objective_witness(
        state_update[0],
        state_update[1],
        _tensor(oracle.state_probe),
    )
    (
        production_model_solution,
        production_model_probe_objective,
        production_model_solved_objective,
        production_model_completion_square_gap,
        production_model_solution_gradient_max_abs,
        model_minimum_witness_passed,
    ) = _production_objective_witness(
        model_update[0],
        model_update[1],
        _tensor(oracle.model_probe),
    )

    information_transpose_rejected = _transpose_rejected(
        assemble_rectangular_information,
        {
            "state_precision": state_precision,
            "state_model_map": state_model_map.transpose(0, 1),
            "model_recoil_residual": recoil_residual,
        },
    )
    state_transpose_kwargs = dict(state_kwargs)
    state_transpose_kwargs["state_model_map"] = state_model_map.transpose(0, 1)
    state_update_transpose_rejected = _transpose_rejected(
        rectangular_state_natural_update, state_transpose_kwargs
    )
    model_transpose_kwargs = dict(model_kwargs)
    model_transpose_kwargs["state_model_map"] = state_model_map.transpose(0, 1)
    model_update_transpose_rejected = _transpose_rejected(
        rectangular_model_natural_update, model_transpose_kwargs
    )

    production = (
        _matrix(information.model_precision_pullback),
        _vector(information.model_recoil_natural),
        _matrix(state_update[0]),
        _vector(state_update[1]),
        _matrix(model_update[0]),
        _vector(model_update[1]),
        _vector(production_state_solution),
        production_state_probe_objective,
        production_state_solved_objective,
        production_state_completion_square_gap,
        production_state_solution_gradient_max_abs,
        _vector(production_model_solution),
        production_model_probe_objective,
        production_model_solved_objective,
        production_model_completion_square_gap,
        production_model_solution_gradient_max_abs,
    )
    expected = (
        oracle.model_precision_pullback,
        oracle.model_recoil_natural,
        oracle.state_precision,
        oracle.state_natural,
        oracle.model_precision,
        oracle.model_natural,
        oracle.state_solution,
        oracle.state_probe_objective,
        oracle.state_solved_objective,
        oracle.state_completion_square_gap,
        oracle.state_solution_gradient_max_abs,
        oracle.model_solution,
        oracle.model_probe_objective,
        oracle.model_solved_objective,
        oracle.model_completion_square_gap,
        oracle.model_solution_gradient_max_abs,
    )
    maximum_error = _maximum_error(production, expected)
    passed = (
        information_transpose_rejected
        and state_update_transpose_rejected
        and model_update_transpose_rejected
        and state_minimum_witness_passed
        and model_minimum_witness_passed
        and maximum_error <= 5.0e-13
    )
    return H2H5RectangularGateResult(
        schema_version="h2-h5-rectangular-gate-result-v1",
        fixture_raw_sha256=checked_fixture.raw_sha256,
        fixture_canonical_sha256=checked_fixture.canonical_sha256,
        oracle_report_sha256=oracle.report_sha256,
        time_index=s,
        production_model_precision_pullback=production[0],
        production_model_recoil_natural=production[1],
        production_state_precision=production[2],
        production_state_natural=production[3],
        production_model_precision=production[4],
        production_model_natural=production[5],
        production_state_solution=production[6],
        production_state_probe_objective=production[7],
        production_state_solved_objective=production[8],
        production_state_completion_square_gap=production[9],
        production_state_solution_gradient_max_abs=production[10],
        production_model_solution=production[11],
        production_model_probe_objective=production[12],
        production_model_solved_objective=production[13],
        production_model_completion_square_gap=production[14],
        production_model_solution_gradient_max_abs=production[15],
        information_transpose_rejected=information_transpose_rejected,
        state_update_transpose_rejected=state_update_transpose_rejected,
        model_update_transpose_rejected=model_update_transpose_rejected,
        state_minimum_witness_passed=state_minimum_witness_passed,
        model_minimum_witness_passed=model_minimum_witness_passed,
        maximum_absolute_error=maximum_error,
        passed=passed,
    )


__all__ = [
    "H2H5RectangularGateResult",
    "evaluate_h2_h5_rectangular_gate",
]
