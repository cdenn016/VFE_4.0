"""Frozen operand-local absolute error budgets for the H3 gate."""

from __future__ import annotations

import math

import numpy as np


EPS = float(np.finfo(np.float64).eps)
C = 4096.0
SOLVER_ALLOWANCE_NATS = 1.0e-7
_DIMENSION = 4


def _finite(value: object, name: str) -> float:
    if type(value) is bool or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be finite numeric data")
    checked = float(value)
    if not math.isfinite(checked):
        raise ValueError(f"{name} must be finite numeric data")
    return checked


def _nonnegative(value: object, name: str) -> float:
    checked = _finite(value, name)
    if checked < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return checked


def _dimension(value: object) -> int:
    if type(value) is not int or value != _DIMENSION:
        raise ValueError("dimension must equal integer four")
    return value


def gamma_n(n: int) -> float:
    if type(n) is not int:
        raise ValueError("n must be an integer")
    numerator = n * EPS
    if n <= 0 or numerator >= 1.0:
        raise ValueError("n must be positive and n*eps must be below one")
    return numerator / (1.0 - numerator)


def operation_count(dimension: int) -> int:
    checked = _dimension(dimension)
    return 16 * checked + 64


def scalar_allowance(
    dimension: int,
    *,
    value: float,
    absolute_sum: float,
    kappas: tuple[float, ...],
    optimized: bool,
) -> float:
    checked_dimension = _dimension(dimension)
    checked_value = _finite(value, "value")
    checked_absolute_sum = _nonnegative(absolute_sum, "absolute_sum")
    if type(kappas) is not tuple or not kappas:
        raise ValueError("kappas must be a nonempty tuple")
    checked_kappas = tuple(
        _nonnegative(kappa, f"kappas[{index}]")
        for index, kappa in enumerate(kappas)
    )
    if type(optimized) is not bool:
        raise ValueError("optimized must be a bool")
    rounding = (
        C
        * gamma_n(operation_count(checked_dimension))
        * max(1.0, *checked_kappas)
        * max(1.0, abs(checked_value), checked_absolute_sum)
    )
    result = (SOLVER_ALLOWANCE_NATS if optimized else 0.0) + rounding
    return _nonnegative(result, "scalar allowance")


def pair_allowance(
    dimension: int,
    *,
    left: float,
    right: float,
    left_allowance: float,
    right_allowance: float,
) -> float:
    checked_dimension = _dimension(dimension)
    checked_left = _finite(left, "left")
    checked_right = _finite(right, "right")
    checked_left_allowance = _nonnegative(left_allowance, "left_allowance")
    checked_right_allowance = _nonnegative(right_allowance, "right_allowance")
    comparison = C * gamma_n(checked_dimension + 2) * max(
        1.0,
        abs(checked_left),
        abs(checked_right),
        abs(checked_left) + abs(checked_right),
    )
    return _nonnegative(
        checked_left_allowance + checked_right_allowance + comparison,
        "pair allowance",
    )


def three_operand_identity_allowance(
    dimension: int,
    *,
    operands: tuple[float, float, float],
    operand_allowances: tuple[float, float, float],
) -> float:
    return _identity_allowance(
        dimension,
        operands=operands,
        operand_allowances=operand_allowances,
        arity=3,
    )


def four_operand_identity_allowance(
    dimension: int,
    *,
    operands: tuple[float, float, float, float],
    operand_allowances: tuple[float, float, float, float],
) -> float:
    return _identity_allowance(
        dimension,
        operands=operands,
        operand_allowances=operand_allowances,
        arity=4,
    )


def _identity_allowance(
    dimension: int,
    *,
    operands: tuple[float, ...],
    operand_allowances: tuple[float, ...],
    arity: int,
) -> float:
    checked_dimension = _dimension(dimension)
    if type(operands) is not tuple or len(operands) != arity:
        raise ValueError(f"operands must be a tuple of length {arity}")
    if type(operand_allowances) is not tuple or len(operand_allowances) != arity:
        raise ValueError(f"operand_allowances must be a tuple of length {arity}")
    checked_operands = tuple(
        _finite(value, f"operands[{index}]")
        for index, value in enumerate(operands)
    )
    checked_allowances = tuple(
        _nonnegative(value, f"operand_allowances[{index}]")
        for index, value in enumerate(operand_allowances)
    )
    reduction = C * gamma_n(checked_dimension + arity) * max(
        1.0, sum(abs(value) for value in checked_operands)
    )
    return _nonnegative(
        sum(checked_allowances) + reduction,
        f"{arity}-operand identity allowance",
    )


def allowance_is_decisive(allowance: float, decisiveness_scale: float) -> bool:
    checked_allowance = _nonnegative(allowance, "allowance")
    checked_scale = _finite(decisiveness_scale, "decisiveness_scale")
    if checked_scale <= 0.0:
        raise ValueError("decisiveness_scale must be positive")
    return checked_allowance < 0.01 * checked_scale


__all__ = [
    "C",
    "EPS",
    "SOLVER_ALLOWANCE_NATS",
    "allowance_is_decisive",
    "four_operand_identity_allowance",
    "gamma_n",
    "operation_count",
    "pair_allowance",
    "scalar_allowance",
    "three_operand_identity_allowance",
]
