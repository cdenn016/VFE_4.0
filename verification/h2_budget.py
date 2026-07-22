"""Literal absolute error budgets preregistered for the H2 gate."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


EPS = float(np.finfo(np.float64).eps)
C = 256.0


def gamma_n(n: int) -> float:
    if type(n) is not int:
        raise ValueError("n must be an integer")
    numerator = n * EPS
    if n <= 0 or numerator >= 1.0:
        raise ValueError("n must be positive and n*eps must be below one")
    return numerator / (1.0 - numerator)


def operation_count(dimension: int) -> int:
    if type(dimension) is not int or dimension <= 0:
        raise ValueError("dimension must be a positive integer")
    return 8 * dimension + 32


def infinity_norm(value: object) -> float:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0 or not bool(np.isfinite(array).all()):
        raise ValueError("norm input must be nonempty and finite")
    result = float(np.max(np.abs(array)))
    return _nonnegative(result, "infinity norm")


def path_allowance(
    dimension: int,
    kappas: Iterable[float],
    output_inf: object,
    absolute_sum_inf: object,
) -> float:
    checked_kappas = tuple(_nonnegative(value, "kappa") for value in kappas)
    if not checked_kappas:
        raise ValueError("kappas must not be empty")
    output = infinity_norm(output_inf)
    absolute_sum = infinity_norm(absolute_sum_inf)
    return _finite(
        C
        * gamma_n(operation_count(dimension))
        * max(1.0, *checked_kappas)
        * max(1.0, output, absolute_sum),
        "path allowance",
    )


def backward_residual_allowance(
    dimension: int,
    matrix_inf: object,
    solution_inf: object,
    rhs_inf: object,
) -> float:
    matrix = infinity_norm(matrix_inf)
    solution = infinity_norm(solution_inf)
    rhs = infinity_norm(rhs_inf)
    return _finite(
        C
        * gamma_n(operation_count(dimension))
        * max(1.0, matrix * solution + rhs),
        "backward residual allowance",
    )


def complete_elbo_allowance(
    signed_terms: Iterable[float], local_allowances: Iterable[float]
) -> float:
    terms = tuple(_finite(value, "signed term") for value in signed_terms)
    allowances = tuple(
        _nonnegative(value, "local allowance") for value in local_allowances
    )
    if len(terms) != 12 or len(allowances) != 12:
        raise ValueError("complete ELBO requires exactly 12 terms and allowances")
    return _finite(
        math.fsum(allowances)
        + C * gamma_n(13) * max(1.0, math.fsum(abs(value) for value in terms)),
        "complete ELBO allowance",
    )


def pair_allowance(
    dimension: int,
    left_allowance: float,
    right_allowance: float,
    left: object,
    right: object,
) -> float:
    left_local = _nonnegative(left_allowance, "left allowance")
    right_local = _nonnegative(right_allowance, "right allowance")
    left_norm = infinity_norm(left)
    right_norm = infinity_norm(right)
    return _finite(
        left_local
        + right_local
        + C * gamma_n(dimension + 2) * max(1.0, left_norm, right_norm),
        "pair allowance",
    )


def _nonnegative(value: object, name: str) -> float:
    checked = _finite(value, name)
    if checked < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return checked


def _finite(value: object, name: str) -> float:
    if type(value) not in (int, float, np.float64) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return float(value)


__all__ = [
    "C",
    "EPS",
    "backward_residual_allowance",
    "complete_elbo_allowance",
    "gamma_n",
    "infinity_norm",
    "operation_count",
    "pair_allowance",
    "path_allowance",
]
