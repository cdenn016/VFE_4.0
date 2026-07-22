"""Operand-shaped binary64 allowances for the deterministic H5 objective."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from vfe4.types.h5_schema import (
    H5_ANALYTIC_FACTOR_OPERATION_COUNTS,
    H5_ANALYTIC_OPERATION_COUNTS,
    H5_C,
    H5_DIAGNOSTIC_TERM_IDS,
    H5_EPS,
    H5_QUADRATURE_ORDERS,
    H5_SIGNED_TERM_IDS,
    H5_SIGNED_TERM_SIGNS,
    emission_operation_count,
    gamma_n,
)


def _finite(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite binary64 float")
    return value


def _nonnegative(value: object, name: str) -> float:
    checked = _finite(value, name)
    if checked < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return checked


def _finite_tuple(
    value: object,
    name: str,
    *,
    nonnegative: bool,
    minimum: float | None = None,
) -> tuple[float, ...]:
    if type(value) is not tuple or not value:
        raise ValueError(f"{name} must be a nonempty tuple")
    checked: list[float] = []
    for index, item in enumerate(value):
        number = _finite(item, f"{name}[{index}]")
        if nonnegative and number < 0.0:
            raise ValueError(f"{name}[{index}] must be nonnegative")
        if minimum is not None and number < minimum:
            raise ValueError(f"{name}[{index}] must be at least {minimum}")
        checked.append(number)
    return tuple(checked)


def _operation_count(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _expected_operation_counts(term_id: str) -> tuple[int, int]:
    if term_id in ("expected_log_emission[1]", "expected_log_emission[2]"):
        return emission_operation_count(21), emission_operation_count(17)
    if term_id in H5_ANALYTIC_OPERATION_COUNTS:
        count = H5_ANALYTIC_OPERATION_COUNTS[term_id]
        return count, count
    raise ValueError("term_id is outside the closed H5 term universe")


def _expected_sign(term_id: str) -> Literal[-1, 0, 1]:
    if term_id in H5_SIGNED_TERM_IDS:
        return H5_SIGNED_TERM_SIGNS[H5_SIGNED_TERM_IDS.index(term_id)]  # type: ignore[return-value]
    if term_id in H5_DIAGNOSTIC_TERM_IDS:
        return 0
    raise ValueError("term_id is outside the closed H5 term universe")


def _rounding_allowance(
    value: float,
    absolute_summands: tuple[float, ...],
    condition_numbers: tuple[float, ...],
    operation_count: int,
) -> float:
    scale = max(1.0, abs(value), math.fsum(absolute_summands))
    condition = max(1.0, *condition_numbers)
    result = H5_C * gamma_n(operation_count) * condition * scale
    return _nonnegative(float(result), "rounding allowance")


def _comparison_rounding(value_order_21: float, value_order_17: float) -> float:
    scale = max(
        1.0,
        abs(value_order_21),
        abs(value_order_17),
        abs(value_order_21) + abs(value_order_17),
    )
    return _nonnegative(
        float(H5_C * gamma_n(3) * scale), "comparison rounding"
    )


@dataclass(frozen=True)
class H5TermAllowance:
    term_id: str
    objective_sign: Literal[-1, 0, 1]
    value_order_21: float
    value_order_17: float
    signed_reported_value: float
    absolute_summands_order_21: tuple[float, ...]
    absolute_summands_order_17: tuple[float, ...]
    condition_numbers_order_21: tuple[float, ...]
    condition_numbers_order_17: tuple[float, ...]
    operation_count_order_21: int
    operation_count_order_17: int
    convergence_estimate: float
    rounding_order_21: float
    rounding_order_17: float
    comparison_rounding: float
    total: float

    def __post_init__(self) -> None:
        if type(self.term_id) is not str or not self.term_id:
            raise ValueError("term_id must be a nonempty string")
        if type(self.objective_sign) is not int or self.objective_sign not in (-1, 0, 1):
            raise ValueError("objective_sign must be -1, 0, or 1")
        expected_sign = _expected_sign(self.term_id)
        if self.objective_sign != expected_sign:
            raise ValueError("objective_sign does not match the H5 term schema")
        value_21 = _finite(self.value_order_21, "value_order_21")
        value_17 = _finite(self.value_order_17, "value_order_17")
        if self.term_id not in (
            "expected_log_emission[1]",
            "expected_log_emission[2]",
        ) and value_21.hex() != value_17.hex():
            raise ValueError("analytic H5 term values must be order-identical")
        signed = _finite(self.signed_reported_value, "signed_reported_value")
        expected_signed = float(self.objective_sign * value_21)
        if signed.hex() != expected_signed.hex():
            raise ValueError("signed_reported_value must be recomputed exactly")
        summands_21 = _finite_tuple(
            self.absolute_summands_order_21,
            "absolute_summands_order_21",
            nonnegative=True,
        )
        summands_17 = _finite_tuple(
            self.absolute_summands_order_17,
            "absolute_summands_order_17",
            nonnegative=True,
        )
        conditions_21 = _finite_tuple(
            self.condition_numbers_order_21,
            "condition_numbers_order_21",
            nonnegative=True,
            minimum=1.0,
        )
        conditions_17 = _finite_tuple(
            self.condition_numbers_order_17,
            "condition_numbers_order_17",
            nonnegative=True,
            minimum=1.0,
        )
        count_21 = _operation_count(
            self.operation_count_order_21, "operation_count_order_21"
        )
        count_17 = _operation_count(
            self.operation_count_order_17, "operation_count_order_17"
        )
        if (count_21, count_17) != _expected_operation_counts(self.term_id):
            raise ValueError("operation counts do not match the frozen H5 table")
        expected_convergence = abs(value_21 - value_17)
        expected_rounding_21 = _rounding_allowance(
            value_21, summands_21, conditions_21, count_21
        )
        expected_rounding_17 = _rounding_allowance(
            value_17, summands_17, conditions_17, count_17
        )
        expected_comparison = _comparison_rounding(value_21, value_17)
        expected_total = math.fsum(
            (
                expected_convergence,
                expected_rounding_21,
                expected_rounding_17,
                expected_comparison,
            )
        )
        for name, actual, expected in (
            ("convergence_estimate", self.convergence_estimate, expected_convergence),
            ("rounding_order_21", self.rounding_order_21, expected_rounding_21),
            ("rounding_order_17", self.rounding_order_17, expected_rounding_17),
            ("comparison_rounding", self.comparison_rounding, expected_comparison),
            ("total", self.total, expected_total),
        ):
            checked = _nonnegative(actual, name)
            if checked.hex() != float(expected).hex():
                raise ValueError(f"{name} must be recomputed exactly")
        object.__setattr__(self, "absolute_summands_order_21", summands_21)
        object.__setattr__(self, "absolute_summands_order_17", summands_17)
        object.__setattr__(self, "condition_numbers_order_21", conditions_21)
        object.__setattr__(self, "condition_numbers_order_17", conditions_17)


@dataclass(frozen=True)
class H5CompleteAllowance:
    term_allowances: tuple[H5TermAllowance, ...]
    reduction_rounding: float
    total: float
    stochastic_contribution: float

    def __post_init__(self) -> None:
        if (
            type(self.term_allowances) is not tuple
            or tuple(item.term_id for item in self.term_allowances)
            != H5_SIGNED_TERM_IDS
            or not all(isinstance(item, H5TermAllowance) for item in self.term_allowances)
        ):
            raise ValueError("term_allowances must equal the signed H5 term order")
        signed_values = tuple(item.signed_reported_value for item in self.term_allowances)
        expected_reduction = H5_C * gamma_n(13) * max(
            1.0, math.fsum(abs(value) for value in signed_values)
        )
        expected_total = math.fsum(
            (math.fsum(item.total for item in self.term_allowances), expected_reduction)
        )
        if _nonnegative(self.reduction_rounding, "reduction_rounding").hex() != float(
            expected_reduction
        ).hex():
            raise ValueError("reduction_rounding must be recomputed exactly")
        if _nonnegative(self.total, "total").hex() != float(expected_total).hex():
            raise ValueError("total must be recomputed exactly")
        if (
            type(self.stochastic_contribution) is not float
            or self.stochastic_contribution.hex() != 0.0.hex()
        ):
            raise ValueError("stochastic_contribution must equal positive zero")


@dataclass(frozen=True)
class H5DeltaAllowance:
    before_total: float
    after_total: float
    subtraction_rounding: float
    stochastic_contribution: float
    epsilon_delta: float

    def __post_init__(self) -> None:
        before = _nonnegative(self.before_total, "before_total")
        after = _nonnegative(self.after_total, "after_total")
        subtraction = _nonnegative(
            self.subtraction_rounding, "subtraction_rounding"
        )
        if (
            type(self.stochastic_contribution) is not float
            or self.stochastic_contribution.hex() != 0.0.hex()
        ):
            raise ValueError("stochastic_contribution must equal positive zero")
        expected = math.fsum((before, after, subtraction))
        if _nonnegative(self.epsilon_delta, "epsilon_delta").hex() != expected.hex():
            raise ValueError("epsilon_delta must be recomputed exactly")


@dataclass(frozen=True)
class H5BudgetConfig:
    quadrature_orders: tuple[Literal[21], Literal[17]]
    epsilon: float
    C: float
    signed_term_ids: tuple[str, ...]
    analytic_operation_counts: Mapping[str, int]
    analytic_factor_operation_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        if self.quadrature_orders != H5_QUADRATURE_ORDERS:
            raise ValueError("quadrature_orders must equal (21, 17)")
        if type(self.epsilon) is not float or self.epsilon.hex() != H5_EPS.hex():
            raise ValueError("epsilon must equal the frozen binary64 epsilon")
        if type(self.C) is not float or self.C.hex() != H5_C.hex():
            raise ValueError("C must equal 4096.0")
        if self.signed_term_ids != H5_SIGNED_TERM_IDS:
            raise ValueError("signed_term_ids must equal the H5 signed term order")
        term_counts = _copy_count_mapping(
            self.analytic_operation_counts, "analytic_operation_counts"
        )
        factor_counts = _copy_count_mapping(
            self.analytic_factor_operation_counts,
            "analytic_factor_operation_counts",
        )
        if term_counts != dict(H5_ANALYTIC_OPERATION_COUNTS):
            raise ValueError("analytic_operation_counts do not match H5")
        if factor_counts != dict(H5_ANALYTIC_FACTOR_OPERATION_COUNTS):
            raise ValueError("analytic_factor_operation_counts do not match H5")
        object.__setattr__(self, "quadrature_orders", tuple(self.quadrature_orders))
        object.__setattr__(self, "signed_term_ids", tuple(self.signed_term_ids))
        object.__setattr__(
            self,
            "analytic_operation_counts",
            MappingProxyType(dict(sorted(term_counts.items()))),
        )
        object.__setattr__(
            self,
            "analytic_factor_operation_counts",
            MappingProxyType(dict(sorted(factor_counts.items()))),
        )


def _copy_count_mapping(value: object, name: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    copied = dict(value)
    for key, count in copied.items():
        if type(key) is not str or not key:
            raise ValueError(f"{name} keys must be nonempty strings")
        _operation_count(count, f"{name}[{key!r}]")
    return copied


DEFAULT_H5_BUDGET_CONFIG = H5BudgetConfig(
    H5_QUADRATURE_ORDERS,
    H5_EPS,
    H5_C,
    H5_SIGNED_TERM_IDS,
    H5_ANALYTIC_OPERATION_COUNTS,
    H5_ANALYTIC_FACTOR_OPERATION_COUNTS,
)


def term_allowance(
    term_id: str,
    *,
    objective_sign: Literal[-1, 0, 1],
    value_order_21: float,
    value_order_17: float,
    absolute_summands_order_21: tuple[float, ...],
    absolute_summands_order_17: tuple[float, ...],
    condition_numbers_order_21: tuple[float, ...],
    condition_numbers_order_17: tuple[float, ...],
    operation_count_order_21: int,
    operation_count_order_17: int,
) -> H5TermAllowance:
    checked_21 = _finite(value_order_21, "value_order_21")
    checked_17 = _finite(value_order_17, "value_order_17")
    checked_summands_21 = _finite_tuple(
        absolute_summands_order_21,
        "absolute_summands_order_21",
        nonnegative=True,
    )
    checked_summands_17 = _finite_tuple(
        absolute_summands_order_17,
        "absolute_summands_order_17",
        nonnegative=True,
    )
    checked_conditions_21 = _finite_tuple(
        condition_numbers_order_21,
        "condition_numbers_order_21",
        nonnegative=True,
        minimum=1.0,
    )
    checked_conditions_17 = _finite_tuple(
        condition_numbers_order_17,
        "condition_numbers_order_17",
        nonnegative=True,
        minimum=1.0,
    )
    checked_count_21 = _operation_count(
        operation_count_order_21, "operation_count_order_21"
    )
    checked_count_17 = _operation_count(
        operation_count_order_17, "operation_count_order_17"
    )
    convergence = abs(checked_21 - checked_17)
    rounding_21 = _rounding_allowance(
        checked_21,
        checked_summands_21,
        checked_conditions_21,
        checked_count_21,
    )
    rounding_17 = _rounding_allowance(
        checked_17,
        checked_summands_17,
        checked_conditions_17,
        checked_count_17,
    )
    comparison = _comparison_rounding(checked_21, checked_17)
    total = math.fsum((convergence, rounding_21, rounding_17, comparison))
    return H5TermAllowance(
        term_id,
        objective_sign,
        checked_21,
        checked_17,
        float(objective_sign * checked_21),
        checked_summands_21,
        checked_summands_17,
        checked_conditions_21,
        checked_conditions_17,
        checked_count_21,
        checked_count_17,
        float(convergence),
        float(rounding_21),
        float(rounding_17),
        float(comparison),
        float(total),
    )


def complete_elbo_allowance(
    term_allowances: tuple[H5TermAllowance, ...],
    signed_terms: tuple[float, ...],
) -> H5CompleteAllowance:
    if (
        type(term_allowances) is not tuple
        or tuple(item.term_id for item in term_allowances) != H5_SIGNED_TERM_IDS
        or not all(isinstance(item, H5TermAllowance) for item in term_allowances)
    ):
        raise ValueError("term_allowances must equal the signed H5 term order")
    if type(signed_terms) is not tuple or len(signed_terms) != len(term_allowances):
        raise ValueError("signed_terms must align with term_allowances")
    checked_signed = tuple(
        _finite(value, f"signed_terms[{index}]")
        for index, value in enumerate(signed_terms)
    )
    if tuple(value.hex() for value in checked_signed) != tuple(
        item.signed_reported_value.hex() for item in term_allowances
    ):
        raise ValueError("signed_terms must equal the allowance signed values")
    reduction = H5_C * gamma_n(13) * max(
        1.0, math.fsum(abs(value) for value in checked_signed)
    )
    total = math.fsum(
        (math.fsum(item.total for item in term_allowances), reduction)
    )
    return H5CompleteAllowance(
        tuple(term_allowances), float(reduction), float(total), 0.0
    )


def subtraction_rounding_allowance(before_elbo: float, after_elbo: float) -> float:
    before = _finite(before_elbo, "before_elbo")
    after = _finite(after_elbo, "after_elbo")
    delta = after - before
    scale = max(
        1.0,
        abs(before),
        abs(after),
        abs(delta),
        abs(before) + abs(after),
    )
    return _nonnegative(
        float(H5_C * gamma_n(3) * scale), "subtraction rounding"
    )


def epsilon_delta(
    before: H5CompleteAllowance,
    after: H5CompleteAllowance,
    *,
    before_elbo: float,
    after_elbo: float,
) -> H5DeltaAllowance:
    if not isinstance(before, H5CompleteAllowance) or not isinstance(
        after, H5CompleteAllowance
    ):
        raise ValueError("before and after must be H5CompleteAllowance records")
    subtraction = subtraction_rounding_allowance(before_elbo, after_elbo)
    total = math.fsum((before.total, after.total, subtraction))
    return H5DeltaAllowance(
        before.total,
        after.total,
        subtraction,
        0.0,
        float(total),
    )


__all__ = [
    "DEFAULT_H5_BUDGET_CONFIG",
    "H5BudgetConfig",
    "H5CompleteAllowance",
    "H5DeltaAllowance",
    "H5TermAllowance",
    "complete_elbo_allowance",
    "epsilon_delta",
    "subtraction_rounding_allowance",
    "term_allowance",
]
