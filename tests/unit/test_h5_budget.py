from __future__ import annotations

import ast
import math
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path

import pytest

import vfe4.numerics as public_numerics
import vfe4.numerics.h5_budget as budget_module
from vfe4.numerics.h5_budget import (
    DEFAULT_H5_BUDGET_CONFIG,
    H5BudgetConfig,
    H5CompleteAllowance,
    H5DeltaAllowance,
    H5TermAllowance,
    complete_elbo_allowance,
    epsilon_delta,
    subtraction_rounding_allowance,
    term_allowance,
)
from vfe4.types.h5_schema import (
    H5_ANALYTIC_FACTOR_OPERATION_COUNTS,
    H5_ANALYTIC_OPERATION_COUNTS,
    H5_C,
    H5_EPS,
    H5_QUADRATURE_ORDERS,
    H5_SIGNED_TERM_IDS,
    emission_operation_count,
    gamma_n,
)


ROOT = Path(__file__).parents[2]


def test_h5_budget_config_is_exact_and_defensively_immutable() -> None:
    term_counts = dict(H5_ANALYTIC_OPERATION_COUNTS)
    factor_counts = dict(H5_ANALYTIC_FACTOR_OPERATION_COUNTS)
    config = H5BudgetConfig(
        H5_QUADRATURE_ORDERS,
        H5_EPS,
        H5_C,
        H5_SIGNED_TERM_IDS,
        term_counts,
        factor_counts,
    )
    term_counts["initial_model_kl"] = 1
    factor_counts["initial_joint"] = 1
    assert dict(config.analytic_operation_counts) == dict(
        H5_ANALYTIC_OPERATION_COUNTS
    )
    assert dict(config.analytic_factor_operation_counts) == dict(
        H5_ANALYTIC_FACTOR_OPERATION_COUNTS
    )
    assert config == DEFAULT_H5_BUDGET_CONFIG

    with pytest.raises(TypeError):
        config.analytic_operation_counts["initial_model_kl"] = 1  # type: ignore[index]
    with pytest.raises(ValueError):
        replace(config, quadrature_orders=(17, 21))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        replace(config, epsilon=math.nextafter(H5_EPS, math.inf))
    with pytest.raises(ValueError):
        replace(config, C=2048.0)
    with pytest.raises(ValueError):
        replace(
            config,
            analytic_operation_counts={
                **dict(H5_ANALYTIC_OPERATION_COUNTS),
                "initial_model_kl": 1,
            },
        )


def test_term_allowance_uses_only_its_order_local_operands() -> None:
    summands_21 = (2.0, 0.5, 0.25)
    summands_17 = (1.75, 0.25)
    conditions_21 = (3.0,)
    conditions_17 = (2.0,)
    allowance = term_allowance(
        "expected_log_emission[1]",
        objective_sign=1,
        value_order_21=-1.75,
        value_order_17=-1.5,
        absolute_summands_order_21=summands_21,
        absolute_summands_order_17=summands_17,
        condition_numbers_order_21=conditions_21,
        condition_numbers_order_17=conditions_17,
        operation_count_order_21=emission_operation_count(21),
        operation_count_order_17=emission_operation_count(17),
    )
    expected_21 = (
        H5_C
        * gamma_n(emission_operation_count(21))
        * 3.0
        * max(1.0, 1.75, math.fsum(summands_21))
    )
    expected_17 = (
        H5_C
        * gamma_n(emission_operation_count(17))
        * 2.0
        * max(1.0, 1.5, math.fsum(summands_17))
    )
    expected_comparison = H5_C * gamma_n(3) * max(
        1.0, 1.75, 1.5, 1.75 + 1.5
    )
    assert allowance.convergence_estimate == 0.25
    assert allowance.rounding_order_21 == expected_21
    assert allowance.rounding_order_17 == expected_17
    assert allowance.comparison_rounding == expected_comparison
    assert allowance.total == math.fsum(
        (0.25, expected_21, expected_17, expected_comparison)
    )
    assert allowance.signed_reported_value == -1.75

    analytic = term_allowance(
        "initial_model_kl",
        objective_sign=-1,
        value_order_21=0.125,
        value_order_17=0.125,
        absolute_summands_order_21=(0.25, 0.125),
        absolute_summands_order_17=(0.25, 0.125),
        condition_numbers_order_21=(1.0,),
        condition_numbers_order_17=(1.0,),
        operation_count_order_21=H5_ANALYTIC_OPERATION_COUNTS["initial_model_kl"],
        operation_count_order_17=H5_ANALYTIC_OPERATION_COUNTS["initial_model_kl"],
    )
    assert analytic.convergence_estimate == 0.0
    assert analytic.signed_reported_value == -0.125


def test_analytic_term_allowance_rejects_any_between_order_difference() -> None:
    value_21 = 0.125
    value_17 = math.nextafter(value_21, math.inf)
    count = H5_ANALYTIC_OPERATION_COUNTS["initial_model_kl"]
    summands = (0.25, 0.125)
    conditions = (1.0,)
    rounding_21 = (
        H5_C
        * gamma_n(count)
        * max(conditions)
        * max(1.0, abs(value_21), math.fsum(summands))
    )
    rounding_17 = (
        H5_C
        * gamma_n(count)
        * max(conditions)
        * max(1.0, abs(value_17), math.fsum(summands))
    )
    comparison = H5_C * gamma_n(3) * max(
        1.0,
        abs(value_21),
        abs(value_17),
        abs(value_21) + abs(value_17),
    )
    convergence = abs(value_21 - value_17)
    total = math.fsum((convergence, rounding_21, rounding_17, comparison))

    with pytest.raises(ValueError, match="analytic.*order"):
        term_allowance(
            "initial_model_kl",
            objective_sign=-1,
            value_order_21=value_21,
            value_order_17=value_17,
            absolute_summands_order_21=summands,
            absolute_summands_order_17=summands,
            condition_numbers_order_21=conditions,
            condition_numbers_order_17=conditions,
            operation_count_order_21=count,
            operation_count_order_17=count,
        )
    with pytest.raises(ValueError, match="analytic.*order"):
        H5TermAllowance(
            "initial_model_kl",
            -1,
            value_21,
            value_17,
            -value_21,
            summands,
            summands,
            conditions,
            conditions,
            count,
            count,
            convergence,
            rounding_21,
            rounding_17,
            comparison,
            total,
        )


def test_complete_and_delta_allowances_follow_the_frozen_formulas() -> None:
    signed = tuple(
        term_allowance(
            term_id,
            objective_sign=1 if index < 2 else -1,
            value_order_21=float(index + 1) / 16.0,
            value_order_17=float(index + 1) / 16.0,
            absolute_summands_order_21=(float(index + 1) / 16.0,),
            absolute_summands_order_17=(float(index + 1) / 16.0,),
            condition_numbers_order_21=(1.0,),
            condition_numbers_order_17=(1.0,),
            operation_count_order_21=(
                emission_operation_count(21)
                if index < 2
                else H5_ANALYTIC_OPERATION_COUNTS[term_id]
            ),
            operation_count_order_17=(
                emission_operation_count(17)
                if index < 2
                else H5_ANALYTIC_OPERATION_COUNTS[term_id]
            ),
        )
        for index, term_id in enumerate(H5_SIGNED_TERM_IDS)
    )
    signed_values = tuple(item.signed_reported_value for item in signed)
    complete = complete_elbo_allowance(signed, signed_values)
    expected_reduction = H5_C * gamma_n(13) * max(
        1.0, math.fsum(abs(value) for value in signed_values)
    )
    assert complete.term_allowances == signed
    assert complete.reduction_rounding == expected_reduction
    assert complete.total == math.fsum(
        (math.fsum(item.total for item in signed), expected_reduction)
    )
    assert complete.stochastic_contribution == 0.0

    after = replace(
        complete,
        reduction_rounding=complete.reduction_rounding,
        total=complete.total,
    )
    before_elbo = -3.0
    after_elbo = -2.75
    subtraction = subtraction_rounding_allowance(before_elbo, after_elbo)
    expected_subtraction = H5_C * gamma_n(3) * max(
        1.0,
        abs(before_elbo),
        abs(after_elbo),
        abs(after_elbo - before_elbo),
        abs(before_elbo) + abs(after_elbo),
    )
    assert subtraction == expected_subtraction
    delta = epsilon_delta(
        complete,
        after,
        before_elbo=before_elbo,
        after_elbo=after_elbo,
    )
    assert delta.before_total == complete.total
    assert delta.after_total == after.total
    assert delta.subtraction_rounding == expected_subtraction
    assert delta.stochastic_contribution == 0.0
    assert delta.epsilon_delta == math.fsum(
        (complete.total, after.total, expected_subtraction)
    )
    with pytest.raises(ValueError):
        replace(delta, stochastic_contribution=math.ulp(1.0))
    with pytest.raises(ValueError):
        replace(delta, epsilon_delta=math.nextafter(delta.epsilon_delta, math.inf))


def test_h5_production_budget_has_no_verification_dependency() -> None:
    source = (ROOT / "vfe4/numerics/h5_budget.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = tuple(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ) + tuple(
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )
    assert not any("verification" in module for module in imported)
    assert all(
        not module.startswith("vfe4.") or module == "vfe4.types.h5_schema"
        for module in imported
    )


def test_h5_budget_public_fields_are_exact_frozen_and_exported() -> None:
    assert tuple(field.name for field in fields(H5TermAllowance)) == (
        "term_id",
        "objective_sign",
        "value_order_21",
        "value_order_17",
        "signed_reported_value",
        "absolute_summands_order_21",
        "absolute_summands_order_17",
        "condition_numbers_order_21",
        "condition_numbers_order_17",
        "operation_count_order_21",
        "operation_count_order_17",
        "convergence_estimate",
        "rounding_order_21",
        "rounding_order_17",
        "comparison_rounding",
        "total",
    )
    assert tuple(field.name for field in fields(H5CompleteAllowance)) == (
        "term_allowances",
        "reduction_rounding",
        "total",
        "stochastic_contribution",
    )
    assert tuple(field.name for field in fields(H5DeltaAllowance)) == (
        "before_total",
        "after_total",
        "subtraction_rounding",
        "stochastic_contribution",
        "epsilon_delta",
    )
    assert tuple(field.name for field in fields(H5BudgetConfig)) == (
        "quadrature_orders",
        "epsilon",
        "C",
        "signed_term_ids",
        "analytic_operation_counts",
        "analytic_factor_operation_counts",
    )
    for record_type in (
        H5TermAllowance,
        H5CompleteAllowance,
        H5DeltaAllowance,
        H5BudgetConfig,
    ):
        assert record_type.__dataclass_params__.frozen is True
    with pytest.raises(FrozenInstanceError):
        DEFAULT_H5_BUDGET_CONFIG.C = 1.0  # type: ignore[misc]

    expected_exports = (
        "DEFAULT_H5_BUDGET_CONFIG",
        "H5BudgetConfig",
        "H5CompleteAllowance",
        "H5DeltaAllowance",
        "H5TermAllowance",
        "complete_elbo_allowance",
        "epsilon_delta",
        "subtraction_rounding_allowance",
        "term_allowance",
    )
    assert budget_module.__all__ == list(expected_exports)
    for name in expected_exports:
        assert name in public_numerics.__all__
        assert getattr(public_numerics, name) is getattr(budget_module, name)
