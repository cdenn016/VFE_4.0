"""Exact, auditable operand-local budgets for frozen H7 invariants.

Every allowance is derived from one closed, category-specific operand scope.
Composite counts are bound to already validated child budgets, and backward
counts are bound to already validated forward and inverse-action budgets.
There is no public aggregate condition, scale, or operation-count input.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Final

from vfe4.types.h7 import (
    H7AllowanceContribution,
    H7BackwardResidualRecord,
    H7BudgetCategory,
    H7BudgetRecord,
    H7OperandRecord,
    H7OperationKind,
)


EPS64: Final = 2.0**-52
ROUNDING_CONSTANT: Final = 4096.0
MAX_ORACLE_RELATIVE_DELTA: Final = Decimal("1e-18")
CONTROL_MINIMUM_RELATIVE_RESIDUAL: Final = 1e-8
CONTROL_ALLOWANCE_MULTIPLE: Final = 100.0
GH_DIMENSION: Final = 2

_VECTOR_CATEGORIES = frozenset({"vector", "information", "offset", "decoder"})
_MATRIX_CATEGORIES = frozenset({"covariance", "precision", "second_moment", "map"})
_LEAF_CHILD_CATEGORIES = frozenset(
    {
        *_VECTOR_CATEGORIES,
        *_MATRIX_CATEGORIES,
        "cocycle",
        "density",
    }
)
_VALID_OPERATION_KINDS = frozenset(
    {
        "exact_identity",
        "direct_solve",
        "matrix_product",
        "quadratic_form",
        "logdet",
        "analytic_density",
        "gauss_hermite",
        "pair_comparison",
    }
)
_CATEGORY_OPERATION_KINDS = {
    "vector": frozenset({"matrix_product", "direct_solve"}),
    "information": frozenset({"matrix_product", "direct_solve"}),
    "offset": frozenset({"matrix_product", "direct_solve"}),
    "decoder": frozenset({"matrix_product", "direct_solve"}),
    "covariance": frozenset({"matrix_product", "direct_solve"}),
    "precision": frozenset({"matrix_product", "direct_solve"}),
    "second_moment": frozenset({"matrix_product", "direct_solve"}),
    "map": frozenset({"matrix_product", "direct_solve"}),
    "cocycle": frozenset({"matrix_product"}),
    "density": frozenset({"analytic_density"}),
    "local_term": frozenset({"pair_comparison", "gauss_hermite"}),
    "complete_objective": frozenset({"pair_comparison", "gauss_hermite"}),
    "backward": frozenset({"direct_solve"}),
}


@dataclass(frozen=True)
class H7BudgetFormula:
    """One closed category formula with semantically typed operand groups."""

    category: H7BudgetCategory
    operation_kind: H7OperationKind
    dimension_operand_id: str | None
    compared_operand_ids: tuple[str, ...]
    source_operand_ids: tuple[str, ...]
    direct_action_operand_ids: tuple[str, ...]
    spd_operand_ids: tuple[str, ...]
    frame_operand_ids: tuple[str, ...]
    link_operand_ids: tuple[str, ...]
    signed_summand_operand_ids: tuple[str, ...]
    child_budgets: tuple[H7BoundBudget, ...]
    forward_budget: H7BoundBudget | None
    inverse_action_budget: H7BoundBudget | None
    quadrature_operand_ids: tuple[str, str] | None = None
    reference_operand_id: str | None = None

    def __post_init__(self) -> None:
        if self.operation_kind not in _VALID_OPERATION_KINDS:
            raise ValueError("formula has an unsupported operation kind")
        if self.operation_kind not in _CATEGORY_OPERATION_KINDS.get(
            self.category,
            frozenset(),
        ):
            raise ValueError("formula operation kind is not valid for its category")
        if self.dimension_operand_id is not None and (
            type(self.dimension_operand_id) is not str or not self.dimension_operand_id
        ):
            raise ValueError("dimension_operand_id must be nonempty or None")
        for name, operand_ids in (
            ("compared_operand_ids", self.compared_operand_ids),
            ("source_operand_ids", self.source_operand_ids),
            ("direct_action_operand_ids", self.direct_action_operand_ids),
            ("spd_operand_ids", self.spd_operand_ids),
            ("frame_operand_ids", self.frame_operand_ids),
            ("link_operand_ids", self.link_operand_ids),
            ("signed_summand_operand_ids", self.signed_summand_operand_ids),
        ):
            _require_id_tuple(name, operand_ids)
        if type(self.child_budgets) is not tuple or any(
            type(item) is not H7BoundBudget for item in self.child_budgets
        ):
            raise ValueError("child budgets must be exact formula-bound records")
        if len({item.bound_sha256 for item in self.child_budgets}) != len(
            self.child_budgets
        ) or len({item.budget.invariant_id for item in self.child_budgets}) != len(
            self.child_budgets
        ):
            raise ValueError("child budgets and invariant IDs must be unique")
        for child in self.child_budgets:
            child.__post_init__()
        for name, budget in (
            ("forward_budget", self.forward_budget),
            ("inverse_action_budget", self.inverse_action_budget),
        ):
            if budget is not None:
                if type(budget) is not H7BoundBudget:
                    raise ValueError(f"{name} must be a formula-bound budget or None")
                budget.__post_init__()
        _require_formula_group_schema(self)
        _require_disjoint_formula_groups(self)
        if self.quadrature_operand_ids is not None:
            _require_id_tuple(
                "quadrature_operand_ids",
                self.quadrature_operand_ids,
                exact_length=2,
            )
            if self.quadrature_operand_ids[0] == self.quadrature_operand_ids[1]:
                raise ValueError(
                    "quadrature operands must name distinct GH41/GH51 values"
                )
        if self.reference_operand_id is not None and (
            type(self.reference_operand_id) is not str or not self.reference_operand_id
        ):
            raise ValueError("reference_operand_id must be nonempty or None")
        if (self.operation_kind == "gauss_hermite") != (
            self.quadrature_operand_ids is not None
        ):
            raise ValueError("only Gauss-Hermite formulas name GH41/GH51 operands")

    @property
    def condition_operand_ids(self) -> tuple[str, ...]:
        """Derive the only admissible condition factors for this category."""

        if self.category in _VECTOR_CATEGORIES:
            return self.direct_action_operand_ids
        if self.category in _MATRIX_CATEGORIES:
            return (*self.direct_action_operand_ids, *self.spd_operand_ids)
        if self.category == "cocycle":
            return (*self.direct_action_operand_ids, *self.frame_operand_ids)
        if self.category == "density":
            return (*self.spd_operand_ids, *self.direct_action_operand_ids)
        if self.category in ("local_term", "complete_objective"):
            return self.spd_operand_ids
        if self.category == "backward":
            return self.direct_action_operand_ids
        raise ValueError("unsupported formula category")

    @property
    def scale_operand_ids(self) -> tuple[str, ...]:
        """Derive the only admissible scale/normalization operands."""

        if self.category in (*_VECTOR_CATEGORIES, *_MATRIX_CATEGORIES):
            return (*self.compared_operand_ids, *self.source_operand_ids)
        if self.category == "cocycle":
            return (*self.link_operand_ids, *self.compared_operand_ids)
        if self.category in ("density", "local_term", "complete_objective"):
            return (*self.compared_operand_ids, *self.signed_summand_operand_ids)
        if self.category == "backward":
            return self.compared_operand_ids
        raise ValueError("unsupported formula category")

    @property
    def formula_sha256(self) -> str:
        semantic = {
            "category": self.category,
            "operation_kind": self.operation_kind,
            "dimension_operand_id": self.dimension_operand_id,
            "compared_operand_ids": self.compared_operand_ids,
            "source_operand_ids": self.source_operand_ids,
            "direct_action_operand_ids": self.direct_action_operand_ids,
            "spd_operand_ids": self.spd_operand_ids,
            "frame_operand_ids": self.frame_operand_ids,
            "link_operand_ids": self.link_operand_ids,
            "signed_summand_operand_ids": self.signed_summand_operand_ids,
            "child_bound_sha256": [item.bound_sha256 for item in self.child_budgets],
            "forward_bound_sha256": (
                None
                if self.forward_budget is None
                else self.forward_budget.bound_sha256
            ),
            "inverse_action_bound_sha256": (
                None
                if self.inverse_action_budget is None
                else self.inverse_action_budget.bound_sha256
            ),
            "quadrature_operand_ids": self.quadrature_operand_ids,
            "reference_operand_id": self.reference_operand_id,
        }
        return hashlib.sha256(
            b"vfe4.h7.budget-formula.v2\x00"
            + json.dumps(
                semantic,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class H7BoundBudget:
    """A budget inseparably bound to a fully reproducible frozen formula."""

    budget: H7BudgetRecord
    formula: H7BudgetFormula
    formula_sha256: str
    bound_sha256: str

    @classmethod
    def create(
        cls,
        budget: H7BudgetRecord,
        formula: H7BudgetFormula,
    ) -> H7BoundBudget:
        formula_sha256 = formula.formula_sha256
        return cls(
            budget=budget,
            formula=formula,
            formula_sha256=formula_sha256,
            bound_sha256=_bound_budget_sha256(
                budget.budget_sha256,
                formula_sha256,
            ),
        )

    def __post_init__(self) -> None:
        if (
            type(self.budget) is not H7BudgetRecord
            or type(self.formula) is not H7BudgetFormula
            or self.formula_sha256 != self.formula.formula_sha256
            or self.bound_sha256
            != _bound_budget_sha256(
                self.budget.budget_sha256,
                self.formula_sha256,
            )
        ):
            raise ValueError("bound budget/formula identity is inconsistent")
        _validate_bound_budget_semantics(self)


def _bound_budget_sha256(budget_sha256: str, formula_sha256: str) -> str:
    return hashlib.sha256(
        b"vfe4.h7.bound-budget.v2\x00"
        + budget_sha256.encode("ascii")
        + b"\x00"
        + formula_sha256.encode("ascii")
    ).hexdigest()


@dataclass(frozen=True)
class H7BackwardOperandInput:
    """One actual backward operand plus its bound forward/action provenance."""

    operand_id: str
    original_sha256: str
    transformed_sha256: str
    recovered_sha256: str
    numerator: float
    normalization: float
    operands: tuple[H7OperandRecord, ...]
    formula: H7BudgetFormula

    def __post_init__(self) -> None:
        if type(self.operand_id) is not str or not self.operand_id:
            raise ValueError("backward operand_id must be nonempty")
        if (
            type(self.numerator) is not float
            or not math.isfinite(self.numerator)
            or self.numerator < 0.0
            or type(self.normalization) is not float
            or not math.isfinite(self.normalization)
            or self.normalization <= 0.0
            or type(self.formula) is not H7BudgetFormula
            or self.formula.category != "backward"
        ):
            raise ValueError("backward numerator/normalization/formula is invalid")
        _require_local_operands("backward", self.operands)
        values_by_role = {
            role: tuple(item for item in self.operands if item.role == role)
            for role in ("original", "transformed", "recovered")
        }
        if any(len(items) != 1 for items in values_by_role.values()):
            raise ValueError(
                "backward requires one original/transformed/recovered value"
            )
        expected_hashes = {
            "original": self.original_sha256,
            "transformed": self.transformed_sha256,
            "recovered": self.recovered_sha256,
        }
        if any(
            values_by_role[role][0].value_sha256 != expected
            for role, expected in expected_hashes.items()
        ):
            raise ValueError("backward hashes must bind their local operand values")
        _resolve_formula_scope("backward", self.operands, self.formula)


@dataclass(frozen=True)
class H7BackwardBudgetAggregate:
    """All per-operand records and their formula-bound budgets."""

    records: tuple[H7BackwardResidualRecord, ...]
    bound_budgets: tuple[H7BoundBudget, ...]
    maximum: float

    def __post_init__(self) -> None:
        if (
            type(self.records) is not tuple
            or not self.records
            or any(type(item) is not H7BackwardResidualRecord for item in self.records)
            or type(self.bound_budgets) is not tuple
            or len(self.bound_budgets) != len(self.records)
            or any(type(item) is not H7BoundBudget for item in self.bound_budgets)
            or any(
                record.budget.budget_sha256 != bound.budget.budget_sha256
                for record, bound in zip(
                    self.records,
                    self.bound_budgets,
                    strict=True,
                )
            )
            or type(self.maximum) is not float
            or not math.isfinite(self.maximum)
            or self.maximum != max(item.value for item in self.records)
        ):
            raise ValueError("backward aggregate must retain every exact record")


def gamma_n(operation_count: int) -> float:
    if type(operation_count) is not int or operation_count <= 0:
        raise ValueError("operation_count must be a positive integer")
    product = operation_count * EPS64
    if product >= 1.0:
        raise ValueError("operation_count is outside the binary64 gamma domain")
    return product / (1.0 - product)


def frozen_operation_count(
    category: H7BudgetCategory,
    operands: tuple[H7OperandRecord, ...],
    formula: H7BudgetFormula,
) -> int:
    """Evaluate one frozen count from closed operands and bound child budgets."""

    scope = _resolve_formula_scope(category, operands, formula)
    if category in (*_VECTOR_CATEGORIES, *_MATRIX_CATEGORIES, "cocycle", "density"):
        dimension_operand = scope[formula.dimension_operand_id]
        n = _dimension_from_shape(category, dimension_operand.shape)
        if category in _VECTOR_CATEGORIES:
            count = 32 * n + 64
        elif category in _MATRIX_CATEGORIES:
            count = 64 * n**3 + 128 * n**2 + 64 * n + 256
        elif category == "cocycle":
            count = 96 * n**3 + 128 * n**2 + 256
        else:
            count = 128 * n**3 + 192 * n**2 + 128 * n + 512
    elif category in ("local_term", "complete_objective"):
        count = (
            sum(_bound_operation_count(item) for item in formula.child_budgets)
            + 32 * len(formula.signed_summand_operand_ids)
            + 64
        )
    elif category == "backward":
        if formula.forward_budget is None or formula.inverse_action_budget is None:
            raise ValueError("backward formula lacks bound operation provenance")
        count = _bound_operation_count(formula.forward_budget) + _bound_operation_count(
            formula.inverse_action_budget
        )
    else:
        raise ValueError("unsupported frozen H7 budget category")
    if count <= 0:
        raise ValueError("frozen operation count must be positive")
    return count


def gauss_hermite_2d_operation_count(order: int) -> int:
    """Return the exact declared two-dimensional tensor-grid node count."""

    if type(order) is not int or order <= 0:
        raise ValueError("quadrature order must be positive")
    return order**GH_DIMENSION


def build_h7_budget(
    *,
    invariant_id: str,
    category: H7BudgetCategory,
    operands: tuple[H7OperandRecord, ...],
    formula: H7BudgetFormula,
) -> H7BoundBudget:
    """Build one budget solely from its closed, hash-bound formula."""

    if type(invariant_id) is not str or not invariant_id:
        raise ValueError("invariant_id must be nonempty")
    if type(formula) is not H7BudgetFormula or formula.category != category:
        raise ValueError("formula category must equal the budget category")
    operation_count = frozen_operation_count(category, operands, formula)
    by_id = _resolve_formula_scope(category, operands, formula)
    condition_operands = tuple(by_id[item] for item in formula.condition_operand_ids)
    scale_operands = tuple(by_id[item] for item in formula.scale_operand_ids)
    local_condition_product = math.prod(
        item.condition_number for item in condition_operands
    )
    local_scale = max(item.scale for item in scale_operands)
    comparison_normalization = max(item.normalization for item in scale_operands)
    if not all(
        math.isfinite(item)
        for item in (
            local_condition_product,
            local_scale,
            comparison_normalization,
        )
    ):
        raise ValueError("budget condition, scale, and normalization must be finite")
    contributions = _expected_contributions(
        invariant_id,
        operands,
        formula,
        operation_count,
        local_condition_product,
        local_scale,
    )
    budget = H7BudgetRecord.create(
        invariant_id=invariant_id,
        category=category,
        operands=operands,
        contributions=contributions,
        comparison_normalization=comparison_normalization,
        total_allowance=math.fsum(item.value for item in contributions),
    )
    return H7BoundBudget.create(budget, formula)


def build_h7_backward_records(
    inputs: tuple[H7BackwardOperandInput, ...],
    *,
    required_operand_ids: tuple[str, ...],
) -> H7BackwardBudgetAggregate:
    """Construct every exact record before reducing to ``r_back_max``."""

    if (
        type(inputs) is not tuple
        or not inputs
        or any(type(item) is not H7BackwardOperandInput for item in inputs)
        or type(required_operand_ids) is not tuple
        or not required_operand_ids
        or any(type(item) is not str or not item for item in required_operand_ids)
        or len(set(required_operand_ids)) != len(required_operand_ids)
    ):
        raise ValueError("backward construction requires an exact frozen inventory")
    operand_ids = tuple(item.operand_id for item in inputs)
    if operand_ids != required_operand_ids:
        raise ValueError("backward operand IDs are missing, extra, or reordered")
    records: list[H7BackwardResidualRecord] = []
    bound_budgets: list[H7BoundBudget] = []
    for item in inputs:
        bound = build_h7_budget(
            invariant_id=f"backward:{item.operand_id}",
            category="backward",
            operands=item.operands,
            formula=item.formula,
        )
        value = item.numerator / item.normalization
        records.append(
            H7BackwardResidualRecord.create(
                operand_id=item.operand_id,
                original_sha256=item.original_sha256,
                transformed_sha256=item.transformed_sha256,
                recovered_sha256=item.recovered_sha256,
                numerator=item.numerator,
                normalization=item.normalization,
                value=value,
                budget=bound.budget,
                passed=value <= bound.budget.total_allowance,
            )
        )
        bound_budgets.append(bound)
    frozen_records = tuple(records)
    return H7BackwardBudgetAggregate(
        records=frozen_records,
        bound_budgets=tuple(bound_budgets),
        maximum=max(item.value for item in frozen_records),
    )


def control_decisiveness_limit(correct_budget: H7BoundBudget) -> float:
    """Use only the category scope already hashed into the correct budget."""

    if type(correct_budget) is not H7BoundBudget:
        raise ValueError("correct_budget must be a formula-bound H7 budget")
    correct_budget.__post_init__()
    budget = correct_budget.budget
    by_id = {item.operand_id: item for item in budget.operands}
    local_scale = max(
        by_id[operand_id].scale
        for operand_id in correct_budget.formula.scale_operand_ids
    )
    return max(
        CONTROL_ALLOWANCE_MULTIPLE * budget.total_allowance,
        CONTROL_MINIMUM_RELATIVE_RESIDUAL * local_scale,
    )


def require_control_decisive(
    residual: float,
    correct_budget: H7BoundBudget,
) -> float:
    if type(residual) is not float or not math.isfinite(residual) or residual < 0.0:
        raise ValueError("control residual must be a finite nonnegative float")
    limit = control_decisiveness_limit(correct_budget)
    if residual <= limit:
        raise ValueError("control-decisiveness boundary was not cleared")
    return limit


def _require_formula_group_schema(formula: H7BudgetFormula) -> None:
    category = formula.category
    empty_child_sources = (
        not formula.child_budgets
        and formula.forward_budget is None
        and formula.inverse_action_budget is None
    )
    if category in _VECTOR_CATEGORIES:
        valid = (
            formula.dimension_operand_id is not None
            and len(formula.compared_operand_ids) == 2
            and len(formula.source_operand_ids) == 1
            and formula.dimension_operand_id == formula.source_operand_ids[0]
            and bool(formula.direct_action_operand_ids)
            and not formula.spd_operand_ids
            and not formula.frame_operand_ids
            and not formula.link_operand_ids
            and not formula.signed_summand_operand_ids
            and empty_child_sources
        )
    elif category in _MATRIX_CATEGORIES:
        expected_spd = (
            formula.source_operand_ids
            if category in {"covariance", "precision", "second_moment"}
            else ()
        )
        valid = (
            formula.dimension_operand_id is not None
            and len(formula.compared_operand_ids) == 2
            and len(formula.source_operand_ids) == 1
            and formula.dimension_operand_id == formula.source_operand_ids[0]
            and bool(formula.direct_action_operand_ids)
            and formula.spd_operand_ids == expected_spd
            and not formula.frame_operand_ids
            and not formula.link_operand_ids
            and not formula.signed_summand_operand_ids
            and empty_child_sources
        )
    elif category == "cocycle":
        valid = (
            formula.dimension_operand_id is not None
            and len(formula.compared_operand_ids) == 2
            and formula.dimension_operand_id == formula.compared_operand_ids[0]
            and not formula.source_operand_ids
            and bool(formula.direct_action_operand_ids)
            and not formula.spd_operand_ids
            and len(formula.frame_operand_ids) >= 2
            and len(formula.link_operand_ids) >= 2
            and not formula.signed_summand_operand_ids
            and empty_child_sources
        )
    elif category == "density":
        valid = (
            formula.dimension_operand_id is not None
            and len(formula.compared_operand_ids) == 2
            and not formula.source_operand_ids
            and bool(formula.direct_action_operand_ids)
            and len(formula.spd_operand_ids) == 1
            and formula.dimension_operand_id == formula.spd_operand_ids[0]
            and not formula.frame_operand_ids
            and not formula.link_operand_ids
            and len(formula.signed_summand_operand_ids) == 2
            and empty_child_sources
        )
    elif category in ("local_term", "complete_objective"):
        allowed_children = (
            _LEAF_CHILD_CATEGORIES
            if category == "local_term"
            else frozenset({*_LEAF_CHILD_CATEGORIES, "local_term"})
        )
        valid = (
            formula.dimension_operand_id is None
            and len(formula.compared_operand_ids) == 2
            and not formula.source_operand_ids
            and not formula.direct_action_operand_ids
            and not formula.frame_operand_ids
            and not formula.link_operand_ids
            and bool(formula.signed_summand_operand_ids)
            and bool(formula.child_budgets)
            and all(
                item.budget.category in allowed_children
                for item in formula.child_budgets
            )
            and formula.forward_budget is None
            and formula.inverse_action_budget is None
        )
    elif category == "backward":
        valid = (
            formula.dimension_operand_id is None
            and len(formula.compared_operand_ids) == 3
            and not formula.source_operand_ids
            and bool(formula.direct_action_operand_ids)
            and not formula.spd_operand_ids
            and not formula.frame_operand_ids
            and not formula.link_operand_ids
            and not formula.signed_summand_operand_ids
            and not formula.child_budgets
            and formula.forward_budget is not None
            and formula.inverse_action_budget is not None
            and formula.forward_budget.budget.category in _LEAF_CHILD_CATEGORIES
            and formula.inverse_action_budget.budget.category in _LEAF_CHILD_CATEGORIES
            and formula.forward_budget.bound_sha256
            != formula.inverse_action_budget.bound_sha256
        )
    else:
        valid = False
    if not valid:
        raise ValueError(f"{category} formula has the wrong closed operand groups")


def _require_disjoint_formula_groups(formula: H7BudgetFormula) -> None:
    groups = (
        formula.compared_operand_ids,
        formula.source_operand_ids,
        formula.direct_action_operand_ids,
        formula.spd_operand_ids,
        formula.frame_operand_ids,
        formula.link_operand_ids,
        formula.signed_summand_operand_ids,
    )
    occurrences: dict[str, int] = {}
    for operand_id in (item for group in groups for item in group):
        occurrences[operand_id] = occurrences.get(operand_id, 0) + 1
    allowed_twice = (
        set(formula.source_operand_ids)
        if formula.category in {"covariance", "precision", "second_moment"}
        else set()
    )
    if any(
        count != (2 if operand_id in allowed_twice else 1)
        for operand_id, count in occurrences.items()
    ):
        raise ValueError(
            "semantic operand groups must be disjoint except for the named "
            "matrix source/SPD operand"
        )


def _resolve_formula_scope(
    category: H7BudgetCategory,
    operands: tuple[H7OperandRecord, ...],
    formula: H7BudgetFormula,
) -> dict[str, H7OperandRecord]:
    _require_local_operands(category, operands)
    if type(formula) is not H7BudgetFormula or formula.category != category:
        raise ValueError("formula category must equal the budget category")
    formula.__post_init__()
    by_id = {item.operand_id: item for item in operands}
    groups = {
        "compared": formula.compared_operand_ids,
        "source": formula.source_operand_ids,
        "direct-action": formula.direct_action_operand_ids,
        "SPD": formula.spd_operand_ids,
        "frame": formula.frame_operand_ids,
        "link": formula.link_operand_ids,
        "signed-summand": formula.signed_summand_operand_ids,
    }
    resolved = {
        name: _resolve_named_operands(by_id, operand_ids, name)
        for name, operand_ids in groups.items()
    }
    if formula.dimension_operand_id is not None:
        _resolve_named_operands(
            by_id,
            (formula.dimension_operand_id,),
            "dimension",
        )
    oracle_ids = tuple(
        (
            ()
            if formula.quadrature_operand_ids is None
            else formula.quadrature_operand_ids
        )
        + (
            ()
            if formula.reference_operand_id is None
            else (formula.reference_operand_id,)
        )
    )
    for operand_id in oracle_ids:
        _named_oracle_operand(operands, operand_id)
    consumed_ids = {
        *(
            ()
            if formula.dimension_operand_id is None
            else (formula.dimension_operand_id,)
        ),
        *(item for operand_ids in groups.values() for item in operand_ids),
        *oracle_ids,
    }
    if set(by_id) != consumed_ids:
        raise ValueError(
            "budget operands are missing, extra, or outside the closed scope"
        )
    _validate_scope_roles_and_shapes(category, formula, resolved, by_id)
    return by_id


def _validate_scope_roles_and_shapes(
    category: H7BudgetCategory,
    formula: H7BudgetFormula,
    groups: dict[str, tuple[H7OperandRecord, ...]],
    by_id: dict[str, H7OperandRecord],
) -> None:
    compared = groups["compared"]
    expected_roles = (
        ("original", "transformed", "recovered")
        if category == "backward"
        else ("original", "transformed")
    )
    if category == "cocycle":
        if compared[0].role not in ("original", "transformed"):
            raise ValueError("cocycle composite must be an evaluated output")
        if compared[1].role != "reference":
            raise ValueError("cocycle endpoint must be a reference")
    elif tuple(item.role for item in compared) != expected_roles:
        raise ValueError("compared operands have the wrong exact roles/order")
    _require_reference_group(groups["source"], "source")
    _require_reference_group(groups["direct-action"], "direct action", square=True)
    _require_reference_group(groups["SPD"], "SPD", square=True)
    _require_reference_group(groups["frame"], "frame", square=True)
    _require_reference_group(
        groups["signed-summand"],
        "signed summand",
        scalar=True,
    )
    if any(item.role not in ("original", "reference") for item in groups["link"]):
        raise ValueError("link operands must be original/reference values")

    if category in (*_VECTOR_CATEGORIES, *_MATRIX_CATEGORIES, "cocycle", "density"):
        if formula.dimension_operand_id is None:
            raise ValueError("leaf formula lacks a dimension operand")
        dimension = _dimension_from_shape(
            category,
            by_id[formula.dimension_operand_id].shape,
        )
        if category in (*_VECTOR_CATEGORIES, *_MATRIX_CATEGORIES):
            source = groups["source"][0]
            if any(item.shape != source.shape for item in compared):
                raise ValueError("leaf compared/source shapes disagree")
        if category == "cocycle":
            matrix_shape = (dimension, dimension)
            if any(
                item.shape != matrix_shape
                for item in (
                    *compared,
                    *groups["link"],
                    *groups["frame"],
                    *groups["direct-action"],
                )
            ):
                raise ValueError("cocycle operand dimensions disagree")
        elif category == "density":
            if any(not _is_scalar_shape(item.shape) for item in compared):
                raise ValueError("density compared outputs must be scalars")
            if groups["SPD"][0].shape != (dimension, dimension) or any(
                item.shape != (dimension, dimension) for item in groups["direct-action"]
            ):
                raise ValueError("density condition dimensions disagree")
        else:
            if any(
                item.shape != (dimension, dimension) for item in groups["direct-action"]
            ):
                raise ValueError("direct-action dimension disagrees with source")
    elif category in ("local_term", "complete_objective"):
        if any(not _is_scalar_shape(item.shape) for item in compared):
            raise ValueError("ELBO compared outputs must be scalars")
    elif category == "backward":
        if len({item.shape for item in compared}) != 1:
            raise ValueError("backward original/transformed/recovered shapes disagree")


def _expected_contributions(
    invariant_id: str,
    operands: tuple[H7OperandRecord, ...],
    formula: H7BudgetFormula,
    operation_count: int,
    condition_product: float,
    scale: float,
) -> tuple[H7AllowanceContribution, ...]:
    operation_unit = ROUNDING_CONSTANT * gamma_n(operation_count)
    rounding_operation_kind: H7OperationKind = (
        "pair_comparison"
        if formula.operation_kind == "gauss_hermite"
        else formula.operation_kind
    )
    contributions = [
        H7AllowanceContribution.create(
            kind="operation_rounding",
            operation_id=f"{invariant_id}:operation",
            operation_kind=rounding_operation_kind,
            operation_count=operation_count,
            quadrature_order=None,
            unit_allowance=operation_unit,
            value=operation_unit * condition_product * scale,
        )
    ]
    consumed_oracle_ids: set[str] = set()
    if formula.quadrature_operand_ids is not None:
        gh41 = _named_oracle_operand(operands, formula.quadrature_operand_ids[0])
        gh51 = _named_oracle_operand(operands, formula.quadrature_operand_ids[1])
        value41 = _decimal_value(gh41)
        value51 = _decimal_value(gh51)
        relative_delta = abs(value51 - value41) / max(Decimal(1), abs(value51))
        if relative_delta > MAX_ORACLE_RELATIVE_DELTA:
            raise ValueError(
                "GH41/GH51 boundary is unresolved under the frozen relative limit"
            )
        exact_convergence = 2 * abs(value51 - value41)
        contributions.append(
            H7AllowanceContribution.create(
                kind="quadrature_convergence",
                operation_id=f"{invariant_id}:quadrature:gh41-gh51",
                operation_kind="gauss_hermite",
                operation_count=gauss_hermite_2d_operation_count(51),
                quadrature_order=51,
                unit_allowance=2.0,
                value=float(exact_convergence),
            )
        )
        consumed_oracle_ids.update(formula.quadrature_operand_ids)
    if formula.reference_operand_id is not None:
        reference = _named_oracle_operand(operands, formula.reference_operand_id)
        reference_value = _decimal_value(reference)
        unit_allowance = 64.0 * EPS64
        contributions.append(
            H7AllowanceContribution.create(
                kind="reference_rounding",
                operation_id=(
                    f"{invariant_id}:reference:{formula.reference_operand_id}"
                ),
                operation_kind="pair_comparison",
                operation_count=max(1, math.prod(reference.shape)),
                quadrature_order=None,
                unit_allowance=unit_allowance,
                value=unit_allowance * max(1.0, abs(float(reference_value))),
            )
        )
        consumed_oracle_ids.add(formula.reference_operand_id)
    declared_oracle_ids = {
        item.operand_id for item in operands if item.role == "oracle"
    }
    if declared_oracle_ids != consumed_oracle_ids:
        raise ValueError("every oracle operand must be named by its contribution")
    return tuple(contributions)


def _validate_bound_budget_semantics(bound: H7BoundBudget) -> None:
    budget = bound.budget
    budget.__post_init__()
    formula = bound.formula
    if budget.category != formula.category:
        raise ValueError("bound budget category disagrees with its formula")
    operation_count = frozen_operation_count(
        budget.category,
        budget.operands,
        formula,
    )
    by_id = _resolve_formula_scope(
        budget.category,
        budget.operands,
        formula,
    )
    condition_product = math.prod(
        by_id[item].condition_number for item in formula.condition_operand_ids
    )
    scale = max(by_id[item].scale for item in formula.scale_operand_ids)
    normalization = max(by_id[item].normalization for item in formula.scale_operand_ids)
    if not all(
        math.isfinite(item) for item in (condition_product, scale, normalization)
    ):
        raise ValueError("bound budget geometry is not finite")
    expected = _expected_contributions(
        budget.invariant_id,
        budget.operands,
        formula,
        operation_count,
        condition_product,
        scale,
    )
    if (
        budget.contributions != expected
        or budget.comparison_normalization != normalization
        or budget.total_allowance != math.fsum(item.value for item in expected)
    ):
        raise ValueError("bound budget is not reproduced by its frozen formula")


def _bound_operation_count(bound: H7BoundBudget) -> int:
    if type(bound) is not H7BoundBudget:
        raise ValueError("operation provenance must be a bound budget")
    bound.__post_init__()
    operation_contributions = tuple(
        item for item in bound.budget.contributions if item.kind == "operation_rounding"
    )
    if len(operation_contributions) != 1:
        raise ValueError("bound budget must contain one operation-rounding count")
    return operation_contributions[0].operation_count


def _dimension_from_shape(
    category: H7BudgetCategory,
    shape: tuple[int, ...],
) -> int:
    if category in ("vector", "information", "offset"):
        if len(shape) == 1 and shape[0] > 0:
            return shape[0]
        if len(shape) == 2 and shape[1] == 1 and shape[0] > 0:
            return shape[0]
        raise ValueError("vector-like dimension operand has the wrong shape")
    if category == "decoder":
        if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
            raise ValueError("decoder dimension operand must have shape (V,n)")
        return shape[1]
    if len(shape) != 2 or shape[0] <= 0 or shape[0] != shape[1]:
        raise ValueError("matrix/density dimension operand must be square")
    return shape[0]


def _require_local_operands(
    category: H7BudgetCategory,
    operands: tuple[H7OperandRecord, ...],
) -> None:
    if (
        type(operands) is not tuple
        or not operands
        or any(type(item) is not H7OperandRecord for item in operands)
    ):
        raise ValueError("budget requires exact local operands")
    for item in operands:
        item.__post_init__()
        if item.category != category:
            raise ValueError("every local operand must share the budget category")
        if not all(
            math.isfinite(value)
            for value in (
                item.scale,
                item.condition_number,
                item.normalization,
            )
        ):
            raise ValueError("local operand geometry must be finite")
    operand_ids = tuple(item.operand_id for item in operands)
    if len(set(operand_ids)) != len(operand_ids):
        raise ValueError("local operand IDs must be unique")


def _require_id_tuple(
    name: str,
    operand_ids: tuple[str, ...],
    *,
    exact_length: int | None = None,
) -> None:
    if (
        type(operand_ids) is not tuple
        or (exact_length is not None and len(operand_ids) != exact_length)
        or any(type(item) is not str or not item for item in operand_ids)
        or len(set(operand_ids)) != len(operand_ids)
    ):
        raise ValueError(f"{name} must contain unique named operands")


def _resolve_named_operands(
    by_id: dict[str, H7OperandRecord],
    operand_ids: tuple[str, ...],
    purpose: str,
) -> tuple[H7OperandRecord, ...]:
    try:
        return tuple(by_id[operand_id] for operand_id in operand_ids)
    except KeyError as error:
        raise ValueError(
            f"{purpose} operand ID {error.args[0]!r} is not local"
        ) from error


def _require_reference_group(
    operands: tuple[H7OperandRecord, ...],
    name: str,
    *,
    square: bool = False,
    scalar: bool = False,
) -> None:
    if any(item.role != "reference" for item in operands):
        raise ValueError(f"{name} operands must be exact references")
    if square and any(
        len(item.shape) != 2 or item.shape[0] <= 0 or item.shape[0] != item.shape[1]
        for item in operands
    ):
        raise ValueError(f"{name} operands must be square matrices")
    if scalar and any(not _is_scalar_shape(item.shape) for item in operands):
        raise ValueError(f"{name} operands must be scalars")


def _is_scalar_shape(shape: tuple[int, ...]) -> bool:
    return shape in ((), (1,), (1, 1))


def _named_oracle_operand(
    operands: tuple[H7OperandRecord, ...],
    operand_id: str,
) -> H7OperandRecord:
    matches = tuple(item for item in operands if item.operand_id == operand_id)
    if len(matches) != 1 or matches[0].role != "oracle":
        raise ValueError(f"{operand_id} is not one exact named oracle operand")
    return matches[0]


def _decimal_value(operand: H7OperandRecord) -> Decimal:
    if operand.oracle_value is None:
        raise ValueError("oracle operand lacks its decimal value")
    try:
        value = Decimal(operand.oracle_value)
    except InvalidOperation as error:
        raise ValueError("oracle operand decimal is invalid") from error
    if not value.is_finite():
        raise ValueError("oracle operand decimal must be finite")
    return value


__all__ = [
    "CONTROL_ALLOWANCE_MULTIPLE",
    "CONTROL_MINIMUM_RELATIVE_RESIDUAL",
    "EPS64",
    "GH_DIMENSION",
    "H7BackwardBudgetAggregate",
    "H7BackwardOperandInput",
    "H7BackwardResidualRecord",
    "H7BoundBudget",
    "H7BudgetFormula",
    "H7OperandRecord",
    "MAX_ORACLE_RELATIVE_DELTA",
    "ROUNDING_CONSTANT",
    "build_h7_backward_records",
    "build_h7_budget",
    "control_decisiveness_limit",
    "frozen_operation_count",
    "gamma_n",
    "gauss_hermite_2d_operation_count",
    "require_control_decisive",
]
