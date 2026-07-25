"""Dependency-free H7 operand-budget protocol for the mpmath oracle.

This module deliberately uses only the Python standard library.  It owns the
minimal immutable records and leaf/backward constructors needed by the H7
oracle, so importing the oracle cannot transitively import production
``vfe4`` types, Torch, or NumPy.  The general verification layer may translate
these records into its production-facing H7 records after the independent
oracle boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields, is_dataclass
from decimal import Decimal, InvalidOperation
from typing import ClassVar, Literal, TypeVar, cast


H7BudgetCategory = Literal[
    "vector",
    "information",
    "offset",
    "decoder",
    "covariance",
    "precision",
    "second_moment",
    "map",
    "backward",
]
H7OperandRole = Literal[
    "original",
    "transformed",
    "reference",
    "recovered",
    "oracle",
]
H7OperationKind = Literal[
    "direct_solve",
    "matrix_product",
]
H7AllowanceKind = Literal["operation_rounding"]

EPS64 = 2.0**-52
ROUNDING_CONSTANT = 4096.0

_VECTOR_CATEGORIES = frozenset({"vector", "information", "offset", "decoder"})
_MATRIX_CATEGORIES = frozenset({"covariance", "precision", "second_moment", "map"})
_LEAF_CATEGORIES = _VECTOR_CATEGORIES | _MATRIX_CATEGORIES
_T = TypeVar("_T", bound="_IntegrityRecord")


def _canonical(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _canonical(getattr(value, item.name)) for item in fields(value)
        }
    if type(value) in (tuple, list):
        return [_canonical(item) for item in cast(tuple[object, ...], value)]
    if type(value) is dict:
        mapping = cast(dict[str, object], value)
        if any(type(key) is not str or not key for key in mapping):
            raise ValueError("protocol hash mappings require nonempty string keys")
        return {key: _canonical(mapping[key]) for key in sorted(mapping)}
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("protocol hash floats must be finite")
        return value.hex()
    if type(value) in (str, int, bool) or value is None:
        return value
    raise ValueError(f"unsupported H7 oracle-protocol value {type(value).__name__}")


def _owned_sha256(domain: str, value: object) -> str:
    if type(domain) is not str or not domain or not domain.isascii():
        raise ValueError("protocol hash domain must be nonempty ASCII")
    payload = json.dumps(
        _canonical(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(domain.encode("ascii") + b"\x00" + payload).hexdigest()


def _require_nonempty(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _require_sha256(value: object, name: str) -> str:
    digest = _require_nonempty(value, name)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return digest


class _IntegrityRecord:
    _integrity_field: ClassVar[str]
    _hash_domain: ClassVar[str]

    @classmethod
    def create(cls: type[_T], **values: object) -> _T:
        if cls._integrity_field in values:
            raise ValueError(
                f"{cls._integrity_field} is owned by {cls.__name__}.create"
            )
        semantic = dict(values)
        semantic[cls._integrity_field] = _owned_sha256(
            cls._hash_domain,
            values,
        )
        return cls(**semantic)  # type: ignore[arg-type]

    def _validate_integrity(self) -> None:
        semantic = {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name != self._integrity_field
        }
        observed = getattr(self, self._integrity_field)
        _require_sha256(observed, self._integrity_field)
        if observed != _owned_sha256(self._hash_domain, semantic):
            raise ValueError(
                f"{self._integrity_field} does not match {type(self).__name__}"
            )


@dataclass(frozen=True)
class H7OperandRecord(_IntegrityRecord):
    _integrity_field: ClassVar[str] = "operand_sha256"
    _hash_domain: ClassVar[str] = "vfe4.h7.operand.v1"

    operand_id: str
    category: H7BudgetCategory
    role: H7OperandRole
    dtype: str
    shape: tuple[int, ...]
    value_sha256: str
    scale: float
    condition_number: float
    normalization: float
    oracle_value: str | None
    operand_sha256: str

    def __post_init__(self) -> None:
        _require_nonempty(self.operand_id, "operand_id")
        _require_nonempty(self.dtype, "dtype")
        _require_sha256(self.value_sha256, "value_sha256")
        if self.category not in (*_LEAF_CATEGORIES, "backward"):
            raise ValueError("unsupported oracle-protocol operand category")
        if self.role not in (
            "original",
            "transformed",
            "reference",
            "recovered",
            "oracle",
        ):
            raise ValueError("unsupported oracle-protocol operand role")
        if (
            type(self.shape) is not tuple
            or any(type(size) is not int or size < 0 for size in self.shape)
            or type(self.scale) is not float
            or not math.isfinite(self.scale)
            or self.scale < 0.0
            or type(self.condition_number) is not float
            or not math.isfinite(self.condition_number)
            or self.condition_number < 1.0
            or type(self.normalization) is not float
            or not math.isfinite(self.normalization)
            or self.normalization <= 0.0
        ):
            raise ValueError("invalid oracle-protocol operand geometry")
        if self.oracle_value is not None:
            if type(self.oracle_value) is not str or not self.oracle_value:
                raise ValueError("oracle_value must be a nonempty decimal")
            try:
                value = Decimal(self.oracle_value)
            except InvalidOperation as error:
                raise ValueError("oracle_value must be an exact decimal") from error
            if not value.is_finite():
                raise ValueError("oracle_value must be finite")
        if (self.role == "oracle") != (self.oracle_value is not None):
            raise ValueError("only oracle operands carry oracle_value")
        self._validate_integrity()


@dataclass(frozen=True)
class H7AllowanceContribution(_IntegrityRecord):
    _integrity_field: ClassVar[str] = "contribution_sha256"
    _hash_domain: ClassVar[str] = "vfe4.h7.allowance-contribution.v1"

    kind: H7AllowanceKind
    operation_id: str
    operation_kind: H7OperationKind
    operation_count: int
    quadrature_order: None
    unit_allowance: float
    value: float
    contribution_sha256: str

    def __post_init__(self) -> None:
        if self.kind != "operation_rounding":
            raise ValueError("oracle protocol permits only operation rounding")
        _require_nonempty(self.operation_id, "operation_id")
        if self.operation_kind not in ("direct_solve", "matrix_product"):
            raise ValueError("unsupported oracle-protocol operation")
        if (
            type(self.operation_count) is not int
            or self.operation_count <= 0
            or self.quadrature_order is not None
            or type(self.unit_allowance) is not float
            or not math.isfinite(self.unit_allowance)
            or self.unit_allowance < 0.0
            or type(self.value) is not float
            or not math.isfinite(self.value)
            or self.value < 0.0
        ):
            raise ValueError("invalid oracle-protocol allowance contribution")
        self._validate_integrity()


@dataclass(frozen=True)
class H7BudgetRecord(_IntegrityRecord):
    _integrity_field: ClassVar[str] = "budget_sha256"
    _hash_domain: ClassVar[str] = "vfe4.h7.budget.v1"

    invariant_id: str
    category: H7BudgetCategory
    operands: tuple[H7OperandRecord, ...]
    contributions: tuple[H7AllowanceContribution, ...]
    comparison_normalization: float
    total_allowance: float
    budget_sha256: str

    def __post_init__(self) -> None:
        _require_nonempty(self.invariant_id, "invariant_id")
        if (
            self.category not in (*_LEAF_CATEGORIES, "backward")
            or type(self.operands) is not tuple
            or not self.operands
            or any(
                type(item) is not H7OperandRecord or item.category != self.category
                for item in self.operands
            )
            or len({item.operand_id for item in self.operands}) != len(self.operands)
            or type(self.contributions) is not tuple
            or len(self.contributions) != 1
            or type(self.contributions[0]) is not H7AllowanceContribution
            or type(self.comparison_normalization) is not float
            or not math.isfinite(self.comparison_normalization)
            or self.comparison_normalization <= 0.0
            or type(self.total_allowance) is not float
            or not math.isfinite(self.total_allowance)
            or self.total_allowance < 0.0
        ):
            raise ValueError("invalid oracle-protocol budget")
        for item in self.operands:
            item.__post_init__()
        self.contributions[0].__post_init__()
        expected = math.fsum(item.value for item in self.contributions)
        if not math.isclose(
            self.total_allowance,
            expected,
            rel_tol=0.0,
            abs_tol=max(math.ulp(expected), math.ulp(self.total_allowance)),
        ):
            raise ValueError("budget allowance must equal its contributions")
        self._validate_integrity()


@dataclass(frozen=True)
class H7BackwardResidualRecord(_IntegrityRecord):
    _integrity_field: ClassVar[str] = "backward_sha256"
    _hash_domain: ClassVar[str] = "vfe4.h7.backward-residual.v1"

    operand_id: str
    original_sha256: str
    transformed_sha256: str
    recovered_sha256: str
    numerator: float
    normalization: float
    value: float
    budget: H7BudgetRecord
    passed: bool
    backward_sha256: str

    def __post_init__(self) -> None:
        _require_nonempty(self.operand_id, "operand_id")
        for name in (
            "original_sha256",
            "transformed_sha256",
            "recovered_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if (
            type(self.numerator) is not float
            or not math.isfinite(self.numerator)
            or self.numerator < 0.0
            or type(self.normalization) is not float
            or not math.isfinite(self.normalization)
            or self.normalization <= 0.0
            or type(self.value) is not float
            or not math.isfinite(self.value)
            or self.value < 0.0
            or type(self.budget) is not H7BudgetRecord
            or self.budget.category != "backward"
            or type(self.passed) is not bool
        ):
            raise ValueError("invalid oracle-protocol backward residual")
        expected = self.numerator / self.normalization
        if not math.isclose(
            self.value,
            expected,
            rel_tol=8.0 * EPS64,
            abs_tol=0.0,
        ):
            raise ValueError("backward value must be numerator/normalization")
        if self.passed != (self.value <= self.budget.total_allowance):
            raise ValueError("backward pass flag disagrees with its budget")
        self.budget.__post_init__()
        self._validate_integrity()


@dataclass(frozen=True)
class H7BudgetFormula:
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
    quadrature_operand_ids: None = None
    reference_operand_id: None = None

    def __post_init__(self) -> None:
        if self.category not in (*_LEAF_CATEGORIES, "backward"):
            raise ValueError("unsupported oracle-protocol formula category")
        if self.operation_kind not in ("direct_solve", "matrix_product"):
            raise ValueError("unsupported oracle-protocol formula operation")
        if self.dimension_operand_id is not None:
            _require_nonempty(
                self.dimension_operand_id,
                "dimension_operand_id",
            )
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
        if (
            type(self.child_budgets) is not tuple
            or self.child_budgets
            or self.quadrature_operand_ids is not None
            or self.reference_operand_id is not None
        ):
            raise ValueError("oracle leaf/backward formula has unsupported sources")
        if self.category in _LEAF_CATEGORIES:
            valid = (
                self.dimension_operand_id is not None
                and len(self.compared_operand_ids) == 2
                and len(self.source_operand_ids) == 1
                and self.dimension_operand_id == self.source_operand_ids[0]
                and bool(self.direct_action_operand_ids)
                and not self.frame_operand_ids
                and not self.link_operand_ids
                and not self.signed_summand_operand_ids
                and self.forward_budget is None
                and self.inverse_action_budget is None
                and self.spd_operand_ids
                == (
                    self.source_operand_ids
                    if self.category in {"covariance", "precision", "second_moment"}
                    else ()
                )
            )
        else:
            valid = (
                self.operation_kind == "direct_solve"
                and self.dimension_operand_id is None
                and len(self.compared_operand_ids) == 3
                and not self.source_operand_ids
                and bool(self.direct_action_operand_ids)
                and not self.spd_operand_ids
                and not self.frame_operand_ids
                and not self.link_operand_ids
                and not self.signed_summand_operand_ids
                and type(self.forward_budget) is H7BoundBudget
                and type(self.inverse_action_budget) is H7BoundBudget
                and self.forward_budget.bound_sha256
                != self.inverse_action_budget.bound_sha256
            )
        if not valid:
            raise ValueError(
                f"{self.category} formula has the wrong closed operand groups"
            )
        if self.forward_budget is not None:
            self.forward_budget.__post_init__()
        if self.inverse_action_budget is not None:
            self.inverse_action_budget.__post_init__()
        _require_disjoint_formula_groups(self)

    @property
    def condition_operand_ids(self) -> tuple[str, ...]:
        if self.category in _VECTOR_CATEGORIES:
            return self.direct_action_operand_ids
        if self.category in _MATRIX_CATEGORIES:
            return (*self.direct_action_operand_ids, *self.spd_operand_ids)
        if self.category == "backward":
            return self.direct_action_operand_ids
        raise ValueError("unsupported oracle-protocol formula category")

    @property
    def scale_operand_ids(self) -> tuple[str, ...]:
        if self.category in _LEAF_CATEGORIES:
            return (*self.compared_operand_ids, *self.source_operand_ids)
        if self.category == "backward":
            return self.compared_operand_ids
        raise ValueError("unsupported oracle-protocol formula category")

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
            "child_bound_sha256": (),
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
            "quadrature_operand_ids": None,
            "reference_operand_id": None,
        }
        return hashlib.sha256(
            b"vfe4.h7.budget-formula.v2\x00"
            + json.dumps(
                _canonical(semantic),
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class H7BoundBudget:
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
            raise ValueError("bound oracle-protocol budget is inconsistent")
        _validate_bound_budget_semantics(self)


@dataclass(frozen=True)
class H7BackwardOperandInput:
    operand_id: str
    original_sha256: str
    transformed_sha256: str
    recovered_sha256: str
    numerator: float
    normalization: float
    operands: tuple[H7OperandRecord, ...]
    formula: H7BudgetFormula

    def __post_init__(self) -> None:
        _require_nonempty(self.operand_id, "operand_id")
        for name in (
            "original_sha256",
            "transformed_sha256",
            "recovered_sha256",
        ):
            _require_sha256(getattr(self, name), name)
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
            raise ValueError("invalid backward operand input")
        by_id = _resolve_formula_scope("backward", self.operands, self.formula)
        expected = {
            "original": self.original_sha256,
            "transformed": self.transformed_sha256,
            "recovered": self.recovered_sha256,
        }
        for role, digest in expected.items():
            matches = tuple(item for item in by_id.values() if item.role == role)
            if len(matches) != 1 or matches[0].value_sha256 != digest:
                raise ValueError("backward hashes do not bind local operands")


@dataclass(frozen=True)
class H7BackwardBudgetAggregate:
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
            raise ValueError(
                "backward aggregate must retain every exact protocol record"
            )
        for record, bound in zip(
            self.records,
            self.bound_budgets,
            strict=True,
        ):
            record.__post_init__()
            bound.__post_init__()


def _bound_budget_sha256(budget_sha256: str, formula_sha256: str) -> str:
    return hashlib.sha256(
        b"vfe4.h7.bound-budget.v2\x00"
        + budget_sha256.encode("ascii")
        + b"\x00"
        + formula_sha256.encode("ascii")
    ).hexdigest()


def gamma_n(operation_count: int) -> float:
    if type(operation_count) is not int or operation_count <= 0:
        raise ValueError("operation_count must be positive")
    product = operation_count * EPS64
    if product >= 1.0:
        raise ValueError("operation_count exceeds the binary64 gamma domain")
    return product / (1.0 - product)


def frozen_operation_count(
    category: H7BudgetCategory,
    operands: tuple[H7OperandRecord, ...],
    formula: H7BudgetFormula,
) -> int:
    scope = _resolve_formula_scope(category, operands, formula)
    if category in _LEAF_CATEGORIES:
        if formula.dimension_operand_id is None:
            raise ValueError("leaf formula lacks a dimension operand")
        n = _dimension_from_shape(
            category,
            scope[formula.dimension_operand_id].shape,
        )
        count = (
            32 * n + 64
            if category in _VECTOR_CATEGORIES
            else 64 * n**3 + 128 * n**2 + 64 * n + 256
        )
    elif category == "backward":
        if formula.forward_budget is None or formula.inverse_action_budget is None:
            raise ValueError("backward formula lacks operation provenance")
        count = _bound_operation_count(formula.forward_budget) + _bound_operation_count(
            formula.inverse_action_budget
        )
    else:
        raise ValueError("unsupported oracle-protocol budget category")
    if count <= 0:
        raise ValueError("frozen operation count must be positive")
    return count


def build_h7_budget(
    *,
    invariant_id: str,
    category: H7BudgetCategory,
    operands: tuple[H7OperandRecord, ...],
    formula: H7BudgetFormula,
) -> H7BoundBudget:
    _require_nonempty(invariant_id, "invariant_id")
    if type(formula) is not H7BudgetFormula or formula.category != category:
        raise ValueError("formula category must equal budget category")
    operation_count = frozen_operation_count(category, operands, formula)
    by_id = _resolve_formula_scope(category, operands, formula)
    condition_product = math.prod(
        by_id[item].condition_number for item in formula.condition_operand_ids
    )
    scale = max(by_id[item].scale for item in formula.scale_operand_ids)
    normalization = max(by_id[item].normalization for item in formula.scale_operand_ids)
    operation_unit = ROUNDING_CONSTANT * gamma_n(operation_count)
    contribution = H7AllowanceContribution.create(
        kind="operation_rounding",
        operation_id=f"{invariant_id}:operation",
        operation_kind=formula.operation_kind,
        operation_count=operation_count,
        quadrature_order=None,
        unit_allowance=operation_unit,
        value=operation_unit * condition_product * scale,
    )
    budget = H7BudgetRecord.create(
        invariant_id=invariant_id,
        category=category,
        operands=operands,
        contributions=(contribution,),
        comparison_normalization=normalization,
        total_allowance=contribution.value,
    )
    return H7BoundBudget.create(budget, formula)


def build_h7_backward_records(
    inputs: tuple[H7BackwardOperandInput, ...],
    *,
    required_operand_ids: tuple[str, ...],
) -> H7BackwardBudgetAggregate:
    if (
        type(inputs) is not tuple
        or not inputs
        or any(type(item) is not H7BackwardOperandInput for item in inputs)
        or type(required_operand_ids) is not tuple
        or not required_operand_ids
        or any(type(item) is not str or not item for item in required_operand_ids)
        or len(set(required_operand_ids)) != len(required_operand_ids)
    ):
        raise ValueError("backward construction requires an exact inventory")
    if tuple(item.operand_id for item in inputs) != required_operand_ids:
        raise ValueError("backward operand IDs are missing, extra, or reordered")
    records: list[H7BackwardResidualRecord] = []
    budgets: list[H7BoundBudget] = []
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
        budgets.append(bound)
    frozen = tuple(records)
    return H7BackwardBudgetAggregate(
        records=frozen,
        bound_budgets=tuple(budgets),
        maximum=max(item.value for item in frozen),
    )


def _validate_bound_budget_semantics(bound: H7BoundBudget) -> None:
    budget = bound.budget
    formula = bound.formula
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
    operation_unit = ROUNDING_CONSTANT * gamma_n(operation_count)
    expected = H7AllowanceContribution.create(
        kind="operation_rounding",
        operation_id=f"{budget.invariant_id}:operation",
        operation_kind=formula.operation_kind,
        operation_count=operation_count,
        quadrature_order=None,
        unit_allowance=operation_unit,
        value=operation_unit * condition_product * scale,
    )
    if (
        budget.contributions != (expected,)
        or budget.comparison_normalization != normalization
        or budget.total_allowance != expected.value
    ):
        raise ValueError("bound budget is not reproduced by its formula")


def _bound_operation_count(bound: H7BoundBudget) -> int:
    if type(bound) is not H7BoundBudget:
        raise ValueError("operation provenance must be a bound budget")
    bound.__post_init__()
    return bound.budget.contributions[0].operation_count


def _resolve_formula_scope(
    category: H7BudgetCategory,
    operands: tuple[H7OperandRecord, ...],
    formula: H7BudgetFormula,
) -> dict[str, H7OperandRecord]:
    if (
        type(operands) is not tuple
        or not operands
        or any(
            type(item) is not H7OperandRecord or item.category != category
            for item in operands
        )
        or len({item.operand_id for item in operands}) != len(operands)
        or type(formula) is not H7BudgetFormula
        or formula.category != category
    ):
        raise ValueError("budget requires exact category-local operands")
    for item in operands:
        item.__post_init__()
    formula.__post_init__()
    by_id = {item.operand_id: item for item in operands}
    declared_ids = {
        *formula.compared_operand_ids,
        *formula.source_operand_ids,
        *formula.direct_action_operand_ids,
        *formula.spd_operand_ids,
    }
    if formula.dimension_operand_id is not None:
        declared_ids.add(formula.dimension_operand_id)
    if set(by_id) != declared_ids:
        raise ValueError("budget operands are outside the closed formula scope")
    compared = tuple(by_id[item] for item in formula.compared_operand_ids)
    expected_roles = (
        ("original", "transformed", "recovered")
        if category == "backward"
        else ("original", "transformed")
    )
    if tuple(item.role for item in compared) != expected_roles:
        raise ValueError("compared operands have the wrong roles/order")
    for operand_id in (
        *formula.source_operand_ids,
        *formula.direct_action_operand_ids,
        *formula.spd_operand_ids,
    ):
        if by_id[operand_id].role != "reference":
            raise ValueError("formula sources/actions must be references")
    if category in _LEAF_CATEGORIES:
        if formula.dimension_operand_id is None:
            raise ValueError("leaf formula lacks a dimension operand")
        source = by_id[formula.source_operand_ids[0]]
        if any(item.shape != source.shape for item in compared):
            raise ValueError("leaf compared/source shapes disagree")
        dimension = _dimension_from_shape(category, source.shape)
        if any(
            by_id[item].shape != (dimension, dimension)
            for item in formula.direct_action_operand_ids
        ):
            raise ValueError("leaf action dimension disagrees with source")
    elif len({item.shape for item in compared}) != 1:
        raise ValueError("backward operand shapes disagree")
    return by_id


def _dimension_from_shape(
    category: H7BudgetCategory,
    shape: tuple[int, ...],
) -> int:
    if category in ("vector", "information", "offset"):
        if len(shape) == 1 and shape[0] > 0:
            return shape[0]
        if len(shape) == 2 and shape[1] == 1 and shape[0] > 0:
            return shape[0]
        raise ValueError("vector-like operand has the wrong shape")
    if category == "decoder":
        if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
            raise ValueError("decoder operand must have shape (V,n)")
        return shape[1]
    if len(shape) != 2 or shape[0] <= 0 or shape[0] != shape[1]:
        raise ValueError("matrix operand must be square")
    return shape[0]


def _require_id_tuple(name: str, values: tuple[str, ...]) -> None:
    if (
        type(values) is not tuple
        or any(type(item) is not str or not item for item in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(f"{name} must contain unique named operands")


def _require_disjoint_formula_groups(formula: H7BudgetFormula) -> None:
    groups = (
        formula.compared_operand_ids,
        formula.source_operand_ids,
        formula.direct_action_operand_ids,
        formula.spd_operand_ids,
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
        raise ValueError("formula semantic operand groups overlap")


__all__ = [
    "EPS64",
    "H7AllowanceContribution",
    "H7BackwardBudgetAggregate",
    "H7BackwardOperandInput",
    "H7BackwardResidualRecord",
    "H7BoundBudget",
    "H7BudgetFormula",
    "H7BudgetRecord",
    "H7OperandRecord",
    "ROUNDING_CONSTANT",
    "build_h7_backward_records",
    "build_h7_budget",
    "frozen_operation_count",
    "gamma_n",
]
