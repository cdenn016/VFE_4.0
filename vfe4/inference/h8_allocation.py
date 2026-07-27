"""Fail-closed H8 allocation observation and negative-control guards.

Logical H8 shapes and physical fixture shapes are intentionally separate.
Classification always uses the frozen logical production layout; storage
liveness uses the unique physical storage spans actually observed.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import traceback
import weakref
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor
from torch.utils._python_dispatch import TorchDispatchMode

from vfe4.numerics.block_layout import (
    H8_MAX_STORAGE_SCALARS,
    BlockChainLayout,
)
from vfe4.types.h8 import (
    BackendCounterSnapshot,
    BlockStorageRecord,
    BlockWorkspaceRecord,
    H8ControlResult,
    H8LossyProfilerRow,
    H8ProfilerAction,
    H8ProfilerEventRecord,
    H8TensorKey,
)
from vfe4.types.results import GateStatus


H8SemanticKind = Literal[
    "scalar_local_channel",
    "block_vector",
    "block_rhs",
    "block_diagonal",
    "block_lower",
    "generator_objective",
]


class H8ForbiddenAllocation(RuntimeError):
    """A witnessed forbidden allocation, operation, or liveness transition."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class H8ProfilerObservabilityGap(RuntimeError):
    """Profiler evidence cannot be joined or enriched losslessly."""


@dataclass(frozen=True, slots=True)
class H8AllocationDecision:
    site: str
    logical_shape: tuple[int, ...]
    operator: str
    classification: str
    float64_equivalent_scalars: int


@dataclass(frozen=True, slots=True)
class H8StorageSpan:
    storage_key: str
    device: str
    pointer: int
    span_start: int
    span_end: int
    nbytes: int

    def __post_init__(self) -> None:
        if not self.storage_key:
            raise ValueError("storage_key must be nonempty")
        if not self.device:
            raise ValueError("device must be nonempty")
        for name in ("pointer", "span_start", "span_end", "nbytes"):
            _nonnegative_int(getattr(self, name), name)
        if self.span_start != self.pointer:
            raise ValueError("storage span must start at its pointer")
        if self.span_end - self.span_start != self.nbytes:
            raise ValueError("storage span must equal nbytes")


@dataclass(frozen=True, slots=True)
class H8DispatchEvent:
    sequence: int
    operator: str
    semantic_site: str | None
    control_id: str | None
    input_shapes: tuple[tuple[int, ...], ...]
    output_shapes: tuple[tuple[int, ...], ...]
    physical_output_shapes: tuple[tuple[int, ...], ...]
    stack_member_shapes: tuple[tuple[int, ...], ...]
    stack_member_count: int
    dtype: str | None
    device: str | None
    float64_equivalent_scalars: int
    classifications: tuple[str, ...]
    storage_spans: tuple[H8StorageSpan, ...]
    alias_storage_keys: tuple[str, ...]
    new_storage_keys: tuple[str, ...]
    allocated_float64_equivalent_scalars: int
    live_float64_equivalent_scalars_by_site: tuple[tuple[str, int], ...]
    stack: tuple[str, ...]
    executed: bool
    forbidden_reason: str | None
    live_storage_bytes_after: int
    population_live_storage_bytes_after: int


@dataclass(frozen=True, slots=True)
class H8NumpyGuardEvent:
    sequence: int
    operator: str
    semantic_site: str | None
    control_id: str | None
    input_shapes: tuple[tuple[int, ...], ...]
    output_shapes: tuple[tuple[int, ...], ...]
    dtype: str | None
    float64_equivalent_scalars: int
    executed: bool
    forbidden_reason: str | None


@dataclass(frozen=True, slots=True)
class H8RawProfilerEvent:
    source_row_index: int
    timestamp_ns: int
    action: H8ProfilerAction
    tensor_key: H8TensorKey
    version: int
    nbytes: int

    def __post_init__(self) -> None:
        _nonnegative_int(self.source_row_index, "source_row_index")
        if self.action not in _PROFILER_ACTIONS:
            raise ValueError("action is outside the frozen profiler union")
        if (
            type(self.timestamp_ns) is not int
            or self.timestamp_ns < -1
            or (self.timestamp_ns == -1 and self.action != "PREEXISTING")
        ):
            raise ValueError(
                "timestamp_ns must be nonnegative except -1 PREEXISTING"
            )
        if type(self.tensor_key) is not H8TensorKey:
            raise ValueError("tensor_key must be H8TensorKey")
        _nonnegative_int(self.version, "version")
        if type(self.nbytes) is not int or type(self.nbytes) is bool:
            raise ValueError("nbytes must be a signed integer")


@dataclass(frozen=True, slots=True)
class H8ProfilerEnrichment:
    source_row_index: int
    tensor_key: H8TensorKey
    version: int
    dtype: str
    operator: str
    stack: tuple[str, ...]
    logical_shape: tuple[int, ...]
    classification: str
    matched_event_node_indices: tuple[int, ...]
    storage_span_start: int
    storage_span_end: int
    storage_nbytes: int
    alias_of: H8TensorKey | None = None

    def __post_init__(self) -> None:
        _nonnegative_int(self.source_row_index, "source_row_index")
        if type(self.tensor_key) is not H8TensorKey:
            raise ValueError("tensor_key must be H8TensorKey")
        _nonnegative_int(self.version, "version")
        for name in ("dtype", "operator", "classification"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be nonempty")
        if (
            type(self.stack) is not tuple
            or not self.stack
            or any(type(frame) is not str or not frame for frame in self.stack)
        ):
            raise ValueError("stack must contain nonempty frames")
        _logical_shape(self.logical_shape)
        if (
            type(self.matched_event_node_indices) is not tuple
            or not self.matched_event_node_indices
            or len(set(self.matched_event_node_indices))
            != len(self.matched_event_node_indices)
            or any(
                type(index) is not int or index < 0
                for index in self.matched_event_node_indices
            )
        ):
            raise ValueError("matched event nodes must be nonempty and unique")
        for name in (
            "storage_span_start",
            "storage_span_end",
            "storage_nbytes",
        ):
            _nonnegative_int(getattr(self, name), name)
        if self.storage_span_start != self.tensor_key.storage_ptr:
            raise ValueError("storage span must start at TensorKey.storage_ptr")
        if self.storage_span_end - self.storage_span_start != self.storage_nbytes:
            raise ValueError("storage span and storage_nbytes disagree")
        if self.storage_nbytes and self.storage_span_start == 0:
            raise ValueError("positive profiler storage needs a nonzero pointer")
        if self.alias_of is not None:
            if type(self.alias_of) is not H8TensorKey:
                raise ValueError("alias_of must be H8TensorKey when present")
            if self.alias_of == self.tensor_key:
                raise ValueError("a profiler identity cannot alias itself")


@dataclass(frozen=True, slots=True)
class H8ProfilerTrace:
    events: tuple[H8ProfilerEventRecord, ...]
    preexisting_storage_count: int
    preexisting_bytes: int
    baseline_live_bytes: int
    live_peak_bytes: int
    all_joined_and_liveness_reconciled: bool
    trace_sha256: str


@dataclass(frozen=True, slots=True)
class H8NegativeControlSpec:
    control_id: str
    requested_operation: str
    logical_shapes: tuple[tuple[int, ...], ...]
    assigned_channels: tuple[str, ...]
    expected_reason: str


@dataclass(frozen=True, slots=True)
class H8AllocationCrossCheck:
    complete: bool
    obligations: tuple[str, ...]
    backend_forbidden_attempt_count: int
    dispatch_forbidden_attempt_count: int
    reconciled_operation_counts: tuple[tuple[str, int, int], ...]


@dataclass(frozen=True, slots=True)
class _SiteSpec:
    kind: H8SemanticKind
    exact_shapes: frozenset[tuple[int, ...]]


@dataclass(frozen=True, slots=True)
class _CompositeOutputSpec:
    logical_shape: tuple[int, ...]
    site: str
    transient_site: str | None = None
    canonical_operators: frozenset[str] = frozenset()

    def resolve_site(self, operator: str) -> str:
        if self.transient_site is not None and operator not in self.canonical_operators:
            return self.transient_site
        return self.site


@dataclass(frozen=True, slots=True)
class _CompositeSiteSpec:
    operators: frozenset[str]
    input_shapes: frozenset[tuple[int, ...]]
    outputs: tuple[_CompositeOutputSpec, ...]
    default_site: str
    unregistered_input_operators: frozenset[str] = frozenset()

    def output_site(
        self,
        *,
        operator: str,
        logical_shape: tuple[int, ...],
    ) -> str:
        matches = tuple(
            output.resolve_site(operator)
            for output in self.outputs
            if output.logical_shape == logical_shape
        )
        if len(matches) != 1:
            raise H8ForbiddenAllocation(
                "unregistered output shape "
                f"{logical_shape!r} for scoped operator {operator!r}"
            )
        return matches[0]


@dataclass(frozen=True, slots=True)
class _DispatchSemanticScope:
    site: str
    logical_output_shapes: tuple[tuple[int, ...], ...]
    composite: bool


_PROFILER_ACTIONS = (
    "PREEXISTING",
    "CREATE",
    "INCREMENT_VERSION",
    "DESTROY",
)
_DENSE_OPERATORS = (
    "eigvalsh",
    "eigh",
    "svd",
    "cholesky",
    "inverse",
    "linalg_inv",
)
_VIEW_OPERATORS = (
    "view",
    "reshape",
    "as_strided",
    "transpose",
    "permute",
    "slice",
    "select",
    "detach",
    "alias",
)
_STACK_OPERATORS = ("stack",)
_CONCATENATE_OPERATORS = ("cat", "concatenate")
_FACTORY_OPERATORS = ("empty", "zeros", "ones", "full")
_H8_COMPOSITE_SITE_NAMES = (
    "production.problem_build",
    "production.assembly",
    "production.factorization",
    "production.mean_solve",
    "production.forward_substitution",
    "production.backward_substitution",
    "production.logdet",
    "production.selected_inverse",
    "production.sample_width_one",
    "production.quadratic",
    "production.sparse_trace",
    "production.condition_estimate",
    "production.entropy",
    "production.log_normalizer",
    "production.complete_objective",
)

_TORCH_TENSORIZATION_OPERATORS = frozenset(
    {
        "aten::tensor",
        "aten::as_tensor",
        "aten::lift_fresh",
        "aten::_to_copy",
    }
)
_TORCH_STRUCTURAL_OPERATORS = frozenset(
    {
        *_TORCH_TENSORIZATION_OPERATORS,
        "aten::alias",
        "aten::as_strided",
        "aten::clone",
        "aten::contiguous",
        "aten::copy_",
        "aten::detach",
        "aten::fill_",
        "aten::index_put_",
        "aten::permute",
        "aten::reshape",
        "aten::select",
        "aten::select_scatter",
        "aten::slice",
        "aten::slice_scatter",
        "aten::squeeze",
        "aten::t",
        "aten::transpose",
        "aten::unbind",
        "aten::unsqueeze",
        "aten::view",
    }
)
_TORCH_FACTORY_OPERATORS = frozenset(
    {
        "aten::empty",
        "aten::empty_like",
        "aten::eye",
        "aten::full",
        "aten::ones",
        "aten::ones_like",
        "aten::zeros",
        "aten::zeros_like",
    }
)
_TORCH_VALIDATION_OPERATORS = frozenset(
    {
        "aten::_local_scalar_dense",
        "aten::abs",
        "aten::all",
        "aten::any",
        "aten::count_nonzero",
        "aten::equal",
        "aten::isfinite",
        "aten::max",
        "aten::min",
        "aten::triu",
    }
)
_TORCH_ARITHMETIC_OPERATORS = frozenset(
    {
        "aten::_log_softmax",
        "aten::add",
        "aten::add_",
        "aten::div",
        "aten::div_",
        "aten::eq",
        "aten::exp",
        "aten::ge",
        "aten::gt",
        "aten::le",
        "aten::log",
        "aten::log_softmax",
        "aten::logsumexp",
        "aten::lt",
        "aten::mul",
        "aten::mul_",
        "aten::ne",
        "aten::neg",
        "aten::sign",
        "aten::sqrt",
        "aten::sub",
        "aten::sub_",
        "aten::where",
    }
)
_TORCH_REDUCTION_OPERATORS = frozenset(
    {
        "aten::argmax",
        "aten::dot",
        "aten::sum",
        "aten::trace",
    }
)
_TORCH_LINEAR_OPERATORS = frozenset(
    {
        "aten::bmm",
        "aten::cholesky_solve",
        "aten::diagonal",
        "aten::einsum",
        "aten::ger",
        "aten::linalg_cholesky",
        "aten::linalg_cholesky_ex",
        "aten::linalg_solve_triangular",
        "aten::matmul",
        "aten::mm",
        "aten::mv",
        "aten::outer",
    }
)
_TORCH_COMBINE_OPERATORS = frozenset({"aten::cat", "aten::stack"})


def _nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _logical_shape(value: object) -> tuple[int, ...]:
    if type(value) is not tuple or any(
        type(dimension) is not int or dimension < 0 for dimension in value
    ):
        raise ValueError("logical shape must be a tuple of nonnegative integers")
    return value


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _operator_name(operator: object) -> str:
    schema = getattr(operator, "_schema", None)
    name = getattr(schema, "name", None)
    overload = getattr(schema, "overload_name", None)
    if isinstance(name, str) and name:
        return f"{name}.{overload}" if overload else name
    return str(operator)


def _operator_base(operator: str) -> str:
    return operator.partition(".")[0].lower()


def _shape_of(value: object) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        result = tuple(int(dimension) for dimension in shape)
    except (TypeError, ValueError):
        return None
    return result if all(dimension >= 0 for dimension in result) else None


def _walk_tensors(value: object) -> Iterable[Tensor]:
    if isinstance(value, Tensor):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _walk_tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _walk_tensors(item)


def _walk_arrays(
    value: object,
) -> Iterable[np.ndarray[Any, Any] | np.generic]:
    if isinstance(value, (np.ndarray, np.generic)):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _walk_arrays(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _walk_arrays(item)


def _float64_equivalent_scalars(
    shape: tuple[int, ...],
    itemsize: int,
) -> int:
    if type(itemsize) is not int or itemsize <= 0:
        itemsize = 8
    return math.ceil(math.prod(shape) * itemsize / 8)


class H8AllocationPolicy:
    """Closed semantic-site allocation policy for one frozen H8 layout."""

    def __init__(self, layout: BlockChainLayout) -> None:
        if type(layout) is not BlockChainLayout:
            raise ValueError("layout must be an exact BlockChainLayout")
        self.layout = layout
        self._sites: dict[str, _SiteSpec] = {}
        self._stack_sites: dict[str, str] = {}
        self._composite_sites: dict[str, _CompositeSiteSpec] = {}
        self._install_standard_sites()
        self._install_composite_sites()

    def _install_standard_sites(self) -> None:
        n = self.layout.population_size
        b = self.layout.block_size
        for name in ("scalar", "local", "channel", "workspace"):
            self._sites[name] = _SiteSpec(
                "scalar_local_channel",
                frozenset(),
            )
        for name in ("information", "block_vector", "sample"):
            self._sites[name] = _SiteSpec("block_vector", frozenset({(n, b)}))
        rhs_shapes = {(n, b)}
        rhs_shapes.update((n, b, width) for width in range(1, b + 1))
        self._sites["rhs"] = _SiteSpec("block_rhs", frozenset(rhs_shapes))
        for name in (
            "precision.diagonal",
            "factor.diagonal",
            "selected.diagonal",
        ):
            self._sites[name] = _SiteSpec(
                "block_diagonal",
                frozenset({(n, b, b)}),
            )
        for name in (
            "precision.lower",
            "factor.lower",
            "selected.lower",
        ):
            self._sites[name] = _SiteSpec(
                "block_lower",
                frozenset({(n - 1, b, b)}),
            )
        vector_sites = (
            "transient.assembly.information",
            "transient.factor.vector",
            "transient.logdet.vector",
            "gaussian.information",
        )
        for name in vector_sites:
            self._sites[name] = _SiteSpec(
                "block_vector",
                frozenset({(n, b)}),
            )
        self._sites["transient.condition.vector"] = _SiteSpec(
            "block_vector",
            frozenset({(n - 1, b)}),
        )
        for name in (
            "transient.assembly.diagonal",
            "transient.factor.diagonal",
            "transient.selected.diagonal",
            "transient.sparse.diagonal",
            "transient.condition.diagonal",
            "objective.diagonal",
        ):
            self._sites[name] = _SiteSpec(
                "block_diagonal",
                frozenset({(n, b, b)}),
            )
        for name in (
            "transient.assembly.lower",
            "transient.factor.lower",
            "transient.selected.lower",
            "transient.sparse.lower",
            "transient.condition.lower",
            "objective.lower",
        ):
            self._sites[name] = _SiteSpec(
                "block_lower",
                frozenset({(n - 1, b, b)}),
            )
        k = self.layout.d_z
        objective_population_shapes = frozenset(
            {
                (n - 1, 3),
                (n - 1, k),
                (n - 1, k, b),
                (n - 1, b, k),
                (n - 1, k, k),
                (n - 1, b),
                (n, k),
                (n, k, b),
                (n, b, k),
                (n, k, k),
            }
        )
        self._sites["objective.population"] = _SiteSpec(
            "generator_objective",
            objective_population_shapes,
        )

    def _install_composite_sites(self) -> None:
        n = self.layout.population_size
        b = self.layout.block_size
        k = self.layout.d_z
        local_shapes = {
            (),
            (0,),
            (1,),
            (3,),
            (17,),
            (21,),
            (k,),
            (b,),
            (1, 3),
            (3, 1),
            (k, 1),
            (1, k),
            (k, 3),
            (3, k),
            (b, 3),
            (3, b),
            (17, 1),
            (21, 1),
            (17, 3),
            (21, 3),
            (k, k),
            (k, b),
            (b, k),
            (b, b),
        }
        local_shapes.update((b, width) for width in range(1, b + 1))
        local_shapes.update((width, b) for width in range(1, b + 1))
        rhs_shapes = {
            (n, b),
            *((n, b, width) for width in range(1, b + 1)),
        }
        objective_population_shapes = set(
            self._sites["objective.population"].exact_shapes
        )
        population_shapes = {
            (n, b),
            (n - 1, b),
            (n, b, b),
            (n - 1, b, b),
            *rhs_shapes,
            *objective_population_shapes,
        }
        input_shapes = frozenset(local_shapes | population_shapes)
        all_operators = frozenset(
            {
                *_TORCH_STRUCTURAL_OPERATORS,
                *_TORCH_FACTORY_OPERATORS,
                *_TORCH_VALIDATION_OPERATORS,
                *_TORCH_ARITHMETIC_OPERATORS,
                *_TORCH_REDUCTION_OPERATORS,
                *_TORCH_LINEAR_OPERATORS,
                *_TORCH_COMBINE_OPERATORS,
            }
        )

        def local_outputs(site: str) -> tuple[_CompositeOutputSpec, ...]:
            return tuple(
                _CompositeOutputSpec(shape, site) for shape in sorted(local_shapes)
            )

        def rhs_outputs(site: str = "rhs") -> tuple[_CompositeOutputSpec, ...]:
            return tuple(
                _CompositeOutputSpec(shape, site) for shape in sorted(rhs_shapes)
            )

        canonical_clone = frozenset({"aten::clone", "aten::empty"})
        self._composite_sites = {
            "production.problem_build": _CompositeSiteSpec(
                operators=frozenset(
                    {
                        *_TORCH_TENSORIZATION_OPERATORS,
                        *_TORCH_STRUCTURAL_OPERATORS,
                        *_TORCH_FACTORY_OPERATORS,
                        *_TORCH_VALIDATION_OPERATORS,
                    }
                ),
                input_shapes=input_shapes,
                outputs=(
                    *local_outputs("local"),
                    _CompositeOutputSpec((n, b), "sample"),
                ),
                default_site="local",
                unregistered_input_operators=_TORCH_TENSORIZATION_OPERATORS,
            ),
            "production.assembly": _CompositeSiteSpec(
                operators=all_operators,
                input_shapes=input_shapes,
                outputs=(
                    *local_outputs("local"),
                    _CompositeOutputSpec(
                        (n, b),
                        "information",
                        "transient.assembly.information",
                        canonical_clone,
                    ),
                    _CompositeOutputSpec(
                        (n, b, b),
                        "precision.diagonal",
                        "transient.assembly.diagonal",
                        canonical_clone,
                    ),
                    _CompositeOutputSpec(
                        (n - 1, b, b),
                        "precision.lower",
                        "transient.assembly.lower",
                        canonical_clone,
                    ),
                ),
                default_site="local",
                unregistered_input_operators=_TORCH_TENSORIZATION_OPERATORS,
            ),
            "production.factorization": _CompositeSiteSpec(
                operators=all_operators,
                input_shapes=input_shapes,
                outputs=(
                    *local_outputs("workspace"),
                    _CompositeOutputSpec(
                        (n, b),
                        "transient.factor.vector",
                    ),
                    _CompositeOutputSpec(
                        (n, b, b),
                        "factor.diagonal",
                        "transient.factor.diagonal",
                        frozenset({"aten::clone"}),
                    ),
                    _CompositeOutputSpec(
                        (n - 1, b, b),
                        "factor.lower",
                        "transient.factor.lower",
                        frozenset({"aten::clone"}),
                    ),
                ),
                default_site="workspace",
            ),
            "production.mean_solve": _CompositeSiteSpec(
                operators=all_operators,
                input_shapes=input_shapes,
                outputs=(*local_outputs("workspace"), *rhs_outputs()),
                default_site="workspace",
            ),
            "production.forward_substitution": _CompositeSiteSpec(
                operators=all_operators,
                input_shapes=input_shapes,
                outputs=(*local_outputs("workspace"), *rhs_outputs()),
                default_site="workspace",
            ),
            "production.backward_substitution": _CompositeSiteSpec(
                operators=all_operators,
                input_shapes=input_shapes,
                outputs=(*local_outputs("workspace"), *rhs_outputs()),
                default_site="workspace",
            ),
            "production.logdet": _CompositeSiteSpec(
                operators=all_operators,
                input_shapes=input_shapes,
                outputs=(
                    *local_outputs("local"),
                    _CompositeOutputSpec(
                        (n, b),
                        "transient.logdet.vector",
                    ),
                ),
                default_site="local",
            ),
            "production.selected_inverse": _CompositeSiteSpec(
                operators=all_operators,
                input_shapes=input_shapes,
                outputs=(
                    *local_outputs("workspace"),
                    _CompositeOutputSpec(
                        (n, b, b),
                        "selected.diagonal",
                        "transient.selected.diagonal",
                        frozenset({"aten::clone"}),
                    ),
                    _CompositeOutputSpec(
                        (n - 1, b, b),
                        "selected.lower",
                        "transient.selected.lower",
                        frozenset({"aten::clone"}),
                    ),
                ),
                default_site="workspace",
            ),
            "production.sample_width_one": _CompositeSiteSpec(
                operators=all_operators,
                input_shapes=input_shapes,
                outputs=(
                    *local_outputs("workspace"),
                    *rhs_outputs("sample"),
                ),
                default_site="workspace",
            ),
            "production.quadratic": _CompositeSiteSpec(
                operators=all_operators,
                input_shapes=input_shapes,
                outputs=(
                    *local_outputs("local"),
                    _CompositeOutputSpec((n, b), "rhs"),
                ),
                default_site="local",
            ),
            "production.sparse_trace": _CompositeSiteSpec(
                operators=all_operators,
                input_shapes=input_shapes,
                outputs=(
                    *local_outputs("local"),
                    *(
                        _CompositeOutputSpec(shape, "rhs")
                        for shape in sorted(rhs_shapes - {(n, b, b)})
                    ),
                    _CompositeOutputSpec(
                        (n, b, b),
                        "transient.sparse.diagonal",
                    ),
                    _CompositeOutputSpec(
                        (n - 1, b, b),
                        "transient.sparse.lower",
                    ),
                ),
                default_site="local",
            ),
            "production.condition_estimate": _CompositeSiteSpec(
                operators=all_operators,
                input_shapes=input_shapes,
                outputs=(
                    *local_outputs("workspace"),
                    *(
                        _CompositeOutputSpec(
                            shape,
                            "transient.condition.vector",
                        )
                        for shape in ((n - 1, b),)
                    ),
                    _CompositeOutputSpec((n, b), "rhs"),
                    *(
                        _CompositeOutputSpec(shape, "rhs")
                        for shape in sorted(rhs_shapes - {(n, b), (n, b, b)})
                    ),
                    _CompositeOutputSpec(
                        (n, b, b),
                        "transient.condition.diagonal",
                    ),
                    _CompositeOutputSpec(
                        (n - 1, b, b),
                        "transient.condition.lower",
                    ),
                ),
                default_site="workspace",
            ),
            "production.entropy": _CompositeSiteSpec(
                operators=all_operators,
                input_shapes=input_shapes,
                outputs=(
                    *local_outputs("local"),
                    _CompositeOutputSpec(
                        (n, b),
                        "transient.logdet.vector",
                    ),
                ),
                default_site="local",
            ),
            "production.log_normalizer": _CompositeSiteSpec(
                operators=all_operators,
                input_shapes=input_shapes,
                outputs=(
                    *local_outputs("workspace"),
                    _CompositeOutputSpec(
                        (n, b),
                        "gaussian.information",
                        "rhs",
                        frozenset({"aten::clone"}),
                    ),
                    *(
                        _CompositeOutputSpec(shape, "rhs")
                        for shape in sorted(rhs_shapes - {(n, b)})
                    ),
                ),
                default_site="workspace",
            ),
            "production.complete_objective": _CompositeSiteSpec(
                operators=all_operators,
                input_shapes=input_shapes,
                outputs=(
                    *local_outputs("local"),
                    *(
                        _CompositeOutputSpec(shape, "rhs")
                        for shape in sorted(
                            rhs_shapes
                            - {(n, b, b)}
                            - objective_population_shapes
                        )
                    ),
                    _CompositeOutputSpec(
                        (n, b, b),
                        "objective.diagonal",
                    ),
                    _CompositeOutputSpec(
                        (n - 1, b, b),
                        "objective.lower",
                    ),
                    *(
                        _CompositeOutputSpec(
                            shape,
                            "objective.population",
                        )
                        for shape in sorted(
                            objective_population_shapes
                            - {(n - 1, b, b), (n, b, b), (n, b)}
                        )
                    ),
                ),
                default_site="local",
                unregistered_input_operators=_TORCH_TENSORIZATION_OPERATORS,
            ),
        }
        if tuple(self._composite_sites) != _H8_COMPOSITE_SITE_NAMES:
            raise RuntimeError("H8 composite semantic-site order drifted")
        for name, spec in self._composite_sites.items():
            if spec.default_site not in self._sites:
                raise RuntimeError(
                    f"composite scope {name!r} has an unknown default site"
                )
            output_shapes = tuple(output.logical_shape for output in spec.outputs)
            if len(set(output_shapes)) != len(output_shapes):
                raise RuntimeError(
                    f"composite scope {name!r} has duplicate output shapes"
                )
            for output in spec.outputs:
                for site in (output.site, output.transient_site):
                    if site is not None and site not in self._sites:
                        raise RuntimeError(
                            f"composite scope {name!r} has an unknown output site"
                        )

    @property
    def registered_sites(self) -> tuple[str, ...]:
        return tuple(self._sites)

    @property
    def registered_composite_sites(self) -> tuple[str, ...]:
        return tuple(self._composite_sites)

    def descriptor(self) -> dict[str, object]:
        """Serialize the executable whitelist without caller-authored policy data."""

        return {
            "schema_version": "h8-allocation-whitelist-v1",
            "layout": {
                "horizon": self.layout.horizon,
                "population_size": self.layout.population_size,
                "d_z": self.layout.d_z,
                "d_m": self.layout.d_m,
                "block_size": self.layout.block_size,
                "population_dimension": self.layout.dimension,
            },
            "registered_sites": tuple(
                {
                    "name": name,
                    "kind": spec.kind,
                    "exact_shapes": tuple(sorted(spec.exact_shapes)),
                }
                for name, spec in sorted(self._sites.items())
            ),
            "registered_composite_sites": tuple(
                {
                    "name": name,
                    "operators": tuple(sorted(spec.operators)),
                    "input_shapes": tuple(sorted(spec.input_shapes)),
                    "outputs": tuple(
                        {
                            "logical_shape": output.logical_shape,
                            "site": output.site,
                            "transient_site": output.transient_site,
                            "canonical_operators": tuple(
                                sorted(output.canonical_operators)
                            ),
                        }
                        for output in sorted(
                            spec.outputs,
                            key=lambda item: item.logical_shape,
                        )
                    ),
                    "default_site": spec.default_site,
                    "unregistered_input_operators": tuple(
                        sorted(spec.unregistered_input_operators)
                    ),
                }
                for name, spec in sorted(self._composite_sites.items())
            ),
            "global_rules": {
                "forbidden_axis_D": self.layout.dimension,
                "float64_equivalent_scalar_cap": H8_MAX_STORAGE_SCALARS,
                "maximum_population_axes": 1,
                "unregistered_shapes_forbidden": True,
            },
        }

    def validate_composite_dispatch(
        self,
        *,
        scope: str,
        operator: str,
        input_shapes: Sequence[tuple[int, ...]],
    ) -> None:
        spec = self._composite_sites.get(scope)
        if spec is None:
            raise ValueError("semantic site is not a registered semantic scope")
        base = _operator_base(operator)
        if base not in spec.operators:
            raise H8ForbiddenAllocation(
                f"unregistered operator {base!r} for semantic scope {scope!r}"
            )
        for shape in input_shapes:
            checked = _logical_shape(shape)
            if checked not in spec.input_shapes:
                raise H8ForbiddenAllocation(
                    f"unregistered input shape {checked!r} for semantic scope {scope!r}"
                )

    def resolve_composite_output_sites(
        self,
        *,
        scope: str,
        operator: str,
        logical_shapes: Sequence[tuple[int, ...]],
    ) -> tuple[str, ...]:
        spec = self._composite_sites.get(scope)
        if spec is None:
            raise ValueError("semantic site is not a registered semantic scope")
        base = _operator_base(operator)
        return tuple(
            spec.output_site(
                operator=base,
                logical_shape=_logical_shape(shape),
            )
            for shape in logical_shapes
        )

    def composite_default_site(self, scope: str) -> str:
        spec = self._composite_sites.get(scope)
        if spec is None:
            raise ValueError("semantic site is not a registered semantic scope")
        return spec.default_site

    def composite_allows_unregistered_inputs(
        self,
        *,
        scope: str,
        operator: str,
    ) -> bool:
        spec = self._composite_sites.get(scope)
        if spec is None:
            return False
        return _operator_base(operator) in spec.unregistered_input_operators

    def register_exact_site(
        self,
        site: str,
        shapes: Iterable[tuple[int, ...]],
        *,
        kind: H8SemanticKind = "scalar_local_channel",
    ) -> None:
        if not isinstance(site, str) or not site:
            raise ValueError("site must be nonempty")
        checked = frozenset(_logical_shape(shape) for shape in shapes)
        if not checked:
            raise ValueError("an exact site needs at least one shape")
        for shape in checked:
            self._preflight_shape(shape, itemsize=8)
        self._sites[site] = _SiteSpec(kind, checked)

    def register_generator_objective_site(
        self,
        site: str,
        shape: tuple[int, ...],
    ) -> None:
        checked = _logical_shape(shape)
        self._require_generator_objective_shape(checked)
        self.register_exact_site(
            site,
            (checked,),
            kind="generator_objective",
        )

    def register_stack_site(self, stack_token: str, site: str) -> None:
        if not isinstance(stack_token, str) or not stack_token:
            raise ValueError("stack token must be nonempty")
        if site not in self._sites:
            raise ValueError("stack site must name a registered semantic site")
        self._stack_sites[stack_token] = site

    def resolve_stack_site(self, stack: Sequence[str]) -> str | None:
        matches = {
            site
            for token, site in self._stack_sites.items()
            if any(token in frame for frame in stack)
        }
        if len(matches) > 1:
            raise H8ProfilerObservabilityGap(
                "dispatch stack matches multiple semantic sites"
            )
        return next(iter(matches), None)

    def classify_allocation(
        self,
        *,
        site: str | None,
        logical_shape: tuple[int, ...],
        operator: str,
        itemsize: int = 8,
        registered_base: bool = True,
    ) -> H8AllocationDecision:
        shape = _logical_shape(logical_shape)
        equivalent = self._preflight_shape(shape, itemsize=itemsize)
        if site is None or site not in self._sites:
            raise H8ForbiddenAllocation("unregistered semantic allocation site")
        spec = self._sites[site]
        if any(token in operator.lower() for token in _VIEW_OPERATORS):
            if not registered_base:
                raise H8ForbiddenAllocation(
                    "view has no registered allowed base storage"
                )
        classification = self._classify_site_shape(site, spec, shape)
        return H8AllocationDecision(
            site=site,
            logical_shape=shape,
            operator=operator,
            classification=classification,
            float64_equivalent_scalars=equivalent,
        )

    def _classify_site_shape(
        self,
        site: str,
        spec: _SiteSpec,
        shape: tuple[int, ...],
    ) -> str:
        if spec.exact_shapes and shape not in spec.exact_shapes:
            raise H8ForbiddenAllocation(
                f"unregistered shape {shape!r} for semantic site {site!r}"
            )
        if spec.kind == "scalar_local_channel":
            b = self.layout.block_size
            if spec.exact_shapes:
                if any(dimension > b and dimension != 3 for dimension in shape):
                    raise H8ForbiddenAllocation("registered local axes exceed b or V=3")
            elif len(shape) > 2 or any(dimension > b for dimension in shape):
                raise H8ForbiddenAllocation(
                    "unregistered nonlocal scalar/channel shape"
                )
            return "local"
        if spec.kind == "generator_objective":
            self._require_generator_objective_shape(shape)
        return spec.kind

    def _require_generator_objective_shape(
        self,
        shape: tuple[int, ...],
    ) -> None:
        t = self.layout.horizon
        n = self.layout.population_size
        b = self.layout.block_size
        population_axes = sum(dimension in (t, n, n - 1) for dimension in shape)
        if population_axes != 1:
            raise H8ForbiddenAllocation(
                "generator/objective arrays need exactly one population axis"
            )
        for dimension in shape:
            if dimension in (t, n, n - 1):
                continue
            if dimension > b and dimension != 3:
                raise H8ForbiddenAllocation(
                    "generator/objective local axes exceed b or V=3"
                )

    def _preflight_shape(
        self,
        shape: tuple[int, ...],
        *,
        itemsize: int,
    ) -> int:
        n = self.layout.population_size
        b = self.layout.block_size
        d = self.layout.dimension
        triangular = n * (n + 1) // 2
        if shape == (d, d):
            raise H8ForbiddenAllocation("dense population matrix is forbidden")
        if shape == (d * d,):
            raise H8ForbiddenAllocation("flat dense population storage is forbidden")
        if shape == (d - 1, d - 1):
            raise H8ForbiddenAllocation(
                "near dense population storage cap is forbidden"
            )
        if d in shape:
            raise H8ForbiddenAllocation("global axis D is forbidden")
        if shape == (n, n, b, b):
            raise H8ForbiddenAllocation("population pair slab is forbidden")
        if shape == (triangular, b, b):
            raise H8ForbiddenAllocation("triangular pair storage is forbidden")
        if shape == (n * n, b, b):
            raise H8ForbiddenAllocation("combined pair slab is forbidden")
        population_values = {self.layout.horizon, n, n - 1}
        if sum(dimension in population_values for dimension in shape) >= 2:
            raise H8ForbiddenAllocation("two population/pair axes are forbidden")
        equivalent = _float64_equivalent_scalars(shape, itemsize)
        if equivalent > H8_MAX_STORAGE_SCALARS:
            raise H8ForbiddenAllocation(
                "single storage exceeds the float64-equivalent storage cap"
            )
        return equivalent

    def preflight_shape(
        self,
        shape: tuple[int, ...],
        *,
        itemsize: int = 8,
    ) -> int:
        return self._preflight_shape(_logical_shape(shape), itemsize=itemsize)

    def preflight_control(
        self,
        spec: H8NegativeControlSpec,
    ) -> None:
        if type(spec) is not H8NegativeControlSpec:
            raise ValueError("spec must be H8NegativeControlSpec")
        classify_h8_operator(
            self,
            operator=spec.requested_operation,
            operand_shapes=spec.logical_shapes[:-1],
            output_shape=spec.logical_shapes[-1],
        )
        self._preflight_shape(spec.logical_shapes[-1], itemsize=8)
        raise H8ForbiddenAllocation(spec.expected_reason)


def classify_h8_operator(
    policy: H8AllocationPolicy,
    *,
    operator: str,
    operand_shapes: Sequence[tuple[int, ...]],
    output_shape: tuple[int, ...] | None,
) -> str:
    """Classify an operation before a global kernel can execute."""

    if type(policy) is not H8AllocationPolicy:
        raise ValueError("policy must be H8AllocationPolicy")
    name = operator.lower()
    checked_operands = tuple(_logical_shape(shape) for shape in operand_shapes)
    checked_output = None if output_shape is None else _logical_shape(output_shape)
    matrix_shapes = tuple(shape for shape in checked_operands if len(shape) >= 2)
    if any(token in name for token in _DENSE_OPERATORS):
        if any(
            shape[-2] > policy.layout.block_size or shape[-1] > policy.layout.block_size
            for shape in matrix_shapes
        ):
            raise H8ForbiddenAllocation(
                "dense population linear-algebra operator is forbidden"
            )
        if checked_output is not None:
            policy.preflight_shape(checked_output)
        if "eig" in name or "svd" in name:
            return "local_eigensolver"
        return "local_factor_operator"
    if "eye" in name or "identity" in name:
        if checked_output == (
            policy.layout.dimension,
            policy.layout.dimension,
        ):
            raise H8ForbiddenAllocation("dense global identity is forbidden")
    if "solve" in name and len(checked_operands) >= 2:
        rhs = checked_operands[-1]
        if policy.layout.dimension in rhs or (
            rhs and rhs[-1] > policy.layout.block_size
        ):
            raise H8ForbiddenAllocation("global/full-width solve RHS is forbidden")
    if any(token in name for token in ("one_hot", "selector", "scatter")):
        all_shapes = checked_operands + (
            (() if checked_output is None else checked_output),
        )
        if any(policy.layout.dimension in shape for shape in all_shapes):
            raise H8ForbiddenAllocation("global selector pattern is forbidden")
    for shape in checked_operands:
        policy.preflight_shape(shape)
    if checked_output is not None:
        policy.preflight_shape(checked_output)
    return "bounded_operator"


@dataclass(slots=True)
class _TensorRegistration:
    reference: weakref.ReferenceType[Tensor]
    site: str
    logical_shape: tuple[int, ...]
    storage_key: str
    population_storage: bool


class H8DispatchTrace(TorchDispatchMode):
    """Pre/post Torch dispatch guard with alias-aware live-storage accounting."""

    def __init__(self, policy: H8AllocationPolicy) -> None:
        if type(policy) is not H8AllocationPolicy:
            raise ValueError("policy must be H8AllocationPolicy")
        super().__init__()
        self.policy = policy
        self._events: list[H8DispatchEvent] = []
        self._semantic_scope: ContextVar[_DispatchSemanticScope | None] = ContextVar(
            f"h8_dispatch_semantic_scope_{id(self)}",
            default=None,
        )
        self._control_stack: list[str] = []
        self._tensors: dict[int, _TensorRegistration] = {}
        self._storage_refcounts: dict[str, int] = {}
        self._storage_population_refcounts: dict[str, int] = {}
        self._storage_spans: dict[str, H8StorageSpan] = {}
        self._storage_sites: dict[str, str] = {}
        self._storage_logical_scalars: dict[str, int] = {}
        self._site_live_logical_scalars: dict[str, int] = {}
        self._baseline_storage_keys: set[str] = set()
        self._baseline_live_bytes = 0
        self._live_storage_bytes = 0
        self._live_peak_bytes = 0
        self._population_live_storage_bytes = 0
        self._population_live_peak_bytes = 0

    @property
    def events(self) -> tuple[H8DispatchEvent, ...]:
        return tuple(self._events)

    @property
    def live_storage_bytes(self) -> int:
        return self._live_storage_bytes

    @property
    def live_peak_bytes(self) -> int:
        return self._live_peak_bytes

    @property
    def population_live_storage_bytes(self) -> int:
        return self._population_live_storage_bytes

    @property
    def population_live_peak_bytes(self) -> int:
        return self._population_live_peak_bytes

    @property
    def baseline_live_bytes(self) -> int:
        return self._baseline_live_bytes

    @property
    def forbidden_attempt_count(self) -> int:
        return sum(event.forbidden_reason is not None for event in self._events)

    @property
    def trace_sha256(self) -> str:
        return _canonical_sha256(self._events)

    @contextmanager
    def negative_control(self, control_id: str) -> Iterable[None]:
        valid_ids = {
            spec.control_id for spec in h8_negative_control_specs(self.policy.layout)
        }
        if control_id not in valid_ids:
            raise ValueError("control_id is outside the frozen inventory")
        self._control_stack.append(control_id)
        try:
            yield
        finally:
            if self._control_stack.pop() != control_id:
                raise RuntimeError("negative-control stack restoration failed")

    @contextmanager
    def semantic_site(
        self,
        site: str,
        *,
        logical_output_shapes: tuple[tuple[int, ...], ...] = (),
    ) -> Iterable[None]:
        leaf = site in self.policy.registered_sites
        composite = site in self.policy.registered_composite_sites
        if not leaf and not composite:
            raise ValueError("semantic site is not a registered semantic scope")
        if self._semantic_scope.get() is not None:
            raise RuntimeError("dispatch semantic scopes cannot be nested")
        checked = tuple(_logical_shape(shape) for shape in logical_output_shapes)
        if composite and checked:
            raise ValueError(
                "composite semantic scopes derive output shapes from the registry"
            )
        scope = _DispatchSemanticScope(
            site=site,
            logical_output_shapes=checked,
            composite=composite,
        )
        token = self._semantic_scope.set(scope)
        try:
            yield
        finally:
            restoration_failed = self._semantic_scope.get() is not scope
            self._semantic_scope.reset(token)
            if restoration_failed:
                raise RuntimeError("dispatch semantic-scope restoration failed")

    def register_preexisting(
        self,
        tensor: Tensor,
        *,
        site: str,
        logical_shape: tuple[int, ...],
        storage_span: H8StorageSpan,
        nbytes: int,
    ) -> None:
        if not isinstance(tensor, Tensor):
            raise ValueError("tensor must be a torch.Tensor")
        if type(storage_span) is not H8StorageSpan:
            raise ValueError("storage_span must be an exact H8StorageSpan")
        if type(nbytes) is not int or nbytes < 0:
            raise ValueError("nbytes must be a nonnegative integer")
        shape = _logical_shape(logical_shape)
        actual_span = _tensor_storage_span(tensor)
        if storage_span != actual_span or nbytes != actual_span.nbytes:
            raise H8ProfilerObservabilityGap(
                "preexisting tensor span/bytes do not match physical storage"
            )
        decision = self.policy.classify_allocation(
            site=site,
            logical_shape=shape,
            operator="PREEXISTING",
            itemsize=int(tensor.element_size()),
        )
        span, new_storage = self._register_tensor(
            tensor,
            site,
            shape,
            population_storage=_is_population_decision(decision, self.policy.layout),
            logical_equivalent_scalars=decision.float64_equivalent_scalars,
        )
        if new_storage:
            self._baseline_storage_keys.add(span.storage_key)
            self._baseline_live_bytes += span.nbytes
        self._append_dispatch_event(
            operator="PREEXISTING",
            site=site,
            control_id=None,
            input_shapes=(),
            output_shapes=(shape,),
            physical_shapes=(_shape_of(tensor) or (),),
            stack_shapes=(),
            dtype=str(tensor.dtype),
            device=str(tensor.device),
            equivalent=decision.float64_equivalent_scalars,
            classifications=(decision.classification,),
            spans=(span,),
            aliases=() if new_storage else (span.storage_key,),
            new_storage_keys=(span.storage_key,) if new_storage else (),
            allocated_equivalent=(
                decision.float64_equivalent_scalars if new_storage else 0
            ),
            stack=("explicit_preexisting_registration",),
            executed=False,
            forbidden_reason=None,
        )

    def __torch_dispatch__(
        self,
        func: object,
        types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        del types
        call_kwargs = {} if kwargs is None else kwargs
        operator = _operator_name(func)
        stack = tuple(
            f"{frame.filename}:{frame.lineno}:{frame.name}"
            for frame in traceback.extract_stack(limit=32)[:-1]
        )
        control_id = self._control_stack[-1] if self._control_stack else None
        active_scope = self._active_scope(stack)
        composite_scope = (
            active_scope.site
            if active_scope is not None and active_scope.composite
            else None
        )
        overrides = (
            ()
            if active_scope is None or active_scope.composite
            else active_scope.logical_output_shapes
        )
        active_site = self._scope_default_site(active_scope)
        input_tensors = tuple(_walk_tensors((args, call_kwargs)))
        input_shapes = tuple(self._logical_tensor_shape(item) for item in input_tensors)
        unregistered_inputs = tuple(
            item for item in input_tensors if not self._is_registered_tensor(item)
        )
        tensorization_inputs_allowed = (
            composite_scope is not None
            and self.policy.composite_allows_unregistered_inputs(
                scope=composite_scope,
                operator=operator,
            )
        )
        if (
            unregistered_inputs
            and control_id is None
            and not tensorization_inputs_allowed
        ):
            reason = "unregistered non-control dispatch input"
            self._append_dispatch_event(
                operator=operator,
                site=active_site,
                control_id=None,
                input_shapes=input_shapes,
                output_shapes=(),
                physical_shapes=(),
                stack_shapes=(),
                dtype=None,
                device=None,
                equivalent=0,
                classifications=(),
                spans=(),
                aliases=(),
                new_storage_keys=(),
                allocated_equivalent=0,
                stack=stack,
                executed=False,
                forbidden_reason=reason,
            )
            raise H8ForbiddenAllocation(reason)
        stack_shapes = self._stack_member_shapes(operator, args)
        inferred = _infer_torch_output_shape(
            operator,
            args,
            call_kwargs,
            input_shapes,
        )
        requested_shapes = overrides or (() if inferred is None else (inferred,))
        registered_base = (
            any(self._is_registered_tensor(item) for item in input_tensors)
            or tensorization_inputs_allowed
        )
        itemsize = _requested_itemsize(call_kwargs, input_tensors)
        decision_sites: tuple[str | None, ...] = ()
        event_site = active_site
        try:
            if composite_scope is not None:
                self.policy.validate_composite_dispatch(
                    scope=composite_scope,
                    operator=operator,
                    input_shapes=input_shapes,
                )
            classify_h8_operator(
                self.policy,
                operator=operator,
                operand_shapes=input_shapes,
                output_shape=inferred,
            )
            decision_sites = self._resolve_output_sites(
                active_scope,
                operator,
                requested_shapes,
            )
            event_site = self._event_site(decision_sites, fallback=active_site)
            decisions = tuple(
                self.policy.classify_allocation(
                    site=site,
                    logical_shape=shape,
                    operator=operator,
                    itemsize=itemsize,
                    registered_base=registered_base,
                )
                for site, shape in zip(
                    decision_sites,
                    requested_shapes,
                    strict=True,
                )
            )
        except H8ForbiddenAllocation as error:
            self._append_dispatch_event(
                operator=operator,
                site=event_site,
                control_id=control_id,
                input_shapes=input_shapes,
                output_shapes=requested_shapes,
                physical_shapes=(),
                stack_shapes=stack_shapes,
                dtype=None,
                device=None,
                equivalent=0,
                classifications=(),
                spans=(),
                aliases=(),
                new_storage_keys=(),
                allocated_equivalent=0,
                stack=stack,
                executed=False,
                forbidden_reason=error.reason,
            )
            raise

        try:
            result = func(*args, **call_kwargs)  # type: ignore[operator]
        except Exception as error:
            reason = (
                "operation executed past dispatch preflight and raised "
                f"{type(error).__name__}"
            )
            self._append_dispatch_event(
                operator=operator,
                site=event_site,
                control_id=control_id,
                input_shapes=input_shapes,
                output_shapes=requested_shapes,
                physical_shapes=(),
                stack_shapes=stack_shapes,
                dtype=None,
                device=None,
                equivalent=sum(
                    decision.float64_equivalent_scalars for decision in decisions
                ),
                classifications=tuple(
                    decision.classification for decision in decisions
                ),
                spans=(),
                aliases=(),
                new_storage_keys=(),
                allocated_equivalent=0,
                stack=stack,
                executed=True,
                forbidden_reason=reason,
            )
            raise
        output_tensors = tuple(_walk_tensors(result))
        physical_shapes = tuple(_shape_of(item) or () for item in output_tensors)
        logical_shapes = self._resolve_output_shapes(
            overrides,
            inferred,
            output_tensors,
        )
        try:
            if len(logical_shapes) != len(output_tensors):
                raise H8ForbiddenAllocation(
                    "dispatch output lacks a unique logical-shape witness"
                )
            decision_sites = self._resolve_output_sites(
                active_scope,
                operator,
                logical_shapes,
            )
            event_site = self._event_site(decision_sites, fallback=active_site)
            decisions = tuple(
                self.policy.classify_allocation(
                    site=site,
                    logical_shape=shape,
                    operator=operator,
                    itemsize=int(tensor.element_size()),
                    registered_base=registered_base,
                )
                for site, shape, tensor in zip(
                    decision_sites,
                    logical_shapes,
                    output_tensors,
                    strict=True,
                )
            )
        except H8ForbiddenAllocation as error:
            self._append_dispatch_event(
                operator=operator,
                site=event_site,
                control_id=control_id,
                input_shapes=input_shapes,
                output_shapes=logical_shapes,
                physical_shapes=physical_shapes,
                stack_shapes=stack_shapes,
                dtype=None,
                device=None,
                equivalent=0,
                classifications=(),
                spans=(),
                aliases=(),
                new_storage_keys=(),
                allocated_equivalent=0,
                stack=stack,
                executed=True,
                forbidden_reason=error.reason,
            )
            raise

        prior_keys = set(self._storage_refcounts)
        spans: list[H8StorageSpan] = []
        aliases: list[str] = []
        new_storage_keys: list[str] = []
        allocated_equivalent = 0
        for site, shape, tensor, decision in zip(
            decision_sites,
            logical_shapes,
            output_tensors,
            decisions,
            strict=True,
        ):
            span = _tensor_storage_span(tensor)
            if span.storage_key in prior_keys:
                aliases.append(span.storage_key)
            spans.append(span)
            if site is None:
                raise H8ForbiddenAllocation("dispatch output lacks a semantic site")
            _registered_span, new_storage = self._register_tensor(
                tensor,
                site,
                shape,
                population_storage=_is_population_decision(
                    decision,
                    self.policy.layout,
                ),
                logical_equivalent_scalars=(decision.float64_equivalent_scalars),
            )
            if new_storage:
                new_storage_keys.append(span.storage_key)
                allocated_equivalent += decision.float64_equivalent_scalars
        dtype = str(output_tensors[0].dtype) if output_tensors else None
        device = str(output_tensors[0].device) if output_tensors else None
        equivalent = sum(decision.float64_equivalent_scalars for decision in decisions)
        self._append_dispatch_event(
            operator=operator,
            site=event_site,
            control_id=control_id,
            input_shapes=input_shapes,
            output_shapes=logical_shapes,
            physical_shapes=physical_shapes,
            stack_shapes=stack_shapes,
            dtype=dtype,
            device=device,
            equivalent=equivalent,
            classifications=tuple(decision.classification for decision in decisions),
            spans=tuple(spans),
            aliases=tuple(dict.fromkeys(aliases)),
            new_storage_keys=tuple(dict.fromkeys(new_storage_keys)),
            allocated_equivalent=allocated_equivalent,
            stack=stack,
            executed=True,
            forbidden_reason=None,
        )
        return result

    def _active_scope(
        self,
        stack: tuple[str, ...],
    ) -> _DispatchSemanticScope | None:
        scope = self._semantic_scope.get()
        if scope is not None:
            return scope
        site = self.policy.resolve_stack_site(stack)
        if site is None:
            return None
        return _DispatchSemanticScope(
            site=site,
            logical_output_shapes=(),
            composite=False,
        )

    def _scope_default_site(
        self,
        scope: _DispatchSemanticScope | None,
    ) -> str | None:
        if scope is None:
            return None
        if scope.composite:
            return self.policy.composite_default_site(scope.site)
        return scope.site

    def _resolve_output_sites(
        self,
        scope: _DispatchSemanticScope | None,
        operator: str,
        logical_shapes: Sequence[tuple[int, ...]],
    ) -> tuple[str | None, ...]:
        if scope is None:
            return tuple(None for _shape in logical_shapes)
        if scope.composite:
            return self.policy.resolve_composite_output_sites(
                scope=scope.site,
                operator=operator,
                logical_shapes=logical_shapes,
            )
        return tuple(scope.site for _shape in logical_shapes)

    @staticmethod
    def _event_site(
        sites: Sequence[str | None],
        *,
        fallback: str | None,
    ) -> str | None:
        unique = tuple(dict.fromkeys(sites))
        if len(unique) > 1:
            raise H8ForbiddenAllocation(
                "one dispatch event resolved to multiple semantic allocation sites"
            )
        return unique[0] if unique else fallback

    def _logical_tensor_shape(self, tensor: Tensor) -> tuple[int, ...]:
        registration = self._tensors.get(id(tensor))
        if registration is not None and registration.reference() is tensor:
            return registration.logical_shape
        shape = _shape_of(tensor)
        return () if shape is None else shape

    def _is_registered_tensor(self, tensor: Tensor) -> bool:
        registration = self._tensors.get(id(tensor))
        return registration is not None and registration.reference() is tensor

    def _stack_member_shapes(
        self,
        operator: str,
        args: tuple[object, ...],
    ) -> tuple[tuple[int, ...], ...]:
        if not any(token in operator.lower() for token in _STACK_OPERATORS):
            return ()
        if not args or not isinstance(args[0], (tuple, list)):
            return ()
        return tuple(
            self._logical_tensor_shape(item)
            for item in args[0]
            if isinstance(item, Tensor)
        )

    def _resolve_output_shapes(
        self,
        overrides: tuple[tuple[int, ...], ...],
        inferred: tuple[int, ...] | None,
        outputs: tuple[Tensor, ...],
    ) -> tuple[tuple[int, ...], ...]:
        if overrides:
            return overrides
        if len(outputs) == 1 and inferred is not None:
            return (inferred,)
        return tuple(_shape_of(item) or () for item in outputs)

    def _register_tensor(
        self,
        tensor: Tensor,
        site: str,
        logical_shape: tuple[int, ...],
        *,
        population_storage: bool,
        logical_equivalent_scalars: int,
    ) -> tuple[H8StorageSpan, bool]:
        tensor_id = id(tensor)
        existing = self._tensors.get(tensor_id)
        if existing is not None and existing.reference() is tensor:
            return self._storage_spans[existing.storage_key], False
        span = _tensor_storage_span(tensor)
        if (span.nbytes + 7) // 8 > H8_MAX_STORAGE_SCALARS:
            raise H8ForbiddenAllocation(
                "physical storage span exceeds the float64-equivalent cap"
            )
        self._storage_spans.setdefault(span.storage_key, span)
        new_storage = self._storage_refcounts.get(span.storage_key, 0) == 0
        if new_storage:
            self._live_storage_bytes += span.nbytes
            self._live_peak_bytes = max(
                self._live_peak_bytes,
                self._live_storage_bytes,
            )
            self._storage_sites[span.storage_key] = site
            self._storage_logical_scalars[span.storage_key] = logical_equivalent_scalars
            self._site_live_logical_scalars[site] = (
                self._site_live_logical_scalars.get(site, 0)
                + logical_equivalent_scalars
            )
        self._storage_refcounts[span.storage_key] = (
            self._storage_refcounts.get(span.storage_key, 0) + 1
        )
        if population_storage:
            if self._storage_population_refcounts.get(span.storage_key, 0) == 0:
                self._population_live_storage_bytes += span.nbytes
                self._population_live_peak_bytes = max(
                    self._population_live_peak_bytes,
                    self._population_live_storage_bytes,
                )
            self._storage_population_refcounts[span.storage_key] = (
                self._storage_population_refcounts.get(span.storage_key, 0) + 1
            )

        def release(_reference: weakref.ReferenceType[Tensor]) -> None:
            registration = self._tensors.get(tensor_id)
            if registration is None or registration.reference is not _reference:
                return
            self._tensors.pop(tensor_id)
            key = registration.storage_key
            if registration.population_storage:
                population_remaining = self._storage_population_refcounts[key] - 1
                if population_remaining:
                    self._storage_population_refcounts[key] = population_remaining
                else:
                    self._storage_population_refcounts.pop(key, None)
                    self._population_live_storage_bytes -= self._storage_spans[
                        key
                    ].nbytes
            remaining = self._storage_refcounts[key] - 1
            if remaining:
                self._storage_refcounts[key] = remaining
                return
            self._storage_refcounts.pop(key, None)
            released = self._storage_spans.pop(key)
            released_site = self._storage_sites.pop(key)
            released_scalars = self._storage_logical_scalars.pop(key)
            remaining_site_scalars = (
                self._site_live_logical_scalars[released_site] - released_scalars
            )
            if remaining_site_scalars:
                self._site_live_logical_scalars[released_site] = remaining_site_scalars
            else:
                self._site_live_logical_scalars.pop(released_site)
            self._live_storage_bytes -= released.nbytes

        reference = weakref.ref(tensor, release)
        self._tensors[tensor_id] = _TensorRegistration(
            reference=reference,
            site=site,
            logical_shape=logical_shape,
            storage_key=span.storage_key,
            population_storage=population_storage,
        )
        return span, new_storage

    def _append_dispatch_event(
        self,
        *,
        operator: str,
        site: str | None,
        control_id: str | None,
        input_shapes: tuple[tuple[int, ...], ...],
        output_shapes: tuple[tuple[int, ...], ...],
        physical_shapes: tuple[tuple[int, ...], ...],
        stack_shapes: tuple[tuple[int, ...], ...],
        dtype: str | None,
        device: str | None,
        equivalent: int,
        classifications: tuple[str, ...],
        spans: tuple[H8StorageSpan, ...],
        aliases: tuple[str, ...],
        new_storage_keys: tuple[str, ...],
        allocated_equivalent: int,
        stack: tuple[str, ...],
        executed: bool,
        forbidden_reason: str | None,
    ) -> None:
        self._events.append(
            H8DispatchEvent(
                sequence=len(self._events),
                operator=operator,
                semantic_site=site,
                control_id=control_id,
                input_shapes=input_shapes,
                output_shapes=output_shapes,
                physical_output_shapes=physical_shapes,
                stack_member_shapes=stack_shapes,
                stack_member_count=len(stack_shapes),
                dtype=dtype,
                device=device,
                float64_equivalent_scalars=equivalent,
                classifications=classifications,
                storage_spans=spans,
                alias_storage_keys=aliases,
                new_storage_keys=new_storage_keys,
                allocated_float64_equivalent_scalars=allocated_equivalent,
                live_float64_equivalent_scalars_by_site=tuple(
                    sorted(self._site_live_logical_scalars.items())
                ),
                stack=stack,
                executed=executed,
                forbidden_reason=forbidden_reason,
                live_storage_bytes_after=self._live_storage_bytes,
                population_live_storage_bytes_after=(
                    self._population_live_storage_bytes
                ),
            )
        )


def _requested_itemsize(
    kwargs: Mapping[str, object],
    inputs: Sequence[Tensor],
) -> int:
    dtype = kwargs.get("dtype")
    dtype_sizes = {
        torch.float64: 8,
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int64: 8,
        torch.int32: 4,
        torch.int16: 2,
        torch.int8: 1,
        torch.uint8: 1,
        torch.bool: 1,
    }
    if dtype in dtype_sizes:
        return dtype_sizes[dtype]
    if inputs:
        return int(inputs[0].element_size())
    return 8


def _is_population_decision(
    decision: H8AllocationDecision,
    layout: BlockChainLayout,
) -> bool:
    population_axes = {
        layout.horizon,
        layout.population_size,
        layout.population_size - 1,
    }
    return decision.classification != "local" and any(
        dimension in population_axes for dimension in decision.logical_shape
    )


def _tensor_storage_span(tensor: Tensor) -> H8StorageSpan:
    storage = tensor.untyped_storage()
    pointer = int(storage.data_ptr())
    nbytes = int(storage.nbytes())
    storage_identity = int(getattr(storage, "_cdata", id(storage)))
    device = str(tensor.device)
    key = f"{device}:{storage_identity}:{pointer}:{nbytes}"
    return H8StorageSpan(
        storage_key=key,
        device=device,
        pointer=pointer,
        span_start=pointer,
        span_end=pointer + nbytes,
        nbytes=nbytes,
    )


def h8_tensor_storage_span(tensor: Tensor) -> H8StorageSpan:
    """Return the exact physical span required for explicit registration."""

    if not isinstance(tensor, Tensor):
        raise ValueError("tensor must be a torch.Tensor")
    return _tensor_storage_span(tensor)


def _infer_torch_output_shape(
    operator: str,
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
    input_shapes: tuple[tuple[int, ...], ...],
) -> tuple[int, ...] | None:
    name = operator.lower()
    if any(token in name for token in _FACTORY_OPERATORS) and args:
        return _shape_argument(args[0])
    if "eye" in name:
        rows = _integer_argument(args, kwargs, 0, "n")
        columns = _integer_argument(args, kwargs, 1, "m")
        if rows is not None:
            return (rows, rows if columns is None else columns)
    if any(token in name for token in ("reshape", "view")) and len(args) > 1:
        return _shape_argument(args[1])
    if "outer" in name and len(input_shapes) >= 2:
        return (math.prod(input_shapes[0]), math.prod(input_shapes[1]))
    if "matmul" in name or name.endswith("::mm"):
        if len(input_shapes) >= 2:
            return _matmul_shape(input_shapes[-2], input_shapes[-1])
    if any(token in name for token in _STACK_OPERATORS) and input_shapes:
        dim = _integer_argument(args, kwargs, 1, "dim") or 0
        base = list(input_shapes[0])
        if dim < 0:
            dim += len(base) + 1
        if 0 <= dim <= len(base):
            base.insert(dim, len(input_shapes))
            return tuple(base)
    if any(token in name for token in _CONCATENATE_OPERATORS) and input_shapes:
        dim = _integer_argument(args, kwargs, 1, "dim") or 0
        if dim < 0:
            dim += len(input_shapes[0])
        if 0 <= dim < len(input_shapes[0]):
            result = list(input_shapes[0])
            result[dim] = sum(shape[dim] for shape in input_shapes)
            return tuple(result)
    return None


def _shape_argument(value: object) -> tuple[int, ...] | None:
    if isinstance(value, torch.Size):
        return tuple(int(item) for item in value)
    if isinstance(value, (tuple, list)) and all(type(item) is int for item in value):
        return tuple(value)
    if type(value) is int:
        return (value,)
    return None


def _integer_argument(
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
    index: int,
    name: str,
) -> int | None:
    value = args[index] if len(args) > index else kwargs.get(name)
    return value if type(value) is int else None


def _matmul_shape(
    left: tuple[int, ...],
    right: tuple[int, ...],
) -> tuple[int, ...] | None:
    if not left or not right:
        return None
    if len(left) == 1 and len(right) == 1:
        return ()
    if len(left) == 1:
        return right[:-2] + (right[-1],)
    if len(right) == 1:
        return left[:-1]
    try:
        batch = torch.broadcast_shapes(left[:-2], right[:-2])
    except RuntimeError:
        return None
    return tuple(batch) + (left[-2], right[-1])


class H8NumpyAllocationGuard:
    """Scoped, restoration-safe NumPy constructor and linear-algebra guard."""

    _NUMPY_NAMES = (
        "add",
        "all",
        "asarray",
        "ascontiguousarray",
        "divide",
        "empty",
        "zeros",
        "ones",
        "full",
        "eye",
        "identity",
        "isfinite",
        "reshape",
        "resize",
        "stack",
        "concatenate",
        "outer",
        "matmul",
        "multiply",
        "sqrt",
        "transpose",
    )
    _LINALG_NAMES = (
        "cholesky",
        "cond",
        "det",
        "eig",
        "eigh",
        "eigvals",
        "eigvalsh",
        "inv",
        "lstsq",
        "matrix_norm",
        "matrix_power",
        "matrix_rank",
        "multi_dot",
        "norm",
        "pinv",
        "qr",
        "slogdet",
        "solve",
        "svd",
        "svdvals",
        "tensorinv",
        "tensorsolve",
        "vector_norm",
    )

    def __init__(self, policy: H8AllocationPolicy) -> None:
        if type(policy) is not H8AllocationPolicy:
            raise ValueError("policy must be H8AllocationPolicy")
        self.policy = policy
        self._events: list[H8NumpyGuardEvent] = []
        self._originals: list[tuple[object, str, object]] = []
        self._site_stack: list[tuple[str, tuple[tuple[int, ...], ...]]] = []
        self._control_stack: list[str] = []
        self._logical_arrays: dict[int, tuple[str, tuple[int, ...]]] = {}
        self._entered = False

    @property
    def events(self) -> tuple[H8NumpyGuardEvent, ...]:
        return tuple(self._events)

    @contextmanager
    def negative_control(self, control_id: str) -> Iterable[None]:
        valid_ids = {
            spec.control_id for spec in h8_negative_control_specs(self.policy.layout)
        }
        if control_id not in valid_ids:
            raise ValueError("control_id is outside the frozen inventory")
        self._control_stack.append(control_id)
        try:
            yield
        finally:
            if self._control_stack.pop() != control_id:
                raise RuntimeError("NumPy negative-control restoration failed")

    @contextmanager
    def semantic_site(
        self,
        site: str,
        *,
        logical_output_shapes: tuple[tuple[int, ...], ...] = (),
    ) -> Iterable[None]:
        if site not in self.policy.registered_sites:
            raise ValueError("semantic site is not registered")
        checked = tuple(_logical_shape(shape) for shape in logical_output_shapes)
        self._site_stack.append((site, checked))
        try:
            yield
        finally:
            if self._site_stack.pop() != (site, checked):
                raise RuntimeError("NumPy semantic-site restoration failed")

    def register_preexisting(
        self,
        array: np.ndarray[Any, Any],
        *,
        site: str,
        logical_shape: tuple[int, ...] | None = None,
    ) -> None:
        if not isinstance(array, np.ndarray):
            raise ValueError("array must be a numpy.ndarray")
        shape = tuple(array.shape) if logical_shape is None else logical_shape
        checked = _logical_shape(shape)
        self.policy.classify_allocation(
            site=site,
            logical_shape=checked,
            operator="PREEXISTING",
            itemsize=int(array.dtype.itemsize),
        )
        self._logical_arrays[id(array)] = (site, checked)

    def standard_normal(
        self,
        generator: np.random.Generator,
        *,
        size: tuple[int, ...],
        dtype: object = np.float64,
    ) -> np.ndarray[Any, Any]:
        """Preflight one Generator draw before its storage can be created."""

        if not self._entered:
            raise RuntimeError("standard_normal requires an active NumPy guard")
        if not isinstance(generator, np.random.Generator):
            raise ValueError("generator must be numpy.random.Generator")
        shape = _numpy_shape_argument(size)
        if shape is None:
            raise H8ForbiddenAllocation(
                "standard_normal shape is not classifiable before allocation"
            )
        site, overrides = self._site_stack[-1] if self._site_stack else (None, ())
        requested = overrides or (shape,)
        operator = "numpy.random.Generator.standard_normal"
        try:
            itemsize = int(np.dtype(dtype).itemsize)
            decisions = tuple(
                self.policy.classify_allocation(
                    site=site,
                    logical_shape=output_shape,
                    operator=operator,
                    itemsize=itemsize,
                )
                for output_shape in requested
            )
            for output_shape in requested:
                classify_h8_operator(
                    self.policy,
                    operator=operator,
                    operand_shapes=(),
                    output_shape=output_shape,
                )
        except (TypeError, ValueError, H8ForbiddenAllocation) as error:
            reason = (
                error.reason
                if isinstance(error, H8ForbiddenAllocation)
                else "standard_normal dtype is not classifiable before allocation"
            )
            self._append_event(
                operator,
                site,
                None,
                (),
                requested,
                None,
                0,
                False,
                reason,
            )
            raise H8ForbiddenAllocation(reason) from error
        try:
            result = generator.standard_normal(size=size, dtype=dtype)
        except Exception as error:
            self._append_event(
                operator,
                site,
                None,
                (),
                requested,
                None,
                sum(item.float64_equivalent_scalars for item in decisions),
                True,
                f"operation executed past NumPy preflight and raised "
                f"{type(error).__name__}",
            )
            raise
        physical_shape = tuple(int(dimension) for dimension in result.shape)
        if len(requested) != 1 or physical_shape != requested[0] or site is None:
            reason = "standard_normal output lacks its exact logical-shape witness"
            self._append_event(
                operator,
                site,
                None,
                (),
                requested,
                str(result.dtype),
                0,
                True,
                reason,
            )
            raise H8ForbiddenAllocation(reason)
        self._logical_arrays[id(result)] = (site, requested[0])
        self._append_event(
            operator,
            site,
            None,
            (),
            requested,
            str(result.dtype),
            sum(item.float64_equivalent_scalars for item in decisions),
            True,
            None,
        )
        return result

    def __enter__(self) -> H8NumpyAllocationGuard:
        if self._entered:
            raise RuntimeError("NumPy allocation guard is not reentrant")
        self._entered = True
        try:
            for name in self._NUMPY_NAMES:
                self._patch(np, name)
            for name in self._LINALG_NAMES:
                if hasattr(np.linalg, name):
                    self._patch(np.linalg, name)
        except BaseException:
            self._restore()
            self._entered = False
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: object,
    ) -> bool:
        del exc_type, exc_value, exc_traceback
        try:
            self._restore()
        finally:
            self._entered = False
            self._site_stack.clear()
            self._control_stack.clear()
        return False

    def _patch(self, owner: object, name: str) -> None:
        original = getattr(owner, name)
        self._originals.append((owner, name, original))
        setattr(owner, name, self._wrapped(name, original, owner is np.linalg))

    def _restore(self) -> None:
        while self._originals:
            owner, name, original = self._originals.pop()
            setattr(owner, name, original)

    def _wrapped(
        self,
        name: str,
        original: Callable[..., object],
        is_linalg: bool,
    ) -> Callable[..., object]:
        operator = f"numpy.linalg.{name}" if is_linalg else f"numpy.{name}"

        @functools.wraps(original)
        def guarded(*args: object, **kwargs: object) -> object:
            input_arrays = tuple(_walk_arrays((args, kwargs)))
            input_shapes = tuple(
                self._logical_array_shape(item) for item in input_arrays
            )
            control_id = self._control_stack[-1] if self._control_stack else None
            site, overrides = self._site_stack[-1] if self._site_stack else (None, ())
            if (
                any(id(item) not in self._logical_arrays for item in input_arrays)
                and control_id is None
            ):
                reason = "unregistered non-control NumPy input"
                self._append_event(
                    operator,
                    site,
                    control_id,
                    input_shapes,
                    (),
                    None,
                    0,
                    False,
                    reason,
                )
                raise H8ForbiddenAllocation(reason)
            try:
                inferred_shapes = _infer_numpy_output_shapes(
                    name,
                    args,
                    kwargs,
                    input_shapes,
                    shape_resolver=self._logical_array_shape,
                )
                requested = overrides or inferred_shapes or ()
                shape_variants = (
                    (overrides,)
                    if overrides
                    else _numpy_output_shape_variants(name, inferred_shapes)
                )
                itemsize = _numpy_requested_itemsize(
                    name,
                    args,
                    kwargs,
                    input_arrays,
                )
            except H8ForbiddenAllocation as error:
                self._append_event(
                    operator,
                    site,
                    control_id,
                    input_shapes,
                    overrides,
                    None,
                    0,
                    False,
                    error.reason,
                )
                raise
            try:
                if shape_variants:
                    for variant in shape_variants:
                        for output_shape in variant:
                            classify_h8_operator(
                                self.policy,
                                operator=operator,
                                operand_shapes=input_shapes,
                                output_shape=output_shape,
                            )
                            self.policy.classify_allocation(
                                site=site,
                                logical_shape=output_shape,
                                operator=operator,
                                itemsize=itemsize,
                            )
                else:
                    classify_h8_operator(
                        self.policy,
                        operator=operator,
                        operand_shapes=input_shapes,
                        output_shape=None,
                    )
                decisions = tuple(
                    self.policy.classify_allocation(
                        site=site,
                        logical_shape=shape,
                        operator=operator,
                        itemsize=itemsize,
                    )
                    for shape in requested
                )
            except H8ForbiddenAllocation as error:
                self._append_event(
                    operator,
                    site,
                    control_id,
                    input_shapes,
                    requested,
                    None,
                    0,
                    False,
                    error.reason,
                )
                raise
            try:
                result = original(*args, **kwargs)
            except Exception as error:
                reason = (
                    "operation executed past NumPy preflight and raised "
                    f"{type(error).__name__}"
                )
                self._append_event(
                    operator,
                    site,
                    control_id,
                    input_shapes,
                    requested,
                    None,
                    sum(decision.float64_equivalent_scalars for decision in decisions),
                    True,
                    reason,
                )
                raise
            outputs = tuple(_walk_arrays(result))
            physical_shapes = tuple(tuple(item.shape) for item in outputs)
            validate_inferred_result = (
                not overrides and name == "lstsq" and inferred_shapes is not None
            )
            logical_shapes = (
                physical_shapes
                if validate_inferred_result
                else (overrides or inferred_shapes or physical_shapes)
            )
            try:
                if validate_inferred_result and physical_shapes not in shape_variants:
                    raise H8ForbiddenAllocation(
                        "NumPy output conflicts with every preflighted shape witness"
                    )
                if len(logical_shapes) != len(outputs):
                    raise H8ForbiddenAllocation(
                        "NumPy output lacks a unique logical-shape witness"
                    )
                decisions = tuple(
                    self.policy.classify_allocation(
                        site=site,
                        logical_shape=shape,
                        operator=operator,
                        itemsize=int(array.dtype.itemsize),
                    )
                    for shape, array in zip(logical_shapes, outputs, strict=True)
                )
            except H8ForbiddenAllocation as error:
                self._append_event(
                    operator,
                    site,
                    control_id,
                    input_shapes,
                    logical_shapes,
                    None,
                    0,
                    True,
                    error.reason,
                )
                raise
            if site is None:
                raise H8ForbiddenAllocation("NumPy output lacks a semantic site")
            for shape, array in zip(logical_shapes, outputs, strict=True):
                self._logical_arrays[id(array)] = (site, shape)
            dtype = str(outputs[0].dtype) if outputs else None
            self._append_event(
                operator,
                site,
                control_id,
                input_shapes,
                logical_shapes,
                dtype,
                sum(item.float64_equivalent_scalars for item in decisions),
                True,
                None,
            )
            return result

        return guarded

    def _logical_array_shape(
        self,
        array: np.ndarray[Any, Any] | np.generic,
    ) -> tuple[int, ...]:
        registration = self._logical_arrays.get(id(array))
        return tuple(array.shape) if registration is None else registration[1]

    def _append_event(
        self,
        operator: str,
        site: str | None,
        control_id: str | None,
        input_shapes: tuple[tuple[int, ...], ...],
        output_shapes: tuple[tuple[int, ...], ...],
        dtype: str | None,
        equivalent: int,
        executed: bool,
        reason: str | None,
    ) -> None:
        self._events.append(
            H8NumpyGuardEvent(
                sequence=len(self._events),
                operator=operator,
                semantic_site=site,
                control_id=control_id,
                input_shapes=input_shapes,
                output_shapes=output_shapes,
                dtype=dtype,
                float64_equivalent_scalars=equivalent,
                executed=executed,
                forbidden_reason=reason,
            )
        )


def _infer_numpy_output_shapes(
    name: str,
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
    input_shapes: tuple[tuple[int, ...], ...],
    *,
    shape_resolver: Callable[
        [np.ndarray[Any, Any] | np.generic],
        tuple[int, ...],
    ],
) -> tuple[tuple[int, ...], ...] | None:
    if name in ("empty", "zeros", "ones", "full"):
        shape = _numpy_shape_argument(_call_argument(args, kwargs, 0, "shape"))
        return None if shape is None else (shape,)
    if name in ("eye", "identity"):
        rows = _integer_call_argument(args, kwargs, 0, "N", "n")
        columns = (
            None
            if name == "identity"
            else _integer_call_argument(args, kwargs, 1, "M", "m")
        )
        if rows is not None:
            return ((rows, rows if columns is None else columns),)
    if name in ("reshape", "resize"):
        names = ("shape", "newshape") if name == "reshape" else ("new_shape",)
        shape = _numpy_shape_argument(_call_argument(args, kwargs, 1, *names))
        if name == "reshape" and shape is not None and input_shapes:
            shape = _resolve_inferred_dimension(shape, input_shapes[0])
        return None if shape is None else (shape,)
    primary_shapes = _numpy_primary_operand_shapes(
        args,
        kwargs,
        shape_resolver=shape_resolver,
    )
    if name in ("asarray", "ascontiguousarray"):
        if primary_shapes:
            return (primary_shapes[0],)
        literal_shape = _numpy_literal_shape(
            _call_argument(args, kwargs, 0, "a")
        )
        return None if literal_shape is None else (literal_shape,)
    if name in ("add", "multiply", "divide"):
        result = _numpy_broadcast_shape(primary_shapes)
        return None if result is None else (result,)
    if name == "isfinite" and primary_shapes:
        return (primary_shapes[0],)
    if name == "all" and primary_shapes:
        return (
            _numpy_reduction_shape(
                primary_shapes[0],
                _call_argument(args, kwargs, 1, "axis"),
                _bool_call_argument(args, kwargs, 3, "keepdims", False),
            ),
        )
    if name == "sqrt":
        return (((),) if not primary_shapes else (primary_shapes[0],))
    if name == "transpose" and primary_shapes:
        axes = _call_argument(args, kwargs, 1, "axes")
        if axes is _MISSING or axes is None:
            return (tuple(reversed(primary_shapes[0])),)
        if isinstance(axes, (tuple, list)) and len(axes) == len(primary_shapes[0]):
            try:
                return (
                    tuple(primary_shapes[0][int(index)] for index in axes),
                )
            except (IndexError, TypeError, ValueError):
                return None
    if name == "outer" and len(primary_shapes) >= 2:
        return (
            (
                math.prod(primary_shapes[0]),
                math.prod(primary_shapes[1]),
            ),
        )
    if name == "matmul" and len(primary_shapes) >= 2:
        result = _numpy_matmul_shape(primary_shapes[0], primary_shapes[1])
        return None if result is None else (result,)
    if name == "stack" and primary_shapes:
        axis = _integer_call_argument(args, kwargs, 1, "axis")
        axis = 0 if axis is None else axis
        if type(axis) is int:
            base = list(primary_shapes[0])
            if axis < 0:
                axis += len(base) + 1
            if 0 <= axis <= len(base):
                base.insert(axis, len(primary_shapes))
                return (tuple(base),)
    if name == "concatenate" and primary_shapes:
        axis_value = _call_argument(args, kwargs, 1, "axis")
        if axis_value is None:
            return ((sum(math.prod(shape) for shape in primary_shapes),),)
        axis = (
            0
            if axis_value is _MISSING
            else _integer_call_argument(args, kwargs, 1, "axis")
        )
        if axis is not None:
            if axis < 0:
                axis += len(primary_shapes[0])
            if 0 <= axis < len(primary_shapes[0]):
                result = list(primary_shapes[0])
                result[axis] = sum(shape[axis] for shape in primary_shapes)
                return (tuple(result),)
    if name in H8NumpyAllocationGuard._LINALG_NAMES:
        return _infer_numpy_linalg_shapes(name, args, kwargs, primary_shapes)
    return None


def _numpy_literal_shape(value: object) -> tuple[int, ...] | None:
    if isinstance(value, (np.ndarray, np.generic)):
        return tuple(value.shape)
    if isinstance(value, (tuple, list)):
        if not value:
            return (0,)
        child_shapes = tuple(_numpy_literal_shape(item) for item in value)
        if any(shape is None for shape in child_shapes):
            return None
        first = child_shapes[0]
        if any(shape != first for shape in child_shapes):
            return None
        return (len(value),) + (() if first is None else first)
    if isinstance(value, (str, bytes, Mapping)):
        return None
    return ()


def _numpy_broadcast_shape(
    shapes: tuple[tuple[int, ...], ...],
) -> tuple[int, ...] | None:
    if not shapes:
        return ()
    result: list[int] = []
    maximum_rank = max(len(shape) for shape in shapes)
    for offset in range(1, maximum_rank + 1):
        dimensions = tuple(
            shape[-offset] if len(shape) >= offset else 1
            for shape in shapes
        )
        nonunit = {dimension for dimension in dimensions if dimension != 1}
        if len(nonunit) > 1:
            return None
        result.append(next(iter(nonunit), 1))
    return tuple(reversed(result))


def _numpy_output_shape_variants(
    name: str,
    inferred_shapes: tuple[tuple[int, ...], ...] | None,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    if inferred_shapes is None:
        return ()
    variants = (inferred_shapes,)
    if name != "lstsq" or len(inferred_shapes) != 4 or inferred_shapes[1] == (0,):
        return variants
    empty_residuals = (
        inferred_shapes[0],
        (0,),
        inferred_shapes[2],
        inferred_shapes[3],
    )
    return variants + (empty_residuals,)


_MISSING = object()


def _call_argument(
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
    index: int,
    *names: str,
) -> object:
    if len(args) > index:
        return args[index]
    for name in names:
        if name in kwargs:
            return kwargs[name]
    return _MISSING


def _integer_call_argument(
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
    index: int,
    *names: str,
) -> int | None:
    value = _call_argument(args, kwargs, index, *names)
    if type(value) is int:
        return value
    if isinstance(value, np.integer) and not isinstance(value, np.bool_):
        return int(value)
    return None


def _numpy_primary_operand_shapes(
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
    *,
    shape_resolver: Callable[
        [np.ndarray[Any, Any] | np.generic],
        tuple[int, ...],
    ],
) -> tuple[tuple[int, ...], ...]:
    operand = _call_argument(
        args,
        kwargs,
        0,
        "a",
        "arrays",
        "tensors",
        "x",
        "x1",
    )
    if operand is _MISSING:
        return ()
    first = tuple(shape_resolver(item) for item in _walk_arrays(operand))
    if isinstance(operand, (tuple, list)):
        return first
    second = _call_argument(args, kwargs, 1, "b", "x2")
    if second is _MISSING:
        return first
    return first + tuple(shape_resolver(item) for item in _walk_arrays(second))


def _infer_numpy_linalg_shapes(
    name: str,
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
    operands: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...] | None:
    if not operands:
        return None
    matrix = operands[0]
    if name == "vector_norm":
        return (
            _numpy_reduction_shape(
                matrix,
                _call_argument(args, kwargs, 1, "axis"),
                _bool_call_argument(args, kwargs, 2, "keepdims", False),
            ),
        )
    if name == "norm" and len(matrix) < 2:
        return (
            _numpy_reduction_shape(
                matrix,
                _call_argument(args, kwargs, 2, "axis"),
                _bool_call_argument(args, kwargs, 3, "keepdims", False),
            ),
        )
    if len(matrix) < 2:
        return None
    batch = matrix[:-2]
    rows, columns = matrix[-2:]
    minimum = min(rows, columns)
    if name in ("cholesky", "inv", "matrix_power"):
        return (matrix,)
    if name in ("eig", "eigh"):
        return (batch + (rows,), matrix)
    if name in ("eigvals", "eigvalsh", "svdvals"):
        return (batch + (minimum if name == "svdvals" else rows,),)
    if name == "pinv":
        return (batch + (columns, rows),)
    if name == "solve" and len(operands) >= 2:
        solution = _numpy_solve_shape(matrix, operands[1])
        return None if solution is None else (solution,)
    if name == "svd":
        compute_uv = _bool_call_argument(args, kwargs, 2, "compute_uv", True)
        if not compute_uv:
            return (batch + (minimum,),)
        full = _bool_call_argument(args, kwargs, 1, "full_matrices", True)
        u_columns = rows if full else minimum
        vh_rows = columns if full else minimum
        return (
            batch + (rows, u_columns),
            batch + (minimum,),
            batch + (vh_rows, columns),
        )
    if name == "qr":
        mode_value = _call_argument(args, kwargs, 1, "mode")
        mode = "reduced" if mode_value is _MISSING else mode_value
        if mode == "r":
            return (batch + (minimum, columns),)
        if mode == "raw":
            return (batch + (columns, rows), batch + (minimum,))
        if mode == "complete":
            return (
                batch + (rows, rows),
                batch + (rows, columns),
            )
        return (
            batch + (rows, minimum),
            batch + (minimum, columns),
        )
    if name == "slogdet":
        return (batch, batch)
    if name in ("det", "cond", "matrix_rank"):
        return (batch,)
    if name == "matrix_norm":
        keepdims = _bool_call_argument(args, kwargs, 1, "keepdims", False)
        axes: object = (-2, -1)
        return (_numpy_reduction_shape(matrix, axes, keepdims),)
    if name == "norm":
        return (
            _numpy_reduction_shape(
                matrix,
                _call_argument(args, kwargs, 2, "axis"),
                _bool_call_argument(args, kwargs, 3, "keepdims", False),
            ),
        )
    if name == "tensorinv":
        ind = _integer_call_argument(args, kwargs, 1, "ind")
        split = 2 if ind is None else ind
        if not 0 < split < len(matrix):
            return None
        return (matrix[split:] + matrix[:split],)
    if name == "tensorsolve" and len(operands) >= 2:
        solution_rank = len(matrix) - len(operands[1])
        if solution_rank < 0:
            return None
        return (matrix[:solution_rank],)
    if name == "multi_dot":
        result = operands[0]
        for operand in operands[1:]:
            result = _numpy_matmul_shape(result, operand)
            if result is None:
                return None
        return (result,)
    if name == "lstsq" and len(operands) >= 2:
        if len(matrix) != 2:
            raise H8ForbiddenAllocation(
                "numpy.linalg.lstsq requires a two-dimensional coefficient matrix"
            )
        rhs = operands[1]
        if len(rhs) not in (1, 2) or rhs[0] != rows:
            raise H8ForbiddenAllocation(
                "numpy.linalg.lstsq right-hand side is incompatible with its matrix"
            )
        solution = (columns,) if len(rhs) == 1 else (columns, rhs[1])
        residuals = (0,) if rows <= columns else ((1,) if len(rhs) == 1 else (rhs[1],))
        return (
            solution,
            residuals,
            (),
            (minimum,),
        )
    return None


def _bool_call_argument(
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
    index: int,
    name: str,
    default: bool,
) -> bool:
    value = _call_argument(args, kwargs, index, name)
    return default if value is _MISSING else bool(value)


def _numpy_solve_shape(
    matrix: tuple[int, ...],
    rhs: tuple[int, ...],
) -> tuple[int, ...] | None:
    if not rhs:
        return None
    matrix_batch = matrix[:-2]
    dimension = matrix[-1]
    if len(rhs) == len(matrix) - 1:
        try:
            batch = np.broadcast_shapes(matrix_batch, rhs[:-1])
        except ValueError:
            return None
        return tuple(batch) + (dimension,)
    if len(rhs) < 2:
        return None
    try:
        batch = np.broadcast_shapes(matrix_batch, rhs[:-2])
    except ValueError:
        return None
    return tuple(batch) + (dimension, rhs[-1])


def _numpy_reduction_shape(
    shape: tuple[int, ...],
    axis_value: object,
    keepdims: bool,
) -> tuple[int, ...]:
    if axis_value is _MISSING or axis_value is None:
        return tuple(1 for _ in shape) if keepdims else ()
    if type(axis_value) is int or (
        isinstance(axis_value, np.integer) and not isinstance(axis_value, np.bool_)
    ):
        axes = (int(axis_value),)
    else:
        axes = tuple(
            int(axis)
            for axis in tuple(axis_value)
            if type(axis) is int
            or (isinstance(axis, np.integer) and not isinstance(axis, np.bool_))
        )
    normalized = {axis + len(shape) if axis < 0 else axis for axis in axes}
    if keepdims:
        return tuple(
            1 if index in normalized else value for index, value in enumerate(shape)
        )
    return tuple(value for index, value in enumerate(shape) if index not in normalized)


def _numpy_shape_argument(value: object) -> tuple[int, ...] | None:
    if type(value) is int:
        return (value,)
    if isinstance(value, np.integer) and not isinstance(value, np.bool_):
        return (int(value),)
    if isinstance(value, (tuple, list)) and all(
        type(item) is int
        or (isinstance(item, np.integer) and not isinstance(item, np.bool_))
        for item in value
    ):
        return tuple(int(item) for item in value)
    return None


def _resolve_inferred_dimension(
    requested: tuple[int, ...],
    source: tuple[int, ...],
) -> tuple[int, ...]:
    if requested.count(-1) != 1:
        return requested
    known = math.prod(dimension for dimension in requested if dimension != -1)
    source_count = math.prod(source)
    if known <= 0 or source_count % known:
        return requested
    inferred = source_count // known
    return tuple(inferred if dimension == -1 else dimension for dimension in requested)


def _numpy_matmul_shape(
    left: tuple[int, ...],
    right: tuple[int, ...],
) -> tuple[int, ...] | None:
    if not left or not right:
        return None
    if len(left) == 1 and len(right) == 1:
        return ()
    if len(left) == 1:
        return right[:-2] + (right[-1],)
    if len(right) == 1:
        return left[:-1]
    try:
        batch = np.broadcast_shapes(left[:-2], right[:-2])
    except ValueError:
        return None
    return tuple(batch) + (left[-2], right[-1])


def _numpy_requested_itemsize(
    name: str,
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
    inputs: Sequence[np.ndarray[Any, Any] | np.generic],
) -> int:
    if name in ("all", "isfinite"):
        return 1
    dtype: object = kwargs.get("dtype", _MISSING)
    if dtype is _MISSING:
        positional_dtype_index = {
            "asarray": 1,
            "ascontiguousarray": 1,
            "empty": 1,
            "zeros": 1,
            "ones": 1,
            "full": 2,
            "eye": 3,
            "identity": 1,
        }.get(name)
        if positional_dtype_index is not None and len(args) > positional_dtype_index:
            dtype = args[positional_dtype_index]
    if dtype is not _MISSING and dtype is not None:
        try:
            return int(np.dtype(dtype).itemsize)
        except (TypeError, ValueError) as error:
            raise H8ForbiddenAllocation(
                "NumPy dtype is not classifiable before allocation"
            ) from error
    if inputs:
        return int(inputs[0].dtype.itemsize)
    if name == "full":
        fill_value = _call_argument(args, kwargs, 1, "fill_value")
        if isinstance(fill_value, (complex, np.complexfloating)):
            return 16
        if isinstance(fill_value, (bool, np.bool_)):
            return 1
    return 8


def parse_h8_lossy_profiler_rows(
    rows: Iterable[H8LossyProfilerRow | tuple[object, object, object, object]],
) -> tuple[H8LossyProfilerRow, ...]:
    result: list[H8LossyProfilerRow] = []
    for row in rows:
        if type(row) is H8LossyProfilerRow:
            result.append(row)
            continue
        if type(row) is not tuple or len(row) != 4:
            raise H8ProfilerObservabilityGap(
                "lossy profiler row lacks its four documented fields"
            )
        try:
            result.append(
                H8LossyProfilerRow(
                    timestamp_ns=row[0],  # type: ignore[arg-type]
                    action=row[1],  # type: ignore[arg-type]
                    nbytes=row[2],  # type: ignore[arg-type]
                    category=row[3],  # type: ignore[arg-type]
                )
            )
        except (TypeError, ValueError) as error:
            raise H8ProfilerObservabilityGap(
                f"invalid lossy profiler row: {error}"
            ) from error
    return tuple(result)


def parse_h8_profiler_events(
    raw_rows: Iterable[H8RawProfilerEvent | Mapping[str, object] | H8LossyProfilerRow],
    enrichments: Iterable[H8ProfilerEnrichment | Mapping[str, object]],
    *,
    policy: H8AllocationPolicy,
) -> H8ProfilerTrace:
    """Join, deduplicate, and reconstruct the pinned raw profiler timeline."""

    converted_rows: list[H8RawProfilerEvent] = []
    for value in raw_rows:
        if type(value) is H8LossyProfilerRow:
            raise H8ProfilerObservabilityGap(
                "lossy profiler rows cannot satisfy raw-event enrichment"
            )
        converted_rows.append(_coerce_raw_profiler_event(value))
    converted_enrichments = tuple(
        _coerce_profiler_enrichment(value) for value in enrichments
    )
    rows = _deduplicate_exact(converted_rows)
    if not rows:
        raise H8ProfilerObservabilityGap("raw profiler primary timeline is empty")
    observed_actions = {row.action for row in rows}
    if observed_actions != set(_PROFILER_ACTIONS):
        missing_actions = tuple(
            action for action in _PROFILER_ACTIONS if action not in observed_actions
        )
        raise H8ProfilerObservabilityGap(
            f"raw profiler primary-action coverage is partial: {missing_actions!r}"
        )
    if not any(row.action == "PREEXISTING" for row in rows):
        raise H8ProfilerObservabilityGap(
            "raw profiler baseline PREEXISTING witness is absent"
        )
    source_rows = tuple(row.source_row_index for row in rows)
    if len(set(source_rows)) != len(source_rows):
        raise H8ProfilerObservabilityGap(
            "raw profiler source-row indices are not unique"
        )
    joined = _join_profiler_enrichment(rows, converted_enrichments)
    ordered = sorted(rows, key=lambda item: (item.timestamp_ns, item.source_row_index))
    joined_keys_by_tensor_id: dict[int, set[H8TensorKey]] = {}
    for enrichment in joined.values():
        joined_keys_by_tensor_id.setdefault(
            enrichment.tensor_key.tensor_id,
            set(),
        ).add(enrichment.tensor_key)

    live: dict[H8TensorKey, tuple[int, int, bool]] = {}
    live_spans: dict[H8TensorKey, tuple[int, int, int]] = {}
    storage_live: dict[tuple[int, str], set[H8TensorKey]] = {}
    storage_sizes: dict[tuple[int, str], int] = {}
    records: list[H8ProfilerEventRecord] = []
    baseline_storage: set[tuple[int, str]] = set()
    established_keys: set[H8TensorKey] = set()
    established_storage: set[tuple[int, str]] = set()
    baseline_bytes = 0
    live_bytes = 0
    peak = 0
    saw_nonpreexisting = False
    for row in ordered:
        enrichment = joined[_join_key(row)]
        _validate_profiler_classification(policy, row, enrichment)
        alias_target = _validate_profiler_storage_witness(
            row,
            enrichment,
            live=live,
            live_spans=live_spans,
            established_keys=established_keys,
            joined_keys_by_tensor_id=joined_keys_by_tensor_id,
        )
        storage_identity = (
            row.tensor_key.allocation_id,
            row.tensor_key.device,
        )
        storage_members = storage_live.get(storage_identity)
        if row.action == "PREEXISTING":
            if saw_nonpreexisting:
                raise H8ForbiddenAllocation(
                    "PREEXISTING appears after the baseline timeline"
                )
            if row.nbytes <= 0:
                raise H8ForbiddenAllocation("PREEXISTING requires positive bytes")
            if row.tensor_key in established_keys:
                raise H8ForbiddenAllocation("duplicate PREEXISTING profiler identity")
            _require_new_profiler_storage_identity(
                storage_identity,
                alias_target=alias_target,
                established_storage=established_storage,
                storage_members=storage_members,
            )
            live[row.tensor_key] = (row.version, row.nbytes, True)
            live_spans[row.tensor_key] = _profiler_span(enrichment)
            if storage_members is None:
                storage_members = set()
                storage_live[storage_identity] = storage_members
                storage_sizes[storage_identity] = row.nbytes
                baseline_bytes += row.nbytes
                live_bytes += row.nbytes
            storage_members.add(row.tensor_key)
            baseline_storage.add(storage_identity)
            established_keys.add(row.tensor_key)
            established_storage.add(storage_identity)
        elif row.action == "CREATE":
            saw_nonpreexisting = True
            if row.version != 0 or row.nbytes <= 0:
                raise H8ForbiddenAllocation(
                    "CREATE requires a dead identity at version zero"
                )
            if row.tensor_key in established_keys:
                raise H8ForbiddenAllocation("duplicate CREATE profiler identity")
            _require_new_profiler_storage_identity(
                storage_identity,
                alias_target=alias_target,
                established_storage=established_storage,
                storage_members=storage_members,
            )
            live[row.tensor_key] = (0, row.nbytes, False)
            live_spans[row.tensor_key] = _profiler_span(enrichment)
            if storage_members is None:
                storage_members = set()
                storage_live[storage_identity] = storage_members
                storage_sizes[storage_identity] = row.nbytes
                live_bytes += row.nbytes
            storage_members.add(row.tensor_key)
            established_keys.add(row.tensor_key)
            established_storage.add(storage_identity)
        elif row.action == "INCREMENT_VERSION":
            saw_nonpreexisting = True
            state = live.get(row.tensor_key)
            if state is None:
                raise H8ForbiddenAllocation(
                    "version increment targets an unknown identity"
                )
            version, nbytes, baseline = state
            if row.version != version + 1 or row.nbytes != 0:
                raise H8ForbiddenAllocation(
                    "profiler version transition is nonmonotone"
                )
            live[row.tensor_key] = (row.version, nbytes, baseline)
        else:
            saw_nonpreexisting = True
            state = live.get(row.tensor_key)
            if state is None:
                raise H8ForbiddenAllocation(
                    "DESTROY targets an unknown or already dead identity"
                )
            version, nbytes, _baseline = state
            if row.version != version or row.nbytes != -nbytes:
                raise H8ForbiddenAllocation(
                    "DESTROY has wrong version or identity byte drift"
                )
            del live[row.tensor_key]
            live_spans.pop(row.tensor_key)
            storage_members = storage_live.get(storage_identity)
            if storage_members is None or row.tensor_key not in storage_members:
                raise H8ForbiddenAllocation("DESTROY storage membership is not live")
            storage_members.remove(row.tensor_key)
            if not storage_members:
                storage_live.pop(storage_identity)
                live_bytes -= storage_sizes[storage_identity]
        if live_bytes < 0:
            raise H8ForbiddenAllocation("profiler live-byte total became negative")
        peak = max(peak, live_bytes)
        witness = _canonical_sha256(
            {
                "raw": row,
                "enrichment": enrichment,
            }
        )
        records.append(
            H8ProfilerEventRecord(
                source_row_index=row.source_row_index,
                timestamp_ns=row.timestamp_ns,
                action=row.action,
                tensor_key=row.tensor_key,
                version=row.version,
                nbytes=row.nbytes,
                dtype=enrichment.dtype,
                device=row.tensor_key.device,
                operator=enrichment.operator,
                stack=enrichment.stack,
                logical_shape=enrichment.logical_shape,
                classification=enrichment.classification,
                matched_event_node_indices=(enrichment.matched_event_node_indices),
                join_witness_sha256=witness,
                live_bytes_after=live_bytes,
            )
        )
    leaked = tuple(key for key, state in live.items() if not state[2])
    if leaked:
        raise H8ForbiddenAllocation("profiler leaked a created identity")
    trace_hash = _canonical_sha256(records)
    return H8ProfilerTrace(
        events=tuple(records),
        preexisting_storage_count=len(baseline_storage),
        preexisting_bytes=baseline_bytes,
        baseline_live_bytes=baseline_bytes,
        live_peak_bytes=peak,
        all_joined_and_liveness_reconciled=True,
        trace_sha256=trace_hash,
    )


def _coerce_raw_profiler_event(
    value: H8RawProfilerEvent | Mapping[str, object],
) -> H8RawProfilerEvent:
    if type(value) is H8RawProfilerEvent:
        return value
    if not isinstance(value, Mapping):
        raise H8ProfilerObservabilityGap(
            "raw profiler row lacks its typed source record"
        )
    required = (
        "source_row_index",
        "timestamp_ns",
        "action",
        "tensor_key",
        "version",
        "nbytes",
    )
    missing = tuple(name for name in required if name not in value)
    if missing:
        raise H8ProfilerObservabilityGap(f"raw profiler row is missing {missing!r}")
    try:
        return H8RawProfilerEvent(
            source_row_index=value["source_row_index"],  # type: ignore[arg-type]
            timestamp_ns=value["timestamp_ns"],  # type: ignore[arg-type]
            action=value["action"],  # type: ignore[arg-type]
            tensor_key=value["tensor_key"],  # type: ignore[arg-type]
            version=value["version"],  # type: ignore[arg-type]
            nbytes=value["nbytes"],  # type: ignore[arg-type]
        )
    except (TypeError, ValueError) as error:
        raise H8ProfilerObservabilityGap(
            f"invalid raw profiler row: {error}"
        ) from error


def _coerce_profiler_enrichment(
    value: H8ProfilerEnrichment | Mapping[str, object],
) -> H8ProfilerEnrichment:
    if type(value) is H8ProfilerEnrichment:
        return value
    if not isinstance(value, Mapping):
        raise H8ProfilerObservabilityGap("profiler enrichment is not typed")
    required = (
        "source_row_index",
        "tensor_key",
        "version",
        "dtype",
        "operator",
        "stack",
        "logical_shape",
        "classification",
        "matched_event_node_indices",
        "storage_span_start",
        "storage_span_end",
        "storage_nbytes",
    )
    missing = tuple(name for name in required if name not in value)
    if missing:
        raise H8ProfilerObservabilityGap(f"profiler enrichment is missing {missing!r}")
    try:
        return H8ProfilerEnrichment(
            source_row_index=value["source_row_index"],  # type: ignore[arg-type]
            tensor_key=value["tensor_key"],  # type: ignore[arg-type]
            version=value["version"],  # type: ignore[arg-type]
            dtype=value["dtype"],  # type: ignore[arg-type]
            operator=value["operator"],  # type: ignore[arg-type]
            stack=value["stack"],  # type: ignore[arg-type]
            logical_shape=value["logical_shape"],  # type: ignore[arg-type]
            classification=value["classification"],  # type: ignore[arg-type]
            matched_event_node_indices=value[  # type: ignore[arg-type]
                "matched_event_node_indices"
            ],
            storage_span_start=value["storage_span_start"],  # type: ignore[arg-type]
            storage_span_end=value["storage_span_end"],  # type: ignore[arg-type]
            storage_nbytes=value["storage_nbytes"],  # type: ignore[arg-type]
            alias_of=value.get("alias_of"),  # type: ignore[arg-type]
        )
    except (TypeError, ValueError) as error:
        raise H8ProfilerObservabilityGap(
            f"invalid profiler enrichment: {error}"
        ) from error


def _deduplicate_exact(
    rows: Iterable[H8RawProfilerEvent],
) -> tuple[H8RawProfilerEvent, ...]:
    return tuple(dict.fromkeys(rows))


def _join_key(
    row: H8RawProfilerEvent | H8ProfilerEnrichment,
) -> tuple[int, H8TensorKey, int]:
    return row.source_row_index, row.tensor_key, row.version


def _join_profiler_enrichment(
    rows: Sequence[H8RawProfilerEvent],
    enrichments: Sequence[H8ProfilerEnrichment],
) -> dict[tuple[int, H8TensorKey, int], H8ProfilerEnrichment]:
    candidates: dict[
        tuple[int, H8TensorKey, int],
        set[H8ProfilerEnrichment],
    ] = {}
    for enrichment in enrichments:
        candidates.setdefault(_join_key(enrichment), set()).add(enrichment)
    joined: dict[
        tuple[int, H8TensorKey, int],
        H8ProfilerEnrichment,
    ] = {}
    for row in rows:
        matches = candidates.get(_join_key(row), set())
        if len(matches) != 1:
            raise H8ProfilerObservabilityGap(
                "profiler event-tree join is missing or nonunique"
            )
        joined[_join_key(row)] = next(iter(matches))
    extra = set(candidates) - {_join_key(row) for row in rows}
    if extra:
        raise H8ProfilerObservabilityGap(
            "profiler enrichment has no unique source-row join"
        )
    return joined


def _validate_profiler_classification(
    policy: H8AllocationPolicy,
    row: H8RawProfilerEvent,
    enrichment: H8ProfilerEnrichment,
) -> None:
    if enrichment.tensor_key.device != row.tensor_key.device:
        raise H8ProfilerObservabilityGap(
            "profiler TensorKey device and enrichment disagree"
        )
    if enrichment.classification in ("unclassified", "unknown", ""):
        raise H8ProfilerObservabilityGap("profiler row has no semantic classification")
    classify_h8_operator(
        policy,
        operator=enrichment.operator,
        operand_shapes=(enrichment.logical_shape,),
        output_shape=enrichment.logical_shape,
    )
    policy.preflight_shape(
        enrichment.logical_shape,
        itemsize=_dtype_itemsize(enrichment.dtype),
    )


def _validate_profiler_storage_witness(
    row: H8RawProfilerEvent,
    enrichment: H8ProfilerEnrichment,
    *,
    live: Mapping[H8TensorKey, tuple[int, int, bool]],
    live_spans: Mapping[H8TensorKey, tuple[int, int, int]],
    established_keys: set[H8TensorKey],
    joined_keys_by_tensor_id: Mapping[int, set[H8TensorKey]],
) -> H8TensorKey | None:
    storage_nbytes = enrichment.storage_nbytes
    if (storage_nbytes + 7) // 8 > H8_MAX_STORAGE_SCALARS:
        raise H8ForbiddenAllocation(
            "profiler storage exceeds the float64-equivalent cap"
        )
    logical_nbytes = math.prod(enrichment.logical_shape) * _dtype_itemsize(
        enrichment.dtype
    )
    if row.action in ("PREEXISTING", "CREATE") and row.nbytes != storage_nbytes:
        raise H8ForbiddenAllocation(
            "live-establishing profiler bytes and storage span disagree"
        )
    alias_target = _validate_profiler_logical_span_alias(
        row,
        enrichment,
        logical_nbytes=logical_nbytes,
        storage_nbytes=storage_nbytes,
        live=live,
        live_spans=live_spans,
        established_keys=established_keys,
        joined_keys_by_tensor_id=joined_keys_by_tensor_id,
    )
    if row.action in ("PREEXISTING", "CREATE"):
        return alias_target
    state = live.get(row.tensor_key)
    if state is None:
        return alias_target
    _version, live_nbytes, _baseline = state
    if storage_nbytes != live_nbytes:
        raise H8ForbiddenAllocation(
            "profiler storage span drifted from its live identity"
        )
    if live_spans.get(row.tensor_key) != _profiler_span(enrichment):
        raise H8ForbiddenAllocation(
            "profiler storage span drifted from its live identity"
        )
    return alias_target


def _validate_profiler_logical_span_alias(
    row: H8RawProfilerEvent,
    enrichment: H8ProfilerEnrichment,
    *,
    logical_nbytes: int,
    storage_nbytes: int,
    live: Mapping[H8TensorKey, tuple[int, int, bool]],
    live_spans: Mapping[H8TensorKey, tuple[int, int, int]],
    established_keys: set[H8TensorKey],
    joined_keys_by_tensor_id: Mapping[int, set[H8TensorKey]],
) -> H8TensorKey | None:
    if logical_nbytes > storage_nbytes:
        raise H8ForbiddenAllocation("profiler logical tensor exceeds its storage span")
    alias = enrichment.alias_of
    if alias is None:
        if logical_nbytes != storage_nbytes:
            raise H8ProfilerObservabilityGap(
                "profiler subspan needs an explicit alias witness"
            )
        return None
    candidates = joined_keys_by_tensor_id.get(alias.tensor_id, set())
    if not candidates:
        raise H8ProfilerObservabilityGap(
            "profiler alias target is nonexistent in the joined timeline"
        )
    if len(candidates) != 1:
        raise H8ProfilerObservabilityGap(
            "profiler alias target is ambiguous in the joined timeline"
        )
    target = next(iter(candidates))
    if target != alias:
        raise H8ProfilerObservabilityGap(
            "profiler alias target does not resolve to its exact joined TensorKey"
        )
    if target not in live or target not in live_spans:
        state = "destroyed" if target in established_keys else "future"
        raise H8ProfilerObservabilityGap(
            f"profiler alias target is {state}, not already live"
        )
    if (
        target.allocation_id,
        target.device,
    ) != (
        row.tensor_key.allocation_id,
        row.tensor_key.device,
    ):
        raise H8ForbiddenAllocation(
            "profiler alias target names an incompatible storage allocation or device"
        )
    target_start, target_end, target_nbytes = live_spans[target]
    current_start, current_end, _current_nbytes = _profiler_span(enrichment)
    if (
        target_end - target_start != target_nbytes
        or target_start > current_start
        or current_end > target_end
        or (target_start == current_start and target_end == current_end)
    ):
        raise H8ForbiddenAllocation(
            "profiler alias target lacks a strict exact containing storage span"
        )
    return target


def _profiler_span(enrichment: H8ProfilerEnrichment) -> tuple[int, int, int]:
    return (
        enrichment.storage_span_start,
        enrichment.storage_span_end,
        enrichment.storage_nbytes,
    )


def _dtype_itemsize(dtype: str) -> int:
    normalized = dtype.lower()
    for token, itemsize in (
        ("float64", 8),
        ("double", 8),
        ("int64", 8),
        ("float32", 4),
        ("int32", 4),
        ("float16", 2),
        ("bfloat16", 2),
        ("int16", 2),
        ("bool", 1),
        ("int8", 1),
        ("uint8", 1),
    ):
        if token in normalized:
            return itemsize
    raise H8ProfilerObservabilityGap("profiler dtype is not classifiable")


def _require_new_profiler_storage_identity(
    storage_identity: tuple[int, str],
    *,
    alias_target: H8TensorKey | None,
    established_storage: set[tuple[int, str]],
    storage_members: set[H8TensorKey] | None,
) -> None:
    if alias_target is None:
        if storage_identity in established_storage or storage_members:
            raise H8ForbiddenAllocation(
                "profiler storage identity is already established"
            )
        return
    if (
        storage_identity not in established_storage
        or storage_members is None
        or alias_target not in storage_members
    ):
        raise H8ForbiddenAllocation(
            "profiler alias target is not a live member of its storage allocation"
        )


def cross_check_h8_backend_dispatch(
    *,
    layout: BlockChainLayout,
    counters: BackendCounterSnapshot,
    storage: BlockStorageRecord,
    workspace: BlockWorkspaceRecord,
    dispatch: H8DispatchTrace,
) -> H8AllocationCrossCheck:
    """Reconcile immutable Task 2 counters with primary dispatch evidence."""

    obligations: list[str] = []
    events = tuple(
        event
        for event in dispatch.events
        if event.control_id is None and event.forbidden_reason is None
    )
    if not events:
        obligations.append("dispatch_event_stream_empty")
    if counters.layout != layout or storage.layout != layout:
        obligations.append("layout_identity_mismatch")
    expected_storage = {
        "precision.diagonal": layout.diagonal_scalar_count,
        "precision.lower": layout.lower_scalar_count,
        "factor.diagonal": layout.diagonal_scalar_count,
        "factor.lower": layout.lower_scalar_count,
        "selected.diagonal": layout.diagonal_scalar_count,
        "selected.lower": layout.lower_scalar_count,
        "information": layout.information_scalar_count,
    }
    observed_storage = _dispatch_site_scalar_peaks(events)
    for site, expected in expected_storage.items():
        observed = observed_storage.get(site, 0)
        if observed != expected:
            obligations.append(
                f"{site}_dispatch_scalars_{observed}_expected_{expected}"
            )
    exact_storage_fields = (
        ("precision_scalar_count", layout.band_storage_scalar_count),
        ("factor_scalar_count", layout.band_storage_scalar_count),
        ("selected_inverse_scalar_count", layout.band_storage_scalar_count),
        ("information_scalar_count", layout.information_scalar_count),
        ("upper_block_scalar_count", 0),
    )
    for name, expected in exact_storage_fields:
        if getattr(storage, name) != expected:
            obligations.append(f"{name}_not_exact_{expected}")

    observed_rhs_width = _dispatch_rhs_width(events, layout, method="solve")
    observed_sample_width = _dispatch_rhs_width(events, layout, method="sample")
    if (
        counters.maximum_rhs_width <= 0
        or counters.maximum_rhs_width > layout.block_size
        or observed_rhs_width != counters.maximum_rhs_width
    ):
        obligations.append("maximum_rhs_width_dispatch_counter_mismatch")
    if (
        counters.maximum_sample_rhs_width != 1
        or observed_sample_width != counters.maximum_sample_rhs_width
    ):
        obligations.append("sample_width_dispatch_counter_mismatch")
    if workspace.maximum_rhs_width != counters.maximum_rhs_width:
        obligations.append("workspace_rhs_counter_mismatch")
    if workspace.maximum_scalar_count > H8_MAX_STORAGE_SCALARS:
        obligations.append("workspace_storage_cap_exceeded")
    workspace_shapes = tuple(
        shape
        for event in events
        if event.semantic_site == "workspace"
        for shape in event.output_shapes
    )
    observed_workspace_max = max(
        (math.prod(shape) for shape in workspace_shapes),
        default=0,
    )
    if (
        not workspace_shapes
        or workspace.maximum_shape not in workspace_shapes
        or observed_workspace_max != workspace.maximum_scalar_count
    ):
        obligations.append("workspace_dispatch_shape_scalar_mismatch")
    forbidden_backend = counters.attempted_forbidden_selected_blocks + len(
        counters.attempted_forbidden_rhs_widths
    )
    operation_fields = (
        (
            "factorize",
            counters.factorization_calls,
            "cholesky_ex",
            layout.population_size,
            None,
        ),
        (
            "solve_factor",
            counters.forward_substitution_calls + counters.backward_substitution_calls,
            "solve_triangular",
            layout.population_size,
            None,
        ),
        (
            "solve",
            counters.solve_calls,
            "solve_triangular",
            2 * layout.population_size,
            None,
        ),
        ("logdet", counters.logdet_calls, "::log", 1, None),
        (
            "selected_inverse",
            counters.selected_inverse_calls,
            "solve_triangular",
            layout.population_size,
            None,
        ),
        (
            "sample",
            counters.sample_calls,
            "solve_triangular",
            layout.population_size,
            None,
        ),
        (
            "quadratic",
            counters.quadratic_calls,
            "::dot",
            layout.population_size,
            None,
        ),
        (
            "trace_inverse_product",
            counters.trace_calls,
            "::add",
            1,
            (),
        ),
        (
            "sparse_matvec",
            counters.sparse_matvec_calls,
            "stack",
            1,
            (layout.population_size, layout.block_size),
        ),
    )
    reconciled: list[tuple[str, int, int]] = []
    for (
        operation,
        backend_count,
        kernel,
        kernels_per_call,
        output_shape,
    ) in operation_fields:
        raw_dispatch_count = _dispatch_kernel_count(
            events,
            method=operation,
            operator_token=kernel,
            output_shape=output_shape,
        )
        dispatch_count, remainder = divmod(
            raw_dispatch_count,
            kernels_per_call,
        )
        reconciled.append((operation, backend_count, dispatch_count))
        if backend_count <= 0:
            obligations.append(f"{operation}_backend_coverage_is_zero")
        if remainder or raw_dispatch_count != kernels_per_call * backend_count:
            obligations.append(f"{operation}_dispatch_counter_mismatch")
    if (
        counters.forward_substitution_calls <= 0
        or counters.backward_substitution_calls <= 0
    ):
        obligations.append("forward_backward_backend_coverage_is_zero")
    if not counters.selected_coverage_complete:
        obligations.append("selected_inverse_coverage_incomplete")
    if any(event.semantic_site is None for event in events):
        obligations.append("unregistered_dispatch_result")
    production_forbidden = sum(
        event.control_id is None and event.forbidden_reason is not None
        for event in dispatch.events
    )
    if forbidden_backend != 0 or production_forbidden != 0:
        obligations.append("production_forbidden_attempt_observed")
    if forbidden_backend != production_forbidden:
        obligations.append("backend_dispatch_forbidden_count_mismatch")
    return H8AllocationCrossCheck(
        complete=not obligations,
        obligations=tuple(obligations),
        backend_forbidden_attempt_count=forbidden_backend,
        dispatch_forbidden_attempt_count=production_forbidden,
        reconciled_operation_counts=tuple(reconciled),
    )


def _dispatch_site_scalar_peaks(
    events: Sequence[H8DispatchEvent],
) -> dict[str, int]:
    peaks: dict[str, int] = {}
    for event in events:
        for site, scalar_count in event.live_float64_equivalent_scalars_by_site:
            peaks[site] = max(peaks.get(site, 0), scalar_count)
    return peaks


def _dispatch_kernel_count(
    events: Sequence[H8DispatchEvent],
    *,
    method: str,
    operator_token: str,
    output_shape: tuple[int, ...] | None,
) -> int:
    method_token = f":{method}"
    return sum(
        any(frame.endswith(method_token) for frame in event.stack)
        and operator_token in event.operator.lower()
        and (output_shape is None or output_shape in event.output_shapes)
        for event in events
    )


def _dispatch_rhs_width(
    events: Sequence[H8DispatchEvent],
    layout: BlockChainLayout,
    *,
    method: str,
) -> int:
    token = f":{method}"
    widths: list[int] = []
    for event in events:
        if not any(frame.endswith(token) for frame in event.stack):
            continue
        for shape in event.input_shapes + event.output_shapes:
            if shape == (layout.population_size, layout.block_size):
                widths.append(1)
            elif len(shape) == 3 and shape[:2] == (
                layout.population_size,
                layout.block_size,
            ):
                widths.append(shape[2])
    return max(widths, default=0)


def h8_negative_control_specs(
    layout: BlockChainLayout,
) -> tuple[H8NegativeControlSpec, ...]:
    if type(layout) is not BlockChainLayout:
        raise ValueError("layout must be an exact BlockChainLayout")
    n = layout.population_size
    b = layout.block_size
    d = layout.dimension
    triangular = n * (n + 1) // 2
    return (
        H8NegativeControlSpec(
            "torch_matrix_d_d",
            "torch.empty",
            ((d, d),),
            ("dispatch",),
            "dense population matrix is forbidden",
        ),
        H8NegativeControlSpec(
            "torch_flat_d2",
            "torch.empty",
            ((d * d,),),
            ("dispatch",),
            "flat dense population storage is forbidden",
        ),
        H8NegativeControlSpec(
            "torch_near_d2",
            "torch.empty",
            ((d - 1, d - 1),),
            ("dispatch",),
            "near dense population storage cap is forbidden",
        ),
        H8NegativeControlSpec(
            "torch_length_d",
            "torch.empty",
            ((d,),),
            ("dispatch",),
            "global axis D is forbidden",
        ),
        H8NegativeControlSpec(
            "torch_block_pair_slab",
            "torch.empty",
            ((n, n, b, b),),
            ("dispatch",),
            "population pair slab is forbidden",
        ),
        H8NegativeControlSpec(
            "torch_triangular_pair_storage",
            "torch.empty",
            ((triangular, b, b),),
            ("dispatch",),
            "triangular pair storage is forbidden",
        ),
        H8NegativeControlSpec(
            "torch_pair_stack",
            "torch.stack",
            ((b, b), (n * n, b, b)),
            ("dispatch",),
            "combined pair slab is forbidden",
        ),
        H8NegativeControlSpec(
            "torch_eye_full_rhs",
            "torch.eye",
            ((d, d), (d, d)),
            ("backend", "dispatch"),
            "dense global identity is forbidden",
        ),
        H8NegativeControlSpec(
            "torch_dense_eigvalsh",
            "torch.linalg.eigvalsh",
            ((d, d), (d,)),
            ("dispatch",),
            "dense population linear-algebra operator is forbidden",
        ),
        H8NegativeControlSpec(
            "numpy_matrix_d_d",
            "numpy.empty",
            ((d, d),),
            ("numpy_guard",),
            "dense population matrix is forbidden",
        ),
        H8NegativeControlSpec(
            "numpy_outer_d_d",
            "numpy.outer",
            ((d,), (d,), (d, d)),
            ("numpy_guard",),
            "global axis D is forbidden",
        ),
        H8NegativeControlSpec(
            "numpy_matmul_d_d",
            "numpy.matmul",
            ((d, 1), (1, d), (d, d)),
            ("numpy_guard",),
            "global axis D is forbidden",
        ),
    )


def execute_h8_torch_negative_control_evidence(
    control_id: str,
    trace: H8DispatchTrace,
    *,
    backend: object | None = None,
    pair_member_meta: Tensor | None = None,
    full_rhs_meta: Tensor | None = None,
    dense_matrix_meta: Tensor | None = None,
) -> tuple[H8ControlResult, object | None]:
    """Execute one assigned Torch control under dispatch pre-execution guards.

    Meta operands are caller-supplied so this function contains no unguarded
    constructor.  The full-width RHS is additionally offered to the backend,
    whose immutable counter delta supplies the second assigned channel.
    """

    spec = _negative_control_spec(trace.policy.layout, control_id)
    if "dispatch" not in spec.assigned_channels:
        raise ValueError("control is not assigned to Torch dispatch")
    before_events = len(trace.events)
    returned = False
    caught_forbidden = False
    with trace:
        with trace.negative_control(control_id):
            try:
                _invoke_torch_control(
                    spec,
                    pair_member_meta=pair_member_meta,
                    dense_matrix_meta=dense_matrix_meta,
                )
                returned = True
            except H8ForbiddenAllocation:
                caught_forbidden = True
            except Exception:
                pass
    observed: list[str] = []
    new_events = tuple(trace.events[before_events:])
    control_events = tuple(
        event for event in new_events if event.control_id == control_id
    )
    executed_or_post_rejected = returned or any(
        event.executed for event in control_events
    )
    dispatch_detected = h8_control_detected_pre_execution(
        spec,
        control_events,
        operation_returned=returned,
        caught_forbidden=caught_forbidden,
    )
    if control_events:
        observed.append("dispatch")
    detected = dispatch_detected

    backend_witness: object | None = None
    if control_id == "torch_eye_full_rhs" and backend is not None:
        if full_rhs_meta is None:
            backend_witness = {"available": False, "reason": "missing_meta_rhs"}
        else:
            counters_before = getattr(backend, "counters", None)
            widths_before = (
                ()
                if counters_before is None
                else counters_before.attempted_forbidden_rhs_widths
            )
            backend_returned = False
            backend_unexpected_exception: str | None = None
            try:
                backend.solve(full_rhs_meta)
                backend_returned = True
            except (TypeError, ValueError, H8ForbiddenAllocation):
                pass
            except Exception as error:
                backend_unexpected_exception = type(error).__name__
            counters_after = getattr(backend, "counters", None)
            widths_after = (
                ()
                if counters_after is None
                else counters_after.attempted_forbidden_rhs_widths
            )
            backend_detected = (
                not backend_returned
                and backend_unexpected_exception is None
                and len(widths_after) == len(widths_before) + 1
                and widths_after[-1] == trace.policy.layout.dimension
            )
            backend_witness = {
                "before": widths_before,
                "after": widths_after,
                "detected": backend_detected,
                "executed_past_detector": (
                    backend_returned or backend_unexpected_exception is not None
                ),
                "unexpected_exception": backend_unexpected_exception,
            }
            if backend_detected:
                observed.append("backend")
            detected = detected and backend_detected
    payload = (
        None
        if not new_events and backend_witness is None
        else {
            "dispatch": new_events,
            "backend": backend_witness,
            "operation_returned": returned,
            "caught_forbidden": caught_forbidden,
            "pre_execution_detected": dispatch_detected,
            "executed_past_detector": executed_or_post_rejected,
        }
    )
    result = make_h8_control_result(
        spec,
        observed_channels=tuple(observed),
        detected=detected,
        event_payload=payload,
    )
    return result, payload


def execute_h8_torch_negative_control(
    control_id: str,
    trace: H8DispatchTrace,
    *,
    backend: object | None = None,
    pair_member_meta: Tensor | None = None,
    full_rhs_meta: Tensor | None = None,
    dense_matrix_meta: Tensor | None = None,
) -> H8ControlResult:
    """Execute one Torch control and return its typed decision summary."""

    result, _payload = execute_h8_torch_negative_control_evidence(
        control_id,
        trace,
        backend=backend,
        pair_member_meta=pair_member_meta,
        full_rhs_meta=full_rhs_meta,
        dense_matrix_meta=dense_matrix_meta,
    )
    return result


def _invoke_torch_control(
    spec: H8NegativeControlSpec,
    *,
    pair_member_meta: Tensor | None,
    dense_matrix_meta: Tensor | None,
) -> object:
    shape = spec.logical_shapes[-1]
    if spec.control_id == "torch_matrix_d_d":
        return torch.empty(shape, dtype=torch.float64, device="cpu")
    if spec.control_id == "torch_flat_d2":
        return torch.empty(shape, dtype=torch.float64, device="cpu")
    if spec.control_id == "torch_near_d2":
        return torch.empty(shape, dtype=torch.float64, device="cpu")
    if spec.control_id == "torch_length_d":
        return torch.empty(shape, dtype=torch.float64, device="cpu")
    if spec.control_id == "torch_block_pair_slab":
        return torch.empty(shape, dtype=torch.float64, device="cpu")
    if spec.control_id == "torch_triangular_pair_storage":
        return torch.empty(shape, dtype=torch.float64, device="cpu")
    if spec.control_id == "torch_pair_stack":
        if pair_member_meta is None:
            raise ValueError("torch_pair_stack requires a caller-supplied meta block")
        member_count = spec.logical_shapes[-1][0]
        return torch.stack([pair_member_meta] * member_count, dim=0)
    if spec.control_id == "torch_eye_full_rhs":
        dimension = spec.logical_shapes[0][0]
        return torch.eye(dimension, dtype=torch.float64, device="cpu")
    if spec.control_id == "torch_dense_eigvalsh":
        if dense_matrix_meta is None:
            raise ValueError(
                "torch_dense_eigvalsh requires a caller-supplied meta matrix"
            )
        return torch.linalg.eigvalsh(dense_matrix_meta)
    raise ValueError("unknown Torch negative control")


def execute_h8_numpy_negative_control_evidence(
    control_id: str,
    guard: H8NumpyAllocationGuard,
    *,
    outer_left: np.ndarray[Any, Any] | None = None,
    outer_right: np.ndarray[Any, Any] | None = None,
    matmul_left: np.ndarray[Any, Any] | None = None,
    matmul_right: np.ndarray[Any, Any] | None = None,
) -> tuple[H8ControlResult, object | None]:
    """Execute one assigned NumPy control with preconstructed safe inputs."""

    spec = _negative_control_spec(guard.policy.layout, control_id)
    if spec.assigned_channels != ("numpy_guard",):
        raise ValueError("control is not assigned solely to the NumPy guard")
    before_events = len(guard.events)
    returned = False
    caught_forbidden = False
    with guard:
        with guard.negative_control(control_id):
            try:
                if control_id == "numpy_matrix_d_d":
                    np.empty(spec.logical_shapes[-1], dtype=np.float64)
                elif control_id == "numpy_outer_d_d":
                    if outer_left is None or outer_right is None:
                        raise ValueError(
                            "numpy_outer_d_d requires preconstructed vectors"
                        )
                    np.outer(outer_left, outer_right)
                elif control_id == "numpy_matmul_d_d":
                    if matmul_left is None or matmul_right is None:
                        raise ValueError(
                            "numpy_matmul_d_d requires preconstructed operands"
                        )
                    np.matmul(matmul_left, matmul_right)
                else:
                    raise ValueError("unknown NumPy negative control")
                returned = True
            except H8ForbiddenAllocation:
                caught_forbidden = True
            except Exception:
                pass
    new_events = tuple(guard.events[before_events:])
    control_events = tuple(
        event for event in new_events if event.control_id == control_id
    )
    executed_or_post_rejected = returned or any(
        event.executed for event in control_events
    )
    detected = h8_control_detected_pre_execution(
        spec,
        control_events,
        operation_returned=returned,
        caught_forbidden=caught_forbidden,
    )
    observed = ("numpy_guard",) if control_events else ()
    payload = (
        {
            "events": new_events,
            "operation_returned": returned,
            "caught_forbidden": caught_forbidden,
            "pre_execution_detected": detected,
            "executed_past_detector": executed_or_post_rejected,
        }
        if new_events
        else None
    )
    result = make_h8_control_result(
        spec,
        observed_channels=observed,
        detected=detected,
        event_payload=payload,
    )
    return result, payload


def execute_h8_numpy_negative_control(
    control_id: str,
    guard: H8NumpyAllocationGuard,
    *,
    outer_left: np.ndarray[Any, Any] | None = None,
    outer_right: np.ndarray[Any, Any] | None = None,
    matmul_left: np.ndarray[Any, Any] | None = None,
    matmul_right: np.ndarray[Any, Any] | None = None,
) -> H8ControlResult:
    """Execute one NumPy control and return its typed decision summary."""

    result, _payload = execute_h8_numpy_negative_control_evidence(
        control_id,
        guard,
        outer_left=outer_left,
        outer_right=outer_right,
        matmul_left=matmul_left,
        matmul_right=matmul_right,
    )
    return result


def h8_control_detected_pre_execution(
    spec: H8NegativeControlSpec,
    events: Sequence[H8DispatchEvent | H8NumpyGuardEvent],
    *,
    operation_returned: bool,
    caught_forbidden: bool,
) -> bool:
    """Return true only for the assigned pre-execution detector and reason."""

    if type(operation_returned) is not bool or type(caught_forbidden) is not bool:
        raise ValueError("control execution flags must be bools")
    if operation_returned or not caught_forbidden or not events:
        return False
    if any(event.executed for event in events):
        return False
    return any(
        event.control_id == spec.control_id
        and event.forbidden_reason == spec.expected_reason
        for event in events
    )


def _negative_control_spec(
    layout: BlockChainLayout,
    control_id: str,
) -> H8NegativeControlSpec:
    matches = tuple(
        spec
        for spec in h8_negative_control_specs(layout)
        if spec.control_id == control_id
    )
    if len(matches) != 1:
        raise ValueError("control_id is outside the frozen inventory")
    return matches[0]


def make_h8_control_result(
    spec: H8NegativeControlSpec,
    *,
    observed_channels: tuple[str, ...],
    detected: bool,
    event_payload: object | None,
) -> H8ControlResult:
    witnessed = event_payload is not None
    event_sha256 = _canonical_sha256(event_payload) if witnessed else None
    assignment_complete = set(spec.assigned_channels).issubset(observed_channels)
    status = (
        GateStatus.INCONCLUSIVE
        if not witnessed
        else GateStatus.FAIL
        if not detected
        else GateStatus.INCONCLUSIVE
        if not assignment_complete
        else GateStatus.PASS
    )
    obligations = (
        ()
        if status is GateStatus.PASS
        else ("assigned_negative_control_channel_unwitnessed",)
        if not witnessed or (detected and not assignment_complete)
        else ("negative_control_executed_past_detector",)
    )
    return H8ControlResult(
        control_id=spec.control_id,
        requested_operation=spec.requested_operation,
        logical_shapes=spec.logical_shapes,
        assigned_channels=spec.assigned_channels,  # type: ignore[arg-type]
        observed_channels=observed_channels,  # type: ignore[arg-type]
        execution_witnessed=witnessed,
        event_sha256=event_sha256,
        assignment_complete=assignment_complete,
        detected=detected,
        status=status,
        obligations=obligations,
    )


__all__ = [
    "H8AllocationCrossCheck",
    "H8AllocationDecision",
    "H8AllocationPolicy",
    "H8DispatchEvent",
    "H8DispatchTrace",
    "H8ForbiddenAllocation",
    "H8NegativeControlSpec",
    "H8NumpyAllocationGuard",
    "H8NumpyGuardEvent",
    "H8ProfilerEnrichment",
    "H8ProfilerObservabilityGap",
    "H8ProfilerTrace",
    "H8RawProfilerEvent",
    "H8StorageSpan",
    "classify_h8_operator",
    "cross_check_h8_backend_dispatch",
    "execute_h8_numpy_negative_control",
    "execute_h8_numpy_negative_control_evidence",
    "execute_h8_torch_negative_control",
    "execute_h8_torch_negative_control_evidence",
    "h8_control_detected_pre_execution",
    "h8_negative_control_specs",
    "h8_tensor_storage_span",
    "make_h8_control_result",
    "parse_h8_lossy_profiler_rows",
    "parse_h8_profiler_events",
]
