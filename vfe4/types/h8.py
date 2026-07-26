"""Immutable structural records for the bounded H8 sparse reference."""

from __future__ import annotations

import math
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Protocol, Sequence, runtime_checkable

import torch
from torch import Tensor

from vfe4.numerics.block_layout import (
    H8_MAX_STORAGE_SCALARS,
    BlockChainLayout,
    BlockId,
)
from vfe4.types.h7 import H7PredecessorReference
from vfe4.types.results import GateStatus


H8_MIN_CHOLESKY_PIVOT = 1e-8
H8_MAXIMUM_ALLOWANCE_SCALE_FRACTION = 1e-4
H8_MAX_SECONDS = 60.0
H8_MAX_PROCESS_INCREMENTAL_BYTES = 128 * 1024 * 1024
H8_MAX_TORCH_POPULATION_BYTES = 64 * 1024 * 1024
H8_H7_PLAN_SHA256 = (
    "3549153ac123b26f1d2372c59e80db93a78ed451fd4724781280dd7f413f1242"
)
H8_INTERPRETATION_SHA256 = (
    "e3fd048126c8133384e026826cf00bbea08280f4e248bc4cd5689e8f9f26e865"
)
H8_PROFILER_MEMORY_SOURCE_SHA256 = (
    "b80b4d5b58e91d581b18082c462ec7f088ec6b46ea50a1a62e2714d517a6a1b1"
)
H8_PROFILER_SOURCE_SHA256 = (
    "2c35f649219fb912728819b7dc0be5a5f1bd54c1efcd9502b62d976aeb278d22"
)
H8_PROFILER_API_CONTRACT_SHA256 = (
    "161a78f04c26fba19bb01ba6417f2cf8c00730ebeb8d007a4af0f4da433ba043"
)
H8_PROBLEM_DRAW_SCHEMA_SHA256 = (
    "7b657e72219f044147a7b414354d34c82bbd5a66d24f669285906d54534723c0"
)
H8_PRODUCTION_SEEDS = (20260721, 20260722, 20260723)
H8_PRODUCTION_SAMPLE_SEED_PAIRS = (
    (20260721, 20261721),
    (20260722, 20261722),
    (20260723, 20261723),
)
H8_CORRECTNESS_CASES = (
    (1, 1, 2026072111, 2026172111),
    (1, 2, 2026072112, 2026172112),
    (1, 4, 2026072114, 2026172114),
    (2, 1, 2026072121, 2026172121),
    (2, 2, 2026072122, 2026172122),
    (2, 4, 2026072124, 2026172124),
    (4, 1, 2026072141, 2026172141),
    (4, 2, 2026072142, 2026172142),
    (4, 4, 2026072144, 2026172144),
    (8, 1, 2026072181, 2026172181),
    (8, 2, 2026072182, 2026172182),
    (8, 4, 2026072184, 2026172184),
)
H8_CORRECTNESS_SOURCES = ("block", "dense_torch", "numpy")
H8_CORRECTNESS_ORDERED_SOURCE_PAIRS = (
    ("block", "dense_torch"),
    ("dense_torch", "block"),
    ("block", "numpy"),
    ("numpy", "block"),
    ("dense_torch", "numpy"),
    ("numpy", "dense_torch"),
)
H8_CORRECTNESS_CONTROL_IDS = (
    "perturbed_solve_element",
    "reversed_logdet_sign",
    "transposed_adjacent_covariance",
    "duplicated_offdiagonal_trace",
    "omitted_entropy",
    "independent_sample_noise",
)
H8_BASE_CORRECTNESS_ENDPOINT_IDS = (
    "factor_reconstruction",
    "forward_substitution",
    "backward_substitution",
    "solve",
    "logdet",
    "quadratic",
    "sample",
    "selected_diagonal",
    "selected_lower",
    "sparse_trace",
    "entropy",
    "log_normalizer",
)
H8_REQUIRED_OPERATIONS = (
    "factorization",
    "forward_substitution",
    "backward_substitution",
    "mean_solve",
    "logdet",
    "selected_inverse",
    "sample_width_one",
    "quadratic",
    "sparse_trace",
    "condition_estimate",
    "entropy",
    "log_normalizer",
    "complete_objective",
)
H8_NEGATIVE_CONTROL_IDS = (
    "torch_matrix_d_d",
    "torch_flat_d2",
    "torch_near_d2",
    "torch_length_d",
    "torch_block_pair_slab",
    "torch_triangular_pair_storage",
    "torch_pair_stack",
    "torch_eye_full_rhs",
    "torch_dense_eigvalsh",
    "numpy_matrix_d_d",
    "numpy_outer_d_d",
    "numpy_matmul_d_d",
)
H8_NONCLAIMS = (
    "no_language_result",
    "no_training_result",
    "no_prediction_result",
    "no_large_language_model_scale",
    "no_asymptotic_scaling_law",
    "no_gpu_claim",
    "no_exact_global_spectrum",
    "no_post_h8_training_memory_transfer",
)
H8_VERIFIER_PREFIX = (
    "H1",
    "H2",
    "H3",
    "H4",
    "H5",
    "H6-Prefix",
    "H7",
    "H8",
)
H8_H7_COMPATIBILITY_REFERENCE_KEYS = (
    "h1_h5",
    "h1_prefix_prior",
    "h6_prefix",
)


def _nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _positive_int(value: object, name: str) -> int:
    value = _nonnegative_int(value, name)
    if value == 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def h8_correctness_endpoint_ids(horizon: int) -> tuple[str, ...]:
    """Return the exact per-source endpoint order for one bounded H8 cell."""

    horizon = _positive_int(horizon, "horizon")
    receivers = tuple(range(1, horizon + 1))
    return (
        *H8_BASE_CORRECTNESS_ENDPOINT_IDS,
        "objective:initial_joint",
        *(f"objective:model_transition:{receiver:04d}" for receiver in receivers),
        *(f"objective:state_transition:{receiver:04d}" for receiver in receivers),
        *(f"objective:emission_order21:{receiver:04d}" for receiver in receivers),
        *(f"objective:emission_order17:{receiver:04d}" for receiver in receivers),
        "objective:recognition_entropy",
        "objective:log_normalizer",
        "objective:model_source_kl",
        "objective:state_source_kl",
        "objective:source_entropy",
        "objective:complete_order21",
    )


def _finite_nonnegative(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative float")
    return value


def _finite(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite float")
    return value


def _nonempty(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _git_object_id(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) not in (40, 64)
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a full lowercase Git object ID")
    return value


def _obligations(
    value: object,
    *,
    status: GateStatus,
    name: str = "obligations",
) -> tuple[str, ...]:
    if (
        type(value) is not tuple
        or any(type(item) is not str or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"{name} must contain unique nonempty strings")
    if status is GateStatus.INCONCLUSIVE:
        if not value:
            raise ValueError("INCONCLUSIVE requires an open obligation")
    elif value:
        raise ValueError("conclusive records cannot retain obligations")
    return value


def _status(value: object, name: str = "status") -> GateStatus:
    if not isinstance(value, GateStatus):
        raise ValueError(f"{name} must be a GateStatus")
    return value


def _shape(value: object, name: str) -> tuple[int, ...]:
    if type(value) is not tuple or not value:
        raise ValueError(f"{name} must be a nonempty shape tuple")
    for index, item in enumerate(value):
        _positive_int(item, f"{name}[{index}]")
    return value


def _freeze_digest_mapping(
    value: object,
    name: str,
    *,
    allow_empty: bool = False,
) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    copied = dict(value)
    if not copied and not allow_empty:
        raise ValueError(f"{name} must be nonempty")
    if len(copied) != len(value):
        raise ValueError(f"{name} cannot contain duplicate keys")
    for key, digest in copied.items():
        _nonempty(key, f"{name} key")
        _sha256(digest, f"{name}[{key!r}]")
    return MappingProxyType(copied)


def _canonical_json_bytes(value: object, name: str) -> bytes:
    if type(value) is not bytes:
        raise ValueError(f"{name} must be immutable bytes")
    try:
        parsed = json.loads(value.decode("utf-8"))
        canonical = json.dumps(
            parsed,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be canonical UTF-8 JSON") from exc
    if canonical != value:
        raise ValueError(f"{name} must be canonical UTF-8 JSON")
    return value


def _owned_block_tensor(
    value: object,
    *,
    expected_shape: tuple[int, ...],
    name: str,
) -> Tensor:
    if type(value) is not Tensor:
        raise ValueError(f"{name} must be a torch.Tensor")
    if value.dtype is not torch.float64:
        raise ValueError(f"{name} must use float64")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU")
    if tuple(value.shape) != expected_shape:
        raise ValueError(f"{name} has the wrong block shape")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    # Mathematical symmetry is adjudicated later with the frozen H8
    # operand-shaped residual budget.  This ownership boundary must preserve
    # the observed bytes, including roundoff-scale skew, for that decision.
    return value.detach().clone().contiguous()


@dataclass(frozen=True, slots=True)
class BlockPatternRecord:
    """The only precision and factor offsets allowed by H8."""

    precision_offsets: tuple[Literal[-1], Literal[0], Literal[1]] = (
        -1,
        0,
        1,
    )
    factor_storage_offsets: tuple[Literal[-1], Literal[0]] = (-1, 0)

    def __post_init__(self) -> None:
        if type(self.precision_offsets) is not tuple or self.precision_offsets != (
            -1,
            0,
            1,
        ):
            raise ValueError("precision offsets must be exactly (-1, 0, 1)")
        if (
            type(self.factor_storage_offsets) is not tuple
            or self.factor_storage_offsets != (-1, 0)
        ):
            raise ValueError("factor storage offsets must be exactly (-1, 0)")


@dataclass(frozen=True, slots=True)
class BlockStorageExpectation:
    """Layout-derived limits, kept separate from observed telemetry."""

    layout: BlockChainLayout
    precision_scalar_count: int
    factor_scalar_count: int
    selected_inverse_scalar_count: int
    information_scalar_count: int
    upper_block_scalar_count: Literal[0] = 0

    @classmethod
    def for_layout(cls, layout: BlockChainLayout) -> "BlockStorageExpectation":
        return cls(
            layout=layout,
            precision_scalar_count=layout.band_storage_scalar_count,
            factor_scalar_count=layout.band_storage_scalar_count,
            selected_inverse_scalar_count=layout.band_storage_scalar_count,
            information_scalar_count=layout.information_scalar_count,
        )

    def __post_init__(self) -> None:
        if type(self.layout) is not BlockChainLayout:
            raise ValueError("layout must be a BlockChainLayout")
        expected_band = self.layout.band_storage_scalar_count
        for name in (
            "precision_scalar_count",
            "factor_scalar_count",
            "selected_inverse_scalar_count",
        ):
            value = _positive_int(getattr(self, name), name)
            if value != expected_band or value > H8_MAX_STORAGE_SCALARS:
                raise ValueError(
                    f"{name} must equal the bounded block-band count"
                )
        if (
            _positive_int(
                self.information_scalar_count,
                "information_scalar_count",
            )
            != self.layout.information_scalar_count
        ):
            raise ValueError(
                "information_scalar_count must equal the block-vector count"
            )
        if type(self.upper_block_scalar_count) is not int or self.upper_block_scalar_count != 0:
            raise ValueError("upper blocks must never own storage")


@dataclass(frozen=True, slots=True)
class BlockStorageRecord:
    """Raw observed storage counts, including values that fail H8 limits."""

    layout: BlockChainLayout
    precision_scalar_count: int
    factor_scalar_count: int
    selected_inverse_scalar_count: int
    information_scalar_count: int
    upper_block_scalar_count: int

    def __post_init__(self) -> None:
        if type(self.layout) is not BlockChainLayout:
            raise ValueError("layout must be a BlockChainLayout")
        for name in (
            "precision_scalar_count",
            "factor_scalar_count",
            "selected_inverse_scalar_count",
            "information_scalar_count",
            "upper_block_scalar_count",
        ):
            _nonnegative_int(getattr(self, name), name)

    @property
    def matches_expectation(self) -> bool:
        expected = BlockStorageExpectation.for_layout(self.layout)
        return (
            self.precision_scalar_count == expected.precision_scalar_count
            and self.factor_scalar_count == expected.factor_scalar_count
            and self.selected_inverse_scalar_count
            == expected.selected_inverse_scalar_count
            and self.information_scalar_count
            == expected.information_scalar_count
            and self.upper_block_scalar_count == 0
        )

    @property
    def over_cap_categories(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, count in (
                ("precision", self.precision_scalar_count),
                ("factor", self.factor_scalar_count),
                ("selected_inverse", self.selected_inverse_scalar_count),
            )
            if count > H8_MAX_STORAGE_SCALARS
        )


@dataclass(frozen=True, slots=True)
class BlockFillRecord:
    layout: BlockChainLayout
    stored_block_ids: tuple[BlockId, ...]
    observed_offband_blocks: int
    duplicated_upper_blocks: int

    def __post_init__(self) -> None:
        if type(self.layout) is not BlockChainLayout:
            raise ValueError("layout must be a BlockChainLayout")
        if type(self.stored_block_ids) is not tuple or any(
            type(block) is not BlockId for block in self.stored_block_ids
        ):
            raise ValueError("stored_block_ids must be a tuple of BlockId")
        for block in self.stored_block_ids:
            self.layout.require_block_id(block)
        _nonnegative_int(
            self.observed_offband_blocks,
            "observed_offband_blocks",
        )
        _nonnegative_int(
            self.duplicated_upper_blocks,
            "duplicated_upper_blocks",
        )

    @property
    def matches_expected_fill(self) -> bool:
        return (
            self.stored_block_ids == self.layout.stored_block_ids
            and self.observed_offband_blocks == 0
            and self.duplicated_upper_blocks == 0
        )


@dataclass(frozen=True, slots=True)
class BlockWorkspaceRecord:
    maximum_shape: tuple[int, ...]
    maximum_scalar_count: int
    maximum_rhs_width: int
    attempted_forbidden_rhs_widths: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if type(self.maximum_shape) is not tuple or not self.maximum_shape:
            raise ValueError("maximum_shape must be a nonempty tuple")
        dimensions = tuple(
            _positive_int(item, "maximum_shape item")
            for item in self.maximum_shape
        )
        if math.prod(dimensions) != _positive_int(
            self.maximum_scalar_count,
            "maximum_scalar_count",
        ):
            raise ValueError("maximum_scalar_count must match maximum_shape")
        _positive_int(self.maximum_rhs_width, "maximum_rhs_width")
        if type(self.attempted_forbidden_rhs_widths) is not tuple or any(
            type(item) is not int or item <= 0
            for item in self.attempted_forbidden_rhs_widths
        ):
            raise ValueError(
                "attempted_forbidden_rhs_widths must be positive integers"
            )


@dataclass(frozen=True, slots=True)
class SparseConditionDiagnostics:
    estimator: Literal["HagerHigham1NormEstimate-v1"]
    kappa_1_estimate: float
    iterations: int
    convergence_reason: str
    index_sha256: str
    sign_sha256: str
    per_block_min_pivots: tuple[float, ...]
    global_min_pivot: float
    per_block_pivot_margins: tuple[float, ...]
    global_pivot_margin: float

    def __post_init__(self) -> None:
        if self.estimator != "HagerHigham1NormEstimate-v1":
            raise ValueError("unsupported sparse condition estimator")
        _finite_nonnegative(self.kappa_1_estimate, "kappa_1_estimate")
        if not 1 <= _positive_int(self.iterations, "iterations") <= 8:
            raise ValueError("iterations must be between one and eight")
        if type(self.convergence_reason) is not str or not self.convergence_reason:
            raise ValueError("convergence_reason must be nonempty")
        _sha256(self.index_sha256, "index_sha256")
        _sha256(self.sign_sha256, "sign_sha256")
        if type(self.per_block_min_pivots) is not tuple or not self.per_block_min_pivots:
            raise ValueError("per_block_min_pivots must be nonempty")
        pivots = tuple(
            _finite_nonnegative(value, "per_block_min_pivot")
            for value in self.per_block_min_pivots
        )
        global_pivot = _finite_nonnegative(
            self.global_min_pivot,
            "global_min_pivot",
        )
        if global_pivot != min(pivots):
            raise ValueError("global_min_pivot must equal the block minimum")
        if type(self.per_block_pivot_margins) is not tuple or len(
            self.per_block_pivot_margins
        ) != len(pivots):
            raise ValueError("pivot margins must align with block pivots")
        expected_margins = tuple(
            pivot - H8_MIN_CHOLESKY_PIVOT for pivot in pivots
        )
        if any(
            type(actual) is not float
            or not math.isfinite(actual)
            or actual != expected
            for actual, expected in zip(
                self.per_block_pivot_margins,
                expected_margins,
                strict=True,
            )
        ):
            raise ValueError("per-block pivot margins are inconsistent")
        if (
            type(self.global_pivot_margin) is not float
            or not math.isfinite(self.global_pivot_margin)
            or self.global_pivot_margin
            != global_pivot - H8_MIN_CHOLESKY_PIVOT
        ):
            raise ValueError("global pivot margin is inconsistent")


H8OperandSource = Literal["block", "dense_torch", "numpy"]


@dataclass(frozen=True, slots=True)
class H8OperandRecord:
    """One observed operand and the local facts used by its H8 budget."""

    operand_id: str
    shape: tuple[int, ...]
    scalar_count: int
    infinity_norm: float
    absolute_sum_bound: float
    local_operation_count: int
    source: H8OperandSource
    condition_provenance: str | None
    solver_produced: bool
    quadrature_convergence: float = 0.0

    def __post_init__(self) -> None:
        _nonempty(self.operand_id, "operand_id")
        dimensions = _shape(self.shape, "shape")
        if math.prod(dimensions) != _positive_int(
            self.scalar_count,
            "scalar_count",
        ):
            raise ValueError("scalar_count must equal the operand shape product")
        _finite_nonnegative(self.infinity_norm, "infinity_norm")
        absolute_sum = _finite_nonnegative(
            self.absolute_sum_bound,
            "absolute_sum_bound",
        )
        if absolute_sum < self.infinity_norm:
            raise ValueError("absolute_sum_bound cannot be below infinity_norm")
        _positive_int(self.local_operation_count, "local_operation_count")
        if self.source not in ("block", "dense_torch", "numpy"):
            raise ValueError("source must be block, dense_torch, or numpy")
        if self.condition_provenance is not None:
            _nonempty(self.condition_provenance, "condition_provenance")
        if type(self.solver_produced) is not bool:
            raise ValueError("solver_produced must be a bool")
        _finite_nonnegative(
            self.quadrature_convergence,
            "quadrature_convergence",
        )


# The plan's public budget signature intentionally uses this short name inside
# vfe4.types.h8.  The package-level export retains the H8-qualified spelling.
OperandRecord = H8OperandRecord


@dataclass(frozen=True, slots=True)
class H8CorrectnessEndpointRecord:
    """One immutable raw endpoint and its exact operand-budget witness."""

    endpoint_id: str
    raw_values: tuple[float, ...]
    operand: H8OperandRecord

    def __post_init__(self) -> None:
        _nonempty(self.endpoint_id, "endpoint_id")
        if (
            type(self.raw_values) is not tuple
            or not self.raw_values
            or any(
                type(value) is not float or not math.isfinite(value)
                for value in self.raw_values
            )
        ):
            raise ValueError("raw_values must be a nonempty tuple of finite floats")
        if type(self.operand) is not H8OperandRecord:
            raise ValueError("operand must be an H8OperandRecord")
        if self.operand.scalar_count != len(self.raw_values):
            raise ValueError("raw endpoint count must match its operand")
        if self.operand.operand_id != (
            f"{self.operand.source}:{self.endpoint_id}"
        ):
            raise ValueError("operand_id must bind source and endpoint identities")
        infinity_norm = max(abs(value) for value in self.raw_values)
        absolute_sum = math.fsum(abs(value) for value in self.raw_values)
        if self.operand.infinity_norm != infinity_norm:
            raise ValueError("operand infinity norm must match the raw endpoint")
        if self.operand.absolute_sum_bound != absolute_sum:
            raise ValueError("operand absolute-sum bound must match the raw endpoint")


@dataclass(frozen=True, slots=True)
class H8CorrectnessSourceResult:
    """The complete ordered raw endpoint inventory from one independent source."""

    source: H8OperandSource
    endpoints: tuple[H8CorrectnessEndpointRecord, ...]

    def __post_init__(self) -> None:
        if self.source not in H8_CORRECTNESS_SOURCES:
            raise ValueError("source is outside the exact H8 correctness inventory")
        if (
            type(self.endpoints) is not tuple
            or not self.endpoints
            or any(
                type(endpoint) is not H8CorrectnessEndpointRecord
                for endpoint in self.endpoints
            )
        ):
            raise ValueError("endpoints must be nonempty typed records")
        endpoint_ids = tuple(endpoint.endpoint_id for endpoint in self.endpoints)
        if len(set(endpoint_ids)) != len(endpoint_ids):
            raise ValueError("source endpoint IDs must be unique")
        if any(endpoint.operand.source != self.source for endpoint in self.endpoints):
            raise ValueError("every endpoint operand must bind the source record")


@dataclass(frozen=True, slots=True)
class H8AllowanceRecord:
    """Literal residual decision for one named ordered operand pair."""

    comparison_id: str
    left: H8OperandRecord
    right: H8OperandRecord
    compared_scalar_count: int
    left_rounding_component: float
    left_solver_component: float
    left_quadrature_component: float
    right_rounding_component: float
    right_solver_component: float
    right_quadrature_component: float
    reduction_component: float
    allowance: float
    scale: float
    residual: float
    allowance_scale_fraction: float
    decisive: bool
    status: GateStatus
    obligations: tuple[str, ...]

    def __post_init__(self) -> None:
        _nonempty(self.comparison_id, "comparison_id")
        if type(self.left) is not H8OperandRecord or type(self.right) is not H8OperandRecord:
            raise ValueError("allowance operands must be H8OperandRecord")
        _positive_int(self.compared_scalar_count, "compared_scalar_count")
        components = (
            self.left_rounding_component,
            self.left_solver_component,
            self.left_quadrature_component,
            self.right_rounding_component,
            self.right_solver_component,
            self.right_quadrature_component,
            self.reduction_component,
        )
        for index, component in enumerate(components):
            _finite_nonnegative(component, f"component[{index}]")
        allowance = _finite_nonnegative(self.allowance, "allowance")
        expected_allowance = math.fsum(components)
        if allowance != expected_allowance:
            raise ValueError("allowance must equal the seven named components")
        scale = _finite(self.scale, "scale")
        if scale < 1.0:
            raise ValueError("allowance scale must be at least one")
        residual = _finite_nonnegative(self.residual, "residual")
        fraction = _finite_nonnegative(
            self.allowance_scale_fraction,
            "allowance_scale_fraction",
        )
        if fraction != allowance / scale:
            raise ValueError("allowance_scale_fraction must equal allowance/scale")
        if type(self.decisive) is not bool:
            raise ValueError("decisive must be a bool")
        expected_decisive = fraction < H8_MAXIMUM_ALLOWANCE_SCALE_FRACTION
        if self.decisive != expected_decisive:
            raise ValueError("decisiveness must use the frozen strict boundary")
        status = _status(self.status)
        expected_status = (
            GateStatus.INCONCLUSIVE
            if not expected_decisive
            else GateStatus.PASS
            if residual <= allowance
            else GateStatus.FAIL
        )
        if status is not expected_status:
            raise ValueError("allowance status does not match its literal decision")
        _obligations(self.obligations, status=status)


AllowanceRecord = H8AllowanceRecord


H8ObjectiveRole = Literal[
    "initial_joint",
    "model_transition",
    "state_transition",
    "emission_order21",
    "emission_order17",
]


@dataclass(frozen=True, slots=True)
class H8ObjectiveTerm:
    factor_id: str
    role: H8ObjectiveRole
    receiver_t: int | None
    value: float
    absolute_sum_bound: float

    def __post_init__(self) -> None:
        _nonempty(self.factor_id, "factor_id")
        if self.role not in (
            "initial_joint",
            "model_transition",
            "state_transition",
            "emission_order21",
            "emission_order17",
        ):
            raise ValueError("objective role is outside the frozen inventory")
        if self.role == "initial_joint":
            if self.receiver_t is not None or self.factor_id != "initial_joint":
                raise ValueError("the initial objective term has one fixed identity")
        else:
            receiver = _positive_int(self.receiver_t, "receiver_t")
            expected_prefix = {
                "model_transition": "model_transition",
                "state_transition": "state_transition",
                "emission_order21": "emission_order21",
                "emission_order17": "emission_order17",
            }[self.role]
            if self.factor_id != f"{expected_prefix}:{receiver:04d}":
                raise ValueError("objective factor_id does not match its role/time")
        value = _finite(self.value, "value")
        absolute_sum = _finite_nonnegative(
            self.absolute_sum_bound,
            "absolute_sum_bound",
        )
        if absolute_sum < abs(value):
            raise ValueError("term absolute_sum_bound cannot be below abs(value)")


def _require_objective_series(
    value: object,
    *,
    role: H8ObjectiveRole,
    horizon: int,
    name: str,
) -> tuple[H8ObjectiveTerm, ...]:
    if (
        type(value) is not tuple
        or len(value) != horizon
        or any(type(item) is not H8ObjectiveTerm for item in value)
    ):
        raise ValueError(f"{name} must contain exactly horizon typed terms")
    expected_receivers = tuple(range(1, horizon + 1))
    if (
        tuple(item.receiver_t for item in value) != expected_receivers
        or any(item.role != role for item in value)
    ):
        raise ValueError(f"{name} must be ordered by receiver_t=1..T")
    return value


@dataclass(frozen=True, slots=True)
class H8ObjectiveTerms:
    """The complete normalized synthetic objective with explicit zero sources."""

    horizon: int
    initial_joint: H8ObjectiveTerm
    model_transitions: tuple[H8ObjectiveTerm, ...]
    state_transitions: tuple[H8ObjectiveTerm, ...]
    emissions_order21: tuple[H8ObjectiveTerm, ...]
    emissions_order17: tuple[H8ObjectiveTerm, ...]
    recognition_entropy: float
    log_normalizer: float
    model_source_kl: Literal[0.0]
    state_source_kl: Literal[0.0]
    source_entropy: Literal[0.0]
    quadrature_absolute_difference: float
    complete_order21: float
    absolute_term_sum: float

    def __post_init__(self) -> None:
        horizon = _positive_int(self.horizon, "horizon")
        if (
            type(self.initial_joint) is not H8ObjectiveTerm
            or self.initial_joint.role != "initial_joint"
        ):
            raise ValueError("initial_joint must be the fixed initial objective term")
        model = _require_objective_series(
            self.model_transitions,
            role="model_transition",
            horizon=horizon,
            name="model_transitions",
        )
        state = _require_objective_series(
            self.state_transitions,
            role="state_transition",
            horizon=horizon,
            name="state_transitions",
        )
        emission21 = _require_objective_series(
            self.emissions_order21,
            role="emission_order21",
            horizon=horizon,
            name="emissions_order21",
        )
        emission17 = _require_objective_series(
            self.emissions_order17,
            role="emission_order17",
            horizon=horizon,
            name="emissions_order17",
        )
        factor_ids = tuple(
            term.factor_id
            for term in (
                self.initial_joint,
                *model,
                *state,
                *emission21,
                *emission17,
            )
        )
        if len(set(factor_ids)) != len(factor_ids):
            raise ValueError("objective factor IDs must be unique")
        entropy = _finite(self.recognition_entropy, "recognition_entropy")
        _finite(self.log_normalizer, "log_normalizer")
        if (
            type(self.model_source_kl) is not float
            or self.model_source_kl != 0.0
            or type(self.state_source_kl) is not float
            or self.state_source_kl != 0.0
            or type(self.source_entropy) is not float
            or self.source_entropy != 0.0
        ):
            raise ValueError("singleton source terms must be explicit float zeros")
        quadrature_difference = _finite_nonnegative(
            self.quadrature_absolute_difference,
            "quadrature_absolute_difference",
        )
        expected_difference = math.fsum(
            abs(left.value - right.value)
            for left, right in zip(emission21, emission17, strict=True)
        )
        if quadrature_difference != expected_difference:
            raise ValueError("quadrature difference must bind all emission pairs once")
        order21_values = (
            self.initial_joint.value,
            *(item.value for item in model),
            *(item.value for item in state),
            *(item.value for item in emission21),
            entropy,
        )
        if _finite(self.complete_order21, "complete_order21") != math.fsum(
            order21_values
        ):
            raise ValueError("complete_order21 must contain every normalized term")
        expected_absolute_sum = math.fsum(abs(value) for value in order21_values)
        if (
            _finite_nonnegative(self.absolute_term_sum, "absolute_term_sum")
            != expected_absolute_sum
        ):
            raise ValueError("absolute_term_sum must bind the complete objective")


@dataclass(frozen=True, slots=True)
class H8InvariantRecord:
    invariant_id: str
    status: GateStatus
    value: float | int | None
    limit: float | int | None
    detail: str
    obligations: tuple[str, ...]

    def __post_init__(self) -> None:
        _nonempty(self.invariant_id, "invariant_id")
        status = _status(self.status)
        for name, value in (("value", self.value), ("limit", self.limit)):
            if value is not None and (
                type(value) not in (int, float) or not math.isfinite(float(value))
            ):
                raise ValueError(f"{name} must be a finite scalar or None")
        _nonempty(self.detail, "detail")
        _obligations(self.obligations, status=status)
        if status is GateStatus.FAIL and (
            self.value is None or self.limit is None
        ):
            raise ValueError("a failed invariant requires witnessed value and limit")


@dataclass(frozen=True, slots=True)
class H8CorrectnessControlResult:
    control_id: str
    residual: float
    allowance: float
    decisive: bool
    status: GateStatus
    obligations: tuple[str, ...]

    def __post_init__(self) -> None:
        _nonempty(self.control_id, "control_id")
        residual = _finite_nonnegative(self.residual, "residual")
        allowance = _finite_nonnegative(self.allowance, "allowance")
        if type(self.decisive) is not bool:
            raise ValueError("decisive must be a bool")
        status = _status(self.status)
        expected_status = (
            GateStatus.INCONCLUSIVE
            if not self.decisive
            else GateStatus.PASS
            if residual > allowance
            else GateStatus.FAIL
        )
        if status is not expected_status:
            raise ValueError("wrong-path control status does not match its witness")
        _obligations(self.obligations, status=status)


@dataclass(frozen=True, slots=True)
class H8CorrectnessCell:
    cell_id: int
    layout: BlockChainLayout
    problem_seed: int
    sample_noise_seed: int
    problem_sha256: str
    sample_noise_sha256: str
    source_results: tuple[H8CorrectnessSourceResult, ...]
    pair_comparisons: tuple[H8AllowanceRecord, ...]
    wrong_path_controls: tuple[H8CorrectnessControlResult, ...]
    invariants: tuple[H8InvariantRecord, ...]
    status: GateStatus
    obligations: tuple[str, ...]

    def __post_init__(self) -> None:
        cell_id = _positive_int(self.cell_id, "cell_id")
        if not 1 <= cell_id <= len(H8_CORRECTNESS_CASES):
            raise ValueError("cell_id is outside the frozen correctness grid")
        if type(self.layout) is not BlockChainLayout:
            raise ValueError("layout must be a BlockChainLayout")
        expected = H8_CORRECTNESS_CASES[cell_id - 1]
        expected_horizon, expected_k, expected_problem, expected_noise = expected
        if (
            self.layout.horizon,
            self.layout.d_z,
            self.layout.d_m,
            self.problem_seed,
            self.sample_noise_seed,
        ) != (
            expected_horizon,
            expected_k,
            expected_k,
            expected_problem,
            expected_noise,
        ):
            raise ValueError("correctness cell does not match its literal seed row")
        _sha256(self.problem_sha256, "problem_sha256")
        _sha256(self.sample_noise_sha256, "sample_noise_sha256")
        if (
            type(self.source_results) is not tuple
            or any(
                type(result) is not H8CorrectnessSourceResult
                for result in self.source_results
            )
            or tuple(result.source for result in self.source_results)
            != H8_CORRECTNESS_SOURCES
        ):
            raise ValueError(
                "source_results must contain block, dense_torch, and numpy in order"
            )
        expected_endpoints = h8_correctness_endpoint_ids(self.layout.horizon)
        endpoint_operands: dict[tuple[str, str], H8OperandRecord] = {}
        scalar_counts: dict[str, int] = {}
        for source_result in self.source_results:
            observed_endpoints = tuple(
                endpoint.endpoint_id for endpoint in source_result.endpoints
            )
            if observed_endpoints != expected_endpoints:
                raise ValueError(
                    f"{source_result.source} endpoint inventory is incomplete or reordered"
                )
            for endpoint in source_result.endpoints:
                endpoint_operands[(source_result.source, endpoint.endpoint_id)] = (
                    endpoint.operand
                )
                prior_count = scalar_counts.setdefault(
                    endpoint.endpoint_id,
                    endpoint.operand.scalar_count,
                )
                if prior_count != endpoint.operand.scalar_count:
                    raise ValueError(
                        "same-named source endpoints must have equal scalar counts"
                    )
        if (
            type(self.pair_comparisons) is not tuple
            or not self.pair_comparisons
            or any(
                type(item) is not H8AllowanceRecord
                for item in self.pair_comparisons
            )
        ):
            raise ValueError("pair_comparisons must be nonempty typed records")
        comparison_ids = tuple(item.comparison_id for item in self.pair_comparisons)
        expected_comparisons = tuple(
            f"{endpoint}:{left}->{right}"
            for endpoint in expected_endpoints
            for left, right in H8_CORRECTNESS_ORDERED_SOURCE_PAIRS
        )
        if comparison_ids != expected_comparisons:
            raise ValueError(
                "pair comparisons must contain every endpoint and six directions"
            )
        for comparison, expected_id in zip(
            self.pair_comparisons,
            expected_comparisons,
            strict=True,
        ):
            endpoint, direction = expected_id.rsplit(":", maxsplit=1)
            left_source, right_source = direction.split("->")
            if (
                comparison.left
                != endpoint_operands[(left_source, endpoint)]
                or comparison.right
                != endpoint_operands[(right_source, endpoint)]
            ):
                raise ValueError(
                    "pair comparison operands must be the retained source endpoints"
                )
        if (
            type(self.wrong_path_controls) is not tuple
            or not self.wrong_path_controls
            or any(
                type(item) is not H8CorrectnessControlResult
                for item in self.wrong_path_controls
            )
        ):
            raise ValueError("wrong_path_controls must be nonempty typed records")
        control_ids = tuple(item.control_id for item in self.wrong_path_controls)
        if control_ids != H8_CORRECTNESS_CONTROL_IDS:
            raise ValueError(
                "wrong-path controls must match the exact six-control inventory"
            )
        if (
            type(self.invariants) is not tuple
            or not self.invariants
            or any(type(item) is not H8InvariantRecord for item in self.invariants)
        ):
            raise ValueError("invariants must be nonempty typed records")
        invariant_ids = tuple(item.invariant_id for item in self.invariants)
        if len(set(invariant_ids)) != len(invariant_ids):
            raise ValueError("correctness invariant IDs must be unique")
        child_statuses = (
            *(item.status for item in self.pair_comparisons),
            *(item.status for item in self.wrong_path_controls),
            *(item.status for item in self.invariants),
        )
        expected_status = (
            GateStatus.FAIL
            if GateStatus.FAIL in child_statuses
            else GateStatus.INCONCLUSIVE
            if GateStatus.INCONCLUSIVE in child_statuses
            else GateStatus.PASS
        )
        status = _status(self.status)
        if status is not expected_status:
            raise ValueError("cell status must summarize every child decision")
        _obligations(self.obligations, status=status)


@dataclass(frozen=True, slots=True)
class BackendCounterSnapshot:
    layout: BlockChainLayout
    factorization_calls: int
    forward_substitution_calls: int
    backward_substitution_calls: int
    solve_calls: int
    logdet_calls: int
    selected_inverse_calls: int
    sample_calls: int
    quadratic_calls: int
    trace_calls: int
    sparse_matvec_calls: int
    maximum_rhs_width: int
    maximum_sample_rhs_width: int
    selected_block_ids: tuple[BlockId, ...]
    selected_block_count: int
    attempted_forbidden_selected_blocks: int
    attempted_forbidden_rhs_widths: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if type(self.layout) is not BlockChainLayout:
            raise ValueError("layout must be a BlockChainLayout")
        for name in (
            "factorization_calls",
            "forward_substitution_calls",
            "backward_substitution_calls",
            "solve_calls",
            "logdet_calls",
            "selected_inverse_calls",
            "sample_calls",
            "quadratic_calls",
            "trace_calls",
            "sparse_matvec_calls",
            "maximum_rhs_width",
            "maximum_sample_rhs_width",
            "selected_block_count",
            "attempted_forbidden_selected_blocks",
        ):
            _nonnegative_int(getattr(self, name), name)
        if type(self.selected_block_ids) is not tuple or any(
            type(block) is not BlockId for block in self.selected_block_ids
        ):
            raise ValueError("selected_block_ids must be a tuple of BlockId")
        for block in self.selected_block_ids:
            self.layout.require_block_id(block)
        if self.selected_block_count != len(self.selected_block_ids):
            raise ValueError(
                "selected_block_count must preserve the observed inventory"
            )
        if type(self.attempted_forbidden_rhs_widths) is not tuple or any(
            type(item) is not int or item <= 0
            for item in self.attempted_forbidden_rhs_widths
        ):
            raise ValueError(
                "attempted_forbidden_rhs_widths must be positive integers"
            )

    @property
    def selected_coverage_complete(self) -> bool:
        return self.selected_block_ids == self.layout.stored_block_ids


H8ProfilerAction = Literal[
    "PREEXISTING",
    "CREATE",
    "INCREMENT_VERSION",
    "DESTROY",
]
H8AllocationChannel = Literal[
    "dispatch",
    "profiler",
    "backend",
    "numpy_guard",
    "os_hwm",
]


@dataclass(frozen=True, slots=True)
class H8TensorKey:
    tensor_id: int
    storage_ptr: int
    allocation_id: int
    device: str

    def __post_init__(self) -> None:
        _nonnegative_int(self.tensor_id, "tensor_id")
        _nonnegative_int(self.storage_ptr, "storage_ptr")
        _nonnegative_int(self.allocation_id, "allocation_id")
        _nonempty(self.device, "device")


@dataclass(frozen=True, slots=True)
class H8ProfilerEventRecord:
    """One lossless, joined profiler timeline action."""

    source_row_index: int
    timestamp_ns: int
    action: H8ProfilerAction
    tensor_key: H8TensorKey
    version: int
    nbytes: int
    dtype: str
    device: str
    operator: str
    stack: tuple[str, ...]
    logical_shape: tuple[int, ...]
    classification: str
    matched_event_node_indices: tuple[int, ...]
    join_witness_sha256: str
    live_bytes_after: int

    def __post_init__(self) -> None:
        _nonnegative_int(self.source_row_index, "source_row_index")
        if self.action not in (
            "PREEXISTING",
            "CREATE",
            "INCREMENT_VERSION",
            "DESTROY",
        ):
            raise ValueError("profiler action is outside the frozen union")
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
        if self.action in ("PREEXISTING", "CREATE") and self.nbytes <= 0:
            raise ValueError("live-establishing actions require positive nbytes")
        if self.action == "INCREMENT_VERSION" and self.nbytes != 0:
            raise ValueError("INCREMENT_VERSION has zero live-byte delta")
        if self.action == "DESTROY" and self.nbytes >= 0:
            raise ValueError("DESTROY requires a negative signed-byte witness")
        for name, value in (
            ("dtype", self.dtype),
            ("device", self.device),
            ("operator", self.operator),
            ("classification", self.classification),
        ):
            _nonempty(value, name)
        if (
            type(self.stack) is not tuple
            or not self.stack
            or any(type(item) is not str or not item for item in self.stack)
        ):
            raise ValueError("stack must contain nonempty source frames")
        if type(self.logical_shape) is not tuple or any(
            type(item) is not int or item < 0 for item in self.logical_shape
        ):
            raise ValueError("logical_shape must be a tuple of dimensions")
        if (
            type(self.matched_event_node_indices) is not tuple
            or not self.matched_event_node_indices
            or any(
                type(item) is not int or item < 0
                for item in self.matched_event_node_indices
            )
            or len(set(self.matched_event_node_indices))
            != len(self.matched_event_node_indices)
        ):
            raise ValueError("matched event-node indices must be unique")
        _sha256(self.join_witness_sha256, "join_witness_sha256")
        _nonnegative_int(self.live_bytes_after, "live_bytes_after")


@dataclass(frozen=True, slots=True)
class H8LossyProfilerRow:
    """Supplementary documented raw-export row; never closure evidence."""

    timestamp_ns: int
    action: H8ProfilerAction
    nbytes: int
    category: str

    def __post_init__(self) -> None:
        if self.action not in (
            "PREEXISTING",
            "CREATE",
            "INCREMENT_VERSION",
            "DESTROY",
        ):
            raise ValueError("profiler action is outside the frozen union")
        if (
            type(self.timestamp_ns) is not int
            or self.timestamp_ns < -1
            or (self.timestamp_ns == -1 and self.action != "PREEXISTING")
        ):
            raise ValueError(
                "timestamp_ns must be nonnegative except -1 PREEXISTING"
            )
        if type(self.nbytes) is not int or type(self.nbytes) is bool:
            raise ValueError("nbytes must be an integer")
        _nonempty(self.category, "category")


@dataclass(frozen=True, slots=True)
class H8AllocationRecord:
    dispatch_trace_sha256: str | None
    dispatch_event_count: int
    dispatch_forbidden_attempt_count: int
    dispatch_live_peak_bytes: int
    torch_population_peak_bytes: int
    profiler_trace_sha256: str | None
    profiler_events: tuple[H8ProfilerEventRecord, ...]
    profiler_lossy_rows: tuple[H8LossyProfilerRow, ...]
    preexisting_storage_count: int | None
    preexisting_bytes: int | None
    baseline_live_bytes: int | None
    profiler_reconstructed_live_peak_bytes: int | None
    profiler_all_joined_and_liveness_reconciled: bool | None
    numpy_guard_event_count: int
    backend_forbidden_attempt_count: int
    observed_channels: tuple[H8AllocationChannel, ...]

    def __post_init__(self) -> None:
        if self.dispatch_trace_sha256 is not None:
            _sha256(self.dispatch_trace_sha256, "dispatch_trace_sha256")
        for name in (
            "dispatch_event_count",
            "dispatch_forbidden_attempt_count",
            "dispatch_live_peak_bytes",
            "torch_population_peak_bytes",
            "numpy_guard_event_count",
            "backend_forbidden_attempt_count",
        ):
            _nonnegative_int(getattr(self, name), name)
        if (
            type(self.profiler_events) is not tuple
            or any(
                type(item) is not H8ProfilerEventRecord
                for item in self.profiler_events
            )
        ):
            raise ValueError("profiler_events must contain typed raw records")
        if (
            type(self.profiler_lossy_rows) is not tuple
            or any(
                type(item) is not H8LossyProfilerRow
                for item in self.profiler_lossy_rows
            )
        ):
            raise ValueError("profiler_lossy_rows must contain typed rows")
        profiler_fields = (
            self.preexisting_storage_count,
            self.preexisting_bytes,
            self.baseline_live_bytes,
            self.profiler_reconstructed_live_peak_bytes,
            self.profiler_all_joined_and_liveness_reconciled,
        )
        profiler_present = self.profiler_trace_sha256 is not None
        if profiler_present:
            _sha256(self.profiler_trace_sha256, "profiler_trace_sha256")
            if not self.profiler_events or any(
                item is None for item in profiler_fields
            ):
                raise ValueError("profiler evidence requires every raw endpoint")
            for name in (
                "preexisting_storage_count",
                "preexisting_bytes",
                "baseline_live_bytes",
                "profiler_reconstructed_live_peak_bytes",
            ):
                _nonnegative_int(getattr(self, name), name)
            if type(self.profiler_all_joined_and_liveness_reconciled) is not bool:
                raise ValueError("profiler reconciliation must be observed")
        elif (
            self.profiler_events
            or self.profiler_lossy_rows
            or any(item is not None for item in profiler_fields)
        ):
            raise ValueError("profiler endpoints cannot exist without its trace hash")
        if (
            type(self.observed_channels) is not tuple
            or not self.observed_channels
            or len(set(self.observed_channels)) != len(self.observed_channels)
            or any(
                item
                not in (
                    "dispatch",
                    "profiler",
                    "backend",
                    "numpy_guard",
                    "os_hwm",
                )
                for item in self.observed_channels
            )
        ):
            raise ValueError("observed_channels must be a unique closed inventory")
        if profiler_present != ("profiler" in self.observed_channels):
            raise ValueError("profiler channel presence must match its endpoints")


@dataclass(frozen=True, slots=True)
class H8ResourceRecord:
    adapter: str
    adapter_sha256: str
    pre_current_rss_bytes: int
    pre_lifetime_peak_bytes: int
    pre_private_bytes: int
    post_current_rss_bytes: int
    post_lifetime_peak_bytes: int
    post_private_bytes: int
    conservative_incremental_hwm_bytes: int
    peak_to_peak_diagnostic_bytes: int
    parent_elapsed_ns: int
    child_elapsed_ns: int

    def __post_init__(self) -> None:
        _nonempty(self.adapter, "adapter")
        _sha256(self.adapter_sha256, "adapter_sha256")
        for name in (
            "pre_current_rss_bytes",
            "pre_lifetime_peak_bytes",
            "pre_private_bytes",
            "post_current_rss_bytes",
            "post_lifetime_peak_bytes",
            "post_private_bytes",
            "conservative_incremental_hwm_bytes",
            "peak_to_peak_diagnostic_bytes",
            "parent_elapsed_ns",
            "child_elapsed_ns",
        ):
            _nonnegative_int(getattr(self, name), name)
        expected_conservative = max(
            0,
            self.post_lifetime_peak_bytes - self.pre_current_rss_bytes,
        )
        if self.conservative_incremental_hwm_bytes != expected_conservative:
            raise ValueError("conservative HWM must use pre-run current RSS")
        expected_peak_delta = max(
            0,
            self.post_lifetime_peak_bytes - self.pre_lifetime_peak_bytes,
        )
        if self.peak_to_peak_diagnostic_bytes != expected_peak_delta:
            raise ValueError("peak-to-peak diagnostic is inconsistent")


@dataclass(frozen=True, slots=True)
class H8ControlResult:
    control_id: str
    requested_operation: str
    logical_shapes: tuple[tuple[int, ...], ...]
    assigned_channels: tuple[H8AllocationChannel, ...]
    observed_channels: tuple[H8AllocationChannel, ...]
    execution_witnessed: bool
    event_sha256: str | None
    assignment_complete: bool
    detected: bool
    status: GateStatus
    obligations: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.control_id not in H8_NEGATIVE_CONTROL_IDS:
            raise ValueError("control_id is outside the frozen ordered inventory")
        _nonempty(self.requested_operation, "requested_operation")
        if (
            type(self.logical_shapes) is not tuple
            or not self.logical_shapes
            or any(
                type(shape) is not tuple
                or any(type(item) is not int or item <= 0 for item in shape)
                for shape in self.logical_shapes
            )
        ):
            raise ValueError("logical_shapes must preserve positive dimensions")
        valid_channels = {
            "dispatch",
            "profiler",
            "backend",
            "numpy_guard",
            "os_hwm",
        }
        for name in ("assigned_channels", "observed_channels"):
            channels = getattr(self, name)
            if (
                type(channels) is not tuple
                or (name == "assigned_channels" and not channels)
                or len(set(channels)) != len(channels)
                or any(item not in valid_channels for item in channels)
            ):
                raise ValueError(f"{name} is not a unique closed inventory")
        if type(self.execution_witnessed) is not bool:
            raise ValueError("execution_witnessed must be a bool")
        if self.event_sha256 is not None:
            _sha256(self.event_sha256, "event_sha256")
        if self.execution_witnessed != (self.event_sha256 is not None):
            raise ValueError("event identity must match execution witness")
        if type(self.assignment_complete) is not bool or type(self.detected) is not bool:
            raise ValueError("control decisions must be bools")
        if self.assignment_complete != set(self.assigned_channels).issubset(
            self.observed_channels
        ):
            raise ValueError("assignment_complete must derive from observed channels")
        expected_status = (
            GateStatus.INCONCLUSIVE
            if not self.execution_witnessed
            else GateStatus.FAIL
            if not self.detected
            else GateStatus.INCONCLUSIVE
            if not self.assignment_complete
            else GateStatus.PASS
        )
        status = _status(self.status)
        if status is not expected_status:
            raise ValueError("control status does not match its witnessed decision")
        _obligations(self.obligations, status=status)


H8ChildMode = Literal["production", "profiler", "negative_control"]


@dataclass(frozen=True, slots=True)
class H8ChildRequest:
    mode: H8ChildMode
    seed: int
    repetition: int | None
    config_sha256: str
    protocol_sha256: str
    control_id: str | None

    def __post_init__(self) -> None:
        if self.mode not in ("production", "profiler", "negative_control"):
            raise ValueError("child mode is outside the frozen union")
        _positive_int(self.seed, "seed")
        _sha256(self.config_sha256, "config_sha256")
        _sha256(self.protocol_sha256, "protocol_sha256")
        if self.mode == "production":
            if (
                type(self.repetition) is not int
                or not 0 <= self.repetition <= 4
                or self.control_id is not None
            ):
                raise ValueError("production requests require repetition 0..4")
        elif self.mode == "profiler":
            if self.repetition is not None or self.control_id is not None:
                raise ValueError("profiler requests have no repetition or control")
        else:
            if self.repetition is not None or self.control_id not in H8_NEGATIVE_CONTROL_IDS:
                raise ValueError("negative-control requests require one frozen control")


@dataclass(frozen=True, slots=True)
class H8ChildResult:
    mode: H8ChildMode
    seed: int
    repetition: int | None
    input_sha256: str
    objective: H8ObjectiveTerms | None
    storage: BlockStorageRecord | None
    fill: BlockFillRecord | None
    workspace: BlockWorkspaceRecord | None
    counters: BackendCounterSnapshot | None
    allocation: H8AllocationRecord
    resources: H8ResourceRecord
    invariants: tuple[H8InvariantRecord, ...]

    def __post_init__(self) -> None:
        if self.mode not in ("production", "profiler", "negative_control"):
            raise ValueError("child mode is outside the frozen union")
        _positive_int(self.seed, "seed")
        _sha256(self.input_sha256, "input_sha256")
        if type(self.allocation) is not H8AllocationRecord:
            raise ValueError("allocation must be H8AllocationRecord")
        if type(self.resources) is not H8ResourceRecord:
            raise ValueError("resources must be H8ResourceRecord")
        if (
            type(self.invariants) is not tuple
            or not self.invariants
            or any(type(item) is not H8InvariantRecord for item in self.invariants)
        ):
            raise ValueError("child invariants must be nonempty typed records")
        invariant_ids = tuple(item.invariant_id for item in self.invariants)
        if len(set(invariant_ids)) != len(invariant_ids):
            raise ValueError("child invariant IDs must be unique")
        endpoints = (
            self.objective,
            self.storage,
            self.fill,
            self.workspace,
            self.counters,
        )
        if self.mode in ("production", "profiler"):
            if self.mode == "production":
                if type(self.repetition) is not int or not 0 <= self.repetition <= 4:
                    raise ValueError("production result requires repetition 0..4")
            elif self.repetition is not None:
                raise ValueError("profiler result cannot carry a repetition")
            expected_types = (
                H8ObjectiveTerms,
                BlockStorageRecord,
                BlockFillRecord,
                BlockWorkspaceRecord,
                BackendCounterSnapshot,
            )
            if any(
                type(value) is not expected
                for value, expected in zip(endpoints, expected_types, strict=True)
            ):
                raise ValueError("complete child endpoints are required")
        elif self.repetition is not None or any(value is not None for value in endpoints):
            raise ValueError("negative-control results cannot carry production endpoints")


def _validate_reference_common(value: object, *, expected_kind: str) -> None:
    if getattr(value, "kind", None) != expected_kind:
        raise ValueError(f"reference kind must be {expected_kind}")
    for name in (
        "artifact_path",
        "result_path",
        "ledger_path",
    ):
        _nonempty(getattr(value, name), name)
    for name in (
        "manifest_sha256",
        "result_sha256",
        "ledger_sha256",
        "producer_dirty_digest",
    ):
        _sha256(getattr(value, name), name)
    _git_object_id(getattr(value, "producer_head"), "producer_head")
    if getattr(value, "status", None) != "pass":
        raise ValueError("H8 prerequisite references must bind PASS artifacts")
    for name in ("content_hashes", "payload_hashes"):
        frozen = _freeze_digest_mapping(getattr(value, name), name)
        object.__setattr__(value, name, frozen)


@dataclass(frozen=True, slots=True)
class H8H1H5Reference:
    kind: Literal["h1_h5"]
    artifact_path: str
    manifest_sha256: str
    result_path: str
    result_sha256: str
    content_hashes: Mapping[str, str]
    payload_hashes: Mapping[str, str]
    ledger_path: str
    ledger_sha256: str
    producer_head: str
    producer_dirty_digest: str
    candidate_junit_sha256: str
    status: Literal["pass"]

    def __post_init__(self) -> None:
        _validate_reference_common(self, expected_kind="h1_h5")
        _sha256(self.candidate_junit_sha256, "candidate_junit_sha256")


@dataclass(frozen=True, slots=True)
class H8H1PrefixPriorReference:
    kind: Literal["h1_prefix_prior"]
    artifact_path: str
    manifest_sha256: str
    result_path: str
    result_sha256: str
    content_hashes: Mapping[str, str]
    payload_hashes: Mapping[str, str]
    ledger_path: str
    ledger_sha256: str
    producer_head: str
    producer_dirty_digest: str
    candidate_junit_sha256: str
    status: Literal["pass"]

    def __post_init__(self) -> None:
        _validate_reference_common(self, expected_kind="h1_prefix_prior")
        _sha256(self.candidate_junit_sha256, "candidate_junit_sha256")


@dataclass(frozen=True, slots=True)
class H8H6PrefixSemanticFamilyReference:
    semantic_family_index: int
    semantic_family_sha256: str
    validation_payload_sha256: str
    certificate_sha256: str

    def __post_init__(self) -> None:
        _nonnegative_int(
            self.semantic_family_index,
            "semantic_family_index",
        )
        for name in (
            "semantic_family_sha256",
            "validation_payload_sha256",
            "certificate_sha256",
        ):
            _sha256(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class H8H6PrefixReference:
    kind: Literal["h6_prefix"]
    artifact_path: str
    manifest_sha256: str
    result_path: str
    result_sha256: str
    content_hashes: Mapping[str, str]
    payload_hashes: Mapping[str, str]
    config_schema: Literal["h6-prefix-config-v3"]
    validation_schema: Literal["h6-prefix-validation-set-v2"]
    certificate_set_schema: Literal["h6-prefix-certificate-set-v2"]
    config_sha256: str
    workload_plan_sha256: str
    validation_payload_sha256: str
    prefix_certificate_set_sha256: str
    semantic_families: tuple[H8H6PrefixSemanticFamilyReference, ...]
    ledger_path: str
    ledger_sha256: str
    producer_head: str
    producer_dirty_digest: str
    candidate_junit_sha256: str
    status: Literal["pass"]

    def __post_init__(self) -> None:
        _validate_reference_common(self, expected_kind="h6_prefix")
        if self.config_schema != "h6-prefix-config-v3":
            raise ValueError("H6-Prefix config schema must be bounded v3")
        if self.validation_schema != "h6-prefix-validation-set-v2":
            raise ValueError("H6-Prefix validation schema must be bounded v2")
        if self.certificate_set_schema != "h6-prefix-certificate-set-v2":
            raise ValueError("H6-Prefix certificate-set schema must be bounded v2")
        for name in (
            "config_sha256",
            "workload_plan_sha256",
            "validation_payload_sha256",
            "prefix_certificate_set_sha256",
            "candidate_junit_sha256",
        ):
            _sha256(getattr(self, name), name)
        if (
            type(self.semantic_families) is not tuple
            or not self.semantic_families
            or any(
                type(row) is not H8H6PrefixSemanticFamilyReference
                for row in self.semantic_families
            )
        ):
            raise ValueError(
                "H6-Prefix semantic families must be a nonempty exact tuple"
            )
        for row in self.semantic_families:
            row.__post_init__()
        if tuple(
            row.semantic_family_index for row in self.semantic_families
        ) != tuple(range(len(self.semantic_families))):
            raise ValueError(
                "H6-Prefix semantic-family indices must be contiguous and ordered"
            )
        family_hashes = tuple(
            row.semantic_family_sha256 for row in self.semantic_families
        )
        if len(set(family_hashes)) != len(family_hashes):
            raise ValueError("H6-Prefix semantic-family hashes must be unique")


@dataclass(frozen=True, slots=True)
class H8LegacyH6PrefixReference:
    """Readable legacy Prefix reference that cannot authorize H8."""

    kind: Literal["h6_prefix"]
    artifact_path: str
    manifest_sha256: str
    result_path: str
    result_sha256: str
    content_hashes: Mapping[str, str]
    payload_hashes: Mapping[str, str]
    certificate_set_sha256: str
    certificate_hashes: Mapping[str, str]
    ledger_path: str
    ledger_sha256: str
    producer_head: str
    producer_dirty_digest: str
    candidate_junit_sha256: str
    status: Literal["pass"]

    def __post_init__(self) -> None:
        _validate_reference_common(self, expected_kind="h6_prefix")
        _sha256(self.certificate_set_sha256, "certificate_set_sha256")
        object.__setattr__(
            self,
            "certificate_hashes",
            _freeze_digest_mapping(
                self.certificate_hashes,
                "certificate_hashes",
            ),
        )
        _sha256(self.candidate_junit_sha256, "candidate_junit_sha256")


@dataclass(frozen=True, slots=True)
class H8H7Reference:
    kind: Literal["h7"]
    artifact_path: str
    manifest_sha256: str
    result_path: str
    result_sha256: str
    content_hashes: Mapping[str, str]
    payload_hashes: Mapping[str, str]
    result_pointer_path: str
    result_pointer_sha256: str
    fixture_set_sha256: str
    ledger_path: str
    ledger_sha256: str
    producer_head: str
    producer_dirty_digest: str
    candidate_junit_sha256: str
    status: Literal["pass"]

    def __post_init__(self) -> None:
        _validate_reference_common(self, expected_kind="h7")
        _nonempty(self.result_pointer_path, "result_pointer_path")
        _sha256(self.result_pointer_sha256, "result_pointer_sha256")
        _sha256(self.fixture_set_sha256, "fixture_set_sha256")
        _sha256(self.candidate_junit_sha256, "candidate_junit_sha256")


@dataclass(frozen=True, slots=True)
class H8LegacyH6PredictionReference:
    """Readable v1 H6-Prediction binding that can never authorize H8."""

    kind: Literal["h6_prediction"]
    artifact_path: str
    manifest_sha256: str
    result_path: str
    result_sha256: str
    content_hashes: Mapping[str, str]
    payload_hashes: Mapping[str, str]
    experiment_sha256: str
    ledger_path: str
    ledger_sha256: str
    producer_head: str
    producer_dirty_digest: str
    candidate_junit_sha256: str | None
    status: Literal["pass"]

    def __post_init__(self) -> None:
        _validate_reference_common(self, expected_kind="h6_prediction")
        _sha256(self.experiment_sha256, "experiment_sha256")
        if self.candidate_junit_sha256 is not None:
            _sha256(self.candidate_junit_sha256, "candidate_junit_sha256")


@dataclass(frozen=True, slots=True)
class H8H6PredictionReference:
    """Amended H6-Prediction reference required by registry v2."""

    kind: Literal["h6_prediction"]
    prediction_schema: Literal["h6-prediction-amended-v2"]
    config_schema: Literal["h6-prediction-config-v2"]
    readiness_schema: Literal["h6-prediction-readiness-v2"]
    metrics_schema: Literal["h6-prediction-metrics-v2"]
    result_schema: Literal["h6-prediction-result-v2"]
    artifact_path: str
    manifest_sha256: str
    result_path: str
    result_sha256: str
    content_hashes: Mapping[str, str]
    payload_hashes: Mapping[str, str]
    experiment_sha256: str
    config_sha256: str
    readiness_artifact_path: str
    readiness_manifest_sha256: str
    readiness_sha256: str
    correctness_artifact_paths: Mapping[str, str]
    h1_prefix_prior_artifact_path: str
    smc_accuracy_artifact_path: str
    smc_accuracy_manifest_sha256: str
    h6_prefix_artifact_path: str
    h6_prefix_manifest_sha256: str
    blinded_data_artifact_path: str
    blinded_data_manifest_sha256: str
    matching_artifact_path: str
    matching_manifest_sha256: str
    matching_set_sha256: str
    h1_prefix_prior_generative_factor_schema_sha256: str
    smc_bias_semantics_sha256: str
    objective_gate_spec_sha256: str
    metrics_sha256: str
    ledger_path: str
    ledger_sha256: str
    producer_head: str
    producer_dirty_digest: str
    candidate_junit_sha256: str | None
    status: Literal["pass"]

    def __post_init__(self) -> None:
        _validate_reference_common(self, expected_kind="h6_prediction")
        if (
            self.prediction_schema != "h6-prediction-amended-v2"
            or self.config_schema != "h6-prediction-config-v2"
            or self.readiness_schema != "h6-prediction-readiness-v2"
            or self.metrics_schema != "h6-prediction-metrics-v2"
            or self.result_schema != "h6-prediction-result-v2"
        ):
            raise ValueError("H6-Prediction reference must retain the amended v2 schemas")
        for name in (
            "readiness_artifact_path",
            "h1_prefix_prior_artifact_path",
            "smc_accuracy_artifact_path",
            "h6_prefix_artifact_path",
            "blinded_data_artifact_path",
            "matching_artifact_path",
        ):
            _nonempty(getattr(self, name), name)
        if (
            not isinstance(self.correctness_artifact_paths, Mapping)
            or tuple(self.correctness_artifact_paths) != ("H1", "H2", "H3", "H5")
            or any(
                type(path) is not str or not path
                for path in self.correctness_artifact_paths.values()
            )
        ):
            raise ValueError(
                "H6-Prediction correctness artifact paths must be exact H1/H2/H3/H5"
            )
        object.__setattr__(
            self,
            "correctness_artifact_paths",
            MappingProxyType(dict(self.correctness_artifact_paths)),
        )
        for name in (
            "experiment_sha256",
            "config_sha256",
            "readiness_manifest_sha256",
            "readiness_sha256",
            "smc_accuracy_manifest_sha256",
            "h6_prefix_manifest_sha256",
            "blinded_data_manifest_sha256",
            "matching_manifest_sha256",
            "matching_set_sha256",
            "h1_prefix_prior_generative_factor_schema_sha256",
            "smc_bias_semantics_sha256",
            "objective_gate_spec_sha256",
            "metrics_sha256",
        ):
            _sha256(getattr(self, name), name)
        if self.experiment_sha256 != self.config_sha256:
            raise ValueError(
                "H6-Prediction experiment identity must equal its resolved config identity"
            )
        if self.candidate_junit_sha256 is not None:
            _sha256(self.candidate_junit_sha256, "candidate_junit_sha256")


@dataclass(frozen=True, slots=True)
class CurrentH8PrerequisiteRefs:
    candidate_head: str
    candidate_dirty_digest: str
    candidate_junit_sha256: str
    h7_compatibility_refs: Mapping[str, H7PredecessorReference]
    h1_h5: H8H1H5Reference
    h1_prefix_prior: H8H1PrefixPriorReference
    h6_prefix: H8H6PrefixReference | H8LegacyH6PrefixReference
    h7: H8H7Reference
    h6_prediction: H8H6PredictionReference | H8LegacyH6PredictionReference
    registry_sha256: str

    def __post_init__(self) -> None:
        _git_object_id(self.candidate_head, "candidate_head")
        _sha256(self.candidate_dirty_digest, "candidate_dirty_digest")
        _sha256(self.candidate_junit_sha256, "candidate_junit_sha256")
        if not isinstance(self.h7_compatibility_refs, Mapping):
            raise ValueError("h7_compatibility_refs must be a mapping")
        frozen_h7 = dict(self.h7_compatibility_refs)
        if tuple(frozen_h7) != H8_H7_COMPATIBILITY_REFERENCE_KEYS:
            raise ValueError("H7 compatibility references must keep exact key order")
        if any(type(value) is not H7PredecessorReference for value in frozen_h7.values()):
            raise ValueError("H7 compatibility references must retain exact types")
        for value in frozen_h7.values():
            if (
                value.git_head != self.candidate_head
                or value.dirty_digest != self.candidate_dirty_digest
                or value.junit_sha256 != self.candidate_junit_sha256
            ):
                raise ValueError("H7 references must bind the same candidate")
        object.__setattr__(
            self,
            "h7_compatibility_refs",
            MappingProxyType(frozen_h7),
        )
        expected_types = (
            (self.h1_h5, H8H1H5Reference),
            (self.h1_prefix_prior, H8H1PrefixPriorReference),
            (self.h7, H8H7Reference),
        )
        if any(type(value) is not expected for value, expected in expected_types):
            raise ValueError("current H8 references must retain exact variants")
        if type(self.h6_prefix) not in (
            H8H6PrefixReference,
            H8LegacyH6PrefixReference,
        ):
            raise ValueError(
                "H6-Prefix must retain an exact bounded or readable legacy variant"
            )
        if type(self.h6_prediction) not in (
            H8H6PredictionReference,
            H8LegacyH6PredictionReference,
        ):
            raise ValueError(
                "H6-Prediction must retain an exact amended or readable legacy variant"
            )
        if (
            type(self.h6_prefix) is H8H6PrefixReference
            and type(self.h6_prediction) is H8LegacyH6PredictionReference
        ):
            raise ValueError(
                "bounded H6-Prefix cannot be combined with the legacy H8 registry"
            )
        for reference in (
            self.h1_h5,
            self.h1_prefix_prior,
            self.h6_prefix,
            self.h7,
        ):
            if (
                reference.producer_head != self.candidate_head
                or reference.producer_dirty_digest != self.candidate_dirty_digest
                or reference.candidate_junit_sha256
                != self.candidate_junit_sha256
            ):
                raise ValueError("current references must bind the same candidate")
        if type(self.h6_prediction) is H8H6PredictionReference and (
            self.h6_prediction.producer_head != self.candidate_head
            or self.h6_prediction.producer_dirty_digest
            != self.candidate_dirty_digest
            or self.h6_prediction.candidate_junit_sha256
            != self.candidate_junit_sha256
        ):
            raise ValueError(
                "amended H6-Prediction must bind the same candidate and JUnit"
            )
        transitive_mismatches: list[str] = []
        for key, reference in (
            ("h1_h5", self.h1_h5),
            ("h1_prefix_prior", self.h1_prefix_prior),
            ("h6_prefix", self.h6_prefix),
        ):
            transitive = frozen_h7[key]
            direct_identity = (
                reference.artifact_path,
                reference.producer_head,
                reference.producer_dirty_digest,
                reference.candidate_junit_sha256,
                reference.manifest_sha256,
                tuple(reference.payload_hashes.items()),
                reference.ledger_path,
                reference.ledger_sha256,
            )
            transitive_identity = (
                transitive.artifact_path,
                transitive.git_head,
                transitive.dirty_digest,
                transitive.junit_sha256,
                transitive.manifest_sha256,
                tuple(transitive.payload_hashes.items()),
                transitive.ledger_path,
                transitive.ledger_sha256,
            )
            if direct_identity != transitive_identity:
                transitive_mismatches.append(key)
        if (
            transitive_mismatches
            and type(self.h6_prediction) is H8H6PredictionReference
        ):
            raise ValueError(
                "H8 direct references differ from H7 transitive references: "
                + ",".join(transitive_mismatches)
            )
        _sha256(self.registry_sha256, "registry_sha256")

    @property
    def registry_schema_version(self) -> Literal[
        "h8-current-candidate-refs-v1",
        "h8-current-candidate-refs-v2",
        "h8-current-candidate-refs-v3",
    ]:
        if (
            type(self.h6_prefix) is H8H6PrefixReference
            and type(self.h6_prediction) is H8H6PredictionReference
        ):
            return "h8-current-candidate-refs-v3"
        if type(self.h6_prediction) is H8H6PredictionReference:
            return "h8-current-candidate-refs-v2"
        return "h8-current-candidate-refs-v1"

    @property
    def prerequisite_obligations(self) -> tuple[str, ...]:
        if self.registry_schema_version == "h8-current-candidate-refs-v3":
            return ()
        obligations: list[str] = []
        if type(self.h6_prefix) is H8LegacyH6PrefixReference:
            obligations.append(
                "h8_prerequisite_legacy_registry_requires_bounded_h6_prefix_v3"
            )
        if type(self.h6_prediction) is H8LegacyH6PredictionReference:
            obligations.append(
                "h8_prerequisite_registry_v1_requires_amended_h6_prediction_v2"
            )
        for key, reference in (
            ("h1_h5", self.h1_h5),
            ("h1_prefix_prior", self.h1_prefix_prior),
            ("h6_prefix", self.h6_prefix),
        ):
            transitive = self.h7_compatibility_refs[key]
            if (
                reference.artifact_path,
                reference.producer_head,
                reference.producer_dirty_digest,
                reference.candidate_junit_sha256,
                reference.manifest_sha256,
                tuple(reference.payload_hashes.items()),
                reference.ledger_path,
                reference.ledger_sha256,
            ) != (
                transitive.artifact_path,
                transitive.git_head,
                transitive.dirty_digest,
                transitive.junit_sha256,
                transitive.manifest_sha256,
                tuple(transitive.payload_hashes.items()),
                transitive.ledger_path,
                transitive.ledger_sha256,
            ):
                obligations.append(
                    f"h8_prerequisite_legacy_{key}_differs_from_h7_transitive"
                )
        return tuple(obligations)


@dataclass(frozen=True, slots=True)
class H8GateEvaluation:
    result: object
    validation_payload_canonical_json: bytes
    validation_payload_sha256: str
    dependency_closure_sha256: str
    preregistration_sha256: str
    interpretation_sha256: str
    evaluation_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.result).__name__ != "H8GateResult"
            or type(self.result).__module__ != "vfe4.types.results"
        ):
            raise ValueError("gate evaluation requires results.py::H8GateResult")
        canonical = _canonical_json_bytes(
            self.validation_payload_canonical_json,
            "validation_payload_canonical_json",
        )
        if hashlib.sha256(canonical).hexdigest() != self.validation_payload_sha256:
            raise ValueError("validation payload hash does not match its bytes")
        for name in (
            "dependency_closure_sha256",
            "preregistration_sha256",
            "interpretation_sha256",
            "evaluation_sha256",
        ):
            _sha256(getattr(self, name), name)


@dataclass(frozen=True, slots=True, init=False)
class BlockTridiagonalPrecision:
    layout: BlockChainLayout
    _diag: Tensor = field(repr=False)
    _lower: Tensor = field(repr=False)

    def __init__(self, layout: BlockChainLayout, diag: Tensor, lower: Tensor) -> None:
        if type(layout) is not BlockChainLayout:
            raise ValueError("layout must be a BlockChainLayout")
        block = layout.block_size
        owned_diag = _owned_block_tensor(
            diag,
            expected_shape=(layout.population_size, block, block),
            name="diag",
        )
        owned_lower = _owned_block_tensor(
            lower,
            expected_shape=(layout.horizon, block, block),
            name="lower",
        )
        object.__setattr__(self, "layout", layout)
        object.__setattr__(self, "_diag", owned_diag)
        object.__setattr__(self, "_lower", owned_lower)

    @property
    def diag(self) -> Tensor:
        return self._diag.clone()

    @property
    def lower(self) -> Tensor:
        return self._lower.clone()

    def _block_refs(self) -> tuple[Tensor, Tensor]:
        """Package-private owned blocks for allocation-aware backend use."""

        return self._diag, self._lower


@dataclass(frozen=True, slots=True, init=False)
class SelectedInverseBlocks:
    layout: BlockChainLayout
    _diag: Tensor = field(repr=False)
    _lower: Tensor = field(repr=False)

    def __init__(self, layout: BlockChainLayout, diag: Tensor, lower: Tensor) -> None:
        if type(layout) is not BlockChainLayout:
            raise ValueError("layout must be a BlockChainLayout")
        block = layout.block_size
        owned_diag = _owned_block_tensor(
            diag,
            expected_shape=(layout.population_size, block, block),
            name="diag",
        )
        owned_lower = _owned_block_tensor(
            lower,
            expected_shape=(layout.horizon, block, block),
            name="lower",
        )
        object.__setattr__(self, "layout", layout)
        object.__setattr__(self, "_diag", owned_diag)
        object.__setattr__(self, "_lower", owned_lower)

    @property
    def diag(self) -> Tensor:
        return self._diag.clone()

    @property
    def lower(self) -> Tensor:
        return self._lower.clone()

    def _block_refs(self) -> tuple[Tensor, Tensor]:
        return self._diag, self._lower


@runtime_checkable
class BlockPrecisionFactor(Protocol):
    @property
    def dimension(self) -> int: ...

    @property
    def layout(self) -> BlockChainLayout: ...

    @property
    def pattern(self) -> BlockPatternRecord: ...

    @property
    def storage(self) -> BlockStorageRecord: ...

    @property
    def fill(self) -> BlockFillRecord: ...

    @property
    def workspace(self) -> BlockWorkspaceRecord: ...

    @property
    def diagnostics(self) -> SparseConditionDiagnostics: ...

    @property
    def counters(self) -> BackendCounterSnapshot: ...

    def solve(self, rhs: Tensor) -> Tensor: ...

    def solve_factor(self, rhs: Tensor, *, transpose: bool) -> Tensor: ...

    def logdet(self) -> Tensor: ...

    def selected_inverse(
        self,
        blocks: Sequence[BlockId],
    ) -> SelectedInverseBlocks: ...

    def sample(self, noise: Tensor) -> Tensor: ...

    def quadratic(self, value: Tensor) -> Tensor: ...

    def trace_inverse_product(
        self,
        left: BlockTridiagonalPrecision,
    ) -> Tensor: ...


__all__ = [
    "AllowanceRecord",
    "BackendCounterSnapshot",
    "BlockFillRecord",
    "BlockPatternRecord",
    "BlockPrecisionFactor",
    "BlockStorageRecord",
    "BlockStorageExpectation",
    "BlockTridiagonalPrecision",
    "BlockWorkspaceRecord",
    "CurrentH8PrerequisiteRefs",
    "H8AllocationChannel",
    "H8AllocationRecord",
    "H8AllowanceRecord",
    "H8ChildMode",
    "H8ChildRequest",
    "H8ChildResult",
    "H8ControlResult",
    "H8CorrectnessEndpointRecord",
    "H8CorrectnessCell",
    "H8CorrectnessControlResult",
    "H8CorrectnessSourceResult",
    "H8GateEvaluation",
    "H8H1H5Reference",
    "H8H1PrefixPriorReference",
    "H8H6PredictionReference",
    "H8H6PrefixReference",
    "H8H6PrefixSemanticFamilyReference",
    "H8H7Reference",
    "H8LegacyH6PrefixReference",
    "H8LegacyH6PredictionReference",
    "H8InvariantRecord",
    "H8LossyProfilerRow",
    "H8ObjectiveRole",
    "H8ObjectiveTerm",
    "H8ObjectiveTerms",
    "H8OperandRecord",
    "H8OperandSource",
    "H8ProfilerAction",
    "H8ProfilerEventRecord",
    "H8ResourceRecord",
    "H8TensorKey",
    "H8_BASE_CORRECTNESS_ENDPOINT_IDS",
    "H8_CORRECTNESS_CASES",
    "H8_CORRECTNESS_CONTROL_IDS",
    "H8_CORRECTNESS_ORDERED_SOURCE_PAIRS",
    "H8_CORRECTNESS_SOURCES",
    "H8_H7_COMPATIBILITY_REFERENCE_KEYS",
    "H8_H7_PLAN_SHA256",
    "H8_INTERPRETATION_SHA256",
    "H8_MAXIMUM_ALLOWANCE_SCALE_FRACTION",
    "H8_MAX_PROCESS_INCREMENTAL_BYTES",
    "H8_MAX_SECONDS",
    "H8_MAX_TORCH_POPULATION_BYTES",
    "H8_MIN_CHOLESKY_PIVOT",
    "H8_NEGATIVE_CONTROL_IDS",
    "H8_NONCLAIMS",
    "H8_PROBLEM_DRAW_SCHEMA_SHA256",
    "H8_PRODUCTION_SAMPLE_SEED_PAIRS",
    "H8_PRODUCTION_SEEDS",
    "H8_PROFILER_API_CONTRACT_SHA256",
    "H8_PROFILER_MEMORY_SOURCE_SHA256",
    "H8_PROFILER_SOURCE_SHA256",
    "H8_REQUIRED_OPERATIONS",
    "H8_VERIFIER_PREFIX",
    "OperandRecord",
    "SelectedInverseBlocks",
    "SparseConditionDiagnostics",
    "h8_correctness_endpoint_ids",
]
