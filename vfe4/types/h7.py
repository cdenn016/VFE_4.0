"""Immutable auxiliary records for H7 population-frame covariance.

The H7 boundary deliberately separates graph-preserving borrowed tensor views
from owned evidence snapshots.  Borrowed views retain the caller's exact
object, storage, version, dtype, shape, device, and autograd identity.  Owned
records defensively copy tensors and mappings and bind their semantic content
with domain-separated SHA-256 identities.

``H7GateResult`` is intentionally not defined here.  Task 7 owns its sole
definition in :mod:`vfe4.types.results`.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, Literal, TypeAlias, TypeVar, cast

import torch

if TYPE_CHECKING:
    from .results import H7GateResult


H7ActionKind = Literal["diagonal_base", "internal_product"]
H7DecoderPolicy = Literal["transform", "fixed"]
H7FrameProfile = Literal["identity", "nonidentity", "h1_v1"]
H7RecognitionFamily = Literal[
    "structured_full_block", "factorized_diagonal_within_fiber"
]
H7RecognitionRepresentation = Literal[
    "structured_full_block",
    "factorized_diagonal_within_fiber",
    "unrestricted_full_block_pushforward",
]
H7TrialRole = Literal["scalar_regression", "positive_covariance", "expected_negative"]
H7ExpectedPredicate = Literal[
    "complete_covariance",
    "centered_decoder_stabilizer_invariance",
    "decisive_outside_stabilizer_change",
]
H7TrialId = Literal[
    "scalar-base-transformed",
    "scalar-internal-transformed",
    "matrix-identity-base-transformed",
    "matrix-identity-internal-transformed",
    "matrix-nonidentity-base-transformed",
    "matrix-nonidentity-internal-transformed",
    "matrix-fixed-decoder-centered-stabilizer",
    "matrix-fixed-decoder-outside-stabilizer",
]
H7ControlId = Literal[
    "wrong_covariance_congruence",
    "wrong_precision_congruence",
    "history_scorer_wrong_source_inverse",
    "reversed_link_order",
    "reverse_arrow_B",
    "wrong_decoder_dual_action",
    "fixed_decoder_outside_stabilizer",
    "omitted_density_jacobian",
    "reversed_logdet_sign",
    "entropy_false_invariance",
    "changed_h1_source_probability",
    "diagonal_for_internal_action",
]
H7SourceBank = Literal["model", "state"]
H7Channel = Literal["z", "m"]
H7BudgetCategory = Literal[
    "vector",
    "information",
    "offset",
    "decoder",
    "covariance",
    "precision",
    "second_moment",
    "map",
    "cocycle",
    "density",
    "local_term",
    "complete_objective",
    "backward",
]
H7OperandRole = Literal["original", "transformed", "reference", "recovered", "oracle"]
H7OperationKind = Literal[
    "exact_identity",
    "direct_solve",
    "matrix_product",
    "quadratic_form",
    "logdet",
    "analytic_density",
    "gauss_hermite",
    "pair_comparison",
]
H7AllowanceKind = Literal[
    "operation_rounding", "quadrature_convergence", "reference_rounding"
]
H7ResidualCategory = Literal[
    "tensor",
    "law",
    "cocycle",
    "density",
    "jacobian",
    "source",
    "decoder",
    "local_term",
    "monolithic",
    "evidence",
    "posterior_kl",
    "absolute",
    "relative",
    "backward",
]
H7DensityRole = Literal["p", "q", "log_ratio"]
H7TrialActionProfile = Literal[
    "scalar_base",
    "scalar_internal",
    "matrix_diagonal",
    "matrix_internal",
    "matrix_fixed_decoder_stabilizer",
]
_H7_OBJECTIVE_LOCAL_TERM_IDS = (
    "expected_log_emission[1]",
    "expected_log_emission[2]",
    "model_source_kl[1]",
    "state_source_kl[1]",
    "model_transition_kl[1]",
    "state_transition_kl[1]",
    "model_source_kl[2]",
    "state_source_kl[2]",
    "model_transition_kl[2]",
    "state_transition_kl[2]",
    "joint_recognition_entropy",
)

H7_SCALAR_TRIAL_IDS: tuple[H7TrialId, H7TrialId] = (
    "scalar-base-transformed",
    "scalar-internal-transformed",
)
H7_MATRIX_TRIAL_IDS: tuple[
    H7TrialId, H7TrialId, H7TrialId, H7TrialId, H7TrialId, H7TrialId
] = (
    "matrix-identity-base-transformed",
    "matrix-identity-internal-transformed",
    "matrix-nonidentity-base-transformed",
    "matrix-nonidentity-internal-transformed",
    "matrix-fixed-decoder-centered-stabilizer",
    "matrix-fixed-decoder-outside-stabilizer",
)
H7_REQUIRED_TRIAL_IDS: tuple[H7TrialId, ...] = (
    *H7_SCALAR_TRIAL_IDS,
    *H7_MATRIX_TRIAL_IDS,
)
H7_CONTROL_IDS: tuple[H7ControlId, ...] = (
    "wrong_covariance_congruence",
    "wrong_precision_congruence",
    "history_scorer_wrong_source_inverse",
    "reversed_link_order",
    "reverse_arrow_B",
    "wrong_decoder_dual_action",
    "fixed_decoder_outside_stabilizer",
    "omitted_density_jacobian",
    "reversed_logdet_sign",
    "entropy_false_invariance",
    "changed_h1_source_probability",
    "diagonal_for_internal_action",
)
H7_DENSITY_PROBE_TABLE_RAW_SHA256 = (
    "4857af296e84a33f47964c3bca65e0d42967009aa5c79a52bcc98d6db04382c6"
)

H7_TRIAL_CONTRACTS: Mapping[
    H7TrialId,
    tuple[
        H7TrialRole,
        H7ExpectedPredicate,
        Literal["h1-v1", "h7-v1"],
        H7FrameProfile,
        H7DecoderPolicy,
        H7TrialActionProfile,
    ],
] = MappingProxyType(
    {
        "scalar-base-transformed": (
            "scalar_regression",
            "complete_covariance",
            "h1-v1",
            "h1_v1",
            "transform",
            "scalar_base",
        ),
        "scalar-internal-transformed": (
            "scalar_regression",
            "complete_covariance",
            "h1-v1",
            "h1_v1",
            "transform",
            "scalar_internal",
        ),
        "matrix-identity-base-transformed": (
            "positive_covariance",
            "complete_covariance",
            "h7-v1",
            "identity",
            "transform",
            "matrix_diagonal",
        ),
        "matrix-identity-internal-transformed": (
            "positive_covariance",
            "complete_covariance",
            "h7-v1",
            "identity",
            "transform",
            "matrix_internal",
        ),
        "matrix-nonidentity-base-transformed": (
            "positive_covariance",
            "complete_covariance",
            "h7-v1",
            "nonidentity",
            "transform",
            "matrix_diagonal",
        ),
        "matrix-nonidentity-internal-transformed": (
            "positive_covariance",
            "complete_covariance",
            "h7-v1",
            "nonidentity",
            "transform",
            "matrix_internal",
        ),
        "matrix-fixed-decoder-centered-stabilizer": (
            "positive_covariance",
            "centered_decoder_stabilizer_invariance",
            "h7-v1",
            "nonidentity",
            "fixed",
            "matrix_fixed_decoder_stabilizer",
        ),
        "matrix-fixed-decoder-outside-stabilizer": (
            "expected_negative",
            "decisive_outside_stabilizer_change",
            "h7-v1",
            "nonidentity",
            "fixed",
            "matrix_diagonal",
        ),
    }
)

_FROZEN_ACTION_PROFILE_VALUES: Mapping[
    H7TrialActionProfile,
    tuple[tuple[tuple[float, ...], ...], ...],
] = MappingProxyType(
    {
        "scalar_base": (((1.25,),), ((1.25,),), ((1.25,),)),
        "scalar_internal": (((0.8,),), ((1.1,),), ((1.4,),)),
        "matrix_diagonal": (
            ((1.2, 0.2), (-0.1, 0.9)),
            ((1.2, 0.2), (-0.1, 0.9)),
            ((1.2, 0.2), (-0.1, 0.9)),
        ),
        "matrix_internal": (
            ((1.25, 0.1), (0.05, 0.95)),
            ((0.85, -0.2), (0.1, 1.15)),
            ((1.05, 0.25), (-0.15, 0.9)),
        ),
        "matrix_fixed_decoder_stabilizer": (
            ((1.0, 0.0), (0.2, 1.1)),
            ((1.0, 0.0), (0.2, 1.1)),
            ((1.0, 0.0), (0.2, 1.1)),
        ),
    }
)

_LOWER_HEX = frozenset("0123456789abcdef")
_T = TypeVar("_T", bound="_H7IntegrityRecord")


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase 64-hex SHA-256")
    return value


def _require_nonempty(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _tensor_raw_bytes(value: torch.Tensor) -> bytes:
    cpu = value.detach().to(device="cpu").contiguous()
    try:
        return cpu.numpy().tobytes(order="C")
    except (TypeError, RuntimeError):
        return bytes(cpu.view(torch.uint8).reshape(-1).tolist())


def _require_owned_tensor(
    value: object,
    name: str,
    *,
    shape: tuple[int, ...] | None = None,
    ndim: int | None = None,
) -> torch.Tensor:
    if type(value) is not H7OwnedTensorSnapshot:
        raise ValueError(f"{name} must be an exact owned tensor snapshot")
    value.assert_intact()
    tensor = value.value()
    if tensor.dtype != torch.float64:
        raise ValueError(f"{name} must use torch.float64")
    if shape is not None and tuple(tensor.shape) != shape:
        raise ValueError(f"{name} has the wrong shape")
    if ndim is not None and tensor.ndim != ndim:
        raise ValueError(f"{name} has the wrong rank")
    return tensor


def _require_receiver_source(
    receiver_t: object,
    source_j: object,
    name: str,
    *,
    allow_initial: bool,
) -> None:
    if allow_initial and receiver_t is None and source_j is None:
        return
    if (
        type(receiver_t) is not int
        or receiver_t not in (1, 2)
        or type(source_j) is not int
        or source_j < 0
        or source_j >= receiver_t
    ):
        raise ValueError(f"{name} receiver/source pair is invalid")


def _require_exact_primitive_field(
    value: object, annotation: object, name: str
) -> None:
    """Reject Python's bool/int and int/float equality aliases at record boundaries."""

    annotation_text = annotation if type(annotation) is str else str(annotation)
    primitive_types = (bool, bytes, float, int, str)
    primitive: type[object] | None = (
        cast(type[object], annotation)
        if annotation in primitive_types
        else {
            "bool": bool,
            "bytes": bytes,
            "float": float,
            "int": int,
            "str": str,
        }.get(annotation_text)
    )
    if primitive is not None and type(value) is not primitive:
        raise ValueError(f"{name} must be an exact {primitive.__name__}")
    optional_primitive: type[object] | None = {
        "float | None": float,
        "int | None": int,
        "str | None": str,
    }.get(annotation_text)
    if (
        optional_primitive is not None
        and value is not None
        and type(value) is not optional_primitive
    ):
        raise ValueError(
            f"{name} must be None or an exact {optional_primitive.__name__}"
        )


def _canonical(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return value.as_posix()
    if type(value) is bytes:
        return {"hex": value.hex(), "length": len(value)}
    if isinstance(value, H7OwnedTensorSnapshot):
        value.assert_intact()
        return {
            "dtype": value.dtype,
            "shape": list(value.shape),
            "device": value.device,
            "raw_bytes_sha256": value.raw_bytes_sha256,
            "snapshot_sha256": value.snapshot_sha256,
        }
    if isinstance(value, (H7BorrowedTensorView, H7BorrowedActionView)):
        raise ValueError(
            "borrowed H7 tensor/action views are unhashed and unpublishable"
        )
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _canonical(getattr(value, item.name))
            for item in fields(value)
            if not item.name.startswith("_H7OwnedTensorSnapshot__")
        }
    if isinstance(value, Mapping):
        entries = sorted(
            ((_canonical(key), _canonical(item)) for key, item in value.items()),
            key=lambda pair: json.dumps(
                pair[0],
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        if all(type(key) is str and key for key in value):
            return {cast(str, key): _canonical(value[key]) for key in sorted(value)}
        return [{"key": key, "value": item} for key, item in entries]
    if type(value) in (tuple, list):
        return [_canonical(item) for item in value]
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("canonical H7 floats must be finite")
        return value.hex()
    if type(value) in (str, int, bool) or value is None:
        return value
    raise ValueError(f"unsupported H7 canonical value {type(value).__name__}")


def canonical_h7_bytes(value: object) -> bytes:
    """Return deterministic canonical bytes for an H7 semantic value."""

    return json.dumps(
        _canonical(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def h7_owned_sha256(domain: str, value: object) -> str:
    """Hash one owned H7 value under an explicit ASCII domain."""

    if type(domain) is not str or not domain or not domain.isascii():
        raise ValueError("H7 hash domain must be nonempty ASCII")
    return hashlib.sha256(
        domain.encode("ascii") + b"\x00" + canonical_h7_bytes(value)
    ).hexdigest()


def _freeze_mapping(value: Mapping[object, object]) -> MappingProxyType:
    recognition_order = {
        "structured_full_block": 0,
        "factorized_diagonal_within_fiber": 1,
    }

    def sort_key(pair: tuple[object, object]) -> tuple[int, object]:
        key = pair[0]
        if type(key) is str and key in recognition_order:
            return (0, recognition_order[key])
        return (1, canonical_h7_bytes(key))

    items = sorted(
        value.items(),
        key=sort_key,
    )
    return MappingProxyType({key: _freeze_container(item) for key, item in items})


def _freeze_container(value: object) -> object:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if type(value) is tuple:
        return tuple(_freeze_container(item) for item in value)
    if type(value) is list:
        return tuple(_freeze_container(item) for item in value)
    return value


class _H7IntegrityRecord:
    """Factory support shared by immutable digest-bearing H7 records."""

    _integrity_field: ClassVar[str]
    _hash_domain: ClassVar[str]

    @classmethod
    def create(cls: type[_T], **values: object) -> _T:
        if cls._integrity_field in values:
            raise ValueError(
                f"{cls._integrity_field} is owned by {cls.__name__}.create"
            )
        semantic = {name: _freeze_container(value) for name, value in values.items()}
        digest = h7_owned_sha256(cls._hash_domain, semantic)
        semantic[cls._integrity_field] = digest
        return cls(**semantic)  # type: ignore[arg-type]

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            frozen = _freeze_container(value)
            if frozen is not value:
                object.__setattr__(self, item.name, frozen)
        for item in fields(self):
            value = getattr(self, item.name)
            _require_exact_primitive_field(value, item.type, item.name)
            if item.name.endswith("_sha256") and type(value) is str:
                _require_sha256(value, item.name)
            if type(value) is float and not math.isfinite(value):
                raise ValueError(f"{item.name} must be finite")
        semantic = {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name != self._integrity_field
        }
        expected = h7_owned_sha256(self._hash_domain, semantic)
        if getattr(self, self._integrity_field) != expected:
            raise ValueError(
                f"{self._integrity_field} does not match {type(self).__name__}"
            )


@dataclass(frozen=True)
class H7RawTensorIdentity:
    object_id: int
    storage_data_ptr: int
    storage_version: int
    raw_bytes_sha256: str
    dtype: str
    shape: tuple[int, ...]
    device: str
    contiguous: bool
    requires_grad: bool

    def __post_init__(self) -> None:
        if (
            type(self.object_id) is not int
            or type(self.storage_data_ptr) is not int
            or type(self.storage_version) is not int
            or self.storage_version < 0
            or type(self.shape) is not tuple
            or any(type(size) is not int or size < 0 for size in self.shape)
            or type(self.contiguous) is not bool
            or type(self.requires_grad) is not bool
        ):
            raise ValueError("invalid raw H7 tensor identity")
        _require_sha256(self.raw_bytes_sha256, "raw_bytes_sha256")
        _require_nonempty(self.dtype, "dtype")
        _require_nonempty(self.device, "device")

    @classmethod
    def capture(cls, value: torch.Tensor) -> "H7RawTensorIdentity":
        if not isinstance(value, torch.Tensor):
            raise ValueError("value must be a torch.Tensor")
        if value.is_sparse:
            raise ValueError("H7 borrowed tensors must be dense")
        storage = value.untyped_storage()
        return cls(
            object_id=id(value),
            storage_data_ptr=int(storage.data_ptr()),
            storage_version=int(value._version),
            raw_bytes_sha256=hashlib.sha256(_tensor_raw_bytes(value)).hexdigest(),
            dtype=str(value.dtype).removeprefix("torch."),
            shape=tuple(int(size) for size in value.shape),
            device=str(value.device),
            contiguous=bool(value.is_contiguous()),
            requires_grad=bool(value.requires_grad),
        )


@dataclass(frozen=True)
class H7BorrowedTensorView:
    __hash__: ClassVar[None] = None

    tensor: torch.Tensor
    identity: H7RawTensorIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.tensor, torch.Tensor):
            raise ValueError("tensor must be a torch.Tensor")
        self.assert_intact()

    @classmethod
    def borrow(cls, value: torch.Tensor) -> "H7BorrowedTensorView":
        return cls(value, H7RawTensorIdentity.capture(value))

    def assert_intact(self) -> None:
        observed = H7RawTensorIdentity.capture(self.tensor)
        if observed != self.identity:
            if observed.storage_version != self.identity.storage_version:
                raise ValueError("borrowed tensor version changed")
            if observed.raw_bytes_sha256 != self.identity.raw_bytes_sha256:
                raise ValueError("borrowed tensor raw bytes changed")
            raise ValueError("borrowed tensor identity is stale")


def _borrowed_tensor_semantic(
    value: H7BorrowedTensorView,
) -> Mapping[str, object]:
    """Return a reproducible live-value identity without replacing its graph.

    Runtime object IDs, storage pointers, and version counters remain useful
    stale-view guards, but they are intentionally absent from this semantic
    payload.  Reading bytes from a temporary CPU representation does not
    replace or detach ``value.tensor`` from the caller's autograd graph.
    """

    if type(value) is not H7BorrowedTensorView:
        raise ValueError("borrowed tensor must be an exact H7 view")
    value.assert_intact()
    return {
        "dtype": value.identity.dtype,
        "shape": value.identity.shape,
        "device": value.identity.device,
        "contiguous": value.identity.contiguous,
        "requires_grad": value.identity.requires_grad,
        "raw_bytes_sha256": value.identity.raw_bytes_sha256,
    }


def _borrowed_tensor_mapping_semantic(
    values: Mapping[str, H7BorrowedTensorView],
) -> Mapping[str, Mapping[str, object]]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError("borrowed tensor mapping must be nonempty")
    result: dict[str, Mapping[str, object]] = {}
    for name, value in values.items():
        _require_nonempty(name, "borrowed tensor name")
        if name in result:
            raise ValueError("borrowed tensor names must be unique")
        result[name] = _borrowed_tensor_semantic(value)
    return MappingProxyType({name: result[name] for name in sorted(result)})


@dataclass(frozen=True, init=False)
class H7OwnedTensorSnapshot:
    __owned: torch.Tensor
    capture_identity: H7RawTensorIdentity
    owned_storage_version: int
    dtype: str
    shape: tuple[int, ...]
    device: str
    raw_bytes: bytes
    raw_bytes_sha256: str
    snapshot_sha256: str

    @classmethod
    def capture(cls, value: torch.Tensor) -> "H7OwnedTensorSnapshot":
        if not isinstance(value, torch.Tensor):
            raise ValueError("value must be a torch.Tensor")
        if value.is_sparse:
            raise ValueError("H7 owned snapshots require dense tensors")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError("H7 owned snapshots require finite tensors")
        capture_identity = H7RawTensorIdentity.capture(value)
        owned = value.contiguous().clone()
        raw_bytes = _tensor_raw_bytes(owned)
        raw_sha = hashlib.sha256(raw_bytes).hexdigest()
        storage_version = int(owned._version)
        # Runtime allocation identity is retained as provenance but excluded
        # from the preregistered semantic hash.  Object IDs, storage pointers,
        # and caller versions are process-local and would make action/config
        # identities irreproducible.  Stable dtype/shape/device/raw bytes stay
        # bound below.
        semantic = {
            "capture_contract": {
                "dtype": capture_identity.dtype,
                "shape": capture_identity.shape,
                "device": capture_identity.device,
                "contiguous": capture_identity.contiguous,
                "requires_grad": capture_identity.requires_grad,
            },
            "owned_storage_version": storage_version,
            "dtype": str(owned.dtype).removeprefix("torch."),
            "shape": tuple(int(size) for size in owned.shape),
            "device": str(owned.device),
            "raw_bytes": raw_bytes,
            "raw_bytes_sha256": raw_sha,
        }
        instance = object.__new__(cls)
        object.__setattr__(instance, "_H7OwnedTensorSnapshot__owned", owned)
        object.__setattr__(instance, "capture_identity", capture_identity)
        object.__setattr__(instance, "owned_storage_version", storage_version)
        object.__setattr__(instance, "dtype", semantic["dtype"])
        object.__setattr__(instance, "shape", semantic["shape"])
        object.__setattr__(instance, "device", semantic["device"])
        object.__setattr__(instance, "raw_bytes", raw_bytes)
        object.__setattr__(instance, "raw_bytes_sha256", raw_sha)
        object.__setattr__(
            instance,
            "snapshot_sha256",
            h7_owned_sha256("vfe4.h7.owned-tensor-snapshot.v1", semantic),
        )
        instance.assert_intact()
        return instance

    def value(self) -> torch.Tensor:
        self.assert_intact()
        return self.__owned.clone()

    def assert_intact(self) -> None:
        observed_bytes = _tensor_raw_bytes(self.__owned)
        semantic = {
            "capture_contract": {
                "dtype": self.capture_identity.dtype,
                "shape": self.capture_identity.shape,
                "device": self.capture_identity.device,
                "contiguous": self.capture_identity.contiguous,
                "requires_grad": self.capture_identity.requires_grad,
            },
            "owned_storage_version": self.owned_storage_version,
            "dtype": self.dtype,
            "shape": self.shape,
            "device": self.device,
            "raw_bytes": self.raw_bytes,
            "raw_bytes_sha256": self.raw_bytes_sha256,
        }
        if (
            int(self.__owned._version) != self.owned_storage_version
            or str(self.__owned.dtype).removeprefix("torch.") != self.dtype
            or tuple(self.__owned.shape) != self.shape
            or str(self.__owned.device) != self.device
            or observed_bytes != self.raw_bytes
            or hashlib.sha256(observed_bytes).hexdigest() != self.raw_bytes_sha256
            or h7_owned_sha256("vfe4.h7.owned-tensor-snapshot.v1", semantic)
            != self.snapshot_sha256
        ):
            raise ValueError("owned H7 tensor snapshot integrity changed")


@dataclass(frozen=True)
class H7BorrowedActionView:
    __hash__: ClassVar[None] = None

    elements: tuple[
        H7BorrowedTensorView,
        H7BorrowedTensorView,
        H7BorrowedTensorView,
    ]
    kind: H7ActionKind
    dimension: Literal[1, 2]
    group: Literal["GL+(1,R)", "GL+(2,R)"]

    def __post_init__(self) -> None:
        if (
            type(self.elements) is not tuple
            or len(self.elements) != 3
            or self.kind not in ("diagonal_base", "internal_product")
            or self.dimension not in (1, 2)
        ):
            raise ValueError("invalid borrowed H7 action")
        expected_group = "GL+(1,R)" if self.dimension == 1 else "GL+(2,R)"
        if self.group != expected_group:
            raise ValueError("borrowed H7 action dimension/group mismatch")
        self.assert_intact()

    def assert_intact(self) -> None:
        for item in self.elements:
            if type(item) is not H7BorrowedTensorView:
                raise ValueError("borrowed action elements must be exact views")
            item.assert_intact()
            if item.identity.shape != (self.dimension, self.dimension):
                raise ValueError("borrowed H7 action element shape mismatch")


def _owned_action_elements(
    elements: tuple[
        H7OwnedTensorSnapshot | torch.Tensor,
        H7OwnedTensorSnapshot | torch.Tensor,
        H7OwnedTensorSnapshot | torch.Tensor,
    ],
    *,
    dimension: int,
) -> tuple[H7OwnedTensorSnapshot, H7OwnedTensorSnapshot, H7OwnedTensorSnapshot]:
    if type(elements) is not tuple or len(elements) != 3:
        raise ValueError("H7 actions require exactly three elements")
    snapshots = tuple(
        item
        if type(item) is H7OwnedTensorSnapshot
        else H7OwnedTensorSnapshot.capture(item)
        for item in elements
    )
    for snapshot in snapshots:
        if type(snapshot) is not H7OwnedTensorSnapshot:
            raise ValueError("H7 action elements must be tensor snapshots")
        snapshot.assert_intact()
        value = snapshot.value()
        if tuple(value.shape) != (dimension, dimension):
            raise ValueError("H7 action element shape disagrees with dimension")
        if value.dtype != torch.float64:
            raise ValueError("H7 owned actions require torch.float64 elements")
        sign, logabsdet = torch.linalg.slogdet(value)
        if (
            not bool(torch.isfinite(value).all().item())
            or not bool(torch.isfinite(logabsdet).item())
            or not bool((sign > 0).item())
        ):
            raise ValueError("H7 direct action requires positive determinant")
    return cast(
        tuple[
            H7OwnedTensorSnapshot,
            H7OwnedTensorSnapshot,
            H7OwnedTensorSnapshot,
        ],
        snapshots,
    )


@dataclass(frozen=True)
class H7ScalarReplayAction:
    elements: tuple[
        H7OwnedTensorSnapshot,
        H7OwnedTensorSnapshot,
        H7OwnedTensorSnapshot,
    ]
    kind: H7ActionKind
    dimension: Literal[1]
    group: Literal["GL+(1,R)"]
    representation: Literal["standard_scalar"]
    action_sha256: str

    @classmethod
    def create(
        cls,
        *,
        elements: tuple[
            H7OwnedTensorSnapshot | torch.Tensor,
            H7OwnedTensorSnapshot | torch.Tensor,
            H7OwnedTensorSnapshot | torch.Tensor,
        ],
        kind: H7ActionKind,
    ) -> "H7ScalarReplayAction":
        owned = _owned_action_elements(elements, dimension=1)
        semantic = {
            "elements": owned,
            "kind": kind,
            "dimension": 1,
            "group": "GL+(1,R)",
            "representation": "standard_scalar",
        }
        return cls(
            **semantic,
            action_sha256=h7_owned_sha256("vfe4.h7.scalar-replay-action.v1", semantic),
        )

    def __post_init__(self) -> None:
        if (
            self.kind not in ("diagonal_base", "internal_product")
            or self.dimension != 1
            or self.group != "GL+(1,R)"
            or self.representation != "standard_scalar"
        ):
            raise ValueError("invalid scalar replay action declaration")
        elements = _owned_action_elements(self.elements, dimension=1)
        if self.kind == "diagonal_base" and not all(
            torch.equal(elements[0].value(), item.value()) for item in elements[1:]
        ):
            raise ValueError("diagonal-base action elements must be identical")
        semantic = {
            "elements": self.elements,
            "kind": self.kind,
            "dimension": self.dimension,
            "group": self.group,
            "representation": self.representation,
        }
        if self.action_sha256 != h7_owned_sha256(
            "vfe4.h7.scalar-replay-action.v1", semantic
        ):
            raise ValueError("scalar action_sha256 does not match action")


@dataclass(frozen=True)
class H7GLPlus2Action:
    elements: tuple[
        H7OwnedTensorSnapshot,
        H7OwnedTensorSnapshot,
        H7OwnedTensorSnapshot,
    ]
    kind: H7ActionKind
    dimension: Literal[2]
    group: Literal["GL+(2,R)"]
    representation: Literal["direct_gl_plus_2"]
    action_sha256: str

    @classmethod
    def create(
        cls,
        *,
        elements: tuple[
            H7OwnedTensorSnapshot | torch.Tensor,
            H7OwnedTensorSnapshot | torch.Tensor,
            H7OwnedTensorSnapshot | torch.Tensor,
        ],
        kind: H7ActionKind,
    ) -> "H7GLPlus2Action":
        owned = _owned_action_elements(elements, dimension=2)
        semantic = {
            "elements": owned,
            "kind": kind,
            "dimension": 2,
            "group": "GL+(2,R)",
            "representation": "direct_gl_plus_2",
        }
        return cls(
            **semantic,
            action_sha256=h7_owned_sha256("vfe4.h7.gl-plus-2-action.v1", semantic),
        )

    def __post_init__(self) -> None:
        if (
            self.kind not in ("diagonal_base", "internal_product")
            or self.dimension != 2
            or self.group != "GL+(2,R)"
            or self.representation != "direct_gl_plus_2"
        ):
            raise ValueError("invalid GL+(2) action declaration")
        elements = _owned_action_elements(self.elements, dimension=2)
        if self.kind == "diagonal_base" and not all(
            torch.equal(elements[0].value(), item.value()) for item in elements[1:]
        ):
            raise ValueError("diagonal-base action elements must be identical")
        semantic = {
            "elements": self.elements,
            "kind": self.kind,
            "dimension": self.dimension,
            "group": self.group,
            "representation": self.representation,
        }
        if self.action_sha256 != h7_owned_sha256(
            "vfe4.h7.gl-plus-2-action.v1", semantic
        ):
            raise ValueError("matrix action_sha256 does not match action")


H7TensorActionSnapshot: TypeAlias = H7ScalarReplayAction | H7GLPlus2Action


def _action_profile_values(
    action: H7TensorActionSnapshot,
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    values: list[tuple[tuple[float, ...], ...]] = []
    for snapshot in action.elements:
        snapshot.assert_intact()
        rows = snapshot.value().tolist()
        values.append(
            tuple(
                tuple(float(item) for item in cast(list[float], row))
                for row in cast(list[list[float]], rows)
            )
        )
    return tuple(values)


@dataclass(frozen=True)
class H7TrialSpec(_H7IntegrityRecord):
    _integrity_field: ClassVar[str] = "trial_sha256"
    _hash_domain: ClassVar[str] = "vfe4.h7.trial-spec.v1"

    trial_id: H7TrialId
    role: H7TrialRole
    expected_predicate: H7ExpectedPredicate
    fixture_id: Literal["h1-v1", "h7-v1"]
    frame_profile: H7FrameProfile
    decoder_policy: H7DecoderPolicy
    action: H7TensorActionSnapshot
    action_sha256: str
    trial_sha256: str

    def __post_init__(self) -> None:
        if type(self.action) not in (H7ScalarReplayAction, H7GLPlus2Action):
            raise ValueError("trial action must use the closed H7 action union")
        if self.action_sha256 != self.action.action_sha256:
            raise ValueError("trial action_sha256 does not match action")
        contract = H7_TRIAL_CONTRACTS.get(self.trial_id)
        if contract is None:
            raise ValueError("trial_id is outside the frozen H7 inventory")
        (
            role,
            predicate,
            fixture_id,
            frame_profile,
            decoder_policy,
            action_profile,
        ) = contract
        if (
            self.role != role
            or self.expected_predicate != predicate
            or self.fixture_id != fixture_id
            or self.frame_profile != frame_profile
            or self.decoder_policy != decoder_policy
        ):
            raise ValueError("trial declaration disagrees with its frozen contract")
        scalar = self.trial_id in H7_SCALAR_TRIAL_IDS
        if scalar != (type(self.action) is H7ScalarReplayAction):
            raise ValueError("trial/action dimension family is inconsistent")
        expected_kind: H7ActionKind = (
            "diagonal_base"
            if action_profile
            in (
                "scalar_base",
                "matrix_diagonal",
                "matrix_fixed_decoder_stabilizer",
            )
            else "internal_product"
        )
        if self.action.kind != expected_kind:
            raise ValueError("trial action kind disagrees with its frozen profile")
        if (
            _action_profile_values(self.action)
            != _FROZEN_ACTION_PROFILE_VALUES[action_profile]
        ):
            raise ValueError("trial action bytes disagree with its frozen profile")
        super().__post_init__()


@dataclass(frozen=True)
class H7HistoryValueView:
    channel: H7Channel
    population_label: int
    value: H7BorrowedTensorView

    def __post_init__(self) -> None:
        if self.channel not in ("z", "m"):
            raise ValueError("history channel must be z or m")
        if type(self.population_label) is not int or self.population_label < 0:
            raise ValueError("history population_label must be nonnegative")
        self.assert_live()

    def assert_live(self) -> None:
        if type(self.value) is not H7BorrowedTensorView:
            raise ValueError("history value must be an exact borrowed view")
        self.value.assert_intact()

    def live_identity(self) -> Mapping[str, object]:
        self.assert_live()
        return MappingProxyType(
            {
                "channel": self.channel,
                "population_label": self.population_label,
                "value": _borrowed_tensor_semantic(self.value),
            }
        )


@dataclass(frozen=True)
class H7HistoryValueSnapshot(_H7IntegrityRecord):
    _integrity_field: ClassVar[str] = "history_sha256"
    _hash_domain: ClassVar[str] = "vfe4.h7.history-value.v1"

    channel: H7Channel
    population_label: int
    value: H7OwnedTensorSnapshot
    history_sha256: str

    def __post_init__(self) -> None:
        if self.channel not in ("z", "m"):
            raise ValueError("history channel must be z or m")
        if type(self.population_label) is not int or self.population_label < 0:
            raise ValueError("history population_label must be nonnegative")
        tensor = _require_owned_tensor(self.value, "history value", ndim=1)
        if tensor.numel() not in (1, 2):
            raise ValueError("history value must have scalar or matrix-fixture width")
        super().__post_init__()


@dataclass(frozen=True)
class H7SourceCovectorSnapshot(_H7IntegrityRecord):
    _integrity_field: ClassVar[str] = "covector_sha256"
    _hash_domain: ClassVar[str] = "vfe4.h7.source-covector.v1"

    bank: H7SourceBank
    channel: H7Channel
    receiver_t: int
    source_j: int
    value: H7OwnedTensorSnapshot
    covector_sha256: str

    def __post_init__(self) -> None:
        if self.bank not in ("model", "state"):
            raise ValueError("source covector bank must be model or state")
        if self.channel not in ("z", "m"):
            raise ValueError("source covector channel must be z or m")
        _require_receiver_source(
            self.receiver_t,
            self.source_j,
            "source covector",
            allow_initial=False,
        )
        tensor = _require_owned_tensor(self.value, "source covector value", ndim=1)
        if tensor.numel() not in (1, 2):
            raise ValueError("source covector must have scalar or matrix-fixture width")
        super().__post_init__()


@dataclass(frozen=True)
class H7SourceScorerRowView:
    bank: H7SourceBank
    receiver_t: int
    source_j: int
    prefix_tokens: tuple[int, ...]
    prefix_bytes: bytes
    prefix_bytes_sha256: str
    alpha_bias: float
    alpha_token_scale: float
    prefix_term: float
    z_history: tuple[H7HistoryValueView, ...]
    m_history: tuple[H7HistoryValueView, ...]
    z_covector: H7BorrowedTensorView
    m_covector: H7BorrowedTensorView
    mask: tuple[bool, ...]
    support: tuple[int, ...]
    raw_scores: H7BorrowedTensorView
    probabilities: H7BorrowedTensorView
    semantic_row_sha256: str

    @classmethod
    def create(
        cls,
        *,
        bank: H7SourceBank,
        receiver_t: int,
        source_j: int,
        prefix_tokens: tuple[int, ...],
        prefix_bytes: bytes,
        alpha_bias: float,
        alpha_token_scale: float,
        prefix_term: float,
        z_history: tuple[H7HistoryValueView, ...],
        m_history: tuple[H7HistoryValueView, ...],
        z_covector: H7BorrowedTensorView,
        m_covector: H7BorrowedTensorView,
        mask: tuple[bool, ...],
        support: tuple[int, ...],
        raw_scores: H7BorrowedTensorView,
        probabilities: H7BorrowedTensorView,
    ) -> "H7SourceScorerRowView":
        prefix_sha256 = hashlib.sha256(prefix_bytes).hexdigest()
        values: dict[str, object] = {
            "bank": bank,
            "receiver_t": receiver_t,
            "source_j": source_j,
            "prefix_tokens": prefix_tokens,
            "prefix_bytes": prefix_bytes,
            "prefix_bytes_sha256": prefix_sha256,
            "alpha_bias": alpha_bias,
            "alpha_token_scale": alpha_token_scale,
            "prefix_term": prefix_term,
            "z_history": z_history,
            "m_history": m_history,
            "z_covector": z_covector,
            "m_covector": m_covector,
            "mask": mask,
            "support": support,
            "raw_scores": raw_scores,
            "probabilities": probabilities,
        }
        digest = h7_owned_sha256(
            "vfe4.h7.source-scorer-row-view.v1",
            cls._live_semantic(values),
        )
        return cls(**values, semantic_row_sha256=digest)  # type: ignore[arg-type]

    @staticmethod
    def _live_semantic(values: Mapping[str, object]) -> Mapping[str, object]:
        z_history = cast(tuple[H7HistoryValueView, ...], values["z_history"])
        m_history = cast(tuple[H7HistoryValueView, ...], values["m_history"])
        return MappingProxyType(
            {
                "bank": values["bank"],
                "receiver_t": values["receiver_t"],
                "source_j": values["source_j"],
                "prefix_tokens": values["prefix_tokens"],
                "prefix_bytes": values["prefix_bytes"],
                "prefix_bytes_sha256": values["prefix_bytes_sha256"],
                "alpha_bias": values["alpha_bias"],
                "alpha_token_scale": values["alpha_token_scale"],
                "prefix_term": values["prefix_term"],
                "z_history": tuple(item.live_identity() for item in z_history),
                "m_history": tuple(item.live_identity() for item in m_history),
                "z_covector": _borrowed_tensor_semantic(
                    cast(H7BorrowedTensorView, values["z_covector"])
                ),
                "m_covector": _borrowed_tensor_semantic(
                    cast(H7BorrowedTensorView, values["m_covector"])
                ),
                "mask": values["mask"],
                "support": values["support"],
                "raw_scores": _borrowed_tensor_semantic(
                    cast(H7BorrowedTensorView, values["raw_scores"])
                ),
                "probabilities": _borrowed_tensor_semantic(
                    cast(H7BorrowedTensorView, values["probabilities"])
                ),
            }
        )

    def __post_init__(self) -> None:
        if self.bank not in ("model", "state"):
            raise ValueError("source scorer bank must be model or state")
        if (
            type(self.receiver_t) is not int
            or self.receiver_t not in (1, 2)
            or type(self.source_j) is not int
            or self.source_j < 0
            or self.source_j >= self.receiver_t
        ):
            raise ValueError("source scorer receiver/source pair is invalid")
        if (
            type(self.prefix_tokens) is not tuple
            or len(self.prefix_tokens) != self.receiver_t
            or any(type(item) is not int or item < 0 for item in self.prefix_tokens)
            or type(self.prefix_bytes) is not bytes
            or hashlib.sha256(self.prefix_bytes).hexdigest() != self.prefix_bytes_sha256
            or type(self.mask) is not tuple
            or type(self.support) is not tuple
            or len(self.mask) != len(self.support)
            or not self.support
            or any(type(item) is not bool for item in self.mask)
            or any(type(item) is not int or item < 0 for item in self.support)
        ):
            raise ValueError("source scorer structural identity is invalid")
        _require_sha256(self.semantic_row_sha256, "semantic_row_sha256")
        self.assert_live()

    def assert_live(self) -> None:
        values = {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name != "semantic_row_sha256"
        }
        expected = h7_owned_sha256(
            "vfe4.h7.source-scorer-row-view.v1",
            self._live_semantic(values),
        )
        if expected != self.semantic_row_sha256:
            raise ValueError("source scorer borrowed identity changed")


@dataclass(frozen=True)
class H7SourceContextView:
    prefix_tokens: tuple[int, ...]
    prefix_bytes: bytes
    prefix_bytes_sha256: str
    z_history: tuple[H7HistoryValueView, ...]
    m_history: tuple[H7HistoryValueView, ...]
    scorer_rows: tuple[H7SourceScorerRowView, ...]
    source_scorer_profile: Literal["h7-linear-history-source-v1"] | None
    semantic_context_sha256: str

    @classmethod
    def create(
        cls,
        *,
        prefix_tokens: tuple[int, ...],
        prefix_bytes: bytes,
        z_history: tuple[H7HistoryValueView, ...],
        m_history: tuple[H7HistoryValueView, ...],
        scorer_rows: tuple[H7SourceScorerRowView, ...],
        source_scorer_profile: Literal["h7-linear-history-source-v1"] | None,
    ) -> "H7SourceContextView":
        values: dict[str, object] = {
            "prefix_tokens": prefix_tokens,
            "prefix_bytes": prefix_bytes,
            "prefix_bytes_sha256": hashlib.sha256(prefix_bytes).hexdigest(),
            "z_history": z_history,
            "m_history": m_history,
            "scorer_rows": scorer_rows,
            "source_scorer_profile": source_scorer_profile,
        }
        digest = h7_owned_sha256(
            "vfe4.h7.source-context-view.v1",
            cls._live_semantic(values),
        )
        return cls(**values, semantic_context_sha256=digest)  # type: ignore[arg-type]

    @staticmethod
    def _live_semantic(values: Mapping[str, object]) -> Mapping[str, object]:
        z_history = cast(tuple[H7HistoryValueView, ...], values["z_history"])
        m_history = cast(tuple[H7HistoryValueView, ...], values["m_history"])
        scorer_rows = cast(tuple[H7SourceScorerRowView, ...], values["scorer_rows"])
        for row in scorer_rows:
            row.assert_live()
        return MappingProxyType(
            {
                "prefix_tokens": values["prefix_tokens"],
                "prefix_bytes": values["prefix_bytes"],
                "prefix_bytes_sha256": values["prefix_bytes_sha256"],
                "z_history": tuple(item.live_identity() for item in z_history),
                "m_history": tuple(item.live_identity() for item in m_history),
                "scorer_rows": tuple(row.semantic_row_sha256 for row in scorer_rows),
                "source_scorer_profile": values["source_scorer_profile"],
            }
        )

    def __post_init__(self) -> None:
        if (
            type(self.prefix_tokens) is not tuple
            or type(self.prefix_bytes) is not bytes
            or hashlib.sha256(self.prefix_bytes).hexdigest() != self.prefix_bytes_sha256
            or self.source_scorer_profile not in (None, "h7-linear-history-source-v1")
        ):
            raise ValueError("source context structural identity is invalid")
        _require_sha256(self.semantic_context_sha256, "semantic_context_sha256")
        self.assert_live()

    def assert_live(self) -> None:
        values = {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name != "semantic_context_sha256"
        }
        expected = h7_owned_sha256(
            "vfe4.h7.source-context-view.v1",
            self._live_semantic(values),
        )
        if expected != self.semantic_context_sha256:
            raise ValueError("source context borrowed identity changed")


@dataclass(frozen=True)
class H7SourceScorerRowSnapshot(_H7IntegrityRecord):
    _integrity_field: ClassVar[str] = "row_sha256"
    _hash_domain: ClassVar[str] = "vfe4.h7.source-scorer-row.v1"

    bank: H7SourceBank
    receiver_t: int
    source_j: int
    prefix_tokens: tuple[int, ...]
    prefix_bytes: bytes
    prefix_bytes_sha256: str
    alpha_bias: float
    alpha_token_scale: float
    prefix_term: float
    z_history: tuple[H7HistoryValueSnapshot, ...]
    m_history: tuple[H7HistoryValueSnapshot, ...]
    z_covector: H7SourceCovectorSnapshot
    m_covector: H7SourceCovectorSnapshot
    mask: tuple[bool, ...]
    support: tuple[int, ...]
    raw_scores: H7OwnedTensorSnapshot
    probabilities: H7OwnedTensorSnapshot
    source_row_raw_bytes: bytes
    row_raw_bytes_sha256: str
    row_sha256: str

    def __post_init__(self) -> None:
        if self.bank not in ("model", "state"):
            raise ValueError("source scorer bank must be model or state")
        _require_receiver_source(
            self.receiver_t,
            self.source_j,
            "source scorer",
            allow_initial=False,
        )
        if (
            type(self.prefix_tokens) is not tuple
            or len(self.prefix_tokens) != self.receiver_t
            or any(type(item) is not int or item < 0 for item in self.prefix_tokens)
            or type(self.prefix_bytes) is not bytes
            or self.prefix_bytes
            != json.dumps(self.prefix_tokens, separators=(",", ":")).encode("ascii")
            or hashlib.sha256(self.prefix_bytes).hexdigest() != self.prefix_bytes_sha256
        ):
            raise ValueError("source scorer prefix identity is invalid")
        weighted_prefix = sum(
            (index + 1) * (token + 1) for index, token in enumerate(self.prefix_tokens)
        )
        expected_prefix_term = (
            self.alpha_bias + self.alpha_token_scale * weighted_prefix
        )
        if self.prefix_term != expected_prefix_term:
            raise ValueError("source scorer prefix_term disagrees with its inputs")
        if (
            type(self.z_history) is not tuple
            or type(self.m_history) is not tuple
            or len(self.z_history) != 2
            or len(self.m_history) != 2
        ):
            raise ValueError("source scorer requires both two-row histories")
        history_dimension: int | None = None
        for channel, history in (("z", self.z_history), ("m", self.m_history)):
            for population_label, item in enumerate(history):
                if (
                    type(item) is not H7HistoryValueSnapshot
                    or item.channel != channel
                    or item.population_label != population_label
                ):
                    raise ValueError("source scorer history order is invalid")
                tensor = _require_owned_tensor(
                    item.value,
                    f"{channel}_history[{population_label}]",
                    ndim=1,
                )
                if history_dimension is None:
                    history_dimension = tensor.numel()
                elif tensor.numel() != history_dimension:
                    raise ValueError("source scorer history widths disagree")
        if history_dimension not in (1, 2):
            raise ValueError("source scorer history width is invalid")
        for channel, covector in (
            ("z", self.z_covector),
            ("m", self.m_covector),
        ):
            if (
                type(covector) is not H7SourceCovectorSnapshot
                or covector.bank != self.bank
                or covector.channel != channel
                or covector.receiver_t != self.receiver_t
                or covector.source_j != self.source_j
            ):
                raise ValueError("source scorer covector identity is invalid")
            _require_owned_tensor(
                covector.value,
                f"{channel}_covector",
                shape=(cast(int, history_dimension),),
            )
        if (
            type(self.mask) is not tuple
            or self.mask != (True,)
            or type(self.support) is not tuple
            or self.support != (self.source_j,)
        ):
            raise ValueError("source scorer support/mask inventory is invalid")
        raw_scores = _require_owned_tensor(
            self.raw_scores, "source scorer raw_scores", shape=(1,)
        )
        probabilities = _require_owned_tensor(
            self.probabilities,
            "source scorer probabilities",
            shape=(1,),
        )
        if not torch.equal(
            probabilities,
            torch.ones(1, dtype=torch.float64, device=probabilities.device),
        ):
            raise ValueError("one-point source scorer probability must be exactly one")
        with torch.no_grad():
            expected_score = (
                self.prefix_term
                + float(
                    self.z_covector.value.value()
                    @ self.z_history[self.source_j].value.value()
                )
                + float(
                    self.m_covector.value.value()
                    @ self.m_history[self.source_j].value.value()
                )
            )
        observed_score = float(raw_scores[0])
        if not math.isclose(
            observed_score,
            expected_score,
            rel_tol=32.0 * torch.finfo(torch.float64).eps,
            abs_tol=32.0 * torch.finfo(torch.float64).eps,
        ):
            raise ValueError("source scorer raw score disagrees with its law")
        expected_row_bytes = canonical_h7_bytes(
            {
                "bank": self.bank,
                "receiver_t": self.receiver_t,
                "source_j": self.source_j,
                "prefix_tokens": self.prefix_tokens,
                "prefix_term": self.prefix_term,
                "z_covector_sha256": self.z_covector.covector_sha256,
                "m_covector_sha256": self.m_covector.covector_sha256,
                "raw_score": observed_score,
                "support": self.support,
            }
        )
        if (
            type(self.source_row_raw_bytes) is not bytes
            or self.source_row_raw_bytes != expected_row_bytes
            or hashlib.sha256(self.source_row_raw_bytes).hexdigest()
            != self.row_raw_bytes_sha256
        ):
            raise ValueError("source scorer raw row bytes are invalid")
        super().__post_init__()


@dataclass(frozen=True)
class H7SourceContextSnapshot(_H7IntegrityRecord):
    _integrity_field: ClassVar[str] = "context_sha256"
    _hash_domain: ClassVar[str] = "vfe4.h7.source-context.v1"

    prefix_tokens: tuple[int, ...]
    prefix_bytes: bytes
    prefix_bytes_sha256: str
    z_history: tuple[H7HistoryValueSnapshot, ...]
    m_history: tuple[H7HistoryValueSnapshot, ...]
    scorer_rows: tuple[H7SourceScorerRowSnapshot, ...]
    source_scorer_profile: Literal["h7-linear-history-source-v1"] | None
    source_scorer_sha256: str | None
    context_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.prefix_tokens) is not tuple
            or len(self.prefix_tokens) != 2
            or any(type(item) is not int or item < 0 for item in self.prefix_tokens)
            or type(self.prefix_bytes) is not bytes
            or self.prefix_bytes
            != json.dumps(self.prefix_tokens, separators=(",", ":")).encode("ascii")
            or hashlib.sha256(self.prefix_bytes).hexdigest() != self.prefix_bytes_sha256
        ):
            raise ValueError("source context prefix identity is invalid")
        if (
            type(self.z_history) is not tuple
            or type(self.m_history) is not tuple
            or tuple(
                (item.channel, item.population_label)
                for item in (*self.z_history, *self.m_history)
                if type(item) is H7HistoryValueSnapshot
            )
            != (("z", 0), ("z", 1), ("m", 0), ("m", 1))
            or len(self.z_history) != 2
            or len(self.m_history) != 2
        ):
            raise ValueError("source context history inventory is invalid")
        if type(self.scorer_rows) is not tuple or any(
            type(item) is not H7SourceScorerRowSnapshot for item in self.scorer_rows
        ):
            raise ValueError("source context scorer rows must be exact records")
        if self.source_scorer_profile is None:
            if self.scorer_rows or self.source_scorer_sha256 is not None:
                raise ValueError("no-scorer context cannot carry scorer evidence")
        else:
            if (
                self.source_scorer_profile != "h7-linear-history-source-v1"
                or self.source_scorer_sha256 is None
                or tuple(
                    (row.bank, row.receiver_t, row.source_j) for row in self.scorer_rows
                )
                != (
                    ("model", 1, 0),
                    ("model", 2, 1),
                    ("state", 1, 0),
                    ("state", 2, 1),
                )
            ):
                raise ValueError("source context scorer inventory is invalid")
            for row in self.scorer_rows:
                if (
                    row.prefix_tokens != self.prefix_tokens[: row.receiver_t]
                    or tuple(item.history_sha256 for item in row.z_history)
                    != tuple(item.history_sha256 for item in self.z_history)
                    or tuple(item.history_sha256 for item in row.m_history)
                    != tuple(item.history_sha256 for item in self.m_history)
                ):
                    raise ValueError("source context row provenance is inconsistent")
            expected_scorer_sha256 = h7_owned_sha256(
                "vfe4.h7.source-scorer.v1",
                tuple(row.row_sha256 for row in self.scorer_rows),
            )
            if self.source_scorer_sha256 != expected_scorer_sha256:
                raise ValueError("source_scorer_sha256 does not match scorer rows")
        super().__post_init__()


@dataclass(frozen=True)
class H7RecognitionContextSnapshot(_H7IntegrityRecord):
    _integrity_field: ClassVar[str] = "context_sha256"
    _hash_domain: ClassVar[str] = "vfe4.h7.recognition-context.v1"

    observation_labels: tuple[int, ...]
    conditioning: Literal["filtering", "smoothing"]
    context_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.observation_labels) is not tuple
            or len(self.observation_labels) != 2
            or any(
                type(item) is not int or item < 0 for item in self.observation_labels
            )
            or self.observation_labels[0] >= self.observation_labels[1]
            or self.conditioning not in ("filtering", "smoothing")
        ):
            raise ValueError("recognition context is invalid")
        super().__post_init__()


H7_H1_SOURCE_PATHS: tuple[
    tuple[
        tuple[int, int],
        tuple[int, int],
        tuple[int, int],
        tuple[int, int],
    ],
    ...,
] = (
    ((0, 0), (0, 0), (0, 0), (0, 0)),
    ((0, 1), (0, 0), (0, 0), (0, 1)),
    ((0, 0), (0, 1), (0, 1), (0, 2)),
    ((0, 1), (0, 1), (0, 1), (0, 3)),
)


@dataclass(frozen=True)
class H7ScalarSourcePathSnapshot(_H7IntegrityRecord):
    _integrity_field: ClassVar[str] = "path_sha256"
    _hash_domain: ClassVar[str] = "vfe4.h7.scalar-source-path.v1"

    path_id: str
    a: tuple[int, int]
    b: tuple[int, int]
    model_kernel_selectors: tuple[int, int]
    state_kernel_selectors: tuple[int, int]
    observation_label_base: Literal[1]
    observation_labels: tuple[int, int]
    decoder_row_indices: tuple[int, int]
    path_sha256: str

    def __post_init__(self) -> None:
        _require_nonempty(self.path_id, "path_id")
        if (
            type(self.a) is not tuple
            or type(self.b) is not tuple
            or type(self.model_kernel_selectors) is not tuple
            or type(self.state_kernel_selectors) is not tuple
            or type(self.observation_labels) is not tuple
            or type(self.decoder_row_indices) is not tuple
            or any(
                len(value) != 2
                or any(type(item) is not int or item < 0 for item in value)
                for value in (
                    self.a,
                    self.b,
                    self.model_kernel_selectors,
                    self.state_kernel_selectors,
                    self.observation_labels,
                    self.decoder_row_indices,
                )
            )
            or self.observation_label_base != 1
        ):
            raise ValueError("invalid scalar source-path coordinates")
        if self.observation_labels != (1, 2):
            raise ValueError("H1 observation labels must remain one-based (1, 2)")
        if self.decoder_row_indices != (0, 1):
            raise ValueError("H1 decoder selectors must match one-based labels")
        try:
            index = tuple(
                (a, b, model, state) for a, b, model, state in H7_H1_SOURCE_PATHS
            ).index(
                (
                    self.a,
                    self.b,
                    self.model_kernel_selectors,
                    self.state_kernel_selectors,
                )
            )
        except ValueError as error:
            raise ValueError(
                "scalar path is outside the frozen H1 inventory"
            ) from error
        expected_path_id = f"h1-path-{index}:a{self.a[1]}-b{self.b[1]}"
        if self.path_id != expected_path_id:
            raise ValueError("scalar path_id disagrees with its frozen coordinates")
        super().__post_init__()


def _require_probability_snapshot(
    value: H7OwnedTensorSnapshot,
    *,
    shape: tuple[int, ...],
    name: str,
    rowwise: bool,
) -> None:
    if type(value) is not H7OwnedTensorSnapshot:
        raise ValueError(f"{name} must be an exact owned tensor snapshot")
    value.assert_intact()
    tensor = value.value()
    if tensor.dtype != torch.float64 or tuple(tensor.shape) != shape:
        raise ValueError(f"{name} has the wrong dtype or shape")
    if bool((tensor < 0.0).any().item()):
        raise ValueError(f"{name} contains a negative probability")
    sums = tensor.sum(dim=-1) if rowwise else tensor.sum()
    ones = torch.ones_like(sums)
    if not bool(torch.allclose(sums, ones, rtol=0.0, atol=1e-15)):
        raise ValueError(f"{name} is not normalized")


def _require_h1_ordered_paths(
    paths: tuple[
        H7ScalarSourcePathSnapshot,
        H7ScalarSourcePathSnapshot,
        H7ScalarSourcePathSnapshot,
        H7ScalarSourcePathSnapshot,
    ],
) -> None:
    if (
        type(paths) is not tuple
        or len(paths) != 4
        or any(type(item) is not H7ScalarSourcePathSnapshot for item in paths)
    ):
        raise ValueError("H1 source law requires exactly four ordered paths")
    observed = tuple(
        (
            item.a,
            item.b,
            item.model_kernel_selectors,
            item.state_kernel_selectors,
        )
        for item in paths
    )
    if observed != H7_H1_SOURCE_PATHS:
        raise ValueError("H1 source paths are missing, duplicated, or reordered")


@dataclass(frozen=True)
class H7ScalarGenerativeSourceLawSnapshot(_H7IntegrityRecord):
    _integrity_field: ClassVar[str] = "source_law_sha256"
    _hash_domain: ClassVar[str] = "vfe4.h7.scalar-generative-source-law.v1"

    model_source_priors: tuple[H7OwnedTensorSnapshot, H7OwnedTensorSnapshot]
    state_source_priors: tuple[H7OwnedTensorSnapshot, H7OwnedTensorSnapshot]
    ordered_paths: tuple[
        H7ScalarSourcePathSnapshot,
        H7ScalarSourcePathSnapshot,
        H7ScalarSourcePathSnapshot,
        H7ScalarSourcePathSnapshot,
    ]
    source_law_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.model_source_priors) is not tuple
            or len(self.model_source_priors) != 2
            or type(self.state_source_priors) is not tuple
            or len(self.state_source_priors) != 2
        ):
            raise ValueError("scalar generative priors require two time rows")
        for bank_name, rows in (
            ("model_source_priors", self.model_source_priors),
            ("state_source_priors", self.state_source_priors),
        ):
            _require_probability_snapshot(
                rows[0], shape=(1,), name=f"{bank_name}[0]", rowwise=False
            )
            _require_probability_snapshot(
                rows[1], shape=(2,), name=f"{bank_name}[1]", rowwise=False
            )
        _require_h1_ordered_paths(self.ordered_paths)
        super().__post_init__()


@dataclass(frozen=True)
class H7ScalarRecognitionSourceLawSnapshot(_H7IntegrityRecord):
    _integrity_field: ClassVar[str] = "source_law_sha256"
    _hash_domain: ClassVar[str] = "vfe4.h7.scalar-recognition-source-law.v1"

    model_source_probabilities: tuple[H7OwnedTensorSnapshot, H7OwnedTensorSnapshot]
    state_source_probabilities_given_model_source: tuple[
        H7OwnedTensorSnapshot, H7OwnedTensorSnapshot
    ]
    ordered_paths: tuple[
        H7ScalarSourcePathSnapshot,
        H7ScalarSourcePathSnapshot,
        H7ScalarSourcePathSnapshot,
        H7ScalarSourcePathSnapshot,
    ]
    source_law_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.model_source_probabilities) is not tuple
            or len(self.model_source_probabilities) != 2
            or type(self.state_source_probabilities_given_model_source) is not tuple
            or len(self.state_source_probabilities_given_model_source) != 2
        ):
            raise ValueError("scalar recognition probabilities require two times")
        _require_probability_snapshot(
            self.model_source_probabilities[0],
            shape=(1,),
            name="model_source_probabilities[0]",
            rowwise=False,
        )
        _require_probability_snapshot(
            self.model_source_probabilities[1],
            shape=(2,),
            name="model_source_probabilities[1]",
            rowwise=False,
        )
        _require_probability_snapshot(
            self.state_source_probabilities_given_model_source[0],
            shape=(1, 1),
            name="state_source_probabilities_given_model_source[0]",
            rowwise=True,
        )
        _require_probability_snapshot(
            self.state_source_probabilities_given_model_source[1],
            shape=(2, 2),
            name="state_source_probabilities_given_model_source[1]",
            rowwise=True,
        )
        _require_h1_ordered_paths(self.ordered_paths)
        super().__post_init__()


@dataclass(frozen=True)
class H7GaussianComponentSnapshot(_H7IntegrityRecord):
    _integrity_field: ClassVar[str] = "component_sha256"
    _hash_domain: ClassVar[str] = "vfe4.h7.gaussian-component.v1"

    component_id: str
    receiver_t: int | None
    source_j: int | None
    mean: H7OwnedTensorSnapshot
    covariance: H7OwnedTensorSnapshot
    precision: H7OwnedTensorSnapshot
    information_vector: H7OwnedTensorSnapshot
    second_moment: H7OwnedTensorSnapshot
    component_sha256: str

    def __post_init__(self) -> None:
        _require_nonempty(self.component_id, "component_id")
        _require_receiver_source(
            self.receiver_t,
            self.source_j,
            "Gaussian component",
            allow_initial=True,
        )
        mean = _require_owned_tensor(self.mean, "Gaussian mean", ndim=1)
        dimension = mean.numel()
        if dimension <= 0:
            raise ValueError("Gaussian component dimension must be positive")
        matrix_shape = (dimension, dimension)
        covariance = _require_owned_tensor(
            self.covariance,
            "Gaussian covariance",
            shape=matrix_shape,
        )
        precision = _require_owned_tensor(
            self.precision,
            "Gaussian precision",
            shape=matrix_shape,
        )
        information = _require_owned_tensor(
            self.information_vector,
            "Gaussian information vector",
            shape=(dimension,),
        )
        second_moment = _require_owned_tensor(
            self.second_moment,
            "Gaussian second moment",
            shape=matrix_shape,
        )
        with torch.no_grad():
            eps = torch.finfo(torch.float64).eps
            for name, matrix in (
                ("covariance", covariance),
                ("precision", precision),
                ("second_moment", second_moment),
            ):
                scale = max(1.0, float(matrix.abs().max()))
                if not torch.allclose(
                    matrix,
                    matrix.T,
                    rtol=256.0 * eps,
                    atol=256.0 * eps * scale,
                ):
                    raise ValueError(f"Gaussian {name} must be symmetric")
            if bool(
                torch.any(torch.linalg.cholesky_ex(covariance).info != 0).item()
            ) or bool(torch.any(torch.linalg.cholesky_ex(precision).info != 0).item()):
                raise ValueError("Gaussian covariance and precision must be SPD")
            identity = torch.eye(
                dimension,
                dtype=torch.float64,
                device=covariance.device,
            )
            relationships = (
                (precision @ covariance, identity, "precision/covariance"),
                (
                    precision @ mean,
                    information,
                    "information vector",
                ),
                (
                    covariance + torch.outer(mean, mean),
                    second_moment,
                    "second moment",
                ),
            )
            for observed, expected, name in relationships:
                scale = max(
                    1.0,
                    float(observed.abs().max()),
                    float(expected.abs().max()),
                )
                if not torch.allclose(
                    observed,
                    expected,
                    rtol=256.0 * eps,
                    atol=256.0 * eps * scale,
                ):
                    raise ValueError(f"Gaussian {name} identity is inconsistent")
        super().__post_init__()


def _require_task5_precision_matrix(
    snapshot: H7OwnedTensorSnapshot,
    name: str,
) -> torch.Tensor:
    matrix = _require_owned_tensor(snapshot, name, ndim=2)
    if (
        snapshot.device != "cpu"
        or matrix.shape[0] <= 0
        or matrix.shape[0] != matrix.shape[1]
    ):
        raise ValueError(f"{name} must be a nonempty square float64 CPU matrix")
    eps = torch.finfo(torch.float64).eps
    with torch.no_grad():
        scale = max(1.0, float(matrix.abs().max()))
        if not torch.allclose(
            matrix,
            matrix.T,
            rtol=256.0 * eps,
            atol=256.0 * eps * scale,
        ):
            raise ValueError(f"{name} must be symmetric")
        if bool(torch.any(torch.linalg.cholesky_ex(matrix).info != 0).item()):
            raise ValueError(f"{name} must be positive definite")
    return matrix


def _require_task5_precision_pair(
    covariance_snapshot: H7OwnedTensorSnapshot,
    precision_snapshot: H7OwnedTensorSnapshot,
) -> None:
    covariance = _require_task5_precision_matrix(
        covariance_snapshot,
        "Task-5 covariance",
    )
    precision = _require_task5_precision_matrix(
        precision_snapshot,
        "Task-5 precision",
    )
    if covariance.shape != precision.shape:
        raise ValueError("Task-5 covariance/precision shapes disagree")
    identity = torch.eye(
        covariance.shape[0],
        dtype=torch.float64,
        device=covariance.device,
    )
    eps = torch.finfo(torch.float64).eps
    with torch.no_grad():
        for observed in (precision @ covariance, covariance @ precision):
            scale = max(1.0, float(observed.abs().max()))
            if not torch.allclose(
                observed,
                identity,
                rtol=256.0 * eps,
                atol=256.0 * eps * scale,
            ):
                raise ValueError(
                    "Task-5 precision is not the two-sided inverse of covariance"
                )


def _task5_precision_gaussian_ids(
    *,
    fixture_id: Literal["h1-v1", "h7-v1"],
    recognition_family: H7RecognitionFamily,
) -> tuple[str, ...]:
    if fixture_id == "h1-v1":
        if recognition_family != "structured_full_block":
            raise ValueError("scalar precision capture has the wrong recognition family")
        return (
            "scalar.p.initial_joint",
            "scalar.q.initial_joint",
            "scalar.p.p.model.receiver_1.source_0.receiver_offset",
            "scalar.p.p.state.receiver_1.source_0.receiver_offset",
            "scalar.p.p.model.receiver_2.source_0.receiver_offset",
            "scalar.p.p.state.receiver_2.source_0.receiver_offset",
            "scalar.p.p.model.receiver_2.source_1.receiver_offset",
            "scalar.p.p.state.receiver_2.source_1.receiver_offset",
            "scalar.q_model.q.model.receiver_1.source_0.receiver_offset",
            "scalar.q_model.q.model.receiver_2.source_0.receiver_offset",
            "scalar.q_model.q.model.receiver_2.source_1.receiver_offset",
            "scalar.q_state.q.state.receiver_1.a_0.b_0.receiver_offset",
            "scalar.q_state.q.state.receiver_2.a_0.b_0.receiver_offset",
            "scalar.q_state.q.state.receiver_2.a_1.b_0.receiver_offset",
            "scalar.q_state.q.state.receiver_2.a_0.b_1.receiver_offset",
            "scalar.q_state.q.state.receiver_2.a_1.b_1.receiver_offset",
            "scalar.q.global[h1-path-0:a0-b0]",
            "scalar.q.global[h1-path-1:a1-b0]",
            "scalar.q.global[h1-path-2:a0-b1]",
            "scalar.q.global[h1-path-3:a1-b1]",
            "scalar.p.global[h1-path-0:a0-b0]",
            "scalar.p.global[h1-path-1:a1-b0]",
            "scalar.p.global[h1-path-2:a0-b1]",
            "scalar.p.global[h1-path-3:a1-b1]",
        )
    prefix = (
        "structured"
        if recognition_family == "structured_full_block"
        else "factorized"
    )
    return (
        f"{prefix}.p.initial_joint",
        f"{prefix}.q.initial_joint",
        f"{prefix}.p.p.model.receiver_1.receiver_offset",
        f"{prefix}.p.p.state.receiver_1.receiver_offset",
        f"{prefix}.p.p.model.receiver_2.receiver_offset",
        f"{prefix}.p.p.state.receiver_2.receiver_offset",
        f"{prefix}.q_model.q.{prefix}.model.receiver_1.receiver_offset",
        f"{prefix}.q_model.q.{prefix}.model.receiver_2.receiver_offset",
        f"{prefix}.q_state.q.{prefix}.state.receiver_1.receiver_offset",
        f"{prefix}.q_state.q.{prefix}.state.receiver_2.receiver_offset",
        f"{prefix}.q.global[matrix-singleton-path]",
        f"{prefix}.p.global[matrix-singleton-path]",
    )


@dataclass(frozen=True)
class H7InjectedGlobalPrecisionSnapshot(_H7IntegrityRecord):
    _integrity_field: ClassVar[str] = "input_sha256"
    _hash_domain: ClassVar[str] = "vfe4.h7.injected-global-precision.v1"

    trial_id: H7TrialId
    gaussian_id: str
    covariance_snapshot_sha256: str
    precision: H7OwnedTensorSnapshot
    input_sha256: str

    def __post_init__(self) -> None:
        if self.trial_id not in H7_REQUIRED_TRIAL_IDS:
            raise ValueError("injected precision trial_id is outside H7")
        _require_nonempty(self.gaussian_id, "gaussian_id")
        if ".global[" not in self.gaussian_id or not self.gaussian_id.endswith("]"):
            raise ValueError("injected precision gaussian_id must select a global law")
        _require_sha256(
            self.covariance_snapshot_sha256,
            "covariance_snapshot_sha256",
        )
        _require_task5_precision_matrix(
            self.precision,
            "injected Task-5 precision",
        )
        super().__post_init__()


@dataclass(frozen=True)
class H7Task5PrecisionOperandSnapshot(_H7IntegrityRecord):
    _integrity_field: ClassVar[str] = "operand_sha256"
    _hash_domain: ClassVar[str] = "vfe4.h7.task5-precision-operand.v1"

    trial_id: H7TrialId
    batch_index: int
    gaussian_id: str
    source_kind: Literal["owned_component", "injected_global"]
    covariance: H7OwnedTensorSnapshot
    precision: H7OwnedTensorSnapshot
    operand_sha256: str

    def __post_init__(self) -> None:
        if (
            self.trial_id not in H7_REQUIRED_TRIAL_IDS
            or type(self.batch_index) is not int
            or self.batch_index < 0
            or self.source_kind not in ("owned_component", "injected_global")
        ):
            raise ValueError("Task-5 precision operand identity is invalid")
        _require_nonempty(self.gaussian_id, "gaussian_id")
        _require_task5_precision_pair(self.covariance, self.precision)
        super().__post_init__()


@dataclass(frozen=True)
class H7Task5PrecisionCaptureBatch(_H7IntegrityRecord):
    _integrity_field: ClassVar[str] = "capture_sha256"
    _hash_domain: ClassVar[str] = "vfe4.h7.task5-precision-capture-batch.v1"

    trial_id: H7TrialId
    fixture_id: Literal["h1-v1", "h7-v1"]
    raw_fixture_sha256: str
    recognition_family: H7RecognitionFamily
    operands: tuple[H7Task5PrecisionOperandSnapshot, ...]
    capture_sha256: str

    def __post_init__(self) -> None:
        contract = H7_TRIAL_CONTRACTS.get(self.trial_id)
        if (
            contract is None
            or self.fixture_id != contract[2]
            or self.recognition_family
            not in (
                "structured_full_block",
                "factorized_diagonal_within_fiber",
            )
        ):
            raise ValueError("Task-5 precision capture identity is invalid")
        _require_sha256(self.raw_fixture_sha256, "raw_fixture_sha256")
        expected_ids = _task5_precision_gaussian_ids(
            fixture_id=self.fixture_id,
            recognition_family=self.recognition_family,
        )
        owned_count = 16 if self.fixture_id == "h1-v1" else 10
        if (
            type(self.operands) is not tuple
            or len(self.operands) != len(expected_ids)
            or any(
                type(item) is not H7Task5PrecisionOperandSnapshot
                for item in self.operands
            )
            or tuple(item.trial_id for item in self.operands)
            != (self.trial_id,) * len(expected_ids)
            or tuple(item.batch_index for item in self.operands)
            != tuple(range(len(expected_ids)))
            or tuple(item.gaussian_id for item in self.operands) != expected_ids
            or tuple(item.source_kind for item in self.operands)
            != (
                *("owned_component" for _ in range(owned_count)),
                *(
                    "injected_global"
                    for _ in range(len(expected_ids) - owned_count)
                ),
            )
        ):
            raise ValueError(
                "Task-5 precision capture changed operand identity/order/cardinality"
            )
        for operand in self.operands:
            operand.__post_init__()
        super().__post_init__()


@dataclass(frozen=True)
class H7AffineComponentSnapshot(_H7IntegrityRecord):
    _integrity_field: ClassVar[str] = "component_sha256"
    _hash_domain: ClassVar[str] = "vfe4.h7.affine-component.v1"

    component_id: str
    bank: H7SourceBank
    receiver_t: int
    source_j: int
    parent_map: H7OwnedTensorSnapshot
    same_receiver_model_map: H7OwnedTensorSnapshot | None
    offset: H7OwnedTensorSnapshot
    receiver_law: H7GaussianComponentSnapshot
    component_sha256: str

    def __post_init__(self) -> None:
        _require_nonempty(self.component_id, "component_id")
        if self.bank not in ("model", "state"):
            raise ValueError("affine component bank must be model or state")
        _require_receiver_source(
            self.receiver_t,
            self.source_j,
            "affine component",
            allow_initial=False,
        )
        offset = _require_owned_tensor(self.offset, "affine offset", ndim=1)
        dimension = offset.numel()
        if dimension <= 0:
            raise ValueError("affine component dimension must be positive")
        _require_owned_tensor(
            self.parent_map,
            "affine parent_map",
            shape=(dimension, dimension),
        )
        if self.bank == "model":
            if self.same_receiver_model_map is not None:
                raise ValueError("model affine component cannot carry a model map")
        else:
            _require_owned_tensor(
                self.same_receiver_model_map,
                "state same_receiver_model_map",
                shape=(dimension, dimension),
            )
        if (
            type(self.receiver_law) is not H7GaussianComponentSnapshot
            or self.receiver_law.component_id != f"{self.component_id}.receiver"
            or self.receiver_law.receiver_t != self.receiver_t
            or self.receiver_law.source_j != self.source_j
            or not torch.equal(self.receiver_law.mean.value(), offset)
            or self.receiver_law.mean.shape != (dimension,)
        ):
            raise ValueError("affine receiver law provenance is inconsistent")
        super().__post_init__()


@dataclass(frozen=True)
class H7TensorLawComponent:
    component_id: str
    receiver_t: int | None
    source_j: int | None
    tensors: Mapping[str, H7BorrowedTensorView]
    component_identity_sha256: str

    @classmethod
    def create(
        cls,
        *,
        component_id: str,
        receiver_t: int | None,
        source_j: int | None,
        tensors: Mapping[str, H7BorrowedTensorView],
    ) -> "H7TensorLawComponent":
        frozen = cast(Mapping[str, H7BorrowedTensorView], _freeze_mapping(tensors))
        semantic = {
            "component_id": component_id,
            "receiver_t": receiver_t,
            "source_j": source_j,
            "tensors": _borrowed_tensor_mapping_semantic(frozen),
        }
        return cls(
            component_id=component_id,
            receiver_t=receiver_t,
            source_j=source_j,
            tensors=frozen,
            component_identity_sha256=h7_owned_sha256(
                "vfe4.h7.tensor-law-component-live.v1", semantic
            ),
        )

    def __post_init__(self) -> None:
        _require_nonempty(self.component_id, "component_id")
        if self.receiver_t is not None and (
            type(self.receiver_t) is not int or self.receiver_t < 0
        ):
            raise ValueError("receiver_t must be nonnegative or None")
        if self.source_j is not None and (
            type(self.source_j) is not int or self.source_j < 0
        ):
            raise ValueError("source_j must be nonnegative or None")
        frozen = _freeze_mapping(self.tensors)
        if frozen is not self.tensors:
            object.__setattr__(self, "tensors", frozen)
        _require_sha256(self.component_identity_sha256, "component_identity_sha256")
        self.assert_live()

    def assert_live(self) -> None:
        semantic = {
            "component_id": self.component_id,
            "receiver_t": self.receiver_t,
            "source_j": self.source_j,
            "tensors": _borrowed_tensor_mapping_semantic(self.tensors),
        }
        expected = h7_owned_sha256("vfe4.h7.tensor-law-component-live.v1", semantic)
        if expected != self.component_identity_sha256:
            raise ValueError("tensor-law component live identity changed")


_JACOBIAN_RECEIVER_SCOPE_ORDER = (
    ("model", 1),
    ("state", 1),
    ("model", 2),
    ("state", 2),
)


def _jacobian_receiver_scope(component_id: str) -> tuple[str, int]:
    _require_nonempty(component_id, "Jacobian receiver component ID")
    fields = component_id.split(".")
    bank_locations = tuple(
        (index, field)
        for index, field in enumerate(fields)
        if field in ("model", "state")
    )
    if len(bank_locations) != 1:
        raise ValueError("Jacobian receiver component ID has an ambiguous bank")
    bank_index, bank = bank_locations[0]
    if bank_index + 1 >= len(fields):
        raise ValueError("Jacobian receiver component ID lacks a receiver")
    receiver_label = fields[bank_index + 1]
    if receiver_label.startswith("receiver_"):
        receiver_label = receiver_label.removeprefix("receiver_")
    else:
        receiver_label = receiver_label.split("<-", maxsplit=1)[0]
    try:
        receiver_t = int(receiver_label)
    except ValueError as error:
        raise ValueError(
            "Jacobian receiver component ID has a malformed receiver"
        ) from error
    if receiver_t not in (1, 2):
        raise ValueError("Jacobian receiver component ID is outside H7")
    return bank, receiver_t


def _jacobian_grouped_local_total(
    initial: torch.Tensor,
    receiver_values: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    if (
        not isinstance(initial, torch.Tensor)
        or initial.shape != ()
        or not isinstance(receiver_values, Mapping)
        or not receiver_values
    ):
        raise ValueError("Jacobian local scopes must be nonempty scalar tensors")
    grouped: dict[tuple[str, int], torch.Tensor] = {}
    for component_id, value in receiver_values.items():
        if not isinstance(value, torch.Tensor) or value.shape != ():
            raise ValueError("Jacobian receiver scopes must be scalar tensors")
        scope = _jacobian_receiver_scope(component_id)
        previous = grouped.get(scope)
        if previous is not None and not torch.equal(previous, value):
            raise ValueError(
                "duplicate Jacobian receiver scopes disagree within a source "
                "family"
            )
        grouped[scope] = value
    if set(grouped) != set(_JACOBIAN_RECEIVER_SCOPE_ORDER):
        raise ValueError(
            "Jacobian receiver scopes must cover model/state at receivers 1/2"
        )
    return initial + torch.stack(
        tuple(grouped[scope] for scope in _JACOBIAN_RECEIVER_SCOPE_ORDER)
    ).sum()


@dataclass(frozen=True)
class H7JacobianMetadataView:
    """Live per-scope measure shifts retained on the autograd graph."""

    scope: Literal["generative", "recognition"]
    initial_logabsdet: H7BorrowedTensorView
    receiver_logabsdet: Mapping[str, H7BorrowedTensorView]
    global_logabsdet: H7BorrowedTensorView
    entropy_shift: H7BorrowedTensorView | None
    semantic_sha256: str

    @classmethod
    def create(
        cls,
        *,
        scope: Literal["generative", "recognition"],
        initial_logabsdet: H7BorrowedTensorView,
        receiver_logabsdet: Mapping[str, H7BorrowedTensorView],
        global_logabsdet: H7BorrowedTensorView,
        entropy_shift: H7BorrowedTensorView | None,
    ) -> "H7JacobianMetadataView":
        frozen_receivers = cast(
            Mapping[str, H7BorrowedTensorView],
            _freeze_mapping(receiver_logabsdet),
        )
        values: dict[str, object] = {
            "scope": scope,
            "initial_logabsdet": initial_logabsdet,
            "receiver_logabsdet": frozen_receivers,
            "global_logabsdet": global_logabsdet,
            "entropy_shift": entropy_shift,
        }
        return cls(
            scope=scope,
            initial_logabsdet=initial_logabsdet,
            receiver_logabsdet=frozen_receivers,
            global_logabsdet=global_logabsdet,
            entropy_shift=entropy_shift,
            semantic_sha256=h7_owned_sha256(
                "vfe4.h7.jacobian-metadata-view.v1",
                cls._live_semantic(values),
            ),
        )

    @staticmethod
    def _live_semantic(values: Mapping[str, object]) -> Mapping[str, object]:
        scope = values["scope"]
        if scope not in ("generative", "recognition"):
            raise ValueError("Jacobian metadata scope is invalid")
        initial = cast(H7BorrowedTensorView, values["initial_logabsdet"])
        receivers = cast(
            Mapping[str, H7BorrowedTensorView],
            values["receiver_logabsdet"],
        )
        global_shift = cast(H7BorrowedTensorView, values["global_logabsdet"])
        entropy = cast(H7BorrowedTensorView | None, values["entropy_shift"])
        if not isinstance(receivers, Mapping) or not receivers:
            raise ValueError("Jacobian metadata requires receiver scopes")
        for name, view in (
            ("initial_logabsdet", initial),
            ("global_logabsdet", global_shift),
            *tuple(receivers.items()),
        ):
            _require_nonempty(name, "Jacobian scope")
            if (
                type(view) is not H7BorrowedTensorView
                or view.identity.shape != ()
                or view.identity.dtype != "float64"
            ):
                raise ValueError("Jacobian metadata requires borrowed float64 scalars")
            view.assert_intact()
        local_total = _jacobian_grouped_local_total(
            initial.tensor,
            {
                component_id: view.tensor
                for component_id, view in receivers.items()
            },
        )
        eps = torch.finfo(torch.float64).eps
        scale = max(
            1.0,
            float(torch.abs(local_total).item()),
            float(torch.abs(global_shift.tensor).item()),
        )
        if not torch.allclose(
            local_total,
            global_shift.tensor,
            rtol=64.0 * eps,
            atol=64.0 * eps * scale,
        ):
            raise ValueError("local Jacobian scopes do not sum to the global shift")
        if scope == "generative":
            if entropy is not None:
                raise ValueError("generative Jacobian metadata cannot carry entropy")
        elif (
            type(entropy) is not H7BorrowedTensorView
            or entropy.identity.shape != ()
            or entropy.identity.dtype != "float64"
        ):
            raise ValueError("recognition Jacobian metadata requires an entropy shift")
        else:
            entropy.assert_intact()
            if not torch.equal(entropy.tensor, global_shift.tensor):
                raise ValueError(
                    "recognition entropy shift must equal positive global logJ"
                )
        return MappingProxyType(
            {
                "scope": scope,
                "initial_logabsdet": _borrowed_tensor_semantic(initial),
                "receiver_logabsdet": _borrowed_tensor_mapping_semantic(receivers),
                "global_logabsdet": _borrowed_tensor_semantic(global_shift),
                "entropy_shift": (
                    None if entropy is None else _borrowed_tensor_semantic(entropy)
                ),
            }
        )

    def __post_init__(self) -> None:
        frozen = _freeze_mapping(self.receiver_logabsdet)
        if frozen is not self.receiver_logabsdet:
            object.__setattr__(self, "receiver_logabsdet", frozen)
        _require_sha256(self.semantic_sha256, "semantic_sha256")
        self.assert_live()

    def assert_live(self) -> None:
        values = {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name != "semantic_sha256"
        }
        expected = h7_owned_sha256(
            "vfe4.h7.jacobian-metadata-view.v1",
            self._live_semantic(values),
        )
        if expected != self.semantic_sha256:
            raise ValueError("Jacobian metadata live identity changed")


@dataclass(frozen=True)
class H7JacobianMetadataSnapshot(_H7IntegrityRecord):
    """Owned per-scope measure shifts bound into evidence hashes."""

    _integrity_field: ClassVar[str] = "metadata_sha256"
    _hash_domain: ClassVar[str] = "vfe4.h7.jacobian-metadata-snapshot.v1"

    scope: Literal["generative", "recognition"]
    initial_logabsdet: H7OwnedTensorSnapshot
    receiver_logabsdet: Mapping[str, H7OwnedTensorSnapshot]
    global_logabsdet: H7OwnedTensorSnapshot
    entropy_shift: H7OwnedTensorSnapshot | None
    metadata_sha256: str

    def __post_init__(self) -> None:
        if self.scope not in ("generative", "recognition"):
            raise ValueError("owned Jacobian metadata scope is invalid")
        initial = _require_owned_tensor(
            self.initial_logabsdet,
            "initial_logabsdet",
            shape=(),
        )
        if not isinstance(self.receiver_logabsdet, Mapping) or not (
            self.receiver_logabsdet
        ):
            raise ValueError("owned Jacobian metadata requires receiver scopes")
        receiver_values: dict[str, torch.Tensor] = {}
        for name, value in self.receiver_logabsdet.items():
            _require_nonempty(name, "Jacobian scope")
            receiver_values[name] = _require_owned_tensor(
                value,
                f"receiver_logabsdet[{name}]",
                shape=(),
            )
        global_shift = _require_owned_tensor(
            self.global_logabsdet,
            "global_logabsdet",
            shape=(),
        )
        with torch.no_grad():
            local_total = _jacobian_grouped_local_total(
                initial,
                receiver_values,
            )
            eps = torch.finfo(torch.float64).eps
            scale = max(
                1.0,
                float(torch.abs(local_total).item()),
                float(torch.abs(global_shift).item()),
            )
            if not torch.allclose(
                local_total,
                global_shift,
                rtol=64.0 * eps,
                atol=64.0 * eps * scale,
            ):
                raise ValueError(
                    "owned local Jacobian scopes do not sum to the global shift"
                )
        if self.scope == "generative":
            if self.entropy_shift is not None:
                raise ValueError(
                    "owned generative Jacobian metadata cannot carry entropy"
                )
        else:
            entropy = _require_owned_tensor(
                self.entropy_shift,
                "entropy_shift",
                shape=(),
            )
            if not torch.equal(entropy, global_shift):
                raise ValueError(
                    "owned recognition entropy shift must equal positive global logJ"
                )
        super().__post_init__()


@dataclass(frozen=True)
class H7DecoderSnapshot(_H7IntegrityRecord):
    _integrity_field: ClassVar[str] = "decoder_sha256"
    _hash_domain: ClassVar[str] = "vfe4.h7.decoder.v1"

    receiver_t: int
    state_weight: H7OwnedTensorSnapshot
    model_weight: H7OwnedTensorSnapshot
    bias: H7OwnedTensorSnapshot
    centered_stabilizer_class: Literal["transformed", "inside", "outside"]
    decoder_sha256: str

    def __post_init__(self) -> None:
        if type(self.receiver_t) is not int or self.receiver_t not in (1, 2):
            raise ValueError("decoder receiver_t must be exactly 1 or 2")
        state_weight = _require_owned_tensor(
            self.state_weight,
            "decoder state_weight",
            ndim=2,
        )
        model_weight = _require_owned_tensor(
            self.model_weight,
            "decoder model_weight",
            ndim=2,
        )
        if (
            tuple(state_weight.shape) != tuple(model_weight.shape)
            or state_weight.shape[0] < 2
            or state_weight.shape[1] not in (1, 2)
        ):
            raise ValueError("decoder weight shapes are inconsistent")
        _require_owned_tensor(
            self.bias,
            "decoder bias",
            shape=(state_weight.shape[0],),
        )
        if self.centered_stabilizer_class not in (
            "transformed",
            "inside",
            "outside",
        ):
            raise ValueError("decoder centered-stabilizer class is invalid")
        super().__post_init__()


@dataclass(frozen=True)
class H7GenerativeSnapshot(_H7IntegrityRecord):
    _integrity_field: ClassVar[str] = "snapshot_sha256"
    _hash_domain: ClassVar[str] = "vfe4.h7.generative-snapshot.v1"

    frames: tuple[H7OwnedTensorSnapshot, ...]
    ordered_links: Mapping[tuple[int, int], H7OwnedTensorSnapshot]
    initial_joint: H7GaussianComponentSnapshot
    transitions: tuple[H7AffineComponentSnapshot, ...]
    source_context: H7SourceContextSnapshot | None
    scalar_source_law: H7ScalarGenerativeSourceLawSnapshot | None
    decoders: tuple[H7DecoderSnapshot, ...]
    support_sha256: str
    jacobian: H7JacobianMetadataSnapshot
    snapshot_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.frames) is not tuple
            or len(self.frames) != 3
            or any(type(item) is not H7OwnedTensorSnapshot for item in self.frames)
        ):
            raise ValueError("generative snapshot requires exactly three frames")
        first_frame = _require_owned_tensor(self.frames[0], "frames[0]", ndim=2)
        if first_frame.shape[0] != first_frame.shape[1] or first_frame.shape[0] not in (
            1,
            2,
        ):
            raise ValueError("generative frame dimension must be exactly 1 or 2")
        dimension = int(first_frame.shape[0])
        frame_values: list[torch.Tensor] = []
        for index, frame in enumerate(self.frames):
            value = _require_owned_tensor(
                frame,
                f"frames[{index}]",
                shape=(dimension, dimension),
            )
            sign, logabsdet = torch.linalg.slogdet(value)
            if not bool(torch.isfinite(logabsdet).item()) or not bool(
                (sign > 0).item()
            ):
                raise ValueError("generative frames must lie in the direct group")
            frame_values.append(value)
        expected_link_keys = {
            (receiver, source)
            for receiver in range(3)
            for source in range(3)
            if receiver != source
        }
        if (
            not isinstance(self.ordered_links, Mapping)
            or set(self.ordered_links) != expected_link_keys
            or any(
                type(key) is not tuple
                or len(key) != 2
                or any(type(item) is not int for item in key)
                or type(value) is not H7OwnedTensorSnapshot
                for key, value in self.ordered_links.items()
            )
        ):
            raise ValueError("generative ordered-link inventory is invalid")
        eps = torch.finfo(torch.float64).eps
        with torch.no_grad():
            for (receiver, source), snapshot in self.ordered_links.items():
                value = _require_owned_tensor(
                    snapshot,
                    f"ordered_links[{receiver},{source}]",
                    shape=(dimension, dimension),
                )
                expected = torch.linalg.solve(
                    frame_values[source].T,
                    frame_values[receiver].T,
                ).T
                scale = max(
                    1.0,
                    float(value.abs().max()),
                    float(expected.abs().max()),
                )
                if not torch.allclose(
                    value,
                    expected,
                    rtol=256.0 * eps,
                    atol=256.0 * eps * scale,
                ):
                    raise ValueError("generative link disagrees with endpoint frames")
        if (
            type(self.initial_joint) is not H7GaussianComponentSnapshot
            or self.initial_joint.receiver_t is not None
            or self.initial_joint.source_j is not None
            or self.initial_joint.mean.shape != (2 * dimension,)
        ):
            raise ValueError("generative initial joint is inconsistent")
        if (
            type(self.transitions) is not tuple
            or not self.transitions
            or any(
                type(item) is not H7AffineComponentSnapshot for item in self.transitions
            )
            or any(item.offset.shape != (dimension,) for item in self.transitions)
            or len({item.component_id for item in self.transitions})
            != len(self.transitions)
        ):
            raise ValueError("generative transition inventory is invalid")
        if (self.source_context is None) == (self.scalar_source_law is None):
            raise ValueError(
                "generative snapshot requires exactly one source-law profile"
            )
        if self.source_context is not None:
            if (
                type(self.source_context) is not H7SourceContextSnapshot
                or dimension != 2
            ):
                raise ValueError("matrix source context is invalid")
        elif (
            type(self.scalar_source_law) is not H7ScalarGenerativeSourceLawSnapshot
            or dimension != 1
        ):
            raise ValueError(
                "scalar generative source law requires the H1 no-scorer profile"
            )
        if (
            type(self.decoders) is not tuple
            or any(type(item) is not H7DecoderSnapshot for item in self.decoders)
            or len(self.decoders) != 2
        ):
            raise ValueError("generative decoder inventory is invalid")
        if tuple(item.receiver_t for item in self.decoders) != (1, 2) or any(
            item.state_weight.shape[1] != dimension for item in self.decoders
        ):
            raise ValueError("generative decoder inventory is invalid")
        _require_sha256(self.support_sha256, "support_sha256")
        if (
            type(self.jacobian) is not H7JacobianMetadataSnapshot
            or self.jacobian.scope != "generative"
        ):
            raise ValueError(
                "generative snapshot requires owned generative Jacobian metadata"
            )
        expected_receiver_scopes = tuple(
            cast(str, name)
            for name in _freeze_mapping(
                {item.component_id: None for item in self.transitions}
            )
        )
        if tuple(self.jacobian.receiver_logabsdet) != expected_receiver_scopes:
            raise ValueError(
                "generative Jacobian receiver scopes do not match the "
                "transition component-ID inventory"
            )
        super().__post_init__()


@dataclass(frozen=True)
class H7GenerativeTensorLaw:
    components: tuple[H7TensorLawComponent, ...]
    source_context: H7SourceContextView | None
    scalar_source_law: H7ScalarGenerativeSourceLawSnapshot | None
    decoder_policy: H7DecoderPolicy
    support_sha256: str
    jacobian: H7JacobianMetadataView
    law_identity_sha256: str

    @classmethod
    def create(
        cls,
        *,
        components: tuple[H7TensorLawComponent, ...],
        source_context: H7SourceContextView | None,
        scalar_source_law: H7ScalarGenerativeSourceLawSnapshot | None,
        decoder_policy: H7DecoderPolicy,
        support_sha256: str,
        jacobian: H7JacobianMetadataView,
    ) -> "H7GenerativeTensorLaw":
        semantic = cls._live_semantic(
            components=components,
            source_context=source_context,
            scalar_source_law=scalar_source_law,
            decoder_policy=decoder_policy,
            support_sha256=support_sha256,
            jacobian=jacobian,
        )
        return cls(
            components=components,
            source_context=source_context,
            scalar_source_law=scalar_source_law,
            decoder_policy=decoder_policy,
            support_sha256=support_sha256,
            jacobian=jacobian,
            law_identity_sha256=h7_owned_sha256(
                "vfe4.h7.generative-tensor-law-live.v1", semantic
            ),
        )

    @staticmethod
    def _live_semantic(
        *,
        components: tuple[H7TensorLawComponent, ...],
        source_context: H7SourceContextView | None,
        scalar_source_law: H7ScalarGenerativeSourceLawSnapshot | None,
        decoder_policy: H7DecoderPolicy,
        support_sha256: str,
        jacobian: H7JacobianMetadataView,
    ) -> Mapping[str, object]:
        if type(components) is not tuple or not components:
            raise ValueError("generative tensor law requires components")
        for component in components:
            if type(component) is not H7TensorLawComponent:
                raise ValueError("generative components must be exact H7 records")
            component.assert_live()
        if source_context is not None:
            if type(source_context) is not H7SourceContextView:
                raise ValueError("source_context must be an exact H7 view")
            source_context.assert_live()
        if (source_context is None) == (scalar_source_law is None):
            raise ValueError(
                "generative tensor law requires exactly one source profile"
            )
        if scalar_source_law is not None and (
            type(scalar_source_law) is not H7ScalarGenerativeSourceLawSnapshot
        ):
            raise ValueError("scalar source law must be an exact H7 snapshot")
        _require_sha256(support_sha256, "support_sha256")
        if (
            type(jacobian) is not H7JacobianMetadataView
            or jacobian.scope != "generative"
        ):
            raise ValueError("generative law requires generative Jacobian metadata")
        jacobian.assert_live()
        return MappingProxyType(
            {
                "components": tuple(
                    component.component_identity_sha256 for component in components
                ),
                "source_context": (
                    None
                    if source_context is None
                    else source_context.semantic_context_sha256
                ),
                "scalar_source_law": scalar_source_law,
                "decoder_policy": decoder_policy,
                "support_sha256": support_sha256,
                "jacobian": jacobian.semantic_sha256,
            }
        )

    def __post_init__(self) -> None:
        if self.decoder_policy not in ("transform", "fixed"):
            raise ValueError("unsupported generative decoder policy")
        _require_sha256(self.law_identity_sha256, "law_identity_sha256")
        self.assert_live()

    def assert_live(self) -> None:
        expected = h7_owned_sha256(
            "vfe4.h7.generative-tensor-law-live.v1",
            self._live_semantic(
                components=self.components,
                source_context=self.source_context,
                scalar_source_law=self.scalar_source_law,
                decoder_policy=self.decoder_policy,
                support_sha256=self.support_sha256,
                jacobian=self.jacobian,
            ),
        )
        if expected != self.law_identity_sha256:
            raise ValueError("generative tensor-law live identity changed")


@dataclass(frozen=True)
class H7RecognitionSnapshot(_H7IntegrityRecord):
    _integrity_field: ClassVar[str] = "snapshot_sha256"
    _hash_domain: ClassVar[str] = "vfe4.h7.recognition-snapshot.v1"

    origin_family: H7RecognitionFamily
    representation: H7RecognitionRepresentation
    initial_joint: H7GaussianComponentSnapshot
    model_conditionals: tuple[H7AffineComponentSnapshot, ...]
    state_conditionals: tuple[H7AffineComponentSnapshot, ...]
    source_rows: tuple[H7SourceScorerRowSnapshot, ...]
    context: H7RecognitionContextSnapshot
    scalar_source_law: H7ScalarRecognitionSourceLawSnapshot | None
    jacobian: H7JacobianMetadataSnapshot
    snapshot_sha256: str

    def __post_init__(self) -> None:
        if self.origin_family not in (
            "structured_full_block",
            "factorized_diagonal_within_fiber",
        ):
            raise ValueError("unsupported recognition origin family")
        if self.representation not in (
            "structured_full_block",
            "factorized_diagonal_within_fiber",
            "unrestricted_full_block_pushforward",
        ):
            raise ValueError("unsupported recognition representation")
        if (
            self.origin_family == "structured_full_block"
            and self.representation != "structured_full_block"
        ):
            raise ValueError("structured origin must retain its representation")
        if (
            self.origin_family == "factorized_diagonal_within_fiber"
            and self.representation == "structured_full_block"
        ):
            raise ValueError("factorized origin cannot be relabeled structured")
        if (
            type(self.initial_joint) is not H7GaussianComponentSnapshot
            or self.initial_joint.receiver_t is not None
            or self.initial_joint.source_j is not None
        ):
            raise ValueError("recognition initial joint is invalid")
        for name, bank, conditionals in (
            ("model_conditionals", "model", self.model_conditionals),
            ("state_conditionals", "state", self.state_conditionals),
        ):
            if (
                type(conditionals) is not tuple
                or not conditionals
                or any(
                    type(item) is not H7AffineComponentSnapshot or item.bank != bank
                    for item in conditionals
                )
                or len({item.component_id for item in conditionals})
                != len(conditionals)
            ):
                raise ValueError(f"recognition {name} inventory is invalid")
        dimensions = {
            item.offset.shape
            for item in (*self.model_conditionals, *self.state_conditionals)
        }
        if len(dimensions) != 1:
            raise ValueError("recognition conditional dimensions disagree")
        (conditional_shape,) = tuple(dimensions)
        if (
            len(conditional_shape) != 1
            or conditional_shape[0] not in (1, 2)
            or self.initial_joint.mean.shape != (2 * conditional_shape[0],)
        ):
            raise ValueError("recognition joint/conditional dimensions disagree")
        if type(self.context) is not H7RecognitionContextSnapshot:
            raise ValueError("recognition context must be an exact snapshot")
        if self.scalar_source_law is not None:
            if (
                type(self.scalar_source_law) is not H7ScalarRecognitionSourceLawSnapshot
                or self.source_rows
                or conditional_shape != (1,)
                or self.origin_family != "structured_full_block"
            ):
                raise ValueError(
                    "scalar recognition source law requires the H1 no-scorer profile"
                )
        else:
            if (
                conditional_shape != (2,)
                or type(self.source_rows) is not tuple
                or tuple(
                    (row.bank, row.receiver_t, row.source_j)
                    for row in self.source_rows
                    if type(row) is H7SourceScorerRowSnapshot
                )
                != (
                    ("model", 1, 0),
                    ("model", 2, 1),
                    ("state", 1, 0),
                    ("state", 2, 1),
                )
                or len(self.source_rows) != 4
            ):
                raise ValueError("matrix recognition source-row inventory is invalid")
        if (
            type(self.jacobian) is not H7JacobianMetadataSnapshot
            or self.jacobian.scope != "recognition"
        ):
            raise ValueError(
                "recognition snapshot requires owned recognition Jacobian metadata"
            )
        expected_receiver_scopes = tuple(
            cast(str, name)
            for name in _freeze_mapping(
                {
                    item.component_id: None
                    for item in (
                        *self.model_conditionals,
                        *self.state_conditionals,
                    )
                }
            )
        )
        if tuple(self.jacobian.receiver_logabsdet) != expected_receiver_scopes:
            raise ValueError(
                "recognition Jacobian receiver scopes do not match the "
                "conditional component-ID inventory"
            )
        super().__post_init__()


@dataclass(frozen=True)
class H7RecognitionTensorLaw:
    origin_family: H7RecognitionFamily
    representation: H7RecognitionRepresentation
    components: tuple[H7TensorLawComponent, ...]
    source_rows: tuple[H7SourceScorerRowView, ...]
    context: H7RecognitionContextSnapshot
    scalar_source_law: H7ScalarRecognitionSourceLawSnapshot | None
    jacobian: H7JacobianMetadataView
    law_identity_sha256: str

    @classmethod
    def create(
        cls,
        *,
        origin_family: H7RecognitionFamily,
        representation: H7RecognitionRepresentation,
        components: tuple[H7TensorLawComponent, ...],
        source_rows: tuple[H7SourceScorerRowView, ...],
        context: H7RecognitionContextSnapshot,
        scalar_source_law: H7ScalarRecognitionSourceLawSnapshot | None,
        jacobian: H7JacobianMetadataView,
    ) -> "H7RecognitionTensorLaw":
        semantic = cls._live_semantic(
            origin_family=origin_family,
            representation=representation,
            components=components,
            source_rows=source_rows,
            context=context,
            scalar_source_law=scalar_source_law,
            jacobian=jacobian,
        )
        return cls(
            origin_family=origin_family,
            representation=representation,
            components=components,
            source_rows=source_rows,
            context=context,
            scalar_source_law=scalar_source_law,
            jacobian=jacobian,
            law_identity_sha256=h7_owned_sha256(
                "vfe4.h7.recognition-tensor-law-live.v1", semantic
            ),
        )

    @staticmethod
    def _live_semantic(
        *,
        origin_family: H7RecognitionFamily,
        representation: H7RecognitionRepresentation,
        components: tuple[H7TensorLawComponent, ...],
        source_rows: tuple[H7SourceScorerRowView, ...],
        context: H7RecognitionContextSnapshot,
        scalar_source_law: H7ScalarRecognitionSourceLawSnapshot | None,
        jacobian: H7JacobianMetadataView,
    ) -> Mapping[str, object]:
        if type(components) is not tuple or not components:
            raise ValueError("recognition tensor law requires components")
        for component in components:
            if type(component) is not H7TensorLawComponent:
                raise ValueError("recognition components must be exact H7 records")
            component.assert_live()
        if type(source_rows) is not tuple:
            raise ValueError("source_rows must be an exact tuple")
        for row in source_rows:
            if type(row) is not H7SourceScorerRowView:
                raise ValueError("source rows must be exact H7 views")
            row.assert_live()
        if type(context) is not H7RecognitionContextSnapshot:
            raise ValueError("context must be an exact H7 snapshot")
        if scalar_source_law is not None and (
            type(scalar_source_law) is not H7ScalarRecognitionSourceLawSnapshot
        ):
            raise ValueError(
                "scalar recognition source law must be an exact H7 snapshot"
            )
        if (not source_rows) != (scalar_source_law is not None):
            raise ValueError(
                "recognition law source rows/scalar source profile disagree"
            )
        if (
            type(jacobian) is not H7JacobianMetadataView
            or jacobian.scope != "recognition"
        ):
            raise ValueError("recognition law requires recognition Jacobian metadata")
        jacobian.assert_live()
        return MappingProxyType(
            {
                "origin_family": origin_family,
                "representation": representation,
                "components": tuple(
                    component.component_identity_sha256 for component in components
                ),
                "source_rows": tuple(row.semantic_row_sha256 for row in source_rows),
                "context": context,
                "scalar_source_law": scalar_source_law,
                "jacobian": jacobian.semantic_sha256,
            }
        )

    def __post_init__(self) -> None:
        if self.origin_family not in (
            "structured_full_block",
            "factorized_diagonal_within_fiber",
        ):
            raise ValueError("unsupported recognition origin family")
        if self.representation not in (
            "structured_full_block",
            "factorized_diagonal_within_fiber",
            "unrestricted_full_block_pushforward",
        ):
            raise ValueError("unsupported recognition representation")
        if (
            self.origin_family == "structured_full_block"
            and self.representation != "structured_full_block"
        ):
            raise ValueError("structured origin must retain its representation")
        if (
            self.origin_family == "factorized_diagonal_within_fiber"
            and self.representation == "structured_full_block"
        ):
            raise ValueError("factorized origin cannot be relabeled structured")
        _require_sha256(self.law_identity_sha256, "law_identity_sha256")
        self.assert_live()

    def assert_live(self) -> None:
        expected = h7_owned_sha256(
            "vfe4.h7.recognition-tensor-law-live.v1",
            self._live_semantic(
                origin_family=self.origin_family,
                representation=self.representation,
                components=self.components,
                source_rows=self.source_rows,
                context=self.context,
                scalar_source_law=self.scalar_source_law,
                jacobian=self.jacobian,
            ),
        )
        if expected != self.law_identity_sha256:
            raise ValueError("recognition tensor-law live identity changed")


@dataclass(frozen=True)
class H7CompleteLawSnapshot(_H7IntegrityRecord):
    _integrity_field: ClassVar[str] = "snapshot_sha256"
    _hash_domain: ClassVar[str] = "vfe4.h7.complete-law-snapshot.v1"

    fixture_id: Literal["h1-v1", "h7-v1"]
    generative: H7GenerativeSnapshot
    recognition: H7RecognitionSnapshot
    raw_fixture_sha256: str
    scalar_probe_set: H7ScalarProbeSetSnapshot | None
    snapshot_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.generative) is not H7GenerativeSnapshot
            or type(self.recognition) is not H7RecognitionSnapshot
        ):
            raise ValueError(
                "complete law requires exact generative/recognition records"
            )
        _require_sha256(self.raw_fixture_sha256, "raw_fixture_sha256")
        if self.fixture_id == "h1-v1":
            if (
                type(self.generative.scalar_source_law)
                is not H7ScalarGenerativeSourceLawSnapshot
                or type(self.recognition.scalar_source_law)
                is not H7ScalarRecognitionSourceLawSnapshot
                or type(self.scalar_probe_set) is not H7ScalarProbeSetSnapshot
            ):
                raise ValueError(
                    "H1 complete law requires explicit source and probe evidence"
                )
            if (
                self.generative.scalar_source_law.ordered_paths
                != self.recognition.scalar_source_law.ordered_paths
            ):
                raise ValueError("H1 generative/recognition path order disagrees")
            if self.scalar_probe_set.raw_fixture_sha256 != self.raw_fixture_sha256:
                raise ValueError("H1 scalar probe set has the wrong raw fixture")
        elif self.fixture_id == "h7-v1":
            if (
                self.generative.scalar_source_law is not None
                or self.recognition.scalar_source_law is not None
                or self.scalar_probe_set is not None
                or type(self.generative.source_context) is not H7SourceContextSnapshot
                or tuple(row.row_sha256 for row in self.recognition.source_rows)
                != tuple(
                    row.row_sha256 for row in self.generative.source_context.scorer_rows
                )
            ):
                raise ValueError("matrix complete law cannot carry scalar evidence")
        else:
            raise ValueError("unsupported complete-law fixture_id")
        super().__post_init__()


@dataclass(frozen=True)
class H7LawPairSnapshot(_H7IntegrityRecord):
    _integrity_field: ClassVar[str] = "law_pair_sha256"
    _hash_domain: ClassVar[str] = "vfe4.h7.law-pair.v1"

    original: H7CompleteLawSnapshot
    transformed: H7CompleteLawSnapshot
    action_sha256: str
    law_pair_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.original) is not H7CompleteLawSnapshot
            or type(self.transformed) is not H7CompleteLawSnapshot
            or self.original.fixture_id != self.transformed.fixture_id
            or self.original.raw_fixture_sha256 != self.transformed.raw_fixture_sha256
            or self.original.recognition.origin_family
            != self.transformed.recognition.origin_family
            or self.original.generative.support_sha256
            != self.transformed.generative.support_sha256
            or self.original.recognition.context != self.transformed.recognition.context
        ):
            raise ValueError("law pair fixture provenance is inconsistent")
        _require_sha256(self.action_sha256, "action_sha256")
        super().__post_init__()


@dataclass(frozen=True)
class H7DensityProbePair(_H7IntegrityRecord):
    _integrity_field: ClassVar[str] = "probe_sha256"
    _hash_domain: ClassVar[str] = "vfe4.h7.density-probe-pair.v1"

    probe_id: str
    fixture_id: Literal["h1-v1", "h7-v1"]
    component_id: str
    source_id: str
    action_sha256: str
    anchor_sha256: str
    anchor_provenance: str
    x: H7OwnedTensorSnapshot
    x_prime: H7OwnedTensorSnapshot
    initial_log_jacobian_shift: float
    receiver_log_jacobian_shift: float
    global_log_jacobian_shift: float
    probe_sha256: str

    def __post_init__(self) -> None:
        _require_nonempty(self.probe_id, "probe_id")
        if self.fixture_id not in ("h1-v1", "h7-v1"):
            raise ValueError("density probe fixture_id is invalid")
        _require_nonempty(self.component_id, "component_id")
        _require_nonempty(self.source_id, "source_id")
        _require_sha256(self.action_sha256, "action_sha256")
        _require_sha256(self.anchor_sha256, "anchor_sha256")
        _require_nonempty(self.anchor_provenance, "anchor_provenance")
        x = _require_owned_tensor(self.x, "density probe x", ndim=1)
        x_prime = _require_owned_tensor(
            self.x_prime,
            "density probe x_prime",
            shape=tuple(x.shape),
        )
        if x.numel() <= 0 or x_prime.numel() != x.numel():
            raise ValueError("density probe vectors must be nonempty and paired")
        if ".initial_joint" in self.component_id:
            if (
                self.receiver_log_jacobian_shift != 0.0
                or self.global_log_jacobian_shift != 0.0
            ):
                raise ValueError("initial density probe has the wrong shift scope")
        elif ".receiver_" in self.component_id:
            if (
                self.initial_log_jacobian_shift != 0.0
                or self.global_log_jacobian_shift != 0.0
            ):
                raise ValueError("receiver density probe has the wrong shift scope")
        elif ".global" in self.component_id:
            if (
                self.initial_log_jacobian_shift != 0.0
                or self.receiver_log_jacobian_shift != 0.0
            ):
                raise ValueError("global density probe has the wrong shift scope")
        else:
            raise ValueError("density probe component has no supported shift scope")
        super().__post_init__()


@dataclass(frozen=True)
class H7ScalarProbeSetSnapshot(_H7IntegrityRecord):
    _integrity_field: ClassVar[str] = "probe_set_sha256"
    _hash_domain: ClassVar[str] = "vfe4.h7.scalar-probe-set.v1"

    raw_fixture_sha256: str
    ordered_source_path_ids: tuple[str, str, str, str]
    scalar_trial_action_sha256: tuple[str, str]
    anchor_provenance: str
    probe_pairs: tuple[
        H7DensityProbePair,
        H7DensityProbePair,
        H7DensityProbePair,
        H7DensityProbePair,
        H7DensityProbePair,
        H7DensityProbePair,
        H7DensityProbePair,
        H7DensityProbePair,
    ]
    probe_set_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.raw_fixture_sha256, "raw_fixture_sha256")
        expected_path_ids = tuple(
            f"h1-path-{index}:a{a[1]}-b{b[1]}"
            for index, (a, b, _, _) in enumerate(H7_H1_SOURCE_PATHS)
        )
        if self.ordered_source_path_ids != expected_path_ids:
            raise ValueError("scalar probe paths are missing, duplicated, or reordered")
        if (
            type(self.scalar_trial_action_sha256) is not tuple
            or len(self.scalar_trial_action_sha256) != 2
        ):
            raise ValueError("scalar probe set requires both trial action hashes")
        for index, digest in enumerate(self.scalar_trial_action_sha256):
            _require_sha256(digest, f"scalar_trial_action_sha256[{index}]")
        _require_nonempty(self.anchor_provenance, "anchor_provenance")
        if type(self.probe_pairs) is not tuple or len(self.probe_pairs) != 8:
            raise ValueError("scalar probe set requires two trials by four paths")
        if any(type(pair) is not H7DensityProbePair for pair in self.probe_pairs):
            raise ValueError("scalar probe pairs must be exact H7 records")
        expected_pairs = tuple(
            (action_sha256, path_id)
            for action_sha256 in self.scalar_trial_action_sha256
            for path_id in self.ordered_source_path_ids
        )
        observed_pairs = tuple(
            (pair.action_sha256, pair.source_id) for pair in self.probe_pairs
        )
        if observed_pairs != expected_pairs:
            raise ValueError("scalar probe pair trial/path order changed")
        for pair in self.probe_pairs:
            expected_anchor_sha256 = h7_owned_sha256(
                "vfe4.h7.scalar-density-anchor.v1",
                {
                    "raw_fixture_sha256": self.raw_fixture_sha256,
                    "source_id": pair.source_id,
                    "anchor": pair.x,
                },
            )
            if (
                pair.fixture_id != "h1-v1"
                or pair.component_id != "h1.p.global.source_path"
                or self.raw_fixture_sha256 not in pair.anchor_provenance
                or self.anchor_provenance not in pair.anchor_provenance
                or pair.source_id not in pair.anchor_provenance
                or pair.anchor_sha256 != expected_anchor_sha256
                or pair.x.shape != (6,)
                or pair.x_prime.shape != (6,)
                or pair.initial_log_jacobian_shift != 0.0
                or pair.receiver_log_jacobian_shift != 0.0
            ):
                raise ValueError("scalar probe pair provenance/scope is invalid")
        super().__post_init__()


@dataclass(frozen=True)
class H7Fixture(_H7IntegrityRecord):
    _integrity_field: ClassVar[str] = "fixture_sha256"
    _hash_domain: ClassVar[str] = "vfe4.h7.fixture.v1"

    fixture_id: Literal["h7-v1"]
    raw_fixture_sha256: str
    frame_profiles: Mapping[H7FrameProfile, tuple[H7OwnedTensorSnapshot, ...]]
    actions: Mapping[str, H7GLPlus2Action]
    generative: H7GenerativeSnapshot
    recognition_families: tuple[H7RecognitionSnapshot, H7RecognitionSnapshot]
    matrix_trial_specs: tuple[H7TrialSpec, ...]
    density_probe_table_raw_sha256: str
    density_probe_pairs: tuple[H7DensityProbePair, ...]
    density_probe_set_sha256: str
    fixture_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.raw_fixture_sha256, "raw_fixture_sha256")
        if (
            self.fixture_id != "h7-v1"
            or not isinstance(self.frame_profiles, Mapping)
            or set(self.frame_profiles) != {"identity", "nonidentity"}
            or any(
                type(frames) is not tuple
                or len(frames) != 3
                or any(type(frame) is not H7OwnedTensorSnapshot for frame in frames)
                for frames in self.frame_profiles.values()
            )
            or not isinstance(self.actions, Mapping)
            or set(self.actions) != {"diagonal", "internal", "fixed_decoder_stabilizer"}
            or any(
                type(action) is not H7GLPlus2Action for action in self.actions.values()
            )
            or type(self.generative) is not H7GenerativeSnapshot
            or type(self.recognition_families) is not tuple
            or len(self.recognition_families) != 2
            or any(
                type(item) is not H7RecognitionSnapshot
                for item in self.recognition_families
            )
            or type(self.matrix_trial_specs) is not tuple
            or len(self.matrix_trial_specs) != len(H7_MATRIX_TRIAL_IDS)
            or any(type(item) is not H7TrialSpec for item in self.matrix_trial_specs)
            or self.density_probe_table_raw_sha256 != H7_DENSITY_PROBE_TABLE_RAW_SHA256
            or type(self.density_probe_pairs) is not tuple
            or len(self.density_probe_pairs) != 486
            or any(
                type(item) is not H7DensityProbePair
                for item in self.density_probe_pairs
            )
        ):
            raise ValueError("H7 fixture inventory or raw table identity changed")
        if (
            tuple(item.trial_id for item in self.matrix_trial_specs)
            != H7_MATRIX_TRIAL_IDS
        ):
            raise ValueError("H7 matrix trial order changed")
        if tuple(item.origin_family for item in self.recognition_families) != (
            "structured_full_block",
            "factorized_diagonal_within_fiber",
        ):
            raise ValueError("H7 recognition-family order changed")
        source_context = self.generative.source_context
        if type(source_context) is not H7SourceContextSnapshot:
            raise ValueError("H7 fixture requires the matrix source context")
        expected_source_rows = tuple(
            row.row_sha256 for row in source_context.scorer_rows
        )
        if any(
            tuple(row.row_sha256 for row in family.source_rows) != expected_source_rows
            for family in self.recognition_families
        ):
            raise ValueError("H7 generative/recognition source rows disagree")
        expected_action_name = {
            "matrix-identity-base-transformed": "diagonal",
            "matrix-identity-internal-transformed": "internal",
            "matrix-nonidentity-base-transformed": "diagonal",
            "matrix-nonidentity-internal-transformed": "internal",
            "matrix-fixed-decoder-centered-stabilizer": ("fixed_decoder_stabilizer"),
            "matrix-fixed-decoder-outside-stabilizer": "diagonal",
        }
        if any(
            spec.action_sha256
            != self.actions[expected_action_name[spec.trial_id]].action_sha256
            for spec in self.matrix_trial_specs
        ):
            raise ValueError("H7 trial/action fixture binding changed")
        allowed_action_hashes = {
            action.action_sha256 for action in self.actions.values()
        }
        if any(
            pair.fixture_id != "h7-v1"
            or pair.action_sha256 not in allowed_action_hashes
            for pair in self.density_probe_pairs
        ):
            raise ValueError("H7 density probe fixture/action binding changed")
        expected_density_probe_set_sha256 = h7_owned_sha256(
            "vfe4.h7.density-probe-set.v1",
            self.density_probe_pairs,
        )
        if self.density_probe_set_sha256 != expected_density_probe_set_sha256:
            raise ValueError(
                "density_probe_set_sha256 does not match density_probe_pairs"
            )
        super().__post_init__()


@dataclass(frozen=True)
class H7OperandRecord(_H7IntegrityRecord):
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
        if self.category not in (
            "vector",
            "information",
            "offset",
            "decoder",
            "covariance",
            "precision",
            "second_moment",
            "map",
            "cocycle",
            "density",
            "local_term",
            "complete_objective",
            "backward",
        ):
            raise ValueError("unsupported H7 operand category")
        if self.role not in (
            "original",
            "transformed",
            "reference",
            "recovered",
            "oracle",
        ):
            raise ValueError("unsupported H7 operand role")
        _require_nonempty(self.dtype, "dtype")
        if (
            type(self.shape) is not tuple
            or any(type(size) is not int or size < 0 for size in self.shape)
            or self.scale < 0.0
            or self.condition_number < 1.0
            or self.normalization <= 0.0
        ):
            raise ValueError("invalid H7 operand geometry")
        _require_sha256(self.value_sha256, "value_sha256")
        if self.oracle_value is not None:
            if type(self.oracle_value) is not str or not self.oracle_value:
                raise ValueError("oracle_value must be a nonempty decimal string")
            try:
                oracle = Decimal(self.oracle_value)
            except InvalidOperation as error:
                raise ValueError("oracle_value must be an exact decimal") from error
            if not oracle.is_finite():
                raise ValueError("oracle_value must be finite")
        if (self.role == "oracle") != (self.oracle_value is not None):
            raise ValueError("only oracle operands carry oracle_value")
        super().__post_init__()


@dataclass(frozen=True)
class H7AllowanceContribution(_H7IntegrityRecord):
    _integrity_field: ClassVar[str] = "contribution_sha256"
    _hash_domain: ClassVar[str] = "vfe4.h7.allowance-contribution.v1"

    kind: H7AllowanceKind
    operation_id: str
    operation_kind: H7OperationKind
    operation_count: int
    quadrature_order: Literal[41, 51] | None
    unit_allowance: float
    value: float
    contribution_sha256: str

    def __post_init__(self) -> None:
        if self.kind not in (
            "operation_rounding",
            "quadrature_convergence",
            "reference_rounding",
        ):
            raise ValueError("unsupported allowance kind")
        _require_nonempty(self.operation_id, "operation_id")
        if self.operation_kind not in (
            "exact_identity",
            "direct_solve",
            "matrix_product",
            "quadratic_form",
            "logdet",
            "analytic_density",
            "gauss_hermite",
            "pair_comparison",
        ):
            raise ValueError("unsupported allowance operation")
        if type(self.operation_count) is not int or self.operation_count <= 0:
            raise ValueError("operation_count must be positive")
        if (
            self.kind == "quadrature_convergence"
            or self.operation_kind == "gauss_hermite"
        ):
            if (
                self.kind != "quadrature_convergence"
                or self.operation_kind != "gauss_hermite"
                or self.quadrature_order not in (41, 51)
            ):
                raise ValueError("quadrature allowance metadata is inconsistent")
        elif self.quadrature_order is not None:
            raise ValueError("analytic allowance cannot name quadrature order")
        if self.unit_allowance < 0.0 or self.value < 0.0:
            raise ValueError("allowance values must be nonnegative")
        super().__post_init__()


@dataclass(frozen=True)
class H7BudgetRecord(_H7IntegrityRecord):
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
            type(self.operands) is not tuple
            or not self.operands
            or type(self.contributions) is not tuple
            or not self.contributions
            or self.comparison_normalization <= 0.0
            or self.total_allowance < 0.0
        ):
            raise ValueError("budget requires local operands and contributions")
        if any(
            type(item) is not H7OperandRecord or item.category != self.category
            for item in self.operands
        ):
            raise ValueError("budget operands must share its category")
        operand_ids = tuple(item.operand_id for item in self.operands)
        if len(set(operand_ids)) != len(operand_ids):
            raise ValueError("budget operand IDs must be unique")
        if any(
            type(item) is not H7AllowanceContribution for item in self.contributions
        ):
            raise ValueError("budget contributions must be exact H7 records")
        contribution_ids = tuple(item.operation_id for item in self.contributions)
        if len(set(contribution_ids)) != len(contribution_ids):
            raise ValueError("allowance operation IDs must be unique")
        expected_total = math.fsum(item.value for item in self.contributions)
        if not math.isclose(
            self.total_allowance,
            expected_total,
            rel_tol=0.0,
            abs_tol=max(math.ulp(expected_total), math.ulp(self.total_allowance)),
        ):
            raise ValueError("total_allowance must equal its contributions")
        super().__post_init__()


@dataclass(frozen=True)
class H7ResidualRecord(_H7IntegrityRecord):
    _integrity_field: ClassVar[str] = "residual_sha256"
    _hash_domain: ClassVar[str] = "vfe4.h7.residual.v1"

    invariant_id: str
    category: H7ResidualCategory
    value: float
    budget: H7BudgetRecord
    passed: bool
    residual_sha256: str

    def __post_init__(self) -> None:
        _require_nonempty(self.invariant_id, "invariant_id")
        if self.category not in (
            "tensor",
            "law",
            "cocycle",
            "density",
            "jacobian",
            "source",
            "decoder",
            "local_term",
            "monolithic",
            "evidence",
            "posterior_kl",
            "absolute",
            "relative",
            "backward",
        ):
            raise ValueError("unsupported residual category")
        if (
            self.value < 0.0
            or type(self.budget) is not H7BudgetRecord
            or self.budget.invariant_id != self.invariant_id
            or type(self.passed) is not bool
        ):
            raise ValueError("residual/budget identity is inconsistent")
        if self.passed != (self.value <= self.budget.total_allowance):
            raise ValueError("residual pass flag disagrees with inclusive budget")
        super().__post_init__()


@dataclass(frozen=True)
class H7BackwardResidualRecord(_H7IntegrityRecord):
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
            self.numerator < 0.0
            or self.normalization <= 0.0
            or self.value < 0.0
            or type(self.budget) is not H7BudgetRecord
            or self.budget.category != "backward"
            or type(self.passed) is not bool
        ):
            raise ValueError("invalid backward residual")
        expected = self.numerator / self.normalization
        if not math.isclose(
            self.value,
            expected,
            rel_tol=8.0 * torch.finfo(torch.float64).eps,
            abs_tol=0.0,
        ):
            raise ValueError("backward value must be numerator/normalization")
        if self.passed != (self.value <= self.budget.total_allowance):
            raise ValueError("backward pass flag disagrees with inclusive budget")
        super().__post_init__()


@dataclass(frozen=True)
class H7EnvelopeOperandRecord(_H7IntegrityRecord):
    _integrity_field: ClassVar[str] = "record_sha256"
    _hash_domain: ClassVar[str] = "vfe4.h7.envelope-operand.v1"

    operand_id: str
    determinant: float | None
    norm_2: float | None
    inverse_norm_2: float | None
    minimum_eigenvalue: float | None
    maximum_eigenvalue: float | None
    condition_number_2: float
    within_envelope: bool
    record_sha256: str

    def __post_init__(self) -> None:
        _require_nonempty(self.operand_id, "operand_id")
        group_fields = (
            self.determinant,
            self.norm_2,
            self.inverse_norm_2,
        )
        spd_fields = (self.minimum_eigenvalue, self.maximum_eigenvalue)
        group = all(value is not None for value in group_fields) and all(
            value is None for value in spd_fields
        )
        spd = all(value is None for value in group_fields) and all(
            value is not None for value in spd_fields
        )
        if not (group or spd) or self.condition_number_2 < 1.0:
            raise ValueError("envelope operand must be exclusively group or SPD")
        if group and (
            cast(float, self.determinant) <= 0.0
            or cast(float, self.norm_2) <= 0.0
            or cast(float, self.inverse_norm_2) <= 0.0
        ):
            raise ValueError("group envelope diagnostics must be positive")
        if spd and (
            cast(float, self.minimum_eigenvalue) <= 0.0
            or cast(float, self.maximum_eigenvalue)
            < cast(float, self.minimum_eigenvalue)
        ):
            raise ValueError("SPD envelope diagnostics must be positive and ordered")
        if type(self.within_envelope) is not bool:
            raise ValueError("within_envelope must be bool")
        expected_within_envelope = (
            cast(float, self.norm_2) <= 2.0 and cast(float, self.inverse_norm_2) <= 2.0
            if group
            else self.condition_number_2 <= 1000.0
        )
        if self.within_envelope != expected_within_envelope:
            raise ValueError("within_envelope disagrees with the frozen H7 limits")
        super().__post_init__()


@dataclass(frozen=True)
class H7EnvelopeRecord(_H7IntegrityRecord):
    _integrity_field: ClassVar[str] = "envelope_sha256"
    _hash_domain: ClassVar[str] = "vfe4.h7.envelope.v1"

    group_operands: tuple[H7EnvelopeOperandRecord, ...]
    spd_operands: tuple[H7EnvelopeOperandRecord, ...]
    inclusive: Literal[True]
    passed: bool
    envelope_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.group_operands) is not tuple
            or not self.group_operands
            or type(self.spd_operands) is not tuple
            or not self.spd_operands
            or self.inclusive is not True
            or type(self.passed) is not bool
        ):
            raise ValueError("invalid H7 envelope inventory")
        all_operands = (*self.group_operands, *self.spd_operands)
        if any(type(item) is not H7EnvelopeOperandRecord for item in all_operands):
            raise ValueError("envelope operands must be exact H7 records")
        ids = tuple(item.operand_id for item in all_operands)
        if len(set(ids)) != len(ids):
            raise ValueError("envelope operand IDs must be unique")
        if self.passed != all(item.within_envelope for item in all_operands):
            raise ValueError("envelope pass flag disagrees with operand records")
        super().__post_init__()


@dataclass(frozen=True)
class H7DensityObservationRecord(_H7IntegrityRecord):
    _integrity_field: ClassVar[str] = "observation_sha256"
    _hash_domain: ClassVar[str] = "vfe4.h7.density-observation.v1"

    probe_sha256: str
    role: H7DensityRole
    original_value: float
    transformed_value: float
    expected_log_jacobian_shift: float
    residual: H7ResidualRecord
    observation_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.probe_sha256, "probe_sha256")
        if self.role not in ("p", "q", "log_ratio"):
            raise ValueError("density observation role is invalid")
        expected_invariant_id = (
            f"density_probe.{self.probe_sha256}.{self.role}"
        )
        expected_residual = abs(
            (self.transformed_value - self.original_value)
            + self.expected_log_jacobian_shift
        )
        operand_roles = (
            set()
            if type(self.residual) is not H7ResidualRecord
            else {item.role for item in self.residual.budget.operands}
        )
        if (
            type(self.residual) is not H7ResidualRecord
            or self.residual.invariant_id != expected_invariant_id
            or self.residual.category != "density"
            or self.residual.value != expected_residual
            or not {"original", "transformed"}.issubset(operand_roles)
        ):
            raise ValueError(
                "density observation does not preserve its exact residual"
            )
        super().__post_init__()


@dataclass(frozen=True)
class H7DensityProbeEvaluation(_H7IntegrityRecord):
    _integrity_field: ClassVar[str] = "evaluation_sha256"
    _hash_domain: ClassVar[str] = "vfe4.h7.density-probe-evaluation.v1"

    probe: H7DensityProbePair
    observations: tuple[H7DensityObservationRecord, ...]
    evaluation_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.probe) is not H7DensityProbePair
            or type(self.observations) is not tuple
            or not self.observations
            or any(
                type(item) is not H7DensityObservationRecord
                for item in self.observations
            )
            or any(
                item.probe_sha256 != self.probe.probe_sha256
                for item in self.observations
            )
        ):
            raise ValueError("density probe evaluation is incomplete")
        roles = tuple(item.role for item in self.observations)
        expected_roles: tuple[H7DensityRole, ...]
        if ".global" in self.probe.component_id:
            expected_roles = ("p", "q", "log_ratio")
        elif self.probe.component_id.startswith("p."):
            expected_roles = ("p",)
        elif self.probe.component_id.startswith("q."):
            expected_roles = ("q",)
        else:
            raise ValueError("density probe component has no p/q role")
        if roles != expected_roles:
            raise ValueError("density probe role inventory is not exact")
        shift = math.fsum(
            (
                self.probe.initial_log_jacobian_shift,
                self.probe.receiver_log_jacobian_shift,
                self.probe.global_log_jacobian_shift,
            )
        )
        if any(
            item.expected_log_jacobian_shift
            != (0.0 if item.role == "log_ratio" else shift)
            for item in self.observations
        ):
            raise ValueError("density observation has the wrong Jacobian shift")
        if roles == ("p", "q", "log_ratio"):
            p_value, q_value, ratio_value = self.observations
            if (
                ratio_value.original_value
                != p_value.original_value - q_value.original_value
                or ratio_value.transformed_value
                != p_value.transformed_value - q_value.transformed_value
            ):
                raise ValueError("density log ratio does not equal log p minus log q")
        super().__post_init__()


@dataclass(frozen=True)
class H7FactorizedPromotionWitness(_H7IntegrityRecord):
    _integrity_field: ClassVar[str] = "witness_sha256"
    _hash_domain: ClassVar[str] = "vfe4.h7.factorized-promotion-witness.v1"

    action_sha256: str
    component_id: str
    covariance_snapshot_sha256: str
    row: int
    column: int
    value: float
    transformed_representation: Literal["unrestricted_full_block_pushforward"]
    witness_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.action_sha256, "action_sha256")
        _require_nonempty(self.component_id, "component_id")
        _require_sha256(
            self.covariance_snapshot_sha256,
            "covariance_snapshot_sha256",
        )
        if (
            type(self.row) is not int
            or type(self.column) is not int
            or self.row < 0
            or self.column < 0
            or self.row >= self.column
            or self.value == 0.0
            or self.transformed_representation
            != "unrestricted_full_block_pushforward"
        ):
            raise ValueError("factorized promotion witness is invalid")
        super().__post_init__()


@dataclass(frozen=True)
class H7IndependentH1EvidenceRecord(_H7IntegrityRecord):
    _integrity_field: ClassVar[str] = "record_sha256"
    _hash_domain: ClassVar[str] = "vfe4.h7.independent-h1-evidence.v1"

    fixture_id: Literal["h1-v1"]
    raw_fixture_sha256: str
    action_sha256: str
    normalization_identity_sha256: str
    producer_identity_sha256: str
    original_log_evidence: float
    transformed_log_evidence: float
    original_posterior_kl: float
    transformed_posterior_kl: float
    record_sha256: str

    def __post_init__(self) -> None:
        if self.fixture_id != "h1-v1":
            raise ValueError("independent evidence is restricted to H1")
        for name in (
            "raw_fixture_sha256",
            "action_sha256",
            "normalization_identity_sha256",
            "producer_identity_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        super().__post_init__()


@dataclass(frozen=True)
class H7InitialJointKlRecord(_H7IntegrityRecord):
    _integrity_field: ClassVar[str] = "record_sha256"
    _hash_domain: ClassVar[str] = "vfe4.h7.initial-joint-kl.v1"

    term_id: Literal["K0_joint_z0_m0"]
    original_factor_ids: tuple[str, ...]
    transformed_factor_ids: tuple[str, ...]
    original_value: float
    transformed_value: float
    residual: H7ResidualRecord
    chain_decomposition: None
    record_sha256: str

    def __post_init__(self) -> None:
        if (
            self.term_id != "K0_joint_z0_m0"
            or len(self.original_factor_ids) != 1
            or len(self.transformed_factor_ids) != 1
            or any(
                type(item) is not str
                or len(item) != 64
                or any(character not in _LOWER_HEX for character in item)
                for item in (
                    *self.original_factor_ids,
                    *self.transformed_factor_ids,
                )
            )
            or self.chain_decomposition is not None
            or type(self.residual) is not H7ResidualRecord
            or self.residual.invariant_id != self.term_id
        ):
            raise ValueError("initial KL must remain one undecomposed joint term")
        super().__post_init__()


@dataclass(frozen=True)
class H7LocalTermRecord(_H7IntegrityRecord):
    _integrity_field: ClassVar[str] = "term_sha256"
    _hash_domain: ClassVar[str] = "vfe4.h7.local-term.v1"

    term_id: str
    original_factor_ids: tuple[str, ...]
    transformed_factor_ids: tuple[str, ...]
    original_value: float
    transformed_value: float
    signed_child_ids: tuple[str, ...]
    residual: H7ResidualRecord
    term_sha256: str

    def __post_init__(self) -> None:
        _require_nonempty(self.term_id, "term_id")
        if (
            type(self.original_factor_ids) is not tuple
            or not self.original_factor_ids
            or type(self.transformed_factor_ids) is not tuple
            or not self.transformed_factor_ids
            or any(
                type(item) is not str
                or len(item) != 64
                or any(character not in _LOWER_HEX for character in item)
                for item in (
                    *self.original_factor_ids,
                    *self.transformed_factor_ids,
                )
            )
            or type(self.signed_child_ids) is not tuple
            or any(type(item) is not str or not item for item in self.signed_child_ids)
            or len(set(self.signed_child_ids)) != len(self.signed_child_ids)
            or type(self.residual) is not H7ResidualRecord
            or self.residual.invariant_id != self.term_id
        ):
            raise ValueError("invalid local-term evidence")
        super().__post_init__()


@dataclass(frozen=True)
class H7ObjectiveCovarianceEvaluation(_H7IntegrityRecord):
    _integrity_field: ClassVar[str] = "evaluation_sha256"
    _hash_domain: ClassVar[str] = "vfe4.h7.objective-covariance.v1"

    original_factor_trace_sha256: str
    transformed_factor_trace_sha256: str
    original_ordered_factor_ids: tuple[str, ...]
    transformed_ordered_factor_ids: tuple[str, ...]
    original_ordered_factor_values: tuple[float, ...]
    transformed_ordered_factor_values: tuple[float, ...]
    original_complete_local_value: float
    transformed_complete_local_value: float
    initial_joint_kl: H7InitialJointKlRecord
    local_terms: tuple[H7LocalTermRecord, ...]
    density_probes: tuple[H7DensityProbePair, ...]
    density_probe_evaluations: tuple[H7DensityProbeEvaluation, ...]
    scorer_residuals: tuple[H7ResidualRecord, ...]
    complete_local: H7ResidualRecord
    complete_monolithic: H7ResidualRecord
    p_density_shift: H7ResidualRecord
    q_density_shift: H7ResidualRecord
    log_ratio: H7ResidualRecord
    entropy_shift: H7ResidualRecord
    scalar_evidence: H7IndependentH1EvidenceRecord | None
    factorized_promotion_witness: H7FactorizedPromotionWitness | None
    evidence: H7ResidualRecord | None
    posterior_kl: H7ResidualRecord | None
    not_applicable_reason: str | None
    evaluation_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "original_factor_trace_sha256",
            "transformed_factor_trace_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if (
            type(self.original_ordered_factor_ids) is not tuple
            or len(self.original_ordered_factor_ids) != 13
            or type(self.transformed_ordered_factor_ids) is not tuple
            or len(self.transformed_ordered_factor_ids) != 13
            or type(self.original_ordered_factor_values) is not tuple
            or len(self.original_ordered_factor_values) != 13
            or type(self.transformed_ordered_factor_values) is not tuple
            or len(self.transformed_ordered_factor_values) != 13
            or any(
                type(item) is not str
                or len(item) != 64
                or any(character not in _LOWER_HEX for character in item)
                for item in (
                    *self.original_ordered_factor_ids,
                    *self.transformed_ordered_factor_ids,
                )
            )
            or len(set(self.original_ordered_factor_ids)) != 13
            or len(set(self.transformed_ordered_factor_ids)) != 13
            or type(self.initial_joint_kl) is not H7InitialJointKlRecord
            or type(self.local_terms) is not tuple
            or not self.local_terms
            or any(type(item) is not H7LocalTermRecord for item in self.local_terms)
            or tuple(item.term_id for item in self.local_terms)
            != _H7_OBJECTIVE_LOCAL_TERM_IDS
            or type(self.density_probes) is not tuple
            or not self.density_probes
            or any(type(item) is not H7DensityProbePair for item in self.density_probes)
            or type(self.density_probe_evaluations) is not tuple
            or len(self.density_probe_evaluations) != len(self.density_probes)
            or any(
                type(item) is not H7DensityProbeEvaluation
                for item in self.density_probe_evaluations
            )
            or tuple(
                item.probe.probe_sha256
                for item in self.density_probe_evaluations
            )
            != tuple(item.probe_sha256 for item in self.density_probes)
            or type(self.scorer_residuals) is not tuple
            or any(type(item) is not H7ResidualRecord for item in self.scorer_residuals)
        ):
            raise ValueError("objective evaluation has incomplete owned evidence")
        original_bound_ids = (
            *self.initial_joint_kl.original_factor_ids,
            *(
                factor_id
                for item in self.local_terms
                for factor_id in item.original_factor_ids
            ),
        )
        transformed_bound_ids = (
            *self.initial_joint_kl.transformed_factor_ids,
            *(
                factor_id
                for item in self.local_terms
                for factor_id in item.transformed_factor_ids
            ),
        )
        if (
            len(original_bound_ids) != 13
            or len(transformed_bound_ids) != 13
            or set(original_bound_ids) != set(self.original_ordered_factor_ids)
            or set(transformed_bound_ids)
            != set(self.transformed_ordered_factor_ids)
        ):
            raise ValueError("local terms do not bind the complete H6 factor trace")
        original_by_id = dict(
            zip(
                self.original_ordered_factor_ids,
                self.original_ordered_factor_values,
                strict=True,
            )
        )
        transformed_by_id = dict(
            zip(
                self.transformed_ordered_factor_ids,
                self.transformed_ordered_factor_values,
                strict=True,
            )
        )
        grouped_records = (self.initial_joint_kl, *self.local_terms)
        if any(
            record.original_value
            != math.fsum(
                original_by_id[factor_id]
                for factor_id in record.original_factor_ids
            )
            or record.transformed_value
            != math.fsum(
                transformed_by_id[factor_id]
                for factor_id in record.transformed_factor_ids
            )
            for record in grouped_records
        ):
            raise ValueError("local term values changed after trace binding")
        required_residuals = (
            self.complete_local,
            self.complete_monolithic,
            self.p_density_shift,
            self.q_density_shift,
            self.log_ratio,
            self.entropy_shift,
        )
        if any(type(item) is not H7ResidualRecord for item in required_residuals):
            raise ValueError("objective evaluation residuals must be exact records")
        if self.complete_local.value != abs(
            self.transformed_complete_local_value
            - self.original_complete_local_value
        ):
            raise ValueError("complete-local residual is not trace-authoritative")
        observations = tuple(
            observation
            for item in self.density_probe_evaluations
            for observation in item.observations
        )
        if len(
            {item.residual.budget.budget_sha256 for item in observations}
        ) != len(observations):
            raise ValueError("each density observation requires its own budget")
        role_maxima = {
            role: max(
                item.residual.value for item in observations if item.role == role
            )
            for role in ("p", "q", "log_ratio")
        }
        if (
            self.p_density_shift.value != role_maxima["p"]
            or self.q_density_shift.value != role_maxima["q"]
            or self.log_ratio.value != role_maxima["log_ratio"]
        ):
            raise ValueError("density maxima must be computed after per-probe records")
        if (
            self.factorized_promotion_witness is not None
            and type(self.factorized_promotion_witness)
            is not H7FactorizedPromotionWitness
        ):
            raise ValueError("factorized promotion witness has the wrong type")
        evidence_available = self.evidence is not None or self.posterior_kl is not None
        if evidence_available:
            if (
                type(self.scalar_evidence) is not H7IndependentH1EvidenceRecord
                or type(self.evidence) is not H7ResidualRecord
                or type(self.posterior_kl) is not H7ResidualRecord
                or self.not_applicable_reason is not None
            ):
                raise ValueError("evidence and posterior KL must appear together")
        elif (
            self.scalar_evidence is not None
            or type(self.not_applicable_reason) is not str
            or not self.not_applicable_reason
        ):
            raise ValueError("unavailable evidence requires a fixed reason")
        super().__post_init__()


def _h7_objective_residuals(
    value: H7ObjectiveCovarianceEvaluation,
) -> tuple[H7ResidualRecord, ...]:
    optional = tuple(
        item
        for item in (value.evidence, value.posterior_kl)
        if type(item) is H7ResidualRecord
    )
    return (
        value.initial_joint_kl.residual,
        *(item.residual for item in value.local_terms),
        *(
            observation.residual
            for item in value.density_probe_evaluations
            for observation in item.observations
        ),
        *value.scorer_residuals,
        value.complete_local,
        value.complete_monolithic,
        value.p_density_shift,
        value.q_density_shift,
        value.log_ratio,
        value.entropy_shift,
        *optional,
    )


@dataclass(frozen=True)
class H7TrialResult(_H7IntegrityRecord):
    _integrity_field: ClassVar[str] = "trial_result_sha256"
    _hash_domain: ClassVar[str] = "vfe4.h7.trial-result.v1"

    spec: H7TrialSpec
    observed_predicate: H7ExpectedPredicate
    predicate_satisfied: bool
    logabsdet_measure_shift: float
    r_abs: H7ResidualRecord
    r_rel: H7ResidualRecord
    backward_by_operand: tuple[H7BackwardResidualRecord, ...]
    r_back_max: H7ResidualRecord
    residuals: tuple[H7ResidualRecord, ...]
    envelope: H7EnvelopeRecord
    law_pairs_by_recognition_family: Mapping[H7RecognitionFamily, H7LawPairSnapshot]
    objective_by_recognition_family: Mapping[
        H7RecognitionFamily, H7ObjectiveCovarianceEvaluation
    ]
    trial_result_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.spec) is not H7TrialSpec
            or self.observed_predicate != self.spec.expected_predicate
            or type(self.predicate_satisfied) is not bool
            or type(self.r_abs) is not H7ResidualRecord
            or type(self.r_rel) is not H7ResidualRecord
            or type(self.backward_by_operand) is not tuple
            or not self.backward_by_operand
            or any(
                type(item) is not H7BackwardResidualRecord
                for item in self.backward_by_operand
            )
            or type(self.r_back_max) is not H7ResidualRecord
            or self.r_back_max.category != "backward"
            or type(self.residuals) is not tuple
            or not self.residuals
            or any(type(item) is not H7ResidualRecord for item in self.residuals)
            or type(self.envelope) is not H7EnvelopeRecord
        ):
            raise ValueError("trial result evidence is incomplete")
        expected_back_max = max(item.value for item in self.backward_by_operand)
        if self.r_back_max.value != expected_back_max:
            raise ValueError("r_back_max must preserve the operand-wise maximum")
        if (
            self.r_abs.category != "absolute"
            or self.r_rel.category != "relative"
            or self.r_back_max.category != "backward"
            or len({item.operand_id for item in self.backward_by_operand})
            != len(self.backward_by_operand)
            or len({item.invariant_id for item in self.residuals})
            != len(self.residuals)
        ):
            raise ValueError("trial aggregate residual inventory is inconsistent")
        law_keys = tuple(self.law_pairs_by_recognition_family)
        objective_keys = tuple(self.objective_by_recognition_family)
        expected_keys: tuple[H7RecognitionFamily, ...] = (
            ("structured_full_block",)
            if self.spec.trial_id in H7_SCALAR_TRIAL_IDS
            else (
                "structured_full_block",
                "factorized_diagonal_within_fiber",
            )
        )
        if law_keys != expected_keys or objective_keys != expected_keys:
            raise ValueError("trial recognition-family inventory is not exact")
        if any(
            type(value) is not H7LawPairSnapshot
            for value in self.law_pairs_by_recognition_family.values()
        ) or any(
            type(value) is not H7ObjectiveCovarianceEvaluation
            for value in self.objective_by_recognition_family.values()
        ):
            raise ValueError("trial mappings must contain owned evidence records")
        for family, pair in self.law_pairs_by_recognition_family.items():
            if pair.action_sha256 != self.spec.action_sha256:
                raise ValueError("law pair action does not match trial action")
            if (
                pair.original.fixture_id != self.spec.fixture_id
                or pair.transformed.fixture_id != self.spec.fixture_id
            ):
                raise ValueError("law pair fixture does not match trial fixture")
            if pair.original.recognition.origin_family != family:
                raise ValueError("law pair origin family disagrees with its key")
            if (
                family == "factorized_diagonal_within_fiber"
                and pair.transformed.recognition.representation
                != "unrestricted_full_block_pushforward"
            ):
                raise ValueError(
                    "matrix factorized origin must publish unrestricted pushforward"
                )
        if (
            self.spec.role == "expected_negative"
            and self.observed_predicate != "decisive_outside_stabilizer_change"
        ):
            raise ValueError("expected-negative trial cannot claim covariance")
        backward_closed = self.r_back_max.passed and all(
            item.passed for item in self.backward_by_operand
        )
        objective_values = tuple(self.objective_by_recognition_family.values())
        if self.spec.role in ("scalar_regression", "positive_covariance"):
            derived_predicate = (
                self.envelope.passed
                and self.r_abs.passed
                and self.r_rel.passed
                and backward_closed
                and all(item.passed for item in self.residuals)
                and all(
                    residual.passed
                    for objective in objective_values
                    for residual in _h7_objective_residuals(objective)
                )
            )
        else:
            unchanged_categories = {
                "tensor",
                "law",
                "cocycle",
                "density",
                "jacobian",
                "source",
                "backward",
            }
            unchanged_residuals_closed = all(
                residual.passed
                for residual in self.residuals
                if residual.category in unchanged_categories
            )
            emission_changed = any(
                residual.category == "decoder" and not residual.passed
                for residual in self.residuals
            )
            objectives_changed = all(
                objective.initial_joint_kl.residual.passed
                and all(item.passed for item in objective.scorer_residuals)
                and objective.entropy_shift.passed
                and not objective.complete_local.passed
                and not objective.complete_monolithic.passed
                and not objective.log_ratio.passed
                and any(not item.residual.passed for item in objective.local_terms)
                for objective in objective_values
            )
            derived_predicate = (
                self.envelope.passed
                and backward_closed
                and unchanged_residuals_closed
                and emission_changed
                and objectives_changed
                and not self.r_abs.passed
                and not self.r_rel.passed
            )
        if self.predicate_satisfied != derived_predicate:
            raise ValueError(
                "predicate_satisfied disagrees with role-specific owned evidence"
            )
        super().__post_init__()


@dataclass(frozen=True)
class H7ControlResult(_H7IntegrityRecord):
    _integrity_field: ClassVar[str] = "control_sha256"
    _hash_domain: ClassVar[str] = "vfe4.h7.control-result.v1"

    control_id: H7ControlId
    target_invariant_id: str
    wrong_residual: float
    invariant_scale: float
    matching_correct_allowance: float
    decisiveness_limit: float
    detected: bool
    control_sha256: str

    def __post_init__(self) -> None:
        if self.control_id not in H7_CONTROL_IDS:
            raise ValueError("control_id is outside the frozen H7 inventory")
        _require_nonempty(self.target_invariant_id, "target_invariant_id")
        if (
            self.wrong_residual < 0.0
            or self.invariant_scale <= 0.0
            or self.matching_correct_allowance < 0.0
            or self.decisiveness_limit < 0.0
            or type(self.detected) is not bool
        ):
            raise ValueError("invalid H7 control values")
        expected_limit = max(
            100.0 * self.matching_correct_allowance,
            1e-8 * self.invariant_scale,
        )
        if self.decisiveness_limit != expected_limit:
            raise ValueError("control decisiveness limit is not frozen")
        if self.detected != (self.wrong_residual > self.decisiveness_limit):
            raise ValueError("control detection must be strict above the limit")
        super().__post_init__()


@dataclass(frozen=True)
class H7PredecessorReference(_H7IntegrityRecord):
    _integrity_field: ClassVar[str] = "reference_sha256"
    _hash_domain: ClassVar[str] = "vfe4.h7.predecessor-reference.v1"

    artifact_path: str
    git_head: str
    dirty_digest: str
    junit_sha256: str
    junit_path: str
    manifest_sha256: str
    payload_hashes: Mapping[str, str]
    ledger_path: str
    ledger_sha256: str
    reference_sha256: str

    def __post_init__(self) -> None:
        _require_nonempty(self.artifact_path, "artifact_path")
        _require_nonempty(self.junit_path, "junit_path")
        _require_nonempty(self.ledger_path, "ledger_path")
        if (
            type(self.git_head) is not str
            or len(self.git_head) not in (40, 64)
            or any(character not in _LOWER_HEX for character in self.git_head)
        ):
            raise ValueError("git_head must be a full lowercase Git object ID")
        _require_sha256(self.dirty_digest, "dirty_digest")
        for name in (
            "junit_sha256",
            "manifest_sha256",
            "ledger_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if not isinstance(self.payload_hashes, Mapping) or not self.payload_hashes:
            raise ValueError("predecessor payload_hashes must be nonempty")
        for key, digest in self.payload_hashes.items():
            _require_nonempty(key, "payload hash key")
            _require_sha256(digest, f"payload_hashes[{key}]")
        super().__post_init__()


@dataclass(frozen=True)
class H7GateEvaluation(_H7IntegrityRecord):
    _integrity_field: ClassVar[str] = "evaluation_sha256"
    _hash_domain: ClassVar[str] = "vfe4.h7.gate-evaluation.v1"

    result: H7GateResult
    validation_payload_canonical_json: bytes
    validation_payload_sha256: str
    fixture_set_sha256: str
    dependency_closure_sha256: str
    evaluation_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.result).__name__ != "H7GateResult"
            or type(self.result).__module__ != "vfe4.types.results"
        ):
            raise ValueError("gate evaluation requires results.py::H7GateResult")
        if type(self.validation_payload_canonical_json) is not bytes:
            raise ValueError("validation payload must be canonical JSON bytes")
        try:
            parsed = json.loads(self.validation_payload_canonical_json.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("validation payload is not UTF-8 JSON") from error
        expected_json = json.dumps(
            parsed,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        if expected_json != self.validation_payload_canonical_json:
            raise ValueError("validation payload bytes are not canonical")
        if (
            hashlib.sha256(self.validation_payload_canonical_json).hexdigest()
            != self.validation_payload_sha256
        ):
            raise ValueError("validation payload hash does not match bytes")
        for name in (
            "fixture_set_sha256",
            "dependency_closure_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        super().__post_init__()


@dataclass(frozen=True)
class H7PassOutcome(_H7IntegrityRecord):
    _integrity_field: ClassVar[str] = "outcome_sha256"
    _hash_domain: ClassVar[str] = "vfe4.h7.pass-outcome.v1"

    kind: Literal["PASS"]
    scalar_trial_ids: tuple[H7TrialId, H7TrialId]
    positive_trial_ids: tuple[H7TrialId, H7TrialId, H7TrialId, H7TrialId, H7TrialId]
    expected_negative_trial_id: H7TrialId
    control_ids: tuple[H7ControlId, ...]
    outcome_sha256: str

    def __post_init__(self) -> None:
        if (
            self.kind != "PASS"
            or self.scalar_trial_ids != H7_SCALAR_TRIAL_IDS
            or self.positive_trial_ids != H7_MATRIX_TRIAL_IDS[:-1]
            or self.expected_negative_trial_id != H7_MATRIX_TRIAL_IDS[-1]
            or self.control_ids != H7_CONTROL_IDS
        ):
            raise ValueError("PASS outcome inventory is not exact")
        super().__post_init__()


@dataclass(frozen=True)
class H7FailOutcome(_H7IntegrityRecord):
    _integrity_field: ClassVar[str] = "outcome_sha256"
    _hash_domain: ClassVar[str] = "vfe4.h7.fail-outcome.v1"

    kind: Literal["FAIL"]
    failed_invariant_ids: tuple[str, ...]
    expected_negative_false_acceptance: bool
    outcome_sha256: str

    def __post_init__(self) -> None:
        if (
            self.kind != "FAIL"
            or type(self.failed_invariant_ids) is not tuple
            or any(
                type(item) is not str or not item for item in self.failed_invariant_ids
            )
            or len(set(self.failed_invariant_ids)) != len(self.failed_invariant_ids)
            or type(self.expected_negative_false_acceptance) is not bool
            or (
                not self.failed_invariant_ids
                and not self.expected_negative_false_acceptance
            )
        ):
            raise ValueError("FAIL outcome requires a concrete refutation")
        super().__post_init__()


@dataclass(frozen=True)
class H7InconclusiveOutcome(_H7IntegrityRecord):
    _integrity_field: ClassVar[str] = "outcome_sha256"
    _hash_domain: ClassVar[str] = "vfe4.h7.inconclusive-outcome.v1"

    kind: Literal["INCONCLUSIVE"]
    obligations: tuple[str, ...]
    outcome_sha256: str

    def __post_init__(self) -> None:
        if (
            self.kind != "INCONCLUSIVE"
            or type(self.obligations) is not tuple
            or not self.obligations
            or any(type(item) is not str or not item for item in self.obligations)
            or len(set(self.obligations)) != len(self.obligations)
        ):
            raise ValueError("INCONCLUSIVE outcome requires unique obligations")
        super().__post_init__()


H7GateOutcome: TypeAlias = H7PassOutcome | H7FailOutcome | H7InconclusiveOutcome
