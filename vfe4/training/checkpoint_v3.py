"""Canonical H6 Prediction v3 checkpoint capture and exact hydration.

The v3 codec never serializes PyTorch object IDs.  Module state is keyed by
qualified parameter or buffer name, and AdamW groups/state are keyed by the
same stable parameter names.  All tensor payloads are canonical contiguous
CPU little-endian bytes before a checkpoint is accepted.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import stat
import sys
import threading
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum
from pathlib import Path
from types import CodeType, FunctionType, ModuleType
from typing import Literal

import torch
from torch import Tensor, nn

from vfe4.types.h6 import TrainingPhase, canonical_json_bytes
from vfe4.types.h6_prediction_v3 import (
    H6AttemptCursorV3,
    H6AttemptSpecV3,
    H6ObjectiveManifestV3,
    H6PredictionRuntimeIdentity,
    H6_CHECKPOINT_CODEC_SHA256,
    H6_DETERMINISTIC_POLICY_SHA256,
    H6_NO_COUNTER_CONSUMPTION_SHA256,
)
from .h6_engine_v3 import (
    H6DetachedBatchRecognitionSnapshotV3,
    H6DetachedRecognitionSnapshotV3,
)


_MAGIC = b"VFE4-H6-CHECKPOINT-V3\x00"
_TRAILER_BYTES = hashlib.sha256().digest_size
_MAX_HEADER_BYTES = 64 * 1024 * 1024
_LOWER_HEX = frozenset("0123456789abcdef")
_MAX_RECORDS = 1_000_000
_MAX_APPLICATION_SOURCE_BYTES = 16 * 1024 * 1024
_MAX_REFERENCED_APPLICATION_OBJECTS = 4096
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MODULE_ROLES = frozenset({"module_parameter", "module_buffer"})
_TENSOR_ROLES = _MODULE_ROLES | {
    "optimizer_state",
    "recognition_snapshot",
}
_SUPPORTED_DTYPES: dict[str, torch.dtype] = {
    "bool": torch.bool,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}
_DTYPE_NAMES = {value: name for name, value in _SUPPORTED_DTYPES.items()}
_ADAMW_GROUP_KEYS = frozenset(
    {
        "lr",
        "betas",
        "eps",
        "weight_decay",
        "amsgrad",
        "maximize",
        "foreach",
        "capturable",
        "differentiable",
        "fused",
        "decoupled_weight_decay",
        "initial_lr",
    }
)
_ADAMW_BASE_STATE = frozenset({"step", "exp_avg", "exp_avg_sq"})
_FACTORY_AUTHORITY_REGISTRY_LOCK = threading.Lock()
_FACTORY_AUTHORITY_REGISTRY: dict[object, tuple[int, object]] = {}
_MODULE_HOOK_MAP_NAMES = (
    "_backward_pre_hooks",
    "_backward_hooks",
    "_forward_hooks",
    "_forward_hooks_always_called",
    "_forward_hooks_with_kwargs",
    "_forward_pre_hooks",
    "_forward_pre_hooks_with_kwargs",
    "_state_dict_hooks",
    "_state_dict_pre_hooks",
    "_load_state_dict_pre_hooks",
    "_load_state_dict_post_hooks",
)
_MODULE_INTERNAL_ATTRIBUTE_NAMES = frozenset(
    {
        "training",
        "_parameters",
        "_buffers",
        "_non_persistent_buffers_set",
        "_modules",
        "_backward_pre_hooks",
        "_backward_hooks",
        "_is_full_backward_hook",
        "_forward_hooks",
        "_forward_hooks_with_kwargs",
        "_forward_hooks_always_called",
        "_forward_pre_hooks",
        "_forward_pre_hooks_with_kwargs",
        "_state_dict_hooks",
        "_state_dict_pre_hooks",
        "_load_state_dict_pre_hooks",
        "_load_state_dict_post_hooks",
    }
)


def _contextmanager_wrapper_probe() -> object:
    yield None


_TRUSTED_CONTEXTMANAGER_WRAPPER_CODE = contextmanager(
    _contextmanager_wrapper_probe
).__code__
del _contextmanager_wrapper_probe


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _owned_hash(domain: str, payload: object) -> str:
    return hashlib.sha256(
        domain.encode("ascii") + b"\x00" + canonical_json_bytes(payload)
    ).hexdigest()


def _require_name(value: object, name: str) -> str:
    if type(value) is not str or not value or "\x00" in value:
        raise ValueError(f"{name} must be a nonempty string without NUL")
    return value


def _require_root_name(value: object, name: str) -> str:
    checked = _require_name(value, name)
    if "." in checked:
        raise ValueError(f"{name} cannot contain a period")
    return checked


def _validate_unique_names(names: tuple[str, ...], *, label: str) -> None:
    if len(set(names)) != len(names):
        raise ValueError(f"{label} contains duplicate names")
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError(f"{label} contains case-colliding names")


def _named_items(
    value: object,
    *,
    label: str,
) -> tuple[tuple[str, object], ...]:
    if isinstance(value, Mapping):
        raw_items = tuple(value.items())
    elif type(value) is tuple:
        raw_items = value
    else:
        raise ValueError(f"{label} must be a mapping or tuple of named items")
    items: list[tuple[str, object]] = []
    for item in raw_items:
        if type(item) is not tuple or len(item) != 2:
            raise ValueError(f"{label} entries must be exact name/value pairs")
        name, owned = item
        items.append((_require_root_name(name, f"{label} name"), owned))
    names = tuple(name for name, _ in items)
    _validate_unique_names(names, label=label)
    return tuple(items)


def _dtype_name(tensor: Tensor) -> str:
    try:
        return _DTYPE_NAMES[tensor.dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported checkpoint tensor dtype {tensor.dtype}") from exc


def _tensor_byte_length(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    count = math.prod(shape)
    return count * torch.empty((), dtype=dtype).element_size()


def _little_endian_bytes(tensor: Tensor) -> bytes:
    if tensor.layout is not torch.strided:
        raise ValueError("sparse or non-strided checkpoint tensors are unsupported")
    if tensor.is_quantized:
        raise ValueError("quantized checkpoint tensors are unsupported")
    if tensor.device.type == "meta":
        raise ValueError("meta checkpoint tensors have no restorable bytes")
    _dtype_name(tensor)
    cpu = tensor.detach().to(device="cpu").contiguous()
    if cpu.is_floating_point() and not bool(torch.isfinite(cpu).all()):
        raise ValueError("checkpoint tensors must contain only finite values")
    raw = bytes(cpu.reshape(-1).view(torch.uint8).tolist())
    if sys.byteorder == "little" or cpu.element_size() == 1:
        return raw
    width = cpu.element_size()
    return b"".join(
        raw[offset : offset + width][::-1] for offset in range(0, len(raw), width)
    )


def _native_endian_bytes(raw: bytes, *, width: int) -> bytes:
    if sys.byteorder == "little" or width == 1:
        return raw
    return b"".join(
        raw[offset : offset + width][::-1] for offset in range(0, len(raw), width)
    )


def _storage_identity(tensor: Tensor) -> tuple[str, int | None, int]:
    storage = tensor.untyped_storage()
    storage_id = getattr(storage, "_cdata", None)
    if type(storage_id) is not int:
        storage_id = storage.data_ptr()
    return tensor.device.type, tensor.device.index, storage_id


def _register_storage(
    tensor: Tensor,
    *,
    name: str,
    aliases: dict[tuple[str, int | None, int], str],
) -> None:
    identity = _storage_identity(tensor)
    previous = aliases.get(identity)
    if previous is not None:
        raise ValueError(
            f"checkpoint rejects shared-storage alias between {previous!r} and {name!r}"
        )
    aliases[identity] = name


def _freeze_scalar(value: object, *, name: str) -> object:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value
    if type(value) is tuple:
        return tuple(_freeze_scalar(item, name=f"{name} item") for item in value)
    raise ValueError(f"{name} uses an unsupported scalar value")


def _thaw_canonical_float(value: object, *, name: str) -> float:
    if type(value) is not str:
        raise ValueError(f"{name} must use one canonical hexadecimal float")
    try:
        result = float.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{name} must use one canonical hexadecimal float") from exc
    if not math.isfinite(result) or result.hex() != value:
        raise ValueError(f"{name} has a noncanonical float encoding")
    return result


@dataclass(frozen=True, slots=True)
class H6TensorRecordV3:
    """One canonical tensor payload with explicit semantic ownership."""

    role: Literal[
        "module_parameter",
        "module_buffer",
        "optimizer_state",
        "recognition_snapshot",
    ]
    name: str
    state_name: str | None
    dtype: str
    shape: tuple[int, ...]
    layout: Literal["contiguous-row-major"]
    byte_order: Literal["little"]
    byte_length: int
    raw_bytes_sha256: str
    _raw_bytes: bytes = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.role not in _TENSOR_ROLES:
            raise ValueError("unsupported checkpoint tensor role")
        _require_name(self.name, "tensor name")
        if self.role == "optimizer_state":
            _require_name(self.state_name, "optimizer tensor state name")
        elif self.state_name is not None:
            raise ValueError("module tensor records cannot carry state_name")
        dtype = _SUPPORTED_DTYPES.get(self.dtype)
        if dtype is None:
            raise ValueError("checkpoint tensor record has unsupported dtype")
        if type(self.shape) is not tuple or any(
            type(size) is not int or size < 0 for size in self.shape
        ):
            raise ValueError("checkpoint tensor shape must be nonnegative integers")
        if self.layout != "contiguous-row-major" or self.byte_order != "little":
            raise ValueError("checkpoint tensor encoding identity is stale")
        if type(self._raw_bytes) is not bytes:
            raise ValueError("checkpoint tensor bytes must be immutable")
        if type(self.byte_length) is not int or self.byte_length < 0:
            raise ValueError("checkpoint tensor byte length must be nonnegative")
        if self.byte_length != len(self._raw_bytes):
            raise ValueError("checkpoint tensor byte length mismatch")
        if self.byte_length != _tensor_byte_length(self.shape, dtype):
            raise ValueError("checkpoint tensor shape/dtype byte length mismatch")
        _require_sha256(self.raw_bytes_sha256, "raw_bytes_sha256")
        if hashlib.sha256(self._raw_bytes).hexdigest() != self.raw_bytes_sha256:
            raise ValueError("checkpoint tensor raw-byte digest mismatch")

    @classmethod
    def capture(
        cls,
        *,
        role: Literal[
            "module_parameter",
            "module_buffer",
            "optimizer_state",
            "recognition_snapshot",
        ],
        name: str,
        tensor: Tensor,
        state_name: str | None = None,
        aliases: dict[tuple[str, int | None, int], str],
    ) -> H6TensorRecordV3:
        if not isinstance(tensor, Tensor):
            raise ValueError("checkpoint state values must be tensors")
        if tensor.layout is not torch.strided:
            raise ValueError("sparse or non-strided checkpoint tensors are unsupported")
        if tensor.is_quantized:
            raise ValueError("quantized checkpoint tensors are unsupported")
        if tensor.device.type == "meta":
            raise ValueError("meta checkpoint tensors have no restorable bytes")
        _register_storage(tensor, name=name, aliases=aliases)
        dtype = _dtype_name(tensor)
        if role in _MODULE_ROLES and tensor.is_floating_point():
            if tensor.dtype is not torch.float64:
                raise ValueError("canonical module floating state must be float64")
        raw = _little_endian_bytes(tensor)
        return cls(
            role=role,
            name=name,
            state_name=state_name,
            dtype=dtype,
            shape=tuple(tensor.shape),
            layout="contiguous-row-major",
            byte_order="little",
            byte_length=len(raw),
            raw_bytes_sha256=hashlib.sha256(raw).hexdigest(),
            _raw_bytes=raw,
        )

    def manifest_payload(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "role": self.role,
            "name": self.name,
            "state_name": self.state_name,
            "dtype": self.dtype,
            "shape": self.shape,
            "layout": self.layout,
            "byte_order": self.byte_order,
            "byte_length": self.byte_length,
            "raw_bytes_sha256": self.raw_bytes_sha256,
        }

    def raw_bytes(self) -> bytes:
        self.__post_init__()
        return bytes(self._raw_bytes)

    def decode_cpu(self) -> Tensor:
        self.__post_init__()
        dtype = _SUPPORTED_DTYPES[self.dtype]
        if self.byte_length == 0:
            return torch.empty(self.shape, dtype=dtype, device="cpu")
        width = torch.empty((), dtype=dtype).element_size()
        native = _native_endian_bytes(self._raw_bytes, width=width)
        flat = torch.frombuffer(bytearray(native), dtype=dtype).clone()
        return flat.reshape(self.shape).contiguous()


@dataclass(frozen=True, slots=True)
class H6AdamWGroupV3:
    """One stable optimizer group with ordered qualified parameter names."""

    name: str
    parameter_names: tuple[str, ...]
    hyperparameters: tuple[tuple[str, object], ...]

    def __post_init__(self) -> None:
        _require_name(self.name, "optimizer group name")
        if type(self.parameter_names) is not tuple or not self.parameter_names:
            raise ValueError("AdamW group requires ordered parameter names")
        for parameter_name in self.parameter_names:
            _require_name(parameter_name, "optimizer parameter name")
        _validate_unique_names(
            self.parameter_names,
            label="AdamW group parameter inventory",
        )
        if type(self.hyperparameters) is not tuple:
            raise ValueError("AdamW hyperparameters must be a tuple")
        keys: list[str] = []
        for item in self.hyperparameters:
            if type(item) is not tuple or len(item) != 2:
                raise ValueError("AdamW hyperparameters must be name/value pairs")
            key, value = item
            _require_name(key, "AdamW hyperparameter name")
            if key not in _ADAMW_GROUP_KEYS:
                raise ValueError(f"unknown AdamW group hyperparameter {key!r}")
            if _freeze_scalar(value, name=f"AdamW hyperparameter {key}") != value:
                raise ValueError("AdamW hyperparameter is not canonical")
            keys.append(key)
        if tuple(keys) != tuple(sorted(keys)):
            raise ValueError("AdamW hyperparameters must be name-sorted")
        _validate_unique_names(tuple(keys), label="AdamW hyperparameters")

    def canonical_payload(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "name": self.name,
            "parameter_names": self.parameter_names,
            "hyperparameters": self.hyperparameters,
        }


@dataclass(frozen=True, slots=True)
class H6AdamWParameterStateV3:
    """AdamW state keyed by one stable qualified parameter name."""

    parameter_name: str
    tensors: tuple[H6TensorRecordV3, ...]
    scalars: tuple[tuple[str, int | float], ...]

    def __post_init__(self) -> None:
        _require_name(self.parameter_name, "AdamW state parameter name")
        if type(self.tensors) is not tuple or type(self.scalars) is not tuple:
            raise ValueError("AdamW parameter state must use tuples")
        tensor_names: list[str] = []
        for tensor in self.tensors:
            if type(tensor) is not H6TensorRecordV3:
                raise ValueError("AdamW tensor state record has the wrong type")
            tensor.__post_init__()
            if tensor.role != "optimizer_state" or tensor.state_name is None:
                raise ValueError("AdamW tensor state has the wrong role")
            tensor_names.append(tensor.state_name)
        scalar_names: list[str] = []
        for item in self.scalars:
            if type(item) is not tuple or len(item) != 2:
                raise ValueError("AdamW scalar states must be name/value pairs")
            name, value = item
            _require_name(name, "AdamW scalar state name")
            if type(value) not in (int, float) or isinstance(value, bool):
                raise ValueError("AdamW scalar state must be int or float")
            if type(value) is float and not math.isfinite(value):
                raise ValueError("AdamW scalar state must be finite")
            scalar_names.append(name)
        all_names = tuple(tensor_names + scalar_names)
        _validate_unique_names(all_names, label="AdamW state")
        if tuple(tensor_names) != tuple(sorted(tensor_names)):
            raise ValueError("AdamW tensor states must be name-sorted")
        if tuple(scalar_names) != tuple(sorted(scalar_names)):
            raise ValueError("AdamW scalar states must be name-sorted")

    def canonical_payload(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "parameter_name": self.parameter_name,
            "tensors": tuple(tensor.manifest_payload() for tensor in self.tensors),
            "scalars": self.scalars,
        }


@dataclass(frozen=True, slots=True)
class H6AdamWRecordV3:
    """A complete AdamW optimizer without ephemeral parameter IDs."""

    name: str
    groups: tuple[H6AdamWGroupV3, ...]
    states: tuple[H6AdamWParameterStateV3, ...]

    def __post_init__(self) -> None:
        _require_root_name(self.name, "optimizer name")
        if type(self.groups) is not tuple or not self.groups:
            raise ValueError("checkpoint AdamW record requires groups")
        if type(self.states) is not tuple:
            raise ValueError("checkpoint AdamW states must be a tuple")
        expected_group_names = tuple(
            f"{self.name}.group.{index:06d}" for index in range(len(self.groups))
        )
        observed_group_names = tuple(group.name for group in self.groups)
        if observed_group_names != expected_group_names:
            raise ValueError("AdamW group names/order are not canonical")
        parameter_names: list[str] = []
        for group in self.groups:
            if type(group) is not H6AdamWGroupV3:
                raise ValueError("AdamW group record has the wrong type")
            group.__post_init__()
            if any(
                parameter_name.split(".", 1)[0] != self.name
                for parameter_name in group.parameter_names
            ):
                raise ValueError("AdamW optimizer may bind only same-root parameters")
            parameter_names.extend(group.parameter_names)
        _validate_unique_names(
            tuple(parameter_names),
            label="AdamW optimizer parameter inventory",
        )
        if len(self.states) != len(parameter_names):
            raise ValueError("AdamW state inventory is incomplete")
        state_parameter_names = tuple(state.parameter_name for state in self.states)
        if state_parameter_names != tuple(sorted(parameter_names)):
            raise ValueError("AdamW states must cover sorted group parameters")
        for state in self.states:
            if type(state) is not H6AdamWParameterStateV3:
                raise ValueError("AdamW parameter state has the wrong type")
            state.__post_init__()

    def canonical_payload(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "name": self.name,
            "groups": tuple(group.canonical_payload() for group in self.groups),
            "states": tuple(state.canonical_payload() for state in self.states),
        }


def _typed_record_payload(
    record: (
        H6AttemptSpecV3
        | H6AttemptCursorV3
        | H6ObjectiveManifestV3
        | H6PredictionRuntimeIdentity
    ),
) -> dict[str, object]:
    payload = record.canonical_payload()
    if type(record) is H6AttemptSpecV3:
        payload["attempt_spec_sha256"] = record.attempt_spec_sha256
    elif type(record) is H6AttemptCursorV3:
        payload["cursor_sha256"] = record.cursor_sha256
    elif type(record) is H6ObjectiveManifestV3:
        payload["objective_manifest_sha256"] = record.objective_manifest_sha256
    elif type(record) is H6PredictionRuntimeIdentity:
        payload["runtime_identity_sha256"] = record.runtime_identity_sha256
    else:
        raise ValueError("unsupported checkpoint v3 typed record")
    return payload


def _snapshot_tensor_records(
    snapshot: H6DetachedBatchRecognitionSnapshotV3,
) -> tuple[tuple[H6TensorRecordV3, ...], ...]:
    if type(snapshot) is not H6DetachedBatchRecognitionSnapshotV3:
        raise ValueError("checkpoint snapshot must be an exact detached batch law")
    snapshot.__post_init__()
    aliases: dict[tuple[str, int | None, int], str] = {}
    return tuple(
        tuple(
            H6TensorRecordV3.capture(
                role="recognition_snapshot",
                name=f"snapshot.example.{example_ordinal}.{name}",
                tensor=tensor,
                aliases=aliases,
            )
            for name, tensor in zip(
                state.names,
                state._tensors,
                strict=True,
            )
        )
        for example_ordinal, state in enumerate(snapshot.states)
    )


def _snapshot_payload(
    snapshot: H6DetachedBatchRecognitionSnapshotV3 | None,
) -> dict[str, object] | None:
    if snapshot is None:
        return None
    records_by_state = _snapshot_tensor_records(snapshot)
    return {
        "authority_sha256": snapshot.authority_sha256,
        "attempt_spec_sha256": snapshot.attempt_spec_sha256,
        "endpoint_config_sha256": snapshot.endpoint_config_sha256,
        "post_recognition_cursor_sha256": (snapshot.post_recognition_cursor_sha256),
        "pass_index": snapshot.pass_index,
        "batch_index": snapshot.batch_index,
        "recognition_update_count": snapshot.recognition_update_count,
        "receiver_count": snapshot.receiver_count,
        "active_target_counts": snapshot.active_target_counts,
        "active_receiver_masks": snapshot.active_receiver_masks,
        "live_batch_state_sha256": snapshot.live_batch_state_sha256,
        "names": snapshot.names,
        "snapshot_sha256": snapshot.snapshot_sha256,
        "states": tuple(
            {
                "attempt_spec_sha256": state.attempt_spec_sha256,
                "endpoint_config_sha256": state.endpoint_config_sha256,
                "post_recognition_cursor_sha256": (
                    state.post_recognition_cursor_sha256
                ),
                "pass_index": state.pass_index,
                "batch_index": state.batch_index,
                "recognition_update_count": (state.recognition_update_count),
                "receiver_count": state.receiver_count,
                "state_categorical_enabled": (state.state_categorical_enabled),
                "model_categorical_enabled": (state.model_categorical_enabled),
                "state_categorical_supports": (state.state_categorical_supports),
                "model_categorical_supports": (state.model_categorical_supports),
                "receiver_components": state.receiver_components,
                "names": state.names,
                "context_sha256": state.context_sha256,
                "recognition_state_sha256": (state.recognition_state_sha256),
                "source_model_sha256": state.source_model_sha256,
                "law_sha256": state.law_sha256,
                "live_state_sha256": state.live_state_sha256,
                "snapshot_sha256": state.snapshot_sha256,
                "tensors": tuple(
                    record.manifest_payload()
                    for record in records_by_state[example_ordinal]
                ),
            }
            for example_ordinal, state in enumerate(snapshot.states)
        ),
    }


def _checkpoint_payload(
    *,
    attempt_spec: H6AttemptSpecV3,
    cursor: H6AttemptCursorV3,
    objective_manifest: H6ObjectiveManifestV3,
    runtime_identity: H6PredictionRuntimeIdentity,
    deterministic_policy_sha256: str,
    module_tensors: tuple[H6TensorRecordV3, ...],
    optimizers: tuple[H6AdamWRecordV3, ...],
    detached_batch_snapshot: (H6DetachedBatchRecognitionSnapshotV3 | None),
) -> dict[str, object]:
    return {
        "checkpoint_schema": "h6-checkpoint-v3",
        "checkpoint_codec_sha256": H6_CHECKPOINT_CODEC_SHA256,
        "attempt_spec": _typed_record_payload(attempt_spec),
        "cursor": _typed_record_payload(cursor),
        "objective_manifest": _typed_record_payload(objective_manifest),
        "runtime_identity": _typed_record_payload(runtime_identity),
        "deterministic_policy_sha256": deterministic_policy_sha256,
        "module_tensors": tuple(tensor.manifest_payload() for tensor in module_tensors),
        "optimizers": tuple(optimizer.canonical_payload() for optimizer in optimizers),
        "detached_batch_snapshot": _snapshot_payload(detached_batch_snapshot),
    }


@dataclass(frozen=True, slots=True)
class H6CheckpointV3:
    """An immutable, self-validating H6 v3 checkpoint payload."""

    checkpoint_schema: Literal["h6-checkpoint-v3"]
    attempt_spec: H6AttemptSpecV3
    cursor: H6AttemptCursorV3
    objective_manifest: H6ObjectiveManifestV3
    runtime_identity: H6PredictionRuntimeIdentity
    deterministic_policy_sha256: str
    module_tensors: tuple[H6TensorRecordV3, ...]
    optimizers: tuple[H6AdamWRecordV3, ...]
    detached_batch_snapshot: H6DetachedBatchRecognitionSnapshotV3 | None
    checkpoint_sha256: str

    def __post_init__(self) -> None:
        if self.checkpoint_schema != "h6-checkpoint-v3":
            raise ValueError("unsupported H6 checkpoint v3 schema")
        for record, expected_type in (
            (self.attempt_spec, H6AttemptSpecV3),
            (self.cursor, H6AttemptCursorV3),
            (self.objective_manifest, H6ObjectiveManifestV3),
            (self.runtime_identity, H6PredictionRuntimeIdentity),
        ):
            if type(record) is not expected_type:
                raise ValueError("checkpoint contains a non-v3 typed record")
            record.__post_init__()
        if self.deterministic_policy_sha256 != H6_DETERMINISTIC_POLICY_SHA256:
            raise ValueError("checkpoint deterministic policy identity is stale")
        if self.attempt_spec.checkpoint_codec_sha256 != H6_CHECKPOINT_CODEC_SHA256:
            raise ValueError("attempt checkpoint codec identity is stale")
        if (
            self.attempt_spec.runtime_identity_sha256
            != self.runtime_identity.runtime_identity_sha256
        ):
            raise ValueError("attempt and checkpoint runtime identities disagree")
        if (
            self.cursor.attempt_spec_sha256 != self.attempt_spec.attempt_spec_sha256
            or self.objective_manifest.attempt_spec_sha256
            != self.attempt_spec.attempt_spec_sha256
            or self.objective_manifest.endpoint_config_sha256
            != self.attempt_spec.endpoint_config_sha256
            or self.objective_manifest.objective_kind
            != self.attempt_spec.objective_kind
            or self.objective_manifest.recognition_estimator_sha256
            != self.attempt_spec.recognition_estimator_sha256
        ):
            raise ValueError("checkpoint records do not bind one exact attempt")
        is_cross_entropy = self.attempt_spec.objective_kind == "cross_entropy"
        if is_cross_entropy and (
            self.cursor.draw_block != 0
            or self.cursor.counter_consumption_sha256
            != H6_NO_COUNTER_CONSUMPTION_SHA256
            or self.objective_manifest.counter_consumption_sha256
            != H6_NO_COUNTER_CONSUMPTION_SHA256
        ):
            raise ValueError(
                "cross-entropy checkpoint requires zero counter consumption"
            )
        if (
            self.objective_manifest.counter_consumption_sha256
            != self.cursor.counter_consumption_sha256
        ):
            raise ValueError(
                "checkpoint objective/cursor counter consumption disagrees"
            )
        if self.detached_batch_snapshot is not None:
            if (
                type(self.detached_batch_snapshot)
                is not H6DetachedBatchRecognitionSnapshotV3
            ):
                raise ValueError("checkpoint detached snapshot has the wrong type")
            self.detached_batch_snapshot.__post_init__()
            if (
                self.cursor.next_phase is not TrainingPhase.MODEL_ADAMW
                or self.detached_batch_snapshot.attempt_spec_sha256
                != self.attempt_spec.attempt_spec_sha256
                or self.detached_batch_snapshot.endpoint_config_sha256
                != self.attempt_spec.endpoint_config_sha256
                or self.detached_batch_snapshot.post_recognition_cursor_sha256
                != self.cursor.cursor_sha256
            ):
                raise ValueError(
                    "checkpoint detached batch law left its model boundary"
                )
        expected_objective_phase = {
            TrainingPhase.MODEL_CE_ADAMW: TrainingPhase.MODEL_CE_ADAMW,
            TrainingPhase.RECOGNITION_ADAMW: TrainingPhase.MODEL_ADAMW,
            TrainingPhase.IMMUTABLE_DETACHED_SNAPSHOT: (
                TrainingPhase.RECOGNITION_ADAMW
            ),
            TrainingPhase.MODEL_ADAMW: TrainingPhase.RECOGNITION_ADAMW,
        }[self.cursor.next_phase]
        if (
            self.objective_manifest.phase is not expected_objective_phase
            or is_cross_entropy
            != (self.cursor.next_phase is TrainingPhase.MODEL_CE_ADAMW)
        ):
            raise ValueError("checkpoint objective/cursor phase transition is invalid")
        if (
            type(self.module_tensors) is not tuple
            or not self.module_tensors
            or type(self.optimizers) is not tuple
            or not self.optimizers
        ):
            raise ValueError("checkpoint requires module and optimizer records")
        if len(self.module_tensors) > _MAX_RECORDS:
            raise ValueError("checkpoint module inventory is unbounded")
        module_names: list[str] = []
        for tensor in self.module_tensors:
            if type(tensor) is not H6TensorRecordV3:
                raise ValueError("module tensor record has the wrong type")
            tensor.__post_init__()
            if tensor.role not in _MODULE_ROLES:
                raise ValueError("module inventory contains optimizer state")
            module_names.append(tensor.name)
        if tuple(module_names) != tuple(sorted(module_names)):
            raise ValueError("module tensor records must be name-sorted")
        _validate_unique_names(
            tuple(module_names),
            label="checkpoint module tensor inventory",
        )
        optimizer_names = tuple(optimizer.name for optimizer in self.optimizers)
        if optimizer_names != tuple(sorted(optimizer_names)):
            raise ValueError("optimizer records must be name-sorted")
        _validate_unique_names(
            optimizer_names,
            label="checkpoint optimizer inventory",
        )
        optimizer_tensor_names: list[str] = []
        for optimizer in self.optimizers:
            if type(optimizer) is not H6AdamWRecordV3:
                raise ValueError("optimizer record has the wrong type")
            optimizer.__post_init__()
            optimizer_tensor_names.extend(
                tensor.name for state in optimizer.states for tensor in state.tensors
            )
        module_roots = tuple(
            sorted({module_name.split(".", 1)[0] for module_name in module_names})
        )
        attempt_has_recognition = (
            self.attempt_spec.recognition_factory_sha256 is not None
        )
        if attempt_has_recognition == is_cross_entropy:
            raise ValueError("checkpoint objective/recognition topology is invalid")
        expected_roots = (
            ("model", "recognition") if attempt_has_recognition else ("model",)
        )
        if module_roots != expected_roots or optimizer_names != expected_roots:
            raise ValueError("checkpoint module/optimizer root inventory is invalid")
        _validate_unique_names(
            tuple(module_names + optimizer_tensor_names),
            label="checkpoint tensor inventory",
        )
        expected = _owned_hash(
            "vfe4.h6.checkpoint.v3",
            _checkpoint_payload(
                attempt_spec=self.attempt_spec,
                cursor=self.cursor,
                objective_manifest=self.objective_manifest,
                runtime_identity=self.runtime_identity,
                deterministic_policy_sha256=(self.deterministic_policy_sha256),
                module_tensors=self.module_tensors,
                optimizers=self.optimizers,
                detached_batch_snapshot=self.detached_batch_snapshot,
            ),
        )
        if self.checkpoint_sha256 != expected:
            raise ValueError("checkpoint v3 digest mismatch")

    def canonical_payload(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **_checkpoint_payload(
                attempt_spec=self.attempt_spec,
                cursor=self.cursor,
                objective_manifest=self.objective_manifest,
                runtime_identity=self.runtime_identity,
                deterministic_policy_sha256=(self.deterministic_policy_sha256),
                module_tensors=self.module_tensors,
                optimizers=self.optimizers,
                detached_batch_snapshot=self.detached_batch_snapshot,
            ),
            "checkpoint_sha256": self.checkpoint_sha256,
        }

    def _ordered_tensor_records(self) -> tuple[H6TensorRecordV3, ...]:
        optimizer_tensors = tuple(
            tensor
            for optimizer in self.optimizers
            for state in optimizer.states
            for tensor in state.tensors
        )
        snapshot_tensors = (
            ()
            if self.detached_batch_snapshot is None
            else tuple(
                record
                for records in _snapshot_tensor_records(self.detached_batch_snapshot)
                for record in records
            )
        )
        return self.module_tensors + optimizer_tensors + snapshot_tensors

    def to_bytes(self) -> bytes:
        """Encode one stable manifest, raw tensor stream, and integrity trailer."""

        self.__post_init__()
        header = canonical_json_bytes(self.canonical_payload())
        payload = b"".join(
            tensor.raw_bytes() for tensor in self._ordered_tensor_records()
        )
        body = _MAGIC + len(header).to_bytes(8, "little") + header + payload
        return body + hashlib.sha256(body).digest()


def _canonical_header(raw: bytes) -> dict[str, object]:
    def reject_duplicates(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("checkpoint header contains duplicate JSON keys")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("checkpoint header is not canonical JSON") from exc
    if type(value) is not dict or canonical_json_bytes(value) != raw:
        raise ValueError("checkpoint header is not one canonical JSON object")
    return value


def _exact_object(
    value: object,
    *,
    keys: frozenset[str],
    label: str,
) -> dict[str, object]:
    if type(value) is not dict or frozenset(value) != keys:
        raise ValueError(f"{label} has missing or unknown fields")
    return value


def _json_tuple(value: object, *, label: str) -> tuple[object, ...]:
    if type(value) is not list:
        raise ValueError(f"{label} must be a JSON array")
    return tuple(
        _json_tuple(item, label=f"{label} item") if type(item) is list else item
        for item in value
    )


def _decode_typed_records(
    header: dict[str, object],
) -> tuple[
    H6AttemptSpecV3,
    H6AttemptCursorV3,
    H6ObjectiveManifestV3,
    H6PredictionRuntimeIdentity,
]:
    attempt_payload = _exact_object(
        header["attempt_spec"],
        keys=frozenset(H6AttemptSpecV3.__dataclass_fields__),
        label="checkpoint attempt spec",
    )
    attempt = H6AttemptSpecV3(**attempt_payload)  # type: ignore[arg-type]

    cursor_payload = _exact_object(
        header["cursor"],
        keys=frozenset(H6AttemptCursorV3.__dataclass_fields__),
        label="checkpoint cursor",
    )
    try:
        cursor_phase = TrainingPhase(cursor_payload["next_phase"])
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint cursor phase is unsupported") from exc
    cursor = H6AttemptCursorV3(
        **{
            **cursor_payload,
            "next_phase": cursor_phase,
        }  # type: ignore[arg-type]
    )

    objective_payload = _exact_object(
        header["objective_manifest"],
        keys=frozenset(H6ObjectiveManifestV3.__dataclass_fields__),
        label="checkpoint objective manifest",
    )
    try:
        objective_phase = TrainingPhase(objective_payload["phase"])
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint objective phase is unsupported") from exc
    raw_bindings = objective_payload["ordered_factor_bindings"]
    if type(raw_bindings) is not list:
        raise ValueError("checkpoint objective bindings must be an array")
    bindings: list[tuple[str, int, str]] = []
    for binding in raw_bindings:
        if type(binding) is not list or len(binding) != 3:
            raise ValueError("checkpoint objective binding is malformed")
        partition, receiver_t, digest = binding
        bindings.append((partition, receiver_t, digest))
    objective = H6ObjectiveManifestV3(
        **{
            **objective_payload,
            "phase": objective_phase,
            "ordered_factor_bindings": tuple(bindings),
        }  # type: ignore[arg-type]
    )

    runtime_payload = _exact_object(
        header["runtime_identity"],
        keys=frozenset(H6PredictionRuntimeIdentity.__dataclass_fields__),
        label="checkpoint runtime identity",
    )
    raw_capability = runtime_payload["cuda_compute_capability"]
    if type(raw_capability) is not list or len(raw_capability) != 2:
        raise ValueError("checkpoint CUDA capability is malformed")
    runtime = H6PredictionRuntimeIdentity(
        **{
            **runtime_payload,
            "cuda_compute_capability": tuple(raw_capability),
        }  # type: ignore[arg-type]
    )
    return attempt, cursor, objective, runtime


def _decode_tensor_record(
    manifest: object,
    *,
    payload: bytes,
    offset: int,
) -> tuple[H6TensorRecordV3, int]:
    item = _exact_object(
        manifest,
        keys=frozenset(
            {
                "role",
                "name",
                "state_name",
                "dtype",
                "shape",
                "layout",
                "byte_order",
                "byte_length",
                "raw_bytes_sha256",
            }
        ),
        label="checkpoint tensor manifest",
    )
    byte_length = item["byte_length"]
    if type(byte_length) is not int or byte_length < 0:
        raise ValueError("checkpoint tensor byte length is invalid")
    end = offset + byte_length
    if end < offset or end > len(payload):
        raise ValueError("checkpoint tensor payload is truncated")
    shape = _json_tuple(item["shape"], label="checkpoint tensor shape")
    record = H6TensorRecordV3(
        role=item["role"],  # type: ignore[arg-type]
        name=item["name"],  # type: ignore[arg-type]
        state_name=item["state_name"],  # type: ignore[arg-type]
        dtype=item["dtype"],  # type: ignore[arg-type]
        shape=shape,  # type: ignore[arg-type]
        layout=item["layout"],  # type: ignore[arg-type]
        byte_order=item["byte_order"],  # type: ignore[arg-type]
        byte_length=byte_length,
        raw_bytes_sha256=item["raw_bytes_sha256"],  # type: ignore[arg-type]
        _raw_bytes=bytes(payload[offset:end]),
    )
    return record, end


def _decode_detached_batch_snapshot(
    value: object,
    *,
    payload: bytes,
    offset: int,
) -> tuple[H6DetachedBatchRecognitionSnapshotV3 | None, int]:
    if value is None:
        return None, offset
    item = _exact_object(
        value,
        keys=frozenset(
            {
                "authority_sha256",
                "attempt_spec_sha256",
                "endpoint_config_sha256",
                "post_recognition_cursor_sha256",
                "pass_index",
                "batch_index",
                "recognition_update_count",
                "receiver_count",
                "active_target_counts",
                "active_receiver_masks",
                "live_batch_state_sha256",
                "names",
                "snapshot_sha256",
                "states",
            }
        ),
        label="checkpoint detached batch snapshot",
    )
    raw_states = item["states"]
    if type(raw_states) is not list or not raw_states:
        raise ValueError("checkpoint detached snapshot states are malformed")
    states: list[H6DetachedRecognitionSnapshotV3] = []
    cursor = offset
    state_keys = frozenset(
        {
            "attempt_spec_sha256",
            "endpoint_config_sha256",
            "post_recognition_cursor_sha256",
            "pass_index",
            "batch_index",
            "recognition_update_count",
            "receiver_count",
            "state_categorical_enabled",
            "model_categorical_enabled",
            "state_categorical_supports",
            "model_categorical_supports",
            "receiver_components",
            "names",
            "context_sha256",
            "recognition_state_sha256",
            "source_model_sha256",
            "law_sha256",
            "live_state_sha256",
            "snapshot_sha256",
            "tensors",
        }
    )
    for example_ordinal, raw_state in enumerate(raw_states):
        state = _exact_object(
            raw_state,
            keys=state_keys,
            label="checkpoint detached example snapshot",
        )
        names = _json_tuple(
            state["names"],
            label="checkpoint detached example names",
        )
        raw_tensors = state["tensors"]
        if type(raw_tensors) is not list or len(raw_tensors) != len(names):
            raise ValueError("checkpoint detached example tensors are malformed")
        tensors: list[Tensor] = []
        for local_name, raw_tensor in zip(
            names,
            raw_tensors,
            strict=True,
        ):
            record, cursor = _decode_tensor_record(
                raw_tensor,
                payload=payload,
                offset=cursor,
            )
            if (
                record.role != "recognition_snapshot"
                or record.state_name is not None
                or record.name != f"snapshot.example.{example_ordinal}.{local_name}"
            ):
                raise ValueError("checkpoint detached tensor ownership is malformed")
            tensors.append(record.decode_cpu())
        states.append(
            H6DetachedRecognitionSnapshotV3(
                attempt_spec_sha256=state["attempt_spec_sha256"],  # type: ignore[arg-type]
                endpoint_config_sha256=state["endpoint_config_sha256"],  # type: ignore[arg-type]
                post_recognition_cursor_sha256=state["post_recognition_cursor_sha256"],  # type: ignore[arg-type]
                pass_index=state["pass_index"],  # type: ignore[arg-type]
                batch_index=state["batch_index"],  # type: ignore[arg-type]
                recognition_update_count=state["recognition_update_count"],  # type: ignore[arg-type]
                receiver_count=state["receiver_count"],  # type: ignore[arg-type]
                state_categorical_enabled=state["state_categorical_enabled"],  # type: ignore[arg-type]
                model_categorical_enabled=state["model_categorical_enabled"],  # type: ignore[arg-type]
                state_categorical_supports=_json_tuple(
                    state["state_categorical_supports"],
                    label="checkpoint state categorical supports",
                ),  # type: ignore[arg-type]
                model_categorical_supports=_json_tuple(
                    state["model_categorical_supports"],
                    label="checkpoint model categorical supports",
                ),  # type: ignore[arg-type]
                receiver_components=_json_tuple(
                    state["receiver_components"],
                    label="checkpoint receiver components",
                ),  # type: ignore[arg-type]
                names=names,  # type: ignore[arg-type]
                context_sha256=state["context_sha256"],  # type: ignore[arg-type]
                recognition_state_sha256=state["recognition_state_sha256"],  # type: ignore[arg-type]
                source_model_sha256=state["source_model_sha256"],  # type: ignore[arg-type]
                law_sha256=state["law_sha256"],  # type: ignore[arg-type]
                live_state_sha256=state["live_state_sha256"],  # type: ignore[arg-type]
                snapshot_sha256=state["snapshot_sha256"],  # type: ignore[arg-type]
                _tensors=tuple(tensors),
            )
        )
    snapshot = H6DetachedBatchRecognitionSnapshotV3(
        authority_sha256=item["authority_sha256"],  # type: ignore[arg-type]
        attempt_spec_sha256=item["attempt_spec_sha256"],  # type: ignore[arg-type]
        endpoint_config_sha256=item["endpoint_config_sha256"],  # type: ignore[arg-type]
        post_recognition_cursor_sha256=item["post_recognition_cursor_sha256"],  # type: ignore[arg-type]
        pass_index=item["pass_index"],  # type: ignore[arg-type]
        batch_index=item["batch_index"],  # type: ignore[arg-type]
        recognition_update_count=item["recognition_update_count"],  # type: ignore[arg-type]
        receiver_count=item["receiver_count"],  # type: ignore[arg-type]
        active_target_counts=_json_tuple(
            item["active_target_counts"],
            label="checkpoint active target counts",
        ),  # type: ignore[arg-type]
        active_receiver_masks=_json_tuple(
            item["active_receiver_masks"],
            label="checkpoint active receiver masks",
        ),  # type: ignore[arg-type]
        live_batch_state_sha256=item["live_batch_state_sha256"],  # type: ignore[arg-type]
        states=tuple(states),
        names=_json_tuple(
            item["names"],
            label="checkpoint detached batch names",
        ),  # type: ignore[arg-type]
        snapshot_sha256=item["snapshot_sha256"],  # type: ignore[arg-type]
    )
    return snapshot, cursor


def _decode_optimizer_record(
    value: object,
    *,
    payload: bytes,
    offset: int,
) -> tuple[H6AdamWRecordV3, int]:
    item = _exact_object(
        value,
        keys=frozenset({"name", "groups", "states"}),
        label="checkpoint optimizer",
    )
    raw_groups = item["groups"]
    if type(raw_groups) is not list or not raw_groups:
        raise ValueError("checkpoint optimizer groups are malformed")
    groups: list[H6AdamWGroupV3] = []
    for raw_group in raw_groups:
        group = _exact_object(
            raw_group,
            keys=frozenset({"name", "parameter_names", "hyperparameters"}),
            label="checkpoint optimizer group",
        )
        parameter_names = _json_tuple(
            group["parameter_names"],
            label="checkpoint optimizer parameter names",
        )
        raw_hyperparameters = group["hyperparameters"]
        if type(raw_hyperparameters) is not list:
            raise ValueError("checkpoint optimizer hyperparameters are malformed")
        hyperparameters: list[tuple[str, object]] = []
        for raw_hyperparameter in raw_hyperparameters:
            if type(raw_hyperparameter) is not list or len(raw_hyperparameter) != 2:
                raise ValueError("checkpoint optimizer hyperparameter is malformed")
            name, raw_value = raw_hyperparameter
            if name == "betas":
                raw_betas = _json_tuple(
                    raw_value,
                    label="checkpoint optimizer betas",
                )
                if len(raw_betas) != 2:
                    raise ValueError("checkpoint optimizer betas are malformed")
                frozen: object = tuple(
                    _thaw_canonical_float(
                        value,
                        name="checkpoint optimizer beta",
                    )
                    for value in raw_betas
                )
            elif name in {
                "eps",
                "lr",
                "weight_decay",
                "initial_lr",
            }:
                frozen = (
                    raw_value
                    if type(raw_value) is int and not isinstance(raw_value, bool)
                    else _thaw_canonical_float(
                        raw_value,
                        name=f"checkpoint optimizer {name}",
                    )
                )
            else:
                frozen = raw_value
            hyperparameters.append((name, frozen))
        groups.append(
            H6AdamWGroupV3(
                name=group["name"],  # type: ignore[arg-type]
                parameter_names=parameter_names,  # type: ignore[arg-type]
                hyperparameters=tuple(hyperparameters),
            )
        )

    raw_states = item["states"]
    if type(raw_states) is not list:
        raise ValueError("checkpoint optimizer states are malformed")
    states: list[H6AdamWParameterStateV3] = []
    cursor = offset
    for raw_state in raw_states:
        state = _exact_object(
            raw_state,
            keys=frozenset({"parameter_name", "tensors", "scalars"}),
            label="checkpoint optimizer parameter state",
        )
        raw_tensors = state["tensors"]
        if type(raw_tensors) is not list:
            raise ValueError("checkpoint optimizer tensor states are malformed")
        tensors: list[H6TensorRecordV3] = []
        for raw_tensor in raw_tensors:
            tensor, cursor = _decode_tensor_record(
                raw_tensor,
                payload=payload,
                offset=cursor,
            )
            tensors.append(tensor)
        raw_scalars = state["scalars"]
        if type(raw_scalars) is not list:
            raise ValueError("checkpoint optimizer scalar states are malformed")
        scalars: list[tuple[str, int | float]] = []
        for raw_scalar in raw_scalars:
            if type(raw_scalar) is not list or len(raw_scalar) != 2:
                raise ValueError("checkpoint optimizer scalar state is malformed")
            scalar_name, scalar_value = raw_scalar
            thawed_scalar = (
                scalar_value
                if type(scalar_value) is int and not isinstance(scalar_value, bool)
                else _thaw_canonical_float(
                    scalar_value,
                    name=f"checkpoint optimizer scalar {scalar_name}",
                )
            )
            scalars.append((scalar_name, thawed_scalar))
        states.append(
            H6AdamWParameterStateV3(
                parameter_name=state["parameter_name"],  # type: ignore[arg-type]
                tensors=tuple(tensors),
                scalars=tuple(scalars),  # type: ignore[arg-type]
            )
        )
    return (
        H6AdamWRecordV3(
            name=item["name"],  # type: ignore[arg-type]
            groups=tuple(groups),
            states=tuple(states),
        ),
        cursor,
    )


def decode_h6_checkpoint_v3(raw: bytes) -> H6CheckpointV3:
    """Parse canonical checkpoint bytes and fail closed on any drift."""

    if type(raw) is not bytes:
        raise ValueError("checkpoint input must be immutable bytes")
    minimum = len(_MAGIC) + 8 + _TRAILER_BYTES
    if len(raw) < minimum or not raw.startswith(_MAGIC):
        raise ValueError("checkpoint schema or integrity marker is invalid")
    body = raw[:-_TRAILER_BYTES]
    trailer = raw[-_TRAILER_BYTES:]
    if hashlib.sha256(body).digest() != trailer:
        raise ValueError("checkpoint integrity trailer mismatch")
    header_length_start = len(_MAGIC)
    header_length = int.from_bytes(
        body[header_length_start : header_length_start + 8],
        "little",
    )
    if header_length <= 0 or header_length > _MAX_HEADER_BYTES:
        raise ValueError("checkpoint header length is invalid")
    header_start = header_length_start + 8
    header_end = header_start + header_length
    if header_end < header_start or header_end > len(body):
        raise ValueError("checkpoint header is truncated")
    header_bytes = body[header_start:header_end]
    tensor_payload = body[header_end:]
    header = _canonical_header(header_bytes)
    _exact_object(
        header,
        keys=frozenset(
            {
                "checkpoint_schema",
                "checkpoint_codec_sha256",
                "attempt_spec",
                "cursor",
                "objective_manifest",
                "runtime_identity",
                "deterministic_policy_sha256",
                "module_tensors",
                "optimizers",
                "detached_batch_snapshot",
                "checkpoint_sha256",
            }
        ),
        label="checkpoint header",
    )
    if (
        header["checkpoint_schema"] != "h6-checkpoint-v3"
        or header["checkpoint_codec_sha256"] != H6_CHECKPOINT_CODEC_SHA256
    ):
        raise ValueError("checkpoint schema or codec identity is stale")
    attempt, cursor, objective, runtime = _decode_typed_records(header)

    raw_module_tensors = header["module_tensors"]
    if type(raw_module_tensors) is not list or not raw_module_tensors:
        raise ValueError("checkpoint module tensor manifest is malformed")
    module_tensors: list[H6TensorRecordV3] = []
    offset = 0
    for raw_tensor in raw_module_tensors:
        tensor, offset = _decode_tensor_record(
            raw_tensor,
            payload=tensor_payload,
            offset=offset,
        )
        module_tensors.append(tensor)

    raw_optimizers = header["optimizers"]
    if type(raw_optimizers) is not list or not raw_optimizers:
        raise ValueError("checkpoint optimizer manifest is malformed")
    optimizers: list[H6AdamWRecordV3] = []
    for raw_optimizer in raw_optimizers:
        optimizer, offset = _decode_optimizer_record(
            raw_optimizer,
            payload=tensor_payload,
            offset=offset,
        )
        optimizers.append(optimizer)
    detached_batch_snapshot, offset = _decode_detached_batch_snapshot(
        header["detached_batch_snapshot"],
        payload=tensor_payload,
        offset=offset,
    )
    if offset != len(tensor_payload):
        raise ValueError("checkpoint tensor payload has trailing bytes")

    checkpoint = H6CheckpointV3(
        checkpoint_schema="h6-checkpoint-v3",
        attempt_spec=attempt,
        cursor=cursor,
        objective_manifest=objective,
        runtime_identity=runtime,
        deterministic_policy_sha256=header["deterministic_policy_sha256"],  # type: ignore[arg-type]
        module_tensors=tuple(module_tensors),
        optimizers=tuple(optimizers),
        detached_batch_snapshot=detached_batch_snapshot,
        checkpoint_sha256=header["checkpoint_sha256"],  # type: ignore[arg-type]
    )
    if (
        canonical_json_bytes(checkpoint.canonical_payload()) != header_bytes
        or checkpoint.to_bytes() != raw
    ):
        raise ValueError("checkpoint canonical encoding changed during decode")
    return checkpoint


def _is_redirect(path: Path, status: os.stat_result) -> bool:
    if stat.S_ISLNK(status.st_mode):
        return True
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if getattr(status, "st_file_attributes", 0) & reparse_flag:
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(callable(is_junction) and is_junction())


def read_h6_checkpoint_file_v3(
    path: Path,
    *,
    maximum_bytes: int,
    expected_checkpoint_sha256: str | None = None,
) -> H6CheckpointV3:
    """Read one bounded, nonredirected canonical v3 checkpoint file."""

    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError("checkpoint path must be an absolute pathlib Path")
    if type(maximum_bytes) is not int or maximum_bytes <= 0:
        raise ValueError("maximum_bytes must be a positive exact integer")
    if expected_checkpoint_sha256 is not None:
        _require_sha256(
            expected_checkpoint_sha256,
            "expected_checkpoint_sha256",
        )
    try:
        parent_before = path.parent.lstat()
        path_before = path.lstat()
    except OSError as exc:
        raise ValueError("checkpoint file is unavailable") from exc
    if (
        not stat.S_ISDIR(parent_before.st_mode)
        or _is_redirect(path.parent, parent_before)
        or not stat.S_ISREG(path_before.st_mode)
        or _is_redirect(path, path_before)
        or path_before.st_size > maximum_bytes
    ):
        raise ValueError("checkpoint file is not a bounded regular file")

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags | nofollow)
    except OSError as exc:
        raise ValueError("checkpoint file cannot be opened safely") from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino)
            != (path_before.st_dev, path_before.st_ino)
            or opened.st_size > maximum_bytes
        ):
            raise ValueError("checkpoint file identity or bound changed")
        chunks: list[bytes] = []
        remaining = maximum_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > maximum_bytes:
            raise ValueError("checkpoint file exceeds the maximum bound")
        opened_after = os.fstat(descriptor)
        if (opened_after.st_dev, opened_after.st_ino, opened_after.st_size) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
        ) or opened_after.st_mtime_ns != opened.st_mtime_ns:
            raise ValueError("checkpoint file changed while reading")
    finally:
        os.close(descriptor)
    try:
        parent_after = path.parent.lstat()
        path_after = path.lstat()
    except OSError as exc:
        raise ValueError("checkpoint file changed after reading") from exc
    if (
        (parent_after.st_dev, parent_after.st_ino)
        != (parent_before.st_dev, parent_before.st_ino)
        or (path_after.st_dev, path_after.st_ino, path_after.st_size)
        != (opened.st_dev, opened.st_ino, opened.st_size)
        or path_after.st_mtime_ns != opened.st_mtime_ns
        or _is_redirect(path.parent, parent_after)
        or _is_redirect(path, path_after)
    ):
        raise ValueError("checkpoint path identity changed while reading")

    checkpoint = decode_h6_checkpoint_v3(raw)
    if (
        expected_checkpoint_sha256 is not None
        and checkpoint.checkpoint_sha256 != expected_checkpoint_sha256
    ):
        raise ValueError("checkpoint digest differs from expected authority")
    return checkpoint


def _module_records(
    named_modules: object,
    *,
    aliases: dict[tuple[str, int | None, int], str],
) -> tuple[
    tuple[H6TensorRecordV3, ...],
    dict[str, nn.Parameter],
    tuple[tuple[str, nn.Module], ...],
]:
    raw_items = _named_items(named_modules, label="named_modules")
    sorted_items = tuple(sorted(raw_items, key=lambda item: item[0]))
    records: list[H6TensorRecordV3] = []
    parameters: dict[str, nn.Parameter] = {}
    for module_name, value in sorted_items:
        if not isinstance(value, nn.Module):
            raise ValueError("named_modules values must be torch modules")
        module = value
        parameter_items = tuple(module.named_parameters(remove_duplicate=False))
        buffer_items = tuple(module.named_buffers(remove_duplicate=False))
        local_parameter_names = tuple(name for name, _ in parameter_items)
        local_buffer_names = tuple(name for name, _ in buffer_items)
        _validate_unique_names(
            tuple(
                f"{module_name}.{name}"
                for name in local_parameter_names + local_buffer_names
            ),
            label="module parameter/buffer inventory",
        )
        parameter_by_local_name = dict(parameter_items)
        buffer_by_local_name = dict(buffer_items)
        state = module.state_dict(keep_vars=True)
        if not state:
            raise ValueError("checkpoint modules must own persistent state")
        state_names = tuple(state)
        _validate_unique_names(
            tuple(f"{module_name}.{name}" for name in state_names),
            label="module state inventory",
        )
        for local_name in sorted(state_names):
            qualified_name = f"{module_name}.{local_name}"
            tensor = state[local_name]
            if local_name in parameter_by_local_name:
                role: Literal["module_parameter", "module_buffer"] = "module_parameter"
                parameter = parameter_by_local_name[local_name]
                if tensor is not parameter:
                    raise ValueError(
                        "module state parameter identity was replaced by a hook"
                    )
                parameters[qualified_name] = parameter
            elif local_name in buffer_by_local_name:
                role = "module_buffer"
                if tensor is not buffer_by_local_name[local_name]:
                    raise ValueError(
                        "module state buffer identity was replaced by a hook"
                    )
            else:
                raise ValueError(
                    "module state contains unknown non-parameter/buffer state"
                )
            records.append(
                H6TensorRecordV3.capture(
                    role=role,
                    name=qualified_name,
                    tensor=tensor,
                    aliases=aliases,
                )
            )
    if len(records) > _MAX_RECORDS:
        raise ValueError("checkpoint module inventory is unbounded")
    records.sort(key=lambda record: record.name)
    _validate_unique_names(
        tuple(record.name for record in records),
        label="checkpoint module tensor inventory",
    )
    return (
        tuple(records),
        parameters,
        tuple(
            (name, value)
            for name, value in sorted_items
            if isinstance(value, nn.Module)
        ),
    )


def _adamw_hyperparameters(
    group: dict[str, object],
) -> tuple[tuple[str, object], ...]:
    keys = tuple(key for key in group if key != "params")
    if any(type(key) is not str for key in keys):
        raise ValueError("AdamW group keys must be strings")
    unknown = set(keys) - _ADAMW_GROUP_KEYS
    if unknown:
        raise ValueError(f"unknown AdamW group hyperparameter {sorted(unknown)[0]!r}")
    required = _ADAMW_GROUP_KEYS - {"initial_lr"}
    missing = required - set(keys)
    if missing:
        raise ValueError(
            f"AdamW group is missing hyperparameter {sorted(missing)[0]!r}"
        )
    expected_policy = {
        "betas": (0.9, 0.999),
        "eps": 1.0e-8,
        "amsgrad": False,
        "maximize": False,
        "foreach": False,
        "capturable": False,
        "differentiable": False,
        "fused": False,
        "decoupled_weight_decay": True,
    }
    for name, expected in expected_policy.items():
        if group[name] != expected or type(group[name]) is not type(expected):
            raise ValueError(f"AdamW group {name} does not match the frozen v3 policy")
    for name in ("lr", "weight_decay"):
        value = group[name]
        if type(value) not in (int, float) or isinstance(value, bool):
            raise ValueError(f"AdamW {name} must be a real scalar")
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"AdamW {name} must be finite and nonnegative")
    return tuple(
        (
            key,
            _freeze_scalar(group[key], name=f"AdamW hyperparameter {key}"),
        )
        for key in sorted(keys)
    )


def _capture_adamw_state(
    *,
    optimizer_name: str,
    parameter_name: str,
    parameter: nn.Parameter,
    state: object,
    amsgrad: bool,
    aliases: dict[tuple[str, int | None, int], str],
) -> H6AdamWParameterStateV3:
    if not isinstance(state, Mapping):
        raise ValueError("AdamW state must be a mapping")
    raw_keys = tuple(state)
    if any(type(key) is not str for key in raw_keys):
        raise ValueError("unknown AdamW state key must be a string")
    keys = set(raw_keys)
    expected = set(_ADAMW_BASE_STATE)
    if amsgrad:
        expected.add("max_exp_avg_sq")
    if not keys:
        return H6AdamWParameterStateV3(parameter_name, (), ())
    unknown = keys - expected
    if unknown:
        raise ValueError(f"unknown AdamW state {sorted(unknown)[0]!r}")
    missing = expected - keys
    if missing:
        raise ValueError(f"incomplete AdamW state missing {sorted(missing)[0]!r}")

    tensors: list[H6TensorRecordV3] = []
    scalars: list[tuple[str, int | float]] = []
    for state_name in sorted(keys):
        value = state[state_name]
        if isinstance(value, Tensor):
            if state_name == "step":
                if (
                    value.numel() != 1
                    or value.ndim != 0
                    or value.dtype not in (torch.float32, torch.float64)
                    or not bool(torch.isfinite(value).all())
                    or float(value.detach().to("cpu").item()) < 0.0
                ):
                    raise ValueError("AdamW tensor step must be finite scalar")
            elif (
                tuple(value.shape) != tuple(parameter.shape)
                or value.dtype is not parameter.dtype
            ):
                raise ValueError(
                    f"AdamW {state_name} shape/dtype does not match parameter"
                )
            tensor_name = f"{optimizer_name}.{parameter_name}.{state_name}"
            tensors.append(
                H6TensorRecordV3.capture(
                    role="optimizer_state",
                    name=tensor_name,
                    state_name=state_name,
                    tensor=value,
                    aliases=aliases,
                )
            )
        elif state_name == "step" and type(value) in (int, float):
            if isinstance(value, bool) or (
                type(value) is float and not math.isfinite(value)
            ):
                raise ValueError("AdamW scalar step must be finite")
            if value < 0:
                raise ValueError("AdamW scalar step must be nonnegative")
            scalars.append((state_name, value))
        else:
            raise ValueError(f"AdamW {state_name} must be a tensor")
    return H6AdamWParameterStateV3(
        parameter_name=parameter_name,
        tensors=tuple(tensors),
        scalars=tuple(scalars),
    )


def _optimizer_records(
    named_optimizers: object,
    *,
    parameters: dict[str, nn.Parameter],
    aliases: dict[tuple[str, int | None, int], str],
) -> tuple[
    tuple[H6AdamWRecordV3, ...],
    tuple[tuple[str, torch.optim.AdamW], ...],
]:
    raw_items = _named_items(named_optimizers, label="named_optimizers")
    if not raw_items:
        raise ValueError("checkpoint requires at least one named AdamW optimizer")
    parameter_names_by_id = {
        id(parameter): name for name, parameter in parameters.items()
    }
    if len(parameter_names_by_id) != len(parameters):
        raise ValueError("module parameter inventory contains aliases")
    globally_bound: set[int] = set()
    records: list[H6AdamWRecordV3] = []
    sorted_items = tuple(sorted(raw_items, key=lambda item: item[0]))
    checked_items: list[tuple[str, torch.optim.AdamW]] = []
    for optimizer_name, value in sorted_items:
        if type(value) is not torch.optim.AdamW:
            raise ValueError("named optimizer must be an exact AdamW")
        optimizer = value
        checked_items.append((optimizer_name, optimizer))
        groups: list[H6AdamWGroupV3] = []
        optimizer_parameters: dict[int, nn.Parameter] = {}
        parameter_amsgrad: dict[int, bool] = {}
        for index, group in enumerate(optimizer.param_groups):
            if type(group) is not dict:
                raise ValueError("AdamW parameter groups must be dictionaries")
            hyperparameters = _adamw_hyperparameters(group)
            raw_parameters = group.get("params")
            if type(raw_parameters) is not list or not raw_parameters:
                raise ValueError("AdamW group requires a nonempty parameter list")
            stable_names: list[str] = []
            for parameter in raw_parameters:
                if type(parameter) is not nn.Parameter:
                    raise ValueError("AdamW groups must contain exact Parameters")
                parameter_name = parameter_names_by_id.get(id(parameter))
                if (
                    parameter_name is None
                    or parameters[parameter_name] is not parameter
                ):
                    raise ValueError(
                        "optimizer parameter is not bound to a named module parameter"
                    )
                identity = id(parameter)
                if identity in globally_bound:
                    raise ValueError("optimizer parameter is bound more than once")
                globally_bound.add(identity)
                optimizer_parameters[identity] = parameter
                parameter_amsgrad[identity] = bool(group["amsgrad"])
                stable_names.append(parameter_name)
            groups.append(
                H6AdamWGroupV3(
                    name=f"{optimizer_name}.group.{index:06d}",
                    parameter_names=tuple(stable_names),
                    hyperparameters=hyperparameters,
                )
            )
        unbound_state = tuple(
            key
            for key in optimizer.state
            if id(key) not in optimizer_parameters
            or optimizer_parameters[id(key)] is not key
        )
        if unbound_state:
            raise ValueError("checkpoint contains unbound optimizer state")
        states = tuple(
            _capture_adamw_state(
                optimizer_name=optimizer_name,
                parameter_name=parameter_names_by_id[identity],
                parameter=parameter,
                state=optimizer.state.get(parameter, {}),
                amsgrad=parameter_amsgrad[identity],
                aliases=aliases,
            )
            for identity, parameter in sorted(
                optimizer_parameters.items(),
                key=lambda item: parameter_names_by_id[item[0]],
            )
        )
        records.append(
            H6AdamWRecordV3(
                name=optimizer_name,
                groups=tuple(groups),
                states=states,
            )
        )
    if globally_bound != set(parameter_names_by_id):
        raise ValueError(
            "every named module parameter must bind exactly one optimizer group"
        )
    return tuple(records), tuple(checked_items)


def capture_h6_checkpoint_v3(
    *,
    attempt_spec: H6AttemptSpecV3,
    cursor: H6AttemptCursorV3,
    objective_manifest: H6ObjectiveManifestV3,
    runtime_identity: H6PredictionRuntimeIdentity,
    named_modules: object,
    named_optimizers: object,
    detached_batch_snapshot: (H6DetachedBatchRecognitionSnapshotV3 | None) = None,
) -> H6CheckpointV3:
    """Capture canonical module and named AdamW state without object IDs."""

    for record, expected_type in (
        (attempt_spec, H6AttemptSpecV3),
        (cursor, H6AttemptCursorV3),
        (objective_manifest, H6ObjectiveManifestV3),
        (runtime_identity, H6PredictionRuntimeIdentity),
    ):
        if type(record) is not expected_type:
            raise ValueError("checkpoint capture requires exact v3 records")
        record.__post_init__()
    aliases: dict[tuple[str, int | None, int], str] = {}
    module_tensors, parameters, _ = _module_records(
        named_modules,
        aliases=aliases,
    )
    optimizers, _ = _optimizer_records(
        named_optimizers,
        parameters=parameters,
        aliases=aliases,
    )
    payload = _checkpoint_payload(
        attempt_spec=attempt_spec,
        cursor=cursor,
        objective_manifest=objective_manifest,
        runtime_identity=runtime_identity,
        deterministic_policy_sha256=H6_DETERMINISTIC_POLICY_SHA256,
        module_tensors=module_tensors,
        optimizers=optimizers,
        detached_batch_snapshot=detached_batch_snapshot,
    )
    return H6CheckpointV3(
        checkpoint_schema="h6-checkpoint-v3",
        attempt_spec=attempt_spec,
        cursor=cursor,
        objective_manifest=objective_manifest,
        runtime_identity=runtime_identity,
        deterministic_policy_sha256=H6_DETERMINISTIC_POLICY_SHA256,
        module_tensors=module_tensors,
        optimizers=optimizers,
        detached_batch_snapshot=detached_batch_snapshot,
        checkpoint_sha256=_owned_hash(
            "vfe4.h6.checkpoint.v3",
            payload,
        ),
    )


def _module_signature(
    records: tuple[H6TensorRecordV3, ...],
) -> tuple[tuple[str, str, str, tuple[int, ...]], ...]:
    return tuple(
        (record.role, record.name, record.dtype, record.shape) for record in records
    )


def _type_identity(value: type[object]) -> str:
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if (
        type(module) is not str
        or not module
        or type(qualname) is not str
        or not qualname
    ):
        raise ValueError("module semantic type has no deterministic identity")
    return f"{module}.{qualname}"


def _code_constant_payload(value: object) -> object:
    if type(value) is CodeType:
        return {"code": _code_semantic_payload(value)}
    return _semantic_value_payload(value, active=set())


def _code_semantic_payload(code: CodeType) -> dict[str, object]:
    return {
        "name": code.co_name,
        "qualname": code.co_qualname,
        "argcount": code.co_argcount,
        "posonlyargcount": code.co_posonlyargcount,
        "kwonlyargcount": code.co_kwonlyargcount,
        "nlocals": code.co_nlocals,
        "stacksize": code.co_stacksize,
        "flags": code.co_flags,
        "bytecode": code.co_code.hex(),
        "exception_table": code.co_exceptiontable.hex(),
        "constants": tuple(_code_constant_payload(value) for value in code.co_consts),
        "names": code.co_names,
        "varnames": code.co_varnames,
        "freevars": code.co_freevars,
        "cellvars": code.co_cellvars,
    }


def _trusted_contextmanager_generator(
    function: FunctionType,
) -> FunctionType | None:
    closure = function.__closure__
    if (
        function.__code__ is not _TRUSTED_CONTEXTMANAGER_WRAPPER_CODE
        or function.__code__.co_freevars != ("func",)
        or closure is None
        or len(closure) != 1
    ):
        return None
    wrapped = getattr(function, "__wrapped__", None)
    if (
        type(wrapped) is not FunctionType
        or closure[0].cell_contents is not wrapped
        or not bool(wrapped.__code__.co_flags & inspect.CO_GENERATOR)
        or function.__module__ != wrapped.__module__
        or function.__qualname__ != wrapped.__qualname__
        or function.__name__ != wrapped.__name__
    ):
        return None
    return wrapped


def _function_semantic_payload(
    function: FunctionType,
    *,
    active: set[int] | None = None,
) -> dict[str, object]:
    owned_active = active if active is not None else set()
    identity = id(function)
    if identity in owned_active:
        raise ValueError("module behavior function closure contains a cycle")
    owned_active.add(identity)
    try:
        wrapped_generator = _trusted_contextmanager_generator(function)
        closure: tuple[tuple[str, dict[str, object]], ...] = ()
        if function.__closure__:
            closure_items: list[tuple[str, dict[str, object]]] = []
            for name, cell in zip(
                function.__code__.co_freevars,
                function.__closure__,
                strict=True,
            ):
                value = cell.cell_contents
                if name == "__class__" and isinstance(value, type):
                    payload: dict[str, object] = {
                        "class": _type_identity(value),
                    }
                elif name == "func" and value is wrapped_generator:
                    payload = {
                        "contextmanager_generator": _function_semantic_payload(
                            wrapped_generator,
                            active=owned_active,
                        )
                    }
                else:
                    raise ValueError(
                        "module behavior cannot close over mutable runtime state"
                    )
                closure_items.append((name, payload))
            closure = tuple(closure_items)
        return {
            "module": function.__module__,
            "qualname": function.__qualname__,
            "code": _code_semantic_payload(function.__code__),
            "closure": closure,
            "defaults": _semantic_value_payload(
                function.__defaults__,
                active=set(),
            ),
            "kwdefaults": _semantic_value_payload(
                function.__kwdefaults__,
                active=set(),
            ),
        }
    finally:
        owned_active.remove(identity)


def _semantic_value_payload(
    value: object,
    *,
    active: set[int],
) -> object:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("module semantic float attributes must be finite")
        return {"float": value.hex()}
    if type(value) is complex:
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ValueError("module semantic complex attributes must be finite")
        return {
            "complex": (value.real.hex(), value.imag.hex()),
        }
    if type(value) is bytes:
        return {"bytes": value.hex()}
    if value is Ellipsis:
        return {"sentinel": "ellipsis"}
    if value is NotImplemented:
        return {"sentinel": "not-implemented"}
    if isinstance(value, Enum):
        return {
            "enum_type": _type_identity(type(value)),
            "name": value.name,
            "value": _semantic_value_payload(value.value, active=active),
        }
    if isinstance(value, torch.dtype):
        return {"torch_dtype": str(value)}
    if isinstance(value, torch.device):
        return {
            "torch_device": {
                "type": value.type,
                "index": value.index,
            }
        }
    if isinstance(value, Path):
        return {
            "path_type": _type_identity(type(value)),
            "value": str(value),
        }
    if isinstance(value, (Tensor, nn.Module)):
        raise ValueError(
            "module semantic attributes cannot hide unregistered tensors or modules"
        )
    identity = id(value)
    if identity in active:
        raise ValueError("module semantic attributes cannot contain cycles")
    if isinstance(value, tuple):
        active.add(identity)
        try:
            return {
                "tuple_type": _type_identity(type(value)),
                "items": tuple(
                    _semantic_value_payload(item, active=active) for item in value
                ),
            }
        finally:
            active.remove(identity)
    if type(value) is list:
        active.add(identity)
        try:
            return {
                "list": tuple(
                    _semantic_value_payload(item, active=active) for item in value
                )
            }
        finally:
            active.remove(identity)
    if isinstance(value, Mapping):
        active.add(identity)
        try:
            items = tuple(
                (
                    _semantic_value_payload(key, active=active),
                    _semantic_value_payload(owned, active=active),
                )
                for key, owned in value.items()
            )
            return {
                "mapping_type": _type_identity(type(value)),
                "items": tuple(
                    sorted(
                        items,
                        key=lambda item: canonical_json_bytes(item[0]),
                    )
                ),
            }
        finally:
            active.remove(identity)
    if isinstance(value, (set, frozenset)):
        active.add(identity)
        try:
            items = tuple(
                _semantic_value_payload(item, active=active) for item in value
            )
            return {
                "set_type": _type_identity(type(value)),
                "items": tuple(sorted(items, key=canonical_json_bytes)),
            }
        finally:
            active.remove(identity)
    if is_dataclass(value) and not isinstance(value, type):
        active.add(identity)
        try:
            return {
                "dataclass_type": _type_identity(type(value)),
                "fields": tuple(
                    (
                        item.name,
                        _semantic_value_payload(
                            object.__getattribute__(value, item.name),
                            active=active,
                        ),
                    )
                    for item in fields(value)
                ),
            }
        finally:
            active.remove(identity)
    if type(value) is FunctionType:
        return {
            "function": _function_semantic_payload(value),
        }
    raise ValueError(
        f"unsupported module semantic attribute type {_type_identity(type(value))}"
    )


def _forward_semantic_payload(module_type: type[nn.Module]) -> dict[str, object]:
    owner: type[object] | None = None
    descriptor: object = None
    for candidate in module_type.__mro__:
        namespace = vars(candidate)
        if "forward" in namespace:
            owner = candidate
            descriptor = namespace["forward"]
            break
    if owner is None:
        raise ValueError("module type has no resolved forward implementation")
    descriptor_kind = "method"
    if type(descriptor) is staticmethod:
        descriptor_kind = "staticmethod"
        descriptor = descriptor.__func__
    elif type(descriptor) is classmethod:
        descriptor_kind = "classmethod"
        descriptor = descriptor.__func__
    if type(descriptor) is FunctionType:
        semantic: object = _function_semantic_payload(descriptor)
    else:
        module_name = getattr(descriptor, "__module__", None)
        qualname = getattr(descriptor, "__qualname__", None)
        if (
            type(module_name) is not str
            or not module_name
            or type(qualname) is not str
            or not qualname
        ):
            raise ValueError("module forward descriptor has no deterministic identity")
        semantic = {
            "descriptor_type": _type_identity(type(descriptor)),
            "module": module_name,
            "qualname": qualname,
        }
    return {
        "owner": _type_identity(owner),
        "kind": descriptor_kind,
        "semantic": semantic,
    }


def _python_descriptor_functions(
    descriptor: object,
) -> tuple[tuple[str, FunctionType], ...]:
    if type(descriptor) is FunctionType:
        return (("method", descriptor),)
    if isinstance(descriptor, staticmethod):
        function = descriptor.__func__
        return (
            (("staticmethod", function),)
            if type(function) is FunctionType
            else ()
        )
    if isinstance(descriptor, classmethod):
        function = descriptor.__func__
        return (
            (("classmethod", function),)
            if type(function) is FunctionType
            else ()
        )
    if isinstance(descriptor, property):
        accessors = (
            ("property_get", descriptor.fget),
            ("property_set", descriptor.fset),
            ("property_delete", descriptor.fdel),
        )
        return tuple(
            (kind, function)
            for kind, function in accessors
            if type(function) is FunctionType
        )
    return ()


def _bounded_application_source_sha256(path: Path) -> str:
    try:
        before = path.lstat()
    except OSError as exc:
        raise ValueError("application source file is unavailable") from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or _is_redirect(path, before)
        or before.st_size > _MAX_APPLICATION_SOURCE_BYTES
    ):
        raise ValueError("application source is not a bounded regular file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags | nofollow)
    except OSError as exc:
        raise ValueError("application source cannot be opened safely") from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
            or opened.st_size > _MAX_APPLICATION_SOURCE_BYTES
        ):
            raise ValueError("application source identity or bound changed")
        digest = hashlib.sha256()
        remaining = _MAX_APPLICATION_SOURCE_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            digest.update(chunk)
            remaining -= len(chunk)
        if remaining == 0:
            raise ValueError("application source exceeds its byte bound")
        after = os.fstat(descriptor)
        if (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
        ):
            raise ValueError("application source changed while hashing")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _application_source_payload(
    value: FunctionType | type[object] | ModuleType,
    *,
    cache: dict[str, dict[str, str]],
) -> dict[str, str] | None:
    if isinstance(value, ModuleType):
        module_name = value.__name__
        module = value
    else:
        module_name = getattr(value, "__module__", None)
        if type(module_name) is not str or not module_name:
            return None
        module = sys.modules.get(module_name)
    raw_path = getattr(module, "__file__", None)
    if type(raw_path) is not str or not raw_path:
        return None
    try:
        source_path = Path(raw_path).resolve(strict=True)
        relative = source_path.relative_to(_PROJECT_ROOT)
    except (OSError, ValueError):
        if not module_name.startswith("vfe4."):
            return None
        raise ValueError("VFE4 application source escaped its project root")
    is_test_source = bool(relative.parts and relative.parts[0] == "tests")
    if not module_name.startswith("vfe4.") and not is_test_source:
        return None
    cache_key = str(source_path)
    payload = cache.get(cache_key)
    if payload is None:
        payload = {
            "module": module_name,
            "path": relative.as_posix(),
            "sha256": _bounded_application_source_sha256(source_path),
        }
        cache[cache_key] = payload
    return payload


def _referenced_application_sources(
    function: FunctionType,
    *,
    cache: dict[str, dict[str, str]],
) -> tuple[dict[str, str], ...]:
    pending: list[FunctionType | type[object] | ModuleType] = [function]
    seen: set[int] = set()
    sources: dict[tuple[str, str], dict[str, str]] = {}
    while pending:
        value = pending.pop()
        identity = id(value)
        if identity in seen:
            continue
        seen.add(identity)
        if len(seen) > _MAX_REFERENCED_APPLICATION_OBJECTS:
            raise ValueError("application behavior dependency graph is unbounded")
        source = _application_source_payload(value, cache=cache)
        if source is not None:
            sources[(source["module"], source["path"])] = source
        if type(value) is FunctionType:
            wrapped_generator = _trusted_contextmanager_generator(value)
            if wrapped_generator is not None:
                pending.append(wrapped_generator)
            for name in value.__code__.co_names:
                referenced = value.__globals__.get(name)
                if (
                    type(referenced) is FunctionType
                    or isinstance(referenced, (type, ModuleType))
                ):
                    if _application_source_payload(referenced, cache=cache) is not None:
                        pending.append(referenced)
        elif isinstance(value, type):
            for candidate in value.__mro__:
                if candidate is nn.Module or candidate is object:
                    break
                for descriptor in vars(candidate).values():
                    pending.extend(
                        function
                        for _kind, function in _python_descriptor_functions(
                            descriptor
                        )
                    )
    return tuple(sources[key] for key in sorted(sources))


def _application_mro_behavior_payload(
    module_type: type[nn.Module],
    *,
    source_cache: dict[str, dict[str, str]],
) -> tuple[dict[str, object], ...]:
    owners: list[dict[str, object]] = []
    for candidate in module_type.__mro__:
        if candidate is nn.Module or candidate is object:
            break
        descriptors: list[dict[str, object]] = []
        for name, descriptor in sorted(vars(candidate).items()):
            functions = _python_descriptor_functions(descriptor)
            if not functions:
                continue
            descriptors.append(
                {
                    "name": name,
                    "accessors": tuple(
                        {
                            "kind": kind,
                            "function": _function_semantic_payload(function),
                            "referenced_application_sources": (
                                _referenced_application_sources(
                                    function,
                                    cache=source_cache,
                                )
                            ),
                        }
                        for kind, function in functions
                    ),
                }
            )
        source = _application_source_payload(candidate, cache=source_cache)
        owners.append(
            {
                "type": _type_identity(candidate),
                "torch_runtime": (
                    str(torch.__version__)
                    if candidate.__module__.startswith("torch.")
                    else None
                ),
                "source": source,
                "descriptors": tuple(descriptors),
            }
        )
    return tuple(owners)


def _behavior_method_names(module_type: type[nn.Module]) -> frozenset[str]:
    names: set[str] = set()
    for candidate in module_type.__mro__:
        for name, descriptor in vars(candidate).items():
            if _python_descriptor_functions(descriptor):
                names.add(name)
    return frozenset(names)


def _local_tensor_contract(
    value: Tensor | None,
    *,
    parameter: bool,
    require_cpu: bool,
) -> object:
    if value is None:
        return None
    if parameter:
        if type(value) is not nn.Parameter:
            raise ValueError(
                "module semantic parameter inventory requires exact Parameters"
            )
    elif type(value) is not Tensor:
        raise ValueError("module semantic buffer inventory requires exact Tensors")
    if require_cpu and value.device.type != "cpu":
        raise ValueError("checkpoint factory authority requires CPU modules")
    return {
        "dtype": str(value.dtype),
        "shape": tuple(value.shape),
        "layout": str(value.layout),
        "requires_grad": bool(value.requires_grad),
    }


def _module_tree_semantic_payload(
    named_modules: tuple[tuple[str, nn.Module], ...],
    *,
    require_cpu: bool,
) -> tuple[dict[str, object], ...]:
    entries: list[dict[str, object]] = []
    seen: dict[int, str] = {}
    source_cache: dict[str, dict[str, str]] = {}

    def visit(path: str, module: nn.Module) -> None:
        identity = id(module)
        previous = seen.get(identity)
        if previous is not None:
            raise ValueError(
                "module semantic tree contains shared module alias "
                f"{previous!r}/{path!r}"
            )
        seen[identity] = path
        instance = object.__getattribute__(module, "__dict__")
        behavior_names = _behavior_method_names(type(module))
        for name, value in instance.items():
            if name in behavior_names and callable(value):
                raise ValueError(
                    f"module {path!r} has an unauthorized instance-level "
                    f"callable override for {name!r}"
                )
        for hook_name in _MODULE_HOOK_MAP_NAMES:
            hooks = instance.get(hook_name)
            if hooks is not None and len(hooks):
                raise ValueError(f"module {path!r} has unauthorized runtime hooks")
        parameters = object.__getattribute__(module, "_parameters")
        buffers = object.__getattribute__(module, "_buffers")
        children = object.__getattribute__(module, "_modules")
        nonpersistent = object.__getattribute__(
            module,
            "_non_persistent_buffers_set",
        )
        if (
            type(parameters) is not dict
            or type(buffers) is not dict
            or type(children) is not dict
            or type(nonpersistent) is not set
        ):
            raise ValueError("module semantic registries are not canonical")
        parameter_contract = tuple(
            (
                name,
                _local_tensor_contract(
                    value,
                    parameter=True,
                    require_cpu=require_cpu,
                ),
            )
            for name, value in parameters.items()
        )
        buffer_contract = tuple(
            (
                name,
                _local_tensor_contract(
                    value,
                    parameter=False,
                    require_cpu=require_cpu,
                ),
                name not in nonpersistent,
            )
            for name, value in buffers.items()
        )
        child_contract: list[tuple[str, str | None]] = []
        for name, child in children.items():
            _require_root_name(name, f"module {path!r} child name")
            if child is not None and not isinstance(child, nn.Module):
                raise ValueError("module semantic child is not a torch module")
            child_contract.append(
                (
                    name,
                    None if child is None else _type_identity(type(child)),
                )
            )
        attributes = tuple(
            (
                name,
                _semantic_value_payload(value, active=set()),
            )
            for name, value in sorted(instance.items())
            if name not in _MODULE_INTERNAL_ATTRIBUTE_NAMES
        )
        module_type = type(module)
        entries.append(
            {
                "path": path,
                "type": _type_identity(module_type),
                "mro": tuple(
                    _type_identity(candidate) for candidate in module_type.__mro__
                ),
                "forward": _forward_semantic_payload(module_type),
                "application_mro_behavior": _application_mro_behavior_payload(
                    module_type,
                    source_cache=source_cache,
                ),
                "training": bool(instance.get("training", False)),
                "parameters": parameter_contract,
                "buffers": buffer_contract,
                "children": tuple(child_contract),
                "attributes": attributes,
            }
        )
        for name, child in children.items():
            if child is not None:
                visit(f"{path}.{name}", child)

    for root_name, module in named_modules:
        visit(root_name, module)
    return tuple(entries)


def _module_tree_semantic_sha256(
    named_modules: tuple[tuple[str, nn.Module], ...],
    *,
    require_cpu: bool,
) -> str:
    return _owned_hash(
        "vfe4.h6.checkpoint-factory-module-tree.v3",
        _module_tree_semantic_payload(
            named_modules,
            require_cpu=require_cpu,
        ),
    )


def _expected_factory_sha256s(
    attempt_spec: H6AttemptSpecV3,
) -> tuple[tuple[str, str], ...]:
    bindings = [("model", attempt_spec.model_factory_sha256)]
    if attempt_spec.recognition_factory_sha256 is not None:
        bindings.append(
            (
                "recognition",
                attempt_spec.recognition_factory_sha256,
            )
        )
    return tuple(sorted(bindings))


@dataclass(frozen=True, slots=True)
class _H6CheckpointFactoryAuthorityV3:
    """One binder-issued construction capability with pinned module semantics."""

    authority_schema: Literal["h6-checkpoint-factory-authority-v3"]
    attempt_spec_sha256: str
    factory_sha256s: tuple[tuple[str, str], ...]
    module_tree_semantic_sha256: str
    authority_sha256: str
    _module_factories: tuple[tuple[str, object], ...] = field(
        repr=False,
        compare=False,
    )
    _issuance_nonce: object = field(repr=False, compare=False)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "authority_schema": self.authority_schema,
            "attempt_spec_sha256": self.attempt_spec_sha256,
            "factory_sha256s": self.factory_sha256s,
            "module_tree_semantic_sha256": (self.module_tree_semantic_sha256),
        }

    def __post_init__(self) -> None:
        if self.authority_schema != "h6-checkpoint-factory-authority-v3":
            raise ValueError("unsupported checkpoint factory authority schema")
        _require_sha256(
            self.attempt_spec_sha256,
            "factory authority attempt_spec_sha256",
        )
        _require_sha256(
            self.module_tree_semantic_sha256,
            "module_tree_semantic_sha256",
        )
        if type(self.factory_sha256s) is not tuple:
            raise ValueError("factory authority bindings must be a tuple")
        names: list[str] = []
        for item in self.factory_sha256s:
            if type(item) is not tuple or len(item) != 2:
                raise ValueError("factory authority bindings must be name/digest pairs")
            name, digest = item
            names.append(_require_root_name(name, "factory authority root"))
            _require_sha256(digest, f"{name} factory authority SHA-256")
        if tuple(names) != tuple(sorted(names)):
            raise ValueError("factory authority bindings must be name-sorted")
        _validate_unique_names(
            tuple(names),
            label="factory authority root inventory",
        )
        factory_items = _named_items(
            self._module_factories,
            label="factory authority constructors",
        )
        if tuple(name for name, _ in factory_items) != tuple(names):
            raise ValueError("factory authority constructor inventory is stale")
        if any(not callable(factory) for _, factory in factory_items):
            raise ValueError("factory authority constructors must be callable")
        _require_sha256(self.authority_sha256, "factory authority SHA-256")
        if self.authority_sha256 != _owned_hash(
            "vfe4.h6.checkpoint-factory-authority.v3",
            self.canonical_payload(),
        ):
            raise ValueError("checkpoint factory authority identity is stale")


def _issue_h6_checkpoint_factory_authority_v3(
    *,
    attempt_spec: H6AttemptSpecV3,
    expected_named_modules: object,
    module_factories: object,
) -> _H6CheckpointFactoryAuthorityV3:
    """Bind constructors to an attempt and an independently expected module tree."""

    if type(attempt_spec) is not H6AttemptSpecV3:
        raise ValueError("factory authority requires an exact attempt spec")
    attempt_spec.__post_init__()
    expected_items = _named_items(
        expected_named_modules,
        label="expected_named_modules",
    )
    factory_items = _named_items(
        module_factories,
        label="module_factories",
    )
    factory_sha256s = _expected_factory_sha256s(attempt_spec)
    expected_names = tuple(name for name, _ in factory_sha256s)
    if (
        tuple(sorted(name for name, _ in expected_items)) != expected_names
        or tuple(sorted(name for name, _ in factory_items)) != expected_names
    ):
        raise ValueError(
            "checkpoint factory authority root inventory differs from attempt"
        )
    if any(not isinstance(module, nn.Module) for _, module in expected_items):
        raise ValueError("expected module semantics require torch modules")
    if any(not callable(factory) for _, factory in factory_items):
        raise ValueError("checkpoint module factories must be callable")
    sorted_expected = tuple(
        (name, module)
        for name, module in sorted(expected_items)
        if isinstance(module, nn.Module)
    )
    module_tree_semantic_sha256 = _module_tree_semantic_sha256(
        sorted_expected,
        require_cpu=True,
    )
    _module_records(sorted_expected, aliases={})
    sorted_factories = tuple(sorted(factory_items))
    payload = {
        "authority_schema": "h6-checkpoint-factory-authority-v3",
        "attempt_spec_sha256": attempt_spec.attempt_spec_sha256,
        "factory_sha256s": factory_sha256s,
        "module_tree_semantic_sha256": module_tree_semantic_sha256,
    }
    issuance_nonce = object()
    authority = _H6CheckpointFactoryAuthorityV3(
        authority_schema="h6-checkpoint-factory-authority-v3",
        attempt_spec_sha256=attempt_spec.attempt_spec_sha256,
        factory_sha256s=factory_sha256s,
        module_tree_semantic_sha256=module_tree_semantic_sha256,
        authority_sha256=_owned_hash(
            "vfe4.h6.checkpoint-factory-authority.v3",
            payload,
        ),
        _module_factories=sorted_factories,
        _issuance_nonce=issuance_nonce,
    )
    with _FACTORY_AUTHORITY_REGISTRY_LOCK:
        _FACTORY_AUTHORITY_REGISTRY[issuance_nonce] = (
            id(authority),
            authority,
        )
    return authority


def _consume_h6_checkpoint_factory_authority_v3(
    authority: _H6CheckpointFactoryAuthorityV3,
) -> None:
    with _FACTORY_AUTHORITY_REGISTRY_LOCK:
        issued = _FACTORY_AUTHORITY_REGISTRY.get(authority._issuance_nonce)
        if (
            issued is None
            or issued[0] != id(authority)
            or issued[1] is not authority
        ):
            raise ValueError(
                "checkpoint factory authority is not an unused sealed issuance"
            )
        del _FACTORY_AUTHORITY_REGISTRY[authority._issuance_nonce]


def _optimizer_groups_signature(
    records: tuple[H6AdamWRecordV3, ...],
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "name": record.name,
            "groups": tuple(group.canonical_payload() for group in record.groups),
        }
        for record in records
    )


def _construct_optimizer(
    record: H6AdamWRecordV3,
    *,
    parameters: dict[str, nn.Parameter],
) -> torch.optim.AdamW:
    groups: list[dict[str, object]] = []
    for group in record.groups:
        values = {name: value for name, value in group.hyperparameters}
        values["params"] = [
            parameters[parameter_name] for parameter_name in group.parameter_names
        ]
        groups.append(values)
    return torch.optim.AdamW(groups)


def _load_module_state(
    *,
    modules: tuple[tuple[str, nn.Module], ...],
    records: tuple[H6TensorRecordV3, ...],
) -> None:
    for module_name, module in modules:
        prefix = f"{module_name}."
        state = {
            record.name[len(prefix) :]: record.decode_cpu()
            for record in records
            if record.name.startswith(prefix)
        }
        result = module.load_state_dict(state, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise ValueError("fresh module inventory changed while loading")


def _load_optimizer_state(
    *,
    optimizers: tuple[tuple[str, torch.optim.AdamW], ...],
    records: tuple[H6AdamWRecordV3, ...],
    parameters: dict[str, nn.Parameter],
) -> None:
    optimizer_by_name = dict(optimizers)
    for optimizer_record in records:
        optimizer = optimizer_by_name[optimizer_record.name]
        for state_record in optimizer_record.states:
            if not state_record.tensors and not state_record.scalars:
                continue
            parameter = parameters[state_record.parameter_name]
            state: dict[str, Tensor | int | float] = {
                name: value for name, value in state_record.scalars
            }
            state.update(
                {
                    tensor.state_name: tensor.decode_cpu()
                    for tensor in state_record.tensors
                    if tensor.state_name is not None
                }
            )
            optimizer.state[parameter] = state


def _module_roots(
    records: tuple[H6TensorRecordV3, ...],
) -> tuple[str, ...]:
    return tuple(sorted({record.name.split(".", 1)[0] for record in records}))


def _validate_loaded_bytes(
    checkpoint: H6CheckpointV3,
    *,
    modules: tuple[tuple[str, nn.Module], ...],
    optimizers: tuple[tuple[str, torch.optim.AdamW], ...],
) -> None:
    aliases: dict[tuple[str, int | None, int], str] = {}
    module_records, parameters, _ = _module_records(
        modules,
        aliases=aliases,
    )
    optimizer_records, _ = _optimizer_records(
        optimizers,
        parameters=parameters,
        aliases=aliases,
    )
    if tuple(record.manifest_payload() for record in module_records) != tuple(
        record.manifest_payload() for record in checkpoint.module_tensors
    ):
        raise ValueError("hydrated module bytes do not match checkpoint")
    if tuple(record.canonical_payload() for record in optimizer_records) != tuple(
        record.canonical_payload() for record in checkpoint.optimizers
    ):
        raise ValueError("hydrated optimizer bytes do not match checkpoint")


def _move_active_state(
    *,
    modules: tuple[tuple[str, nn.Module], ...],
    optimizers: tuple[tuple[str, torch.optim.AdamW], ...],
    device: torch.device,
) -> None:
    parameter_ids = {
        name: tuple(id(parameter) for parameter in module.parameters())
        for name, module in modules
    }
    for _, module in modules:
        module.to(device=device)
    for name, module in modules:
        if (
            tuple(id(parameter) for parameter in module.parameters())
            != (parameter_ids[name])
        ):
            raise RuntimeError(
                "device move replaced parameters after optimizer construction"
            )
        for tensor in module.state_dict().values():
            if tensor.device != device:
                raise RuntimeError("module device move exposed fallback")
            if tensor.is_floating_point() and tensor.dtype is not torch.float64:
                raise RuntimeError("module device move changed float64 policy")
    for _, optimizer in optimizers:
        for state in optimizer.state.values():
            for name, value in tuple(state.items()):
                if isinstance(value, Tensor):
                    state[name] = value.to(device=device)
        for state in optimizer.state.values():
            for value in state.values():
                if isinstance(value, Tensor) and value.device != device:
                    raise RuntimeError("optimizer state device move exposed fallback")


def _move_detached_batch_snapshot(
    snapshot: H6DetachedBatchRecognitionSnapshotV3 | None,
    *,
    device: torch.device,
) -> H6DetachedBatchRecognitionSnapshotV3 | None:
    if snapshot is None:
        return None
    moved_states = tuple(
        replace(
            state,
            _tensors=tuple(
                tensor.to(device=device).contiguous() for tensor in state._tensors
            ),
        )
        for state in snapshot.states
    )
    moved = replace(snapshot, states=moved_states)
    if any(
        tensor.device != device for state in moved.states for tensor in state._tensors
    ):
        raise RuntimeError("detached recognition snapshot device move exposed fallback")
    return moved


@dataclass(frozen=True, slots=True)
class H6HydratedCheckpointV3:
    """Fresh modules/optimizers positioned at the checkpoint's next phase."""

    named_modules: tuple[tuple[str, nn.Module], ...]
    named_optimizers: tuple[tuple[str, torch.optim.AdamW], ...]
    cursor: H6AttemptCursorV3
    checkpoint_sha256: str
    authorized_device: str
    detached_batch_snapshot: H6DetachedBatchRecognitionSnapshotV3 | None

    def __post_init__(self) -> None:
        _require_sha256(self.checkpoint_sha256, "checkpoint_sha256")
        if type(self.cursor) is not H6AttemptCursorV3:
            raise ValueError("hydrated cursor must be an exact v3 cursor")
        self.cursor.__post_init__()
        _named_items(self.named_modules, label="hydrated named_modules")
        _named_items(self.named_optimizers, label="hydrated named_optimizers")
        _require_name(self.authorized_device, "authorized_device")
        if self.detached_batch_snapshot is not None:
            if (
                type(self.detached_batch_snapshot)
                is not H6DetachedBatchRecognitionSnapshotV3
            ):
                raise ValueError("hydrated detached snapshot has the wrong type")
            self.detached_batch_snapshot.__post_init__()
            if (
                self.detached_batch_snapshot.post_recognition_cursor_sha256
                != self.cursor.cursor_sha256
            ):
                raise ValueError("hydrated detached snapshot left its exact cursor")


def hydrate_h6_checkpoint_v3(
    checkpoint: H6CheckpointV3,
    *,
    expected_attempt_spec: H6AttemptSpecV3,
    expected_runtime_identity: H6PredictionRuntimeIdentity,
    live_deterministic_policy_sha256: str,
    factory_authority: _H6CheckpointFactoryAuthorityV3,
    authorized_device: str,
    allow_synthetic_cpu: bool = False,
) -> H6HydratedCheckpointV3:
    """Hydrate in CPU module/group/state/validate/device order, or refuse."""

    if type(checkpoint) is not H6CheckpointV3:
        raise ValueError("checkpoint must be an exact H6CheckpointV3")
    checkpoint.__post_init__()
    if type(expected_attempt_spec) is not H6AttemptSpecV3:
        raise ValueError("expected_attempt_spec must be exact v3")
    expected_attempt_spec.__post_init__()
    if expected_attempt_spec != checkpoint.attempt_spec:
        raise RuntimeError("checkpoint attempt identity drift")
    if type(expected_runtime_identity) is not H6PredictionRuntimeIdentity:
        raise ValueError("expected_runtime_identity must be exact v3")
    expected_runtime_identity.__post_init__()
    if expected_runtime_identity != checkpoint.runtime_identity:
        raise RuntimeError("checkpoint runtime identity drift")
    _require_sha256(
        live_deterministic_policy_sha256,
        "live_deterministic_policy_sha256",
    )
    if live_deterministic_policy_sha256 != checkpoint.deterministic_policy_sha256:
        raise RuntimeError("checkpoint deterministic policy drift")
    if type(allow_synthetic_cpu) is not bool:
        raise ValueError("allow_synthetic_cpu must be an exact bool")
    if type(authorized_device) is not str:
        raise ValueError("authorized_device must be a string")
    if authorized_device == "cpu":
        if not allow_synthetic_cpu:
            raise RuntimeError("CPU hydration is limited to bounded synthetic fixtures")
    elif authorized_device != expected_runtime_identity.training_device:
        raise RuntimeError("checkpoint authorized device drift")
    device = torch.device(authorized_device)

    # 1. Validate the independently issued semantic authority, then construct
    # each fresh CPU module exactly once.  Constructor outputs remain
    # untrusted until their exact semantic/type tree matches the authority.
    if type(factory_authority) is not _H6CheckpointFactoryAuthorityV3:
        raise ValueError(
            "factory_authority must be an exact binder-issued v3 authority"
        )
    factory_authority.__post_init__()
    if (
        factory_authority.attempt_spec_sha256
        != expected_attempt_spec.attempt_spec_sha256
        or factory_authority.factory_sha256s
        != _expected_factory_sha256s(expected_attempt_spec)
    ):
        raise RuntimeError("checkpoint factory authority drift")
    factory_items = factory_authority._module_factories
    expected_roots = _module_roots(checkpoint.module_tensors)
    if tuple(name for name, _ in factory_items) != expected_roots:
        raise ValueError("factory authority inventory does not match checkpoint")
    _consume_h6_checkpoint_factory_authority_v3(factory_authority)
    constructed: list[tuple[str, nn.Module]] = []
    for name, factory in factory_items:
        module = factory(expected_attempt_spec)
        if not isinstance(module, nn.Module):
            raise ValueError("authorized module factory must return a torch module")
        constructed.append((name, module))
    modules = tuple(constructed)
    if (
        _module_tree_semantic_sha256(modules, require_cpu=True)
        != factory_authority.module_tree_semantic_sha256
    ):
        raise ValueError(
            "constructed module semantic/type signature differs from authority"
        )
    aliases: dict[tuple[str, int | None, int], str] = {}
    fresh_records, parameters, _ = _module_records(
        modules,
        aliases=aliases,
    )
    if _module_signature(fresh_records) != _module_signature(checkpoint.module_tensors):
        raise ValueError("fresh CPU module inventory does not match checkpoint")
    # 2-3. Inventory is exact; construct AdamW groups from stable names.
    optimizers = tuple(
        (
            record.name,
            _construct_optimizer(record, parameters=parameters),
        )
        for record in checkpoint.optimizers
    )
    group_aliases: dict[tuple[str, int | None, int], str] = {}
    _, group_parameters, _ = _module_records(
        modules,
        aliases=group_aliases,
    )
    empty_optimizer_records, _ = _optimizer_records(
        optimizers,
        parameters=group_parameters,
        aliases=group_aliases,
    )
    if _optimizer_groups_signature(
        empty_optimizer_records
    ) != _optimizer_groups_signature(checkpoint.optimizers):
        raise ValueError("fresh AdamW group inventory does not match checkpoint")

    # 4. Decode and load canonical module and optimizer state on CPU.
    _load_module_state(modules=modules, records=checkpoint.module_tensors)
    _load_optimizer_state(
        optimizers=optimizers,
        records=checkpoint.optimizers,
        parameters=parameters,
    )

    # 5. Revalidate every tensor byte/hash and the complete next-phase cursor.
    checkpoint.cursor.__post_init__()
    if (
        checkpoint.cursor.attempt_spec_sha256
        != expected_attempt_spec.attempt_spec_sha256
    ):
        raise RuntimeError("checkpoint cursor attempt identity drift")
    _validate_loaded_bytes(
        checkpoint,
        modules=modules,
        optimizers=optimizers,
    )

    # 6. Only validated active state may move to the authorized device.
    _move_active_state(
        modules=modules,
        optimizers=optimizers,
        device=device,
    )
    detached_batch_snapshot = _move_detached_batch_snapshot(
        checkpoint.detached_batch_snapshot,
        device=device,
    )

    # 7. The caller resumes from this exact next phase without replay.
    return H6HydratedCheckpointV3(
        named_modules=modules,
        named_optimizers=optimizers,
        cursor=checkpoint.cursor,
        checkpoint_sha256=checkpoint.checkpoint_sha256,
        authorized_device=str(device),
        detached_batch_snapshot=detached_batch_snapshot,
    )


__all__ = [
    "H6AdamWGroupV3",
    "H6AdamWParameterStateV3",
    "H6AdamWRecordV3",
    "H6CheckpointV3",
    "H6HydratedCheckpointV3",
    "H6TensorRecordV3",
    "capture_h6_checkpoint_v3",
    "decode_h6_checkpoint_v3",
    "hydrate_h6_checkpoint_v3",
    "read_h6_checkpoint_file_v3",
]
