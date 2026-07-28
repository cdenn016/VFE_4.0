"""Canonical H6 Prediction v3 checkpoint capture and exact hydration.

The v3 codec never serializes PyTorch object IDs.  Module state is keyed by
qualified parameter or buffer name, and AdamW groups/state are keyed by the
same stable parameter names.  All tensor payloads are canonical contiguous
CPU little-endian bytes before a checkpoint is accepted.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
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
)


_MAGIC = b"VFE4-H6-CHECKPOINT-V3\x00"
_TRAILER_BYTES = hashlib.sha256().digest_size
_MAX_HEADER_BYTES = 64 * 1024 * 1024
_LOWER_HEX = frozenset("0123456789abcdef")
_MAX_RECORDS = 1_000_000
_MODULE_ROLES = frozenset({"module_parameter", "module_buffer"})
_TENSOR_ROLES = _MODULE_ROLES | {"optimizer_state"}
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


@dataclass(frozen=True, slots=True)
class H6TensorRecordV3:
    """One canonical tensor payload with explicit semantic ownership."""

    role: Literal[
        "module_parameter",
        "module_buffer",
        "optimizer_state",
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


def _checkpoint_payload(
    *,
    attempt_spec: H6AttemptSpecV3,
    cursor: H6AttemptCursorV3,
    objective_manifest: H6ObjectiveManifestV3,
    runtime_identity: H6PredictionRuntimeIdentity,
    deterministic_policy_sha256: str,
    module_tensors: tuple[H6TensorRecordV3, ...],
    optimizers: tuple[H6AdamWRecordV3, ...],
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
        if (
            self.objective_manifest.counter_consumption_sha256
            != self.cursor.counter_consumption_sha256
        ):
            raise ValueError(
                "checkpoint objective/cursor counter consumption disagrees"
            )
        expected_objective_phase = {
            TrainingPhase.MODEL_CE_ADAMW: TrainingPhase.MODEL_CE_ADAMW,
            TrainingPhase.RECOGNITION_ADAMW: TrainingPhase.MODEL_ADAMW,
            TrainingPhase.IMMUTABLE_DETACHED_SNAPSHOT: (
                TrainingPhase.RECOGNITION_ADAMW
            ),
            TrainingPhase.MODEL_ADAMW: TrainingPhase.RECOGNITION_ADAMW,
        }[self.cursor.next_phase]
        is_cross_entropy = self.attempt_spec.objective_kind == "cross_entropy"
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
        return self.module_tensors + optimizer_tensors

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
            frozen = (
                _json_tuple(
                    raw_value,
                    label="checkpoint optimizer hyperparameter value",
                )
                if type(raw_value) is list
                else raw_value
            )
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
            scalars.append((scalar_name, scalar_value))
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
        checkpoint_sha256=header["checkpoint_sha256"],  # type: ignore[arg-type]
    )
    if (
        canonical_json_bytes(checkpoint.canonical_payload()) != header_bytes
        or checkpoint.to_bytes() != raw
    ):
        raise ValueError("checkpoint canonical encoding changed during decode")
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


@dataclass(frozen=True, slots=True)
class H6HydratedCheckpointV3:
    """Fresh modules/optimizers positioned at the checkpoint's next phase."""

    named_modules: tuple[tuple[str, nn.Module], ...]
    named_optimizers: tuple[tuple[str, torch.optim.AdamW], ...]
    cursor: H6AttemptCursorV3
    checkpoint_sha256: str
    authorized_device: str

    def __post_init__(self) -> None:
        _require_sha256(self.checkpoint_sha256, "checkpoint_sha256")
        if type(self.cursor) is not H6AttemptCursorV3:
            raise ValueError("hydrated cursor must be an exact v3 cursor")
        self.cursor.__post_init__()
        _named_items(self.named_modules, label="hydrated named_modules")
        _named_items(self.named_optimizers, label="hydrated named_optimizers")
        _require_name(self.authorized_device, "authorized_device")


def hydrate_h6_checkpoint_v3(
    checkpoint: H6CheckpointV3,
    *,
    expected_attempt_spec: H6AttemptSpecV3,
    expected_runtime_identity: H6PredictionRuntimeIdentity,
    live_deterministic_policy_sha256: str,
    module_factories: object,
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

    # 1. Construct fresh CPU float64 modules from the bound attempt.
    factory_items = _named_items(module_factories, label="module_factories")
    expected_roots = _module_roots(checkpoint.module_tensors)
    if tuple(sorted(name for name, _ in factory_items)) != expected_roots:
        raise ValueError("module factory inventory does not match checkpoint")
    constructed: list[tuple[str, nn.Module]] = []
    for name, factory in sorted(factory_items, key=lambda item: item[0]):
        if not callable(factory):
            raise ValueError("module factory must be callable")
        module = factory(expected_attempt_spec)
        if not isinstance(module, nn.Module):
            raise ValueError("module factory must return a torch module")
        constructed.append((name, module))
    modules = tuple(constructed)
    aliases: dict[tuple[str, int | None, int], str] = {}
    fresh_records, parameters, _ = _module_records(
        modules,
        aliases=aliases,
    )
    if _module_signature(fresh_records) != _module_signature(checkpoint.module_tensors):
        raise ValueError("fresh CPU module inventory does not match checkpoint")
    if any(
        tensor.device.type != "cpu"
        for _, module in modules
        for tensor in module.state_dict().values()
    ):
        raise ValueError("fresh modules must be constructed on CPU")

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

    # 7. The caller resumes from this exact next phase without replay.
    return H6HydratedCheckpointV3(
        named_modules=modules,
        named_optimizers=optimizers,
        cursor=checkpoint.cursor,
        checkpoint_sha256=checkpoint.checkpoint_sha256,
        authorized_device=str(device),
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
]
