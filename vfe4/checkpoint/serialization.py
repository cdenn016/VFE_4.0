"""Exact primitive/tensor checkpoint serialization and scientific hashing."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import math
import struct

import torch

from .schema import (
    CheckpointSchemaError,
    CheckpointSecurityError,
    ResumeContract,
)


SCIENTIFIC_STATE_KEYS = frozenset(
    {
        "model_state",
        "recognition_state",
        "optimizer_state",
        "scheduler_state",
        "amp_scaler_state",
        "rng_state",
        "estimator_state",
        "data_cursor_state",
        "update_trace_state",
        "metric_state",
        "next_prediction_fixture",
    }
)
_ALLOWED_TENSOR_DTYPES = frozenset(
    {
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }
)
_MAXIMUM_TENSOR_RANK = 16
_SCIENTIFIC_STATE_DOMAIN = b"vfe4.wt103.scientific-state.v1\x00"


@dataclasses.dataclass(frozen=True, slots=True, order=True)
class TensorInventoryEntry:
    path: str
    dtype: str
    shape: tuple[int, ...]
    numel: int
    nbytes: int
    tensor_sha256: str

    def __post_init__(self) -> None:
        if type(self.path) is not str or not self.path:
            raise CheckpointSchemaError("tensor inventory path is invalid")
        if type(self.dtype) is not str or not self.dtype.startswith("torch."):
            raise CheckpointSchemaError("tensor inventory dtype is invalid")
        if (
            type(self.shape) is not tuple
            or len(self.shape) > _MAXIMUM_TENSOR_RANK
            or any(
                type(dimension) is not int or dimension < 0 for dimension in self.shape
            )
        ):
            raise CheckpointSchemaError("tensor inventory shape is invalid")
        if type(self.numel) is not int or self.numel < 0:
            raise CheckpointSchemaError("tensor inventory numel is invalid")
        if type(self.nbytes) is not int or self.nbytes < 0:
            raise CheckpointSchemaError("tensor inventory nbytes is invalid")
        if (
            type(self.tensor_sha256) is not str
            or len(self.tensor_sha256) != 64
            or any(
                character not in "0123456789abcdef" for character in self.tensor_sha256
            )
        ):
            raise CheckpointSchemaError("tensor inventory hash is invalid")

    def payload(self) -> dict[str, object]:
        return {
            "path": self.path,
            "dtype": self.dtype,
            "shape": self.shape,
            "numel": self.numel,
            "nbytes": self.nbytes,
            "tensor_sha256": self.tensor_sha256,
        }


@dataclasses.dataclass(slots=True)
class _Traversal:
    tensor_count: int = 0
    total_tensor_bytes: int = 0
    item_count: int = 0
    inventory: list[TensorInventoryEntry] = dataclasses.field(default_factory=list)


def _forbidden_manifest_key(key: str) -> bool:
    normalized = key.casefold().replace("-", "_")
    return "run_manifest" in normalized or "terminal_manifest" in normalized


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    if tensor.device.type != "cpu" or not tensor.is_contiguous():
        raise CheckpointSchemaError("tensor bytes require contiguous CPU storage")
    flattened = tensor.reshape(-1)
    if flattened.numel() == 0:
        return b""
    return flattened.view(torch.uint8).numpy().tobytes(order="C")


def _key_path(path: str, key: str | int) -> str:
    if type(key) is str:
        escaped = (
            key.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
            .replace("\r", "\\r")
        )
        return f'{path}["{escaped}"]'
    return f"{path}[{key}]"


def _record_item(traversal: _Traversal, contract: ResumeContract) -> None:
    traversal.item_count += 1
    if traversal.item_count > contract.maximum_container_items:
        raise CheckpointSchemaError("recursive checkpoint item count exceeds its bound")


def _normalize_value(
    value: object,
    *,
    path: str,
    depth: int,
    contract: ResumeContract,
    traversal: _Traversal,
    require_cpu: bool,
    clone_tensors: bool,
) -> object:
    if depth > contract.maximum_recursion_depth:
        raise CheckpointSchemaError("checkpoint recursion depth exceeds its bound")
    _record_item(traversal, contract)
    if type(value) is torch.Tensor:
        tensor = value
        if tensor.layout is not torch.strided:
            raise CheckpointSchemaError(
                "checkpoint tensors must use exact dense strided layout"
            )
        if tensor.dtype not in _ALLOWED_TENSOR_DTYPES:
            raise CheckpointSchemaError(
                f"checkpoint tensor dtype is not allowed: {tensor.dtype}"
            )
        if tensor.dim() > _MAXIMUM_TENSOR_RANK:
            raise CheckpointSchemaError("checkpoint tensor rank exceeds its bound")
        numel = tensor.numel()
        nbytes = numel * tensor.element_size()
        if nbytes > contract.maximum_tensor_bytes:
            raise CheckpointSchemaError(
                "checkpoint tensor bytes exceed the per-tensor bound"
            )
        traversal.tensor_count += 1
        traversal.total_tensor_bytes += nbytes
        if traversal.tensor_count > contract.maximum_tensor_count:
            raise CheckpointSchemaError("checkpoint tensor count exceeds its bound")
        if traversal.total_tensor_bytes > contract.maximum_total_tensor_bytes:
            raise CheckpointSchemaError(
                "checkpoint tensor bytes exceed the aggregate bound"
            )
        if require_cpu and tensor.device.type != "cpu":
            raise CheckpointSecurityError(
                "weights-only checkpoint tensors must load on CPU"
            )
        if tensor.device.type == "meta":
            raise CheckpointSchemaError("meta tensors cannot be checkpointed")
        normalized = tensor.detach().to(device="cpu").contiguous()
        if clone_tensors:
            normalized = normalized.clone()
        raw = _tensor_bytes(normalized)
        if len(raw) != nbytes:
            raise CheckpointSchemaError(
                "tensor byte inventory differs from dtype/shape/numel"
            )
        traversal.inventory.append(
            TensorInventoryEntry(
                path=path,
                dtype=str(normalized.dtype),
                shape=tuple(normalized.shape),
                numel=numel,
                nbytes=nbytes,
                tensor_sha256=hashlib.sha256(raw).hexdigest(),
            )
        )
        return normalized
    if value is None or type(value) in (bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise CheckpointSchemaError("checkpoint primitive floats must be finite")
        return value
    if type(value) is str:
        if len(value.encode("utf-8")) > contract.maximum_checkpoint_bytes:
            raise CheckpointSchemaError("checkpoint string exceeds the payload bound")
        return value
    if type(value) is dict:
        if len(value) > contract.maximum_container_items:
            raise CheckpointSchemaError("checkpoint dict exceeds its item bound")
        result: dict[str | int, object] = {}
        for key, item in value.items():
            if type(key) not in (str, int):
                raise CheckpointSchemaError(
                    "checkpoint dict keys must be exact str or int primitives"
                )
            if type(key) is str and _forbidden_manifest_key(key):
                raise CheckpointSchemaError(
                    "checkpoint state cannot depend on a later run manifest; "
                    "that dependency would be circular"
                )
            result[key] = _normalize_value(
                item,
                path=_key_path(path, key),
                depth=depth + 1,
                contract=contract,
                traversal=traversal,
                require_cpu=require_cpu,
                clone_tensors=clone_tensors,
            )
        return result
    if type(value) in (list, tuple):
        if len(value) > contract.maximum_container_items:
            raise CheckpointSchemaError("checkpoint sequence exceeds its item bound")
        normalized_items = tuple(
            _normalize_value(
                item,
                path=f"{path}[{index}]",
                depth=depth + 1,
                contract=contract,
                traversal=traversal,
                require_cpu=require_cpu,
                clone_tensors=clone_tensors,
            )
            for index, item in enumerate(value)
        )
        return list(normalized_items) if type(value) is list else normalized_items
    raise CheckpointSchemaError(
        "unsupported checkpoint value outside the exact "
        "primitive/tensor whitelist: "
        f"{type(value).__name__}"
    )


def _require_mapping_state(
    state: dict[str, object],
    *,
    key: str,
    allow_none: bool = False,
) -> dict[object, object] | None:
    value = state[key]
    if allow_none and value is None:
        return None
    if type(value) is not dict or not value:
        raise CheckpointSchemaError(f"{key} must be a complete nonempty exact dict")
    return value


def _validate_scientific_structure(state: dict[str, object]) -> None:
    _require_mapping_state(state, key="model_state")
    _require_mapping_state(
        state,
        key="recognition_state",
        allow_none=True,
    )
    _require_mapping_state(state, key="optimizer_state")
    _require_mapping_state(state, key="scheduler_state")
    _require_mapping_state(
        state,
        key="amp_scaler_state",
        allow_none=True,
    )
    rng = _require_mapping_state(state, key="rng_state")
    if rng is None or set(rng) != {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
    }:
        raise CheckpointSchemaError(
            "RNG state must completely cover Python, NumPy, PyTorch CPU, "
            "and every ordered CUDA-device stream"
        )
    if (
        type(rng["python"]) is not tuple
        or type(rng["numpy"]) is not tuple
        or type(rng["torch_cpu"]) is not torch.Tensor
        or rng["torch_cpu"].dtype is not torch.uint8
        or type(rng["torch_cuda"]) is not tuple
        or any(
            type(item) is not torch.Tensor or item.dtype is not torch.uint8
            for item in rng["torch_cuda"]
        )
    ):
        raise CheckpointSchemaError(
            "RNG state has an incomplete or noncanonical stream representation"
        )
    estimator = _require_mapping_state(state, key="estimator_state")
    if estimator is None or "stream_counters" not in estimator:
        raise CheckpointSchemaError(
            "estimator state must contain its complete counter streams"
        )
    cursor = _require_mapping_state(state, key="data_cursor_state")
    if cursor is None or not {
        "pass_index",
        "batch_index",
        "next_window_ids",
        "permutation_bytes",
        "permutation_sha256",
    }.issubset(cursor):
        raise CheckpointSchemaError(
            "data cursor state must contain position, next windows, "
            "and permutation identity"
        )
    if (
        type(cursor["permutation_bytes"]) is not torch.Tensor
        or cursor["permutation_bytes"].dtype is not torch.uint8
    ):
        raise CheckpointSchemaError(
            "data cursor permutation bytes must be one exact uint8 tensor"
        )
    update_trace = _require_mapping_state(state, key="update_trace_state")
    if update_trace is None or not {
        "global_step",
        "successful_updates",
        "rejected_updates",
        "counted_targets",
    }.issubset(update_trace):
        raise CheckpointSchemaError(
            "update trace state must contain complete step/update/target counters"
        )
    metric = _require_mapping_state(state, key="metric_state")
    if metric is None or not {
        "next_ordinal",
        "hash_chain_head",
        "nll_numerator",
        "nll_denominator",
        "failure_ledger_head",
    }.issubset(metric):
        raise CheckpointSchemaError(
            "metric state must contain its next ordinal, chain heads, "
            "and scientific numerator/denominator projection"
        )
    fixture = state["next_prediction_fixture"]
    if (
        type(fixture) is not tuple
        or len(fixture) != 2
        or any(type(item) is not torch.Tensor for item in fixture)
    ):
        raise CheckpointSchemaError(
            "next prediction fixture must contain exactly two tensor batches"
        )


def normalize_scientific_state(
    state: object,
    *,
    contract: ResumeContract,
    require_cpu: bool,
    clone_tensors: bool,
) -> tuple[
    dict[str, object],
    tuple[TensorInventoryEntry, ...],
    int,
]:
    if type(contract) is not ResumeContract:
        raise CheckpointSchemaError(
            "scientific serialization requires an exact ResumeContract"
        )
    contract.__post_init__()
    if type(state) is not dict or set(state) != SCIENTIFIC_STATE_KEYS:
        raise CheckpointSchemaError(
            "scientific state keys must exactly cover model, recognition, "
            "optimizer, scheduler, AMP, RNG, estimator, cursor, update trace, "
            "metric state, and next predictions"
        )
    _validate_scientific_structure(state)
    traversal = _Traversal()
    normalized = _normalize_value(
        state,
        path="$",
        depth=0,
        contract=contract,
        traversal=traversal,
        require_cpu=require_cpu,
        clone_tensors=clone_tensors,
    )
    if type(normalized) is not dict:
        raise CheckpointSchemaError("scientific state did not remain an exact dict")
    inventory = tuple(sorted(traversal.inventory))
    if len({entry.path for entry in inventory}) != len(inventory):
        raise CheckpointSchemaError("tensor inventory paths are not unique")
    return normalized, inventory, traversal.total_tensor_bytes


def validate_whitelist_tree(
    value: object,
    *,
    contract: ResumeContract,
    require_cpu: bool,
) -> object:
    traversal = _Traversal()
    return _normalize_value(
        value,
        path="$",
        depth=0,
        contract=contract,
        traversal=traversal,
        require_cpu=require_cpu,
        clone_tensors=False,
    )


def _length_prefixed(tag: bytes, payload: bytes) -> bytes:
    return tag + len(payload).to_bytes(8, "big") + payload


def _canonical_key_bytes(key: str | int) -> bytes:
    if type(key) is str:
        return _length_prefixed(b"K", key.encode("utf-8"))
    if type(key) is int:
        return _length_prefixed(b"J", str(key).encode("ascii"))
    raise CheckpointSchemaError("canonical checkpoint key is unsupported")


def _canonical_value_bytes(value: object) -> bytes:
    if value is None:
        return b"N"
    if type(value) is bool:
        return b"B1" if value else b"B0"
    if type(value) is int:
        return _length_prefixed(b"I", str(value).encode("ascii"))
    if type(value) is float:
        return b"F" + struct.pack(">d", value)
    if type(value) is str:
        return _length_prefixed(b"S", value.encode("utf-8"))
    if type(value) is torch.Tensor:
        tensor = value
        dtype = str(tensor.dtype).encode("ascii")
        shape = b"".join(
            int(dimension).to_bytes(8, "big", signed=False)
            for dimension in tensor.shape
        )
        raw = _tensor_bytes(tensor)
        return (
            _length_prefixed(b"Y", dtype)
            + _length_prefixed(b"H", shape)
            + _length_prefixed(b"R", raw)
        )
    if type(value) in (list, tuple):
        tag = b"L" if type(value) is list else b"T"
        children = b"".join(_canonical_value_bytes(item) for item in value)
        return tag + len(value).to_bytes(8, "big") + children
    if type(value) is dict:
        rows = tuple(
            sorted(
                (
                    _canonical_key_bytes(key),
                    _canonical_value_bytes(item),
                )
                for key, item in value.items()
            )
        )
        return (
            b"D"
            + len(rows).to_bytes(8, "big")
            + b"".join(key + item for key, item in rows)
        )
    raise CheckpointSchemaError(
        f"canonical checkpoint value is unsupported: {type(value).__name__}"
    )


def scientific_state_sha256(state: dict[str, object]) -> str:
    return hashlib.sha256(
        _SCIENTIFIC_STATE_DOMAIN + _canonical_value_bytes(state)
    ).hexdigest()


def inventory_payload(
    inventory: tuple[TensorInventoryEntry, ...],
) -> tuple[dict[str, object], ...]:
    if (
        type(inventory) is not tuple
        or any(type(entry) is not TensorInventoryEntry for entry in inventory)
        or tuple(sorted(inventory)) != inventory
    ):
        raise CheckpointSchemaError("tensor inventory is not exact and sorted")
    return tuple(entry.payload() for entry in inventory)


def inventory_from_payload(
    value: object,
) -> tuple[TensorInventoryEntry, ...]:
    if type(value) is not tuple:
        raise CheckpointSecurityError(
            "checkpoint tensor inventory must be an exact tuple"
        )
    entries: list[TensorInventoryEntry] = []
    expected_keys = {
        "path",
        "dtype",
        "shape",
        "numel",
        "nbytes",
        "tensor_sha256",
    }
    for row in value:
        if type(row) is not dict or set(row) != expected_keys:
            raise CheckpointSecurityError(
                "checkpoint tensor inventory row is not exact"
            )
        try:
            entries.append(TensorInventoryEntry(**row))
        except (CheckpointSchemaError, TypeError) as exc:
            raise CheckpointSecurityError(
                "checkpoint tensor inventory row is invalid"
            ) from exc
    inventory = tuple(entries)
    if inventory != tuple(sorted(inventory)) or len(
        {entry.path for entry in inventory}
    ) != len(inventory):
        raise CheckpointSecurityError(
            "checkpoint tensor inventory is not exact, unique, and sorted"
        )
    return inventory


def serialize_checkpoint_envelope(envelope: object) -> bytes:
    buffer = io.BytesIO()
    try:
        torch.save(envelope, buffer)
    except Exception as exc:
        raise CheckpointSchemaError(
            f"checkpoint tensor serialization failed: {type(exc).__name__}"
        ) from exc
    return buffer.getvalue()


def deserialize_weights_only_cpu(payload: bytes) -> object:
    if type(payload) is not bytes:
        raise CheckpointSecurityError("checkpoint payload must be exact bytes")
    try:
        return torch.load(
            io.BytesIO(payload),
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise CheckpointSecurityError(
            "weights-only safe checkpoint load rejected the payload"
        ) from exc


__all__ = [
    "SCIENTIFIC_STATE_KEYS",
    "TensorInventoryEntry",
    "deserialize_weights_only_cpu",
    "inventory_from_payload",
    "inventory_payload",
    "normalize_scientific_state",
    "scientific_state_sha256",
    "serialize_checkpoint_envelope",
    "validate_whitelist_tree",
]
