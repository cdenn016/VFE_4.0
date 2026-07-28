"""Phase-owned H6 training engine contracts."""

from __future__ import annotations

import hashlib
import math
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from vfe4.types.h6 import TrainingPhase, canonical_json_bytes
from vfe4.types.h6_prediction_v3 import (
    H6AttemptCursorV3,
    H6_NO_COUNTER_CONSUMPTION_SHA256,
)

from .matching import H6_ADAMW_POLICY


_LOWER_HEX = frozenset("0123456789abcdef")
_ABSENT_CATEGORICAL_SUPPORT: tuple[None] = (None,)


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


def _require_nonempty(value: object, name: str) -> str:
    if type(value) is not str or not value or "\x00" in value:
        raise ValueError(f"{name} must be a nonempty string without NUL")
    return value


def _tensor_raw_bytes(tensor: Tensor) -> bytes:
    if tensor.layout is not torch.strided or tensor.is_quantized:
        raise ValueError("H6 engine tensors must be dense and nonquantized")
    if tensor.device.type == "meta":
        raise ValueError("H6 engine tensors cannot use the meta device")
    cpu = tensor.detach().to(device="cpu").contiguous()
    raw = bytes(cpu.reshape(-1).view(torch.uint8).tolist())
    if sys.byteorder == "little" or cpu.element_size() == 1:
        return raw
    width = cpu.element_size()
    return b"".join(
        raw[offset : offset + width][::-1] for offset in range(0, len(raw), width)
    )


def _tensor_manifest(tensor: Tensor) -> dict[str, object]:
    raw = _tensor_raw_bytes(tensor)
    return {
        "dtype": str(tensor.dtype),
        "shape": tuple(tensor.shape),
        "raw_bytes_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _recognition_inventory_names(
    *,
    receiver_count: int,
    receiver_components: tuple[tuple[int, tuple[str, ...]], ...],
    state_categorical_enabled: bool,
    state_categorical_supports: tuple[tuple[int, ...], ...] | tuple[None],
    model_categorical_enabled: bool,
    model_categorical_supports: tuple[tuple[int, ...], ...] | tuple[None],
) -> tuple[str, ...]:
    if type(receiver_count) is not int or receiver_count < 2:
        raise ValueError("recognition receiver count must represent 0..T")
    if (
        type(receiver_components) is not tuple
        or len(receiver_components) != receiver_count
    ):
        raise ValueError(
            "recognition continuous receiver inventory must cover exactly 0..T"
        )
    receivers: list[int] = []
    names: list[str] = []
    for item in receiver_components:
        if type(item) is not tuple or len(item) != 2:
            raise ValueError("recognition receiver inventory entries are malformed")
        receiver_t, component_ids = item
        if type(receiver_t) is not int or receiver_t < 0:
            raise ValueError(
                "recognition receiver indices must be nonnegative integers"
            )
        if (
            type(component_ids) is not tuple
            or not component_ids
            or len(set(component_ids)) != len(component_ids)
        ):
            raise ValueError(
                "recognition component inventory must be nonempty and unique"
            )
        receivers.append(receiver_t)
        for component_id in component_ids:
            _require_nonempty(component_id, "recognition component ID")
            if "." in component_id:
                raise ValueError("recognition component IDs cannot contain periods")
            names.append(f"receiver.{receiver_t}.component.{component_id}.mean")
        names.append(f"receiver.{receiver_t}.shared_precision_cholesky")
    if tuple(receivers) != tuple(range(receiver_count)):
        raise ValueError(
            "recognition continuous receiver inventory must cover exactly 0..T"
        )
    for bank, enabled, supports in (
        ("state", state_categorical_enabled, state_categorical_supports),
        ("model", model_categorical_enabled, model_categorical_supports),
    ):
        if type(enabled) is not bool:
            raise ValueError("recognition categorical topology must use exact bools")
        if enabled:
            if type(supports) is not tuple or len(supports) != receiver_count - 1:
                raise ValueError(
                    f"{bank} categorical bank must cover every receiver 1..T"
                )
            for receiver_t, support in enumerate(supports, start=1):
                if (
                    type(support) is not tuple
                    or not support
                    or any(type(source_j) is not int for source_j in support)
                    or tuple(sorted(set(support))) != support
                    or any(
                        source_j < 0 or source_j >= receiver_t for source_j in support
                    )
                ):
                    raise ValueError(
                        f"{bank} categorical bank has malformed ragged support"
                    )
                names.extend(
                    (
                        f"{bank}.receiver.{receiver_t}.support",
                        f"{bank}.receiver.{receiver_t}.categorical_row",
                    )
                )
        elif supports != _ABSENT_CATEGORICAL_SUPPORT:
            raise ValueError(
                f"{bank} absent categorical bank must use the explicit singleton"
            )
        else:
            names.extend(
                (
                    f"{bank}.absent.support",
                    f"{bank}.absent.categorical_row",
                )
            )
    return tuple(sorted(names))


def _validate_recognition_tensors(
    *,
    receiver_components: tuple[tuple[int, tuple[str, ...]], ...],
    state_categorical_enabled: bool,
    state_categorical_supports: tuple[tuple[int, ...], ...] | tuple[None],
    model_categorical_enabled: bool,
    model_categorical_supports: tuple[tuple[int, ...], ...] | tuple[None],
    tensors: Mapping[str, Tensor],
) -> None:
    for receiver_t, component_ids in receiver_components:
        means = tuple(
            tensors[f"receiver.{receiver_t}.component.{component_id}.mean"]
            for component_id in component_ids
        )
        if any(
            not isinstance(mean, Tensor)
            or mean.ndim < 1
            or not mean.is_floating_point()
            or not bool(torch.isfinite(mean.detach()).all())
            for mean in means
        ):
            raise ValueError("recognition continuous component means must be finite")
        dimensions = {mean.shape[-1] for mean in means}
        if len(dimensions) != 1:
            raise ValueError("recognition component mean dimensions disagree")
        dimension = next(iter(dimensions))
        precision = tensors[f"receiver.{receiver_t}.shared_precision_cholesky"]
        if (
            not isinstance(precision, Tensor)
            or not precision.is_floating_point()
            or precision.shape[-2:] != (dimension, dimension)
            or not bool(torch.isfinite(precision.detach()).all())
            or not bool(
                (
                    torch.diagonal(
                        precision.detach(),
                        dim1=-2,
                        dim2=-1,
                    )
                    > 0
                ).all()
            )
        ):
            raise ValueError("recognition shared precision Cholesky is malformed")
    for bank, enabled, supports in (
        ("state", state_categorical_enabled, state_categorical_supports),
        ("model", model_categorical_enabled, model_categorical_supports),
    ):
        if enabled:
            for receiver_t, expected_support in enumerate(supports, start=1):
                assert isinstance(expected_support, tuple)
                support = tensors[f"{bank}.receiver.{receiver_t}.support"]
                row = tensors[f"{bank}.receiver.{receiver_t}.categorical_row"]
                if (
                    not isinstance(support, Tensor)
                    or support.dtype not in (torch.int32, torch.int64)
                    or support.ndim != 1
                    or tuple(int(value) for value in support.detach().cpu().tolist())
                    != expected_support
                    or not isinstance(row, Tensor)
                    or not row.is_floating_point()
                    or row.shape != support.shape
                    or not bool(torch.isfinite(row.detach()).all())
                    or not bool((row.detach() >= 0).all())
                    or not bool(
                        torch.allclose(
                            row.detach().sum(),
                            row.detach().new_ones(()),
                            rtol=1e-12,
                            atol=1e-12,
                        )
                    )
                ):
                    raise ValueError(
                        f"{bank} categorical support/row inventory is malformed"
                    )
        else:
            support = tensors[f"{bank}.absent.support"]
            row = tensors[f"{bank}.absent.categorical_row"]
            if (
                not isinstance(support, Tensor)
                or support.dtype not in (torch.int32, torch.int64)
                or support.shape != (1,)
                or int(support.detach().cpu().item()) != -1
                or not isinstance(row, Tensor)
                or not row.is_floating_point()
                or row.shape != (1,)
                or not bool(row.detach()[0] == 1.0)
            ):
                raise ValueError(
                    f"{bank} absent categorical bank must be an explicit singleton"
                )


@dataclass(frozen=True, slots=True)
class H6LiveRecognitionStateV3(Mapping[str, Tensor]):
    """Complete live recognition law returned by a recognition forward."""

    endpoint_config_sha256: str
    receiver_count: int
    state_categorical_enabled: bool
    model_categorical_enabled: bool
    state_categorical_supports: tuple[tuple[int, ...], ...] | tuple[None]
    model_categorical_supports: tuple[tuple[int, ...], ...] | tuple[None]
    receiver_components: tuple[tuple[int, tuple[str, ...]], ...]
    names: tuple[str, ...]
    context_sha256: str
    recognition_state_sha256: str
    source_model_sha256: str
    law_sha256: str
    live_state_sha256: str
    _tensors: tuple[Tensor, ...]

    def __iter__(self):
        return iter(self.names)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, name: str) -> Tensor:
        try:
            return self._tensors[self.names.index(name)]
        except ValueError as exc:
            raise KeyError(name) from exc

    def canonical_payload(self) -> dict[str, object]:
        return {
            "endpoint_config_sha256": self.endpoint_config_sha256,
            "receiver_count": self.receiver_count,
            "state_categorical_enabled": self.state_categorical_enabled,
            "model_categorical_enabled": self.model_categorical_enabled,
            "state_categorical_supports": self.state_categorical_supports,
            "model_categorical_supports": self.model_categorical_supports,
            "receiver_components": self.receiver_components,
            "context_sha256": self.context_sha256,
            "recognition_state_sha256": self.recognition_state_sha256,
            "source_model_sha256": self.source_model_sha256,
            "law_sha256": self.law_sha256,
            "tensors": tuple(
                {"name": name, **_tensor_manifest(tensor)}
                for name, tensor in zip(self.names, self._tensors, strict=True)
            ),
        }

    def __post_init__(self) -> None:
        _require_sha256(self.endpoint_config_sha256, "endpoint_config_sha256")
        expected_names = _recognition_inventory_names(
            receiver_count=self.receiver_count,
            receiver_components=self.receiver_components,
            state_categorical_enabled=self.state_categorical_enabled,
            state_categorical_supports=self.state_categorical_supports,
            model_categorical_enabled=self.model_categorical_enabled,
            model_categorical_supports=self.model_categorical_supports,
        )
        if self.names != expected_names or len(self._tensors) != len(self.names):
            raise ValueError(
                "recognition tensor inventory is missing a continuous state, "
                "precision, categorical row, or ragged support"
            )
        for name in (
            "context_sha256",
            "recognition_state_sha256",
            "source_model_sha256",
            "law_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        _validate_recognition_tensors(
            receiver_components=self.receiver_components,
            state_categorical_enabled=self.state_categorical_enabled,
            state_categorical_supports=self.state_categorical_supports,
            model_categorical_enabled=self.model_categorical_enabled,
            model_categorical_supports=self.model_categorical_supports,
            tensors=dict(zip(self.names, self._tensors, strict=True)),
        )
        if self.live_state_sha256 != _owned_hash(
            "vfe4.h6.live-recognition-state.v3", self.canonical_payload()
        ):
            raise ValueError("live recognition-state identity is stale")

    @classmethod
    def create(
        cls,
        *,
        endpoint_config_sha256: str,
        receiver_count: int,
        state_categorical_enabled: bool,
        model_categorical_enabled: bool,
        state_categorical_supports: tuple[tuple[int, ...], ...] | tuple[None],
        model_categorical_supports: tuple[tuple[int, ...], ...] | tuple[None],
        receiver_components: tuple[tuple[int, tuple[str, ...]], ...],
        tensors: Mapping[str, Tensor],
        context_sha256: str,
        recognition_state_sha256: str,
        source_model_sha256: str,
        law_sha256: str,
    ) -> H6LiveRecognitionStateV3:
        expected_names = _recognition_inventory_names(
            receiver_count=receiver_count,
            receiver_components=receiver_components,
            state_categorical_enabled=state_categorical_enabled,
            state_categorical_supports=state_categorical_supports,
            model_categorical_enabled=model_categorical_enabled,
            model_categorical_supports=model_categorical_supports,
        )
        if not isinstance(tensors, Mapping) or tuple(sorted(tensors)) != expected_names:
            raise ValueError(
                "recognition tensor inventory is missing a continuous state, "
                "precision, categorical row, or ragged support"
            )
        owned = tuple(tensors[name] for name in expected_names)
        values = {
            "endpoint_config_sha256": endpoint_config_sha256,
            "receiver_count": receiver_count,
            "state_categorical_enabled": state_categorical_enabled,
            "model_categorical_enabled": model_categorical_enabled,
            "state_categorical_supports": state_categorical_supports,
            "model_categorical_supports": model_categorical_supports,
            "receiver_components": receiver_components,
            "names": expected_names,
            "context_sha256": context_sha256,
            "recognition_state_sha256": recognition_state_sha256,
            "source_model_sha256": source_model_sha256,
            "law_sha256": law_sha256,
            "_tensors": owned,
        }
        provisional = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(provisional, name, value)
        return cls(
            **values,
            live_state_sha256=_owned_hash(
                "vfe4.h6.live-recognition-state.v3",
                provisional.canonical_payload(),
            ),
        )


def _example_qualified_names(
    states: tuple[H6LiveRecognitionStateV3, ...]
    | tuple["H6DetachedRecognitionSnapshotV3", ...],
) -> tuple[str, ...]:
    return tuple(
        f"example.{example_ordinal}.{name}"
        for example_ordinal, state in enumerate(states)
        for name in state.names
    )


def _validate_active_receiver_inventory(
    *,
    receiver_count: int,
    state_count: int,
    active_target_counts: tuple[int, ...],
    active_receiver_masks: tuple[tuple[bool, ...], ...],
) -> None:
    if (
        type(active_target_counts) is not tuple
        or len(active_target_counts) != state_count
        or type(active_receiver_masks) is not tuple
        or len(active_receiver_masks) != state_count
    ):
        raise ValueError(
            "batch recognition active-target inventory must align with examples"
        )
    for target_count, mask in zip(
        active_target_counts,
        active_receiver_masks,
        strict=True,
    ):
        if (
            type(target_count) is not int
            or not 1 <= target_count < receiver_count
            or type(mask) is not tuple
            or len(mask) != receiver_count
            or any(type(active) is not bool for active in mask)
            or mask
            != (True,) * (target_count + 1)
            + (False,) * (receiver_count - target_count - 1)
        ):
            raise ValueError(
                "each batch example requires its exact active receiver prefix"
            )


@dataclass(frozen=True, slots=True)
class H6BatchLiveRecognitionStateV3(Mapping[str, Tensor]):
    """A complete example-qualified live recognition law for one update."""

    authority_sha256: str
    endpoint_config_sha256: str
    receiver_count: int
    active_target_counts: tuple[int, ...]
    active_receiver_masks: tuple[tuple[bool, ...], ...]
    states: tuple[H6LiveRecognitionStateV3, ...]
    names: tuple[str, ...]
    batch_live_state_sha256: str

    @property
    def example_ordinals(self) -> tuple[int, ...]:
        return tuple(range(len(self.states)))

    def __iter__(self):
        return iter(self.names)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, name: str) -> Tensor:
        _require_nonempty(name, "batch recognition tensor name")
        parts = name.split(".", 2)
        if len(parts) != 3 or parts[0] != "example":
            raise KeyError(name)
        try:
            example_ordinal = int(parts[1])
        except ValueError as exc:
            raise KeyError(name) from exc
        if (
            not 0 <= example_ordinal < len(self.states)
            or name not in self.names
        ):
            raise KeyError(name)
        return self.states[example_ordinal][parts[2]]

    def canonical_payload(self) -> dict[str, object]:
        return {
            "authority_sha256": self.authority_sha256,
            "endpoint_config_sha256": self.endpoint_config_sha256,
            "receiver_count": self.receiver_count,
            "example_ordinals": self.example_ordinals,
            "active_target_counts": self.active_target_counts,
            "active_receiver_masks": self.active_receiver_masks,
            "example_live_state_sha256s": tuple(
                state.live_state_sha256 for state in self.states
            ),
            "tensors": tuple(
                {"name": name, **_tensor_manifest(self[name])}
                for name in self.names
            ),
        }

    def __post_init__(self) -> None:
        for name in ("authority_sha256", "endpoint_config_sha256"):
            _require_sha256(getattr(self, name), name)
        if (
            type(self.receiver_count) is not int
            or self.receiver_count < 2
            or type(self.states) is not tuple
            or not self.states
            or len(self.states) > 8
            or any(
                type(state) is not H6LiveRecognitionStateV3
                for state in self.states
            )
        ):
            raise ValueError(
                "batch recognition requires one to eight exact live states"
            )
        for state in self.states:
            state.__post_init__()
        first = self.states[0]
        topology = (
            first.endpoint_config_sha256,
            first.state_categorical_enabled,
            first.model_categorical_enabled,
            first.source_model_sha256,
        )
        if (
            self.endpoint_config_sha256 != first.endpoint_config_sha256
            or any(
                (
                    state.endpoint_config_sha256,
                    state.state_categorical_enabled,
                    state.model_categorical_enabled,
                    state.source_model_sha256,
                )
                != topology
                for state in self.states[1:]
            )
        ):
            raise ValueError(
                "batch recognition examples do not share one endpoint topology"
            )
        _validate_active_receiver_inventory(
            receiver_count=self.receiver_count,
            state_count=len(self.states),
            active_target_counts=self.active_target_counts,
            active_receiver_masks=self.active_receiver_masks,
        )
        if any(
            state.receiver_count != active_target_count + 1
            for state, active_target_count in zip(
                self.states,
                self.active_target_counts,
                strict=True,
            )
        ):
            raise ValueError(
                "each ragged recognition state must end at its active target"
            )
        for bank in ("state", "model"):
            enabled = getattr(first, f"{bank}_categorical_enabled")
            if not enabled:
                continue
            supports_by_state = tuple(
                getattr(state, f"{bank}_categorical_supports")
                for state in self.states
            )
            for receiver_t in range(1, self.receiver_count):
                visible = tuple(
                    supports[receiver_t - 1]
                    for supports in supports_by_state
                    if receiver_t <= len(supports)
                )
                if visible and any(
                    support != visible[0] for support in visible[1:]
                ):
                    raise ValueError(
                        "ragged categorical supports are not one prefix law"
                    )
        expected_names = _example_qualified_names(self.states)
        if self.names != expected_names or len(set(self.names)) != len(
            self.names
        ):
            raise ValueError(
                "batch recognition tensor inventory is incomplete or duplicated"
            )
        if self.batch_live_state_sha256 != _owned_hash(
            "vfe4.h6.batch-live-recognition-state.v3",
            self.canonical_payload(),
        ):
            raise ValueError("batch live recognition-state identity is stale")

    @classmethod
    def create(
        cls,
        *,
        authority: "H6EngineAuthorityV3",
        states: tuple[H6LiveRecognitionStateV3, ...],
        active_target_counts: tuple[int, ...],
        active_receiver_masks: tuple[tuple[bool, ...], ...],
    ) -> "H6BatchLiveRecognitionStateV3":
        if type(authority) is not H6EngineAuthorityV3:
            raise ValueError(
                "batch recognition requires an exact engine authority"
            )
        authority.__post_init__()
        if (
            type(states) is not tuple
            or type(active_target_counts) is not tuple
            or len(states) != len(active_target_counts)
        ):
            raise ValueError(
                "batch recognition state/count inventories must align"
            )
        if (
            any(
                type(state) is not H6LiveRecognitionStateV3
                for state in states
            )
            or any(
                state.endpoint_config_sha256
                != authority.endpoint_config_sha256
                or state.receiver_count
                != active_target_count + 1
                or state.state_categorical_enabled
                != authority.state_categorical_enabled
                or state.model_categorical_enabled
                != authority.model_categorical_enabled
                for state, active_target_count in zip(
                    states,
                    active_target_counts,
                    strict=True,
                )
            )
        ):
            raise ValueError(
                "batch recognition state is outside engine authority"
            )
        values = {
            "authority_sha256": authority.authority_sha256,
            "endpoint_config_sha256": authority.endpoint_config_sha256,
            "receiver_count": authority.receiver_count,
            "active_target_counts": tuple(active_target_counts),
            "active_receiver_masks": tuple(active_receiver_masks),
            "states": tuple(states),
            "names": _example_qualified_names(tuple(states)),
        }
        provisional = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(provisional, name, value)
        return cls(
            **values,
            batch_live_state_sha256=_owned_hash(
                "vfe4.h6.batch-live-recognition-state.v3",
                provisional.canonical_payload(),
            ),
        )


@dataclass(frozen=True, slots=True, eq=False)
class H6DetachedRecognitionSnapshotV3:
    """Complete detached recognition law bound to the exact model phase."""

    attempt_spec_sha256: str
    endpoint_config_sha256: str
    post_recognition_cursor_sha256: str
    pass_index: int
    batch_index: int
    recognition_update_count: int
    receiver_count: int
    state_categorical_enabled: bool
    model_categorical_enabled: bool
    state_categorical_supports: tuple[tuple[int, ...], ...] | tuple[None]
    model_categorical_supports: tuple[tuple[int, ...], ...] | tuple[None]
    receiver_components: tuple[tuple[int, tuple[str, ...]], ...]
    names: tuple[str, ...]
    context_sha256: str
    recognition_state_sha256: str
    source_model_sha256: str
    law_sha256: str
    live_state_sha256: str
    snapshot_sha256: str
    _tensors: tuple[Tensor, ...]

    def canonical_payload(self) -> dict[str, object]:
        return {
            "attempt_spec_sha256": self.attempt_spec_sha256,
            "endpoint_config_sha256": self.endpoint_config_sha256,
            "post_recognition_cursor_sha256": (self.post_recognition_cursor_sha256),
            "pass_index": self.pass_index,
            "batch_index": self.batch_index,
            "recognition_update_count": self.recognition_update_count,
            "receiver_count": self.receiver_count,
            "state_categorical_enabled": self.state_categorical_enabled,
            "model_categorical_enabled": self.model_categorical_enabled,
            "state_categorical_supports": self.state_categorical_supports,
            "model_categorical_supports": self.model_categorical_supports,
            "receiver_components": self.receiver_components,
            "context_sha256": self.context_sha256,
            "recognition_state_sha256": self.recognition_state_sha256,
            "source_model_sha256": self.source_model_sha256,
            "law_sha256": self.law_sha256,
            "live_state_sha256": self.live_state_sha256,
            "tensors": tuple(
                {"name": name, **_tensor_manifest(tensor)}
                for name, tensor in zip(self.names, self._tensors, strict=True)
            ),
        }

    def __post_init__(self) -> None:
        for name in (
            "attempt_spec_sha256",
            "endpoint_config_sha256",
            "post_recognition_cursor_sha256",
            "context_sha256",
            "recognition_state_sha256",
            "source_model_sha256",
            "law_sha256",
            "live_state_sha256",
            "snapshot_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if (
            type(self.pass_index) is not int
            or self.pass_index < 0
            or type(self.batch_index) is not int
            or self.batch_index < 0
            or type(self.recognition_update_count) is not int
            or self.recognition_update_count <= 0
        ):
            raise ValueError("snapshot cursor coordinates are invalid")
        expected_names = _recognition_inventory_names(
            receiver_count=self.receiver_count,
            receiver_components=self.receiver_components,
            state_categorical_enabled=self.state_categorical_enabled,
            state_categorical_supports=self.state_categorical_supports,
            model_categorical_enabled=self.model_categorical_enabled,
            model_categorical_supports=self.model_categorical_supports,
        )
        if self.names != expected_names:
            raise ValueError("snapshot receiver/component inventory is incomplete")
        if len(self._tensors) != len(self.names):
            raise ValueError("snapshot tensor inventory does not match its names")
        storage_ids: set[tuple[str, int | None, int]] = set()
        for tensor in self._tensors:
            if (
                not isinstance(tensor, Tensor)
                or tensor.requires_grad
                or tensor.grad_fn is not None
                or not tensor.is_contiguous()
            ):
                raise ValueError("snapshot tensors must be owned and detached")
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                raise ValueError("snapshot tensors must be finite")
            storage = tensor.untyped_storage()
            storage_id = (
                tensor.device.type,
                tensor.device.index,
                int(getattr(storage, "_cdata", storage.data_ptr())),
            )
            if storage_id in storage_ids:
                raise ValueError("snapshot tensors cannot alias storage")
            storage_ids.add(storage_id)
        _validate_recognition_tensors(
            receiver_components=self.receiver_components,
            state_categorical_enabled=self.state_categorical_enabled,
            state_categorical_supports=self.state_categorical_supports,
            model_categorical_enabled=self.model_categorical_enabled,
            model_categorical_supports=self.model_categorical_supports,
            tensors=dict(zip(self.names, self._tensors, strict=True)),
        )
        if self.snapshot_sha256 != _owned_hash(
            "vfe4.h6.detached-recognition-snapshot.v3",
            self.canonical_payload(),
        ):
            raise ValueError("detached recognition snapshot identity is stale")

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is H6DetachedRecognitionSnapshotV3
            and self.snapshot_sha256 == other.snapshot_sha256
        )

    @classmethod
    def capture(
        cls,
        state: H6LiveRecognitionStateV3,
        *,
        authority: H6EngineAuthorityV3,
        post_recognition_cursor: H6AttemptCursorV3,
        allow_receiver_prefix: bool = False,
    ) -> H6DetachedRecognitionSnapshotV3:
        if type(state) is not H6LiveRecognitionStateV3:
            raise ValueError(
                "snapshot capture requires a complete live recognition state"
            )
        state.__post_init__()
        if type(authority) is not H6EngineAuthorityV3:
            raise ValueError("snapshot requires an exact H6 engine authority")
        authority.__post_init__()
        if (
            state.endpoint_config_sha256 != authority.endpoint_config_sha256
            or (
                state.receiver_count != authority.receiver_count
                if not allow_receiver_prefix
                else not 2 <= state.receiver_count <= authority.receiver_count
            )
            or state.state_categorical_enabled != authority.state_categorical_enabled
            or state.model_categorical_enabled != authority.model_categorical_enabled
        ):
            raise ValueError(
                "recognition snapshot inventory does not match endpoint topology"
            )
        if type(post_recognition_cursor) is not H6AttemptCursorV3:
            raise ValueError("snapshot requires an exact post-recognition cursor")
        post_recognition_cursor.__post_init__()
        if (
            post_recognition_cursor.attempt_spec_sha256 != authority.attempt_spec_sha256
            or post_recognition_cursor.next_phase is not TrainingPhase.MODEL_ADAMW
            or post_recognition_cursor.recognition_update_count
            != post_recognition_cursor.model_update_count + 1
        ):
            raise ValueError(
                "snapshot is not bound to the exact post-recognition model phase"
            )
        tensors = tuple(
            state[name].detach().clone(memory_format=torch.contiguous_format)
            for name in state.names
        )
        values = {
            "attempt_spec_sha256": authority.attempt_spec_sha256,
            "endpoint_config_sha256": authority.endpoint_config_sha256,
            "post_recognition_cursor_sha256": (post_recognition_cursor.cursor_sha256),
            "pass_index": post_recognition_cursor.pass_index,
            "batch_index": post_recognition_cursor.batch_index,
            "recognition_update_count": (
                post_recognition_cursor.recognition_update_count
            ),
            "receiver_count": state.receiver_count,
            "state_categorical_enabled": state.state_categorical_enabled,
            "model_categorical_enabled": state.model_categorical_enabled,
            "state_categorical_supports": state.state_categorical_supports,
            "model_categorical_supports": state.model_categorical_supports,
            "receiver_components": state.receiver_components,
            "names": state.names,
            "context_sha256": state.context_sha256,
            "recognition_state_sha256": state.recognition_state_sha256,
            "source_model_sha256": state.source_model_sha256,
            "law_sha256": state.law_sha256,
            "live_state_sha256": state.live_state_sha256,
            "_tensors": tensors,
        }
        provisional = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(provisional, name, value)
        return cls(
            **values,
            snapshot_sha256=_owned_hash(
                "vfe4.h6.detached-recognition-snapshot.v3",
                provisional.canonical_payload(),
            ),
        )

    def tensor(self, name: str) -> Tensor:
        """Return an owned clone so callers cannot mutate the snapshot."""

        _require_nonempty(name, "snapshot tensor name")
        try:
            index = self.names.index(name)
        except ValueError as exc:
            raise KeyError(name) from exc
        self.__post_init__()
        return self._tensors[index].clone(memory_format=torch.contiguous_format)


@dataclass(frozen=True, slots=True, eq=False)
class H6DetachedBatchRecognitionSnapshotV3(Mapping[str, Tensor]):
    """Every example law detached at one exact post-recognition boundary."""

    authority_sha256: str
    attempt_spec_sha256: str
    endpoint_config_sha256: str
    post_recognition_cursor_sha256: str
    pass_index: int
    batch_index: int
    recognition_update_count: int
    receiver_count: int
    active_target_counts: tuple[int, ...]
    active_receiver_masks: tuple[tuple[bool, ...], ...]
    live_batch_state_sha256: str
    states: tuple[H6DetachedRecognitionSnapshotV3, ...]
    names: tuple[str, ...]
    snapshot_sha256: str

    @property
    def example_ordinals(self) -> tuple[int, ...]:
        return tuple(range(len(self.states)))

    def __iter__(self):
        return iter(self.names)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, name: str) -> Tensor:
        _require_nonempty(name, "batch snapshot tensor name")
        parts = name.split(".", 2)
        if len(parts) != 3 or parts[0] != "example":
            raise KeyError(name)
        try:
            example_ordinal = int(parts[1])
        except ValueError as exc:
            raise KeyError(name) from exc
        if (
            not 0 <= example_ordinal < len(self.states)
            or name not in self.names
        ):
            raise KeyError(name)
        return self.states[example_ordinal].tensor(parts[2])

    def canonical_payload(self) -> dict[str, object]:
        return {
            "authority_sha256": self.authority_sha256,
            "attempt_spec_sha256": self.attempt_spec_sha256,
            "endpoint_config_sha256": self.endpoint_config_sha256,
            "post_recognition_cursor_sha256": (
                self.post_recognition_cursor_sha256
            ),
            "pass_index": self.pass_index,
            "batch_index": self.batch_index,
            "recognition_update_count": self.recognition_update_count,
            "receiver_count": self.receiver_count,
            "example_ordinals": self.example_ordinals,
            "active_target_counts": self.active_target_counts,
            "active_receiver_masks": self.active_receiver_masks,
            "live_batch_state_sha256": self.live_batch_state_sha256,
            "example_snapshot_sha256s": tuple(
                state.snapshot_sha256 for state in self.states
            ),
            "tensors": tuple(
                {"name": name, **_tensor_manifest(self[name])}
                for name in self.names
            ),
        }

    def __post_init__(self) -> None:
        for name in (
            "authority_sha256",
            "attempt_spec_sha256",
            "endpoint_config_sha256",
            "post_recognition_cursor_sha256",
            "live_batch_state_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if (
            type(self.pass_index) is not int
            or self.pass_index < 0
            or type(self.batch_index) is not int
            or self.batch_index < 0
            or type(self.recognition_update_count) is not int
            or self.recognition_update_count <= 0
            or type(self.receiver_count) is not int
            or self.receiver_count < 2
            or type(self.states) is not tuple
            or not self.states
            or len(self.states) > 8
            or any(
                type(state) is not H6DetachedRecognitionSnapshotV3
                for state in self.states
            )
        ):
            raise ValueError("detached batch recognition metadata is invalid")
        for state, active_target_count in zip(
            self.states,
            self.active_target_counts,
            strict=True,
        ):
            state.__post_init__()
            if (
                state.attempt_spec_sha256 != self.attempt_spec_sha256
                or state.endpoint_config_sha256
                != self.endpoint_config_sha256
                or state.post_recognition_cursor_sha256
                != self.post_recognition_cursor_sha256
                or state.pass_index != self.pass_index
                or state.batch_index != self.batch_index
                or state.recognition_update_count
                != self.recognition_update_count
                or state.receiver_count != active_target_count + 1
            ):
                raise ValueError(
                    "detached example snapshot left the batch boundary"
                )
        _validate_active_receiver_inventory(
            receiver_count=self.receiver_count,
            state_count=len(self.states),
            active_target_counts=self.active_target_counts,
            active_receiver_masks=self.active_receiver_masks,
        )
        expected_names = _example_qualified_names(self.states)
        if self.names != expected_names or len(set(self.names)) != len(
            self.names
        ):
            raise ValueError(
                "detached batch tensor inventory is incomplete or duplicated"
            )
        storage_ids: set[tuple[str, int | None, int]] = set()
        for state in self.states:
            for tensor in state._tensors:
                if tensor.requires_grad or tensor.grad_fn is not None:
                    raise ValueError(
                        "detached batch snapshot retained a recognition graph"
                    )
                storage = tensor.untyped_storage()
                identity = (
                    tensor.device.type,
                    tensor.device.index,
                    int(getattr(storage, "_cdata", storage.data_ptr())),
                )
                if identity in storage_ids:
                    raise ValueError(
                        "detached batch snapshot tensors cannot alias storage"
                    )
                storage_ids.add(identity)
        if self.snapshot_sha256 != _owned_hash(
            "vfe4.h6.detached-batch-recognition-snapshot.v3",
            self.canonical_payload(),
        ):
            raise ValueError(
                "detached batch recognition snapshot identity is stale"
            )

    @classmethod
    def capture(
        cls,
        state: H6BatchLiveRecognitionStateV3,
        *,
        authority: "H6EngineAuthorityV3",
        post_recognition_cursor: H6AttemptCursorV3,
    ) -> "H6DetachedBatchRecognitionSnapshotV3":
        if type(state) is not H6BatchLiveRecognitionStateV3:
            raise ValueError(
                "batch snapshot capture requires an exact live batch law"
            )
        state.__post_init__()
        if type(authority) is not H6EngineAuthorityV3:
            raise ValueError(
                "batch snapshot requires an exact engine authority"
            )
        authority.__post_init__()
        if (
            state.authority_sha256 != authority.authority_sha256
            or type(post_recognition_cursor) is not H6AttemptCursorV3
        ):
            raise ValueError(
                "batch snapshot state/authority identity drift"
            )
        post_recognition_cursor.__post_init__()
        snapshots = tuple(
            H6DetachedRecognitionSnapshotV3.capture(
                example_state,
                authority=authority,
                post_recognition_cursor=post_recognition_cursor,
                allow_receiver_prefix=True,
            )
            for example_state in state.states
        )
        values = {
            "authority_sha256": authority.authority_sha256,
            "attempt_spec_sha256": authority.attempt_spec_sha256,
            "endpoint_config_sha256": authority.endpoint_config_sha256,
            "post_recognition_cursor_sha256": (
                post_recognition_cursor.cursor_sha256
            ),
            "pass_index": post_recognition_cursor.pass_index,
            "batch_index": post_recognition_cursor.batch_index,
            "recognition_update_count": (
                post_recognition_cursor.recognition_update_count
            ),
            "receiver_count": authority.receiver_count,
            "active_target_counts": state.active_target_counts,
            "active_receiver_masks": state.active_receiver_masks,
            "live_batch_state_sha256": state.batch_live_state_sha256,
            "states": snapshots,
            "names": _example_qualified_names(snapshots),
        }
        provisional = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(provisional, name, value)
        return cls(
            **values,
            snapshot_sha256=_owned_hash(
                "vfe4.h6.detached-batch-recognition-snapshot.v3",
                provisional.canonical_payload(),
            ),
        )


@dataclass(frozen=True, slots=True)
class H6EngineAuthorityV3:
    """Immutable identities authorizing one H6 training attempt."""

    attempt_spec_sha256: str
    endpoint_config_sha256: str
    readiness_sha256: str
    readiness_matching_set_sha256: str
    matching_set_sha256: str
    matching_policy_sha256: str
    readiness_training_schedule_sha256: str
    training_schedule_sha256: str
    readiness_runtime_identity_sha256: str
    runtime_identity_sha256: str
    planned_attempt_sha256: str
    endpoint_config_id: str
    matching_ledger_sha256: str
    matching_report_sha256s: tuple[str, ...]
    receiver_count: int
    state_categorical_enabled: bool
    model_categorical_enabled: bool
    tuning_cell_sha256: str
    optimizer_policy_sha256: str
    optimizer_learning_rate: float
    optimizer_betas: tuple[float, float]
    optimizer_eps: float
    optimizer_weight_decay: float
    optimizer_amsgrad: bool
    optimizer_maximize: bool
    optimizer_foreach: bool
    optimizer_capturable: bool
    optimizer_differentiable: bool
    optimizer_fused: bool
    gradient_clip_max_norm: float
    objective_kind: Literal[
        "cross_entropy",
        "complete_elbo",
        "emission_only_ablation_non_elbo",
    ]
    latent_enabled: bool
    authority_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name) for name in tuple(self.__dataclass_fields__)[:-1]
        }

    def __post_init__(self) -> None:
        for name in (
            "attempt_spec_sha256",
            "endpoint_config_sha256",
            "readiness_sha256",
            "readiness_matching_set_sha256",
            "matching_set_sha256",
            "matching_policy_sha256",
            "readiness_training_schedule_sha256",
            "training_schedule_sha256",
            "readiness_runtime_identity_sha256",
            "runtime_identity_sha256",
            "planned_attempt_sha256",
            "matching_ledger_sha256",
            "tuning_cell_sha256",
            "optimizer_policy_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        _require_nonempty(self.endpoint_config_id, "endpoint_config_id")
        if (
            type(self.matching_report_sha256s) is not tuple
            or not self.matching_report_sha256s
            or len(set(self.matching_report_sha256s))
            != len(self.matching_report_sha256s)
        ):
            raise ValueError(
                "engine authority requires a unique matching-report inventory"
            )
        for digest in self.matching_report_sha256s:
            _require_sha256(digest, "matching report SHA-256")
        if type(self.receiver_count) is not int or self.receiver_count < 2:
            raise ValueError("engine authority receiver count must represent 0..T")
        if (
            type(self.state_categorical_enabled) is not bool
            or type(self.model_categorical_enabled) is not bool
        ):
            raise ValueError("engine authority categorical topology is malformed")
        for name in (
            "optimizer_learning_rate",
            "optimizer_eps",
            "optimizer_weight_decay",
            "gradient_clip_max_norm",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite nonnegative float")
        if self.optimizer_learning_rate <= 0.0 or self.gradient_clip_max_norm <= 0.0:
            raise ValueError("AdamW learning rate and gradient clip must be positive")
        policy_fields = {
            "optimizer_policy_sha256": H6_ADAMW_POLICY.optimizer_policy_sha256,
            "optimizer_betas": H6_ADAMW_POLICY.betas,
            "optimizer_eps": H6_ADAMW_POLICY.eps,
            "optimizer_amsgrad": H6_ADAMW_POLICY.amsgrad,
            "optimizer_maximize": H6_ADAMW_POLICY.maximize,
            "optimizer_foreach": H6_ADAMW_POLICY.foreach,
            "optimizer_capturable": H6_ADAMW_POLICY.capturable,
            "optimizer_differentiable": H6_ADAMW_POLICY.differentiable,
            "optimizer_fused": H6_ADAMW_POLICY.fused,
            "gradient_clip_max_norm": H6_ADAMW_POLICY.gradient_clip_max_norm,
        }
        if any(getattr(self, name) != value for name, value in policy_fields.items()):
            raise ValueError(
                "engine authority does not bind the complete H6 AdamW policy"
            )
        if self.readiness_matching_set_sha256 != self.matching_set_sha256:
            raise ValueError("readiness and matching-set identity drift")
        if self.readiness_training_schedule_sha256 != self.training_schedule_sha256:
            raise ValueError("readiness and training-schedule identity drift")
        if self.readiness_runtime_identity_sha256 != self.runtime_identity_sha256:
            raise ValueError("readiness and runtime identity drift")
        if type(self.latent_enabled) is not bool:
            raise ValueError("latent_enabled must be an exact bool")
        if self.objective_kind not in (
            "cross_entropy",
            "complete_elbo",
            "emission_only_ablation_non_elbo",
        ):
            raise ValueError("unsupported H6 objective kind")
        if self.latent_enabled == (self.objective_kind == "cross_entropy"):
            raise ValueError("objective kind and latent endpoint disagree")
        if not self.latent_enabled and (
            self.state_categorical_enabled or self.model_categorical_enabled
        ):
            raise ValueError("no-latent endpoint cannot bind categorical banks")
        if self.authority_sha256 != _owned_hash(
            "vfe4.h6.engine-authority.v3", self.canonical_payload()
        ):
            raise ValueError("H6 engine authority identity is stale")

    @classmethod
    def create(
        cls,
        *,
        attempt_spec_sha256: str,
        endpoint_config_sha256: str,
        readiness_sha256: str,
        readiness_matching_set_sha256: str,
        matching_set_sha256: str,
        matching_policy_sha256: str,
        readiness_training_schedule_sha256: str,
        training_schedule_sha256: str,
        readiness_runtime_identity_sha256: str,
        runtime_identity_sha256: str,
        planned_attempt_sha256: str,
        endpoint_config_id: str,
        matching_ledger_sha256: str,
        matching_report_sha256s: tuple[str, ...],
        receiver_count: int,
        state_categorical_enabled: bool,
        model_categorical_enabled: bool,
        tuning_cell_sha256: str,
        optimizer_policy_sha256: str,
        optimizer_learning_rate: float,
        optimizer_weight_decay: float,
        objective_kind: Literal[
            "cross_entropy",
            "complete_elbo",
            "emission_only_ablation_non_elbo",
        ],
        latent_enabled: bool,
    ) -> H6EngineAuthorityV3:
        values = {
            "attempt_spec_sha256": attempt_spec_sha256,
            "endpoint_config_sha256": endpoint_config_sha256,
            "readiness_sha256": readiness_sha256,
            "readiness_matching_set_sha256": readiness_matching_set_sha256,
            "matching_set_sha256": matching_set_sha256,
            "matching_policy_sha256": matching_policy_sha256,
            "readiness_training_schedule_sha256": (readiness_training_schedule_sha256),
            "training_schedule_sha256": training_schedule_sha256,
            "readiness_runtime_identity_sha256": (readiness_runtime_identity_sha256),
            "runtime_identity_sha256": runtime_identity_sha256,
            "planned_attempt_sha256": planned_attempt_sha256,
            "endpoint_config_id": endpoint_config_id,
            "matching_ledger_sha256": matching_ledger_sha256,
            "matching_report_sha256s": tuple(matching_report_sha256s),
            "receiver_count": receiver_count,
            "state_categorical_enabled": state_categorical_enabled,
            "model_categorical_enabled": model_categorical_enabled,
            "tuning_cell_sha256": tuning_cell_sha256,
            "optimizer_policy_sha256": optimizer_policy_sha256,
            "optimizer_learning_rate": optimizer_learning_rate,
            "optimizer_betas": H6_ADAMW_POLICY.betas,
            "optimizer_eps": H6_ADAMW_POLICY.eps,
            "optimizer_weight_decay": optimizer_weight_decay,
            "optimizer_amsgrad": H6_ADAMW_POLICY.amsgrad,
            "optimizer_maximize": H6_ADAMW_POLICY.maximize,
            "optimizer_foreach": H6_ADAMW_POLICY.foreach,
            "optimizer_capturable": H6_ADAMW_POLICY.capturable,
            "optimizer_differentiable": H6_ADAMW_POLICY.differentiable,
            "optimizer_fused": H6_ADAMW_POLICY.fused,
            "gradient_clip_max_norm": H6_ADAMW_POLICY.gradient_clip_max_norm,
            "objective_kind": objective_kind,
            "latent_enabled": latent_enabled,
        }
        return cls(
            **values,  # type: ignore[arg-type]
            authority_sha256=_owned_hash(
                "vfe4.h6.engine-authority.v3",
                values,
            ),
        )

    @classmethod
    def from_planned_attempt(
        cls,
        *,
        planned_attempt: object,
    ) -> H6EngineAuthorityV3:
        """Bind execution to one exact, outcome-free planned tuning attempt."""

        from .h6_experiment_v3 import H6PlannedAttemptV3, H6TuningCellV3

        if type(planned_attempt) is not H6PlannedAttemptV3:
            raise ValueError("engine authority requires an exact planned attempt")
        planned_attempt.__post_init__()
        if type(planned_attempt.tuning_cell) is not H6TuningCellV3:
            raise ValueError(
                "confirmatory execution requires its selected tuning cell authority"
            )
        cell = planned_attempt.tuning_cell
        cell.__post_init__()
        spec = planned_attempt.attempt_spec
        if spec.optimizer_policy_sha256 != H6_ADAMW_POLICY.optimizer_policy_sha256:
            raise ValueError("planned attempt does not bind the H6 AdamW policy")
        return cls.create(
            attempt_spec_sha256=spec.attempt_spec_sha256,
            endpoint_config_sha256=planned_attempt.endpoint_config_sha256,
            readiness_sha256=spec.readiness_sha256,
            readiness_matching_set_sha256=planned_attempt.matching_set_sha256,
            matching_set_sha256=planned_attempt.matching_set_sha256,
            matching_policy_sha256=planned_attempt.matching_policy_sha256,
            readiness_training_schedule_sha256=spec.training_schedule_sha256,
            training_schedule_sha256=spec.training_schedule_sha256,
            readiness_runtime_identity_sha256=spec.runtime_identity_sha256,
            runtime_identity_sha256=spec.runtime_identity_sha256,
            planned_attempt_sha256=planned_attempt.planned_attempt_sha256,
            endpoint_config_id=planned_attempt.endpoint_config_id,
            matching_ledger_sha256=planned_attempt.matching_ledger_sha256,
            matching_report_sha256s=(planned_attempt.matching_report_sha256s),
            receiver_count=planned_attempt.receiver_count,
            state_categorical_enabled=planned_attempt.state_categorical_enabled,
            model_categorical_enabled=planned_attempt.model_categorical_enabled,
            tuning_cell_sha256=cell.cell_sha256,
            optimizer_policy_sha256=spec.optimizer_policy_sha256,
            optimizer_learning_rate=cell.learning_rate,
            optimizer_weight_decay=cell.weight_decay,
            objective_kind=spec.objective_kind,
            latent_enabled=spec.recognition_factory_sha256 is not None,
        )


@dataclass(frozen=True, slots=True)
class H6LiveObjectiveTermV3:
    """One live objective contribution with an explicit partition."""

    partition: str
    receiver_t: int
    value: Tensor

    def __post_init__(self) -> None:
        _require_nonempty(self.partition, "objective partition")
        if type(self.receiver_t) is not int or self.receiver_t < 0:
            raise ValueError("objective receiver_t must be a nonnegative integer")
        if type(self.value) is not Tensor or self.value.numel() != 1:
            raise ValueError("live objective value must be an exact scalar tensor")
        if not bool(torch.isfinite(self.value.detach()).all()):
            raise ValueError("live objective value must be finite")

    @classmethod
    def create(
        cls,
        *,
        partition: str,
        receiver_t: int,
        value: Tensor,
    ) -> H6LiveObjectiveTermV3:
        return cls(partition=partition, receiver_t=receiver_t, value=value)


@dataclass(frozen=True, slots=True)
class H6PhaseObjectiveV3:
    """The phase-local objective presented to the optimizer."""

    objective_kind: Literal[
        "cross_entropy",
        "complete_elbo",
        "emission_only_ablation_non_elbo",
    ]
    is_elbo: bool
    terms: tuple[H6LiveObjectiveTermV3, ...]

    @property
    def partitions(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(term.partition for term in self.terms))

    @property
    def value(self) -> Tensor:
        first, *rest = (term.value for term in self.terms)
        result = first
        for value in rest:
            result = result + value
        return result

    def __post_init__(self) -> None:
        if self.objective_kind not in (
            "cross_entropy",
            "complete_elbo",
            "emission_only_ablation_non_elbo",
        ):
            raise ValueError("unsupported phase objective kind")
        if type(self.is_elbo) is not bool or self.is_elbo != (
            self.objective_kind == "complete_elbo"
        ):
            raise ValueError("phase objective kind and is_elbo disagree")
        if type(self.terms) is not tuple or not self.terms:
            raise ValueError("phase objective requires live terms")
        for term in self.terms:
            if type(term) is not H6LiveObjectiveTermV3:
                raise ValueError("phase objective terms must be exact live terms")
            term.__post_init__()
        devices = {term.value.device for term in self.terms}
        dtypes = {term.value.dtype for term in self.terms}
        if len(devices) != 1 or len(dtypes) != 1:
            raise ValueError("phase objective terms must share device and dtype")
        if self.objective_kind in (
            "cross_entropy",
            "emission_only_ablation_non_elbo",
        ) and any(term.partition != "emission" for term in self.terms):
            raise ValueError("non-ELBO objective accepts only live emission terms")
        if self.objective_kind == "cross_entropy" and len(self.terms) != 1:
            raise ValueError("cross-entropy phase requires exactly one CE/NLL term")
        if self.objective_kind == "complete_elbo" and "emission" not in self.partitions:
            raise ValueError("complete ELBO phase is missing its emission partition")
        if not bool(torch.isfinite(self.value.detach()).all()):
            raise ValueError("phase objective total must be finite")

    @classmethod
    def complete_elbo(
        cls,
        terms: tuple[H6LiveObjectiveTermV3, ...],
    ) -> H6PhaseObjectiveV3:
        return cls("complete_elbo", True, tuple(terms))

    @classmethod
    def emission_only(
        cls,
        terms: tuple[H6LiveObjectiveTermV3, ...],
    ) -> H6PhaseObjectiveV3:
        return cls("emission_only_ablation_non_elbo", False, tuple(terms))

    @classmethod
    def cross_entropy(
        cls,
        term: H6LiveObjectiveTermV3,
    ) -> H6PhaseObjectiveV3:
        return cls("cross_entropy", False, (term,))


@dataclass(frozen=True, slots=True)
class H6PhaseRecordV3:
    """Immutable audit record for one active optimizer phase."""

    phase: TrainingPhase
    objective_kind: str
    is_elbo: bool
    partitions: tuple[str, ...]
    noise_sha256: str
    objective_value: float
    loss_value: float
    gradient_norm: float

    def canonical_payload(self) -> dict[str, object]:
        return {
            "phase": self.phase.value,
            "objective_kind": self.objective_kind,
            "is_elbo": self.is_elbo,
            "partitions": self.partitions,
            "noise_sha256": self.noise_sha256,
            "objective_value": self.objective_value,
            "loss_value": self.loss_value,
            "gradient_norm": self.gradient_norm,
        }

    def __post_init__(self) -> None:
        if type(self.phase) is not TrainingPhase:
            raise ValueError("phase record requires an exact TrainingPhase")
        if self.objective_kind not in (
            "cross_entropy",
            "complete_elbo",
            "emission_only_ablation_non_elbo",
        ):
            raise ValueError("phase record has an unsupported objective kind")
        if type(self.is_elbo) is not bool or self.is_elbo != (
            self.objective_kind == "complete_elbo"
        ):
            raise ValueError("phase record objective kind and is_elbo disagree")
        if type(self.partitions) is not tuple or not self.partitions:
            raise ValueError("phase record requires objective partitions")
        _require_sha256(self.noise_sha256, "noise_sha256")
        for name in ("objective_value", "loss_value", "gradient_norm"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite float")


@dataclass(frozen=True, slots=True)
class H6MetricRecordV3:
    """Cumulative metric row bound to the post-update cursor counts."""

    ordinal: int
    phase: TrainingPhase
    recognition_update_count: int
    model_update_count: int
    objective_value: float
    loss_value: float
    gradient_norm: float
    metric_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "phase": self.phase.value,
            "recognition_update_count": self.recognition_update_count,
            "model_update_count": self.model_update_count,
            "objective_value": self.objective_value,
            "loss_value": self.loss_value,
            "gradient_norm": self.gradient_norm,
        }

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("metric ordinal must be nonnegative")
        if type(self.phase) is not TrainingPhase:
            raise ValueError("metric phase must be exact")
        for name in ("recognition_update_count", "model_update_count"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be nonnegative")
        for name in ("objective_value", "loss_value", "gradient_norm"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise ValueError(f"metric {name} must be finite")
        if self.metric_sha256 != _owned_hash(
            "vfe4.h6.engine-metric-record.v3", self.canonical_payload()
        ):
            raise ValueError("metric record identity is stale")

    @classmethod
    def from_phase_record(
        cls,
        record: H6PhaseRecordV3,
        *,
        ordinal: int,
        cursor: H6AttemptCursorV3,
    ) -> H6MetricRecordV3:
        values = {
            "ordinal": ordinal,
            "phase": record.phase,
            "recognition_update_count": cursor.recognition_update_count,
            "model_update_count": cursor.model_update_count,
            "objective_value": record.objective_value,
            "loss_value": record.loss_value,
            "gradient_norm": record.gradient_norm,
        }
        payload = {**values, "phase": record.phase.value}
        return cls(
            **values,
            metric_sha256=_owned_hash("vfe4.h6.engine-metric-record.v3", payload),
        )


@dataclass(frozen=True, slots=True)
class H6TrainingBatchResultV3:
    """Complete cumulative state at one exact phase boundary."""

    authority_sha256: str
    latent_enabled: bool
    cursor: H6AttemptCursorV3
    snapshot: (
        H6DetachedRecognitionSnapshotV3
        | H6DetachedBatchRecognitionSnapshotV3
        | None
    )
    phase_records: tuple[H6PhaseRecordV3, ...]
    metric_records: tuple[H6MetricRecordV3, ...]
    recognition_update_count: int
    model_update_count: int
    gradient_clip_count: int
    checkpoint_phases: tuple[TrainingPhase, ...]
    result_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "authority_sha256": self.authority_sha256,
            "latent_enabled": self.latent_enabled,
            "cursor_sha256": self.cursor.cursor_sha256,
            "snapshot_sha256": (
                None if self.snapshot is None else self.snapshot.snapshot_sha256
            ),
            "phase_records": tuple(
                record.canonical_payload() for record in self.phase_records
            ),
            "metric_record_sha256s": tuple(
                record.metric_sha256 for record in self.metric_records
            ),
            "recognition_update_count": self.recognition_update_count,
            "model_update_count": self.model_update_count,
            "gradient_clip_count": self.gradient_clip_count,
            "checkpoint_phases": tuple(phase.value for phase in self.checkpoint_phases),
        }

    def __post_init__(self) -> None:
        _require_sha256(self.authority_sha256, "authority_sha256")
        if type(self.latent_enabled) is not bool:
            raise ValueError("result latent_enabled must be exact")
        if type(self.cursor) is not H6AttemptCursorV3:
            raise ValueError("training result requires an exact v3 cursor")
        self.cursor.__post_init__()
        if self.snapshot is not None:
            if type(self.snapshot) not in (
                H6DetachedRecognitionSnapshotV3,
                H6DetachedBatchRecognitionSnapshotV3,
            ):
                raise ValueError("training result snapshot has the wrong type")
            self.snapshot.__post_init__()
        if type(self.phase_records) is not tuple or any(
            type(record) is not H6PhaseRecordV3 for record in self.phase_records
        ):
            raise ValueError("training result phase records must be exact")
        for record in self.phase_records:
            record.__post_init__()
        if (
            type(self.metric_records) is not tuple
            or len(self.metric_records) != len(self.phase_records)
            or any(
                type(record) is not H6MetricRecordV3 for record in self.metric_records
            )
        ):
            raise ValueError(
                "training result requires one cumulative metric per active phase"
            )
        for ordinal, (phase_record, metric_record) in enumerate(
            zip(self.phase_records, self.metric_records, strict=True)
        ):
            metric_record.__post_init__()
            if (
                metric_record.ordinal != ordinal
                or metric_record.phase is not phase_record.phase
                or metric_record.objective_value != phase_record.objective_value
                or metric_record.loss_value != phase_record.loss_value
                or metric_record.gradient_norm != phase_record.gradient_norm
            ):
                raise ValueError("phase and metric record histories disagree")
        if (
            self.recognition_update_count != self.cursor.recognition_update_count
            or self.model_update_count != self.cursor.model_update_count
        ):
            raise ValueError("training result update counts disagree with its cursor")
        if type(self.gradient_clip_count) is not int or self.gradient_clip_count != len(
            self.phase_records
        ):
            raise ValueError("training result gradient-clip count is invalid")
        if type(self.checkpoint_phases) is not tuple or any(
            type(phase) is not TrainingPhase for phase in self.checkpoint_phases
        ):
            raise ValueError("training result checkpoint phases must be exact")
        if self.latent_enabled:
            if (
                self.cursor.next_phase is TrainingPhase.IMMUTABLE_DETACHED_SNAPSHOT
                and self.snapshot is not None
            ):
                raise ValueError(
                    "pre-snapshot result cannot carry a recognition snapshot"
                )
            if self.cursor.next_phase is TrainingPhase.MODEL_ADAMW:
                if (
                    self.snapshot is None
                    or self.snapshot.post_recognition_cursor_sha256
                    != self.cursor.cursor_sha256
                ):
                    raise ValueError(
                        "model-phase result lacks its exact persisted snapshot"
                    )
        elif self.snapshot is not None:
            raise ValueError("no-latent result cannot carry recognition state")
        _require_sha256(self.result_sha256, "result_sha256")
        if self.result_sha256 != _owned_hash(
            "vfe4.h6.training-batch-result.v3", self.canonical_payload()
        ):
            raise ValueError("training result identity is stale")


def _training_result(
    *,
    authority: H6EngineAuthorityV3,
    cursor: H6AttemptCursorV3,
    snapshot: (
        H6DetachedRecognitionSnapshotV3
        | H6DetachedBatchRecognitionSnapshotV3
        | None
    ),
    phase_records: tuple[H6PhaseRecordV3, ...],
    metric_records: tuple[H6MetricRecordV3, ...],
    checkpoint_phases: tuple[TrainingPhase, ...],
) -> H6TrainingBatchResultV3:
    values = {
        "authority_sha256": authority.authority_sha256,
        "latent_enabled": authority.latent_enabled,
        "cursor": cursor,
        "snapshot": snapshot,
        "phase_records": phase_records,
        "metric_records": metric_records,
        "recognition_update_count": cursor.recognition_update_count,
        "model_update_count": cursor.model_update_count,
        "gradient_clip_count": len(phase_records),
        "checkpoint_phases": checkpoint_phases,
    }
    provisional = object.__new__(H6TrainingBatchResultV3)
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    return H6TrainingBatchResultV3(
        **values,
        result_sha256=_owned_hash(
            "vfe4.h6.training-batch-result.v3",
            provisional.canonical_payload(),
        ),
    )


def _module_parameters(module: nn.Module, *, label: str) -> tuple[nn.Parameter, ...]:
    if not isinstance(module, nn.Module):
        raise ValueError(f"{label} must be a torch module")
    parameters = tuple(module.parameters())
    if not parameters:
        raise ValueError(f"{label} must expose trainable parameters")
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise ValueError(f"{label} contains duplicate parameters")
    return parameters


def _optimizer_parameters(
    optimizer: torch.optim.AdamW,
    *,
    label: str,
) -> tuple[nn.Parameter, ...]:
    if type(optimizer) is not torch.optim.AdamW:
        raise ValueError(f"{label} must be an exact AdamW optimizer")
    parameters: list[nn.Parameter] = []
    for group in optimizer.param_groups:
        raw = group.get("params")
        if type(raw) is not list or not raw:
            raise ValueError(f"{label} has an invalid parameter group")
        for parameter in raw:
            if type(parameter) is not nn.Parameter:
                raise ValueError(f"{label} must bind exact Parameters")
            parameters.append(parameter)
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise ValueError(f"{label} binds a parameter more than once")
    return tuple(parameters)


def _validate_optimizer_binding(
    module_parameters: tuple[nn.Parameter, ...],
    optimizer: torch.optim.AdamW,
    *,
    label: str,
) -> None:
    optimizer_parameters = _optimizer_parameters(optimizer, label=label)
    if {id(parameter) for parameter in optimizer_parameters} != {
        id(parameter) for parameter in module_parameters
    }:
        raise ValueError(f"{label} does not bind the exact module parameter set")


def _validate_optimizer_cell(
    optimizer: torch.optim.AdamW,
    *,
    authority: H6EngineAuthorityV3,
    label: str,
) -> None:
    expected_policy: dict[str, object] = {
        "lr": authority.optimizer_learning_rate,
        "betas": authority.optimizer_betas,
        "eps": authority.optimizer_eps,
        "weight_decay": authority.optimizer_weight_decay,
        "amsgrad": authority.optimizer_amsgrad,
        "maximize": authority.optimizer_maximize,
        "foreach": authority.optimizer_foreach,
        "capturable": authority.optimizer_capturable,
        "differentiable": authority.optimizer_differentiable,
        "fused": authority.optimizer_fused,
    }
    for group in optimizer.param_groups:
        for field, expected in expected_policy.items():
            observed = group.get(field)
            if type(observed) is not type(expected) or observed != expected:
                raise ValueError(
                    f"{label} AdamW policy field {field} does not match "
                    "the bound H6 policy"
                )


def _set_phase_ownership(
    *,
    active: tuple[nn.Parameter, ...],
    inactive: tuple[nn.Parameter, ...],
    active_optimizer: torch.optim.AdamW,
    inactive_optimizer: torch.optim.AdamW | None,
) -> None:
    active_optimizer.zero_grad(set_to_none=True)
    if inactive_optimizer is not None:
        inactive_optimizer.zero_grad(set_to_none=True)
    for parameter in inactive:
        parameter.requires_grad_(False)
        parameter.grad = None
    for parameter in active:
        parameter.requires_grad_(True)
        parameter.grad = None


def _noise_for_phase(
    noise_factory: Callable[
        [TrainingPhase, H6AttemptCursorV3],
        tuple[Tensor, str],
    ],
    *,
    phase: TrainingPhase,
    cursor: H6AttemptCursorV3,
    consumed: set[str],
) -> tuple[Tensor, str]:
    produced = noise_factory(phase, cursor)
    if type(produced) is not tuple or len(produced) != 2:
        raise ValueError("noise_factory must return an exact (tensor, digest) pair")
    noise, digest = produced
    if not isinstance(noise, Tensor):
        raise ValueError("training noise must be a tensor")
    if noise.dtype is not torch.float64 or noise.device.type != "cpu":
        raise ValueError("counter-based training noise must be CPU float64")
    if noise.requires_grad or not bool(torch.isfinite(noise).all()):
        raise ValueError("counter-based training noise must be finite and detached")
    _require_sha256(digest, "counter-consumption digest")
    if digest == cursor.counter_consumption_sha256 or digest in consumed:
        raise ValueError("training phases cannot reuse counter-based noise")
    consumed.add(digest)
    return noise, digest


def _advanced_cursor(
    cursor: H6AttemptCursorV3,
    *,
    next_phase: TrainingPhase,
    counter_consumption_sha256: str | None = None,
    draw_block_delta: int = 0,
    recognition_update_delta: int = 0,
    model_update_delta: int = 0,
    batch_delta: int = 0,
    checkpoint_boundary_delta: int = 0,
) -> H6AttemptCursorV3:
    return H6AttemptCursorV3.create(
        attempt_spec_sha256=cursor.attempt_spec_sha256,
        pass_index=cursor.pass_index,
        batch_index=cursor.batch_index + batch_delta,
        next_phase=next_phase,
        example_ordinal=cursor.example_ordinal,
        draw_block=cursor.draw_block + draw_block_delta,
        counter_consumption_sha256=(
            cursor.counter_consumption_sha256
            if counter_consumption_sha256 is None
            else counter_consumption_sha256
        ),
        permutation_sha256=cursor.permutation_sha256,
        recognition_update_count=(
            cursor.recognition_update_count + recognition_update_delta
        ),
        model_update_count=cursor.model_update_count + model_update_delta,
        validation_boundary_count=cursor.validation_boundary_count,
        checkpoint_boundary_count=(
            cursor.checkpoint_boundary_count + checkpoint_boundary_delta
        ),
    )


def _objective_for_phase(
    objective_forward: Callable[..., H6PhaseObjectiveV3],
    *,
    authority: H6EngineAuthorityV3,
    phase: TrainingPhase,
    cursor: H6AttemptCursorV3,
    recognition_state: H6LiveRecognitionStateV3
    | H6BatchLiveRecognitionStateV3
    | H6DetachedRecognitionSnapshotV3
    | H6DetachedBatchRecognitionSnapshotV3
    | None,
    noise: Tensor,
    noise_sha256: str,
) -> H6PhaseObjectiveV3:
    allowed_partitions = (
        ("emission",)
        if authority.objective_kind
        in ("cross_entropy", "emission_only_ablation_non_elbo")
        else None
    )
    objective = objective_forward(
        phase=phase,
        recognition_state=recognition_state,
        noise=noise,
        noise_sha256=noise_sha256,
        cursor=cursor,
        allowed_partitions=allowed_partitions,
    )
    if type(objective) is not H6PhaseObjectiveV3:
        raise ValueError("objective_forward must return an exact phase objective")
    objective.__post_init__()
    if objective.objective_kind != authority.objective_kind:
        raise ValueError("phase objective kind is outside endpoint authority")
    if allowed_partitions is not None and objective.partitions != allowed_partitions:
        raise ValueError("phase objective escaped its allowed partition")
    return objective


def _optimizer_step(
    *,
    phase: TrainingPhase,
    objective: H6PhaseObjectiveV3,
    noise_sha256: str,
    active: tuple[nn.Parameter, ...],
    inactive: tuple[nn.Parameter, ...],
    optimizer: torch.optim.AdamW,
    gradient_clip_max_norm: float,
) -> H6PhaseRecordV3:
    loss = -objective.value
    if loss.numel() != 1 or not bool(torch.isfinite(loss.detach()).all()):
        raise FloatingPointError("training loss must be a finite scalar")
    loss.backward()
    if any(parameter.grad is not None for parameter in inactive):
        optimizer.zero_grad(set_to_none=True)
        raise RuntimeError("inactive phase parameters received gradients")
    gradients = tuple(
        parameter.grad for parameter in active if parameter.grad is not None
    )
    if not gradients:
        optimizer.zero_grad(set_to_none=True)
        raise RuntimeError("active phase produced no parameter gradients")
    if any(not bool(torch.isfinite(gradient).all()) for gradient in gradients):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("active phase produced nonfinite gradients")
    norm = torch.nn.utils.clip_grad_norm_(
        active,
        max_norm=gradient_clip_max_norm,
        error_if_nonfinite=True,
    )
    norm_value = float(norm.detach().to(device="cpu").item())
    if not math.isfinite(norm_value):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("active phase gradient norm is nonfinite")
    objective_value = float(objective.value.detach().to(device="cpu").item())
    loss_value = float(loss.detach().to(device="cpu").item())
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return H6PhaseRecordV3(
        phase=phase,
        objective_kind=objective.objective_kind,
        is_elbo=objective.is_elbo,
        partitions=objective.partitions,
        noise_sha256=noise_sha256,
        objective_value=objective_value,
        loss_value=loss_value,
        gradient_norm=norm_value,
    )


def _capture_recognition_snapshot(
    recognition_forward: Callable[
        [],
        H6LiveRecognitionStateV3 | H6BatchLiveRecognitionStateV3,
    ],
    *,
    authority: H6EngineAuthorityV3,
    post_recognition_cursor: H6AttemptCursorV3,
) -> (
    H6DetachedRecognitionSnapshotV3
    | H6DetachedBatchRecognitionSnapshotV3
):
    with torch.no_grad():
        state = recognition_forward()
    if type(state) is H6BatchLiveRecognitionStateV3:
        return H6DetachedBatchRecognitionSnapshotV3.capture(
            state,
            authority=authority,
            post_recognition_cursor=post_recognition_cursor,
        )
    if type(state) is not H6LiveRecognitionStateV3:
        raise ValueError(
            "snapshot capture requires a complete live recognition state"
        )
    return H6DetachedRecognitionSnapshotV3.capture(
        state,
        authority=authority,
        post_recognition_cursor=post_recognition_cursor,
    )


def _validate_start_cursor(
    *,
    authority: H6EngineAuthorityV3,
    cursor: H6AttemptCursorV3,
) -> None:
    if cursor.attempt_spec_sha256 != authority.attempt_spec_sha256:
        raise ValueError("cursor and engine authority attempt identity drift")
    if authority.latent_enabled:
        if cursor.next_phase is TrainingPhase.MODEL_CE_ADAMW:
            raise ValueError("latent endpoint cannot enter the CE-only phase")
        expected_delta = (
            0 if cursor.next_phase is TrainingPhase.RECOGNITION_ADAMW else 1
        )
        if (
            cursor.recognition_update_count - cursor.model_update_count
            != expected_delta
        ):
            raise ValueError("cursor update counts would replay or skip a latent phase")
    else:
        if (
            cursor.next_phase is not TrainingPhase.MODEL_CE_ADAMW
            or cursor.recognition_update_count != 0
        ):
            raise ValueError("no-latent endpoint requires the single CE/NLL phase")
        if (
            cursor.draw_block != 0
            or cursor.counter_consumption_sha256
            != H6_NO_COUNTER_CONSUMPTION_SHA256
        ):
            raise ValueError(
                "no-latent CE requires zero counter consumption"
            )


def _resume_histories(
    *,
    authority: H6EngineAuthorityV3,
    cursor: H6AttemptCursorV3,
    resume_state: H6TrainingBatchResultV3 | None,
) -> tuple[
    list[H6PhaseRecordV3],
    list[H6MetricRecordV3],
    list[TrainingPhase],
    H6DetachedRecognitionSnapshotV3
    | H6DetachedBatchRecognitionSnapshotV3
    | None,
]:
    pristine = (
        cursor.recognition_update_count == 0
        and cursor.model_update_count == 0
        and cursor.next_phase
        is (
            TrainingPhase.RECOGNITION_ADAMW
            if authority.latent_enabled
            else TrainingPhase.MODEL_CE_ADAMW
        )
    )
    if pristine:
        if resume_state is not None:
            raise ValueError("a pristine batch cannot import prior resume records")
        return [], [], [], None
    batch_boundary = (
        (
            authority.latent_enabled
            and cursor.next_phase is TrainingPhase.RECOGNITION_ADAMW
            and cursor.recognition_update_count
            == cursor.model_update_count
        )
        or (
            not authority.latent_enabled
            and cursor.next_phase is TrainingPhase.MODEL_CE_ADAMW
            and cursor.recognition_update_count == 0
        )
    )
    if batch_boundary and resume_state is None:
        # Completed batches are already committed in modules, optimizers, and
        # the cursor. Their graph-bearing records must not accumulate across
        # the corpus; only an interrupted in-batch phase imports resume state.
        return [], [], [], None
    if resume_state is None:
        raise ValueError(
            "noninitial phase requires persisted resume state with prior "
            "records and snapshot"
        )
    if type(resume_state) is not H6TrainingBatchResultV3:
        raise ValueError("resume_state must be an exact training result")
    resume_state.__post_init__()
    if (
        resume_state.authority_sha256 != authority.authority_sha256
        or resume_state.latent_enabled != authority.latent_enabled
        or resume_state.cursor != cursor
    ):
        raise ValueError("persisted resume state belongs to another authority")
    if resume_state.metric_records:
        last_metric = resume_state.metric_records[-1]
        if (
            last_metric.recognition_update_count != cursor.recognition_update_count
            or last_metric.model_update_count != cursor.model_update_count
        ):
            raise ValueError("persisted prior metric records are incomplete")
    if cursor.next_phase is TrainingPhase.MODEL_ADAMW:
        snapshot = resume_state.snapshot
        if (
            snapshot is None
            or snapshot.post_recognition_cursor_sha256 != cursor.cursor_sha256
        ):
            raise ValueError("model-phase resume requires its exact persisted snapshot")
    return (
        list(resume_state.phase_records),
        list(resume_state.metric_records),
        list(resume_state.checkpoint_phases),
        resume_state.snapshot,
    )


def _append_metric(
    *,
    phase_records: list[H6PhaseRecordV3],
    metric_records: list[H6MetricRecordV3],
    record: H6PhaseRecordV3,
    cursor: H6AttemptCursorV3,
) -> None:
    phase_records.append(record)
    metric_records.append(
        H6MetricRecordV3.from_phase_record(
            record,
            ordinal=len(metric_records),
            cursor=cursor,
        )
    )


def run_h6_training_batch_v3(
    *,
    authority: H6EngineAuthorityV3,
    cursor: H6AttemptCursorV3,
    model: nn.Module,
    recognition: nn.Module | None,
    model_optimizer: torch.optim.AdamW,
    recognition_optimizer: torch.optim.AdamW | None,
    recognition_forward: (
        Callable[
            [],
            H6LiveRecognitionStateV3 | H6BatchLiveRecognitionStateV3,
        ]
        | None
    ),
    objective_forward: Callable[..., H6PhaseObjectiveV3],
    noise_factory: Callable[
        [TrainingPhase, H6AttemptCursorV3],
        tuple[Tensor, str],
    ],
    stop_after_phase: TrainingPhase | None = None,
    declared_checkpoint_phases: tuple[TrainingPhase, ...] = (),
    checkpoint_at_batch_end: bool = True,
    resume_state: H6TrainingBatchResultV3 | None = None,
) -> H6TrainingBatchResultV3:
    """Execute each remaining phase exactly once from the supplied cursor."""

    if type(authority) is not H6EngineAuthorityV3:
        raise ValueError("training requires an exact H6 engine authority")
    authority.__post_init__()
    if type(cursor) is not H6AttemptCursorV3:
        raise ValueError("training requires an exact H6 attempt cursor")
    cursor.__post_init__()
    _validate_start_cursor(authority=authority, cursor=cursor)
    if not callable(objective_forward) or not callable(noise_factory):
        raise ValueError("training forward/noise factories must be callable")
    if stop_after_phase is not None and type(stop_after_phase) is not TrainingPhase:
        raise ValueError("stop_after_phase must be an exact TrainingPhase")
    if type(checkpoint_at_batch_end) is not bool:
        raise ValueError("checkpoint_at_batch_end must be an exact bool")
    if (
        type(declared_checkpoint_phases) is not tuple
        or any(type(phase) is not TrainingPhase for phase in declared_checkpoint_phases)
        or len(set(declared_checkpoint_phases)) != len(declared_checkpoint_phases)
    ):
        raise ValueError("declared checkpoint phases must be unique and exact")
    clip_norm = authority.gradient_clip_max_norm

    model_parameters = _module_parameters(model, label="model")
    _validate_optimizer_binding(
        model_parameters,
        model_optimizer,
        label="model_optimizer",
    )
    _validate_optimizer_cell(
        model_optimizer,
        authority=authority,
        label="model_optimizer",
    )
    if authority.latent_enabled:
        if (
            recognition is None
            or recognition_optimizer is None
            or recognition_forward is None
            or not callable(recognition_forward)
        ):
            raise ValueError("latent endpoint requires recognition phase bindings")
        recognition_parameters = _module_parameters(
            recognition,
            label="recognition",
        )
        _validate_optimizer_binding(
            recognition_parameters,
            recognition_optimizer,
            label="recognition_optimizer",
        )
        _validate_optimizer_cell(
            recognition_optimizer,
            authority=authority,
            label="recognition_optimizer",
        )
        if {id(parameter) for parameter in model_parameters} & {
            id(parameter) for parameter in recognition_parameters
        }:
            raise ValueError("model and recognition optimizer ownership overlaps")
    else:
        if (
            recognition is not None
            or recognition_optimizer is not None
            or recognition_forward is not None
        ):
            raise ValueError("no-latent endpoint cannot construct recognition state")
        recognition_parameters = ()

    (
        phase_records,
        metric_records,
        checkpoint_phases,
        snapshot,
    ) = _resume_histories(
        authority=authority,
        cursor=cursor,
        resume_state=resume_state,
    )
    current = cursor
    consumed_noise = {record.noise_sha256 for record in phase_records}

    while True:
        phase = current.next_phase
        boundary_already_accounted = False
        if phase is TrainingPhase.RECOGNITION_ADAMW:
            assert recognition_optimizer is not None
            assert recognition_forward is not None
            _set_phase_ownership(
                active=recognition_parameters,
                inactive=model_parameters,
                active_optimizer=recognition_optimizer,
                inactive_optimizer=model_optimizer,
            )
            noise, noise_sha256 = _noise_for_phase(
                noise_factory,
                phase=phase,
                cursor=current,
                consumed=consumed_noise,
            )
            recognition_state = recognition_forward()
            if type(recognition_state) not in (
                H6LiveRecognitionStateV3,
                H6BatchLiveRecognitionStateV3,
            ):
                raise ValueError(
                    "recognition_forward must return a complete live recognition state"
                )
            recognition_state.__post_init__()
            objective = _objective_for_phase(
                objective_forward,
                authority=authority,
                phase=phase,
                cursor=current,
                recognition_state=recognition_state,
                noise=noise,
                noise_sha256=noise_sha256,
            )
            record = _optimizer_step(
                phase=phase,
                objective=objective,
                noise_sha256=noise_sha256,
                active=recognition_parameters,
                inactive=model_parameters,
                optimizer=recognition_optimizer,
                gradient_clip_max_norm=clip_norm,
            )
            current = _advanced_cursor(
                current,
                next_phase=TrainingPhase.IMMUTABLE_DETACHED_SNAPSHOT,
                counter_consumption_sha256=noise_sha256,
                draw_block_delta=1,
                recognition_update_delta=1,
            )
            _append_metric(
                phase_records=phase_records,
                metric_records=metric_records,
                record=record,
                cursor=current,
            )
        elif phase is TrainingPhase.IMMUTABLE_DETACHED_SNAPSHOT:
            assert recognition_forward is not None
            for parameter in model_parameters:
                parameter.requires_grad_(False)
                parameter.grad = None
            post_recognition_cursor = _advanced_cursor(
                current,
                next_phase=TrainingPhase.MODEL_ADAMW,
                checkpoint_boundary_delta=(
                    1 if phase in declared_checkpoint_phases else 0
                ),
            )
            snapshot = _capture_recognition_snapshot(
                recognition_forward,
                authority=authority,
                post_recognition_cursor=post_recognition_cursor,
            )
            current = post_recognition_cursor
            boundary_already_accounted = phase in declared_checkpoint_phases
        elif phase is TrainingPhase.MODEL_ADAMW:
            assert recognition_optimizer is not None
            if snapshot is None:
                raise ValueError(
                    "model-phase resume requires the exact persisted snapshot"
                )
            snapshot.__post_init__()
            if (
                snapshot.attempt_spec_sha256 != authority.attempt_spec_sha256
                or snapshot.post_recognition_cursor_sha256 != current.cursor_sha256
            ):
                raise ValueError(
                    "persisted snapshot does not bind this exact model-phase cursor"
                )
            _set_phase_ownership(
                active=model_parameters,
                inactive=recognition_parameters,
                active_optimizer=model_optimizer,
                inactive_optimizer=recognition_optimizer,
            )
            noise, noise_sha256 = _noise_for_phase(
                noise_factory,
                phase=phase,
                cursor=current,
                consumed=consumed_noise,
            )
            objective = _objective_for_phase(
                objective_forward,
                authority=authority,
                phase=phase,
                cursor=current,
                recognition_state=snapshot,
                noise=noise,
                noise_sha256=noise_sha256,
            )
            record = _optimizer_step(
                phase=phase,
                objective=objective,
                noise_sha256=noise_sha256,
                active=model_parameters,
                inactive=recognition_parameters,
                optimizer=model_optimizer,
                gradient_clip_max_norm=clip_norm,
            )
            current = _advanced_cursor(
                current,
                next_phase=TrainingPhase.RECOGNITION_ADAMW,
                counter_consumption_sha256=noise_sha256,
                draw_block_delta=1,
                model_update_delta=1,
                batch_delta=1,
            )
            _append_metric(
                phase_records=phase_records,
                metric_records=metric_records,
                record=record,
                cursor=current,
            )
        elif phase is TrainingPhase.MODEL_CE_ADAMW:
            _set_phase_ownership(
                active=model_parameters,
                inactive=(),
                active_optimizer=model_optimizer,
                inactive_optimizer=None,
            )
            noise = torch.empty(
                (0,),
                dtype=torch.float64,
                device="cpu",
            ).detach()
            noise_sha256 = H6_NO_COUNTER_CONSUMPTION_SHA256
            objective = _objective_for_phase(
                objective_forward,
                authority=authority,
                phase=phase,
                cursor=current,
                recognition_state=None,
                noise=noise,
                noise_sha256=noise_sha256,
            )
            record = _optimizer_step(
                phase=phase,
                objective=objective,
                noise_sha256=noise_sha256,
                active=model_parameters,
                inactive=(),
                optimizer=model_optimizer,
                gradient_clip_max_norm=clip_norm,
            )
            current = _advanced_cursor(
                current,
                next_phase=TrainingPhase.MODEL_CE_ADAMW,
                model_update_delta=1,
                batch_delta=1,
            )
            _append_metric(
                phase_records=phase_records,
                metric_records=metric_records,
                record=record,
                cursor=current,
            )
        else:  # pragma: no cover - exhaustive over the exact enum.
            raise ValueError("unsupported H6 training phase")

        terminal = phase in (
            TrainingPhase.MODEL_ADAMW,
            TrainingPhase.MODEL_CE_ADAMW,
        )
        if phase in declared_checkpoint_phases or (
            terminal and checkpoint_at_batch_end
        ):
            checkpoint_phases.append(phase)
            if not boundary_already_accounted:
                current = _advanced_cursor(
                    current,
                    next_phase=current.next_phase,
                    checkpoint_boundary_delta=1,
                )
        if stop_after_phase is phase or terminal:
            break

    return _training_result(
        authority=authority,
        cursor=current,
        snapshot=snapshot,
        phase_records=tuple(phase_records),
        metric_records=tuple(metric_records),
        checkpoint_phases=tuple(checkpoint_phases),
    )


def _canonical_scalar(value: object, *, name: str) -> object:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value
    if type(value) in (tuple, list):
        return tuple(_canonical_scalar(item, name=f"{name} item") for item in value)
    raise ValueError(f"{name} is not canonically serializable")


def _state_tensor_record(
    *,
    role: str,
    name: str,
    tensor: Tensor,
    raw_stream: bytearray,
) -> dict[str, object]:
    raw = _tensor_raw_bytes(tensor)
    raw_stream.extend(len(raw).to_bytes(8, "little"))
    raw_stream.extend(raw)
    return {
        "role": role,
        "name": name,
        "dtype": str(tensor.dtype),
        "shape": tuple(tensor.shape),
        "requires_grad": bool(tensor.requires_grad),
        "byte_length": len(raw),
        "raw_bytes_sha256": hashlib.sha256(raw).hexdigest(),
    }


def canonical_engine_state_bytes_v3(
    *,
    model: nn.Module,
    recognition: nn.Module | None,
    model_optimizer: torch.optim.AdamW,
    recognition_optimizer: torch.optim.AdamW | None,
    result: H6TrainingBatchResultV3,
) -> bytes:
    """Return complete object-ID-free engine, snapshot, and history bytes."""

    if not isinstance(model, nn.Module):
        raise ValueError("model must be a torch module")
    if type(result) is not H6TrainingBatchResultV3:
        raise ValueError("result must be an exact cumulative training result")
    result.__post_init__()
    cursor = result.cursor
    modules: tuple[tuple[str, nn.Module], ...] = (
        (("model", model),)
        if recognition is None
        else (("model", model), ("recognition", recognition))
    )
    optimizers: tuple[tuple[str, torch.optim.AdamW], ...] = (
        (("model_optimizer", model_optimizer),)
        if recognition_optimizer is None
        else (
            ("model_optimizer", model_optimizer),
            ("recognition_optimizer", recognition_optimizer),
        )
    )
    if (recognition is None) != (recognition_optimizer is None):
        raise ValueError("recognition module and optimizer must be present together")
    if result.latent_enabled != (recognition is not None):
        raise ValueError("engine modules disagree with cumulative result authority")

    raw_stream = bytearray()
    module_records: list[dict[str, object]] = []
    parameter_names: dict[int, str] = {}
    for module_name, module in modules:
        if not isinstance(module, nn.Module):
            raise ValueError("canonical engine modules must be torch modules")
        named_parameters = dict(module.named_parameters())
        named_buffers = dict(module.named_buffers())
        if set(named_parameters) & set(named_buffers):
            raise ValueError("module parameter and buffer names collide")
        for local_name, tensor in sorted(
            (*named_parameters.items(), *named_buffers.items()),
            key=lambda item: item[0],
        ):
            qualified_name = f"{module_name}.{local_name}"
            role = "parameter" if local_name in named_parameters else "buffer"
            module_records.append(
                _state_tensor_record(
                    role=role,
                    name=qualified_name,
                    tensor=tensor,
                    raw_stream=raw_stream,
                )
            )
            if role == "parameter":
                if id(tensor) in parameter_names:
                    raise ValueError("canonical engine modules share a parameter")
                parameter_names[id(tensor)] = qualified_name

    optimizer_records: list[dict[str, object]] = []
    globally_bound: set[int] = set()
    for optimizer_name, optimizer in optimizers:
        parameters = _optimizer_parameters(optimizer, label=optimizer_name)
        group_records: list[dict[str, object]] = []
        for group_index, group in enumerate(optimizer.param_groups):
            bound_names: list[str] = []
            for parameter in group["params"]:
                qualified_name = parameter_names.get(id(parameter))
                if qualified_name is None or parameter is not next(
                    tensor
                    for _, module in modules
                    for tensor in module.parameters()
                    if id(tensor) == id(parameter)
                ):
                    raise ValueError("optimizer binds a foreign parameter")
                if id(parameter) in globally_bound:
                    raise ValueError("parameter is bound by multiple optimizers")
                globally_bound.add(id(parameter))
                bound_names.append(qualified_name)
            hyperparameters = tuple(
                (
                    key,
                    _canonical_scalar(
                        value,
                        name=f"{optimizer_name} {key}",
                    ),
                )
                for key, value in sorted(group.items())
                if key != "params"
            )
            group_records.append(
                {
                    "index": group_index,
                    "parameter_names": tuple(bound_names),
                    "hyperparameters": hyperparameters,
                }
            )
        state_records: list[dict[str, object]] = []
        for parameter in sorted(
            parameters,
            key=lambda item: parameter_names[id(item)],
        ):
            qualified_name = parameter_names[id(parameter)]
            values: list[dict[str, object] | tuple[str, object]] = []
            for state_name, value in sorted(optimizer.state.get(parameter, {}).items()):
                if isinstance(value, Tensor):
                    values.append(
                        _state_tensor_record(
                            role="optimizer_state",
                            name=f"{optimizer_name}.{qualified_name}.{state_name}",
                            tensor=value,
                            raw_stream=raw_stream,
                        )
                    )
                else:
                    values.append(
                        (
                            state_name,
                            _canonical_scalar(
                                value,
                                name=f"{optimizer_name} state {state_name}",
                            ),
                        )
                    )
            state_records.append(
                {
                    "parameter_name": qualified_name,
                    "values": tuple(values),
                }
            )
        optimizer_records.append(
            {
                "name": optimizer_name,
                "groups": tuple(group_records),
                "states": tuple(state_records),
            }
        )
    if globally_bound != set(parameter_names):
        raise ValueError("not every engine parameter is bound exactly once")

    snapshot_records: list[dict[str, object]] = []
    if result.snapshot is not None:
        result.snapshot.__post_init__()
        snapshot_records = [
            _state_tensor_record(
                role="detached_recognition_snapshot",
                name=name,
                tensor=tensor,
                raw_stream=raw_stream,
            )
            for name, tensor in zip(
                result.snapshot.names,
                result.snapshot._tensors,
                strict=True,
            )
        ]
    manifest = {
        "schema": "h6-canonical-engine-state-v3",
        "modules": tuple(module_records),
        "optimizers": tuple(optimizer_records),
        "snapshot_tensors": tuple(snapshot_records),
        "cumulative_result": {
            **result.canonical_payload(),
            "result_sha256": result.result_sha256,
        },
        "cursor": {
            **cursor.canonical_payload(),
            "cursor_sha256": cursor.cursor_sha256,
        },
    }
    header = canonical_json_bytes(manifest)
    return (
        b"VFE4-H6-ENGINE-STATE-V3\x00"
        + len(header).to_bytes(8, "little")
        + header
        + bytes(raw_stream)
    )


__all__ = [
    "H6BatchLiveRecognitionStateV3",
    "H6DetachedBatchRecognitionSnapshotV3",
    "H6DetachedRecognitionSnapshotV3",
    "H6EngineAuthorityV3",
    "H6LiveRecognitionStateV3",
    "H6LiveObjectiveTermV3",
    "H6MetricRecordV3",
    "H6_NO_COUNTER_CONSUMPTION_SHA256",
    "H6PhaseObjectiveV3",
    "H6PhaseRecordV3",
    "H6TrainingBatchResultV3",
    "canonical_engine_state_bytes_v3",
    "run_h6_training_batch_v3",
]
