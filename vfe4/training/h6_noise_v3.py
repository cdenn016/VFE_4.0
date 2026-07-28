"""Stateless counter-based continuous noise for H6 Prediction v3.

The stream deliberately owns no mutable generator.  A complete key names one
example and phase, SHA-256 expands blocks into four little-endian uint64
words, and the existing open-uniform/paired-Box-Muller numerical mapping is
reused exactly.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass

import torch
from torch import Tensor

from vfe4.types.h6 import TrainingPhase, canonical_json_bytes
from vfe4.types.h6_prediction_v3 import H6_COUNTER_MAPPING_SHA256


_LOWER_HEX = frozenset("0123456789abcdef")
_TRAINING_NORMAL_DOMAIN = b"vfe4.h6.training-rmc-normal.v1\x00"
_TRAINING_BATCH_CONSUMPTION_DOMAIN = (
    b"vfe4.h6.training-batch-counter-consumption.v3\x00"
)
_TRAINING_BATCH_KEY_INVENTORY_DOMAIN = (
    b"vfe4.h6.training-batch-counter-key-inventory.v3\x00"
)


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class H6TrainingCounterKeyV3:
    """Every coordinate required to reconstruct one phase's base noise."""

    attempt_spec_sha256: str
    pass_index: int
    batch_index: int
    phase: TrainingPhase
    example_ordinal: int
    sample_ordinal: int
    draw_block: int

    def __post_init__(self) -> None:
        _require_sha256(self.attempt_spec_sha256, "attempt_spec_sha256")
        for name in (
            "pass_index",
            "batch_index",
            "example_ordinal",
            "sample_ordinal",
            "draw_block",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if type(self.phase) is not TrainingPhase:
            raise ValueError("phase must be an exact TrainingPhase")
        if self.sample_ordinal != 0:
            raise ValueError("H6 v3 has exactly sample_ordinal zero")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "attempt_spec_sha256": self.attempt_spec_sha256,
            "pass_index": self.pass_index,
            "batch_index": self.batch_index,
            "phase": self.phase.value,
            "example_ordinal": self.example_ordinal,
            "sample_ordinal": self.sample_ordinal,
            "draw_block": self.draw_block,
        }

    @property
    def key_sha256(self) -> str:
        return hashlib.sha256(
            _TRAINING_NORMAL_DOMAIN + canonical_json_bytes(self.canonical_payload())
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class H6TrainingBatchNoiseV3:
    """Fixed-shape noise plus every ordered per-example counter identity."""

    tensor: Tensor
    keys: tuple[H6TrainingCounterKeyV3, ...]
    active_receiver_counts: tuple[int, ...]
    example_consumption_sha256s: tuple[str, ...]
    receiver_count: int
    latent_dimension: int
    consumption_sha256: str

    @property
    def key_sha256s(self) -> tuple[str, ...]:
        return tuple(key.key_sha256 for key in self.keys)

    @property
    def key_inventory_sha256(self) -> str:
        return hashlib.sha256(
            _TRAINING_BATCH_KEY_INVENTORY_DOMAIN
            + canonical_json_bytes(self.key_sha256s)
        ).hexdigest()

    def canonical_payload(self) -> dict[str, object]:
        return {
            "key_sha256s": self.key_sha256s,
            "key_inventory_sha256": self.key_inventory_sha256,
            "active_receiver_counts": self.active_receiver_counts,
            "example_consumption_sha256s": (
                self.example_consumption_sha256s
            ),
            "receiver_count": self.receiver_count,
            "latent_dimension": self.latent_dimension,
        }

    def __post_init__(self) -> None:
        if (
            type(self.keys) is not tuple
            or not self.keys
            or any(
                type(key) is not H6TrainingCounterKeyV3
                for key in self.keys
            )
        ):
            raise ValueError(
                "batch noise requires an ordered nonempty exact key inventory"
            )
        for key in self.keys:
            key.__post_init__()
        first = self.keys[0]
        shared_coordinates = (
            first.attempt_spec_sha256,
            first.pass_index,
            first.batch_index,
            first.phase,
            first.sample_ordinal,
            first.draw_block,
        )
        if any(
            (
                key.attempt_spec_sha256,
                key.pass_index,
                key.batch_index,
                key.phase,
                key.sample_ordinal,
                key.draw_block,
            )
            != shared_coordinates
            for key in self.keys[1:]
        ) or tuple(key.example_ordinal for key in self.keys) != tuple(
            range(len(self.keys))
        ):
            raise ValueError(
                "batch noise keys must share one phase and cover exact ordinals"
            )
        if (
            type(self.receiver_count) is not int
            or self.receiver_count <= 0
            or type(self.latent_dimension) is not int
            or self.latent_dimension <= 0
            or type(self.active_receiver_counts) is not tuple
            or len(self.active_receiver_counts) != len(self.keys)
            or any(
                type(count) is not int
                or not 1 <= count <= self.receiver_count
                for count in self.active_receiver_counts
            )
        ):
            raise ValueError(
                "batch noise active receiver inventory is invalid"
            )
        if (
            type(self.example_consumption_sha256s) is not tuple
            or len(self.example_consumption_sha256s) != len(self.keys)
        ):
            raise ValueError(
                "batch noise requires one consumption digest per example"
            )
        for digest in self.example_consumption_sha256s:
            _require_sha256(digest, "example_consumption_sha256")
        if (
            not isinstance(self.tensor, Tensor)
            or tuple(self.tensor.shape)
            != (
                len(self.keys),
                self.receiver_count,
                self.latent_dimension,
            )
            or self.tensor.dtype is not torch.float64
            or self.tensor.requires_grad
            or self.tensor.grad_fn is not None
            or not self.tensor.is_contiguous()
            or not bool(torch.isfinite(self.tensor).all())
        ):
            raise ValueError(
                "batch counter tensor must be finite contiguous float64"
            )
        _require_sha256(self.consumption_sha256, "consumption_sha256")
        expected = hashlib.sha256(
            _TRAINING_BATCH_CONSUMPTION_DOMAIN
            + bytes.fromhex(H6_COUNTER_MAPPING_SHA256)
            + canonical_json_bytes(
                {
                    "key_sha256s": self.key_sha256s,
                    "active_receiver_counts": (
                        self.active_receiver_counts
                    ),
                    "example_consumption_sha256s": (
                        self.example_consumption_sha256s
                    ),
                    "receiver_count": self.receiver_count,
                    "latent_dimension": self.latent_dimension,
                }
            )
        ).hexdigest()
        if self.consumption_sha256 != expected:
            raise ValueError("batch counter consumption identity is stale")


def _counter_word(key: H6TrainingCounterKeyV3, draw_index: int) -> int:
    if type(key) is not H6TrainingCounterKeyV3:
        raise ValueError("key must be an exact H6TrainingCounterKeyV3")
    key.__post_init__()
    if type(draw_index) is not int or draw_index < 0:
        raise ValueError("draw_index must be a nonnegative integer")
    block_index, word_index = divmod(draw_index, 4)
    digest = hashlib.sha256(
        _TRAINING_NORMAL_DOMAIN
        + canonical_json_bytes(key.canonical_payload())
        + block_index.to_bytes(8, "little")
    ).digest()
    offset = 8 * word_index
    return int.from_bytes(digest[offset : offset + 8], "little")


def training_open_uniform_v3(
    key: H6TrainingCounterKeyV3,
    *,
    draw_index: int,
) -> float:
    """Map one counter word to the same clamped open interval as H6 SMC."""

    word = _counter_word(key, draw_index)
    value = (float(word) + 0.5) / float(2**64)
    if value <= 0.0:
        return math.nextafter(0.0, 1.0)
    if value >= 1.0:
        return math.nextafter(1.0, 0.0)
    return value


def training_normal_values_v3(
    key: H6TrainingCounterKeyV3,
    *,
    count: int,
) -> tuple[float, ...]:
    """Expand one immutable key with the paired Box-Muller mapping."""

    if type(count) is not int or count <= 0:
        raise ValueError("count must be a positive integer")
    values: list[float] = []
    for pair_index in range((count + 1) // 2):
        first = training_open_uniform_v3(
            key,
            draw_index=2 * pair_index,
        )
        second = training_open_uniform_v3(
            key,
            draw_index=2 * pair_index + 1,
        )
        radius = math.sqrt(-2.0 * math.log(first))
        angle = 2.0 * math.pi * second
        values.append(radius * math.cos(angle))
        if len(values) < count:
            values.append(radius * math.sin(angle))
    return tuple(values)


def training_normal_tensor_v3(
    key: H6TrainingCounterKeyV3,
    *,
    receiver_count: int,
    latent_dimension: int,
    device: torch.device | str,
) -> tuple[Tensor, str]:
    """Generate CPU float64 receiver/channel noise, then transfer once."""

    for value, name in (
        (receiver_count, "receiver_count"),
        (latent_dimension, "latent_dimension"),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    destination = torch.device(device)
    values = training_normal_values_v3(
        key,
        count=receiver_count * latent_dimension,
    )
    cpu = torch.tensor(
        values,
        dtype=torch.float64,
        device="cpu",
    ).reshape(receiver_count, latent_dimension)
    if not cpu.is_contiguous():
        raise RuntimeError("training counter tensor must be contiguous")
    raw_bytes = b"".join(struct.pack("<d", value) for value in values)
    consumption_sha256 = hashlib.sha256(
        b"vfe4.h6.training-counter-consumption.v1\x00"
        + bytes.fromhex(H6_COUNTER_MAPPING_SHA256)
        + bytes.fromhex(key.key_sha256)
        + receiver_count.to_bytes(8, "little")
        + latent_dimension.to_bytes(8, "little")
        + raw_bytes
    ).hexdigest()
    result = cpu if destination.type == "cpu" else cpu.to(destination)
    return result, consumption_sha256


def training_batch_normal_tensor_v3(
    *,
    attempt_spec_sha256: str,
    pass_index: int,
    batch_index: int,
    phase: TrainingPhase,
    draw_block: int,
    example_count: int,
    receiver_count: int,
    active_receiver_counts: tuple[int, ...],
    latent_dimension: int,
    device: torch.device | str,
) -> H6TrainingBatchNoiseV3:
    """Generate one exact counter stream per example in ordinal order."""

    if type(example_count) is not int or not 1 <= example_count <= 8:
        raise ValueError("example_count must be between one and eight")
    if (
        type(active_receiver_counts) is not tuple
        or len(active_receiver_counts) != example_count
        or type(receiver_count) is not int
        or receiver_count <= 0
        or any(
            type(count) is not int or not 1 <= count <= receiver_count
            for count in active_receiver_counts
        )
    ):
        raise ValueError(
            "active_receiver_counts must bind every exact active prefix"
        )
    keys = tuple(
        H6TrainingCounterKeyV3(
            attempt_spec_sha256=attempt_spec_sha256,
            pass_index=pass_index,
            batch_index=batch_index,
            phase=phase,
            example_ordinal=example_ordinal,
            sample_ordinal=0,
            draw_block=draw_block,
        )
        for example_ordinal in range(example_count)
    )
    active_rows_and_digests = tuple(
        training_normal_tensor_v3(
            key,
            receiver_count=active_receiver_count,
            latent_dimension=latent_dimension,
            device="cpu",
        )
        for key, active_receiver_count in zip(
            keys,
            active_receiver_counts,
            strict=True,
        )
    )
    padded_rows: list[Tensor] = []
    for row, active_receiver_count in zip(
        (item[0] for item in active_rows_and_digests),
        active_receiver_counts,
        strict=True,
    ):
        padded = torch.zeros(
            (receiver_count, latent_dimension),
            dtype=torch.float64,
            device="cpu",
        )
        padded[:active_receiver_count].copy_(row)
        padded_rows.append(padded)
    cpu = torch.stack(
        tuple(padded_rows),
        dim=0,
    ).contiguous()
    destination = torch.device(device)
    tensor = cpu if destination.type == "cpu" else cpu.to(destination)
    example_digests = tuple(
        digest for _row, digest in active_rows_and_digests
    )
    consumption_sha256 = hashlib.sha256(
        _TRAINING_BATCH_CONSUMPTION_DOMAIN
        + bytes.fromhex(H6_COUNTER_MAPPING_SHA256)
        + canonical_json_bytes(
            {
                "key_sha256s": tuple(key.key_sha256 for key in keys),
                "active_receiver_counts": tuple(active_receiver_counts),
                "example_consumption_sha256s": example_digests,
                "receiver_count": receiver_count,
                "latent_dimension": latent_dimension,
            }
        )
    ).hexdigest()
    return H6TrainingBatchNoiseV3(
        tensor=tensor,
        keys=keys,
        active_receiver_counts=tuple(active_receiver_counts),
        example_consumption_sha256s=example_digests,
        receiver_count=receiver_count,
        latent_dimension=latent_dimension,
        consumption_sha256=consumption_sha256,
    )


__all__ = [
    "H6TrainingBatchNoiseV3",
    "H6TrainingCounterKeyV3",
    "training_batch_normal_tensor_v3",
    "training_normal_tensor_v3",
    "training_normal_values_v3",
    "training_open_uniform_v3",
]
