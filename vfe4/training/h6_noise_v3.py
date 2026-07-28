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


__all__ = [
    "H6TrainingCounterKeyV3",
    "training_normal_tensor_v3",
    "training_normal_values_v3",
    "training_open_uniform_v3",
]
