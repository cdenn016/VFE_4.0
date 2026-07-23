"""Exact causal windows, validation safety fixture, and frozen H6 schedules."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Final, Iterable, Literal

import torch

from vfe4.types.h6 import (
    EncodedTokenStorageIdentity,
    FrozenBatchSchedule,
    ValidationSafetyFixture,
    VocabularyIdentity,
)

from .byte_tokenizer import BOS_ID, IGNORE_TARGET_ID, VOCABULARY_SIZE, ByteTokenizerV1


SEQUENCE_LENGTH: Final = 32
WINDOW_STRIDE: Final = 32
BATCH_SIZE: Final = 8
SHARED_DATA_ORDER_SEED: Final = 2026072199
VALIDATION_SAFETY_FIXTURE_COUNT: Final = 4096

_VALIDATION_RANK_DOMAIN = b"VFE4-H6-VALIDATION-SAFETY-RANK-V1\x00"
_VALIDATION_FIXTURE_DOMAIN = b"VFE4-H6-VALIDATION-SAFETY-FIXTURE-V1\x00"
_BATCH_DRAW_DOMAIN = b"VFE4-H6-BATCH-PERMUTATION-DRAW-V1\x00"


def _exact_ids(token_ids: Iterable[int]) -> tuple[int, ...]:
    try:
        values = tuple(token_ids)
    except TypeError as exc:
        raise ValueError("token_ids must be an immutable-compatible integer sequence") from exc
    if len(values) < 2:
        raise ValueError("at least two tokens are required to form a causal target")
    if any(type(value) is not int or not 0 <= value < VOCABULARY_SIZE for value in values):
        raise ValueError("token IDs must be exact integers in 0..257")
    return values


_CAUSAL_PREFIX_DOMAIN = b"VFE4-H6-CAUSAL-PREFIX-V1\x00"


@dataclass(frozen=True, init=False)
class CausalPrefix:
    """Target-free token history available strictly before one receiver."""

    receiver_t: int
    vocabulary: VocabularyIdentity
    prefix_sha256: str
    _token_ids: torch.Tensor = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.receiver_t) is not int or self.receiver_t <= 0:
            raise ValueError("receiver_t must be a positive integer")
        if type(self.vocabulary) is not VocabularyIdentity:
            raise ValueError("vocabulary must be an exact VocabularyIdentity")
        self.vocabulary.__post_init__()
        if self.vocabulary.size > 65536:
            raise ValueError("causal prefix vocabulary must fit canonical uint16-le IDs")
        tensor = self._token_ids
        if (
            type(tensor) is not torch.Tensor
            or tensor.device.type != "cpu"
            or tensor.dtype is not torch.int64
            or tensor.ndim != 1
            or tuple(tensor.shape) != (self.receiver_t - 1,)
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                "causal prefix token IDs must be contiguous CPU int64 shape (receiver_t-1,)"
            )
        if tensor.numel() and (
            bool(torch.any(tensor < 0).item())
            or bool(torch.any(tensor >= self.vocabulary.size).item())
        ):
            raise ValueError("causal prefix token IDs fall outside the vocabulary")
        if self.prefix_sha256 != hashlib.sha256(self._preimage()).hexdigest():
            raise ValueError("prefix_sha256 does not match the owned causal prefix")

    def _preimage(self) -> bytes:
        vocabulary_id = self.vocabulary.vocabulary_id.encode("utf-8")
        canonical_ids = b"".join(
            int(value).to_bytes(2, "little", signed=False)
            for value in self._token_ids.tolist()
        )
        return (
            _CAUSAL_PREFIX_DOMAIN
            + len(vocabulary_id).to_bytes(2, "little")
            + vocabulary_id
            + self.vocabulary.size.to_bytes(8, "little")
            + bytes.fromhex(self.vocabulary.tokenizer_spec_sha256)
            + self.receiver_t.to_bytes(8, "little")
            + len(self._token_ids).to_bytes(8, "little")
            + canonical_ids
        )

    @classmethod
    def create(
        cls,
        *,
        receiver_t: int,
        vocabulary: VocabularyIdentity,
        token_ids: torch.Tensor,
    ) -> "CausalPrefix":
        if type(receiver_t) is not int or receiver_t <= 0:
            raise ValueError("receiver_t must be a positive integer")
        if type(vocabulary) is not VocabularyIdentity:
            raise ValueError("vocabulary must be an exact VocabularyIdentity")
        if vocabulary.size > 65536:
            raise ValueError("causal prefix vocabulary must fit canonical uint16-le IDs")
        if type(token_ids) is not torch.Tensor:
            raise ValueError("token_ids must be an exact torch.Tensor")
        if (
            token_ids.device.type != "cpu"
            or token_ids.dtype is not torch.int64
            or token_ids.ndim != 1
            or tuple(token_ids.shape) != (receiver_t - 1,)
            or not token_ids.is_contiguous()
        ):
            raise ValueError(
                "causal prefix token IDs must be contiguous CPU int64 shape (receiver_t-1,)"
            )
        owned = token_ids.clone(memory_format=torch.contiguous_format)
        temporary = object.__new__(cls)
        object.__setattr__(temporary, "receiver_t", receiver_t)
        object.__setattr__(temporary, "vocabulary", vocabulary)
        object.__setattr__(temporary, "prefix_sha256", "0" * 64)
        object.__setattr__(temporary, "_token_ids", owned)
        digest = hashlib.sha256(temporary._preimage()).hexdigest()
        object.__setattr__(temporary, "prefix_sha256", digest)
        temporary.__post_init__()
        return temporary

    @property
    def token_ids(self) -> torch.Tensor:
        self.__post_init__()
        return self._token_ids.clone(memory_format=torch.contiguous_format)


@dataclass(frozen=True)
class CausalWindows:
    """A complete deterministic split windowing with no dropped target."""

    split: Literal["train", "validation", "test"]
    inputs: tuple[tuple[int, ...], ...]
    targets: tuple[tuple[int, ...], ...]
    attention_masks: tuple[tuple[bool, ...], ...]
    starts: tuple[int, ...]
    real_target_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.split not in ("train", "validation", "test"):
            raise ValueError("unsupported causal-window split")
        count = len(self.inputs)
        if (
            count <= 0
            or len(self.targets) != count
            or len(self.attention_masks) != count
            or len(self.starts) != count
            or len(self.real_target_counts) != count
        ):
            raise ValueError("causal-window columns must be nonempty and aligned")
        if self.starts != tuple(range(0, count * WINDOW_STRIDE, WINDOW_STRIDE)):
            raise ValueError("causal-window starts must be the exact stride-32 sequence")
        for inputs, targets, attention_mask, real_count in zip(
            self.inputs,
            self.targets,
            self.attention_masks,
            self.real_target_counts,
            strict=True,
        ):
            if len(inputs) != SEQUENCE_LENGTH or len(targets) != SEQUENCE_LENGTH:
                raise ValueError("every causal window must have length 32")
            if type(real_count) is not int or not 1 <= real_count <= SEQUENCE_LENGTH:
                raise ValueError("real target counts must be exact integers in 1..32")
            real_input_count = min(SEQUENCE_LENGTH, real_count + 1)
            expected_mask = (True,) * real_input_count + (False,) * (
                SEQUENCE_LENGTH - real_input_count
            )
            if attention_mask != expected_mask:
                raise ValueError("attention masks must be exact left-aligned input masks")
            for index, (input_id, target_id, active) in enumerate(
                zip(inputs, targets, attention_mask, strict=True)
            ):
                if type(input_id) is not int or not 0 <= input_id < VOCABULARY_SIZE:
                    raise ValueError("causal-window inputs must be IDs in 0..257")
                if type(target_id) is not int or (
                    target_id != IGNORE_TARGET_ID
                    and not 0 <= target_id < VOCABULARY_SIZE
                ):
                    raise ValueError("causal-window targets must be IDs or -100")
                if not active and input_id != BOS_ID:
                    raise ValueError("masked inputs must use BOS padding")
                if index >= real_count and target_id != IGNORE_TARGET_ID:
                    raise ValueError("targets beyond real_target_count must be ignored")
                if index < real_count and target_id == IGNORE_TARGET_ID:
                    raise ValueError("real targets cannot be ignored")
                if index + 1 < real_input_count and target_id != inputs[index + 1]:
                    raise ValueError("causal targets must equal the next real input")

    def __len__(self) -> int:
        return len(self.inputs)

    @property
    def counted_target_total(self) -> int:
        return sum(self.real_target_counts)

    def causal_prefix(
        self,
        *,
        window_index: int,
        receiver_t: int,
        vocabulary: VocabularyIdentity,
    ) -> CausalPrefix:
        if type(window_index) is not int or not 0 <= window_index < len(self):
            raise IndexError("causal-window index is out of range")
        real_input_count = sum(self.attention_masks[window_index])
        if type(receiver_t) is not int or not 1 <= receiver_t <= real_input_count + 1:
            raise ValueError("receiver_t exceeds the causal window input boundary")
        token_ids = torch.tensor(
            self.inputs[window_index][: receiver_t - 1],
            dtype=torch.int64,
            device="cpu",
        )
        return CausalPrefix.create(
            receiver_t=receiver_t,
            vocabulary=vocabulary,
            token_ids=token_ids,
        )


def build_causal_windows(
    token_ids: Iterable[int], *, split: Literal["train", "validation", "test"]
) -> CausalWindows:
    tokens = _exact_ids(token_ids)
    if split not in ("train", "validation", "test"):
        raise ValueError("unsupported causal-window split")
    starts = tuple(range(0, len(tokens) - 1, WINDOW_STRIDE))
    inputs: list[tuple[int, ...]] = []
    targets: list[tuple[int, ...]] = []
    attention_masks: list[tuple[bool, ...]] = []
    target_counts: list[int] = []
    for start in starts:
        input_values = tokens[start : start + SEQUENCE_LENGTH]
        target_values = tokens[start + 1 : start + 1 + SEQUENCE_LENGTH]
        real_target_count = len(target_values)
        inputs.append(
            input_values + (BOS_ID,) * (SEQUENCE_LENGTH - len(input_values))
        )
        targets.append(
            target_values
            + (IGNORE_TARGET_ID,) * (SEQUENCE_LENGTH - real_target_count)
        )
        target_counts.append(real_target_count)
        real_input_count = min(SEQUENCE_LENGTH, len(input_values))
        attention_masks.append(
            (True,) * real_input_count
            + (False,) * (SEQUENCE_LENGTH - real_input_count)
        )
    return CausalWindows(
        split,
        tuple(inputs),
        tuple(targets),
        tuple(attention_masks),
        starts,
        tuple(target_counts),
    )


def materialize_validation_safety_fixture(
    *,
    validation_tokens: Iterable[int],
    validation_storage_identity: EncodedTokenStorageIdentity,
) -> ValidationSafetyFixture:
    tokens = _exact_ids(validation_tokens)
    if type(validation_storage_identity) is not EncodedTokenStorageIdentity:
        raise ValueError("validation_storage_identity must be the exact identity record")
    encoded = ByteTokenizerV1().serialize(tokens)
    try:
        validation_storage_identity.verify_encoded_token_bytes(encoded)
    except ValueError as exc:
        raise ValueError("validation token identity does not match supplied tokens") from exc
    windows = build_causal_windows(tokens, split="validation")
    if len(windows) < VALIDATION_SAFETY_FIXTURE_COUNT:
        raise ValueError("validation split must contain at least 4,096 stride-32 windows")
    token_digest = bytes.fromhex(validation_storage_identity.encoded_token_sha256)
    ranked_indices = sorted(
        range(len(windows)),
        key=lambda index: (
            hashlib.sha256(
                _VALIDATION_RANK_DOMAIN
                + token_digest
                + index.to_bytes(8, "little")
            ).digest(),
            index,
        ),
    )[:VALIDATION_SAFETY_FIXTURE_COUNT]
    fixture_bytes = bytearray(
        _VALIDATION_FIXTURE_DOMAIN
        + token_digest
        + VALIDATION_SAFETY_FIXTURE_COUNT.to_bytes(4, "little")
    )
    starts: list[int] = []
    real_target_counts: list[int] = []
    for index in ranked_indices:
        start = windows.starts[index]
        real_target_count = windows.real_target_counts[index]
        row = tokens[start : start + SEQUENCE_LENGTH + 1]
        row += (BOS_ID,) * (SEQUENCE_LENGTH + 1 - len(row))
        fixture_bytes += start.to_bytes(8, "little")
        fixture_bytes += real_target_count.to_bytes(2, "little")
        fixture_bytes += b"".join(
            value.to_bytes(2, "little", signed=False) for value in row
        )
        starts.append(start)
        real_target_counts.append(real_target_count)
    return ValidationSafetyFixture.create(
        validation_token_sha256=validation_storage_identity.encoded_token_sha256,
        starts=tuple(starts),
        real_target_counts=tuple(real_target_counts),
        fixture_bytes=bytes(fixture_bytes),
    )


def _permutation(window_count: int, zero_based_pass_index: int) -> tuple[int, ...]:
    values = list(range(window_count))
    buffered_words: list[int] = []
    counter = 0
    for index in range(window_count - 1, 0, -1):
        modulus = index + 1
        limit = (1 << 64) - ((1 << 64) % modulus)
        while True:
            if not buffered_words:
                digest = hashlib.sha256(
                    _BATCH_DRAW_DOMAIN
                    + SHARED_DATA_ORDER_SEED.to_bytes(8, "little")
                    + zero_based_pass_index.to_bytes(8, "little")
                    + counter.to_bytes(8, "little")
                ).digest()
                buffered_words.extend(
                    int.from_bytes(digest[offset : offset + 8], "little")
                    for offset in (0, 8, 16, 24)
                )
                counter += 1
            draw = buffered_words.pop(0)
            if draw < limit:
                break
        swap_index = draw % modulus
        values[index], values[swap_index] = values[swap_index], values[index]
    return tuple(values)


def frozen_batch_schedule(
    *, window_count: int, zero_based_pass_index: int
) -> FrozenBatchSchedule:
    if type(window_count) is not int or window_count <= 0:
        raise ValueError("window_count must be a positive integer")
    if type(zero_based_pass_index) is not int or zero_based_pass_index < 0:
        raise ValueError("zero_based_pass_index must be a nonnegative integer")
    return FrozenBatchSchedule.create(
        zero_based_pass_index=zero_based_pass_index,
        window_count=window_count,
        permutation=_permutation(window_count, zero_based_pass_index),
    )


def schedule_batches(schedule: FrozenBatchSchedule) -> tuple[tuple[int, ...], ...]:
    if type(schedule) is not FrozenBatchSchedule:
        raise ValueError("schedule must be an exact FrozenBatchSchedule")
    schedule.__post_init__()
    return tuple(
        schedule.permutation[offset : offset + BATCH_SIZE]
        for offset in range(0, schedule.window_count, BATCH_SIZE)
    )


def quarter_pass_batches(
    schedule: FrozenBatchSchedule,
) -> tuple[tuple[int, ...], ...]:
    batches = schedule_batches(schedule)
    return batches[: math.ceil(len(batches) / 4)]


def evaluation_batches(window_count: int) -> tuple[tuple[int, ...], ...]:
    if type(window_count) is not int or window_count <= 0:
        raise ValueError("window_count must be a positive integer")
    return tuple(
        tuple(range(offset, min(offset + BATCH_SIZE, window_count)))
        for offset in range(0, window_count, BATCH_SIZE)
    )


__all__ = [
    "BATCH_SIZE",
    "CausalPrefix",
    "CausalWindows",
    "SEQUENCE_LENGTH",
    "SHARED_DATA_ORDER_SEED",
    "VALIDATION_SAFETY_FIXTURE_COUNT",
    "WINDOW_STRIDE",
    "build_causal_windows",
    "evaluation_batches",
    "frozen_batch_schedule",
    "materialize_validation_safety_fixture",
    "quarter_pass_batches",
    "schedule_batches",
]
