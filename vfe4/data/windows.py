"""Exact causal windows, validation safety fixture, and frozen H6 schedules."""

from __future__ import annotations

import hashlib
import math
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Iterable, Iterator, Literal

import numpy as np
import torch

from vfe4.artifacts.paths import owned_payload_path, regular_nonlink_payload
from vfe4.types.h6 import (
    EncodedTokenStorageIdentity,
    FrozenBatchSchedule,
    ValidationSafetyFixture,
    VocabularyIdentity,
)
from vfe4.types.training import DataCursor, PermutationManifest, WindowManifest

from .byte_tokenizer import BOS_ID, IGNORE_TARGET_ID, VOCABULARY_SIZE, ByteTokenizerV1
from .tokenizer import (
    FixtureDurabilityBackend,
    SyntheticFixtureSplitCapability,
    SyntheticFixtureTokenCacheRecord,
    SyntheticFixtureTokenizerSpec,
    open_fixture_token_cache,
)


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


# WikiText-103 uses a parallel, explicitly prefixed contract.  The H6 constants
# and types above remain unchanged.
WT103_SEQUENCE_LENGTH: Final = 128
WT103_WINDOW_STRIDE: Final = 128
WT103_BATCH_SIZE: Final = 128
WT103_EOT_TOKEN_ID: Final = 50_256
WT103_IGNORE_TARGET_ID: Final = -100
WT103_DATA_ORDER_SEED: Final = 2026072199

_WT103_WINDOW_ROWS_DOMAIN = b"VFE4-WT103-WINDOW-ROWS-V1\x00"
_WT103_SCHEDULE_DOMAIN = b"VFE4-WT103-WINDOW-SCHEDULE-V1\x00"


@dataclass(frozen=True, slots=True)
class WT103WindowRow:
    """One exactly-once block of adjacent token transitions."""

    window_id: int
    start_transition: int
    counted_targets: int

    def __post_init__(self) -> None:
        if (
            type(self.window_id) is not int
            or self.window_id < 0
            or type(self.start_transition) is not int
            or self.start_transition != self.window_id * WT103_WINDOW_STRIDE
            or type(self.counted_targets) is not int
            or not 1 <= self.counted_targets <= WT103_SEQUENCE_LENGTH
        ):
            raise ValueError("WT103 window row is not canonical")

    def canonical_bytes(self) -> bytes:
        self.__post_init__()
        return (
            self.window_id.to_bytes(8, "little")
            + self.start_transition.to_bytes(8, "little")
            + self.counted_targets.to_bytes(4, "little")
        )


def enumerate_wt103_window_rows(
    token_count: int,
) -> tuple[WT103WindowRow, ...]:
    """Enumerate every transition ``token[t] -> token[t+1]`` once."""

    if type(token_count) is not int or token_count < 2:
        raise ValueError("token_count must be an exact integer of at least two")
    transition_count = token_count - 1
    rows = tuple(
        WT103WindowRow(
            window_id=window_id,
            start_transition=start,
            counted_targets=min(
                WT103_SEQUENCE_LENGTH, transition_count - start
            ),
        )
        for window_id, start in enumerate(
            range(0, transition_count, WT103_WINDOW_STRIDE)
        )
    )
    if (
        sum(row.counted_targets for row in rows) != transition_count
        or tuple(row.window_id for row in rows) != tuple(range(len(rows)))
    ):
        raise RuntimeError("WT103 window enumeration lost a transition")
    return rows


def _regular_payload(path: Path, *, size: int, sha256: str) -> bytes:
    try:
        status = path.lstat()
    except OSError as exc:
        raise ValueError(f"window payload is unavailable: {exc}") from exc
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if (
        not stat.S_ISREG(status.st_mode)
        or stat.S_ISLNK(status.st_mode)
        or bool(getattr(status, "st_file_attributes", 0) & reparse)
    ):
        raise ValueError("window payload must be a regular nonlink file")
    if status.st_size != size:
        raise ValueError("window payload size does not match")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"window payload cannot be reopened: {exc}") from exc
    if hashlib.sha256(payload).hexdigest() != sha256:
        raise ValueError("window payload hash does not match")
    return payload


def _publish_window_payload(
    *,
    backend: FixtureDurabilityBackend,
    path: Path,
    payload: bytes,
) -> None:
    if not callable(getattr(backend, "publish_bytes", None)):
        raise ValueError("durability backend must expose publish_bytes")
    try:
        backend.publish_bytes(path, payload)
    except Exception as exc:
        raise ValueError(
            f"durable window publication failed: {exc}"
        ) from exc
    _regular_payload(
        path,
        size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


@dataclass(frozen=True, slots=True)
class CausalWindow:
    window_id: int
    start_transition: int
    inputs: torch.Tensor = field(repr=False, compare=False)
    targets: torch.Tensor = field(repr=False, compare=False)
    attention_mask: torch.Tensor = field(repr=False, compare=False)
    counted_targets: int

    def __post_init__(self) -> None:
        if type(self.window_id) is not int or self.window_id < 0:
            raise ValueError("window_id must be a nonnegative integer")
        if (
            type(self.start_transition) is not int
            or self.start_transition
            != self.window_id * WT103_WINDOW_STRIDE
        ):
            raise ValueError("start_transition does not match window_id")
        if (
            type(self.counted_targets) is not int
            or not 1 <= self.counted_targets <= WT103_SEQUENCE_LENGTH
        ):
            raise ValueError("counted_targets is outside 1..128")
        for tensor, dtype, name in (
            (self.inputs, torch.int64, "inputs"),
            (self.targets, torch.int64, "targets"),
            (self.attention_mask, torch.bool, "attention_mask"),
        ):
            if (
                type(tensor) is not torch.Tensor
                or tensor.device.type != "cpu"
                or tensor.dtype is not dtype
                or tuple(tensor.shape) != (WT103_SEQUENCE_LENGTH,)
                or not tensor.is_contiguous()
            ):
                raise ValueError(f"{name} must be contiguous CPU length 128")
        real = self.counted_targets
        if not bool(torch.all(self.attention_mask[:real])):
            raise ValueError("real input positions must be attended")
        if bool(torch.any(self.attention_mask[real:])):
            raise ValueError("padded input positions must be masked")
        if not bool(torch.all(self.inputs[real:] == WT103_EOT_TOKEN_ID)):
            raise ValueError("padded inputs must use the GPT-2 EOT token")
        if not bool(
            torch.all(self.targets[real:] == WT103_IGNORE_TARGET_ID)
        ):
            raise ValueError("padded targets must use -100")


@dataclass(frozen=True, slots=True)
class CausalWindowSet:
    """Manifest-bound view over one memory-mapped int32 split."""

    split: Literal["train", "validation", "test"]
    cache_record: SyntheticFixtureTokenCacheRecord
    tokenizer_spec: SyntheticFixtureTokenizerSpec
    manifest: WindowManifest
    rows: tuple[WT103WindowRow, ...]
    row_payload_relative_path: str
    token_payload_path: Path
    _tokens: np.memmap = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.split not in ("train", "validation", "test"):
            raise ValueError("WT103 window split is invalid")
        if type(self.cache_record) is not SyntheticFixtureTokenCacheRecord:
            raise ValueError("cache_record must be the exact synthetic record")
        self.cache_record.__post_init__()
        if self.cache_record.split != self.split:
            raise ValueError("cache split does not match the window set")
        if type(self.tokenizer_spec) is not SyntheticFixtureTokenizerSpec:
            raise ValueError("tokenizer_spec must be the exact synthetic spec")
        self.tokenizer_spec.__post_init__()
        if self.cache_record.tokenizer != self.tokenizer_spec:
            raise ValueError("window cache does not bind the tokenizer spec")
        if type(self.manifest) is not WindowManifest:
            raise ValueError("manifest must be the exact WindowManifest")
        self.manifest.__post_init__()
        if (
            self.manifest.split != self.split
            or self.manifest.token_payload_sha256
            != self.cache_record.payload_sha256
            or self.manifest.window_count != len(self.rows)
            or self.manifest.counted_targets
            != self.cache_record.token_count - 1
            or tuple(row.window_id for row in self.rows)
            != tuple(range(len(self.rows)))
        ):
            raise ValueError("window manifest does not bind the complete rows")
        if (
            not isinstance(self.token_payload_path, Path)
            or type(self._tokens) is not np.memmap
            or self._tokens.dtype != np.dtype("<i4")
            or tuple(self._tokens.shape)
            != (self.cache_record.token_count,)
        ):
            raise ValueError("window token source must be an exact int32 memmap")

    def window(self, window_id: int) -> CausalWindow:
        self.__post_init__()
        if type(window_id) is not int or not 0 <= window_id < len(self.rows):
            raise IndexError("WT103 window_id is out of range")
        row = self.rows[window_id]
        start = row.start_transition
        real = row.counted_targets
        input_values = np.asarray(
            self._tokens[start : start + real], dtype=np.int64
        ).copy()
        target_values = np.asarray(
            self._tokens[start + 1 : start + 1 + real],
            dtype=np.int64,
        ).copy()
        inputs = torch.full(
            (WT103_SEQUENCE_LENGTH,),
            WT103_EOT_TOKEN_ID,
            dtype=torch.int64,
            device="cpu",
        )
        targets = torch.full(
            (WT103_SEQUENCE_LENGTH,),
            WT103_IGNORE_TARGET_ID,
            dtype=torch.int64,
            device="cpu",
        )
        attention_mask = torch.zeros(
            (WT103_SEQUENCE_LENGTH,), dtype=torch.bool, device="cpu"
        )
        inputs[:real] = torch.from_numpy(input_values)
        targets[:real] = torch.from_numpy(target_values)
        attention_mask[:real] = True
        return CausalWindow(
            window_id=window_id,
            start_transition=start,
            inputs=inputs,
            targets=targets,
            attention_mask=attention_mask,
            counted_targets=real,
        )


def materialize_causal_window_set(
    *,
    cache_record: SyntheticFixtureTokenCacheRecord,
    tokenizer_spec: SyntheticFixtureTokenizerSpec,
    cache_root: Path,
    split_capability: SyntheticFixtureSplitCapability,
    artifact_root: Path,
    durability_backend: FixtureDurabilityBackend,
) -> CausalWindowSet:
    """Close row bytes, then expose an on-demand memory-mapped window view."""

    if type(cache_record) is not SyntheticFixtureTokenCacheRecord:
        raise ValueError("cache_record must be an exact synthetic cache record")
    cache_record.__post_init__()
    # This validates the complete payload and capability before mapping it.
    open_fixture_token_cache(
        identity=cache_record,
        spec=tokenizer_spec,
        cache_root=cache_root,
        capability=split_capability,
    )
    rows = enumerate_wt103_window_rows(cache_record.token_count)
    row_payload = _WT103_WINDOW_ROWS_DOMAIN + b"".join(
        row.canonical_bytes() for row in rows
    )
    row_payload_sha256 = hashlib.sha256(row_payload).hexdigest()
    row_relative = (
        f"window-manifests/{cache_record.split}/"
        f"{row_payload_sha256}.rows"
    )
    row_path = owned_payload_path(
        root=artifact_root,
        relative_path=row_relative,
        prepare_parents=True,
        forbidden_component_substrings=("v3_transformer",),
    )
    _publish_window_payload(
        backend=durability_backend,
        path=row_path,
        payload=row_payload,
    )
    manifest = WindowManifest.create(
        split=cache_record.split,
        token_payload_sha256=cache_record.payload_sha256,
        window_count=len(rows),
        counted_targets=cache_record.token_count - 1,
        payload_sha256=row_payload_sha256,
    )
    token_path = owned_payload_path(
        root=cache_root,
        relative_path=cache_record.cache_relative_path,
        prepare_parents=False,
        forbidden_component_substrings=("v3_transformer",),
    )
    regular_nonlink_payload(
        token_path,
        expected_size=cache_record.payload_size_bytes,
    )
    tokens = np.memmap(
        token_path,
        mode="r",
        dtype=np.dtype("<i4"),
        shape=(cache_record.token_count,),
    )
    return CausalWindowSet(
        split=cache_record.split,
        cache_record=cache_record,
        tokenizer_spec=tokenizer_spec,
        manifest=manifest,
        rows=rows,
        row_payload_relative_path=row_relative,
        token_payload_path=token_path,
        _tokens=tokens,
    )


@dataclass(frozen=True, slots=True)
class WindowSchedule:
    schema_version: str
    split: Literal["train", "validation", "test"]
    pass_index: int
    window_manifest_sha256: str
    permutation_manifest: PermutationManifest | None
    window_ids: tuple[int, ...]
    batch_size: int
    schedule_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != "wt103-window-schedule-v1":
            raise ValueError("unsupported WT103 schedule schema")
        if self.split not in ("train", "validation", "test"):
            raise ValueError("WT103 schedule split is invalid")
        if type(self.pass_index) is not int or self.pass_index < 0:
            raise ValueError("schedule pass_index must be nonnegative")
        if (
            type(self.window_manifest_sha256) is not str
            or len(self.window_manifest_sha256) != 64
        ):
            raise ValueError("window_manifest_sha256 is invalid")
        if (
            type(self.window_ids) is not tuple
            or not self.window_ids
            or any(type(item) is not int or item < 0 for item in self.window_ids)
            or sorted(self.window_ids) != list(range(len(self.window_ids)))
        ):
            raise ValueError(
                "schedule window IDs must be one complete unique inventory"
            )
        if type(self.batch_size) is not int or self.batch_size <= 0:
            raise ValueError("schedule batch_size must be positive")
        if self.split == "train":
            if (
                type(self.permutation_manifest) is not PermutationManifest
                or self.permutation_manifest.pass_index != self.pass_index
                or self.permutation_manifest.window_manifest_sha256
                != self.window_manifest_sha256
            ):
                raise ValueError("train schedule lacks its exact permutation")
            self.permutation_manifest.__post_init__()
        elif self.permutation_manifest is not None:
            raise ValueError("evaluation schedules cannot carry a permutation")
        expected = _schedule_sha256(
            split=self.split,
            pass_index=self.pass_index,
            window_manifest_sha256=self.window_manifest_sha256,
            permutation_manifest=self.permutation_manifest,
            window_ids=self.window_ids,
            batch_size=self.batch_size,
        )
        if self.schedule_sha256 != expected:
            raise ValueError("schedule_sha256 does not match schedule")


def _schedule_sha256(
    *,
    split: str,
    pass_index: int,
    window_manifest_sha256: str,
    permutation_manifest: PermutationManifest | None,
    window_ids: tuple[int, ...],
    batch_size: int,
) -> str:
    split_bytes = split.encode("ascii")
    permutation_sha = (
        b"\x00" * 32
        if permutation_manifest is None
        else bytes.fromhex(permutation_manifest.manifest_sha256)
    )
    return hashlib.sha256(
        _WT103_SCHEDULE_DOMAIN
        + len(split_bytes).to_bytes(1, "little")
        + split_bytes
        + pass_index.to_bytes(8, "little")
        + bytes.fromhex(window_manifest_sha256)
        + permutation_sha
        + batch_size.to_bytes(8, "little")
        + len(window_ids).to_bytes(8, "little")
        + b"".join(
            window_id.to_bytes(8, "little") for window_id in window_ids
        )
    ).hexdigest()


def build_train_schedule(
    *,
    windows: CausalWindowSet,
    pass_index: Literal[0, 1],
    artifact_root: Path,
    durability_backend: FixtureDurabilityBackend,
    batch_size: int = WT103_BATCH_SIZE,
) -> WindowSchedule:
    if type(windows) is not CausalWindowSet or windows.split != "train":
        raise ValueError("train schedule requires an exact train window set")
    windows.__post_init__()
    if type(pass_index) is not int or pass_index not in (0, 1):
        raise ValueError("train pass_index must be 0 or 1")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be positive")
    seed_sequence = np.random.SeedSequence(
        (WT103_DATA_ORDER_SEED, pass_index)
    )
    generator = np.random.Generator(np.random.PCG64(seed_sequence))
    permutation = generator.permutation(
        windows.manifest.window_count
    ).astype("<u8", copy=False)
    window_ids = tuple(int(item) for item in permutation.tolist())
    payload = permutation.tobytes(order="C")
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    manifest = PermutationManifest.create(
        pass_index=pass_index,
        numpy_version=np.__version__,
        window_manifest_sha256=windows.manifest.manifest_sha256,
        payload_sha256=payload_sha256,
    )
    relative = (
        f"permutations/pass-{pass_index}/{payload_sha256}.u64le"
    )
    permutation_path = owned_payload_path(
        root=artifact_root,
        relative_path=relative,
        prepare_parents=True,
        forbidden_component_substrings=("v3_transformer",),
    )
    _publish_window_payload(
        backend=durability_backend,
        path=permutation_path,
        payload=payload,
    )
    return WindowSchedule(
        schema_version="wt103-window-schedule-v1",
        split="train",
        pass_index=pass_index,
        window_manifest_sha256=windows.manifest.manifest_sha256,
        permutation_manifest=manifest,
        window_ids=window_ids,
        batch_size=batch_size,
        schedule_sha256=_schedule_sha256(
            split="train",
            pass_index=pass_index,
            window_manifest_sha256=windows.manifest.manifest_sha256,
            permutation_manifest=manifest,
            window_ids=window_ids,
            batch_size=batch_size,
        ),
    )


def build_evaluation_schedule(windows: CausalWindowSet) -> WindowSchedule:
    if (
        type(windows) is not CausalWindowSet
        or windows.split not in ("validation", "test")
    ):
        raise ValueError(
            "evaluation schedule requires validation or test windows"
        )
    windows.__post_init__()
    window_ids = tuple(range(windows.manifest.window_count))
    return WindowSchedule(
        schema_version="wt103-window-schedule-v1",
        split=windows.split,
        pass_index=0,
        window_manifest_sha256=windows.manifest.manifest_sha256,
        permutation_manifest=None,
        window_ids=window_ids,
        batch_size=WT103_BATCH_SIZE,
        schedule_sha256=_schedule_sha256(
            split=windows.split,
            pass_index=0,
            window_manifest_sha256=windows.manifest.manifest_sha256,
            permutation_manifest=None,
            window_ids=window_ids,
            batch_size=WT103_BATCH_SIZE,
        ),
    )


@dataclass(frozen=True, slots=True)
class CausalBatch:
    window_ids: tuple[int, ...]
    inputs: torch.Tensor = field(repr=False, compare=False)
    targets: torch.Tensor = field(repr=False, compare=False)
    attention_mask: torch.Tensor = field(repr=False, compare=False)
    counted_targets: int

    def __post_init__(self) -> None:
        if (
            type(self.window_ids) is not tuple
            or not self.window_ids
            or any(type(item) is not int or item < 0 for item in self.window_ids)
        ):
            raise ValueError("batch window_ids are invalid")
        batch_size = len(self.window_ids)
        for tensor, dtype, name in (
            (self.inputs, torch.int64, "inputs"),
            (self.targets, torch.int64, "targets"),
            (self.attention_mask, torch.bool, "attention_mask"),
        ):
            if (
                type(tensor) is not torch.Tensor
                or tensor.dtype is not dtype
                or tensor.device.type != "cpu"
                or tuple(tensor.shape)
                != (batch_size, WT103_SEQUENCE_LENGTH)
                or not tensor.is_contiguous()
            ):
                raise ValueError(f"batch {name} has an invalid representation")
        if (
            type(self.counted_targets) is not int
            or self.counted_targets <= 0
            or self.counted_targets
            != int(torch.sum(self.targets != WT103_IGNORE_TARGET_ID).item())
        ):
            raise ValueError("batch counted_targets is not exact")


def _batch_ids(
    schedule: WindowSchedule, *, batch_size: int
) -> tuple[tuple[int, ...], ...]:
    if batch_size != schedule.batch_size:
        raise ValueError("batch_size differs from the frozen schedule")
    return tuple(
        schedule.window_ids[offset : offset + batch_size]
        for offset in range(0, len(schedule.window_ids), batch_size)
    )


def _cursor_binding_sha256(schedule: WindowSchedule) -> str:
    if schedule.permutation_manifest is None:
        return schedule.schedule_sha256
    return schedule.permutation_manifest.manifest_sha256


def cursor_after_batches(
    *,
    windows: CausalWindowSet,
    schedule: WindowSchedule,
    completed_batch_count: int,
    batch_size: int = WT103_BATCH_SIZE,
) -> DataCursor:
    windows.__post_init__()
    schedule.__post_init__()
    if (
        schedule.window_manifest_sha256
        != windows.manifest.manifest_sha256
        or schedule.split != windows.split
    ):
        raise ValueError("schedule does not bind the window set")
    batches = _batch_ids(schedule, batch_size=batch_size)
    if (
        type(completed_batch_count) is not int
        or not 0 <= completed_batch_count <= len(batches)
    ):
        raise ValueError("completed_batch_count is outside the schedule")
    consumed_ids = tuple(
        window_id
        for batch in batches[:completed_batch_count]
        for window_id in batch
    )
    counted_targets = sum(
        windows.rows[window_id].counted_targets for window_id in consumed_ids
    )
    next_ids = (
        batches[completed_batch_count]
        if completed_batch_count < len(batches)
        else ()
    )
    return DataCursor.create(
        split=schedule.split,
        pass_index=schedule.pass_index,
        permutation_sha256=_cursor_binding_sha256(schedule),
        next_batch_ordinal=completed_batch_count,
        next_window_ids=next_ids,
        counted_targets=counted_targets,
    )


def _validate_cursor(
    *,
    cursor: DataCursor,
    windows: CausalWindowSet,
    schedule: WindowSchedule,
    batches: tuple[tuple[int, ...], ...],
) -> int:
    if type(cursor) is not DataCursor:
        raise ValueError("cursor must be the exact DataCursor")
    cursor.__post_init__()
    if (
        cursor.split != schedule.split
        or cursor.pass_index != schedule.pass_index
        or cursor.permutation_sha256 != _cursor_binding_sha256(schedule)
        or cursor.next_batch_ordinal > len(batches)
    ):
        raise ValueError("cursor does not bind the exact schedule")
    expected = cursor_after_batches(
        windows=windows,
        schedule=schedule,
        completed_batch_count=cursor.next_batch_ordinal,
        batch_size=schedule.batch_size,
    )
    if cursor != expected:
        raise ValueError("cursor next batch or counted denominator changed")
    return cursor.next_batch_ordinal


def iter_causal_batches(
    *,
    windows: CausalWindowSet,
    schedule: WindowSchedule,
    batch_size: int = WT103_BATCH_SIZE,
    cursor: DataCursor | None = None,
) -> Iterator[CausalBatch]:
    windows.__post_init__()
    schedule.__post_init__()
    if (
        schedule.window_manifest_sha256
        != windows.manifest.manifest_sha256
        or schedule.split != windows.split
    ):
        raise ValueError("schedule does not bind the supplied window set")
    batches = _batch_ids(schedule, batch_size=batch_size)
    start = (
        0
        if cursor is None
        else _validate_cursor(
            cursor=cursor,
            windows=windows,
            schedule=schedule,
            batches=batches,
        )
    )
    for window_ids in batches[start:]:
        rows = tuple(windows.window(window_id) for window_id in window_ids)
        yield CausalBatch(
            window_ids=window_ids,
            inputs=torch.stack(tuple(row.inputs for row in rows), dim=0),
            targets=torch.stack(tuple(row.targets for row in rows), dim=0),
            attention_mask=torch.stack(
                tuple(row.attention_mask for row in rows), dim=0
            ),
            counted_targets=sum(row.counted_targets for row in rows),
        )


__all__ = [
    "BATCH_SIZE",
    "CausalBatch",
    "CausalPrefix",
    "CausalWindow",
    "CausalWindowSet",
    "CausalWindows",
    "DataCursor",
    "PermutationManifest",
    "SEQUENCE_LENGTH",
    "SHARED_DATA_ORDER_SEED",
    "VALIDATION_SAFETY_FIXTURE_COUNT",
    "WINDOW_STRIDE",
    "WT103_BATCH_SIZE",
    "WT103_DATA_ORDER_SEED",
    "WT103_EOT_TOKEN_ID",
    "WT103_IGNORE_TARGET_ID",
    "WT103_SEQUENCE_LENGTH",
    "WT103_WINDOW_STRIDE",
    "WT103WindowRow",
    "WindowManifest",
    "WindowSchedule",
    "build_evaluation_schedule",
    "build_train_schedule",
    "build_causal_windows",
    "evaluation_batches",
    "enumerate_wt103_window_rows",
    "frozen_batch_schedule",
    "materialize_validation_safety_fixture",
    "materialize_causal_window_set",
    "cursor_after_batches",
    "iter_causal_batches",
    "quarter_pass_batches",
    "schedule_batches",
]
