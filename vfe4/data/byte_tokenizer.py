"""Frozen byte-level tokenizer and endian-stable H6 token storage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Iterable

from vfe4.types.h6 import EncodedTokenStorageIdentity, VocabularyIdentity


BOS_ID: Final = 256
EOS_ID: Final = 257
VOCABULARY_SIZE: Final = 258
IGNORE_TARGET_ID: Final = -100

TOKENIZER_SPEC_BYTES: Final = (
    b"VFE4-H6-BYTE-TOKENIZER-V1\x00"
    b"raw-byte-to-identical-id:0..255\x00"
    b"BOS=256\x00EOS=257\x00vocabulary-size=258\x00ignored-target=-100\x00"
)


def _validated_ids(token_ids: Iterable[int]) -> tuple[int, ...]:
    try:
        values = tuple(token_ids)
    except TypeError as exc:
        raise ValueError("token_ids must be an iterable of integers") from exc
    if not values:
        raise ValueError("token_ids must be nonempty")
    if any(type(value) is not int or value < 0 or value >= VOCABULARY_SIZE for value in values):
        raise ValueError("token IDs must be exact integers in 0..257")
    return values


@dataclass(frozen=True)
class ByteTokenizerV1:
    """The exact split-local H6 mapping `[BOS] + raw bytes + [EOS]`."""

    @property
    def vocabulary_size(self) -> int:
        return VOCABULARY_SIZE

    @property
    def ignored_target_id(self) -> int:
        return IGNORE_TARGET_ID

    @property
    def vocabulary_identity(self) -> VocabularyIdentity:
        return VocabularyIdentity.from_tokenizer_spec(
            vocabulary_id="wikitext-2-byte-v1",
            size=VOCABULARY_SIZE,
            tokenizer_spec_bytes=TOKENIZER_SPEC_BYTES,
        )

    def encode(self, raw_bytes: bytes) -> tuple[int, ...]:
        if type(raw_bytes) is not bytes:
            raise TypeError("raw_bytes must be immutable bytes")
        return (BOS_ID, *raw_bytes, EOS_ID)

    def decode(self, token_ids: Iterable[int]) -> bytes:
        values = _validated_ids(token_ids)
        if len(values) < 2 or values[0] != BOS_ID or values[-1] != EOS_ID:
            raise ValueError("split token IDs must begin with BOS and end with EOS")
        if any(value > 255 for value in values[1:-1]):
            raise ValueError("only the split boundaries may contain BOS or EOS")
        return bytes(values[1:-1])

    def serialize(self, token_ids: Iterable[int]) -> bytes:
        values = _validated_ids(token_ids)
        return b"".join(value.to_bytes(2, "little", signed=False) for value in values)

    def deserialize(
        self, encoded_token_bytes: bytes, *, require_split_boundaries: bool = False
    ) -> tuple[int, ...]:
        if type(encoded_token_bytes) is not bytes:
            raise TypeError("encoded_token_bytes must be immutable bytes")
        if not encoded_token_bytes or len(encoded_token_bytes) % 2:
            raise ValueError("encoded token storage must contain complete uint16-le values")
        values = tuple(
            int.from_bytes(encoded_token_bytes[offset : offset + 2], "little")
            for offset in range(0, len(encoded_token_bytes), 2)
        )
        _validated_ids(values)
        if require_split_boundaries:
            self.decode(values)
        return values

    def storage_identity(
        self, token_ids: Iterable[int]
    ) -> EncodedTokenStorageIdentity:
        values = _validated_ids(token_ids)
        encoded = self.serialize(values)
        return EncodedTokenStorageIdentity.create(
            token_count=len(values), encoded_token_bytes=encoded
        )


__all__ = [
    "BOS_ID",
    "ByteTokenizerV1",
    "EOS_ID",
    "IGNORE_TARGET_ID",
    "TOKENIZER_SPEC_BYTES",
    "VOCABULARY_SIZE",
]
