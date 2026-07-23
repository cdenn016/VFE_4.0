from __future__ import annotations

import hashlib

import pytest

from vfe4.data.byte_tokenizer import (
    BOS_ID,
    EOS_ID,
    IGNORE_TARGET_ID,
    TOKENIZER_SPEC_BYTES,
    VOCABULARY_SIZE,
    ByteTokenizerV1,
)


def test_byte_tokenizer_preserves_exact_bytes_and_split_boundaries() -> None:
    raw = b"a\r\n\x00\xff"
    tokenizer = ByteTokenizerV1()

    encoded = tokenizer.encode(raw)

    assert encoded == (BOS_ID, 97, 13, 10, 0, 255, EOS_ID)
    assert tokenizer.decode(encoded) == raw
    assert (tokenizer.vocabulary_size, tokenizer.ignored_target_id) == (
        VOCABULARY_SIZE,
        IGNORE_TARGET_ID,
    )
    assert tokenizer.vocabulary_identity.vocabulary_id == "wikitext-2-byte-v1"
    assert tokenizer.vocabulary_identity.tokenizer_spec_sha256 == hashlib.sha256(
        TOKENIZER_SPEC_BYTES
    ).hexdigest()


def test_u16le_serialization_is_explicit_and_identity_bound() -> None:
    tokenizer = ByteTokenizerV1()
    encoded = (BOS_ID, 0, 255, EOS_ID)
    exact_bytes = b"\x00\x01\x00\x00\xff\x00\x01\x01"

    assert tokenizer.serialize(encoded) == exact_bytes
    assert tokenizer.deserialize(exact_bytes) == encoded
    identity = tokenizer.storage_identity(encoded)
    preimage = (
        b"VFE4-H6-U16LE-TOKENS-V1\x00"
        + len(encoded).to_bytes(8, "little")
        + exact_bytes
    )
    assert identity.byte_length == len(preimage)
    assert identity.encoded_token_sha256 == hashlib.sha256(preimage).hexdigest()
    identity.verify_encoded_token_bytes(exact_bytes)


def test_byte_tokenizer_fails_closed_on_nonbytes_or_invalid_storage() -> None:
    tokenizer = ByteTokenizerV1()
    for invalid in ("text", bytearray(b"x"), memoryview(b"x")):
        with pytest.raises((TypeError, ValueError)):
            tokenizer.encode(invalid)  # type: ignore[arg-type]
    for invalid in (
        b"\x00",
        (258).to_bytes(2, "little"),
        tokenizer.serialize((0, 1, 2)),
    ):
        with pytest.raises(ValueError):
            tokenizer.deserialize(invalid, require_split_boundaries=True)

