"""Exact bounded ordinary-tokenization streams for production source locking."""

from __future__ import annotations

import codecs
import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Protocol

from vfe4.artifacts.durability import (
    ContentAddressedDurabilityBackend,
    DurableFileIdentity,
    validate_regular_nonlink_sha256,
)

_LOWER_HEX = frozenset("0123456789abcdef")
_SIGNED_INT32_MIN = -(2**31)
_SIGNED_INT32_MAX = 2**31 - 1


class TokenizerStreamError(ValueError):
    """The bounded tokenizer stream failed closed."""


class ExactPieceTokenizerAdapter(Protocol):
    """Minimal adapter needed to reproduce ordinary regex-piece encoding."""

    vocabulary_size: int

    def encode_ordinary(self, text: str) -> list[int]: ...

    def split_regex_pieces(self, text: str) -> tuple[str, ...]: ...

    def encode_single_piece(self, piece: str) -> list[int]: ...

    def decode_token_bytes(self, token_ids: list[int]) -> bytes: ...


@dataclass(frozen=True, slots=True)
class TokenizerStreamLimits:
    """Explicit input, retained-piece, and emitted-payload memory bounds."""

    input_chunk_size_bytes: int = 1_048_576
    retained_piece_size_bytes: int = 1_048_576
    output_chunk_size_bytes: int = 1_048_576

    def __post_init__(self) -> None:
        if (
            type(self.input_chunk_size_bytes) is not int
            or self.input_chunk_size_bytes <= 0
            or type(self.retained_piece_size_bytes) is not int
            or self.retained_piece_size_bytes <= 0
            or type(self.output_chunk_size_bytes) is not int
            or self.output_chunk_size_bytes < 4
            or self.output_chunk_size_bytes % 4 != 0
        ):
            raise TokenizerStreamError(
                "tokenizer stream limits must be positive and the output "
                "chunk bound must be a multiple of four bytes"
            )


@dataclass(frozen=True, slots=True)
class TokenPayloadFacts:
    """Exact facts accumulated without retaining the token corpus."""

    token_count: int
    minimum_token_id: int
    maximum_token_id: int
    payload_size_bytes: int
    payload_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.token_count) is not int
            or self.token_count <= 0
            or type(self.minimum_token_id) is not int
            or type(self.maximum_token_id) is not int
            or not _SIGNED_INT32_MIN
            <= self.minimum_token_id
            <= self.maximum_token_id
            <= _SIGNED_INT32_MAX
            or self.payload_size_bytes != self.token_count * 4
            or type(self.payload_sha256) is not str
            or len(self.payload_sha256) != 64
            or any(character not in _LOWER_HEX for character in self.payload_sha256)
        ):
            raise TokenizerStreamError("token payload facts are malformed")


@dataclass(frozen=True, slots=True)
class PublishedTokenPayload:
    """Content-addressed publication and its independently accumulated facts."""

    path: Path
    durable_identity: DurableFileIdentity
    facts: TokenPayloadFacts

    def __post_init__(self) -> None:
        if (
            not isinstance(self.path, Path)
            or not self.path.is_absolute()
            or type(self.durable_identity) is not DurableFileIdentity
            or type(self.facts) is not TokenPayloadFacts
            or self.path.name != f"{self.facts.payload_sha256}{self.path.suffix}"
            or self.durable_identity.size_bytes != self.facts.payload_size_bytes
            or self.durable_identity.sha256 != self.facts.payload_sha256
        ):
            raise TokenizerStreamError(
                "content-addressed token publication does not match stream facts"
            )


def _require_sha256(value: str, name: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise TokenizerStreamError(f"{name} must be lowercase SHA-256")


def _validated_regex_pieces(
    adapter: ExactPieceTokenizerAdapter,
    text: str,
) -> tuple[str, ...]:
    pieces = adapter.split_regex_pieces(text)
    if (
        type(pieces) is not tuple
        or any(type(piece) is not str or not piece for piece in pieces)
        or "".join(pieces) != text
    ):
        raise TokenizerStreamError(
            "tokenizer regex coverage does not exactly cover retained text"
        )
    return pieces


def _validated_piece_tokens(
    adapter: ExactPieceTokenizerAdapter,
    piece: str,
) -> list[int]:
    token_ids = adapter.encode_single_piece(piece)
    if type(token_ids) is not list or not token_ids:
        raise TokenizerStreamError(
            "single regex piece must produce a nonempty canonical token list"
        )
    if any(
        type(token_id) is not int
        or not _SIGNED_INT32_MIN <= token_id <= _SIGNED_INT32_MAX
        for token_id in token_ids
    ):
        raise TokenizerStreamError(
            "token ID is outside the canonical signed int32 range"
        )
    if any(
        not 0 <= token_id < adapter.vocabulary_size
        for token_id in token_ids
    ):
        raise TokenizerStreamError(
            "token ID is outside the tokenizer vocabulary"
        )
    decoded = adapter.decode_token_bytes(token_ids)
    if type(decoded) is not bytes or decoded != piece.encode("utf-8"):
        raise TokenizerStreamError(
            "tokenizer piece decode mismatch under strict UTF-8 bytes"
        )
    return token_ids


class ExactBoundedTokenizerStream:
    """Single-use iterable of canonical bounded little-endian int32 chunks."""

    def __init__(
        self,
        *,
        raw_chunks: Iterable[bytes],
        expected_raw_size_bytes: int,
        expected_raw_sha256: str,
        adapter: ExactPieceTokenizerAdapter,
        limits: TokenizerStreamLimits,
    ) -> None:
        if (
            type(expected_raw_size_bytes) is not int
            or expected_raw_size_bytes < 0
            or type(limits) is not TokenizerStreamLimits
            or type(getattr(adapter, "vocabulary_size", None)) is not int
            or not 0
            < adapter.vocabulary_size
            <= _SIGNED_INT32_MAX
            or any(
                not callable(getattr(adapter, name, None))
                for name in (
                    "encode_ordinary",
                    "split_regex_pieces",
                    "encode_single_piece",
                    "decode_token_bytes",
                )
            )
        ):
            raise TokenizerStreamError("bounded tokenizer stream inputs are malformed")
        _require_sha256(expected_raw_sha256, "expected_raw_sha256")
        self._raw_chunks = raw_chunks
        self._expected_raw_size_bytes = expected_raw_size_bytes
        self._expected_raw_sha256 = expected_raw_sha256
        self._adapter = adapter
        self._limits = limits
        self._started = False
        self._facts: TokenPayloadFacts | None = None

    @property
    def facts(self) -> TokenPayloadFacts:
        if self._facts is None:
            raise TokenizerStreamError(
                "token payload facts are unavailable before full stream consumption"
            )
        return self._facts

    def __iter__(self) -> Iterator[bytes]:
        if self._started:
            raise TokenizerStreamError("tokenizer payload stream is single-use")
        self._started = True
        return self._iter_payload()

    def _iter_payload(self) -> Iterator[bytes]:
        decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
        raw_hasher = hashlib.sha256()
        raw_size = 0
        payload_hasher = hashlib.sha256()
        payload_size = 0
        token_count = 0
        minimum_token_id: int | None = None
        maximum_token_id: int | None = None
        retained_piece = ""
        token_buffer: list[int] = []
        tokens_per_chunk = self._limits.output_chunk_size_bytes // 4

        def accept_piece(piece: str) -> Iterator[bytes]:
            nonlocal payload_size
            nonlocal token_count
            nonlocal minimum_token_id
            nonlocal maximum_token_id
            for token_id in _validated_piece_tokens(self._adapter, piece):
                token_count += 1
                minimum_token_id = (
                    token_id
                    if minimum_token_id is None
                    else min(minimum_token_id, token_id)
                )
                maximum_token_id = (
                    token_id
                    if maximum_token_id is None
                    else max(maximum_token_id, token_id)
                )
                token_buffer.append(token_id)
                if len(token_buffer) == tokens_per_chunk:
                    payload = struct.pack(
                        f"<{tokens_per_chunk}i",
                        *token_buffer,
                    )
                    token_buffer.clear()
                    payload_hasher.update(payload)
                    payload_size += len(payload)
                    if len(payload) > self._limits.output_chunk_size_bytes:
                        raise TokenizerStreamError(
                            "emitted token-payload chunk exceeds configured bound"
                        )
                    yield payload

        try:
            for raw_chunk in self._raw_chunks:
                if type(raw_chunk) is not bytes:
                    raise TokenizerStreamError(
                        "raw input chunks must be exact bytes"
                    )
                if len(raw_chunk) > self._limits.input_chunk_size_bytes:
                    raise TokenizerStreamError(
                        "raw input chunk exceeds configured bound"
                    )
                raw_hasher.update(raw_chunk)
                raw_size += len(raw_chunk)
                decoded = decoder.decode(raw_chunk, final=False)
                if not decoded:
                    continue
                pieces = _validated_regex_pieces(
                    self._adapter,
                    retained_piece + decoded,
                )
                retained_piece = pieces[-1] if pieces else ""
                if (
                    len(retained_piece.encode("utf-8"))
                    > self._limits.retained_piece_size_bytes
                ):
                    raise TokenizerStreamError(
                        "retained regex piece exceeds configured bound"
                    )
                for piece in pieces[:-1]:
                    yield from accept_piece(piece)
            decoded_final = decoder.decode(b"", final=True)
        except UnicodeDecodeError as exc:
            raise TokenizerStreamError(
                "raw source is not strict UTF-8"
            ) from exc

        if (
            raw_size != self._expected_raw_size_bytes
            or raw_hasher.hexdigest() != self._expected_raw_sha256
        ):
            raise TokenizerStreamError("raw source identity drift detected")

        final_text = retained_piece + decoded_final
        if final_text:
            for piece in _validated_regex_pieces(self._adapter, final_text):
                yield from accept_piece(piece)
        if not token_count:
            raise TokenizerStreamError("tokenizer stream produced no token IDs")
        if token_buffer:
            payload = struct.pack(f"<{len(token_buffer)}i", *token_buffer)
            token_buffer.clear()
            payload_hasher.update(payload)
            payload_size += len(payload)
            if len(payload) > self._limits.output_chunk_size_bytes:
                raise TokenizerStreamError(
                    "emitted token-payload chunk exceeds configured bound"
                )
            yield payload
        assert minimum_token_id is not None
        assert maximum_token_id is not None
        self._facts = TokenPayloadFacts(
            token_count=token_count,
            minimum_token_id=minimum_token_id,
            maximum_token_id=maximum_token_id,
            payload_size_bytes=payload_size,
            payload_sha256=payload_hasher.hexdigest(),
        )


def _file_chunks(path: Path, block_size: int) -> Iterator[bytes]:
    with path.open("rb", buffering=0) as handle:
        while True:
            chunk = handle.read(block_size)
            if not chunk:
                return
            yield chunk


def publish_exact_bounded_token_payload(
    *,
    raw_path: Path,
    expected_raw_size_bytes: int,
    expected_raw_sha256: str,
    adapter: ExactPieceTokenizerAdapter,
    backend: ContentAddressedDurabilityBackend,
    destination_directory: Path,
    suffix: str = ".int32le",
    limits: TokenizerStreamLimits = TokenizerStreamLimits(),
) -> PublishedTokenPayload:
    """Publish one exact token stream through the Task 1 durability API."""

    if (
        not isinstance(raw_path, Path)
        or not raw_path.is_absolute()
        or not isinstance(destination_directory, Path)
        or not destination_directory.is_absolute()
        or not callable(
            getattr(backend, "publish_content_addressed_stream", None)
        )
    ):
        raise TokenizerStreamError("token publication inputs are malformed")
    validate_regular_nonlink_sha256(
        raw_path,
        expected_size_bytes=expected_raw_size_bytes,
        expected_sha256=expected_raw_sha256,
        block_size=limits.input_chunk_size_bytes,
    )
    stream = ExactBoundedTokenizerStream(
        raw_chunks=_file_chunks(raw_path, limits.input_chunk_size_bytes),
        expected_raw_size_bytes=expected_raw_size_bytes,
        expected_raw_sha256=expected_raw_sha256,
        adapter=adapter,
        limits=limits,
    )
    durable_identity = backend.publish_content_addressed_stream(
        destination_directory,
        stream,
        suffix=suffix,
        chunk_size_limit=limits.output_chunk_size_bytes,
    )
    validate_regular_nonlink_sha256(
        raw_path,
        expected_size_bytes=expected_raw_size_bytes,
        expected_sha256=expected_raw_sha256,
        block_size=limits.input_chunk_size_bytes,
    )
    facts = stream.facts
    return PublishedTokenPayload(
        path=destination_directory / f"{facts.payload_sha256}{suffix}",
        durable_identity=durable_identity,
        facts=facts,
    )


def verify_piece_stream_golden_vectors(
    *,
    adapter: ExactPieceTokenizerAdapter,
    golden_vectors: tuple[tuple[str, str, tuple[int, ...]], ...],
    limits: TokenizerStreamLimits = TokenizerStreamLimits(),
) -> None:
    """Cross-check bounded piece streaming against frozen ordinary vectors."""

    if type(golden_vectors) is not tuple or not golden_vectors:
        raise TokenizerStreamError("golden vector set must be a nonempty tuple")
    for label, text, expected_token_ids in golden_vectors:
        if (
            type(label) is not str
            or not label
            or type(text) is not str
            or type(expected_token_ids) is not tuple
        ):
            raise TokenizerStreamError("golden vector row is malformed")
        raw = text.encode("utf-8")
        raw_chunks = tuple(
            raw[index : index + limits.input_chunk_size_bytes]
            for index in range(0, len(raw), limits.input_chunk_size_bytes)
        )
        stream = ExactBoundedTokenizerStream(
            raw_chunks=raw_chunks,
            expected_raw_size_bytes=len(raw),
            expected_raw_sha256=hashlib.sha256(raw).hexdigest(),
            adapter=adapter,
            limits=limits,
        )
        piece_token_ids = tuple(
            value[0]
            for payload in stream
            for value in struct.iter_unpack("<i", payload)
        )
        ordinary_token_ids = adapter.encode_ordinary(text)
        if (
            type(ordinary_token_ids) is not list
            or any(type(token_id) is not int for token_id in ordinary_token_ids)
            or piece_token_ids != tuple(ordinary_token_ids)
            or piece_token_ids != expected_token_ids
        ):
            raise TokenizerStreamError(
                f"golden vector {label!r} piece stream differs from ordinary encoding"
            )
