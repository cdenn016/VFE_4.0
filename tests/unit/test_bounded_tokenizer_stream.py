from __future__ import annotations

import hashlib
import json
import re
import struct
from dataclasses import asdict
from pathlib import Path

import pytest


class _RegexByteAdapter:
    """Hermetic ordinary encoder with independently implemented piece splitting."""

    _pattern = re.compile(r"[A-Za-z]+|[^\x00-\x7f]|\s+|[^A-Za-z\s]+")
    vocabulary_size = 256

    def encode_ordinary(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def split_regex_pieces(self, text: str) -> tuple[str, ...]:
        return tuple(match.group(0) for match in self._pattern.finditer(text))

    def encode_single_piece(self, piece: str) -> list[int]:
        return list(piece.encode("utf-8"))

    def decode_token_bytes(self, token_ids: list[int]) -> bytes:
        return bytes(token_ids)


class _MergeSensitiveAdapter:
    """Independent fake whose BPE merges are not invariant to chunking."""

    vocabulary_size = 1_024
    _pattern = re.compile(r" ?[A-Za-z]+|\s+(?!\S)|\s+|[^A-Za-z\s]+")
    _merged_piece_tokens = {
        b" mergeable": 700,
        b" next": 701,
        b"mergeable": 702,
    }
    _merged_token_bytes = {
        token_id: piece
        for piece, token_id in _merged_piece_tokens.items()
    }
    _ordinary_fixtures = {
        "mergeable next": [702, 701],
        "  mergeable next": [32, 700, 701],
    }

    def encode_ordinary(self, text: str) -> list[int]:
        try:
            return list(self._ordinary_fixtures[text])
        except KeyError as exc:
            raise AssertionError(
                f"ordinary fixture was not frozen for {text!r}"
            ) from exc

    def split_regex_pieces(self, text: str) -> tuple[str, ...]:
        return tuple(match.group(0) for match in self._pattern.finditer(text))

    def encode_single_piece(self, piece: str) -> list[int]:
        raw = piece.encode()
        merged = self._merged_piece_tokens.get(raw)
        return [merged] if merged is not None else list(raw)

    def decode_token_bytes(self, token_ids: list[int]) -> bytes:
        decoded = bytearray()
        for token_id in token_ids:
            merged = self._merged_token_bytes.get(token_id)
            if merged is None:
                decoded.append(token_id)
            else:
                decoded.extend(merged)
        return bytes(decoded)


def _limits(
    *,
    input_chunk_size_bytes: int = 32,
    retained_piece_size_bytes: int = 32,
    output_chunk_size_bytes: int = 16,
):
    from vfe4.training.tokenizer_stream import TokenizerStreamLimits

    return TokenizerStreamLimits(
        input_chunk_size_bytes=input_chunk_size_bytes,
        retained_piece_size_bytes=retained_piece_size_bytes,
        output_chunk_size_bytes=output_chunk_size_bytes,
    )


def _consume(
    raw_chunks: tuple[bytes, ...],
    *,
    raw: bytes,
    adapter: object | None = None,
    limits: object | None = None,
):
    from vfe4.training.tokenizer_stream import ExactBoundedTokenizerStream

    stream = ExactBoundedTokenizerStream(
        raw_chunks=raw_chunks,
        expected_raw_size_bytes=len(raw),
        expected_raw_sha256=hashlib.sha256(raw).hexdigest(),
        adapter=adapter or _RegexByteAdapter(),
        limits=limits or _limits(),
    )
    payload_chunks = tuple(stream)
    payload = b"".join(payload_chunks)
    token_ids = tuple(
        value[0] for value in struct.iter_unpack("<i", payload)
    )
    return stream, payload_chunks, payload, token_ids


@pytest.mark.parametrize(
    "text",
    (
        "plain ASCII words!",
        "naïve café 世界",
        " \t  whitespace  ",
        "alpha\n\nbeta\r\n",
        "literal <|endoftext|> text",
    ),
)
def test_stream_matches_independent_ordinary_fixture_at_every_byte_boundary(
    text: str,
) -> None:
    raw = text.encode("utf-8")
    ordinary = tuple(_RegexByteAdapter().encode_ordinary(text))
    expected_payload = b"".join(
        struct.pack("<i", token_id) for token_id in ordinary
    )

    for boundary in range(len(raw) + 1):
        stream, chunks, payload, token_ids = _consume(
            (raw[:boundary], raw[boundary:]),
            raw=raw,
        )
        assert payload == expected_payload
        assert token_ids == ordinary
        assert all(len(chunk) <= 16 for chunk in chunks)
        assert all(len(chunk) == 16 for chunk in chunks[:-1])
        assert stream.facts.token_count == len(ordinary)
        assert stream.facts.minimum_token_id == min(ordinary)
        assert stream.facts.maximum_token_id == max(ordinary)
        assert stream.facts.payload_size_bytes == len(expected_payload)
        assert stream.facts.payload_sha256 == hashlib.sha256(
            expected_payload
        ).hexdigest()


@pytest.mark.parametrize(
    ("text", "expected_token_ids"),
    (
        ("mergeable next", (702, 701)),
        ("  mergeable next", (32, 700, 701)),
    ),
)
def test_stream_retains_merge_sensitive_word_and_whitespace_carry(
    text: str,
    expected_token_ids: tuple[int, ...],
) -> None:
    raw = text.encode()
    adapter = _MergeSensitiveAdapter()
    assert tuple(adapter.encode_ordinary(text)) == expected_token_ids

    for boundary in range(len(raw) + 1):
        _, _, _, token_ids = _consume(
            (raw[:boundary], raw[boundary:]),
            raw=raw,
            adapter=adapter,
        )
        assert token_ids == expected_token_ids


def test_stream_rejects_invalid_strict_utf8() -> None:
    raw = b"valid\xfftail"
    with pytest.raises(ValueError, match="strict UTF-8"):
        _consume((raw[:6], raw[6:]), raw=raw)


def test_stream_rejects_tokenizer_regex_coverage_gap() -> None:
    class _CoverageGapAdapter(_RegexByteAdapter):
        def split_regex_pieces(self, text: str) -> tuple[str, ...]:
            return tuple(piece for piece in super().split_regex_pieces(text) if piece != "!")

    raw = b"coverage!"
    with pytest.raises(ValueError, match="regex coverage"):
        _consume((raw,), raw=raw, adapter=_CoverageGapAdapter())


@pytest.mark.parametrize(
    ("expected_size_delta", "expected_sha256"),
    (
        (1, hashlib.sha256(b"source").hexdigest()),
        (0, hashlib.sha256(b"drifted").hexdigest()),
    ),
)
def test_stream_rejects_raw_source_identity_drift(
    expected_size_delta: int,
    expected_sha256: str,
) -> None:
    from vfe4.training.tokenizer_stream import ExactBoundedTokenizerStream

    raw = b"source"
    stream = ExactBoundedTokenizerStream(
        raw_chunks=(raw,),
        expected_raw_size_bytes=len(raw) + expected_size_delta,
        expected_raw_sha256=expected_sha256,
        adapter=_RegexByteAdapter(),
        limits=_limits(),
    )
    with pytest.raises(ValueError, match="raw source identity drift"):
        tuple(stream)


def test_stream_rejects_token_ids_outside_signed_int32() -> None:
    class _OutOfRangeAdapter(_RegexByteAdapter):
        def encode_single_piece(self, piece: str) -> list[int]:
            del piece
            return [2**31]

    raw = b"x"
    with pytest.raises(ValueError, match="signed int32"):
        _consume((raw,), raw=raw, adapter=_OutOfRangeAdapter())


@pytest.mark.parametrize("bad_token_id", (-1, 256))
def test_stream_rejects_token_ids_outside_adapter_vocabulary(
    bad_token_id: int,
) -> None:
    class _VocabularyRangeAdapter(_RegexByteAdapter):
        def encode_single_piece(self, piece: str) -> list[int]:
            del piece
            return [bad_token_id]

        def decode_token_bytes(self, token_ids: list[int]) -> bytes:
            del token_ids
            return b"x"

    raw = b"x"
    with pytest.raises(ValueError, match="tokenizer vocabulary"):
        _consume((raw,), raw=raw, adapter=_VocabularyRangeAdapter())


def test_stream_rejects_per_piece_decode_mismatch() -> None:
    class _DecodeMismatchAdapter(_RegexByteAdapter):
        def decode_token_bytes(self, token_ids: list[int]) -> bytes:
            del token_ids
            return b"different"

    raw = b"piece"
    with pytest.raises(ValueError, match="piece decode mismatch"):
        _consume((raw,), raw=raw, adapter=_DecodeMismatchAdapter())


def test_stream_rejects_pathological_unbounded_retained_piece() -> None:
    raw = b"a" * 9
    with pytest.raises(ValueError, match="retained regex piece"):
        _consume(
            (raw,),
            raw=raw,
            limits=_limits(retained_piece_size_bytes=8),
        )


def test_stream_rejects_an_input_chunk_above_its_bound() -> None:
    raw = b"ab"
    with pytest.raises(ValueError, match="raw input chunk"):
        _consume(
            (raw,),
            raw=raw,
            limits=_limits(input_chunk_size_bytes=1),
        )


def test_stream_never_emits_a_payload_chunk_above_its_bound() -> None:
    raw = b"abcdefghij"
    _, chunks, _, _ = _consume(
        (raw,),
        raw=raw,
        limits=_limits(output_chunk_size_bytes=8),
    )
    assert tuple(len(chunk) for chunk in chunks) == (8, 8, 8, 8, 8)


def test_golden_piece_stream_cross_check_rejects_ordinary_divergence() -> None:
    from vfe4.training.tokenizer_stream import (
        verify_piece_stream_golden_vectors,
    )

    class _DivergentAdapter(_RegexByteAdapter):
        def encode_single_piece(self, piece: str) -> list[int]:
            return list(reversed(piece.encode("utf-8")))

        def decode_token_bytes(self, token_ids: list[int]) -> bytes:
            return bytes(reversed(token_ids))

    golden = (("ascii", "golden text", tuple(b"golden text")),)
    with pytest.raises(ValueError, match="golden.*ordinary"):
        verify_piece_stream_golden_vectors(
            adapter=_DivergentAdapter(),
            golden_vectors=golden,
            limits=_limits(),
        )


def test_tokenizer_table_authority_binds_regex_engine_distribution() -> None:
    from vfe4.types.training import production_tokenizer_tables_sha256

    facts = {
        "regex_pattern_sha256": hashlib.sha256(b"pattern").hexdigest(),
        "mergeable_ranks_sha256": hashlib.sha256(b"ranks").hexdigest(),
        "special_tokens_sha256": hashlib.sha256(b"special").hexdigest(),
        "golden_vectors_sha256": hashlib.sha256(b"golden").hexdigest(),
    }
    first = production_tokenizer_tables_sha256(
        **facts,
        regex_engine_distribution_name="regex",
        regex_engine_distribution_version="2026.1.1",
        regex_engine_distribution_record_sha256=hashlib.sha256(
            b"regex-engine-a"
        ).hexdigest(),
    )
    second = production_tokenizer_tables_sha256(
        **facts,
        regex_engine_distribution_name="regex",
        regex_engine_distribution_version="2026.1.1",
        regex_engine_distribution_record_sha256=hashlib.sha256(
            b"regex-engine-b"
        ).hexdigest(),
    )
    assert first != second
    changed_version = production_tokenizer_tables_sha256(
        **facts,
        regex_engine_distribution_name="regex",
        regex_engine_distribution_version="2026.1.2",
        regex_engine_distribution_record_sha256=hashlib.sha256(
            b"regex-engine-a"
        ).hexdigest(),
    )
    assert first != changed_version


def _durable_production_tokenizer_spec():
    from vfe4.types.training import (
        ProductionTokenizerSpec,
        production_tokenizer_tables_sha256,
    )

    facts = {
        "distribution_record_sha256": hashlib.sha256(
            b"tiktoken-record"
        ).hexdigest(),
        "regex_pattern_sha256": hashlib.sha256(b"pattern").hexdigest(),
        "regex_engine_distribution_name": "regex",
        "regex_engine_distribution_version": "2026.1.1",
        "regex_engine_distribution_record_sha256": hashlib.sha256(
            b"regex-record"
        ).hexdigest(),
        "mergeable_ranks_sha256": hashlib.sha256(b"ranks").hexdigest(),
        "special_tokens_sha256": hashlib.sha256(b"special").hexdigest(),
        "golden_vectors_sha256": hashlib.sha256(b"golden").hexdigest(),
    }
    tables_sha256 = production_tokenizer_tables_sha256(
        **{
            name: value
            for name, value in facts.items()
            if name != "distribution_record_sha256"
        }
    )
    return ProductionTokenizerSpec.create_verified(
        **facts,
        tokenizer_tables_sha256=tables_sha256,
    )


def test_durable_tokenizer_spec_recomputes_table_authority_on_reopen() -> None:
    from vfe4.types.training import ProductionTokenizerSpec, owned_sha256

    spec = _durable_production_tokenizer_spec()
    reopened = ProductionTokenizerSpec(
        **json.loads(json.dumps(asdict(spec), sort_keys=True))
    )
    assert reopened == spec

    forged = asdict(spec)
    forged["regex_engine_distribution_version"] = "2026.1.2"
    forged["spec_sha256"] = owned_sha256(
        "vfe4.wt103.production-tokenizer-spec.v1",
        {
            name: value
            for name, value in forged.items()
            if name != "spec_sha256"
        },
    )
    with pytest.raises(ValueError, match="tokenizer table"):
        ProductionTokenizerSpec(**forged)


def test_production_tokenizer_issuance_rejects_forged_adapter_table_hash() -> None:
    from vfe4.training.production import (
        ProductionOperationError,
        _production_tokenizer_spec,
    )

    class _ForgedAdapter:
        distribution_name = "tiktoken"
        distribution_version = "0.12.0"
        encoding_name = "gpt2"
        vocabulary_size = 50_257
        eot_token_id = 50_256
        distribution_record_sha256 = hashlib.sha256(
            b"tiktoken-record"
        ).hexdigest()
        regex_pattern_sha256 = hashlib.sha256(b"pattern").hexdigest()
        regex_engine_distribution_name = "regex"
        regex_engine_distribution_version = "2026.1.1"
        regex_engine_distribution_record_sha256 = hashlib.sha256(
            b"regex-record"
        ).hexdigest()
        mergeable_ranks_sha256 = hashlib.sha256(b"ranks").hexdigest()
        special_tokens_sha256 = hashlib.sha256(b"special").hexdigest()
        golden_vectors_sha256 = hashlib.sha256(b"golden").hexdigest()
        tokenizer_tables_sha256 = "f" * 64

    with pytest.raises(
        ProductionOperationError,
        match="tokenizer table",
    ):
        _production_tokenizer_spec(_ForgedAdapter())


def test_source_lock_tokenizer_cross_link_includes_table_authority() -> None:
    from vfe4.training.production import (
        _tokenizer_authority_cross_links_match,
    )

    spec = _durable_production_tokenizer_spec()
    assert _tokenizer_authority_cross_links_match(
        finalized_spec_sha256=spec.spec_sha256,
        finalized_tables_sha256=spec.tokenizer_tables_sha256,
        tokenizer=spec,
    )
    assert not _tokenizer_authority_cross_links_match(
        finalized_spec_sha256=spec.spec_sha256,
        finalized_tables_sha256="f" * 64,
        tokenizer=spec,
    )


def test_live_adapter_exposes_exact_piece_operations_to_golden_cross_check() -> None:
    from vfe4.training.production import _LiveTokenizer
    from vfe4.training.tokenizer_stream import (
        verify_piece_stream_golden_vectors,
    )

    class _CoreBpe:
        @staticmethod
        def encode_single_piece(piece: bytes) -> list[int]:
            return list(piece)

    class _Encoding:
        _core_bpe = _CoreBpe()

        @staticmethod
        def encode_ordinary(text: str) -> list[int]:
            return list(text.encode())

        @staticmethod
        def decode(token_ids: list[int]) -> str:
            return bytes(token_ids).decode()

        @staticmethod
        def decode_bytes(token_ids: list[int]) -> bytes:
            return bytes(token_ids)

    digest = hashlib.sha256(b"live-adapter-fixture").hexdigest()
    adapter = _LiveTokenizer(
        encoding=_Encoding(),
        distribution_record_sha256=digest,
        regex_pattern_sha256=digest,
        mergeable_ranks_sha256=digest,
        special_tokens_sha256=digest,
        golden_vectors_sha256=digest,
        tokenizer_tables_sha256=digest,
        regex_engine_distribution_name="regex",
        regex_engine_distribution_version="2026.1.1",
        regex_engine_distribution_record_sha256=digest,
        regex_pattern=re.compile(r"[A-Za-z]+|\s+|[^A-Za-z\s]+"),
    )
    golden = (("live-fixture", "one two!", tuple(b"one two!")),)
    verify_piece_stream_golden_vectors(
        adapter=adapter,
        golden_vectors=golden,
        limits=_limits(),
    )


def test_publication_uses_task1_content_addressed_stream_api(
    tmp_path: Path,
) -> None:
    from vfe4.artifacts.durability import WindowsDurabilityBackend
    from vfe4.training.tokenizer_stream import (
        publish_exact_bounded_token_payload,
    )

    raw = "publish 世界\n".encode()
    raw_path = tmp_path / "raw.txt"
    raw_path.write_bytes(raw)
    destination = tmp_path / "cache"
    destination.mkdir()

    publication = publish_exact_bounded_token_payload(
        raw_path=raw_path,
        expected_raw_size_bytes=len(raw),
        expected_raw_sha256=hashlib.sha256(raw).hexdigest(),
        adapter=_RegexByteAdapter(),
        backend=WindowsDurabilityBackend(),
        destination_directory=destination,
        suffix=".int32le",
        limits=_limits(input_chunk_size_bytes=3, output_chunk_size_bytes=8),
    )

    assert publication.path == destination / (
        publication.facts.payload_sha256 + ".int32le"
    )
    assert publication.path.read_bytes() == b"".join(
        struct.pack("<i", value) for value in raw
    )
    assert publication.durable_identity.sha256 == (
        publication.facts.payload_sha256
    )
    assert publication.durable_identity.size_bytes == (
        publication.facts.payload_size_bytes
    )
