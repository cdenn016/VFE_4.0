from __future__ import annotations

import dataclasses
import hashlib
import importlib
import sys
from pathlib import Path

import pytest


class _ByteAdapter:
    distribution_name = "tiktoken"
    distribution_version = "0.12.0"
    encoding_name = "gpt2"
    vocabulary_size = 50_257
    special_tokens = (("<|endoftext|>", 50_256),)
    regex_pattern_sha256 = hashlib.sha256(b"synthetic-regex-v1").hexdigest()
    mergeable_ranks_sha256 = hashlib.sha256(b"synthetic-ranks-v1").hexdigest()
    ordinary_encoding_policy = "encode_ordinary_no_special_tokens"
    fitted_state_sha256 = None
    implementation_sha256 = hashlib.sha256(
        b"tests.wt103.synthetic-byte-adapter.v1"
    ).hexdigest()

    def encode_ordinary(self, text: str) -> tuple[int, ...]:
        return tuple(text.encode("utf-8"))

    def decode(self, token_ids: tuple[int, ...]) -> str:
        return bytes(token_ids).decode("utf-8")


def _contract():
    from vfe4.data.tokenizer import SyntheticTokenizerFixtureContract

    adapter = _ByteAdapter()
    vectors = (
        ("ascii", "Hello, VFE4!\n", adapter.encode_ordinary("Hello, VFE4!\n")),
        ("unicode", "π gauge Δ\n", adapter.encode_ordinary("π gauge Δ\n")),
        ("newlines", "\n\nwiki\r\n", adapter.encode_ordinary("\n\nwiki\r\n")),
    )
    return SyntheticTokenizerFixtureContract.create(
        distribution_name="tiktoken",
        distribution_version="0.12.0",
        encoding_name="gpt2",
        vocabulary_size=50_257,
        special_tokens=(("<|endoftext|>", 50_256),),
        regex_pattern_sha256=adapter.regex_pattern_sha256,
        mergeable_ranks_sha256=adapter.mergeable_ranks_sha256,
        ordinary_encoding_policy="encode_ordinary_no_special_tokens",
        golden_vectors=vectors,
    )


def test_module_is_hermetic_and_never_imports_live_distribution_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocked: list[str] = []
    real_import = importlib.import_module

    def _guard(name: str, package: str | None = None):
        if name == "tiktoken" or name.startswith("importlib.metadata"):
            blocked.append(name)
            raise AssertionError(f"live discovery attempted: {name}")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", _guard)
    module = importlib.import_module("vfe4.data.tokenizer")
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "import tiktoken" not in source
    assert "importlib.metadata" not in source
    assert blocked == []


def test_candidate_contract_and_fixture_spec_are_frozen_and_stable() -> None:
    from vfe4.data import tokenizer as tokenizer_module
    from vfe4.data.tokenizer import build_synthetic_fixture_tokenizer_spec
    from vfe4.types import training as training_types

    contract = _contract()
    spec_a = build_synthetic_fixture_tokenizer_spec(contract, _ByteAdapter())
    spec_b = build_synthetic_fixture_tokenizer_spec(contract, _ByteAdapter())

    assert dataclasses.is_dataclass(contract)
    assert dataclasses.is_dataclass(spec_a)
    assert (
        contract.schema_version
        == "wt103-synthetic-tokenizer-fixture-contract-v1"
    )
    assert (
        spec_a.schema_version
        == "wt103-synthetic-fixture-tokenizer-spec-v1"
    )
    assert spec_a.fixture_sha256 == contract.contract_sha256
    assert spec_a.spec_sha256 == spec_b.spec_sha256
    assert (
        tokenizer_module.CandidateTokenizerContract
        is training_types.CandidateTokenizerContract
    )
    assert (
        tokenizer_module.SyntheticFixtureTokenizerSpec
        is training_types.SyntheticFixtureTokenizerSpec
    )
    assert (
        tokenizer_module.SyntheticFixtureTokenCacheIdentity
        is training_types.SyntheticFixtureTokenCacheIdentity
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec_a.fixture_sha256 = "0" * 64  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("distribution_name", "other"),
        ("distribution_version", "0.11.0"),
        ("encoding_name", "other"),
        ("vocabulary_size", 50_256),
        ("special_tokens", (("<|endoftext|>", 1),)),
        ("regex_pattern_sha256", "1" * 64),
        ("mergeable_ranks_sha256", "2" * 64),
        ("ordinary_encoding_policy", "encode"),
        ("fitted_state_sha256", "3" * 64),
        ("implementation_sha256", "not-a-sha256"),
    ),
)
def test_validator_rejects_each_adapter_contract_mutation(
    field: str, replacement: object
) -> None:
    from vfe4.data.tokenizer import TokenizerContractError, validate_tokenizer_adapter

    class Mutated(_ByteAdapter):
        pass

    setattr(Mutated, field, replacement)
    with pytest.raises(TokenizerContractError, match=field):
        validate_tokenizer_adapter(_contract(), Mutated())


def test_validator_rejects_golden_encode_and_decode_mutations() -> None:
    from vfe4.data.tokenizer import TokenizerContractError, validate_tokenizer_adapter

    class EncodeMutated(_ByteAdapter):
        def encode_ordinary(self, text: str) -> tuple[int, ...]:
            values = super().encode_ordinary(text)
            return values + (1,) if text.startswith("Hello") else values

    class DecodeMutated(_ByteAdapter):
        def decode(self, token_ids: tuple[int, ...]) -> str:
            return super().decode(token_ids) + "x"

    with pytest.raises(TokenizerContractError, match="golden vector"):
        validate_tokenizer_adapter(_contract(), EncodeMutated())
    with pytest.raises(TokenizerContractError, match="round trip"):
        validate_tokenizer_adapter(_contract(), DecodeMutated())


def test_adapter_identity_binds_behavior_beyond_the_golden_vectors() -> None:
    from vfe4.data.tokenizer import build_synthetic_fixture_tokenizer_spec

    class GoldenEquivalentButDivergent(_ByteAdapter):
        def encode_ordinary(self, text: str) -> tuple[int, ...]:
            if text == "outside-the-golden-set":
                return (1, 2, 3)
            return super().encode_ordinary(text)

    contract = _contract()
    baseline = build_synthetic_fixture_tokenizer_spec(contract, _ByteAdapter())
    divergent = build_synthetic_fixture_tokenizer_spec(
        contract,
        GoldenEquivalentButDivergent(),
    )

    assert baseline.adapter_sha256 != divergent.adapter_sha256
    assert baseline.spec_sha256 != divergent.spec_sha256


class _MemoryBackend:
    def __init__(self) -> None:
        self.writes: list[tuple[Path, bytes]] = []

    def publish_bytes(self, path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        self.writes.append((path, payload))


def test_fixture_split_encoding_is_exact_int32_little_endian_and_reopenable(
    tmp_path: Path,
) -> None:
    from vfe4.data.tokenizer import (
        build_synthetic_fixture_tokenizer_spec,
        encode_fixture_split_record,
        issue_fixture_split_capability,
        open_fixture_token_cache,
    )

    raw = "Aπ\n".encode("utf-8")
    raw_sha = hashlib.sha256(raw).hexdigest()
    adapter = _ByteAdapter()
    fixture_contract = _contract()
    spec = build_synthetic_fixture_tokenizer_spec(fixture_contract, adapter)
    backend = _MemoryBackend()
    identity = encode_fixture_split_record(
        split="train",
        raw_bytes=raw,
        raw_parent_sha256=raw_sha,
        spec=spec,
        fixture_contract=fixture_contract,
        adapter=adapter,
        cache_root=tmp_path,
        durability_backend=backend,
        available_disk_bytes=20 * 1024**3,
        available_host_ram_bytes=20 * 1024**3,
        smoke_token_bytes_per_raw_byte=4.0,
        smoke_ram_bytes_per_raw_byte=2.0,
    )

    expected_ids = tuple(raw)
    expected_payload = b"".join(value.to_bytes(4, "little") for value in expected_ids)
    assert backend.writes and backend.writes[-1][1] == expected_payload
    assert identity.authority == "nonproduction_synthetic_fixture"
    assert identity.split == "train"
    assert identity.raw_parent_sha256 == raw_sha
    assert identity.tokenizer == spec
    assert identity.cache_identity.tokenizer == spec
    assert identity.token_count == len(expected_ids)
    assert identity.payload_size_bytes == 4 * len(expected_ids)
    assert identity.payload_sha256 == hashlib.sha256(expected_payload).hexdigest()
    assert identity.minimum_token_id == min(expected_ids)
    assert identity.maximum_token_id == max(expected_ids)

    capability = issue_fixture_split_capability(
        allowed_splits=("train",),
        cache_identities=(identity,),
    )
    assert open_fixture_token_cache(
        identity=identity,
        spec=spec,
        cache_root=tmp_path,
        capability=capability,
    ) == expected_ids


def test_public_encode_api_returns_exact_frozen_task1_identity(
    tmp_path: Path,
) -> None:
    from vfe4.data.tokenizer import (
        SyntheticFixtureTokenCacheIdentity,
        build_synthetic_fixture_tokenizer_spec,
        encode_fixture_split,
    )

    raw = b"canonical identity"
    adapter = _ByteAdapter()
    contract = _contract()
    spec = build_synthetic_fixture_tokenizer_spec(contract, adapter)
    identity = encode_fixture_split(
        split="train",
        raw_bytes=raw,
        raw_parent_sha256=hashlib.sha256(raw).hexdigest(),
        spec=spec,
        fixture_contract=contract,
        adapter=adapter,
        cache_root=tmp_path,
        durability_backend=_MemoryBackend(),
        available_disk_bytes=20 * 1024**3,
        available_host_ram_bytes=20 * 1024**3,
        smoke_token_bytes_per_raw_byte=4.0,
        smoke_ram_bytes_per_raw_byte=2.0,
    )

    assert type(identity) is SyntheticFixtureTokenCacheIdentity
    assert identity.tokenizer is spec


def test_real_durability_backend_provisions_owned_cache_parents(
    tmp_path: Path,
) -> None:
    from vfe4.artifacts.durability import (
        PosixDurabilityBackend,
        WindowsDurabilityBackend,
    )
    from vfe4.data.tokenizer import (
        build_synthetic_fixture_tokenizer_spec,
        encode_fixture_split_record,
    )

    raw = b"real durability"
    adapter = _ByteAdapter()
    contract = _contract()
    spec = build_synthetic_fixture_tokenizer_spec(contract, adapter)
    backend = (
        WindowsDurabilityBackend()
        if sys.platform == "win32"
        else PosixDurabilityBackend()
    )
    record = encode_fixture_split_record(
        split="validation",
        raw_bytes=raw,
        raw_parent_sha256=hashlib.sha256(raw).hexdigest(),
        spec=spec,
        fixture_contract=contract,
        adapter=adapter,
        cache_root=tmp_path,
        durability_backend=backend,
        available_disk_bytes=20 * 1024**3,
        available_host_ram_bytes=20 * 1024**3,
        smoke_token_bytes_per_raw_byte=4.0,
        smoke_ram_bytes_per_raw_byte=2.0,
    )

    target = tmp_path / record.cache_relative_path
    assert target.is_file()
    assert hashlib.sha256(target.read_bytes()).hexdigest() == record.payload_sha256


def test_cache_path_rejects_in_root_symlink_escape(tmp_path: Path) -> None:
    from vfe4.data.tokenizer import (
        TokenizerContractError,
        build_synthetic_fixture_tokenizer_spec,
        encode_fixture_split_record,
    )

    destination = tmp_path / "V3_Transformer"
    destination.mkdir()
    link = tmp_path / "synthetic-fixture"
    try:
        link.symlink_to(destination, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable on this platform: {exc}")

    raw = b"quarantined"
    adapter = _ByteAdapter()
    contract = _contract()
    spec = build_synthetic_fixture_tokenizer_spec(contract, adapter)
    with pytest.raises(
        TokenizerContractError,
        match="nonlink|symlink|junction|reparse|V3",
    ):
        encode_fixture_split_record(
            split="train",
            raw_bytes=raw,
            raw_parent_sha256=hashlib.sha256(raw).hexdigest(),
            spec=spec,
            fixture_contract=contract,
            adapter=adapter,
            cache_root=tmp_path,
            durability_backend=_MemoryBackend(),
            available_disk_bytes=20 * 1024**3,
            available_host_ram_bytes=20 * 1024**3,
            smoke_token_bytes_per_raw_byte=4.0,
            smoke_ram_bytes_per_raw_byte=2.0,
        )


def test_split_parentage_and_test_bytes_remain_sealed(tmp_path: Path) -> None:
    from vfe4.data.tokenizer import (
        TokenizerContractError,
        build_synthetic_fixture_tokenizer_spec,
        encode_fixture_split_record,
        issue_fixture_split_capability,
        open_fixture_token_cache,
    )

    adapter = _ByteAdapter()
    fixture_contract = _contract()
    spec = build_synthetic_fixture_tokenizer_spec(fixture_contract, adapter)
    backend = _MemoryBackend()
    raw = b"same text"
    identities = tuple(
        encode_fixture_split_record(
            split=split,
            raw_bytes=raw,
            raw_parent_sha256=hashlib.sha256(split.encode() + raw).hexdigest(),
            spec=spec,
            fixture_contract=fixture_contract,
            adapter=adapter,
            cache_root=tmp_path,
            durability_backend=backend,
            available_disk_bytes=20 * 1024**3,
            available_host_ram_bytes=20 * 1024**3,
            smoke_token_bytes_per_raw_byte=4.0,
            smoke_ram_bytes_per_raw_byte=2.0,
        )
        for split in ("train", "validation", "test")
    )
    assert len({identity.record_sha256 for identity in identities}) == 3
    assert len({identity.cache_relative_path for identity in identities}) == 3

    capability = issue_fixture_split_capability(
        allowed_splits=("train", "validation"),
        cache_identities=identities,
    )
    with pytest.raises(TokenizerContractError, match="test"):
        open_fixture_token_cache(
            identity=identities[2],
            spec=spec,
            cache_root=tmp_path,
            capability=capability,
        )


def test_cache_reopen_rejects_payload_manifest_and_provenance_mismatch(
    tmp_path: Path,
) -> None:
    from vfe4.data.tokenizer import (
        TokenizerContractError,
        build_synthetic_fixture_tokenizer_spec,
        encode_fixture_split_record,
        issue_fixture_split_capability,
        open_fixture_token_cache,
    )

    adapter = _ByteAdapter()
    fixture_contract = _contract()
    spec = build_synthetic_fixture_tokenizer_spec(fixture_contract, adapter)
    identity = encode_fixture_split_record(
        split="validation",
        raw_bytes=b"validation",
        raw_parent_sha256=hashlib.sha256(b"validation-parent").hexdigest(),
        spec=spec,
        fixture_contract=fixture_contract,
        adapter=adapter,
        cache_root=tmp_path,
        durability_backend=_MemoryBackend(),
        available_disk_bytes=20 * 1024**3,
        available_host_ram_bytes=20 * 1024**3,
        smoke_token_bytes_per_raw_byte=4.0,
        smoke_ram_bytes_per_raw_byte=2.0,
    )
    capability = issue_fixture_split_capability(
        allowed_splits=("validation",),
        cache_identities=(identity,),
    )
    cache_path = tmp_path / identity.cache_relative_path
    cache_path.write_bytes(cache_path.read_bytes() + b"\x00")
    with pytest.raises(TokenizerContractError, match="size|hash"):
        open_fixture_token_cache(
            identity=identity,
            spec=spec,
            cache_root=tmp_path,
            capability=capability,
        )

    with pytest.raises(TokenizerContractError, match="identity|path|V3|escapes"):
        mutated = dataclasses.replace(
            identity, cache_relative_path="../V3_Transformer/cache.bin"
        )
        open_fixture_token_cache(
            identity=mutated,
            spec=spec,
            cache_root=tmp_path,
            capability=capability,
        )


def test_encoding_rejects_invalid_utf8_out_of_range_ids_and_nonroundtrip(
    tmp_path: Path,
) -> None:
    from vfe4.data.tokenizer import (
        TokenizerContractError,
        build_synthetic_fixture_tokenizer_spec,
        encode_fixture_split_record,
    )

    adapter = _ByteAdapter()
    fixture_contract = _contract()
    spec = build_synthetic_fixture_tokenizer_spec(fixture_contract, adapter)
    common = dict(
        split="train",
        raw_parent_sha256="a" * 64,
        spec=spec,
        fixture_contract=fixture_contract,
        cache_root=tmp_path,
        durability_backend=_MemoryBackend(),
        available_disk_bytes=20 * 1024**3,
        available_host_ram_bytes=20 * 1024**3,
        smoke_token_bytes_per_raw_byte=4.0,
        smoke_ram_bytes_per_raw_byte=2.0,
    )
    with pytest.raises(TokenizerContractError, match="UTF-8"):
        encode_fixture_split_record(raw_bytes=b"\xff", adapter=adapter, **common)

    class OutOfRange(_ByteAdapter):
        def encode_ordinary(self, text: str) -> tuple[int, ...]:
            if text == "x":
                return (50_257,)
            return super().encode_ordinary(text)

    class NotRoundTrip(_ByteAdapter):
        def encode_ordinary(self, text: str) -> tuple[int, ...]:
            if text == "x":
                return (65,)
            return super().encode_ordinary(text)

    out_of_range = OutOfRange()
    with pytest.raises(TokenizerContractError, match="token ID"):
        encode_fixture_split_record(
            raw_bytes=b"x",
            adapter=out_of_range,
            **common
            | {
                "spec": build_synthetic_fixture_tokenizer_spec(
                    fixture_contract,
                    out_of_range,
                )
            },
        )
    not_round_trip = NotRoundTrip()
    with pytest.raises(TokenizerContractError, match="round trip"):
        encode_fixture_split_record(
            raw_bytes=b"x",
            adapter=not_round_trip,
            **common
            | {
                "spec": build_synthetic_fixture_tokenizer_spec(
                    fixture_contract,
                    not_round_trip,
                )
            },
        )


def test_preprocessing_forecast_enforces_disk_and_host_ram_before_encode(
    tmp_path: Path,
) -> None:
    from vfe4.data.tokenizer import (
        TokenizerContractError,
        build_synthetic_fixture_tokenizer_spec,
        encode_fixture_split_record,
        forecast_preprocessing_resources,
    )

    forecast = forecast_preprocessing_resources(
        raw_size_bytes=100,
        smoke_token_bytes_per_raw_byte=4.0,
        smoke_ram_bytes_per_raw_byte=2.0,
    )
    assert forecast.token_payload_bytes == 400
    assert forecast.peak_ram_bytes == 200
    assert forecast.required_disk_bytes == 2 * 400 + 10 * 1024**3

    adapter = _ByteAdapter()
    fixture_contract = _contract()
    spec = build_synthetic_fixture_tokenizer_spec(fixture_contract, adapter)
    with pytest.raises(TokenizerContractError, match="disk"):
        encode_fixture_split_record(
            split="train",
            raw_bytes=b"x",
            raw_parent_sha256="b" * 64,
            spec=spec,
            fixture_contract=fixture_contract,
            adapter=adapter,
            cache_root=tmp_path,
            durability_backend=_MemoryBackend(),
            available_disk_bytes=10,
            available_host_ram_bytes=20 * 1024**3,
            smoke_token_bytes_per_raw_byte=4.0,
            smoke_ram_bytes_per_raw_byte=2.0,
        )
    with pytest.raises(TokenizerContractError, match="host RAM"):
        encode_fixture_split_record(
            split="train",
            raw_bytes=b"x",
            raw_parent_sha256="b" * 64,
            spec=spec,
            fixture_contract=fixture_contract,
            adapter=adapter,
            cache_root=tmp_path,
            durability_backend=_MemoryBackend(),
            available_disk_bytes=20 * 1024**3,
            available_host_ram_bytes=1,
            smoke_token_bytes_per_raw_byte=4.0,
            smoke_ram_bytes_per_raw_byte=2.0,
        )
