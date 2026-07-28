"""Hermetic candidate-tokenizer validation and synthetic fixture caches.

This module intentionally has no production tokenizer adapter.  In particular,
it never imports ``tiktoken`` or distribution metadata.  Tasks before the
separately authorized source-lock operation can validate only injected,
nonproduction adapters and can create only synthetic-fixture identities.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import stat
from collections.abc import Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import CodeType
from typing import Literal, Protocol, runtime_checkable

from vfe4.types.training import (
    CandidateTokenizerContract,
    SyntheticFixtureTokenCacheIdentity,
    SyntheticFixtureTokenizerSpec,
)


SplitName = Literal["train", "validation", "test"]
ReadableFixtureSplit = Literal["train", "validation"]

_FIXTURE_CONTRACT_DOMAIN = (
    b"VFE4-WT103-SYNTHETIC-TOKENIZER-FIXTURE-CONTRACT-V1\x00"
)
_ADAPTER_OBSERVATION_DOMAIN = (
    b"VFE4-WT103-SYNTHETIC-TOKENIZER-ADAPTER-OBSERVATION-V1\x00"
)
_FIXTURE_CACHE_DOMAIN = b"VFE4-WT103-SYNTHETIC-TOKEN-CACHE-V1\x00"
_FIXTURE_CAPABILITY_DOMAIN = b"VFE4-WT103-SYNTHETIC-SPLIT-CAPABILITY-V1\x00"
_TEN_GIB = 10 * 1024**3


class TokenizerContractError(ValueError):
    """An injected tokenizer or fixture cache violated the frozen contract."""


def _canonical_json_bytes(value: object) -> bytes:
    def convert(item: object) -> object:
        if dataclasses.is_dataclass(item) and not isinstance(item, type):
            return {
                field.name: convert(getattr(item, field.name))
                for field in dataclasses.fields(item)
            }
        if isinstance(item, tuple):
            return [convert(value) for value in item]
        if item is None or type(item) in (str, bool, int):
            return item
        if type(item) is float:
            if not math.isfinite(item):
                raise TokenizerContractError("canonical values must be finite")
            return item
        raise TokenizerContractError(
            f"unsupported canonical value type: {type(item).__name__}"
        )

    return json.dumps(
        convert(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TokenizerContractError(f"{field} must be a lowercase SHA-256")
    return value


def _require_plain_string(value: object, *, field: str) -> str:
    if type(value) is not str or not value:
        raise TokenizerContractError(f"{field} must be a nonempty plain string")
    return value


def _require_split(value: object) -> SplitName:
    if value not in ("train", "validation", "test"):
        raise TokenizerContractError("split must be train, validation, or test")
    return value  # type: ignore[return-value]


def _exact_special_tokens(value: object) -> tuple[tuple[str, int], ...]:
    if type(value) is not tuple:
        raise TokenizerContractError("special_tokens must be an immutable tuple")
    rows: list[tuple[str, int]] = []
    seen_names: set[str] = set()
    seen_ids: set[int] = set()
    for row in value:
        if (
            type(row) is not tuple
            or len(row) != 2
            or type(row[0]) is not str
            or not row[0]
            or type(row[1]) is not int
            or row[1] < 0
        ):
            raise TokenizerContractError(
                "special_tokens rows must be exact (nonempty str, nonnegative int)"
            )
        if row[0] in seen_names or row[1] in seen_ids:
            raise TokenizerContractError("special_tokens must be unique")
        seen_names.add(row[0])
        seen_ids.add(row[1])
        rows.append(row)
    return tuple(rows)


def _exact_golden_vectors(
    value: object,
    *,
    vocabulary_size: int,
) -> tuple[tuple[str, str, tuple[int, ...]], ...]:
    if type(value) is not tuple or not value:
        raise TokenizerContractError("golden_vectors must be a nonempty tuple")
    result: list[tuple[str, str, tuple[int, ...]]] = []
    names: set[str] = set()
    for row in value:
        if (
            type(row) is not tuple
            or len(row) != 3
            or type(row[0]) is not str
            or not row[0]
            or type(row[1]) is not str
            or type(row[2]) is not tuple
            or not row[2]
        ):
            raise TokenizerContractError(
                "golden_vectors rows must be exact (name, text, token tuple)"
            )
        if row[0] in names:
            raise TokenizerContractError("golden vector names must be unique")
        names.add(row[0])
        if any(
            type(token_id) is not int
            or token_id < 0
            or token_id >= vocabulary_size
            for token_id in row[2]
        ):
            raise TokenizerContractError(
                f"golden vector {row[0]!r} contains an invalid token ID"
            )
        result.append((row[0], row[1], row[2]))
    return tuple(result)


@runtime_checkable
class TokenizerDistributionAdapter(Protocol):
    """Pure observation/encoding seam supplied by a test or source-lock owner."""

    distribution_name: str
    distribution_version: str
    encoding_name: str
    vocabulary_size: int
    special_tokens: tuple[tuple[str, int], ...]
    regex_pattern_sha256: str
    mergeable_ranks_sha256: str
    ordinary_encoding_policy: str
    fitted_state_sha256: str | None
    implementation_sha256: str

    def encode_ordinary(self, text: str) -> Sequence[int]:
        """Encode a complete text with no special-token interpretation."""

    def decode(self, token_ids: tuple[int, ...]) -> str:
        """Decode an exact token sequence."""


@dataclasses.dataclass(frozen=True, slots=True)
class SyntheticTokenizerFixtureContract:
    """Synthetic observations used only to test the candidate contract."""

    schema_version: str
    candidate: CandidateTokenizerContract
    distribution_name: str
    distribution_version: str
    encoding_name: str
    vocabulary_size: int
    special_tokens: tuple[tuple[str, int], ...]
    regex_pattern_sha256: str
    mergeable_ranks_sha256: str
    ordinary_encoding_policy: str
    golden_vectors: tuple[tuple[str, str, tuple[int, ...]], ...]
    contract_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != "wt103-synthetic-tokenizer-fixture-contract-v1":
            raise TokenizerContractError("unsupported synthetic fixture schema")
        if type(self.candidate) is not CandidateTokenizerContract:
            raise TokenizerContractError(
                "candidate must be the exact three-string candidate contract"
            )
        self.candidate.__post_init__()
        _require_plain_string(self.distribution_name, field="distribution_name")
        _require_plain_string(self.distribution_version, field="distribution_version")
        _require_plain_string(self.encoding_name, field="encoding_name")
        if (
            self.distribution_name,
            self.distribution_version,
            self.encoding_name,
        ) != (
            self.candidate.distribution,
            self.candidate.version,
            self.candidate.encoding_name,
        ):
            raise TokenizerContractError(
                "synthetic fixture strings do not match the candidate contract"
            )
        if type(self.vocabulary_size) is not int or self.vocabulary_size <= 0:
            raise TokenizerContractError("vocabulary_size must be a positive integer")
        special_tokens = _exact_special_tokens(self.special_tokens)
        if any(token_id >= self.vocabulary_size for _, token_id in special_tokens):
            raise TokenizerContractError(
                "special_tokens contains an out-of-vocabulary token ID"
            )
        _require_sha256(self.regex_pattern_sha256, field="regex_pattern_sha256")
        _require_sha256(
            self.mergeable_ranks_sha256, field="mergeable_ranks_sha256"
        )
        _require_plain_string(
            self.ordinary_encoding_policy, field="ordinary_encoding_policy"
        )
        _exact_golden_vectors(
            self.golden_vectors, vocabulary_size=self.vocabulary_size
        )
        expected = _digest(
            _FIXTURE_CONTRACT_DOMAIN,
            (
                self.schema_version,
                self.candidate,
                self.distribution_name,
                self.distribution_version,
                self.encoding_name,
                self.vocabulary_size,
                self.special_tokens,
                self.regex_pattern_sha256,
                self.mergeable_ranks_sha256,
                self.ordinary_encoding_policy,
                self.golden_vectors,
            ),
        )
        if self.contract_sha256 != expected:
            raise TokenizerContractError("contract_sha256 does not match the contract")

    @classmethod
    def create(
        cls,
        *,
        distribution_name: str,
        distribution_version: str,
        encoding_name: str,
        vocabulary_size: int,
        special_tokens: tuple[tuple[str, int], ...],
        regex_pattern_sha256: str,
        mergeable_ranks_sha256: str,
        ordinary_encoding_policy: str,
        golden_vectors: tuple[tuple[str, str, tuple[int, ...]], ...],
        candidate: CandidateTokenizerContract | None = None,
    ) -> "SyntheticTokenizerFixtureContract":
        exact_candidate = (
            CandidateTokenizerContract() if candidate is None else candidate
        )
        values = (
            "wt103-synthetic-tokenizer-fixture-contract-v1",
            exact_candidate,
            distribution_name,
            distribution_version,
            encoding_name,
            vocabulary_size,
            special_tokens,
            regex_pattern_sha256,
            mergeable_ranks_sha256,
            ordinary_encoding_policy,
            golden_vectors,
        )
        return cls(*values, _digest(_FIXTURE_CONTRACT_DOMAIN, values))


def validate_tokenizer_adapter(
    contract: SyntheticTokenizerFixtureContract,
    adapter: TokenizerDistributionAdapter,
) -> None:
    """Validate only injected observations and synthetic golden vectors."""

    if type(contract) is not SyntheticTokenizerFixtureContract:
        raise TokenizerContractError(
            "contract must be an exact SyntheticTokenizerFixtureContract"
        )
    contract.__post_init__()
    fields = (
        "distribution_name",
        "distribution_version",
        "encoding_name",
        "vocabulary_size",
        "special_tokens",
        "regex_pattern_sha256",
        "mergeable_ranks_sha256",
        "ordinary_encoding_policy",
    )
    for field in fields:
        try:
            observed = getattr(adapter, field)
        except (AttributeError, RuntimeError) as exc:
            raise TokenizerContractError(f"adapter is missing {field}") from exc
        expected = getattr(contract, field)
        if type(observed) is not type(expected) or observed != expected:
            raise TokenizerContractError(
                f"{field} does not match the candidate contract"
            )
    try:
        fitted_state = adapter.fitted_state_sha256
    except (AttributeError, RuntimeError) as exc:
        raise TokenizerContractError(
            "adapter is missing fitted_state_sha256"
        ) from exc
    if fitted_state is not None:
        raise TokenizerContractError(
            "fitted_state_sha256 must be absent; fitted state is forbidden"
        )
    try:
        implementation_sha256 = adapter.implementation_sha256
    except (AttributeError, RuntimeError) as exc:
        raise TokenizerContractError(
            "adapter is missing implementation_sha256"
        ) from exc
    _require_sha256(
        implementation_sha256,
        field="implementation_sha256",
    )
    if not callable(getattr(adapter, "encode_ordinary", None)):
        raise TokenizerContractError("adapter encode_ordinary must be callable")
    if not callable(getattr(adapter, "decode", None)):
        raise TokenizerContractError("adapter decode must be callable")
    for name, text, expected_ids in contract.golden_vectors:
        try:
            encoded = tuple(adapter.encode_ordinary(text))
        except Exception as exc:
            raise TokenizerContractError(
                f"golden vector {name!r} encoding failed"
            ) from exc
        if encoded != expected_ids:
            raise TokenizerContractError(
                f"golden vector {name!r} does not match the candidate contract"
            )
        if any(type(token_id) is not int for token_id in encoded):
            raise TokenizerContractError(
                f"golden vector {name!r} returned a noninteger token ID"
            )
        try:
            decoded = adapter.decode(encoded)
        except Exception as exc:
            raise TokenizerContractError(
                f"golden vector {name!r} round trip failed"
            ) from exc
        if type(decoded) is not str or decoded != text:
            raise TokenizerContractError(
                f"golden vector {name!r} round trip does not match"
            )


def _code_constant_observation(value: object) -> tuple[object, ...]:
    if value is None:
        return ("none",)
    if type(value) in (str, bool, int):
        return (type(value).__name__, value)
    if type(value) is float:
        if not math.isfinite(value):
            raise TokenizerContractError(
                "adapter implementation contains a nonfinite constant"
            )
        return ("float", value)
    if type(value) is bytes:
        return ("bytes", value.hex())
    if type(value) is tuple:
        return (
            "tuple",
            tuple(_code_constant_observation(item) for item in value),
        )
    if isinstance(value, CodeType):
        return ("code", _code_observation(value))
    if value is Ellipsis:
        return ("ellipsis",)
    raise TokenizerContractError(
        "adapter implementation contains an unsupported code constant "
        f"{type(value).__name__}"
    )


def _code_observation(code: CodeType) -> tuple[object, ...]:
    """Describe executable semantics without source paths or line numbers."""

    return (
        code.co_argcount,
        code.co_posonlyargcount,
        code.co_kwonlyargcount,
        code.co_nlocals,
        code.co_flags,
        hashlib.sha256(code.co_code).hexdigest(),
        tuple(_code_constant_observation(item) for item in code.co_consts),
        tuple(code.co_names),
        tuple(code.co_varnames),
        tuple(code.co_freevars),
        tuple(code.co_cellvars),
    )


def _callable_observation(value: object, *, field: str) -> tuple[object, ...]:
    function = getattr(value, "__func__", value)
    code = getattr(function, "__code__", None)
    if code is None:
        return ("declared_implementation_only", field)
    if not isinstance(code, CodeType):
        raise TokenizerContractError(
            f"{field} exposes a malformed Python code object"
        )
    defaults = getattr(function, "__defaults__", None)
    keyword_defaults = getattr(function, "__kwdefaults__", None)
    if keyword_defaults:
        keyword_rows = tuple(
            (key, _code_constant_observation(item))
            for key, item in sorted(keyword_defaults.items())
        )
    else:
        keyword_rows = ()
    return (
        "python_code_v1",
        field,
        _code_observation(code),
        _code_constant_observation(defaults),
        keyword_rows,
    )


def _adapter_observation(adapter: TokenizerDistributionAdapter) -> tuple[object, ...]:
    """Bind the declared source digest and observable Python callables.

    This is a bounded synthetic-fixture observation, not a proof of extensional
    equality for an arbitrary tokenizer implementation.  Task 13 separately
    authenticates the installed production distribution and tables.
    """

    return (
        adapter.implementation_sha256,
        _callable_observation(
            adapter.encode_ordinary,
            field="encode_ordinary",
        ),
        _callable_observation(adapter.decode, field="decode"),
    )


def build_synthetic_fixture_tokenizer_spec(
    contract: SyntheticTokenizerFixtureContract,
    adapter: TokenizerDistributionAdapter,
) -> SyntheticFixtureTokenizerSpec:
    """Create the only tokenizer identity available before source lock."""

    validate_tokenizer_adapter(contract, adapter)
    adapter_sha256 = _digest(
        _ADAPTER_OBSERVATION_DOMAIN,
        (
            contract.distribution_name,
            contract.distribution_version,
            contract.encoding_name,
            contract.vocabulary_size,
            contract.special_tokens,
            contract.regex_pattern_sha256,
            contract.mergeable_ranks_sha256,
            contract.ordinary_encoding_policy,
            contract.golden_vectors,
            None,
            _adapter_observation(adapter),
        ),
    )
    return SyntheticFixtureTokenizerSpec.create(
        adapter_sha256=adapter_sha256,
        fixture_sha256=contract.contract_sha256,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class PreprocessingResourceForecast:
    schema_version: str
    raw_size_bytes: int
    smoke_token_bytes_per_raw_byte: float
    smoke_ram_bytes_per_raw_byte: float
    token_payload_bytes: int
    peak_ram_bytes: int
    required_disk_bytes: int

    def __post_init__(self) -> None:
        if self.schema_version != "wt103-preprocessing-forecast-v1":
            raise TokenizerContractError("unsupported preprocessing forecast schema")
        if type(self.raw_size_bytes) is not int or self.raw_size_bytes <= 0:
            raise TokenizerContractError("raw_size_bytes must be positive")
        for field in (
            "smoke_token_bytes_per_raw_byte",
            "smoke_ram_bytes_per_raw_byte",
        ):
            value = getattr(self, field)
            if type(value) is not float or not math.isfinite(value) or value <= 0:
                raise TokenizerContractError(f"{field} must be a positive finite float")
        expected_payload = math.ceil(
            self.raw_size_bytes * self.smoke_token_bytes_per_raw_byte
        )
        expected_ram = math.ceil(
            self.raw_size_bytes * self.smoke_ram_bytes_per_raw_byte
        )
        if self.token_payload_bytes != expected_payload:
            raise TokenizerContractError("token_payload_bytes is not derived")
        if self.peak_ram_bytes != expected_ram:
            raise TokenizerContractError("peak_ram_bytes is not derived")
        if self.required_disk_bytes != 2 * expected_payload + _TEN_GIB:
            raise TokenizerContractError("required_disk_bytes is not derived")


def forecast_preprocessing_resources(
    *,
    raw_size_bytes: int,
    smoke_token_bytes_per_raw_byte: float,
    smoke_ram_bytes_per_raw_byte: float,
) -> PreprocessingResourceForecast:
    if type(raw_size_bytes) is not int or raw_size_bytes <= 0:
        raise TokenizerContractError("raw_size_bytes must be positive")
    if (
        type(smoke_token_bytes_per_raw_byte) is not float
        or not math.isfinite(smoke_token_bytes_per_raw_byte)
        or smoke_token_bytes_per_raw_byte <= 0
    ):
        raise TokenizerContractError(
            "smoke_token_bytes_per_raw_byte must be a positive finite float"
        )
    if (
        type(smoke_ram_bytes_per_raw_byte) is not float
        or not math.isfinite(smoke_ram_bytes_per_raw_byte)
        or smoke_ram_bytes_per_raw_byte <= 0
    ):
        raise TokenizerContractError(
            "smoke_ram_bytes_per_raw_byte must be a positive finite float"
        )
    payload = math.ceil(raw_size_bytes * smoke_token_bytes_per_raw_byte)
    peak_ram = math.ceil(raw_size_bytes * smoke_ram_bytes_per_raw_byte)
    return PreprocessingResourceForecast(
        schema_version="wt103-preprocessing-forecast-v1",
        raw_size_bytes=raw_size_bytes,
        smoke_token_bytes_per_raw_byte=smoke_token_bytes_per_raw_byte,
        smoke_ram_bytes_per_raw_byte=smoke_ram_bytes_per_raw_byte,
        token_payload_bytes=payload,
        peak_ram_bytes=peak_ram,
        required_disk_bytes=2 * payload + _TEN_GIB,
    )


def _preflight_resources(
    forecast: PreprocessingResourceForecast,
    *,
    available_disk_bytes: int,
    available_host_ram_bytes: int,
) -> None:
    forecast.__post_init__()
    if (
        type(available_disk_bytes) is not int
        or available_disk_bytes < forecast.required_disk_bytes
    ):
        raise TokenizerContractError(
            "insufficient disk for the fixed full-split preprocessing forecast"
        )
    if (
        type(available_host_ram_bytes) is not int
        or available_host_ram_bytes < forecast.peak_ram_bytes
    ):
        raise TokenizerContractError(
            "insufficient host RAM for the fixed full-split preprocessing forecast"
        )


def _cache_relative_path(
    *, split: SplitName, payload_sha256: str
) -> str:
    return f"synthetic-fixture/{split}/{payload_sha256}.int32le"


def _validate_relative_cache_path(path: object, *, split: SplitName) -> str:
    if type(path) is not str or not path or "\\" in path:
        raise TokenizerContractError("cache_relative_path must be canonical POSIX")
    posix = PurePosixPath(path)
    windows = PureWindowsPath(path)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or bool(windows.root)
        or posix.as_posix() != path
        or any(part in ("", ".", "..") for part in posix.parts)
        or len(posix.parts) != 3
        or posix.parts[0] != "synthetic-fixture"
        or posix.parts[1] != split
        or posix.suffix != ".int32le"
    ):
        raise TokenizerContractError(
            "cache_relative_path is noncanonical or escapes its split"
        )
    if any("v3_transformer" in part.casefold() for part in posix.parts):
        raise TokenizerContractError("V3 cache provenance is forbidden")
    return path


@dataclasses.dataclass(frozen=True, slots=True)
class SyntheticFixtureTokenCacheRecord:
    """Rich split record around the canonical readiness-ineligible identity."""

    schema_version: str
    authority: str
    split: SplitName
    raw_parent_sha256: str
    fixture_contract: SyntheticTokenizerFixtureContract
    tokenizer: SyntheticFixtureTokenizerSpec
    cache_identity: SyntheticFixtureTokenCacheIdentity
    token_count: int
    minimum_token_id: int
    maximum_token_id: int
    payload_size_bytes: int
    payload_sha256: str
    cache_relative_path: str
    record_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != "wt103-synthetic-token-cache-record-v1":
            raise TokenizerContractError("unsupported synthetic cache schema")
        if self.authority != "nonproduction_synthetic_fixture":
            raise TokenizerContractError("synthetic cache authority cannot be promoted")
        split = _require_split(self.split)
        _require_sha256(self.raw_parent_sha256, field="raw_parent_sha256")
        if type(self.fixture_contract) is not SyntheticTokenizerFixtureContract:
            raise TokenizerContractError(
                "fixture_contract must be the exact rich synthetic contract"
            )
        self.fixture_contract.__post_init__()
        if type(self.tokenizer) is not SyntheticFixtureTokenizerSpec:
            raise TokenizerContractError(
                "tokenizer must be the exact canonical synthetic fixture spec"
            )
        self.tokenizer.__post_init__()
        if self.tokenizer.fixture_sha256 != self.fixture_contract.contract_sha256:
            raise TokenizerContractError(
                "canonical tokenizer spec does not bind the fixture contract"
            )
        if type(self.cache_identity) is not SyntheticFixtureTokenCacheIdentity:
            raise TokenizerContractError(
                "cache_identity must be the exact canonical synthetic identity"
            )
        self.cache_identity.__post_init__()
        if type(self.token_count) is not int or self.token_count <= 0:
            raise TokenizerContractError("token_count must be positive")
        if (
            type(self.minimum_token_id) is not int
            or type(self.maximum_token_id) is not int
            or self.minimum_token_id < 0
            or self.maximum_token_id < self.minimum_token_id
        ):
            raise TokenizerContractError("token ID bounds are invalid")
        if self.payload_size_bytes != 4 * self.token_count:
            raise TokenizerContractError(
                "payload_size_bytes must be four bytes per token"
            )
        _require_sha256(self.payload_sha256, field="payload_sha256")
        if (
            self.cache_identity.tokenizer != self.tokenizer
            or self.cache_identity.payload_sha256 != self.payload_sha256
        ):
            raise TokenizerContractError(
                "canonical cache identity does not bind this tokenizer/payload"
            )
        _validate_relative_cache_path(self.cache_relative_path, split=split)
        expected = _digest(
            _FIXTURE_CACHE_DOMAIN,
            (
                self.schema_version,
                self.authority,
                self.split,
                self.raw_parent_sha256,
                self.fixture_contract,
                self.tokenizer,
                self.cache_identity,
                self.token_count,
                self.minimum_token_id,
                self.maximum_token_id,
                self.payload_size_bytes,
                self.payload_sha256,
                self.cache_relative_path,
            ),
        )
        if self.record_sha256 != expected:
            raise TokenizerContractError(
                "record_sha256 does not match the fixture token cache"
            )

    @classmethod
    def create(
        cls,
        *,
        split: SplitName,
        raw_parent_sha256: str,
        fixture_contract: SyntheticTokenizerFixtureContract,
        tokenizer: SyntheticFixtureTokenizerSpec,
        token_ids: tuple[int, ...],
        payload_sha256: str,
    ) -> "SyntheticFixtureTokenCacheRecord":
        split = _require_split(split)
        if type(fixture_contract) is not SyntheticTokenizerFixtureContract:
            raise TokenizerContractError(
                "fixture_contract must be the exact rich synthetic contract"
            )
        fixture_contract.__post_init__()
        if type(tokenizer) is not SyntheticFixtureTokenizerSpec:
            raise TokenizerContractError(
                "tokenizer must be the exact canonical synthetic fixture spec"
            )
        tokenizer.__post_init__()
        if not token_ids:
            raise TokenizerContractError("fixture token cache cannot be empty")
        relative = _cache_relative_path(
            split=split, payload_sha256=payload_sha256
        )
        values = (
            "wt103-synthetic-token-cache-record-v1",
            "nonproduction_synthetic_fixture",
            split,
            raw_parent_sha256,
            fixture_contract,
            tokenizer,
            SyntheticFixtureTokenCacheIdentity.create(
                tokenizer=tokenizer,
                payload_sha256=payload_sha256,
            ),
            len(token_ids),
            min(token_ids),
            max(token_ids),
            4 * len(token_ids),
            payload_sha256,
            relative,
        )
        return cls(*values, _digest(_FIXTURE_CACHE_DOMAIN, values))


class FixtureDurabilityBackend(Protocol):
    """Small write seam implemented by the generic Task 2 backend."""

    def publish_bytes(self, path: Path, payload: bytes) -> None:
        """Durably publish ``payload`` at ``path`` and return only on reopen."""


def _is_redirect_or_reparse(path: Path, status: os.stat_result) -> bool:
    if stat.S_ISLNK(status.st_mode):
        return True
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if bool(getattr(status, "st_file_attributes", 0) & reparse):
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction is not None and is_junction())


def _require_regular_nonlink_directory(path: Path) -> None:
    try:
        status = path.lstat()
    except OSError as exc:
        raise TokenizerContractError(
            f"cache directory is unavailable: {path}: {exc}"
        ) from exc
    if not stat.S_ISDIR(status.st_mode) or _is_redirect_or_reparse(path, status):
        raise TokenizerContractError(
            f"cache directory must be a regular nonlink directory: {path}"
        )


def _same_lexical_path(left: Path, right: Path) -> bool:
    return os.path.normcase(os.path.normpath(os.fspath(left))) == os.path.normcase(
        os.path.normpath(os.fspath(right))
    )


def _prepare_owned_subdirectories(root: Path, relative_parent: PurePosixPath) -> None:
    current = root
    for part in relative_parent.parts:
        current = current / part
        try:
            status = current.lstat()
        except FileNotFoundError:
            try:
                current.mkdir()
            except FileExistsError:
                pass
            except OSError as exc:
                raise TokenizerContractError(
                    f"cache directory creation failed: {current}: {exc}"
                ) from exc
            _require_regular_nonlink_directory(current)
            continue
        except OSError as exc:
            raise TokenizerContractError(
                f"cache directory metadata failed: {current}: {exc}"
            ) from exc
        if not stat.S_ISDIR(status.st_mode) or _is_redirect_or_reparse(
            current, status
        ):
            raise TokenizerContractError(
                f"cache directory must be a regular nonlink directory: {current}"
            )


def _resolved_cache_path(
    cache_root: Path,
    relative_path: str,
    *,
    split: SplitName,
    prepare_parents: bool = False,
) -> Path:
    if not isinstance(cache_root, Path):
        raise TokenizerContractError("cache_root must be a Path")
    if not cache_root.is_absolute():
        raise TokenizerContractError("cache_root must be absolute")
    relative = PurePosixPath(
        _validate_relative_cache_path(relative_path, split=split)
    )
    root = Path(os.path.abspath(cache_root))
    if prepare_parents:
        try:
            root.lstat()
        except FileNotFoundError:
            _require_regular_nonlink_directory(root.parent)
            try:
                root.mkdir()
            except FileExistsError:
                pass
            except OSError as exc:
                raise TokenizerContractError(
                    f"cache root creation failed: {root}: {exc}"
                ) from exc
        except OSError as exc:
            raise TokenizerContractError(
                f"cache root metadata failed: {root}: {exc}"
            ) from exc
    _require_regular_nonlink_directory(root)
    resolved_root = root.resolve(strict=True)
    if not _same_lexical_path(root, resolved_root):
        raise TokenizerContractError(
            "cache_root contains a symlink, junction, or reparse component"
        )
    if any(
        "v3_transformer" in part.casefold()
        for part in (*root.parts, *resolved_root.parts)
    ):
        raise TokenizerContractError("V3 cache roots are quarantined")
    if prepare_parents:
        _prepare_owned_subdirectories(root, relative.parent)
    else:
        current = root
        for part in relative.parent.parts:
            current = current / part
            _require_regular_nonlink_directory(current)
    target = root / Path(*relative.parts)
    try:
        target_status = target.lstat()
    except FileNotFoundError:
        target_status = None
    except OSError as exc:
        raise TokenizerContractError(
            f"cache payload metadata failed: {target}: {exc}"
        ) from exc
    if target_status is not None and _is_redirect_or_reparse(target, target_status):
        raise TokenizerContractError(
            "cache payload cannot be a symlink, junction, or reparse point"
        )
    resolved_target = target.resolve(strict=False)
    if not _same_lexical_path(target, resolved_target):
        raise TokenizerContractError(
            "cache path does not preserve its declared relative identity"
        )
    try:
        resolved_target.relative_to(resolved_root)
    except ValueError as exc:
        raise TokenizerContractError("cache path escapes the cache root") from exc
    if any(
        "v3_transformer" in part.casefold()
        for part in (*target.parts, *resolved_target.parts)
    ):
        raise TokenizerContractError("V3 cache provenance is forbidden")
    return target


def _regular_nonlink(path: Path) -> os.stat_result:
    try:
        status = path.lstat()
    except OSError as exc:
        raise TokenizerContractError(f"cache payload is unavailable: {exc}") from exc
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if (
        not stat.S_ISREG(status.st_mode)
        or stat.S_ISLNK(status.st_mode)
        or bool(getattr(status, "st_file_attributes", 0) & reparse)
    ):
        raise TokenizerContractError(
            "cache payload must be a regular nonlink file"
        )
    return status


def encode_fixture_split_record(
    *,
    split: SplitName,
    raw_bytes: bytes,
    raw_parent_sha256: str,
    spec: SyntheticFixtureTokenizerSpec,
    fixture_contract: SyntheticTokenizerFixtureContract,
    adapter: TokenizerDistributionAdapter,
    cache_root: Path,
    durability_backend: FixtureDurabilityBackend,
    available_disk_bytes: int,
    available_host_ram_bytes: int,
    smoke_token_bytes_per_raw_byte: float,
    smoke_ram_bytes_per_raw_byte: float,
) -> SyntheticFixtureTokenCacheRecord:
    """Publish one split and return its nonproduction operational record."""

    split = _require_split(split)
    if type(raw_bytes) is not bytes or not raw_bytes:
        raise TokenizerContractError("raw_bytes must be nonempty exact bytes")
    _require_sha256(raw_parent_sha256, field="raw_parent_sha256")
    if type(spec) is not SyntheticFixtureTokenizerSpec:
        raise TokenizerContractError(
            "spec must be an exact SyntheticFixtureTokenizerSpec"
        )
    spec.__post_init__()
    if type(fixture_contract) is not SyntheticTokenizerFixtureContract:
        raise TokenizerContractError(
            "fixture_contract must be an exact synthetic fixture contract"
        )
    fixture_contract.__post_init__()
    if build_synthetic_fixture_tokenizer_spec(
        fixture_contract, adapter
    ) != spec:
        raise TokenizerContractError(
            "synthetic tokenizer spec does not bind the supplied fixture/adapter"
        )
    if not callable(getattr(durability_backend, "publish_bytes", None)):
        raise TokenizerContractError(
            "durability_backend must expose publish_bytes"
        )
    forecast = forecast_preprocessing_resources(
        raw_size_bytes=len(raw_bytes),
        smoke_token_bytes_per_raw_byte=smoke_token_bytes_per_raw_byte,
        smoke_ram_bytes_per_raw_byte=smoke_ram_bytes_per_raw_byte,
    )
    _preflight_resources(
        forecast,
        available_disk_bytes=available_disk_bytes,
        available_host_ram_bytes=available_host_ram_bytes,
    )
    try:
        text = raw_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise TokenizerContractError(
            "raw split is not strict UTF-8"
        ) from exc
    try:
        token_ids = tuple(adapter.encode_ordinary(text))
    except Exception as exc:
        raise TokenizerContractError("complete-split encoding failed") from exc
    if not token_ids:
        raise TokenizerContractError("complete-split encoding returned no token IDs")
    if any(
        type(token_id) is not int
        or token_id < 0
        or token_id >= fixture_contract.vocabulary_size
        for token_id in token_ids
    ):
        raise TokenizerContractError(
            "complete-split encoding returned an out-of-range token ID"
        )
    try:
        decoded = adapter.decode(token_ids)
    except Exception as exc:
        raise TokenizerContractError("complete-split round trip failed") from exc
    if type(decoded) is not str or decoded.encode("utf-8") != raw_bytes:
        raise TokenizerContractError(
            "complete-split round trip does not reproduce exact raw bytes"
        )
    payload = b"".join(
        token_id.to_bytes(4, "little", signed=True) for token_id in token_ids
    )
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    identity = SyntheticFixtureTokenCacheRecord.create(
        split=split,
        raw_parent_sha256=raw_parent_sha256,
        fixture_contract=fixture_contract,
        tokenizer=spec,
        token_ids=token_ids,
        payload_sha256=payload_sha256,
    )
    target = _resolved_cache_path(
        cache_root,
        identity.cache_relative_path,
        split=split,
        prepare_parents=True,
    )
    try:
        durability_backend.publish_bytes(target, payload)
    except TokenizerContractError:
        raise
    except Exception as exc:
        raise TokenizerContractError(
            f"durability-backed cache publication failed: {exc}"
        ) from exc
    status = _regular_nonlink(target)
    if status.st_size != identity.payload_size_bytes:
        raise TokenizerContractError(
            "published cache size does not match its identity"
        )
    try:
        installed = target.read_bytes()
    except OSError as exc:
        raise TokenizerContractError(
            f"published cache could not be reopened: {exc}"
        ) from exc
    if hashlib.sha256(installed).hexdigest() != identity.payload_sha256:
        raise TokenizerContractError(
            "published cache hash does not match its identity"
        )
    return identity


def encode_fixture_split(
    *,
    split: SplitName,
    raw_bytes: bytes,
    raw_parent_sha256: str,
    spec: SyntheticFixtureTokenizerSpec,
    fixture_contract: SyntheticTokenizerFixtureContract,
    adapter: TokenizerDistributionAdapter,
    cache_root: Path,
    durability_backend: FixtureDurabilityBackend,
    available_disk_bytes: int,
    available_host_ram_bytes: int,
    smoke_token_bytes_per_raw_byte: float,
    smoke_ram_bytes_per_raw_byte: float,
) -> SyntheticFixtureTokenCacheIdentity:
    """Publish one split and return the frozen canonical Task 1 identity."""

    return encode_fixture_split_record(
        split=split,
        raw_bytes=raw_bytes,
        raw_parent_sha256=raw_parent_sha256,
        spec=spec,
        fixture_contract=fixture_contract,
        adapter=adapter,
        cache_root=cache_root,
        durability_backend=durability_backend,
        available_disk_bytes=available_disk_bytes,
        available_host_ram_bytes=available_host_ram_bytes,
        smoke_token_bytes_per_raw_byte=smoke_token_bytes_per_raw_byte,
        smoke_ram_bytes_per_raw_byte=smoke_ram_bytes_per_raw_byte,
    ).cache_identity


@dataclasses.dataclass(frozen=True, slots=True)
class SyntheticFixtureSplitCapability:
    schema_version: str
    authority: str
    allowed_splits: tuple[ReadableFixtureSplit, ...]
    sealed_cache_identity_sha256: tuple[tuple[SplitName, str], ...]
    capability_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != "wt103-synthetic-split-capability-v1":
            raise TokenizerContractError("unsupported fixture capability schema")
        if self.authority != "nonproduction_synthetic_fixture":
            raise TokenizerContractError("fixture capability cannot be promoted")
        if (
            type(self.allowed_splits) is not tuple
            or not self.allowed_splits
            or any(split not in ("train", "validation") for split in self.allowed_splits)
            or len(set(self.allowed_splits)) != len(self.allowed_splits)
        ):
            raise TokenizerContractError(
                "allowed_splits must be unique train/validation values"
            )
        if type(self.sealed_cache_identity_sha256) is not tuple:
            raise TokenizerContractError(
                "sealed_cache_identity_sha256 must be an immutable tuple"
            )
        observed_splits: set[str] = set()
        for split, digest in self.sealed_cache_identity_sha256:
            _require_split(split)
            _require_sha256(digest, field="sealed cache identity")
            if split in observed_splits:
                raise TokenizerContractError(
                    "sealed cache identities must have unique splits"
                )
            observed_splits.add(split)
        if any(split not in observed_splits for split in self.allowed_splits):
            raise TokenizerContractError(
                "every allowed split must have a sealed cache identity"
            )
        expected = _digest(
            _FIXTURE_CAPABILITY_DOMAIN,
            (
                self.schema_version,
                self.authority,
                self.allowed_splits,
                self.sealed_cache_identity_sha256,
            ),
        )
        if self.capability_sha256 != expected:
            raise TokenizerContractError(
                "capability_sha256 does not match fixture capability"
            )


def issue_fixture_split_capability(
    *,
    allowed_splits: tuple[ReadableFixtureSplit, ...],
    cache_identities: tuple[SyntheticFixtureTokenCacheRecord, ...],
) -> SyntheticFixtureSplitCapability:
    if (
        type(cache_identities) is not tuple
        or not cache_identities
        or any(
            type(identity) is not SyntheticFixtureTokenCacheRecord
            for identity in cache_identities
        )
    ):
        raise TokenizerContractError(
            "cache_identities must contain exact synthetic fixture identities"
        )
    for identity in cache_identities:
        identity.__post_init__()
    sealed = tuple(
        sorted(
            (
                (identity.split, identity.record_sha256)
                for identity in cache_identities
            ),
            key=lambda row: ("train", "validation", "test").index(row[0]),
        )
    )
    values = (
        "wt103-synthetic-split-capability-v1",
        "nonproduction_synthetic_fixture",
        allowed_splits,
        sealed,
    )
    return SyntheticFixtureSplitCapability(
        *values, _digest(_FIXTURE_CAPABILITY_DOMAIN, values)
    )


def open_fixture_token_cache(
    *,
    identity: SyntheticFixtureTokenCacheRecord,
    spec: SyntheticFixtureTokenizerSpec,
    cache_root: Path,
    capability: SyntheticFixtureSplitCapability,
) -> tuple[int, ...]:
    """Reopen train/validation fixture tokens after exact identity validation."""

    if type(identity) is not SyntheticFixtureTokenCacheRecord:
        raise TokenizerContractError(
            "identity must be an exact synthetic fixture cache record"
        )
    if type(spec) is not SyntheticFixtureTokenizerSpec:
        raise TokenizerContractError(
            "spec must be an exact synthetic fixture tokenizer spec"
        )
    if type(capability) is not SyntheticFixtureSplitCapability:
        raise TokenizerContractError(
            "capability must be an exact synthetic split capability"
        )
    identity.__post_init__()
    spec.__post_init__()
    capability.__post_init__()
    if identity.split == "test":
        raise TokenizerContractError(
            "test fixture tokens remain sealed outside the test-opening owner"
        )
    if identity.split not in capability.allowed_splits:
        raise TokenizerContractError(
            f"capability does not permit split {identity.split!r}"
        )
    sealed = dict(capability.sealed_cache_identity_sha256)
    if sealed.get(identity.split) != identity.record_sha256:
        raise TokenizerContractError(
            "capability does not bind the requested cache identity"
        )
    if identity.tokenizer != spec:
        raise TokenizerContractError(
            "cache identity does not bind the supplied tokenizer spec"
        )
    target = _resolved_cache_path(
        cache_root, identity.cache_relative_path, split=identity.split
    )
    status = _regular_nonlink(target)
    if status.st_size != identity.payload_size_bytes:
        raise TokenizerContractError("cache payload size does not match identity")
    try:
        payload = target.read_bytes()
    except OSError as exc:
        raise TokenizerContractError(
            f"cache payload could not be read: {exc}"
        ) from exc
    if hashlib.sha256(payload).hexdigest() != identity.payload_sha256:
        raise TokenizerContractError("cache payload hash does not match identity")
    if len(payload) % 4:
        raise TokenizerContractError("cache payload is malformed int32 data")
    token_ids = tuple(
        int.from_bytes(payload[offset : offset + 4], "little", signed=True)
        for offset in range(0, len(payload), 4)
    )
    if (
        len(token_ids) != identity.token_count
        or min(token_ids) != identity.minimum_token_id
        or max(token_ids) != identity.maximum_token_id
        or any(
            token_id < 0
            or token_id >= identity.fixture_contract.vocabulary_size
            for token_id in token_ids
        )
    ):
        raise TokenizerContractError(
            "cache payload token inventory does not match identity"
        )
    return token_ids


__all__ = [
    "CandidateTokenizerContract",
    "FixtureDurabilityBackend",
    "PreprocessingResourceForecast",
    "SyntheticFixtureSplitCapability",
    "SyntheticFixtureTokenCacheIdentity",
    "SyntheticFixtureTokenCacheRecord",
    "SyntheticFixtureTokenizerSpec",
    "SyntheticTokenizerFixtureContract",
    "TokenizerContractError",
    "TokenizerDistributionAdapter",
    "build_synthetic_fixture_tokenizer_spec",
    "encode_fixture_split",
    "encode_fixture_split_record",
    "forecast_preprocessing_resources",
    "issue_fixture_split_capability",
    "open_fixture_token_cache",
    "validate_tokenizer_adapter",
]
