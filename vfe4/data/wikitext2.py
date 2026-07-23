"""Bounded, identity-bound, blinded acquisition of the official WikiText-2 raw archive."""

from __future__ import annotations

import contextlib
import ctypes
import errno
import hashlib
import io
import json
import os
import shutil
import stat
import sys
import unicodedata
import urllib.request
import uuid
import zipfile
import zlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import BinaryIO, Final

from vfe4.artifacts.atomic import canonical_json_bytes
from vfe4.config import H6DataConfig, H6PredictionResolvedConfig
from vfe4.types.h6 import (
    DataIdentity,
    SealedSplitHandle,
    ValidationSafetyFixture,
)

from .byte_tokenizer import ByteTokenizerV1
from .windows import materialize_validation_safety_fixture

WIKITEXT2_RAW_URL: Final = (
    "https://s3.amazonaws.com/research.metamind.io/wikitext/"
    "wikitext-2-raw-v1.zip"
)
ARCHIVE_ENTRY_ORDER: Final = (
    "wikitext-2-raw/",
    "wikitext-2-raw/wiki.train.raw",
    "wikitext-2-raw/wiki.valid.raw",
    "wikitext-2-raw/wiki.test.raw",
)
BINARY_PAYLOAD_ORDER: Final = (
    "sealed/wiki.train.raw",
    "sealed/wiki.valid.raw",
    "sealed/wiki.test.raw",
    "validation_safety_fixture.bin",
    "data_identity.json",
)

_MANIFEST_DOMAIN = b"VFE4-H6-BINARY-DIRECTORY-MANIFEST-V1\x00"
_ACCESS_POLICY_BYTES = (
    b"VFE4-H6-BLINDED-DATA-ACCESS-POLICY-V1\x00"
    b"validation-safety-before-readiness\x00"
    b"train-validation-after-readiness-pass\x00"
    b"test-after-durable-opening-only\x00"
)
ACCESS_POLICY_SHA256: Final = hashlib.sha256(_ACCESS_POLICY_BYTES).hexdigest()


class BlindedDataError(RuntimeError):
    """The exact official archive, identity, or blinded-store contract failed."""


class BinaryPublicationError(RuntimeError):
    """A five-payload blinded directory could not be atomically published."""


@dataclass(frozen=True)
class H6DataAcquisitionRequest:
    """Exact resolved inputs for one identity-bound blinded acquisition."""

    data: H6DataConfig
    artifact_root: Path
    access_policy_sha256: str = ACCESS_POLICY_SHA256
    expected_data_identity_sha256: str | None = None

    def __post_init__(self) -> None:
        if type(self.data) is not H6DataConfig:
            raise ValueError("data must be the exact resolved H6DataConfig")
        if not isinstance(self.artifact_root, Path) or not self.artifact_root.is_absolute():
            raise ValueError("artifact_root must be an absolute pathlib.Path")
        if self.access_policy_sha256 != ACCESS_POLICY_SHA256:
            raise ValueError("access policy does not match the frozen H6 data policy")
        if self.expected_data_identity_sha256 is not None:
            digest = self.expected_data_identity_sha256
            if (
                type(digest) is not str
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("expected data identity must be lowercase SHA-256 hex")

    @classmethod
    def from_prediction_config(
        cls, config: H6PredictionResolvedConfig
    ) -> "H6DataAcquisitionRequest":
        if type(config) is not H6PredictionResolvedConfig:
            raise ValueError("config must be the exact resolved H6 Prediction config")
        return cls(
            data=config.data,
            artifact_root=config.artifact_root,
            access_policy_sha256=config.access_policy_sha256,
            expected_data_identity_sha256=config.data_identity_sha256,
        )


@dataclass(frozen=True)
class BinaryPayloadRecord:
    path: str
    raw_length: int
    raw_content_sha256: str


@dataclass(frozen=True)
class BinaryDirectoryReference:
    directory: Path
    manifest_sha256: str
    payloads: tuple[BinaryPayloadRecord, ...]


@dataclass(frozen=True)
class BlindedCorpusStore:
    """Public identities and opaque handles; access owns all filesystem state."""

    data_identity_sha256: str
    sealed_train_handle: SealedSplitHandle
    sealed_validation_handle: SealedSplitHandle
    frozen_validation_fixture: ValidationSafetyFixture
    sealed_test_handle: SealedSplitHandle
    _data_identity: DataIdentity = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self._data_identity) is not DataIdentity:
            raise ValueError("store requires an exact DataIdentity")
        self._data_identity.__post_init__()
        if self.data_identity_sha256 != self._data_identity.data_identity_sha256:
            raise ValueError("store data identity does not match its retained record")
        expected_handles = {
            "train": self.sealed_train_handle,
            "validation": self.sealed_validation_handle,
            "test": self.sealed_test_handle,
        }
        for split, handle in expected_handles.items():
            if type(handle) is not SealedSplitHandle:
                raise ValueError("store split handles must be exact sealed handles")
            handle.__post_init__()
            if (
                handle.split != split
                or handle.data_identity_sha256 != self.data_identity_sha256
                or handle.access_policy_sha256 != self._data_identity.access_policy_sha256
            ):
                raise ValueError("sealed split handle does not match the store")
        if type(self.frozen_validation_fixture) is not ValidationSafetyFixture:
            raise ValueError("store requires the frozen validation safety fixture")
        self.frozen_validation_fixture.__post_init__()
        if self.frozen_validation_fixture != self._data_identity.validation_fixture:
            raise ValueError("store validation fixture does not match its data identity")

    @property
    def data_identity(self) -> DataIdentity:
        self._data_identity.__post_init__()
        return self._data_identity


def _source_bytes(source: object) -> bytes:
    if type(source) is bytes:
        return source
    if isinstance(source, Path):
        with source.open("rb") as handle:
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = handle.read(65536)
                if not chunk:
                    break
                total += len(chunk)
                if total > 33_554_432:
                    raise BinaryPublicationError("binary payload exceeds the frozen bound")
                chunks.append(chunk)
        return b"".join(chunks)
    raise BinaryPublicationError("binary payload sources must be immutable bytes or Path")


def _validate_data_identity_json(content: bytes) -> None:
    try:
        decoded = json.loads(content.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise BinaryPublicationError("data_identity.json must be canonical UTF-8 JSON") from exc
    if not isinstance(decoded, Mapping):
        raise BinaryPublicationError("data_identity.json must contain one JSON object")

    forbidden = {
        "manifest_sha256",
        "manifest_path",
        "directory_manifest_identity",
        "directory_manifest_sha256",
    }

    def walk(value: object) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                if type(key) is not str:
                    raise BinaryPublicationError("data identity JSON keys must be strings")
                if key.casefold() in forbidden:
                    raise BinaryPublicationError(
                        "data_identity.json cannot contain its enclosing manifest identity"
                    )
                walk(nested)
        elif isinstance(value, list):
            for nested in value:
                walk(nested)

    walk(decoded)
    if canonical_json_bytes(decoded) != content:
        raise BinaryPublicationError("data_identity.json must use exact canonical JSON bytes")


def _payload_items(payloads: Mapping[str, object]) -> dict[str, bytes]:
    if not isinstance(payloads, Mapping):
        raise BinaryPublicationError("payloads must be a mapping")
    try:
        items = list(payloads.items())
    except (TypeError, AttributeError) as exc:
        raise BinaryPublicationError("payload mapping cannot be enumerated") from exc
    names = [name for name, _ in items]
    if any(type(name) is not str for name in names):
        raise BinaryPublicationError("payload paths must be exact strings")
    if len(names) != len(set(names)) or len(names) != len({name.casefold() for name in names}):
        raise BinaryPublicationError("duplicate or case-colliding payload paths are forbidden")
    if set(names) != set(BINARY_PAYLOAD_ORDER) or len(names) != len(BINARY_PAYLOAD_ORDER):
        raise BinaryPublicationError("caller must supply exactly the five frozen payload paths")
    result = {name: _source_bytes(source) for name, source in items}
    _validate_data_identity_json(result["data_identity.json"])
    return result


def _exclusive_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _install_directory_no_replace(source: Path, destination: Path) -> None:
    """Use an OS no-replace directory move or fail closed when unavailable."""
    if os.name == "nt":
        os.rename(source, destination)
        return
    if sys.platform.startswith("linux"):
        renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
        if renameat2 is None:
            raise OSError(errno.ENOTSUP, "renameat2 no-replace is unavailable")
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(destination),
            1,
        )
        if result != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), destination)
        return
    raise OSError(errno.ENOTSUP, "no OS no-replace directory primitive is available")


def publish_blinded_binary_directory(
    destination: Path, payloads: Mapping[str, object]
) -> BinaryDirectoryReference:
    """Durably publish exactly five binary payloads plus a self-excluding manifest."""
    if not isinstance(destination, Path):
        raise BinaryPublicationError("destination must be a pathlib.Path")
    content_by_name = _payload_items(payloads)
    staging: Path | None = None
    installed = False
    try:
        final = destination.resolve(strict=False)
        parent = final.parent
        parent.mkdir(parents=True, exist_ok=True)
        if final.exists():
            raise BinaryPublicationError("blinded destination already exists")
        staging = parent / f".{final.name}.staging-{uuid.uuid4().hex}"
        staging.mkdir()
        if staging.stat().st_dev != parent.stat().st_dev:
            raise BinaryPublicationError("blinded stage must be on the destination volume")
        records: list[BinaryPayloadRecord] = []
        for name in BINARY_PAYLOAD_ORDER:
            relative = PurePosixPath(name)
            target = staging.joinpath(*relative.parts)
            _exclusive_write(target, content_by_name[name])
            records.append(
                BinaryPayloadRecord(
                    name,
                    len(content_by_name[name]),
                    hashlib.sha256(content_by_name[name]).hexdigest(),
                )
            )
        manifest_preimage = bytearray(
            _MANIFEST_DOMAIN + len(records).to_bytes(4, "little")
        )
        for record in records:
            name_bytes = record.path.encode("utf-8")
            manifest_preimage += len(name_bytes).to_bytes(2, "little")
            manifest_preimage += name_bytes
            manifest_preimage += record.raw_length.to_bytes(8, "little")
            manifest_preimage += bytes.fromhex(record.raw_content_sha256)
        manifest_sha256 = hashlib.sha256(manifest_preimage).hexdigest()
        _exclusive_write(
            staging / "manifest.sha256", (manifest_sha256 + "\n").encode("ascii")
        )
        for directory in sorted(
            (path for path in staging.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            _fsync_directory(directory)
        _fsync_directory(staging)
        if final.exists():
            raise BinaryPublicationError("blinded destination already exists")
        _install_directory_no_replace(staging, final)
        installed = True
        _fsync_directory(parent)
        return BinaryDirectoryReference(final, manifest_sha256, tuple(records))
    except BinaryPublicationError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError, UnicodeError) as exc:
        raise BinaryPublicationError(f"blinded binary publication failed: {exc}") from exc
    finally:
        if staging is not None and not installed and staging.exists():
            try:
                shutil.rmtree(staging)
            except OSError:
                pass


def _official_urlopen(url: str) -> BinaryIO:
    return urllib.request.urlopen(url)  # noqa: S310 - exact frozen HTTPS URL only


def _read_bounded_stream(stream: BinaryIO, maximum: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = stream.read(min(65536, maximum + 1 - total))
        if not chunk:
            break
        if type(chunk) is not bytes:
            raise BlindedDataError("archive opener must yield binary bytes")
        total += len(chunk)
        if total > maximum:
            raise BlindedDataError("archive exceeds the frozen compressed-byte bound")
        chunks.append(chunk)
    return b"".join(chunks)


def _canonical_archive_path(name: str) -> str:
    if type(name) is not str or not name or "\\" in name:
        raise BlindedDataError("archive path is not canonical POSIX text")
    if unicodedata.normalize("NFC", name) != name:
        raise BlindedDataError("archive path is not NFC canonical")
    posix = PurePosixPath(name)
    windows = PureWindowsPath(name)
    canonical_posix = posix.as_posix() + ("/" if name.endswith("/") else "")
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or bool(windows.root)
        or canonical_posix != name
        or any(part in (".", "..") for part in posix.parts)
    ):
        raise BlindedDataError("archive path escapes the frozen member root")
    return name


def _validate_zip_entry_type(info: zipfile.ZipInfo) -> None:
    mode = (info.external_attr >> 16) & 0xFFFF
    kind = stat.S_IFMT(mode)
    if info.is_dir():
        if info.filename != ARCHIVE_ENTRY_ORDER[0]:
            raise BlindedDataError("archive contains an unexpected directory")
        if info.file_size != 0 or info.compress_size != 0:
            raise BlindedDataError("the sole archive directory entry must be empty")
        if kind not in (0, stat.S_IFDIR):
            raise BlindedDataError("archive directory has an unsafe file type")
    elif kind not in (0, stat.S_IFREG):
        raise BlindedDataError("links and special archive members are forbidden")


def _stream_zip_member(
    archive: zipfile.ZipFile, info: zipfile.ZipInfo, maximum: int
) -> bytes:
    chunks: list[bytes] = []
    total = 0
    crc = 0
    try:
        with archive.open(info, "r") as member:
            while True:
                chunk = member.read(min(65536, maximum + 1 - total))
                if not chunk:
                    break
                total += len(chunk)
                if total > maximum:
                    raise BlindedDataError("archive member decompressed beyond its bound")
                crc = zlib.crc32(chunk, crc)
                chunks.append(chunk)
    except (OSError, EOFError, zipfile.BadZipFile, RuntimeError) as exc:
        raise BlindedDataError(f"archive member stream failed: {exc}") from exc
    content = b"".join(chunks)
    if total != info.file_size:
        raise BlindedDataError("archive member streamed size disagrees with its directory")
    if (crc & 0xFFFFFFFF) != info.CRC:
        raise BlindedDataError("archive member streamed CRC disagrees with its directory")
    return content


def _normalized_request(
    config: H6DataAcquisitionRequest,
) -> tuple[H6DataConfig, Path, str, str | None]:
    if type(config) is not H6DataAcquisitionRequest:
        raise BlindedDataError("acquisition requires the exact H6DataAcquisitionRequest")
    try:
        config.__post_init__()
    except ValueError as exc:
        raise BlindedDataError("H6DataAcquisitionRequest failed validation") from exc
    data = config.data
    if data.source_url != WIKITEXT2_RAW_URL:
        raise BlindedDataError("H6 data source URL is not the exact official URL")
    if (
        data.schema_version != "h6-data-config-v1"
        or data.max_archive_bytes != 16_777_216
        or data.member_paths != ARCHIVE_ENTRY_ORDER
        or data.allowed_compression_methods != (0, 8)
        or data.max_member_bytes != 16_777_216
        or data.max_total_uncompressed_bytes != 33_554_432
        or data.max_compression_ratio != 100
    ):
        raise BlindedDataError("H6 data bounds differ from the frozen archive contract")
    if data.observed_archive is None:
        raise BlindedDataError("observed archive identities must be frozen before acquisition")
    artifact_root = config.artifact_root
    access_policy_sha256 = config.access_policy_sha256
    if access_policy_sha256 != ACCESS_POLICY_SHA256:
        raise BlindedDataError("access policy does not match the frozen H6 policy")
    expected_data_identity = config.expected_data_identity_sha256
    if expected_data_identity is not None and (
        type(expected_data_identity) is not str or len(expected_data_identity) != 64
    ):
        raise BlindedDataError("expected data identity must be a SHA-256 hex digest")
    return data, artifact_root, access_policy_sha256, expected_data_identity


def _data_identity_json(identity: DataIdentity) -> bytes:
    return canonical_json_bytes(
        {
            "access_policy_sha256": identity.access_policy_sha256,
            "archive_sha256": identity.archive_sha256,
            "data_identity_sha256": identity.data_identity_sha256,
            "data_schema": identity.data_schema,
            "splits": {
                "test": {
                    "raw_sha256": identity.test_raw_sha256,
                    "token_count": identity.test_tokens.token_count,
                    "token_sha256": identity.test_tokens.encoded_token_sha256,
                },
                "train": {
                    "raw_sha256": identity.train_raw_sha256,
                    "token_count": identity.train_tokens.token_count,
                    "token_sha256": identity.train_tokens.encoded_token_sha256,
                },
                "validation": {
                    "raw_sha256": identity.validation_raw_sha256,
                    "token_count": identity.validation_tokens.token_count,
                    "token_sha256": identity.validation_tokens.encoded_token_sha256,
                },
            },
            "validation_fixture_sha256": identity.validation_fixture.fixture_sha256,
        }
    )


def _acquire_wikitext2_blinded_impl(
    config: H6DataAcquisitionRequest, opener, register_store
) -> BlindedCorpusStore:
    """Shared acquisition mechanics behind fixed production and synthetic wrappers."""
    data_config, artifact_root, access_policy_sha256, expected_data_identity = (
        _normalized_request(config)
    )
    if not callable(opener):
        raise BlindedDataError("archive opener must be callable")
    response = opener(WIKITEXT2_RAW_URL)
    if not hasattr(response, "read") or not hasattr(response, "close"):
        raise BlindedDataError("archive opener must return a closable binary stream")
    with contextlib.closing(response):
        archive_bytes = _read_bounded_stream(response, data_config.max_archive_bytes)
    observed = data_config.observed_archive
    assert observed is not None
    if len(archive_bytes) != observed.archive_byte_length:
        raise BlindedDataError("archive_byte_length does not match the frozen observation")
    if hashlib.sha256(archive_bytes).hexdigest() != observed.archive_sha256:
        raise BlindedDataError("archive_sha256 does not match the frozen observation")

    expected_members = {member.path: member for member in observed.members}
    if tuple(expected_members) != ARCHIVE_ENTRY_ORDER[1:]:
        raise BlindedDataError("observed member inventory/order is not the exact three files")
    raw_by_path: dict[str, bytes] = {}
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as archive:
            entries = archive.infolist()
            names = [_canonical_archive_path(info.filename) for info in entries]
            if (
                tuple(names) != ARCHIVE_ENTRY_ORDER
                or len(set(names)) != len(names)
                or len({name.casefold() for name in names}) != len(names)
            ):
                raise BlindedDataError("archive must contain exactly one directory and three files")
            total_uncompressed = 0
            for info in entries:
                _validate_zip_entry_type(info)
                if info.flag_bits & 0x1:
                    raise BlindedDataError("encrypted archive members are forbidden")
                if info.is_dir():
                    continue
                if info.compress_type not in data_config.allowed_compression_methods:
                    raise BlindedDataError("archive compression method is not frozen")
                if (
                    info.compress_size <= 0
                    or info.file_size <= 0
                    or info.compress_size > data_config.max_member_bytes
                    or info.file_size > data_config.max_member_bytes
                    or info.file_size > data_config.max_compression_ratio * info.compress_size
                ):
                    raise BlindedDataError("archive member violates size or ratio bounds")
                total_uncompressed += info.file_size
                expectation = expected_members.get(info.filename)
                if expectation is None:
                    raise BlindedDataError("archive member has no frozen expectation")
                if (
                    info.compress_size != expectation.compressed_size
                    or info.file_size != expectation.uncompressed_size
                    or info.compress_type != expectation.compression_method
                    or info.CRC != expectation.crc32
                ):
                    raise BlindedDataError("archive member metadata differs from the frozen observation")
                raw = _stream_zip_member(archive, info, data_config.max_member_bytes)
                if hashlib.sha256(raw).hexdigest() != expectation.raw_sha256:
                    raise BlindedDataError("archive member raw hash differs from the frozen observation")
                raw_by_path[info.filename] = raw
            if total_uncompressed > data_config.max_total_uncompressed_bytes:
                raise BlindedDataError("archive total uncompressed bytes exceed the frozen bound")
    except BlindedDataError:
        raise
    except (OSError, EOFError, zipfile.BadZipFile, RuntimeError, ValueError) as exc:
        raise BlindedDataError(f"archive validation failed: {exc}") from exc

    tokenizer = ByteTokenizerV1()
    train_raw, validation_raw, test_raw = (
        raw_by_path[path] for path in ARCHIVE_ENTRY_ORDER[1:]
    )
    train_tokens = tokenizer.encode(train_raw)
    validation_tokens = tokenizer.encode(validation_raw)
    test_tokens = tokenizer.encode(test_raw)
    train_identity = tokenizer.storage_identity(train_tokens)
    validation_identity = tokenizer.storage_identity(validation_tokens)
    test_identity = tokenizer.storage_identity(test_tokens)
    validation_fixture = materialize_validation_safety_fixture(
        validation_tokens=validation_tokens,
        validation_storage_identity=validation_identity,
    )
    data_identity = DataIdentity.create(
        archive_sha256=observed.archive_sha256,
        train_raw_sha256=hashlib.sha256(train_raw).hexdigest(),
        validation_raw_sha256=hashlib.sha256(validation_raw).hexdigest(),
        test_raw_sha256=hashlib.sha256(test_raw).hexdigest(),
        train_tokens=train_identity,
        validation_tokens=validation_identity,
        test_tokens=test_identity,
        validation_fixture=validation_fixture,
        access_policy_sha256=access_policy_sha256,
    )
    if (
        expected_data_identity is not None
        and expected_data_identity != data_identity.data_identity_sha256
    ):
        raise BlindedDataError("derived data identity differs from the frozen config")

    fixture_bytes = object.__getattribute__(validation_fixture, "_fixture_bytes")
    reference = publish_blinded_binary_directory(
        artifact_root / "wikitext2-blinded",
        {
            "sealed/wiki.train.raw": train_raw,
            "sealed/wiki.valid.raw": validation_raw,
            "sealed/wiki.test.raw": test_raw,
            "validation_safety_fixture.bin": fixture_bytes,
            "data_identity.json": _data_identity_json(data_identity),
        },
    )
    handles = {
        split: SealedSplitHandle.create(
            split=split,
            data_identity_sha256=data_identity.data_identity_sha256,
            sealed_content_sha256=hashlib.sha256(raw).hexdigest(),
            access_policy_sha256=access_policy_sha256,
        )
        for split, raw in (
            ("train", train_raw),
            ("validation", validation_raw),
            ("test", test_raw),
        )
    }
    store = BlindedCorpusStore(
        data_identity.data_identity_sha256,
        handles["train"],
        handles["validation"],
        validation_fixture,
        handles["test"],
        data_identity,
    )
    register_store(store, reference.directory)
    return store


def _acquire_wikitext2_blinded(
    config: H6DataAcquisitionRequest, opener
) -> BlindedCorpusStore:
    """Explicitly synthetic acquisition seam used only by bounded unit fixtures."""
    from .access import _register_synthetic_blinded_store

    return _acquire_wikitext2_blinded_impl(
        config,
        opener,
        _register_synthetic_blinded_store,
    )


def acquire_wikitext2_blinded(
    config: H6DataAcquisitionRequest,
) -> BlindedCorpusStore:
    """Acquire only the exact official URL through the private fixed opener."""
    from .access import _register_production_blinded_store

    return _acquire_wikitext2_blinded_impl(
        config,
        _official_urlopen,
        _register_production_blinded_store,
    )


__all__ = [
    "ACCESS_POLICY_SHA256",
    "BINARY_PAYLOAD_ORDER",
    "BinaryDirectoryReference",
    "BinaryPayloadRecord",
    "BinaryPublicationError",
    "BlindedCorpusStore",
    "BlindedDataError",
    "H6DataAcquisitionRequest",
    "WIKITEXT2_RAW_URL",
    "acquire_wikitext2_blinded",
    "publish_blinded_binary_directory",
]
