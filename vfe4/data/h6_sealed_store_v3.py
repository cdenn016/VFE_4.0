"""Authenticated, read-only reopening of the sealed H6 WikiText-2 store."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import zlib
from pathlib import Path
from typing import Final

from vfe4.artifacts.atomic import canonical_json_bytes
from vfe4.config import H6DataConfig
from vfe4.h6_validation_fixture import ValidationSafetyFixtureReference
from vfe4.types.h6 import (
    DataIdentity,
    EncodedTokenStorageIdentity,
    SealedSplitHandle,
)

from .byte_tokenizer import (
    BOS_ID,
    EOS_ID,
    IGNORE_TARGET_ID,
    TOKENIZER_SPEC_BYTES,
    VOCABULARY_SIZE,
    ByteTokenizerV1,
)
from .windows import (
    SEQUENCE_LENGTH,
    WINDOW_STRIDE,
    CausalWindows,
    build_causal_windows,
    materialize_validation_safety_fixture,
)
from .wikitext2 import (
    ACCESS_POLICY_SHA256,
    ARCHIVE_ENTRY_ORDER,
    BINARY_PAYLOAD_ORDER,
    WIKITEXT2_RAW_URL,
    AuthenticatedReopenedDataIdentity,
    AuthenticatedSealedTokenIdentity,
    BinaryDirectoryReference,
    BinaryPayloadRecord,
    BlindedCorpusStore,
    BlindedDataError,
    _data_identity_json,
    _validation_fixture_reference,
)


AUTHENTICATED_BLINDED_STORE_MANIFEST_V3_FILENAME: Final = (
    "wikitext2-blinded-store-v3.json"
)

_SCHEMA_VERSION = "h6-authenticated-blinded-store-manifest-v3"
_SEALED_DIRECTORY_NAME = "wikitext2-blinded"
_SPLITS = ("train", "validation", "test")
_MEMBER_PATHS = ARCHIVE_ENTRY_ORDER[1:]
_PAYLOAD_PATHS = (
    "sealed/wiki.train.raw",
    "sealed/wiki.valid.raw",
    "sealed/wiki.test.raw",
)
_MANIFEST_DOMAIN = b"VFE4-H6-AUTHENTICATED-BLINDED-STORE-MANIFEST-V3\x00"
_BINARY_DIRECTORY_MANIFEST_DOMAIN = b"VFE4-H6-BINARY-DIRECTORY-MANIFEST-V1\x00"
_WINDOW_MANIFEST_DOMAIN = b"VFE4-H6-CAUSAL-WINDOW-MANIFEST-V1\x00"
_MAXIMUM_MANIFEST_BYTES = 262_144


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise BlindedDataError(f"{name} must be lowercase SHA-256 hex")
    return value


def _require_int(
    value: object,
    name: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if type(value) is not int or value < minimum:
        raise BlindedDataError(f"{name} must be an exact bounded integer")
    if maximum is not None and value > maximum:
        raise BlindedDataError(f"{name} must be an exact bounded integer")
    return value


def _require_mapping(
    value: object,
    expected_keys: set[str],
    name: str,
) -> dict[str, object]:
    if type(value) is not dict or set(value) != expected_keys:
        raise BlindedDataError(f"{name} schema is not exact")
    return value


def _is_redirect(path: Path, path_stat: os.stat_result) -> bool:
    if stat.S_ISLNK(path_stat.st_mode):
        return True
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if reparse_flag and (getattr(path_stat, "st_file_attributes", 0) & reparse_flag):
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(callable(is_junction) and is_junction())


def _require_exact_directory(path: Path, name: str) -> Path:
    try:
        path_stat = os.lstat(path)
    except OSError as exc:
        raise BlindedDataError(f"{name} is unavailable") from exc
    if not stat.S_ISDIR(path_stat.st_mode) or _is_redirect(path, path_stat):
        raise BlindedDataError(f"{name} must be a non-redirected directory")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise BlindedDataError(f"{name} cannot be resolved") from exc
    if os.path.normcase(os.fspath(resolved)) != os.path.normcase(os.fspath(path)):
        raise BlindedDataError(f"{name} path is redirected")
    return resolved


def _require_exact_regular_file(path: Path, name: str) -> os.stat_result:
    try:
        path_stat = os.lstat(path)
    except OSError as exc:
        raise BlindedDataError(f"{name} is unavailable") from exc
    if not stat.S_ISREG(path_stat.st_mode) or _is_redirect(path, path_stat):
        raise BlindedDataError(f"{name} must be a non-redirected regular file")
    return path_stat


def _read_exact_file(path: Path, *, maximum_bytes: int, name: str) -> bytes:
    try:
        parent_stat = os.lstat(path.parent)
    except OSError as exc:
        raise BlindedDataError(f"{name} is unavailable") from exc
    path_stat = _require_exact_regular_file(path, name)
    if (
        not stat.S_ISDIR(parent_stat.st_mode)
        or _is_redirect(path.parent, parent_stat)
        or path_stat.st_size > maximum_bytes
    ):
        raise BlindedDataError(f"{name} must be a bounded non-redirected file")

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise BlindedDataError(f"{name} cannot be opened") from exc
    chunks: list[bytes] = []
    total = 0
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (path_stat.st_dev, path_stat.st_ino)
            or opened.st_size != path_stat.st_size
        ):
            raise BlindedDataError(f"{name} identity changed before reading")
        while total <= maximum_bytes:
            chunk = os.read(descriptor, min(65_536, maximum_bytes + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            chunks.append(chunk)
        closed_snapshot = os.fstat(descriptor)
    except BlindedDataError:
        raise
    except OSError as exc:
        raise BlindedDataError(f"{name} cannot be read") from exc
    finally:
        os.close(descriptor)
    try:
        current_parent = os.lstat(path.parent)
        current_path = os.lstat(path)
    except OSError as exc:
        raise BlindedDataError(f"{name} changed while reading") from exc
    if (
        total > maximum_bytes
        or (closed_snapshot.st_dev, closed_snapshot.st_ino, closed_snapshot.st_size)
        != (opened.st_dev, opened.st_ino, opened.st_size)
        or (current_path.st_dev, current_path.st_ino, current_path.st_size)
        != (opened.st_dev, opened.st_ino, opened.st_size)
        or (current_parent.st_dev, current_parent.st_ino)
        != (parent_stat.st_dev, parent_stat.st_ino)
    ):
        raise BlindedDataError(f"{name} changed while reading")
    content = b"".join(chunks)
    if len(content) != opened.st_size:
        raise BlindedDataError(f"{name} length changed while reading")
    return content


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


def _exclusive_manifest_write(path: Path, content: bytes) -> None:
    """Publish one canonical manifest without replacing or repairing anything."""

    if type(content) is not bytes or len(content) > _MAXIMUM_MANIFEST_BYTES:
        raise BlindedDataError("authenticated manifest bytes violate their bound")
    if os.path.lexists(path):
        raise BlindedDataError("authenticated manifest already exists")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise BlindedDataError("authenticated manifest cannot be created") from exc
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    _fsync_directory(path.parent)


def _window_manifest(
    *,
    split: str,
    token_sha256: str,
    windows: CausalWindows,
) -> dict[str, object]:
    if split not in _SPLITS or windows.split != split:
        raise BlindedDataError("causal-window split identity is invalid")
    _require_sha256(token_sha256, "window token identity")
    windows.__post_init__()
    split_bytes = split.encode("ascii")
    digest = hashlib.sha256()
    digest.update(_WINDOW_MANIFEST_DOMAIN)
    digest.update(len(split_bytes).to_bytes(1, "little"))
    digest.update(split_bytes)
    digest.update(bytes.fromhex(token_sha256))
    digest.update(SEQUENCE_LENGTH.to_bytes(2, "little"))
    digest.update(WINDOW_STRIDE.to_bytes(2, "little"))
    digest.update(len(windows).to_bytes(8, "little"))
    digest.update(windows.counted_target_total.to_bytes(8, "little"))
    for inputs, targets, attention_mask, start, real_target_count in zip(
        windows.inputs,
        windows.targets,
        windows.attention_masks,
        windows.starts,
        windows.real_target_counts,
        strict=True,
    ):
        digest.update(start.to_bytes(8, "little"))
        digest.update(real_target_count.to_bytes(2, "little"))
        for token_id in inputs:
            digest.update(token_id.to_bytes(2, "little", signed=False))
        for target_id in targets:
            digest.update(target_id.to_bytes(2, "little", signed=True))
        digest.update(bytes(attention_mask))
    return {
        "schema_version": "vfe4-h6-causal-window-manifest-v1",
        "sequence_length": SEQUENCE_LENGTH,
        "stride": WINDOW_STRIDE,
        "window_count": len(windows),
        "counted_target_total": windows.counted_target_total,
        "window_manifest_sha256": digest.hexdigest(),
    }


def _archive_payload(data_config: H6DataConfig) -> dict[str, object]:
    if type(data_config) is not H6DataConfig:
        raise BlindedDataError("manifest publication requires exact H6DataConfig")
    observed = data_config.observed_archive
    if (
        data_config.schema_version != "h6-data-config-v1"
        or data_config.source_url != WIKITEXT2_RAW_URL
        or data_config.max_archive_bytes != 16_777_216
        or data_config.member_paths != ARCHIVE_ENTRY_ORDER
        or data_config.allowed_compression_methods != (0, 8)
        or data_config.max_member_bytes != 16_777_216
        or data_config.max_total_uncompressed_bytes != 33_554_432
        or data_config.max_compression_ratio != 100
        or observed is None
    ):
        raise BlindedDataError(
            "manifest archive contract is not the frozen H6 contract"
        )
    if tuple(member.path for member in observed.members) != _MEMBER_PATHS:
        raise BlindedDataError("manifest archive member order is not exact")
    return {
        "source_url": data_config.source_url,
        "max_archive_bytes": data_config.max_archive_bytes,
        "member_paths": list(data_config.member_paths),
        "allowed_compression_methods": list(data_config.allowed_compression_methods),
        "max_member_bytes": data_config.max_member_bytes,
        "max_total_uncompressed_bytes": data_config.max_total_uncompressed_bytes,
        "max_compression_ratio": data_config.max_compression_ratio,
        "archive_byte_length": observed.archive_byte_length,
        "archive_sha256": observed.archive_sha256,
        "members": [
            {
                "path": member.path,
                "compressed_size": member.compressed_size,
                "uncompressed_size": member.uncompressed_size,
                "compression_method": member.compression_method,
                "crc32": member.crc32,
                "raw_sha256": member.raw_sha256,
            }
            for member in observed.members
        ],
    }


def _tokenizer_payload() -> dict[str, object]:
    tokenizer = ByteTokenizerV1()
    vocabulary = tokenizer.vocabulary_identity
    return {
        "schema_version": "vfe4-h6-byte-tokenizer-v1",
        "tokenizer_spec_sha256": hashlib.sha256(TOKENIZER_SPEC_BYTES).hexdigest(),
        "vocabulary_id": vocabulary.vocabulary_id,
        "vocabulary_size": VOCABULARY_SIZE,
        "bos_id": BOS_ID,
        "eos_id": EOS_ID,
        "ignore_target_id": IGNORE_TARGET_ID,
        "storage_schema": "vfe4-h6-u16le-tokens-v1",
    }


def _payload_record_map(
    directory_reference: BinaryDirectoryReference,
) -> dict[str, object]:
    if (
        type(directory_reference) is not BinaryDirectoryReference
        or tuple(record.path for record in directory_reference.payloads)
        != BINARY_PAYLOAD_ORDER
    ):
        raise BlindedDataError("binary directory reference inventory is not exact")
    return {record.path: record for record in directory_reference.payloads}


def _manifest_payload(
    *,
    data_config: H6DataConfig,
    directory_reference: BinaryDirectoryReference,
    data_identity: DataIdentity,
    token_splits: tuple[
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
    ],
) -> dict[str, object]:
    if type(data_identity) is not DataIdentity:
        raise BlindedDataError("manifest publication requires exact DataIdentity")
    try:
        data_identity.__post_init__()
    except ValueError as exc:
        raise BlindedDataError("manifest data identity is stale") from exc
    if (
        type(token_splits) is not tuple
        or len(token_splits) != 3
        or any(type(tokens) is not tuple for tokens in token_splits)
    ):
        raise BlindedDataError("manifest token split inventory is not exact")

    archive = _archive_payload(data_config)
    members = archive["members"]
    assert type(members) is list
    records = _payload_record_map(directory_reference)
    token_identities = (
        data_identity.train_tokens,
        data_identity.validation_tokens,
        data_identity.test_tokens,
    )
    raw_hashes = (
        data_identity.train_raw_sha256,
        data_identity.validation_raw_sha256,
        data_identity.test_raw_sha256,
    )
    split_payload: dict[str, object] = {}
    tokenizer = ByteTokenizerV1()
    for index, split in enumerate(_SPLITS):
        member = members[index]
        assert type(member) is dict
        record = records[_PAYLOAD_PATHS[index]]
        tokens = token_splits[index]
        token_identity = tokenizer.storage_identity(tokens)
        if (
            token_identity != token_identities[index]
            or record.raw_content_sha256 != raw_hashes[index]
            or record.raw_content_sha256 != member["raw_sha256"]
            or record.raw_length != member["uncompressed_size"]
        ):
            raise BlindedDataError("manifest split identities disagree")
        windows = build_causal_windows(tokens, split=split)  # type: ignore[arg-type]
        split_payload[split] = {
            "member_path": _MEMBER_PATHS[index],
            "payload_path": _PAYLOAD_PATHS[index],
            "raw_length": record.raw_length,
            "raw_sha256": record.raw_content_sha256,
            "tokens": {
                "storage_schema": token_identity.storage_schema,
                "token_count": token_identity.token_count,
                "byte_length": token_identity.byte_length,
                "encoded_token_sha256": token_identity.encoded_token_sha256,
            },
            "windows": _window_manifest(
                split=split,
                token_sha256=token_identity.encoded_token_sha256,
                windows=windows,
            ),
        }

    fixture_record = records["validation_safety_fixture.bin"]
    fixture = data_identity.validation_fixture
    if (
        fixture_record.raw_content_sha256 != fixture.fixture_sha256
        or fixture_record.raw_length != 311_369
    ):
        raise BlindedDataError("manifest validation fixture identity disagrees")
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "artifact_directory": _SEALED_DIRECTORY_NAME,
        "access_policy_sha256": data_identity.access_policy_sha256,
        "data_identity_sha256": data_identity.data_identity_sha256,
        "archive": archive,
        "tokenizer": _tokenizer_payload(),
        "splits": split_payload,
        "validation_fixture": {
            "schema_version": fixture.policy,
            "relative_path": "wikitext2-blinded/validation_safety_fixture.bin",
            "validation_token_sha256": fixture.validation_token_sha256,
            "fixture_sha256": fixture.fixture_sha256,
            "fixture_raw_length": fixture_record.raw_length,
            "row_count": len(fixture.starts),
        },
        "enclosing_manifest": {
            "schema_version": "vfe4-h6-binary-directory-manifest-v1",
            "relative_path": "wikitext2-blinded/manifest.sha256",
            "manifest_sha256": directory_reference.manifest_sha256,
            "payloads": [
                {
                    "path": record.path,
                    "raw_length": record.raw_length,
                    "raw_content_sha256": record.raw_content_sha256,
                }
                for record in directory_reference.payloads
            ],
        },
    }
    payload["manifest_sha256"] = hashlib.sha256(
        _MANIFEST_DOMAIN + canonical_json_bytes(payload)
    ).hexdigest()
    return payload


def _publish_authenticated_blinded_store_manifest_v3(
    *,
    artifact_root: Path,
    data_config: H6DataConfig,
    directory_reference: BinaryDirectoryReference,
    data_identity: DataIdentity,
    token_splits: tuple[
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
    ],
) -> Path:
    """Publish the additive v3 reopen record after the legacy store is sealed."""

    if not isinstance(artifact_root, Path) or not artifact_root.is_absolute():
        raise BlindedDataError("artifact_root must be an absolute pathlib.Path")
    root = _require_exact_directory(artifact_root, "artifact_root")
    if directory_reference.directory != root / _SEALED_DIRECTORY_NAME:
        raise BlindedDataError("binary directory is outside the exact artifact root")
    payload = _manifest_payload(
        data_config=data_config,
        directory_reference=directory_reference,
        data_identity=data_identity,
        token_splits=token_splits,
    )
    manifest_path = root / AUTHENTICATED_BLINDED_STORE_MANIFEST_V3_FILENAME
    _exclusive_manifest_write(manifest_path, canonical_json_bytes(payload))
    return manifest_path


def _decode_manifest(content: bytes) -> dict[str, object]:
    try:
        decoded = json.loads(content.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise BlindedDataError("authenticated manifest is not UTF-8 JSON") from exc
    payload = _require_mapping(
        decoded,
        {
            "schema_version",
            "artifact_directory",
            "access_policy_sha256",
            "data_identity_sha256",
            "archive",
            "tokenizer",
            "splits",
            "validation_fixture",
            "enclosing_manifest",
            "manifest_sha256",
        },
        "authenticated manifest",
    )
    if canonical_json_bytes(payload) != content:
        raise BlindedDataError("authenticated manifest JSON is not canonical")
    manifest_sha256 = _require_sha256(
        payload["manifest_sha256"],
        "manifest_sha256",
    )
    unsigned = dict(payload)
    unsigned.pop("manifest_sha256")
    expected = hashlib.sha256(
        _MANIFEST_DOMAIN + canonical_json_bytes(unsigned)
    ).hexdigest()
    if manifest_sha256 != expected:
        raise BlindedDataError("authenticated manifest hash changed")
    return payload


def _validate_archive(archive_value: object) -> tuple[dict[str, object], ...]:
    archive = _require_mapping(
        archive_value,
        {
            "source_url",
            "max_archive_bytes",
            "member_paths",
            "allowed_compression_methods",
            "max_member_bytes",
            "max_total_uncompressed_bytes",
            "max_compression_ratio",
            "archive_byte_length",
            "archive_sha256",
            "members",
        },
        "archive identity",
    )
    if (
        archive["source_url"] != WIKITEXT2_RAW_URL
        or archive["max_archive_bytes"] != 16_777_216
        or archive["member_paths"] != list(ARCHIVE_ENTRY_ORDER)
        or archive["allowed_compression_methods"] != [0, 8]
        or archive["max_member_bytes"] != 16_777_216
        or archive["max_total_uncompressed_bytes"] != 33_554_432
        or archive["max_compression_ratio"] != 100
    ):
        raise BlindedDataError("archive identity is not the frozen H6 contract")
    _require_int(
        archive["archive_byte_length"],
        "archive_byte_length",
        minimum=1,
        maximum=16_777_216,
    )
    _require_sha256(archive["archive_sha256"], "archive_sha256")
    member_values = archive["members"]
    if type(member_values) is not list or len(member_values) != 3:
        raise BlindedDataError("archive member inventory is not exact")
    members: list[dict[str, object]] = []
    total = 0
    for index, value in enumerate(member_values):
        member = _require_mapping(
            value,
            {
                "path",
                "compressed_size",
                "uncompressed_size",
                "compression_method",
                "crc32",
                "raw_sha256",
            },
            f"archive member {index}",
        )
        compressed_size = _require_int(
            member["compressed_size"],
            f"archive member {index} compressed_size",
            minimum=1,
            maximum=16_777_216,
        )
        uncompressed_size = _require_int(
            member["uncompressed_size"],
            f"archive member {index} uncompressed_size",
            minimum=1,
            maximum=16_777_216,
        )
        if (
            member["path"] != _MEMBER_PATHS[index]
            or member["compression_method"] not in (0, 8)
            or uncompressed_size > 100 * compressed_size
        ):
            raise BlindedDataError("archive member identity is not exact")
        _require_int(
            member["crc32"],
            f"archive member {index} crc32",
            maximum=0xFFFFFFFF,
        )
        _require_sha256(
            member["raw_sha256"],
            f"archive member {index} raw_sha256",
        )
        total += uncompressed_size
        members.append(member)
    if total > 33_554_432:
        raise BlindedDataError("archive member total exceeds the frozen bound")
    return tuple(members)


def _validate_tokenizer(tokenizer_value: object) -> None:
    tokenizer = _require_mapping(
        tokenizer_value,
        {
            "schema_version",
            "tokenizer_spec_sha256",
            "vocabulary_id",
            "vocabulary_size",
            "bos_id",
            "eos_id",
            "ignore_target_id",
            "storage_schema",
        },
        "tokenizer identity",
    )
    if tokenizer != _tokenizer_payload():
        raise BlindedDataError("tokenizer identity differs from the frozen tokenizer")


def _validate_windows(value: object, split: str) -> dict[str, object]:
    windows = _require_mapping(
        value,
        {
            "schema_version",
            "sequence_length",
            "stride",
            "window_count",
            "counted_target_total",
            "window_manifest_sha256",
        },
        f"{split} window identity",
    )
    if (
        windows["schema_version"] != "vfe4-h6-causal-window-manifest-v1"
        or windows["sequence_length"] != 32
        or windows["stride"] != 32
    ):
        raise BlindedDataError(f"{split} window contract is not frozen")
    _require_int(windows["window_count"], f"{split} window_count", minimum=1)
    _require_int(
        windows["counted_target_total"],
        f"{split} counted_target_total",
        minimum=1,
    )
    _require_sha256(
        windows["window_manifest_sha256"],
        f"{split} window_manifest_sha256",
    )
    return windows


def _validate_splits(
    split_value: object,
    members: tuple[dict[str, object], ...],
) -> tuple[dict[str, object], ...]:
    splits = _require_mapping(
        split_value,
        set(_SPLITS),
        "split inventory",
    )
    result: list[dict[str, object]] = []
    for index, split in enumerate(_SPLITS):
        record = _require_mapping(
            splits[split],
            {
                "member_path",
                "payload_path",
                "raw_length",
                "raw_sha256",
                "tokens",
                "windows",
            },
            f"{split} split identity",
        )
        tokens = _require_mapping(
            record["tokens"],
            {
                "storage_schema",
                "token_count",
                "byte_length",
                "encoded_token_sha256",
            },
            f"{split} token identity",
        )
        if (
            record["member_path"] != _MEMBER_PATHS[index]
            or record["payload_path"] != _PAYLOAD_PATHS[index]
            or record["raw_length"] != members[index]["uncompressed_size"]
            or record["raw_sha256"] != members[index]["raw_sha256"]
            or tokens["storage_schema"] != "vfe4-h6-u16le-tokens-v1"
        ):
            raise BlindedDataError(f"{split} split identity is not exact")
        _require_int(record["raw_length"], f"{split} raw_length", minimum=1)
        _require_sha256(record["raw_sha256"], f"{split} raw_sha256")
        _require_int(tokens["token_count"], f"{split} token_count", minimum=2)
        _require_int(tokens["byte_length"], f"{split} token byte_length", minimum=1)
        _require_sha256(
            tokens["encoded_token_sha256"],
            f"{split} encoded_token_sha256",
        )
        _validate_windows(record["windows"], split)
        result.append(record)
    return tuple(result)


def _validate_fixture(value: object) -> dict[str, object]:
    fixture = _require_mapping(
        value,
        {
            "schema_version",
            "relative_path",
            "validation_token_sha256",
            "fixture_sha256",
            "fixture_raw_length",
            "row_count",
        },
        "validation fixture identity",
    )
    if (
        fixture["schema_version"] != "vfe4-h6-validation-safety-fixture-v1"
        or fixture["relative_path"] != "wikitext2-blinded/validation_safety_fixture.bin"
        or fixture["fixture_raw_length"] != 311_369
        or fixture["row_count"] != 4096
    ):
        raise BlindedDataError("validation fixture identity is not exact")
    _require_sha256(
        fixture["validation_token_sha256"],
        "fixture validation_token_sha256",
    )
    _require_sha256(fixture["fixture_sha256"], "fixture_sha256")
    return fixture


def _validate_enclosing_manifest(
    value: object,
) -> tuple[
    dict[str, object],
    tuple[dict[str, object], ...],
]:
    enclosing = _require_mapping(
        value,
        {
            "schema_version",
            "relative_path",
            "manifest_sha256",
            "payloads",
        },
        "enclosing manifest identity",
    )
    if (
        enclosing["schema_version"] != "vfe4-h6-binary-directory-manifest-v1"
        or enclosing["relative_path"] != "wikitext2-blinded/manifest.sha256"
    ):
        raise BlindedDataError("enclosing manifest identity is not exact")
    _require_sha256(
        enclosing["manifest_sha256"],
        "enclosing manifest_sha256",
    )
    values = enclosing["payloads"]
    if type(values) is not list or len(values) != len(BINARY_PAYLOAD_ORDER):
        raise BlindedDataError("enclosing payload inventory is not exact")
    records: list[dict[str, object]] = []
    for index, value_record in enumerate(values):
        record = _require_mapping(
            value_record,
            {"path", "raw_length", "raw_content_sha256"},
            f"enclosing payload {index}",
        )
        if record["path"] != BINARY_PAYLOAD_ORDER[index]:
            raise BlindedDataError("enclosing payload order is not exact")
        _require_int(
            record["raw_length"],
            f"enclosing payload {index} raw_length",
            minimum=1,
        )
        _require_sha256(
            record["raw_content_sha256"],
            f"enclosing payload {index} raw_content_sha256",
        )
        records.append(record)
    return enclosing, tuple(records)


def _validate_manifest_structure(
    payload: dict[str, object],
) -> tuple[
    tuple[dict[str, object], ...],
    tuple[dict[str, object], ...],
    dict[str, object],
    dict[str, object],
    tuple[dict[str, object], ...],
]:
    if (
        payload["schema_version"] != _SCHEMA_VERSION
        or payload["artifact_directory"] != _SEALED_DIRECTORY_NAME
    ):
        raise BlindedDataError("authenticated manifest header is not exact")
    if (
        _require_sha256(
            payload["access_policy_sha256"],
            "access_policy_sha256",
        )
        != ACCESS_POLICY_SHA256
    ):
        raise BlindedDataError("authenticated manifest access policy is stale")
    _require_sha256(
        payload["data_identity_sha256"],
        "data_identity_sha256",
    )
    members = _validate_archive(payload["archive"])
    _validate_tokenizer(payload["tokenizer"])
    splits = _validate_splits(payload["splits"], members)
    fixture = _validate_fixture(payload["validation_fixture"])
    enclosing, payload_records = _validate_enclosing_manifest(
        payload["enclosing_manifest"]
    )
    if (
        fixture["validation_token_sha256"]
        != _require_mapping(
            splits[1]["tokens"],
            {
                "storage_schema",
                "token_count",
                "byte_length",
                "encoded_token_sha256",
            },
            "validation token identity",
        )["encoded_token_sha256"]
    ):
        raise BlindedDataError("fixture does not bind the validation token identity")
    for index, split in enumerate(splits):
        if (
            payload_records[index]["path"] != split["payload_path"]
            or payload_records[index]["raw_length"] != split["raw_length"]
            or payload_records[index]["raw_content_sha256"] != split["raw_sha256"]
        ):
            raise BlindedDataError("enclosing manifest split identities disagree")
    if (
        payload_records[3]["raw_length"] != fixture["fixture_raw_length"]
        or payload_records[3]["raw_content_sha256"] != fixture["fixture_sha256"]
    ):
        raise BlindedDataError("enclosing manifest fixture identity disagrees")
    return members, splits, fixture, enclosing, payload_records


def _validate_exact_sealed_inventory(sealed_directory: Path) -> None:
    expected_root = {
        "sealed",
        "validation_safety_fixture.bin",
        "data_identity.json",
        "manifest.sha256",
    }
    try:
        root_entries = {
            entry.name: sealed_directory / entry.name
            for entry in os.scandir(sealed_directory)
        }
    except OSError as exc:
        raise BlindedDataError("sealed payload inventory is unavailable") from exc
    if set(root_entries) != expected_root:
        raise BlindedDataError("sealed payload inventory is not exact")
    _require_exact_directory(root_entries["sealed"], "sealed split directory")
    for name in expected_root - {"sealed"}:
        _require_exact_regular_file(
            root_entries[name],
            f"sealed payload {name}",
        )

    expected_splits = {"wiki.train.raw", "wiki.valid.raw", "wiki.test.raw"}
    split_directory = root_entries["sealed"]
    try:
        split_entries = {
            entry.name: split_directory / entry.name
            for entry in os.scandir(split_directory)
        }
    except OSError as exc:
        raise BlindedDataError("sealed split inventory is unavailable") from exc
    if set(split_entries) != expected_splits:
        raise BlindedDataError("sealed split inventory is not exact")
    for name, path in split_entries.items():
        _require_exact_regular_file(path, f"sealed split {name}")


def _typed_payload_records(
    payload_records: tuple[dict[str, object], ...],
) -> tuple[BinaryPayloadRecord, ...]:
    return tuple(
        BinaryPayloadRecord(
            path=record["path"],  # type: ignore[arg-type]
            raw_length=record["raw_length"],  # type: ignore[arg-type]
            raw_content_sha256=record["raw_content_sha256"],  # type: ignore[arg-type]
        )
        for record in payload_records
    )


def _binary_directory_manifest_sha256(
    records: tuple[BinaryPayloadRecord, ...],
) -> str:
    preimage = bytearray(
        _BINARY_DIRECTORY_MANIFEST_DOMAIN + len(records).to_bytes(4, "little")
    )
    for record in records:
        name_bytes = record.path.encode("utf-8")
        preimage += len(name_bytes).to_bytes(2, "little")
        preimage += name_bytes
        preimage += record.raw_length.to_bytes(8, "little")
        preimage += bytes.fromhex(record.raw_content_sha256)
    return hashlib.sha256(preimage).hexdigest()


def _identity_summary(content: bytes) -> dict[str, object]:
    try:
        summary = json.loads(content.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise BlindedDataError("blinded data identity JSON is invalid") from exc
    if type(summary) is not dict or canonical_json_bytes(summary) != content:
        raise BlindedDataError("blinded data identity JSON is not canonical")
    if set(summary) != {
        "access_policy_sha256",
        "archive_sha256",
        "data_identity_sha256",
        "data_schema",
        "splits",
        "validation_fixture_sha256",
    }:
        raise BlindedDataError("blinded data identity summary schema is not exact")
    split_summaries = _require_mapping(
        summary["splits"],
        set(_SPLITS),
        "blinded data identity splits",
    )
    for split in _SPLITS:
        _require_mapping(
            split_summaries[split],
            {"raw_sha256", "token_count", "token_sha256"},
            f"blinded data identity {split}",
        )
    return summary


def _validate_restricted_reopen_materials(
    *,
    payload: dict[str, object],
    members: tuple[dict[str, object], ...],
    splits: tuple[dict[str, object], ...],
    fixture: dict[str, object],
    enclosing: dict[str, object],
    payload_records: tuple[dict[str, object], ...],
    sealed_directory: Path,
) -> tuple[
    AuthenticatedReopenedDataIdentity,
    ValidationSafetyFixtureReference,
]:
    """Authenticate a reopen while leaving the held-out payload unopened."""

    archive = _require_mapping(
        payload["archive"],
        {
            "source_url",
            "max_archive_bytes",
            "member_paths",
            "allowed_compression_methods",
            "max_member_bytes",
            "max_total_uncompressed_bytes",
            "max_compression_ratio",
            "archive_byte_length",
            "archive_sha256",
            "members",
        },
        "archive identity",
    )
    _validate_exact_sealed_inventory(sealed_directory)
    typed_records = _typed_payload_records(payload_records)
    expected_enclosing_sha256 = _binary_directory_manifest_sha256(typed_records)
    if expected_enclosing_sha256 != enclosing["manifest_sha256"]:
        raise BlindedDataError("enclosing payload identity changed")
    enclosing_bytes = _read_exact_file(
        sealed_directory / "manifest.sha256",
        maximum_bytes=65,
        name="enclosing manifest",
    )
    if enclosing_bytes != (expected_enclosing_sha256 + "\n").encode("ascii"):
        raise BlindedDataError("enclosing manifest does not bind its payloads")

    content_by_name: dict[str, bytes] = {}
    test_metadata: os.stat_result | None = None
    for record in typed_records:
        target = sealed_directory.joinpath(*Path(record.path).parts)
        if record.path == "sealed/wiki.test.raw":
            test_metadata = _require_exact_regular_file(
                target,
                "sealed test payload",
            )
            if test_metadata.st_size != record.raw_length:
                raise BlindedDataError("sealed test payload length changed")
            continue
        content = _read_exact_file(
            target,
            maximum_bytes=record.raw_length,
            name=f"sealed payload {record.path}",
        )
        if (
            len(content) != record.raw_length
            or hashlib.sha256(content).hexdigest() != record.raw_content_sha256
        ):
            raise BlindedDataError(f"sealed payload {record.path} content changed")
        content_by_name[record.path] = content
    if test_metadata is None:
        raise BlindedDataError("sealed test payload metadata is unavailable")

    tokenizer = ByteTokenizerV1()
    opened_tokens: list[tuple[int, ...]] = []
    opened_identities: list[EncodedTokenStorageIdentity] = []
    for index, split_name in enumerate(("train", "validation")):
        raw = content_by_name[_PAYLOAD_PATHS[index]]
        tokens = tokenizer.encode(raw)
        split = splits[index]
        token_record = _require_mapping(
            split["tokens"],
            {
                "storage_schema",
                "token_count",
                "byte_length",
                "encoded_token_sha256",
            },
            f"{split_name} token identity",
        )
        observed_token_identity = tokenizer.storage_identity(tokens)
        expected_token_payload = {
            "storage_schema": observed_token_identity.storage_schema,
            "token_count": observed_token_identity.token_count,
            "byte_length": observed_token_identity.byte_length,
            "encoded_token_sha256": observed_token_identity.encoded_token_sha256,
        }
        if (
            len(raw) != split["raw_length"]
            or hashlib.sha256(raw).hexdigest() != split["raw_sha256"]
            or (zlib.crc32(raw) & 0xFFFFFFFF) != members[index]["crc32"]
            or split["raw_sha256"] != members[index]["raw_sha256"]
            or token_record != expected_token_payload
        ):
            raise BlindedDataError(f"reopened {split_name} identity changed")
        observed_windows = build_causal_windows(
            tokens,
            split=split_name,  # type: ignore[arg-type]
        )
        expected_windows = _window_manifest(
            split=split_name,
            token_sha256=observed_token_identity.encoded_token_sha256,
            windows=observed_windows,
        )
        if split["windows"] != expected_windows:
            raise BlindedDataError(f"reopened {split_name} window identity changed")
        opened_tokens.append(tokens)
        opened_identities.append(observed_token_identity)

    test_split = splits[2]
    test_token_record = _require_mapping(
        test_split["tokens"],
        {
            "storage_schema",
            "token_count",
            "byte_length",
            "encoded_token_sha256",
        },
        "test token identity",
    )
    test_token_identity = AuthenticatedSealedTokenIdentity(
        storage_schema=test_token_record["storage_schema"],  # type: ignore[arg-type]
        token_count=test_token_record["token_count"],  # type: ignore[arg-type]
        byte_length=test_token_record["byte_length"],  # type: ignore[arg-type]
        encoded_token_sha256=test_token_record[  # type: ignore[arg-type]
            "encoded_token_sha256"
        ],
    )
    test_windows = _validate_windows(test_split["windows"], "test")
    expected_test_window_count = (
        test_token_identity.token_count - 2
    ) // WINDOW_STRIDE + 1
    if (
        test_metadata.st_size != test_split["raw_length"]
        or test_token_identity.token_count != test_metadata.st_size + 2
        or test_windows["window_count"] != expected_test_window_count
        or test_windows["counted_target_total"] != test_token_identity.token_count - 1
    ):
        raise BlindedDataError("sealed test metadata is inconsistent")

    validation_fixture = materialize_validation_safety_fixture(
        validation_tokens=opened_tokens[1],
        validation_storage_identity=opened_identities[1],
    )
    fixture_bytes = content_by_name["validation_safety_fixture.bin"]
    try:
        validation_fixture.verify_fixture_bytes(fixture_bytes)
    except ValueError as exc:
        raise BlindedDataError("reopened validation fixture changed") from exc

    identity_bytes = content_by_name["data_identity.json"]
    summary = _identity_summary(identity_bytes)
    split_summaries = summary["splits"]
    assert type(split_summaries) is dict
    semantic_token_identities = (
        opened_identities[0],
        opened_identities[1],
        test_token_identity,
    )
    for index, split_name in enumerate(_SPLITS):
        split_summary = split_summaries[split_name]
        assert type(split_summary) is dict
        token_identity = semantic_token_identities[index]
        if split_summary != {
            "raw_sha256": splits[index]["raw_sha256"],
            "token_count": token_identity.token_count,
            "token_sha256": token_identity.encoded_token_sha256,
        }:
            raise BlindedDataError(
                f"reopened {split_name} data identity summary changed"
            )

    identity = AuthenticatedReopenedDataIdentity(
        data_schema=summary["data_schema"],  # type: ignore[arg-type]
        archive_sha256=summary["archive_sha256"],  # type: ignore[arg-type]
        train_raw_sha256=splits[0]["raw_sha256"],  # type: ignore[arg-type]
        validation_raw_sha256=splits[1]["raw_sha256"],  # type: ignore[arg-type]
        test_raw_sha256=splits[2]["raw_sha256"],  # type: ignore[arg-type]
        train_tokens=opened_identities[0],
        validation_tokens=opened_identities[1],
        test_tokens=test_token_identity,
        validation_fixture=validation_fixture,
        access_policy_sha256=summary["access_policy_sha256"],  # type: ignore[arg-type]
        data_identity_sha256=summary["data_identity_sha256"],  # type: ignore[arg-type]
    )
    if (
        identity.archive_sha256 != archive["archive_sha256"]
        or identity.data_identity_sha256 != payload["data_identity_sha256"]
        or identity.access_policy_sha256 != payload["access_policy_sha256"]
        or summary["validation_fixture_sha256"]
        != identity.validation_fixture.fixture_sha256
        or _data_identity_json(identity) != identity_bytes
    ):
        raise BlindedDataError("reopened typed data identity changed")

    directory_reference = BinaryDirectoryReference(
        sealed_directory,
        expected_enclosing_sha256,
        typed_records,
    )
    fixture_reference = _validation_fixture_reference(
        directory_reference=directory_reference,
        fixture_payload=validation_fixture,
        data_identity=identity,
    )
    reopened_fixture = identity.validation_fixture
    if (
        reopened_fixture.policy != fixture["schema_version"]
        or reopened_fixture.validation_token_sha256
        != fixture["validation_token_sha256"]
        or reopened_fixture.fixture_sha256 != fixture["fixture_sha256"]
        or len(reopened_fixture.starts) != fixture["row_count"]
        or fixture_reference.binary_directory_manifest_sha256
        != enclosing["manifest_sha256"]
        or fixture_reference.fixture_raw_sha256 != fixture["fixture_sha256"]
        or fixture_reference.fixture_raw_length != fixture["fixture_raw_length"]
    ):
        raise BlindedDataError("reopened validation fixture identity changed")
    return identity, fixture_reference


def reopen_authenticated_blinded_store_v3(
    manifest_path: Path,
    artifact_root: Path,
) -> BlindedCorpusStore:
    """Reopen one complete sealed store without downloading, writing, or unsealing."""

    if (
        not isinstance(manifest_path, Path)
        or not manifest_path.is_absolute()
        or not isinstance(artifact_root, Path)
        or not artifact_root.is_absolute()
    ):
        raise BlindedDataError(
            "manifest_path and artifact_root must be absolute pathlib.Path values"
        )
    root = _require_exact_directory(artifact_root, "artifact_root")
    expected_manifest_path = root / AUTHENTICATED_BLINDED_STORE_MANIFEST_V3_FILENAME
    if os.path.normcase(os.fspath(manifest_path)) != os.path.normcase(
        os.fspath(expected_manifest_path)
    ):
        raise BlindedDataError("manifest_path is not the exact artifact manifest")
    manifest_bytes = _read_exact_file(
        manifest_path,
        maximum_bytes=_MAXIMUM_MANIFEST_BYTES,
        name="authenticated manifest",
    )
    payload = _decode_manifest(manifest_bytes)
    members, splits, fixture, enclosing, payload_records = _validate_manifest_structure(
        payload
    )
    sealed_directory = _require_exact_directory(
        root / _SEALED_DIRECTORY_NAME,
        "sealed directory",
    )
    _require_exact_directory(
        sealed_directory / "sealed",
        "sealed split directory",
    )
    for relative_path in (*BINARY_PAYLOAD_ORDER, "manifest.sha256"):
        _require_exact_regular_file(
            sealed_directory.joinpath(*Path(relative_path).parts),
            f"sealed payload {relative_path}",
        )
    identity, fixture_reference = _validate_restricted_reopen_materials(
        payload=payload,
        members=members,
        splits=splits,
        fixture=fixture,
        enclosing=enclosing,
        payload_records=payload_records,
        sealed_directory=sealed_directory,
    )

    handles = {
        split: SealedSplitHandle.create(
            split=split,
            data_identity_sha256=identity.data_identity_sha256,
            sealed_content_sha256=raw_sha256,
            access_policy_sha256=identity.access_policy_sha256,
        )
        for split, raw_sha256 in (
            ("train", identity.train_raw_sha256),
            ("validation", identity.validation_raw_sha256),
            ("test", identity.test_raw_sha256),
        )
    }
    store = BlindedCorpusStore(
        data_identity_sha256=identity.data_identity_sha256,
        sealed_train_handle=handles["train"],
        sealed_validation_handle=handles["validation"],
        frozen_validation_fixture=identity.validation_fixture,
        validation_safety_fixture_reference=fixture_reference,
        sealed_test_handle=handles["test"],
        _data_identity=identity,
    )
    from .access import (
        OpeningCapabilityError,
        _register_reopened_blinded_store_v3,
    )

    try:
        _register_reopened_blinded_store_v3(
            store,
            sealed_directory,
        )
    except (OpeningCapabilityError, OSError, ValueError) as exc:
        raise BlindedDataError(
            f"sealed store registration or opening reservation failed: {exc}"
        ) from exc
    return store


__all__ = [
    "AUTHENTICATED_BLINDED_STORE_MANIFEST_V3_FILENAME",
    "reopen_authenticated_blinded_store_v3",
]
