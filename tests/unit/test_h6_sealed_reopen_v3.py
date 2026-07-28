from __future__ import annotations

import builtins
import gc
import hashlib
import io
import json
import os
import zipfile
from collections.abc import Callable
from pathlib import Path

import pytest

from vfe4.artifacts.atomic import canonical_json_bytes
from vfe4.config import H6ArchiveMemberExpectation, H6DataConfig, H6ObservedArchive
from vfe4.data.access import (
    OpeningCapabilityError,
    materialize_validation_safety_fixture,
    validate_durable_test_opening_capability,
)
from vfe4.data.wikitext2 import (
    ACCESS_POLICY_SHA256,
    WIKITEXT2_RAW_URL,
    BlindedDataError,
    H6DataAcquisitionRequest,
    _acquire_wikitext2_blinded,
)


_MEMBERS = (
    "wikitext-2-raw/wiki.train.raw",
    "wikitext-2-raw/wiki.valid.raw",
    "wikitext-2-raw/wiki.test.raw",
)
_MANIFEST_DOMAIN = b"VFE4-H6-AUTHENTICATED-BLINDED-STORE-MANIFEST-V3\x00"


def _archive_bytes() -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        directory = zipfile.ZipInfo("wikitext-2-raw/")
        directory.external_attr = (0o40755 << 16) | 0x10
        archive.writestr(directory, b"")
        archive.writestr(_MEMBERS[0], b"train bytes" * 10)
        archive.writestr(_MEMBERS[1], bytes(range(256)) * 513)
        archive.writestr(_MEMBERS[2], b"held-out test bytes")
    return stream.getvalue()


def _request(archive_bytes: bytes, artifact_root: Path) -> H6DataAcquisitionRequest:
    with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as archive:
        members = tuple(
            H6ArchiveMemberExpectation(
                info.filename,  # type: ignore[arg-type]
                info.compress_size,
                info.file_size,
                info.compress_type,  # type: ignore[arg-type]
                info.CRC,
                hashlib.sha256(archive.read(info)).hexdigest(),
            )
            for info in archive.infolist()
            if not info.is_dir()
        )
    return H6DataAcquisitionRequest(
        data=H6DataConfig(
            "h6-data-config-v1",
            WIKITEXT2_RAW_URL,
            16_777_216,
            ("wikitext-2-raw/", *_MEMBERS),
            (0, 8),
            16_777_216,
            33_554_432,
            100,
            H6ObservedArchive(
                len(archive_bytes),
                hashlib.sha256(archive_bytes).hexdigest(),
                members,
            ),
        ),
        artifact_root=artifact_root,
    )


def _acquire(artifact_root: Path):
    archive_bytes = _archive_bytes()
    return _acquire_wikitext2_blinded(
        _request(archive_bytes, artifact_root),
        lambda _: io.BytesIO(archive_bytes),
    )


def _manifest_path(artifact_root: Path) -> Path:
    from vfe4.data.h6_sealed_store_v3 import (
        AUTHENTICATED_BLINDED_STORE_MANIFEST_V3_FILENAME,
    )

    return artifact_root / AUTHENTICATED_BLINDED_STORE_MANIFEST_V3_FILENAME


def _rewrite_authenticated_manifest(
    manifest_path: Path,
    mutation: Callable[[dict[str, object]], None],
) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert type(payload) is dict
    payload.pop("manifest_sha256")
    mutation(payload)
    payload["manifest_sha256"] = hashlib.sha256(
        _MANIFEST_DOMAIN + canonical_json_bytes(payload)
    ).hexdigest()
    manifest_path.write_bytes(canonical_json_bytes(payload))


def _nested_dict(payload: dict[str, object], name: str) -> dict[str, object]:
    value = payload[name]
    assert type(value) is dict
    return value


def _reservation_path(artifact_root: Path, data_identity_sha256: str) -> Path:
    marker_sha256 = hashlib.sha256(
        b"VFE4-H6-TEST-OPENING-MARKER-V1\x00"
        + bytes.fromhex(data_identity_sha256)
        + bytes.fromhex(ACCESS_POLICY_SHA256)
    ).hexdigest()
    return (
        artifact_root.parent
        / ".vfe4-h6-synthetic-opening-reservations"
        / f"{marker_sha256}.reservation.bin"
    )


def _assert_same_semantic_data_identity(left: object, right: object) -> None:
    fields = (
        "data_schema",
        "archive_sha256",
        "train_raw_sha256",
        "validation_raw_sha256",
        "test_raw_sha256",
        "access_policy_sha256",
        "data_identity_sha256",
    )
    for field in fields:
        assert getattr(left, field) == getattr(right, field)
    for split in ("train", "validation", "test"):
        left_tokens = getattr(left, f"{split}_tokens")
        right_tokens = getattr(right, f"{split}_tokens")
        assert (
            left_tokens.storage_schema,
            left_tokens.token_count,
            left_tokens.byte_length,
            left_tokens.encoded_token_sha256,
        ) == (
            right_tokens.storage_schema,
            right_tokens.token_count,
            right_tokens.byte_length,
            right_tokens.encoded_token_sha256,
        )
    assert getattr(left, "validation_fixture") == getattr(
        right,
        "validation_fixture",
    )


def test_reopen_revalidates_manifest_inventory_and_tokenized_splits(
    tmp_path: Path,
) -> None:
    from vfe4.data.h6_sealed_store_v3 import (
        reopen_authenticated_blinded_store_v3,
    )

    artifact_root = (tmp_path / "valid").resolve()
    acquired = _acquire(artifact_root)
    manifest_path = _manifest_path(artifact_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == "h6-authenticated-blinded-store-manifest-v3"
    assert manifest["data_identity_sha256"] == acquired.data_identity_sha256
    assert _nested_dict(manifest, "archive")["members"] == [
        {
            "compressed_size": 110,
            "compression_method": 0,
            "crc32": 1394064396,
            "path": _MEMBERS[0],
            "raw_sha256": hashlib.sha256(b"train bytes" * 10).hexdigest(),
            "uncompressed_size": 110,
        },
        {
            "compressed_size": 131328,
            "compression_method": 0,
            "crc32": 813796358,
            "path": _MEMBERS[1],
            "raw_sha256": hashlib.sha256(bytes(range(256)) * 513).hexdigest(),
            "uncompressed_size": 131328,
        },
        {
            "compressed_size": 19,
            "compression_method": 0,
            "crc32": 3148457503,
            "path": _MEMBERS[2],
            "raw_sha256": hashlib.sha256(b"held-out test bytes").hexdigest(),
            "uncompressed_size": 19,
        },
    ]
    assert set(_nested_dict(manifest, "splits")) == {
        "train",
        "validation",
        "test",
    }
    assert set(_nested_dict(manifest, "enclosing_manifest")) == {
        "manifest_sha256",
        "payloads",
        "relative_path",
        "schema_version",
    }

    acquired_identity = acquired.data_identity
    del acquired
    gc.collect()
    reopened = reopen_authenticated_blinded_store_v3(
        manifest_path,
        artifact_root,
    )
    _assert_same_semantic_data_identity(reopened.data_identity, acquired_identity)

    def mutate_member(payload: dict[str, object]) -> None:
        archive = _nested_dict(payload, "archive")
        members = archive["members"]
        assert type(members) is list and type(members[0]) is dict
        members[0]["raw_sha256"] = "0" * 64

    def mutate_member_crc(payload: dict[str, object]) -> None:
        archive = _nested_dict(payload, "archive")
        members = archive["members"]
        assert type(members) is list and type(members[0]) is dict
        members[0]["crc32"] = 0

    def mutate_tokenizer(payload: dict[str, object]) -> None:
        _nested_dict(payload, "tokenizer")["tokenizer_spec_sha256"] = "0" * 64

    def mutate_tokens(payload: dict[str, object]) -> None:
        splits = _nested_dict(payload, "splits")
        train = splits["train"]
        assert type(train) is dict
        tokens = train["tokens"]
        assert type(tokens) is dict
        tokens["encoded_token_sha256"] = "0" * 64

    def mutate_fixture(payload: dict[str, object]) -> None:
        _nested_dict(payload, "validation_fixture")["fixture_sha256"] = "0" * 64

    def mutate_window(payload: dict[str, object]) -> None:
        splits = _nested_dict(payload, "splits")
        validation = splits["validation"]
        assert type(validation) is dict
        windows = validation["windows"]
        assert type(windows) is dict
        windows["window_manifest_sha256"] = "0" * 64

    def mutate_enclosing(payload: dict[str, object]) -> None:
        _nested_dict(payload, "enclosing_manifest")["manifest_sha256"] = "0" * 64

    mutations = (
        mutate_member,
        mutate_member_crc,
        mutate_tokenizer,
        mutate_tokens,
        mutate_fixture,
        mutate_window,
        mutate_enclosing,
        lambda payload: payload.__setitem__("unexpected", True),
    )
    original_manifest = manifest_path.read_bytes()
    for mutation in mutations:
        manifest_path.write_bytes(original_manifest)
        _rewrite_authenticated_manifest(manifest_path, mutation)
        with pytest.raises(BlindedDataError):
            reopen_authenticated_blinded_store_v3(manifest_path, artifact_root)


def test_reopen_returns_registered_store_without_test_rows(tmp_path: Path) -> None:
    from vfe4.data.h6_sealed_store_v3 import (
        reopen_authenticated_blinded_store_v3,
    )

    artifact_root = (tmp_path / "store").resolve()
    original = _acquire(artifact_root)
    identity = original.data_identity
    del original
    gc.collect()

    reopened = reopen_authenticated_blinded_store_v3(
        _manifest_path(artifact_root),
        artifact_root,
    )

    _assert_same_semantic_data_identity(reopened.data_identity, identity)
    assert reopened.sealed_train_handle.split == "train"
    assert reopened.sealed_validation_handle.split == "validation"
    assert materialize_validation_safety_fixture(reopened) is (
        reopened.frozen_validation_fixture
    )
    assert not hasattr(reopened, "test")
    assert not hasattr(reopened, "opening_capability")
    assert not any(
        type(value) is bytes
        for name, value in vars(reopened).items()
        if name != "_data_identity"
    )
    with pytest.raises(OpeningCapabilityError, match="no opening proof"):
        validate_durable_test_opening_capability(reopened, object())  # type: ignore[arg-type]


def test_reopen_refuses_partial_hash_mismatch_or_consumed_opening(
    tmp_path: Path,
) -> None:
    from vfe4.data.h6_sealed_store_v3 import (
        reopen_authenticated_blinded_store_v3,
    )

    partial_root = (tmp_path / "partial").resolve()
    _acquire(partial_root)
    (partial_root / "wikitext2-blinded" / "sealed" / "wiki.train.raw").unlink()
    with pytest.raises(BlindedDataError):
        reopen_authenticated_blinded_store_v3(
            _manifest_path(partial_root),
            partial_root,
        )

    drift_root = (tmp_path / "drift").resolve()
    _acquire(drift_root)
    drift_path = drift_root / "wikitext2-blinded" / "sealed" / "wiki.valid.raw"
    changed = bytearray(drift_path.read_bytes())
    changed[len(changed) // 2] ^= 1
    drift_path.write_bytes(changed)
    with pytest.raises(BlindedDataError):
        reopen_authenticated_blinded_store_v3(
            _manifest_path(drift_root),
            drift_root,
        )

    consumed_root = (tmp_path / "consumed").resolve()
    consumed = _acquire(consumed_root)
    reservation_path = _reservation_path(
        consumed_root,
        consumed.data_identity_sha256,
    )
    reservation_path.parent.mkdir()
    reservation_path.write_bytes(b"already reserved or consumed")
    with pytest.raises(BlindedDataError, match="reservation|consumed"):
        reopen_authenticated_blinded_store_v3(
            _manifest_path(consumed_root),
            consumed_root,
        )


def test_reopen_refuses_redirected_store_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vfe4.data.h6_sealed_store_v3 import (
        reopen_authenticated_blinded_store_v3,
    )

    artifact_root = (tmp_path / "redirect").resolve()
    _acquire(artifact_root)
    fixture = artifact_root / "wikitext2-blinded" / "validation_safety_fixture.bin"
    original = fixture.with_name("fixture-original.bin")
    fixture.rename(original)
    try:
        os.symlink(original, fixture)
    except OSError:
        original.rename(fixture)
        path_type = type(fixture)
        original_is_junction = getattr(path_type, "is_junction", None)

        def fake_is_junction(path: Path) -> bool:
            if path == fixture:
                return True
            return bool(callable(original_is_junction) and original_is_junction(path))

        monkeypatch.setattr(
            path_type,
            "is_junction",
            fake_is_junction,
            raising=False,
        )

    with pytest.raises(BlindedDataError, match="redirect"):
        reopen_authenticated_blinded_store_v3(
            _manifest_path(artifact_root),
            artifact_root,
        )


def test_reopen_never_downloads_replaces_or_repairs_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vfe4.data.h6_sealed_store_v3 as sealed_v3
    import vfe4.data.wikitext2 as wikitext2

    artifact_root = (tmp_path / "read-only").resolve()
    _acquire(artifact_root)
    manifest_path = _manifest_path(artifact_root)
    before = {
        path.relative_to(artifact_root).as_posix(): path.read_bytes()
        for path in artifact_root.rglob("*")
        if path.is_file()
    }

    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("reopen attempted a download, publication, or repair")

    monkeypatch.setattr(wikitext2, "_official_urlopen", forbidden)
    monkeypatch.setattr(wikitext2, "publish_blinded_binary_directory", forbidden)
    monkeypatch.setattr(sealed_v3, "_exclusive_manifest_write", forbidden)

    sealed_v3.reopen_authenticated_blinded_store_v3(
        manifest_path,
        artifact_root,
    )

    after = {
        path.relative_to(artifact_root).as_posix(): path.read_bytes()
        for path in artifact_root.rglob("*")
        if path.is_file()
    }
    assert after == before

    missing = artifact_root / "wikitext2-blinded" / "sealed" / "wiki.train.raw"
    missing.unlink()
    partial_before = {
        path.relative_to(artifact_root).as_posix(): path.read_bytes()
        for path in artifact_root.rglob("*")
        if path.is_file()
    }
    with pytest.raises(BlindedDataError):
        sealed_v3.reopen_authenticated_blinded_store_v3(
            manifest_path,
            artifact_root,
        )
    partial_after = {
        path.relative_to(artifact_root).as_posix(): path.read_bytes()
        for path in artifact_root.rglob("*")
        if path.is_file()
    }
    assert partial_after == partial_before


def test_reopen_never_opens_reads_tokenizes_or_builds_test_raw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vfe4.data.h6_sealed_store_v3 as sealed_v3

    artifact_root = (tmp_path / "test-remains-sealed").resolve()
    _acquire(artifact_root)
    test_path = artifact_root / "wikitext2-blinded" / "sealed" / "wiki.test.raw"
    canonical_test_path = os.path.normcase(os.fspath(test_path))

    def is_test_path(value: object) -> bool:
        if isinstance(value, int):
            return False
        try:
            normalized = os.path.normcase(
                os.path.abspath(os.fspath(value))  # type: ignore[arg-type]
            )
        except TypeError:
            return False
        return normalized == canonical_test_path

    original_builtin_open = builtins.open
    original_io_open = io.open
    original_os_open = os.open
    original_path_open = Path.open
    original_read_bytes = Path.read_bytes
    original_encode = sealed_v3.ByteTokenizerV1.encode
    original_build_windows = sealed_v3.build_causal_windows

    def guarded_builtin_open(file: object, *args: object, **kwargs: object):
        if is_test_path(file):
            raise AssertionError("reopen opened the held-out test payload")
        return original_builtin_open(file, *args, **kwargs)

    def guarded_io_open(file: object, *args: object, **kwargs: object):
        if is_test_path(file):
            raise AssertionError("reopen opened the held-out test payload")
        return original_io_open(file, *args, **kwargs)

    def guarded_os_open(path: object, *args: object, **kwargs: object) -> int:
        if is_test_path(path):
            raise AssertionError("reopen os.opened the held-out test payload")
        return original_os_open(path, *args, **kwargs)  # type: ignore[arg-type]

    def guarded_path_open(
        path: Path,
        *args: object,
        **kwargs: object,
    ):
        if is_test_path(path):
            raise AssertionError("reopen Path.opened the held-out test payload")
        return original_path_open(path, *args, **kwargs)

    def guarded_read_bytes(path: Path) -> bytes:
        if is_test_path(path):
            raise AssertionError(
                "reopen Path.read_bytes read the held-out test payload"
            )
        return original_read_bytes(path)

    def guarded_encode(tokenizer: object, raw: bytes):
        if raw == b"held-out test bytes":
            raise AssertionError("reopen tokenized the held-out test payload")
        return original_encode(tokenizer, raw)

    def guarded_build_windows(tokens: object, *, split: str):
        if split == "test":
            raise AssertionError("reopen built held-out test windows")
        return original_build_windows(tokens, split=split)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "open", guarded_builtin_open)
    monkeypatch.setattr(io, "open", guarded_io_open)
    monkeypatch.setattr(os, "open", guarded_os_open)
    monkeypatch.setattr(Path, "open", guarded_path_open)
    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    monkeypatch.setattr(sealed_v3.ByteTokenizerV1, "encode", guarded_encode)
    monkeypatch.setattr(sealed_v3, "build_causal_windows", guarded_build_windows)

    sealed_v3.reopen_authenticated_blinded_store_v3(
        _manifest_path(artifact_root),
        artifact_root,
    )


def test_reopen_registration_namespace_cannot_be_relabeled_by_manifest(
    tmp_path: Path,
) -> None:
    from vfe4.data.h6_sealed_store_v3 import (
        reopen_authenticated_blinded_store_v3,
    )

    artifact_root = (tmp_path / "immutable-registration-authority").resolve()
    acquired = _acquire(artifact_root)
    manifest_path = _manifest_path(artifact_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert type(manifest) is dict
    assert "registration_mode" not in manifest

    reservation_path = _reservation_path(
        artifact_root,
        acquired.data_identity_sha256,
    )
    reservation_path.parent.mkdir()
    reservation_path.write_bytes(b"already reserved or consumed")

    with pytest.raises(BlindedDataError, match="reservation|consumed"):
        reopen_authenticated_blinded_store_v3(manifest_path, artifact_root)

    def self_assert_synthetic(payload: dict[str, object]) -> None:
        payload["registration_mode"] = "synthetic"

    _rewrite_authenticated_manifest(manifest_path, self_assert_synthetic)
    with pytest.raises(BlindedDataError, match="schema|registration"):
        reopen_authenticated_blinded_store_v3(manifest_path, artifact_root)
