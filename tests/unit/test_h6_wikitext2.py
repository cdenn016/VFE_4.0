from __future__ import annotations

import hashlib
import io
import json
import zipfile
import dataclasses
import warnings
from collections.abc import Iterator, Mapping
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from vfe4.artifacts.atomic import canonical_json_bytes
from vfe4.config import H6ArchiveMemberExpectation, H6DataConfig, H6ObservedArchive
from vfe4.data.access import (
    _revalidate_blinded_data_identity_for_readiness,
)
from vfe4.data.byte_tokenizer import ByteTokenizerV1
from vfe4.data.wikitext2 import (
    ACCESS_POLICY_SHA256,
    BINARY_PAYLOAD_ORDER,
    WIKITEXT2_RAW_URL,
    BlindedDataError,
    BinaryPublicationError,
    H6DataAcquisitionRequest,
    _acquire_wikitext2_blinded,
    acquire_wikitext2_blinded,
    publish_blinded_binary_directory,
)


MEMBERS = (
    "wikitext-2-raw/wiki.train.raw",
    "wikitext-2-raw/wiki.valid.raw",
    "wikitext-2-raw/wiki.test.raw",
)


def _archive_bytes(
    *,
    validation: bytes | None = None,
    extra_name: str | None = None,
    duplicate_train: bool = False,
    symlink_train: bool = False,
    compression: int = zipfile.ZIP_STORED,
) -> bytes:
    validation = validation or (bytes(range(256)) * 513)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=compression) as archive:
        directory = zipfile.ZipInfo("wikitext-2-raw/")
        directory.external_attr = (0o40755 << 16) | 0x10
        archive.writestr(directory, b"")
        for name, payload in (
            (MEMBERS[0], b"train\r\nbytes"),
            (MEMBERS[1], validation),
            (MEMBERS[2], b"test\nbytes"),
        ):
            if name == MEMBERS[0] and symlink_train:
                info = zipfile.ZipInfo(name)
                info.external_attr = 0o120777 << 16
                archive.writestr(info, payload)
            else:
                archive.writestr(name, payload)
        if duplicate_train:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                archive.writestr(MEMBERS[0], b"duplicate")
        if extra_name is not None:
            archive.writestr(extra_name, b"extra")
    return stream.getvalue()


def _config(archive_bytes: bytes, artifact_root: Path) -> H6DataAcquisitionRequest:
    with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as archive:
        expected_members = tuple(
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
            ("wikitext-2-raw/", *MEMBERS),
            (0, 8),
            16_777_216,
            33_554_432,
            100,
            H6ObservedArchive(
                len(archive_bytes),
                hashlib.sha256(archive_bytes).hexdigest(),
                expected_members,
            ),
        ),
        artifact_root=artifact_root,
    )


class _Response(io.BytesIO):
    requested_url: str | None = None

    def __init__(self, payload: bytes, url: str) -> None:
        super().__init__(payload)
        self.requested_url = url


def test_internal_acquisition_streams_exact_archive_and_seals_identity(tmp_path: Path) -> None:
    archive_bytes = _archive_bytes()
    response: _Response | None = None

    def opener(url: str) -> _Response:
        nonlocal response
        response = _Response(archive_bytes, url)
        return response

    store = _acquire_wikitext2_blinded(
        _config(archive_bytes, tmp_path), opener
    )

    assert response is not None and response.closed
    assert response.requested_url == WIKITEXT2_RAW_URL
    assert store.data_identity.archive_sha256 == hashlib.sha256(archive_bytes).hexdigest()
    assert store.data_identity.access_policy_sha256 == ACCESS_POLICY_SHA256
    assert store.sealed_train_handle.data_identity_sha256 == store.data_identity_sha256
    assert store.sealed_test_handle.split == "test"
    assert len(store.frozen_validation_fixture.starts) == 4096
    assert (tmp_path / "wikitext2-blinded" / "manifest.sha256").is_file()
    assert not tuple((tmp_path / "wikitext2-blinded").rglob("*.tokens"))
    encoded_train = ByteTokenizerV1().encode(b"train\r\nbytes")
    assert store.data_identity.train_tokens.token_count == len(encoded_train)
    rehydrated = _revalidate_blinded_data_identity_for_readiness(
        tmp_path / "wikitext2-blinded",
        expected_archive_sha256=store.data_identity.archive_sha256,
        expected_data_identity_sha256=store.data_identity_sha256,
        expected_access_policy_sha256=store.data_identity.access_policy_sha256,
    )
    assert rehydrated == store.data_identity

    fixture_path = (
        tmp_path / "wikitext2-blinded" / "validation_safety_fixture.bin"
    )
    fixture_bytes = fixture_path.read_bytes()
    fixture_path.write_bytes(fixture_bytes[:-1] + bytes((fixture_bytes[-1] ^ 1,)))
    with pytest.raises(BlindedDataError, match="manifest|fixture"):
        _revalidate_blinded_data_identity_for_readiness(
            tmp_path / "wikitext2-blinded",
            expected_archive_sha256=store.data_identity.archive_sha256,
            expected_data_identity_sha256=store.data_identity_sha256,
            expected_access_policy_sha256=store.data_identity.access_policy_sha256,
        )


def test_blinded_acquisition_emits_fixture_reference_without_split_paths_or_handles(
    tmp_path: Path,
) -> None:
    archive_bytes = _archive_bytes()
    store = _acquire_wikitext2_blinded(
        _config(archive_bytes, tmp_path),
        lambda url: _Response(archive_bytes, url),
    )

    reference = store.validation_safety_fixture_reference
    public_fields = {
        field.name: getattr(reference, field.name)
        for field in dataclasses.fields(reference)
    }

    assert type(reference).__name__ == "ValidationSafetyFixtureReference"
    assert reference.schema_version == (
        "vfe4-h6-validation-safety-fixture-reference-v1"
    )
    assert reference.logical_payload_name == "validation_safety_fixture.bin"
    assert reference.local_payload_path == (
        tmp_path / "wikitext2-blinded" / "validation_safety_fixture.bin"
    ).resolve()
    assert (
        reference.binary_directory_manifest_sha256
        == (
            tmp_path / "wikitext2-blinded" / "manifest.sha256"
        ).read_text(encoding="ascii").strip()
    )
    assert reference.data_identity_sha256 == store.data_identity_sha256
    assert (
        reference.access_policy_sha256
        == store.data_identity.access_policy_sha256
    )
    assert (
        reference.validation_token_sha256
        == store.data_identity.validation_tokens.encoded_token_sha256
    )
    assert (
        reference.fixture_raw_sha256
        == store.frozen_validation_fixture.fixture_sha256
    )
    assert reference.fixture_raw_length == 311_369
    assert reference.row_count == 4096
    assert len(reference.reference_sha256) == 64
    assert [
        (name, value)
        for name, value in public_fields.items()
        if isinstance(value, Path)
    ] == [("local_payload_path", reference.local_payload_path)]
    assert not any(type(value) is bytes for value in public_fields.values())
    assert not any("handle" in name or "split" in name for name in public_fields)
    assert not hasattr(reference, "__dict__")


def test_public_acquisition_has_no_url_argument_and_uses_only_official_opener(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_bytes = _archive_bytes()
    requested: list[str] = []

    def fake_urlopen(url: str) -> _Response:
        requested.append(url)
        return _Response(archive_bytes, url)

    monkeypatch.setattr("vfe4.data.wikitext2._official_urlopen", fake_urlopen)
    store = acquire_wikitext2_blinded(_config(archive_bytes, tmp_path))

    assert requested == [WIKITEXT2_RAW_URL]
    assert store.data_identity_sha256 == store.data_identity.data_identity_sha256
    with pytest.raises(TypeError):
        acquire_wikitext2_blinded(  # type: ignore[call-arg]
            _config(archive_bytes, tmp_path / "other"), opener=fake_urlopen
        )
    with pytest.raises(TypeError):
        acquire_wikitext2_blinded(  # type: ignore[call-arg]
            _config(archive_bytes, tmp_path / "synthetic"),
            synthetic=True,
        )


def test_acquisition_request_is_exact_frozen_and_rejects_duck_typed_configs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_bytes = _archive_bytes()
    request = _config(archive_bytes, tmp_path)
    with pytest.raises(FrozenInstanceError):
        request.artifact_root = tmp_path / "changed"  # type: ignore[misc]

    class DuckRequest:
        data = request.data
        artifact_root = request.artifact_root
        access_policy_sha256 = request.access_policy_sha256

    calls: list[str] = []
    monkeypatch.setattr(
        "vfe4.data.wikitext2._official_urlopen",
        lambda url: calls.append(url),
    )
    with pytest.raises(BlindedDataError, match="H6DataAcquisitionRequest"):
        acquire_wikitext2_blinded(DuckRequest())  # type: ignore[arg-type]
    with pytest.raises(BlindedDataError, match="H6DataAcquisitionRequest"):
        _acquire_wikitext2_blinded(  # type: ignore[arg-type]
            request.data,
            lambda _: (_ for _ in ()).throw(AssertionError("opener reached")),
        )
    assert calls == []


def _data_identity_payload() -> bytes:
    return canonical_json_bytes(
        {
            "access_policy_sha256": ACCESS_POLICY_SHA256,
            "data_identity_sha256": "a" * 64,
            "data_schema": "vfe4-h6-data-identity-v1",
        }
    )


def _payloads() -> dict[str, bytes]:
    return {
        "sealed/wiki.train.raw": b"train",
        "sealed/wiki.valid.raw": b"validation",
        "sealed/wiki.test.raw": b"test",
        "validation_safety_fixture.bin": b"fixture",
        "data_identity.json": _data_identity_payload(),
    }


def test_binary_publisher_uses_fixed_order_exact_self_excluding_manifest(
    tmp_path: Path,
) -> None:
    payloads = _payloads()
    reversed_payloads = dict(reversed(tuple(payloads.items())))
    reference = publish_blinded_binary_directory(tmp_path / "sealed", reversed_payloads)

    expected_preimage = bytearray(
        b"VFE4-H6-BINARY-DIRECTORY-MANIFEST-V1\x00"
        + len(BINARY_PAYLOAD_ORDER).to_bytes(4, "little")
    )
    for name in BINARY_PAYLOAD_ORDER:
        name_bytes = name.encode("utf-8")
        content = payloads[name]
        expected_preimage += len(name_bytes).to_bytes(2, "little") + name_bytes
        expected_preimage += len(content).to_bytes(8, "little")
        expected_preimage += hashlib.sha256(content).digest()
    expected_digest = hashlib.sha256(expected_preimage).hexdigest()

    assert reference.manifest_sha256 == expected_digest
    assert tuple(record.path for record in reference.payloads) == BINARY_PAYLOAD_ORDER
    assert (reference.directory / "manifest.sha256").read_bytes() == (
        expected_digest + "\n"
    ).encode("ascii")
    assert "manifest" not in json.loads(
        (reference.directory / "data_identity.json").read_text(encoding="utf-8")
    )


class _DuplicateItems(Mapping[str, bytes]):
    def __getitem__(self, key: str) -> bytes:
        return _payloads()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(_payloads())

    def __len__(self) -> int:
        return len(_payloads())

    def items(self):  # type: ignore[override]
        values = list(_payloads().items())
        return (*values, values[0])


def test_binary_publisher_rejects_payload_and_identity_contract_mutants(
    tmp_path: Path,
) -> None:
    mutations: list[Mapping[str, bytes]] = []
    missing = _payloads()
    missing.pop("sealed/wiki.test.raw")
    mutations.append(missing)
    mutations.append({**_payloads(), "extra.bin": b"extra"})
    mutations.append({**_payloads(), "manifest.sha256": b"forged"})
    mutations.append(_DuplicateItems())
    mutations.append(
        {
            **_payloads(),
            "data_identity.json": b'{"manifest_sha256":"' + b"a" * 64 + b'"}',
        }
    )
    mutations.append(
        {**_payloads(), "data_identity.json": b'{"z":1, "a":2}'}
    )

    for index, mutation in enumerate(mutations):
        with pytest.raises(BinaryPublicationError):
            publish_blinded_binary_directory(tmp_path / f"bad-{index}", mutation)


def test_binary_publisher_is_no_replace_and_cleans_only_owned_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "sealed"
    destination.mkdir()
    (destination / "sentinel").write_bytes(b"owned elsewhere")
    with pytest.raises(BinaryPublicationError):
        publish_blinded_binary_directory(destination, _payloads())
    assert (destination / "sentinel").read_bytes() == b"owned elsewhere"

    sibling = tmp_path / ".unrelated-stage"
    sibling.mkdir()
    monkeypatch.setattr(
        "vfe4.data.wikitext2._install_directory_no_replace",
        lambda source, target: (_ for _ in ()).throw(OSError("install failed")),
    )
    with pytest.raises(BinaryPublicationError):
        publish_blinded_binary_directory(tmp_path / "failed", _payloads())
    assert sibling.is_dir()
    assert not tuple(tmp_path.glob(".failed.staging-*"))


def test_archive_contract_rejects_wrong_identity_or_unsafe_member_sets(
    tmp_path: Path,
) -> None:
    valid = _archive_bytes()
    original = _config(valid, tmp_path / "wrong")
    observed = original.data.observed_archive
    assert observed is not None
    wrong = H6DataAcquisitionRequest(
        data=dataclasses.replace(
            original.data,
            observed_archive=dataclasses.replace(observed, archive_sha256="0" * 64),
        ),
        artifact_root=original.artifact_root,
    )
    with pytest.raises(BlindedDataError, match="archive_sha256"):
        _acquire_wikitext2_blinded(wrong, lambda _: _Response(valid, WIKITEXT2_RAW_URL))

    unsafe_archives = (
        _archive_bytes(extra_name="wikitext-2-raw/extra.raw"),
        _archive_bytes(extra_name="../escape.raw"),
        _archive_bytes(duplicate_train=True),
        _archive_bytes(symlink_train=True),
        _archive_bytes(
            validation=b"x" * 131_400,
            compression=zipfile.ZIP_DEFLATED,
        ),
    )
    for index, archive_bytes in enumerate(unsafe_archives):
        with pytest.raises(BlindedDataError):
            _acquire_wikitext2_blinded(
                _config(archive_bytes, tmp_path / f"unsafe-{index}"),
                lambda _, content=archive_bytes: _Response(content, WIKITEXT2_RAW_URL),
            )
