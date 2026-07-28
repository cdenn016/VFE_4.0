from __future__ import annotations

import ast
import hashlib
import importlib
import json
import os
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from typing import BinaryIO

import pytest


def _durability():
    return importlib.import_module("vfe4.artifacts.durability")


def _manifest():
    return importlib.import_module("vfe4.artifacts.manifest")


class _DirectoryHandle:
    pass


class _PortablePosixSyscalls:
    def __init__(self, volume_facts: object) -> None:
        self.volume_facts_value = volume_facts
        self.calls: list[tuple[object, ...]] = []
        self.fail_flush = False
        self.raise_after_replace = False

    def volume_facts(self, path: Path) -> object:
        self.calls.append(("volume_facts", path))
        return self.volume_facts_value

    def open_exclusive(self, path: Path, flags: int, mode: int) -> BinaryIO:
        self.calls.append(("open_exclusive", path, flags, mode))
        return path.open("xb", buffering=0)

    def write_all(self, handle: BinaryIO, payload: bytes) -> None:
        self.calls.append(("write_all", payload))
        handle.write(payload)

    def flush_file(self, handle: BinaryIO) -> None:
        self.calls.append(("flush_file",))
        if self.fail_flush:
            raise OSError(5, "injected flush failure")
        handle.flush()
        os.fsync(handle.fileno())

    def close(self, handle: BinaryIO | _DirectoryHandle) -> None:
        self.calls.append(("close", type(handle).__name__))
        if not isinstance(handle, _DirectoryHandle):
            handle.close()

    def read_regular_bytes(self, path: Path) -> bytes:
        self.calls.append(("read_regular_bytes", path))
        return path.read_bytes()

    def replace(self, source: Path, destination: Path) -> None:
        self.calls.append(("replace", source, destination))
        os.replace(source, destination)
        if self.raise_after_replace:
            raise OSError(5, "injected post-replace failure")

    def open_directory(self, path: Path, flags: int) -> _DirectoryHandle:
        self.calls.append(("open_directory", path, flags))
        return _DirectoryHandle()

    def flush_directory(self, handle: _DirectoryHandle) -> None:
        self.calls.append(("flush_directory",))

    def unlink(self, path: Path) -> None:
        self.calls.append(("unlink", path))
        path.unlink(missing_ok=True)


class _PortableWindowsSyscalls:
    def __init__(self, volume_facts: object) -> None:
        self.volume_facts_value = volume_facts
        self.calls: list[tuple[object, ...]] = []
        self.raise_after_move = False

    def volume_facts(self, path: Path) -> object:
        self.calls.append(("volume_facts", path))
        return self.volume_facts_value

    def create_file(
        self,
        path: Path,
        *,
        desired_access: int,
        share_mode: int,
        creation_disposition: int,
        flags_and_attributes: int,
    ) -> BinaryIO:
        self.calls.append(
            (
                "create_file",
                path,
                desired_access,
                share_mode,
                creation_disposition,
                flags_and_attributes,
            )
        )
        return path.open("xb", buffering=0)

    def write_all(self, handle: BinaryIO, payload: bytes) -> None:
        self.calls.append(("write_all", payload))
        handle.write(payload)

    def flush_file(self, handle: BinaryIO) -> None:
        self.calls.append(("flush_file",))
        handle.flush()
        os.fsync(handle.fileno())

    def close(self, handle: BinaryIO) -> None:
        self.calls.append(("close",))
        handle.close()

    def read_regular_bytes(self, path: Path) -> bytes:
        self.calls.append(("read_regular_bytes", path))
        return path.read_bytes()

    def move_file_ex(self, source: Path, destination: Path, flags: int) -> None:
        self.calls.append(("move_file_ex", source, destination, flags))
        os.replace(source, destination)
        if self.raise_after_move:
            raise OSError(5, "injected post-move failure")

    def unlink(self, path: Path) -> None:
        self.calls.append(("unlink", path))
        path.unlink(missing_ok=True)


def _closed_manifest_bytes(entries: list[dict[str, object]]) -> bytes:
    return json.dumps(
        {
            "entries": entries,
            "schema_version": "vfe4-closed-manifest-v1",
        },
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _entry(path: str, payload: bytes, *, kind: str = "file") -> dict[str, object]:
    return {
        "kind": kind,
        "relative_path": path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def test_posix_backend_uses_exclusive_flush_reopen_replace_and_directory_fsync(
    tmp_path: Path,
) -> None:
    durability = _durability()
    volume = durability.VolumeFacts(
        volume_path="/",
        volume_serial="device-7",
        filesystem_type="ext4",
        is_remote=False,
    )
    syscalls = _PortablePosixSyscalls(volume)
    backend = durability.PosixDurabilityBackend(syscalls=syscalls)
    target = tmp_path / "record.json"

    created = backend.create_exclusive(target, b'{"old":true}')
    replaced = backend.replace_durable(target, b'{"new":true}')

    assert target.read_bytes() == b'{"new":true}'
    assert created.reopen_verified is True
    assert replaced.sha256 == hashlib.sha256(b'{"new":true}').hexdigest()
    assert replaced.volume_identity == volume.identity
    open_calls = [call for call in syscalls.calls if call[0] == "open_exclusive"]
    assert all(call[2] & os.O_CREAT and call[2] & os.O_EXCL for call in open_calls)
    assert all(call[3] == 0o600 for call in open_calls)
    replace_ordinal = next(
        index for index, call in enumerate(syscalls.calls) if call[0] == "replace"
    )
    assert any(
        call[0] == "read_regular_bytes"
        for call in syscalls.calls[:replace_ordinal]
    )
    assert any(
        call[0] == "flush_directory"
        for call in syscalls.calls[replace_ordinal + 1 :]
    )


def test_exclusive_collision_preserves_existing_bytes(tmp_path: Path) -> None:
    durability = _durability()
    volume = durability.VolumeFacts("/", "device-7", "ext4", False)
    syscalls = _PortablePosixSyscalls(volume)
    backend = durability.PosixDurabilityBackend(syscalls=syscalls)
    target = tmp_path / "reservation.json"
    backend.create_exclusive(target, b"first")

    with pytest.raises(durability.DurabilityCollisionError):
        backend.create_exclusive(target, b"second")

    assert target.read_bytes() == b"first"
    assert len(
        [call for call in syscalls.calls if call[0] == "open_exclusive"]
    ) == 2


def test_failure_before_replace_preserves_old_target_and_is_determinate(
    tmp_path: Path,
) -> None:
    durability = _durability()
    volume = durability.VolumeFacts("/", "device-7", "ext4", False)
    syscalls = _PortablePosixSyscalls(volume)
    backend = durability.PosixDurabilityBackend(syscalls=syscalls)
    target = tmp_path / "checkpoint.bin"
    backend.create_exclusive(target, b"old")
    syscalls.fail_flush = True

    with pytest.raises(durability.DurabilityOperationError) as captured:
        backend.replace_durable(target, b"new")

    assert captured.value.phase == "flush_staging"
    assert captured.value.indeterminate is False
    assert target.read_bytes() == b"old"


def test_failure_returned_after_replace_is_explicitly_indeterminate(
    tmp_path: Path,
) -> None:
    durability = _durability()
    volume = durability.VolumeFacts("/", "device-7", "ext4", False)
    syscalls = _PortablePosixSyscalls(volume)
    backend = durability.PosixDurabilityBackend(syscalls=syscalls)
    target = tmp_path / "checkpoint.bin"
    backend.create_exclusive(target, b"old")
    syscalls.raise_after_replace = True

    with pytest.raises(durability.DurabilityOperationError) as captured:
        backend.replace_durable(target, b"new")

    assert captured.value.indeterminate is True
    assert captured.value.error_code == 5
    assert captured.value.obligations


def test_windows_backend_uses_required_write_through_and_move_flags(
    tmp_path: Path,
) -> None:
    durability = _durability()
    volume = durability.VolumeFacts(
        volume_path="C:\\",
        volume_serial="A1B2-C3D4",
        filesystem_type="NTFS",
        is_remote=False,
    )
    syscalls = _PortableWindowsSyscalls(volume)
    backend = durability.WindowsDurabilityBackend(syscalls=syscalls)
    target = tmp_path / "manifest.json"

    backend.create_exclusive(target, b"old")
    identity = backend.replace_durable(target, b"new")

    create_calls = [call for call in syscalls.calls if call[0] == "create_file"]
    assert create_calls
    assert all(
        call[4] == durability.WINDOWS_CREATE_NEW
        and call[5]
        == (
            durability.WINDOWS_FILE_ATTRIBUTE_NORMAL
            | durability.WINDOWS_FILE_FLAG_WRITE_THROUGH
        )
        for call in create_calls
    )
    move_call = next(
        call for call in syscalls.calls if call[0] == "move_file_ex"
    )
    assert move_call[3] == (
        durability.WINDOWS_MOVEFILE_REPLACE_EXISTING
        | durability.WINDOWS_MOVEFILE_WRITE_THROUGH
    )
    assert any(call[0] == "flush_file" for call in syscalls.calls)
    assert identity.volume_identity == volume.identity


def test_cross_volume_staging_fails_before_create(tmp_path: Path) -> None:
    durability = _durability()
    first = durability.VolumeFacts("/", "device-7", "ext4", False)
    second = durability.VolumeFacts("/", "device-8", "ext4", False)

    class CrossVolume(_PortablePosixSyscalls):
        def volume_facts(self, path: Path) -> object:
            self.calls.append(("volume_facts", path))
            return second if ".vfe4-stage-" in path.name else first

    syscalls = CrossVolume(first)
    backend = durability.PosixDurabilityBackend(syscalls=syscalls)

    with pytest.raises(durability.DurabilityOperationError) as captured:
        backend.replace_durable(tmp_path / "target.bin", b"payload")

    assert captured.value.phase == "same_volume"
    assert not any(call[0] == "open_exclusive" for call in syscalls.calls)


def test_active_backend_probe_is_passed_and_removes_only_probe_files(
    tmp_path: Path,
) -> None:
    durability = _durability()
    sentinel = tmp_path / "user-owned.txt"
    sentinel.write_bytes(b"keep")

    identity = durability.probe_durability(tmp_path)

    assert identity.status == "pass"
    assert identity.backend_kind == ("windows" if os.name == "nt" else "posix")
    assert len(identity.implementation_sha256) == 64
    assert identity.create_sha256 != identity.replace_sha256
    assert identity.obligations == ()
    assert sentinel.read_bytes() == b"keep"
    assert tuple(tmp_path.iterdir()) == (sentinel,)


def test_probe_rejects_remote_or_unknown_filesystem_without_writing(
    tmp_path: Path,
) -> None:
    durability = _durability()
    volume = durability.VolumeFacts("/", "remote", "fuse.sshfs", True)
    syscalls = _PortablePosixSyscalls(volume)
    identity = durability.PosixDurabilityBackend(syscalls=syscalls).probe(tmp_path)

    assert identity.status == "inconclusive"
    assert identity.obligations
    assert not any(call[0] == "open_exclusive" for call in syscalls.calls)


def test_probe_collision_never_deletes_a_preexisting_probe_named_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    durability = _durability()
    volume = durability.VolumeFacts("/", "device-7", "ext4", False)
    syscalls = _PortablePosixSyscalls(volume)
    backend = durability.PosixDurabilityBackend(syscalls=syscalls)
    monkeypatch.setattr(
        durability.uuid,
        "uuid4",
        lambda: SimpleNamespace(hex="fixed-token"),
    )
    preexisting = (
        tmp_path / ".vfe4-durability-probe-fixed-token-create"
    )
    preexisting.write_bytes(b"user-owned")

    identity = backend.probe(tmp_path)

    assert identity.status == "inconclusive"
    assert preexisting.read_bytes() == b"user-owned"


def test_durability_backed_canonical_json_write_is_byte_stable(
    tmp_path: Path,
) -> None:
    durability = _durability()
    volume = durability.VolumeFacts("/", "device-7", "ext4", False)
    backend = durability.PosixDurabilityBackend(
        syscalls=_PortablePosixSyscalls(volume)
    )
    target = tmp_path / "canonical.json"

    first = durability.create_canonical_json(
        target,
        {"z": 2, "a": [True, None, 1.25]},
        backend=backend,
    )

    assert target.read_bytes() == b'{"a":[true,null,1.25],"z":2}'
    assert first.sha256 == hashlib.sha256(target.read_bytes()).hexdigest()


def test_publish_bytes_creates_then_replaces_with_reopen_verified_identity(
    tmp_path: Path,
) -> None:
    durability = _durability()
    volume = durability.VolumeFacts("/", "device-7", "ext4", False)
    backend = durability.PosixDurabilityBackend(
        syscalls=_PortablePosixSyscalls(volume)
    )
    target = tmp_path / "cache.bin"

    created = backend.publish_bytes(target, b"first")
    replaced = backend.publish_bytes(target, b"second")

    assert created.operation == "exclusive_create"
    assert replaced.operation == "replace"
    assert created.reopen_verified is True
    assert replaced.reopen_verified is True
    assert target.read_bytes() == b"second"


def test_closed_manifest_validates_canonical_recursive_regular_files(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    nested = tmp_path / "nested"
    nested.mkdir()
    root_payload = b"root payload"
    nested_payload = b"nested payload"
    (tmp_path / "root.bin").write_bytes(root_payload)
    (nested / "payload.bin").write_bytes(nested_payload)
    nested_manifest = _closed_manifest_bytes(
        [_entry("payload.bin", nested_payload)]
    )
    (nested / "manifest.json").write_bytes(nested_manifest)
    root_manifest = _closed_manifest_bytes(
        [
            _entry(
                "nested/manifest.json",
                nested_manifest,
                kind="manifest",
            ),
            _entry("root.bin", root_payload),
        ]
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(root_manifest)

    identity = manifest.validate_closed_manifest(manifest_path)

    assert identity.manifest.relative_path == "manifest.json"
    assert tuple(record.relative_path for record in identity.entries) == (
        "nested/manifest.json",
        "nested/payload.bin",
        "root.bin",
    )
    assert all(
        isinstance(record, manifest.ArtifactIntegrityRecord)
        for record in identity.entries
    )
    with pytest.raises(FrozenInstanceError):
        identity.entries[0].size_bytes = 0


@pytest.mark.parametrize(
    "payload",
    [
        b'{\n  "entries": [],\n  "schema_version": "vfe4-closed-manifest-v1"\n}',
        _closed_manifest_bytes([])[:-1]
        + b',"unexpected":true}',
        _closed_manifest_bytes(
            [
                {
                    **_entry("payload.bin", b"x"),
                    "unexpected": True,
                }
            ]
        ),
    ],
)
def test_closed_manifest_rejects_noncanonical_or_open_key_sets(
    tmp_path: Path,
    payload: bytes,
) -> None:
    manifest = _manifest()
    path = tmp_path / "manifest.json"
    path.write_bytes(payload)

    with pytest.raises(manifest.IntegrityValidationError):
        manifest.validate_closed_manifest(path)


def test_closed_manifest_checks_declared_size_before_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest()
    payload = b"payload"
    (tmp_path / "payload.bin").write_bytes(payload)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(
        _closed_manifest_bytes(
            [
                {
                    **_entry("payload.bin", payload),
                    "size_bytes": len(payload) + 1,
                }
            ]
        )
    )
    real_hasher = manifest._sha256_regular_file

    def reject_payload_hash(path: Path, size_bytes: int) -> object:
        if path.name == "payload.bin":
            pytest.fail("hashed payload before declared-size validation")
        return real_hasher(path, size_bytes)

    monkeypatch.setattr(
        manifest,
        "_sha256_regular_file",
        reject_payload_hash,
    )

    with pytest.raises(manifest.IntegrityValidationError, match="size"):
        manifest.validate_closed_manifest(manifest_path)


def test_closed_manifest_rejects_nonregular_entries(tmp_path: Path) -> None:
    manifest = _manifest()
    directory = tmp_path / "payload.bin"
    directory.mkdir()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(
        _closed_manifest_bytes(
            [
                {
                    "kind": "file",
                    "relative_path": "payload.bin",
                    "sha256": "0" * 64,
                    "size_bytes": 0,
                }
            ]
        )
    )

    with pytest.raises(manifest.IntegrityValidationError, match="regular"):
        manifest.validate_closed_manifest(manifest_path)


def test_generic_integrity_modules_have_no_domain_dependencies() -> None:
    durability = _durability()
    manifest = _manifest()
    forbidden = {
        "vfe4.artifacts.readiness",
        "vfe4.artifacts.run_directory",
        "vfe4.checkpoint",
        "vfe4.data",
        "vfe4.evaluation",
        "vfe4.figures",
        "vfe4.generative",
        "vfe4.objective",
        "vfe4.predictive",
        "vfe4.recognition",
        "vfe4.training",
    }
    for module in (durability, manifest):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imports.update(
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        )
        assert not any(
            imported == blocked or imported.startswith(f"{blocked}.")
            for imported in imports
            for blocked in forbidden
        )


def test_task2_public_interfaces_are_exported_from_artifacts_package() -> None:
    artifacts = importlib.import_module("vfe4.artifacts")
    durability = _durability()
    manifest = _manifest()

    expected = {
        "ArtifactIntegrityRecord": manifest.ArtifactIntegrityRecord,
        "DurabilityBackend": durability.DurabilityBackend,
        "PosixDurabilityBackend": durability.PosixDurabilityBackend,
        "WindowsDurabilityBackend": durability.WindowsDurabilityBackend,
        "probe_durability": durability.probe_durability,
        "validate_closed_manifest": manifest.validate_closed_manifest,
    }
    assert {
        name: getattr(artifacts, name, None) for name in expected
    } == expected
