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


class _BoundedReader:
    def __init__(
        self,
        path: Path,
        calls: list[tuple[object, ...]],
        *,
        fail: bool,
    ) -> None:
        self._handle = path.open("rb", buffering=0)
        self._calls = calls
        self._fail = fail

    def __enter__(self) -> _BoundedReader:
        return self

    def __exit__(self, *args: object) -> None:
        self._handle.close()

    def read(self, size: int = -1) -> bytes:
        self._calls.append(("read_stream", size))
        if self._fail:
            raise OSError(5, "injected streamed reopen failure")
        return self._handle.read(size)


class _PortablePosixSyscalls:
    def __init__(self, volume_facts: object) -> None:
        self.volume_facts_value = volume_facts
        self.calls: list[tuple[object, ...]] = []
        self.fail_flush = False
        self.fail_stream_write = False
        self.fail_stream_reopen = False
        self.fail_stream_promotion = False
        self.raise_after_replace = False

    def volume_facts(self, path: Path) -> object:
        self.calls.append(("volume_facts", path))
        return self.volume_facts_value

    def open_exclusive(self, path: Path, flags: int, mode: int) -> BinaryIO:
        self.calls.append(("open_exclusive", path, flags, mode))
        return path.open("xb", buffering=0)

    def write_all(self, handle: BinaryIO, payload: bytes) -> None:
        self.calls.append(("write_all", payload))
        if self.fail_stream_write:
            raise OSError(5, "injected streamed write failure")
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

    def open_regular_read(self, path: Path) -> _BoundedReader:
        self.calls.append(("open_regular_read", path))
        return _BoundedReader(
            path,
            self.calls,
            fail=self.fail_stream_reopen,
        )

    def replace(self, source: Path, destination: Path) -> None:
        self.calls.append(("replace", source, destination))
        os.replace(source, destination)
        if self.raise_after_replace:
            raise OSError(5, "injected post-replace failure")

    def link(self, source: Path, destination: Path) -> None:
        self.calls.append(("link", source, destination))
        if self.fail_stream_promotion:
            raise OSError(5, "injected streamed promotion failure")
        os.link(source, destination)

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
        self.fail_stream_write = False
        self.fail_stream_reopen = False
        self.fail_stream_promotion = False
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
        if self.fail_stream_write:
            raise OSError(5, "injected streamed write failure")
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

    def open_regular_read(self, path: Path) -> _BoundedReader:
        self.calls.append(("open_regular_read", path))
        return _BoundedReader(
            path,
            self.calls,
            fail=self.fail_stream_reopen,
        )

    def move_file_ex(self, source: Path, destination: Path, flags: int) -> None:
        self.calls.append(("move_file_ex", source, destination, flags))
        if self.fail_stream_promotion:
            raise OSError(5, "injected streamed promotion failure")
        if not flags & 0x1 and destination.exists():
            raise FileExistsError(17, "destination exists", destination)
        if not flags & 0x1:
            os.rename(source, destination)
            return
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


def test_verified_identity_can_be_constructed_without_payload_bytes() -> None:
    durability = _durability()
    digest = hashlib.sha256(b"streamed").hexdigest()

    identity = durability.DurableFileIdentity.create_verified(
        operation="content_addressed",
        size_bytes=8,
        sha256=digest,
        volume_identity="volume-identity",
    )

    assert identity.operation == "content_addressed"
    assert identity.size_bytes == 8
    assert identity.sha256 == digest
    assert identity.reopen_verified is True


def test_regular_nonlink_validator_hashes_only_bounded_blocks(
    tmp_path: Path,
) -> None:
    durability = _durability()
    payload = b"0123456789abcdef"
    path = tmp_path / "payload.bin"
    path.write_bytes(payload)
    calls: list[tuple[object, ...]] = []

    facts = durability.validate_regular_nonlink_sha256(
        path,
        expected_size_bytes=len(payload),
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        block_size=3,
        opener=lambda candidate: _BoundedReader(
            candidate,
            calls,
            fail=False,
        ),
    )

    assert facts.size_bytes == len(payload)
    assert facts.sha256 == hashlib.sha256(payload).hexdigest()
    assert calls
    assert all(call[1] == 3 for call in calls if call[0] == "read_stream")


def _stream_backend(
    durability: object,
    backend_kind: str,
    volume: object,
) -> tuple[object, object]:
    if backend_kind == "posix":
        syscalls = _PortablePosixSyscalls(volume)
        return durability.PosixDurabilityBackend(syscalls=syscalls), syscalls
    syscalls = _PortableWindowsSyscalls(volume)
    return durability.WindowsDurabilityBackend(syscalls=syscalls), syscalls


@pytest.mark.parametrize("backend_kind", ["posix", "windows"])
def test_content_addressed_stream_is_chunked_bounded_and_digest_derived(
    tmp_path: Path,
    backend_kind: str,
) -> None:
    durability = _durability()
    volume = durability.VolumeFacts("/", "device-7", "NTFS", False)
    backend, syscalls = _stream_backend(durability, backend_kind, volume)
    chunks = (b"abc", b"de", b"f")
    payload = b"abcdef"
    digest = hashlib.sha256(payload).hexdigest()

    identity = backend.publish_content_addressed_stream(
        tmp_path,
        iter(chunks),
        suffix=".bin",
        chunk_size_limit=3,
        reopen_block_size=2,
    )

    target = tmp_path / f"{digest}.bin"
    assert target.read_bytes() == payload
    assert identity.operation == "content_addressed"
    assert identity.size_bytes == len(payload)
    assert identity.sha256 == digest
    assert identity.volume_identity == volume.identity
    assert [call[1] for call in syscalls.calls if call[0] == "write_all"] == [
        b"abc",
        b"de",
        b"f",
    ]
    assert all(
        call[1] == 2 for call in syscalls.calls if call[0] == "read_stream"
    )
    reopen_paths = [
        call[1] for call in syscalls.calls if call[0] == "open_regular_read"
    ]
    assert len(reopen_paths) == 2
    assert reopen_paths[0].parent == target.parent
    assert reopen_paths[1] == target
    assert not any(call[0] == "read_regular_bytes" for call in syscalls.calls)
    volume_paths = [
        call[1] for call in syscalls.calls if call[0] == "volume_facts"
    ]
    assert volume_paths[0] == target.parent
    assert volume_paths[1].parent == target.parent
    stage_creates = [
        call for call in syscalls.calls if call[0] in {"open_exclusive", "create_file"}
    ]
    assert len(stage_creates) == 1
    assert stage_creates[0][1].parent == target.parent
    if backend_kind == "posix":
        assert len([call for call in syscalls.calls if call[0] == "link"]) == 1
    else:
        move = next(call for call in syscalls.calls if call[0] == "move_file_ex")
        assert move[3] == durability.WINDOWS_MOVEFILE_WRITE_THROUGH


@pytest.mark.parametrize("backend_kind", ["posix", "windows"])
def test_content_addressed_stream_rejects_an_oversized_chunk_before_publication(
    tmp_path: Path,
    backend_kind: str,
) -> None:
    durability = _durability()
    volume = durability.VolumeFacts("/", "device-7", "NTFS", False)
    backend, _ = _stream_backend(durability, backend_kind, volume)
    payload = b"oversized"
    destination = tmp_path / f"{hashlib.sha256(payload).hexdigest()}.bin"

    with pytest.raises(durability.DurabilityOperationError) as captured:
        backend.publish_content_addressed_stream(
            tmp_path,
            (payload,),
            suffix=".bin",
            chunk_size_limit=len(payload) - 1,
            reopen_block_size=3,
        )

    assert captured.value.phase == "validate_stream_chunk"
    assert not destination.exists()


@pytest.mark.parametrize("backend_kind", ["posix", "windows"])
def test_content_addressed_stream_cleans_stage_after_late_oversized_chunk(
    tmp_path: Path,
    backend_kind: str,
) -> None:
    durability = _durability()
    volume = durability.VolumeFacts("/", "device-7", "NTFS", False)
    backend, syscalls = _stream_backend(durability, backend_kind, volume)
    valid = b"valid"
    oversized = b"too-large"

    with pytest.raises(durability.DurabilityOperationError) as captured:
        backend.publish_content_addressed_stream(
            tmp_path,
            (valid, oversized),
            suffix=".bin",
            chunk_size_limit=len(oversized) - 1,
            reopen_block_size=3,
        )

    assert captured.value.phase == "validate_stream_chunk"
    assert [
        call[1] for call in syscalls.calls if call[0] == "write_all"
    ] == [valid]
    stage_creates = [
        call
        for call in syscalls.calls
        if call[0] in {"open_exclusive", "create_file"}
    ]
    assert len(stage_creates) == 1
    assert stage_creates[0][1].name.startswith(".vfe4-stream-stage-")
    assert not tuple(tmp_path.glob(".vfe4-stream-stage-*"))


@pytest.mark.parametrize("backend_kind", ["posix", "windows"])
def test_content_addressed_stream_recovers_idempotently_only_for_exact_target(
    tmp_path: Path,
    backend_kind: str,
) -> None:
    durability = _durability()
    volume = durability.VolumeFacts("/", "device-7", "NTFS", False)
    backend, _ = _stream_backend(durability, backend_kind, volume)
    payload = b"same-content"
    digest = hashlib.sha256(payload).hexdigest()

    first = backend.publish_content_addressed_stream(
        tmp_path,
        (payload,),
        suffix=".raw",
        chunk_size_limit=len(payload),
        reopen_block_size=4,
    )
    second = backend.publish_content_addressed_stream(
        tmp_path,
        (b"same-", b"content"),
        suffix=".raw",
        chunk_size_limit=7,
        reopen_block_size=3,
    )

    assert second == first
    assert (tmp_path / f"{digest}.raw").read_bytes() == payload
    assert not tuple(tmp_path.glob(".vfe4-stream-stage-*"))


@pytest.mark.parametrize("backend_kind", ["posix", "windows"])
def test_content_addressed_stream_never_overwrites_conflicting_target(
    tmp_path: Path,
    backend_kind: str,
) -> None:
    durability = _durability()
    volume = durability.VolumeFacts("/", "device-7", "NTFS", False)
    backend, syscalls = _stream_backend(durability, backend_kind, volume)
    payload = b"intended"
    digest = hashlib.sha256(payload).hexdigest()
    target = tmp_path / f"{digest}.bin"
    target.write_bytes(b"conflict")

    with pytest.raises(durability.DurabilityCollisionError):
        backend.publish_content_addressed_stream(
            tmp_path,
            (payload,),
            suffix=".bin",
            chunk_size_limit=len(payload),
            reopen_block_size=3,
        )

    assert target.read_bytes() == b"conflict"
    assert not any(
        call[0] in {"link", "move_file_ex"} for call in syscalls.calls
    )


@pytest.mark.parametrize("backend_kind", ["posix", "windows"])
@pytest.mark.parametrize(
    ("failure_flag", "expected_phase"),
    [
        ("fail_stream_write", "write_stream"),
        ("fail_stream_reopen", "reopen_staging_stream"),
        ("fail_stream_promotion", "promote_stream"),
    ],
)
def test_content_addressed_stream_reports_injected_failures(
    tmp_path: Path,
    backend_kind: str,
    failure_flag: str,
    expected_phase: str,
) -> None:
    durability = _durability()
    volume = durability.VolumeFacts("/", "device-7", "NTFS", False)
    backend, syscalls = _stream_backend(durability, backend_kind, volume)
    setattr(syscalls, failure_flag, True)
    payload = b"failure-fixture"
    digest = hashlib.sha256(payload).hexdigest()

    with pytest.raises(durability.DurabilityOperationError) as captured:
        backend.publish_content_addressed_stream(
            tmp_path,
            (payload,),
            suffix=".bin",
            chunk_size_limit=len(payload),
            reopen_block_size=4,
        )

    assert captured.value.phase == expected_phase
    assert not (tmp_path / f"{digest}.bin").exists()


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
