from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import pytest

from vfe4.artifacts import (
    ArtifactPublicationError,
    canonical_json_bytes,
    publish_run_directory,
)


@dataclass(frozen=True)
class _Payload:
    value: float
    mapping: object


def _verify_manifest(run_dir: Path) -> list[str]:
    lines = (run_dir / "manifest.sha256").read_text(encoding="utf-8").splitlines()
    for line in lines:
        digest, relative = line.split("  ", 1)
        assert digest == hashlib.sha256((run_dir / relative).read_bytes()).hexdigest()
        assert "\\" not in relative
    return lines


def test_json_adapter_handles_frozen_records_and_mappingproxy_without_asdict() -> None:
    value = 0.1
    encoded = canonical_json_bytes(
        _Payload(value, MappingProxyType({"z": (2, 1), "a": Path("x/y")}))
    )

    decoded = json.loads(encoded)
    assert float(decoded["value"]).hex() == value.hex()
    assert decoded["mapping"] == {"a": "x/y", "z": [2, 1]}
    assert encoded == canonical_json_bytes(
        _Payload(value, MappingProxyType({"a": Path("x/y"), "z": (2, 1)}))
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_json_adapter_rejects_nonfinite_floats(value: float) -> None:
    with pytest.raises(ArtifactPublicationError, match="nonfinite"):
        canonical_json_bytes({"value": value})


def test_publish_run_is_atomic_and_manifest_is_sorted_and_content_bound(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "root with spaces"
    payloads = {
        "validation/h1.json": {"state": "pass"},
        "environment.json": {"dtype": "float64"},
        "provenance.json": {"git_head": "a" * 40},
        "config.json": {"schema_version": 1},
    }

    run_dir = publish_run_directory(run_root, "verify-h1-frozen", payloads)

    assert run_dir == run_root / "verify-h1-frozen"
    assert sorted(path.relative_to(run_dir).as_posix() for path in run_dir.rglob("*.*")) == [
        "config.json",
        "environment.json",
        "manifest.sha256",
        "provenance.json",
        "validation/h1.json",
    ]
    lines = _verify_manifest(run_dir)
    assert [line.split("  ", 1)[1] for line in lines] == [
        "config.json",
        "environment.json",
        "provenance.json",
        "validation/h1.json",
    ]


def test_publish_run_cleans_staging_and_temporary_files_after_injected_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import vfe4.artifacts.atomic as atomic

    calls = 0
    real_writer = atomic._atomic_write_bytes

    def fail_second(path: Path, content: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected write failure")
        real_writer(path, content)

    monkeypatch.setattr(atomic, "_atomic_write_bytes", fail_second)

    with pytest.raises(ArtifactPublicationError, match="injected write failure"):
        publish_run_directory(
            tmp_path / "runs",
            "verify-h1-frozen",
            {"config.json": {}, "validation/h1.json": {}},
        )

    assert not (tmp_path / "runs" / "verify-h1-frozen").exists()
    assert list((tmp_path / "runs").glob(".*staging*")) == []
    assert list((tmp_path / "runs").rglob("*.tmp")) == []


def test_publish_run_never_overwrites_an_existing_run(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    first = publish_run_directory(root, "verify-h1-frozen", {"config.json": {"n": 1}})
    before = {path.relative_to(first): path.read_bytes() for path in first.rglob("*") if path.is_file()}

    with pytest.raises(ArtifactPublicationError, match="already exists"):
        publish_run_directory(root, "verify-h1-frozen", {"config.json": {"n": 2}})

    assert {path.relative_to(first): path.read_bytes() for path in first.rglob("*") if path.is_file()} == before


def test_publish_run_rejects_manifest_path_and_non_json_payload_names(tmp_path: Path) -> None:
    for payloads in (
        {"manifest.sha256": {}},
        {"payload.txt": {}},
        {"../escape.json": {}},
        {"/absolute.json": {}},
    ):
        with pytest.raises(ArtifactPublicationError):
            publish_run_directory(tmp_path / "runs", "verify-h1-frozen", payloads)


@pytest.mark.parametrize("phase", ["fsync", "final_rename"])
def test_publish_run_cleans_staging_for_durability_and_rename_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    import vfe4.artifacts.atomic as atomic

    if phase == "fsync":
        monkeypatch.setattr(atomic.os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("fsync failure")))
    else:
        monkeypatch.setattr(atomic.os, "rename", lambda source, target: (_ for _ in ()).throw(OSError("rename failure")))

    with pytest.raises(ArtifactPublicationError, match="failure"):
        publish_run_directory(tmp_path / "runs", "verify-h1-frozen", {"config.json": {}})

    assert not (tmp_path / "runs" / "verify-h1-frozen").exists()
    assert list((tmp_path / "runs").glob(".staging-*")) == []


def test_manifest_write_failure_cleans_all_staging_debris(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import vfe4.artifacts.atomic as atomic

    original = atomic._atomic_write_bytes

    def fail_manifest(path: Path, content: bytes) -> None:
        if path.name == "manifest.sha256":
            raise OSError("manifest failure")
        original(path, content)

    monkeypatch.setattr(atomic, "_atomic_write_bytes", fail_manifest)
    with pytest.raises(ArtifactPublicationError, match="manifest failure"):
        publish_run_directory(tmp_path / "runs", "verify-h1-frozen", {"config.json": {}})
    assert list((tmp_path / "runs").iterdir()) == []
