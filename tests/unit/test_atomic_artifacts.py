from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from vfe4.artifacts import (
    ArtifactPublicationError,
    build_provenance,
    canonical_json_bytes,
    dirty_content_digest,
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


@pytest.mark.parametrize(
    "run_name",
    [
        ".",
        "..",
        "a/b",
        "a\\b",
        "/absolute",
        "C:\\absolute",
        "C:relative",
        "\\\\server\\share",
        "name/../escape",
        "name.",
        "name ",
        "bad:name",
        "bad?name",
        "bad*name",
        "bad|name",
        "NUL",
    ],
)
def test_publish_run_rejects_noncanonical_or_escaping_run_names(
    tmp_path: Path, run_name: str
) -> None:
    with pytest.raises(ArtifactPublicationError, match="run_name"):
        publish_run_directory(tmp_path / "runs", run_name, {"config.json": {}})


def test_publish_run_wraps_path_resolution_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        Path, "resolve", lambda self: (_ for _ in ()).throw(OSError("resolve failure"))
    )

    with pytest.raises(ArtifactPublicationError, match="resolve failure"):
        publish_run_directory(tmp_path / "runs", "verify-h1", {"config.json": {}})


def test_dirty_digest_is_content_bound_and_excludes_only_control_and_run_descendant(
    tmp_path: Path,
) -> None:
    import subprocess

    repo = tmp_path / "digest repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    tracked = repo / "tracked.txt"
    tracked.write_text("first", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    run_root = repo / "runs"

    baseline = dirty_content_digest(repo, run_root)
    assert len(baseline) == 64 and baseline != "0" * 64
    tracked.write_text("second", encoding="utf-8")
    changed = dirty_content_digest(repo, run_root)
    assert changed != baseline

    control = repo / ".verification" / "control.json"
    control.parent.mkdir()
    control.write_text("ignored control", encoding="utf-8")
    assert dirty_content_digest(repo, run_root) == changed
    artifact = run_root / "run" / "result.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("ignored artifact", encoding="utf-8")
    assert dirty_content_digest(repo, run_root) == changed

    relevant = repo / "relevant.txt"
    relevant.write_text("included", encoding="utf-8")
    assert dirty_content_digest(repo, run_root) != changed


def test_dirty_digest_rejects_repo_or_ancestor_as_exclusion_root(tmp_path: Path) -> None:
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    for unsafe in (repo, repo.parent):
        with pytest.raises(ArtifactPublicationError, match="strict descendant"):
            dirty_content_digest(repo, unsafe)


def test_dirty_digest_never_excludes_tracked_source_beneath_run_root(
    tmp_path: Path,
) -> None:
    import subprocess

    repo = tmp_path / "repo"
    run_root = repo / "vfe4"
    run_root.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    source = run_root / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "vfe4/source.py"], cwd=repo, check=True)

    baseline = dirty_content_digest(repo, run_root)
    source.write_text("VALUE = 2\n", encoding="utf-8")
    changed = dirty_content_digest(repo, run_root)
    assert changed != baseline

    artifact = run_root / "untracked-run.json"
    artifact.write_text("publication", encoding="utf-8")
    assert dirty_content_digest(repo, run_root) == changed


@pytest.mark.parametrize(
    ("gate_state", "observed"),
    [
        ("pass", None),
        ("fail", "b" * 64),
        ("pass", "not-a-sha"),
    ],
)
def test_provenance_rejects_untruthful_closed_fixture_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gate_state: str,
    observed: str | None,
) -> None:
    import vfe4.artifacts.provenance as provenance

    monkeypatch.setattr(provenance, "git_head", lambda root: "c" * 40)
    monkeypatch.setattr(provenance, "dirty_content_digest", lambda root, runs: "d" * 64)
    config = SimpleNamespace(
        objective_schema_version="vfe4-state-elbo-v1",
        config_sha256="e" * 64,
        artifacts=SimpleNamespace(run_root=tmp_path / "runs"),
        run=SimpleNamespace(
            device="cpu", dtype="float64", seed=1, deterministic=True
        ),
    )

    with pytest.raises(ArtifactPublicationError, match="fixture"):
        build_provenance(
            repo_root=tmp_path,
            fixture_expected_sha256="a" * 64,
            fixture_observed_sha256=observed,
            config=config,
            started_utc="2026-07-21T00:00:00Z",
            ended_utc="2026-07-21T00:00:01Z",
            gate_state=gate_state,
        )


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
