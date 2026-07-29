from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


class _RecordingPublisher:
    def __init__(self) -> None:
        self.writes: list[tuple[Path, bytes]] = []
        self.operations: list[str] = []

    def create_exclusive(self, path: Path, payload: bytes) -> None:
        from vfe4.artifacts.durability import DurabilityCollisionError

        if path.exists():
            raise DurabilityCollisionError("fixture collision")
        path.write_bytes(payload)
        self.writes.append((path, payload))
        self.operations.append("create_exclusive")

    def replace_durable(self, path: Path, payload: bytes) -> None:
        path.write_bytes(payload)
        self.writes.append((path, payload))
        self.operations.append("replace_durable")

    def publish_bytes(self, path: Path, payload: bytes) -> None:
        path.write_bytes(payload)
        self.writes.append((path, payload))
        self.operations.append("publish_bytes")


def _unresolved_lock_pair() -> tuple[object, bytes, bytes]:
    from vfe4.artifacts.durability import canonical_json_bytes_generic
    from vfe4.artifacts.environment import (
        LockInputManifest,
        LockRequirement,
        render_dependency_lock,
    )

    requirement = LockRequirement(
        name="tiktoken",
        version="0.12.0",
        environment_marker='python_version >= "3.12"',
        artifact_filename="tiktoken-fixture.whl",
        artifact_url="https://example.invalid/tiktoken-fixture.whl",
        artifact_size_bytes=17,
        artifact_sha256s=("1" * 64,),
        expected_installed_record_sha256=None,
        task13_obligation=(
            "task13_capture_exact_installed_record_sha256:tiktoken"
        ),
    )
    manifest = LockInputManifest.create(
        writer_code_sha256="2" * 64,
        target_python_version="3.12",
        requirements=(requirement,),
    )
    return (
        manifest,
        canonical_json_bytes_generic(manifest) + b"\n",
        render_dependency_lock(manifest),
    )


def _dependency_seam(
    *,
    repository_root: Path,
    backend: object,
    record_sha256: str,
) -> object:
    from vfe4.artifacts.environment import DistributionIdentity

    return SimpleNamespace(
        repository_root=repository_root,
        durability_backend=backend,
        installed_distributions=(
            DistributionIdentity(
                name="tiktoken",
                version="0.12.0",
                record_sha256=record_sha256,
            ),
        ),
    )


def test_exact_publication_is_idempotent_and_never_replaces_bytes(
    tmp_path: Path,
) -> None:
    from vfe4.training import production

    path = tmp_path / "final.json"
    backend = _RecordingPublisher()
    original = b'{"source":"first"}\n'

    production._publish(backend, path, original)
    production._publish(backend, path, original)

    assert path.read_bytes() == original
    assert backend.writes == [(path, original)]
    assert backend.operations == ["create_exclusive"]

    with pytest.raises(
        production.ProductionOperationError,
        match="immutable production artifact differs",
    ):
        production._publish(backend, path, b'{"source":"changed"}\n')

    assert path.read_bytes() == original
    assert backend.writes == [(path, original)]
    assert backend.operations == ["create_exclusive"]


def test_competing_exclusive_create_is_reopened_without_replacement(
    tmp_path: Path,
) -> None:
    from vfe4.artifacts.durability import DurabilityCollisionError
    from vfe4.training import production

    path = tmp_path / "final.json"
    competing = b'{"source":"competing"}\n'

    class _CompetingPublisher(_RecordingPublisher):
        def create_exclusive(self, path: Path, payload: bytes) -> None:
            del payload
            path.write_bytes(competing)
            raise DurabilityCollisionError("injected competing create")

        def publish_bytes(self, path: Path, payload: bytes) -> None:
            # This models the reviewed race: publish_bytes rechecks after the
            # competing create, sees a present target, and replaces it.
            path.write_bytes(competing)
            path.write_bytes(payload)
            self.writes.append((path, payload))
            self.operations.append("publish_bytes")

    backend = _CompetingPublisher()
    with pytest.raises(
        production.ProductionOperationError,
        match="competing exclusive create",
    ):
        production._publish(
            backend,
            path,
            b'{"source":"candidate"}\n',
        )

    assert path.read_bytes() == competing
    assert backend.writes == []
    assert backend.operations == []


def test_dependency_lock_transition_is_one_way_then_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vfe4.artifacts.environment import parse_lock_input_manifest
    from vfe4.training import production

    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    _, manifest_payload, lock_payload = _unresolved_lock_pair()
    manifest_path = repository_root / "requirements-wt103.lock-input.json"
    lock_path = repository_root / "requirements-wt103.lock"
    manifest_path.write_bytes(manifest_payload)
    lock_path.write_bytes(lock_payload)
    backend = _RecordingPublisher()
    monkeypatch.setattr(
        production,
        "_validate_lock_writer_source",
        lambda manifest: manifest.writer_code_sha256,
    )
    dependencies = _dependency_seam(
        repository_root=repository_root,
        backend=backend,
        record_sha256="3" * 64,
    )

    resolved_lock, resolved_sha = production._resolve_dependency_lock(
        dependencies
    )
    resolved_manifest_payload = manifest_path.read_bytes()
    resolved_lock_payload = lock_path.read_bytes()
    resolved_manifest = parse_lock_input_manifest(
        resolved_manifest_payload
    )

    assert resolved_lock == resolved_lock_payload
    assert resolved_sha == hashlib.sha256(resolved_lock_payload).hexdigest()
    assert resolved_manifest.task13_obligations == ()
    assert (
        resolved_manifest.requirements[0].expected_installed_record_sha256
        == "3" * 64
    )
    assert len(backend.writes) == 2
    assert backend.operations == ["replace_durable", "replace_durable"]

    assert production._resolve_dependency_lock(dependencies) == (
        resolved_lock_payload,
        resolved_sha,
    )
    assert len(backend.writes) == 2

    changed_dependencies = _dependency_seam(
        repository_root=repository_root,
        backend=backend,
        record_sha256="4" * 64,
    )
    with pytest.raises(
        production.ProductionOperationError,
        match="resolved dependency lock differs",
    ):
        production._resolve_dependency_lock(changed_dependencies)

    assert manifest_path.read_bytes() == resolved_manifest_payload
    assert lock_path.read_bytes() == resolved_lock_payload
    assert len(backend.writes) == 2
    assert backend.operations == ["replace_durable", "replace_durable"]


def test_dependency_lock_transition_rejects_noncanonical_predecessor_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vfe4.training import production

    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    _, manifest_payload, _ = _unresolved_lock_pair()
    manifest_path = repository_root / "requirements-wt103.lock-input.json"
    lock_path = repository_root / "requirements-wt103.lock"
    manifest_path.write_bytes(manifest_payload)
    tampered = b"# not the reviewed candidate lock\n"
    lock_path.write_bytes(tampered)
    backend = _RecordingPublisher()
    monkeypatch.setattr(
        production,
        "_validate_lock_writer_source",
        lambda manifest: manifest.writer_code_sha256,
    )

    with pytest.raises(
        production.ProductionOperationError,
        match="reviewed unresolved dependency-lock pair differs",
    ):
        production._resolve_dependency_lock(
            _dependency_seam(
                repository_root=repository_root,
                backend=backend,
                record_sha256="3" * 64,
            )
        )

    assert manifest_path.read_bytes() == manifest_payload
    assert lock_path.read_bytes() == tampered
    assert backend.writes == []


def test_dependency_transition_fails_on_changed_pre_replace_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vfe4.training import production

    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    _, manifest_payload, lock_payload = _unresolved_lock_pair()
    manifest_path = repository_root / "requirements-wt103.lock-input.json"
    lock_path = repository_root / "requirements-wt103.lock"
    manifest_path.write_bytes(manifest_payload)
    lock_path.write_bytes(lock_payload)
    backend = _RecordingPublisher()
    monkeypatch.setattr(
        production,
        "_validate_lock_writer_source",
        lambda manifest: manifest.writer_code_sha256,
    )
    regular_nonlink_bytes = production._regular_nonlink_bytes
    lock_reads = 0
    competing = b"# competing lock mutation\n"

    def inject_between_observations(
        path: Path,
        *,
        expected_size: int | None = None,
        expected_sha256: str | None = None,
    ) -> bytes:
        nonlocal lock_reads
        observed = regular_nonlink_bytes(
            path,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
        )
        if path == lock_path:
            lock_reads += 1
            if lock_reads == 2:
                path.write_bytes(competing)
        return observed

    monkeypatch.setattr(
        production,
        "_regular_nonlink_bytes",
        inject_between_observations,
    )

    with pytest.raises(
        production.ProductionOperationError,
        match="changed before the authorized transition",
    ):
        production._resolve_dependency_lock(
            _dependency_seam(
                repository_root=repository_root,
                backend=backend,
                record_sha256="3" * 64,
            )
        )

    assert lock_path.read_bytes() == competing
    assert manifest_path.read_bytes() == manifest_payload
    assert backend.writes == []
    assert backend.operations == []


def test_current_reviewed_lock_pair_supports_the_one_way_transition(
    tmp_path: Path,
) -> None:
    from vfe4.artifacts.environment import (
        DistributionIdentity,
        parse_lock_input_manifest,
    )
    from vfe4.training import production

    source_root = Path(__file__).resolve().parents[2]
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    manifest_name = "requirements-wt103.lock-input.json"
    lock_name = "requirements-wt103.lock"
    manifest_payload = (source_root / manifest_name).read_bytes()
    lock_payload = (source_root / lock_name).read_bytes()
    (repository_root / manifest_name).write_bytes(manifest_payload)
    (repository_root / lock_name).write_bytes(lock_payload)
    manifest = parse_lock_input_manifest(manifest_payload)
    installed = tuple(
        DistributionIdentity(
            name=requirement.name,
            version=requirement.version,
            record_sha256=hashlib.sha256(
                f"fixture-record:{requirement.name}".encode("ascii")
            ).hexdigest(),
        )
        for requirement in manifest.requirements
    )
    backend = _RecordingPublisher()
    dependencies = SimpleNamespace(
        repository_root=repository_root,
        durability_backend=backend,
        installed_distributions=installed,
    )

    resolved_lock, resolved_sha = production._resolve_dependency_lock(
        dependencies
    )
    resolved_manifest = parse_lock_input_manifest(
        (repository_root / manifest_name).read_bytes()
    )

    assert resolved_manifest.task13_obligations == ()
    assert tuple(
        requirement.expected_installed_record_sha256
        for requirement in resolved_manifest.requirements
    ) == tuple(item.record_sha256 for item in installed)
    assert resolved_lock == (repository_root / lock_name).read_bytes()
    assert resolved_sha == hashlib.sha256(resolved_lock).hexdigest()
    assert [path.name for path, _ in backend.writes] == [
        lock_name,
        manifest_name,
    ]
    assert backend.operations == ["replace_durable", "replace_durable"]


def test_bundle_before_record_recovery_requires_the_exact_same_bundle(
    tmp_path: Path,
) -> None:
    from vfe4.training import production

    backend = _RecordingPublisher()
    bundle_path = tmp_path / "cache" / "bundle.json"
    source_path = tmp_path / "tracked" / "source.json"
    bundle_path.parent.mkdir()
    source_path.parent.mkdir()
    bundle = b'{"bundle":"original"}\n'
    source = b'{"source":"original"}\n'

    production._publish(backend, bundle_path, bundle)
    assert not source_path.exists()

    with pytest.raises(
        production.ProductionOperationError,
        match="immutable production artifact differs",
    ):
        production._publish(
            backend,
            bundle_path,
            b'{"bundle":"different-retry"}\n',
        )
    assert bundle_path.read_bytes() == bundle
    assert not source_path.exists()

    production._publish(backend, bundle_path, bundle)
    production._publish(backend, source_path, source)

    assert bundle_path.read_bytes() == bundle
    assert source_path.read_bytes() == source
    assert backend.writes == [
        (bundle_path, bundle),
        (source_path, source),
    ]


def test_final_source_marker_reopens_before_live_or_mutating_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import train_vfe4 as launcher
    from vfe4.training import production

    config = copy.deepcopy(launcher.CONFIG)
    config["training"]["operation"] = "source_lock"
    config["paths"]["cache_root"] = str(tmp_path / "cache")
    config["paths"]["run_root"] = str(tmp_path / "runs")
    config["paths"]["source_record_path"] = str(
        tmp_path / "tracked" / "source-record.json"
    )
    config["paths"]["resume_experiment_plan_path"] = str(
        tmp_path / "runs" / "experiment-plan.json"
    )
    config["authorization"] = launcher.SOURCE_LOCK_AUTHORIZATION
    training, paths, _ = launcher._resolve_launcher(config)
    paths.source_record_path.parent.mkdir()
    paths.source_record_path.write_bytes(b"final marker\n")
    sentinel = object()
    reopen_calls: list[tuple[object, object, Path]] = []

    def reopen(
        *,
        training: object,
        paths: object,
        repository_root: Path,
    ) -> object:
        reopen_calls.append((training, paths, repository_root))
        return sentinel

    monkeypatch.setattr(production, "_reopen_source_lock", reopen)
    monkeypatch.setattr(
        production,
        "_live_source_lock_dependencies",
        lambda: (_ for _ in ()).throw(
            AssertionError("live dependency discovery was reached")
        ),
    )

    assert (
        production.run_source_lock(
            training=training,
            paths=paths,
            dependencies=None,
        )
        is sentinel
    )
    assert reopen_calls == [
        (
            training,
            paths,
            Path(production.__file__).resolve().parents[2],
        )
    ]
    assert not paths.cache_root.exists()


def test_source_lock_execution_lease_excludes_a_competing_process(
    tmp_path: Path,
) -> None:
    from vfe4.training import production

    source_record_path = (tmp_path / "tracked" / "source.json").absolute()
    script = "\n".join(
        (
            "import sys",
            "from pathlib import Path",
            "from vfe4.training.production import "
            "_acquire_source_lock_execution_lease",
            "lease = _acquire_source_lock_execution_lease("
            "Path(sys.argv[1]))",
            "print('READY', flush=True)",
            "sys.stdin.readline()",
            "lease.release()",
        )
    )
    child = subprocess.Popen(
        [sys.executable, "-c", script, str(source_record_path)],
        cwd=Path(__file__).resolve().parents[2],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "READY"
        with pytest.raises(
            production.ProductionOperationError,
            match="another source-lock transaction",
        ):
            production._acquire_source_lock_execution_lease(
                source_record_path
            )
    finally:
        if child.stdin is not None:
            child.stdin.write("\n")
            child.stdin.flush()
        stdout, stderr = child.communicate(timeout=10)
        assert stdout == ""
        assert child.returncode == 0, stderr

    lease = production._acquire_source_lock_execution_lease(
        source_record_path
    )
    lease.release()


def test_source_lock_lease_set_serializes_shared_repository_mutation(
    tmp_path: Path,
) -> None:
    from vfe4.training import production

    repository_root = (tmp_path / "repository").absolute()
    cache_root = (tmp_path / "cache").absolute()
    first_source = (tmp_path / "tracked-a" / "source.json").absolute()
    second_source = (tmp_path / "tracked-b" / "source.json").absolute()
    script = "\n".join(
        (
            "import sys",
            "from pathlib import Path",
            "from vfe4.training.production import "
            "_acquire_source_lock_execution_leases",
            "leases = _acquire_source_lock_execution_leases(tuple("
            "Path(value) for value in sys.argv[1:]))",
            "print('READY', flush=True)",
            "sys.stdin.readline()",
            "[lease.release() for lease in reversed(leases)]",
        )
    )
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            script,
            str(repository_root),
            str(cache_root),
            str(first_source),
        ],
        cwd=Path(__file__).resolve().parents[2],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "READY"
        with pytest.raises(
            production.ProductionOperationError,
            match="another source-lock transaction",
        ):
            production._acquire_source_lock_execution_leases(
                (repository_root, cache_root, second_source)
            )
    finally:
        if child.stdin is not None:
            child.stdin.write("\n")
            child.stdin.flush()
        stdout, stderr = child.communicate(timeout=10)
        assert stdout == ""
        assert child.returncode == 0, stderr

    leases = production._acquire_source_lock_execution_leases(
        (repository_root, cache_root, second_source)
    )
    for lease in reversed(leases):
        lease.release()
