from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import verification.h6_smc_gate as smc_gate
from vfe4.predictive import EstimatorIdentity
from vfe4.types import EstimatorSpec, GateStatus
from vfe4.types.h6 import (
    SmcAccuracyArtifactRef,
    canonical_json_bytes as h6_canonical_json_bytes,
)


_GIT_HEAD = "1" * 40
_DIRTY_DIGEST = "2" * 64
_SOURCE_SHA256 = "3" * 64


def test_finite_smc_fixture_and_oracle_are_validated_once_per_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_path = (
        Path(__file__).resolve().parents[2]
        / "verification"
        / "fixtures"
        / "h6_smc_finite_01.json"
    )
    fixture_bytes = fixture_path.read_bytes()
    validation_calls = 0
    oracle_calls = 0
    real_validation = smc_gate.FiniteSmcFixture.__post_init__
    real_oracle = smc_gate.exact_finite_oracle

    def validate_once(fixture: smc_gate.FiniteSmcFixture) -> None:
        nonlocal validation_calls
        validation_calls += 1
        real_validation(fixture)

    def exact_once(
        fixture: smc_gate.FiniteSmcFixture,
    ) -> smc_gate.ExactFiniteOracle:
        nonlocal oracle_calls
        oracle_calls += 1
        return real_oracle(fixture)

    monkeypatch.setattr(
        smc_gate.FiniteSmcFixture,
        "__post_init__",
        validate_once,
    )
    monkeypatch.setattr(smc_gate, "exact_finite_oracle", exact_once)
    monkeypatch.setattr(
        smc_gate,
        "FINITE_FIXTURE_SHA256",
        (hashlib.sha256(fixture_bytes).hexdigest(),),
    )

    report = smc_gate._run_h6_smc_gate_from_fixture_bytes(
        fixture_snapshots=((fixture_path.name, fixture_bytes),),
        replicate_seeds=(7, 11),
        particle_count=1,
        horizon_limit=1,
    )

    assert report.executed_replicates == 2
    assert report.executed_cells == 4
    assert validation_calls == 1
    assert oracle_calls == 1


def _report(
    *,
    fixture_sha256: tuple[str, ...],
    particle_count: int,
) -> smc_gate.SmcAccuracyReport:
    identity = EstimatorIdentity.from_spec(
        EstimatorSpec.create(
            kind="weighted_smc",
            particle_count=particle_count,
            resampling="systematic_ess_half",
        )
    )
    payload = {
        "gate": "H6-SMC-Accuracy",
        "status": "PASS",
        "validation_path": smc_gate.SMC_VALIDATION_RELATIVE_PATH,
        "fixture_sha256": fixture_sha256,
        "estimator_semantic_sha256": identity.semantic_sha256,
        "estimator_artifact_bytes_sha256": identity.artifact_bytes_sha256,
        "critical_values_sha256": smc_gate.CRITICAL_VALUES_PROTOCOL_SHA256,
        "executed_replicates": 512,
        "executed_cells": 76,
        "particle_count": particle_count,
        "error_trace_sha256": "4" * 64,
        "obligations": (),
    }
    report_sha256 = hashlib.sha256(
        b"vfe4.h6.smc-accuracy-report.v1\x00"
        + h6_canonical_json_bytes(payload)
    ).hexdigest()
    return smc_gate.SmcAccuracyReport(
        **payload,
        report_sha256=report_sha256,
    )


def test_smc_publisher_snapshots_once_and_publishes_direct_readiness_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_root = tmp_path / "repo"
    artifact_root = repository_root / "runs"
    fixture_root = repository_root / "verification" / "fixtures"
    fixture_root.mkdir(parents=True)
    fixture_paths = tuple(
        fixture_root / f"h6_smc_finite_{index:02d}.json"
        for index in range(1, 5)
    )
    fixture_bytes = tuple(
        json.dumps(
            {"fixture_id": f"h6-smc-finite-{index}"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        for index in range(1, 5)
    )
    for path, content in zip(fixture_paths, fixture_bytes, strict=True):
        path.write_bytes(content)

    events: list[str] = []
    read_counts = {path: 0 for path in fixture_paths}
    real_snapshot = smc_gate._read_fixture_bytes

    def source_identity(repo: Path, runs: Path) -> tuple[str, str, str]:
        assert repo == repository_root.resolve()
        assert runs == artifact_root
        events.append("source")
        return _GIT_HEAD, _DIRTY_DIGEST, _SOURCE_SHA256

    def snapshot(path: Path) -> bytes:
        events.append(f"fixture:{path.name}")
        read_counts[path] += 1
        return real_snapshot(path)

    expected_report = _report(
        fixture_sha256=tuple(
            hashlib.sha256(content).hexdigest() for content in fixture_bytes
        ),
        particle_count=256,
    )

    def evaluate(**values: object) -> smc_gate.SmcAccuracyReport:
        events.append("evaluate")
        assert values == {
            "fixture_snapshots": tuple(
                (path.name, content)
                for path, content in zip(
                    fixture_paths, fixture_bytes, strict=True
                )
            ),
            "replicate_seeds": (7, 11),
            "particle_count": 256,
            "horizon_limit": 3,
        }
        return expected_report

    monkeypatch.setattr(smc_gate, "current_source_identity", source_identity)
    monkeypatch.setattr(smc_gate, "_read_fixture_bytes", snapshot)
    monkeypatch.setattr(
        smc_gate,
        "_run_h6_smc_gate_from_fixture_bytes",
        evaluate,
    )

    report, run_directory = smc_gate.publish_h6_smc_accuracy_artifact(
        repository_root=repository_root,
        artifact_root=artifact_root,
        run_name="h6-smc-accuracy-test",
        fixture_paths=fixture_paths,
        replicate_seeds=(7, 11),
        particle_count=256,
        horizon_limit=3,
    )

    assert report is expected_report
    assert events[0] == "source"
    assert all(count == 1 for count in read_counts.values())
    assert run_directory == artifact_root / "h6-smc-accuracy-test"
    assert tuple(
        path.relative_to(run_directory).as_posix()
        for path in sorted(run_directory.rglob("*"))
        if path.is_file()
    ) == (
        "config.json",
        "fixtures/finite_smc.json",
        "manifest.sha256",
        "protocol/estimator.json",
        "validation/h6_smc_accuracy.json",
    )

    config_bytes = (run_directory / "config.json").read_bytes()
    estimator_bytes = (
        run_directory / "protocol" / "estimator.json"
    ).read_bytes()
    fixture_set_bytes = (
        run_directory / "fixtures" / "finite_smc.json"
    ).read_bytes()
    validation_bytes = (
        run_directory / "validation" / "h6_smc_accuracy.json"
    ).read_bytes()
    validation = json.loads(validation_bytes)
    fixture_set = json.loads(fixture_set_bytes)

    assert json.loads(config_bytes) == {
        "critical_values_sha256": smc_gate.CRITICAL_VALUES_PROTOCOL_SHA256,
        "estimator_artifact_bytes_sha256": (
            expected_report.estimator_artifact_bytes_sha256
        ),
        "estimator_semantic_sha256": (
            expected_report.estimator_semantic_sha256
        ),
        "fixture_sha256": list(expected_report.fixture_sha256),
        "horizon_limit": 3,
        "particle_count": 256,
        "replicate_seeds": [7, 11],
        "schema_version": "h6-smc-accuracy-config-v1",
    }
    assert fixture_set == {
        "encoding": "hex",
        "fixtures": [
            {
                "filename": path.name,
                "raw_bytes_hex": content.hex(),
                "raw_sha256": hashlib.sha256(content).hexdigest(),
            }
            for path, content in zip(
                fixture_paths, fixture_bytes, strict=True
            )
        ],
        "schema_version": "h6-finite-smc-fixture-set-v1",
    }
    assert validation["status"] == "pass"
    assert validation["producer_validation"]["status"] == "PASS"
    assert validation["git_head"] == _GIT_HEAD
    assert validation["dirty_digest"] == _DIRTY_DIGEST
    assert validation["source_sha256"] == _SOURCE_SHA256
    assert validation["config_sha256"] == hashlib.sha256(
        config_bytes
    ).hexdigest()
    assert validation["estimator_sha256"] == hashlib.sha256(
        estimator_bytes
    ).hexdigest()
    assert validation["fixture_set_sha256"] == hashlib.sha256(
        fixture_set_bytes
    ).hexdigest()
    assert (
        validation["estimator_semantic_sha256"]
        == expected_report.estimator_semantic_sha256
    )
    assert (
        validation["estimator_sha256"]
        == expected_report.estimator_artifact_bytes_sha256
    )
    assert (
        validation["estimator_semantic_sha256"]
        != validation["estimator_sha256"]
    )

    reference = SmcAccuracyArtifactRef.from_bytes(
        artifact_path=Path(smc_gate.SMC_VALIDATION_RELATIVE_PATH),
        manifest_bytes=(run_directory / "manifest.sha256").read_bytes(),
        git_head=_GIT_HEAD,
        dirty_digest=_DIRTY_DIGEST,
        estimator_preimage_bytes=estimator_bytes,
        fixture_set_bytes=fixture_set_bytes,
        config_bytes=config_bytes,
        validation_payload_bytes=validation_bytes,
    )
    assert reference.status is GateStatus.PASS


def test_existing_smc_gate_api_uses_snapshots_and_preserves_direct_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_root = tmp_path / "repo"
    fixture_path = repository_root / "fixture.json"
    fixture_path.parent.mkdir()
    fixture_bytes = b'{"fixture":"snapshot"}'
    fixture_path.write_bytes(fixture_bytes)
    report = _report(
        fixture_sha256=(hashlib.sha256(fixture_bytes).hexdigest(),),
        particle_count=8,
    )
    snapshots: list[Path] = []
    real_snapshot = smc_gate._read_fixture_bytes

    def snapshot(path: Path) -> bytes:
        snapshots.append(path)
        return real_snapshot(path)

    def evaluate(**values: object) -> smc_gate.SmcAccuracyReport:
        assert values == {
            "fixture_snapshots": ((fixture_path.name, fixture_bytes),),
            "replicate_seeds": (5,),
            "particle_count": 8,
            "horizon_limit": None,
        }
        return report

    monkeypatch.setattr(smc_gate, "_read_fixture_bytes", snapshot)
    monkeypatch.setattr(
        smc_gate,
        "_run_h6_smc_gate_from_fixture_bytes",
        evaluate,
    )
    output_path = (
        repository_root / "validation" / "h6_smc_accuracy.json"
    )

    observed = smc_gate.run_h6_smc_gate(
        fixture_paths=(fixture_path,),
        replicate_seeds=(5,),
        particle_count=8,
        output_path=output_path,
        repository_root=repository_root,
    )

    assert observed is report
    assert snapshots == [fixture_path]
    assert output_path.read_bytes() == report.artifact_bytes()
