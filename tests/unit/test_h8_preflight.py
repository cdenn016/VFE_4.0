from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from verification.h8_preflight import (
    H8PreflightResult,
    canonical_h8_preflight_bytes,
    inspect_h8_preflight,
)


def _request() -> dict[str, object]:
    return {
        "schema_version": "h8-preflight-config-v1",
        "operation": "H8-Preflight",
        "target_operation": "h8",
        "inspection_policy": "metadata_only",
        "write_artifact": False,
    }


def _candidate() -> dict[str, str]:
    return {
        "git_head": "1" * 40,
        "dirty_digest": "2" * 64,
        "source_sha256": "3" * 64,
    }


def _target() -> dict[str, object]:
    import verify_vfe4

    return copy.deepcopy(verify_vfe4.CONFIG["operations"]["h8"]["config"])


def _state(result: H8PreflightResult, name: str) -> str:
    return next(item.state for item in result.prerequisites if item.name == name)


def _registry_payload(
    tmp_path: Path,
    *,
    schema_version: str = "h8-current-candidate-refs-v3",
    stale: bool = False,
) -> dict[str, object]:
    candidate = _candidate()
    junit = tmp_path / "candidate-junit.xml"
    junit.write_bytes(b'<testsuites tests="1" failures="0"/>\n')
    junit_sha256 = hashlib.sha256(junit.read_bytes()).hexdigest()
    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}\n", encoding="utf-8")
    evidence_path = evidence.as_posix()
    junit_path = junit.as_posix()
    registry_head = "9" * 40 if stale else candidate["git_head"]
    registry_dirty = "8" * 64 if stale else candidate["dirty_digest"]

    compatibility: dict[str, dict[str, object]] = {}
    for key in ("h1_h5", "h1_prefix_prior", "h6_prefix"):
        semantic = {
            "artifact_path": evidence_path,
            "git_head": registry_head,
            "dirty_digest": registry_dirty,
            "junit_sha256": junit_sha256,
            "junit_path": junit_path,
            "manifest_sha256": "4" * 64,
            "payload_hashes": {"validation/result.json": "5" * 64},
            "ledger_path": evidence_path,
            "ledger_sha256": "6" * 64,
        }
        semantic["reference_sha256"] = hashlib.sha256(
            b"vfe4.h7.predecessor-reference.v1\x00"
            + canonical_h8_preflight_bytes(semantic)
        ).hexdigest()
        compatibility[key] = semantic

    def common(key: str) -> dict[str, object]:
        return {
            "kind": key,
            "artifact_path": evidence_path,
            "manifest_sha256": "4" * 64,
            "result_path": evidence_path,
            "result_sha256": "4" * 64,
            "content_hashes": {"validation/result.json": "5" * 64},
            "payload_hashes": {"validation/result.json": "5" * 64},
            "ledger_path": evidence_path,
            "ledger_sha256": "6" * 64,
            "status": "pass",
            "producer_head": registry_head,
            "producer_dirty_digest": registry_dirty,
            "candidate_junit_sha256": junit_sha256,
        }

    h6_prediction = {
        **common("h6_prediction"),
        "prediction_schema": "h6-prediction-amended-v2",
        "config_schema": "h6-prediction-config-v2",
        "readiness_schema": "h6-prediction-readiness-v2",
        "metrics_schema": "h6-prediction-metrics-v2",
        "result_schema": "h6-prediction-result-v2",
        "experiment_sha256": "4" * 64,
        "config_sha256": "4" * 64,
        "readiness_artifact_path": evidence_path,
        "readiness_manifest_sha256": "4" * 64,
        "readiness_sha256": "4" * 64,
        "correctness_artifact_paths": {
            gate: evidence_path for gate in ("H1", "H2", "H3", "H5")
        },
        "h1_prefix_prior_artifact_path": evidence_path,
        "smc_accuracy_artifact_path": evidence_path,
        "smc_accuracy_manifest_sha256": "4" * 64,
        "h6_prefix_artifact_path": evidence_path,
        "h6_prefix_manifest_sha256": "4" * 64,
        "blinded_data_artifact_path": evidence_path,
        "blinded_data_manifest_sha256": "4" * 64,
        "matching_artifact_path": evidence_path,
        "matching_manifest_sha256": "4" * 64,
        "matching_set_sha256": "4" * 64,
        "h1_prefix_prior_generative_factor_schema_sha256": "4" * 64,
        "smc_bias_semantics_sha256": "4" * 64,
        "objective_gate_spec_sha256": "4" * 64,
        "metrics_sha256": "4" * 64,
    }
    if schema_version == "h8-current-candidate-refs-v1":
        h6_prediction = {
            **common("h6_prediction"),
            "experiment_sha256": "4" * 64,
        }
    if schema_version == "h8-current-candidate-refs-v3":
        h6_prefix = {
            **common("h6_prefix"),
            "config_schema": "h6-prefix-config-v3",
            "validation_schema": "h6-prefix-validation-set-v2",
            "certificate_set_schema": "h6-prefix-certificate-set-v2",
            "config_sha256": "4" * 64,
            "workload_plan_sha256": "4" * 64,
            "validation_payload_sha256": "4" * 64,
            "prefix_certificate_set_sha256": "4" * 64,
            "semantic_families": [
                {
                    "semantic_family_index": 0,
                    "semantic_family_sha256": "a" * 64,
                    "validation_payload_sha256": "b" * 64,
                    "certificate_sha256": "c" * 64,
                },
                {
                    "semantic_family_index": 1,
                    "semantic_family_sha256": "d" * 64,
                    "validation_payload_sha256": "e" * 64,
                    "certificate_sha256": "f" * 64,
                },
            ],
        }
    else:
        h6_prefix = {
            **common("h6_prefix"),
            "certificate_set_sha256": "4" * 64,
            "certificate_hashes": {"certificate.json": "5" * 64},
        }
    references = {
        "h1_h5": common("h1_h5"),
        "h1_prefix_prior": common("h1_prefix_prior"),
        "h6_prefix": h6_prefix,
        "h7": {
            **common("h7"),
            "result_pointer_path": evidence_path,
            "result_pointer_sha256": "4" * 64,
            "fixture_set_sha256": "4" * 64,
        },
        "h6_prediction": h6_prediction,
    }
    return {
        "schema_version": schema_version,
        "candidate": {
            "git_head": registry_head,
            "dirty_digest": registry_dirty,
            "junit_sha256": junit_sha256,
        },
        "h7_compatibility_refs": compatibility,
        "references": references,
    }


def _write_registry(
    tmp_path: Path,
    payload: dict[str, object],
) -> Path:
    verification = tmp_path / ".verification"
    verification.mkdir(exist_ok=True)
    path = verification / f"h8-current-candidate-{_candidate()['git_head']}-refs.json"
    path.write_bytes(canonical_h8_preflight_bytes(payload))
    return path


def test_missing_registry_returns_blocked_exact_forecast_without_writes(
    tmp_path: Path,
) -> None:
    preregistration = (
        tmp_path / "docs" / "preregistrations" / "2026-07-21-h8-sparse-scale.md"
    )
    preregistration.parent.mkdir(parents=True)
    preregistration.write_text("frozen preregistration\n", encoding="utf-8")
    before = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    result = inspect_h8_preflight(
        repository_root=tmp_path,
        target_scientific_config=_target(),
        request=_request(),
        candidate=_candidate(),
    )

    after = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert result.disposition == "blocked"
    assert result.scientific_status == "not_evaluated"
    assert _state(result, "h8_registry_v3") == "missing"
    assert _state(result, "h1_h5") == "missing"
    assert _state(result, "h8_runtime_orchestrator") == "missing"

    correctness = result.workload_forecast["correctness"]
    assert correctness == {
        "cells": 12,
        "sources_per_cell": 3,
        "source_evaluations": 36,
        "retained_source_endpoint_records": 1224,
        "ordered_source_pairs_per_cell": 6,
        "ordered_pair_endpoint_comparisons": 2448,
        "wrong_path_control_decisions": 72,
    }
    resources = result.resource_forecast
    assert resources["children"] == {
        "production": 15,
        "profiler": 3,
        "isolated_allocation_controls": 12,
        "total": 30,
        "retries": 0,
    }
    assert resources["layout"] == {"N": 129, "b": 40, "D": 5160}
    assert resources["storage"] == {
        "information_vector": {"scalars": 5160, "bytes": 41280},
        "precision": {"scalars": 411200, "bytes": 3289600},
        "factor": {"scalars": 411200, "bytes": 3289600},
        "selected_inverse": {"scalars": 411200, "bytes": 3289600},
        "maximum_local_workspace": {"scalars": 1600, "bytes": 12800},
        "forbidden_dense_population": {
            "scalars": 26625600,
            "bytes": 213004800,
        },
    }
    assert resources["sequential_resource_child_ceiling_seconds"] == 1800.0
    assert resources["estimated_total_wall_seconds"] is None
    assert resources["measured_runtime_seconds"] is None
    assert resources["measured_memory_mib"] is None
    assert result.execution_policy["scientific_children_launched"] == 0
    assert result.execution_policy["artifact_writes"] == 0
    with pytest.raises(TypeError):
        result.resource_forecast["layout"]["N"] = 1  # type: ignore[index]

    unhashed = result.as_dict(include_result_sha256=False)
    assert (
        result.result_sha256
        == hashlib.sha256(canonical_h8_preflight_bytes(unhashed)).hexdigest()
    )
    assert json.loads(canonical_h8_preflight_bytes(result.as_dict()))


def test_current_v3_registry_is_present_but_never_scientific_pass(
    tmp_path: Path,
) -> None:
    preregistration = (
        tmp_path / "docs" / "preregistrations" / "2026-07-21-h8-sparse-scale.md"
    )
    preregistration.parent.mkdir(parents=True)
    preregistration.write_text("frozen preregistration\n", encoding="utf-8")
    registry_path = _write_registry(tmp_path, _registry_payload(tmp_path))
    from verification.run_gates import parse_h8_reference_registry_bytes

    parsed = parse_h8_reference_registry_bytes(registry_path.read_bytes())
    assert parsed.registry_schema_version == "h8-current-candidate-refs-v3"
    runtime_root = tmp_path / "verification"
    runtime_root.mkdir()
    (runtime_root / "run_gates.py").write_text(
        "def run_h8_verification():\n"
        "    return assemble_h8_gate_evaluation(\n"
        "        correctness=correctness_records,\n"
        "        production_runs=production_records,\n"
        "        profiler_runs=profiler_records,\n"
        "        controls=control_records,\n"
        "    )\n",
        encoding="utf-8",
    )
    (runtime_root / "h8_gate.py").write_text(
        "def classify_h8_status(*, exact_inventory_complete):\n"
        "    return exact_inventory_complete\n"
        "\n"
        "def assemble_h8_gate_evaluation(*, runtime_sections):\n"
        "    complete = bool(runtime_sections)\n"
        "    return runtime_sections, classify_h8_status(\n"
        "        exact_inventory_complete=complete,\n"
        "    )\n",
        encoding="utf-8",
    )

    result = inspect_h8_preflight(
        repository_root=tmp_path,
        target_scientific_config=_target(),
        request=_request(),
        candidate=_candidate(),
    )

    assert _state(result, "h8_registry_v3") == "present_unvalidated"
    assert _state(result, "candidate_junit") == "present_unvalidated"
    assert _state(result, "h1_h5") == "present_unvalidated"
    assert _state(result, "h1_prefix_prior_v2") == "present_unvalidated"
    assert _state(result, "h6_prefix") == "present_unvalidated"
    assert _state(result, "h6_prediction_v2") == "present_unvalidated"
    assert _state(result, "h7_compatibility_registry") == ("present_unvalidated")
    assert _state(result, "h7") == "present_unvalidated"
    assert _state(result, "h8_runtime_orchestrator") == "present_unvalidated"
    assert _state(result, "h8_complete_runtime_cross_binding") == (
        "present_unvalidated"
    )
    assert result.scientific_status == "not_evaluated"
    assert result.disposition == "metadata_complete_unvalidated"
    assert result.obligations == ()
    assert not hasattr(result, "status")
    assert "pass" not in result.as_dict().values()


def test_registry_v1_malformed_and_stale_are_nonauthorizing(
    tmp_path: Path,
) -> None:
    cases = (
        (
            _registry_payload(
                tmp_path,
                schema_version="h8-current-candidate-refs-v1",
            ),
            "blocked",
            "blocked",
        ),
        ({"schema_version": "broken"}, "malformed", "missing"),
        (_registry_payload(tmp_path, stale=True), "stale", "stale"),
    )
    for payload, expected_state, prediction_state in cases:
        _write_registry(tmp_path, payload)

        result = inspect_h8_preflight(
            repository_root=tmp_path,
            target_scientific_config=_target(),
            request=_request(),
            candidate=_candidate(),
        )

        assert _state(result, "h8_registry_v3") == expected_state
        assert _state(result, "h6_prediction_v2") == prediction_state
        assert result.disposition == "blocked"
        assert result.scientific_status == "not_evaluated"
        assert f"h8_registry_v3:{expected_state}" in result.obligations
        assert f"h6_prediction_v2:{prediction_state}" in result.obligations


def test_request_rejects_artifact_writes(tmp_path: Path) -> None:
    request = _request()
    request["write_artifact"] = True

    with pytest.raises(ValueError, match="write_artifact must be false"):
        inspect_h8_preflight(
            repository_root=tmp_path,
            target_scientific_config=_target(),
            request=request,
            candidate=_candidate(),
        )


def test_target_rejects_any_frozen_h8_protocol_drift(tmp_path: Path) -> None:
    target = _target()
    target["h8"]["max_seconds"] = 61.0

    with pytest.raises(ValueError, match="complete frozen protocol"):
        inspect_h8_preflight(
            repository_root=tmp_path,
            target_scientific_config=target,
            request=_request(),
            candidate=_candidate(),
        )
