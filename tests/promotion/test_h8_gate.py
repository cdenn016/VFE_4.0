from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import verification.h8_gate as h8_gate
from vfe4.artifacts import source_candidate_sha256
from vfe4.artifacts.h6 import CandidateArtifactReference
from vfe4.types.h7 import H7PredecessorReference
from vfe4.types.h8 import (
    CurrentH8PrerequisiteRefs,
    H8H1H5Reference,
    H8H1PrefixPriorReference,
    H8H6PredictionReference,
    H8H6PrefixReference,
    H8H7Reference,
)
from vfe4.types.results import GateStatus


def _current_refs(*, registry_sha256: str | None = None) -> CurrentH8PrerequisiteRefs:
    digest = "a" * 64
    head = "1" * 40
    compatibility = {
        key: H7PredecessorReference.create(
            artifact_path=f"{key}-artifact",
            git_head=head,
            dirty_digest=digest,
            junit_sha256=digest,
            junit_path=f"{key}-junit.xml",
            manifest_sha256=digest,
            payload_hashes={f"{key}.json": digest},
            ledger_path=f"{key}-ledger",
            ledger_sha256=digest,
        )
        for key in ("h1_h5", "h1_prefix_prior", "h6_prefix")
    }

    def common(key: str) -> dict[str, object]:
        transitive = compatibility[key]
        return {
            "artifact_path": transitive.artifact_path,
            "manifest_sha256": transitive.manifest_sha256,
            "result_path": f"{key}-result",
            "result_sha256": digest,
            "content_hashes": {f"{key}-content.json": digest},
            "payload_hashes": dict(transitive.payload_hashes),
            "ledger_path": transitive.ledger_path,
            "ledger_sha256": transitive.ledger_sha256,
            "producer_head": transitive.git_head,
            "producer_dirty_digest": transitive.dirty_digest,
            "candidate_junit_sha256": transitive.junit_sha256,
            "status": "pass",
        }

    h7_common: dict[str, object] = {
        "artifact_path": "h7-artifact",
        "manifest_sha256": digest,
        "result_path": "h7-result",
        "result_sha256": digest,
        "content_hashes": {"h7-content.json": digest},
        "payload_hashes": {"h7.json": digest},
        "ledger_path": "h7-ledger",
        "ledger_sha256": digest,
        "producer_head": head,
        "producer_dirty_digest": digest,
        "candidate_junit_sha256": digest,
        "status": "pass",
    }
    prediction_common = {
        **h7_common,
        "artifact_path": "prediction-artifact",
        "result_path": "prediction-result",
        "content_hashes": {"prediction-content.json": digest},
        "payload_hashes": {"prediction.json": digest},
        "ledger_path": "prediction-ledger",
        "candidate_junit_sha256": digest,
    }
    return CurrentH8PrerequisiteRefs(
        candidate_head=head,
        candidate_dirty_digest=digest,
        candidate_junit_sha256=digest,
        h7_compatibility_refs=compatibility,
        h1_h5=H8H1H5Reference(
            kind="h1_h5", **common("h1_h5")  # type: ignore[arg-type]
        ),
        h1_prefix_prior=H8H1PrefixPriorReference(
            kind="h1_prefix_prior",
            **common("h1_prefix_prior"),  # type: ignore[arg-type]
        ),
        h6_prefix=H8H6PrefixReference(
            kind="h6_prefix",
            certificate_set_sha256=digest,
            certificate_hashes={"certificate.json": digest},
            **common("h6_prefix"),  # type: ignore[arg-type]
        ),
        h7=H8H7Reference(
            kind="h7",
            result_pointer_path="h7-result-pointer",
            result_pointer_sha256=digest,
            fixture_set_sha256=digest,
            **h7_common,  # type: ignore[arg-type]
        ),
        h6_prediction=H8H6PredictionReference(
            kind="h6_prediction",
            prediction_schema="h6-prediction-amended-v2",
            config_schema="h6-prediction-config-v2",
            readiness_schema="h6-prediction-readiness-v2",
            metrics_schema="h6-prediction-metrics-v2",
            result_schema="h6-prediction-result-v2",
            experiment_sha256=digest,
            config_sha256=digest,
            readiness_artifact_path="prediction-readiness",
            readiness_manifest_sha256=digest,
            readiness_sha256=digest,
            correctness_artifact_paths={
                gate: f"prediction-{gate.lower()}-correctness"
                for gate in ("H1", "H2", "H3", "H5")
            },
            h1_prefix_prior_artifact_path="prediction-h1-prefix-prior",
            smc_accuracy_artifact_path="prediction-smc-accuracy",
            smc_accuracy_manifest_sha256=digest,
            h6_prefix_artifact_path="prediction-h6-prefix",
            h6_prefix_manifest_sha256=digest,
            blinded_data_artifact_path="prediction-blinded-data",
            blinded_data_manifest_sha256=digest,
            matching_artifact_path="prediction-matching",
            matching_manifest_sha256=digest,
            matching_set_sha256=digest,
            h1_prefix_prior_generative_factor_schema_sha256=digest,
            smc_bias_semantics_sha256=digest,
            objective_gate_spec_sha256=digest,
            metrics_sha256=digest,
            **prediction_common,  # type: ignore[arg-type]
        ),
        registry_sha256=registry_sha256 or digest,
    )


def test_h8_preflight_keeps_optional_hashes_and_failure_dominates() -> None:
    digest = "b" * 64
    result = h8_gate.make_h8_preflight_inconclusive(
        config_sha256=digest,
        obligations=("obtain current H7 and H6-Prediction references",),
        candidate_junit_sha256=None,
        current_refs_registry_sha256=None,
        h7_manifest_sha256=None,
        h6_prediction_manifest_sha256=None,
    )

    assert result.status is GateStatus.INCONCLUSIVE
    assert result.candidate_junit_sha256 is None
    assert result.current_refs_registry_sha256 is None
    assert (
        h8_gate.classify_h8_status(
            retained_statuses=(GateStatus.INCONCLUSIVE, GateStatus.FAIL),
            exact_inventory_complete=False,
            open_obligations=("later evidence is unavailable",),
        )
        is GateStatus.FAIL
    )


def test_h8_prerequisite_reopen_is_fail_closed_without_reconstructing_refs() -> None:
    refs = _current_refs()

    validation = h8_gate.validate_h8_prerequisite_artifacts(refs)
    obligations = validation.obligations

    assert obligations == (
        "h8_prerequisite_h7_transitive_junit_artifact_revalidation_required",
        "h8_prerequisite_h1_h5_immutable_artifact_revalidation_required",
        "h8_prerequisite_h1_prefix_prior_immutable_artifact_revalidation_required",
        "h8_prerequisite_h6_prefix_immutable_artifact_revalidation_required",
        "h8_prerequisite_h7_immutable_artifact_revalidation_required",
        "h8_prerequisite_h6_prediction_v2_artifact_revalidation_required",
    )
    evaluation = h8_gate.assemble_h8_gate_evaluation(
        config_sha256="b" * 64,
        current_refs=refs,
        correctness=(),
        production_runs=(),
        profiler_runs=(),
        controls=(),
        dependency_closure_sha256="c" * 64,
        preregistration_sha256="d" * 64,
        prerequisite_validation=validation,
    )
    payload = h8_gate.h8_validation_payload(evaluation)
    assert payload["status"] == "inconclusive"
    assert payload["prerequisites"]["obligations"] == list(obligations)
    assert payload["prerequisites"]["all_current_and_pass"] is False
    assert payload["invariants"]["prerequisites_current_and_pass"] is False


def test_h8_payload_inventories_are_exact_and_private() -> None:
    refs = _current_refs()
    config_bytes = b'{"operation":"H8"}'
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    evaluation = h8_gate.assemble_h8_gate_evaluation(
        config_sha256=config_sha256,
        current_refs=refs,
        correctness=(),
        production_runs=(),
        profiler_runs=(),
        controls=(),
        dependency_closure_sha256="c" * 64,
        preregistration_sha256="d" * 64,
    )
    validation = h8_gate.h8_validation_payload(evaluation)
    config = SimpleNamespace(
        canonical_json=config_bytes.decode("ascii"),
        config_sha256=config_sha256,
        h8_current_refs=refs,
    )
    payloads = h8_gate.build_h8_publication_payloads(
        config,
        evaluation,
        h7_reference=refs.h7,
        h6_prediction_reference=refs.h6_prediction,
    )

    assert tuple(validation) == h8_gate.H8_VALIDATION_TOP_LEVEL_KEYS
    assert tuple(payloads) == h8_gate.H8_PUBLICATION_PAYLOAD_KEYS
    assert "manifest.sha256" not in payloads
    assert "run_h8_child" not in vars(h8_gate)


def test_h8_external_pointer_round_trips_all_reference_variants(
    tmp_path: Path,
) -> None:
    base_refs = _current_refs()
    registry_bytes = h8_gate.canonical_h8_json_bytes(
        h8_gate.h8_current_refs_registry_payload(base_refs)
    )
    refs = replace(
        base_refs,
        registry_sha256=hashlib.sha256(registry_bytes).hexdigest(),
    )
    assert registry_bytes == h8_gate.canonical_h8_json_bytes(
        h8_gate.h8_current_refs_registry_payload(refs)
    )
    refs_root = tmp_path / ".verification"
    refs_root.mkdir()
    (refs_root / f"h8-current-candidate-{refs.candidate_head}-refs.json").write_bytes(
        registry_bytes
    )
    config_payload = {
        "validation": {"gates": list(h8_gate.H8_VERIFIER_PREFIX)},
        "h8": {
            "h7_plan_sha256": h8_gate.H8_H7_PLAN_SHA256,
            "interpretation_sha256": h8_gate.H8_INTERPRETATION_SHA256,
        },
        "h8_current_refs": json.loads(
            h8_gate.canonical_h8_json_bytes(refs)
        ),
    }
    config_sha256 = hashlib.sha256(
        h8_gate.canonical_h8_json_bytes(config_payload)
    ).hexdigest()
    evaluation = h8_gate.assemble_h8_gate_evaluation(
        config_sha256=config_sha256,
        current_refs=refs,
        correctness=(),
        production_runs=(),
        profiler_runs=(),
        controls=(),
        dependency_closure_sha256="c" * 64,
        preregistration_sha256="d" * 64,
    )
    validation_payload = h8_gate.h8_validation_payload(evaluation)
    validation_sha256 = evaluation.validation_payload_sha256
    source_sha256 = source_candidate_sha256(
        git_head_value=refs.candidate_head,
        dirty_digest_value=refs.candidate_dirty_digest,
    )
    payload_values = {
        "config.json": config_payload,
        "provenance.json": {
            "schema_version": "vfe4-h8-provenance-v1",
            "git_head": refs.candidate_head,
            "dirty_digest": refs.candidate_dirty_digest,
            "dirty_content_digest": refs.candidate_dirty_digest,
            "source_sha256": source_sha256,
            "config_sha256": config_sha256,
            "junit_sha256": refs.candidate_junit_sha256,
            "current_refs_registry_sha256": refs.registry_sha256,
            "reference_registry": {
                "path": (
                    refs_root
                    / f"h8-current-candidate-{refs.candidate_head}-refs.json"
                ).resolve(strict=False).as_posix(),
                "sha256": refs.registry_sha256,
            },
            "dependency_closure_sha256": evaluation.dependency_closure_sha256,
            "preregistration_sha256": evaluation.preregistration_sha256,
            "interpretation_sha256": evaluation.interpretation_sha256,
            "validation_sha256": validation_sha256,
            "evaluation_sha256": evaluation.evaluation_sha256,
            "status": evaluation.result.status.value,
            "obligations": list(evaluation.result.obligations),
            "selected_operation": "H8",
            "ordered_gates": list(h8_gate.H8_VERIFIER_PREFIX),
            "execution_scope": "source-only-empty-runtime-records",
            "external_pointer_in_artifact": False,
            "started_utc": "2026-07-23T00:00:00.000000Z",
            "ended_utc": "2026-07-23T00:00:00.000001Z",
        },
        "environment.json": validation_payload["environment"],
        "references/h7.json": json.loads(
            h8_gate.canonical_h8_json_bytes(refs.h7)
        ),
        "references/h6_prediction.json": json.loads(
            h8_gate.canonical_h8_json_bytes(refs.h6_prediction)
        ),
        "validation/h8.json": validation_payload,
    }
    artifact_root = tmp_path / "artifact"
    payload_hashes: dict[str, str] = {}
    for name in h8_gate.H8_PUBLICATION_PAYLOAD_KEYS:
        payload_bytes = h8_gate.canonical_h8_json_bytes(payload_values[name])
        payload_path = artifact_root / Path(*name.split("/"))
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        payload_path.write_bytes(payload_bytes)
        payload_hashes[name] = hashlib.sha256(payload_bytes).hexdigest()
    manifest_bytes = "".join(
        f"{payload_hashes[name]}  {name}\n"
        for name in h8_gate.H8_PUBLICATION_PAYLOAD_KEYS
    ).encode("ascii")
    (artifact_root / "manifest.sha256").write_bytes(manifest_bytes)
    assert payload_hashes["config.json"] == config_sha256
    assert payload_hashes["validation/h8.json"] == validation_sha256
    artifact = CandidateArtifactReference(
        artifact_path=artifact_root,
        git_head=refs.candidate_head,
        dirty_digest=refs.candidate_dirty_digest,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        payload_hashes=payload_hashes,
    )
    pointer = h8_gate.h8_current_candidate_result_payload(
        artifact,
        repo_root=tmp_path,
        config_sha256=config_sha256,
        validation_sha256=validation_sha256,
        junit_sha256=refs.candidate_junit_sha256,
        current_refs=refs,
        evaluation=evaluation,
        source_sha256=source_sha256,
        registry_path=(
            refs_root
            / f"h8-current-candidate-{refs.candidate_head}-refs.json"
        ),
        registry_bytes=registry_bytes,
    )

    assert tuple(pointer) == (
        "schema_version",
        "candidate",
        "artifact",
        "current_refs",
        "predecessors",
    )
    for name in (
        "h1_h5",
        "h1_prefix_prior",
        "h6_prefix",
        "h7",
        "h6_prediction",
    ):
        assert json.loads(
            h8_gate.canonical_h8_json_bytes(getattr(refs, name))
        ) == pointer["predecessors"][name]
