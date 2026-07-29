from __future__ import annotations

import copy
import hashlib
import inspect
import json
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_support.h8_runtime_fakes import (
    make_fake_h8_process_record,
    make_pass_correctness_cells,
    make_test_parent_identities,
)

import verification.h8_gate as h8_gate
from vfe4.artifacts import source_candidate_sha256
from vfe4.artifacts.h6 import CandidateArtifactReference
from vfe4.types.h6 import (
    H6_A0_DIRECT_EXACT_PREFIX_REQUIRED_CHECKS,
    A0DirectExactPrefixCertificateV1,
    BoundedPrefixCertificateSet,
    EvidenceStatus,
)
from vfe4.types.h7 import H7PredecessorReference
from vfe4.types.h8 import (
    CurrentH8PrerequisiteRefs,
    H8ChildRequest,
    H8ChildResult,
    H8ControlResult,
    H8GateEvaluation,
    H8H1H5Reference,
    H8H1PrefixPriorReference,
    H8H6PredictionV3Reference,
    H8H6PrefixReference,
    H8H6PrefixSemanticFamilyReference,
    H8H7Reference,
    H8LegacyH6PrefixV4Reference,
)
from vfe4.types.results import GateStatus, H8GateResult


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
        "content_hashes": {"result.json": digest},
        "payload_hashes": {
            "metrics.json": digest,
            "raw_inventory.json": digest,
            "result.json": digest,
        },
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
            config_schema="h6-prefix-config-v3",
            validation_schema="h6-prefix-validation-set-v2",
            certificate_set_schema="h6-prefix-certificate-set-v2",
            config_sha256=digest,
            workload_plan_sha256="b" * 64,
            validation_payload_sha256="c" * 64,
            prefix_certificate_set_sha256="d" * 64,
            a0_direct_exact_prefix_certificate_sha256="5" * 64,
            semantic_families=(
                H8H6PrefixSemanticFamilyReference(
                    semantic_family_index=0,
                    semantic_family_sha256="e" * 64,
                    validation_payload_sha256="f" * 64,
                    certificate_sha256="1" * 64,
                ),
                H8H6PrefixSemanticFamilyReference(
                    semantic_family_index=1,
                    semantic_family_sha256="2" * 64,
                    validation_payload_sha256="3" * 64,
                    certificate_sha256="4" * 64,
                ),
            ),
            **common("h6_prefix"),  # type: ignore[arg-type]
        ),
        h7=H8H7Reference(
            kind="h7",
            result_pointer_path="h7-result-pointer",
            result_pointer_sha256=digest,
            fixture_set_sha256=digest,
            **h7_common,  # type: ignore[arg-type]
        ),
        h6_prediction=H8H6PredictionV3Reference(
            kind="h6_prediction",
            config_schema="h6-prediction-config-v3",
            readiness_schema="h6-prediction-readiness-v3",
            raw_inventory_schema="h6-raw-endpoint-inventory-v4",
            metrics_schema="h6-prediction-metrics-v3",
            result_schema="h6-prediction-result-v3",
            authorities_path="prediction-authorities",
            authorities_manifest_sha256=digest,
            authorities_sha256=digest,
            config_sha256=digest,
            readiness_sha256=digest,
            plan_sha256=digest,
            matching_set_sha256=digest,
            validation_bundle_path="prediction-validation",
            validation_bundle_manifest_sha256=digest,
            validation_bundle_sha256=digest,
            checkpoint_selection_sha256=digest,
            reservation_path="prediction-reservation.json",
            reservation_sha256=digest,
            reservation_file_sha256=digest,
            terminal_path="prediction-terminal",
            terminal_sha256=digest,
            terminal_manifest_sha256=digest,
            finalized_path="prediction-finalized",
            finalized_manifest_sha256=digest,
            pointer_path="prediction-pointer",
            pointer_sha256=digest,
            pointer_manifest_sha256=digest,
            experiment_identity_sha256=digest,
            opening_proof_sha256=digest,
            raw_inventory_sha256=digest,
            metrics_sha256=digest,
            result_record_sha256=digest,
            ledger_validator_sha256=digest,
            artifact_revision=f"git:{head}:sha256:{digest}",
            candidate_junit_path="prediction-junit.xml",
            **prediction_common,  # type: ignore[arg-type]
        ),
        registry_sha256=registry_sha256 or digest,
    )


def _timeout_parent_authority(tmp_path: Path, refs: CurrentH8PrerequisiteRefs):
    from verification.h8_budget import H8ChildProcessRecord
    from verification.h8_orchestrator import (
        _mint_h8_parent_attempt_authority,
        _run_h8_parent_attempt_for_test,
        derive_h8_child_start_authorization,
    )
    from vfe4.config.schema import H8ValidationConfig

    prerequisite_validation = h8_gate.H8PrerequisiteArtifactValidation.create(
        registry_sha256=refs.registry_sha256,
        revalidated_reference_names=h8_gate.H8_POINTER_PREDECESSOR_KEYS,
        obligations=(),
    )
    authorization = derive_h8_child_start_authorization(
        config=H8ValidationConfig.create(),
        current_registry_sha256=refs.registry_sha256,
        prerequisite_validation=prerequisite_validation,
        correctness_statuses=tuple(
            (cell_id, GateStatus.PASS) for cell_id in range(1, 13)
        ),
    )
    identities = make_test_parent_identities()

    def timed_out_child(_invocation):
        return H8ChildProcessRecord(
            timed_out=True,
            exit_code=None,
            stdout=b"",
            stderr=b"fake timeout",
            parent_elapsed_ns=60_000_000_001,
        )

    parent_run = _run_h8_parent_attempt_for_test(
        authorization=authorization,
        repository_root=tmp_path,
        identities=identities,
        child_runner=timed_out_child,
    )
    return (
        _mint_h8_parent_attempt_authority(parent_run),
        prerequisite_validation,
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


def test_h8_gate_retains_parent_attempt_failure_without_a_child_result(
    tmp_path: Path,
) -> None:
    from verification.h8_budget import (
        H8ChildProcessRecord,
        build_h8_child_invocation,
        classify_h8_child_outcome,
        make_h8_child_attempt_record,
    )
    from vfe4.config.schema import H8ValidationConfig

    refs = _current_refs()
    validation_config = H8ValidationConfig.create()
    request = H8ChildRequest(
        mode="production",
        seed=20260721,
        repetition=0,
        config_sha256=validation_config.config_sha256,
        protocol_sha256="c" * 64,
        control_id=None,
    )
    identities = make_test_parent_identities()
    invocation = build_h8_child_invocation(
        {
            "mode": request.mode,
            "seed": request.seed,
            "repetition": request.repetition,
            "config_sha256": request.config_sha256,
            "protocol_sha256": request.protocol_sha256,
            "control_id": request.control_id,
        },
        repository_root=tmp_path,
        identities=identities,
        base_environment={},
    )
    process_record = H8ChildProcessRecord(
        timed_out=True,
        exit_code=None,
        stdout=b"",
        stderr=b"fake timeout",
        parent_elapsed_ns=60_000_000_001,
    )
    decision = classify_h8_child_outcome(
        process_record,
        valid_start=True,
        invocation=invocation,
    )
    attempt = make_h8_child_attempt_record(
        request,
        invocation,
        process_record,
        decision,
    )
    prerequisite_validation = h8_gate.H8PrerequisiteArtifactValidation.create(
        registry_sha256=refs.registry_sha256,
        revalidated_reference_names=h8_gate.H8_POINTER_PREDECESSOR_KEYS,
        obligations=(),
    )

    evaluation = h8_gate.assemble_h8_source_only_evaluation(
        config_sha256="b" * 64,
        current_refs=refs,
        correctness=(),
        child_attempts=(attempt,),
        production_runs=(),
        profiler_runs=(),
        controls=(),
        dependency_closure_sha256="c" * 64,
        preregistration_sha256="d" * 64,
        prerequisite_validation=prerequisite_validation,
    )
    payload = h8_gate.h8_validation_payload(evaluation)

    assert evaluation.result.status is GateStatus.FAIL
    assert evaluation.result.obligations == ()
    assert payload["status"] == "fail"
    assert payload["production_runs"] == []
    assert payload["child_attempts"] == [
        {
            "request": {
                "mode": "production",
                "seed": 20260721,
                "repetition": 0,
                "config_sha256": validation_config.config_sha256,
                "protocol_sha256": "c" * 64,
                "control_id": None,
            },
            "status": "fail",
            "reasons": ["child_timeout"],
            "result_kind": None,
            "result_identity": None,
            "nonpass_envelope": None,
            "timed_out": True,
            "exit_code": None,
            "parent_elapsed_ns": 60_000_000_001,
            "request_sha256": attempt.request_sha256,
            "identities_sha256": attempt.identities_sha256,
            "stdout_sha256": attempt.stdout_sha256,
            "stderr_sha256": attempt.stderr_sha256,
            "operation_reachability": None,
            "residuals": None,
            "resource_decisions": None,
        }
    ]


def test_h8_gate_requires_authority_and_keeps_two_config_commitments_distinct(
    tmp_path: Path,
) -> None:
    from vfe4.config.schema import H8ValidationConfig

    refs = _current_refs()
    validation_config = H8ValidationConfig.create()
    publication_config_sha256 = "f" * 64
    authority, prerequisite_validation = _timeout_parent_authority(
        tmp_path,
        refs,
    )

    signature = inspect.signature(h8_gate.assemble_h8_gate_evaluation)
    assert "parent_authority" in signature.parameters
    assert {
        "child_attempts",
        "production_runs",
        "profiler_runs",
        "controls",
    }.isdisjoint(signature.parameters)
    assert authority.child_config_sha256 == validation_config.config_sha256
    assert authority.child_config_sha256 != publication_config_sha256
    assert authority.current_registry_sha256 == refs.registry_sha256
    assert (
        authority.prerequisite_validation_sha256
        == prerequisite_validation.validation_sha256
    )

    with pytest.raises(ValueError, match="correctness"):
        h8_gate.assemble_h8_gate_evaluation(
            publication_config_sha256=publication_config_sha256,
            current_refs=refs,
            correctness=(),
            parent_authority=authority,
            dependency_closure_sha256="c" * 64,
            preregistration_sha256="d" * 64,
            prerequisite_validation=prerequisite_validation,
        )


def test_h8_authority_incomplete_inventory_retains_runtime_pass_locks(
    tmp_path: Path,
) -> None:
    from vfe4.config.schema import H8ValidationConfig
    from verification.h8_budget import H8ChildProcessRecord
    from verification.h8_orchestrator import (
        _mint_h8_parent_attempt_authority,
        _run_h8_parent_attempt_for_test,
        derive_h8_child_start_authorization,
    )

    refs = _current_refs()
    correctness = make_pass_correctness_cells()
    prerequisite_validation = h8_gate.H8PrerequisiteArtifactValidation.create(
        registry_sha256=refs.registry_sha256,
        revalidated_reference_names=h8_gate.H8_POINTER_PREDECESSOR_KEYS,
        obligations=(),
    )
    authorization = derive_h8_child_start_authorization(
        config=H8ValidationConfig.create(),
        current_registry_sha256=refs.registry_sha256,
        prerequisite_validation=prerequisite_validation,
        correctness_statuses=tuple(
            (cell.cell_id, cell.status) for cell in correctness
        ),
    )

    def inconclusive_child(invocation):
        passing = make_fake_h8_process_record(invocation)
        payload = json.loads(passing.stdout)
        payload.update(
            {
                "status": "inconclusive",
                "obligations": ["test-only incomplete child evidence"],
                "result": None,
                "control": None,
            }
        )
        return H8ChildProcessRecord.from_payload(payload)

    parent_run = _run_h8_parent_attempt_for_test(
        authorization=authorization,
        repository_root=tmp_path,
        identities=make_test_parent_identities(),
        child_runner=inconclusive_child,
    )
    authority = _mint_h8_parent_attempt_authority(parent_run)
    evaluation = h8_gate.assemble_h8_gate_evaluation(
        publication_config_sha256="f" * 64,
        current_refs=refs,
        correctness=correctness,
        parent_authority=authority,
        dependency_closure_sha256="c" * 64,
        preregistration_sha256="d" * 64,
        prerequisite_validation=prerequisite_validation,
    )

    assert evaluation.result.status is GateStatus.INCONCLUSIVE
    assert evaluation.runtime_sections is None
    assert set(h8_gate.H8_SOURCE_ONLY_OBLIGATIONS[4:]).issubset(
        evaluation.result.obligations
    )


def _passing_parent_authority(
    tmp_path: Path,
    refs: CurrentH8PrerequisiteRefs,
):
    from vfe4.config.schema import H8ValidationConfig
    from verification.h8_orchestrator import (
        _mint_h8_parent_attempt_authority,
        _run_h8_parent_attempt_for_test,
        derive_h8_child_start_authorization,
    )

    correctness = make_pass_correctness_cells()
    prerequisite_validation = h8_gate.H8PrerequisiteArtifactValidation.create(
        registry_sha256=refs.registry_sha256,
        revalidated_reference_names=h8_gate.H8_POINTER_PREDECESSOR_KEYS,
        obligations=(),
    )
    authorization = derive_h8_child_start_authorization(
        config=H8ValidationConfig.create(),
        current_registry_sha256=refs.registry_sha256,
        prerequisite_validation=prerequisite_validation,
        correctness_statuses=tuple(
            (cell.cell_id, cell.status) for cell in correctness
        ),
    )
    parent_run = _run_h8_parent_attempt_for_test(
        authorization=authorization,
        repository_root=tmp_path,
        identities=make_test_parent_identities(),
        child_runner=make_fake_h8_process_record,
    )
    return (
        _mint_h8_parent_attempt_authority(parent_run),
        prerequisite_validation,
        correctness,
    )


def test_h8_complete_inventories_cannot_clear_locks_without_parent_authority(
    tmp_path: Path,
) -> None:
    refs = _current_refs()
    authority, prerequisite_validation, correctness = _passing_parent_authority(
        tmp_path,
        refs,
    )
    child_attempts = authority.attempts
    production_runs = tuple(
        attempt.result
        for attempt in child_attempts
        if attempt.request.mode == "production"
        and type(attempt.result) is H8ChildResult
    )
    profiler_runs = tuple(
        attempt.result
        for attempt in child_attempts
        if attempt.request.mode == "profiler"
        and type(attempt.result) is H8ChildResult
    )
    controls = tuple(
        attempt.result
        for attempt in child_attempts
        if attempt.request.mode == "negative_control"
        and type(attempt.result) is H8ControlResult
    )
    bypass = getattr(
        h8_gate,
        "_assemble_h8_gate_evaluation_from_inventories",
        None,
    )
    assert bypass is None

    source_only = h8_gate.assemble_h8_source_only_evaluation(
        config_sha256="f" * 64,
        current_refs=refs,
        correctness=correctness,
        child_attempts=child_attempts,
        production_runs=production_runs,
        profiler_runs=profiler_runs,
        controls=controls,
        dependency_closure_sha256="c" * 64,
        preregistration_sha256="d" * 64,
        prerequisite_validation=prerequisite_validation,
    )
    assert source_only.result.status is GateStatus.INCONCLUSIVE
    assert set(h8_gate.H8_SOURCE_ONLY_OBLIGATIONS[4:]).issubset(
        source_only.result.obligations
    )


def test_h8_direct_complete_pass_result_requires_authoritative_factory(
    tmp_path: Path,
) -> None:
    refs = _current_refs()
    authority, prerequisite_validation, correctness = _passing_parent_authority(
        tmp_path,
        refs,
    )
    evaluation = h8_gate.assemble_h8_gate_evaluation(
        publication_config_sha256="f" * 64,
        current_refs=refs,
        correctness=correctness,
        parent_authority=authority,
        dependency_closure_sha256="c" * 64,
        preregistration_sha256="d" * 64,
        prerequisite_validation=prerequisite_validation,
    )
    authoritative = evaluation.result
    assert authoritative.status is GateStatus.PASS

    with pytest.raises(ValueError, match="factory-issued"):
        H8GateResult(
            **{
                field.name: getattr(authoritative, field.name)
                for field in fields(authoritative)
            }
        )


def test_verification_run_result_rejects_shallow_copied_h8_pass(
    tmp_path: Path,
) -> None:
    from verification.run_gates import VerificationRunResult

    refs = _current_refs()
    authority, prerequisite_validation, correctness = _passing_parent_authority(
        tmp_path,
        refs,
    )
    evaluation = h8_gate.assemble_h8_gate_evaluation(
        publication_config_sha256="f" * 64,
        current_refs=refs,
        correctness=correctness,
        parent_authority=authority,
        dependency_closure_sha256="c" * 64,
        preregistration_sha256="d" * 64,
        prerequisite_validation=prerequisite_validation,
    )
    authoritative = evaluation.result
    copied = copy.copy(authoritative)
    assert type(copied) is H8GateResult
    assert copied is not authoritative

    with pytest.raises(ValueError, match="factory-issued"):
        VerificationRunResult((copied,), tmp_path)

    accepted = VerificationRunResult((authoritative,), tmp_path)
    assert accepted.gate_results == (authoritative,)


def test_h8_gate_rejects_parent_authority_replayed_to_another_registry(
    tmp_path: Path,
) -> None:
    source_refs = _current_refs()
    authority, _source_validation = _timeout_parent_authority(
        tmp_path,
        source_refs,
    )
    replay_refs = _current_refs(registry_sha256="e" * 64)
    replay_validation = h8_gate.H8PrerequisiteArtifactValidation.create(
        registry_sha256=replay_refs.registry_sha256,
        revalidated_reference_names=h8_gate.H8_POINTER_PREDECESSOR_KEYS,
        obligations=(),
    )

    with pytest.raises(ValueError, match="registry"):
        h8_gate.assemble_h8_gate_evaluation(
            publication_config_sha256="f" * 64,
            current_refs=replay_refs,
            correctness=(),
            parent_authority=authority,
            dependency_closure_sha256="c" * 64,
            preregistration_sha256="d" * 64,
            prerequisite_validation=replay_validation,
        )


def test_h8_gate_rejects_parent_authority_replayed_with_other_prerequisites(
    tmp_path: Path,
) -> None:
    refs = _current_refs()
    authority, _source_validation = _timeout_parent_authority(tmp_path, refs)
    replay_validation = h8_gate.H8PrerequisiteArtifactValidation.create(
        registry_sha256=refs.registry_sha256,
        revalidated_reference_names=h8_gate.H8_POINTER_PREDECESSOR_KEYS,
        obligations=("synthetic_replay_obligation",),
    )

    with pytest.raises(ValueError, match="prerequisite"):
        h8_gate.assemble_h8_gate_evaluation(
            publication_config_sha256="f" * 64,
            current_refs=refs,
            correctness=(),
            parent_authority=authority,
            dependency_closure_sha256="c" * 64,
            preregistration_sha256="d" * 64,
            prerequisite_validation=replay_validation,
        )


def test_h8_gate_rejects_parent_authority_replayed_with_other_correctness(
    tmp_path: Path,
) -> None:
    refs = _current_refs()
    authority, prerequisite_validation = _timeout_parent_authority(
        tmp_path,
        refs,
    )

    with pytest.raises(ValueError, match="correctness"):
        h8_gate.assemble_h8_gate_evaluation(
            publication_config_sha256="f" * 64,
            current_refs=refs,
            correctness=(),
            parent_authority=authority,
            dependency_closure_sha256="c" * 64,
            preregistration_sha256="d" * 64,
            prerequisite_validation=prerequisite_validation,
        )


def _forge_h8_validation_payload(
    evaluation: H8GateEvaluation,
    payload: dict[str, object],
) -> H8GateEvaluation:
    result = evaluation.result
    assert type(result) is H8GateResult
    payload_bytes = h8_gate.canonical_h8_json_bytes(payload)
    payload_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    return replace(
        evaluation,
        validation_payload_canonical_json=payload_bytes,
        validation_payload_sha256=payload_sha256,
        evaluation_sha256=h8_gate._evaluation_sha256(
            result=result,
            current_refs=evaluation.current_refs,
            prerequisite_obligations=evaluation.prerequisite_obligations,
            runtime_sections=evaluation.runtime_sections,
            validation_payload_sha256=payload_sha256,
            dependency_closure_sha256=evaluation.dependency_closure_sha256,
            preregistration_sha256=evaluation.preregistration_sha256,
        ),
    )


@pytest.mark.parametrize(
    "inventory_name",
    (
        "correctness",
        "child_attempts",
        "controls",
        "production_runs",
        "profiler_runs",
    ),
)
def test_h8_validation_payload_rejects_inventory_drift_from_typed_result(
    inventory_name: str,
) -> None:
    refs = _current_refs()
    evaluation = h8_gate.assemble_h8_source_only_evaluation(
        config_sha256="b" * 64,
        current_refs=refs,
        correctness=(),
        production_runs=(),
        profiler_runs=(),
        controls=(),
        dependency_closure_sha256="c" * 64,
        preregistration_sha256="d" * 64,
    )
    payload = h8_gate.h8_validation_payload(evaluation)
    payload[inventory_name] = [{"unexpected": inventory_name}]
    forged = _forge_h8_validation_payload(evaluation, payload)

    with pytest.raises(ValueError, match="inventories.*typed result"):
        h8_gate.h8_validation_payload(forged)


@pytest.mark.parametrize(
    ("section_name", "field_name", "forged_value"),
    (
        ("bounded_claim", None, h8_gate.H8_BOUNDED_CLAIM),
        ("revision", "git_head", "f" * 40),
        ("config", "selected_operation", "H7"),
        ("prerequisites", "all_current_and_pass", True),
        ("interpretation", "K", 21),
        ("protocol", "factor_schema", "forged-factor-schema"),
        ("environment", "cpu_count", 999),
        ("problems", None, [{"unexpected": "problem"}]),
        ("storage", "h_scalars", 1),
        ("factor", "algorithm", "forged-factor"),
        ("allocation", "all_observable", True),
        ("budgets", "max_seconds", 1.0),
        ("invariants", "all_pass", True),
        ("artifacts", "validation_path", "forged-validation.json"),
    ),
)
def test_h8_validation_payload_rejects_noninventory_context_drift(
    section_name: str,
    field_name: str | None,
    forged_value: object,
) -> None:
    refs = _current_refs()
    evaluation = h8_gate.assemble_h8_source_only_evaluation(
        config_sha256="b" * 64,
        current_refs=refs,
        correctness=(),
        production_runs=(),
        profiler_runs=(),
        controls=(),
        dependency_closure_sha256="c" * 64,
        preregistration_sha256="d" * 64,
    )
    payload = h8_gate.h8_validation_payload(evaluation)
    if field_name is None:
        payload[section_name] = forged_value
    else:
        section = payload[section_name]
        assert isinstance(section, dict)
        section[field_name] = forged_value
    forged = _forge_h8_validation_payload(evaluation, payload)

    with pytest.raises(ValueError, match="exact context reconstruction"):
        h8_gate.h8_validation_payload(forged)


def test_h8_evaluation_rejects_interpretation_context_drift() -> None:
    evaluation = h8_gate.assemble_h8_source_only_evaluation(
        config_sha256="b" * 64,
        current_refs=_current_refs(),
        correctness=(),
        production_runs=(),
        profiler_runs=(),
        controls=(),
        dependency_closure_sha256="c" * 64,
        preregistration_sha256="d" * 64,
    )

    with pytest.raises(ValueError, match="interpretation_sha256"):
        replace(evaluation, interpretation_sha256="f" * 64)


def test_h8_supplied_runtime_context_round_trips_in_declared_schema_order() -> None:
    refs = _current_refs()
    source_only = h8_gate.assemble_h8_source_only_evaluation(
        config_sha256="b" * 64,
        current_refs=refs,
        correctness=(),
        production_runs=(),
        profiler_runs=(),
        controls=(),
        dependency_closure_sha256="c" * 64,
        preregistration_sha256="d" * 64,
    )
    runtime_sections = h8_gate._source_only_sections(
        result=source_only.result,
        refs=refs,
        dependency_closure_sha256=source_only.dependency_closure_sha256,
        preregistration_sha256=source_only.preregistration_sha256,
        prerequisites_current_and_pass=False,
    )

    evaluation = h8_gate.assemble_h8_source_only_evaluation(
        config_sha256="b" * 64,
        current_refs=refs,
        correctness=(),
        production_runs=(),
        profiler_runs=(),
        controls=(),
        dependency_closure_sha256="c" * 64,
        preregistration_sha256="d" * 64,
        runtime_sections=runtime_sections,
    )

    assert evaluation.runtime_sections is not None
    assert h8_gate.h8_validation_payload(evaluation)["protocol"] == (
        h8_gate.h8_validation_payload(source_only)["protocol"]
    )


def test_h8_artifact_environment_consumes_frozen_config_torch_version() -> None:
    import verify_vfe4
    from vfe4.artifacts.provenance import build_h8_environment
    from vfe4.config import bind_h8_current_refs

    refs = _current_refs()
    scientific = copy.deepcopy(
        verify_vfe4.CONFIG["operations"]["h8"]["config"]  # type: ignore[index]
    )
    config = bind_h8_current_refs(scientific, refs)
    source_only = h8_gate.assemble_h8_source_only_evaluation(
        config_sha256=config.config_sha256,
        current_refs=refs,
        correctness=(),
        production_runs=(),
        profiler_runs=(),
        controls=(),
        dependency_closure_sha256="c" * 64,
        preregistration_sha256="d" * 64,
    )
    runtime_sections = h8_gate._source_only_sections(
        result=source_only.result,
        refs=refs,
        dependency_closure_sha256=source_only.dependency_closure_sha256,
        preregistration_sha256=source_only.preregistration_sha256,
        prerequisites_current_and_pass=False,
    )

    retained = build_h8_environment(
        config=config,
        validation_environment=runtime_sections["environment"],
    )

    assert config.h8 is not None
    assert retained["pytorch_version"] == config.h8.torch_version
    assert retained["pytorch_version"] == "2.10.0.dev20251210+cu128"


def test_h8_direct_pass_remains_unavailable_without_parent_orchestration() -> None:
    with pytest.raises(ValueError, match="PASS H8 remains unavailable"):
        H8GateResult(
            gate="H8",
            status=GateStatus.PASS,
            config_sha256="a" * 64,
            candidate_junit_sha256="b" * 64,
            current_refs_registry_sha256="c" * 64,
            h7_manifest_sha256="d" * 64,
            h6_prediction_manifest_sha256="e" * 64,
            correctness=(),
            child_attempts=(),
            production_runs=(),
            profiler_runs=(),
            controls=(),
            obligations=(),
        )


def test_h8_prerequisite_reopen_is_fail_closed_without_reconstructing_refs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refs = _current_refs()
    prefix_reference = refs.h6_prefix
    assert type(prefix_reference) is H8H6PrefixReference
    original_certificates = tuple(
        SimpleNamespace(
            semantic_family_sha256=row.semantic_family_sha256,
            validation_payload_sha256=row.validation_payload_sha256,
            certificate_sha256=row.certificate_sha256,
        )
        for row in prefix_reference.semantic_families
    )

    def bounded_set(
        **changes: object,
    ) -> BoundedPrefixCertificateSet:
        values = {
            "schema_version": prefix_reference.certificate_set_schema,
            "config_sha256": prefix_reference.config_sha256,
            "workload_plan_sha256": (
                prefix_reference.workload_plan_sha256
            ),
            "validation_payload_sha256": (
                prefix_reference.validation_payload_sha256
            ),
            "prefix_certificate_set_sha256": (
                prefix_reference.prefix_certificate_set_sha256
            ),
            "certificates": original_certificates,
            **changes,
        }
        value = object.__new__(BoundedPrefixCertificateSet)
        for name, item in values.items():
            object.__setattr__(value, name, item)
        return value

    reopened = bounded_set()
    direct = object.__new__(A0DirectExactPrefixCertificateV1)
    for name, value in {
        "certificate_sha256": (
            prefix_reference.a0_direct_exact_prefix_certificate_sha256
        ),
        "status": EvidenceStatus.PASS,
        "obligations": (),
        "checks": {
            name: True
            for name in H6_A0_DIRECT_EXACT_PREFIX_REQUIRED_CHECKS
        },
        "bounded_a0_certificate_sha256": (
            original_certificates[0].certificate_sha256
        ),
    }.items():
        object.__setattr__(direct, name, value)
    reopen_calls: list[dict[str, object]] = []

    def reopen_prefix(**kwargs: object) -> tuple[object, object]:
        reopen_calls.append(kwargs)
        return reopened, direct

    monkeypatch.setattr(
        h8_gate,
        "reopen_h6_prefix_authorities",
        reopen_prefix,
    )
    prefix_payloads = {
        name: b"{}"
        for name in (
            "certificates/a0_direct_exact.json",
            "certificates/prefix_set.json",
            "config.json",
            "environment.json",
            "provenance.json",
            "validation/h6_prefix.json",
        )
    }
    h8_gate._revalidate_h6_prefix_certificates(
        prefix_reference,
        prefix_payloads,
    )
    assert reopen_calls == [
        {
            "root": Path(prefix_reference.artifact_path),
            "expected_manifest_sha256": (
                prefix_reference.manifest_sha256
            ),
            "expected_git_head": prefix_reference.producer_head,
            "expected_dirty_digest": (
                prefix_reference.producer_dirty_digest
            ),
            "expected_junit_sha256": (
                prefix_reference.candidate_junit_sha256
            ),
        }
    ]

    legacy = H8LegacyH6PrefixV4Reference(
        kind="h6_prefix",
        config_schema=prefix_reference.config_schema,
        validation_schema=prefix_reference.validation_schema,
        certificate_set_schema=prefix_reference.certificate_set_schema,
        config_sha256=prefix_reference.config_sha256,
        workload_plan_sha256=prefix_reference.workload_plan_sha256,
        validation_payload_sha256=prefix_reference.validation_payload_sha256,
        prefix_certificate_set_sha256=(
            prefix_reference.prefix_certificate_set_sha256
        ),
        semantic_families=prefix_reference.semantic_families,
        artifact_path=prefix_reference.artifact_path,
        manifest_sha256=prefix_reference.manifest_sha256,
        result_path=prefix_reference.result_path,
        result_sha256=prefix_reference.result_sha256,
        content_hashes=prefix_reference.content_hashes,
        payload_hashes=prefix_reference.payload_hashes,
        ledger_path=prefix_reference.ledger_path,
        ledger_sha256=prefix_reference.ledger_sha256,
        producer_head=prefix_reference.producer_head,
        producer_dirty_digest=prefix_reference.producer_dirty_digest,
        candidate_junit_sha256=(
            prefix_reference.candidate_junit_sha256
        ),
        status="pass",
    )
    legacy_refs = replace(refs, h6_prefix=legacy)
    assert (
        legacy_refs.registry_schema_version
        == "h8-current-candidate-refs-v4"
    )
    assert legacy_refs.prerequisite_obligations == (
        "h8_prerequisite_registry_v1_v2_v3_v4_requires_direct_a0_prefix",
    )
    import verify_vfe4
    from vfe4.config import bind_h8_current_refs

    with pytest.raises(ValueError, match="only the exact H8 v5 registry"):
        bind_h8_current_refs(
            copy.deepcopy(
                verify_vfe4.CONFIG["operations"]["h8"]["config"]  # type: ignore[index]
            ),
            legacy_refs,
        )
    with pytest.raises(ValueError, match="bounded H6-Prefix reference"):
        h8_gate._revalidate_h6_prefix_certificates(
            legacy,
            prefix_payloads,
        )

    mutations = (
        {"certificates": tuple(reversed(original_certificates))},
        {"certificates": original_certificates[:-1]},
        {
            "certificates": (
                original_certificates[0],
                original_certificates[0],
            )
        },
        {
            "certificates": (
                replace(
                    prefix_reference.semantic_families[0],
                    validation_payload_sha256="7" * 64,
                ),
                original_certificates[1],
            )
        },
        {
            "certificates": (
                replace(
                    prefix_reference.semantic_families[0],
                    certificate_sha256="8" * 64,
                ),
                original_certificates[1],
            )
        },
        {"config_sha256": "9" * 64},
        {"workload_plan_sha256": "0" * 64},
        {"validation_payload_sha256": "5" * 64},
        {"prefix_certificate_set_sha256": "6" * 64},
    )
    for mutation in mutations:
        monkeypatch.setattr(
            h8_gate,
            "reopen_h6_prefix_authorities",
            lambda **_kwargs: (bounded_set(**mutation), direct),
        )
        with pytest.raises(ValueError, match="bounded H6-Prefix"):
            h8_gate._revalidate_h6_prefix_certificates(
                prefix_reference,
                prefix_payloads,
            )

    stale_direct = object.__new__(A0DirectExactPrefixCertificateV1)
    for name in (
        "certificate_sha256",
        "status",
        "obligations",
        "checks",
        "bounded_a0_certificate_sha256",
    ):
        object.__setattr__(
            stale_direct,
            name,
            (
                "7" * 64
                if name == "certificate_sha256"
                else getattr(direct, name)
            ),
        )
    monkeypatch.setattr(
        h8_gate,
        "reopen_h6_prefix_authorities",
        lambda **_kwargs: (reopened, stale_direct),
    )
    with pytest.raises(ValueError, match="bounded H6-Prefix"):
        h8_gate._revalidate_h6_prefix_certificates(
            prefix_reference,
            prefix_payloads,
        )

    for field_name, legacy_schema in (
        ("config_schema", "h6-prefix-config-v2"),
        ("validation_schema", "h6-prefix-validation-set-v1"),
        ("certificate_set_schema", "h6-prefix-certificate-set-v1"),
    ):
        with pytest.raises(ValueError, match="schema must be bounded"):
            replace(prefix_reference, **{field_name: legacy_schema})

    validation = h8_gate.validate_h8_prerequisite_artifacts(refs)
    obligations = validation.obligations

    assert obligations == (
        "h8_prerequisite_h7_transitive_junit_artifact_revalidation_required",
        "h8_prerequisite_h1_h5_immutable_artifact_revalidation_required",
        "h8_prerequisite_h1_prefix_prior_immutable_artifact_revalidation_required",
        "h8_prerequisite_h6_prefix_immutable_artifact_revalidation_required",
        "h8_prerequisite_h7_immutable_artifact_revalidation_required",
        "h8_prerequisite_h6_prediction_v3_artifact_revalidation_required",
    )
    evaluation = h8_gate.assemble_h8_source_only_evaluation(
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


def _synthetic_h7_revalidation_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    H8H7Reference,
    dict[str, bytes],
    dict[str, Path],
    dict[str, str],
]:
    from verification.mp_oracles import h7_covariance
    from verification.mp_oracles.h7_covariance import H7MPOracleResult
    from vfe4.types.h7 import h7_owned_sha256
    from vfe4.types.results import H7GateResult
    import vfe4.validation

    repository_root = tmp_path / "repo"
    fixture_root = repository_root / "vfe4" / "validation" / "fixtures"
    fixture_root.mkdir(parents=True)
    (repository_root / "verification").mkdir()
    monkeypatch.setattr(
        h8_gate,
        "__file__",
        str(repository_root / "verification" / "h8_gate.py"),
    )

    semantic_hashes = {
        "density_probe_set_sha256": "1" * 64,
        "scalar_probe_set_sha256": "2" * 64,
        "precision_operand_set_sha256": "3" * 64,
        "oracle_inventory_sha256": "4" * 64,
    }
    fixture_payloads = {
        "h1_fixture_raw_sha256": b'{"fixture":"h1"}\n',
        "h7_fixture_raw_sha256": b'{"fixture":"h7"}\n',
        "density_probe_table_raw_sha256": h8_gate.canonical_h8_json_bytes(
            {
                "probe_set_sha256": semantic_hashes[
                    "density_probe_set_sha256"
                ]
            }
        )
        + b"\n",
        "scalar_probe_table_raw_sha256": h8_gate.canonical_h8_json_bytes(
            {
                "probe_set_sha256": semantic_hashes[
                    "scalar_probe_set_sha256"
                ]
            }
        )
        + b"\n",
        "precision_operand_table_raw_sha256": h8_gate.canonical_h8_json_bytes(
            {
                "precision_set_sha256": semantic_hashes[
                    "precision_operand_set_sha256"
                ]
            }
        )
        + b"\n",
    }
    fixture_names = {
        "h1_fixture_raw_sha256": "h1_v1.json",
        "h7_fixture_raw_sha256": "h7_v1.json",
        "density_probe_table_raw_sha256": "h7_density_probes_v1.json",
        "scalar_probe_table_raw_sha256": (
            "h7_scalar_density_probes_v1.json"
        ),
        "precision_operand_table_raw_sha256": (
            "h7_precision_operands_v2.json"
        ),
    }
    fixture_paths = {
        key: fixture_root / fixture_names[key] for key in fixture_payloads
    }
    for key, path in fixture_paths.items():
        path.write_bytes(fixture_payloads[key])

    monkeypatch.setattr(
        vfe4.validation,
        "parse_h7_fixture_bytes",
        lambda _payload: SimpleNamespace(
            density_probe_set_sha256=semantic_hashes[
                "density_probe_set_sha256"
            ]
        ),
    )

    def independently_validated(
        h1_fixture_bytes: bytes,
        h7_fixture_bytes: bytes,
        h7_density_probe_bytes: bytes,
        h1_scalar_probe_bytes: bytes | None = None,
        precision_operand_bytes: bytes | None = None,
    ) -> H7MPOracleResult:
        assert h1_scalar_probe_bytes is not None
        assert precision_operand_bytes is not None
        return H7MPOracleResult(
            status="EVIDENCE_VERIFIED",
            open_obligations=(),
            decimal_precision=100,
            gauss_hermite_orders=(41, 51),
            raw_fixture_sha256=tuple(
                hashlib.sha256(payload).hexdigest()
                for payload in (
                    h1_fixture_bytes,
                    h7_fixture_bytes,
                    h7_density_probe_bytes,
                    h1_scalar_probe_bytes,
                    precision_operand_bytes,
                )
            ),
            h1_source_paths=(),
            h7_source_path=SimpleNamespace(),
            trials=(),
            inventory_sha256=semantic_hashes[
                "oracle_inventory_sha256"
            ],
        )

    monkeypatch.setattr(
        h7_covariance,
        "evaluate_h7_from_raw_bytes",
        independently_validated,
    )

    fixture_hashes: dict[str, str] = {
        "h1_fixture_raw_sha256": hashlib.sha256(
            fixture_payloads["h1_fixture_raw_sha256"]
        ).hexdigest(),
        "h7_fixture_raw_sha256": hashlib.sha256(
            fixture_payloads["h7_fixture_raw_sha256"]
        ).hexdigest(),
        "density_probe_table_raw_sha256": hashlib.sha256(
            fixture_payloads["density_probe_table_raw_sha256"]
        ).hexdigest(),
        "density_probe_set_sha256": semantic_hashes[
            "density_probe_set_sha256"
        ],
        "scalar_probe_table_raw_sha256": hashlib.sha256(
            fixture_payloads["scalar_probe_table_raw_sha256"]
        ).hexdigest(),
        "scalar_probe_set_sha256": semantic_hashes[
            "scalar_probe_set_sha256"
        ],
        "precision_operand_table_raw_sha256": hashlib.sha256(
            fixture_payloads["precision_operand_table_raw_sha256"]
        ).hexdigest(),
        "precision_operand_set_sha256": semantic_hashes[
            "precision_operand_set_sha256"
        ],
        "oracle_inventory_sha256": semantic_hashes[
            "oracle_inventory_sha256"
        ],
    }
    assert tuple(fixture_hashes) == H7GateResult.fixture_hash_keys
    fixture_set_sha256 = h7_owned_sha256(
        "vfe4.h7.fixture-set.v1",
        fixture_hashes,
    )
    validation = {
        "schema": "h7-frame-covariance-validation-v1",
        "fixture_set_sha256": fixture_set_sha256,
        "result": {
            "status": "pass",
            "obligations": [],
            "fixture_hashes": fixture_hashes,
        },
    }
    payloads = {
        "validation/h7.json": h8_gate.canonical_h8_json_bytes(validation)
    }
    reference = replace(
        _current_refs().h7,
        fixture_set_sha256=fixture_set_sha256,
    )
    return reference, payloads, fixture_paths, semantic_hashes


def test_h8_h7_revalidation_reopens_all_nine_fixture_identities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference, payloads, _fixture_paths, _semantic_hashes = (
        _synthetic_h7_revalidation_case(tmp_path, monkeypatch)
    )

    h8_gate._revalidate_h7_fixture_set(reference, payloads)


@pytest.mark.parametrize(
    ("fixture_key", "semantic_key"),
    (
        ("scalar_probe_table_raw_sha256", None),
        ("precision_operand_table_raw_sha256", None),
        ("scalar_probe_table_raw_sha256", "probe_set_sha256"),
        (
            "precision_operand_table_raw_sha256",
            "precision_set_sha256",
        ),
    ),
)
def test_h8_h7_revalidation_rejects_scalar_and_precision_mutations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_key: str,
    semantic_key: str | None,
) -> None:
    from vfe4.types.h7 import h7_owned_sha256

    reference, payloads, fixture_paths, _semantic_hashes = (
        _synthetic_h7_revalidation_case(tmp_path, monkeypatch)
    )
    path = fixture_paths[fixture_key]
    if semantic_key is None:
        value = json.loads(path.read_bytes())
        value["mutated"] = True
        path.write_bytes(h8_gate.canonical_h8_json_bytes(value) + b"\n")
    else:
        value = json.loads(path.read_bytes())
        value[semantic_key] = "9" * 64
        path.write_bytes(h8_gate.canonical_h8_json_bytes(value) + b"\n")
        validation = json.loads(payloads["validation/h7.json"])
        fixture_hashes = validation["result"]["fixture_hashes"]
        fixture_hashes[fixture_key] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        fixture_set_sha256 = h7_owned_sha256(
            "vfe4.h7.fixture-set.v1",
            fixture_hashes,
        )
        validation["fixture_set_sha256"] = fixture_set_sha256
        payloads = {
            "validation/h7.json": h8_gate.canonical_h8_json_bytes(
                validation
            )
        }
        reference = replace(
            reference,
            fixture_set_sha256=fixture_set_sha256,
        )
        semantic_fixture_key = (
            "scalar_probe_set_sha256"
            if semantic_key == "probe_set_sha256"
            else "precision_operand_set_sha256"
        )
        assert fixture_hashes[fixture_key] == hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        assert fixture_hashes[semantic_fixture_key] != value[semantic_key]

    with pytest.raises(ValueError, match="fixture hashes"):
        h8_gate._revalidate_h7_fixture_set(reference, payloads)


@pytest.mark.parametrize(
    ("status", "obligations", "inventory_sha256"),
    (
        ("INCONCLUSIVE", (), "4" * 64),
        ("EVIDENCE_VERIFIED", ("unexpected obligation",), "4" * 64),
        ("EVIDENCE_VERIFIED", (), None),
    ),
)
def test_h8_h7_revalidation_rejects_nonclosing_oracle_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    obligations: tuple[str, ...],
    inventory_sha256: str | None,
) -> None:
    from verification.mp_oracles import h7_covariance
    from verification.mp_oracles.h7_covariance import H7MPOracleResult

    reference, payloads, _fixture_paths, _semantic_hashes = (
        _synthetic_h7_revalidation_case(tmp_path, monkeypatch)
    )

    def nonclosing_oracle(
        h1_fixture_bytes: bytes,
        h7_fixture_bytes: bytes,
        h7_density_probe_bytes: bytes,
        h1_scalar_probe_bytes: bytes | None = None,
        precision_operand_bytes: bytes | None = None,
    ) -> H7MPOracleResult:
        assert h1_scalar_probe_bytes is not None
        assert precision_operand_bytes is not None
        return H7MPOracleResult(
            status=status,  # type: ignore[arg-type]
            open_obligations=obligations,
            decimal_precision=100,
            gauss_hermite_orders=(41, 51),
            raw_fixture_sha256=tuple(
                hashlib.sha256(payload).hexdigest()
                for payload in (
                    h1_fixture_bytes,
                    h7_fixture_bytes,
                    h7_density_probe_bytes,
                    h1_scalar_probe_bytes,
                    precision_operand_bytes,
                )
            ),
            h1_source_paths=(),
            h7_source_path=SimpleNamespace(),
            trials=(),
            inventory_sha256=inventory_sha256,
        )

    monkeypatch.setattr(
        h7_covariance,
        "evaluate_h7_from_raw_bytes",
        nonclosing_oracle,
    )

    with pytest.raises(ValueError, match="oracle"):
        h8_gate._revalidate_h7_fixture_set(reference, payloads)


def test_h8_h7_revalidation_rejects_oracle_inventory_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from verification.mp_oracles import h7_covariance

    reference, payloads, _fixture_paths, _semantic_hashes = (
        _synthetic_h7_revalidation_case(tmp_path, monkeypatch)
    )
    original = h7_covariance.evaluate_h7_from_raw_bytes

    def changed_inventory(*args: object, **kwargs: object):
        result = original(*args, **kwargs)
        return replace(result, inventory_sha256="8" * 64)

    monkeypatch.setattr(
        h7_covariance,
        "evaluate_h7_from_raw_bytes",
        changed_inventory,
    )

    with pytest.raises(ValueError, match="fixture hashes"):
        h8_gate._revalidate_h7_fixture_set(reference, payloads)


def test_h8_h7_revalidation_rejects_second_read_source_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference, payloads, fixture_paths, _semantic_hashes = (
        _synthetic_h7_revalidation_case(tmp_path, monkeypatch)
    )
    target = fixture_paths["scalar_probe_table_raw_sha256"]
    original_read_bytes = Path.read_bytes
    reads = 0

    def swapped_read_bytes(path: Path) -> bytes:
        nonlocal reads
        payload = original_read_bytes(path)
        if path == target:
            reads += 1
            if reads == 2:
                return payload + b" "
        return payload

    monkeypatch.setattr(Path, "read_bytes", swapped_read_bytes)

    with pytest.raises(ValueError, match="changed"):
        h8_gate._revalidate_h7_fixture_set(reference, payloads)


@pytest.mark.parametrize("mutation", ("missing", "reordered"))
def test_h8_h7_revalidation_rejects_missing_or_reordered_hash_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    reference, payloads, _fixture_paths, _semantic_hashes = (
        _synthetic_h7_revalidation_case(tmp_path, monkeypatch)
    )
    validation = json.loads(payloads["validation/h7.json"])
    hashes = validation["result"]["fixture_hashes"]
    if mutation == "missing":
        hashes.pop("scalar_probe_set_sha256")
    else:
        validation["result"]["fixture_hashes"] = dict(
            reversed(tuple(hashes.items()))
        )
    monkeypatch.setattr(
        h8_gate,
        "_canonical_mapping",
        lambda _payload, _name: validation,
    )

    with pytest.raises(ValueError, match="fixture hashes"):
        h8_gate._revalidate_h7_fixture_set(reference, payloads)


def test_h8_payload_inventories_are_exact_and_private(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import verify_vfe4
    from vfe4.artifacts.provenance import build_h8_environment, build_h8_provenance
    from vfe4.config import bind_h8_current_refs

    refs = _current_refs()
    scientific = copy.deepcopy(
        verify_vfe4.CONFIG["operations"]["h8"]["config"]  # type: ignore[index]
    )
    config = bind_h8_current_refs(scientific, refs)
    assert config.h8 is not None
    expected_profiler_source_hashes = {
        "memory_profile": config.h8.profiler_memory_source_sha256,
        "profiler": config.h8.profiler_source_sha256,
    }
    monkeypatch.setattr(
        h8_gate,
        "H8_PROFILER_MEMORY_SOURCE_SHA256",
        "0" * 64,
        raising=False,
    )
    monkeypatch.setattr(
        h8_gate,
        "H8_PROFILER_SOURCE_SHA256",
        "1" * 64,
        raising=False,
    )
    monkeypatch.setattr(
        h8_gate,
        "H8_PROFILER_API_CONTRACT_SHA256",
        "2" * 64,
        raising=False,
    )
    evaluation = h8_gate.assemble_h8_source_only_evaluation(
        config_sha256=config.config_sha256,
        current_refs=refs,
        correctness=(),
        production_runs=(),
        profiler_runs=(),
        controls=(),
        dependency_closure_sha256="c" * 64,
        preregistration_sha256="d" * 64,
    )
    validation = h8_gate.h8_validation_payload(evaluation)
    environment = build_h8_environment(
        config=config,
        validation_environment=validation["environment"],  # type: ignore[arg-type]
    )
    configured_operation = "H8-config-owned-operation"
    object.__setattr__(config.h8, "operation", configured_operation)
    provenance = build_h8_provenance(
        config=config,
        evaluation=evaluation,
        git_head_value=refs.candidate_head,
        dirty_digest_value=refs.candidate_dirty_digest,
        source_sha256_value=source_candidate_sha256(
            git_head_value=refs.candidate_head,
            dirty_digest_value=refs.candidate_dirty_digest,
        ),
        reference_registry_path=Path("h8-current-refs.json"),
        reference_registry_sha256=refs.registry_sha256,
        started_utc="2026-07-27T00:00:00.000000Z",
        ended_utc="2026-07-27T00:00:00.000001Z",
    )
    payloads = h8_gate.build_h8_publication_payloads(
        config,
        evaluation,
        h7_reference=refs.h7,
        h6_prediction_reference=refs.h6_prediction,
        provenance=provenance,
        environment=environment,
    )

    assert tuple(validation) == h8_gate.H8_VALIDATION_TOP_LEVEL_KEYS
    assert tuple(payloads) == h8_gate.H8_PUBLICATION_PAYLOAD_KEYS
    assert validation["environment"]["pytorch_version"] == config.h8.torch_version
    assert validation["protocol"]["profiler_source_hashes"] == (
        expected_profiler_source_hashes
    )
    assert validation["protocol"]["profiler_api_contract_sha256"] == (
        config.h8.profiler_api_contract_sha256
    )
    assert payloads["environment.json"]["pytorch_version"] == config.h8.torch_version
    assert payloads["provenance.json"]["selected_operation"] == configured_operation
    assert config.h8.torch_version == "2.10.0.dev20251210+cu128"
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
    evaluation = h8_gate.assemble_h8_source_only_evaluation(
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
