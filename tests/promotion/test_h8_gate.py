from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import verification.h8_gate as h8_gate
from vfe4.artifacts import source_candidate_sha256
from vfe4.artifacts.h6 import CandidateArtifactReference
from vfe4.types.h6 import BoundedPrefixCertificateSet
from vfe4.types.h7 import H7PredecessorReference
from vfe4.types.h8 import (
    CurrentH8PrerequisiteRefs,
    H8ChildRequest,
    H8GateEvaluation,
    H8H1H5Reference,
    H8H1PrefixPriorReference,
    H8H6PredictionReference,
    H8H6PrefixReference,
    H8H6PrefixSemanticFamilyReference,
    H8H7Reference,
    H8LegacyH6PrefixReference,
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
            config_schema="h6-prefix-config-v3",
            validation_schema="h6-prefix-validation-set-v2",
            certificate_set_schema="h6-prefix-certificate-set-v2",
            config_sha256=digest,
            workload_plan_sha256="b" * 64,
            validation_payload_sha256="c" * 64,
            prefix_certificate_set_sha256="d" * 64,
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


def test_h8_gate_retains_parent_attempt_failure_without_a_child_result(
    tmp_path: Path,
) -> None:
    from verification.h8_budget import (
        H8ChildProcessRecord,
        build_h8_child_invocation,
        classify_h8_child_outcome,
        make_h8_child_attempt_record,
        make_h8_identity_record,
    )

    refs = _current_refs()
    request = H8ChildRequest(
        mode="production",
        seed=20260721,
        repetition=0,
        config_sha256="b" * 64,
        protocol_sha256="c" * 64,
        control_id=None,
    )
    identities = {
        name: make_h8_identity_record(name, payload)
        for name, payload in (
            (
                "hardware",
                {
                    "platform": "test",
                    "release": "test-release",
                    "system": "test",
                    "machine": "test",
                    "processor": "test",
                    "cpu_count": 1,
                    "python": "test",
                    "implementation": "test",
                },
            ),
            ("affinity", {"adapter": "test", "cpus": [0]}),
            (
                "thread",
                {
                    "environment": {
                        "OMP_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1",
                        "NUMEXPR_NUM_THREADS": "1",
                        "VECLIB_MAXIMUM_THREADS": "1",
                    },
                    "torch_num_threads": 1,
                    "torch_num_interop_threads": 1,
                },
            ),
            (
                "blas",
                {
                    "torch_version": "2.9.1",
                    "numpy_version": "test",
                    "torch_config": "test",
                    "numpy_config": "test",
                },
            ),
        )
    }
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

    evaluation = h8_gate.assemble_h8_gate_evaluation(
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
                "config_sha256": "b" * 64,
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
    evaluation = h8_gate.assemble_h8_gate_evaluation(
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
    evaluation = h8_gate.assemble_h8_gate_evaluation(
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
    evaluation = h8_gate.assemble_h8_gate_evaluation(
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
    source_only = h8_gate.assemble_h8_gate_evaluation(
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

    evaluation = h8_gate.assemble_h8_gate_evaluation(
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
    reopen_calls: list[dict[str, object]] = []

    def reopen_bounded(**kwargs: object) -> object:
        reopen_calls.append(kwargs)
        return reopened

    monkeypatch.setattr(
        h8_gate,
        "reopen_bounded_prefix_certificate_set",
        reopen_bounded,
    )
    prefix_payloads = {
        name: b"{}"
        for name in (
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

    legacy = H8LegacyH6PrefixReference(
        kind="h6_prefix",
        certificate_set_sha256="5" * 64,
        certificate_hashes={"legacy-case-key": "6" * 64},
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
        == "h8-current-candidate-refs-v2"
    )
    assert legacy_refs.prerequisite_obligations == (
        "h8_prerequisite_legacy_registry_requires_bounded_h6_prefix_v3",
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
            "reopen_bounded_prefix_certificate_set",
            lambda **_kwargs: bounded_set(**mutation),
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
