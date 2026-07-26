from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest

import verification.h6_prefix_gate as h6_prefix_gate
from verification.h6_prefix_gate import (
    H6PrefixReportBundle,
    compose_prefix_certificate,
    h6_prefix_artifact_payloads,
    publish_h6_prefix_artifact,
)
from vfe4.artifacts.atomic import ArtifactPublicationError
from vfe4.config import (
    H6PrefixResolvedConfig,
    H6SourceIdentity,
    resolve_h6_prefix_config,
)
from vfe4.config.schema import H6_PREFIX_V2_AUTHORIZATION_SHA256
from vfe4.predictive import EstimatorIdentity
from vfe4.types import (
    ArmConfig,
    ArmId,
    CapacityAllocation,
    CausalDag,
    CausalDagRow,
    EstimatorSpec,
    EvidenceStatus,
    H6LanguageStructure,
    H6PrefixProfilePair,
    H6PrefixWorkloadPlan,
    PrefixCaseKey,
    PrefixReportBinding,
    VocabularyIdentity,
    ZeroDimensionalBase,
    arm_model_family_sha256,
)
from vfe4.types.h6 import canonical_json_bytes
from vfe4.validation.h6_prefix import (
    DynamicCheckResult,
    DynamicExecutionPlan,
    DynamicPrefixCase,
    DynamicPrefixReport,
    PairSideHarness,
    run_dynamic_prefix_checks,
)
from vfe4.validation.h6_static_audit import (
    StaticAuditCheck,
    StaticAuditReport,
)


_DYNAMIC_CHECK_NAMES = (
    "signature_and_identity",
    "dynamic_target_suffix_leakage",
    "cache_identity",
    "source_mask",
    "case_inventory",
    "validation_data_safety",
)
_STATIC_CHECK_NAMES = (
    "import_signature_access",
    "taint_cache_capability",
    "mask_normalization_support",
    "inventory_identity",
)


def _owned_hash(domain: str, payload: object) -> str:
    return hashlib.sha256(
        domain.encode("ascii") + b"\x00" + canonical_json_bytes(payload)
    ).hexdigest()


def _structure(horizon: int) -> H6LanguageStructure:
    base = ZeroDimensionalBase.create()
    dag = CausalDag.create(
        node_labels=tuple(range(horizon + 1)),
        rows=tuple(
            CausalDagRow(receiver, tuple(range(receiver)))
            for receiver in range(1, horizon + 1)
        ),
    )
    return H6LanguageStructure.create(
        base=base,
        dag=dag,
        receiver_labels=tuple(range(1, horizon + 1)),
    )


def _arm_config(*, vocabulary: VocabularyIdentity, horizon: int, width: int) -> ArmConfig:
    return ArmConfig.create(
        arm=ArmId.A0,
        config_id="h6-a0-transformer-v2",
        vocabulary=vocabulary,
        horizon=horizon,
        latent_enabled=False,
        state_channel_enabled=False,
        model_channel_enabled=False,
        source_mode="absent",
        map_mode="absent",
        recognition_family="absent",
        recognition_conditioning="absent",
        prior_variant="absent",
        mixture_mode="absent",
        objective_kind="cross_entropy",
        capacity_allocation=CapacityAllocation.create(
            emission_width=width,
            latent_width=None,
            recognition_width=None,
        ),
    )


def _categorical_arm_config(
    *,
    arm: ArmId,
    vocabulary: VocabularyIdentity,
    horizon: int,
) -> ArmConfig:
    if arm is ArmId.A2:
        return ArmConfig.create(
            arm=arm,
            config_id="h6-a2-generic-map-v1",
            vocabulary=vocabulary,
            horizon=horizon,
            latent_enabled=True,
            state_channel_enabled=True,
            model_channel_enabled=True,
            source_mode="categorical",
            map_mode="generic_fixed_frame_non_coboundary",
            recognition_family="structured",
            recognition_conditioning="smoothing",
            prior_variant="fixed",
            mixture_mode="exact",
            objective_kind="complete_elbo",
            capacity_allocation=CapacityAllocation.create(
                emission_width=48,
                latent_width=8,
                recognition_width=32,
            ),
        )
    if arm is ArmId.A5:
        return ArmConfig.create(
            arm=arm,
            config_id=(
                "h6-a5-structured-parent-specific-prefix-exact-complete-"
                "latent-smoothing-v2"
            ),
            vocabulary=vocabulary,
            horizon=horizon,
            latent_enabled=True,
            state_channel_enabled=True,
            model_channel_enabled=True,
            source_mode="categorical",
            map_mode="shared_vertex_coboundary",
            recognition_family="structured",
            recognition_conditioning="smoothing",
            prior_variant="parent_specific_pooled_prefix",
            mixture_mode="exact",
            objective_kind="complete_elbo",
            capacity_allocation=CapacityAllocation.create(
                emission_width=48,
                latent_width=8,
                recognition_width=32,
                prior_context_width=2,
            ),
        )
    raise ValueError("test helper supports the two categorical profiles")


def _categorical_profile(arm: ArmId) -> H6PrefixProfilePair:
    small = _categorical_arm_config(
        arm=arm,
        vocabulary=VocabularyIdentity("h6-prefix-small-v1", 3, "a" * 64),
        horizon=4,
    )
    production = _categorical_arm_config(
        arm=arm,
        vocabulary=VocabularyIdentity("wikitext-2-byte-v1", 258, "b" * 64),
        horizon=32,
    )
    estimator = EstimatorSpec.create(
        kind="weighted_smc",
        particle_count=4,
        resampling="systematic_ess_half",
    )
    return H6PrefixProfilePair.create(
        profile_id=f"h6-{arm.value.lower()}-categorical-smc-4",
        small_arm_config=small,
        production_arm_config=production,
        estimator=estimator,
        small_structure=_structure(4),
        production_structure=_structure(32),
        data_safety_sha256=hashlib.sha256(
            b"VFE4-H6-TARGET-FREE-PREDICTIVE-BOUNDARY-V1"
        ).hexdigest(),
        small_model_family_sha256=arm_model_family_sha256(small),
        production_model_family_sha256=arm_model_family_sha256(production),
    )


def test_runner_binds_source_mask_observations_for_each_prefix_case_family(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Catch a runner that omits categorical source-mask evidence arguments."""

    case = DynamicPrefixCase.create(
        ordinal=0,
        receiver_t=2,
        shared_prefix=(1,),
        left_tail=(0,),
        right_tail=(2,),
    )
    captured: list[dict[str, object]] = []

    def capture_dynamic_checks(**kwargs: object) -> object:
        assert (
            "source_mask_observations" in kwargs
            and "all_invalid_observation" in kwargs
            and "source_mask_observer" in kwargs
        ), "run_h6_prefix() omits the source-mask observer/all-invalid evidence"
        captured.append(kwargs)
        return object()

    monkeypatch.setattr(h6_prefix_gate, "_validate_resolved_runner_config", lambda _: None)
    monkeypatch.setattr(h6_prefix_gate, "_small_cases", lambda _: (case,))
    monkeypatch.setattr(
        h6_prefix_gate,
        "load_frozen_validation_perturbations",
        lambda: SimpleNamespace(dynamic_cases=(case,)),
    )
    monkeypatch.setattr(
        h6_prefix_gate,
        "run_dynamic_prefix_checks",
        capture_dynamic_checks,
    )
    monkeypatch.setattr(h6_prefix_gate, "audit_h6_static_source", lambda *_: object())
    monkeypatch.setattr(h6_prefix_gate, "compose_prefix_certificate", lambda **_: object())
    monkeypatch.setattr(h6_prefix_gate, "H6PrefixReportBundle", lambda **_: object())
    monkeypatch.setattr(
        h6_prefix_gate,
        "publish_h6_prefix_artifact",
        lambda **_: (object(), tmp_path / "published"),
    )
    config = SimpleNamespace(
        source=SimpleNamespace(
            git_head="1" * 40,
            dirty_digest="2" * 64,
            source_sha256="3" * 64,
        ),
        execution_mode="focused_subset",
        profiles=(
            _categorical_profile(ArmId.A2),
            _categorical_profile(ArmId.A5),
        ),
        authorization_sha256=None,
        artifact_root=tmp_path,
        canonical_json="{}",
        config_sha256="4" * 64,
    )

    h6_prefix_gate.run_h6_prefix(config=config, junit_sha256=None)

    assert len(captured) == 4
    for values in captured:
        arm_config = values["arm_config"]
        observations = values["source_mask_observations"]
        observer = values["source_mask_observer"]
        all_invalid = values["all_invalid_observation"]
        assert observations is None
        assert callable(observer)
        assert getattr(all_invalid, "config_sha256") == arm_config.config_sha256
        assert getattr(all_invalid, "outcome") == "rejected"
        assert getattr(all_invalid, "observed_type") == "AllInvalidSourceRowError"


def _model_family_sha256(config: ArmConfig) -> str:
    factory = (
        "build_a0@h6-arm-v2"
        if config.arm is ArmId.A0
        else f"build_{config.arm.value.lower()}@h6-arm-v1"
    )
    return _owned_hash(
        "vfe4.h6.arm-model-family.v1",
        {
            "config_sha256": config.config_sha256,
            "factory": factory,
        },
    )


def _vocabulary_sha256(vocabulary: VocabularyIdentity) -> str:
    return _owned_hash(
        "vfe4.h6.vocabulary-identity.v1",
        {
            "vocabulary_id": vocabulary.vocabulary_id,
            "size": vocabulary.size,
            "tokenizer_spec_sha256": vocabulary.tokenizer_spec_sha256,
        },
    )


def _key(
    *,
    config: ArmConfig,
    estimator: EstimatorSpec,
    model_family_sha256: str,
    data_safety_sha256: str,
) -> PrefixCaseKey:
    return PrefixCaseKey(
        arm=config.arm,
        predictor_config_sha256=config.config_sha256,
        estimator_sha256=estimator.estimator_sha256,
        model_family_sha256=model_family_sha256,
        vocabulary_sha256=_vocabulary_sha256(config.vocabulary),
        data_safety_sha256=data_safety_sha256,
        git_head="1" * 40,
        dirty_digest="2" * 64,
    )


def _dynamic_report(
    *,
    key: PrefixCaseKey,
    expected_by_position: tuple[int, ...],
    expected_total: int,
    mode: str,
    salt: str,
) -> DynamicPrefixReport:
    checks = []
    for name in _DYNAMIC_CHECK_NAMES:
        status = EvidenceStatus.PASS
        completed = expected_total
        violations = 0
        first_counterexample = None
        obligations: tuple[str, ...] = ()
        if mode == "inconclusive" and name == "case_inventory":
            status = EvidenceStatus.INCONCLUSIVE
            completed = min(16, expected_total)
            obligations = ("focused synthetic inventory is incomplete",)
        elif mode == "fail" and name == "dynamic_target_suffix_leakage":
            status = EvidenceStatus.FAIL
            violations = 1
            first_counterexample = "synthetic target-suffix witness"
        checks.append(
            DynamicCheckResult.create(
                name=name,
                status=status,
                expected_count=expected_total,
                completed_count=completed,
                violation_count=violations,
                first_counterexample=first_counterexample,
                obligations=obligations,
            )
        )
    status = (
        EvidenceStatus.FAIL
        if mode == "fail"
        else EvidenceStatus.INCONCLUSIVE
        if mode == "inconclusive"
        else EvidenceStatus.PASS
    )
    obligations = (
        ("focused synthetic inventory is incomplete",)
        if status is EvidenceStatus.INCONCLUSIVE
        else ()
    )
    values = {
        "schema_version": "h6-dynamic-prefix-report-v1",
        "key": key,
        "execution_plan_sha256": _owned_hash("test.plan", salt),
        "model_state_sha256": _owned_hash("test.model", salt),
        "proposal_identity_sha256": _owned_hash("test.proposal", salt),
        "estimator_semantic_sha256": _owned_hash("test.estimator-semantic", salt),
        "estimator_artifact_bytes_sha256": _owned_hash(
            "test.estimator-artifact", salt
        ),
        "stream_seed": 2026072197,
        "completed_by_position": (
            expected_by_position
            if mode != "inconclusive"
            else tuple(min(value, 4) for value in expected_by_position)
        ),
        "checks": tuple(checks),
        "status": status,
        "obligations": obligations,
        "unresolved_diagnostics": (),
        "first_counterexample": (
            "synthetic target-suffix witness"
            if status is EvidenceStatus.FAIL
            else None
        ),
        "case_result_manifest_sha256": _owned_hash("test.cases", salt),
        "cache_manifest_sha256": _owned_hash("test.cache", salt),
        "pair_harness_manifest_sha256": _owned_hash("test.pairs", salt),
        "mask_manifest_sha256": _owned_hash("test.mask", salt),
        "complete_case_manifest_sha256": (
            None
            if status is EvidenceStatus.INCONCLUSIVE
            else _owned_hash("test.complete-cases", salt)
        ),
    }
    provisional = object.__new__(DynamicPrefixReport)
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    return DynamicPrefixReport(
        **values,
        report_sha256=_owned_hash(
            "vfe4.h6.dynamic-prefix-report.v1",
            provisional.canonical_payload(),
        ),
    )


def _case_key_manifest(keys: tuple[PrefixCaseKey, ...]) -> str:
    ordered = tuple(
        payload
        for _, payload in sorted(
            (
                (canonical_json_bytes(key.canonical_payload()), key.canonical_payload())
                for key in keys
            ),
            key=lambda item: item[0],
        )
    )
    return _owned_hash("vfe4.h6.static-audit-case-keys.v1", ordered)


def _static_report(keys: tuple[PrefixCaseKey, ...]) -> StaticAuditReport:
    checks = tuple(
        StaticAuditCheck(
            name=name,
            status=EvidenceStatus.PASS,
            finding_sha256s=(),
            obligations=(),
            check_sha256=_owned_hash(
                "vfe4.h6.static-audit-check.v1",
                {
                    "name": name,
                    "status": EvidenceStatus.PASS.value,
                    "finding_sha256s": (),
                    "obligations": (),
                },
            ),
        )
        for name in _STATIC_CHECK_NAMES
    )
    values = {
        "schema_version": "h6-static-audit-v1",
        "source_manifest_sha256": _owned_hash("test.static-sources", "sources"),
        "rules_sha256": _owned_hash("test.static-rules", "rules"),
        "case_key_manifest_sha256": _case_key_manifest(keys),
        "checks": checks,
        "findings": (),
        "status": EvidenceStatus.PASS,
        "obligations": (),
    }
    return StaticAuditReport(
        **values,
        report_sha256=_owned_hash(
            "vfe4.h6.static-audit-report.v1",
            {
                "schema_version": values["schema_version"],
                "source_manifest_sha256": values["source_manifest_sha256"],
                "rules_sha256": values["rules_sha256"],
                "case_key_manifest_sha256": values[
                    "case_key_manifest_sha256"
                ],
                "checks": tuple(check.check_sha256 for check in checks),
                "findings": (),
                "status": EvidenceStatus.PASS.value,
                "obligations": (),
            },
        ),
    )


def _unsafe_stale_report(report: DynamicPrefixReport) -> DynamicPrefixReport:
    stale = object.__new__(DynamicPrefixReport)
    for item in fields(DynamicPrefixReport):
        object.__setattr__(
            stale,
            item.name,
            "f" * 64
            if item.name == "report_sha256"
            else getattr(report, item.name),
        )
    return stale


def test_h6_prefix_reports_bind_status_and_publish_only_the_independent_artifact(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts"
    small_vocabulary = VocabularyIdentity("h6-prefix-small-v1", 3, "a" * 64)
    production_vocabulary = VocabularyIdentity(
        "wikitext-2-byte-v1", 258, "b" * 64
    )
    small_config = _arm_config(
        vocabulary=small_vocabulary, horizon=4, width=4
    )
    production_config = _arm_config(
        vocabulary=production_vocabulary, horizon=32, width=52
    )
    estimator = EstimatorSpec.create(
        kind="weighted_smc",
        particle_count=128,
        resampling="systematic_ess_half",
    )
    data_safety_sha256 = "c" * 64
    profile = H6PrefixProfilePair.create(
        profile_id="a0-weighted-smc-128",
        small_arm_config=small_config,
        production_arm_config=production_config,
        estimator=estimator,
        small_structure=_structure(4),
        production_structure=_structure(32),
        data_safety_sha256=data_safety_sha256,
        small_model_family_sha256=_model_family_sha256(small_config),
        production_model_family_sha256=_model_family_sha256(
            production_config
        ),
    )
    small_key = _key(
        config=small_config,
        estimator=estimator,
        model_family_sha256=profile.small_model_family_sha256,
        data_safety_sha256=data_safety_sha256,
    )
    validation_key = _key(
        config=production_config,
        estimator=estimator,
        model_family_sha256=profile.production_model_family_sha256,
        data_safety_sha256=data_safety_sha256,
    )
    static_report = _static_report((small_key, validation_key))

    def reports(mode: str) -> tuple[DynamicPrefixReport, DynamicPrefixReport]:
        return (
            _dynamic_report(
                key=small_key,
                expected_by_position=(6561, 2187, 729, 243),
                expected_total=9720,
                mode=mode,
                salt=f"small-{mode}",
            ),
            _dynamic_report(
                key=validation_key,
                expected_by_position=(4096,),
                expected_total=4096,
                mode=mode,
                salt=f"validation-{mode}",
            ),
        )

    small_pass, validation_pass = reports("pass")
    certificate = compose_prefix_certificate(
        profile=profile,
        small_report=small_pass,
        validation_report=validation_pass,
        static_report=static_report,
    )
    assert certificate.key == validation_key
    assert certificate.status is EvidenceStatus.PASS
    certificate_payload = json.loads(
        certificate.validation_payload_canonical_json
    )
    binding = PrefixReportBinding(**certificate_payload["report_binding"])
    assert binding.small_report_sha256 == small_pass.report_sha256
    assert (
        binding.small_case_manifest_sha256
        == small_pass.complete_case_manifest_sha256
    )
    assert binding.validation_report_sha256 == validation_pass.report_sha256
    assert (
        binding.validation_case_manifest_sha256
        == validation_pass.complete_case_manifest_sha256
    )
    assert certificate_payload["small_key"] == small_key.canonical_payload()
    assert certificate_payload["key"] == validation_key.canonical_payload()
    assert all(certificate_payload["checks"].values())

    small_inconclusive, validation_inconclusive = reports("inconclusive")
    inconclusive = compose_prefix_certificate(
        profile=profile,
        small_report=small_inconclusive,
        validation_report=validation_inconclusive,
        static_report=static_report,
    )
    assert inconclusive.status is EvidenceStatus.INCONCLUSIVE
    assert inconclusive.obligations

    small_fail, validation_fail = reports("fail")
    failed = compose_prefix_certificate(
        profile=profile,
        small_report=small_fail,
        validation_report=validation_fail,
        static_report=static_report,
    )
    assert failed.status is EvidenceStatus.FAIL
    assert failed.obligations == ()

    with pytest.raises(ValueError, match="identity is stale"):
        compose_prefix_certificate(
            profile=profile,
            small_report=_unsafe_stale_report(small_pass),
            validation_report=validation_pass,
            static_report=static_report,
        )

    focused_estimator = EstimatorSpec.create(
        kind="weighted_smc",
        particle_count=4,
        resampling="systematic_ess_half",
    )
    focused_profile = H6PrefixProfilePair.create(
        profile_id="a0-weighted-smc-4",
        small_arm_config=small_config,
        production_arm_config=production_config,
        estimator=focused_estimator,
        small_structure=_structure(4),
        production_structure=_structure(32),
        data_safety_sha256=hashlib.sha256(
            b"VFE4-H6-TARGET-FREE-PREDICTIVE-BOUNDARY-V1"
        ).hexdigest(),
        small_model_family_sha256=_model_family_sha256(small_config),
        production_model_family_sha256=_model_family_sha256(
            production_config
        ),
    )
    focused_small_key = _key(
        config=small_config,
        estimator=focused_estimator,
        model_family_sha256=focused_profile.small_model_family_sha256,
        data_safety_sha256=focused_profile.data_safety_sha256,
    )
    focused_validation_key = _key(
        config=production_config,
        estimator=focused_estimator,
        model_family_sha256=focused_profile.production_model_family_sha256,
        data_safety_sha256=focused_profile.data_safety_sha256,
    )
    focused_small = _dynamic_report(
        key=focused_small_key,
        expected_by_position=(6561, 2187, 729, 243),
        expected_total=9720,
        mode="inconclusive",
        salt="focused-small",
    )
    focused_validation = _dynamic_report(
        key=focused_validation_key,
        expected_by_position=(4096,),
        expected_total=4096,
        mode="inconclusive",
        salt="focused-validation",
    )
    focused_static = _static_report(
        (focused_small_key, focused_validation_key)
    )
    focused_certificate = compose_prefix_certificate(
        profile=focused_profile,
        small_report=focused_small,
        validation_report=focused_validation,
        static_report=focused_static,
    )
    bundle = H6PrefixReportBundle(
        profile=focused_profile,
        small_report=focused_small,
        validation_report=focused_validation,
        static_report=focused_static,
        certificate=focused_certificate,
    )

    def raw_arm(config: ArmConfig) -> dict[str, object]:
        payload = config.canonical_payload()
        payload.pop("capacity_allocation_sha256")
        return payload

    def raw_structure(structure: H6LanguageStructure) -> dict[str, object]:
        return {
            "base": {"base_id": "C0", "points": ["*"], "dimension": 0},
            "dag": {
                "labeling": "zero_based",
                "node_labels": list(structure.dag.node_labels),
                "rows": [
                    {
                        "receiver_t": row.receiver_t,
                        "parents": list(row.parents),
                    }
                    for row in structure.dag.rows
                ],
            },
            "receiver_labels": list(structure.receiver_labels),
        }

    raw_profile = {
        "profile_id": focused_profile.profile_id,
        "small_arm_config": raw_arm(small_config),
        "production_arm_config": raw_arm(production_config),
        "estimator": {
            "schema_version": focused_estimator.schema_version,
            "kind": focused_estimator.kind,
            "particle_count": focused_estimator.particle_count,
            "resampling": focused_estimator.resampling,
            "dtype": focused_estimator.dtype,
            "device": focused_estimator.device,
        },
        "small_structure": raw_structure(focused_profile.small_structure),
        "production_structure": raw_structure(
            focused_profile.production_structure
        ),
        "data_safety_sha256": focused_profile.data_safety_sha256,
        "small_model_family_sha256": (
            focused_profile.small_model_family_sha256
        ),
        "production_model_family_sha256": (
            focused_profile.production_model_family_sha256
        ),
        "profile_pair_sha256": focused_profile.profile_pair_sha256,
    }
    resolved = resolve_h6_prefix_config(
        {
            "schema_version": "h6-prefix-config-v1",
            "operation": "H6-Prefix",
            "source": {
                "git_head": validation_key.git_head,
                "dirty_digest": validation_key.dirty_digest,
                "source_sha256": "3" * 64,
            },
            "execution_mode": "focused_subset",
            "profiles": [raw_profile],
            "authorization_sha256": None,
            "artifact_root": str(artifact_root),
        },
        repo_root=tmp_path / "synthetic-repository",
    )
    config_payload = json.loads(resolved.canonical_json)
    assert config_payload["profiles"][0]["profile_pair_sha256"] == (
        focused_profile.profile_pair_sha256
    )
    assert "profile_pair_sha256s" not in config_payload
    stale_config_payload = dict(config_payload)
    stale_config_payload["profiles"] = [
        {
            **config_payload["profiles"][0],
            "profile_pair_sha256": "f" * 64,
        }
    ]
    with pytest.raises(ValueError, match="ordered embedded"):
        h6_prefix_artifact_payloads(
            config_payload=stale_config_payload,
            provenance_payload={
                "schema_version": "h6-prefix-provenance-v1",
                "git_head": validation_key.git_head,
                "dirty_digest": validation_key.dirty_digest,
                "source_sha256": "3" * 64,
                "junit_sha256": None,
            },
            environment_payload={
                "schema_version": "h6-prefix-environment-v1",
                "device": "cpu",
                "dtype": "float64",
            },
            report_bundles=(bundle,),
        )
    legacy_config_payload = {
        "schema_version": "h6-prefix-config-v1",
        "operation": "H6-Prefix",
        "profile_pair_sha256s": [focused_profile.profile_pair_sha256],
    }
    with pytest.raises(ValueError, match="exact resolved"):
        h6_prefix_artifact_payloads(
            config_payload=legacy_config_payload,
            provenance_payload={
                "schema_version": "h6-prefix-provenance-v1",
                "git_head": validation_key.git_head,
                "dirty_digest": validation_key.dirty_digest,
                "source_sha256": "3" * 64,
                "junit_sha256": None,
            },
            environment_payload={
                "schema_version": "h6-prefix-environment-v1",
                "device": "cpu",
                "dtype": "float64",
            },
            report_bundles=(bundle,),
        )
    provenance_payload = {
        "schema_version": "h6-prefix-provenance-v1",
        "git_head": validation_key.git_head,
        "dirty_digest": validation_key.dirty_digest,
        "source_sha256": "3" * 64,
        "junit_sha256": None,
    }
    environment_payload = {
        "schema_version": "h6-prefix-environment-v1",
        "device": "cpu",
        "dtype": "float64",
    }
    payloads = h6_prefix_artifact_payloads(
        config_payload=config_payload,
        provenance_payload=provenance_payload,
        environment_payload=environment_payload,
        report_bundles=(bundle,),
    )
    assert tuple(sorted(payloads)) == (
        "certificates/prefix_set.json",
        "config.json",
        "environment.json",
        "provenance.json",
        "validation/h6_prefix.json",
    )

    result, run_dir = publish_h6_prefix_artifact(
        artifact_root=artifact_root,
        run_name="synthetic-h6-prefix",
        config_payload=config_payload,
        provenance_payload=provenance_payload,
        environment_payload=environment_payload,
        report_bundles=(bundle,),
    )
    assert result.status.value == "inconclusive"
    assert tuple(
        sorted(
            path.relative_to(run_dir).as_posix()
            for path in run_dir.rglob("*")
            if path.is_file()
        )
    ) == (
        "certificates/prefix_set.json",
        "config.json",
        "environment.json",
        "manifest.sha256",
        "provenance.json",
        "validation/h6_prefix.json",
    )
    artifact_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in run_dir.rglob("*.json")
    )
    for forbidden in (
        '"H1"',
        '"H2"',
        '"H3"',
        '"H4"',
        '"H5"',
        "predecessor",
        "checkpoint",
        "test_opening",
    ):
        assert forbidden not in artifact_text
    with pytest.raises(ArtifactPublicationError, match="already exists"):
        publish_h6_prefix_artifact(
            artifact_root=artifact_root,
            run_name="synthetic-h6-prefix",
            config_payload=config_payload,
            provenance_payload=provenance_payload,
            environment_payload=environment_payload,
            report_bundles=(bundle,),
        )


def _bounded_profile_ladder() -> tuple[H6PrefixProfilePair, ...]:
    base = _categorical_profile(ArmId.A2)
    return tuple(
        H6PrefixProfilePair.create(
            profile_id=f"a2-weighted-smc-{particle_count}",
            small_arm_config=base.small_arm_config,
            production_arm_config=base.production_arm_config,
            estimator=EstimatorSpec.create(
                kind="weighted_smc",
                particle_count=particle_count,
                resampling="systematic_ess_half",
            ),
            small_structure=base.small_structure,
            production_structure=base.production_structure,
            data_safety_sha256=base.data_safety_sha256,
            small_model_family_sha256=base.small_model_family_sha256,
            production_model_family_sha256=base.production_model_family_sha256,
        )
        for particle_count in (128, 256, 512, 1024)
    )


def _bounded_resolved_config() -> H6PrefixResolvedConfig:
    source = H6SourceIdentity("1" * 40, "2" * 64, "3" * 64)
    profiles = _bounded_profile_ladder()
    workload = H6PrefixWorkloadPlan()
    artifact_root = Path("artifacts")
    payload = {
        "schema_version": "h6-prefix-config-v2",
        "operation": "H6-Prefix",
        "source": {
            "git_head": source.git_head,
            "dirty_digest": source.dirty_digest,
            "source_sha256": source.source_sha256,
        },
        "execution_mode": "authorized_full",
        "profiles": tuple(
            h6_prefix_gate._resolved_profile_payload(profile)
            for profile in profiles
        ),
        "artifact_root": artifact_root.as_posix(),
        "workload_plan": workload.canonical_payload(),
        "workload_plan_sha256": workload.workload_plan_sha256,
        "authorization_sha256": H6_PREFIX_V2_AUTHORIZATION_SHA256,
    }
    canonical = canonical_json_bytes(payload)
    return H6PrefixResolvedConfig(
        schema_version="h6-prefix-config-v2",
        operation="H6-Prefix",
        source=source,
        execution_mode="authorized_full",
        profiles=profiles,
        workload_plan=workload,
        workload_plan_sha256=workload.workload_plan_sha256,
        authorization_sha256=H6_PREFIX_V2_AUTHORIZATION_SHA256,
        artifact_root=artifact_root,
        canonical_json=canonical.decode("ascii"),
        config_sha256=hashlib.sha256(canonical).hexdigest(),
    )


def test_bounded_runner_plan_freezes_semantic_groups_jobs_and_call_budget() -> None:
    config = _bounded_resolved_config()

    plan = h6_prefix_gate._build_h6_prefix_runner_plan(config)

    assert len(plan.semantic_families) == 1
    family = plan.semantic_families[0]
    assert tuple(
        (
            job.particle_count,
            job.case_family,
            job.scope,
            job.expected_case_count,
            job.collect_source_masks,
            job.collect_validation_safety,
        )
        for job in family.dynamic_jobs
    ) == (
        (128, "small", "representative_exhaustive", 9720, True, False),
        (128, "validation", "representative_exhaustive", 4096, True, True),
        (256, "small", "estimator_stratified", 16, False, False),
        (256, "validation", "estimator_stratified", 16, False, False),
        (512, "small", "estimator_stratified", 16, False, False),
        (512, "validation", "estimator_stratified", 16, False, False),
        (1024, "small", "estimator_stratified", 16, False, False),
        (1024, "validation", "estimator_stratified", 16, False, False),
    )
    assert (
        family.arm_build_count,
        family.predictor_boundary_count,
        family.dynamic_report_count,
        family.representative_mask_collector_count,
        family.expected_case_count,
        family.expected_predictor_call_count,
        family.expected_particle_call_units,
    ) == (2, 8, 8, 2, 13_912, 69_560, 9_128_960)
    assert plan.fixture_load_count == plan.static_audit_count == 1
    assert plan.static_audit_job.report_keys == tuple(
        job.report_key for job in family.dynamic_jobs
    )


def test_bounded_runner_fake_executor_builds_once_per_family_and_never_adds_a_sixth_call() -> None:
    plan = h6_prefix_gate._build_h6_prefix_runner_plan(
        _bounded_resolved_config()
    )
    events: list[tuple[object, ...]] = []

    class FakeExecutor:
        def load_validation_perturbations(
            self, *, expected_vocabulary: VocabularyIdentity
        ) -> object:
            events.append(("fixture", expected_vocabulary))
            return object()

        def build_arm(
            self,
            *,
            arm_config: ArmConfig,
            structure: H6LanguageStructure,
        ) -> object:
            events.append(("build", arm_config, structure))
            return (arm_config, structure)

        def build_predictor_boundary(
            self,
            *,
            built_arm: object,
            estimator: EstimatorSpec,
        ) -> object:
            events.append(("boundary", built_arm, estimator.particle_count))
            return (built_arm, estimator)

        def execute_dynamic_job(
            self,
            *,
            job: object,
            built_arm: object,
            predictor: object,
            validation_perturbations: object,
        ) -> object:
            del built_arm, predictor, validation_perturbations
            events.append(
                (
                    "dynamic",
                    job.particle_count,
                    job.case_family,
                    job.selected_global_indices,
                )
            )
            return SimpleNamespace(
                report=object(),
                observed_predictor_call_count=(
                    job.expected_predictor_call_count
                ),
            )

        def execute_static_audit(self, *, job: object) -> object:
            events.append(("static", job.report_keys))
            return object()

    evidence = h6_prefix_gate._execute_h6_prefix_plan(plan, FakeExecutor())

    assert tuple(event[0] for event in events) == (
        ("fixture", "build", "build")
        + ("boundary", "boundary", "dynamic", "dynamic") * 4
        + ("static",)
    )
    assert len([event for event in events if event[0] == "build"]) == 2
    assert len([event for event in events if event[0] == "boundary"]) == 8
    assert len([event for event in events if event[0] == "dynamic"]) == 8
    assert evidence.observed_predictor_call_count == 69_560
    assert all(
        result.observed_predictor_call_count
        == result.job.expected_case_count * 5
        for result in evidence.dynamic_results
    )


def test_source_mask_observer_reuses_first_prediction_without_a_sixth_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arm_config = _categorical_arm_config(
        arm=ArmId.A2,
        vocabulary=VocabularyIdentity("h6-prefix-small-v1", 3, "a" * 64),
        horizon=4,
    )
    estimator = EstimatorSpec.create(
        kind="weighted_smc",
        particle_count=4,
        resampling="systematic_ess_half",
    )
    case = DynamicPrefixCase.create(
        ordinal=0,
        receiver_t=2,
        shared_prefix=(1,),
        left_tail=(0,),
        right_tail=(2,),
    )
    validated: list[object] = []
    observed: list[object] = []

    class TinyIdentity:
        tensor = "same"

        def cache_payload(self) -> dict[str, str]:
            return {"cache": "same"}

        def tensor_payload(self) -> dict[str, str]:
            return {"tensor": "same"}

    identity = TinyIdentity()

    class TinyDeterministicPredictor:
        vocabulary = arm_config.vocabulary
        estimator_identity = EstimatorIdentity.from_spec(estimator)
        model_state_sha256 = "c" * 64
        proposal_identity_sha256 = "d" * 64
        call_count = 0

        def next_token_log_probs(
            self,
            prefix_tokens: object,
            estimator_rng: object,
            cache: object = None,
        ) -> object:
            del prefix_tokens, estimator_rng, cache
            self.call_count += 1
            return SimpleNamespace(
                cache=object(),
                call_ordinal=self.call_count,
            )

    predictor = TinyDeterministicPredictor()

    def validate_prediction(prediction: object, **_: object) -> object:
        validated.append(prediction)
        return identity

    def observe_first(
        observed_case: DynamicPrefixCase,
        prediction: object,
    ) -> tuple[object, ...]:
        assert observed_case is case
        assert validated[-1] is prediction
        observed.append(prediction)
        return ()

    monkeypatch.setattr(
        "vfe4.validation.h6_prefix._signature_and_identity_assessment",
        lambda *_: None,
    )
    monkeypatch.setattr(
        "vfe4.validation.h6_prefix._validate_prediction_identity",
        validate_prediction,
    )

    run_dynamic_prefix_checks(
        key=_key(
            config=arm_config,
            estimator=estimator,
            model_family_sha256=_model_family_sha256(arm_config),
            data_safety_sha256=hashlib.sha256(
                b"VFE4-H6-TARGET-FREE-PREDICTIVE-BOUNDARY-V1"
            ).hexdigest(),
        ),
        predictor=predictor,
        arm_config=arm_config,
        cases=(case,),
        plan=DynamicExecutionPlan.create(
            mode="focused_subset",
            case_family="small",
        ),
        stream_seed=2026072197,
        pair_side_harness=PairSideHarness(),
        source_mask_observer=observe_first,
    )

    assert predictor.call_count == 5
    assert len(validated) == 4
    assert len(observed) == 1
    assert observed[0].call_ordinal == 1
