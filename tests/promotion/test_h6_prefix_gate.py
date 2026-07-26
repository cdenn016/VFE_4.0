from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import verification.h6_prefix_gate as h6_prefix_gate
import verification.h6_validation_candidate as h6_validation_candidate
from verification.h6_prefix_gate import (
    H6PrefixReportBundle,
    compose_prefix_certificate,
    h6_prefix_artifact_payloads,
    publish_h6_prefix_artifact,
)
from vfe4.artifacts.atomic import ArtifactPublicationError
from vfe4.artifacts.provenance import source_candidate_sha256
from vfe4.config import (
    H6PrefixResolvedConfig,
    H6PrefixV3ResolvedConfig,
    H6SourceIdentity,
    resolve_h6_prefix_config,
)
from vfe4.config.schema import (
    H6_PREFIX_V2_AUTHORIZATION_SHA256,
    H6_PREFIX_V3_AUTHORIZATION_SHA256,
)
from vfe4.h6_validation_fixture import (
    H6ValidationPerturbationArtifactReference,
    ValidationSafetyFixturePayload,
    ValidationSafetyFixtureReference,
)
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
_V3_TOKENIZER_SPEC_SHA256 = (
    "1c924ca10bed173c8aaa0e2cb6389df02524269d6405bb1339aa3903834689d4"
)
_V3_VOCABULARY_SHA256 = (
    "5aea771bc9b54b0e6ad0ce9b5cddbd6d32e89a4201e4f9cd11bb00bf8713dd68"
)
_V3_FIXTURE_DOMAIN = b"VFE4-H6-VALIDATION-SAFETY-FIXTURE-V1\x00"
_V3_FIXTURE_ROW = struct.Struct("<QH33H")


def _v3_fixture_bytes() -> bytes:
    raw = bytearray(
        _V3_FIXTURE_DOMAIN + bytes.fromhex("7" * 64) + struct.pack("<I", 4096)
    )
    token_ids = (0,) * 33
    for index in range(4096):
        raw += _V3_FIXTURE_ROW.pack(index, 1, *token_ids)
    result = bytes(raw)
    assert len(result) == 311_369
    return result


_V3_FIXTURE_BYTES = _v3_fixture_bytes()
_V3_CANDIDATE_BYTES = b'{"synthetic_v3_candidate":true}'


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


def _categorical_profile(
    arm: ArmId,
    *,
    production_tokenizer_spec_sha256: str = "b" * 64,
) -> H6PrefixProfilePair:
    small = _categorical_arm_config(
        arm=arm,
        vocabulary=VocabularyIdentity("h6-prefix-small-v1", 3, "a" * 64),
        horizon=4,
    )
    production = _categorical_arm_config(
        arm=arm,
        vocabulary=VocabularyIdentity(
            "wikitext-2-byte-v1",
            258,
            production_tokenizer_spec_sha256,
        ),
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


def _bounded_profile_ladder(
    *,
    data_safety_sha256: str | None = None,
    production_tokenizer_spec_sha256: str = "b" * 64,
) -> tuple[H6PrefixProfilePair, ...]:
    base = _categorical_profile(
        ArmId.A2,
        production_tokenizer_spec_sha256=(
            production_tokenizer_spec_sha256
        ),
    )
    resolved_data_safety_sha256 = (
        base.data_safety_sha256
        if data_safety_sha256 is None
        else data_safety_sha256
    )
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
            data_safety_sha256=resolved_data_safety_sha256,
            small_model_family_sha256=base.small_model_family_sha256,
            production_model_family_sha256=base.production_model_family_sha256,
        )
        for particle_count in (128, 256, 512, 1024)
    )


def _bounded_resolved_config(
    tmp_path: Path,
    *,
    data_safety_sha256: str | None = None,
) -> H6PrefixResolvedConfig:
    source = H6SourceIdentity("1" * 40, "2" * 64, "3" * 64)
    profiles = _bounded_profile_ladder(
        data_safety_sha256=data_safety_sha256
    )
    workload = H6PrefixWorkloadPlan()
    artifact_root = (tmp_path / "bounded-artifacts").resolve()
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


def _v3_references(
    tmp_path: Path,
) -> tuple[
    ValidationSafetyFixtureReference,
    H6ValidationPerturbationArtifactReference,
]:
    fixture = ValidationSafetyFixtureReference.create(
        local_payload_path=(
            tmp_path / "v3-fixture" / "validation_safety_fixture.bin"
        ),
        binary_directory_manifest_sha256="4" * 64,
        data_identity_sha256="5" * 64,
        access_policy_sha256="6" * 64,
        validation_token_sha256="7" * 64,
        fixture_raw_sha256=hashlib.sha256(_V3_FIXTURE_BYTES).hexdigest(),
        fixture_raw_length=311_369,
        row_count=4096,
    )
    git_head = "8" * 40
    dirty_digest = "9" * 64
    source_sha256 = hashlib.sha256(
        b"VFE4-H6-SOURCE-CANDIDATE-V1\x00"
        + bytes.fromhex(git_head)
        + bytes.fromhex(dirty_digest)
    ).hexdigest()
    payload_sha256s = (
        ("config.json", "a" * 64),
        ("provenance.json", "b" * 64),
        (
            "validation/h6_validation_perturbations_v1.json",
            "c" * 64,
        ),
    )
    identity = {
        "access_policy_sha256": fixture.access_policy_sha256,
        "binary_directory_manifest_sha256": (
            fixture.binary_directory_manifest_sha256
        ),
        "config_sha256": "d" * 64,
        "data_identity_sha256": fixture.data_identity_sha256,
        "directory_manifest_sha256": "e" * 64,
        "fixture_raw_sha256": fixture.fixture_raw_sha256,
        "full_count": 4096,
        "generator_version": "h6-validation-perturbations-v1",
        "materialized_count": 4096,
        "payload_sha256s": [
            {"path": path, "sha256": sha256}
            for path, sha256 in payload_sha256s
        ],
        "perturbation_inner_manifest_sha256": "f" * 64,
        "perturbation_raw_sha256": hashlib.sha256(
            _V3_CANDIDATE_BYTES
        ).hexdigest(),
        "perturbation_schema_version": "h6-validation-perturbations-v1",
        "seed": 2026072197,
        "source": {
            "dirty_digest": dirty_digest,
            "git_head": git_head,
            "source_sha256": source_sha256,
        },
        "validation_fixture_reference_sha256": fixture.reference_sha256,
        "validation_token_sha256": fixture.validation_token_sha256,
        "vocabulary": {
            "size": 258,
            "tokenizer_spec_sha256": _V3_TOKENIZER_SPEC_SHA256,
            "vocabulary_id": "wikitext-2-byte-v1",
            "vocabulary_sha256": _V3_VOCABULARY_SHA256,
        },
    }
    reference_sha256 = hashlib.sha256(
        b"vfe4.h6.validation-perturbation-artifact-reference.v1\x00"
        + canonical_json_bytes(identity)
    ).hexdigest()
    candidate = H6ValidationPerturbationArtifactReference(
        local_artifact_path=(tmp_path / "v3-candidate").resolve(),
        git_head=git_head,
        dirty_digest=dirty_digest,
        source_sha256=source_sha256,
        config_sha256="d" * 64,
        validation_fixture_reference_sha256=fixture.reference_sha256,
        binary_directory_manifest_sha256=(
            fixture.binary_directory_manifest_sha256
        ),
        data_identity_sha256=fixture.data_identity_sha256,
        access_policy_sha256=fixture.access_policy_sha256,
        validation_token_sha256=fixture.validation_token_sha256,
        fixture_raw_sha256=fixture.fixture_raw_sha256,
        vocabulary_id="wikitext-2-byte-v1",
        vocabulary_size=258,
        tokenizer_spec_sha256=_V3_TOKENIZER_SPEC_SHA256,
        vocabulary_sha256=_V3_VOCABULARY_SHA256,
        perturbation_schema_version="h6-validation-perturbations-v1",
        generator_version="h6-validation-perturbations-v1",
        seed=2026072197,
        full_count=4096,
        materialized_count=4096,
        perturbation_inner_manifest_sha256="f" * 64,
        perturbation_raw_sha256=hashlib.sha256(
            _V3_CANDIDATE_BYTES
        ).hexdigest(),
        payload_sha256s=payload_sha256s,
        directory_manifest_sha256="e" * 64,
        reference_sha256=reference_sha256,
    )
    return fixture, candidate


def _bounded_v3_resolved_config(tmp_path: Path) -> H6PrefixV3ResolvedConfig:
    fixture, candidate = _v3_references(tmp_path)
    profiles = _bounded_profile_ladder(
        production_tokenizer_spec_sha256=_V3_TOKENIZER_SPEC_SHA256
    )
    workload = H6PrefixWorkloadPlan()
    resolved = resolve_h6_prefix_config(
        {
            "schema_version": "h6-prefix-config-v3",
            "operation": "H6-Prefix",
            "source": {
                "git_head": "1" * 40,
                "dirty_digest": "2" * 64,
                "source_sha256": "3" * 64,
            },
            "execution_mode": "authorized_full",
            "profiles": [
                h6_prefix_gate._resolver_profile_payload(profile)
                for profile in profiles
            ],
            "workload_plan_sha256": workload.workload_plan_sha256,
            "workload_authorization_sha256": (
                H6_PREFIX_V2_AUTHORIZATION_SHA256
            ),
            "validation_fixture_reference": fixture.to_payload(),
            "validation_perturbation_reference": candidate.to_payload(),
            "authorization_sha256": H6_PREFIX_V3_AUTHORIZATION_SHA256,
            "artifact_root": str(
                (tmp_path / "bounded-v3-artifacts").resolve()
            ),
        },
        repo_root=Path(__file__).resolve().parents[2],
    )
    assert type(resolved) is H6PrefixV3ResolvedConfig
    return resolved


def _v3_fixture_payload(
    config: H6PrefixV3ResolvedConfig,
) -> ValidationSafetyFixturePayload:
    reference = config.validation_fixture_reference
    return ValidationSafetyFixturePayload(
        reference=reference,
        fixture_bytes=_V3_FIXTURE_BYTES,
        validation_token_sha256=reference.validation_token_sha256,
        starts=tuple(range(4096)),
        real_target_counts=(1,) * 4096,
    )


def _v3_full_perturbations(
    config: H6PrefixV3ResolvedConfig,
    expected_vocabulary: VocabularyIdentity,
    *,
    materialized_count: int = 4096,
) -> SimpleNamespace:
    reference = config.validation_perturbation_reference
    return SimpleNamespace(
        schema_version="h6-validation-perturbations-v1",
        generator_version="h6-validation-perturbations-v1",
        seed=2026072197,
        vocabulary=expected_vocabulary,
        vocabulary_sha256=reference.vocabulary_sha256,
        validation_token_sha256=reference.validation_token_sha256,
        validation_safety_fixture_sha256=reference.fixture_raw_sha256,
        full_count=4096,
        materialized_count=materialized_count,
        materialization=(
            "authorized_full"
            if materialized_count == 4096
            else "focused_subset"
        ),
        records=tuple(
            SimpleNamespace(case_index=index)
            for index in range(materialized_count)
        ),
        manifest_sha256=(
            reference.perturbation_inner_manifest_sha256
        ),
        source_fixture_verified=True,
        raw_sha256=reference.perturbation_raw_sha256,
        canonical_bytes=_V3_CANDIDATE_BYTES,
    )


class _V3FakeExecutor:
    def __init__(
        self,
        *,
        loader: object,
        events: list[str],
    ) -> None:
        self.loader = loader
        self.events = events

    def load_validation_perturbations(
        self,
        *,
        expected_vocabulary: VocabularyIdentity,
    ) -> object:
        return self.loader(expected_vocabulary)

    def build_arm(
        self,
        *,
        arm_config: ArmConfig,
        structure: H6LanguageStructure,
    ) -> object:
        self.events.append("build-arm")
        return (arm_config, structure)

    def build_predictor_boundary(
        self,
        *,
        built_arm: object,
        estimator: EstimatorSpec,
    ) -> object:
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
        execution_plan = _bounded_execution_plan(job)
        return SimpleNamespace(
            execution_plan=execution_plan,
            report=_bounded_dynamic_report(
                job=job,
                execution_plan=execution_plan,
            ),
            observed_predictor_call_count=(
                job.expected_predictor_call_count
            ),
        )

    def execute_static_audit(self, *, job: object) -> object:
        return _static_report(job.report_keys)


def test_bounded_runner_plan_freezes_semantic_groups_jobs_and_call_budget(
    tmp_path: Path,
) -> None:
    config = _bounded_resolved_config(tmp_path)

    plan = h6_prefix_gate._build_h6_prefix_runner_plan(config)

    assert plan.config.artifact_root == (
        tmp_path / "bounded-artifacts"
    ).resolve()
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
    resolver_invalid = _bounded_resolved_config(
        tmp_path,
        data_safety_sha256="e" * 64,
    )
    with pytest.raises(
        ValueError,
        match="implemented H6 predictor safety boundary",
    ):
        h6_prefix_gate._build_h6_prefix_runner_plan(resolver_invalid)


def _bounded_execution_plan(job: object) -> DynamicExecutionPlan:
    selection_rows = tuple(
        (
            index,
            _owned_hash(
                "test.h6.bounded-selection-case",
                {
                    "scope": job.scope,
                    "case_family": job.case_family,
                    "particle_count": job.particle_count,
                    "global_index": index,
                },
            ),
        )
        for index in job.selected_global_indices
    )
    return DynamicExecutionPlan.create_scoped(
        scope=job.scope,
        case_family=job.case_family,
        particle_count=job.particle_count,
        workload_plan=H6PrefixWorkloadPlan(),
        authorization_sha256=H6_PREFIX_V2_AUTHORIZATION_SHA256,
        selection_rows=selection_rows,
    )


def _bounded_dynamic_report(
    *,
    job: object,
    execution_plan: DynamicExecutionPlan,
    key: PrefixCaseKey | None = None,
) -> DynamicPrefixReport:
    expected_total = sum(execution_plan.expected_by_position)
    applicable = (
        _DYNAMIC_CHECK_NAMES
        if job.scope == "representative_exhaustive"
        else _DYNAMIC_CHECK_NAMES[:3]
    )
    checks = tuple(
        DynamicCheckResult.create(
            name=name,
            status=EvidenceStatus.PASS,
            expected_count=(
                0
                if name not in applicable
                else expected_total
            ),
            completed_count=(
                0
                if name not in applicable
                else expected_total
            ),
        )
        for name in _DYNAMIC_CHECK_NAMES
    )
    salt = {
        "scope": job.scope,
        "case_family": job.case_family,
        "particle_count": job.particle_count,
    }
    values = {
        "schema_version": "h6-dynamic-prefix-report-v2",
        "key": job.report_key if key is None else key,
        "execution_plan_sha256": execution_plan.plan_sha256,
        "model_state_sha256": _owned_hash("test.h6.bounded-model", salt),
        "proposal_identity_sha256": _owned_hash(
            "test.h6.bounded-proposal", salt
        ),
        "estimator_semantic_sha256": _owned_hash(
            "test.h6.bounded-estimator-semantic", salt
        ),
        "estimator_artifact_bytes_sha256": _owned_hash(
            "test.h6.bounded-estimator-artifact", salt
        ),
        "stream_seed": 2026072197,
        "completed_by_position": execution_plan.expected_by_position,
        "checks": checks,
        "status": EvidenceStatus.PASS,
        "obligations": (),
        "unresolved_diagnostics": (),
        "first_counterexample": None,
        "case_result_manifest_sha256": _owned_hash(
            "test.h6.bounded-case-results", salt
        ),
        "cache_manifest_sha256": _owned_hash(
            "test.h6.bounded-cache", salt
        ),
        "pair_harness_manifest_sha256": _owned_hash(
            "test.h6.bounded-pairs", salt
        ),
        "mask_manifest_sha256": _owned_hash(
            "test.h6.bounded-masks", salt
        ),
        "complete_case_manifest_sha256": (
            _owned_hash("test.h6.bounded-complete-cases", salt)
            if job.scope == "representative_exhaustive"
            else None
        ),
        "scope": job.scope,
        "case_family": job.case_family,
        "particle_count": job.particle_count,
        "workload_plan_sha256": execution_plan.workload_plan_sha256,
        "selected_global_indices": job.selected_global_indices,
        "selection_manifest_sha256": (
            execution_plan.selection_manifest_sha256
        ),
        "applicable_check_names": applicable,
    }
    provisional = object.__new__(DynamicPrefixReport)
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    return DynamicPrefixReport(
        **values,
        report_sha256=_owned_hash(
            "vfe4.h6.dynamic-prefix-report.v2",
            provisional.canonical_payload(),
        ),
    )


def _bounded_certificate_job(
    profile: H6PrefixProfilePair,
    case_family: str,
    *,
    key: PrefixCaseKey | None = None,
) -> SimpleNamespace:
    workload = H6PrefixWorkloadPlan()
    representative = profile.estimator.particle_count == 128
    scope = (
        "representative_exhaustive"
        if representative
        else "estimator_stratified"
    )
    selected = (
        tuple(range(9720 if case_family == "small" else 4096))
        if representative
        else (
            workload.small_global_case_indices
            if case_family == "small"
            else workload.validation_global_case_indices
        )
    )
    config = (
        profile.small_arm_config
        if case_family == "small"
        else profile.production_arm_config
    )
    model_family_sha256 = (
        profile.small_model_family_sha256
        if case_family == "small"
        else profile.production_model_family_sha256
    )
    return SimpleNamespace(
        profile=profile,
        case_family=case_family,
        scope=scope,
        particle_count=profile.estimator.particle_count,
        selected_global_indices=selected,
        report_key=(
            _key(
                config=config,
                estimator=profile.estimator,
                model_family_sha256=model_family_sha256,
                data_safety_sha256=profile.data_safety_sha256,
            )
            if key is None
            else key
        ),
    )


def _bounded_certificate_report(
    profile: H6PrefixProfilePair,
    case_family: str,
    *,
    mode: str = "pass",
    key: PrefixCaseKey | None = None,
    selection_tag: str = "shared",
    model_tag: str = "shared",
) -> DynamicPrefixReport:
    job = _bounded_certificate_job(profile, case_family, key=key)
    selection_rows = tuple(
        (
            index,
            _owned_hash(
                "test.h6.bounded-certificate-selection",
                {
                    "case_family": case_family,
                    "global_index": index,
                    "tag": selection_tag,
                },
            ),
        )
        for index in job.selected_global_indices
    )
    execution_plan = DynamicExecutionPlan.create_scoped(
        scope=job.scope,
        case_family=job.case_family,
        particle_count=job.particle_count,
        workload_plan=H6PrefixWorkloadPlan(),
        authorization_sha256=H6_PREFIX_V2_AUTHORIZATION_SHA256,
        selection_rows=selection_rows,
    )
    base = _bounded_dynamic_report(job=job, execution_plan=execution_plan)
    applicable = base.applicable_check_names or ()
    checks = []
    for check in base.checks:
        status = EvidenceStatus.PASS
        violation_count = 0
        counterexample = None
        obligations: tuple[str, ...] = ()
        if mode == "fail" and check.name == "dynamic_target_suffix_leakage":
            status = EvidenceStatus.FAIL
            violation_count = 1
            counterexample = "synthetic bounded leakage witness"
        elif mode == "inconclusive" and check.name == "signature_and_identity":
            status = EvidenceStatus.INCONCLUSIVE
            obligations = ("synthetic bounded identity obligation",)
        checks.append(
            DynamicCheckResult.create(
                name=check.name,
                status=status,
                expected_count=check.expected_count,
                completed_count=check.completed_count,
                violation_count=violation_count,
                first_counterexample=counterexample,
                obligations=obligations,
            )
            if check.name in applicable
            else check
        )
    estimator_identity = EstimatorIdentity.from_spec(profile.estimator)
    values = {
        descriptor.name: getattr(base, descriptor.name)
        for descriptor in fields(DynamicPrefixReport)
        if descriptor.name != "report_sha256"
    }
    values.update(
        {
            "model_state_sha256": _owned_hash(
                "test.h6.bounded-certificate-model",
                {"case_family": case_family, "tag": model_tag},
            ),
            "proposal_identity_sha256": _owned_hash(
                "test.h6.bounded-certificate-proposal",
                {"case_family": case_family, "tag": model_tag},
            ),
            "estimator_semantic_sha256": estimator_identity.semantic_sha256,
            "estimator_artifact_bytes_sha256": (
                estimator_identity.artifact_bytes_sha256
            ),
            "checks": tuple(checks),
            "status": (
                EvidenceStatus.FAIL
                if mode == "fail"
                else EvidenceStatus.INCONCLUSIVE
                if mode == "inconclusive"
                else EvidenceStatus.PASS
            ),
            "obligations": (
                ("synthetic bounded report obligation",)
                if mode == "inconclusive"
                else ()
            ),
            "first_counterexample": (
                "synthetic bounded leakage witness"
                if mode == "fail"
                else None
            ),
        }
    )
    provisional = object.__new__(DynamicPrefixReport)
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    return DynamicPrefixReport(
        **values,
        report_sha256=_owned_hash(
            "vfe4.h6.dynamic-prefix-report.v2",
            provisional.canonical_payload(),
        ),
    )


def _bounded_certificate_fixture(
    *,
    report_mode: tuple[int, str] | None = None,
) -> SimpleNamespace:
    profiles = _bounded_profile_ladder()
    reports = tuple(
        _bounded_certificate_report(
            profile,
            case_family,
            mode=(
                report_mode[1]
                if report_mode is not None and report_mode[0] == report_index
                else "pass"
            ),
        )
        for report_index, (profile, case_family) in enumerate(
            (profile, case_family)
            for profile in profiles
            for case_family in ("small", "validation")
        )
    )
    keys = tuple(report.key for report in reports)
    return SimpleNamespace(
        profiles=profiles,
        reports=reports,
        global_keys=keys,
        static_report=_static_report(keys),
        source_sha256=source_candidate_sha256(
            git_head_value="1" * 40,
            dirty_digest_value="2" * 64,
        ),
    )


def _bounded_family_bundle(
    profiles: tuple[H6PrefixProfilePair, ...],
    reports: tuple[DynamicPrefixReport, ...],
) -> object:
    return h6_prefix_gate.H6BoundedPrefixFamilyBundle(
        profiles=profiles,
        reports=reports,
    )


def _chain_structure(horizon: int) -> H6LanguageStructure:
    base = ZeroDimensionalBase.create()
    return H6LanguageStructure.create(
        base=base,
        dag=CausalDag.create(
            node_labels=tuple(range(horizon + 1)),
            rows=tuple(
                CausalDagRow(receiver, (receiver - 1,))
                for receiver in range(1, horizon + 1)
            ),
        ),
        receiver_labels=tuple(range(1, horizon + 1)),
    )


def _unsafe_stale_static_identity(
    report: StaticAuditReport,
    field_name: str,
) -> StaticAuditReport:
    stale = object.__new__(StaticAuditReport)
    for descriptor in fields(StaticAuditReport):
        object.__setattr__(
            stale,
            descriptor.name,
            "f" * 64
            if descriptor.name == field_name
            else getattr(report, descriptor.name),
        )
    return stale


def test_bounded_certificate_requires_exact_eight_report_matrix_and_global_static_report() -> None:
    fixture = _bounded_certificate_fixture()
    compose = h6_prefix_gate.compose_bounded_prefix_certificate
    certificate = compose(
        family_bundle=_bounded_family_bundle(
            fixture.profiles,
            fixture.reports,
        ),
        static_report=fixture.static_report,
        global_case_keys=fixture.global_keys,
        source_sha256=fixture.source_sha256,
    )

    assert certificate.schema_version == "h6-prefix-certificate-v2"
    assert certificate.status is EvidenceStatus.PASS
    assert certificate.obligations == ()
    assert all(certificate.checks.values())
    binding = certificate.report_binding
    assert (
        binding.schema_version
        == "h6-bounded-prefix-report-binding-v2"
    )
    assert (
        binding.workload_plan_sha256
        == H6PrefixWorkloadPlan().workload_plan_sha256
    )
    assert binding.profile_pair_sha256s == tuple(
        profile.profile_pair_sha256 for profile in fixture.profiles
    )
    assert tuple(
        (reference.particle_count, reference.case_family, reference.scope)
        for reference in binding.report_references
    ) == (
        (128, "small", "representative_exhaustive"),
        (128, "validation", "representative_exhaustive"),
        (256, "small", "estimator_stratified"),
        (256, "validation", "estimator_stratified"),
        (512, "small", "estimator_stratified"),
        (512, "validation", "estimator_stratified"),
        (1024, "small", "estimator_stratified"),
        (1024, "validation", "estimator_stratified"),
    )
    assert binding.report_references[1].completed_by_position == (4096,)
    assert len(
        {
            binding.report_references[index].selection_manifest_sha256
            for index in (2, 4, 6)
        }
    ) == 1
    assert len(
        {
            binding.report_references[index].selection_manifest_sha256
            for index in (3, 5, 7)
        }
    ) == 1
    assert certificate.semantic_family_sha256 == binding.semantic_family_sha256
    assert binding.git_head == "1" * 40
    assert binding.dirty_digest == "2" * 64
    assert binding.source_sha256 == fixture.source_sha256
    assert binding.global_case_key_order_sha256 == _owned_hash(
        "vfe4.h6.bounded-prefix-global-case-key-order.v2",
        tuple(key.canonical_payload() for key in fixture.global_keys),
    )
    assert certificate.checks["artifact_identity"] is True
    for profile_index, profile in enumerate(fixture.profiles):
        estimator_identity = EstimatorIdentity.from_spec(profile.estimator)
        for reference in binding.report_references[
            2 * profile_index : 2 * profile_index + 2
        ]:
            assert (
                reference.estimator_semantic_sha256
                == estimator_identity.semantic_sha256
            )
            assert (
                reference.estimator_artifact_bytes_sha256
                == estimator_identity.artifact_bytes_sha256
            )

    reports = fixture.reports
    source_key = PrefixCaseKey(
        **{
            **reports[0].key.__dict__,
            "dirty_digest": "d" * 64,
        }
    )
    cross_source = _bounded_certificate_report(
        fixture.profiles[0],
        "small",
        key=source_key,
    )
    wrong_model = _bounded_certificate_report(
        fixture.profiles[2],
        "small",
        model_tag="different",
    )
    wrong_selection = _bounded_certificate_report(
        fixture.profiles[2],
        "small",
        selection_tag="different",
    )
    structure_profile = H6PrefixProfilePair.create(
        profile_id=fixture.profiles[0].profile_id,
        small_arm_config=fixture.profiles[0].small_arm_config,
        production_arm_config=fixture.profiles[0].production_arm_config,
        estimator=fixture.profiles[0].estimator,
        small_structure=_chain_structure(4),
        production_structure=fixture.profiles[0].production_structure,
        data_safety_sha256=fixture.profiles[0].data_safety_sha256,
        small_model_family_sha256=(
            fixture.profiles[0].small_model_family_sha256
        ),
        production_model_family_sha256=(
            fixture.profiles[0].production_model_family_sha256
        ),
    )
    mutations = (
        (fixture.profiles, reports[:-1], fixture.static_report, fixture.global_keys),
        (
            fixture.profiles,
            (*reports[:-1], reports[-2]),
            fixture.static_report,
            fixture.global_keys,
        ),
        (
            fixture.profiles,
            (reports[1], reports[0], *reports[2:]),
            fixture.static_report,
            fixture.global_keys,
        ),
        (
            fixture.profiles,
            (reports[2], reports[1], reports[0], *reports[3:]),
            fixture.static_report,
            fixture.global_keys,
        ),
        (
            fixture.profiles,
            (cross_source, *reports[1:]),
            fixture.static_report,
            fixture.global_keys,
        ),
        (
            (structure_profile, *fixture.profiles[1:]),
            reports,
            fixture.static_report,
            fixture.global_keys,
        ),
        (
            fixture.profiles,
            (*reports[:4], wrong_model, *reports[5:]),
            fixture.static_report,
            fixture.global_keys,
        ),
        (
            fixture.profiles,
            (*reports[:4], wrong_selection, *reports[5:]),
            fixture.static_report,
            fixture.global_keys,
        ),
        (
            fixture.profiles,
            reports,
            _static_report(fixture.global_keys[:-1]),
            fixture.global_keys,
        ),
        (
            fixture.profiles,
            reports,
            fixture.static_report,
            (
                fixture.global_keys[1],
                fixture.global_keys[0],
                *fixture.global_keys[2:],
            ),
        ),
        (
            fixture.profiles,
            reports,
            _unsafe_stale_static_identity(
                fixture.static_report,
                "source_manifest_sha256",
            ),
            fixture.global_keys,
        ),
        (
            fixture.profiles,
            reports,
            _unsafe_stale_static_identity(
                fixture.static_report,
                "rules_sha256",
            ),
            fixture.global_keys,
        ),
    )
    for profiles, mutated_reports, static_report, global_keys in mutations:
        with pytest.raises(ValueError):
            compose(
                family_bundle=_bounded_family_bundle(
                    profiles,
                    mutated_reports,
                ),
                static_report=static_report,
                global_case_keys=global_keys,
                source_sha256=fixture.source_sha256,
            )
    with pytest.raises(ValueError, match="source candidate"):
        compose(
            family_bundle=_bounded_family_bundle(
                fixture.profiles,
                fixture.reports,
            ),
            static_report=fixture.static_report,
            global_case_keys=fixture.global_keys,
            source_sha256="e" * 64,
        )

    inconclusive_fixture = _bounded_certificate_fixture(
        report_mode=(4, "inconclusive")
    )
    inconclusive = compose(
        family_bundle=_bounded_family_bundle(
            inconclusive_fixture.profiles,
            inconclusive_fixture.reports,
        ),
        static_report=inconclusive_fixture.static_report,
        global_case_keys=inconclusive_fixture.global_keys,
        source_sha256=inconclusive_fixture.source_sha256,
    )
    assert inconclusive.status is EvidenceStatus.INCONCLUSIVE
    assert inconclusive.obligations
    failed_fixture = _bounded_certificate_fixture(report_mode=(6, "fail"))
    failed = compose(
        family_bundle=_bounded_family_bundle(
            failed_fixture.profiles,
            failed_fixture.reports,
        ),
        static_report=failed_fixture.static_report,
        global_case_keys=failed_fixture.global_keys,
        source_sha256=failed_fixture.source_sha256,
    )
    assert failed.status is EvidenceStatus.FAIL
    assert failed.obligations == ()


def test_scoped_pass_alone_cannot_mint_a_certificate_and_v1_stays_stable() -> None:
    fixture = _bounded_certificate_fixture()
    profile = fixture.profiles[0]
    small_key = fixture.reports[0].key
    validation_key = fixture.reports[1].key
    small_v1 = _dynamic_report(
        key=small_key,
        expected_by_position=(6561, 2187, 729, 243),
        expected_total=9720,
        mode="pass",
        salt="v1-stability-small",
    )
    validation_v1 = _dynamic_report(
        key=validation_key,
        expected_by_position=(4096,),
        expected_total=4096,
        mode="pass",
        salt="v1-stability-validation",
    )
    static_v1 = _static_report((small_key, validation_key))
    legacy = compose_prefix_certificate(
        profile=profile,
        small_report=small_v1,
        validation_report=validation_v1,
        static_report=static_v1,
    )
    binding = PrefixReportBinding.create(
        small_report_sha256=small_v1.report_sha256,
        small_case_manifest_sha256=small_v1.complete_case_manifest_sha256,
        validation_report_sha256=validation_v1.report_sha256,
        validation_case_manifest_sha256=(
            validation_v1.complete_case_manifest_sha256
        ),
        static_report_sha256=static_v1.report_sha256,
        static_source_manifest_sha256=static_v1.source_manifest_sha256,
        static_rules_sha256=static_v1.rules_sha256,
        static_case_key_manifest_sha256=(
            static_v1.case_key_manifest_sha256
        ),
    )
    expected_bytes = canonical_json_bytes(
        {
            "schema_version": "h6-prefix-certificate-validation-v2",
            "profile": {
                **profile.canonical_payload(),
                "profile_pair_sha256": profile.profile_pair_sha256,
            },
            "small_key": small_key.canonical_payload(),
            "key": validation_key.canonical_payload(),
            "report_binding": binding.canonical_payload(),
            "checks": {name: True for name in h6_prefix_gate.H6_PREFIX_REQUIRED_CHECKS},
            "status": EvidenceStatus.PASS.value,
            "obligations": (),
        }
    )
    assert legacy.validation_payload_canonical_json == expected_bytes
    assert binding.binding_sha256 == _owned_hash(
        "vfe4.h6.prefix-report-binding.v1",
        binding.canonical_payload(include_binding=False),
    )
    assert legacy.certificate_sha256 == _owned_hash(
        "vfe4.h6.prefix-certificate.v1",
        {
            "key": validation_key.canonical_payload(),
            "validation_payload_sha256": hashlib.sha256(
                expected_bytes
            ).hexdigest(),
            "status": EvidenceStatus.PASS.value,
            "obligations": (),
        },
    )

    with pytest.raises(ValueError, match="v2"):
        compose_prefix_certificate(
            profile=profile,
            small_report=fixture.reports[0],
            validation_report=fixture.reports[1],
            static_report=fixture.static_report,
        )
    compose = h6_prefix_gate.compose_bounded_prefix_certificate
    for reports in (fixture.reports[:1], fixture.reports[:2]):
        with pytest.raises(ValueError):
            compose(
                family_bundle=_bounded_family_bundle(
                    fixture.profiles[:1],
                    reports,
                ),
                static_report=fixture.static_report,
                global_case_keys=fixture.global_keys,
                source_sha256=fixture.source_sha256,
            )


def test_bounded_runner_fake_executor_builds_once_per_family_and_never_adds_a_sixth_call(
    tmp_path: Path,
) -> None:
    plan = h6_prefix_gate._build_h6_prefix_runner_plan(
        _bounded_resolved_config(tmp_path)
    )
    events: list[tuple[object, ...]] = []

    class FakeExecutor:
        def __init__(
            self,
            *,
            fault: str | None = None,
            record_events: bool = True,
        ) -> None:
            self.fault = fault
            self.record_events = record_events
            self.dynamic_count = 0

        def record(self, event: tuple[object, ...]) -> None:
            if self.record_events:
                events.append(event)

        def load_validation_perturbations(
            self, *, expected_vocabulary: VocabularyIdentity
        ) -> object:
            self.record(("fixture", expected_vocabulary))
            return object()

        def build_arm(
            self,
            *,
            arm_config: ArmConfig,
            structure: H6LanguageStructure,
        ) -> object:
            self.record(("build", arm_config, structure))
            return (arm_config, structure)

        def build_predictor_boundary(
            self,
            *,
            built_arm: object,
            estimator: EstimatorSpec,
        ) -> object:
            self.record(("boundary", built_arm, estimator.particle_count))
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
            self.record(
                (
                    "dynamic",
                    job.particle_count,
                    job.case_family,
                    job.selected_global_indices,
                )
            )
            execution_plan = _bounded_execution_plan(job)
            report_key = None
            if self.fault == "dynamic_binding" and self.dynamic_count == 0:
                report_key = plan.static_audit_job.report_keys[-1]
            self.dynamic_count += 1
            return SimpleNamespace(
                execution_plan=execution_plan,
                report=_bounded_dynamic_report(
                    job=job,
                    execution_plan=execution_plan,
                    key=report_key,
                ),
                observed_predictor_call_count=(
                    job.expected_predictor_call_count
                ),
            )

        def execute_static_audit(self, *, job: object) -> object:
            self.record(("static", job.report_keys))
            keys = (
                job.report_keys[:-1]
                if self.fault == "static_manifest"
                else job.report_keys
            )
            return _static_report(keys)

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
    with pytest.raises(RuntimeError, match="dynamic job result"):
        h6_prefix_gate._execute_h6_prefix_plan(
            plan,
            FakeExecutor(
                fault="dynamic_binding",
                record_events=False,
            ),
        )
    with pytest.raises(RuntimeError, match="static audit report"):
        h6_prefix_gate._execute_h6_prefix_plan(
            plan,
            FakeExecutor(
                fault="static_manifest",
                record_events=False,
            ),
        )
    with pytest.raises(RuntimeError, match="language structure"):
        h6_prefix_gate._require_live_structure_binding(
            plan.semantic_families[0].profiles[0].small_structure,
            "f" * 64,
        )


def test_v3_private_executor_loads_bound_inputs_before_arms_and_keeps_v2_workload_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _bounded_v3_resolved_config(tmp_path)
    plan = h6_prefix_gate._build_h6_prefix_runner_plan(config)
    events: list[str] = []

    def read_fixture(reference: object) -> object:
        events.append("fixture")
        assert reference == config.validation_fixture_reference
        return _v3_fixture_payload(config)

    def read_candidate(path: Path) -> object:
        events.append("candidate")
        assert path == (
            config.validation_perturbation_reference.local_artifact_path
        )
        return (
            h6_validation_candidate.H6ValidationPerturbationArtifactPayload(
                reference=config.validation_perturbation_reference,
                candidate_bytes=_V3_CANDIDATE_BYTES,
            )
        )

    def production_load(
        source: bytes,
        *,
        expected_vocabulary: VocabularyIdentity,
        validation_fixture: object,
        validation_fixture_bytes: bytes,
    ) -> object:
        events.append("production-load")
        assert source == _V3_CANDIDATE_BYTES
        assert expected_vocabulary == plan.expected_validation_vocabulary
        assert validation_fixture.fixture_sha256 == (
            config.validation_fixture_reference.fixture_raw_sha256
        )
        assert validation_fixture_bytes == _V3_FIXTURE_BYTES
        return _v3_full_perturbations(config, expected_vocabulary)

    monkeypatch.setattr(
        h6_prefix_gate,
        "read_validation_safety_fixture_payload",
        read_fixture,
    )
    monkeypatch.setattr(
        h6_prefix_gate,
        "load_h6_validation_perturbation_artifact_payload",
        read_candidate,
    )
    monkeypatch.setattr(
        h6_prefix_gate,
        "load_frozen_validation_perturbations",
        production_load,
    )
    executor = _V3FakeExecutor(
        events=events,
        loader=lambda expected: (
            h6_prefix_gate._load_h6_prefix_v3_validation_inputs(
                config,
                expected_vocabulary=expected,
            )
        ),
    )

    evidence = h6_prefix_gate._execute_h6_prefix_v3_plan(plan, executor)

    assert events[:4] == [
        "fixture",
        "candidate",
        "production-load",
        "build-arm",
    ]
    assert events.count("build-arm") == 2
    assert config.authorization_sha256 == H6_PREFIX_V3_AUTHORIZATION_SHA256
    assert (
        config.workload_authorization_sha256
        == H6_PREFIX_V2_AUTHORIZATION_SHA256
    )
    assert all(
        result.execution_plan.authorization_sha256
        == H6_PREFIX_V2_AUTHORIZATION_SHA256
        for result in evidence.dynamic_results
    )


def test_v3_input_mismatch_blocks_arms_and_public_v2_v3_publication_remains_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _bounded_v3_resolved_config(tmp_path)
    plan = h6_prefix_gate._build_h6_prefix_runner_plan(config)

    for failure in (
        "fixture-bytes",
        "fixture-starts",
        "fixture-counts",
        "candidate",
        "incomplete",
    ):
        events: list[str] = []
        fixture_payload = _v3_fixture_payload(config)
        if failure == "fixture-bytes":
            object.__setattr__(
                fixture_payload,
                "fixture_bytes",
                fixture_payload.fixture_bytes[:-1],
            )
        elif failure == "fixture-starts":
            object.__setattr__(
                fixture_payload,
                "starts",
                tuple(value + 1 for value in fixture_payload.starts),
            )
        elif failure == "fixture-counts":
            object.__setattr__(
                fixture_payload,
                "real_target_counts",
                (2, *fixture_payload.real_target_counts[1:]),
            )
        candidate_reference = config.validation_perturbation_reference
        if failure == "candidate":
            candidate_reference = replace(
                candidate_reference,
                local_artifact_path=(
                    tmp_path / "other-v3-candidate"
                ).resolve(),
            )
        candidate_payload = (
            h6_validation_candidate.H6ValidationPerturbationArtifactPayload(
                reference=candidate_reference,
                candidate_bytes=_V3_CANDIDATE_BYTES,
            )
        )

        monkeypatch.setattr(
            h6_prefix_gate,
            "read_validation_safety_fixture_payload",
            lambda reference, payload=fixture_payload: payload,
        )
        monkeypatch.setattr(
            h6_prefix_gate,
            "load_h6_validation_perturbation_artifact_payload",
            lambda path, payload=candidate_payload: payload,
        )
        monkeypatch.setattr(
            h6_prefix_gate,
            "load_frozen_validation_perturbations",
            lambda source, *, expected_vocabulary, validation_fixture,
            validation_fixture_bytes: _v3_full_perturbations(
                config,
                expected_vocabulary,
                materialized_count=(2 if failure == "incomplete" else 4096),
            ),
        )
        executor = _V3FakeExecutor(
            events=events,
            loader=lambda expected: (
                h6_prefix_gate._load_h6_prefix_v3_validation_inputs(
                    config,
                    expected_vocabulary=expected,
                )
            ),
        )
        with pytest.raises((ValueError, RuntimeError)):
            h6_prefix_gate._execute_h6_prefix_v3_plan(plan, executor)
        assert "build-arm" not in events

    monkeypatch.setattr(
        h6_prefix_gate,
        "current_source_identity",
        lambda repo_root, artifact_root: (
            config.source.git_head,
            config.source.dirty_digest,
            config.source.source_sha256,
        ),
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("public blocked path reached input or arm work")

    monkeypatch.setattr(
        h6_prefix_gate,
        "read_validation_safety_fixture_payload",
        forbidden,
    )
    monkeypatch.setattr(
        h6_prefix_gate,
        "load_h6_validation_perturbation_artifact_payload",
        forbidden,
    )
    monkeypatch.setattr(
        h6_prefix_gate,
        "load_frozen_validation_perturbations",
        forbidden,
    )
    monkeypatch.setattr(h6_prefix_gate, "build_arm", forbidden)
    with pytest.raises(RuntimeError, match="h6-prefix-config-v2.*blocked"):
        h6_prefix_gate.run_h6_prefix(
            config=_bounded_resolved_config(tmp_path),
            junit_sha256=None,
        )
    with pytest.raises(RuntimeError, match="Task 3E2C3|publication"):
        h6_prefix_gate.run_h6_prefix(
            config=config,
            junit_sha256=None,
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
