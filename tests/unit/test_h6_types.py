from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest
import torch

from vfe4.types import GateStatus, H6PrefixGateResult, H6PredictionResult
from vfe4.types.h6 import (
    H6_PREFIX_REQUIRED_CHECKS,
    ArmId,
    CausalDag,
    CausalDagRow,
    CheckpointIdentity,
    DataIdentity,
    DurableTestOpeningCapability,
    EmissionOnlyAblationTerms,
    EncodedTokenStorageIdentity,
    EndpointSmcProtocol,
    EvidenceStatus,
    FrozenTensorSnapshot,
    FrozenBatchSchedule,
    H5UpdateBinding,
    H6ArmPhaseSchedule,
    H6LanguageStructure,
    H6FactorTerm,
    H6LanguageElboTerms,
    H6OuterSchedule,
    H6PredictionReadinessToken,
    H6TrainingSchedule,
    H1PrefixPriorArtifactRef,
    PredictionCorrectnessArtifactRef,
    PredictionDecision,
    PrefixCaseKey,
    PrefixCertificate,
    SmcAccuracyArtifactRef,
    TrainingPhase,
    ValidationSafetyFixture,
    VocabularyIdentity,
    ZeroDimensionalBase,
    canonical_json_bytes,
    issue_prediction_readiness,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
GIT_HEAD = "1" * 40


def _dag() -> CausalDag:
    return CausalDag.create(
        node_labels=(0, 1, 2, 3, 4),
        rows=(
            CausalDagRow(1, (0,)),
            CausalDagRow(2, (0, 1)),
            CausalDagRow(3, (0, 1, 2)),
            CausalDagRow(4, (0, 1, 2, 3)),
        ),
    )


def _prefix_key() -> PrefixCaseKey:
    return PrefixCaseKey(
        arm=ArmId.A5,
        predictor_config_sha256=SHA_A,
        estimator_sha256=SHA_B,
        model_family_sha256=SHA_C,
        vocabulary_sha256=SHA_D,
        data_safety_sha256="e" * 64,
        git_head=GIT_HEAD,
        dirty_digest="f" * 64,
    )


def _manifest_bytes(entries: dict[str, bytes]) -> bytes:
    return "".join(
        f"{hashlib.sha256(entries[path]).hexdigest()}  {path}\n"
        for path in sorted(entries)
    ).encode("ascii")


def _producer_payload(
    *,
    gate: str,
    config_bytes: bytes,
    dirty_digest: str = SHA_A,
    status: GateStatus = GateStatus.PASS,
    obligations: tuple[str, ...] = (),
    extra: dict[str, object] | None = None,
) -> bytes:
    return canonical_json_bytes(
        {
            "gate": gate,
            "status": status.value,
            "obligations": obligations,
            "git_head": GIT_HEAD,
            "dirty_digest": dirty_digest,
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            **({} if extra is None else extra),
        }
    )


def _h5_binding() -> H5UpdateBinding:
    return H5UpdateBinding.from_producer_preimages(
        producer_preimages={
            "h5_manifest_sha256": b"manifest",
            "h5_payload_sha256": b"payload",
            "update_spec_raw_sha256": b"raw-update",
            "update_spec_canonical_sha256": b"canonical-update",
            "objective_schema_sha256": b"objective",
            "factor_input_schema_sha256": b"factor-input",
            "reference_sha256": b"reference",
            "recognition_state_sha256": b"recognition",
            "model_state_sha256": b"model",
            "validation_payload_sha256": b"validation",
        },
        enabled_update_labels=(
            "exact_coordinate",
            "generalized_em",
            "natural_gradient_proposal",
        ),
    )


def _data_identity() -> DataIdentity:
    encoded = (
        (256).to_bytes(2, "little")
        + (7).to_bytes(2, "little")
        + (257).to_bytes(2, "little")
    )
    storage = EncodedTokenStorageIdentity.create(
        token_count=3, encoded_token_bytes=encoded
    )
    fixture = ValidationSafetyFixture.create(
        validation_token_sha256=storage.encoded_token_sha256,
        starts=tuple(range(4096)),
        real_target_counts=(32,) * 4096,
        fixture_bytes=b"fixture",
    )
    return DataIdentity.create(
        archive_sha256=SHA_A,
        train_raw_sha256=SHA_B,
        validation_raw_sha256=SHA_C,
        test_raw_sha256=SHA_D,
        train_tokens=storage,
        validation_tokens=storage,
        test_tokens=storage,
        validation_fixture=fixture,
        access_policy_sha256="e" * 64,
    )


def _endpoint_protocol() -> EndpointSmcProtocol:
    return EndpointSmcProtocol.create(
        particle_counts=(128, 256, 512, 1024),
        replicate_count=64,
        registry_root_seed=2026072198,
        common_stream_domain="h6-wt2-endpoint-mc-v1",
        simultaneous_interval_count=352,
        familywise_alpha=0.01,
        critical_value_df63=4.5144904535377144,
        remainder_contraction=0.75,
    )


def _correctness_ref(gate: str) -> PredictionCorrectnessArtifactRef:
    config = f"config-{gate}".encode()
    path = Path(f"validation/{gate.lower()}.json")
    payload = _producer_payload(gate=gate, config_bytes=config)
    return PredictionCorrectnessArtifactRef.from_bytes(
        gate=gate,  # type: ignore[arg-type]
        artifact_path=path,
        manifest_bytes=_manifest_bytes(
            {"config.json": config, path.as_posix(): payload}
        ),
        git_head=GIT_HEAD,
        dirty_digest=SHA_A,
        config_bytes=config,
        validation_payload_bytes=payload,
    )


def _h1_prefix_ref() -> H1PrefixPriorArtifactRef:
    config = b"h1-prefix-config"
    schema = b"generative-schema"
    path = Path("validation/h1_prefix_prior.json")
    payload = _producer_payload(
        gate="H1-Prefix-Prior",
        config_bytes=config,
        extra={
            "generative_factor_schema_sha256": hashlib.sha256(schema).hexdigest()
        },
    )
    return H1PrefixPriorArtifactRef.from_bytes(
        artifact_path=path,
        manifest_bytes=_manifest_bytes(
            {
                "config.json": config,
                "schemas/generative_factor.json": schema,
                path.as_posix(): payload,
            }
        ),
        git_head=GIT_HEAD,
        dirty_digest=SHA_A,
        generative_factor_schema_bytes=schema,
        config_bytes=config,
        validation_payload_bytes=payload,
    )


def _smc_ref() -> SmcAccuracyArtifactRef:
    config = b"smc-config"
    estimator = b"estimator"
    fixture_set = b"fixture-set"
    path = Path("validation/h6_smc_accuracy.json")
    payload = _producer_payload(
        gate="H6-SMC-Accuracy",
        config_bytes=config,
        extra={
            "estimator_sha256": hashlib.sha256(estimator).hexdigest(),
            "fixture_set_sha256": hashlib.sha256(fixture_set).hexdigest(),
        },
    )
    return SmcAccuracyArtifactRef.from_bytes(
        artifact_path=path,
        manifest_bytes=_manifest_bytes(
            {
                "config.json": config,
                "protocol/estimator.json": estimator,
                "fixtures/finite_smc.json": fixture_set,
                path.as_posix(): payload,
            }
        ),
        git_head=GIT_HEAD,
        dirty_digest=SHA_A,
        estimator_preimage_bytes=estimator,
        fixture_set_bytes=fixture_set,
        config_bytes=config,
        validation_payload_bytes=payload,
    )


def test_zero_dimensional_base_dag_and_structure_bind_all_semantics() -> None:
    base = ZeroDimensionalBase.create()
    dag = _dag()
    structure = H6LanguageStructure.create(
        base=base,
        dag=dag,
        receiver_labels=(1, 2, 3, 4),
    )

    assert base.points == ("*",) and base.dimension == 0
    assert dag.labeling == "zero_based"
    assert structure.receiver_labels == tuple(row.receiver_t for row in dag.rows)
    assert len({base.canonical_sha256, dag.canonical_sha256, structure.structure_sha256}) == 3

    with pytest.raises(ValueError, match="canonical_sha256"):
        dataclasses.replace(base, canonical_sha256="0" * 64)
    with pytest.raises(ValueError, match="canonical_sha256"):
        dataclasses.replace(dag, node_labels=(0, 1, 2, 3, 5))
    with pytest.raises(ValueError, match="structure_sha256"):
        dataclasses.replace(structure, receiver_labels=(1, 2, 4, 3))


@pytest.mark.parametrize(
    "rows",
    [
        (CausalDagRow(1, (0,)), CausalDagRow(3, (0, 1))),
        (CausalDagRow(1, (0,)), CausalDagRow(1, (0,))),
    ],
)
def test_causal_dag_rejects_missing_or_duplicate_receivers(
    rows: tuple[CausalDagRow, ...],
) -> None:
    with pytest.raises(ValueError, match="receiver"):
        CausalDag.create(node_labels=(0, 1, 2), rows=rows)


@pytest.mark.parametrize(
    "row",
    [
        (1, (1,)),
        (1, (2,)),
        (2, (0, 0)),
        (2, (1, 0)),
        (0, ()),
    ],
)
def test_causal_dag_row_rejects_noncausal_or_ambiguous_parent_sets(
    row: tuple[int, tuple[int, ...]],
) -> None:
    with pytest.raises(ValueError):
        CausalDagRow(*row)


def test_frozen_tensor_snapshot_owns_bytes_and_preserves_autograd() -> None:
    source = torch.tensor([1.0, -0.0], dtype=torch.float64, requires_grad=True)
    snapshot = FrozenTensorSnapshot.capture(source)
    original_hash = snapshot.raw_bytes_sha256

    with torch.no_grad():
        source.add_(10.0)
    assert snapshot.value().tolist() == [1.0, -0.0]
    returned = snapshot.value()
    with torch.no_grad():
        returned.mul_(0.0)
    assert snapshot.raw_bytes_sha256 == original_hash
    assert snapshot.value().tolist() == [1.0, -0.0]

    snapshot.value().sum().backward()
    assert source.grad is not None
    assert source.grad.tolist() == [1.0, 1.0]

    private = getattr(snapshot, "_FrozenTensorSnapshot__owned")
    with torch.no_grad():
        private.add_(1.0)
    with pytest.raises(ValueError, match="integrity"):
        snapshot.assert_intact()
    with pytest.raises(ValueError, match="integrity"):
        snapshot.value()


def test_vocabulary_identity_verifies_named_tokenizer_bytes_independently() -> None:
    spec = b'{"tokenizer":"byte-v1"}'
    identity = VocabularyIdentity.from_tokenizer_spec(
        vocabulary_id="h6-prefix-small-v1",
        size=3,
        tokenizer_spec_bytes=spec,
    )
    identity.verify_tokenizer_spec(spec)
    with pytest.raises(ValueError, match="tokenizer_spec_sha256"):
        identity.verify_tokenizer_spec(spec + b"!")


def test_prefix_certificate_is_data_safety_bound_and_fail_closed() -> None:
    checks = {name: True for name in H6_PREFIX_REQUIRED_CHECKS}
    certificate = PrefixCertificate.create(
        key=_prefix_key(),
        status=EvidenceStatus.PASS,
        checks=checks,
        obligations=(),
    )

    assert certificate.status is EvidenceStatus.PASS
    assert certificate.key.data_safety_sha256 == "e" * 64
    assert certificate.validation_payload_sha256 == hashlib.sha256(
        certificate.validation_payload_canonical_json
    ).hexdigest()

    with pytest.raises(ValueError, match="certificate_sha256"):
        dataclasses.replace(
            certificate,
            key=dataclasses.replace(certificate.key, data_safety_sha256="0" * 64),
        )
    with pytest.raises(ValueError, match="validation_payload_sha256"):
        dataclasses.replace(
            certificate,
            validation_payload_canonical_json=(
                certificate.validation_payload_canonical_json + b" "
            ),
        )
    with pytest.raises(ValueError, match="PASS"):
        PrefixCertificate.create(
            key=_prefix_key(),
            status=EvidenceStatus.PASS,
            checks={**checks, H6_PREFIX_REQUIRED_CHECKS[0]: False},
            obligations=(),
        )
    with pytest.raises(ValueError, match="obligation"):
        PrefixCertificate.create(
            key=_prefix_key(),
            status=EvidenceStatus.PASS,
            checks=checks,
            obligations=("unresolved",),
        )
    with pytest.raises(ValueError, match="FAIL"):
        PrefixCertificate.create(
            key=_prefix_key(),
            status=EvidenceStatus.FAIL,
            checks=checks,
            obligations=(),
        )


def test_prefix_certificate_payload_obligations_must_match_record() -> None:
    checks = {name: True for name in H6_PREFIX_REQUIRED_CHECKS}
    certificate = PrefixCertificate.create(
        key=_prefix_key(),
        status=EvidenceStatus.PASS,
        checks=checks,
        obligations=(),
    )
    payload = json.loads(certificate.validation_payload_canonical_json)
    payload["obligations"] = ["forged"]
    payload_bytes = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    payload_sha = hashlib.sha256(payload_bytes).hexdigest()
    forged_sha = hashlib.sha256(
        b"vfe4.h6.prefix-certificate.v1\x00"
        + json.dumps(
            {
                "key": certificate.key.canonical_payload(),
                "obligations": [],
                "status": "PASS",
                "validation_payload_sha256": payload_sha,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(ValueError, match="obligations"):
        PrefixCertificate(
            certificate.key,
            payload_bytes,
            payload_sha,
            EvidenceStatus.PASS,
            (),
            forged_sha,
        )


def test_public_arm_inventory_and_phase_schedules_are_exact() -> None:
    assert tuple(ArmId) == tuple(ArmId(f"A{i}") for i in range(6))
    outer = H6OuterSchedule.create(optimizer_policy_sha256=SHA_A)
    no_latent = H6ArmPhaseSchedule.create(
        endpoint_config_sha256=SHA_B,
        latent_enabled=False,
        phases=(TrainingPhase.MODEL_CE_ADAMW,),
    )
    latent = H6ArmPhaseSchedule.create(
        endpoint_config_sha256=SHA_C,
        latent_enabled=True,
        phases=(
            TrainingPhase.RECOGNITION_ADAMW,
            TrainingPhase.IMMUTABLE_DETACHED_SNAPSHOT,
            TrainingPhase.MODEL_ADAMW,
        ),
    )
    schedule = H6TrainingSchedule.create(
        outer=outer,
        endpoint_phases=(no_latent, latent),
    )

    assert schedule.schedule_schema == "h6-training-schedule-v2"
    assert no_latent.recognition_updates_per_batch == 0
    assert latent.recognition_updates_per_batch == 1
    with pytest.raises(ValueError, match="phase"):
        H6ArmPhaseSchedule.create(
            endpoint_config_sha256=SHA_D,
            latent_enabled=True,
            phases=(TrainingPhase.MODEL_ADAMW, TrainingPhase.RECOGNITION_ADAMW),
        )


def test_h5_binding_verifies_each_producer_preimage() -> None:
    preimages = {
        "h5_manifest_sha256": b"manifest",
        "h5_payload_sha256": b"payload",
        "update_spec_raw_sha256": b"raw-update",
        "update_spec_canonical_sha256": b"canonical-update",
        "objective_schema_sha256": b"objective",
        "factor_input_schema_sha256": b"factor-input",
        "reference_sha256": b"reference",
        "recognition_state_sha256": b"recognition",
        "model_state_sha256": b"model",
        "validation_payload_sha256": b"validation",
    }
    binding = H5UpdateBinding.from_producer_preimages(
        producer_preimages=preimages,
        enabled_update_labels=(
            "exact_coordinate",
            "generalized_em",
            "natural_gradient_proposal",
        ),
    )
    binding.verify_producer_preimages(preimages)
    with pytest.raises(ValueError, match="h5_payload_sha256"):
        binding.verify_producer_preimages({**preimages, "h5_payload_sha256": b"changed"})
    with pytest.raises(ValueError, match="binding_sha256"):
        dataclasses.replace(binding, binding_sha256="0" * 64)


def test_prediction_artifact_reference_excludes_h4_and_verifies_bytes() -> None:
    config = b"config"
    artifact_path = Path("validation/h3.json")
    payload = _producer_payload(gate="H3", config_bytes=config)
    manifest = _manifest_bytes(
        {"config.json": config, artifact_path.as_posix(): payload}
    )
    reference = PredictionCorrectnessArtifactRef.from_bytes(
        gate="H3",
        artifact_path=artifact_path,
        manifest_bytes=manifest,
        git_head=GIT_HEAD,
        dirty_digest=SHA_A,
        config_bytes=config,
        validation_payload_bytes=payload,
    )
    assert reference.status is GateStatus.PASS
    reference.verify_bytes(
        manifest_bytes=manifest,
        config_bytes=config,
        validation_payload_bytes=payload,
    )
    with pytest.raises(ValueError, match="manifest_sha256"):
        reference.verify_bytes(
            manifest_bytes=b"changed",
            config_bytes=config,
            validation_payload_bytes=payload,
        )
    with pytest.raises(TypeError):
        PredictionCorrectnessArtifactRef(
            "H3", artifact_path, SHA_A, GIT_HEAD,
            SHA_A, SHA_B, SHA_C, GateStatus.PASS,
        )
    with pytest.raises(ValueError, match="gate"):
        PredictionCorrectnessArtifactRef.from_bytes(
            gate="H4",  # type: ignore[arg-type]
            artifact_path=Path("validation/h4.json"),
            manifest_bytes=manifest,
            git_head=GIT_HEAD,
            dirty_digest=SHA_A,
            config_bytes=config,
            validation_payload_bytes=payload,
        )
    forged_payload = _producer_payload(
        gate="H3", config_bytes=config, obligations=("still open",)
    )
    forged_manifest = _manifest_bytes(
        {"config.json": config, artifact_path.as_posix(): forged_payload}
    )
    with pytest.raises(ValueError, match="PASS.*obligation"):
        PredictionCorrectnessArtifactRef.from_bytes(
            gate="H3",
            artifact_path=artifact_path,
            manifest_bytes=forged_manifest,
            git_head=GIT_HEAD,
            dirty_digest=SHA_A,
            config_bytes=config,
            validation_payload_bytes=forged_payload,
        )


def test_language_elbo_and_ablation_recompute_owned_digests() -> None:
    partitions = (
        "initial", "state_source", "model_source", "state_transition",
        "model_transition", "emission", "entropy",
    )
    terms = tuple(
        H6FactorTerm(
            0 if partition == "initial" else 1,
            partition,  # type: ignore[arg-type]
            hashlib.sha256(partition.encode()).hexdigest(),
            FrozenTensorSnapshot.capture(torch.tensor(1.0, dtype=torch.float64)),
        )
        for partition in partitions
    )
    independent_total = torch.tensor(7.0, dtype=torch.float64)
    elbo = H6LanguageElboTerms.create(
        horizon=1,
        ordered_factor_terms=terms,
        total_language_elbo=independent_total,
    )
    assert elbo.complete_decomposition.value().item() == 7.0
    assert all(
        getattr(elbo, f"{partition}_terms")
        for partition in partitions
    )
    with pytest.raises(TypeError):
        EmissionOnlyAblationTerms.create(
            ordered_emission_terms=elbo.emission_terms  # type: ignore[call-arg]
        )
    with pytest.raises(ValueError, match="seven partitions"):
        H6LanguageElboTerms.create(
            horizon=1,
            ordered_factor_terms=(terms[5],),
            total_language_elbo=torch.tensor(1.0, dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="does not equal"):
        H6LanguageElboTerms.create(
            horizon=1,
            ordered_factor_terms=terms,
            total_language_elbo=torch.tensor(8.0, dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="canonical_sha256"):
        dataclasses.replace(elbo, canonical_sha256="0" * 64)


def test_h1_prefix_and_smc_references_require_named_producer_bytes() -> None:
    h1_config = b"h1-prefix-config"
    schema = b"generative-schema"
    h1_path = Path("validation/h1_prefix_prior.json")
    h1_payload = _producer_payload(
        gate="H1-Prefix-Prior",
        config_bytes=h1_config,
        extra={
            "generative_factor_schema_sha256": hashlib.sha256(schema).hexdigest()
        },
    )
    h1_manifest = _manifest_bytes(
        {
            "config.json": h1_config,
            "schemas/generative_factor.json": schema,
            h1_path.as_posix(): h1_payload,
        }
    )
    h1 = H1PrefixPriorArtifactRef.from_bytes(
        artifact_path=h1_path,
        manifest_bytes=h1_manifest,
        git_head=GIT_HEAD,
        dirty_digest=SHA_A,
        generative_factor_schema_bytes=schema,
        config_bytes=h1_config,
        validation_payload_bytes=h1_payload,
    )
    estimator = b"estimator"
    fixture_set = b"fixture-set"
    smc_config = b"smc-config"
    smc_path = Path("validation/h6_smc_accuracy.json")
    smc_payload = _producer_payload(
        gate="H6-SMC-Accuracy",
        config_bytes=smc_config,
        extra={
            "estimator_sha256": hashlib.sha256(estimator).hexdigest(),
            "fixture_set_sha256": hashlib.sha256(fixture_set).hexdigest(),
        },
    )
    smc_manifest = _manifest_bytes(
        {
            "config.json": smc_config,
            "protocol/estimator.json": estimator,
            "fixtures/finite_smc.json": fixture_set,
            smc_path.as_posix(): smc_payload,
        }
    )
    smc = SmcAccuracyArtifactRef.from_bytes(
        artifact_path=smc_path,
        manifest_bytes=smc_manifest,
        git_head=GIT_HEAD,
        dirty_digest=SHA_A,
        estimator_preimage_bytes=estimator,
        fixture_set_bytes=fixture_set,
        config_bytes=smc_config,
        validation_payload_bytes=smc_payload,
    )
    h1.verify_bytes(
        manifest_bytes=h1_manifest,
        generative_factor_schema_bytes=schema,
        config_bytes=h1_config,
        validation_payload_bytes=h1_payload,
    )
    smc.verify_bytes(
        manifest_bytes=smc_manifest,
        estimator_preimage_bytes=estimator,
        fixture_set_bytes=fixture_set,
        config_bytes=smc_config,
        validation_payload_bytes=smc_payload,
    )
    with pytest.raises(ValueError, match="fixture_set_sha256"):
        smc.verify_bytes(
            manifest_bytes=smc_manifest,
            estimator_preimage_bytes=estimator,
            fixture_set_bytes=b"changed",
            config_bytes=smc_config,
            validation_payload_bytes=smc_payload,
        )
    with pytest.raises(TypeError):
        H1PrefixPriorArtifactRef(
            Path("x"), SHA_A, GIT_HEAD, SHA_A, SHA_B, SHA_C, SHA_D,
            GateStatus.PASS,
        )
    with pytest.raises(TypeError):
        SmcAccuracyArtifactRef(
            Path("x"), SHA_A, GIT_HEAD, SHA_A, SHA_B, SHA_C, SHA_D,
            GateStatus.PASS,
        )


def test_data_checkpoint_schedule_and_capability_identity_surface_is_frozen() -> None:
    storage = EncodedTokenStorageIdentity.create(
        token_count=3,
        encoded_token_bytes=(256).to_bytes(2, "little")
        + (7).to_bytes(2, "little")
        + (257).to_bytes(2, "little"),
    )
    fixture = ValidationSafetyFixture.create(
        validation_token_sha256=storage.encoded_token_sha256,
        starts=tuple(range(4096)),
        real_target_counts=(32,) * 4096,
        fixture_bytes=b"fixture",
    )
    schedule = FrozenBatchSchedule.create(
        zero_based_pass_index=0,
        window_count=4,
        permutation=(2, 0, 3, 1),
    )
    data = DataIdentity.create(
        archive_sha256=SHA_A,
        train_raw_sha256=SHA_B,
        validation_raw_sha256=SHA_C,
        test_raw_sha256=SHA_D,
        train_tokens=storage,
        validation_tokens=storage,
        test_tokens=storage,
        validation_fixture=fixture,
        access_policy_sha256="e" * 64,
    )
    checkpoint = CheckpointIdentity.create(
        experiment_identity_sha256=SHA_A,
        config_sha256=SHA_B,
        model_state_sha256=SHA_C,
        data_identity_sha256=data.data_identity_sha256,
        estimator_sha256=SHA_D,
        batch_schedule_sha256=schedule.schedule_sha256,
    )
    assert checkpoint.data_identity_sha256 == data.data_identity_sha256
    with pytest.raises(TypeError):
        DurableTestOpeningCapability(
            SHA_A, SHA_B, SHA_C, SHA_D, "e" * 64, "f" * 64, b"proof"
        )


def test_prediction_decision_is_derived_from_frozen_interval_rule() -> None:
    passed = PredictionDecision.classify(
        primary_interval=(0.011, 0.02), estimator_complete=True
    )
    failed = PredictionDecision.classify(
        primary_interval=(-0.02, 0.0), estimator_complete=True
    )
    unresolved = PredictionDecision.classify(
        primary_interval=(0.0, 0.01), estimator_complete=True
    )
    assert passed.status is EvidenceStatus.PASS
    assert failed.status is EvidenceStatus.FAIL
    assert unresolved.status is EvidenceStatus.INCONCLUSIVE
    with pytest.raises(TypeError):
        PredictionDecision(EvidenceStatus.PASS, (0.1, 0.2), ())


def _readiness_token() -> H6PredictionReadinessToken:
    correctness = tuple(_correctness_ref(gate) for gate in ("H1", "H2", "H3", "H5"))
    checks = {name: True for name in H6_PREFIX_REQUIRED_CHECKS}
    certificate = PrefixCertificate.create(
        key=dataclasses.replace(_prefix_key(), dirty_digest=SHA_A),
        status=EvidenceStatus.PASS,
        checks=checks,
        obligations=(),
    )
    outer = H6OuterSchedule.create(optimizer_policy_sha256=SHA_A)
    phase = H6ArmPhaseSchedule.create(
        endpoint_config_sha256=SHA_B,
        latent_enabled=False,
        phases=(TrainingPhase.MODEL_CE_ADAMW,),
    )
    schedule = H6TrainingSchedule.create(outer=outer, endpoint_phases=(phase,))
    h1_prefix = _h1_prefix_ref()
    smc = _smc_ref()
    token = issue_prediction_readiness(
        git_head=GIT_HEAD,
        dirty_digest=SHA_A,
        experiment_config_sha256=SHA_B,
        correctness_artifacts=correctness,
        h1_prefix_prior_artifact=h1_prefix,
        h5_update_binding=_h5_binding(),
        h6_training_schedule=schedule,
        smc_accuracy_artifact=smc,
        critical_values_sha256="0" * 64,
        endpoint_smc_protocol=_endpoint_protocol(),
        attribution_matrix_sha256="2" * 64,
        matching_set_sha256="3" * 64,
        prefix_certificates={certificate.key: certificate},
        data_identity=_data_identity(),
    )
    assert token.status == "PASS"
    with pytest.raises(TypeError):
        H6PredictionReadinessToken(
            "h6-prediction-readiness-v1", GIT_HEAD, SHA_A, SHA_B, (), None,
            SHA_D, schedule.schedule_sha256, SHA_A, SHA_A, SHA_A, SHA_A,
            SHA_A, SHA_A, SHA_A, SHA_A, SHA_A, "PASS",
        )
    return token


def test_readiness_token_requires_validated_pass_artifacts() -> None:
    assert _readiness_token().status == "PASS"


def test_h6_results_are_explicit_and_do_not_widen_legacy_gate_result() -> None:
    certificate = PrefixCertificate.create(
        key=_prefix_key(),
        status=EvidenceStatus.PASS,
        checks={name: True for name in H6_PREFIX_REQUIRED_CHECKS},
        obligations=(),
    )
    prefix = H6PrefixGateResult.from_certificates(
        {_prefix_key(): certificate}
    )
    metrics = json.dumps(
        {
            "schema": "h6-prediction-metrics-v1",
            "estimator_complete": True,
            "primary_interval": {"lower": 0.0, "upper": 0.01},
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    prediction = H6PredictionResult.from_metrics(
        readiness=_readiness_token(), metrics_bytes=metrics
    )
    assert prefix.gate == "H6-Prefix"
    assert prediction.status is GateStatus.INCONCLUSIVE
    assert prediction.smc_bias_semantics_sha256 is None
    assert not hasattr(H6PredictionResult, "from_decision")
    with pytest.raises(TypeError):
        H6PrefixGateResult("H6-Prefix", GateStatus.PASS, SHA_A, SHA_B, ())
    with pytest.raises(TypeError):
        H6PredictionResult(
            "H6-Prediction", GateStatus.PASS, SHA_A, SHA_B, ()
        )
