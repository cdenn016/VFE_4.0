from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import pytest
import torch

from vfe4.types import GateStatus, H6PrefixGateResult, H6PredictionResult
from vfe4.types.h6 import (
    H6_PREFIX_REQUIRED_CHECKS,
    ArmId,
    CausalDag,
    CausalDagRow,
    EvidenceStatus,
    FrozenTensorSnapshot,
    H5UpdateBinding,
    H6ArmPhaseSchedule,
    H6LanguageStructure,
    H6OuterSchedule,
    H6PredictionReadinessToken,
    H6TrainingSchedule,
    PredictionCorrectnessArtifactRef,
    PrefixCaseKey,
    PrefixCertificate,
    TrainingPhase,
    VocabularyIdentity,
    ZeroDimensionalBase,
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
    reference = PredictionCorrectnessArtifactRef.from_bytes(
        gate="H3",
        artifact_path=Path("runs/h3/validation.json"),
        manifest_bytes=b"manifest",
        git_head=GIT_HEAD,
        dirty_digest=SHA_A,
        config_sha256=SHA_B,
        validation_payload_bytes=b"payload",
        status=GateStatus.PASS,
    )
    reference.verify_bytes(manifest_bytes=b"manifest", validation_payload_bytes=b"payload")
    with pytest.raises(ValueError, match="manifest_sha256"):
        reference.verify_bytes(manifest_bytes=b"changed", validation_payload_bytes=b"payload")
    with pytest.raises(ValueError, match="gate"):
        dataclasses.replace(reference, gate="H4")


def test_readiness_token_requires_exact_h1_h2_h3_h5_inventory() -> None:
    token = H6PredictionReadinessToken.create(
        git_head=GIT_HEAD,
        dirty_digest=SHA_A,
        experiment_config_sha256=SHA_B,
        correctness_manifests=(
            ("H1", "1" * 64),
            ("H2", "2" * 64),
            ("H3", "3" * 64),
            ("H5", "5" * 64),
        ),
        h1_prefix_prior_manifest_sha256=SHA_C,
        h5_update_binding_sha256=SHA_D,
        h6_training_schedule_sha256="e" * 64,
        smc_validation_manifest_sha256="f" * 64,
        critical_values_sha256="0" * 64,
        endpoint_smc_protocol_sha256="1" * 64,
        attribution_matrix_sha256="2" * 64,
        matching_set_sha256="3" * 64,
        prefix_certificate_set_sha256="4" * 64,
        data_identity_sha256="5" * 64,
        access_policy_sha256="6" * 64,
    )
    assert token.status == "PASS"
    with pytest.raises(ValueError, match="H1.*H2.*H3.*H5"):
        H6PredictionReadinessToken.create(
            **{
                field.name: getattr(token, field.name)
                for field in dataclasses.fields(token)
                if field.name not in {"readiness_sha256", "status", "correctness_manifests"}
            },
            correctness_manifests=(
                ("H1", "1" * 64),
                ("H2", "2" * 64),
                ("H3", "3" * 64),
                ("H4", "4" * 64),
                ("H5", "5" * 64),
            ),
        )


def test_h6_results_are_explicit_and_do_not_widen_legacy_gate_result() -> None:
    prefix = H6PrefixGateResult(
        gate="H6-Prefix",
        status=GateStatus.PASS,
        validation_payload_sha256=SHA_A,
        prefix_certificate_set_sha256=SHA_B,
        obligations=(),
    )
    prediction = H6PredictionResult(
        gate="H6-Prediction",
        status=GateStatus.INCONCLUSIVE,
        readiness_sha256=SHA_C,
        metrics_sha256=None,
        obligations=("deferred evidence operation",),
    )
    assert prefix.gate == "H6-Prefix"
    assert prediction.status is GateStatus.INCONCLUSIVE

