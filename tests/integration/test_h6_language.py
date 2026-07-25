from __future__ import annotations

import math

import pytest
import torch

from vfe4.data.windows import CausalPrefix, build_causal_windows
from vfe4.evaluation.prior_nll import score_prior_nll_replicate
from vfe4.predictive import (
    BootstrapSmcPredictor,
    EstimatorIdentity,
    EstimatorStream,
    PriorPrediction,
    vocabulary_identity_sha256,
)
from vfe4.recognition import RecognitionConditioning
from vfe4.training.arms import (
    ArmTargetFreeProposalAdapter,
    MeanPooledPrefixFloor,
    build_a5,
)
from vfe4.training.language import (
    ArmObjectiveInventory,
    ArmTrainingObjectiveAdapter,
    DetachedRecognitionLawSnapshot,
    plan_h6_attempt,
)
from vfe4.training.matching import (
    ArmConfig,
    CapacityAllocation,
    MatchingReport,
)
from vfe4.types import (
    ArmId,
    EstimatorSpec,
    EvidenceStatus,
    H6_PREFIX_REQUIRED_CHECKS,
    H6ArmPhaseSchedule,
    PrefixCaseKey,
    PrefixCertificate,
    TrainingPhase,
    VocabularyIdentity,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
EXPECTED_PARTITIONS = {
    ArmId.A0: ("emission",),
    ArmId.A1: (
        "initial",
        "state_transition",
        "emission",
        "entropy",
    ),
    ArmId.A2: (
        "initial",
        "state_source",
        "model_source",
        "state_transition",
        "model_transition",
        "emission",
        "entropy",
    ),
    ArmId.A3: (
        "initial",
        "state_transition",
        "model_transition",
        "emission",
        "entropy",
    ),
    ArmId.A4: (
        "initial",
        "state_source",
        "state_transition",
        "emission",
        "entropy",
    ),
    ArmId.A5: (
        "initial",
        "state_source",
        "model_source",
        "state_transition",
        "model_transition",
        "emission",
        "entropy",
    ),
}
_FIELDS = (
    "config_id",
    "latent_enabled",
    "state_channel_enabled",
    "model_channel_enabled",
    "source_mode",
    "map_mode",
    "recognition_family",
    "recognition_conditioning",
    "prior_variant",
    "mixture_mode",
    "objective_kind",
)
_SEMANTICS = {
    ArmId.A0: (
        "h6-a0-transformer-v2",
        False,
        False,
        False,
        "absent",
        "absent",
        "absent",
        "absent",
        "absent",
        "absent",
        "cross_entropy",
    ),
    ArmId.A1: (
        "h6-a1-ordinary-latent-v1",
        True,
        True,
        False,
        "absent",
        "absent",
        "structured",
        "smoothing",
        "absent",
        "absent",
        "complete_elbo",
    ),
    ArmId.A2: (
        "h6-a2-generic-map-v1",
        True,
        True,
        True,
        "categorical",
        "generic_fixed_frame_non_coboundary",
        "structured",
        "smoothing",
        "fixed",
        "exact",
        "complete_elbo",
    ),
    ArmId.A3: (
        "h6-a3-immediate-predecessor-v1",
        True,
        True,
        True,
        "immediate_predecessor",
        "shared_vertex_coboundary",
        "structured",
        "smoothing",
        "absent",
        "absent",
        "complete_elbo",
    ),
    ArmId.A4: (
        "h6-a4-state-only-v1",
        True,
        True,
        False,
        "categorical",
        "shared_vertex_coboundary",
        "structured",
        "smoothing",
        "fixed",
        "exact",
        "complete_elbo",
    ),
    ArmId.A5: (
        "h6-a5-structured-fixed-exact-complete-latent-smoothing-v1",
        True,
        True,
        True,
        "categorical",
        "shared_vertex_coboundary",
        "structured",
        "smoothing",
        "fixed",
        "exact",
        "complete_elbo",
    ),
}


def _arm_config(arm: ArmId) -> ArmConfig:
    values = dict(zip(_FIELDS, _SEMANTICS[arm], strict=True))
    latent_enabled = values["latent_enabled"] is True
    recognition_enabled = values["recognition_family"] != "absent"
    return ArmConfig.create(
        arm=arm,
        vocabulary=VocabularyIdentity("h6-task8-small-v1", 3, SHA_A),
        horizon=2,
        capacity_allocation=CapacityAllocation.create(
            emission_width=4,
            latent_width=2 if latent_enabled else None,
            recognition_width=4 if recognition_enabled else None,
        ),
        **values,
    )


def _inconclusive_matching_report(
    *, endpoint_config_sha256: str
) -> MatchingReport:
    return MatchingReport.from_totals(
        matching_config_sha256=SHA_A,
        endpoint_config_sha256=endpoint_config_sha256,
        reference_config_sha256=SHA_C,
        endpoint_parameter_count=100,
        reference_parameter_count=100,
        endpoint_training_flops=100,
        reference_training_flops=100,
        parameter_relative_tolerance=0.01,
        flop_relative_tolerance=0.05,
        ownership_valid=True,
        common_schedule=True,
        optimizer_policy_match=True,
        training_flop_ledger_complete=False,
        training_flop_obligations=("whole-schedule ledger is deferred",),
        semantic_interventions=("whole_declared_architecture",),
        named_factor="whole_declared_architecture",
        nuisance_capacity_fields=(),
        common_schedule_sha256=SHA_D,
    )


def test_literal_families_have_only_their_live_objective_partitions() -> None:
    for arm, expected in EXPECTED_PARTITIONS.items():
        inventory = ArmObjectiveInventory.for_config(config=_arm_config(arm))
        assert inventory.partitions == expected


def test_detached_snapshot_is_clone_only_and_inconclusive_plan_refuses() -> None:
    config = _arm_config(ArmId.A5)
    built = build_a5(config)
    parameter_store = built.recognition_store
    assert parameter_store is not None
    conditioning = RecognitionConditioning.create(
        mode="smoothing",
        horizon=2,
        observed_tokens=torch.tensor([0, 1], dtype=torch.int64),
    )
    snapshot = DetachedRecognitionLawSnapshot.capture_from_store(
        config=config,
        parameter_store=parameter_store,
        conditioning=conditioning,
    )
    expected_mean = snapshot.tensor("mean")
    with torch.no_grad():
        next(parameter_store.parameters()).add_(10.0)
    returned = snapshot.tensor("mean")
    returned.add_(20.0)

    assert snapshot.tensor("mean").requires_grad is False
    assert snapshot.tensor("mean").grad_fn is None
    assert torch.equal(snapshot.tensor("mean"), expected_mean)

    phases = (
        TrainingPhase.RECOGNITION_ADAMW,
        TrainingPhase.IMMUTABLE_DETACHED_SNAPSHOT,
        TrainingPhase.MODEL_ADAMW,
    )
    phase_schedule = H6ArmPhaseSchedule.create(
        endpoint_config_sha256=config.config_sha256,
        latent_enabled=True,
        phases=phases,
    )
    inventory = ArmObjectiveInventory.for_config(config=config)
    adapter = ArmTrainingObjectiveAdapter.create(
        config=config,
        inventory=inventory,
        phase_schedule=phase_schedule,
    )
    assert adapter.phases == phases

    report = _inconclusive_matching_report(
        endpoint_config_sha256=config.config_sha256
    )
    assert report.status == "INCONCLUSIVE"
    with pytest.raises(ValueError, match="INCONCLUSIVE|FLOP"):
        plan_h6_attempt(
            config=config,
            objective_adapter=adapter,
            phase_schedule=phase_schedule,
            matching_report=report,
        )


class _RecordingPredictor:
    def __init__(self, delegate: BootstrapSmcPredictor) -> None:
        self._delegate = delegate
        self.prefixes: list[tuple[int, ...]] = []
        for name in (
            "proposal",
            "estimator_spec",
            "estimator_identity",
            "predictor_config_sha256",
            "data_safety_sha256",
            "vocabulary",
            "vocabulary_sha256",
            "model_family_sha256",
            "model_state_sha256",
            "proposal_identity_sha256",
            "particle_count",
        ):
            setattr(self, name, getattr(delegate, name))

    def next_token_log_probs(
        self,
        prefix_tokens: CausalPrefix,
        estimator_rng: EstimatorStream,
        cache=None,
    ) -> PriorPrediction:
        self.prefixes.append(tuple(prefix_tokens.token_ids.tolist()))
        return self._delegate.next_token_log_probs(
            prefix_tokens,
            estimator_rng,
            cache,
        )


def test_tiny_prior_scorer_is_certificate_bound_and_target_blind() -> None:
    vocabulary = VocabularyIdentity("h6-v3-test", 3, SHA_A)
    model = MeanPooledPrefixFloor(
        vocabulary=vocabulary,
        emission_width=4,
    )
    proposal = ArmTargetFreeProposalAdapter(
        model=model,
        model_family_sha256=SHA_C,
    )
    estimator_spec = EstimatorSpec.create(
        kind="weighted_smc",
        particle_count=128,
        resampling="systematic_ess_half",
    )
    estimator_identity = EstimatorIdentity.from_spec(estimator_spec)
    delegate = BootstrapSmcPredictor(
        proposal=proposal,
        estimator_spec=estimator_spec,
        estimator_identity=estimator_identity,
        predictor_config_sha256=SHA_B,
        data_safety_sha256=SHA_D,
    )
    predictor = _RecordingPredictor(delegate)
    stream = EstimatorStream.create(
        stream_seed=2026072301,
        estimator_identity=estimator_identity,
    )
    certificate = PrefixCertificate.create(
        key=PrefixCaseKey(
            arm=ArmId.A5,
            predictor_config_sha256=SHA_B,
            estimator_sha256=estimator_spec.estimator_sha256,
            model_family_sha256=SHA_C,
            vocabulary_sha256=vocabulary_identity_sha256(vocabulary),
            data_safety_sha256=SHA_D,
            git_head="1" * 40,
            dirty_digest=SHA_A,
        ),
        status=EvidenceStatus.PASS,
        checks={name: True for name in H6_PREFIX_REQUIRED_CHECKS},
        obligations=(),
    )
    windows = build_causal_windows((0, 1, 2), split="test")
    result = score_prior_nll_replicate(
        predictor,
        windows,
        stream,
        128,
        certificate,
    )

    empty = CausalPrefix.create(
        receiver_t=1,
        vocabulary=vocabulary,
        token_ids=torch.empty(0, dtype=torch.int64),
    )
    prior_one = CausalPrefix.create(
        receiver_t=2,
        vocabulary=vocabulary,
        token_ids=torch.tensor([1], dtype=torch.int64),
    )
    expected_sum = -math.fsum(
        (
            float(model.prefix_log_probs(empty)[1].item()),
            float(model.prefix_log_probs(prior_one)[2].item()),
        )
    )
    assert predictor.prefixes == [(), (1,)]
    assert result.negative_log_likelihood_sum == pytest.approx(expected_sum)
    assert result.counted_targets == 2
    assert result.nats_per_token == pytest.approx(expected_sum / 2)
