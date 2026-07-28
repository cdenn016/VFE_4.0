from __future__ import annotations

import dataclasses
import hashlib
import math
from functools import cache

import pytest
import torch

import test_h6_validation_v3 as validation_fixtures
import vfe4.training.h6_heldout_scoring_v3 as heldout
from vfe4.artifacts.h6_prediction_v3 import (
    H6CheckpointCandidateV3,
    select_h6_tuning_v3,
)
from vfe4.data.byte_tokenizer import ByteTokenizerV1
from vfe4.data.windows import CausalPrefix, build_causal_windows
from vfe4.predictive import EstimatorIdentity
from vfe4.training.arms import (
    CapacityAllocation,
    build_a5,
    build_arm,
    build_arm_model,
)
from vfe4.training.h6_experiment_v3 import (
    H6_TUNED_ENDPOINT_CONFIG_IDS_V3,
    H6_TUNING_CELLS_V3,
    H6ExperimentPlanV3,
    H6TuningCellV3,
    plan_h6_experiment_v3,
)
from vfe4.training.h6_heldout_scoring_v3 import (
    H6HeldoutCheckpointArmV3,
    h6_weighted_estimator_stream_seed_v3,
    score_h6_exact_a0_total_v3,
    score_h6_heldout_inventory_v3,
    score_h6_weighted_a5_total_v3,
)
from vfe4.training.h6_matching_v3 import (
    H6_MATCHING_V3_ENDPOINT_CONFIG_IDS,
    H6MatchingSetV3,
    H6TrainingWorkloadV3,
)
from vfe4.training.h6_transformer import H6CausalTransformer
from vfe4.training.h6_validation_v3 import build_h6_evaluation_arm_v3
from vfe4.types import (
    H6_A0_DIRECT_EXACT_PREFIX_REQUIRED_CHECKS,
    H6_A0_DIRECT_EXACT_PREFIX_WITNESS_CHECKS,
    A0DirectExactPrefixCertificateV1,
    A0DirectExactPrefixWitnessV1,
    ArmConfig,
    ArmId,
    BoundedPrefixCertificate,
    BoundedPrefixCertificateSet,
    BoundedPrefixReportBinding,
    BoundedPrefixReportReference,
    EstimatorSpec,
    EvidenceStatus,
    H6PredictionV3ReadinessToken,
    H6PrefixWorkloadPlan,
    H6_PREFIX_REQUIRED_CHECKS,
    NllTotals,
    PrefixCaseKey,
    VocabularyIdentity,
)
from vfe4.types.h6 import canonical_json_bytes


_SHA = "a" * 64


def _tiny_a0() -> tuple[
    ArmConfig,
    H6CausalTransformer,
    A0DirectExactPrefixCertificateV1,
]:
    config = ArmConfig.create(
        arm=ArmId.A0,
        config_id=H6_MATCHING_V3_ENDPOINT_CONFIG_IDS[0],
        vocabulary=VocabularyIdentity("h6-heldout-a0-test-v1", 3, _SHA),
        horizon=4,
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
            emission_width=4,
            latent_width=None,
            recognition_width=None,
        ),
    )
    arm = build_arm(config.arm, config)
    model = arm.model
    assert type(model) is H6CausalTransformer
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    bounded = _bounded_certificate(
        arm,
        git_head="1" * 40,
        dirty_digest="2" * 64,
        semantic_label="tiny-a0",
    )
    return config, model, _direct_a0_certificate(
        config=config,
        bounded=bounded,
        model_family_sha256=arm.model_family_sha256,
    )


def test_exact_a0_scores_only_real_target_blind_prefixes() -> None:
    config, model, certificate = _tiny_a0()
    windows = build_causal_windows((0, 1, 2, 1, 0), split="test")
    observed_prefixes: list[tuple[int, ...]] = []
    original = model.prefix_log_probs

    def recording_prefix_log_probs(prefix: CausalPrefix) -> torch.Tensor:
        observed_prefixes.append(tuple(prefix.token_ids.tolist()))
        return original(prefix)

    model.prefix_log_probs = recording_prefix_log_probs  # type: ignore[method-assign]
    totals = score_h6_exact_a0_total_v3(
        config=config,
        model=model,
        windows=windows,
        certificate=certificate,
    )

    expected_terms = []
    history: list[int] = []
    for target in (1, 2, 1, 0):
        prefix = CausalPrefix.create(
            receiver_t=len(history) + 1,
            vocabulary=model.vocabulary,
            token_ids=torch.tensor(history, dtype=torch.int64),
        )
        expected_terms.append(-float(original(prefix)[target].item()))
        history.append(target)

    assert observed_prefixes == [(), (1,), (1, 2), (1, 2, 1)]
    assert totals.counted_targets == 4
    assert totals.negative_log_likelihood_sum == pytest.approx(
        math.fsum(expected_terms)
    )


def _tiny_complete_a5():
    config = ArmConfig.create(
        arm=ArmId.A5,
        config_id=(
            "h6-a5-structured-fixed-exact-complete-latent-smoothing-v1"
        ),
        vocabulary=VocabularyIdentity("h6-heldout-a5-test-v1", 3, _SHA),
        horizon=2,
        latent_enabled=True,
        state_channel_enabled=True,
        model_channel_enabled=True,
        source_mode="categorical",
        map_mode="shared_vertex_coboundary",
        recognition_family="structured",
        recognition_conditioning="smoothing",
        prior_variant="fixed",
        mixture_mode="exact",
        objective_kind="complete_elbo",
        capacity_allocation=CapacityAllocation.create(
            emission_width=48,
            latent_width=8,
            recognition_width=32,
            prior_context_width=None,
        ),
    )
    arm = build_a5(config)
    arm.model.eval()
    for parameter in arm.model.parameters():
        parameter.requires_grad_(False)
    return arm


def _source_sha256(git_head: str, dirty_digest: str) -> str:
    return hashlib.sha256(
        b"VFE4-H6-SOURCE-CANDIDATE-V1\x00"
        + bytes.fromhex(git_head)
        + bytes.fromhex(dirty_digest)
    ).hexdigest()


def _bounded_certificate(
    arm,
    *,
    git_head: str,
    dirty_digest: str,
    semantic_label: str,
    status: EvidenceStatus = EvidenceStatus.PASS,
    obligations: tuple[str, ...] = (),
) -> BoundedPrefixCertificate:
    workload_sha256 = H6PrefixWorkloadPlan().workload_plan_sha256
    semantic_sha256 = _digest(f"semantic:{semantic_label}")
    profile_pair_sha256s = tuple(
        _digest(f"profile:{semantic_label}:{particle_count}")
        for particle_count in (128, 256, 512, 1024)
    )
    references = []
    for profile_pair_sha256, particle_count in zip(
        profile_pair_sha256s,
        (128, 256, 512, 1024),
        strict=True,
    ):
        spec = EstimatorSpec.create(
            kind="weighted_smc",
            particle_count=particle_count,
            resampling="systematic_ess_half",
        )
        _, predictor = arm.rebuild_predictive_boundary(spec)
        assert predictor.estimator_identity == EstimatorIdentity.from_spec(spec)
        key = PrefixCaseKey(
            arm=arm.config.arm,
            predictor_config_sha256=predictor.predictor_config_sha256,
            estimator_sha256=spec.estimator_sha256,
            model_family_sha256=predictor.model_family_sha256,
            vocabulary_sha256=predictor.vocabulary_sha256,
            data_safety_sha256=predictor.data_safety_sha256,
            git_head=git_head,
            dirty_digest=dirty_digest,
        )
        for case_family in ("small", "validation"):
            references.append(
                BoundedPrefixReportReference.create(
                    profile_pair_sha256=profile_pair_sha256,
                    particle_count=particle_count,
                    case_family=case_family,
                    scope=(
                        "representative_exhaustive"
                        if particle_count == 128
                        else "estimator_stratified"
                    ),
                    report_key=key,
                    report_sha256=_digest(
                        f"report:{semantic_label}:{particle_count}:{case_family}"
                    ),
                    execution_plan_sha256=_digest(
                        f"execution:{semantic_label}:{particle_count}:{case_family}"
                    ),
                    workload_plan_sha256=workload_sha256,
                    selected_global_indices=(0,),
                    selection_manifest_sha256=_digest(
                        f"selection:{semantic_label}:{case_family}"
                    ),
                    completed_by_position=(1,),
                    complete_case_manifest_sha256=(
                        _digest(
                            f"complete:{semantic_label}:{case_family}"
                        )
                        if particle_count == 128
                        else None
                    ),
                    model_state_sha256=predictor.model_state_sha256,
                    proposal_identity_sha256=(
                        predictor.proposal_identity_sha256
                    ),
                    estimator_semantic_sha256=(
                        predictor.estimator_identity.semantic_sha256
                    ),
                    estimator_artifact_bytes_sha256=(
                        predictor.estimator_identity.artifact_bytes_sha256
                    ),
                )
            )
    binding = BoundedPrefixReportBinding.create(
        workload_plan_sha256=workload_sha256,
        semantic_family_sha256=semantic_sha256,
        git_head=git_head,
        dirty_digest=dirty_digest,
        source_sha256=_source_sha256(git_head, dirty_digest),
        global_case_key_order_sha256=_digest(
            f"case-order:{semantic_label}"
        ),
        profile_pair_sha256s=profile_pair_sha256s,
        report_references=tuple(references),
        higher_n_small_selection_manifest_sha256=_digest(
            f"selection:{semantic_label}:small"
        ),
        higher_n_validation_selection_manifest_sha256=_digest(
            f"selection:{semantic_label}:validation"
        ),
        static_report_sha256=_digest(f"static-report:{semantic_label}"),
        static_source_manifest_sha256=_digest(
            f"static-source:{semantic_label}"
        ),
        static_rules_sha256=_digest(f"static-rules:{semantic_label}"),
        static_case_key_manifest_sha256=_digest(
            f"static-cases:{semantic_label}"
        ),
    )
    return BoundedPrefixCertificate.create(
        semantic_family_sha256=semantic_sha256,
        report_binding=binding,
        status=status,
        checks={name: True for name in H6_PREFIX_REQUIRED_CHECKS},
        obligations=obligations,
    )


def _direct_a0_certificate(
    *,
    config: ArmConfig,
    bounded: BoundedPrefixCertificate,
    model_family_sha256: str,
) -> A0DirectExactPrefixCertificateV1:
    references = bounded.report_binding.report_references
    small_reference = next(
        reference
        for reference in references
        if (
            reference.particle_count == 128
            and reference.case_family == "small"
        )
    )
    validation_reference = next(
        reference
        for reference in references
        if (
            reference.particle_count == 128
            and reference.case_family == "validation"
        )
    )
    direct_predictor_path_sha256 = _digest("direct-a0-predictor-path")
    witness = A0DirectExactPrefixWitnessV1.create(
        small_complete_case_manifest_sha256=(
            small_reference.complete_case_manifest_sha256
        ),
        validation_complete_case_manifest_sha256=(
            validation_reference.complete_case_manifest_sha256
        ),
        small_model_state_sha256=small_reference.model_state_sha256,
        production_model_state_sha256=(
            validation_reference.model_state_sha256
        ),
        small_proposal_identity_sha256=(
            small_reference.proposal_identity_sha256
        ),
        production_proposal_identity_sha256=(
            validation_reference.proposal_identity_sha256
        ),
        direct_predictor_path_sha256=direct_predictor_path_sha256,
        case_witness_manifest_sha256=_digest("direct-a0-case-witness"),
        checks={
            name: True
            for name in H6_A0_DIRECT_EXACT_PREFIX_WITNESS_CHECKS
        },
        first_counterexample_sha256=None,
    )
    return A0DirectExactPrefixCertificateV1.create(
        endpoint_config=config,
        estimator=EstimatorSpec.create(
            kind="deterministic_exact",
            particle_count=None,
            resampling="none",
        ),
        model_family_sha256=model_family_sha256,
        data_safety_sha256=(
            small_reference.report_key.data_safety_sha256
        ),
        git_head=bounded.report_binding.git_head,
        dirty_digest=bounded.report_binding.dirty_digest,
        source_sha256=bounded.report_binding.source_sha256,
        direct_predictor_path_sha256=direct_predictor_path_sha256,
        heldout_scorer_path_sha256=_digest("heldout-scorer-path"),
        bounded_a0_certificate=bounded,
        direct_witness=witness,
        static_report_sha256=(
            bounded.report_binding.static_report_sha256
        ),
        static_report_status=EvidenceStatus.PASS,
        checks={
            name: True
            for name in H6_A0_DIRECT_EXACT_PREFIX_REQUIRED_CHECKS
        },
        status=EvidenceStatus.PASS,
        obligations=(),
    )


def _direct_certificate_with_status(
    certificate: A0DirectExactPrefixCertificateV1,
    status: EvidenceStatus,
) -> A0DirectExactPrefixCertificateV1:
    checks = dict(certificate.checks)
    if status is EvidenceStatus.FAIL:
        checks["target_read_after_prediction"] = False
        static_status = EvidenceStatus.PASS
        obligations: tuple[str, ...] = ()
    elif status is EvidenceStatus.INCONCLUSIVE:
        checks["static_report"] = False
        static_status = EvidenceStatus.INCONCLUSIVE
        obligations = ("static: synthetic unresolved authority",)
    else:
        raise ValueError("synthetic status must be FAIL or INCONCLUSIVE")
    return A0DirectExactPrefixCertificateV1.create(
        endpoint_config=certificate.endpoint_config,
        estimator=certificate.estimator,
        model_family_sha256=certificate.model_family_sha256,
        data_safety_sha256=certificate.data_safety_sha256,
        git_head=certificate.git_head,
        dirty_digest=certificate.dirty_digest,
        source_sha256=certificate.source_sha256,
        direct_predictor_path_sha256=(
            certificate.direct_predictor_path_sha256
        ),
        heldout_scorer_path_sha256=(
            certificate.heldout_scorer_path_sha256
        ),
        bounded_a0_certificate=certificate.bounded_a0_certificate,
        direct_witness=certificate.direct_witness,
        static_report_sha256=certificate.static_report_sha256,
        static_report_status=static_status,
        checks=checks,
        status=status,
        obligations=obligations,
    )


def test_weighted_a5_uses_frozen_replicate_stream_and_real_partial_horizon() -> None:
    arm = _tiny_complete_a5()
    windows = build_causal_windows((0, 1, 2), split="test")
    totals = score_h6_weighted_a5_total_v3(
        config=arm.config,
        model=arm.model,
        windows=windows,
        particle_count=128,
        replicate_id=0,
        certificate=_bounded_certificate(
            arm,
            git_head="1" * 40,
            dirty_digest="2" * 64,
            semantic_label="tiny-complete",
        ),
    )

    assert totals.counted_targets == 2
    assert math.isfinite(totals.negative_log_likelihood_sum)
    assert totals.negative_log_likelihood_sum >= 0.0
    with pytest.raises(ValueError, match="replicate"):
        score_h6_weighted_a5_total_v3(
            config=arm.config,
            model=arm.model,
            windows=windows,
            particle_count=128,
            replicate_id=64,
            certificate=_bounded_certificate(
                arm,
                git_head="1" * 40,
                dirty_digest="2" * 64,
                semantic_label="tiny-complete",
            ),
        )


def test_weighted_stream_seed_mapping_is_canonical_and_frozen() -> None:
    assert tuple(
        h6_weighted_estimator_stream_seed_v3(replicate_id=replicate_id)
        for replicate_id in (0, 1, 63)
    ) == (
        5_551_064_237_968_966_580,
        2_515_697_631_527_028_451,
        10_593_027_223_154_148_081,
    )
    with pytest.raises(ValueError, match="replicate"):
        h6_weighted_estimator_stream_seed_v3(replicate_id=64)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _matching_set() -> H6MatchingSetV3:
    return H6MatchingSetV3.create(
        git_head=validation_fixtures._GIT_HEAD,
        dirty_digest=validation_fixtures._DIRTY_DIGEST,
        workload=H6TrainingWorkloadV3.from_train_tokens(
            train_token_count=258,
            train_token_sha256=_digest("train-tokens"),
        ),
        endpoint_templates=tuple(
            validation_fixtures._template(
                config_id,
                ByteTokenizerV1().vocabulary_identity,
            )
            for config_id in H6_MATCHING_V3_ENDPOINT_CONFIG_IDS
        ),
    )


def _heldout_certificate_set_for_configs(
    configs: tuple[ArmConfig, ...],
    *,
    git_head: str,
    dirty_digest: str,
) -> tuple[
    BoundedPrefixCertificateSet,
    A0DirectExactPrefixCertificateV1,
]:
    by_id = {config.config_id: config for config in configs}
    certificates = []
    a0_config = by_id[H6_MATCHING_V3_ENDPOINT_CONFIG_IDS[0]]
    a0_arm = build_arm(a0_config.arm, a0_config)
    a0_bounded = _bounded_certificate(
        a0_arm,
        git_head=git_head,
        dirty_digest=dirty_digest,
        semantic_label=H6_MATCHING_V3_ENDPOINT_CONFIG_IDS[0],
    )
    certificates.append(a0_bounded)
    for endpoint_id in (
        H6_MATCHING_V3_ENDPOINT_CONFIG_IDS[5],
        H6_MATCHING_V3_ENDPOINT_CONFIG_IDS[9],
    ):
        config = by_id[endpoint_id]
        arm = build_a5(config)
        certificates.append(
            _bounded_certificate(
                arm,
                git_head=git_head,
                dirty_digest=dirty_digest,
                semantic_label=endpoint_id,
            )
        )
    frozen = tuple(certificates)
    certificate_set = BoundedPrefixCertificateSet.create(
        config_sha256=_digest("bounded-prefix-config"),
        semantic_family_sha256s=tuple(
            certificate.semantic_family_sha256
            for certificate in frozen
        ),
        certificates=frozen,
    )
    return certificate_set, _direct_a0_certificate(
        config=a0_config,
        bounded=a0_bounded,
        model_family_sha256=a0_arm.model_family_sha256,
    )


def _readiness_for_certificate_set(
    base: H6PredictionV3ReadinessToken,
    certificate_set: BoundedPrefixCertificateSet,
    direct_certificate: A0DirectExactPrefixCertificateV1,
) -> H6PredictionV3ReadinessToken:
    return H6PredictionV3ReadinessToken.create(
        git_head=base.git_head,
        dirty_digest=base.dirty_digest,
        experiment_config_sha256=base.experiment_config_sha256,
        correctness_manifests=base.correctness_manifests,
        h1_prefix_prior_manifest_sha256=(
            base.h1_prefix_prior_manifest_sha256
        ),
        h1_prefix_prior_generative_factor_schema_sha256=(
            base.h1_prefix_prior_generative_factor_schema_sha256
        ),
        smc_bias_semantics_sha256=base.smc_bias_semantics_sha256,
        smc_validation_manifest_sha256=(
            base.smc_validation_manifest_sha256
        ),
        prefix_certificate_set_sha256=(
            certificate_set.prefix_certificate_set_sha256
        ),
        a0_direct_exact_prefix_certificate_sha256=(
            direct_certificate.certificate_sha256
        ),
        h5_update_binding_sha256=base.h5_update_binding_sha256,
        critical_values_sha256=base.critical_values_sha256,
        endpoint_smc_protocol_sha256=base.endpoint_smc_protocol_sha256,
        attribution_matrix_sha256=base.attribution_matrix_sha256,
        objective_gate_spec_sha256=base.objective_gate_spec_sha256,
        matching_policy_sha256=base.matching_policy_sha256,
        matching_set_sha256=base.matching_set_sha256,
        training_schedule_sha256=base.training_schedule_sha256,
        recognition_estimator_sha256=base.recognition_estimator_sha256,
        runtime_identity_sha256=base.runtime_identity_sha256,
        counter_mapping_sha256=base.counter_mapping_sha256,
        phase_ownership_sha256=base.phase_ownership_sha256,
        objective_manifest_schema_sha256=(
            base.objective_manifest_schema_sha256
        ),
        data_identity_sha256=base.data_identity_sha256,
        access_policy_sha256=base.access_policy_sha256,
    )


def _tuning_selection_for_plan(plan: H6ExperimentPlanV3):
    endpoint_indices = {
        endpoint_id: index
        for index, endpoint_id in enumerate(
            H6_TUNED_ENDPOINT_CONFIG_IDS_V3
        )
    }
    cell_indices = {
        H6TuningCellV3.create(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        ).cell_sha256: index
        for index, (learning_rate, weight_decay) in enumerate(
            H6_TUNING_CELLS_V3
        )
    }
    records = tuple(
        validation_fixtures._validation_record(
            plan=plan,
            attempt=attempt,
            mean_prior_nll=(
                1.0
                if endpoint_indices[attempt.endpoint_config_id] == 0
                else float(
                    cell_indices[attempt.tuning_cell.cell_sha256]
                    != endpoint_indices[attempt.endpoint_config_id]
                )
            ),
        )
        for attempt in plan.tuning_attempts
        if attempt.tuning_cell is not None
    )
    return select_h6_tuning_v3(records, plan)


@cache
def _heldout_authority_fixture() -> tuple[
    tuple[H6HeldoutCheckpointArmV3, ...],
    BoundedPrefixCertificateSet,
    A0DirectExactPrefixCertificateV1,
    H6PredictionV3ReadinessToken,
]:
    matching = _matching_set()
    base_plan, runtime, base_readiness = validation_fixtures._authorities()
    assert matching.matching_set_sha256 == base_plan.matching_set_sha256
    certificate_set, direct_certificate = (
        _heldout_certificate_set_for_configs(
            matching.endpoint_configs,
            git_head=base_readiness.git_head,
            dirty_digest=base_readiness.dirty_digest,
        )
    )
    readiness = _readiness_for_certificate_set(
        base_readiness,
        certificate_set,
        direct_certificate,
    )
    plan = plan_h6_experiment_v3(
        readiness=readiness,
        matching_set=matching,
        training_schedule=base_plan.training_schedule,
        runtime_identity=runtime,
    )
    tuning_selection = _tuning_selection_for_plan(plan)
    selected_cells = {
        item.endpoint_config_id: item.tuning_cell
        for item in tuning_selection.endpoint_selections
    }
    configs = {
        config.config_id: config for config in plan.endpoint_configs
    }
    heldout_ids = {
        H6_MATCHING_V3_ENDPOINT_CONFIG_IDS[0],
        H6_MATCHING_V3_ENDPOINT_CONFIG_IDS[5],
        H6_MATCHING_V3_ENDPOINT_CONFIG_IDS[9],
    }
    bindings = []
    for attempt in plan.confirmatory_attempts:
        if attempt.endpoint_config_id not in heldout_ids:
            continue
        model = build_arm_model(configs[attempt.endpoint_config_id])
        checkpoint = validation_fixtures._terminal_checkpoint(
            attempt,
            runtime=runtime,
            cell=selected_cells[attempt.endpoint_config_id],
            model=model,
        )
        candidate = H6CheckpointCandidateV3.create(
            checkpoint=checkpoint,
            planned_attempt=attempt,
            plan=plan,
            tuning_selection=tuning_selection,
        )
        evaluation = build_h6_evaluation_arm_v3(
            checkpoint,
            plan=plan,
            planned_attempt=attempt,
            evaluation_role="heldout",
        )
        bindings.append(
            H6HeldoutCheckpointArmV3(
                candidate=candidate,
                evaluation=evaluation,
            )
        )
    return (
        tuple(bindings),
        certificate_set,
        direct_certificate,
        readiness,
    )


@pytest.mark.parametrize(
    "authority_case",
    ("missing", "fail", "inconclusive", "readiness_sha_drift"),
)
def test_direct_a0_authority_rejected_before_test_window_access(
    authority_case: str,
) -> None:
    (
        bindings,
        certificate_set,
        direct_certificate,
        readiness,
    ) = _heldout_authority_fixture()

    class PoisonTestWindows:
        @property
        def split(self) -> str:
            raise AssertionError(
                "test-window bytes were inspected before direct-A0 closure"
            )

    candidate: object = direct_certificate
    candidate_readiness = readiness
    if authority_case == "missing":
        candidate = None
    elif authority_case == "fail":
        candidate = _direct_certificate_with_status(
            direct_certificate,
            EvidenceStatus.FAIL,
        )
    elif authority_case == "inconclusive":
        candidate = _direct_certificate_with_status(
            direct_certificate,
            EvidenceStatus.INCONCLUSIVE,
        )
    else:
        candidate_readiness = _readiness_for_certificate_set(
            validation_fixtures._authorities()[2],
            certificate_set,
            direct_certificate,
        )
        forged = object.__new__(H6PredictionV3ReadinessToken)
        for field in dataclasses.fields(candidate_readiness):
            object.__setattr__(
                forged,
                field.name,
                (
                    _digest("another-direct-certificate")
                    if field.name
                    == "a0_direct_exact_prefix_certificate_sha256"
                    else getattr(candidate_readiness, field.name)
                ),
            )
        values = forged.canonical_payload()
        object.__setattr__(
            forged,
            "readiness_sha256",
            hashlib.sha256(
                b"vfe4.h6.prediction-readiness.v3\x00"
                + canonical_json_bytes(values)
            ).hexdigest(),
        )
        candidate_readiness = forged

    with pytest.raises(ValueError, match="direct-A0|PASS|readiness"):
        score_h6_heldout_inventory_v3(
            windows=PoisonTestWindows(),  # type: ignore[arg-type]
            opening_proof_sha256=_digest("opening"),
            checkpoint_arms=bindings,
            prefix_certificate_set=certificate_set,
            a0_direct_exact_prefix_certificate=candidate,  # type: ignore[arg-type]
            readiness=candidate_readiness,
        )


def test_inventory_scores_only_a0_and_two_a5_endpoints_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        bindings,
        certificate_set,
        direct_certificate,
        readiness,
    ) = _heldout_authority_fixture()
    windows = build_causal_windows((0, 1, 2), split="test")
    exact_calls: list[str] = []
    weighted_calls: list[tuple[str, int, int]] = []

    def exact_total(*, model, windows, **_) -> NllTotals:
        exact_calls.append(model.family_label)
        return NllTotals(2.0, windows.counted_target_total)

    def weighted_total(
        *,
        config,
        model,
        windows,
        particle_count,
        replicate_id,
        certificate,
    ) -> NllTotals:
        del model, certificate
        weighted_calls.append(
            (config.config_id, replicate_id, particle_count)
        )
        value = (
            1.0
            + replicate_id * 1.0e-8
            + 1.0 / float(particle_count)
        )
        return NllTotals(
            value * windows.counted_target_total,
            windows.counted_target_total,
        )

    monkeypatch.setattr(heldout, "score_h6_exact_a0_total_v3", exact_total)
    monkeypatch.setattr(
        heldout,
        "score_h6_weighted_a5_total_v3",
        weighted_total,
    )
    inventory = score_h6_heldout_inventory_v3(
        windows=windows,
        opening_proof_sha256=_digest("opening"),
        checkpoint_arms=bindings,
        prefix_certificate_set=certificate_set,
        a0_direct_exact_prefix_certificate=direct_certificate,
        readiness=readiness,
    )

    assert inventory.logical_row_count == 4104
    assert len(inventory.exact_a0_rows) == len(exact_calls) == 8
    assert len(inventory.complete_a5_rows) == 2048
    assert len(inventory.emission_a5_rows) == 2048
    assert len(weighted_calls) == 4096
    assert {row.endpoint_config_id for row in inventory.exact_a0_rows} == {
        H6_MATCHING_V3_ENDPOINT_CONFIG_IDS[0]
    }
    assert {
        row.endpoint_config_id for row in inventory.complete_a5_rows
    } == {H6_MATCHING_V3_ENDPOINT_CONFIG_IDS[5]}
    assert {
        row.endpoint_config_id for row in inventory.emission_a5_rows
    } == {H6_MATCHING_V3_ENDPOINT_CONFIG_IDS[9]}
    assert not hasattr(inventory.exact_a0_rows[0], "particle_count")
    assert not hasattr(inventory.exact_a0_rows[0], "replicate_id")
    assert not hasattr(
        inventory.exact_a0_rows[0],
        "monte_carlo_half_width",
    )
    assert not hasattr(inventory.exact_a0_rows[0], "smc_bias_bound")


def test_non_pass_certificate_is_rejected_before_any_test_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        bindings,
        certificate_set,
        direct_certificate,
        readiness,
    ) = _heldout_authority_fixture()
    first = certificate_set.certificates[0]
    inconclusive = BoundedPrefixCertificate.create(
        semantic_family_sha256=first.semantic_family_sha256,
        report_binding=first.report_binding,
        status=EvidenceStatus.INCONCLUSIVE,
        checks={name: True for name in H6_PREFIX_REQUIRED_CHECKS},
        obligations=("prefix evidence is incomplete",),
    )
    inconclusive_set = BoundedPrefixCertificateSet.create(
        config_sha256=certificate_set.config_sha256,
        semantic_family_sha256s=certificate_set.semantic_family_sha256s,
        certificates=(inconclusive, *certificate_set.certificates[1:]),
    )
    monkeypatch.setattr(
        heldout,
        "score_h6_exact_a0_total_v3",
        lambda **_: (_ for _ in ()).throw(
            AssertionError("test bytes reached a scorer before PASS closure")
        ),
    )

    with pytest.raises(ValueError, match="PASS"):
        score_h6_heldout_inventory_v3(
            windows=build_causal_windows((0, 1, 2), split="test"),
            opening_proof_sha256=_digest("opening"),
            checkpoint_arms=bindings,
            prefix_certificate_set=inconclusive_set,
            a0_direct_exact_prefix_certificate=direct_certificate,
            readiness=_readiness_for_certificate_set(
                readiness,
                inconclusive_set,
                direct_certificate,
            ),
        )


def test_stale_certificate_source_is_rejected_before_any_test_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        bindings,
        certificate_set,
        direct_certificate,
        readiness,
    ) = _heldout_authority_fixture()
    stale = object.__new__(BoundedPrefixCertificateSet)
    for field in dataclasses.fields(certificate_set):
        object.__setattr__(
            stale,
            field.name,
            (
                "f" * 64
                if field.name == "dirty_digest"
                else getattr(certificate_set, field.name)
            ),
        )
    monkeypatch.setattr(
        heldout,
        "score_h6_exact_a0_total_v3",
        lambda **_: (_ for _ in ()).throw(
            AssertionError("test bytes reached a scorer before source closure")
        ),
    )

    with pytest.raises(ValueError, match="source"):
        score_h6_heldout_inventory_v3(
            windows=build_causal_windows((0, 1, 2), split="test"),
            opening_proof_sha256=_digest("opening"),
            checkpoint_arms=bindings,
            prefix_certificate_set=stale,
            a0_direct_exact_prefix_certificate=direct_certificate,
            readiness=readiness,
        )


def test_readiness_owned_digest_rejects_forged_set_before_any_test_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        bindings,
        certificate_set,
        direct_certificate,
        readiness,
    ) = _heldout_authority_fixture()
    forged = object.__new__(BoundedPrefixCertificateSet)
    for field in dataclasses.fields(certificate_set):
        object.__setattr__(
            forged,
            field.name,
            (
                "e" * 64
                if field.name == "prefix_certificate_set_sha256"
                else getattr(certificate_set, field.name)
            ),
        )
    monkeypatch.setattr(
        heldout,
        "score_h6_exact_a0_total_v3",
        lambda **_: (_ for _ in ()).throw(
            AssertionError("test bytes reached a scorer before set closure")
        ),
    )

    with pytest.raises(ValueError, match="certificate-set"):
        score_h6_heldout_inventory_v3(
            windows=build_causal_windows((0, 1, 2), split="test"),
            opening_proof_sha256=_digest("opening"),
            checkpoint_arms=bindings,
            prefix_certificate_set=forged,
            a0_direct_exact_prefix_certificate=direct_certificate,
            readiness=readiness,
        )


def test_full_certificate_set_may_contain_an_unrelated_same_estimator_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        bindings,
        certificate_set,
        direct_certificate,
        readiness,
    ) = _heldout_authority_fixture()
    matching = _matching_set()
    heldout_ids = {
        H6_MATCHING_V3_ENDPOINT_CONFIG_IDS[5],
        H6_MATCHING_V3_ENDPOINT_CONFIG_IDS[9],
    }
    unrelated_config = next(
        config
        for config in matching.endpoint_configs
        if config.arm is ArmId.A5 and config.config_id not in heldout_ids
    )
    unrelated = _bounded_certificate(
        build_a5(unrelated_config),
        git_head=readiness.git_head,
        dirty_digest=readiness.dirty_digest,
        semantic_label="unrelated-family",
    )
    full_certificates = (unrelated, *certificate_set.certificates)
    full_set = BoundedPrefixCertificateSet.create(
        config_sha256=certificate_set.config_sha256,
        semantic_family_sha256s=tuple(
            certificate.semantic_family_sha256
            for certificate in full_certificates
        ),
        certificates=full_certificates,
    )
    monkeypatch.setattr(
        heldout,
        "score_h6_exact_a0_total_v3",
        lambda *, windows, **_: NllTotals(
            float(windows.counted_target_total),
            windows.counted_target_total,
        ),
    )
    monkeypatch.setattr(
        heldout,
        "score_h6_weighted_a5_total_v3",
        lambda *, windows, replicate_id, particle_count, **_: NllTotals(
            (
                1.0
                + replicate_id * 1.0e-8
                + 1.0 / float(particle_count)
            )
            * windows.counted_target_total,
            windows.counted_target_total,
        ),
    )

    inventory = score_h6_heldout_inventory_v3(
        windows=build_causal_windows((0, 1, 2), split="test"),
        opening_proof_sha256=_digest("opening"),
        checkpoint_arms=bindings,
        prefix_certificate_set=full_set,
        a0_direct_exact_prefix_certificate=direct_certificate,
        readiness=_readiness_for_certificate_set(
            readiness,
            full_set,
            direct_certificate,
        ),
    )
    assert inventory.logical_row_count == 4104


def test_ineligible_endpoint_aggregate_aborts_before_rows_are_finalized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        bindings,
        certificate_set,
        direct_certificate,
        readiness,
    ) = _heldout_authority_fixture()
    windows = build_causal_windows((0, 1, 2), split="test")
    particle_totals = {
        128: 1.0,
        256: 2.0,
        512: 1.0,
        1024: 2.0,
    }
    monkeypatch.setattr(
        heldout,
        "score_h6_exact_a0_total_v3",
        lambda *, windows, **_: NllTotals(
            float(windows.counted_target_total),
            windows.counted_target_total,
        ),
    )
    monkeypatch.setattr(
        heldout,
        "score_h6_weighted_a5_total_v3",
        lambda *, windows, particle_count, **_: NllTotals(
            particle_totals[particle_count] * windows.counted_target_total,
            windows.counted_target_total,
        ),
    )

    with pytest.raises(ValueError, match="uncertainty|eligible|PASS"):
        score_h6_heldout_inventory_v3(
            windows=windows,
            opening_proof_sha256=_digest("opening"),
            checkpoint_arms=bindings,
            prefix_certificate_set=certificate_set,
            a0_direct_exact_prefix_certificate=direct_certificate,
            readiness=readiness,
        )
