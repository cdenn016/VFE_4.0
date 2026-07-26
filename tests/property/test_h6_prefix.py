from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest
import torch

from verification.numpy_oracles.h6_prefix import (
    IndependentVocabularyIdentity,
    OrderedTailPair,
    enumerate_ordered_tail_pairs,
    generate_frozen_validation_perturbations,
    load_frozen_validation_perturbations as load_oracle_perturbations,
    ordered_tail_pair_counts,
    ordered_tail_pair_inventory,
)
from vfe4.config.schema import H6_PREFIX_V2_AUTHORIZATION_SHA256
from vfe4.data.byte_tokenizer import ByteTokenizerV1
from vfe4.predictive import vocabulary_identity_sha256
from vfe4.training.arms import (
    ArmConfig,
    BuiltArm,
    CapacityAllocation,
    build_a0,
    build_a2,
)
from vfe4.types import (
    ArmId,
    EvidenceStatus,
    EstimatorSpec,
    H6PrefixWorkloadPlan,
    PrefixCaseKey,
    VocabularyIdentity,
)
from vfe4.validation.h6_prefix import (
    DynamicCheckResult,
    DynamicExecutionPlan,
    DynamicPrefixCase,
    PairSideHarness,
    PERTURBATION_FIXTURE_PATH,
    SourceMaskObservation,
    load_frozen_validation_perturbations,
    observe_all_invalid_source_rejection,
    run_dynamic_prefix_checks,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SPARSE_CASE_INDICES = (1, 6562, 8749, 9478)
VALIDATION_FIXTURE_DOMAIN = b"VFE4-H6-VALIDATION-SAFETY-FIXTURE-V1\x00"


class _RecordingPredictor:
    def __init__(self, delegate: object) -> None:
        self._delegate = delegate
        self.call_count = 0

    def __getattr__(self, name: str) -> object:
        return getattr(self._delegate, name)

    def next_token_log_probs(
        self,
        prefix_tokens: object,
        estimator_rng: object,
        cache: object = None,
    ) -> object:
        self.call_count += 1
        return self._delegate.next_token_log_probs(  # type: ignore[attr-defined]
            prefix_tokens,
            estimator_rng,
            cache,
        )


@pytest.fixture(scope="module")
def sparse_ordered_pairs() -> tuple[OrderedTailPair, ...]:
    return tuple(
        enumerate_ordered_tail_pairs(
            vocabulary_size=3,
            horizon=4,
            case_indices=SPARSE_CASE_INDICES + (0,),
            max_cases=4,
        )
    )


def _arm_config(
    arm: ArmId,
    vocabulary: VocabularyIdentity | None = None,
) -> ArmConfig:
    vocabulary = vocabulary or VocabularyIdentity(
        "h6-prefix-small-v1", 3, SHA_A
    )
    if arm is ArmId.A0:
        semantics = {
            "config_id": "h6-a0-transformer-v2",
            "latent_enabled": False,
            "state_channel_enabled": False,
            "model_channel_enabled": False,
            "source_mode": "absent",
            "map_mode": "absent",
            "recognition_family": "absent",
            "recognition_conditioning": "absent",
            "prior_variant": "absent",
            "mixture_mode": "absent",
            "objective_kind": "cross_entropy",
        }
        allocation = CapacityAllocation.create(
            emission_width=4,
            latent_width=None,
            recognition_width=None,
        )
    elif arm is ArmId.A2:
        semantics = {
            "config_id": "h6-a2-generic-map-v1",
            "latent_enabled": True,
            "state_channel_enabled": True,
            "model_channel_enabled": True,
            "source_mode": "categorical",
            "map_mode": "generic_fixed_frame_non_coboundary",
            "recognition_family": "structured",
            "recognition_conditioning": "smoothing",
            "prior_variant": "fixed",
            "mixture_mode": "exact",
            "objective_kind": "complete_elbo",
        }
        allocation = CapacityAllocation.create(
            emission_width=4,
            latent_width=2,
            recognition_width=4,
        )
    else:
        raise AssertionError("this focused fixture supports only A0 and A2")
    return ArmConfig.create(
        arm=arm,
        vocabulary=vocabulary,
        horizon=4,
        capacity_allocation=allocation,
        **semantics,
    )


def _case_key(built: BuiltArm) -> PrefixCaseKey:
    predictor = built.predictor
    return PrefixCaseKey(
        arm=built.config.arm,
        predictor_config_sha256=predictor.predictor_config_sha256,
        estimator_sha256=predictor.estimator_spec.estimator_sha256,
        model_family_sha256=built.model_family_sha256,
        vocabulary_sha256=vocabulary_identity_sha256(built.config.vocabulary),
        data_safety_sha256=predictor.data_safety_sha256,
        git_head="1" * 40,
        dirty_digest=SHA_B,
    )


def _dynamic_cases(
    sparse_ordered_pairs: tuple[OrderedTailPair, ...],
) -> tuple[DynamicPrefixCase, ...]:
    return tuple(
        DynamicPrefixCase.create(
            ordinal=pair.case_index,
            receiver_t=pair.receiver_t,
            shared_prefix=pair.prefix,
            left_tail=pair.left_tail,
            right_tail=pair.right_tail,
        )
        for pair in sparse_ordered_pairs
    )


def _tiny_validation_fixture_bytes() -> bytes:
    header = (
        VALIDATION_FIXTURE_DOMAIN
        + bytes.fromhex(SHA_B)
        + (2).to_bytes(4, "little")
    )
    rows = bytearray()
    for source_index, (start, target_count) in enumerate(((0, 4), (32, 3))):
        token_ids = tuple(
            (17 * source_index + coordinate) % 258
            for coordinate in range(33)
        )
        rows += start.to_bytes(8, "little")
        rows += target_count.to_bytes(2, "little")
        rows += b"".join(
            token_id.to_bytes(2, "little") for token_id in token_ids
        )
    return header + bytes(rows)


def test_closed_form_counts_and_sparse_iterator_cap(
    sparse_ordered_pairs: tuple[OrderedTailPair, ...],
) -> None:
    inventory = ordered_tail_pair_inventory(vocabulary_size=3, horizon=4)

    assert ordered_tail_pair_counts(vocabulary_size=3, horizon=4) == (
        6561,
        2187,
        729,
        243,
    )
    assert inventory.total_count == 9720
    assert tuple(pair.case_index for pair in sparse_ordered_pairs) == (
        SPARSE_CASE_INDICES
    )
    assert tuple(pair.receiver_t for pair in sparse_ordered_pairs) == (1, 2, 3, 4)
    assert all(
        pair.left_tail != pair.right_tail for pair in sparse_ordered_pairs
    )
    assert len(sparse_ordered_pairs) == 4


def test_two_record_v258_case_file_has_one_canonical_loader_contract() -> None:
    fixture_bytes = _tiny_validation_fixture_bytes()
    oracle_vocabulary = IndependentVocabularyIdentity.create(
        vocabulary_id="h6-byte-tokenizer-v1",
        size=258,
        tokenizer_spec_sha256=SHA_A,
    )
    production_vocabulary = VocabularyIdentity(
        oracle_vocabulary.vocabulary_id,
        oracle_vocabulary.size,
        oracle_vocabulary.tokenizer_spec_sha256,
    )
    oracle_generated = generate_frozen_validation_perturbations(
        fixture_bytes,
        vocabulary=oracle_vocabulary,
        max_cases=2,
    )
    canonical_bytes = oracle_generated.canonical_bytes
    oracle_loaded = load_oracle_perturbations(
        canonical_bytes,
        max_cases=2,
    )
    production_loaded = load_frozen_validation_perturbations(
        canonical_bytes,
        expected_vocabulary=production_vocabulary,
    )

    assert oracle_vocabulary.vocabulary_sha256 == vocabulary_identity_sha256(
        production_vocabulary
    )
    assert hashlib.sha256(canonical_bytes).hexdigest() == oracle_loaded.raw_sha256
    assert production_loaded.canonical_bytes == oracle_loaded.canonical_bytes
    assert production_loaded.manifest_sha256 == oracle_loaded.manifest_sha256
    assert production_loaded.materialized_count == oracle_loaded.materialized_count == 2
    assert tuple(record.case_sha256 for record in production_loaded.records) == tuple(
        record.case_sha256 for record in oracle_loaded.records
    )
    committed = load_frozen_validation_perturbations(
        PERTURBATION_FIXTURE_PATH,
        expected_vocabulary=ByteTokenizerV1().vocabulary_identity,
    )
    assert committed.materialized_count == 2
    assert committed.materialization == "focused_subset"

    validation_plan = DynamicExecutionPlan.create(
        mode="focused_subset",
        case_family="validation",
    )
    v258_a0 = build_a0(_arm_config(ArmId.A0, production_vocabulary))
    v258_harness = PairSideHarness()
    v258_report = run_dynamic_prefix_checks(
        key=_case_key(v258_a0),
        predictor=v258_a0.predictor,
        arm_config=v258_a0.config,
        cases=production_loaded.dynamic_cases,
        plan=validation_plan,
        stream_seed=2026072302,
        perturbations=production_loaded,
        pair_side_harness=v258_harness,
    )
    v258_checks = {check.name: check for check in v258_report.checks}
    assert v258_a0.predictor.particle_count == 4
    assert v258_report.status is EvidenceStatus.INCONCLUSIVE
    assert v258_report.completed_by_position == (2,)
    assert v258_checks["dynamic_target_suffix_leakage"].status is EvidenceStatus.PASS
    assert v258_checks["cache_identity"].status is EvidenceStatus.PASS
    assert v258_checks["validation_data_safety"].status is EvidenceStatus.INCONCLUSIVE
    assert v258_harness.trace_count == 8


def test_focused_dynamic_reports_fail_only_on_a_witnessed_defect(
    sparse_ordered_pairs: tuple[OrderedTailPair, ...],
) -> None:
    cases = _dynamic_cases(sparse_ordered_pairs)
    plan = DynamicExecutionPlan.create(
        mode="focused_subset",
        case_family="small",
    )
    a0 = build_a0(_arm_config(ArmId.A0))
    correct_harness = PairSideHarness()
    correct = run_dynamic_prefix_checks(
        key=_case_key(a0),
        predictor=a0.predictor,
        arm_config=a0.config,
        cases=cases,
        plan=plan,
        stream_seed=2026072301,
        pair_side_harness=correct_harness,
    )
    correct_checks = {check.name: check for check in correct.checks}

    assert a0.predictor.particle_count == 4
    assert correct.status is EvidenceStatus.INCONCLUSIVE
    assert correct.completed_by_position == (1, 1, 1, 1)
    assert correct_checks["signature_and_identity"].status is EvidenceStatus.PASS
    assert (
        correct_checks["dynamic_target_suffix_leakage"].status
        is EvidenceStatus.PASS
    )
    assert correct_checks["cache_identity"].status is EvidenceStatus.PASS
    assert correct_checks["case_inventory"].status is EvidenceStatus.INCONCLUSIVE
    assert correct_harness.trace_count == 4 * len(cases)

    class TargetAwareSignature:
        def __init__(self, delegate: object) -> None:
            self._delegate = delegate

        def __getattr__(self, name: str) -> object:
            return getattr(self._delegate, name)

        def next_token_log_probs(
            self,
            prefix_tokens: object,
            estimator_rng: object,
            cache: object = None,
            target_tokens: object = None,
        ) -> object:
            return self._delegate.next_token_log_probs(  # type: ignore[attr-defined]
                prefix_tokens,
                estimator_rng,
                cache,
            )

    signature_mutant = run_dynamic_prefix_checks(
        key=_case_key(a0),
        predictor=TargetAwareSignature(a0.predictor),
        arm_config=a0.config,
        cases=(cases[0],),
        plan=plan,
        stream_seed=2026072301,
    )
    assert signature_mutant.status is EvidenceStatus.FAIL
    assert signature_mutant.checks[0].status is EvidenceStatus.FAIL
    assert "noncausal argument" in (signature_mutant.first_counterexample or "")

    class SuffixAwareSignature(TargetAwareSignature):
        def next_token_log_probs(
            self,
            prefix_tokens: object,
            estimator_rng: object,
            cache: object = None,
            suffix_tokens: object = None,
        ) -> object:
            return self._delegate.next_token_log_probs(  # type: ignore[attr-defined]
                prefix_tokens,
                estimator_rng,
                cache,
            )

    suffix_mutant = run_dynamic_prefix_checks(
        key=_case_key(a0),
        predictor=SuffixAwareSignature(a0.predictor),
        arm_config=a0.config,
        cases=(cases[0],),
        plan=plan,
        stream_seed=2026072301,
    )
    assert suffix_mutant.status is EvidenceStatus.FAIL

    class HiddenTailReader(TargetAwareSignature):
        def __init__(self, delegate: object, harness: PairSideHarness) -> None:
            super().__init__(delegate)
            self._harness = harness

        def next_token_log_probs(
            self,
            prefix_tokens: object,
            estimator_rng: object,
            cache: object = None,
        ) -> object:
            prediction = self._delegate.next_token_log_probs(  # type: ignore[attr-defined]
                prefix_tokens,
                estimator_rng,
                cache,
            )
            tail = self._harness.current_tail
            if tail is not None and tail[-1] == 1:
                owned = prediction.log_probs._FrozenTensorSnapshot__owned
                with torch.no_grad():
                    owned[0].add_(0.125)
            return prediction

    hidden_harness = PairSideHarness()
    hidden_tail_mutant = run_dynamic_prefix_checks(
        key=_case_key(a0),
        predictor=HiddenTailReader(a0.predictor, hidden_harness),
        arm_config=a0.config,
        cases=(cases[0],),
        plan=plan,
        stream_seed=2026072301,
        pair_side_harness=hidden_harness,
    )
    hidden_checks = {
        check.name: check for check in hidden_tail_mutant.checks
    }
    assert hidden_checks["signature_and_identity"].status is EvidenceStatus.PASS
    assert (
        hidden_checks["dynamic_target_suffix_leakage"].status
        is EvidenceStatus.FAIL
    )

    class MissingCacheConfigIdentity(TargetAwareSignature):
        def next_token_log_probs(
            self,
            prefix_tokens: object,
            estimator_rng: object,
            cache: object = None,
        ) -> object:
            prediction = self._delegate.next_token_log_probs(  # type: ignore[attr-defined]
                prefix_tokens,
                estimator_rng,
                cache,
            )
            object.__setattr__(
                prediction.cache.key,
                "predictor_config_sha256",
                "0" * 64,
            )
            return prediction

    cache_mutant = run_dynamic_prefix_checks(
        key=_case_key(a0),
        predictor=MissingCacheConfigIdentity(a0.predictor),
        arm_config=a0.config,
        cases=(cases[0],),
        plan=plan,
        stream_seed=2026072301,
    )
    cache_checks = {check.name: check for check in cache_mutant.checks}
    assert cache_mutant.status is EvidenceStatus.FAIL
    assert cache_checks["cache_identity"].status is EvidenceStatus.FAIL

    a2 = build_a2(_arm_config(ArmId.A2))
    missing_mask_evidence = run_dynamic_prefix_checks(
        key=_case_key(a2),
        predictor=a2.predictor,
        arm_config=a2.config,
        cases=(cases[1],),
        plan=plan,
        stream_seed=2026072301,
    )
    missing_checks = {
        check.name: check for check in missing_mask_evidence.checks
    }

    assert a2.predictor.particle_count == 4
    assert missing_mask_evidence.status is EvidenceStatus.INCONCLUSIVE
    assert missing_checks["source_mask"].status is EvidenceStatus.INCONCLUSIVE
    assert any(
        "source-mask" in obligation or "source-row" in obligation
        for obligation in missing_checks["source_mask"].obligations
    )

    receiver_t = cases[1].receiver_t
    invalid_post_softmax = a2.model.state_source_log_probs(receiver_t).detach().clone()
    invalid_post_softmax[0] += 0.25
    mask_mutant = run_dynamic_prefix_checks(
        key=_case_key(a2),
        predictor=a2.predictor,
        arm_config=a2.config,
        cases=(cases[1],),
        plan=plan,
        stream_seed=2026072301,
        source_mask_observations=(
            SourceMaskObservation.capture(
                case_sha256=cases[1].case_sha256,
                config_sha256=a2.config.config_sha256,
                bank="state",
                receiver_t=receiver_t,
                declared_parents=tuple(range(receiver_t)),
                log_probabilities=invalid_post_softmax,
            ),
        ),
        all_invalid_observation=observe_all_invalid_source_rejection(
            config_sha256=a2.config.config_sha256,
            receiver_t=receiver_t,
            probe=lambda: torch.zeros(receiver_t, dtype=torch.float64),
        ),
    )
    mask_checks = {check.name: check for check in mask_mutant.checks}
    assert mask_mutant.status is EvidenceStatus.FAIL
    assert mask_checks["source_mask"].status is EvidenceStatus.FAIL
    assert "all-invalid" in (mask_checks["source_mask"].first_counterexample or "") or (
        "post-softmax" in (mask_checks["source_mask"].first_counterexample or "")
    )


def test_scoped_dynamic_plans_freeze_bounded_workload_and_preserve_v1() -> None:
    workload = H6PrefixWorkloadPlan()

    def synthetic_rows(
        family: str,
        indices: tuple[int, ...],
    ) -> tuple[tuple[int, str], ...]:
        return tuple(
            (
                index,
                hashlib.sha256(
                    f"h6-prefix-{family}-case-{index}".encode("ascii")
                ).hexdigest(),
            )
            for index in indices
        )

    legacy = DynamicExecutionPlan.create(
        mode="focused_subset",
        case_family="small",
    )
    assert legacy.schema_version == "h6-dynamic-execution-plan-v1"
    assert (
        legacy.plan_sha256
        == "78e337c103655f01ba11b8d070c44aa77705345c4e23e41e4c38dd01febde729"
    )

    matrix = (
        (
            "representative_exhaustive",
            "small",
            128,
            tuple(range(9_720)),
            (6_561, 2_187, 729, 243),
            9_720,
        ),
        (
            "representative_exhaustive",
            "validation",
            128,
            tuple(range(4_096)),
            (4_096,),
            4_096,
        ),
        (
            "estimator_stratified",
            "small",
            256,
            workload.small_global_case_indices,
            (4, 4, 4, 4),
            9_720,
        ),
        (
            "estimator_stratified",
            "small",
            512,
            workload.small_global_case_indices,
            (4, 4, 4, 4),
            9_720,
        ),
        (
            "estimator_stratified",
            "small",
            1_024,
            workload.small_global_case_indices,
            (4, 4, 4, 4),
            9_720,
        ),
        (
            "estimator_stratified",
            "validation",
            256,
            workload.validation_global_case_indices,
            (16,),
            4_096,
        ),
        (
            "estimator_stratified",
            "validation",
            512,
            workload.validation_global_case_indices,
            (16,),
            4_096,
        ),
        (
            "estimator_stratified",
            "validation",
            1_024,
            workload.validation_global_case_indices,
            (16,),
            4_096,
        ),
    )
    plans: dict[tuple[str, str, int], DynamicExecutionPlan] = {}
    for scope, family, particle_count, indices, counts, denominator in matrix:
        plan = DynamicExecutionPlan.create_scoped(
            scope=scope,
            case_family=family,
            particle_count=particle_count,
            workload_plan=workload,
            authorization_sha256=H6_PREFIX_V2_AUTHORIZATION_SHA256,
            selection_rows=synthetic_rows(family, indices),
        )
        plans[(scope, family, particle_count)] = plan
        assert plan.schema_version == "h6-dynamic-execution-plan-v2"
        assert plan.scope == scope
        assert plan.case_family == family
        assert plan.particle_count == particle_count
        assert plan.expected_by_position == counts
        assert plan.full_expected_count == denominator
        assert plan.selected_global_indices == indices
        assert plan.workload_plan_sha256 == workload.workload_plan_sha256
        assert plan.authorization_sha256 == H6_PREFIX_V2_AUTHORIZATION_SHA256

    for family in ("small", "validation"):
        family_plans = tuple(
            plans[("estimator_stratified", family, particle_count)]
            for particle_count in (256, 512, 1_024)
        )
        assert len(
            {plan.selection_manifest_sha256 for plan in family_plans}
        ) == 1
        assert len({plan.plan_sha256 for plan in family_plans}) == 3

    small_rows = synthetic_rows(
        "small", workload.small_global_case_indices
    )
    scoped_kwargs = {
        "scope": "estimator_stratified",
        "case_family": "small",
        "particle_count": 256,
        "workload_plan": workload,
        "authorization_sha256": H6_PREFIX_V2_AUTHORIZATION_SHA256,
        "selection_rows": small_rows,
    }
    with pytest.raises(ValueError):
        DynamicExecutionPlan.create_scoped(
            **{**scoped_kwargs, "particle_count": True}
        )
    with pytest.raises(ValueError):
        DynamicExecutionPlan.create_scoped(
            **{
                **scoped_kwargs,
                "scope": "representative_exhaustive",
                "particle_count": 256,
            }
        )
    with pytest.raises(ValueError):
        DynamicExecutionPlan.create_scoped(
            **{**scoped_kwargs, "particle_count": 128}
        )
    with pytest.raises(ValueError):
        DynamicExecutionPlan.create_scoped(
            **{**scoped_kwargs, "authorization_sha256": None}
        )
    with pytest.raises(ValueError):
        DynamicExecutionPlan.create_scoped(
            **{**scoped_kwargs, "selection_rows": small_rows[::-1]}
        )
    with pytest.raises(ValueError):
        DynamicExecutionPlan.create_scoped(
            **{
                **scoped_kwargs,
                "selection_rows": (small_rows[0],) + small_rows,
            }
        )
    with pytest.raises(ValueError):
        DynamicExecutionPlan.create_scoped(
            **{
                **scoped_kwargs,
                "selection_rows": small_rows
                + (
                    (
                        9_718,
                        hashlib.sha256(b"extra-case").hexdigest(),
                    ),
                ),
            }
        )
    with pytest.raises(TypeError):
        DynamicExecutionPlan.create_scoped(
            **scoped_kwargs,
            expected_by_position=(16,),  # type: ignore[call-arg]
        )
    with pytest.raises(ValueError):
        replace(
            plans[("estimator_stratified", "small", 256)],
            workload_plan_sha256=SHA_A,
        )

    pairs = tuple(
        enumerate_ordered_tail_pairs(
            vocabulary_size=3,
            horizon=4,
            case_indices=workload.small_global_case_indices,
            max_cases=16,
        )
    )
    cases = _dynamic_cases(pairs)
    a0 = build_a0(_arm_config(ArmId.A0))
    legacy_predictor = _RecordingPredictor(a0.predictor)
    legacy_report = run_dynamic_prefix_checks(
        key=_case_key(a0),
        predictor=legacy_predictor,
        arm_config=a0.config,
        cases=(cases[0],),
        plan=legacy,
        stream_seed=2026072501,
        pair_side_harness=PairSideHarness(),
    )
    assert legacy_report.schema_version == "h6-dynamic-prefix-report-v1"
    assert legacy_report.status is EvidenceStatus.INCONCLUSIVE
    assert legacy_report.obligations == (
        "focused subset is not H6-Prefix evidence",
        "companion 4,096 validation-family report is required",
    )
    assert legacy_predictor.call_count == 5

    plan = DynamicExecutionPlan.create_scoped(
        **{
            **scoped_kwargs,
            "selection_rows": tuple(
                (case.ordinal, case.case_sha256) for case in cases
            ),
        }
    )
    for supplied_cases in (
        cases[::-1],
        (
            DynamicPrefixCase.create(
                ordinal=cases[0].ordinal,
                receiver_t=cases[0].receiver_t,
                shared_prefix=cases[0].shared_prefix,
                left_tail=cases[0].left_tail,
                right_tail=(
                    (cases[0].right_tail[0] + 1) % 3,
                    *cases[0].right_tail[1:],
                ),
            ),
            *cases[1:],
        ),
    ):
        predictor = _RecordingPredictor(a0.predictor)
        harness = PairSideHarness()
        with pytest.raises(ValueError, match="selection"):
            run_dynamic_prefix_checks(
                key=_case_key(a0),
                predictor=predictor,
                arm_config=a0.config,
                cases=supplied_cases,
                plan=plan,
                stream_seed=2026072501,
                pair_side_harness=harness,
            )
        assert predictor.call_count == 0
        assert harness.trace_count == 0


def test_stratified_dynamic_report_is_scope_complete_and_explicitly_applicable() -> None:
    workload = H6PrefixWorkloadPlan()
    pairs = tuple(
        enumerate_ordered_tail_pairs(
            vocabulary_size=3,
            horizon=4,
            case_indices=workload.small_global_case_indices,
            max_cases=16,
        )
    )
    cases = _dynamic_cases(pairs)
    selection_rows = tuple(
        (case.ordinal, case.case_sha256) for case in cases
    )
    plan = DynamicExecutionPlan.create_scoped(
        scope="estimator_stratified",
        case_family="small",
        particle_count=256,
        workload_plan=workload,
        authorization_sha256=H6_PREFIX_V2_AUTHORIZATION_SHA256,
        selection_rows=selection_rows,
    )
    a0 = build_a0(_arm_config(ArmId.A0))
    _, scoped_predictor = a0.rebuild_predictive_boundary(
        EstimatorSpec.create(
            kind="weighted_smc",
            particle_count=256,
            resampling="systematic_ess_half",
        )
    )
    key = PrefixCaseKey(
        arm=a0.config.arm,
        predictor_config_sha256=scoped_predictor.predictor_config_sha256,
        estimator_sha256=scoped_predictor.estimator_spec.estimator_sha256,
        model_family_sha256=a0.model_family_sha256,
        vocabulary_sha256=vocabulary_identity_sha256(a0.config.vocabulary),
        data_safety_sha256=scoped_predictor.data_safety_sha256,
        git_head="1" * 40,
        dirty_digest=SHA_B,
    )
    predictor = _RecordingPredictor(scoped_predictor)
    harness = PairSideHarness()
    report = run_dynamic_prefix_checks(
        key=key,
        predictor=predictor,
        arm_config=a0.config,
        cases=cases,
        plan=plan,
        stream_seed=2026072502,
        pair_side_harness=harness,
    )
    checks = {check.name: check for check in report.checks}

    assert report.schema_version == "h6-dynamic-prefix-report-v2"
    assert report.scope == "estimator_stratified"
    assert report.case_family == "small"
    assert report.particle_count == 256
    assert report.workload_plan_sha256 == workload.workload_plan_sha256
    assert report.selected_global_indices == workload.small_global_case_indices
    assert report.selection_manifest_sha256 == plan.selection_manifest_sha256
    assert report.applicable_check_names == (
        "signature_and_identity",
        "dynamic_target_suffix_leakage",
        "cache_identity",
    )
    assert report.completed_by_position == (4, 4, 4, 4)
    assert report.status is EvidenceStatus.PASS
    assert report.obligations == ()
    assert report.complete_case_manifest_sha256 is None
    assert checks["signature_and_identity"].status is EvidenceStatus.PASS
    for name in ("dynamic_target_suffix_leakage", "cache_identity"):
        assert checks[name].status is EvidenceStatus.PASS
        assert checks[name].completed_count == checks[name].expected_count == 16
    for name in ("source_mask", "case_inventory", "validation_data_safety"):
        assert checks[name].status is EvidenceStatus.PASS
        assert checks[name].expected_count == checks[name].completed_count == 0
        assert checks[name].violation_count == 0
        assert checks[name].first_counterexample is None
        assert checks[name].obligations == ()
    assert predictor.call_count == 5 * len(cases)
    assert harness.trace_count == 4 * len(cases)

    with pytest.raises(ValueError):
        replace(plan, selection_manifest_sha256=SHA_A)
    with pytest.raises(ValueError):
        replace(report, particle_count=512)
    with pytest.raises(ValueError, match="scope-complete"):
        replace(report, completed_by_position=(3, 4, 4, 4))

    mismatch_plan = DynamicExecutionPlan.create_scoped(
        scope="estimator_stratified",
        case_family="small",
        particle_count=512,
        workload_plan=workload,
        authorization_sha256=H6_PREFIX_V2_AUTHORIZATION_SHA256,
        selection_rows=selection_rows,
    )
    mismatch_predictor = _RecordingPredictor(scoped_predictor)
    mismatch_harness = PairSideHarness()
    with pytest.raises(ValueError, match="particle count"):
        run_dynamic_prefix_checks(
            key=key,
            predictor=mismatch_predictor,
            arm_config=a0.config,
            cases=cases,
            plan=mismatch_plan,
            stream_seed=2026072502,
            pair_side_harness=mismatch_harness,
        )
    assert mismatch_predictor.call_count == 0
    assert mismatch_harness.trace_count == 0

    source_evidence_predictor = _RecordingPredictor(scoped_predictor)
    source_evidence_harness = PairSideHarness()
    with pytest.raises(ValueError, match="source-mask"):
        run_dynamic_prefix_checks(
            key=key,
            predictor=source_evidence_predictor,
            arm_config=a0.config,
            cases=cases,
            plan=plan,
            stream_seed=2026072502,
            source_mask_observations=(),
            pair_side_harness=source_evidence_harness,
        )
    assert source_evidence_predictor.call_count == 0
    assert source_evidence_harness.trace_count == 0

    incomplete_leakage = DynamicCheckResult.create(
        name="dynamic_target_suffix_leakage",
        status=EvidenceStatus.PASS,
        expected_count=16,
        completed_count=15,
    )
    with pytest.raises(ValueError, match="applicable PASS"):
        replace(
            report,
            checks=(
                report.checks[0],
                incomplete_leakage,
                *report.checks[2:],
            ),
        )
