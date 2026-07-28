from __future__ import annotations

import dataclasses
from functools import cache

import pytest

from vfe4.training.h6_matching_v3 import (
    H6_MATCHING_V3_ENDPOINT_CONFIG_IDS,
    H6_MATCHING_POLICY_V3,
    H6_MATCHING_V3_ESTIMATOR_TERM_NAMES,
    H6_MATCHING_V3_EXCLUDED_OPERATIONS,
    H6MatchingSetV3,
    H6TrainingWorkloadV3,
    analytical_training_flop_ledger_v3,
    build_h6_matching_set_v3,
    endpoint_parameter_count_v3,
    primary_matching_diagnostics_v3,
)
from vfe4.training.matching import (
    A5_REFERENCE_ALLOCATION,
    endpoint_formula_profile,
)
from vfe4.training.parameter_counts import (
    arm_parameter_count,
    arm_parameter_count_v3,
    arm_source_bank_count_v3,
    fixed_source_prior_parameter_count,
    parent_specific_pooled_prefix_source_prior_parameter_count,
    recognition_source_parameter_count_v3,
)
from vfe4.types.h6 import (
    ArmConfig,
    ArmId,
    CapacityAllocation,
    VocabularyIdentity,
)

_GIT_HEAD = "1" * 40
_DIRTY_DIGEST = "2" * 64


def _config(
    config_id: str,
    allocation: CapacityAllocation,
) -> ArmConfig:
    profile = endpoint_formula_profile(config_id)
    return ArmConfig.create(
        arm=ArmId(profile.arm),
        config_id=config_id,
        vocabulary=VocabularyIdentity(
            vocabulary_id="h6-matching-v3-test",
            size=258,
            tokenizer_spec_sha256="b" * 64,
        ),
        horizon=32,
        latent_enabled=profile.latent_enabled,
        state_channel_enabled=profile.channel_count >= 1,
        model_channel_enabled=profile.channel_count == 2,
        source_mode=profile.source_mode,
        map_mode=profile.map_mode,
        recognition_family=profile.recognition_family,
        recognition_conditioning=profile.recognition_conditioning,
        prior_variant=profile.prior_variant,
        mixture_mode=profile.mixture_mode,
        objective_kind=profile.objective_kind,
        capacity_allocation=allocation,
    )


@cache
def _endpoint_templates() -> tuple[ArmConfig, ...]:
    templates: list[ArmConfig] = []
    for config_id in H6_MATCHING_V3_ENDPOINT_CONFIG_IDS:
        profile = endpoint_formula_profile(config_id)
        if config_id == "h6-a0-transformer-v2":
            allocation = CapacityAllocation.create(
                emission_width=52,
                latent_width=None,
                recognition_width=None,
            )
        elif not profile.latent_enabled:
            allocation = CapacityAllocation.create(
                emission_width=64,
                latent_width=None,
                recognition_width=None,
            )
        elif profile.prior_variant == "parent_specific_pooled_prefix":
            allocation = CapacityAllocation.create(
                emission_width=89,
                latent_width=2,
                recognition_width=113,
                prior_context_width=6,
            )
        else:
            allocation = A5_REFERENCE_ALLOCATION
        templates.append(_config(config_id, allocation))
    return tuple(templates)


@cache
def _matching_set() -> H6MatchingSetV3:
    workload = H6TrainingWorkloadV3.from_train_tokens(
        train_token_count=258,
        train_token_sha256="a" * 64,
    )
    return H6MatchingSetV3.create(
        git_head=_GIT_HEAD,
        dirty_digest=_DIRTY_DIGEST,
        workload=workload,
        endpoint_templates=_endpoint_templates(),
    )


def test_public_matching_builder_regenerates_the_exact_frozen_set() -> None:
    expected = _matching_set()

    rebuilt = build_h6_matching_set_v3(
        git_head=_GIT_HEAD,
        dirty_digest=_DIRTY_DIGEST,
        train_token_count=expected.workload.train_token_count,
        train_token_sha256=expected.workload.train_token_sha256,
        vocabulary=expected.endpoint_configs[0].vocabulary,
        horizon=expected.endpoint_configs[0].horizon,
    )

    assert rebuilt == expected


def test_matching_v3_counts_terminal_source_parameters_by_arm() -> None:
    expected_banks = {
        "A0": 0,
        "A1": 0,
        "A2": 2,
        "A3": 0,
        "A4": 1,
        "A5": 2,
    }
    for arm, bank_count in expected_banks.items():
        assert arm_source_bank_count_v3(arm) == bank_count

    assert recognition_source_parameter_count_v3(
        bank_count=2,
        recognition_width=64,
        latent_width=16,
    ) == 2 * (64 + 1 + 16)
    for arm in ("A1", "A2", "A3", "A4", "A5"):
        base = arm_parameter_count(
            arm,
            vocabulary_size=258,
            horizon=32,
            emission_width=64,
            latent_width=16,
            recognition_width=64,
        )
        amended = arm_parameter_count_v3(
            arm,
            vocabulary_size=258,
            horizon=32,
            emission_width=64,
            latent_width=16,
            recognition_width=64,
        )
        assert amended - base == expected_banks[arm] * (64 + 1 + 16)


def test_endpoint_parameter_count_v3_uses_canonical_factorized_family() -> None:
    factorized = _config(
        "h6-a5-factorized-fixed-exact-complete-latent-smoothing-v1",
        A5_REFERENCE_ALLOCATION,
    )

    assert factorized.recognition_family == "factorized"
    assert endpoint_parameter_count_v3(factorized) == 63_540


def test_endpoint_parameter_count_v3_counts_parent_specific_prior_exactly() -> None:
    fixed = _config(
        "h6-a5-structured-fixed-exact-complete-latent-smoothing-v1",
        CapacityAllocation.create(
            emission_width=80,
            latent_width=8,
            recognition_width=96,
        ),
    )
    parent_specific = _config(
        ("h6-a5-structured-parent-specific-prefix-exact-complete-latent-smoothing-v2"),
        CapacityAllocation.create(
            emission_width=80,
            latent_width=8,
            recognition_width=96,
            prior_context_width=6,
        ),
    )

    fixed_count = arm_parameter_count_v3(
        "A5",
        vocabulary_size=258,
        horizon=32,
        emission_width=80,
        latent_width=8,
        recognition_width=96,
    )
    expected_parent_count = (
        fixed_count
        - fixed_source_prior_parameter_count(
            horizon=32,
            bank_count=2,
        )
        + parent_specific_pooled_prefix_source_prior_parameter_count(
            vocabulary_size=258,
            horizon=32,
            latent_width=8,
            context_width=6,
            gauge_anchored=True,
        )
    )

    assert endpoint_parameter_count_v3(fixed) == fixed_count
    assert endpoint_parameter_count_v3(parent_specific) == expected_parent_count
    assert endpoint_parameter_count_v3(parent_specific) != fixed_count


def test_matching_v3_has_complete_named_estimator_flop_terms() -> None:
    workload = H6TrainingWorkloadV3.from_train_tokens(
        train_token_count=258,
        train_token_sha256="a" * 64,
    )
    config = _config(
        "h6-a5-structured-fixed-exact-complete-latent-smoothing-v1",
        A5_REFERENCE_ALLOCATION,
    )
    ledger = analytical_training_flop_ledger_v3(
        endpoint_config=config,
        workload=workload,
    )
    v3_names = {
        term.operation.removeprefix("v3::")
        for term in ledger.terms
        if term.operation.startswith("v3::")
    }
    assert set(H6_MATCHING_V3_ESTIMATOR_TERM_NAMES) <= v3_names
    assert ledger.source_bank_count == 2
    assert ledger.source_parameter_count == 2 * (64 + 1 + 16)
    assert ledger.total_arithmetic_flops == sum(
        term.total_arithmetic_flops for term in ledger.terms
    )
    incremental_clip = tuple(
        term for term in ledger.terms if term.operation == "v3::global_norm_clipping"
    )
    assert len(incremental_clip) == 1
    assert (
        incremental_clip[0].arithmetic_flops_per_repetition
        == 3 * ledger.source_parameter_count
    )
    terminal_shift = tuple(
        term
        for term in ledger.terms
        if term.operation == "v3::terminal_rank_one_component_shift"
    )
    assert len(terminal_shift) == 2
    assert all(
        term.arithmetic_flops_per_repetition == 2 * 2 * 32**2 * 16
        for term in terminal_shift
    )


def test_matching_v3_exclusion_inventory_is_exact() -> None:
    assert H6_MATCHING_POLICY_V3.excluded_operations == (
        "data_io",
        "validation",
        "checkpoint_serialization",
        "test_scoring",
        "cpu_to_cuda_noise_transfer",
        "prediction_particle_propagation",
        "prediction_cache",
    )
    assert (
        H6_MATCHING_POLICY_V3.excluded_operations == H6_MATCHING_V3_EXCLUDED_OPERATIONS
    )
    assert H6_MATCHING_POLICY_V3.selection_rule == ("first_lexicographic_hard_eligible")


def test_matching_v3_policy_binds_primary_grid_amendment() -> None:
    assert H6_MATCHING_POLICY_V3.primary_emission_width_candidates == (
        72,
        84,
        85,
        86,
        87,
        88,
        89,
    )
    assert H6_MATCHING_POLICY_V3.primary_joint_candidate_count == 378

    with pytest.raises(ValueError, match="frozen design"):
        dataclasses.replace(
            H6_MATCHING_POLICY_V3,
            primary_emission_width_candidates=(84, 85, 86, 87, 88, 89),
        )


def test_matching_set_v3_requires_complete_regenerated_inventory() -> None:
    workload = H6TrainingWorkloadV3.from_train_tokens(
        train_token_count=258,
        train_token_sha256="a" * 64,
    )
    templates = _endpoint_templates()

    with pytest.raises(ValueError, match="twelve|12|complete.*inventory"):
        H6MatchingSetV3.create(
            git_head=_GIT_HEAD,
            dirty_digest=_DIRTY_DIGEST,
            workload=workload,
            endpoint_templates=templates[:-1],
        )
    with pytest.raises(ValueError, match="reordered|inventory"):
        H6MatchingSetV3.create(
            git_head=_GIT_HEAD,
            dirty_digest=_DIRTY_DIGEST,
            workload=workload,
            endpoint_templates=(templates[1], templates[0], *templates[2:]),
        )


def test_matching_set_v3_freezes_gates_and_first_lexicographic_selection() -> None:
    matching_set = _matching_set()

    assert tuple(config.config_id for config in matching_set.endpoint_configs) == (
        H6_MATCHING_V3_ENDPOINT_CONFIG_IDS
    )
    assert len(matching_set.endpoint_ledgers) == 12
    assert matching_set.endpoint_ledgers[10].source_bank_count == 0
    assert matching_set.endpoint_ledgers[10].source_parameter_count == 0
    assert tuple(record.row.row_id for record in matching_set.matrix_reports) == (
        "PRIMARY",
        "MAP",
        "STRUCTURE",
        "PRIOR",
        "MIXTURE",
        "OBJECTIVE",
        "LATENT",
        "RECOGNITION",
    )
    assert matching_set.parameter_relative_tolerance == 0.01
    assert matching_set.flop_relative_tolerance == 0.05
    assert matching_set.selection_rule == "first_lexicographic_hard_eligible"
    assert len(matching_set.primary_selection.candidates) == 378
    assert matching_set.primary_selection.status == "ELIGIBLE"
    assert matching_set.status == "ELIGIBLE"
    assert matching_set.obligations == ()
    selected = matching_set.primary_selection.selected_candidate
    assert selected is not None
    assert selected.ordinal == 88
    assert selected.allocation_key == (2, 8, 72, 117)
    assert selected.a0_parameter_count == 61_982
    assert selected.endpoint_parameter_count == 61_454
    assert selected.parameter_relative_difference == pytest.approx(0.00851860217482495)
    assert selected.a0_training_flops == 178_715_214
    assert selected.endpoint_training_flops == 187_045_140
    assert selected.flop_relative_difference == pytest.approx(0.04661005525808228)
    assert tuple(
        candidate.allocation_key
        for candidate in matching_set.primary_selection.candidates
        if candidate.hard_eligible
    ) == (
        (2, 8, 72, 117),
        (2, 8, 72, 118),
    )
    assert tuple(
        selection.config_id for selection in matching_set.component_selections
    ) == (
        "h6-a1-ordinary-latent-v1",
        "h6-a2-generic-map-v1",
        "h6-a3-immediate-predecessor-v1",
        "h6-a4-state-only-v1",
        "h6-a5-factorized-fixed-exact-complete-latent-smoothing-v1",
        "h6-a5-structured-fixed-projection-complete-latent-smoothing-v1",
        ("h6-a5-structured-fixed-exact-complete-nolatent-norecognition-v1"),
        "h6-a5-structured-fixed-exact-complete-latent-filtering-v1",
    )
    assert tuple(
        selection.status for selection in matching_set.component_selections
    ) == (
        "INCONCLUSIVE",
        "INCONCLUSIVE",
        "INCONCLUSIVE",
        "INCONCLUSIVE",
        "ELIGIBLE",
        "INCONCLUSIVE",
        "INCONCLUSIVE",
        "ELIGIBLE",
    )

    diagnostics = primary_matching_diagnostics_v3(
        matching_set.primary_selection,
    )
    assert diagnostics == (
        {
            "criterion": "minimum_flop_relative_difference",
            "allocation": (2, 4, 72, 113),
            "parameter_count": 55_382,
            "training_flops": 182_019_828,
            "parameter_relative_difference": pytest.approx(0.10648252718531187),
            "flop_relative_difference": pytest.approx(0.018490949517034404),
            "hard_eligible": False,
        },
        {
            "criterion": "minimum_parameter_relative_difference",
            "allocation": (2, 6, 88, 113),
            "parameter_count": 62_082,
            "training_flops": 213_465_444,
            "parameter_relative_difference": pytest.approx(0.0016133716240198767),
            "flop_relative_difference": pytest.approx(0.19444472142142302),
            "hard_eligible": False,
        },
        {
            "criterion": "minimum_flop_gap_within_parameter_gate",
            "allocation": (2, 8, 72, 117),
            "parameter_count": 61_454,
            "training_flops": 187_045_140,
            "parameter_relative_difference": pytest.approx(0.00851860217482495),
            "flop_relative_difference": pytest.approx(0.04661005525808228),
            "hard_eligible": True,
        },
    )

    with pytest.raises(ValueError, match="tolerance|frozen"):
        dataclasses.replace(
            matching_set,
            parameter_relative_tolerance=0.02,
        )
    with pytest.raises(ValueError, match="first|eligible|selection"):
        dataclasses.replace(
            matching_set.primary_selection,
            selected_candidate_sha256=(
                matching_set.primary_selection.candidates[89].candidate_sha256
            ),
            status="ELIGIBLE",
            obligations=(),
        )

    factorized = matching_set.component_selections[4]
    later_config = _config(
        factorized.config_id,
        CapacityAllocation.create(
            emission_width=96,
            latent_width=16,
            recognition_width=32,
        ),
    )
    later_ledger = analytical_training_flop_ledger_v3(
        endpoint_config=later_config,
        workload=matching_set.workload,
    )
    reference = factorized.reference_config
    reference_ledger = analytical_training_flop_ledger_v3(
        endpoint_config=reference,
        workload=matching_set.workload,
    )
    later_parameter_count = endpoint_parameter_count_v3(later_config)
    reference_parameter_count = endpoint_parameter_count_v3(reference)
    with pytest.raises(ValueError, match="first-lexicographic"):
        dataclasses.replace(
            factorized,
            candidate_count_evaluated=52,
            selected_endpoint_config=later_config,
            selected_ledger=later_ledger,
            parameter_count=later_parameter_count,
            training_flops=later_ledger.total_arithmetic_flops,
            parameter_relative_difference=(
                abs(later_parameter_count - reference_parameter_count)
                / reference_parameter_count
            ),
            flop_relative_difference=(
                abs(
                    later_ledger.total_arithmetic_flops
                    - reference_ledger.total_arithmetic_flops
                )
                / reference_ledger.total_arithmetic_flops
            ),
        )
