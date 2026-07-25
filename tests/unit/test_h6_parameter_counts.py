from vfe4.training.parameter_counts import (
    A5_REFERENCE_PARAMETER_COUNT,
    AMENDED_EMISSION_WIDTH_CANDIDATES,
    AMENDED_LATENT_WIDTH_CANDIDATES,
    AMENDED_RECOGNITION_WIDTH_CANDIDATES,
    arm_parameter_count,
    h6_a0_parameter_count,
    outcome_blind_feasibility_assessments,
    parent_specific_pooled_prefix_source_prior_parameter_count,
)
from vfe4.training.h6_transformer import (
    H6A0ArchitectureProfile,
    H6CausalTransformer,
)
from vfe4.types import VocabularyIdentity


_SHA = "a" * 64


def test_h6_a0_transformer_formula_matches_live_tensors() -> None:
    profile = H6A0ArchitectureProfile.create()
    assert h6_a0_parameter_count(
        vocabulary_size=profile.vocabulary_size,
        position_capacity=profile.position_capacity,
        hidden_width=profile.hidden_width,
    ) == 61_982
    model = H6CausalTransformer(
        vocabulary=VocabularyIdentity("h6-byte-v1", 258, _SHA),
        profile=profile,
    )
    assert sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    ) == 61_982

    try:
        h6_a0_parameter_count(
            vocabulary_size=258,
            position_capacity=32,
            hidden_width=53,
        )
    except ValueError as exc:
        assert "two equal heads" in str(exc)
    else:
        raise AssertionError("unequal H6 A0 heads must be rejected")


def test_symbolic_parameter_counts_match_the_constructor_inventory() -> None:
    dimensions = {
        "vocabulary_size": 258,
        "horizon": 32,
    }
    assert arm_parameter_count(
        "A5",
        emission_width=64,
        latent_width=16,
        recognition_width=64,
        **dimensions,
    ) == A5_REFERENCE_PARAMETER_COUNT == 63_634
    assert arm_parameter_count(
        "A5",
        emission_width=64,
        latent_width=16,
        recognition_width=64,
        recognition_family="factorized",
        **dimensions,
    ) == 63_378
    assert parent_specific_pooled_prefix_source_prior_parameter_count(
        vocabulary_size=258,
        horizon=32,
        latent_width=8,
        context_width=6,
        gauge_anchored=False,
    ) == 9_036
    assert parent_specific_pooled_prefix_source_prior_parameter_count(
        vocabulary_size=258,
        horizon=32,
        latent_width=8,
        context_width=6,
        gauge_anchored=True,
    ) == 8_588


def test_outcome_blind_witnesses_match_the_amended_width_policy() -> None:
    assert AMENDED_EMISSION_WIDTH_CANDIDATES == (48, 64, 80, 96, 123)
    assert AMENDED_LATENT_WIDTH_CANDIDATES == (2, 8, 16, 24, 32)
    assert AMENDED_RECOGNITION_WIDTH_CANDIDATES == (32, 64, 96)

    assessments = outcome_blind_feasibility_assessments()
    assert len(assessments) == 12
    observed = {
        item.config_id: (item.status, item.parameter_count)
        for item in assessments
    }
    assert observed == {
        "h6-a0-transformer-v2": ("AVAILABLE", 61_982),
        "h6-a1-ordinary-latent-v1": ("AVAILABLE", 63_012),
        "h6-a2-generic-map-v1": ("AVAILABLE", 63_010),
        "h6-a3-immediate-predecessor-v1": ("AVAILABLE", 63_730),
        "h6-a4-state-only-v1": ("AVAILABLE", 63_316),
        "h6-a5-structured-fixed-exact-complete-latent-smoothing-v1": (
            "AVAILABLE",
            63_634,
        ),
        "h6-a5-factorized-fixed-exact-complete-latent-smoothing-v1": (
            "AVAILABLE",
            63_378,
        ),
        (
            "h6-a5-structured-parent-specific-prefix-exact-complete-"
            "latent-smoothing-v2"
        ): (
            "AVAILABLE",
            63_430,
        ),
        "h6-a5-structured-fixed-projection-complete-latent-smoothing-v1": (
            "AVAILABLE",
            63_634,
        ),
        (
            "h6-a5-structured-parent-specific-prefix-exact-emission-"
            "latent-smoothing-v2"
        ): (
            "AVAILABLE",
            63_430,
        ),
        "h6-a5-structured-fixed-exact-complete-nolatent-norecognition-v1": (
            "AVAILABLE",
            63_849,
        ),
        "h6-a5-structured-fixed-exact-complete-latent-filtering-v1": (
            "AVAILABLE",
            63_634,
        ),
    }
    prior = next(
        item
        for item in assessments
        if item.config_id
        == (
            "h6-a5-structured-parent-specific-prefix-exact-complete-"
            "latent-smoothing-v2"
        )
    )
    assert prior.prior_context_width == 6
    assert prior.parameter_count == 63_430
    assert prior.planned_parameter_count is None
    assert prior.obligations == ()
    assert all(
        item.parameter_within_tolerance
        for item in assessments
        if item.status == "AVAILABLE" and item.arm != "A0"
    )
    a0 = next(item for item in assessments if item.arm == "A0")
    assert not a0.parameter_within_tolerance
