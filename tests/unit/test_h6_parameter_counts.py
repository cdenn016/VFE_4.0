from vfe4.training.parameter_counts import (
    A5_REFERENCE_PARAMETER_COUNT,
    AMENDED_EMISSION_WIDTH_CANDIDATES,
    AMENDED_LATENT_WIDTH_CANDIDATES,
    AMENDED_RECOGNITION_WIDTH_CANDIDATES,
    arm_parameter_count,
    outcome_blind_feasibility_assessments,
    prefix_conditioned_source_prior_parameter_count,
)


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
    assert prefix_conditioned_source_prior_parameter_count(
        vocabulary_size=258,
        horizon=32,
        latent_width=8,
        context_width=6,
        gauge_anchored=False,
    ) == 9_036
    assert prefix_conditioned_source_prior_parameter_count(
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
        "h6-a0-ar-v1": ("AVAILABLE", 63_849),
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
        "h6-a5-structured-prefix-exact-complete-latent-smoothing-v1": (
            "AVAILABLE",
            63_430,
        ),
        "h6-a5-structured-fixed-projection-complete-latent-smoothing-v1": (
            "AVAILABLE",
            63_634,
        ),
        "h6-a5-structured-fixed-exact-emission-latent-smoothing-v1": (
            "AVAILABLE",
            63_634,
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
        == "h6-a5-structured-prefix-exact-complete-latent-smoothing-v1"
    )
    assert prior.prior_context_width == 6
    assert prior.parameter_count == 63_430
    assert prior.planned_parameter_count is None
    assert prior.obligations == ()
    assert all(
        item.parameter_within_tolerance
        for item in assessments
        if item.status == "AVAILABLE"
    )
