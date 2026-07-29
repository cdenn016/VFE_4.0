from __future__ import annotations

import math

import pytest


def test_objective_metric_projection_sums_only_contiguous_exact_terms() -> None:
    from vfe4.training.production_observability import (
        ProductionObservationError,
        project_objective_metrics,
    )

    terms = {
        "expected_log_emission[0]": -3.0,
        "expected_log_emission[1]": -5.0,
        "initial_model_cross_entropy": 0.25,
        "initial_state_cross_entropy": 0.5,
        "model_source_cross_entropy[0]": 0.0,
        "model_source_cross_entropy[1]": 0.75,
        "model_source_kl[0]": 0.0,
        "model_source_kl[1]": 0.5,
        "model_transition_cross_entropy[0]": 0.0,
        "model_transition_cross_entropy[1]": 1.0,
        "state_source_cross_entropy[0]": 0.0,
        "state_source_cross_entropy[1]": 1.25,
        "state_source_kl[0]": 0.0,
        "state_source_kl[1]": 1.0,
        "state_transition_cross_entropy[0]": 0.0,
        "state_transition_cross_entropy[1]": 1.5,
        "continuous_recognition_entropy": 1.25,
        "conditional_source_entropy_estimate": 0.5,
        "joint_recognition_entropy_estimate": 1.75,
        "complete_elbo_numerator": -11.5,
    }
    projected = project_objective_metrics(
        objective_kind="complete_elbo",
        objective_terms=terms,
        complete_elbo_numerator=-11.5,
        complete_elbo_value=-11.5 / 4,
        counted_targets=4,
    )

    assert projected["expected_log_emission"].numerator == -8.0
    assert projected["expected_log_emission"].denominator == 4
    assert projected["expected_log_emission"].value == -2.0
    assert projected["model_source_cross_entropy"].numerator == 0.75
    assert projected["model_source_kl"].numerator == 0.5
    assert projected["model_transition_cross_entropy"].numerator == 1.0
    assert projected["state_source_cross_entropy"].numerator == 1.25
    assert projected["state_source_kl"].numerator == 1.0
    assert projected["state_transition_cross_entropy"].numerator == 1.5
    assert projected["continuous_recognition_entropy"].numerator == 1.25
    assert (
        projected["conditional_source_entropy_estimate"].numerator
        == 0.5
    )
    assert projected["joint_recognition_entropy_estimate"].numerator == 1.75
    assert "estimator_error_bound" not in projected
    assert projected["complete_elbo"].value == -11.5 / 4

    broken = dict(terms)
    broken.pop("model_source_cross_entropy[0]")
    with pytest.raises(ProductionObservationError, match="contiguous"):
        project_objective_metrics(
            objective_kind="complete_elbo",
            objective_terms=broken,
            complete_elbo_numerator=-11.5,
            complete_elbo_value=-11.5 / 4,
            counted_targets=4,
        )


def test_objective_metric_projection_rejects_factor_identity_drift() -> None:
    from vfe4.training.production_observability import (
        ProductionObservationError,
        project_objective_metrics,
    )

    terms = {
        "expected_log_emission[0]": -2.0,
        "initial_model_cross_entropy": 0.25,
        "initial_state_cross_entropy": 0.5,
        "model_source_cross_entropy[0]": 0.75,
        "model_source_kl[0]": 0.5,
        "model_transition_cross_entropy[0]": 1.0,
        "state_source_cross_entropy[0]": 1.25,
        "state_source_kl[0]": 1.0,
        "state_transition_cross_entropy[0]": 1.5,
        "continuous_recognition_entropy": 1.25,
        "conditional_source_entropy_estimate": 0.5,
        "joint_recognition_entropy_estimate": 1.75,
        "complete_elbo_numerator": -5.5,
    }

    bad_joint = dict(terms)
    bad_joint["joint_recognition_entropy_estimate"] = 1.5
    with pytest.raises(ProductionObservationError, match="chain-rule"):
        project_objective_metrics(
            objective_kind="complete_elbo",
            objective_terms=bad_joint,
            complete_elbo_numerator=-5.5,
            complete_elbo_value=-5.5,
            counted_targets=1,
        )

    bad_source_kl = dict(terms)
    bad_source_kl["state_source_kl[0]"] = 0.75
    with pytest.raises(ProductionObservationError, match="source KL"):
        project_objective_metrics(
            objective_kind="complete_elbo",
            objective_terms=bad_source_kl,
            complete_elbo_numerator=-5.5,
            complete_elbo_value=-5.5,
            counted_targets=1,
        )

    bad_complete = dict(terms)
    bad_complete["complete_elbo_numerator"] = -5.25
    with pytest.raises(
        ProductionObservationError,
        match="factor reconstruction",
    ):
        project_objective_metrics(
            objective_kind="complete_elbo",
            objective_terms=bad_complete,
            complete_elbo_numerator=-5.25,
            complete_elbo_value=-5.25,
            counted_targets=1,
        )

    fabricated_bound = dict(terms)
    fabricated_bound["estimator_error_bound"] = 0.0
    with pytest.raises(ProductionObservationError, match="not applicable"):
        project_objective_metrics(
            objective_kind="complete_elbo",
            objective_terms=fabricated_bound,
            complete_elbo_numerator=-5.5,
            complete_elbo_value=-5.5,
            counted_targets=1,
        )


def test_objective_metric_projection_never_substitutes_validation_nll() -> None:
    from vfe4.training.production_observability import (
        project_objective_metrics,
    )

    projected = project_objective_metrics(
        objective_kind="cross_entropy",
        objective_terms={"cross_entropy_value": 9.0},
        complete_elbo_numerator=None,
        complete_elbo_value=None,
        counted_targets=3,
    )
    assert tuple(projected) == ("train_cross_entropy",)
    assert projected["train_cross_entropy"].numerator == 9.0
    assert projected["train_cross_entropy"].denominator == 3
    assert projected["train_cross_entropy"].value == 3.0


def test_phase_timer_records_disjoint_actual_event_durations() -> None:
    from vfe4.training.production_observability import (
        PhaseTimer,
        ProductionObservationError,
    )

    ticks = iter((0, 7, 10, 15, 20, 24, 30, 43, 50, 61))
    synchronizations: list[str] = []
    timer = PhaseTimer(
        monotonic_ns=lambda: next(ticks),
        synchronize=lambda: synchronizations.append("sync"),
    )

    assert timer.run("forward", lambda: "forward-result") == "forward-result"
    assert timer.run("recognition_forward", lambda: "e-forward") == (
        "e-forward"
    )
    assert timer.run("immutable_detached_snapshot", lambda: "snapshot") == (
        "snapshot"
    )
    assert timer.run("recognition_backward", lambda: "backward-result") == (
        "backward-result"
    )
    assert timer.run("model_adamw", lambda: 4) == 4
    with pytest.raises(ProductionObservationError, match="unclassified"):
        timer.run("recognition_adam_proposal", lambda: None)

    observation = timer.observation(
        data_wait_ns=2,
        evaluation_ns=17,
        checkpoint_ns=19,
        wall_ns=100,
    )
    assert observation.forward_ns == 7
    assert observation.backward_ns == 13
    assert observation.inference_ns == 9
    assert observation.update_ns == 11
    assert observation.data_wait_ns == 2
    assert observation.evaluation_ns == 17
    assert observation.checkpoint_ns == 19
    assert observation.wall_ns == 100
    assert len(synchronizations) == 10


def test_production_observations_reject_fabricated_or_nonfinite_values() -> None:
    from vfe4.training.production_observability import (
        MemoryObservation,
        NumericalObservation,
        ProductionObservationError,
        SourceObservation,
    )

    empty_source = SourceObservation(
        entropy_sum=0.0,
        source_row_count=0,
        support_size_sum=0.0,
    )
    with pytest.raises(ProductionObservationError, match="not applicable"):
        _ = empty_source.mean_entropy
    with pytest.raises(ProductionObservationError, match="zero rows"):
        SourceObservation(
            entropy_sum=1.0,
            source_row_count=0,
            support_size_sum=0.0,
        )
    with pytest.raises(ProductionObservationError, match="finite"):
        NumericalObservation(
            minimum_cholesky_pivot=1.0,
            failed_pivots=0,
            condition_estimate=math.inf,
            solve_residual=0.0,
            nonfinite_count=0,
        )
    with pytest.raises(ProductionObservationError, match="HWM"):
        MemoryObservation(
            process_rss_bytes=10,
            process_hwm_bytes=9,
            cuda_allocated_bytes=0,
            cuda_reserved_bytes=0,
            cuda_peak_allocated_bytes=0,
            cuda_peak_reserved_bytes=0,
        )


def test_memory_capture_uses_exact_host_and_cuda_providers() -> None:
    from vfe4.training.production_observability import (
        capture_memory_observation,
    )

    observed = capture_memory_observation(
        host_provider=lambda: (100, 120),
        cuda_provider=lambda: (30, 50, 45, 80),
    )
    assert observed.process_rss_bytes == 100
    assert observed.process_hwm_bytes == 120
    assert observed.cuda_allocated_bytes == 30
    assert observed.cuda_reserved_bytes == 50
    assert observed.cuda_peak_allocated_bytes == 45
    assert observed.cuda_peak_reserved_bytes == 80
