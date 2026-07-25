from __future__ import annotations

import hashlib
import inspect
import math
from dataclasses import fields, replace
from pathlib import Path

import pytest
import torch

import vfe4.inference as public_inference
import vfe4.inference.h5_updates as updates
from vfe4.inference.h5_updates import (
    H5_CANDIDATE_DRAFT_DOMAIN,
    AttemptFailureReason,
    AttemptPhase,
    CompletedUpdateAttempt,
    DifferentiableModelState,
    DifferentiableRecognitionState,
    FailedUpdateAttempt,
    H5CandidateDraft,
    H5FaultInjection,
    H5FaultKind,
    UpdateHashRecord,
    canonical_frozen_complement_bytes,
    differentiable_h5_complete_elbo_order_21,
    exact_conjugate_gaussian_e_update,
    exact_gaussian_m_update,
    exact_source_row_update,
    execute_update,
    freeze_candidate,
    propose_generalized_em,
    propose_natural_gradient,
)
from vfe4.numerics.h5_budget import DEFAULT_H5_BUDGET_CONFIG
from vfe4.objective.h5_complete import (
    StaleFactorCacheError,
    evaluate_h5_complete_elbo,
)
from vfe4.types.h5_schema import H5_FACTOR_UNIVERSE, H5_FROZEN_COMPLEMENT_DOMAIN
from vfe4.types.updates import (
    H5_RULE_CONTRACTS,
    H5CandidateSnapshot,
    H5UpdateRule,
    UpdateRequest,
    initial_live,
)
from vfe4.validation.h5_update_spec import build_h5_reference_state


ROOT = Path(__file__).parents[2]
H1_BYTES = (ROOT / "vfe4/validation/fixtures/h1_v1.json").read_bytes()
H5_BYTES = (
    ROOT / "vfe4/validation/fixtures/h5_conditional_update_v1.json"
).read_bytes()


def _reference():
    return build_h5_reference_state(H1_BYTES, H5_BYTES)


def _request(rule: H5UpdateRule) -> UpdateRequest:
    label, variables, parameters, schedule = H5_RULE_CONTRACTS[rule]
    return UpdateRequest(
        "h5-update-request-v1",
        f"test-{rule.value}",
        rule,
        label,
        variables,
        parameters,
        schedule,
    )


class _Evaluator:
    def __init__(self, reference) -> None:
        self.reference = reference
        self.calls: list[object] = []

    def evaluate(self, state, *, frozen_complement_sha256, cache=None):
        self.calls.append(state)
        return evaluate_h5_complete_elbo(
            self.reference,
            state,
            frozen_complement_sha256=frozen_complement_sha256,
            cache=cache,
        )


def _gaussian(snapshot, coordinate_id: str) -> tuple[float, float]:
    item = next(x for x in snapshot.gaussians if x.coordinate_id == coordinate_id)
    return item.mean.values[0], item.variance.values[0]


def _categorical(snapshot, coordinate_id: str) -> tuple[float, ...]:
    item = next(
        x for x in snapshot.categoricals if x.coordinate_id == coordinate_id
    )
    return item.probabilities.values


def _block(snapshot, block_id: str) -> dict[str, tuple[float, ...]]:
    item = next(x for x in snapshot.parameter_blocks if x.block_id == block_id)
    return {name: value.values for name, value in item.values}


def test_task7_public_records_signature_and_exports_are_exact() -> None:
    assert H5_CANDIDATE_DRAFT_DOMAIN == b"vfe4.h5.candidate-draft.v1\x00"
    assert tuple(field.name for field in fields(H5CandidateDraft)) == (
        "schema_version",
        "rule",
        "request_sha256",
        "producer_label",
        "variables",
        "parameters",
        "damping",
        "numerical_diagnostics",
        "recognition",
        "model",
        "candidate_draft_sha256",
    )
    assert tuple(field.name for field in fields(UpdateHashRecord)) == (
        "schema_version",
        "request_sha256",
        "before_live_sha256",
        "before_recognition_sha256",
        "before_model_sha256",
        "before_optimizer_sha256",
        "before_rng_sha256",
        "predecision_live_sha256",
        "predecision_optimizer_sha256",
        "predecision_rng_sha256",
        "candidate_draft_sha256",
        "candidate_sha256",
        "candidate_recognition_sha256",
        "candidate_model_sha256",
        "frozen_complement_sha256",
        "final_live_sha256",
        "final_recognition_sha256",
        "final_model_sha256",
        "final_optimizer_sha256",
        "final_rng_sha256",
    )
    signature = inspect.signature(freeze_candidate)
    assert tuple(signature.parameters) == (
        "reference",
        "live",
        "recognition_working",
        "model_working",
        "request",
        "producer_label",
        "damping",
        "expected_frozen_complement_sha256",
    )
    assert all(
        signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        for name in (
            "request",
            "producer_label",
            "damping",
            "expected_frozen_complement_sha256",
        )
    )
    expected = (
        "H5_CANDIDATE_DRAFT_DOMAIN",
        "AttemptPhase",
        "AttemptFailureReason",
        "DecisionReason",
        "H5FaultKind",
        "H5FaultInjection",
        "H5CandidateDraft",
        "UpdateHashRecord",
        "PartialFactorEvaluation",
        "DeterministicReevaluationRecord",
        "CompletedUpdateAttempt",
        "FailedUpdateAttempt",
        "H5AttemptOutcome",
        "H5TransactionResult",
        "DifferentiableRecognitionState",
        "DifferentiableModelState",
        "canonical_h5_candidate_draft_bytes",
        "exact_conjugate_gaussian_e_update",
        "exact_source_row_update",
        "exact_gaussian_m_update",
        "differentiable_h5_complete_elbo_order_21",
        "propose_generalized_em",
        "propose_natural_gradient",
        "freeze_candidate",
        "canonical_frozen_complement_bytes",
        "execute_update",
    )
    assert updates.__all__ == list(expected)
    assert public_inference.__all__[-len(expected) :] == list(expected)
    assert not any("DraftRejected" in name for name in public_inference.__all__)


def test_three_exact_coordinates_match_displayed_closed_form() -> None:
    reference = _reference()
    live = initial_live(reference)

    z0 = exact_conjugate_gaussian_e_update(
        reference, live, _request(H5UpdateRule.EXACT_Z0)
    )
    covariance = torch.tensor(
        [[0.8, 0.18], [0.18, 0.65]], dtype=torch.float64
    )
    precision = torch.linalg.inv(covariance)
    information = precision @ torch.tensor((0.2, -0.15), dtype=torch.float64)
    gamma = _categorical(live.recognition, "q[model_source_b2]")
    beta0 = _categorical(live.recognition, "q[source_row_a2]")
    beta1 = _categorical(live.recognition, "q[state_source_a2_b1]")
    w20 = gamma[0] * beta0[0] + gamma[1] * beta1[0]
    transition = _block(live.model, "theta[state_transition_2]")
    shared = _block(live.model, "theta[shared_decoder_transition]")["s"][0]
    z1, _ = _gaussian(live.recognition, "q[z1]")
    z2, _ = _gaussian(live.recognition, "q[z2]")
    m0, _ = _gaussian(live.recognition, "q[m0]")
    m1, _ = _gaussian(live.recognition, "q[m1]")
    m2, _ = _gaussian(live.recognition, "q[m2]")
    J = float(precision[0, 0]) + 1.25**2 / 0.37 + w20 * 0.8**2 / 0.48
    h = (
        float(information[0])
        - float(precision[0, 1]) * m0
        + 1.25 * (z1 - 0.45 * m1 - (-0.12)) / 0.37
        + w20
        * 0.8
        * (z2 - (transition["B_base"][0] + shared) * m2 - transition["c"][0])
        / 0.48
    )
    assert _gaussian(z0.recognition, "q[z0]") == pytest.approx(
        (h / J, 1.0 / J), rel=0.0, abs=3e-15
    )

    source = exact_source_row_update(
        reference, live, _request(H5UpdateRule.EXACT_SOURCE_ROW_A2)
    )
    logits = []
    for a, alpha in enumerate((0.8, 0.64)):
        parent_mean, parent_variance = _gaussian(
            live.recognition, f"q[z{a}]"
        )
        residual = (
            z2
            - alpha * parent_mean
            - (transition["B_base"][0] + shared) * m2
            - transition["c"][0]
        )
        _, z2_variance = _gaussian(live.recognition, "q[z2]")
        _, m2_variance = _gaussian(live.recognition, "q[m2]")
        ell = -0.5 * (
            math.log(2.0 * math.pi * transition["R"][0])
            + (
                z2_variance
                + alpha * alpha * parent_variance
                + (transition["B_base"][0] + shared) ** 2 * m2_variance
                + residual * residual
            )
            / transition["R"][0]
        )
        logits.append(math.log((0.55, 0.45)[a]) + ell)
    denominator = math.fsum(math.exp(x - max(logits)) for x in logits)
    expected = tuple(math.exp(x - max(logits)) / denominator for x in logits)
    assert _categorical(source.recognition, "q[source_row_a2]") == pytest.approx(
        expected, rel=0.0, abs=3e-15
    )

    m_step = exact_gaussian_m_update(
        reference, live, _request(H5UpdateRule.EXACT_STATE_TRANSITION_2_M)
    )
    m_values = _block(m_step.model, "theta[state_transition_2]")
    assert set(m_values) == {"alpha_0", "alpha_1", "B_base", "c", "R"}
    assert m_values["R"][0] > 0.0
    assert m_step.recognition.state_sha256 == live.recognition.state_sha256
    assert m_step.recognition is not live.recognition
    assert m_step.numerical_diagnostics[0][0] == "G_condition_number"
    assert m_step.numerical_diagnostics[0][1] >= 1.0


def test_differentiable_objective_matches_complete_objective_and_scopes_autograd() -> None:
    reference = _reference()
    live = initial_live(reference)
    request = _request(H5UpdateRule.NATURAL_GRADIENT_Z1)
    mean = torch.tensor(
        _gaussian(live.recognition, "q[z1]")[0],
        dtype=torch.float64,
        requires_grad=True,
    )
    recognition = DifferentiableRecognitionState(
        request.variables, {"q[z1]": mean}, {}, {}
    )
    model = DifferentiableModelState((), {})
    value = differentiable_h5_complete_elbo_order_21(
        reference, live, recognition, model
    )
    complement = hashlib.sha256(
        H5_FROZEN_COMPLEMENT_DOMAIN
        + canonical_frozen_complement_bytes(reference, live, request)
    ).hexdigest()
    expected = evaluate_h5_complete_elbo(
        reference, live, frozen_complement_sha256=complement
    ).terms.complete_elbo
    assert value.detach().item() == pytest.approx(expected, rel=0.0, abs=3e-12)
    (gradient,) = torch.autograd.grad(value, (mean,))
    assert torch.isfinite(gradient)
    assert live.state_sha256 == initial_live(reference).state_sha256


def test_valid_freeze_has_defensive_draft_and_distinct_final_hashes() -> None:
    reference = _reference()
    live = initial_live(reference)
    request = _request(H5UpdateRule.NATURAL_GRADIENT_Z1)
    mean = torch.tensor(0.125, dtype=torch.float64, requires_grad=True)
    recognition = DifferentiableRecognitionState(
        request.variables, {"q[z1]": mean}, {}, {}
    )
    model = DifferentiableModelState((), {})
    expected = hashlib.sha256(
        H5_FROZEN_COMPLEMENT_DOMAIN
        + canonical_frozen_complement_bytes(reference, live, request)
    ).hexdigest()
    candidate = freeze_candidate(
        reference,
        live,
        recognition,
        model,
        request=request,
        producer_label=request.requested_label,
        damping=64.0,
        expected_frozen_complement_sha256=expected,
    )
    assert isinstance(candidate, H5CandidateSnapshot)
    assert candidate.recognition is not live.recognition
    assert _gaussian(candidate.recognition, "q[z1]")[0] == 0.125
    with pytest.raises(ValueError, match="frozen complement"):
        freeze_candidate(
            reference,
            live,
            recognition,
            model,
            request=request,
            producer_label=request.requested_label,
            damping=64.0,
            expected_frozen_complement_sha256="00" * 32,
        )


@pytest.mark.parametrize(
    "rule",
    (
        H5UpdateRule.EXACT_Z0,
        H5UpdateRule.EXACT_SOURCE_ROW_A2,
        H5UpdateRule.EXACT_STATE_TRANSITION_2_M,
        H5UpdateRule.GENERALIZED_EM_EMISSION_1,
        H5UpdateRule.NATURAL_GRADIENT_Z1,
    ),
)
def test_execute_update_is_transactional_and_dependency_complete(rule) -> None:
    reference = _reference()
    live = initial_live(reference)
    evaluator = _Evaluator(reference)
    result = execute_update(
        reference,
        live,
        _request(rule),
        evaluator,
        DEFAULT_H5_BUDGET_CONFIG,
    )
    assert isinstance(result.outcome, CompletedUpdateAttempt)
    attempt = result.outcome
    assert attempt.expected_affected_factor_ids == attempt.observed_affected_factor_ids
    assert attempt.missing_factor_ids == attempt.extra_factor_ids == ()
    assert attempt.hashes.candidate_draft_sha256 is not None
    assert attempt.hashes.candidate_sha256 is not None
    assert (
        attempt.hashes.candidate_draft_sha256
        != attempt.hashes.candidate_sha256
    )
    if attempt.accepted:
        assert result.live is not live
    else:
        assert result.live is live
        assert attempt.hashes.final_live_sha256 == attempt.hashes.before_live_sha256
    if rule is H5UpdateRule.GENERALIZED_EM_EMISSION_1:
        assert attempt.accepted
        assert attempt.line_search_step is not None
    if rule is H5UpdateRule.NATURAL_GRADIENT_Z1:
        assert not attempt.accepted
        assert attempt.delta_elbo < -attempt.allowance.epsilon_delta


def test_mislabel_failure_retains_only_draft_candidate_hash_and_rolls_back(
    monkeypatch,
) -> None:
    reference = _reference()
    live = initial_live(reference)
    evaluator = _Evaluator(reference)
    final_construction_calls = 0
    original_candidate_type = updates.H5CandidateSnapshot

    def tracked_candidate(*args, **kwargs):
        nonlocal final_construction_calls
        final_construction_calls += 1
        return original_candidate_type(*args, **kwargs)

    monkeypatch.setattr(updates, "H5CandidateSnapshot", tracked_candidate)
    result = execute_update(
        reference,
        live,
        _request(H5UpdateRule.NATURAL_GRADIENT_Z1),
        evaluator,
        DEFAULT_H5_BUDGET_CONFIG,
        fault_injection=H5FaultInjection(
            H5FaultKind.MISLABEL_NATURAL_AS_EXACT, None, None
        ),
    )
    assert result.live is live
    assert isinstance(result.outcome, FailedUpdateAttempt)
    failure = result.outcome
    assert failure.phase is AttemptPhase.FREEZE
    assert failure.reason is AttemptFailureReason.LABEL_PROVENANCE_MISMATCH
    assert failure.hashes.candidate_draft_sha256 is not None
    assert failure.hashes.candidate_recognition_sha256 is not None
    assert failure.hashes.candidate_model_sha256 is not None
    assert failure.hashes.candidate_sha256 is None
    assert failure.hashes.predecision_live_sha256 is None
    assert failure.hashes.predecision_optimizer_sha256 is None
    assert failure.hashes.predecision_rng_sha256 is None
    assert failure.hashes.final_live_sha256 == failure.hashes.before_live_sha256
    assert failure.before is not None
    assert len(evaluator.calls) == 1
    assert final_construction_calls == 1


@pytest.mark.parametrize(
    ("rule", "fault", "phase", "reason", "missing"),
    (
        (
            H5UpdateRule.EXACT_Z0,
            H5FaultKind.OMIT_CHILD,
            AttemptPhase.AFTER_EVALUATION,
            AttemptFailureReason.FACTOR_COVERAGE_MISMATCH,
            ("state_transition[2]",),
        ),
        (
            H5UpdateRule.GENERALIZED_EM_EMISSION_1,
            H5FaultKind.OMIT_EMISSION,
            AttemptPhase.AFTER_EVALUATION,
            AttemptFailureReason.FACTOR_COVERAGE_MISMATCH,
            ("emission[1]",),
        ),
        (
            H5UpdateRule.GENERALIZED_EM_EMISSION_1,
            H5FaultKind.FORCE_UNRESOLVED_GEM_ACCEPT,
            AttemptPhase.DECISION,
            AttemptFailureReason.DECISION_POLICY_VIOLATION,
            (),
        ),
        (
            H5UpdateRule.NATURAL_GRADIENT_Z1,
            H5FaultKind.MUTATE_REJECTED_LIVE_AND_RNG,
            AttemptPhase.COMMIT_OR_ROLLBACK,
            AttemptFailureReason.ROLLBACK_HASH_MISMATCH,
            (),
        ),
    ),
)
def test_faults_fail_at_their_first_typed_phase(
    rule, fault, phase, reason, missing
) -> None:
    reference = _reference()
    live = initial_live(reference)
    result = execute_update(
        reference,
        live,
        _request(rule),
        _Evaluator(reference),
        DEFAULT_H5_BUDGET_CONFIG,
        fault_injection=H5FaultInjection(fault, None, None),
    )
    assert result.live is live
    assert isinstance(result.outcome, FailedUpdateAttempt)
    assert result.outcome.phase is phase
    assert result.outcome.reason is reason
    assert result.outcome.missing_factor_ids == missing
    if phase in (AttemptPhase.DECISION, AttemptPhase.COMMIT_OR_ROLLBACK):
        assert result.outcome.decision_delta_elbo is not None
        assert result.outcome.decision_epsilon_delta is not None
        assert result.outcome.attempted_accept is not None
    if fault is H5FaultKind.MUTATE_REJECTED_LIVE_AND_RNG:
        assert (
            result.outcome.hashes.final_live_sha256
            != result.outcome.hashes.before_live_sha256
        )
        assert (
            result.outcome.hashes.final_rng_sha256
            != result.outcome.hashes.before_rng_sha256
        )


def test_changed_input_equal_value_and_same_input_changed_value_are_separate() -> None:
    reference = _reference()
    live = initial_live(reference)
    request = _request(H5UpdateRule.EXACT_STATE_TRANSITION_2_M)
    reflected = execute_update(
        reference,
        live,
        request,
        _Evaluator(reference),
        DEFAULT_H5_BUDGET_CONFIG,
        fault_injection=H5FaultInjection(
            H5FaultKind.CHANGE_INPUT_KEEP_VALUE, "state_transition[2]", None
        ),
    )
    assert isinstance(reflected.outcome, CompletedUpdateAttempt)
    assert reflected.outcome.observed_affected_factor_ids == (
        "state_transition[2]",
    )
    assert "state_transition[2]" not in reflected.outcome.value_changed_factor_ids

    corrupted_request = _request(H5UpdateRule.GENERALIZED_EM_EMISSION_1)
    corrupted = execute_update(
        reference,
        live,
        corrupted_request,
        _Evaluator(reference),
        DEFAULT_H5_BUDGET_CONFIG,
        fault_injection=H5FaultInjection(
            H5FaultKind.CHANGE_VALUE_KEEP_INPUT,
            "state_transition[2]",
            1.0e-6,
        ),
    )
    assert corrupted.live is live
    assert isinstance(corrupted.outcome, FailedUpdateAttempt)
    assert corrupted.outcome.phase is AttemptPhase.AFTER_EVALUATION
    assert (
        corrupted.outcome.reason
        is AttemptFailureReason.DETERMINISTIC_REEVALUATION_MISMATCH
    )
    assert "state_transition[2]" not in corrupted.outcome.observed_affected_factor_ids
    assert "state_transition[2]" in corrupted.outcome.value_changed_factor_ids
    assert corrupted.outcome.deterministic_reevaluation is not None
    recheck = corrupted.outcome.deterministic_reevaluation
    before_record = next(
        item
        for item in corrupted.outcome.before.factor_records
        if item.factor_id == "state_transition[2]"
    )
    assert recheck.input_sha256 == before_record.input_hash.input_sha256
    assert recheck.recomputed_value_order_21.hex() == before_record.value_order_21.hex()
    assert recheck.recomputed_value_order_17.hex() == before_record.value_order_17.hex()


@pytest.mark.parametrize(
    ("rule", "proposal"),
    (
        (
            H5UpdateRule.GENERALIZED_EM_EMISSION_1,
            lambda reference, live, request: propose_generalized_em(
                reference, live, request, 1.0
            ),
        ),
        (
            H5UpdateRule.NATURAL_GRADIENT_Z1,
            lambda reference, live, request: propose_natural_gradient(
                reference, live, request, 64.0
            ),
        ),
    ),
)
def test_autograd_is_blocked_until_differentiable_and_task6_objectives_agree(
    monkeypatch, rule, proposal
) -> None:
    reference = _reference()
    live = initial_live(reference)
    request = _request(rule)
    original_objective = updates.differentiable_h5_complete_elbo_order_21
    original_grad = torch.autograd.grad
    grad_calls = 0

    def corrupted_objective(*args, **kwargs):
        return original_objective(*args, **kwargs) + 1.0

    def tracked_grad(*args, **kwargs):
        nonlocal grad_calls
        grad_calls += 1
        return original_grad(*args, **kwargs)

    monkeypatch.setattr(
        updates, "differentiable_h5_complete_elbo_order_21", corrupted_objective
    )
    monkeypatch.setattr(torch.autograd, "grad", tracked_grad)
    with pytest.raises(ValueError, match="differentiable.*Task 6"):
        proposal(reference, live, request)
    assert grad_calls == 0


class _StaleCandidateEvaluator(_Evaluator):
    def evaluate(self, state, *, frozen_complement_sha256, cache=None):
        if self.calls:
            raise StaleFactorCacheError("emission[1]")
        return super().evaluate(
            state,
            frozen_complement_sha256=frozen_complement_sha256,
            cache=cache,
        )


def test_candidate_cache_failure_is_after_evaluation_not_proposal() -> None:
    reference = _reference()
    live = initial_live(reference)
    result = execute_update(
        reference,
        live,
        _request(H5UpdateRule.GENERALIZED_EM_EMISSION_1),
        _StaleCandidateEvaluator(reference),
        DEFAULT_H5_BUDGET_CONFIG,
    )
    assert result.live is live
    assert isinstance(result.outcome, FailedUpdateAttempt)
    assert result.outcome.phase is AttemptPhase.AFTER_EVALUATION
    assert result.outcome.reason is AttemptFailureReason.STALE_CACHE
    assert result.outcome.hashes.candidate_draft_sha256 is not None
    assert result.outcome.hashes.candidate_sha256 is not None
    assert result.outcome.hashes.predecision_live_sha256 is not None


def test_generic_freeze_failure_has_no_candidate_hash_and_is_not_proposal(
    monkeypatch,
) -> None:
    reference = _reference()
    live = initial_live(reference)

    def invalid_freeze(*args, **kwargs):
        raise ValueError("frozen complement SHA-256 mismatch")

    monkeypatch.setattr(updates, "_freeze_with_draft", invalid_freeze)
    result = execute_update(
        reference,
        live,
        _request(H5UpdateRule.EXACT_Z0),
        _Evaluator(reference),
        DEFAULT_H5_BUDGET_CONFIG,
    )
    assert result.live is live
    assert isinstance(result.outcome, FailedUpdateAttempt)
    assert result.outcome.phase is AttemptPhase.FREEZE
    assert result.outcome.reason is AttemptFailureReason.NONFINITE_OR_INVALID_CANDIDATE
    assert result.outcome.hashes.candidate_draft_sha256 is None
    assert result.outcome.hashes.candidate_sha256 is None
    assert result.outcome.hashes.predecision_live_sha256 is None


def test_completed_attempt_requires_disjoint_exact_cache_partition_and_bound_damping() -> None:
    reference = _reference()
    live = initial_live(reference)
    exact = execute_update(
        reference,
        live,
        _request(H5UpdateRule.EXACT_Z0),
        _Evaluator(reference),
        DEFAULT_H5_BUDGET_CONFIG,
    ).outcome
    assert isinstance(exact, CompletedUpdateAttempt)
    overlap_id = exact.reevaluated_factor_ids[0]
    overlapping_reused = tuple(
        factor_id
        for factor_id in H5_FACTOR_UNIVERSE
        if factor_id in set(exact.reused_factor_ids) | {overlap_id}
    )
    with pytest.raises(ValueError, match="disjoint"):
        replace(exact, reused_factor_ids=overlapping_reused)

    gem_request = _request(H5UpdateRule.GENERALIZED_EM_EMISSION_1)
    gem = execute_update(
        reference,
        live,
        gem_request,
        _Evaluator(reference),
        DEFAULT_H5_BUDGET_CONFIG,
    ).outcome
    assert isinstance(gem, CompletedUpdateAttempt)
    wrong_damping = gem_request.damping_schedule[
        1 if gem.line_search_step == 0 else 0
    ]
    with pytest.raises(ValueError, match="line-search.*damping"):
        replace(gem, damping=wrong_damping)


class _DelayResolvedGEMEvaluator(_Evaluator):
    def __init__(self, reference, unresolved_candidates: int) -> None:
        super().__init__(reference)
        self.unresolved_candidates = unresolved_candidates
        self.before_evaluation = None
        self.candidate_calls = 0

    def evaluate(self, state, *, frozen_complement_sha256, cache=None):
        if self.before_evaluation is None:
            result = super().evaluate(
                state,
                frozen_complement_sha256=frozen_complement_sha256,
                cache=cache,
            )
            self.before_evaluation = result
            return result
        self.calls.append(state)
        self.candidate_calls += 1
        if self.candidate_calls <= self.unresolved_candidates:
            return self.before_evaluation
        return evaluate_h5_complete_elbo(
            self.reference,
            state,
            frozen_complement_sha256=frozen_complement_sha256,
            cache=cache,
        )


def test_gem_records_the_first_resolved_damping_after_all_earlier_unresolved() -> None:
    reference = _reference()
    live = initial_live(reference)
    request = _request(H5UpdateRule.GENERALIZED_EM_EMISSION_1)
    evaluator = _DelayResolvedGEMEvaluator(reference, unresolved_candidates=2)
    result = execute_update(
        reference,
        live,
        request,
        evaluator,
        DEFAULT_H5_BUDGET_CONFIG,
    )
    assert isinstance(result.outcome, CompletedUpdateAttempt)
    assert result.outcome.accepted
    assert evaluator.candidate_calls == 3
    assert result.outcome.line_search_step == 2
    assert result.outcome.damping.hex() == request.damping_schedule[2].hex()


def test_rectangular_state_and_model_updates_match_independent_oracle() -> None:
    import importlib

    import numpy as np

    fixture_module = importlib.import_module(
        "vfe4.validation.h2_h5_rectangular_fixture"
    )
    rectangular_oracle = importlib.import_module(
        "verification.numpy_oracles.h2_h5_rectangular"
    )
    rectangular_gate = importlib.import_module(
        "verification.h2_h5_rectangular_gate"
    )

    fixture = fixture_module.load_h2_h5_rectangular_fixture()
    oracle = rectangular_oracle.evaluate_rectangular_update_oracle(
        fixture, time_index=2
    )
    result = rectangular_gate.evaluate_h2_h5_rectangular_gate(
        fixture, time_index=2
    )
    forged_offsets = tuple(
        (
            (row[0] + 0.125,) + row[1:]
            if index == 1
            else row
        )
        for index, row in enumerate(fixture.state_offsets)
    )
    forged = replace(fixture, state_offsets=forged_offsets)
    with pytest.raises(ValueError, match="frozen rectangular C5"):
        rectangular_gate.evaluate_h2_h5_rectangular_gate(
            forged, time_index=2
        )

    assert result.fixture_raw_sha256 == fixture.raw_sha256
    assert result.fixture_canonical_sha256 == fixture.canonical_sha256
    assert result.oracle_report_sha256 == oracle.report_sha256
    for production, independent, shape in (
        (result.production_state_precision, oracle.state_precision, (2, 2)),
        (result.production_state_natural, oracle.state_natural, (2,)),
        (result.production_state_solution, oracle.state_solution, (2,)),
        (result.production_model_precision, oracle.model_precision, (3, 3)),
        (result.production_model_natural, oracle.model_natural, (3,)),
        (result.production_model_solution, oracle.model_solution, (3,)),
    ):
        actual = np.asarray(production)
        expected = np.asarray(independent)
        assert actual.shape == shape
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=5.0e-13)
    assert result.information_transpose_rejected
    assert result.state_update_transpose_rejected
    assert result.model_update_transpose_rejected
    assert result.state_minimum_witness_passed
    assert result.model_minimum_witness_passed
    for channel in ("state", "model"):
        production_probe = getattr(
            result, f"production_{channel}_probe_objective"
        )
        production_solved = getattr(
            result, f"production_{channel}_solved_objective"
        )
        production_gap = getattr(
            result, f"production_{channel}_completion_square_gap"
        )
        assert production_probe > production_solved
        assert production_gap > 0.0
        assert production_probe - production_solved == pytest.approx(
            production_gap,
            rel=5.0e-13,
            abs=5.0e-13,
        )
        assert (
            getattr(
                result,
                f"production_{channel}_solution_gradient_max_abs",
            )
            <= 5.0e-13
        )
        for suffix in (
            "probe_objective",
            "solved_objective",
            "completion_square_gap",
            "solution_gradient_max_abs",
        ):
            assert getattr(
                result, f"production_{channel}_{suffix}"
            ) == pytest.approx(
                getattr(oracle, f"{channel}_{suffix}"),
                rel=5.0e-13,
                abs=5.0e-13,
            )
    assert result.maximum_absolute_error <= 5.0e-13
    assert result.passed
