from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import pytest
import torch

import vfe4.generative.pushforward as generative_pushforward
import vfe4.objective.h7_covariance as covariance
import vfe4.recognition.pushforward as recognition_pushforward
from vfe4.geometry.group_action import borrow_h7_action
from vfe4.objective.language_elbo import require_h7_complete_factor_trace
from vfe4.types.h6 import FrozenTensorSnapshot, H6FactorTerm, H6LanguageElboTerms
from vfe4.types.h7 import (
    H7AllowanceContribution,
    H7BudgetCategory,
    H7BudgetRecord,
    H7CompleteLawSnapshot,
    H7IndependentH1EvidenceRecord,
    H7LawPairSnapshot,
    H7OperandRecord,
)
from vfe4.validation.h7_fixture import (
    H1_FIXTURE_RAW_SHA256,
    H7_FIXTURE_PATH,
    adapt_optional_h1_fixture_bytes,
    h7_scalar_trial_specs,
    parse_h7_fixture_bytes,
)


_FACTOR_SLOTS = (
    ("initial", 0),
    ("model_source", 1),
    ("model_transition", 1),
    ("state_source", 1),
    ("state_transition", 1),
    ("emission", 1),
    ("entropy", 1),
    ("model_source", 2),
    ("model_transition", 2),
    ("state_source", 2),
    ("state_transition", 2),
    ("emission", 2),
    ("entropy", 2),
)
_SIGNED_VALUES = (
    -0.31,
    -0.07,
    -0.11,
    -0.05,
    -0.09,
    -0.83,
    0.24,
    -0.08,
    -0.12,
    -0.06,
    -0.10,
    -0.77,
    0.21,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _complete_trace(
    prefix: str,
    *,
    entropy_shift: float = 0.0,
):
    values = list(_SIGNED_VALUES)
    values[5] -= entropy_shift
    values[6] += entropy_shift
    terms = tuple(
        H6FactorTerm(
            receiver_t=receiver_t,
            partition=partition,
            factor_identity_sha256=_sha(
                f"{prefix}:{partition}:{receiver_t}"
            ),
            value=FrozenTensorSnapshot.capture(
                torch.tensor(value, dtype=torch.float64)
            ),
        )
        for (partition, receiver_t), value in zip(
            _FACTOR_SLOTS,
            values,
            strict=True,
        )
    )
    total = terms[0].value.value()
    for term in terms[1:]:
        total = total + term.value.value()
    return require_h7_complete_factor_trace(
        H6LanguageElboTerms.create(
            horizon=2,
            ordered_factor_terms=terms,
            total_language_elbo=total,
        )
    )


def _budget(
    invariant_id: str,
    category: H7BudgetCategory,
    *,
    allowance: float,
    operand_roles: tuple[str, ...] = ("original", "transformed"),
) -> H7BudgetRecord:
    operands = tuple(
        H7OperandRecord.create(
            operand_id=f"{invariant_id}:{role}",
            category=category,
            role=role,
            dtype="float64",
            shape=(),
            value_sha256=_sha(f"{invariant_id}:{role}:value"),
            scale=1.0,
            condition_number=1.0,
            normalization=1.0,
            oracle_value=None,
        )
        for role in operand_roles
    )
    contribution = H7AllowanceContribution.create(
        kind="operation_rounding",
        operation_id=f"{invariant_id}:comparison",
        operation_kind="pair_comparison",
        operation_count=1,
        quadrature_order=None,
        unit_allowance=allowance,
        value=allowance,
    )
    return H7BudgetRecord.create(
        invariant_id=invariant_id,
        category=category,
        operands=operands,
        contributions=(contribution,),
        comparison_normalization=1.0,
        total_allowance=allowance,
    )


def _density_roles(probe) -> tuple[str, ...]:
    if ".global" in probe.component_id:
        return ("p", "q", "log_ratio")
    if probe.component_id.startswith("p."):
        return ("p",)
    if probe.component_id.startswith("q."):
        return ("q",)
    raise AssertionError("test probe lacks a density role")


def _objective_budgets(
    probes,
    *,
    include_matrix_scorers: bool,
    include_scalar_evidence: bool,
) -> dict[str, H7BudgetRecord]:
    declarations: dict[
        str,
        tuple[H7BudgetCategory, float],
    ] = {
        "K0_joint_z0_m0": ("local_term", 2.0e-12),
        **{
            term_id: ("local_term", 3.0e-12 + index * 1.0e-13)
            for index, term_id in enumerate(
                covariance.H7_COMPLETE_LOCAL_TERM_IDS
            )
        },
        covariance.H7_COMPLETE_LOCAL_INVARIANT_ID: (
            "complete_objective",
            5.0e-11,
        ),
        covariance.H7_COMPLETE_MONOLITHIC_INVARIANT_ID: (
            "complete_objective",
            7.0e-11,
        ),
        covariance.H7_POINTWISE_P_SHIFT_INVARIANT_ID: (
            "density",
            2.0e-9,
        ),
        covariance.H7_POINTWISE_Q_SHIFT_INVARIANT_ID: (
            "density",
            3.0e-9,
        ),
        covariance.H7_POINTWISE_LOG_RATIO_INVARIANT_ID: (
            "density",
            4.0e-9,
        ),
        covariance.H7_ENTROPY_SHIFT_INVARIANT_ID: (
            "density",
            6.0e-11,
        ),
    }
    if include_matrix_scorers:
        for index, invariant_id in enumerate(
            covariance.H7_MATRIX_SCORER_RESIDUAL_IDS
        ):
            declarations[invariant_id] = (
                "vector",
                8.0e-11 + index * 1.0e-12,
            )
    if include_scalar_evidence:
        declarations[covariance.H7_SCALAR_EVIDENCE_INVARIANT_ID] = (
            "complete_objective",
            9.0e-11,
        )
        declarations[covariance.H7_SCALAR_POSTERIOR_KL_INVARIANT_ID] = (
            "complete_objective",
            1.1e-10,
        )
    probe_allowances = {
        "p": 1.2e-9,
        "q": 1.4e-9,
        "log_ratio": 1.8e-9,
    }
    for probe_index, probe in enumerate(probes):
        for role in _density_roles(probe):
            declarations[
                f"density_probe.{probe.probe_sha256}.{role}"
            ] = (
                "density",
                probe_allowances[role] + probe_index * 1.0e-14,
            )
    return {
        invariant_id: _budget(
            invariant_id,
            category,
            allowance=allowance,
        )
        for invariant_id, (category, allowance) in declarations.items()
    }


def _scalar_law_pair():
    fixture_path = (
        Path(__file__).parents[2]
        / "vfe4"
        / "validation"
        / "fixtures"
        / "h1_v1.json"
    )
    original = adapt_optional_h1_fixture_bytes(
        fixture_path.read_bytes(),
        required_scalar_trials=(
            "scalar-base-transformed",
            "scalar-internal-transformed",
        ),
    )
    assert original is not None
    action = h7_scalar_trial_specs()[0].action
    borrowed_action = borrow_h7_action(
        tuple(item.value() for item in action.elements),
        kind=action.kind,
        dimension=1,
    )
    transformed_generative = generative_pushforward.freeze_h7_generative(
        generative_pushforward._pushforward_h7_generative_snapshot(
            original.generative,
            borrowed_action,
        ),
        action=borrowed_action,
    )
    transformed_recognition = recognition_pushforward.freeze_h7_recognition(
        recognition_pushforward._pushforward_h7_recognition_snapshot(
            original.recognition,
            borrowed_action,
        )
    )
    transformed = H7CompleteLawSnapshot.create(
        fixture_id="h1-v1",
        generative=transformed_generative,
        recognition=transformed_recognition,
        raw_fixture_sha256=original.raw_fixture_sha256,
        scalar_probe_set=original.scalar_probe_set,
    )
    return original, transformed, action


def test_complete_h7_objective_binds_trace_probe_and_scalar_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original, transformed, action = _scalar_law_pair()
    original_trace = _complete_trace("original")
    expected_entropy_shift = covariance._global_log_jacobian(action)
    assert expected_entropy_shift > 0.0
    transformed_trace = _complete_trace(
        "transformed",
        entropy_shift=expected_entropy_shift,
    )
    probes = tuple(
        pair
        for pair in original.scalar_probe_set.probe_pairs
        if pair.action_sha256 == action.action_sha256
    )
    budgets = _objective_budgets(
        probes,
        include_matrix_scorers=False,
        include_scalar_evidence=True,
    )
    # Wiring-only synthetic values: Task 6 supplies the scientific oracle and
    # real operand-local budgets.
    evidence = H7IndependentH1EvidenceRecord.create(
        fixture_id="h1-v1",
        raw_fixture_sha256=H1_FIXTURE_RAW_SHA256,
        action_sha256=action.action_sha256,
        normalization_identity_sha256=(
            covariance.H7_INDEPENDENT_H1_NORMALIZATION_IDENTITY_SHA256
        ),
        producer_identity_sha256=(
            covariance.H7_INDEPENDENT_H1_PRODUCER_IDENTITY_SHA256
        ),
        original_log_evidence=original_trace.total_value + 0.25,
        transformed_log_evidence=transformed_trace.total_value + 0.25,
        original_posterior_kl=0.25,
        transformed_posterior_kl=0.25,
    )
    quadrature_orders: list[int] = []
    production_quadrature = covariance.probabilists_gauss_hermite

    def recording_quadrature(order: int, *, dtype: torch.dtype):
        quadrature_orders.append(order)
        return production_quadrature(order, dtype=dtype)

    monkeypatch.setattr(
        covariance,
        "probabilists_gauss_hermite",
        recording_quadrature,
    )
    result = covariance.evaluate_h7_complete_covariance(
        original,
        transformed,
        action,
        original_factor_trace=original_trace,
        transformed_factor_trace=transformed_trace,
        density_probe_pairs=None,
        quadrature_orders=(41, 51),
        budgets_by_invariant=budgets,
        scalar_evidence=evidence,
    )

    assert result.original_factor_trace_sha256 == original_trace.trace_sha256
    assert result.original_ordered_factor_ids == original_trace.ordered_factor_ids
    assert (
        result.original_ordered_factor_values
        == original_trace.ordered_factor_values
    )
    assert result.original_complete_local_value == original_trace.total_value
    assert result.initial_joint_kl.original_value == _SIGNED_VALUES[0]
    assert any(item.original_value < 0.0 for item in result.local_terms)
    assert result.complete_local.value <= 5.0e-11
    entropy = next(
        item
        for item in result.local_terms
        if item.term_id == "joint_recognition_entropy"
    )
    assert entropy.transformed_value > entropy.original_value
    assert entropy.residual.value <= 6.0e-11
    assert result.entropy_shift.passed
    assert result.complete_monolithic.category == "monolithic"
    assert quadrature_orders == [51] * 16
    assert tuple(
        evaluation.probe.source_id
        for evaluation in result.density_probe_evaluations
    ) == tuple(probe.source_id for probe in probes)
    assert all(
        tuple(item.role for item in evaluation.observations)
        == ("p", "q", "log_ratio")
        for evaluation in result.density_probe_evaluations
    )
    assert len(
        {
            item.residual.budget.budget_sha256
            for evaluation in result.density_probe_evaluations
            for item in evaluation.observations
        }
    ) == 3 * len(probes)
    assert result.scalar_evidence == evidence
    assert result.not_applicable_reason is None

    law_pair = H7LawPairSnapshot.create(
        original=original,
        transformed=transformed,
        action_sha256=action.action_sha256,
    )
    with pytest.raises(ValueError, match="law_pair"):
        covariance.evaluate_h7_law_pair_covariance(
            object(),  # type: ignore[arg-type]
            action,
            original_factor_trace=original_trace,
            transformed_factor_trace=transformed_trace,
            density_probe_pairs=probes,
            quadrature_orders=(41, 51),
            budgets_by_invariant=budgets,
            scalar_evidence=evidence,
        )
    with pytest.raises(ValueError):
        dataclasses.replace(evidence, action_sha256="0" * 64)
    wrong_producer = H7IndependentH1EvidenceRecord.create(
        fixture_id="h1-v1",
        raw_fixture_sha256=H1_FIXTURE_RAW_SHA256,
        action_sha256=action.action_sha256,
        normalization_identity_sha256=(
            covariance.H7_INDEPENDENT_H1_NORMALIZATION_IDENTITY_SHA256
        ),
        producer_identity_sha256=_sha("not-the-independent-h1-producer"),
        original_log_evidence=evidence.original_log_evidence,
        transformed_log_evidence=evidence.transformed_log_evidence,
        original_posterior_kl=evidence.original_posterior_kl,
        transformed_posterior_kl=evidence.transformed_posterior_kl,
    )
    with pytest.raises(ValueError, match="producer identity"):
        covariance.evaluate_h7_complete_covariance(
            original,
            transformed,
            action,
            original_factor_trace=original_trace,
            transformed_factor_trace=transformed_trace,
            density_probe_pairs=None,
            quadrature_orders=(41, 51),
            budgets_by_invariant=budgets,
            scalar_evidence=wrong_producer,
        )
    wrong_normalization = H7IndependentH1EvidenceRecord.create(
        fixture_id="h1-v1",
        raw_fixture_sha256=H1_FIXTURE_RAW_SHA256,
        action_sha256=action.action_sha256,
        normalization_identity_sha256=_sha("not-the-h1-normalization"),
        producer_identity_sha256=(
            covariance.H7_INDEPENDENT_H1_PRODUCER_IDENTITY_SHA256
        ),
        original_log_evidence=evidence.original_log_evidence,
        transformed_log_evidence=evidence.transformed_log_evidence,
        original_posterior_kl=evidence.original_posterior_kl,
        transformed_posterior_kl=evidence.transformed_posterior_kl,
    )
    with pytest.raises(ValueError, match="normalization identity"):
        covariance.evaluate_h7_complete_covariance(
            original,
            transformed,
            action,
            original_factor_trace=original_trace,
            transformed_factor_trace=transformed_trace,
            density_probe_pairs=None,
            quadrature_orders=(41, 51),
            budgets_by_invariant=budgets,
            scalar_evidence=wrong_normalization,
        )
    with pytest.raises(ValueError, match="exact complete post-H6 trace"):
        covariance.evaluate_h7_complete_covariance(
            original,
            transformed,
            action,
            original_factor_trace=object(),  # type: ignore[arg-type]
            transformed_factor_trace=transformed_trace,
            density_probe_pairs=None,
            quadrature_orders=(41, 51),
            budgets_by_invariant=budgets,
            scalar_evidence=evidence,
        )
    with pytest.raises(ValueError, match="values changed"):
        dataclasses.replace(
            original_trace,
            total_value=original_trace.total_value + 1.0,
        )
    assert law_pair.original.generative.scalar_source_law is not None
    assert law_pair.original.recognition.scalar_source_law is not None


def test_matrix_inventory_and_factorized_promotion_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert len(covariance.H7_MATRIX_SCORER_RESIDUAL_IDS) == 12
    assert covariance.require_h7_matrix_scorer_residual_inventory(
        covariance.H7_MATRIX_SCORER_RESIDUAL_IDS
    ) == covariance.H7_MATRIX_SCORER_RESIDUAL_IDS
    with pytest.raises(ValueError, match="exact twelve"):
        covariance.require_h7_matrix_scorer_residual_inventory(
            covariance.H7_MATRIX_SCORER_RESIDUAL_IDS[:-1]
        )

    fixture = parse_h7_fixture_bytes(H7_FIXTURE_PATH.read_bytes())
    factorized = fixture.recognition_families[1]
    action = fixture.actions["internal"]
    law = H7CompleteLawSnapshot.create(
        fixture_id="h7-v1",
        generative=fixture.generative,
        recognition=factorized,
        raw_fixture_sha256=fixture.raw_fixture_sha256,
        scalar_probe_set=None,
    )
    expected_entropy_shift = covariance._global_log_jacobian(action)
    assert expected_entropy_shift > 0.0
    matrix_original_trace = _complete_trace("matrix-original")
    matrix_transformed_trace = _complete_trace(
        "matrix-transformed",
        entropy_shift=expected_entropy_shift,
    )
    unpromoted_pair = H7LawPairSnapshot.create(
        original=law,
        transformed=law,
        action_sha256=action.action_sha256,
    )
    with pytest.raises(ValueError, match="unrestricted"):
        covariance.evaluate_h7_law_pair_covariance(
            unpromoted_pair,
            action,
            original_factor_trace=matrix_original_trace,
            transformed_factor_trace=matrix_transformed_trace,
            density_probe_pairs=None,
            quadrature_orders=(41, 51),
            budgets_by_invariant={},
            scalar_evidence=None,
        )

    borrowed_action = borrow_h7_action(
        tuple(item.value() for item in action.elements),
        kind=action.kind,
        dimension=2,
    )
    transformed = H7CompleteLawSnapshot.create(
        fixture_id="h7-v1",
        generative=generative_pushforward.freeze_h7_generative(
            generative_pushforward._pushforward_h7_generative_snapshot(
                law.generative,
                borrowed_action,
            ),
            action=borrowed_action,
        ),
        recognition=recognition_pushforward.freeze_h7_recognition(
            recognition_pushforward._pushforward_h7_recognition_snapshot(
                law.recognition,
                borrowed_action,
            )
        ),
        raw_fixture_sha256=law.raw_fixture_sha256,
        scalar_probe_set=None,
    )
    probes = tuple(
        pair
        for pair in fixture.density_probe_pairs
        if pair.action_sha256 == action.action_sha256
        and (
            pair.component_id.startswith("p.")
            or pair.component_id.startswith("q.factorized.")
        )
    )
    assert len(probes) == 108
    budgets = _objective_budgets(
        probes,
        include_matrix_scorers=True,
        include_scalar_evidence=False,
    )
    quadrature_orders: list[int] = []
    production_quadrature = covariance.probabilists_gauss_hermite

    def recording_quadrature(order: int, *, dtype: torch.dtype):
        quadrature_orders.append(order)
        return production_quadrature(order, dtype=dtype)

    monkeypatch.setattr(
        covariance,
        "probabilists_gauss_hermite",
        recording_quadrature,
    )
    result = covariance.evaluate_h7_complete_covariance(
        law,
        transformed,
        action,
        original_factor_trace=matrix_original_trace,
        transformed_factor_trace=matrix_transformed_trace,
        density_probe_pairs=probes,
        quadrature_orders=(41, 51),
        budgets_by_invariant=budgets,
        scalar_evidence=None,
    )

    assert (
        result.original_factor_trace_sha256
        == matrix_original_trace.trace_sha256
    )
    assert (
        result.transformed_factor_trace_sha256
        == matrix_transformed_trace.trace_sha256
    )
    assert result.factorized_promotion_witness is not None
    assert result.factorized_promotion_witness.value != 0.0
    assert result.initial_joint_kl.original_value == _SIGNED_VALUES[0]
    assert result.initial_joint_kl.residual.value == 0.0
    assert quadrature_orders == [51] * 4
    assert tuple(
        item.invariant_id for item in result.scorer_residuals
    ) == covariance.H7_MATRIX_SCORER_RESIDUAL_IDS
    assert all(item.passed for item in result.scorer_residuals)
    assert result.density_probes == probes
    assert len(result.density_probe_evaluations) == 108
    assert sum(
        len(item.observations)
        for item in result.density_probe_evaluations
    ) == 208
    assert len(
        {
            observation.residual.budget.budget_sha256
            for item in result.density_probe_evaluations
            for observation in item.observations
        }
    ) == 208
    global_p = next(
        observation
        for item in result.density_probe_evaluations
        if item.probe.component_id == "p.global"
        for observation in item.observations
        if observation.role == "p"
    )
    assert global_p.expected_log_jacobian_shift > 0.0
    assert global_p.transformed_value < global_p.original_value
    assert global_p.residual.passed
    assert result.p_density_shift.value <= 2.0e-9
    assert result.q_density_shift.value <= 3.0e-9
    assert result.log_ratio.value <= 4.0e-9
    assert result.p_density_shift.passed
    assert result.q_density_shift.passed
    assert result.log_ratio.passed
    entropy = next(
        item
        for item in result.local_terms
        if item.term_id == "joint_recognition_entropy"
    )
    compensated_emission = next(
        item
        for item in result.local_terms
        if item.term_id == "expected_log_emission[1]"
    )
    assert entropy.transformed_value > entropy.original_value
    assert entropy.residual.passed
    assert result.entropy_shift.passed
    assert compensated_emission.transformed_value < (
        compensated_emission.original_value
    )
    # This synthetic trace deliberately closes the entropy/total signs while
    # leaving one local and the monolithic scientific obligations open.
    assert not compensated_emission.residual.passed
    assert result.complete_local.passed
    assert result.complete_monolithic.category == "monolithic"
    assert not result.complete_monolithic.passed
    assert (
        result.complete_monolithic.value
        > result.complete_monolithic.budget.total_allowance
    )
    assert result.evidence is None
    assert result.posterior_kl is None
    assert (
        result.not_applicable_reason
        == covariance.H7_MATRIX_EVIDENCE_NOT_APPLICABLE_REASON
    )

    first_probe = probes[0]
    malformed_invariant = (
        f"density_probe.{first_probe.probe_sha256}.p"
    )
    malformed_budget = _budget(
        malformed_invariant,
        "density",
        allowance=1.0e-9,
        operand_roles=("original",),
    )
    with pytest.raises(ValueError, match="original/transformed operands"):
        covariance._make_density_observation(
            pair=first_probe,
            role="p",
            original_value=0.0,
            transformed_value=0.0,
            expected_shift=covariance._probe_scope_shift(first_probe),
            budgets={malformed_invariant: malformed_budget},
            used_budget_ids=set(),
        )

    reversed_entropy_trace = _complete_trace(
        "matrix-reversed-entropy",
        entropy_shift=-expected_entropy_shift,
    )
    with pytest.raises(ValueError, match="entropy-shift sign"):
        covariance.evaluate_h7_complete_covariance(
            law,
            transformed,
            action,
            original_factor_trace=matrix_original_trace,
            transformed_factor_trace=reversed_entropy_trace,
            density_probe_pairs=probes,
            quadrature_orders=(41, 51),
            budgets_by_invariant=budgets,
            scalar_evidence=None,
        )
