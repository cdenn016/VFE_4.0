from __future__ import annotations

import ast
import copy
import dataclasses
import hashlib
import inspect
import json
import math
from pathlib import Path
import struct
import textwrap

import pytest
import torch

import vfe4.artifacts as artifacts
import vfe4.generative.pushforward as generative_pushforward
import vfe4.objective.h7_covariance as covariance
import vfe4.recognition.pushforward as recognition_pushforward
import verification.mp_oracles.h7_covariance as mp_oracle
from vfe4.geometry.group_action import borrow_h7_action
from vfe4.objective.language_elbo import require_h7_complete_factor_trace
from vfe4.types.h6 import FrozenTensorSnapshot, H6FactorTerm, H6LanguageElboTerms
from vfe4.types.h7 import (
    H7AllowanceContribution,
    H7AssembledGlobalPrecisionSnapshot,
    H7BudgetCategory,
    H7BudgetRecord,
    H7CompleteLawSnapshot,
    H7GenerativeSnapshot,
    H7IndependentH1EvidenceRecord,
    H7LawPairSnapshot,
    H7OperandRecord,
    H7OwnedTensorSnapshot,
    h7_owned_sha256,
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
_SCALAR_PRECISION_IDS = (
    "scalar.p.initial_joint",
    "scalar.q.initial_joint",
    "scalar.p.p.model.receiver_1.source_0.receiver_offset",
    "scalar.p.p.state.receiver_1.source_0.receiver_offset",
    "scalar.p.p.model.receiver_2.source_0.receiver_offset",
    "scalar.p.p.state.receiver_2.source_0.receiver_offset",
    "scalar.p.p.model.receiver_2.source_1.receiver_offset",
    "scalar.p.p.state.receiver_2.source_1.receiver_offset",
    "scalar.q_model.q.model.receiver_1.source_0.receiver_offset",
    "scalar.q_model.q.model.receiver_2.source_0.receiver_offset",
    "scalar.q_model.q.model.receiver_2.source_1.receiver_offset",
    "scalar.q_state.q.state.receiver_1.a_0.b_0.receiver_offset",
    "scalar.q_state.q.state.receiver_2.a_0.b_0.receiver_offset",
    "scalar.q_state.q.state.receiver_2.a_1.b_0.receiver_offset",
    "scalar.q_state.q.state.receiver_2.a_0.b_1.receiver_offset",
    "scalar.q_state.q.state.receiver_2.a_1.b_1.receiver_offset",
    "scalar.q.global[h1-path-0:a0-b0]",
    "scalar.q.global[h1-path-1:a1-b0]",
    "scalar.q.global[h1-path-2:a0-b1]",
    "scalar.q.global[h1-path-3:a1-b1]",
    "scalar.p.global[h1-path-0:a0-b0]",
    "scalar.p.global[h1-path-1:a1-b0]",
    "scalar.p.global[h1-path-2:a0-b1]",
    "scalar.p.global[h1-path-3:a1-b1]",
)
_STRUCTURED_PRECISION_IDS = (
    "structured.p.initial_joint",
    "structured.q.initial_joint",
    "structured.p.p.model.receiver_1.receiver_offset",
    "structured.p.p.state.receiver_1.receiver_offset",
    "structured.p.p.model.receiver_2.receiver_offset",
    "structured.p.p.state.receiver_2.receiver_offset",
    "structured.q_model.q.structured.model.receiver_1.receiver_offset",
    "structured.q_model.q.structured.model.receiver_2.receiver_offset",
    "structured.q_state.q.structured.state.receiver_1.receiver_offset",
    "structured.q_state.q.structured.state.receiver_2.receiver_offset",
    "structured.q.global[matrix-singleton-path]",
    "structured.p.global[matrix-singleton-path]",
)
_FACTORIZED_PRECISION_IDS = (
    "factorized.p.initial_joint",
    "factorized.q.initial_joint",
    "factorized.p.p.model.receiver_1.receiver_offset",
    "factorized.p.p.state.receiver_1.receiver_offset",
    "factorized.p.p.model.receiver_2.receiver_offset",
    "factorized.p.p.state.receiver_2.receiver_offset",
    "factorized.q_model.q.factorized.model.receiver_1.receiver_offset",
    "factorized.q_model.q.factorized.model.receiver_2.receiver_offset",
    "factorized.q_state.q.factorized.state.receiver_1.receiver_offset",
    "factorized.q_state.q.factorized.state.receiver_2.receiver_offset",
    "factorized.q.global[matrix-singleton-path]",
    "factorized.p.global[matrix-singleton-path]",
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
        generative_pushforward.pushforward_h7_generative_snapshot(
            original.generative,
            borrowed_action,
        ),
        action=borrowed_action,
    )
    transformed_recognition = recognition_pushforward.freeze_h7_recognition(
        recognition_pushforward.pushforward_h7_recognition_snapshot(
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


def _matrix_law_pair(recognition_index: int):
    fixture = parse_h7_fixture_bytes(H7_FIXTURE_PATH.read_bytes())
    trial_spec = next(
        item
        for item in fixture.matrix_trial_specs
        if item.trial_id == "matrix-nonidentity-internal-transformed"
    )
    action = trial_spec.action
    original = H7CompleteLawSnapshot.create(
        fixture_id="h7-v1",
        generative=fixture.generative,
        recognition=fixture.recognition_families[recognition_index],
        raw_fixture_sha256=fixture.raw_fixture_sha256,
        scalar_probe_set=None,
    )
    borrowed_action = borrow_h7_action(
        tuple(item.value() for item in action.elements),
        kind=action.kind,
        dimension=2,
    )
    transformed = H7CompleteLawSnapshot.create(
        fixture_id="h7-v1",
        generative=generative_pushforward.freeze_h7_generative(
            generative_pushforward.pushforward_h7_generative_snapshot(
                original.generative,
                borrowed_action,
            ),
            action=borrowed_action,
        ),
        recognition=recognition_pushforward.freeze_h7_recognition(
            recognition_pushforward.pushforward_h7_recognition_snapshot(
                original.recognition,
                borrowed_action,
            )
        ),
        raw_fixture_sha256=original.raw_fixture_sha256,
        scalar_probe_set=None,
    )
    return (
        H7LawPairSnapshot.create(
            original=original,
            transformed=transformed,
            action_sha256=action.action_sha256,
        ),
        action,
        trial_spec,
    )


def _install_task5_moment_only_evaluator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def moment_only_evaluator(
        law: H7CompleteLawSnapshot,
        *,
        factor_trace,
        quadrature_order: int,
    ):
        assert quadrature_order == 51
        paths = covariance._source_paths(law)
        return covariance._CompleteValues(
            factor_trace=factor_trace,
            initial_joint_kl=float(factor_trace.ordered_factor_values[0]),
            initial_factor_ids=(factor_trace.ordered_factor_ids[0],),
            local_terms={
                term_id: 0.0
                for term_id in covariance.H7_COMPLETE_LOCAL_TERM_IDS
            },
            local_factor_ids={
                term_id: (f"task5-moment-only:{term_id}",)
                for term_id in covariance.H7_COMPLETE_LOCAL_TERM_IDS
            },
            complete_local=float(factor_trace.total_value),
            complete_monolithic=float(factor_trace.total_value),
            q_moments={
                path.path_id: covariance._recognition_joint_moments(
                    law.recognition,
                    path,
                )
                for path in paths
            },
            p_moments={
                path.path_id: covariance._generative_joint_moments(
                    law.generative,
                    path,
                )
                for path in paths
            },
            paths=paths,
        )

    monkeypatch.setattr(
        covariance,
        "_evaluate_complete_law",
        moment_only_evaluator,
    )


def _task5_identity_generative(fixture) -> H7GenerativeSnapshot:
    dimension = fixture.frame_profiles["identity"][0].shape[0]
    identity_links = {
        key: H7OwnedTensorSnapshot.capture(
            torch.eye(dimension, dtype=torch.float64)
        )
        for key in fixture.generative.ordered_links
    }
    return H7GenerativeSnapshot.create(
        frames=fixture.frame_profiles["identity"],
        ordered_links=identity_links,
        initial_joint=fixture.generative.initial_joint,
        transitions=fixture.generative.transitions,
        source_context=fixture.generative.source_context,
        scalar_source_law=fixture.generative.scalar_source_law,
        decoders=fixture.generative.decoders,
        support_sha256=fixture.generative.support_sha256,
        jacobian=fixture.generative.jacobian,
    )


def _task5_capture_cases():
    scalar_path = (
        Path(__file__).parents[2]
        / "vfe4"
        / "validation"
        / "fixtures"
        / "h1_v1.json"
    )
    scalar = adapt_optional_h1_fixture_bytes(
        scalar_path.read_bytes(),
        required_scalar_trials=(
            "scalar-base-transformed",
            "scalar-internal-transformed",
        ),
    )
    assert scalar is not None
    cases = [
        (
            H7LawPairSnapshot.create(
                original=scalar,
                transformed=scalar,
                action_sha256=spec.action_sha256,
            ),
            spec.action,
            spec,
            _SCALAR_PRECISION_IDS,
            16,
        )
        for spec in h7_scalar_trial_specs()
    ]

    fixture = parse_h7_fixture_bytes(H7_FIXTURE_PATH.read_bytes())
    generative_by_frame = {
        "identity": _task5_identity_generative(fixture),
        "nonidentity": fixture.generative,
    }
    family_contracts = (
        (fixture.recognition_families[0], _STRUCTURED_PRECISION_IDS),
        (fixture.recognition_families[1], _FACTORIZED_PRECISION_IDS),
    )
    for spec in fixture.matrix_trial_specs:
        for recognition, expected_ids in family_contracts:
            original = H7CompleteLawSnapshot.create(
                fixture_id="h7-v1",
                generative=generative_by_frame[spec.frame_profile],
                recognition=recognition,
                raw_fixture_sha256=fixture.raw_fixture_sha256,
                scalar_probe_set=None,
            )
            cases.append(
                (
                    H7LawPairSnapshot.create(
                        original=original,
                        transformed=original,
                        action_sha256=spec.action_sha256,
                    ),
                    spec.action,
                    spec,
                    expected_ids,
                    10,
                )
            )
    return tuple(cases)


def _task5_expected_components(
    law: H7CompleteLawSnapshot,
    law_kind: str,
    path,
):
    if law_kind == "q":
        selected = [law.recognition.initial_joint]
        for receiver_t in (1, 2):
            source_j = path.a[receiver_t - 1]
            model_source_j = path.b[receiver_t - 1]
            model = next(
                item
                for item in law.recognition.model_conditionals
                if item.receiver_t == receiver_t
                and item.source_j == model_source_j
            )
            candidates = tuple(
                item
                for item in law.recognition.state_conditionals
                if item.receiver_t == receiver_t
            )
            if law.fixture_id == "h1-v1":
                marker = f".a_{source_j}.b_{model_source_j}."
                state = next(
                    item for item in candidates if marker in item.component_id
                )
            else:
                state = next(
                    item for item in candidates if item.source_j == source_j
                )
            selected.extend((model, state))
    else:
        selected = [law.generative.initial_joint]
        for receiver_t in (1, 2):
            selected.extend(
                (
                    next(
                        item
                        for item in law.generative.transitions
                        if item.bank == "model"
                        and item.receiver_t == receiver_t
                        and item.source_j == path.b[receiver_t - 1]
                    ),
                    next(
                        item
                        for item in law.generative.transitions
                        if item.bank == "state"
                        and item.receiver_t == receiver_t
                        and item.source_j == path.a[receiver_t - 1]
                    ),
                )
            )
    return tuple(selected)


def _task5_factor_energy(
    components,
    path,
    point: torch.Tensor,
) -> torch.Tensor:
    initial, model_1, state_1, model_2, state_2 = components
    dimension = initial.mean.shape[0] // 2
    initial_point = point[: 2 * dimension]
    value = (
        0.5
        * initial_point
        @ initial.precision.value()
        @ initial_point
        - initial.information_vector.value() @ initial_point
    )
    for receiver_t, model, state in (
        (1, model_1, state_1),
        (2, model_2, state_2),
    ):
        model_target = point[
            list(covariance._block_indices("m", receiver_t, dimension))
        ]
        model_parent = point[
            list(
                covariance._block_indices(
                    "m",
                    path.b[receiver_t - 1],
                    dimension,
                )
            )
        ]
        model_residual = (
            model_target
            - model.parent_map.value() @ model_parent
            - model.offset.value()
        )
        value = (
            value
            + 0.5
            * model_residual
            @ model.receiver_law.precision.value()
            @ model_residual
        )

        state_target = point[
            list(covariance._block_indices("z", receiver_t, dimension))
        ]
        state_parent = point[
            list(
                covariance._block_indices(
                    "z",
                    path.a[receiver_t - 1],
                    dimension,
                )
            )
        ]
        assert state.same_receiver_model_map is not None
        state_residual = (
            state_target
            - state.parent_map.value() @ state_parent
            - state.same_receiver_model_map.value() @ model_target
            - state.offset.value()
        )
        value = (
            value
            + 0.5
            * state_residual
            @ state.receiver_law.precision.value()
            @ state_residual
        )
    return value


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
            generative_pushforward.pushforward_h7_generative_snapshot(
                law.generative,
                borrowed_action,
            ),
            action=borrowed_action,
        ),
        recognition=recognition_pushforward.freeze_h7_recognition(
            recognition_pushforward.pushforward_h7_recognition_snapshot(
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


def test_task5_assembles_40_global_canonical_pairs_in_exact_batch_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_task5_moment_only_evaluator(monkeypatch)
    cases = _task5_capture_cases()
    assert len(cases) == 14
    scalar_path_ids = (
        "h1-path-0:a0-b0",
        "h1-path-1:a1-b0",
        "h1-path-2:a0-b1",
        "h1-path-3:a1-b1",
    )
    expected_global_order = tuple(
        (
            trial_spec.trial_id,
            law_pair.original.recognition.origin_family,
            law_kind,
            path_id,
        )
        for law_pair, _action, trial_spec, _ids, _owned_count in cases
        for law_kind in ("q", "p")
        for path_id in (
            scalar_path_ids
            if law_pair.original.fixture_id == "h1-v1"
            else ("matrix-singleton-path",)
        )
    )

    batches = []
    assembled_rows = []
    observed_global_order = []
    eps = torch.finfo(torch.float64).eps
    for case_index, (
        law_pair,
        action,
        trial_spec,
        expected_ids,
        owned_count,
    ) in enumerate(cases):
        factor_trace = _complete_trace(f"task5-capture-{case_index}")
        batch = covariance.capture_h7_task5_precision_batch(
            law_pair,
            action,
            trial_spec=trial_spec,
            original_factor_trace=factor_trace,
        )
        batches.append(batch)
        assert tuple(item.gaussian_id for item in batch.operands) == expected_ids
        assert tuple(item.batch_index for item in batch.operands) == tuple(
            range(len(expected_ids))
        )
        assert tuple(item.source_kind for item in batch.operands) == (
            *("owned_component" for _ in range(owned_count)),
            *(
                "assembled_global"
                for _ in range(len(expected_ids) - owned_count)
            ),
        )
        assert all(
            item.assembled_global is None
            for item in batch.operands[:owned_count]
        )
        paths = {
            path.path_id: path
            for path in covariance._source_paths(law_pair.original)
        }
        for operand in batch.operands[owned_count:]:
            assembled = operand.assembled_global
            assert type(assembled) is H7AssembledGlobalPrecisionSnapshot
            assert assembled.trial_id == trial_spec.trial_id
            expected_global_dimension = (
                6 if law_pair.original.fixture_id == "h1-v1" else 12
            )
            assert assembled.mean.shape == (expected_global_dimension,)
            assert assembled.covariance.shape == (
                expected_global_dimension,
                expected_global_dimension,
            )
            assert assembled.precision.shape == (
                expected_global_dimension,
                expected_global_dimension,
            )
            assert assembled.information_vector.shape == (
                expected_global_dimension,
            )
            assert assembled.original_law_snapshot_sha256 == (
                law_pair.original.snapshot_sha256
            )
            assert operand.covariance.snapshot_sha256 == (
                assembled.covariance.snapshot_sha256
            )
            assert operand.precision.snapshot_sha256 == (
                assembled.precision.snapshot_sha256
            )
            path = paths[assembled.path_id]
            components = _task5_expected_components(
                law_pair.original,
                assembled.law_kind,
                path,
            )
            assert assembled.selected_component_sha256s == tuple(
                component.component_sha256 for component in components
            )
            moments = (
                covariance._recognition_joint_moments(
                    law_pair.original.recognition,
                    path,
                )
                if assembled.law_kind == "q"
                else covariance._generative_joint_moments(
                    law_pair.original.generative,
                    path,
                )
            )
            assert assembled.mean.snapshot_sha256 == (
                H7OwnedTensorSnapshot.capture(moments.mean).snapshot_sha256
            )
            precision = assembled.precision.value()
            covariance_value = assembled.covariance.value()
            information = assembled.information_vector.value()
            identity = torch.eye(precision.shape[0], dtype=torch.float64)
            for observed, expected in (
                (precision @ covariance_value, identity),
                (covariance_value @ precision, identity),
                (precision @ moments.mean, information),
            ):
                scale = max(
                    1.0,
                    float(observed.abs().max()),
                    float(expected.abs().max()),
                )
                assert torch.allclose(
                    observed,
                    expected,
                    rtol=256.0 * eps,
                    atol=256.0 * eps * scale,
                )

            point = torch.linspace(
                -0.375,
                0.625,
                precision.shape[0],
                dtype=torch.float64,
            )
            zero = torch.zeros_like(point)
            factor_delta = _task5_factor_energy(
                components,
                path,
                point,
            ) - _task5_factor_energy(components, path, zero)
            global_delta = (
                0.5 * point @ precision @ point - information @ point
            )
            scale = max(
                1.0,
                abs(float(factor_delta)),
                abs(float(global_delta)),
            )
            assert torch.allclose(
                factor_delta,
                global_delta,
                rtol=256.0 * eps,
                atol=256.0 * eps * scale,
            )
            assembled_rows.append(assembled)
            observed_global_order.append(
                (
                    batch.trial_id,
                    batch.recognition_family,
                    assembled.law_kind,
                    assembled.path_id,
                )
            )

    assert sum(len(batch.operands) for batch in batches) == 192
    assert sum(
        sum(item.source_kind == "owned_component" for item in batch.operands)
        for batch in batches
    ) == 152
    assert len(assembled_rows) == 40
    assert tuple(observed_global_order) == expected_global_order

    first = assembled_rows[0]
    with pytest.raises(ValueError, match="path/Gaussian identity"):
        H7AssembledGlobalPrecisionSnapshot.create(
            trial_id=first.trial_id,
            gaussian_id="scalar.q.global[wrong-path]",
            law_kind=first.law_kind,
            path_id="wrong-path",
            original_law_snapshot_sha256=(
                first.original_law_snapshot_sha256
            ),
            selected_component_sha256s=first.selected_component_sha256s,
            mean=first.mean,
            covariance=first.covariance,
            precision=first.precision,
            information_vector=first.information_vector,
        )
    for wrong_dimension in (3, 12):
        wrong_mean = torch.zeros(wrong_dimension, dtype=torch.float64)
        wrong_matrix = torch.eye(wrong_dimension, dtype=torch.float64)
        with pytest.raises(ValueError, match="global dimension"):
            H7AssembledGlobalPrecisionSnapshot.create(
                trial_id=first.trial_id,
                gaussian_id=first.gaussian_id,
                law_kind=first.law_kind,
                path_id=first.path_id,
                original_law_snapshot_sha256=(
                    first.original_law_snapshot_sha256
                ),
                selected_component_sha256s=(
                    first.selected_component_sha256s
                ),
                mean=H7OwnedTensorSnapshot.capture(wrong_mean),
                covariance=H7OwnedTensorSnapshot.capture(wrong_matrix),
                precision=H7OwnedTensorSnapshot.capture(wrong_matrix),
                information_vector=H7OwnedTensorSnapshot.capture(wrong_mean),
            )

    first_matrix = next(
        item
        for item in assembled_rows
        if item.trial_id == "matrix-identity-base-transformed"
    )
    for wrong_dimension in (6, 15):
        wrong_mean = torch.zeros(wrong_dimension, dtype=torch.float64)
        wrong_matrix = torch.eye(wrong_dimension, dtype=torch.float64)
        with pytest.raises(ValueError, match="global dimension"):
            H7AssembledGlobalPrecisionSnapshot.create(
                trial_id=first_matrix.trial_id,
                gaussian_id=first_matrix.gaussian_id,
                law_kind=first_matrix.law_kind,
                path_id=first_matrix.path_id,
                original_law_snapshot_sha256=(
                    first_matrix.original_law_snapshot_sha256
                ),
                selected_component_sha256s=(
                    first_matrix.selected_component_sha256s
                ),
                mean=H7OwnedTensorSnapshot.capture(wrong_mean),
                covariance=H7OwnedTensorSnapshot.capture(wrong_matrix),
                precision=H7OwnedTensorSnapshot.capture(wrong_matrix),
                information_vector=H7OwnedTensorSnapshot.capture(wrong_mean),
            )
    with pytest.raises(ValueError, match="path/Gaussian identity"):
        H7AssembledGlobalPrecisionSnapshot.create(
            trial_id=first.trial_id,
            gaussian_id=first_matrix.gaussian_id,
            law_kind=first_matrix.law_kind,
            path_id=first_matrix.path_id,
            original_law_snapshot_sha256=(
                first_matrix.original_law_snapshot_sha256
            ),
            selected_component_sha256s=(
                first_matrix.selected_component_sha256s
            ),
            mean=first_matrix.mean,
            covariance=first_matrix.covariance,
            precision=first_matrix.precision,
            information_vector=first_matrix.information_vector,
        )

    def semantic_without(record: object, integrity_field: str) -> dict[str, object]:
        return {
            field.name: getattr(record, field.name)
            for field in dataclasses.fields(record)
            if field.name != integrity_field
        }

    first_global_operand = next(
        item
        for batch in batches
        for item in batch.operands
        if item.source_kind == "assembled_global"
    )
    assert first_global_operand.operand_sha256 == h7_owned_sha256(
        "vfe4.h7.task5-precision-operand.v2",
        semantic_without(first_global_operand, "operand_sha256"),
    )
    legacy_operand_sha256 = h7_owned_sha256(
        "vfe4.h7.task5-precision-operand.v1",
        semantic_without(first_global_operand, "operand_sha256"),
    )
    with pytest.raises(ValueError, match="operand_sha256"):
        dataclasses.replace(
            first_global_operand,
            operand_sha256=legacy_operand_sha256,
        )

    first_batch = batches[0]
    assert first_batch.capture_sha256 == h7_owned_sha256(
        "vfe4.h7.task5-precision-capture-batch.v2",
        semantic_without(first_batch, "capture_sha256"),
    )
    legacy_capture_sha256 = h7_owned_sha256(
        "vfe4.h7.task5-precision-capture-batch.v1",
        semantic_without(first_batch, "capture_sha256"),
    )
    with pytest.raises(ValueError, match="capture_sha256"):
        dataclasses.replace(
            first_batch,
            capture_sha256=legacy_capture_sha256,
        )

    defensive = H7AssembledGlobalPrecisionSnapshot.create(
        trial_id=first.trial_id,
        gaussian_id=first.gaussian_id,
        law_kind=first.law_kind,
        path_id=first.path_id,
        original_law_snapshot_sha256=first.original_law_snapshot_sha256,
        selected_component_sha256s=first.selected_component_sha256s,
        mean=H7OwnedTensorSnapshot.capture(first.mean.value()),
        covariance=H7OwnedTensorSnapshot.capture(first.covariance.value()),
        precision=H7OwnedTensorSnapshot.capture(first.precision.value()),
        information_vector=H7OwnedTensorSnapshot.capture(
            first.information_vector.value()
        ),
    )
    owned = object.__getattribute__(
        defensive.precision,
        "_H7OwnedTensorSnapshot__owned",
    )
    owned.add_(1.0)
    with pytest.raises(ValueError, match="integrity changed"):
        defensive.__post_init__()


def test_task5_assembler_rejects_wrong_bindings_and_inverse_synthesis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = textwrap.dedent(
        inspect.getsource(covariance._assemble_task5_global_canonical)
    )
    tree = ast.parse(source)

    def called_name(node: ast.expr) -> str:
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        return ".".join(reversed(parts))

    forbidden_terminals = {
        "cholesky_inverse",
        "cholesky_solve",
        "inv",
        "lu_solve",
        "pinv",
        "solve",
        "solve_triangular",
    }
    calls = {
        called_name(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    assert not {
        call
        for call in calls
        if call.rsplit(".", maxsplit=1)[-1] in forbidden_terminals
    }
    assert "injected_global_precisions" not in inspect.signature(
        covariance.capture_h7_task5_precision_batch
    ).parameters

    _install_task5_moment_only_evaluator(monkeypatch)
    law_pair, _action, trial_spec = _matrix_law_pair(0)
    path = covariance._source_paths(law_pair.original)[0]
    moments = covariance._recognition_joint_moments(
        law_pair.original.recognition,
        path,
    )
    components = covariance._task5_selected_global_components(
        law_pair.original,
        "q",
        path,
    )
    gaussian_id = "structured.q.global[matrix-singleton-path]"

    wrong_path = dataclasses.replace(path, a=(0, 0))
    with pytest.raises(ValueError, match="path/component binding"):
        covariance._assemble_task5_global_canonical(
            law_pair.original,
            trial_id=trial_spec.trial_id,
            gaussian_id=gaussian_id,
            law_kind="q",
            path=wrong_path,
            moments=moments,
            components=components,
        )
    wrong_components = (
        components[0],
        components[2],
        components[1],
        components[3],
        components[4],
    )
    with pytest.raises(ValueError, match="path/component binding"):
        covariance._assemble_task5_global_canonical(
            law_pair.original,
            trial_id=trial_spec.trial_id,
            gaussian_id=gaussian_id,
            law_kind="q",
            path=path,
            moments=moments,
            components=wrong_components,
        )

    stale_precision = components[1].receiver_law.precision
    stale_owned = object.__getattribute__(
        stale_precision,
        "_H7OwnedTensorSnapshot__owned",
    )
    stale_owned.add_(1.0)
    with pytest.raises(ValueError, match="integrity changed"):
        covariance._assemble_task5_global_canonical(
            law_pair.original,
            trial_id=trial_spec.trial_id,
            gaussian_id=gaussian_id,
            law_kind="q",
            path=path,
            moments=moments,
            components=components,
        )


_TASK5_PRECISION_V2_ROOT_FIELDS = {
    "precision_table_schema",
    "h1_raw_fixture_sha256",
    "h7_raw_fixture_sha256",
    "ordered_trial_ids",
    "source_contract",
    "binary64_text_policy",
    "precision_set_sha256",
    "records",
}
_TASK5_PRECISION_V2_ROW_FIELDS = {
    "row_index",
    "trial_id",
    "gaussian_id",
    "source_kind",
    "shape",
    "covariance_values",
    "covariance_values_sha256",
    "covariance_snapshot_sha256",
    "precision_values",
    "precision_values_sha256",
    "precision_snapshot_sha256",
    "record_sha256",
}
_TASK5_PRECISION_V2_TABLE_CACHE: bytes | None = None


def _task5_precision_v2_table_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> bytes:
    writer = getattr(
        artifacts,
        "build_h7_task5_precision_operand_table_bytes",
        None,
    )
    assert callable(writer), "missing H7 Task-5 precision v2 table writer"
    global _TASK5_PRECISION_V2_TABLE_CACHE
    if _TASK5_PRECISION_V2_TABLE_CACHE is None:
        _install_task5_moment_only_evaluator(monkeypatch)
        batches = tuple(
            covariance.capture_h7_task5_precision_batch(
                law_pair,
                action,
                trial_spec=trial_spec,
                original_factor_trace=_complete_trace(
                    f"task5-precision-v2-{case_index}"
                ),
            )
            for case_index, (
                law_pair,
                action,
                trial_spec,
                _expected_ids,
                _owned_count,
            ) in enumerate(_task5_capture_cases())
        )
        _TASK5_PRECISION_V2_TABLE_CACHE = writer(batches)
    return _TASK5_PRECISION_V2_TABLE_CACHE


def _canonical_table_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _walk_binary64_tokens(value: object):
    if type(value) is list:
        for item in value:
            yield from _walk_binary64_tokens(item)
        return
    assert type(value) is str
    parsed = float(value)
    assert math.isfinite(parsed)
    assert repr(parsed) == value
    assert struct.pack("<d", float(repr(parsed))) == struct.pack("<d", parsed)
    yield value


def _snapshot_sha256_from_v2_values(values: object) -> str:
    assert type(values) is list

    def to_binary64(value: object) -> object:
        if type(value) is list:
            return [to_binary64(item) for item in value]
        assert type(value) is str
        return float(value)

    tensor = torch.tensor(to_binary64(values), dtype=torch.float64)
    return H7OwnedTensorSnapshot.capture(tensor).snapshot_sha256


def _rehash_task5_precision_v2_table(table: dict[str, object]) -> None:
    records = table["records"]
    assert type(records) is list
    for record in records:
        assert type(record) is dict
        identity = {
            "trial_id": record["trial_id"],
            "gaussian_id": record["gaussian_id"],
            "source_kind": record["source_kind"],
            "shape": record["shape"],
        }
        record["covariance_values_sha256"] = h7_owned_sha256(
            "vfe4.h7.mp-serialized-covariance-values.v2",
            {**identity, "covariance_values": record["covariance_values"]},
        )
        record["precision_values_sha256"] = h7_owned_sha256(
            "vfe4.h7.mp-serialized-precision-values.v2",
            {**identity, "precision_values": record["precision_values"]},
        )
        record["record_sha256"] = h7_owned_sha256(
            "vfe4.h7.mp-serialized-precision-operand.v2",
            {
                key: record[key]
                for key in _TASK5_PRECISION_V2_ROW_FIELDS
                if key != "record_sha256"
            },
        )
    table["precision_set_sha256"] = h7_owned_sha256(
        "vfe4.h7.mp-serialized-precision-set.v2",
        {
            key: table[key]
            for key in _TASK5_PRECISION_V2_ROOT_FIELDS
            if key != "precision_set_sha256"
        },
    )


def test_task5_precision_operand_v2_round_trip_binds_production_values_and_independent_covariance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table_bytes = _task5_precision_v2_table_bytes(monkeypatch)
    table = json.loads(table_bytes.decode("ascii"))
    assert type(table) is dict
    assert set(table) == _TASK5_PRECISION_V2_ROOT_FIELDS
    assert table_bytes == _canonical_table_bytes(table)
    assert table["precision_table_schema"] == "h7-mp-precision-operands-v2"
    assert (
        table["source_contract"]
        == "task5-production-covariance-and-precision-v2"
    )
    assert (
        table["binary64_text_policy"]
        == "python-repr-binary64-roundtrip-v1"
    )
    assert table["h1_raw_fixture_sha256"] == H1_FIXTURE_RAW_SHA256
    assert table["h7_raw_fixture_sha256"] == hashlib.sha256(
        H7_FIXTURE_PATH.read_bytes()
    ).hexdigest()
    assert tuple(table["ordered_trial_ids"]) == mp_oracle.H7_REQUIRED_TRIAL_IDS

    records = table["records"]
    assert type(records) is list
    assert len(records) == 192
    assert sum(row["source_kind"] == "owned_component" for row in records) == 152
    assert sum(row["source_kind"] == "assembled_global" for row in records) == 40
    for row_index, row in enumerate(records):
        assert set(row) == _TASK5_PRECISION_V2_ROW_FIELDS
        assert row["row_index"] == str(row_index)
        assert all(
            type(item) is str and str(int(item)) == item
            for item in row["shape"]
        )
        covariance_tokens = tuple(
            _walk_binary64_tokens(row["covariance_values"])
        )
        precision_tokens = tuple(
            _walk_binary64_tokens(row["precision_values"])
        )
        assert len(covariance_tokens) == int(row["shape"][0]) ** 2
        assert len(precision_tokens) == int(row["shape"][0]) ** 2
        assert row["covariance_snapshot_sha256"] == (
            _snapshot_sha256_from_v2_values(row["covariance_values"])
        )
        assert row["precision_snapshot_sha256"] == (
            _snapshot_sha256_from_v2_values(row["precision_values"])
        )
        identity = {
            "trial_id": row["trial_id"],
            "gaussian_id": row["gaussian_id"],
            "source_kind": row["source_kind"],
            "shape": row["shape"],
        }
        assert row["covariance_values_sha256"] == h7_owned_sha256(
            "vfe4.h7.mp-serialized-covariance-values.v2",
            {**identity, "covariance_values": row["covariance_values"]},
        )
        assert row["precision_values_sha256"] == h7_owned_sha256(
            "vfe4.h7.mp-serialized-precision-values.v2",
            {**identity, "precision_values": row["precision_values"]},
        )
        assert row["record_sha256"] == h7_owned_sha256(
            "vfe4.h7.mp-serialized-precision-operand.v2",
            {
                key: row[key]
                for key in _TASK5_PRECISION_V2_ROW_FIELDS
                if key != "record_sha256"
            },
        )
    assert table["precision_set_sha256"] == h7_owned_sha256(
        "vfe4.h7.mp-serialized-precision-set.v2",
        {
            key: table[key]
            for key in _TASK5_PRECISION_V2_ROOT_FIELDS
            if key != "precision_set_sha256"
        },
    )

    parsed = mp_oracle._parse_raw_json(table_bytes)
    source = mp_oracle._validate_precision_operand_table(parsed)
    previous_precision = mp_oracle.mp.mp.dps
    mp_oracle.mp.mp.dps = 100
    try:
        for row_index, row in enumerate(records):
            independent_covariance = mp_oracle._matrix(
                row["covariance_values"]
            )
            if row_index == 0:
                independent_covariance[0, 0] += mp_oracle.mp.mpf("1e-80")
            source.consume(
                trial_id=row["trial_id"],
                gaussian_id=row["gaussian_id"],
                covariance=independent_covariance,
            )
        source.require_complete()
    finally:
        mp_oracle.mp.mp.dps = previous_precision


def test_task5_precision_operand_v2_rejects_legacy_injected_and_malformed_tables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid = json.loads(
        _task5_precision_v2_table_bytes(monkeypatch).decode("ascii")
    )

    mutations = []

    legacy = copy.deepcopy(valid)
    legacy["precision_table_schema"] = "h7-mp-precision-operands-v1"
    _rehash_task5_precision_v2_table(legacy)
    mutations.append(legacy)

    injected = copy.deepcopy(valid)
    injected["records"][0]["source_kind"] = "injected"
    _rehash_task5_precision_v2_table(injected)
    mutations.append(injected)

    stale_covariance_snapshot = copy.deepcopy(valid)
    stale_covariance_snapshot["records"][0][
        "covariance_snapshot_sha256"
    ] = _sha("stale covariance snapshot")
    _rehash_task5_precision_v2_table(stale_covariance_snapshot)
    mutations.append(stale_covariance_snapshot)

    stale_precision_snapshot = copy.deepcopy(valid)
    stale_precision_snapshot["records"][0][
        "precision_snapshot_sha256"
    ] = _sha("stale precision snapshot")
    _rehash_task5_precision_v2_table(stale_precision_snapshot)
    mutations.append(stale_precision_snapshot)

    reordered = copy.deepcopy(valid)
    reordered["records"][0], reordered["records"][1] = (
        reordered["records"][1],
        reordered["records"][0],
    )
    for index, record in enumerate(reordered["records"]):
        record["row_index"] = str(index)
    _rehash_task5_precision_v2_table(reordered)
    mutations.append(reordered)

    duplicated = copy.deepcopy(valid)
    duplicated["records"][1] = copy.deepcopy(duplicated["records"][0])
    duplicated["records"][1]["row_index"] = "1"
    _rehash_task5_precision_v2_table(duplicated)
    mutations.append(duplicated)

    missing_field = copy.deepcopy(valid)
    del missing_field["records"][0]["covariance_values_sha256"]
    mutations.append(missing_field)

    extra_field = copy.deepcopy(valid)
    extra_field["records"][0]["unexpected"] = "closed-schema violation"
    mutations.append(extra_field)

    nonfinite = copy.deepcopy(valid)
    nonfinite["records"][0]["covariance_values"][0][0] = "nan"
    _rehash_task5_precision_v2_table(nonfinite)
    mutations.append(nonfinite)

    noncanonical = copy.deepcopy(valid)
    noncanonical["records"][0]["covariance_values"][0][0] = "1.00"
    _rehash_task5_precision_v2_table(noncanonical)
    mutations.append(noncanonical)

    nonroundtripping = copy.deepcopy(valid)
    nonroundtripping["records"][0]["covariance_values"][0][0] = (
        "0.10000000000000001"
    )
    _rehash_task5_precision_v2_table(nonroundtripping)
    mutations.append(nonroundtripping)

    for mutation in mutations:
        raw = _canonical_table_bytes(mutation)
        with pytest.raises(mp_oracle._H7ExternalDataError):
            mp_oracle._validate_precision_operand_table(
                mp_oracle._parse_raw_json(raw)
            )
