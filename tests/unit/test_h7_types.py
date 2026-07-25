"""Compact boundary and mutation checks for the H7 auxiliary records."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from vfe4.types.h7 import (
    H7AllowanceContribution,
    H7BorrowedTensorView,
    H7BudgetRecord,
    H7ControlResult,
    H7DensityProbePair,
    H7HistoryValueSnapshot,
    H7OperandRecord,
    H7OwnedTensorSnapshot,
    H7PassOutcome,
    H7ResidualRecord,
    H7ScalarGenerativeSourceLawSnapshot,
    H7ScalarReplayAction,
    H7ScalarSourcePathSnapshot,
    H7SourceCovectorSnapshot,
    H7TensorLawComponent,
    H7TrialSpec,
    canonical_h7_bytes,
)
from vfe4.validation.h7_fixture import (
    h7_trial_specs_from_config,
    h7_validation_config_mapping,
)


def _h1_paths() -> tuple[H7ScalarSourcePathSnapshot, ...]:
    declarations = (
        ((0, 0), (0, 0), (0, 0), (0, 0)),
        ((0, 1), (0, 0), (0, 0), (0, 1)),
        ((0, 0), (0, 1), (0, 1), (0, 2)),
        ((0, 1), (0, 1), (0, 1), (0, 3)),
    )
    return tuple(
        H7ScalarSourcePathSnapshot.create(
            path_id=f"h1-path-{index}:a{a[1]}-b{b[1]}",
            a=a,
            b=b,
            model_kernel_selectors=model,
            state_kernel_selectors=state,
            observation_label_base=1,
            observation_labels=(1, 2),
            decoder_row_indices=(0, 1),
        )
        for index, (a, b, model, state) in enumerate(declarations)
    )


def test_borrowed_tensor_law_identity_is_reproducible_live_and_graph_preserving() -> (
    None
):
    left = torch.tensor([0.25, -0.5], dtype=torch.float64, requires_grad=True)
    right = left.detach().clone().requires_grad_(True)
    left_view = H7BorrowedTensorView.borrow(left)
    right_view = H7BorrowedTensorView.borrow(right)

    left_component = H7TensorLawComponent.create(
        component_id="q.initial.mean",
        receiver_t=None,
        source_j=None,
        tensors={"mean": left_view},
    )
    right_component = H7TensorLawComponent.create(
        component_id="q.initial.mean",
        receiver_t=None,
        source_j=None,
        tensors={"mean": right_view},
    )

    assert left_component.component_identity_sha256 == (
        right_component.component_identity_sha256
    )
    assert left_component.tensors["mean"].tensor is left
    left_component.tensors["mean"].tensor.square().sum().backward()
    assert left.grad is not None
    with pytest.raises(TypeError):
        hash(left_view)
    with pytest.raises(ValueError, match="unhashed and unpublishable"):
        canonical_h7_bytes(left_view)

    with torch.no_grad():
        left.add_(1.0)
    with pytest.raises(ValueError, match="version changed"):
        left_component.assert_live()


def test_borrowed_tensor_detects_raw_storage_mutation_without_version_change() -> None:
    value = torch.tensor([0.25, -0.5], dtype=torch.float64)
    view = H7BorrowedTensorView.borrow(value)
    captured_version = value._version

    value.data.add_(1.0)

    assert value._version == captured_version
    with pytest.raises(ValueError, match="raw bytes changed"):
        view.assert_intact()


def test_owned_constructor_semantics_reject_self_hashed_invalid_records() -> None:
    vector = H7OwnedTensorSnapshot.capture(
        torch.tensor([0.25, -0.5], dtype=torch.float64)
    )
    with pytest.raises(ValueError, match="history channel"):
        H7HistoryValueSnapshot.create(
            channel="state",
            population_label=0,
            value=vector,
        )
    with pytest.raises(ValueError, match="receiver/source"):
        H7SourceCovectorSnapshot.create(
            bank="model",
            channel="z",
            receiver_t=1,
            source_j=1,
            value=vector,
        )

    pair = H7DensityProbePair.create(
        probe_id="probe",
        fixture_id="h7-v1",
        component_id="p.model.receiver_1",
        source_id="source",
        action_sha256="1" * 64,
        anchor_sha256="2" * 64,
        anchor_provenance="fixture-owned anchor",
        x=vector,
        x_prime=vector,
        initial_log_jacobian_shift=0.0,
        receiver_log_jacobian_shift=0.0,
        global_log_jacobian_shift=0.0,
    )
    with pytest.raises(ValueError, match="fixture_id"):
        replace(pair, fixture_id="h0-v1")


def test_h7_config_helper_rejects_equal_but_wrong_nested_numeric_types() -> None:
    raw = h7_validation_config_mapping()
    raw["oracle_decimal_precision"] = 100.0

    with pytest.raises(ValueError, match="frozen mapping"):
        h7_trial_specs_from_config(raw)


def test_trial_and_source_path_contracts_reject_role_and_order_mutations() -> None:
    action = H7ScalarReplayAction.create(
        elements=tuple(torch.tensor([[1.25]], dtype=torch.float64) for _ in range(3)),
        kind="diagonal_base",
    )
    spec = H7TrialSpec.create(
        trial_id="scalar-base-transformed",
        role="scalar_regression",
        expected_predicate="complete_covariance",
        fixture_id="h1-v1",
        frame_profile="h1_v1",
        decoder_policy="transform",
        action=action,
        action_sha256=action.action_sha256,
    )
    with pytest.raises(ValueError, match="frozen contract"):
        replace(spec, role="positive_covariance")

    paths = _h1_paths()
    priors = (
        H7OwnedTensorSnapshot.capture(torch.tensor([1.0], dtype=torch.float64)),
        H7OwnedTensorSnapshot.capture(torch.tensor([0.35, 0.65], dtype=torch.float64)),
    )
    H7ScalarGenerativeSourceLawSnapshot.create(
        model_source_priors=priors,
        state_source_priors=priors,
        ordered_paths=paths,
    )
    with pytest.raises(ValueError, match="reordered"):
        H7ScalarGenerativeSourceLawSnapshot.create(
            model_source_priors=priors,
            state_source_priors=priors,
            ordered_paths=(paths[1], paths[0], paths[2], paths[3]),
        )


def test_budget_control_and_pass_outcome_close_their_own_predicates() -> None:
    value_sha = "1" * 64
    operand = H7OperandRecord.create(
        operand_id="covariance.original",
        category="covariance",
        role="original",
        dtype="float64",
        shape=(2, 2),
        value_sha256=value_sha,
        scale=1.0,
        condition_number=2.0,
        normalization=1.0,
        oracle_value=None,
    )
    contribution = H7AllowanceContribution.create(
        kind="operation_rounding",
        operation_id="covariance.congruence",
        operation_kind="matrix_product",
        operation_count=16,
        quadrature_order=None,
        unit_allowance=1e-15,
        value=1.6e-14,
    )
    budget = H7BudgetRecord.create(
        invariant_id="covariance",
        category="covariance",
        operands=(operand,),
        contributions=(contribution,),
        comparison_normalization=1.0,
        total_allowance=contribution.value,
    )
    H7ResidualRecord.create(
        invariant_id="covariance",
        category="tensor",
        value=budget.total_allowance,
        budget=budget,
        passed=True,
    )
    with pytest.raises(ValueError, match="must equal"):
        H7BudgetRecord.create(
            invariant_id="covariance",
            category="covariance",
            operands=(operand,),
            contributions=(contribution,),
            comparison_normalization=1.0,
            total_allowance=2.0 * contribution.value,
        )

    boundary = max(100.0 * contribution.value, 1e-8)
    H7ControlResult.create(
        control_id="wrong_covariance_congruence",
        target_invariant_id="covariance",
        wrong_residual=boundary,
        invariant_scale=1.0,
        matching_correct_allowance=contribution.value,
        decisiveness_limit=boundary,
        detected=False,
    )
    with pytest.raises(ValueError, match="strict above"):
        H7ControlResult.create(
            control_id="wrong_covariance_congruence",
            target_invariant_id="covariance",
            wrong_residual=boundary,
            invariant_scale=1.0,
            matching_correct_allowance=contribution.value,
            decisiveness_limit=boundary,
            detected=True,
        )

    H7PassOutcome.create(
        kind="PASS",
        scalar_trial_ids=(
            "scalar-base-transformed",
            "scalar-internal-transformed",
        ),
        positive_trial_ids=(
            "matrix-identity-base-transformed",
            "matrix-identity-internal-transformed",
            "matrix-nonidentity-base-transformed",
            "matrix-nonidentity-internal-transformed",
            "matrix-fixed-decoder-centered-stabilizer",
        ),
        expected_negative_trial_id=("matrix-fixed-decoder-outside-stabilizer"),
        control_ids=(
            "wrong_covariance_congruence",
            "wrong_precision_congruence",
            "history_scorer_wrong_source_inverse",
            "reversed_link_order",
            "reverse_arrow_B",
            "wrong_decoder_dual_action",
            "fixed_decoder_outside_stabilizer",
            "omitted_density_jacobian",
            "reversed_logdet_sign",
            "entropy_false_invariance",
            "changed_h1_source_probability",
            "diagonal_for_internal_action",
        ),
    )
