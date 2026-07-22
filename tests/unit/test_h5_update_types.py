from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

import pytest
import torch

from vfe4.types.updates import (
    H5_RULE_CONTRACTS,
    CategoricalRecognitionCoordinate,
    FrozenByteState,
    FrozenTensorValue,
    GaussianRecognitionCoordinate,
    H5CandidateSnapshot,
    H5LiveState,
    H5ModelSnapshot,
    H5ReferenceState,
    H5UpdateRule,
    ModelParameterBlock,
    RecognitionSnapshot,
    UpdateLabel,
    UpdateRequest,
    UpdateSpecification,
    canonical_h5_reference_state_bytes,
    canonical_h5_semantic_state_bytes,
    initial_live,
)
from vfe4.validation.h5_update_spec import build_h5_reference_state


ROOT = Path(__file__).parents[2]
H1_BYTES = (ROOT / "vfe4/validation/fixtures/h1_v1.json").read_bytes()
H5_BYTES = (
    ROOT / "vfe4/validation/fixtures/h5_conditional_update_v1.json"
).read_bytes()


def _reference() -> H5ReferenceState:
    return build_h5_reference_state(H1_BYTES, H5_BYTES)


def test_displayed_h5_record_field_order_is_frozen() -> None:
    assert tuple(field.name for field in fields(FrozenTensorValue)) == (
        "dtype",
        "shape",
        "values",
    )
    assert tuple(field.name for field in fields(FrozenByteState)) == (
        "schema_version",
        "payload",
        "state_sha256",
    )
    assert tuple(field.name for field in fields(GaussianRecognitionCoordinate)) == (
        "coordinate_id",
        "mean",
        "variance",
    )
    assert tuple(field.name for field in fields(CategoricalRecognitionCoordinate)) == (
        "coordinate_id",
        "support",
        "conditioned_on",
        "probabilities",
    )
    assert tuple(field.name for field in fields(RecognitionSnapshot)) == (
        "schema_version",
        "gaussians",
        "categoricals",
        "state_sha256",
    )
    assert tuple(field.name for field in fields(ModelParameterBlock)) == (
        "block_id",
        "values",
    )
    assert tuple(field.name for field in fields(H5ModelSnapshot)) == (
        "schema_version",
        "parameter_blocks",
        "reconstruction_records",
        "shared_groups",
        "objective_schema_sha256",
        "state_sha256",
    )
    assert tuple(field.name for field in fields(UpdateSpecification)) == (
        "raw_bytes",
        "fixture_id",
        "fixture_schema_version",
        "recognition_family",
        "h1_fixture_id",
        "h1_fixture_sha256",
        "factor_input_schema_version",
        "factor_input_schema_sha256",
        "factor_universe",
        "recognition_coordinate_universe",
        "model_block_universe",
        "quadrature_orders",
        "reconstruction_records",
        "shared_groups",
        "initial_recognition",
        "initial_model",
        "canonical_bytes",
        "canonical_sha256",
        "raw_sha256",
    )
    assert tuple(field.name for field in fields(H5ReferenceState)) == (
        "schema_version",
        "raw_h1_fixture_bytes",
        "raw_update_spec_bytes",
        "h1_fixture_sha256",
        "update_spec_raw_sha256",
        "objective_schema_sha256",
        "factor_input_schema_sha256",
        "specification",
        "initial_recognition",
        "initial_model",
        "initial_optimizer_state",
        "initial_rng_state",
        "reference_sha256",
    )
    assert tuple(field.name for field in fields(H5LiveState)) == (
        "schema_version",
        "recognition",
        "model",
        "optimizer_state",
        "rng_state",
        "state_sha256",
    )
    assert tuple(field.name for field in fields(UpdateRequest)) == (
        "schema_version",
        "request_id",
        "rule",
        "requested_label",
        "variables",
        "parameters",
        "damping_schedule",
        "request_sha256",
    )
    assert tuple(field.name for field in fields(H5CandidateSnapshot)) == (
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
        "candidate_sha256",
    )


def test_frozen_tensor_detaches_clones_and_preserves_signed_zero() -> None:
    source = torch.tensor([-0.0, 1.5], dtype=torch.float64, requires_grad=True)
    frozen = FrozenTensorValue.from_tensor(source)
    source.detach()[1] = 99.0
    assert frozen == FrozenTensorValue("float64", (2,), (-0.0, 1.5))
    restored = frozen.to_tensor()
    assert restored.dtype is torch.float64
    assert restored.device.type == "cpu"
    assert restored.requires_grad is False
    assert restored.data_ptr() != source.data_ptr()

    positive_zero = FrozenTensorValue("float64", (), (0.0,))
    negative_zero = FrozenTensorValue("float64", (), (-0.0,))
    reference = _reference()
    first = reference.initial_recognition.gaussians[0]
    plus = replace(
        reference.initial_recognition,
        gaussians=(replace(first, mean=positive_zero),)
        + reference.initial_recognition.gaussians[1:],
    )
    minus = replace(
        reference.initial_recognition,
        gaussians=(replace(first, mean=negative_zero),)
        + reference.initial_recognition.gaussians[1:],
    )
    assert plus.state_sha256 != minus.state_sha256
    assert canonical_h5_semantic_state_bytes(plus, reference.initial_model) != (
        canonical_h5_semantic_state_bytes(minus, reference.initial_model)
    )


def test_reference_and_initial_live_rebuild_all_four_owned_states() -> None:
    reference = _reference()
    live = initial_live(reference)
    assert live.recognition.state_sha256 == reference.initial_recognition.state_sha256
    assert live.model.state_sha256 == reference.initial_model.state_sha256
    assert live.optimizer_state.state_sha256 == (
        reference.initial_optimizer_state.state_sha256
    )
    assert live.rng_state.state_sha256 == reference.initial_rng_state.state_sha256
    assert live.recognition is not reference.initial_recognition
    assert live.model is not reference.initial_model
    assert live.optimizer_state is not reference.initial_optimizer_state
    assert live.rng_state is not reference.initial_rng_state
    assert reference.raw_h1_fixture_bytes == H1_BYTES
    assert reference.raw_update_spec_bytes == H5_BYTES
    assert canonical_h5_reference_state_bytes(reference)


def test_reference_and_live_state_reject_swapped_optimizer_rng_schemas() -> None:
    reference = _reference()
    swapped_optimizer = FrozenByteState(
        "h5-deterministic-rng-v1", b'{"kind":"none"}'
    )
    swapped_rng = FrozenByteState(
        "h5-no-optimizer-v1", b'{"algorithm":"none","counter":0}'
    )
    with pytest.raises(ValueError, match="optimizer.*schema"):
        replace(reference, initial_optimizer_state=swapped_optimizer)
    with pytest.raises(ValueError, match="RNG.*schema"):
        replace(reference, initial_rng_state=swapped_rng)

    live = initial_live(reference)
    with pytest.raises(ValueError, match="optimizer.*schema"):
        replace(live, optimizer_state=swapped_optimizer)
    with pytest.raises(ValueError, match="RNG.*schema"):
        replace(live, rng_state=swapped_rng)


def test_signed_zero_changes_reference_and_semantic_hashes() -> None:
    reference = _reference()
    block = reference.initial_model.parameter_blocks[2]
    name, _ = block.values[0]
    plus_block = replace(
        block, values=((name, FrozenTensorValue("float64", (), (0.0,))),)
    )
    minus_block = replace(
        block, values=((name, FrozenTensorValue("float64", (), (-0.0,))),)
    )
    plus_model = replace(
        reference.initial_model,
        parameter_blocks=reference.initial_model.parameter_blocks[:2] + (plus_block,),
    )
    minus_model = replace(
        reference.initial_model,
        parameter_blocks=reference.initial_model.parameter_blocks[:2] + (minus_block,),
    )
    assert plus_model.state_sha256 != minus_model.state_sha256
    assert canonical_h5_semantic_state_bytes(
        reference.initial_recognition, plus_model
    ) != canonical_h5_semantic_state_bytes(reference.initial_recognition, minus_model)

    plus_spec = replace(
        reference.specification,
        initial_model=plus_model,
        raw_bytes=reference.specification.raw_bytes,
    )
    minus_spec = replace(
        reference.specification,
        initial_model=minus_model,
        raw_bytes=reference.specification.raw_bytes,
    )
    plus_reference = replace(
        reference,
        specification=plus_spec,
        initial_model=plus_model,
    )
    minus_reference = replace(
        reference,
        specification=minus_spec,
        initial_model=minus_model,
    )
    assert plus_reference.reference_sha256 != minus_reference.reference_sha256


def test_closed_rule_contracts_have_five_producers_and_no_valid_mm() -> None:
    assert tuple(H5UpdateRule) == (
        H5UpdateRule.EXACT_Z0,
        H5UpdateRule.EXACT_SOURCE_ROW_A2,
        H5UpdateRule.EXACT_STATE_TRANSITION_2_M,
        H5UpdateRule.GENERALIZED_EM_EMISSION_1,
        H5UpdateRule.NATURAL_GRADIENT_Z1,
    )
    assert UpdateLabel.VALID_MM not in tuple(
        contract[0] for contract in H5_RULE_CONTRACTS.values()
    )
    expected = {
        H5UpdateRule.EXACT_Z0: (
            UpdateLabel.EXACT_COORDINATE,
            ("q[z0]",),
            (),
            (1.0,),
        ),
        H5UpdateRule.EXACT_SOURCE_ROW_A2: (
            UpdateLabel.EXACT_COORDINATE,
            ("q[source_row_a2]",),
            (),
            (1.0,),
        ),
        H5UpdateRule.EXACT_STATE_TRANSITION_2_M: (
            UpdateLabel.EXACT_COORDINATE,
            (),
            ("theta[state_transition_2]",),
            (1.0,),
        ),
        H5UpdateRule.GENERALIZED_EM_EMISSION_1: (
            UpdateLabel.GENERALIZED_EM,
            (),
            ("theta[emission_1]",),
            (
                1.0,
                0.5,
                0.25,
                0.125,
                0.0625,
                0.03125,
                0.015625,
                0.0078125,
                0.00390625,
                0.001953125,
                0.0009765625,
            ),
        ),
        H5UpdateRule.NATURAL_GRADIENT_Z1: (
            UpdateLabel.NATURAL_GRADIENT_PROPOSAL,
            ("q[z1]",),
            (),
            (64.0,),
        ),
    }
    assert dict(H5_RULE_CONTRACTS) == expected


@pytest.mark.parametrize("rule", tuple(H5UpdateRule))
def test_update_request_accepts_only_its_exact_rule_contract(rule: H5UpdateRule) -> None:
    label, variables, parameters, schedule = H5_RULE_CONTRACTS[rule]
    request = UpdateRequest(
        "h5-update-request-v1",
        f"request-{rule.value}",
        rule,
        label,
        variables,
        parameters,
        schedule,
    )
    assert len(request.request_sha256) == 64
    with pytest.raises(ValueError):
        replace(request, requested_label=UpdateLabel.VALID_MM)
    with pytest.raises(ValueError):
        replace(request, variables=("q[model_source_b1]",))
    with pytest.raises(ValueError):
        replace(request, damping_schedule=schedule + (0.0,))


def test_candidate_provenance_and_diagnostics_are_rule_closed() -> None:
    reference = _reference()
    label, variables, parameters, schedule = H5_RULE_CONTRACTS[
        H5UpdateRule.EXACT_STATE_TRANSITION_2_M
    ]
    request = UpdateRequest(
        "h5-update-request-v1",
        "exact-m",
        H5UpdateRule.EXACT_STATE_TRANSITION_2_M,
        label,
        variables,
        parameters,
        schedule,
    )
    candidate = H5CandidateSnapshot(
        "h5-candidate-v1",
        request.rule,
        request.request_sha256,
        request.requested_label,
        request.variables,
        request.parameters,
        1.0,
        (("G_condition_number", 2.0),),
        reference.initial_recognition,
        reference.initial_model,
    )
    assert candidate.recognition is not reference.initial_recognition
    assert candidate.model is not reference.initial_model
    assert len(candidate.candidate_sha256) == 64
    with pytest.raises(ValueError):
        replace(candidate, producer_label=UpdateLabel.GENERALIZED_EM)
    with pytest.raises(ValueError):
        replace(candidate, numerical_diagnostics=())
