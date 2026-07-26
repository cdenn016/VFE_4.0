from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest
import torch

from vfe4.types.h7 import (
    H7_CONTROL_IDS,
    H7_MATRIX_TRIAL_IDS,
    H7_SCALAR_TRIAL_IDS,
    H7BorrowedActionView,
    H7BorrowedTensorView,
    H7GLPlus2Action,
    H7OwnedTensorSnapshot,
    H7RawTensorIdentity,
    H7ScalarReplayAction,
    h7_owned_sha256,
)
from vfe4.validation.h7_fixture import (
    H1_FIXTURE_RAW_SHA256,
    H7_DENSITY_PROBE_SET_SHA256,
    H7_DENSITY_PROBE_TABLE_PATH,
    H7_DENSITY_PROBE_TABLE_RAW_SHA256,
    H7_FIXTURE_PATH,
    H7_FIXTURE_RAW_SHA256,
    H7_SCALAR_PROBE_SET_SHA256,
    H7_SCALAR_PROBE_TABLE_PATH,
    H7_SCALAR_PROBE_TABLE_RAW_SHA256,
    adapt_optional_h1_fixture_bytes,
    parse_h7_fixture_bytes,
)
from verification.mp_oracles.h7_covariance import build_h7_scalar_probe_table_bytes


def test_h7_raw_fixture_parses_to_the_frozen_matrix_contract() -> None:
    data = H7_FIXTURE_PATH.read_bytes()
    fixture = parse_h7_fixture_bytes(data)

    assert hashlib.sha256(data).hexdigest() == H7_FIXTURE_RAW_SHA256
    assert fixture.fixture_id == "h7-v1"
    assert fixture.raw_fixture_sha256 == H7_FIXTURE_RAW_SHA256
    assert tuple(fixture.frame_profiles) == ("identity", "nonidentity")
    assert tuple(fixture.actions) == (
        "diagonal",
        "fixed_decoder_stabilizer",
        "internal",
    )
    assert all(type(action) is H7GLPlus2Action for action in fixture.actions.values())
    assert tuple(spec.trial_id for spec in fixture.matrix_trial_specs) == (
        H7_MATRIX_TRIAL_IDS
    )
    component_dimensions = (
        ("p.initial_joint", 4),
        ("p.model.receiver_1", 2),
        ("p.model.receiver_2", 2),
        ("p.state.receiver_1", 2),
        ("p.state.receiver_2", 2),
        ("q.structured.initial_joint", 4),
        ("q.structured.model.receiver_1", 2),
        ("q.structured.model.receiver_2", 2),
        ("q.structured.state.receiver_1", 2),
        ("q.structured.state.receiver_2", 2),
        ("q.factorized.initial_joint", 4),
        ("q.factorized.model.receiver_1", 2),
        ("q.factorized.model.receiver_2", 2),
        ("q.factorized.state.receiver_1", 2),
        ("q.factorized.state.receiver_2", 2),
        ("p.global", 12),
        ("q.structured.global", 12),
        ("q.factorized.global", 12),
    )
    expected_probe_ids = tuple(
        f"{action}:{component}:{direction}"
        for action in ("diagonal", "fixed_decoder_stabilizer", "internal")
        for component, dimension in component_dimensions
        for direction in (
            "zero",
            *(
                signed
                for index in range(dimension)
                for signed in (f"+e{index}", f"-e{index}")
            ),
        )
    )
    assert len(expected_probe_ids) == 486
    assert (
        tuple(pair.probe_id for pair in fixture.density_probe_pairs)
        == expected_probe_ids
    )
    assert fixture.density_probe_set_sha256 == H7_DENSITY_PROBE_SET_SHA256
    assert (
        h7_owned_sha256(
            "vfe4.h7.density-probe-set.v1",
            fixture.density_probe_pairs,
        )
        == H7_DENSITY_PROBE_SET_SHA256
    )
    with pytest.raises(ValueError, match="density_probe_set_sha256"):
        dataclasses.replace(fixture, density_probe_set_sha256="0" * 64)
    table_bytes = H7_DENSITY_PROBE_TABLE_PATH.read_bytes()
    assert hashlib.sha256(table_bytes).hexdigest() == H7_DENSITY_PROBE_TABLE_RAW_SHA256
    assert tuple(family.origin_family for family in fixture.recognition_families) == (
        "structured_full_block",
        "factorized_diagonal_within_fiber",
    )
    assert fixture.recognition_families[1].representation == (
        "factorized_diagonal_within_fiber"
    )
    assert tuple(row.bank for row in fixture.generative.source_context.scorer_rows) == (
        "model",
        "model",
        "state",
        "state",
    )
    scorer_rows = fixture.generative.source_context.scorer_rows
    assert tuple((row.receiver_t, row.source_j) for row in scorer_rows) == (
        (1, 0),
        (2, 1),
        (1, 0),
        (2, 1),
    )
    torch.testing.assert_close(
        torch.stack(tuple(row.raw_scores.value()[0] for row in scorer_rows)),
        torch.tensor(
            [0.367, -0.0845, -0.094, 0.2315],
            dtype=torch.float64,
        ),
        rtol=0.0,
        atol=1.0e-15,
    )
    assert all(
        torch.equal(row.probabilities.value(), torch.ones(1, dtype=torch.float64))
        for row in scorer_rows
    )
    assert all(
        family.source_rows == scorer_rows
        and family.context.observation_labels == (0, 2)
        and family.context.conditioning == "smoothing"
        for family in fixture.recognition_families
    )


def test_h7_raw_fixture_rejects_any_byte_or_schema_drift() -> None:
    data = H7_FIXTURE_PATH.read_bytes()
    with pytest.raises(ValueError, match="raw SHA-256"):
        parse_h7_fixture_bytes(data.replace(b'"GL+(2,R)"', b'"GL(2,R)"'))
    with pytest.raises(ValueError):
        parse_h7_fixture_bytes(data + b"\n")
    with pytest.raises(ValueError):
        parse_h7_fixture_bytes(memoryview(data))  # type: ignore[arg-type]


def test_frozen_h7_scalar_probe_table_matches_independent_builder() -> None:
    h1_bytes = (
        Path(__file__).parents[2] / "vfe4" / "validation" / "fixtures" / "h1_v1.json"
    ).read_bytes()
    table_bytes = H7_SCALAR_PROBE_TABLE_PATH.read_bytes()
    table = json.loads(table_bytes)

    assert table_bytes == build_h7_scalar_probe_table_bytes(h1_bytes)
    assert hashlib.sha256(table_bytes).hexdigest() == H7_SCALAR_PROBE_TABLE_RAW_SHA256
    assert table["fixture_id"] == "h1-v1"
    assert table["probe_set_sha256"] == H7_SCALAR_PROBE_SET_SHA256
    assert tuple(record["row_index"] for record in table["records"]) == tuple(
        str(index) for index in range(8)
    )


def test_borrowed_views_preserve_identity_and_owned_snapshots_clone() -> None:
    value = torch.tensor(
        [[1.2, 0.1], [0.0, 0.9]],
        dtype=torch.float64,
        requires_grad=True,
    )
    identity = H7RawTensorIdentity.capture(value)
    borrowed = H7BorrowedTensorView.borrow(value)
    owned = H7OwnedTensorSnapshot.capture(value)

    assert borrowed.tensor is value
    assert borrowed.identity == identity
    borrowed.assert_intact()
    assert owned.value().data_ptr() != value.data_ptr()
    assert torch.equal(owned.value(), value.detach())
    owned.assert_intact()

    with torch.no_grad():
        value[0, 0] += 0.01
    with pytest.raises(ValueError, match="version"):
        borrowed.assert_intact()
    assert torch.equal(
        owned.value(),
        torch.tensor([[1.2, 0.1], [0.0, 0.9]], dtype=torch.float64),
    )


def test_owned_action_types_are_disjoint_and_self_hashing() -> None:
    scalar = H7ScalarReplayAction.create(
        elements=tuple(
            torch.tensor([[item]], dtype=torch.float64) for item in (0.8, 1.1, 1.4)
        ),
        kind="internal_product",
    )
    matrix = H7GLPlus2Action.create(
        elements=(
            torch.tensor([[1.25, 0.1], [0.05, 0.95]], dtype=torch.float64),
            torch.tensor([[0.85, -0.2], [0.1, 1.15]], dtype=torch.float64),
            torch.tensor([[1.05, 0.25], [-0.15, 0.9]], dtype=torch.float64),
        ),
        kind="internal_product",
    )

    assert (scalar.dimension, scalar.group, scalar.representation) == (
        1,
        "GL+(1,R)",
        "standard_scalar",
    )
    assert (matrix.dimension, matrix.group, matrix.representation) == (
        2,
        "GL+(2,R)",
        "direct_gl_plus_2",
    )
    assert scalar.action_sha256 != matrix.action_sha256
    with pytest.raises(ValueError):
        dataclasses.replace(scalar, action_sha256="0" * 64)
    with pytest.raises(ValueError):
        H7ScalarReplayAction.create(
            elements=tuple(item.value() for item in matrix.elements),
            kind="internal_product",
        )


def test_borrowed_action_rejects_group_dimension_shape_disagreement() -> None:
    elements = tuple(
        H7BorrowedTensorView.borrow(torch.eye(2, dtype=torch.float64)) for _ in range(3)
    )
    with pytest.raises(ValueError):
        H7BorrowedActionView(
            elements=elements,
            kind="diagonal_base",
            dimension=1,
            group="GL+(1,R)",
        )


def test_optional_h1_adapter_has_exact_four_case_contract() -> None:
    h1_bytes = (
        Path(__file__).parents[2] / "vfe4" / "validation" / "fixtures" / "h1_v1.json"
    ).read_bytes()
    required = H7_SCALAR_TRIAL_IDS

    snapshot = adapt_optional_h1_fixture_bytes(
        h1_bytes, required_scalar_trials=required
    )
    assert snapshot is not None
    assert snapshot.fixture_id == "h1-v1"
    assert snapshot.raw_fixture_sha256 == H1_FIXTURE_RAW_SHA256
    generative_sources = snapshot.generative.scalar_source_law
    recognition_sources = snapshot.recognition.scalar_source_law
    probes = snapshot.scalar_probe_set
    assert generative_sources is not None
    assert recognition_sources is not None
    assert probes is not None
    assert tuple(path.path_id for path in generative_sources.ordered_paths) == (
        "h1-path-0:a0-b0",
        "h1-path-1:a1-b0",
        "h1-path-2:a0-b1",
        "h1-path-3:a1-b1",
    )
    assert tuple(
        (
            path.a,
            path.b,
            path.model_kernel_selectors,
            path.state_kernel_selectors,
            path.observation_label_base,
            path.observation_labels,
            path.decoder_row_indices,
        )
        for path in recognition_sources.ordered_paths
    ) == (
        ((0, 0), (0, 0), (0, 0), (0, 0), 1, (1, 2), (0, 1)),
        ((0, 1), (0, 0), (0, 0), (0, 1), 1, (1, 2), (0, 1)),
        ((0, 0), (0, 1), (0, 1), (0, 2), 1, (1, 2), (0, 1)),
        ((0, 1), (0, 1), (0, 1), (0, 3), 1, (1, 2), (0, 1)),
    )
    assert tuple(
        tuple(row.value().tolist()) for row in generative_sources.model_source_priors
    ) == ((1.0,), (0.35, 0.65))
    assert tuple(
        tuple(row.value().tolist()) for row in generative_sources.state_source_priors
    ) == ((1.0,), (0.55, 0.45))
    assert tuple(
        tuple(row.value().tolist())
        for row in recognition_sources.model_source_probabilities
    ) == ((1.0,), (0.4, 0.6))
    assert tuple(
        tuple(tuple(row) for row in table.value().tolist())
        for table in (recognition_sources.state_source_probabilities_given_model_source)
    ) == (((1.0,),), ((0.75, 0.25), (0.2, 0.8)))
    assert tuple(
        item.source_j for item in snapshot.recognition.state_conditionals[-4:]
    ) == (0, 0, 1, 1)
    assert tuple(
        item.component_id for item in snapshot.recognition.state_conditionals[-4:]
    ) == (
        "h1.q.state.2.a_0.b_0.row_0",
        "h1.q.state.2.a_1.b_0.row_1",
        "h1.q.state.2.a_0.b_1.row_2",
        "h1.q.state.2.a_1.b_1.row_3",
    )
    assert len(probes.probe_pairs) == 8
    assert tuple(
        (pair.action_sha256, pair.source_id) for pair in probes.probe_pairs
    ) == tuple(
        (action_sha256, path_id)
        for action_sha256 in probes.scalar_trial_action_sha256
        for path_id in probes.ordered_source_path_ids
    )
    expected_x = torch.tensor(
        [0.2, -0.15, 0.090625, -0.0875, 0.2995, -0.17],
        dtype=torch.float64,
    )
    expected_base_prime = torch.tensor(
        [0.25, -0.1875, 0.11328125, -0.109375, 0.374375, -0.2125],
        dtype=torch.float64,
    )
    expected_internal_prime = torch.tensor(
        [0.16, -0.12, 0.0996875, -0.09625, 0.4193, -0.238],
        dtype=torch.float64,
    )
    assert torch.equal(probes.probe_pairs[0].x.value(), expected_x)
    assert torch.equal(probes.probe_pairs[0].x_prime.value(), expected_base_prime)
    assert torch.equal(probes.probe_pairs[4].x.value(), expected_x)
    assert torch.equal(probes.probe_pairs[4].x_prime.value(), expected_internal_prime)
    assert all(
        H1_FIXTURE_RAW_SHA256 in pair.anchor_provenance
        and pair.source_id in pair.anchor_provenance
        for pair in probes.probe_pairs
    )
    with pytest.raises(ValueError):
        dataclasses.replace(probes, probe_set_sha256="0" * 64)
    with pytest.raises(ValueError, match="required"):
        adapt_optional_h1_fixture_bytes(None, required_scalar_trials=required)
    assert adapt_optional_h1_fixture_bytes(None, required_scalar_trials=()) is None
    with pytest.raises(ValueError, match="unused"):
        adapt_optional_h1_fixture_bytes(h1_bytes, required_scalar_trials=())
    with pytest.raises(ValueError):
        adapt_optional_h1_fixture_bytes(
            h1_bytes,
            required_scalar_trials=(H7_SCALAR_TRIAL_IDS[0],),
        )


def test_h7_control_inventory_is_exact_and_closed() -> None:
    assert H7_CONTROL_IDS == (
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
    )
