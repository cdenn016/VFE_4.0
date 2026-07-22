from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from vfe4.objective.dependency_graph import (
    FactorDependencyGraph,
    build_h5_reference_dependency_graph,
    expected_affected_factors,
)
from vfe4.types.h5_schema import (
    H5_FACTOR_UNIVERSE,
    H5_MODEL_BLOCK_UNIVERSE,
    H5_PARAMETER_DEPENDENCY_ROWS,
    H5_RECOGNITION_COORDINATE_UNIVERSE,
    H5_VARIABLE_DEPENDENCY_ROWS,
)
from vfe4.validation.h5_update_spec import build_h5_reference_state


ROOT = Path(__file__).parents[2]


def _graph() -> FactorDependencyGraph:
    reference = build_h5_reference_state(
        (ROOT / "vfe4/validation/fixtures/h1_v1.json").read_bytes(),
        (
            ROOT / "vfe4/validation/fixtures/h5_conditional_update_v1.json"
        ).read_bytes(),
    )
    return build_h5_reference_dependency_graph(reference.specification)


def test_closed_dependency_rows_are_exact_and_universe_ordered() -> None:
    assert H5_VARIABLE_DEPENDENCY_ROWS == (
        ("q[z0]", ("initial_joint", "state_transition[1]", "state_transition[2]", "recognition_entropy")),
        ("q[m0]", ("initial_joint", "model_transition[1]", "model_transition[2]", "recognition_entropy")),
        ("q[z1]", ("state_transition[1]", "emission[1]", "state_transition[2]", "recognition_entropy")),
        ("q[m1]", ("model_transition[1]", "state_transition[1]", "emission[1]", "model_transition[2]", "recognition_entropy")),
        ("q[z2]", ("state_transition[2]", "emission[2]", "recognition_entropy")),
        ("q[m2]", ("model_transition[2]", "state_transition[2]", "emission[2]", "recognition_entropy")),
        ("q[model_source_b1]", ("model_source[1]", "model_transition[1]", "state_source[1]", "state_transition[1]", "recognition_entropy")),
        ("q[state_source_a1_b0]", ("state_source[1]", "state_transition[1]", "recognition_entropy")),
        ("q[model_source_b2]", ("model_source[2]", "model_transition[2]", "state_source[2]", "state_transition[2]", "recognition_entropy")),
        ("q[source_row_a2]", ("state_source[2]", "state_transition[2]", "recognition_entropy")),
        ("q[state_source_a2_b1]", ("state_source[2]", "state_transition[2]", "recognition_entropy")),
    )
    assert H5_PARAMETER_DEPENDENCY_ROWS == (
        ("theta[state_transition_2]", ("state_transition[2]",)),
        ("theta[emission_1]", ("emission[1]",)),
        (
            "theta[shared_decoder_transition]",
            ("state_transition[2]", "emission[1]", "emission[2]"),
        ),
    )
    graph = _graph()
    assert graph.factor_universe == H5_FACTOR_UNIVERSE
    assert graph.recognition_coordinate_universe == H5_RECOGNITION_COORDINATE_UNIVERSE
    assert graph.model_block_universe == H5_MODEL_BLOCK_UNIVERSE
    assert graph.variable_dependencies == H5_VARIABLE_DEPENDENCY_ROWS
    assert graph.parameter_dependencies == H5_PARAMETER_DEPENDENCY_ROWS


@pytest.mark.parametrize(
    ("variables", "parameters", "expected"),
    (
        (
            ("q[z0]",),
            (),
            ("initial_joint", "state_transition[1]", "state_transition[2]", "recognition_entropy"),
        ),
        (
            ("q[source_row_a2]",),
            (),
            ("state_source[2]", "state_transition[2]", "recognition_entropy"),
        ),
        ((), ("theta[state_transition_2]",), ("state_transition[2]",)),
        ((), ("theta[emission_1]",), ("emission[1]",)),
        (
            (),
            ("theta[shared_decoder_transition]",),
            ("emission[1]", "state_transition[2]", "emission[2]"),
        ),
        (
            ("q[z1]",),
            ("theta[shared_decoder_transition]",),
            ("state_transition[1]", "emission[1]", "state_transition[2]", "emission[2]", "recognition_entropy"),
        ),
    ),
)
def test_expected_affected_factors_is_the_universe_ordered_union(
    variables: tuple[str, ...],
    parameters: tuple[str, ...],
    expected: tuple[str, ...],
) -> None:
    assert expected_affected_factors(
        _graph(), variables=variables, parameters=parameters
    ) == expected


def test_every_missing_or_extra_dependency_edge_is_rejected() -> None:
    graph = _graph()
    for row_index, (coordinate, factors) in enumerate(graph.variable_dependencies):
        for factor_index in range(len(factors)):
            changed = list(graph.variable_dependencies)
            changed[row_index] = (
                coordinate,
                factors[:factor_index] + factors[factor_index + 1 :],
            )
            with pytest.raises(ValueError):
                replace(graph, variable_dependencies=tuple(changed))
    for row_index, (block, factors) in enumerate(graph.parameter_dependencies):
        for factor_index in range(len(factors)):
            changed = list(graph.parameter_dependencies)
            changed[row_index] = (
                block,
                factors[:factor_index] + factors[factor_index + 1 :],
            )
            with pytest.raises(ValueError):
                replace(graph, parameter_dependencies=tuple(changed))

    coordinate, factors = graph.variable_dependencies[0]
    changed_variables = ((coordinate, factors + ("unknown",)),) + graph.variable_dependencies[1:]
    with pytest.raises(ValueError):
        replace(graph, variable_dependencies=changed_variables)
    block, factors = graph.parameter_dependencies[0]
    changed_parameters = ((block, factors + ("emission[1]",)),) + graph.parameter_dependencies[1:]
    with pytest.raises(ValueError):
        replace(graph, parameter_dependencies=changed_parameters)


def test_graph_rejects_duplicate_unknown_empty_and_out_of_order_updates() -> None:
    graph = _graph()
    with pytest.raises(ValueError):
        expected_affected_factors(graph, variables=(), parameters=())
    with pytest.raises(ValueError):
        expected_affected_factors(graph, variables=("q[z0]", "q[z0]"), parameters=())
    with pytest.raises(ValueError):
        expected_affected_factors(graph, variables=("q[alias]",), parameters=())
    with pytest.raises(ValueError):
        expected_affected_factors(
            graph,
            variables=("q[m0]", "q[z0]"),
            parameters=(),
        )
