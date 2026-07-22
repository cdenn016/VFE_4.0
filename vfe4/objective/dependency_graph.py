"""Closed H5 recognition/model-block to complete-factor dependency graph."""

from __future__ import annotations

from dataclasses import dataclass

from vfe4.types.h5_schema import (
    H5_FACTOR_UNIVERSE,
    H5_MODEL_BLOCK_UNIVERSE,
    H5_PARAMETER_DEPENDENCY_ROWS,
    H5_RECOGNITION_COORDINATE_UNIVERSE,
    H5_VARIABLE_DEPENDENCY_ROWS,
)
from vfe4.types.updates import UpdateSpecification


DependencyRows = tuple[tuple[str, tuple[str, ...]], ...]


@dataclass(frozen=True)
class FactorDependencyGraph:
    factor_universe: tuple[str, ...]
    recognition_coordinate_universe: tuple[str, ...]
    model_block_universe: tuple[str, ...]
    variable_dependencies: DependencyRows
    parameter_dependencies: DependencyRows

    def __post_init__(self) -> None:
        for name, value in (
            ("factor_universe", self.factor_universe),
            ("recognition_coordinate_universe", self.recognition_coordinate_universe),
            ("model_block_universe", self.model_block_universe),
            ("variable_dependencies", self.variable_dependencies),
            ("parameter_dependencies", self.parameter_dependencies),
        ):
            if type(value) is not tuple:
                raise ValueError(f"{name} must be a tuple")
        if self.factor_universe != H5_FACTOR_UNIVERSE:
            raise ValueError("factor universe must equal the closed H5 universe")
        if self.recognition_coordinate_universe != H5_RECOGNITION_COORDINATE_UNIVERSE:
            raise ValueError("recognition universe must equal the closed H5 universe")
        if self.model_block_universe != H5_MODEL_BLOCK_UNIVERSE:
            raise ValueError("model-block universe must equal the closed H5 universe")
        _validate_rows(
            self.variable_dependencies,
            self.recognition_coordinate_universe,
            "variable_dependencies",
        )
        _validate_rows(
            self.parameter_dependencies,
            self.model_block_universe,
            "parameter_dependencies",
        )
        if self.variable_dependencies != H5_VARIABLE_DEPENDENCY_ROWS:
            raise ValueError("variable dependencies must equal the closed H5 graph")
        if self.parameter_dependencies != H5_PARAMETER_DEPENDENCY_ROWS:
            raise ValueError("parameter dependencies must equal the closed H5 graph")
        object.__setattr__(self, "factor_universe", tuple(self.factor_universe))
        object.__setattr__(
            self,
            "recognition_coordinate_universe",
            tuple(self.recognition_coordinate_universe),
        )
        object.__setattr__(self, "model_block_universe", tuple(self.model_block_universe))
        object.__setattr__(
            self,
            "variable_dependencies",
            tuple((identifier, tuple(factors)) for identifier, factors in self.variable_dependencies),
        )
        object.__setattr__(
            self,
            "parameter_dependencies",
            tuple((identifier, tuple(factors)) for identifier, factors in self.parameter_dependencies),
        )


def _validate_rows(rows: object, universe: tuple[str, ...], name: str) -> None:
    if type(rows) is not tuple or len(rows) != len(universe):
        raise ValueError(f"{name} must contain exactly one row per identifier")
    identifiers: list[str] = []
    for row_index, row in enumerate(rows):
        if type(row) is not tuple or len(row) != 2 or type(row[0]) is not str:
            raise ValueError(f"{name}[{row_index}] must be an identifier/factors pair")
        identifier, factors = row
        if type(factors) is not tuple or not factors:
            raise ValueError(f"{name}[{row_index}] factors must be a nonempty tuple")
        if any(type(factor) is not str or factor not in H5_FACTOR_UNIVERSE for factor in factors):
            raise ValueError(f"{name}[{row_index}] contains an unknown factor")
        if len(set(factors)) != len(factors):
            raise ValueError(f"{name}[{row_index}] contains duplicate factors")
        identifiers.append(identifier)
    if tuple(identifiers) != universe:
        raise ValueError(f"{name} must contain every identifier exactly once in universe order")


def build_h5_reference_dependency_graph(
    specification: UpdateSpecification,
) -> FactorDependencyGraph:
    if not isinstance(specification, UpdateSpecification):
        raise ValueError("specification must be an UpdateSpecification")
    if (
        specification.factor_universe != H5_FACTOR_UNIVERSE
        or specification.recognition_coordinate_universe
        != H5_RECOGNITION_COORDINATE_UNIVERSE
        or specification.model_block_universe != H5_MODEL_BLOCK_UNIVERSE
    ):
        raise ValueError("specification universes do not match the closed H5 graph")
    return FactorDependencyGraph(
        H5_FACTOR_UNIVERSE,
        H5_RECOGNITION_COORDINATE_UNIVERSE,
        H5_MODEL_BLOCK_UNIVERSE,
        H5_VARIABLE_DEPENDENCY_ROWS,
        H5_PARAMETER_DEPENDENCY_ROWS,
    )


def expected_affected_factors(
    graph: FactorDependencyGraph,
    *,
    variables: tuple[str, ...],
    parameters: tuple[str, ...],
) -> tuple[str, ...]:
    if not isinstance(graph, FactorDependencyGraph):
        raise ValueError("graph must be a FactorDependencyGraph")
    _validate_active_ids(
        variables,
        graph.recognition_coordinate_universe,
        "variables",
    )
    _validate_active_ids(parameters, graph.model_block_universe, "parameters")
    if not variables and not parameters:
        raise ValueError("an H5 update must activate at least one coordinate or block")
    variable_map = dict(graph.variable_dependencies)
    parameter_map = dict(graph.parameter_dependencies)
    affected = {
        factor
        for identifier in variables
        for factor in variable_map[identifier]
    }
    affected.update(
        factor
        for identifier in parameters
        for factor in parameter_map[identifier]
    )
    return tuple(factor for factor in graph.factor_universe if factor in affected)


def _validate_active_ids(value: object, universe: tuple[str, ...], name: str) -> None:
    if type(value) is not tuple:
        raise ValueError(f"{name} must be a tuple")
    if any(type(identifier) is not str or identifier not in universe for identifier in value):
        raise ValueError(f"{name} contains an unknown H5 identifier")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} contains duplicate identifiers")
    expected_order = tuple(identifier for identifier in universe if identifier in value)
    if value != expected_order:
        raise ValueError(f"{name} must use universe order")


__all__ = [
    "FactorDependencyGraph",
    "build_h5_reference_dependency_graph",
    "expected_affected_factors",
]
