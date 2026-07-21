"""Immutable structural records for the fixed H1 fixture."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch


ParentSets = tuple[tuple[int, ...], tuple[int, ...]]


@dataclass(frozen=True)
class StructuralData:
    horizon: Literal[2]
    d_z: Literal[1]
    d_m: Literal[1]
    vocabulary_size: Literal[3]
    state_parent_sets: ParentSets
    model_parent_sets: ParentSets
    state_source_support: ParentSets
    model_source_support: ParentSets

    def __post_init__(self) -> None:
        for name, value, expected in (
            ("horizon", self.horizon, 2),
            ("d_z", self.d_z, 1),
            ("d_m", self.d_m, 1),
            ("vocabulary_size", self.vocabulary_size, 3),
        ):
            if type(value) is not int or value != expected:
                raise ValueError(f"{name} must equal {expected}")
        _require_parent_sets(self.state_parent_sets, "state_parent_sets")
        _require_parent_sets(self.model_parent_sets, "model_parent_sets")
        _require_support(
            self.state_source_support, self.state_parent_sets, "state_source_support"
        )
        _require_support(
            self.model_source_support, self.model_parent_sets, "model_source_support"
        )


@dataclass(frozen=True, init=False)
class PopulationFrames:
    """Three fixed scalar population frames used only by the H1 fixture."""

    _values: torch.Tensor = field(repr=False, compare=False)

    def __init__(self, values: torch.Tensor) -> None:
        if not isinstance(values, torch.Tensor):
            raise ValueError("values must be a torch.Tensor")
        if values.dtype is not torch.float64:
            raise ValueError("values must use float64")
        if values.ndim != 1 or values.shape != (3,):
            raise ValueError("values must have shape (3,)")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("values must be finite")
        if bool(torch.eq(values, 0).any()):
            raise ValueError("values must be nonzero")
        object.__setattr__(self, "_values", values.detach().clone())

    @property
    def values(self) -> torch.Tensor:
        """Return an owned copy so callers cannot mutate the frozen record."""
        return self._values.detach().clone()

    def omega(self, receiver: int, source: int) -> torch.Tensor:
        _require_frame_index(receiver, "receiver")
        _require_frame_index(source, "source")
        ratio = self._values[receiver] / self._values[source]
        if not bool(torch.isfinite(ratio)):
            raise ValueError("omega must be finite")
        return ratio.detach().clone()


@dataclass(frozen=True)
class SourcePath:
    a: tuple[int, int]
    b: tuple[int, int]

    def __post_init__(self) -> None:
        _require_coordinate(self.a, "a")
        _require_coordinate(self.b, "b")


def _require_parent_sets(value: object, name: str) -> None:
    if type(value) is not tuple or len(value) != 2:
        raise ValueError(f"{name} must contain two parent tuples")
    for time, parents in enumerate(value):
        if type(parents) is not tuple or not parents:
            raise ValueError(f"{name}[{time}] must be a nonempty tuple")
        if any(
            type(parent) is not int or parent < 0 or parent > time
            for parent in parents
        ):
            raise ValueError(f"{name}[{time}] contains an out-of-range parent")
        if len(set(parents)) != len(parents):
            raise ValueError(f"{name}[{time}] contains duplicate parents")


def _require_support(value: object, parents: ParentSets, name: str) -> None:
    _require_parent_sets(value, name)
    for time, sources in enumerate(value):
        if not set(sources).issubset(parents[time]):
            raise ValueError(f"{name}[{time}] is outside its parent support")


def _require_frame_index(value: int, name: str) -> None:
    if type(value) is not int or value < 0 or value >= 3:
        raise ValueError(f"{name} index must be in [0, 3)")


def _require_coordinate(value: object, name: str) -> None:
    if type(value) is not tuple or len(value) != 2:
        raise ValueError(f"{name} must be a pair")
    if any(type(item) is not int or item < 0 for item in value):
        raise ValueError(f"{name} must contain nonnegative integer coordinates")
