"""Flat contextual bases; adapted from MAgent_Model Torus geometry."""

from math import prod
from typing import Protocol

import torch

from ..config import SimulationConfig
from ..state import Domain


class BaseManifold(Protocol):
    def grid(self) -> torch.Tensor: ...
    def wrap(self, points: torch.Tensor) -> torch.Tensor: ...
    def displacement(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor: ...
    def volume_weights(self) -> torch.Tensor: ...


class PointBase:
    def __init__(self, device: torch.device | str = 'cpu', dtype: torch.dtype = torch.float64) -> None:
        self.device, self.dtype = device, dtype

    def grid(self) -> torch.Tensor:
        return torch.empty((1, 0), device=self.device, dtype=self.dtype)

    def wrap(self, points: torch.Tensor) -> torch.Tensor:
        return points

    def displacement(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return target - source

    def volume_weights(self) -> torch.Tensor:
        return torch.ones(1, device=self.device, dtype=self.dtype)


class FlatTorus:
    def __init__(self, periods: tuple[float, ...], grid_shape: tuple[int, ...],
                 device: torch.device | str = 'cpu', dtype: torch.dtype = torch.float64) -> None:
        self.periods = torch.tensor(periods, device=device, dtype=dtype)
        self.grid_shape, self.device, self.dtype = tuple(grid_shape), device, dtype
        self.spacing = self.periods / torch.tensor(grid_shape, device=device, dtype=dtype)

    def grid(self) -> torch.Tensor:
        axes = [torch.arange(n, device=self.device, dtype=self.dtype) * self.spacing[a]
                for a, n in enumerate(self.grid_shape)]
        return torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1).reshape(-1, len(axes))

    def wrap(self, points: torch.Tensor) -> torch.Tensor:
        return torch.remainder(points, self.periods)

    def displacement(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""Minimum-image displacement in the flat torus: delta in [-L/2,L/2)."""
        return torch.remainder(target - source + self.periods / 2, self.periods) - self.periods / 2

    def distance(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(self.displacement(source, target), dim=-1)

    def volume_weights(self) -> torch.Tensor:
        return torch.ones(prod(self.grid_shape), device=self.device, dtype=self.dtype) * self.spacing.prod()

    def path(self, source: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return source, source + self.displacement(source, target)


def build_domain(config: SimulationConfig) -> Domain:
    values = config['manifold']
    manifold = (PointBase(config.device, config.dtype) if values['kind'] == 'point' else
                FlatTorus(values['periods'], values['grid_shape'], config.device, config.dtype))
    return Domain(manifold.grid(), torch.tensor(values['periods'], device=config.device, dtype=config.dtype),
                  tuple(values['grid_shape']), manifold.volume_weights(), manifold)
