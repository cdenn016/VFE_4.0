"""Synthetic truth is an observation fixture, never an inference prior."""

import torch

from .config import SimulationConfig
from .state import Domain, Incidence


def synthetic_truth(domain: Domain, incidence: Incidence, config: SimulationConfig,
                    generator: torch.Generator) -> torch.Tensor:
    K = config['bundle']['K']
    if not domain.grid_shape:
        return torch.randn(incidence.agent_id.numel(), K, device=config.device,
                           dtype=config.dtype, generator=generator)
    d = len(domain.grid_shape)
    coefficients = torch.randn(d, K, device=config.device, dtype=config.dtype, generator=generator)
    coefficients = coefficients / max(d, 1)**0.5
    phase = 2 * torch.pi * domain.contexts / domain.periods
    field = torch.sin(phase) @ coefficients + 0.25 * torch.cos(2 * phase) @ coefficients
    return field[incidence.context_id]
