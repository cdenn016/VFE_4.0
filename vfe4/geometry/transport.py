"""Frame-induced base transport and independent interaction twists are separate maps."""

import torch

from ..config import SimulationConfig
from ..state import Domain
from .groups import GLReal


class ConstantFrameField:
    def __init__(self, seed_frames: torch.Tensor) -> None:
        self.seed_frames = seed_frames

    def evaluate(self, agent_ids: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        return self.seed_frames[agent_ids]


class FourierFrameField(ConstantFrameField):
    def __init__(self, seed_frames: torch.Tensor, coefficients: torch.Tensor, periods: torch.Tensor) -> None:
        super().__init__(seed_frames)
        self.coefficients, self.periods = coefficients, periods

    def evaluate(self, agent_ids: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        phase = torch.sin(2 * torch.pi * points / self.periods)
        algebra = (phase[..., :, None, None] * self.coefficients[agent_ids]).sum(-3)
        return self.seed_frames[agent_ids] @ torch.linalg.matrix_exp(algebra)


class FlatReferenceConnection:
    def reference_transport(self, path: tuple[torch.Tensor, torch.Tensor], channel: str,
                            K: int) -> torch.Tensor:
        source = path[0]
        return torch.eye(K, device=source.device, dtype=source.dtype).expand(*source.shape[:-1], K, K)


def build_frame_field(domain: Domain, config: SimulationConfig) -> ConstantFrameField:
    N, K = config['population']['n_agents'], config['bundle']['K']
    generator = torch.Generator(device=config.device).manual_seed(config['runtime']['seed'] + 1)
    seeds = torch.linalg.matrix_exp(config['bundle']['frame_spread'] * torch.randn(
        N, K, K, device=config.device, dtype=config.dtype, generator=generator))
    if config['bundle']['frame_field'] == 'constant' or not domain.grid_shape:
        return ConstantFrameField(seeds)
    coefficients = config['bundle']['frame_amplitude'] * torch.randn(
        N, len(domain.grid_shape), K, K, device=config.device, dtype=config.dtype, generator=generator)
    return FourierFrameField(seeds, coefficients, domain.periods)


def base_transport(U_target: torch.Tensor, U_source: torch.Tensor) -> torch.Tensor:
    r"""Omega(c'<-c)=U(c')U(c)^{-1} for A0=0, with A=-dU U^{-1}."""
    return GLReal(U_source.shape[-1]).ratio(U_target, U_source)


def interaction_transport(U_receiver: torch.Tensor, U_sender: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    r"""Theta_e=U_i exp(omega_e) U_j^{-1}; opposite receiving rows own different omega."""
    return GLReal(omega.shape[-1]).ratio(U_receiver @ torch.linalg.matrix_exp(omega), U_sender)
