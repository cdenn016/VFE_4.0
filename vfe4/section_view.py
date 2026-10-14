"""Detached coefficient snapshots and selected agent-frame section images."""

from dataclasses import dataclass
from operator import index

import numpy as np
import torch

from .geometry.sections import evaluate_section
from .model import SimulationModel
from .state import GaussianBucket, PatchGaussians, SectionState


@dataclass(frozen=True)
class SectionSnapshot:
    """Frozen sweep metadata and detached copies of both coefficient laws."""

    sweep:  int
    belief: PatchGaussians
    model:  PatchGaussians


@dataclass(frozen=True)
class SectionImage:
    """Selected scalar values, explicit support mask, and physical plane bounds."""

    values: np.ma.MaskedArray
    extent: tuple[float, float, float, float] | None
    z:      float | None


def _copy_laws(laws: PatchGaussians) -> PatchGaussians:
    """Copy coefficient tensors on their device while sharing immutable IDs."""
    return PatchGaussians(tuple(GaussianBucket(
        bucket.agent_ids, bucket.incidence_ids,
        bucket.mu.detach().clone(), bucket.chol.detach().clone(),
    ) for bucket in laws.buckets), laws.covariance_structure)


def capture_sections(state: SectionState) -> SectionSnapshot:
    """Capture coefficients without copying links, sampling, or moving to CPU."""
    return SectionSnapshot(state.sweep, _copy_laws(state.belief), _copy_laws(state.model))


@torch.no_grad()
def section_image(
    snapshot:    SectionSnapshot,
    model:       SimulationModel,
    agent:       int,
    channel:     str,
    component:   int,
    statistic:   str,
    slice_index: int = 0,
) -> SectionImage:
    r"""Display mu_a(c) or sqrt(Sigma_aa(c)) of one agent's finite section.

    The existing evaluator pushes the complete coefficient covariance through
    the basis and frame, Sigma(c)=U(c) E(c) Sigma E(c)^T U(c)^T. Spatial image
    axes use doubled-grid cell centers; a 3D plane has z=j L_z/(2 N_z).
    Unsupported values are masked. Only selected scalars and the mask reach CPU.
    """
    agent, component, slice_index = index(agent), index(component), index(slice_index)
    if not 0 <= agent < model.config['population']['n_agents']:
        raise ValueError('Agent is outside the configured population')
    if channel not in ('belief', 'model'):
        raise ValueError('Channel must be belief or model')
    if not 0 <= component < model.config['bundle']['K']:
        raise ValueError('Component is outside the configured fiber')
    if statistic not in ('mean', 'std'):
        raise ValueError('Statistic must be mean or std')
    grid = model.domain.grid_shape
    limit = 2 * grid[2] if len(grid) == 3 else 1
    if not 0 <= slice_index < limit:
        raise ValueError('Slice index is outside the displayed base')

    coefficient = getattr(snapshot, channel).buckets[0].mu
    device, dtype = coefficient.device, coefficient.dtype
    extent, z = None, None
    if not grid:
        points = coefficient.new_empty(1, 0)
        image_shape = (1,)
    else:
        periods = model.config['manifold']['periods']
        image_shape = (2 * grid[0], 2 * grid[1])
        axes = [(torch.arange(size, device=device, dtype=dtype) + 0.5) * periods[a] / size
                for a, size in enumerate(image_shape)]
        points = torch.stack(torch.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 2)
        extent = (0.0, float(periods[0]), 0.0, float(periods[1]))
        if len(grid) == 3:
            z = slice_index * float(periods[2]) / (2 * grid[2])
            points = torch.cat((points, points.new_full((points.shape[0], 1), z)), -1)
    agents = torch.full((points.shape[0],), agent, device=device, dtype=torch.long)
    mean, covariance, present = evaluate_section(snapshot, model, agents, points, channel)
    selected = mean[:, component] if statistic == 'mean' else covariance[:, component, component].sqrt()
    values = selected.reshape(image_shape).detach().cpu().numpy()
    mask = (~present).reshape(image_shape).detach().cpu().numpy()
    return SectionImage(np.ma.array(values, mask=mask), extent, z)
