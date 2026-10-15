"""Detached dashboard snapshots, population nodal means, and exact source rows."""

from dataclasses import dataclass
from math import prod
from operator import index

import numpy as np
import torch

from .model import SimulationModel
from .section_view import SectionSnapshot, capture_sections
from .state import SectionState


@dataclass(frozen=True)
class DashboardSnapshot(SectionSnapshot):
    """Section coefficients and source probabilities copied on their device."""

    beta: torch.Tensor


@dataclass(frozen=True)
class ReferenceImage:
    """Population nodal averages in coefficient coordinates, with support masks."""

    mean:   np.ma.MaskedArray
    truth:  np.ma.MaskedArray
    error:  np.ma.MaskedArray
    extent: tuple[float, float, float, float] | None
    z:      float | None


@dataclass(frozen=True)
class RoutingView:
    """Unmodified source probabilities for one receiver, channel, and context."""

    status:        str
    source_agents: np.ndarray
    probabilities: np.ndarray


def capture_dashboard(state: SectionState) -> DashboardSnapshot:
    """Capture section coefficients once, then detach and copy beta."""
    sections = capture_sections(state)
    return DashboardSnapshot(sections.sweep, sections.belief, sections.model, state.beta.detach().clone())


def _channel_number(channel: str) -> int:
    if channel not in ('belief', 'model'):
        raise ValueError('Channel must be belief or model')
    return 0 if channel == 'belief' else 1


def _nodal_average(
    values: torch.Tensor,
    model:  SimulationModel,
    plane:  int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Average selected incidence values: a(c)=sum_{s:c_s=c} v_s / n(c).

    In point mode the grouping key is the agent. For a 3D base only incidences
    on the requested nodal z plane contribute to the returned (Nx Ny, 2) array.
    """
    grid = model.domain.grid_shape
    nodes = model.incidence.context_id if grid else model.incidence.agent_id
    size = prod(grid[:2]) if grid else model.config['population']['n_agents']
    if len(grid) == 3:
        selected = nodes.remainder(grid[2]) == plane
        nodes, values = nodes[selected] // grid[2], values[selected]
    total = values.new_zeros(size, 2)
    count = values.new_zeros(size)
    total.index_add_(0, nodes, values)
    count.index_add_(0, nodes, torch.ones_like(nodes, dtype=values.dtype))
    return total / count.clamp_min(1)[:, None], count > 0


@torch.no_grad()
def reference_image(
    snapshot:    DashboardSnapshot,
    model:       SimulationModel,
    channel:     str,
    component:   int,
    slice_index: int = 0,
) -> ReferenceImage:
    r"""Compare population nodal coefficient means with diagnostic truth.

    These incidence averages are descriptive reference coordinates. For 3D,
    z=j L_z/(2 N_z); odd j uses periodic linear interpolation of adjacent nodal
    averages, requiring both planes' coverage. The error is mean minus truth.
    Only the selected plane values and support mask are transferred to CPU.
    """
    _channel_number(channel)
    component, slice_index = index(component), index(slice_index)
    if not 0 <= component < model.config['bundle']['K']:
        raise ValueError('Component is outside the configured fiber')
    grid = model.domain.grid_shape
    limit = 2 * grid[2] if len(grid) == 3 else 1
    if not 0 <= slice_index < limit:
        raise ValueError('Slice index is outside the displayed base')

    K, S = model.config['bundle']['K'], model.incidence.agent_id.numel()
    mean = getattr(snapshot, channel).means(S, K)[:, component]
    values = torch.stack((mean, model.truth[:, component]), -1)  # (S, 2), mean and truth
    plane = slice_index // 2
    averaged, covered = _nodal_average(values, model, plane)
    if len(grid) == 3 and slice_index % 2:
        adjacent, adjacent_covered = _nodal_average(values, model, (plane + 1) % grid[2])
        averaged = (averaged + adjacent) / 2
        covered = covered & adjacent_covered

    shape = grid[:2] if grid else (model.config['population']['n_agents'],)
    displayed = averaged.reshape(*shape, 2).cpu().numpy()
    mask = (~covered).reshape(shape).cpu().numpy()
    reference_mean = np.ma.array(displayed[..., 0], mask=mask)
    truth = np.ma.array(displayed[..., 1], mask=mask)
    extent, z = None, None
    if grid:
        periods = model.config['manifold']['periods']
        extent = (0.0, float(periods[0]), 0.0, float(periods[1]))
        if len(grid) == 3:
            z = slice_index * float(periods[2]) / (2 * grid[2])
    return ReferenceImage(reference_mean, truth, reference_mean - truth, extent, z)


@torch.no_grad()
def routing_view(
    snapshot:   DashboardSnapshot,
    model:      SimulationModel,
    agent:      int,
    channel:    str,
    context_id: int,
) -> RoutingView:
    """Extract the actual beta row, including self, without pooling or normalization."""
    agent, context_id = index(agent), index(context_id)
    channel_number = _channel_number(channel)
    if not 0 <= agent < model.config['population']['n_agents']:
        raise ValueError('Agent is outside the configured population')
    if not 0 <= context_id < model.domain.contexts.shape[0]:
        raise ValueError('Context is outside the finite design')

    empty = RoutingView('channel_disabled', np.empty(0, dtype=np.int64),
                        snapshot.beta.new_empty(0).cpu().numpy())
    if channel not in model.config['communication']['channels']:
        return empty
    incidence = model.incidence
    receiver = ((incidence.agent_id == agent) & (incidence.context_id == context_id)).nonzero().flatten()
    if not receiver.numel():
        return RoutingView('agent_absent', empty.source_agents, empty.probabilities)
    rows = model.rows
    selected = ((rows.receiver_incidence == receiver[0]) &
                (rows.channel[rows.row_id] == channel_number)).nonzero().flatten()
    if not selected.numel():
        return RoutingView('no_candidates', empty.source_agents, empty.probabilities)
    sources = incidence.agent_id[rows.source_incidence[selected]].cpu().numpy()
    probabilities = snapshot.beta[selected].cpu().numpy()
    return RoutingView('active', sources, probabilities)
