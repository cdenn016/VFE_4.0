"""Structural source priors and independently oriented interaction copies."""

from math import prod

import torch

from ..config import SimulationConfig
from ..gaussian import ragged_log_softmax
from ..state import CandidateRows, CopyTable, Domain, Incidence, LoopTable, SourceSupport


def _support(receiver: torch.Tensor, source: torch.Tensor, context: torch.Tensor,
             rec_inc: torch.Tensor, src_inc: torch.Tensor, weights: torch.Tensor, S: int) -> SourceSupport:
    order = torch.argsort(rec_inc, stable=True)
    receiver, source, context, rec_inc, src_inc, weights = (x[order] for x in
        (receiver, source, context, rec_inc, src_inc, weights))
    lengths = torch.bincount(rec_inc, minlength=S)
    offsets = torch.cat([lengths.new_zeros(1), lengths.cumsum(0)])
    return SourceSupport(receiver, source, context, ragged_log_softmax(weights, offsets), offsets,
                         rec_inc, src_inc, weights, S)


def agent_lattice_support(n_agents: int, lattice_shape: tuple[int, ...], family: str,
                          radius: int | None, decay: float | None, alpha: float | None,
                          device: torch.device | str = 'cpu', dtype: torch.dtype = torch.float64) -> SourceSupport:
    r"""Source locality at c* is a prior on the agent set, with toroidal Manhattan distance."""
    if prod(lattice_shape) != n_agents:
        raise ValueError('lattice_shape must multiply to n_agents')
    axes = [torch.arange(n, device=device) for n in lattice_shape]
    coords = torch.stack(torch.meshgrid(*axes, indexing='ij'), -1).reshape(n_agents, -1)
    size = torch.tensor(lattice_shape, device=device)
    delta = torch.remainder(coords[None, :, :] - coords[:, None, :] + size // 2, size) - size // 2
    distance = delta.abs().sum(-1)
    mask = distance > 0
    if family in ('window', 'directed_window'):
        if radius is None or radius < 1:
            raise ValueError('window radius must be positive')
        mask &= distance <= radius
    elif family not in ('complete', 'exponential', 'power'):
        raise ValueError(f'Unsupported prior_family {family}')
    if family == 'directed_window':
        first = (delta != 0).long().argmax(-1)
        direction = delta.gather(-1, first.unsqueeze(-1)).squeeze(-1)
        mask &= direction > 0
    receiver, source = mask.nonzero(as_tuple=True)
    dist = distance[receiver, source].to(dtype)
    weights = torch.zeros_like(dist)
    if family == 'exponential':
        weights = -(1.0 if decay is None else decay) * dist
    if family == 'power':
        weights = -(2.0 if alpha is None else alpha) * dist.log()
    return _support(receiver, source, torch.zeros_like(receiver), receiver, source, weights, n_agents)


def overlap_support(incidence: Incidence, domain: Domain, config: SimulationConfig) -> SourceSupport:
    r"""Build actual shared-context candidates in O(S+E), without N*N*M storage."""
    context, agent = incidence.context_id, incidence.agent_id
    S, M = context.numel(), domain.contexts.shape[0]
    counts = torch.bincount(context, minlength=M)
    lengths = counts[context] - 1
    rec_inc = torch.arange(S, device=context.device).repeat_interleave(lengths)
    starts = torch.cat([lengths.new_zeros(1), lengths.cumsum(0)])
    local = torch.arange(rec_inc.numel(), device=context.device) - starts[rec_inc]
    order = torch.argsort(context, stable=True)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(S, device=context.device)
    group_start = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
    own = rank[rec_inc] - group_start[context[rec_inc]]
    source_rank = group_start[context[rec_inc]] + local + (local >= own)
    src_inc = order[source_rank]
    distance = domain.manifold.distance(incidence.centers[agent[rec_inc]], incidence.centers[agent[src_inc]])
    comm = config['communication']
    family = comm['prior_family']
    weights = torch.zeros_like(distance)
    if family == 'exponential':
        weights = -comm['decay'] * distance
    elif family == 'power':
        weights = -comm['alpha'] * torch.log1p(distance / domain.manifold.spacing.min())
    elif family in ('window', 'directed_window'):
        keep = distance <= comm['radius'] * domain.manifold.spacing.min()
        if family == 'directed_window':
            delta = domain.manifold.displacement(incidence.centers[agent[src_inc]], incidence.centers[agent[rec_inc]])
            axis = (delta != 0).long().argmax(-1)
            keep &= delta.gather(-1, axis[:, None]).squeeze(-1) > 0
        rec_inc, src_inc, weights, distance = (x[keep] for x in (rec_inc, src_inc, weights, distance))
        lengths = torch.bincount(rec_inc, minlength=S)
        starts = torch.cat([lengths.new_zeros(1), lengths.cumsum(0)])
    elif family != 'complete':
        raise ValueError(f'Unsupported prior_family {family}')
    maximum = config['communication']['max_sources']
    if maximum is not None:
        if maximum < 1:
            raise ValueError('max_sources must be positive')
        ranked = torch.argsort(distance, stable=True)
        ranked = ranked[torch.argsort(rec_inc[ranked], stable=True)]
        slots = torch.arange(ranked.numel(), device=context.device) - starts[rec_inc[ranked]]
        keep = ranked[slots < maximum]
        rec_inc, src_inc, weights = rec_inc[keep], src_inc[keep], weights[keep]
    support = _support(agent[rec_inc], agent[src_inc], context[rec_inc], rec_inc, src_inc, weights, S)
    support.incidence_agent, support.incidence_context = agent, context
    return support


def build_copy_table(support: SourceSupport, include_self: bool) -> tuple[CandidateRows, CopyTable, LoopTable]:
    rec, src = support.receiver_incidence, support.source_incidence
    receiver, sender, context, weights = support.receiver_agent, support.source_agent, support.context_id, support.log_weights
    S, device = support.n_incidence, rec.device
    if include_self:
        self_inc = torch.arange(S, device=device)
        # Each support row represents one incidence; infer its agent/context from a row or point fallback.
        row_agent = torch.arange(S, device=device)
        row_context = torch.zeros(S, device=device, dtype=torch.long)
        if rec.numel():
            row_agent[rec], row_context[rec] = receiver, context
        if support.incidence_agent is not None:
            row_agent, row_context = support.incidence_agent, support.incidence_context
        rec, src = torch.cat([rec, self_inc]), torch.cat([src, self_inc])
        receiver, sender = torch.cat([receiver, row_agent]), torch.cat([sender, row_agent])
        context, weights = torch.cat([context, row_context]), torch.cat([weights, weights.new_zeros(S)])
    order = torch.argsort(rec, stable=True)
    rec, src, receiver, sender, context, weights = (x[order] for x in (rec, src, receiver, sender, context, weights))
    lengths = torch.bincount(rec, minlength=S)
    if torch.any(lengths == 0):
        raise ValueError('Empty source alphabet; include_self or provide a nonempty support')
    offsets = torch.cat([lengths.new_zeros(1), lengths.cumsum(0)])
    self_mask = rec == src
    primitives = (~self_mask).long().cumsum(0) - 1
    primitives[self_mask] = -1
    C = rec.numel()
    copy = CopyTable(receiver, sender, context, primitives, torch.ones_like(rec), self_mask)
    rows = CandidateRows(offsets, rec, src, torch.arange(C, device=device), ragged_log_softmax(weights, offsets),
                         torch.arange(S, device=device).repeat_interleave(lengths), torch.zeros(S, device=device, dtype=torch.long))
    keys = rec * S + src
    sorted_key, sorted_index = keys.sort()
    reverse_pos = torch.searchsorted(sorted_key, src * S + rec)
    safe = reverse_pos.clamp_max(C - 1)
    valid = (reverse_pos < C) & (sorted_key[safe] == src * S + rec) & ~self_mask
    first = primitives[valid]
    second = primitives[sorted_index[safe[valid]]]
    loops = torch.stack([first, second], -1)
    loop = LoopTable(torch.arange(first.numel() + 1, device=device) * 2, loops.flatten(),
                     torch.ones(first.numel() * 2, device=device, dtype=torch.long), rec[valid], weights.new_ones(first.numel()))
    return rows, copy, loop
