"""Finite random sections from smooth cardinal local coefficients (B1 finite design)."""

from math import prod

import torch

from ..config import SimulationConfig
from ..state import Domain, GaussianBucket, Incidence, PatchGaussians


def build_incidence(domain: Domain, config: SimulationConfig) -> Incidence:
    N, population = config['population']['n_agents'], config['population']
    device, dtype = config.device, config.dtype
    if not domain.grid_shape:
        ids = torch.arange(N, device=device)
        return Incidence(ids, torch.zeros_like(ids), torch.arange(N + 1, device=device),
                         torch.zeros_like(ids), torch.empty(N, 0, device=device, dtype=dtype))
    shape = domain.grid_shape
    d = len(shape)
    if population['placement'] == 'grid':
        axes = [torch.arange(n, device=device) for n in population['center_grid']]
        lattice = torch.stack(torch.meshgrid(*axes, indexing='ij'), -1).reshape(N, d)
        center_index = torch.floor(lattice * torch.tensor(shape, device=device) /
                                   torch.tensor(population['center_grid'], device=device)).long()
    else:
        rng = torch.Generator(device=device).manual_seed(config['runtime']['seed'])
        center_index = torch.stack([torch.randint(n, (N,), generator=rng, device=device) for n in shape], -1)
    axes = [torch.arange(-(w // 2), w // 2 + 1, device=device) for w in population['patch_shape']]
    offsets = torch.stack(torch.meshgrid(*axes, indexing='ij'), -1).reshape(-1, d)
    P = offsets.shape[0]
    nodes = torch.remainder(center_index[:, None, :] + offsets[None, :, :], torch.tensor(shape, device=device))
    strides = torch.tensor([prod(shape[a + 1:]) for a in range(d)], device=device)
    context = (nodes * strides).sum(-1).long().flatten()
    covered = torch.bincount(context, minlength=domain.contexts.shape[0])
    if population['coverage'] == 'required' and torch.any(covered == 0):
        raise ValueError('Agent patches leave uncovered design points; choose different placement or allow_uncovered')
    centers = center_index * domain.manifold.spacing
    return Incidence(torch.arange(N, device=device).repeat_interleave(P), context,
                     torch.arange(N + 1, device=device) * P,
                     torch.arange(P, device=device).repeat(N), centers.to(dtype))


def basis_weights(
    domain:    Domain,
    incidence: Incidence,
    agent_ids: torch.Tensor,
    points:    torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""phi_r=psi_r/sum_s psi_s with C-infinity bumps of scaled radius 0.95."""
    if not domain.grid_shape:
        return points.new_ones(points.shape[0], 1), torch.ones(points.shape[0], device=points.device, dtype=torch.bool)
    counts = incidence.patch_offset.diff()[agent_ids]
    if not torch.all(counts == counts[0]):
        raise ValueError('Evaluate differing patch sizes in separate buckets')
    P = int(counts[0])
    ids = incidence.patch_offset[agent_ids, None] + torch.arange(P, device=agent_ids.device)
    knots = domain.contexts[incidence.context_id[ids]]
    delta = domain.manifold.displacement(knots, points[:, None, :]) / domain.manifold.spacing
    r2 = delta.square().sum(-1) / 0.95**2
    inside = r2 < 1
    # Log normalization avoids underflow close to the boundary of the open support.
    log_bump = torch.where(inside, -1 / (1 - r2).clamp_min(torch.finfo(points.dtype).tiny), -torch.inf)
    present = inside.any(-1)
    safe = torch.where(present[:, None], log_bump, torch.zeros_like(log_bump))
    weights = torch.softmax(safe, -1) * present[:, None]
    return weights, present


def evaluate_gaussian_coefficients(
    mu_coeff:   torch.Tensor,
    chol_coeff: torch.Tensor,
    weights:    torch.Tensor,
    frame:      torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Push forward the entire coefficient Gaussian: Sigma(c)=U E Sigma E^T U^T."""
    K = frame.shape[-1]
    eye = torch.eye(K, device=frame.device, dtype=frame.dtype)
    evaluation = (weights[..., :, None, None] * eye).transpose(-3, -2).reshape(*weights.shape[:-1], K, -1)
    mean = (evaluation @ mu_coeff.unsqueeze(-1)).squeeze(-1)
    if chol_coeff.ndim == weights.ndim + 2:  # (...,P,K,K) node-block factors
        blocks = chol_coeff @ chol_coeff.mT
        covariance = (weights.square()[..., :, None, None] * blocks).sum(-3)
    else:
        pushed_chol = evaluation @ chol_coeff
        covariance = pushed_chol @ pushed_chol.mT
    return (frame @ mean.unsqueeze(-1)).squeeze(-1), frame @ covariance @ frame.mT


def initialize_gaussians(incidence: Incidence, config: SimulationConfig, std: float) -> PatchGaussians:
    K = config['bundle']['K']
    buckets = []
    counts = incidence.patch_offset.diff()
    for P in counts.unique().tolist():
        agents = (counts == P).nonzero().flatten()
        ids = incidence.patch_offset[agents, None] + torch.arange(P, device=config.device)
        mu = torch.zeros(agents.numel(), P * K, device=config.device, dtype=config.dtype)
        if config['recognition']['covariance_structure'] == 'full_patch':
            L = torch.eye(P * K, device=config.device, dtype=config.dtype).expand(agents.numel(), -1, -1).clone() * std
        else:
            L = torch.eye(K, device=config.device, dtype=config.dtype).expand(agents.numel(), P, -1, -1).clone() * std
        buckets.append(GaussianBucket(agents, ids, mu, L))
    return PatchGaussians(tuple(buckets), config['recognition']['covariance_structure'])


def evaluate_section(state: object, model: object, agent_ids: torch.Tensor,
                     points: torch.Tensor, channel: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    laws = getattr(state, channel)
    weights, present = basis_weights(model.domain, model.incidence, agent_ids, points)
    K = model.config['bundle']['K']
    mu, covariance = points.new_zeros(points.shape[0], K), points.new_zeros(points.shape[0], K, K)
    for bucket in laws.buckets:
        selected = agent_ids[:, None] == bucket.agent_ids[None, :]
        wanted = selected.any(-1)
        indices = selected[wanted].long().argmax(-1)
        if wanted.any():
            frame = model.frame_field.evaluate(agent_ids[wanted], points[wanted])
            mu[wanted], covariance[wanted] = evaluate_gaussian_coefficients(
                bucket.mu[indices], bucket.chol[indices], weights[wanted], frame)
    return mu, covariance, present
