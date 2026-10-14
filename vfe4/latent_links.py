"""Gaussian generator recognition using the projected expected true Hessian."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from functools import partial

import torch

from .state import LinkState, SectionState

if TYPE_CHECKING:
    from .model import SimulationModel


def gaussian_blr_step(mu: torch.Tensor, precision: torch.Tensor, gradient: torch.Tensor,
                      hessian: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    r"""P_new=(1-dt)P+dt H; mu_new=mu-dt P_new^{-1}g, c08b Eq. BLR."""
    new_precision = (1 - dt) * precision + dt * hessian
    if precision.ndim == mu.ndim:
        if not bool((new_precision > 0).all()):
            raise ValueError('Projected generator precision is not positive')
        displacement = gradient / new_precision
    else:
        L, info = torch.linalg.cholesky_ex(new_precision)
        if bool((info != 0).any()):
            raise ValueError('Projected generator precision is not SPD')
        displacement = torch.cholesky_solve(gradient[..., None], L).squeeze(-1)
    return mu - dt * displacement, new_precision


def record_integrand(omega: torch.Tensor, z: torch.Tensor, mu_sender: torch.Tensor,
                     Sigma_sender: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    r"""1/2 E[(z-exp(Omega)k)^T G(z-exp(Omega)k)] in Frobenius coordinates."""
    K = mu_sender.numel()
    T = torch.matrix_exp(omega.reshape(K, K))
    delta = z - T @ mu_sender
    return 0.5 * (delta @ G @ delta + torch.trace(G @ T @ Sigma_sender @ T.mT))


def derivatives(omega: torch.Tensor, z: torch.Tensor, mu_sender: torch.Tensor,
                Sigma_sender: torch.Tensor, G: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gradient = torch.func.jacrev(record_integrand, argnums=0)
    hessian = torch.func.jacrev(gradient, argnums=0)
    with torch.enable_grad():
        return (torch.func.vmap(gradient)(omega, z, mu_sender, Sigma_sender, G),
                torch.func.vmap(hessian)(omega, z, mu_sender, Sigma_sender, G))


def refresh_samples(links: LinkState, samples: int) -> None:
    """Advance the checkpointed integration stream once, using antithetic pairs."""
    if samples < 2 or samples % 2:
        raise ValueError('links.samples must be a positive even number >= 2')
    generator = torch.Generator(device=links.omega_mean.device)
    generator.set_state(links.rng_state)
    normals = torch.randn(samples // 2, *links.omega_mean.shape, generator=generator,
                          device=links.omega_mean.device, dtype=links.omega_mean.dtype)
    links.standard_normals = torch.cat((normals, -normals))
    links.rng_state = generator.get_state()


def sampled_generators(links: LinkState, samples: int) -> torch.Tensor:
    if links.standard_normals is None:
        refresh_samples(links, samples)
    if links.precision.ndim == 2:
        perturbation = links.standard_normals / links.precision.sqrt()
    else:
        L = torch.linalg.cholesky(links.precision)
        perturbation = torch.linalg.solve_triangular(L.mT, links.standard_normals[..., None], upper=True).squeeze(-1)
    return links.omega_mean + perturbation


def link_kl(links: LinkState, prior_std: float) -> torch.Tensor:
    r"""KL(prod_e N(mu_e,P_e^{-1}) || prod_e N(0,sigma_prior^2 I))."""
    D = links.omega_mean.shape[-1]
    if links.precision.ndim == 2:
        trace = links.precision.reciprocal().sum(-1)
        logdet = links.precision.log().sum(-1)
    else:
        L = torch.linalg.cholesky(links.precision)
        trace = torch.cholesky_inverse(L).diagonal(dim1=-2, dim2=-1).sum(-1)
        logdet = 2 * L.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    return 0.5 * ((trace + links.omega_mean.square().sum(-1)) / prior_std**2 - D
                  + D * math.log(prior_std**2) + logdet).sum()


def loop_integrand(pair: torch.Tensor, K: int) -> torch.Tensor:
    """One based mutual loop, with distinct independently attending generators."""
    T = torch.matrix_exp(pair.reshape(2, K, K))
    defect = torch.eye(K, device=pair.device, dtype=pair.dtype) - T[0] @ T[1]
    return 0.5 * defect.square().sum()


def loop_sample_costs(links: LinkState, model: SimulationModel) -> torch.Tensor:
    K = model.config['bundle']['K']
    omega = sampled_generators(links, model.config['links']['samples'])
    samples = omega.shape[0]
    result = omega.new_zeros(samples)
    if not model.loops.weight.numel():
        return result
    pairs = model.loops.primitive_id.reshape(-1, 2)
    repeats = 1 if model.config['links']['shared_channels'] else 2
    copies = omega.shape[1] // repeats
    chunk = model.config['links']['copy_chunk_size']
    for channel in range(repeats):
        for start in range(0, len(pairs), chunk):
            ids = pairs[start:start + chunk] + channel * copies
            T = torch.matrix_exp(omega[:, ids].reshape(samples, -1, 2, K, K))
            defect = torch.eye(K, device=omega.device, dtype=omega.dtype) - T[:, :, 0] @ T[:, :, 1]
            result += (0.5 * defect.square().sum((-1, -2)) * model.loops.weight[start:start + chunk]).sum(-1)
    # The same holonomy penalty acts on both associated channels when links are shared.
    return result * (2 if repeats == 1 else 1) * model.config['links']['mutual_loop_coefficient']


def independent_diagnostics(state: SectionState, model: SimulationModel) -> dict[str, float]:
    """Independent antithetic estimate; uncertainty is computed across pair means."""
    from .objective import evaluate_vfe
    links = state.links
    if links.mode != 'latent_gaussian':
        return {}
    samples = model.config['links']['samples']
    generator = torch.Generator(device=links.omega_mean.device)
    if links.diagnostic_rng_state is None:
        generator.manual_seed(model.config['runtime']['seed'] + 101)
    else:
        generator.set_state(links.diagnostic_rng_state)
    normals = torch.randn(samples // 2, *links.omega_mean.shape, generator=generator,
                          device=links.omega_mean.device, dtype=links.omega_mean.dtype)
    normals = torch.cat((normals, -normals))
    links.diagnostic_rng_state = generator.get_state()
    saved = links.standard_normals
    values = []
    try:
        for sample in normals:
            links.standard_normals = sample.unsqueeze(0)
            values.append(evaluate_vfe(state, model).total)
    finally:
        links.standard_normals = saved
    estimates = torch.stack(values)
    pair_mean = (estimates[:samples // 2] + estimates[samples // 2:]) * 0.5
    error = float(pair_mean.std(unbiased=True) / math.sqrt(samples // 2)) if samples >= 4 else math.nan
    return {'diagnostic_vfe_mean': float(estimates.mean()), 'diagnostic_vfe_standard_error': error,
            'diagnostic_independent_pairs': samples // 2}


def update_links(state: SectionState, model: SimulationModel) -> int:
    """Project the true expected Hessian to per-copy Gaussian blocks, then admit by VFE."""
    from .objective import evaluate_vfe
    config, links = model.config, state.links
    C, D = links.omega_mean.shape
    if not C:
        return 0
    K, S = config['bundle']['K'], model.incidence.agent_id.numel()
    omega = sampled_generators(links, config['links']['samples'])
    sample_count = omega.shape[0]
    prior_precision = config['links']['prior_std']**-2
    gradient = prior_precision * links.omega_mean.clone()
    full_hessian = torch.eye(D, device=omega.device, dtype=omega.dtype).expand(C, D, D).clone() * prior_precision
    chunk = config['links']['copy_chunk_size']
    for bank in model.records:
        if bank.kind != 'communication':
            continue
        mu, Sigma = (state.belief if bank.channel == 0 else state.model).marginals(S, K)
        primitive = model.record_primitives(bank, state)
        valid = (primitive >= 0).nonzero().flatten()
        for start in range(0, len(valid), chunk):
            ids = valid[start:start + chunk]
            p = primitive[ids]
            # Comparison rows are the first R coordinates: embed the residual into K.
            residual = (bank.query[ids] @ mu[bank.receiver[ids], :, None]).squeeze(-1) - bank.observed[ids]
            z = (bank.key[ids].mT @ residual[..., None]).squeeze(-1)
            G = bank.key[ids].mT @ bank.precision[ids] @ bank.key[ids]
            shape = (sample_count, len(ids))
            def expand(x: torch.Tensor) -> torch.Tensor:
                return x.unsqueeze(0).expand(sample_count, *x.shape).flatten(0, 1)
            g, H = derivatives(omega[:, p].flatten(0, 1), expand(z), expand(mu[bank.sender[ids]]),
                               expand(Sigma[bank.sender[ids]]), expand(G))
            weight = state.beta[bank.candidate_id[ids]]
            gradient.index_add_(0, p, g.reshape(*shape, D).mean(0) * weight[:, None])
            full_hessian.index_add_(0, p, H.reshape(*shape, D, D).mean(0) * weight[:, None, None])
    coefficient = config['links']['mutual_loop_coefficient']
    if coefficient and model.loops.weight.numel():
        pairs = model.loops.primitive_id.reshape(-1, 2)
        repeats = 1 if config['links']['shared_channels'] else 2
        copies = C // repeats
        loss = partial(loop_integrand, K=K)
        grad_fn, hess_fn = torch.func.jacrev(loss), torch.func.jacrev(torch.func.jacrev(loss))
        for channel in range(repeats):
            for start in range(0, len(pairs), chunk):
                p = pairs[start:start + chunk] + channel * copies
                x = omega[:, p].reshape(-1, 2 * D)
                with torch.enable_grad():
                    g = torch.func.vmap(grad_fn)(x).reshape(sample_count, -1, 2, D).mean(0)
                    H = torch.func.vmap(hess_fn)(x).reshape(sample_count, -1, 2, D, 2, D).mean(0)
                weights = coefficient * model.loops.weight[start:start + chunk] * (2 if repeats == 1 else 1)
                for side in range(2):
                    gradient.index_add_(0, p[:, side], g[:, side] * weights[:, None])
                    full_hessian.index_add_(0, p[:, side], H[:, side, :, side, :] * weights[:, None, None])
    projected = full_hessian.diagonal(dim1=-2, dim2=-1) if links.precision.ndim == 2 else full_hessian
    original_mu, original_precision = links.omega_mean, links.precision
    before = float(evaluate_vfe(state, model).total)
    dt = config['inference']['link_dt']
    for retry in range(12):
        try:
            links.omega_mean, links.precision = gaussian_blr_step(original_mu, original_precision, gradient, projected, dt)
            after = float(evaluate_vfe(state, model).total)
            if math.isfinite(after) and after <= before + 1e-9 * max(1, abs(before)):
                return retry
        except (ValueError, RuntimeError):
            pass
        links.omega_mean, links.precision = original_mu, original_precision
        dt *= 0.5
    return 12

