"""One collective finite-design VFE, preserving Gaussian and categorical normalizers."""

from dataclasses import dataclass
import math

import torch

from .gaussian import gaussian_entropy, ragged_log_softmax
from .model import SimulationModel
from .state import CandidateRows, SectionState


@dataclass
class VFEReport:
    terms: dict[str, torch.Tensor]
    normalizer_status: str = 'exact'
    estimator_error: float = 0.0

    @property
    def total(self) -> torch.Tensor:
        return sum(self.terms.values())


def optimal_rows(costs: torch.Tensor, rows: CandidateRows, log_pi: torch.Tensor | None = None) -> torch.Tensor:
    r"""beta* = softmax_row(log pi - E[-log likelihood]), c06 recognition optimum."""
    return ragged_log_softmax((rows.log_pi if log_pi is None else log_pi) - costs, rows.row_offset).exp()


def evaluate_vfe(state: SectionState, model: SimulationModel) -> VFEReport:
    K, S = model.config['bundle']['K'], model.incidence.agent_id.numel()
    mu_b, Sigma_b = state.belief.marginals(S, K)
    mu_m, Sigma_m = state.model.marginals(S, K)
    H = model.config['generative']['model_prior_std']**-2
    Q = model.config['generative']['bridge_std']**2
    terms = {
        'model_prior': 0.5 * (H * (mu_m.square().sum(-1) + Sigma_m.diagonal(dim1=-2, dim2=-1).sum(-1))
                             + K * math.log(2 * math.pi / H)).sum(),
        'bridge': 0.5 * (((mu_b - mu_m).square().sum(-1) + (Sigma_b + Sigma_m).diagonal(dim1=-2, dim2=-1).sum(-1)) / Q
                        + K * math.log(2 * math.pi * Q)).sum(),
        'recognition_entropy': -sum(gaussian_entropy(bucket.chol).sum()
                                   for laws in (state.belief, state.model) for bucket in laws.buckets),
        'categorical_kl': (torch.special.xlogy(state.beta, state.beta) - state.beta * state.log_pi).sum(),
        'private': mu_b.new_zeros(()), 'spatial': mu_b.new_zeros(()), 'communication': mu_b.new_zeros(()),
    }
    for bank in model.records:
        costs = model.bank_costs(state, bank)
        weight = state.beta[bank.candidate_id] if bank.kind == 'communication' else torch.ones_like(costs)
        terms[bank.kind] = terms[bank.kind] + (weight * costs).sum()
    if state.links.mode == 'latent_gaussian':
        from .latent_links import link_kl, loop_sample_costs
        terms['generator_kl'] = link_kl(state.links, model.config['links']['prior_std'])
        terms['loop_prior'] = loop_sample_costs(state.links, model).mean()
        status = ('relative_vfe_unknown_logZ' if model.config['links']['mutual_loop_coefficient'] and model.loops.weight.numel()
                  else 'estimated_vfe_known_normalizer')
        return VFEReport(terms, status, math.nan)
    return VFEReport(terms)


def dense_action(state: SectionState, model: SimulationModel) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Tiny exact oracle: exp(-1/2 y^T J y+h^T y-c), labels fixed under Q."""
    S, K = model.incidence.agent_id.numel(), model.config['bundle']['K']
    if state.links.mode != 'held':
        raise ValueError('The dense Gaussian oracle requires conditioned held generators')
    D = S * K
    if 2 * D > model.config['analytics']['coarse_max_dim']:
        raise ValueError('Dense posterior unavailable above analytics.coarse_max_dim')
    eye = torch.eye(D, device=state.beta.device, dtype=state.beta.dtype)
    precision_bridge = model.config['generative']['bridge_std']**-2
    H = model.config['generative']['model_prior_std']**-2
    J = torch.cat([torch.cat([precision_bridge * eye, -precision_bridge * eye], 1),
                   torch.cat([-precision_bridge * eye, (H + precision_bridge) * eye], 1)], 0)
    h = eye.new_zeros(2 * D)
    constant = eye.new_tensor(0.5 * D * (math.log(2 * math.pi / H) + math.log(2 * math.pi / precision_bridge)))
    constant += (torch.special.xlogy(state.beta, state.beta) - state.beta * state.log_pi).sum()
    for bank in model.records:
        F, R = bank.query.shape[:2]
        A, B = model.record_maps(bank, state)
        linear = eye.new_zeros(F, R, 2 * D)
        index = bank.receiver[:, None, None] * K + torch.arange(K, device=eye.device)[None, None, :] + bank.channel * D
        linear.scatter_add_(2, index.expand(F, R, K), A)
        index = bank.sender.clamp_min(0)[:, None, None] * K + torch.arange(K, device=eye.device)[None, None, :] + bank.channel * D
        linear.scatter_add_(2, index.expand(F, R, K), -B)
        weight = state.beta[bank.candidate_id] if bank.kind == 'communication' else eye.new_ones(F)
        weighted = bank.precision * weight[:, None, None]
        root = torch.linalg.cholesky(bank.precision).mT @ linear * weight.sqrt()[:, None, None]
        flattened = root.reshape(-1, 2 * D)
        J += flattened.T @ flattened
        h += (linear.mT @ weighted @ bank.observed[:, :, None]).squeeze(-1).sum(0)
        constant += (weight * (bank.normalizer() + 0.5 * torch.einsum('fi,fij,fj->f', bank.observed, bank.precision, bank.observed))).sum()
    return J, h, constant
