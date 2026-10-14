"""Chromatic Gaussian block inference with every incident sender term (c08)."""

from dataclasses import dataclass

import torch

from .config import SimulationConfig
from .gaussian import checked_cholesky, damp_moments
from .model import SimulationModel
from .objective import evaluate_vfe, optimal_rows
from .state import GaussianBucket, PatchGaussians, SectionState


@dataclass
class UpdateSchedule:
    colors: tuple[torch.Tensor, ...]


@dataclass
class StepReport:
    objective_before: float
    objective_after: float
    accepted: bool
    retries: int = 0
    min_spd_status: int = 0
    estimator_error: float = 0.0
    normalizer_status: str = 'exact'


def make_schedule(model: SimulationModel, config: SimulationConfig) -> UpdateSchedule:
    N = config['population']['n_agents']
    graph = [set() for _ in range(N)]
    receivers = model.copies.receiver_agent.cpu().tolist()
    senders = model.copies.source_agent.cpu().tolist()
    # Conservative source-row hypergraph coloring, not just directed attending edges.
    rows = model.rows.row_offset.cpu().tolist()
    rec_inc = model.rows.receiver_incidence.cpu().tolist()
    src_inc = model.rows.source_incidence.cpu().tolist()
    agents = model.incidence.agent_id.cpu().tolist()
    for lo, hi in zip(rows[:-1], rows[1:]):
        scope = {agents[rec_inc[lo]], *(agents[x] for x in src_inc[lo:hi])}
        for i in scope:
            graph[i].update(scope - {i})
    colors = [-1] * N
    for agent in sorted(range(N), key=lambda i: (-len(graph[i]), i)):
        used = {colors[j] for j in graph[agent] if colors[j] >= 0}
        color = 0
        while color in used:
            color += 1
        colors[agent] = color
    groups = tuple(torch.tensor([i for i, c in enumerate(colors) if c == color], device=config.device)
                   for color in range(max(colors) + 1))
    return UpdateSchedule(groups)


def _add_blocks(target: torch.Tensor, agents: torch.Tensor, local_r: torch.Tensor,
                local_s: torch.Tensor, values: torch.Tensor, K: int) -> None:
    D = target.shape[-1]
    coordinate = torch.arange(K, device=target.device)
    row = local_r[:, None, None] * K + coordinate[None, :, None]
    col = local_s[:, None, None] * K + coordinate[None, None, :]
    index = (agents[:, None, None] * D + row) * D + col
    target.view(-1).index_add_(0, index.expand(-1, K, K).flatten(), values.flatten())


def _assemble(bucket: GaussianBucket, state: SectionState, model: SimulationModel,
              channel: int) -> tuple[torch.Tensor, torch.Tensor, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    K = model.config['bundle']['K']
    S, N = model.incidence.agent_id.numel(), model.config['population']['n_agents']
    Bn, D = bucket.mu.shape
    P = bucket.incidence_ids.shape[1]
    precision_bridge = model.config['generative']['bridge_std']**-2
    H = model.config['generative']['model_prior_std']**-2 if channel else 0.0
    precision = torch.eye(D, device=bucket.mu.device, dtype=bucket.mu.dtype).expand(Bn, D, D).clone() * (precision_bridge + H)
    # precision: (agents_in_bucket, P*K, P*K), never a global precision.
    other = state.model if channel == 0 else state.belief
    other_mu = other.marginals(S, K)[0]
    info = precision_bridge * other_mu[bucket.incidence_ids].reshape(Bn, D)
    mapping = torch.full((N,), -1, device=bucket.mu.device, dtype=torch.long)
    mapping[bucket.agent_ids] = torch.arange(Bn, device=bucket.mu.device)
    messages = []
    for bank in model.records:
        if bank.channel != channel or not bank.receiver.numel():
            continue
        A, B, second_gram = model.record_statistics(bank, state)
        weight = state.beta[bank.candidate_id] if bank.kind == 'communication' else bucket.mu.new_ones(bank.receiver.numel())
        W = bank.precision * weight[:, None, None]
        sender = bank.sender.clamp_min(0)
        ra, sa = model.incidence.agent_id[bank.receiver], model.incidence.agent_id[sender]
        r, s = model.incidence.local_node_id[bank.receiver], model.incidence.local_node_id[sender]
        same = (bank.sender >= 0) & (ra == sa)
        self_copy = same & (bank.receiver == sender)
        A2, B2 = A.clone(), B.clone()
        A2[self_copy] -= B2[self_copy]
        B2[self_copy] = 0
        PA = A2.mT @ W @ A2
        PB = second_gram * weight[:, None, None]
        PB = torch.where(self_copy[:, None, None], torch.zeros_like(PB), PB)
        cross = A2.mT @ W @ B2
        ha = (A2.mT @ W @ bank.observed[:, :, None]).squeeze(-1)
        hb = -(B2.mT @ W @ bank.observed[:, :, None]).squeeze(-1)
        valid_r = mapping[ra] >= 0
        if valid_r.any():
            _add_blocks(precision, mapping[ra[valid_r]], r[valid_r], r[valid_r], PA[valid_r], K)
            idx = mapping[ra[valid_r], None] * D + r[valid_r, None] * K + torch.arange(K, device=info.device)
            info.view(-1).index_add_(0, idx.flatten(), ha[valid_r].flatten())
        valid_s = (bank.sender >= 0) & (mapping[sa] >= 0)
        if valid_s.any():
            _add_blocks(precision, mapping[sa[valid_s]], s[valid_s], s[valid_s], PB[valid_s], K)
            idx = mapping[sa[valid_s], None] * D + s[valid_s, None] * K + torch.arange(K, device=info.device)
            info.view(-1).index_add_(0, idx.flatten(), hb[valid_s].flatten())
        internal = same & valid_r & ~self_copy
        if internal.any():
            _add_blocks(precision, mapping[ra[internal]], r[internal], s[internal], -cross[internal], K)
            _add_blocks(precision, mapping[ra[internal]], s[internal], r[internal], -cross[internal].mT, K)
        external = ~same & (bank.sender >= 0)
        if external.any():
            messages.append((bank.receiver[external], sender[external], cross[external]))
    return precision, info, messages


def _channel_sweep(state: SectionState, model: SimulationModel, schedule: UpdateSchedule,
                   config: SimulationConfig, channel: int, weight: float) -> None:
    laws = state.belief if channel == 0 else state.model
    K, S = config['bundle']['K'], model.incidence.agent_id.numel()
    for bucket in laws.buckets:
        precision, static_info, messages = _assemble(bucket, state, model, channel)
        L = checked_cholesky(precision)
        Sigma_star = torch.cholesky_inverse(L)
        if laws.covariance_structure == 'node_blocks':
            P = bucket.incidence_ids.shape[1]
            diag = precision.reshape(-1, P, K, P, K).diagonal(dim1=1, dim2=3).permute(0, 3, 1, 2)
            Sigma_nodes = torch.cholesky_inverse(checked_cholesky(diag))
        for color in schedule.colors:
            selected = torch.isin(bucket.agent_ids, color).nonzero().flatten()
            if not selected.numel():
                continue
            mu_nodes = laws.marginals(S, K)[0]
            incoming = mu_nodes.new_zeros(S, K)
            for rec, src, C in messages:
                incoming.index_add_(0, rec, (C @ mu_nodes[src, :, None]).squeeze(-1))
                incoming.index_add_(0, src, (C.mT @ mu_nodes[rec, :, None]).squeeze(-1))
            rhs = static_info[selected] + incoming[bucket.incidence_ids[selected]].reshape(selected.numel(), -1)
            target_mu = torch.cholesky_solve(rhs[:, :, None], L[selected]).squeeze(-1)
            old_mu = bucket.mu[selected]
            if config['inference']['method'] == 'fisher_natural' and laws.covariance_structure == 'full_patch':
                old_precision = torch.cholesky_inverse(bucket.chol[selected])
                new_precision = (1 - weight) * old_precision + weight * precision[selected]
                new_L = checked_cholesky(new_precision)
                new_info = (1 - weight) * (old_precision @ old_mu[:, :, None]).squeeze(-1) + weight * rhs
                new_mu = torch.cholesky_solve(new_info[:, :, None], new_L).squeeze(-1)
                new_cov = torch.cholesky_inverse(new_L)
            elif laws.covariance_structure == 'full_patch':
                old_cov = bucket.covariance()[selected]
                new_mu, new_cov = damp_moments(old_mu, target_mu, old_cov, Sigma_star[selected], weight)
            else:
                old_cov = bucket.chol[selected] @ bucket.chol[selected].mT
                mean, new_cov = damp_moments(old_mu.reshape(-1, P, K), target_mu.reshape(-1, P, K),
                                             old_cov, Sigma_nodes[selected], weight)
                new_mu = mean.flatten(1)
            bucket.mu[selected] = new_mu
            bucket.chol[selected] = checked_cholesky(new_cov)


@torch.no_grad()
def step(state: SectionState, model: SimulationModel, schedule: UpdateSchedule,
         config: SimulationConfig) -> tuple[SectionState, StepReport]:
    retries = 0
    if state.links.mode == 'latent_gaussian':
        from .latent_links import refresh_samples
        refresh_samples(state.links, config['links']['samples'])
    before = float(evaluate_vfe(state, model).total)
    state.beta = optimal_rows(model.record_costs(state), model.rows, state.log_pi)
    _channel_sweep(state, model, schedule, config, 0, config['inference']['belief_weight'])
    if (state.sweep + 1) % config['inference']['model_every'] == 0:
        _channel_sweep(state, model, schedule, config, 1, config['inference']['model_weight'])
    if (state.sweep + 1) % config['communication']['source_every'] == 0:
        from .source_learning import source_prior_step
        state.log_pi = source_prior_step(state.log_pi, state.beta, model.rows.row_offset,
                                         config['communication']['source_rate'])
    if state.links.mode == 'latent_gaussian' and (state.sweep + 1) % config['inference']['link_every'] == 0:
        from .latent_links import update_links
        retries = update_links(state, model)
    state.sweep += 1
    after_report = evaluate_vfe(state, model)
    after = float(after_report.total)
    tolerance = (1e-9 if config.dtype == torch.float64 else 1e-5) * max(1.0, abs(before))
    if not math_isfinite(after) or after > before + tolerance:
        raise RuntimeError(f'Collective VFE failed descent at sweep {state.sweep}: {before} -> {after}')
    return state, StepReport(before, after, True, retries, estimator_error=after_report.estimator_error,
                            normalizer_status=after_report.normalizer_status)


def math_isfinite(value: float) -> bool:
    import math
    return math.isfinite(value)
