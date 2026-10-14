import torch

from ..model import SimulationModel
from ..state import SectionState
from .attention import row_statistics
from .coherence import coherent_partition
from .holonomy import fixed_sector_rank, loop_holonomy_defect


def dense_recognition(state: SectionState, model: SimulationModel) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact joint moments of the declared product across agent/channel patch laws."""
    K, S = model.config['bundle']['K'], model.incidence.agent_id.numel()
    D = S * K
    mu = state.beta.new_zeros(2 * D)
    Sigma = state.beta.new_zeros(2 * D, 2 * D)
    for channel, laws in enumerate((state.belief, state.model)):
        for bucket in laws.buckets:
            index = (bucket.incidence_ids[..., None] * K + torch.arange(K, device=mu.device)).flatten(1) + channel * D
            mu[index] = bucket.mu
            Sigma[index[..., None], index[:, None, :]] = bucket.covariance()
    return mu, Sigma


def collect_observables(state: SectionState, model: SimulationModel,
                        cache: dict[str, float]) -> dict[str, float]:
    """c13 mean-field coherence, c10 fixed-sector kernel, c09 exact Gaussian KL loss."""
    K, S = model.config['bundle']['K'], model.incidence.agent_id.numel()
    mu = state.belief.marginals(S, K)[0]
    out = {name: float(value.mean()) for name, value in row_statistics(state.beta, model.rows).items()}
    out['field_rmse_reference'] = float((mu - model.truth).square().mean().sqrt())
    # Structural and coherence observables concern one network in one fiber.
    context_id = 0
    bank = next((b for b in model.records if b.kind == 'communication' and b.channel == 0), None)
    if bank is not None:
        selected = model.incidence.context_id[bank.receiver] == context_id
        vertices = (model.incidence.context_id == context_id).nonzero().flatten()
        if vertices.numel():
            rec, src = bank.receiver[selected], bank.sender[selected]
            local_r, local_s = torch.searchsorted(vertices, rec), torch.searchsorted(vertices, src)
            A, B = model.record_maps(bank, state)
            residual = (A[selected] @ mu[rec, :, None] - B[selected] @ mu[src, :, None]).squeeze(-1)
            norm = torch.linalg.vector_norm(residual, dim=-1)
            for resolution in model.config['analytics']['coherence_resolutions']:
                labels = coherent_partition(norm, local_s, local_r, resolution, vertices.numel())
                out[f'coherent_blocks_mean_field_a{resolution:g}'] = float(labels.unique().numel())
            key = 'fixed_sector_rank_context0'
            if key not in cache and model.config['links']['mode'] == 'held':
                if bank.query.shape[-2] == K and model.config['communication']['comparison'] == 'full_state':
                    cache[key] = float(fixed_sector_rank(B[selected], local_s, local_r, vertices.numel()))
            out.update(cache)
            out['analytics_context_id'] = float(context_id)
            out['analytics_context_agents'] = float(vertices.numel())
            N = vertices.numel()
            pair_key = local_r * N + local_s
            order = pair_key.argsort()
            reverse_key = local_s * N + local_r
            positions = torch.searchsorted(pair_key[order], reverse_key)
            valid = (positions < len(order)) & (pair_key[order[positions.clamp_max(len(order) - 1)]] == reverse_key)
            weights = state.beta[bank.candidate_id[selected]]
            out['routing_reciprocity_reference_mean_abs'] = float((weights[valid] - weights[order[positions[valid]]]).abs().mean()) if valid.any() else 0.0
    omega = state.links.omega_mean.reshape(-1, K, K)
    if not model.config['links']['shared_channels']:
        omega = omega[:omega.shape[0] // 2]
    defects = loop_holonomy_defect(omega, model.loops)
    out['holonomy_mean_defect_reference'] = float(defects.mean()) if defects.numel() else 0.0
    out['discarded_conditional_kl_available'] = 0.0
    if (state.links.mode == 'held' and bool((model.rows.row_offset.diff() == 1).all())
            and 2 * S * K <= model.config['analytics']['coarse_max_dim']):
        from ..objective import dense_action
        from .coarse import discarded_conditional_kl
        J, h, _ = dense_action(state, model)
        L = torch.linalg.cholesky(J)
        Sigma_p = torch.cholesky_inverse(L)
        mu_p = torch.cholesky_solve(h[:, None], L).squeeze(-1)
        mu_q, Sigma_q = dense_recognition(state, model)
        vertices = (model.incidence.context_id == context_id).nonzero().flatten()
        if vertices.numel():
            reduction = mu_q.new_zeros(2 * K, 2 * S * K)
            eye = torch.eye(K, device=mu_q.device, dtype=mu_q.dtype) / vertices.numel()
            for channel in range(2):
                for vertex in vertices:
                    reduction[channel * K:(channel + 1) * K, channel * S * K + vertex * K:channel * S * K + (vertex + 1) * K] = eye
            out['discarded_conditional_kl'] = float(discarded_conditional_kl(mu_q, Sigma_q, mu_p, Sigma_p, reduction))
            out['discarded_conditional_kl_available'] = 1.0
    return out
