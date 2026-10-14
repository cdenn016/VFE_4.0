"""Typed context restriction consumable as MultiAgentELBO GaussianAction(J,h,c)."""

from copy import deepcopy
from pathlib import Path
import math
from typing import Any

import torch

from .model import SimulationModel
from .state import SectionState


def context_snapshot(state: SectionState, model: SimulationModel, context_id: int,
                     boundary_values: torch.Tensor | None = None,
                     labels: torch.Tensor | None = None) -> dict[str, Any]:
    r"""Restrict the complete joint action: J'=J_rr, h'=h_r-J_rb v_b.

    Source labels and generator realizations are explicitly conditioned. Boundary
    values default to the recorded recognition means, rather than being integrated
    out. The recognition law is conditioned on the same boundary values.
    """
    K, S = model.config['bundle']['K'], model.incidence.agent_id.numel()
    if not 0 <= context_id < len(model.domain.contexts):
        raise ValueError('context_id is outside the finite domain')
    vertices = (model.incidence.context_id == context_id).nonzero().flatten()
    if not len(vertices):
        raise ValueError('Chosen context has no agents')
    C, D = len(vertices), len(vertices) * K
    if 2 * D > model.config['analytics']['coarse_max_dim']:
        raise ValueError('Context action exceeds analytics.coarse_max_dim')
    selected_state = deepcopy(state)
    selected_state.links.mode = 'held'
    # Every row retains one explicit source, including rows outside the context.
    if labels is None:
        chosen = []
        for lo, hi in zip(model.rows.row_offset[:-1].tolist(), model.rows.row_offset[1:].tolist()):
            chosen.append(lo + int(state.beta[lo:hi].argmax()))
        labels = torch.tensor(chosen, device=state.beta.device)
        label_policy = 'MAP recognition source in each row'
    else:
        label_policy = 'caller supplied candidate ID in each row'
    if (labels.shape != model.rows.row_offset[:-1].shape or
            bool(((labels < model.rows.row_offset[:-1]) | (labels >= model.rows.row_offset[1:])).any())):
        raise ValueError('labels must contain one valid candidate ID per original row')
    selected_state.beta = torch.zeros_like(state.beta)
    selected_state.beta[labels] = 1
    means = torch.stack([law.marginals(S, K)[0] for law in (state.belief, state.model)])
    boundary_policy = 'recognition means' if boundary_values is None else 'caller supplied'
    boundary = means.clone() if boundary_values is None else boundary_values.clone()
    if boundary.shape != (2, S, K) or not bool(torch.isfinite(boundary).all()):
        raise ValueError('boundary_values must be finite with shape (2,S,K)')
    fixed = boundary.clone()
    fixed[:, vertices] = 0
    mapping = torch.full((S,), -1, device=state.beta.device, dtype=torch.long)
    mapping[vertices] = torch.arange(C, device=state.beta.device)
    eye = torch.eye(D, device=state.beta.device, dtype=state.beta.dtype)
    Qinv, H = model.config['generative']['bridge_std']**-2, model.config['generative']['model_prior_std']**-2
    J = torch.cat((torch.cat((Qinv * eye, -Qinv * eye), 1),
                   torch.cat((-Qinv * eye, (Qinv + H) * eye), 1)), 0)
    h = eye.new_zeros(2 * D)
    constant = eye.new_tensor(0.5 * S * K * (math.log(2 * math.pi / H) + math.log(2 * math.pi / Qinv)))
    constant += 0.5 * (H * fixed[1].square().sum() + Qinv * (fixed[0] - fixed[1]).square().sum())
    constant -= state.log_pi[labels].sum()
    for bank in model.records:
        A, B = model.record_maps(bank, selected_state)
        include = selected_state.beta[bank.candidate_id].bool() if bank.kind == 'communication' else torch.ones(len(bank.receiver), device=eye.device, dtype=torch.bool)
        rec, src = bank.receiver[include], bank.sender[include]
        A, B, W = A[include], B[include], bank.precision[include]
        effective = bank.observed[include] - (A @ fixed[bank.channel, rec, :, None]).squeeze(-1)
        effective += (B @ fixed[bank.channel, src.clamp_min(0), :, None]).squeeze(-1)
        constant += (bank.normalizer()[include] + 0.5 * torch.einsum('fi,fij,fj->f', effective, W, effective)).sum()
        incident = (mapping[rec] >= 0) | ((src >= 0) & (mapping[src.clamp_min(0)] >= 0))
        rec, src, A, B, W, effective = rec[incident], src[incident], A[incident], B[incident], W[incident], effective[incident]
        R = A.shape[-2]
        linear = eye.new_zeros(len(rec), R, 2 * D)
        for endpoints, maps in ((rec, A), (src, -B)):
            valid = (endpoints >= 0) & (mapping[endpoints.clamp_min(0)] >= 0)
            index = mapping[endpoints.clamp_min(0)].clamp_min(0)[:, None, None] * K + torch.arange(K, device=eye.device)[None, None, :] + bank.channel * D
            linear.scatter_add_(2, index.expand(-1, R, K), maps * valid[:, None, None])
        flattened = (torch.linalg.cholesky(W).mT @ linear).reshape(-1, 2 * D)
        J += flattened.mT @ flattened
        h += (linear.mT @ W @ effective[..., None]).squeeze(-1).sum(0)
    q_mu, q_cov = eye.new_zeros(2 * D), eye.new_zeros(2 * D, 2 * D)
    for channel, laws in enumerate((state.belief, state.model)):
        for bucket in laws.buckets:
            full = bucket.covariance()
            for position, incidence_ids in enumerate(bucket.incidence_ids):
                present = mapping[incidence_ids] >= 0
                if not bool(present.any()):
                    continue
                local_node = int(present.nonzero().flatten()[0])
                keep = torch.arange(K, device=eye.device) + local_node * K
                omit = torch.arange(bucket.mu.shape[-1], device=eye.device)
                omit = omit[~torch.isin(omit, keep)]
                mean, covariance = bucket.mu[position, keep], full[position][keep[:, None], keep[None, :]]
                if len(omit):
                    cross = full[position][keep[:, None], omit[None, :]]
                    L = torch.linalg.cholesky(full[position][omit[:, None], omit[None, :]])
                    values = boundary[channel, incidence_ids].flatten()[omit] - bucket.mu[position, omit]
                    mean = mean + cross @ torch.cholesky_solve(values[:, None], L).squeeze(-1)
                    covariance = covariance - cross @ torch.cholesky_solve(cross.mT, L)
                target = keep - local_node * K + mapping[incidence_ids[local_node]] * K + channel * D
                q_mu[target], q_cov[target[:, None], target[None, :]] = mean, covariance
    retained = torch.cat([vertices[:, None] * K + torch.arange(K, device=eye.device),
                          vertices[:, None] * K + torch.arange(K, device=eye.device) + S * K]).flatten()
    return {'schema_version': 1, 'constructor': 'multiagent_elbo.realizations.gaussian.network.GaussianAction',
            'model_type': 'conditioned' if model.config['spatial']['enabled'] else 'spatial_disabled',
            'J': J, 'h': h, 'c': constant, 'recognition_mean': q_mu, 'recognition_covariance': q_cov,
            'recognition_type': 'conditioned original patch law', 'context_id': context_id,
            'coordinate_order': 'all retained belief coordinates, then all retained model coordinates',
            'physical_agent_ids': model.incidence.agent_id[vertices], 'retained_coordinates': retained,
            'boundary_values': boundary, 'boundary_policy': boundary_policy,
            'conditioned_source_labels': labels, 'label_policy': label_policy,
            'generator_conditioning': 'omega_mean realization; generator uncertainty is not marginalized',
            'generators': deepcopy(state.links), 'source_rows': model.rows, 'log_pi': state.log_pi.clone(),
            'beta': state.beta.clone(), 'copies': model.copies, 'loops': model.loops,
            'records': model.records, 'observation_normalizers': tuple(bank.normalizer() for bank in model.records),
            'priors': deepcopy(model.config['generative']),
            'frames': model.frame_field.evaluate(model.incidence.agent_id[vertices], model.domain.contexts[context_id].expand(C, -1)),
            'constant_convention': 'complete joint restricted to all recorded boundary values and conditioned labels; includes boundary-only factors'}


def export_context(path: Path, state: SectionState, model: SimulationModel, context_id: int,
                   boundary_values: torch.Tensor | None = None, labels: torch.Tensor | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(context_snapshot(state, model, context_id, boundary_values, labels), path)
    return path
