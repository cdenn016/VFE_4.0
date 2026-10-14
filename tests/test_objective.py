import math

import torch

from vfe4.config import load_config
from vfe4.presets import resolve_preset
from vfe4.model import build_model, initialize_sections
from vfe4.objective import evaluate_vfe, dense_action, optimal_rows
from vfe4.gaussian import gaussian_kl


def test_singleton_evidence_identity_and_self_record() -> None:
    cfg = load_config(resolve_preset('tiny_point', {
        'population': {'n_agents': 1, 'lattice_shape': (1,)},
        'observations': {'private_fraction': 1.0},
    }))
    model = build_model(cfg)
    state = initialize_sections(model, cfg)
    J, h, constant = dense_action(state, model)
    posterior_cov = torch.linalg.inv(J)
    posterior_mean = torch.linalg.solve(J, h)
    log_evidence = 0.5 * (h @ posterior_mean + h.numel() * math.log(2 * math.pi) - torch.logdet(J)) - constant
    mu = torch.cat([state.belief.buckets[0].mu.flatten(), state.model.buckets[0].mu.flatten()])
    Sigma = torch.block_diag(state.belief.buckets[0].covariance()[0], state.model.buckets[0].covariance()[0])
    expected = gaussian_kl(mu, Sigma, posterior_mean, posterior_cov)
    torch.testing.assert_close(evaluate_vfe(state, model).total + log_evidence, expected)
    costs = model.record_costs(state)
    # Identity self residual has no state variance, even if q is broad.
    state.belief.buckets[0].chol *= 2
    torch.testing.assert_close(model.record_costs(state)[model.channel_edge == 0], costs[model.channel_edge == 0])


def test_expected_rows_retain_source_variance() -> None:
    cfg = load_config(resolve_preset('tiny_point', {}))
    model, state = build_model(cfg), None
    state = initialize_sections(model, cfg)
    costs = model.record_costs(state)
    state.belief.buckets[0].chol[0] *= 2
    changed = model.record_costs(state)
    assert not torch.allclose(costs, changed)
    beta = optimal_rows(changed, model.rows, state.log_pi)
    sums = torch.segment_reduce(beta, 'sum', lengths=model.rows.row_offset.diff())
    torch.testing.assert_close(sums, torch.ones_like(sums))


def test_spatial_self_copy_uses_physical_ids() -> None:
    cfg = load_config(resolve_preset('tiny_spatial_2d', {}))
    model = build_model(cfg)
    assert model.copies.receiver_agent.max() < cfg['population']['n_agents']
    state = initialize_sections(model, cfg)
    assert torch.isfinite(evaluate_vfe(state, model).total)
