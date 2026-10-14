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


def _directed_three() -> tuple[object, object, object]:
    # Each receiver hears exactly one other agent: singleton rows with real sender terms.
    cfg = load_config(resolve_preset('tiny_point', {
        'population': {'n_agents': 3, 'lattice_shape': (3,)},
        'communication': {'prior_family': 'directed_window', 'include_self': False},
        'links': {'initial_twist_std': 0.3}, 'observations': {'private_fraction': 1.0}}))
    model = build_model(cfg)
    state = initialize_sections(model, cfg)
    generator = torch.Generator().manual_seed(1)
    for laws in (state.belief, state.model):
        bucket = laws.buckets[0]
        bucket.mu += torch.randn(bucket.mu.shape, generator=generator, dtype=cfg.dtype)
        bucket.chol += 0.2 * torch.randn(bucket.chol.shape, generator=generator, dtype=cfg.dtype).tril(-1)
    return cfg, model, state


def test_cross_agent_evidence_identity() -> None:
    cfg, model, state = _directed_three()
    assert bool((model.rows.row_offset.diff() == 1).all())
    J, h, constant = dense_action(state, model)
    posterior_cov = torch.linalg.inv(J)
    posterior_mean = torch.linalg.solve(J, h)
    log_evidence = 0.5 * (h @ posterior_mean + h.numel() * math.log(2 * math.pi) - torch.logdet(J)) - constant
    from vfe4.analytics.observables import dense_recognition
    mu, Sigma = dense_recognition(state, model)
    expected = gaussian_kl(mu, Sigma, posterior_mean, posterior_cov)
    torch.testing.assert_close(evaluate_vfe(state, model).total + log_evidence, expected)


def test_record_cost_carries_sender_variance() -> None:
    r"""Changing only the sender covariance shifts its row cost by tr(W B dSigma B^T)/2."""
    cfg, model, state = _directed_three()
    bank = next(b for b in model.records if b.kind == 'communication' and b.channel == 0)
    before = model.bank_costs(state, bank)
    bucket = state.belief.buckets[0]
    old = bucket.covariance()[2]
    bucket.chol[2] *= 1.5
    delta_Sigma = bucket.covariance()[2] - old
    after = model.bank_costs(state, bank)
    _, B = model.record_maps(bank, state)
    sends = bank.sender == 2
    expected = 0.5 * torch.einsum('fij,fjk,fki->f', bank.precision[sends], B[sends] @ delta_Sigma, B[sends].mT)
    torch.testing.assert_close(after[sends] - before[sends], expected)
    assert bool(expected.abs().min() > 0)


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
