"""Tiny systems validate formulas/schedules; they do not establish large-N behavior."""

import math

import pytest
import torch

from vfe4.analytics.coarse import discarded_conditional_kl
from vfe4.gaussian import gaussian_kl
from vfe4.latent_links import gaussian_blr_step, link_kl, independent_diagnostics
from vfe4.config import load_config
from vfe4.presets import resolve_preset
from vfe4.model import build_model, initialize_sections
from vfe4.inference import make_schedule
from vfe4.geometry.transport import base_transport
from vfe4.aggregate import RunTable, aggregate_metric


def test_conditional_kl_correlated_oracle() -> None:
    generator = torch.Generator().manual_seed(0)
    A, B = [torch.randn(3, 3, generator=generator, dtype=torch.float64) for _ in range(2)]
    Sigma_q, Sigma_p = A @ A.mT + torch.eye(3), B @ B.mT + torch.eye(3)
    mu_q, mu_p = [torch.randn(3, generator=generator, dtype=torch.float64) for _ in range(2)]
    T = torch.tensor([[1., 1., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=mu_q.dtype)
    q, p, Q, P = T @ mu_q, T @ mu_p, T @ Sigma_q @ T.mT, T @ Sigma_p @ T.mT
    a, b = Q[1:, :1] / Q[0, 0], P[1:, :1] / P[0, 0]
    q_cov, p_cov = Q[1:, 1:] - a @ Q[:1, 1:], P[1:, 1:] - b @ P[:1, 1:]
    delta_mean = q[1:] - p[1:] - (b[:, 0] * (q[0] - p[0]))
    expected = gaussian_kl(delta_mean, q_cov, torch.zeros(2, dtype=q.dtype), p_cov)
    expected += 0.5 * Q[0, 0] * ((a - b).mT @ torch.linalg.solve(p_cov, a - b)).squeeze()
    torch.testing.assert_close(discarded_conditional_kl(mu_q, Sigma_q, mu_p, Sigma_p, T[:1]), expected)


def test_hyperedge_colors_and_jacobi_counterexample() -> None:
    config = load_config(resolve_preset('tiny_point', {'population': {'n_agents': 3, 'lattice_shape': (3,)}}))
    schedule = make_schedule(build_model(config), config)
    assert len(schedule.colors) == 3
    # General SPD dense factor counterexample from c08; not a claim about a formation mechanism.
    J = torch.full((3, 3), 0.9, dtype=torch.float64)
    J.diagonal().fill_(1)
    mu = torch.ones(3, dtype=J.dtype)
    jacobi = -(J - torch.eye(3)) @ mu
    energy = lambda x: 0.5 * x @ J @ x
    assert energy(jacobi) > energy(mu)
    for color in schedule.colors:
        mu[color] = -((J - torch.eye(3)) @ mu)[color]
    assert energy(mu) < energy(torch.ones_like(mu))


def test_latent_kl_spd_and_independent_stream() -> None:
    config = load_config(resolve_preset('tiny_point', {'population': {'n_agents': 2, 'lattice_shape': (2,)},
                        'bundle': {'K': 1}, 'links': {'mode': 'latent_gaussian', 'samples': 4}}))
    model = build_model(config)
    state = initialize_sections(model, config)
    links = state.links
    expected = 0.5 * (links.omega_mean.square() / config['links']['prior_std']**2).sum()
    torch.testing.assert_close(link_kl(links, config['links']['prior_std']), expected)
    with pytest.raises(ValueError, match='positive'):
        gaussian_blr_step(torch.ones(1), torch.ones(1), torch.ones(1), -torch.ones(1), 1.)
    diagnostics = independent_diagnostics(state, model)
    assert math.isfinite(diagnostics['diagnostic_vfe_standard_error'])
    # Initial integration points are drawn lazily, but diagnostic RNG is distinct.
    first_integration = links.rng_state.clone()
    independent_diagnostics(state, model)
    assert torch.equal(first_integration, links.rng_state)


def test_swept_models_require_separate_groups() -> None:
    table = RunTable([{'run_id': 'a', 'sweep': 0, 'x': 1., 'bundle.K': 2},
                      {'run_id': 'b', 'sweep': 0, 'x': 2., 'bundle.K': 3}], {}, ('bundle.K',))
    with pytest.raises(ValueError, match='swept keys'):
        aggregate_metric(table, 'x', 'sweep', 99, 'percentile', 0)


def test_fourier_frame_transport_cocycle() -> None:
    config = load_config(resolve_preset('tiny_spatial_2d', {}))
    model = build_model(config)
    agent = torch.tensor([0])
    a, b, c = [torch.tensor([point], dtype=config.dtype) for point in ((.1, .2), (.3, .4), (.6, .7))]
    Ua, Ub, Uc = [model.frame_field.evaluate(agent, point) for point in (a, b, c)]
    torch.testing.assert_close(base_transport(Uc, Ub) @ base_transport(Ub, Ua), base_transport(Uc, Ua))
