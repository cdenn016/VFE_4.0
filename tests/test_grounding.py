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


def _belief_vector(state: object, S: int, K: int) -> torch.Tensor:
    return state.belief.means(S, K).flatten()


def test_colored_sweep_is_dense_block_gauss_seidel() -> None:
    r"""Engine belief means equal Gauss-Seidel on the dense beta-weighted action, in color order.

    The dense J,h come from dense_action, independent of the bucket assembly. A Jacobi
    sweep (all colors reading start-of-sweep means) gives a different result.
    """
    from vfe4.inference import _channel_sweep
    from vfe4.objective import dense_action
    config = load_config(resolve_preset('tiny_point', {'population': {'n_agents': 3, 'lattice_shape': (3,)},
                         'links': {'initial_twist_std': 0.4}, 'observations': {'private_fraction': 1.0}}))
    model = build_model(config)
    state = initialize_sections(model, config)
    generator = torch.Generator().manual_seed(3)
    for laws in (state.belief, state.model):
        laws.buckets[0].mu += torch.randn(laws.buckets[0].mu.shape, generator=generator, dtype=config.dtype)
    schedule = make_schedule(model, config)
    assert len(schedule.colors) == 3
    S, K = model.incidence.agent_id.numel(), config['bundle']['K']
    J, h, _ = dense_action(state, model)
    x = torch.cat([_belief_vector(state, S, K), state.model.means(S, K).flatten()])
    start = x.clone()

    def block_update(source: torch.Tensor, agents: torch.Tensor) -> torch.Tensor:
        target = source.clone()
        for agent in agents.tolist():
            a = torch.arange(agent * K, (agent + 1) * K)
            rest = torch.ones(x.numel(), dtype=torch.bool)
            rest[a] = False
            target[a] = torch.linalg.solve(J[a][:, a], h[a] - J[a][:, rest] @ source[rest])
        return target

    for color in schedule.colors:
        x = block_update(x, color)
    jacobi = block_update(start, torch.arange(S))
    _channel_sweep(state, model, schedule, config, 0, 1.0)
    engine = _belief_vector(state, S, K)
    torch.testing.assert_close(engine, x[:S * K], rtol=1e-10, atol=1e-12)
    assert not torch.allclose(engine, jacobi[:S * K], atol=1e-6)


def test_pairwise_records_make_jacobi_a_descent_too() -> None:
    r"""With only pairwise record scopes, 2D-J is a sum of S Q S >= 0, so even one color descends.

    The c08 counterexample needs a record coupling three blocks, which this record family lacks.
    """
    from vfe4.inference import UpdateSchedule, step
    config = load_config(resolve_preset('tiny_point', {'population': {'n_agents': 5, 'lattice_shape': (5,)},
                         'communication': {'radius': 2, 'tau': 0.05}, 'links': {'initial_twist_std': 1.0}}))
    model = build_model(config)
    state = initialize_sections(model, config)
    single = UpdateSchedule((torch.arange(5),))
    for _ in range(3):
        state, report = step(state, model, single, config)
        assert report.objective_after <= report.objective_before + 1e-9 * abs(report.objective_before)


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


def test_fourier_frames_are_periodic_so_base_loops_are_trivial() -> None:
    r"""Omega around a noncontractible cycle is U(c+L e_k)U(c)^{-1}; it is I only for periodic frames."""
    config = load_config(resolve_preset('tiny_spatial_2d', {'bundle': {'frame_amplitude': 0.8}}))
    model = build_model(config)
    agent = torch.tensor([1])
    point = torch.tensor([[.13, .37]], dtype=config.dtype)
    U = model.frame_field.evaluate(agent, point)
    assert not torch.allclose(model.frame_field.evaluate(agent, point + 0.25), U)
    for axis in range(2):
        shifted = point.clone()
        shifted[0, axis] += float(model.domain.periods[axis])
        loop = base_transport(model.frame_field.evaluate(agent, shifted), U)
        torch.testing.assert_close(loop, torch.eye(config['bundle']['K'], dtype=config.dtype).expand_as(loop))
