from pathlib import Path

import torch

from vfe4.latent_links import gaussian_blr_step, record_integrand, derivatives, sampled_generators
from vfe4.config import load_config
from vfe4.presets import resolve_preset
from vfe4.model import build_model, initialize_sections
from vfe4.objective import evaluate_vfe
from vfe4.simulation import run_simulation, resume_simulation


def test_quadratic_blr_step() -> None:
    mu = torch.tensor([0.3, -0.2], dtype=torch.float64)
    precision = torch.eye(2, dtype=mu.dtype)
    H = torch.tensor([[2.0, 0.4], [0.4, 3.0]], dtype=mu.dtype)
    b = torch.tensor([1.0, 2.0], dtype=mu.dtype)
    new_mu, new_precision = gaussian_blr_step(mu, precision, H @ mu - b, H, 1.0)
    torch.testing.assert_close(new_mu, torch.linalg.solve(H, b))
    torch.testing.assert_close(new_precision, H)


def test_record_true_hessian_by_finite_difference() -> None:
    omega = torch.tensor([[0.1, 0.2, -0.1, 0.05]], dtype=torch.float64)
    z, mu = torch.tensor([[0.2, 0.8]], dtype=omega.dtype), torch.tensor([[0.4, -0.3]], dtype=omega.dtype)
    Sigma = torch.eye(2, dtype=omega.dtype).unsqueeze(0) * 0.2
    G = torch.eye(2, dtype=omega.dtype).unsqueeze(0)
    gradient, H = derivatives(omega, z, mu, Sigma, G)
    eps = 1e-5
    columns = []
    for axis in range(4):
        delta = torch.zeros_like(omega)
        delta[0, axis] = eps
        plus = derivatives(omega + delta, z, mu, Sigma, G)[0]
        minus = derivatives(omega - delta, z, mu, Sigma, G)[0]
        columns.append((plus - minus)[0] / (2 * eps))
    torch.testing.assert_close(H[0], torch.stack(columns, -1), rtol=1e-6, atol=1e-7)


def test_unknown_loop_normalizer_is_relative() -> None:
    config = load_config(resolve_preset('tiny_point', {'links': {
        'mode': 'latent_gaussian', 'samples': 2, 'mutual_loop_coefficient': 0.1}}))
    model = build_model(config)
    state = initialize_sections(model, config)
    report = evaluate_vfe(state, model)
    assert report.normalizer_status == 'relative_vfe_unknown_logZ'
    assert torch.isfinite(report.total)


def test_sampled_link_resume(tmp_path: Path) -> None:
    raw = {'population': {'n_agents': 2, 'lattice_shape': (2,)}, 'bundle': {'K': 1},
           'links': {'mode': 'latent_gaussian', 'samples': 4},
           'inference': {'link_every': 1}, 'analytics': {'every': 1},
           'output': {'directory': str(tmp_path), 'figures': False}}
    a = run_simulation(load_config(resolve_preset('tiny_point', raw)))
    raw['inference']['sweeps'] = 2
    b = run_simulation(load_config(resolve_preset('tiny_point', raw)))
    c = resume_simulation(b.run_directory / 'final_state.pt', 2)
    assert torch.equal(a.final_state.links.omega_mean, c.final_state.links.omega_mean)
    assert torch.equal(a.final_state.links.precision, c.final_state.links.precision)
    assert torch.equal(a.final_state.beta, c.final_state.beta)
    assert torch.equal(a.final_state.links.rng_state, c.final_state.links.rng_state)
    assert torch.equal(a.final_state.links.diagnostic_rng_state, c.final_state.links.diagnostic_rng_state)
    # The diagnostic stream advances across calls rather than restarting from its seed.
    omega = a.final_state.links.omega_mean
    fresh = torch.Generator().manual_seed(load_config(resolve_preset('tiny_point', raw))['runtime']['seed'] + 101)
    torch.randn(2, *omega.shape, generator=fresh, dtype=omega.dtype)   # one call: samples // 2 = 2
    assert not torch.equal(fresh.get_state(), c.final_state.links.diagnostic_rng_state)


def test_sampled_transport_chunking() -> None:
    config = load_config(resolve_preset('tiny_point', {
        'population': {'n_agents': 3, 'lattice_shape': (3,)}, 'bundle': {'K': 4},
        'links': {'mode': 'latent_gaussian', 'samples': 4, 'copy_chunk_size': 2}}))
    model = build_model(config)
    state = initialize_sections(model, config)
    omega = sampled_generators(state.links, 4)
    T = torch.matrix_exp(omega.reshape(4, -1, 4, 4))
    G = torch.eye(4, dtype=config.dtype) / config['communication']['tau']
    mean, gram = model.transport_moments(state)
    torch.testing.assert_close(mean, T.mean(0))
    torch.testing.assert_close(gram, (T.mT @ G @ T).mean(0))


def _latent_three(anchor: str = 'zero', prior_std: float = 0.1) -> tuple[object, object, object]:
    config = load_config(resolve_preset('tiny_point', {
        'population': {'n_agents': 3, 'lattice_shape': (3,)},
        'links': {'mode': 'latent_gaussian', 'samples': 4, 'initial_twist_std': 0.3,
                  'prior_anchor': anchor, 'prior_std': prior_std}}))
    model = build_model(config)
    return config, model, initialize_sections(model, config)


def test_latent_recognition_does_not_start_at_fixture_twist() -> None:
    config, model, state = _latent_three()
    assert bool(model.fixture_twist.abs().max() > 0)
    assert torch.equal(state.links.omega_mean, torch.zeros_like(model.fixture_twist))
    held = build_model(load_config(resolve_preset('tiny_point', {
        'population': {'n_agents': 3, 'lattice_shape': (3,)}, 'links': {'initial_twist_std': 0.3}})))
    # Same records in both modes: only recognition initialization differs.
    for a, b in zip(model.records, held.records):
        assert torch.equal(a.observed, b.observed)
    assert torch.equal(held.initial_links.omega_mean, model.fixture_twist)


def test_fixture_anchor_reaches_held_limit() -> None:
    from vfe4.latent_links import update_links, refresh_samples
    config, model, state = _latent_three('fixture_twist', 1e-4)
    state.links.omega_mean = state.links.omega_mean + 0.05
    refresh_samples(state.links, 4)
    config['inference']['link_dt'] = 1.0
    for _ in range(5):
        update_links(state, model)
    torch.testing.assert_close(state.links.omega_mean, model.fixture_twist, atol=1e-6, rtol=0)


def test_link_gradient_matches_objective_finite_difference(monkeypatch: object) -> None:
    import vfe4.latent_links as latent
    config, model, state = _latent_three()
    generator = torch.Generator().manual_seed(5)
    state.links.omega_mean = 0.3 * torch.randn(state.links.omega_mean.shape, generator=generator, dtype=config.dtype)
    state.belief.buckets[0].mu += torch.randn(state.belief.buckets[0].mu.shape, generator=generator, dtype=config.dtype)
    latent.refresh_samples(state.links, 4)
    captured = {}

    def capture(mu: torch.Tensor, precision: torch.Tensor, gradient: torch.Tensor,
                hessian: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
        captured['gradient'] = gradient.clone()
        return mu, precision

    monkeypatch.setattr(latent, 'gaussian_blr_step', capture)
    latent.update_links(state, model)
    eps = 1e-6
    for copy, entry in ((0, 1), (1, 2), (3, 0)):
        values = []
        for sign in (1, -1):
            state.links.omega_mean[copy, entry] += sign * eps
            values.append(float(evaluate_vfe(state, model).total))
            state.links.omega_mean[copy, entry] -= sign * eps
        fd = (values[0] - values[1]) / (2 * eps)
        assert abs(fd - float(captured['gradient'][copy, entry])) < 1e-5 * max(1.0, abs(fd))
