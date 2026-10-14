"""Task 9 formula and admission oracles on tiny CPU tensors."""

import math

import pytest
import torch

from vfe4.config import load_config
from vfe4.latent_links import (derivatives, independent_diagnostics, link_kl,
                               loop_integrand, record_integrand, sampled_generators,
                               update_links)
from vfe4.model import build_model, initialize_sections
from vfe4.objective import evaluate_vfe
from vfe4.presets import resolve_preset
from vfe4.state import LinkState


@pytest.mark.parametrize('family', ['diagonal', 'full'])
def test_generator_kl_nontrivial_covariance(family: str) -> None:
    mu = torch.tensor([[0.3, -0.4], [-0.2, 0.1]], dtype=torch.float64)
    P = torch.tensor([[[2.0, 0.4], [0.4, 1.3]], [[1.2, -0.3], [-0.3, 3.0]]], dtype=mu.dtype)
    precision = P.diagonal(dim1=-2, dim2=-1).clone() if family == 'diagonal' else P
    covariance = torch.diag_embed(precision.reciprocal()) if family == 'diagonal' else torch.linalg.inv(P)
    links = LinkState('latent_gaussian', mu, precision, torch.Generator().get_state())
    prior_std = 0.8
    q = torch.distributions.MultivariateNormal(mu, covariance_matrix=covariance)
    p = torch.distributions.MultivariateNormal(torch.zeros_like(mu),
                                              covariance_matrix=torch.eye(2, dtype=mu.dtype) * prior_std**2)
    expected = torch.distributions.kl_divergence(q, p).sum()
    torch.testing.assert_close(link_kl(links, prior_std), expected)


def test_quadratic_generator_expectation_exact_sigma_points() -> None:
    mu = torch.tensor([[0.3, -0.2]], dtype=torch.float64)
    P = torch.tensor([[[2.0, 0.6], [0.6, 1.4]]], dtype=mu.dtype)
    H = torch.tensor([[1.5, -0.2], [-0.2, 2.3]], dtype=mu.dtype)
    b = torch.tensor([0.7, -0.3], dtype=mu.dtype)
    normal = math.sqrt(2) * torch.eye(2, dtype=mu.dtype)
    links = LinkState('latent_gaussian', mu, P, torch.Generator().get_state(),
                      torch.cat((normal, -normal)).unsqueeze(1))
    points = sampled_generators(links, 4)[:, 0]
    observed = (0.5 * torch.einsum('si,ij,sj->s', points, H, points) - points @ b + 0.4).mean()
    expected = 0.5 * mu[0] @ H @ mu[0] + 0.5 * torch.trace(H @ torch.linalg.inv(P[0])) - b @ mu[0] + 0.4
    torch.testing.assert_close(observed, expected)


def test_record_gradient_against_loss_finite_difference() -> None:
    omega = torch.tensor([[0.1, 0.2, -0.1, 0.05]], dtype=torch.float64)
    z, mu = torch.tensor([[0.2, 0.8]], dtype=omega.dtype), torch.tensor([[0.4, -0.3]], dtype=omega.dtype)
    Sigma = torch.tensor([[[0.2, 0.03], [0.03, 0.3]]], dtype=omega.dtype)
    G = torch.tensor([[[2.0, 0.3], [0.3, 1.0]]], dtype=omega.dtype)
    gradient = derivatives(omega, z, mu, Sigma, G)[0][0]
    expected = []
    for axis in range(4):
        delta = torch.zeros_like(omega[0])
        delta[axis] = 1e-5
        plus = record_integrand(omega[0] + delta, z[0], mu[0], Sigma[0], G[0])
        minus = record_integrand(omega[0] - delta, z[0], mu[0], Sigma[0], G[0])
        expected.append((plus - minus) / 2e-5)
    torch.testing.assert_close(gradient, torch.stack(expected), rtol=1e-6, atol=1e-7)


def test_two_copy_nonabelian_loop_derivatives() -> None:
    pair = torch.tensor([0.1, 0.2, -0.1, 0.05, -0.1, 0.03, 0.2, 0.1], dtype=torch.float64)

    def both_bases(x: torch.Tensor) -> torch.Tensor:
        return loop_integrand(x, 2) + loop_integrand(x.reshape(2, 4).flip(0).flatten(), 2)

    gradient_fn = torch.func.jacrev(both_bases)
    gradient, H = gradient_fn(pair), torch.func.jacrev(gradient_fn)(pair)
    finite_gradient, finite_columns = [], []
    for axis in range(8):
        delta = torch.zeros_like(pair)
        delta[axis] = 1e-5
        finite_gradient.append((both_bases(pair + delta) - both_bases(pair - delta)) / 2e-5)
        finite_columns.append((gradient_fn(pair + delta) - gradient_fn(pair - delta)) / 2e-5)
    torch.testing.assert_close(gradient, torch.stack(finite_gradient), rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(H, torch.stack(finite_columns, -1), rtol=1e-6, atol=1e-7)
    assert H[:4, 4:].abs().max() > 0.1


def test_sampled_record_uses_second_transport_moment() -> None:
    config = load_config(resolve_preset('tiny_point', {'population': {'n_agents': 2, 'lattice_shape': (2,)},
        'bundle': {'K': 1}, 'links': {'mode': 'latent_gaussian', 'samples': 2}}))
    model = build_model(config)
    state = initialize_sections(model, config)
    bucket = state.belief.buckets[0]
    bucket.mu[:, 0] = torch.tensor([0.3, 1.0], dtype=config.dtype)
    bucket.chol[:, 0, 0] = torch.tensor([0.4, 0.2], dtype=config.dtype).sqrt()
    state.links.omega_mean.fill_(0.1)
    state.links.precision.fill_(4.0)
    state.links.standard_normals = torch.stack((torch.ones_like(state.links.omega_mean),
                                               -torch.ones_like(state.links.omega_mean)))
    bank = next(bank for bank in model.records if bank.kind == 'communication' and bank.channel == 0)
    selected = ((bank.receiver == 0) & (bank.sender == 1)).nonzero().flatten()[0]
    bank.observed[selected] = 0.6
    T = torch.exp(torch.tensor([0.1 + 0.5, 0.1 - 0.5], dtype=config.dtype))
    quadratic = ((0.3 - 0.6 - T)**2 + 0.4 + T.square() * 0.2).mean()
    expected = bank.normalizer()[selected] + 0.5 * quadratic / config['communication']['tau']
    torch.testing.assert_close(model.bank_costs(state, bank)[selected], expected)


def test_indefinite_generator_hessian_triggers_smaller_step(monkeypatch: pytest.MonkeyPatch) -> None:
    from vfe4 import latent_links
    config = load_config(resolve_preset('tiny_point', {'population': {'n_agents': 2, 'lattice_shape': (2,)},
        'bundle': {'K': 1}, 'communication': {'channels': ('belief',)},
        'links': {'mode': 'latent_gaussian', 'samples': 2, 'prior_std': 1.0, 'initial_twist_std': 0.0},
        'inference': {'link_dt': 1.0}}))
    model = build_model(config)
    state = initialize_sections(model, config)
    state.belief.buckets[0].mu[:, 0] = torch.tensor([0.0, 1.0], dtype=config.dtype)
    state.belief.buckets[0].chol.fill_(math.sqrt(0.001))
    state.links.standard_normals = torch.zeros(2, *state.links.omega_mean.shape, dtype=config.dtype)
    bank = next(bank for bank in model.records if bank.kind == 'communication')
    bank.observed[bank.receiver == 0] = -10.0
    state.beta.zero_()
    chosen = ((bank.receiver == 0) & (bank.sender == 1)) | ((bank.receiver == 1) & (bank.sender == 1))
    state.beta[bank.candidate_id[chosen]] = 1.0
    original = latent_links.gaussian_blr_step
    rejected_steps = []

    def capture(mu: torch.Tensor, precision: torch.Tensor, gradient: torch.Tensor,
                hessian: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            return original(mu, precision, gradient, hessian, dt)
        except ValueError:
            rejected_steps.append(dt)
            raise

    monkeypatch.setattr(latent_links, 'gaussian_blr_step', capture)
    before = evaluate_vfe(state, model).total.clone()
    retries = update_links(state, model)
    assert rejected_steps and rejected_steps[0] == 1.0 and 0 < retries < 12
    assert bool((state.links.precision > 0).all())
    assert evaluate_vfe(state, model).total < before


def test_paired_estimates_remain_identical_after_diagnostics() -> None:
    config = load_config(resolve_preset('tiny_point', {'population': {'n_agents': 2, 'lattice_shape': (2,)},
        'bundle': {'K': 2}, 'links': {'mode': 'latent_gaussian', 'samples': 4, 'mutual_loop_coefficient': 0.1}}))
    model = build_model(config)
    state = initialize_sections(model, config)
    first = evaluate_vfe(state, model).total.clone()
    samples, integration_rng = state.links.standard_normals.clone(), state.links.rng_state.clone()
    assert torch.equal(first, evaluate_vfe(state, model).total)
    independent_diagnostics(state, model)
    assert torch.equal(state.links.standard_normals, samples)
    assert torch.equal(state.links.rng_state, integration_rng)
    assert torch.equal(first, evaluate_vfe(state, model).total)


def test_sample_count_changes_diagnostic_uncertainty_without_model_change() -> None:
    models, errors = [], []
    for count in (4, 8):
        config = load_config(resolve_preset('tiny_point', {'population': {'n_agents': 2, 'lattice_shape': (2,)},
            'bundle': {'K': 1}, 'links': {'mode': 'latent_gaussian', 'samples': count}}))
        model = build_model(config)
        state = initialize_sections(model, config)
        state.links.standard_normals = torch.zeros(1, *state.links.omega_mean.shape, dtype=config.dtype)
        baseline = evaluate_vfe(state, model).total
        coefficient = torch.zeros(len(state.links.omega_mean), dtype=config.dtype)
        for bank in model.records:
            if bank.kind != 'communication':
                continue
            primitive = model.record_primitives(bank, state)
            valid = primitive >= 0
            Sigma = (state.belief if bank.channel == 0 else state.model).marginals(2, 1)[1]
            weights = state.beta[bank.candidate_id[valid]]
            terms = 0.5 * bank.precision[valid, 0, 0] * Sigma[bank.sender[valid], 0, 0] * weights
            coefficient.index_add_(0, primitive[valid], terms)
        generator = torch.Generator().manual_seed(config['runtime']['seed'] + 101)
        normals = torch.randn(count // 2, *state.links.omega_mean.shape, dtype=config.dtype, generator=generator)
        normals = torch.cat((normals, -normals))
        omega = state.links.omega_mean + normals / state.links.precision.sqrt()
        # Means are zero, so the complete sampled objective depends on each link
        # through its sender covariance coefficient times exp(2*omega).
        values = baseline + ((torch.exp(2 * omega[..., 0]) - torch.exp(2 * state.links.omega_mean[:, 0])) * coefficient).sum(-1)
        pair_means = 0.5 * (values[:count // 2] + values[count // 2:])
        expected_error = pair_means.std(unbiased=True) / math.sqrt(count // 2)
        diagnostics = independent_diagnostics(state, model)
        torch.testing.assert_close(torch.tensor(diagnostics['diagnostic_vfe_standard_error'], dtype=config.dtype), expected_error)
        torch.testing.assert_close(torch.tensor(diagnostics['diagnostic_vfe_mean'], dtype=config.dtype), values.mean())
        assert diagnostics['diagnostic_independent_pairs'] == count // 2
        models.append(model)
        errors.append(diagnostics['diagnostic_vfe_standard_error'])
    assert errors[0] != errors[1]  # No monotonic-error assertion for these random samples.
    assert torch.equal(models[0].initial_links.omega_mean, models[1].initial_links.omega_mean)
    assert torch.equal(models[0].initial_links.precision, models[1].initial_links.precision)
    assert torch.equal(models[0].rows.log_pi, models[1].rows.log_pi)
    assert all(torch.equal(a.observed, b.observed) for a, b in zip(models[0].records, models[1].records))
