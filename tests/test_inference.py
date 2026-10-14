import torch

from vfe4.config import load_config
from vfe4.presets import resolve_preset
from vfe4.model import build_model, initialize_sections
from vfe4.objective import evaluate_vfe
from vfe4.inference import make_schedule, step
from vfe4.gaussian import damp_moments


def test_moment_damping() -> None:
    mu = torch.tensor([0.0], dtype=torch.float64)
    target = torch.tensor([2.0], dtype=mu.dtype)
    mu_new, covariance = damp_moments(mu, target, torch.tensor([[1.0]], dtype=mu.dtype),
                                    torch.tensor([[3.0]], dtype=mu.dtype), 0.5)
    torch.testing.assert_close(mu_new, torch.tensor([1.0], dtype=mu.dtype))
    torch.testing.assert_close(covariance, torch.tensor([[3.0]], dtype=mu.dtype))


def test_colored_point_and_spatial_descent() -> None:
    for name in ('tiny_point', 'tiny_spatial_2d', 'tiny_spatial_3d'):
        config = load_config(resolve_preset(name, {}))
        model = build_model(config)
        state = initialize_sections(model, config)
        schedule = make_schedule(model, config)
        old = evaluate_vfe(state, model).total.item()
        for _ in range(3):
            state, report = step(state, model, schedule, config)
            assert report.objective_after <= old + 1e-8
            old = report.objective_after


def test_patch_cross_node_coupling_descent() -> None:
    config = load_config(resolve_preset('tiny_spatial_2d', {
        'manifold': {'grid_shape': (3, 3)},
        'population': {'n_agents': 1, 'center_grid': (1, 1), 'patch_shape': (3, 3)},
        'observations': {'private_fraction': 1.0},
        'bundle': {'K': 1},
    }))
    model = build_model(config)
    state = initialize_sections(model, config)
    state, report = step(state, model, make_schedule(model, config), config)
    assert report.objective_after < report.objective_before
    cov = state.belief.buckets[0].covariance()[0]
    assert torch.count_nonzero(cov - torch.diag(cov.diagonal())) > 0
