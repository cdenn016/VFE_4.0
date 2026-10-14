from pathlib import Path

import torch

from vfe4.config import load_config
from vfe4.presets import resolve_preset
from vfe4.model import build_model, initialize_sections
from vfe4.objective import dense_action
from vfe4.exchange import context_snapshot, export_context
from vfe4.inference import make_schedule, step
from vfe4.analytics.observables import dense_recognition


def test_conditioned_context_matches_dense_action(tmp_path: Path) -> None:
    config = load_config(resolve_preset('tiny_spatial_2d', {'manifold': {'grid_shape': (3, 3)},
        'bundle': {'K': 1}, 'population': {'center_grid': (1, 1), 'n_agents': 1, 'patch_shape': (3, 3)}}))
    model = build_model(config)
    state = initialize_sections(model, config)
    state, _ = step(state, model, make_schedule(model, config), config)
    boundary = torch.arange(18, dtype=config.dtype).reshape(2, 9, 1) * 0.03
    snapshot = context_snapshot(state, model, 0, boundary)
    J, h, c = dense_action(state, model)
    retained = snapshot['retained_coordinates']
    x = boundary.flatten().clone()
    x[retained] = 0
    torch.testing.assert_close(snapshot['J'], J[retained[:, None], retained[None, :]])
    torch.testing.assert_close(snapshot['h'], h[retained] - (J @ x)[retained])
    torch.testing.assert_close(snapshot['c'], c + 0.5 * x @ J @ x - h @ x)
    mu_q, Sigma_q = dense_recognition(state, model)
    omitted = torch.arange(18)[~torch.isin(torch.arange(18), retained)]
    cross = Sigma_q[retained[:, None], omitted[None, :]]
    sigma_bb = Sigma_q[omitted[:, None], omitted[None, :]]
    expected_mu = mu_q[retained] + cross @ torch.linalg.solve(sigma_bb, boundary.flatten()[omitted] - mu_q[omitted])
    expected_cov = Sigma_q[retained[:, None], retained[None, :]] - cross @ torch.linalg.solve(sigma_bb, cross.mT)
    torch.testing.assert_close(snapshot['recognition_mean'], expected_mu)
    torch.testing.assert_close(snapshot['recognition_covariance'], expected_cov)
    path = export_context(tmp_path / 'context.pt', state, model, 0, boundary)
    assert torch.load(path, weights_only=False)['model_type'] == 'conditioned'


def test_spatial_disabled_reduction() -> None:
    config = load_config(resolve_preset('tiny_point', {'population': {'n_agents': 1, 'lattice_shape': (1,)}}))
    model = build_model(config)
    state = initialize_sections(model, config)
    snapshot = context_snapshot(state, model, 0)
    J, h, c = dense_action(state, model)
    torch.testing.assert_close(snapshot['J'], J)
    torch.testing.assert_close(snapshot['h'], h)
    torch.testing.assert_close(snapshot['c'], c)
    assert snapshot['model_type'] == 'spatial_disabled'
