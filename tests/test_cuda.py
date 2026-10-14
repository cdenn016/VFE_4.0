"""Explicit CUDA lane; tiny fixed-data agreement, never production scale."""

from copy import deepcopy
from dataclasses import fields, is_dataclass
import os

import pytest
import torch

from vfe4.config import load_config, SimulationConfig
from vfe4.presets import resolve_preset
from vfe4.model import build_model, initialize_sections
from vfe4.objective import evaluate_vfe
from vfe4.inference import make_schedule, step
from vfe4.geometry.groups import GLReal
from vfe4.latent_links import derivatives

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(os.environ.get('VFE4_TEST_DEVICE') != 'cuda', reason='explicit CUDA lane only')]


def _move(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.to('cuda')
    if isinstance(value, SimulationConfig):
        raw = deepcopy(value.raw)
        raw['runtime']['device'] = 'cuda'
        return load_config(raw)
    if isinstance(value, tuple):
        return tuple(_move(item) for item in value)
    if is_dataclass(value):
        for field in fields(value):
            setattr(value, field.name, _move(getattr(value, field.name)))
    elif hasattr(value, '__dict__') and value.__class__.__module__.startswith('vfe4.'):
        for key, item in vars(value).copy().items():
            setattr(value, key, _move(item))
    return value


@pytest.mark.parametrize('preset', ['tiny_point', 'tiny_spatial_2d', 'tiny_spatial_3d'])
@pytest.mark.parametrize('dtype', ['float64', 'float32'])
def test_same_data_cpu_cuda(preset: str, dtype: str) -> None:
    assert torch.cuda.is_available(), 'CUDA lane must select the actual GPU'
    overrides = {'runtime': {'dtype': dtype}}
    if preset != 'tiny_point':
        d = 2 if preset == 'tiny_spatial_2d' else 3
        overrides.update(manifold={'grid_shape': (3,) * d},
                         population={'n_agents': 1, 'center_grid': (1,) * d, 'patch_shape': (3,) * d})
    config = load_config(resolve_preset(preset, overrides))
    cpu_model = build_model(config)
    cpu_state = initialize_sections(cpu_model, config)
    gpu_model, gpu_state = _move(deepcopy(cpu_model)), _move(deepcopy(cpu_state))
    assert gpu_state.beta.is_cuda and gpu_model.domain.contexts.is_cuda
    tolerance = {'atol': 1e-8, 'rtol': 1e-7} if dtype == 'float64' else {'atol': 2e-4, 'rtol': 2e-4}
    torch.testing.assert_close(evaluate_vfe(cpu_state, cpu_model).total,
                               evaluate_vfe(gpu_state, gpu_model).total.cpu(), **tolerance)
    cpu_state, cpu_report = step(cpu_state, cpu_model, make_schedule(cpu_model, config), config)
    gpu_state, gpu_report = step(gpu_state, gpu_model, make_schedule(gpu_model, gpu_model.config), gpu_model.config)
    assert cpu_report.accepted and gpu_report.accepted
    for left, right in zip(cpu_state.belief.buckets, gpu_state.belief.buckets):
        torch.testing.assert_close(left.mu, right.mu.cpu(), **tolerance)
        torch.testing.assert_close(left.covariance(), right.covariance().cpu(), **tolerance)


def test_gl_shear_scale_cuda() -> None:
    left = torch.tensor([[1.3, 0.4], [0., 0.8]], dtype=torch.float64)
    right = torch.tensor([[0.7, 0.], [-0.2, 1.2]], dtype=torch.float64)
    group = GLReal(2)
    torch.testing.assert_close(group.ratio(left, right), group.ratio(left.cuda(), right.cuda()).cpu(), atol=1e-8, rtol=1e-7)


def test_true_generator_derivatives_cuda() -> None:
    omega = torch.tensor([[0.1, 0.2, -0.1, 0.05]], dtype=torch.float64)
    z = torch.tensor([[0.2, 0.8]], dtype=omega.dtype)
    mu = torch.tensor([[0.4, -0.3]], dtype=omega.dtype)
    Sigma = torch.eye(2, dtype=omega.dtype).unsqueeze(0) * 0.2
    G = torch.eye(2, dtype=omega.dtype).unsqueeze(0)
    cpu = derivatives(omega, z, mu, Sigma, G)
    gpu = derivatives(*(value.cuda() for value in (omega, z, mu, Sigma, G)))
    for expected, actual in zip(cpu, gpu):
        torch.testing.assert_close(expected, actual.cpu(), atol=1e-8, rtol=1e-7)


def test_native_cuda_latent_step() -> None:
    config = load_config(resolve_preset('tiny_point', {'runtime': {'device': 'cuda'},
        'population': {'n_agents': 2, 'lattice_shape': (2,)},
        'links': {'mode': 'latent_gaussian', 'samples': 4, 'copy_chunk_size': 16, 'mutual_loop_coefficient': 0.1},
        'inference': {'link_every': 1}}))
    model = build_model(config)
    state = initialize_sections(model, config)
    assert state.links.omega_mean.is_cuda
    state, report = step(state, model, make_schedule(model, config), config)
    assert report.accepted and report.objective_after <= report.objective_before + 1e-7
    assert bool(torch.isfinite(state.links.precision).all())


def test_sampled_k4_transport_chunks_cuda() -> None:
    from vfe4.latent_links import sampled_generators
    config = load_config(resolve_preset('tiny_point', {
        'population': {'n_agents': 3, 'lattice_shape': (3,)}, 'bundle': {'K': 4},
        'links': {'mode': 'latent_gaussian', 'samples': 4, 'copy_chunk_size': 2}}))
    model = build_model(config)
    state = initialize_sections(model, config)
    omega = sampled_generators(state.links, 4)
    T = torch.matrix_exp(omega.reshape(4, -1, 4, 4))
    G = torch.eye(4, dtype=config.dtype) / config['communication']['tau']
    gpu_model, gpu_state = _move(deepcopy(model)), _move(deepcopy(state))
    mean, gram = gpu_model.transport_moments(gpu_state)
    torch.testing.assert_close(mean.cpu(), T.mean(0), atol=1e-8, rtol=1e-7)
    torch.testing.assert_close(gram.cpu(), (T.mT @ G @ T).mean(0), atol=1e-8, rtol=1e-7)
