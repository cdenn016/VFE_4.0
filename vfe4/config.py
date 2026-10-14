"""Validated resolved configuration with no implicit device fallback."""

from dataclasses import dataclass
from math import prod
from typing import Any

import torch

from .presets import DEFAULTS, strict_merge


@dataclass(frozen=True)
class SimulationConfig:
    raw: dict[str, Any]
    device: torch.device
    dtype: torch.dtype

    def __getitem__(self, key: str) -> dict[str, Any]:
        return self.raw[key]


def load_config(raw: dict[str, Any]) -> SimulationConfig:
    config = strict_merge(DEFAULTS, raw)
    device = torch.device(config['runtime']['device'])
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('runtime.device must be cpu or cuda')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; explicitly choose CPU for tiny validation')
    if config['runtime']['dtype'] not in ('float64', 'float32'):
        raise ValueError('runtime.dtype must be float64 or float32')
    dtype = getattr(torch, config['runtime']['dtype'])
    manifold, population = config['manifold'], config['population']
    N, K = population['n_agents'], config['bundle']['K']
    if not isinstance(N, int) or N < 1 or not isinstance(K, int) or K < 1:
        raise ValueError('n_agents and K must be positive integers')
    if config['bundle']['group'] != 'GL_R':
        raise ValueError('Only the implemented GL_R backend is available')
    if manifold['kind'] == 'point':
        if manifold['dimension'] != 0 or manifold['periods'] or manifold['grid_shape']:
            raise ValueError('point mode has dimension 0 and no periods/grid_shape')
        if prod(population['lattice_shape']) != N:
            raise ValueError('population.lattice_shape must multiply to n_agents')
        if config['spatial']['enabled'] or config['communication']['source_support'] != 'agent_lattice':
            raise ValueError('point mode requires agent_lattice support and no spatial records')
        if (population['placement'] != 'grid' or population['center_grid'] or population['patch_shape']
                or population['coverage'] != 'required' or config['bundle']['frame_field'] != 'constant'):
            raise ValueError('point mode reads no placement, center_grid, patch_shape, coverage or Fourier frames; leave their defaults')
    elif manifold['kind'] == 'flat_torus':
        d = manifold['dimension']
        if d not in (2, 3) or any(len(manifold[key]) != d for key in ('periods', 'grid_shape')):
            raise ValueError('manifold dimension must agree with periods and grid_shape in 2D/3D')
        if any(x <= 0 for x in manifold['periods']) or any(x < 2 for x in manifold['grid_shape']):
            raise ValueError('periods must be positive and grid_shape at least 2')
        if len(population['patch_shape']) != d or any(x < 1 or x % 2 == 0 for x in population['patch_shape']):
            raise ValueError('patch_shape requires positive odd widths in each dimension')
        if any(p > n for p, n in zip(population['patch_shape'], manifold['grid_shape'])):
            raise ValueError('patch_shape cannot exceed grid_shape')
        if population['placement'] == 'grid' and (len(population['center_grid']) != d or prod(population['center_grid']) != N):
            raise ValueError('population.center_grid must match dimension and n_agents')
        if config['communication']['source_support'] != 'overlap':
            raise ValueError('spatial mode requires overlap source support')
    else:
        raise ValueError('Only point and flat_torus manifolds are implemented')
    allowed = {
        ('recognition', 'covariance_structure'): ('full_patch', 'node_blocks'),
        ('links', 'mode'): ('held', 'latent_gaussian'),
        ('links', 'generator_precision'): ('diagonal', 'full'),
        ('links', 'prior_anchor'): ('zero', 'fixture_twist'),
        ('bundle', 'frame_field'): ('constant', 'periodic_fourier'),
        ('inference', 'method'): ('chromatic_cavi', 'fisher_natural'),
        ('communication', 'comparison'): ('full_state', 'projection'),
        ('population', 'placement'): ('grid', 'random'),
        ('population', 'coverage'): ('required', 'allow_uncovered'),
    }
    for (section, key), values in allowed.items():
        if config[section][key] not in values:
            raise ValueError(f'{section}.{key} must be one of {values}')
    for section, key in [('generative', 'bridge_std'), ('generative', 'model_prior_std'),
                         ('observations', 'private_std'), ('communication', 'tau'),
                         ('spatial', 'noise_std'), ('links', 'prior_std'), ('runtime', 'max_vram_gb')]:
        if config[section][key] <= 0:
            raise ValueError(f'{section}.{key} must be positive')
    for section, key in [('inference', 'belief_weight'), ('inference', 'model_weight')]:
        if not 0 < config[section][key] <= 1:
            raise ValueError(f'{section}.{key} must belong to (0,1]')
    if not 0 <= config['communication']['source_rate'] <= 1:
        raise ValueError('source_rate must belong to [0,1]')
    if not 0 <= config['observations']['private_fraction'] <= 1:
        raise ValueError('private_fraction must belong to [0,1]')
    for section, key in [('inference', 'model_every'), ('inference', 'link_every'),
                         ('communication', 'source_every'), ('links', 'samples'),
                         ('links', 'copy_chunk_size'), ('analytics', 'every'),
                         ('output', 'snapshot_every'), ('output', 'checkpoint_every')]:
        if not isinstance(config[section][key], int) or config[section][key] < 1:
            raise ValueError(f'{section}.{key} must be a positive integer')
    sweeps = config['inference']['sweeps']
    if not isinstance(sweeps, int) or isinstance(sweeps, bool) or sweeps < 0:
        raise ValueError('inference.sweeps must be a nonnegative integer')
    if any(not resolution > 0 for resolution in config['analytics']['coherence_resolutions']):
        raise ValueError('analytics.coherence_resolutions must be positive')
    max_dim = config['analytics']['coarse_max_dim']
    if not isinstance(max_dim, int) or isinstance(max_dim, bool) or max_dim < 1:
        raise ValueError('analytics.coarse_max_dim must be a positive integer')
    if config['links']['samples'] < 2 or config['links']['samples'] % 2:
        raise ValueError('links.samples must be even and at least 2 for antithetic integration')
    if not 0 < config['inference']['link_dt'] <= 1:
        raise ValueError('inference.link_dt must belong to (0,1]')
    if config['links']['mutual_loop_coefficient'] < 0:
        raise ValueError('links.mutual_loop_coefficient must be nonnegative')
    channels = config['communication']['channels']
    if not channels or len(set(channels)) != len(channels) or any(x not in ('belief', 'model') for x in channels):
        raise ValueError('communication.channels must be unique belief/model names')
    comm = config['communication']
    if comm['prior_family'] not in ('window', 'directed_window', 'exponential', 'power', 'complete'):
        raise ValueError('Unsupported communication.prior_family')
    if not isinstance(comm['radius'], int) or comm['radius'] < 1 or comm['decay'] < 0 or comm['alpha'] < 0:
        raise ValueError('radius must be a positive integer; decay/alpha must be nonnegative')
    if comm['max_sources'] is not None and (not isinstance(comm['max_sources'], int) or comm['max_sources'] < 1):
        raise ValueError('communication.max_sources must be a positive integer or None')
    if manifold['kind'] == 'point' and comm['max_sources'] is not None:
        raise ValueError('max_sources applies to overlap support; use radius or prior_family for a point lattice')
    if config['inference']['method'] == 'fisher_natural' and config['recognition']['covariance_structure'] != 'full_patch':
        raise ValueError('fisher_natural currently requires full_patch recognition')
    if config['generative']['bridge'] != 'identity_gaussian' or config['spatial']['record_value'] != 'zero':
        raise ValueError('Only identity_gaussian bridge and zero spatial records are implemented')
    expected_kind = 'synthetic_population' if manifold['kind'] == 'point' else 'synthetic_periodic_field'
    if config['observations']['kind'] != expected_kind:
        raise ValueError(f'observations.kind must be {expected_kind}')
    torch.use_deterministic_algorithms(config['runtime']['deterministic'])
    return SimulationConfig(config, device, dtype)
