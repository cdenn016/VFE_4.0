"""Click-to-run configurations; all modes use the same explicit schema."""

from copy import deepcopy
from math import prod
from typing import Any


DEFAULTS: dict[str, Any] = {
    'runtime': {'device': 'cpu', 'dtype': 'float64', 'seed': 20261013,
                'max_vram_gb': 12.0, 'deterministic': False},
    'manifold': {'kind': 'point', 'dimension': 0, 'periods': (), 'grid_shape': ()},
    'population': {'n_agents': 4, 'lattice_shape': (4,), 'placement': 'grid',
                   'center_grid': (), 'patch_shape': (), 'coverage': 'required'},
    'bundle': {'group': 'GL_R', 'K': 2, 'frame_field': 'constant',
               'frame_spread': 0.1, 'frame_amplitude': 0.1},
    'recognition': {'covariance_structure': 'full_patch'},
    'generative': {'bridge': 'identity_gaussian', 'bridge_std': 0.5, 'model_prior_std': 2.0},
    'observations': {'kind': 'synthetic_population', 'private_std': 0.15, 'private_fraction': 0.25},
    'communication': {'comparison': 'full_state', 'comparison_dim': None,
                      'channels': ('belief', 'model'), 'source_support': 'agent_lattice',
                      'prior_family': 'window', 'radius': 1, 'decay': 1.0, 'alpha': 2.0,
                      'include_self': True, 'max_sources': None, 'tau': 0.5,
                      'source_rate': 0.0, 'source_every': 1},
    'links': {'mode': 'held', 'shared_channels': True, 'initial_twist_std': 0.05,
              'prior_std': 0.1, 'prior_anchor': 'zero', 'generator_precision': 'diagonal',
              'mutual_loop_coefficient': 0.0, 'samples': 8, 'copy_chunk_size': 4096},
    'spatial': {'enabled': False, 'record_value': 'zero', 'noise_std': 0.5},
    'inference': {'method': 'chromatic_cavi', 'sweeps': 4, 'belief_weight': 1.0,
                  'model_weight': 0.1, 'model_every': 1, 'link_every': 5, 'link_dt': 0.02},
    'analytics': {'every': 5, 'coherence_resolutions': (0.1, 0.3, 1.0), 'coarse_max_dim': 2048},
    'output': {'directory': 'runs', 'viewer': False, 'snapshot_every': 5,
               'checkpoint_every': 25, 'figures': True},
}


def strict_merge(base: dict[str, Any], overrides: dict[str, Any], path: str = '') -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in overrides.items():
        location = f'{path}.{key}' if path else key
        if key not in base:
            raise KeyError(location)
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise TypeError(f'{location} must be a dictionary')
            result[key] = strict_merge(base[key], value, location)
        else:
            result[key] = deepcopy(value)
    return result


def _spatial(dimension: int, large: bool = False) -> dict[str, Any]:
    grid = (32, 32) if dimension == 2 else (16, 16, 16)
    centers = (32, 16) if dimension == 2 else (8, 8, 8)
    patch = (5, 5) if dimension == 2 else (3, 3, 3)
    if not large:
        # Smallest nondegenerate layout: overlapping 3^d patches give spatial records,
        # cross-node covariance and inter-agent copies.
        grid, centers, patch = (4,) * dimension, (2,) * dimension, (3,) * dimension
    return strict_merge(DEFAULTS, {
        'runtime': {'device': 'cuda' if large else 'cpu'},
        'manifold': {'kind': 'flat_torus', 'dimension': dimension,
                     'periods': (1.0,) * dimension, 'grid_shape': grid},
        'population': {'n_agents': prod(centers), 'center_grid': centers, 'patch_shape': patch},
        'bundle': {'K': 4 if large else 2, 'frame_field': 'periodic_fourier'},
        'observations': {'kind': 'synthetic_periodic_field'},
        'communication': {'source_support': 'overlap', 'prior_family': 'exponential'},
        'spatial': {'enabled': True}, 'inference': {'sweeps': 100 if large else 4},
    })


PRESETS = {
    'tiny_point': deepcopy(DEFAULTS),
    'tiny_spatial_2d': _spatial(2),
    'tiny_spatial_3d': _spatial(3),
    'point_lattice_512': strict_merge(DEFAULTS, {
        'runtime': {'device': 'cuda'}, 'population': {'n_agents': 512, 'lattice_shape': (32, 16)},
        'bundle': {'K': 4}, 'communication': {'radius': 2}, 'inference': {'sweeps': 100}}),
    'spatial_2d_512': _spatial(2, True), 'spatial_3d_512': _spatial(3, True),
}
PRESETS['spatial_2d_1024_sparse'] = strict_merge(PRESETS['spatial_2d_512'], {
    'population': {'n_agents': 1024, 'center_grid': (32, 32), 'patch_shape': (3, 3)},
    'communication': {'max_sources': 4}, 'inference': {'sweeps': 20}})


def resolve_preset(name: str, overrides: dict[str, Any]) -> dict[str, Any]:
    if name not in PRESETS:
        raise KeyError(f'Unknown preset {name!r}; choose {tuple(PRESETS)}')
    return strict_merge(PRESETS[name], overrides)
