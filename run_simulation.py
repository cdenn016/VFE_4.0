"""Edit CONFIG below, select C:/anaconda/python.exe, then press Run."""

from vfe4.config import load_config
from vfe4.simulation import run_simulation


# All supported settings are visible here and passed directly to load_config.
# These values preserve the validated point512 run. README lists the coordinated
# changes for the validated periodic 2D and 3D configurations.
CONFIG = {
    'runtime': {
        'device': 'cuda',                 # 'cuda' or 'cpu'
        'dtype': 'float64',               # 'float64' or 'float32'
        'seed': 20261013,
        'max_vram_gb': 12.0,
        'deterministic': False,
    },
    'manifold': {
        'kind': 'point',                  # 'point' or 'flat_torus'
        'dimension': 0,                   # 0 for a point; 2 or 3 for flat_torus
        'periods': (),                    # Spatial example: (1.0, 1.0)
        'grid_shape': (),                 # Spatial example: (32, 32)
    },
    'population': {
        'n_agents': 512,
        'lattice_shape': (32, 16),        # Point-mode shape; product equals n_agents
        'placement': 'grid',              # 'grid' or 'random'
        'center_grid': (),                # Spatial grid placement; product equals n_agents
        'patch_shape': (),                # Spatial patch widths, positive and odd
        'coverage': 'required',           # 'required' or 'allow_uncovered'
    },
    'bundle': {
        'group': 'GL_R',                  # Implemented group backend
        'K': 4,
        'frame_field': 'constant',        # 'constant' or 'periodic_fourier'
        'frame_spread': 0.1,
        'frame_amplitude': 0.1,           # Used by periodic_fourier frames
    },
    'recognition': {
        'covariance_structure': 'full_patch',  # 'full_patch' or 'node_blocks'
    },
    'generative': {
        'bridge': 'identity_gaussian',
        'bridge_std': 0.5,
        'model_prior_std': 2.0,
    },
    'observations': {
        'kind': 'synthetic_population',   # Spatial mode: 'synthetic_periodic_field'
        'private_std': 0.15,
        'private_fraction': 0.25,
    },
    'communication': {
        'comparison': 'full_state',      # 'full_state' or 'projection'
        'comparison_dim': None,          # Set 1..K when using projection
        'channels': ('belief', 'model'),
        'source_support': 'agent_lattice',  # Spatial mode: 'overlap'
        'prior_family': 'window',         # window, directed_window, exponential, power, complete
        'radius': 2,
        'decay': 1.0,                     # Per lattice step or grid spacing
        'alpha': 2.0,
        'include_self': True,
        'max_sources': None,             # Optional overlap candidate cap
        'tau': 0.5,
        'source_rate': 0.0,               # 0 freezes source priors; positive values learn them
        'source_every': 1,
    },
    'links': {
        'mode': 'held',                   # 'held' or 'latent_gaussian'
        'shared_channels': True,
        'initial_twist_std': 0.05,
        'prior_std': 0.1,
        'prior_anchor': 'zero',           # 'zero' or 'fixture_twist' (held-link limit check only)
        'generator_precision': 'diagonal',  # 'diagonal' or 'full'
        'mutual_loop_coefficient': 0.0,   # Nonzero latent loop prior has unknown log Z
        'samples': 8,                     # Latent antithetic integration: even, at least 2
        'copy_chunk_size': 4096,          # Measured latent profile used 4 samples/chunks of 64
    },
    'spatial': {
        'enabled': False,                # Enable for spatial continuity records
        'record_value': 'zero',
        'noise_std': 0.5,
    },
    'inference': {
        'method': 'chromatic_cavi',       # Or 'fisher_natural' with full_patch recognition
        'sweeps': 100,
        'belief_weight': 1.0,
        'model_weight': 0.1,
        'model_every': 1,
        'link_every': 5,
        'link_dt': 0.02,
    },
    'analytics': {
        'every': 5,
        'coherence_resolutions': (0.1, 0.3, 1.0),
        'coarse_max_dim': 2048,
    },
    'output': {
        'directory': 'runs',
        'viewer': False,                  # True opens pause/resume/one-sweep/stop controls
        'snapshot_every': 5,
        'checkpoint_every': 25,
        'figures': True,
    },
}


if __name__ == '__main__':
    result = run_simulation(load_config(CONFIG))
    print(f'Finished at sweep {result.completed_sweeps}; results: {result.run_directory}')
