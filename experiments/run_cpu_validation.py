"""Tiny repeatable artifact validation, safe while another CUDA workload is active."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from vfe4.aggregate import load_runs, aggregate_metric, save_aggregate
from vfe4.batch import run_batch
from vfe4.config import load_config
from vfe4.exchange import export_context
from vfe4.io import load_checkpoint, write_json
from vfe4.presets import resolve_preset
from vfe4.simulation import run_simulation
from vfe4.visualization import render_aggregate
from experiments.run_formation import run_controls


def run_validation() -> dict:
    torch.set_num_threads(1)
    output = {'runs': [], 'batch': '', 'controls': {}}
    for preset in ('tiny_point', 'tiny_spatial_2d', 'tiny_spatial_3d'):
        overrides = {'output': {'directory': 'runs/cpu-validation-final'}}
        if preset != 'tiny_point':
            d = 2 if preset == 'tiny_spatial_2d' else 3
            overrides.update(manifold={'grid_shape': (3,) * d}, population={'patch_shape': (3,) * d})
        config = load_config(resolve_preset(preset, overrides))
        result = run_simulation(config)
        model, state, _ = load_checkpoint(result.run_directory / 'final_state.pt')
        export_context(result.run_directory / 'context0.pt', state, model, 0)
        output['runs'].append(str(result.run_directory))
    config = load_config(resolve_preset('tiny_point', {'population': {'n_agents': 2, 'lattice_shape': (2,)},
        'links': {'mode': 'latent_gaussian', 'generator_precision': 'full', 'samples': 4, 'mutual_loop_coefficient': 0.1},
        'communication': {'comparison': 'projection', 'comparison_dim': 1, 'source_rate': 0.02},
        'inference': {'link_every': 1, 'sweeps': 5}, 'analytics': {'every': 1},
        'output': {'directory': 'runs/cpu-validation-final'}}))
    result = run_simulation(config)
    output['runs'].append(str(result.run_directory))
    base = resolve_preset('tiny_point', {'output': {'directory': 'runs/cpu-batches-final', 'figures': False}})
    paths = run_batch(base, [1, 2, 3], {})
    table = load_runs(paths[0].parent)
    aggregate = aggregate_metric(table, 'vfe_total', 'sweep', 999, 'percentile', 0)
    save_aggregate(paths[0].parent / 'aggregate.csv', aggregate)
    render_aggregate(paths[0].parent, aggregate, 'sweep', 'vfe_total')
    output['batch'] = str(paths[0].parent)
    base = resolve_preset('tiny_point', {'communication': {'source_rate': 0.1},
        'output': {'directory': 'runs/cpu-formation', 'figures': False}, 'analytics': {'every': 1}})
    controls = run_controls(base, [1])
    output['controls'] = {arm: [str(path) for path in paths] for arm, paths in controls.items()}
    write_json(ROOT / 'docs/implementation/cpu-artifacts.json', output)
    return output


if __name__ == '__main__':
    print(run_validation())
