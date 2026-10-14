"""Large populations are experiments, executed sequentially after CUDA admission."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vfe4.aggregate import load_runs, aggregate_metric, save_aggregate
from vfe4.batch import run_batch
from vfe4.config import load_config
from vfe4.io import write_json
from vfe4.preflight import storage_preflight
from vfe4.presets import resolve_preset
from vfe4.simulation import run_simulation, ensure_cuda_capacity
from vfe4.visualization import render_aggregate

PROFILES = ['point_lattice_512', 'spatial_2d_512', 'spatial_3d_512', 'spatial_2d_1024_sparse', 'point_512_uncertain']
SEEDS = [20261013, 20261014, 20261015, 20261016, 20261017]
EVIDENCE = ROOT / 'docs/performance/scaling.json'


def profile_config(name: str) -> dict:
    if name == 'point_512_uncertain':
        return resolve_preset('point_lattice_512', {'communication': {'radius': 1},
            'links': {'mode': 'latent_gaussian', 'samples': 4, 'copy_chunk_size': 64},
            'inference': {'sweeps': 20, 'link_every': 5}, 'analytics': {'every': 10},
            'output': {'directory': 'runs/scaling/point_512_uncertain', 'figures': False}})
    return resolve_preset(name, {'output': {'directory': f'runs/scaling/{name}', 'figures': False}})


def run_scaling() -> dict:
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    evidence = json.loads(EVIDENCE.read_text()) if EVIDENCE.exists() else {'profiles': {}, 'batch': {}}
    evidence['last_checked_utc'] = datetime.now(timezone.utc).isoformat()
    for name in PROFILES:
        if evidence['profiles'].get(name, {}).get('status') == 'completed':
            continue
        config = load_config(profile_config(name))
        record = {'config': config.raw, 'preflight': storage_preflight(config)}
        try:
            ensure_cuda_capacity(config)
            result = run_simulation(config)
            record.update(json.loads((result.run_directory / 'summary.json').read_text()))
            record['run_directory'] = str(result.run_directory)
        except RuntimeError as error:
            record.update(status='deferred' if 'compute processes' in str(error) else 'failed', reason=str(error))
        except MemoryError as error:
            record.update(status='not_admitted', reason=str(error))
        evidence['profiles'][name] = record
        write_json(EVIDENCE, evidence)
        if record['status'] == 'deferred':
            # Record every configured profile without launching another job.
            for remaining in PROFILES:
                if remaining not in evidence['profiles']:
                    remaining_config = load_config(profile_config(remaining))
                    evidence['profiles'][remaining] = {'status': 'deferred', 'reason': record['reason'],
                        'config': remaining_config.raw, 'preflight': storage_preflight(remaining_config)}
            evidence['batch'] = {'status': 'deferred', 'reason': record['reason'], 'seeds': SEEDS}
            write_json(EVIDENCE, evidence)
            return evidence
    if evidence['batch'].get('status') != 'completed':
        try:
            raw = resolve_preset('point_lattice_512', {'output': {'directory': 'runs/scaling/seeds', 'figures': False}})
            paths = run_batch(raw, SEEDS, {})
            folder = paths[0].parent
            data = aggregate_metric(load_runs(folder), 'vfe_total', 'sweep', 9999, 'BCa', 20261013)
            save_aggregate(folder / 'aggregate.csv', data)
            render_aggregate(folder, data, 'sweep', 'vfe_total')
            evidence['batch'] = {'status': 'completed', 'directory': str(folder), 'seeds': SEEDS}
        except (RuntimeError, ValueError, MemoryError) as error:
            evidence['batch'] = {'status': 'deferred' if 'compute processes' in str(error) else 'failed', 'reason': str(error)}
        write_json(EVIDENCE, evidence)
    return evidence


if __name__ == '__main__':
    result = run_scaling()
    print({name: value['status'] for name, value in result['profiles'].items()}, result['batch'])
