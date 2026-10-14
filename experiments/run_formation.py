"""Matched source-learning/frozen controls; descriptive, not a formation theorem test."""

from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vfe4.batch import run_batch
from vfe4.presets import resolve_preset


PRESET = 'point_lattice_512'
OVERRIDES = {'communication': {'source_rate': 0.02}, 'output': {'viewer': False}}
SEEDS = [20261013, 20261014, 20261015, 20261016, 20261017]


def run_controls(base: dict, seeds: list[int]) -> dict[str, list[Path]]:
    results = {}
    for arm in ('learned', 'frozen'):
        raw = deepcopy(base)
        raw['output']['directory'] = str(Path(base['output']['directory']) / arm)
        if arm == 'frozen':
            raw['communication']['source_rate'] = 0.0
        results[arm] = run_batch(raw, seeds, {})
    render_controls(results)
    return results


def render_controls(results: dict[str, list[Path]]) -> None:
    """Matched arms on every formation chart; private fixture is the declared seed datum."""
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from vfe4.io import write_json
    destination = results['learned'][0].parents[2] / 'formation-comparison'
    destination.mkdir(parents=True, exist_ok=True)
    histories = {arm: [list(csv.DictReader((path / 'metrics.csv').open())) for path in paths]
                 for arm, paths in results.items()}
    # VFE falls in the learned arm by construction of the source M-step; fixture RMSE is the external referent.
    keys = ['vfe_total', 'vfe_categorical_kl', 'field_rmse_reference', 'effective_degree', 'self_weight',
            'routing_reciprocity_reference_mean_abs']
    keys += [key for key in histories['learned'][0][0] if key.startswith('coherent_blocks_')]
    for metric in keys:
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        for arm, runs in histories.items():
            records = [{int(row['sweep']): float(row[metric]) for row in run if row.get(metric)} for run in runs]
            times = sorted(set.intersection(*(set(run) for run in records)))
            values = np.array([[run[t] for t in times] for run in records])
            ax.plot(times, values.mean(0), label=arm)
        ax.set(xlabel='Inference sweep', ylabel=metric.replace('_', ' '), title='Matched source learning controls; descriptive mean-field runs')
        ax.title.set_fontsize(10)
        ax.legend()
        for extension in ('png', 'pdf', 'svg'):
            fig.savefig(destination / f'{metric}.{extension}', bbox_inches='tight', dpi=300)
        plt.close(fig)
    write_json(destination / 'design.json', {'arms': {arm: [str(path) for path in paths] for arm, paths in results.items()},
        'matching': 'same seed, geometry, private observations and inter-agent records; only source_rate differs',
        'symmetry_group': 'translations of the periodic agent lattice (population.lattice_shape)',
        'symmetry_breaking_datum': ('seeded asymmetric synthetic population, noisy private observations, and the seeded '
                                    'per-copy twist realization; initial recognition means are zero, so sweep-0 '
                                    'coherence is fixed by initialization'),
        'learned_arm_vfe': 'lower by construction: the source step minimizes the categorical KL at fixed Q',
        'scope': 'descriptive finite mean-field CAVI; does not test stochastic formation conjectures'})


if __name__ == '__main__':
    print(run_controls(resolve_preset(PRESET, OVERRIDES), SEEDS))
