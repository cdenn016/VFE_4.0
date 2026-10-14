"""Choose a batch folder and metric, then press Run."""

from pathlib import Path

from vfe4.aggregate import load_runs, aggregate_metric, save_aggregate
from vfe4.visualization import render_aggregate


CONFIG = {'runs_dir': 'runs/select-a-batch-folder', 'metric': 'vfe_total', 'by': 'sweep',
          'n_resamples': 9999, 'method': 'BCa', 'seed': 20261013}


if __name__ == '__main__':
    folder = Path(CONFIG['runs_dir'])
    if not folder.is_absolute():
        folder = Path(__file__).resolve().parent / folder
    result = aggregate_metric(load_runs(folder), CONFIG['metric'], CONFIG['by'],
                              CONFIG['n_resamples'], CONFIG['method'], CONFIG['seed'])
    save_aggregate(folder / 'aggregate.csv', result)
    render_aggregate(folder, result, CONFIG['by'], CONFIG['metric'])
    print(folder / 'aggregate.csv')
