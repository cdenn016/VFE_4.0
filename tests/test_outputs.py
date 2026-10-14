from pathlib import Path
import csv
import json

from vfe4.config import load_config
from vfe4.presets import resolve_preset
from vfe4.simulation import run_simulation


def test_self_describing_run(tmp_path: Path) -> None:
    config = load_config(resolve_preset('tiny_spatial_2d', {
        'output': {'directory': str(tmp_path), 'figures': False}, 'inference': {'sweeps': 2}}))
    result = run_simulation(config)
    for name in ('config.json', 'environment.json', 'geometry.pt', 'observations.pt',
                 'metrics.csv', 'summary.json', 'final_state.pt'):
        assert (result.run_directory / name).is_file()
    rows = list(csv.DictReader((result.run_directory / 'metrics.csv').open()))
    assert len(rows) == 3
    assert float(rows[-1]['vfe_total']) <= float(rows[0]['vfe_total'])
    assert json.loads((result.run_directory / 'summary.json').read_text())['status'] == 'completed'
