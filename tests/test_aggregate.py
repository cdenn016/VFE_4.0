from pathlib import Path
import json

import pytest

from vfe4.aggregate import RunTable, aggregate_metric, load_runs


def test_constant_interval() -> None:
    table = RunTable.from_columns({'run_id': ['a', 'b', 'c', 'a', 'b', 'c'],
        'sweep': [0, 0, 0, 1, 1, 1], 'vfe_total': [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]}, {})
    out = aggregate_metric(table, 'vfe_total', 'sweep', 99, 'percentile', 0)
    assert out['mean'] == [1.0, 2.0]
    assert out['ci_low'] == [1.0, 2.0] and out['ci_high'] == [1.0, 2.0]


def test_mixed_models_refused(tmp_path: Path) -> None:
    for index, K in enumerate((2, 3)):
        folder = tmp_path / str(index)
        folder.mkdir()
        (folder / 'config.json').write_text(json.dumps({'runtime': {'seed': index}, 'bundle': {'K': K}}))
        (folder / 'metrics.csv').write_text('sweep,vfe_total\n0,1\n1,2\n')
    with pytest.raises(ValueError, match='bundle.K'):
        load_runs(tmp_path)

