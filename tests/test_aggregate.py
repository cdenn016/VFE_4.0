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
        (folder / 'summary.json').write_text(json.dumps({'status': 'completed'}))
    with pytest.raises(ValueError, match='bundle.K'):
        load_runs(tmp_path)


def _fake_run(folder: Path, seed: int, summary: bool = True, K: object = 2) -> None:
    folder.mkdir()
    (folder / 'config.json').write_text(json.dumps({'runtime': {'seed': seed}, 'bundle': {'K': K}}))
    (folder / 'metrics.csv').write_text('sweep,vfe_total\n0,1\n1,2\n')
    if summary:
        (folder / 'summary.json').write_text(json.dumps({'status': 'completed'}))


def test_continuations_and_unfinished_runs_are_not_pooled(tmp_path: Path) -> None:
    _fake_run(tmp_path / 'a', 1)
    _fake_run(tmp_path / 'b', 2)
    _fake_run(tmp_path / 'c', 1)   # a resumed continuation repeats seed 1
    with pytest.raises(ValueError, match='share seed'):
        load_runs(tmp_path)
    (tmp_path / 'c' / 'summary.json').unlink()
    with pytest.raises(ValueError, match='no summary'):
        load_runs(tmp_path)


def test_tuple_valued_swept_key_groups(tmp_path: Path) -> None:
    _fake_run(tmp_path / 'a', 1, K=[3, 3])
    _fake_run(tmp_path / 'b', 2, K=[3, 3])
    _fake_run(tmp_path / 'c', 1, K=[5, 5])
    _fake_run(tmp_path / 'd', 2, K=[5, 5])
    table = load_runs(tmp_path, ('bundle.K',))
    out = aggregate_metric(table, 'vfe_total', ('sweep', 'bundle.K'), 99, 'percentile', 0)
    assert out['bundle.K'] == [(3, 3), (5, 5), (3, 3), (5, 5)]


def test_relative_vfe_not_compared_across_loop_priors() -> None:
    rows = [{'run_id': r, 'sweep': 0, 'vfe_total': v, 'links.mutual_loop_coefficient': c,
             'normalizer_status': 'relative_vfe_unknown_logZ'}
            for r, v, c in (('a', 1., .1), ('b', 2., .1), ('c', 3., .2), ('d', 4., .2))]
    table = RunTable(rows, {}, ('links.mutual_loop_coefficient',))
    with pytest.raises(ValueError, match='log Z'):
        aggregate_metric(table, 'vfe_total', ('sweep', 'links.mutual_loop_coefficient'), 99, 'percentile', 0)
