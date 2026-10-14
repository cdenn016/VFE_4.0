"""Seed-level bootstrap aggregation; model differences never disappear silently."""

from copy import deepcopy
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import bootstrap


@dataclass
class RunTable:
    rows: list[dict[str, Any]]
    configs: dict[str, dict[str, Any]]
    swept_keys: tuple[str, ...] = ()

    @classmethod
    def from_columns(cls, columns: dict[str, list], configs: dict) -> 'RunTable':
        lengths = {len(values) for values in columns.values()}
        if len(lengths) != 1:
            raise ValueError('Columns must have the same number of rows')
        return cls([dict(zip(columns, row)) for row in zip(*columns.values())], configs)


def _without_seed(config: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(config)
    result.get('runtime', {}).pop('seed', None)
    return result


def _difference(a: dict, b: dict, path: str = '') -> str | None:
    for key in sorted(a.keys() | b.keys()):
        location = f'{path}.{key}' if path else key
        if key not in a or key not in b:
            return location
        if isinstance(a[key], dict) and isinstance(b[key], dict):
            nested = _difference(a[key], b[key], location)
            if nested:
                return nested
        elif a[key] != b[key]:
            return location
    return None


def load_runs(runs_dir: Path, swept_keys: tuple[str, ...] = ()) -> RunTable:
    rows, configs = [], {}
    reference = None
    for folder in sorted(runs_dir.iterdir()):
        if not (folder / 'config.json').is_file() or not (folder / 'metrics.csv').is_file():
            continue
        config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
        normalized = _without_seed(config)
        values = {}
        for dotted in swept_keys:
            parts = dotted.split('.')
            section = normalized
            for part in parts[:-1]:
                section = section[part]
            values[dotted] = section.pop(parts[-1])
        if reference is None:
            reference = normalized
        difference = _difference(reference, normalized)
        if difference:
            raise ValueError(f'Cannot pool different resolved configs: {difference}')
        configs[folder.name] = config
        if (folder / 'summary.json').exists():
            summary = json.loads((folder / 'summary.json').read_text())
            if summary['status'] != 'completed':
                raise ValueError(f'Run {folder.name} failed; inspect its summary before aggregation')
        with (folder / 'metrics.csv').open(newline='', encoding='utf-8') as file:
            for row in csv.DictReader(file):
                converted: dict[str, Any] = {'run_id': folder.name, **values}
                for key, value in row.items():
                    try:
                        converted[key] = float(value) if value else None
                    except ValueError:
                        converted[key] = value
                rows.append(converted)
    if not rows:
        raise ValueError(f'No run metrics in {runs_dir}')
    return RunTable(rows, configs, swept_keys)


def aggregate_metric(table: RunTable, metric: str, by: str | tuple[str, ...], n_resamples: int,
                     method: str, seed: int) -> dict[str, list[float]]:
    groups: dict[Any, list[float]] = {}
    group_keys = (by,) if isinstance(by, str) else by
    if any(key not in group_keys for key in table.swept_keys):
        raise ValueError('All swept keys must be explicit aggregation groups')
    seen = set()
    for row in table.rows:
        if row.get(metric) is None:
            continue
        group = tuple(row[key] for key in group_keys)
        identity = (row['run_id'], group)
        if identity in seen:
            raise ValueError('Aggregation requires one measurement per independent run per group')
        seen.add(identity)
        groups.setdefault(group, []).append(float(row[metric]))
    out = {**{key: [] for key in group_keys}, 'mean': [], 'ci_low': [], 'ci_high': [], 'n_runs': []}
    for group, values in sorted(groups.items()):
        sample = np.asarray(values)
        if sample.size < 2:
            raise ValueError('A bootstrap interval requires at least two independent runs')
        if not np.isfinite(sample).all():
            raise ValueError('Cannot aggregate nonfinite measurements')
        if np.all(sample == sample[0]):
            low = high = float(sample[0])
        else:
            interval = bootstrap((sample,), np.mean, method=method, n_resamples=n_resamples,
                                 rng=np.random.default_rng(seed)).confidence_interval
            low, high = float(interval.low), float(interval.high)
            if not np.isfinite([low, high]).all():
                raise ValueError('Bootstrap interval is degenerate; choose percentile or more seeds explicitly')
        for key, value in zip(group_keys, group):
            out[key].append(value)
        out['mean'].append(float(sample.mean()))
        out['ci_low'].append(low)
        out['ci_high'].append(high)
        out['n_runs'].append(int(sample.size))
    return out


def save_aggregate(path: Path, data: dict[str, list[float]]) -> None:
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(data)
        writer.writerows(zip(*data.values()))
