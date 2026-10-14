"""Read the completed October 14 controls and compare their saved inputs on CPU."""

import csv
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SEEDS = list(range(20261013, 20261018))
BATCHES = {'learned': 'batch-6093bf9e', 'frozen': 'batch-c4d1e9ba'}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flatten(value: object, prefix: str = '') -> dict[str, object]:
    if isinstance(value, dict):
        return {key: leaf for name, part in value.items()
                for key, leaf in flatten(part, f'{prefix}.{name}'.strip('.')).items()}
    return {prefix: value}


def compare(left: object, right: object, path: str, tensors: list[dict], differences: list[str]) -> None:
    if type(left) is not type(right):
        differences.append(path + ': type differs')
    elif isinstance(left, torch.Tensor):
        same = left.shape == right.shape and left.dtype == right.dtype
        if left.device.type != 'cpu' or right.device.type != 'cpu':
            raise RuntimeError('Artifact tensors must remain on CPU')
        if same:
            same = torch.equal(left.contiguous().reshape(-1).view(torch.uint8),
                               right.contiguous().reshape(-1).view(torch.uint8))
        tensors.append({'path': path, 'shape': list(left.shape), 'dtype': str(left.dtype),
                        'elements': left.numel(), 'bitwise_equal': same})
        if not same:
            differences.append(path + ': tensor differs')
    elif is_dataclass(left):
        for field in fields(left):
            compare(getattr(left, field.name), getattr(right, field.name),
                    path + '.' + field.name, tensors, differences)
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            differences.append(path + ': keys differ')
        for key in left.keys() & right.keys():
            compare(left[key], right[key], path + '.' + str(key), tensors, differences)
    elif isinstance(left, (tuple, list)):
        if len(left) != len(right):
            differences.append(path + ': length differs')
        for index, (a, b) in enumerate(zip(left, right)):
            compare(a, b, f'{path}[{index}]', tensors, differences)
    elif hasattr(left, '__dict__'):
        compare(vars(left), vars(right), path, tensors, differences)
    elif left != right:
        differences.append(path + ': value differs')


def main() -> None:
    design_path = ROOT / 'runs/formation-comparison/design.json'
    design = read_json(design_path)
    arms = {}
    configs = {}
    for arm, batch in BATCHES.items():
        entries = {}
        configs[arm] = {}
        for text in design['arms'][arm]:
            path = Path(text)
            if path.parent != ROOT / 'runs' / arm / batch:
                raise ValueError('Comparison design references a different batch')
            config = read_json(path / 'config.json')
            seed = config['runtime']['seed']
            if seed in entries:
                raise ValueError('Duplicate control seed')
            summary = read_json(path / 'summary.json')
            with (path / 'metrics.csv').open(newline='', encoding='utf-8') as handle:
                metrics = list(csv.DictReader(handle))
            entries[seed] = {'path': str(path), 'config_sha256': digest(path / 'config.json'),
                             'source_rate': config['communication']['source_rate'],
                             'summary': summary, 'environment': read_json(path / 'environment.json'),
                             'recorded_sweeps': [int(row['sweep']) for row in metrics],
                             'final_metrics': metrics[-1]}
            configs[arm][seed] = config
        if sorted(entries) != SEEDS:
            raise ValueError('Control seeds differ from the runner')
        arms[arm] = entries
    pairs = []
    for seed in SEEDS:
        learned, frozen = (Path(arms[arm][seed]['path']) for arm in BATCHES)
        a, b = (flatten(configs[arm][seed]) for arm in BATCHES)
        config_differences = {key: {'learned': a.get(key), 'frozen': b.get(key)}
                              for key in sorted(a.keys() | b.keys()) if a.get(key) != b.get(key)}
        tensors, differences, inputs = [], [], []
        for filename in ('geometry.pt', 'observations.pt'):
            left = torch.load(learned / filename, map_location='cpu', weights_only=False)
            right = torch.load(frozen / filename, map_location='cpu', weights_only=False)
            compare(left, right, filename, tensors, differences)
            inputs.append({'filename': filename, 'learned_sha256': digest(learned / filename),
                           'frozen_sha256': digest(frozen / filename)})
        matched = set(config_differences) == {'communication.source_rate', 'output.directory'}
        matched &= config_differences.get('communication.source_rate') == {'learned': 0.02, 'frozen': 0.0}
        completed = all(arms[arm][seed]['summary']['status'] == 'completed'
                        and arms[arm][seed]['summary']['sweeps'] == 100
                        and arms[arm][seed]['recorded_sweeps'] == list(range(101)) for arm in BATCHES)
        pairs.append({'seed': seed, 'configs_match_except_declared_fields': matched,
                      'config_differences': config_differences, 'both_completed_100_sweeps': completed,
                      'saved_geometry_records_truth_equal': not differences,
                      'tensor_count': len(tensors), 'tensor_elements_compared': sum(t['elements'] for t in tensors),
                      'tensor_comparisons': tensors, 'differences': differences, 'input_files': inputs})
    figures = sorted(str(path) for path in design_path.parent.iterdir() if path.suffix in ('.png', '.pdf', '.svg'))
    value = {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'preset': 'point_lattice_512',
             'runner': str(ROOT / 'experiments/run_formation.py'), 'runner_sha256': digest(ROOT / 'experiments/run_formation.py'),
             'check_script': str(Path(__file__).resolve()), 'check_script_sha256': digest(Path(__file__).resolve()),
             'check_execution': {'interpreter': sys.executable, 'torch': torch.__version__,
                                 'map_location': 'cpu', 'weights_only': False, 'tensor_comparison': 'dtype, shape, raw bytes'},
             'seeds': SEEDS, 'arms': arms, 'pairs': pairs, 'comparison_design': str(design_path),
             'comparison_design_sha256': digest(design_path), 'comparison_figures': figures,
             'comparison_figure_count': len(figures),
             'all_pairs_pass': all(p['configs_match_except_declared_fields'] and p['both_completed_100_sweeps']
                                   and p['saved_geometry_records_truth_equal'] for p in pairs),
             'scope': 'Saved-input matching and recorded completion of deterministic finite mean-field CAVI controls; '
                      'not a test of stochastic formation conjectures, population effects, or a formation theorem.',
             'path_note': 'The preset resolves output.directory to runs; these GPU controls are separate from runs/cpu-formation.'}
    (ROOT / 'docs/implementation/source-controls.json').write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'all_pairs_pass': value['all_pairs_pass'], 'pairs': len(pairs),
                      'tensor_counts': [p['tensor_count'] for p in pairs], 'comparison_figures': len(figures)}))


if __name__ == '__main__':
    main()
