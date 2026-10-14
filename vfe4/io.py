"""Self-contained run folders and continuation of the same fixed-data model."""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from uuid import uuid4

import torch

from .config import SimulationConfig
from .model import SimulationModel
from .recorder import RecorderState
from .state import SectionState


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


class RunWriter:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        run_dir.mkdir(parents=True, exist_ok=False)

    @classmethod
    def create(cls, config: SimulationConfig) -> 'RunWriter':
        root = Path(config['output']['directory'])
        if not root.is_absolute():
            root = Path(__file__).resolve().parents[1] / root
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        name = f'{stamp}-{config["manifold"]["kind"]}-{uuid4().hex[:8]}'
        writer = cls(root / name)
        write_json(writer.run_dir / 'config.json', config.raw)
        write_json(writer.run_dir / 'environment.json', {'interpreter': sys.executable,
            'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
            'selected_device': str(config.device), 'dtype': str(config.dtype),
            'cuda_available': torch.cuda.is_available(),
            'device_name': torch.cuda.get_device_name(config.device) if config.device.type == 'cuda' else 'CPU',
            'recognition': 'product across agents and channels; covariance ' + config['recognition']['covariance_structure'],
            'model': 'normalized Gaussian roots, bridges, private/spatial/source-label records'})
        return writer

    def save_model(self, model: SimulationModel) -> None:
        torch.save({'domain': model.domain, 'incidence': model.incidence, 'rows': model.rows,
                    'copies': model.copies, 'loops': model.loops, 'frame_field': model.frame_field}, self.run_dir / 'geometry.pt')
        torch.save({'records': model.records, 'truth_fixture': model.truth}, self.run_dir / 'observations.pt')

    def write_metrics(self, state: RecorderState) -> None:
        names = list(dict.fromkeys(key for row in state.rows for key in row))
        with (self.run_dir / 'metrics.csv').open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, names)
            writer.writeheader()
            writer.writerows(state.rows)

    def summary(self, **values: object) -> None:
        write_json(self.run_dir / 'summary.json', values)


def save_checkpoint(path: Path, model: SimulationModel, state: SectionState,
                    recorder: RecorderState) -> None:
    data = {'model': model, 'state': state, 'recorder': recorder, 'cpu_rng': torch.get_rng_state()}
    if model.config.device.type == 'cuda':
        data['cuda_rng'] = torch.cuda.get_rng_state(model.config.device)
    temporary = path.with_suffix('.tmp')
    torch.save(data, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path) -> tuple[SimulationModel, SectionState, RecorderState]:
    data = torch.load(path, weights_only=False)
    torch.set_rng_state(data['cpu_rng'].cpu())
    if 'cuda_rng' in data:
        torch.cuda.set_rng_state(data['cuda_rng'].cpu(), data['model'].config.device)
    return data['model'], data['state'], data['recorder']
