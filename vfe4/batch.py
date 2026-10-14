from copy import deepcopy
from itertools import product
from pathlib import Path
from uuid import uuid4

from .config import load_config
from .simulation import run_simulation


def run_batch(base: dict, seeds: list[int], grid: dict[str, list]) -> list[Path]:
    root = Path(base['output']['directory'])
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[1] / root
    batch_root = root / ('batch-' + uuid4().hex[:8])
    paths = []
    for combination in product(*grid.values()):
        for seed in seeds:
            raw = deepcopy(base)
            raw['runtime']['seed'] = seed
            raw['output']['directory'] = str(batch_root)
            for dotted, value in zip(grid, combination):
                section = raw
                parts = dotted.split('.')
                for part in parts[:-1]:
                    section = section[part]
                if parts[-1] not in section:
                    raise KeyError(dotted)
                section[parts[-1]] = value
            paths.append(run_simulation(load_config(raw)).run_directory)
    return paths
