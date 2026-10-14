"""Edit PRESET/OVERRIDES, then press Run. No command-line arguments."""

from vfe4.config import load_config
from vfe4.presets import resolve_preset
from vfe4.simulation import run_simulation


PRESET = 'point_lattice_512'
OVERRIDES = {'bundle': {'K': 4}, 'inference': {'sweeps': 100}}
CONFIG = resolve_preset(PRESET, OVERRIDES)


if __name__ == '__main__':
    result = run_simulation(load_config(CONFIG))
    print(f'Finished at sweep {result.completed_sweeps}; results: {result.run_directory}')
