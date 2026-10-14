"""Edit dictionaries and press Run; seeds run sequentially on one selected device."""

from vfe4.batch import run_batch
from vfe4.presets import resolve_preset


PRESET = 'point_lattice_512'
OVERRIDES = {'output': {'viewer': False}}
SEEDS = [20261013, 20261014, 20261015, 20261016, 20261017]
GRID = {}


if __name__ == '__main__':
    print(run_batch(resolve_preset(PRESET, OVERRIDES), SEEDS, GRID))
