"""Open trusted saved checkpoints for display without continuing inference."""

from copy import deepcopy
from pathlib import Path

import torch

from .config import SimulationConfig
from .model import SimulationModel
from .recorder import RecorderState
from .state import SectionState
from .viewer import SimulationViewer


def load_inspection_checkpoint(
    path:   Path,
    device: str = 'cpu',
) -> tuple[SimulationModel, SectionState, RecorderState]:
    """Map checkpoint tensors to the display device without restoring saved RNG."""
    display_device = torch.device(device)
    if display_device.type not in ('cpu', 'cuda'):
        raise ValueError('Display device must be cpu or cuda')
    path = Path(path).expanduser()
    if path.is_dir():
        path = path / 'final_state.pt'
    data = torch.load(path, map_location=display_device, weights_only=False)
    model = data['model']
    raw = deepcopy(model.config.raw)
    raw['runtime']['device'] = str(display_device)
    model.config = SimulationConfig(raw, display_device, model.config.dtype)
    model.domain.manifold.device = display_device
    return model, data['state'], data['recorder']


def open_saved_run(
    path:   Path,
    device: str  = 'cpu',
    block:  bool = True,
) -> SimulationViewer:
    """Display one stored state and its recorded history, with no inference step."""
    model, state, recorder = load_inspection_checkpoint(path, device)
    history = [(int(row['sweep']), float(row['vfe_total'])) for row in recorder.rows]
    viewer = SimulationViewer(inspection_only=True)
    viewer.update(state, model, None, recorded_history=history)
    viewer.figure.suptitle(f'Saved checkpoint: {Path(path).name}; sweep {state.sweep}', fontsize=13)
    viewer.figure.canvas.draw_idle()
    viewer.plt.show(block=block)
    return viewer
