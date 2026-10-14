"""Exercise the real Qt event loop with application-level button events on CPU."""

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent
import torch

from vfe4.config import load_config
from vfe4.inference import make_schedule, step
from vfe4.io import write_json
from vfe4.model import SimulationModel, build_model, initialize_sections
from vfe4.presets import resolve_preset
from vfe4.simulation import run_simulation
from vfe4.state import SectionState
from vfe4.viewer import SimulationViewer


def click_button(viewer: SimulationViewer, index: int) -> None:
    """Dispatch Matplotlib press/release events inside the displayed button axes."""
    viewer.figure.canvas.draw()
    bounds = viewer.buttons[index].ax.bbox
    x, y = (bounds.x0 + bounds.x1) / 2, (bounds.y0 + bounds.y1) / 2
    for name in ('button_press_event', 'button_release_event'):
        event = MouseEvent(name, viewer.figure.canvas, x, y, button=1)
        viewer.figure.canvas.callbacks.process(name, event)


def run_validation() -> dict:
    torch.set_num_threads(1)
    config = load_config(resolve_preset('tiny_point', {
        'population': {'n_agents': 2, 'lattice_shape': (2,)}, 'bundle': {'K': 1}}))
    model = build_model(config)
    state = initialize_sections(model, config)
    viewer = SimulationViewer()
    checks = {}
    try:
        click_button(viewer, 0)
        assert viewer.paused
        checks['pause_button'] = True
        click_button(viewer, 1)
        assert viewer.paused and viewer.once and viewer.ready() and not viewer.once
        state, report = step(state, model, make_schedule(model, config), config)
        viewer.update(state, model, report.objective_after)
        assert state.sweep == 1 and viewer.paused
        checks['one_sweep_button'] = True
        timer = viewer.figure.canvas.new_timer(interval=100)
        timer.single_shot = True

        def resume_event() -> None:
            click_button(viewer, 0)

        timer.add_callback(resume_event)
        timer.start()
        assert viewer.ready() and not viewer.paused
        checks['resume_while_waiting'] = True
        snapshot = ROOT / 'docs/implementation/viewer-snapshot.png'
        viewer.figure.savefig(snapshot, dpi=160, bbox_inches='tight')
        click_button(viewer, 2)
        assert viewer.stopped and not viewer.ready()
        checks['stop_button'] = True
    finally:
        plt.close(viewer.figure)

    # The ordinary orchestrator must save a stopped summary and resumable state.
    from vfe4 import viewer as viewer_module
    original = viewer_module.SimulationViewer

    class StopAfterSnapshot(SimulationViewer):
        def update(self, state: SectionState, model: SimulationModel, objective: float) -> None:
            super().update(state, model, objective)
            click_button(self, 2)

    viewer_module.SimulationViewer = StopAfterSnapshot
    try:
        run_config = load_config(resolve_preset('tiny_point', {
            'population': {'n_agents': 2, 'lattice_shape': (2,)}, 'bundle': {'K': 1},
            'inference': {'sweeps': 5}, 'output': {'viewer': True, 'snapshot_every': 1,
                'figures': False, 'directory': 'runs/viewer-validation'}}))
        result = run_simulation(run_config)
        summary = json.loads((result.run_directory / 'summary.json').read_text())
        assert summary['status'] == 'stopped' and result.completed_sweeps == 1
        assert (result.run_directory / 'final_state.pt').is_file()
        checks['orchestrator_stop_checkpoint'] = True
    finally:
        viewer_module.SimulationViewer = original
        plt.close('all')
    output = {'status': 'completed', 'backend': matplotlib.get_backend(), 'device': 'cpu',
              'input_method': 'synthetic Matplotlib mouse events in a real Qt window; no manual interaction',
              'checks': checks, 'snapshot': str(snapshot), 'run_directory': str(result.run_directory)}
    write_json(ROOT / 'docs/implementation/viewer-validation.json', output)
    return output


if __name__ == '__main__':
    print(run_validation())
