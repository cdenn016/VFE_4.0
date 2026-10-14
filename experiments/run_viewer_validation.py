"""Exercise section-inspector controls and orchestration in real Qt windows on CPU."""

from dataclasses import fields, is_dataclass
import hashlib
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent
import numpy as np
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


def tensor_copies(value: object, prefix: str = '') -> dict[str, torch.Tensor]:
    """Capture every tensor of the recognition state without evaluating it."""
    if isinstance(value, torch.Tensor):
        return {prefix: value.detach().clone()}
    if is_dataclass(value):
        return {key: tensor for field in fields(value)
                for key, tensor in tensor_copies(getattr(value, field.name), prefix + '.' + field.name).items()}
    if isinstance(value, (tuple, list)):
        return {key: tensor for index, item in enumerate(value)
                for key, tensor in tensor_copies(item, f'{prefix}[{index}]').items()}
    return {}


def inspect_spatial(dimension: int) -> dict:
    """Dispatch widget callbacks against one captured tiny spatial state."""
    center_grid = (3, 1) if dimension == 2 else (1, 1, 3)
    config = load_config(resolve_preset(f'tiny_spatial_{dimension}d', {
        'manifold': {'grid_shape': (3,) * dimension},
        'population': {'n_agents': 3, 'center_grid': center_grid, 'patch_shape': (1,) * dimension,
                       'coverage': 'allow_uncovered'},
        'bundle': {'K': 2}}))
    model = build_model(config)
    state = initialize_sections(model, config)
    # Distinct deterministic fixture means make both channel/component choices visible.
    for channel, shift in ((state.belief, 0.2), (state.model, 0.8)):
        for bucket in channel.buckets:
            bucket.mu.copy_(torch.arange(bucket.mu.numel(), dtype=bucket.mu.dtype).reshape_as(bucket.mu) / 10 + shift)
    before = tensor_copies(state)
    cpu_rng = torch.get_rng_state().clone()
    viewer = SimulationViewer()
    checks = {}
    try:
        viewer.update(state, model, 12.5)
        click_button(viewer, 0)
        assert viewer.paused
        history = list(viewer.history)
        viewer.agent_input.set_val('1')
        assert viewer.selection_agent == 1
        checks['agent_textbox_callback'] = True
        if dimension == 3:
            center = float(model.incidence.centers[1, 2])
            assert abs(float(viewer.last_section_image.z) - center) < 1e-12
            checks['agent_recenters_z_at_center'] = True
        viewer.channel_buttons.set_active(1)
        assert viewer.selection_channel == 'model'
        checks['channel_radio_callback'] = True
        model_image = viewer.last_section_image.values.copy()
        viewer.channel_buttons.set_active(0)
        assert viewer.selection_channel == 'belief'
        belief_image = viewer.last_section_image.values.copy()
        assert not np.array_equal(model_image.compressed(), belief_image.compressed())
        checks['channel_changes_displayed_section'] = True
        viewer.component_input.set_val('1')
        assert viewer.selection_component == 1
        assert not np.array_equal(belief_image.compressed(), viewer.last_section_image.values.compressed())
        checks['component_textbox_callback'] = True
        viewer.statistic_buttons.set_active(1)
        assert viewer.selection_statistic == 'std'
        checks['uncertainty_radio_callback'] = True
        image = viewer.last_section_image
        mask = np.ma.getmaskarray(image.values)
        assert mask.any() and (~mask).any()
        assert np.isfinite(image.values.compressed()).all() and (image.values.compressed() >= 0).all()
        checks['unsupported_points_masked'] = True
        checks['finite_supported_standard_deviation'] = True
        assert tuple(float(x) for x in image.extent) == (0.0, 1.0, 0.0, 1.0)
        checks['physical_period_extent'] = True
        for widget, value, attribute in ((viewer.agent_input, 'invalid', 'selection_agent'),
                                         (viewer.agent_input, '3', 'selection_agent'),
                                         (viewer.component_input, '1.5', 'selection_component'),
                                         (viewer.component_input, '2', 'selection_component')):
            prior = getattr(viewer, attribute)
            widget.set_val(value)
            assert getattr(viewer, attribute) == prior
            assert viewer.status_text.get_text().strip()
        checks['invalid_numeric_input_preserves_selection_and_reports_status'] = True
        viewer.agent_input.set_val('1')
        viewer.component_input.set_val('1')
        if dimension == 3:
            prior_z = float(viewer.last_section_image.z)
            viewer.slice_slider.set_val(0)
            assert viewer.selection_slice == 0 and float(viewer.last_section_image.z) != prior_z
            checks['slice_slider_callback'] = True
            viewer.agent_input.set_val('2')
            center = float(model.incidence.centers[2, 2])
            assert abs(float(viewer.last_section_image.z) - center) < 1e-12
            assert viewer.last_section_image.values.count() > 0
            checks['new_agent_restores_supported_center_slice'] = True
        else:
            assert viewer.slice_slider is None
            checks['no_slice_slider_in_2d'] = True
        assert viewer.history == history and state.sweep == 0 and viewer.paused
        checks['controls_do_not_advance_or_duplicate_history'] = True
        after = tensor_copies(state)
        assert before.keys() == after.keys() and all(torch.equal(value, after[key]) for key, value in before.items())
        assert torch.equal(cpu_rng, torch.get_rng_state())
        checks['recognition_and_rng_unchanged_by_controls'] = True
        snapshot = ROOT / f'docs/implementation/section-viewer-{dimension}d.png'
        viewer.figure.canvas.draw()
        viewer.figure.savefig(snapshot, dpi=160, bbox_inches='tight')
        displayed = viewer.last_section_image
        return {'dimension': dimension, 'n_agents': 3, 'K': 2, 'grid_shape': [3] * dimension,
                'patch_shape': [1] * dimension, 'checks': checks, 'snapshot': str(snapshot),
                'selection': {'agent': viewer.selection_agent, 'channel': viewer.selection_channel,
                              'component': viewer.selection_component, 'statistic': viewer.selection_statistic,
                              'slice_index': viewer.selection_slice, 'z': displayed.z},
                'image_shape': list(displayed.values.shape),
                'supported_preview_points': int(displayed.values.count()),
                'masked_preview_points': int(np.ma.getmaskarray(displayed.values).sum()),
                'physical_extent': list(displayed.extent)}
    finally:
        plt.close(viewer.figure)


def run_validation() -> dict:
    torch.set_num_threads(1)
    historical_paths = [ROOT / 'docs/implementation/viewer-validation.json',
                        ROOT / 'docs/implementation/viewer-snapshot.png']
    historical = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in historical_paths if path.is_file()}
    config = load_config(resolve_preset('tiny_point', {
        'population': {'n_agents': 2, 'lattice_shape': (2,)}, 'bundle': {'K': 2}}))
    model = build_model(config)
    state = initialize_sections(model, config)
    viewer = SimulationViewer()
    checks = {}
    try:
        viewer.update(state, model, 0.0)
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
            'population': {'n_agents': 2, 'lattice_shape': (2,)}, 'bundle': {'K': 2},
            'inference': {'sweeps': 5}, 'output': {'viewer': True, 'snapshot_every': 1,
                'figures': False, 'directory': 'runs/section-viewer-validation'}}))
        result = run_simulation(run_config)
        summary = json.loads((result.run_directory / 'summary.json').read_text())
        assert summary['status'] == 'stopped' and result.completed_sweeps == 1
        assert (result.run_directory / 'final_state.pt').is_file()
        checks['orchestrator_stop_checkpoint'] = True
    finally:
        viewer_module.SimulationViewer = original
        plt.close('all')
    spatial = [inspect_spatial(dimension) for dimension in (2, 3)]
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected for path, expected in historical.items())
    checks['historical_artifacts_preserved'] = True
    output = {'status': 'completed', 'backend': matplotlib.get_backend(), 'device': 'cpu',
              'interpreter': sys.executable, 'torch': torch.__version__, 'torch_threads': torch.get_num_threads(),
              'input_method': {'orchestration_buttons': 'Matplotlib MouseEvent press/release in real Qt windows',
                               'numeric_inputs': 'Matplotlib TextBox.set_val dispatching registered callbacks',
                               'channel_and_statistic': 'RadioButtons.set_active dispatching registered callbacks',
                               'slice': 'Slider.set_val dispatching the registered callback',
                               'resume': 'Qt-backed Matplotlib timer invokes the pause/resume mouse events',
                               'manual_interaction': False},
              'checks': checks, 'spatial_windows': spatial, 'run_directory': str(result.run_directory),
              'stop_summary': summary, 'historical_artifact_sha256': historical}
    write_json(ROOT / 'docs/implementation/section-viewer-validation.json', output)
    return output


if __name__ == '__main__':
    print(run_validation())
