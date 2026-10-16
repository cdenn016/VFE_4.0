"""Exercise saved-checkpoint inspection in real Qt windows on tiny CPU fixtures."""

from contextlib import ExitStack
from copy import deepcopy
from dataclasses import fields, is_dataclass
import hashlib
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent
import numpy as np
import torch

from vfe4.config import SimulationConfig, load_config
from vfe4.inspection import load_inspection_checkpoint, open_saved_run
from vfe4.io import write_json
from vfe4.model import build_model, initialize_sections
from vfe4.presets import resolve_preset
from vfe4.recorder import RecorderState
from vfe4.viewer import SimulationViewer


def tensor_copies(value: object, prefix: str = '') -> dict[str, torch.Tensor]:
    """Capture tensor contents without evaluating inference or using CUDA."""
    if isinstance(value, torch.Tensor):
        return {prefix: value.detach().clone()}
    if is_dataclass(value):
        return {key: tensor for field in fields(value)
                for key, tensor in tensor_copies(getattr(value, field.name), prefix + '.' + field.name).items()}
    if isinstance(value, (tuple, list)):
        return {key: tensor for index, item in enumerate(value)
                for key, tensor in tensor_copies(item, f'{prefix}[{index}]').items()}
    return {}


def assert_tensors_equal(before: dict[str, torch.Tensor], value: object) -> None:
    """Require exact tensor values and CPU placement after loading or callbacks."""
    after = tensor_copies(value)
    assert before.keys() == after.keys()
    assert all(tensor.device.type == 'cpu' and torch.equal(tensor, before[key])
               for key, tensor in after.items())


def click_button(viewer: SimulationViewer, index: int) -> None:
    """Dispatch registered mouse callbacks within a displayed Qt button."""
    viewer.figure.canvas.draw()
    bounds = viewer.buttons[index].ax.bbox
    x, y = (bounds.x0 + bounds.x1) / 2, (bounds.y0 + bounds.y1) / 2
    events = [MouseEvent(name, viewer.figure.canvas, x, y, button=1)
              for name in ('button_press_event', 'button_release_event')]
    for event in events:
        viewer.figure.canvas.callbacks.process(event.name, event)


def inspect_spatial(dimension: int, directory: Path) -> dict:
    """Load a saved snapshot with CUDA metadata while every tensor is on CPU."""
    center_grid = (3, 1) if dimension == 2 else (1, 1, 3)
    config = load_config(resolve_preset(f'tiny_spatial_{dimension}d', {
        'runtime': {'device': 'cpu', 'dtype': 'float64'},
        'manifold': {'grid_shape': (3,) * dimension},
        'population': {'n_agents': 3, 'center_grid': center_grid,
                       'patch_shape': (1,) * dimension, 'coverage': 'allow_uncovered'},
        'bundle': {'K': 2}}))
    model = build_model(config)
    state = initialize_sections(model, config)
    state.sweep = 7
    for law, shift in ((state.belief, 0.2), (state.model, 1.2)):
        for bucket in law.buckets:
            bucket.mu.copy_(torch.arange(bucket.mu.numel(), dtype=config.dtype).reshape_as(bucket.mu) / 10 + shift)
    recorder = RecorderState([{'sweep': 1, 'vfe_total': 18.25}, {'sweep': 4, 'vfe_total': 7.5}])
    expected_history = [(row['sweep'], row['vfe_total']) for row in recorder.rows]
    expected_model, expected_state = tensor_copies(model), tensor_copies(state)
    saved_raw = deepcopy(config.raw)
    saved_raw['runtime']['device'] = 'cuda'
    model.config = SimulationConfig(saved_raw, torch.device('cuda'), config.dtype)
    model.domain.manifold.device = torch.device('cuda')
    checkpoint = directory / f'fixture-{dimension}d' / 'final_state.pt'
    checkpoint.parent.mkdir()
    saved_rng = torch.Generator(device='cpu').manual_seed(16 + dimension).get_state()
    # Metadata and RNG fields mimic a CUDA checkpoint; the fixture uses only CPU tensors.
    torch.save({'model': model, 'state': state, 'recorder': recorder,
                'cpu_rng': saved_rng, 'cuda_rng': saved_rng.clone()}, checkpoint)
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    files_before = sorted(str(path.relative_to(directory)) for path in directory.rglob('*'))
    live_rng = torch.get_rng_state().clone()
    live_determinism = (torch.are_deterministic_algorithms_enabled(),
                        torch.is_deterministic_algorithms_warn_only_enabled())
    assert not torch.equal(saved_rng, live_rng)
    checks, viewer = {}, None
    cuda_names = ('is_available', 'device_count', 'current_device', 'get_device_properties',
                  'get_device_name', 'init', '_lazy_init', 'get_rng_state', 'get_rng_state_all',
                  'set_rng_state', 'set_rng_state_all')
    try:
        with ExitStack() as stack:
            cuda_mocks = {name: stack.enter_context(patch.object(torch.cuda, name,
                side_effect=AssertionError(f'CPU inspection called CUDA {name}'))) for name in cuda_names}
            restore_rng = stack.enter_context(patch.object(torch, 'set_rng_state',
                side_effect=AssertionError('Inspection restored the live CPU RNG')))
            set_determinism = stack.enter_context(patch.object(torch, 'use_deterministic_algorithms',
                side_effect=AssertionError('Inspection changed global deterministic settings')))
            loaded_model, loaded_state, loaded_recorder = load_inspection_checkpoint(checkpoint, device='cpu')
            assert loaded_model.config.device.type == 'cpu'
            assert loaded_model.config['runtime']['device'] == 'cpu'
            assert torch.device(loaded_model.domain.manifold.device).type == 'cpu'
            assert loaded_model.config.dtype == torch.float64
            assert loaded_recorder.rows == recorder.rows and loaded_state.sweep == 7
            assert_tensors_equal(expected_model, loaded_model)
            assert_tensors_equal(expected_state, loaded_state)
            assert torch.equal(live_rng, torch.get_rng_state())
            checks['cpu_loader_maps_saved_cuda_metadata_without_cuda_or_rng_restoration'] = True
            checks['checkpoint_coefficients_fixed_data_dtype_and_recorder_retained'] = True
            # A directory exercises the public run-folder path selection as well.
            viewer = open_saved_run(checkpoint.parent, device='cpu', block=False)
            assert 'qt' in matplotlib.get_backend().lower() and plt.fignum_exists(viewer.figure.number)
            assert viewer.inspection_only and viewer.paused and not viewer.once
            assert viewer.history == expected_history and viewer.snapshot.sweep == 7
            assert viewer.history[-1][0] < viewer.snapshot.sweep
            np.testing.assert_array_equal(viewer.axes[0].lines[0].get_xdata(), [1, 4])
            np.testing.assert_array_equal(viewer.axes[0].lines[0].get_ydata(), [18.25, 7.5])
            checks['real_qt_window_uses_recorded_history_without_synthetic_endpoint'] = True
            snapshot_before = tensor_copies(viewer.snapshot)
            model_before = tensor_copies(viewer._model)
            axes_count = len(viewer.figure.axes)
            assert not viewer.buttons[0].get_active() and not viewer.buttons[1].get_active()
            assert viewer.buttons[2].get_active() and viewer.buttons[2].label.get_text() == 'Close'
            click_button(viewer, 0)
            click_button(viewer, 1)
            assert viewer.paused and not viewer.once and not viewer.stopped
            assert not viewer.ready()
            checks['inference_buttons_disabled_and_mouse_callbacks_inactive'] = True
            viewer.agent_input.set_val('1')
            assert viewer.selection_agent == 1
            if dimension == 3:
                assert viewer.selection_slice == 2
                assert abs(viewer.last_reference_image.z - 1 / 3) < 1e-12
            checks['agent_selector_updates_stored_section'] = True
            belief_reference = viewer.last_reference_image.mean.copy()
            viewer.channel_buttons.set_active(1)
            assert viewer.selection_channel == 'model'
            assert not np.array_equal(belief_reference.compressed(), viewer.last_reference_image.mean.compressed())
            checks['channel_radio_updates_displayed_reference'] = True
            component_zero = viewer.last_reference_image.mean.copy()
            viewer.component_input.set_val('1')
            assert viewer.selection_component == 1
            assert not np.array_equal(component_zero.compressed(), viewer.last_reference_image.mean.compressed())
            checks['component_textbox_updates_displayed_reference'] = True
            viewer.statistic_buttons.set_active(1)
            assert viewer.selection_statistic == 'std'
            assert np.isfinite(viewer.last_section_image.values.compressed()).all()
            assert (viewer.last_section_image.values.compressed() >= 0).all()
            checks['uncertainty_radio_remains_active'] = True
            context = 3 if dimension == 2 else 1
            viewer.context_input.set_val(str(context))
            assert viewer.selection_context == context
            if dimension == 3:
                assert viewer.selection_slice == 2
                viewer.slice_slider.set_val(3)
                assert viewer.selection_slice == 3 and viewer.last_reference_image.z == 0.5
                assert viewer.last_section_image.z == viewer.last_reference_image.z
                viewer.context_input.set_val(str(context))
                assert viewer.selection_slice == 2
                checks['slice_and_context_callbacks_keep_section_and_reference_synchronized'] = True
            else:
                assert viewer.slice_slider is None
            checks['explicit_context_selector_remains_active'] = True
            viewer.statistic_buttons.set_active(0)
            assert viewer.history == expected_history and viewer.snapshot.sweep == 7
            assert_tensors_equal(snapshot_before, viewer.snapshot)
            assert_tensors_equal(model_before, viewer._model)
            assert_tensors_equal(expected_state, loaded_state)
            assert loaded_recorder.rows == recorder.rows
            assert torch.equal(live_rng, torch.get_rng_state())
            assert len(viewer.figure.axes) == axes_count
            checks['selectors_preserve_snapshot_state_rng_history_and_axes'] = True
            screenshot = ROOT / f'docs/implementation/inspector-viewer-{dimension}d.png'
            viewer.figure.canvas.draw()
            context_bounds = viewer.context_range.get_window_extent(viewer.figure.canvas.get_renderer())
            assert all(not context_bounds.overlaps(button.ax.bbox) for button in viewer.buttons)
            checks['context_range_clear_of_buttons'] = True
            viewer.figure.savefig(screenshot, dpi=160, bbox_inches='tight')
            click_button(viewer, 2)
            assert viewer.stopped and not plt.fignum_exists(viewer.figure.number)
            checks['close_mouse_callback_exits_real_qt_window'] = True
            assert restore_rng.call_count == 0 and all(mock.call_count == 0 for mock in cuda_mocks.values())
            assert set_determinism.call_count == 0
            assert (torch.are_deterministic_algorithms_enabled(),
                    torch.is_deterministic_algorithms_warn_only_enabled()) == live_determinism
            checks['inspection_preserves_global_deterministic_settings_without_setting_them'] = True
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checkpoint_sha
        assert sorted(str(path.relative_to(directory)) for path in directory.rglob('*')) == files_before
        assert model.config['runtime']['device'] == 'cuda'
        assert torch.equal(live_rng, torch.get_rng_state())
        checks['inspection_writes_no_run_files_or_checkpoint_and_leaves_rng_unchanged'] = True
        return {'dimension': dimension, 'n_agents': 3, 'K': 2, 'grid_shape': [3] * dimension,
                'patch_shape': [1] * dimension, 'snapshot_sweep': 7,
                'recorded_history': expected_history, 'saved_device_metadata': 'cuda',
                'actual_fixture_tensor_device': 'cpu', 'display_device': 'cpu',
                'checks': checks, 'snapshot': str(screenshot), 'plot_axis_count': axes_count,
                'cuda_functions_blocked': list(cuda_names), 'cuda_call_count': 0,
                'cpu_rng_restore_call_count': 0, 'deterministic_setting_call_count': 0,
                'checkpoint_sha256': checkpoint_sha,
                'selection': {'agent': viewer.selection_agent, 'channel': viewer.selection_channel,
                              'component': viewer.selection_component, 'context': viewer.selection_context,
                              'slice_index': viewer.selection_slice, 'z': viewer.last_reference_image.z}}
    finally:
        if viewer is not None:
            plt.close(viewer.figure)


def run_validation() -> dict:
    """Run only saved-run inspection checks and retain all previous viewer evidence."""
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    prior_paths = [path for path in (ROOT / 'experiments').glob('run_*validation.py')
                   if path.name != Path(__file__).name]
    prior_paths += [path for path in (ROOT / 'docs/implementation').glob('*viewer*')
                    if path.is_file() and not path.name.startswith('inspector-')]
    historical = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in prior_paths}
    with TemporaryDirectory(prefix='vfe4-inspector-qt-') as temporary:
        windows = [inspect_spatial(dimension, Path(temporary)) for dimension in (2, 3)]
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == value for path, value in historical.items())
    result = {'status': 'completed', 'backend': matplotlib.get_backend(), 'device': 'cpu',
              'dtype': 'float64', 'interpreter': sys.executable, 'torch': torch.__version__,
              'torch_threads': torch.get_num_threads(), 'torch_interop_threads': torch.get_num_interop_threads(),
              'thread_environment': {name: os.environ[name] for name in
                  ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
              'gpu_work_or_query': False,
              'input_method': {'buttons': 'Matplotlib MouseEvent press/release in real Qt windows',
                               'numeric_inputs': 'TextBox.set_val dispatches registered submit callbacks',
                               'channel_and_statistic': 'RadioButtons.set_active dispatches registered callbacks',
                               'slice': 'Slider.set_val dispatches the registered changed callback',
                               'manual_interaction': False},
              'checks': {'historical_scripts_and_evidence_preserved': True},
              'spatial_windows': windows, 'historical_artifact_sha256': historical,
              'scope': 'Tiny CPU Qt inspection of CPU tensors with simulated CUDA metadata; no inference, CUDA run, or manual human interaction.'}
    write_json(ROOT / 'docs/implementation/inspector-viewer-validation.json', result)
    return result


if __name__ == '__main__':
    print(run_validation())
