"""Validate the spatial dashboard with real Qt callbacks and tiny CPU fixtures."""

from dataclasses import fields, is_dataclass
import hashlib
from pathlib import Path
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
from vfe4.io import write_json
from vfe4.model import SimulationModel, build_model, initialize_sections
from vfe4.presets import resolve_preset
from vfe4.viewer import SimulationViewer


def tensor_copies(value: object, prefix: str = '') -> dict[str, torch.Tensor]:
    """Copy recognition/snapshot tensors for exact before-and-after comparisons."""
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
    """Require exact preservation of every previously captured tensor."""
    after = tensor_copies(value)
    assert before.keys() == after.keys()
    assert all(torch.equal(tensor, after[key]) for key, tensor in before.items())


def click_button(viewer: SimulationViewer, index: int) -> None:
    """Dispatch press/release events inside an actual displayed Qt button."""
    viewer.figure.canvas.draw()
    bounds = viewer.buttons[index].ax.bbox
    x, y = (bounds.x0 + bounds.x1) / 2, (bounds.y0 + bounds.y1) / 2
    for name in ('button_press_event', 'button_release_event'):
        event = MouseEvent(name, viewer.figure.canvas, x, y, button=1)
        viewer.figure.canvas.callbacks.process(name, event)


def click_population(viewer: SimulationViewer, agent: int) -> dict:
    """Click near a physical center through Matplotlib's real event registry."""
    viewer.figure.canvas.draw()
    position = viewer.population_positions[agent, :2].copy()
    # Keep centers on the periodic boundary inside the plotting rectangle.
    position += 0.005
    x, y = viewer.population_ax.transData.transform(position)
    for name in ('button_press_event', 'button_release_event'):
        event = MouseEvent(name, viewer.figure.canvas, x, y, button=1)
        assert event.inaxes is viewer.population_ax
        viewer.figure.canvas.callbacks.process(name, event)
    assert viewer.selection_agent == agent
    assert int(viewer.agent_input.text) == agent
    return {'target_agent': agent, 'physical_xy': position.tolist(),
            'selected_agent': viewer.selection_agent}


def assert_masked_equal(actual: np.ma.MaskedArray, expected: np.ma.MaskedArray) -> None:
    """Compare displayed supported values and every mask location."""
    np.testing.assert_array_equal(np.ma.getmaskarray(actual), np.ma.getmaskarray(expected))
    np.testing.assert_allclose(actual.compressed(), expected.compressed(), rtol=1e-12, atol=1e-12)


def expected_reference(viewer: SimulationViewer, model: SimulationModel) -> tuple:
    r"""Independently compute mean_c = sum_{s:c_s=c} mu_s / count_c."""
    context_ids = model.incidence.context_id.numpy()
    n_contexts, K = model.domain.contexts.shape[0], model.config['bundle']['K']
    coefficients = np.empty((len(context_ids), K))
    for bucket in getattr(viewer.snapshot, viewer.selection_channel).buckets:
        coefficients[bucket.incidence_ids.numpy().ravel()] = bucket.mu.numpy().reshape(-1, K)
    counts = np.bincount(context_ids, minlength=n_contexts)
    sums, truth_sums = np.zeros(n_contexts), np.zeros(n_contexts)
    np.add.at(sums, context_ids, coefficients[:, viewer.selection_component])
    np.add.at(truth_sums, context_ids, model.truth[:, viewer.selection_component].numpy())
    shape = model.domain.grid_shape
    means = np.ma.array((sums / np.maximum(counts, 1)).reshape(shape), mask=(counts == 0).reshape(shape))
    truth = np.ma.array((truth_sums / np.maximum(counts, 1)).reshape(shape), mask=(counts == 0).reshape(shape))
    if len(shape) == 3:
        lower, remainder = divmod(viewer.selection_slice, 2)
        upper = (lower + 1) % shape[2]
        if remainder:
            means = (means[:, :, lower] + means[:, :, upper]) / 2
            truth = (truth[:, :, lower] + truth[:, :, upper]) / 2
        else:
            means, truth = means[:, :, lower], truth[:, :, lower]
    return means, truth, means - truth


def assert_reference(viewer: SimulationViewer, model: SimulationModel) -> None:
    """Check helper outputs and the arrays actually drawn in all three panels."""
    reference = viewer.last_reference_image
    expected = expected_reference(viewer, model)
    for name, values in zip(('mean', 'truth', 'error'), expected):
        assert_masked_equal(getattr(reference, name), values)
    for values, ax in zip((expected[1], expected[0], expected[2]), viewer.reference_axes):
        assert len(ax.images) == 1
        assert_masked_equal(ax.images[0].get_array(), values.T)
    assert tuple(reference.extent) == (0.0, 1.0, 0.0, 1.0)
    assert reference.z == viewer.last_section_image.z
    if len(model.domain.grid_shape) == 3:
        assert reference.z == viewer.selection_slice / (2 * model.domain.grid_shape[2])


def assert_routing(viewer: SimulationViewer, model: SimulationModel) -> dict:
    """Check the selected channel/context row directly against candidate tensors."""
    incidence = ((model.incidence.agent_id == viewer.selection_agent) &
                 (model.incidence.context_id == viewer.selection_context)).nonzero().flatten()
    assert incidence.numel() == 1
    channel = 0 if viewer.selection_channel == 'belief' else 1
    candidates = ((model.rows.receiver_incidence == incidence[0]) &
                  (model.channel_edge == channel)).nonzero().flatten()
    sources = model.incidence.agent_id[model.rows.source_incidence[candidates]].numpy()
    probabilities = viewer.snapshot.beta[candidates].numpy()
    np.testing.assert_array_equal(viewer.last_routing.source_agents, sources)
    np.testing.assert_array_equal(viewer.last_routing.probabilities, probabilities)
    assert len(sources) > 1 and np.count_nonzero(sources == viewer.selection_agent) == 1
    assert abs(float(probabilities.sum()) - 1.0) < 1e-12
    assert 'self' in viewer.population_ax.get_title().lower()
    return {'agent': viewer.selection_agent, 'channel': viewer.selection_channel,
            'context': viewer.selection_context, 'status': viewer.last_routing.status,
            'candidate_ids': candidates.tolist(), 'source_agents': sources.tolist(),
            'probabilities': probabilities.tolist()}


def assert_nearest_context(viewer: SimulationViewer, model: SimulationModel) -> None:
    """Require a nearest torus node, accepting either minimizer in an exact tie."""
    point = model.incidence.centers[viewer.selection_agent].clone()
    if len(model.domain.grid_shape) == 3:
        point[2] = viewer.selection_slice / (2 * model.domain.grid_shape[2])
    delta = model.domain.manifold.displacement(point, model.domain.contexts)
    distances = delta.square().sum(-1)
    assert abs(float(distances[viewer.selection_context] - distances.min())) < 1e-12
    assert int(viewer.context_input.text) == viewer.selection_context


def inspect_spatial(dimension: int) -> dict:
    """Exercise one real Qt dashboard using overlapping, partially covered patches."""
    centers = (2, 1) if dimension == 2 else (2, 1, 2)
    n_agents = int(np.prod(centers))
    config = load_config(resolve_preset(f'tiny_spatial_{dimension}d', {
        'runtime': {'device': 'cpu', 'dtype': 'float64'},
        'manifold': {'grid_shape': (4,) * dimension},
        'population': {'n_agents': n_agents, 'center_grid': centers,
                       'patch_shape': (3,) * dimension, 'coverage': 'allow_uncovered'},
        'bundle': {'K': 2}}))
    model = build_model(config)
    state = initialize_sections(model, config)
    for channel, shift in ((state.belief, 0.2), (state.model, 1.2)):
        for bucket in channel.buckets:
            bucket.mu.copy_(torch.arange(bucket.mu.numel(), dtype=bucket.mu.dtype).reshape_as(bucket.mu) / 50 + shift)
    # Make every nontrivial row distinct across context and channel.
    for row in range(model.rows.row_offset.numel() - 1):
        start, stop = model.rows.row_offset[row:row + 2].tolist()
        scores = torch.arange(stop - start, dtype=state.beta.dtype) * (row + 1) / 31
        state.beta[start:stop] = scores.softmax(0)
    before = tensor_copies(state)
    cpu_rng = torch.get_rng_state().clone()
    viewer = SimulationViewer()
    checks, clicks, displayed_rows = {}, [], []
    try:
        viewer.update(state, model, 12.5)
        assert 'qt' in matplotlib.get_backend().lower()
        assert len(viewer.reference_axes) == 3
        assert viewer.snapshot.beta.device.type == 'cpu' and not viewer.snapshot.beta.requires_grad
        assert viewer.snapshot.beta.data_ptr() != state.beta.data_ptr()
        snapshot_before = tensor_copies(viewer.snapshot)
        axis_count, history = len(viewer.figure.axes), list(viewer.history)
        click_button(viewer, 0)
        assert viewer.paused
        click_button(viewer, 1)
        assert viewer.paused and viewer.once and viewer.ready() and not viewer.once
        checks['pause_and_single_sweep_request_mouse_callbacks'] = True
        clicks.append(click_population(viewer, 1 if dimension == 2 else 2))
        assert_nearest_context(viewer, model)
        checks['population_mouse_selects_agent_and_recenters_context'] = True
        if dimension == 3:
            viewer.slice_slider.set_val(4)
            assert_nearest_context(viewer, model)
            clicks.append(click_population(viewer, 3))
            assert viewer.selection_slice == 4
            checks['population_hit_testing_uses_current_z'] = True
        # x=1/4, y=0 (and z=1/4) is covered by every selected receiver.
        context = 4 if dimension == 2 else 17
        viewer.context_input.set_val(str(context))
        assert viewer.selection_context == context
        if dimension == 3:
            assert viewer.selection_slice == 2 and viewer.last_reference_image.z == 0.25
        displayed_rows.append(assert_routing(viewer, model))
        assert_reference(viewer, model)
        checks['explicit_context_moves_slice_and_displays_exact_row'] = True
        belief_image = viewer.last_reference_image.mean.copy()
        viewer.channel_buttons.set_active(1)
        displayed_rows.append(assert_routing(viewer, model))
        assert_reference(viewer, model)
        assert not np.array_equal(belief_image.compressed(), viewer.last_reference_image.mean.compressed())
        checks['channel_callback_changes_reference_and_exact_source_row'] = True
        component_zero = viewer.last_reference_image.mean.copy()
        viewer.component_input.set_val('1')
        assert viewer.selection_component == 1
        assert_reference(viewer, model)
        assert not np.array_equal(component_zero.compressed(), viewer.last_reference_image.mean.compressed())
        checks['component_callback_changes_displayed_reference'] = True
        viewer.statistic_buttons.set_active(1)
        assert viewer.selection_statistic == 'std'
        assert np.isfinite(viewer.last_section_image.values.compressed()).all()
        assert (viewer.last_section_image.values.compressed() >= 0).all()
        assert_reference(viewer, model)
        checks['uncertainty_callback_preserves_reference_mean_definition'] = True
        if dimension == 3:
            for selected_slice in (3, 7, 2):
                viewer.slice_slider.set_val(selected_slice)
                assert_nearest_context(viewer, model)
                assert_reference(viewer, model)
            checks['slice_context_section_reference_sync_including_periodic_interpolation'] = True
        else:
            assert viewer.slice_slider is None
            checks['no_slice_slider_in_2d'] = True
        viewer.context_input.set_val(str(context))
        viewer.statistic_buttons.set_active(0)
        displayed_rows.append(assert_routing(viewer, model))
        assert_reference(viewer, model)
        mask = np.ma.getmaskarray(viewer.last_reference_image.mean)
        assert mask.any() and (~mask).any()
        assert np.ma.getmaskarray(viewer.last_section_image.values).any()
        checks['uncovered_reference_and_unsupported_section_points_masked'] = True
        assert len(viewer.figure.axes) == axis_count
        assert viewer.history == history and state.sweep == 0 and viewer.paused
        assert_tensors_equal(before, state)
        assert_tensors_equal(snapshot_before, viewer.snapshot)
        assert torch.equal(cpu_rng, torch.get_rng_state())
        checks['controls_preserve_state_snapshot_rng_history_and_axes'] = True
        # A subsequent live recognition update must not alter the displayed snapshot.
        saved_beta = state.beta.clone()
        saved_mu = state.model.buckets[0].mu.clone()
        state.beta.zero_()
        state.model.buckets[0].mu.fill_(999.0)
        viewer.component_input.set_val('0')
        viewer.component_input.set_val('1')
        assert_tensors_equal(snapshot_before, viewer.snapshot)
        assert_routing(viewer, model)
        assert_reference(viewer, model)
        state.beta.copy_(saved_beta)
        state.model.buckets[0].mu.copy_(saved_mu)
        assert_tensors_equal(before, state)
        assert len(viewer.figure.axes) == axis_count and viewer.history == history
        assert torch.equal(cpu_rng, torch.get_rng_state())
        checks['later_live_updates_leave_captured_beta_and_reference_unchanged'] = True
        path = ROOT / f'docs/implementation/dashboard-viewer-{dimension}d.png'
        viewer.figure.canvas.draw()
        context_bounds = viewer.context_range.get_window_extent(viewer.figure.canvas.get_renderer())
        assert all(not context_bounds.overlaps(button.ax.bbox) for button in viewer.buttons)
        checks['context_range_clear_of_orchestration_buttons'] = True
        viewer.figure.savefig(path, dpi=160, bbox_inches='tight')
        click_button(viewer, 2)
        assert viewer.stopped and not viewer.ready()
        checks['stop_button_mouse_callback'] = True
        return {'dimension': dimension, 'n_agents': n_agents, 'K': 2,
                'grid_shape': [4] * dimension, 'patch_shape': [3] * dimension,
                'coverage': 'allow_uncovered', 'checks': checks, 'population_clicks': clicks,
                'displayed_rows': displayed_rows, 'plot_axis_count': axis_count,
                'reference_shape': list(viewer.last_reference_image.mean.shape),
                'supported_reference_nodes': int(viewer.last_reference_image.mean.count()),
                'masked_reference_nodes': int(mask.sum()), 'snapshot': str(path),
                'selection': {'agent': viewer.selection_agent, 'context': viewer.selection_context,
                              'channel': viewer.selection_channel, 'component': viewer.selection_component,
                              'statistic': viewer.selection_statistic, 'slice_index': viewer.selection_slice,
                              'z': viewer.last_reference_image.z}}
    finally:
        plt.close(viewer.figure)


def run_validation() -> dict:
    """Run only CPU dashboard checks, preserving earlier validation artifacts."""
    torch.set_num_threads(1)
    historical_paths = [ROOT / 'experiments/run_viewer_validation.py']
    historical_paths += [path for path in (ROOT / 'docs/implementation').glob('*viewer*')
                         if path.is_file() and not path.name.startswith('dashboard-')]
    historical = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in historical_paths}
    windows = [inspect_spatial(dimension) for dimension in (2, 3)]
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest for path, digest in historical.items())
    result = {'status': 'completed', 'backend': matplotlib.get_backend(), 'device': 'cpu',
              'dtype': 'float64', 'interpreter': sys.executable, 'torch': torch.__version__,
              'torch_threads': torch.get_num_threads(), 'gpu_work_or_query': False,
              'input_method': {'population_and_buttons': 'Matplotlib MouseEvent press/release in real Qt windows',
                               'numeric_inputs': 'TextBox.set_val dispatches registered submit callbacks',
                               'channel_and_statistic': 'RadioButtons.set_active dispatches registered callbacks',
                               'slice': 'Slider.set_val dispatches the registered changed callback',
                               'manual_interaction': False},
              'checks': {'historical_scripts_and_evidence_preserved': True},
              'spatial_windows': windows, 'historical_artifact_sha256': historical,
              'scope': 'Tiny CPU Qt callback/display validation; CUDA and manual human interaction remain for the user.'}
    write_json(ROOT / 'docs/implementation/dashboard-viewer-validation.json', result)
    return result


if __name__ == '__main__':
    print(run_validation())
