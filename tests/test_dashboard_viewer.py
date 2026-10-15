"""Whole-population controls must inspect one frozen inference snapshot."""

import matplotlib
matplotlib.use('Agg')

from matplotlib.backend_bases import MouseEvent
import matplotlib.pyplot as plt
import numpy as np

from vfe4.config import load_config
from vfe4.model import SimulationModel, build_model, initialize_sections
from vfe4.presets import resolve_preset
from vfe4.state import SectionState
from vfe4.viewer import SimulationViewer


def _fixture(dimension: int) -> tuple[SectionState, SimulationModel]:
    config = load_config(resolve_preset(f'tiny_spatial_{dimension}d', {
        'manifold': {'grid_shape': (4,) * dimension},
        'population': {'n_agents': 2, 'center_grid': (2, 1) if dimension == 2 else (1, 1, 2),
                       'patch_shape': (3, 3) if dimension == 2 else (1, 1, 3),
                       'coverage': 'allow_uncovered'},
        'bundle': {'K': 2}}))
    model = build_model(config)
    state = initialize_sections(model, config)
    for channel, offset in ((state.belief, 1.), (state.model, 4.)):
        channel.buckets[0].mu.reshape(2, -1, 2)[...] = channel.buckets[0].mu.new_tensor([offset, offset + 1.])
    return state, model


def _click_agent(viewer: SimulationViewer, agent: int) -> None:
    viewer.figure.canvas.draw()
    x, y = viewer.population_ax.transData.transform(viewer.population_positions[agent, :2])
    event = MouseEvent('button_press_event', viewer.figure.canvas, x, y, button=1)
    viewer.figure.canvas.callbacks.process('button_press_event', event)


def test_population_click_and_reference_controls_use_frozen_snapshot() -> None:
    state, model = _fixture(2)
    viewer = SimulationViewer()
    try:
        viewer.update(state, model, 2.)
        axes_count = len(viewer.figure.axes)
        assert len(viewer.reference_axes) == 3
        _click_agent(viewer, 1)
        assert viewer.selection_agent == 1 and viewer.agent_input.text == '1'
        assert 'context' in viewer.population_ax.get_title().lower()
        reference = viewer.last_reference_image.mean.copy()
        routing = viewer.last_routing.probabilities.copy()
        state.belief.buckets[0].mu.fill_(999.)
        state.beta.zero_()
        viewer.channel_buttons.set_active(1)
        assert not np.array_equal(reference.compressed(), viewer.last_reference_image.mean.compressed())
        viewer.channel_buttons.set_active(0)
        np.testing.assert_array_equal(reference, viewer.last_reference_image.mean)
        np.testing.assert_array_equal(routing, viewer.last_routing.probabilities)
        viewer.component_input.set_val('1')
        assert not np.array_equal(reference.compressed(), viewer.last_reference_image.mean.compressed())
        assert len(viewer.figure.axes) == axes_count and len(viewer.history) == 1 and state.sweep == 0
        assert all('reference' in ax.get_title().lower() for ax in viewer.reference_axes)
        viewer.figure.canvas.draw()
        bounds = viewer.context_range.get_window_extent(viewer.figure.canvas.get_renderer())
        assert all(not bounds.overlaps(button.ax.bbox) for button in viewer.buttons)
    finally:
        plt.close(viewer.figure)


def test_context_and_3d_slice_selection_stay_synchronized() -> None:
    state, model = _fixture(3)
    viewer = SimulationViewer()
    try:
        viewer.update(state, model, 2.)
        viewer.slice_slider.set_val(4)
        assert viewer.selection_context == 2 and viewer.last_reference_image.z == 0.5
        _click_agent(viewer, 1)
        assert viewer.selection_agent == 1
        viewer.context_input.set_val('1')
        assert viewer.selection_slice == 2 and viewer.last_reference_image.z == 0.25
        assert viewer.last_section_image.z == 0.25
        assert viewer.last_routing.status == 'active'
        viewer.context_input.set_val('64')
        assert viewer.selection_context == 1 and viewer.status_text.get_text()
        viewer.slice_slider.set_val(3)
        assert viewer.last_reference_image.z == viewer.last_section_image.z == 0.375
        assert any('interpol' in ax.get_title().lower() for ax in viewer.reference_axes)
        assert len(viewer.history) == 1
    finally:
        plt.close(viewer.figure)


def test_missing_receiver_context_is_reported_without_an_overlay() -> None:
    state, model = _fixture(3)
    viewer = SimulationViewer()
    try:
        viewer.update(state, model, 2.)
        viewer.context_input.set_val('16')  # x=0.25, outside the one-node xy patch.
        assert viewer.last_routing.status == 'agent_absent'
        assert not viewer.last_routing.probabilities.size
        assert 'absent' in viewer.population_ax.get_title().lower()
    finally:
        plt.close(viewer.figure)


def test_dashboard_retains_point_mode_and_labels_probability() -> None:
    config = load_config(resolve_preset('tiny_point', {
        'population': {'n_agents': 2, 'lattice_shape': (2,)}, 'bundle': {'K': 1}}))
    model = build_model(config)
    state = initialize_sections(model, config)
    viewer = SimulationViewer()
    try:
        viewer.update(state, model, 0.)
        assert viewer.last_reference_image.mean.shape == (2,)
        assert viewer.last_routing.status == 'active'
        assert 'self' in viewer.population_ax.get_title().lower()
        _click_agent(viewer, 1)
        assert viewer.selection_agent == 1
        assert viewer.slice_slider is None and viewer.selection_context == 0
    finally:
        plt.close(viewer.figure)
