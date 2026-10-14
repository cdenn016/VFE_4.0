"""The live viewer must declare the same objective convention as saved output."""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest

from vfe4.config import load_config
from vfe4.model import SimulationModel, build_model, initialize_sections
from vfe4.objective import evaluate_vfe
from vfe4.presets import resolve_preset
from vfe4.state import SectionState
from vfe4.viewer import SimulationViewer


@pytest.mark.parametrize('coefficient, required', [(0.0, 'sampled'), (0.1, 'relative')])
def test_sampled_viewer_objective_convention(coefficient: float, required: str) -> None:
    config = load_config(resolve_preset('tiny_point', {
        'population': {'n_agents': 2, 'lattice_shape': (2,)}, 'bundle': {'K': 1},
        'links': {'mode': 'latent_gaussian', 'samples': 2, 'mutual_loop_coefficient': coefficient}}))
    model = build_model(config)
    state = initialize_sections(model, config)
    viewer = SimulationViewer()
    try:
        viewer.update(state, model, float(evaluate_vfe(state, model).total))
        assert required in viewer.axes[0].get_ylabel().lower()
    finally:
        plt.close(viewer.figure)


def test_interactive_section_selection_uses_frozen_agent_law() -> None:
    config = load_config(resolve_preset('tiny_spatial_2d', {
        'manifold': {'grid_shape': (4, 4)},
        'population': {'n_agents': 2, 'center_grid': (2, 1), 'patch_shape': (1, 1),
                       'coverage': 'allow_uncovered'},
        'bundle': {'frame_field': 'constant', 'frame_spread': 0.0}}))
    model = build_model(config)
    state = initialize_sections(model, config)
    state.model.buckets[0].mu[:] = state.model.buckets[0].mu.new_tensor([[10., 20.], [30., 40.]])
    viewer = SimulationViewer()
    try:
        viewer.update(state, model, 1.25)
        viewer.agent_input.set_val('1')
        viewer.component_input.set_val('1')
        viewer.channel_buttons.set_active(1)
        assert viewer.selection_agent == 1 and viewer.selection_channel == 'model'
        np.testing.assert_allclose(viewer.last_section_image.values.compressed(), 40.)
        assert viewer.last_section_image.values.mask.any()
        assert 'agent frame' in viewer.axes[1].get_title().lower()
        axis_count = len(viewer.figure.axes)
        state.model.buckets[0].mu.fill_(999.)
        viewer.statistic_buttons.set_active(1)
        assert np.all(viewer.last_section_image.values.compressed() >= 0)
        viewer.statistic_buttons.set_active(0)
        np.testing.assert_allclose(viewer.last_section_image.values.compressed(), 40.)
        assert len(viewer.history) == 1 and len(viewer.figure.axes) == axis_count
        assert state.sweep == 0
    finally:
        plt.close(viewer.figure)


def test_interactive_3d_slice_and_invalid_selection() -> None:
    config = load_config(resolve_preset('tiny_spatial_3d', {
        'manifold': {'grid_shape': (4, 4, 4)},
        'population': {'n_agents': 2, 'center_grid': (1, 1, 2), 'patch_shape': (1, 1, 1),
                       'coverage': 'allow_uncovered'}}))
    model = build_model(config)
    state = initialize_sections(model, config)
    viewer = SimulationViewer()
    try:
        viewer.update(state, model, 0.5)
        assert viewer.last_section_image.z == 0.0
        viewer.slice_slider.set_val(4)
        assert viewer.last_section_image.z == 0.5 and viewer.last_section_image.values.mask.all()
        viewer.agent_input.set_val('1')
        assert viewer.selection_slice == 4 and not viewer.last_section_image.values.mask.all()
        viewer.agent_input.set_val('2')
        assert viewer.selection_agent == 1 and viewer.status_text.get_text()
        viewer.component_input.set_val('-1')
        assert viewer.selection_component == 0 and viewer.status_text.get_text()
        assert len(viewer.history) == 1
    finally:
        plt.close(viewer.figure)


def test_paused_single_sweep_refreshes_before_snapshot_cadence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vfe4 import viewer as viewer_module
    from vfe4.simulation import run_simulation
    frames = []

    class PausedViewer:
        def __init__(self) -> None:
            self.paused, self.calls = True, 0

        def ready(self) -> bool:
            self.calls += 1
            return self.calls == 1

        def update(self, state: SectionState, model: SimulationModel, objective: float) -> None:
            frames.append(state.sweep)

    monkeypatch.setattr(viewer_module, 'SimulationViewer', PausedViewer)
    config = load_config(resolve_preset('tiny_point', {
        'population': {'n_agents': 2, 'lattice_shape': (2,)}, 'bundle': {'K': 1},
        'inference': {'sweeps': 3}, 'output': {'directory': str(tmp_path), 'viewer': True,
            'snapshot_every': 5, 'figures': False}}))
    result = run_simulation(config)
    assert result.completed_sweeps == 1 and frames == [1]


def test_first_snapshot_visible_and_controls_clear_labels() -> None:
    config = load_config(resolve_preset('tiny_point', {
        'population': {'n_agents': 2, 'lattice_shape': (2,)}, 'bundle': {'K': 1}}))
    model = build_model(config)
    state = initialize_sections(model, config)
    viewer = SimulationViewer()
    try:
        viewer.update(state, model, float(evaluate_vfe(state, model).total))
        viewer.figure.canvas.draw()
        assert viewer.axes[0].lines[0].get_marker() not in ('None', '', ' ')
        renderer = viewer.figure.canvas.get_renderer()
        label_bounds = viewer.axes[0].xaxis.label.get_window_extent(renderer)
        assert all(not label_bounds.overlaps(button.ax.bbox) for button in viewer.buttons)
    finally:
        plt.close(viewer.figure)
