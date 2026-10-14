"""The live viewer must declare the same objective convention as saved output."""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytest

from vfe4.config import load_config
from vfe4.model import build_model, initialize_sections
from vfe4.objective import evaluate_vfe
from vfe4.presets import resolve_preset
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
