from dataclasses import FrozenInstanceError
from importlib import import_module
from types import ModuleType

import numpy as np
import pytest
import torch

from vfe4.config import load_config
from vfe4.geometry.transport import ConstantFrameField
from vfe4.model import SimulationModel, build_model, initialize_sections
from vfe4.presets import resolve_preset
from vfe4.state import CandidateRows, Incidence, PatchGaussians, SectionState


def _api() -> ModuleType:
    return import_module('vfe4.dashboard')


def _fixture(dimension: int = 2) -> tuple[SimulationModel, SectionState]:
    if dimension == 0:
        raw = resolve_preset('tiny_point', {
            'runtime': {'device': 'cpu'}, 'bundle': {'K': 2, 'frame_spread': 0.0},
            'population': {'n_agents': 2, 'lattice_shape': (2,)}})
    else:
        raw = resolve_preset(f'tiny_spatial_{dimension}d', {
            'runtime': {'device': 'cpu'}, 'bundle': {'K': 2, 'frame_field': 'constant', 'frame_spread': 0.0},
            'manifold': {'periods': (2.0, 4.0) if dimension == 2 else (2.0, 4.0, 6.0),
                         'grid_shape': (2, 2) if dimension == 2 else (2, 2, 3)},
            'population': {'n_agents': 2, 'center_grid': (2,) + (1,) * (dimension - 1),
                           'patch_shape': (1,) * dimension, 'coverage': 'allow_uncovered'},
            'communication': {'channels': ('belief', 'model')}})
    config = load_config(raw)
    model = build_model(config)
    if dimension:
        contexts = [0, 1, 0, 2] if dimension == 2 else [0, 1, 2, 3, 0, 1, 2, 8]
        P = len(contexts) // 2
        model.incidence = Incidence(torch.arange(2).repeat_interleave(P), torch.tensor(contexts),
            torch.arange(3) * P, torch.arange(P).repeat(2), model.incidence.centers)
    model.frame_field = ConstantFrameField(torch.tensor([
        [[2., 1.], [0., 3.]], [[1., -2.], [1., 4.]]], dtype=config.dtype))
    state = initialize_sections(model, config)
    if dimension == 0:
        belief = [[1., 2.], [3., 5.]]
        other = [[7., 11.], [13., 17.]]
        truth = [[.5, 1.], [2., 3.]]
    elif dimension == 2:
        belief = [[2., 20., 6., 60.], [4., 40., 8., 80.]]
        other = [[12., 120., 16., 160.], [14., 140., 18., 180.]]
        truth = [[1., 10.], [4., 40.], [5., 50.], [7., 70.]]
    else:
        belief = [[2., 20., 8., 80., 18., 180., 10., 100.],
                  [6., 60., 12., 120., 22., 220., 30., 300.]]
        other = [[102., 120., 108., 180., 118., 280., 110., 200.],
                 [106., 160., 112., 220., 122., 320., 130., 400.]]
        truth = [[1., 10.], [4., 40.], [9., 90.], [5., 50.],
                 [3., 30.], [6., 60.], [11., 110.], [15., 150.]]
    state.belief.buckets[0].mu.copy_(torch.tensor(belief, dtype=config.dtype))
    state.model.buckets[0].mu.copy_(torch.tensor(other, dtype=config.dtype))
    model.truth = torch.tensor(truth, dtype=config.dtype)
    return model, state


@pytest.mark.parametrize('channel,component,mean,truth,error', [
    ('belief', 0, [3., 6., 8.], [3., 4., 7.], [0., 2., 1.]),
    ('belief', 1, [30., 60., 80.], [30., 40., 70.], [0., 20., 10.]),
    ('model', 0, [13., 16., 18.], [3., 4., 7.], [10., 12., 11.]),
    ('model', 1, [130., 160., 180.], [30., 40., 70.], [100., 120., 110.]),
])
def test_reference_averages_nodal_coefficients_without_agent_frames(
    channel: str, component: int, mean: list[float], truth: list[float], error: list[float],
) -> None:
    api = _api()
    model, state = _fixture()
    image = api.reference_image(api.capture_dashboard(state), model, channel, component)
    assert image.mean.shape == (2, 2)
    assert image.extent == (0., 2., 0., 4.) and image.z is None
    for values, expected in ((image.mean, mean), (image.truth, truth), (image.error, error)):
        np.testing.assert_array_equal(np.ma.getmaskarray(values), [[False, False], [False, True]])
        np.testing.assert_allclose(values.compressed(), expected)
        assert np.ma.is_masked(values[1, 1])


def test_reference_uses_means_without_materializing_covariance(monkeypatch: pytest.MonkeyPatch) -> None:
    api = _api()
    model, state = _fixture()
    snapshot = api.capture_dashboard(state)

    def forbidden_marginals(self: PatchGaussians, n_incidence: int, K: int) -> None:
        raise AssertionError('Reference means do not need covariance')

    monkeypatch.setattr(PatchGaussians, 'marginals', forbidden_marginals)
    assert api.reference_image(snapshot, model, 'belief', 0).mean[0, 0] == 3.


@pytest.mark.parametrize('slice_index,mean,mask', [
    (0, [4., 10.], [[False, False], [True, True]]),
    (1, [7.], [[False, True], [True, True]]),
    (2, [10.], [[False, True], [True, True]]),
    (3, [15.], [[False, True], [True, True]]),
    (4, [20., 30.], [[False, True], [False, True]]),
    (5, [12.], [[False, True], [True, True]]),
])
def test_reference_3d_uses_nodal_planes_and_periodic_linear_interpolation(
    slice_index: int, mean: list[float], mask: list[list[bool]],
) -> None:
    api = _api()
    model, state = _fixture(3)
    image = api.reference_image(api.capture_dashboard(state), model, 'belief', 0, slice_index)
    assert image.z == float(slice_index)
    assert image.extent == (0., 2., 0., 4.)
    np.testing.assert_array_equal(np.ma.getmaskarray(image.mean), mask)
    np.testing.assert_array_equal(np.ma.getmaskarray(image.truth), mask)
    np.testing.assert_array_equal(np.ma.getmaskarray(image.error), mask)
    np.testing.assert_allclose(image.mean.compressed(), mean)
    np.testing.assert_allclose(image.truth.compressed(), np.asarray(mean) / 2)
    np.testing.assert_allclose(image.error.compressed(), np.asarray(mean) / 2)


@pytest.mark.parametrize('channel,component,mean,truth', [
    ('belief', 0, [1., 3.], [.5, 2.]), ('belief', 1, [2., 5.], [1., 3.]),
    ('model', 0, [7., 13.], [.5, 2.]), ('model', 1, [11., 17.], [1., 3.]),
])
def test_point_reference_preserves_per_agent_values(
    channel: str, component: int, mean: list[float], truth: list[float],
) -> None:
    api = _api()
    model, state = _fixture(0)
    image = api.reference_image(api.capture_dashboard(state), model, channel, component)
    assert image.mean.shape == (2,)
    assert image.extent is None and image.z is None
    assert not np.ma.getmaskarray(image.mean).any()
    np.testing.assert_allclose(image.mean, mean)
    np.testing.assert_allclose(image.truth, truth)
    np.testing.assert_allclose(image.error, np.asarray(mean) - np.asarray(truth))


def _routing_fixture() -> tuple[SimulationModel, SectionState]:
    model, state = _fixture()
    # Four incidences, then four rows per channel. Model row 5 is explicitly empty.
    model.rows = CandidateRows(torch.tensor([0, 2, 3, 4, 5, 7, 7, 8, 9]),
        torch.tensor([0, 0, 1, 2, 3, 0, 0, 2, 3]),
        torch.tensor([0, 2, 1, 0, 3, 2, 0, 2, 3]),
        torch.arange(9), torch.zeros(9, dtype=model.config.dtype),
        torch.tensor([0, 0, 1, 2, 3, 4, 4, 6, 7]),
        torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]))
    state.beta = torch.tensor([.2, .6, .7, .9, .4, .35, .15, .55, .8], dtype=model.config.dtype)
    return model, state


@pytest.mark.parametrize('agent,channel,context,sources,probabilities', [
    (0, 'belief', 0, [0, 1], [.2, .6]), (0, 'belief', 1, [0], [.7]),
    (1, 'belief', 0, [0], [.9]), (1, 'belief', 2, [1], [.4]),
    (0, 'model', 0, [1, 0], [.35, .15]), (1, 'model', 0, [1], [.55]),
])
def test_routing_selects_one_actual_receiver_channel_context_row(
    agent: int, channel: str, context: int, sources: list[int], probabilities: list[float],
) -> None:
    api = _api()
    model, state = _routing_fixture()
    view = api.routing_view(api.capture_dashboard(state), model, agent, channel, context)
    assert view.status == 'active'
    np.testing.assert_array_equal(view.source_agents, sources)
    # Deliberately unnormalized beta detects accidental renormalization or pooling.
    np.testing.assert_allclose(view.probabilities, probabilities)


@pytest.mark.parametrize('agent,channel,context,status', [
    (0, 'model', 1, 'no_candidates'), (0, 'belief', 2, 'agent_absent'),
])
def test_routing_reports_missing_incidence_or_empty_row(
    agent: int, channel: str, context: int, status: str,
) -> None:
    api = _api()
    model, state = _routing_fixture()
    view = api.routing_view(api.capture_dashboard(state), model, agent, channel, context)
    assert view.status == status
    assert view.source_agents.shape == (0,) and view.probabilities.shape == (0,)


def test_routing_reports_disabled_channel() -> None:
    api = _api()
    model, state = _routing_fixture()
    model.config.raw['communication']['channels'] = ('belief',)
    view = api.routing_view(api.capture_dashboard(state), model, 0, 'model', 0)
    assert view.status == 'channel_disabled'
    assert view.source_agents.size == view.probabilities.size == 0


def test_dashboard_snapshot_freezes_beta_and_both_coefficient_laws() -> None:
    api = _api()
    model, state = _routing_fixture()
    state.sweep = 9
    state.beta.requires_grad_(True)
    for laws in (state.belief, state.model):
        laws.buckets[0].mu.requires_grad_(True)
        laws.buckets[0].chol.requires_grad_(True)
    snapshot = api.capture_dashboard(state)
    assert isinstance(snapshot, import_module('vfe4.section_view').SectionSnapshot)
    for before, after in [(state.beta, snapshot.beta)] + [
        (getattr(source, name), getattr(captured, name))
        for source_laws, captured_laws in ((state.belief, snapshot.belief), (state.model, snapshot.model))
        for source, captured in zip(source_laws.buckets, captured_laws.buckets)
        for name in ('mu', 'chol')]:
        assert before.device == after.device
        assert before.data_ptr() != after.data_ptr()
        assert not after.requires_grad and after.grad_fn is None
    with torch.no_grad():
        state.beta.zero_()
        for laws in (state.belief, state.model):
            laws.buckets[0].mu.add_(1000.)
            laws.buckets[0].chol.mul_(2.)
    state.sweep = 10
    assert snapshot.sweep == 9
    np.testing.assert_allclose(api.routing_view(snapshot, model, 0, 'model', 0).probabilities, [.35, .15])
    assert api.reference_image(snapshot, model, 'belief', 0).mean[0, 0] == 3.
    assert api.reference_image(snapshot, model, 'model', 0).mean[0, 0] == 13.
    with pytest.raises(FrozenInstanceError):
        snapshot.beta = state.beta


def test_dashboard_reads_leave_state_snapshot_truth_and_rng_unchanged() -> None:
    api = _api()
    model, state = _routing_fixture()
    snapshot = api.capture_dashboard(state)
    tensors = [state.beta, snapshot.beta, model.truth, state.links.rng_state]
    tensors += [tensor for value in (state, snapshot) for laws in (value.belief, value.model)
                for bucket in laws.buckets for tensor in (bucket.mu, bucket.chol)]
    before = [tensor.clone() for tensor in tensors]
    rng = torch.get_rng_state().clone()
    api.reference_image(snapshot, model, 'belief', 0)
    api.reference_image(snapshot, model, 'model', 1)
    api.routing_view(snapshot, model, 0, 'belief', 0)
    api.routing_view(snapshot, model, 0, 'model', 1)
    for original, actual in zip(before, tensors):
        assert torch.equal(original, actual)
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize('channel,component,slice_index', [
    ('other', 0, 0), ('belief', -1, 0), ('belief', 2, 0),
    ('belief', .5, 0), ('belief', 0, 1), ('belief', 0, .5), ('belief', 0, -1),
])
def test_reference_rejects_invalid_selectors(channel: str, component: int, slice_index: int) -> None:
    api = _api()
    model, state = _fixture(0)
    with pytest.raises((ValueError, TypeError)):
        api.reference_image(api.capture_dashboard(state), model, channel, component, slice_index)


@pytest.mark.parametrize('slice_index', [-1, 6])
def test_reference_rejects_out_of_range_3d_slice(slice_index: int) -> None:
    api = _api()
    model, state = _fixture(3)
    with pytest.raises(ValueError):
        api.reference_image(api.capture_dashboard(state), model, 'belief', 0, slice_index)


@pytest.mark.parametrize('agent,channel,context', [
    (-1, 'belief', 0), (2, 'belief', 0), (.5, 'belief', 0),
    (0, 'other', 0), (0, 'belief', -1), (0, 'belief', 4), (0, 'belief', .5),
])
def test_routing_rejects_invalid_selectors(agent: int, channel: str, context: int) -> None:
    api = _api()
    model, state = _routing_fixture()
    with pytest.raises((ValueError, TypeError)):
        api.routing_view(api.capture_dashboard(state), model, agent, channel, context)
