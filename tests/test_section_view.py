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
from vfe4.state import Incidence, SectionState


def _api() -> ModuleType:
    return import_module('vfe4.section_view')


def _point_fixture() -> tuple[SimulationModel, SectionState]:
    config = load_config(resolve_preset('tiny_point', {
        'runtime': {'device': 'cpu'}, 'bundle': {'K': 2, 'frame_spread': 0.0},
        'population': {'n_agents': 2, 'lattice_shape': (2,)}}))
    model = build_model(config)
    state = initialize_sections(model, config)
    state.belief.buckets[0].mu.copy_(torch.tensor([[1., 2.], [3., 5.]], dtype=config.dtype))
    state.model.buckets[0].mu.copy_(torch.tensor([[7., 11.], [13., 17.]], dtype=config.dtype))
    return model, state


def _spatial_fixture(
    dimension: int = 2,
    structure: str = 'full_patch',
) -> tuple[SimulationModel, SectionState]:
    config = load_config(resolve_preset(f'tiny_spatial_{dimension}d', {
        'runtime': {'device': 'cpu'}, 'bundle': {'K': 2, 'frame_field': 'constant', 'frame_spread': 0.0},
        'manifold': {'periods': (2.0,) * dimension, 'grid_shape': (2,) * dimension},
        'population': {'n_agents': 1, 'center_grid': (1,) * dimension,
                       'patch_shape': (1,) * dimension, 'coverage': 'allow_uncovered'},
        'recognition': {'covariance_structure': structure}}))
    model = build_model(config)
    return model, initialize_sections(model, config)


def _two_node_fixture(structure: str = 'full_patch') -> tuple[SimulationModel, SectionState]:
    model, _ = _spatial_fixture(structure=structure)
    # A declared two-node incidence isolates interpolation from the stencil builder.
    model.incidence = Incidence(torch.tensor([0, 0]), torch.tensor([0, 3]),
        torch.tensor([0, 2]), torch.tensor([0, 1]), torch.zeros(1, 2, dtype=model.config.dtype))
    model.frame_field = ConstantFrameField(torch.tensor([[[2., 1.], [0., 1.5]]], dtype=model.config.dtype))
    state = initialize_sections(model, model.config)
    bucket = state.belief.buckets[0]
    bucket.mu.copy_(torch.tensor([[1., 2., 3., -1.]], dtype=model.config.dtype))
    if structure == 'full_patch':
        Sigma = torch.tensor([[2., 0., .4, 0.], [0., 1., 0., .2],
                              [.4, 0., 1., 0.], [0., .2, 0., 3.]], dtype=model.config.dtype)
        bucket.chol.copy_(torch.linalg.cholesky(Sigma).unsqueeze(0))
    else:
        blocks = torch.diag_embed(torch.tensor([[[2., 1.], [1., 3.]]], dtype=model.config.dtype))
        bucket.chol.copy_(torch.linalg.cholesky(blocks))
    return model, state


def test_section_image_retains_cross_node_covariance_and_agent_frame() -> None:
    api = _api()
    model, state = _two_node_fixture()
    snapshot = api.capture_sections(state)
    mean = api.section_image(snapshot, model, 0, 'belief', 0, 'mean')
    std = api.section_image(snapshot, model, 0, 'belief', 0, 'std')
    # At (.75,.25), the two diagonal knots have equal bump weights.
    # Reference mean=(2,.5), covariance=diag(.95,1.1); U=[[2,1],[0,1.5]].
    assert mean.values.shape == (4, 4)
    assert mean.extent == (0.0, 2.0, 0.0, 2.0)
    assert mean.z is None
    assert not mean.values.mask[1, 0]
    assert mean.values[1, 0] == pytest.approx(4.5)
    assert std.values[1, 0] == pytest.approx(np.sqrt(4.9))


def test_section_image_masks_absent_support_and_wraps_the_seam() -> None:
    api = _api()
    model, state = _spatial_fixture()
    state.belief.buckets[0].mu.fill_(3.0)
    image = api.section_image(api.capture_sections(state), model, 0, 'belief', 0, 'mean')
    assert not image.values.mask[0, 0]
    assert image.values[0, 0] == pytest.approx(3.0)
    assert image.values.mask[1, 1]
    assert np.ma.is_masked(image.values[1, 1])
    assert not image.values.mask[3, 3]
    assert image.values[3, 3] == pytest.approx(3.0)


def test_section_image_uses_endpoint_exclusive_3d_slice() -> None:
    api = _api()
    model, state = _spatial_fixture(3)
    state.model.buckets[0].mu.copy_(torch.tensor([[7., 11.]], dtype=model.config.dtype))
    snapshot = api.capture_sections(state)
    distant = api.section_image(snapshot, model, 0, 'model', 1, 'mean', 2)
    seam = api.section_image(snapshot, model, 0, 'model', 1, 'mean', 3)
    assert distant.z == 1.0
    assert distant.values.mask.all()
    assert seam.z == 1.5
    assert seam.values.shape == (4, 4)
    assert seam.values[0, 0] == pytest.approx(11.0)


def test_point_image_selects_agent_channel_component_and_std() -> None:
    api = _api()
    model, state = _point_fixture()
    state.model.buckets[0].chol[1].copy_(torch.tensor([[2., 0.], [.5, 3.]], dtype=model.config.dtype))
    snapshot = api.capture_sections(state)
    image = api.section_image(snapshot, model, 1, 'model', 1, 'mean')
    std = api.section_image(snapshot, model, 1, 'model', 1, 'std')
    assert image.values.shape == (1,)
    assert image.extent is None and image.z is None
    assert image.values[0] == pytest.approx(17.0)
    assert std.values[0] == pytest.approx(np.sqrt(9.25))
    assert api.section_image(snapshot, model, 0, 'belief', 1, 'mean').values[0] == pytest.approx(2.0)


def test_section_image_honors_node_block_recognition() -> None:
    api = _api()
    model, state = _two_node_fixture('node_blocks')
    image = api.section_image(api.capture_sections(state), model, 0, 'belief', 0, 'std')
    # Independent node covariance gives diag(.75,1) before the frame congruence.
    assert image.values[1, 0] == pytest.approx(2.0)


def test_snapshot_freezes_detached_coefficients_and_shares_immutable_ids() -> None:
    api = _api()
    model, state = _point_fixture()
    state.sweep = 7
    state.belief.buckets[0].mu.requires_grad_(True)
    state.belief.buckets[0].chol.requires_grad_(True)
    snapshot = api.capture_sections(state)
    for source, captured in zip(state.belief.buckets, snapshot.belief.buckets):
        assert captured.agent_ids is source.agent_ids
        assert captured.incidence_ids is source.incidence_ids
        assert captured.mu.device == source.mu.device
        assert not captured.mu.requires_grad and not captured.chol.requires_grad
        assert captured.mu.data_ptr() != source.mu.data_ptr()
        assert captured.chol.data_ptr() != source.chol.data_ptr()
    with torch.no_grad():
        state.belief.buckets[0].mu.add_(100)
        state.belief.buckets[0].chol.mul_(2)
        state.model.buckets[0].mu.add_(100)
    state.sweep = 8
    assert snapshot.sweep == 7
    assert api.section_image(snapshot, model, 0, 'belief', 0, 'mean').values[0] == pytest.approx(1.0)
    assert api.section_image(snapshot, model, 1, 'model', 1, 'mean').values[0] == pytest.approx(17.0)
    with pytest.raises(FrozenInstanceError):
        snapshot.sweep = 9


def test_section_rendering_is_read_only_and_does_not_advance_rng() -> None:
    api = _api()
    model, state = _spatial_fixture()
    global_rng, link_rng = torch.get_rng_state().clone(), state.links.rng_state.clone()
    originals = [tensor.clone() for laws in (state.belief, state.model)
                 for bucket in laws.buckets for tensor in (bucket.mu, bucket.chol)]
    snapshot = api.capture_sections(state)
    captured = [tensor.clone() for laws in (snapshot.belief, snapshot.model)
                for bucket in laws.buckets for tensor in (bucket.mu, bucket.chol)]
    api.section_image(snapshot, model, 0, 'belief', 0, 'mean')
    api.section_image(snapshot, model, 0, 'model', 1, 'std')
    current = [tensor for laws in (state.belief, state.model)
               for bucket in laws.buckets for tensor in (bucket.mu, bucket.chol)]
    frozen = [tensor for laws in (snapshot.belief, snapshot.model)
              for bucket in laws.buckets for tensor in (bucket.mu, bucket.chol)]
    for before, after in zip(originals + captured, current + frozen):
        assert torch.equal(before, after)
    assert torch.equal(global_rng, torch.get_rng_state())
    assert torch.equal(link_rng, state.links.rng_state)


@pytest.mark.parametrize('agent,channel,component,statistic,slice_index', [
    (-1, 'belief', 0, 'mean', 0), (2, 'belief', 0, 'mean', 0),
    (0, 'other', 0, 'mean', 0), (0, 'belief', -1, 'mean', 0),
    (0, 'belief', 2, 'mean', 0), (0, 'belief', 0, 'variance', 0),
    (0, 'belief', 0, 'mean', 1), (0.5, 'belief', 0, 'mean', 0),
    (0, 'belief', 0.5, 'mean', 0), (0, 'belief', 0, 'mean', 0.5),
])
def test_section_image_rejects_invalid_selectors(
    agent: int, channel: str, component: int, statistic: str, slice_index: int,
) -> None:
    api = _api()
    model, state = _point_fixture()
    with pytest.raises((ValueError, TypeError)):
        api.section_image(api.capture_sections(state), model, agent, channel, component, statistic, slice_index)


@pytest.mark.parametrize('slice_index', [-1, 4])
def test_section_image_rejects_out_of_range_3d_slice(slice_index: int) -> None:
    api = _api()
    model, state = _spatial_fixture(3)
    with pytest.raises(ValueError):
        api.section_image(api.capture_sections(state), model, 0, 'belief', 0, 'mean', slice_index)
