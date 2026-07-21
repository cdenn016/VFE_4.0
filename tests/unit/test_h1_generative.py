from __future__ import annotations

import math

import pytest
import torch

from vfe4.generative import H1GenerativeModel
from vfe4.numerics import gaussian_log_prob, selected_log_softmax
from vfe4.validation import enumerate_source_paths, load_h1_fixture


def _direct_component(model: H1GenerativeModel, path: object) -> tuple[torch.Tensor, torch.Tensor]:
    factors = model.factors
    transform = torch.zeros((6, 6), dtype=torch.float64)
    transform[0, 0] = 1.0
    transform[1, 1] = 1.0
    mean = torch.zeros(6, dtype=torch.float64)
    mean[:2] = factors.initial_joint.mean
    noise_covariance = torch.zeros((6, 6), dtype=torch.float64)
    noise_covariance[:2, :2] = factors.initial_joint.covariance

    for time in (1, 2):
        a = path.a[time - 1]  # type: ignore[attr-defined]
        b = path.b[time - 1]  # type: ignore[attr-defined]
        model_record = factors.model_transitions[time - 1]
        state_record = factors.state_transitions[time - 1]
        m_index = 2 * time + 1
        z_index = 2 * time
        m_source = 2 * b + 1
        z_source = 2 * a
        m_slope = model_record.source_slopes[b]
        z_slope = state_record.source_slopes[a]
        transform[m_index] = m_slope * transform[m_source]
        transform[m_index, m_index] += 1.0
        mean[m_index] = m_slope * mean[m_source] + model_record.offset
        transform[z_index] = (
            z_slope * transform[z_source]
            + state_record.model_slope * transform[m_index]
        )
        transform[z_index, z_index] += 1.0
        mean[z_index] = (
            z_slope * mean[z_source]
            + state_record.model_slope * mean[m_index]
            + state_record.offset
        )
        noise_covariance[m_index, m_index] = model_record.variance
        noise_covariance[z_index, z_index] = state_record.variance
    return mean, transform @ noise_covariance @ transform.T


def test_source_log_prob_contains_all_four_generative_priors() -> None:
    fixture = load_h1_fixture()
    model = H1GenerativeModel.from_fixture(fixture)

    values = [model.source_log_prob(path).exp().item() for path in enumerate_source_paths(fixture)]

    assert values == pytest.approx([
        1.0 * 1.0 * 0.35 * 0.55,
        1.0 * 1.0 * 0.35 * 0.45,
        1.0 * 1.0 * 0.65 * 0.55,
        1.0 * 1.0 * 0.65 * 0.45,
    ])
    assert math.fsum(values) == pytest.approx(1.0)


def test_emission_uses_one_based_labels_and_selected_decoder_rows() -> None:
    model = H1GenerativeModel.from_fixture(load_h1_fixture())
    y = torch.tensor([0.0, 0.0, 0.3, -0.2, -0.4, 0.5], dtype=torch.float64)

    actual = model.emission_log_prob(y, (1, 2))
    factors = model.factors
    logits_1 = factors.emissions[0].w_z * y[2] + factors.emissions[0].w_m * y[3] + factors.emissions[0].bias
    logits_2 = factors.emissions[1].w_z * y[4] + factors.emissions[1].w_m * y[5] + factors.emissions[1].bias

    assert actual.item() == pytest.approx(
        (selected_log_softmax(logits_1, 0) + selected_log_softmax(logits_2, 1)).item()
    )
    with pytest.raises(ValueError, match="label"):
        model.emission_log_prob(y, (0, 2))
    with pytest.raises(ValueError, match="label"):
        model.emission_log_prob(y, (1, 4))


def test_log_joint_includes_normalized_continuous_source_and_emission_factors() -> None:
    model = H1GenerativeModel.from_fixture(load_h1_fixture())
    path = enumerate_source_paths(load_h1_fixture())[3]
    component = model.joint_component(path)
    y = component.mean + torch.tensor([0.1, -0.2, 0.05, 0.12, -0.07, 0.09], dtype=torch.float64)

    expected = (
        gaussian_log_prob(y, component.mean, component.covariance)
        + model.source_log_prob(path)
        + model.emission_log_prob(y, (1, 2))
    )

    assert torch.isfinite(model.log_joint(y, path))
    assert model.log_joint(y, path).item() == pytest.approx(expected.item(), abs=1e-12)


@pytest.mark.parametrize("path_index", range(4))
def test_joint_component_matches_direct_affine_noise_assembly(path_index: int) -> None:
    fixture = load_h1_fixture()
    model = H1GenerativeModel.from_fixture(fixture)
    path = enumerate_source_paths(fixture)[path_index]

    component = model.joint_component(path)
    expected_mean, expected_covariance = _direct_component(model, path)

    assert component.mean.shape == (6,)
    assert component.covariance.shape == (6, 6)
    assert torch.allclose(component.mean, expected_mean, atol=1e-14, rtol=0.0)
    assert torch.allclose(component.covariance, expected_covariance, atol=1e-14, rtol=0.0)
    torch.linalg.cholesky(component.covariance)


def test_generative_factor_records_are_defensive() -> None:
    model = H1GenerativeModel.from_fixture(load_h1_fixture())
    factors = model.factors
    slopes = factors.model_transitions[1].source_slopes
    slopes[0] = 99.0

    assert model.factors.model_transitions[1].source_slopes[0].item() == pytest.approx(0.8)
