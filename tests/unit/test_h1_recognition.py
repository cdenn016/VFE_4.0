from __future__ import annotations

import math

import pytest
import torch

from vfe4.numerics import gaussian_log_prob
from vfe4.recognition import H1RecognitionLaw
from vfe4.validation import enumerate_source_paths, load_h1_fixture


def _direct_component(law: H1RecognitionLaw, path: object) -> tuple[torch.Tensor, torch.Tensor]:
    factors = law.factors
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
        model_kernel = factors.model_kernels[time - 1]
        state_kernel = factors.state_kernels[time - 1]
        state_slot = 0 if time == 1 else a + 2 * b
        m_index = 2 * time + 1
        z_index = 2 * time
        m_source = 2 * b + 1
        z_source = 2 * a
        m_slope = model_kernel.slopes[b]
        z_slope = state_kernel.z_slopes[state_slot]
        state_m_slope = state_kernel.m_slopes[state_slot]
        transform[m_index] = m_slope * transform[m_source]
        transform[m_index, m_index] += 1.0
        mean[m_index] = m_slope * mean[m_source] + model_kernel.offsets[b]
        transform[z_index] = z_slope * transform[z_source] + state_m_slope * transform[m_index]
        transform[z_index, z_index] += 1.0
        mean[z_index] = (
            z_slope * mean[z_source]
            + state_m_slope * mean[m_index]
            + state_kernel.offsets[state_slot]
        )
        noise_covariance[m_index, m_index] = model_kernel.variances[b]
        noise_covariance[z_index, z_index] = state_kernel.variances[state_slot]
    return mean, transform @ noise_covariance @ transform.T


def test_recognition_has_exact_four_path_weights_in_frozen_order() -> None:
    fixture = load_h1_fixture()
    law = H1RecognitionLaw.from_fixture(fixture)

    weights = [law.source_probability(path).item() for path in enumerate_source_paths(fixture)]

    assert weights == pytest.approx([0.30, 0.10, 0.12, 0.48])
    assert math.fsum(weights) == pytest.approx(1.0)


def test_recognition_log_prob_includes_source_mass_and_normalized_gaussian() -> None:
    law = H1RecognitionLaw.from_fixture(load_h1_fixture())
    path = enumerate_source_paths(load_h1_fixture())[2]
    component = law.joint_component(path)
    y = component.mean + torch.tensor([-0.1, 0.1, 0.03, -0.08, 0.12, -0.04], dtype=torch.float64)

    expected = gaussian_log_prob(y, component.mean, component.covariance) + torch.log(
        law.source_probability(path)
    )

    assert torch.isfinite(law.log_prob(y, path))
    assert law.log_prob(y, path).item() == pytest.approx(expected.item(), abs=1e-12)


@pytest.mark.parametrize("path_index", range(4))
def test_recognition_component_matches_its_own_affine_noise_assembly(path_index: int) -> None:
    fixture = load_h1_fixture()
    law = H1RecognitionLaw.from_fixture(fixture)
    path = enumerate_source_paths(fixture)[path_index]

    component = law.joint_component(path)
    expected_mean, expected_covariance = _direct_component(law, path)

    assert component.mean.shape == (6,)
    assert component.covariance.shape == (6, 6)
    assert torch.allclose(component.mean, expected_mean, atol=1e-14, rtol=0.0)
    assert torch.allclose(component.covariance, expected_covariance, atol=1e-14, rtol=0.0)
    torch.linalg.cholesky(component.covariance)


def test_recognition_source_tables_and_kernels_are_normalized_and_defensive() -> None:
    law = H1RecognitionLaw.from_fixture(load_h1_fixture())
    factors = law.factors

    for row in factors.model_source_probabilities:
        assert row.sum().item() == pytest.approx(1.0)
    for table in factors.state_source_probabilities_given_model_source:
        assert torch.allclose(table.sum(dim=1), torch.ones(table.shape[0], dtype=torch.float64))
    returned = factors.state_kernels[1].z_slopes
    returned[0] = 99.0
    assert law.factors.state_kernels[1].z_slopes[0].item() == pytest.approx(0.65)
