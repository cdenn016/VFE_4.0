"""Normalized recognition law for the frozen H1 reference fixture."""

from __future__ import annotations

import torch
from torch import Tensor

from vfe4.numerics import gaussian_log_prob
from vfe4.types.h1 import GaussianLaw, H1Fixture, H1RecognitionFactorRecord
from vfe4.types.structural import SourcePath


def _value(y: Tensor) -> Tensor:
    if not isinstance(y, Tensor) or y.dtype is not torch.float64:
        raise ValueError("y must be a float64 tensor")
    if y.shape != (6,) or not bool(torch.isfinite(y).all()):
        raise ValueError("y must be a finite vector with shape (6,)")
    return y


def _path(path: SourcePath) -> SourcePath:
    if not isinstance(path, SourcePath) or path.a[0] != 0 or path.b[0] != 0 or path.a[1] not in (0, 1) or path.b[1] not in (0, 1):
        raise ValueError("path is outside the frozen H1 source support")
    return path


class H1RecognitionLaw:
    def __init__(self, factors: H1RecognitionFactorRecord) -> None:
        self._factors = factors

    @classmethod
    def from_fixture(cls, fixture: H1Fixture) -> "H1RecognitionLaw":
        if not isinstance(fixture, H1Fixture):
            raise ValueError("fixture must be an H1Fixture")
        record = fixture.recognition
        return cls(H1RecognitionFactorRecord(
            record.initial_joint,
            record.model_source_probabilities,
            record.state_source_probabilities_given_model_source,
            record.model_kernels,
            record.state_kernels,
        ))

    @property
    def factors(self) -> H1RecognitionFactorRecord:
        return self._factors

    def source_probability(self, path: SourcePath) -> Tensor:
        checked = _path(path)
        result = torch.ones((), dtype=torch.float64)
        model_probabilities = self._factors.model_source_probabilities
        state_probabilities = self._factors.state_source_probabilities_given_model_source
        for time in range(2):
            b = checked.b[time]
            a = checked.a[time]
            result = result * model_probabilities[time][b] * state_probabilities[time][b, a]
        return result

    def log_prob(self, y: Tensor, path: SourcePath) -> Tensor:
        checked_y = _value(y)
        checked_path = _path(path)
        result = gaussian_log_prob(
            checked_y[:2], self._factors.initial_joint.mean, self._factors.initial_joint.covariance
        ) + torch.log(self.source_probability(checked_path))
        for time in (1, 2):
            a = checked_path.a[time - 1]
            b = checked_path.b[time - 1]
            model_kernel = self._factors.model_kernels[time - 1]
            state_kernel = self._factors.state_kernels[time - 1]
            state_slot = 0 if time == 1 else a + 2 * b
            m_value = checked_y[2 * time + 1]
            z_value = checked_y[2 * time]
            m_mean = model_kernel.slopes[b] * checked_y[2 * b + 1] + model_kernel.offsets[b]
            z_mean = (
                state_kernel.z_slopes[state_slot] * checked_y[2 * a]
                + state_kernel.m_slopes[state_slot] * m_value
                + state_kernel.offsets[state_slot]
            )
            result = result + gaussian_log_prob(
                m_value.reshape(1), m_mean.reshape(1), model_kernel.variances[b].reshape(1, 1)
            )
            result = result + gaussian_log_prob(
                z_value.reshape(1), z_mean.reshape(1), state_kernel.variances[state_slot].reshape(1, 1)
            )
        return result

    def joint_component(self, path: SourcePath) -> GaussianLaw:
        checked_path = _path(path)
        transform = torch.zeros((6, 6), dtype=torch.float64)
        transform[0, 0] = 1.0
        transform[1, 1] = 1.0
        mean = torch.zeros(6, dtype=torch.float64)
        mean[:2] = self._factors.initial_joint.mean
        noise_covariance = torch.zeros((6, 6), dtype=torch.float64)
        noise_covariance[:2, :2] = self._factors.initial_joint.covariance
        for time in (1, 2):
            a = checked_path.a[time - 1]
            b = checked_path.b[time - 1]
            model_kernel = self._factors.model_kernels[time - 1]
            state_kernel = self._factors.state_kernels[time - 1]
            state_slot = 0 if time == 1 else a + 2 * b
            m_index, z_index = 2 * time + 1, 2 * time
            m_source, z_source = 2 * b + 1, 2 * a
            m_slope = model_kernel.slopes[b]
            z_slope = state_kernel.z_slopes[state_slot]
            state_m_slope = state_kernel.m_slopes[state_slot]
            transform[m_index] = m_slope * transform[m_source]
            transform[m_index, m_index] += 1.0
            mean[m_index] = m_slope * mean[m_source] + model_kernel.offsets[b]
            transform[z_index] = z_slope * transform[z_source] + state_m_slope * transform[m_index]
            transform[z_index, z_index] += 1.0
            mean[z_index] = z_slope * mean[z_source] + state_m_slope * mean[m_index] + state_kernel.offsets[state_slot]
            noise_covariance[m_index, m_index] = model_kernel.variances[b]
            noise_covariance[z_index, z_index] = state_kernel.variances[state_slot]
        covariance = transform @ noise_covariance @ transform.transpose(0, 1)
        covariance = 0.5 * (covariance + covariance.transpose(0, 1))
        return GaussianLaw(mean, covariance)
