"""Normalized generative law for the frozen H1 reference fixture."""

from __future__ import annotations

import torch
from torch import Tensor

from vfe4.numerics import gaussian_log_prob, selected_log_softmax
from vfe4.types.h1 import GaussianLaw, H1Fixture, H1GenerativeFactorRecord
from vfe4.types.structural import SourcePath
from vfe4.validation import label_to_index


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


class H1GenerativeModel:
    def __init__(self, factors: H1GenerativeFactorRecord, observation_labels: tuple[int, int]) -> None:
        self._factors = factors
        self._observation_labels = observation_labels

    @classmethod
    def from_fixture(cls, fixture: H1Fixture) -> "H1GenerativeModel":
        if not isinstance(fixture, H1Fixture):
            raise ValueError("fixture must be an H1Fixture")
        return cls(
            H1GenerativeFactorRecord(
                fixture.initial_joint,
                fixture.model_source_priors,
                fixture.state_source_priors,
                fixture.model_transitions,
                fixture.state_transitions,
                fixture.emissions,
            ),
            fixture.observation_labels,
        )

    @property
    def factors(self) -> H1GenerativeFactorRecord:
        return self._factors

    def source_log_prob(self, path: SourcePath) -> Tensor:
        checked = _path(path)
        result = torch.zeros((), dtype=torch.float64)
        model_priors = self._factors.model_source_priors
        state_priors = self._factors.state_source_priors
        for time in range(2):
            model_probability = model_priors[time][checked.b[time]]
            state_probability = state_priors[time][checked.a[time]]
            result = result + torch.log(model_probability) + torch.log(state_probability)
        return result

    def log_joint(self, y: Tensor, path: SourcePath) -> Tensor:
        checked_y = _value(y)
        checked_path = _path(path)
        result = gaussian_log_prob(
            checked_y[:2], self._factors.initial_joint.mean, self._factors.initial_joint.covariance
        ) + self.source_log_prob(checked_path)
        for time in (1, 2):
            a = checked_path.a[time - 1]
            b = checked_path.b[time - 1]
            model_record = self._factors.model_transitions[time - 1]
            state_record = self._factors.state_transitions[time - 1]
            m_value = checked_y[2 * time + 1]
            z_value = checked_y[2 * time]
            m_mean = model_record.source_slopes[b] * checked_y[2 * b + 1] + model_record.offset
            z_mean = (
                state_record.source_slopes[a] * checked_y[2 * a]
                + state_record.model_slope * m_value
                + state_record.offset
            )
            result = result + gaussian_log_prob(
                m_value.reshape(1), m_mean.reshape(1), model_record.variance.reshape(1, 1)
            )
            result = result + gaussian_log_prob(
                z_value.reshape(1), z_mean.reshape(1), state_record.variance.reshape(1, 1)
            )
        return result + self.emission_log_prob(checked_y, self._observation_labels)

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
            model_record = self._factors.model_transitions[time - 1]
            state_record = self._factors.state_transitions[time - 1]
            m_index, z_index = 2 * time + 1, 2 * time
            m_source, z_source = 2 * b + 1, 2 * a
            m_slope = model_record.source_slopes[b]
            z_slope = state_record.source_slopes[a]
            transform[m_index] = m_slope * transform[m_source]
            transform[m_index, m_index] += 1.0
            mean[m_index] = m_slope * mean[m_source] + model_record.offset
            transform[z_index] = z_slope * transform[z_source] + state_record.model_slope * transform[m_index]
            transform[z_index, z_index] += 1.0
            mean[z_index] = z_slope * mean[z_source] + state_record.model_slope * mean[m_index] + state_record.offset
            noise_covariance[m_index, m_index] = model_record.variance
            noise_covariance[z_index, z_index] = state_record.variance
        covariance = transform @ noise_covariance @ transform.transpose(0, 1)
        covariance = 0.5 * (covariance + covariance.transpose(0, 1))
        return GaussianLaw(mean, covariance)

    def emission_log_prob(self, y: Tensor, observations: tuple[int, int]) -> Tensor:
        checked_y = _value(y)
        if type(observations) is not tuple or len(observations) != 2:
            raise ValueError("observations must be a pair of one-based labels")
        result = torch.zeros((), dtype=torch.float64)
        for time, (observation, emission) in enumerate(zip(observations, self._factors.emissions), start=1):
            index = label_to_index(observation, vocabulary_size=3)
            logits = emission.w_z * checked_y[2 * time] + emission.w_m * checked_y[2 * time + 1] + emission.bias
            result = result + selected_log_softmax(logits, index)
        return result
