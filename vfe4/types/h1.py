"""Immutable records for the frozen normalized H1 reference law."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor

from vfe4.numerics import require_probability_vector, require_spd
from vfe4.types.structural import PopulationFrames, StructuralData


def _vector(value: Tensor, size: int, name: str) -> Tensor:
    if not isinstance(value, Tensor) or value.dtype is not torch.float64:
        raise ValueError(f"{name} must be a float64 tensor")
    if value.shape != (size,) or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be a finite vector of shape ({size},)")
    return value.detach().clone()


def _scalar(value: Tensor, name: str, *, positive: bool = False) -> Tensor:
    if not isinstance(value, Tensor) or value.dtype is not torch.float64:
        raise ValueError(f"{name} must be a float64 tensor")
    if value.shape != () or not bool(torch.isfinite(value)):
        raise ValueError(f"{name} must be a finite scalar")
    if positive and not bool(value > 0):
        raise ValueError(f"{name} must be positive")
    return value.detach().clone()


def _require_initial_shape(value: GaussianLaw, name: str) -> None:
    if not isinstance(value, GaussianLaw):
        raise ValueError(f"{name} must be a GaussianLaw")
    if value.mean.shape != (2,) or value.covariance.shape != (2, 2):
        raise ValueError(f"{name} must have a two-dimensional mean and covariance")


@dataclass(frozen=True, init=False)
class GaussianLaw:
    _mean: Tensor = field(repr=False, compare=False)
    _covariance: Tensor = field(repr=False, compare=False)

    def __init__(self, mean: Tensor, covariance: Tensor) -> None:
        checked_mean = _vector(mean, mean.numel() if isinstance(mean, Tensor) and mean.ndim == 1 else 0, "mean")
        checked_covariance = require_spd(covariance, name="covariance")
        if checked_covariance.shape != (checked_mean.numel(), checked_mean.numel()):
            raise ValueError("covariance shape must match mean")
        object.__setattr__(self, "_mean", checked_mean)
        object.__setattr__(self, "_covariance", checked_covariance)

    @property
    def mean(self) -> Tensor:
        return self._mean.detach().clone()

    @property
    def covariance(self) -> Tensor:
        return self._covariance.detach().clone()


@dataclass(frozen=True, init=False)
class ModelTransitionRecord:
    sources: tuple[int, ...]
    _source_slopes: Tensor = field(repr=False, compare=False)
    _offset: Tensor = field(repr=False, compare=False)
    _variance: Tensor = field(repr=False, compare=False)

    def __init__(self, sources: tuple[int, ...], source_slopes: Tensor, offset: Tensor, variance: Tensor) -> None:
        if type(sources) is not tuple or not sources or any(type(v) is not int or v < 0 for v in sources):
            raise ValueError("sources must be nonnegative integer indices")
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "_source_slopes", _vector(source_slopes, len(sources), "source_slopes"))
        object.__setattr__(self, "_offset", _scalar(offset, "offset"))
        object.__setattr__(self, "_variance", _scalar(variance, "variance", positive=True))

    @property
    def source_slopes(self) -> Tensor:
        return self._source_slopes.detach().clone()

    @property
    def offset(self) -> Tensor:
        return self._offset.detach().clone()

    @property
    def variance(self) -> Tensor:
        return self._variance.detach().clone()


@dataclass(frozen=True, init=False)
class StateTransitionRecord(ModelTransitionRecord):
    _model_slope: Tensor = field(repr=False, compare=False)

    def __init__(
        self,
        sources: tuple[int, ...],
        source_slopes: Tensor,
        model_slope: Tensor,
        offset: Tensor,
        variance: Tensor,
    ) -> None:
        super().__init__(sources, source_slopes, offset, variance)
        object.__setattr__(self, "_model_slope", _scalar(model_slope, "model_slope"))

    @property
    def model_slope(self) -> Tensor:
        return self._model_slope.detach().clone()


@dataclass(frozen=True, init=False)
class EmissionRecord:
    _w_z: Tensor = field(repr=False, compare=False)
    _w_m: Tensor = field(repr=False, compare=False)
    _bias: Tensor = field(repr=False, compare=False)

    def __init__(self, w_z: Tensor, w_m: Tensor, bias: Tensor) -> None:
        object.__setattr__(self, "_w_z", _vector(w_z, 3, "w_z"))
        object.__setattr__(self, "_w_m", _vector(w_m, 3, "w_m"))
        object.__setattr__(self, "_bias", _vector(bias, 3, "bias"))

    @property
    def w_z(self) -> Tensor:
        return self._w_z.detach().clone()

    @property
    def w_m(self) -> Tensor:
        return self._w_m.detach().clone()

    @property
    def bias(self) -> Tensor:
        return self._bias.detach().clone()


@dataclass(frozen=True, init=False)
class RecognitionModelKernelRecord:
    _slopes: Tensor = field(repr=False, compare=False)
    _offsets: Tensor = field(repr=False, compare=False)
    _variances: Tensor = field(repr=False, compare=False)

    def __init__(self, slopes: Tensor, offsets: Tensor, variances: Tensor) -> None:
        size = slopes.numel() if isinstance(slopes, Tensor) and slopes.ndim == 1 else 0
        object.__setattr__(self, "_slopes", _vector(slopes, size, "slopes"))
        object.__setattr__(self, "_offsets", _vector(offsets, size, "offsets"))
        checked = _vector(variances, size, "variances")
        if bool(torch.any(checked <= 0)):
            raise ValueError("variances must be positive")
        object.__setattr__(self, "_variances", checked)

    @property
    def slopes(self) -> Tensor:
        return self._slopes.detach().clone()

    @property
    def offsets(self) -> Tensor:
        return self._offsets.detach().clone()

    @property
    def variances(self) -> Tensor:
        return self._variances.detach().clone()


@dataclass(frozen=True, init=False)
class RecognitionStateKernelRecord:
    _z_slopes: Tensor = field(repr=False, compare=False)
    _m_slopes: Tensor = field(repr=False, compare=False)
    _offsets: Tensor = field(repr=False, compare=False)
    _variances: Tensor = field(repr=False, compare=False)

    def __init__(self, z_slopes: Tensor, m_slopes: Tensor, offsets: Tensor, variances: Tensor) -> None:
        size = z_slopes.numel() if isinstance(z_slopes, Tensor) and z_slopes.ndim == 1 else 0
        object.__setattr__(self, "_z_slopes", _vector(z_slopes, size, "z_slopes"))
        object.__setattr__(self, "_m_slopes", _vector(m_slopes, size, "m_slopes"))
        object.__setattr__(self, "_offsets", _vector(offsets, size, "offsets"))
        checked = _vector(variances, size, "variances")
        if bool(torch.any(checked <= 0)):
            raise ValueError("variances must be positive")
        object.__setattr__(self, "_variances", checked)

    @property
    def z_slopes(self) -> Tensor:
        return self._z_slopes.detach().clone()

    @property
    def m_slopes(self) -> Tensor:
        return self._m_slopes.detach().clone()

    @property
    def offsets(self) -> Tensor:
        return self._offsets.detach().clone()

    @property
    def variances(self) -> Tensor:
        return self._variances.detach().clone()


@dataclass(frozen=True, init=False)
class RecognitionParameterRecord:
    initial_joint: GaussianLaw
    _model_source_probabilities: tuple[Tensor, Tensor] = field(repr=False, compare=False)
    _state_source_probabilities_given_model_source: tuple[Tensor, Tensor] = field(repr=False, compare=False)
    model_kernels: tuple[RecognitionModelKernelRecord, RecognitionModelKernelRecord]
    state_kernels: tuple[RecognitionStateKernelRecord, RecognitionStateKernelRecord]

    def __init__(
        self,
        initial_joint: GaussianLaw,
        model_source_probabilities: tuple[Tensor, Tensor],
        state_source_probabilities_given_model_source: tuple[Tensor, Tensor],
        model_kernels: tuple[RecognitionModelKernelRecord, RecognitionModelKernelRecord],
        state_kernels: tuple[RecognitionStateKernelRecord, RecognitionStateKernelRecord],
    ) -> None:
        _require_initial_shape(initial_joint, "initial_joint")
        if type(model_source_probabilities) is not tuple or len(model_source_probabilities) != 2:
            raise ValueError("model_source_probabilities must be a pair")
        model_rows = tuple(require_probability_vector(row, name=f"model_source_probabilities[{i}]") for i, row in enumerate(model_source_probabilities))
        if tuple(row.shape for row in model_rows) != ((1,), (2,)):
            raise ValueError("model source probability shapes must be (1,) and (2,)")
        if type(state_source_probabilities_given_model_source) is not tuple or len(state_source_probabilities_given_model_source) != 2:
            raise ValueError("state source probabilities must be a pair")
        state_tables: list[Tensor] = []
        for time, (table, shape) in enumerate(zip(state_source_probabilities_given_model_source, ((1, 1), (2, 2)))):
            if not isinstance(table, Tensor) or table.dtype is not torch.float64 or table.shape != shape or not bool(torch.isfinite(table).all()):
                raise ValueError(f"state source probability table {time} has invalid shape or values")
            for row_index, row in enumerate(table):
                require_probability_vector(row, name=f"state_source_probabilities[{time}][{row_index}]")
            state_tables.append(table.detach().clone())
        if type(model_kernels) is not tuple or len(model_kernels) != 2 or not all(isinstance(v, RecognitionModelKernelRecord) for v in model_kernels):
            raise ValueError("model_kernels must contain two records")
        if type(state_kernels) is not tuple or len(state_kernels) != 2 or not all(isinstance(v, RecognitionStateKernelRecord) for v in state_kernels):
            raise ValueError("state_kernels must contain two records")
        if tuple(record.slopes.shape for record in model_kernels) != ((1,), (2,)):
            raise ValueError("model kernel shapes must be (1,) and (2,)")
        if tuple(record.z_slopes.shape for record in state_kernels) != ((1,), (4,)):
            raise ValueError("state kernel shapes must be (1,) and (4,)")
        object.__setattr__(self, "initial_joint", initial_joint)
        object.__setattr__(self, "_model_source_probabilities", model_rows)
        object.__setattr__(self, "_state_source_probabilities_given_model_source", tuple(state_tables))
        object.__setattr__(self, "model_kernels", model_kernels)
        object.__setattr__(self, "state_kernels", state_kernels)

    @property
    def model_source_probabilities(self) -> tuple[Tensor, Tensor]:
        return tuple(row.detach().clone() for row in self._model_source_probabilities)  # type: ignore[return-value]

    @property
    def state_source_probabilities_given_model_source(self) -> tuple[Tensor, Tensor]:
        return tuple(table.detach().clone() for table in self._state_source_probabilities_given_model_source)  # type: ignore[return-value]


@dataclass(frozen=True, init=False)
class H1GenerativeFactorRecord:
    initial_joint: GaussianLaw
    _model_source_priors: tuple[Tensor, Tensor] = field(repr=False, compare=False)
    _state_source_priors: tuple[Tensor, Tensor] = field(repr=False, compare=False)
    model_transitions: tuple[ModelTransitionRecord, ModelTransitionRecord]
    state_transitions: tuple[StateTransitionRecord, StateTransitionRecord]
    emissions: tuple[EmissionRecord, EmissionRecord]

    def __init__(self, initial_joint: GaussianLaw, model_source_priors: tuple[Tensor, Tensor], state_source_priors: tuple[Tensor, Tensor], model_transitions: tuple[ModelTransitionRecord, ModelTransitionRecord], state_transitions: tuple[StateTransitionRecord, StateTransitionRecord], emissions: tuple[EmissionRecord, EmissionRecord]) -> None:
        _require_initial_shape(initial_joint, "initial_joint")
        if type(model_source_priors) is not tuple or len(model_source_priors) != 2:
            raise ValueError("model_source_priors must be a pair")
        if type(state_source_priors) is not tuple or len(state_source_priors) != 2:
            raise ValueError("state_source_priors must be a pair")
        model_rows = tuple(require_probability_vector(row, name=f"model_source_priors[{i}]") for i, row in enumerate(model_source_priors))
        state_rows = tuple(require_probability_vector(row, name=f"state_source_priors[{i}]") for i, row in enumerate(state_source_priors))
        if tuple(row.shape for row in model_rows) != ((1,), (2,)) or tuple(row.shape for row in state_rows) != ((1,), (2,)):
            raise ValueError("source prior shapes must be (1,) and (2,)")
        if type(model_transitions) is not tuple or len(model_transitions) != 2 or not all(isinstance(v, ModelTransitionRecord) for v in model_transitions):
            raise ValueError("model_transitions must contain two records")
        if type(state_transitions) is not tuple or len(state_transitions) != 2 or not all(isinstance(v, StateTransitionRecord) for v in state_transitions):
            raise ValueError("state_transitions must contain two records")
        if type(emissions) is not tuple or len(emissions) != 2 or not all(isinstance(v, EmissionRecord) for v in emissions):
            raise ValueError("emissions must contain two records")
        if tuple(record.sources for record in model_transitions) != ((0,), (0, 1)):
            raise ValueError("model transition sources must match H1")
        if tuple(record.sources for record in state_transitions) != ((0,), (0, 1)):
            raise ValueError("state transition sources must match H1")
        object.__setattr__(self, "initial_joint", initial_joint)
        object.__setattr__(self, "_model_source_priors", model_rows)
        object.__setattr__(self, "_state_source_priors", state_rows)
        object.__setattr__(self, "model_transitions", model_transitions)
        object.__setattr__(self, "state_transitions", state_transitions)
        object.__setattr__(self, "emissions", emissions)

    @property
    def model_source_priors(self) -> tuple[Tensor, Tensor]:
        return tuple(row.detach().clone() for row in self._model_source_priors)  # type: ignore[return-value]

    @property
    def state_source_priors(self) -> tuple[Tensor, Tensor]:
        return tuple(row.detach().clone() for row in self._state_source_priors)  # type: ignore[return-value]


@dataclass(frozen=True, init=False)
class H1RecognitionFactorRecord:
    initial_joint: GaussianLaw
    _model_source_probabilities: tuple[Tensor, Tensor] = field(repr=False, compare=False)
    _state_source_probabilities_given_model_source: tuple[Tensor, Tensor] = field(repr=False, compare=False)
    model_kernels: tuple[RecognitionModelKernelRecord, RecognitionModelKernelRecord]
    state_kernels: tuple[RecognitionStateKernelRecord, RecognitionStateKernelRecord]

    def __init__(self, initial_joint: GaussianLaw, model_source_probabilities: tuple[Tensor, Tensor], state_source_probabilities_given_model_source: tuple[Tensor, Tensor], model_kernels: tuple[RecognitionModelKernelRecord, RecognitionModelKernelRecord], state_kernels: tuple[RecognitionStateKernelRecord, RecognitionStateKernelRecord]) -> None:
        checked = RecognitionParameterRecord(
            initial_joint,
            model_source_probabilities,
            state_source_probabilities_given_model_source,
            model_kernels,
            state_kernels,
        )
        object.__setattr__(self, "initial_joint", initial_joint)
        object.__setattr__(self, "_model_source_probabilities", checked.model_source_probabilities)
        object.__setattr__(self, "_state_source_probabilities_given_model_source", checked.state_source_probabilities_given_model_source)
        object.__setattr__(self, "model_kernels", model_kernels)
        object.__setattr__(self, "state_kernels", state_kernels)

    @property
    def model_source_probabilities(self) -> tuple[Tensor, Tensor]:
        return tuple(row.detach().clone() for row in self._model_source_probabilities)  # type: ignore[return-value]

    @property
    def state_source_probabilities_given_model_source(self) -> tuple[Tensor, Tensor]:
        return tuple(table.detach().clone() for table in self._state_source_probabilities_given_model_source)  # type: ignore[return-value]


@dataclass(frozen=True, init=False)
class H1Fixture:
    fixture_schema_version: Literal[1]
    fixture_id: Literal["h1-v1"]
    continuous_order: tuple[str, str, str, str, str, str]
    structural: StructuralData
    frames: PopulationFrames
    observation_labels: tuple[int, int]
    initial_joint: GaussianLaw
    _model_source_priors: tuple[Tensor, Tensor] = field(repr=False, compare=False)
    _state_source_priors: tuple[Tensor, Tensor] = field(repr=False, compare=False)
    model_transitions: tuple[ModelTransitionRecord, ModelTransitionRecord]
    state_transitions: tuple[StateTransitionRecord, StateTransitionRecord]
    emissions: tuple[EmissionRecord, EmissionRecord]
    recognition: RecognitionParameterRecord
    quadrature_order: Literal[21]
    convergence_check_order: Literal[17]
    maximum_convergence_estimate: float

    def __init__(self, fixture_schema_version: Literal[1], fixture_id: Literal["h1-v1"], continuous_order: tuple[str, str, str, str, str, str], structural: StructuralData, frames: PopulationFrames, observation_labels: tuple[int, int], initial_joint: GaussianLaw, model_source_priors: tuple[Tensor, Tensor], state_source_priors: tuple[Tensor, Tensor], model_transitions: tuple[ModelTransitionRecord, ModelTransitionRecord], state_transitions: tuple[StateTransitionRecord, StateTransitionRecord], emissions: tuple[EmissionRecord, EmissionRecord], recognition: RecognitionParameterRecord, quadrature_order: Literal[21], convergence_check_order: Literal[17], maximum_convergence_estimate: float) -> None:
        if fixture_schema_version != 1 or fixture_id != "h1-v1":
            raise ValueError("unsupported H1 fixture identity")
        if continuous_order != ("z0", "m0", "z1", "m1", "z2", "m2"):
            raise ValueError("continuous_order must match h1-v1")
        if not isinstance(structural, StructuralData) or (
            structural.horizon, structural.d_z, structural.d_m, structural.vocabulary_size,
            structural.state_parent_sets, structural.model_parent_sets,
            structural.state_source_support, structural.model_source_support,
        ) != (2, 1, 1, 3, ((0,), (0, 1)), ((0,), (0, 1)), ((0,), (0, 1)), ((0,), (0, 1))):
            raise ValueError("structural data must match h1-v1")
        if not isinstance(frames, PopulationFrames):
            raise ValueError("frames must be PopulationFrames")
        if type(observation_labels) is not tuple or len(observation_labels) != 2 or any(type(label) is not int or label < 1 or label > 3 for label in observation_labels):
            raise ValueError("observation_labels must be two labels from {1,2,3}")
        if not isinstance(recognition, RecognitionParameterRecord):
            raise ValueError("recognition must be a RecognitionParameterRecord")
        if quadrature_order != 21 or convergence_check_order != 17:
            raise ValueError("quadrature orders must match h1-v1")
        if type(maximum_convergence_estimate) is not float or not math.isfinite(maximum_convergence_estimate) or maximum_convergence_estimate != 1e-9:
            raise ValueError("maximum_convergence_estimate must equal 1e-9")
        factors = H1GenerativeFactorRecord(initial_joint, model_source_priors, state_source_priors, model_transitions, state_transitions, emissions)
        for name, value in (
            ("fixture_schema_version", fixture_schema_version), ("fixture_id", fixture_id),
            ("continuous_order", continuous_order), ("structural", structural), ("frames", frames),
            ("observation_labels", observation_labels), ("initial_joint", initial_joint),
            ("model_transitions", model_transitions), ("state_transitions", state_transitions),
            ("emissions", emissions), ("recognition", recognition), ("quadrature_order", quadrature_order),
            ("convergence_check_order", convergence_check_order), ("maximum_convergence_estimate", maximum_convergence_estimate),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_model_source_priors", factors.model_source_priors)
        object.__setattr__(self, "_state_source_priors", factors.state_source_priors)

    @property
    def model_source_priors(self) -> tuple[Tensor, Tensor]:
        return tuple(row.detach().clone() for row in self._model_source_priors)  # type: ignore[return-value]

    @property
    def state_source_priors(self) -> tuple[Tensor, Tensor]:
        return tuple(row.detach().clone() for row in self._state_source_priors)  # type: ignore[return-value]
