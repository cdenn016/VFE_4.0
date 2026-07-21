"""Independent NumPy oracle for the frozen normalized H1 fixture.

This module intentionally has no dependency on :mod:`vfe4`.  It parses the
data-only fixture, assembles both directed Gaussian laws, evaluates evidence,
and evaluates the ELBO identity from independently implemented formulas.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Literal

import numpy as np


_EPSILON = float(np.finfo(np.float64).eps)
_ROOT_FIELDS = {
    "fixture_schema_version",
    "fixture_id",
    "continuous_order",
    "vocabulary_labels",
    "observation_label_base",
    "observation_labels",
    "frames",
    "initial_joint",
    "model_source_priors",
    "state_source_priors",
    "model_offsets",
    "model_variances",
    "state_offsets",
    "state_variances",
    "state_model_slopes",
    "decoder",
    "recognition",
    "quadrature",
}
_PATHS = ((0, 0), (1, 0), (0, 1), (1, 1))


@dataclass(frozen=True)
class IndependentNumericalAllowance:
    convergence_estimate: float
    rounding_allowance: float

    def __post_init__(self) -> None:
        _nonnegative(self.convergence_estimate, "convergence_estimate")
        _nonnegative(self.rounding_allowance, "rounding_allowance")
        _finite(self.total, "total allowance")
        if self.convergence_estimate > 1.0e-9:
            raise ValueError("convergence_estimate exceeds the frozen H1 maximum")

    @property
    def total(self) -> float:
        return self.convergence_estimate + self.rounding_allowance


@dataclass(frozen=True)
class H1EvidenceRecord:
    observation_labels: tuple[int, int]
    probability: float
    log_probability: float
    probability_allowance: IndependentNumericalAllowance
    log_probability_allowance: IndependentNumericalAllowance

    def __post_init__(self) -> None:
        _observation_pair(self.observation_labels)
        probability = _positive(self.probability, "evidence probability")
        if probability > 1.0:
            raise ValueError("evidence probability must not exceed one")
        _finite(self.log_probability, "log evidence")
        if not isinstance(self.probability_allowance, IndependentNumericalAllowance):
            raise ValueError("probability_allowance must be independent")
        if not isinstance(self.log_probability_allowance, IndependentNumericalAllowance):
            raise ValueError("log_probability_allowance must be independent")
        expected_log = math.log(probability)
        consistency_allowance = 8.0 * _EPSILON * max(1.0, abs(expected_log))
        if abs(self.log_probability - expected_log) > consistency_allowance:
            raise ValueError("probability and log_probability are inconsistent")


@dataclass(frozen=True)
class H1IdentityRecord:
    evidence: H1EvidenceRecord
    posterior_kl: float
    elbo_from_identity: float
    quadrature_order: Literal[21]
    convergence_check_order: Literal[17]
    posterior_kl_allowance: IndependentNumericalAllowance
    identity_allowance: IndependentNumericalAllowance

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, H1EvidenceRecord):
            raise ValueError("evidence must be an H1EvidenceRecord")
        _finite(self.posterior_kl, "posterior KL")
        _finite(self.elbo_from_identity, "identity ELBO")
        if type(self.quadrature_order) is not int or self.quadrature_order != 21:
            raise ValueError("quadrature_order must equal 21")
        if (
            type(self.convergence_check_order) is not int
            or self.convergence_check_order != 17
        ):
            raise ValueError("convergence_check_order must equal 17")
        if not isinstance(self.posterior_kl_allowance, IndependentNumericalAllowance):
            raise ValueError("posterior_kl_allowance must be independent")
        if not isinstance(self.identity_allowance, IndependentNumericalAllowance):
            raise ValueError("identity_allowance must be independent")
        if self.posterior_kl < -self.posterior_kl_allowance.total:
            raise ValueError("posterior KL is negative beyond its allowance")
        expected = self.evidence.log_probability - self.posterior_kl
        consistency_allowance = 16.0 * _EPSILON * math.fsum(
            (1.0, abs(expected), abs(self.elbo_from_identity))
        )
        if abs(self.elbo_from_identity - expected) > consistency_allowance:
            raise ValueError("identity ELBO is inconsistent with log evidence and posterior KL")


@dataclass(frozen=True)
class IndependentTermAllowances:
    expected_log_emission: tuple[
        IndependentNumericalAllowance, IndependentNumericalAllowance
    ]
    initial_model_kl: IndependentNumericalAllowance
    initial_state_kl: IndependentNumericalAllowance
    model_source_kl: tuple[
        IndependentNumericalAllowance, IndependentNumericalAllowance
    ]
    model_transition_kl: tuple[
        IndependentNumericalAllowance, IndependentNumericalAllowance
    ]
    state_source_kl: tuple[
        IndependentNumericalAllowance, IndependentNumericalAllowance
    ]
    state_transition_kl: tuple[
        IndependentNumericalAllowance, IndependentNumericalAllowance
    ]
    joint_recognition_entropy: IndependentNumericalAllowance
    complete_elbo: IndependentNumericalAllowance

    def __post_init__(self) -> None:
        for name in (
            "expected_log_emission",
            "model_source_kl",
            "model_transition_kl",
            "state_source_kl",
            "state_transition_kl",
        ):
            value = getattr(self, name)
            if type(value) is not tuple or len(value) != 2 or not all(
                isinstance(item, IndependentNumericalAllowance) for item in value
            ):
                raise ValueError(f"{name} must be a pair of independent allowances")
        for name in (
            "initial_model_kl",
            "initial_state_kl",
            "joint_recognition_entropy",
            "complete_elbo",
        ):
            if not isinstance(getattr(self, name), IndependentNumericalAllowance):
                raise ValueError(f"{name} must be an independent allowance")


@dataclass(frozen=True)
class IndependentTermRecord:
    expected_log_emission: tuple[float, float]
    initial_model_kl: float
    initial_state_kl: float
    model_source_kl: tuple[float, float]
    model_transition_kl: tuple[float, float]
    state_source_kl: tuple[float, float]
    state_transition_kl: tuple[float, float]
    joint_recognition_entropy: float
    complete_elbo: float
    allowances: IndependentTermAllowances

    def __post_init__(self) -> None:
        for name in (
            "expected_log_emission",
            "model_source_kl",
            "model_transition_kl",
            "state_source_kl",
            "state_transition_kl",
        ):
            values = getattr(self, name)
            if type(values) is not tuple or len(values) != 2:
                raise ValueError(f"{name} must be a pair")
            for value in values:
                _finite(value, name)
        for name in (
            "initial_model_kl",
            "initial_state_kl",
            "joint_recognition_entropy",
            "complete_elbo",
        ):
            _finite(getattr(self, name), name)
        _nonnegative(self.initial_model_kl, "initial_model_kl")
        _nonnegative(self.initial_state_kl, "initial_state_kl")
        for name in (
            "model_source_kl",
            "model_transition_kl",
            "state_source_kl",
            "state_transition_kl",
        ):
            for value in getattr(self, name):
                _nonnegative(value, name)
        if not isinstance(self.allowances, IndependentTermAllowances):
            raise ValueError("allowances must be IndependentTermAllowances")
        expected = math.fsum(
            (
                *self.expected_log_emission,
                -self.initial_model_kl,
                -self.initial_state_kl,
                -self.model_source_kl[0],
                -self.model_transition_kl[0],
                -self.model_source_kl[1],
                -self.model_transition_kl[1],
                -self.state_source_kl[0],
                -self.state_transition_kl[0],
                -self.state_source_kl[1],
                -self.state_transition_kl[1],
            )
        )
        arithmetic_allowance = 256.0 * _EPSILON * math.fsum(
            (
                1.0,
                *(abs(value) for value in self.expected_log_emission),
                abs(self.initial_model_kl),
                abs(self.initial_state_kl),
                *(abs(value) for value in self.model_source_kl),
                *(abs(value) for value in self.model_transition_kl),
                *(abs(value) for value in self.state_source_kl),
                *(abs(value) for value in self.state_transition_kl),
            )
        )
        if abs(self.complete_elbo - expected) > arithmetic_allowance:
            raise ValueError("complete_elbo is inconsistent with its local terms")


@dataclass(frozen=True)
class _Reduction:
    value: float
    absolute_sum: float

    def __post_init__(self) -> None:
        _finite(self.value, "reduction value")
        _nonnegative(self.absolute_sum, "reduction absolute sum")


@dataclass(frozen=True)
class _Gaussian:
    mean: np.ndarray
    covariance: np.ndarray


@dataclass(frozen=True)
class _Emission:
    w_z: np.ndarray
    w_m: np.ndarray
    bias: np.ndarray


@dataclass(frozen=True)
class _GenerativeFixture:
    observations: tuple[int, int]
    frames: np.ndarray
    initial: _Gaussian
    model_priors: tuple[np.ndarray, np.ndarray]
    state_priors: tuple[np.ndarray, np.ndarray]
    model_offsets: np.ndarray
    model_variances: np.ndarray
    state_offsets: np.ndarray
    state_variances: np.ndarray
    state_model_slopes: np.ndarray
    emissions: tuple[_Emission, _Emission]
    maximum_convergence_estimate: float


@dataclass(frozen=True)
class _ModelKernel:
    slopes: np.ndarray
    offsets: np.ndarray
    variances: np.ndarray


@dataclass(frozen=True)
class _StateKernel:
    z_slopes: np.ndarray
    m_slopes: np.ndarray
    offsets: np.ndarray
    variances: np.ndarray


@dataclass(frozen=True)
class _RecognitionFixture:
    initial: _Gaussian
    model_probabilities: tuple[np.ndarray, np.ndarray]
    state_probabilities: tuple[np.ndarray, np.ndarray]
    model_kernels: tuple[_ModelKernel, _ModelKernel]
    state_kernels: tuple[_StateKernel, _StateKernel]


@dataclass(frozen=True)
class _CompleteFixture:
    generative: _GenerativeFixture
    recognition: _RecognitionFixture


@dataclass(frozen=True)
class _TermEvaluation:
    expected_log_emission: tuple[_Reduction, _Reduction]
    initial_model_kl: _Reduction
    initial_state_kl: _Reduction
    model_source_kl: tuple[_Reduction, _Reduction]
    model_transition_kl: tuple[_Reduction, _Reduction]
    state_source_kl: tuple[_Reduction, _Reduction]
    state_transition_kl: tuple[_Reduction, _Reduction]
    joint_recognition_entropy: _Reduction
    complete_elbo: _Reduction


@dataclass(frozen=True)
class _IdentityEvaluation:
    evidence: _Reduction
    log_evidence: _Reduction
    posterior_kl: _Reduction
    elbo: _Reduction


def _finite(value: object, name: str) -> float:
    if type(value) not in (int, float) and not isinstance(value, (np.integer, np.floating)):
        raise ValueError(f"{name} must be finite numeric data")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite numeric data")
    return float(value)


def _nonnegative(value: object, name: str) -> float:
    checked = _finite(value, name)
    if checked < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return checked


def _positive(value: object, name: str) -> float:
    checked = _finite(value, name)
    if checked <= 0.0:
        raise ValueError(f"{name} must be positive")
    return checked


def _fields(value: object, expected: set[str], name: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"{name} fields must equal {sorted(expected)}")
    return value


def _sequence(value: object, size: int, name: str) -> list[object]:
    if type(value) is not list or len(value) != size:
        raise ValueError(f"{name} must be a list of length {size}")
    return value


def _integer(value: object, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def _vector(value: object, size: int, name: str) -> np.ndarray:
    sequence = _sequence(value, size, name)
    result = np.asarray(
        [_finite(item, f"{name}[{index}]") for index, item in enumerate(sequence)],
        dtype=np.float64,
    )
    result.setflags(write=False)
    return result


def _matrix(value: object, rows: int, columns: int, name: str) -> np.ndarray:
    outer = _sequence(value, rows, name)
    result = np.asarray(
        [
            [
                _finite(item, f"{name}[{row_index}][{column_index}]")
                for column_index, item in enumerate(
                    _sequence(row, columns, f"{name}[{row_index}]")
                )
            ]
            for row_index, row in enumerate(outer)
        ],
        dtype=np.float64,
    )
    result.setflags(write=False)
    return result


def _probability_vector(value: object, size: int, name: str) -> np.ndarray:
    result = _vector(value, size, name)
    if bool(np.any(result < 0.0)):
        raise ValueError(f"{name} must be nonnegative")
    if abs(math.fsum(float(item) for item in result) - 1.0) > 64.0 * _EPSILON:
        raise ValueError(f"{name} must sum to one")
    return result


def _probability_matrix(value: object, rows: int, columns: int, name: str) -> np.ndarray:
    result = _matrix(value, rows, columns, name)
    if bool(np.any(result < 0.0)):
        raise ValueError(f"{name} must be nonnegative")
    for row_index, row in enumerate(result):
        if abs(math.fsum(float(item) for item in row) - 1.0) > 64.0 * _EPSILON:
            raise ValueError(f"{name}[{row_index}] must sum to one")
    return result


def _gaussian(mean: object, covariance: object, size: int, name: str) -> _Gaussian:
    checked_mean = _vector(mean, size, f"{name}.mean")
    checked_covariance = _matrix(covariance, size, size, f"{name}.covariance")
    if not np.array_equal(checked_covariance, checked_covariance.T):
        raise ValueError(f"{name}.covariance must be symmetric")
    _checked_spd(checked_covariance, f"{name}.covariance")
    return _Gaussian(checked_mean, checked_covariance)


def _checked_spd(covariance: np.ndarray, name: str) -> np.ndarray:
    if not isinstance(covariance, np.ndarray) or covariance.ndim != 2:
        raise ValueError(f"{name} must be a matrix")
    if covariance.shape[0] != covariance.shape[1]:
        raise ValueError(f"{name} must be square")
    if not bool(np.isfinite(covariance).all()):
        raise ValueError(f"{name} must be finite")
    try:
        chol = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be SPD") from exc
    sign, slogdet = np.linalg.slogdet(covariance)
    chol_logdet = 2.0 * math.fsum(math.log(float(item)) for item in np.diag(chol))
    if sign != 1.0 or not math.isfinite(float(slogdet)):
        raise ValueError(f"{name} must have a finite positive determinant")
    agreement = 128.0 * _EPSILON * max(1.0, abs(float(slogdet)), abs(chol_logdet))
    if abs(float(slogdet) - chol_logdet) > agreement:
        raise ValueError(f"{name} Cholesky and slogdet disagree")
    if not bool(np.isfinite(chol).all()):
        raise ValueError(f"{name} Cholesky factor must be finite")
    return chol


def _read_root(path: Path) -> dict[str, object]:
    if not isinstance(path, Path):
        raise ValueError("fixture_path must be a pathlib.Path")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"fixture JSON could not be loaded: {exc}") from exc
    return _fields(raw, _ROOT_FIELDS, "fixture")


def _parse_generative(root: dict[str, object]) -> _GenerativeFixture:
    if _integer(root["fixture_schema_version"], "fixture_schema_version") != 1:
        raise ValueError("fixture_schema_version must equal 1")
    if root["fixture_id"] != "h1-v1":
        raise ValueError("fixture_id must equal h1-v1")
    if tuple(_sequence(root["continuous_order"], 6, "continuous_order")) != (
        "z0",
        "m0",
        "z1",
        "m1",
        "z2",
        "m2",
    ):
        raise ValueError("continuous_order must match h1-v1")
    vocabulary = tuple(
        _integer(item, f"vocabulary_labels[{index}]")
        for index, item in enumerate(
            _sequence(root["vocabulary_labels"], 3, "vocabulary_labels")
        )
    )
    if vocabulary != (1, 2, 3):
        raise ValueError("vocabulary_labels must equal [1, 2, 3]")
    if _integer(root["observation_label_base"], "observation_label_base") != 1:
        raise ValueError("observation_label_base must equal 1")
    observations = _observation_pair(
        tuple(
            _integer(item, f"observation_labels[{index}]")
            for index, item in enumerate(
                _sequence(root["observation_labels"], 2, "observation_labels")
            )
        )
    )
    frames = _vector(root["frames"], 3, "frames")
    if bool(np.any(frames == 0.0)):
        raise ValueError("frames must be nonzero")

    initial_raw = _fields(root["initial_joint"], {"mean", "covariance"}, "initial_joint")
    initial = _gaussian(
        initial_raw["mean"], initial_raw["covariance"], 2, "initial_joint"
    )
    model_prior_raw = _sequence(root["model_source_priors"], 2, "model_source_priors")
    state_prior_raw = _sequence(root["state_source_priors"], 2, "state_source_priors")
    model_priors = (
        _probability_vector(model_prior_raw[0], 1, "model_source_priors[0]"),
        _probability_vector(model_prior_raw[1], 2, "model_source_priors[1]"),
    )
    state_priors = (
        _probability_vector(state_prior_raw[0], 1, "state_source_priors[0]"),
        _probability_vector(state_prior_raw[1], 2, "state_source_priors[1]"),
    )
    model_offsets = _vector(root["model_offsets"], 2, "model_offsets")
    model_variances = _vector(root["model_variances"], 2, "model_variances")
    state_offsets = _vector(root["state_offsets"], 2, "state_offsets")
    state_variances = _vector(root["state_variances"], 2, "state_variances")
    state_model_slopes = _vector(root["state_model_slopes"], 2, "state_model_slopes")
    if bool(np.any(model_variances <= 0.0)) or bool(np.any(state_variances <= 0.0)):
        raise ValueError("generative conditional variances must be positive")

    decoder_raw = _sequence(root["decoder"], 2, "decoder")
    emissions_list: list[_Emission] = []
    for time, value in enumerate(decoder_raw):
        record = _fields(value, {"w_z", "w_m", "bias"}, f"decoder[{time}]")
        emissions_list.append(
            _Emission(
                _vector(record["w_z"], 3, f"decoder[{time}].w_z"),
                _vector(record["w_m"], 3, f"decoder[{time}].w_m"),
                _vector(record["bias"], 3, f"decoder[{time}].bias"),
            )
        )
    quadrature = _fields(
        root["quadrature"],
        {"order", "convergence_check_order", "maximum_convergence_estimate"},
        "quadrature",
    )
    if _integer(quadrature["order"], "quadrature.order") != 21:
        raise ValueError("quadrature.order must equal 21")
    if _integer(quadrature["convergence_check_order"], "quadrature.convergence_check_order") != 17:
        raise ValueError("quadrature.convergence_check_order must equal 17")
    maximum_raw = quadrature["maximum_convergence_estimate"]
    if type(maximum_raw) is not float or maximum_raw != 1.0e-9:
        raise ValueError(
            "maximum_convergence_estimate must be the float 1e-9"
        )
    maximum = maximum_raw
    return _GenerativeFixture(
        observations,
        frames,
        initial,
        model_priors,
        state_priors,
        model_offsets,
        model_variances,
        state_offsets,
        state_variances,
        state_model_slopes,
        (emissions_list[0], emissions_list[1]),
        maximum,
    )


def _parse_recognition(
    root: dict[str, object], generative: _GenerativeFixture
) -> _RecognitionFixture:
    raw = _fields(
        root["recognition"],
        {
            "initial_mean",
            "initial_covariance",
            "model_source_probabilities",
            "state_source_probabilities_given_model_source",
            "model_kernels",
            "state_kernels",
        },
        "recognition",
    )
    initial = _gaussian(
        raw["initial_mean"], raw["initial_covariance"], 2, "recognition.initial"
    )
    model_probability_raw = _sequence(
        raw["model_source_probabilities"], 2, "recognition.model_source_probabilities"
    )
    model_probabilities = (
        _probability_vector(
            model_probability_raw[0], 1, "recognition.model_source_probabilities[0]"
        ),
        _probability_vector(
            model_probability_raw[1], 2, "recognition.model_source_probabilities[1]"
        ),
    )
    state_probability_raw = _sequence(
        raw["state_source_probabilities_given_model_source"],
        2,
        "recognition.state_source_probabilities_given_model_source",
    )
    state_probabilities = (
        _probability_matrix(
            state_probability_raw[0],
            1,
            1,
            "recognition.state_source_probabilities_given_model_source[0]",
        ),
        _probability_matrix(
            state_probability_raw[1],
            2,
            2,
            "recognition.state_source_probabilities_given_model_source[1]",
        ),
    )

    model_kernel_raw = _sequence(raw["model_kernels"], 2, "recognition.model_kernels")
    model_kernels_list: list[_ModelKernel] = []
    for time, size in enumerate((1, 2)):
        records = [
            _fields(item, {"slope", "offset", "variance"}, f"recognition.model_kernels[{time}][{slot}]")
            for slot, item in enumerate(
                _sequence(model_kernel_raw[time], size, f"recognition.model_kernels[{time}]")
            )
        ]
        variances = np.asarray(
            [_positive(record["variance"], "recognition model variance") for record in records],
            dtype=np.float64,
        )
        model_kernels_list.append(
            _ModelKernel(
                np.asarray([_finite(record["slope"], "recognition model slope") for record in records], dtype=np.float64),
                np.asarray([_finite(record["offset"], "recognition model offset") for record in records], dtype=np.float64),
                variances,
            )
        )

    state_kernel_raw = _sequence(raw["state_kernels"], 2, "recognition.state_kernels")
    state_kernels_list: list[_StateKernel] = []
    for time, size in enumerate((1, 4)):
        expected = {"z_slope", "m_slope", "offset", "variance"}
        if time == 1:
            expected = expected | {"a", "b"}
        records = [
            _fields(item, expected, f"recognition.state_kernels[{time}][{slot}]")
            for slot, item in enumerate(
                _sequence(state_kernel_raw[time], size, f"recognition.state_kernels[{time}]")
            )
        ]
        if time == 1:
            tags = tuple(
                (_integer(record["a"], "state-kernel a"), _integer(record["b"], "state-kernel b"))
                for record in records
            )
            if tags != ((0, 0), (1, 0), (0, 1), (1, 1)):
                raise ValueError("recognition state-kernel source order is invalid")
        state_kernels_list.append(
            _StateKernel(
                np.asarray([_finite(record["z_slope"], "recognition z slope") for record in records], dtype=np.float64),
                np.asarray([_finite(record["m_slope"], "recognition m slope") for record in records], dtype=np.float64),
                np.asarray([_finite(record["offset"], "recognition state offset") for record in records], dtype=np.float64),
                np.asarray([_positive(record["variance"], "recognition state variance") for record in records], dtype=np.float64),
            )
        )

    recognition = _RecognitionFixture(
        initial,
        model_probabilities,
        state_probabilities,
        (model_kernels_list[0], model_kernels_list[1]),
        (state_kernels_list[0], state_kernels_list[1]),
    )
    _validate_positive_support(generative, recognition)
    return recognition


def _validate_positive_support(
    generative: _GenerativeFixture, recognition: _RecognitionFixture
) -> None:
    for time in range(2):
        for b, q_b in enumerate(recognition.model_probabilities[time]):
            if q_b > 0.0 and generative.model_priors[time][b] <= 0.0:
                raise ValueError(
                    "recognition model-source mass lies outside positive generative support"
                )
            for a, q_a in enumerate(recognition.state_probabilities[time][b]):
                if q_b > 0.0 and q_a > 0.0 and generative.state_priors[time][a] <= 0.0:
                    raise ValueError(
                        "recognition state-source mass lies outside positive generative support"
                    )


def _load_generative_fixture(path: Path) -> _GenerativeFixture:
    return _parse_generative(_read_root(path))


def _load_complete_fixture(path: Path) -> _CompleteFixture:
    root = _read_root(path)
    generative = _parse_generative(root)
    return _CompleteFixture(generative, _parse_recognition(root, generative))


def _observation_pair(value: object) -> tuple[int, int]:
    if type(value) is not tuple or len(value) != 2:
        raise ValueError("observation labels must be a pair")
    first = _label_to_index(value[0])
    second = _label_to_index(value[1])
    return (first + 1, second + 1)


def _label_to_index(label: int) -> int:
    if type(label) is not int or label < 1 or label > 3:
        raise ValueError("label must be an integer in [1, 3]")
    return label - 1


def _validate_orders(quadrature_order: object, convergence_check_order: object) -> None:
    if type(quadrature_order) is not int or quadrature_order != 21:
        raise ValueError("quadrature_order must equal the frozen order 21")
    if type(convergence_check_order) is not int or convergence_check_order != 17:
        raise ValueError("convergence_check_order must equal the frozen order 17")


def _validate_single_order(order: object) -> int:
    if type(order) is not int or order not in (17, 21):
        raise ValueError("quadrature order must be 17 or 21")
    return order


@lru_cache(maxsize=4)
def _standard_normal_grid(order: int, dimension: int) -> tuple[np.ndarray, np.ndarray]:
    """Return lexicographically ordered standard-normal Hermite nodes."""
    _validate_single_order(order)
    if type(dimension) is not int or dimension not in (2, 4):
        raise ValueError("quadrature dimension must be two or four")
    physicists_nodes, physicists_weights = np.polynomial.hermite.hermgauss(order)
    nodes = math.sqrt(2.0) * physicists_nodes
    weights = physicists_weights / math.sqrt(math.pi)
    lexicographic_indices = np.asarray(
        tuple(product(range(order), repeat=dimension)), dtype=np.int64
    )
    points = nodes[lexicographic_indices]
    point_weights = np.prod(weights[lexicographic_indices], axis=1)
    if not bool(np.isfinite(points).all()) or not bool(np.isfinite(point_weights).all()):
        raise ValueError("quadrature grid must be finite")
    if abs(math.fsum(float(item) for item in point_weights) - 1.0) > 64.0 * _EPSILON:
        raise ValueError("quadrature weights must sum to one")
    points.setflags(write=False)
    point_weights.setflags(write=False)
    return points, point_weights


def _assemble_generative_component(
    fixture: _GenerativeFixture, path: tuple[int, int]
) -> _Gaussian:
    """Assemble one generative 6x6 law from its own affine-noise chain."""
    if path not in _PATHS:
        raise ValueError("path is outside H1 source support")
    a_second, b_second = path
    transform = np.zeros((6, 6), dtype=np.float64)
    transform[0, 0] = 1.0
    transform[1, 1] = 1.0
    mean = np.zeros(6, dtype=np.float64)
    mean[:2] = fixture.initial.mean
    noise_covariance = np.zeros((6, 6), dtype=np.float64)
    noise_covariance[:2, :2] = fixture.initial.covariance

    for time in (1, 2):
        a = 0 if time == 1 else a_second
        b = 0 if time == 1 else b_second
        m_index = 2 * time + 1
        z_index = 2 * time
        m_source = 2 * b + 1
        z_source = 2 * a
        m_slope = fixture.frames[time] / fixture.frames[b]
        z_slope = fixture.frames[time] / fixture.frames[a]

        transform[m_index] = m_slope * transform[m_source]
        transform[m_index, m_index] += 1.0
        mean[m_index] = (
            m_slope * mean[m_source] + fixture.model_offsets[time - 1]
        )
        transform[z_index] = (
            z_slope * transform[z_source]
            + fixture.state_model_slopes[time - 1] * transform[m_index]
        )
        transform[z_index, z_index] += 1.0
        mean[z_index] = (
            z_slope * mean[z_source]
            + fixture.state_model_slopes[time - 1] * mean[m_index]
            + fixture.state_offsets[time - 1]
        )
        noise_covariance[m_index, m_index] = fixture.model_variances[time - 1]
        noise_covariance[z_index, z_index] = fixture.state_variances[time - 1]

    covariance = transform @ noise_covariance @ transform.T
    covariance = 0.5 * (covariance + covariance.T)
    if not bool(np.isfinite(mean).all()) or not bool(np.isfinite(covariance).all()):
        raise ValueError("generative component must be finite")
    _checked_spd(covariance, "generative component covariance")
    return _Gaussian(mean, covariance)


def _assemble_recognition_component(
    fixture: _RecognitionFixture, path: tuple[int, int]
) -> _Gaussian:
    """Separately assemble one recognition 6x6 affine-noise law."""
    if path not in _PATHS:
        raise ValueError("path is outside H1 source support")
    a_second, b_second = path
    transform = np.zeros((6, 6), dtype=np.float64)
    transform[0, 0] = 1.0
    transform[1, 1] = 1.0
    mean = np.zeros(6, dtype=np.float64)
    mean[:2] = fixture.initial.mean
    noise_covariance = np.zeros((6, 6), dtype=np.float64)
    noise_covariance[:2, :2] = fixture.initial.covariance

    for time in (1, 2):
        a = 0 if time == 1 else a_second
        b = 0 if time == 1 else b_second
        state_slot = 0 if time == 1 else a + 2 * b
        model_kernel = fixture.model_kernels[time - 1]
        state_kernel = fixture.state_kernels[time - 1]
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
        transform[z_index] = (
            z_slope * transform[z_source] + state_m_slope * transform[m_index]
        )
        transform[z_index, z_index] += 1.0
        mean[z_index] = (
            z_slope * mean[z_source]
            + state_m_slope * mean[m_index]
            + state_kernel.offsets[state_slot]
        )
        noise_covariance[m_index, m_index] = model_kernel.variances[b]
        noise_covariance[z_index, z_index] = state_kernel.variances[state_slot]

    covariance = transform @ noise_covariance @ transform.T
    covariance = 0.5 * (covariance + covariance.T)
    if not bool(np.isfinite(mean).all()) or not bool(np.isfinite(covariance).all()):
        raise ValueError("recognition component must be finite")
    _checked_spd(covariance, "recognition component covariance")
    return _Gaussian(mean, covariance)


def _generative_source_weight(
    fixture: _GenerativeFixture, path: tuple[int, int]
) -> float:
    a, b = path
    factors = (
        fixture.model_priors[0][0],
        fixture.state_priors[0][0],
        fixture.model_priors[1][b],
        fixture.state_priors[1][a],
    )
    weight = math.prod(float(item) for item in factors)
    return _nonnegative(weight, "generative source weight")


def _recognition_source_weight(
    fixture: _RecognitionFixture, path: tuple[int, int]
) -> float:
    a, b = path
    factors = (
        fixture.model_probabilities[0][0],
        fixture.state_probabilities[0][0, 0],
        fixture.model_probabilities[1][b],
        fixture.state_probabilities[1][b, a],
    )
    return _nonnegative(
        math.prod(float(item) for item in factors), "recognition source weight"
    )


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    result = exponentials / np.sum(exponentials, axis=1, keepdims=True)
    if not bool(np.isfinite(result).all()):
        raise ValueError("softmax probabilities must be finite")
    return result


def _component_all_likelihoods(
    component: _Gaussian,
    emissions: tuple[_Emission, _Emission],
    order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.asarray((2, 3, 4, 5), dtype=np.int64)
    mean = component.mean[indices]
    covariance = component.covariance[np.ix_(indices, indices)]
    chol = np.linalg.cholesky(covariance)
    standard, quadrature_weights = _standard_normal_grid(order, 4)
    values = mean + standard @ chol.T
    first_logits = (
        values[:, 0, None] * emissions[0].w_z[None, :]
        + values[:, 1, None] * emissions[0].w_m[None, :]
        + emissions[0].bias[None, :]
    )
    second_logits = (
        values[:, 2, None] * emissions[1].w_z[None, :]
        + values[:, 3, None] * emissions[1].w_m[None, :]
        + emissions[1].bias[None, :]
    )
    return _softmax(first_logits), _softmax(second_logits), quadrature_weights


def _all_evidence_from_components(
    fixture: _GenerativeFixture,
    components: tuple[_Gaussian, ...],
    source_weights: tuple[float, ...],
    order: int,
) -> tuple[tuple[_Reduction, _Reduction, _Reduction], ...]:
    if not components or len(components) != len(source_weights):
        raise ValueError("components and source weights must have equal nonzero length")
    per_path_values: list[np.ndarray] = []
    per_path_absolute: list[np.ndarray] = []
    for source_weight, component in zip(source_weights, components):
        first, second, quadrature_weights = _component_all_likelihoods(
            component, fixture.emissions, order
        )
        values = np.zeros((3, 3), dtype=np.float64)
        absolute = np.zeros((3, 3), dtype=np.float64)
        for first_index, second_index in product(range(3), repeat=2):
            contributions = (
                source_weight
                * quadrature_weights
                * first[:, first_index]
                * second[:, second_index]
            )
            values[first_index, second_index] = math.fsum(
                float(item) for item in contributions
            )
            absolute[first_index, second_index] = math.fsum(
                abs(float(item)) for item in contributions
            )
        per_path_values.append(values)
        per_path_absolute.append(absolute)

    return tuple(
        tuple(
            _Reduction(
                math.fsum(
                    float(values[first_index, second_index])
                    for values in per_path_values
                ),
                math.fsum(
                    float(values[first_index, second_index])
                    for values in per_path_absolute
                ),
            )
            for second_index in range(3)
        )
        for first_index in range(3)
    )


def _all_evidence_source_loop(
    fixture: _GenerativeFixture, order: int
) -> tuple[tuple[_Reduction, _Reduction, _Reduction], ...]:
    """Evidence enumerator whose input is generative data only."""
    weights = tuple(_generative_source_weight(fixture, path) for path in _PATHS)
    if abs(math.fsum(weights) - 1.0) > 64.0 * _EPSILON:
        raise ValueError("generative source weights must sum to one")
    per_path_values: list[np.ndarray] = []
    per_path_absolute: list[np.ndarray] = []
    for path, source_weight in zip(_PATHS, weights):
        component = _assemble_generative_component(fixture, path)
        first, second, quadrature_weights = _component_all_likelihoods(
            component, fixture.emissions, order
        )
        values = np.zeros((3, 3), dtype=np.float64)
        absolute = np.zeros((3, 3), dtype=np.float64)
        for first_index, second_index in product(range(3), repeat=2):
            contributions = (
                source_weight
                * quadrature_weights
                * first[:, first_index]
                * second[:, second_index]
            )
            values[first_index, second_index] = math.fsum(
                float(item) for item in contributions
            )
            absolute[first_index, second_index] = math.fsum(
                abs(float(item)) for item in contributions
            )
        per_path_values.append(values)
        per_path_absolute.append(absolute)
    return tuple(
        tuple(
            _Reduction(
                math.fsum(
                    float(values[first_index, second_index])
                    for values in per_path_values
                ),
                math.fsum(
                    float(values[first_index, second_index])
                    for values in per_path_absolute
                ),
            )
            for second_index in range(3)
        )
        for first_index in range(3)
    )


def _all_evidence_component_table(
    fixture: _GenerativeFixture,
    component_table: tuple[_Gaussian, _Gaussian, _Gaussian, _Gaussian],
    order: int,
) -> tuple[tuple[_Reduction, _Reduction, _Reduction], ...]:
    """Independent enumeration seam over a prebuilt generative table."""
    weights = tuple(_generative_source_weight(fixture, path) for path in _PATHS)
    return _all_evidence_from_components(
        fixture, component_table, weights, order  # type: ignore[arg-type]
    )


def _allowance(reported: _Reduction, check: _Reduction) -> IndependentNumericalAllowance:
    return IndependentNumericalAllowance(
        convergence_estimate=abs(reported.value - check.value),
        rounding_allowance=32.0 * _EPSILON * reported.absolute_sum,
    )


def _evidence_record(
    observation_labels: tuple[int, int], reported: _Reduction, check: _Reduction
) -> H1EvidenceRecord:
    probability = _positive(reported.value, "evidence probability")
    check_probability = _positive(check.value, "check evidence probability")
    log_probability = math.log(probability)
    check_log_probability = math.log(check_probability)
    probability_allowance = _allowance(reported, check)
    log_convergence = abs(log_probability - check_log_probability)
    probability_rounding = 32.0 * _EPSILON * reported.absolute_sum
    propagated_probability_rounding = probability_rounding / probability
    intrinsic_log_rounding = 32.0 * _EPSILON * max(1.0, abs(log_probability))
    log_allowance = IndependentNumericalAllowance(
        convergence_estimate=log_convergence,
        rounding_allowance=propagated_probability_rounding + intrinsic_log_rounding,
    )
    return H1EvidenceRecord(
        observation_labels,
        probability,
        log_probability,
        probability_allowance,
        log_allowance,
    )


def h1_log_evidence(
    fixture_path: Path,
    observation_labels: tuple[int, int],
    *,
    quadrature_order: int,
    convergence_check_order: int,
) -> H1EvidenceRecord:
    """Evaluate one observation evidence from raw generative fixture data only."""
    _validate_orders(quadrature_order, convergence_check_order)
    labels = _observation_pair(observation_labels)
    generative = _load_generative_fixture(fixture_path)
    reported_table = _all_evidence_source_loop(generative, quadrature_order)
    check_table = _all_evidence_source_loop(generative, convergence_check_order)
    first, second = (_label_to_index(label) for label in labels)
    return _evidence_record(
        labels,
        reported_table[first][second],
        check_table[first][second],
    )


def h1_all_observation_evidences(
    fixture_path: Path,
    *,
    quadrature_order: int,
    convergence_check_order: int,
) -> tuple[
    H1EvidenceRecord,
    H1EvidenceRecord,
    H1EvidenceRecord,
    H1EvidenceRecord,
    H1EvidenceRecord,
    H1EvidenceRecord,
    H1EvidenceRecord,
    H1EvidenceRecord,
    H1EvidenceRecord,
]:
    """Evaluate all nine observation pairs in one vectorized component pass."""
    _validate_orders(quadrature_order, convergence_check_order)
    generative = _load_generative_fixture(fixture_path)
    reported = _all_evidence_source_loop(generative, quadrature_order)
    check = _all_evidence_source_loop(generative, convergence_check_order)
    records = tuple(
        _evidence_record(
            (first + 1, second + 1),
            reported[first][second],
            check[first][second],
        )
        for first, second in product(range(3), repeat=2)
    )
    normalization_residual = abs(
        math.fsum(record.probability for record in records) - 1.0
    )
    normalization_allowance = math.fsum(
        record.probability_allowance.total for record in records
    )
    if normalization_residual > normalization_allowance:
        raise ValueError("all-pairs evidence table failed probability normalization")
    return records  # type: ignore[return-value]


def h1_evidence_enumeration_pair(
    fixture_path: Path,
    observation_labels: tuple[int, int],
    *,
    quadrature_order: int,
) -> tuple[float, float]:
    """Return direct-loop and prebuilt-table evidence for a crosscheck."""
    order = _validate_single_order(quadrature_order)
    labels = _observation_pair(observation_labels)
    generative = _load_generative_fixture(fixture_path)
    direct = _all_evidence_source_loop(generative, order)
    component_table = tuple(
        _assemble_generative_component(generative, path) for path in _PATHS
    )
    table = _all_evidence_component_table(
        generative, component_table, order  # type: ignore[arg-type]
    )
    first, second = (_label_to_index(label) for label in labels)
    return direct[first][second].value, table[first][second].value


def h1_wrong_recognition_mixture_evidence(
    fixture_path: Path,
    observation_labels: tuple[int, int],
    *,
    quadrature_order: int,
    convergence_check_order: int,
) -> H1EvidenceRecord:
    """Failure injection: use recognition components and weights as evidence."""
    _validate_orders(quadrature_order, convergence_check_order)
    labels = _observation_pair(observation_labels)
    complete = _load_complete_fixture(fixture_path)
    weighted_paths = tuple(
        (_recognition_source_weight(complete.recognition, path), path)
        for path in _PATHS
    )
    active = tuple((weight, path) for weight, path in weighted_paths if weight > 0.0)
    weights = tuple(weight for weight, _ in active)
    components = tuple(
        _assemble_recognition_component(complete.recognition, path)
        for _, path in active
    )
    reported = _all_evidence_from_components(
        complete.generative,
        components,  # type: ignore[arg-type]
        weights,  # type: ignore[arg-type]
        quadrature_order,
    )
    check = _all_evidence_from_components(
        complete.generative,
        components,  # type: ignore[arg-type]
        weights,  # type: ignore[arg-type]
        convergence_check_order,
    )
    first, second = (_label_to_index(label) for label in labels)
    return _evidence_record(labels, reported[first][second], check[first][second])


def _mixed_evidence_record(
    complete: _CompleteFixture,
    labels: tuple[int, int],
    components: tuple[_Gaussian, ...],
    weights: tuple[float, ...],
    quadrature_order: int,
    convergence_check_order: int,
) -> H1EvidenceRecord:
    reported = _all_evidence_from_components(
        complete.generative, components, weights, quadrature_order
    )
    check = _all_evidence_from_components(
        complete.generative, components, weights, convergence_check_order
    )
    first, second = (_label_to_index(label) for label in labels)
    return _evidence_record(labels, reported[first][second], check[first][second])


def h1_q_weights_p_components_evidence(
    fixture_path: Path,
    observation_labels: tuple[int, int],
    *,
    quadrature_order: int,
    convergence_check_order: int,
) -> H1EvidenceRecord:
    """Failure injection: recognition weights with generative components."""
    _validate_orders(quadrature_order, convergence_check_order)
    labels = _observation_pair(observation_labels)
    complete = _load_complete_fixture(fixture_path)
    weighted_paths = tuple(
        (_recognition_source_weight(complete.recognition, path), path)
        for path in _PATHS
    )
    active = tuple((weight, path) for weight, path in weighted_paths if weight > 0.0)
    weights = tuple(weight for weight, _ in active)
    components = tuple(
        _assemble_generative_component(complete.generative, path)
        for _, path in active
    )
    return _mixed_evidence_record(
        complete,
        labels,
        components,  # type: ignore[arg-type]
        weights,  # type: ignore[arg-type]
        quadrature_order,
        convergence_check_order,
    )


def h1_p_weights_q_components_evidence(
    fixture_path: Path,
    observation_labels: tuple[int, int],
    *,
    quadrature_order: int,
    convergence_check_order: int,
) -> H1EvidenceRecord:
    """Failure injection: generative weights with recognition components."""
    _validate_orders(quadrature_order, convergence_check_order)
    labels = _observation_pair(observation_labels)
    complete = _load_complete_fixture(fixture_path)
    components = tuple(
        _assemble_recognition_component(complete.recognition, path) for path in _PATHS
    )
    weights = tuple(
        _generative_source_weight(complete.generative, path) for path in _PATHS
    )
    return _mixed_evidence_record(
        complete,
        labels,
        components,  # type: ignore[arg-type]
        weights,  # type: ignore[arg-type]
        quadrature_order,
        convergence_check_order,
    )


def _permuted_component(component: _Gaussian) -> _Gaussian:
    permutation = np.asarray((1, 0, 3, 2, 5, 4), dtype=np.int64)
    return _Gaussian(
        component.mean[permutation],
        component.covariance[np.ix_(permutation, permutation)],
    )


def h1_permuted_zm_evidence(
    fixture_path: Path,
    observation_labels: tuple[int, int],
    *,
    quadrature_order: int,
    convergence_check_order: int,
) -> H1EvidenceRecord:
    """Failure injection: interchange every z/m coordinate in p components."""
    _validate_orders(quadrature_order, convergence_check_order)
    labels = _observation_pair(observation_labels)
    generative = _load_generative_fixture(fixture_path)
    components = tuple(
        _permuted_component(_assemble_generative_component(generative, path))
        for path in _PATHS
    )
    weights = tuple(_generative_source_weight(generative, path) for path in _PATHS)
    reported = _all_evidence_from_components(
        generative,
        components,  # type: ignore[arg-type]
        weights,  # type: ignore[arg-type]
        quadrature_order,
    )
    check = _all_evidence_from_components(
        generative,
        components,  # type: ignore[arg-type]
        weights,  # type: ignore[arg-type]
        convergence_check_order,
    )
    first, second = (_label_to_index(label) for label in labels)
    return _evidence_record(labels, reported[first][second], check[first][second])


def _expected_emission_probability_component(
    component: _Gaussian,
    emission: _Emission,
    *,
    time: int,
    order: int,
) -> tuple[_Reduction, _Reduction, _Reduction]:
    indices = np.asarray((2 * time, 2 * time + 1), dtype=np.int64)
    mean = component.mean[indices]
    covariance = component.covariance[np.ix_(indices, indices)]
    chol = _checked_spd(covariance, "emission marginal covariance")
    standard, weights = _standard_normal_grid(order, 2)
    values = mean + standard @ chol.T
    logits = (
        values[:, 0, None] * emission.w_z[None, :]
        + values[:, 1, None] * emission.w_m[None, :]
        + emission.bias[None, :]
    )
    probabilities = _softmax(logits)
    return tuple(
        _Reduction(
            math.fsum(float(item) for item in weights * probabilities[:, index]),
            math.fsum(abs(float(item)) for item in weights * probabilities[:, index]),
        )
        for index in range(3)
    )  # type: ignore[return-value]


def _factorized_time_evidence_table(
    generative: _GenerativeFixture, order: int
) -> tuple[tuple[_Reduction, _Reduction, _Reduction], ...]:
    component_values: list[np.ndarray] = []
    for path in _PATHS:
        weight = _generative_source_weight(generative, path)
        component = _assemble_generative_component(generative, path)
        first = _expected_emission_probability_component(
            component, generative.emissions[0], time=1, order=order
        )
        second = _expected_emission_probability_component(
            component, generative.emissions[1], time=2, order=order
        )
        component_values.append(
            weight
            * np.outer(
                np.asarray([item.value for item in first], dtype=np.float64),
                np.asarray([item.value for item in second], dtype=np.float64),
            )
        )
    return tuple(
        tuple(
            _Reduction(
                math.fsum(
                    float(values[first_index, second_index])
                    for values in component_values
                ),
                math.fsum(
                    abs(float(values[first_index, second_index]))
                    for values in component_values
                ),
            )
            for second_index in range(3)
        )
        for first_index in range(3)
    )


def h1_factorized_time_evidence(
    fixture_path: Path,
    observation_labels: tuple[int, int],
    *,
    quadrature_order: int,
    convergence_check_order: int,
) -> H1EvidenceRecord:
    """Failure injection: factorize E[L1 L2] within each source path."""
    _validate_orders(quadrature_order, convergence_check_order)
    labels = _observation_pair(observation_labels)
    generative = _load_generative_fixture(fixture_path)
    reported = _factorized_time_evidence_table(generative, quadrature_order)
    check = _factorized_time_evidence_table(generative, convergence_check_order)
    first, second = (_label_to_index(label) for label in labels)
    return _evidence_record(labels, reported[first][second], check[first][second])


def _expected_log_emission_component(
    component: _Gaussian,
    emission: _Emission,
    *,
    time: int,
    selected_index: int,
    order: int,
) -> _Reduction:
    indices = np.asarray((2 * time, 2 * time + 1), dtype=np.int64)
    mean = component.mean[indices]
    covariance = component.covariance[np.ix_(indices, indices)]
    chol = np.linalg.cholesky(covariance)
    standard, weights = _standard_normal_grid(order, 2)
    values = mean + standard @ chol.T
    logits = (
        values[:, 0, None] * emission.w_z[None, :]
        + values[:, 1, None] * emission.w_m[None, :]
        + emission.bias[None, :]
    )
    maximum = np.max(logits, axis=1)
    log_normalizer = maximum + np.log(
        np.sum(np.exp(logits - maximum[:, None]), axis=1)
    )
    selected = logits[:, selected_index] - log_normalizer
    contributions = weights * selected
    return _Reduction(
        math.fsum(float(item) for item in contributions),
        math.fsum(abs(float(item)) for item in contributions),
    )


def _gaussian_kl(q: _Gaussian, p: _Gaussian) -> float:
    if q.mean.shape != p.mean.shape:
        raise ValueError("Gaussian dimensions must agree")
    dimension = q.mean.size
    q_chol = _checked_spd(q.covariance, "q covariance")
    p_chol = _checked_spd(p.covariance, "p covariance")
    solved_covariance = np.linalg.solve(
        p_chol.T, np.linalg.solve(p_chol, q.covariance)
    )
    displacement = q.mean - p.mean
    solved_displacement = np.linalg.solve(
        p_chol.T, np.linalg.solve(p_chol, displacement)
    )
    if not bool(np.isfinite(solved_covariance).all()) or not bool(
        np.isfinite(solved_displacement).all()
    ):
        raise ValueError("Gaussian KL solves must be finite")
    q_sign, q_log_determinant = np.linalg.slogdet(q.covariance)
    p_sign, p_log_determinant = np.linalg.slogdet(p.covariance)
    if q_sign != 1.0 or p_sign != 1.0:
        raise ValueError("Gaussian covariances must have positive determinant")
    value = 0.5 * (
        float(np.trace(solved_covariance))
        + float(displacement @ solved_displacement)
        - dimension
        + float(p_log_determinant)
        - float(q_log_determinant)
    )
    return _nonnegative(value, "Gaussian KL")


def _identity_order(
    complete: _CompleteFixture,
    observation_labels: tuple[int, int],
    order: int,
) -> _IdentityEvaluation:
    first_index, second_index = (
        _label_to_index(label) for label in observation_labels
    )
    evidence_table = _all_evidence_source_loop(complete.generative, order)
    evidence = evidence_table[first_index][second_index]
    log_evidence = math.log(_positive(evidence.value, "identity evidence"))
    log_reduction = _Reduction(log_evidence, max(1.0, abs(log_evidence)))

    posterior_contributions: list[float] = []
    posterior_absolute: list[float] = []
    recognition_weights = tuple(
        _recognition_source_weight(complete.recognition, path) for path in _PATHS
    )
    if abs(math.fsum(recognition_weights) - 1.0) > 64.0 * _EPSILON:
        raise ValueError("recognition source weights must sum to one")
    for path, q_weight in zip(_PATHS, recognition_weights):
        if q_weight == 0.0:
            continue
        p_weight = _positive(
            _generative_source_weight(complete.generative, path),
            "generative source weight on recognition support",
        )
        q_component = _assemble_recognition_component(complete.recognition, path)
        p_component = _assemble_generative_component(complete.generative, path)
        source_log_ratio = math.log(q_weight) - math.log(p_weight)
        continuous_kl = _gaussian_kl(q_component, p_component)
        first_emission = _expected_log_emission_component(
            q_component,
            complete.generative.emissions[0],
            time=1,
            selected_index=first_index,
            order=order,
        )
        second_emission = _expected_log_emission_component(
            q_component,
            complete.generative.emissions[1],
            time=2,
            selected_index=second_index,
            order=order,
        )
        path_terms = (
            source_log_ratio,
            continuous_kl,
            -first_emission.value,
            -second_emission.value,
            log_evidence,
        )
        posterior_contributions.append(q_weight * math.fsum(path_terms))
        posterior_absolute.extend(
            (
                abs(q_weight * source_log_ratio),
                abs(q_weight * continuous_kl),
                q_weight * first_emission.absolute_sum,
                q_weight * second_emission.absolute_sum,
                abs(q_weight * log_evidence),
            )
        )

    posterior = _Reduction(
        math.fsum(posterior_contributions),
        math.fsum(posterior_absolute),
    )
    elbo_value = math.fsum((log_evidence, -posterior.value))
    elbo = _Reduction(elbo_value, abs(log_evidence) + posterior.absolute_sum)
    return _IdentityEvaluation(evidence, log_reduction, posterior, elbo)


def h1_evidence_and_posterior_kl(
    fixture_path: Path,
    *,
    quadrature_order: int,
    convergence_check_order: int,
) -> H1IdentityRecord:
    """Evaluate ``log p(x) - KL(Q || p(.|x))`` independently."""
    _validate_orders(quadrature_order, convergence_check_order)
    complete = _load_complete_fixture(fixture_path)
    reported = _identity_order(
        complete, complete.generative.observations, quadrature_order
    )
    check = _identity_order(
        complete, complete.generative.observations, convergence_check_order
    )
    evidence = _evidence_record(
        complete.generative.observations, reported.evidence, check.evidence
    )
    return H1IdentityRecord(
        evidence=evidence,
        posterior_kl=reported.posterior_kl.value,
        elbo_from_identity=reported.elbo.value,
        quadrature_order=21,
        convergence_check_order=17,
        posterior_kl_allowance=_allowance(
            reported.posterior_kl, check.posterior_kl
        ),
        identity_allowance=_allowance(reported.elbo, check.elbo),
    )


def _conditional_gaussian_kl(
    q_variance: float, p_variance: float, mean_square: float
) -> float:
    q_checked = _positive(q_variance, "q conditional variance")
    p_checked = _positive(p_variance, "p conditional variance")
    mean_checked = _nonnegative(mean_square, "conditional mean square")
    value = 0.5 * (
        math.log(p_checked / q_checked)
        + (q_checked + mean_checked) / p_checked
        - 1.0
    )
    return _nonnegative(value, "conditional Gaussian KL")


def _initial_state_conditional(gaussian: _Gaussian) -> tuple[float, float, float]:
    model_variance = _positive(gaussian.covariance[1, 1], "initial model variance")
    slope = float(gaussian.covariance[0, 1]) / model_variance
    offset = float(gaussian.mean[0]) - slope * float(gaussian.mean[1])
    variance = float(gaussian.covariance[0, 0]) - (
        float(gaussian.covariance[0, 1]) ** 2 / model_variance
    )
    return (
        _finite(slope, "initial conditional slope"),
        _finite(offset, "initial conditional offset"),
        _positive(variance, "initial conditional variance"),
    )


def _initial_kls(
    q_initial: _Gaussian, p_initial: _Gaussian
) -> tuple[_Reduction, _Reduction]:
    model_kl = _conditional_gaussian_kl(
        float(q_initial.covariance[1, 1]),
        float(p_initial.covariance[1, 1]),
        (float(q_initial.mean[1]) - float(p_initial.mean[1])) ** 2,
    )
    q_slope, q_offset, q_variance = _initial_state_conditional(q_initial)
    p_slope, p_offset, p_variance = _initial_state_conditional(p_initial)
    slope_difference = q_slope - p_slope
    offset_difference = q_offset - p_offset
    mean_square = (
        slope_difference * slope_difference * float(q_initial.covariance[1, 1])
        + (slope_difference * float(q_initial.mean[1]) + offset_difference) ** 2
    )
    state_kl = _conditional_gaussian_kl(q_variance, p_variance, mean_square)
    return _Reduction(model_kl, abs(model_kl)), _Reduction(state_kl, abs(state_kl))


def _categorical_kl(q: np.ndarray, p: np.ndarray) -> _Reduction:
    if q.shape != p.shape:
        raise ValueError("categorical vectors must have equal shape")
    contributions: list[float] = []
    for q_value, p_value in zip(q, p):
        q_float = _nonnegative(float(q_value), "q source probability")
        p_float = _nonnegative(float(p_value), "p source probability")
        if q_float > 0.0:
            if p_float <= 0.0:
                raise ValueError(
                    "recognition source mass lies outside positive generative support"
                )
            contributions.append(q_float * (math.log(q_float) - math.log(p_float)))
    value = math.fsum(contributions)
    return _Reduction(
        _nonnegative(value, "categorical KL"),
        math.fsum(abs(item) for item in contributions),
    )


def _state_source_kl(
    q_b: np.ndarray, q_a_given_b: np.ndarray, p_a: np.ndarray
) -> _Reduction:
    contributions: list[float] = []
    for b in range(q_b.size):
        for a in range(p_a.size):
            q_b_value = float(q_b[b])
            q_a_value = float(q_a_given_b[b, a])
            p_a_value = float(p_a[a])
            if q_b_value > 0.0 and q_a_value > 0.0:
                if p_a_value <= 0.0:
                    raise ValueError(
                        "recognition state-source mass lies outside positive generative support"
                    )
                contributions.append(
                    q_b_value
                    * q_a_value
                    * (math.log(q_a_value) - math.log(p_a_value))
                )
    value = math.fsum(contributions)
    return _Reduction(
        _nonnegative(value, "state-source KL"),
        math.fsum(abs(item) for item in contributions),
    )


def _mixture_expected_log_emission(
    complete: _CompleteFixture,
    components: tuple[_Gaussian, ...],
    weights: tuple[float, ...],
    *,
    time: int,
    selected_index: int,
    order: int,
) -> _Reduction:
    values: list[float] = []
    absolute: list[float] = []
    for weight, component in zip(weights, components):
        reduction = _expected_log_emission_component(
            component,
            complete.generative.emissions[time - 1],
            time=time,
            selected_index=selected_index,
            order=order,
        )
        values.append(weight * reduction.value)
        absolute.append(weight * reduction.absolute_sum)
    return _Reduction(math.fsum(values), math.fsum(absolute))


def _model_transition_kl(
    complete: _CompleteFixture,
    components: tuple[_Gaussian, ...],
    weights: tuple[float, ...],
    paths: tuple[tuple[int, int], ...],
    time: int,
) -> _Reduction:
    contributions: list[float] = []
    for weight, path, component in zip(weights, paths, components):
        _, b_second = path
        b = 0 if time == 1 else b_second
        parent_index = 2 * b + 1
        q_kernel = complete.recognition.model_kernels[time - 1]
        q_slope = float(q_kernel.slopes[b])
        p_slope = float(
            complete.generative.frames[time] / complete.generative.frames[b]
        )
        slope_difference = q_slope - p_slope
        offset_difference = (
            float(q_kernel.offsets[b])
            - float(complete.generative.model_offsets[time - 1])
        )
        parent_mean = float(component.mean[parent_index])
        parent_variance = float(component.covariance[parent_index, parent_index])
        mean_square = (
            slope_difference * slope_difference * parent_variance
            + (slope_difference * parent_mean + offset_difference) ** 2
        )
        kl = _conditional_gaussian_kl(
            float(q_kernel.variances[b]),
            float(complete.generative.model_variances[time - 1]),
            mean_square,
        )
        contributions.append(weight * kl)
    return _Reduction(math.fsum(contributions), math.fsum(abs(item) for item in contributions))


def _state_transition_kl(
    complete: _CompleteFixture,
    components: tuple[_Gaussian, ...],
    weights: tuple[float, ...],
    paths: tuple[tuple[int, int], ...],
    time: int,
) -> _Reduction:
    contributions: list[float] = []
    for weight, path, component in zip(weights, paths, components):
        a_second, b_second = path
        a = 0 if time == 1 else a_second
        b = 0 if time == 1 else b_second
        slot = 0 if time == 1 else a + 2 * b
        q_kernel = complete.recognition.state_kernels[time - 1]
        p_state_slope = float(
            complete.generative.frames[time] / complete.generative.frames[a]
        )
        slope_difference = np.asarray(
            (
                float(q_kernel.z_slopes[slot]) - p_state_slope,
                float(q_kernel.m_slopes[slot])
                - float(complete.generative.state_model_slopes[time - 1]),
            ),
            dtype=np.float64,
        )
        offset_difference = (
            float(q_kernel.offsets[slot])
            - float(complete.generative.state_offsets[time - 1])
        )
        indices = np.asarray((2 * a, 2 * time + 1), dtype=np.int64)
        parent_mean = component.mean[indices]
        parent_covariance = component.covariance[np.ix_(indices, indices)]
        centered_mean = float(slope_difference @ parent_mean) + offset_difference
        mean_square = (
            float(slope_difference @ parent_covariance @ slope_difference)
            + centered_mean * centered_mean
        )
        kl = _conditional_gaussian_kl(
            float(q_kernel.variances[slot]),
            float(complete.generative.state_variances[time - 1]),
            mean_square,
        )
        contributions.append(weight * kl)
    return _Reduction(math.fsum(contributions), math.fsum(abs(item) for item in contributions))


def _recognition_entropy(
    recognition: _RecognitionFixture,
    weights: tuple[float, ...],
    paths: tuple[tuple[int, int], ...],
) -> _Reduction:
    contributions = [
        -weight * math.log(weight)
        for weight in weights
        if weight > 0.0
    ]
    sign, initial_log_determinant = np.linalg.slogdet(recognition.initial.covariance)
    if sign != 1.0:
        raise ValueError("recognition initial covariance must have positive determinant")
    contributions.append(
        0.5 * (2.0 * (1.0 + math.log(2.0 * math.pi)) + initial_log_determinant)
    )
    for weight, path in zip(weights, paths):
        if weight == 0.0:
            continue
        a_second, b_second = path
        for time in (1, 2):
            a = 0 if time == 1 else a_second
            b = 0 if time == 1 else b_second
            slot = 0 if time == 1 else a + 2 * b
            model_variance = float(recognition.model_kernels[time - 1].variances[b])
            state_variance = float(recognition.state_kernels[time - 1].variances[slot])
            contributions.append(
                weight * 0.5 * math.log(2.0 * math.pi * math.e * model_variance)
            )
            contributions.append(
                weight * 0.5 * math.log(2.0 * math.pi * math.e * state_variance)
            )
    return _Reduction(
        math.fsum(contributions), math.fsum(abs(item) for item in contributions)
    )


def _local_order(complete: _CompleteFixture, order: int) -> _TermEvaluation:
    all_weights = tuple(
        _recognition_source_weight(complete.recognition, path) for path in _PATHS
    )
    if abs(math.fsum(all_weights) - 1.0) > 64.0 * _EPSILON:
        raise ValueError("recognition source weights must sum to one")
    active = tuple(
        (weight, path)
        for weight, path in zip(all_weights, _PATHS)
        if weight > 0.0
    )
    weights = tuple(weight for weight, _ in active)
    paths = tuple(path for _, path in active)
    components = tuple(
        _assemble_recognition_component(complete.recognition, path) for path in paths
    )
    first_index, second_index = (
        _label_to_index(label) for label in complete.generative.observations
    )
    emissions = (
        _mixture_expected_log_emission(
            complete,
            components,  # type: ignore[arg-type]
            weights,  # type: ignore[arg-type]
            time=1,
            selected_index=first_index,
            order=order,
        ),
        _mixture_expected_log_emission(
            complete,
            components,  # type: ignore[arg-type]
            weights,  # type: ignore[arg-type]
            time=2,
            selected_index=second_index,
            order=order,
        ),
    )
    initial_model, initial_state = _initial_kls(
        complete.recognition.initial, complete.generative.initial
    )
    model_source = tuple(
        _categorical_kl(
            complete.recognition.model_probabilities[time],
            complete.generative.model_priors[time],
        )
        for time in range(2)
    )
    state_source = tuple(
        _state_source_kl(
            complete.recognition.model_probabilities[time],
            complete.recognition.state_probabilities[time],
            complete.generative.state_priors[time],
        )
        for time in range(2)
    )
    model_transition = tuple(
        _model_transition_kl(
            complete,
            components,  # type: ignore[arg-type]
            weights,  # type: ignore[arg-type]
            paths,
            time,
        )
        for time in (1, 2)
    )
    state_transition = tuple(
        _state_transition_kl(
            complete,
            components,  # type: ignore[arg-type]
            weights,  # type: ignore[arg-type]
            paths,
            time,
        )
        for time in (1, 2)
    )
    entropy = _recognition_entropy(
        complete.recognition, weights, paths
    )
    objective_terms = (
        *emissions,
        initial_model,
        initial_state,
        *model_source,
        *model_transition,
        *state_source,
        *state_transition,
    )
    complete_value = math.fsum(
        (
            emissions[0].value,
            emissions[1].value,
            -initial_model.value,
            -initial_state.value,
            -model_source[0].value,
            -model_transition[0].value,
            -model_source[1].value,
            -model_transition[1].value,
            -state_source[0].value,
            -state_transition[0].value,
            -state_source[1].value,
            -state_transition[1].value,
        )
    )
    complete_reduction = _Reduction(
        complete_value, math.fsum(item.absolute_sum for item in objective_terms)
    )
    return _TermEvaluation(
        emissions,
        initial_model,
        initial_state,
        model_source,  # type: ignore[arg-type]
        model_transition,  # type: ignore[arg-type]
        state_source,  # type: ignore[arg-type]
        state_transition,  # type: ignore[arg-type]
        entropy,
        complete_reduction,
    )


def h1_local_diagnostics(
    fixture_path: Path,
    *,
    quadrature_order: int,
    convergence_check_order: int,
) -> IndependentTermRecord:
    """Independently evaluate all local H1 terms and term-shaped allowances."""
    _validate_orders(quadrature_order, convergence_check_order)
    complete = _load_complete_fixture(fixture_path)
    reported = _local_order(complete, quadrature_order)
    check = _local_order(complete, convergence_check_order)
    allowances = IndependentTermAllowances(
        expected_log_emission=tuple(
            _allowance(reported.expected_log_emission[index], check.expected_log_emission[index])
            for index in range(2)
        ),  # type: ignore[arg-type]
        initial_model_kl=_allowance(reported.initial_model_kl, check.initial_model_kl),
        initial_state_kl=_allowance(reported.initial_state_kl, check.initial_state_kl),
        model_source_kl=tuple(
            _allowance(reported.model_source_kl[index], check.model_source_kl[index])
            for index in range(2)
        ),  # type: ignore[arg-type]
        model_transition_kl=tuple(
            _allowance(reported.model_transition_kl[index], check.model_transition_kl[index])
            for index in range(2)
        ),  # type: ignore[arg-type]
        state_source_kl=tuple(
            _allowance(reported.state_source_kl[index], check.state_source_kl[index])
            for index in range(2)
        ),  # type: ignore[arg-type]
        state_transition_kl=tuple(
            _allowance(reported.state_transition_kl[index], check.state_transition_kl[index])
            for index in range(2)
        ),  # type: ignore[arg-type]
        joint_recognition_entropy=_allowance(
            reported.joint_recognition_entropy, check.joint_recognition_entropy
        ),
        complete_elbo=_allowance(reported.complete_elbo, check.complete_elbo),
    )
    return IndependentTermRecord(
        expected_log_emission=tuple(
            item.value for item in reported.expected_log_emission
        ),  # type: ignore[arg-type]
        initial_model_kl=reported.initial_model_kl.value,
        initial_state_kl=reported.initial_state_kl.value,
        model_source_kl=tuple(
            item.value for item in reported.model_source_kl
        ),  # type: ignore[arg-type]
        model_transition_kl=tuple(
            item.value for item in reported.model_transition_kl
        ),  # type: ignore[arg-type]
        state_source_kl=tuple(
            item.value for item in reported.state_source_kl
        ),  # type: ignore[arg-type]
        state_transition_kl=tuple(
            item.value for item in reported.state_transition_kl
        ),  # type: ignore[arg-type]
        joint_recognition_entropy=reported.joint_recognition_entropy.value,
        complete_elbo=reported.complete_elbo.value,
        allowances=allowances,
    )
