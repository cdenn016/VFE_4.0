"""Strict data-only loader for the frozen h1-v1 fixture."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from vfe4.types.h1 import (
    EmissionRecord,
    GaussianLaw,
    H1Fixture,
    ModelTransitionRecord,
    RecognitionModelKernelRecord,
    RecognitionParameterRecord,
    RecognitionStateKernelRecord,
    StateTransitionRecord,
)
from vfe4.types.structural import PopulationFrames, SourcePath, StructuralData


_ROOT_FIELDS = {
    "fixture_schema_version", "fixture_id", "continuous_order", "vocabulary_labels",
    "observation_label_base", "observation_labels", "frames", "initial_joint",
    "model_source_priors", "state_source_priors", "model_offsets", "model_variances",
    "state_offsets", "state_variances", "state_model_slopes", "decoder", "recognition",
    "quadrature",
}


def _fields(value: object, expected: set[str], name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"{name} fields must equal {sorted(expected)}")
    return value


def _sequence(value: object, size: int, name: str) -> list[Any]:
    if type(value) is not list or len(value) != size:
        raise ValueError(f"{name} must be a list of length {size}")
    return value


def _number(value: object, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite numeric data")
    return float(value)


def _tensor_vector(value: object, size: int, name: str) -> torch.Tensor:
    return torch.tensor([_number(item, f"{name}[{i}]") for i, item in enumerate(_sequence(value, size, name))], dtype=torch.float64)


def _tensor_matrix(value: object, rows: int, columns: int, name: str) -> torch.Tensor:
    outer = _sequence(value, rows, name)
    return torch.tensor([
        [_number(item, f"{name}[{i}][{j}]") for j, item in enumerate(_sequence(row, columns, f"{name}[{i}]"))]
        for i, row in enumerate(outer)
    ], dtype=torch.float64)


def _positive_vector(value: object, size: int, name: str) -> torch.Tensor:
    result = _tensor_vector(value, size, name)
    if bool(torch.any(result <= 0)):
        raise ValueError(f"{name} variance values must be positive")
    return result


def label_to_index(label: int, *, vocabulary_size: int = 3) -> int:
    if type(vocabulary_size) is not int or vocabulary_size <= 0:
        raise ValueError("vocabulary_size must be a positive integer")
    if type(label) is not int or label < 1 or label > vocabulary_size:
        raise ValueError(f"label must be in [1, {vocabulary_size}]")
    return label - 1


def enumerate_source_paths(fixture: H1Fixture) -> tuple[SourcePath, ...]:
    if not isinstance(fixture, H1Fixture):
        raise ValueError("fixture must be an H1Fixture")
    return tuple(
        SourcePath(a=(0, a_2), b=(0, b_2))
        for b_2 in fixture.structural.model_source_support[1]
        for a_2 in fixture.structural.state_source_support[1]
    )

def load_h1_fixture(path: Path | None = None) -> H1Fixture:
    fixture_path = path if path is not None else Path(__file__).with_name("fixtures") / "h1_v1.json"
    if not isinstance(fixture_path, Path):
        raise ValueError("path must be a pathlib.Path")
    try:
        raw = json.loads(fixture_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"fixture JSON could not be loaded: {exc}") from exc
    root = _fields(raw, _ROOT_FIELDS, "fixture")
    if type(root["fixture_schema_version"]) is not int or root["fixture_schema_version"] != 1:
        raise ValueError("fixture_schema_version must equal 1")
    if root["fixture_id"] != "h1-v1":
        raise ValueError("fixture_id must equal h1-v1")
    continuous_order = tuple(_sequence(root["continuous_order"], 6, "continuous_order"))
    if continuous_order != ("z0", "m0", "z1", "m1", "z2", "m2"):
        raise ValueError("continuous_order must match h1-v1")
    if root["vocabulary_labels"] != [1, 2, 3] or root["observation_label_base"] != 1:
        raise ValueError("vocabulary labels and label base must match h1-v1")
    observation_data = _sequence(root["observation_labels"], 2, "observation_labels")
    observation_labels = tuple(int(item) if type(item) is int else -1 for item in observation_data)
    for label in observation_labels:
        label_to_index(label)

    structure = StructuralData(
        horizon=2, d_z=1, d_m=1, vocabulary_size=3,
        state_parent_sets=((0,), (0, 1)), model_parent_sets=((0,), (0, 1)),
        state_source_support=((0,), (0, 1)), model_source_support=((0,), (0, 1)),
    )
    frames = PopulationFrames(_tensor_vector(root["frames"], 3, "frames"))
    initial_raw = _fields(root["initial_joint"], {"mean", "covariance"}, "initial_joint")
    initial_joint = GaussianLaw(
        _tensor_vector(initial_raw["mean"], 2, "initial_joint.mean"),
        _tensor_matrix(initial_raw["covariance"], 2, 2, "initial_joint.covariance"),
    )
    model_priors = (
        _tensor_vector(_sequence(root["model_source_priors"], 2, "model_source_priors")[0], 1, "model_source_priors[0]"),
        _tensor_vector(_sequence(root["model_source_priors"], 2, "model_source_priors")[1], 2, "model_source_priors[1]"),
    )
    state_priors = (
        _tensor_vector(_sequence(root["state_source_priors"], 2, "state_source_priors")[0], 1, "state_source_priors[0]"),
        _tensor_vector(_sequence(root["state_source_priors"], 2, "state_source_priors")[1], 2, "state_source_priors[1]"),
    )
    model_offsets = _tensor_vector(root["model_offsets"], 2, "model_offsets")
    model_variances = _positive_vector(root["model_variances"], 2, "model_variances")
    state_offsets = _tensor_vector(root["state_offsets"], 2, "state_offsets")
    state_variances = _positive_vector(root["state_variances"], 2, "state_variances")
    state_model_slopes = _tensor_vector(root["state_model_slopes"], 2, "state_model_slopes")
    supports = ((0,), (0, 1))
    model_transitions = tuple(
        ModelTransitionRecord(
            supports[t - 1],
            torch.stack(tuple(frames.omega(t, source) for source in supports[t - 1])),
            model_offsets[t - 1], model_variances[t - 1],
        ) for t in (1, 2)
    )
    state_transitions = tuple(
        StateTransitionRecord(
            supports[t - 1],
            torch.stack(tuple(frames.omega(t, source) for source in supports[t - 1])),
            state_model_slopes[t - 1], state_offsets[t - 1], state_variances[t - 1],
        ) for t in (1, 2)
    )
    decoder_data = _sequence(root["decoder"], 2, "decoder")
    emissions = tuple(
        EmissionRecord(
            _tensor_vector(record["w_z"], 3, f"decoder[{time}].w_z"),
            _tensor_vector(record["w_m"], 3, f"decoder[{time}].w_m"),
            _tensor_vector(record["bias"], 3, f"decoder[{time}].bias"),
        )
        for time, value in enumerate(decoder_data)
        for record in (_fields(value, {"w_z", "w_m", "bias"}, f"decoder[{time}]"),)
    )

    recognition_raw = _fields(root["recognition"], {
        "initial_mean", "initial_covariance", "model_source_probabilities",
        "state_source_probabilities_given_model_source", "model_kernels", "state_kernels",
    }, "recognition")
    recognition_initial = GaussianLaw(
        _tensor_vector(recognition_raw["initial_mean"], 2, "recognition.initial_mean"),
        _tensor_matrix(recognition_raw["initial_covariance"], 2, 2, "recognition.initial_covariance"),
    )
    rec_model_probability_data = _sequence(recognition_raw["model_source_probabilities"], 2, "recognition.model_source_probabilities")
    rec_model_probabilities = (
        _tensor_vector(rec_model_probability_data[0], 1, "recognition.model_source_probabilities[0]"),
        _tensor_vector(rec_model_probability_data[1], 2, "recognition.model_source_probabilities[1]"),
    )
    rec_state_probability_data = _sequence(recognition_raw["state_source_probabilities_given_model_source"], 2, "recognition.state_source_probabilities")
    rec_state_probabilities = (
        _tensor_matrix(rec_state_probability_data[0], 1, 1, "recognition.state_source_probabilities[0]"),
        _tensor_matrix(rec_state_probability_data[1], 2, 2, "recognition.state_source_probabilities[1]"),
    )
    model_kernel_data = _sequence(recognition_raw["model_kernels"], 2, "recognition.model_kernels")
    model_kernels_list: list[RecognitionModelKernelRecord] = []
    for time, size in enumerate((1, 2)):
        records = [_fields(item, {"slope", "offset", "variance"}, f"recognition.model_kernels[{time}]") for item in _sequence(model_kernel_data[time], size, f"recognition.model_kernels[{time}]")]
        model_kernels_list.append(RecognitionModelKernelRecord(
            torch.tensor([_number(item["slope"], "slope") for item in records], dtype=torch.float64),
            torch.tensor([_number(item["offset"], "offset") for item in records], dtype=torch.float64),
            torch.tensor([_number(item["variance"], "variance") for item in records], dtype=torch.float64),
        ))
    state_kernel_data = _sequence(recognition_raw["state_kernels"], 2, "recognition.state_kernels")
    state_kernels_list: list[RecognitionStateKernelRecord] = []
    for time, size in enumerate((1, 4)):
        expected_fields = {"z_slope", "m_slope", "offset", "variance"} if time == 0 else {"a", "b", "z_slope", "m_slope", "offset", "variance"}
        records = [_fields(item, expected_fields, f"recognition.state_kernels[{time}]") for item in _sequence(state_kernel_data[time], size, f"recognition.state_kernels[{time}]")]
        if time == 1 and [(item["a"], item["b"]) for item in records] != [(0, 0), (1, 0), (0, 1), (1, 1)]:
            raise ValueError("recognition state kernel source order must be (0,0),(1,0),(0,1),(1,1)")
        state_kernels_list.append(RecognitionStateKernelRecord(
            torch.tensor([_number(item["z_slope"], "z_slope") for item in records], dtype=torch.float64),
            torch.tensor([_number(item["m_slope"], "m_slope") for item in records], dtype=torch.float64),
            torch.tensor([_number(item["offset"], "offset") for item in records], dtype=torch.float64),
            torch.tensor([_number(item["variance"], "variance") for item in records], dtype=torch.float64),
        ))
    recognition = RecognitionParameterRecord(
        recognition_initial, rec_model_probabilities, rec_state_probabilities,
        tuple(model_kernels_list), tuple(state_kernels_list),  # type: ignore[arg-type]
    )
    for time in range(2):
        for b, q_b in enumerate(recognition.model_source_probabilities[time]):
            if bool(q_b > 0) and not bool(model_priors[time][b] > 0):
                raise ValueError("recognition model-source mass lies outside positive generative support")
            for a, q_a in enumerate(recognition.state_source_probabilities_given_model_source[time][b]):
                if bool(q_b > 0) and bool(q_a > 0) and not bool(state_priors[time][a] > 0):
                    raise ValueError("recognition state-source mass lies outside positive generative support")
    quadrature = _fields(root["quadrature"], {"order", "convergence_check_order", "maximum_convergence_estimate"}, "quadrature")
    return H1Fixture(
        1, "h1-v1", continuous_order, structure, frames, observation_labels, initial_joint,
        model_priors, state_priors, model_transitions, state_transitions, emissions, recognition,
        quadrature["order"], quadrature["convergence_check_order"],
        _number(quadrature["maximum_convergence_estimate"], "maximum_convergence_estimate"),
    )
