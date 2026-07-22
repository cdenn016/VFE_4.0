"""Frozen records that define the supported H1/H2 configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class RunConfig:
    mode: Literal["verify"]
    seed: int
    device: Literal["cpu"]
    dtype: Literal["float64"]
    deterministic: bool


@dataclass(frozen=True)
class ValidationConfig:
    gates: tuple[Literal["H1"]] | tuple[Literal["H1"], Literal["H2"]]
    fixture_id: Literal["h1-v1"]
    quadrature_order: Literal[21]
    convergence_check_order: Literal[17]
    maximum_convergence_estimate: float


@dataclass(frozen=True)
class DataConfig:
    kind: Literal["frozen_fixture"]
    identity: Literal["h1-v1"]


@dataclass(frozen=True)
class ModelConfig:
    horizon: Literal[2]
    d_z: Literal[1]
    d_m: Literal[1]
    vocabulary_size: Literal[3]
    state_parent_sets: tuple[tuple[int, ...], tuple[int, ...]]
    model_parent_sets: tuple[tuple[int, ...], tuple[int, ...]]
    state_source_support: tuple[tuple[int, ...], tuple[int, ...]]
    model_source_support: tuple[tuple[int, ...], tuple[int, ...]]
    geometry: Literal["fixed_population_frames"]


@dataclass(frozen=True)
class RecognitionConfig:
    conditioning: Literal["smoothing"]
    family: Literal["structured_linear_gaussian_mixture"]
    source_treatment: Literal["exact_enumeration"]


@dataclass(frozen=True)
class InferenceConfig:
    operation: Literal["evaluate_only"]
    estimator: Literal["deterministic_quadrature"]


@dataclass(frozen=True)
class OptimizationConfig:
    e_like_update: Literal["none"]
    m_like_update: Literal["none"]
    expected_autograd_scope: Literal["none"]


@dataclass(frozen=True)
class ArtifactConfig:
    run_root: Path


@dataclass(frozen=True)
class ResolvedConfig:
    schema_version: Literal[1]
    objective_schema_version: Literal["vfe4-state-elbo-v1"]
    run: RunConfig
    data: DataConfig
    model: ModelConfig
    recognition: RecognitionConfig
    inference: InferenceConfig
    optimization: OptimizationConfig
    validation: ValidationConfig
    artifacts: ArtifactConfig
    canonical_json: str
    config_sha256: str
