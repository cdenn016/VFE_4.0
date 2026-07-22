"""Strict configuration types and resolution for VFE 4.0."""

from .resolve import resolve_config
from .schema import (
    ArtifactConfig,
    DataConfig,
    H3ValidationConfig,
    InferenceConfig,
    ModelConfig,
    OptimizationConfig,
    RecognitionConfig,
    ResolvedConfig,
    RunConfig,
    ValidationConfig,
)

__all__ = [
    "ArtifactConfig",
    "DataConfig",
    "H3ValidationConfig",
    "InferenceConfig",
    "ModelConfig",
    "OptimizationConfig",
    "RecognitionConfig",
    "ResolvedConfig",
    "RunConfig",
    "ValidationConfig",
    "resolve_config",
]
