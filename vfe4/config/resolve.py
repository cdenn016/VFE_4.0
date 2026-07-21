"""Strict, side-effect-free resolution of the frozen H1 configuration."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .schema import (
    ArtifactConfig,
    DataConfig,
    InferenceConfig,
    ModelConfig,
    OptimizationConfig,
    RecognitionConfig,
    ResolvedConfig,
    RunConfig,
    ValidationConfig,
)


_ROOT_KEYS = frozenset(
    {
        "schema_version",
        "objective_schema_version",
        "run",
        "data",
        "model",
        "recognition",
        "inference",
        "optimization",
        "validation",
        "artifacts",
    }
)
_RUN_KEYS = frozenset({"mode", "seed", "device", "dtype", "deterministic"})
_DATA_KEYS = frozenset({"kind", "identity"})
_MODEL_KEYS = frozenset(
    {
        "horizon",
        "d_z",
        "d_m",
        "vocabulary_size",
        "state_parent_sets",
        "model_parent_sets",
        "state_source_support",
        "model_source_support",
        "geometry",
    }
)
_RECOGNITION_KEYS = frozenset({"conditioning", "family", "source_treatment"})
_INFERENCE_KEYS = frozenset({"operation", "estimator"})
_OPTIMIZATION_KEYS = frozenset(
    {"e_like_update", "m_like_update", "expected_autograd_scope"}
)
_VALIDATION_KEYS = frozenset(
    {
        "gates",
        "fixture_id",
        "quadrature_order",
        "convergence_check_order",
        "maximum_convergence_estimate",
    }
)
_ARTIFACT_KEYS = frozenset({"run_root"})
_PARENT_SETS = ((0,), (0, 1))


def resolve_config(raw: Mapping[str, object], *, repo_root: Path) -> ResolvedConfig:
    """Validate and freeze the only configuration supported by the H1 gate."""
    root = _require_mapping(raw, "config")
    _validate_keys(root, _ROOT_KEYS, "config")

    run_raw = _section(root, "run", _RUN_KEYS)
    run = RunConfig(
        mode=_require_exact(run_raw["mode"], "verify", "run.mode"),
        seed=_require_int(run_raw["seed"], "run.seed"),
        device=_require_exact(run_raw["device"], "cpu", "run.device"),
        dtype=_require_exact(run_raw["dtype"], "float64", "run.dtype"),
        deterministic=_require_exact(
            run_raw["deterministic"], True, "run.deterministic"
        ),
    )

    data_raw = _section(root, "data", _DATA_KEYS)
    data = DataConfig(
        kind=_require_exact(data_raw["kind"], "frozen_fixture", "data.kind"),
        identity=_require_exact(data_raw["identity"], "h1-v1", "data.identity"),
    )

    model_raw = _section(root, "model", _MODEL_KEYS)
    model = ModelConfig(
        horizon=_require_exact(model_raw["horizon"], 2, "model.horizon"),
        d_z=_require_exact(model_raw["d_z"], 1, "model.d_z"),
        d_m=_require_exact(model_raw["d_m"], 1, "model.d_m"),
        vocabulary_size=_require_exact(
            model_raw["vocabulary_size"], 3, "model.vocabulary_size"
        ),
        state_parent_sets=_require_parent_sets(
            model_raw["state_parent_sets"], "model.state_parent_sets"
        ),
        model_parent_sets=_require_parent_sets(
            model_raw["model_parent_sets"], "model.model_parent_sets"
        ),
        state_source_support=_require_parent_sets(
            model_raw["state_source_support"], "model.state_source_support"
        ),
        model_source_support=_require_parent_sets(
            model_raw["model_source_support"], "model.model_source_support"
        ),
        geometry=_require_exact(
            model_raw["geometry"], "fixed_population_frames", "model.geometry"
        ),
    )

    recognition_raw = _section(root, "recognition", _RECOGNITION_KEYS)
    recognition = RecognitionConfig(
        conditioning=_require_exact(
            recognition_raw["conditioning"], "smoothing", "recognition.conditioning"
        ),
        family=_require_exact(
            recognition_raw["family"],
            "structured_linear_gaussian_mixture",
            "recognition.family",
        ),
        source_treatment=_require_exact(
            recognition_raw["source_treatment"],
            "exact_enumeration",
            "recognition.source_treatment",
        ),
    )

    inference_raw = _section(root, "inference", _INFERENCE_KEYS)
    inference = InferenceConfig(
        operation=_require_exact(
            inference_raw["operation"], "evaluate_only", "inference.operation"
        ),
        estimator=_require_exact(
            inference_raw["estimator"],
            "deterministic_quadrature",
            "inference.estimator",
        ),
    )

    optimization_raw = _section(root, "optimization", _OPTIMIZATION_KEYS)
    optimization = OptimizationConfig(
        e_like_update=_require_exact(
            optimization_raw["e_like_update"], "none", "optimization.e_like_update"
        ),
        m_like_update=_require_exact(
            optimization_raw["m_like_update"], "none", "optimization.m_like_update"
        ),
        expected_autograd_scope=_require_exact(
            optimization_raw["expected_autograd_scope"],
            "none",
            "optimization.expected_autograd_scope",
        ),
    )

    validation_raw = _section(root, "validation", _VALIDATION_KEYS)
    validation = ValidationConfig(
        gates=_require_gates(validation_raw["gates"]),
        fixture_id=_require_exact(
            validation_raw["fixture_id"], "h1-v1", "validation.fixture_id"
        ),
        quadrature_order=_require_exact(
            validation_raw["quadrature_order"], 21, "validation.quadrature_order"
        ),
        convergence_check_order=_require_exact(
            validation_raw["convergence_check_order"],
            17,
            "validation.convergence_check_order",
        ),
        maximum_convergence_estimate=_require_exact(
            validation_raw["maximum_convergence_estimate"],
            1e-9,
            "validation.maximum_convergence_estimate",
        ),
    )

    artifacts_raw = _section(root, "artifacts", _ARTIFACT_KEYS)
    artifacts = ArtifactConfig(
        run_root=_resolve_run_root(artifacts_raw["run_root"], repo_root)
    )

    canonical_json = _canonical_json(
        schema_version=1,
        objective_schema_version="vfe4-state-elbo-v1",
        run=run,
        data=data,
        model=model,
        recognition=recognition,
        inference=inference,
        optimization=optimization,
        validation=validation,
        artifacts=artifacts,
    )
    config_sha256 = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    return ResolvedConfig(
        schema_version=_require_exact(root["schema_version"], 1, "schema_version"),
        objective_schema_version=_require_exact(
            root["objective_schema_version"],
            "vfe4-state-elbo-v1",
            "objective_schema_version",
        ),
        run=run,
        data=data,
        model=model,
        recognition=recognition,
        inference=inference,
        optimization=optimization,
        validation=validation,
        artifacts=artifacts,
        canonical_json=canonical_json,
        config_sha256=config_sha256,
    )


def _require_mapping(value: object, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{location} keys must be strings")
    return value


def _validate_keys(
    mapping: Mapping[str, object], expected: frozenset[str], location: str
) -> None:
    actual = frozenset(mapping)
    unknown = actual - expected
    missing = expected - actual
    if unknown or missing:
        raise ValueError(
            f"{location} has unknown keys {sorted(unknown)!r} or missing keys {sorted(missing)!r}"
        )


def _section(
    root: Mapping[str, object], name: str, expected: frozenset[str]
) -> Mapping[str, object]:
    section = _require_mapping(root[name], name)
    _validate_keys(section, expected, name)
    return section


def _require_exact(value: object, expected: Any, location: str) -> Any:
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"{location} must equal {expected!r}")
    return value


def _require_int(value: object, location: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{location} must be an integer")
    return value


def _require_parent_sets(value: object, location: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if type(value) is not list or len(value) != 2:
        raise ValueError(f"{location} must contain two lists")
    rows: list[tuple[int, ...]] = []
    for row_index, row in enumerate(value):
        if type(row) is not list:
            raise ValueError(f"{location}[{row_index}] must be a list")
        rows.append(
            tuple(_require_int(item, f"{location}[{row_index}]") for item in row)
        )
    parent_sets = tuple(rows)
    if parent_sets != _PARENT_SETS:
        raise ValueError(f"{location} must equal {list(map(list, _PARENT_SETS))!r}")
    return parent_sets  # type: ignore[return-value]


def _require_gates(value: object) -> tuple[str]:
    if type(value) is not list or value != ["H1"]:
        raise ValueError("validation.gates must equal ['H1']")
    return ("H1",)


def _resolve_run_root(value: object, repo_root: Path) -> Path:
    if not isinstance(value, (str, Path)):
        raise ValueError("artifacts.run_root must be a path string")
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _canonical_json(
    *,
    schema_version: int,
    objective_schema_version: str,
    run: RunConfig,
    data: DataConfig,
    model: ModelConfig,
    recognition: RecognitionConfig,
    inference: InferenceConfig,
    optimization: OptimizationConfig,
    validation: ValidationConfig,
    artifacts: ArtifactConfig,
) -> str:
    payload = {
        "schema_version": schema_version,
        "objective_schema_version": objective_schema_version,
        "run": {
            "mode": run.mode,
            "seed": run.seed,
            "device": run.device,
            "dtype": run.dtype,
            "deterministic": run.deterministic,
        },
        "data": {"kind": data.kind, "identity": data.identity},
        "model": {
            "horizon": model.horizon,
            "d_z": model.d_z,
            "d_m": model.d_m,
            "vocabulary_size": model.vocabulary_size,
            "state_parent_sets": model.state_parent_sets,
            "model_parent_sets": model.model_parent_sets,
            "state_source_support": model.state_source_support,
            "model_source_support": model.model_source_support,
            "geometry": model.geometry,
        },
        "recognition": {
            "conditioning": recognition.conditioning,
            "family": recognition.family,
            "source_treatment": recognition.source_treatment,
        },
        "inference": {
            "operation": inference.operation,
            "estimator": inference.estimator,
        },
        "optimization": {
            "e_like_update": optimization.e_like_update,
            "m_like_update": optimization.m_like_update,
            "expected_autograd_scope": optimization.expected_autograd_scope,
        },
        "validation": {
            "gates": validation.gates,
            "fixture_id": validation.fixture_id,
            "quadrature_order": validation.quadrature_order,
            "convergence_check_order": validation.convergence_check_order,
            "maximum_convergence_estimate": validation.maximum_convergence_estimate,
        },
        "artifacts": {"run_root": artifacts.run_root.as_posix()},
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
