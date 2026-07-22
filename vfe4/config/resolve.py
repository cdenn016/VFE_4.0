"""Strict resolution of the frozen ordered H1/H2/H3 configuration."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from vfe4.types.h3 import (
    H3DecisionConfig,
    H3InitializationConfig,
    H3OptimizationConfig,
)

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
from .control_paths import is_repository_control_path, is_same_or_descendant


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
_ROOT_KEYS_WITH_H3 = _ROOT_KEYS | {"h3"}
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
_H3_KEYS = frozenset(
    {
        "coupled_fixture_id",
        "coupled_expected_sha256",
        "zero_control_fixture_id",
        "zero_control_expected_sha256",
        "recognition_families",
        "common_initialization",
        "optimization_operation",
        "expected_autograd_scope",
        "optimizer",
        "decision",
        "solver_allowance_nats",
        "threshold_decision_rule",
        "minimum_resolved_fraction",
        "coupled_gap_inconclusive_obligation",
        "structured_closure_inconclusive_obligation",
    }
)
_H3_INITIALIZATION_KEYS = frozenset({"mean", "precision"})
_H3_OPTIMIZER_KEYS = frozenset(
    {
        "learning_rate",
        "maximum_iterations_per_step",
        "maximum_evaluations_per_step",
        "tolerance_gradient",
        "tolerance_change",
        "history_size",
        "line_search",
        "maximum_accepted_iterations",
        "maximum_closure_evaluations",
        "terminal_gradient_infinity_norm",
        "terminal_objective_change",
        "required_consecutive_accepted_iterations",
    }
)
_H3_DECISION_KEYS = frozenset(
    {
        "dimension",
        "minimum_precision_eigenvalue",
        "maximum_precision_eigenvalue",
        "maximum_precision_condition_number",
        "maximum_mean_infinity_norm",
        "minimum_coupled_gap_nats",
        "maximum_structured_gap_fraction",
        "maximum_allowance_fraction",
    }
)
_PARENT_SETS = ((0,), (0, 1))
_H3_GATES = ("H1", "H2", "H3")
_H3_FAMILIES = ("structured_full_spd", "fine_factorized_diagonal")
_H3_ZERO_MEAN = (0.0, 0.0, 0.0, 0.0)
_H3_IDENTITY_PRECISION = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


def resolve_config(raw: Mapping[str, object], *, repo_root: Path) -> ResolvedConfig:
    """Validate and freeze an ordered implemented H1/H2/H3 gate prefix."""
    root = _require_mapping(raw, "config")
    expected_root_keys = _ROOT_KEYS_WITH_H3 if "h3" in root else _ROOT_KEYS
    _validate_keys(root, expected_root_keys, "config")

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

    h3 = _resolve_h3(root, validation.gates)

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
        h3=h3,
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
        h3=h3,
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


def _require_gates(
    value: object,
) -> (
    tuple[Literal["H1"]]
    | tuple[Literal["H1"], Literal["H2"]]
    | tuple[Literal["H1"], Literal["H2"], Literal["H3"]]
):
    if type(value) is not list or value not in (
        ["H1"],
        ["H1", "H2"],
        ["H1", "H2", "H3"],
    ):
        raise ValueError(
            "validation.gates must equal ['H1'], ['H1', 'H2'], or "
            "['H1', 'H2', 'H3']"
        )
    return tuple(value)  # type: ignore[return-value]


def _resolve_h3(
    root: Mapping[str, object], gates: tuple[str, ...]
) -> H3ValidationConfig | None:
    requested = gates == _H3_GATES
    present = "h3" in root
    if requested != present:
        raise ValueError(
            "h3 must be present exactly when validation.gates equals "
            "['H1', 'H2', 'H3']"
        )
    if not requested:
        return None

    raw = _section(root, "h3", _H3_KEYS)
    initialization_raw = _nested_section(
        raw,
        "common_initialization",
        _H3_INITIALIZATION_KEYS,
        "h3.common_initialization",
    )
    optimizer_raw = _nested_section(
        raw, "optimizer", _H3_OPTIMIZER_KEYS, "h3.optimizer"
    )
    decision_raw = _nested_section(
        raw, "decision", _H3_DECISION_KEYS, "h3.decision"
    )

    common_initialization = H3InitializationConfig(
        mean=_require_exact_list(
            initialization_raw["mean"],
            _H3_ZERO_MEAN,
            "h3.common_initialization.mean",
        ),  # type: ignore[arg-type]
        precision=_require_exact_matrix(
            initialization_raw["precision"],
            _H3_IDENTITY_PRECISION,
            "h3.common_initialization.precision",
        ),  # type: ignore[arg-type]
    )
    optimizer = H3OptimizationConfig(
        learning_rate=_require_exact(
            optimizer_raw["learning_rate"], 1.0, "h3.optimizer.learning_rate"
        ),
        maximum_iterations_per_step=_require_exact(
            optimizer_raw["maximum_iterations_per_step"],
            1,
            "h3.optimizer.maximum_iterations_per_step",
        ),
        maximum_evaluations_per_step=_require_exact(
            optimizer_raw["maximum_evaluations_per_step"],
            25,
            "h3.optimizer.maximum_evaluations_per_step",
        ),
        tolerance_gradient=_require_exact(
            optimizer_raw["tolerance_gradient"],
            1.0e-12,
            "h3.optimizer.tolerance_gradient",
        ),
        tolerance_change=_require_exact(
            optimizer_raw["tolerance_change"],
            1.0e-18,
            "h3.optimizer.tolerance_change",
        ),
        history_size=_require_exact(
            optimizer_raw["history_size"], 20, "h3.optimizer.history_size"
        ),
        line_search=_require_exact(
            optimizer_raw["line_search"],
            "strong_wolfe",
            "h3.optimizer.line_search",
        ),
        maximum_accepted_iterations=_require_exact(
            optimizer_raw["maximum_accepted_iterations"],
            200,
            "h3.optimizer.maximum_accepted_iterations",
        ),
        maximum_closure_evaluations=_require_exact(
            optimizer_raw["maximum_closure_evaluations"],
            5_000,
            "h3.optimizer.maximum_closure_evaluations",
        ),
        terminal_gradient_infinity_norm=_require_exact(
            optimizer_raw["terminal_gradient_infinity_norm"],
            1.0e-8,
            "h3.optimizer.terminal_gradient_infinity_norm",
        ),
        terminal_objective_change=_require_exact(
            optimizer_raw["terminal_objective_change"],
            1.0e-12,
            "h3.optimizer.terminal_objective_change",
        ),
        required_consecutive_accepted_iterations=_require_exact(
            optimizer_raw["required_consecutive_accepted_iterations"],
            3,
            "h3.optimizer.required_consecutive_accepted_iterations",
        ),
    )
    decision = H3DecisionConfig(
        dimension=_require_exact(
            decision_raw["dimension"], 4, "h3.decision.dimension"
        ),
        minimum_precision_eigenvalue=_require_exact(
            decision_raw["minimum_precision_eigenvalue"],
            1.0e-4,
            "h3.decision.minimum_precision_eigenvalue",
        ),
        maximum_precision_eigenvalue=_require_exact(
            decision_raw["maximum_precision_eigenvalue"],
            1.0e4,
            "h3.decision.maximum_precision_eigenvalue",
        ),
        maximum_precision_condition_number=_require_exact(
            decision_raw["maximum_precision_condition_number"],
            1.0e6,
            "h3.decision.maximum_precision_condition_number",
        ),
        maximum_mean_infinity_norm=_require_exact(
            decision_raw["maximum_mean_infinity_norm"],
            4.0,
            "h3.decision.maximum_mean_infinity_norm",
        ),
        minimum_coupled_gap_nats=_require_exact(
            decision_raw["minimum_coupled_gap_nats"],
            0.50,
            "h3.decision.minimum_coupled_gap_nats",
        ),
        maximum_structured_gap_fraction=_require_exact(
            decision_raw["maximum_structured_gap_fraction"],
            0.01,
            "h3.decision.maximum_structured_gap_fraction",
        ),
        maximum_allowance_fraction=_require_exact(
            decision_raw["maximum_allowance_fraction"],
            0.01,
            "h3.decision.maximum_allowance_fraction",
        ),
    )
    return H3ValidationConfig(
        coupled_fixture_id=_require_exact(
            raw["coupled_fixture_id"], "h3-coupled-v1", "h3.coupled_fixture_id"
        ),
        coupled_expected_sha256=_require_exact(
            raw["coupled_expected_sha256"],
            "6779f5b0a2e27aa5e203764bcc4d84c1b1daedb9423fcefdf28dce3cf7e40e03",
            "h3.coupled_expected_sha256",
        ),
        zero_control_fixture_id=_require_exact(
            raw["zero_control_fixture_id"],
            "h3-zero-control-v1",
            "h3.zero_control_fixture_id",
        ),
        zero_control_expected_sha256=_require_exact(
            raw["zero_control_expected_sha256"],
            "ba600e09e0ae7e2b7576fbf4446a8e5b38a605c7621eb0cd5586689dccb89acf",
            "h3.zero_control_expected_sha256",
        ),
        recognition_families=_require_exact_list(
            raw["recognition_families"],
            _H3_FAMILIES,
            "h3.recognition_families",
        ),  # type: ignore[arg-type]
        common_initialization=common_initialization,
        optimization_operation=_require_exact(
            raw["optimization_operation"],
            "maximize_direct_h3_elbo_lbfgs",
            "h3.optimization_operation",
        ),
        expected_autograd_scope=_require_exact(
            raw["expected_autograd_scope"],
            "h3_recognition_only",
            "h3.expected_autograd_scope",
        ),
        optimizer=optimizer,
        decision=decision,
        solver_allowance_nats=_require_exact(
            raw["solver_allowance_nats"], 1.0e-7, "h3.solver_allowance_nats"
        ),
        threshold_decision_rule=_require_exact(
            raw["threshold_decision_rule"],
            "signed_margin_three_way",
            "h3.threshold_decision_rule",
        ),
        minimum_resolved_fraction=_require_exact(
            raw["minimum_resolved_fraction"],
            0.99,
            "h3.minimum_resolved_fraction",
        ),
        coupled_gap_inconclusive_obligation=_require_exact(
            raw["coupled_gap_inconclusive_obligation"],
            "resolve coupled gap threshold outside allowance",
            "h3.coupled_gap_inconclusive_obligation",
        ),
        structured_closure_inconclusive_obligation=_require_exact(
            raw["structured_closure_inconclusive_obligation"],
            "resolve structured closure threshold outside allowance",
            "h3.structured_closure_inconclusive_obligation",
        ),
    )


def _nested_section(
    parent: Mapping[str, object],
    name: str,
    expected: frozenset[str],
    location: str,
) -> Mapping[str, object]:
    section = _require_mapping(parent[name], location)
    _validate_keys(section, expected, location)
    return section


def _require_exact_list(
    value: object, expected: tuple[Any, ...], location: str
) -> tuple[Any, ...]:
    if type(value) is not list or len(value) != len(expected):
        raise ValueError(f"{location} must equal {list(expected)!r}")
    return tuple(
        _require_exact(item, expected[index], f"{location}[{index}]")
        for index, item in enumerate(value)
    )


def _require_exact_matrix(
    value: object, expected: tuple[tuple[float, ...], ...], location: str
) -> tuple[tuple[float, ...], ...]:
    if type(value) is not list or len(value) != len(expected):
        raise ValueError(f"{location} must contain {len(expected)} rows")
    return tuple(
        _require_exact_list(row, expected[index], f"{location}[{index}]")
        for index, row in enumerate(value)
    )  # type: ignore[return-value]


def _resolve_run_root(value: object, repo_root: Path) -> Path:
    if not isinstance(value, (str, Path)):
        raise ValueError("artifacts.run_root must be a path string")
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    resolved = path.resolve()
    repository = repo_root.resolve()
    if is_repository_control_path(resolved, repository):
        raise ValueError("artifacts.run_root must not enter a repository control tree")
    if is_same_or_descendant(repository, resolved):
        raise ValueError("artifacts.run_root must not equal or contain the repository")
    return resolved


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
    h3: H3ValidationConfig | None,
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
    if h3 is not None:
        payload["h3"] = {
            "coupled_fixture_id": h3.coupled_fixture_id,
            "coupled_expected_sha256": h3.coupled_expected_sha256,
            "zero_control_fixture_id": h3.zero_control_fixture_id,
            "zero_control_expected_sha256": h3.zero_control_expected_sha256,
            "recognition_families": h3.recognition_families,
            "common_initialization": {
                "mean": h3.common_initialization.mean,
                "precision": h3.common_initialization.precision,
            },
            "optimization_operation": h3.optimization_operation,
            "expected_autograd_scope": h3.expected_autograd_scope,
            "optimizer": {
                "learning_rate": h3.optimizer.learning_rate,
                "maximum_iterations_per_step": (
                    h3.optimizer.maximum_iterations_per_step
                ),
                "maximum_evaluations_per_step": (
                    h3.optimizer.maximum_evaluations_per_step
                ),
                "tolerance_gradient": h3.optimizer.tolerance_gradient,
                "tolerance_change": h3.optimizer.tolerance_change,
                "history_size": h3.optimizer.history_size,
                "line_search": h3.optimizer.line_search,
                "maximum_accepted_iterations": (
                    h3.optimizer.maximum_accepted_iterations
                ),
                "maximum_closure_evaluations": (
                    h3.optimizer.maximum_closure_evaluations
                ),
                "terminal_gradient_infinity_norm": (
                    h3.optimizer.terminal_gradient_infinity_norm
                ),
                "terminal_objective_change": (
                    h3.optimizer.terminal_objective_change
                ),
                "required_consecutive_accepted_iterations": (
                    h3.optimizer.required_consecutive_accepted_iterations
                ),
            },
            "decision": {
                "dimension": h3.decision.dimension,
                "minimum_precision_eigenvalue": (
                    h3.decision.minimum_precision_eigenvalue
                ),
                "maximum_precision_eigenvalue": (
                    h3.decision.maximum_precision_eigenvalue
                ),
                "maximum_precision_condition_number": (
                    h3.decision.maximum_precision_condition_number
                ),
                "maximum_mean_infinity_norm": (
                    h3.decision.maximum_mean_infinity_norm
                ),
                "minimum_coupled_gap_nats": h3.decision.minimum_coupled_gap_nats,
                "maximum_structured_gap_fraction": (
                    h3.decision.maximum_structured_gap_fraction
                ),
                "maximum_allowance_fraction": (
                    h3.decision.maximum_allowance_fraction
                ),
            },
            "solver_allowance_nats": h3.solver_allowance_nats,
            "threshold_decision_rule": h3.threshold_decision_rule,
            "minimum_resolved_fraction": h3.minimum_resolved_fraction,
            "coupled_gap_inconclusive_obligation": (
                h3.coupled_gap_inconclusive_obligation
            ),
            "structured_closure_inconclusive_obligation": (
                h3.structured_closure_inconclusive_obligation
            ),
        }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
