"""Strict resolution of the frozen ordered H1/H2/H3 configuration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from vfe4.types.h3 import (
    H3DecisionConfig,
    H3InitializationConfig,
    H3OptimizationConfig,
)
from vfe4.types.h4 import H4_PRIMARY_TIMED_BALANCE, H4_PROBLEM_SEEDS, H4SolveProtocol

from .schema import (
    ArtifactConfig,
    DataConfig,
    H3ValidationConfig,
    H4AllowanceConfig,
    H4BootstrapConfig,
    H4ConditionEnvelopeConfig,
    H4EnvironmentConfig,
    H4TimingConfig,
    H4TraversalConfig,
    H4ValidationConfig,
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

_H4_KEYS = frozenset({
    "schema_version", "solve_protocol", "traversal", "timing", "bootstrap",
    "condition_envelope", "allowance", "environment", "primary_effect_threshold",
    "maximum_validation_payload_bytes",
})
_H4_SOLVE_PROTOCOL_KEYS = frozenset({
    "protocol_id", "dtype", "device", "factor_passes", "solver_relative_budget",
    "stopping_rule",
})
_H4_TRAVERSAL_KEYS = frozenset({
    "horizons", "seeds", "kinds", "d_z", "d_m", "dimensions",
    "primary_horizon", "primary_kind", "primary_dimension",
})
_H4_TIMING_KEYS = frozenset({
    "parity_expression", "warmup_pair_indices", "timed_pair_indices",
    "timed_repetitions_per_problem", "warmups_count_toward_balance",
    "primary_timed_balance", "primary_5_ab_6_ba_rows", "primary_6_ab_5_ba_rows",
    "primary_timed_ab_total", "primary_timed_ba_total", "clock", "timer_boundary",
    "between_repetitions",
})
_H4_BOOTSTRAP_KEYS = frozenset({
    "seed", "replicates", "inferential_units", "index_low", "index_high", "endpoint",
    "index_dtype", "index_shape", "statistic", "percentiles", "percentile_method",
    "percentile_space", "digest_domain", "expected_index_sha256",
})
_H4_CONDITION_KEYS = frozenset({
    "posterior_minimum_eigenvalue", "posterior_maximum_eigenvalue",
    "posterior_maximum_condition_number", "posterior_minimum_cholesky_pivot",
    "posterior_maximum_mean_infinity_norm", "innovation_minimum_eigenvalue",
    "innovation_maximum_eigenvalue", "innovation_maximum_condition_number", "inclusive",
})
_H4_ALLOWANCE_KEYS = frozenset({
    "float64_epsilon", "rounding_constant", "solver_relative_budget",
    "maximum_allowance_scale_fraction", "decisiveness_comparison",
    "element_stream_domain", "maximum_chunk_rows",
})
_H4_ENVIRONMENT_KEYS = frozenset({
    "device", "dtype", "intra_op_threads", "alter_inter_op_threads", "cuda_expected",
    "gc_policy", "power_policy_field_order", "power_policy_capture",
})


def resolve_h4_validation_config(raw_h4: Mapping[str, object]) -> H4ValidationConfig:
    """Strictly resolve the standalone frozen H4 section without changing gate prefixes."""

    root = _require_mapping(raw_h4, "h4")
    _validate_keys(root, _H4_KEYS, "h4")
    protocol_raw = _nested_section(root, "solve_protocol", _H4_SOLVE_PROTOCOL_KEYS, "h4.solve_protocol")
    protocol = H4SolveProtocol(
        protocol_id=_require_exact(protocol_raw["protocol_id"], "h4-single-pass-v1", "h4.solve_protocol.protocol_id"),
        dtype=_require_exact(protocol_raw["dtype"], "float64", "h4.solve_protocol.dtype"),
        device=_require_exact(protocol_raw["device"], "cpu", "h4.solve_protocol.device"),
        factor_passes=_require_exact(protocol_raw["factor_passes"], 1, "h4.solve_protocol.factor_passes"),
        solver_relative_budget=_require_exact(protocol_raw["solver_relative_budget"], 1.0e-9, "h4.solve_protocol.solver_relative_budget"),
        stopping_rule=_require_exact(protocol_raw["stopping_rule"], "complete_schedule_finite_spd", "h4.solve_protocol.stopping_rule"),
    )
    traversal_raw = _nested_section(root, "traversal", _H4_TRAVERSAL_KEYS, "h4.traversal")
    traversal = H4TraversalConfig(
        horizons=_require_exact_list(traversal_raw["horizons"], (7, 15, 31), "h4.traversal.horizons"),
        seeds=_require_exact_list(traversal_raw["seeds"], H4_PROBLEM_SEEDS, "h4.traversal.seeds"),
        kinds=_require_exact_list(traversal_raw["kinds"], ("coupled", "zero_control"), "h4.traversal.kinds"),
        d_z=_require_exact(traversal_raw["d_z"], 4, "h4.traversal.d_z"),
        d_m=_require_exact(traversal_raw["d_m"], 4, "h4.traversal.d_m"),
        dimensions=_require_exact_list(traversal_raw["dimensions"], (64, 128, 256), "h4.traversal.dimensions"),
        primary_horizon=_require_exact(traversal_raw["primary_horizon"], 31, "h4.traversal.primary_horizon"),
        primary_kind=_require_exact(traversal_raw["primary_kind"], "coupled", "h4.traversal.primary_kind"),
        primary_dimension=_require_exact(traversal_raw["primary_dimension"], 256, "h4.traversal.primary_dimension"),
    )
    timing_raw = _nested_section(root, "timing", _H4_TIMING_KEYS, "h4.timing")
    balance_value = timing_raw["primary_timed_balance"]
    if type(balance_value) is not list or len(balance_value) != 20:
        raise ValueError("h4.timing.primary_timed_balance must contain 20 rows")
    balance = tuple(
        _require_exact_list(row, H4_PRIMARY_TIMED_BALANCE[index], f"h4.timing.primary_timed_balance[{index}]")
        for index, row in enumerate(balance_value)
    )
    timing = H4TimingConfig(
        parity_expression=_require_exact(timing_raw["parity_expression"], "(horizon_index + seed_index + kind_index + pair_index) % 2 == 0", "h4.timing.parity_expression"),
        warmup_pair_indices=_require_exact_list(timing_raw["warmup_pair_indices"], (0, 1, 2), "h4.timing.warmup_pair_indices"),
        timed_pair_indices=_require_exact_list(timing_raw["timed_pair_indices"], tuple(range(3, 14)), "h4.timing.timed_pair_indices"),
        timed_repetitions_per_problem=_require_exact(timing_raw["timed_repetitions_per_problem"], 11, "h4.timing.timed_repetitions_per_problem"),
        warmups_count_toward_balance=_require_exact(timing_raw["warmups_count_toward_balance"], False, "h4.timing.warmups_count_toward_balance"),
        primary_timed_balance=balance,
        primary_5_ab_6_ba_rows=_require_exact(timing_raw["primary_5_ab_6_ba_rows"], 10, "h4.timing.primary_5_ab_6_ba_rows"),
        primary_6_ab_5_ba_rows=_require_exact(timing_raw["primary_6_ab_5_ba_rows"], 10, "h4.timing.primary_6_ab_5_ba_rows"),
        primary_timed_ab_total=_require_exact(timing_raw["primary_timed_ab_total"], 110, "h4.timing.primary_timed_ab_total"),
        primary_timed_ba_total=_require_exact(timing_raw["primary_timed_ba_total"], 110, "h4.timing.primary_timed_ba_total"),
        clock=_require_exact(timing_raw["clock"], "time.perf_counter_ns", "h4.timing.clock"),
        timer_boundary=_require_exact(timing_raw["timer_boundary"], "fresh_native_solver_call_v1", "h4.timing.timer_boundary"),
        between_repetitions=_require_exact(timing_raw["between_repetitions"], "timer_reads_and_preallocated_assignments_only", "h4.timing.between_repetitions"),
    )
    _validate_h4_orders(traversal, timing)
    bootstrap_raw = _nested_section(root, "bootstrap", _H4_BOOTSTRAP_KEYS, "h4.bootstrap")
    bootstrap = H4BootstrapConfig(
        seed=_require_exact(bootstrap_raw["seed"], 20260721, "h4.bootstrap.seed"),
        replicates=_require_exact(bootstrap_raw["replicates"], 100000, "h4.bootstrap.replicates"),
        inferential_units=_require_exact(bootstrap_raw["inferential_units"], 20, "h4.bootstrap.inferential_units"),
        index_low=_require_exact(bootstrap_raw["index_low"], 0, "h4.bootstrap.index_low"),
        index_high=_require_exact(bootstrap_raw["index_high"], 20, "h4.bootstrap.index_high"),
        endpoint=_require_exact(bootstrap_raw["endpoint"], False, "h4.bootstrap.endpoint"),
        index_dtype=_require_exact(bootstrap_raw["index_dtype"], "<i8", "h4.bootstrap.index_dtype"),
        index_shape=_require_exact_list(bootstrap_raw["index_shape"], (100000, 20), "h4.bootstrap.index_shape"),
        statistic=_require_exact(bootstrap_raw["statistic"], "mean_log_seed_ratio", "h4.bootstrap.statistic"),
        percentiles=_require_exact_list(bootstrap_raw["percentiles"], (2.5, 97.5), "h4.bootstrap.percentiles"),
        percentile_method=_require_exact(bootstrap_raw["percentile_method"], "linear", "h4.bootstrap.percentile_method"),
        percentile_space=_require_exact(bootstrap_raw["percentile_space"], "log_then_exp", "h4.bootstrap.percentile_space"),
        digest_domain=_require_exact(bootstrap_raw["digest_domain"], "vfe4.h4.bootstrap-indices.v1", "h4.bootstrap.digest_domain"),
        expected_index_sha256=_require_exact(bootstrap_raw["expected_index_sha256"], "a254e18bccc519a719e9f4b409f45cc9ae4a2a321903531cd8fd73433687cd14", "h4.bootstrap.expected_index_sha256"),
    )
    condition_raw = _nested_section(root, "condition_envelope", _H4_CONDITION_KEYS, "h4.condition_envelope")
    condition = H4ConditionEnvelopeConfig(**{
        name: _require_exact(condition_raw[name], expected, f"h4.condition_envelope.{name}")
        for name, expected in (
            ("posterior_minimum_eigenvalue", 1.0e-6), ("posterior_maximum_eigenvalue", 1.0e6),
            ("posterior_maximum_condition_number", 1.0e8), ("posterior_minimum_cholesky_pivot", 1.0e-3),
            ("posterior_maximum_mean_infinity_norm", 16.0), ("innovation_minimum_eigenvalue", 1.0e-6),
            ("innovation_maximum_eigenvalue", 1.0e6), ("innovation_maximum_condition_number", 1.0e8),
            ("inclusive", True),
        )
    })
    allowance_raw = _nested_section(root, "allowance", _H4_ALLOWANCE_KEYS, "h4.allowance")
    allowance = H4AllowanceConfig(
        float64_epsilon=_require_exact(allowance_raw["float64_epsilon"], 2.220446049250313e-16, "h4.allowance.float64_epsilon"),
        rounding_constant=_require_exact(allowance_raw["rounding_constant"], 4096, "h4.allowance.rounding_constant"),
        solver_relative_budget=_require_exact(allowance_raw["solver_relative_budget"], 1.0e-9, "h4.allowance.solver_relative_budget"),
        maximum_allowance_scale_fraction=_require_exact(allowance_raw["maximum_allowance_scale_fraction"], 1.0e-4, "h4.allowance.maximum_allowance_scale_fraction"),
        decisiveness_comparison=_require_exact(allowance_raw["decisiveness_comparison"], "strict_less_than", "h4.allowance.decisiveness_comparison"),
        element_stream_domain=_require_exact(allowance_raw["element_stream_domain"], "vfe4.h4.allowance-element-stream.v1", "h4.allowance.element_stream_domain"),
        maximum_chunk_rows=_require_exact(allowance_raw["maximum_chunk_rows"], 4096, "h4.allowance.maximum_chunk_rows"),
    )
    environment_raw = _nested_section(root, "environment", _H4_ENVIRONMENT_KEYS, "h4.environment")
    environment = H4EnvironmentConfig(
        device=_require_exact(environment_raw["device"], "cpu", "h4.environment.device"),
        dtype=_require_exact(environment_raw["dtype"], "float64", "h4.environment.dtype"),
        intra_op_threads=_require_exact(environment_raw["intra_op_threads"], 1, "h4.environment.intra_op_threads"),
        alter_inter_op_threads=_require_exact(environment_raw["alter_inter_op_threads"], False, "h4.environment.alter_inter_op_threads"),
        cuda_expected=_require_exact(environment_raw["cuda_expected"], False, "h4.environment.cuda_expected"),
        gc_policy=_require_exact(environment_raw["gc_policy"], "restore_exact_prior_enabled_state", "h4.environment.gc_policy"),
        power_policy_field_order=_require_exact_list(environment_raw["power_policy_field_order"], ("active_power_scheme", "cpu_frequency_governor", "energy_performance_preference", "low_power_mode"), "h4.environment.power_policy_field_order"),
        power_policy_capture=_require_exact(environment_raw["power_policy_capture"], "typed_best_effort_outside_timing", "h4.environment.power_policy_capture"),
    )
    primary_effect_threshold = _require_exact(root["primary_effect_threshold"], 0.80, "h4.primary_effect_threshold")
    payload_limit = _require_exact(root["maximum_validation_payload_bytes"], 67_108_864, "h4.maximum_validation_payload_bytes")
    payload = {
        "schema_version": "h4-validation-config-v1", "solve_protocol": asdict(protocol),
        "traversal": asdict(traversal), "timing": asdict(timing),
        "bootstrap": asdict(bootstrap), "condition_envelope": asdict(condition),
        "allowance": asdict(allowance), "environment": asdict(environment),
        "primary_effect_threshold": primary_effect_threshold,
        "maximum_validation_payload_bytes": payload_limit,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return H4ValidationConfig(
        "h4-validation-config-v1", protocol, traversal, timing, bootstrap, condition,
        allowance, environment, primary_effect_threshold, payload_limit, canonical,
        hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )


def _validate_h4_orders(traversal: H4TraversalConfig, timing: H4TimingConfig) -> None:
    primary_rows: list[tuple[int, int, int]] = []
    problem_index = 0
    for horizon_index, horizon in enumerate(traversal.horizons):
        for seed_index, seed in enumerate(traversal.seeds):
            for kind_index, _kind in enumerate(traversal.kinds):
                if problem_index != ((horizon_index * 20 + seed_index) * 2 + kind_index):
                    raise ValueError("H4 problem traversal index mismatch")
                for pair_index in (*timing.warmup_pair_indices, *timing.timed_pair_indices):
                    _ = (horizon_index + seed_index + kind_index + pair_index) % 2 == 0
                if horizon == 31 and kind_index == 0:
                    ab = sum(
                        (horizon_index + seed_index + kind_index + pair_index) % 2 == 0
                        for pair_index in timing.timed_pair_indices
                    )
                    primary_rows.append((seed, ab, 11 - ab))
                problem_index += 1
    if problem_index != 120 or tuple(primary_rows) != timing.primary_timed_balance:
        raise ValueError("H4 timing balance does not follow independent-index parity")


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
