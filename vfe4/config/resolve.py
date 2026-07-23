"""Strict resolution of the frozen ordered H1--H5 configuration prefixes."""

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
from vfe4.types.h5_schema import (
    H5_FACTOR_INPUT_SCHEMA_SHA256,
    H5_FACTOR_INPUT_SCHEMA_VERSION,
    H5_FACTOR_UNIVERSE,
    H5_H1_FIXTURE_RAW_SHA256,
    H5_MODEL_BLOCK_UNIVERSE,
    H5_OBJECTIVE_SCHEMA_SHA256,
    H5_QUADRATURE_ORDERS,
    H5_RECOGNITION_COORDINATE_UNIVERSE,
)
from vfe4.types.updates import H5UpdateRule, UpdateLabel
from vfe4.types.h6 import (
    CausalDag,
    CausalDagRow,
    EndpointSmcProtocol,
    EstimatorSpec,
    H6ArmPhaseSchedule,
    H6LanguageStructure,
    H6OuterSchedule,
    H6TrainingSchedule,
    TrainingPhase,
    VocabularyIdentity,
    ZeroDimensionalBase,
)
from vfe4.validation.h5_update_spec import EXPECTED_H5_UPDATE_SPEC_RAW_SHA256

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
    H5ValidationConfig,
    H5_CONTROL_IDS,
    H5_POSITIVE_CASE_IDS,
    H5_UPDATE_SPEC_CANONICAL_SHA256,
    InferenceConfig,
    H6PredictionResolvedConfig,
    H6PrefixResolvedConfig,
    H6SourceIdentity,
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
_ROOT_KEYS_WITH_H4_H5 = _ROOT_KEYS | {"h3", "h4", "h5"}
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
_H5_KEYS = frozenset({
    "schema_version", "fixture_id", "fixture_schema_version", "recognition_family",
    "h1_fixture_id", "h1_fixture_raw_sha256", "update_spec_raw_sha256",
    "update_spec_canonical_sha256", "objective_schema_sha256",
    "factor_input_schema_version", "factor_input_schema_sha256", "factor_universe",
    "recognition_coordinate_universe", "model_block_universe",
    "enabled_update_rules", "enabled_update_labels", "positive_case_ids",
    "control_ids", "quadrature_orders", "allowance_policy", "rounding_constant",
    "stochastic_contribution", "epsilon_delta_formula", "mm_proof_artifact",
})
_PARENT_SETS = ((0,), (0, 1))
_H3_GATES = ("H1", "H2", "H3")
_H5_GATES = ("H1", "H2", "H3", "H4", "H5")
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


def _require_h6_sha256(value: object, location: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{location} must be a lowercase 64-hex SHA-256")
    return value


def _resolve_h6_source(value: object) -> H6SourceIdentity:
    raw = _require_mapping(value, "source")
    _validate_keys(raw, frozenset({"git_head", "dirty_digest", "source_sha256"}), "source")
    git_head = raw["git_head"]
    if (
        type(git_head) is not str
        or len(git_head) != 40
        or any(character not in "0123456789abcdef" for character in git_head)
    ):
        raise ValueError("source.git_head must be a lowercase 40-hex Git object name")
    return H6SourceIdentity(
        git_head,
        _require_h6_sha256(raw["dirty_digest"], "source.dirty_digest"),
        _require_h6_sha256(raw["source_sha256"], "source.source_sha256"),
    )


def _h6_json(payload: Mapping[str, object]) -> tuple[str, str]:
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return canonical, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _resolve_h6_prefix_config(
    raw: Mapping[str, object], *, repo_root: Path
) -> H6PrefixResolvedConfig:
    """Resolve the independent predecessor-free H6 Prefix configuration."""
    root = _require_mapping(raw, "h6_prefix")
    _validate_keys(
        root,
        frozenset(
            {
                "schema_version", "operation", "source", "structure",
                "model_family_sha256", "vocabulary", "estimator",
                "data_safety_sha256", "artifact_root",
            }
        ),
        "h6_prefix",
    )
    schema_version = _require_exact(
        root["schema_version"], "h6-prefix-config-v1", "h6_prefix.schema_version"
    )
    operation = _require_exact(root["operation"], "H6-Prefix", "h6_prefix.operation")
    source = _resolve_h6_source(root["source"])

    structure_raw = _require_mapping(root["structure"], "h6_prefix.structure")
    _validate_keys(
        structure_raw, frozenset({"base", "dag", "receiver_labels"}),
        "h6_prefix.structure",
    )
    base_raw = _require_mapping(structure_raw["base"], "h6_prefix.structure.base")
    _validate_keys(
        base_raw, frozenset({"base_id", "points", "dimension"}),
        "h6_prefix.structure.base",
    )
    _require_exact(base_raw["base_id"], "C0", "h6_prefix.structure.base.base_id")
    _require_exact(base_raw["points"], ["*"], "h6_prefix.structure.base.points")
    _require_exact(base_raw["dimension"], 0, "h6_prefix.structure.base.dimension")
    base = ZeroDimensionalBase.create()

    dag_raw = _require_mapping(structure_raw["dag"], "h6_prefix.structure.dag")
    _validate_keys(
        dag_raw, frozenset({"labeling", "node_labels", "rows"}),
        "h6_prefix.structure.dag",
    )
    _require_exact(dag_raw["labeling"], "zero_based", "h6_prefix.structure.dag.labeling")
    node_labels_raw = dag_raw["node_labels"]
    if type(node_labels_raw) is not list or any(type(item) is not int for item in node_labels_raw):
        raise ValueError("h6_prefix.structure.dag.node_labels must be an integer list")
    rows_raw = dag_raw["rows"]
    if type(rows_raw) is not list:
        raise ValueError("h6_prefix.structure.dag.rows must be a list")
    rows: list[CausalDagRow] = []
    for index, value in enumerate(rows_raw):
        row = _require_mapping(value, f"h6_prefix.structure.dag.rows[{index}]")
        _validate_keys(
            row, frozenset({"receiver_t", "parents"}),
            f"h6_prefix.structure.dag.rows[{index}]",
        )
        parents = row["parents"]
        if type(parents) is not list or any(type(item) is not int for item in parents):
            raise ValueError(f"h6_prefix.structure.dag.rows[{index}].parents must be integers")
        rows.append(CausalDagRow(_require_int(row["receiver_t"], "receiver_t"), tuple(parents)))
    dag = CausalDag.create(node_labels=tuple(node_labels_raw), rows=tuple(rows))
    receivers_raw = structure_raw["receiver_labels"]
    if type(receivers_raw) is not list or any(type(item) is not int for item in receivers_raw):
        raise ValueError("h6_prefix.structure.receiver_labels must be an integer list")
    structure = H6LanguageStructure.create(
        base=base, dag=dag, receiver_labels=tuple(receivers_raw)
    )

    model_family_sha256 = _require_h6_sha256(
        root["model_family_sha256"], "h6_prefix.model_family_sha256"
    )
    vocabulary_raw = _require_mapping(root["vocabulary"], "h6_prefix.vocabulary")
    _validate_keys(
        vocabulary_raw, frozenset({"vocabulary_id", "size", "tokenizer_spec_sha256"}),
        "h6_prefix.vocabulary",
    )
    vocabulary_id = vocabulary_raw["vocabulary_id"]
    vocabulary_size = _require_int(
        vocabulary_raw["size"], "h6_prefix.vocabulary.size"
    )
    if type(vocabulary_id) is not str or (vocabulary_id, vocabulary_size) not in {
        ("h6-prefix-small-v1", 3),
        ("wikitext-2-byte-v1", 258),
    }:
        raise ValueError(
            "h6_prefix.vocabulary must equal either "
            "('h6-prefix-small-v1', 3) or ('wikitext-2-byte-v1', 258)"
        )
    vocabulary = VocabularyIdentity(
        vocabulary_id,
        vocabulary_size,
        _require_h6_sha256(
            vocabulary_raw["tokenizer_spec_sha256"],
            "h6_prefix.vocabulary.tokenizer_spec_sha256",
        ),
    )

    estimator_raw = _require_mapping(root["estimator"], "h6_prefix.estimator")
    _validate_keys(
        estimator_raw,
        frozenset({"schema_version", "kind", "particle_count", "resampling", "dtype", "device"}),
        "h6_prefix.estimator",
    )
    _require_exact(estimator_raw["schema_version"], "h6-estimator-v1", "estimator.schema_version")
    estimator = EstimatorSpec.create(
        kind=_require_exact(estimator_raw["kind"], "deterministic_exact", "estimator.kind"),
        particle_count=_require_exact(estimator_raw["particle_count"], None, "estimator.particle_count"),
        resampling=_require_exact(estimator_raw["resampling"], "none", "estimator.resampling"),
        dtype=_require_exact(estimator_raw["dtype"], "float64", "estimator.dtype"),
        device=_require_exact(estimator_raw["device"], "cpu", "estimator.device"),
    )
    data_safety_sha256 = _require_h6_sha256(
        root["data_safety_sha256"], "h6_prefix.data_safety_sha256"
    )
    artifact_root = _resolve_run_root(root["artifact_root"], repo_root)
    payload = {
        "schema_version": schema_version,
        "operation": operation,
        "source": {
            "git_head": source.git_head,
            "dirty_digest": source.dirty_digest,
            "source_sha256": source.source_sha256,
        },
        "structure": {
            "base_sha256": base.canonical_sha256,
            "dag_sha256": dag.canonical_sha256,
            "receiver_labels": structure.receiver_labels,
            "structure_sha256": structure.structure_sha256,
        },
        "model_family_sha256": model_family_sha256,
        "vocabulary": {
            "vocabulary_id": vocabulary.vocabulary_id,
            "size": vocabulary.size,
            "tokenizer_spec_sha256": vocabulary.tokenizer_spec_sha256,
        },
        "estimator": {
            "schema_version": estimator.schema_version,
            "kind": estimator.kind,
            "particle_count": estimator.particle_count,
            "resampling": estimator.resampling,
            "dtype": estimator.dtype,
            "device": estimator.device,
            "estimator_sha256": estimator.estimator_sha256,
        },
        "data_safety_sha256": data_safety_sha256,
        "artifact_root": artifact_root.as_posix(),
    }
    canonical_json, config_sha256 = _h6_json(payload)
    return H6PrefixResolvedConfig(
        schema_version, operation, source, structure, model_family_sha256,
        vocabulary, estimator, data_safety_sha256, artifact_root,
        canonical_json, config_sha256,
    )


def _resolve_h6_prediction_config(
    raw: Mapping[str, object], *, repo_root: Path
) -> H6PredictionResolvedConfig:
    """Resolve H6 Prediction without admitting H4 as a prerequisite."""
    root = _require_mapping(raw, "h6_prediction")
    _validate_keys(
        root,
        frozenset(
            {
                "schema_version", "operation", "source", "prerequisites",
                "h5_update_binding_sha256", "training_schedule",
                "critical_values_sha256", "endpoint_smc_protocol",
                "attribution_matrix_sha256", "matching_set_sha256",
                "data_identity_sha256", "access_policy_sha256", "artifact_root",
            }
        ),
        "h6_prediction",
    )
    schema_version = _require_exact(
        root["schema_version"], "h6-prediction-config-v1",
        "h6_prediction.schema_version",
    )
    operation = _require_exact(
        root["operation"], "H6-Prediction", "h6_prediction.operation"
    )
    source = _resolve_h6_source(root["source"])
    prerequisites = _require_mapping(root["prerequisites"], "h6_prediction.prerequisites")
    _validate_keys(
        prerequisites,
        frozenset(
            {
                "correctness_manifests", "h1_prefix_prior_manifest_sha256",
                "smc_validation_manifest_sha256", "prefix_certificate_set_sha256",
            }
        ),
        "h6_prediction.prerequisites",
    )
    correctness_raw = _require_mapping(
        prerequisites["correctness_manifests"],
        "h6_prediction.prerequisites.correctness_manifests",
    )
    if set(correctness_raw) != {"H1", "H2", "H3", "H5"}:
        raise ValueError("correctness manifests must be exactly H1, H2, H3, H5; H4 is forbidden")
    correctness_manifests = tuple(
        (gate, _require_h6_sha256(correctness_raw[gate], f"correctness_manifests.{gate}"))
        for gate in ("H1", "H2", "H3", "H5")
    )
    prefix_value = prerequisites["prefix_certificate_set_sha256"]
    if prefix_value is None:
        raise ValueError("exact H6-Prefix certificate set is required")
    prefix_certificate_set_sha256 = _require_h6_sha256(
        prefix_value, "prefix_certificate_set_sha256"
    )
    h1_prefix_prior_manifest_sha256 = _require_h6_sha256(
        prerequisites["h1_prefix_prior_manifest_sha256"],
        "h1_prefix_prior_manifest_sha256",
    )
    smc_validation_manifest_sha256 = _require_h6_sha256(
        prerequisites["smc_validation_manifest_sha256"],
        "smc_validation_manifest_sha256",
    )

    schedule_raw = _require_mapping(root["training_schedule"], "h6_prediction.training_schedule")
    _validate_keys(
        schedule_raw, frozenset({"schedule_schema", "outer", "endpoint_phases"}),
        "h6_prediction.training_schedule",
    )
    _require_exact(
        schedule_raw["schedule_schema"], "h6-training-schedule-v2",
        "training_schedule.schedule_schema",
    )
    outer_raw = _require_mapping(schedule_raw["outer"], "training_schedule.outer")
    _validate_keys(
        outer_raw,
        frozenset(
            {
                "schedule_schema", "optimizer_class", "optimizer_policy_sha256",
                "model_updates_per_batch", "validation_twentieths_per_pass", "full_passes",
            }
        ),
        "training_schedule.outer",
    )
    for name, expected in (
        ("schedule_schema", "h6-outer-schedule-v1"),
        ("optimizer_class", "AdamW"),
        ("model_updates_per_batch", 1),
        ("validation_twentieths_per_pass", 20),
        ("full_passes", 2),
    ):
        _require_exact(outer_raw[name], expected, f"training_schedule.outer.{name}")
    outer = H6OuterSchedule.create(
        optimizer_policy_sha256=_require_h6_sha256(
            outer_raw["optimizer_policy_sha256"],
            "training_schedule.outer.optimizer_policy_sha256",
        )
    )
    phases_raw = schedule_raw["endpoint_phases"]
    if type(phases_raw) is not list or not phases_raw:
        raise ValueError("training_schedule.endpoint_phases must be a nonempty list")
    phase_records: list[H6ArmPhaseSchedule] = []
    for index, value in enumerate(phases_raw):
        phase_raw = _require_mapping(value, f"endpoint_phases[{index}]")
        _validate_keys(
            phase_raw,
            frozenset(
                {
                    "endpoint_config_sha256", "latent_enabled", "phases",
                    "recognition_updates_per_batch", "model_updates_per_batch", "no_op_phases",
                }
            ),
            f"endpoint_phases[{index}]",
        )
        latent_enabled = phase_raw["latent_enabled"]
        if type(latent_enabled) is not bool:
            raise ValueError("endpoint latent_enabled must be a bool")
        raw_phase_names = phase_raw["phases"]
        if type(raw_phase_names) is not list:
            raise ValueError("endpoint phases must be a list")
        try:
            phases = tuple(TrainingPhase(name) for name in raw_phase_names)
        except (TypeError, ValueError) as exc:
            raise ValueError("endpoint phase name is unsupported") from exc
        record = H6ArmPhaseSchedule.create(
            endpoint_config_sha256=_require_h6_sha256(
                phase_raw["endpoint_config_sha256"], "endpoint_config_sha256"
            ),
            latent_enabled=latent_enabled,
            phases=phases,
        )
        for name in ("recognition_updates_per_batch", "model_updates_per_batch", "no_op_phases"):
            if _require_int(phase_raw[name], name) != getattr(record, name):
                raise ValueError(f"endpoint {name} does not match phase schedule")
        phase_records.append(record)
    training_schedule = H6TrainingSchedule.create(
        outer=outer, endpoint_phases=tuple(phase_records)
    )

    endpoint_raw = _require_mapping(root["endpoint_smc_protocol"], "endpoint_smc_protocol")
    _validate_keys(
        endpoint_raw,
        frozenset(
            {
                "protocol_schema", "particle_counts", "replicate_count",
                "registry_root_seed", "common_stream_domain",
                "simultaneous_interval_count", "familywise_alpha",
                "critical_value_df63", "remainder_contraction",
            }
        ),
        "endpoint_smc_protocol",
    )
    _require_exact(endpoint_raw["protocol_schema"], "h6-endpoint-smc-v1", "protocol_schema")
    particles = endpoint_raw["particle_counts"]
    if type(particles) is not list or any(type(item) is not int for item in particles):
        raise ValueError("particle_counts must be an integer list")
    endpoint_protocol = EndpointSmcProtocol.create(
        particle_counts=tuple(particles),
        replicate_count=_require_int(endpoint_raw["replicate_count"], "replicate_count"),
        registry_root_seed=_require_int(endpoint_raw["registry_root_seed"], "registry_root_seed"),
        common_stream_domain=_require_exact(
            endpoint_raw["common_stream_domain"], "h6-wt2-endpoint-mc-v1", "common_stream_domain"
        ),
        simultaneous_interval_count=_require_int(
            endpoint_raw["simultaneous_interval_count"], "simultaneous_interval_count"
        ),
        familywise_alpha=_require_exact(endpoint_raw["familywise_alpha"], 0.01, "familywise_alpha"),
        critical_value_df63=_require_exact(
            endpoint_raw["critical_value_df63"], 4.5144904535377144, "critical_value_df63"
        ),
        remainder_contraction=_require_exact(
            endpoint_raw["remainder_contraction"], 0.75, "remainder_contraction"
        ),
    )
    digest_names = (
        "h5_update_binding_sha256", "critical_values_sha256",
        "attribution_matrix_sha256", "matching_set_sha256",
        "data_identity_sha256", "access_policy_sha256",
    )
    digests = {name: _require_h6_sha256(root[name], name) for name in digest_names}
    artifact_root = _resolve_run_root(root["artifact_root"], repo_root)
    payload = {
        "schema_version": schema_version,
        "operation": operation,
        "source": {
            "git_head": source.git_head,
            "dirty_digest": source.dirty_digest,
            "source_sha256": source.source_sha256,
        },
        "prerequisites": {
            "correctness_manifests": dict(correctness_manifests),
            "h1_prefix_prior_manifest_sha256": h1_prefix_prior_manifest_sha256,
            "smc_validation_manifest_sha256": smc_validation_manifest_sha256,
            "prefix_certificate_set_sha256": prefix_certificate_set_sha256,
        },
        "h5_update_binding_sha256": digests["h5_update_binding_sha256"],
        "training_schedule": {
            "schedule_schema": training_schedule.schedule_schema,
            "outer_schedule_sha256": outer.outer_schedule_sha256,
            "phase_schedule_sha256": tuple(item.phase_schedule_sha256 for item in phase_records),
            "schedule_sha256": training_schedule.schedule_sha256,
        },
        "critical_values_sha256": digests["critical_values_sha256"],
        "endpoint_smc_protocol_sha256": endpoint_protocol.protocol_sha256,
        "attribution_matrix_sha256": digests["attribution_matrix_sha256"],
        "matching_set_sha256": digests["matching_set_sha256"],
        "data_identity_sha256": digests["data_identity_sha256"],
        "access_policy_sha256": digests["access_policy_sha256"],
        "artifact_root": artifact_root.as_posix(),
    }
    canonical_json, config_sha256 = _h6_json(payload)
    return H6PredictionResolvedConfig(
        schema_version,
        operation,
        source,
        correctness_manifests,
        h1_prefix_prior_manifest_sha256,
        smc_validation_manifest_sha256,
        prefix_certificate_set_sha256,
        digests["h5_update_binding_sha256"],
        training_schedule,
        digests["critical_values_sha256"],
        endpoint_protocol,
        digests["attribution_matrix_sha256"],
        digests["matching_set_sha256"],
        digests["data_identity_sha256"],
        digests["access_policy_sha256"],
        artifact_root,
        canonical_json,
        config_sha256,
    )


def resolve_config(
    raw: Mapping[str, object], *, repo_root: Path
) -> ResolvedConfig | H6PrefixResolvedConfig | H6PredictionResolvedConfig:
    """Resolve one frozen H1--H5, H6 Prefix, or H6 Prediction configuration."""

    root = _require_mapping(raw, "config")
    discriminator = (root.get("schema_version"), root.get("operation"))
    if discriminator == ("h6-prefix-config-v1", "H6-Prefix"):
        return _resolve_h6_prefix_config(root, repo_root=repo_root)
    if discriminator == ("h6-prediction-config-v1", "H6-Prediction"):
        return _resolve_h6_prediction_config(root, repo_root=repo_root)
    if type(root.get("schema_version")) is str or "operation" in root:
        raise ValueError(
            "config schema_version and operation must identify a supported "
            "H6 Prefix or H6 Prediction configuration"
        )
    return _resolve_h1_h5_config(root, repo_root=repo_root)


def resolve_h6_prefix_config(
    raw: Mapping[str, object], *, repo_root: Path
) -> H6PrefixResolvedConfig:
    """Resolve H6 Prefix through the single public configuration resolver."""

    resolved = resolve_config(raw, repo_root=repo_root)
    if not isinstance(resolved, H6PrefixResolvedConfig):
        raise ValueError("configuration is not an H6 Prefix configuration")
    return resolved


def resolve_h6_prediction_config(
    raw: Mapping[str, object], *, repo_root: Path
) -> H6PredictionResolvedConfig:
    """Resolve H6 Prediction through the single public configuration resolver."""

    resolved = resolve_config(raw, repo_root=repo_root)
    if not isinstance(resolved, H6PredictionResolvedConfig):
        raise ValueError("configuration is not an H6 Prediction configuration")
    return resolved


def _resolve_h1_h5_config(
    raw: Mapping[str, object], *, repo_root: Path
) -> ResolvedConfig:
    """Validate and freeze one implemented ordered H1--H5 gate prefix."""
    root = _require_mapping(raw, "config")
    if "h4" in root or "h5" in root:
        expected_root_keys = _ROOT_KEYS_WITH_H4_H5
    else:
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
    coupled = validation.gates == _H5_GATES
    if coupled:
        if "h4" not in root or "h5" not in root:
            raise ValueError(
                "h4 and h5 are both required for the coupled H1--H5 prefix"
            )
        h4 = resolve_h4_validation_config(
            _require_mapping(root["h4"], "h4")
        )
        h5 = _resolve_h5_validation_config(
            _require_mapping(root["h5"], "h5")
        )
    else:
        if "h4" in root or "h5" in root:
            raise ValueError(
                "h4 and h5 must be present exactly for the coupled H1--H5 prefix"
            )
        h4 = None
        h5 = None

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
        h4=h4,
        h5=h5,
    )
    if h4 is not None:
        projected_h4 = json.dumps(
            json.loads(canonical_json)["h4"],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        if (
            projected_h4 != h4.canonical_json
            or hashlib.sha256(projected_h4.encode("utf-8")).hexdigest()
            != h4.config_sha256
        ):
            raise ValueError("full resolved configuration H4 projection drifted")
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
        h4=h4,
        h5=h5,
    )


def _resolve_h5_validation_config(raw_h5: Mapping[str, object]) -> H5ValidationConfig:
    """Resolve H5 only from frozen declarations; never inspect fixture bytes."""
    root = _require_mapping(raw_h5, "h5")
    _validate_keys(root, _H5_KEYS, "h5")

    def exact_list(name: str, expected: tuple[object, ...]) -> tuple[object, ...]:
        return _require_exact_list(root[name], expected, f"h5.{name}")

    rules = tuple(H5UpdateRule)
    labels = (
        UpdateLabel.EXACT_COORDINATE,
        UpdateLabel.GENERALIZED_EM,
        UpdateLabel.NATURAL_GRADIENT_PROPOSAL,
    )
    resolved_rules = tuple(
        H5UpdateRule(value)
        for value in exact_list(
            "enabled_update_rules", tuple(item.value for item in rules)
        )
    )
    raw_labels = root["enabled_update_labels"]
    if type(raw_labels) is not list:
        raise ValueError("h5.enabled_update_labels must be an exact list")
    if UpdateLabel.VALID_MM.value in raw_labels and root["mm_proof_artifact"] is None:
        raise ValueError("VALID_MM requires a revision-bound MM proof artifact")
    resolved_labels = tuple(
        UpdateLabel(value)
        for value in _require_exact_list(
            raw_labels, tuple(item.value for item in labels),
            "h5.enabled_update_labels",
        )
    )
    if root["mm_proof_artifact"] is not None:
        raise ValueError("H5 v1 does not admit an MM proof artifact")
    payload = {
        "schema_version": _require_exact(root["schema_version"], "h5-validation-config-v1", "h5.schema_version"),
        "fixture_id": _require_exact(root["fixture_id"], "h5-conditional-update-v1", "h5.fixture_id"),
        "fixture_schema_version": _require_exact(root["fixture_schema_version"], 1, "h5.fixture_schema_version"),
        "recognition_family": _require_exact(root["recognition_family"], "continuous_mean_field_conditional_categorical", "h5.recognition_family"),
        "h1_fixture_id": _require_exact(root["h1_fixture_id"], "h1-v1", "h5.h1_fixture_id"),
        "h1_fixture_raw_sha256": _require_exact(root["h1_fixture_raw_sha256"], H5_H1_FIXTURE_RAW_SHA256, "h5.h1_fixture_raw_sha256"),
        "update_spec_raw_sha256": _require_exact(root["update_spec_raw_sha256"], EXPECTED_H5_UPDATE_SPEC_RAW_SHA256, "h5.update_spec_raw_sha256"),
        "update_spec_canonical_sha256": _require_exact(root["update_spec_canonical_sha256"], H5_UPDATE_SPEC_CANONICAL_SHA256, "h5.update_spec_canonical_sha256"),
        "objective_schema_sha256": _require_exact(root["objective_schema_sha256"], H5_OBJECTIVE_SCHEMA_SHA256, "h5.objective_schema_sha256"),
        "factor_input_schema_version": _require_exact(root["factor_input_schema_version"], H5_FACTOR_INPUT_SCHEMA_VERSION, "h5.factor_input_schema_version"),
        "factor_input_schema_sha256": _require_exact(root["factor_input_schema_sha256"], H5_FACTOR_INPUT_SCHEMA_SHA256, "h5.factor_input_schema_sha256"),
        "factor_universe": exact_list("factor_universe", H5_FACTOR_UNIVERSE),
        "recognition_coordinate_universe": exact_list("recognition_coordinate_universe", H5_RECOGNITION_COORDINATE_UNIVERSE),
        "model_block_universe": exact_list("model_block_universe", H5_MODEL_BLOCK_UNIVERSE),
        "enabled_update_rules": resolved_rules,
        "enabled_update_labels": resolved_labels,
        "positive_case_ids": exact_list("positive_case_ids", H5_POSITIVE_CASE_IDS),
        "control_ids": exact_list("control_ids", H5_CONTROL_IDS),
        "quadrature_orders": exact_list("quadrature_orders", H5_QUADRATURE_ORDERS),
        "allowance_policy": _require_exact(root["allowance_policy"], "deterministic_convergence_plus_rounding_v1", "h5.allowance_policy"),
        "rounding_constant": _require_exact(root["rounding_constant"], 4096, "h5.rounding_constant"),
        "stochastic_contribution": _require_exact(root["stochastic_contribution"], 0.0, "h5.stochastic_contribution"),
        "epsilon_delta_formula": _require_exact(root["epsilon_delta_formula"], "before_total+after_total+subtraction_rounding", "h5.epsilon_delta_formula"),
        "mm_proof_artifact": None,
    }
    canonical_payload = {
        **payload,
        "enabled_update_rules": tuple(item.value for item in resolved_rules),
        "enabled_update_labels": tuple(item.value for item in resolved_labels),
    }
    canonical = json.dumps(
        canonical_payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return H5ValidationConfig(
        **payload,
        canonical_json=canonical,
        config_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
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
    | tuple[
        Literal["H1"], Literal["H2"], Literal["H3"], Literal["H4"], Literal["H5"]
    ]
):
    if type(value) is not list or value not in (
        ["H1"],
        ["H1", "H2"],
        ["H1", "H2", "H3"],
        ["H1", "H2", "H3", "H4", "H5"],
    ):
        raise ValueError(
            "validation.gates must equal ['H1'], ['H1', 'H2'], or "
            "['H1', 'H2', 'H3'], or ['H1', 'H2', 'H3', 'H4', 'H5']"
        )
    return tuple(value)  # type: ignore[return-value]


def _resolve_h3(
    root: Mapping[str, object], gates: tuple[str, ...]
) -> H3ValidationConfig | None:
    requested = gates[:3] == _H3_GATES and len(gates) >= 3
    present = "h3" in root
    if requested != present:
        raise ValueError(
            "h3 must be present exactly when validation.gates equals "
            "an implemented prefix containing ['H1', 'H2', 'H3']"
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
    h4: H4ValidationConfig | None,
    h5: H5ValidationConfig | None,
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
    if (h4 is None) != (h5 is None):
        raise ValueError("canonical H4/H5 sections must appear together")
    if h4 is not None and h5 is not None:
        payload["h4"] = json.loads(h4.canonical_json)
        payload["h5"] = json.loads(h5.canonical_json)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
