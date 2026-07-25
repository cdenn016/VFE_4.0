"""Frozen records for supported ordered verification prefixes through H8."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from vfe4.types.h3 import (
    H3DecisionConfig,
    H3InitializationConfig,
    H3OptimizationConfig,
    H3RecognitionFamily,
)
from vfe4.types.h4 import (
    H4_PRIMARY_TIMED_AB_TOTAL,
    H4_PRIMARY_TIMED_BALANCE,
    H4_PRIMARY_TIMED_BA_TOTAL,
    H4_PROBLEM_SEEDS,
    H4ProblemKind,
    H4SolveProtocol,
)
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
from vfe4.types.h6 import (
    AdamWPolicyRecord,
    ArmConfig,
    ArmId,
    CapacityAllocation,
    EndpointSmcProtocol,
    H6PrefixProfilePair,
    H6TrainingSchedule,
)
from vfe4.types.h7 import (
    H7_CONTROL_IDS,
    H7_REQUIRED_TRIAL_IDS,
    H7ControlId,
    H7TrialSpec,
)
from vfe4.types.h8 import (
    CurrentH8PrerequisiteRefs,
    H8_CORRECTNESS_CASES,
    H8_H7_PLAN_SHA256,
    H8_INTERPRETATION_SHA256,
    H8_PROBLEM_DRAW_SCHEMA_SHA256,
    H8_PRODUCTION_SAMPLE_SEED_PAIRS,
    H8_PRODUCTION_SEEDS,
    H8_PROFILER_API_CONTRACT_SHA256,
    H8_PROFILER_MEMORY_SOURCE_SHA256,
    H8_PROFILER_SOURCE_SHA256,
)

from vfe4.types.updates import H5_RULE_CONTRACTS, H5UpdateRule, UpdateLabel
from vfe4.validation.h5_update_spec import EXPECTED_H5_UPDATE_SPEC_RAW_SHA256


_H7_H1_FIXTURE_RAW_SHA256 = (
    "388e38cc8c16d8b5e2c61919c1e712a134d88fb0bbd8ec1f2939b9859c9a583b"
)
_H7_FIXTURE_RAW_SHA256 = (
    "d2ed126c3deab3eafc7b94f81f13152be63eb854e3e62e03f1494dea163666d4"
)
_H7_DENSITY_PROBE_TABLE_RAW_SHA256 = (
    "4857af296e84a33f47964c3bca65e0d42967009aa5c79a52bcc98d6db04382c6"
)
_H7_DENSITY_PROBE_SET_SHA256 = (
    "f002618a32270846c83fedf9888bc06a01d755019edc6421526aee33f89fb42f"
)


@dataclass(frozen=True)
class RunConfig:
    mode: Literal["verify"]
    seed: int
    device: Literal["cpu"]
    dtype: Literal["float64"]
    deterministic: bool


@dataclass(frozen=True)
class ValidationConfig:
    gates: (
        tuple[Literal["H1"]]
        | tuple[Literal["H1"], Literal["H2"]]
        | tuple[Literal["H1"], Literal["H2"], Literal["H3"]]
        | tuple[
            Literal["H1"], Literal["H2"], Literal["H3"],
            Literal["H4"], Literal["H5"],
        ]
        | tuple[
            Literal["H1"], Literal["H2"], Literal["H3"],
            Literal["H4"], Literal["H5"], Literal["H6-Prefix"],
            Literal["H7"],
        ]
        | tuple[
            Literal["H1"], Literal["H2"], Literal["H3"],
            Literal["H4"], Literal["H5"], Literal["H6-Prefix"],
            Literal["H7"], Literal["H8"],
        ]
    )
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
class H3ValidationConfig:
    coupled_fixture_id: Literal["h3-coupled-v1"]
    coupled_expected_sha256: Literal[
        "6779f5b0a2e27aa5e203764bcc4d84c1b1daedb9423fcefdf28dce3cf7e40e03"
    ]
    zero_control_fixture_id: Literal["h3-zero-control-v1"]
    zero_control_expected_sha256: Literal[
        "ba600e09e0ae7e2b7576fbf4446a8e5b38a605c7621eb0cd5586689dccb89acf"
    ]
    recognition_families: tuple[H3RecognitionFamily, H3RecognitionFamily]
    common_initialization: H3InitializationConfig
    optimization_operation: Literal["maximize_direct_h3_elbo_lbfgs"]
    expected_autograd_scope: Literal["h3_recognition_only"]
    optimizer: H3OptimizationConfig
    decision: H3DecisionConfig
    solver_allowance_nats: float
    threshold_decision_rule: Literal["signed_margin_three_way"]
    minimum_resolved_fraction: float
    coupled_gap_inconclusive_obligation: Literal[
        "resolve coupled gap threshold outside allowance"
    ]
    structured_closure_inconclusive_obligation: Literal[
        "resolve structured closure threshold outside allowance"
    ]


@dataclass(frozen=True, slots=True)
class H4TraversalConfig:
    horizons: tuple[int, int, int]
    seeds: tuple[int, ...]
    kinds: tuple[H4ProblemKind, H4ProblemKind]
    d_z: Literal[4]
    d_m: Literal[4]
    dimensions: tuple[int, int, int]
    primary_horizon: Literal[31]
    primary_kind: Literal["coupled"]
    primary_dimension: Literal[256]

    def __post_init__(self) -> None:
        if (
            self.horizons != (7, 15, 31)
            or self.seeds != H4_PROBLEM_SEEDS
            or self.kinds != ("coupled", "zero_control")
            or (self.d_z, self.d_m) != (4, 4)
            or self.dimensions != (64, 128, 256)
            or (self.primary_horizon, self.primary_kind, self.primary_dimension)
            != (31, "coupled", 256)
            or any(type(item) is not int for item in (*self.horizons, *self.seeds, *self.dimensions, self.d_z, self.d_m, self.primary_horizon, self.primary_dimension))
        ):
            raise ValueError("H4 traversal configuration is frozen")


@dataclass(frozen=True, slots=True)
class H4TimingConfig:
    parity_expression: Literal[
        "(horizon_index + seed_index + kind_index + pair_index) % 2 == 0"
    ]
    warmup_pair_indices: tuple[int, int, int]
    timed_pair_indices: tuple[int, ...]
    timed_repetitions_per_problem: Literal[11]
    warmups_count_toward_balance: Literal[False]
    primary_timed_balance: tuple[tuple[int, int, int], ...]
    primary_5_ab_6_ba_rows: Literal[10]
    primary_6_ab_5_ba_rows: Literal[10]
    primary_timed_ab_total: Literal[110]
    primary_timed_ba_total: Literal[110]
    clock: Literal["time.perf_counter_ns"]
    timer_boundary: Literal["fresh_native_solver_call_v1"]
    between_repetitions: Literal["timer_reads_and_preallocated_assignments_only"]

    def __post_init__(self) -> None:
        expected = (
            "(horizon_index + seed_index + kind_index + pair_index) % 2 == 0",
            (0, 1, 2), tuple(range(3, 14)), 11, False,
            H4_PRIMARY_TIMED_BALANCE, 10, 10,
            H4_PRIMARY_TIMED_AB_TOTAL, H4_PRIMARY_TIMED_BA_TOTAL,
            "time.perf_counter_ns", "fresh_native_solver_call_v1",
            "timer_reads_and_preallocated_assignments_only",
        )
        if tuple(getattr(self, field) for field in self.__dataclass_fields__) != expected:
            raise ValueError("H4 timing configuration is frozen")


@dataclass(frozen=True, slots=True)
class H4BootstrapConfig:
    seed: Literal[20260721]
    replicates: Literal[100000]
    inferential_units: Literal[20]
    index_low: Literal[0]
    index_high: Literal[20]
    endpoint: Literal[False]
    index_dtype: Literal["<i8"]
    index_shape: tuple[Literal[100000], Literal[20]]
    statistic: Literal["mean_log_seed_ratio"]
    percentiles: tuple[float, float]
    percentile_method: Literal["linear"]
    percentile_space: Literal["log_then_exp"]
    digest_domain: Literal["vfe4.h4.bootstrap-indices.v1"]
    expected_index_sha256: Literal[
        "a254e18bccc519a719e9f4b409f45cc9ae4a2a321903531cd8fd73433687cd14"
    ]

    def __post_init__(self) -> None:
        if (
            (self.seed, self.replicates, self.inferential_units, self.index_low, self.index_high)
            != (20260721, 100000, 20, 0, 20)
            or self.endpoint is not False
            or self.index_dtype != "<i8"
            or self.index_shape != (100000, 20)
            or self.statistic != "mean_log_seed_ratio"
            or self.percentiles != (2.5, 97.5)
            or self.percentile_method != "linear"
            or self.percentile_space != "log_then_exp"
            or self.digest_domain != "vfe4.h4.bootstrap-indices.v1"
            or self.expected_index_sha256 != "a254e18bccc519a719e9f4b409f45cc9ae4a2a321903531cd8fd73433687cd14"
            or any(type(value) is not int for value in (self.seed, self.replicates, self.inferential_units, self.index_low, self.index_high, *self.index_shape))
            or any(type(value) is not float for value in self.percentiles)
        ):
            raise ValueError("H4 bootstrap configuration is frozen")


@dataclass(frozen=True, slots=True)
class H4ConditionEnvelopeConfig:
    posterior_minimum_eigenvalue: float
    posterior_maximum_eigenvalue: float
    posterior_maximum_condition_number: float
    posterior_minimum_cholesky_pivot: float
    posterior_maximum_mean_infinity_norm: float
    innovation_minimum_eigenvalue: float
    innovation_maximum_eigenvalue: float
    innovation_maximum_condition_number: float
    inclusive: Literal[True]

    def __post_init__(self) -> None:
        values = tuple(getattr(self, name) for name in tuple(self.__dataclass_fields__)[:-1])
        if (
            any(type(value) is not float or not math.isfinite(value) for value in values)
            or values != (1.0e-6, 1.0e6, 1.0e8, 1.0e-3, 16.0, 1.0e-6, 1.0e6, 1.0e8)
            or self.inclusive is not True
        ):
            raise ValueError("H4 condition envelope is frozen and inclusive")


@dataclass(frozen=True, slots=True)
class H4AllowanceConfig:
    float64_epsilon: float
    rounding_constant: Literal[4096]
    solver_relative_budget: float
    maximum_allowance_scale_fraction: float
    decisiveness_comparison: Literal["strict_less_than"]
    element_stream_domain: Literal["vfe4.h4.allowance-element-stream.v1"]
    maximum_chunk_rows: Literal[4096]

    def __post_init__(self) -> None:
        if (
            type(self.float64_epsilon) is not float
            or self.float64_epsilon != 2.220446049250313e-16
            or type(self.rounding_constant) is not int or self.rounding_constant != 4096
            or type(self.solver_relative_budget) is not float or self.solver_relative_budget != 1.0e-9
            or type(self.maximum_allowance_scale_fraction) is not float or self.maximum_allowance_scale_fraction != 1.0e-4
            or self.decisiveness_comparison != "strict_less_than"
            or self.element_stream_domain != "vfe4.h4.allowance-element-stream.v1"
            or type(self.maximum_chunk_rows) is not int or self.maximum_chunk_rows != 4096
        ):
            raise ValueError("H4 allowance configuration is frozen")


@dataclass(frozen=True, slots=True)
class H4EnvironmentConfig:
    device: Literal["cpu"]
    dtype: Literal["float64"]
    intra_op_threads: Literal[1]
    alter_inter_op_threads: Literal[False]
    cuda_expected: Literal[False]
    gc_policy: Literal["restore_exact_prior_enabled_state"]
    power_policy_field_order: tuple[
        Literal["active_power_scheme"], Literal["cpu_frequency_governor"],
        Literal["energy_performance_preference"], Literal["low_power_mode"],
    ]
    power_policy_capture: Literal["typed_best_effort_outside_timing"]

    def __post_init__(self) -> None:
        if (
            (self.device, self.dtype, self.intra_op_threads, self.alter_inter_op_threads, self.cuda_expected, self.gc_policy)
            != ("cpu", "float64", 1, False, False, "restore_exact_prior_enabled_state")
            or self.power_policy_field_order != (
                "active_power_scheme", "cpu_frequency_governor",
                "energy_performance_preference", "low_power_mode",
            )
            or self.power_policy_capture != "typed_best_effort_outside_timing"
            or type(self.intra_op_threads) is not int
        ):
            raise ValueError("H4 environment configuration is frozen")


@dataclass(frozen=True, slots=True)
class H4ValidationConfig:
    schema_version: Literal["h4-validation-config-v1"]
    solve_protocol: H4SolveProtocol
    traversal: H4TraversalConfig
    timing: H4TimingConfig
    bootstrap: H4BootstrapConfig
    condition_envelope: H4ConditionEnvelopeConfig
    allowance: H4AllowanceConfig
    environment: H4EnvironmentConfig
    primary_effect_threshold: float
    maximum_validation_payload_bytes: Literal[67108864]
    canonical_json: str
    config_sha256: str

    def __post_init__(self) -> None:
        if type(self.solve_protocol) is not H4SolveProtocol:
            raise ValueError("solve_protocol must be the exact public H4SolveProtocol")
        H4SolveProtocol(**asdict(self.solve_protocol))
        nested_types = (
            (self.traversal, H4TraversalConfig), (self.timing, H4TimingConfig),
            (self.bootstrap, H4BootstrapConfig),
            (self.condition_envelope, H4ConditionEnvelopeConfig),
            (self.allowance, H4AllowanceConfig),
            (self.environment, H4EnvironmentConfig),
        )
        if any(type(value) is not expected for value, expected in nested_types):
            raise ValueError("H4 validation config requires exact nested records")
        if (
            self.schema_version != "h4-validation-config-v1"
            or self.solve_protocol.dtype != self.environment.dtype
            or self.solve_protocol.device != self.environment.device
            or self.solve_protocol.solver_relative_budget != self.allowance.solver_relative_budget
            or self.solve_protocol.factor_passes != 1
            or len(self.timing.timed_pair_indices) != self.timing.timed_repetitions_per_problem
            or self.timing.warmup_pair_indices != (0, 1, 2)
            or self.timing.timed_pair_indices != tuple(range(3, 14))
            or self.bootstrap.inferential_units != self.bootstrap.index_high
            or self.bootstrap.index_high != len(self.traversal.seeds)
            or self.traversal.dimensions != tuple(
                (horizon + 1) * (self.traversal.d_z + self.traversal.d_m)
                for horizon in self.traversal.horizons
            )
            or self.traversal.primary_dimension != (
                (self.traversal.primary_horizon + 1)
                * (self.traversal.d_z + self.traversal.d_m)
            )
            or type(self.primary_effect_threshold) is not float
            or self.primary_effect_threshold != 0.80
            or type(self.maximum_validation_payload_bytes) is not int
            or self.maximum_validation_payload_bytes != 67_108_864
        ):
            raise ValueError("H4 validation cross-field identity failed")
        payload = _h4_validation_payload(self)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if type(self.canonical_json) is not str or self.canonical_json != canonical:
            raise ValueError("H4 canonical JSON does not match resolved fields")
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if self.config_sha256 != digest:
            raise ValueError("H4 config SHA-256 does not match canonical JSON")


H5_POSITIVE_CASE_IDS = (
    "exact_gaussian_e_coordinate",
    "exact_categorical_source_coordinate",
    "exact_gaussian_m_coordinate_fixed_recognition",
    "accepted_resolved_generalized_em",
    "rejected_proposal_rollback",
)
H5_CONTROL_IDS = (
    "child_factor_omission_detected",
    "emission_factor_omission_detected",
    "unresolved_gem_acceptance_detected",
    "natural_gradient_mislabel_detected",
    "rejection_mutation_detected",
    "changed_input_equal_value_detected",
    "changed_value_unchanged_input_not_affected",
)
H5_UPDATE_SPEC_CANONICAL_SHA256 = (
    "0e4e870dd725aeaec77ffd128ba85dbf619df5b0261b2178e6a115a8970715d6"
)


@dataclass(frozen=True, slots=True)
class H5ValidationConfig:
    schema_version: Literal["h5-validation-config-v1"]
    fixture_id: Literal["h5-conditional-update-v1"]
    fixture_schema_version: Literal[1]
    recognition_family: Literal["continuous_mean_field_conditional_categorical"]
    h1_fixture_id: Literal["h1-v1"]
    h1_fixture_raw_sha256: str
    update_spec_raw_sha256: str
    update_spec_canonical_sha256: str
    objective_schema_sha256: str
    factor_input_schema_version: Literal["h5-factor-input-v1"]
    factor_input_schema_sha256: str
    factor_universe: tuple[str, ...]
    recognition_coordinate_universe: tuple[str, ...]
    model_block_universe: tuple[str, ...]
    enabled_update_rules: tuple[H5UpdateRule, ...]
    enabled_update_labels: tuple[UpdateLabel, ...]
    positive_case_ids: tuple[str, ...]
    control_ids: tuple[str, ...]
    quadrature_orders: tuple[Literal[21], Literal[17]]
    allowance_policy: Literal["deterministic_convergence_plus_rounding_v1"]
    rounding_constant: Literal[4096]
    stochastic_contribution: float
    epsilon_delta_formula: Literal[
        "before_total+after_total+subtraction_rounding"
    ]
    mm_proof_artifact: None
    canonical_json: str
    config_sha256: str

    def __post_init__(self) -> None:
        rules = tuple(H5UpdateRule)
        labels: list[UpdateLabel] = []
        for rule in rules:
            producer = H5_RULE_CONTRACTS[rule][0]
            if producer not in labels:
                labels.append(producer)
        expected_labels = tuple(labels)
        expected = (
            self.schema_version == "h5-validation-config-v1"
            and self.fixture_id == "h5-conditional-update-v1"
            and type(self.fixture_schema_version) is int
            and self.fixture_schema_version == 1
            and self.recognition_family
            == "continuous_mean_field_conditional_categorical"
            and self.h1_fixture_id == "h1-v1"
            and self.h1_fixture_raw_sha256 == H5_H1_FIXTURE_RAW_SHA256
            and self.update_spec_raw_sha256 == EXPECTED_H5_UPDATE_SPEC_RAW_SHA256
            and self.update_spec_canonical_sha256
            == H5_UPDATE_SPEC_CANONICAL_SHA256
            and self.objective_schema_sha256 == H5_OBJECTIVE_SCHEMA_SHA256
            and self.factor_input_schema_version == H5_FACTOR_INPUT_SCHEMA_VERSION
            and self.factor_input_schema_sha256 == H5_FACTOR_INPUT_SCHEMA_SHA256
            and self.factor_universe == H5_FACTOR_UNIVERSE
            and self.recognition_coordinate_universe
            == H5_RECOGNITION_COORDINATE_UNIVERSE
            and self.model_block_universe == H5_MODEL_BLOCK_UNIVERSE
            and self.enabled_update_rules == rules
            and self.enabled_update_labels == expected_labels
            and UpdateLabel.VALID_MM not in self.enabled_update_labels
            and self.positive_case_ids == H5_POSITIVE_CASE_IDS
            and self.control_ids == H5_CONTROL_IDS
            and self.quadrature_orders == H5_QUADRATURE_ORDERS
            and self.allowance_policy
            == "deterministic_convergence_plus_rounding_v1"
            and type(self.rounding_constant) is int
            and self.rounding_constant == 4096
            and type(self.stochastic_contribution) is float
            and self.stochastic_contribution == 0.0
            and self.epsilon_delta_formula
            == "before_total+after_total+subtraction_rounding"
            and self.mm_proof_artifact is None
        )
        if not expected:
            raise ValueError("H5 validation configuration is not the frozen v1 contract")
        payload = _h5_validation_payload(self)
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        if self.canonical_json != canonical:
            raise ValueError("H5 canonical JSON does not match resolved fields")
        if self.config_sha256 != hashlib.sha256(canonical.encode("utf-8")).hexdigest():
            raise ValueError("H5 config SHA-256 does not match canonical JSON")


def _h5_validation_payload(config: H5ValidationConfig) -> dict[str, object]:
    return {
        "schema_version": config.schema_version,
        "fixture_id": config.fixture_id,
        "fixture_schema_version": config.fixture_schema_version,
        "recognition_family": config.recognition_family,
        "h1_fixture_id": config.h1_fixture_id,
        "h1_fixture_raw_sha256": config.h1_fixture_raw_sha256,
        "update_spec_raw_sha256": config.update_spec_raw_sha256,
        "update_spec_canonical_sha256": config.update_spec_canonical_sha256,
        "objective_schema_sha256": config.objective_schema_sha256,
        "factor_input_schema_version": config.factor_input_schema_version,
        "factor_input_schema_sha256": config.factor_input_schema_sha256,
        "factor_universe": config.factor_universe,
        "recognition_coordinate_universe": config.recognition_coordinate_universe,
        "model_block_universe": config.model_block_universe,
        "enabled_update_rules": tuple(item.value for item in config.enabled_update_rules),
        "enabled_update_labels": tuple(item.value for item in config.enabled_update_labels),
        "positive_case_ids": config.positive_case_ids,
        "control_ids": config.control_ids,
        "quadrature_orders": config.quadrature_orders,
        "allowance_policy": config.allowance_policy,
        "rounding_constant": config.rounding_constant,
        "stochastic_contribution": config.stochastic_contribution,
        "epsilon_delta_formula": config.epsilon_delta_formula,
        "mm_proof_artifact": config.mm_proof_artifact,
    }


def _h4_validation_payload(config: H4ValidationConfig) -> dict[str, object]:
    return {
        "schema_version": config.schema_version,
        "solve_protocol": asdict(config.solve_protocol),
        "traversal": asdict(config.traversal),
        "timing": asdict(config.timing),
        "bootstrap": asdict(config.bootstrap),
        "condition_envelope": asdict(config.condition_envelope),
        "allowance": asdict(config.allowance),
        "environment": asdict(config.environment),
        "primary_effect_threshold": config.primary_effect_threshold,
        "maximum_validation_payload_bytes": config.maximum_validation_payload_bytes,
    }


@dataclass(frozen=True)
class ArtifactConfig:
    run_root: Path


@dataclass(frozen=True)
class H6SourceIdentity:
    git_head: str
    dirty_digest: str
    source_sha256: str


@dataclass(frozen=True)
class H6ArchiveMemberExpectation:
    path: Literal[
        "wikitext-2-raw/wiki.train.raw",
        "wikitext-2-raw/wiki.valid.raw",
        "wikitext-2-raw/wiki.test.raw",
    ]
    compressed_size: int
    uncompressed_size: int
    compression_method: Literal[0, 8]
    crc32: int
    raw_sha256: str


@dataclass(frozen=True)
class H6ObservedArchive:
    archive_byte_length: int
    archive_sha256: str
    members: tuple[H6ArchiveMemberExpectation, ...]


@dataclass(frozen=True)
class H6DataConfig:
    schema_version: Literal["h6-data-config-v1"]
    source_url: Literal[
        "https://s3.amazonaws.com/research.metamind.io/wikitext/"
        "wikitext-2-raw-v1.zip"
    ]
    max_archive_bytes: Literal[16777216]
    member_paths: tuple[str, ...]
    allowed_compression_methods: tuple[Literal[0, 8], ...]
    max_member_bytes: Literal[16777216]
    max_total_uncompressed_bytes: Literal[33554432]
    max_compression_ratio: Literal[100]
    observed_archive: H6ObservedArchive | None


@dataclass(frozen=True)
class H1PrefixPriorResolvedConfig:
    schema_version: Literal["h1-prefix-prior-config-v1"]
    operation: Literal["H1-Prefix-Prior"]
    source: H6SourceIdentity
    fixture_id: Literal["h1-prefix-prior-v1"]
    fixture_sha256: str
    base_fixture_sha256: str
    generative_factor_schema_sha256: str
    horizon: Literal[2]
    d_z: Literal[1]
    d_m: Literal[1]
    vocabulary_size: Literal[3]
    state_parent_sets: tuple[tuple[int, ...], tuple[int, ...]]
    model_parent_sets: tuple[tuple[int, ...], tuple[int, ...]]
    latent_projection_policy: Literal["exact_zero"]
    prefix_policy: Literal["strictly_prior_tokens"]
    quadrature_order: Literal[21]
    convergence_check_order: Literal[17]
    maximum_convergence_estimate: Literal[1e-9]
    artifact_root: Path
    canonical_json: str
    config_sha256: str


@dataclass(frozen=True)
class H6PrefixResolvedConfig:
    schema_version: Literal["h6-prefix-config-v1"]
    operation: Literal["H6-Prefix"]
    source: H6SourceIdentity
    execution_mode: Literal["focused_subset", "authorized_full"]
    profiles: tuple[H6PrefixProfilePair, ...]
    authorization_sha256: str | None
    artifact_root: Path
    canonical_json: str
    config_sha256: str


@dataclass(frozen=True)
class H6PredictionResolvedConfig:
    schema_version: Literal["h6-prediction-config-v1"]
    operation: Literal["H6-Prediction"]
    source: H6SourceIdentity
    data: H6DataConfig
    correctness_manifests: tuple[tuple[str, str], ...]
    h1_prefix_prior_manifest_sha256: str
    smc_validation_manifest_sha256: str
    prefix_certificate_set_sha256: str
    h5_update_binding_sha256: str
    training_schedule: H6TrainingSchedule
    critical_values_sha256: str
    endpoint_smc_protocol: EndpointSmcProtocol
    attribution_matrix_sha256: str
    matching_set_sha256: str
    data_identity_sha256: str
    access_policy_sha256: str
    artifact_root: Path
    canonical_json: str
    config_sha256: str


@dataclass(frozen=True, slots=True)
class H6ArmMatchingResolvedConfig:
    """Standalone typed Task 7 projection; not an H6-Prediction v1 section."""

    schema_version: Literal["h6-arm-matching-config-v1"]
    operation: Literal["H6-Arm-Matching"]
    arm_configs: tuple[
        ArmConfig,
        ArmConfig,
        ArmConfig,
        ArmConfig,
        ArmConfig,
        ArmConfig,
    ]
    adamw_policy: AdamWPolicyRecord
    reference_allocation: CapacityAllocation
    emission_width_candidates: tuple[int, int, int, int]
    latent_width_candidates: tuple[int, int, int, int]
    recognition_width_candidates: tuple[int, int, int]
    parameter_relative_tolerance: float
    flop_relative_tolerance: float
    matching_schedule_sha256: str
    canonical_json: str
    config_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != "h6-arm-matching-config-v1"
            or self.operation != "H6-Arm-Matching"
            or type(self.arm_configs) is not tuple
            or len(self.arm_configs) != 6
            or any(type(item) is not ArmConfig for item in self.arm_configs)
            or tuple(item.arm for item in self.arm_configs) != tuple(ArmId)
            or type(self.adamw_policy) is not AdamWPolicyRecord
            or type(self.reference_allocation) is not CapacityAllocation
            or (
                self.reference_allocation.emission_width,
                self.reference_allocation.latent_width,
                self.reference_allocation.recognition_width,
            )
            != (64, 16, 64)
            or self.emission_width_candidates != (48, 64, 80, 96)
            or self.latent_width_candidates != (8, 16, 24, 32)
            or self.recognition_width_candidates != (32, 64, 96)
            or type(self.parameter_relative_tolerance) is not float
            or self.parameter_relative_tolerance != 0.01
            or type(self.flop_relative_tolerance) is not float
            or self.flop_relative_tolerance != 0.05
            or type(self.matching_schedule_sha256) is not str
            or len(self.matching_schedule_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.matching_schedule_sha256
            )
        ):
            raise ValueError("H6 arm-matching configuration is not frozen")
        a2_payload = self.arm_configs[2].semantic_payload()
        a5_payload = self.arm_configs[5].semantic_payload()
        if (
            a2_payload["map_mode"]
            != "generic_fixed_frame_non_coboundary"
            or a5_payload["map_mode"] != "shared_vertex_coboundary"
            or any(
                a2_payload[name] != a5_payload[name]
                for name in a2_payload
                if name != "map_mode"
            )
        ):
            raise ValueError(
                "A2 and A5 must differ semantically only in map_mode"
            )
        payload = {
            "schema_version": self.schema_version,
            "operation": self.operation,
            "arm_config_sha256": tuple(
                item.config_sha256 for item in self.arm_configs
            ),
            "optimizer_policy_sha256": (
                self.adamw_policy.optimizer_policy_sha256
            ),
            "reference_allocation_sha256": (
                self.reference_allocation.allocation_sha256
            ),
            "emission_width_candidates": self.emission_width_candidates,
            "latent_width_candidates": self.latent_width_candidates,
            "recognition_width_candidates": (
                self.recognition_width_candidates
            ),
            "parameter_relative_tolerance": (
                self.parameter_relative_tolerance
            ),
            "flop_relative_tolerance": self.flop_relative_tolerance,
            "matching_schedule_sha256": self.matching_schedule_sha256,
        }
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        if self.canonical_json != canonical:
            raise ValueError(
                "H6 arm-matching canonical JSON does not match fields"
            )
        if self.config_sha256 != hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest():
            raise ValueError(
                "H6 arm-matching config SHA-256 does not match fields"
            )


def _h7_trial_payload(spec: H7TrialSpec) -> dict[str, object]:
    return {
        "trial_id": spec.trial_id,
        "role": spec.role,
        "expected_predicate": spec.expected_predicate,
        "fixture_id": spec.fixture_id,
        "frame_profile": spec.frame_profile,
        "decoder_policy": spec.decoder_policy,
        "action": {
            "kind": spec.action.kind,
            "dimension": spec.action.dimension,
            "group": spec.action.group,
            "representation": spec.action.representation,
            "elements": [
                {
                    "dtype": item.dtype,
                    "shape": item.shape,
                    "device": item.device,
                    "raw_bytes_hex": item.raw_bytes.hex(),
                    "raw_bytes_sha256": item.raw_bytes_sha256,
                    "snapshot_sha256": item.snapshot_sha256,
                }
                for item in spec.action.elements
            ],
            "action_sha256": spec.action.action_sha256,
        },
        "action_sha256": spec.action_sha256,
        "trial_sha256": spec.trial_sha256,
    }


def _h7_validation_payload(
    *,
    schema_version: str,
    required_trial_specs: tuple[H7TrialSpec, ...],
    required_control_ids: tuple[H7ControlId, ...],
    recognition_families: tuple[str, str],
    h1_fixture_raw_sha256: str,
    h7_fixture_raw_sha256: str,
    density_probe_table_raw_sha256: str,
    density_probe_set_sha256: str,
    oracle_decimal_precision: int,
    gauss_hermite_orders: tuple[int, int],
    group_norm_limit: float,
    group_inverse_norm_limit: float,
    spd_condition_limit: float,
    predecessor_keys: tuple[str, str, str],
) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "required_trial_specs": [
            _h7_trial_payload(spec) for spec in required_trial_specs
        ],
        "required_control_ids": required_control_ids,
        "recognition_families": recognition_families,
        "h1_fixture_raw_sha256": h1_fixture_raw_sha256,
        "h7_fixture_raw_sha256": h7_fixture_raw_sha256,
        "density_probe_table_raw_sha256": (
            density_probe_table_raw_sha256
        ),
        "density_probe_set_sha256": density_probe_set_sha256,
        "oracle_decimal_precision": oracle_decimal_precision,
        "gauss_hermite_orders": gauss_hermite_orders,
        "group_norm_limit": group_norm_limit,
        "group_inverse_norm_limit": group_inverse_norm_limit,
        "spd_condition_limit": spd_condition_limit,
        "predecessor_keys": predecessor_keys,
    }


@dataclass(frozen=True)
class H7ValidationConfig:
    schema_version: Literal["h7-validation-config-v1"]
    required_trial_specs: tuple[H7TrialSpec, ...]
    required_control_ids: tuple[H7ControlId, ...]
    recognition_families: tuple[
        Literal["structured_full_block"],
        Literal["factorized_diagonal_within_fiber"],
    ]
    h1_fixture_raw_sha256: str
    h7_fixture_raw_sha256: str
    density_probe_table_raw_sha256: str
    density_probe_set_sha256: str
    oracle_decimal_precision: Literal[100]
    gauss_hermite_orders: tuple[Literal[41], Literal[51]]
    group_norm_limit: float
    group_inverse_norm_limit: float
    spd_condition_limit: float
    predecessor_keys: tuple[
        Literal["h1_h5"],
        Literal["h1_prefix_prior"],
        Literal["h6_prefix"],
    ]
    canonical_json: str
    config_sha256: str

    @classmethod
    def create(
        cls,
        *,
        required_trial_specs: tuple[H7TrialSpec, ...],
        required_control_ids: tuple[H7ControlId, ...],
        recognition_families: tuple[
            Literal["structured_full_block"],
            Literal["factorized_diagonal_within_fiber"],
        ],
        h1_fixture_raw_sha256: str,
        h7_fixture_raw_sha256: str,
        density_probe_table_raw_sha256: str,
        density_probe_set_sha256: str,
        oracle_decimal_precision: Literal[100],
        gauss_hermite_orders: tuple[Literal[41], Literal[51]],
        group_norm_limit: float,
        group_inverse_norm_limit: float,
        spd_condition_limit: float,
        predecessor_keys: tuple[
            Literal["h1_h5"],
            Literal["h1_prefix_prior"],
            Literal["h6_prefix"],
        ],
    ) -> "H7ValidationConfig":
        payload = _h7_validation_payload(
            schema_version="h7-validation-config-v1",
            required_trial_specs=required_trial_specs,
            required_control_ids=required_control_ids,
            recognition_families=recognition_families,
            h1_fixture_raw_sha256=h1_fixture_raw_sha256,
            h7_fixture_raw_sha256=h7_fixture_raw_sha256,
            density_probe_table_raw_sha256=(
                density_probe_table_raw_sha256
            ),
            density_probe_set_sha256=density_probe_set_sha256,
            oracle_decimal_precision=oracle_decimal_precision,
            gauss_hermite_orders=gauss_hermite_orders,
            group_norm_limit=group_norm_limit,
            group_inverse_norm_limit=group_inverse_norm_limit,
            spd_condition_limit=spd_condition_limit,
            predecessor_keys=predecessor_keys,
        )
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return cls(
            schema_version="h7-validation-config-v1",
            required_trial_specs=required_trial_specs,
            required_control_ids=required_control_ids,
            recognition_families=recognition_families,
            h1_fixture_raw_sha256=h1_fixture_raw_sha256,
            h7_fixture_raw_sha256=h7_fixture_raw_sha256,
            density_probe_table_raw_sha256=(
                density_probe_table_raw_sha256
            ),
            density_probe_set_sha256=density_probe_set_sha256,
            oracle_decimal_precision=oracle_decimal_precision,
            gauss_hermite_orders=gauss_hermite_orders,
            group_norm_limit=group_norm_limit,
            group_inverse_norm_limit=group_inverse_norm_limit,
            spd_condition_limit=spd_condition_limit,
            predecessor_keys=predecessor_keys,
            canonical_json=canonical,
            config_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        )

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not str
            or self.schema_version != "h7-validation-config-v1"
            or type(self.required_trial_specs) is not tuple
            or any(
                type(spec) is not H7TrialSpec
                for spec in self.required_trial_specs
            )
            or tuple(
                spec.trial_id for spec in self.required_trial_specs
            )
            != H7_REQUIRED_TRIAL_IDS
            or type(self.required_control_ids) is not tuple
            or self.required_control_ids != H7_CONTROL_IDS
            or type(self.recognition_families) is not tuple
            or self.recognition_families
            != (
                "structured_full_block",
                "factorized_diagonal_within_fiber",
            )
            or type(self.oracle_decimal_precision) is not int
            or self.oracle_decimal_precision != 100
            or type(self.gauss_hermite_orders) is not tuple
            or any(type(order) is not int for order in self.gauss_hermite_orders)
            or self.gauss_hermite_orders != (41, 51)
            or type(self.group_norm_limit) is not float
            or self.group_norm_limit != 2.0
            or type(self.group_inverse_norm_limit) is not float
            or self.group_inverse_norm_limit != 2.0
            or type(self.spd_condition_limit) is not float
            or self.spd_condition_limit != 1000.0
            or type(self.predecessor_keys) is not tuple
            or self.predecessor_keys
            != ("h1_h5", "h1_prefix_prior", "h6_prefix")
            or self.h1_fixture_raw_sha256
            != _H7_H1_FIXTURE_RAW_SHA256
            or self.h7_fixture_raw_sha256 != _H7_FIXTURE_RAW_SHA256
            or self.density_probe_table_raw_sha256
            != _H7_DENSITY_PROBE_TABLE_RAW_SHA256
            or self.density_probe_set_sha256
            != _H7_DENSITY_PROBE_SET_SHA256
        ):
            raise ValueError("H7 validation configuration is frozen")
        for name in (
            "h1_fixture_raw_sha256",
            "h7_fixture_raw_sha256",
            "density_probe_table_raw_sha256",
            "density_probe_set_sha256",
            "config_sha256",
        ):
            value = getattr(self, name)
            if (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be lowercase SHA-256")
        payload = _h7_validation_payload(
            schema_version=self.schema_version,
            required_trial_specs=self.required_trial_specs,
            required_control_ids=self.required_control_ids,
            recognition_families=self.recognition_families,
            h1_fixture_raw_sha256=self.h1_fixture_raw_sha256,
            h7_fixture_raw_sha256=self.h7_fixture_raw_sha256,
            density_probe_table_raw_sha256=(
                self.density_probe_table_raw_sha256
            ),
            density_probe_set_sha256=self.density_probe_set_sha256,
            oracle_decimal_precision=self.oracle_decimal_precision,
            gauss_hermite_orders=self.gauss_hermite_orders,
            group_norm_limit=self.group_norm_limit,
            group_inverse_norm_limit=self.group_inverse_norm_limit,
            spd_condition_limit=self.spd_condition_limit,
            predecessor_keys=self.predecessor_keys,
        )
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        if self.canonical_json != canonical:
            raise ValueError("H7 canonical JSON does not match fields")
        if self.config_sha256 != hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest():
            raise ValueError("H7 config SHA-256 does not match fields")


H8_PROBLEM_DRAW_DESCRIPTOR = (
    "numpy.Generator(numpy.PCG64(problem_seed))"
    "|method=standard_normal|dtype=float64|order=C"
    "|initial:mu0[b],Q0[b,b]"
    "|transition:t=1..T:{A_m[K,K],c_m[K],Q_m[K,K],A_z[K,K],B[K,K],"
    "c_z[K],Q_z[K,K]}"
    "|recognition_initial:mu_q0[b],Q_q0[b,b]"
    "|recognition_transition:t=1..T:{A_q[b,b],c_q[b],Q_q[b,b]}"
    "|emission:t=1..T:{w[b],beta[V]}"
    "|normal_map_variance=1/dim=>multiply_standard_normal_by_1/sqrt(dim)"
    "|serialize=after_all_problem_draws_before_sample_rng"
    "|bytes=little-endian-f8-C-contiguous"
)


def _h8_validation_payload(
    *,
    schema_version: str,
    operation: str,
    choice_kind: str,
    k_semantics: str,
    coordinate_order: str,
    T: int,
    N: int,
    K: int,
    d_z: int,
    d_m: int,
    b: int,
    D: int,
    V: int,
    generator_schema: str,
    sample_schema: str,
    problem_draw_descriptor: str,
    problem_draw_schema_sha256: str,
    serialization_point: str,
    seeds: tuple[int, ...],
    production_sample_seed_pairs: tuple[tuple[int, int], ...],
    cold_repetitions: int,
    correctness_seed_table: tuple[tuple[int, int, int, int], ...],
    max_seconds: float,
    max_process_incremental_mib: int,
    max_torch_population_mib: int,
    max_rhs_width: int,
    sample_width: int,
    torch_version: str,
    profiler_memory_source_sha256: str,
    profiler_source_sha256: str,
    profiler_api_contract_sha256: str,
    interpretation_sha256: str,
    h7_plan_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "operation": operation,
        "choice_kind": choice_kind,
        "k_semantics": k_semantics,
        "coordinate_order": coordinate_order,
        "T": T,
        "N": N,
        "K": K,
        "d_z": d_z,
        "d_m": d_m,
        "b": b,
        "D": D,
        "V": V,
        "generator_schema": generator_schema,
        "sample_schema": sample_schema,
        "problem_draw_descriptor": problem_draw_descriptor,
        "problem_draw_schema_sha256": problem_draw_schema_sha256,
        "serialization_point": serialization_point,
        "seeds": seeds,
        "production_sample_seed_pairs": production_sample_seed_pairs,
        "cold_repetitions": cold_repetitions,
        "correctness_seed_table": correctness_seed_table,
        "max_seconds": max_seconds,
        "max_process_incremental_mib": max_process_incremental_mib,
        "max_torch_population_mib": max_torch_population_mib,
        "max_rhs_width": max_rhs_width,
        "sample_width": sample_width,
        "torch_version": torch_version,
        "profiler_memory_source_sha256": profiler_memory_source_sha256,
        "profiler_source_sha256": profiler_source_sha256,
        "profiler_api_contract_sha256": profiler_api_contract_sha256,
        "interpretation_sha256": interpretation_sha256,
        "h7_plan_sha256": h7_plan_sha256,
    }


def _h8_frozen_values() -> dict[str, object]:
    return {
        "schema_version": "h8-validation-config-v1",
        "operation": "H8",
        "choice_kind": "operational_preregistration_not_manuscript_theorem",
        "k_semantics": "each_channel_dimension",
        "coordinate_order": "[z_0,m_0,...,z_T,m_T]",
        "T": 128,
        "N": 129,
        "K": 20,
        "d_z": 20,
        "d_m": 20,
        "b": 40,
        "D": 5160,
        "V": 3,
        "generator_schema": "h8-synthetic-chain-v1",
        "sample_schema": "h8-pcg64-sample-v1",
        "problem_draw_descriptor": H8_PROBLEM_DRAW_DESCRIPTOR,
        "problem_draw_schema_sha256": H8_PROBLEM_DRAW_SCHEMA_SHA256,
        "serialization_point": "after_all_problem_draws_before_sample_rng",
        "seeds": H8_PRODUCTION_SEEDS,
        "production_sample_seed_pairs": H8_PRODUCTION_SAMPLE_SEED_PAIRS,
        "cold_repetitions": 5,
        "correctness_seed_table": H8_CORRECTNESS_CASES,
        "max_seconds": 60.0,
        "max_process_incremental_mib": 128,
        "max_torch_population_mib": 64,
        "max_rhs_width": 40,
        "sample_width": 1,
        "torch_version": "2.9.1",
        "profiler_memory_source_sha256": H8_PROFILER_MEMORY_SOURCE_SHA256,
        "profiler_source_sha256": H8_PROFILER_SOURCE_SHA256,
        "profiler_api_contract_sha256": H8_PROFILER_API_CONTRACT_SHA256,
        "interpretation_sha256": H8_INTERPRETATION_SHA256,
        "h7_plan_sha256": H8_H7_PLAN_SHA256,
    }


@dataclass(frozen=True, slots=True)
class H8ValidationConfig:
    """Frozen, unmeasured H8 synthetic systems protocol."""

    schema_version: Literal["h8-validation-config-v1"]
    operation: Literal["H8"]
    choice_kind: Literal["operational_preregistration_not_manuscript_theorem"]
    k_semantics: Literal["each_channel_dimension"]
    coordinate_order: Literal["[z_0,m_0,...,z_T,m_T]"]
    T: Literal[128]
    N: Literal[129]
    K: Literal[20]
    d_z: Literal[20]
    d_m: Literal[20]
    b: Literal[40]
    D: Literal[5160]
    V: Literal[3]
    generator_schema: Literal["h8-synthetic-chain-v1"]
    sample_schema: Literal["h8-pcg64-sample-v1"]
    problem_draw_descriptor: str
    problem_draw_schema_sha256: str
    serialization_point: Literal[
        "after_all_problem_draws_before_sample_rng"
    ]
    seeds: tuple[Literal[20260721], Literal[20260722], Literal[20260723]]
    production_sample_seed_pairs: tuple[
        tuple[Literal[20260721], Literal[20261721]],
        tuple[Literal[20260722], Literal[20261722]],
        tuple[Literal[20260723], Literal[20261723]],
    ]
    cold_repetitions: Literal[5]
    correctness_seed_table: tuple[tuple[int, int, int, int], ...]
    max_seconds: float
    max_process_incremental_mib: Literal[128]
    max_torch_population_mib: Literal[64]
    max_rhs_width: Literal[40]
    sample_width: Literal[1]
    torch_version: Literal["2.9.1"]
    profiler_memory_source_sha256: str
    profiler_source_sha256: str
    profiler_api_contract_sha256: str
    interpretation_sha256: str
    h7_plan_sha256: str
    canonical_json: str
    config_sha256: str

    @classmethod
    def create(cls) -> "H8ValidationConfig":
        values = _h8_frozen_values()
        canonical = json.dumps(
            _h8_validation_payload(**values),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return cls(
            **values,
            canonical_json=canonical,
            config_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        )

    def __post_init__(self) -> None:
        expected = _h8_frozen_values()
        for name in self.__dataclass_fields__:
            if name in ("canonical_json", "config_sha256"):
                continue
            if (
                type(getattr(self, name)) is not type(expected[name])
                or getattr(self, name) != expected[name]
            ):
                raise ValueError("H8 validation configuration is frozen")
        payload = _h8_validation_payload(
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name not in ("canonical_json", "config_sha256")
            }
        )
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        if self.canonical_json != canonical:
            raise ValueError("H8 canonical JSON does not match fields")
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if self.config_sha256 != digest:
            raise ValueError("H8 config SHA-256 does not match fields")


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
    h3: H3ValidationConfig | None = None
    h4: H4ValidationConfig | None = None
    h5: H5ValidationConfig | None = None
    h6_prefix: H6PrefixResolvedConfig | None = None
    h6_prediction: H6PredictionResolvedConfig | None = None
    h7: H7ValidationConfig | None = None
    h8: H8ValidationConfig | None = None
    h8_current_refs: CurrentH8PrerequisiteRefs | None = None
