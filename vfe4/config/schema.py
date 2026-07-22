"""Frozen records that define the supported ordered H1/H2/H3 configuration."""

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
