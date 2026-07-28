"""Immutable WikiText-103 training protocol records.

This module is deliberately limited to the Python standard library.  It
defines scientific identities and closed records; it does not inspect an
installed tokenizer, import a model runtime, read files, or perform I/O.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Literal

from .results import GateStatus


WT103_ARM_IDS = (
    "WT103-A0-AR-v1",
    "WT103-A5-PARENT-SPECIFIC-PREFIX-COMPLETE-v1",
    "WT103-A5-FIXED-COMPLETE-v1",
    "WT103-A5-PARENT-SPECIFIC-PREFIX-EMISSION-v1",
    "WT103-A5-NOLATENT-v1",
)
WT103_GATE_IDS = (
    "SOURCE_LOCK",
    "H8_EXACT_REVISION",
    "POST_H8_READINESS",
    "OBJECTIVE",
    "PRIMARY",
    "PRIOR_CONTROL",
    "LATENT_PATH_CONTROL",
)
WT103_TUNING_CELLS = (
    (1.0e-4, 0.0),
    (1.0e-4, 1.0e-2),
    (3.0e-4, 0.0),
    (3.0e-4, 1.0e-2),
    (1.0e-3, 0.0),
    (1.0e-3, 1.0e-2),
)
WT103_TUNING_SEED_IDS = (2026072199, 2026072200)
WT103_CONFIRMATORY_SEED_IDS = tuple(range(2026072101, 2026072109))
WT103_VALIDATION_STREAM_IDS = tuple(range(8))
WT103_TEST_STREAM_IDS = tuple(range(64))
WT103_PARTICLE_COUNTS = (128, 256, 512, 1024)
WT103_A0_HIDDEN_WIDTH_CANDIDATES = (
    20,
    24,
    28,
    32,
    36,
    40,
    44,
    48,
    52,
    56,
    60,
    64,
    72,
    80,
    96,
    112,
    128,
    160,
)
WT103_FIGURE_PANEL_KEYS = (
    "training-objective-and-validation",
    "terminal-prior-nll-ppl",
    "complete-elbo-decomposition",
    "source-entropy-effective-count",
    "update-acceptance",
    "spd-health",
    "throughput-memory",
    "seed-variability",
)

_HEX = frozenset("0123456789abcdef")


def _plain(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _plain(getattr(value, item.name)) for item in fields(value)}
    if type(value) is tuple:
        return [_plain(item) for item in value]
    if type(value) is list:
        return [_plain(item) for item in value]
    if type(value) is dict:
        return {str(key): _plain(item) for key, item in value.items()}
    if type(value) in (str, int, float, bool) or value is None:
        return value
    raise ValueError(f"unsupported canonical value {type(value).__name__}")


def canonical_json_bytes(value: object) -> bytes:
    """Return stable canonical JSON bytes for a supported immutable record."""

    return json.dumps(
        _plain(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def owned_sha256(domain: str, value: object) -> str:
    """Hash a canonical payload in a type-specific domain."""

    if type(domain) is not str or not domain or "\x00" in domain:
        raise ValueError("hash domain must be nonempty text without NUL")
    return hashlib.sha256(
        domain.encode("ascii") + b"\x00" + canonical_json_bytes(value)
    ).hexdigest()


def _record_payload(value: object, *, omit: tuple[str, ...]) -> dict[str, object]:
    if not is_dataclass(value) or isinstance(value, type):
        raise ValueError("value must be a dataclass instance")
    return {
        item.name: _plain(getattr(value, item.name))
        for item in fields(value)
        if item.name not in omit
    }


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _require_text(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be nonempty text")
    return value


def _require_int(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be a plain int >= {minimum}")
    return value


def _require_float(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite plain float")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _require_text_tuple(value: object, name: str) -> tuple[str, ...]:
    if (
        type(value) is not tuple
        or any(type(item) is not str or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"{name} must be a unique tuple of nonempty strings")
    return value


@dataclass(frozen=True, slots=True)
class AdamWProfile:
    optimizer: Literal["AdamW"]
    betas: tuple[float, float]
    epsilon: float
    amsgrad: Literal[False]
    foreach: Literal[False]
    fused: Literal[False]
    gradient_clip: Literal["per_active_block_global_l2"]
    gradient_clip_max_norm: float
    proposal_acceptance: Literal["validity_only_no_monotonicity_claim"]
    reject_on: tuple[str, ...]

    def __post_init__(self) -> None:
        expected_rejections = (
            "nonfinite_objective",
            "nonfinite_gradient",
            "amp_overflow",
            "invalid_support",
            "non_spd",
            "scope_mismatch",
            "snapshot_alias",
            "optimizer_access_mismatch",
        )
        if (
            self.optimizer != "AdamW"
            or self.betas != (0.9, 0.999)
            or any(type(item) is not float for item in self.betas)
            or type(self.epsilon) is not float
            or self.epsilon != 1.0e-8
            or self.amsgrad is not False
            or self.foreach is not False
            or self.fused is not False
            or self.gradient_clip != "per_active_block_global_l2"
            or type(self.gradient_clip_max_norm) is not float
            or self.gradient_clip_max_norm != 1.0
            or self.proposal_acceptance != "validity_only_no_monotonicity_claim"
            or self.reject_on != expected_rejections
        ):
            raise ValueError("optimizer profile is frozen")


@dataclass(frozen=True, slots=True)
class SchedulerProfile:
    scheduler: Literal["linear_warmup_then_cosine"]
    warmup_optimizer_steps: Literal[100]
    minimum_lr_ratio: float
    restart_count: Literal[0]
    horizon: Literal["planned_active_optimizer_steps_for_attempt"]

    def __post_init__(self) -> None:
        if (
            self.scheduler != "linear_warmup_then_cosine"
            or type(self.warmup_optimizer_steps) is not int
            or self.warmup_optimizer_steps != 100
            or type(self.minimum_lr_ratio) is not float
            or self.minimum_lr_ratio != 0.1
            or type(self.restart_count) is not int
            or self.restart_count != 0
            or self.horizon != "planned_active_optimizer_steps_for_attempt"
        ):
            raise ValueError("scheduler profile is frozen")


@dataclass(frozen=True, slots=True)
class PrecisionProfile:
    real_training_device: Literal["cuda:0"]
    parameter_dtype: Literal["float32"]
    optimizer_state_dtype: Literal["float32"]
    autocast_enabled: Literal[True]
    autocast_dtype: Literal["bfloat16"]
    grad_scaler_enabled: Literal[False]
    grad_scaler_fixed_scale: float
    spd_factor_solve_logdet_dtype: Literal["float32"]
    smc_log_weight_dtype: Literal["float64"]
    metric_corpus_accumulator: Literal["python_math_fsum_float64"]
    torch_deterministic_algorithms: Literal[True]
    cudnn_deterministic: Literal[True]
    cudnn_benchmark: Literal[False]
    allow_tf32_matmul: Literal[False]
    allow_tf32_cudnn: Literal[False]
    allow_fp16_reduced_precision_reduce: Literal[False]
    cublas_workspace_config: Literal[":4096:8"]

    def __post_init__(self) -> None:
        expected = (
            "cuda:0",
            "float32",
            "float32",
            True,
            "bfloat16",
            False,
            1.0,
            "float32",
            "float64",
            "python_math_fsum_float64",
            True,
            True,
            False,
            False,
            False,
            False,
            ":4096:8",
        )
        if tuple(getattr(self, item.name) for item in fields(self)) != expected:
            raise ValueError("precision profile is frozen")
        if type(self.grad_scaler_fixed_scale) is not float:
            raise ValueError("grad_scaler_fixed_scale must be a plain float")


@dataclass(frozen=True, slots=True)
class CadenceProfile:
    validation_boundaries_per_pass: Literal[20]
    checkpoint_at_every_validation: Literal[True]
    confirmatory_passes: Literal[2]
    early_stopping: Literal[False]
    validation_boundary_rule: Literal["stable_unique_ceil_k_batches_per_pass_over_20"]

    def __post_init__(self) -> None:
        if (
            type(self.validation_boundaries_per_pass) is not int
            or self.validation_boundaries_per_pass != 20
            or self.checkpoint_at_every_validation is not True
            or type(self.confirmatory_passes) is not int
            or self.confirmatory_passes != 2
            or self.early_stopping is not False
            or self.validation_boundary_rule
            != "stable_unique_ceil_k_batches_per_pass_over_20"
        ):
            raise ValueError("cadence profile is frozen")


@dataclass(frozen=True, slots=True)
class CheckpointProfile:
    rolling_checkpoints_retained: Literal[2]
    rolling_role: Literal["resume_only"]
    terminal_checkpoint_retained: Literal[True]
    terminal_role: Literal["terminal_scoring"]
    best_checkpoint_selection: Literal[False]

    def __post_init__(self) -> None:
        if (
            type(self.rolling_checkpoints_retained) is not int
            or self.rolling_checkpoints_retained != 2
            or self.rolling_role != "resume_only"
            or self.terminal_checkpoint_retained is not True
            or self.terminal_role != "terminal_scoring"
            or self.best_checkpoint_selection is not False
        ):
            raise ValueError("checkpoint profile is frozen")


@dataclass(frozen=True, slots=True)
class StatisticalProfile:
    learning_rate_grid: tuple[float, ...]
    weight_decay_grid: tuple[float, ...]
    tuning_seed_ids: tuple[int, ...]
    confirmatory_seed_ids: tuple[int, ...]
    data_order_seed: Literal[2026072199]
    validation_stream_ids: tuple[int, ...]
    test_stream_ids: tuple[int, ...]
    validation_particle_count: Literal[256]
    particle_counts: tuple[int, ...]
    simultaneous_constant: float
    practical_threshold: float
    contraction_ratio: float
    one_opening_policy: Literal["durable_exclusive_single_test_transaction"]

    def __post_init__(self) -> None:
        expected = (
            (1.0e-4, 3.0e-4, 1.0e-3),
            (0.0, 1.0e-2),
            WT103_TUNING_SEED_IDS,
            WT103_CONFIRMATORY_SEED_IDS,
            2026072199,
            WT103_VALIDATION_STREAM_IDS,
            WT103_TEST_STREAM_IDS,
            256,
            WT103_PARTICLE_COUNTS,
            4.5144904535377144,
            0.01005033585350145,
            0.75,
            "durable_exclusive_single_test_transaction",
        )
        if tuple(getattr(self, item.name) for item in fields(self)) != expected:
            raise ValueError("statistical profile is frozen")
        for name in (
            "learning_rate_grid",
            "weight_decay_grid",
        ):
            if any(type(item) is not float for item in getattr(self, name)):
                raise ValueError(f"{name} must contain plain floats")
        for name in (
            "tuning_seed_ids",
            "confirmatory_seed_ids",
            "validation_stream_ids",
            "test_stream_ids",
            "particle_counts",
        ):
            if any(type(item) is not int for item in getattr(self, name)):
                raise ValueError(f"{name} must contain plain ints")


@dataclass(frozen=True, slots=True)
class ResourceProfile:
    maximum_gpu_hours: float
    maximum_wall_hours: float
    maximum_energy_kwh: float
    forecast_headroom_factor: float
    maximum_device_fraction: float
    power_sample_interval_ms: Literal[100]

    def __post_init__(self) -> None:
        expected = (720.0, 840.0, 500.0, 1.25, 0.85, 100)
        if tuple(getattr(self, item.name) for item in fields(self)) != expected:
            raise ValueError("resource profile is frozen")
        for item in fields(self)[:-1]:
            _require_float(getattr(self, item.name), item.name, minimum=0.0)
        _require_int(self.power_sample_interval_ms, "power_sample_interval_ms")


@dataclass(frozen=True, slots=True)
class SchemaProfile:
    h6_prediction_schema: Literal["h6-prediction-result-v3"]
    h8_schema: Literal["h8-sparse-scale-v5"]
    h8_config_schema: Literal["h8-validation-config-v3"]
    h8_parent_child_protocol: Literal["vfe4.h8.parent-child-protocol.v3"]
    training_sparsity_schema: Literal["wt103-training-sparsity-v1"]
    metric_schema: Literal["wt103-metric-record-v1"]
    figure_schema: Literal["wt103-figure-spec-v1"]
    checkpoint_schema: Literal["wt103-checkpoint-v1"]

    def __post_init__(self) -> None:
        expected = (
            "h6-prediction-result-v3",
            "h8-sparse-scale-v5",
            "h8-validation-config-v3",
            "vfe4.h8.parent-child-protocol.v3",
            "wt103-training-sparsity-v1",
            "wt103-metric-record-v1",
            "wt103-figure-spec-v1",
            "wt103-checkpoint-v1",
        )
        if tuple(getattr(self, item.name) for item in fields(self)) != expected:
            raise ValueError("schema profile is frozen")


@dataclass(frozen=True, slots=True)
class NonclaimProfile:
    backprop_free: Literal[False]
    h8_training_memory_transfer: Literal[False]
    h6_byte_vocabulary_transfer: Literal[False]
    v3_checkpoint_or_config_reuse: Literal[False]
    h8_asymptotic_scaling_law: Literal[False]
    monotone_elbo_or_coordinate_ascent: Literal[False]
    component_attribution_from_primary: Literal[False]

    def __post_init__(self) -> None:
        if any(getattr(self, item.name) is not False for item in fields(self)):
            raise ValueError("WT103 nonclaims must all remain false")


@dataclass(frozen=True, slots=True)
class ScientificPreconditionProfile:
    schema_version: Literal["wt103-scientific-preconditions-v1"]
    h6_prediction_authority: Literal["native_executable_v3"]
    h6_prediction_schema: Literal["h6-prediction-result-v3"]
    h8_schema: Literal["h8-sparse-scale-v5"]
    h8_config_schema: Literal["h8-validation-config-v3"]
    h8_parent_child_protocol: Literal["vfe4.h8.parent-child-protocol.v3"]
    training_sparsity_schema: Literal["wt103-training-sparsity-v1"]
    h8_reference_required: Literal[True]
    training_sparsity_reference_required: Literal[True]
    target_blind_predictor_safety_required: Literal[True]
    h8_can_satisfy_training_sparsity: Literal[False]
    capacity_can_satisfy_training_sparsity: Literal[False]
    preconditions_sha256: str

    def __post_init__(self) -> None:
        expected = (
            "wt103-scientific-preconditions-v1",
            "native_executable_v3",
            "h6-prediction-result-v3",
            "h8-sparse-scale-v5",
            "h8-validation-config-v3",
            "vfe4.h8.parent-child-protocol.v3",
            "wt103-training-sparsity-v1",
            True,
            True,
            True,
            False,
            False,
        )
        names = tuple(self.__dataclass_fields__)[:-1]
        if tuple(getattr(self, name) for name in names) != expected:
            raise ValueError("scientific precondition profile is frozen")
        digest = owned_sha256(
            "vfe4.wt103.scientific-preconditions.v1",
            _record_payload(self, omit=("preconditions_sha256",)),
        )
        _require_sha256(
            self.preconditions_sha256,
            "preconditions_sha256",
        )
        if self.preconditions_sha256 != digest:
            raise ValueError("preconditions_sha256 does not match scientific inputs")

    @classmethod
    def create(cls) -> "ScientificPreconditionProfile":
        payload = {
            "schema_version": "wt103-scientific-preconditions-v1",
            "h6_prediction_authority": "native_executable_v3",
            "h6_prediction_schema": "h6-prediction-result-v3",
            "h8_schema": "h8-sparse-scale-v5",
            "h8_config_schema": "h8-validation-config-v3",
            "h8_parent_child_protocol": ("vfe4.h8.parent-child-protocol.v3"),
            "training_sparsity_schema": "wt103-training-sparsity-v1",
            "h8_reference_required": True,
            "training_sparsity_reference_required": True,
            "target_blind_predictor_safety_required": True,
            "h8_can_satisfy_training_sparsity": False,
            "capacity_can_satisfy_training_sparsity": False,
        }
        return cls(
            **payload,
            preconditions_sha256=owned_sha256(
                "vfe4.wt103.scientific-preconditions.v1",
                payload,
            ),
        )  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class WT103ExperimentProfile:
    schema_version: Literal["wt103-experiment-profile-v1"]
    dataset_schema: Literal["wikitext-103-raw-v1"]
    tokenizer_schema: Literal["gpt2-tiktoken-v1"]
    vocabulary_size: Literal[50257]
    sequence_length: Literal[128]
    stride: Literal[128]
    batch_size: Literal[128]
    gradient_accumulation_steps: Literal[1]
    num_workers: Literal[0]
    pin_memory: Literal[True]
    drop_last: Literal[False]
    model_depth: Literal[1]
    d_z: Literal[20]
    d_m: Literal[20]
    K: Literal[20]
    combined_latent_block: Literal[40]
    source_lookback: Literal[20]
    state_parent_rule: Literal["range(max(0,t-20),t)"]
    model_parent_rule: Literal["range(max(0,t-20),t)"]
    population_frame_profile: Literal["h7-direct-glplus-v1"]
    decoder_profile: Literal["categorical_linear_chunked"]
    decoder_train_token_chunk: Literal[512]
    decoder_eval_token_chunk: Literal[256]
    smc_particle_chunk: Literal[32]
    dropout_probability: float
    input_output_embedding_tied: Literal[False]
    optimizer: AdamWProfile
    scheduler: SchedulerProfile
    precision: PrecisionProfile
    cadence: CadenceProfile
    checkpoints: CheckpointProfile
    statistics: StatisticalProfile
    resources: ResourceProfile
    schemas: SchemaProfile
    nonclaims: NonclaimProfile
    profile_sha256: str

    def __post_init__(self) -> None:
        scalar_expected = (
            "wt103-experiment-profile-v1",
            "wikitext-103-raw-v1",
            "gpt2-tiktoken-v1",
            50257,
            128,
            128,
            128,
            1,
            0,
            True,
            False,
            1,
            20,
            20,
            20,
            40,
            20,
            "range(max(0,t-20),t)",
            "range(max(0,t-20),t)",
            "h7-direct-glplus-v1",
            "categorical_linear_chunked",
            512,
            256,
            32,
            0.0,
            False,
        )
        scalar_names = tuple(self.__dataclass_fields__)[:26]
        if tuple(getattr(self, name) for name in scalar_names) != scalar_expected:
            raise ValueError("WT103 shared experiment profile is frozen")
        integer_names = (
            "vocabulary_size",
            "sequence_length",
            "stride",
            "batch_size",
            "gradient_accumulation_steps",
            "num_workers",
            "model_depth",
            "d_z",
            "d_m",
            "K",
            "combined_latent_block",
            "source_lookback",
            "decoder_train_token_chunk",
            "decoder_eval_token_chunk",
            "smc_particle_chunk",
        )
        if any(type(getattr(self, name)) is not int for name in integer_names):
            raise ValueError("WT103 profile integers must have exact int type")
        if type(self.dropout_probability) is not float:
            raise ValueError("dropout_probability must be a plain float")
        exact_nested = (
            (self.optimizer, AdamWProfile),
            (self.scheduler, SchedulerProfile),
            (self.precision, PrecisionProfile),
            (self.cadence, CadenceProfile),
            (self.checkpoints, CheckpointProfile),
            (self.statistics, StatisticalProfile),
            (self.resources, ResourceProfile),
            (self.schemas, SchemaProfile),
            (self.nonclaims, NonclaimProfile),
        )
        if any(type(value) is not expected for value, expected in exact_nested):
            raise ValueError("WT103 profile requires exact nested record types")
        expected_hash = owned_sha256(
            "vfe4.wt103.experiment-profile.v1",
            _record_payload(self, omit=("profile_sha256",)),
        )
        _require_sha256(self.profile_sha256, "profile_sha256")
        if self.profile_sha256 != expected_hash:
            raise ValueError("profile_sha256 does not match the profile")

    @classmethod
    def create(cls, **values: object) -> "WT103ExperimentProfile":
        payload = {
            "schema_version": "wt103-experiment-profile-v1",
            **values,
        }
        profile_hash = owned_sha256(
            "vfe4.wt103.experiment-profile.v1",
            payload,
        )
        return cls(**payload, profile_sha256=profile_hash)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class A0ArchitectureProfile:
    schema_version: Literal["wt103-a0-architecture-v1"]
    block_count: Literal[1]
    hidden_width: int
    attention_heads: Literal[2]
    head_width: int
    attention_context: Literal["full_causal_inclusive_self"]
    attention_allowed_keys: Literal["range(0,q+1)"]
    attention_semantic_pair_count: Literal["L*(L+1)//2"]
    attention_implementation: Literal[
        "torch.nn.functional.scaled_dot_product_attention"
    ]
    attention_backend_policy: Literal["flash_attention_only_no_fallback"]
    pytorch_sdpa_api_binding: Literal[
        "torch.nn.attention.sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION])"
    ]
    enabled_backends: tuple[Literal["FLASH_ATTENTION"], ...]
    alternative_backends_disabled: Literal[True]
    source_lock_scope: Literal[
        "candidate_unverified", "production_source_lock_verified"
    ]
    pytorch_version: str
    sdpa_api_sha256: str
    flash_backend_sha256: str
    attention_is_causal: Literal[True]
    attention_mask_argument: None
    attention_scale: Literal["1/sqrt(head_width)"]
    attention_dropout_probability: float
    attention_returns_weights: Literal[False]
    grouped_query_attention: Literal[False]
    backend_fallback_allowed: Literal[False]
    fused_full_attention_allowed: Literal[True]
    fused_attention_materialization: Literal["forbidden"]
    token_embedding: Literal["learned[V,h]_no_bias"]
    positional_encoding: Literal["learned_absolute"]
    positional_capacity: Literal[128]
    position_interpolation: Literal[False]
    input_composition: Literal["token_embedding_plus_position_embedding"]
    normalization: Literal["LayerNorm(eps=1e-5,elementwise_affine=true,bias=true)"]
    normalization_placement: Literal["pre_norm_with_final_norm"]
    residual_topology: Literal["x=x+attn(ln1(x));x=x+mlp(ln2(x));y=ln_f(x)"]
    qkv_projection: Literal["Linear(in=h,out=3h,weight[3h,h],bias[3h])"]
    attention_output_projection: Literal["Linear(in=h,out=h,weight[h,h],bias[h])"]
    mlp_input_projection: Literal["Linear(in=h,out=4h,weight[4h,h],bias[4h])"]
    activation: Literal["gelu_tanh_approximation"]
    mlp_output_projection: Literal["Linear(in=4h,out=h,weight[h,4h],bias[h])"]
    decoder_projection: Literal["untied_Linear(in=h,out=V,weight[V,h],bias[V])"]
    all_dropout_probabilities: float
    candidate_hidden_widths: tuple[int, ...]
    parameter_relative_tolerance: float
    flop_relative_tolerance: float
    candidate_selection_key: Literal[
        "abs_log_parameter_ratio_abs_log_flop_ratio_hidden_width"
    ]
    parameter_formula_schema: Literal["wt103-a0-parameter-formula-v1"]
    flop_formula_schema: Literal["wt103-a0-semantic-train-flops-v1"]
    formula_sha256: str
    architecture_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != "wt103-a0-architecture-v1":
            raise ValueError("unsupported A0 architecture schema")
        if type(self.block_count) is not int or self.block_count != 1:
            raise ValueError("A0 block_count must be exactly one")
        _require_int(self.hidden_width, "hidden_width", minimum=1)
        if (
            self.hidden_width not in WT103_A0_HIDDEN_WIDTH_CANDIDATES
            or self.hidden_width % 2
            or type(self.attention_heads) is not int
            or self.attention_heads != 2
            or type(self.head_width) is not int
            or self.head_width != self.hidden_width // 2
        ):
            raise ValueError("A0 hidden/head widths violate the frozen candidates")
        fixed = (
            (self.attention_context, "full_causal_inclusive_self"),
            (self.attention_allowed_keys, "range(0,q+1)"),
            (self.attention_semantic_pair_count, "L*(L+1)//2"),
            (
                self.attention_implementation,
                "torch.nn.functional.scaled_dot_product_attention",
            ),
            (
                self.attention_backend_policy,
                "flash_attention_only_no_fallback",
            ),
            (
                self.pytorch_sdpa_api_binding,
                "torch.nn.attention.sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION])",
            ),
        )
        if any(value != expected for value, expected in fixed):
            raise ValueError("A0 attention contract is frozen")
        if (
            type(self.enabled_backends) is not tuple
            or self.enabled_backends != ("FLASH_ATTENTION",)
            or self.alternative_backends_disabled is not True
            or self.source_lock_scope
            not in (
                "candidate_unverified",
                "production_source_lock_verified",
            )
        ):
            raise ValueError("A0 must use the single Flash backend context")
        _require_text(self.pytorch_version, "pytorch_version")
        if self.source_lock_scope == "production_source_lock_verified":
            _require_sha256(self.sdpa_api_sha256, "sdpa_api_sha256")
            _require_sha256(
                self.flash_backend_sha256,
                "flash_backend_sha256",
            )
        elif (
            self.sdpa_api_sha256 != "unresolved_until_task13_source_lock"
            or self.flash_backend_sha256 != "unresolved_until_task13_source_lock"
            or self.pytorch_version != "unresolved_until_task13_source_lock"
        ):
            raise ValueError(
                "candidate API identities must remain explicitly unresolved"
            )
        if (
            self.attention_is_causal is not True
            or self.attention_mask_argument is not None
            or self.attention_scale != "1/sqrt(head_width)"
            or type(self.attention_dropout_probability) is not float
            or self.attention_dropout_probability != 0.0
            or self.attention_returns_weights is not False
            or self.grouped_query_attention is not False
            or self.backend_fallback_allowed is not False
            or self.fused_full_attention_allowed is not True
            or self.fused_attention_materialization != "forbidden"
        ):
            raise ValueError("A0 attention fallback/materialization is forbidden")
        architecture_literals = (
            (self.token_embedding, "learned[V,h]_no_bias"),
            (self.positional_encoding, "learned_absolute"),
            (self.position_interpolation, False),
            (
                self.input_composition,
                "token_embedding_plus_position_embedding",
            ),
            (
                self.normalization,
                "LayerNorm(eps=1e-5,elementwise_affine=true,bias=true)",
            ),
            (self.normalization_placement, "pre_norm_with_final_norm"),
            (
                self.residual_topology,
                "x=x+attn(ln1(x));x=x+mlp(ln2(x));y=ln_f(x)",
            ),
            (
                self.qkv_projection,
                "Linear(in=h,out=3h,weight[3h,h],bias[3h])",
            ),
            (
                self.attention_output_projection,
                "Linear(in=h,out=h,weight[h,h],bias[h])",
            ),
            (
                self.mlp_input_projection,
                "Linear(in=h,out=4h,weight[4h,h],bias[4h])",
            ),
            (self.activation, "gelu_tanh_approximation"),
            (
                self.mlp_output_projection,
                "Linear(in=4h,out=h,weight[h,4h],bias[h])",
            ),
            (
                self.decoder_projection,
                "untied_Linear(in=h,out=V,weight[V,h],bias[V])",
            ),
        )
        if any(value != expected for value, expected in architecture_literals):
            raise ValueError("A0 architectural literals are frozen")
        if (
            type(self.positional_capacity) is not int
            or self.positional_capacity != 128
            or type(self.all_dropout_probabilities) is not float
            or self.all_dropout_probabilities != 0.0
            or self.candidate_hidden_widths != WT103_A0_HIDDEN_WIDTH_CANDIDATES
            or any(type(item) is not int for item in self.candidate_hidden_widths)
            or type(self.parameter_relative_tolerance) is not float
            or self.parameter_relative_tolerance != 0.01
            or type(self.flop_relative_tolerance) is not float
            or self.flop_relative_tolerance != 0.05
            or self.candidate_selection_key
            != "abs_log_parameter_ratio_abs_log_flop_ratio_hidden_width"
            or self.parameter_formula_schema != "wt103-a0-parameter-formula-v1"
            or self.flop_formula_schema != "wt103-a0-semantic-train-flops-v1"
        ):
            raise ValueError("A0 capacity matching/formula policy is frozen")
        _require_sha256(self.formula_sha256, "formula_sha256")
        expected_hash = owned_sha256(
            "vfe4.wt103.a0-architecture.v1",
            _record_payload(self, omit=("architecture_sha256",)),
        )
        _require_sha256(self.architecture_sha256, "architecture_sha256")
        if self.architecture_sha256 != expected_hash:
            raise ValueError("architecture_sha256 does not match A0 architecture")

    @classmethod
    def create(cls, **values: object) -> "A0ArchitectureProfile":
        payload = {
            "schema_version": "wt103-a0-architecture-v1",
            **values,
        }
        digest = owned_sha256("vfe4.wt103.a0-architecture.v1", payload)
        return cls(**payload, architecture_sha256=digest)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class A0FormulaRecord:
    schema_version: Literal["wt103-a0-formula-record-v1"]
    vocabulary_size: Literal[50257]
    positional_capacity: Literal[128]
    hidden_width: int
    parameter_count: int
    semantic_train_flops: int
    multiply_flops: Literal[1]
    add_subtract_divide_exp_log_tanh_rsqrt_flops: Literal[1]
    comparison_and_indexing_flops: Literal[0]
    formula_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != "wt103-a0-formula-record-v1"
            or type(self.vocabulary_size) is not int
            or self.vocabulary_size != 50257
            or type(self.positional_capacity) is not int
            or self.positional_capacity != 128
            or type(self.hidden_width) is not int
            or self.hidden_width not in WT103_A0_HIDDEN_WIDTH_CANDIDATES
            or type(self.parameter_count) is not int
            or self.parameter_count
            != (
                2 * self.vocabulary_size * self.hidden_width
                + 128 * self.hidden_width
                + 12 * self.hidden_width**2
                + 15 * self.hidden_width
                + self.vocabulary_size
            )
            or type(self.semantic_train_flops) is not int
            or self.semantic_train_flops <= 0
            or type(self.multiply_flops) is not int
            or self.multiply_flops != 1
            or type(self.add_subtract_divide_exp_log_tanh_rsqrt_flops) is not int
            or self.add_subtract_divide_exp_log_tanh_rsqrt_flops != 1
            or type(self.comparison_and_indexing_flops) is not int
            or self.comparison_and_indexing_flops != 0
        ):
            raise ValueError("A0 formula record is invalid")
        expected = owned_sha256(
            "vfe4.wt103.a0-formula-record.v1",
            _record_payload(self, omit=("formula_sha256",)),
        )
        _require_sha256(self.formula_sha256, "formula_sha256")
        if self.formula_sha256 != expected:
            raise ValueError("formula_sha256 does not match formula record")


@dataclass(frozen=True, slots=True)
class WT103ArmSpec:
    schema_version: Literal["wt103-arm-spec-v1"]
    arm_id: str
    factory_id: str
    training_objective: str
    prior_variant: str
    source_mixture: str
    latent_enabled: bool
    recognition_enabled: bool
    recognition_family: str
    recognition_iterations_per_batch: int
    update_phases: tuple[str, ...]
    scorer_kind: str
    tuning_grid_id: Literal["wt103-six-cell-v1"]
    confirmatory_seed_ids: tuple[int, ...]
    terminal_checkpoint_role: Literal["terminal_scoring"]
    result_role: str
    nonclaims: tuple[str, ...]
    arm_spec_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != "wt103-arm-spec-v1":
            raise ValueError("unsupported WT103 arm schema")
        if self.arm_id not in WT103_ARM_IDS:
            raise ValueError("unknown WT103 arm_id")
        _require_text(self.factory_id, "factory_id")
        if self.training_objective not in (
            "cross_entropy",
            "complete_elbo",
            "emission_only_ablation_non_elbo",
        ):
            raise ValueError("unknown training objective")
        if self.prior_variant not in (
            "absent",
            "fixed",
            "parent_specific_pooled_prefix",
        ):
            raise ValueError("unknown prior variant")
        if self.source_mixture not in ("absent", "exact"):
            raise ValueError("unknown source mixture")
        if type(self.latent_enabled) is not bool:
            raise ValueError("latent_enabled must be a plain bool")
        if type(self.recognition_enabled) is not bool:
            raise ValueError("recognition_enabled must be a plain bool")
        if self.recognition_family not in (
            "absent",
            "structured_block_tridiagonal_smoothing",
        ):
            raise ValueError("unknown recognition family")
        if type(
            self.recognition_iterations_per_batch
        ) is not int or self.recognition_iterations_per_batch not in (0, 1):
            raise ValueError("recognition iterations must be zero or one")
        _require_text_tuple(self.update_phases, "update_phases")
        if self.scorer_kind not in ("exact_autoregressive", "weighted_smc"):
            raise ValueError("unknown scorer kind")
        if self.tuning_grid_id != "wt103-six-cell-v1":
            raise ValueError("every arm must use the six-cell grid")
        if (
            type(self.confirmatory_seed_ids) is not tuple
            or self.confirmatory_seed_ids != WT103_CONFIRMATORY_SEED_IDS
            or any(type(item) is not int for item in self.confirmatory_seed_ids)
        ):
            raise ValueError("confirmatory seed inventory is frozen")
        if self.terminal_checkpoint_role != "terminal_scoring":
            raise ValueError("arm terminal checkpoint role is frozen")
        if self.result_role not in (
            "PRIMARY_REFERENCE",
            "PRIMARY_ENDPOINT",
            "PRIOR_CONTROL",
            "OBJECTIVE_GATE",
            "LATENT_PATH_CONTROL",
        ):
            raise ValueError("unknown arm result role")
        _require_text_tuple(self.nonclaims, "nonclaims")
        latent = self.latent_enabled
        if latent and (
            self.recognition_enabled is not True
            or self.source_mixture != "exact"
            or self.recognition_family != "structured_block_tridiagonal_smoothing"
            or self.recognition_iterations_per_batch != 1
            or self.update_phases
            != (
                "recognition_adam_proposal",
                "immutable_detached_snapshot",
                "model_adam_proposal",
            )
            or self.scorer_kind != "weighted_smc"
        ):
            raise ValueError("latent arm semantics must be explicit and exact")
        if not latent and (
            self.recognition_enabled is not False
            or self.source_mixture != "absent"
            or self.recognition_family != "absent"
            or self.recognition_iterations_per_batch != 0
            or self.update_phases != ("model_ce_adam_proposal",)
            or self.scorer_kind != "exact_autoregressive"
        ):
            raise ValueError("nonlatent arm semantics must be explicit and exact")
        expected = owned_sha256(
            "vfe4.wt103.arm-spec.v1",
            _record_payload(self, omit=("arm_spec_sha256",)),
        )
        _require_sha256(self.arm_spec_sha256, "arm_spec_sha256")
        if self.arm_spec_sha256 != expected:
            raise ValueError("arm_spec_sha256 does not match the arm")

    @classmethod
    def create(cls, **values: object) -> "WT103ArmSpec":
        payload = {"schema_version": "wt103-arm-spec-v1", **values}
        digest = owned_sha256("vfe4.wt103.arm-spec.v1", payload)
        return cls(**payload, arm_spec_sha256=digest)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class WT103GateSpec:
    schema_version: Literal["wt103-gate-spec-v1"]
    gate_id: str
    ordinal: int
    prerequisite_gate_ids: tuple[str, ...]
    result_arm_ids: tuple[str, ...]
    disposition_rule_id: str
    gate_spec_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != "wt103-gate-spec-v1":
            raise ValueError("unsupported WT103 gate schema")
        if self.gate_id not in WT103_GATE_IDS:
            raise ValueError("unknown WT103 gate_id")
        if type(self.ordinal) is not int or self.ordinal != WT103_GATE_IDS.index(
            self.gate_id
        ):
            raise ValueError("gate ordinal must follow exact gate order")
        _require_text_tuple(
            self.prerequisite_gate_ids,
            "prerequisite_gate_ids",
        )
        if any(
            item not in WT103_GATE_IDS[: self.ordinal]
            for item in self.prerequisite_gate_ids
        ):
            raise ValueError("gate prerequisites must name earlier gates")
        _require_text_tuple(self.result_arm_ids, "result_arm_ids")
        if any(item not in WT103_ARM_IDS for item in self.result_arm_ids):
            raise ValueError("gate result_arm_ids contain an unknown arm")
        _require_text(self.disposition_rule_id, "disposition_rule_id")
        expected = owned_sha256(
            "vfe4.wt103.gate-spec.v1",
            _record_payload(self, omit=("gate_spec_sha256",)),
        )
        _require_sha256(self.gate_spec_sha256, "gate_spec_sha256")
        if self.gate_spec_sha256 != expected:
            raise ValueError("gate_spec_sha256 does not match the gate")

    @classmethod
    def create(cls, **values: object) -> "WT103GateSpec":
        payload = {"schema_version": "wt103-gate-spec-v1", **values}
        digest = owned_sha256("vfe4.wt103.gate-spec.v1", payload)
        return cls(**payload, gate_spec_sha256=digest)  # type: ignore[arg-type]


def default_wt103_arm_specs() -> tuple[WT103ArmSpec, ...]:
    """Return the exact ordered five-row arm inventory."""

    common = {
        "tuning_grid_id": "wt103-six-cell-v1",
        "confirmatory_seed_ids": WT103_CONFIRMATORY_SEED_IDS,
        "terminal_checkpoint_role": "terminal_scoring",
    }
    return (
        WT103ArmSpec.create(
            arm_id="WT103-A0-AR-v1",
            factory_id="build_wt103_a0@wt103-arm-v1",
            training_objective="cross_entropy",
            prior_variant="absent",
            source_mixture="absent",
            latent_enabled=False,
            recognition_enabled=False,
            recognition_family="absent",
            recognition_iterations_per_batch=0,
            update_phases=("model_ce_adam_proposal",),
            scorer_kind="exact_autoregressive",
            result_role="PRIMARY_REFERENCE",
            nonclaims=(
                "reference_is_not_vfe4_complete_elbo",
                "no_component_attribution",
            ),
            **common,
        ),
        WT103ArmSpec.create(
            arm_id="WT103-A5-PARENT-SPECIFIC-PREFIX-COMPLETE-v1",
            factory_id="build_wt103_a5_parent_specific@wt103-arm-v1",
            training_objective="complete_elbo",
            prior_variant="parent_specific_pooled_prefix",
            source_mixture="exact",
            latent_enabled=True,
            recognition_enabled=True,
            recognition_family="structured_block_tridiagonal_smoothing",
            recognition_iterations_per_batch=1,
            update_phases=(
                "recognition_adam_proposal",
                "immutable_detached_snapshot",
                "model_adam_proposal",
            ),
            scorer_kind="weighted_smc",
            result_role="PRIMARY_ENDPOINT",
            nonclaims=(
                "whole_architecture_not_component_attribution",
                "adam_proposal_not_coordinate_ascent",
            ),
            **common,
        ),
        WT103ArmSpec.create(
            arm_id="WT103-A5-FIXED-COMPLETE-v1",
            factory_id="build_wt103_a5_fixed@wt103-arm-v1",
            training_objective="complete_elbo",
            prior_variant="fixed",
            source_mixture="exact",
            latent_enabled=True,
            recognition_enabled=True,
            recognition_family="structured_block_tridiagonal_smoothing",
            recognition_iterations_per_batch=1,
            update_phases=(
                "recognition_adam_proposal",
                "immutable_detached_snapshot",
                "model_adam_proposal",
            ),
            scorer_kind="weighted_smc",
            result_role="PRIOR_CONTROL",
            nonclaims=(
                "changed_joint_not_prior_only_causal_estimate",
                "control_cannot_rescue_primary",
            ),
            **common,
        ),
        WT103ArmSpec.create(
            arm_id="WT103-A5-PARENT-SPECIFIC-PREFIX-EMISSION-v1",
            factory_id="build_wt103_a5_parent_specific@wt103-arm-v1",
            training_objective="emission_only_ablation_non_elbo",
            prior_variant="parent_specific_pooled_prefix",
            source_mixture="exact",
            latent_enabled=True,
            recognition_enabled=True,
            recognition_family="structured_block_tridiagonal_smoothing",
            recognition_iterations_per_batch=1,
            update_phases=(
                "recognition_adam_proposal",
                "immutable_detached_snapshot",
                "model_adam_proposal",
            ),
            scorer_kind="weighted_smc",
            result_role="OBJECTIVE_GATE",
            nonclaims=(
                "emission_only_is_not_an_elbo",
                "objective_gate_cannot_rescue_primary",
            ),
            **common,
        ),
        WT103ArmSpec.create(
            arm_id="WT103-A5-NOLATENT-v1",
            factory_id="build_wt103_a5_nolatent@wt103-arm-v1",
            training_objective="cross_entropy",
            prior_variant="absent",
            source_mixture="absent",
            latent_enabled=False,
            recognition_enabled=False,
            recognition_family="absent",
            recognition_iterations_per_batch=0,
            update_phases=("model_ce_adam_proposal",),
            scorer_kind="exact_autoregressive",
            result_role="LATENT_PATH_CONTROL",
            nonclaims=(
                "bundled_control_not_held_fixed_attribution",
                "no_dormant_latent_or_recognition_state",
            ),
            **common,
        ),
    )


def default_wt103_gate_specs() -> tuple[WT103GateSpec, ...]:
    """Return the exact ordered seven-gate logical inventory."""

    return (
        WT103GateSpec.create(
            gate_id="SOURCE_LOCK",
            ordinal=0,
            prerequisite_gate_ids=(),
            result_arm_ids=(),
            disposition_rule_id="finalized_source_or_stop",
        ),
        WT103GateSpec.create(
            gate_id="H8_EXACT_REVISION",
            ordinal=1,
            prerequisite_gate_ids=("SOURCE_LOCK",),
            result_arm_ids=(),
            disposition_rule_id="same_revision_h8_v5_pass_or_stop",
        ),
        WT103GateSpec.create(
            gate_id="POST_H8_READINESS",
            ordinal=2,
            prerequisite_gate_ids=("H8_EXACT_REVISION",),
            result_arm_ids=(),
            disposition_rule_id="all_static_and_resource_inputs_pass_or_stop",
        ),
        WT103GateSpec.create(
            gate_id="OBJECTIVE",
            ordinal=3,
            prerequisite_gate_ids=("POST_H8_READINESS",),
            result_arm_ids=("WT103-A5-PARENT-SPECIFIC-PREFIX-EMISSION-v1",),
            disposition_rule_id="objective_gate_before_primary",
        ),
        WT103GateSpec.create(
            gate_id="PRIMARY",
            ordinal=4,
            prerequisite_gate_ids=("OBJECTIVE",),
            result_arm_ids=(
                "WT103-A0-AR-v1",
                "WT103-A5-PARENT-SPECIFIC-PREFIX-COMPLETE-v1",
            ),
            disposition_rule_id="paired_estimator_inflated_primary_decision",
        ),
        WT103GateSpec.create(
            gate_id="PRIOR_CONTROL",
            ordinal=5,
            prerequisite_gate_ids=("POST_H8_READINESS",),
            result_arm_ids=("WT103-A5-FIXED-COMPLETE-v1",),
            disposition_rule_id="retain_control_without_primary_promotion",
        ),
        WT103GateSpec.create(
            gate_id="LATENT_PATH_CONTROL",
            ordinal=6,
            prerequisite_gate_ids=("POST_H8_READINESS",),
            result_arm_ids=("WT103-A5-NOLATENT-v1",),
            disposition_rule_id="retain_control_without_primary_promotion",
        ),
    )


@dataclass(frozen=True, slots=True)
class EstimatorProtocol:
    schema_version: Literal["wt103-estimator-protocol-v1"]
    validation_particle_count: Literal[256]
    validation_stream_ids: tuple[int, ...]
    test_stream_ids: tuple[int, ...]
    particle_counts: tuple[int, ...]
    validation_stream_domain: Literal[
        "post-h8-wt103-validation-v1|2026072198|stream_id|purpose"
    ]
    test_stream_domain: Literal["post-h8-wt103-test-v1|2026072198|stream_id|purpose"]
    protocol_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != "wt103-estimator-protocol-v1"
            or type(self.validation_particle_count) is not int
            or self.validation_particle_count != 256
            or self.validation_stream_ids != WT103_VALIDATION_STREAM_IDS
            or self.test_stream_ids != WT103_TEST_STREAM_IDS
            or self.particle_counts != WT103_PARTICLE_COUNTS
            or self.validation_stream_domain
            != "post-h8-wt103-validation-v1|2026072198|stream_id|purpose"
            or self.test_stream_domain
            != "post-h8-wt103-test-v1|2026072198|stream_id|purpose"
            or any(
                type(item) is not int
                for inventory in (
                    self.validation_stream_ids,
                    self.test_stream_ids,
                    self.particle_counts,
                )
                for item in inventory
            )
        ):
            raise ValueError("estimator protocol is frozen")
        expected = owned_sha256(
            "vfe4.wt103.estimator-protocol.v1",
            _record_payload(self, omit=("protocol_sha256",)),
        )
        _require_sha256(self.protocol_sha256, "protocol_sha256")
        if self.protocol_sha256 != expected:
            raise ValueError("protocol_sha256 does not match estimator protocol")

    @classmethod
    def create(cls) -> "EstimatorProtocol":
        payload = {
            "schema_version": "wt103-estimator-protocol-v1",
            "validation_particle_count": 256,
            "validation_stream_ids": WT103_VALIDATION_STREAM_IDS,
            "test_stream_ids": WT103_TEST_STREAM_IDS,
            "particle_counts": WT103_PARTICLE_COUNTS,
            "validation_stream_domain": (
                "post-h8-wt103-validation-v1|2026072198|stream_id|purpose"
            ),
            "test_stream_domain": (
                "post-h8-wt103-test-v1|2026072198|stream_id|purpose"
            ),
        }
        return cls(
            **payload,
            protocol_sha256=owned_sha256(
                "vfe4.wt103.estimator-protocol.v1",
                payload,
            ),
        )  # type: ignore[arg-type]


def _score_record_keys(
    *,
    endpoint_key: str,
    scorer_kind: str,
    split: Literal["validation", "test"],
    protocol: EstimatorProtocol,
) -> tuple[str, ...]:
    if scorer_kind == "exact_autoregressive":
        return (f"raw-score/{split}/{endpoint_key}/exact",)
    if scorer_kind != "weighted_smc":
        raise ValueError("unknown scorer kind")
    if split == "validation":
        return tuple(
            f"raw-score/validation/{endpoint_key}/particles="
            f"{protocol.validation_particle_count}/stream={stream}"
            for stream in protocol.validation_stream_ids
        )
    return tuple(
        f"raw-score/test/{endpoint_key}/particles={particles}/stream={stream}"
        for particles in protocol.particle_counts
        for stream in protocol.test_stream_ids
    )


def _figure_series_keys(arms: tuple[WT103ArmSpec, ...]) -> tuple[str, ...]:
    series: list[str] = []
    for arm in arms:
        series.extend(
            (
                f"training-objective-and-validation/{arm.arm_id}/train-objective",
                f"training-objective-and-validation/{arm.arm_id}/validation-nll",
            )
        )
    for arm in arms:
        series.extend(
            (
                f"terminal-prior-nll-ppl/{arm.arm_id}/nll",
                f"terminal-prior-nll-ppl/{arm.arm_id}/ppl",
            )
        )
    for arm in arms:
        if arm.training_objective == "complete_elbo":
            series.append(f"complete-elbo-decomposition/{arm.arm_id}/complete")
        elif arm.training_objective == "emission_only_ablation_non_elbo":
            series.append(f"complete-elbo-decomposition/{arm.arm_id}/emission-non-elbo")
    for arm in arms:
        if arm.latent_enabled:
            series.extend(
                (
                    f"source-entropy-effective-count/{arm.arm_id}/entropy",
                    f"source-entropy-effective-count/{arm.arm_id}/effective-count",
                )
            )
    for arm in arms:
        series.append(f"update-acceptance/{arm.arm_id}/updates")
    for arm in arms:
        if arm.latent_enabled:
            series.append(f"spd-health/{arm.arm_id}/health")
    for arm in arms:
        series.append(f"throughput-memory/{arm.arm_id}/resources")
    for arm in arms:
        series.append(f"seed-variability/{arm.arm_id}/variability")
    return tuple(series)


@dataclass(frozen=True, slots=True)
class EndpointInventory:
    schema_version: Literal["wt103-endpoint-inventory-v1"]
    arms: tuple[WT103ArmSpec, ...]
    gates: tuple[WT103GateSpec, ...]
    tuning_cells: tuple[tuple[float, float], ...]
    tuning_seed_ids: tuple[int, ...]
    confirmatory_seed_ids: tuple[int, ...]
    validation_stream_ids: tuple[int, ...]
    test_stream_ids: tuple[int, ...]
    particle_counts: tuple[int, ...]
    estimator_protocol_sha256: str
    tuning_attempt_keys: tuple[str, ...]
    terminal_checkpoint_keys: tuple[str, ...]
    validation_endpoint_keys: tuple[str, ...]
    test_endpoint_keys: tuple[str, ...]
    raw_score_record_keys: tuple[str, ...]
    result_row_keys: tuple[str, ...]
    figure_panel_keys: tuple[str, ...]
    figure_series_keys: tuple[str, ...]
    endpoint_inventory_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != "wt103-endpoint-inventory-v1":
            raise ValueError("unsupported endpoint inventory schema")
        if (
            type(self.arms) is not tuple
            or any(type(item) is not WT103ArmSpec for item in self.arms)
            or tuple(item.arm_id for item in self.arms) != WT103_ARM_IDS
            or tuple(item.arm_spec_sha256 for item in self.arms)
            != tuple(item.arm_spec_sha256 for item in default_wt103_arm_specs())
        ):
            raise ValueError("endpoint inventory arms are not exact")
        if (
            type(self.gates) is not tuple
            or any(type(item) is not WT103GateSpec for item in self.gates)
            or tuple(item.gate_id for item in self.gates) != WT103_GATE_IDS
            or tuple(item.gate_spec_sha256 for item in self.gates)
            != tuple(item.gate_spec_sha256 for item in default_wt103_gate_specs())
        ):
            raise ValueError("endpoint inventory gates are not exact")
        if "OBJECTIVE" not in self.gates[4].prerequisite_gate_ids:
            raise ValueError("OBJECTIVE must be a prerequisite of PRIMARY")
        if (
            self.tuning_cells != WT103_TUNING_CELLS
            or self.tuning_seed_ids != WT103_TUNING_SEED_IDS
            or self.confirmatory_seed_ids != WT103_CONFIRMATORY_SEED_IDS
            or self.validation_stream_ids != WT103_VALIDATION_STREAM_IDS
            or self.test_stream_ids != WT103_TEST_STREAM_IDS
            or self.particle_counts != WT103_PARTICLE_COUNTS
        ):
            raise ValueError("endpoint inventory scalar inventories are frozen")
        for name in (
            "tuning_attempt_keys",
            "terminal_checkpoint_keys",
            "validation_endpoint_keys",
            "test_endpoint_keys",
            "raw_score_record_keys",
            "result_row_keys",
            "figure_panel_keys",
            "figure_series_keys",
        ):
            value = getattr(self, name)
            _require_text_tuple(value, name)
        if self.figure_panel_keys != WT103_FIGURE_PANEL_KEYS:
            raise ValueError("figure panel inventory is frozen")
        _require_sha256(
            self.estimator_protocol_sha256,
            "estimator_protocol_sha256",
        )
        expected = owned_sha256(
            "vfe4.wt103.endpoint-inventory.v1",
            _record_payload(self, omit=("endpoint_inventory_sha256",)),
        )
        _require_sha256(
            self.endpoint_inventory_sha256,
            "endpoint_inventory_sha256",
        )
        if self.endpoint_inventory_sha256 != expected:
            raise ValueError(
                "endpoint_inventory_sha256 does not match derived inventory"
            )

    @classmethod
    def create(
        cls,
        arms: tuple[WT103ArmSpec, ...],
        gates: tuple[WT103GateSpec, ...],
        tuning_cells: tuple[tuple[float, float], ...],
        tuning_seeds: tuple[int, ...],
        confirmatory_seeds: tuple[int, ...],
        estimator_protocol: EstimatorProtocol,
    ) -> "EndpointInventory":
        if type(estimator_protocol) is not EstimatorProtocol:
            raise ValueError("estimator_protocol must have exact type")
        estimator_protocol.__post_init__()
        if type(arms) is not tuple or type(gates) is not tuple:
            raise ValueError("arms and gates must be immutable tuples")
        if type(tuning_cells) is not tuple:
            raise ValueError("tuning_cells must be an immutable tuple")
        if type(tuning_seeds) is not tuple:
            raise ValueError("tuning_seeds must be an immutable tuple")
        if type(confirmatory_seeds) is not tuple:
            raise ValueError("confirmatory_seeds must be an immutable tuple")

        tuning_attempts: list[str] = []
        terminal_checkpoints: list[str] = []
        validation_endpoints: list[str] = []
        test_endpoints: list[str] = []
        raw_scores: list[str] = []
        for arm in arms:
            for cell_index, _cell in enumerate(tuning_cells):
                for seed in tuning_seeds:
                    attempt = f"tuning/{arm.arm_id}/cell={cell_index}/seed={seed}"
                    tuning_attempts.append(attempt)
                    endpoint = f"validation/{attempt}"
                    validation_endpoints.append(endpoint)
                    raw_scores.extend(
                        _score_record_keys(
                            endpoint_key=endpoint,
                            scorer_kind=arm.scorer_kind,
                            split="validation",
                            protocol=estimator_protocol,
                        )
                    )
            for seed in confirmatory_seeds:
                terminal = f"terminal/{arm.arm_id}/seed={seed}"
                terminal_checkpoints.append(terminal)
                for pass_index in range(2):
                    for boundary in range(1, 21):
                        endpoint = (
                            f"validation/confirm/{arm.arm_id}/seed={seed}/"
                            f"pass={pass_index}/boundary={boundary}"
                        )
                        validation_endpoints.append(endpoint)
                        raw_scores.extend(
                            _score_record_keys(
                                endpoint_key=endpoint,
                                scorer_kind=arm.scorer_kind,
                                split="validation",
                                protocol=estimator_protocol,
                            )
                        )
                test_endpoint = f"test/{terminal}"
                test_endpoints.append(test_endpoint)
                raw_scores.extend(
                    _score_record_keys(
                        endpoint_key=test_endpoint,
                        scorer_kind=arm.scorer_kind,
                        split="test",
                        protocol=estimator_protocol,
                    )
                )
        payload = {
            "schema_version": "wt103-endpoint-inventory-v1",
            "arms": arms,
            "gates": gates,
            "tuning_cells": tuning_cells,
            "tuning_seed_ids": tuning_seeds,
            "confirmatory_seed_ids": confirmatory_seeds,
            "validation_stream_ids": estimator_protocol.validation_stream_ids,
            "test_stream_ids": estimator_protocol.test_stream_ids,
            "particle_counts": estimator_protocol.particle_counts,
            "estimator_protocol_sha256": estimator_protocol.protocol_sha256,
            "tuning_attempt_keys": tuple(tuning_attempts),
            "terminal_checkpoint_keys": tuple(terminal_checkpoints),
            "validation_endpoint_keys": tuple(validation_endpoints),
            "test_endpoint_keys": tuple(test_endpoints),
            "raw_score_record_keys": tuple(raw_scores),
            "result_row_keys": tuple(f"result-row/{arm.arm_id}" for arm in arms),
            "figure_panel_keys": WT103_FIGURE_PANEL_KEYS,
            "figure_series_keys": _figure_series_keys(arms),
        }
        digest = owned_sha256("vfe4.wt103.endpoint-inventory.v1", payload)
        return cls(
            **payload,
            endpoint_inventory_sha256=digest,
        )  # type: ignore[arg-type]

    @property
    def tuning_attempt_count(self) -> int:
        return len(self.tuning_attempt_keys)

    @property
    def terminal_checkpoint_count(self) -> int:
        return len(self.terminal_checkpoint_keys)

    @property
    def validation_endpoint_count(self) -> int:
        return len(self.validation_endpoint_keys)

    @property
    def test_endpoint_count(self) -> int:
        return len(self.test_endpoint_keys)

    @property
    def raw_score_record_count(self) -> int:
        return len(self.raw_score_record_keys)

    @property
    def result_row_count(self) -> int:
        return len(self.result_row_keys)

    @property
    def figure_panel_count(self) -> int:
        return len(self.figure_panel_keys)

    @property
    def figure_series_count(self) -> int:
        return len(self.figure_series_keys)


def validate_endpoint_inventory(
    inventory: EndpointInventory,
    *,
    expected_sha256: str,
) -> None:
    """Reconstruct and validate an endpoint inventory without external reads."""

    if type(inventory) is not EndpointInventory:
        raise ValueError("inventory must be an exact EndpointInventory")
    _require_sha256(expected_sha256, "expected_sha256")
    inventory.__post_init__()
    rebuilt = EndpointInventory.create(
        inventory.arms,
        inventory.gates,
        inventory.tuning_cells,
        inventory.tuning_seed_ids,
        inventory.confirmatory_seed_ids,
        EstimatorProtocol.create(),
    )
    if rebuilt != inventory:
        raise ValueError("endpoint inventory contains non-derived fields")
    if inventory.endpoint_inventory_sha256 != expected_sha256:
        raise ValueError("endpoint_inventory_sha256 differs from expected")


@dataclass(frozen=True, slots=True)
class RedirectHop:
    request_url: str
    response_url: str
    status_code: int

    def __post_init__(self) -> None:
        _require_text(self.request_url, "request_url")
        _require_text(self.response_url, "response_url")
        if type(self.status_code) is not int or self.status_code not in (
            301,
            302,
            303,
            307,
            308,
        ):
            raise ValueError("redirect status_code is not an HTTP redirect")


@dataclass(frozen=True, slots=True)
class ArchiveMemberIdentity:
    split: Literal["train", "validation", "test"]
    member_name: str
    compression_method: int
    compressed_size_bytes: int
    uncompressed_size_bytes: int
    crc32: int
    payload_sha256: str

    def __post_init__(self) -> None:
        if self.split not in ("train", "validation", "test"):
            raise ValueError("archive member split is invalid")
        _require_text(self.member_name, "member_name")
        _require_int(
            self.compression_method,
            "compression_method",
            minimum=0,
        )
        _require_int(
            self.compressed_size_bytes,
            "compressed_size_bytes",
            minimum=0,
        )
        _require_int(
            self.uncompressed_size_bytes,
            "uncompressed_size_bytes",
            minimum=1,
        )
        if type(self.crc32) is not int or self.crc32 < 0 or self.crc32 > 0xFFFFFFFF:
            raise ValueError("crc32 must be an unsigned 32-bit integer")
        _require_sha256(self.payload_sha256, "payload_sha256")


@dataclass(frozen=True, slots=True)
class StagedWikiText103AcquisitionObservation:
    schema_version: Literal["wt103-staged-acquisition-observation-v1"]
    evidence_scope: Literal["nonproduction_staged_observation"]
    archive_sha256: str
    central_directory_sha256: str
    source_page_sha256: str
    license_raw_slice_sha256: str
    observation_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != "wt103-staged-acquisition-observation-v1"
            or self.evidence_scope != "nonproduction_staged_observation"
        ):
            raise ValueError("staged acquisition observation has invalid scope")
        for name in (
            "archive_sha256",
            "central_directory_sha256",
            "source_page_sha256",
            "license_raw_slice_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        expected = owned_sha256(
            "vfe4.wt103.staged-acquisition-observation.v1",
            _record_payload(self, omit=("observation_sha256",)),
        )
        _require_sha256(self.observation_sha256, "observation_sha256")
        if self.observation_sha256 != expected:
            raise ValueError("observation_sha256 does not match staged observation")

    @classmethod
    def create(
        cls,
        *,
        archive_sha256: str,
        central_directory_sha256: str,
        source_page_sha256: str,
        license_raw_slice_sha256: str,
    ) -> "StagedWikiText103AcquisitionObservation":
        payload = {
            "schema_version": "wt103-staged-acquisition-observation-v1",
            "evidence_scope": "nonproduction_staged_observation",
            "archive_sha256": archive_sha256,
            "central_directory_sha256": central_directory_sha256,
            "source_page_sha256": source_page_sha256,
            "license_raw_slice_sha256": license_raw_slice_sha256,
        }
        return cls(
            **payload,
            observation_sha256=owned_sha256(
                "vfe4.wt103.staged-acquisition-observation.v1",
                payload,
            ),
        )  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class FinalizedWikiText103SourceRecord:
    schema_version: Literal["wt103-finalized-source-record-v1"]
    acquisition_observation_sha256: str
    archive_request_url: str
    archive_final_url: str
    archive_redirect_chain: tuple[RedirectHop, ...]
    source_page_request_url: str
    source_page_final_url: str
    source_page_redirect_chain: tuple[RedirectHop, ...]
    archive_size_bytes: int
    archive_sha256: str
    archive_content_type: str | None
    central_directory_sha256: str
    members: tuple[
        ArchiveMemberIdentity,
        ArchiveMemberIdentity,
        ArchiveMemberIdentity,
    ]
    source_page_size_bytes: int
    source_page_content_type: str
    source_page_sha256: str
    license_paragraph_start_byte: int
    license_paragraph_end_byte: int
    license_raw_slice_sha256: str
    license_declaration: str
    license_hrefs: tuple[str, ...]
    installed_distribution_sha256: str
    tokenizer_tables_sha256: str
    production_tokenizer_spec_sha256: str
    production_token_cache_set_sha256: str
    schedule_set_sha256: str
    dependency_lock_sha256: str
    validator_sha256: str
    record_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != "wt103-finalized-source-record-v1":
            raise ValueError("unsupported finalized source schema")
        for name in (
            "acquisition_observation_sha256",
            "archive_sha256",
            "central_directory_sha256",
            "source_page_sha256",
            "license_raw_slice_sha256",
            "installed_distribution_sha256",
            "tokenizer_tables_sha256",
            "production_tokenizer_spec_sha256",
            "production_token_cache_set_sha256",
            "schedule_set_sha256",
            "dependency_lock_sha256",
            "validator_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        for name in (
            "archive_request_url",
            "archive_final_url",
            "source_page_request_url",
            "source_page_final_url",
            "source_page_content_type",
            "license_declaration",
        ):
            _require_text(getattr(self, name), name)
        if self.archive_content_type is not None:
            _require_text(self.archive_content_type, "archive_content_type")
        for name in ("archive_redirect_chain", "source_page_redirect_chain"):
            chain = getattr(self, name)
            if type(chain) is not tuple or any(
                type(item) is not RedirectHop for item in chain
            ):
                raise ValueError(f"{name} must contain exact RedirectHop records")
        if (
            type(self.members) is not tuple
            or len(self.members) != 3
            or any(type(item) is not ArchiveMemberIdentity for item in self.members)
            or tuple(item.split for item in self.members)
            != ("train", "validation", "test")
        ):
            raise ValueError("members must be exact train/validation/test records")
        _require_int(
            self.archive_size_bytes,
            "archive_size_bytes",
            minimum=1,
        )
        _require_int(
            self.source_page_size_bytes,
            "source_page_size_bytes",
            minimum=1,
        )
        _require_int(
            self.license_paragraph_start_byte,
            "license_paragraph_start_byte",
            minimum=0,
        )
        _require_int(
            self.license_paragraph_end_byte,
            "license_paragraph_end_byte",
            minimum=1,
        )
        if not (
            self.license_paragraph_start_byte
            < self.license_paragraph_end_byte
            <= self.source_page_size_bytes
        ):
            raise ValueError("license paragraph offsets are outside source page")
        _require_text_tuple(self.license_hrefs, "license_hrefs")
        expected = owned_sha256(
            "vfe4.wt103.finalized-source-record.v1",
            _record_payload(self, omit=("record_sha256",)),
        )
        _require_sha256(self.record_sha256, "record_sha256")
        if self.record_sha256 != expected:
            raise ValueError("record_sha256 does not match finalized source")

    @classmethod
    def create(cls, **values: object) -> "FinalizedWikiText103SourceRecord":
        payload = {
            "schema_version": "wt103-finalized-source-record-v1",
            **values,
        }
        return cls(
            **payload,
            record_sha256=owned_sha256(
                "vfe4.wt103.finalized-source-record.v1",
                payload,
            ),
        )  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class CandidateTokenizerContract:
    """The only production-scope tokenizer facts allowed before Task 13."""

    distribution: Literal["tiktoken"] = "tiktoken"
    version: Literal["0.12.0"] = "0.12.0"
    encoding_name: Literal["gpt2"] = "gpt2"

    def __post_init__(self) -> None:
        if (
            self.distribution,
            self.version,
            self.encoding_name,
        ) != ("tiktoken", "0.12.0", "gpt2"):
            raise ValueError("candidate tokenizer contract is frozen")


@dataclass(frozen=True, slots=True)
class SyntheticFixtureTokenizerSpec:
    schema_version: Literal["wt103-synthetic-fixture-tokenizer-spec-v1"]
    hash_domain: Literal["vfe4.wt103.synthetic-fixture-tokenizer-spec.v1\x00"]
    adapter_sha256: str
    fixture_sha256: str
    spec_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != "wt103-synthetic-fixture-tokenizer-spec-v1"
            or self.hash_domain != "vfe4.wt103.synthetic-fixture-tokenizer-spec.v1\x00"
        ):
            raise ValueError("synthetic fixture tokenizer domain is frozen")
        _require_sha256(self.adapter_sha256, "adapter_sha256")
        _require_sha256(self.fixture_sha256, "fixture_sha256")
        expected = owned_sha256(
            "vfe4.wt103.synthetic-fixture-tokenizer-spec.v1",
            _record_payload(self, omit=("spec_sha256",)),
        )
        _require_sha256(self.spec_sha256, "spec_sha256")
        if self.spec_sha256 != expected:
            raise ValueError("spec_sha256 does not match synthetic tokenizer")

    @classmethod
    def create(
        cls,
        *,
        adapter_sha256: str,
        fixture_sha256: str,
    ) -> "SyntheticFixtureTokenizerSpec":
        payload = {
            "schema_version": ("wt103-synthetic-fixture-tokenizer-spec-v1"),
            "hash_domain": ("vfe4.wt103.synthetic-fixture-tokenizer-spec.v1\x00"),
            "adapter_sha256": adapter_sha256,
            "fixture_sha256": fixture_sha256,
        }
        return cls(
            **payload,
            spec_sha256=owned_sha256(
                "vfe4.wt103.synthetic-fixture-tokenizer-spec.v1",
                payload,
            ),
        )  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class ProductionTokenizerSpec:
    schema_version: Literal["wt103-production-tokenizer-spec-v1"]
    hash_domain: Literal["vfe4.wt103.production-tokenizer-spec.v1\x00"]
    distribution: Literal["tiktoken"]
    version: Literal["0.12.0"]
    encoding_name: Literal["gpt2"]
    vocabulary_size: Literal[50257]
    eot_token_id: Literal[50256]
    corpus_method: Literal["encode_ordinary"]
    distribution_record_sha256: str
    regex_pattern_sha256: str
    mergeable_ranks_sha256: str
    special_tokens_sha256: str
    golden_vectors_sha256: str
    spec_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != "wt103-production-tokenizer-spec-v1"
            or self.hash_domain != "vfe4.wt103.production-tokenizer-spec.v1\x00"
            or self.distribution != "tiktoken"
            or self.version != "0.12.0"
            or self.encoding_name != "gpt2"
            or type(self.vocabulary_size) is not int
            or self.vocabulary_size != 50257
            or type(self.eot_token_id) is not int
            or self.eot_token_id != 50256
            or self.corpus_method != "encode_ordinary"
        ):
            raise ValueError("production tokenizer literal contract is frozen")
        for name in (
            "distribution_record_sha256",
            "regex_pattern_sha256",
            "mergeable_ranks_sha256",
            "special_tokens_sha256",
            "golden_vectors_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        expected = owned_sha256(
            "vfe4.wt103.production-tokenizer-spec.v1",
            _record_payload(self, omit=("spec_sha256",)),
        )
        _require_sha256(self.spec_sha256, "spec_sha256")
        if self.spec_sha256 != expected:
            raise ValueError("spec_sha256 does not match production tokenizer")

    @classmethod
    def create_verified(
        cls,
        *,
        distribution_record_sha256: str,
        regex_pattern_sha256: str,
        mergeable_ranks_sha256: str,
        special_tokens_sha256: str,
        golden_vectors_sha256: str,
    ) -> "ProductionTokenizerSpec":
        payload = {
            "schema_version": "wt103-production-tokenizer-spec-v1",
            "hash_domain": "vfe4.wt103.production-tokenizer-spec.v1\x00",
            "distribution": "tiktoken",
            "version": "0.12.0",
            "encoding_name": "gpt2",
            "vocabulary_size": 50257,
            "eot_token_id": 50256,
            "corpus_method": "encode_ordinary",
            "distribution_record_sha256": distribution_record_sha256,
            "regex_pattern_sha256": regex_pattern_sha256,
            "mergeable_ranks_sha256": mergeable_ranks_sha256,
            "special_tokens_sha256": special_tokens_sha256,
            "golden_vectors_sha256": golden_vectors_sha256,
        }
        return cls(
            **payload,
            spec_sha256=owned_sha256(
                "vfe4.wt103.production-tokenizer-spec.v1",
                payload,
            ),
        )  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class SyntheticFixtureTokenCacheIdentity:
    schema_version: Literal["wt103-synthetic-fixture-token-cache-v1"]
    tokenizer: SyntheticFixtureTokenizerSpec
    payload_sha256: str
    cache_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != "wt103-synthetic-fixture-token-cache-v1":
            raise ValueError("unsupported synthetic token cache schema")
        if type(self.tokenizer) is not SyntheticFixtureTokenizerSpec:
            raise ValueError("tokenizer must be an exact SyntheticFixtureTokenizerSpec")
        self.tokenizer.__post_init__()
        _require_sha256(self.payload_sha256, "payload_sha256")
        expected = owned_sha256(
            "vfe4.wt103.synthetic-fixture-token-cache.v1",
            _record_payload(self, omit=("cache_sha256",)),
        )
        _require_sha256(self.cache_sha256, "cache_sha256")
        if self.cache_sha256 != expected:
            raise ValueError("cache_sha256 does not match synthetic cache")

    @classmethod
    def create(
        cls,
        *,
        tokenizer: SyntheticFixtureTokenizerSpec,
        payload_sha256: str,
    ) -> "SyntheticFixtureTokenCacheIdentity":
        if type(tokenizer) is not SyntheticFixtureTokenizerSpec:
            raise ValueError("tokenizer must be an exact SyntheticFixtureTokenizerSpec")
        payload = {
            "schema_version": "wt103-synthetic-fixture-token-cache-v1",
            "tokenizer": tokenizer,
            "payload_sha256": payload_sha256,
        }
        return cls(
            **payload,
            cache_sha256=owned_sha256(
                "vfe4.wt103.synthetic-fixture-token-cache.v1",
                payload,
            ),
        )  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class ProductionTokenCacheIdentity:
    schema_version: Literal["wt103-production-token-cache-v1"]
    tokenizer: ProductionTokenizerSpec
    split: Literal["train", "validation", "test"]
    payload_sha256: str
    cache_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != "wt103-production-token-cache-v1":
            raise ValueError("unsupported production token cache schema")
        if type(self.tokenizer) is not ProductionTokenizerSpec:
            raise ValueError("tokenizer must be an exact ProductionTokenizerSpec")
        self.tokenizer.__post_init__()
        if self.split not in ("train", "validation", "test"):
            raise ValueError("production cache split is invalid")
        _require_sha256(self.payload_sha256, "payload_sha256")
        expected = owned_sha256(
            "vfe4.wt103.production-token-cache.v1",
            _record_payload(self, omit=("cache_sha256",)),
        )
        _require_sha256(self.cache_sha256, "cache_sha256")
        if self.cache_sha256 != expected:
            raise ValueError("cache_sha256 does not match production cache")

    @classmethod
    def create(
        cls,
        *,
        tokenizer: ProductionTokenizerSpec,
        split: Literal["train", "validation", "test"],
        payload_sha256: str,
    ) -> "ProductionTokenCacheIdentity":
        if type(tokenizer) is not ProductionTokenizerSpec:
            raise ValueError("tokenizer must be an exact ProductionTokenizerSpec")
        payload = {
            "schema_version": "wt103-production-token-cache-v1",
            "tokenizer": tokenizer,
            "split": split,
            "payload_sha256": payload_sha256,
        }
        return cls(
            **payload,
            cache_sha256=owned_sha256(
                "vfe4.wt103.production-token-cache.v1",
                payload,
            ),
        )  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class WindowManifest:
    schema_version: Literal["wt103-window-manifest-v1"]
    split: Literal["train", "validation", "test"]
    token_payload_sha256: str
    sequence_length: Literal[128]
    stride: Literal[128]
    window_count: int
    counted_targets: int
    payload_sha256: str
    manifest_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != "wt103-window-manifest-v1"
            or self.split not in ("train", "validation", "test")
            or type(self.sequence_length) is not int
            or self.sequence_length != 128
            or type(self.stride) is not int
            or self.stride != 128
        ):
            raise ValueError("window manifest literals are invalid")
        _require_sha256(self.token_payload_sha256, "token_payload_sha256")
        _require_int(self.window_count, "window_count", minimum=1)
        _require_int(self.counted_targets, "counted_targets", minimum=1)
        _require_sha256(self.payload_sha256, "payload_sha256")
        expected = owned_sha256(
            "vfe4.wt103.window-manifest.v1",
            _record_payload(self, omit=("manifest_sha256",)),
        )
        _require_sha256(self.manifest_sha256, "manifest_sha256")
        if self.manifest_sha256 != expected:
            raise ValueError("manifest_sha256 does not match window manifest")

    @classmethod
    def create(
        cls,
        *,
        split: Literal["train", "validation", "test"],
        token_payload_sha256: str,
        window_count: int,
        counted_targets: int,
        payload_sha256: str,
    ) -> "WindowManifest":
        payload = {
            "schema_version": "wt103-window-manifest-v1",
            "split": split,
            "token_payload_sha256": token_payload_sha256,
            "sequence_length": 128,
            "stride": 128,
            "window_count": window_count,
            "counted_targets": counted_targets,
            "payload_sha256": payload_sha256,
        }
        return cls(
            **payload,
            manifest_sha256=owned_sha256(
                "vfe4.wt103.window-manifest.v1",
                payload,
            ),
        )  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class PermutationManifest:
    schema_version: Literal["wt103-permutation-manifest-v1"]
    split: Literal["train"]
    pass_index: Literal[0, 1]
    data_order_seed: Literal[2026072199]
    bit_generator: Literal["PCG64"]
    numpy_version: str
    window_manifest_sha256: str
    payload_sha256: str
    manifest_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != "wt103-permutation-manifest-v1"
            or self.split != "train"
            or type(self.pass_index) is not int
            or self.pass_index not in (0, 1)
            or type(self.data_order_seed) is not int
            or self.data_order_seed != 2026072199
            or self.bit_generator != "PCG64"
        ):
            raise ValueError("permutation manifest literals are invalid")
        _require_text(self.numpy_version, "numpy_version")
        _require_sha256(
            self.window_manifest_sha256,
            "window_manifest_sha256",
        )
        _require_sha256(self.payload_sha256, "payload_sha256")
        expected = owned_sha256(
            "vfe4.wt103.permutation-manifest.v1",
            _record_payload(self, omit=("manifest_sha256",)),
        )
        _require_sha256(self.manifest_sha256, "manifest_sha256")
        if self.manifest_sha256 != expected:
            raise ValueError("manifest_sha256 does not match permutation manifest")

    @classmethod
    def create(
        cls,
        *,
        pass_index: Literal[0, 1],
        numpy_version: str,
        window_manifest_sha256: str,
        payload_sha256: str,
    ) -> "PermutationManifest":
        payload = {
            "schema_version": "wt103-permutation-manifest-v1",
            "split": "train",
            "pass_index": pass_index,
            "data_order_seed": 2026072199,
            "bit_generator": "PCG64",
            "numpy_version": numpy_version,
            "window_manifest_sha256": window_manifest_sha256,
            "payload_sha256": payload_sha256,
        }
        return cls(
            **payload,
            manifest_sha256=owned_sha256(
                "vfe4.wt103.permutation-manifest.v1",
                payload,
            ),
        )  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class DataCursor:
    schema_version: Literal["wt103-data-cursor-v1"]
    split: Literal["train", "validation", "test"]
    pass_index: int
    permutation_sha256: str
    next_batch_ordinal: int
    next_window_ids: tuple[int, ...]
    counted_targets: int
    cursor_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != "wt103-data-cursor-v1" or self.split not in (
            "train",
            "validation",
            "test",
        ):
            raise ValueError("data cursor schema/split is invalid")
        _require_int(self.pass_index, "pass_index", minimum=0)
        _require_sha256(self.permutation_sha256, "permutation_sha256")
        _require_int(
            self.next_batch_ordinal,
            "next_batch_ordinal",
            minimum=0,
        )
        if (
            type(self.next_window_ids) is not tuple
            or any(type(item) is not int or item < 0 for item in self.next_window_ids)
            or len(set(self.next_window_ids)) != len(self.next_window_ids)
        ):
            raise ValueError("next_window_ids must be an immutable unique int tuple")
        _require_int(self.counted_targets, "counted_targets", minimum=0)
        expected = owned_sha256(
            "vfe4.wt103.data-cursor.v1",
            _record_payload(self, omit=("cursor_sha256",)),
        )
        _require_sha256(self.cursor_sha256, "cursor_sha256")
        if self.cursor_sha256 != expected:
            raise ValueError("cursor_sha256 does not match data cursor")

    @classmethod
    def create(
        cls,
        *,
        split: Literal["train", "validation", "test"],
        pass_index: int,
        permutation_sha256: str,
        next_batch_ordinal: int,
        next_window_ids: tuple[int, ...],
        counted_targets: int,
    ) -> "DataCursor":
        payload = {
            "schema_version": "wt103-data-cursor-v1",
            "split": split,
            "pass_index": pass_index,
            "permutation_sha256": permutation_sha256,
            "next_batch_ordinal": next_batch_ordinal,
            "next_window_ids": next_window_ids,
            "counted_targets": counted_targets,
        }
        return cls(
            **payload,
            cursor_sha256=owned_sha256(
                "vfe4.wt103.data-cursor.v1",
                payload,
            ),
        )  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class TrainingSparsityCertificate:
    schema_version: Literal["wt103-training-sparsity-v1"]
    git_head: str
    dirty_digest: str
    profile_sha256: str
    factory_set_sha256: str
    endpoint_inventory_sha256: str
    whitelist_sha256: str
    forbidden_shape_sha256: str
    trace_set_sha256: str
    formula_reconciliation_sha256: str
    negative_controls_sha256: str
    status: GateStatus
    obligations: tuple[str, ...]
    certificate_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != "wt103-training-sparsity-v1":
            raise ValueError("unsupported training sparsity schema")
        if (
            type(self.git_head) is not str
            or len(self.git_head) not in (40, 64)
            or any(character not in _HEX for character in self.git_head)
        ):
            raise ValueError("git_head must be a concrete hex object id")
        for name in (
            "dirty_digest",
            "profile_sha256",
            "factory_set_sha256",
            "endpoint_inventory_sha256",
            "whitelist_sha256",
            "forbidden_shape_sha256",
            "trace_set_sha256",
            "formula_reconciliation_sha256",
            "negative_controls_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if type(self.status) is not GateStatus:
            raise ValueError("status must be an exact GateStatus")
        _require_text_tuple(self.obligations, "obligations")
        if self.status is GateStatus.PASS and self.obligations:
            raise ValueError("PASS sparsity certificate cannot retain obligations")
        if self.status is not GateStatus.PASS and not self.obligations:
            raise ValueError("non-PASS sparsity certificate requires an obligation")
        expected = owned_sha256(
            "vfe4.wt103.training-sparsity-certificate.v1",
            _record_payload(self, omit=("certificate_sha256",)),
        )
        _require_sha256(self.certificate_sha256, "certificate_sha256")
        if self.certificate_sha256 != expected:
            raise ValueError("certificate_sha256 does not match sparsity certificate")


@dataclass(frozen=True, slots=True)
class CheckpointBundle:
    schema_version: Literal["wt103-checkpoint-bundle-v1"]
    logical_key: str
    arm_spec_sha256: str
    experiment_plan_sha256: str
    config_sha256: str
    scientific_state_sha256: str
    bundle_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != "wt103-checkpoint-bundle-v1":
            raise ValueError("unsupported checkpoint bundle schema")
        _require_text(self.logical_key, "logical_key")
        for name in (
            "arm_spec_sha256",
            "experiment_plan_sha256",
            "config_sha256",
            "scientific_state_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        expected = owned_sha256(
            "vfe4.wt103.checkpoint-bundle.v1",
            _record_payload(self, omit=("bundle_sha256",)),
        )
        _require_sha256(self.bundle_sha256, "bundle_sha256")
        if self.bundle_sha256 != expected:
            raise ValueError("bundle_sha256 does not match checkpoint bundle")


@dataclass(frozen=True, slots=True)
class WT103CheckpointIdentity:
    schema_version: Literal["wt103-checkpoint-identity-v1"]
    logical_key: str
    checkpoint_role: Literal["resume_only", "terminal_scoring"]
    scientific_state_sha256: str
    checkpoint_payload_sha256: str
    checkpoint_manifest_body_sha256: str
    artifact_sha256: str
    size_bytes: int
    checkpoint_identity_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != "wt103-checkpoint-identity-v1":
            raise ValueError("unsupported checkpoint identity schema")
        _require_text(self.logical_key, "logical_key")
        if self.checkpoint_role not in ("resume_only", "terminal_scoring"):
            raise ValueError("unknown checkpoint role")
        for name in (
            "scientific_state_sha256",
            "checkpoint_payload_sha256",
            "checkpoint_manifest_body_sha256",
            "artifact_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        expected_artifact = hashlib.sha256(
            b"vfe4-checkpoint-artifact-v1\x00"
            + bytes.fromhex(self.checkpoint_payload_sha256)
            + bytes.fromhex(self.checkpoint_manifest_body_sha256)
        ).hexdigest()
        if self.artifact_sha256 != expected_artifact:
            raise ValueError("artifact_sha256 does not match checkpoint parts")
        _require_int(self.size_bytes, "size_bytes", minimum=1)
        expected = owned_sha256(
            "vfe4.wt103.checkpoint-identity.v1",
            _record_payload(self, omit=("checkpoint_identity_sha256",)),
        )
        _require_sha256(
            self.checkpoint_identity_sha256,
            "checkpoint_identity_sha256",
        )
        if self.checkpoint_identity_sha256 != expected:
            raise ValueError("checkpoint identity hash does not match")


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    bundle: CheckpointBundle
    identity: WT103CheckpointIdentity

    def __post_init__(self) -> None:
        if type(self.bundle) is not CheckpointBundle:
            raise ValueError("bundle must be an exact CheckpointBundle")
        if type(self.identity) is not WT103CheckpointIdentity:
            raise ValueError("identity must be an exact WT103CheckpointIdentity")
        if (
            self.bundle.logical_key != self.identity.logical_key
            or self.bundle.scientific_state_sha256
            != self.identity.scientific_state_sha256
        ):
            raise ValueError("loaded checkpoint bundle/identity disagree")


@dataclass(frozen=True, slots=True)
class WT103NllTotals:
    schema_version: Literal["wt103-nll-totals-v1"]
    scorer_kind: Literal["exact_autoregressive", "weighted_smc"]
    summed_nll: float
    counted_targets: int
    nll_per_token: float
    perplexity: float
    estimator_stream_id: int | None
    particle_count: int | None
    cache_audit_sha256: str
    totals_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != "wt103-nll-totals-v1" or self.scorer_kind not in (
            "exact_autoregressive",
            "weighted_smc",
        ):
            raise ValueError("NLL totals schema/scorer is invalid")
        _require_float(self.summed_nll, "summed_nll", minimum=0.0)
        _require_int(self.counted_targets, "counted_targets", minimum=1)
        _require_float(self.nll_per_token, "nll_per_token", minimum=0.0)
        _require_float(self.perplexity, "perplexity", minimum=1.0)
        expected_nll = self.summed_nll / self.counted_targets
        if self.nll_per_token != expected_nll:
            raise ValueError("nll_per_token must be derived from corpus sums")
        expected_ppl = math.exp(self.nll_per_token)
        if self.perplexity != expected_ppl:
            raise ValueError("perplexity must be exp(nll_per_token)")
        if self.scorer_kind == "exact_autoregressive":
            if self.estimator_stream_id is not None or self.particle_count is not None:
                raise ValueError("exact scoring cannot fabricate SMC identity")
        else:
            _require_int(
                self.estimator_stream_id,
                "estimator_stream_id",
                minimum=0,
            )
            if (
                type(self.particle_count) is not int
                or self.particle_count not in WT103_PARTICLE_COUNTS
            ):
                raise ValueError("weighted SMC particle count is not frozen")
        _require_sha256(self.cache_audit_sha256, "cache_audit_sha256")
        expected = owned_sha256(
            "vfe4.wt103.nll-totals.v1",
            _record_payload(self, omit=("totals_sha256",)),
        )
        _require_sha256(self.totals_sha256, "totals_sha256")
        if self.totals_sha256 != expected:
            raise ValueError("totals_sha256 does not match NLL totals")


@dataclass(frozen=True, slots=True)
class MetricValue:
    name: str
    applicability: Literal["applicable", "not_applicable"]
    reason: str
    numerator: float | None
    denominator: int | None
    value: float | None
    units: str

    def __post_init__(self) -> None:
        _require_text(self.name, "name")
        if self.applicability not in ("applicable", "not_applicable"):
            raise ValueError("metric applicability is invalid")
        _require_text(self.reason, "reason")
        _require_text(self.units, "units")
        if self.applicability == "not_applicable":
            if any(
                item is not None
                for item in (self.numerator, self.denominator, self.value)
            ):
                raise ValueError(
                    "not-applicable metric values cannot fabricate zero/null data"
                )
            return
        if self.numerator is not None:
            _require_float(self.numerator, "numerator")
        if self.denominator is not None:
            _require_int(self.denominator, "denominator", minimum=1)
        if self.value is not None:
            _require_float(self.value, "value")
        if self.value is None:
            raise ValueError("applicable metric requires a value")


@dataclass(frozen=True, slots=True)
class MetricRecord:
    schema_version: Literal["wt103-metric-record-v1"]
    ordinal: int
    run_id: str
    arm_id: str
    seed_id: int
    phase: str
    split: Literal["train", "validation", "test", "not_applicable"]
    step: int
    pass_index: int
    previous_record_sha256: str
    values: tuple[MetricValue, ...]
    record_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != "wt103-metric-record-v1":
            raise ValueError("unsupported metric schema")
        _require_int(self.ordinal, "ordinal", minimum=0)
        _require_text(self.run_id, "run_id")
        if self.arm_id not in WT103_ARM_IDS:
            raise ValueError("metric arm_id is not in the frozen inventory")
        _require_int(self.seed_id, "seed_id", minimum=0)
        _require_text(self.phase, "phase")
        if self.split not in (
            "train",
            "validation",
            "test",
            "not_applicable",
        ):
            raise ValueError("metric split is invalid")
        _require_int(self.step, "step", minimum=0)
        _require_int(self.pass_index, "pass_index", minimum=0)
        _require_sha256(
            self.previous_record_sha256,
            "previous_record_sha256",
        )
        if (
            type(self.values) is not tuple
            or not self.values
            or any(type(item) is not MetricValue for item in self.values)
            or len({item.name for item in self.values}) != len(self.values)
        ):
            raise ValueError("metric values must be unique immutable records")
        expected = owned_sha256(
            "vfe4.wt103.metric-record.v1",
            _record_payload(self, omit=("record_sha256",)),
        )
        _require_sha256(self.record_sha256, "record_sha256")
        if self.record_sha256 != expected:
            raise ValueError("record_sha256 does not match metric record")


@dataclass(frozen=True, slots=True)
class WT103UpdateRecord:
    schema_version: Literal["wt103-update-record-v1"]
    arm_id: str
    phase: str
    update_label: Literal["adam_proposal"]
    accepted: bool
    rejection_reason: str | None
    expected_autograd_scope: Literal["m_step", "e_and_m"]
    observed_autograd_scope: Literal["m_step", "e_and_m"]
    snapshot_sha256: str | None
    optimizer_state_sha256: str
    scheduler_state_sha256: str
    update_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != "wt103-update-record-v1"
            or self.arm_id not in WT103_ARM_IDS
            or self.update_label != "adam_proposal"
            or type(self.accepted) is not bool
            or self.expected_autograd_scope not in ("m_step", "e_and_m")
            or self.observed_autograd_scope not in ("m_step", "e_and_m")
            or self.observed_autograd_scope != self.expected_autograd_scope
        ):
            raise ValueError("update record literals/scope are invalid")
        _require_text(self.phase, "phase")
        if self.accepted and self.rejection_reason is not None:
            raise ValueError("accepted update cannot retain a rejection reason")
        if not self.accepted:
            _require_text(self.rejection_reason, "rejection_reason")
        if self.snapshot_sha256 is not None:
            _require_sha256(self.snapshot_sha256, "snapshot_sha256")
        for name in ("optimizer_state_sha256", "scheduler_state_sha256"):
            _require_sha256(getattr(self, name), name)
        expected = owned_sha256(
            "vfe4.wt103.update-record.v1",
            _record_payload(self, omit=("update_sha256",)),
        )
        _require_sha256(self.update_sha256, "update_sha256")
        if self.update_sha256 != expected:
            raise ValueError("update_sha256 does not match update record")


@dataclass(frozen=True, slots=True)
class WT103EvaluationRecord:
    schema_version: Literal["wt103-evaluation-record-v1"]
    endpoint_key: str
    checkpoint: WT103CheckpointIdentity
    totals: WT103NllTotals
    target_blind_cache_audit_passed: Literal[True]
    evaluation_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != "wt103-evaluation-record-v1":
            raise ValueError("unsupported evaluation record schema")
        _require_text(self.endpoint_key, "endpoint_key")
        if type(self.checkpoint) is not WT103CheckpointIdentity:
            raise ValueError("evaluation checkpoint identity has wrong type")
        if self.checkpoint.checkpoint_role != "terminal_scoring":
            raise ValueError("evaluation requires a terminal-scoring checkpoint")
        if type(self.totals) is not WT103NllTotals:
            raise ValueError("evaluation totals have wrong type")
        if self.target_blind_cache_audit_passed is not True:
            raise ValueError("evaluation must pass target-blind cache audit")
        expected = owned_sha256(
            "vfe4.wt103.evaluation-record.v1",
            _record_payload(self, omit=("evaluation_sha256",)),
        )
        _require_sha256(self.evaluation_sha256, "evaluation_sha256")
        if self.evaluation_sha256 != expected:
            raise ValueError("evaluation_sha256 does not match record")


@dataclass(frozen=True, slots=True)
class WT103RunRecord:
    schema_version: Literal["wt103-run-record-v1"]
    run_id: str
    arm_spec_sha256: str
    seed_id: int
    config_sha256: str
    endpoint_inventory_sha256: str
    terminal_checkpoint_identity_sha256: str
    metric_head_sha256: str
    failure_head_sha256: str
    disposition: Literal["complete", "failed", "inconclusive"]
    run_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != "wt103-run-record-v1":
            raise ValueError("unsupported run record schema")
        _require_text(self.run_id, "run_id")
        _require_int(self.seed_id, "seed_id", minimum=0)
        for name in (
            "arm_spec_sha256",
            "config_sha256",
            "endpoint_inventory_sha256",
            "terminal_checkpoint_identity_sha256",
            "metric_head_sha256",
            "failure_head_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if self.disposition not in ("complete", "failed", "inconclusive"):
            raise ValueError("unknown run disposition")
        expected = owned_sha256(
            "vfe4.wt103.run-record.v1",
            _record_payload(self, omit=("run_sha256",)),
        )
        _require_sha256(self.run_sha256, "run_sha256")
        if self.run_sha256 != expected:
            raise ValueError("run_sha256 does not match run record")


@dataclass(frozen=True, slots=True)
class WT103ExperimentRecord:
    schema_version: Literal["wt103-experiment-record-v1"]
    endpoint_inventory_sha256: str
    source_record_sha256: str
    training_config_sha256: str
    run_record_sha256s: tuple[str, ...]
    result_row_sha256s: tuple[str, ...]
    figure_input_sha256: str
    terminal_status: GateStatus
    obligations: tuple[str, ...]
    experiment_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != "wt103-experiment-record-v1":
            raise ValueError("unsupported experiment record schema")
        for name in (
            "endpoint_inventory_sha256",
            "source_record_sha256",
            "training_config_sha256",
            "figure_input_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        for name in ("run_record_sha256s", "result_row_sha256s"):
            value = getattr(self, name)
            if type(value) is not tuple or not value or len(set(value)) != len(value):
                raise ValueError(f"{name} must be a nonempty unique tuple")
            for index, digest in enumerate(value):
                _require_sha256(digest, f"{name}[{index}]")
        if type(self.terminal_status) is not GateStatus:
            raise ValueError("terminal_status must be an exact GateStatus")
        _require_text_tuple(self.obligations, "obligations")
        if self.terminal_status is GateStatus.PASS and self.obligations:
            raise ValueError("PASS experiment cannot retain obligations")
        if self.terminal_status is not GateStatus.PASS and not self.obligations:
            raise ValueError("non-PASS experiment requires an obligation")
        expected = owned_sha256(
            "vfe4.wt103.experiment-record.v1",
            _record_payload(self, omit=("experiment_sha256",)),
        )
        _require_sha256(self.experiment_sha256, "experiment_sha256")
        if self.experiment_sha256 != expected:
            raise ValueError("experiment_sha256 does not match record")


__all__ = [
    "A0ArchitectureProfile",
    "A0FormulaRecord",
    "AdamWProfile",
    "ArchiveMemberIdentity",
    "CandidateTokenizerContract",
    "CadenceProfile",
    "CheckpointBundle",
    "CheckpointProfile",
    "DataCursor",
    "EndpointInventory",
    "EstimatorProtocol",
    "FinalizedWikiText103SourceRecord",
    "LoadedCheckpoint",
    "MetricRecord",
    "MetricValue",
    "NonclaimProfile",
    "PermutationManifest",
    "PrecisionProfile",
    "ProductionTokenCacheIdentity",
    "ProductionTokenizerSpec",
    "RedirectHop",
    "ResourceProfile",
    "SchedulerProfile",
    "ScientificPreconditionProfile",
    "SchemaProfile",
    "StagedWikiText103AcquisitionObservation",
    "StatisticalProfile",
    "SyntheticFixtureTokenCacheIdentity",
    "SyntheticFixtureTokenizerSpec",
    "TrainingSparsityCertificate",
    "WT103ArmSpec",
    "WT103CheckpointIdentity",
    "WT103EvaluationRecord",
    "WT103ExperimentProfile",
    "WT103ExperimentRecord",
    "WT103GateSpec",
    "WT103NllTotals",
    "WT103RunRecord",
    "WT103UpdateRecord",
    "WT103_A0_HIDDEN_WIDTH_CANDIDATES",
    "WT103_ARM_IDS",
    "WT103_CONFIRMATORY_SEED_IDS",
    "WT103_FIGURE_PANEL_KEYS",
    "WT103_GATE_IDS",
    "WT103_PARTICLE_COUNTS",
    "WT103_TEST_STREAM_IDS",
    "WT103_TUNING_CELLS",
    "WT103_TUNING_SEED_IDS",
    "WT103_VALIDATION_STREAM_IDS",
    "WindowManifest",
    "canonical_json_bytes",
    "default_wt103_arm_specs",
    "default_wt103_gate_specs",
    "owned_sha256",
    "validate_endpoint_inventory",
]
