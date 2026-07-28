"""Additive executable H6-Prediction v3 protocol records.

These records do not widen any v1/v2 schema.  They bind the authorities needed
by the executable path while remaining pure: importing this module performs no
Torch, CUDA, corpus, artifact, or repository operation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

from .h6 import (
    H6ArmPhaseSchedule,
    H6OuterSchedule,
    TrainingPhase,
    canonical_json_bytes,
)


_LOWER_HEX = frozenset("0123456789abcdef")


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_git_head(value: object, name: str = "git_head") -> str:
    if (
        type(value) is not str
        or len(value) != 40
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase Git object name")
    return value


def _require_nonempty(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _owned_hash(domain: str, payload: object) -> str:
    return hashlib.sha256(
        domain.encode("ascii") + b"\x00" + canonical_json_bytes(payload)
    ).hexdigest()


H6_DETERMINISTIC_POLICY_DESCRIPTOR = {
    "schema_version": "h6-cuda-determinism-policy-v1",
    "cublas_workspace_config": ":4096:8",
    "deterministic_algorithms": True,
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
    "cuda_matmul_allow_fp16_reduced_precision_reduction": False,
    "cuda_matmul_allow_bf16_reduced_precision_reduction": False,
    "cuda_matmul_allow_fp16_accumulation": False,
    "flash_sdp": False,
    "memory_efficient_sdp": False,
    "cudnn_sdp": False,
    "math_sdp": True,
}
H6_DETERMINISTIC_POLICY_SHA256 = _owned_hash(
    "vfe4.h6.cuda-determinism-policy.v1",
    H6_DETERMINISTIC_POLICY_DESCRIPTOR,
)
H6_COUNTER_MAPPING_SHA256 = _owned_hash(
    "vfe4.h6.training-counter-mapping.v1",
    {
        "normal_domain": "vfe4.h6.training-rmc-normal.v1",
        "block_expansion": "sha256-four-little-endian-uint64-v1",
        "uniform_mapping": "estimator-stream-open-uniform-v1",
        "normal_mapping": "estimator-stream-paired-box-muller-v1",
        "key_coordinates": (
            "attempt_spec_sha256",
            "pass_index",
            "batch_index",
            "phase",
            "example_ordinal",
            "sample_ordinal",
            "draw_block",
        ),
        "sample_ordinal": 0,
    },
)
H6_NO_COUNTER_CONSUMPTION_SHA256 = _owned_hash(
    "vfe4.h6.no-counter-consumption.v1",
    {
        "continuous_noise": "absent",
        "draw_blocks_consumed": 0,
    },
)
H6_PHASE_OWNERSHIP_SHA256 = _owned_hash(
    "vfe4.h6.phase-ownership.v3",
    {
        "latent_phases": (
            TrainingPhase.RECOGNITION_ADAMW.value,
            TrainingPhase.IMMUTABLE_DETACHED_SNAPSHOT.value,
            TrainingPhase.MODEL_ADAMW.value,
        ),
        "snapshot": "fresh-post-recognition-step-detached-v1",
        "optimizer_overlap": "forbidden",
        "minimized_complete_scalar": "negative-elbo",
        "minimized_emission_scalar": "negative-live-emission-non-elbo",
    },
)
H6_CHECKPOINT_CODEC_SHA256 = _owned_hash(
    "vfe4.h6.checkpoint-codec.v3",
    {
        "tensor_order": "qualified-name-unicode-codepoint",
        "tensor_encoding": "contiguous-row-major-little-endian",
        "optimizer_binding": "stable-qualified-parameter-name",
        "aliases": "rejected",
        "hydration": (
            "fresh-cpu-module",
            "named-groups",
            "cpu-state",
            "validate",
            "device-move",
        ),
    },
)
H6_SCORING_INVENTORY_SHA256 = _owned_hash(
    "vfe4.h6.raw-endpoint-inventory.v4",
    {
        "exact_a0_corpus_totals": 8,
        "complete_a5_weighted_rows": 2048,
        "emission_a5_weighted_rows": 2048,
        "logical_row_count": 4104,
        "particle_counts": (128, 256, 512, 1024),
        "replicates_per_seed": 64,
    },
)
H6_PREDICTION_CONFIG_SCHEMA = "h6-prediction-config-v3"
H6_PREDICTION_READINESS_SCHEMA = "h6-prediction-readiness-v3"
H6_TRAINING_SCHEDULE_SCHEMA = "h6-training-schedule-v3"
H6_ATTEMPT_SPEC_SCHEMA = "h6-attempt-spec-v3"
H6_ATTEMPT_CURSOR_SCHEMA = "h6-attempt-cursor-v3"
H6_OBJECTIVE_MANIFEST_SCHEMA = "h6-objective-manifest-v3"
H6_OBJECTIVE_MANIFEST_SCHEMA_SHA256 = _owned_hash(
    "vfe4.h6.objective-manifest-schema.v3",
    {
        "schema_version": H6_OBJECTIVE_MANIFEST_SCHEMA,
        "objective_kinds": (
            "cross_entropy",
            "complete_elbo",
            "emission_only_ablation_non_elbo",
        ),
        "phase_field": "TrainingPhase",
        "recognition_law": "evaluated-v3-or-none",
        "detached_snapshot": "fresh-post-recognition-step-or-none",
        "factor_binding_order": "partition_receiver_digest",
        "raw_total_binding": "canonical-float64-bytes-sha256",
    },
)
H6_CHECKPOINT_SCHEMA = "h6-checkpoint-v3"
H6_RAW_ENDPOINT_INVENTORY_SCHEMA = "h6-raw-endpoint-inventory-v4"
H6_PREDICTION_METRICS_SCHEMA = "h6-prediction-metrics-v3"
H6_PREDICTION_RESULT_SCHEMA = "h6-prediction-result-v3"


@dataclass(frozen=True, slots=True)
class H6RecognitionEstimatorSpec:
    schema_version: Literal["h6-recognition-estimator-v3"]
    evaluation_method: Literal["reparameterized_mc"]
    continuous_base_samples_per_receiver_per_example_per_phase: Literal[1]
    categorical_evaluation: Literal["exact_support_sum"]
    gaussian_entropy: Literal["analytic"]
    component_sampling: Literal["common_random_numbers_per_receiver"]
    estimator_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name) for name in tuple(self.__dataclass_fields__)[:-1]
        }

    def __post_init__(self) -> None:
        expected = {
            "schema_version": "h6-recognition-estimator-v3",
            "evaluation_method": "reparameterized_mc",
            "continuous_base_samples_per_receiver_per_example_per_phase": 1,
            "categorical_evaluation": "exact_support_sum",
            "gaussian_entropy": "analytic",
            "component_sampling": "common_random_numbers_per_receiver",
        }
        if self.canonical_payload() != expected:
            raise ValueError("recognition estimator is not the frozen v3 estimator")
        if self.estimator_sha256 != _owned_hash(
            "vfe4.h6.recognition-estimator.v3", expected
        ):
            raise ValueError("recognition estimator identity is stale")

    @classmethod
    def create(cls) -> "H6RecognitionEstimatorSpec":
        payload = {
            "schema_version": "h6-recognition-estimator-v3",
            "evaluation_method": "reparameterized_mc",
            "continuous_base_samples_per_receiver_per_example_per_phase": 1,
            "categorical_evaluation": "exact_support_sum",
            "gaussian_entropy": "analytic",
            "component_sampling": "common_random_numbers_per_receiver",
        }
        return cls(
            **payload,  # type: ignore[arg-type]
            estimator_sha256=_owned_hash("vfe4.h6.recognition-estimator.v3", payload),
        )


@dataclass(frozen=True, slots=True)
class H6PredictionRuntimeIdentity:
    schema_version: Literal["h6-prediction-runtime-v3"]
    python_executable: Literal["C:/anaconda/python.exe"]
    python_version: str
    torch_full_version: str
    cuda_runtime_version: str
    training_device: Literal["cuda:0"]
    training_dtype: Literal["float64"]
    validation_device: Literal["cpu"]
    heldout_scoring_device: Literal["cpu"]
    scoring_dtype: Literal["float64"]
    cuda_device_name: str
    cuda_compute_capability: tuple[int, int]
    deterministic_policy_sha256: str
    runtime_identity_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name) for name in tuple(self.__dataclass_fields__)[:-1]
        }

    def __post_init__(self) -> None:
        if (
            self.schema_version != "h6-prediction-runtime-v3"
            or self.python_executable != "C:/anaconda/python.exe"
            or self.training_device != "cuda:0"
            or self.training_dtype != "float64"
            or self.validation_device != "cpu"
            or self.heldout_scoring_device != "cpu"
            or self.scoring_dtype != "float64"
        ):
            raise ValueError("runtime identity does not select the v3 device policy")
        for name in (
            "python_version",
            "torch_full_version",
            "cuda_runtime_version",
            "cuda_device_name",
        ):
            _require_nonempty(getattr(self, name), name)
        if (
            type(self.cuda_compute_capability) is not tuple
            or len(self.cuda_compute_capability) != 2
            or any(
                type(value) is not int or value < 0
                for value in self.cuda_compute_capability
            )
        ):
            raise ValueError("CUDA compute capability must be an integer pair")
        if self.deterministic_policy_sha256 != H6_DETERMINISTIC_POLICY_SHA256:
            raise ValueError("runtime deterministic policy identity is stale")
        if self.runtime_identity_sha256 != _owned_hash(
            "vfe4.h6.prediction-runtime.v3", self.canonical_payload()
        ):
            raise ValueError("runtime identity digest is stale")

    @classmethod
    def create(
        cls,
        *,
        python_version: str,
        torch_full_version: str,
        cuda_runtime_version: str,
        cuda_device_name: str,
        cuda_compute_capability: tuple[int, int],
    ) -> "H6PredictionRuntimeIdentity":
        payload = {
            "schema_version": "h6-prediction-runtime-v3",
            "python_executable": "C:/anaconda/python.exe",
            "python_version": python_version,
            "torch_full_version": torch_full_version,
            "cuda_runtime_version": cuda_runtime_version,
            "training_device": "cuda:0",
            "training_dtype": "float64",
            "validation_device": "cpu",
            "heldout_scoring_device": "cpu",
            "scoring_dtype": "float64",
            "cuda_device_name": cuda_device_name,
            "cuda_compute_capability": tuple(cuda_compute_capability),
            "deterministic_policy_sha256": H6_DETERMINISTIC_POLICY_SHA256,
        }
        return cls(
            **payload,  # type: ignore[arg-type]
            runtime_identity_sha256=_owned_hash(
                "vfe4.h6.prediction-runtime.v3", payload
            ),
        )


@dataclass(frozen=True, slots=True)
class H6TrainingScheduleV3:
    schedule_schema: Literal["h6-training-schedule-v3"]
    outer: H6OuterSchedule
    endpoint_phases: tuple[H6ArmPhaseSchedule, ...]
    recognition_estimator_sha256: str
    runtime_identity_sha256: str
    training_noise_domain: Literal["vfe4.h6.training-rmc-normal.v1"]
    counter_mapping_sha256: str
    phase_ownership_sha256: str
    checkpoint_codec_sha256: str
    schedule_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schedule_schema": self.schedule_schema,
            "outer_schedule_sha256": self.outer.outer_schedule_sha256,
            "phase_schedule_sha256": tuple(
                item.phase_schedule_sha256 for item in self.endpoint_phases
            ),
            "recognition_estimator_sha256": self.recognition_estimator_sha256,
            "runtime_identity_sha256": self.runtime_identity_sha256,
            "training_noise_domain": self.training_noise_domain,
            "counter_mapping_sha256": self.counter_mapping_sha256,
            "phase_ownership_sha256": self.phase_ownership_sha256,
            "checkpoint_codec_sha256": self.checkpoint_codec_sha256,
        }

    def __post_init__(self) -> None:
        if (
            self.schedule_schema != "h6-training-schedule-v3"
            or type(self.outer) is not H6OuterSchedule
            or type(self.endpoint_phases) is not tuple
            or not self.endpoint_phases
            or any(
                type(item) is not H6ArmPhaseSchedule for item in self.endpoint_phases
            )
        ):
            raise ValueError("v3 training schedule has invalid typed records")
        self.outer.__post_init__()
        for item in self.endpoint_phases:
            item.__post_init__()
        endpoints = tuple(item.endpoint_config_sha256 for item in self.endpoint_phases)
        if len(set(endpoints)) != len(endpoints):
            raise ValueError("v3 endpoint schedules must be unique")
        for name in (
            "recognition_estimator_sha256",
            "runtime_identity_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if (
            self.training_noise_domain != "vfe4.h6.training-rmc-normal.v1"
            or self.counter_mapping_sha256 != H6_COUNTER_MAPPING_SHA256
            or self.phase_ownership_sha256 != H6_PHASE_OWNERSHIP_SHA256
            or self.checkpoint_codec_sha256 != H6_CHECKPOINT_CODEC_SHA256
        ):
            raise ValueError("v3 schedule authority is stale")
        if self.schedule_sha256 != _owned_hash(
            "vfe4.h6.training-schedule.v3", self.canonical_payload()
        ):
            raise ValueError("v3 training schedule identity is stale")

    @classmethod
    def create(
        cls,
        *,
        outer: H6OuterSchedule,
        endpoint_phases: tuple[H6ArmPhaseSchedule, ...],
        estimator: H6RecognitionEstimatorSpec,
        runtime: H6PredictionRuntimeIdentity,
    ) -> "H6TrainingScheduleV3":
        estimator.__post_init__()
        runtime.__post_init__()
        values = {
            "schedule_schema": "h6-training-schedule-v3",
            "outer": outer,
            "endpoint_phases": tuple(endpoint_phases),
            "recognition_estimator_sha256": estimator.estimator_sha256,
            "runtime_identity_sha256": runtime.runtime_identity_sha256,
            "training_noise_domain": "vfe4.h6.training-rmc-normal.v1",
            "counter_mapping_sha256": H6_COUNTER_MAPPING_SHA256,
            "phase_ownership_sha256": H6_PHASE_OWNERSHIP_SHA256,
            "checkpoint_codec_sha256": H6_CHECKPOINT_CODEC_SHA256,
        }
        payload = {
            "schedule_schema": values["schedule_schema"],
            "outer_schedule_sha256": outer.outer_schedule_sha256,
            "phase_schedule_sha256": tuple(
                item.phase_schedule_sha256 for item in endpoint_phases
            ),
            "recognition_estimator_sha256": estimator.estimator_sha256,
            "runtime_identity_sha256": runtime.runtime_identity_sha256,
            "training_noise_domain": values["training_noise_domain"],
            "counter_mapping_sha256": values["counter_mapping_sha256"],
            "phase_ownership_sha256": values["phase_ownership_sha256"],
            "checkpoint_codec_sha256": values["checkpoint_codec_sha256"],
        }
        return cls(
            **values,  # type: ignore[arg-type]
            schedule_sha256=_owned_hash(
                "vfe4.h6.training-schedule.v3",
                payload,
            ),
        )


@dataclass(frozen=True, slots=True)
class H6PredictionV3ReadinessToken:
    readiness_schema: Literal["h6-prediction-readiness-v3"]
    status: Literal["PASS"]
    git_head: str
    dirty_digest: str
    experiment_config_sha256: str
    correctness_manifests: tuple[tuple[str, str], ...]
    h1_prefix_prior_manifest_sha256: str
    h1_prefix_prior_generative_factor_schema_sha256: str
    smc_bias_semantics_sha256: str
    smc_validation_manifest_sha256: str
    prefix_certificate_set_sha256: str
    a0_direct_exact_prefix_certificate_sha256: str
    h5_update_binding_sha256: str
    critical_values_sha256: str
    endpoint_smc_protocol_sha256: str
    attribution_matrix_sha256: str
    objective_gate_spec_sha256: str
    matching_policy_sha256: str
    matching_set_sha256: str
    training_schedule_sha256: str
    recognition_estimator_sha256: str
    runtime_identity_sha256: str
    counter_mapping_sha256: str
    phase_ownership_sha256: str
    deterministic_policy_sha256: str
    checkpoint_codec_sha256: str
    objective_manifest_schema_sha256: str
    scoring_inventory_sha256: str
    data_identity_sha256: str
    access_policy_sha256: str
    readiness_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name) for name in tuple(self.__dataclass_fields__)[:-1]
        }

    def __post_init__(self) -> None:
        if (
            self.readiness_schema != "h6-prediction-readiness-v3"
            or self.status != "PASS"
        ):
            raise ValueError("v3 readiness must be an exact PASS token")
        _require_git_head(self.git_head)
        if type(self.correctness_manifests) is not tuple or tuple(
            gate for gate, _ in self.correctness_manifests
        ) != ("H1", "H2", "H3", "H5"):
            raise ValueError("v3 readiness correctness manifests must be H1,H2,H3,H5")
        for _, digest in self.correctness_manifests:
            _require_sha256(digest, "correctness manifest")
        for name in (
            "dirty_digest",
            "experiment_config_sha256",
            "h1_prefix_prior_manifest_sha256",
            "h1_prefix_prior_generative_factor_schema_sha256",
            "smc_bias_semantics_sha256",
            "smc_validation_manifest_sha256",
            "prefix_certificate_set_sha256",
            "a0_direct_exact_prefix_certificate_sha256",
            "h5_update_binding_sha256",
            "critical_values_sha256",
            "endpoint_smc_protocol_sha256",
            "attribution_matrix_sha256",
            "objective_gate_spec_sha256",
            "matching_policy_sha256",
            "matching_set_sha256",
            "training_schedule_sha256",
            "recognition_estimator_sha256",
            "runtime_identity_sha256",
            "counter_mapping_sha256",
            "phase_ownership_sha256",
            "deterministic_policy_sha256",
            "checkpoint_codec_sha256",
            "objective_manifest_schema_sha256",
            "scoring_inventory_sha256",
            "data_identity_sha256",
            "access_policy_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if (
            self.counter_mapping_sha256 != H6_COUNTER_MAPPING_SHA256
            or self.phase_ownership_sha256 != H6_PHASE_OWNERSHIP_SHA256
            or self.deterministic_policy_sha256 != H6_DETERMINISTIC_POLICY_SHA256
            or self.checkpoint_codec_sha256 != H6_CHECKPOINT_CODEC_SHA256
            or self.objective_manifest_schema_sha256
            != H6_OBJECTIVE_MANIFEST_SCHEMA_SHA256
            or self.scoring_inventory_sha256 != H6_SCORING_INVENTORY_SHA256
        ):
            raise ValueError("v3 readiness authority is stale")
        if self.readiness_sha256 != _owned_hash(
            "vfe4.h6.prediction-readiness.v3", self.canonical_payload()
        ):
            raise ValueError("v3 readiness identity is stale")

    @classmethod
    def create(
        cls,
        *,
        git_head: str,
        dirty_digest: str,
        experiment_config_sha256: str,
        correctness_manifests: tuple[tuple[str, str], ...],
        h1_prefix_prior_manifest_sha256: str,
        h1_prefix_prior_generative_factor_schema_sha256: str,
        smc_bias_semantics_sha256: str,
        smc_validation_manifest_sha256: str,
        prefix_certificate_set_sha256: str,
        a0_direct_exact_prefix_certificate_sha256: str,
        h5_update_binding_sha256: str,
        critical_values_sha256: str,
        endpoint_smc_protocol_sha256: str,
        attribution_matrix_sha256: str,
        objective_gate_spec_sha256: str,
        matching_policy_sha256: str,
        matching_set_sha256: str,
        training_schedule_sha256: str,
        recognition_estimator_sha256: str,
        runtime_identity_sha256: str,
        counter_mapping_sha256: str,
        phase_ownership_sha256: str,
        objective_manifest_schema_sha256: str,
        data_identity_sha256: str,
        access_policy_sha256: str,
    ) -> "H6PredictionV3ReadinessToken":
        values = {
            "readiness_schema": "h6-prediction-readiness-v3",
            "status": "PASS",
            "git_head": git_head,
            "dirty_digest": dirty_digest,
            "experiment_config_sha256": experiment_config_sha256,
            "correctness_manifests": tuple(correctness_manifests),
            "h1_prefix_prior_manifest_sha256": (h1_prefix_prior_manifest_sha256),
            "h1_prefix_prior_generative_factor_schema_sha256": (
                h1_prefix_prior_generative_factor_schema_sha256
            ),
            "smc_bias_semantics_sha256": smc_bias_semantics_sha256,
            "smc_validation_manifest_sha256": (smc_validation_manifest_sha256),
            "prefix_certificate_set_sha256": (prefix_certificate_set_sha256),
            "a0_direct_exact_prefix_certificate_sha256": (
                a0_direct_exact_prefix_certificate_sha256
            ),
            "h5_update_binding_sha256": h5_update_binding_sha256,
            "critical_values_sha256": critical_values_sha256,
            "endpoint_smc_protocol_sha256": endpoint_smc_protocol_sha256,
            "attribution_matrix_sha256": attribution_matrix_sha256,
            "objective_gate_spec_sha256": objective_gate_spec_sha256,
            "matching_policy_sha256": matching_policy_sha256,
            "matching_set_sha256": matching_set_sha256,
            "training_schedule_sha256": training_schedule_sha256,
            "recognition_estimator_sha256": (recognition_estimator_sha256),
            "runtime_identity_sha256": runtime_identity_sha256,
            "counter_mapping_sha256": counter_mapping_sha256,
            "phase_ownership_sha256": phase_ownership_sha256,
            "deterministic_policy_sha256": (H6_DETERMINISTIC_POLICY_SHA256),
            "checkpoint_codec_sha256": H6_CHECKPOINT_CODEC_SHA256,
            "objective_manifest_schema_sha256": (objective_manifest_schema_sha256),
            "scoring_inventory_sha256": H6_SCORING_INVENTORY_SHA256,
            "data_identity_sha256": data_identity_sha256,
            "access_policy_sha256": access_policy_sha256,
        }
        return cls(
            **values,  # type: ignore[arg-type]
            readiness_sha256=_owned_hash(
                "vfe4.h6.prediction-readiness.v3",
                values,
            ),
        )


@dataclass(frozen=True, slots=True)
class H6AttemptSpecV3:
    attempt_schema: Literal["h6-attempt-spec-v3"]
    git_head: str
    dirty_digest: str
    readiness_sha256: str
    experiment_config_sha256: str
    endpoint_id: str
    arm_id: str
    endpoint_config_sha256: str
    objective_kind: Literal[
        "cross_entropy",
        "complete_elbo",
        "emission_only_ablation_non_elbo",
    ]
    model_factory_sha256: str
    recognition_factory_sha256: str | None
    initialization_sha256: str
    optimizer_policy_sha256: str
    training_seed: int
    data_identity_sha256: str
    window_schedule_sha256: str
    batch_schedule_sha256: str
    phase_schedule_sha256: str
    training_schedule_sha256: str
    recognition_estimator_sha256: str
    runtime_identity_sha256: str
    counter_mapping_sha256: str
    checkpoint_codec_sha256: str
    attempt_spec_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name) for name in tuple(self.__dataclass_fields__)[:-1]
        }

    def __post_init__(self) -> None:
        if self.attempt_schema != "h6-attempt-spec-v3":
            raise ValueError("unsupported v3 attempt schema")
        _require_git_head(self.git_head)
        for name in (
            "dirty_digest",
            "readiness_sha256",
            "experiment_config_sha256",
            "endpoint_config_sha256",
            "model_factory_sha256",
            "initialization_sha256",
            "optimizer_policy_sha256",
            "data_identity_sha256",
            "window_schedule_sha256",
            "batch_schedule_sha256",
            "phase_schedule_sha256",
            "training_schedule_sha256",
            "recognition_estimator_sha256",
            "runtime_identity_sha256",
            "counter_mapping_sha256",
            "checkpoint_codec_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        _require_nonempty(self.endpoint_id, "endpoint_id")
        _require_nonempty(self.arm_id, "arm_id")
        if self.objective_kind not in (
            "cross_entropy",
            "complete_elbo",
            "emission_only_ablation_non_elbo",
        ):
            raise ValueError("unsupported v3 attempt objective")
        if self.recognition_factory_sha256 is not None:
            _require_sha256(
                self.recognition_factory_sha256,
                "recognition_factory_sha256",
            )
        if type(self.training_seed) is not int or self.training_seed < 0:
            raise ValueError("training_seed must be a nonnegative integer")
        if (
            self.counter_mapping_sha256 != H6_COUNTER_MAPPING_SHA256
            or self.checkpoint_codec_sha256 != H6_CHECKPOINT_CODEC_SHA256
        ):
            raise ValueError("v3 attempt authority is stale")
        if self.attempt_spec_sha256 != _owned_hash(
            "vfe4.h6.attempt-spec.v3", self.canonical_payload()
        ):
            raise ValueError("v3 attempt identity is stale")

    @classmethod
    def create(
        cls,
        *,
        git_head: str,
        dirty_digest: str,
        readiness_sha256: str,
        experiment_config_sha256: str,
        endpoint_id: str,
        arm_id: str,
        endpoint_config_sha256: str,
        objective_kind: Literal[
            "cross_entropy",
            "complete_elbo",
            "emission_only_ablation_non_elbo",
        ],
        model_factory_sha256: str,
        recognition_factory_sha256: str | None,
        initialization_sha256: str,
        optimizer_policy_sha256: str,
        training_seed: int,
        data_identity_sha256: str,
        window_schedule_sha256: str,
        batch_schedule_sha256: str,
        phase_schedule_sha256: str,
        training_schedule_sha256: str,
        recognition_estimator_sha256: str,
        runtime_identity_sha256: str,
    ) -> "H6AttemptSpecV3":
        values = {
            "attempt_schema": "h6-attempt-spec-v3",
            "git_head": git_head,
            "dirty_digest": dirty_digest,
            "readiness_sha256": readiness_sha256,
            "experiment_config_sha256": experiment_config_sha256,
            "endpoint_id": endpoint_id,
            "arm_id": arm_id,
            "endpoint_config_sha256": endpoint_config_sha256,
            "objective_kind": objective_kind,
            "model_factory_sha256": model_factory_sha256,
            "recognition_factory_sha256": recognition_factory_sha256,
            "initialization_sha256": initialization_sha256,
            "optimizer_policy_sha256": optimizer_policy_sha256,
            "training_seed": training_seed,
            "data_identity_sha256": data_identity_sha256,
            "window_schedule_sha256": window_schedule_sha256,
            "batch_schedule_sha256": batch_schedule_sha256,
            "phase_schedule_sha256": phase_schedule_sha256,
            "training_schedule_sha256": training_schedule_sha256,
            "recognition_estimator_sha256": (recognition_estimator_sha256),
            "runtime_identity_sha256": runtime_identity_sha256,
            "counter_mapping_sha256": H6_COUNTER_MAPPING_SHA256,
            "checkpoint_codec_sha256": H6_CHECKPOINT_CODEC_SHA256,
        }
        return cls(
            **values,  # type: ignore[arg-type]
            attempt_spec_sha256=_owned_hash(
                "vfe4.h6.attempt-spec.v3",
                values,
            ),
        )


@dataclass(frozen=True, slots=True)
class H6AttemptCursorV3:
    cursor_schema: Literal["h6-attempt-cursor-v3"]
    attempt_spec_sha256: str
    pass_index: int
    batch_index: int
    next_phase: TrainingPhase
    example_ordinal: int
    sample_ordinal: Literal[0]
    draw_block: int
    counter_consumption_sha256: str
    permutation_sha256: str
    recognition_update_count: int
    model_update_count: int
    validation_boundary_count: int
    checkpoint_boundary_count: int
    cursor_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "cursor_schema": self.cursor_schema,
            "attempt_spec_sha256": self.attempt_spec_sha256,
            "pass_index": self.pass_index,
            "batch_index": self.batch_index,
            "next_phase": self.next_phase.value,
            "example_ordinal": self.example_ordinal,
            "sample_ordinal": self.sample_ordinal,
            "draw_block": self.draw_block,
            "counter_consumption_sha256": self.counter_consumption_sha256,
            "permutation_sha256": self.permutation_sha256,
            "recognition_update_count": self.recognition_update_count,
            "model_update_count": self.model_update_count,
            "validation_boundary_count": self.validation_boundary_count,
            "checkpoint_boundary_count": self.checkpoint_boundary_count,
        }

    def __post_init__(self) -> None:
        if self.cursor_schema != "h6-attempt-cursor-v3":
            raise ValueError("unsupported v3 cursor schema")
        _require_sha256(self.attempt_spec_sha256, "attempt_spec_sha256")
        _require_sha256(self.counter_consumption_sha256, "counter_consumption_sha256")
        _require_sha256(self.permutation_sha256, "permutation_sha256")
        if type(self.next_phase) is not TrainingPhase:
            raise ValueError("next_phase must be an exact TrainingPhase")
        for name in (
            "pass_index",
            "batch_index",
            "example_ordinal",
            "sample_ordinal",
            "draw_block",
            "recognition_update_count",
            "model_update_count",
            "validation_boundary_count",
            "checkpoint_boundary_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.sample_ordinal != 0:
            raise ValueError("v3 uses exactly sample_ordinal zero")
        if self.cursor_sha256 != _owned_hash(
            "vfe4.h6.attempt-cursor.v3", self.canonical_payload()
        ):
            raise ValueError("v3 cursor identity is stale")

    @classmethod
    def create(
        cls,
        *,
        attempt_spec_sha256: str,
        pass_index: int,
        batch_index: int,
        next_phase: TrainingPhase,
        example_ordinal: int,
        draw_block: int,
        counter_consumption_sha256: str,
        permutation_sha256: str,
        recognition_update_count: int = 0,
        model_update_count: int = 0,
        validation_boundary_count: int = 0,
        checkpoint_boundary_count: int = 0,
    ) -> "H6AttemptCursorV3":
        values = {
            "cursor_schema": "h6-attempt-cursor-v3",
            "attempt_spec_sha256": attempt_spec_sha256,
            "pass_index": pass_index,
            "batch_index": batch_index,
            "next_phase": next_phase,
            "example_ordinal": example_ordinal,
            "sample_ordinal": 0,
            "draw_block": draw_block,
            "counter_consumption_sha256": counter_consumption_sha256,
            "permutation_sha256": permutation_sha256,
            "recognition_update_count": recognition_update_count,
            "model_update_count": model_update_count,
            "validation_boundary_count": validation_boundary_count,
            "checkpoint_boundary_count": checkpoint_boundary_count,
        }
        payload = {
            **values,
            "next_phase": next_phase.value,
        }
        return cls(
            **values,  # type: ignore[arg-type]
            cursor_sha256=_owned_hash(
                "vfe4.h6.attempt-cursor.v3",
                payload,
            ),
        )


@dataclass(frozen=True, slots=True)
class H6ObjectiveManifestV3:
    objective_schema: Literal["h6-objective-manifest-v3"]
    attempt_spec_sha256: str
    endpoint_config_sha256: str
    objective_kind: Literal[
        "cross_entropy",
        "complete_elbo",
        "emission_only_ablation_non_elbo",
    ]
    is_elbo: bool
    phase: TrainingPhase
    recognition_estimator_sha256: str
    counter_consumption_sha256: str
    recognition_law_sha256: str | None
    detached_snapshot_sha256: str | None
    ordered_factor_bindings: tuple[tuple[str, int, str], ...]
    total_raw_bytes_sha256: str
    objective_manifest_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "objective_schema": self.objective_schema,
            "attempt_spec_sha256": self.attempt_spec_sha256,
            "endpoint_config_sha256": self.endpoint_config_sha256,
            "objective_kind": self.objective_kind,
            "is_elbo": self.is_elbo,
            "phase": self.phase.value,
            "recognition_estimator_sha256": self.recognition_estimator_sha256,
            "counter_consumption_sha256": self.counter_consumption_sha256,
            "recognition_law_sha256": self.recognition_law_sha256,
            "detached_snapshot_sha256": self.detached_snapshot_sha256,
            "ordered_factor_bindings": self.ordered_factor_bindings,
            "total_raw_bytes_sha256": self.total_raw_bytes_sha256,
        }

    def __post_init__(self) -> None:
        if self.objective_schema != "h6-objective-manifest-v3":
            raise ValueError("unsupported v3 objective schema")
        if type(self.is_elbo) is not bool or self.is_elbo != (
            self.objective_kind == "complete_elbo"
        ):
            raise ValueError("objective kind and is_elbo disagree")
        if type(self.phase) is not TrainingPhase:
            raise ValueError("objective phase must be exact")
        for name in (
            "attempt_spec_sha256",
            "endpoint_config_sha256",
            "recognition_estimator_sha256",
            "counter_consumption_sha256",
            "total_raw_bytes_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        for name in ("recognition_law_sha256", "detached_snapshot_sha256"):
            value = getattr(self, name)
            if value is not None:
                _require_sha256(value, name)
        if (
            type(self.ordered_factor_bindings) is not tuple
            or not self.ordered_factor_bindings
        ):
            raise ValueError("objective factor bindings must be nonempty")
        for partition, receiver_t, digest in self.ordered_factor_bindings:
            _require_nonempty(partition, "partition")
            if type(receiver_t) is not int or receiver_t < 0:
                raise ValueError("factor receiver must be nonnegative")
            _require_sha256(digest, "factor digest")
        if self.objective_manifest_sha256 != _owned_hash(
            "vfe4.h6.objective-manifest.v3", self.canonical_payload()
        ):
            raise ValueError("v3 objective identity is stale")

    @classmethod
    def create(
        cls,
        *,
        attempt_spec_sha256: str,
        endpoint_config_sha256: str,
        objective_kind: Literal[
            "cross_entropy",
            "complete_elbo",
            "emission_only_ablation_non_elbo",
        ],
        phase: TrainingPhase,
        recognition_estimator_sha256: str,
        counter_consumption_sha256: str,
        recognition_law_sha256: str | None,
        detached_snapshot_sha256: str | None,
        ordered_factor_bindings: tuple[tuple[str, int, str], ...],
        total_raw_bytes_sha256: str,
    ) -> "H6ObjectiveManifestV3":
        values = {
            "objective_schema": "h6-objective-manifest-v3",
            "attempt_spec_sha256": attempt_spec_sha256,
            "endpoint_config_sha256": endpoint_config_sha256,
            "objective_kind": objective_kind,
            "is_elbo": objective_kind == "complete_elbo",
            "phase": phase,
            "recognition_estimator_sha256": (recognition_estimator_sha256),
            "counter_consumption_sha256": (counter_consumption_sha256),
            "recognition_law_sha256": recognition_law_sha256,
            "detached_snapshot_sha256": detached_snapshot_sha256,
            "ordered_factor_bindings": tuple(ordered_factor_bindings),
            "total_raw_bytes_sha256": total_raw_bytes_sha256,
        }
        payload = {
            **values,
            "phase": phase.value,
        }
        return cls(
            **values,  # type: ignore[arg-type]
            objective_manifest_sha256=_owned_hash(
                "vfe4.h6.objective-manifest.v3",
                payload,
            ),
        )


__all__ = [
    "H6_ATTEMPT_CURSOR_SCHEMA",
    "H6_ATTEMPT_SPEC_SCHEMA",
    "H6AttemptCursorV3",
    "H6AttemptSpecV3",
    "H6_CHECKPOINT_SCHEMA",
    "H6_CHECKPOINT_CODEC_SHA256",
    "H6_COUNTER_MAPPING_SHA256",
    "H6_DETERMINISTIC_POLICY_DESCRIPTOR",
    "H6_DETERMINISTIC_POLICY_SHA256",
    "H6_NO_COUNTER_CONSUMPTION_SHA256",
    "H6ObjectiveManifestV3",
    "H6_OBJECTIVE_MANIFEST_SCHEMA",
    "H6_OBJECTIVE_MANIFEST_SCHEMA_SHA256",
    "H6_PHASE_OWNERSHIP_SHA256",
    "H6_PREDICTION_CONFIG_SCHEMA",
    "H6_PREDICTION_METRICS_SCHEMA",
    "H6_PREDICTION_READINESS_SCHEMA",
    "H6_PREDICTION_RESULT_SCHEMA",
    "H6PredictionRuntimeIdentity",
    "H6PredictionV3ReadinessToken",
    "H6_RAW_ENDPOINT_INVENTORY_SCHEMA",
    "H6RecognitionEstimatorSpec",
    "H6_SCORING_INVENTORY_SHA256",
    "H6_TRAINING_SCHEDULE_SCHEMA",
    "H6TrainingScheduleV3",
]
