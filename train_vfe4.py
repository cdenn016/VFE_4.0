"""Click-to-run WikiText-103 VFE4 training surface.

Edit ``CONFIG`` and click Run. Import is pure: it performs no data access,
device initialization, run reservation, checkpoint I/O, or training.
"""

from __future__ import annotations

import hmac
import os
import stat
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from vfe4.config.schema import TrainingConfig


OPERATIONS = (
    "idle",
    "synthetic_smoke",
    "source_lock",
    "readiness",
    "train",
    "resume",
)
SOURCE_LOCK_AUTHORIZATION = "AUTHORIZE_VFE4_WT103_SOURCE_LOCK_V1"
PRODUCTION_AUTHORIZATION = "AUTHORIZE_VFE4_WT103_PRODUCTION_TRAINING_V1"
_REPO_ROOT = Path(__file__).resolve().parent


def _default_training_mapping() -> dict[str, object]:
    """Return the complete editable profile without importing ``vfe4``.

    Keeping this literal at the launcher boundary is intentional. Importing
    the broad package configuration surface transitively imports Torch, while
    merely opening this click-to-run file must remain free of runtime/model
    imports. Resolution still happens exactly once inside :func:`main`.
    """

    confirmatory_seeds = tuple(range(2026072101, 2026072109))
    validation_streams = tuple(range(8))
    test_streams = tuple(range(64))
    tuning_grid = (1.0e-4, 3.0e-4, 1.0e-3)
    particle_counts = (128, 256, 512, 1024)

    def arm(
        *,
        arm_id: str,
        factory_id: str,
        training_objective: str,
        prior_variant: str,
        source_mixture: str,
        latent_enabled: bool,
        recognition_enabled: bool,
        recognition_family: str,
        recognition_iterations_per_batch: int,
        update_phases: tuple[str, ...],
        scorer_kind: str,
        result_role: str,
        nonclaims: tuple[str, ...],
    ) -> dict[str, object]:
        return {
            "schema_version": "wt103-arm-spec-v1",
            "arm_id": arm_id,
            "factory_id": factory_id,
            "training_objective": training_objective,
            "prior_variant": prior_variant,
            "source_mixture": source_mixture,
            "latent_enabled": latent_enabled,
            "recognition_enabled": recognition_enabled,
            "recognition_family": recognition_family,
            "recognition_iterations_per_batch": (
                recognition_iterations_per_batch
            ),
            "update_phases": update_phases,
            "scorer_kind": scorer_kind,
            "tuning_grid_id": "wt103-six-cell-v1",
            "confirmatory_seed_ids": confirmatory_seeds,
            "terminal_checkpoint_role": "terminal_scoring",
            "result_role": result_role,
            "nonclaims": nonclaims,
        }

    arms = [
        arm(
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
        ),
        arm(
            arm_id="WT103-A5-PARENT-SPECIFIC-PREFIX-COMPLETE-v1",
            factory_id=(
                "build_wt103_a5_parent_specific@wt103-arm-v1"
            ),
            training_objective="complete_elbo",
            prior_variant="parent_specific_pooled_prefix",
            source_mixture="exact",
            latent_enabled=True,
            recognition_enabled=True,
            recognition_family=(
                "structured_block_tridiagonal_smoothing"
            ),
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
        ),
        arm(
            arm_id="WT103-A5-FIXED-COMPLETE-v1",
            factory_id="build_wt103_a5_fixed@wt103-arm-v1",
            training_objective="complete_elbo",
            prior_variant="fixed",
            source_mixture="exact",
            latent_enabled=True,
            recognition_enabled=True,
            recognition_family=(
                "structured_block_tridiagonal_smoothing"
            ),
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
        ),
        arm(
            arm_id="WT103-A5-PARENT-SPECIFIC-PREFIX-EMISSION-v1",
            factory_id=(
                "build_wt103_a5_parent_specific@wt103-arm-v1"
            ),
            training_objective="emission_only_ablation_non_elbo",
            prior_variant="parent_specific_pooled_prefix",
            source_mixture="exact",
            latent_enabled=True,
            recognition_enabled=True,
            recognition_family=(
                "structured_block_tridiagonal_smoothing"
            ),
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
        ),
        arm(
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
        ),
    ]

    def gate(
        gate_id: str,
        ordinal: int,
        prerequisites: tuple[str, ...],
        result_arm_ids: tuple[str, ...],
        disposition: str,
    ) -> dict[str, object]:
        return {
            "schema_version": "wt103-gate-spec-v1",
            "gate_id": gate_id,
            "ordinal": ordinal,
            "prerequisite_gate_ids": prerequisites,
            "result_arm_ids": result_arm_ids,
            "disposition_rule_id": disposition,
        }

    gates = [
        gate("SOURCE_LOCK", 0, (), (), "finalized_source_or_stop"),
        gate(
            "H8_EXACT_REVISION",
            1,
            ("SOURCE_LOCK",),
            (),
            "same_revision_h8_v5_pass_or_stop",
        ),
        gate(
            "POST_H8_READINESS",
            2,
            ("H8_EXACT_REVISION",),
            (),
            "all_static_and_resource_inputs_pass_or_stop",
        ),
        gate(
            "OBJECTIVE",
            3,
            ("POST_H8_READINESS",),
            ("WT103-A5-PARENT-SPECIFIC-PREFIX-EMISSION-v1",),
            "objective_gate_before_primary",
        ),
        gate(
            "PRIMARY",
            4,
            ("OBJECTIVE",),
            (
                "WT103-A0-AR-v1",
                "WT103-A5-PARENT-SPECIFIC-PREFIX-COMPLETE-v1",
            ),
            "paired_estimator_inflated_primary_decision",
        ),
        gate(
            "PRIOR_CONTROL",
            5,
            ("POST_H8_READINESS",),
            ("WT103-A5-FIXED-COMPLETE-v1",),
            "retain_control_without_primary_promotion",
        ),
        gate(
            "LATENT_PATH_CONTROL",
            6,
            ("POST_H8_READINESS",),
            ("WT103-A5-NOLATENT-v1",),
            "retain_control_without_primary_promotion",
        ),
    ]

    profile = {
        "schema_version": "wt103-experiment-profile-v1",
        "dataset_schema": "wikitext-103-raw-v1",
        "tokenizer_schema": "gpt2-tiktoken-v1",
        "vocabulary_size": 50_257,
        "sequence_length": 128,
        "stride": 128,
        "batch_size": 128,
        "gradient_accumulation_steps": 1,
        "num_workers": 0,
        "pin_memory": True,
        "drop_last": False,
        "model_depth": 1,
        "d_z": 20,
        "d_m": 20,
        "K": 20,
        "combined_latent_block": 40,
        "source_lookback": 20,
        "state_parent_rule": "range(max(0,t-20),t)",
        "model_parent_rule": "range(max(0,t-20),t)",
        "population_frame_profile": "h7-direct-glplus-v1",
        "decoder_profile": "categorical_linear_chunked",
        "decoder_train_token_chunk": 512,
        "decoder_eval_token_chunk": 256,
        "smc_particle_chunk": 32,
        "dropout_probability": 0.0,
        "input_output_embedding_tied": False,
        "optimizer": {
            "optimizer": "AdamW",
            "betas": (0.9, 0.999),
            "epsilon": 1.0e-8,
            "amsgrad": False,
            "foreach": False,
            "fused": False,
            "gradient_clip": "per_active_block_global_l2",
            "gradient_clip_max_norm": 1.0,
            "proposal_acceptance": (
                "validity_only_no_monotonicity_claim"
            ),
            "reject_on": (
                "nonfinite_objective",
                "nonfinite_gradient",
                "amp_overflow",
                "invalid_support",
                "non_spd",
                "scope_mismatch",
                "snapshot_alias",
                "optimizer_access_mismatch",
            ),
        },
        "scheduler": {
            "scheduler": "linear_warmup_then_cosine",
            "warmup_optimizer_steps": 100,
            "minimum_lr_ratio": 0.1,
            "restart_count": 0,
            "horizon": "planned_active_optimizer_steps_for_attempt",
        },
        "precision": {
            "real_training_device": "cuda:0",
            "parameter_dtype": "float32",
            "optimizer_state_dtype": "float32",
            "autocast_enabled": True,
            "autocast_dtype": "bfloat16",
            "grad_scaler_enabled": False,
            "grad_scaler_fixed_scale": 1.0,
            "spd_factor_solve_logdet_dtype": "float32",
            "smc_log_weight_dtype": "float64",
            "metric_corpus_accumulator": "python_math_fsum_float64",
            "torch_deterministic_algorithms": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "allow_tf32_matmul": False,
            "allow_tf32_cudnn": False,
            "allow_fp16_reduced_precision_reduce": False,
            "cublas_workspace_config": ":4096:8",
        },
        "cadence": {
            "validation_boundaries_per_pass": 20,
            "checkpoint_at_every_validation": True,
            "confirmatory_passes": 2,
            "early_stopping": False,
            "validation_boundary_rule": (
                "stable_unique_ceil_k_batches_per_pass_over_20"
            ),
        },
        "checkpoints": {
            "rolling_checkpoints_retained": 2,
            "rolling_role": "resume_only",
            "terminal_checkpoint_retained": True,
            "terminal_role": "terminal_scoring",
            "best_checkpoint_selection": False,
        },
        "statistics": {
            "learning_rate_grid": tuning_grid,
            "weight_decay_grid": (0.0, 1.0e-2),
            "tuning_seed_ids": (2026072199, 2026072200),
            "confirmatory_seed_ids": confirmatory_seeds,
            "data_order_seed": 2026072199,
            "validation_stream_ids": validation_streams,
            "test_stream_ids": test_streams,
            "validation_particle_count": 256,
            "particle_counts": particle_counts,
            "simultaneous_constant": 4.5144904535377144,
            "practical_threshold": 0.01005033585350145,
            "contraction_ratio": 0.75,
            "one_opening_policy": (
                "durable_exclusive_single_test_transaction"
            ),
        },
        "resources": {
            "maximum_gpu_hours": 720.0,
            "maximum_wall_hours": 840.0,
            "maximum_energy_kwh": 500.0,
            "forecast_headroom_factor": 1.25,
            "maximum_device_fraction": 0.85,
            "power_sample_interval_ms": 100,
        },
        "schemas": {
            "h6_prediction_schema": "h6-prediction-result-v3",
            "h8_schema": "h8-sparse-scale-v5",
            "h8_config_schema": "h8-validation-config-v3",
            "h8_parent_child_protocol": (
                "vfe4.h8.parent-child-protocol.v3"
            ),
            "training_sparsity_schema": "wt103-training-sparsity-v1",
            "metric_schema": "wt103-metric-record-v1",
            "figure_schema": "wt103-figure-spec-v1",
            "checkpoint_schema": "wt103-checkpoint-v1",
        },
        "nonclaims": {
            "backprop_free": False,
            "h8_training_memory_transfer": False,
            "h6_byte_vocabulary_transfer": False,
            "v3_checkpoint_or_config_reuse": False,
            "h8_asymptotic_scaling_law": False,
            "monotone_elbo_or_coordinate_ascent": False,
            "component_attribution_from_primary": False,
        },
    }
    a0 = {
        "schema_version": "wt103-a0-architecture-v1",
        "block_count": 1,
        "hidden_width": 20,
        "attention_heads": 2,
        "head_width": 10,
        "attention_context": "full_causal_inclusive_self",
        "attention_allowed_keys": "range(0,q+1)",
        "attention_semantic_pair_count": "L*(L+1)//2",
        "attention_implementation": (
            "torch.nn.functional.scaled_dot_product_attention"
        ),
        "attention_backend_policy": "flash_attention_only_no_fallback",
        "pytorch_sdpa_api_binding": (
            "torch.nn.attention.sdpa_kernel("
            "backends=[SDPBackend.FLASH_ATTENTION])"
        ),
        "enabled_backends": ("FLASH_ATTENTION",),
        "alternative_backends_disabled": True,
        "source_lock_scope": "candidate_unverified",
        "pytorch_version": "unresolved_until_task13_source_lock",
        "sdpa_api_sha256": "unresolved_until_task13_source_lock",
        "flash_backend_sha256": "unresolved_until_task13_source_lock",
        "attention_is_causal": True,
        "attention_mask_argument": None,
        "attention_scale": "1/sqrt(head_width)",
        "attention_dropout_probability": 0.0,
        "attention_returns_weights": False,
        "grouped_query_attention": False,
        "backend_fallback_allowed": False,
        "fused_full_attention_allowed": True,
        "fused_attention_materialization": "forbidden",
        "token_embedding": "learned[V,h]_no_bias",
        "positional_encoding": "learned_absolute",
        "positional_capacity": 128,
        "position_interpolation": False,
        "input_composition": (
            "token_embedding_plus_position_embedding"
        ),
        "normalization": (
            "LayerNorm(eps=1e-5,elementwise_affine=true,bias=true)"
        ),
        "normalization_placement": "pre_norm_with_final_norm",
        "residual_topology": (
            "x=x+attn(ln1(x));x=x+mlp(ln2(x));y=ln_f(x)"
        ),
        "qkv_projection": (
            "Linear(in=h,out=3h,weight[3h,h],bias[3h])"
        ),
        "attention_output_projection": (
            "Linear(in=h,out=h,weight[h,h],bias[h])"
        ),
        "mlp_input_projection": (
            "Linear(in=h,out=4h,weight[4h,h],bias[4h])"
        ),
        "activation": "gelu_tanh_approximation",
        "mlp_output_projection": (
            "Linear(in=4h,out=h,weight[h,4h],bias[h])"
        ),
        "decoder_projection": (
            "untied_Linear(in=h,out=V,weight[V,h],bias[V])"
        ),
        "all_dropout_probabilities": 0.0,
        "candidate_hidden_widths": (
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
        ),
        "parameter_relative_tolerance": 0.01,
        "flop_relative_tolerance": 0.05,
        "candidate_selection_key": (
            "abs_log_parameter_ratio_abs_log_flop_ratio_hidden_width"
        ),
        "parameter_formula_schema": "wt103-a0-parameter-formula-v1",
        "flop_formula_schema": "wt103-a0-semantic-train-flops-v1",
    }
    return {
        "schema_version": "wt103-training-config-v1",
        "operation": "idle",
        "synthetic_authority": "nonproduction_synthetic_smoke",
        "candidate_tokenizer": {
            "distribution": "tiktoken",
            "version": "0.12.0",
            "encoding_name": "gpt2",
        },
        "profile": profile,
        "a0_architecture": a0,
        "estimator_protocol": {
            "schema_version": "wt103-estimator-protocol-v1",
            "validation_particle_count": 256,
            "validation_stream_ids": validation_streams,
            "test_stream_ids": test_streams,
            "particle_counts": particle_counts,
            "validation_stream_domain": (
                "post-h8-wt103-validation-v1|"
                "2026072198|stream_id|purpose"
            ),
            "test_stream_domain": (
                "post-h8-wt103-test-v1|"
                "2026072198|stream_id|purpose"
            ),
        },
        "arms": arms,
        "gates": gates,
        "scientific_preconditions": {
            "schema_version": "wt103-scientific-preconditions-v1",
            "h6_prediction_authority": "native_executable_v3",
            "h6_prediction_schema": "h6-prediction-result-v3",
            "h8_schema": "h8-sparse-scale-v5",
            "h8_config_schema": "h8-validation-config-v3",
            "h8_parent_child_protocol": (
                "vfe4.h8.parent-child-protocol.v3"
            ),
            "training_sparsity_schema": "wt103-training-sparsity-v1",
            "h8_reference_required": True,
            "training_sparsity_reference_required": True,
            "target_blind_predictor_safety_required": True,
            "h8_can_satisfy_training_sparsity": False,
            "capacity_can_satisfy_training_sparsity": False,
        },
    }


def _config_template() -> dict[str, object]:
    return {
        "launcher_schema": "wt103-click-launcher-v2",
        "training": _default_training_mapping(),
        "paths": {
            "cache_root": str(
                Path.home() / ".cache" / "vfe4" / "wikitext103"
            ),
            "run_root": str(_REPO_ROOT / "artifacts" / "wt103-runs"),
            "source_record_path": str(
                _REPO_ROOT
                / "docs"
                / "data"
                / "wikitext103-raw-v1-source-record.json"
            ),
            "resume_experiment_plan_path": str(
                _REPO_ROOT
                / "artifacts"
                / "wt103-runs"
                / "unresolved-experiment"
                / "experiment-plan.json"
            ),
            "smoke_run_id": "wt103-nonproduction-synthetic-smoke",
        },
        "authorization": None,
    }


# Edit this dictionary, then click Run.
CONFIG: dict[str, object] = _config_template()


class TrainingLaunchError(RuntimeError):
    """The click launcher failed before or during an authorized operation."""


@dataclass(frozen=True, slots=True)
class TrainingLauncherResult:
    launcher_schema: Literal["wt103-click-launcher-result-v1"]
    operation: Literal[
        "idle",
        "synthetic_smoke",
        "source_lock",
        "readiness",
        "train",
        "resume",
    ]
    status: Literal["IDLE", "COMPLETED"]
    payload: object | None = None

    def __post_init__(self) -> None:
        if (
            self.launcher_schema != "wt103-click-launcher-result-v1"
            or self.operation not in OPERATIONS
            or self.status not in ("IDLE", "COMPLETED")
            or (self.status == "IDLE") != (self.operation == "idle")
            or (self.status == "IDLE") != (self.payload is None)
        ):
            raise ValueError("training launcher result is inconsistent")


@dataclass(frozen=True, slots=True)
class _LauncherPaths:
    cache_root: Path
    run_root: Path
    source_record_path: Path
    resume_experiment_plan_path: Path
    smoke_run_id: str


@runtime_checkable
class TrainingOperationDriver(Protocol):
    """Concrete production operation seam used by the click surface."""

    def source_lock(
        self,
        *,
        training: "TrainingConfig",
        paths: _LauncherPaths,
    ) -> object: ...

    def reopen_source_lock(
        self,
        *,
        training: "TrainingConfig",
        paths: _LauncherPaths,
    ) -> object: ...

    def readiness(
        self,
        *,
        training: "TrainingConfig",
        paths: _LauncherPaths,
        source_lock: object,
    ) -> object: ...

    def train(
        self,
        *,
        training: "TrainingConfig",
        paths: _LauncherPaths,
        source_lock: object,
        readiness: object,
    ) -> object: ...

    def resume(
        self,
        *,
        training: "TrainingConfig",
        paths: _LauncherPaths,
        source_lock: object,
        readiness: object,
    ) -> object: ...


class _DefaultTrainingOperationDriver:
    """Lazy bridge to concrete source/readiness/training implementations."""

    def source_lock(
        self,
        *,
        training: "TrainingConfig",
        paths: _LauncherPaths,
    ) -> object:
        from vfe4.training.production import run_source_lock

        return run_source_lock(training=training, paths=paths)

    def reopen_source_lock(
        self,
        *,
        training: "TrainingConfig",
        paths: _LauncherPaths,
    ) -> object:
        from vfe4.training.production import reopen_source_lock

        return reopen_source_lock(training=training, paths=paths)

    def readiness(
        self,
        *,
        training: "TrainingConfig",
        paths: _LauncherPaths,
        source_lock: object,
    ) -> object:
        from vfe4.training.production import run_readiness

        return run_readiness(
            training=training,
            paths=paths,
            source_lock=source_lock,
        )

    def train(
        self,
        *,
        training: "TrainingConfig",
        paths: _LauncherPaths,
        source_lock: object,
        readiness: object,
    ) -> object:
        from vfe4.training.production import run_training

        return run_training(
            training=training,
            paths=paths,
            source_lock=source_lock,
            readiness=readiness,
            mode="train",
        )

    def resume(
        self,
        *,
        training: "TrainingConfig",
        paths: _LauncherPaths,
        source_lock: object,
        readiness: object,
    ) -> object:
        from vfe4.training.production import run_training

        return run_training(
            training=training,
            paths=paths,
            source_lock=source_lock,
            readiness=readiness,
            mode="resume",
        )


def _default_driver() -> TrainingOperationDriver:
    return _DefaultTrainingOperationDriver()


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TrainingLaunchError(f"{name} must be a mapping")
    return value


def _is_reparse_or_link(path: Path, status: os.stat_result) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    is_junction = getattr(path, "is_junction", None)
    return (
        stat.S_ISLNK(status.st_mode)
        or bool(getattr(status, "st_file_attributes", 0) & reparse_flag)
        or bool(is_junction is not None and is_junction())
    )


def _has_v3_component(path: Path) -> bool:
    for part in path.parts:
        compact = "".join(
            character for character in part.casefold() if character.isalnum()
        )
        if compact.startswith("v3transformer"):
            return True
    return False


def _reject_existing_reparse_components(path: Path, name: str) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            status = current.lstat()
        except FileNotFoundError:
            return
        except OSError as exc:
            raise TrainingLaunchError(
                f"{name} component metadata is unavailable: {current}: {exc}"
            ) from exc
        if _is_reparse_or_link(current, status):
            raise TrainingLaunchError(
                f"{name} cannot traverse a symlink, junction, or reparse point"
            )


def _uses_windows_device_namespace(value: str) -> bool:
    normalized_separators = value.replace("/", "\\")
    return normalized_separators.startswith(("\\\\?\\", "\\\\.\\"))


def _absolute_path(value: object, name: str) -> Path:
    if type(value) is not str or not value:
        raise TrainingLaunchError(f"{name} must be nonempty path text")
    if _uses_windows_device_namespace(value):
        raise TrainingLaunchError(
            f"{name} cannot use a Windows device namespace"
        )
    if any(character in value for character in "*?[]"):
        raise TrainingLaunchError(f"{name} cannot contain a glob")
    declared = Path(value)
    if not declared.is_absolute() or ".." in declared.parts:
        raise TrainingLaunchError(
            f"{name} must be an absolute normalized path"
        )
    absolute_text = os.path.abspath(declared)
    if _uses_windows_device_namespace(absolute_text):
        raise TrainingLaunchError(
            f"{name} cannot use a Windows device namespace"
        )
    path = Path(absolute_text)
    legacy_token_cache_root = Path(
        os.path.abspath(Path.home() / ".cache" / "tokenized_cache")
    )
    if _same_or_ancestor(legacy_token_cache_root, path):
        raise TrainingLaunchError(
            f"{name} cannot use the legacy V3 token cache tree"
        )
    if _has_v3_component(path):
        raise TrainingLaunchError(f"{name} cannot use a V3 path")
    return path


def _same_or_ancestor(left: Path, right: Path) -> bool:
    left_text = os.path.normcase(str(left))
    right_text = os.path.normcase(str(right))
    if left_text == right_text:
        return True
    try:
        Path(right_text).relative_to(Path(left_text))
    except ValueError:
        return False
    return True


def _resolve_paths(value: object) -> _LauncherPaths:
    raw = _mapping(value, "CONFIG.paths")
    if "resume_experiment_index_path" in raw:
        raise TrainingLaunchError(
            "resume_experiment_index_path was renamed to "
            "resume_experiment_plan_path; point it to the exact "
            "experiment-plan.json"
        )
    expected = {
        "cache_root",
        "run_root",
        "source_record_path",
        "resume_experiment_plan_path",
        "smoke_run_id",
    }
    if set(raw) != expected:
        raise TrainingLaunchError("CONFIG.paths has unknown or missing keys")
    smoke_run_id = raw["smoke_run_id"]
    if (
        type(smoke_run_id) is not str
        or not smoke_run_id
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789-_"
            for character in smoke_run_id
        )
    ):
        raise TrainingLaunchError(
            "CONFIG.paths.smoke_run_id is not a portable lowercase component"
        )
    paths = _LauncherPaths(
        cache_root=_absolute_path(raw["cache_root"], "cache_root"),
        run_root=_absolute_path(raw["run_root"], "run_root"),
        source_record_path=_absolute_path(
            raw["source_record_path"],
            "source_record_path",
        ),
        resume_experiment_plan_path=_absolute_path(
            raw["resume_experiment_plan_path"],
            "resume_experiment_plan_path",
        ),
        smoke_run_id=smoke_run_id,
    )
    if _same_or_ancestor(
        paths.cache_root,
        paths.run_root,
    ) or _same_or_ancestor(paths.run_root, paths.cache_root):
        raise TrainingLaunchError(
            "cache_root and run_root must be disjoint, non-nested roots"
        )
    if paths.source_record_path in (
        paths.cache_root,
        paths.run_root,
    ):
        raise TrainingLaunchError(
            "source_record_path cannot equal an artifact root"
        )
    if paths.resume_experiment_plan_path.name != "experiment-plan.json":
        raise TrainingLaunchError(
            "resume_experiment_plan_path must name experiment-plan.json"
        )
    if paths.resume_experiment_plan_path == paths.cache_root:
        raise TrainingLaunchError(
            "resume_experiment_plan_path cannot equal cache_root"
        )
    for name, path in (
        ("cache_root", paths.cache_root),
        ("run_root", paths.run_root),
        ("source_record_path", paths.source_record_path),
        (
            "resume_experiment_plan_path",
            paths.resume_experiment_plan_path,
        ),
    ):
        _reject_existing_reparse_components(path, name)
    return paths


def _resolve_launcher(
    value: object,
) -> tuple[TrainingConfig, _LauncherPaths, str | None]:
    from vfe4.config.training import resolve_training_config

    raw = _mapping(value, "CONFIG")
    if set(raw) != {
        "launcher_schema",
        "training",
        "paths",
        "authorization",
    }:
        raise TrainingLaunchError("CONFIG has unknown or missing root keys")
    path_mapping = _mapping(raw["paths"], "CONFIG.paths")
    if "resume_experiment_index_path" in path_mapping:
        raise TrainingLaunchError(
            "resume_experiment_index_path was renamed to "
            "resume_experiment_plan_path; point it to the exact "
            "experiment-plan.json"
        )
    if raw["launcher_schema"] != "wt103-click-launcher-v2":
        raise TrainingLaunchError("CONFIG launcher_schema is unsupported")
    try:
        training = resolve_training_config(raw["training"])
    except (TypeError, ValueError) as exc:
        raise TrainingLaunchError(
            f"training configuration is invalid: {exc}"
        ) from exc
    if training.operation not in OPERATIONS:
        raise TrainingLaunchError("training operation is unsupported")
    authorization = raw["authorization"]
    if authorization is not None and type(authorization) is not str:
        raise TrainingLaunchError(
            "CONFIG.authorization must be text or None"
        )
    return training, _resolve_paths(raw["paths"]), authorization


def _authorize(observed: str | None, expected: str, operation: str) -> None:
    if type(observed) is not str or not hmac.compare_digest(
        observed,
        expected,
    ):
        raise PermissionError(
            f"{operation} requires its exact explicit authorization phrase"
        )


def _require_finalized_source(path: Path) -> None:
    if not path.is_file():
        raise TrainingLaunchError(
            "finalized production source record is absent; run the "
            "separately authorized source-lock transaction first"
        )


def _run_synthetic_smoke(
    training: TrainingConfig,
    paths: _LauncherPaths,
) -> object:
    from vfe4.training.smoke import run_wt103_synthetic_smoke

    return run_wt103_synthetic_smoke(
        config=training,
        cache_root=paths.cache_root,
        run_root=paths.run_root,
        smoke_run_id=paths.smoke_run_id,
    )


def main(
    config: object = CONFIG,
    *,
    driver: TrainingOperationDriver | None = None,
) -> TrainingLauncherResult:
    training, paths, authorization = _resolve_launcher(config)
    operation = training.operation
    if operation == "idle":
        return TrainingLauncherResult(
            "wt103-click-launcher-result-v1",
            "idle",
            "IDLE",
        )
    if operation == "synthetic_smoke":
        if authorization is not None:
            raise TrainingLaunchError(
                "synthetic_smoke does not accept production authorization"
            )
        payload = _run_synthetic_smoke(training, paths)
    elif operation == "source_lock":
        _authorize(
            authorization,
            SOURCE_LOCK_AUTHORIZATION,
            operation,
        )
        operations = _default_driver() if driver is None else driver
        if not isinstance(operations, TrainingOperationDriver):
            raise TrainingLaunchError(
                "production driver does not implement the exact operation seam"
            )
        payload = operations.source_lock(
            training=training,
            paths=paths,
        )
    else:
        if operation in ("train", "resume"):
            _authorize(
                authorization,
                PRODUCTION_AUTHORIZATION,
                operation,
            )
        elif authorization is not None:
            raise TrainingLaunchError(
                "readiness does not accept training authorization"
            )
        _require_finalized_source(paths.source_record_path)
        operations = _default_driver() if driver is None else driver
        if not isinstance(operations, TrainingOperationDriver):
            raise TrainingLaunchError(
                "production driver does not implement the exact operation seam"
            )
        source_lock = operations.reopen_source_lock(
            training=training,
            paths=paths,
        )
        readiness = operations.readiness(
            training=training,
            paths=paths,
            source_lock=source_lock,
        )
        if operation == "readiness":
            payload = readiness
        elif operation == "train":
            payload = operations.train(
                training=training,
                paths=paths,
                source_lock=source_lock,
                readiness=readiness,
            )
        elif operation == "resume":
            payload = operations.resume(
                training=training,
                paths=paths,
                source_lock=source_lock,
                readiness=readiness,
            )
        else:
            raise AssertionError("unreachable production operation")
    return TrainingLauncherResult(
        "wt103-click-launcher-result-v1",
        operation,
        "COMPLETED",
        payload,
    )


def _script_main() -> int:
    try:
        result = main()
    except (OSError, PermissionError, RuntimeError, TypeError, ValueError) as exc:
        print(f"VFE4 WikiText-103 operation unavailable: {exc}", file=sys.stderr)
        return 2
    if result.status == "IDLE":
        print(
            "VFE4 WikiText-103 launcher is idle; edit "
            "CONFIG['training']['operation'], then click Run."
        )
    else:
        print(f"VFE4 WikiText-103 operation completed: {result.operation}")
        print(f"artifact={result.payload}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_script_main())
