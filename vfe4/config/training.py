"""Strict, side-effect-free WikiText-103 configuration resolution."""

from __future__ import annotations

import hashlib
from dataclasses import fields, is_dataclass

from vfe4.config.schema import FigureConfig, TrainingConfig
from vfe4.types.figures import FigureSpec, default_figure_specs
from vfe4.types.training import (
    A0ArchitectureProfile,
    AdamWProfile,
    CadenceProfile,
    CandidateTokenizerContract,
    CheckpointProfile,
    EndpointInventory,
    EstimatorProtocol,
    NonclaimProfile,
    PrecisionProfile,
    ResourceProfile,
    SchedulerProfile,
    SchemaProfile,
    ScientificPreconditionProfile,
    StatisticalProfile,
    WT103ExperimentProfile,
    WT103_TUNING_CELLS,
    canonical_json_bytes,
    default_wt103_arm_specs,
    default_wt103_gate_specs,
    owned_sha256,
)


_OPERATIONS = (
    "idle",
    "synthetic_smoke",
    "source_lock",
    "readiness",
    "train",
    "resume",
)


def _mapping(value: object, location: str) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{location} must be a plain mapping")
    if any(type(key) is not str for key in value):
        raise ValueError(f"{location} keys must be plain strings")
    return value


def _keys(
    value: dict[str, object],
    expected: tuple[str, ...],
    location: str,
) -> None:
    missing = tuple(key for key in expected if key not in value)
    unknown = tuple(key for key in value if key not in expected)
    if missing or unknown:
        raise ValueError(
            f"{location} keys differ; missing={missing}, unknown={unknown}"
        )


def _exact(value: object, expected: object, location: str) -> object:
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"{location} must equal {expected!r}")
    return value


def _text(value: object, location: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{location} must be nonempty plain text")
    return value


def _integer(value: object, location: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{location} must be a plain int")
    return value


def _floating(value: object, location: str) -> float:
    if type(value) is not float:
        raise ValueError(f"{location} must be a plain float")
    return value


def _boolean(value: object, location: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{location} must be a plain bool")
    return value


def _tuple(value: object, location: str) -> tuple[object, ...]:
    if type(value) is not tuple:
        raise ValueError(f"{location} must be an immutable tuple")
    return value


def _record_mapping(
    value: object,
    *,
    omit: tuple[str, ...],
) -> dict[str, object]:
    if not is_dataclass(value) or isinstance(value, type):
        raise ValueError("value must be a dataclass instance")
    return {
        item.name: _raw_value(getattr(value, item.name), omit=omit)
        for item in fields(value)
        if item.name not in omit
    }


def _raw_value(value: object, *, omit: tuple[str, ...] = ()) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _record_mapping(value, omit=omit)
    if type(value) is tuple:
        return tuple(_raw_value(item, omit=omit) for item in value)
    if type(value) in (str, int, float, bool) or value is None:
        return value
    raise ValueError(f"unsupported raw config value {type(value).__name__}")


def _default_profile_mapping() -> dict[str, object]:
    return {
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
            "proposal_acceptance": "validity_only_no_monotonicity_claim",
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
            "learning_rate_grid": (1.0e-4, 3.0e-4, 1.0e-3),
            "weight_decay_grid": (0.0, 1.0e-2),
            "tuning_seed_ids": (2026072199, 2026072200),
            "confirmatory_seed_ids": tuple(range(2026072101, 2026072109)),
            "data_order_seed": 2026072199,
            "validation_stream_ids": tuple(range(8)),
            "test_stream_ids": tuple(range(64)),
            "validation_particle_count": 256,
            "particle_counts": (128, 256, 512, 1024),
            "simultaneous_constant": 4.5144904535377144,
            "practical_threshold": 0.01005033585350145,
            "contraction_ratio": 0.75,
            "one_opening_policy": ("durable_exclusive_single_test_transaction"),
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
            "h8_parent_child_protocol": ("vfe4.h8.parent-child-protocol.v3"),
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


def _formula_sha256() -> str:
    return owned_sha256(
        "vfe4.wt103.a0-formula-schema.v1",
        {
            "parameter_formula_schema": "wt103-a0-parameter-formula-v1",
            "flop_formula_schema": "wt103-a0-semantic-train-flops-v1",
            "parameter_formula": ("2*V*h+128*h+12*h**2+15*h+V"),
            "primitive_flop_policy": (
                "multiply_and_named_arithmetic_each_one_comparison_zero"
            ),
        },
    )


def _default_a0_mapping() -> dict[str, object]:
    return {
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
            "torch.nn.attention.sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION])"
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
        "input_composition": "token_embedding_plus_position_embedding",
        "normalization": ("LayerNorm(eps=1e-5,elementwise_affine=true,bias=true)"),
        "normalization_placement": "pre_norm_with_final_norm",
        "residual_topology": ("x=x+attn(ln1(x));x=x+mlp(ln2(x));y=ln_f(x)"),
        "qkv_projection": "Linear(in=h,out=3h,weight[3h,h],bias[3h])",
        "attention_output_projection": ("Linear(in=h,out=h,weight[h,h],bias[h])"),
        "mlp_input_projection": ("Linear(in=h,out=4h,weight[4h,h],bias[4h])"),
        "activation": "gelu_tanh_approximation",
        "mlp_output_projection": ("Linear(in=4h,out=h,weight[h,4h],bias[h])"),
        "decoder_projection": ("untied_Linear(in=h,out=V,weight[V,h],bias[V])"),
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


def _default_estimator_mapping() -> dict[str, object]:
    return {
        "schema_version": "wt103-estimator-protocol-v1",
        "validation_particle_count": 256,
        "validation_stream_ids": tuple(range(8)),
        "test_stream_ids": tuple(range(64)),
        "particle_counts": (128, 256, 512, 1024),
        "validation_stream_domain": (
            "post-h8-wt103-validation-v1|2026072198|stream_id|purpose"
        ),
        "test_stream_domain": ("post-h8-wt103-test-v1|2026072198|stream_id|purpose"),
    }


def _default_preconditions_mapping() -> dict[str, object]:
    return {
        "schema_version": "wt103-scientific-preconditions-v1",
        "h6_prediction_authority": "native_executable_v3",
        "h6_prediction_schema": "h6-prediction-result-v3",
        "h8_schema": "h8-sparse-scale-v5",
        "h8_config_schema": "h8-validation-config-v3",
        "h8_parent_child_protocol": "vfe4.h8.parent-child-protocol.v3",
        "training_sparsity_schema": "wt103-training-sparsity-v1",
        "h8_reference_required": True,
        "training_sparsity_reference_required": True,
        "target_blind_predictor_safety_required": True,
        "h8_can_satisfy_training_sparsity": False,
        "capacity_can_satisfy_training_sparsity": False,
    }


def default_training_config_mapping() -> dict[str, object]:
    """Return a fresh launcher-friendly, hermetic Task 1 mapping."""

    return {
        "schema_version": "wt103-training-config-v1",
        "operation": "idle",
        "synthetic_authority": "nonproduction_synthetic_smoke",
        "candidate_tokenizer": {
            "distribution": "tiktoken",
            "version": "0.12.0",
            "encoding_name": "gpt2",
        },
        "profile": _default_profile_mapping(),
        "a0_architecture": _default_a0_mapping(),
        "estimator_protocol": _default_estimator_mapping(),
        "arms": [
            _record_mapping(arm, omit=("arm_spec_sha256",))
            for arm in default_wt103_arm_specs()
        ],
        "gates": [
            _record_mapping(gate, omit=("gate_spec_sha256",))
            for gate in default_wt103_gate_specs()
        ],
        "scientific_preconditions": _default_preconditions_mapping(),
    }


def _resolve_optimizer(value: object) -> AdamWProfile:
    raw = _mapping(value, "config.profile.optimizer")
    expected = tuple(_default_profile_mapping()["optimizer"])  # type: ignore[arg-type]
    _keys(raw, expected, "config.profile.optimizer")
    return AdamWProfile(
        optimizer=_exact(raw["optimizer"], "AdamW", "optimizer"),  # type: ignore[arg-type]
        betas=tuple(
            _floating(item, f"optimizer.betas[{index}]")
            for index, item in enumerate(_tuple(raw["betas"], "optimizer.betas"))
        ),  # type: ignore[arg-type]
        epsilon=_floating(raw["epsilon"], "optimizer.epsilon"),
        amsgrad=_boolean(raw["amsgrad"], "optimizer.amsgrad"),  # type: ignore[arg-type]
        foreach=_boolean(raw["foreach"], "optimizer.foreach"),  # type: ignore[arg-type]
        fused=_boolean(raw["fused"], "optimizer.fused"),  # type: ignore[arg-type]
        gradient_clip=_text(raw["gradient_clip"], "optimizer.gradient_clip"),  # type: ignore[arg-type]
        gradient_clip_max_norm=_floating(
            raw["gradient_clip_max_norm"],
            "optimizer.gradient_clip_max_norm",
        ),
        proposal_acceptance=_text(
            raw["proposal_acceptance"],
            "optimizer.proposal_acceptance",
        ),  # type: ignore[arg-type]
        reject_on=tuple(
            _text(item, f"optimizer.reject_on[{index}]")
            for index, item in enumerate(
                _tuple(raw["reject_on"], "optimizer.reject_on")
            )
        ),
    )


def _construct_exact_record(
    raw_value: object,
    *,
    location: str,
    expected: dict[str, object],
    record_type: type,
) -> object:
    raw = _mapping(raw_value, location)
    _keys(raw, tuple(expected), location)
    for key, expected_value in expected.items():
        value = raw[key]
        if type(expected_value) is tuple:
            observed_tuple = _tuple(value, f"{location}.{key}")
            if len(observed_tuple) != len(expected_value):
                raise ValueError(f"{location}.{key} length differs")
            for index, (observed, wanted) in enumerate(
                zip(observed_tuple, expected_value)
            ):
                _exact(observed, wanted, f"{location}.{key}[{index}]")
        else:
            _exact(value, expected_value, f"{location}.{key}")
    return record_type(**raw)


def _resolve_profile(value: object) -> WT103ExperimentProfile:
    raw = _mapping(value, "config.profile")
    expected = _default_profile_mapping()
    _keys(raw, tuple(expected), "config.profile")
    nested_names = (
        "optimizer",
        "scheduler",
        "precision",
        "cadence",
        "checkpoints",
        "statistics",
        "resources",
        "schemas",
        "nonclaims",
    )
    for key, expected_value in expected.items():
        if key in nested_names:
            continue
        _exact(raw[key], expected_value, f"config.profile.{key}")
    optimizer = _resolve_optimizer(raw["optimizer"])
    nested_types = {
        "scheduler": SchedulerProfile,
        "precision": PrecisionProfile,
        "cadence": CadenceProfile,
        "checkpoints": CheckpointProfile,
        "statistics": StatisticalProfile,
        "resources": ResourceProfile,
        "schemas": SchemaProfile,
        "nonclaims": NonclaimProfile,
    }
    nested: dict[str, object] = {"optimizer": optimizer}
    for name, record_type in nested_types.items():
        nested[name] = _construct_exact_record(
            raw[name],
            location=f"config.profile.{name}",
            expected=expected[name],  # type: ignore[arg-type]
            record_type=record_type,
        )
    scalars = {
        key: raw[key]
        for key in expected
        if key not in nested_names and key != "schema_version"
    }
    return WT103ExperimentProfile.create(**scalars, **nested)


def _resolve_a0(value: object) -> A0ArchitectureProfile:
    raw = _mapping(value, "config.a0_architecture")
    expected = _default_a0_mapping()
    _keys(raw, tuple(expected), "config.a0_architecture")
    for key, wanted in expected.items():
        observed = raw[key]
        if type(wanted) is tuple:
            observed_tuple = _tuple(
                observed,
                f"config.a0_architecture.{key}",
            )
            if observed_tuple != wanted or any(
                type(item) is not type(expected_item)
                for item, expected_item in zip(observed_tuple, wanted)
            ):
                raise ValueError(f"config.a0_architecture.{key} is not exact")
        else:
            _exact(observed, wanted, f"config.a0_architecture.{key}")
    values = {key: raw[key] for key in expected if key != "schema_version"}
    return A0ArchitectureProfile.create(
        **values,
        formula_sha256=_formula_sha256(),
    )


def _resolve_estimator(value: object) -> EstimatorProtocol:
    raw = _mapping(value, "config.estimator_protocol")
    expected = _default_estimator_mapping()
    _keys(raw, tuple(expected), "config.estimator_protocol")
    for key, wanted in expected.items():
        observed = raw[key]
        if type(wanted) is tuple:
            _exact(
                _tuple(observed, f"config.estimator_protocol.{key}"),
                wanted,
                f"config.estimator_protocol.{key}",
            )
        else:
            _exact(observed, wanted, f"config.estimator_protocol.{key}")
    return EstimatorProtocol.create()


def _resolve_arms(value: object) -> tuple:
    if type(value) is not list:
        raise ValueError("config.arms must be an ordered list of raw rows")
    expected = [
        _record_mapping(item, omit=("arm_spec_sha256",))
        for item in default_wt103_arm_specs()
    ]
    if len(value) != len(expected):
        raise ValueError("config.arms must contain exactly five rows")
    for index, (raw_row, expected_row) in enumerate(zip(value, expected)):
        row = _mapping(raw_row, f"config.arms[{index}]")
        _keys(row, tuple(expected_row), f"config.arms[{index}]")
        for key, wanted in expected_row.items():
            observed = row[key]
            if type(wanted) is tuple:
                _exact(
                    _tuple(observed, f"config.arms[{index}].{key}"),
                    wanted,
                    f"config.arms[{index}].{key}",
                )
            else:
                _exact(observed, wanted, f"config.arms[{index}].{key}")
    return default_wt103_arm_specs()


def _resolve_gates(value: object) -> tuple:
    if type(value) is not list:
        raise ValueError("config.gates must be an ordered list of raw rows")
    expected = [
        _record_mapping(item, omit=("gate_spec_sha256",))
        for item in default_wt103_gate_specs()
    ]
    if len(value) != len(expected):
        raise ValueError("config.gates must contain exactly seven rows")
    for index, (raw_row, expected_row) in enumerate(zip(value, expected)):
        row = _mapping(raw_row, f"config.gates[{index}]")
        _keys(row, tuple(expected_row), f"config.gates[{index}]")
        for key, wanted in expected_row.items():
            observed = row[key]
            if type(wanted) is tuple:
                _exact(
                    _tuple(observed, f"config.gates[{index}].{key}"),
                    wanted,
                    f"config.gates[{index}].{key}",
                )
            else:
                _exact(observed, wanted, f"config.gates[{index}].{key}")
    return default_wt103_gate_specs()


def _resolve_preconditions(value: object) -> ScientificPreconditionProfile:
    raw = _mapping(value, "config.scientific_preconditions")
    expected = _default_preconditions_mapping()
    _keys(raw, tuple(expected), "config.scientific_preconditions")
    for key, wanted in expected.items():
        _exact(raw[key], wanted, f"config.scientific_preconditions.{key}")
    return ScientificPreconditionProfile.create()


def resolve_training_config(raw: object) -> TrainingConfig:
    """Resolve a raw launcher mapping once into immutable records."""

    root = _mapping(raw, "config")
    expected_keys = (
        "schema_version",
        "operation",
        "synthetic_authority",
        "candidate_tokenizer",
        "profile",
        "a0_architecture",
        "estimator_protocol",
        "arms",
        "gates",
        "scientific_preconditions",
    )
    _keys(root, expected_keys, "config")
    _exact(
        root["schema_version"],
        "wt103-training-config-v1",
        "config.schema_version",
    )
    operation = _text(root["operation"], "config.operation")
    if operation not in _OPERATIONS:
        raise ValueError("config.operation is not a supported WT103 operation")
    _exact(
        root["synthetic_authority"],
        "nonproduction_synthetic_smoke",
        "config.synthetic_authority",
    )
    tokenizer_raw = _mapping(
        root["candidate_tokenizer"],
        "config.candidate_tokenizer",
    )
    _keys(
        tokenizer_raw,
        ("distribution", "version", "encoding_name"),
        "config.candidate_tokenizer",
    )
    candidate = CandidateTokenizerContract(
        distribution=_exact(
            tokenizer_raw["distribution"],
            "tiktoken",
            "config.candidate_tokenizer.distribution",
        ),  # type: ignore[arg-type]
        version=_exact(
            tokenizer_raw["version"],
            "0.12.0",
            "config.candidate_tokenizer.version",
        ),  # type: ignore[arg-type]
        encoding_name=_exact(
            tokenizer_raw["encoding_name"],
            "gpt2",
            "config.candidate_tokenizer.encoding_name",
        ),  # type: ignore[arg-type]
    )
    profile = _resolve_profile(root["profile"])
    architecture = _resolve_a0(root["a0_architecture"])
    estimator = _resolve_estimator(root["estimator_protocol"])
    arms = _resolve_arms(root["arms"])
    gates = _resolve_gates(root["gates"])
    preconditions = _resolve_preconditions(root["scientific_preconditions"])
    inventory = EndpointInventory.create(
        arms,
        gates,
        WT103_TUNING_CELLS,
        profile.statistics.tuning_seed_ids,
        profile.statistics.confirmatory_seed_ids,
        estimator,
    )
    if profile.statistics.validation_stream_ids != inventory.validation_stream_ids:
        raise ValueError("profile/inventory validation streams disagree")
    if profile.statistics.test_stream_ids != inventory.test_stream_ids:
        raise ValueError("profile/inventory test streams disagree")
    if profile.statistics.particle_counts != inventory.particle_counts:
        raise ValueError("profile/inventory particle counts disagree")
    if (
        profile.schemas.h6_prediction_schema != preconditions.h6_prediction_schema
        or profile.schemas.h8_schema != preconditions.h8_schema
        or profile.schemas.h8_config_schema != preconditions.h8_config_schema
        or profile.schemas.h8_parent_child_protocol
        != preconditions.h8_parent_child_protocol
        or profile.schemas.training_sparsity_schema
        != preconditions.training_sparsity_schema
    ):
        raise ValueError("profile/precondition schema identities disagree")
    payload = {
        "schema_version": "wt103-training-config-v1",
        "operation": operation,
        "synthetic_authority": "nonproduction_synthetic_smoke",
        "candidate_tokenizer": candidate,
        "profile": profile,
        "a0_architecture": architecture,
        "endpoint_inventory": inventory,
        "scientific_preconditions": preconditions,
    }
    canonical = canonical_json_bytes(payload).decode("utf-8")
    return TrainingConfig(
        **payload,
        canonical_json=canonical,
        config_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )  # type: ignore[arg-type]


def _figure_raw_semantics(spec: FigureSpec) -> dict[str, object]:
    def convert(value: object) -> object:
        if is_dataclass(value) and not isinstance(value, type):
            return {
                item.name: convert(getattr(value, item.name))
                for item in fields(value)
                if not item.name.endswith("_sha256")
            }
        if type(value) is tuple:
            return tuple(convert(item) for item in value)
        if type(value) in (str, int, float, bool) or value is None:
            return value
        raise ValueError("unsupported figure semantic value")

    converted = convert(spec)
    if type(converted) is not dict:
        raise ValueError("figure spec did not convert to a mapping")
    return converted


def default_figure_config_mapping(
    inventory: EndpointInventory,
) -> dict[str, object]:
    """Return a pure renderer mapping bound to one exact endpoint inventory."""

    if type(inventory) is not EndpointInventory:
        raise ValueError("inventory must be an exact EndpointInventory")
    inventory.__post_init__()
    return {
        "schema_version": "wt103-figure-config-v1",
        "operation": "idle",
        "run_group_manifest_path": "unresolved-until-render",
        "figure_root": "figures",
        "endpoint_inventory_sha256": inventory.endpoint_inventory_sha256,
        "rendering": {
            "backend": "Agg",
            "matplotlib_version": "3.10.6",
            "font_family": "DejaVu Sans",
            "svg_hashsalt": "vfe4-wt103-figure-v1",
            "metadata_policy": "fixed_no_current_timestamp",
        },
        "specs": [
            _figure_raw_semantics(spec) for spec in default_figure_specs(inventory)
        ],
    }


def resolve_figure_config(raw: object) -> FigureConfig:
    """Resolve an import-safe, no-training figure configuration."""

    root = _mapping(raw, "figure_config")
    _keys(
        root,
        (
            "schema_version",
            "operation",
            "run_group_manifest_path",
            "figure_root",
            "endpoint_inventory_sha256",
            "rendering",
            "specs",
        ),
        "figure_config",
    )
    _exact(
        root["schema_version"],
        "wt103-figure-config-v1",
        "figure_config.schema_version",
    )
    operation = _text(root["operation"], "figure_config.operation")
    if operation not in ("idle", "render"):
        raise ValueError("figure_config.operation must be idle or render")
    run_path = _text(
        root["run_group_manifest_path"],
        "figure_config.run_group_manifest_path",
    )
    figure_root = _text(root["figure_root"], "figure_config.figure_root")
    if "v3_transformer" in (run_path + figure_root).casefold():
        raise ValueError("figure config cannot reference a V3 path")
    inventory = EndpointInventory.create(
        default_wt103_arm_specs(),
        default_wt103_gate_specs(),
        WT103_TUNING_CELLS,
        (2026072199, 2026072200),
        tuple(range(2026072101, 2026072109)),
        EstimatorProtocol.create(),
    )
    _exact(
        root["endpoint_inventory_sha256"],
        inventory.endpoint_inventory_sha256,
        "figure_config.endpoint_inventory_sha256",
    )
    rendering = _mapping(root["rendering"], "figure_config.rendering")
    expected_rendering = {
        "backend": "Agg",
        "matplotlib_version": "3.10.6",
        "font_family": "DejaVu Sans",
        "svg_hashsalt": "vfe4-wt103-figure-v1",
        "metadata_policy": "fixed_no_current_timestamp",
    }
    _keys(rendering, tuple(expected_rendering), "figure_config.rendering")
    for key, wanted in expected_rendering.items():
        _exact(rendering[key], wanted, f"figure_config.rendering.{key}")
    specs = default_figure_specs(inventory)
    provided_specs = root["specs"]
    if type(provided_specs) is not list:
        raise ValueError("figure_config.specs must be an ordered raw list")
    expected_specs = [_figure_raw_semantics(spec) for spec in specs]
    if provided_specs != expected_specs:
        raise ValueError(
            "figure_config.specs differ from the inventory-derived registry"
        )
    payload = {
        "schema_version": "wt103-figure-config-v1",
        "operation": operation,
        "run_group_manifest_path": run_path,
        "figure_root": figure_root,
        "endpoint_inventory_sha256": inventory.endpoint_inventory_sha256,
        "backend": "Agg",
        "matplotlib_version": "3.10.6",
        "font_family": "DejaVu Sans",
        "svg_hashsalt": "vfe4-wt103-figure-v1",
        "metadata_policy": "fixed_no_current_timestamp",
        "specs": specs,
    }
    canonical = canonical_json_bytes(payload).decode("utf-8")
    return FigureConfig(
        **payload,
        canonical_json=canonical,
        config_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )  # type: ignore[arg-type]


__all__ = [
    "default_figure_config_mapping",
    "default_training_config_mapping",
    "resolve_figure_config",
    "resolve_training_config",
]
