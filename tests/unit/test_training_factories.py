from __future__ import annotations

import dataclasses

import pytest
import torch
from torch import nn

from vfe4.config import (
    default_training_config_mapping,
    resolve_training_config,
)
from vfe4.training.factories import (
    A0FactoryInputs,
    build_wt103_a0,
    build_wt103_a5_fixed,
    build_wt103_a5_nolatent,
    build_wt103_a5_parent_specific,
    build_wt103_arm,
    audit_arm_matching,
    scorer_dispatch,
)
from vfe4.training.formulas import (
    A0FlopWorkload,
    build_a0_architecture_profile,
    build_a0_formula_record,
    reconstruct_a0_flops,
    reconstruct_a0_parameters,
)
from vfe4.training.wt103_models import (
    OptimizerParameterBinding,
    WT103A0Model,
    WT103ArmRuntimeComponents,
)
from vfe4.types.results import GateStatus


class _TinyLatentRuntime(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model_weight = nn.Parameter(torch.ones(2, 2))
        self.latent_weight = nn.Parameter(torch.ones(2))
        self.source_weight = nn.Parameter(torch.ones(2))
        self.recognition_weight = nn.Parameter(torch.ones(2))


class _TinyNoLatentRuntime(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model_weight = nn.Parameter(torch.ones(2, 2))


def _resolved():
    return resolve_training_config(default_training_config_mapping())


def _tiny_latent_components() -> WT103ArmRuntimeComponents:
    return WT103ArmRuntimeComponents.create(
        model=_TinyLatentRuntime(),
        model_parameter_names=("model_weight",),
        latent_parameter_names=("latent_weight",),
        source_parameter_names=("source_weight",),
        frame_parameter_names=(),
        recognition_parameter_names=("recognition_weight",),
        optimizer_bindings=(
            OptimizerParameterBinding(
                optimizer_id="recognition_adamw",
                parameter_names=("recognition_weight",),
            ),
            OptimizerParameterBinding(
                optimizer_id="model_adamw",
                parameter_names=(
                    "model_weight",
                    "latent_weight",
                    "source_weight",
                ),
            ),
        ),
        filler_parameter_names=(),
        dormant_parameter_names=(),
    )


def _tiny_nolatent_components() -> WT103ArmRuntimeComponents:
    return WT103ArmRuntimeComponents.create(
        model=_TinyNoLatentRuntime(),
        model_parameter_names=("model_weight",),
        latent_parameter_names=(),
        source_parameter_names=(),
        frame_parameter_names=(),
        recognition_parameter_names=(),
        optimizer_bindings=(
            OptimizerParameterBinding(
                optimizer_id="model_adamw",
                parameter_names=("model_weight",),
            ),
        ),
        filler_parameter_names=(),
        dormant_parameter_names=(),
    )


def test_exact_a0_named_shapes_reconstruct_the_closed_parameter_formula() -> None:
    tiny = WT103A0Model(
        vocabulary_size=11,
        positional_capacity=8,
        hidden_width=4,
        attention_heads=2,
        layer_norm_epsilon=1.0e-5,
        device=torch.device("meta"),
        dtype=torch.float32,
    )
    tiny_inventory = reconstruct_a0_parameters(
        tiny,
        vocabulary_size=11,
        positional_capacity=8,
        hidden_width=4,
    )

    assert tiny_inventory.parameter_count == 383
    assert tuple(item.name for item in tiny_inventory.parameters) == (
        "token_embedding.weight",
        "position_embedding.weight",
        "block.ln1.weight",
        "block.ln1.bias",
        "block.qkv.weight",
        "block.qkv.bias",
        "block.attention_output.weight",
        "block.attention_output.bias",
        "block.ln2.weight",
        "block.ln2.bias",
        "block.mlp_input.weight",
        "block.mlp_input.bias",
        "block.mlp_output.weight",
        "block.mlp_output.bias",
        "final_norm.weight",
        "final_norm.bias",
        "decoder.weight",
        "decoder.bias",
    )
    assert len({id(parameter) for parameter in tiny.parameters()}) == len(
        tuple(tiny.parameters())
    )
    assert not {
        "latent",
        "recognition",
        "source_prior",
        "frame",
    } & set(vars(tiny))

    production = WT103A0Model(
        vocabulary_size=50_257,
        positional_capacity=128,
        hidden_width=20,
        attention_heads=2,
        layer_norm_epsilon=1.0e-5,
        device=torch.device("meta"),
        dtype=torch.float32,
    )
    production_inventory = reconstruct_a0_parameters(
        production,
        vocabulary_size=50_257,
        positional_capacity=128,
        hidden_width=20,
    )
    assert production_inventory.parameter_count == 2_068_197


def test_tiny_independent_flop_ledger_is_exact_and_decoder_rechunk_invariant() -> None:
    workload = A0FlopWorkload(
        batch_size=1,
        sequence_length=2,
        vocabulary_size=5,
        hidden_width=4,
        parameter_count=329,
        decoder_chunk_size=1,
        optimizer_steps=2,
        validation_batches=1,
    )
    ledger = reconstruct_a0_flops(workload)
    rechunked = reconstruct_a0_flops(
        dataclasses.replace(workload, decoder_chunk_size=2)
    )

    assert ledger.forward_and_ce_flops_per_batch == 1_522
    assert ledger.backward_flops_per_batch == 2_658
    assert ledger.adamw_flops_per_step == 4_935
    assert ledger.train_flops_per_step == 9_115
    assert ledger.semantic_train_flops == 19_752
    assert ledger.terms == rechunked.terms
    assert ledger.semantic_train_flops == rechunked.semantic_train_flops
    assert ledger.ledger_sha256 == rechunked.ledger_sha256


def test_a0_architecture_binds_formula_and_resolved_flash_api_identity() -> None:
    model = WT103A0Model(
        vocabulary_size=50_257,
        positional_capacity=128,
        hidden_width=20,
        attention_heads=2,
        layer_norm_epsilon=1.0e-5,
        device=torch.device("meta"),
        dtype=torch.float32,
    )
    inventory = reconstruct_a0_parameters(
        model,
        vocabulary_size=50_257,
        positional_capacity=128,
        hidden_width=20,
    )
    ledger = reconstruct_a0_flops(
        A0FlopWorkload(
            batch_size=1,
            sequence_length=2,
            vocabulary_size=50_257,
            hidden_width=20,
            parameter_count=inventory.parameter_count,
            decoder_chunk_size=2,
            optimizer_steps=1,
            validation_batches=0,
        )
    )
    formula = build_a0_formula_record(inventory=inventory, ledger=ledger)
    architecture = build_a0_architecture_profile(
        hidden_width=20,
        formula=formula,
        source_lock_scope="production_source_lock_verified",
        pytorch_version="2.8.0+cu128",
        sdpa_api_sha256="1" * 64,
        flash_backend_sha256="2" * 64,
    )
    changed_api = build_a0_architecture_profile(
        hidden_width=20,
        formula=formula,
        source_lock_scope="production_source_lock_verified",
        pytorch_version="2.8.0+cu128",
        sdpa_api_sha256="3" * 64,
        flash_backend_sha256="2" * 64,
    )

    assert architecture.attention_heads == 2
    assert architecture.attention_context == "full_causal_inclusive_self"
    assert architecture.attention_backend_policy == (
        "flash_attention_only_no_fallback"
    )
    assert architecture.pytorch_sdpa_api_binding == (
        "torch.nn.attention.sdpa_kernel("
        "backends=[SDPBackend.FLASH_ATTENTION])"
    )
    assert architecture.enabled_backends == ("FLASH_ATTENTION",)
    assert architecture.alternative_backends_disabled is True
    assert architecture.attention_mask_argument is None
    assert architecture.attention_returns_weights is False
    assert architecture.backend_fallback_allowed is False
    assert architecture.fused_attention_materialization == "forbidden"
    assert architecture.positional_encoding == "learned_absolute"
    assert architecture.normalization_placement == "pre_norm_with_final_norm"
    assert architecture.residual_topology == (
        "x=x+attn(ln1(x));x=x+mlp(ln2(x));y=ln_f(x)"
    )
    assert architecture.activation == "gelu_tanh_approximation"
    assert architecture.architecture_sha256 != changed_api.architecture_sha256
    with pytest.raises(ValueError):
        dataclasses.replace(
            architecture,
            enabled_backends=("FLASH_ATTENTION", "MATH"),
        )
    with pytest.raises(ValueError):
        dataclasses.replace(architecture, backend_fallback_allowed=True)


def test_five_exact_specs_dispatch_through_four_direct_constructors() -> None:
    resolved = _resolved()
    profile = resolved.profile
    arms = resolved.endpoint_inventory.arms
    workload = A0FlopWorkload(
        batch_size=1,
        sequence_length=2,
        vocabulary_size=profile.vocabulary_size,
        hidden_width=20,
        parameter_count=2_068_197,
        decoder_chunk_size=2,
        optimizer_steps=1,
        validation_batches=0,
    )
    target_ledger = reconstruct_a0_flops(workload)
    matching = audit_arm_matching(
        profile=profile,
        endpoint_inventory=resolved.endpoint_inventory,
        primary_parameter_count=2_068_197,
        primary_semantic_train_flops=target_ledger.semantic_train_flops,
        workload_template=workload,
        optimizer_access_exact=True,
    )
    assert matching.status is GateStatus.PASS
    assert matching.selected_hidden_width == 20
    model = WT103A0Model(
        vocabulary_size=profile.vocabulary_size,
        positional_capacity=128,
        hidden_width=matching.selected_hidden_width,
        attention_heads=2,
        layer_norm_epsilon=1.0e-5,
        device=torch.device("meta"),
        dtype=torch.float32,
    )
    inventory = reconstruct_a0_parameters(
        model,
        vocabulary_size=profile.vocabulary_size,
        positional_capacity=128,
        hidden_width=20,
    )
    formula = build_a0_formula_record(inventory=inventory, ledger=target_ledger)
    architecture = build_a0_architecture_profile(
        hidden_width=20,
        formula=formula,
        source_lock_scope="candidate_unverified",
        pytorch_version="unresolved_until_task13_source_lock",
        sdpa_api_sha256="unresolved_until_task13_source_lock",
        flash_backend_sha256="unresolved_until_task13_source_lock",
    )
    a0_inputs = A0FactoryInputs(
        architecture=architecture,
        formula=formula,
        flop_ledger=target_ledger,
        matching=matching,
        device=torch.device("meta"),
        dtype=torch.float32,
    )

    builds = (
        build_wt103_a0(
            spec=arms[0],
            profile=profile,
            inputs=a0_inputs,
            execution_scope="nonproduction_synthetic_smoke",
        ),
        build_wt103_a5_parent_specific(
            spec=arms[1],
            profile=profile,
            runtime=_tiny_latent_components(),
            execution_scope="nonproduction_synthetic_smoke",
        ),
        build_wt103_a5_fixed(
            spec=arms[2],
            profile=profile,
            runtime=_tiny_latent_components(),
            execution_scope="nonproduction_synthetic_smoke",
        ),
        build_wt103_a5_parent_specific(
            spec=arms[3],
            profile=profile,
            runtime=_tiny_latent_components(),
            execution_scope="nonproduction_synthetic_smoke",
        ),
        build_wt103_a5_nolatent(
            spec=arms[4],
            profile=profile,
            runtime=_tiny_nolatent_components(),
            execution_scope="nonproduction_synthetic_smoke",
        ),
    )

    assert tuple(build.record.spec.arm_id for build in builds) == tuple(
        arm.arm_id for arm in arms
    )
    assert tuple(build.record.constructor_id for build in builds) == (
        "build_wt103_a0",
        "build_wt103_a5_parent_specific",
        "build_wt103_a5_fixed",
        "build_wt103_a5_parent_specific",
        "build_wt103_a5_nolatent",
    )
    assert len({build.record.constructor_id for build in builds}) == 4
    assert builds[1].record.training_objective == "complete_elbo"
    assert builds[3].record.training_objective == (
        "emission_only_ablation_non_elbo"
    )
    assert builds[4].record.recognition_parameter_names == ()
    assert builds[4].record.latent_parameter_names == ()
    assert builds[4].record.source_parameter_names == ()
    assert builds[4].record.optimizer_bindings == (
        OptimizerParameterBinding(
            optimizer_id="model_adamw",
            parameter_names=("model_weight",),
        ),
    )
    assert scorer_dispatch(builds[0]) == "exact_autoregressive"
    assert scorer_dispatch(builds[1]) == "weighted_smc"

    dispatched = tuple(
        build_wt103_arm(
            spec=arm,
            profile=profile,
            a0_inputs=a0_inputs if index == 0 else None,
            runtime=(
                None
                if index == 0
                else (
                    _tiny_nolatent_components()
                    if index == 4
                    else _tiny_latent_components()
                )
            ),
            execution_scope="nonproduction_synthetic_smoke",
        ).record.constructor_id
        for index, arm in enumerate(arms)
    )
    assert dispatched == tuple(build.record.constructor_id for build in builds)
    with pytest.raises(ValueError):
        build_wt103_a5_parent_specific(
            spec=arms[2],
            profile=profile,
            runtime=_tiny_latent_components(),
            execution_scope="nonproduction_synthetic_smoke",
        )


def test_matching_is_deterministic_and_enforces_both_primary_margins() -> None:
    resolved = _resolved()
    profile = resolved.profile
    template = A0FlopWorkload(
        batch_size=1,
        sequence_length=2,
        vocabulary_size=profile.vocabulary_size,
        hidden_width=20,
        parameter_count=2_068_197,
        decoder_chunk_size=2,
        optimizer_steps=1,
        validation_batches=0,
    )
    exact_flops = reconstruct_a0_flops(template).semantic_train_flops
    exact = audit_arm_matching(
        profile=profile,
        endpoint_inventory=resolved.endpoint_inventory,
        primary_parameter_count=2_068_197,
        primary_semantic_train_flops=exact_flops,
        workload_template=template,
        optimizer_access_exact=True,
    )
    outside_parameter_margin = audit_arm_matching(
        profile=profile,
        endpoint_inventory=resolved.endpoint_inventory,
        primary_parameter_count=round(2_068_197 / 1.0101),
        primary_semantic_train_flops=exact_flops,
        workload_template=template,
        optimizer_access_exact=True,
    )
    absent_optimizer_access = audit_arm_matching(
        profile=profile,
        endpoint_inventory=resolved.endpoint_inventory,
        primary_parameter_count=2_068_197,
        primary_semantic_train_flops=exact_flops,
        workload_template=template,
        optimizer_access_exact=False,
    )

    assert exact.status is GateStatus.PASS
    assert exact.selected_hidden_width == 20
    assert outside_parameter_margin.status is GateStatus.INCONCLUSIVE
    assert outside_parameter_margin.selected_hidden_width is None
    assert absent_optimizer_access.status is GateStatus.INCONCLUSIVE
    assert all(not row.eligible for row in absent_optimizer_access.rows)


def test_runtime_inventory_rejects_filler_dormant_and_duplicate_optimizer_state() -> None:
    runtime = _TinyLatentRuntime()
    with pytest.raises(ValueError, match="filler"):
        WT103ArmRuntimeComponents.create(
            model=runtime,
            model_parameter_names=("model_weight",),
            latent_parameter_names=("latent_weight",),
            source_parameter_names=("source_weight",),
            frame_parameter_names=(),
            recognition_parameter_names=("recognition_weight",),
            optimizer_bindings=(
                OptimizerParameterBinding(
                    optimizer_id="recognition_adamw",
                    parameter_names=("recognition_weight",),
                ),
                OptimizerParameterBinding(
                    optimizer_id="model_adamw",
                    parameter_names=(
                        "model_weight",
                        "latent_weight",
                        "source_weight",
                    ),
                ),
            ),
            filler_parameter_names=("unused",),
            dormant_parameter_names=(),
        )
    with pytest.raises(ValueError, match="exactly once"):
        WT103ArmRuntimeComponents.create(
            model=runtime,
            model_parameter_names=("model_weight",),
            latent_parameter_names=("latent_weight",),
            source_parameter_names=("source_weight",),
            frame_parameter_names=(),
            recognition_parameter_names=("recognition_weight",),
            optimizer_bindings=(
                OptimizerParameterBinding(
                    optimizer_id="recognition_adamw",
                    parameter_names=("recognition_weight",),
                ),
                OptimizerParameterBinding(
                    optimizer_id="model_adamw",
                    parameter_names=(
                        "model_weight",
                        "latent_weight",
                        "source_weight",
                        "recognition_weight",
                    ),
                ),
            ),
            filler_parameter_names=(),
            dormant_parameter_names=(),
        )
