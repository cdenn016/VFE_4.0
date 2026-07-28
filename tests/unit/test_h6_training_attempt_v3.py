from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from torch import nn

from vfe4.data.byte_tokenizer import BOS_ID, IGNORE_TARGET_ID
from vfe4.data.windows import CausalPrefix, CausalWindows
from vfe4.training.h6_engine_v3 import (
    H6BatchLiveRecognitionStateV3,
    H6EngineAuthorityV3,
    H6LiveObjectiveTermV3,
    H6LiveRecognitionStateV3,
    H6PhaseObjectiveV3,
    run_h6_training_batch_v3,
)
from vfe4.training.arms import (
    BuiltArm,
    H6CausalTransformer,
    LatentLanguageArmModel,
    MeanPooledPrefixFloor,
    build_a0,
    build_a5,
)
from vfe4.training.h6_training_attempt_v3 import (
    _H6BatchCallbacksV3,
    _H6TrainingWindowBatchV3,
    _window_batch_v3,
)
from vfe4.training.h6_transformer_v3 import (
    H6TrainingCausalTransformerV3,
)
from vfe4.training.matching import ArmConfig, CapacityAllocation
from vfe4.types.h6 import ArmId, TrainingPhase, VocabularyIdentity
from vfe4.types.h6_prediction_v3 import (
    H6AttemptCursorV3,
    H6_NO_COUNTER_CONSUMPTION_SHA256,
)


def _vocabulary() -> VocabularyIdentity:
    return VocabularyIdentity.from_tokenizer_spec(
        vocabulary_id="h6-training-attempt-v3-tests",
        size=19,
        tokenizer_spec_bytes=b"h6-training-attempt-v3",
    )


def _model(*, prior_variant: str) -> LatentLanguageArmModel:
    return LatentLanguageArmModel(
        arm=ArmId.A5,
        vocabulary=_vocabulary(),
        horizon=3,
        emission_width=4,
        latent_width=2,
        state_channel_enabled=True,
        model_channel_enabled=True,
        source_mode="categorical",
        map_mode="shared_vertex_coboundary",
        prior_variant=prior_variant,
        prior_context_width=5
        if prior_variant == "parent_specific_pooled_prefix"
        else None,
        predictor_config_sha256="1" * 64,
        model_family_sha256="2" * 64,
    )


def _categorical_a5_config() -> ArmConfig:
    return ArmConfig.create(
        arm=ArmId.A5,
        config_id=(
            "h6-a5-structured-parent-specific-prefix-exact-complete-latent-smoothing-v2"
        ),
        vocabulary=_vocabulary(),
        horizon=3,
        latent_enabled=True,
        state_channel_enabled=True,
        model_channel_enabled=True,
        source_mode="categorical",
        map_mode="shared_vertex_coboundary",
        recognition_family="structured",
        recognition_conditioning="smoothing",
        prior_variant="parent_specific_pooled_prefix",
        mixture_mode="exact",
        objective_kind="complete_elbo",
        capacity_allocation=CapacityAllocation.create(
            emission_width=4,
            latent_width=2,
            recognition_width=5,
            prior_context_width=5,
        ),
    )


def _emission_only_a5_config() -> ArmConfig:
    return ArmConfig.create(
        arm=ArmId.A5,
        config_id=(
            "h6-a5-structured-parent-specific-prefix-exact-emission-latent-smoothing-v2"
        ),
        vocabulary=_vocabulary(),
        horizon=3,
        latent_enabled=True,
        state_channel_enabled=True,
        model_channel_enabled=True,
        source_mode="categorical",
        map_mode="shared_vertex_coboundary",
        recognition_family="structured",
        recognition_conditioning="smoothing",
        prior_variant="parent_specific_pooled_prefix",
        mixture_mode="exact",
        objective_kind="emission_only_ablation_non_elbo",
        capacity_allocation=CapacityAllocation.create(
            emission_width=4,
            latent_width=2,
            recognition_width=5,
            prior_context_width=5,
        ),
    )


def _no_latent_a5_config() -> ArmConfig:
    return ArmConfig.create(
        arm=ArmId.A5,
        config_id=("h6-a5-structured-fixed-exact-complete-nolatent-norecognition-v1"),
        vocabulary=_vocabulary(),
        horizon=3,
        latent_enabled=False,
        state_channel_enabled=False,
        model_channel_enabled=False,
        source_mode="absent",
        map_mode="absent",
        recognition_family="absent",
        recognition_conditioning="absent",
        prior_variant="absent",
        mixture_mode="absent",
        objective_kind="cross_entropy",
        capacity_allocation=CapacityAllocation.create(
            emission_width=4,
            latent_width=None,
            recognition_width=None,
            prior_context_width=None,
        ),
    )


def _engine_authority(*, receiver_count: int = 2) -> H6EngineAuthorityV3:
    return H6EngineAuthorityV3.create(
        attempt_spec_sha256="3" * 64,
        endpoint_config_sha256="4" * 64,
        readiness_sha256="5" * 64,
        readiness_matching_set_sha256="6" * 64,
        matching_set_sha256="6" * 64,
        matching_policy_sha256="7" * 64,
        readiness_training_schedule_sha256="8" * 64,
        training_schedule_sha256="8" * 64,
        readiness_runtime_identity_sha256="9" * 64,
        runtime_identity_sha256="9" * 64,
        planned_attempt_sha256="a" * 64,
        endpoint_config_id="h6-batch-wrapper-test",
        matching_ledger_sha256="b" * 64,
        matching_report_sha256s=("c" * 64,),
        receiver_count=receiver_count,
        state_categorical_enabled=True,
        model_categorical_enabled=False,
        tuning_cell_sha256="d" * 64,
        optimizer_policy_sha256=(
            "67b498399b293d4f267cb7ffbe5f0e329ac0025adaaa5f86869588ad720f5ce8"
        ),
        optimizer_learning_rate=1.0e-3,
        optimizer_weight_decay=0.0,
        objective_kind="complete_elbo",
        latent_enabled=True,
    )


def _live_state(example_ordinal: int) -> H6LiveRecognitionStateV3:
    value = float(example_ordinal + 1)
    return H6LiveRecognitionStateV3.create(
        endpoint_config_sha256="4" * 64,
        receiver_count=2,
        state_categorical_enabled=True,
        model_categorical_enabled=False,
        state_categorical_supports=((0,),),
        model_categorical_supports=(None,),
        receiver_components=((0, ("base",)), (1, ("terminal",))),
        tensors={
            "receiver.0.component.base.mean": torch.tensor(
                (value, value + 0.5),
                dtype=torch.float64,
                requires_grad=True,
            ),
            "receiver.0.shared_precision_cholesky": torch.eye(
                2, dtype=torch.float64, requires_grad=True
            ),
            "receiver.1.component.terminal.mean": torch.tensor(
                (value + 1.0, value + 1.5),
                dtype=torch.float64,
                requires_grad=True,
            ),
            "receiver.1.shared_precision_cholesky": torch.eye(
                2, dtype=torch.float64, requires_grad=True
            ),
            "state.receiver.1.support": torch.tensor((0,), dtype=torch.int64),
            "state.receiver.1.categorical_row": torch.ones(
                1, dtype=torch.float64, requires_grad=True
            ),
            "model.absent.support": torch.tensor((-1,), dtype=torch.int64),
            "model.absent.categorical_row": torch.ones(1, dtype=torch.float64),
        },
        context_sha256=f"{example_ordinal + 1:x}" * 64,
        recognition_state_sha256=f"{example_ordinal + 3:x}" * 64,
        source_model_sha256="e" * 64,
        law_sha256=f"{example_ordinal + 5:x}" * 64,
    )


def _ragged_live_state(
    *,
    active_target_count: int,
    example_ordinal: int,
) -> H6LiveRecognitionStateV3:
    receiver_count = active_target_count + 1
    tensors: dict[str, torch.Tensor] = {}
    receiver_components: list[tuple[int, tuple[str, ...]]] = []
    for receiver_t in range(receiver_count):
        component_id = "terminal" if receiver_t == active_target_count else "ordinary"
        receiver_components.append((receiver_t, (component_id,)))
        tensors[f"receiver.{receiver_t}.component.{component_id}.mean"] = torch.full(
            (2,),
            float(receiver_t + example_ordinal + 1),
            dtype=torch.float64,
            requires_grad=True,
        )
        tensors[f"receiver.{receiver_t}.shared_precision_cholesky"] = torch.eye(
            2, dtype=torch.float64, requires_grad=True
        )
    state_supports = tuple(
        tuple(range(receiver_t)) for receiver_t in range(1, receiver_count)
    )
    for receiver_t, support in enumerate(state_supports, start=1):
        tensors[f"state.receiver.{receiver_t}.support"] = torch.tensor(
            support, dtype=torch.int64
        )
        tensors[f"state.receiver.{receiver_t}.categorical_row"] = torch.full(
            (len(support),),
            1.0 / len(support),
            dtype=torch.float64,
            requires_grad=True,
        )
    tensors["model.absent.support"] = torch.tensor((-1,), dtype=torch.int64)
    tensors["model.absent.categorical_row"] = torch.ones(1, dtype=torch.float64)
    return H6LiveRecognitionStateV3.create(
        endpoint_config_sha256="4" * 64,
        receiver_count=receiver_count,
        state_categorical_enabled=True,
        model_categorical_enabled=False,
        state_categorical_supports=state_supports,
        model_categorical_supports=(None,),
        receiver_components=tuple(receiver_components),
        tensors=tensors,
        context_sha256=f"{example_ordinal + 1:x}" * 64,
        recognition_state_sha256=f"{example_ordinal + 3:x}" * 64,
        source_model_sha256="e" * 64,
        law_sha256=f"{example_ordinal + 5:x}" * 64,
    )


class _ScalarModule(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.tensor(value, dtype=torch.float64))


def _live_state_from_parameter(
    parameter: torch.Tensor,
    *,
    example_ordinal: int,
) -> H6LiveRecognitionStateV3:
    value = parameter + float(example_ordinal + 1)
    return H6LiveRecognitionStateV3.create(
        endpoint_config_sha256="4" * 64,
        receiver_count=2,
        state_categorical_enabled=True,
        model_categorical_enabled=False,
        state_categorical_supports=((0,),),
        model_categorical_supports=(None,),
        receiver_components=((0, ("base",)), (1, ("terminal",))),
        tensors={
            "receiver.0.component.base.mean": value.repeat(2),
            "receiver.0.shared_precision_cholesky": torch.eye(2, dtype=torch.float64),
            "receiver.1.component.terminal.mean": (value + 0.25).repeat(2),
            "receiver.1.shared_precision_cholesky": torch.eye(2, dtype=torch.float64),
            "state.receiver.1.support": torch.tensor((0,), dtype=torch.int64),
            "state.receiver.1.categorical_row": torch.ones(1, dtype=torch.float64),
            "model.absent.support": torch.tensor((-1,), dtype=torch.int64),
            "model.absent.categorical_row": torch.ones(1, dtype=torch.float64),
        },
        context_sha256=f"{example_ordinal + 1:x}" * 64,
        recognition_state_sha256=f"{example_ordinal + 3:x}" * 64,
        source_model_sha256="e" * 64,
        law_sha256=f"{example_ordinal + 5:x}" * 64,
    )


def _adamw(module: nn.Module) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        module.parameters(),
        lr=1.0e-3,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=0.0,
        amsgrad=False,
        maximize=False,
        foreach=False,
        capturable=False,
        differentiable=False,
        fused=False,
    )


def _tiny_a0_config() -> ArmConfig:
    return ArmConfig.create(
        arm=ArmId.A0,
        config_id="h6-a0-transformer-v2",
        vocabulary=VocabularyIdentity(
            vocabulary_id="h6-training-attempt-v3-a0-validation",
            size=3,
            tokenizer_spec_sha256="f" * 64,
        ),
        horizon=4,
        latent_enabled=False,
        state_channel_enabled=False,
        model_channel_enabled=False,
        source_mode="absent",
        map_mode="absent",
        recognition_family="absent",
        recognition_conditioning="absent",
        prior_variant="absent",
        mixture_mode="absent",
        objective_kind="cross_entropy",
        capacity_allocation=CapacityAllocation.create(
            emission_width=4,
            latent_width=None,
            recognition_width=None,
        ),
    )


def _tiny_training_windows() -> CausalWindows:
    return CausalWindows(
        split="train",
        inputs=((0, 1, 2) + (BOS_ID,) * 29,),
        targets=((1, 2) + (IGNORE_TARGET_ID,) * 30,),
        attention_masks=((True, True, True) + (False,) * 29,),
        starts=(0,),
        real_target_counts=(2,),
    )


def _tiny_callback_authority(config: ArmConfig) -> H6EngineAuthorityV3:
    return H6EngineAuthorityV3.create(
        attempt_spec_sha256="3" * 64,
        endpoint_config_sha256=config.config_sha256,
        readiness_sha256="5" * 64,
        readiness_matching_set_sha256="6" * 64,
        matching_set_sha256="6" * 64,
        matching_policy_sha256="7" * 64,
        readiness_training_schedule_sha256="8" * 64,
        training_schedule_sha256="8" * 64,
        readiness_runtime_identity_sha256="9" * 64,
        runtime_identity_sha256="9" * 64,
        planned_attempt_sha256="a" * 64,
        endpoint_config_id=config.config_id,
        matching_ledger_sha256="b" * 64,
        matching_report_sha256s=("c" * 64,),
        receiver_count=config.horizon + 1,
        state_categorical_enabled=(
            config.latent_enabled and config.source_mode == "categorical"
        ),
        model_categorical_enabled=(
            config.latent_enabled
            and config.model_channel_enabled
            and config.source_mode == "categorical"
        ),
        tuning_cell_sha256="d" * 64,
        optimizer_policy_sha256=(
            "67b498399b293d4f267cb7ffbe5f0e329ac0025adaaa5f86869588ad720f5ce8"
        ),
        optimizer_learning_rate=1.0e-3,
        optimizer_weight_decay=0.0,
        objective_kind=config.objective_kind,
        latent_enabled=config.latent_enabled,
    )


def _tiny_callback_cursor(
    authority: H6EngineAuthorityV3,
) -> H6AttemptCursorV3:
    return H6AttemptCursorV3.create(
        attempt_spec_sha256=authority.attempt_spec_sha256,
        pass_index=0,
        batch_index=0,
        next_phase=(
            TrainingPhase.RECOGNITION_ADAMW
            if authority.latent_enabled
            else TrainingPhase.MODEL_CE_ADAMW
        ),
        example_ordinal=0,
        draw_block=0,
        counter_consumption_sha256=H6_NO_COUNTER_CONSUMPTION_SHA256,
        permutation_sha256="1" * 64,
        recognition_update_count=0,
        model_update_count=0,
        validation_boundary_count=0,
        checkpoint_boundary_count=0,
    )


@dataclass(frozen=True)
class _TinySyntheticCallbackFixtureV3:
    """One bounded CPU callback case reusable by the Task 12 integration test."""

    built_arm: BuiltArm
    model: nn.Module
    recognition: nn.Module | None
    authority: H6EngineAuthorityV3
    windows: CausalWindows
    batch: _H6TrainingWindowBatchV3
    cursor: H6AttemptCursorV3
    model_optimizer: torch.optim.AdamW
    recognition_optimizer: torch.optim.AdamW | None

    def callbacks(self) -> _H6BatchCallbacksV3:
        return _H6BatchCallbacksV3(
            built_arm=self.built_arm,
            model=self.model,
            recognition=self.recognition,
            authority=self.authority,
            windows=self.windows,
            batch=self.batch,
        )


def _tiny_synthetic_callback_fixture_v3(
    arm: ArmId,
) -> _TinySyntheticCallbackFixtureV3:
    """Build only one active two-target window on the sanctioned CPU path."""

    windows = _tiny_training_windows()
    if arm is ArmId.A0:
        config = _tiny_a0_config()
        built = build_a0(config)
        assert type(built.model) is H6CausalTransformer
        model: nn.Module = H6TrainingCausalTransformerV3(
            vocabulary=config.vocabulary,
            profile=built.model.profile,
            allow_synthetic_cpu=True,
        )
        loaded = model.load_state_dict(built.model.state_dict(), strict=True)
        assert not loaded.missing_keys
        assert not loaded.unexpected_keys
        recognition: nn.Module | None = None
    elif arm is ArmId.A5:
        config = _categorical_a5_config()
        built = build_a5(config)
        model = built.model
        recognition = built.recognition_store
        assert type(model) is LatentLanguageArmModel
        assert recognition is not None
    else:
        raise ValueError("tiny callback fixture supports only A0 and A5")
    authority = _tiny_callback_authority(config)
    return _TinySyntheticCallbackFixtureV3(
        built_arm=built,
        model=model,
        recognition=recognition,
        authority=authority,
        windows=windows,
        batch=_window_batch_v3(
            windows=windows,
            window_indices=(0,),
            maximum_horizon=config.horizon,
        ),
        cursor=_tiny_callback_cursor(authority),
        model_optimizer=_adamw(model),
        recognition_optimizer=(None if recognition is None else _adamw(recognition)),
    )


def _assert_same_module_state(left: nn.Module, right: nn.Module) -> None:
    left_state = left.state_dict()
    right_state = right.state_dict()
    assert tuple(left_state) == tuple(right_state)
    for name in left_state:
        torch.testing.assert_close(left_state[name], right_state[name], rtol=0, atol=0)


@pytest.fixture(scope="module")
def _tiny_attempt_authority_v3(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[object, object, object]:
    import test_h6_readiness_v3 as readiness_fixtures
    from vfe4.artifacts.h6_prediction_v3 import (
        H6PredictionV3Authorities,
    )
    from vfe4.data.access import H6TrainingDataV3
    from vfe4.data.windows import build_causal_windows
    from vfe4.training.h6_experiment_v3 import plan_h6_experiment_v3
    from vfe4.training.h6_matching_v3 import (
        H6MatchingSetV3,
        H6TrainingWorkloadV3,
    )
    from vfe4.training.h6_readiness import (
        _derive_h6_prediction_readiness_v3,
    )
    from vfe4.training.h6_runtime_v3 import H6SyntheticCpuRuntimeV3

    workload = H6TrainingWorkloadV3.from_train_tokens(
        train_token_count=2,
        train_token_sha256="4" * 64,
    )
    matching_set = H6MatchingSetV3.create(
        git_head=readiness_fixtures._GIT_HEAD,
        dirty_digest=readiness_fixtures._DIRTY_DIGEST,
        workload=workload,
        endpoint_templates=readiness_fixtures._endpoint_templates(),
    )
    config = readiness_fixtures._config(
        matching_set=matching_set,
        artifact_root=tmp_path_factory.mktemp("h6-attempt-recovery-authority"),
    )
    readiness = _derive_h6_prediction_readiness_v3(
        config=config,
        matching_set=matching_set,
        git_head=readiness_fixtures._GIT_HEAD,
        dirty_digest=readiness_fixtures._DIRTY_DIGEST,
    )
    plan = plan_h6_experiment_v3(
        readiness=readiness,
        matching_set=matching_set,
        training_schedule=config.training_schedule,
        runtime_identity=config.runtime,
    )
    authorities = H6PredictionV3Authorities.create(
        config=config,
        matching_set=matching_set,
        readiness=readiness,
        plan=plan,
    )
    training_data = H6TrainingDataV3(
        data_identity_sha256=config.data_identity_sha256,
        readiness_sha256=readiness.readiness_sha256,
        plan_sha256=plan.plan_sha256,
        matching_set_sha256=matching_set.matching_set_sha256,
        runtime_identity_sha256=config.runtime.runtime_identity_sha256,
        windows=build_causal_windows((2, 3), split="train"),
        vocabulary=plan.endpoint_configs[0].vocabulary,
    )
    return (
        authorities,
        training_data,
        H6SyntheticCpuRuntimeV3(fixture_id="h6-attempt-recovery-bounded-cpu"),
    )


def test_live_prior_provider_uses_exact_causal_source_factor_without_model_gradients() -> (
    None
):
    from vfe4.training.h6_training_attempt_v3 import (
        H6GenerativePriorFeatureProviderV3,
    )

    model = _model(prior_variant="parent_specific_pooled_prefix")
    provider = H6GenerativePriorFeatureProviderV3(model=model)
    means = torch.tensor(
        (
            (0.1, 0.2, 0.3, 0.4),
            (0.2, 0.3, 0.4, 0.5),
        ),
        dtype=torch.float64,
        requires_grad=True,
    )
    prefix = CausalPrefix.create(
        receiver_t=2,
        vocabulary=model.vocabulary,
        token_ids=torch.tensor((4,), dtype=torch.int64),
    )

    feature = provider(
        bank="state",
        causal_prefix=prefix,
        earlier_recognition_means=means,
    )
    expected = model.state_source_log_probs(
        2,
        prefix=prefix,
        earlier_latents=means[:, : model.latent_width],
    )

    assert provider.source_model is model
    assert feature.support == (0, 1)
    torch.testing.assert_close(feature.log_prior_features, expected)

    feature.log_prior_features.sum().backward()
    assert means.grad is not None
    assert bool(torch.any(means.grad != 0.0))
    assert all(parameter.grad is None for parameter in model.parameters())


def test_categorical_arm_factory_constructs_declared_trainable_source_banks() -> None:
    built = build_a5(_categorical_a5_config())

    assert built.recognition_store is not None
    assert built.recognition_store.trainable_source_banks == (
        "state",
        "model",
    )
    assert {
        name
        for name, _parameter in built.recognition_store.named_parameters()
        if name.startswith("source_")
    } == {
        "source_residual_vectors.state",
        "source_residual_vectors.model",
        "source_lag_scalars.state",
        "source_lag_scalars.model",
        "source_shift_vectors.state",
        "source_shift_vectors.model",
    }


def test_batch_live_law_and_snapshot_keep_every_example_qualified_tensor() -> None:
    from vfe4.training.h6_engine_v3 import (
        H6BatchLiveRecognitionStateV3,
        H6DetachedBatchRecognitionSnapshotV3,
    )

    authority = _engine_authority()
    live = H6BatchLiveRecognitionStateV3.create(
        authority=authority,
        states=(_live_state(0), _live_state(1)),
        active_target_counts=(1, 1),
        active_receiver_masks=(
            (True, True),
            (True, True),
        ),
    )

    assert live.example_ordinals == (0, 1)
    assert len(live.names) == 2 * len(_live_state(0).names)
    assert "example.0.state.receiver.1.categorical_row" in live.names
    assert "example.1.state.receiver.1.categorical_row" in live.names
    assert live["example.0.receiver.0.component.base.mean"].requires_grad

    cursor = H6AttemptCursorV3.create(
        attempt_spec_sha256=authority.attempt_spec_sha256,
        pass_index=0,
        batch_index=0,
        next_phase=TrainingPhase.MODEL_ADAMW,
        example_ordinal=0,
        draw_block=1,
        counter_consumption_sha256="f" * 64,
        permutation_sha256="1" * 64,
        recognition_update_count=1,
        model_update_count=0,
        validation_boundary_count=0,
        checkpoint_boundary_count=0,
    )
    snapshot = H6DetachedBatchRecognitionSnapshotV3.capture(
        live,
        authority=authority,
        post_recognition_cursor=cursor,
    )

    assert snapshot.example_ordinals == (0, 1)
    assert snapshot.active_target_counts == (1, 1)
    assert snapshot.names == live.names
    assert all(
        not snapshot[name].requires_grad and snapshot[name].grad_fn is None
        for name in snapshot.names
    )


def test_batch_recognition_law_persists_exact_ragged_prefix_states() -> None:
    from vfe4.training.h6_engine_v3 import (
        H6BatchLiveRecognitionStateV3,
        H6DetachedBatchRecognitionSnapshotV3,
    )

    authority = _engine_authority(receiver_count=4)
    live = H6BatchLiveRecognitionStateV3.create(
        authority=authority,
        states=(
            _ragged_live_state(
                active_target_count=1,
                example_ordinal=0,
            ),
            _ragged_live_state(
                active_target_count=3,
                example_ordinal=1,
            ),
        ),
        active_target_counts=(1, 3),
        active_receiver_masks=(
            (True, True, False, False),
            (True, True, True, True),
        ),
    )
    assert live.receiver_count == 4
    assert tuple(state.receiver_count for state in live.states) == (2, 4)
    assert "example.0.receiver.1.component.terminal.mean" in live.names
    assert "example.0.receiver.2.component.ordinary.mean" not in live.names

    cursor = H6AttemptCursorV3.create(
        attempt_spec_sha256=authority.attempt_spec_sha256,
        pass_index=0,
        batch_index=0,
        next_phase=TrainingPhase.MODEL_ADAMW,
        example_ordinal=0,
        draw_block=1,
        counter_consumption_sha256="f" * 64,
        permutation_sha256="1" * 64,
        recognition_update_count=1,
        model_update_count=0,
        validation_boundary_count=0,
        checkpoint_boundary_count=0,
    )
    snapshot = H6DetachedBatchRecognitionSnapshotV3.capture(
        live,
        authority=authority,
        post_recognition_cursor=cursor,
    )
    assert tuple(state.receiver_count for state in snapshot.states) == (2, 4)
    assert snapshot.active_receiver_masks[0] == (
        True,
        True,
        False,
        False,
    )


def test_engine_runs_one_recognition_and_one_model_update_for_whole_batch() -> None:
    from vfe4.training.h6_engine_v3 import (
        H6BatchLiveRecognitionStateV3,
        H6DetachedBatchRecognitionSnapshotV3,
    )

    authority = _engine_authority()
    model = _ScalarModule(0.75)
    recognition = _ScalarModule(0.25)
    model_before = model.value.detach().clone()
    recognition_before = recognition.value.detach().clone()

    def recognition_forward() -> H6BatchLiveRecognitionStateV3:
        return H6BatchLiveRecognitionStateV3.create(
            authority=authority,
            states=tuple(
                _live_state_from_parameter(
                    recognition.value,
                    example_ordinal=ordinal,
                )
                for ordinal in range(2)
            ),
            active_target_counts=(1, 1),
            active_receiver_masks=((True, True), (True, True)),
        )

    def objective_forward(
        *,
        recognition_state: object,
        **_values: object,
    ) -> H6PhaseObjectiveV3:
        if type(recognition_state) is H6BatchLiveRecognitionStateV3:
            value = sum(
                (
                    recognition_state[
                        f"example.{ordinal}.receiver.1.component.terminal.mean"
                    ].sum()
                    for ordinal in recognition_state.example_ordinals
                ),
                start=recognition.value.new_zeros(()),
            )
        elif type(recognition_state) is H6DetachedBatchRecognitionSnapshotV3:
            detached_total = sum(
                (
                    recognition_state[
                        f"example.{ordinal}.receiver.1.component.terminal.mean"
                    ].sum()
                    for ordinal in recognition_state.example_ordinals
                ),
                start=model.value.new_zeros(()),
            )
            value = model.value * detached_total
        else:
            raise AssertionError("engine passed a non-batch recognition law")
        return H6PhaseObjectiveV3.complete_elbo(
            (
                H6LiveObjectiveTermV3.create(
                    partition="emission",
                    receiver_t=1,
                    value=value,
                ),
            )
        )

    cursor = H6AttemptCursorV3.create(
        attempt_spec_sha256=authority.attempt_spec_sha256,
        pass_index=0,
        batch_index=0,
        next_phase=TrainingPhase.RECOGNITION_ADAMW,
        example_ordinal=0,
        draw_block=0,
        counter_consumption_sha256=H6_NO_COUNTER_CONSUMPTION_SHA256,
        permutation_sha256="1" * 64,
        recognition_update_count=0,
        model_update_count=0,
        validation_boundary_count=0,
        checkpoint_boundary_count=0,
    )
    result = run_h6_training_batch_v3(
        authority=authority,
        cursor=cursor,
        model=model,
        recognition=recognition,
        model_optimizer=_adamw(model),
        recognition_optimizer=_adamw(recognition),
        recognition_forward=recognition_forward,
        objective_forward=objective_forward,
        noise_factory=lambda phase, _cursor: (
            torch.zeros((2, 2, 2), dtype=torch.float64),
            "2" * 64 if phase is TrainingPhase.RECOGNITION_ADAMW else "3" * 64,
        ),
    )

    assert result.cursor.recognition_update_count == 1
    assert result.cursor.model_update_count == 1
    assert len(result.phase_records) == 2
    assert type(result.snapshot) is H6DetachedBatchRecognitionSnapshotV3
    assert not torch.equal(recognition.value.detach(), recognition_before)
    assert not torch.equal(model.value.detach(), model_before)


def test_engine_starts_next_batch_without_retaining_prior_batch_graph_history() -> None:
    from vfe4.training.h6_engine_v3 import H6BatchLiveRecognitionStateV3

    authority = _engine_authority()
    model = _ScalarModule(0.75)
    recognition = _ScalarModule(0.25)
    model_optimizer = _adamw(model)
    recognition_optimizer = _adamw(recognition)

    def recognition_forward() -> H6BatchLiveRecognitionStateV3:
        return H6BatchLiveRecognitionStateV3.create(
            authority=authority,
            states=(
                _live_state_from_parameter(
                    recognition.value,
                    example_ordinal=0,
                ),
            ),
            active_target_counts=(1,),
            active_receiver_masks=((True, True),),
        )

    def objective_forward(
        *,
        recognition_state: object,
        **_values: object,
    ) -> H6PhaseObjectiveV3:
        value = (
            recognition.value * 1.0
            if type(recognition_state) is H6BatchLiveRecognitionStateV3
            else model.value * 1.0
        )
        return H6PhaseObjectiveV3.complete_elbo(
            (
                H6LiveObjectiveTermV3.create(
                    partition="emission",
                    receiver_t=1,
                    value=value,
                ),
            )
        )

    cursor = H6AttemptCursorV3.create(
        attempt_spec_sha256=authority.attempt_spec_sha256,
        pass_index=0,
        batch_index=0,
        next_phase=TrainingPhase.RECOGNITION_ADAMW,
        example_ordinal=0,
        draw_block=0,
        counter_consumption_sha256=H6_NO_COUNTER_CONSUMPTION_SHA256,
        permutation_sha256="1" * 64,
        recognition_update_count=0,
        model_update_count=0,
        validation_boundary_count=0,
        checkpoint_boundary_count=0,
    )
    first = run_h6_training_batch_v3(
        authority=authority,
        cursor=cursor,
        model=model,
        recognition=recognition,
        model_optimizer=model_optimizer,
        recognition_optimizer=recognition_optimizer,
        recognition_forward=recognition_forward,
        objective_forward=objective_forward,
        noise_factory=lambda phase, _cursor: (
            torch.zeros((1, 2, 2), dtype=torch.float64),
            "2" * 64 if phase is TrainingPhase.RECOGNITION_ADAMW else "3" * 64,
        ),
    )
    second = run_h6_training_batch_v3(
        authority=authority,
        cursor=first.cursor,
        model=model,
        recognition=recognition,
        model_optimizer=model_optimizer,
        recognition_optimizer=recognition_optimizer,
        recognition_forward=recognition_forward,
        objective_forward=objective_forward,
        noise_factory=lambda phase, _cursor: (
            torch.zeros((1, 2, 2), dtype=torch.float64),
            "4" * 64 if phase is TrainingPhase.RECOGNITION_ADAMW else "5" * 64,
        ),
    )

    assert second.cursor.batch_index == 2
    assert second.cursor.recognition_update_count == 2
    assert second.cursor.model_update_count == 2
    assert len(second.phase_records) == 2
    assert len(second.metric_records) == 2


def test_batch_counter_owns_one_key_per_example_and_one_ordered_digest() -> None:
    from vfe4.training.h6_noise_v3 import training_batch_normal_tensor_v3

    first = training_batch_normal_tensor_v3(
        attempt_spec_sha256="3" * 64,
        pass_index=2,
        batch_index=7,
        phase=TrainingPhase.MODEL_ADAMW,
        draw_block=11,
        example_count=3,
        receiver_count=4,
        active_receiver_counts=(4, 2, 3),
        latent_dimension=2,
        device="cpu",
    )
    repeated = training_batch_normal_tensor_v3(
        attempt_spec_sha256="3" * 64,
        pass_index=2,
        batch_index=7,
        phase=TrainingPhase.MODEL_ADAMW,
        draw_block=11,
        example_count=3,
        receiver_count=4,
        active_receiver_counts=(4, 2, 3),
        latent_dimension=2,
        device="cpu",
    )

    assert first.tensor.shape == (3, 4, 2)
    assert first.active_receiver_counts == (4, 2, 3)
    assert tuple(key.example_ordinal for key in first.keys) == (0, 1, 2)
    assert len(set(first.key_sha256s)) == 3
    assert len(set(first.example_consumption_sha256s)) == 3
    assert first.consumption_sha256 == repeated.consumption_sha256
    torch.testing.assert_close(first.tensor, repeated.tensor)

    shorter = training_batch_normal_tensor_v3(
        attempt_spec_sha256="3" * 64,
        pass_index=2,
        batch_index=7,
        phase=TrainingPhase.MODEL_ADAMW,
        draw_block=11,
        example_count=2,
        receiver_count=4,
        active_receiver_counts=(4, 2),
        latent_dimension=2,
        device="cpu",
    )
    assert shorter.consumption_sha256 != first.consumption_sha256


def test_batch_counter_binds_exact_ragged_consumed_bytes() -> None:
    from vfe4.training.h6_noise_v3 import training_batch_normal_tensor_v3

    common = {
        "attempt_spec_sha256": "3" * 64,
        "pass_index": 2,
        "batch_index": 7,
        "phase": TrainingPhase.MODEL_ADAMW,
        "draw_block": 11,
        "example_count": 3,
        "receiver_count": 4,
        "latent_dimension": 2,
        "device": "cpu",
    }
    first = training_batch_normal_tensor_v3(
        **common,
        active_receiver_counts=(4, 2, 3),
    )
    reordered = training_batch_normal_tensor_v3(
        **common,
        active_receiver_counts=(4, 3, 2),
    )

    assert first.consumption_sha256 != reordered.consumption_sha256
    assert (
        first.example_consumption_sha256s[0] == reordered.example_consumption_sha256s[0]
    )
    for example_ordinal, shared_count in enumerate((4, 2, 2)):
        torch.testing.assert_close(
            first.tensor[example_ordinal, :shared_count],
            reordered.tensor[example_ordinal, :shared_count],
            rtol=0,
            atol=0,
        )
    assert torch.count_nonzero(first.tensor[1, 2:]) == 0
    assert torch.count_nonzero(reordered.tensor[2, 2:]) == 0


def test_prior_feature_batch_captures_complete_stopped_state_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vfe4.recognition.h6_prediction_v3 as recognition_v3
    import vfe4.training.h6_training_attempt_v3 as attempt_v3

    config = _categorical_a5_config()
    built = build_a5(config)
    assert type(built.model) is LatentLanguageArmModel
    assert built.recognition_store is not None
    windows = _tiny_training_windows()
    callbacks = _H6BatchCallbacksV3(
        built_arm=built,
        model=built.model,
        recognition=built.recognition_store,
        authority=_tiny_callback_authority(config),
        windows=windows,
        batch=_window_batch_v3(
            windows=windows,
            window_indices=(0,),
            maximum_horizon=config.horizon,
        ),
    )
    calls = {"attempt": 0, "recognition": 0}
    attempt_capture = attempt_v3._stopped_module_state
    recognition_capture = recognition_v3._stopped_provider_state

    def counted_attempt(module: nn.Module) -> object:
        calls["attempt"] += 1
        return attempt_capture(module)

    def counted_recognition(provider: object) -> object:
        calls["recognition"] += 1
        return recognition_capture(provider)  # type: ignore[arg-type]

    monkeypatch.setattr(attempt_v3, "_stopped_module_state", counted_attempt)
    monkeypatch.setattr(
        recognition_v3,
        "_stopped_provider_state",
        counted_recognition,
    )

    callbacks.recognition_forward()

    assert calls == {"attempt": 1, "recognition": 0}


def test_latent_batch_reuses_exact_post_step_forward_for_model_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vfe4.training.h6_training_attempt_v3 as attempt_v3

    baseline_case = _tiny_synthetic_callback_fixture_v3(ArmId.A5)
    observed_case = _tiny_synthetic_callback_fixture_v3(ArmId.A5)
    assert baseline_case.recognition is not None
    assert observed_case.recognition is not None
    baseline_callbacks = baseline_case.callbacks()
    baseline = run_h6_training_batch_v3(
        authority=baseline_case.authority,
        cursor=baseline_case.cursor,
        model=baseline_case.model,
        recognition=baseline_case.recognition,
        model_optimizer=baseline_case.model_optimizer,
        recognition_optimizer=baseline_case.recognition_optimizer,
        recognition_forward=baseline_callbacks.recognition_forward,
        objective_forward=baseline_callbacks.objective_forward,
        noise_factory=baseline_callbacks.noise_factory,
        checkpoint_at_batch_end=False,
    )

    captures = 0
    original_capture = attempt_v3._stopped_module_state

    def counted_capture(module: nn.Module) -> object:
        nonlocal captures
        captures += 1
        return original_capture(module)

    monkeypatch.setattr(
        attempt_v3,
        "_stopped_module_state",
        counted_capture,
    )
    observed_callbacks = observed_case.callbacks()
    recognition_forwards = 0

    def counted_forward() -> H6BatchLiveRecognitionStateV3:
        nonlocal recognition_forwards
        recognition_forwards += 1
        return observed_callbacks.recognition_forward()

    observed = run_h6_training_batch_v3(
        authority=observed_case.authority,
        cursor=observed_case.cursor,
        model=observed_case.model,
        recognition=observed_case.recognition,
        model_optimizer=observed_case.model_optimizer,
        recognition_optimizer=observed_case.recognition_optimizer,
        recognition_forward=counted_forward,
        objective_forward=observed_callbacks.objective_forward,
        noise_factory=observed_callbacks.noise_factory,
        checkpoint_at_batch_end=False,
    )

    assert recognition_forwards == 2
    assert captures == 2
    assert observed.result_sha256 == baseline.result_sha256
    assert (
        observed_callbacks.latest_total_raw_bytes_sha256
        == baseline_callbacks.latest_total_raw_bytes_sha256
    )
    assert (
        observed_callbacks.latest_factor_bindings
        == baseline_callbacks.latest_factor_bindings
    )
    _assert_same_module_state(observed_case.model, baseline_case.model)
    _assert_same_module_state(
        observed_case.recognition,
        baseline_case.recognition,
    )


def test_emission_only_objective_never_executes_non_emission_operators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vfe4.objective.h6_prediction_v3 as objective_v3
    import vfe4.recognition.h6_prediction_v3 as recognition_v3
    from vfe4.recognition.h6_prediction_v3 import (
        build_receiver_contexts,
    )
    from vfe4.recognition.language import RecognitionConditioning

    config = _emission_only_a5_config()
    built = build_a5(config)
    oracle_built = build_a5(config)
    assert built.elbo_factor_inventory == ("emission",)
    assert type(built.model) is LatentLanguageArmModel
    assert type(oracle_built.model) is LatentLanguageArmModel
    assert built.recognition_store is not None
    assert oracle_built.recognition_store is not None
    _assert_same_module_state(built.model, oracle_built.model)
    _assert_same_module_state(
        built.recognition_store,
        oracle_built.recognition_store,
    )
    windows = _tiny_training_windows()
    callbacks = _H6BatchCallbacksV3(
        built_arm=built,
        model=built.model,
        recognition=built.recognition_store,
        authority=_tiny_callback_authority(config),
        windows=windows,
        batch=_window_batch_v3(
            windows=windows,
            window_indices=(0,),
            maximum_horizon=config.horizon,
        ),
    )

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("non-emission operator executed")

    for name in (
        "initial_log_prob",
        "state_source_log_probs",
        "model_source_log_probs",
        "state_transition_mean",
        "model_transition_mean",
    ):
        monkeypatch.setattr(built.model, name, forbidden)
    monkeypatch.setattr(
        objective_v3,
        "_precision_gaussian_entropy",
        forbidden,
    )
    monkeypatch.setattr(
        recognition_v3,
        "_categorical_entropy",
        forbidden,
    )
    for model in (built.model, oracle_built.model):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    live = callbacks.recognition_forward()
    noise, _digest = callbacks.noise_factory(
        TrainingPhase.RECOGNITION_ADAMW,
        _tiny_callback_cursor(callbacks.authority),
    )
    result = callbacks.objective_forward(
        phase=TrainingPhase.RECOGNITION_ADAMW,
        recognition_state=live,
        noise=noise,
    )

    # Independent source-prior-free oracle: reconstruct the recognition-only
    # categorical law and receiver Gaussians directly from primitive
    # parameters, then evaluate only the public emission operator.
    oracle_store = oracle_built.recognition_store
    targets = callbacks.batch.observed_targets[0]
    binding = callbacks.batch.active_horizons[0]
    conditioning = RecognitionConditioning.create(
        mode=config.recognition_conditioning,
        horizon=binding.active_horizon,
        observed_tokens=targets,
    )
    embedded = oracle_store.token_embedding(
        targets.to(device=oracle_store.token_embedding.weight.device)
    )
    contexts = build_receiver_contexts(
        conditioning=conditioning,
        token_embeddings=embedded,
    )
    base_means = torch.nn.functional.linear(
        contexts,
        oracle_store.mean_weight,
        oracle_store.mean_bias,
    )
    dimension = oracle_store.gaussian_dimension
    row, column = torch.tril_indices(
        dimension,
        dimension,
        device=oracle_store.packed_precision_cholesky.device,
    )
    packed = oracle_store.packed_precision_cholesky.clone()
    diagonal = row == column
    packed[diagonal] = torch.nn.functional.softplus(packed[diagonal]) + 1.0e-6
    precision_cholesky = torch.zeros(
        (dimension, dimension),
        dtype=torch.float64,
        device=packed.device,
    ).index_put((row, column), packed)
    nonterminal_t = 1
    nonterminal_displacement = torch.linalg.solve_triangular(
        precision_cholesky.transpose(-1, -2),
        noise[0, nonterminal_t].unsqueeze(-1),
        upper=True,
    ).squeeze(-1)
    nonterminal_sample = base_means[nonterminal_t] + nonterminal_displacement
    nonterminal_emission = oracle_built.model.emission_log_probs(
        state=nonterminal_sample[: oracle_built.model.latent_width],
        model=nonterminal_sample[
            oracle_built.model.latent_width : 2 * oracle_built.model.latent_width
        ],
    )[int(targets[nonterminal_t - 1].item())]

    terminal_t = binding.active_horizon
    support = torch.arange(
        terminal_t,
        dtype=torch.int64,
        device=contexts.device,
    )
    context_differences = contexts[terminal_t].unsqueeze(0) - contexts.index_select(
        0, support
    )
    lags = terminal_t - support.to(dtype=torch.float64)

    def recognition_only_source_bank(
        bank: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residuals = context_differences @ oracle_store.source_residual_vectors[
            bank
        ] + oracle_store.source_lag_scalars[bank][0] * torch.log1p(lags)
        return (
            torch.log_softmax(residuals, dim=0),
            residuals.unsqueeze(1)
            * oracle_store.source_shift_vectors[bank].unsqueeze(0),
        )

    state_log_probabilities, state_shifts = recognition_only_source_bank("state")
    model_log_probabilities, model_shifts = recognition_only_source_bank("model")
    terminal_emission = precision_cholesky.new_zeros(())
    latent_width = oracle_built.model.latent_width
    terminal_displacement = torch.linalg.solve_triangular(
        precision_cholesky.transpose(-1, -2),
        noise[0, terminal_t].unsqueeze(-1),
        upper=True,
    ).squeeze(-1)
    for state_source_j in range(terminal_t):
        for model_source_j in range(terminal_t):
            terminal_mean = torch.cat(
                (
                    base_means[terminal_t, :latent_width]
                    + state_shifts[state_source_j],
                    base_means[terminal_t, latent_width:]
                    + model_shifts[model_source_j],
                )
            )
            terminal_sample = terminal_mean + terminal_displacement
            terminal_emission = (
                terminal_emission
                + (
                    state_log_probabilities[state_source_j]
                    + model_log_probabilities[model_source_j]
                ).exp()
                * oracle_built.model.emission_log_probs(
                    state=terminal_sample[:latent_width],
                    model=terminal_sample[latent_width:],
                )[int(targets[terminal_t - 1].item())]
            )
    oracle_value = (nonterminal_emission + terminal_emission) / binding.active_horizon

    observed_named_parameters = tuple(built.recognition_store.named_parameters())
    oracle_named_parameters = tuple(oracle_store.named_parameters())
    assert tuple(name for name, _ in observed_named_parameters) == tuple(
        name for name, _ in oracle_named_parameters
    )
    observed_gradients = torch.autograd.grad(
        -result.value,
        tuple(parameter for _name, parameter in observed_named_parameters),
        allow_unused=True,
    )
    oracle_gradients = torch.autograd.grad(
        -oracle_value,
        tuple(parameter for _name, parameter in oracle_named_parameters),
        allow_unused=True,
    )

    torch.testing.assert_close(result.value, oracle_value, rtol=1e-12, atol=1e-12)
    for (name, _parameter), observed, expected in zip(
        observed_named_parameters,
        observed_gradients,
        oracle_gradients,
        strict=True,
    ):
        if expected is None:
            assert observed is None, f"{name} unexpectedly received a gradient"
        else:
            assert observed is not None, f"{name} is missing its emission gradient"
            torch.testing.assert_close(
                observed,
                expected,
                rtol=1e-12,
                atol=1e-12,
            )

    assert result.objective_kind == "emission_only_ablation_non_elbo"
    assert result.is_elbo is False
    assert result.partitions == built.elbo_factor_inventory
    assert (
        tuple(
            dict.fromkeys(
                name.rsplit(".", 1)[-1]
                for name, _receiver_t, _digest in callbacks.latest_factor_bindings
            )
        )
        == built.elbo_factor_inventory
    )


def test_catalog_publication_loss_adopts_ordinal_zero_orphan_without_replay(
    _tiny_attempt_authority_v3: tuple[object, object, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vfe4.training.h6_training_attempt_v3 as attempt_v3
    from vfe4.training.h6_execution_v3 import (
        bind_h6_executable_attempt_v3,
    )

    authorities, training_data, runtime = _tiny_attempt_authority_v3
    endpoint_config_id = (
        "h6-a5-structured-parent-specific-prefix-exact-complete-latent-smoothing-v2"
    )
    planned = next(
        attempt
        for attempt in authorities.plan.tuning_attempts  # type: ignore[union-attr]
        if attempt.endpoint_config_id == endpoint_config_id
    )
    executable = bind_h6_executable_attempt_v3(
        authorities=authorities,  # type: ignore[arg-type]
        planned_attempt=planned,
    )
    maximum_bytes = 256 * 1024 * 1024
    checkpoint_path = (
        tmp_path / f"{planned.endpoint_config_id}.catalog-loss.h6v3"
    ).resolve()
    progress_path = attempt_v3.h6_training_attempt_progress_path_v3(checkpoint_path)
    original_publish = attempt_v3._publish_progress_catalog_v3
    publication_attempts = 0

    def fail_before_catalog_publication(*args: object, **kwargs: object) -> object:
        nonlocal publication_attempts
        publication_attempts += 1
        if publication_attempts == 1:
            assert not progress_path.exists()
            raise RuntimeError("simulated pre-catalog process loss")
        return original_publish(*args, **kwargs)

    monkeypatch.setattr(
        attempt_v3,
        "_publish_progress_catalog_v3",
        fail_before_catalog_publication,
    )
    with pytest.raises(RuntimeError, match="pre-catalog process loss"):
        attempt_v3._execute_new_training_attempt_v3(
            executable=executable,
            training_data=training_data,
            runtime=runtime,
            checkpoint_path=checkpoint_path,
            maximum_checkpoint_bytes=maximum_bytes,
        )
    assert publication_attempts == 1
    assert not progress_path.exists()
    orphan_inventory = tuple(
        (
            boundary_kind,
            checkpoint.exists(),
            history.exists(),
        )
        for boundary_kind in (
            "post_recognition",
            "batch_boundary",
            "terminal",
        )
        for checkpoint, history in (
            attempt_v3._recovery_artifact_paths_v3(
                checkpoint_path=checkpoint_path,
                ordinal=0,
                boundary_kind=boundary_kind,
            ),
        )
    )
    assert orphan_inventory == (
        ("post_recognition", True, True),
        ("batch_boundary", False, False),
        ("terminal", False, False),
    )

    original_run_batch = attempt_v3.run_h6_training_batch_v3
    resumed_entries: list[tuple[int, int, bool]] = []

    def require_persisted_recognition_update(**kwargs: object) -> object:
        cursor = kwargs["cursor"]
        resume_state = kwargs.get("resume_state")
        assert isinstance(cursor, H6AttemptCursorV3)
        resumed_entries.append(
            (
                cursor.recognition_update_count,
                cursor.model_update_count,
                resume_state is not None,
            )
        )
        assert cursor.recognition_update_count == 1
        assert cursor.model_update_count == 0
        assert resume_state is not None
        return original_run_batch(**kwargs)

    monkeypatch.setattr(
        attempt_v3,
        "run_h6_training_batch_v3",
        require_persisted_recognition_update,
    )
    resumed = attempt_v3._execute_new_training_attempt_v3(
        executable=executable,
        training_data=training_data,
        runtime=runtime,
        checkpoint_path=checkpoint_path,
        maximum_checkpoint_bytes=maximum_bytes,
    )
    progress = attempt_v3.read_h6_training_attempt_progress_v3(
        path=progress_path,
        maximum_bytes=maximum_bytes,
    )

    assert resumed_entries == [(1, 0, True)]
    assert tuple(
        (boundary.ordinal, boundary.boundary_kind) for boundary in progress.boundaries
    ) == ((0, "post_recognition"), (1, "terminal"))
    assert resumed.terminal_cursor.recognition_update_count == 1
    assert resumed.terminal_cursor.model_update_count == 1


def test_no_latent_batch_uses_only_previously_observed_target_prefixes() -> None:
    from vfe4.training.h6_training_attempt_v3 import (
        _H6BatchCallbacksV3,
        _window_batch_v3,
    )

    class RecordingFloor(MeanPooledPrefixFloor):
        def __init__(self) -> None:
            super().__init__(
                vocabulary=_vocabulary(),
                emission_width=4,
            )
            self.seen: list[tuple[int, ...]] = []

        def prefix_log_probs(self, prefix: CausalPrefix) -> torch.Tensor:
            self.seen.append(tuple(int(value) for value in prefix.token_ids.tolist()))
            return super().prefix_log_probs(prefix)

    windows = CausalWindows(
        split="train",
        inputs=(
            (2, 7) + (BOS_ID,) * 30,
            (3, 5, 7, 9) + (BOS_ID,) * 28,
        ),
        targets=(
            (7,) + (IGNORE_TARGET_ID,) * 31,
            (5, 7, 9) + (IGNORE_TARGET_ID,) * 29,
        ),
        attention_masks=(
            (True, True) + (False,) * 30,
            (True, True, True, True) + (False,) * 28,
        ),
        starts=(0, 32),
        real_target_counts=(1, 3),
    )
    config = _no_latent_a5_config()
    built = build_a5(config)
    model = RecordingFloor()
    authority = H6EngineAuthorityV3.create(
        attempt_spec_sha256="3" * 64,
        endpoint_config_sha256=config.config_sha256,
        readiness_sha256="5" * 64,
        readiness_matching_set_sha256="6" * 64,
        matching_set_sha256="6" * 64,
        matching_policy_sha256="7" * 64,
        readiness_training_schedule_sha256="8" * 64,
        training_schedule_sha256="8" * 64,
        readiness_runtime_identity_sha256="9" * 64,
        runtime_identity_sha256="9" * 64,
        planned_attempt_sha256="a" * 64,
        endpoint_config_id=config.config_id,
        matching_ledger_sha256="b" * 64,
        matching_report_sha256s=("c" * 64,),
        receiver_count=4,
        state_categorical_enabled=False,
        model_categorical_enabled=False,
        tuning_cell_sha256="d" * 64,
        optimizer_policy_sha256=(
            "67b498399b293d4f267cb7ffbe5f0e329ac0025adaaa5f86869588ad720f5ce8"
        ),
        optimizer_learning_rate=1.0e-3,
        optimizer_weight_decay=0.0,
        objective_kind="cross_entropy",
        latent_enabled=False,
    )
    batch = _window_batch_v3(
        windows=windows,
        window_indices=(0, 1),
        maximum_horizon=3,
    )
    callbacks = _H6BatchCallbacksV3(
        built_arm=built,
        model=model,
        recognition=None,
        authority=authority,
        windows=windows,
        batch=batch,
    )

    objective = callbacks.objective_forward(
        phase=TrainingPhase.MODEL_CE_ADAMW,
        recognition_state=None,
        noise=torch.empty(0, dtype=torch.float64),
    )

    assert objective.objective_kind == "cross_entropy"
    assert model.seen == [(), (), (5,), (5, 7)]


def test_tiny_synthetic_a0_executes_one_callback_batch() -> None:
    case = _tiny_synthetic_callback_fixture_v3(ArmId.A0)
    callbacks = case.callbacks()
    model_before = {
        name: tensor.detach().clone()
        for name, tensor in case.model.state_dict().items()
    }

    result = run_h6_training_batch_v3(
        authority=case.authority,
        cursor=case.cursor,
        model=case.model,
        recognition=None,
        model_optimizer=case.model_optimizer,
        recognition_optimizer=None,
        recognition_forward=None,
        objective_forward=callbacks.objective_forward,
        noise_factory=callbacks.noise_factory,
        checkpoint_at_batch_end=False,
    )

    assert result.cursor.batch_index == 1
    assert result.cursor.model_update_count == 1
    assert result.phase_records[-1].phase is TrainingPhase.MODEL_CE_ADAMW
    assert callbacks.latest_phase is TrainingPhase.MODEL_CE_ADAMW
    assert any(
        not torch.equal(model_before[name], tensor)
        for name, tensor in case.model.state_dict().items()
    )


def test_model_phase_resume_authenticates_snapshot_with_fresh_callbacks() -> None:
    from vfe4.objective.h6_prediction_v3 import (
        _trajectory_live_tensors,
    )
    from vfe4.training.h6_engine_v3 import (
        H6DetachedBatchRecognitionSnapshotV3,
    )
    from vfe4.training.h6_training_attempt_v3 import (
        _trajectory_live_state_v3,
    )

    uninterrupted_case = _tiny_synthetic_callback_fixture_v3(ArmId.A5)
    resumed_case = _tiny_synthetic_callback_fixture_v3(ArmId.A5)
    assert uninterrupted_case.recognition is not None
    assert resumed_case.recognition is not None
    _assert_same_module_state(
        uninterrupted_case.model,
        resumed_case.model,
    )
    _assert_same_module_state(
        uninterrupted_case.recognition,
        resumed_case.recognition,
    )

    uninterrupted_callbacks = uninterrupted_case.callbacks()
    uninterrupted = run_h6_training_batch_v3(
        authority=uninterrupted_case.authority,
        cursor=uninterrupted_case.cursor,
        model=uninterrupted_case.model,
        recognition=uninterrupted_case.recognition,
        model_optimizer=uninterrupted_case.model_optimizer,
        recognition_optimizer=(uninterrupted_case.recognition_optimizer),
        recognition_forward=uninterrupted_callbacks.recognition_forward,
        objective_forward=uninterrupted_callbacks.objective_forward,
        noise_factory=uninterrupted_callbacks.noise_factory,
        checkpoint_at_batch_end=False,
    )

    first_callbacks = resumed_case.callbacks()
    first = run_h6_training_batch_v3(
        authority=resumed_case.authority,
        cursor=resumed_case.cursor,
        model=resumed_case.model,
        recognition=resumed_case.recognition,
        model_optimizer=resumed_case.model_optimizer,
        recognition_optimizer=resumed_case.recognition_optimizer,
        recognition_forward=first_callbacks.recognition_forward,
        objective_forward=first_callbacks.objective_forward,
        noise_factory=first_callbacks.noise_factory,
        stop_after_phase=TrainingPhase.IMMUTABLE_DETACHED_SNAPSHOT,
        checkpoint_at_batch_end=False,
    )
    assert type(first.snapshot) is H6DetachedBatchRecognitionSnapshotV3
    assert first.cursor.next_phase is TrainingPhase.MODEL_ADAMW

    # A resumed process has no trusted in-memory trajectory sidecar.  It must
    # deterministically reconstruct and byte-authenticate against the exact
    # persisted snapshot before the model objective is evaluated.
    resumed_callbacks = resumed_case.callbacks()
    snapshot_state = first.snapshot.states[0]
    snapshot_tensor_index = next(
        index
        for index, name in enumerate(snapshot_state.names)
        if name.endswith(".mean")
    )
    snapshot_tensor = snapshot_state._tensors[snapshot_tensor_index]
    snapshot_tensor_before = snapshot_tensor.detach().clone()
    model_before_failure = {
        name: tensor.detach().clone()
        for name, tensor in resumed_case.model.state_dict().items()
    }
    with torch.no_grad():
        snapshot_tensor.add_(0.25)
    with pytest.raises(ValueError, match="snapshot identity is stale"):
        run_h6_training_batch_v3(
            authority=resumed_case.authority,
            cursor=first.cursor,
            model=resumed_case.model,
            recognition=resumed_case.recognition,
            model_optimizer=resumed_case.model_optimizer,
            recognition_optimizer=resumed_case.recognition_optimizer,
            recognition_forward=resumed_callbacks.recognition_forward,
            objective_forward=resumed_callbacks.objective_forward,
            noise_factory=resumed_callbacks.noise_factory,
            checkpoint_at_batch_end=False,
            resume_state=first,
        )
    for name, tensor in resumed_case.model.state_dict().items():
        torch.testing.assert_close(
            tensor,
            model_before_failure[name],
            rtol=0,
            atol=0,
        )
    with torch.no_grad():
        snapshot_tensor.copy_(snapshot_tensor_before)

    resumed_callbacks = resumed_case.callbacks()
    authenticated = resumed_callbacks._authenticated_detached_trajectories(
        first.snapshot
    )
    for example_ordinal, (trajectory, binding) in enumerate(
        zip(
            authenticated,
            resumed_case.batch.active_horizons,
            strict=True,
        )
    ):
        assert all(
            not tensor.requires_grad and tensor.grad_fn is None
            for tensor in _trajectory_live_tensors(trajectory)
        )
        projected = _trajectory_live_state_v3(
            trajectory=trajectory,
            active_horizon=binding,
            endpoint_config_sha256=(resumed_case.authority.endpoint_config_sha256),
        )
        persisted = first.snapshot.states[example_ordinal]
        assert projected.names == persisted.names
        for name in projected.names:
            torch.testing.assert_close(
                projected[name],
                persisted.tensor(name),
                rtol=0,
                atol=0,
            )

    resumed = run_h6_training_batch_v3(
        authority=resumed_case.authority,
        cursor=first.cursor,
        model=resumed_case.model,
        recognition=resumed_case.recognition,
        model_optimizer=resumed_case.model_optimizer,
        recognition_optimizer=resumed_case.recognition_optimizer,
        recognition_forward=resumed_callbacks.recognition_forward,
        objective_forward=resumed_callbacks.objective_forward,
        noise_factory=resumed_callbacks.noise_factory,
        checkpoint_at_batch_end=False,
        resume_state=first,
    )

    assert resumed.cursor.batch_index == 1
    assert resumed.cursor.recognition_update_count == 1
    assert resumed.cursor.model_update_count == 1
    assert resumed.phase_records == uninterrupted.phase_records
    assert resumed.metric_records == uninterrupted.metric_records
    _assert_same_module_state(
        resumed_case.model,
        uninterrupted_case.model,
    )
    assert resumed_callbacks.latest_detached_snapshot_sha256 == (
        first.snapshot.snapshot_sha256
    )


@pytest.mark.parametrize(
    ("endpoint_config_id", "interruption_boundary"),
    (
        ("h6-a0-transformer-v2", "terminal"),
        (
            "h6-a5-structured-parent-specific-prefix-exact-complete-"
            "latent-smoothing-v2",
            "post_recognition",
        ),
    ),
)
def test_process_loss_resume_reproduces_terminal_and_history_bytes(
    endpoint_config_id: str,
    interruption_boundary: str,
    _tiny_attempt_authority_v3: tuple[object, object, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vfe4.training.h6_training_attempt_v3 as attempt_v3
    from vfe4.training.h6_execution_v3 import (
        bind_h6_executable_attempt_v3,
    )

    authorities, training_data, runtime = _tiny_attempt_authority_v3
    planned = next(
        attempt
        for attempt in authorities.plan.tuning_attempts  # type: ignore[union-attr]
        if attempt.endpoint_config_id == endpoint_config_id
    )
    executable = bind_h6_executable_attempt_v3(
        authorities=authorities,  # type: ignore[arg-type]
        planned_attempt=planned,
    )
    maximum_bytes = 256 * 1024 * 1024
    uninterrupted_path = (
        tmp_path / f"{planned.endpoint_config_id}.uninterrupted.h6v3"
    ).resolve()
    resumed_path = (tmp_path / f"{planned.endpoint_config_id}.resumed.h6v3").resolve()
    uninterrupted = attempt_v3._execute_new_training_attempt_v3(
        executable=executable,
        training_data=training_data,
        runtime=runtime,
        checkpoint_path=uninterrupted_path,
        maximum_checkpoint_bytes=maximum_bytes,
    )
    uninterrupted_history = attempt_v3.read_h6_training_attempt_history_v3(
        checkpoint_path=uninterrupted_path,
        maximum_bytes=maximum_bytes,
    )

    original_persist = attempt_v3._persist_attempt_boundary_v3
    interrupted = False

    def persist_then_interrupt(**kwargs: object) -> object:
        nonlocal interrupted
        persisted = original_persist(**kwargs)
        if not interrupted and kwargs["boundary_kind"] == interruption_boundary:
            interrupted = True
            raise RuntimeError("simulated process loss")
        return persisted

    monkeypatch.setattr(
        attempt_v3,
        "_persist_attempt_boundary_v3",
        persist_then_interrupt,
    )
    with pytest.raises(RuntimeError, match="simulated process loss"):
        attempt_v3._execute_new_training_attempt_v3(
            executable=executable,
            training_data=training_data,
            runtime=runtime,
            checkpoint_path=resumed_path,
            maximum_checkpoint_bytes=maximum_bytes,
        )
    assert interrupted
    resumed = attempt_v3._execute_new_training_attempt_v3(
        executable=executable,
        training_data=training_data,
        runtime=runtime,
        checkpoint_path=resumed_path,
        maximum_checkpoint_bytes=maximum_bytes,
    )
    resumed_history = attempt_v3.read_h6_training_attempt_history_v3(
        checkpoint_path=resumed_path,
        maximum_bytes=maximum_bytes,
    )
    resumed_progress = attempt_v3.read_h6_training_attempt_progress_v3(
        path=attempt_v3.h6_training_attempt_progress_path_v3(resumed_path),
        maximum_bytes=maximum_bytes,
    )

    assert resumed_progress.latest_boundary.boundary_kind == "terminal"
    assert resumed.checkpoint_bytes_sha256 == (uninterrupted.checkpoint_bytes_sha256)
    assert resumed_path.read_bytes() == uninterrupted_path.read_bytes()
    assert resumed_history == uninterrupted_history
    assert len(resumed_history.validation_boundary_history) == 1
    assert len(resumed_history.metric_history) == (
        1 if endpoint_config_id == "h6-a0-transformer-v2" else 2
    )
    assert resumed.terminal_progress == resumed_progress
    assert resumed.terminal_history == resumed_history
    assert resumed.progress_sha256 == resumed_progress.progress_sha256
    assert resumed.history_sha256 == resumed_history.history_sha256
    assert resumed.metric_history_count == len(resumed_history.metric_history)
    assert resumed.validation_boundary_history_count == len(
        resumed_history.validation_boundary_history
    )

    progress_path = attempt_v3.h6_training_attempt_progress_path_v3(resumed_path)
    if endpoint_config_id == "h6-a0-transformer-v2":
        assert (
            attempt_v3.reopen_h6_terminal_training_attempt_v3(
                executable=executable,
                checkpoint_path=resumed_path,
                maximum_checkpoint_bytes=maximum_bytes,
            )
            == resumed
        )
        progress_raw = progress_path.read_bytes()
        progress_path.write_bytes(progress_raw + b"\ncorrupt")
        try:
            with pytest.raises((RuntimeError, ValueError)):
                attempt_v3.reopen_h6_terminal_training_attempt_v3(
                    executable=executable,
                    checkpoint_path=resumed_path,
                    maximum_checkpoint_bytes=maximum_bytes,
                )
        finally:
            progress_path.write_bytes(progress_raw)

        latest_history_path = (
            resumed_path.parent / resumed_progress.latest_boundary.history_filename
        )
        latest_history_raw = latest_history_path.read_bytes()
        latest_history_path.unlink()
        try:
            with pytest.raises(FileNotFoundError):
                attempt_v3.reopen_h6_terminal_training_attempt_v3(
                    executable=executable,
                    checkpoint_path=resumed_path,
                    maximum_checkpoint_bytes=maximum_bytes,
                )
        finally:
            latest_history_path.write_bytes(latest_history_raw)
    else:
        first_history_path = (
            resumed_path.parent / resumed_progress.boundaries[0].history_filename
        )
        first_history_raw = first_history_path.read_bytes()

        def factory_must_not_run(**_kwargs: object) -> object:
            raise AssertionError(
                "recovery constructed modules before inventory validation"
            )

        monkeypatch.setattr(
            attempt_v3,
            "_fresh_cpu_training_modules_v3",
            factory_must_not_run,
        )
        first_history_path.write_bytes(first_history_raw + b"\ncorrupt")
        try:
            with pytest.raises((RuntimeError, ValueError)):
                attempt_v3.recover_h6_training_attempt_v3(
                    executable=executable,
                    runtime=runtime,
                    checkpoint_path=resumed_path,
                    maximum_checkpoint_bytes=maximum_bytes,
                )
        finally:
            first_history_path.write_bytes(first_history_raw)

        first_history_path.unlink()
        try:
            with pytest.raises(FileNotFoundError):
                attempt_v3.recover_h6_training_attempt_v3(
                    executable=executable,
                    runtime=runtime,
                    checkpoint_path=resumed_path,
                    maximum_checkpoint_bytes=maximum_bytes,
                )
        finally:
            first_history_path.write_bytes(first_history_raw)


def test_training_package_lazily_exports_attempt_recovery_and_history_api() -> None:
    import vfe4.training as training
    import vfe4.training.h6_training_attempt_v3 as attempt_v3

    names = (
        "H6AttemptMetricHistoryRecordV3",
        "H6ValidationBoundaryHistoryRecordV3",
        "H6AttemptRecoveryBoundaryV3",
        "H6TrainingAttemptProgressV3",
        "H6TrainingAttemptHistoryV3",
        "H6RecoveredTrainingAttemptV3",
        "h6_training_attempt_progress_path_v3",
        "read_h6_training_attempt_progress_v3",
        "read_h6_training_attempt_history_v3",
        "recover_h6_training_attempt_v3",
        "reopen_h6_terminal_training_attempt_v3",
        "execute_h6_training_attempt_v3",
    )
    for name in names:
        assert name in training.__all__
        assert getattr(training, name) is getattr(attempt_v3, name)
