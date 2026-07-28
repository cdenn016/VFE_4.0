from __future__ import annotations

import dataclasses

import pytest
import torch
from torch import nn

from vfe4.training import h6_engine_v3 as engine_v3
from vfe4.training.h6_engine_v3 import (
    H6DetachedRecognitionSnapshotV3,
    H6EngineAuthorityV3,
    H6LiveObjectiveTermV3,
    H6PhaseObjectiveV3,
    canonical_engine_state_bytes_v3,
    run_h6_training_batch_v3,
)
from vfe4.training.matching import H6_ADAMW_POLICY
from vfe4.types.h6 import TrainingPhase
from vfe4.types.h6_prediction_v3 import H6AttemptCursorV3


def _sha(character: str) -> str:
    return character * 64


class _ScalarModule(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(value, dtype=torch.float64))


def _authority(
    *,
    objective_kind: str = "complete_elbo",
    latent_enabled: bool = True,
    receiver_count: int = 2,
    state_categorical_enabled: bool = True,
    model_categorical_enabled: bool = False,
) -> H6EngineAuthorityV3:
    return H6EngineAuthorityV3.create(
        attempt_spec_sha256=_sha("1"),
        endpoint_config_sha256=_sha("2"),
        readiness_sha256=_sha("3"),
        readiness_matching_set_sha256=_sha("4"),
        matching_set_sha256=_sha("4"),
        matching_policy_sha256=_sha("f"),
        readiness_training_schedule_sha256=_sha("5"),
        training_schedule_sha256=_sha("5"),
        readiness_runtime_identity_sha256=_sha("6"),
        runtime_identity_sha256=_sha("6"),
        planned_attempt_sha256=_sha("b"),
        endpoint_config_id="synthetic-latent",
        matching_ledger_sha256=_sha("c"),
        matching_report_sha256s=(_sha("d"),),
        receiver_count=receiver_count,
        state_categorical_enabled=state_categorical_enabled,
        model_categorical_enabled=model_categorical_enabled,
        tuning_cell_sha256=_sha("e"),
        optimizer_policy_sha256=H6_ADAMW_POLICY.optimizer_policy_sha256,
        optimizer_learning_rate=0.1,
        optimizer_weight_decay=0.0,
        objective_kind=objective_kind,
        latent_enabled=latent_enabled,
    )


def _cursor(authority: H6EngineAuthorityV3) -> H6AttemptCursorV3:
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
        counter_consumption_sha256=_sha("7"),
        permutation_sha256=_sha("8"),
    )


def _latent_case(
    *,
    objective_kind: str = "complete_elbo",
) -> tuple[
    H6EngineAuthorityV3,
    _ScalarModule,
    _ScalarModule,
    torch.optim.AdamW,
    torch.optim.AdamW,
    list[tuple[TrainingPhase, tuple[str, ...] | None]],
    list[TrainingPhase],
    object,
    object,
    object,
]:
    authority = _authority(objective_kind=objective_kind)
    model = _ScalarModule(0.25)
    recognition = _ScalarModule(-0.5)
    model_optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.1, weight_decay=0.0, foreach=False, fused=False
    )
    recognition_optimizer = torch.optim.AdamW(
        recognition.parameters(),
        lr=0.1,
        weight_decay=0.0,
        foreach=False,
        fused=False,
    )
    calls: list[tuple[TrainingPhase, tuple[str, ...] | None]] = []
    noise_phases: list[TrainingPhase] = []

    def noise(phase: TrainingPhase, _: H6AttemptCursorV3) -> tuple[torch.Tensor, str]:
        noise_phases.append(phase)
        value = 1.0 if phase is TrainingPhase.RECOGNITION_ADAMW else 2.0
        return torch.tensor([value], dtype=torch.float64), _sha(
            "9" if value == 1.0 else "a"
        )

    def recognition_forward() -> engine_v3.H6LiveRecognitionStateV3:
        return engine_v3.H6LiveRecognitionStateV3.create(
            endpoint_config_sha256=authority.endpoint_config_sha256,
            receiver_count=authority.receiver_count,
            state_categorical_enabled=authority.state_categorical_enabled,
            model_categorical_enabled=authority.model_categorical_enabled,
            state_categorical_supports=((0,),),
            model_categorical_supports=(None,),
            receiver_components=((0, ("initial",)), (1, ("main",))),
            tensors={
                "receiver.0.component.initial.mean": (
                    torch.zeros(1, dtype=torch.float64)
                    + recognition.weight.reshape(1) * 0.0
                ),
                "receiver.0.shared_precision_cholesky": (
                    torch.ones((1, 1), dtype=torch.float64) + recognition.weight * 0.0
                ),
                "receiver.1.component.main.mean": (recognition.weight.reshape(1) * 1.0),
                "receiver.1.shared_precision_cholesky": (
                    torch.ones((1, 1), dtype=torch.float64) + recognition.weight * 0.0
                ),
                "state.receiver.1.support": torch.tensor([0], dtype=torch.int64),
                "state.receiver.1.categorical_row": torch.softmax(
                    recognition.weight.reshape(1), dim=0
                ),
                "model.absent.support": torch.tensor([-1], dtype=torch.int64),
                "model.absent.categorical_row": torch.ones(1, dtype=torch.float64),
            },
            context_sha256=_sha("b"),
            recognition_state_sha256=_sha("c"),
            source_model_sha256=_sha("d"),
            law_sha256=_sha("e"),
        )

    def objective(
        *,
        phase: TrainingPhase,
        recognition_state: object,
        allowed_partitions: tuple[str, ...] | None,
        **_: object,
    ) -> H6PhaseObjectiveV3:
        calls.append((phase, allowed_partitions))
        if phase is TrainingPhase.RECOGNITION_ADAMW:
            target = -(recognition.weight - model.weight.detach()).square()
        else:
            assert isinstance(recognition_state, H6DetachedRecognitionSnapshotV3)
            target = -(
                model.weight
                - recognition_state.tensor("receiver.1.component.main.mean").squeeze(0)
            ).square()
        emission = H6LiveObjectiveTermV3.create(
            partition="emission", receiver_t=1, value=target
        )
        if objective_kind == "emission_only_ablation_non_elbo":
            return H6PhaseObjectiveV3.emission_only((emission,))
        source = H6LiveObjectiveTermV3.create(
            partition="state_source",
            receiver_t=1,
            value=target * 0.0,
        )
        return H6PhaseObjectiveV3.complete_elbo((source, emission))

    return (
        authority,
        model,
        recognition,
        model_optimizer,
        recognition_optimizer,
        calls,
        noise_phases,
        noise,
        recognition_forward,
        objective,
    )


def _run_latent(
    case: tuple[object, ...],
    *,
    cursor: H6AttemptCursorV3 | None = None,
    stop_after_phase: TrainingPhase | None = None,
    resume_state: object | None = None,
    gradient_clip_max_norm: float | None = None,
    declared_checkpoint_phases: tuple[TrainingPhase, ...] = (
        TrainingPhase.RECOGNITION_ADAMW,
    ),
):
    (
        authority,
        model,
        recognition,
        model_optimizer,
        recognition_optimizer,
        _,
        _,
        noise,
        recognition_forward,
        objective,
    ) = case
    assert isinstance(authority, H6EngineAuthorityV3)
    values = dict(
        authority=authority,
        cursor=_cursor(authority) if cursor is None else cursor,
        model=model,
        recognition=recognition,
        model_optimizer=model_optimizer,
        recognition_optimizer=recognition_optimizer,
        recognition_forward=recognition_forward,
        objective_forward=objective,
        noise_factory=noise,
        stop_after_phase=stop_after_phase,
        declared_checkpoint_phases=declared_checkpoint_phases,
    )
    if resume_state is not None:
        values["resume_state"] = resume_state
    if gradient_clip_max_norm is not None:
        values["gradient_clip_max_norm"] = gradient_clip_max_norm
    return run_h6_training_batch_v3(**values)


def test_recognition_step_cannot_mutate_model_parameters() -> None:
    case = _latent_case()
    model = case[1]
    before = model.weight.detach().clone()

    result = _run_latent(case, stop_after_phase=TrainingPhase.RECOGNITION_ADAMW)

    assert torch.equal(model.weight.detach(), before)
    assert model.weight.grad is None
    assert result.cursor.next_phase is TrainingPhase.IMMUTABLE_DETACHED_SNAPSHOT
    assert result.recognition_update_count == 1
    assert result.model_update_count == 0
    assert result.gradient_clip_count == 1


def test_snapshot_is_fresh_post_step_complete_and_detached() -> None:
    case = _latent_case()
    recognition = case[2]
    before = recognition.weight.detach().clone()

    result = _run_latent(case)

    assert not torch.equal(recognition.weight.detach(), before)
    assert result.snapshot is not None
    assert result.snapshot.names == (
        "model.absent.categorical_row",
        "model.absent.support",
        "receiver.0.component.initial.mean",
        "receiver.0.shared_precision_cholesky",
        "receiver.1.component.main.mean",
        "receiver.1.shared_precision_cholesky",
        "state.receiver.1.categorical_row",
        "state.receiver.1.support",
    )
    assert torch.equal(
        result.snapshot.tensor("receiver.1.component.main.mean"),
        recognition.weight.detach().reshape(1),
    )
    assert all(
        not result.snapshot.tensor(name).requires_grad for name in result.snapshot.names
    )
    mutated = result.snapshot.tensor("receiver.1.component.main.mean")
    mutated.add_(100.0)
    assert not torch.equal(
        mutated,
        result.snapshot.tensor("receiver.1.component.main.mean"),
    )


def test_model_step_cannot_mutate_recognition_parameters() -> None:
    case = _latent_case()
    recognition = case[2]

    partial = _run_latent(case, stop_after_phase=TrainingPhase.RECOGNITION_ADAMW)
    after_recognition = recognition.weight.detach().clone()
    completed = _run_latent(
        case,
        cursor=partial.cursor,
        resume_state=partial,
    )

    assert torch.equal(recognition.weight.detach(), after_recognition)
    assert recognition.weight.grad is None
    assert completed.recognition_update_count == 1
    assert completed.model_update_count == 1
    assert completed.gradient_clip_count == 2


def test_emission_only_endpoint_optimizes_only_live_emission_and_is_not_elbo() -> None:
    case = _latent_case(objective_kind="emission_only_ablation_non_elbo")

    result = _run_latent(case)

    calls = case[5]
    assert calls == [
        (TrainingPhase.RECOGNITION_ADAMW, ("emission",)),
        (TrainingPhase.MODEL_ADAMW, ("emission",)),
    ]
    assert tuple(record.is_elbo for record in result.phase_records) == (
        False,
        False,
    )
    assert all(record.partitions == ("emission",) for record in result.phase_records)


def test_a0_no_latent_has_one_ce_nll_model_phase() -> None:
    authority = _authority(
        objective_kind="cross_entropy",
        latent_enabled=False,
        state_categorical_enabled=False,
    )
    model = _ScalarModule(0.25)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.1, weight_decay=0.0, foreach=False, fused=False
    )
    calls: list[TrainingPhase] = []

    def objective(**values: object) -> H6PhaseObjectiveV3:
        phase = values["phase"]
        assert isinstance(phase, TrainingPhase)
        calls.append(phase)
        term = H6LiveObjectiveTermV3.create(
            partition="emission",
            receiver_t=1,
            value=-(model.weight - 1.0).square(),
        )
        return H6PhaseObjectiveV3.cross_entropy(term)

    result = run_h6_training_batch_v3(
        authority=authority,
        cursor=_cursor(authority),
        model=model,
        recognition=None,
        model_optimizer=optimizer,
        recognition_optimizer=None,
        recognition_forward=None,
        objective_forward=objective,
        noise_factory=lambda phase, cursor: (
            torch.ones(1, dtype=torch.float64),
            _sha("9"),
        ),
    )

    assert calls == [TrainingPhase.MODEL_CE_ADAMW]
    assert result.model_update_count == 1
    assert result.recognition_update_count == 0
    assert result.snapshot is None


def test_tiny_cpu_resume_matches_uninterrupted_terminal_bytes() -> None:
    uninterrupted = _latent_case()
    resumed = _latent_case()

    full = _run_latent(uninterrupted)
    partial = _run_latent(resumed, stop_after_phase=TrainingPhase.RECOGNITION_ADAMW)
    resumed_result = _run_latent(
        resumed,
        cursor=partial.cursor,
        resume_state=partial,
    )

    assert full.cursor == resumed_result.cursor
    assert full.phase_records == resumed_result.phase_records
    assert full.metric_records == resumed_result.metric_records
    assert full.snapshot == resumed_result.snapshot
    assert canonical_engine_state_bytes_v3(
        model=uninterrupted[1],
        recognition=uninterrupted[2],
        model_optimizer=uninterrupted[3],
        recognition_optimizer=uninterrupted[4],
        result=full,
    ) == canonical_engine_state_bytes_v3(
        model=resumed[1],
        recognition=resumed[2],
        model_optimizer=resumed[3],
        recognition_optimizer=resumed[4],
        result=resumed_result,
    )


@pytest.mark.parametrize(
    ("field", "mutated"),
    (
        ("lr", 0.2),
        ("betas", (0.8, 0.999)),
        ("eps", 1.0e-7),
        ("weight_decay", 0.01),
        ("amsgrad", True),
        ("maximize", True),
        ("foreach", True),
        ("capturable", True),
        ("differentiable", True),
        ("fused", True),
    ),
)
def test_train_refuses_any_unbound_adamw_policy_before_forward_effect(
    field: str,
    mutated: object,
) -> None:
    case = _latent_case()
    authority = case[0]
    calls = case[5]
    noise_phases = case[6]
    model_before = case[1].weight.detach().clone()
    recognition_before = case[2].weight.detach().clone()

    bound = H6EngineAuthorityV3.create(
        attempt_spec_sha256=authority.attempt_spec_sha256,
        endpoint_config_sha256=authority.endpoint_config_sha256,
        readiness_sha256=authority.readiness_sha256,
        readiness_matching_set_sha256=authority.matching_set_sha256,
        matching_set_sha256=authority.matching_set_sha256,
        matching_policy_sha256=authority.matching_policy_sha256,
        readiness_training_schedule_sha256=authority.training_schedule_sha256,
        training_schedule_sha256=authority.training_schedule_sha256,
        readiness_runtime_identity_sha256=authority.runtime_identity_sha256,
        runtime_identity_sha256=authority.runtime_identity_sha256,
        objective_kind=authority.objective_kind,
        latent_enabled=authority.latent_enabled,
        planned_attempt_sha256=_sha("b"),
        endpoint_config_id="synthetic-latent",
        matching_ledger_sha256=_sha("c"),
        matching_report_sha256s=(_sha("d"),),
        receiver_count=authority.receiver_count,
        state_categorical_enabled=authority.state_categorical_enabled,
        model_categorical_enabled=authority.model_categorical_enabled,
        tuning_cell_sha256=_sha("e"),
        optimizer_policy_sha256=authority.optimizer_policy_sha256,
        optimizer_learning_rate=0.1,
        optimizer_weight_decay=0.0,
    )
    case[3].param_groups[0][field] = mutated

    with pytest.raises(ValueError, match="AdamW.*policy|policy.*AdamW"):
        _run_latent((bound, *case[1:]))

    assert calls == []
    assert noise_phases == []
    assert torch.equal(case[1].weight.detach(), model_before)
    assert torch.equal(case[2].weight.detach(), recognition_before)


def test_train_has_no_caller_gradient_clip_override() -> None:
    case = _latent_case()

    with pytest.raises(TypeError, match="gradient_clip_max_norm"):
        _run_latent(case, gradient_clip_max_norm=0.5)

    assert case[5] == []
    assert case[6] == []


def test_snapshot_requires_complete_inventory_and_exact_phase_binding() -> None:
    authority = _authority(receiver_count=3)
    post_recognition_cursor = H6AttemptCursorV3.create(
        attempt_spec_sha256=authority.attempt_spec_sha256,
        pass_index=0,
        batch_index=0,
        next_phase=TrainingPhase.MODEL_ADAMW,
        example_ordinal=0,
        draw_block=1,
        counter_consumption_sha256=_sha("9"),
        permutation_sha256=_sha("8"),
        recognition_update_count=1,
    )
    live_state_type = engine_v3.H6LiveRecognitionStateV3
    complete = live_state_type.create(
        endpoint_config_sha256=authority.endpoint_config_sha256,
        receiver_count=3,
        state_categorical_enabled=True,
        model_categorical_enabled=False,
        receiver_components=(
            (0, ("base",)),
            (1, ("main",)),
            (2, ("state0", "state1")),
        ),
        state_categorical_supports=((0,), (0, 1)),
        model_categorical_supports=(None,),
        tensors={
            "receiver.0.component.base.mean": torch.tensor([0.0], dtype=torch.float64),
            "receiver.0.shared_precision_cholesky": torch.tensor(
                [[1.0]], dtype=torch.float64
            ),
            "receiver.1.component.main.mean": torch.tensor([0.5], dtype=torch.float64),
            "receiver.1.shared_precision_cholesky": torch.tensor(
                [[1.0]], dtype=torch.float64
            ),
            "receiver.2.component.state0.mean": torch.tensor(
                [0.25], dtype=torch.float64
            ),
            "receiver.2.component.state1.mean": torch.tensor(
                [0.75], dtype=torch.float64
            ),
            "receiver.2.shared_precision_cholesky": torch.tensor(
                [[1.0]], dtype=torch.float64
            ),
            "state.receiver.1.support": torch.tensor([0], dtype=torch.int64),
            "state.receiver.1.categorical_row": torch.tensor(
                [1.0], dtype=torch.float64
            ),
            "state.receiver.2.support": torch.tensor([0, 1], dtype=torch.int64),
            "state.receiver.2.categorical_row": torch.tensor(
                [0.25, 0.75], dtype=torch.float64
            ),
            "model.absent.support": torch.tensor([-1], dtype=torch.int64),
            "model.absent.categorical_row": torch.tensor([1.0], dtype=torch.float64),
        },
        context_sha256=_sha("b"),
        recognition_state_sha256=_sha("c"),
        source_model_sha256=_sha("d"),
        law_sha256=_sha("e"),
    )
    snapshot = H6DetachedRecognitionSnapshotV3.capture(
        complete,
        authority=authority,
        post_recognition_cursor=post_recognition_cursor,
    )

    assert snapshot.receiver_components == complete.receiver_components
    assert snapshot.receiver_count == 3
    assert snapshot.state_categorical_supports == ((0,), (0, 1))
    assert snapshot.model_categorical_supports == (None,)
    assert snapshot.attempt_spec_sha256 == authority.attempt_spec_sha256
    assert (
        snapshot.post_recognition_cursor_sha256 == post_recognition_cursor.cursor_sha256
    )
    assert snapshot.context_sha256 == _sha("b")
    assert snapshot.recognition_state_sha256 == _sha("c")
    assert snapshot.source_model_sha256 == _sha("d")
    assert snapshot.law_sha256 == _sha("e")

    with pytest.raises(ValueError, match="inventory|precision|categorical|support"):
        live_state_type.create(
            endpoint_config_sha256=authority.endpoint_config_sha256,
            receiver_count=3,
            state_categorical_enabled=True,
            model_categorical_enabled=False,
            receiver_components=((0, ("base",)), (1, ("main",))),
            state_categorical_supports=((0,), (0, 1)),
            model_categorical_supports=(None,),
            tensors={
                "receiver.0.component.base.mean": torch.tensor(
                    [0.0], dtype=torch.float64
                ),
                "receiver.0.shared_precision_cholesky": torch.tensor(
                    [[1.0]], dtype=torch.float64
                ),
                "receiver.1.component.main.mean": torch.tensor(
                    [0.5], dtype=torch.float64
                ),
                "receiver.1.shared_precision_cholesky": torch.tensor(
                    [[1.0]], dtype=torch.float64
                ),
                "state.receiver.1.support": torch.tensor([0], dtype=torch.int64),
                "state.receiver.1.categorical_row": torch.tensor(
                    [1.0], dtype=torch.float64
                ),
                "state.receiver.2.support": torch.tensor([0, 1], dtype=torch.int64),
                "state.receiver.2.categorical_row": torch.tensor(
                    [0.25, 0.75], dtype=torch.float64
                ),
                "model.absent.support": torch.tensor([-1], dtype=torch.int64),
                "model.absent.categorical_row": torch.tensor(
                    [1.0], dtype=torch.float64
                ),
            },
            context_sha256=_sha("b"),
            recognition_state_sha256=_sha("c"),
            source_model_sha256=_sha("d"),
            law_sha256=_sha("e"),
        )


def test_model_phase_requires_exact_persisted_snapshot_and_prior_records() -> None:
    case = _latent_case()
    recognition_done = _run_latent(
        case,
        stop_after_phase=TrainingPhase.RECOGNITION_ADAMW,
    )
    snapshot_done = _run_latent(
        case,
        cursor=recognition_done.cursor,
        resume_state=recognition_done,
        stop_after_phase=TrainingPhase.IMMUTABLE_DETACHED_SNAPSHOT,
        declared_checkpoint_phases=(
            TrainingPhase.RECOGNITION_ADAMW,
            TrainingPhase.IMMUTABLE_DETACHED_SNAPSHOT,
        ),
    )

    assert snapshot_done.snapshot is not None
    with pytest.raises(
        ValueError,
        match="persisted.*resume|resume.*snapshot|prior.*records",
    ):
        _run_latent(case, cursor=snapshot_done.cursor)

    completed = _run_latent(
        case,
        cursor=snapshot_done.cursor,
        resume_state=snapshot_done,
    )
    assert len(completed.phase_records) == 2
    assert len(completed.metric_records) == 2


def test_train_refuses_readiness_or_matching_identity_drift() -> None:
    case = _latent_case()
    authority = case[0]

    with pytest.raises(ValueError, match="readiness.*matching|matching.*readiness"):
        _run_latent(
            (
                dataclasses.replace(
                    authority,
                    readiness_matching_set_sha256=_sha("f"),
                ),
                *case[1:],
            )
        )
    with pytest.raises(ValueError, match="schedule.*identity|identity.*schedule"):
        _run_latent(
            (
                dataclasses.replace(
                    authority,
                    training_schedule_sha256=_sha("f"),
                ),
                *case[1:],
            )
        )
