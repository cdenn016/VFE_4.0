from __future__ import annotations

import copy
import hashlib
from types import SimpleNamespace

import pytest
import torch

from vfe4.types.training import default_wt103_arm_specs


class _ScalarModule(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = torch.nn.Parameter(
            torch.tensor(value, dtype=torch.float64)
        )


class _ScientificCounter:
    def __init__(self) -> None:
        self.value = 0

    def capture_state(self) -> object:
        return self.value

    def restore_state(self, state: object) -> None:
        assert type(state) is int
        self.value = state

    def state_sha256(self) -> str:
        return hashlib.sha256(str(self.value).encode("ascii")).hexdigest()


class _AttemptSink:
    def __init__(self) -> None:
        self.steps: list[tuple[int, int, object]] = []
        self.validations: list[tuple[int, int]] = []
        self.failures: list[tuple[int, int, object, object]] = []

    def record_step(
        self,
        *,
        step_index: int,
        cumulative_counted_targets: int,
        result: object,
    ) -> None:
        self.steps.append(
            (step_index, cumulative_counted_targets, result)
        )

    def validate_target_blind(
        self,
        *,
        step_index: int,
        cumulative_counted_targets: int,
    ) -> None:
        self.validations.append(
            (step_index, cumulative_counted_targets)
        )

    def record_terminal_failure(
        self,
        *,
        step_index: int,
        cumulative_counted_targets: int,
        result: object,
        exception: object,
    ) -> None:
        self.failures.append(
            (
                step_index,
                cumulative_counted_targets,
                result,
                exception,
            )
        )


def _sha_state(value: object) -> str:
    return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()


def test_forward_terms_keep_raw_numerators_and_normalize_the_loss_once() -> None:
    from vfe4.training.engine import ForwardTerms

    raw_nll = torch.tensor(12.0, dtype=torch.float64, requires_grad=True)
    cross_entropy = ForwardTerms.cross_entropy(
        value=raw_nll,
        counted_targets=4,
    )
    assert cross_entropy.detached_values() == {
        "cross_entropy_value": 12.0,
    }
    assert cross_entropy.objective_numerator().item() == -12.0
    assert cross_entropy.objective().item() == -3.0
    assert cross_entropy.loss().item() == 3.0

    raw_emission = torch.tensor(-8.0, dtype=torch.float64)
    one_nat = torch.tensor(1.0, dtype=torch.float64)
    zero = torch.tensor(0.0, dtype=torch.float64)
    complete = ForwardTerms.complete_elbo(
        expected_log_emission=(raw_emission,),
        initial_model_cross_entropy=one_nat,
        initial_state_cross_entropy=one_nat,
        model_source_cross_entropy=(zero,),
        model_transition_cross_entropy=(one_nat,),
        state_source_cross_entropy=(zero,),
        state_transition_cross_entropy=(one_nat,),
        model_source_kl=(zero,),
        state_source_kl=(zero,),
        continuous_recognition_entropy=zero,
        conditional_source_entropy_estimate=zero,
        joint_recognition_entropy_estimate=zero,
        estimator_error_bound=None,
        counted_targets=4,
    )
    assert complete.partition_schema == "wt103-structured-factor-elbo-v1"
    assert complete.objective_numerator().item() == -12.0
    assert complete.objective().item() == -3.0
    complete_values = complete.detached_values()
    assert complete_values["complete_elbo_numerator"] == -12.0
    assert complete_values["model_source_kl[0]"] == 0.0
    assert complete_values["state_source_kl[0]"] == 0.0
    assert "estimator_error_bound" not in complete_values
    assert {
        "initial_model_kl",
        "initial_state_kl",
        "model_transition_kl[0]",
        "state_transition_kl[0]",
        "joint_recognition_entropy",
    }.isdisjoint(complete_values)
    assert complete.complete_elbo_numerator() == -12.0
    assert complete.complete_elbo_value() == -3.0

    partial_zero = torch.tensor(0.0, dtype=torch.float32)
    partial = ForwardTerms.complete_elbo(
        expected_log_emission=(
            torch.tensor(-0.1, dtype=torch.float32),
        ),
        initial_model_cross_entropy=partial_zero,
        initial_state_cross_entropy=partial_zero,
        model_source_cross_entropy=(partial_zero,),
        model_transition_cross_entropy=(partial_zero,),
        state_source_cross_entropy=(partial_zero,),
        state_transition_cross_entropy=(partial_zero,),
        model_source_kl=(partial_zero,),
        state_source_kl=(partial_zero,),
        continuous_recognition_entropy=partial_zero,
        conditional_source_entropy_estimate=partial_zero,
        joint_recognition_entropy_estimate=partial_zero,
        estimator_error_bound=None,
        counted_targets=3,
    )
    partial_numerator = partial.complete_elbo_numerator()
    assert partial_numerator is not None
    assert partial.complete_elbo_value() == partial_numerator / 3
    assert partial.complete_elbo_value() != float(
        partial.objective().detach().cpu().item()
    )


def test_synthetic_smoke_complete_elbo_uses_structured_factors() -> None:
    from vfe4.training.smoke import _SyntheticBatch, _make_compute_terms

    spec = default_wt103_arm_specs()[1]
    model = _ScalarModule(0.4)
    recognition = _ScalarModule(0.2)
    science = SimpleNamespace(
        trace=SimpleNamespace(
            forward_path="language_generative_complete_elbo"
        ),
        totals=SimpleNamespace(nll_per_token=1.0),
        source_entropy=0.5,
    )
    compute_terms = _make_compute_terms(
        spec=spec,
        model=model,
        recognition=recognition,
        science=science,
    )
    terms = compute_terms(
        "recognition_adam_proposal",
        _SyntheticBatch(
            context=torch.tensor(0.75, dtype=torch.float64),
            target=torch.tensor(1.25, dtype=torch.float64),
        ),
        None,
    )

    assert terms.partition_schema == "wt103-structured-factor-elbo-v1"
    values = terms.detached_values()
    assert {
        "initial_model_cross_entropy",
        "initial_state_cross_entropy",
        "model_source_cross_entropy[0]",
        "model_transition_cross_entropy[0]",
        "state_source_cross_entropy[0]",
        "state_transition_cross_entropy[0]",
        "model_source_kl[0]",
        "state_source_kl[0]",
        "continuous_recognition_entropy",
        "conditional_source_entropy_estimate",
        "joint_recognition_entropy_estimate",
    } <= set(values)
    assert "estimator_error_bound" not in values
    assert terms.estimator_error_bound is None
    assert values["joint_recognition_entropy_estimate"] == pytest.approx(
        values["continuous_recognition_entropy"]
        + values["conditional_source_entropy_estimate"]
    )
    assert (
        values["model_source_kl[0]"] + values["state_source_kl[0]"]
        == pytest.approx(
            values["model_source_cross_entropy[0]"]
            + values["state_source_cross_entropy[0]"]
            - values["conditional_source_entropy_estimate"]
        )
    )


def test_a0_executes_one_reverse_mode_model_proposal_without_latent_state() -> None:
    from vfe4.training.engine import (
        ArmExecutionRuntime,
        ForwardTerms,
        train_step,
    )

    arm = default_wt103_arm_specs()[0]
    model = _ScalarModule(2.0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.05,
        betas=(0.9, 0.95),
        eps=1.0e-8,
        weight_decay=0.0,
        foreach=False,
        fused=False,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda _step: 1.0,
    )

    def forward(phase: str, _batch: object, snapshot: object) -> ForwardTerms:
        assert phase == "model_ce_adam_proposal"
        assert snapshot is None
        return ForwardTerms.cross_entropy(
            value=3.0 * (model.value - 1.0).square(),
            counted_targets=3,
        )

    runtime = ArmExecutionRuntime(
        arm_spec=arm,
        model=model,
        recognition=None,
        model_optimizer=optimizer,
        recognition_optimizer=None,
        model_scheduler=scheduler,
        recognition_scheduler=None,
        grad_scaler=None,
        compute_terms=forward,
        support_validator=lambda: True,
        spd_validator=lambda: True,
        damping_observer=lambda: False,
        projection_observer=lambda: False,
        state_participants=(),
        gradient_clip_norm=1.0,
    )
    before = model.value.detach().clone()
    result = train_step(runtime, batch=object())

    assert result.accepted is True
    assert result.phase_order == ("model_ce_adam_proposal",)
    assert result.reverse_mode_autograd is True
    assert result.monotonicity_claim is False
    assert result.complete_elbo_value is None
    assert result.counted_targets == 3
    assert len(result.updates) == 1
    assert result.updates[0].expected_autograd_scope == "m_step"
    assert result.updates[0].update_label == "adam_proposal"
    control = result.update_controls[0]
    assert control.pre_clip_inf_norm == pytest.approx(2.0)
    assert control.post_clip_inf_norm == pytest.approx(1.0)
    assert not torch.equal(before, model.value.detach())


def test_latent_complete_elbo_runs_e_snapshot_m_with_exact_term_sum() -> None:
    from vfe4.training.engine import (
        ArmExecutionRuntime,
        ForwardTerms,
        RecognitionSnapshot,
        train_step,
    )

    arm = default_wt103_arm_specs()[1]
    model = _ScalarModule(0.4)
    recognition = _ScalarModule(0.2)
    model_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.01,
        foreach=False,
        fused=False,
    )
    recognition_optimizer = torch.optim.AdamW(
        recognition.parameters(),
        lr=0.01,
        foreach=False,
        fused=False,
    )
    observed: list[tuple[str, RecognitionSnapshot | None]] = []
    execution_events: list[str] = []

    def run_event(name: str, operation):
        execution_events.append(name)
        return operation()

    def forward(
        phase: str,
        _batch: object,
        snapshot: RecognitionSnapshot | None,
    ) -> ForwardTerms:
        observed.append((phase, snapshot))
        if phase == "recognition_adam_proposal":
            assert recognition.value.requires_grad is True
            assert model.value.requires_grad is False
            phi = recognition.value
        else:
            assert snapshot is not None
            assert model.value.requires_grad is True
            assert recognition.value.requires_grad is False
            snapshot.assert_nonaliasing(recognition)
            phi = snapshot.tensor("value")
        zero = phi.square() * 0.0
        continuous_entropy = 0.1 * phi
        return ForwardTerms.complete_elbo(
            expected_log_emission=(-(model.value - phi).square(),),
            initial_model_cross_entropy=0.05 * model.value.square(),
            initial_state_cross_entropy=0.25 * phi.square(),
            model_source_cross_entropy=(0.10 * phi.square(),),
            model_transition_cross_entropy=(0.20 * phi.square(),),
            state_source_cross_entropy=(0.30 * phi.square(),),
            state_transition_cross_entropy=(0.40 * phi.square(),),
            model_source_kl=(0.10 * phi.square(),),
            state_source_kl=(0.30 * phi.square(),),
            continuous_recognition_entropy=continuous_entropy,
            conditional_source_entropy_estimate=zero,
            joint_recognition_entropy_estimate=continuous_entropy,
            estimator_error_bound=None,
            counted_targets=5,
        )

    runtime = ArmExecutionRuntime(
        arm_spec=arm,
        model=model,
        recognition=recognition,
        model_optimizer=model_optimizer,
        recognition_optimizer=recognition_optimizer,
        model_scheduler=None,
        recognition_scheduler=None,
        grad_scaler=None,
        compute_terms=forward,
        support_validator=lambda: True,
        spd_validator=lambda: True,
        damping_observer=lambda: False,
        projection_observer=lambda: False,
        state_participants=(),
        gradient_clip_norm=10.0,
        execution_event_runner=run_event,
    )
    result = train_step(runtime, batch=object())

    assert result.accepted is True
    assert result.phase_order == arm.update_phases
    assert result.snapshot_sha256 is not None
    assert result.complete_elbo_numerator == pytest.approx(
        result.objective_terms["complete_elbo_numerator"]
    )
    assert result.complete_elbo_value == pytest.approx(
        result.complete_elbo_numerator / result.counted_targets
    )
    assert tuple(name for name, _ in observed) == (
        "recognition_adam_proposal",
        "recognition_adam_proposal",
        "model_adam_proposal",
        "model_adam_proposal",
    )
    assert observed[0][1] is None
    assert observed[1][1] is None
    assert observed[2][1] is not None
    assert observed[3][1] is not None
    assert tuple(execution_events) == (
        "recognition_forward",
        "recognition_backward",
        "recognition_adamw",
        "recognition_forward",
        "immutable_detached_snapshot",
        "complete_elbo",
        "model_backward",
        "model_adamw",
        "complete_elbo",
    )
    assert tuple(
        update.expected_autograd_scope for update in result.updates
    ) == ("e_step", "m_step")
    assert tuple(
        update.observed_autograd_scope for update in result.updates
    ) == ("e_step", "m_step")
    assert result.expected_autograd_scope == "e_and_m"
    assert result.observed_autograd_scope == "e_and_m"
    assert len(result.proposal_evidence) == 2
    assert all(
        evidence.objective_before_value is not None
        and evidence.objective_after_value is not None
        and evidence.rollback_applied is False
        for evidence in result.proposal_evidence
    )
    assert len(result.update_controls) == 2
    assert all(
        control.adamw_foreach is False
        for control in result.update_controls
    )


def test_rejected_proposal_restores_parameters_optimizer_scheduler_rng_and_counter() -> None:
    from vfe4.training.engine import (
        ArmExecutionRuntime,
        ForwardTerms,
        train_step,
    )

    arm = default_wt103_arm_specs()[0]
    model = _ScalarModule(2.0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.05,
        foreach=False,
        fused=False,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1,
        gamma=0.5,
    )

    def forward(_phase: str, _batch: object, _snapshot: object) -> ForwardTerms:
        torch.rand(4)
        return ForwardTerms.cross_entropy(
            value=model.value.square(),
            counted_targets=1,
        )

    runtime = ArmExecutionRuntime(
        arm_spec=arm,
        model=model,
        recognition=None,
        model_optimizer=optimizer,
        recognition_optimizer=None,
        model_scheduler=scheduler,
        recognition_scheduler=None,
        grad_scaler=None,
        compute_terms=forward,
        support_validator=lambda: True,
        spd_validator=lambda: False,
        damping_observer=lambda: False,
        projection_observer=lambda: False,
        state_participants=(),
        gradient_clip_norm=1.0,
    )
    parameter_before = model.value.detach().clone()
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    scheduler_before = copy.deepcopy(scheduler.state_dict())
    rng_before = torch.get_rng_state().clone()
    counter_before = runtime.update_counter

    result = train_step(runtime, batch=object())

    assert result.accepted is False
    assert result.failure_kind == "spd_validation_failed"
    assert torch.equal(model.value.detach(), parameter_before)
    assert _sha_state(optimizer.state_dict()) == _sha_state(optimizer_before)
    assert _sha_state(scheduler.state_dict()) == _sha_state(scheduler_before)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert runtime.update_counter == counter_before
    assert result.updates[0].accepted is False
    assert result.updates[0].rejection_reason == "spd_validation_failed"


def test_emission_ablation_is_explicitly_not_reported_as_complete_elbo() -> None:
    from vfe4.training.engine import (
        ArmExecutionRuntime,
        ForwardTerms,
        train_step,
    )

    arm = default_wt103_arm_specs()[3]
    model = _ScalarModule(0.4)
    recognition = _ScalarModule(0.2)

    def forward(phase: str, _batch: object, snapshot: object) -> ForwardTerms:
        phi = recognition.value if snapshot is None else snapshot.tensor("value")
        return ForwardTerms.emission_only(
            expected_log_emission=(-(model.value - phi).square(),),
            counted_targets=2,
        )

    runtime = ArmExecutionRuntime(
        arm_spec=arm,
        model=model,
        recognition=recognition,
        model_optimizer=torch.optim.AdamW(
            model.parameters(),
            lr=0.01,
            foreach=False,
            fused=False,
        ),
        recognition_optimizer=torch.optim.AdamW(
            recognition.parameters(),
            lr=0.01,
            foreach=False,
            fused=False,
        ),
        model_scheduler=None,
        recognition_scheduler=None,
        grad_scaler=None,
        compute_terms=forward,
        support_validator=lambda: True,
        spd_validator=lambda: True,
        damping_observer=lambda: False,
        projection_observer=lambda: False,
        state_participants=(),
        gradient_clip_norm=1.0,
    )
    result = train_step(runtime, batch=object())

    assert result.accepted is True
    assert result.complete_elbo_value is None
    assert result.objective_kind == "emission_only_ablation_non_elbo"


def test_nonfinite_forward_rejects_before_scientific_state_can_advance() -> None:
    from vfe4.training.engine import (
        ArmExecutionRuntime,
        ForwardTerms,
        train_step,
    )

    arm = default_wt103_arm_specs()[4]
    model = _ScalarModule(0.4)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.01,
        foreach=False,
        fused=False,
    )

    def forward(_phase: str, _batch: object, _snapshot: object) -> ForwardTerms:
        return ForwardTerms.cross_entropy(
            value=model.value * torch.tensor(float("nan")),
            counted_targets=1,
        )

    runtime = ArmExecutionRuntime(
        arm_spec=arm,
        model=model,
        recognition=None,
        model_optimizer=optimizer,
        recognition_optimizer=None,
        model_scheduler=None,
        recognition_scheduler=None,
        grad_scaler=None,
        compute_terms=forward,
        support_validator=lambda: True,
        spd_validator=lambda: True,
        damping_observer=lambda: False,
        projection_observer=lambda: False,
        state_participants=(),
        gradient_clip_norm=1.0,
    )
    before = model.value.detach().clone()
    result = train_step(runtime, batch=object())

    assert result.accepted is False
    assert result.failure_kind == "nonfinite_objective"
    assert torch.equal(model.value.detach(), before)
    assert result.objective_diagnostics_applicable is False
    assert result.objective_terms is None
    assert result.complete_elbo_value is None
    assert result.counted_targets is None


def test_callback_exception_restores_state_without_fabricating_objective() -> None:
    from vfe4.training.engine import ArmExecutionRuntime, train_step

    arm = default_wt103_arm_specs()[0]
    model = _ScalarModule(2.0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.05,
        foreach=False,
        fused=False,
    )

    def forward(_phase: str, _batch: object, _snapshot: object):
        with torch.no_grad():
            model.value.add_(7.0)
        torch.rand(3)
        raise TypeError("injected callback failure")

    runtime = ArmExecutionRuntime(
        arm_spec=arm,
        model=model,
        recognition=None,
        model_optimizer=optimizer,
        recognition_optimizer=None,
        model_scheduler=None,
        recognition_scheduler=None,
        grad_scaler=None,
        compute_terms=forward,
        support_validator=lambda: True,
        spd_validator=lambda: True,
        damping_observer=lambda: False,
        projection_observer=lambda: False,
        state_participants=(),
        gradient_clip_norm=1.0,
    )
    parameter_before = model.value.detach().clone()
    rng_before = torch.get_rng_state().clone()

    result = train_step(runtime, batch=object())

    assert result.accepted is False
    assert result.failure_kind == "proposal_exception:TypeError"
    assert torch.equal(model.value.detach(), parameter_before)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert result.objective_diagnostics_applicable is False
    assert result.objective_terms is None
    assert result.counted_targets is None


def test_optimizer_access_mismatch_is_rejected_before_forward() -> None:
    from vfe4.training.engine import (
        ArmExecutionRuntime,
        ForwardTerms,
        TrainingEngineError,
        train_step,
    )

    arm = default_wt103_arm_specs()[0]
    model = torch.nn.Sequential(_ScalarModule(1.0), _ScalarModule(2.0))
    optimizer = torch.optim.AdamW(
        (model[0].value,),
        lr=0.01,
        foreach=False,
        fused=False,
    )
    called = False

    def forward(_phase: str, _batch: object, _snapshot: object) -> ForwardTerms:
        nonlocal called
        called = True
        return ForwardTerms.cross_entropy(
            value=model[0].value.square() + model[1].value.square(),
            counted_targets=1,
        )

    runtime = ArmExecutionRuntime(
        arm_spec=arm,
        model=model,
        recognition=None,
        model_optimizer=optimizer,
        recognition_optimizer=None,
        model_scheduler=None,
        recognition_scheduler=None,
        grad_scaler=None,
        compute_terms=forward,
        support_validator=lambda: True,
        spd_validator=lambda: True,
        damping_observer=lambda: False,
        projection_observer=lambda: False,
        state_participants=(),
        gradient_clip_norm=1.0,
    )

    with pytest.raises(TrainingEngineError, match="optimizer access"):
        train_step(runtime, batch=object())
    assert called is False


def test_inactive_parameter_mutation_rejects_and_restores_exact_proposal() -> None:
    from vfe4.training.engine import (
        ArmExecutionRuntime,
        ForwardTerms,
        train_step,
    )

    arm = default_wt103_arm_specs()[1]
    model = _ScalarModule(0.4)
    recognition = _ScalarModule(0.2)
    model_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.01,
        foreach=False,
        fused=False,
    )
    recognition_optimizer = torch.optim.AdamW(
        recognition.parameters(),
        lr=0.01,
        foreach=False,
        fused=False,
    )

    def forward(phase: str, _batch: object, snapshot: object) -> ForwardTerms:
        if phase == "recognition_adam_proposal":
            with torch.no_grad():
                model.value.add_(1.0)
            phi = recognition.value
        else:
            phi = snapshot.tensor("value")
        zero = phi.square() * 0.0
        continuous_entropy = 0.1 * phi
        return ForwardTerms.complete_elbo(
            expected_log_emission=(-(model.value - phi).square(),),
            initial_model_cross_entropy=0.05 * model.value.square(),
            initial_state_cross_entropy=0.25 * phi.square(),
            model_source_cross_entropy=(0.10 * phi.square(),),
            model_transition_cross_entropy=(0.20 * phi.square(),),
            state_source_cross_entropy=(0.30 * phi.square(),),
            state_transition_cross_entropy=(0.40 * phi.square(),),
            model_source_kl=(0.10 * phi.square(),),
            state_source_kl=(0.30 * phi.square(),),
            continuous_recognition_entropy=continuous_entropy,
            conditional_source_entropy_estimate=zero,
            joint_recognition_entropy_estimate=continuous_entropy,
            estimator_error_bound=None,
            counted_targets=1,
        )

    runtime = ArmExecutionRuntime(
        arm_spec=arm,
        model=model,
        recognition=recognition,
        model_optimizer=model_optimizer,
        recognition_optimizer=recognition_optimizer,
        model_scheduler=None,
        recognition_scheduler=None,
        grad_scaler=None,
        compute_terms=forward,
        support_validator=lambda: True,
        spd_validator=lambda: True,
        damping_observer=lambda: False,
        projection_observer=lambda: False,
        state_participants=(),
        gradient_clip_norm=1.0,
    )
    model_before = model.value.detach().clone()
    recognition_before = recognition.value.detach().clone()

    result = train_step(runtime, batch=object())

    assert result.accepted is False
    assert result.failure_kind == "inactive_parameter_mutation"
    assert torch.equal(model.value.detach(), model_before)
    assert torch.equal(recognition.value.detach(), recognition_before)


def test_rejection_restores_registered_buffers_and_scientific_participants() -> None:
    from vfe4.training.engine import (
        ArmExecutionRuntime,
        ForwardTerms,
        train_step,
    )

    arm = default_wt103_arm_specs()[0]
    model = _ScalarModule(2.0)
    model.register_buffer(
        "scientific_counter",
        torch.tensor(0, dtype=torch.int64),
    )
    participant = _ScientificCounter()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.01,
        foreach=False,
        fused=False,
    )

    def forward(_phase: str, _batch: object, _snapshot: object) -> ForwardTerms:
        participant.value += 1
        return ForwardTerms.cross_entropy(
            value=model.value.square(),
            counted_targets=1,
        )

    runtime = ArmExecutionRuntime(
        arm_spec=arm,
        model=model,
        recognition=None,
        model_optimizer=optimizer,
        recognition_optimizer=None,
        model_scheduler=None,
        recognition_scheduler=None,
        grad_scaler=None,
        compute_terms=forward,
        support_validator=lambda: True,
        spd_validator=lambda: False,
        damping_observer=lambda: False,
        projection_observer=lambda: False,
        state_participants=(participant,),
        gradient_clip_norm=1.0,
    )

    result = train_step(runtime, batch=object())

    assert result.failure_kind == "spd_validation_failed"
    assert participant.value == 0
    assert model.scientific_counter.item() == 0
    assert result.proposal_evidence[0].rollback_applied is True


def test_forward_buffer_mutation_is_rejected_and_restored() -> None:
    from vfe4.training.engine import (
        ArmExecutionRuntime,
        ForwardTerms,
        train_step,
    )

    arm = default_wt103_arm_specs()[0]
    model = _ScalarModule(2.0)
    model.register_buffer(
        "scientific_counter",
        torch.tensor(0, dtype=torch.int64),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.01,
        foreach=False,
        fused=False,
    )

    def forward(_phase: str, _batch: object, _snapshot: object) -> ForwardTerms:
        model.scientific_counter.add_(1)
        return ForwardTerms.cross_entropy(
            value=model.value.square(),
            counted_targets=1,
        )

    runtime = ArmExecutionRuntime(
        arm_spec=arm,
        model=model,
        recognition=None,
        model_optimizer=optimizer,
        recognition_optimizer=None,
        model_scheduler=None,
        recognition_scheduler=None,
        grad_scaler=None,
        compute_terms=forward,
        support_validator=lambda: True,
        spd_validator=lambda: True,
        damping_observer=lambda: False,
        projection_observer=lambda: False,
        state_participants=(),
        gradient_clip_norm=1.0,
    )

    result = train_step(runtime, batch=object())

    assert result.failure_kind == "module_buffer_mutation"
    assert model.scientific_counter.item() == 0


def test_complete_elbo_rejects_missing_or_misaligned_partitions() -> None:
    from vfe4.training.engine import ForwardTerms, TrainingEngineError

    scalar = torch.tensor(0.0, requires_grad=True)
    with pytest.raises(TrainingEngineError, match="horizon partitions"):
        ForwardTerms.complete_elbo(
            expected_log_emission=(scalar, scalar),
            initial_model_cross_entropy=scalar,
            initial_state_cross_entropy=scalar,
            model_source_cross_entropy=(scalar,),
            model_transition_cross_entropy=(scalar, scalar),
            state_source_cross_entropy=(scalar, scalar),
            state_transition_cross_entropy=(scalar, scalar),
            model_source_kl=(scalar,),
            state_source_kl=(scalar, scalar),
            continuous_recognition_entropy=scalar,
            conditional_source_entropy_estimate=scalar,
            joint_recognition_entropy_estimate=scalar,
            estimator_error_bound=None,
            counted_targets=2,
        )


def test_complete_elbo_rejects_inconsistent_source_kl_diagnostics() -> None:
    from vfe4.training.engine import ForwardTerms, TrainingEngineError

    zero = torch.tensor(0.0, dtype=torch.float64)
    one = torch.tensor(1.0, dtype=torch.float64)
    two = torch.tensor(2.0, dtype=torch.float64)
    three = torch.tensor(3.0, dtype=torch.float64)
    with pytest.raises(
        TrainingEngineError,
        match="source KL diagnostics",
    ):
        ForwardTerms.complete_elbo(
            expected_log_emission=(-one,),
            initial_model_cross_entropy=zero,
            initial_state_cross_entropy=zero,
            model_source_cross_entropy=(two,),
            model_transition_cross_entropy=(zero,),
            state_source_cross_entropy=(three,),
            state_transition_cross_entropy=(zero,),
            model_source_kl=(one,),
            state_source_kl=(two,),
            continuous_recognition_entropy=one,
            conditional_source_entropy_estimate=one,
            joint_recognition_entropy_estimate=two,
            estimator_error_bound=None,
            counted_targets=1,
        )


def test_complete_elbo_rejects_fabricated_finite_estimator_bound() -> None:
    from vfe4.training.engine import ForwardTerms, TrainingEngineError

    zero = torch.tensor(0.0, dtype=torch.float64)
    with pytest.raises(TrainingEngineError, match="not applicable"):
        ForwardTerms.complete_elbo(
            expected_log_emission=(zero,),
            initial_model_cross_entropy=zero,
            initial_state_cross_entropy=zero,
            model_source_cross_entropy=(zero,),
            model_transition_cross_entropy=(zero,),
            state_source_cross_entropy=(zero,),
            state_transition_cross_entropy=(zero,),
            model_source_kl=(zero,),
            state_source_kl=(zero,),
            continuous_recognition_entropy=zero,
            conditional_source_entropy_estimate=zero,
            joint_recognition_entropy_estimate=zero,
            estimator_error_bound=zero,
            counted_targets=1,
        )


def test_attempt_accumulates_targets_records_steps_and_validates_boundaries() -> None:
    from vfe4.training.engine import (
        ArmExecutionRuntime,
        AttemptResult,
        ForwardTerms,
        train_attempt,
    )

    arm = default_wt103_arm_specs()[0]
    model = _ScalarModule(2.0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.01,
        foreach=False,
        fused=False,
    )

    def forward(_phase: str, _batch: object, _snapshot: object) -> ForwardTerms:
        return ForwardTerms.cross_entropy(
            value=model.value.square(),
            counted_targets=3,
        )

    runtime = ArmExecutionRuntime(
        arm_spec=arm,
        model=model,
        recognition=None,
        model_optimizer=optimizer,
        recognition_optimizer=None,
        model_scheduler=None,
        recognition_scheduler=None,
        grad_scaler=None,
        compute_terms=forward,
        support_validator=lambda: True,
        spd_validator=lambda: True,
        damping_observer=lambda: False,
        projection_observer=lambda: False,
        state_participants=(),
        gradient_clip_norm=1.0,
    )
    sink = _AttemptSink()

    result = train_attempt(
        runtime,
        batches=(object(), object()),
        validation_step_boundaries=(1, 2),
        event_sink=sink,
    )

    assert type(result) is AttemptResult
    assert result.cumulative_counted_targets == 6
    assert result.completed_validation_step_boundaries == (1, 2)
    assert tuple((row[0], row[1]) for row in sink.steps) == (
        (1, 3),
        (2, 6),
    )
    assert sink.validations == [(1, 3), (2, 6)]
    assert sink.failures == []


def test_attempt_routes_rejection_to_terminal_failure_sink() -> None:
    from vfe4.training.engine import (
        ArmExecutionRuntime,
        ForwardTerms,
        train_attempt,
    )

    arm = default_wt103_arm_specs()[0]
    model = _ScalarModule(2.0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.01,
        foreach=False,
        fused=False,
    )

    def forward(_phase: str, _batch: object, _snapshot: object) -> ForwardTerms:
        return ForwardTerms.cross_entropy(
            value=model.value.square(),
            counted_targets=3,
        )

    runtime = ArmExecutionRuntime(
        arm_spec=arm,
        model=model,
        recognition=None,
        model_optimizer=optimizer,
        recognition_optimizer=None,
        model_scheduler=None,
        recognition_scheduler=None,
        grad_scaler=None,
        compute_terms=forward,
        support_validator=lambda: False,
        spd_validator=lambda: True,
        damping_observer=lambda: False,
        projection_observer=lambda: False,
        state_participants=(),
        gradient_clip_norm=1.0,
    )
    sink = _AttemptSink()

    result = train_attempt(
        runtime,
        batches=(object(), object()),
        validation_step_boundaries=(1, 2),
        event_sink=sink,
    )

    assert len(result.steps) == 1
    assert result.cumulative_counted_targets == 0
    assert result.completed_validation_step_boundaries == ()
    assert result.terminal_failure_recorded is True
    assert len(sink.failures) == 1
    assert sink.failures[0][2] is result.steps[0]
    assert sink.failures[0][3] is None
