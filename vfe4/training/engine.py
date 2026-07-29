"""Typed reverse-mode WT103 training phases with exact proposal rollback."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import math
import random
import struct
from collections.abc import Callable, Iterable, Mapping
from typing import Literal, Protocol

import numpy as np
import torch

from vfe4.recording.metrics import UpdateControlRecord
from vfe4.types.training import (
    WT103ArmSpec,
    WT103UpdateRecord,
    owned_sha256,
)


ObjectiveKind = Literal[
    "cross_entropy",
    "complete_elbo",
    "emission_only_ablation_non_elbo",
]
WT103_STRUCTURED_FACTOR_ELBO_SCHEMA = "wt103-structured-factor-elbo-v1"
WT103_STRUCTURED_FACTOR_ELBO_SCHEMA_SHA256 = owned_sha256(
    "vfe4.wt103.structured-factor-elbo-schema.v1",
    {
        "schema_version": WT103_STRUCTURED_FACTOR_ELBO_SCHEMA,
        "objective_reconstruction": (
            "sum(expected_log_emission)"
            "-initial_model_cross_entropy"
            "-initial_state_cross_entropy"
            "-sum(model_source_cross_entropy)"
            "-sum(model_transition_cross_entropy)"
            "-sum(state_source_cross_entropy)"
            "-sum(state_transition_cross_entropy)"
            "+joint_recognition_entropy_estimate"
        ),
        "joint_entropy_chain_rule": (
            "continuous_recognition_entropy"
            "+conditional_source_entropy_estimate"
        ),
        "derived_diagnostics": (
            "model_source_kl",
            "state_source_kl",
        ),
        "estimator_error_bound": (
            "not_applicable_only:"
            "no_preregistered_finite_bound_for_single_sample_mc"
        ),
        "h5_kl_partition_schema": "reference_only_not_reused",
    },
)


class TrainingEngineError(RuntimeError):
    """The typed training contract was malformed before a proposal."""


class _ProposalRejected(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _scalar_tensor(value: object, *, name: str) -> torch.Tensor:
    if (
        type(value) is not torch.Tensor
        or value.numel() != 1
        or not value.dtype.is_floating_point
    ):
        raise TrainingEngineError(
            f"{name} must be one exact floating scalar tensor"
        )
    return value


def _same_float32_reduction(left: torch.Tensor, right: torch.Tensor) -> bool:
    left_value = float(left.detach().cpu().item())
    right_value = float(right.detach().cpu().item())
    scale = max(abs(left_value), abs(right_value), 1.0)
    return math.isclose(
        left_value,
        right_value,
        rel_tol=0.0,
        abs_tol=8.0 * scale * 2.0**-23,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class ForwardTerms:
    """One complete forward pass under an explicit WT103 factor schema."""

    objective_kind: ObjectiveKind
    partition_schema: (
        Literal["wt103-structured-factor-elbo-v1"] | None
    )
    counted_targets: int
    cross_entropy_value: torch.Tensor | None
    expected_log_emission: tuple[torch.Tensor, ...] | None
    initial_model_cross_entropy: torch.Tensor | None
    initial_state_cross_entropy: torch.Tensor | None
    model_source_cross_entropy: tuple[torch.Tensor, ...] | None
    model_transition_cross_entropy: tuple[torch.Tensor, ...] | None
    state_source_cross_entropy: tuple[torch.Tensor, ...] | None
    state_transition_cross_entropy: tuple[torch.Tensor, ...] | None
    model_source_kl: tuple[torch.Tensor, ...] | None
    state_source_kl: tuple[torch.Tensor, ...] | None
    continuous_recognition_entropy: torch.Tensor | None
    conditional_source_entropy_estimate: torch.Tensor | None
    joint_recognition_entropy_estimate: torch.Tensor | None
    estimator_error_bound: torch.Tensor | None

    @staticmethod
    def _scalar_tuple(
        value: object,
        *,
        name: str,
    ) -> tuple[torch.Tensor, ...]:
        if type(value) is not tuple or not value:
            raise TrainingEngineError(
                f"{name} must be a nonempty exact tensor tuple"
            )
        for index, tensor in enumerate(value):
            _scalar_tensor(tensor, name=f"{name}[{index}]")
        return value

    def __post_init__(self) -> None:
        if type(self.counted_targets) is not int or self.counted_targets <= 0:
            raise TrainingEngineError("counted_targets must be positive")
        if self.objective_kind == "cross_entropy":
            if self.partition_schema is not None:
                raise TrainingEngineError(
                    "cross entropy cannot claim an ELBO factor schema"
                )
            _scalar_tensor(
                self.cross_entropy_value,
                name="cross_entropy_value",
            )
            if any(
                value is not None
                for value in (
                    self.expected_log_emission,
                    self.initial_model_cross_entropy,
                    self.initial_state_cross_entropy,
                    self.model_source_cross_entropy,
                    self.model_transition_cross_entropy,
                    self.state_source_cross_entropy,
                    self.state_transition_cross_entropy,
                    self.model_source_kl,
                    self.state_source_kl,
                    self.continuous_recognition_entropy,
                    self.conditional_source_entropy_estimate,
                    self.joint_recognition_entropy_estimate,
                    self.estimator_error_bound,
                )
            ):
                raise TrainingEngineError(
                    "cross entropy cannot carry ELBO partitions"
                )
            return
        if self.objective_kind == "emission_only_ablation_non_elbo":
            if self.partition_schema is not None:
                raise TrainingEngineError(
                    "emission-only cannot claim an ELBO factor schema"
                )
            emissions = self._scalar_tuple(
                self.expected_log_emission,
                name="expected_log_emission",
            )
            if self.cross_entropy_value is not None or any(
                value is not None
                for value in (
                    self.initial_model_cross_entropy,
                    self.initial_state_cross_entropy,
                    self.model_source_cross_entropy,
                    self.model_transition_cross_entropy,
                    self.state_source_cross_entropy,
                    self.state_transition_cross_entropy,
                    self.model_source_kl,
                    self.state_source_kl,
                    self.continuous_recognition_entropy,
                    self.conditional_source_entropy_estimate,
                    self.joint_recognition_entropy_estimate,
                    self.estimator_error_bound,
                )
            ):
                raise TrainingEngineError(
                    "emission-only ablation cannot carry ELBO partitions"
                )
            if not emissions:
                raise TrainingEngineError(
                    "emission-only objective requires emissions"
                )
            return
        if self.objective_kind != "complete_elbo":
            raise TrainingEngineError("unknown objective kind")
        if self.partition_schema != WT103_STRUCTURED_FACTOR_ELBO_SCHEMA:
            raise TrainingEngineError(
                "complete ELBO requires the structured-factor schema"
            )
        if self.cross_entropy_value is not None:
            raise TrainingEngineError(
                "complete ELBO cannot carry cross entropy"
            )
        emissions = self._scalar_tuple(
            self.expected_log_emission,
            name="expected_log_emission",
        )
        partitions = tuple(
            self._scalar_tuple(getattr(self, name), name=name)
            for name in (
                "model_source_cross_entropy",
                "model_transition_cross_entropy",
                "state_source_cross_entropy",
                "state_transition_cross_entropy",
                "model_source_kl",
                "state_source_kl",
            )
        )
        if any(len(value) != len(emissions) for value in partitions):
            raise TrainingEngineError(
                "ELBO horizon partitions must have identical lengths"
            )
        for name in (
            "initial_model_cross_entropy",
            "initial_state_cross_entropy",
            "continuous_recognition_entropy",
            "conditional_source_entropy_estimate",
            "joint_recognition_entropy_estimate",
        ):
            _scalar_tensor(getattr(self, name), name=name)
        assert self.conditional_source_entropy_estimate is not None
        assert self.continuous_recognition_entropy is not None
        assert self.joint_recognition_entropy_estimate is not None
        if (
            float(
                self.conditional_source_entropy_estimate.detach()
                .cpu()
                .item()
            )
            < 0.0
        ):
            raise TrainingEngineError(
                "conditional source entropy estimate must be nonnegative"
            )
        expected_joint = (
            self.continuous_recognition_entropy
            + self.conditional_source_entropy_estimate
        )
        if not _same_float32_reduction(
            expected_joint,
            self.joint_recognition_entropy_estimate,
        ):
            raise TrainingEngineError(
                "joint recognition entropy estimate changed chain-rule sum"
            )
        assert self.model_source_cross_entropy is not None
        assert self.state_source_cross_entropy is not None
        assert self.model_source_kl is not None
        assert self.state_source_kl is not None
        source_cross_entropy = sum(
            (
                *self.model_source_cross_entropy,
                *self.state_source_cross_entropy,
            )
        )
        source_kl = sum((*self.model_source_kl, *self.state_source_kl))
        expected_source_kl = (
            source_cross_entropy
            - self.conditional_source_entropy_estimate
        )
        if not _same_float32_reduction(source_kl, expected_source_kl):
            raise TrainingEngineError(
                "source KL diagnostics changed "
                "cross-entropy-minus-entropy"
            )
        if self.estimator_error_bound is not None:
            raise TrainingEngineError(
                "structured-factor estimator_error_bound is not applicable"
            )

    @classmethod
    def cross_entropy(
        cls,
        *,
        value: torch.Tensor,
        counted_targets: int,
    ) -> "ForwardTerms":
        return cls(
            objective_kind="cross_entropy",
            partition_schema=None,
            counted_targets=counted_targets,
            cross_entropy_value=value,
            expected_log_emission=None,
            initial_model_cross_entropy=None,
            initial_state_cross_entropy=None,
            model_source_cross_entropy=None,
            model_transition_cross_entropy=None,
            state_source_cross_entropy=None,
            state_transition_cross_entropy=None,
            model_source_kl=None,
            state_source_kl=None,
            continuous_recognition_entropy=None,
            conditional_source_entropy_estimate=None,
            joint_recognition_entropy_estimate=None,
            estimator_error_bound=None,
        )

    @classmethod
    def complete_elbo(
        cls,
        *,
        expected_log_emission: tuple[torch.Tensor, ...],
        initial_model_cross_entropy: torch.Tensor,
        initial_state_cross_entropy: torch.Tensor,
        model_source_cross_entropy: tuple[torch.Tensor, ...],
        model_transition_cross_entropy: tuple[torch.Tensor, ...],
        state_source_cross_entropy: tuple[torch.Tensor, ...],
        state_transition_cross_entropy: tuple[torch.Tensor, ...],
        model_source_kl: tuple[torch.Tensor, ...],
        state_source_kl: tuple[torch.Tensor, ...],
        continuous_recognition_entropy: torch.Tensor,
        conditional_source_entropy_estimate: torch.Tensor,
        joint_recognition_entropy_estimate: torch.Tensor,
        estimator_error_bound: torch.Tensor | None,
        counted_targets: int,
    ) -> "ForwardTerms":
        return cls(
            objective_kind="complete_elbo",
            partition_schema=WT103_STRUCTURED_FACTOR_ELBO_SCHEMA,
            counted_targets=counted_targets,
            cross_entropy_value=None,
            expected_log_emission=expected_log_emission,
            initial_model_cross_entropy=initial_model_cross_entropy,
            initial_state_cross_entropy=initial_state_cross_entropy,
            model_source_cross_entropy=model_source_cross_entropy,
            model_transition_cross_entropy=(
                model_transition_cross_entropy
            ),
            state_source_cross_entropy=state_source_cross_entropy,
            state_transition_cross_entropy=(
                state_transition_cross_entropy
            ),
            model_source_kl=model_source_kl,
            state_source_kl=state_source_kl,
            continuous_recognition_entropy=(
                continuous_recognition_entropy
            ),
            conditional_source_entropy_estimate=(
                conditional_source_entropy_estimate
            ),
            joint_recognition_entropy_estimate=(
                joint_recognition_entropy_estimate
            ),
            estimator_error_bound=estimator_error_bound,
        )

    @classmethod
    def emission_only(
        cls,
        *,
        expected_log_emission: tuple[torch.Tensor, ...],
        counted_targets: int,
    ) -> "ForwardTerms":
        return cls(
            objective_kind="emission_only_ablation_non_elbo",
            partition_schema=None,
            counted_targets=counted_targets,
            cross_entropy_value=None,
            expected_log_emission=expected_log_emission,
            initial_model_cross_entropy=None,
            initial_state_cross_entropy=None,
            model_source_cross_entropy=None,
            model_transition_cross_entropy=None,
            state_source_cross_entropy=None,
            state_transition_cross_entropy=None,
            model_source_kl=None,
            state_source_kl=None,
            continuous_recognition_entropy=None,
            conditional_source_entropy_estimate=None,
            joint_recognition_entropy_estimate=None,
            estimator_error_bound=None,
        )

    def objective_numerator(self) -> torch.Tensor:
        """Return the raw summed objective in nats before token normalization."""

        self.__post_init__()
        if self.objective_kind == "cross_entropy":
            assert self.cross_entropy_value is not None
            return -self.cross_entropy_value
        assert self.expected_log_emission is not None
        if self.objective_kind == "emission_only_ablation_non_elbo":
            return sum(self.expected_log_emission)
        assert self.initial_model_cross_entropy is not None
        assert self.initial_state_cross_entropy is not None
        assert self.model_source_cross_entropy is not None
        assert self.model_transition_cross_entropy is not None
        assert self.state_source_cross_entropy is not None
        assert self.state_transition_cross_entropy is not None
        assert self.model_source_kl is not None
        assert self.state_source_kl is not None
        assert self.joint_recognition_entropy_estimate is not None
        return (
            sum(self.expected_log_emission)
            - self.initial_model_cross_entropy
            - self.initial_state_cross_entropy
            - sum(self.model_source_cross_entropy)
            - sum(self.model_transition_cross_entropy)
            - sum(self.state_source_cross_entropy)
            - sum(self.state_transition_cross_entropy)
            + self.joint_recognition_entropy_estimate
        )

    def objective(self) -> torch.Tensor:
        """Return the single counted-target-normalized optimization objective."""

        return self.objective_numerator() / self.counted_targets

    def loss(self) -> torch.Tensor:
        objective = self.objective()
        return -objective

    def detached_values(self) -> dict[str, float]:
        self.__post_init__()
        if self.objective_kind == "cross_entropy":
            assert self.cross_entropy_value is not None
            return {
                "cross_entropy_value": float(
                    self.cross_entropy_value.detach().cpu().item()
                )
            }
        assert self.expected_log_emission is not None
        values = {
            f"expected_log_emission[{index}]": float(
                tensor.detach().cpu().item()
            )
            for index, tensor in enumerate(self.expected_log_emission)
        }
        if self.objective_kind == "emission_only_ablation_non_elbo":
            values["emission_only_non_elbo"] = float(
                self.objective_numerator().detach().cpu().item()
            )
            return values
        for name in (
            "model_source_cross_entropy",
            "model_transition_cross_entropy",
            "state_source_cross_entropy",
            "state_transition_cross_entropy",
            "model_source_kl",
            "state_source_kl",
        ):
            tensors = getattr(self, name)
            assert tensors is not None
            for index, tensor in enumerate(tensors):
                values[f"{name}[{index}]"] = float(
                    tensor.detach().cpu().item()
                )
        for name in (
            "initial_model_cross_entropy",
            "initial_state_cross_entropy",
            "continuous_recognition_entropy",
            "conditional_source_entropy_estimate",
            "joint_recognition_entropy_estimate",
        ):
            tensor = getattr(self, name)
            assert tensor is not None
            values[name] = float(tensor.detach().cpu().item())
        if self.estimator_error_bound is not None:
            values["estimator_error_bound"] = float(
                self.estimator_error_bound.detach().cpu().item()
            )
        values["complete_elbo_numerator"] = float(
            self.objective_numerator().detach().cpu().item()
        )
        return values

    def complete_elbo_numerator(self) -> float | None:
        if self.objective_kind != "complete_elbo":
            return None
        return float(
            self.objective_numerator().detach().cpu().item()
        )

    def complete_elbo_value(self) -> float | None:
        if self.objective_kind != "complete_elbo":
            return None
        numerator = self.complete_elbo_numerator()
        assert numerator is not None
        return numerator / self.counted_targets


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    owned = tensor.detach().to(device="cpu").contiguous()
    try:
        return owned.numpy().tobytes(order="C")
    except (RuntimeError, TypeError):
        return owned.view(torch.uint8).numpy().tobytes(order="C")


def _tensor_identity(name: str, tensor: torch.Tensor) -> tuple[object, ...]:
    return (
        name,
        str(tensor.dtype),
        tuple(int(item) for item in tensor.shape),
        hashlib.sha256(_tensor_bytes(tensor)).hexdigest(),
    )


@dataclasses.dataclass(frozen=True, slots=True)
class RecognitionSnapshot:
    """Detached, cloned, hash-bound recognition parameters."""

    tensors: tuple[tuple[str, torch.Tensor], ...]
    snapshot_sha256: str

    @classmethod
    def capture(cls, recognition: torch.nn.Module) -> "RecognitionSnapshot":
        if not isinstance(recognition, torch.nn.Module):
            raise TrainingEngineError("recognition must be a torch module")
        rows = tuple(
            (
                name,
                parameter.detach().clone(
                    memory_format=torch.contiguous_format
                ),
            )
            for name, parameter in recognition.named_parameters()
        )
        if not rows:
            raise TrainingEngineError(
                "recognition snapshot cannot be empty"
            )
        for _, tensor in rows:
            tensor.requires_grad_(False)
        digest = owned_sha256(
            "vfe4.wt103.recognition-snapshot.v1",
            tuple(_tensor_identity(name, tensor) for name, tensor in rows),
        )
        snapshot = cls(rows, digest)
        snapshot.assert_nonaliasing(recognition)
        snapshot.assert_intact()
        return snapshot

    def assert_intact(self) -> None:
        if (
            type(self.tensors) is not tuple
            or not self.tensors
            or len({name for name, _ in self.tensors}) != len(self.tensors)
        ):
            raise TrainingEngineError("recognition snapshot inventory is invalid")
        for name, tensor in self.tensors:
            if (
                type(name) is not str
                or not name
                or type(tensor) is not torch.Tensor
                or tensor.requires_grad
            ):
                raise TrainingEngineError(
                    "recognition snapshot tensors must be detached exact tensors"
                )
        observed = owned_sha256(
            "vfe4.wt103.recognition-snapshot.v1",
            tuple(
                _tensor_identity(name, tensor)
                for name, tensor in self.tensors
            ),
        )
        if observed != self.snapshot_sha256:
            raise TrainingEngineError("recognition snapshot was mutated")

    def assert_nonaliasing(self, recognition: torch.nn.Module) -> None:
        self.assert_intact()
        source = dict(recognition.named_parameters())
        if tuple(source) != tuple(name for name, _ in self.tensors):
            raise TrainingEngineError(
                "recognition snapshot inventory no longer matches"
            )
        for name, tensor in self.tensors:
            if tensor.data_ptr() == source[name].data_ptr():
                raise TrainingEngineError(
                    "recognition snapshot aliases a live parameter"
                )

    def tensor(self, name: str) -> torch.Tensor:
        self.assert_intact()
        for observed_name, tensor in self.tensors:
            if observed_name == name:
                return tensor.clone(memory_format=torch.contiguous_format)
        raise KeyError(name)


class ForwardCallback(Protocol):
    def __call__(
        self,
        phase: str,
        batch: object,
        recognition_snapshot: RecognitionSnapshot | None,
    ) -> ForwardTerms: ...


class ExecutionEventRunner(Protocol):
    """Optional observer seam that wraps an actual engine operation."""

    def __call__(
        self,
        event_name: str,
        operation: Callable[[], object],
    ) -> object: ...


class ScientificStateParticipant(Protocol):
    """Explicit non-module scientific state participating in rollback."""

    def capture_state(self) -> object: ...

    def restore_state(self, state: object) -> None: ...

    def state_sha256(self) -> str: ...


def _optimizer_parameters(
    optimizer: torch.optim.Optimizer,
) -> tuple[torch.nn.Parameter, ...]:
    parameters: list[torch.nn.Parameter] = []
    for group in optimizer.param_groups:
        raw_parameters = group.get("params")
        if type(raw_parameters) is not list:
            raise TrainingEngineError(
                "optimizer parameter groups must expose concrete lists"
            )
        for parameter in raw_parameters:
            if not isinstance(parameter, torch.nn.Parameter):
                raise TrainingEngineError(
                    "optimizer access contains a non-Parameter value"
                )
            parameters.append(parameter)
    return tuple(parameters)


def _validate_optimizer_access(
    optimizer: torch.optim.Optimizer,
    expected: tuple[torch.nn.Parameter, ...],
    *,
    name: str,
) -> None:
    if len(optimizer.param_groups) != 1:
        raise TrainingEngineError(
            f"{name} optimizer access requires one exact parameter group"
        )
    observed = _optimizer_parameters(optimizer)
    observed_ids = tuple(id(parameter) for parameter in observed)
    expected_ids = tuple(id(parameter) for parameter in expected)
    if (
        len(set(observed_ids)) != len(observed_ids)
        or set(observed_ids) != set(expected_ids)
    ):
        raise TrainingEngineError(
            f"{name} optimizer access does not match its active block"
        )
    group = optimizer.param_groups[0]
    for flag in (
        "amsgrad",
        "maximize",
        "capturable",
        "differentiable",
        "foreach",
        "fused",
    ):
        if type(group.get(flag)) is not bool:
            raise TrainingEngineError(
                f"{name} optimizer {flag} must be an explicit bool"
            )


def _validate_scheduler_binding(
    scheduler: object | None,
    optimizer: torch.optim.Optimizer,
    *,
    name: str,
) -> None:
    if scheduler is None:
        return
    if getattr(scheduler, "optimizer", None) is not optimizer:
        raise TrainingEngineError(
            f"{name} scheduler is not bound to its optimizer"
        )
    for operation in ("state_dict", "load_state_dict", "step"):
        if not callable(getattr(scheduler, operation, None)):
            raise TrainingEngineError(
                f"{name} scheduler lacks {operation}"
            )


@dataclasses.dataclass(slots=True)
class ArmExecutionRuntime:
    arm_spec: WT103ArmSpec
    model: torch.nn.Module
    recognition: torch.nn.Module | None
    model_optimizer: torch.optim.Optimizer
    recognition_optimizer: torch.optim.Optimizer | None
    model_scheduler: object | None
    recognition_scheduler: object | None
    grad_scaler: object | None
    compute_terms: ForwardCallback
    support_validator: Callable[[], bool]
    spd_validator: Callable[[], bool]
    damping_observer: Callable[[], bool]
    projection_observer: Callable[[], bool]
    state_participants: tuple[ScientificStateParticipant, ...]
    gradient_clip_norm: float
    update_counter: int = 0
    execution_event_runner: ExecutionEventRunner | None = None

    def validate(self) -> None:
        if type(self.arm_spec) is not WT103ArmSpec:
            raise TrainingEngineError("arm_spec must be an exact WT103ArmSpec")
        self.arm_spec.__post_init__()
        if not isinstance(self.model, torch.nn.Module):
            raise TrainingEngineError("model must be a torch module")
        if not isinstance(self.model_optimizer, torch.optim.Optimizer):
            raise TrainingEngineError("model optimizer is missing")
        if type(self.model_optimizer) is not torch.optim.AdamW:
            raise TrainingEngineError("model optimizer must be exact AdamW")
        if not callable(self.compute_terms) or any(
            not callable(callback)
            for callback in (
                self.support_validator,
                self.spd_validator,
                self.damping_observer,
                self.projection_observer,
            )
        ):
            raise TrainingEngineError("runtime callbacks must be callable")
        if (
            type(self.state_participants) is not tuple
            or len({id(value) for value in self.state_participants})
            != len(self.state_participants)
        ):
            raise TrainingEngineError(
                "state participants must be a unique exact tuple"
            )
        for participant in self.state_participants:
            for operation in (
                "capture_state",
                "restore_state",
                "state_sha256",
            ):
                if not callable(getattr(participant, operation, None)):
                    raise TrainingEngineError(
                        f"scientific state participant lacks {operation}"
                    )
        if (
            type(self.gradient_clip_norm) is not float
            or not math.isfinite(self.gradient_clip_norm)
            or self.gradient_clip_norm <= 0.0
        ):
            raise TrainingEngineError(
                "gradient_clip_norm must be a positive finite float"
            )
        if type(self.update_counter) is not int or self.update_counter < 0:
            raise TrainingEngineError("update_counter must be nonnegative")
        if (
            self.execution_event_runner is not None
            and not callable(self.execution_event_runner)
        ):
            raise TrainingEngineError(
                "execution_event_runner must be callable or None"
            )
        if self.arm_spec.latent_enabled:
            if (
                not isinstance(self.recognition, torch.nn.Module)
                or not isinstance(
                    self.recognition_optimizer,
                    torch.optim.Optimizer,
                )
            ):
                raise TrainingEngineError(
                    "latent arms require recognition state and optimizer"
                )
            if type(self.recognition_optimizer) is not torch.optim.AdamW:
                raise TrainingEngineError(
                    "recognition optimizer must be exact AdamW"
                )
        elif any(
            item is not None
            for item in (
                self.recognition,
                self.recognition_optimizer,
                self.recognition_scheduler,
            )
        ):
            raise TrainingEngineError(
                "nonlatent arms cannot carry dormant recognition state"
            )
        module_parameters = tuple(self.model.parameters())
        if not module_parameters:
            raise TrainingEngineError("model parameter inventory is empty")
        if self.recognition is not None:
            recognition_parameters = tuple(self.recognition.parameters())
            if not recognition_parameters:
                raise TrainingEngineError(
                    "recognition parameter inventory is empty"
                )
            if {
                id(parameter) for parameter in module_parameters
            } & {
                id(parameter) for parameter in recognition_parameters
            }:
                raise TrainingEngineError(
                    "model and recognition parameter inventories overlap"
                )
        else:
            recognition_parameters = ()
        _validate_optimizer_access(
            self.model_optimizer,
            module_parameters,
            name="model",
        )
        if self.recognition_optimizer is not None:
            _validate_optimizer_access(
                self.recognition_optimizer,
                recognition_parameters,
                name="recognition",
            )
        _validate_scheduler_binding(
            self.model_scheduler,
            self.model_optimizer,
            name="model",
        )
        if self.recognition_optimizer is not None:
            _validate_scheduler_binding(
                self.recognition_scheduler,
                self.recognition_optimizer,
                name="recognition",
            )


PhaseAutogradScope = Literal["e_step", "m_step"]
AttemptAutogradScope = Literal["m_step", "e_and_m"]
ObservedAttemptAutogradScope = Literal[
    "m_step",
    "e_and_m",
    "partial",
    "not_observed",
]


@dataclasses.dataclass(frozen=True, slots=True)
class ProposalEvidence:
    schema_version: Literal["wt103-adam-proposal-evidence-v1"]
    phase: str
    affected_block: Literal["recognition", "model"]
    expected_autograd_scope: PhaseAutogradScope
    observed_autograd_scope: PhaseAutogradScope | None
    objective_before_terms: tuple[tuple[str, float], ...] | None
    objective_after_terms: tuple[tuple[str, float], ...] | None
    objective_before_value: float | None
    objective_after_value: float | None
    counted_targets: int | None
    estimator_error_bound_before: float | None
    estimator_error_bound_after: float | None
    support_valid: bool | None
    spd_valid: bool | None
    damping_applied: bool | None
    projection_applied: bool | None
    rollback_applied: bool
    evidence_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != "wt103-adam-proposal-evidence-v1"
            or type(self.phase) is not str
            or not self.phase
            or self.affected_block not in ("recognition", "model")
            or self.expected_autograd_scope not in ("e_step", "m_step")
            or self.observed_autograd_scope
            not in (None, "e_step", "m_step")
            or type(self.rollback_applied) is not bool
        ):
            raise TrainingEngineError("proposal evidence literals are invalid")
        before_applicable = self.objective_before_terms is not None
        after_applicable = self.objective_after_terms is not None
        for name, rows, value in (
            (
                "before",
                self.objective_before_terms,
                self.objective_before_value,
            ),
            (
                "after",
                self.objective_after_terms,
                self.objective_after_value,
            ),
        ):
            if rows is None:
                if value is not None:
                    raise TrainingEngineError(
                        f"{name} objective cannot fabricate a scalar"
                    )
                continue
            if (
                type(rows) is not tuple
                or not rows
                or len({row[0] for row in rows}) != len(rows)
                or any(
                    type(row) is not tuple
                    or len(row) != 2
                    or type(row[0]) is not str
                    or not row[0]
                    or type(row[1]) is not float
                    or not math.isfinite(row[1])
                    for row in rows
                )
                or type(value) is not float
                or not math.isfinite(value)
            ):
                raise TrainingEngineError(
                    f"{name} objective evidence is invalid"
                )
        if before_applicable != (self.counted_targets is not None):
            raise TrainingEngineError(
                "proposal counted-target applicability is inconsistent"
            )
        if self.counted_targets is not None and (
            type(self.counted_targets) is not int
            or self.counted_targets <= 0
        ):
            raise TrainingEngineError(
                "proposal counted_targets must be positive"
            )
        if after_applicable and not before_applicable:
            raise TrainingEngineError(
                "after evidence cannot exist without before evidence"
            )
        for name in (
            "estimator_error_bound_before",
            "estimator_error_bound_after",
        ):
            value = getattr(self, name)
            if value is not None and (
                type(value) is not float
                or not math.isfinite(value)
                or value < 0.0
            ):
                raise TrainingEngineError(
                    f"{name} must be a nonnegative finite float"
                )
        for name in (
            "support_valid",
            "spd_valid",
            "damping_applied",
            "projection_applied",
        ):
            value = getattr(self, name)
            if value is not None and type(value) is not bool:
                raise TrainingEngineError(f"{name} must be exact bool or None")
        if self.rollback_applied:
            if after_applicable:
                raise TrainingEngineError(
                    "rolled-back proposal cannot publish after-state evidence"
                )
        elif not (
            before_applicable
            and after_applicable
            and self.observed_autograd_scope
            == self.expected_autograd_scope
            and self.support_valid is True
            and self.spd_valid is True
            and type(self.damping_applied) is bool
            and type(self.projection_applied) is bool
        ):
            raise TrainingEngineError(
                "accepted proposal evidence is incomplete"
            )
        expected = owned_sha256(
            "vfe4.wt103.adam-proposal-evidence.v1",
            {
                field.name: getattr(self, field.name)
                for field in dataclasses.fields(self)
                if field.name != "evidence_sha256"
            },
        )
        if self.evidence_sha256 != expected:
            raise TrainingEngineError(
                "proposal evidence hash does not match"
            )

    @classmethod
    def create(cls, **values: object) -> "ProposalEvidence":
        payload = {
            "schema_version": "wt103-adam-proposal-evidence-v1",
            **values,
        }
        return cls(
            **payload,
            evidence_sha256=owned_sha256(
                "vfe4.wt103.adam-proposal-evidence.v1",
                payload,
            ),
        )  # type: ignore[arg-type]


@dataclasses.dataclass(frozen=True, slots=True)
class StepResult:
    arm_id: str
    objective_kind: ObjectiveKind
    phase_order: tuple[str, ...]
    updates: tuple[WT103UpdateRecord, ...]
    update_controls: tuple[UpdateControlRecord, ...]
    proposal_evidence: tuple[ProposalEvidence, ...]
    snapshot_sha256: str | None
    objective_diagnostics_applicable: bool
    objective_terms: dict[str, float] | None
    complete_elbo_numerator: float | None
    complete_elbo_value: float | None
    counted_targets: int | None
    accepted: bool
    failure_kind: str | None
    expected_autograd_scope: AttemptAutogradScope
    observed_autograd_scope: ObservedAttemptAutogradScope
    reverse_mode_autograd: Literal[True]
    monotonicity_claim: Literal[False]

    def __post_init__(self) -> None:
        if type(self.phase_order) is not tuple or not self.phase_order:
            raise TrainingEngineError("step phase_order cannot be empty")
        if type(self.updates) is not tuple or not self.updates:
            raise TrainingEngineError("step must contain an update record")
        if (
            type(self.update_controls) is not tuple
            or len(self.update_controls) != len(self.updates)
            or any(
                type(control) is not UpdateControlRecord
                for control in self.update_controls
            )
        ):
            raise TrainingEngineError(
                "every proposal requires one exact update-control record"
            )
        if (
            type(self.proposal_evidence) is not tuple
            or len(self.proposal_evidence) != len(self.updates)
            or any(
                type(evidence) is not ProposalEvidence
                for evidence in self.proposal_evidence
            )
        ):
            raise TrainingEngineError(
                "every proposal requires exact before/after evidence"
            )
        if type(self.accepted) is not bool:
            raise TrainingEngineError("accepted must be exact bool")
        if self.accepted != all(update.accepted for update in self.updates):
            raise TrainingEngineError(
                "step acceptance must equal all proposal dispositions"
            )
        if self.accepted and self.failure_kind is not None:
            raise TrainingEngineError(
                "accepted step cannot retain a failure kind"
            )
        if not self.accepted and (
            type(self.failure_kind) is not str or not self.failure_kind
        ):
            raise TrainingEngineError(
                "rejected step requires a failure kind"
            )
        if self.reverse_mode_autograd is not True:
            raise TrainingEngineError("training uses reverse-mode autograd")
        if self.expected_autograd_scope not in ("m_step", "e_and_m"):
            raise TrainingEngineError(
                "attempt expected autograd scope is invalid"
            )
        if self.accepted:
            if self.observed_autograd_scope != self.expected_autograd_scope:
                raise TrainingEngineError(
                    "accepted attempt did not observe its exact scope"
                )
        elif self.observed_autograd_scope not in (
            "m_step",
            "e_and_m",
            "partial",
            "not_observed",
        ):
            raise TrainingEngineError(
                "rejected attempt observed-scope classification is invalid"
            )
        if self.monotonicity_claim is not False:
            raise TrainingEngineError(
                "Adam proposals cannot claim monotonicity"
            )
        if type(self.objective_diagnostics_applicable) is not bool:
            raise TrainingEngineError(
                "objective diagnostic applicability must be exact bool"
            )
        if self.objective_diagnostics_applicable:
            if (
                type(self.objective_terms) is not dict
                or not self.objective_terms
                or type(self.counted_targets) is not int
                or self.counted_targets <= 0
                or any(
                    type(value) is not float or not math.isfinite(value)
                    for value in self.objective_terms.values()
                )
            ):
                raise TrainingEngineError(
                    "applicable objective diagnostics are incomplete"
                )
        elif (
            self.objective_terms is not None
            or self.complete_elbo_numerator is not None
            or self.complete_elbo_value is not None
            or self.counted_targets is not None
        ):
            raise TrainingEngineError(
                "inapplicable objective diagnostics cannot fabricate values"
            )
        if (
            self.complete_elbo_numerator is None
        ) != (self.complete_elbo_value is None):
            raise TrainingEngineError(
                "complete ELBO numerator/value applicability differs"
            )
        if self.complete_elbo_value is not None:
            assert self.complete_elbo_numerator is not None
            assert self.counted_targets is not None
            if (
                self.objective_kind != "complete_elbo"
                or not self.objective_diagnostics_applicable
                or not math.isfinite(self.complete_elbo_numerator)
                or not math.isfinite(self.complete_elbo_value)
                or self.complete_elbo_value
                != self.complete_elbo_numerator / self.counted_targets
                or self.objective_terms is None
                or self.objective_terms.get("complete_elbo_numerator")
                != self.complete_elbo_numerator
            ):
                raise TrainingEngineError(
                    "complete ELBO diagnostic arithmetic is inconsistent"
                )
        elif (
            self.objective_kind == "complete_elbo"
            and self.objective_diagnostics_applicable
        ):
            raise TrainingEngineError(
                "applicable complete ELBO requires numerator and value"
            )


class AttemptEventSink(Protocol):
    """Mandatory durable-recording and target-blind validation seam."""

    def record_step(
        self,
        *,
        step_index: int,
        cumulative_counted_targets: int,
        result: StepResult,
    ) -> None: ...

    def validate_target_blind(
        self,
        *,
        step_index: int,
        cumulative_counted_targets: int,
    ) -> None: ...

    def record_terminal_failure(
        self,
        *,
        step_index: int,
        cumulative_counted_targets: int,
        result: StepResult | None,
        exception: Exception | None,
    ) -> None: ...


@dataclasses.dataclass(frozen=True, slots=True)
class AttemptResult:
    """One bounded attempt with exact target and boundary accounting."""

    steps: tuple[StepResult, ...]
    cumulative_counted_targets: int
    validation_step_boundaries: tuple[int, ...]
    completed_validation_step_boundaries: tuple[int, ...]
    terminal_failure_recorded: bool

    def __post_init__(self) -> None:
        if (
            type(self.steps) is not tuple
            or not self.steps
            or any(type(step) is not StepResult for step in self.steps)
        ):
            raise TrainingEngineError(
                "attempt result requires exact step records"
            )
        expected_targets = sum(
            step.counted_targets or 0
            for step in self.steps
            if step.accepted
        )
        if (
            type(self.cumulative_counted_targets) is not int
            or self.cumulative_counted_targets != expected_targets
        ):
            raise TrainingEngineError(
                "attempt counted-target accumulation is inconsistent"
            )
        _validate_validation_boundaries(self.validation_step_boundaries)
        expected_completed = tuple(
            boundary
            for boundary in self.validation_step_boundaries
            if boundary <= len(self.steps)
            and all(step.accepted for step in self.steps[:boundary])
        )
        if self.completed_validation_step_boundaries != expected_completed:
            raise TrainingEngineError(
                "completed validation boundaries are inconsistent"
            )
        expected_terminal = not self.steps[-1].accepted
        if (
            type(self.terminal_failure_recorded) is not bool
            or self.terminal_failure_recorded != expected_terminal
        ):
            raise TrainingEngineError(
                "terminal failure recording is inconsistent"
            )


def _validate_validation_boundaries(
    boundaries: tuple[int, ...],
) -> None:
    if (
        type(boundaries) is not tuple
        or any(type(boundary) is not int or boundary <= 0 for boundary in boundaries)
        or tuple(sorted(set(boundaries))) != boundaries
    ):
        raise TrainingEngineError(
            "validation boundaries must be a strictly ascending integer tuple"
        )


def _validate_attempt_event_sink(event_sink: AttemptEventSink) -> None:
    for operation in (
        "record_step",
        "validate_target_blind",
        "record_terminal_failure",
    ):
        if not callable(getattr(event_sink, operation, None)):
            raise TrainingEngineError(
                f"attempt event sink lacks {operation}"
            )


@dataclasses.dataclass(slots=True)
class _RuntimeState:
    parameters: tuple[tuple[torch.nn.Parameter, torch.Tensor], ...]
    buffers: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    gradients: tuple[torch.Tensor | None, ...]
    requires_grad: tuple[bool, ...]
    optimizer_states: tuple[tuple[torch.optim.Optimizer, dict[str, object]], ...]
    scheduler_states: tuple[tuple[object, dict[str, object]], ...]
    scaler_state: dict[str, object] | None
    cpu_rng: torch.Tensor
    cuda_rng: tuple[torch.Tensor, ...]
    python_rng: object
    numpy_rng: object
    participant_states: tuple[
        tuple[ScientificStateParticipant, object, str],
        ...,
    ]
    update_counter: int


def _modules(runtime: ArmExecutionRuntime) -> tuple[torch.nn.Module, ...]:
    return (
        (runtime.model,)
        if runtime.recognition is None
        else (runtime.model, runtime.recognition)
    )


def _parameters(
    runtime: ArmExecutionRuntime,
) -> tuple[torch.nn.Parameter, ...]:
    return tuple(
        parameter
        for module in _modules(runtime)
        for parameter in module.parameters()
    )


def _buffers(runtime: ArmExecutionRuntime) -> tuple[torch.Tensor, ...]:
    return tuple(
        buffer
        for module in _modules(runtime)
        for buffer in module.buffers()
    )


def _participant_digest(
    participant: ScientificStateParticipant,
) -> str:
    value = participant.state_sha256()
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TrainingEngineError(
            "scientific state participant returned an invalid SHA-256"
        )
    return value


def _capture_state(runtime: ArmExecutionRuntime) -> _RuntimeState:
    parameters = _parameters(runtime)
    buffers = _buffers(runtime)
    optimizers = tuple(
        optimizer
        for optimizer in (
            runtime.model_optimizer,
            runtime.recognition_optimizer,
        )
        if optimizer is not None
    )
    schedulers = tuple(
        scheduler
        for scheduler in (
            runtime.model_scheduler,
            runtime.recognition_scheduler,
        )
        if scheduler is not None
    )
    for scheduler in schedulers:
        if not callable(getattr(scheduler, "state_dict", None)):
            raise TrainingEngineError("scheduler lacks state_dict")
    scaler_state = None
    if runtime.grad_scaler is not None:
        if not callable(getattr(runtime.grad_scaler, "state_dict", None)):
            raise TrainingEngineError("grad scaler lacks state_dict")
        scaler_state = copy.deepcopy(runtime.grad_scaler.state_dict())
    cuda_rng: tuple[torch.Tensor, ...] = ()
    if any(parameter.device.type == "cuda" for parameter in parameters):
        cuda_rng = tuple(state.clone() for state in torch.cuda.get_rng_state_all())
    return _RuntimeState(
        parameters=tuple(
            (parameter, parameter.detach().clone())
            for parameter in parameters
        ),
        buffers=tuple(
            (buffer, buffer.detach().clone())
            for buffer in buffers
        ),
        gradients=tuple(
            None if parameter.grad is None else parameter.grad.detach().clone()
            for parameter in parameters
        ),
        requires_grad=tuple(parameter.requires_grad for parameter in parameters),
        optimizer_states=tuple(
            (optimizer, copy.deepcopy(optimizer.state_dict()))
            for optimizer in optimizers
        ),
        scheduler_states=tuple(
            (scheduler, copy.deepcopy(scheduler.state_dict()))
            for scheduler in schedulers
        ),
        scaler_state=scaler_state,
        cpu_rng=torch.get_rng_state().clone(),
        cuda_rng=cuda_rng,
        python_rng=copy.deepcopy(random.getstate()),
        numpy_rng=copy.deepcopy(np.random.get_state()),
        participant_states=tuple(
            (
                participant,
                copy.deepcopy(participant.capture_state()),
                _participant_digest(participant),
            )
            for participant in runtime.state_participants
        ),
        update_counter=runtime.update_counter,
    )


def _restore_state(
    runtime: ArmExecutionRuntime,
    state: _RuntimeState,
) -> None:
    with torch.no_grad():
        for parameter, value in state.parameters:
            parameter.copy_(value)
        for buffer, value in state.buffers:
            buffer.copy_(value)
    for (parameter, _), gradient, requires_grad in zip(
        state.parameters,
        state.gradients,
        state.requires_grad,
        strict=True,
    ):
        parameter.requires_grad_(requires_grad)
        parameter.grad = None if gradient is None else gradient.clone()
    for optimizer, optimizer_state in state.optimizer_states:
        optimizer.load_state_dict(copy.deepcopy(optimizer_state))
    for scheduler, scheduler_state in state.scheduler_states:
        scheduler.load_state_dict(copy.deepcopy(scheduler_state))
    if runtime.grad_scaler is not None and state.scaler_state is not None:
        runtime.grad_scaler.load_state_dict(copy.deepcopy(state.scaler_state))
    torch.set_rng_state(state.cpu_rng)
    if state.cuda_rng:
        torch.cuda.set_rng_state_all(list(state.cuda_rng))
    random.setstate(copy.deepcopy(state.python_rng))
    np.random.set_state(copy.deepcopy(state.numpy_rng))
    for participant, value, expected_digest in state.participant_states:
        participant.restore_state(copy.deepcopy(value))
        if _participant_digest(participant) != expected_digest:
            raise TrainingEngineError(
                "scientific state participant did not restore exactly"
            )
    runtime.update_counter = state.update_counter


def _set_active_module(
    runtime: ArmExecutionRuntime,
    *,
    active: torch.nn.Module,
) -> tuple[torch.nn.Parameter, ...]:
    active_parameters = tuple(active.parameters())
    if not active_parameters:
        raise _ProposalRejected("empty_active_parameter_set")
    active_ids = {id(parameter) for parameter in active_parameters}
    for parameter in _parameters(runtime):
        parameter.grad = None
        parameter.requires_grad_(id(parameter) in active_ids)
    return active_parameters


def _float_bytes(value: float) -> bytes:
    return struct.pack("<d", value)


def _state_digest(value: object) -> str:
    digest = hashlib.sha256(b"VFE4-WT103-RUNTIME-STATE-V1\x00")

    def update(item: object) -> None:
        if item is None:
            digest.update(b"N")
        elif type(item) is bool:
            digest.update(b"B1" if item else b"B0")
        elif type(item) is int:
            encoded = str(item).encode("ascii")
            digest.update(b"I" + len(encoded).to_bytes(8, "little") + encoded)
        elif type(item) is float:
            if not math.isfinite(item):
                raise TrainingEngineError(
                    "runtime state contains a nonfinite float"
                )
            digest.update(b"F" + _float_bytes(item))
        elif type(item) is str:
            encoded = item.encode("utf-8")
            digest.update(b"S" + len(encoded).to_bytes(8, "little") + encoded)
        elif type(item) is bytes:
            digest.update(b"Y" + len(item).to_bytes(8, "little") + item)
        elif type(item) is torch.Tensor:
            identity = _tensor_identity("", item)
            digest.update(b"T")
            update(identity)
        elif isinstance(item, Mapping):
            digest.update(b"M" + len(item).to_bytes(8, "little"))
            rows = sorted(item.items(), key=lambda row: repr(row[0]))
            for key, child in rows:
                update(key)
                update(child)
        elif type(item) in (tuple, list):
            digest.update(
                (b"Q" if type(item) is tuple else b"L")
                + len(item).to_bytes(8, "little")
            )
            for child in item:
                update(child)
        else:
            raise TrainingEngineError(
                "unsupported runtime state value "
                f"{type(item).__name__}"
            )

    update(value)
    return digest.hexdigest()


def _scheduler_state(scheduler: object | None) -> object:
    if scheduler is None:
        return ("not_applicable",)
    return scheduler.state_dict()


def _make_update_record(
    *,
    runtime: ArmExecutionRuntime,
    phase: str,
    accepted: bool,
    rejection_reason: str | None,
    snapshot_sha256: str | None,
    optimizer: torch.optim.Optimizer,
    scheduler: object | None,
    expected_autograd_scope: PhaseAutogradScope,
    observed_autograd_scope: Literal[
        "e_step",
        "m_step",
        "not_observed",
    ],
) -> WT103UpdateRecord:
    payload = {
        "schema_version": "wt103-update-record-v1",
        "arm_id": runtime.arm_spec.arm_id,
        "phase": phase,
        "update_label": "adam_proposal",
        "accepted": accepted,
        "rejection_reason": rejection_reason,
        "expected_autograd_scope": expected_autograd_scope,
        "observed_autograd_scope": observed_autograd_scope,
        "snapshot_sha256": snapshot_sha256,
        "optimizer_state_sha256": _state_digest(optimizer.state_dict()),
        "scheduler_state_sha256": _state_digest(
            _scheduler_state(scheduler)
        ),
    }
    return WT103UpdateRecord(
        **payload,
        update_sha256=owned_sha256(
            "vfe4.wt103.update-record.v1",
            payload,
        ),
    )  # type: ignore[arg-type]


def _assert_pre_step_state_unmodified(
    state: _RuntimeState,
    *,
    active_ids: set[int],
) -> None:
    for parameter, original in state.parameters:
        if id(parameter) not in active_ids and not torch.equal(
            parameter.detach(),
            original,
        ):
            raise _ProposalRejected("inactive_parameter_mutation")
    for buffer, original in state.buffers:
        if not torch.equal(buffer.detach(), original):
            raise _ProposalRejected("module_buffer_mutation")
    for optimizer, original in state.optimizer_states:
        if _state_digest(optimizer.state_dict()) != _state_digest(original):
            raise _ProposalRejected("optimizer_access_mismatch")
    for scheduler, original in state.scheduler_states:
        if _state_digest(scheduler.state_dict()) != _state_digest(original):
            raise _ProposalRejected("scheduler_access_mismatch")


def _assert_inactive_state_unchanged(
    state: _RuntimeState,
    *,
    active_ids: set[int],
    optimizer: torch.optim.Optimizer,
    scheduler: object | None,
) -> None:
    for parameter, original in state.parameters:
        if id(parameter) not in active_ids and not torch.equal(
            parameter.detach(),
            original,
        ):
            raise _ProposalRejected("inactive_parameter_mutation")
    for buffer, original in state.buffers:
        if not torch.equal(buffer.detach(), original):
            raise _ProposalRejected("module_buffer_mutation")
    for candidate, original in state.optimizer_states:
        if candidate is not optimizer and (
            _state_digest(candidate.state_dict()) != _state_digest(original)
        ):
            raise _ProposalRejected("inactive_optimizer_mutation")
    for candidate, original in state.scheduler_states:
        if candidate is not scheduler and (
            _state_digest(candidate.state_dict()) != _state_digest(original)
        ):
            raise _ProposalRejected("inactive_scheduler_mutation")


def _gradient_norm(parameters: tuple[torch.nn.Parameter, ...]) -> float:
    squared = []
    for parameter in parameters:
        if parameter.grad is None:
            raise _ProposalRejected("missing_active_gradient")
        squared.append(
            float(
                parameter.grad.detach()
                .to(dtype=torch.float64)
                .square()
                .sum()
                .item()
            )
        )
    value = math.sqrt(math.fsum(squared))
    if not math.isfinite(value):
        raise _ProposalRejected("nonfinite_gradient_norm")
    return value


def _gradient_inf_norm(
    parameters: tuple[torch.nn.Parameter, ...],
) -> float:
    maxima: list[float] = []
    for parameter in parameters:
        if parameter.grad is None:
            raise _ProposalRejected("missing_active_gradient")
        maxima.append(
            float(
                parameter.grad.detach()
                .abs()
                .amax()
                .to(dtype=torch.float64)
                .item()
            )
        )
    value = max(maxima, default=0.0)
    if not math.isfinite(value):
        raise _ProposalRejected("nonfinite_gradient_inf_norm")
    return value


def _plain_group_float(
    group: Mapping[str, object],
    key: str,
) -> float:
    raw = group.get(key)
    if type(raw) not in (float, int) or type(raw) is bool:
        raise TrainingEngineError(
            f"AdamW {key} must be a plain numeric scalar"
        )
    value = float(raw)
    if not math.isfinite(value):
        raise TrainingEngineError(f"AdamW {key} must be finite")
    return value


def _scheduler_ordinal(scheduler: object | None) -> int:
    if scheduler is None:
        return 0
    state = scheduler.state_dict()
    ordinal = state.get("last_epoch")
    if type(ordinal) is not int or ordinal < 0:
        raise TrainingEngineError(
            "scheduler last_epoch must be a nonnegative exact integer"
        )
    return ordinal


def _make_update_control(
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: object | None,
    learning_rate: float,
    clipping_threshold: float,
    pre_clip_norm: float | None,
    post_clip_norm: float | None,
    pre_clip_inf_norm: float | None,
    post_clip_inf_norm: float | None,
    amp_scale: float | None,
    amp_overflow: bool | None,
) -> UpdateControlRecord:
    if len(optimizer.param_groups) != 1:
        raise TrainingEngineError(
            "update controls require one exact AdamW parameter group"
        )
    group = optimizer.param_groups[0]
    betas = group.get("betas")
    if (
        type(betas) is not tuple
        or len(betas) != 2
        or any(type(value) is not float for value in betas)
    ):
        raise TrainingEngineError("AdamW betas must be two plain floats")
    gradient_applicability: Literal["applicable", "not_applicable"] = (
        "applicable"
        if pre_clip_norm is not None and post_clip_norm is not None
        else "not_applicable"
    )
    if (
        (pre_clip_norm is None)
        != (post_clip_norm is None)
        or (pre_clip_norm is None)
        != (pre_clip_inf_norm is None)
        or (pre_clip_norm is None)
        != (post_clip_inf_norm is None)
    ):
        raise TrainingEngineError(
            "gradient norm observations must be jointly applicable"
        )
    return UpdateControlRecord.create(
        learning_rate=learning_rate,
        scheduler_ordinal=_scheduler_ordinal(scheduler),
        scheduler_state_sha256=_state_digest(_scheduler_state(scheduler)),
        amp_applicability=(
            "not_applicable" if amp_scale is None else "applicable"
        ),
        amp_scale=amp_scale,
        amp_overflow=amp_overflow,
        clipping_threshold=clipping_threshold,
        gradient_norm_applicability=gradient_applicability,
        pre_clip_norm=pre_clip_norm,
        post_clip_norm=post_clip_norm,
        pre_clip_inf_norm=pre_clip_inf_norm,
        post_clip_inf_norm=post_clip_inf_norm,
        clipped=(
            None
            if pre_clip_norm is None
            else pre_clip_norm > clipping_threshold
        ),
        adamw_beta1=betas[0],
        adamw_beta2=betas[1],
        adamw_epsilon=_plain_group_float(group, "eps"),
        adamw_weight_decay=_plain_group_float(group, "weight_decay"),
        adamw_amsgrad=group["amsgrad"],
        adamw_maximize=group["maximize"],
        adamw_capturable=group["capturable"],
        adamw_differentiable=group["differentiable"],
        adamw_foreach=group["foreach"],
        adamw_fused=group["fused"],
    )


def _finite_objective_diagnostics(
    terms: ForwardTerms | None,
) -> tuple[
    dict[str, float] | None,
    float | None,
    float | None,
]:
    if terms is None:
        return None, None, None
    values = terms.detached_values()
    if any(not math.isfinite(value) for value in values.values()):
        return None, None, None
    numerator = terms.complete_elbo_numerator()
    value = terms.complete_elbo_value()
    if (
        numerator is None
    ) != (value is None) or any(
        item is not None and not math.isfinite(item)
        for item in (numerator, value)
    ):
        return None, None, None
    return values, numerator, value


def _phase_scope_from_gradients(
    runtime: ArmExecutionRuntime,
) -> PhaseAutogradScope:
    model_ids = {
        id(parameter)
        for parameter in runtime.model.parameters()
        if parameter.grad is not None
    }
    recognition_ids = (
        set()
        if runtime.recognition is None
        else {
            id(parameter)
            for parameter in runtime.recognition.parameters()
            if parameter.grad is not None
        }
    )
    all_model_ids = {id(parameter) for parameter in runtime.model.parameters()}
    all_recognition_ids = (
        set()
        if runtime.recognition is None
        else {
            id(parameter)
            for parameter in runtime.recognition.parameters()
        }
    )
    if model_ids == all_model_ids and not recognition_ids:
        return "m_step"
    if (
        all_recognition_ids
        and recognition_ids == all_recognition_ids
        and not model_ids
    ):
        return "e_step"
    raise _ProposalRejected("scope_mismatch")


def _objective_rows(
    terms: ForwardTerms | None,
) -> tuple[
    tuple[tuple[str, float], ...] | None,
    float | None,
    float | None,
]:
    values, _numerator, _value = _finite_objective_diagnostics(terms)
    if values is None or terms is None:
        return None, None, None
    objective = float(terms.objective().detach().cpu().item())
    if not math.isfinite(objective):
        return None, None, None
    error = None
    if terms.estimator_error_bound is not None:
        error = float(
            terms.estimator_error_bound.detach().cpu().item()
        )
        if not math.isfinite(error) or error < 0.0:
            return None, None, None
    return tuple(values.items()), objective, error


def _make_proposal_evidence(
    *,
    phase: str,
    affected_block: Literal["recognition", "model"],
    expected_scope: PhaseAutogradScope,
    observed_scope: PhaseAutogradScope | None,
    before_terms: ForwardTerms | None,
    after_terms: ForwardTerms | None,
    support_valid: bool | None,
    spd_valid: bool | None,
    damping_applied: bool | None,
    projection_applied: bool | None,
    rollback_applied: bool,
) -> ProposalEvidence:
    before_rows, before_value, before_error = _objective_rows(before_terms)
    after_rows, after_value, after_error = _objective_rows(after_terms)
    if rollback_applied:
        after_rows = None
        after_value = None
        after_error = None
    return ProposalEvidence.create(
        phase=phase,
        affected_block=affected_block,
        expected_autograd_scope=expected_scope,
        observed_autograd_scope=observed_scope,
        objective_before_terms=before_rows,
        objective_after_terms=after_rows,
        objective_before_value=before_value,
        objective_after_value=after_value,
        counted_targets=(
            None if before_rows is None else before_terms.counted_targets
        ),
        estimator_error_bound_before=before_error,
        estimator_error_bound_after=after_error,
        support_valid=support_valid,
        spd_valid=spd_valid,
        damping_applied=damping_applied,
        projection_applied=projection_applied,
        rollback_applied=rollback_applied,
    )


def _run_execution_event(
    runtime: ArmExecutionRuntime,
    event_name: str,
    operation: Callable[[], object],
) -> object:
    runner = runtime.execution_event_runner
    if runner is None:
        return operation()
    return runner(event_name, operation)


def _run_proposal(
    runtime: ArmExecutionRuntime,
    *,
    phase: str,
    batch: object,
    snapshot: RecognitionSnapshot | None,
    expected_counted_targets: int | None,
) -> tuple[
    WT103UpdateRecord,
    UpdateControlRecord,
    ProposalEvidence,
    ForwardTerms | None,
    str | None,
]:
    if phase == "recognition_adam_proposal":
        assert runtime.recognition is not None
        assert runtime.recognition_optimizer is not None
        active_module = runtime.recognition
        optimizer = runtime.recognition_optimizer
        scheduler = runtime.recognition_scheduler
        affected_block: Literal["recognition", "model"] = "recognition"
        expected_scope: PhaseAutogradScope = "e_step"
        compute_event = "recognition_forward"
        backward_event = "recognition_backward"
    elif phase in ("model_adam_proposal", "model_ce_adam_proposal"):
        active_module = runtime.model
        optimizer = runtime.model_optimizer
        scheduler = runtime.model_scheduler
        affected_block = "model"
        expected_scope = "m_step"
        compute_event = (
            "forward"
            if phase == "model_ce_adam_proposal"
            else runtime.arm_spec.training_objective
        )
        backward_event = (
            "backward"
            if phase == "model_ce_adam_proposal"
            else "model_backward"
        )
    else:
        raise TrainingEngineError(f"unknown proposal phase {phase!r}")

    state = _capture_state(runtime)
    group = optimizer.param_groups[0]
    learning_rate = _plain_group_float(group, "lr")
    amp_scale_before: float | None = None
    if runtime.grad_scaler is not None:
        if not callable(getattr(runtime.grad_scaler, "get_scale", None)):
            raise TrainingEngineError("grad scaler lacks get_scale")
        amp_scale_before = float(runtime.grad_scaler.get_scale())
        if not math.isfinite(amp_scale_before) or amp_scale_before <= 0.0:
            raise TrainingEngineError("grad scaler scale is invalid")
    terms: ForwardTerms | None = None
    after_terms: ForwardTerms | None = None
    rejection: str | None = None
    pre_clip_norm: float | None = None
    post_clip_norm: float | None = None
    pre_clip_inf_norm: float | None = None
    post_clip_inf_norm: float | None = None
    amp_overflow: bool | None = (
        None if amp_scale_before is None else False
    )
    observed_scope: PhaseAutogradScope | None = None
    support_valid: bool | None = None
    spd_valid: bool | None = None
    damping_applied: bool | None = None
    projection_applied: bool | None = None
    committed = False
    try:
        active_parameters = _set_active_module(runtime, active=active_module)
        active_ids = {id(parameter) for parameter in active_parameters}
        computed = _run_execution_event(
            runtime,
            compute_event,
            lambda: runtime.compute_terms(phase, batch, snapshot),
        )
        if type(computed) is not ForwardTerms:
            raise _ProposalRejected("invalid_forward_terms")
        computed.__post_init__()
        if computed.objective_kind != runtime.arm_spec.training_objective:
            raise _ProposalRejected("objective_kind_mismatch")
        if (
            expected_counted_targets is not None
            and computed.counted_targets != expected_counted_targets
        ):
            raise _ProposalRejected("counted_target_mismatch")
        terms = computed
        if phase == "model_ce_adam_proposal":
            loss = _run_execution_event(
                runtime,
                "cross_entropy",
                computed.loss,
            )
        else:
            loss = computed.loss()
        if not bool(torch.isfinite(loss.detach()).item()):
            raise _ProposalRejected("nonfinite_objective")
        optimizer.zero_grad(set_to_none=True)
        def backward() -> None:
            if runtime.grad_scaler is None:
                loss.backward()
            else:
                runtime.grad_scaler.scale(loss).backward()
                runtime.grad_scaler.unscale_(optimizer)

        _run_execution_event(runtime, backward_event, backward)
        active_ids = {id(parameter) for parameter in active_parameters}
        inactive = tuple(
            parameter
            for parameter in _parameters(runtime)
            if id(parameter) not in active_ids
        )
        if any(parameter.grad is not None for parameter in inactive):
            raise _ProposalRejected("inactive_gradient_observed")
        if any(parameter.grad is None for parameter in active_parameters):
            raise _ProposalRejected("missing_active_gradient")
        if any(
            not bool(torch.isfinite(parameter.grad).all().item())
            for parameter in active_parameters
            if parameter.grad is not None
        ):
            raise _ProposalRejected("nonfinite_gradient")
        observed_scope = _phase_scope_from_gradients(runtime)
        if observed_scope != expected_scope:
            raise _ProposalRejected("scope_mismatch")
        _assert_pre_step_state_unmodified(
            state,
            active_ids=active_ids,
        )
        pre_clip_inf_norm = _gradient_inf_norm(active_parameters)
        total_norm = torch.nn.utils.clip_grad_norm_(
            active_parameters,
            max_norm=runtime.gradient_clip_norm,
            error_if_nonfinite=True,
        )
        if not bool(torch.isfinite(total_norm).item()):
            raise _ProposalRejected("nonfinite_gradient_norm")
        pre_clip_norm = float(total_norm.detach().cpu().item())
        post_clip_norm = _gradient_norm(active_parameters)
        post_clip_inf_norm = _gradient_inf_norm(active_parameters)
        optimizer_event = (
            "recognition_adamw"
            if phase == "recognition_adam_proposal"
            else "model_adamw"
        )

        def optimizer_update() -> None:
            nonlocal amp_overflow
            if runtime.grad_scaler is None:
                optimizer.step()
            else:
                runtime.grad_scaler.step(optimizer)
                runtime.grad_scaler.update()
                amp_scale_after = float(runtime.grad_scaler.get_scale())
                if (
                    not math.isfinite(amp_scale_after)
                    or amp_scale_after <= 0.0
                ):
                    raise _ProposalRejected("invalid_amp_scale")
                amp_overflow = amp_scale_after < amp_scale_before
                if amp_overflow:
                    raise _ProposalRejected("amp_overflow")
            if scheduler is not None:
                scheduler.step()
            runtime.update_counter += 1

        _run_execution_event(runtime, optimizer_event, optimizer_update)
        if any(
            not bool(torch.isfinite(parameter.detach()).all().item())
            for parameter in active_parameters
        ):
            raise _ProposalRejected("nonfinite_parameter")
        _assert_inactive_state_unchanged(
            state,
            active_ids=active_ids,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        post_parameters = tuple(
            (parameter, parameter.detach().clone())
            for parameter in _parameters(runtime)
        )
        post_buffers = tuple(
            (buffer, buffer.detach().clone())
            for buffer in _buffers(runtime)
        )
        post_optimizer_hashes = tuple(
            (
                candidate,
                _state_digest(candidate.state_dict()),
            )
            for candidate, _ in state.optimizer_states
        )
        post_scheduler_hashes = tuple(
            (
                candidate,
                _state_digest(candidate.state_dict()),
            )
            for candidate, _ in state.scheduler_states
        )
        computed_after = _run_execution_event(
            runtime,
            compute_event,
            lambda: runtime.compute_terms(phase, batch, snapshot),
        )
        if type(computed_after) is not ForwardTerms:
            raise _ProposalRejected("invalid_post_forward_terms")
        computed_after.__post_init__()
        if (
            computed_after.objective_kind
            != runtime.arm_spec.training_objective
            or computed_after.counted_targets != computed.counted_targets
        ):
            raise _ProposalRejected("post_objective_contract_mismatch")
        after_objective = computed_after.objective()
        if not bool(torch.isfinite(after_objective.detach()).item()):
            raise _ProposalRejected("nonfinite_post_objective")
        after_terms = computed_after
        if any(
            not torch.equal(parameter.detach(), value)
            for parameter, value in post_parameters
        ) or any(
            not torch.equal(buffer.detach(), value)
            for buffer, value in post_buffers
        ):
            raise _ProposalRejected("post_evaluation_state_mutation")
        if any(
            _state_digest(candidate.state_dict()) != digest
            for candidate, digest in post_optimizer_hashes
        ) or any(
            _state_digest(candidate.state_dict()) != digest
            for candidate, digest in post_scheduler_hashes
        ):
            raise _ProposalRejected("post_evaluation_state_mutation")
        try:
            support_valid = runtime.support_validator()
        except Exception as exc:
            raise _ProposalRejected("support_validation_error") from exc
        if support_valid is not True:
            raise _ProposalRejected("invalid_support")
        try:
            spd_valid = runtime.spd_validator()
        except Exception as exc:
            raise _ProposalRejected("spd_validation_error") from exc
        if spd_valid is not True:
            raise _ProposalRejected("spd_validation_failed")
        try:
            damping_applied = runtime.damping_observer()
            projection_applied = runtime.projection_observer()
        except Exception as exc:
            raise _ProposalRejected(
                "numerical_action_observation_error"
            ) from exc
        if (
            type(damping_applied) is not bool
            or type(projection_applied) is not bool
        ):
            raise _ProposalRejected(
                "invalid_numerical_action_observation"
            )
        if any(
            not torch.equal(parameter.detach(), value)
            for parameter, value in post_parameters
        ) or any(
            not torch.equal(buffer.detach(), value)
            for buffer, value in post_buffers
        ):
            raise _ProposalRejected("validator_state_mutation")
        if snapshot is not None:
            assert runtime.recognition is not None
            snapshot.assert_nonaliasing(runtime.recognition)
        record = _make_update_record(
            runtime=runtime,
            phase=phase,
            accepted=True,
            rejection_reason=None,
            snapshot_sha256=(
                None if snapshot is None else snapshot.snapshot_sha256
            ),
            optimizer=optimizer,
            scheduler=scheduler,
            expected_autograd_scope=expected_scope,
            observed_autograd_scope=observed_scope,
        )
        control = _make_update_control(
            optimizer=optimizer,
            scheduler=scheduler,
            learning_rate=learning_rate,
            clipping_threshold=runtime.gradient_clip_norm,
            pre_clip_norm=pre_clip_norm,
            post_clip_norm=post_clip_norm,
            pre_clip_inf_norm=pre_clip_inf_norm,
            post_clip_inf_norm=post_clip_inf_norm,
            amp_scale=amp_scale_before,
            amp_overflow=amp_overflow,
        )
        evidence = _make_proposal_evidence(
            phase=phase,
            affected_block=affected_block,
            expected_scope=expected_scope,
            observed_scope=observed_scope,
            before_terms=computed,
            after_terms=computed_after,
            support_valid=support_valid,
            spd_valid=spd_valid,
            damping_applied=damping_applied,
            projection_applied=projection_applied,
            rollback_applied=False,
        )
        committed = True
        return record, control, evidence, computed_after, None
    except _ProposalRejected as exc:
        rejection = exc.reason
    except Exception as exc:
        rejection = f"proposal_exception:{type(exc).__name__}"
    finally:
        if not committed:
            _restore_state(runtime, state)
        else:
            for (parameter, _), requires_grad in zip(
                state.parameters,
                state.requires_grad,
                strict=True,
            ):
                parameter.requires_grad_(requires_grad)
                parameter.grad = None
    assert rejection is not None
    record = _make_update_record(
        runtime=runtime,
        phase=phase,
        accepted=False,
        rejection_reason=rejection,
        snapshot_sha256=None if snapshot is None else snapshot.snapshot_sha256,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_autograd_scope=expected_scope,
        observed_autograd_scope=(
            "not_observed" if observed_scope is None else observed_scope
        ),
    )
    control = _make_update_control(
        optimizer=optimizer,
        scheduler=scheduler,
        learning_rate=learning_rate,
        clipping_threshold=runtime.gradient_clip_norm,
        pre_clip_norm=pre_clip_norm,
        post_clip_norm=post_clip_norm,
        pre_clip_inf_norm=pre_clip_inf_norm,
        post_clip_inf_norm=post_clip_inf_norm,
        amp_scale=amp_scale_before,
        amp_overflow=amp_overflow,
    )
    evidence = _make_proposal_evidence(
        phase=phase,
        affected_block=affected_block,
        expected_scope=expected_scope,
        observed_scope=observed_scope,
        before_terms=terms,
        after_terms=after_terms,
        support_valid=support_valid,
        spd_valid=spd_valid,
        damping_applied=damping_applied,
        projection_applied=projection_applied,
        rollback_applied=True,
    )
    return record, control, evidence, terms, rejection


def train_step(
    runtime: ArmExecutionRuntime,
    *,
    batch: object,
) -> StepResult:
    """Execute one frozen arm step using ordinary reverse-mode autograd."""

    if type(runtime) is not ArmExecutionRuntime:
        raise TrainingEngineError(
            "runtime must be an exact ArmExecutionRuntime"
        )
    runtime.validate()
    phase_order: list[str] = []
    updates: list[WT103UpdateRecord] = []
    update_controls: list[UpdateControlRecord] = []
    proposal_evidence: list[ProposalEvidence] = []
    snapshot: RecognitionSnapshot | None = None
    final_terms: ForwardTerms | None = None
    expected_targets: int | None = None
    failure: str | None = None
    for phase in runtime.arm_spec.update_phases:
        phase_order.append(phase)
        if phase == "immutable_detached_snapshot":
            if runtime.recognition is None:
                raise TrainingEngineError(
                    "snapshot phase requires recognition state"
                )
            captured = _run_execution_event(
                runtime,
                "immutable_detached_snapshot",
                lambda: RecognitionSnapshot.capture(runtime.recognition),
            )
            if type(captured) is not RecognitionSnapshot:
                raise TrainingEngineError(
                    "snapshot instrumentation changed the result"
                )
            snapshot = captured
            continue
        update, control, evidence, terms, rejection = _run_proposal(
            runtime,
            phase=phase,
            batch=batch,
            snapshot=snapshot,
            expected_counted_targets=expected_targets,
        )
        updates.append(update)
        update_controls.append(control)
        proposal_evidence.append(evidence)
        final_terms = terms
        if expected_targets is None and terms is not None:
            expected_targets = terms.counted_targets
        if rejection is not None:
            failure = rejection
            break
    if not updates:
        raise TrainingEngineError("step executed no proposal")
    accepted = failure is None and all(update.accepted for update in updates)
    expected_scope: AttemptAutogradScope = (
        "e_and_m" if runtime.arm_spec.latent_enabled else "m_step"
    )
    observed_phase_scopes = tuple(
        evidence.observed_autograd_scope
        for evidence in proposal_evidence
        if evidence.observed_autograd_scope is not None
    )
    if runtime.arm_spec.latent_enabled and observed_phase_scopes == (
        "e_step",
        "m_step",
    ):
        observed_scope: ObservedAttemptAutogradScope = "e_and_m"
    elif not runtime.arm_spec.latent_enabled and observed_phase_scopes == (
        "m_step",
    ):
        observed_scope = "m_step"
    elif not observed_phase_scopes:
        observed_scope = "not_observed"
    else:
        observed_scope = "partial"
    (
        objective_values,
        complete_elbo_numerator,
        complete_elbo_value,
    ) = _finite_objective_diagnostics(
        final_terms
    )
    diagnostics_applicable = (
        objective_values is not None and expected_targets is not None
    )
    if accepted and not diagnostics_applicable:
        raise TrainingEngineError(
            "accepted step lacks finite objective diagnostics"
        )
    return StepResult(
        arm_id=runtime.arm_spec.arm_id,
        objective_kind=runtime.arm_spec.training_objective,  # type: ignore[arg-type]
        phase_order=tuple(phase_order),
        updates=tuple(updates),
        update_controls=tuple(update_controls),
        proposal_evidence=tuple(proposal_evidence),
        snapshot_sha256=(
            None if snapshot is None else snapshot.snapshot_sha256
        ),
        objective_diagnostics_applicable=diagnostics_applicable,
        objective_terms=objective_values,
        complete_elbo_numerator=complete_elbo_numerator,
        complete_elbo_value=complete_elbo_value,
        counted_targets=(
            expected_targets if diagnostics_applicable else None
        ),
        accepted=accepted,
        failure_kind=failure,
        expected_autograd_scope=expected_scope,
        observed_autograd_scope=observed_scope,
        reverse_mode_autograd=True,
        monotonicity_claim=False,
    )


def train_attempt(
    runtime: ArmExecutionRuntime,
    *,
    batches: Iterable[object],
    validation_step_boundaries: tuple[int, ...],
    event_sink: AttemptEventSink,
) -> AttemptResult:
    """Run a bounded attempt with mandatory recording and validation seams."""

    _validate_validation_boundaries(validation_step_boundaries)
    _validate_attempt_event_sink(event_sink)
    results: list[StepResult] = []
    completed_boundaries: list[int] = []
    cumulative_targets = 0
    for step_index, batch in enumerate(batches, start=1):
        try:
            result = train_step(runtime, batch=batch)
        except Exception as exc:
            event_sink.record_terminal_failure(
                step_index=step_index,
                cumulative_counted_targets=cumulative_targets,
                result=None,
                exception=exc,
            )
            raise
        results.append(result)
        if result.accepted:
            assert result.counted_targets is not None
            cumulative_targets += result.counted_targets
        try:
            event_sink.record_step(
                step_index=step_index,
                cumulative_counted_targets=cumulative_targets,
                result=result,
            )
        except Exception as exc:
            event_sink.record_terminal_failure(
                step_index=step_index,
                cumulative_counted_targets=cumulative_targets,
                result=result,
                exception=exc,
            )
            raise
        if not result.accepted:
            event_sink.record_terminal_failure(
                step_index=step_index,
                cumulative_counted_targets=cumulative_targets,
                result=result,
                exception=None,
            )
            break
        if step_index in validation_step_boundaries:
            try:
                event_sink.validate_target_blind(
                    step_index=step_index,
                    cumulative_counted_targets=cumulative_targets,
                )
            except Exception as exc:
                event_sink.record_terminal_failure(
                    step_index=step_index,
                    cumulative_counted_targets=cumulative_targets,
                    result=result,
                    exception=exc,
                )
                raise
            completed_boundaries.append(step_index)
    if not results:
        raise TrainingEngineError("train_attempt requires at least one batch")
    return AttemptResult(
        steps=tuple(results),
        cumulative_counted_targets=cumulative_targets,
        validation_step_boundaries=validation_step_boundaries,
        completed_validation_step_boundaries=tuple(completed_boundaries),
        terminal_failure_recorded=not results[-1].accepted,
    )


__all__ = [
    "AttemptEventSink",
    "AttemptResult",
    "ArmExecutionRuntime",
    "ExecutionEventRunner",
    "ForwardTerms",
    "ProposalEvidence",
    "RecognitionSnapshot",
    "ScientificStateParticipant",
    "StepResult",
    "TrainingEngineError",
    "WT103_STRUCTURED_FACTOR_ELBO_SCHEMA",
    "WT103_STRUCTURED_FACTOR_ELBO_SCHEMA_SHA256",
    "train_attempt",
    "train_step",
]
