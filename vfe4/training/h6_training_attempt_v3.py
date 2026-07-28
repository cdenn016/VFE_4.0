"""Authenticated H6 v3 training-attempt and batch bridges."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor, nn

from vfe4.artifacts.h6_prediction_v3 import (
    H6PredictionV3Authorities,
)
from vfe4.data.access import (
    H6TrainingDataV3,
    issue_h6_train_capability_v3,
    open_train_for_training_v3,
)
from vfe4.data.wikitext2 import BlindedCorpusStore
from vfe4.data.windows import (
    CausalPrefix,
    CausalWindows,
    frozen_batch_schedule,
    quarter_pass_batches,
    schedule_batches,
)
from vfe4.objective.h6_prediction_v3 import (
    evaluate_h6_no_latent_cross_entropy_v3,
    evaluate_h6_prediction_emission_only_v3,
    evaluate_h6_prediction_elbo_v3,
)
from vfe4.predictive.identities import canonical_model_state_sha256
from vfe4.recognition.h6_prediction_v3 import (
    AbsentSourceBank,
    CategoricalSourceBank,
    H6ActiveHorizonV3,
    LanguageRecognitionTrajectory,
    RecognitionPriorFeature,
    RecognitionPriorFeatureProvider,
    SourceBankName,
    project_active_recognition_topology_v3,
)
from vfe4.recognition.language import RecognitionConditioning
from vfe4.recognition.parameter_store import (
    LanguageRecognitionParameterStore,
    RecognitionStoreStateBindingV3,
)
from vfe4.types.h6 import TrainingPhase, canonical_json_bytes
from vfe4.types.h6_prediction_v3 import (
    H6AttemptCursorV3,
    H6ObjectiveManifestV3,
    H6_DETERMINISTIC_POLICY_SHA256,
    H6_NO_COUNTER_CONSUMPTION_SHA256,
)

from .arms import (
    BuiltArm,
    H6CausalTransformer,
    LatentLanguageArmModel,
    MeanPooledPrefixFloor,
)
from .checkpoint_v3 import (
    H6CheckpointV3,
    H6HydratedCheckpointV3,
    _issue_h6_checkpoint_factory_authority_v3,
    capture_h6_checkpoint_v3,
    hydrate_h6_checkpoint_v3,
    read_h6_checkpoint_file_v3,
)
from .h6_engine_v3 import (
    H6BatchLiveRecognitionStateV3,
    H6DetachedBatchRecognitionSnapshotV3,
    H6EngineAuthorityV3,
    H6LiveObjectiveTermV3,
    H6LiveRecognitionStateV3,
    H6MetricRecordV3,
    H6PhaseObjectiveV3,
    H6PhaseRecordV3,
    H6TrainingBatchResultV3,
    run_h6_training_batch_v3,
)
from .h6_execution_v3 import (
    H6ExecutableAttemptV3,
    bind_h6_executable_attempt_v3,
)
from .h6_experiment_v3 import (
    H6PlannedAttemptV3,
    realize_seeded_initialization_v3,
    seeded_initialization_sha256_v3,
)
from .h6_noise_v3 import training_batch_normal_tensor_v3
from .h6_runtime_v3 import (
    H6InstalledRuntimeBindingV3,
    H6RuntimeBindingV3,
    H6SyntheticCpuRuntimeV3,
    prepare_training_module_v3,
)
from .h6_transformer_v3 import H6TrainingCausalTransformerV3
from .matching import H6_ADAMW_POLICY


class _GenerativePriorCallV3(nn.Module):
    """Call one exact source-prior method through a stateless module boundary."""

    def __init__(self, model: LatentLanguageArmModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        *,
        bank: SourceBankName,
        causal_prefix: CausalPrefix,
        earlier_latents: Tensor,
    ) -> Tensor:
        kwargs: dict[str, object]
        if self.model.prior_variant == "fixed":
            kwargs = {}
        else:
            kwargs = {
                "prefix": causal_prefix,
                "earlier_latents": earlier_latents,
            }
        if bank == "state":
            return self.model.state_source_log_probs(
                causal_prefix.receiver_t,
                **kwargs,
            )
        return self.model.model_source_log_probs(
            causal_prefix.receiver_t,
            **kwargs,
        )


def _stopped_module_state(module: nn.Module) -> Mapping[str, Tensor]:
    state: dict[str, Tensor] = {}
    for name, tensor in (
        *module.named_parameters(),
        *module.named_buffers(),
    ):
        if name in state:
            raise ValueError("source model parameter and buffer names overlap")
        state[name] = tensor.detach().clone(memory_format=torch.preserve_format)
    return state


class H6GenerativePriorFeatureProviderV3(RecognitionPriorFeatureProvider):
    """Mean-evaluate the exact live arm prior with stopped model parameters."""

    uses_internal_stopped_state_v3 = True

    def __init__(self, *, model: LatentLanguageArmModel) -> None:
        super().__init__()
        if type(model) is not LatentLanguageArmModel:
            raise ValueError("H6 v3 prior features require an exact latent arm model")
        if model.source_mode != "categorical" or model.source_prior is None:
            raise ValueError("H6 v3 prior features require a categorical source prior")
        self._call = _GenerativePriorCallV3(model)
        self._stopped_state = _stopped_module_state(self._call)
        self._stopped_state_sha256 = canonical_model_state_sha256(model)
        self._source_inventory = tuple(
            (
                name,
                tensor,
                id(tensor),
                tensor.data_ptr(),
                tensor._version,
                tuple(tensor.shape),
                tensor.dtype,
                tensor.device,
            )
            for name, tensor in (
                *model.named_parameters(),
                *model.named_buffers(),
            )
        )

    @property
    def source_model(self) -> nn.Module:
        return self._call.model

    @property
    def stopped_state_sha256_v3(self) -> str:
        return self._stopped_state_sha256

    def stopped_state_v3(self) -> Mapping[str, Tensor]:
        self.assert_stopped_source_intact_v3()
        return self._stopped_state

    def assert_stopped_source_intact_v3(self) -> None:
        current = tuple(
            (name, tensor)
            for name, tensor in (
                *self._call.model.named_parameters(),
                *self._call.model.named_buffers(),
            )
        )
        if len(current) != len(self._source_inventory):
            raise ValueError("source model state inventory mutated")
        for (name, tensor), recorded in zip(
            current,
            self._source_inventory,
            strict=True,
        ):
            (
                expected_name,
                expected_tensor,
                expected_id,
                expected_pointer,
                expected_version,
                expected_shape,
                expected_dtype,
                expected_device,
            ) = recorded
            if (
                name != expected_name
                or tensor is not expected_tensor
                or id(tensor) != expected_id
                or tensor.data_ptr() != expected_pointer
                or tensor._version != expected_version
                or tuple(tensor.shape) != expected_shape
                or tensor.dtype != expected_dtype
                or tensor.device != expected_device
            ):
                raise ValueError("source model mutated during stopped-state evaluation")

    def forward(
        self,
        *,
        bank: SourceBankName,
        causal_prefix: CausalPrefix,
        earlier_recognition_means: Tensor,
    ) -> RecognitionPriorFeature:
        model = self._call.model
        if bank not in ("state", "model"):
            raise ValueError("source bank must be state or model")
        if bank == "model" and not model.model_channel_enabled:
            raise ValueError("model source bank is structurally absent")
        if type(causal_prefix) is not CausalPrefix:
            raise ValueError("prior features require an exact target-free CausalPrefix")
        causal_prefix.__post_init__()
        receiver_t = causal_prefix.receiver_t
        channel_count = 2 if model.model_channel_enabled else 1
        dimension = channel_count * model.latent_width
        if (
            causal_prefix.vocabulary != model.vocabulary
            or receiver_t > model.horizon
            or type(earlier_recognition_means) is not Tensor
            or earlier_recognition_means.dtype is not torch.float64
            or earlier_recognition_means.shape != (receiver_t, dimension)
            or not bool(torch.isfinite(earlier_recognition_means.detach()).all())
        ):
            raise ValueError(
                "prior feature prefix/recognition history does not match the live arm"
            )
        model_device = next(model.parameters()).device
        if earlier_recognition_means.device != model_device:
            raise ValueError(
                "prior feature recognition history must share the model device"
            )
        offset = 0 if bank == "state" else model.latent_width
        earlier_latents = earlier_recognition_means[
            :, offset : offset + model.latent_width
        ]
        self.assert_stopped_source_intact_v3()
        log_probs = torch.func.functional_call(
            self._call,
            self._stopped_state,
            (),
            {
                "bank": bank,
                "causal_prefix": causal_prefix,
                "earlier_latents": earlier_latents,
            },
            strict=True,
        )
        self.assert_stopped_source_intact_v3()
        if (
            type(log_probs) is not Tensor
            or log_probs.dtype is not torch.float64
            or log_probs.shape != (receiver_t,)
            or log_probs.device != model_device
            or bool(torch.isnan(log_probs.detach()).any())
            or bool(torch.isposinf(log_probs.detach()).any())
        ):
            raise ValueError("live source prior returned an invalid normalized row")
        support = tuple(
            int(index)
            for index in torch.nonzero(
                torch.isfinite(log_probs.detach()),
                as_tuple=False,
            )
            .reshape(-1)
            .to(device="cpu")
            .tolist()
        )
        if not support:
            raise ValueError("live source prior has no positive-mass parent")
        indices = torch.tensor(
            support,
            dtype=torch.int64,
            device=log_probs.device,
        )
        return RecognitionPriorFeature(
            bank=bank,
            causal_prefix_sha256=causal_prefix.prefix_sha256,
            support=support,
            log_prior_features=log_probs.index_select(0, indices),
        )


def _owned_hash(domain: str, payload: object) -> str:
    return hashlib.sha256(
        domain.encode("ascii") + b"\x00" + canonical_json_bytes(payload)
    ).hexdigest()


def _tensor_sha256(value: Tensor) -> str:
    cpu = value.detach().to(device="cpu").contiguous()
    return hashlib.sha256(bytes(cpu.reshape(-1).view(torch.uint8).tolist())).hexdigest()


def _trajectory_live_state_v3(
    *,
    trajectory: LanguageRecognitionTrajectory,
    active_horizon: H6ActiveHorizonV3,
    endpoint_config_sha256: str,
) -> H6LiveRecognitionStateV3:
    """Project one exact active trajectory without padding its terminal law."""

    if type(trajectory) is not LanguageRecognitionTrajectory:
        raise ValueError("trajectory must be exact")
    trajectory.__post_init__()
    if (
        type(active_horizon) is not H6ActiveHorizonV3
        or trajectory.active_horizon_binding != active_horizon
    ):
        raise ValueError("trajectory does not carry the exact active-horizon binding")
    active_horizon.__post_init__()
    topology = project_active_recognition_topology_v3(
        trajectory=trajectory,
        active_horizon=active_horizon,
    )
    tensors: dict[str, Tensor] = {}
    for receiver_t, components in enumerate(trajectory.receiver_components):
        component_ids = topology.receiver_components[receiver_t][1]
        for component_id, component in zip(
            component_ids,
            components,
            strict=True,
        ):
            tensors[f"receiver.{receiver_t}.component.{component_id}.mean"] = (
                component.mean
            )
        tensors[f"receiver.{receiver_t}.shared_precision_cholesky"] = (
            trajectory.shared_precision_cholesky
        )
    for bank_name, bank in (
        ("state", trajectory.state_source),
        ("model", trajectory.model_source),
    ):
        if type(bank) is CategoricalSourceBank:
            for row in bank.rows:
                tensors[f"{bank_name}.receiver.{row.receiver_t}.support"] = (
                    torch.tensor(
                        row.support,
                        dtype=torch.int64,
                        device=row.log_probabilities.device,
                    )
                )
                tensors[f"{bank_name}.receiver.{row.receiver_t}.categorical_row"] = (
                    row.probabilities
                )
        elif type(bank) is AbsentSourceBank:
            reference = trajectory.base_means
            tensors[f"{bank_name}.absent.support"] = torch.tensor(
                (-1,),
                dtype=torch.int64,
                device=reference.device,
            )
            tensors[f"{bank_name}.absent.categorical_row"] = reference.new_ones((1,))
        else:
            raise ValueError("trajectory source bank has the wrong type")
    return H6LiveRecognitionStateV3.create(
        endpoint_config_sha256=endpoint_config_sha256,
        receiver_count=topology.receiver_count,
        state_categorical_enabled=topology.state_categorical_enabled,
        model_categorical_enabled=topology.model_categorical_enabled,
        state_categorical_supports=topology.state_categorical_supports,
        model_categorical_supports=topology.model_categorical_supports,
        receiver_components=topology.receiver_components,
        tensors=tensors,
        context_sha256=trajectory.horizon_scope_identity_sha256,
        recognition_state_sha256=(trajectory.recognition_store_state_sha256),
        source_model_sha256=trajectory.source_model_state_sha256,
        law_sha256=topology.topology_identity_sha256,
    )


@dataclass(frozen=True, slots=True)
class _H6TrainingWindowBatchV3:
    window_indices: tuple[int, ...]
    active_horizons: tuple[H6ActiveHorizonV3, ...]
    observed_targets: tuple[Tensor, ...]

    def __post_init__(self) -> None:
        if (
            type(self.window_indices) is not tuple
            or not self.window_indices
            or len(self.window_indices) > 8
            or len(self.active_horizons) != len(self.window_indices)
            or len(self.observed_targets) != len(self.window_indices)
        ):
            raise ValueError("training window batch inventory is invalid")
        for binding, targets in zip(
            self.active_horizons,
            self.observed_targets,
            strict=True,
        ):
            binding.__post_init__()
            if (
                type(targets) is not Tensor
                or targets.device.type != "cpu"
                or targets.dtype is not torch.int64
                or targets.shape != (binding.active_horizon,)
                or not targets.is_contiguous()
            ):
                raise ValueError(
                    "batch targets must be exact active CPU int64 prefixes"
                )


def _window_batch_v3(
    *,
    windows: CausalWindows,
    window_indices: tuple[int, ...],
    maximum_horizon: int,
) -> _H6TrainingWindowBatchV3:
    bindings: list[H6ActiveHorizonV3] = []
    targets: list[Tensor] = []
    for window_index in window_indices:
        if not 0 <= window_index < len(windows):
            raise IndexError("training window index is outside the split")
        active = windows.real_target_counts[window_index]
        binding = H6ActiveHorizonV3.create(
            maximum_horizon=maximum_horizon,
            active_horizon=active,
            active_receiver_mask=(
                (True,) * (active + 1) + (False,) * (maximum_horizon - active)
            ),
        )
        bindings.append(binding)
        targets.append(
            torch.tensor(
                windows.targets[window_index][:active],
                dtype=torch.int64,
                device="cpu",
            ).contiguous()
        )
    return _H6TrainingWindowBatchV3(
        window_indices=tuple(window_indices),
        active_horizons=tuple(bindings),
        observed_targets=tuple(targets),
    )


class _H6BatchCallbacksV3:
    """Own the live batch trajectories and phase-local objective evidence."""

    def __init__(
        self,
        *,
        built_arm: BuiltArm,
        model: nn.Module,
        recognition: nn.Module | None,
        authority: H6EngineAuthorityV3,
        windows: CausalWindows,
        batch: _H6TrainingWindowBatchV3,
    ) -> None:
        self.built_arm = built_arm
        self.model = model
        self.recognition = recognition
        self.authority = authority
        self.windows = windows
        self.batch = batch
        self._trajectories: tuple[LanguageRecognitionTrajectory, ...] = ()
        self._live_batch: H6BatchLiveRecognitionStateV3 | None = None
        self.latest_phase: TrainingPhase | None = None
        self.latest_factor_bindings: tuple[tuple[str, int, str], ...] = ()
        self.latest_total_raw_bytes_sha256: str | None = None
        self.latest_recognition_law_sha256: str | None = None
        self.latest_detached_snapshot_sha256: str | None = None
        self._prior_provider: H6GenerativePriorFeatureProviderV3 | None = None

    @property
    def latent_dimension(self) -> int:
        config = self.built_arm.config
        latent_width = config.capacity_allocation.latent_width
        if type(latent_width) is not int or latent_width <= 0:
            raise ValueError("latent batch lacks a positive latent width")
        return latent_width * (2 if config.model_channel_enabled else 1)

    def recognition_forward(self) -> H6BatchLiveRecognitionStateV3:
        if (
            type(self.model) is not LatentLanguageArmModel
            or type(self.recognition) is not LanguageRecognitionParameterStore
        ):
            raise ValueError(
                "latent callbacks require exact model and recognition store"
            )
        provider = H6GenerativePriorFeatureProviderV3(model=self.model)
        store_binding = RecognitionStoreStateBindingV3.capture(self.recognition)
        trajectories: list[LanguageRecognitionTrajectory] = []
        states: list[H6LiveRecognitionStateV3] = []
        mode = self.built_arm.config.recognition_conditioning
        if mode not in ("filtering", "smoothing"):
            raise ValueError("latent endpoint has no recognition conditioning")
        trajectory_builder = (
            self.recognition.source_prior_free_recognition_trajectory
            if (self.authority.objective_kind == "emission_only_ablation_non_elbo")
            else self.recognition.recognition_trajectory
        )
        for targets, binding in zip(
            self.batch.observed_targets,
            self.batch.active_horizons,
            strict=True,
        ):
            conditioning = RecognitionConditioning.create(
                mode=mode,
                horizon=binding.active_horizon,
                observed_tokens=targets,
            )
            trajectory = trajectory_builder(
                conditioning,
                prior_feature_provider=provider,
                active_horizon=binding,
                state_binding=store_binding,
            )
            trajectories.append(trajectory)
            states.append(
                _trajectory_live_state_v3(
                    trajectory=trajectory,
                    active_horizon=binding,
                    endpoint_config_sha256=(self.authority.endpoint_config_sha256),
                )
            )
        live = H6BatchLiveRecognitionStateV3.create(
            authority=self.authority,
            states=tuple(states),
            active_target_counts=tuple(
                binding.active_horizon for binding in self.batch.active_horizons
            ),
            active_receiver_masks=tuple(
                binding.active_receiver_mask for binding in self.batch.active_horizons
            ),
        )
        self._trajectories = tuple(trajectories)
        self._live_batch = live
        store_binding.assert_intact(self.recognition)
        provider.assert_stopped_source_intact_v3()
        self._prior_provider = provider
        return live

    def noise_factory(
        self,
        phase: TrainingPhase,
        cursor: H6AttemptCursorV3,
    ) -> tuple[Tensor, str]:
        noise = training_batch_normal_tensor_v3(
            attempt_spec_sha256=cursor.attempt_spec_sha256,
            pass_index=cursor.pass_index,
            batch_index=cursor.batch_index,
            phase=phase,
            draw_block=cursor.draw_block,
            example_count=len(self.batch.window_indices),
            receiver_count=self.authority.receiver_count,
            active_receiver_counts=tuple(
                binding.active_horizon + 1 for binding in self.batch.active_horizons
            ),
            latent_dimension=self.latent_dimension,
            # The engine authenticates counter streams on CPU.  Each exact
            # active slice is transferred once, immediately before the live
            # ELBO consumes it.
            device="cpu",
        )
        return noise.tensor, noise.consumption_sha256

    def _record_evidence(
        self,
        *,
        phase: TrainingPhase,
        factors: tuple[tuple[str, int, str], ...],
        total: Tensor,
        recognition_law_sha256: str | None,
        detached_snapshot_sha256: str | None,
    ) -> None:
        self.latest_phase = phase
        self.latest_factor_bindings = factors
        self.latest_total_raw_bytes_sha256 = _tensor_sha256(total)
        self.latest_recognition_law_sha256 = recognition_law_sha256
        self.latest_detached_snapshot_sha256 = detached_snapshot_sha256

    def _authenticated_detached_trajectories(
        self,
        snapshot: H6DetachedBatchRecognitionSnapshotV3,
    ) -> tuple[LanguageRecognitionTrajectory, ...]:
        """Authenticate the fresh post-step law, reconstructing only on resume."""

        reconstructed = self._live_batch
        if reconstructed is None or not self._trajectories:
            # A new process has no trusted in-memory trajectory sidecar.  It
            # reconstructs once and authenticates that law against the
            # persisted detached snapshot.  The uninterrupted path already
            # owns the exact post-step forward used to create ``snapshot`` and
            # must not perform a third full recognition computation.
            with torch.no_grad():
                reconstructed = self.recognition_forward()
        if (
            reconstructed.batch_live_state_sha256 != snapshot.live_batch_state_sha256
            or reconstructed.names != snapshot.names
            or reconstructed.active_target_counts != snapshot.active_target_counts
            or reconstructed.active_receiver_masks != snapshot.active_receiver_masks
        ):
            raise ValueError(
                "recomputed recognition law differs from the persisted snapshot"
            )
        for name in snapshot.names:
            live_tensor = reconstructed[name]
            persisted_tensor = snapshot[name]
            if (
                live_tensor.dtype != persisted_tensor.dtype
                or live_tensor.shape != persisted_tensor.shape
                or _tensor_sha256(live_tensor) != _tensor_sha256(persisted_tensor)
            ):
                raise ValueError(
                    "recomputed recognition tensor differs from the persisted snapshot"
                )
        trajectories = self._trajectories
        if len(trajectories) != len(snapshot.example_ordinals):
            raise ValueError("recomputed detached recognition trajectory is incomplete")
        return trajectories

    def _latent_objective(
        self,
        *,
        phase: TrainingPhase,
        recognition_state: object,
        noise: Tensor,
    ) -> H6PhaseObjectiveV3:
        if type(self.model) is not LatentLanguageArmModel:
            raise ValueError("latent objective requires a latent arm model")
        if type(recognition_state) is H6BatchLiveRecognitionStateV3:
            if (
                self._live_batch is None
                or recognition_state.batch_live_state_sha256
                != self._live_batch.batch_live_state_sha256
            ):
                raise ValueError("live recognition batch sidecar drift")
            active_block: Literal["recognition", "model"] = "recognition"
            detached_sha256 = None
            law_sha256 = recognition_state.batch_live_state_sha256
            trajectories = self._trajectories
        elif type(recognition_state) is H6DetachedBatchRecognitionSnapshotV3:
            if recognition_state.active_target_counts != tuple(
                binding.active_horizon for binding in self.batch.active_horizons
            ):
                raise ValueError("detached recognition batch sidecar drift")
            active_block = "model"
            detached_sha256 = recognition_state.snapshot_sha256
            law_sha256 = recognition_state.live_batch_state_sha256
            trajectories = self._authenticated_detached_trajectories(recognition_state)
        else:
            raise ValueError("latent objective requires an exact batch law")
        if len(trajectories) != len(self.batch.window_indices):
            raise ValueError("recognition trajectory sidecar is incomplete")
        if type(noise) is not Tensor or noise.shape != (
            len(self.batch.window_indices),
            self.authority.receiver_count,
            self.latent_dimension,
        ):
            raise ValueError("batch latent noise shape is invalid")

        denominator = sum(
            binding.active_horizon for binding in self.batch.active_horizons
        )
        provider = self._prior_provider
        if provider is None:
            raise ValueError("latent objective lacks its stopped source-state binding")
        provider.assert_stopped_source_intact_v3()
        expected_partitions = tuple(
            ("gaussian_entropy" if partition == "entropy" else partition)
            for partition in self.built_arm.elbo_factor_inventory
        )
        live_terms: list[H6LiveObjectiveTermV3] = []
        factor_bindings: list[tuple[str, int, str]] = []
        for example_ordinal, (trajectory, targets, binding) in enumerate(
            zip(
                trajectories,
                self.batch.observed_targets,
                self.batch.active_horizons,
                strict=True,
            )
        ):
            active_noise = noise[example_ordinal, : binding.active_horizon + 1].to(
                device=trajectory.base_means.device,
                dtype=torch.float64,
            )
            if self.authority.objective_kind == "emission_only_ablation_non_elbo":
                estimate = evaluate_h6_prediction_emission_only_v3(
                    model=self.model,
                    trajectory=trajectory,
                    observed_tokens=targets,
                    base_noise=active_noise,
                    active_parameter_block=active_block,
                    active_horizon=binding,
                    preauthenticated_source_model_state_sha256=(
                        provider.stopped_state_sha256_v3
                    ),
                )
            else:
                estimate = evaluate_h6_prediction_elbo_v3(
                    model=self.model,
                    trajectory=trajectory,
                    observed_tokens=targets,
                    base_noise=active_noise,
                    mixture_mode=self.built_arm.config.mixture_mode,
                    active_parameter_block=active_block,
                    active_horizon=binding,
                    preauthenticated_source_model_state_sha256=(
                        provider.stopped_state_sha256_v3
                    ),
                )
            observed_partitions = frozenset(
                term.partition for term in estimate.ordered_terms
            )
            if observed_partitions != frozenset(expected_partitions):
                raise ValueError(
                    "live objective factor inventory differs from BuiltArm"
                )
            selected = tuple(
                term
                for partition in expected_partitions
                for term in estimate.ordered_terms
                if term.partition == partition
            )
            for term in selected:
                scaled = term.value / denominator
                live_terms.append(
                    H6LiveObjectiveTermV3.create(
                        partition=term.partition,
                        receiver_t=term.receiver_t,
                        value=scaled,
                    )
                )
                factor_bindings.append(
                    (
                        f"example.{example_ordinal}.{term.partition}",
                        term.receiver_t,
                        _owned_hash(
                            "vfe4.h6.batch-objective-factor.v3",
                            {
                                "example_ordinal": example_ordinal,
                                "term_identity_sha256": (term.term_identity_sha256),
                                "active_target_denominator": denominator,
                            },
                        ),
                    )
                )
        provider.assert_stopped_source_intact_v3()
        terms = tuple(live_terms)
        objective = (
            H6PhaseObjectiveV3.complete_elbo(terms)
            if self.authority.objective_kind == "complete_elbo"
            else H6PhaseObjectiveV3.emission_only(terms)
        )
        self._record_evidence(
            phase=phase,
            factors=tuple(factor_bindings),
            total=objective.value,
            recognition_law_sha256=law_sha256,
            detached_snapshot_sha256=detached_sha256,
        )
        return objective

    def _cross_entropy_objective(
        self,
        *,
        phase: TrainingPhase,
    ) -> H6PhaseObjectiveV3:
        if not isinstance(
            self.model,
            (H6TrainingCausalTransformerV3, MeanPooledPrefixFloor),
        ):
            raise ValueError("cross-entropy endpoint has the wrong training model")
        logit_rows: list[Tensor] = []
        target_rows: list[Tensor] = []
        factor_bindings: list[tuple[str, int, str]] = []
        for example_ordinal, (_window_index, binding, targets) in enumerate(
            zip(
                self.batch.window_indices,
                self.batch.active_horizons,
                self.batch.observed_targets,
                strict=True,
            )
        ):
            active_rows: list[Tensor] = []
            for receiver_t in range(1, binding.active_horizon + 1):
                prefix = CausalPrefix.create(
                    receiver_t=receiver_t,
                    vocabulary=self.built_arm.config.vocabulary,
                    token_ids=targets[: receiver_t - 1].contiguous(),
                )
                active_rows.append(self.model.prefix_log_probs(prefix))
            logit_rows.extend(active_rows)
            target_rows.append(targets)
            factor_bindings.append(
                (
                    f"example.{example_ordinal}.emission",
                    binding.active_horizon,
                    binding.evaluation_identity_sha256,
                )
            )
        nll = evaluate_h6_no_latent_cross_entropy_v3(
            logits=torch.stack(tuple(logit_rows)),
            targets=torch.cat(tuple(target_rows)).contiguous(),
            active_horizons=self.batch.active_horizons,
        )
        objective = H6PhaseObjectiveV3.cross_entropy(
            H6LiveObjectiveTermV3.create(
                partition="emission",
                receiver_t=0,
                value=-nll,
            )
        )
        self._record_evidence(
            phase=phase,
            factors=tuple(factor_bindings),
            total=objective.value,
            recognition_law_sha256=None,
            detached_snapshot_sha256=None,
        )
        return objective

    def objective_forward(
        self,
        *,
        phase: TrainingPhase,
        recognition_state: object,
        noise: Tensor,
        **_unused: object,
    ) -> H6PhaseObjectiveV3:
        if self.authority.latent_enabled:
            return self._latent_objective(
                phase=phase,
                recognition_state=recognition_state,
                noise=noise,
            )
        if recognition_state is not None:
            raise ValueError("cross-entropy objective cannot receive recognition")
        return self._cross_entropy_objective(phase=phase)


@dataclass(frozen=True, slots=True)
class H6TrainingAttemptResultV3:
    """One terminal, canonically reopened training-attempt checkpoint."""

    result_schema: Literal["h6-training-attempt-result-v3"]
    stage: Literal["tuning", "confirmatory"]
    endpoint_config_id: str
    planned_attempt_sha256: str
    executable_attempt: H6ExecutableAttemptV3
    executable_attempt_sha256: str
    terminal_cursor: H6AttemptCursorV3
    terminal_checkpoint: H6CheckpointV3
    checkpoint_path: Path
    terminal_progress: H6TrainingAttemptProgressV3
    terminal_history: H6TrainingAttemptHistoryV3
    progress_path: Path
    progress_sha256: str
    history_sha256: str
    metric_history_count: int
    metric_history_sha256: str
    validation_boundary_history_count: int
    validation_history_sha256: str
    checkpoint_bytes_sha256: str
    checkpoint_byte_count: int
    batch_count: int
    result_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "result_schema": self.result_schema,
            "stage": self.stage,
            "endpoint_config_id": self.endpoint_config_id,
            "planned_attempt_sha256": self.planned_attempt_sha256,
            "executable_attempt_sha256": self.executable_attempt_sha256,
            "terminal_cursor_sha256": self.terminal_cursor.cursor_sha256,
            "terminal_checkpoint_sha256": (self.terminal_checkpoint.checkpoint_sha256),
            "checkpoint_path": self.checkpoint_path.as_posix(),
            "progress_path": self.progress_path.as_posix(),
            "progress_sha256": self.progress_sha256,
            "history_sha256": self.history_sha256,
            "metric_history_count": self.metric_history_count,
            "metric_history_sha256": self.metric_history_sha256,
            "validation_boundary_history_count": (
                self.validation_boundary_history_count
            ),
            "validation_history_sha256": self.validation_history_sha256,
            "checkpoint_bytes_sha256": self.checkpoint_bytes_sha256,
            "checkpoint_byte_count": self.checkpoint_byte_count,
            "batch_count": self.batch_count,
        }

    def __post_init__(self) -> None:
        if (
            self.result_schema != "h6-training-attempt-result-v3"
            or self.stage not in ("tuning", "confirmatory")
            or type(self.endpoint_config_id) is not str
            or not self.endpoint_config_id
        ):
            raise ValueError("training-attempt result schema is invalid")
        for name in (
            "planned_attempt_sha256",
            "executable_attempt_sha256",
            "progress_sha256",
            "history_sha256",
            "metric_history_sha256",
            "validation_history_sha256",
            "checkpoint_bytes_sha256",
            "result_sha256",
        ):
            value = getattr(self, name)
            if (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if type(self.terminal_cursor) is not H6AttemptCursorV3:
            raise ValueError("training result requires an exact terminal cursor")
        self.terminal_cursor.__post_init__()
        if type(self.executable_attempt) is not H6ExecutableAttemptV3:
            raise ValueError("training result requires an exact executable attempt")
        self.executable_attempt.__post_init__()
        if type(self.terminal_checkpoint) is not H6CheckpointV3:
            raise ValueError("training result requires an exact v3 checkpoint")
        self.terminal_checkpoint.__post_init__()
        if (
            type(self.terminal_progress) is not H6TrainingAttemptProgressV3
            or type(self.terminal_history) is not H6TrainingAttemptHistoryV3
        ):
            raise ValueError(
                "training result requires exact progress and history authorities"
            )
        self.terminal_progress.__post_init__()
        self.terminal_history.__post_init__()
        if (
            self.executable_attempt.executable_attempt_sha256
            != self.executable_attempt_sha256
            or self.executable_attempt.planned_attempt.planned_attempt_sha256
            != self.planned_attempt_sha256
            or self.executable_attempt.planned_attempt.stage != self.stage
            or self.executable_attempt.endpoint_config.config_id
            != self.endpoint_config_id
            or self.terminal_checkpoint.cursor != self.terminal_cursor
            or not isinstance(self.checkpoint_path, Path)
            or not self.checkpoint_path.is_absolute()
            or not isinstance(self.progress_path, Path)
            or self.progress_path
            != h6_training_attempt_progress_path_v3(self.checkpoint_path)
            or self.terminal_progress.progress_sha256 != self.progress_sha256
            or self.terminal_history.history_sha256 != self.history_sha256
            or self.terminal_progress.attempt_spec_sha256
            != self.terminal_checkpoint.attempt_spec.attempt_spec_sha256
            or self.terminal_history.attempt_spec_sha256
            != self.terminal_progress.attempt_spec_sha256
            or self.metric_history_count != len(self.terminal_history.metric_history)
            or self.metric_history_count != self.terminal_progress.metric_history_count
            or self.metric_history_sha256
            != self.terminal_progress.metric_history_sha256
            or self.metric_history_sha256 != self.terminal_history.metric_history_sha256
            or self.validation_boundary_history_count
            != len(self.terminal_history.validation_boundary_history)
            or self.validation_boundary_history_count
            != self.terminal_progress.validation_history_count
            or self.validation_history_sha256
            != self.terminal_progress.validation_history_sha256
            or self.validation_history_sha256
            != self.terminal_history.validation_history_sha256
            or type(self.checkpoint_byte_count) is not int
            or self.checkpoint_byte_count <= 0
            or type(self.batch_count) is not int
            or self.batch_count <= 0
            or self.batch_count != self.terminal_cursor.model_update_count
            or self.metric_history_count
            != (
                self.terminal_cursor.recognition_update_count
                + self.terminal_cursor.model_update_count
            )
            or self.validation_boundary_history_count
            != self.terminal_cursor.validation_boundary_count
        ):
            raise ValueError("training-attempt terminal result is inconsistent")
        raw = self.terminal_checkpoint.to_bytes()
        if (
            len(raw) != self.checkpoint_byte_count
            or hashlib.sha256(raw).hexdigest() != self.checkpoint_bytes_sha256
            or self.result_sha256
            != _owned_hash(
                "vfe4.h6.training-attempt-result.v3",
                self.canonical_payload(),
            )
        ):
            raise ValueError("training-attempt result identity is stale")


_EMPTY_METRIC_HISTORY_SHA256_V3 = _owned_hash(
    "vfe4.h6.attempt-metric-history.v3",
    {"count": 0, "record_sha256s": ()},
)
_EMPTY_VALIDATION_HISTORY_SHA256_V3 = _owned_hash(
    "vfe4.h6.attempt-validation-history.v3",
    {"count": 0, "record_sha256s": ()},
)
H6_VALIDATION_BOUNDARY_CONTRACT_SHA256_V3 = _owned_hash(
    "vfe4.h6.validation-boundary-contract.v3",
    {
        "kind": "authenticated_training_cadence_boundary",
        "heldout_scoring": "separate_blinded_validation_campaign",
        "binds": (
            "post_update_cursor",
            "canonical_checkpoint",
            "ordered_graph_free_training_metric_history",
        ),
    },
)


def _require_digest_v3(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _history_chain_sha256_v3(
    *,
    domain: str,
    prior_sha256: str,
    prior_count: int,
    appended_sha256s: tuple[str, ...],
) -> str:
    _require_digest_v3(prior_sha256, "prior history SHA-256")
    if type(prior_count) is not int or prior_count < 0:
        raise ValueError("prior history count must be nonnegative")
    for digest in appended_sha256s:
        _require_digest_v3(digest, "appended history SHA-256")
    return _owned_hash(
        domain,
        {
            "prior_sha256": prior_sha256,
            "prior_count": prior_count,
            "appended_sha256s": appended_sha256s,
            "result_count": prior_count + len(appended_sha256s),
        },
    )


@dataclass(frozen=True, slots=True)
class H6AttemptMetricHistoryRecordV3:
    """One graph-free engine metric placed in the global attempt order."""

    ordinal: int
    pass_index: int
    batch_index: int
    phase_record: H6PhaseRecordV3
    metric_record: H6MetricRecordV3
    record_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "pass_index": self.pass_index,
            "batch_index": self.batch_index,
            "phase_record": self.phase_record.canonical_payload(),
            "metric_record": {
                **self.metric_record.canonical_payload(),
                "metric_sha256": self.metric_record.metric_sha256,
            },
        }

    def __post_init__(self) -> None:
        if (
            type(self.ordinal) is not int
            or self.ordinal < 0
            or type(self.pass_index) is not int
            or self.pass_index < 0
            or type(self.batch_index) is not int
            or self.batch_index < 0
        ):
            raise ValueError("attempt metric coordinates must be nonnegative")
        if (
            type(self.phase_record) is not H6PhaseRecordV3
            or type(self.metric_record) is not H6MetricRecordV3
        ):
            raise ValueError("attempt metric requires exact engine records")
        self.phase_record.__post_init__()
        self.metric_record.__post_init__()
        if (
            self.phase_record.phase is not self.metric_record.phase
            or self.phase_record.objective_value != self.metric_record.objective_value
            or self.phase_record.loss_value != self.metric_record.loss_value
            or self.phase_record.gradient_norm != self.metric_record.gradient_norm
        ):
            raise ValueError("attempt metric phase and metric records disagree")
        if self.record_sha256 != _owned_hash(
            "vfe4.h6.attempt-metric-record.v3",
            self.canonical_payload(),
        ):
            raise ValueError("attempt metric record identity is stale")

    @classmethod
    def create(
        cls,
        *,
        ordinal: int,
        pass_index: int,
        batch_index: int,
        phase_record: H6PhaseRecordV3,
        metric_record: H6MetricRecordV3,
    ) -> "H6AttemptMetricHistoryRecordV3":
        values = {
            "ordinal": ordinal,
            "pass_index": pass_index,
            "batch_index": batch_index,
            "phase_record": phase_record,
            "metric_record": metric_record,
        }
        provisional = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(provisional, name, value)
        return cls(
            **values,
            record_sha256=_owned_hash(
                "vfe4.h6.attempt-metric-record.v3",
                provisional.canonical_payload(),
            ),
        )


@dataclass(frozen=True, slots=True)
class H6ValidationBoundaryHistoryRecordV3:
    """One exact cadence marker; held-out scoring remains a separate campaign."""

    ordinal: int
    pass_index: int
    completed_batch_count: int
    cursor_sha256: str
    checkpoint_sha256: str
    metric_history_sha256: str
    contract_sha256: str
    record_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "pass_index": self.pass_index,
            "completed_batch_count": self.completed_batch_count,
            "cursor_sha256": self.cursor_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "metric_history_sha256": self.metric_history_sha256,
            "contract_sha256": self.contract_sha256,
        }

    def __post_init__(self) -> None:
        if (
            type(self.ordinal) is not int
            or self.ordinal < 0
            or type(self.pass_index) is not int
            or self.pass_index < 0
            or type(self.completed_batch_count) is not int
            or self.completed_batch_count <= 0
        ):
            raise ValueError(
                "validation-boundary coordinates must be exact nonnegative integers"
            )
        for name in (
            "cursor_sha256",
            "checkpoint_sha256",
            "metric_history_sha256",
            "contract_sha256",
        ):
            _require_digest_v3(getattr(self, name), name)
        if self.contract_sha256 != H6_VALIDATION_BOUNDARY_CONTRACT_SHA256_V3:
            raise ValueError("validation-boundary contract identity drift")
        if self.record_sha256 != _owned_hash(
            "vfe4.h6.validation-boundary-history-record.v3",
            self.canonical_payload(),
        ):
            raise ValueError("validation-boundary record identity is stale")

    @classmethod
    def create(
        cls,
        *,
        ordinal: int,
        pass_index: int,
        completed_batch_count: int,
        cursor_sha256: str,
        checkpoint_sha256: str,
        metric_history_sha256: str,
    ) -> "H6ValidationBoundaryHistoryRecordV3":
        values = {
            "ordinal": ordinal,
            "pass_index": pass_index,
            "completed_batch_count": completed_batch_count,
            "cursor_sha256": cursor_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "metric_history_sha256": metric_history_sha256,
            "contract_sha256": H6_VALIDATION_BOUNDARY_CONTRACT_SHA256_V3,
        }
        provisional = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(provisional, name, value)
        return cls(
            **values,
            record_sha256=_owned_hash(
                "vfe4.h6.validation-boundary-history-record.v3",
                provisional.canonical_payload(),
            ),
        )


@dataclass(frozen=True, slots=True)
class H6AttemptHistoryShardV3:
    """One immutable append-only metric/validation delta at a recovery boundary."""

    shard_schema: Literal["h6-attempt-history-shard-v3"]
    attempt_spec_sha256: str
    boundary_ordinal: int
    prior_boundary_sha256: str | None
    metric_records: tuple[H6AttemptMetricHistoryRecordV3, ...]
    validation_records: tuple[H6ValidationBoundaryHistoryRecordV3, ...]
    resume_phase_records: tuple[H6PhaseRecordV3, ...]
    resume_metric_records: tuple[H6MetricRecordV3, ...]
    resume_checkpoint_phases: tuple[TrainingPhase, ...]
    resume_result_sha256: str | None
    shard_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "shard_schema": self.shard_schema,
            "attempt_spec_sha256": self.attempt_spec_sha256,
            "boundary_ordinal": self.boundary_ordinal,
            "prior_boundary_sha256": self.prior_boundary_sha256,
            "metric_records": tuple(
                {
                    **record.canonical_payload(),
                    "record_sha256": record.record_sha256,
                }
                for record in self.metric_records
            ),
            "validation_records": tuple(
                {
                    **record.canonical_payload(),
                    "record_sha256": record.record_sha256,
                }
                for record in self.validation_records
            ),
            "resume_phase_records": tuple(
                record.canonical_payload() for record in self.resume_phase_records
            ),
            "resume_metric_records": tuple(
                {
                    **record.canonical_payload(),
                    "metric_sha256": record.metric_sha256,
                }
                for record in self.resume_metric_records
            ),
            "resume_checkpoint_phases": tuple(
                phase.value for phase in self.resume_checkpoint_phases
            ),
            "resume_result_sha256": self.resume_result_sha256,
        }

    def __post_init__(self) -> None:
        if (
            self.shard_schema != "h6-attempt-history-shard-v3"
            or type(self.boundary_ordinal) is not int
            or self.boundary_ordinal < 0
        ):
            raise ValueError("attempt history shard schema is invalid")
        _require_digest_v3(
            self.attempt_spec_sha256,
            "history shard attempt_spec_sha256",
        )
        if self.prior_boundary_sha256 is not None:
            _require_digest_v3(
                self.prior_boundary_sha256,
                "prior_boundary_sha256",
            )
        for record in self.metric_records:
            if type(record) is not H6AttemptMetricHistoryRecordV3:
                raise ValueError("history shard metric record has wrong type")
            record.__post_init__()
        for record in self.validation_records:
            if type(record) is not H6ValidationBoundaryHistoryRecordV3:
                raise ValueError("history shard validation record has wrong type")
            record.__post_init__()
        if (
            len(self.resume_phase_records) != len(self.resume_metric_records)
            or any(
                type(record) is not H6PhaseRecordV3
                for record in self.resume_phase_records
            )
            or any(
                type(record) is not H6MetricRecordV3
                for record in self.resume_metric_records
            )
            or any(
                type(phase) is not TrainingPhase
                for phase in self.resume_checkpoint_phases
            )
        ):
            raise ValueError("history shard resume records are invalid")
        for ordinal, (phase_record, metric_record) in enumerate(
            zip(
                self.resume_phase_records,
                self.resume_metric_records,
                strict=True,
            )
        ):
            phase_record.__post_init__()
            metric_record.__post_init__()
            if (
                metric_record.ordinal != ordinal
                or metric_record.phase is not phase_record.phase
                or metric_record.objective_value != phase_record.objective_value
                or metric_record.loss_value != phase_record.loss_value
                or metric_record.gradient_norm != phase_record.gradient_norm
            ):
                raise ValueError("history shard resume histories disagree")
        if self.resume_result_sha256 is None:
            if (
                self.resume_phase_records
                or self.resume_metric_records
                or self.resume_checkpoint_phases
            ):
                raise ValueError("history shard resume records lack a result identity")
        else:
            _require_digest_v3(
                self.resume_result_sha256,
                "resume_result_sha256",
            )
        if self.shard_sha256 != _owned_hash(
            "vfe4.h6.attempt-history-shard.v3",
            self.canonical_payload(),
        ):
            raise ValueError("attempt history shard identity is stale")

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(
            {
                **self.canonical_payload(),
                "shard_sha256": self.shard_sha256,
            }
        )


@dataclass(frozen=True, slots=True)
class H6AttemptRecoveryBoundaryV3:
    """One immutable checkpoint/history pair in the recovery catalog."""

    boundary_kind: Literal[
        "post_recognition",
        "batch_boundary",
        "terminal",
    ]
    ordinal: int
    checkpoint_filename: str
    checkpoint_sha256: str
    checkpoint_bytes_sha256: str
    checkpoint_byte_count: int
    history_filename: str
    history_sha256: str
    history_bytes_sha256: str
    history_byte_count: int
    cursor_sha256: str
    next_phase: TrainingPhase
    metric_history_count: int
    metric_history_sha256: str
    validation_history_count: int
    validation_history_sha256: str
    boundary_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            name: (
                getattr(self, name).value
                if name == "next_phase"
                else getattr(self, name)
            )
            for name in tuple(self.__dataclass_fields__)[:-1]
        }

    def __post_init__(self) -> None:
        if (
            self.boundary_kind not in ("post_recognition", "batch_boundary", "terminal")
            or type(self.ordinal) is not int
            or self.ordinal < 0
            or type(self.next_phase) is not TrainingPhase
        ):
            raise ValueError("recovery boundary schema is invalid")
        for name in ("checkpoint_filename", "history_filename"):
            value = getattr(self, name)
            if (
                type(value) is not str
                or not value
                or Path(value).name != value
                or value in (".", "..")
            ):
                raise ValueError("recovery boundary filename is unsafe")
        for name in (
            "checkpoint_sha256",
            "checkpoint_bytes_sha256",
            "history_sha256",
            "history_bytes_sha256",
            "cursor_sha256",
            "metric_history_sha256",
            "validation_history_sha256",
        ):
            _require_digest_v3(getattr(self, name), name)
        for name in (
            "checkpoint_byte_count",
            "history_byte_count",
            "metric_history_count",
            "validation_history_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.checkpoint_byte_count == 0 or self.history_byte_count == 0:
            raise ValueError("recovery boundary artifacts must be nonempty")
        if self.boundary_sha256 != _owned_hash(
            "vfe4.h6.attempt-recovery-boundary.v3",
            self.canonical_payload(),
        ):
            raise ValueError("recovery boundary identity is stale")


@dataclass(frozen=True, slots=True)
class H6TrainingAttemptProgressV3:
    """Small canonical catalog over immutable recovery/history shards."""

    progress_schema: Literal["h6-training-attempt-progress-v3"]
    attempt_spec_sha256: str
    planned_attempt_sha256: str
    executable_attempt_sha256: str
    tuning_cell_sha256: str
    model_factory_sha256: str
    recognition_factory_sha256: str | None
    initialization_sha256: str
    runtime_identity_sha256: str
    deterministic_policy_sha256: str
    terminal_checkpoint_filename: str
    boundaries: tuple[H6AttemptRecoveryBoundaryV3, ...]
    metric_history_count: int
    metric_history_sha256: str
    validation_history_count: int
    validation_history_sha256: str
    progress_sha256: str

    @property
    def latest_boundary(self) -> H6AttemptRecoveryBoundaryV3:
        if not self.boundaries:
            raise ValueError("attempt progress has no recovery boundary")
        return self.boundaries[-1]

    def canonical_payload(self) -> dict[str, object]:
        return {
            "progress_schema": self.progress_schema,
            "attempt_spec_sha256": self.attempt_spec_sha256,
            "planned_attempt_sha256": self.planned_attempt_sha256,
            "executable_attempt_sha256": self.executable_attempt_sha256,
            "tuning_cell_sha256": self.tuning_cell_sha256,
            "model_factory_sha256": self.model_factory_sha256,
            "recognition_factory_sha256": (self.recognition_factory_sha256),
            "initialization_sha256": self.initialization_sha256,
            "runtime_identity_sha256": self.runtime_identity_sha256,
            "deterministic_policy_sha256": (self.deterministic_policy_sha256),
            "terminal_checkpoint_filename": (self.terminal_checkpoint_filename),
            "boundaries": tuple(
                {
                    **boundary.canonical_payload(),
                    "boundary_sha256": boundary.boundary_sha256,
                }
                for boundary in self.boundaries
            ),
            "metric_history_count": self.metric_history_count,
            "metric_history_sha256": self.metric_history_sha256,
            "validation_history_count": self.validation_history_count,
            "validation_history_sha256": self.validation_history_sha256,
        }

    def __post_init__(self) -> None:
        if self.progress_schema != "h6-training-attempt-progress-v3":
            raise ValueError("attempt progress schema is invalid")
        for name in (
            "attempt_spec_sha256",
            "planned_attempt_sha256",
            "executable_attempt_sha256",
            "tuning_cell_sha256",
            "model_factory_sha256",
            "initialization_sha256",
            "runtime_identity_sha256",
            "deterministic_policy_sha256",
            "metric_history_sha256",
            "validation_history_sha256",
        ):
            _require_digest_v3(getattr(self, name), name)
        if self.recognition_factory_sha256 is not None:
            _require_digest_v3(
                self.recognition_factory_sha256,
                "recognition_factory_sha256",
            )
        if self.deterministic_policy_sha256 != H6_DETERMINISTIC_POLICY_SHA256:
            raise ValueError("attempt progress deterministic policy drift")
        if (
            type(self.terminal_checkpoint_filename) is not str
            or Path(self.terminal_checkpoint_filename).name
            != self.terminal_checkpoint_filename
        ):
            raise ValueError("attempt progress terminal filename is unsafe")
        if type(self.boundaries) is not tuple or not self.boundaries:
            raise ValueError("attempt progress requires a recovery boundary")
        prior_metric_count = 0
        prior_validation_count = 0
        filenames: set[str] = set()
        for ordinal, boundary in enumerate(self.boundaries):
            if type(boundary) is not H6AttemptRecoveryBoundaryV3:
                raise ValueError("attempt progress boundary has wrong type")
            boundary.__post_init__()
            if (
                boundary.ordinal != ordinal
                or boundary.metric_history_count < prior_metric_count
                or boundary.validation_history_count < prior_validation_count
                or boundary.checkpoint_filename in filenames
                or boundary.history_filename in filenames
            ):
                raise ValueError("attempt progress boundary order is invalid")
            prior_metric_count = boundary.metric_history_count
            prior_validation_count = boundary.validation_history_count
            filenames.update((boundary.checkpoint_filename, boundary.history_filename))
        if (
            self.metric_history_count != prior_metric_count
            or self.validation_history_count != prior_validation_count
            or self.metric_history_sha256 != self.boundaries[-1].metric_history_sha256
            or self.validation_history_sha256
            != self.boundaries[-1].validation_history_sha256
        ):
            raise ValueError("attempt progress aggregate history is stale")
        if self.progress_sha256 != _owned_hash(
            "vfe4.h6.training-attempt-progress.v3",
            self.canonical_payload(),
        ):
            raise ValueError("attempt progress identity is stale")

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(
            {
                **self.canonical_payload(),
                "progress_sha256": self.progress_sha256,
            }
        )


@dataclass(frozen=True, slots=True)
class H6TrainingAttemptHistoryV3:
    """Fully reopened ordered histories, loaded only when explicitly requested."""

    attempt_spec_sha256: str
    metric_history: tuple[H6AttemptMetricHistoryRecordV3, ...]
    validation_boundary_history: tuple[H6ValidationBoundaryHistoryRecordV3, ...]
    metric_history_sha256: str
    validation_history_sha256: str
    history_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "attempt_spec_sha256": self.attempt_spec_sha256,
            "metric_record_sha256s": tuple(
                record.record_sha256 for record in self.metric_history
            ),
            "validation_record_sha256s": tuple(
                record.record_sha256 for record in self.validation_boundary_history
            ),
            "metric_history_sha256": self.metric_history_sha256,
            "validation_history_sha256": self.validation_history_sha256,
        }

    def __post_init__(self) -> None:
        for name in (
            "attempt_spec_sha256",
            "metric_history_sha256",
            "validation_history_sha256",
            "history_sha256",
        ):
            _require_digest_v3(getattr(self, name), name)
        if any(
            type(record) is not H6AttemptMetricHistoryRecordV3
            for record in self.metric_history
        ) or any(
            type(record) is not H6ValidationBoundaryHistoryRecordV3
            for record in self.validation_boundary_history
        ):
            raise ValueError("attempt history record inventory is invalid")
        for ordinal, record in enumerate(self.metric_history):
            record.__post_init__()
            if record.ordinal != ordinal:
                raise ValueError("attempt metric history is not append-only")
        for ordinal, record in enumerate(self.validation_boundary_history):
            record.__post_init__()
            if record.ordinal != ordinal:
                raise ValueError("validation history is not append-only")
        if self.history_sha256 != _owned_hash(
            "vfe4.h6.training-attempt-history.v3",
            self.canonical_payload(),
        ):
            raise ValueError("attempt history identity is stale")


def _adamw_for_module_v3(
    module: nn.Module,
    *,
    authority: H6EngineAuthorityV3,
) -> torch.optim.AdamW:
    parameters = tuple(module.parameters())
    if not parameters:
        raise ValueError("H6 optimizer module has no trainable parameters")
    return torch.optim.AdamW(
        parameters,
        lr=authority.optimizer_learning_rate,
        betas=H6_ADAMW_POLICY.betas,
        eps=H6_ADAMW_POLICY.eps,
        weight_decay=authority.optimizer_weight_decay,
        amsgrad=H6_ADAMW_POLICY.amsgrad,
        maximize=H6_ADAMW_POLICY.maximize,
        foreach=H6_ADAMW_POLICY.foreach,
        capturable=H6_ADAMW_POLICY.capturable,
        differentiable=H6_ADAMW_POLICY.differentiable,
        fused=H6_ADAMW_POLICY.fused,
    )


def _fresh_cpu_training_modules_v3(
    *,
    executable: H6ExecutableAttemptV3,
    runtime: H6RuntimeBindingV3,
) -> tuple[BuiltArm, nn.Module, nn.Module | None]:
    built = realize_seeded_initialization_v3(
        executable.endpoint_config,
        executable.planned_attempt.training_seed,
    )
    if (
        seeded_initialization_sha256_v3(built)
        != executable.planned_attempt.attempt_spec.initialization_sha256
    ):
        raise ValueError("seed-realized initialization differs from the planned bytes")
    cpu_model: nn.Module = built.model
    if type(cpu_model) is H6CausalTransformer:
        training_transformer = H6TrainingCausalTransformerV3(
            vocabulary=cpu_model.vocabulary,
            profile=cpu_model.profile,
            allow_synthetic_cpu=(type(runtime) is H6SyntheticCpuRuntimeV3),
        )
        loaded = training_transformer.load_state_dict(
            cpu_model.state_dict(),
            strict=True,
        )
        if loaded.missing_keys or loaded.unexpected_keys:
            raise ValueError("training Transformer state inventory differs from A0")
        cpu_model = training_transformer
    recognition = built.recognition_store
    if executable.engine_authority.latent_enabled != (recognition is not None):
        raise ValueError(
            "fresh training module inventory differs from endpoint authority"
        )
    if any(
        tensor.device.type != "cpu"
        for module in (cpu_model, recognition)
        if module is not None
        for tensor in module.state_dict().values()
    ):
        raise ValueError("seed-realized training factories must remain on CPU")
    return built, cpu_model, recognition


@dataclass(frozen=True, slots=True)
class _H6CanonicalCheckpointHydrationPreflightV3:
    """In-memory canonical hydration completed before any catalog write."""

    checkpoint_sha256: str
    built_arm: BuiltArm
    hydrated: H6HydratedCheckpointV3

    def __post_init__(self) -> None:
        _require_digest_v3(
            self.checkpoint_sha256,
            "preflight checkpoint_sha256",
        )
        if (
            type(self.built_arm) is not BuiltArm
            or type(self.hydrated) is not H6HydratedCheckpointV3
            or self.hydrated.checkpoint_sha256 != self.checkpoint_sha256
        ):
            raise ValueError("canonical checkpoint hydration preflight is stale")
        self.hydrated.__post_init__()


def _canonical_checkpoint_hydration_preflight_v3(
    *,
    checkpoint: H6CheckpointV3,
    executable: H6ExecutableAttemptV3,
    runtime: H6RuntimeBindingV3,
) -> _H6CanonicalCheckpointHydrationPreflightV3:
    """Construct, type-bind, and fully hydrate without filesystem effects."""

    if type(checkpoint) is not H6CheckpointV3:
        raise ValueError("canonical hydration requires an exact checkpoint")
    checkpoint.__post_init__()
    if type(executable) is not H6ExecutableAttemptV3:
        raise ValueError("canonical hydration requires an exact executable")
    executable.__post_init__()
    if type(runtime) not in (
        H6InstalledRuntimeBindingV3,
        H6SyntheticCpuRuntimeV3,
    ):
        raise ValueError("canonical hydration requires an exact runtime binding")
    planned = executable.planned_attempt
    built, cpu_model, cpu_recognition = _fresh_cpu_training_modules_v3(
        executable=executable,
        runtime=runtime,
    )
    expected_named_modules: tuple[tuple[str, nn.Module], ...] = (
        (("model", cpu_model),)
        if cpu_recognition is None
        else (
            ("model", cpu_model),
            ("recognition", cpu_recognition),
        )
    )
    used_factories: set[str] = set()

    def closed_factory(
        name: str,
        module: nn.Module,
    ):
        def factory(attempt_spec: object) -> nn.Module:
            if attempt_spec != planned.attempt_spec or name in used_factories:
                raise RuntimeError("closed recovery factory invocation is invalid")
            used_factories.add(name)
            return module

        return factory

    module_factories = tuple(
        (
            name,
            closed_factory(name, module),
        )
        for name, module in expected_named_modules
    )
    factory_authority = _issue_h6_checkpoint_factory_authority_v3(
        attempt_spec=planned.attempt_spec,
        expected_named_modules=expected_named_modules,
        module_factories=module_factories,
    )
    hydrated = hydrate_h6_checkpoint_v3(
        checkpoint,
        expected_attempt_spec=planned.attempt_spec,
        expected_runtime_identity=executable.authorities.config.runtime,
        live_deterministic_policy_sha256=H6_DETERMINISTIC_POLICY_SHA256,
        factory_authority=factory_authority,
        authorized_device=runtime.training_device,
        allow_synthetic_cpu=type(runtime) is H6SyntheticCpuRuntimeV3,
    )
    if used_factories != {name for name, _ in expected_named_modules}:
        raise RuntimeError("canonical hydration left a factory unused")
    return _H6CanonicalCheckpointHydrationPreflightV3(
        checkpoint_sha256=checkpoint.checkpoint_sha256,
        built_arm=built,
        hydrated=hydrated,
    )


def _fresh_training_modules_v3(
    *,
    executable: H6ExecutableAttemptV3,
    runtime: H6RuntimeBindingV3,
) -> tuple[BuiltArm, nn.Module, nn.Module | None]:
    built, cpu_model, recognition = _fresh_cpu_training_modules_v3(
        executable=executable,
        runtime=runtime,
    )
    prepared_model = prepare_training_module_v3(
        cpu_module=cpu_model,
        runtime=runtime,
    ).module
    prepared_recognition = (
        None
        if recognition is None
        else prepare_training_module_v3(
            cpu_module=recognition,
            runtime=runtime,
        ).module
    )
    return built, prepared_model, prepared_recognition


def _reissue_cursor_v3(
    cursor: H6AttemptCursorV3,
    *,
    pass_index: int | None = None,
    batch_index: int | None = None,
    permutation_sha256: str | None = None,
    validation_boundary_delta: int = 0,
) -> H6AttemptCursorV3:
    if type(validation_boundary_delta) is not int:
        raise ValueError("validation boundary delta must be an exact integer")
    return H6AttemptCursorV3.create(
        attempt_spec_sha256=cursor.attempt_spec_sha256,
        pass_index=(cursor.pass_index if pass_index is None else pass_index),
        batch_index=(cursor.batch_index if batch_index is None else batch_index),
        next_phase=cursor.next_phase,
        example_ordinal=cursor.example_ordinal,
        draw_block=cursor.draw_block,
        counter_consumption_sha256=cursor.counter_consumption_sha256,
        permutation_sha256=(
            cursor.permutation_sha256
            if permutation_sha256 is None
            else permutation_sha256
        ),
        recognition_update_count=cursor.recognition_update_count,
        model_update_count=cursor.model_update_count,
        validation_boundary_count=(
            cursor.validation_boundary_count + validation_boundary_delta
        ),
        checkpoint_boundary_count=cursor.checkpoint_boundary_count,
    )


def _initial_cursor_v3(
    executable: H6ExecutableAttemptV3,
    *,
    permutation_sha256: str,
) -> H6AttemptCursorV3:
    return H6AttemptCursorV3.create(
        attempt_spec_sha256=(
            executable.planned_attempt.attempt_spec.attempt_spec_sha256
        ),
        pass_index=0,
        batch_index=0,
        next_phase=(
            TrainingPhase.RECOGNITION_ADAMW
            if executable.engine_authority.latent_enabled
            else TrainingPhase.MODEL_CE_ADAMW
        ),
        example_ordinal=0,
        draw_block=0,
        counter_consumption_sha256=(H6_NO_COUNTER_CONSUMPTION_SHA256),
        permutation_sha256=permutation_sha256,
        recognition_update_count=0,
        model_update_count=0,
        validation_boundary_count=0,
        checkpoint_boundary_count=0,
    )


def _terminal_cursor_matches_plan_v3(
    cursor: H6AttemptCursorV3,
    planned: H6PlannedAttemptV3,
) -> bool:
    return (
        cursor.attempt_spec_sha256 == planned.attempt_spec.attempt_spec_sha256
        and cursor.pass_index == planned.terminal_pass_index
        and cursor.batch_index == planned.terminal_batch_index
        and cursor.example_ordinal == planned.terminal_example_ordinal
        and cursor.draw_block == planned.terminal_draw_block
        and cursor.counter_consumption_sha256
        == planned.terminal_counter_consumption_sha256
        and cursor.permutation_sha256 == planned.terminal_permutation_sha256
        and cursor.recognition_update_count == planned.terminal_recognition_update_count
        and cursor.model_update_count == planned.terminal_model_update_count
        and cursor.validation_boundary_count
        == planned.terminal_validation_boundary_count
        and cursor.checkpoint_boundary_count
        == planned.terminal_checkpoint_boundary_count
    )


def _validate_checkpoint_tuning_cell_v3(
    checkpoint: H6CheckpointV3,
    *,
    executable: H6ExecutableAttemptV3,
) -> None:
    """Refuse another tuning cell before any module state is loaded."""

    observed_cells: set[tuple[float, float]] = set()
    for optimizer in checkpoint.optimizers:
        for group in optimizer.groups:
            values = dict(group.hyperparameters)
            learning_rate = values.get("lr")
            weight_decay = values.get("weight_decay")
            if type(learning_rate) is not float or type(weight_decay) is not float:
                raise ValueError("checkpoint optimizer lacks exact tuning values")
            observed_cells.add((learning_rate, weight_decay))
    if observed_cells != {
        (
            executable.tuning_cell.learning_rate,
            executable.tuning_cell.weight_decay,
        )
    }:
        raise ValueError("checkpoint tuning-cell identity drift")


def _validate_terminal_checkpoint_v3(
    checkpoint: H6CheckpointV3,
    *,
    executable: H6ExecutableAttemptV3,
) -> None:
    if type(checkpoint) is not H6CheckpointV3:
        raise ValueError("terminal checkpoint must be exact H6CheckpointV3")
    checkpoint.__post_init__()
    planned = executable.planned_attempt
    terminal_phase = (
        TrainingPhase.MODEL_ADAMW
        if executable.engine_authority.latent_enabled
        else TrainingPhase.MODEL_CE_ADAMW
    )
    expected_next_phase = (
        TrainingPhase.RECOGNITION_ADAMW
        if executable.engine_authority.latent_enabled
        else TrainingPhase.MODEL_CE_ADAMW
    )
    if (
        checkpoint.attempt_spec != planned.attempt_spec
        or checkpoint.runtime_identity != executable.authorities.config.runtime
        or checkpoint.objective_manifest.phase is not terminal_phase
        or checkpoint.cursor.next_phase is not expected_next_phase
        or checkpoint.detached_batch_snapshot is not None
        or not _terminal_cursor_matches_plan_v3(
            checkpoint.cursor,
            planned,
        )
    ):
        raise ValueError("checkpoint does not match the executable terminal boundary")
    _validate_checkpoint_tuning_cell_v3(
        checkpoint,
        executable=executable,
    )


def _attempt_result_v3(
    *,
    executable: H6ExecutableAttemptV3,
    checkpoint: H6CheckpointV3,
    checkpoint_path: Path,
    progress: H6TrainingAttemptProgressV3,
    history: H6TrainingAttemptHistoryV3,
) -> H6TrainingAttemptResultV3:
    _validate_terminal_checkpoint_v3(
        checkpoint,
        executable=executable,
    )
    raw = checkpoint.to_bytes()
    values = {
        "result_schema": "h6-training-attempt-result-v3",
        "stage": executable.planned_attempt.stage,
        "endpoint_config_id": (executable.planned_attempt.endpoint_config_id),
        "planned_attempt_sha256": (executable.planned_attempt.planned_attempt_sha256),
        "executable_attempt": executable,
        "executable_attempt_sha256": (executable.executable_attempt_sha256),
        "terminal_cursor": checkpoint.cursor,
        "terminal_checkpoint": checkpoint,
        "checkpoint_path": checkpoint_path,
        "terminal_progress": progress,
        "terminal_history": history,
        "progress_path": h6_training_attempt_progress_path_v3(checkpoint_path),
        "progress_sha256": progress.progress_sha256,
        "history_sha256": history.history_sha256,
        "metric_history_count": len(history.metric_history),
        "metric_history_sha256": history.metric_history_sha256,
        "validation_boundary_history_count": len(history.validation_boundary_history),
        "validation_history_sha256": history.validation_history_sha256,
        "checkpoint_bytes_sha256": hashlib.sha256(raw).hexdigest(),
        "checkpoint_byte_count": len(raw),
        "batch_count": checkpoint.cursor.model_update_count,
    }
    provisional = object.__new__(H6TrainingAttemptResultV3)
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    return H6TrainingAttemptResultV3(
        **values,  # type: ignore[arg-type]
        result_sha256=_owned_hash(
            "vfe4.h6.training-attempt-result.v3",
            provisional.canonical_payload(),
        ),
    )


def _is_redirect_v3(path: Path, status: os.stat_result) -> bool:
    if stat.S_ISLNK(status.st_mode):
        return True
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if getattr(status, "st_file_attributes", 0) & reparse_flag:
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(callable(is_junction) and is_junction())


def _publish_checkpoint_v3(
    checkpoint: H6CheckpointV3,
    *,
    path: Path,
    maximum_bytes: int,
) -> H6CheckpointV3:
    if (
        not isinstance(path, Path)
        or not path.is_absolute()
        or type(maximum_bytes) is not int
        or maximum_bytes <= 0
    ):
        raise ValueError(
            "checkpoint publication requires an absolute path and byte bound"
        )
    raw = checkpoint.to_bytes()
    if len(raw) > maximum_bytes:
        raise ValueError("terminal checkpoint exceeds its configured byte bound")
    path.parent.mkdir(parents=True, exist_ok=True)
    parent_status = path.parent.lstat()
    if not stat.S_ISDIR(parent_status.st_mode) or _is_redirect_v3(
        path.parent, parent_status
    ):
        raise ValueError("checkpoint parent is not a safe regular directory")
    if os.path.lexists(path):
        raise FileExistsError("checkpoint destination already exists")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.complete")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o600)
    installed = False
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        if os.name == "nt":
            os.rename(temporary, path)
        else:
            os.link(temporary, path)
            temporary.unlink()
        installed = True
        try:
            directory_descriptor = os.open(
                path.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
        except OSError:
            directory_descriptor = -1
        if directory_descriptor >= 0:
            try:
                os.fsync(directory_descriptor)
            except OSError:
                pass
            finally:
                os.close(directory_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if not installed:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
    return read_h6_checkpoint_file_v3(
        path,
        maximum_bytes=maximum_bytes,
        expected_checkpoint_sha256=checkpoint.checkpoint_sha256,
    )


def h6_training_attempt_progress_path_v3(
    checkpoint_path: Path,
) -> Path:
    """Return the deterministic sibling recovery-catalog path."""

    if (
        not isinstance(checkpoint_path, Path)
        or not checkpoint_path.is_absolute()
        or checkpoint_path.resolve(strict=False) != checkpoint_path
    ):
        raise ValueError("checkpoint_path must be an absolute canonical Path")
    return checkpoint_path.with_name(f"{checkpoint_path.name}.progress-v3.json")


def _recovery_artifact_paths_v3(
    *,
    checkpoint_path: Path,
    ordinal: int,
    boundary_kind: str,
) -> tuple[Path, Path]:
    if type(ordinal) is not int or ordinal < 0:
        raise ValueError("recovery ordinal must be nonnegative")
    if boundary_kind not in (
        "post_recognition",
        "batch_boundary",
        "terminal",
    ):
        raise ValueError("unsupported recovery boundary kind")
    stem = f"{checkpoint_path.name}.recovery-{ordinal:06d}-{boundary_kind}"
    return (
        checkpoint_path.with_name(f"{stem}.h6v3"),
        checkpoint_path.with_name(f"{stem}.history-v3.json"),
    )


def _read_bounded_regular_bytes_v3(
    path: Path,
    *,
    maximum_bytes: int,
    label: str,
) -> bytes:
    if (
        not isinstance(path, Path)
        or not path.is_absolute()
        or type(maximum_bytes) is not int
        or maximum_bytes <= 0
    ):
        raise ValueError(f"{label} read contract is invalid")
    status = path.lstat()
    if (
        not stat.S_ISREG(status.st_mode)
        or _is_redirect_v3(path, status)
        or status.st_size <= 0
        or status.st_size > maximum_bytes
    ):
        raise ValueError(f"{label} is not a bounded regular file")
    with path.open("rb") as handle:
        raw = handle.read(maximum_bytes + 1)
    if len(raw) != status.st_size or len(raw) > maximum_bytes:
        raise ValueError(f"{label} changed during bounded read")
    return raw


def _publish_immutable_bytes_v3(
    raw: bytes,
    *,
    path: Path,
    maximum_bytes: int,
    label: str,
) -> bytes:
    if (
        type(raw) is not bytes
        or not raw
        or len(raw) > maximum_bytes
        or not path.is_absolute()
    ):
        raise ValueError(f"{label} publication contract is invalid")
    path.parent.mkdir(parents=True, exist_ok=True)
    parent_status = path.parent.lstat()
    if not stat.S_ISDIR(parent_status.st_mode) or _is_redirect_v3(
        path.parent, parent_status
    ):
        raise ValueError(f"{label} parent is not a safe regular directory")
    if os.path.lexists(path):
        reopened = _read_bounded_regular_bytes_v3(
            path,
            maximum_bytes=maximum_bytes,
            label=label,
        )
        if reopened != raw:
            raise FileExistsError(f"{label} destination contains different bytes")
        return reopened
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.complete")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o600)
    installed = False
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        if os.name == "nt":
            os.rename(temporary, path)
        else:
            os.link(temporary, path)
            temporary.unlink()
        installed = True
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if not installed:
            temporary.unlink(missing_ok=True)
    reopened = _read_bounded_regular_bytes_v3(
        path,
        maximum_bytes=maximum_bytes,
        label=label,
    )
    if reopened != raw:
        raise RuntimeError(f"{label} bytes changed during publication")
    return reopened


def _publish_progress_catalog_v3(
    progress: H6TrainingAttemptProgressV3,
    *,
    path: Path,
    maximum_bytes: int,
    expected_prior_sha256: str | None,
) -> H6TrainingAttemptProgressV3:
    progress.__post_init__()
    raw = progress.to_bytes()
    if len(raw) > maximum_bytes:
        raise ValueError("attempt progress catalog exceeds its byte bound")
    path.parent.mkdir(parents=True, exist_ok=True)
    parent_status = path.parent.lstat()
    if not stat.S_ISDIR(parent_status.st_mode) or _is_redirect_v3(
        path.parent, parent_status
    ):
        raise ValueError("attempt progress parent is not a safe regular directory")
    if os.path.lexists(path):
        prior = read_h6_training_attempt_progress_v3(
            path=path,
            maximum_bytes=maximum_bytes,
        )
        if (
            expected_prior_sha256 is None
            or prior.progress_sha256 != expected_prior_sha256
            or progress.boundaries[:-1] != prior.boundaries
        ):
            raise RuntimeError(
                "attempt progress catalog is not an append-only extension"
            )
    elif expected_prior_sha256 is not None:
        raise RuntimeError("attempt progress catalog disappeared")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.complete")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o600)
    installed = False
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        installed = True
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if not installed:
            temporary.unlink(missing_ok=True)
    reopened = read_h6_training_attempt_progress_v3(
        path=path,
        maximum_bytes=maximum_bytes,
    )
    if reopened != progress:
        raise RuntimeError("attempt progress catalog changed after publication")
    return reopened


def _exact_json_mapping_v3(
    value: object,
    *,
    keys: tuple[str, ...],
    label: str,
) -> dict[str, object]:
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(f"{label} has a noncanonical field inventory")
    return value


def _json_float_v3(value: object, name: str) -> float:
    if type(value) is not str:
        raise ValueError(f"{name} must be a canonical hexadecimal float")
    try:
        result = float.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a canonical hexadecimal float") from exc
    if result.hex() != value:
        raise ValueError(f"{name} hexadecimal float is not canonical")
    return result


def _phase_record_from_json_v3(value: object) -> H6PhaseRecordV3:
    payload = _exact_json_mapping_v3(
        value,
        keys=(
            "phase",
            "objective_kind",
            "is_elbo",
            "partitions",
            "noise_sha256",
            "objective_value",
            "loss_value",
            "gradient_norm",
        ),
        label="phase record",
    )
    partitions = payload["partitions"]
    if type(partitions) is not list or any(
        type(partition) is not str for partition in partitions
    ):
        raise ValueError("phase record partitions are invalid")
    try:
        phase = TrainingPhase(payload["phase"])
    except (TypeError, ValueError) as exc:
        raise ValueError("phase record phase is invalid") from exc
    return H6PhaseRecordV3(
        phase=phase,
        objective_kind=payload["objective_kind"],  # type: ignore[arg-type]
        is_elbo=payload["is_elbo"],  # type: ignore[arg-type]
        partitions=tuple(partitions),
        noise_sha256=payload["noise_sha256"],  # type: ignore[arg-type]
        objective_value=_json_float_v3(
            payload["objective_value"],
            "objective_value",
        ),
        loss_value=_json_float_v3(payload["loss_value"], "loss_value"),
        gradient_norm=_json_float_v3(
            payload["gradient_norm"],
            "gradient_norm",
        ),
    )


def _metric_record_from_json_v3(value: object) -> H6MetricRecordV3:
    payload = _exact_json_mapping_v3(
        value,
        keys=(
            "ordinal",
            "phase",
            "recognition_update_count",
            "model_update_count",
            "objective_value",
            "loss_value",
            "gradient_norm",
            "metric_sha256",
        ),
        label="metric record",
    )
    try:
        phase = TrainingPhase(payload["phase"])
    except (TypeError, ValueError) as exc:
        raise ValueError("metric record phase is invalid") from exc
    return H6MetricRecordV3(
        ordinal=payload["ordinal"],  # type: ignore[arg-type]
        phase=phase,
        recognition_update_count=payload[  # type: ignore[arg-type]
            "recognition_update_count"
        ],
        model_update_count=payload["model_update_count"],  # type: ignore[arg-type]
        objective_value=_json_float_v3(
            payload["objective_value"],
            "metric objective_value",
        ),
        loss_value=_json_float_v3(
            payload["loss_value"],
            "metric loss_value",
        ),
        gradient_norm=_json_float_v3(
            payload["gradient_norm"],
            "metric gradient_norm",
        ),
        metric_sha256=payload["metric_sha256"],  # type: ignore[arg-type]
    )


def _attempt_metric_from_json_v3(
    value: object,
) -> H6AttemptMetricHistoryRecordV3:
    payload = _exact_json_mapping_v3(
        value,
        keys=(
            "ordinal",
            "pass_index",
            "batch_index",
            "phase_record",
            "metric_record",
            "record_sha256",
        ),
        label="attempt metric record",
    )
    return H6AttemptMetricHistoryRecordV3(
        ordinal=payload["ordinal"],  # type: ignore[arg-type]
        pass_index=payload["pass_index"],  # type: ignore[arg-type]
        batch_index=payload["batch_index"],  # type: ignore[arg-type]
        phase_record=_phase_record_from_json_v3(payload["phase_record"]),
        metric_record=_metric_record_from_json_v3(payload["metric_record"]),
        record_sha256=payload["record_sha256"],  # type: ignore[arg-type]
    )


def _validation_record_from_json_v3(
    value: object,
) -> H6ValidationBoundaryHistoryRecordV3:
    payload = _exact_json_mapping_v3(
        value,
        keys=(
            "ordinal",
            "pass_index",
            "completed_batch_count",
            "cursor_sha256",
            "checkpoint_sha256",
            "metric_history_sha256",
            "contract_sha256",
            "record_sha256",
        ),
        label="validation-boundary record",
    )
    return H6ValidationBoundaryHistoryRecordV3(
        ordinal=payload["ordinal"],  # type: ignore[arg-type]
        pass_index=payload["pass_index"],  # type: ignore[arg-type]
        completed_batch_count=payload[  # type: ignore[arg-type]
            "completed_batch_count"
        ],
        cursor_sha256=payload["cursor_sha256"],  # type: ignore[arg-type]
        checkpoint_sha256=payload["checkpoint_sha256"],  # type: ignore[arg-type]
        metric_history_sha256=payload[  # type: ignore[arg-type]
            "metric_history_sha256"
        ],
        contract_sha256=payload["contract_sha256"],  # type: ignore[arg-type]
        record_sha256=payload["record_sha256"],  # type: ignore[arg-type]
    )


def _history_shard_from_bytes_v3(raw: bytes) -> H6AttemptHistoryShardV3:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("attempt history shard is not canonical JSON") from exc
    if canonical_json_bytes(value) != raw:
        raise ValueError("attempt history shard JSON bytes are not canonical")
    payload = _exact_json_mapping_v3(
        value,
        keys=(
            "shard_schema",
            "attempt_spec_sha256",
            "boundary_ordinal",
            "prior_boundary_sha256",
            "metric_records",
            "validation_records",
            "resume_phase_records",
            "resume_metric_records",
            "resume_checkpoint_phases",
            "resume_result_sha256",
            "shard_sha256",
        ),
        label="attempt history shard",
    )
    for name in (
        "metric_records",
        "validation_records",
        "resume_phase_records",
        "resume_metric_records",
        "resume_checkpoint_phases",
    ):
        if type(payload[name]) is not list:
            raise ValueError(f"attempt history shard {name} is invalid")
    try:
        resume_checkpoint_phases = tuple(
            TrainingPhase(value)
            for value in payload["resume_checkpoint_phases"]  # type: ignore[union-attr]
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("attempt history shard checkpoint phase is invalid") from exc
    return H6AttemptHistoryShardV3(
        shard_schema=payload["shard_schema"],  # type: ignore[arg-type]
        attempt_spec_sha256=payload[  # type: ignore[arg-type]
            "attempt_spec_sha256"
        ],
        boundary_ordinal=payload["boundary_ordinal"],  # type: ignore[arg-type]
        prior_boundary_sha256=payload[  # type: ignore[arg-type]
            "prior_boundary_sha256"
        ],
        metric_records=tuple(
            _attempt_metric_from_json_v3(item)
            for item in payload["metric_records"]  # type: ignore[union-attr]
        ),
        validation_records=tuple(
            _validation_record_from_json_v3(item)
            for item in payload["validation_records"]  # type: ignore[union-attr]
        ),
        resume_phase_records=tuple(
            _phase_record_from_json_v3(item)
            for item in payload["resume_phase_records"]  # type: ignore[union-attr]
        ),
        resume_metric_records=tuple(
            _metric_record_from_json_v3(item)
            for item in payload["resume_metric_records"]  # type: ignore[union-attr]
        ),
        resume_checkpoint_phases=resume_checkpoint_phases,
        resume_result_sha256=payload["resume_result_sha256"],  # type: ignore[arg-type]
        shard_sha256=payload["shard_sha256"],  # type: ignore[arg-type]
    )


def _boundary_from_json_v3(
    value: object,
) -> H6AttemptRecoveryBoundaryV3:
    payload = _exact_json_mapping_v3(
        value,
        keys=(
            "boundary_kind",
            "ordinal",
            "checkpoint_filename",
            "checkpoint_sha256",
            "checkpoint_bytes_sha256",
            "checkpoint_byte_count",
            "history_filename",
            "history_sha256",
            "history_bytes_sha256",
            "history_byte_count",
            "cursor_sha256",
            "next_phase",
            "metric_history_count",
            "metric_history_sha256",
            "validation_history_count",
            "validation_history_sha256",
            "boundary_sha256",
        ),
        label="attempt recovery boundary",
    )
    try:
        next_phase = TrainingPhase(payload["next_phase"])
    except (TypeError, ValueError) as exc:
        raise ValueError("recovery boundary phase is invalid") from exc
    return H6AttemptRecoveryBoundaryV3(
        boundary_kind=payload["boundary_kind"],  # type: ignore[arg-type]
        ordinal=payload["ordinal"],  # type: ignore[arg-type]
        checkpoint_filename=payload["checkpoint_filename"],  # type: ignore[arg-type]
        checkpoint_sha256=payload["checkpoint_sha256"],  # type: ignore[arg-type]
        checkpoint_bytes_sha256=payload[  # type: ignore[arg-type]
            "checkpoint_bytes_sha256"
        ],
        checkpoint_byte_count=payload[  # type: ignore[arg-type]
            "checkpoint_byte_count"
        ],
        history_filename=payload["history_filename"],  # type: ignore[arg-type]
        history_sha256=payload["history_sha256"],  # type: ignore[arg-type]
        history_bytes_sha256=payload[  # type: ignore[arg-type]
            "history_bytes_sha256"
        ],
        history_byte_count=payload["history_byte_count"],  # type: ignore[arg-type]
        cursor_sha256=payload["cursor_sha256"],  # type: ignore[arg-type]
        next_phase=next_phase,
        metric_history_count=payload[  # type: ignore[arg-type]
            "metric_history_count"
        ],
        metric_history_sha256=payload[  # type: ignore[arg-type]
            "metric_history_sha256"
        ],
        validation_history_count=payload[  # type: ignore[arg-type]
            "validation_history_count"
        ],
        validation_history_sha256=payload[  # type: ignore[arg-type]
            "validation_history_sha256"
        ],
        boundary_sha256=payload["boundary_sha256"],  # type: ignore[arg-type]
    )


def read_h6_training_attempt_progress_v3(
    *,
    path: Path,
    maximum_bytes: int,
) -> H6TrainingAttemptProgressV3:
    """Reopen and authenticate the small recovery catalog only."""

    raw = _read_bounded_regular_bytes_v3(
        path,
        maximum_bytes=maximum_bytes,
        label="attempt progress catalog",
    )
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("attempt progress catalog is not canonical JSON") from exc
    if canonical_json_bytes(value) != raw:
        raise ValueError("attempt progress catalog JSON bytes are not canonical")
    payload = _exact_json_mapping_v3(
        value,
        keys=(
            "progress_schema",
            "attempt_spec_sha256",
            "planned_attempt_sha256",
            "executable_attempt_sha256",
            "tuning_cell_sha256",
            "model_factory_sha256",
            "recognition_factory_sha256",
            "initialization_sha256",
            "runtime_identity_sha256",
            "deterministic_policy_sha256",
            "terminal_checkpoint_filename",
            "boundaries",
            "metric_history_count",
            "metric_history_sha256",
            "validation_history_count",
            "validation_history_sha256",
            "progress_sha256",
        ),
        label="attempt progress catalog",
    )
    if type(payload["boundaries"]) is not list:
        raise ValueError("attempt progress boundary inventory is invalid")
    return H6TrainingAttemptProgressV3(
        progress_schema=payload["progress_schema"],  # type: ignore[arg-type]
        attempt_spec_sha256=payload[  # type: ignore[arg-type]
            "attempt_spec_sha256"
        ],
        planned_attempt_sha256=payload[  # type: ignore[arg-type]
            "planned_attempt_sha256"
        ],
        executable_attempt_sha256=payload[  # type: ignore[arg-type]
            "executable_attempt_sha256"
        ],
        tuning_cell_sha256=payload["tuning_cell_sha256"],  # type: ignore[arg-type]
        model_factory_sha256=payload["model_factory_sha256"],  # type: ignore[arg-type]
        recognition_factory_sha256=payload[  # type: ignore[arg-type]
            "recognition_factory_sha256"
        ],
        initialization_sha256=payload[  # type: ignore[arg-type]
            "initialization_sha256"
        ],
        runtime_identity_sha256=payload[  # type: ignore[arg-type]
            "runtime_identity_sha256"
        ],
        deterministic_policy_sha256=payload[  # type: ignore[arg-type]
            "deterministic_policy_sha256"
        ],
        terminal_checkpoint_filename=payload[  # type: ignore[arg-type]
            "terminal_checkpoint_filename"
        ],
        boundaries=tuple(
            _boundary_from_json_v3(item)
            for item in payload["boundaries"]  # type: ignore[union-attr]
        ),
        metric_history_count=payload[  # type: ignore[arg-type]
            "metric_history_count"
        ],
        metric_history_sha256=payload[  # type: ignore[arg-type]
            "metric_history_sha256"
        ],
        validation_history_count=payload[  # type: ignore[arg-type]
            "validation_history_count"
        ],
        validation_history_sha256=payload[  # type: ignore[arg-type]
            "validation_history_sha256"
        ],
        progress_sha256=payload["progress_sha256"],  # type: ignore[arg-type]
    )


def _read_history_shard_for_boundary_v3(
    *,
    parent: Path,
    boundary: H6AttemptRecoveryBoundaryV3,
    maximum_bytes: int,
) -> H6AttemptHistoryShardV3:
    path = parent / boundary.history_filename
    raw = _read_bounded_regular_bytes_v3(
        path,
        maximum_bytes=maximum_bytes,
        label="attempt history shard",
    )
    if (
        len(raw) != boundary.history_byte_count
        or hashlib.sha256(raw).hexdigest() != boundary.history_bytes_sha256
    ):
        raise ValueError("attempt history shard bytes differ from catalog")
    shard = _history_shard_from_bytes_v3(raw)
    if (
        shard.shard_sha256 != boundary.history_sha256
        or shard.boundary_ordinal != boundary.ordinal
    ):
        raise ValueError("attempt history shard differs from its boundary")
    return shard


def read_h6_training_attempt_history_v3(
    *,
    checkpoint_path: Path,
    maximum_bytes: int,
) -> H6TrainingAttemptHistoryV3:
    """Explicitly materialize all append-only metric/validation shards."""

    progress_path = h6_training_attempt_progress_path_v3(checkpoint_path)
    progress = read_h6_training_attempt_progress_v3(
        path=progress_path,
        maximum_bytes=maximum_bytes,
    )
    metric_history: list[H6AttemptMetricHistoryRecordV3] = []
    validation_history: list[H6ValidationBoundaryHistoryRecordV3] = []
    metric_sha256 = _EMPTY_METRIC_HISTORY_SHA256_V3
    validation_sha256 = _EMPTY_VALIDATION_HISTORY_SHA256_V3
    prior_boundary_sha256: str | None = None
    for boundary in progress.boundaries:
        shard = _read_history_shard_for_boundary_v3(
            parent=progress_path.parent,
            boundary=boundary,
            maximum_bytes=maximum_bytes,
        )
        if (
            shard.attempt_spec_sha256 != progress.attempt_spec_sha256
            or shard.prior_boundary_sha256 != prior_boundary_sha256
        ):
            raise ValueError("attempt history shard chain is broken")
        expected_metric_ordinal = len(metric_history)
        for record in shard.metric_records:
            if record.ordinal != expected_metric_ordinal:
                raise ValueError("attempt metric history is not append-only")
            metric_history.append(record)
            expected_metric_ordinal += 1
        metric_sha256 = _history_chain_sha256_v3(
            domain="vfe4.h6.attempt-metric-history.v3",
            prior_sha256=metric_sha256,
            prior_count=len(metric_history) - len(shard.metric_records),
            appended_sha256s=tuple(
                record.record_sha256 for record in shard.metric_records
            ),
        )
        expected_validation_ordinal = len(validation_history)
        for record in shard.validation_records:
            if record.ordinal != expected_validation_ordinal:
                raise ValueError("attempt validation history is not append-only")
            validation_history.append(record)
            expected_validation_ordinal += 1
        validation_sha256 = _history_chain_sha256_v3(
            domain="vfe4.h6.attempt-validation-history.v3",
            prior_sha256=validation_sha256,
            prior_count=(len(validation_history) - len(shard.validation_records)),
            appended_sha256s=tuple(
                record.record_sha256 for record in shard.validation_records
            ),
        )
        if (
            len(metric_history) != boundary.metric_history_count
            or metric_sha256 != boundary.metric_history_sha256
            or len(validation_history) != boundary.validation_history_count
            or validation_sha256 != boundary.validation_history_sha256
        ):
            raise ValueError("attempt history aggregate differs from catalog")
        prior_boundary_sha256 = boundary.boundary_sha256
    values = {
        "attempt_spec_sha256": progress.attempt_spec_sha256,
        "metric_history": tuple(metric_history),
        "validation_boundary_history": tuple(validation_history),
        "metric_history_sha256": metric_sha256,
        "validation_history_sha256": validation_sha256,
    }
    provisional = object.__new__(H6TrainingAttemptHistoryV3)
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    return H6TrainingAttemptHistoryV3(
        **values,
        history_sha256=_owned_hash(
            "vfe4.h6.training-attempt-history.v3",
            provisional.canonical_payload(),
        ),
    )


def _validate_recovery_inventory_v3(
    *,
    progress_path: Path,
    progress: H6TrainingAttemptProgressV3,
    executable: H6ExecutableAttemptV3,
    maximum_bytes: int,
) -> H6TrainingAttemptHistoryV3:
    """Authenticate every immutable checkpoint/history boundary in order."""

    terminal_checkpoint_path = progress_path.parent / (
        progress.terminal_checkpoint_filename
    )
    prior_cursor: H6AttemptCursorV3 | None = None
    for boundary in progress.boundaries:
        expected_checkpoint_path, expected_history_path = _recovery_artifact_paths_v3(
            checkpoint_path=terminal_checkpoint_path,
            ordinal=boundary.ordinal,
            boundary_kind=boundary.boundary_kind,
        )
        if (
            expected_checkpoint_path.name != boundary.checkpoint_filename
            or expected_history_path.name != boundary.history_filename
        ):
            raise RuntimeError("recovery boundary path inventory is not deterministic")
        checkpoint = read_h6_checkpoint_file_v3(
            expected_checkpoint_path,
            maximum_bytes=maximum_bytes,
            expected_checkpoint_sha256=boundary.checkpoint_sha256,
        )
        if (
            checkpoint.attempt_spec != executable.planned_attempt.attempt_spec
            or checkpoint.runtime_identity != executable.authorities.config.runtime
            or checkpoint.objective_manifest.endpoint_config_sha256
            != executable.endpoint_config.config_sha256
            or checkpoint.objective_manifest.objective_kind
            != executable.endpoint_config.objective_kind
        ):
            raise RuntimeError(
                "recovery checkpoint differs from executable authorities"
            )
        _validate_checkpoint_tuning_cell_v3(
            checkpoint,
            executable=executable,
        )
        raw = checkpoint.to_bytes()
        if (
            len(raw) != boundary.checkpoint_byte_count
            or hashlib.sha256(raw).hexdigest() != boundary.checkpoint_bytes_sha256
            or checkpoint.cursor.cursor_sha256 != boundary.cursor_sha256
            or checkpoint.cursor.next_phase is not boundary.next_phase
            or checkpoint.attempt_spec.attempt_spec_sha256
            != progress.attempt_spec_sha256
            or checkpoint.runtime_identity.runtime_identity_sha256
            != progress.runtime_identity_sha256
            or checkpoint.deterministic_policy_sha256
            != progress.deterministic_policy_sha256
            or checkpoint.attempt_spec.model_factory_sha256
            != progress.model_factory_sha256
            or checkpoint.attempt_spec.recognition_factory_sha256
            != progress.recognition_factory_sha256
            or checkpoint.attempt_spec.initialization_sha256
            != progress.initialization_sha256
            or boundary.metric_history_count
            != (
                checkpoint.cursor.recognition_update_count
                + checkpoint.cursor.model_update_count
            )
            or boundary.validation_history_count
            != checkpoint.cursor.validation_boundary_count
        ):
            raise RuntimeError("recovery checkpoint inventory differs from its catalog")
        if prior_cursor is not None and (
            checkpoint.cursor.recognition_update_count
            < prior_cursor.recognition_update_count
            or checkpoint.cursor.model_update_count < prior_cursor.model_update_count
            or checkpoint.cursor.validation_boundary_count
            < prior_cursor.validation_boundary_count
            or checkpoint.cursor.checkpoint_boundary_count
            < prior_cursor.checkpoint_boundary_count
        ):
            raise RuntimeError("recovery checkpoint cursor order regressed")
        prior_cursor = checkpoint.cursor
    history = read_h6_training_attempt_history_v3(
        checkpoint_path=terminal_checkpoint_path,
        maximum_bytes=maximum_bytes,
    )
    if (
        history.attempt_spec_sha256 != progress.attempt_spec_sha256
        or len(history.metric_history) != progress.metric_history_count
        or history.metric_history_sha256 != progress.metric_history_sha256
        or len(history.validation_boundary_history) != progress.validation_history_count
        or history.validation_history_sha256 != progress.validation_history_sha256
    ):
        raise RuntimeError("reopened attempt history differs from the recovery catalog")
    return history


def _objective_manifest_v3(
    *,
    executable: H6ExecutableAttemptV3,
    cursor: H6AttemptCursorV3,
    callbacks: _H6BatchCallbacksV3,
    detached_snapshot_sha256: str | None = None,
) -> H6ObjectiveManifestV3:
    if (
        callbacks.latest_phase is None
        or callbacks.latest_total_raw_bytes_sha256 is None
        or not callbacks.latest_factor_bindings
    ):
        raise ValueError("terminal batch lacks objective evidence")
    return H6ObjectiveManifestV3.create(
        attempt_spec_sha256=cursor.attempt_spec_sha256,
        endpoint_config_sha256=(executable.endpoint_config.config_sha256),
        objective_kind=executable.endpoint_config.objective_kind,
        phase=callbacks.latest_phase,
        recognition_estimator_sha256=(
            executable.planned_attempt.attempt_spec.recognition_estimator_sha256
        ),
        counter_consumption_sha256=(cursor.counter_consumption_sha256),
        recognition_law_sha256=(callbacks.latest_recognition_law_sha256),
        detached_snapshot_sha256=(
            callbacks.latest_detached_snapshot_sha256
            if detached_snapshot_sha256 is None
            else detached_snapshot_sha256
        ),
        ordered_factor_bindings=callbacks.latest_factor_bindings,
        total_raw_bytes_sha256=(callbacks.latest_total_raw_bytes_sha256),
    )


def _new_history_shard_v3(
    *,
    attempt_spec_sha256: str,
    boundary_ordinal: int,
    prior_boundary_sha256: str | None,
    metric_records: tuple[H6AttemptMetricHistoryRecordV3, ...],
    validation_records: tuple[H6ValidationBoundaryHistoryRecordV3, ...],
    resume_state: H6TrainingBatchResultV3 | None,
) -> H6AttemptHistoryShardV3:
    values = {
        "shard_schema": "h6-attempt-history-shard-v3",
        "attempt_spec_sha256": attempt_spec_sha256,
        "boundary_ordinal": boundary_ordinal,
        "prior_boundary_sha256": prior_boundary_sha256,
        "metric_records": metric_records,
        "validation_records": validation_records,
        "resume_phase_records": (
            () if resume_state is None else resume_state.phase_records
        ),
        "resume_metric_records": (
            () if resume_state is None else resume_state.metric_records
        ),
        "resume_checkpoint_phases": (
            () if resume_state is None else resume_state.checkpoint_phases
        ),
        "resume_result_sha256": (
            None if resume_state is None else resume_state.result_sha256
        ),
    }
    provisional = object.__new__(H6AttemptHistoryShardV3)
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    return H6AttemptHistoryShardV3(
        **values,  # type: ignore[arg-type]
        shard_sha256=_owned_hash(
            "vfe4.h6.attempt-history-shard.v3",
            provisional.canonical_payload(),
        ),
    )


def _new_recovery_boundary_v3(
    *,
    boundary_kind: Literal[
        "post_recognition",
        "batch_boundary",
        "terminal",
    ],
    ordinal: int,
    checkpoint_path: Path,
    checkpoint: H6CheckpointV3,
    history_path: Path,
    history_shard: H6AttemptHistoryShardV3,
    metric_history_count: int,
    metric_history_sha256: str,
    validation_history_count: int,
    validation_history_sha256: str,
) -> H6AttemptRecoveryBoundaryV3:
    checkpoint_raw = checkpoint.to_bytes()
    history_raw = history_shard.to_bytes()
    values = {
        "boundary_kind": boundary_kind,
        "ordinal": ordinal,
        "checkpoint_filename": checkpoint_path.name,
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "checkpoint_bytes_sha256": hashlib.sha256(checkpoint_raw).hexdigest(),
        "checkpoint_byte_count": len(checkpoint_raw),
        "history_filename": history_path.name,
        "history_sha256": history_shard.shard_sha256,
        "history_bytes_sha256": hashlib.sha256(history_raw).hexdigest(),
        "history_byte_count": len(history_raw),
        "cursor_sha256": checkpoint.cursor.cursor_sha256,
        "next_phase": checkpoint.cursor.next_phase,
        "metric_history_count": metric_history_count,
        "metric_history_sha256": metric_history_sha256,
        "validation_history_count": validation_history_count,
        "validation_history_sha256": validation_history_sha256,
    }
    provisional = object.__new__(H6AttemptRecoveryBoundaryV3)
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    return H6AttemptRecoveryBoundaryV3(
        **values,
        boundary_sha256=_owned_hash(
            "vfe4.h6.attempt-recovery-boundary.v3",
            provisional.canonical_payload(),
        ),
    )


def _validate_progress_identity_v3(
    progress: H6TrainingAttemptProgressV3,
    *,
    executable: H6ExecutableAttemptV3,
    checkpoint_path: Path,
) -> None:
    progress.__post_init__()
    planned = executable.planned_attempt
    spec = planned.attempt_spec
    if (
        progress.attempt_spec_sha256 != spec.attempt_spec_sha256
        or progress.planned_attempt_sha256 != planned.planned_attempt_sha256
        or progress.executable_attempt_sha256 != executable.executable_attempt_sha256
        or progress.tuning_cell_sha256 != executable.tuning_cell.cell_sha256
        or progress.model_factory_sha256 != spec.model_factory_sha256
        or progress.recognition_factory_sha256 != spec.recognition_factory_sha256
        or progress.initialization_sha256 != spec.initialization_sha256
        or progress.runtime_identity_sha256
        != executable.authorities.config.runtime.runtime_identity_sha256
        or progress.deterministic_policy_sha256 != H6_DETERMINISTIC_POLICY_SHA256
        or progress.terminal_checkpoint_filename != checkpoint_path.name
    ):
        raise RuntimeError(
            "attempt progress differs from the exact executable authority"
        )


def _new_progress_v3(
    *,
    executable: H6ExecutableAttemptV3,
    checkpoint_path: Path,
    prior: H6TrainingAttemptProgressV3 | None,
    boundary: H6AttemptRecoveryBoundaryV3,
) -> H6TrainingAttemptProgressV3:
    if prior is not None:
        _validate_progress_identity_v3(
            prior,
            executable=executable,
            checkpoint_path=checkpoint_path,
        )
        if boundary.ordinal != len(prior.boundaries):
            raise ValueError("attempt recovery boundary ordinal is not append-only")
        boundaries = (*prior.boundaries, boundary)
    else:
        if boundary.ordinal != 0:
            raise ValueError("first attempt recovery boundary must be ordinal zero")
        boundaries = (boundary,)
    spec = executable.planned_attempt.attempt_spec
    values = {
        "progress_schema": "h6-training-attempt-progress-v3",
        "attempt_spec_sha256": spec.attempt_spec_sha256,
        "planned_attempt_sha256": (executable.planned_attempt.planned_attempt_sha256),
        "executable_attempt_sha256": (executable.executable_attempt_sha256),
        "tuning_cell_sha256": executable.tuning_cell.cell_sha256,
        "model_factory_sha256": spec.model_factory_sha256,
        "recognition_factory_sha256": (spec.recognition_factory_sha256),
        "initialization_sha256": spec.initialization_sha256,
        "runtime_identity_sha256": (
            executable.authorities.config.runtime.runtime_identity_sha256
        ),
        "deterministic_policy_sha256": (H6_DETERMINISTIC_POLICY_SHA256),
        "terminal_checkpoint_filename": checkpoint_path.name,
        "boundaries": boundaries,
        "metric_history_count": boundary.metric_history_count,
        "metric_history_sha256": boundary.metric_history_sha256,
        "validation_history_count": boundary.validation_history_count,
        "validation_history_sha256": (boundary.validation_history_sha256),
    }
    provisional = object.__new__(H6TrainingAttemptProgressV3)
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    return H6TrainingAttemptProgressV3(
        **values,  # type: ignore[arg-type]
        progress_sha256=_owned_hash(
            "vfe4.h6.training-attempt-progress.v3",
            provisional.canonical_payload(),
        ),
    )


def _persist_attempt_boundary_v3(
    *,
    executable: H6ExecutableAttemptV3,
    terminal_checkpoint_path: Path,
    maximum_bytes: int,
    prior_progress: H6TrainingAttemptProgressV3 | None,
    boundary_kind: Literal[
        "post_recognition",
        "batch_boundary",
        "terminal",
    ],
    checkpoint: H6CheckpointV3,
    metric_records: tuple[H6AttemptMetricHistoryRecordV3, ...],
    validation_pass_index: int | None,
    validation_completed_batch_count: int | None,
    resume_state: H6TrainingBatchResultV3 | None,
) -> tuple[
    H6TrainingAttemptProgressV3,
    H6CheckpointV3,
]:
    """Publish checkpoint, history shard, then append the small catalog."""

    ordinal = 0 if prior_progress is None else len(prior_progress.boundaries)
    recovery_checkpoint_path, history_path = _recovery_artifact_paths_v3(
        checkpoint_path=terminal_checkpoint_path,
        ordinal=ordinal,
        boundary_kind=boundary_kind,
    )
    prior_metric_count = (
        0 if prior_progress is None else prior_progress.metric_history_count
    )
    prior_metric_sha256 = (
        _EMPTY_METRIC_HISTORY_SHA256_V3
        if prior_progress is None
        else prior_progress.metric_history_sha256
    )
    if tuple(record.ordinal for record in metric_records) != tuple(
        range(
            prior_metric_count,
            prior_metric_count + len(metric_records),
        )
    ):
        raise ValueError("attempt metric delta is not append-only")
    metric_history_sha256 = _history_chain_sha256_v3(
        domain="vfe4.h6.attempt-metric-history.v3",
        prior_sha256=prior_metric_sha256,
        prior_count=prior_metric_count,
        appended_sha256s=tuple(record.record_sha256 for record in metric_records),
    )
    metric_history_count = prior_metric_count + len(metric_records)
    prior_validation_count = (
        0 if prior_progress is None else prior_progress.validation_history_count
    )
    prior_validation_sha256 = (
        _EMPTY_VALIDATION_HISTORY_SHA256_V3
        if prior_progress is None
        else prior_progress.validation_history_sha256
    )
    if (validation_pass_index is None) != (validation_completed_batch_count is None):
        raise ValueError("validation-boundary coordinates are incomplete")
    validation_records: tuple[H6ValidationBoundaryHistoryRecordV3, ...]
    if validation_pass_index is None:
        validation_records = ()
    else:
        validation_records = (
            H6ValidationBoundaryHistoryRecordV3.create(
                ordinal=prior_validation_count,
                pass_index=validation_pass_index,
                completed_batch_count=validation_completed_batch_count,
                cursor_sha256=checkpoint.cursor.cursor_sha256,
                checkpoint_sha256=checkpoint.checkpoint_sha256,
                metric_history_sha256=metric_history_sha256,
            ),
        )
    validation_history_sha256 = _history_chain_sha256_v3(
        domain="vfe4.h6.attempt-validation-history.v3",
        prior_sha256=prior_validation_sha256,
        prior_count=prior_validation_count,
        appended_sha256s=tuple(record.record_sha256 for record in validation_records),
    )
    validation_history_count = prior_validation_count + len(validation_records)
    shard = _new_history_shard_v3(
        attempt_spec_sha256=(
            executable.planned_attempt.attempt_spec.attempt_spec_sha256
        ),
        boundary_ordinal=ordinal,
        prior_boundary_sha256=(
            None
            if prior_progress is None
            else prior_progress.latest_boundary.boundary_sha256
        ),
        metric_records=metric_records,
        validation_records=validation_records,
        resume_state=resume_state,
    )
    try:
        reopened_checkpoint = _publish_checkpoint_v3(
            checkpoint,
            path=recovery_checkpoint_path,
            maximum_bytes=maximum_bytes,
        )
    except FileExistsError:
        reopened_checkpoint = read_h6_checkpoint_file_v3(
            recovery_checkpoint_path,
            maximum_bytes=maximum_bytes,
            expected_checkpoint_sha256=checkpoint.checkpoint_sha256,
        )
    history_raw = shard.to_bytes()
    _publish_immutable_bytes_v3(
        history_raw,
        path=history_path,
        maximum_bytes=maximum_bytes,
        label="attempt history shard",
    )
    boundary = _new_recovery_boundary_v3(
        boundary_kind=boundary_kind,
        ordinal=ordinal,
        checkpoint_path=recovery_checkpoint_path,
        checkpoint=reopened_checkpoint,
        history_path=history_path,
        history_shard=shard,
        metric_history_count=metric_history_count,
        metric_history_sha256=metric_history_sha256,
        validation_history_count=validation_history_count,
        validation_history_sha256=validation_history_sha256,
    )
    progress = _new_progress_v3(
        executable=executable,
        checkpoint_path=terminal_checkpoint_path,
        prior=prior_progress,
        boundary=boundary,
    )
    progress = _publish_progress_catalog_v3(
        progress,
        path=h6_training_attempt_progress_path_v3(terminal_checkpoint_path),
        maximum_bytes=maximum_bytes,
        expected_prior_sha256=(
            None if prior_progress is None else prior_progress.progress_sha256
        ),
    )
    return progress, reopened_checkpoint


@dataclass(frozen=True, slots=True)
class H6RecoveredTrainingAttemptV3:
    """Hydrated exact state at the latest append-only recovery boundary."""

    progress: H6TrainingAttemptProgressV3
    boundary: H6AttemptRecoveryBoundaryV3
    checkpoint: H6CheckpointV3
    history_shard: H6AttemptHistoryShardV3
    history: H6TrainingAttemptHistoryV3
    built_arm: BuiltArm
    model: nn.Module
    recognition: nn.Module | None
    model_optimizer: torch.optim.AdamW
    recognition_optimizer: torch.optim.AdamW | None
    cursor: H6AttemptCursorV3
    resume_state: H6TrainingBatchResultV3 | None

    def __post_init__(self) -> None:
        self.progress.__post_init__()
        self.boundary.__post_init__()
        self.checkpoint.__post_init__()
        self.history_shard.__post_init__()
        self.history.__post_init__()
        if (
            self.boundary != self.progress.latest_boundary
            or self.checkpoint.checkpoint_sha256 != self.boundary.checkpoint_sha256
            or self.cursor != self.checkpoint.cursor
            or self.history.attempt_spec_sha256 != self.progress.attempt_spec_sha256
            or self.history.metric_history_sha256 != self.progress.metric_history_sha256
            or self.history.validation_history_sha256
            != self.progress.validation_history_sha256
            or not isinstance(self.model, nn.Module)
            or type(self.model_optimizer) is not torch.optim.AdamW
        ):
            raise ValueError("recovered training attempt is inconsistent")
        if (self.recognition is None) != (self.recognition_optimizer is None):
            raise ValueError("recovered recognition inventory is incomplete")
        if self.resume_state is not None:
            self.resume_state.__post_init__()
            if self.resume_state.cursor != self.cursor:
                raise ValueError("recovered resume state left its cursor")


def _restore_resume_state_v3(
    *,
    authority: H6EngineAuthorityV3,
    checkpoint: H6CheckpointV3,
    shard: H6AttemptHistoryShardV3,
) -> H6TrainingBatchResultV3 | None:
    if checkpoint.cursor.next_phase is not TrainingPhase.MODEL_ADAMW:
        if shard.resume_result_sha256 is not None:
            raise ValueError("batch-boundary shard retains in-batch history")
        return None
    snapshot = checkpoint.detached_batch_snapshot
    if snapshot is None or shard.resume_result_sha256 is None:
        raise ValueError("model-phase recovery lacks persisted batch state")
    values = {
        "authority_sha256": authority.authority_sha256,
        "latent_enabled": authority.latent_enabled,
        "cursor": checkpoint.cursor,
        "snapshot": snapshot,
        "phase_records": shard.resume_phase_records,
        "metric_records": shard.resume_metric_records,
        "recognition_update_count": (checkpoint.cursor.recognition_update_count),
        "model_update_count": checkpoint.cursor.model_update_count,
        "gradient_clip_count": len(shard.resume_phase_records),
        "checkpoint_phases": shard.resume_checkpoint_phases,
    }
    provisional = object.__new__(H6TrainingBatchResultV3)
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    restored = H6TrainingBatchResultV3(
        **values,
        result_sha256=_owned_hash(
            "vfe4.h6.training-batch-result.v3",
            provisional.canonical_payload(),
        ),
    )
    if restored.result_sha256 != shard.resume_result_sha256:
        raise ValueError("persisted batch result identity differs after reopen")
    return restored


def _adopt_next_recovery_orphan_v3(
    *,
    executable: H6ExecutableAttemptV3,
    runtime: H6RuntimeBindingV3,
    checkpoint_path: Path,
    progress_path: Path,
    maximum_bytes: int,
    prior_progress: H6TrainingAttemptProgressV3 | None,
) -> (
    tuple[
        H6TrainingAttemptProgressV3,
        _H6CanonicalCheckpointHydrationPreflightV3,
    ]
    | None
):
    """Adopt one complete pair or leave one partial pair for exact replay."""

    boundary_kinds: tuple[
        Literal["post_recognition", "batch_boundary", "terminal"], ...
    ] = ("post_recognition", "batch_boundary", "terminal")
    ordinal = 0 if prior_progress is None else len(prior_progress.boundaries)
    complete: list[
        tuple[
            Literal["post_recognition", "batch_boundary", "terminal"],
            Path,
            Path,
        ]
    ] = []
    partial: list[str] = []
    for boundary_kind in boundary_kinds:
        recovery_checkpoint_path, history_path = _recovery_artifact_paths_v3(
            checkpoint_path=checkpoint_path,
            ordinal=ordinal,
            boundary_kind=boundary_kind,
        )
        checkpoint_exists = os.path.lexists(recovery_checkpoint_path)
        history_exists = os.path.lexists(history_path)
        if checkpoint_exists != history_exists:
            partial.append(boundary_kind)
        elif checkpoint_exists:
            complete.append(
                (
                    boundary_kind,
                    recovery_checkpoint_path,
                    history_path,
                )
            )
    if partial and (complete or len(partial) > 1):
        raise RuntimeError(
            "next recovery ordinal contains an ambiguous partial orphan inventory"
        )
    if partial:
        # A lone half-pair has no resumable authority.  Leave it immutable and
        # replay from the prior catalog; the idempotent publishers must then
        # reopen the existing half at the exact regenerated bytes.
        return None
    if len(complete) > 1:
        raise RuntimeError("next recovery ordinal contains multiple orphan pairs")
    if not complete:
        return None

    boundary_kind, recovery_checkpoint_path, history_path = complete[0]
    checkpoint = read_h6_checkpoint_file_v3(
        recovery_checkpoint_path,
        maximum_bytes=maximum_bytes,
    )
    history_raw = _read_bounded_regular_bytes_v3(
        history_path,
        maximum_bytes=maximum_bytes,
        label="attempt history shard",
    )
    shard = _history_shard_from_bytes_v3(history_raw)
    planned = executable.planned_attempt
    spec = planned.attempt_spec
    if (
        checkpoint.attempt_spec != spec
        or checkpoint.runtime_identity != executable.authorities.config.runtime
        or checkpoint.deterministic_policy_sha256 != H6_DETERMINISTIC_POLICY_SHA256
        or checkpoint.objective_manifest.endpoint_config_sha256
        != executable.endpoint_config.config_sha256
        or checkpoint.objective_manifest.objective_kind
        != executable.endpoint_config.objective_kind
    ):
        raise RuntimeError(
            "orphan recovery checkpoint differs from executable authorities"
        )
    _validate_checkpoint_tuning_cell_v3(
        checkpoint,
        executable=executable,
    )

    prior_boundary = None if prior_progress is None else prior_progress.latest_boundary
    if (
        shard.boundary_ordinal != ordinal
        or shard.attempt_spec_sha256 != spec.attempt_spec_sha256
        or shard.prior_boundary_sha256
        != (None if prior_boundary is None else prior_boundary.boundary_sha256)
    ):
        raise RuntimeError("orphan recovery history chain is invalid")
    prior_metric_count = (
        0 if prior_boundary is None else prior_boundary.metric_history_count
    )
    prior_metric_sha256 = (
        _EMPTY_METRIC_HISTORY_SHA256_V3
        if prior_boundary is None
        else prior_boundary.metric_history_sha256
    )
    if tuple(record.ordinal for record in shard.metric_records) != tuple(
        range(
            prior_metric_count,
            prior_metric_count + len(shard.metric_records),
        )
    ):
        raise RuntimeError("orphan recovery metric delta is not append-only")
    metric_history_sha256 = _history_chain_sha256_v3(
        domain="vfe4.h6.attempt-metric-history.v3",
        prior_sha256=prior_metric_sha256,
        prior_count=prior_metric_count,
        appended_sha256s=tuple(record.record_sha256 for record in shard.metric_records),
    )
    metric_history_count = prior_metric_count + len(shard.metric_records)
    prior_validation_count = (
        0 if prior_boundary is None else prior_boundary.validation_history_count
    )
    prior_validation_sha256 = (
        _EMPTY_VALIDATION_HISTORY_SHA256_V3
        if prior_boundary is None
        else prior_boundary.validation_history_sha256
    )
    if len(shard.validation_records) > 1 or tuple(
        record.ordinal for record in shard.validation_records
    ) != tuple(
        range(
            prior_validation_count,
            prior_validation_count + len(shard.validation_records),
        )
    ):
        raise RuntimeError("orphan recovery validation delta is not append-only")
    validation_history_sha256 = _history_chain_sha256_v3(
        domain="vfe4.h6.attempt-validation-history.v3",
        prior_sha256=prior_validation_sha256,
        prior_count=prior_validation_count,
        appended_sha256s=tuple(
            record.record_sha256 for record in shard.validation_records
        ),
    )
    validation_history_count = prior_validation_count + len(shard.validation_records)
    if (
        checkpoint.cursor.recognition_update_count
        + checkpoint.cursor.model_update_count
        != metric_history_count
        or checkpoint.cursor.validation_boundary_count != validation_history_count
        or any(
            record.cursor_sha256 != checkpoint.cursor.cursor_sha256
            or record.checkpoint_sha256 != checkpoint.checkpoint_sha256
            or record.metric_history_sha256 != metric_history_sha256
            for record in shard.validation_records
        )
    ):
        raise RuntimeError("orphan recovery aggregate differs from its checkpoint")
    if prior_boundary is not None:
        prior_checkpoint_path, _prior_history_path = _recovery_artifact_paths_v3(
            checkpoint_path=checkpoint_path,
            ordinal=prior_boundary.ordinal,
            boundary_kind=prior_boundary.boundary_kind,
        )
        prior_checkpoint = read_h6_checkpoint_file_v3(
            prior_checkpoint_path,
            maximum_bytes=maximum_bytes,
            expected_checkpoint_sha256=(prior_boundary.checkpoint_sha256),
        )
        if (
            checkpoint.cursor.recognition_update_count
            < prior_checkpoint.cursor.recognition_update_count
            or checkpoint.cursor.model_update_count
            < prior_checkpoint.cursor.model_update_count
            or checkpoint.cursor.validation_boundary_count
            < prior_checkpoint.cursor.validation_boundary_count
            or checkpoint.cursor.checkpoint_boundary_count
            < prior_checkpoint.cursor.checkpoint_boundary_count
        ):
            raise RuntimeError("orphan recovery cursor order regressed")

    resume_state = _restore_resume_state_v3(
        authority=executable.engine_authority,
        checkpoint=checkpoint,
        shard=shard,
    )
    terminal = _terminal_cursor_matches_plan_v3(
        checkpoint.cursor,
        planned,
    )
    if (boundary_kind == "post_recognition") != (resume_state is not None) or (
        boundary_kind == "terminal"
    ) != terminal:
        raise RuntimeError("orphan recovery kind differs from its persisted state")
    if boundary_kind == "terminal":
        _validate_terminal_checkpoint_v3(
            checkpoint,
            executable=executable,
        )
    hydration_preflight = _canonical_checkpoint_hydration_preflight_v3(
        checkpoint=checkpoint,
        executable=executable,
        runtime=runtime,
    )
    boundary = _new_recovery_boundary_v3(
        boundary_kind=boundary_kind,
        ordinal=ordinal,
        checkpoint_path=recovery_checkpoint_path,
        checkpoint=checkpoint,
        history_path=history_path,
        history_shard=shard,
        metric_history_count=metric_history_count,
        metric_history_sha256=metric_history_sha256,
        validation_history_count=validation_history_count,
        validation_history_sha256=validation_history_sha256,
    )
    progress = _new_progress_v3(
        executable=executable,
        checkpoint_path=checkpoint_path,
        prior=prior_progress,
        boundary=boundary,
    )
    _validate_progress_identity_v3(
        progress,
        executable=executable,
        checkpoint_path=checkpoint_path,
    )
    return (
        _publish_progress_catalog_v3(
            progress,
            path=progress_path,
            maximum_bytes=maximum_bytes,
            expected_prior_sha256=(
                None if prior_progress is None else prior_progress.progress_sha256
            ),
        ),
        hydration_preflight,
    )


def recover_h6_training_attempt_v3(
    *,
    executable: H6ExecutableAttemptV3,
    runtime: H6RuntimeBindingV3,
    checkpoint_path: Path,
    maximum_checkpoint_bytes: int,
) -> H6RecoveredTrainingAttemptV3 | None:
    """Reopen, bind, hydrate, and position the latest exact recovery state."""

    if type(executable) is not H6ExecutableAttemptV3:
        raise ValueError("recovery requires an exact executable attempt")
    executable.__post_init__()
    if type(runtime) is H6InstalledRuntimeBindingV3:
        runtime.assert_production_authorized()
        if runtime.identity != executable.authorities.config.runtime:
            raise ValueError("recovery runtime differs from experiment authority")
    elif type(runtime) is H6SyntheticCpuRuntimeV3:
        runtime.__post_init__()
    else:
        raise ValueError(
            "recovery requires an exact installed or synthetic runtime binding"
        )
    _validate_checkpoint_destination_v3(
        checkpoint_path,
        maximum_checkpoint_bytes,
    )
    progress_path = h6_training_attempt_progress_path_v3(checkpoint_path)
    progress: H6TrainingAttemptProgressV3 | None = None
    history: H6TrainingAttemptHistoryV3 | None = None
    if os.path.lexists(progress_path):
        progress = read_h6_training_attempt_progress_v3(
            path=progress_path,
            maximum_bytes=maximum_checkpoint_bytes,
        )
        _validate_progress_identity_v3(
            progress,
            executable=executable,
            checkpoint_path=checkpoint_path,
        )
        history = _validate_recovery_inventory_v3(
            progress_path=progress_path,
            progress=progress,
            executable=executable,
            maximum_bytes=maximum_checkpoint_bytes,
        )
    adopted = _adopt_next_recovery_orphan_v3(
        executable=executable,
        runtime=runtime,
        checkpoint_path=checkpoint_path,
        progress_path=progress_path,
        maximum_bytes=maximum_checkpoint_bytes,
        prior_progress=progress,
    )
    adopted_preflight: _H6CanonicalCheckpointHydrationPreflightV3 | None = None
    if adopted is not None:
        progress, adopted_preflight = adopted
        history = _validate_recovery_inventory_v3(
            progress_path=progress_path,
            progress=progress,
            executable=executable,
            maximum_bytes=maximum_checkpoint_bytes,
        )
    if progress is None or history is None:
        return None
    boundary = progress.latest_boundary
    expected_checkpoint_path, expected_history_path = _recovery_artifact_paths_v3(
        checkpoint_path=checkpoint_path,
        ordinal=boundary.ordinal,
        boundary_kind=boundary.boundary_kind,
    )
    if (
        boundary.checkpoint_filename != expected_checkpoint_path.name
        or boundary.history_filename != expected_history_path.name
    ):
        raise RuntimeError("recovery catalog path inventory is not deterministic")
    checkpoint = read_h6_checkpoint_file_v3(
        expected_checkpoint_path,
        maximum_bytes=maximum_checkpoint_bytes,
        expected_checkpoint_sha256=boundary.checkpoint_sha256,
    )
    checkpoint_raw = checkpoint.to_bytes()
    if (
        len(checkpoint_raw) != boundary.checkpoint_byte_count
        or hashlib.sha256(checkpoint_raw).hexdigest()
        != boundary.checkpoint_bytes_sha256
        or checkpoint.cursor.cursor_sha256 != boundary.cursor_sha256
        or checkpoint.cursor.next_phase is not boundary.next_phase
    ):
        raise RuntimeError("recovery checkpoint differs from its catalog entry")
    shard = _read_history_shard_for_boundary_v3(
        parent=progress_path.parent,
        boundary=boundary,
        maximum_bytes=maximum_checkpoint_bytes,
    )
    prior_boundary = (
        None if boundary.ordinal == 0 else progress.boundaries[boundary.ordinal - 1]
    )
    if (
        shard.attempt_spec_sha256 != progress.attempt_spec_sha256
        or shard.prior_boundary_sha256
        != (None if prior_boundary is None else prior_boundary.boundary_sha256)
    ):
        raise RuntimeError("latest recovery history shard chain is broken")
    prior_metric_count = (
        0 if prior_boundary is None else prior_boundary.metric_history_count
    )
    prior_metric_sha256 = (
        _EMPTY_METRIC_HISTORY_SHA256_V3
        if prior_boundary is None
        else prior_boundary.metric_history_sha256
    )
    if tuple(record.ordinal for record in shard.metric_records) != tuple(
        range(
            prior_metric_count,
            prior_metric_count + len(shard.metric_records),
        )
    ):
        raise RuntimeError("latest recovery metric delta is not append-only")
    expected_metric_sha256 = _history_chain_sha256_v3(
        domain="vfe4.h6.attempt-metric-history.v3",
        prior_sha256=prior_metric_sha256,
        prior_count=prior_metric_count,
        appended_sha256s=tuple(record.record_sha256 for record in shard.metric_records),
    )
    prior_validation_count = (
        0 if prior_boundary is None else prior_boundary.validation_history_count
    )
    prior_validation_sha256 = (
        _EMPTY_VALIDATION_HISTORY_SHA256_V3
        if prior_boundary is None
        else prior_boundary.validation_history_sha256
    )
    if tuple(record.ordinal for record in shard.validation_records) != tuple(
        range(
            prior_validation_count,
            prior_validation_count + len(shard.validation_records),
        )
    ):
        raise RuntimeError("latest validation delta is not append-only")
    expected_validation_sha256 = _history_chain_sha256_v3(
        domain="vfe4.h6.attempt-validation-history.v3",
        prior_sha256=prior_validation_sha256,
        prior_count=prior_validation_count,
        appended_sha256s=tuple(
            record.record_sha256 for record in shard.validation_records
        ),
    )
    if (
        boundary.metric_history_count != prior_metric_count + len(shard.metric_records)
        or boundary.metric_history_sha256 != expected_metric_sha256
        or boundary.validation_history_count
        != prior_validation_count + len(shard.validation_records)
        or boundary.validation_history_sha256 != expected_validation_sha256
    ):
        raise RuntimeError("latest recovery history aggregate is stale")
    planned = executable.planned_attempt
    if (
        checkpoint.attempt_spec != planned.attempt_spec
        or checkpoint.runtime_identity != executable.authorities.config.runtime
        or checkpoint.deterministic_policy_sha256 != H6_DETERMINISTIC_POLICY_SHA256
        or checkpoint.objective_manifest.endpoint_config_sha256
        != executable.endpoint_config.config_sha256
        or checkpoint.objective_manifest.objective_kind
        != executable.endpoint_config.objective_kind
    ):
        raise RuntimeError("recovery checkpoint differs from executable authorities")
    _validate_checkpoint_tuning_cell_v3(
        checkpoint,
        executable=executable,
    )

    # All executable/runtime/cell/factory/init identities are now bound.
    # An adopted orphan was hydrated before its catalog became visible; reuse
    # that exact in-memory result.  Existing catalog entries receive the same
    # side-effect-free canonical preflight here.
    if (
        adopted_preflight is not None
        and adopted_preflight.checkpoint_sha256 == checkpoint.checkpoint_sha256
    ):
        hydration_preflight = adopted_preflight
    else:
        hydration_preflight = _canonical_checkpoint_hydration_preflight_v3(
            checkpoint=checkpoint,
            executable=executable,
            runtime=runtime,
        )
    built = hydration_preflight.built_arm
    hydrated = hydration_preflight.hydrated
    modules = dict(hydrated.named_modules)
    optimizers = dict(hydrated.named_optimizers)
    model = modules["model"]
    recognition = modules.get("recognition")
    model_optimizer = optimizers["model"]
    recognition_optimizer = optimizers.get("recognition")
    resume_state = _restore_resume_state_v3(
        authority=executable.engine_authority,
        checkpoint=checkpoint,
        shard=shard,
    )
    return H6RecoveredTrainingAttemptV3(
        progress=progress,
        boundary=boundary,
        checkpoint=checkpoint,
        history_shard=shard,
        history=history,
        built_arm=built,
        model=model,
        recognition=recognition,
        model_optimizer=model_optimizer,
        recognition_optimizer=recognition_optimizer,
        cursor=hydrated.cursor,
        resume_state=resume_state,
    )


def reopen_h6_terminal_training_attempt_v3(
    *,
    executable: H6ExecutableAttemptV3,
    checkpoint_path: Path,
    maximum_checkpoint_bytes: int,
) -> H6TrainingAttemptResultV3:
    """Reopen a terminal result only with its complete recovery/history chain."""

    _validate_checkpoint_destination_v3(
        checkpoint_path,
        maximum_checkpoint_bytes,
    )
    if not os.path.lexists(checkpoint_path):
        raise FileNotFoundError("terminal checkpoint does not exist")
    progress_path = h6_training_attempt_progress_path_v3(checkpoint_path)
    if not os.path.lexists(progress_path):
        raise RuntimeError("terminal checkpoint lacks its required progress catalog")
    progress = read_h6_training_attempt_progress_v3(
        path=progress_path,
        maximum_bytes=maximum_checkpoint_bytes,
    )
    _validate_progress_identity_v3(
        progress,
        executable=executable,
        checkpoint_path=checkpoint_path,
    )
    history = _validate_recovery_inventory_v3(
        progress_path=progress_path,
        progress=progress,
        executable=executable,
        maximum_bytes=maximum_checkpoint_bytes,
    )
    boundary = progress.latest_boundary
    if boundary.boundary_kind != "terminal":
        raise RuntimeError(
            "terminal checkpoint points to a nonterminal recovery boundary"
        )
    expected_recovery_path, _history_path = _recovery_artifact_paths_v3(
        checkpoint_path=checkpoint_path,
        ordinal=boundary.ordinal,
        boundary_kind="terminal",
    )
    recovery_checkpoint = read_h6_checkpoint_file_v3(
        expected_recovery_path,
        maximum_bytes=maximum_checkpoint_bytes,
        expected_checkpoint_sha256=boundary.checkpoint_sha256,
    )
    terminal_checkpoint = read_h6_checkpoint_file_v3(
        checkpoint_path,
        maximum_bytes=maximum_checkpoint_bytes,
        expected_checkpoint_sha256=boundary.checkpoint_sha256,
    )
    terminal_raw = terminal_checkpoint.to_bytes()
    if (
        terminal_raw != recovery_checkpoint.to_bytes()
        or len(terminal_raw) != boundary.checkpoint_byte_count
        or hashlib.sha256(terminal_raw).hexdigest() != boundary.checkpoint_bytes_sha256
        or len(history.metric_history)
        != (
            terminal_checkpoint.cursor.recognition_update_count
            + terminal_checkpoint.cursor.model_update_count
        )
        or len(history.validation_boundary_history)
        != terminal_checkpoint.cursor.validation_boundary_count
    ):
        raise RuntimeError(
            "terminal checkpoint/history bytes differ from recovery authority"
        )
    _validate_terminal_checkpoint_v3(
        terminal_checkpoint,
        executable=executable,
    )
    return _attempt_result_v3(
        executable=executable,
        checkpoint=terminal_checkpoint,
        checkpoint_path=checkpoint_path,
        progress=progress,
        history=history,
    )


def _validate_checkpoint_destination_v3(
    checkpoint_path: Path,
    maximum_checkpoint_bytes: int,
) -> None:
    if not isinstance(checkpoint_path, Path) or not checkpoint_path.is_absolute():
        raise ValueError("checkpoint_path must be an absolute pathlib Path")
    if checkpoint_path.resolve(strict=False) != checkpoint_path:
        raise ValueError("checkpoint_path must be canonical")
    if type(maximum_checkpoint_bytes) is not int or maximum_checkpoint_bytes <= 0:
        raise ValueError("maximum_checkpoint_bytes must be a positive exact integer")


def _attempt_metric_delta_v3(
    *,
    result: H6TrainingBatchResultV3,
    start_index: int,
    pass_index: int,
    batch_index: int,
    first_ordinal: int,
) -> tuple[H6AttemptMetricHistoryRecordV3, ...]:
    result.__post_init__()
    if (
        type(start_index) is not int
        or not 0 <= start_index <= len(result.phase_records)
        or type(first_ordinal) is not int
        or first_ordinal < 0
    ):
        raise ValueError("attempt metric delta bounds are invalid")
    return tuple(
        H6AttemptMetricHistoryRecordV3.create(
            ordinal=first_ordinal + offset,
            pass_index=pass_index,
            batch_index=batch_index,
            phase_record=phase_record,
            metric_record=metric_record,
        )
        for offset, (phase_record, metric_record) in enumerate(
            zip(
                result.phase_records[start_index:],
                result.metric_records[start_index:],
                strict=True,
            )
        )
    )


def _capture_attempt_checkpoint_v3(
    *,
    executable: H6ExecutableAttemptV3,
    cursor: H6AttemptCursorV3,
    callbacks: _H6BatchCallbacksV3,
    model: nn.Module,
    recognition: nn.Module | None,
    model_optimizer: torch.optim.AdamW,
    recognition_optimizer: torch.optim.AdamW | None,
    detached_batch_snapshot: (H6DetachedBatchRecognitionSnapshotV3 | None),
) -> H6CheckpointV3:
    if (recognition is None) != (recognition_optimizer is None):
        raise ValueError("checkpoint recognition inventory is incomplete")
    objective_manifest = _objective_manifest_v3(
        executable=executable,
        cursor=cursor,
        callbacks=callbacks,
        detached_snapshot_sha256=(
            None
            if detached_batch_snapshot is None
            else detached_batch_snapshot.snapshot_sha256
        ),
    )
    named_modules: tuple[tuple[str, nn.Module], ...] = (
        (("model", model),)
        if recognition is None
        else (("model", model), ("recognition", recognition))
    )
    named_optimizers: tuple[tuple[str, torch.optim.AdamW], ...] = (
        (("model", model_optimizer),)
        if recognition_optimizer is None
        else (
            ("model", model_optimizer),
            ("recognition", recognition_optimizer),
        )
    )
    return capture_h6_checkpoint_v3(
        attempt_spec=executable.planned_attempt.attempt_spec,
        cursor=cursor,
        objective_manifest=objective_manifest,
        runtime_identity=executable.authorities.config.runtime,
        named_modules=named_modules,
        named_optimizers=named_optimizers,
        detached_batch_snapshot=detached_batch_snapshot,
    )


def _execute_new_training_attempt_v3(
    *,
    executable: H6ExecutableAttemptV3,
    training_data: H6TrainingDataV3,
    runtime: H6RuntimeBindingV3,
    checkpoint_path: Path,
    maximum_checkpoint_bytes: int,
) -> H6TrainingAttemptResultV3:
    if type(executable) is not H6ExecutableAttemptV3:
        raise ValueError("training requires an exact executable attempt")
    executable.__post_init__()
    if type(training_data) is not H6TrainingDataV3:
        raise ValueError("training requires exact capability-opened data")
    training_data.__post_init__()
    _validate_checkpoint_destination_v3(
        checkpoint_path,
        maximum_checkpoint_bytes,
    )
    if type(runtime) is H6InstalledRuntimeBindingV3:
        runtime.assert_production_authorized()
        if runtime.identity != executable.authorities.config.runtime:
            raise ValueError("training runtime differs from experiment authority")
    elif type(runtime) is H6SyntheticCpuRuntimeV3:
        runtime.__post_init__()
    else:
        raise ValueError(
            "training requires an exact installed or synthetic runtime binding"
        )
    plan = executable.authorities.plan
    workload = executable.authorities.matching_set.workload
    if (
        training_data.data_identity_sha256
        != executable.planned_attempt.attempt_spec.data_identity_sha256
        or training_data.readiness_sha256 != plan.readiness_sha256
        or training_data.plan_sha256 != plan.plan_sha256
        or training_data.matching_set_sha256 != plan.matching_set_sha256
        or training_data.runtime_identity_sha256
        != plan.training_schedule.runtime_identity_sha256
        or training_data.vocabulary != executable.endpoint_config.vocabulary
        or len(training_data.windows) != workload.window_count
    ):
        raise ValueError("training data differs from executable plan authority")
    if os.path.lexists(checkpoint_path):
        return reopen_h6_terminal_training_attempt_v3(
            executable=executable,
            checkpoint_path=checkpoint_path,
            maximum_checkpoint_bytes=maximum_checkpoint_bytes,
        )

    authority = executable.engine_authority
    schedules = tuple(
        frozen_batch_schedule(
            window_count=workload.window_count,
            zero_based_pass_index=pass_index,
        )
        for pass_index in (
            (0,)
            if executable.planned_attempt.stage == "tuning"
            else tuple(range(workload.full_passes))
        )
    )
    if tuple(schedule.schedule_sha256 for schedule in schedules) != (
        executable.planned_attempt.consumed_permutation_sha256s
    ):
        raise ValueError("live batch permutations differ from the plan")

    recovered = recover_h6_training_attempt_v3(
        executable=executable,
        runtime=runtime,
        checkpoint_path=checkpoint_path,
        maximum_checkpoint_bytes=maximum_checkpoint_bytes,
    )
    progress: H6TrainingAttemptProgressV3 | None
    active_resume_state: H6TrainingBatchResultV3 | None
    if recovered is None:
        built, model, recognition = _fresh_training_modules_v3(
            executable=executable,
            runtime=runtime,
        )
        model_optimizer = _adamw_for_module_v3(
            model,
            authority=authority,
        )
        recognition_optimizer = (
            None
            if recognition is None
            else _adamw_for_module_v3(
                recognition,
                authority=authority,
            )
        )
        cursor = _initial_cursor_v3(
            executable,
            permutation_sha256=schedules[0].schedule_sha256,
        )
        progress = None
        active_resume_state = None
    else:
        built = recovered.built_arm
        model = recovered.model
        recognition = recovered.recognition
        model_optimizer = recovered.model_optimizer
        recognition_optimizer = recovered.recognition_optimizer
        cursor = recovered.cursor
        progress = recovered.progress
        active_resume_state = recovered.resume_state
        if _terminal_cursor_matches_plan_v3(
            cursor,
            executable.planned_attempt,
        ):
            _validate_terminal_checkpoint_v3(
                recovered.checkpoint,
                executable=executable,
            )
            try:
                _publish_checkpoint_v3(
                    recovered.checkpoint,
                    path=checkpoint_path,
                    maximum_bytes=maximum_checkpoint_bytes,
                )
            except FileExistsError:
                read_h6_checkpoint_file_v3(
                    checkpoint_path,
                    maximum_bytes=maximum_checkpoint_bytes,
                    expected_checkpoint_sha256=(recovered.checkpoint.checkpoint_sha256),
                )
            return reopen_h6_terminal_training_attempt_v3(
                executable=executable,
                checkpoint_path=checkpoint_path,
                maximum_checkpoint_bytes=maximum_checkpoint_bytes,
            )

    final_callbacks: _H6BatchCallbacksV3 | None = None
    completed_batches = cursor.model_update_count
    pending_metrics: list[H6AttemptMetricHistoryRecordV3] = []
    next_metric_ordinal = 0 if progress is None else progress.metric_history_count
    terminal_recovery_checkpoint: H6CheckpointV3 | None = None
    for schedule_ordinal, schedule in enumerate(schedules):
        batches = (
            quarter_pass_batches(schedule)
            if executable.planned_attempt.stage == "tuning"
            else schedule_batches(schedule)
        )
        for batch_index, window_indices in enumerate(batches):
            if schedule.zero_based_pass_index < cursor.pass_index:
                continue
            if (
                schedule.zero_based_pass_index == cursor.pass_index
                and batch_index < cursor.batch_index
            ):
                continue
            if (
                cursor.pass_index != schedule.zero_based_pass_index
                or cursor.batch_index != batch_index
                or cursor.permutation_sha256 != schedule.schedule_sha256
            ):
                raise ValueError("training cursor left its exact batch schedule")
            batch = _window_batch_v3(
                windows=training_data.windows,
                window_indices=window_indices,
                maximum_horizon=executable.endpoint_config.horizon,
            )
            callbacks = _H6BatchCallbacksV3(
                built_arm=built,
                model=model,
                recognition=recognition,
                authority=authority,
                windows=training_data.windows,
                batch=batch,
            )
            is_last = (
                schedule_ordinal == len(schedules) - 1
                and batch_index == len(batches) - 1
            )
            completed_in_pass = batch_index + 1
            is_validation_boundary = (
                completed_in_pass in workload.validation_boundaries_per_pass
            )
            if (
                authority.latent_enabled
                and active_resume_state is None
                and is_validation_boundary
            ):
                first = run_h6_training_batch_v3(
                    authority=authority,
                    cursor=cursor,
                    model=model,
                    recognition=recognition,
                    model_optimizer=model_optimizer,
                    recognition_optimizer=recognition_optimizer,
                    recognition_forward=callbacks.recognition_forward,
                    objective_forward=callbacks.objective_forward,
                    noise_factory=callbacks.noise_factory,
                    stop_after_phase=(TrainingPhase.IMMUTABLE_DETACHED_SNAPSHOT),
                    checkpoint_at_batch_end=False,
                )
                metric_delta = _attempt_metric_delta_v3(
                    result=first,
                    start_index=0,
                    pass_index=schedule.zero_based_pass_index,
                    batch_index=batch_index,
                    first_ordinal=next_metric_ordinal,
                )
                pending_metrics.extend(metric_delta)
                next_metric_ordinal += len(metric_delta)
                if type(first.snapshot) is not H6DetachedBatchRecognitionSnapshotV3:
                    raise ValueError("post-recognition boundary lacks its detached law")
                partial_checkpoint = _capture_attempt_checkpoint_v3(
                    executable=executable,
                    cursor=first.cursor,
                    callbacks=callbacks,
                    model=model,
                    recognition=recognition,
                    model_optimizer=model_optimizer,
                    recognition_optimizer=recognition_optimizer,
                    detached_batch_snapshot=first.snapshot,
                )
                progress, _ = _persist_attempt_boundary_v3(
                    executable=executable,
                    terminal_checkpoint_path=checkpoint_path,
                    maximum_bytes=maximum_checkpoint_bytes,
                    prior_progress=progress,
                    boundary_kind="post_recognition",
                    checkpoint=partial_checkpoint,
                    metric_records=tuple(pending_metrics),
                    validation_pass_index=None,
                    validation_completed_batch_count=None,
                    resume_state=first,
                )
                pending_metrics.clear()
                active_resume_state = first
                cursor = first.cursor

            resume_prefix_count = (
                0
                if active_resume_state is None
                else len(active_resume_state.phase_records)
            )
            result = run_h6_training_batch_v3(
                authority=authority,
                cursor=cursor,
                model=model,
                recognition=recognition,
                model_optimizer=model_optimizer,
                recognition_optimizer=recognition_optimizer,
                recognition_forward=(
                    None if recognition is None else callbacks.recognition_forward
                ),
                objective_forward=callbacks.objective_forward,
                noise_factory=callbacks.noise_factory,
                checkpoint_at_batch_end=is_last,
                resume_state=active_resume_state,
            )
            metric_delta = _attempt_metric_delta_v3(
                result=result,
                start_index=resume_prefix_count,
                pass_index=schedule.zero_based_pass_index,
                batch_index=batch_index,
                first_ordinal=next_metric_ordinal,
            )
            pending_metrics.extend(metric_delta)
            next_metric_ordinal += len(metric_delta)
            cursor = result.cursor
            active_resume_state = None
            completed_batches += 1
            if is_validation_boundary:
                cursor = _reissue_cursor_v3(
                    cursor,
                    validation_boundary_delta=1,
                )
            if (
                batch_index == len(batches) - 1
                and executable.planned_attempt.stage == "confirmatory"
            ):
                next_pass_index = schedule.zero_based_pass_index + 1
                next_permutation = (
                    schedule.schedule_sha256
                    if is_last
                    else schedules[schedule_ordinal + 1].schedule_sha256
                )
                cursor = _reissue_cursor_v3(
                    cursor,
                    pass_index=next_pass_index,
                    batch_index=0,
                    permutation_sha256=next_permutation,
                )
            final_callbacks = callbacks
            if is_validation_boundary or is_last:
                boundary_kind: Literal[
                    "batch_boundary",
                    "terminal",
                ] = "terminal" if is_last else "batch_boundary"
                boundary_checkpoint = _capture_attempt_checkpoint_v3(
                    executable=executable,
                    cursor=cursor,
                    callbacks=callbacks,
                    model=model,
                    recognition=recognition,
                    model_optimizer=model_optimizer,
                    recognition_optimizer=recognition_optimizer,
                    detached_batch_snapshot=None,
                )
                progress, reopened_boundary = _persist_attempt_boundary_v3(
                    executable=executable,
                    terminal_checkpoint_path=checkpoint_path,
                    maximum_bytes=maximum_checkpoint_bytes,
                    prior_progress=progress,
                    boundary_kind=boundary_kind,
                    checkpoint=boundary_checkpoint,
                    metric_records=tuple(pending_metrics),
                    validation_pass_index=(
                        schedule.zero_based_pass_index
                        if is_validation_boundary
                        else None
                    ),
                    validation_completed_batch_count=(
                        completed_in_pass if is_validation_boundary else None
                    ),
                    resume_state=None,
                )
                pending_metrics.clear()
                if is_last:
                    terminal_recovery_checkpoint = reopened_boundary

    planned = executable.planned_attempt
    if (
        final_callbacks is None
        or completed_batches != planned.terminal_model_update_count
        or not _terminal_cursor_matches_plan_v3(cursor, planned)
        or pending_metrics
        or terminal_recovery_checkpoint is None
    ):
        raise ValueError(
            "completed training attempt differs from the planned terminal cursor"
        )
    _validate_terminal_checkpoint_v3(
        terminal_recovery_checkpoint,
        executable=executable,
    )
    try:
        _publish_checkpoint_v3(
            terminal_recovery_checkpoint,
            path=checkpoint_path,
            maximum_bytes=maximum_checkpoint_bytes,
        )
    except FileExistsError:
        read_h6_checkpoint_file_v3(
            checkpoint_path,
            maximum_bytes=maximum_checkpoint_bytes,
            expected_checkpoint_sha256=(terminal_recovery_checkpoint.checkpoint_sha256),
        )
    return reopen_h6_terminal_training_attempt_v3(
        executable=executable,
        checkpoint_path=checkpoint_path,
        maximum_checkpoint_bytes=maximum_checkpoint_bytes,
    )


def execute_h6_training_attempt_v3(
    *,
    executable: H6ExecutableAttemptV3,
    training_data: H6TrainingDataV3,
    runtime: H6RuntimeBindingV3,
    checkpoint_path: Path,
    maximum_checkpoint_bytes: int,
) -> H6TrainingAttemptResultV3:
    """Execute or resume one already-bound exact H6 v3 training attempt."""

    return _execute_new_training_attempt_v3(
        executable=executable,
        training_data=training_data,
        runtime=runtime,
        checkpoint_path=checkpoint_path,
        maximum_checkpoint_bytes=maximum_checkpoint_bytes,
    )


def run_h6_training_attempt_v3(
    *,
    authorities: H6PredictionV3Authorities,
    store: BlindedCorpusStore,
    runtime: H6InstalledRuntimeBindingV3,
    planned_attempt_sha256: str,
    checkpoint_path: Path,
    maximum_checkpoint_bytes: int,
    validation_bundle_directory: Path,
) -> H6TrainingAttemptResultV3:
    """Execute one exact plan member using only authenticated train access."""

    if type(authorities) is not H6PredictionV3Authorities:
        raise ValueError("training requires exact H6 v3 authorities")
    authorities.__post_init__()
    if type(store) is not BlindedCorpusStore:
        raise ValueError("training requires an exact blinded corpus store")
    if type(runtime) is not H6InstalledRuntimeBindingV3:
        raise ValueError("training requires the configured installed CUDA runtime")
    runtime.assert_production_authorized()
    if runtime.identity != authorities.config.runtime:
        raise ValueError("configured runtime differs from authority identity")
    _validate_checkpoint_destination_v3(
        checkpoint_path,
        maximum_checkpoint_bytes,
    )
    if (
        type(planned_attempt_sha256) is not str
        or len(planned_attempt_sha256) != 64
        or any(
            character not in "0123456789abcdef" for character in planned_attempt_sha256
        )
    ):
        raise ValueError("planned_attempt_sha256 must be lowercase SHA-256")
    by_sha256 = {
        attempt.planned_attempt_sha256: attempt for attempt in authorities.plan.attempts
    }
    planned = by_sha256.get(planned_attempt_sha256)
    if type(planned) is not H6PlannedAttemptV3:
        raise ValueError("planned attempt is not in the exact experiment plan")
    tuning_selection: object = None
    if planned.stage == "confirmatory":
        from .h6_validation_campaign_v3 import (
            h6_tuning_selection_directory_v3,
            read_h6_tuning_selection_v3,
        )

        if (
            not isinstance(validation_bundle_directory, Path)
            or not validation_bundle_directory.is_absolute()
        ):
            raise ValueError(
                "confirmatory training requires an absolute validation bundle"
            )
        tuning_selection = read_h6_tuning_selection_v3(
            h6_tuning_selection_directory_v3(validation_bundle_directory),
            expected_plan_sha256=authorities.plan.plan_sha256,
            expected_experiment_config_sha256=(authorities.config.config_sha256),
        )
    executable = bind_h6_executable_attempt_v3(
        authorities=authorities,
        planned_attempt=planned,
        tuning_selection=tuning_selection,
    )
    if os.path.lexists(checkpoint_path):
        return reopen_h6_terminal_training_attempt_v3(
            executable=executable,
            checkpoint_path=checkpoint_path,
            maximum_checkpoint_bytes=maximum_checkpoint_bytes,
        )
    capability = issue_h6_train_capability_v3(
        store,
        authorities.readiness,
        authorities.plan,
    )
    training_data = open_train_for_training_v3(
        capability,
        plan=authorities.plan,
    )
    return _execute_new_training_attempt_v3(
        executable=executable,
        training_data=training_data,
        runtime=runtime,
        checkpoint_path=checkpoint_path,
        maximum_checkpoint_bytes=maximum_checkpoint_bytes,
    )


__all__ = [
    "H6AttemptHistoryShardV3",
    "H6AttemptMetricHistoryRecordV3",
    "H6AttemptRecoveryBoundaryV3",
    "H6GenerativePriorFeatureProviderV3",
    "H6RecoveredTrainingAttemptV3",
    "H6TrainingAttemptHistoryV3",
    "H6TrainingAttemptProgressV3",
    "H6TrainingAttemptResultV3",
    "H6ValidationBoundaryHistoryRecordV3",
    "H6_VALIDATION_BOUNDARY_CONTRACT_SHA256_V3",
    "execute_h6_training_attempt_v3",
    "h6_training_attempt_progress_path_v3",
    "read_h6_training_attempt_history_v3",
    "read_h6_training_attempt_progress_v3",
    "recover_h6_training_attempt_v3",
    "reopen_h6_terminal_training_attempt_v3",
    "run_h6_training_attempt_v3",
]
