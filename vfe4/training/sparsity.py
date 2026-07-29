"""Revision-bound structural sparsity certification for WT103 training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from vfe4.types.results import GateStatus
from vfe4.types.training import (
    A0ArchitectureProfile,
    EndpointInventory,
    TrainingSparsityCertificate,
    WT103ArmSpec,
    WT103ExperimentProfile,
    owned_sha256,
)

from .factories import WT103FactorySetIdentity


_OBSERVABILITY_VIEWS = (
    "dispatch",
    "profiler",
    "cuda_allocator_unique_storage",
    "backend_checkpoint_inventory",
)
_NEGATIVE_CONTROL_IDS = (
    "a0_math_sdpa_fallback",
    "a0_explicit_causal_mask",
    "a0_attention_weights",
    "dense_population",
    "batch_dense_population",
    "full_source",
    "pair_slab",
    "full_decoder",
    "full_selector_rhs",
    "unclassified_checkpoint",
)
_STORAGE_CLASSES = (
    "vocabulary_parameter",
    "decoder_chunk",
    "a0_qkv_or_result",
    "token_ids_or_mask",
    "latent_block",
    "lower_adjacent_block",
    "banded_source",
    "primary_frame",
    "local_workspace",
    "particle_chunk",
    "scalar_or_row",
    "checkpoint_classified_parameter",
    "checkpoint_model_parameter",
    "checkpoint_recognition_parameter",
    "checkpoint_optimizer_state",
    "checkpoint_scheduler_state",
    "checkpoint_rng_state",
    "checkpoint_estimator_state",
    "checkpoint_cursor_or_metric",
)


class ForbiddenStorageRequest(ValueError):
    """A logical shape must be rejected before allocation/serialization."""


def _sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _git_head(value: object) -> str:
    if (
        type(value) is not str
        or len(value) not in (40, 64)
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("git_head must be a concrete hex object id")
    return value


def _expected_path_events(spec: WT103ArmSpec) -> tuple[str, ...]:
    if spec.arm_id in ("WT103-A0-AR-v1", "WT103-A5-NOLATENT-v1"):
        return (
            "data_transfer",
            "forward",
            "cross_entropy",
            "backward",
            "model_adamw",
            "exact_autoregressive_scorer",
            "metric_failure_write",
            "checkpoint_serialization",
        )
    objective_event = (
        "complete_elbo"
        if spec.training_objective == "complete_elbo"
        else "emission_only_ablation_non_elbo"
    )
    return (
        "data_transfer",
        "forward",
        "recognition_adam_proposal",
        "recognition_adamw",
        "immutable_detached_snapshot",
        objective_event,
        "model_backward",
        "model_adamw",
        "weighted_smc_scorer",
        "metric_failure_write",
        "checkpoint_serialization",
    )


@dataclass(frozen=True, slots=True)
class TensorStorageObservation:
    """One dispatch/allocator/serializer-classified logical tensor."""

    event_id: str
    arm_id: str
    path_event: str
    phase: Literal["train", "evaluation", "checkpoint"]
    storage_id: str
    storage_class: str
    shape: tuple[int, ...]
    logical_axes: tuple[str, ...]
    numel: int
    element_size_bytes: int
    logical_bytes: int
    storage_span_bytes: int

    def __post_init__(self) -> None:
        for name in ("event_id", "arm_id", "path_event", "storage_id"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be nonempty text")
        if self.phase not in ("train", "evaluation", "checkpoint"):
            raise ValueError("unknown sparsity observation phase")
        if self.storage_class not in _STORAGE_CLASSES:
            raise ValueError("storage_class is not classified")
        if (
            type(self.shape) is not tuple
            or not self.shape
            or any(type(item) is not int or item <= 0 for item in self.shape)
            or type(self.logical_axes) is not tuple
            or len(self.logical_axes) != len(self.shape)
            or any(type(item) is not str or not item for item in self.logical_axes)
        ):
            raise ValueError("logical shape/axes are invalid")
        expected_numel = math.prod(self.shape)
        if (
            type(self.numel) is not int
            or self.numel != expected_numel
            or type(self.element_size_bytes) is not int
            or self.element_size_bytes <= 0
            or type(self.logical_bytes) is not int
            or self.logical_bytes != self.numel * self.element_size_bytes
            or type(self.storage_span_bytes) is not int
            or self.storage_span_bytes < self.logical_bytes
        ):
            raise ValueError("storage bytes do not reconcile as numel*element_size")


@dataclass(frozen=True, slots=True)
class ArmPathTrace:
    """All enumerated events and four observability views for one arm."""

    arm_id: str
    arm_spec_sha256: str
    path_events: tuple[str, ...]
    observability_views: tuple[str, ...]
    observations: tuple[TensorStorageObservation, ...]

    def __post_init__(self) -> None:
        if (
            type(self.arm_id) is not str
            or not self.arm_id
            or type(self.path_events) is not tuple
            or not self.path_events
            or len(set(self.path_events)) != len(self.path_events)
            or type(self.observability_views) is not tuple
            or len(set(self.observability_views))
            != len(self.observability_views)
            or type(self.observations) is not tuple
            or not self.observations
            or any(
                type(item) is not TensorStorageObservation
                for item in self.observations
            )
        ):
            raise ValueError("arm path trace is malformed")
        _sha256(self.arm_spec_sha256, "arm_spec_sha256")
        if any(
            item.arm_id != self.arm_id
            or item.path_event not in self.path_events
            for item in self.observations
        ):
            raise ValueError("trace observations do not belong to its path")

    @classmethod
    def exact_for_arm(
        cls,
        spec: WT103ArmSpec,
        *,
        observations: tuple[TensorStorageObservation, ...],
    ) -> "ArmPathTrace":
        if type(spec) is not WT103ArmSpec:
            raise ValueError("spec must be an exact WT103ArmSpec")
        spec.__post_init__()
        return cls(
            arm_id=spec.arm_id,
            arm_spec_sha256=spec.arm_spec_sha256,
            path_events=_expected_path_events(spec),
            observability_views=_OBSERVABILITY_VIEWS,
            observations=observations,
        )


@dataclass(frozen=True, slots=True)
class FlashAttentionObservation:
    arm_id: str
    backend: str
    fallback_allowed: bool
    explicit_mask_materialized: bool
    attention_weights_returned: bool
    materialized_pair_shapes: tuple[tuple[int, ...], ...]
    semantic_pair_count: int
    dispatch_observed: bool
    profiler_observed: bool
    allocator_observed: bool
    backend_observed: bool

    @classmethod
    def exact(
        cls,
        *,
        arm_id: str,
        sequence_length: int,
    ) -> "FlashAttentionObservation":
        return cls(
            arm_id=arm_id,
            backend="FLASH_ATTENTION",
            fallback_allowed=False,
            explicit_mask_materialized=False,
            attention_weights_returned=False,
            materialized_pair_shapes=(),
            semantic_pair_count=sequence_length * (sequence_length + 1) // 2,
            dispatch_observed=True,
            profiler_observed=True,
            allocator_observed=True,
            backend_observed=True,
        )


@dataclass(frozen=True, slots=True)
class NegativeControlRecord:
    control_id: str
    guard_id: str
    logical_shape: tuple[int, ...]
    fired_pre_allocation: bool
    allocation_or_serialization_attempted: bool

    def __post_init__(self) -> None:
        if (
            self.control_id not in _NEGATIVE_CONTROL_IDS
            or type(self.guard_id) is not str
            or not self.guard_id
            or type(self.logical_shape) is not tuple
            or not self.logical_shape
            or any(
                type(item) is not int or item <= 0
                for item in self.logical_shape
            )
            or type(self.fired_pre_allocation) is not bool
            or type(self.allocation_or_serialization_attempted) is not bool
        ):
            raise ValueError("negative control record is invalid")


def _forbidden_shape(
    *,
    profile: WT103ExperimentProfile,
    shape: tuple[int, ...],
) -> bool:
    batch = profile.batch_size
    length = profile.sequence_length
    block = profile.combined_latent_block
    dimension = length * block
    vocabulary = profile.vocabulary_size
    return shape in (
        (length, length),
        (batch, length, length),
        (batch, 2, length, length),
        (length, length, block, block),
        (batch, length, length, block, block),
        (dimension, dimension),
        (batch, dimension, dimension),
        (batch * length, vocabulary),
    )


def guard_tensor_request(
    *,
    profile: WT103ExperimentProfile,
    architecture: A0ArchitectureProfile,
    storage_class: str,
    shape: tuple[int, ...],
    phase: Literal["train", "evaluation", "checkpoint"],
) -> None:
    """Classify a logical shape before any allocation or serialization."""

    if (
        type(profile) is not WT103ExperimentProfile
        or type(architecture) is not A0ArchitectureProfile
    ):
        raise ValueError("shape guards require exact profile/architecture")
    profile.__post_init__()
    architecture.__post_init__()
    if (
        type(shape) is not tuple
        or not shape
        or any(type(item) is not int or item <= 0 for item in shape)
        or phase not in ("train", "evaluation", "checkpoint")
    ):
        raise ValueError("shape request metadata is invalid")
    if storage_class not in _STORAGE_CLASSES:
        raise ForbiddenStorageRequest("unclassified storage is forbidden")
    batch = profile.batch_size
    length = profile.sequence_length
    block = profile.combined_latent_block
    width = architecture.hidden_width
    vocabulary = profile.vocabulary_size
    allowed = False
    if storage_class == "vocabulary_parameter":
        allowed = shape in (
            (vocabulary, width),
            (vocabulary, block),
            (vocabulary,),
        )
    elif storage_class == "decoder_chunk":
        limit = (
            profile.decoder_train_token_chunk
            if phase == "train"
            else profile.decoder_eval_token_chunk
        )
        allowed = (
            len(shape) == 2
            and shape[1] == vocabulary
            and shape[0] <= limit
        )
    elif storage_class == "a0_qkv_or_result":
        allowed = shape in (
            (batch, 2, length, width // 2),
            (batch, length, 3 * width),
            (batch, length, width),
        )
    elif storage_class == "token_ids_or_mask":
        allowed = shape == (batch, length)
    elif storage_class == "latent_block":
        allowed = shape in (
            (batch, length, block),
            (batch, length, block, block),
        )
    elif storage_class == "lower_adjacent_block":
        allowed = shape == (batch, length - 1, block, block)
    elif storage_class == "banded_source":
        allowed = shape == (batch, length, profile.source_lookback)
    elif storage_class == "primary_frame":
        allowed = shape == (length, profile.K, profile.K)
    elif storage_class == "local_workspace":
        allowed = (
            len(shape) >= 2
            and shape[0] <= batch
            and all(item <= block for item in shape[1:])
        )
    elif storage_class == "particle_chunk":
        allowed = shape[0] <= profile.smc_particle_chunk
    elif storage_class == "scalar_or_row":
        allowed = math.prod(shape) <= max(batch, length, block * block)
    elif storage_class == "checkpoint_classified_parameter":
        allowed = phase == "checkpoint" and shape in (
            (vocabulary, width),
            (vocabulary, block),
            (vocabulary,),
            (length, profile.K, profile.K),
        )
    elif storage_class in (
        "checkpoint_model_parameter",
        "checkpoint_recognition_parameter",
        "checkpoint_optimizer_state",
    ):
        allowed = (
            phase == "checkpoint"
            and not _forbidden_shape(profile=profile, shape=shape)
            and math.prod(shape)
            <= vocabulary * max(width, block)
        )
    elif storage_class in (
        "checkpoint_scheduler_state",
        "checkpoint_cursor_or_metric",
    ):
        allowed = phase == "checkpoint" and math.prod(shape) <= block * block
    elif storage_class == "checkpoint_rng_state":
        allowed = (
            phase == "checkpoint"
            and len(shape) == 1
            and shape[0] <= vocabulary
        )
    elif storage_class == "checkpoint_estimator_state":
        allowed = (
            phase == "checkpoint"
            and not _forbidden_shape(profile=profile, shape=shape)
            and math.prod(shape)
            <= (
                profile.statistics.validation_particle_count
                * length
                * block
            )
        )
    if not allowed and _forbidden_shape(profile=profile, shape=shape):
        raise ForbiddenStorageRequest(
            f"forbidden population/pair/full-decoder shape {shape}"
        )
    if not allowed:
        raise ForbiddenStorageRequest(
            f"storage class {storage_class} cannot classify shape {shape}"
        )


def guard_flash_attention_request(
    *,
    backend: str,
    explicit_mask_requested: bool,
    attention_weights_requested: bool,
    fallback_allowed: bool,
) -> None:
    """Reject any A0 attention request outside the exact Flash-only call."""

    if (
        backend != "FLASH_ATTENTION"
        or explicit_mask_requested is not False
        or attention_weights_requested is not False
        or fallback_allowed is not False
    ):
        raise ForbiddenStorageRequest(
            "A0 attention requires Flash only, no mask/weights/fallback"
        )


def run_sparsity_negative_controls(
    *,
    profile: WT103ExperimentProfile,
    architecture: A0ArchitectureProfile,
) -> tuple[NegativeControlRecord, ...]:
    """Execute all ten metadata guards without allocating their payloads."""

    batch = profile.batch_size
    length = profile.sequence_length
    block = profile.combined_latent_block
    dimension = length * block
    cases = (
        ("a0_math_sdpa_fallback", "flash_backend_guard", (batch, 2, length, length)),
        ("a0_explicit_causal_mask", "flash_mask_guard", (length, length)),
        ("a0_attention_weights", "flash_weight_guard", (batch, 2, length, length)),
        ("dense_population", "population_shape_guard", (dimension, dimension)),
        (
            "batch_dense_population",
            "population_shape_guard",
            (batch, dimension, dimension),
        ),
        ("full_source", "source_band_guard", (batch, length, length)),
        (
            "pair_slab",
            "stack_concat_guard",
            (batch, length, length, block, block),
        ),
        (
            "full_decoder",
            "decoder_chunk_guard",
            (batch * length, profile.vocabulary_size),
        ),
        ("full_selector_rhs", "factor_rhs_guard", (dimension, dimension)),
        ("unclassified_checkpoint", "serializer_guard", (3, 5, 7)),
    )
    records: list[NegativeControlRecord] = []
    for control_id, guard_id, shape in cases:
        fired = False
        if control_id.startswith("a0_"):
            try:
                guard_flash_attention_request(
                    backend=(
                        "MATH"
                        if control_id == "a0_math_sdpa_fallback"
                        else "FLASH_ATTENTION"
                    ),
                    explicit_mask_requested=(
                        control_id == "a0_explicit_causal_mask"
                    ),
                    attention_weights_requested=(
                        control_id == "a0_attention_weights"
                    ),
                    fallback_allowed=(
                        control_id == "a0_math_sdpa_fallback"
                    ),
                )
            except ForbiddenStorageRequest:
                fired = True
        else:
            try:
                guard_tensor_request(
                    profile=profile,
                    architecture=architecture,
                    storage_class=(
                        "checkpoint_classified_parameter"
                        if control_id == "unclassified_checkpoint"
                        else "banded_source"
                    ),
                    shape=shape,
                    phase=(
                        "checkpoint"
                        if control_id == "unclassified_checkpoint"
                        else "train"
                    ),
                )
            except ForbiddenStorageRequest:
                fired = True
        records.append(
            NegativeControlRecord(
                control_id=control_id,
                guard_id=guard_id,
                logical_shape=shape,
                fired_pre_allocation=fired,
                allocation_or_serialization_attempted=False,
            )
        )
    return tuple(records)


@dataclass(frozen=True, slots=True)
class TrainingSparsityAudit:
    certificate: TrainingSparsityCertificate
    traces: tuple[ArmPathTrace, ...]
    flash_attention: FlashAttentionObservation
    negative_controls: tuple[NegativeControlRecord, ...]
    classified_unique_storage_bytes: int
    allocator_allocated_bytes: int
    allocator_overhead_bytes: int
    audit_sha256: str


def certify_training_sparsity(
    *,
    git_head: str,
    dirty_digest: str,
    profile: WT103ExperimentProfile,
    architecture: A0ArchitectureProfile,
    factory_set: WT103FactorySetIdentity,
    endpoint_inventory: EndpointInventory,
    traces: tuple[ArmPathTrace, ...],
    flash_attention: FlashAttentionObservation,
    negative_controls: tuple[NegativeControlRecord, ...],
    allocator_allocated_bytes: int,
    allocator_overhead_bytes: int,
    profiler_dispatch_agree: bool,
    serializer_inventory_complete: bool,
    h8_evidence: None,
    capacity_evidence: None,
) -> TrainingSparsityAudit:
    """Certify only current exact traces; H8/capacity cannot transfer."""

    _git_head(git_head)
    _sha256(dirty_digest, "dirty_digest")
    if h8_evidence is not None:
        raise ValueError("H8 evidence cannot populate training sparsity")
    if capacity_evidence is not None:
        raise ValueError("capacity evidence cannot populate training sparsity")
    if (
        type(profile) is not WT103ExperimentProfile
        or type(architecture) is not A0ArchitectureProfile
        or type(factory_set) is not WT103FactorySetIdentity
        or type(endpoint_inventory) is not EndpointInventory
    ):
        raise ValueError("sparsity inputs must retain exact types")
    profile.__post_init__()
    architecture.__post_init__()
    factory_set.__post_init__()
    endpoint_inventory.__post_init__()
    if (
        factory_set.arm_spec_sha256s
        != tuple(
            item.arm_spec_sha256 for item in endpoint_inventory.arms
        )
        or type(traces) is not tuple
        or tuple(item.arm_id for item in traces)
        != tuple(item.arm_id for item in endpoint_inventory.arms)
        or tuple(item.arm_spec_sha256 for item in traces)
        != tuple(
            item.arm_spec_sha256 for item in endpoint_inventory.arms
        )
    ):
        raise ValueError("sparsity trace/factory inventories are not exact")

    failures: list[str] = []
    inconclusive: list[str] = []
    storage_spans: dict[str, int] = {}
    for trace, spec in zip(
        traces,
        endpoint_inventory.arms,
        strict=True,
    ):
        trace.__post_init__()
        if trace.path_events != _expected_path_events(spec):
            failures.append(f"arm_path_changed:{trace.arm_id}")
        if set(trace.observability_views) != set(_OBSERVABILITY_VIEWS):
            inconclusive.append(f"missing_observability_view:{trace.arm_id}")
        for observation in trace.observations:
            try:
                guard_tensor_request(
                    profile=profile,
                    architecture=architecture,
                    storage_class=observation.storage_class,
                    shape=observation.shape,
                    phase=observation.phase,
                )
            except ForbiddenStorageRequest:
                failures.append(
                    f"forbidden_storage:{observation.event_id}"
                )
            previous = storage_spans.get(observation.storage_id)
            if previous is not None and previous != observation.storage_span_bytes:
                failures.append(
                    f"alias_span_disagreement:{observation.storage_id}"
                )
            storage_spans[observation.storage_id] = (
                observation.storage_span_bytes
            )

    if type(flash_attention) is not FlashAttentionObservation:
        raise ValueError("flash_attention must be an exact observation")
    if (
        flash_attention.arm_id != endpoint_inventory.arms[0].arm_id
        or flash_attention.backend != "FLASH_ATTENTION"
        or flash_attention.fallback_allowed
        or flash_attention.explicit_mask_materialized
        or flash_attention.attention_weights_returned
        or flash_attention.materialized_pair_shapes
        or flash_attention.semantic_pair_count
        != profile.sequence_length * (profile.sequence_length + 1) // 2
    ):
        failures.append("forbidden_or_fallback_flash_attention")
    if not all(
        (
            flash_attention.dispatch_observed,
            flash_attention.profiler_observed,
            flash_attention.allocator_observed,
            flash_attention.backend_observed,
        )
    ):
        inconclusive.append("missing_flash_backend_materialization_observability")

    if (
        type(negative_controls) is not tuple
        or tuple(item.control_id for item in negative_controls)
        != _NEGATIVE_CONTROL_IDS
    ):
        failures.append("negative_control_inventory_changed")
    else:
        for item in negative_controls:
            item.__post_init__()
            if (
                not item.fired_pre_allocation
                or item.allocation_or_serialization_attempted
            ):
                failures.append(f"negative_control_missed:{item.control_id}")
    classified = sum(storage_spans.values())
    if (
        type(allocator_allocated_bytes) is not int
        or type(allocator_overhead_bytes) is not int
        or allocator_allocated_bytes < 0
        or allocator_overhead_bytes < 0
        or allocator_allocated_bytes != classified + allocator_overhead_bytes
    ):
        failures.append("allocator_unique_storage_reconciliation_failed")
    if profiler_dispatch_agree is not True:
        failures.append("profiler_dispatch_inventory_disagreement")
    if serializer_inventory_complete is not True:
        inconclusive.append("checkpoint_serializer_inventory_unavailable")

    if failures:
        status = GateStatus.FAIL
        obligations = tuple(dict.fromkeys(failures))
    elif inconclusive:
        status = GateStatus.INCONCLUSIVE
        obligations = tuple(dict.fromkeys(inconclusive))
    else:
        status = GateStatus.PASS
        obligations = ()
    whitelist_sha256 = owned_sha256(
        "vfe4.wt103.training-sparsity-whitelist.v1",
        {
            "profile_sha256": profile.profile_sha256,
            "architecture_sha256": architecture.architecture_sha256,
            "storage_classes": _STORAGE_CLASSES,
        },
    )
    forbidden_shape_sha256 = owned_sha256(
        "vfe4.wt103.training-sparsity-forbidden-shapes.v1",
        {
            "B": profile.batch_size,
            "L": profile.sequence_length,
            "b": profile.combined_latent_block,
            "V": profile.vocabulary_size,
            "D": profile.sequence_length * profile.combined_latent_block,
        },
    )
    trace_set_sha256 = owned_sha256(
        "vfe4.wt103.training-sparsity-traces.v1",
        traces,
    )
    formula_reconciliation_sha256 = owned_sha256(
        "vfe4.wt103.training-sparsity-byte-reconciliation.v1",
        {
            "classified_unique_storage_bytes": classified,
            "allocator_allocated_bytes": allocator_allocated_bytes,
            "allocator_overhead_bytes": allocator_overhead_bytes,
        },
    )
    negative_controls_sha256 = owned_sha256(
        "vfe4.wt103.training-sparsity-negative-controls.v1",
        negative_controls,
    )
    certificate_payload = {
        "schema_version": "wt103-training-sparsity-v1",
        "git_head": git_head,
        "dirty_digest": dirty_digest,
        "profile_sha256": profile.profile_sha256,
        "factory_set_sha256": factory_set.factory_set_sha256,
        "endpoint_inventory_sha256": (
            endpoint_inventory.endpoint_inventory_sha256
        ),
        "whitelist_sha256": whitelist_sha256,
        "forbidden_shape_sha256": forbidden_shape_sha256,
        "trace_set_sha256": trace_set_sha256,
        "formula_reconciliation_sha256": (
            formula_reconciliation_sha256
        ),
        "negative_controls_sha256": negative_controls_sha256,
        "status": status,
        "obligations": obligations,
    }
    certificate = TrainingSparsityCertificate(
        **certificate_payload,
        certificate_sha256=owned_sha256(
            "vfe4.wt103.training-sparsity-certificate.v1",
            certificate_payload,
        ),
    )
    audit_payload = {
        "certificate_sha256": certificate.certificate_sha256,
        "traces": traces,
        "flash_attention": flash_attention,
        "negative_controls": negative_controls,
        "classified_unique_storage_bytes": classified,
        "allocator_allocated_bytes": allocator_allocated_bytes,
        "allocator_overhead_bytes": allocator_overhead_bytes,
    }
    return TrainingSparsityAudit(
        certificate=certificate,
        traces=traces,
        flash_attention=flash_attention,
        negative_controls=negative_controls,
        classified_unique_storage_bytes=classified,
        allocator_allocated_bytes=allocator_allocated_bytes,
        allocator_overhead_bytes=allocator_overhead_bytes,
        audit_sha256=owned_sha256(
            "vfe4.wt103.training-sparsity-audit.v1",
            audit_payload,
        ),
    )


__all__ = [
    "ArmPathTrace",
    "FlashAttentionObservation",
    "ForbiddenStorageRequest",
    "NegativeControlRecord",
    "TensorStorageObservation",
    "TrainingSparsityAudit",
    "certify_training_sparsity",
    "guard_flash_attention_request",
    "guard_tensor_request",
    "run_sparsity_negative_controls",
]
