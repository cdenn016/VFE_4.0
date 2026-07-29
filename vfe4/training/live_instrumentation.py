"""Live, revision-bound storage and profiler instrumentation for WT103 arms."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import (
    Callable,
    Generic,
    Iterator,
    Literal,
    TypeVar,
    cast,
)

from vfe4.artifacts.live_readiness import (
    Task14ReadinessBundle,
    publish_task14_readiness_bundle,
    reopen_and_issue_task14_readiness,
)
from vfe4.types.training import (
    A0ArchitectureProfile,
    EndpointInventory,
    WT103ArmSpec,
    WT103ExperimentProfile,
)

from .factories import WT103FactorySetIdentity
from .sparsity import (
    ArmPathTrace,
    FlashAttentionObservation,
    TensorStorageObservation,
    TrainingSparsityAudit,
    certify_training_sparsity,
    guard_flash_attention_request,
    guard_tensor_request,
    run_sparsity_negative_controls,
)


_T = TypeVar("_T")
LiveProbeAuthority = Literal[
    "production_shape_identical",
    "nonproduction_test_adapter",
]


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
class ProfiledLivePath(Generic[_T]):
    """One complete arm path collected under dispatch/profiler/allocator views."""

    authority: LiveProbeAuthority
    trace: ArmPathTrace
    flash_attention: FlashAttentionObservation | None
    profiler_marker_names: tuple[str, ...]
    profiler_dispatch_agree: bool
    serializer_inventory_complete: bool
    serializer_unique_tensor_count: int
    serializer_inventory_sha256: str | None
    unique_storage_bytes: int
    unique_cuda_storage_bytes: int
    allocator_allocated_bytes: int
    allocator_overhead_bytes: int
    peak_device_allocated_bytes: int
    peak_device_reserved_bytes: int
    checkpoint_duplicate_bytes: int
    operation_result: _T

    def __post_init__(self) -> None:
        if self.authority not in (
            "production_shape_identical",
            "nonproduction_test_adapter",
        ):
            raise ValueError("unknown live-probe authority")
        if type(self.trace) is not ArmPathTrace:
            raise ValueError("trace must be exact")
        self.trace.__post_init__()
        if (
            self.flash_attention is not None
            and type(self.flash_attention) is not FlashAttentionObservation
        ):
            raise ValueError("flash observation must be exact")
        if (
            type(self.profiler_marker_names) is not tuple
            or any(
                type(value) is not str or not value
                for value in self.profiler_marker_names
            )
            or type(self.profiler_dispatch_agree) is not bool
            or type(self.serializer_inventory_complete) is not bool
        ):
            raise ValueError("live observability fields are malformed")
        for name in (
            "serializer_unique_tensor_count",
            "unique_storage_bytes",
            "unique_cuda_storage_bytes",
            "allocator_allocated_bytes",
            "allocator_overhead_bytes",
            "peak_device_allocated_bytes",
            "peak_device_reserved_bytes",
            "checkpoint_duplicate_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be an exact nonnegative int")
        if self.serializer_inventory_complete:
            if (
                self.serializer_unique_tensor_count <= 0
                or type(self.serializer_inventory_sha256) is not str
                or len(self.serializer_inventory_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in self.serializer_inventory_sha256
                )
            ):
                raise ValueError(
                    "complete serializer inventory must carry a count and SHA-256"
                )
        elif (
            self.serializer_unique_tensor_count != 0
            or self.serializer_inventory_sha256 is not None
        ):
            raise ValueError(
                "incomplete serializer inventory cannot carry claimed identity"
            )
        if (
            self.unique_cuda_storage_bytes > self.unique_storage_bytes
            or self.allocator_allocated_bytes
            != self.unique_storage_bytes + self.allocator_overhead_bytes
        ):
            raise ValueError("live allocator/storage bytes do not reconcile")
        if (
            self.authority == "production_shape_identical"
            and (
                not self.profiler_dispatch_agree
                or not self.serializer_inventory_complete
                or self.serializer_unique_tensor_count <= 0
            )
        ):
            raise ValueError("production live path lacks required observability")


class LivePathObserver:
    """Runtime observer that classifies real tensor storage as it is used."""

    def __init__(
        self,
        *,
        arm: WT103ArmSpec,
        profile: WT103ExperimentProfile,
        architecture: A0ArchitectureProfile,
    ) -> None:
        if type(arm) is not WT103ArmSpec:
            raise ValueError("arm must be exact")
        if type(profile) is not WT103ExperimentProfile:
            raise ValueError("profile must be exact")
        if type(architecture) is not A0ArchitectureProfile:
            raise ValueError("architecture must be exact")
        arm.__post_init__()
        profile.__post_init__()
        architecture.__post_init__()
        self._arm = arm
        self._profile = profile
        self._architecture = architecture
        self._expected_path_events = _expected_path_events(arm)
        self._marked_events: list[str] = []
        self._current_event: str | None = None
        self._observations: list[TensorStorageObservation] = []
        self._storage_ids: dict[tuple[str, int, int], str] = {}
        self._storage_devices: dict[str, str] = {}
        self._flash_attention: FlashAttentionObservation | None = None
        self._checkpoint_duplicate_bytes = 0

    @property
    def expected_path_events(self) -> tuple[str, ...]:
        return self._expected_path_events

    @property
    def flash_attention(self) -> FlashAttentionObservation | None:
        return self._flash_attention

    @property
    def checkpoint_duplicate_bytes(self) -> int:
        return self._checkpoint_duplicate_bytes

    def mark_path_event(self, name: str) -> None:
        """Mark one runtime dispatch event and emit a profiler-visible marker."""

        if name not in self._expected_path_events:
            raise ValueError(f"unexpected live path event: {name}")
        import torch

        with torch.profiler.record_function(f"vfe4.path.{name}"):
            pass
        self._marked_events.append(name)
        self._current_event = name

    @contextmanager
    def path_event(self, name: str) -> Iterator[None]:
        """Wrap an operation in the exact profiler marker for its dispatch row."""

        if name not in self._expected_path_events:
            raise ValueError(f"unexpected live path event: {name}")
        if self._current_event is not None:
            raise ValueError("live path events cannot be nested")
        import torch

        self._marked_events.append(name)
        self._current_event = name
        try:
            with torch.profiler.record_function(f"vfe4.path.{name}"):
                yield
        finally:
            self._current_event = None

    def guard_shape(
        self,
        *,
        storage_class: str,
        shape: tuple[int, ...],
        phase: Literal["train", "evaluation", "checkpoint"],
    ) -> None:
        """Run the sparsity guard before the runtime allocates a tensor."""

        guard_tensor_request(
            profile=self._profile,
            architecture=self._architecture,
            storage_class=storage_class,
            shape=shape,
            phase=phase,
        )

    def observe_tensor(
        self,
        tensor: object,
        storage_class: str,
        logical_axes: tuple[str, ...],
        phase: str,
        event_id: str,
    ) -> None:
        """Classify one real tensor and preserve alias-aware storage identity."""

        import torch

        if not isinstance(tensor, torch.Tensor):
            raise ValueError("observed value must be a torch.Tensor")
        if self._current_event is None:
            raise ValueError("tensor observation requires a marked path event")
        if phase not in ("train", "evaluation", "checkpoint"):
            raise ValueError("unknown live tensor phase")
        typed_phase = cast(
            Literal["train", "evaluation", "checkpoint"],
            phase,
        )
        shape = tuple(int(value) for value in tensor.shape)
        self.guard_shape(
            storage_class=storage_class,
            shape=shape,
            phase=typed_phase,
        )
        if (
            type(logical_axes) is not tuple
            or len(logical_axes) != len(shape)
            or any(type(value) is not str or not value for value in logical_axes)
        ):
            raise ValueError("logical axes do not match observed tensor")
        if type(event_id) is not str or not event_id:
            raise ValueError("event_id must be nonempty text")
        storage = tensor.untyped_storage()
        span = int(storage.nbytes())
        data_ptr = int(storage.data_ptr())
        device = str(tensor.device)
        storage_key = (device, data_ptr, span)
        storage_id = self._storage_ids.get(storage_key)
        if storage_id is None:
            storage_id = (
                f"{self._arm.arm_id}/storage-{len(self._storage_ids):08d}"
            )
            self._storage_ids[storage_key] = storage_id
            self._storage_devices[storage_id] = device
        numel = int(tensor.numel())
        element_size = int(tensor.element_size())
        self._observations.append(
            TensorStorageObservation(
                event_id=event_id,
                arm_id=self._arm.arm_id,
                path_event=self._current_event,
                phase=typed_phase,
                storage_id=storage_id,
                storage_class=storage_class,
                shape=shape,
                logical_axes=logical_axes,
                numel=numel,
                element_size_bytes=element_size,
                logical_bytes=numel * element_size,
                storage_span_bytes=span,
            )
        )

    def observe_flash_attention(
        self,
        *,
        sequence_length: int,
        backend: str,
        fallback_allowed: bool,
        explicit_mask_materialized: bool,
        attention_weights_returned: bool,
    ) -> None:
        """Bind the runtime call-site facts for the A0 Flash-only operator."""

        if self._arm.arm_id != "WT103-A0-AR-v1":
            raise ValueError("Flash attention observation belongs only to A0")
        if (
            type(sequence_length) is not int
            or sequence_length != self._profile.sequence_length
        ):
            raise ValueError("Flash sequence length differs from the profile")
        guard_flash_attention_request(
            backend=backend,
            explicit_mask_requested=explicit_mask_materialized,
            attention_weights_requested=attention_weights_returned,
            fallback_allowed=fallback_allowed,
        )
        self._flash_attention = FlashAttentionObservation(
            arm_id=self._arm.arm_id,
            backend=backend,
            fallback_allowed=fallback_allowed,
            explicit_mask_materialized=explicit_mask_materialized,
            attention_weights_returned=attention_weights_returned,
            materialized_pair_shapes=(),
            semantic_pair_count=(
                sequence_length * (sequence_length + 1) // 2
            ),
            dispatch_observed=True,
            profiler_observed=False,
            allocator_observed=False,
            backend_observed=False,
        )

    def record_checkpoint_duplicate_bytes(self, value: int) -> None:
        if type(value) is not int or value <= 0:
            raise ValueError(
                "checkpoint duplicate bytes must be an exact positive int"
            )
        self._checkpoint_duplicate_bytes = value

    def _trace(self) -> ArmPathTrace:
        if not self._observations:
            raise ValueError("live path emitted no classified tensor storage")
        trace = ArmPathTrace.exact_for_arm(
            self._arm,
            observations=tuple(self._observations),
        )
        if trace.path_events != self._expected_path_events:
            raise ValueError("live observer path vocabulary drifted")
        return trace

    def _storage_bytes(self) -> tuple[int, int]:
        unique: dict[str, int] = {}
        for observation in self._observations:
            previous = unique.get(observation.storage_id)
            if (
                previous is not None
                and previous != observation.storage_span_bytes
            ):
                raise ValueError("aliased storage span changed during probe")
            unique[observation.storage_id] = observation.storage_span_bytes
        total = sum(unique.values())
        cuda = sum(
            span
            for storage_id, span in unique.items()
            if self._storage_devices[storage_id].startswith("cuda")
        )
        return total, cuda


def _flash_profiler_seen(names: tuple[str, ...]) -> bool:
    lowered = tuple(name.lower() for name in names)
    return any(
        "flash_attention" in name
        or "flashattention" in name
        or "_scaled_dot_product_flash" in name
        for name in lowered
    )


def collect_profiled_live_path(
    *,
    arm: WT103ArmSpec,
    profile: WT103ExperimentProfile,
    architecture: A0ArchitectureProfile,
    operation: Callable[[LivePathObserver], _T],
    authority: LiveProbeAuthority,
    require_cuda: bool,
    device_ordinal: int = 0,
) -> ProfiledLivePath[_T]:
    """Execute one complete arm path under a real PyTorch profiler."""

    if authority not in (
        "production_shape_identical",
        "nonproduction_test_adapter",
    ):
        raise ValueError("unknown live-probe authority")
    if type(require_cuda) is not bool:
        raise ValueError("require_cuda must be an exact bool")
    if (
        authority == "production_shape_identical"
        and require_cuda is not True
    ):
        raise ValueError("production live probes require CUDA")
    if type(device_ordinal) is not int or device_ordinal < 0:
        raise ValueError("device_ordinal must be an exact nonnegative int")
    if not callable(operation):
        raise ValueError("operation must be callable")
    import torch

    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for production live probe")
    observer = LivePathObserver(
        arm=arm,
        profile=profile,
        architecture=architecture,
    )
    activities = [torch.profiler.ProfilerActivity.CPU]
    if require_cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        torch.cuda.synchronize(device_ordinal)
        torch.cuda.reset_peak_memory_stats(device_ordinal)
    with torch.profiler.profile(activities=activities) as profiler:
        operation_result = operation(observer)
    if require_cuda:
        torch.cuda.synchronize(device_ordinal)
        peak_allocated = int(
            torch.cuda.max_memory_allocated(device_ordinal)
        )
        peak_reserved = int(torch.cuda.max_memory_reserved(device_ordinal))
    else:
        peak_allocated = 0
        peak_reserved = 0
    profiler_names = tuple(
        sorted({str(item.key) for item in profiler.key_averages()})
    )
    marker_names = tuple(
        name for name in profiler_names if name.startswith("vfe4.path.")
    )
    expected_markers = tuple(
        f"vfe4.path.{name}" for name in observer.expected_path_events
    )
    marked_exact = (
        tuple(dict.fromkeys(observer._marked_events))
        == observer.expected_path_events
    )
    profiler_dispatch_agree = (
        marked_exact and set(marker_names) == set(expected_markers)
    )
    flash = observer.flash_attention
    if flash is not None:
        profiler_seen = _flash_profiler_seen(profiler_names)
        flash = FlashAttentionObservation(
            arm_id=flash.arm_id,
            backend=flash.backend,
            fallback_allowed=flash.fallback_allowed,
            explicit_mask_materialized=flash.explicit_mask_materialized,
            attention_weights_returned=flash.attention_weights_returned,
            materialized_pair_shapes=flash.materialized_pair_shapes,
            semantic_pair_count=flash.semantic_pair_count,
            dispatch_observed=flash.dispatch_observed,
            profiler_observed=profiler_seen,
            allocator_observed=require_cuda,
            backend_observed=profiler_seen,
        )
    if arm.arm_id == "WT103-A0-AR-v1" and authority == "production_shape_identical":
        if flash is None or not all(
            (
                flash.profiler_observed,
                flash.allocator_observed,
                flash.backend_observed,
            )
        ):
            raise ValueError("production A0 probe did not prove Flash dispatch")
    trace = observer._trace()
    unique_storage, unique_cuda_storage = observer._storage_bytes()
    allocator_overhead = max(
        0,
        peak_allocated - unique_cuda_storage,
    )
    allocator_allocated = unique_storage + allocator_overhead
    checkpoint_bytes = observer.checkpoint_duplicate_bytes
    result_checkpoint_bytes = getattr(
        operation_result,
        "checkpoint_duplicate_bytes",
        None,
    )
    if type(result_checkpoint_bytes) is int and result_checkpoint_bytes > 0:
        if checkpoint_bytes not in (0, result_checkpoint_bytes):
            raise ValueError("checkpoint byte observations disagree")
        checkpoint_bytes = result_checkpoint_bytes
    result_serializer_complete = getattr(
        operation_result,
        "serializer_inventory_complete",
        None,
    )
    result_serializer_count = getattr(
        operation_result,
        "serializer_unique_tensor_count",
        None,
    )
    result_serializer_sha256 = getattr(
        operation_result,
        "serializer_inventory_sha256",
        None,
    )
    serializer_complete = (
        "checkpoint_serialization" in observer._marked_events
        and checkpoint_bytes > 0
        and result_serializer_complete is True
        and type(result_serializer_count) is int
        and result_serializer_count > 0
        and type(result_serializer_sha256) is str
        and len(result_serializer_sha256) == 64
        and all(
            character in "0123456789abcdef"
            for character in result_serializer_sha256
        )
    )
    if authority == "production_shape_identical" and not serializer_complete:
        raise ValueError(
            "live probe did not provide a complete authenticated serializer inventory"
        )
    serializer_count = (
        cast(int, result_serializer_count) if serializer_complete else 0
    )
    serializer_sha256 = (
        cast(str, result_serializer_sha256) if serializer_complete else None
    )
    return ProfiledLivePath(
        authority=authority,
        trace=trace,
        flash_attention=flash,
        profiler_marker_names=marker_names,
        profiler_dispatch_agree=profiler_dispatch_agree,
        serializer_inventory_complete=serializer_complete,
        serializer_unique_tensor_count=serializer_count,
        serializer_inventory_sha256=serializer_sha256,
        unique_storage_bytes=unique_storage,
        unique_cuda_storage_bytes=unique_cuda_storage,
        allocator_allocated_bytes=allocator_allocated,
        allocator_overhead_bytes=allocator_overhead,
        peak_device_allocated_bytes=peak_allocated,
        peak_device_reserved_bytes=peak_reserved,
        checkpoint_duplicate_bytes=checkpoint_bytes,
        operation_result=operation_result,
    )


def certify_profiled_runtime_set(
    *,
    git_head: str,
    dirty_digest: str,
    profile: WT103ExperimentProfile,
    architecture: A0ArchitectureProfile,
    factory_set: WT103FactorySetIdentity,
    endpoint_inventory: EndpointInventory,
    paths: tuple[ProfiledLivePath[object], ...],
) -> TrainingSparsityAudit:
    """Close the sparsity certificate from all five clean-child path reports."""

    if type(profile) is not WT103ExperimentProfile:
        raise ValueError("profile must be exact")
    if type(architecture) is not A0ArchitectureProfile:
        raise ValueError("architecture must be exact")
    if type(factory_set) is not WT103FactorySetIdentity:
        raise ValueError("factory_set must be exact")
    if type(endpoint_inventory) is not EndpointInventory:
        raise ValueError("endpoint_inventory must be exact")
    profile.__post_init__()
    architecture.__post_init__()
    factory_set.__post_init__()
    endpoint_inventory.__post_init__()
    if (
        type(paths) is not tuple
        or len(paths) != len(endpoint_inventory.arms)
        or any(type(item) is not ProfiledLivePath for item in paths)
    ):
        raise ValueError("profiled path inventory is not exact")
    for path, arm in zip(paths, endpoint_inventory.arms, strict=True):
        path.__post_init__()
        if (
            path.authority != "production_shape_identical"
            or path.trace.arm_id != arm.arm_id
            or path.trace.arm_spec_sha256 != arm.arm_spec_sha256
        ):
            raise ValueError(
                "profiled path is not production-authorized for its arm"
            )
    flash = paths[0].flash_attention
    if type(flash) is not FlashAttentionObservation:
        raise ValueError("A0 production path lacks exact Flash evidence")
    return certify_training_sparsity(
        git_head=git_head,
        dirty_digest=dirty_digest,
        profile=profile,
        architecture=architecture,
        factory_set=factory_set,
        endpoint_inventory=endpoint_inventory,
        traces=tuple(item.trace for item in paths),
        flash_attention=flash,
        negative_controls=run_sparsity_negative_controls(
            profile=profile,
            architecture=architecture,
        ),
        allocator_allocated_bytes=sum(
            item.allocator_allocated_bytes for item in paths
        ),
        allocator_overhead_bytes=sum(
            item.allocator_overhead_bytes for item in paths
        ),
        profiler_dispatch_agree=all(
            item.profiler_dispatch_agree for item in paths
        ),
        serializer_inventory_complete=all(
            item.serializer_inventory_complete for item in paths
        ),
        h8_evidence=None,
        capacity_evidence=None,
    )
__all__ = [
    "LivePathObserver",
    "LiveProbeAuthority",
    "ProfiledLivePath",
    "Task14ReadinessBundle",
    "certify_profiled_runtime_set",
    "collect_profiled_live_path",
    "publish_task14_readiness_bundle",
    "reopen_and_issue_task14_readiness",
]
