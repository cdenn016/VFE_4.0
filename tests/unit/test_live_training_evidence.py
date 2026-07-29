from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from vfe4.artifacts.durability import (
    DurabilityIdentity,
    VolumeFacts,
    canonical_json_bytes_generic,
)
from vfe4.artifacts.environment import (
    AllocationPreflightRecord,
    ComponentForecast,
    DependencyLockIdentity,
    DistributionIdentity,
    EnvironmentObservation,
    LockInputManifest,
    LockRequirement,
    PowerProviderIdentity,
    ResourceForecast,
    ResourceWorkload,
    TrainingExecutionIdentity,
    capture_environment,
    render_dependency_lock,
    required_resource_components,
)
from vfe4.artifacts.live_readiness import (
    Task14ReadinessBundle,
    read_task14_readiness_bundle,
)
from vfe4.artifacts.provenance import TrainingProvenanceRecord
from vfe4.artifacts.readiness import ReadinessValidationError
from vfe4.artifacts.live_environment import (
    benchmark_live_component,
    measure_shape_identical_allocation,
)
from vfe4.config import (
    default_training_config_mapping,
    resolve_training_config,
)
from vfe4.training.formulas import (
    A0FlopWorkload,
    build_a0_architecture_profile,
    build_a0_formula_record,
    reconstruct_a0_flops,
    reconstruct_a0_parameters,
)
from vfe4.training.live_instrumentation import (
    LivePathObserver,
    collect_profiled_live_path,
)
from vfe4.training.readiness import StaticScientificPreconditionRecord
from vfe4.training.sparsity import ForbiddenStorageRequest
from vfe4.types.results import GateStatus
from vfe4.types.training import (
    EndpointInventory,
    TrainingSparsityCertificate,
    owned_sha256,
)
from vfe4.training.wt103_models import WT103A0Model


def _sha(character: str) -> str:
    return character * 64


def _resolved_and_architecture():
    resolved = resolve_training_config(default_training_config_mapping())
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
        source_lock_scope="candidate_unverified",
        pytorch_version="unresolved_until_task13_source_lock",
        sdpa_api_sha256="unresolved_until_task13_source_lock",
        flash_backend_sha256="unresolved_until_task13_source_lock",
    )
    return resolved, architecture


def _dependency_lock() -> DependencyLockIdentity:
    installed = DistributionIdentity(
        name="fixture",
        version="1.0",
        record_sha256=_sha("6"),
    )
    manifest = LockInputManifest.create(
        writer_code_sha256=_sha("7"),
        target_python_version="3.12",
        requirements=(
            LockRequirement(
                name="fixture",
                version="1.0",
                environment_marker='python_version >= "3.10"',
                artifact_filename="fixture-1.0.whl",
                artifact_url="https://example.invalid/fixture-1.0.whl",
                artifact_size_bytes=123,
                artifact_sha256s=(_sha("8"),),
                expected_installed_record_sha256=installed.record_sha256,
                task13_obligation=None,
            ),
        ),
    )
    lock_bytes = render_dependency_lock(manifest)
    return DependencyLockIdentity.capture(
        lock_relative_path="requirements-wt103.lock",
        lock_bytes=lock_bytes,
        expected_sha256=hashlib.sha256(lock_bytes).hexdigest(),
        lock_input_manifest=manifest,
        installed_distributions=(installed,),
    )


def _environment_and_execution():
    dependency = _dependency_lock()
    environment = capture_environment(
        EnvironmentObservation(
            captured_utc="2026-07-28T04:00:00Z",
            device_work_started=False,
            python_version="3.12.0",
            pytorch_version="fixture",
            cuda_runtime_version="12.8",
            cudnn_version="fixture",
            driver_version="fixture",
            os_name="nt",
            platform_system="Windows",
            platform_release="11",
            cpu_name="fixture-cpu",
            logical_cpu_count=8,
            physical_ram_bytes=100_000,
            gpu_names=("fixture-gpu",),
            gpu_device_uuids=("GPU-fixture-0001",),
            gpu_total_bytes=(10_000,),
            compute_capabilities=("12.0",),
            blas_identity_sha256=_sha("9"),
            thread_settings_sha256=_sha("a"),
            deterministic_algorithms=True,
            cudnn_benchmark=False,
            locale_name="en_US",
            timezone_name="America/Chicago",
        ),
        dependency_lock=dependency,
    )
    execution = TrainingExecutionIdentity.create(
        git_identity_sha256=_sha("1"),
        git_head="1" * 40,
        dirty_digest=_sha("2"),
        config_sha256=_sha("3"),
        profile_sha256=_sha("4"),
        factory_set_sha256=_sha("5"),
        environment_sha256=environment.environment_sha256,
    )
    return environment, execution


def _sparsity(
    inventory: EndpointInventory,
) -> TrainingSparsityCertificate:
    payload = {
        "schema_version": "wt103-training-sparsity-v1",
        "git_head": "1" * 40,
        "dirty_digest": _sha("2"),
        "profile_sha256": _sha("4"),
        "factory_set_sha256": _sha("5"),
        "endpoint_inventory_sha256": (
            inventory.endpoint_inventory_sha256
        ),
        "whitelist_sha256": _sha("6"),
        "forbidden_shape_sha256": _sha("7"),
        "trace_set_sha256": _sha("8"),
        "formula_reconciliation_sha256": _sha("9"),
        "negative_controls_sha256": _sha("a"),
        "status": GateStatus.PASS,
        "obligations": (),
    }
    return TrainingSparsityCertificate(
        **payload,
        certificate_sha256=owned_sha256(
            "vfe4.wt103.training-sparsity-certificate.v1",
            payload,
        ),
    )


def _static_scientific(
    inventory: EndpointInventory,
    sparsity: TrainingSparsityCertificate,
) -> StaticScientificPreconditionRecord:
    payload = {
        "schema_version": "wt103-static-scientific-preconditions-v1",
        "git_head": "1" * 40,
        "dirty_digest": _sha("2"),
        "h6_prediction_schema": "h6-prediction-result-v3",
        "h8_schema": "h8-sparse-scale-v5",
        "profile_sha256": sparsity.profile_sha256,
        "architecture_sha256": _sha("b"),
        "formula_sha256": _sha("c"),
        "factory_set_sha256": sparsity.factory_set_sha256,
        "endpoint_inventory_sha256": (
            inventory.endpoint_inventory_sha256
        ),
        "objective_sha256": _sha("d"),
        "update_policy_sha256": _sha("e"),
        "snapshot_policy_sha256": _sha("f"),
        "estimator_protocol_sha256": (
            inventory.estimator_protocol_sha256
        ),
        "predecessor_reference_sha256s": (
            _sha("0"),
            _sha("1"),
            _sha("2"),
            _sha("3"),
            _sha("4"),
        ),
        "predictor_safety_sha256": _sha("5"),
        "training_sparsity_sha256": sparsity.certificate_sha256,
        "status": GateStatus.PASS,
        "obligations": (),
        "production_readiness_token_issued": False,
    }
    return StaticScientificPreconditionRecord(
        **payload,
        record_sha256=owned_sha256(
            "vfe4.wt103.static-scientific-preconditions.v1",
            payload,
        ),
    )


def _allocation_preflight(
    inventory: EndpointInventory,
    environment_sha256: str,
    execution: TrainingExecutionIdentity,
) -> AllocationPreflightRecord:
    payload = {
        "schema_version": "wt103-allocation-preflight-v1",
        "endpoint_inventory_sha256": (
            inventory.endpoint_inventory_sha256
        ),
        "environment_sha256": environment_sha256,
        "execution_identity": execution,
        "observation_sha256s": (_sha("a"),),
        "maximum_device_fraction": 0.85,
        "maximum_peak_allocated_fraction": 0.1,
        "maximum_peak_reserved_fraction": 0.2,
        "status": GateStatus.PASS,
        "obligations": (),
        "h8_evidence_accepted": False,
    }
    return AllocationPreflightRecord(
        **payload,
        record_sha256=owned_sha256(
            "vfe4.wt103.allocation-preflight.v1",
            payload,
        ),
    )


def _resource_forecast(
    inventory: EndpointInventory,
    execution: TrainingExecutionIdentity,
) -> ResourceForecast:
    component = ComponentForecast(
        component_id="task14-fixture",
        component_spec_sha256=_sha("a"),
        benchmark_sha256=_sha("b"),
        predicted_seconds=3_600.0,
        predicted_gpu_seconds=3_600.0,
    )
    payload = {
        "schema_version": "wt103-resource-forecast-v1",
        "endpoint_inventory_sha256": (
            inventory.endpoint_inventory_sha256
        ),
        "execution_identity": execution,
        "component_forecasts": (component,),
        "component_benchmark_sha256s": (component.benchmark_sha256,),
        "disk_forecast_sha256": _sha("c"),
        "available_disk_bytes": 1,
        "raw_gpu_hours": 1.0,
        "raw_wall_hours": 1.0,
        "raw_energy_kwh": 0.1,
        "forecast_gpu_hours": 1.25,
        "forecast_wall_hours": 1.25,
        "forecast_energy_kwh": 0.125,
        "maximum_gpu_hours": 720.0,
        "maximum_wall_hours": 840.0,
        "maximum_energy_kwh": 500.0,
        "forecast_headroom_factor": 1.25,
        "power_provider_identity_sha256": _sha("d"),
        "maximum_observed_board_power_watts": 100.0,
        "reported_power_limit_watts": 100.0,
        "conservative_power_watts": 100.0,
        "status": GateStatus.PASS,
        "obligations": (),
    }
    return ResourceForecast(
        **payload,
        forecast_sha256=owned_sha256(
            "vfe4.wt103.resource-forecast.v1",
            payload,
        ),
    )


def _provenance(
    inventory: EndpointInventory,
    environment,
) -> TrainingProvenanceRecord:
    payload = {
        "schema_version": "wt103-training-provenance-v1",
        "git_identity_sha256": _sha("1"),
        "git_head": "1" * 40,
        "dirty_digest": _sha("2"),
        "source_is_clean": False,
        "environment_sha256": environment.environment_sha256,
        "hardware_identity_sha256": (
            environment.hardware_identity_sha256
        ),
        "runtime_identity_sha256": environment.runtime_identity_sha256,
        "dependency_lock_identity_sha256": (
            environment.dependency_lock_identity_sha256
        ),
        "source_record_sha256": _sha("6"),
        "tokenizer_spec_sha256": _sha("7"),
        "token_cache_set_sha256": _sha("8"),
        "schedule_set_sha256": _sha("9"),
        "config_sha256": _sha("3"),
        "objective_sha256": _sha("d"),
        "factory_set_sha256": _sha("5"),
        "endpoint_inventory_sha256": (
            inventory.endpoint_inventory_sha256
        ),
        "artifact_integrity_sha256s": (
            _sha("a"),
            _sha("b"),
            _sha("c"),
        ),
        "parent_checkpoint_identity_sha256": None,
    }
    return TrainingProvenanceRecord(
        **payload,
        provenance_sha256=owned_sha256(
            "vfe4.wt103.training-provenance.v1",
            payload,
        ),
    )


def _task14_bundle() -> Task14ReadinessBundle:
    resolved, _ = _resolved_and_architecture()
    inventory = resolved.endpoint_inventory
    environment, execution = _environment_and_execution()
    sparsity = _sparsity(inventory)
    return Task14ReadinessBundle.create(
        training_config_sha256=_sha("3"),
        source_lock_sha256=_sha("4"),
        static_scientific=_static_scientific(inventory, sparsity),
        training_sparsity=sparsity,
        dependency_lock=_dependency_lock(),
        durability=DurabilityIdentity.create(
            backend_kind="posix",
            implementation_sha256=_sha("5"),
            volume=VolumeFacts(
                volume_path="/fixture",
                volume_serial="fixture-volume",
                filesystem_type="ext4",
                is_remote=False,
            ),
            create_sha256=_sha("6"),
            replace_sha256=_sha("7"),
            errors=(),
            obligations=(),
        ),
        allocation_preflight=_allocation_preflight(
            inventory,
            environment.environment_sha256,
            execution,
        ),
        environment=environment,
        resource_forecast=_resource_forecast(inventory, execution),
        provenance=_provenance(inventory, environment),
        endpoint_inventory=inventory,
        live_integration_artifact_sha256=_sha("8"),
    )


class _ExclusiveFileBackend:
    def create_exclusive(self, path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
        assert path.read_bytes() == payload


def test_task14_bundle_is_published_as_canonical_typed_evidence_and_reopens(
    tmp_path: Path,
) -> None:
    bundle = _task14_bundle()
    payload = canonical_json_bytes_generic(bundle)
    path = tmp_path / "task14-readiness.json"

    _ExclusiveFileBackend().create_exclusive(path, payload)
    reopened = read_task14_readiness_bundle(path)

    assert path.read_bytes() == payload
    assert type(reopened) is Task14ReadinessBundle
    assert reopened == bundle


def test_task14_reader_rejects_duplicate_unknown_noncanonical_and_tampered_json(
    tmp_path: Path,
) -> None:
    raw = canonical_json_bytes_generic(_task14_bundle())
    duplicate = (
        b'{"schema_version":"wt103-task14-readiness-bundle-v1",'
        + raw[1:]
    )
    unknown_mapping = json.loads(raw)
    unknown_mapping["unknown_task14_field"] = True
    tampered_mapping = json.loads(raw)
    tampered_mapping["training_config_sha256"] = _sha("f")
    cases = {
        "duplicate.json": duplicate,
        "unknown.json": canonical_json_bytes_generic(unknown_mapping),
        "noncanonical.json": raw + b"\n",
        "tampered.json": canonical_json_bytes_generic(tampered_mapping),
    }

    for name, payload in cases.items():
        path = tmp_path / name
        path.write_bytes(payload)
        with pytest.raises(ReadinessValidationError):
            read_task14_readiness_bundle(path)


def test_live_instrumentation_compatibility_exports_are_canonical() -> None:
    from vfe4.artifacts import live_readiness as canonical
    from vfe4.training import live_instrumentation as compatibility

    assert (
        compatibility.Task14ReadinessBundle
        is canonical.Task14ReadinessBundle
    )
    assert (
        compatibility.publish_task14_readiness_bundle
        is canonical.publish_task14_readiness_bundle
    )
    assert (
        compatibility.reopen_and_issue_task14_readiness
        is canonical.reopen_and_issue_task14_readiness
    )


def test_live_readiness_import_does_not_load_training_instrumentation() -> None:
    command = (
        "import sys\n"
        "import vfe4.artifacts.live_readiness\n"
        "assert 'vfe4.training.live_instrumentation' not in sys.modules\n"
    )

    completed = subprocess.run(
        (sys.executable, "-c", command),
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_live_observer_profiles_real_storage_and_rejects_dense_requests() -> None:
    resolved, architecture = _resolved_and_architecture()
    arm = resolved.endpoint_inventory.arms[-1]

    def operation(observer: LivePathObserver) -> int:
        value = torch.ones(1, dtype=torch.float32)
        for event in observer.expected_path_events:
            with observer.path_event(event):
                if event == "metric_failure_write":
                    observer.observe_tensor(
                        value,
                        event_id="metric-value",
                        storage_class="scalar_or_row",
                        logical_axes=("metric",),
                        phase="evaluation",
                    )
                    observer.observe_tensor(
                        value.view(1),
                        event_id="metric-alias",
                        storage_class="scalar_or_row",
                        logical_axes=("metric",),
                        phase="evaluation",
                    )
        return 7

    result = collect_profiled_live_path(
        arm=arm,
        profile=resolved.profile,
        architecture=architecture,
        operation=operation,
        authority="nonproduction_test_adapter",
        require_cuda=False,
    )

    assert result.operation_result == 7
    assert result.profiler_dispatch_agree is True
    assert result.serializer_inventory_complete is False
    assert result.serializer_unique_tensor_count == 0
    assert result.serializer_inventory_sha256 is None
    assert result.unique_storage_bytes == 4
    assert len(result.trace.observations) == 2
    assert (
        result.trace.observations[0].storage_id
        == result.trace.observations[1].storage_id
    )
    with pytest.raises(ValueError, match="observability"):
        dataclasses.replace(
            result,
            authority="production_shape_identical",
        )

    def serialized_operation(observer: LivePathObserver) -> SimpleNamespace:
        value = torch.ones(1, dtype=torch.float32)
        for event in observer.expected_path_events:
            with observer.path_event(event):
                if event == "metric_failure_write":
                    observer.observe_tensor(
                        value,
                        event_id="serialized-metric",
                        storage_class="scalar_or_row",
                        logical_axes=("metric",),
                        phase="evaluation",
                    )
                if event == "checkpoint_serialization":
                    observer.record_checkpoint_duplicate_bytes(4)
        return SimpleNamespace(
            checkpoint_duplicate_bytes=4,
            serializer_inventory_complete=True,
            serializer_unique_tensor_count=1,
            serializer_inventory_sha256=_sha("a"),
        )

    serialized = collect_profiled_live_path(
        arm=arm,
        profile=resolved.profile,
        architecture=architecture,
        operation=serialized_operation,
        authority="nonproduction_test_adapter",
        require_cuda=False,
    )
    assert serialized.serializer_inventory_complete is True
    assert serialized.serializer_unique_tensor_count == 1
    assert serialized.serializer_inventory_sha256 == _sha("a")
    production_record = dataclasses.replace(
        serialized,
        authority="production_shape_identical",
    )
    assert production_record.serializer_inventory_complete is True
    with pytest.raises(ValueError, match="count and SHA-256"):
        dataclasses.replace(
            serialized,
            serializer_unique_tensor_count=0,
        )
    with pytest.raises(ValueError, match="claimed identity"):
        dataclasses.replace(
            serialized,
            serializer_inventory_complete=False,
        )

    observer = LivePathObserver(
        arm=arm,
        profile=resolved.profile,
        architecture=architecture,
    )
    with pytest.raises(ForbiddenStorageRequest):
        observer.guard_shape(
            storage_class="banded_source",
            shape=(
                resolved.profile.batch_size,
                resolved.profile.sequence_length,
                resolved.profile.sequence_length,
            ),
            phase="train",
        )


class _FakeAllocationBackend:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.sync_calls = 0

    def reset_peak_memory_stats(self, device_ordinal: int) -> None:
        assert device_ordinal == 0
        self.reset_calls += 1

    def synchronize(self, device_ordinal: int) -> None:
        assert device_ordinal == 0
        self.sync_calls += 1

    def peak_memory_allocated(self, device_ordinal: int) -> int:
        return 7_500

    def peak_memory_reserved(self, device_ordinal: int) -> int:
        return 8_000

    def device_uuid(self, device_ordinal: int) -> str:
        return "GPU-fixture-0001"

    def physical_device_bytes(self, device_ordinal: int) -> int:
        return 10_000

    def host_available_bytes(self) -> int:
        return 9_000

    def disk_available_bytes(self, path: Path) -> int:
        assert path == Path("artifact-root")
        return 8_000


def test_live_allocation_uses_the_captured_device_denominator() -> None:
    resolved, _ = _resolved_and_architecture()
    environment, execution = _environment_and_execution()
    backend = _FakeAllocationBackend()
    calls: list[str] = []

    observation = measure_shape_identical_allocation(
        spec=resolved.endpoint_inventory.arms[0],
        execution_identity=execution,
        environment=environment,
        device_ordinal=0,
        checkpoint_duplicate_bytes=1_000,
        checkpoint_root=Path("artifact-root"),
        operation=lambda: calls.append("ran"),
        backend=backend,
    )

    assert calls == ["ran"]
    assert observation.device_uuid == "GPU-fixture-0001"
    assert observation.physical_device_bytes == 10_000
    assert observation.peak_device_allocated_bytes == 7_500
    assert observation.peak_device_reserved_bytes == 8_000
    assert backend.reset_calls == backend.sync_calls == 1

    backend.physical_device_bytes = lambda ordinal: 20_000  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="captured environment"):
        measure_shape_identical_allocation(
            spec=resolved.endpoint_inventory.arms[0],
            execution_identity=execution,
            environment=environment,
            device_ordinal=0,
            checkpoint_duplicate_bytes=1_000,
            checkpoint_root=Path("artifact-root"),
            operation=lambda: None,
            backend=backend,
        )


class _FakePowerSampler:
    def sample(self, operation):
        result = operation()
        return result, (110.0, 125.0, 120.0)


def test_live_benchmark_runs_exact_frozen_counts_and_requires_power_evidence() -> None:
    resolved, _ = _resolved_and_architecture()
    _, execution = _environment_and_execution()
    workload = ResourceWorkload(
        train_batches_per_pass=4,
        validation_batches_per_full_evaluation=2,
        test_batches_per_full_evaluation=2,
        preparation_source_work_units=1,
        preparation_tokenizer_work_units=1,
        preparation_window_work_units=1,
    )
    component = next(
        item
        for item in required_resource_components(
            resolved.endpoint_inventory,
            workload,
        )
        if item.component_id
        == f"tuning/train/{resolved.endpoint_inventory.arms[0].arm_id}"
    )
    provider = PowerProviderIdentity.create(
        provider_kind="nvidia-smi",
        provider_version="fixture",
        provider_executable_sha256=_sha("f"),
        sample_interval_ms=100,
        reported_power_limit_watts=600.0,
    )
    calls: list[int] = []

    with pytest.raises(ValueError, match="power"):
        benchmark_live_component(
            component=component,
            execution_identity=execution,
            operation=lambda: 1,
            power_provider=None,
            power_sampler=None,
        )

    benchmark = benchmark_live_component(
        component=component,
        execution_identity=execution,
        operation=lambda: calls.append(1) or 1,
        power_provider=provider,
        power_sampler=_FakePowerSampler(),
    )

    assert len(calls) == component.warmup_count + component.sample_count
    assert benchmark.maximum_board_power_watts == 125.0
    assert benchmark.minimum_throughput_per_second > 0.0
    assert benchmark.maximum_duration_seconds >= 0.0
