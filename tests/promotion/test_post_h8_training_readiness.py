from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

from vfe4.artifacts.durability import DurabilityIdentity, VolumeFacts
from vfe4.artifacts.environment import (
    AllocationObservation,
    ComponentBenchmark,
    DependencyLockIdentity,
    DiskByteForecast,
    DistributionIdentity,
    EnvironmentObservation,
    LockInputManifest,
    LockRequirement,
    PowerProviderIdentity,
    ResourceWorkload,
    TrainingExecutionIdentity,
    capture_environment,
    forecast_resources,
    required_resource_components,
    render_dependency_lock,
    run_allocation_preflight,
)
from vfe4.artifacts.manifest import ArtifactIntegrityRecord
from vfe4.artifacts.provenance import (
    build_training_provenance,
    capture_git_identity,
    production_token_cache_set_sha256,
)
from vfe4.artifacts.readiness import (
    PostH8ReadinessToken,
    ReadinessValidationError,
    Task14LiveIntegrationEvidence,
    validate_post_h8_readiness,
)
import vfe4.artifacts.readiness as readiness_module
from vfe4.training.readiness import StaticScientificPreconditionRecord
from vfe4.types.results import GateStatus
from vfe4.types.training import (
    ArchiveMemberIdentity,
    EndpointInventory,
    EstimatorProtocol,
    FinalizedWikiText103SourceRecord,
    ProductionTokenCacheIdentity,
    ProductionTokenizerSpec,
    ResourceProfile,
    SyntheticFixtureTokenCacheIdentity,
    SyntheticFixtureTokenizerSpec,
    TrainingSparsityCertificate,
    WT103_CONFIRMATORY_SEED_IDS,
    WT103_TUNING_CELLS,
    WT103_TUNING_SEED_IDS,
    default_wt103_arm_specs,
    default_wt103_gate_specs,
    owned_sha256,
    production_tokenizer_tables_sha256,
)


def _sha(character: str) -> str:
    return character * 64


def _inventory() -> EndpointInventory:
    return EndpointInventory.create(
        default_wt103_arm_specs(),
        default_wt103_gate_specs(),
        WT103_TUNING_CELLS,
        WT103_TUNING_SEED_IDS,
        WT103_CONFIRMATORY_SEED_IDS,
        EstimatorProtocol.create(),
    )


def _tokenizer() -> ProductionTokenizerSpec:
    table_facts = {
        "regex_pattern_sha256": _sha("2"),
        "regex_engine_distribution_name": "regex",
        "regex_engine_distribution_version": "2026.1.1",
        "regex_engine_distribution_record_sha256": _sha("3"),
        "mergeable_ranks_sha256": _sha("4"),
        "special_tokens_sha256": _sha("5"),
        "golden_vectors_sha256": _sha("6"),
    }
    return ProductionTokenizerSpec.create_verified(
        distribution_record_sha256=_sha("1"),
        **table_facts,
        tokenizer_tables_sha256=production_tokenizer_tables_sha256(
            **table_facts
        ),
    )


def _caches(
    tokenizer: ProductionTokenizerSpec,
) -> tuple[ProductionTokenCacheIdentity, ...]:
    return tuple(
        ProductionTokenCacheIdentity.create(
            tokenizer=tokenizer,
            split=split,
            payload_sha256=payload_sha,
        )
        for split, payload_sha in (
            ("train", _sha("6")),
            ("validation", _sha("7")),
            ("test", _sha("8")),
        )
    )


def _dependency_lock(
    *,
    installed_match: bool = True,
) -> DependencyLockIdentity:
    locked = DistributionIdentity(
        name="torch",
        version="2.10.0.dev20251210+cu128",
        record_sha256=_sha("9"),
    )
    installed = (
        locked
        if installed_match
        else DistributionIdentity(
            name="torch",
            version="2.9.1",
            record_sha256=_sha("a"),
        )
    )
    manifest = LockInputManifest.create(
        writer_code_sha256=_sha("8"),
        target_python_version="3.13",
        requirements=(
            LockRequirement(
                name=locked.name,
                version=locked.version,
                environment_marker='python_version >= "3.10"',
                artifact_filename="torch-fixture.whl",
                artifact_url="https://example.invalid/torch-fixture.whl",
                artifact_size_bytes=123,
                artifact_sha256s=(_sha("7"),),
                expected_installed_record_sha256=(
                    locked.record_sha256
                ),
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


def _source(
    tokenizer: ProductionTokenizerSpec,
    caches: tuple[ProductionTokenCacheIdentity, ...],
    dependency: DependencyLockIdentity,
    *,
    dependency_lock_sha256: str | None = None,
    tokenizer_tables_sha256: str | None = None,
) -> FinalizedWikiText103SourceRecord:
    members = tuple(
        ArchiveMemberIdentity(
            split=split,
            member_name=f"wikitext-103/{split}.txt",
            compression_method=8,
            compressed_size_bytes=10,
            uncompressed_size_bytes=20,
            crc32=ordinal,
            payload_sha256=payload_sha,
        )
        for ordinal, (split, payload_sha) in enumerate(
            (
                ("train", _sha("b")),
                ("validation", _sha("c")),
                ("test", _sha("d")),
            ),
            start=1,
        )
    )
    return FinalizedWikiText103SourceRecord.create(
        acquisition_observation_sha256=_sha("e"),
        archive_request_url="https://example.invalid/archive.zip",
        archive_final_url="https://example.invalid/archive.zip",
        archive_redirect_chain=(),
        source_page_request_url="https://example.invalid/source",
        source_page_final_url="https://example.invalid/source",
        source_page_redirect_chain=(),
        archive_size_bytes=30,
        archive_sha256=_sha("f"),
        archive_content_type="application/zip",
        central_directory_sha256=_sha("0"),
        members=members,
        source_page_size_bytes=100,
        source_page_content_type="text/html",
        source_page_sha256=_sha("1"),
        license_paragraph_start_byte=10,
        license_paragraph_end_byte=20,
        license_raw_slice_sha256=_sha("2"),
        license_declaration="Creative Commons Attribution-ShareAlike",
        license_hrefs=("https://example.invalid/license",),
        installed_distribution_sha256=_sha("3"),
        tokenizer_tables_sha256=(
            tokenizer.tokenizer_tables_sha256
            if tokenizer_tables_sha256 is None
            else tokenizer_tables_sha256
        ),
        production_tokenizer_spec_sha256=tokenizer.spec_sha256,
        production_token_cache_set_sha256=(
            production_token_cache_set_sha256(caches)
        ),
        schedule_set_sha256=_sha("5"),
        dependency_lock_sha256=(
            dependency.lock_sha256
            if dependency_lock_sha256 is None
            else dependency_lock_sha256
        ),
        validator_sha256=_sha("6"),
    )


def _environment(dependency: DependencyLockIdentity):
    return capture_environment(
        EnvironmentObservation(
            captured_utc="2026-07-28T05:00:00Z",
            device_work_started=False,
            python_version="3.13.5",
            pytorch_version="2.10.0.dev20251210+cu128",
            cuda_runtime_version="12.8",
            cudnn_version="91002",
            driver_version="576.80",
            os_name="nt",
            platform_system="Windows",
            platform_release="11",
            cpu_name="fixture-cpu",
            logical_cpu_count=8,
            physical_ram_bytes=64 * 1024**3,
            gpu_names=("fixture-gpu",),
            gpu_device_uuids=("GPU-fixture-0001",),
            gpu_total_bytes=(32 * 1024**3,),
            compute_capabilities=("12.0",),
            blas_identity_sha256=_sha("7"),
            thread_settings_sha256=_sha("8"),
            deterministic_algorithms=True,
            cudnn_benchmark=False,
            locale_name="en_US",
            timezone_name="America/Chicago",
        ),
        dependency_lock=dependency,
    )


def _sparsity(
    inventory: EndpointInventory,
) -> TrainingSparsityCertificate:
    payload = {
        "schema_version": "wt103-training-sparsity-v1",
        "git_head": "1" * 40,
        "dirty_digest": _sha("2"),
        "profile_sha256": _sha("3"),
        "factory_set_sha256": _sha("4"),
        "endpoint_inventory_sha256": inventory.endpoint_inventory_sha256,
        "whitelist_sha256": _sha("5"),
        "forbidden_shape_sha256": _sha("6"),
        "trace_set_sha256": _sha("7"),
        "formula_reconciliation_sha256": _sha("8"),
        "negative_controls_sha256": _sha("9"),
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


def _static_science(
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
        "architecture_sha256": _sha("a"),
        "formula_sha256": _sha("b"),
        "factory_set_sha256": sparsity.factory_set_sha256,
        "endpoint_inventory_sha256": inventory.endpoint_inventory_sha256,
        "objective_sha256": _sha("c"),
        "update_policy_sha256": _sha("d"),
        "snapshot_policy_sha256": _sha("e"),
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


def _integrity(path: str, payload: bytes) -> ArtifactIntegrityRecord:
    return ArtifactIntegrityRecord.create(
        kind="file",
        relative_path=path,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _resource_profile() -> ResourceProfile:
    return ResourceProfile(720.0, 840.0, 500.0, 1.25, 0.85, 100)


def _capacity(
    inventory: EndpointInventory,
    execution_identity: TrainingExecutionIdentity,
    environment,
):
    observations = tuple(
        AllocationObservation.shape_identical_for_arm(
            arm,
            execution_identity=execution_identity,
            device_ordinal=0,
            device_uuid="GPU-fixture-0001",
            physical_device_bytes=32 * 1024**3,
            peak_device_allocated_bytes=8_000,
            peak_device_reserved_bytes=8_500,
            host_available_bytes=5_000,
            checkpoint_duplicate_bytes=2_000,
            disk_available_bytes=5_000,
        )
        for arm in inventory.arms
    )
    return run_allocation_preflight(
        endpoint_inventory=inventory,
        observations=observations,
        execution_identity=execution_identity,
        environment=environment,
        maximum_device_fraction=0.85,
        h8_allocation_evidence=None,
    )


def _forecast(
    inventory: EndpointInventory,
    execution_identity: TrainingExecutionIdentity,
    *,
    power_available: bool = True,
):
    workload = ResourceWorkload(
        train_batches_per_pass=4,
        validation_batches_per_full_evaluation=2,
        test_batches_per_full_evaluation=3,
        preparation_source_work_units=1,
        preparation_tokenizer_work_units=1,
        preparation_window_work_units=1,
    )
    components = required_resource_components(inventory, workload)
    power = PowerProviderIdentity.create(
        provider_kind="nvidia-smi",
        provider_version="fixture-1",
        provider_executable_sha256=_sha("a"),
        sample_interval_ms=100,
        reported_power_limit_watts=600.0,
    )
    benchmarks = tuple(
        ComponentBenchmark.observed_for(
            component,
            execution_identity=execution_identity,
            minimum_throughput_per_second=1_000_000.0,
            maximum_duration_seconds=0.001,
            maximum_board_power_watts=(
                500.0
                if component.uses_gpu and power_available
                else None
                if component.uses_gpu
                else 0.0
            ),
            power_provider=(
                power
                if component.uses_gpu and power_available
                else None
            ),
        )
        for component in components
    )
    disk = DiskByteForecast.create(
        archive_staging_bytes=1,
        extracted_member_bytes=2,
        int32_token_cache_bytes=3,
        schedule_bytes=4,
        retained_checkpoint_bytes=5,
        jsonl_csv_bytes=6,
        test_record_bytes=7,
        figure_bytes=8,
    )
    return forecast_resources(
        endpoint_inventory=inventory,
        workload=workload,
        component_benchmarks=benchmarks,
        execution_identity=execution_identity,
        disk_forecast=disk,
        available_disk_bytes=disk.required_available_bytes,
        resource_profile=_resource_profile(),
        power_provider=power if power_available else None,
    )


def _durability() -> DurabilityIdentity:
    volume = VolumeFacts(
        volume_path="/fixture",
        volume_serial="fixture-volume",
        filesystem_type="ext4",
        is_remote=False,
    )
    return DurabilityIdentity.create(
        backend_kind="posix",
        implementation_sha256=_sha("b"),
        volume=volume,
        create_sha256=_sha("c"),
        replace_sha256=_sha("d"),
        errors=(),
        obligations=(),
    )


def _ready_inputs(
    *,
    installed_match: bool = True,
    power_available: bool = True,
    source_lock_sha256: str | None = None,
    source_tokenizer_tables_sha256: str | None = None,
) -> dict[str, object]:
    inventory = _inventory()
    tokenizer = _tokenizer()
    caches = _caches(tokenizer)
    dependency = _dependency_lock(installed_match=installed_match)
    source = _source(
        tokenizer,
        caches,
        dependency,
        dependency_lock_sha256=source_lock_sha256,
        tokenizer_tables_sha256=source_tokenizer_tables_sha256,
    )
    environment = _environment(dependency)
    git_identity = capture_git_identity(
        git_head_value="1" * 40,
        dirty_digest_value=_sha("2"),
        status_porcelain=b"",
    )
    sparsity = _sparsity(inventory)
    static_scientific = _static_science(inventory, sparsity)
    execution_identity = TrainingExecutionIdentity.create(
        git_identity_sha256=git_identity.identity_sha256,
        git_head=git_identity.git_head,
        dirty_digest=git_identity.dirty_digest,
        config_sha256=_sha("e"),
        profile_sha256=static_scientific.profile_sha256,
        factory_set_sha256=static_scientific.factory_set_sha256,
        environment_sha256=environment.environment_sha256,
    )
    provenance = build_training_provenance(
        git_identity=git_identity,
        environment=environment,
        source_record_sha256=source.record_sha256,
        tokenizer_spec_sha256=tokenizer.spec_sha256,
        token_cache_set_sha256=production_token_cache_set_sha256(caches),
        schedule_set_sha256=source.schedule_set_sha256,
        config_sha256=_sha("e"),
        objective_sha256=_sha("c"),
        factory_set_sha256=_sha("4"),
        endpoint_inventory_sha256=inventory.endpoint_inventory_sha256,
        data_integrity=_integrity("data/source.json", b"source"),
        evidence_integrity=(_integrity("evidence/h8.json", b"h8"),),
        inventory_integrity=_integrity(
            "inventory/endpoints.json",
            b"inventory",
        ),
        parent_checkpoint=None,
    )
    return {
        "static_scientific": static_scientific,
        "training_sparsity": sparsity,
        "finalized_source": source,
        "tokenizer": tokenizer,
        "token_caches": caches,
        "dependency_lock": dependency,
        "durability": _durability(),
        "allocation_preflight": _capacity(
            inventory,
            execution_identity,
            environment,
        ),
        "environment": environment,
        "resource_forecast": _forecast(
            inventory,
            execution_identity,
            power_available=power_available,
        ),
        "provenance": provenance,
        "endpoint_inventory": inventory,
        "h8_allocation_evidence": None,
    }


def test_final_readiness_composes_task6_one_way_without_issuing_task10_token() -> None:
    inputs = _ready_inputs()

    assessment = validate_post_h8_readiness(
        **inputs,
        task14_evidence=None,
    )

    assert assessment.status is GateStatus.INCONCLUSIVE
    assert assessment.obligations == (
        "task14_live_integration_evidence_missing",
    )
    assert assessment.token is None
    assert assessment.production_token_issued is False
    assert len(assessment.assessment_sha256) == 64


def test_final_readiness_is_inconclusive_for_lock_or_power_obligations() -> None:
    lock_mismatch = validate_post_h8_readiness(
        **_ready_inputs(installed_match=False),
        task14_evidence=None,
    )
    assert lock_mismatch.status is GateStatus.INCONCLUSIVE
    assert "dependency_lock_not_exact_match" in lock_mismatch.obligations

    source_mismatch = validate_post_h8_readiness(
        **_ready_inputs(source_lock_sha256=_sha("f")),
        task14_evidence=None,
    )
    assert source_mismatch.status is GateStatus.INCONCLUSIVE
    assert "source_dependency_lock_mismatch" in source_mismatch.obligations

    tokenizer_tables_mismatch = validate_post_h8_readiness(
        **_ready_inputs(source_tokenizer_tables_sha256=_sha("f")),
        task14_evidence=None,
    )
    assert tokenizer_tables_mismatch.status is GateStatus.INCONCLUSIVE
    assert "source_tokenizer_mismatch" in tokenizer_tables_mismatch.obligations

    missing_power = validate_post_h8_readiness(
        **_ready_inputs(power_available=False),
        task14_evidence=None,
    )
    assert missing_power.status is GateStatus.INCONCLUSIVE
    assert "resource_forecast_not_pass" in missing_power.obligations


def test_synthetic_tokenizer_and_h8_allocation_can_never_issue_readiness() -> None:
    inputs = _ready_inputs()
    synthetic_tokenizer = SyntheticFixtureTokenizerSpec.create(
        adapter_sha256=_sha("1"),
        fixture_sha256=_sha("2"),
    )
    synthetic_cache = SyntheticFixtureTokenCacheIdentity.create(
        tokenizer=synthetic_tokenizer,
        payload_sha256=_sha("3"),
    )
    inputs["tokenizer"] = synthetic_tokenizer
    inputs["token_caches"] = (synthetic_cache,)

    with pytest.raises(ReadinessValidationError, match="production tokenizer"):
        validate_post_h8_readiness(
            **inputs,
            task14_evidence=None,
        )

    h8_inputs = _ready_inputs()
    h8_inputs["h8_allocation_evidence"] = object()
    with pytest.raises(ReadinessValidationError, match="H8 allocation"):
        validate_post_h8_readiness(
            **h8_inputs,
            task14_evidence=None,
        )


def test_task10_cannot_spoof_task14_authority_and_constructors_are_sealed() -> None:
    with pytest.raises(ReadinessValidationError, match="Task 10 cannot"):
        validate_post_h8_readiness(
            **_ready_inputs(),
            task14_evidence="task14_live_integration",  # type: ignore[arg-type]
        )
    with pytest.raises(ReadinessValidationError, match="Task 14"):
        PostH8ReadinessToken()  # type: ignore[call-arg]
    with pytest.raises(ReadinessValidationError, match="Task 14"):
        Task14LiveIntegrationEvidence()  # type: ignore[call-arg]


def test_task10_has_no_live_evidence_or_production_token_issuer() -> None:
    source = Path(readiness_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "_seal_task14_live_integration_evidence" not in functions
    assert "_issue_token" not in functions
    assert "_TASK14_EVIDENCE_CONSTRUCTOR_KEY" not in source
    assert "_TOKEN_CONSTRUCTOR_KEY" not in source


@pytest.mark.parametrize(
    "field",
    (
        "git_identity_sha256",
        "git_head",
        "dirty_digest",
        "config_sha256",
        "profile_sha256",
        "factory_set_sha256",
        "environment_sha256",
    ),
)
def test_readiness_rejects_capacity_evidence_from_another_execution(
    field: str,
) -> None:
    inputs = _ready_inputs()
    current = inputs["allocation_preflight"].execution_identity
    values = {
        name: getattr(current, name)
        for name in (
            "git_identity_sha256",
            "git_head",
            "dirty_digest",
            "config_sha256",
            "profile_sha256",
            "factory_set_sha256",
            "environment_sha256",
        )
    }
    values[field] = "2" * 40 if field == "git_head" else _sha("f")
    mismatched = TrainingExecutionIdentity.create(**values)
    inventory = inputs["endpoint_inventory"]
    if field == "environment_sha256":
        with pytest.raises(ValueError, match="captured environment"):
            _capacity(
                inventory,
                mismatched,
                inputs["environment"],
            )
        return
    inputs["allocation_preflight"] = _capacity(
        inventory,
        mismatched,
        inputs["environment"],
    )
    inputs["resource_forecast"] = _forecast(
        inventory,
        mismatched,
    )

    assessment = validate_post_h8_readiness(
        **inputs,
        task14_evidence=None,
    )

    assert assessment.status is GateStatus.INCONCLUSIVE
    assert (
        "allocation_execution_identity_mismatch"
        in assessment.obligations
    )
    assert "resource_execution_identity_mismatch" in assessment.obligations


def test_dependency_direction_has_one_way_task6_composition_and_no_live_dependencies() -> None:
    import vfe4.artifacts.environment as environment
    import vfe4.artifacts.provenance as provenance
    import vfe4.artifacts.readiness as readiness
    import vfe4.artifacts.run_directory as run_directory
    import vfe4.training.readiness as static_readiness

    static_tree = ast.parse(
        Path(static_readiness.__file__).read_text(encoding="utf-8")
    )
    static_imports = {
        node.module
        for node in ast.walk(static_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "vfe4.artifacts.readiness" not in static_imports

    forbidden = ("tiktoken", "vfe4.figures")
    for module in (environment, provenance, readiness, run_directory):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imports.update(
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        )
        assert not any(
            name == blocked or name.startswith(f"{blocked}.")
            for name in imports
            for blocked in forbidden
        )
