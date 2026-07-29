"""Canonical durable Task 14 readiness artifacts and token issuance."""

from __future__ import annotations

import dataclasses
import json
import types
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal, Union, get_args, get_origin, get_type_hints

from vfe4.artifacts.durability import (
    DurabilityBackend,
    DurabilityIdentity,
    canonical_json_bytes_generic,
)
from vfe4.artifacts.environment import (
    AllocationPreflightRecord,
    DependencyLockIdentity,
    EnvironmentRecord,
    ResourceForecast,
    TrainingExecutionIdentity,
)
from vfe4.artifacts.provenance import TrainingProvenanceRecord
from vfe4.artifacts.readiness import (
    PostH8ReadinessAssessment,
    PostH8ReadinessToken,
    ReadinessValidationError,
    Task14LiveIntegrationEvidence,
    validate_post_h8_readiness,
)
from vfe4.config.schema import TrainingConfig
from vfe4.training.readiness import StaticScientificPreconditionRecord
from vfe4.types.results import GateStatus
from vfe4.types.training import (
    EndpointInventory,
    TrainingSparsityCertificate,
    owned_sha256,
)


@dataclass(frozen=True, slots=True)
class Task14ReadinessBundle:
    """Canonical durable inputs to the sole Task 14 readiness issuer."""

    schema_version: Literal["wt103-task14-readiness-bundle-v1"]
    training_config_sha256: str
    source_lock_sha256: str
    static_scientific: StaticScientificPreconditionRecord
    training_sparsity: TrainingSparsityCertificate
    dependency_lock: DependencyLockIdentity
    durability: DurabilityIdentity
    allocation_preflight: AllocationPreflightRecord
    environment: EnvironmentRecord
    resource_forecast: ResourceForecast
    provenance: TrainingProvenanceRecord
    endpoint_inventory: EndpointInventory
    live_integration_artifact_sha256: str
    bundle_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in tuple(self.__dataclass_fields__)[:-1]
        }

    def __post_init__(self) -> None:
        if self.schema_version != "wt103-task14-readiness-bundle-v1":
            raise ReadinessValidationError(
                "Task 14 readiness bundle schema is invalid"
            )
        exact = (
            (self.static_scientific, StaticScientificPreconditionRecord),
            (self.training_sparsity, TrainingSparsityCertificate),
            (self.dependency_lock, DependencyLockIdentity),
            (self.durability, DurabilityIdentity),
            (self.allocation_preflight, AllocationPreflightRecord),
            (self.environment, EnvironmentRecord),
            (self.resource_forecast, ResourceForecast),
            (self.provenance, TrainingProvenanceRecord),
            (self.endpoint_inventory, EndpointInventory),
        )
        if any(type(value) is not expected for value, expected in exact):
            raise ReadinessValidationError(
                "Task 14 readiness bundle contains a nonexact record"
            )
        for value, _ in exact:
            validator = getattr(value, "__post_init__", None)
            if validator is not None:
                validator()
        for name in (
            "training_config_sha256",
            "source_lock_sha256",
            "live_integration_artifact_sha256",
            "bundle_sha256",
        ):
            value = getattr(self, name)
            if (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ReadinessValidationError(
                    f"{name} must be a lowercase SHA-256"
                )
        execution = self.allocation_preflight.execution_identity
        if (
            self.static_scientific.status is not GateStatus.PASS
            or self.training_sparsity.status is not GateStatus.PASS
            or self.static_scientific.training_sparsity_sha256
            != self.training_sparsity.certificate_sha256
            or self.static_scientific.endpoint_inventory_sha256
            != self.endpoint_inventory.endpoint_inventory_sha256
            or self.training_sparsity.endpoint_inventory_sha256
            != self.endpoint_inventory.endpoint_inventory_sha256
            or self.allocation_preflight.endpoint_inventory_sha256
            != self.endpoint_inventory.endpoint_inventory_sha256
            or self.resource_forecast.endpoint_inventory_sha256
            != self.endpoint_inventory.endpoint_inventory_sha256
            or self.allocation_preflight.status is not GateStatus.PASS
            or self.resource_forecast.status is not GateStatus.PASS
            or execution != self.resource_forecast.execution_identity
            or execution.environment_sha256 != self.environment.environment_sha256
            or self.provenance.environment_sha256
            != self.environment.environment_sha256
            or self.provenance.git_head != self.static_scientific.git_head
            or self.provenance.dirty_digest
            != self.static_scientific.dirty_digest
            or self.provenance.config_sha256 != self.training_config_sha256
            or self.provenance.endpoint_inventory_sha256
            != self.endpoint_inventory.endpoint_inventory_sha256
        ):
            raise ReadinessValidationError(
                "Task 14 readiness bundle cross-links are not exact PASS"
            )
        expected = owned_sha256(
            "vfe4.wt103.task14-readiness-bundle.v1",
            self.semantic_payload(),
        )
        if self.bundle_sha256 != expected:
            raise ReadinessValidationError(
                "Task 14 readiness bundle hash does not match"
            )

    @classmethod
    def create(
        cls,
        *,
        training_config_sha256: str,
        source_lock_sha256: str,
        static_scientific: StaticScientificPreconditionRecord,
        training_sparsity: TrainingSparsityCertificate,
        dependency_lock: DependencyLockIdentity,
        durability: DurabilityIdentity,
        allocation_preflight: AllocationPreflightRecord,
        environment: EnvironmentRecord,
        resource_forecast: ResourceForecast,
        provenance: TrainingProvenanceRecord,
        endpoint_inventory: EndpointInventory,
        live_integration_artifact_sha256: str,
    ) -> "Task14ReadinessBundle":
        payload = {
            "schema_version": "wt103-task14-readiness-bundle-v1",
            "training_config_sha256": training_config_sha256,
            "source_lock_sha256": source_lock_sha256,
            "static_scientific": static_scientific,
            "training_sparsity": training_sparsity,
            "dependency_lock": dependency_lock,
            "durability": durability,
            "allocation_preflight": allocation_preflight,
            "environment": environment,
            "resource_forecast": resource_forecast,
            "provenance": provenance,
            "endpoint_inventory": endpoint_inventory,
            "live_integration_artifact_sha256": (
                live_integration_artifact_sha256
            ),
        }
        return cls(
            **payload,
            bundle_sha256=owned_sha256(
                "vfe4.wt103.task14-readiness-bundle.v1",
                payload,
            ),
        )  # type: ignore[arg-type]


def _reject_duplicate_json_pairs(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ReadinessValidationError(
                f"duplicate Task 14 JSON key: {key}"
            )
        result[key] = value
    return result


def _decode_typed(annotation: object, value: object, *, name: str) -> object:
    """Strictly reconstruct a fixed annotated dataclass tree from JSON."""

    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin is Literal:
        if not any(type(value) is type(item) and value == item for item in arguments):
            raise ReadinessValidationError(f"{name} differs from its Literal")
        return value
    if origin is tuple:
        if type(value) is not list:
            raise ReadinessValidationError(f"{name} must be a JSON array")
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return tuple(
                _decode_typed(arguments[0], item, name=f"{name}[{index}]")
                for index, item in enumerate(value)
            )
        if len(value) != len(arguments):
            raise ReadinessValidationError(f"{name} tuple length changed")
        return tuple(
            _decode_typed(expected, item, name=f"{name}[{index}]")
            for index, (expected, item) in enumerate(
                zip(arguments, value, strict=True)
            )
        )
    if origin in (types.UnionType, Union):
        if value is None and type(None) in arguments:
            return None
        failures: list[BaseException] = []
        for expected in arguments:
            if expected is type(None):
                continue
            try:
                return _decode_typed(expected, value, name=name)
            except (ReadinessValidationError, TypeError, ValueError) as exc:
                failures.append(exc)
        raise ReadinessValidationError(
            f"{name} does not match any allowed union member"
        ) from (failures[-1] if failures else None)
    if origin is dict:
        if type(value) is not dict or len(arguments) != 2:
            raise ReadinessValidationError(f"{name} must be a typed mapping")
        key_type, item_type = arguments
        if key_type is not str:
            raise ReadinessValidationError(
                f"{name} mapping keys are not fixed strings"
            )
        return {
            key: _decode_typed(item_type, item, name=f"{name}.{key}")
            for key, item in value.items()
        }
    if annotation is Path:
        if type(value) is not str or not value:
            raise ReadinessValidationError(f"{name} must be a path string")
        return Path(value)
    if annotation in (str, bool, int, float):
        if type(value) is not annotation:
            raise ReadinessValidationError(
                f"{name} must retain exact {annotation.__name__} type"
            )
        return value
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        try:
            return annotation(value)
        except (TypeError, ValueError) as exc:
            raise ReadinessValidationError(
                f"{name} has an unknown enum value"
            ) from exc
    if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        if type(value) is not dict:
            raise ReadinessValidationError(
                f"{name} must be a JSON object"
            )
        hints = get_type_hints(annotation)
        fields = tuple(dataclasses.fields(annotation))
        expected_keys = {field.name for field in fields}
        if set(value) != expected_keys:
            raise ReadinessValidationError(
                f"{name} has unknown or missing keys"
            )
        kwargs = {
            field.name: _decode_typed(
                hints[field.name],
                value[field.name],
                name=f"{name}.{field.name}",
            )
            for field in fields
        }
        try:
            return annotation(**kwargs)
        except (TypeError, ValueError) as exc:
            raise ReadinessValidationError(
                f"{name} failed typed reconstruction"
            ) from exc
    raise ReadinessValidationError(
        f"{name} has unsupported annotation {annotation!r}"
    )


def _read_canonical_task14_bundle(path: Path) -> Task14ReadinessBundle:
    if not isinstance(path, Path):
        raise ReadinessValidationError("Task 14 bundle path must be a Path")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ReadinessValidationError(
            "Task 14 readiness bundle is unavailable"
        ) from exc
    reparse_flag = getattr(metadata, "st_file_attributes", 0) & 0x400
    if (
        not path.is_file()
        or path.is_symlink()
        or reparse_flag
        or metadata.st_size <= 0
        or metadata.st_size > 64 * 1024 * 1024
    ):
        raise ReadinessValidationError(
            "Task 14 readiness bundle is not a bounded regular file"
        )
    try:
        raw = path.read_bytes()
        decoded = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_json_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ReadinessValidationError(
                    f"nonfinite Task 14 JSON constant: {value}"
                )
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReadinessValidationError(
            "Task 14 readiness bundle could not be decoded"
        ) from exc
    bundle = _decode_typed(
        Task14ReadinessBundle,
        decoded,
        name="task14_bundle",
    )
    if type(bundle) is not Task14ReadinessBundle:
        raise ReadinessValidationError(
            "Task 14 readiness bundle reconstruction failed"
        )
    bundle.__post_init__()
    if raw != canonical_json_bytes_generic(bundle):
        raise ReadinessValidationError(
            "Task 14 readiness bundle is not canonical JSON"
        )
    return bundle


def read_task14_readiness_bundle(path: Path) -> Task14ReadinessBundle:
    """Strictly reopen the canonical typed Task 14 input bundle."""

    return _read_canonical_task14_bundle(path)


def publish_task14_readiness_bundle(
    *,
    path: Path,
    backend: DurabilityBackend,
    training: TrainingConfig,
    source_lock: object,
    static_scientific: StaticScientificPreconditionRecord,
    training_sparsity: TrainingSparsityCertificate,
    dependency_lock: DependencyLockIdentity,
    durability: DurabilityIdentity,
    allocation_preflight: AllocationPreflightRecord,
    environment: EnvironmentRecord,
    resource_forecast: ResourceForecast,
    provenance: TrainingProvenanceRecord,
    endpoint_inventory: EndpointInventory,
    live_integration_artifact_sha256: str,
) -> Task14ReadinessBundle:
    """Exclusively publish and reopen the complete Task 14 input bundle."""

    from vfe4.training.production import ProductionSourceLock

    if type(training) is not TrainingConfig:
        raise ReadinessValidationError("training config must be exact")
    if type(source_lock) is not ProductionSourceLock:
        raise ReadinessValidationError("source lock must be exact")
    source_lock.__post_init__()
    bundle = Task14ReadinessBundle.create(
        training_config_sha256=training.experiment_config_sha256,
        source_lock_sha256=source_lock.source_lock_sha256,
        static_scientific=static_scientific,
        training_sparsity=training_sparsity,
        dependency_lock=dependency_lock,
        durability=durability,
        allocation_preflight=allocation_preflight,
        environment=environment,
        resource_forecast=resource_forecast,
        provenance=provenance,
        endpoint_inventory=endpoint_inventory,
        live_integration_artifact_sha256=live_integration_artifact_sha256,
    )
    if not callable(getattr(backend, "create_exclusive", None)):
        raise ReadinessValidationError("durability backend is invalid")
    backend.create_exclusive(path, canonical_json_bytes_generic(bundle))
    reopened = _read_canonical_task14_bundle(path)
    if reopened != bundle:
        raise ReadinessValidationError(
            "published Task 14 readiness bundle changed on reopen"
        )
    return reopened


def _sealed_task14_evidence(
    *,
    execution_identity: TrainingExecutionIdentity,
    allocation_preflight_sha256: str,
    resource_forecast_sha256: str,
    live_integration_artifact_sha256: str,
) -> Task14LiveIntegrationEvidence:
    payload = {
        "schema_version": "wt103-task14-live-integration-evidence-v1",
        "execution_identity": execution_identity,
        "allocation_preflight_sha256": allocation_preflight_sha256,
        "resource_forecast_sha256": resource_forecast_sha256,
        "live_integration_artifact_sha256": (
            live_integration_artifact_sha256
        ),
    }
    evidence = object.__new__(Task14LiveIntegrationEvidence)
    for name, value in (
        *payload.items(),
        (
            "evidence_sha256",
            owned_sha256(
                "vfe4.wt103.task14-live-integration-evidence.v1",
                payload,
            ),
        ),
    ):
        object.__setattr__(evidence, name, value)
    evidence.__post_init__()
    return evidence


def _sealed_readiness_token(
    *,
    assessment_sha256: str,
    bundle: Task14ReadinessBundle,
    source_lock: object,
    evidence: Task14LiveIntegrationEvidence,
) -> PostH8ReadinessToken:
    finalized = getattr(source_lock, "finalized_source")
    payload = {
        "schema_version": "wt103-post-h8-readiness-token-v1",
        "authority": "production_task14_live_integration",
        "assessment_sha256": assessment_sha256,
        "git_head": bundle.provenance.git_head,
        "finalized_source_record_sha256": finalized.record_sha256,
        "environment_sha256": bundle.environment.environment_sha256,
        "provenance_sha256": bundle.provenance.provenance_sha256,
        "allocation_preflight_sha256": (
            bundle.allocation_preflight.record_sha256
        ),
        "resource_forecast_sha256": bundle.resource_forecast.forecast_sha256,
        "task14_live_integration_evidence_sha256": evidence.evidence_sha256,
    }
    token = object.__new__(PostH8ReadinessToken)
    for name, value in (
        *payload.items(),
        (
            "token_sha256",
            owned_sha256(
                "vfe4.wt103.post-h8-readiness-token.v1",
                payload,
            ),
        ),
    ):
        object.__setattr__(token, name, value)
    token.__post_init__()
    return token


def reopen_and_issue_task14_readiness(
    *,
    path: Path,
    training: TrainingConfig,
    source_lock: object,
) -> tuple[PostH8ReadinessAssessment, PostH8ReadinessToken]:
    """Durably reopen Task 14 evidence and issue the sole production token."""

    from vfe4.training.production import ProductionSourceLock

    if type(training) is not TrainingConfig:
        raise ReadinessValidationError("training config must be exact")
    if type(source_lock) is not ProductionSourceLock:
        raise ReadinessValidationError("source lock must be exact")
    source_lock.__post_init__()
    bundle = _read_canonical_task14_bundle(path)
    if (
        bundle.training_config_sha256
        != training.experiment_config_sha256
        or bundle.source_lock_sha256 != source_lock.source_lock_sha256
        or bundle.endpoint_inventory != training.endpoint_inventory
        or bundle.provenance.source_record_sha256
        != source_lock.finalized_source.record_sha256
        or bundle.provenance.tokenizer_spec_sha256
        != source_lock.tokenizer.spec_sha256
        or bundle.provenance.schedule_set_sha256
        != source_lock.schedules.schedule_set_sha256
        or bundle.dependency_lock.lock_sha256
        != source_lock.finalized_source.dependency_lock_sha256
    ):
        raise ReadinessValidationError(
            "Task 14 bundle differs from config or source lock"
        )
    initial = validate_post_h8_readiness(
        static_scientific=bundle.static_scientific,
        training_sparsity=bundle.training_sparsity,
        finalized_source=source_lock.finalized_source,
        tokenizer=source_lock.tokenizer,
        token_caches=tuple(
            item.cache_identity for item in source_lock.token_caches
        ),
        dependency_lock=bundle.dependency_lock,
        durability=bundle.durability,
        allocation_preflight=bundle.allocation_preflight,
        environment=bundle.environment,
        resource_forecast=bundle.resource_forecast,
        provenance=bundle.provenance,
        endpoint_inventory=bundle.endpoint_inventory,
        h8_allocation_evidence=None,
        task14_evidence=None,
    )
    if (
        initial.status is not GateStatus.INCONCLUSIVE
        or initial.obligations
        != ("task14_live_integration_evidence_missing",)
        or initial.production_token_issued
        or initial.token is not None
    ):
        raise ReadinessValidationError(
            "non-Task14 readiness obligations remain open"
        )
    evidence = _sealed_task14_evidence(
        execution_identity=bundle.allocation_preflight.execution_identity,
        allocation_preflight_sha256=(
            bundle.allocation_preflight.record_sha256
        ),
        resource_forecast_sha256=bundle.resource_forecast.forecast_sha256,
        live_integration_artifact_sha256=(
            bundle.live_integration_artifact_sha256
        ),
    )
    assessment_payload = {
        name: getattr(initial, name)
        for name in (
            "schema_version",
            "static_scientific_record_sha256",
            "training_sparsity_sha256",
            "finalized_source_record_sha256",
            "tokenizer_spec_sha256",
            "token_cache_set_sha256",
            "dependency_lock_identity_sha256",
            "durability_identity_sha256",
            "allocation_preflight_sha256",
            "environment_sha256",
            "resource_forecast_sha256",
            "provenance_sha256",
            "endpoint_inventory_sha256",
        )
    } | {
        "task14_live_integration_evidence_sha256": evidence.evidence_sha256,
        "status": GateStatus.PASS,
        "obligations": (),
    }
    assessment_sha256 = owned_sha256(
        "vfe4.wt103.post-h8-readiness-assessment.v1",
        assessment_payload,
    )
    token = _sealed_readiness_token(
        assessment_sha256=assessment_sha256,
        bundle=bundle,
        source_lock=source_lock,
        evidence=evidence,
    )
    assessment = PostH8ReadinessAssessment(
        **assessment_payload,
        production_token_issued=True,
        token=token,
        assessment_sha256=assessment_sha256,
    )
    assessment.__post_init__()
    return assessment, token


__all__ = [
    "Task14ReadinessBundle",
    "publish_task14_readiness_bundle",
    "read_task14_readiness_bundle",
    "reopen_and_issue_task14_readiness",
]
