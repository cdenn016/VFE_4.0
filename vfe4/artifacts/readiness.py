"""Final one-way composition of Task 6 science with Task 10 operations."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

from vfe4.training.readiness import StaticScientificPreconditionRecord
from vfe4.types.results import GateStatus
from vfe4.types.training import (
    EndpointInventory,
    FinalizedWikiText103SourceRecord,
    ProductionTokenCacheIdentity,
    ProductionTokenizerSpec,
    TrainingSparsityCertificate,
    owned_sha256,
)

from .durability import DurabilityIdentity, canonical_json_bytes_generic
from .environment import (
    AllocationPreflightRecord,
    DependencyLockIdentity,
    EnvironmentRecord,
    ResourceForecast,
    TrainingExecutionIdentity,
)
from .provenance import (
    TrainingProvenanceRecord,
    production_token_cache_set_sha256,
)


class ReadinessValidationError(RuntimeError):
    """Task 10 readiness inputs or issuance authority were invalid."""


def _sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ReadinessValidationError(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True, init=False)
class Task14LiveIntegrationEvidence:
    """Sealed typed evidence emitted only after Task 14 live integration."""

    schema_version: Literal["wt103-task14-live-integration-evidence-v1"]
    execution_identity: TrainingExecutionIdentity
    allocation_preflight_sha256: str
    resource_forecast_sha256: str
    live_integration_artifact_sha256: str
    evidence_sha256: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise ReadinessValidationError(
            "Task 14 live-integration evidence is unavailable before the "
            "durable Task 14 issuer exists"
        )

    def semantic_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in tuple(self.__dataclass_fields__)[:-1]
        }

    def __post_init__(self) -> None:
        if (
            self.schema_version
            != "wt103-task14-live-integration-evidence-v1"
            or type(self.execution_identity) is not TrainingExecutionIdentity
        ):
            raise ReadinessValidationError(
                "Task 14 live-integration evidence schema is invalid"
            )
        try:
            self.execution_identity.__post_init__()
        except ValueError as exc:
            raise ReadinessValidationError(
                "Task 14 execution identity is invalid"
            ) from exc
        for name in (
            "allocation_preflight_sha256",
            "resource_forecast_sha256",
            "live_integration_artifact_sha256",
        ):
            _sha256(getattr(self, name), name)
        expected = owned_sha256(
            "vfe4.wt103.task14-live-integration-evidence.v1",
            self.semantic_payload(),
        )
        _sha256(self.evidence_sha256, "evidence_sha256")
        if self.evidence_sha256 != expected:
            raise ReadinessValidationError(
                "Task 14 live-integration evidence hash does not match"
            )


def _validate_durability_identity(value: DurabilityIdentity) -> None:
    body = {
        name: getattr(value, name)
        for name in value.__dataclass_fields__
        if name != "identity_sha256"
    }
    expected = hashlib.sha256(
        b"vfe4.durability-identity.v1\0"
        + canonical_json_bytes_generic(body)
    ).hexdigest()
    if value.identity_sha256 != expected:
        raise ReadinessValidationError(
            "durability identity does not match its semantic payload"
        )


@dataclass(frozen=True, slots=True, init=False)
class PostH8ReadinessToken:
    """Sealed production authority issued only by the Task 14 call path."""

    schema_version: Literal["wt103-post-h8-readiness-token-v1"]
    authority: Literal["production_task14_live_integration"]
    assessment_sha256: str
    git_head: str
    finalized_source_record_sha256: str
    environment_sha256: str
    provenance_sha256: str
    allocation_preflight_sha256: str
    resource_forecast_sha256: str
    task14_live_integration_evidence_sha256: str
    token_sha256: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise ReadinessValidationError(
            "Task 14 production token issuance is unavailable before the "
            "durable Task 14 issuer exists"
        )

    def semantic_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in tuple(self.__dataclass_fields__)[:-1]
        }

    def __post_init__(self) -> None:
        if (
            self.schema_version != "wt103-post-h8-readiness-token-v1"
            or self.authority != "production_task14_live_integration"
            or type(self.git_head) is not str
            or len(self.git_head) not in (40, 64)
            or any(
                character not in "0123456789abcdef"
                for character in self.git_head
            )
        ):
            raise ReadinessValidationError("readiness token schema is invalid")
        for name in (
            "assessment_sha256",
            "finalized_source_record_sha256",
            "environment_sha256",
            "provenance_sha256",
            "allocation_preflight_sha256",
            "resource_forecast_sha256",
            "task14_live_integration_evidence_sha256",
        ):
            _sha256(getattr(self, name), name)
        expected = owned_sha256(
            "vfe4.wt103.post-h8-readiness-token.v1",
            self.semantic_payload(),
        )
        _sha256(self.token_sha256, "token_sha256")
        if self.token_sha256 != expected:
            raise ReadinessValidationError(
                "readiness token hash does not match"
            )


@dataclass(frozen=True, slots=True)
class PostH8ReadinessAssessment:
    """Complete validation result; issuance remains a separate gated fact."""

    schema_version: Literal["wt103-post-h8-readiness-assessment-v1"]
    static_scientific_record_sha256: str
    training_sparsity_sha256: str
    finalized_source_record_sha256: str
    tokenizer_spec_sha256: str
    token_cache_set_sha256: str
    dependency_lock_identity_sha256: str
    durability_identity_sha256: str
    allocation_preflight_sha256: str
    environment_sha256: str
    resource_forecast_sha256: str
    provenance_sha256: str
    endpoint_inventory_sha256: str
    task14_live_integration_evidence_sha256: str | None
    status: GateStatus
    obligations: tuple[str, ...]
    production_token_issued: bool
    token: PostH8ReadinessToken | None
    assessment_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in tuple(self.__dataclass_fields__)[:-3]
        } | {
            "status": self.status,
            "obligations": self.obligations,
        }

    def __post_init__(self) -> None:
        if (
            self.schema_version
            != "wt103-post-h8-readiness-assessment-v1"
            or type(self.status) is not GateStatus
            or type(self.obligations) is not tuple
            or any(type(item) is not str or not item for item in self.obligations)
            or type(self.production_token_issued) is not bool
        ):
            raise ReadinessValidationError(
                "readiness assessment schema is invalid"
            )
        for name in (
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
        ):
            _sha256(getattr(self, name), name)
        if self.task14_live_integration_evidence_sha256 is not None:
            _sha256(
                self.task14_live_integration_evidence_sha256,
                "task14_live_integration_evidence_sha256",
            )
        if (
            (self.status is GateStatus.PASS and self.obligations)
            or (self.status is not GateStatus.PASS and not self.obligations)
        ):
            raise ReadinessValidationError(
                "readiness status/obligations disagree"
            )
        if self.production_token_issued is not (
            self.task14_live_integration_evidence_sha256 is not None
        ):
            raise ReadinessValidationError(
                "Task 14 evidence/token issuance state disagrees"
            )
        if self.production_token_issued:
            if (
                type(self.token) is not PostH8ReadinessToken
                or self.status is not GateStatus.PASS
                or self.token.assessment_sha256 != self.assessment_sha256
                or self.token.task14_live_integration_evidence_sha256
                != self.task14_live_integration_evidence_sha256
            ):
                raise ReadinessValidationError(
                    "issued token does not bind this PASS assessment"
                )
            self.token.__post_init__()
        elif self.token is not None:
            raise ReadinessValidationError(
                "unissued assessment cannot contain a token"
            )
        expected = owned_sha256(
            "vfe4.wt103.post-h8-readiness-assessment.v1",
            self.semantic_payload(),
        )
        _sha256(self.assessment_sha256, "assessment_sha256")
        if self.assessment_sha256 != expected:
            raise ReadinessValidationError(
                "readiness assessment hash does not match"
            )


def validate_post_h8_readiness(
    *,
    static_scientific: StaticScientificPreconditionRecord,
    training_sparsity: TrainingSparsityCertificate,
    finalized_source: FinalizedWikiText103SourceRecord,
    tokenizer: ProductionTokenizerSpec,
    token_caches: tuple[ProductionTokenCacheIdentity, ...],
    dependency_lock: DependencyLockIdentity,
    durability: DurabilityIdentity,
    allocation_preflight: AllocationPreflightRecord,
    environment: EnvironmentRecord,
    resource_forecast: ResourceForecast,
    provenance: TrainingProvenanceRecord,
    endpoint_inventory: EndpointInventory,
    h8_allocation_evidence: None,
    task14_evidence: Task14LiveIntegrationEvidence | None,
) -> PostH8ReadinessAssessment:
    """Compose exact PASS records in the sole Task 6 -> Task 10 direction."""

    if h8_allocation_evidence is not None:
        raise ReadinessValidationError(
            "H8 allocation evidence cannot populate training readiness"
        )
    if type(tokenizer) is not ProductionTokenizerSpec:
        raise ReadinessValidationError(
            "final readiness requires an exact production tokenizer"
        )
    if (
        type(token_caches) is not tuple
        or any(
            type(item) is not ProductionTokenCacheIdentity
            for item in token_caches
        )
    ):
        raise ReadinessValidationError(
            "final readiness requires exact production token caches"
        )
    exact_inputs = (
        (static_scientific, StaticScientificPreconditionRecord),
        (training_sparsity, TrainingSparsityCertificate),
        (finalized_source, FinalizedWikiText103SourceRecord),
        (dependency_lock, DependencyLockIdentity),
        (durability, DurabilityIdentity),
        (allocation_preflight, AllocationPreflightRecord),
        (environment, EnvironmentRecord),
        (resource_forecast, ResourceForecast),
        (provenance, TrainingProvenanceRecord),
        (endpoint_inventory, EndpointInventory),
    )
    if any(type(value) is not expected for value, expected in exact_inputs):
        raise ReadinessValidationError(
            "final readiness requires exact typed records"
        )
    try:
        for value, _ in exact_inputs:
            post_init = getattr(value, "__post_init__", None)
            if post_init is not None:
                post_init()
        _validate_durability_identity(durability)
        tokenizer.__post_init__()
        cache_set_sha = production_token_cache_set_sha256(token_caches)
    except (TypeError, ValueError) as exc:
        raise ReadinessValidationError(
            "final readiness input validation failed"
        ) from exc
    if task14_evidence is not None:
        raise ReadinessValidationError(
            "Task 10 cannot consume or issue Task 14 production authority; "
            "the later issuer must durably reopen canonical Task 14 evidence"
        )

    failures: list[str] = []
    inconclusive: list[str] = [
        "task14_live_integration_evidence_missing"
    ]
    try:
        expected_execution_identity = TrainingExecutionIdentity.create(
            git_identity_sha256=provenance.git_identity_sha256,
            git_head=provenance.git_head,
            dirty_digest=provenance.dirty_digest,
            config_sha256=provenance.config_sha256,
            profile_sha256=static_scientific.profile_sha256,
            factory_set_sha256=static_scientific.factory_set_sha256,
            environment_sha256=environment.environment_sha256,
        )
    except ValueError as exc:
        raise ReadinessValidationError(
            "readiness execution identity could not be derived"
        ) from exc
    if static_scientific.status is not GateStatus.PASS:
        inconclusive.append("static_scientific_preconditions_not_pass")
    if (
        training_sparsity.status is not GateStatus.PASS
        or static_scientific.training_sparsity_sha256
        != training_sparsity.certificate_sha256
        or static_scientific.profile_sha256
        != training_sparsity.profile_sha256
        or static_scientific.factory_set_sha256
        != training_sparsity.factory_set_sha256
        or static_scientific.endpoint_inventory_sha256
        != training_sparsity.endpoint_inventory_sha256
    ):
        inconclusive.append("training_sparsity_not_exact_pass")
    if (
        static_scientific.git_head != training_sparsity.git_head
        or static_scientific.dirty_digest
        != training_sparsity.dirty_digest
    ):
        inconclusive.append("scientific_revision_mismatch")
    if (
        static_scientific.endpoint_inventory_sha256
        != endpoint_inventory.endpoint_inventory_sha256
    ):
        inconclusive.append("static_inventory_mismatch")
    if (
        finalized_source.production_tokenizer_spec_sha256
        != tokenizer.spec_sha256
        or finalized_source.tokenizer_tables_sha256
        != tokenizer.tokenizer_tables_sha256
    ):
        inconclusive.append("source_tokenizer_mismatch")
    if (
        finalized_source.production_token_cache_set_sha256
        != cache_set_sha
    ):
        inconclusive.append("source_token_cache_set_mismatch")
    if (
        finalized_source.dependency_lock_sha256
        != dependency_lock.lock_sha256
    ):
        inconclusive.append("source_dependency_lock_mismatch")
    if dependency_lock.status is not GateStatus.PASS:
        inconclusive.append("dependency_lock_not_exact_match")
    if durability.status != "pass":
        inconclusive.append("durability_probe_not_pass")
    if (
        allocation_preflight.status is GateStatus.FAIL
        or allocation_preflight.endpoint_inventory_sha256
        != endpoint_inventory.endpoint_inventory_sha256
    ):
        failures.append("allocation_preflight_not_pass")
    elif allocation_preflight.status is not GateStatus.PASS:
        inconclusive.append("allocation_preflight_not_pass")
    if (
        allocation_preflight.execution_identity
        != expected_execution_identity
        or allocation_preflight.environment_sha256
        != environment.environment_sha256
    ):
        inconclusive.append("allocation_execution_identity_mismatch")
    if (
        environment.captured_before_device_work is not True
        or environment.dependency_lock_identity_sha256
        != dependency_lock.identity_sha256
    ):
        inconclusive.append("environment_not_exact_pre_device_capture")
    if resource_forecast.status is GateStatus.FAIL:
        failures.append("resource_forecast_not_pass")
    elif resource_forecast.status is not GateStatus.PASS:
        inconclusive.append("resource_forecast_not_pass")
    if (
        resource_forecast.endpoint_inventory_sha256
        != endpoint_inventory.endpoint_inventory_sha256
    ):
        inconclusive.append("resource_inventory_mismatch")
    if resource_forecast.execution_identity != expected_execution_identity:
        inconclusive.append("resource_execution_identity_mismatch")
    if (
        not provenance.source_is_clean
        or provenance.git_head != static_scientific.git_head
        or provenance.dirty_digest != static_scientific.dirty_digest
    ):
        inconclusive.append("clean_same_revision_provenance_missing")
    if (
        provenance.environment_sha256 != environment.environment_sha256
        or provenance.dependency_lock_identity_sha256
        != dependency_lock.identity_sha256
    ):
        inconclusive.append("runtime_provenance_mismatch")
    if (
        provenance.source_record_sha256 != finalized_source.record_sha256
        or provenance.tokenizer_spec_sha256 != tokenizer.spec_sha256
        or provenance.token_cache_set_sha256 != cache_set_sha
        or provenance.schedule_set_sha256
        != finalized_source.schedule_set_sha256
        or provenance.factory_set_sha256
        != static_scientific.factory_set_sha256
        or provenance.objective_sha256
        != static_scientific.objective_sha256
        or provenance.endpoint_inventory_sha256
        != endpoint_inventory.endpoint_inventory_sha256
    ):
        inconclusive.append("scientific_provenance_identity_mismatch")
    obligations = tuple(dict.fromkeys((*failures, *inconclusive)))
    status = (
        GateStatus.FAIL
        if failures
        else GateStatus.INCONCLUSIVE
        if inconclusive
        else GateStatus.PASS
    )
    identity_payload = {
        "schema_version": "wt103-post-h8-readiness-assessment-v1",
        "static_scientific_record_sha256": (
            static_scientific.record_sha256
        ),
        "training_sparsity_sha256": (
            training_sparsity.certificate_sha256
        ),
        "finalized_source_record_sha256": finalized_source.record_sha256,
        "tokenizer_spec_sha256": tokenizer.spec_sha256,
        "token_cache_set_sha256": cache_set_sha,
        "dependency_lock_identity_sha256": dependency_lock.identity_sha256,
        "durability_identity_sha256": durability.identity_sha256,
        "allocation_preflight_sha256": (
            allocation_preflight.record_sha256
        ),
        "environment_sha256": environment.environment_sha256,
        "resource_forecast_sha256": resource_forecast.forecast_sha256,
        "provenance_sha256": provenance.provenance_sha256,
        "endpoint_inventory_sha256": (
            endpoint_inventory.endpoint_inventory_sha256
        ),
        "task14_live_integration_evidence_sha256": None,
        "status": status,
        "obligations": obligations,
    }
    assessment_sha = owned_sha256(
        "vfe4.wt103.post-h8-readiness-assessment.v1",
        identity_payload,
    )
    token = None
    return PostH8ReadinessAssessment(
        **identity_payload,
        production_token_issued=token is not None,
        token=token,
        assessment_sha256=assessment_sha,
    )


__all__ = [
    "PostH8ReadinessAssessment",
    "PostH8ReadinessToken",
    "ReadinessValidationError",
    "Task14LiveIntegrationEvidence",
    "validate_post_h8_readiness",
]
