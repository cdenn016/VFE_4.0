"""Closed WikiText-103 checkpoint identities and resume compatibility."""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Literal

from vfe4.types.training import (
    CheckpointBundle,
    WT103CheckpointIdentity,
    owned_sha256,
)


CHECKPOINT_ENVELOPE_SCHEMA = "wt103-checkpoint-envelope-v1"
CHECKPOINT_ROLE_PURPOSES = (
    "confirmation",
    "test",
    "endpoint",
    "figure",
)
_LOWER_HEX = frozenset("0123456789abcdef")
_COMPATIBILITY_HASH_FIELDS = (
    "arm_spec_sha256",
    "experiment_plan_sha256",
    "config_sha256",
    "objective_sha256",
    "model_schema_sha256",
    "recognition_schema_sha256",
    "optimizer_schema_sha256",
    "scheduler_schema_sha256",
    "amp_schema_sha256",
    "rng_schema_sha256",
    "estimator_schema_sha256",
    "cursor_schema_sha256",
    "metric_schema_sha256",
    "update_trace_schema_sha256",
    "precision_profile_sha256",
    "dependency_lock_sha256",
    "source_sha256",
    "tokenizer_sha256",
    "data_sha256",
    "window_sha256",
    "permutation_sha256",
    "evidence_sha256",
    "environment_sha256",
)


class CheckpointError(RuntimeError):
    """Base class for a fail-closed WikiText-103 checkpoint rejection."""


class CheckpointSchemaError(CheckpointError):
    """A checkpoint or requested contract has an invalid closed schema."""


class CheckpointSecurityError(CheckpointError):
    """Checkpoint bytes or their filesystem carrier are unsafe or corrupt."""


class CheckpointCompatibilityError(CheckpointError):
    """A valid checkpoint does not match the exact requested resume contract."""


class CheckpointMigrationError(CheckpointError):
    """No explicit permitted VFE4 checkpoint migration exists."""


def _require_sha256(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise CheckpointSchemaError(f"{field} must be a lowercase 64-hex SHA-256")
    return value


def _require_text(value: object, *, field: str) -> str:
    if type(value) is not str or not value or not value.strip():
        raise CheckpointSchemaError(f"{field} must be nonempty exact text")
    return value


def identifier_is_v3(value: object) -> bool:
    if type(value) is not str:
        return False
    normalized = value.casefold().replace("\\", "/").replace("-", "_")
    components = tuple(component for component in normalized.split("/") if component)
    return any(
        component == "v3"
        or component.startswith("v3_")
        or "v3_transformer" in component
        for component in components
    )


def reject_v3_identifier(value: object, *, field: str) -> str:
    text = _require_text(value, field=field)
    if identifier_is_v3(text):
        raise CheckpointSchemaError(
            f"{field} is a V3 checkpoint identifier; V3 is permanently "
            "outside the VFE4 checkpoint schema"
        )
    return text


@dataclasses.dataclass(frozen=True, slots=True)
class ResumeContract:
    """Exact pre-attempt compatibility and resource contract for one restore."""

    schema_version: Literal["wt103-resume-contract-v1"]
    logical_key: str
    checkpoint_role: Literal["resume_only", "terminal_scoring"]
    training_complete: bool
    arm_spec_sha256: str
    experiment_plan_sha256: str
    config_sha256: str
    objective_sha256: str
    model_schema_sha256: str
    recognition_schema_sha256: str
    optimizer_schema_sha256: str
    scheduler_schema_sha256: str
    amp_schema_sha256: str
    rng_schema_sha256: str
    estimator_schema_sha256: str
    cursor_schema_sha256: str
    metric_schema_sha256: str
    update_trace_schema_sha256: str
    precision_profile_sha256: str
    dependency_lock_sha256: str
    source_sha256: str
    tokenizer_sha256: str
    data_sha256: str
    window_sha256: str
    permutation_sha256: str
    evidence_sha256: str
    environment_sha256: str
    maximum_checkpoint_bytes: int
    maximum_tensor_bytes: int
    maximum_total_tensor_bytes: int
    maximum_tensor_count: int
    maximum_container_items: int
    maximum_recursion_depth: int
    contract_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != "wt103-resume-contract-v1":
            raise CheckpointSchemaError("unsupported resume contract schema")
        reject_v3_identifier(self.logical_key, field="logical_key")
        if self.checkpoint_role not in ("resume_only", "terminal_scoring"):
            raise CheckpointSchemaError("checkpoint role is not closed")
        if type(self.training_complete) is not bool:
            raise CheckpointSchemaError("training_complete must be an exact bool")
        if (
            self.checkpoint_role == "terminal_scoring"
            and self.training_complete is not True
        ):
            raise CheckpointSchemaError(
                "terminal_scoring requires a complete planned training pass"
            )
        for field in _COMPATIBILITY_HASH_FIELDS:
            _require_sha256(getattr(self, field), field=field)
        for field in (
            "maximum_checkpoint_bytes",
            "maximum_tensor_bytes",
            "maximum_total_tensor_bytes",
            "maximum_tensor_count",
            "maximum_container_items",
            "maximum_recursion_depth",
        ):
            value = getattr(self, field)
            if type(value) is not int or value <= 0:
                raise CheckpointSchemaError(f"{field} must be a positive exact int")
        if self.maximum_tensor_bytes > self.maximum_total_tensor_bytes:
            raise CheckpointSchemaError(
                "per-tensor bytes cannot exceed total tensor bytes"
            )
        if self.maximum_total_tensor_bytes > self.maximum_checkpoint_bytes:
            raise CheckpointSchemaError(
                "total tensor bytes cannot exceed checkpoint bytes"
            )
        if self.maximum_tensor_count > self.maximum_container_items:
            raise CheckpointSchemaError(
                "tensor count cannot exceed the recursive item bound"
            )
        expected = owned_sha256(
            "vfe4.wt103.resume-contract.v1",
            self.canonical_payload(include_contract_sha256=False),
        )
        _require_sha256(self.contract_sha256, field="contract_sha256")
        if self.contract_sha256 != expected:
            raise CheckpointSchemaError(
                "contract_sha256 does not match the resume contract"
            )

    def canonical_payload(
        self,
        *,
        include_contract_sha256: bool = True,
    ) -> dict[str, object]:
        payload = {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name != "contract_sha256"
        }
        if include_contract_sha256:
            payload["contract_sha256"] = self.contract_sha256
        return payload

    @classmethod
    def create(
        cls,
        *,
        logical_key: str,
        checkpoint_role: Literal["resume_only", "terminal_scoring"],
        training_complete: bool,
        arm_spec_sha256: str,
        experiment_plan_sha256: str,
        config_sha256: str,
        objective_sha256: str,
        model_schema_sha256: str,
        recognition_schema_sha256: str,
        optimizer_schema_sha256: str,
        scheduler_schema_sha256: str,
        amp_schema_sha256: str,
        rng_schema_sha256: str,
        estimator_schema_sha256: str,
        cursor_schema_sha256: str,
        metric_schema_sha256: str,
        update_trace_schema_sha256: str,
        precision_profile_sha256: str,
        dependency_lock_sha256: str,
        source_sha256: str,
        tokenizer_sha256: str,
        data_sha256: str,
        window_sha256: str,
        permutation_sha256: str,
        evidence_sha256: str,
        environment_sha256: str,
        maximum_checkpoint_bytes: int,
        maximum_tensor_bytes: int,
        maximum_total_tensor_bytes: int,
        maximum_tensor_count: int,
        maximum_container_items: int,
        maximum_recursion_depth: int,
    ) -> "ResumeContract":
        values: dict[str, object] = {
            "schema_version": "wt103-resume-contract-v1",
            "logical_key": logical_key,
            "checkpoint_role": checkpoint_role,
            "training_complete": training_complete,
            "arm_spec_sha256": arm_spec_sha256,
            "experiment_plan_sha256": experiment_plan_sha256,
            "config_sha256": config_sha256,
            "objective_sha256": objective_sha256,
            "model_schema_sha256": model_schema_sha256,
            "recognition_schema_sha256": recognition_schema_sha256,
            "optimizer_schema_sha256": optimizer_schema_sha256,
            "scheduler_schema_sha256": scheduler_schema_sha256,
            "amp_schema_sha256": amp_schema_sha256,
            "rng_schema_sha256": rng_schema_sha256,
            "estimator_schema_sha256": estimator_schema_sha256,
            "cursor_schema_sha256": cursor_schema_sha256,
            "metric_schema_sha256": metric_schema_sha256,
            "update_trace_schema_sha256": update_trace_schema_sha256,
            "precision_profile_sha256": precision_profile_sha256,
            "dependency_lock_sha256": dependency_lock_sha256,
            "source_sha256": source_sha256,
            "tokenizer_sha256": tokenizer_sha256,
            "data_sha256": data_sha256,
            "window_sha256": window_sha256,
            "permutation_sha256": permutation_sha256,
            "evidence_sha256": evidence_sha256,
            "environment_sha256": environment_sha256,
            "maximum_checkpoint_bytes": maximum_checkpoint_bytes,
            "maximum_tensor_bytes": maximum_tensor_bytes,
            "maximum_total_tensor_bytes": maximum_total_tensor_bytes,
            "maximum_tensor_count": maximum_tensor_count,
            "maximum_container_items": maximum_container_items,
            "maximum_recursion_depth": maximum_recursion_depth,
        }
        digest = owned_sha256("vfe4.wt103.resume-contract.v1", values)
        return cls(**values, contract_sha256=digest)  # type: ignore[arg-type]


def make_checkpoint_bundle(
    *,
    logical_key: str,
    arm_spec_sha256: str,
    experiment_plan_sha256: str,
    config_sha256: str,
    scientific_state_sha256: str,
) -> CheckpointBundle:
    payload = {
        "schema_version": "wt103-checkpoint-bundle-v1",
        "logical_key": reject_v3_identifier(logical_key, field="logical_key"),
        "arm_spec_sha256": _require_sha256(
            arm_spec_sha256,
            field="arm_spec_sha256",
        ),
        "experiment_plan_sha256": _require_sha256(
            experiment_plan_sha256,
            field="experiment_plan_sha256",
        ),
        "config_sha256": _require_sha256(
            config_sha256,
            field="config_sha256",
        ),
        "scientific_state_sha256": _require_sha256(
            scientific_state_sha256,
            field="scientific_state_sha256",
        ),
    }
    return CheckpointBundle(
        **payload,
        bundle_sha256=owned_sha256(
            "vfe4.wt103.checkpoint-bundle.v1",
            payload,
        ),
    )  # type: ignore[arg-type]


def make_checkpoint_identity(
    *,
    logical_key: str,
    checkpoint_role: Literal["resume_only", "terminal_scoring"],
    scientific_state_sha256: str,
    checkpoint_payload_sha256: str,
    checkpoint_manifest_body_sha256: str,
    size_bytes: int,
) -> WT103CheckpointIdentity:
    if checkpoint_role not in ("resume_only", "terminal_scoring"):
        raise CheckpointSchemaError("checkpoint role is not closed")
    payload_sha256 = _require_sha256(
        checkpoint_payload_sha256,
        field="checkpoint_payload_sha256",
    )
    manifest_sha256 = _require_sha256(
        checkpoint_manifest_body_sha256,
        field="checkpoint_manifest_body_sha256",
    )
    artifact_sha256 = hashlib.sha256(
        b"vfe4-checkpoint-artifact-v1\x00"
        + bytes.fromhex(payload_sha256)
        + bytes.fromhex(manifest_sha256)
    ).hexdigest()
    values = {
        "schema_version": "wt103-checkpoint-identity-v1",
        "logical_key": reject_v3_identifier(
            logical_key,
            field="logical_key",
        ),
        "checkpoint_role": checkpoint_role,
        "scientific_state_sha256": _require_sha256(
            scientific_state_sha256,
            field="scientific_state_sha256",
        ),
        "checkpoint_payload_sha256": payload_sha256,
        "checkpoint_manifest_body_sha256": manifest_sha256,
        "artifact_sha256": artifact_sha256,
        "size_bytes": size_bytes,
    }
    if type(size_bytes) is not int or size_bytes <= 0:
        raise CheckpointSchemaError("size_bytes must be a positive exact int")
    return WT103CheckpointIdentity(
        **values,
        checkpoint_identity_sha256=owned_sha256(
            "vfe4.wt103.checkpoint-identity.v1",
            values,
        ),
    )  # type: ignore[arg-type]


def require_terminal_scoring(
    identity: WT103CheckpointIdentity,
    *,
    purpose: str,
) -> None:
    if type(identity) is not WT103CheckpointIdentity:
        raise CheckpointSchemaError(
            "checkpoint identity must be exact WT103CheckpointIdentity"
        )
    try:
        identity.__post_init__()
    except ValueError as exc:
        raise CheckpointSchemaError("checkpoint identity is invalid") from exc
    if purpose not in CHECKPOINT_ROLE_PURPOSES:
        raise CheckpointSchemaError("checkpoint role purpose is not closed")
    if identity.checkpoint_role != "terminal_scoring":
        raise CheckpointCompatibilityError(
            f"{purpose} requires a terminal_scoring checkpoint"
        )


__all__ = [
    "CHECKPOINT_ENVELOPE_SCHEMA",
    "CHECKPOINT_ROLE_PURPOSES",
    "CheckpointCompatibilityError",
    "CheckpointError",
    "CheckpointMigrationError",
    "CheckpointSchemaError",
    "CheckpointSecurityError",
    "ResumeContract",
    "identifier_is_v3",
    "make_checkpoint_bundle",
    "make_checkpoint_identity",
    "reject_v3_identifier",
    "require_terminal_scoring",
]
