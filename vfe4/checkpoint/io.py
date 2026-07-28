"""Durability-backed, bounded, weights-only WikiText-103 checkpoint I/O."""

from __future__ import annotations

import hashlib
import math
import os
import stat
from pathlib import Path
from typing import Protocol, runtime_checkable

from vfe4.artifacts.durability import (
    DurableFileIdentity,
    DurabilityBackend,
    canonical_json_bytes_generic,
)
from vfe4.types.training import (
    CheckpointBundle,
    LoadedCheckpoint,
    WT103CheckpointIdentity,
)

from .migrations import select_migration
from .schema import (
    CHECKPOINT_ENVELOPE_SCHEMA,
    CheckpointCompatibilityError,
    CheckpointError,
    CheckpointSchemaError,
    CheckpointSecurityError,
    ResumeContract,
    identifier_is_v3,
    make_checkpoint_bundle,
    make_checkpoint_identity,
)
from .serialization import (
    TensorInventoryEntry,
    deserialize_weights_only_cpu,
    inventory_from_payload,
    inventory_payload,
    normalize_scientific_state,
    scientific_state_sha256,
    serialize_checkpoint_envelope,
    validate_whitelist_tree,
)


_ENVELOPE_KEYS = frozenset(
    {
        "schema_version",
        "resume_contract",
        "bundle",
        "manifest_body",
        "tensor_inventory",
        "scientific_state",
    }
)
_BUNDLE_KEYS = frozenset(
    {
        "schema_version",
        "logical_key",
        "arm_spec_sha256",
        "experiment_plan_sha256",
        "config_sha256",
        "scientific_state_sha256",
        "bundle_sha256",
    }
)
_MANIFEST_BODY_KEYS = frozenset(
    {
        "schema_version",
        "checkpoint_schema",
        "logical_key",
        "checkpoint_role",
        "tensor_count",
        "total_tensor_bytes",
        "operational_metadata",
    }
)
_OPERATIONAL_METADATA_KEYS = frozenset(
    {
        "process_id",
        "utc_timestamp",
        "monotonic_seconds",
        "elapsed_seconds",
        "path_hint",
        "write_ordinal",
    }
)
_MANIFEST_BODY_DOMAIN = b"vfe4.wt103.checkpoint-manifest-body.v1\x00"
_READ_CHUNK_BYTES = 1024 * 1024


@runtime_checkable
class FreshCheckpointTarget(Protocol):
    """Training-owned fresh object boundary used after full validation."""

    checkpoint_contract_sha256: str

    def is_fresh_checkpoint_target(self) -> bool: ...

    def validate_checkpoint_state(
        self,
        state: dict[str, object],
    ) -> None: ...

    def restore_checkpoint_state(
        self,
        state: dict[str, object],
    ) -> None: ...


def _is_redirect_or_reparse(path: Path, status: os.stat_result) -> bool:
    if stat.S_ISLNK(status.st_mode):
        return True
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if getattr(status, "st_file_attributes", 0) & reparse:
        return True
    is_junction = getattr(path, "is_junction", None)
    try:
        return bool(is_junction is not None and is_junction())
    except OSError as exc:
        raise CheckpointSecurityError(
            f"checkpoint reparse/junction metadata failed: {exc}"
        ) from exc


def _reject_v3_path(path: Path) -> None:
    if any(identifier_is_v3(part) for part in path.parts):
        raise CheckpointSecurityError("V3 checkpoint paths are permanently rejected")


def _checkpoint_path(path: Path, *, require_file: bool) -> Path:
    if not isinstance(path, Path) or path.name in ("", ".", ".."):
        raise CheckpointSchemaError(
            "checkpoint path must be a concrete pathlib.Path file"
        )
    declared = path.absolute()
    _reject_v3_path(declared)
    try:
        parent_status = declared.parent.lstat()
    except OSError as exc:
        raise CheckpointSecurityError(
            f"checkpoint parent is unavailable: {exc}"
        ) from exc
    if not stat.S_ISDIR(parent_status.st_mode) or _is_redirect_or_reparse(
        declared.parent, parent_status
    ):
        raise CheckpointSecurityError(
            "checkpoint parent must be a regular nonlink directory"
        )
    try:
        resolved_parent = declared.parent.resolve(strict=True)
    except OSError as exc:
        raise CheckpointSecurityError(
            f"checkpoint parent identity cannot be resolved: {exc}"
        ) from exc
    if resolved_parent != declared.parent:
        raise CheckpointSecurityError(
            "checkpoint parent resolved identity differs from its declared path"
        )
    _reject_v3_path(resolved_parent)
    try:
        target_status = declared.lstat()
    except FileNotFoundError:
        if require_file:
            raise CheckpointSecurityError("checkpoint file is missing")
    except OSError as exc:
        raise CheckpointSecurityError(
            f"checkpoint metadata is unavailable: {exc}"
        ) from exc
    else:
        if not stat.S_ISREG(target_status.st_mode) or _is_redirect_or_reparse(
            declared, target_status
        ):
            raise CheckpointSecurityError(
                "checkpoint must be a regular nonlink, non-reparse, non-junction file"
            )
        try:
            resolved = declared.resolve(strict=True)
        except OSError as exc:
            raise CheckpointSecurityError(
                f"checkpoint identity cannot be resolved: {exc}"
            ) from exc
        if resolved != declared:
            raise CheckpointSecurityError(
                "checkpoint resolved identity differs from its declared path"
            )
        _reject_v3_path(resolved)
    return declared


def _same_file_identity(
    left: os.stat_result,
    right: os.stat_result,
) -> bool:
    return (
        left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
    )


def _read_authenticated_payload(
    path: Path,
    *,
    expected_identity: WT103CheckpointIdentity,
    maximum_checkpoint_bytes: int,
) -> bytes:
    declared = _checkpoint_path(path, require_file=True)
    if expected_identity.size_bytes > maximum_checkpoint_bytes:
        raise CheckpointSecurityError(
            "checkpoint declared size exceeds the immutable maximum bound"
        )
    try:
        before = declared.lstat()
    except OSError as exc:
        raise CheckpointSecurityError(
            f"checkpoint metadata cannot be reopened: {exc}"
        ) from exc
    if before.st_size != expected_identity.size_bytes:
        raise CheckpointSecurityError(
            "checkpoint actual size differs from the declared size"
        )
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(declared, flags)
        opened_before = os.fstat(descriptor)
        if not _same_file_identity(before, opened_before):
            raise CheckpointSecurityError(
                "checkpoint identity changed before bounded read"
            )
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, _READ_CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if total > maximum_checkpoint_bytes:
                raise CheckpointSecurityError(
                    "checkpoint read exceeded the immutable maximum bound"
                )
            chunks.append(chunk)
        opened_after = os.fstat(descriptor)
    except CheckpointError:
        raise
    except OSError as exc:
        raise CheckpointSecurityError(f"checkpoint bounded read failed: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    try:
        after = declared.lstat()
    except OSError as exc:
        raise CheckpointSecurityError(
            f"checkpoint metadata changed after read: {exc}"
        ) from exc
    if not _same_file_identity(before, opened_after) or not _same_file_identity(
        before, after
    ):
        raise CheckpointSecurityError("checkpoint identity changed during bounded read")
    payload = b"".join(chunks)
    if len(payload) != expected_identity.size_bytes:
        raise CheckpointSecurityError(
            "checkpoint bounded read size differs from its identity"
        )
    if (
        hashlib.sha256(payload).hexdigest()
        != expected_identity.checkpoint_payload_sha256
    ):
        raise CheckpointSecurityError(
            "checkpoint payload SHA-256 differs from its identity"
        )
    return payload


def _operational_metadata(value: object) -> dict[str, object]:
    if value is None:
        return {}
    if type(value) is not dict:
        raise CheckpointSchemaError("operational_metadata must be an exact dict")
    for key in value:
        if type(key) is str and (
            "run_manifest" in key.casefold().replace("-", "_")
            or "terminal_manifest" in key.casefold().replace("-", "_")
        ):
            raise CheckpointSchemaError(
                "checkpoint manifest cannot depend on a later run manifest; "
                "that dependency would be circular"
            )
    if not set(value).issubset(_OPERATIONAL_METADATA_KEYS):
        raise CheckpointSchemaError("operational_metadata contains an unknown field")
    result: dict[str, object] = {}
    for key, item in value.items():
        if key in ("process_id", "write_ordinal"):
            if type(item) is not int or item < 0:
                raise CheckpointSchemaError(
                    f"operational {key} must be a nonnegative exact int"
                )
        elif key in ("monotonic_seconds", "elapsed_seconds"):
            if type(item) is not float or not math.isfinite(item) or item < 0.0:
                raise CheckpointSchemaError(
                    f"operational {key} must be a finite nonnegative float"
                )
        elif type(item) is not str or not item:
            raise CheckpointSchemaError(
                f"operational {key} must be nonempty exact text"
            )
        result[key] = item
    return result


def _manifest_body(
    *,
    contract: ResumeContract,
    inventory: tuple[TensorInventoryEntry, ...],
    total_tensor_bytes: int,
    operational_metadata: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "wt103-checkpoint-manifest-body-v1",
        "checkpoint_schema": CHECKPOINT_ENVELOPE_SCHEMA,
        "logical_key": contract.logical_key,
        "checkpoint_role": contract.checkpoint_role,
        "tensor_count": len(inventory),
        "total_tensor_bytes": total_tensor_bytes,
        "operational_metadata": operational_metadata,
    }


def _manifest_body_sha256(body: dict[str, object]) -> str:
    return hashlib.sha256(
        _MANIFEST_BODY_DOMAIN + canonical_json_bytes_generic(body)
    ).hexdigest()


def _bundle_payload(bundle: CheckpointBundle) -> dict[str, object]:
    return {
        "schema_version": bundle.schema_version,
        "logical_key": bundle.logical_key,
        "arm_spec_sha256": bundle.arm_spec_sha256,
        "experiment_plan_sha256": bundle.experiment_plan_sha256,
        "config_sha256": bundle.config_sha256,
        "scientific_state_sha256": bundle.scientific_state_sha256,
        "bundle_sha256": bundle.bundle_sha256,
    }


def _bundle_from_payload(value: object) -> CheckpointBundle:
    if type(value) is not dict or set(value) != _BUNDLE_KEYS:
        raise CheckpointSecurityError("checkpoint bundle payload is not exact")
    try:
        return CheckpointBundle(**value)
    except (TypeError, ValueError) as exc:
        raise CheckpointSecurityError("checkpoint bundle identity is invalid") from exc


def _validate_manifest_body(
    value: object,
    *,
    contract: ResumeContract,
    identity: WT103CheckpointIdentity,
    inventory: tuple[TensorInventoryEntry, ...],
    total_tensor_bytes: int,
) -> None:
    if type(value) is not dict or set(value) != _MANIFEST_BODY_KEYS:
        raise CheckpointSecurityError("checkpoint manifest body is not exact")
    if (
        value["schema_version"] != "wt103-checkpoint-manifest-body-v1"
        or value["checkpoint_schema"] != CHECKPOINT_ENVELOPE_SCHEMA
        or value["logical_key"] != contract.logical_key
        or value["checkpoint_role"] != contract.checkpoint_role
        or value["tensor_count"] != len(inventory)
        or value["total_tensor_bytes"] != total_tensor_bytes
    ):
        raise CheckpointCompatibilityError(
            "checkpoint manifest body has a compatibility mismatch"
        )
    metadata = _operational_metadata(value["operational_metadata"])
    if metadata != value["operational_metadata"]:
        raise CheckpointSecurityError("checkpoint operational metadata is not exact")
    if _manifest_body_sha256(value) != (identity.checkpoint_manifest_body_sha256):
        raise CheckpointSecurityError(
            "checkpoint manifest body SHA-256 differs from its identity"
        )


def _decode_validated_envelope(
    payload: bytes,
    *,
    expected_contract: ResumeContract,
    expected_identity: WT103CheckpointIdentity,
) -> tuple[LoadedCheckpoint, dict[str, object]]:
    envelope = deserialize_weights_only_cpu(payload)
    try:
        envelope = validate_whitelist_tree(
            envelope,
            contract=expected_contract,
            require_cpu=True,
        )
    except CheckpointSecurityError:
        raise
    except CheckpointSchemaError as exc:
        raise CheckpointSecurityError(
            "loaded checkpoint violates the exact primitive/tensor whitelist"
        ) from exc
    if type(envelope) is not dict or set(envelope) != _ENVELOPE_KEYS:
        raise CheckpointSecurityError("checkpoint envelope keys are not exact")
    source_schema = envelope["schema_version"]
    if type(source_schema) is not str:
        raise CheckpointSecurityError("checkpoint envelope schema is not exact text")
    select_migration(
        source_schema=source_schema,
        destination_schema=CHECKPOINT_ENVELOPE_SCHEMA,
    )
    if envelope["resume_contract"] != (expected_contract.canonical_payload()):
        raise CheckpointCompatibilityError("checkpoint compatibility contract mismatch")
    bundle = _bundle_from_payload(envelope["bundle"])
    if (
        bundle.logical_key != expected_contract.logical_key
        or bundle.arm_spec_sha256 != expected_contract.arm_spec_sha256
        or bundle.experiment_plan_sha256 != expected_contract.experiment_plan_sha256
        or bundle.config_sha256 != expected_contract.config_sha256
    ):
        raise CheckpointCompatibilityError("checkpoint bundle compatibility mismatch")
    try:
        state, observed_inventory, total_tensor_bytes = normalize_scientific_state(
            envelope["scientific_state"],
            contract=expected_contract,
            require_cpu=True,
            clone_tensors=False,
        )
    except CheckpointSecurityError:
        raise
    except CheckpointSchemaError as exc:
        raise CheckpointSecurityError(
            "loaded scientific state violates its closed schema"
        ) from exc
    declared_inventory = inventory_from_payload(envelope["tensor_inventory"])
    if declared_inventory != observed_inventory:
        raise CheckpointSecurityError(
            "checkpoint tensor inventory differs from loaded tensors"
        )
    observed_scientific_sha256 = scientific_state_sha256(state)
    if (
        bundle.scientific_state_sha256 != observed_scientific_sha256
        or expected_identity.scientific_state_sha256 != observed_scientific_sha256
    ):
        raise CheckpointSecurityError(
            "checkpoint scientific state SHA-256 differs from its identity"
        )
    _validate_manifest_body(
        envelope["manifest_body"],
        contract=expected_contract,
        identity=expected_identity,
        inventory=observed_inventory,
        total_tensor_bytes=total_tensor_bytes,
    )
    if (
        expected_identity.logical_key != expected_contract.logical_key
        or expected_identity.checkpoint_role != expected_contract.checkpoint_role
    ):
        raise CheckpointCompatibilityError("checkpoint identity compatibility mismatch")
    try:
        loaded = LoadedCheckpoint(bundle=bundle, identity=expected_identity)
    except ValueError as exc:
        raise CheckpointSecurityError(
            "checkpoint loaded identity and bundle disagree"
        ) from exc
    return loaded, state


def _validate_contract_and_identity(
    *,
    contract: ResumeContract,
    identity: WT103CheckpointIdentity | None = None,
) -> None:
    if type(contract) is not ResumeContract:
        raise CheckpointSchemaError("checkpoint requires an exact ResumeContract")
    contract.__post_init__()
    if identity is None:
        return
    if type(identity) is not WT103CheckpointIdentity:
        raise CheckpointSchemaError(
            "expected_identity must be exact WT103CheckpointIdentity"
        )
    try:
        identity.__post_init__()
    except ValueError as exc:
        raise CheckpointSchemaError("expected checkpoint identity is invalid") from exc
    if (
        identity.logical_key != contract.logical_key
        or identity.checkpoint_role != contract.checkpoint_role
    ):
        raise CheckpointCompatibilityError(
            "checkpoint identity and resume contract mismatch"
        )


def _validate_fresh_target(
    target: FreshCheckpointTarget,
    *,
    contract: ResumeContract,
) -> None:
    if not isinstance(target, FreshCheckpointTarget):
        raise CheckpointCompatibilityError(
            "fresh checkpoint target does not expose the closed restore protocol"
        )
    if target.checkpoint_contract_sha256 != contract.contract_sha256:
        raise CheckpointCompatibilityError("fresh checkpoint target contract mismatch")
    try:
        fresh = target.is_fresh_checkpoint_target()
    except Exception as exc:
        raise CheckpointCompatibilityError(
            "fresh checkpoint target could not prove freshness"
        ) from exc
    if fresh is not True:
        raise CheckpointCompatibilityError(
            "checkpoint restoration requires a fresh object"
        )


def save_checkpoint(
    path: Path,
    *,
    contract: ResumeContract,
    scientific_state: dict[str, object],
    durability_backend: DurabilityBackend,
    operational_metadata: dict[str, object] | None = None,
) -> WT103CheckpointIdentity:
    """Validate, serialize, durably publish, and reopen one checkpoint."""

    _validate_contract_and_identity(contract=contract)
    destination = _checkpoint_path(path, require_file=False)
    if not isinstance(durability_backend, DurabilityBackend):
        raise CheckpointSchemaError(
            "durability_backend must implement Task 2 durability primitives"
        )
    state, inventory, total_tensor_bytes = normalize_scientific_state(
        scientific_state,
        contract=contract,
        require_cpu=False,
        clone_tensors=True,
    )
    state_sha256 = scientific_state_sha256(state)
    bundle = make_checkpoint_bundle(
        logical_key=contract.logical_key,
        arm_spec_sha256=contract.arm_spec_sha256,
        experiment_plan_sha256=contract.experiment_plan_sha256,
        config_sha256=contract.config_sha256,
        scientific_state_sha256=state_sha256,
    )
    metadata = _operational_metadata(operational_metadata)
    manifest_body = _manifest_body(
        contract=contract,
        inventory=inventory,
        total_tensor_bytes=total_tensor_bytes,
        operational_metadata=metadata,
    )
    envelope = {
        "schema_version": CHECKPOINT_ENVELOPE_SCHEMA,
        "resume_contract": contract.canonical_payload(),
        "bundle": _bundle_payload(bundle),
        "manifest_body": manifest_body,
        "tensor_inventory": inventory_payload(inventory),
        "scientific_state": state,
    }
    validate_whitelist_tree(
        envelope,
        contract=contract,
        require_cpu=True,
    )
    payload = serialize_checkpoint_envelope(envelope)
    if len(payload) > contract.maximum_checkpoint_bytes:
        raise CheckpointSchemaError(
            "serialized checkpoint exceeds the immutable maximum byte bound"
        )
    identity = make_checkpoint_identity(
        logical_key=contract.logical_key,
        checkpoint_role=contract.checkpoint_role,
        scientific_state_sha256=state_sha256,
        checkpoint_payload_sha256=hashlib.sha256(payload).hexdigest(),
        checkpoint_manifest_body_sha256=_manifest_body_sha256(manifest_body),
        size_bytes=len(payload),
    )
    _decode_validated_envelope(
        payload,
        expected_contract=contract,
        expected_identity=identity,
    )
    try:
        published = durability_backend.publish_bytes(destination, payload)
    except Exception as exc:
        raise CheckpointSecurityError(
            f"durability-backed checkpoint publication failed: {exc}"
        ) from exc
    if type(published) is not DurableFileIdentity:
        raise CheckpointSecurityError(
            "durability backend returned an untyped file identity"
        )
    if (
        published.size_bytes != len(payload)
        or published.sha256 != identity.checkpoint_payload_sha256
        or published.reopen_verified is not True
    ):
        raise CheckpointSecurityError(
            "durability identity differs from the checkpoint payload"
        )
    reopened = _read_authenticated_payload(
        destination,
        expected_identity=identity,
        maximum_checkpoint_bytes=contract.maximum_checkpoint_bytes,
    )
    _decode_validated_envelope(
        reopened,
        expected_contract=contract,
        expected_identity=identity,
    )
    return identity


def load_checkpoint(
    path: Path,
    *,
    expected_identity: WT103CheckpointIdentity,
    expected_contract: ResumeContract,
    fresh_target: FreshCheckpointTarget,
) -> LoadedCheckpoint:
    """Authenticate and validate completely before restoring a fresh target."""

    _validate_contract_and_identity(
        contract=expected_contract,
        identity=expected_identity,
    )
    _validate_fresh_target(fresh_target, contract=expected_contract)
    payload = _read_authenticated_payload(
        path,
        expected_identity=expected_identity,
        maximum_checkpoint_bytes=expected_contract.maximum_checkpoint_bytes,
    )
    loaded, state = _decode_validated_envelope(
        payload,
        expected_contract=expected_contract,
        expected_identity=expected_identity,
    )
    try:
        fresh_target.validate_checkpoint_state(state)
    except Exception as exc:
        raise CheckpointCompatibilityError(
            "fresh target rejected checkpoint state before mutation"
        ) from exc
    if fresh_target.is_fresh_checkpoint_target() is not True:
        raise CheckpointCompatibilityError(
            "fresh target mutated during pre-restore validation"
        )
    try:
        fresh_target.restore_checkpoint_state(state)
    except Exception as exc:
        raise CheckpointCompatibilityError(
            "fresh target failed while restoring validated state"
        ) from exc
    if fresh_target.is_fresh_checkpoint_target() is not False:
        raise CheckpointCompatibilityError(
            "fresh target did not record the completed restoration"
        )
    return loaded


__all__ = [
    "FreshCheckpointTarget",
    "load_checkpoint",
    "save_checkpoint",
]
