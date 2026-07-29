"""Hermetic environment, capacity, and resource authorization records.

Task 10 owns only record construction and exact arithmetic.  Live hardware,
package, power, and corpus observations are supplied by later integration
tasks, which keeps this module import-safe and unit-testable.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Literal

from vfe4.artifacts.durability import canonical_json_bytes_generic
from vfe4.types.results import GateStatus
from vfe4.types.training import (
    EndpointInventory,
    ResourceProfile,
    WT103ArmSpec,
    owned_sha256,
)


_GIB = 1024**3
_ALLOCATION_EVENTS = (
    "data_transfer",
    "forward",
    "model_backward",
    "optimizer_update",
    "validation_scorer",
    "metric_record",
    "checkpoint_serialization",
)


def _sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _text(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _exact_int(
    value: object,
    name: str,
    *,
    minimum: int = 0,
) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an exact int >= {minimum}")
    return value


def _finite_float(
    value: object,
    name: str,
    *,
    minimum: float = 0.0,
) -> float:
    if (
        type(value) is not float
        or not math.isfinite(value)
        or value < minimum
    ):
        raise ValueError(f"{name} must be a finite float >= {minimum}")
    return value


def _git_head(value: object) -> str:
    if (
        type(value) is not str
        or len(value) not in (40, 64)
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("git_head must be a concrete lowercase object ID")
    return value


@dataclass(frozen=True, slots=True)
class DistributionIdentity:
    """One exact locked or installed distribution record."""

    name: str
    version: str
    record_sha256: str

    def __post_init__(self) -> None:
        _text(self.name, "distribution name")
        _text(self.version, "distribution version")
        _sha256(self.record_sha256, "distribution record_sha256")
        if self.name != self.name.lower():
            raise ValueError("distribution name must be normalized lowercase")


def _distribution_inventory(
    values: object,
    name: str,
) -> tuple[DistributionIdentity, ...]:
    if (
        type(values) is not tuple
        or any(type(item) is not DistributionIdentity for item in values)
    ):
        raise ValueError(f"{name} must contain exact DistributionIdentity records")
    for item in values:
        item.__post_init__()
    names = tuple(item.name for item in values)
    if names != tuple(sorted(names)) or len(names) != len(set(names)):
        raise ValueError(f"{name} must be sorted and unique")
    return values


_LOCK_MARKER = re.compile(
    r'^python_version >= "(?P<major>[0-9]+)\.(?P<minor>[0-9]+)"$'
)


def _normalized_distribution_name(value: str) -> str:
    _text(value, "distribution name")
    return re.sub(r"[-_.]+", "-", value).lower()


@dataclass(frozen=True, slots=True)
class LockRequirement:
    """One reviewed direct distribution declaration for the lock writer."""

    name: str
    version: str
    environment_marker: str
    artifact_filename: str
    artifact_url: str
    artifact_size_bytes: int
    artifact_sha256s: tuple[str, ...]
    expected_installed_record_sha256: str | None
    task13_obligation: str | None

    def __post_init__(self) -> None:
        if self.name != _normalized_distribution_name(self.name):
            raise ValueError("lock requirement name must be normalized")
        _text(self.version, "lock requirement version")
        _text(self.artifact_filename, "artifact_filename")
        _text(self.artifact_url, "artifact_url")
        _exact_int(
            self.artifact_size_bytes,
            "artifact_size_bytes",
            minimum=1,
        )
        if (
            "/" in self.artifact_filename
            or "\\" in self.artifact_filename
            or not self.artifact_url.startswith("https://")
            or not self.artifact_url.endswith(self.artifact_filename)
        ):
            raise ValueError("lock requirement artifact source is invalid")
        match = _LOCK_MARKER.fullmatch(self.environment_marker)
        if match is None:
            raise ValueError("lock requirement marker is unresolved")
        if (
            type(self.artifact_sha256s) is not tuple
            or not self.artifact_sha256s
            or tuple(sorted(self.artifact_sha256s))
            != self.artifact_sha256s
            or len(set(self.artifact_sha256s))
            != len(self.artifact_sha256s)
        ):
            raise ValueError(
                "lock requirement needs sorted unique artifact hashes"
            )
        for digest in self.artifact_sha256s:
            _sha256(digest, "artifact_sha256")
        if self.expected_installed_record_sha256 is None:
            expected = (
                "task13_capture_exact_installed_record_sha256:"
                f"{self.name}"
            )
            if self.task13_obligation != expected:
                raise ValueError(
                    "missing installed RECORD identity needs its exact "
                    "Task13 obligation"
                )
        else:
            _sha256(
                self.expected_installed_record_sha256,
                "expected_installed_record_sha256",
            )
            if self.task13_obligation is not None:
                raise ValueError(
                    "resolved installed RECORD identity cannot retain "
                    "a Task13 obligation"
                )


@dataclass(frozen=True, slots=True)
class LockInputManifest:
    """Canonical reviewed input to the deterministic dependency-lock writer."""

    schema_version: Literal["wt103-lock-input-manifest-v1"]
    writer_schema_version: Literal["wt103-lock-writer-v1"]
    writer_code_sha256: str
    source_kind: Literal["reviewed_direct_declarations"]
    target_python_version: str
    requirements: tuple[LockRequirement, ...]
    task13_obligations: tuple[str, ...]
    manifest_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in tuple(self.__dataclass_fields__)[:-1]
        }

    def __post_init__(self) -> None:
        if (
            self.schema_version != "wt103-lock-input-manifest-v1"
            or self.writer_schema_version != "wt103-lock-writer-v1"
            or self.source_kind != "reviewed_direct_declarations"
            or re.fullmatch(r"[0-9]+\.[0-9]+", self.target_python_version)
            is None
            or type(self.requirements) is not tuple
            or not self.requirements
            or any(
                type(item) is not LockRequirement
                for item in self.requirements
            )
        ):
            raise ValueError("lock input manifest schema is invalid")
        _sha256(self.writer_code_sha256, "writer_code_sha256")
        for requirement in self.requirements:
            requirement.__post_init__()
        names = tuple(item.name for item in self.requirements)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise ValueError(
                "lock input requirements must be sorted and unique"
            )
        expected_obligations = tuple(
            item.task13_obligation
            for item in self.requirements
            if item.task13_obligation is not None
        )
        if self.task13_obligations != expected_obligations:
            raise ValueError(
                "lock input Task13 obligations are not requirement-derived"
            )
        expected = owned_sha256(
            "vfe4.wt103.lock-input-manifest.v1",
            self.semantic_payload(),
        )
        _sha256(self.manifest_sha256, "manifest_sha256")
        if self.manifest_sha256 != expected:
            raise ValueError("lock input manifest hash does not match")

    @classmethod
    def create(
        cls,
        *,
        writer_code_sha256: str,
        target_python_version: str,
        requirements: tuple[LockRequirement, ...],
    ) -> "LockInputManifest":
        obligations = tuple(
            item.task13_obligation
            for item in requirements
            if item.task13_obligation is not None
        )
        payload = {
            "schema_version": "wt103-lock-input-manifest-v1",
            "writer_schema_version": "wt103-lock-writer-v1",
            "writer_code_sha256": writer_code_sha256,
            "source_kind": "reviewed_direct_declarations",
            "target_python_version": target_python_version,
            "requirements": requirements,
            "task13_obligations": obligations,
        }
        return cls(
            **payload,
            manifest_sha256=owned_sha256(
                "vfe4.wt103.lock-input-manifest.v1",
                payload,
            ),
        )  # type: ignore[arg-type]


def _marker_applies(requirement: LockRequirement, target: str) -> bool:
    match = _LOCK_MARKER.fullmatch(requirement.environment_marker)
    if match is None:
        raise ValueError("lock requirement marker is unresolved")
    target_parts = tuple(int(item) for item in target.split("."))
    minimum = (int(match.group("major")), int(match.group("minor")))
    return target_parts >= minimum


def render_dependency_lock(manifest: LockInputManifest) -> bytes:
    """Render reviewed declarations without imports, installs, or pip state."""

    if type(manifest) is not LockInputManifest:
        raise ValueError("lock_input_manifest must be exact")
    manifest.__post_init__()
    lines = (
        "# Generated from requirements-wt103.lock-input.json.",
        f"# writer-schema={manifest.writer_schema_version}",
        f"# writer-code-sha256={manifest.writer_code_sha256}",
        f"# target-python-version={manifest.target_python_version}",
    )
    rendered = list(lines)
    for requirement in manifest.requirements:
        if not _marker_applies(
            requirement,
            manifest.target_python_version,
        ):
            raise ValueError(
                f"lock requirement marker is unresolved:{requirement.name}"
            )
        if requirement.expected_installed_record_sha256 is None:
            rendered.append(f"# obligation={requirement.task13_obligation}")
        else:
            rendered.append(
                "# expected-installed-record-sha256="
                f"{requirement.name}:"
                f"{requirement.expected_installed_record_sha256}"
            )
        rendered.extend(
            (
                f"# artifact-filename={requirement.artifact_filename}",
                f"# artifact-size-bytes={requirement.artifact_size_bytes}",
                f"# artifact-url={requirement.artifact_url}",
            )
        )
        rendered.append(
            f"{requirement.name}=={requirement.version} ; "
            f'{requirement.environment_marker} \\'
        )
        hashes = requirement.artifact_sha256s
        for index, digest in enumerate(hashes):
            suffix = " \\" if index < len(hashes) - 1 else ""
            rendered.append(f"    --hash=sha256:{digest}{suffix}")
    return ("\n".join(rendered) + "\n").encode("utf-8")


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate lock-manifest key:{key}")
        result[key] = value
    return result


def parse_lock_input_manifest(payload: bytes) -> LockInputManifest:
    """Reopen exact canonical tracked manifest bytes without runtime probes."""

    if type(payload) is not bytes or not payload:
        raise ValueError("lock input manifest bytes must be nonempty")
    try:
        document = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=lambda value: (_raise_json_constant(value)),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("lock input manifest JSON is invalid") from exc
    if type(document) is not dict:
        raise ValueError("lock input manifest must be a JSON object")
    if payload != canonical_json_bytes_generic(document) + b"\n":
        raise ValueError("lock input manifest is not canonical JSONL")
    expected_keys = {
        "manifest_sha256",
        "requirements",
        "schema_version",
        "source_kind",
        "target_python_version",
        "task13_obligations",
        "writer_code_sha256",
        "writer_schema_version",
    }
    if set(document) != expected_keys:
        raise ValueError("lock input manifest key set is open")
    raw_requirements = document["requirements"]
    if type(raw_requirements) is not list:
        raise ValueError("lock input requirements must be a list")
    requirement_keys = {
        "artifact_filename",
        "artifact_sha256s",
        "artifact_size_bytes",
        "artifact_url",
        "environment_marker",
        "expected_installed_record_sha256",
        "name",
        "task13_obligation",
        "version",
    }
    requirements: list[LockRequirement] = []
    for raw in raw_requirements:
        if type(raw) is not dict or set(raw) != requirement_keys:
            raise ValueError("lock requirement key set is open")
        values = dict(raw)
        hashes = values["artifact_sha256s"]
        if type(hashes) is not list:
            raise ValueError("artifact_sha256s must be a list")
        values["artifact_sha256s"] = tuple(hashes)
        requirements.append(LockRequirement(**values))
    values = dict(document)
    values["requirements"] = tuple(requirements)
    obligations = values["task13_obligations"]
    if type(obligations) is not list:
        raise ValueError("task13_obligations must be a list")
    values["task13_obligations"] = tuple(obligations)
    return LockInputManifest(**values)


def _raise_json_constant(value: str) -> object:
    raise ValueError(f"nonfinite lock-manifest constant:{value}")


def _locked_distributions_from_manifest(
    manifest: LockInputManifest,
) -> tuple[DistributionIdentity, ...]:
    if manifest.task13_obligations:
        return ()
    result = tuple(
        DistributionIdentity(
            name=item.name,
            version=item.version,
            record_sha256=item.expected_installed_record_sha256,
        )
        for item in manifest.requirements
    )
    _distribution_inventory(result, "locked_distributions")
    return result


def _distribution_coordinates(
    values: tuple[DistributionIdentity, ...],
) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (item.name, item.version, item.record_sha256) for item in values
    )


@dataclass(frozen=True, slots=True)
class DependencyLockIdentity:
    """Exact tracked lock bytes plus installed-distribution comparison."""

    schema_version: Literal["wt103-dependency-lock-identity-v2"]
    lock_relative_path: Literal["requirements-wt103.lock"]
    lock_text: str
    lock_size_bytes: int
    lock_sha256: str
    expected_lock_sha256: str
    lock_input_manifest: LockInputManifest
    lock_input_manifest_sha256: str
    task13_obligations: tuple[str, ...]
    locked_distributions: tuple[DistributionIdentity, ...]
    installed_distributions: tuple[DistributionIdentity, ...]
    installed_match: bool
    status: GateStatus
    obligations: tuple[str, ...]
    identity_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in tuple(self.__dataclass_fields__)[:-1]
        }

    def __post_init__(self) -> None:
        if (
            self.schema_version != "wt103-dependency-lock-identity-v2"
            or self.lock_relative_path != "requirements-wt103.lock"
            or type(self.lock_text) is not str
            or not self.lock_text
            or type(self.installed_match) is not bool
            or type(self.status) is not GateStatus
            or type(self.obligations) is not tuple
            or any(type(item) is not str or not item for item in self.obligations)
        ):
            raise ValueError("dependency lock identity schema is invalid")
        _exact_int(self.lock_size_bytes, "lock_size_bytes", minimum=1)
        try:
            exact_lock_bytes = self.lock_text.encode(
                "utf-8",
                errors="strict",
            )
        except UnicodeError as exc:
            raise ValueError("dependency lock text is not exact UTF-8") from exc
        if (
            len(exact_lock_bytes) != self.lock_size_bytes
            or hashlib.sha256(exact_lock_bytes).hexdigest()
            != self.lock_sha256
        ):
            raise ValueError("dependency lock bytes do not match their identity")
        _sha256(self.lock_sha256, "lock_sha256")
        _sha256(self.expected_lock_sha256, "expected_lock_sha256")
        _sha256(
            self.lock_input_manifest_sha256,
            "lock_input_manifest_sha256",
        )
        if type(self.lock_input_manifest) is not LockInputManifest:
            raise ValueError("lock input manifest must be exact")
        self.lock_input_manifest.__post_init__()
        if (
            self.lock_input_manifest_sha256
            != self.lock_input_manifest.manifest_sha256
            or exact_lock_bytes
            != render_dependency_lock(self.lock_input_manifest)
            or self.task13_obligations
            != self.lock_input_manifest.task13_obligations
        ):
            raise ValueError(
                "dependency lock is not bound to its exact manifest"
            )
        if (
            type(self.task13_obligations) is not tuple
            or any(
                type(item) is not str or not item
                for item in self.task13_obligations
            )
        ):
            raise ValueError("Task13 lock obligations are invalid")
        _distribution_inventory(
            self.locked_distributions,
            "locked_distributions",
        )
        _distribution_inventory(
            self.installed_distributions,
            "installed_distributions",
        )
        if self.locked_distributions != _locked_distributions_from_manifest(
            self.lock_input_manifest
        ):
            raise ValueError(
                "locked distributions are not derived from their manifest"
            )
        expected_obligations: list[str] = []
        if self.lock_sha256 != self.expected_lock_sha256:
            expected_obligations.append("tracked_lock_sha256_mismatch")
        expected_obligations.extend(self.task13_obligations)
        if _distribution_coordinates(
            self.locked_distributions
        ) != _distribution_coordinates(self.installed_distributions):
            expected_obligations.append(
                "installed_distributions_do_not_match_lock"
            )
        expected_match = not expected_obligations
        if self.installed_match is not expected_match:
            raise ValueError("installed_match is not derived from lock records")
        if self.obligations != tuple(expected_obligations):
            raise ValueError(
                "dependency lock obligations are not exact"
            )
        if (
            (self.status is GateStatus.PASS and self.obligations)
            or (self.status is not GateStatus.PASS and not self.obligations)
            or self.status
            is not (
                GateStatus.PASS
                if expected_match
                else GateStatus.INCONCLUSIVE
            )
        ):
            raise ValueError("dependency lock status/obligations disagree")
        expected = owned_sha256(
            "vfe4.wt103.dependency-lock-identity.v2",
            self.semantic_payload(),
        )
        _sha256(self.identity_sha256, "identity_sha256")
        if self.identity_sha256 != expected:
            raise ValueError("dependency lock identity hash does not match")

    @classmethod
    def capture(
        cls,
        *,
        lock_relative_path: str,
        lock_bytes: bytes,
        expected_sha256: str,
        lock_input_manifest: LockInputManifest,
        installed_distributions: tuple[DistributionIdentity, ...],
    ) -> "DependencyLockIdentity":
        if lock_relative_path != "requirements-wt103.lock":
            raise ValueError("tracked lock path must be requirements-wt103.lock")
        if type(lock_bytes) is not bytes or not lock_bytes:
            raise ValueError("lock_bytes must be nonempty exact bytes")
        _sha256(expected_sha256, "expected_sha256")
        if type(lock_input_manifest) is not LockInputManifest:
            raise ValueError("lock_input_manifest must be exact")
        lock_input_manifest.__post_init__()
        _distribution_inventory(
            installed_distributions,
            "installed_distributions",
        )
        try:
            lock_text = lock_bytes.decode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise ValueError("lock_bytes must be exact UTF-8") from exc
        if lock_bytes != render_dependency_lock(lock_input_manifest):
            raise ValueError(
                "dependency lock bytes are not generated from their manifest"
            )
        locked_distributions = _locked_distributions_from_manifest(
            lock_input_manifest
        )
        observed = hashlib.sha256(lock_bytes).hexdigest()
        obligations: list[str] = []
        if observed != expected_sha256:
            obligations.append("tracked_lock_sha256_mismatch")
        obligations.extend(lock_input_manifest.task13_obligations)
        if _distribution_coordinates(
            locked_distributions
        ) != _distribution_coordinates(installed_distributions):
            obligations.append("installed_distributions_do_not_match_lock")
        payload = {
            "schema_version": "wt103-dependency-lock-identity-v2",
            "lock_relative_path": "requirements-wt103.lock",
            "lock_text": lock_text,
            "lock_size_bytes": len(lock_bytes),
            "lock_sha256": observed,
            "expected_lock_sha256": expected_sha256,
            "lock_input_manifest": lock_input_manifest,
            "lock_input_manifest_sha256": (
                lock_input_manifest.manifest_sha256
            ),
            "task13_obligations": (
                lock_input_manifest.task13_obligations
            ),
            "locked_distributions": locked_distributions,
            "installed_distributions": installed_distributions,
            "installed_match": not obligations,
            "status": (
                GateStatus.PASS
                if not obligations
                else GateStatus.INCONCLUSIVE
            ),
            "obligations": tuple(obligations),
        }
        return cls(
            **payload,
            identity_sha256=owned_sha256(
                "vfe4.wt103.dependency-lock-identity.v2",
                payload,
            ),
        )  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class EnvironmentObservation:
    """Injected facts captured before any device work starts."""

    captured_utc: str
    device_work_started: bool
    python_version: str
    pytorch_version: str
    cuda_runtime_version: str
    cudnn_version: str
    driver_version: str
    os_name: str
    platform_system: str
    platform_release: str
    cpu_name: str
    logical_cpu_count: int
    physical_ram_bytes: int
    gpu_names: tuple[str, ...]
    gpu_device_uuids: tuple[str, ...]
    gpu_total_bytes: tuple[int, ...]
    compute_capabilities: tuple[str, ...]
    blas_identity_sha256: str
    thread_settings_sha256: str
    deterministic_algorithms: bool
    cudnn_benchmark: bool
    locale_name: str
    timezone_name: str

    def __post_init__(self) -> None:
        for name in (
            "captured_utc",
            "python_version",
            "pytorch_version",
            "cuda_runtime_version",
            "cudnn_version",
            "driver_version",
            "os_name",
            "platform_system",
            "platform_release",
            "cpu_name",
            "locale_name",
            "timezone_name",
        ):
            _text(getattr(self, name), name)
        if (
            type(self.device_work_started) is not bool
            or type(self.deterministic_algorithms) is not bool
            or type(self.cudnn_benchmark) is not bool
        ):
            raise ValueError("environment flags must be exact bool values")
        _exact_int(self.logical_cpu_count, "logical_cpu_count", minimum=1)
        _exact_int(self.physical_ram_bytes, "physical_ram_bytes", minimum=1)
        if (
            type(self.gpu_names) is not tuple
            or not self.gpu_names
            or any(type(item) is not str or not item for item in self.gpu_names)
            or type(self.gpu_device_uuids) is not tuple
            or any(
                type(item) is not str or not item
                for item in self.gpu_device_uuids
            )
            or type(self.gpu_total_bytes) is not tuple
            or type(self.compute_capabilities) is not tuple
            or len(self.gpu_names) != len(self.gpu_device_uuids)
            or len(self.gpu_names) != len(self.gpu_total_bytes)
            or len(self.gpu_names) != len(self.compute_capabilities)
            or any(type(item) is not int or item <= 0 for item in self.gpu_total_bytes)
            or any(
                type(item) is not str or not item
                for item in self.compute_capabilities
            )
        ):
            raise ValueError("GPU inventories must be aligned nonempty tuples")
        _sha256(self.blas_identity_sha256, "blas_identity_sha256")
        _sha256(
            self.thread_settings_sha256,
            "thread_settings_sha256",
        )


@dataclass(frozen=True, slots=True)
class EnvironmentRecord:
    """Content-bound environment record produced before device work."""

    schema_version: Literal["wt103-environment-record-v1"]
    observation: EnvironmentObservation
    dependency_lock_identity_sha256: str
    captured_before_device_work: Literal[True]
    hardware_identity_sha256: str
    runtime_identity_sha256: str
    environment_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in tuple(self.__dataclass_fields__)[:-1]
        }

    def __post_init__(self) -> None:
        if (
            self.schema_version != "wt103-environment-record-v1"
            or type(self.observation) is not EnvironmentObservation
            or self.captured_before_device_work is not True
            or self.observation.device_work_started is not False
        ):
            raise ValueError("environment was not captured before device work")
        self.observation.__post_init__()
        for name in (
            "dependency_lock_identity_sha256",
            "hardware_identity_sha256",
            "runtime_identity_sha256",
        ):
            _sha256(getattr(self, name), name)
        expected = owned_sha256(
            "vfe4.wt103.environment-record.v1",
            self.semantic_payload(),
        )
        _sha256(self.environment_sha256, "environment_sha256")
        if self.environment_sha256 != expected:
            raise ValueError("environment_sha256 does not match record")


def capture_environment(
    observation: EnvironmentObservation,
    *,
    dependency_lock: DependencyLockIdentity,
) -> EnvironmentRecord:
    """Bind injected pre-device facts without importing a live runtime."""

    if type(observation) is not EnvironmentObservation:
        raise ValueError("observation must be exact EnvironmentObservation")
    if type(dependency_lock) is not DependencyLockIdentity:
        raise ValueError("dependency_lock must be exact")
    observation.__post_init__()
    dependency_lock.__post_init__()
    if observation.device_work_started:
        raise ValueError("environment must be captured before device work")
    hardware = owned_sha256(
        "vfe4.wt103.hardware-identity.v1",
        {
            "cpu_name": observation.cpu_name,
            "logical_cpu_count": observation.logical_cpu_count,
            "physical_ram_bytes": observation.physical_ram_bytes,
            "gpu_names": observation.gpu_names,
            "gpu_device_uuids": observation.gpu_device_uuids,
            "gpu_total_bytes": observation.gpu_total_bytes,
            "compute_capabilities": observation.compute_capabilities,
        },
    )
    runtime = owned_sha256(
        "vfe4.wt103.runtime-identity.v1",
        {
            "python_version": observation.python_version,
            "pytorch_version": observation.pytorch_version,
            "cuda_runtime_version": observation.cuda_runtime_version,
            "cudnn_version": observation.cudnn_version,
            "driver_version": observation.driver_version,
            "blas_identity_sha256": observation.blas_identity_sha256,
            "thread_settings_sha256": observation.thread_settings_sha256,
            "deterministic_algorithms": observation.deterministic_algorithms,
            "cudnn_benchmark": observation.cudnn_benchmark,
            "locale_name": observation.locale_name,
            "timezone_name": observation.timezone_name,
        },
    )
    payload = {
        "schema_version": "wt103-environment-record-v1",
        "observation": observation,
        "dependency_lock_identity_sha256": dependency_lock.identity_sha256,
        "captured_before_device_work": True,
        "hardware_identity_sha256": hardware,
        "runtime_identity_sha256": runtime,
    }
    return EnvironmentRecord(
        **payload,
        environment_sha256=owned_sha256(
            "vfe4.wt103.environment-record.v1",
            payload,
        ),
    )


@dataclass(frozen=True, slots=True)
class TrainingExecutionIdentity:
    """Exact source/config/profile/factory/runtime identity for live evidence."""

    schema_version: Literal["wt103-training-execution-identity-v1"]
    git_identity_sha256: str
    git_head: str
    dirty_digest: str
    config_sha256: str
    profile_sha256: str
    factory_set_sha256: str
    environment_sha256: str
    identity_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in tuple(self.__dataclass_fields__)[:-1]
        }

    def __post_init__(self) -> None:
        if self.schema_version != "wt103-training-execution-identity-v1":
            raise ValueError("training execution identity schema is invalid")
        _git_head(self.git_head)
        for name in (
            "git_identity_sha256",
            "dirty_digest",
            "config_sha256",
            "profile_sha256",
            "factory_set_sha256",
            "environment_sha256",
        ):
            _sha256(getattr(self, name), name)
        expected = owned_sha256(
            "vfe4.wt103.training-execution-identity.v1",
            self.semantic_payload(),
        )
        _sha256(self.identity_sha256, "identity_sha256")
        if self.identity_sha256 != expected:
            raise ValueError("training execution identity hash does not match")

    @classmethod
    def create(
        cls,
        *,
        git_identity_sha256: str,
        git_head: str,
        dirty_digest: str,
        config_sha256: str,
        profile_sha256: str,
        factory_set_sha256: str,
        environment_sha256: str,
    ) -> "TrainingExecutionIdentity":
        payload = {
            "schema_version": "wt103-training-execution-identity-v1",
            "git_identity_sha256": git_identity_sha256,
            "git_head": git_head,
            "dirty_digest": dirty_digest,
            "config_sha256": config_sha256,
            "profile_sha256": profile_sha256,
            "factory_set_sha256": factory_set_sha256,
            "environment_sha256": environment_sha256,
        }
        return cls(
            **payload,
            identity_sha256=owned_sha256(
                "vfe4.wt103.training-execution-identity.v1",
                payload,
            ),
        )  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class PowerProviderIdentity:
    """Provenance-bound 100 ms board-power sampling provider."""

    schema_version: Literal["wt103-power-provider-v1"]
    provider_kind: Literal["nvml", "nvidia-smi"]
    provider_version: str
    provider_executable_sha256: str
    sample_interval_ms: Literal[100]
    reported_power_limit_watts: float
    identity_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in tuple(self.__dataclass_fields__)[:-1]
        }

    def __post_init__(self) -> None:
        if (
            self.schema_version != "wt103-power-provider-v1"
            or self.provider_kind not in ("nvml", "nvidia-smi")
            or type(self.sample_interval_ms) is not int
            or self.sample_interval_ms != 100
        ):
            raise ValueError("power provider schema is not frozen")
        _text(self.provider_version, "provider_version")
        _sha256(
            self.provider_executable_sha256,
            "provider_executable_sha256",
        )
        _finite_float(
            self.reported_power_limit_watts,
            "reported_power_limit_watts",
            minimum=0.000001,
        )
        expected = owned_sha256(
            "vfe4.wt103.power-provider.v1",
            self.semantic_payload(),
        )
        _sha256(self.identity_sha256, "identity_sha256")
        if self.identity_sha256 != expected:
            raise ValueError("power provider identity hash does not match")

    @classmethod
    def create(
        cls,
        *,
        provider_kind: Literal["nvml", "nvidia-smi"],
        provider_version: str,
        provider_executable_sha256: str,
        sample_interval_ms: int,
        reported_power_limit_watts: float,
    ) -> "PowerProviderIdentity":
        payload = {
            "schema_version": "wt103-power-provider-v1",
            "provider_kind": provider_kind,
            "provider_version": provider_version,
            "provider_executable_sha256": provider_executable_sha256,
            "sample_interval_ms": sample_interval_ms,
            "reported_power_limit_watts": reported_power_limit_watts,
        }
        return cls(
            **payload,
            identity_sha256=owned_sha256(
                "vfe4.wt103.power-provider.v1",
                payload,
            ),
        )  # type: ignore[arg-type]


def _allocation_events(spec: WT103ArmSpec) -> tuple[str, ...]:
    events = list(_ALLOCATION_EVENTS)
    if spec.recognition_enabled:
        insertion = events.index("model_backward")
        events[insertion:insertion] = (
            "recognition_proposal",
            "immutable_snapshot",
        )
    return tuple(events)


@dataclass(frozen=True, slots=True)
class AllocationObservation:
    """One independent shape-identical arm-path capacity observation."""

    schema_version: Literal["wt103-allocation-observation-v2"]
    arm_id: str
    arm_spec_sha256: str
    execution_identity: TrainingExecutionIdentity
    path_events: tuple[str, ...]
    shape_identical: Literal[True]
    device_ordinal: int
    device_uuid: str
    physical_device_bytes: int
    peak_device_allocated_bytes: int
    peak_device_reserved_bytes: int
    host_available_bytes: int
    checkpoint_duplicate_bytes: int
    disk_available_bytes: int
    observation_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in tuple(self.__dataclass_fields__)[:-1]
        }

    def __post_init__(self) -> None:
        if (
            self.schema_version != "wt103-allocation-observation-v2"
            or self.shape_identical is not True
        ):
            raise ValueError("allocation observation schema is invalid")
        _text(self.arm_id, "arm_id")
        _sha256(self.arm_spec_sha256, "arm_spec_sha256")
        if type(self.execution_identity) is not TrainingExecutionIdentity:
            raise ValueError("allocation execution identity must be exact")
        self.execution_identity.__post_init__()
        if (
            type(self.path_events) is not tuple
            or not self.path_events
            or any(type(item) is not str or not item for item in self.path_events)
        ):
            raise ValueError("allocation path events are invalid")
        _exact_int(self.device_ordinal, "device_ordinal")
        _text(self.device_uuid, "device_uuid")
        _exact_int(
            self.physical_device_bytes,
            "physical_device_bytes",
            minimum=1,
        )
        for name in (
            "peak_device_allocated_bytes",
            "peak_device_reserved_bytes",
            "host_available_bytes",
            "checkpoint_duplicate_bytes",
            "disk_available_bytes",
        ):
            _exact_int(getattr(self, name), name, minimum=0)
        expected = owned_sha256(
            "vfe4.wt103.allocation-observation.v2",
            self.semantic_payload(),
        )
        _sha256(self.observation_sha256, "observation_sha256")
        if self.observation_sha256 != expected:
            raise ValueError("allocation observation hash does not match")

    @classmethod
    def shape_identical_for_arm(
        cls,
        spec: WT103ArmSpec,
        *,
        execution_identity: TrainingExecutionIdentity,
        device_ordinal: int,
        device_uuid: str,
        physical_device_bytes: int,
        peak_device_allocated_bytes: int,
        peak_device_reserved_bytes: int,
        host_available_bytes: int,
        checkpoint_duplicate_bytes: int,
        disk_available_bytes: int,
    ) -> "AllocationObservation":
        if type(spec) is not WT103ArmSpec:
            raise ValueError("spec must be exact WT103ArmSpec")
        if type(execution_identity) is not TrainingExecutionIdentity:
            raise ValueError("execution_identity must be exact")
        spec.__post_init__()
        execution_identity.__post_init__()
        payload = {
            "schema_version": "wt103-allocation-observation-v2",
            "arm_id": spec.arm_id,
            "arm_spec_sha256": spec.arm_spec_sha256,
            "execution_identity": execution_identity,
            "path_events": _allocation_events(spec),
            "shape_identical": True,
            "device_ordinal": device_ordinal,
            "device_uuid": device_uuid,
            "physical_device_bytes": physical_device_bytes,
            "peak_device_allocated_bytes": peak_device_allocated_bytes,
            "peak_device_reserved_bytes": peak_device_reserved_bytes,
            "host_available_bytes": host_available_bytes,
            "checkpoint_duplicate_bytes": checkpoint_duplicate_bytes,
            "disk_available_bytes": disk_available_bytes,
        }
        return cls(
            **payload,
            observation_sha256=owned_sha256(
                "vfe4.wt103.allocation-observation.v2",
                payload,
            ),
        )  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class AllocationPreflightRecord:
    """Inventory-wide capacity record independent of H8 sparsity evidence."""

    schema_version: Literal["wt103-allocation-preflight-v1"]
    endpoint_inventory_sha256: str
    environment_sha256: str
    execution_identity: TrainingExecutionIdentity
    observation_sha256s: tuple[str, ...]
    maximum_device_fraction: Literal[0.85]
    maximum_peak_allocated_fraction: float
    maximum_peak_reserved_fraction: float
    status: GateStatus
    obligations: tuple[str, ...]
    h8_evidence_accepted: Literal[False]
    record_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in tuple(self.__dataclass_fields__)[:-1]
        }

    def __post_init__(self) -> None:
        if (
            self.schema_version != "wt103-allocation-preflight-v1"
            or type(self.maximum_device_fraction) is not float
            or self.maximum_device_fraction != 0.85
            or type(self.status) is not GateStatus
            or self.h8_evidence_accepted is not False
            or type(self.obligations) is not tuple
        ):
            raise ValueError("allocation preflight schema is invalid")
        _sha256(
            self.endpoint_inventory_sha256,
            "endpoint_inventory_sha256",
        )
        _sha256(self.environment_sha256, "environment_sha256")
        if type(self.execution_identity) is not TrainingExecutionIdentity:
            raise ValueError("allocation execution identity must be exact")
        self.execution_identity.__post_init__()
        if (
            type(self.observation_sha256s) is not tuple
            or not self.observation_sha256s
        ):
            raise ValueError("allocation observation inventory is empty")
        for value in self.observation_sha256s:
            _sha256(value, "observation_sha256")
        for name in (
            "maximum_peak_allocated_fraction",
            "maximum_peak_reserved_fraction",
        ):
            value = _finite_float(getattr(self, name), name)
            if value > 1.0:
                raise ValueError(f"{name} cannot exceed one")
        if (
            (self.status is GateStatus.PASS and self.obligations)
            or (self.status is not GateStatus.PASS and not self.obligations)
        ):
            raise ValueError("allocation status/obligations disagree")
        expected = owned_sha256(
            "vfe4.wt103.allocation-preflight.v1",
            self.semantic_payload(),
        )
        _sha256(self.record_sha256, "record_sha256")
        if self.record_sha256 != expected:
            raise ValueError("allocation preflight hash does not match")


def run_allocation_preflight(
    *,
    endpoint_inventory: EndpointInventory,
    observations: tuple[AllocationObservation, ...],
    execution_identity: TrainingExecutionIdentity,
    environment: EnvironmentRecord,
    maximum_device_fraction: float,
    h8_allocation_evidence: None,
) -> AllocationPreflightRecord:
    """Validate every arm independently against the exact 85% device cap."""

    if h8_allocation_evidence is not None:
        raise ValueError("H8 allocation evidence cannot populate training capacity")
    if type(endpoint_inventory) is not EndpointInventory:
        raise ValueError("endpoint_inventory must be exact")
    if type(execution_identity) is not TrainingExecutionIdentity:
        raise ValueError("execution_identity must be exact")
    if type(environment) is not EnvironmentRecord:
        raise ValueError("environment must be exact")
    endpoint_inventory.__post_init__()
    execution_identity.__post_init__()
    environment.__post_init__()
    if execution_identity.environment_sha256 != environment.environment_sha256:
        raise ValueError(
            "allocation execution identity differs from captured environment"
        )
    if (
        type(maximum_device_fraction) is not float
        or maximum_device_fraction != 0.85
    ):
        raise ValueError("maximum_device_fraction is frozen at 0.85")
    if (
        type(observations) is not tuple
        or tuple(item.arm_id for item in observations)
        != tuple(item.arm_id for item in endpoint_inventory.arms)
        or any(type(item) is not AllocationObservation for item in observations)
    ):
        raise ValueError("allocation observations must exactly follow arm inventory")
    obligations: list[str] = []
    allocated_fractions: list[float] = []
    reserved_fractions: list[float] = []
    for observation, arm in zip(
        observations,
        endpoint_inventory.arms,
        strict=True,
    ):
        observation.__post_init__()
        if (
            observation.execution_identity != execution_identity
            or observation.arm_spec_sha256 != arm.arm_spec_sha256
            or observation.path_events != _allocation_events(arm)
        ):
            raise ValueError(
                "allocation path/execution identity is not "
                f"shape-identical:{arm.arm_id}"
            )
        ordinal = observation.device_ordinal
        if (
            ordinal >= len(environment.observation.gpu_names)
            or observation.device_uuid
            != environment.observation.gpu_device_uuids[ordinal]
            or observation.physical_device_bytes
            != environment.observation.gpu_total_bytes[ordinal]
        ):
            raise ValueError(
                "allocation denominator differs from captured device "
                f"ordinal/UUID/bytes:{arm.arm_id}"
            )
        allocated_fractions.append(
            observation.peak_device_allocated_bytes
            / observation.physical_device_bytes
        )
        reserved_fractions.append(
            observation.peak_device_reserved_bytes
            / observation.physical_device_bytes
        )
        if (
            observation.peak_device_allocated_bytes
            > observation.peak_device_reserved_bytes
        ):
            obligations.append(
                f"device_allocated_exceeds_reserved:{arm.arm_id}"
            )
        if (
            observation.peak_device_allocated_bytes * 100
            > observation.physical_device_bytes * 85
        ):
            obligations.append(
                f"device_allocated_over_85_percent:{arm.arm_id}"
            )
        if (
            observation.peak_device_reserved_bytes * 100
            > observation.physical_device_bytes * 85
        ):
            obligations.append(
                f"device_reserved_over_85_percent:{arm.arm_id}"
            )
        if (
            observation.host_available_bytes
            < observation.checkpoint_duplicate_bytes
        ):
            obligations.append(
                f"host_checkpoint_duplicate_headroom_insufficient:{arm.arm_id}"
            )
        if (
            observation.disk_available_bytes
            < observation.checkpoint_duplicate_bytes
        ):
            obligations.append(
                f"disk_checkpoint_duplicate_headroom_insufficient:{arm.arm_id}"
            )
    payload = {
        "schema_version": "wt103-allocation-preflight-v1",
        "endpoint_inventory_sha256": (
            endpoint_inventory.endpoint_inventory_sha256
        ),
        "environment_sha256": environment.environment_sha256,
        "execution_identity": execution_identity,
        "observation_sha256s": tuple(
            item.observation_sha256 for item in observations
        ),
        "maximum_device_fraction": 0.85,
        "maximum_peak_allocated_fraction": max(allocated_fractions),
        "maximum_peak_reserved_fraction": max(reserved_fractions),
        "status": GateStatus.PASS if not obligations else GateStatus.FAIL,
        "obligations": tuple(dict.fromkeys(obligations)),
        "h8_evidence_accepted": False,
    }
    return AllocationPreflightRecord(
        **payload,
        record_sha256=owned_sha256(
            "vfe4.wt103.allocation-preflight.v1",
            payload,
        ),
    )


@dataclass(frozen=True, slots=True)
class ResourceWorkload:
    """Corpus work counts; attempt/endpoint multipliers remain inventory-owned."""

    train_batches_per_pass: int
    validation_batches_per_full_evaluation: int
    test_batches_per_full_evaluation: int
    preparation_source_work_units: int
    preparation_tokenizer_work_units: int
    preparation_window_work_units: int

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _exact_int(getattr(self, name), name, minimum=1)


@dataclass(frozen=True, slots=True)
class ResourceComponentSpec:
    component_id: str
    work_unit_kind: str
    work_units: int
    operation_count: int
    uses_gpu: bool
    warmup_count: int
    sample_count: int
    endpoint_inventory_sha256: str
    spec_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in tuple(self.__dataclass_fields__)[:-1]
        }

    def __post_init__(self) -> None:
        _text(self.component_id, "component_id")
        _text(self.work_unit_kind, "work_unit_kind")
        _exact_int(self.work_units, "work_units", minimum=1)
        _exact_int(self.operation_count, "operation_count", minimum=1)
        if type(self.uses_gpu) is not bool:
            raise ValueError("uses_gpu must be exact bool")
        _exact_int(self.warmup_count, "warmup_count")
        _exact_int(self.sample_count, "sample_count", minimum=1)
        _sha256(
            self.endpoint_inventory_sha256,
            "endpoint_inventory_sha256",
        )
        expected = owned_sha256(
            "vfe4.wt103.resource-component-spec.v1",
            self.semantic_payload(),
        )
        _sha256(self.spec_sha256, "spec_sha256")
        if self.spec_sha256 != expected:
            raise ValueError("resource component spec hash does not match")


def _component(
    *,
    component_id: str,
    work_unit_kind: str,
    work_units: int,
    operation_count: int,
    uses_gpu: bool,
    warmup_count: int,
    sample_count: int,
    endpoint_inventory_sha256: str,
) -> ResourceComponentSpec:
    payload = {
        "component_id": component_id,
        "work_unit_kind": work_unit_kind,
        "work_units": work_units,
        "operation_count": operation_count,
        "uses_gpu": uses_gpu,
        "warmup_count": warmup_count,
        "sample_count": sample_count,
        "endpoint_inventory_sha256": endpoint_inventory_sha256,
    }
    return ResourceComponentSpec(
        **payload,
        spec_sha256=owned_sha256(
            "vfe4.wt103.resource-component-spec.v1",
            payload,
        ),
    )


def required_resource_components(
    endpoint_inventory: EndpointInventory,
    workload: ResourceWorkload,
) -> tuple[ResourceComponentSpec, ...]:
    """Derive the complete benchmark/forecast component inventory."""

    if type(endpoint_inventory) is not EndpointInventory:
        raise ValueError("endpoint_inventory must be exact")
    if type(workload) is not ResourceWorkload:
        raise ValueError("workload must be exact")
    endpoint_inventory.__post_init__()
    workload.__post_init__()
    identity = endpoint_inventory.endpoint_inventory_sha256
    components: list[ResourceComponentSpec] = []
    for suffix, units in (
        ("source", workload.preparation_source_work_units),
        ("tokenizer", workload.preparation_tokenizer_work_units),
        ("windows", workload.preparation_window_work_units),
    ):
        components.append(
            _component(
                component_id=f"preparation/{suffix}",
                work_unit_kind="preparation_work_unit",
                work_units=units,
                operation_count=1,
                uses_gpu=False,
                warmup_count=0,
                sample_count=1,
                endpoint_inventory_sha256=identity,
            )
        )

    for arm in endpoint_inventory.arms:
        tuning_attempts = sum(
            key.startswith(f"tuning/{arm.arm_id}/")
            for key in endpoint_inventory.tuning_attempt_keys
        )
        confirmation_attempts = sum(
            key.startswith(f"terminal/{arm.arm_id}/")
            for key in endpoint_inventory.terminal_checkpoint_keys
        )
        quarter_batches = (
            workload.train_batches_per_pass + 3
        ) // 4
        components.extend(
            (
                _component(
                    component_id=f"tuning/train/{arm.arm_id}",
                    work_unit_kind="optimizer_update",
                    work_units=tuning_attempts * quarter_batches,
                    operation_count=tuning_attempts,
                    uses_gpu=True,
                    warmup_count=5,
                    sample_count=20,
                    endpoint_inventory_sha256=identity,
                ),
                _component(
                    component_id=f"confirmation/train/{arm.arm_id}",
                    work_unit_kind="optimizer_update",
                    work_units=(
                        confirmation_attempts
                        * 2
                        * workload.train_batches_per_pass
                    ),
                    operation_count=confirmation_attempts,
                    uses_gpu=True,
                    warmup_count=5,
                    sample_count=20,
                    endpoint_inventory_sha256=identity,
                ),
                _component(
                    component_id=f"validation/{arm.arm_id}",
                    work_unit_kind="validation_batch",
                    work_units=(
                        tuning_attempts
                        + confirmation_attempts * 2 * 20
                    )
                    * workload.validation_batches_per_full_evaluation,
                    operation_count=(
                        tuning_attempts
                        + confirmation_attempts * 2 * 20
                    ),
                    uses_gpu=True,
                    warmup_count=0,
                    sample_count=10,
                    endpoint_inventory_sha256=identity,
                ),
            )
        )
        test_prefix = f"raw-score/test/test/terminal/{arm.arm_id}/"
        if arm.scorer_kind == "exact_autoregressive":
            record_count = sum(
                key.startswith(test_prefix)
                for key in endpoint_inventory.raw_score_record_keys
            )
            components.append(
                _component(
                    component_id=f"test/{arm.arm_id}/exact",
                    work_unit_kind="test_batch",
                    work_units=(
                        record_count
                        * workload.test_batches_per_full_evaluation
                    ),
                    operation_count=record_count,
                    uses_gpu=True,
                    warmup_count=0,
                    sample_count=10,
                    endpoint_inventory_sha256=identity,
                )
            )
        else:
            for particle_count in endpoint_inventory.particle_counts:
                marker = f"/particles={particle_count}"
                record_count = sum(
                    key.startswith(test_prefix) and marker in key
                    for key in endpoint_inventory.raw_score_record_keys
                )
                components.append(
                    _component(
                        component_id=(
                            f"test/{arm.arm_id}/particles={particle_count}"
                        ),
                        work_unit_kind="particle_test_batch",
                        work_units=(
                            record_count
                            * workload.test_batches_per_full_evaluation
                        ),
                        operation_count=record_count,
                        uses_gpu=True,
                        warmup_count=0,
                        sample_count=10,
                        endpoint_inventory_sha256=identity,
                    )
                )

    components.extend(
        (
            _component(
                component_id="checkpoint/wt103",
                work_unit_kind="durable_checkpoint",
                work_units=(
                    endpoint_inventory.terminal_checkpoint_count
                    * (2 * 20 + 1)
                ),
                operation_count=(
                    endpoint_inventory.terminal_checkpoint_count
                    * (2 * 20 + 1)
                ),
                uses_gpu=False,
                warmup_count=0,
                sample_count=1,
                endpoint_inventory_sha256=identity,
            ),
            _component(
                component_id="table/final",
                work_unit_kind="result_row",
                work_units=endpoint_inventory.result_row_count,
                operation_count=1,
                uses_gpu=False,
                warmup_count=0,
                sample_count=1,
                endpoint_inventory_sha256=identity,
            ),
        )
    )
    for figure_id in endpoint_inventory.figure_panel_keys:
        series_count = max(
            1,
            sum(
                key.startswith(f"{figure_id}/")
                for key in endpoint_inventory.figure_series_keys
            ),
        )
        components.append(
            _component(
                component_id=f"figure/{figure_id}",
                work_unit_kind="figure_series",
                work_units=series_count,
                operation_count=1,
                uses_gpu=False,
                warmup_count=0,
                sample_count=1,
                endpoint_inventory_sha256=identity,
            )
        )
    components.append(
        _component(
            component_id="review-export",
            work_unit_kind="review_export_operation",
            work_units=1,
            operation_count=1,
            uses_gpu=False,
            warmup_count=0,
            sample_count=1,
            endpoint_inventory_sha256=identity,
        )
    )
    return tuple(components)


@dataclass(frozen=True, slots=True)
class ComponentBenchmark:
    """Worst-case post-warmup measurement for one exact component."""

    schema_version: Literal["wt103-component-benchmark-v1"]
    component_id: str
    component_spec_sha256: str
    execution_identity: TrainingExecutionIdentity
    warmup_count: int
    sample_count: int
    minimum_throughput_per_second: float
    maximum_duration_seconds: float
    maximum_board_power_watts: float | None
    power_provider_identity_sha256: str | None
    benchmark_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in tuple(self.__dataclass_fields__)[:-1]
        }

    def __post_init__(self) -> None:
        if self.schema_version != "wt103-component-benchmark-v1":
            raise ValueError("component benchmark schema is invalid")
        _text(self.component_id, "component_id")
        _sha256(self.component_spec_sha256, "component_spec_sha256")
        if type(self.execution_identity) is not TrainingExecutionIdentity:
            raise ValueError("benchmark execution identity must be exact")
        self.execution_identity.__post_init__()
        _exact_int(self.warmup_count, "warmup_count")
        _exact_int(self.sample_count, "sample_count", minimum=1)
        _finite_float(
            self.minimum_throughput_per_second,
            "minimum_throughput_per_second",
            minimum=0.000000001,
        )
        _finite_float(
            self.maximum_duration_seconds,
            "maximum_duration_seconds",
        )
        if self.maximum_board_power_watts is not None:
            _finite_float(
                self.maximum_board_power_watts,
                "maximum_board_power_watts",
            )
        if self.power_provider_identity_sha256 is not None:
            _sha256(
                self.power_provider_identity_sha256,
                "power_provider_identity_sha256",
            )
        expected = owned_sha256(
            "vfe4.wt103.component-benchmark.v1",
            self.semantic_payload(),
        )
        _sha256(self.benchmark_sha256, "benchmark_sha256")
        if self.benchmark_sha256 != expected:
            raise ValueError("component benchmark hash does not match")

    @classmethod
    def observed_for(
        cls,
        component: ResourceComponentSpec,
        *,
        execution_identity: TrainingExecutionIdentity,
        minimum_throughput_per_second: float,
        maximum_duration_seconds: float,
        maximum_board_power_watts: float | None,
        power_provider: PowerProviderIdentity | None,
    ) -> "ComponentBenchmark":
        if type(component) is not ResourceComponentSpec:
            raise ValueError("component must be exact")
        if type(execution_identity) is not TrainingExecutionIdentity:
            raise ValueError("execution_identity must be exact")
        component.__post_init__()
        execution_identity.__post_init__()
        if component.uses_gpu:
            if power_provider is None:
                if maximum_board_power_watts is not None:
                    raise ValueError(
                        "missing GPU power provider requires missing power"
                    )
                provider_sha = None
            else:
                if type(power_provider) is not PowerProviderIdentity:
                    raise ValueError(
                        "power_provider must be exact"
                    )
                power_provider.__post_init__()
                if maximum_board_power_watts is None:
                    raise ValueError(
                        "GPU power provider requires observed power"
                    )
                provider_sha = power_provider.identity_sha256
        else:
            if power_provider is not None or maximum_board_power_watts != 0.0:
                raise ValueError("CPU-only benchmark cannot claim board power")
            provider_sha = None
        payload = {
            "schema_version": "wt103-component-benchmark-v1",
            "component_id": component.component_id,
            "component_spec_sha256": component.spec_sha256,
            "execution_identity": execution_identity,
            "warmup_count": component.warmup_count,
            "sample_count": component.sample_count,
            "minimum_throughput_per_second": (
                minimum_throughput_per_second
            ),
            "maximum_duration_seconds": maximum_duration_seconds,
            "maximum_board_power_watts": maximum_board_power_watts,
            "power_provider_identity_sha256": provider_sha,
        }
        return cls(
            **payload,
            benchmark_sha256=owned_sha256(
                "vfe4.wt103.component-benchmark.v1",
                payload,
            ),
        )


@dataclass(frozen=True, slots=True)
class DiskByteForecast:
    """Exact byte inventory including the 25% temporary-write overhead."""

    archive_staging_bytes: int
    extracted_member_bytes: int
    int32_token_cache_bytes: int
    schedule_bytes: int
    retained_checkpoint_bytes: int
    jsonl_csv_bytes: int
    test_record_bytes: int
    figure_bytes: int
    payload_bytes: int
    temporary_write_overhead_bytes: int
    forecast_bytes: int
    required_available_bytes: int
    forecast_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in tuple(self.__dataclass_fields__)[:-1]
        }

    def __post_init__(self) -> None:
        categories = tuple(self.__dataclass_fields__)[:8]
        for name in categories:
            _exact_int(getattr(self, name), name)
        expected_payload = sum(getattr(self, name) for name in categories)
        expected_overhead = (expected_payload + 3) // 4
        expected_forecast = expected_payload + expected_overhead
        expected_available = 2 * expected_forecast + 10 * _GIB
        if (
            self.payload_bytes != expected_payload
            or self.temporary_write_overhead_bytes != expected_overhead
            or self.forecast_bytes != expected_forecast
            or self.required_available_bytes != expected_available
        ):
            raise ValueError("disk forecast fields are not exactly derived")
        expected = owned_sha256(
            "vfe4.wt103.disk-byte-forecast.v1",
            self.semantic_payload(),
        )
        _sha256(self.forecast_sha256, "forecast_sha256")
        if self.forecast_sha256 != expected:
            raise ValueError("disk forecast hash does not match")

    @classmethod
    def create(cls, **categories: int) -> "DiskByteForecast":
        expected_names = tuple(cls.__dataclass_fields__)[:8]
        if tuple(categories) != expected_names:
            raise ValueError("disk byte categories must retain exact order")
        payload_bytes = sum(categories.values())
        overhead = (payload_bytes + 3) // 4
        forecast = payload_bytes + overhead
        payload = {
            **categories,
            "payload_bytes": payload_bytes,
            "temporary_write_overhead_bytes": overhead,
            "forecast_bytes": forecast,
            "required_available_bytes": 2 * forecast + 10 * _GIB,
        }
        return cls(
            **payload,
            forecast_sha256=owned_sha256(
                "vfe4.wt103.disk-byte-forecast.v1",
                payload,
            ),
        )


@dataclass(frozen=True, slots=True)
class ComponentForecast:
    component_id: str
    component_spec_sha256: str
    benchmark_sha256: str
    predicted_seconds: float
    predicted_gpu_seconds: float

    def __post_init__(self) -> None:
        _text(self.component_id, "component_id")
        _sha256(self.component_spec_sha256, "component_spec_sha256")
        _sha256(self.benchmark_sha256, "benchmark_sha256")
        _finite_float(self.predicted_seconds, "predicted_seconds")
        _finite_float(
            self.predicted_gpu_seconds,
            "predicted_gpu_seconds",
        )
        if self.predicted_gpu_seconds > self.predicted_seconds:
            raise ValueError("GPU seconds cannot exceed component wall seconds")


@dataclass(frozen=True, slots=True)
class ResourceForecast:
    """Complete disk/time/device-hour/energy forecast under frozen ceilings."""

    schema_version: Literal["wt103-resource-forecast-v1"]
    endpoint_inventory_sha256: str
    execution_identity: TrainingExecutionIdentity
    component_forecasts: tuple[ComponentForecast, ...]
    component_benchmark_sha256s: tuple[str, ...]
    disk_forecast_sha256: str
    available_disk_bytes: int
    raw_gpu_hours: float
    raw_wall_hours: float
    raw_energy_kwh: float | None
    forecast_gpu_hours: float
    forecast_wall_hours: float
    forecast_energy_kwh: float | None
    maximum_gpu_hours: Literal[720.0]
    maximum_wall_hours: Literal[840.0]
    maximum_energy_kwh: Literal[500.0]
    forecast_headroom_factor: Literal[1.25]
    power_provider_identity_sha256: str | None
    maximum_observed_board_power_watts: float | None
    reported_power_limit_watts: float | None
    conservative_power_watts: float | None
    status: GateStatus
    obligations: tuple[str, ...]
    forecast_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in tuple(self.__dataclass_fields__)[:-1]
        }

    def __post_init__(self) -> None:
        if (
            self.schema_version != "wt103-resource-forecast-v1"
            or self.maximum_gpu_hours != 720.0
            or self.maximum_wall_hours != 840.0
            or self.maximum_energy_kwh != 500.0
            or self.forecast_headroom_factor != 1.25
            or type(self.status) is not GateStatus
            or type(self.obligations) is not tuple
        ):
            raise ValueError("resource forecast schema/ceilings are invalid")
        _sha256(
            self.endpoint_inventory_sha256,
            "endpoint_inventory_sha256",
        )
        if type(self.execution_identity) is not TrainingExecutionIdentity:
            raise ValueError("resource execution identity must be exact")
        self.execution_identity.__post_init__()
        _sha256(self.disk_forecast_sha256, "disk_forecast_sha256")
        _exact_int(self.available_disk_bytes, "available_disk_bytes")
        if (
            type(self.component_forecasts) is not tuple
            or not self.component_forecasts
            or any(
                type(item) is not ComponentForecast
                for item in self.component_forecasts
            )
            or type(self.component_benchmark_sha256s) is not tuple
            or len(self.component_benchmark_sha256s)
            != len(self.component_forecasts)
        ):
            raise ValueError("resource component forecast inventory is invalid")
        for item in self.component_forecasts:
            item.__post_init__()
        for value in self.component_benchmark_sha256s:
            _sha256(value, "component_benchmark_sha256")
        for name in (
            "raw_gpu_hours",
            "raw_wall_hours",
            "forecast_gpu_hours",
            "forecast_wall_hours",
        ):
            _finite_float(getattr(self, name), name)
        if self.raw_energy_kwh is None:
            if (
                self.forecast_energy_kwh is not None
                or self.power_provider_identity_sha256 is not None
                or self.maximum_observed_board_power_watts is not None
                or self.reported_power_limit_watts is not None
                or self.conservative_power_watts is not None
            ):
                raise ValueError(
                    "missing power must leave energy and power authority open"
                )
        else:
            _finite_float(self.raw_energy_kwh, "raw_energy_kwh")
            if (
                self.forecast_energy_kwh is None
                or type(self.maximum_observed_board_power_watts) is not float
                or type(self.reported_power_limit_watts) is not float
                or type(self.conservative_power_watts) is not float
            ):
                raise ValueError("forecast energy is missing")
            _finite_float(
                self.forecast_energy_kwh,
                "forecast_energy_kwh",
            )
            _finite_float(
                self.maximum_observed_board_power_watts,
                "maximum_observed_board_power_watts",
            )
            _finite_float(
                self.reported_power_limit_watts,
                "reported_power_limit_watts",
            )
            _finite_float(
                self.conservative_power_watts,
                "conservative_power_watts",
            )
            _sha256(
                self.power_provider_identity_sha256,
                "power_provider_identity_sha256",
            )
            if (
                self.raw_gpu_hours <= 0.0
                or self.reported_power_limit_watts <= 0.0
                or self.conservative_power_watts
                != max(
                    self.maximum_observed_board_power_watts,
                    self.reported_power_limit_watts,
                )
                or self.raw_energy_kwh
                != self.raw_gpu_hours
                * self.conservative_power_watts
                / 1000.0
            ):
                raise ValueError(
                    "conservative power authority or energy does not match"
                )
        if (
            self.forecast_gpu_hours
            != self.raw_gpu_hours * self.forecast_headroom_factor
            or self.forecast_wall_hours
            != self.raw_wall_hours * self.forecast_headroom_factor
            or (
                self.raw_energy_kwh is not None
                and self.forecast_energy_kwh
                != self.raw_energy_kwh * self.forecast_headroom_factor
            )
        ):
            raise ValueError("resource headroom arithmetic does not match")
        if (
            (self.status is GateStatus.PASS and self.obligations)
            or (self.status is not GateStatus.PASS and not self.obligations)
        ):
            raise ValueError("resource forecast status/obligations disagree")
        expected = owned_sha256(
            "vfe4.wt103.resource-forecast.v1",
            self.semantic_payload(),
        )
        _sha256(self.forecast_sha256, "forecast_sha256")
        if self.forecast_sha256 != expected:
            raise ValueError("resource forecast hash does not match")


def forecast_resources(
    *,
    endpoint_inventory: EndpointInventory,
    workload: ResourceWorkload,
    component_benchmarks: tuple[ComponentBenchmark, ...],
    execution_identity: TrainingExecutionIdentity,
    disk_forecast: DiskByteForecast,
    available_disk_bytes: int,
    resource_profile: ResourceProfile,
    power_provider: PowerProviderIdentity | None,
) -> ResourceForecast:
    """Apply exact inventory work and conservative extrema to frozen ceilings."""

    if type(resource_profile) is not ResourceProfile:
        raise ValueError("resource_profile must be exact")
    if type(execution_identity) is not TrainingExecutionIdentity:
        raise ValueError("execution_identity must be exact")
    resource_profile.__post_init__()
    execution_identity.__post_init__()
    if type(disk_forecast) is not DiskByteForecast:
        raise ValueError("disk_forecast must be exact")
    disk_forecast.__post_init__()
    _exact_int(available_disk_bytes, "available_disk_bytes")
    specs = required_resource_components(endpoint_inventory, workload)
    if (
        type(component_benchmarks) is not tuple
        or tuple(item.component_id for item in component_benchmarks)
        != tuple(item.component_id for item in specs)
        or any(
            type(item) is not ComponentBenchmark
            for item in component_benchmarks
        )
    ):
        raise ValueError("component benchmark inventory is not exact")
    forecasts: list[ComponentForecast] = []
    for spec, benchmark in zip(specs, component_benchmarks, strict=True):
        benchmark.__post_init__()
        if (
            benchmark.execution_identity != execution_identity
            or benchmark.component_spec_sha256 != spec.spec_sha256
            or benchmark.warmup_count != spec.warmup_count
            or benchmark.sample_count != spec.sample_count
        ):
            raise ValueError(
                "component benchmark does not bind execution identity/spec:"
                f"{spec.component_id}"
            )
        duration = max(
            spec.work_units / benchmark.minimum_throughput_per_second,
            spec.operation_count * benchmark.maximum_duration_seconds,
        )
        forecasts.append(
            ComponentForecast(
                component_id=spec.component_id,
                component_spec_sha256=spec.spec_sha256,
                benchmark_sha256=benchmark.benchmark_sha256,
                predicted_seconds=duration,
                predicted_gpu_seconds=duration if spec.uses_gpu else 0.0,
            )
        )
    raw_wall_hours = sum(
        item.predicted_seconds for item in forecasts
    ) / 3600.0
    raw_gpu_hours = sum(
        item.predicted_gpu_seconds for item in forecasts
    ) / 3600.0
    failures: list[str] = []
    inconclusive: list[str] = []
    if available_disk_bytes < disk_forecast.required_available_bytes:
        failures.append("disk_headroom_insufficient")
    if power_provider is None:
        for spec, benchmark in zip(
            specs,
            component_benchmarks,
            strict=True,
        ):
            if spec.uses_gpu and (
                benchmark.power_provider_identity_sha256 is not None
                or benchmark.maximum_board_power_watts is not None
            ):
                raise ValueError(
                    f"component power evidence has no provider:"
                    f"{spec.component_id}"
                )
        inconclusive.append("power_provider_missing")
        raw_energy: float | None = None
        forecast_energy: float | None = None
        power_sha: str | None = None
        measured_max: float | None = None
        reported_limit: float | None = None
        conservative_watts: float | None = None
    else:
        if type(power_provider) is not PowerProviderIdentity:
            raise ValueError("power_provider must be exact")
        power_provider.__post_init__()
        if power_provider.sample_interval_ms != resource_profile.power_sample_interval_ms:
            raise ValueError("power provider sample interval differs from profile")
        for spec, benchmark in zip(specs, component_benchmarks, strict=True):
            expected = (
                power_provider.identity_sha256 if spec.uses_gpu else None
            )
            if benchmark.power_provider_identity_sha256 != expected:
                raise ValueError(
                    f"component power identity mismatch:{spec.component_id}"
                )
        measured_max = max(
            item.maximum_board_power_watts
            for item in component_benchmarks
            if item.maximum_board_power_watts is not None
        )
        conservative_watts = max(
            measured_max,
            power_provider.reported_power_limit_watts,
        )
        raw_energy = raw_gpu_hours * conservative_watts / 1000.0
        forecast_energy = (
            raw_energy * resource_profile.forecast_headroom_factor
        )
        power_sha = power_provider.identity_sha256
        reported_limit = power_provider.reported_power_limit_watts
    forecast_gpu = (
        raw_gpu_hours * resource_profile.forecast_headroom_factor
    )
    forecast_wall = (
        raw_wall_hours * resource_profile.forecast_headroom_factor
    )
    if forecast_gpu > resource_profile.maximum_gpu_hours:
        failures.append("gpu_hour_ceiling_exceeded")
    if forecast_wall > resource_profile.maximum_wall_hours:
        failures.append("wall_hour_ceiling_exceeded")
    if (
        forecast_energy is not None
        and forecast_energy > resource_profile.maximum_energy_kwh
    ):
        failures.append("energy_ceiling_exceeded")
    obligations = tuple(dict.fromkeys((*failures, *inconclusive)))
    status = (
        GateStatus.FAIL
        if failures
        else GateStatus.INCONCLUSIVE
        if inconclusive
        else GateStatus.PASS
    )
    payload = {
        "schema_version": "wt103-resource-forecast-v1",
        "endpoint_inventory_sha256": (
            endpoint_inventory.endpoint_inventory_sha256
        ),
        "execution_identity": execution_identity,
        "component_forecasts": tuple(forecasts),
        "component_benchmark_sha256s": tuple(
            item.benchmark_sha256 for item in component_benchmarks
        ),
        "disk_forecast_sha256": disk_forecast.forecast_sha256,
        "available_disk_bytes": available_disk_bytes,
        "raw_gpu_hours": raw_gpu_hours,
        "raw_wall_hours": raw_wall_hours,
        "raw_energy_kwh": raw_energy,
        "forecast_gpu_hours": forecast_gpu,
        "forecast_wall_hours": forecast_wall,
        "forecast_energy_kwh": forecast_energy,
        "maximum_gpu_hours": resource_profile.maximum_gpu_hours,
        "maximum_wall_hours": resource_profile.maximum_wall_hours,
        "maximum_energy_kwh": resource_profile.maximum_energy_kwh,
        "forecast_headroom_factor": (
            resource_profile.forecast_headroom_factor
        ),
        "power_provider_identity_sha256": power_sha,
        "maximum_observed_board_power_watts": measured_max,
        "reported_power_limit_watts": reported_limit,
        "conservative_power_watts": conservative_watts,
        "status": status,
        "obligations": obligations,
    }
    return ResourceForecast(
        **payload,
        forecast_sha256=owned_sha256(
            "vfe4.wt103.resource-forecast.v1",
            payload,
        ),
    )


@dataclass(frozen=True, slots=True)
class ResourceUsageEvent:
    schema_version: Literal["wt103-resource-usage-event-v1"]
    attempt_id: str
    segment_ordinal: int
    device_seconds: float
    wall_seconds: float
    sampled_energy_kwh: float
    usage_evidence_sha256: str
    event_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in tuple(self.__dataclass_fields__)[:-1]
        }

    def __post_init__(self) -> None:
        if self.schema_version != "wt103-resource-usage-event-v1":
            raise ValueError("resource usage event schema is invalid")
        _text(self.attempt_id, "attempt_id")
        _exact_int(self.segment_ordinal, "segment_ordinal", minimum=0)
        for name in (
            "device_seconds",
            "wall_seconds",
            "sampled_energy_kwh",
        ):
            _finite_float(getattr(self, name), name)
        if self.device_seconds > self.wall_seconds:
            raise ValueError("device seconds cannot exceed wall seconds")
        _sha256(
            self.usage_evidence_sha256,
            "usage_evidence_sha256",
        )
        expected = owned_sha256(
            "vfe4.wt103.resource-usage-event.v1",
            self.semantic_payload(),
        )
        _sha256(self.event_sha256, "event_sha256")
        if self.event_sha256 != expected:
            raise ValueError("resource usage event hash does not match")

    @classmethod
    def create(
        cls,
        *,
        attempt_id: str,
        segment_ordinal: int,
        device_seconds: float,
        wall_seconds: float,
        sampled_energy_kwh: float,
        usage_evidence_sha256: str,
    ) -> "ResourceUsageEvent":
        payload = {
            "schema_version": "wt103-resource-usage-event-v1",
            "attempt_id": attempt_id,
            "segment_ordinal": segment_ordinal,
            "device_seconds": device_seconds,
            "wall_seconds": wall_seconds,
            "sampled_energy_kwh": sampled_energy_kwh,
            "usage_evidence_sha256": usage_evidence_sha256,
        }
        return cls(
            **payload,
            event_sha256=owned_sha256(
                "vfe4.wt103.resource-usage-event.v1",
                payload,
            ),
        )


@dataclass(frozen=True, slots=True)
class ResourceUsageLedger:
    """Immutable append-by-replacement ledger bound to one experiment plan."""

    schema_version: Literal["wt103-resource-usage-ledger-v1"]
    experiment_plan_sha256: str
    maximum_gpu_hours: Literal[720.0]
    maximum_wall_hours: Literal[840.0]
    maximum_energy_kwh: Literal[500.0]
    forecast_headroom_factor: Literal[1.25]
    events: tuple[ResourceUsageEvent, ...]
    ledger_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in tuple(self.__dataclass_fields__)[:-1]
        }

    def __post_init__(self) -> None:
        if (
            self.schema_version != "wt103-resource-usage-ledger-v1"
            or self.maximum_gpu_hours != 720.0
            or self.maximum_wall_hours != 840.0
            or self.maximum_energy_kwh != 500.0
            or self.forecast_headroom_factor != 1.25
            or type(self.events) is not tuple
            or any(type(item) is not ResourceUsageEvent for item in self.events)
        ):
            raise ValueError("resource usage ledger schema is invalid")
        _sha256(
            self.experiment_plan_sha256,
            "experiment_plan_sha256",
        )
        for item in self.events:
            item.__post_init__()
        if len({item.event_sha256 for item in self.events}) != len(self.events):
            raise ValueError("resource usage events must be unique")
        seen_attempts: set[str] = set()
        active_attempt: str | None = None
        next_segment = 0
        for item in self.events:
            if item.attempt_id != active_attempt:
                if item.attempt_id in seen_attempts:
                    raise ValueError(
                        "resource usage segments must be contiguous"
                    )
                seen_attempts.add(item.attempt_id)
                active_attempt = item.attempt_id
                next_segment = 0
            if item.segment_ordinal != next_segment:
                raise ValueError(
                    "resource usage segment ordinal is discontinuous"
                )
            next_segment += 1
        expected = owned_sha256(
            "vfe4.wt103.resource-usage-ledger.v1",
            self.semantic_payload(),
        )
        _sha256(self.ledger_sha256, "ledger_sha256")
        if self.ledger_sha256 != expected:
            raise ValueError("resource usage ledger hash does not match")

    @classmethod
    def create(
        cls,
        *,
        experiment_plan_sha256: str,
        resource_profile: ResourceProfile,
    ) -> "ResourceUsageLedger":
        if type(resource_profile) is not ResourceProfile:
            raise ValueError("resource_profile must be exact")
        resource_profile.__post_init__()
        payload = {
            "schema_version": "wt103-resource-usage-ledger-v1",
            "experiment_plan_sha256": experiment_plan_sha256,
            "maximum_gpu_hours": resource_profile.maximum_gpu_hours,
            "maximum_wall_hours": resource_profile.maximum_wall_hours,
            "maximum_energy_kwh": resource_profile.maximum_energy_kwh,
            "forecast_headroom_factor": (
                resource_profile.forecast_headroom_factor
            ),
            "events": (),
        }
        return cls(
            **payload,
            ledger_sha256=owned_sha256(
                "vfe4.wt103.resource-usage-ledger.v1",
                payload,
            ),
        )

    def append(self, event: ResourceUsageEvent) -> "ResourceUsageLedger":
        if type(event) is not ResourceUsageEvent:
            raise ValueError("event must be exact ResourceUsageEvent")
        event.__post_init__()
        if event.event_sha256 in {item.event_sha256 for item in self.events}:
            raise ValueError("resource usage event is already present")
        payload = {
            "schema_version": self.schema_version,
            "experiment_plan_sha256": self.experiment_plan_sha256,
            "maximum_gpu_hours": self.maximum_gpu_hours,
            "maximum_wall_hours": self.maximum_wall_hours,
            "maximum_energy_kwh": self.maximum_energy_kwh,
            "forecast_headroom_factor": self.forecast_headroom_factor,
            "events": (*self.events, event),
        }
        return ResourceUsageLedger(
            **payload,
            ledger_sha256=owned_sha256(
                "vfe4.wt103.resource-usage-ledger.v1",
                payload,
            ),
        )

    @property
    def used_gpu_hours(self) -> float:
        return sum(item.device_seconds for item in self.events) / 3600.0

    @property
    def used_wall_hours(self) -> float:
        return sum(item.wall_seconds for item in self.events) / 3600.0

    @property
    def used_energy_kwh(self) -> float:
        return sum(item.sampled_energy_kwh for item in self.events)

    @property
    def remaining_gpu_hours(self) -> float:
        return self.maximum_gpu_hours - self.used_gpu_hours

    @property
    def remaining_wall_hours(self) -> float:
        return self.maximum_wall_hours - self.used_wall_hours

    @property
    def remaining_energy_kwh(self) -> float:
        return self.maximum_energy_kwh - self.used_energy_kwh


@dataclass(frozen=True, slots=True)
class TestReservationPreflight:
    schema_version: Literal["wt103-test-reservation-preflight-v1"]
    resource_usage_ledger_sha256: str
    required_gpu_hours: float
    required_wall_hours: float
    required_energy_kwh: float
    required_disk_bytes: float
    remaining_gpu_hours: float
    remaining_wall_hours: float
    remaining_energy_kwh: float
    available_disk_bytes: int
    status: GateStatus
    obligations: tuple[str, ...]
    record_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in tuple(self.__dataclass_fields__)[:-1]
        }

    def __post_init__(self) -> None:
        if (
            self.schema_version
            != "wt103-test-reservation-preflight-v1"
            or type(self.status) is not GateStatus
            or type(self.obligations) is not tuple
        ):
            raise ValueError("test reservation preflight schema is invalid")
        _sha256(
            self.resource_usage_ledger_sha256,
            "resource_usage_ledger_sha256",
        )
        for name in (
            "required_gpu_hours",
            "required_wall_hours",
            "required_energy_kwh",
            "required_disk_bytes",
        ):
            _finite_float(getattr(self, name), name)
        for name in (
            "remaining_gpu_hours",
            "remaining_wall_hours",
            "remaining_energy_kwh",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        _exact_int(self.available_disk_bytes, "available_disk_bytes")
        if (
            (self.status is GateStatus.PASS and self.obligations)
            or (self.status is not GateStatus.PASS and not self.obligations)
        ):
            raise ValueError("test reservation status/obligations disagree")
        expected = owned_sha256(
            "vfe4.wt103.test-reservation-preflight.v1",
            self.semantic_payload(),
        )
        _sha256(self.record_sha256, "record_sha256")
        if self.record_sha256 != expected:
            raise ValueError("test reservation preflight hash does not match")


def authorize_test_reservation(
    *,
    ledger: ResourceUsageLedger,
    raw_test_gpu_hours: float,
    raw_test_wall_hours: float,
    raw_test_energy_kwh: float,
    raw_test_disk_bytes: int,
    available_disk_bytes: int,
) -> TestReservationPreflight:
    """Recompute the irreversible test transaction against remaining budget."""

    if type(ledger) is not ResourceUsageLedger:
        raise ValueError("ledger must be exact")
    ledger.__post_init__()
    for name, value in (
        ("raw_test_gpu_hours", raw_test_gpu_hours),
        ("raw_test_wall_hours", raw_test_wall_hours),
        ("raw_test_energy_kwh", raw_test_energy_kwh),
    ):
        _finite_float(value, name)
    _exact_int(raw_test_disk_bytes, "raw_test_disk_bytes")
    _exact_int(available_disk_bytes, "available_disk_bytes")
    factor = ledger.forecast_headroom_factor
    required_gpu = raw_test_gpu_hours * factor
    required_wall = raw_test_wall_hours * factor
    required_energy = raw_test_energy_kwh * factor
    required_disk = raw_test_disk_bytes * factor
    obligations: list[str] = []
    if ledger.remaining_gpu_hours < required_gpu:
        obligations.append("remaining_gpu_hours_insufficient")
    if ledger.remaining_wall_hours < required_wall:
        obligations.append("remaining_wall_hours_insufficient")
    if ledger.remaining_energy_kwh < required_energy:
        obligations.append("remaining_energy_kwh_insufficient")
    if available_disk_bytes < required_disk:
        obligations.append("remaining_disk_bytes_insufficient")
    payload = {
        "schema_version": "wt103-test-reservation-preflight-v1",
        "resource_usage_ledger_sha256": ledger.ledger_sha256,
        "required_gpu_hours": required_gpu,
        "required_wall_hours": required_wall,
        "required_energy_kwh": required_energy,
        "required_disk_bytes": required_disk,
        "remaining_gpu_hours": ledger.remaining_gpu_hours,
        "remaining_wall_hours": ledger.remaining_wall_hours,
        "remaining_energy_kwh": ledger.remaining_energy_kwh,
        "available_disk_bytes": available_disk_bytes,
        "status": GateStatus.PASS if not obligations else GateStatus.FAIL,
        "obligations": tuple(obligations),
    }
    return TestReservationPreflight(
        **payload,
        record_sha256=owned_sha256(
            "vfe4.wt103.test-reservation-preflight.v1",
            payload,
        ),
    )


__all__ = [
    "AllocationObservation",
    "AllocationPreflightRecord",
    "ComponentBenchmark",
    "ComponentForecast",
    "DependencyLockIdentity",
    "DiskByteForecast",
    "DistributionIdentity",
    "EnvironmentObservation",
    "EnvironmentRecord",
    "LockInputManifest",
    "LockRequirement",
    "PowerProviderIdentity",
    "ResourceComponentSpec",
    "ResourceForecast",
    "ResourceUsageEvent",
    "ResourceUsageLedger",
    "ResourceWorkload",
    "TestReservationPreflight",
    "TrainingExecutionIdentity",
    "authorize_test_reservation",
    "capture_environment",
    "forecast_resources",
    "required_resource_components",
    "parse_lock_input_manifest",
    "render_dependency_lock",
    "run_allocation_preflight",
]
