"""Fail-closed assembly boundary for the H7 frame-covariance gate.

Numerical H7 work remains in the production/reference modules that own it.
This module validates immutable evidence, current-candidate predecessors, and
the exact source dependency closure before assembling the sole public result.
It never upgrades missing runtime evidence to a PASS.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Literal, cast

from vfe4.artifacts.provenance import source_candidate_sha256
from vfe4.types.h6 import (
    H1_PREFIX_PRIOR_V2_GENERATIVE_FACTOR_SCHEMA_SHA256,
)
from vfe4.types.h7 import (
    H7_CONTROL_IDS,
    H7_MATRIX_TRIAL_IDS,
    H7_REQUIRED_TRIAL_IDS,
    H7_SCALAR_TRIAL_IDS,
    H7ControlResult,
    H7FailOutcome,
    H7GateEvaluation,
    H7InconclusiveOutcome,
    H7PassOutcome,
    H7PredecessorReference,
    H7TrialResult,
    canonical_h7_bytes,
    h7_owned_sha256,
)
from vfe4.types.results import (
    GateStatus,
    H7GateResult,
    _h7_expected_negative_state,
)


H7_VERIFICATION_PREFIX = ("H1", "H2", "H3", "H4", "H5", "H6-Prefix", "H7")
H7_PREDECESSOR_KEYS = H7GateResult.predecessor_keys
H7_PREDECESSOR_CLAIM_IDS: Mapping[str, str] = MappingProxyType(
    {
        "h1_h5": "h7-predecessor-h1-h5-closure",
        "h1_prefix_prior": "h7-predecessor-h1-prefix-prior-closure",
        "h6_prefix": "h7-predecessor-h6-prefix-closure",
    }
)
H7_ACTIVE_SCORER_PROFILE = "h7-linear-history-source-v1"
H7_VALIDATION_SCHEMA = "h7-frame-covariance-validation-v1"
H7_SOURCE_ONLY_OBLIGATIONS = (
    "Task-6 scalar density-probe table hashes are unmeasured",
    "Task-6 precision-operand table hashes are unmeasured",
    "Task-6 required SPD condition extrema are unmeasured",
    "Task-6 GH41/GH51 deltas are unmeasured",
    "Task-6 independent oracle inventory hash is unmeasured",
    "Task-7 focused runtime evidence has not been executed",
)
H7_NONCLAIMS = (
    "optimizer_equivariance",
    "gradient_flow_equivariance",
    "training_benefit",
    "predictive_benefit",
    "h8_scaling",
    "orientation_reversing_gl2",
    "base_curvature_or_holonomy",
)
H7_FROZEN_SOURCE_FIXTURE_HASHES: Mapping[str, str] = MappingProxyType(
    {
        "h1_fixture_raw_sha256": (
            "388e38cc8c16d8b5e2c61919c1e712a134d88fb0bbd8ec1f2939b9859c9a583b"
        ),
        "h7_fixture_raw_sha256": (
            "d2ed126c3deab3eafc7b94f81f13152be63eb854e3e62e03f1494dea163666d4"
        ),
        "density_probe_table_raw_sha256": (
            "4857af296e84a33f47964c3bca65e0d42967009aa5c79a52bcc98d6db04382c6"
        ),
        "density_probe_set_sha256": (
            "f002618a32270846c83fedf9888bc06a01d755019edc6421526aee33f89fb42f"
        ),
    }
)
H7_REQUIRED_DEPENDENCY_PATHS = (
    "docs/preregistrations/2026-07-21-h7-frame-covariance.md",
    "verification/h7_budget.py",
    "verification/h7_gate.py",
    "verification/mp_oracles/h7_budget_protocol.py",
    "verification/mp_oracles/h7_covariance.py",
    "vfe4/generative/pushforward.py",
    "vfe4/geometry/group_action.py",
    "vfe4/objective/h7_covariance.py",
    "vfe4/recognition/pushforward.py",
    "vfe4/types/__init__.py",
    "vfe4/types/h7.py",
    "vfe4/types/results.py",
    "vfe4/validation/fixtures/h1_v1.json",
    "vfe4/validation/fixtures/h7_density_probes_v1.json",
    "vfe4/validation/fixtures/h7_v1.json",
    "vfe4/validation/h7_fixture.py",
)
H7_CAPTURED_FIXTURE_PATHS = (
    "vfe4/validation/fixtures/h1_v1.json",
    "vfe4/validation/fixtures/h7_v1.json",
)
H7_H1_PREFIX_PRIOR_V2_FIXTURE_SHA256 = (
    "6b0e855482b8f335bec73e4b0976a1317d7ce4cf3ff050670b3950e271c57fde"
)
H7_H1_PREFIX_PRIOR_BASE_FIXTURE_SHA256 = (
    "388e38cc8c16d8b5e2c61919c1e712a134d88fb0bbd8ec1f2939b9859c9a583b"
)

_LOWER_HEX = frozenset("0123456789abcdef")


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase 64-hex SHA-256")
    return value


def _require_git_head(value: object) -> str:
    if (
        type(value) is not str
        or len(value) not in (40, 64)
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise ValueError("git_head must be a full lowercase Git object ID")
    return value


def _candidate_path(
    value: object,
    *,
    repo_root: Path,
    name: str,
    require_inside_repo: bool,
) -> Path:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty path string")
    raw = Path(value)
    if not raw.is_absolute() and any(part in ("", ".", "..") for part in raw.parts):
        raise ValueError(f"{name} relative path is not canonical")
    candidate = raw if raw.is_absolute() else repo_root / raw
    probe = candidate
    while True:
        if probe.is_symlink():
            raise ValueError(f"{name} cannot traverse a symlink")
        if probe == repo_root or probe.parent == probe:
            break
        probe = probe.parent
    resolved = candidate.resolve(strict=True)
    if require_inside_repo:
        try:
            resolved.relative_to(repo_root)
        except ValueError as error:
            raise ValueError(
                f"{name} must remain inside the candidate repository"
            ) from error
    return resolved


def h7_predecessor_closure_binding_sha256(
    key: str,
    *,
    repo_root: Path,
    artifact_path: str,
    git_head: str,
    dirty_digest: str,
    junit_path: str,
    junit_sha256: str,
    manifest_sha256: str,
    payload_hashes: Mapping[str, str],
) -> str:
    """Bind one predecessor's candidate, payload, and JUnit preimages."""

    if key not in H7_PREDECESSOR_KEYS:
        raise ValueError("predecessor closure key is outside the frozen inventory")
    if not isinstance(repo_root, Path):
        raise ValueError("repo_root must be a Path")
    root = repo_root.resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("repo_root must be a real directory")
    artifact_root = _candidate_path(
        artifact_path,
        repo_root=root,
        name="predecessor artifact",
        require_inside_repo=False,
    )
    junit_file = _candidate_path(
        junit_path,
        repo_root=root,
        name="candidate JUnit",
        require_inside_repo=False,
    )
    if not artifact_root.is_dir() or artifact_root.is_symlink():
        raise ValueError("predecessor artifact must be a real directory")
    if not junit_file.is_file() or junit_file.is_symlink():
        raise ValueError("candidate JUnit must be a regular file")
    _require_git_head(git_head)
    _require_sha256(dirty_digest, "dirty_digest")
    _require_sha256(junit_sha256, "junit_sha256")
    _require_sha256(manifest_sha256, "manifest_sha256")
    if (
        not isinstance(payload_hashes, Mapping)
        or not payload_hashes
        or any(type(name) is not str or not name for name in payload_hashes)
    ):
        raise ValueError("payload_hashes must be a nonempty string-keyed mapping")
    normalized_payload_hashes: dict[str, str] = {}
    for name, digest in sorted(payload_hashes.items()):
        normalized_payload_hashes[name] = _require_sha256(
            digest,
            f"payload_hashes[{name!r}]",
        )
    semantic = {
        "key": key,
        "artifact_path": artifact_root.as_posix(),
        "git_head": git_head,
        "dirty_digest": dirty_digest,
        "junit_path": junit_file.as_posix(),
        "junit_sha256": junit_sha256,
        "manifest_sha256": manifest_sha256,
        "payload_hashes": normalized_payload_hashes,
    }
    return h7_owned_sha256(
        "vfe4.h7.predecessor-closure-claim.v1",
        semantic,
    )


def h7_predecessor_closure_claim_contract(
    key: str,
    *,
    repo_root: Path,
    artifact_path: str,
    git_head: str,
    dirty_digest: str,
    junit_path: str,
    junit_sha256: str,
    manifest_sha256: str,
    payload_hashes: Mapping[str, str],
) -> dict[str, object]:
    """Return the exact noncircular ledger claim fields H7 consumes."""

    binding_sha256 = h7_predecessor_closure_binding_sha256(
        key,
        repo_root=repo_root,
        artifact_path=artifact_path,
        git_head=git_head,
        dirty_digest=dirty_digest,
        junit_path=junit_path,
        junit_sha256=junit_sha256,
        manifest_sha256=manifest_sha256,
        payload_hashes=payload_hashes,
    )
    root = repo_root.resolve(strict=True)
    artifact_root = _candidate_path(
        artifact_path,
        repo_root=root,
        name="predecessor artifact",
        require_inside_repo=False,
    )
    junit_file = _candidate_path(
        junit_path,
        repo_root=root,
        name="candidate JUnit",
        require_inside_repo=False,
    )
    claim_id = H7_PREDECESSOR_CLAIM_IDS[key]
    return {
        "id": claim_id,
        "domain": "code",
        "statement": (
            "H7 predecessor closure binding sha256:"
            f"{binding_sha256}"
        ),
        "evidence": (
            {
                "id": f"e-{claim_id}-artifact-manifest",
                "kind": "mechanical",
                "location": (artifact_root / "manifest.sha256").as_posix(),
            },
            {
                "id": f"e-{claim_id}-candidate-junit",
                "kind": "mechanical",
                "location": junit_file.as_posix(),
            },
        ),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _ordered_unique(values: tuple[str, ...], name: str) -> tuple[str, ...]:
    if (
        type(values) is not tuple
        or any(type(item) is not str or not item for item in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(f"{name} must be an ordered tuple of unique strings")
    return values


@dataclass(frozen=True)
class _VerificationGateApi:
    validate_ledger: Callable[[dict[str, object]], list[str]]
    capture_artifact_revision: Callable[..., str]


def _verification_ledger_validator_path() -> Path:
    path = (
        Path.home()
        / ".codex"
        / "skills"
        / "verification"
        / "scripts"
        / "verification_gate.py"
    ).resolve(strict=True)
    if not path.is_file() or path.is_symlink():
        raise ValueError("installed verification-ledger validator is unavailable")
    return path


def _load_verification_gate_api(
    *,
    expected_sha256: str,
) -> _VerificationGateApi:
    """Load the exact installed deterministic validator bound by source hash."""

    _require_sha256(expected_sha256, "ledger_validator_sha256")
    path = _verification_ledger_validator_path()
    before = path.read_bytes()
    if hashlib.sha256(before).hexdigest() != expected_sha256:
        raise ValueError("verification-ledger validator changed after source capture")
    spec = importlib.util.spec_from_file_location("_vfe4_h7_ledger_validator", path)
    if spec is None or spec.loader is None:
        raise ValueError("verification-ledger validator cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("verification-ledger validator changed while loading")
    validate_ledger = getattr(module, "validate_ledger", None)
    capture_revision = getattr(module, "capture_artifact_revision", None)
    if not callable(validate_ledger) or not callable(capture_revision):
        raise ValueError("verification-ledger validator API is incomplete")
    return _VerificationGateApi(
        validate_ledger=cast(
            Callable[[dict[str, object]], list[str]],
            validate_ledger,
        ),
        capture_artifact_revision=cast(Callable[..., str], capture_revision),
    )


@dataclass(frozen=True)
class H7DependencyClosure:
    file_sha256: Mapping[str, str]
    ledger_validator_sha256: str
    dependency_closure_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.file_sha256, Mapping):
            raise ValueError("file_sha256 must be a mapping")
        if tuple(self.file_sha256) != H7_REQUIRED_DEPENDENCY_PATHS:
            raise ValueError("H7 dependency inventory is not exact or ordered")
        owned = {
            path: _require_sha256(self.file_sha256[path], f"file_sha256[{path!r}]")
            for path in H7_REQUIRED_DEPENDENCY_PATHS
        }
        frozen = MappingProxyType(owned)
        object.__setattr__(self, "file_sha256", frozen)
        validator_sha256 = _require_sha256(
            self.ledger_validator_sha256,
            "ledger_validator_sha256",
        )
        expected = h7_owned_sha256(
            "vfe4.h7.source-dependency-closure.v1",
            {
                "file_sha256": frozen,
                "ledger_validator_sha256": validator_sha256,
            },
        )
        if self.dependency_closure_sha256 != expected:
            raise ValueError("dependency closure hash does not match its files")


def _require_captured_fixture_bytes(
    captured_fixture_bytes: Mapping[str, bytes],
) -> Mapping[str, bytes]:
    if (
        not isinstance(captured_fixture_bytes, Mapping)
        or tuple(captured_fixture_bytes) != H7_CAPTURED_FIXTURE_PATHS
        or any(
            type(value) is not bytes or not value
            for value in captured_fixture_bytes.values()
        )
    ):
        raise ValueError(
            "captured_fixture_bytes must contain exact ordered H1/H7 bytes"
        )
    return MappingProxyType(
        {
            relative: captured_fixture_bytes[relative]
            for relative in H7_CAPTURED_FIXTURE_PATHS
        }
    )


def capture_h7_dependency_closure(
    repo_root: Path,
    *,
    captured_fixture_bytes: Mapping[str, bytes],
) -> H7DependencyClosure:
    """Hash the exact H7 source/config/fixture dependency inventory."""

    if not isinstance(repo_root, Path):
        raise ValueError("repo_root must be a Path")
    captured = _require_captured_fixture_bytes(captured_fixture_bytes)
    root = repo_root.resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("repo_root must be a real directory")
    file_sha256: dict[str, str] = {}
    for relative in H7_REQUIRED_DEPENDENCY_PATHS:
        path = (root / Path(*PurePosixPath(relative).parts)).resolve(strict=True)
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("H7 dependency escaped the repository root") from error
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"H7 dependency is not a regular file: {relative}")
        content = (
            captured[relative]
            if relative in captured
            else path.read_bytes()
        )
        file_sha256[relative] = hashlib.sha256(content).hexdigest()
    frozen = MappingProxyType(file_sha256)
    ledger_validator_sha256 = hashlib.sha256(
        _verification_ledger_validator_path().read_bytes()
    ).hexdigest()
    return H7DependencyClosure(
        file_sha256=frozen,
        ledger_validator_sha256=ledger_validator_sha256,
        dependency_closure_sha256=h7_owned_sha256(
            "vfe4.h7.source-dependency-closure.v1",
            {
                "file_sha256": frozen,
                "ledger_validator_sha256": ledger_validator_sha256,
            },
        ),
    )


@dataclass(frozen=True)
class H7PredecessorValidation:
    references: Mapping[str, H7PredecessorReference]
    obligations: tuple[str, ...]
    registry_sha256: str

    @classmethod
    def create(
        cls,
        *,
        references: Mapping[str, H7PredecessorReference],
        obligations: tuple[str, ...],
    ) -> "H7PredecessorValidation":
        if not isinstance(references, Mapping) or set(references).difference(
            H7_PREDECESSOR_KEYS
        ):
            raise ValueError("predecessor references use a closed key inventory")
        owned = MappingProxyType(
            {key: references[key] for key in H7_PREDECESSOR_KEYS if key in references}
        )
        obligations = _ordered_unique(obligations, "predecessor obligations")
        semantic = {"references": owned, "obligations": obligations}
        return cls(
            references=owned,
            obligations=obligations,
            registry_sha256=h7_owned_sha256(
                "vfe4.h7.predecessor-registry.v1",
                semantic,
            ),
        )

    def __post_init__(self) -> None:
        if not isinstance(self.references, Mapping):
            raise ValueError("predecessor validation references must be a mapping")
        if tuple(self.references) != tuple(
            key for key in H7_PREDECESSOR_KEYS if key in self.references
        ):
            raise ValueError("predecessor validation order is not canonical")
        for value in self.references.values():
            if type(value) is not H7PredecessorReference:
                raise ValueError("predecessor validation requires exact references")
            value.__post_init__()
        _ordered_unique(self.obligations, "predecessor obligations")
        semantic = {
            "references": self.references,
            "obligations": self.obligations,
        }
        if self.registry_sha256 != h7_owned_sha256(
            "vfe4.h7.predecessor-registry.v1",
            semantic,
        ):
            raise ValueError("predecessor registry hash is stale")


def validate_h7_predecessor_registry(
    entries: tuple[tuple[str, H7PredecessorReference], ...],
    *,
    repo_root: Path,
    git_head: str,
    dirty_digest: str,
    junit_sha256: str,
    scorer_profile: str,
    ledger_validator_sha256: str,
) -> H7PredecessorValidation:
    """Validate current-candidate artifact, payload, and ledger references."""

    _require_git_head(git_head)
    _require_sha256(dirty_digest, "dirty_digest")
    _require_sha256(junit_sha256, "junit_sha256")
    _require_sha256(ledger_validator_sha256, "ledger_validator_sha256")
    if not isinstance(repo_root, Path):
        raise ValueError("repo_root must be a Path")
    root = repo_root.resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("repo_root must be a real directory")
    if scorer_profile != H7_ACTIVE_SCORER_PROFILE:
        raise ValueError("the frozen H7 scorer profile must activate H1-Prefix-Prior")
    if type(entries) is not tuple:
        raise ValueError("predecessor entries must be an exact tuple")

    references: dict[str, H7PredecessorReference] = {}
    obligations: list[str] = []
    validator_api = _load_verification_gate_api(
        expected_sha256=ledger_validator_sha256,
    )
    try:
        live_artifact_revision = validator_api.capture_artifact_revision(root)
    except (OSError, RuntimeError, ValueError) as error:
        live_artifact_revision = None
        obligations.append(
            f"current verification artifact revision is unavailable: {error}"
        )
    if live_artifact_revision is not None:
        revision_parts = live_artifact_revision.split(":")
        if (
            len(revision_parts) != 4
            or revision_parts[0] != "git"
            or revision_parts[1] != git_head
            or revision_parts[2] != "sha256"
            or _require_sha256(
                revision_parts[3],
                "live artifact revision digest",
            )
            != dirty_digest
        ):
            obligations.append(
                "live verification artifact revision differs from the H7 candidate"
            )
    for key, reference in entries:
        if key not in H7_PREDECESSOR_KEYS:
            obligations.append(f"unexpected predecessor key: {key}")
            continue
        if key in references:
            obligations.append(f"duplicate predecessor key: {key}")
            continue
        if type(reference) is not H7PredecessorReference:
            obligations.append(f"invalid predecessor record: {key}")
            continue
        references[key] = reference
    if tuple(references) != H7_PREDECESSOR_KEYS:
        for key in H7_PREDECESSOR_KEYS:
            if key not in references:
                obligations.append(f"missing predecessor: {key}")
        if set(references) == set(H7_PREDECESSOR_KEYS):
            obligations.append("predecessor registry order is not exact")

    candidate_junit_path: Path | None = None
    for key in H7_PREDECESSOR_KEYS:
        reference = references.get(key)
        if reference is None:
            continue
        try:
            reference.__post_init__()
            if (
                reference.git_head != git_head
                or reference.dirty_digest != dirty_digest
                or reference.junit_sha256 != junit_sha256
            ):
                obligations.append(f"stale predecessor candidate identity: {key}")
                continue
            resolved_junit_path = _candidate_path(
                reference.junit_path,
                repo_root=root,
                name="candidate JUnit",
                require_inside_repo=False,
            )
            if candidate_junit_path is None:
                candidate_junit_path = resolved_junit_path
            elif resolved_junit_path != candidate_junit_path:
                obligations.append(
                    f"predecessor references do not share one candidate JUnit: {key}"
                )
                continue
            _validate_predecessor_files(
                key,
                reference,
                repo_root=root,
                validator_api=validator_api,
                live_artifact_revision=live_artifact_revision,
            )
        except (OSError, RuntimeError, ValueError) as error:
            obligations.append(f"invalid predecessor {key}: {error}")
    return H7PredecessorValidation.create(
        references=references,
        obligations=tuple(obligations),
    )


def _validate_h1_prefix_prior_v2_payloads(
    payloads: Mapping[str, object],
    *,
    repo_root: Path,
    git_head: str,
    dirty_digest: str,
    junit_sha256: str,
) -> None:
    """Require H7's H1 predecessor to be the exact scorer-v2 producer."""

    expected_names = {
        "config.json",
        "schemas/generative_factor.json",
        "validation/h1_prefix_prior.json",
    }
    if set(payloads) != expected_names:
        raise ValueError("H1 prefix-prior artifact inventory is not scorer-v2")
    config_payload = payloads["config.json"]
    schema_payload = payloads["schemas/generative_factor.json"]
    validation = payloads["validation/h1_prefix_prior.json"]
    if (
        type(config_payload) is not dict
        or type(schema_payload) is not dict
        or type(validation) is not dict
    ):
        raise ValueError("H1 scorer-v2 payloads must be JSON objects")

    from vfe4.config import resolve_h1_prefix_prior_v2_config

    try:
        resolved = resolve_h1_prefix_prior_v2_config(
            config_payload,
            repo_root=repo_root,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("H1 predecessor config is not exact scorer-v2") from error
    expected_validation_fields = {
        "schema_version",
        "gate",
        "status",
        "obligations",
        "git_head",
        "dirty_digest",
        "source_sha256",
        "config_sha256",
        "junit_sha256",
        "fixture_id",
        "fixture_sha256",
        "base_fixture_sha256",
        "generative_factor_schema_sha256",
        "scorer_schema",
        "latent_projection_policy",
        "parent_history_policy",
        "invariants",
        "computation",
    }
    schema_sha256 = hashlib.sha256(
        canonical_h7_bytes(schema_payload)
    ).hexdigest()
    expected_source_sha256 = source_candidate_sha256(
        git_head_value=git_head,
        dirty_digest_value=dirty_digest,
    )
    if (
        set(validation) != expected_validation_fields
        or validation["schema_version"]
        != "h1-prefix-prior-validation-v3"
        or validation["gate"] != "H1-Prefix-Prior"
        or validation["status"] != "pass"
        or validation["obligations"] != []
        or validation["git_head"] != git_head
        or validation["dirty_digest"] != dirty_digest
        or validation["junit_sha256"] != junit_sha256
        or validation["source_sha256"] != expected_source_sha256
        or resolved.source.source_sha256 != expected_source_sha256
        or validation["config_sha256"] != resolved.config_sha256
        or validation["fixture_id"] != "h1-prefix-prior-scorer-v2"
        or validation["fixture_sha256"]
        != H7_H1_PREFIX_PRIOR_V2_FIXTURE_SHA256
        or validation["fixture_sha256"] != resolved.fixture_sha256
        or validation["base_fixture_sha256"]
        != H7_H1_PREFIX_PRIOR_BASE_FIXTURE_SHA256
        or validation["base_fixture_sha256"]
        != resolved.base_fixture_sha256
        or validation["generative_factor_schema_sha256"]
        != H1_PREFIX_PRIOR_V2_GENERATIVE_FACTOR_SCHEMA_SHA256
        or resolved.generative_factor_schema_sha256
        != H1_PREFIX_PRIOR_V2_GENERATIVE_FACTOR_SCHEMA_SHA256
        or schema_sha256
        != H1_PREFIX_PRIOR_V2_GENERATIVE_FACTOR_SCHEMA_SHA256
        or validation["scorer_schema"]
        != "parent-specific-pooled-prefix-bilinear-v1"
        or resolved.scorer_schema
        != "parent-specific-pooled-prefix-bilinear-v1"
        or validation["latent_projection_policy"]
        != "nonzero_bank_projections"
        or validation["parent_history_policy"]
        != "active_swapped_distinct_nonzero"
        or resolved.source.git_head != git_head
        or resolved.source.dirty_digest != dirty_digest
    ):
        raise ValueError(
            "H1 predecessor lacks exact scorer-v2 fixture, scorer, schema, "
            "source, and JUnit bindings"
        )


def _validate_predecessor_closure_claim(
    key: str,
    reference: H7PredecessorReference,
    *,
    repo_root: Path,
    claims: list[object],
    live_artifact_revision: str,
) -> None:
    contract = h7_predecessor_closure_claim_contract(
        key,
        repo_root=repo_root,
        artifact_path=reference.artifact_path,
        git_head=reference.git_head,
        dirty_digest=reference.dirty_digest,
        junit_path=reference.junit_path,
        junit_sha256=reference.junit_sha256,
        manifest_sha256=reference.manifest_sha256,
        payload_hashes=reference.payload_hashes,
    )
    matches = tuple(
        claim
        for claim in claims
        if isinstance(claim, Mapping)
        and claim.get("id") == contract["id"]
    )
    if len(matches) != 1:
        raise ValueError(
            "predecessor ledger lacks one exact predecessor-closure claim"
        )
    claim = matches[0]
    if (
        claim.get("domain") != contract["domain"]
        or claim.get("statement") != contract["statement"]
        or claim.get("artifact_revision") != live_artifact_revision
        or claim.get("state") != "EVIDENCE_VERIFIED"
        or claim.get("open_obligations") != []
        or claim.get("evidence_invalidated") is not False
    ):
        raise ValueError(
            "predecessor-closure claim does not bind this artifact and JUnit"
        )
    evidence = claim.get("evidence")
    if not isinstance(evidence, list):
        raise ValueError("predecessor-closure claim lacks typed evidence")
    by_id: dict[str, Mapping[str, object]] = {}
    for record in evidence:
        if not isinstance(record, Mapping):
            continue
        evidence_id = record.get("id")
        if type(evidence_id) is str:
            if evidence_id in by_id:
                raise ValueError(
                    "predecessor-closure claim duplicates an evidence ID"
                )
            by_id[evidence_id] = record
    for expected in contract["evidence"]:
        if not isinstance(expected, Mapping):
            raise RuntimeError("internal predecessor claim contract is invalid")
        evidence_id = expected["id"]
        record = by_id.get(evidence_id)
        if (
            record is None
            or record.get("kind") != expected["kind"]
            or record.get("location") != expected["location"]
            or record.get("artifact_revision") != live_artifact_revision
        ):
            raise ValueError(
                "predecessor-closure claim evidence does not bind its "
                "artifact manifest and candidate JUnit"
            )


def _validate_predecessor_files(
    key: str,
    reference: H7PredecessorReference,
    *,
    repo_root: Path,
    validator_api: _VerificationGateApi,
    live_artifact_revision: str | None,
) -> None:
    root = _candidate_path(
        reference.artifact_path,
        repo_root=repo_root,
        name="predecessor artifact",
        require_inside_repo=False,
    )
    manifest_path = root / "manifest.sha256"
    if not root.is_dir() or root.is_symlink() or manifest_path.is_symlink():
        raise ValueError("artifact root or manifest is not a regular owned path")
    manifest_bytes = manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != reference.manifest_sha256:
        raise ValueError("manifest hash mismatch")
    manifest = _manifest_hashes(manifest_bytes)
    if manifest != dict(reference.payload_hashes):
        raise ValueError("manifest and referenced payload inventory differ")
    observed_files: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("predecessor artifact contains a symlink")
        if path.is_file():
            observed_files.add(path.relative_to(root).as_posix())
    if observed_files != {*manifest, "manifest.sha256"}:
        raise ValueError("predecessor artifact has unlisted or missing files")

    payloads: dict[str, object] = {}
    for name, digest in manifest.items():
        path = (root / Path(*PurePosixPath(name).parts)).resolve(strict=True)
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("artifact payload escaped its root") from error
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"artifact payload is missing: {name}")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"artifact payload hash mismatch: {name}")
        if name.endswith(".json"):
            payloads[name] = json.loads(raw)

    if key == "h1_prefix_prior":
        _validate_h1_prefix_prior_v2_payloads(
            payloads,
            repo_root=repo_root,
            git_head=reference.git_head,
            dirty_digest=reference.dirty_digest,
            junit_sha256=reference.junit_sha256,
        )

    expected_source_sha256 = source_candidate_sha256(
        git_head_value=reference.git_head,
        dirty_digest_value=reference.dirty_digest,
    )
    expected_gate = {
        "h1_h5": "H5",
        "h1_prefix_prior": "H1-Prefix-Prior",
        "h6_prefix": "H6-Prefix",
    }[key]
    expected_schema: object = {
        "h1_h5": 1,
        "h1_prefix_prior": "h1-prefix-prior-validation-v3",
        "h6_prefix": "h6-prefix-validation-set-v1",
    }[key]
    validation_values = tuple(
        value
        for name, value in payloads.items()
        if name.startswith("validation/") and isinstance(value, Mapping)
    )
    if not any(
        value.get("gate") == expected_gate
        and value.get("status") == "pass"
        and value.get("schema_version") == expected_schema
        for value in validation_values
    ):
        raise ValueError(
            "artifact lacks its required producer schema and PASS validation"
        )
    for field, expected in (
        ("git_head", reference.git_head),
        ("dirty_digest", reference.dirty_digest),
        ("source_sha256", expected_source_sha256),
        ("junit_sha256", reference.junit_sha256),
    ):
        values = _field_values(payloads, field)
        if not values or any(value != expected for value in values):
            raise ValueError(f"artifact {field} differs from the H7 candidate")
    serialized_payloads = canonical_h7_bytes(payloads)
    if b"H7" in serialized_payloads or b"H6-Prediction" in serialized_payloads:
        raise ValueError("predecessor artifact was produced after or beyond H7 scope")
    if key == "h6_prefix" and b"predecessor_refs" in serialized_payloads:
        for value in payloads.values():
            if _nonempty_predecessor_mapping(value):
                raise ValueError("H6-Prefix contains forbidden predecessor identity")

    junit_path = _candidate_path(
        reference.junit_path,
        repo_root=repo_root,
        name="candidate JUnit",
        require_inside_repo=False,
    )
    if (
        not junit_path.is_file()
        or junit_path.is_symlink()
        or _sha256_file(junit_path) != reference.junit_sha256
    ):
        raise ValueError("candidate JUnit preimage hash mismatch")
    ledger_path = _candidate_path(
        reference.ledger_path,
        repo_root=repo_root,
        name="predecessor ledger",
        require_inside_repo=True,
    )
    if not ledger_path.is_file() or ledger_path.is_symlink():
        raise ValueError("predecessor ledger is unavailable")
    ledger_bytes = ledger_path.read_bytes()
    if hashlib.sha256(ledger_bytes).hexdigest() != reference.ledger_sha256:
        raise ValueError("predecessor ledger hash mismatch")
    ledger = json.loads(ledger_bytes)
    if type(ledger) is not dict:
        raise ValueError("predecessor ledger must encode one JSON object")
    ledger_errors = validator_api.validate_ledger(ledger)
    if type(ledger_errors) is not list or any(
        type(error) is not str for error in ledger_errors
    ):
        raise ValueError("deterministic ledger validator returned an invalid result")
    if ledger_errors:
        raise ValueError(
            "deterministic ledger validation failed: " + "; ".join(ledger_errors)
        )
    claims = ledger.get("claims")
    if (
        live_artifact_revision is None
        or ledger.get("schema_version") != "1.0"
        or ledger.get("mode") != "closure"
        or ledger.get("artifact_revision") != live_artifact_revision
        or not isinstance(claims, list)
        or not claims
        or any(
            not isinstance(claim, Mapping)
            or claim.get("artifact_revision") != live_artifact_revision
            or claim.get("state") != "EVIDENCE_VERIFIED"
            or claim.get("open_obligations") != []
            or claim.get("evidence_invalidated") is not False
            for claim in claims
        )
    ):
        raise ValueError("predecessor ledger is not validated at this candidate")
    _validate_predecessor_closure_claim(
        key,
        reference,
        repo_root=repo_root,
        claims=claims,
        live_artifact_revision=live_artifact_revision,
    )


def _manifest_hashes(data: bytes) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in data.decode("utf-8").splitlines():
        pieces = line.split("  ", 1)
        if len(pieces) != 2:
            raise ValueError("manifest line is not canonical")
        digest = _require_sha256(pieces[0], "manifest digest")
        name = PurePosixPath(pieces[1])
        if (
            name.is_absolute()
            or name.as_posix() != pieces[1]
            or any(part in ("", ".", "..") for part in name.parts)
            or pieces[1] in result
        ):
            raise ValueError("manifest payload path is invalid or duplicated")
        result[pieces[1]] = digest
    if not result:
        raise ValueError("manifest cannot be empty")
    return result


def _nonempty_predecessor_mapping(value: object) -> bool:
    if isinstance(value, Mapping):
        if "predecessor_refs" in value and value["predecessor_refs"] not in ({}, None):
            return True
        return any(_nonempty_predecessor_mapping(item) for item in value.values())
    if isinstance(value, list):
        return any(_nonempty_predecessor_mapping(item) for item in value)
    return False


def _field_values(value: object, field: str) -> tuple[object, ...]:
    result: list[object] = []
    if isinstance(value, Mapping):
        if field in value:
            result.append(value[field])
        for item in value.values():
            result.extend(_field_values(item, field))
    elif isinstance(value, list):
        for item in value:
            result.extend(_field_values(item, field))
    return tuple(result)


_H7ExpectedNegativeState = Literal[
    "success",
    "false_acceptance",
    "inconclusive",
]


def _classify_h7_status_from_state(
    *,
    obligations: tuple[str, ...],
    failed_invariant_ids: tuple[str, ...],
    expected_negative_state: _H7ExpectedNegativeState,
) -> GateStatus:
    """Apply the preregistered INCONCLUSIVE-before-FAIL-before-PASS order."""

    _ordered_unique(obligations, "H7 obligations")
    _ordered_unique(failed_invariant_ids, "failed invariant IDs")
    if expected_negative_state not in (
        "success",
        "false_acceptance",
        "inconclusive",
    ):
        raise ValueError("expected-negative state is outside the closed inventory")
    if obligations or expected_negative_state == "inconclusive":
        return GateStatus.INCONCLUSIVE
    if failed_invariant_ids or expected_negative_state == "false_acceptance":
        return GateStatus.FAIL
    return GateStatus.PASS


def classify_h7_status(
    *,
    obligations: tuple[str, ...],
    failed_invariant_ids: tuple[str, ...],
    expected_negative_trial: H7TrialResult | None,
) -> GateStatus:
    """Derive status from the owned outside-stabilizer trial, never a flag."""

    state: _H7ExpectedNegativeState = (
        "inconclusive"
        if expected_negative_trial is None
        else _h7_expected_negative_state(expected_negative_trial)
    )
    return _classify_h7_status_from_state(
        obligations=obligations,
        failed_invariant_ids=failed_invariant_ids,
        expected_negative_state=state,
    )


def _inventory_obligations(
    *,
    trial_ids: tuple[str, ...],
    control_ids: tuple[str, ...],
) -> tuple[str, ...]:
    obligations: list[str] = []
    if trial_ids != H7_REQUIRED_TRIAL_IDS:
        obligations.append(
            "required H7 trial inventory is missing, duplicated, or reordered"
        )
    if control_ids != H7_CONTROL_IDS:
        obligations.append(
            "required H7 control inventory is missing, duplicated, or reordered"
        )
    return tuple(obligations)


def assemble_h7_gate_evaluation(
    *,
    repo_root: Path,
    captured_fixture_bytes: Mapping[str, bytes],
    predecessor_entries: tuple[tuple[str, H7PredecessorReference], ...],
    git_head: str,
    dirty_digest: str,
    junit_sha256: str,
    scorer_profile: str,
    fixture_hashes: Mapping[str, str],
    trials: tuple[H7TrialResult, ...],
    controls: tuple[H7ControlResult, ...],
    oracle_obligations: tuple[str, ...],
    additional_obligations: tuple[str, ...] = (),
) -> H7GateEvaluation:
    """Apply H7 status precedence and freeze one canonical gate evaluation."""

    captured = _require_captured_fixture_bytes(captured_fixture_bytes)
    dependency_closure = capture_h7_dependency_closure(
        repo_root,
        captured_fixture_bytes=captured,
    )
    predecessor_validation = validate_h7_predecessor_registry(
        predecessor_entries,
        repo_root=repo_root,
        git_head=git_head,
        dirty_digest=dirty_digest,
        junit_sha256=junit_sha256,
        scorer_profile=scorer_profile,
        ledger_validator_sha256=dependency_closure.ledger_validator_sha256,
    )
    _ordered_unique(oracle_obligations, "oracle obligations")
    _ordered_unique(additional_obligations, "additional obligations")

    obligations = list(predecessor_validation.obligations)
    obligations.extend(oracle_obligations)
    obligations.extend(additional_obligations)
    effective_fixture_hashes: dict[str, str] = {
        "h1_fixture_raw_sha256": hashlib.sha256(
            captured[H7_CAPTURED_FIXTURE_PATHS[0]]
        ).hexdigest(),
        "h7_fixture_raw_sha256": hashlib.sha256(
            captured[H7_CAPTURED_FIXTURE_PATHS[1]]
        ).hexdigest(),
    }
    for key in fixture_hashes:
        if key not in (
            "density_probe_table_raw_sha256",
            "density_probe_set_sha256",
        ):
            obligations.append(f"unexpected fixture identity: {key}")
            continue
        effective_fixture_hashes[key] = fixture_hashes[key]
    fixture_keys = tuple(
        key
        for key in H7GateResult.fixture_hash_keys
        if key in effective_fixture_hashes
    )
    for key in H7GateResult.fixture_hash_keys:
        if key not in effective_fixture_hashes:
            obligations.append(f"missing fixture identity: {key}")
    for key, expected in H7_FROZEN_SOURCE_FIXTURE_HASHES.items():
        if effective_fixture_hashes.get(key) != expected:
            obligations.append(f"frozen fixture identity mismatch: {key}")

    trial_ids = tuple(item.spec.trial_id for item in trials)
    control_ids = tuple(item.control_id for item in controls)
    obligations.extend(
        _inventory_obligations(
            trial_ids=trial_ids,
            control_ids=control_ids,
        )
    )
    for trial in trials:
        trial.__post_init__()
        if not trial.envelope.passed:
            obligations.append(
                f"required trial is outside the envelope: {trial.spec.trial_id}"
            )
    for control in controls:
        control.__post_init__()
        if not control.detected:
            obligations.append(f"negative control is nondecisive: {control.control_id}")
    expected_negative_trials = tuple(
        trial for trial in trials if trial.spec.role == "expected_negative"
    )
    expected_negative_trial = (
        expected_negative_trials[0]
        if len(expected_negative_trials) == 1
        else None
    )
    expected_negative_state: _H7ExpectedNegativeState = (
        "inconclusive"
        if expected_negative_trial is None
        else _h7_expected_negative_state(expected_negative_trial)
    )
    if expected_negative_state == "inconclusive":
        obligations.append(
            "expected-negative trial is missing or nondecisive"
        )
    obligations = list(dict.fromkeys(obligations))

    failed_invariant_ids = tuple(
        f"{trial.spec.trial_id}:{trial.spec.expected_predicate}"
        for trial in trials
        if trial.spec.role in ("scalar_regression", "positive_covariance")
        and trial.envelope.passed
        and not trial.predicate_satisfied
    )
    final_obligations = tuple(obligations)
    status = classify_h7_status(
        obligations=final_obligations,
        failed_invariant_ids=failed_invariant_ids,
        expected_negative_trial=expected_negative_trial,
    )
    if status is GateStatus.INCONCLUSIVE:
        outcome = H7InconclusiveOutcome.create(
            kind="INCONCLUSIVE",
            obligations=final_obligations,
        )
    elif status is GateStatus.FAIL:
        final_obligations = ()
        outcome = H7FailOutcome.create(
            kind="FAIL",
            failed_invariant_ids=failed_invariant_ids,
            expected_negative_false_acceptance=(
                expected_negative_state == "false_acceptance"
            ),
        )
    else:
        final_obligations = ()
        outcome = H7PassOutcome.create(
            kind="PASS",
            scalar_trial_ids=H7_SCALAR_TRIAL_IDS,
            positive_trial_ids=H7_MATRIX_TRIAL_IDS[:-1],
            expected_negative_trial_id=H7_MATRIX_TRIAL_IDS[-1],
            control_ids=H7_CONTROL_IDS,
        )

    result = H7GateResult.create(
        gate="H7",
        status=status,
        fixture_hashes={
            key: effective_fixture_hashes[key] for key in fixture_keys
        },
        predecessor_references=predecessor_validation.references,
        trials=trials,
        controls=controls,
        outcome=outcome,
        obligations=final_obligations,
    )
    fixture_set_sha256 = h7_owned_sha256(
        "vfe4.h7.fixture-set.v1",
        result.fixture_hashes,
    )
    combined_dependency_sha256 = h7_owned_sha256(
        "vfe4.h7.complete-dependency-closure.v1",
        {
            "source": dependency_closure.dependency_closure_sha256,
            "predecessors": predecessor_validation.registry_sha256,
        },
    )
    payload = canonical_h7_bytes(
        {
            "schema": H7_VALIDATION_SCHEMA,
            "verification_prefix": H7_VERIFICATION_PREFIX,
            "result": result,
            "fixture_set_sha256": fixture_set_sha256,
            "dependency_closure_sha256": combined_dependency_sha256,
            "source_dependencies": dependency_closure.file_sha256,
            "verification_ledger_validator_sha256": (
                dependency_closure.ledger_validator_sha256
            ),
            "predecessor_registry_sha256": predecessor_validation.registry_sha256,
            "group_claim": "direct GL+(2,R) forward covariance",
            "scalar_regression_nonclaim": "GL+(1,R) replay is not GL+(2,R) evidence",
            "nonclaims": H7_NONCLAIMS,
        }
    )
    return H7GateEvaluation.create(
        result=result,
        validation_payload_canonical_json=payload,
        validation_payload_sha256=hashlib.sha256(payload).hexdigest(),
        fixture_set_sha256=fixture_set_sha256,
        dependency_closure_sha256=combined_dependency_sha256,
    )


def h7_validation_payload(evaluation: H7GateEvaluation) -> dict[str, object]:
    """Validate all canonical bindings and return a fresh JSON payload."""

    if type(evaluation) is not H7GateEvaluation:
        raise ValueError("evaluation must be an exact H7GateEvaluation")
    evaluation.__post_init__()
    payload = json.loads(evaluation.validation_payload_canonical_json)
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != H7_VALIDATION_SCHEMA
        or payload.get("fixture_set_sha256") != evaluation.fixture_set_sha256
        or payload.get("dependency_closure_sha256")
        != evaluation.dependency_closure_sha256
    ):
        raise ValueError("H7 validation payload is not bound to its evaluation")
    result_value = payload.get("result")
    result = (
        cast(Mapping[str, object], result_value)
        if isinstance(result_value, Mapping)
        else None
    )
    payload_obligations = None if result is None else result.get("obligations")
    if (
        result is None
        or result.get("result_sha256") != evaluation.result.result_sha256
        or result.get("status") != evaluation.result.status.value
        or not isinstance(payload_obligations, list)
        or tuple(payload_obligations)
        != evaluation.result.obligations
    ):
        raise ValueError("H7 validation payload is not bound to its gate result")
    return payload


__all__ = [
    "H7_ACTIVE_SCORER_PROFILE",
    "H7_CAPTURED_FIXTURE_PATHS",
    "H7DependencyClosure",
    "H7_FROZEN_SOURCE_FIXTURE_HASHES",
    "H7_NONCLAIMS",
    "H7_PREDECESSOR_CLAIM_IDS",
    "H7_PREDECESSOR_KEYS",
    "H7_REQUIRED_DEPENDENCY_PATHS",
    "H7_SOURCE_ONLY_OBLIGATIONS",
    "H7_VALIDATION_SCHEMA",
    "H7_VERIFICATION_PREFIX",
    "H7PredecessorValidation",
    "assemble_h7_gate_evaluation",
    "capture_h7_dependency_closure",
    "classify_h7_status",
    "h7_predecessor_closure_binding_sha256",
    "h7_predecessor_closure_claim_contract",
    "h7_validation_payload",
    "validate_h7_predecessor_registry",
]
