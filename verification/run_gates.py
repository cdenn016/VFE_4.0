"""Ordered, atomic publication for the implemented H1/H2/H3 prefixes."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from verification.h1_gate import (
    EXPECTED_H1_FIXTURE_SHA256,
    FIXTURE_PATH,
    H1GateEvaluation,
    evaluate_h1,
)
from verification.h2_gate import H2GateEvaluation, evaluate_h2, h2_validation_payload
from verification.h3_gate import H3GateEvaluation, evaluate_h3, h3_validation_payload
from verification.h4_gate import (
    H4GateEvaluation,
    evaluate_h4,
    h4_validation_artifact,
    h4_validation_payload,
)
from verification.h5_gate import (
    H5GateEvaluation,
    H5GateResult,
    H5PreflightPhase,
    evaluate_h5,
    h5_update_binding_preimages,
    h5_validation_payload,
)
from verification.h7_budget import (
    CONTROL_ALLOWANCE_MULTIPLE,
    CONTROL_MINIMUM_RELATIVE_RESIDUAL,
    EPS64,
    MAX_ORACLE_RELATIVE_DELTA,
    ROUNDING_CONSTANT,
)
from verification.h7_gate import (
    H7_ACTIVE_SCORER_PROFILE,
    H7_CAPTURED_FIXTURE_PATHS,
    H7_FROZEN_SOURCE_FIXTURE_HASHES,
    H7_NONCLAIMS,
    H7_PREDECESSOR_KEYS,
    H7_SOURCE_ONLY_OBLIGATIONS,
    H7_VERIFICATION_PREFIX,
    assemble_h7_gate_evaluation,
    h7_validation_payload,
)
from verification.h8_correctness import produce_h8_correctness_grid
from verification.h8_gate import (
    H8_PUBLICATION_PAYLOAD_KEYS,
    H8_VERIFIER_PREFIX,
    assemble_h8_gate_evaluation,
    assemble_h8_source_only_evaluation,
    build_h8_publication_payloads,
    canonical_h8_json_bytes,
    h8_current_candidate_result_payload,
    h8_current_refs_registry_payload,
    validate_h8_prerequisite_artifacts,
)
from verification.h8_orchestrator import (
    derive_h8_child_start_authorization,
    run_h8_parent_attempt,
)
from verification.h8_wire import require_h8_startup_environment
from vfe4.artifacts import (
    CandidateArtifactReference,
    build_environment,
    build_provenance,
    canonical_json_bytes,
    current_source_identity,
    publish_run_directory,
    source_candidate_sha256,
)
from vfe4.artifacts.provenance import (
    build_h7_provenance,
    build_h8_environment,
    build_h8_provenance,
)
from vfe4.config import ResolvedConfig, resolve_config
from vfe4.types import (
    GateResult,
    GateStatus,
    H3GateResult,
    H4GateResult,
    H7GateResult,
    H8GateResult,
)
from vfe4.types.h7 import H7PredecessorReference
from vfe4.types.h8 import (
    CurrentH8PrerequisiteRefs,
    H8H1H5Reference,
    H8H1PrefixPriorReference,
    H8H6PredictionReference,
    H8H6PredictionV3Reference,
    H8H6PrefixReference,
    H8H6PrefixSemanticFamilyReference,
    H8H7Reference,
    H8LegacyH6PrefixReference,
    H8LegacyH6PrefixV4Reference,
    H8LegacyH6PredictionReference,
)
from vfe4.validation import (
    H3_COUPLED_FIXTURE_PATH,
    H3_ZERO_CONTROL_FIXTURE_PATH,
    H7_FIXTURE_PATH,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
H5_UPDATE_SPEC_FIXTURE_PATH = (
    REPO_ROOT / "vfe4" / "validation" / "fixtures" / "h5_conditional_update_v1.json"
)
H8_PREREGISTRATION_PATH = (
    REPO_ROOT / "docs" / "preregistrations" / "2026-07-21-h8-sparse-scale.md"
)
_ALLOWED_PREFIXES = (
    ("H1",),
    ("H1", "H2"),
    ("H1", "H2", "H3"),
    ("H1", "H2", "H3", "H4", "H5"),
)
_ALLOWED_RESULT_PREFIXES = (*_ALLOWED_PREFIXES, ("H7",))
_ALLOWED_RESULT_PREFIXES = (*_ALLOWED_RESULT_PREFIXES, ("H8",))
_PREDICTION_CORRECTNESS_GATES: tuple[
    Literal["H1", "H2", "H3", "H5"], ...
] = ("H1", "H2", "H3", "H5")
_LOWER_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class VerificationRunResult:
    gate_results: tuple[
        GateResult
        | H3GateResult
        | H4GateResult
        | H5GateResult
        | H7GateResult
        | H8GateResult,
        ...,
    ]
    run_directory: Path
    prediction_correctness_artifacts: tuple[
        tuple[Literal["H1", "H2", "H3", "H5"], Path, str], ...
    ] = ()

    def __post_init__(self) -> None:
        if type(self.gate_results) is not tuple or not all(
            type(result)
            in (
                GateResult,
                H3GateResult,
                H4GateResult,
                H5GateResult,
                H7GateResult,
                H8GateResult,
            )
            for result in self.gate_results
        ):
            raise ValueError(
                "gate_results must contain exact immutable gate results"
            )
        for result in self.gate_results:
            if type(result) is H8GateResult:
                result.__post_init__()
        gate_names = tuple(result.gate for result in self.gate_results)
        if gate_names not in _ALLOWED_RESULT_PREFIXES:
            raise ValueError("gate_results must contain an implemented ordered prefix")
        if not isinstance(self.run_directory, Path):
            raise ValueError("run_directory must be a Path")
        if type(self.prediction_correctness_artifacts) is not tuple:
            raise ValueError("prediction_correctness_artifacts must be a tuple")
        correctness_gates: list[str] = []
        correctness_roots: list[Path] = []
        for reference in self.prediction_correctness_artifacts:
            if type(reference) is not tuple or len(reference) != 3:
                raise ValueError(
                    "prediction correctness references must be "
                    "(gate, root, manifest_sha256) tuples"
                )
            gate, root, manifest_sha256 = reference
            if gate not in _PREDICTION_CORRECTNESS_GATES:
                raise ValueError("prediction correctness gate is unsupported")
            if not isinstance(root, Path):
                raise ValueError("prediction correctness root must be a Path")
            if (
                type(manifest_sha256) is not str
                or len(manifest_sha256) != 64
                or any(character not in _LOWER_HEX for character in manifest_sha256)
            ):
                raise ValueError(
                    "prediction correctness manifest SHA-256 must be lowercase 64-hex"
                )
            correctness_gates.append(gate)
            correctness_roots.append(root)
        if correctness_gates and tuple(correctness_gates) != _PREDICTION_CORRECTNESS_GATES:
            raise ValueError(
                "prediction correctness references must contain H1, H2, H3, H5 "
                "in frozen order"
            )
        if correctness_gates and gate_names != ("H1", "H2", "H3", "H4", "H5"):
            raise ValueError(
                "prediction correctness references require the full H1--H5 result"
            )
        if len(set(correctness_roots)) != len(correctness_roots):
            raise ValueError("prediction correctness roots must be distinct")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def candidate_artifact_reference_to_h7_reference(
    candidate: CandidateArtifactReference,
    *,
    junit_sha256: str,
    junit_path: Path,
    ledger_path: Path,
    ledger_sha256: str,
) -> H7PredecessorReference:
    """Losslessly add the candidate JUnit/ledger binding required by H7."""

    if type(candidate) is not CandidateArtifactReference:
        raise ValueError("candidate must be an exact CandidateArtifactReference")
    if not isinstance(ledger_path, Path):
        raise ValueError("ledger_path must be a Path")
    if not isinstance(junit_path, Path):
        raise ValueError("junit_path must be a Path")
    return H7PredecessorReference.create(
        artifact_path=candidate.artifact_path.as_posix(),
        git_head=candidate.git_head,
        dirty_digest=candidate.dirty_digest,
        junit_sha256=junit_sha256,
        junit_path=junit_path.resolve(strict=False).as_posix(),
        manifest_sha256=candidate.manifest_sha256,
        payload_hashes=candidate.payload_hashes,
        ledger_path=ledger_path.resolve(strict=False).as_posix(),
        ledger_sha256=ledger_sha256,
    )


def h7_reference_registry_bytes(
    references: Mapping[str, H7PredecessorReference],
) -> bytes:
    """Serialize the exact ordered H7 reference registry canonically."""

    if (
        not isinstance(references, Mapping)
        or tuple(references) != H7_PREDECESSOR_KEYS
        or any(
            type(reference) is not H7PredecessorReference
            for reference in references.values()
        )
    ):
        raise ValueError("H7 registry must contain exact ordered references")
    for reference in references.values():
        reference.__post_init__()
    return canonical_json_bytes(
        {key: references[key] for key in H7_PREDECESSOR_KEYS}
    )


def parse_h7_reference_registry_bytes(
    registry_bytes: bytes,
) -> tuple[tuple[str, H7PredecessorReference], ...]:
    """Decode once through CandidateArtifactReference and reject lossy bytes."""

    if type(registry_bytes) is not bytes or not registry_bytes:
        raise ValueError("H7 reference registry must be nonempty bytes")
    try:
        payload = json.loads(registry_bytes.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("H7 reference registry is not UTF-8 JSON") from exc
    if (
        not isinstance(payload, dict)
        or tuple(payload) != H7_PREDECESSOR_KEYS
        or canonical_json_bytes(payload) != registry_bytes
    ):
        raise ValueError("H7 reference registry is not exact ordered canonical JSON")
    expected_fields = {
        "artifact_path",
        "git_head",
        "dirty_digest",
        "junit_sha256",
        "junit_path",
        "manifest_sha256",
        "payload_hashes",
        "ledger_path",
        "ledger_sha256",
        "reference_sha256",
    }
    entries: list[tuple[str, H7PredecessorReference]] = []
    for key in H7_PREDECESSOR_KEYS:
        raw = payload[key]
        if (
            not isinstance(raw, dict)
            or set(raw) != expected_fields
            or not isinstance(raw["payload_hashes"], dict)
        ):
            raise ValueError(f"H7 reference registry entry is malformed: {key}")
        candidate = CandidateArtifactReference(
            artifact_path=Path(raw["artifact_path"]),
            git_head=raw["git_head"],
            dirty_digest=raw["dirty_digest"],
            manifest_sha256=raw["manifest_sha256"],
            payload_hashes=raw["payload_hashes"],
        )
        reference = candidate_artifact_reference_to_h7_reference(
            candidate,
            junit_sha256=raw["junit_sha256"],
            junit_path=Path(raw["junit_path"]),
            ledger_path=Path(raw["ledger_path"]),
            ledger_sha256=raw["ledger_sha256"],
        )
        if json.loads(canonical_json_bytes(reference)) != raw:
            raise ValueError(f"H7 reference registry entry is lossy: {key}")
        entries.append((key, reference))
    references = {key: reference for key, reference in entries}
    if h7_reference_registry_bytes(references) != registry_bytes:
        raise ValueError("H7 reference registry changed during typed adaptation")
    return tuple(entries)


def parse_h8_reference_registry_bytes(
    registry_bytes: bytes,
) -> CurrentH8PrerequisiteRefs:
    """Decode the exact current-candidate H8 registry without path discovery."""

    if type(registry_bytes) is not bytes or not registry_bytes:
        raise ValueError("H8 reference registry must be nonempty bytes")
    try:
        payload = json.loads(registry_bytes.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("H8 reference registry is not strict UTF-8 JSON") from exc
    if (
        not isinstance(payload, dict)
        or set(payload)
        != {
            "schema_version",
            "candidate",
            "h7_compatibility_refs",
            "references",
        }
        or payload["schema_version"]
        not in (
            "h8-current-candidate-refs-v1",
            "h8-current-candidate-refs-v2",
            "h8-current-candidate-refs-v3",
            "h8-current-candidate-refs-v4",
            "h8-current-candidate-refs-v5",
        )
        or canonical_h8_json_bytes(payload) != registry_bytes
    ):
        raise ValueError("H8 reference registry is not exact canonical JSON")
    candidate = payload["candidate"]
    raw_compatibility = payload["h7_compatibility_refs"]
    raw_references = payload["references"]
    if (
        not isinstance(candidate, dict)
        or set(candidate) != {"git_head", "dirty_digest", "junit_sha256"}
        or not isinstance(raw_compatibility, dict)
        or not isinstance(raw_references, dict)
    ):
        raise ValueError("H8 reference registry has malformed identity sections")

    compatibility_bytes = canonical_json_bytes(raw_compatibility)
    compatibility = dict(parse_h7_reference_registry_bytes(compatibility_bytes))
    common_fields = {
        "kind",
        "artifact_path",
        "manifest_sha256",
        "result_path",
        "result_sha256",
        "content_hashes",
        "payload_hashes",
        "ledger_path",
        "ledger_sha256",
        "producer_head",
        "producer_dirty_digest",
        "candidate_junit_sha256",
        "status",
    }
    registry_schema = payload["schema_version"]
    h6_prediction_type: type[object]
    h6_prediction_fields: set[str]
    if registry_schema in (
        "h8-current-candidate-refs-v4",
        "h8-current-candidate-refs-v5",
    ):
        h6_prediction_type = H8H6PredictionV3Reference
        h6_prediction_fields = common_fields | {
            "config_schema",
            "readiness_schema",
            "raw_inventory_schema",
            "metrics_schema",
            "result_schema",
            "authorities_path",
            "authorities_manifest_sha256",
            "authorities_sha256",
            "config_sha256",
            "readiness_sha256",
            "plan_sha256",
            "matching_set_sha256",
            "validation_bundle_path",
            "validation_bundle_manifest_sha256",
            "validation_bundle_sha256",
            "checkpoint_selection_sha256",
            "reservation_path",
            "reservation_sha256",
            "reservation_file_sha256",
            "terminal_path",
            "terminal_sha256",
            "terminal_manifest_sha256",
            "finalized_path",
            "finalized_manifest_sha256",
            "pointer_path",
            "pointer_sha256",
            "pointer_manifest_sha256",
            "experiment_identity_sha256",
            "opening_proof_sha256",
            "raw_inventory_sha256",
            "metrics_sha256",
            "result_record_sha256",
            "ledger_validator_sha256",
            "artifact_revision",
            "candidate_junit_path",
        }
    elif registry_schema in (
        "h8-current-candidate-refs-v2",
        "h8-current-candidate-refs-v3",
    ):
        h6_prediction_type = H8H6PredictionReference
        h6_prediction_fields = common_fields | {
            "prediction_schema",
            "config_schema",
            "readiness_schema",
            "metrics_schema",
            "result_schema",
            "experiment_sha256",
            "config_sha256",
            "readiness_artifact_path",
            "readiness_manifest_sha256",
            "readiness_sha256",
            "correctness_artifact_paths",
            "h1_prefix_prior_artifact_path",
            "smc_accuracy_artifact_path",
            "smc_accuracy_manifest_sha256",
            "h6_prefix_artifact_path",
            "h6_prefix_manifest_sha256",
            "blinded_data_artifact_path",
            "blinded_data_manifest_sha256",
            "matching_artifact_path",
            "matching_manifest_sha256",
            "matching_set_sha256",
            "h1_prefix_prior_generative_factor_schema_sha256",
            "smc_bias_semantics_sha256",
            "objective_gate_spec_sha256",
            "metrics_sha256",
        }
    else:
        h6_prediction_type = H8LegacyH6PredictionReference
        h6_prediction_fields = common_fields | {"experiment_sha256"}
    if registry_schema == "h8-current-candidate-refs-v5":
        h6_prefix_type: type[object] = H8H6PrefixReference
        h6_prefix_fields = common_fields | {
            "config_schema",
            "validation_schema",
            "certificate_set_schema",
            "config_sha256",
            "workload_plan_sha256",
            "validation_payload_sha256",
            "prefix_certificate_set_sha256",
            "a0_direct_exact_prefix_certificate_sha256",
            "semantic_families",
        }
    elif registry_schema in (
        "h8-current-candidate-refs-v3",
        "h8-current-candidate-refs-v4",
    ):
        h6_prefix_type = H8LegacyH6PrefixV4Reference
        h6_prefix_fields = common_fields | {
            "config_schema",
            "validation_schema",
            "certificate_set_schema",
            "config_sha256",
            "workload_plan_sha256",
            "validation_payload_sha256",
            "prefix_certificate_set_sha256",
            "semantic_families",
        }
    else:
        h6_prefix_type = H8LegacyH6PrefixReference
        h6_prefix_fields = common_fields | {
            "certificate_set_sha256",
            "certificate_hashes",
        }
    variants: tuple[tuple[str, type[object], set[str]], ...] = (
        ("h1_h5", H8H1H5Reference, common_fields),
        ("h1_prefix_prior", H8H1PrefixPriorReference, common_fields),
        (
            "h6_prefix",
            h6_prefix_type,
            h6_prefix_fields,
        ),
        (
            "h7",
            H8H7Reference,
            common_fields
            | {
                "result_pointer_path",
                "result_pointer_sha256",
                "fixture_set_sha256",
            },
        ),
        (
            "h6_prediction",
            h6_prediction_type,
            h6_prediction_fields,
        ),
    )
    typed: dict[str, object] = {}
    if set(raw_references) != {name for name, _, _ in variants}:
        raise ValueError("H8 reference registry variant inventory is not exact")
    for name, expected_type, expected_fields in variants:
        raw = raw_references[name]
        if (
            not isinstance(raw, dict)
            or set(raw) != expected_fields
            or not isinstance(raw["content_hashes"], dict)
            or not isinstance(raw["payload_hashes"], dict)
            or (
                name == "h6_prediction"
                and registry_schema
                in (
                    "h8-current-candidate-refs-v2",
                    "h8-current-candidate-refs-v3",
                )
                and not isinstance(raw["correctness_artifact_paths"], dict)
            )
        ):
            raise ValueError(f"H8 reference registry entry is malformed: {name}")
        typed_raw = dict(raw)
        if (
            name == "h6_prefix"
            and registry_schema
            in (
                "h8-current-candidate-refs-v3",
                "h8-current-candidate-refs-v4",
                "h8-current-candidate-refs-v5",
            )
        ):
            raw_families = raw["semantic_families"]
            family_fields = {
                "semantic_family_index",
                "semantic_family_sha256",
                "validation_payload_sha256",
                "certificate_sha256",
            }
            if (
                type(raw_families) is not list
                or not raw_families
                or any(
                    type(row) is not dict or set(row) != family_fields
                    for row in raw_families
                )
            ):
                raise ValueError(
                    "H8 bounded Prefix semantic-family rows are malformed"
                )
            typed_raw["semantic_families"] = tuple(
                H8H6PrefixSemanticFamilyReference(**row)
                for row in raw_families
            )
        value = expected_type(**typed_raw)
        if json.loads(canonical_h8_json_bytes(value)) != raw:
            raise ValueError(f"H8 reference registry entry is lossy: {name}")
        typed[name] = value

    refs = CurrentH8PrerequisiteRefs(
        candidate_head=candidate["git_head"],
        candidate_dirty_digest=candidate["dirty_digest"],
        candidate_junit_sha256=candidate["junit_sha256"],
        h7_compatibility_refs=compatibility,
        h1_h5=typed["h1_h5"],  # type: ignore[arg-type]
        h1_prefix_prior=typed["h1_prefix_prior"],  # type: ignore[arg-type]
        h6_prefix=typed["h6_prefix"],  # type: ignore[arg-type]
        h7=typed["h7"],  # type: ignore[arg-type]
        h6_prediction=typed["h6_prediction"],  # type: ignore[arg-type]
        registry_sha256=hashlib.sha256(registry_bytes).hexdigest(),
    )
    if canonical_h8_json_bytes(h8_current_refs_registry_payload(refs)) != registry_bytes:
        raise ValueError("H8 reference registry changed during typed adaptation")
    return refs


def _run_name(timestamp: str, config_hash: str, gates: tuple[str, ...]) -> str:
    safe = timestamp.replace("-", "").replace(":", "").replace(".", "")
    prefix = "-".join(gate.lower() for gate in gates)
    return f"verify-{prefix}-{safe}-{config_hash[:12]}"


def _config_payload(config: ResolvedConfig) -> dict[str, object]:
    payload = json.loads(config.canonical_json)
    payload["config_sha256"] = config.config_sha256
    return payload


def _raw_config_payload(config: ResolvedConfig) -> tuple[dict[str, object], bytes]:
    """Recover the exact canonical config bytes consumed by one gate."""

    if type(config) is not ResolvedConfig:
        raise ValueError("prediction correctness config must be a ResolvedConfig")
    try:
        payload = json.loads(config.canonical_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("prediction correctness config JSON is invalid") from exc
    if type(payload) is not dict:
        raise ValueError("prediction correctness config must be one JSON object")
    raw_bytes = canonical_json_bytes(payload)
    if raw_bytes != config.canonical_json.encode("utf-8"):
        raise ValueError(
            "prediction correctness config bytes differ from canonical_json"
        )
    if hashlib.sha256(raw_bytes).hexdigest() != config.config_sha256:
        raise ValueError(
            "prediction correctness raw config SHA-256 differs from config"
        )
    return payload, raw_bytes


def _ordered_prediction_inputs(
    value: object,
    *,
    name: str,
) -> dict[str, object]:
    if (
        type(value) is not tuple
        or any(type(item) is not tuple or len(item) != 2 for item in value)
    ):
        raise ValueError(f"{name} must be an ordered tuple of gate/value pairs")
    gates = tuple(item[0] for item in value)
    if gates != _PREDICTION_CORRECTNESS_GATES:
        raise ValueError(f"{name} must contain H1, H2, H3, H5 in frozen order")
    return {item[0]: item[1] for item in value}


def _prediction_correctness_run_name(
    timestamp: str,
    gate: Literal["H1", "H2", "H3", "H5"],
    config_sha256: str,
) -> str:
    safe = timestamp.replace("-", "").replace(":", "").replace(".", "")
    return (
        f"verify-prediction-correctness-{gate.lower()}-"
        f"{safe}-{config_sha256[:12]}"
    )


def _producer_validation_result_fields(
    *,
    gate: Literal["H1", "H2", "H3", "H5"],
    producer_validation: Mapping[str, object],
) -> tuple[str, str, tuple[str, ...]]:
    if gate in ("H1", "H2"):
        record = producer_validation.get("gate_result")
    elif gate == "H5":
        record = producer_validation.get("result")
    else:
        record = producer_validation
    if isinstance(record, Mapping):
        nested_gate = record.get("gate")
        nested_status = record.get("status")
        nested_obligations = record.get("obligations")
    else:
        nested_gate = getattr(record, "gate", None)
        nested_status = getattr(record, "status", None)
        nested_obligations = getattr(record, "obligations", None)
    if isinstance(nested_status, GateStatus):
        nested_status = nested_status.value
    if (
        type(nested_gate) is not str
        or type(nested_status) is not str
        or type(nested_obligations) not in (tuple, list)
        or any(
            type(item) is not str or not item
            for item in nested_obligations
        )
    ):
        raise ValueError(
            f"{gate} producer validation does not expose a typed result identity"
        )
    return nested_gate, nested_status, tuple(nested_obligations)


def _publish_prediction_correctness_artifacts(
    *,
    run_root: Path,
    started_utc: str,
    source_provenance: Mapping[str, object],
    gate_configs: tuple[
        tuple[Literal["H1", "H2", "H3", "H5"], ResolvedConfig], ...
    ],
    gate_results: tuple[
        tuple[
            Literal["H1", "H2", "H3", "H5"],
            GateResult | H3GateResult | H5GateResult,
        ],
        ...,
    ],
    producer_validations: tuple[
        tuple[Literal["H1", "H2", "H3", "H5"], object], ...
    ],
) -> tuple[
    tuple[Literal["H1", "H2", "H3", "H5"], Path, str], ...
]:
    """Publish Prediction-only gate roots from already-computed evaluations."""

    if not isinstance(run_root, Path):
        raise ValueError("prediction correctness run_root must be a Path")
    if type(started_utc) is not str or not started_utc:
        raise ValueError("prediction correctness started_utc must be nonempty")
    if not isinstance(source_provenance, Mapping):
        raise ValueError("prediction correctness source provenance must be a mapping")
    configs = _ordered_prediction_inputs(gate_configs, name="gate_configs")
    results = _ordered_prediction_inputs(gate_results, name="gate_results")
    validations = _ordered_prediction_inputs(
        producer_validations,
        name="producer_validations",
    )
    git_head = source_provenance.get("git_head")
    dirty_digest = source_provenance.get("dirty_digest")
    if (
        type(git_head) is not str
        or len(git_head) != 40
        or any(character not in _LOWER_HEX for character in git_head)
    ):
        raise ValueError("prediction correctness git_head must be lowercase 40-hex")
    if (
        type(dirty_digest) is not str
        or len(dirty_digest) != 64
        or any(character not in _LOWER_HEX for character in dirty_digest)
    ):
        raise ValueError(
            "prediction correctness dirty_digest must be lowercase 64-hex"
        )
    recorded_dirty_content = source_provenance.get("dirty_content_digest")
    if (
        recorded_dirty_content is not None
        and recorded_dirty_content != dirty_digest
    ):
        raise ValueError(
            "prediction correctness source dirty digests disagree"
        )
    source_sha256 = source_candidate_sha256(
        git_head_value=git_head,
        dirty_digest_value=dirty_digest,
    )

    prepared: list[
        tuple[
            Literal["H1", "H2", "H3", "H5"],
            str,
            dict[str, object],
            dict[str, object],
            dict[str, object],
        ]
    ] = []
    for gate in _PREDICTION_CORRECTNESS_GATES:
        config = configs[gate]
        if type(config) is not ResolvedConfig:
            raise ValueError(f"{gate} correctness config has the wrong type")
        config_payload, config_bytes = _raw_config_payload(config)
        config_sha256 = hashlib.sha256(config_bytes).hexdigest()
        result = results[gate]
        if getattr(result, "gate", None) != gate:
            raise ValueError(f"{gate} correctness result gate differs")
        status = getattr(result, "status", None)
        obligations = getattr(result, "obligations", None)
        if not isinstance(status, GateStatus):
            raise ValueError(f"{gate} correctness status is not typed")
        if (
            type(obligations) is not tuple
            or any(type(item) is not str or not item for item in obligations)
        ):
            raise ValueError(
                f"{gate} correctness obligations must be nonempty strings"
            )
        producer_validation = validations[gate]
        if not isinstance(producer_validation, Mapping):
            raise ValueError(
                f"{gate} producer_validation must be a mapping"
            )
        nested_gate, nested_status, nested_obligations = (
            _producer_validation_result_fields(
                gate=gate,
                producer_validation=producer_validation,
            )
        )
        if (
            nested_gate != gate
            or nested_status != status.value
            or nested_obligations != obligations
        ):
            raise ValueError(
                f"{gate} producer validation differs from its typed gate result"
            )
        validation_payload = {
            "schema_version": "vfe4-prediction-correctness-v1",
            "gate": gate,
            "git_head": git_head,
            "dirty_digest": dirty_digest,
            "config_sha256": config_sha256,
            "status": status.value,
            "obligations": obligations,
            "producer_validation": producer_validation,
        }
        provenance_payload = {
            "schema_version": "vfe4-prediction-correctness-provenance-v1",
            "gate": gate,
            "git_head": git_head,
            "dirty_digest": dirty_digest,
            "dirty_content_digest": dirty_digest,
            "source_sha256": source_sha256,
            "config_sha256": config_sha256,
            "status": status.value,
        }
        if gate == "H5":
            h5_fields = (
                "h5_config",
                "h5_state_hashes",
                "h5_update_hash_records",
                "h5_update_binding_preimages",
            )
            missing_h5_fields = tuple(
                name for name in h5_fields if source_provenance.get(name) is None
            )
            if status is GateStatus.PASS and missing_h5_fields:
                raise RuntimeError(
                    "PASS H5 correctness publication lacks producer provenance: "
                    + ", ".join(missing_h5_fields)
                )
            for name in h5_fields:
                value = source_provenance.get(name)
                if value is not None:
                    provenance_payload[name] = value
        prepared.append(
            (
                gate,
                config_sha256,
                config_payload,
                validation_payload,
                provenance_payload,
            )
        )

    references: list[
        tuple[Literal["H1", "H2", "H3", "H5"], Path, str]
    ] = []
    for (
        gate,
        config_sha256,
        config_payload,
        validation_payload,
        provenance_payload,
    ) in prepared:
        root = publish_run_directory(
            run_root,
            _prediction_correctness_run_name(
                started_utc,
                gate,
                config_sha256,
            ),
            {
                "config.json": config_payload,
                "provenance.json": provenance_payload,
                f"validation/{gate.lower()}.json": validation_payload,
            },
        )
        manifest_sha256 = hashlib.sha256(
            (root / "manifest.sha256").read_bytes()
        ).hexdigest()
        references.append((gate, root, manifest_sha256))
    return tuple(references)


def _canonical_config(config: object) -> ResolvedConfig:
    if type(config) is not ResolvedConfig:
        raise ValueError("config must have exact type ResolvedConfig")
    try:
        raw = json.loads(config.canonical_json)
        reproduced = resolve_config(raw, repo_root=REPO_ROOT)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"resolved config cannot be reproduced: {exc}") from exc
    if reproduced != config:
        raise ValueError("resolved config differs from its canonical reproduction")
    return reproduced


def _legacy_projection(config: ResolvedConfig) -> ResolvedConfig:
    """Project a validated prefix onto the unchanged H1/H2 config contract."""

    raw = json.loads(config.canonical_json)
    legacy_gates = ["H1"] if config.validation.gates == ("H1",) else ["H1", "H2"]
    raw["validation"]["gates"] = legacy_gates
    raw.pop("h3", None)
    raw.pop("h4", None)
    raw.pop("h5", None)
    return resolve_config(raw, repo_root=REPO_ROOT)


def _h3_projection(config: ResolvedConfig) -> ResolvedConfig:
    """Project the coupled config onto H3's unchanged exact contract."""

    raw = json.loads(config.canonical_json)
    raw["validation"]["gates"] = ["H1", "H2", "H3"]
    raw.pop("h4", None)
    raw.pop("h5", None)
    return resolve_config(raw, repo_root=REPO_ROOT)


def _aggregate_state(
    results: tuple[GateResult | H3GateResult | H4GateResult | H5GateResult, ...],
) -> GateStatus:
    if any(result.status is GateStatus.FAIL for result in results):
        return GateStatus.FAIL
    if any(result.status is GateStatus.INCONCLUSIVE for result in results):
        return GateStatus.INCONCLUSIVE
    return GateStatus.PASS


def _combined_provenance(
    config: ResolvedConfig,
    h1: H1GateEvaluation,
    h2: H2GateEvaluation | None,
    h3: H3GateEvaluation | None,
    h4: H4GateEvaluation | None,
    h5: H5GateEvaluation | None,
    started_utc: str,
    ended_utc: str,
    candidate_junit_sha256: str | None,
) -> dict[str, object]:
    if h2 is not None and h1.fixture_observed_sha256 != h2.fixture_observed_sha256:
        raise ValueError("ordered legacy gates reported different fixture snapshots")
    evaluations = (
        h1,
        *((h2,) if h2 is not None else ()),
        *((h3,) if h3 is not None else ()),
        *((h4,) if h4 is not None else ()),
        *((h5,) if h5 is not None else ()),
    )
    results = tuple(evaluation.result for evaluation in evaluations)
    provenance = build_provenance(
        repo_root=REPO_ROOT,
        fixture_expected_sha256=EXPECTED_H1_FIXTURE_SHA256,
        fixture_observed_sha256=h1.fixture_observed_sha256,
        config=config,
        started_utc=started_utc,
        ended_utc=ended_utc,
        gate_state=_aggregate_state(results).value,
        candidate_junit_sha256=candidate_junit_sha256,
    )
    provenance["gate_states"] = {
        result.gate: result.status.value for result in results
    }
    provenance["fixture_consumers"] = tuple(
        result.gate for result in results if result.gate in ("H1", "H2", "H5")
    )
    if h3 is not None:
        hashes = h3.fixture_hashes
        provenance["fixture_hashes"] = {
            "h1-v1": {
                "expected_sha256": EXPECTED_H1_FIXTURE_SHA256,
                "observed_sha256": h1.fixture_observed_sha256,
                "hash_domain": "raw_fixture_bytes",
            },
            "h3-coupled-v1": {
                "expected_sha256": hashes.coupled_expected_sha256,
                "observed_sha256": hashes.coupled_observed_sha256,
                "hash_domain": "raw_fixture_bytes",
            },
            "h3-zero-control-v1": {
                "expected_sha256": hashes.zero_control_expected_sha256,
                "observed_sha256": hashes.zero_control_observed_sha256,
                "hash_domain": "raw_fixture_bytes",
            },
        }
        provenance["gate_fixture_consumers"] = {
            "H1": ("h1-v1",),
            "H2": ("h1-v1",),
            "H3": ("h3-coupled-v1", "h3-zero-control-v1"),
        }
        canonical_payload = json.loads(config.canonical_json)
        provenance["h3_profile"] = canonical_payload["h3"]
    if h4 is not None and h5 is not None:
        if config.h4 is None or config.h5 is None:
            raise ValueError("coupled provenance requires typed H4 and H5 config")
        if h4.h4_config_sha256 != config.h4.config_sha256:
            raise ValueError("H4 evaluation/config identity differs")
        h4_projection = json.dumps(
            json.loads(config.canonical_json)["h4"],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        if (
            h4_projection != config.h4.canonical_json
            or hashlib.sha256(h4_projection.encode("utf-8")).hexdigest()
            != config.h4.config_sha256
        ):
            raise ValueError("full resolved config H4 projection drifted")
        h5_raw = h5.result.update_spec_raw_sha256
        if (
            h5.result.preflight.phase is H5PreflightPhase.READY
            and h5_raw != config.h5.update_spec_raw_sha256
        ):
            raise ValueError("H5 evaluation/config raw fixture identity differs")
        provenance["fixture_hashes"]["h5-conditional-update-v1"] = {
            "expected_sha256": config.h5.update_spec_raw_sha256,
            "observed_sha256": h5_raw,
            "hash_domain": "raw_fixture_bytes",
        }
        provenance["gate_fixture_consumers"]["H4"] = (
            "h3-coupled-v1", "h3-zero-control-v1",
        )
        provenance["gate_fixture_consumers"]["H5"] = (
            "h1-v1", "h5-conditional-update-v1",
        )
        provenance["h4_config_sha256"] = config.h4.config_sha256
        provenance["h4_projection_sha256"] = hashlib.sha256(
            h4_projection.encode("utf-8")
        ).hexdigest()
        provenance["full_config_sha256"] = config.config_sha256
        provenance["h4_bounded_claim"] = h4.bounded_claim
        provenance["h4_nonclaims"] = h4.nonclaims
        reference = h5.reference
        provenance["h5_config"] = {
            "config_sha256": config.h5.config_sha256,
            "update_spec_raw_sha256": config.h5.update_spec_raw_sha256,
            "update_spec_canonical_sha256": h5.result.update_spec_canonical_sha256,
            "objective_schema_sha256": h5.result.objective_schema_sha256,
            "factor_input_schema_version": h5.result.factor_input_schema_version,
            "factor_input_schema_sha256": h5.result.factor_input_schema_sha256,
            "recognition_family": config.h5.recognition_family,
            "factor_universe": config.h5.factor_universe,
            "recognition_coordinate_universe": config.h5.recognition_coordinate_universe,
            "model_block_universe": config.h5.model_block_universe,
            "enabled_update_rules": tuple(item.value for item in config.h5.enabled_update_rules),
            "enabled_update_labels": tuple(item.value for item in config.h5.enabled_update_labels),
            "positive_case_ids": config.h5.positive_case_ids,
            "control_ids": config.h5.control_ids,
            "quadrature_orders": config.h5.quadrature_orders,
            "allowance_policy": config.h5.allowance_policy,
            "rounding_constant": config.h5.rounding_constant,
            "stochastic_contribution": config.h5.stochastic_contribution,
            "epsilon_delta_formula": config.h5.epsilon_delta_formula,
        }
        provenance["h5_state_hashes"] = {
            "reference_sha256": h5.result.reference_sha256,
            "recognition_sha256": (
                reference.initial_recognition.state_sha256 if reference is not None else None
            ),
            "model_sha256": (
                reference.initial_model.state_sha256 if reference is not None else None
            ),
            "validation_payload_sha256": h5.validation_payload.payload_sha256,
        }
        provenance["h5_update_hash_records"] = {
            "positive": tuple(
                {
                    "case_id": item.case_id.value,
                    "hashes": asdict(item.outcome.hashes),
                }
                for item in (h5.result.positive_cases or ())
            ),
            "controls": tuple(
                {
                    "control_id": item.control_id.value,
                    "hashes": asdict(item.outcome.hashes),
                }
                for item in (h5.result.controls or ())
            ),
        }
        provenance["h5_bounded_claim"] = (
            "deterministic complete-objective update coherence for the frozen H5 cases"
        )
        provenance["h5_nonclaims"] = h5.validation_payload.nonclaims
        if (
            h5.result.preflight.phase is H5PreflightPhase.READY
            and h5.reference is not None
        ):
            provenance["h5_update_binding_preimages"] = (
                h5_update_binding_preimages(h5)
            )
    return provenance


def run_h7_verification(config: ResolvedConfig) -> VerificationRunResult:
    """Publish one reference-only H7 result without running predecessor gates."""

    canonical = _canonical_config(config)
    if canonical.validation.gates != H7_VERIFICATION_PREFIX or canonical.h7 is None:
        raise ValueError("run_h7_verification requires the exact H7 operation")

    started = _utc_now()
    git_head_value, dirty_digest_value, source_sha256_value = current_source_identity(
        REPO_ROOT,
        canonical.artifacts.run_root,
    )
    registry_path = (
        REPO_ROOT
        / ".verification"
        / f"h7-current-candidate-{git_head_value}-refs.json"
    )
    if (
        not registry_path.is_file()
        or registry_path.is_symlink()
        or registry_path.parent.resolve(strict=False)
        != (REPO_ROOT / ".verification").resolve(strict=False)
    ):
        raise ValueError(
            "the exact current-candidate H7 reference registry is unavailable"
        )
    registry_bytes = registry_path.read_bytes()
    predecessor_entries = parse_h7_reference_registry_bytes(registry_bytes)
    references = {
        key: reference for key, reference in predecessor_entries
    }
    junit_sha256 = predecessor_entries[0][1].junit_sha256

    h1_fixture_bytes = FIXTURE_PATH.read_bytes()
    h7_fixture_bytes = H7_FIXTURE_PATH.read_bytes()
    captured_fixture_bytes = {
        H7_CAPTURED_FIXTURE_PATHS[0]: h1_fixture_bytes,
        H7_CAPTURED_FIXTURE_PATHS[1]: h7_fixture_bytes,
    }
    fixture_observed_sha256 = {
        "h1_fixture_raw_sha256": hashlib.sha256(h1_fixture_bytes).hexdigest(),
        "h7_fixture_raw_sha256": hashlib.sha256(h7_fixture_bytes).hexdigest(),
        "density_probe_table_raw_sha256": (
            canonical.h7.density_probe_table_raw_sha256
        ),
        "density_probe_set_sha256": canonical.h7.density_probe_set_sha256,
    }
    evaluation = assemble_h7_gate_evaluation(
        repo_root=REPO_ROOT,
        captured_fixture_bytes=captured_fixture_bytes,
        predecessor_entries=predecessor_entries,
        git_head=git_head_value,
        dirty_digest=dirty_digest_value,
        junit_sha256=junit_sha256,
        scorer_profile=H7_ACTIVE_SCORER_PROFILE,
        fixture_hashes={
            "density_probe_table_raw_sha256": (
                canonical.h7.density_probe_table_raw_sha256
            ),
            "density_probe_set_sha256": (
                canonical.h7.density_probe_set_sha256
            ),
        },
        trials=(),
        controls=(),
        oracle_obligations=H7_SOURCE_ONLY_OBLIGATIONS,
    )
    ended = _utc_now()
    provenance = build_h7_provenance(
        config=canonical,
        evaluation=evaluation,
        git_head_value=git_head_value,
        dirty_digest_value=dirty_digest_value,
        source_sha256_value=source_sha256_value,
        junit_sha256=junit_sha256,
        reference_registry_path=registry_path,
        reference_registry_sha256=hashlib.sha256(registry_bytes).hexdigest(),
        fixture_expected_sha256=H7_FROZEN_SOURCE_FIXTURE_HASHES,
        fixture_observed_sha256=fixture_observed_sha256,
        predecessor_references=references,
        scorer_profile=H7_ACTIVE_SCORER_PROFILE,
        nonclaims=H7_NONCLAIMS,
        budget_constants={
            "eps64": EPS64,
            "rounding_constant": ROUNDING_CONSTANT,
            "maximum_oracle_relative_delta": str(MAX_ORACLE_RELATIVE_DELTA),
            "control_minimum_relative_residual": (
                CONTROL_MINIMUM_RELATIVE_RESIDUAL
            ),
            "control_allowance_multiple": CONTROL_ALLOWANCE_MULTIPLE,
            "policy": "category-and-operand-local-v1",
        },
        started_utc=started,
        ended_utc=ended,
    )
    reference_payloads = {
        f"references/{key}.json": json.loads(canonical_json_bytes(reference))
        for key, reference in predecessor_entries
    }
    payloads = {
        "config.json": _config_payload(canonical),
        "provenance.json": provenance,
        "environment.json": build_environment(canonical),
        **reference_payloads,
        "validation/h7.json": h7_validation_payload(evaluation),
    }
    run_directory = publish_run_directory(
        canonical.artifacts.run_root,
        _run_name(started, canonical.config_sha256, ("H7",)),
        payloads,
    )
    return VerificationRunResult((evaluation.result,), run_directory)


def _canonical_h8_config(config: object) -> ResolvedConfig:
    if type(config) is not ResolvedConfig:
        raise ValueError("H8 config must have exact type ResolvedConfig")
    refs = config.h8_current_refs
    if type(refs) is not CurrentH8PrerequisiteRefs:
        raise ValueError("H8 config requires exact bound current references")
    try:
        raw = json.loads(config.canonical_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("H8 canonical config is not JSON") from exc
    if not isinstance(raw, dict):
        raise ValueError("H8 canonical config must be one object")
    if (
        canonical_h8_json_bytes(raw) != config.canonical_json.encode("utf-8")
        or hashlib.sha256(config.canonical_json.encode("utf-8")).hexdigest()
        != config.config_sha256
        or raw.get("h8_current_refs")
        != json.loads(canonical_h8_json_bytes(refs))
    ):
        raise ValueError("H8 canonical config lost its bound references")
    refs.__post_init__()
    return config


def _h8_dependency_closure_sha256(
    *,
    source_sha256: str,
    config_sha256: str,
    registry_sha256: str,
    a0_direct_exact_prefix_certificate_sha256: str,
    preregistration_sha256: str,
) -> str:
    return hashlib.sha256(
        canonical_h8_json_bytes(
            {
                "domain": "vfe4.h8.selected-runtime-dependency-closure.v1",
                "source_sha256": source_sha256,
                "config_sha256": config_sha256,
                "registry_sha256": registry_sha256,
                "a0_direct_exact_prefix_certificate_sha256": (
                    a0_direct_exact_prefix_certificate_sha256
                ),
                "preregistration_sha256": preregistration_sha256,
            }
        )
    ).hexdigest()


def _h8_published_reference(
    run_directory: Path,
    *,
    git_head_value: str,
    dirty_digest_value: str,
) -> CandidateArtifactReference:
    manifest_path = run_directory / "manifest.sha256"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("published H8 artifact lacks a regular manifest")
    manifest_bytes = manifest_path.read_bytes()
    if not manifest_bytes.endswith(b"\n"):
        raise ValueError("published H8 manifest is not newline terminated")
    lines = manifest_bytes.decode("ascii", errors="strict").splitlines()
    manifest_names = tuple(sorted(H8_PUBLICATION_PAYLOAD_KEYS))
    if len(lines) != len(manifest_names):
        raise ValueError("published H8 manifest inventory is not exact")
    payload_hashes: dict[str, str] = {}
    for expected_name, line in zip(manifest_names, lines, strict=True):
        digest, separator, name = line.partition("  ")
        if (
            separator != "  "
            or name != expected_name
            or len(digest) != 64
            or any(character not in _LOWER_HEX for character in digest)
        ):
            raise ValueError("published H8 manifest entry is invalid")
        payload_path = run_directory / Path(*name.split("/"))
        if (
            not payload_path.is_file()
            or payload_path.is_symlink()
            or hashlib.sha256(payload_path.read_bytes()).hexdigest() != digest
        ):
            raise ValueError(f"published H8 payload hash mismatch: {name}")
        payload_hashes[name] = digest
    return CandidateArtifactReference(
        artifact_path=run_directory,
        git_head=git_head_value,
        dirty_digest=dirty_digest_value,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        payload_hashes=payload_hashes,
    )


def run_h8_verification(
    config: ResolvedConfig,
    *,
    registry_path: Path,
    registry_bytes: bytes,
) -> VerificationRunResult:
    """Run and publish the exact selected H8 parent-orchestrated operation."""

    canonical = _canonical_h8_config(config)
    refs = canonical.h8_current_refs
    if (
        canonical.validation.gates != H8_VERIFIER_PREFIX
        or canonical.h8 is None
        or type(refs) is not CurrentH8PrerequisiteRefs
        or refs.registry_schema_version != "h8-current-candidate-refs-v5"
        or type(refs.h6_prefix) is not H8H6PrefixReference
        or type(refs.h6_prediction) is not H8H6PredictionV3Reference
        or refs.prerequisite_obligations
    ):
        raise ValueError("run_h8_verification requires the exact bound H8 operation")

    started = _utc_now()
    git_head_value, dirty_digest_value, source_sha256_value = current_source_identity(
        REPO_ROOT,
        canonical.artifacts.run_root,
    )
    if (
        refs.candidate_head != git_head_value
        or refs.candidate_dirty_digest != dirty_digest_value
    ):
        raise ValueError("bound H8 references do not describe the current source")
    expected_registry_path = (
        REPO_ROOT
        / ".verification"
        / f"h8-current-candidate-{git_head_value}-refs.json"
    ).resolve(strict=False)
    if (
        not isinstance(registry_path, Path)
        or registry_path.resolve(strict=False) != expected_registry_path
        or type(registry_bytes) is not bytes
        or not registry_bytes
        or hashlib.sha256(registry_bytes).hexdigest() != refs.registry_sha256
        or registry_bytes
        != canonical_h8_json_bytes(h8_current_refs_registry_payload(refs))
    ):
        raise ValueError("captured H8 registry differs from the bound current refs")
    prerequisite_validation = validate_h8_prerequisite_artifacts(refs)
    if (
        not H8_PREREGISTRATION_PATH.is_file()
        or H8_PREREGISTRATION_PATH.is_symlink()
    ):
        raise ValueError("H8 preregistration is unavailable")
    preregistration_sha256 = hashlib.sha256(
        H8_PREREGISTRATION_PATH.read_bytes()
    ).hexdigest()
    dependency_closure_sha256 = _h8_dependency_closure_sha256(
        source_sha256=source_sha256_value,
        config_sha256=canonical.config_sha256,
        registry_sha256=refs.registry_sha256,
        a0_direct_exact_prefix_certificate_sha256=(
            refs.h6_prefix.a0_direct_exact_prefix_certificate_sha256
        ),
        preregistration_sha256=preregistration_sha256,
    )
    startup_environment = require_h8_startup_environment(os.environ)
    correctness = produce_h8_correctness_grid()
    if require_h8_startup_environment(os.environ) != startup_environment:
        raise ValueError(
            "H8 startup environment drifted during correctness execution"
        )
    authorization = derive_h8_child_start_authorization(
        config=canonical.h8,
        current_registry_sha256=refs.registry_sha256,
        prerequisite_validation=prerequisite_validation,
        correctness_statuses=tuple(
            (cell.cell_id, cell.status)
            for cell in correctness
        ),
    )
    if not authorization.valid_start:
        evaluation = assemble_h8_source_only_evaluation(
            config_sha256=canonical.config_sha256,
            current_refs=refs,
            correctness=correctness,
            production_runs=(),
            profiler_runs=(),
            controls=(),
            dependency_closure_sha256=dependency_closure_sha256,
            preregistration_sha256=preregistration_sha256,
            additional_obligations=authorization.obligations,
            prerequisite_validation=prerequisite_validation,
        )
    else:
        parent_authority = run_h8_parent_attempt(
            authorization=authorization,
            repository_root=REPO_ROOT,
        )
        evaluation = assemble_h8_gate_evaluation(
            publication_config_sha256=canonical.config_sha256,
            current_refs=refs,
            correctness=correctness,
            parent_authority=parent_authority,
            dependency_closure_sha256=dependency_closure_sha256,
            preregistration_sha256=preregistration_sha256,
            prerequisite_validation=prerequisite_validation,
        )
    ended = _utc_now()
    environment = build_h8_environment(
        config=canonical,
        validation_environment=json.loads(
            evaluation.validation_payload_canonical_json
        )["environment"],
    )
    provenance = build_h8_provenance(
        config=canonical,
        evaluation=evaluation,
        git_head_value=git_head_value,
        dirty_digest_value=dirty_digest_value,
        source_sha256_value=source_sha256_value,
        reference_registry_path=registry_path,
        reference_registry_sha256=refs.registry_sha256,
        started_utc=started,
        ended_utc=ended,
    )
    payloads = build_h8_publication_payloads(
        canonical,
        evaluation,
        h7_reference=refs.h7,
        h6_prediction_reference=refs.h6_prediction,
        provenance=provenance,
        environment=environment,
    )
    run_directory = publish_run_directory(
        canonical.artifacts.run_root,
        _run_name(started, canonical.config_sha256, ("H8",)),
        payloads,
    )
    artifact = _h8_published_reference(
        run_directory,
        git_head_value=git_head_value,
        dirty_digest_value=dirty_digest_value,
    )
    # Revalidate the complete published reference and construct the external
    # pointer value in memory only. Task 8 owns its one-time external write.
    h8_current_candidate_result_payload(
        artifact,
        repo_root=REPO_ROOT,
        config_sha256=canonical.config_sha256,
        validation_sha256=evaluation.validation_payload_sha256,
        junit_sha256=refs.candidate_junit_sha256,
        current_refs=refs,
        evaluation=evaluation,
        source_sha256=source_sha256_value,
        registry_path=registry_path,
        registry_bytes=registry_bytes,
    )
    return VerificationRunResult((evaluation.result,), run_directory)


def run_verification(
    config: ResolvedConfig,
    *,
    publish_prediction_correctness: bool = False,
    candidate_junit_sha256: str | None = None,
) -> VerificationRunResult:
    """Evaluate one implemented prefix from one capture set and publish once."""

    if type(publish_prediction_correctness) is not bool:
        raise ValueError("publish_prediction_correctness must be a boolean")
    if candidate_junit_sha256 is not None and (
        type(candidate_junit_sha256) is not str
        or len(candidate_junit_sha256) != 64
        or any(
            character not in _LOWER_HEX
            for character in candidate_junit_sha256
        )
    ):
        raise ValueError(
            "candidate_junit_sha256 must be None or lowercase 64-hex"
        )
    canonical = _canonical_config(config)
    gates = canonical.validation.gates
    if gates not in _ALLOWED_PREFIXES:
        raise ValueError("run_verification requires an implemented ordered gate prefix")
    if (
        publish_prediction_correctness
        and gates != ("H1", "H2", "H3", "H4", "H5")
    ):
        raise ValueError(
            "Prediction correctness publication requires the full H1--H5 "
            "evaluation operation"
        )
    legacy = _legacy_projection(canonical)
    h3_config = _h3_projection(canonical) if "H3" in gates else None
    started = _utc_now()

    h1_bytes = FIXTURE_PATH.read_bytes()
    coupled_bytes: bytes | None = None
    zero_control_bytes: bytes | None = None
    h5_update_spec_bytes: bytes | None = None
    observed_h5_sha256: str | None = None
    expected_h5_sha256: str | None = None
    h5_update_spec_digest_matches: bool | None = None
    if "H3" in gates:
        coupled_bytes = H3_COUPLED_FIXTURE_PATH.read_bytes()
        zero_control_bytes = H3_ZERO_CONTROL_FIXTURE_PATH.read_bytes()
    if gates == ("H1", "H2", "H3", "H4", "H5"):
        if canonical.h5 is None:
            raise ValueError("coupled prefix lacks its typed H5 config")
        h5_update_spec_bytes = H5_UPDATE_SPEC_FIXTURE_PATH.read_bytes()
        observed_h5_sha256 = hashlib.sha256(h5_update_spec_bytes).hexdigest()
        expected_h5_sha256 = canonical.h5.update_spec_raw_sha256
        h5_update_spec_digest_matches = observed_h5_sha256 == expected_h5_sha256

    h1 = evaluate_h1(legacy, fixture_bytes=h1_bytes)
    h2: H2GateEvaluation | None = None
    h3: H3GateEvaluation | None = None
    h4: H4GateEvaluation | None = None
    h5: H5GateEvaluation | None = None
    results: list[GateResult | H3GateResult | H4GateResult | H5GateResult] = [h1.result]
    validation_payloads: dict[str, object] = {
        "validation/h1.json": h1.validation_payload,
    }
    if "H2" in gates:
        h2 = evaluate_h2(legacy, fixture_bytes=h1_bytes)
        results.append(h2.result)
        validation_payloads["validation/h2.json"] = h2_validation_payload(h2)
    if "H3" in gates:
        if coupled_bytes is None or zero_control_bytes is None:
            raise RuntimeError("H3 fixture capture is unavailable")
        if h3_config is None:
            raise RuntimeError("H3 config projection is unavailable")
        h3 = evaluate_h3(
            h3_config,
            coupled_fixture_bytes=coupled_bytes,
            zero_control_fixture_bytes=zero_control_bytes,
        )
        results.append(h3.result)
        validation_payloads["validation/h3.json"] = h3_validation_payload(h3)
    if "H4" in gates:
        if canonical.h4 is None or coupled_bytes is None or zero_control_bytes is None:
            raise RuntimeError("H4 typed config or captured anchor bytes are unavailable")
        h4 = evaluate_h4(
            canonical.h4,
            h3_coupled_bytes=coupled_bytes,
            h3_zero_bytes=zero_control_bytes,
        )
        h4_artifact = h4_validation_artifact(h4)
        results.append(h4.result)
        validation_payloads["validation/h4.json"] = h4_validation_payload(h4_artifact)
    if "H5" in gates:
        if (
            h5_update_spec_bytes is None
            or observed_h5_sha256 is None
            or expected_h5_sha256 is None
            or h5_update_spec_digest_matches is None
        ):
            raise RuntimeError("H5 update-spec preflight capture is unavailable")
        h5 = evaluate_h5(
            canonical,
            h1_fixture_bytes=h1_bytes,
            h5_update_spec_bytes=h5_update_spec_bytes,
        )
        if (
            h5.result.update_spec_raw_sha256 != observed_h5_sha256
            or (
                h5.result.update_spec_raw_sha256 == expected_h5_sha256
            ) is not h5_update_spec_digest_matches
        ):
            raise RuntimeError("H5 typed result differs from runner raw digest preflight")
        results.append(h5.result)
        validation_payloads["validation/h5.json"] = h5_validation_payload(h5)

    frozen_results = tuple(results)
    ended = _utc_now()
    provenance = _combined_provenance(
        canonical,
        h1,
        h2,
        h3,
        h4,
        h5,
        started,
        ended,
        candidate_junit_sha256,
    )
    payloads = {
        "config.json": _config_payload(canonical),
        "provenance.json": provenance,
        "environment.json": build_environment(canonical),
        **validation_payloads,
    }
    run_directory = publish_run_directory(
        canonical.artifacts.run_root,
        _run_name(started, canonical.config_sha256, gates),
        payloads,
    )
    correctness_artifacts: tuple[
        tuple[Literal["H1", "H2", "H3", "H5"], Path, str], ...
    ] = ()
    if publish_prediction_correctness:
        if h2 is None or h3 is None or h3_config is None or h5 is None:
            raise RuntimeError(
                "Prediction correctness publication lacks computed H1/H2/H3/H5 "
                "evaluations"
            )
        correctness_artifacts = _publish_prediction_correctness_artifacts(
            run_root=canonical.artifacts.run_root,
            started_utc=started,
            source_provenance=provenance,
            gate_configs=(
                ("H1", legacy),
                ("H2", legacy),
                ("H3", h3_config),
                ("H5", canonical),
            ),
            gate_results=(
                ("H1", h1.result),
                ("H2", h2.result),
                ("H3", h3.result),
                ("H5", h5.result),
            ),
            producer_validations=(
                ("H1", validation_payloads["validation/h1.json"]),
                ("H2", validation_payloads["validation/h2.json"]),
                ("H3", validation_payloads["validation/h3.json"]),
                ("H5", validation_payloads["validation/h5.json"]),
            ),
        )
    return VerificationRunResult(
        frozen_results,
        run_directory,
        correctness_artifacts,
    )


__all__ = [
    "VerificationRunResult",
    "candidate_artifact_reference_to_h7_reference",
    "h7_reference_registry_bytes",
    "parse_h8_reference_registry_bytes",
    "parse_h7_reference_registry_bytes",
    "run_h7_verification",
    "run_h8_verification",
    "run_verification",
]
