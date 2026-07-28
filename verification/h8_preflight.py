"""Read-only, zero-scientific-compute preflight for the frozen H8 protocol."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import stat
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal, cast


H8_PREFLIGHT_CONFIG_SCHEMA = "h8-preflight-config-v1"
H8_PREFLIGHT_RESULT_SCHEMA = "h8-preflight-result-v1"
H8_REGISTRY_SCHEMA = "h8-current-candidate-refs-v5"
H8_FROZEN_SECTION_SHA256 = (
    "c11969f7e27bc4835f1768ee6757c48d46626718ec387d50a27808f8d35373bb"
)
H8_PREREGISTRATION_RELATIVE_PATH = (
    Path("docs") / "preregistrations" / "2026-07-21-h8-sparse-scale.md"
)
H8_REFERENCE_KEYS = (
    "h1_h5",
    "h1_prefix_prior",
    "h6_prefix",
    "h7",
    "h6_prediction",
)
H8_COMPATIBILITY_KEYS = ("h1_h5", "h1_prefix_prior", "h6_prefix")
H8_PREDICTION_V3_SCHEMAS = {
    "config_schema": "h6-prediction-config-v3",
    "readiness_schema": "h6-prediction-readiness-v3",
    "raw_inventory_schema": "h6-raw-endpoint-inventory-v4",
    "metrics_schema": "h6-prediction-metrics-v3",
    "result_schema": "h6-prediction-result-v3",
}
H8_SELECTED_RUNTIME_CALL_NAMES = (
    "validate_h8_prerequisite_artifacts",
    "produce_h8_correctness_grid",
    "derive_h8_child_start_authorization",
    "run_h8_parent_attempt",
    "assemble_h8_gate_evaluation",
)
H8_SELECTED_GATE_ARGUMENT_NAMES = (
    "correctness",
    "parent_authority",
    "prerequisite_validation",
)
H7_COMPATIBILITY_FIELDS = frozenset(
    {
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
)
H8_REFERENCE_COMMON_FIELDS = frozenset(
    {
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
)
H8_REFERENCE_FIELDS = {
    "h1_h5": H8_REFERENCE_COMMON_FIELDS,
    "h1_prefix_prior": H8_REFERENCE_COMMON_FIELDS,
    "h6_prefix": H8_REFERENCE_COMMON_FIELDS
    | frozenset(
        {
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
    ),
    "h7": H8_REFERENCE_COMMON_FIELDS
    | frozenset(
        {
            "result_pointer_path",
            "result_pointer_sha256",
            "fixture_set_sha256",
        }
    ),
    "h6_prediction": H8_REFERENCE_COMMON_FIELDS
    | frozenset(H8_PREDICTION_V3_SCHEMAS)
    | frozenset(
        {
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
    ),
}

PrerequisiteState = Literal[
    "missing",
    "present_unvalidated",
    "malformed",
    "stale",
    "blocked",
]
PreflightDisposition = Literal["blocked", "metadata_complete_unvalidated"]


def _canonical_value(value: object) -> object:
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("nonfinite floats are not canonical JSON values")
        return value
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ValueError("canonical mappings require string keys")
        return {cast(str, key): _canonical_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonical_value(item) for item in value]
    raise ValueError(f"unsupported canonical preflight value: {type(value).__name__}")


def canonical_h8_preflight_bytes(value: object) -> bytes:
    """Return deterministic UTF-8 JSON bytes for one preflight record."""

    return json.dumps(
        _canonical_value(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {cast(str, key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _lower_hex(value: object, length: int, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be {length}-character lowercase hex")
    return value


def _path_text(path: Path) -> str:
    return str(path.resolve(strict=False))


def _require_regular_nonsymlink_file(path: Path, *, name: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise ValueError(f"{name} must be an existing regular non-symlink file") from exc
    if not stat.S_ISREG(mode):
        raise ValueError(f"{name} must be an existing regular non-symlink file")


def read_h8_exact_test_nodes(
    path: Path,
    *,
    repository_root: Path,
) -> tuple[tuple[str, ...], bytes, str]:
    """Read and validate the ordered exact-node manifest for H8 verification."""

    try:
        root = Path(repository_root).resolve(strict=True)
    except OSError as exc:
        raise ValueError("repository_root must be an existing directory") from exc
    if not root.is_dir():
        raise ValueError("repository_root must be an existing directory")

    manifest_path = Path(path)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    _require_regular_nonsymlink_file(
        manifest_path,
        name="exact-test-node manifest",
    )
    try:
        original_bytes = manifest_path.read_bytes()
    except OSError as exc:
        raise ValueError("exact-test-node manifest must be readable") from exc

    if original_bytes.startswith(b"\xef\xbb\xbf"):
        raise ValueError("exact-test-node manifest must not contain a UTF-8 BOM")
    if b"\r" in original_bytes:
        raise ValueError("exact-test-node manifest must use LF-only newlines")
    if not original_bytes.endswith(b"\n") or original_bytes.endswith(b"\n\n"):
        raise ValueError(
            "exact-test-node manifest must end with exactly one terminal LF"
        )
    try:
        text = original_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("exact-test-node manifest must be strict UTF-8") from exc

    entries = text[:-1].split("\n")
    if not entries or entries == [""]:
        raise ValueError("exact-test-node manifest must contain at least one entry")

    nodes: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        if not entry:
            raise ValueError("exact-test-node manifest must not contain blank lines")
        if entry.startswith("#"):
            raise ValueError("exact-test-node manifest must not contain comments")
        if entry in seen:
            raise ValueError(f"duplicate exact-test-node entry: {entry}")
        seen.add(entry)
        if any(character.isspace() or not character.isprintable() for character in entry):
            raise ValueError("exact-test-node entries must not contain whitespace or controls")
        if "*" in entry or "?" in entry:
            raise ValueError("exact-test-node entries must not contain wildcards")
        if not entry.startswith("tests/"):
            raise ValueError("exact-test-node entries must start with tests/")
        if "::" not in entry:
            raise ValueError(
                "exact-test-node entries must select a .py file and exact node"
            )

        file_selector, node_selector = entry.split("::", 1)
        path_parts = file_selector.split("/")
        if (
            not file_selector.endswith(".py")
            or "\\" in file_selector
            or ":" in file_selector
            or any(part in ("", ".", "..") for part in path_parts)
        ):
            raise ValueError(
                "exact-test-node entries must select a traversal-free .py file"
            )
        node_parts = node_selector.split("::")
        if (
            any(not part or part in (".", "..") or part.startswith("-") for part in node_parts)
            or "/" in node_selector
            or "\\" in node_selector
        ):
            raise ValueError(
                "exact-test-node entries must select one exact node without options"
            )

        test_path = root.joinpath(*path_parts)
        _require_regular_nonsymlink_file(
            test_path,
            name=f"selected test file {file_selector}",
        )
        try:
            resolved_test_path = test_path.resolve(strict=True)
        except OSError as exc:
            raise ValueError(
                f"selected test file {file_selector} must resolve inside repository_root"
            ) from exc
        if not resolved_test_path.is_relative_to(root):
            raise ValueError(
                f"selected test file {file_selector} must resolve inside repository_root"
            )
        nodes.append(entry)

    return (
        tuple(nodes),
        original_bytes,
        hashlib.sha256(original_bytes).hexdigest(),
    )


@dataclass(frozen=True, slots=True)
class H8PreflightPrerequisite:
    """One read-only prerequisite observation."""

    name: str
    state: PrerequisiteState
    path: str | None
    expected_schema: str | None
    detail: str

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise ValueError("preflight prerequisite name must be nonempty")
        if self.state not in (
            "missing",
            "present_unvalidated",
            "malformed",
            "stale",
            "blocked",
        ):
            raise ValueError("preflight prerequisite state is unsupported")
        if self.path is not None and (type(self.path) is not str or not self.path):
            raise ValueError("preflight prerequisite path must be null or nonempty")
        if self.expected_schema is not None and (
            type(self.expected_schema) is not str or not self.expected_schema
        ):
            raise ValueError("preflight prerequisite schema must be null or nonempty")
        if type(self.detail) is not str or not self.detail:
            raise ValueError("preflight prerequisite detail must be nonempty")

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "state": self.state,
            "path": self.path,
            "expected_schema": self.expected_schema,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class H8PreflightResult:
    """Advisory metadata result; deliberately not an H8 gate result."""

    schema_version: Literal["h8-preflight-result-v1"]
    operation: Literal["H8-Preflight"]
    disposition: PreflightDisposition
    scientific_status: Literal["not_evaluated"]
    candidate: Mapping[str, str]
    target_config_sha256: str
    prerequisites: tuple[H8PreflightPrerequisite, ...]
    workload_forecast: Mapping[str, object]
    resource_forecast: Mapping[str, object]
    execution_policy: Mapping[str, object]
    obligations: tuple[str, ...]
    result_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != H8_PREFLIGHT_RESULT_SCHEMA:
            raise ValueError("preflight result schema is unsupported")
        if self.operation != "H8-Preflight":
            raise ValueError("preflight result operation is unsupported")
        if self.disposition not in (
            "blocked",
            "metadata_complete_unvalidated",
        ):
            raise ValueError("preflight result disposition is unsupported")
        if self.scientific_status != "not_evaluated":
            raise ValueError("preflight cannot declare a scientific status")
        _validate_candidate(self.candidate)
        _lower_hex(
            self.target_config_sha256,
            64,
            "target_config_sha256",
        )
        _lower_hex(self.result_sha256, 64, "result_sha256")
        if not self.prerequisites:
            raise ValueError("preflight prerequisites cannot be empty")
        if len({item.name for item in self.prerequisites}) != len(self.prerequisites):
            raise ValueError("preflight prerequisite names must be unique")
        object.__setattr__(
            self,
            "candidate",
            cast(Mapping[str, str], _freeze_json(self.candidate)),
        )
        object.__setattr__(
            self,
            "workload_forecast",
            cast(Mapping[str, object], _freeze_json(self.workload_forecast)),
        )
        object.__setattr__(
            self,
            "resource_forecast",
            cast(Mapping[str, object], _freeze_json(self.resource_forecast)),
        )
        object.__setattr__(
            self,
            "execution_policy",
            cast(Mapping[str, object], _freeze_json(self.execution_policy)),
        )

    def as_dict(self, *, include_result_sha256: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "operation": self.operation,
            "disposition": self.disposition,
            "scientific_status": self.scientific_status,
            "candidate": dict(self.candidate),
            "target_config_sha256": self.target_config_sha256,
            "prerequisites": [item.as_dict() for item in self.prerequisites],
            "workload_forecast": _canonical_value(self.workload_forecast),
            "resource_forecast": _canonical_value(self.resource_forecast),
            "execution_policy": _canonical_value(self.execution_policy),
            "obligations": list(self.obligations),
        }
        if include_result_sha256:
            payload["result_sha256"] = self.result_sha256
        return payload


def _validate_request(request: Mapping[str, object]) -> None:
    expected = {
        "schema_version",
        "operation",
        "target_operation",
        "inspection_policy",
        "write_artifact",
    }
    if set(request) != expected:
        raise ValueError("H8 preflight request has unknown or missing keys")
    if request["schema_version"] != H8_PREFLIGHT_CONFIG_SCHEMA:
        raise ValueError("H8 preflight config schema is unsupported")
    if request["operation"] != "H8-Preflight":
        raise ValueError("H8 preflight operation must be H8-Preflight")
    if request["target_operation"] != "h8":
        raise ValueError("H8 preflight target_operation must be h8")
    if request["inspection_policy"] != "metadata_only":
        raise ValueError("H8 preflight inspection_policy must be metadata_only")
    if request["write_artifact"] is not False:
        raise ValueError("H8 preflight write_artifact must be false")


def _validate_candidate(candidate: Mapping[str, object]) -> dict[str, str]:
    if set(candidate) != {"git_head", "dirty_digest", "source_sha256"}:
        raise ValueError("H8 preflight candidate has unknown or missing keys")
    return {
        "git_head": _lower_hex(candidate["git_head"], 40, "candidate.git_head"),
        "dirty_digest": _lower_hex(
            candidate["dirty_digest"],
            64,
            "candidate.dirty_digest",
        ),
        "source_sha256": _lower_hex(
            candidate["source_sha256"],
            64,
            "candidate.source_sha256",
        ),
    }


def capture_current_candidate(
    *,
    repository_root: Path,
    target_scientific_config: Mapping[str, object],
) -> dict[str, str]:
    """Capture current Git/source identity without importing scientific runners."""

    root = repository_root.resolve(strict=False)
    artifacts = _mapping(
        target_scientific_config.get("artifacts"),
        "target_scientific_config.artifacts",
    )
    run_root_value = artifacts.get("run_root")
    if type(run_root_value) is not str or not run_root_value:
        raise ValueError("target artifacts.run_root must be a nonempty path")
    run_root = Path(run_root_value)
    if not run_root.is_absolute():
        run_root = root / run_root
    configured_run_root = run_root.resolve(strict=False)
    if configured_run_root == root or configured_run_root in root.parents:
        raise ValueError("target artifacts.run_root cannot contain the repository")
    for control_name in (".git", ".verification"):
        control = (root / control_name).resolve(strict=False)
        if configured_run_root == control or control in configured_run_root.parents:
            raise ValueError("target artifacts.run_root enters a control tree")

    def git_bytes(*arguments: str) -> bytes:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Git metadata capture failed: {detail}")
        return completed.stdout

    git_head = git_bytes("rev-parse", "HEAD").decode("ascii", errors="strict").strip()
    _lower_hex(git_head, 40, "current Git HEAD")
    try:
        tracked = {
            item
            for item in git_bytes("ls-files", "--cached", "-z")
            .decode("utf-8", errors="strict")
            .split("\0")
            if item
        }
        untracked = {
            item
            for item in git_bytes(
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
            )
            .decode("utf-8", errors="strict")
            .split("\0")
            if item
        }
    except UnicodeError as exc:
        raise RuntimeError("Git paths are not strict UTF-8") from exc
    excluded_run_root = (
        configured_run_root if root in configured_run_root.parents else None
    )
    digest = hashlib.sha256()
    for name in sorted(tracked | untracked):
        relative = Path(name)
        normalized = relative.as_posix()
        if (
            normalized == ".git"
            or normalized.startswith(".git/")
            or normalized == ".verification"
            or normalized.startswith(".verification/")
        ):
            continue
        absolute = (root / relative).resolve(strict=False)
        if root != absolute and root not in absolute.parents:
            raise RuntimeError(f"Git path escapes the repository: {name}")
        if (
            name not in tracked
            and excluded_run_root is not None
            and (absolute == excluded_run_root or excluded_run_root in absolute.parents)
        ):
            continue
        try:
            content = absolute.read_bytes()
        except FileNotFoundError:
            content = b"<deleted>"
        encoded_name = normalized.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    dirty_digest = digest.hexdigest()
    source_sha256 = hashlib.sha256(
        b"VFE4-H6-SOURCE-CANDIDATE-V1\x00"
        + bytes.fromhex(git_head)
        + bytes.fromhex(dirty_digest)
    ).hexdigest()
    return {
        "git_head": git_head,
        "dirty_digest": dirty_digest,
        "source_sha256": source_sha256,
    }


def _validate_target_h8(
    target_scientific_config: Mapping[str, object],
) -> Mapping[str, object]:
    h8 = _mapping(
        target_scientific_config.get("h8"),
        "target_scientific_config.h8",
    )
    if (
        h8.get("schema_version") != "h8-validation-config-v3"
        or h8.get("operation") != "H8"
    ):
        raise ValueError("target must retain the frozen H8 v3 operation")
    integer_fields = ("T", "N", "K", "d_z", "d_m", "b", "D", "V")
    for name in integer_fields:
        if type(h8.get(name)) is not int or cast(int, h8[name]) <= 0:
            raise ValueError(f"target h8.{name} must be a positive integer")
    if (
        h8["N"] != cast(int, h8["T"]) + 1
        or h8["b"] != cast(int, h8["d_z"]) + cast(int, h8["d_m"])
        or h8["D"] != cast(int, h8["N"]) * cast(int, h8["b"])
        or h8["K"] != h8["d_z"]
        or h8["K"] != h8["d_m"]
    ):
        raise ValueError("target H8 layout identities are inconsistent")
    seed_table = h8.get("correctness_seed_table")
    if (
        type(seed_table) is not list
        or len(seed_table) != 12
        or any(
            type(row) is not list
            or len(row) != 4
            or any(type(value) is not int for value in row)
            for row in seed_table
        )
    ):
        raise ValueError("target H8 correctness seed table must contain 12 rows")
    cells = cast(list[list[int]], seed_table)
    if tuple((row[0], row[1]) for row in cells) != tuple(
        (horizon, dimension) for horizon in (1, 2, 4, 8) for dimension in (1, 2, 4)
    ):
        raise ValueError("target H8 correctness cells are not the frozen grid")
    production_pairs = h8.get("production_sample_seed_pairs")
    if (
        type(production_pairs) is not list
        or len(production_pairs) != 3
        or any(
            type(row) is not list
            or len(row) != 2
            or any(type(value) is not int for value in row)
            for row in production_pairs
        )
    ):
        raise ValueError("target H8 production seed pairs must contain three rows")
    for name in (
        "cold_repetitions",
        "max_process_incremental_mib",
        "max_torch_population_mib",
        "max_rhs_width",
        "sample_width",
    ):
        if type(h8.get(name)) is not int or cast(int, h8[name]) <= 0:
            raise ValueError(f"target h8.{name} must be a positive integer")
    if (
        type(h8.get("max_seconds")) not in (int, float)
        or not math.isfinite(cast(float, h8["max_seconds"]))
        or cast(float, h8["max_seconds"]) <= 0
    ):
        raise ValueError("target h8.max_seconds must be finite and positive")
    observed_sha256 = hashlib.sha256(canonical_h8_preflight_bytes(h8)).hexdigest()
    if observed_sha256 != H8_FROZEN_SECTION_SHA256:
        raise ValueError(
            "target H8 section differs from the complete frozen protocol: "
            f"{observed_sha256}"
        )
    return h8


def _workload_forecasts(
    h8: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    cells = cast(list[list[int]], h8["correctness_seed_table"])
    sources_per_cell = 3
    ordered_pairs_per_cell = 6
    wrong_path_controls_per_cell = 6
    endpoints_per_source = sum(19 + 4 * row[0] for row in cells)
    correctness = {
        "cells": len(cells),
        "sources_per_cell": sources_per_cell,
        "source_evaluations": len(cells) * sources_per_cell,
        "retained_source_endpoint_records": (endpoints_per_source * sources_per_cell),
        "ordered_source_pairs_per_cell": ordered_pairs_per_cell,
        "ordered_pair_endpoint_comparisons": (
            endpoints_per_source * ordered_pairs_per_cell
        ),
        "wrong_path_control_decisions": (len(cells) * wrong_path_controls_per_cell),
    }
    production_children = len(
        cast(list[list[int]], h8["production_sample_seed_pairs"])
    ) * cast(int, h8["cold_repetitions"])
    profiler_children = len(cast(list[list[int]], h8["production_sample_seed_pairs"]))
    isolated_controls = len(cells)
    total_children = production_children + profiler_children + isolated_controls
    n = cast(int, h8["N"])
    b = cast(int, h8["b"])
    d = cast(int, h8["D"])
    float64_bytes = 8
    information_scalars = n * b
    band_scalars = (2 * n - 1) * b * b
    workspace_scalars = b * b
    dense_scalars = d * d
    resource = {
        "children": {
            "production": production_children,
            "profiler": profiler_children,
            "isolated_allocation_controls": isolated_controls,
            "total": total_children,
            "retries": 0,
        },
        "layout": {"N": n, "b": b, "D": d},
        "storage": {
            "information_vector": {
                "scalars": information_scalars,
                "bytes": information_scalars * float64_bytes,
            },
            "precision": {
                "scalars": band_scalars,
                "bytes": band_scalars * float64_bytes,
            },
            "factor": {
                "scalars": band_scalars,
                "bytes": band_scalars * float64_bytes,
            },
            "selected_inverse": {
                "scalars": band_scalars,
                "bytes": band_scalars * float64_bytes,
            },
            "maximum_local_workspace": {
                "scalars": workspace_scalars,
                "bytes": workspace_scalars * float64_bytes,
            },
            "forbidden_dense_population": {
                "scalars": dense_scalars,
                "bytes": dense_scalars * float64_bytes,
            },
        },
        "per_child_caps": {
            "seconds": float(h8["max_seconds"]),
            "incremental_process_mib": h8["max_process_incremental_mib"],
            "torch_population_mib": h8["max_torch_population_mib"],
            "rhs_width": h8["max_rhs_width"],
            "sample_width": h8["sample_width"],
        },
        "sequential_resource_child_ceiling_seconds": (
            total_children * float(h8["max_seconds"])
        ),
        "sequential_ceiling_scope": (
            "resource_children_only; excludes uncapped correctness parent; "
            "arithmetic ceiling, not a runtime prediction"
        ),
        "estimated_total_wall_seconds": None,
        "measured_runtime_seconds": None,
        "measured_memory_mib": None,
    }
    return {"correctness": correctness}, resource


def _record(
    name: str,
    state: PrerequisiteState,
    *,
    path: Path | None = None,
    expected_schema: str | None = None,
    detail: str,
) -> H8PreflightPrerequisite:
    return H8PreflightPrerequisite(
        name=name,
        state=state,
        path=None if path is None else _path_text(path),
        expected_schema=expected_schema,
        detail=detail,
    )


def _regular_file_record(
    name: str,
    path: Path,
    *,
    expected_schema: str | None = None,
) -> H8PreflightPrerequisite:
    if path.is_symlink():
        return _record(
            name,
            "malformed",
            path=path,
            expected_schema=expected_schema,
            detail="control path is a symlink",
        )
    if not path.is_file():
        return _record(
            name,
            "missing",
            path=path,
            expected_schema=expected_schema,
            detail="required regular file is absent",
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return _record(
        name,
        "present_unvalidated",
        path=path,
        expected_schema=expected_schema,
        detail=f"regular file observed; sha256={digest}; contents not reopened",
    )


def _reference_paths(reference: Mapping[str, object]) -> tuple[str, ...]:
    paths: list[str] = []
    for key, value in reference.items():
        if key.endswith("_path") or key in (
            "artifact_path",
            "result_path",
            "ledger_path",
        ):
            if type(value) is str and value:
                paths.append(value)
        elif key == "correctness_artifact_paths" and isinstance(value, Mapping):
            paths.extend(
                cast(str, item) for item in value.values() if type(item) is str and item
            )
    return tuple(dict.fromkeys(paths))


def _hash_mapping_error(value: object, name: str) -> str | None:
    if not isinstance(value, Mapping) or not value:
        return f"{name} must be a nonempty mapping"
    for key, digest in value.items():
        if type(key) is not str or not key:
            return f"{name} contains an invalid key"
        try:
            _lower_hex(digest, 64, f"{name}[{key}]")
        except ValueError as exc:
            return str(exc)
    return None


def _direct_reference_structure_error(
    key: str,
    reference: Mapping[str, object],
) -> str | None:
    if set(reference) != H8_REFERENCE_FIELDS[key]:
        return "reference field inventory is not exact"
    for path_name in (
        "artifact_path",
        "result_path",
        "ledger_path",
    ):
        if type(reference[path_name]) is not str or not reference[path_name]:
            return f"{path_name} must be a nonempty path"
    for name, value in reference.items():
        if name.endswith("_sha256"):
            try:
                _lower_hex(value, 64, name)
            except ValueError as exc:
                return str(exc)
    try:
        _lower_hex(reference["producer_head"], 40, "producer_head")
        _lower_hex(
            reference["producer_dirty_digest"],
            64,
            "producer_dirty_digest",
        )
    except ValueError as exc:
        return str(exc)
    for mapping_name in ("content_hashes", "payload_hashes"):
        error = _hash_mapping_error(reference[mapping_name], mapping_name)
        if error is not None:
            return error
    if key == "h6_prefix":
        if (
            reference["config_schema"] != "h6-prefix-config-v3"
            or reference["validation_schema"]
            != "h6-prefix-validation-set-v2"
            or reference["certificate_set_schema"]
            != "h6-prefix-certificate-set-v2"
        ):
            return "bounded H6-Prefix schema discriminators are stale"
        families = reference["semantic_families"]
        family_fields = {
            "semantic_family_index",
            "semantic_family_sha256",
            "validation_payload_sha256",
            "certificate_sha256",
        }
        if (
            type(families) is not list
            or not families
            or any(
                type(row) is not dict or set(row) != family_fields
                for row in families
            )
        ):
            return "bounded H6-Prefix semantic families are malformed"
        observed_indices: list[int] = []
        observed_family_hashes: list[str] = []
        for row in families:
            assert isinstance(row, dict)
            index = row["semantic_family_index"]
            if type(index) is not int or index < 0:
                return "bounded H6-Prefix semantic-family index is invalid"
            observed_indices.append(index)
            for name in (
                "semantic_family_sha256",
                "validation_payload_sha256",
                "certificate_sha256",
            ):
                try:
                    _lower_hex(row[name], 64, name)
                except ValueError as exc:
                    return str(exc)
            observed_family_hashes.append(
                cast(str, row["semantic_family_sha256"])
            )
        if observed_indices != list(range(len(families))):
            return "bounded H6-Prefix semantic-family order is not contiguous"
        if len(set(observed_family_hashes)) != len(observed_family_hashes):
            return "bounded H6-Prefix semantic families are duplicated"
    if key == "h7" and (
        type(reference["result_pointer_path"]) is not str
        or not reference["result_pointer_path"]
    ):
        return "result_pointer_path must be a nonempty path"
    if key == "h6_prediction":
        for path_name in (
            "authorities_path",
            "validation_bundle_path",
            "reservation_path",
            "terminal_path",
            "finalized_path",
            "pointer_path",
            "candidate_junit_path",
        ):
            if type(reference[path_name]) is not str or not reference[path_name]:
                return f"{path_name} must be a nonempty path"
        expected_revision = (
            f"git:{reference['producer_head']}:sha256:"
            f"{reference['producer_dirty_digest']}"
        )
        if reference["artifact_revision"] != expected_revision:
            return "Prediction-v3 artifact revision does not bind its producer"
        if tuple(cast(Mapping[str, object], reference["payload_hashes"])) != (
            "metrics.json",
            "raw_inventory.json",
            "result.json",
        ):
            return "Prediction-v3 payload hashes must retain metrics/raw/result order"
    return None


def _compatibility_structure_error(reference: object) -> str | None:
    if not isinstance(reference, Mapping) or any(
        type(key) is not str for key in reference
    ):
        return "compatibility entry is not a string-keyed mapping"
    checked = cast(Mapping[str, object], reference)
    if set(checked) != H7_COMPATIBILITY_FIELDS:
        return "compatibility field inventory is not exact"
    for path_name in ("artifact_path", "junit_path", "ledger_path"):
        if type(checked[path_name]) is not str or not checked[path_name]:
            return f"{path_name} must be a nonempty path"
        if Path(cast(str, checked[path_name])).as_posix() != checked[path_name]:
            return f"{path_name} must use canonical forward slashes"
    for hash_name in (
        "dirty_digest",
        "junit_sha256",
        "manifest_sha256",
        "ledger_sha256",
        "reference_sha256",
    ):
        try:
            _lower_hex(checked[hash_name], 64, hash_name)
        except ValueError as exc:
            return str(exc)
    try:
        _lower_hex(checked["git_head"], 40, "git_head")
    except ValueError as exc:
        return str(exc)
    mapping_error = _hash_mapping_error(
        checked["payload_hashes"],
        "payload_hashes",
    )
    if mapping_error is not None:
        return mapping_error
    semantic = {
        key: value for key, value in checked.items() if key != "reference_sha256"
    }
    expected_reference_sha256 = hashlib.sha256(
        b"vfe4.h7.predecessor-reference.v1\x00" + canonical_h8_preflight_bytes(semantic)
    ).hexdigest()
    if checked["reference_sha256"] != expected_reference_sha256:
        return "reference_sha256 does not bind the compatibility record"
    return None


def _resolve_declared_path(repository_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repository_root / path


def _inspect_direct_reference(
    *,
    repository_root: Path,
    key: str,
    display_name: str,
    raw: object,
    candidate: Mapping[str, str],
    junit_sha256: str,
    registry_is_current: bool,
    registry_schema: object,
) -> H8PreflightPrerequisite:
    expected_schema = (
        "h6-prediction-result-v3" if key == "h6_prediction" else None
    )
    if not isinstance(raw, Mapping) or any(type(item) is not str for item in raw):
        return _record(
            display_name,
            "malformed",
            expected_schema=expected_schema,
            detail="reference entry is not a string-keyed mapping",
        )
    reference = cast(Mapping[str, object], raw)
    if reference.get("kind") != key or reference.get("status") != "pass":
        return _record(
            display_name,
            "blocked",
            expected_schema=expected_schema,
            detail="reference kind/status is not the required passing variant",
        )
    if key == "h6_prediction" and registry_schema != H8_REGISTRY_SCHEMA:
        return _record(
            display_name,
            "blocked",
            expected_schema=expected_schema,
            detail="historical H6-Prediction reference is nonauthorizing",
        )
    if key == "h6_prediction" and any(
        reference.get(name) != expected
        for name, expected in H8_PREDICTION_V3_SCHEMAS.items()
    ):
        return _record(
            display_name,
            "blocked",
            expected_schema=expected_schema,
            detail="H6-Prediction does not declare every executable v3 schema",
        )
    structure_error = _direct_reference_structure_error(key, reference)
    if structure_error is not None:
        return _record(
            display_name,
            "malformed",
            expected_schema=expected_schema,
            detail=structure_error,
        )
    if not registry_is_current:
        return _record(
            display_name,
            "stale",
            expected_schema=expected_schema,
            detail="registry does not bind the exact H8 candidate",
        )
    if key != "h6_prediction" and (
        reference.get("producer_head") != candidate["git_head"]
        or reference.get("producer_dirty_digest") != candidate["dirty_digest"]
        or reference.get("candidate_junit_sha256") != junit_sha256
    ):
        return _record(
            display_name,
            "stale",
            expected_schema=expected_schema,
            detail="current-chain reference does not bind the H8 candidate and JUnit",
        )
    declared_paths = _reference_paths(reference)
    if not declared_paths:
        return _record(
            display_name,
            "malformed",
            expected_schema=expected_schema,
            detail="reference declares no evidence paths",
        )
    missing_paths = [
        value
        for value in declared_paths
        if not _resolve_declared_path(repository_root, value).exists()
        or _resolve_declared_path(repository_root, value).is_symlink()
    ]
    if missing_paths:
        return _record(
            display_name,
            "missing",
            path=_resolve_declared_path(repository_root, missing_paths[0]),
            expected_schema=expected_schema,
            detail=(
                f"{len(missing_paths)} declared evidence path(s) are absent "
                "or symlinked"
            ),
        )
    return _record(
        display_name,
        "present_unvalidated",
        path=_resolve_declared_path(repository_root, declared_paths[0]),
        expected_schema=expected_schema,
        detail=(
            f"{len(declared_paths)} declared evidence path(s) observed; "
            "cryptographic reopening deferred to H8"
        ),
    )


def _inspect_registry(
    *,
    repository_root: Path,
    registry_path: Path,
    candidate: Mapping[str, str],
) -> tuple[H8PreflightPrerequisite, ...]:
    absent = (
        _record(
            "h8_registry_v5",
            "missing",
            path=registry_path,
            expected_schema=H8_REGISTRY_SCHEMA,
            detail="exact current-HEAD H8 registry is absent",
        ),
        _record(
            "candidate_junit",
            "missing",
            expected_schema="JUnit XML",
            detail="same-candidate JUnit is not discoverable without the registry",
        ),
        _record(
            "h1_h5",
            "missing",
            detail="H1--H5 evidence is not discoverable without the registry",
        ),
        _record(
            "h1_prefix_prior_v2",
            "missing",
            expected_schema="h1-prefix-prior-scorer-v2",
            detail="scorer-v2 prefix evidence is not discoverable",
        ),
        _record(
            "h6_prefix",
            "missing",
            detail="independent H6-Prefix evidence is not discoverable",
        ),
        _record(
            "h6_prediction_v3",
            "missing",
            expected_schema="h6-prediction-result-v3",
            detail="executable H6-Prediction v3 evidence is not discoverable",
        ),
        _record(
            "h7_compatibility_registry",
            "missing",
            expected_schema="h7-current-candidate-refs-v1",
            detail="H7 compatibility references are not discoverable",
        ),
        _record(
            "h7",
            "missing",
            detail="H7 evidence is not discoverable without the registry",
        ),
    )
    if registry_path.is_symlink():
        return (
            _record(
                "h8_registry_v5",
                "malformed",
                path=registry_path,
                expected_schema=H8_REGISTRY_SCHEMA,
                detail="exact registry path is a symlink",
            ),
            *absent[1:],
        )
    if not registry_path.is_file():
        return absent
    try:
        registry_bytes = registry_path.read_bytes()
        payload = json.loads(registry_bytes.decode("utf-8", errors="strict"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return (
            _record(
                "h8_registry_v5",
                "malformed",
                path=registry_path,
                expected_schema=H8_REGISTRY_SCHEMA,
                detail=f"registry is not readable strict UTF-8 JSON: {exc}",
            ),
            *absent[1:],
        )
    if (
        not isinstance(payload, dict)
        or set(payload)
        != {
            "schema_version",
            "candidate",
            "h7_compatibility_refs",
            "references",
        }
        or canonical_h8_preflight_bytes(payload) != registry_bytes
    ):
        return (
            _record(
                "h8_registry_v5",
                "malformed",
                path=registry_path,
                expected_schema=H8_REGISTRY_SCHEMA,
                detail="registry shape or canonical JSON bytes are invalid",
            ),
            *absent[1:],
        )
    schema = payload["schema_version"]
    raw_candidate = payload["candidate"]
    raw_compatibility = payload["h7_compatibility_refs"]
    raw_references = payload["references"]
    if (
        not isinstance(raw_candidate, dict)
        or set(raw_candidate) != {"git_head", "dirty_digest", "junit_sha256"}
        or not isinstance(raw_compatibility, dict)
        or not isinstance(raw_references, dict)
    ):
        return (
            _record(
                "h8_registry_v5",
                "malformed",
                path=registry_path,
                expected_schema=H8_REGISTRY_SCHEMA,
                detail="registry identity/reference sections are malformed",
            ),
            *absent[1:],
        )
    try:
        registry_head = _lower_hex(
            raw_candidate["git_head"], 40, "registry candidate head"
        )
        registry_dirty = _lower_hex(
            raw_candidate["dirty_digest"], 64, "registry candidate dirty digest"
        )
        junit_sha256 = _lower_hex(
            raw_candidate["junit_sha256"], 64, "registry candidate JUnit"
        )
    except ValueError as exc:
        return (
            _record(
                "h8_registry_v5",
                "malformed",
                path=registry_path,
                expected_schema=H8_REGISTRY_SCHEMA,
                detail=str(exc),
            ),
            *absent[1:],
        )
    registry_is_current = (
        registry_head == candidate["git_head"]
        and registry_dirty == candidate["dirty_digest"]
    )
    if schema != H8_REGISTRY_SCHEMA:
        registry_record = _record(
            "h8_registry_v5",
            "blocked",
            path=registry_path,
            expected_schema=H8_REGISTRY_SCHEMA,
            detail=f"registry schema {schema!r} is nonauthorizing",
        )
    elif not registry_is_current:
        registry_record = _record(
            "h8_registry_v5",
            "stale",
            path=registry_path,
            expected_schema=H8_REGISTRY_SCHEMA,
            detail="registry candidate identity differs from the current source",
        )
    elif set(raw_references) != set(H8_REFERENCE_KEYS):
        registry_record = _record(
            "h8_registry_v5",
            "malformed",
            path=registry_path,
            expected_schema=H8_REGISTRY_SCHEMA,
            detail="registry direct-reference inventory is not exact",
        )
    else:
        registry_record = _record(
            "h8_registry_v5",
            "present_unvalidated",
            path=registry_path,
            expected_schema=H8_REGISTRY_SCHEMA,
            detail=(
                "exact schema/candidate observed; artifact reopening deferred; "
                f"sha256={hashlib.sha256(registry_bytes).hexdigest()}"
            ),
        )

    compatibility_errors = tuple(
        _compatibility_structure_error(raw_compatibility.get(key))
        for key in H8_COMPATIBILITY_KEYS
    )
    compatibility_structure_is_exact = tuple(
        raw_compatibility
    ) == H8_COMPATIBILITY_KEYS and not any(compatibility_errors)
    compatibility_current = compatibility_structure_is_exact and all(
        cast(Mapping[str, object], raw_compatibility[key]).get("git_head")
        == candidate["git_head"]
        and cast(Mapping[str, object], raw_compatibility[key]).get("dirty_digest")
        == candidate["dirty_digest"]
        and cast(Mapping[str, object], raw_compatibility[key]).get("junit_sha256")
        == junit_sha256
        for key in H8_COMPATIBILITY_KEYS
    )
    junit_paths = tuple(
        cast(Mapping[str, object], raw_compatibility[key])["junit_path"]
        for key in H8_COMPATIBILITY_KEYS
        if compatibility_structure_is_exact
    )
    resolved_junit = tuple(
        _resolve_declared_path(repository_root, cast(str, value)).resolve(strict=False)
        for value in junit_paths
    )
    if not compatibility_structure_is_exact:
        junit_record = _record(
            "candidate_junit",
            "malformed",
            expected_schema="JUnit XML",
            detail="H7 compatibility references are structurally malformed",
        )
    elif len(set(resolved_junit)) != 1:
        junit_record = _record(
            "candidate_junit",
            "malformed",
            path=resolved_junit[0],
            expected_schema="JUnit XML",
            detail="H7 compatibility references do not share one JUnit path",
        )
    else:
        valid_junit = tuple(
            path.is_file()
            and not path.is_symlink()
            and hashlib.sha256(path.read_bytes()).hexdigest() == junit_sha256
            for path in resolved_junit
        )
        junit_record = _record(
            "candidate_junit",
            "present_unvalidated" if all(valid_junit) else "missing",
            path=resolved_junit[0],
            expected_schema="JUnit XML",
            detail=(
                "same-candidate JUnit paths and SHA-256 observed; XML not parsed"
                if all(valid_junit)
                else "one or more declared JUnit paths/hashes are unavailable"
            ),
        )
    if not compatibility_structure_is_exact:
        compatibility_record = _record(
            "h7_compatibility_registry",
            "malformed",
            expected_schema="h7-current-candidate-refs-v1",
            detail=next(
                (error for error in compatibility_errors if error is not None),
                "H7 compatibility inventory is not exact or ordered",
            ),
        )
    elif not compatibility_current:
        compatibility_record = _record(
            "h7_compatibility_registry",
            "stale",
            expected_schema="h7-current-candidate-refs-v1",
            detail="H7 compatibility references do not bind the same candidate",
        )
    else:
        compatibility_record = _record(
            "h7_compatibility_registry",
            "present_unvalidated",
            expected_schema="h7-current-candidate-refs-v1",
            detail="three exact compatibility identities are declared",
        )
    names = {
        "h1_h5": "h1_h5",
        "h1_prefix_prior": "h1_prefix_prior_v2",
        "h6_prefix": "h6_prefix",
        "h7": "h7",
        "h6_prediction": "h6_prediction_v3",
    }
    direct = {
        key: _inspect_direct_reference(
            repository_root=repository_root,
            key=key,
            display_name=names[key],
            raw=raw_references.get(key),
            candidate=candidate,
            junit_sha256=junit_sha256,
            registry_is_current=registry_is_current,
            registry_schema=schema,
        )
        for key in H8_REFERENCE_KEYS
    }
    transitive_mismatches: list[str] = []
    if compatibility_structure_is_exact:
        for key in H8_COMPATIBILITY_KEYS:
            raw_direct = raw_references.get(key)
            if not isinstance(raw_direct, Mapping):
                continue
            direct_reference = cast(Mapping[str, object], raw_direct)
            compatibility_reference = cast(
                Mapping[str, object],
                raw_compatibility[key],
            )
            if (
                direct_reference.get("artifact_path"),
                direct_reference.get("producer_head"),
                direct_reference.get("producer_dirty_digest"),
                direct_reference.get("candidate_junit_sha256"),
                direct_reference.get("manifest_sha256"),
                direct_reference.get("payload_hashes"),
                direct_reference.get("ledger_path"),
                direct_reference.get("ledger_sha256"),
            ) != (
                compatibility_reference["artifact_path"],
                compatibility_reference["git_head"],
                compatibility_reference["dirty_digest"],
                compatibility_reference["junit_sha256"],
                compatibility_reference["manifest_sha256"],
                compatibility_reference["payload_hashes"],
                compatibility_reference["ledger_path"],
                compatibility_reference["ledger_sha256"],
            ):
                transitive_mismatches.append(key)
    if transitive_mismatches:
        compatibility_record = _record(
            "h7_compatibility_registry",
            "stale",
            expected_schema="h7-current-candidate-refs-v1",
            detail=(
                "direct references differ from H7 transitive identities: "
                + ",".join(transitive_mismatches)
            ),
        )
    return (
        registry_record,
        junit_record,
        direct["h1_h5"],
        direct["h1_prefix_prior"],
        direct["h6_prefix"],
        direct["h6_prediction"],
        compatibility_record,
        direct["h7"],
    )


def _runtime_records(
    repository_root: Path,
) -> tuple[H8PreflightPrerequisite, H8PreflightPrerequisite]:
    runner_path = repository_root / "verification" / "run_gates.py"
    gate_path = repository_root / "verification" / "h8_gate.py"
    if runner_path.is_symlink() or gate_path.is_symlink():
        return (
            _record(
                "h8_runtime_orchestrator",
                "malformed",
                path=runner_path,
                detail="runtime source path is a symlink",
            ),
            _record(
                "h8_complete_runtime_cross_binding",
                "malformed",
                path=gate_path,
                detail="H8 gate source path is a symlink",
            ),
        )
    if not runner_path.is_file() or not gate_path.is_file():
        return (
            _record(
                "h8_runtime_orchestrator",
                "missing",
                path=runner_path,
                detail="H8 runtime source is unavailable",
            ),
            _record(
                "h8_complete_runtime_cross_binding",
                "missing",
                path=gate_path,
                detail="H8 gate source is unavailable",
            ),
        )
    try:
        runner_source = runner_path.read_text(encoding="utf-8")
        runner_tree = ast.parse(runner_source, filename=str(runner_path))
        gate_source = gate_path.read_text(encoding="utf-8")
        gate_tree = ast.parse(gate_source, filename=str(gate_path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        return (
            _record(
                "h8_runtime_orchestrator",
                "malformed",
                path=runner_path,
                detail=f"runtime source is not parseable UTF-8 Python: {exc}",
            ),
            _record(
                "h8_complete_runtime_cross_binding",
                "malformed",
                path=gate_path,
                detail=f"H8 gate source is not parseable UTF-8 Python: {exc}",
            ),
        )
    def called_name(call: ast.Call) -> str | None:
        called = call.func
        if isinstance(called, ast.Name):
            return called.id
        return None

    def top_level_function(
        tree: ast.Module,
        name: str,
    ) -> ast.FunctionDef | None:
        matches = tuple(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        return matches[0] if len(matches) == 1 else None

    class ScopeBindingVisitor(ast.NodeVisitor):
        """Collect bindings in one lexical scope without entering nested scopes."""

        def __init__(self, *, include_imports: bool) -> None:
            self.include_imports = include_imports
            self.names: set[str] = set()
            self.counts: dict[str, int] = {}

        def _record(self, name: str) -> None:
            self.names.add(name)
            self.counts[name] = self.counts.get(name, 0) + 1

        def visit_Name(self, node: ast.Name) -> None:
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                self._record(node.id)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._record(node.name)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._record(node.name)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self._record(node.name)

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return

        def visit_Import(self, node: ast.Import) -> None:
            if self.include_imports:
                for item in node.names:
                    self._record(
                        item.asname or item.name.split(".", maxsplit=1)[0]
                    )

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            if self.include_imports:
                for item in node.names:
                    self._record(item.asname or item.name)

    def lexical_rebindings(
        statements: list[ast.stmt],
        protected_names: frozenset[str],
        *,
        include_imports: bool,
    ) -> frozenset[str]:
        visitor = ScopeBindingVisitor(include_imports=include_imports)
        for statement in statements:
            visitor.visit(statement)
        return frozenset(visitor.names & protected_names)

    def lexical_binding_count(
        statements: list[ast.stmt],
        name: str,
        *,
        include_imports: bool,
    ) -> int:
        visitor = ScopeBindingVisitor(include_imports=include_imports)
        for statement in statements:
            visitor.visit(statement)
        return visitor.counts.get(name, 0)

    def function_rebindings(
        function: ast.FunctionDef,
        protected_names: frozenset[str],
    ) -> frozenset[str]:
        parameters = {
            item.arg
            for item in (
                *function.args.posonlyargs,
                *function.args.args,
                *function.args.kwonlyargs,
            )
        }
        if function.args.vararg is not None:
            parameters.add(function.args.vararg.arg)
        if function.args.kwarg is not None:
            parameters.add(function.args.kwarg.arg)
        return lexical_rebindings(
            function.body,
            protected_names,
            include_imports=True,
        ) | frozenset(parameters & protected_names)

    class ModuleImportBindingVisitor(ast.NodeVisitor):
        """Collect import bindings in one module without entering nested scopes."""

        def __init__(self, protected_names: frozenset[str]) -> None:
            self.protected_names = protected_names
            self.bindings: dict[str, list[tuple[str, str, str]]] = {}
            self.unsafe = False

        def _record(
            self,
            local_name: str,
            binding: tuple[str, str, str],
        ) -> None:
            if local_name in self.protected_names:
                self.bindings.setdefault(local_name, []).append(binding)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            return

        def visit_AsyncFunctionDef(
            self,
            node: ast.AsyncFunctionDef,
        ) -> None:
            return

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            return

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return

        def _record_direct_import(self, node: ast.Import) -> None:
            for item in node.names:
                local_name = item.asname or item.name.split(".", maxsplit=1)[0]
                self._record(
                    local_name,
                    ("import", item.name, item.asname or ""),
                )

        def _record_direct_import_from(self, node: ast.ImportFrom) -> None:
            module_name = "." * node.level + (node.module or "")
            for item in node.names:
                if item.name == "*":
                    self.unsafe = True
                    continue
                self._record(
                    item.asname or item.name,
                    ("from", module_name, item.name),
                )

        def visit_Import(self, node: ast.Import) -> None:
            if any(
                (item.asname or item.name.split(".", maxsplit=1)[0])
                in self.protected_names
                for item in node.names
            ):
                self.unsafe = True

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            if any(
                item.name == "*"
                or (item.asname or item.name) in self.protected_names
                for item in node.names
            ):
                self.unsafe = True

        def collect(self, tree: ast.Module) -> None:
            for statement in tree.body:
                if isinstance(statement, ast.Import):
                    self._record_direct_import(statement)
                elif isinstance(statement, ast.ImportFrom):
                    self._record_direct_import_from(statement)
                else:
                    self.visit(statement)

    def exact_module_import_bindings(
        tree: ast.Module,
        protected_names: frozenset[str],
        expected: Mapping[str, tuple[str, str]],
    ) -> bool:
        visitor = ModuleImportBindingVisitor(protected_names)
        visitor.collect(tree)
        return not visitor.unsafe and all(
            tuple(visitor.bindings.get(name, ()))
            == (
                (("from", *expected[name]),)
                if name in expected
                else ()
            )
            for name in protected_names
        )

    def direct_call(statement: ast.stmt) -> ast.Call | None:
        value: ast.expr | None
        if isinstance(statement, ast.Assign):
            value = statement.value
        elif isinstance(statement, ast.AnnAssign):
            value = statement.value
        elif isinstance(statement, (ast.Expr, ast.Return)):
            value = statement.value
        else:
            value = None
        return value if isinstance(value, ast.Call) else None

    def assigned_call(
        statement: ast.stmt,
        *,
        target_name: str,
        call_name: str,
    ) -> ast.Call | None:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            return None
        targets = (
            statement.targets
            if isinstance(statement, ast.Assign)
            else (statement.target,)
        )
        call = direct_call(statement)
        if (
            call is None
            or called_name(call) != call_name
            or not any(
                isinstance(target, ast.Name)
                and target.id == target_name
                for target in targets
            )
        ):
            return None
        return call

    def assigned_attribute(
        statement: ast.stmt,
        *,
        target_name: str,
        owner_name: str,
        attribute_name: str,
    ) -> bool:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            return False
        targets = (
            statement.targets
            if isinstance(statement, ast.Assign)
            else (statement.target,)
        )
        value = statement.value
        return (
            any(
                isinstance(target, ast.Name)
                and target.id == target_name
                for target in targets
            )
            and isinstance(value, ast.Attribute)
            and isinstance(value.value, ast.Name)
            and value.value.id == owner_name
            and value.attr == attribute_name
        )

    def keyword_name_bindings(call: ast.Call) -> dict[str, str]:
        return {
            cast(str, keyword.arg): keyword.value.id
            for keyword in call.keywords
            if keyword.arg is not None
            and isinstance(keyword.value, ast.Name)
        }

    class ReachableCallVisitor(ast.NodeVisitor):
        """Collect calls without counting nested definition decoys."""

        def __init__(self) -> None:
            self.calls: list[ast.Call] = []

        def visit_Call(self, node: ast.Call) -> None:
            self.calls.append(node)
            self.generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            return

        def visit_AsyncFunctionDef(
            self,
            node: ast.AsyncFunctionDef,
        ) -> None:
            return

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            return

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return

    def reachable_calls(statements: list[ast.stmt]) -> tuple[ast.Call, ...]:
        visitor = ReachableCallVisitor()
        for statement in statements:
            visitor.visit(statement)
        return tuple(visitor.calls)

    def dominating_calls(statements: list[ast.stmt]) -> tuple[ast.Call, ...]:
        """Collect exact calls evaluated before every path reaching the next statement."""

        class DirectScopeReturnVisitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.found = False

            def visit_Return(self, node: ast.Return) -> None:
                self.found = True

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                return

            def visit_AsyncFunctionDef(
                self,
                node: ast.AsyncFunctionDef,
            ) -> None:
                return

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                return

            def visit_Lambda(self, node: ast.Lambda) -> None:
                return

        return_visitor = DirectScopeReturnVisitor()
        for statement in statements:
            return_visitor.visit(statement)
        if return_visitor.found:
            return ()

        calls: list[ast.Call] = []
        for statement in statements:
            if (call := direct_call(statement)) is not None:
                calls.append(call)
            elif (
                isinstance(statement, ast.If)
                and isinstance(statement.test, ast.Compare)
                and len(statement.test.ops) == 1
                and len(statement.test.comparators) == 1
                and isinstance(statement.test.left, ast.Call)
            ):
                calls.append(statement.test.left)
            if isinstance(statement, (ast.Return, ast.Raise)):
                break
        return tuple(calls)

    def invalid_start_guard(statement: ast.stmt) -> bool:
        if not isinstance(statement, ast.If):
            return False
        test = statement.test
        return (
            isinstance(test, ast.UnaryOp)
            and isinstance(test.op, ast.Not)
            and isinstance(test.operand, ast.Attribute)
            and isinstance(test.operand.value, ast.Name)
            and test.operand.value.id == "authorization"
            and test.operand.attr == "valid_start"
        )

    def terminal_branch(statements: list[ast.stmt]) -> bool:
        return bool(statements) and isinstance(
            statements[-1],
            (ast.Return, ast.Raise),
        )

    def direct_assignment_index(
        statements: list[ast.stmt],
        *,
        target_name: str,
        call_name: str,
    ) -> int | None:
        return next(
            (
                index
                for index, statement in enumerate(statements)
                if assigned_call(
                    statement,
                    target_name=target_name,
                    call_name=call_name,
                )
                is not None
            ),
            None,
        )

    runner_function = top_level_function(
        runner_tree,
        "run_h8_verification",
    )
    runtime_available = False
    runtime_failure = "run_h8_verification is absent or duplicated"
    if runner_function is not None:
        runner_body = runner_function.body
        runner_protected_names = frozenset(
            (*H8_SELECTED_RUNTIME_CALL_NAMES, "assemble_h8_source_only_evaluation")
        )
        runner_import_bindings = {
            "produce_h8_correctness_grid": (
                "verification.h8_correctness",
                "produce_h8_correctness_grid",
            ),
            "assemble_h8_gate_evaluation": (
                "verification.h8_gate",
                "assemble_h8_gate_evaluation",
            ),
            "assemble_h8_source_only_evaluation": (
                "verification.h8_gate",
                "assemble_h8_source_only_evaluation",
            ),
            "validate_h8_prerequisite_artifacts": (
                "verification.h8_gate",
                "validate_h8_prerequisite_artifacts",
            ),
            "derive_h8_child_start_authorization": (
                "verification.h8_orchestrator",
                "derive_h8_child_start_authorization",
            ),
            "run_h8_parent_attempt": (
                "verification.h8_orchestrator",
                "run_h8_parent_attempt",
            ),
        }
        runner_bindings_clean = (
            exact_module_import_bindings(
                runner_tree,
                runner_protected_names,
                runner_import_bindings,
            )
            and not function_rebindings(
                runner_function,
                runner_protected_names,
            )
            and not lexical_rebindings(
                runner_tree.body,
                runner_protected_names,
                include_imports=False,
            )
        )
        prerequisite_index = direct_assignment_index(
            runner_body,
            target_name="prerequisite_validation",
            call_name="validate_h8_prerequisite_artifacts",
        )
        correctness_index = direct_assignment_index(
            runner_body,
            target_name="correctness",
            call_name="produce_h8_correctness_grid",
        )
        authorization_index = direct_assignment_index(
            runner_body,
            target_name="authorization",
            call_name="derive_h8_child_start_authorization",
        )
        guard_index = next(
            (
                index
                for index, statement in enumerate(runner_body)
                if invalid_start_guard(statement)
            ),
            None,
        )
        ordered_prefix = (
            prerequisite_index is not None
            and correctness_index is not None
            and authorization_index is not None
            and guard_index is not None
            and prerequisite_index < correctness_index
            < authorization_index < guard_index
            and not any(
                isinstance(statement, ast.Return)
                for statement in runner_body[:guard_index]
            )
        )
        if not ordered_prefix:
            runtime_failure = (
                "selected prerequisite/correctness/authorization chain does "
                "not dominate the invalid-start guard"
            )
        else:
            guard = cast(ast.If, runner_body[cast(int, guard_index)])
            invalid_calls = reachable_calls(guard.body)
            invalid_call = (
                direct_call(guard.body[0])
                if len(guard.body) == 1
                else None
            )
            invalid_calls_are_source_only = (
                invalid_call is not None
                and called_name(invalid_call)
                == "assemble_h8_source_only_evaluation"
                and invalid_calls == (invalid_call,)
            )
            if guard.orelse:
                valid_branch = guard.orelse
            elif terminal_branch(guard.body):
                valid_branch = runner_body[cast(int, guard_index) + 1 :]
            else:
                valid_branch = []
            parent_index = direct_assignment_index(
                valid_branch,
                target_name="parent_authority",
                call_name="run_h8_parent_attempt",
            )
            assembly_index = next(
                (
                    index
                    for index, statement in enumerate(valid_branch)
                    if (
                        (call := direct_call(statement)) is not None
                        and called_name(call)
                        == "assemble_h8_gate_evaluation"
                    )
                ),
                None,
            )
            assembly_call = (
                direct_call(valid_branch[assembly_index])
                if assembly_index is not None
                else None
            )
            assembly_bindings = (
                keyword_name_bindings(assembly_call)
                if assembly_call is not None
                else {}
            )
            exact_assembly_bindings = all(
                assembly_bindings.get(name) == name
                for name in H8_SELECTED_GATE_ARGUMENT_NAMES
            )
            runtime_available = (
                runner_bindings_clean
                and invalid_calls_are_source_only
                and parent_index is not None
                and assembly_index is not None
                and parent_index < assembly_index
                and exact_assembly_bindings
            )
            if not runner_bindings_clean:
                runtime_failure = (
                    "selected runtime call names are rebound outside their "
                    "direct module bindings"
                )
            elif not invalid_calls_are_source_only:
                runtime_failure = (
                    "invalid-start branch contains a call outside the exact "
                    "source-only gate"
                )
            elif not runtime_available:
                runtime_failure = (
                    "authority launch and exact gate bindings do not share "
                    "the guarded valid-start branch"
                )
    if runner_function is None:
        runtime_detail = runtime_failure
    elif not runtime_available:
        runtime_detail = runtime_failure
    else:
        runtime_detail = (
            "selected prerequisite, correctness, authorization, parent, and "
            "gate chain shares one dominated branch; behavior unvalidated"
        )
    runtime_record = _record(
        "h8_runtime_orchestrator",
        "present_unvalidated" if runtime_available else "blocked",
        path=runner_path,
        detail=runtime_detail,
    )
    gate_function = top_level_function(
        gate_tree,
        "assemble_h8_gate_evaluation",
    )
    forbidden_legacy_authorization = any(
        (
            isinstance(node, ast.Name)
            and node.id
            in {
                "runtime_authorized",
                "_assemble_h8_gate_evaluation_from_inventories",
            }
        )
        or (
            isinstance(node, ast.keyword)
            and node.arg == "runtime_authorized"
        )
        for node in ast.walk(gate_tree)
    )
    cross_binding_available = False
    cross_binding_failure = (
        "authoritative H8 gate is absent or duplicated"
    )
    if gate_function is not None and not forbidden_legacy_authorization:
        gate_protected_names = frozenset(
            {
                "build_h8_v4_runtime_sections",
                "require_h8_parent_attempt_authority",
                "_issue_h8_gate_pass_result",
                "_finalize_h8_gate_evaluation",
            }
        )
        gate_import_bindings = {
            "build_h8_v4_runtime_sections": (
                "verification.h8_runtime",
                "build_h8_v4_runtime_sections",
            ),
            "require_h8_parent_attempt_authority": (
                "verification.h8_parent_authority",
                "require_h8_parent_attempt_authority",
            ),
            "_issue_h8_gate_pass_result": (
                "vfe4.types.results",
                "_issue_h8_gate_pass_result",
            ),
        }
        gate_imported_names = frozenset(gate_import_bindings)
        gate_bindings_clean = (
            exact_module_import_bindings(
                gate_tree,
                gate_protected_names,
                gate_import_bindings,
            )
            and top_level_function(
                gate_tree,
                "_finalize_h8_gate_evaluation",
            )
            is not None
            and lexical_binding_count(
                gate_tree.body,
                "_finalize_h8_gate_evaluation",
                include_imports=False,
            )
            == 1
            and not function_rebindings(
                gate_function,
                gate_protected_names,
            )
            and not lexical_rebindings(
                gate_tree.body,
                gate_imported_names,
                include_imports=False,
            )
        )
        argument_names = {
            item.arg
            for item in (
                *gate_function.args.args,
                *gate_function.args.kwonlyargs,
            )
        }
        gate_body = gate_function.body
        authority_index = direct_assignment_index(
            gate_body,
            target_name="authority",
            call_name="require_h8_parent_attempt_authority",
        )
        attempts_index = next(
            (
                index
                for index, statement in enumerate(gate_body)
                if assigned_attribute(
                    statement,
                    target_name="child_attempts",
                    owner_name="authority",
                    attribute_name="attempts",
                )
            ),
            None,
        )
        pass_index = next(
            (
                index
                for index, statement in enumerate(gate_body)
                if (
                    isinstance(statement, ast.If)
                    and isinstance(statement.test, ast.Compare)
                    and isinstance(statement.test.left, ast.Name)
                    and statement.test.left.id == "status"
                    and len(statement.test.ops) == 1
                    and isinstance(statement.test.ops[0], ast.Is)
                    and len(statement.test.comparators) == 1
                    and isinstance(
                        statement.test.comparators[0],
                        ast.Attribute,
                    )
                    and isinstance(
                        statement.test.comparators[0].value,
                        ast.Name,
                    )
                    and statement.test.comparators[0].value.id
                    == "GateStatus"
                    and statement.test.comparators[0].attr == "PASS"
                )
            ),
            None,
        )
        finalize_index = next(
            (
                index
                for index, statement in enumerate(gate_body)
                if (
                    (call := direct_call(statement)) is not None
                    and called_name(call)
                    == "_finalize_h8_gate_evaluation"
                )
            ),
            None,
        )
        authority_call = (
            direct_call(gate_body[authority_index])
            if authority_index is not None
            else None
        )
        authority_input_exact = (
            authority_call is not None
            and len(authority_call.args) == 1
            and isinstance(authority_call.args[0], ast.Name)
            and authority_call.args[0].id == "parent_authority"
        )
        pass_calls: tuple[ast.Call, ...] = ()
        if pass_index is not None:
            pass_calls = dominating_calls(
                cast(ast.If, gate_body[pass_index]).body
            )
        named_pass_calls = {
            name: tuple(
                call
                for call in pass_calls
                if called_name(call) == name
            )
            for name in (
                "build_h8_v4_runtime_sections",
                "require_h8_parent_attempt_authority",
                "_issue_h8_gate_pass_result",
            )
        }
        build_calls = named_pass_calls["build_h8_v4_runtime_sections"]
        revalidation_calls = named_pass_calls[
            "require_h8_parent_attempt_authority"
        ]
        issuer_calls = named_pass_calls["_issue_h8_gate_pass_result"]
        pass_calls_ordered = (
            bool(build_calls)
            and bool(revalidation_calls)
            and bool(issuer_calls)
            and build_calls[0].lineno
            < revalidation_calls[0].lineno
            < issuer_calls[0].lineno
        )
        build_bindings = (
            keyword_name_bindings(build_calls[0])
            if build_calls
            else {}
        )
        issuer_bindings = (
            keyword_name_bindings(issuer_calls[0])
            if issuer_calls
            else {}
        )
        exact_inventory_bindings = all(
            build_bindings.get(name) == name
            and issuer_bindings.get(name) == name
            for name in (
                "correctness",
                "child_attempts",
                "production_runs",
                "profiler_runs",
                "controls",
            )
        )
        exact_revalidation = any(
            len(call.args) == 1
            and isinstance(call.args[0], ast.Name)
            and call.args[0].id == "authority"
            for call in revalidation_calls
        )
        finalize_call = (
            direct_call(gate_body[finalize_index])
            if finalize_index is not None
            else None
        )
        finalize_bindings = (
            keyword_name_bindings(finalize_call)
            if finalize_call is not None
            else {}
        )
        exact_finalization = (
            finalize_bindings.get("result") == "result"
            and finalize_bindings.get("retained_runtime_sections")
            == "retained_runtime_sections"
        )
        cross_binding_available = (
            set(H8_SELECTED_GATE_ARGUMENT_NAMES).issubset(argument_names)
            and gate_bindings_clean
            and authority_input_exact
            and authority_index is not None
            and attempts_index is not None
            and pass_index is not None
            and finalize_index is not None
            and authority_index < attempts_index < pass_index < finalize_index
            and pass_calls_ordered
            and exact_inventory_bindings
            and exact_revalidation
            and exact_finalization
        )
        if not cross_binding_available:
            if not gate_bindings_clean:
                cross_binding_failure = (
                    "authoritative PASS call names are rebound outside their "
                    "direct module bindings"
                )
            else:
                cross_binding_failure = (
                    "authority, owned-v4 PASS issuance, and final assembly do "
                    "not share one reachable dominated branch"
                )
    elif forbidden_legacy_authorization:
        cross_binding_failure = (
            "legacy caller-provided runtime authorization remains reachable"
        )
    if not cross_binding_available:
        cross_binding_detail = cross_binding_failure
    else:
        cross_binding_detail = (
            "authority projection, owned-v4 revalidation, private PASS "
            "issuance, and final assembly are visible; behavior unvalidated"
        )
    cross_binding_record = _record(
        "h8_complete_runtime_cross_binding",
        "present_unvalidated" if cross_binding_available else "blocked",
        path=gate_path,
        detail=cross_binding_detail,
    )
    return runtime_record, cross_binding_record


def inspect_h8_preflight(
    *,
    repository_root: Path,
    target_scientific_config: Mapping[str, object],
    request: Mapping[str, object],
    candidate: Mapping[str, object],
) -> H8PreflightResult:
    """Inspect H8 readiness without running or publishing scientific work."""

    if not isinstance(repository_root, Path):
        raise ValueError("repository_root must be a pathlib.Path")
    root = repository_root.resolve(strict=False)
    request_mapping = _mapping(request, "request")
    target_mapping = _mapping(
        target_scientific_config,
        "target_scientific_config",
    )
    _validate_request(request_mapping)
    checked_candidate = _validate_candidate(_mapping(candidate, "candidate"))
    h8 = _validate_target_h8(target_mapping)
    target_config_sha256 = hashlib.sha256(
        canonical_h8_preflight_bytes(target_mapping)
    ).hexdigest()
    workload, resource = _workload_forecasts(h8)

    active_path = root / ".verification" / "active.json"
    if active_path.is_symlink():
        active_record = _record(
            "verification_marker_clear",
            "malformed",
            path=active_path,
            detail="active verification marker is a symlink",
        )
    elif active_path.exists():
        active_record = _record(
            "verification_marker_clear",
            "blocked",
            path=active_path,
            detail="active verification marker must be cleared before H8 execution",
        )
    else:
        active_record = _record(
            "verification_marker_clear",
            "present_unvalidated",
            path=active_path,
            detail="no active verification marker is present",
        )
    preregistration_record = _regular_file_record(
        "h8_preregistration",
        root / H8_PREREGISTRATION_RELATIVE_PATH,
        expected_schema="2026-07-21-h8-sparse-scale",
    )
    registry_path = (
        root
        / ".verification"
        / (f"h8-current-candidate-{checked_candidate['git_head']}-refs.json")
    )
    registry_records = _inspect_registry(
        repository_root=root,
        registry_path=registry_path,
        candidate=checked_candidate,
    )
    runtime_records = _runtime_records(root)
    prerequisites = (
        active_record,
        preregistration_record,
        *registry_records,
        *runtime_records,
    )
    obligations = tuple(
        f"{item.name}:{item.state}"
        for item in prerequisites
        if item.state != "present_unvalidated"
    )
    disposition: PreflightDisposition = (
        "blocked" if obligations else "metadata_complete_unvalidated"
    )
    execution_policy = {
        "inspection_policy": "metadata_only",
        "tests_launched": 0,
        "training_runs_launched": 0,
        "scientific_evaluations_launched": 0,
        "profiler_runs_launched": 0,
        "scientific_children_launched": 0,
        "artifact_writes": 0,
        "result_delivery": "stdout_and_return_value_only",
        "scientific_status_authority": "none",
    }
    payload = {
        "schema_version": H8_PREFLIGHT_RESULT_SCHEMA,
        "operation": "H8-Preflight",
        "disposition": disposition,
        "scientific_status": "not_evaluated",
        "candidate": checked_candidate,
        "target_config_sha256": target_config_sha256,
        "prerequisites": [item.as_dict() for item in prerequisites],
        "workload_forecast": workload,
        "resource_forecast": resource,
        "execution_policy": execution_policy,
        "obligations": list(obligations),
    }
    return H8PreflightResult(
        schema_version=H8_PREFLIGHT_RESULT_SCHEMA,
        operation="H8-Preflight",
        disposition=disposition,
        scientific_status="not_evaluated",
        candidate=checked_candidate,
        target_config_sha256=target_config_sha256,
        prerequisites=prerequisites,
        workload_forecast=workload,
        resource_forecast=resource,
        execution_policy=execution_policy,
        obligations=obligations,
        result_sha256=hashlib.sha256(canonical_h8_preflight_bytes(payload)).hexdigest(),
    )


__all__ = [
    "H8_PREFLIGHT_CONFIG_SCHEMA",
    "H8_PREFLIGHT_RESULT_SCHEMA",
    "H8PreflightPrerequisite",
    "H8PreflightResult",
    "canonical_h8_preflight_bytes",
    "capture_current_candidate",
    "inspect_h8_preflight",
    "read_h8_exact_test_nodes",
]
