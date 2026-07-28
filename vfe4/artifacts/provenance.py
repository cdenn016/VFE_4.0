"""Revision- and content-bound provenance for ordered verification publications."""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch

from vfe4.config import ResolvedConfig
from vfe4.config.control_paths import is_repository_control_path, is_same_or_descendant
from vfe4.types.h7 import H7GateEvaluation, H7PredecessorReference
from vfe4.types.h8 import CurrentH8PrerequisiteRefs, H8GateEvaluation
from vfe4.types.results import H7GateResult, H8GateResult

from .atomic import ArtifactPublicationError


def _canonical_cpu_affinity(cpu_ids: object) -> tuple[int, ...]:
    try:
        observed = tuple(cpu_ids)  # type: ignore[arg-type]
    except TypeError as exc:
        raise RuntimeError("process CPU affinity provider returned a non-iterable") from exc
    if not observed or any(type(cpu_id) is not int or cpu_id < 0 for cpu_id in observed):
        raise RuntimeError("process CPU affinity provider returned invalid CPU IDs")
    return tuple(sorted(set(observed)))


def process_cpu_affinity() -> tuple[int, ...]:
    """Return the process's real, canonical CPU-affinity IDs or raise."""
    failures: list[str] = []
    sched_getaffinity = getattr(os, "sched_getaffinity", None)
    if sched_getaffinity is not None:
        try:
            return _canonical_cpu_affinity(sched_getaffinity(0))
        except Exception as exc:
            failures.append(f"os.sched_getaffinity: {type(exc).__name__}: {exc}")

    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            get_current_process = kernel32.GetCurrentProcess
            get_current_process.argtypes = ()
            get_current_process.restype = wintypes.HANDLE
            get_process_affinity_mask = kernel32.GetProcessAffinityMask
            get_process_affinity_mask.argtypes = (
                wintypes.HANDLE,
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.POINTER(ctypes.c_size_t),
            )
            get_process_affinity_mask.restype = wintypes.BOOL
            process_mask = ctypes.c_size_t()
            system_mask = ctypes.c_size_t()
            if not get_process_affinity_mask(
                get_current_process(), ctypes.byref(process_mask), ctypes.byref(system_mask),
            ):
                error_code = ctypes.get_last_error()
                raise OSError(error_code, "GetProcessAffinityMask failed")
            return _canonical_cpu_affinity(
                cpu_id
                for cpu_id in range(process_mask.value.bit_length())
                if process_mask.value & (1 << cpu_id)
            )
        except Exception as exc:
            failures.append(f"GetProcessAffinityMask: {type(exc).__name__}: {exc}")

    try:
        import psutil  # type: ignore[import-not-found]

        return _canonical_cpu_affinity(psutil.Process().cpu_affinity())
    except Exception as exc:
        failures.append(f"psutil.Process.cpu_affinity: {type(exc).__name__}: {exc}")

    detail = "; ".join(failures) or "no affinity provider is available"
    raise RuntimeError(f"process CPU affinity is unavailable: {detail}")


def _git(repo_root: Path, *arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ArtifactPublicationError(f"Git provenance unavailable: {exc}") from exc
    return completed.stdout


def git_head(repo_root: Path) -> str:
    value = _git(repo_root, "rev-parse", "HEAD").decode("ascii", errors="strict").strip()
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ArtifactPublicationError("Git HEAD is not an exact SHA-1 revision")
    return value


def source_candidate_sha256(
    *, git_head_value: str, dirty_digest_value: str
) -> str:
    """Bind the exact revision and dirty-content digest as one source identity."""

    if (
        type(git_head_value) is not str
        or len(git_head_value) != 40
        or any(character not in "0123456789abcdef" for character in git_head_value)
    ):
        raise ArtifactPublicationError("source candidate Git head is invalid")
    if (
        type(dirty_digest_value) is not str
        or len(dirty_digest_value) != 64
        or any(
            character not in "0123456789abcdef"
            for character in dirty_digest_value
        )
    ):
        raise ArtifactPublicationError("source candidate dirty digest is invalid")
    return hashlib.sha256(
        b"VFE4-H6-SOURCE-CANDIDATE-V1\x00"
        + bytes.fromhex(git_head_value)
        + bytes.fromhex(dirty_digest_value)
    ).hexdigest()


def current_source_identity(
    repo_root: Path,
    run_root: Path,
) -> tuple[str, str, str]:
    """Capture the current Git, dirty-content, and bound source identities."""

    head = git_head(repo_root)
    dirty = dirty_content_digest(repo_root, run_root)
    return (
        head,
        dirty,
        source_candidate_sha256(
            git_head_value=head,
            dirty_digest_value=dirty,
        ),
    )


def dirty_content_digest(repo_root: Path, run_root: Path) -> str:
    """Hash tracked and nonignored untracked bytes, excluding publication state."""
    tracked_names = _git(repo_root, "ls-files", "--cached", "-z")
    untracked_names = _git(repo_root, "ls-files", "--others", "--exclude-standard", "-z")
    try:
        tracked = {
            name
            for name in tracked_names.decode("utf-8", errors="strict").split("\0")
            if name
        }
        untracked = {
            name
            for name in untracked_names.decode("utf-8", errors="strict").split("\0")
            if name
        }
    except UnicodeError as exc:
        raise ArtifactPublicationError(f"Git path decoding failed: {exc}") from exc
    root = repo_root.resolve()
    configured = run_root.resolve()
    if is_repository_control_path(configured, root):
        raise ArtifactPublicationError("run_root must not enter a repository control tree")
    if is_same_or_descendant(root, configured):
        raise ArtifactPublicationError(
            "run_root must be a strict descendant or external to the repository"
        )
    if is_same_or_descendant(configured, root):
        excluded_root = configured
    else:
        excluded_root = None
    digest = hashlib.sha256()
    for name in sorted(tracked | untracked):
        relative = Path(name)
        normalized = relative.as_posix()
        if normalized == ".git" or normalized.startswith(".git/"):
            continue
        if normalized == ".verification" or normalized.startswith(".verification/"):
            continue
        absolute = (root / relative).resolve()
        if (
            name not in tracked
            and excluded_root is not None
            and is_same_or_descendant(absolute, excluded_root)
        ):
            continue
        if not is_same_or_descendant(absolute, root):
            raise ArtifactPublicationError(f"provenance path escapes repository: {name}")
        try:
            content = absolute.read_bytes()
        except FileNotFoundError:
            content = b"<deleted>"
        except OSError as exc:
            raise ArtifactPublicationError(f"provenance content unreadable: {name}: {exc}") from exc
        encoded_name = normalized.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def build_provenance(
    *,
    repo_root: Path,
    fixture_expected_sha256: str,
    fixture_observed_sha256: str | None,
    config: ResolvedConfig,
    started_utc: str,
    ended_utc: str,
    gate_state: str,
    candidate_junit_sha256: str | None = None,
) -> dict[str, object]:
    lowercase_hex = frozenset("0123456789abcdef")
    if (
        type(fixture_expected_sha256) is not str
        or len(fixture_expected_sha256) != 64
        or any(character not in lowercase_hex for character in fixture_expected_sha256)
    ):
        raise ArtifactPublicationError("fixture expected SHA-256 is invalid")
    if fixture_observed_sha256 is not None and (
        type(fixture_observed_sha256) is not str
        or len(fixture_observed_sha256) != 64
        or any(character not in lowercase_hex for character in fixture_observed_sha256)
    ):
        raise ArtifactPublicationError("fixture observed SHA-256 is invalid")
    if gate_state not in {"pass", "fail", "inconclusive"}:
        raise ArtifactPublicationError("gate state is invalid")
    if gate_state != "inconclusive" and fixture_observed_sha256 != fixture_expected_sha256:
        raise ArtifactPublicationError(
            "fixture observed SHA-256 must match expected SHA-256 for a closed gate"
        )
    if candidate_junit_sha256 is not None and (
        type(candidate_junit_sha256) is not str
        or len(candidate_junit_sha256) != 64
        or any(character not in lowercase_hex for character in candidate_junit_sha256)
    ):
        raise ArtifactPublicationError(
            "candidate JUnit SHA-256 must be lowercase 64-hex"
        )
    objective_input = f"objective_schema_version:{config.objective_schema_version}"
    dirty_digest = dirty_content_digest(repo_root, config.artifacts.run_root)
    values: dict[str, object] = {
        "git_head": git_head(repo_root),
        "dirty_digest": dirty_digest,
        "dirty_content_digest": dirty_digest,
        "config_sha256": config.config_sha256,
        "objective_schema_input": objective_input,
        "objective_schema_sha256": hashlib.sha256(objective_input.encode("utf-8")).hexdigest(),
        "fixture_expected_sha256": fixture_expected_sha256,
        "fixture_observed_sha256": fixture_observed_sha256,
        "fixture_available": fixture_observed_sha256 is not None,
        "python_version": sys.version.split()[0],
        "pytorch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "device": config.run.device,
        "dtype": config.run.dtype,
        "seed": config.run.seed,
        "deterministic": config.run.deterministic,
        "stochastic_policy": "no-stochastic-operations",
        "started_utc": started_utc,
        "ended_utc": ended_utc,
        "gate_state": gate_state,
    }
    if candidate_junit_sha256 is not None:
        values["junit_sha256"] = candidate_junit_sha256
    required_values = {
        key: value for key, value in values.items() if key != "fixture_observed_sha256"
    }
    if any(value is None or value == "" for value in required_values.values()):
        raise ArtifactPublicationError("provenance contains a missing value")
    return values


def build_h7_provenance(
    *,
    config: ResolvedConfig,
    evaluation: H7GateEvaluation,
    git_head_value: str,
    dirty_digest_value: str,
    source_sha256_value: str,
    junit_sha256: str,
    reference_registry_path: Path,
    reference_registry_sha256: str,
    fixture_expected_sha256: Mapping[str, str],
    fixture_observed_sha256: Mapping[str, str],
    predecessor_references: Mapping[str, H7PredecessorReference],
    scorer_profile: str,
    nonclaims: tuple[str, ...],
    budget_constants: Mapping[str, object],
    started_utc: str,
    ended_utc: str,
) -> dict[str, object]:
    """Build the reference-only, source-bound H7 provenance record."""

    if (
        type(config) is not ResolvedConfig
        or config.validation.gates
        != ("H1", "H2", "H3", "H4", "H5", "H6-Prefix", "H7")
        or config.h7 is None
    ):
        raise ArtifactPublicationError("H7 provenance requires the exact H7 config")
    if type(evaluation) is not H7GateEvaluation:
        raise ArtifactPublicationError("H7 provenance requires an owned evaluation")
    evaluation.__post_init__()
    if type(evaluation.result) is not H7GateResult:
        raise ArtifactPublicationError("H7 provenance requires results.py::H7GateResult")
    if (
        source_candidate_sha256(
            git_head_value=git_head_value,
            dirty_digest_value=dirty_digest_value,
        )
        != source_sha256_value
    ):
        raise ArtifactPublicationError("H7 source identity is not content-bound")
    for name, value in (
        ("junit_sha256", junit_sha256),
        ("reference_registry_sha256", reference_registry_sha256),
    ):
        if (
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ArtifactPublicationError(f"H7 {name} is invalid")
    if (
        not isinstance(reference_registry_path, Path)
        or not isinstance(fixture_expected_sha256, Mapping)
        or not isinstance(fixture_observed_sha256, Mapping)
        or tuple(fixture_expected_sha256)
        != (
            "h1_fixture_raw_sha256",
            "h7_fixture_raw_sha256",
            "density_probe_table_raw_sha256",
            "density_probe_set_sha256",
        )
        or tuple(fixture_observed_sha256) != tuple(fixture_expected_sha256)
    ):
        raise ArtifactPublicationError("H7 fixture provenance inventory is not exact")
    expected_predecessors = ("h1_h5", "h1_prefix_prior", "h6_prefix")
    if (
        not isinstance(predecessor_references, Mapping)
        or tuple(predecessor_references) != expected_predecessors
        or any(
            type(value) is not H7PredecessorReference
            for value in predecessor_references.values()
        )
    ):
        raise ArtifactPublicationError("H7 predecessor provenance is not exact")
    if (
        type(scorer_profile) is not str
        or not scorer_profile
        or type(nonclaims) is not tuple
        or not nonclaims
        or not isinstance(budget_constants, Mapping)
    ):
        raise ArtifactPublicationError("H7 protocol provenance is incomplete")

    h7_config = json.loads(config.h7.canonical_json)
    if not isinstance(h7_config, dict):
        raise ArtifactPublicationError("H7 canonical config is not a JSON object")
    predecessor_hashes = {
        key: {
            "artifact_path": reference.artifact_path,
            "manifest_sha256": reference.manifest_sha256,
            "payload_hashes": reference.payload_hashes,
            "ledger_path": reference.ledger_path,
            "ledger_sha256": reference.ledger_sha256,
            "reference_sha256": reference.reference_sha256,
        }
        for key, reference in predecessor_references.items()
    }
    fixture_hashes = {
        key: {
            "expected_sha256": fixture_expected_sha256[key],
            "observed_sha256": fixture_observed_sha256[key],
            "hash_domain": "raw_fixture_bytes"
            if key != "density_probe_set_sha256"
            else "domain_separated_canonical_probe_set",
        }
        for key in fixture_expected_sha256
    }
    objective_input = f"objective_schema_version:{config.objective_schema_version}"
    return {
        "schema_version": "vfe4-h7-provenance-v1",
        "git_head": git_head_value,
        "dirty_digest": dirty_digest_value,
        "dirty_content_digest": dirty_digest_value,
        "source_sha256": source_sha256_value,
        "config_sha256": config.config_sha256,
        "objective_schema_input": objective_input,
        "objective_schema_sha256": hashlib.sha256(
            objective_input.encode("utf-8")
        ).hexdigest(),
        "junit_sha256": junit_sha256,
        "reference_registry": {
            "path": reference_registry_path.resolve(strict=False).as_posix(),
            "sha256": reference_registry_sha256,
        },
        "production_order": (*expected_predecessors, "h7"),
        "predecessor_references": predecessor_hashes,
        "fixture_hashes": fixture_hashes,
        "fixture_capture": "h1-and-h7-raw-bytes-once-before-evaluation",
        "h7_protocol": {
            "selected_operation": "H7",
            "ordered_gates": config.validation.gates,
            "group_claim": "selected direct GL+(2,R) matrix elements only",
            "scalar_regression": (
                "separately typed GL+(1,R) complete-law regression; "
                "not GL+(2,R) evidence"
            ),
            "required_trial_specs": h7_config["required_trial_specs"],
            "required_control_ids": h7_config["required_control_ids"],
            "recognition_origins": h7_config["recognition_families"],
            "factorized_pushforward_representation": (
                "unrestricted_full_block_pushforward"
            ),
            "scorer_profile": scorer_profile,
            "history_scorer_law": (
                "alpha_b,t,j(prefix)+r_z^T z_j+r_m^T m_j; "
                "both covectors use the source inverse transpose"
            ),
            "history_scorer_control_id": (
                "history_scorer_wrong_source_inverse"
            ),
            "density_probe_table_raw_sha256": (
                config.h7.density_probe_table_raw_sha256
            ),
            "density_probe_set_sha256": config.h7.density_probe_set_sha256,
            "joint_initial_kl_schema": "K0_joint_z0_m0",
            "entropy_law": "continuous recognition entropy shifts by +logJ_G",
            "fixed_decoder_scope": (
                "centered-softmax stabilizer C_V W g^-1=C_V W"
            ),
            "oracle": {
                "implementation": "verification.mp_oracles.h7_covariance",
                "decimal_precision": config.h7.oracle_decimal_precision,
                "gauss_hermite_orders": config.h7.gauss_hermite_orders,
            },
            "envelope": {
                "group_norm_limit": config.h7.group_norm_limit,
                "group_inverse_norm_limit": config.h7.group_inverse_norm_limit,
                "spd_condition_limit": config.h7.spd_condition_limit,
                "inclusive": True,
            },
            "operand_budget": dict(budget_constants),
        },
        "h7_result": {
            "status": evaluation.result.status.value,
            "obligations": evaluation.result.obligations,
            "result_sha256": evaluation.result.result_sha256,
            "evaluation_sha256": evaluation.evaluation_sha256,
            "fixture_set_sha256": evaluation.fixture_set_sha256,
            "dependency_closure_sha256": evaluation.dependency_closure_sha256,
        },
        "nonclaims": nonclaims,
        "started_utc": started_utc,
        "ended_utc": ended_utc,
    }


_H8_ENVIRONMENT_KEYS = (
    "platform",
    "platform_release",
    "processor",
    "cpu_count",
    "affinity",
    "python_version",
    "pytorch_version",
    "numpy_version",
    "device",
    "dtype",
    "grad_enabled",
    "intraop_threads",
    "interop_threads",
    "thread_environment",
    "blas_identity",
    "hardware_identity_sha256",
    "affinity_sha256",
    "thread_identity_sha256",
    "blas_identity_sha256",
)

H8_PROVENANCE_KEYS = (
    "schema_version",
    "git_head",
    "dirty_digest",
    "dirty_content_digest",
    "source_sha256",
    "config_sha256",
    "junit_sha256",
    "current_refs_registry_sha256",
    "reference_registry",
    "dependency_closure_sha256",
    "preregistration_sha256",
    "interpretation_sha256",
    "validation_sha256",
    "evaluation_sha256",
    "status",
    "obligations",
    "selected_operation",
    "ordered_gates",
    "execution_scope",
    "external_pointer_in_artifact",
    "started_utc",
    "ended_utc",
)


def build_h8_environment(
    *,
    config: ResolvedConfig,
    validation_environment: Mapping[str, object],
) -> dict[str, object]:
    """Retain the already-validated H8 environment without probing it again."""

    if (
        type(config) is not ResolvedConfig
        or config.validation.gates
        != ("H1", "H2", "H3", "H4", "H5", "H6-Prefix", "H7", "H8")
        or config.h8 is None
        or type(config.h8_current_refs) is not CurrentH8PrerequisiteRefs
    ):
        raise ArtifactPublicationError(
            "H8 environment requires the exact bound H8 config"
        )
    if (
        not isinstance(validation_environment, Mapping)
        or set(validation_environment) != set(_H8_ENVIRONMENT_KEYS)
        or validation_environment.get("device") != "cpu"
        or validation_environment.get("dtype") != "float64"
        or validation_environment.get("grad_enabled") is not False
        or validation_environment.get("pytorch_version")
        != config.h8.torch_version
    ):
        raise ArtifactPublicationError(
            "H8 environment does not match the frozen schema"
        )
    return {
        name: validation_environment[name]
        for name in _H8_ENVIRONMENT_KEYS
    }


def build_h8_provenance(
    *,
    config: ResolvedConfig,
    evaluation: H8GateEvaluation,
    git_head_value: str,
    dirty_digest_value: str,
    source_sha256_value: str,
    reference_registry_path: Path,
    reference_registry_sha256: str,
    started_utc: str,
    ended_utc: str,
) -> dict[str, object]:
    """Build an H8 record bound to exact current inputs and retained evidence."""

    refs = config.h8_current_refs
    if (
        type(config) is not ResolvedConfig
        or config.validation.gates
        != ("H1", "H2", "H3", "H4", "H5", "H6-Prefix", "H7", "H8")
        or config.h8 is None
        or type(refs) is not CurrentH8PrerequisiteRefs
    ):
        raise ArtifactPublicationError("H8 provenance requires the exact bound config")
    if (
        type(evaluation) is not H8GateEvaluation
        or type(evaluation.result) is not H8GateResult
    ):
        raise ArtifactPublicationError("H8 provenance requires an owned evaluation")
    evaluation.__post_init__()
    if (
        source_candidate_sha256(
            git_head_value=git_head_value,
            dirty_digest_value=dirty_digest_value,
        )
        != source_sha256_value
        or refs.candidate_head != git_head_value
        or refs.candidate_dirty_digest != dirty_digest_value
        or evaluation.result.config_sha256 != config.config_sha256
        or evaluation.result.candidate_junit_sha256
        != refs.candidate_junit_sha256
        or evaluation.result.current_refs_registry_sha256 != refs.registry_sha256
    ):
        raise ArtifactPublicationError("H8 provenance identities are inconsistent")
    if (
        not isinstance(reference_registry_path, Path)
        or reference_registry_sha256 != refs.registry_sha256
        or len(reference_registry_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in reference_registry_sha256
        )
        or type(started_utc) is not str
        or not started_utc
        or type(ended_utc) is not str
        or not ended_utc
        or started_utc > ended_utc
    ):
        raise ArtifactPublicationError("H8 provenance registry/time inputs are invalid")
    provenance = {
        "schema_version": "vfe4-h8-provenance-v1",
        "git_head": git_head_value,
        "dirty_digest": dirty_digest_value,
        "dirty_content_digest": dirty_digest_value,
        "source_sha256": source_sha256_value,
        "config_sha256": config.config_sha256,
        "junit_sha256": refs.candidate_junit_sha256,
        "current_refs_registry_sha256": refs.registry_sha256,
        "reference_registry": {
            "path": reference_registry_path.resolve(strict=False).as_posix(),
            "sha256": reference_registry_sha256,
        },
        "dependency_closure_sha256": evaluation.dependency_closure_sha256,
        "preregistration_sha256": evaluation.preregistration_sha256,
        "interpretation_sha256": evaluation.interpretation_sha256,
        "validation_sha256": evaluation.validation_payload_sha256,
        "evaluation_sha256": evaluation.evaluation_sha256,
        "status": evaluation.result.status.value,
        "obligations": evaluation.result.obligations,
        "selected_operation": "H8",
        "ordered_gates": config.validation.gates,
        "execution_scope": (
            "h8-parent-orchestrated-runtime-v1"
            if evaluation.result.child_attempts
            else "source-only-empty-runtime-records"
        ),
        "external_pointer_in_artifact": False,
        "started_utc": started_utc,
        "ended_utc": ended_utc,
    }
    if tuple(provenance) != H8_PROVENANCE_KEYS:
        raise RuntimeError("internal H8 provenance inventory drifted")
    return provenance


def build_environment(config: ResolvedConfig) -> dict[str, object]:
    clock = time.get_clock_info("perf_counter")
    torch_config_text = str(torch.__config__.show())
    numpy_buffer = io.StringIO()
    with redirect_stdout(numpy_buffer):
        np.show_config()
    numpy_blas_text = numpy_buffer.getvalue()
    try:
        affinity: tuple[int, ...] | None = process_cpu_affinity()
    except RuntimeError:
        affinity = None
    try:
        import psutil  # type: ignore[import-not-found]

        physical_cpu_count: int | None = psutil.cpu_count(logical=False)
    except (ImportError, AttributeError, OSError):
        physical_cpu_count = None
    thread_environment = {
        name: {"present": name in os.environ, "value": os.environ.get(name)}
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        )
    }
    return {
        "python_version": sys.version.split()[0],
        "pytorch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "device": config.run.device,
        "dtype": config.run.dtype,
        "seed": config.run.seed,
        "deterministic": config.run.deterministic,
        "stochastic_policy": "no-stochastic-operations",
        "timing_clock": {
            "name": "perf_counter",
            "function": "time.perf_counter_ns",
            "implementation": clock.implementation,
            "resolution_seconds": float(clock.resolution),
            "monotonic": bool(clock.monotonic),
            "adjustable": bool(clock.adjustable),
        },
        "process_cpu_affinity": affinity,
        "logical_cpu_count": os.cpu_count(),
        "physical_cpu_count": physical_cpu_count,
        "processor": platform.processor(),
        "platform": platform.platform(),
        "torch_intra_op_threads": torch.get_num_threads(),
        "torch_inter_op_threads": torch.get_num_interop_threads(),
        "torch_config_sha256": hashlib.sha256(
            torch_config_text.encode("utf-8")
        ).hexdigest(),
        "torch_config_text": torch_config_text,
        "numpy_blas_config_sha256": hashlib.sha256(
            numpy_blas_text.encode("utf-8")
        ).hexdigest(),
        "numpy_blas_config_text": numpy_blas_text,
        "cuda_available": bool(torch.cuda.is_available()),
        "thread_environment": thread_environment,
    }
