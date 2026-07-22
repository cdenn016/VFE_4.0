"""Revision- and content-bound provenance for ordered verification publications."""

from __future__ import annotations

import hashlib
import io
import os
import platform
import subprocess
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch

from vfe4.config import ResolvedConfig
from vfe4.config.control_paths import is_repository_control_path, is_same_or_descendant

from .atomic import ArtifactPublicationError


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
    required_values = {
        key: value for key, value in values.items() if key != "fixture_observed_sha256"
    }
    if any(value is None or value == "" for value in required_values.values()):
        raise ArtifactPublicationError("provenance contains a missing value")
    return values


def build_environment(config: ResolvedConfig) -> dict[str, object]:
    clock = time.get_clock_info("perf_counter")
    torch_config_text = str(torch.__config__.show())
    numpy_buffer = io.StringIO()
    with redirect_stdout(numpy_buffer):
        np.show_config()
    numpy_blas_text = numpy_buffer.getvalue()
    try:
        affinity: tuple[int, ...] | None = tuple(sorted(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        try:
            import psutil  # type: ignore[import-not-found]

            affinity = tuple(sorted(psutil.Process().cpu_affinity()))
        except (ImportError, AttributeError, OSError):
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
