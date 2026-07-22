"""Revision- and content-bound provenance for H1 publications."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from vfe4.config import ResolvedConfig
from vfe4.config.control_paths import is_repository_control_path

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
    try:
        configured_relative = configured.relative_to(root)
    except ValueError:
        try:
            root.relative_to(configured)
        except ValueError:
            excluded_root: Path | None = None
        else:
            raise ArtifactPublicationError(
                "run_root must be a strict descendant or external to the repository"
            )
    else:
        if not configured_relative.parts:
            raise ArtifactPublicationError(
                "run_root must be a strict descendant or external to the repository"
            )
        excluded_root = configured
    digest = hashlib.sha256()
    for name in sorted(tracked | untracked):
        relative = Path(name)
        normalized = relative.as_posix()
        if normalized == ".git" or normalized.startswith(".git/"):
            continue
        if normalized == ".verification" or normalized.startswith(".verification/"):
            continue
        absolute = (root / relative).resolve()
        if name not in tracked and excluded_root is not None:
            try:
                absolute.relative_to(excluded_root)
            except ValueError:
                pass
            else:
                continue
        try:
            absolute.relative_to(root)
        except ValueError as exc:
            raise ArtifactPublicationError(f"provenance path escapes repository: {name}") from exc
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
    values: dict[str, object] = {
        "git_head": git_head(repo_root),
        "dirty_digest": dirty_content_digest(repo_root, config.artifacts.run_root),
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
    return {
        "python_version": sys.version.split()[0],
        "pytorch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "device": config.run.device,
        "dtype": config.run.dtype,
        "seed": config.run.seed,
        "deterministic": config.run.deterministic,
        "stochastic_policy": "no-stochastic-operations",
    }
