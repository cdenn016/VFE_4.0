"""Revision- and content-bound provenance for H1 publications."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from vfe4.config import ResolvedConfig

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
    names = _git(repo_root, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
    try:
        decoded = [name for name in names.decode("utf-8", errors="strict").split("\0") if name]
    except UnicodeError as exc:
        raise ArtifactPublicationError(f"Git path decoding failed: {exc}") from exc
    root = repo_root.resolve()
    configured = run_root.resolve()
    digest = hashlib.sha256()
    for name in sorted(set(decoded)):
        relative = Path(name)
        normalized = relative.as_posix()
        if normalized == ".git" or normalized.startswith(".git/"):
            continue
        if normalized == ".verification" or normalized.startswith(".verification/"):
            continue
        absolute = (root / relative).resolve()
        try:
            absolute.relative_to(configured)
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
    fixture_path: Path,
    config: ResolvedConfig,
    started_utc: str,
    ended_utc: str,
    gate_state: str,
) -> dict[str, object]:
    try:
        fixture_sha256 = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ArtifactPublicationError(f"fixture identity unavailable: {exc}") from exc
    objective_input = f"objective_schema_version:{config.objective_schema_version}"
    values: dict[str, object] = {
        "git_head": git_head(repo_root),
        "dirty_digest": dirty_content_digest(repo_root, config.artifacts.run_root),
        "config_sha256": config.config_sha256,
        "objective_schema_input": objective_input,
        "objective_schema_sha256": hashlib.sha256(objective_input.encode("utf-8")).hexdigest(),
        "fixture_sha256": fixture_sha256,
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
    if any(value is None or value == "" for value in values.values()):
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
