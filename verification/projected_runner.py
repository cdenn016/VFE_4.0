"""Verification-owned execution adapter for projected H1/H6 candidates."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from vfe4.artifacts.h6 import (
    CandidateArtifactReference,
    ProjectedCurrentCandidateConfig,
    _install_projected_current_candidate_runner,
    run_projected_current_candidate as _run_projected_current_candidate,
)


def _execute_projected_gate(
    operation: Literal["H1-Prefix-Prior", "H6-Prefix"],
    resolved_config: object,
    junit_sha256: str | None,
) -> tuple[object, Path]:
    if operation == "H1-Prefix-Prior":
        from verification.h1_prefix_prior_gate import (
            run_h1_prefix_prior,
            run_parent_specific_h1_prefix_prior,
        )
        from vfe4.config import H1PrefixPriorV2ResolvedConfig

        if type(resolved_config) is H1PrefixPriorV2ResolvedConfig:
            return run_parent_specific_h1_prefix_prior(
                resolved_config,
                junit_sha256=junit_sha256,
            )
        return run_h1_prefix_prior(resolved_config)  # type: ignore[arg-type]
    if operation == "H6-Prefix":
        from verification.h6_prefix_gate import run_h6_prefix

        return run_h6_prefix(
            config=resolved_config,  # type: ignore[arg-type]
            junit_sha256=junit_sha256,
        )
    raise ValueError("unsupported projected current-candidate operation")


def run_projected_current_candidate(
    *,
    config: ProjectedCurrentCandidateConfig,
    junit_sha256: str | None,
    predecessor_refs: Mapping[str, CandidateArtifactReference],
) -> CandidateArtifactReference:
    """Bind the eligible verifier callback, then invoke the frozen seam."""

    _install_projected_current_candidate_runner(_execute_projected_gate)
    return _run_projected_current_candidate(
        config=config,
        junit_sha256=junit_sha256,
        predecessor_refs=predecessor_refs,
    )


__all__ = ["run_projected_current_candidate"]
