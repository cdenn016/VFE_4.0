"""Nonproduction source-record boundary for hermetic Tasks 1--12.

Production source-record construction intentionally does not exist here.  The
separately authorized source-lock transaction owns the distinct finalized
record after live source, tokenizer, cache, schedule, and dependency facts are
all closed.
"""

from __future__ import annotations

from pathlib import Path

from .wikitext103 import (
    ArchiveMemberObservation,
    LicenseObservation,
    SealedSplitRef,
    SourceAcquisitionError,
    StagedWikiText103AcquisitionObservation,
    StagedWikiText103AcquisitionRecord,
    reopen_staged_wikitext103,
)


def validate_staged_source_record(
    observation: StagedWikiText103AcquisitionRecord,
    *,
    staging_root: Path,
    expected_record_sha256: str,
) -> StagedWikiText103AcquisitionRecord:
    """Reopen a staged record under one explicitly expected identity."""

    if (
        type(expected_record_sha256) is not str
        or expected_record_sha256 != observation.record_sha256
    ):
        raise SourceAcquisitionError(
            "expected staged observation identity does not match"
        )
    return reopen_staged_wikitext103(
        observation=observation,
        staging_root=staging_root,
    )


__all__ = [
    "ArchiveMemberObservation",
    "LicenseObservation",
    "SealedSplitRef",
    "SourceAcquisitionError",
    "StagedWikiText103AcquisitionObservation",
    "StagedWikiText103AcquisitionRecord",
    "validate_staged_source_record",
]
