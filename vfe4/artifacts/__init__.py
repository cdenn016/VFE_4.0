"""Public atomic artifact interfaces."""

from .atomic import (
    ArtifactPublicationError,
    canonical_json_bytes,
    publish_run_directory,
)
from .provenance import (
    build_environment,
    build_provenance,
    current_source_identity,
    dirty_content_digest,
    git_head,
    process_cpu_affinity,
    source_candidate_sha256,
)

_H6_EXPORTS = frozenset(
    {
        "CandidateArtifactReference",
        "ProjectedCurrentCandidateConfig",
        "project_h1_prefix_prior_config",
        "project_h1_prefix_prior_v2_config",
        "project_h6_prefix_config",
        "publish_h6_prediction_result",
        "read_h6_prediction_result",
        "reopen_bounded_prefix_certificate_set",
        "run_projected_current_candidate",
    }
)
_H6_MATCHING_EXPORTS = frozenset(
    {
        "H6MatchingOwnershipRecord",
        "H6MatchingSetRecord",
        "derive_h6_inference_inclusive_compute_report",
        "derive_h6_matrix_reports",
        "publish_h6_matching_set",
        "read_h6_matching_set",
    }
)
_H6_PREDICTION_V3_EXPORTS = frozenset(
    {
        "H6CheckpointCandidateV3",
        "H6CheckpointSelectionV3",
        "H6EndpointTuningSelectionV3",
        "H6TuningSelectionV3",
        "H6ValidationBundleV3",
        "H6ValidationRecordV3",
        "bind_h6_checkpoint_selection_v3",
        "publish_h6_validation_bundle_v3",
        "read_h6_validation_bundle_v3",
        "select_h6_tuning_v3",
    }
)
_H7_EXPORTS = frozenset(
    {
        "build_h7_task5_precision_operand_table_bytes",
    }
)


def __getattr__(name: str) -> object:
    """Load H6 orchestration surfaces only when explicitly requested."""

    if name in _H6_EXPORTS:
        from . import h6

        return getattr(h6, name)
    if name in _H6_MATCHING_EXPORTS:
        from . import h6_matching

        return getattr(h6_matching, name)
    if name in _H6_PREDICTION_V3_EXPORTS:
        from . import h6_prediction_v3

        return getattr(h6_prediction_v3, name)
    if name in _H7_EXPORTS:
        from . import h7

        return getattr(h7, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ArtifactPublicationError",
    "CandidateArtifactReference",
    "H6MatchingOwnershipRecord",
    "H6MatchingSetRecord",
    "H6CheckpointCandidateV3",
    "H6CheckpointSelectionV3",
    "H6EndpointTuningSelectionV3",
    "H6TuningSelectionV3",
    "H6ValidationBundleV3",
    "H6ValidationRecordV3",
    "ProjectedCurrentCandidateConfig",
    "canonical_json_bytes",
    "publish_run_directory",
    "project_h1_prefix_prior_config",
    "project_h1_prefix_prior_v2_config",
    "project_h6_prefix_config",
    "publish_h6_prediction_result",
    "read_h6_prediction_result",
    "reopen_bounded_prefix_certificate_set",
    "run_projected_current_candidate",
    "build_environment",
    "build_h7_task5_precision_operand_table_bytes",
    "build_provenance",
    "bind_h6_checkpoint_selection_v3",
    "current_source_identity",
    "derive_h6_inference_inclusive_compute_report",
    "dirty_content_digest",
    "derive_h6_matrix_reports",
    "git_head",
    "process_cpu_affinity",
    "publish_h6_matching_set",
    "publish_h6_validation_bundle_v3",
    "read_h6_validation_bundle_v3",
    "read_h6_matching_set",
    "select_h6_tuning_v3",
    "source_candidate_sha256",
]
