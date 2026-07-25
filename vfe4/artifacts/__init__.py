"""Public atomic artifact interfaces."""

from .atomic import (
    ArtifactPublicationError,
    canonical_json_bytes,
    publish_run_directory,
)
from .h6 import (
    CandidateArtifactReference,
    ProjectedCurrentCandidateConfig,
    project_h1_prefix_prior_config,
    project_h6_prefix_config,
    run_projected_current_candidate,
)
from .h6_matching import (
    H6MatchingOwnershipRecord,
    H6MatchingSetRecord,
    derive_h6_matrix_reports,
    publish_h6_matching_set,
    read_h6_matching_set,
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

__all__ = [
    "ArtifactPublicationError",
    "CandidateArtifactReference",
    "H6MatchingOwnershipRecord",
    "H6MatchingSetRecord",
    "ProjectedCurrentCandidateConfig",
    "canonical_json_bytes",
    "publish_run_directory",
    "project_h1_prefix_prior_config",
    "project_h6_prefix_config",
    "run_projected_current_candidate",
    "build_environment",
    "build_provenance",
    "current_source_identity",
    "dirty_content_digest",
    "derive_h6_matrix_reports",
    "git_head",
    "process_cpu_affinity",
    "publish_h6_matching_set",
    "read_h6_matching_set",
    "source_candidate_sha256",
]
