"""H6 arm construction and deterministic capacity matching."""

from .arms import (
    ARM_MATRIX_ROWS,
    ARM_MATRIX_SHA256,
    ArmTargetFreeProposalAdapter,
    BuiltArm,
    CausalAutoregressiveModel,
    H6_TARGET_FREE_DATA_SAFETY_SHA256,
    LatentLanguageArmModel,
    build_a0,
    build_a1,
    build_a2,
    build_a3,
    build_a4,
    build_a5,
    build_arm,
    literal_arm_semantic_payload,
    shared_a2_a5_semantic_payload,
)

from .checkpoint import (
    H6CheckpointManifest,
    load_h6_checkpoint,
    save_h6_checkpoint,
)
from .language import (
    ArmObjectiveInventory,
    ArmTrainingObjectiveAdapter,
    DetachedRecognitionLawSnapshot,
    H6AttemptCursor,
    H6AttemptSpec,
    H6CrossEntropyTerms,
    H6ObjectiveManifest,
    H6ReducedLanguageElboTerms,
    H6TrainingAuthorization,
    H6TypedTrainingObjective,
    plan_h6_attempt,
    train_h6_attempt,
)
from .matching import (
    AMENDED_MATCHING_SCHEDULE_POLICY,
    AmendedMatchingSchedulePolicy,
    ArmConfig,
    ArmMatrixRow,
    CapacityAllocation,
    EndpointFormulaProfile,
    FlopTerm,
    H6AnalyticalFlopLedger,
    H6FormulaSelection,
    H6TrainingWorkload,
    MatchingReport,
    OptimizerBinding,
    ParameterRoleRecord,
    analytical_training_flop_ledger,
    arm_matrix_sha256,
    audit_arm_matching,
    endpoint_formula_profile,
    select_outcome_blind_allocation,
    stable_parameter_key,
)
from .parameter_counts import (
    A5_REFERENCE_PARAMETER_COUNT,
    AMENDED_EMISSION_WIDTH_CANDIDATES,
    AMENDED_LATENT_WIDTH_CANDIDATES,
    AMENDED_RECOGNITION_WIDTH_CANDIDATES,
    PROPOSED_PREFIX_PRIOR_CONTEXT_WIDTH,
    ParameterCountAssessment,
    arm_parameter_count,
    fixed_source_prior_parameter_count,
    outcome_blind_feasibility_assessments,
    parameter_count_within_tolerance,
    prefix_conditioned_source_prior_parameter_count,
    recognition_parameter_count,
)

_READINESS_EXPORTS = frozenset(
    {
        "CurrentPredictionPrerequisiteRefs",
        "ProducerCompatibilityError",
        "validate_h6_prediction_readiness",
    }
)
_EXPERIMENT_EXPORTS = frozenset(
    {"H6ExperimentRunResult", "run_h6_experiment"}
)


def __getattr__(name: str) -> object:
    """Load effectful H6 orchestration surfaces only when requested."""

    if name in _READINESS_EXPORTS:
        from . import h6_readiness

        return getattr(h6_readiness, name)
    if name in _EXPERIMENT_EXPORTS:
        from . import h6_experiment

        return getattr(h6_experiment, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "A5_REFERENCE_PARAMETER_COUNT",
    "AMENDED_MATCHING_SCHEDULE_POLICY",
    "AMENDED_EMISSION_WIDTH_CANDIDATES",
    "AMENDED_LATENT_WIDTH_CANDIDATES",
    "AMENDED_RECOGNITION_WIDTH_CANDIDATES",
    "ARM_MATRIX_ROWS",
    "ARM_MATRIX_SHA256",
    "ArmConfig",
    "ArmMatrixRow",
    "AmendedMatchingSchedulePolicy",
    "ArmObjectiveInventory",
    "ArmTargetFreeProposalAdapter",
    "ArmTrainingObjectiveAdapter",
    "BuiltArm",
    "CapacityAllocation",
    "CausalAutoregressiveModel",
    "CurrentPredictionPrerequisiteRefs",
    "DetachedRecognitionLawSnapshot",
    "EndpointFormulaProfile",
    "FlopTerm",
    "H6AnalyticalFlopLedger",
    "H6AttemptCursor",
    "H6AttemptSpec",
    "H6CheckpointManifest",
    "H6CrossEntropyTerms",
    "H6ExperimentRunResult",
    "H6FormulaSelection",
    "H6ObjectiveManifest",
    "H6ReducedLanguageElboTerms",
    "H6TrainingAuthorization",
    "H6TrainingWorkload",
    "H6TypedTrainingObjective",
    "H6_TARGET_FREE_DATA_SAFETY_SHA256",
    "LatentLanguageArmModel",
    "MatchingReport",
    "OptimizerBinding",
    "ParameterRoleRecord",
    "ParameterCountAssessment",
    "PROPOSED_PREFIX_PRIOR_CONTEXT_WIDTH",
    "analytical_training_flop_ledger",
    "arm_parameter_count",
    "ProducerCompatibilityError",
    "arm_matrix_sha256",
    "audit_arm_matching",
    "build_a0",
    "build_a1",
    "build_a2",
    "build_a3",
    "build_a4",
    "build_a5",
    "build_arm",
    "fixed_source_prior_parameter_count",
    "endpoint_formula_profile",
    "literal_arm_semantic_payload",
    "load_h6_checkpoint",
    "outcome_blind_feasibility_assessments",
    "parameter_count_within_tolerance",
    "plan_h6_attempt",
    "prefix_conditioned_source_prior_parameter_count",
    "recognition_parameter_count",
    "save_h6_checkpoint",
    "select_outcome_blind_allocation",
    "shared_a2_a5_semantic_payload",
    "stable_parameter_key",
    "train_h6_attempt",
    "run_h6_experiment",
    "validate_h6_prediction_readiness",
]
