"""H6 arm construction and deterministic capacity matching."""

from .arms import (
    ARM_MATRIX_ROWS,
    ARM_MATRIX_SHA256,
    ArmTargetFreeProposalAdapter,
    BuiltArm,
    H6A0ArchitectureProfile,
    H6A0ValidationProfile,
    H6CausalTransformer,
    H6_TARGET_FREE_DATA_SAFETY_SHA256,
    LatentLanguageArmModel,
    MeanPooledPrefixFloor,
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
from .h7_assembly import (
    H7FixedSourceAssemblyReceipt,
    H7FixedSourceAssemblySpec,
    build_h7_fixed_a5_arm,
    require_h7_fixed_source_assembly,
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
    H6PrimaryJointCandidate,
    H6PrimaryJointSelection,
    H6TrainingWorkload,
    MatchingReport,
    OptimizerBinding,
    ParameterRoleRecord,
    analytical_training_flop_ledger,
    arm_matrix_sha256,
    audit_arm_matching,
    endpoint_formula_profile,
    select_outcome_blind_allocation,
    select_parent_specific_primary_allocation,
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
    h6_a0_parameter_count,
    mean_pooled_no_latent_parameter_count,
    outcome_blind_feasibility_assessments,
    parent_specific_pooled_prefix_source_prior_parameter_count,
    parameter_count_within_tolerance,
    recognition_parameter_count,
)
from .engine import (
    ArmExecutionRuntime,
    AttemptEventSink,
    AttemptResult,
    ForwardTerms,
    ProposalEvidence,
    RecognitionSnapshot,
    ScientificStateParticipant,
    StepResult,
    TrainingEngineError,
    WT103_STRUCTURED_FACTOR_ELBO_SCHEMA,
    WT103_STRUCTURED_FACTOR_ELBO_SCHEMA_SHA256,
    train_attempt,
    train_step,
)
from .factories import (
    A0FactoryInputs,
    A0MatchRow,
    ArmMatchingReport,
    WT103FactorySetIdentity,
    audit_arm_matching as audit_wt103_arm_matching,
    build_wt103_a0,
    build_wt103_a5_fixed,
    build_wt103_a5_nolatent,
    build_wt103_a5_parent_specific,
    build_wt103_arm,
    scorer_dispatch,
)
from .formulas import (
    A0FlopLedger,
    A0FlopTerm,
    A0FlopWorkload,
    A0ParameterInventory,
    NamedParameterShape,
    a0_primitive_formula_sha256,
    build_a0_architecture_profile,
    build_a0_formula_record,
    reconstruct_a0_flops,
    reconstruct_a0_parameters,
)
from .readiness import (
    PredictorPerturbationObservation,
    StaticScientificArtifactRef,
    StaticScientificPreconditionRecord,
    WT103PredictorSafetyCertificate,
    certify_wt103_predictor_safety,
    validate_static_scientific_preconditions,
)
from .sparsity import (
    ArmPathTrace,
    FlashAttentionObservation,
    ForbiddenStorageRequest,
    NegativeControlRecord,
    TensorStorageObservation,
    TrainingSparsityAudit,
    certify_training_sparsity,
    guard_flash_attention_request,
    guard_tensor_request,
    run_sparsity_negative_controls,
)
from .wt103_models import (
    BuiltWT103Arm,
    ExecutionScope,
    OptimizerParameterBinding,
    WT103A0Model,
    WT103ArmBuildRecord,
    WT103ArmRuntimeComponents,
)

_READINESS_EXPORTS = frozenset(
    {
        "CurrentPredictionPrerequisiteRefs",
        "H6PredictionV3PrerequisiteEvidence",
        "ProducerCompatibilityError",
        "read_h6_bounded_prefix_certificate_set_for_scoring_v3",
        "reopen_h6_prediction_v3_prerequisite_evidence",
        "validate_existing_h6_prediction_readiness_v3",
        "validate_h6_prediction_readiness",
        "validate_h6_prediction_readiness_v3",
    }
)
_EXPERIMENT_EXPORTS = frozenset({"H6ExperimentRunResult", "run_h6_experiment"})
_EXPERIMENT_V3_EXPORTS = frozenset(
    {"prepare_h6_test_transaction_v3", "run_h6_experiment_v3"}
)
_CHECKPOINT_CATALOG_V3_EXPORTS = frozenset(
    {
        "H6CheckpointCatalogEntryV3",
        "H6CheckpointCatalogItemV3",
        "H6CheckpointCatalogV3",
        "publish_h6_checkpoint_catalog_entry_v3",
        "read_h6_checkpoint_catalog_v3",
    }
)
_VALIDATION_CAMPAIGN_V3_EXPORTS = frozenset(
    {
        "H6ValidationCampaignResultV3",
        "h6_tuning_selection_directory_v3",
        "publish_h6_tuning_selection_v3",
        "read_h6_tuning_selection_v3",
        "run_h6_validation_campaign_v3",
    }
)
_VALIDATION_V3_EXPORTS = frozenset(
    {
        "H6EvaluationArmV3",
        "build_h6_evaluation_arm_v3",
        "score_h6_validation_checkpoint_v3",
    }
)
_TRAINING_ATTEMPT_V3_EXPORTS = frozenset(
    {
        "H6AttemptHistoryShardV3",
        "H6AttemptMetricHistoryRecordV3",
        "H6AttemptRecoveryBoundaryV3",
        "H6GenerativePriorFeatureProviderV3",
        "H6RecoveredTrainingAttemptV3",
        "H6TrainingAttemptHistoryV3",
        "H6TrainingAttemptProgressV3",
        "H6TrainingAttemptResultV3",
        "H6ValidationBoundaryHistoryRecordV3",
        "H6_VALIDATION_BOUNDARY_CONTRACT_SHA256_V3",
        "execute_h6_training_attempt_v3",
        "h6_training_attempt_progress_path_v3",
        "read_h6_training_attempt_history_v3",
        "read_h6_training_attempt_progress_v3",
        "recover_h6_training_attempt_v3",
        "reopen_h6_terminal_training_attempt_v3",
        "run_h6_training_attempt_v3",
    }
)
_TEST_TRANSACTION_V3_EXPORTS = frozenset(
    {
        "execute_h6_test_transaction_v3",
        "finalize_h6_test_transaction_v3",
        "read_h6_prediction_pointer_v3",
        "read_h6_test_reservation_v3",
        "read_h6_test_terminal_v3",
        "recover_h6_test_transaction_v3",
    }
)


def __getattr__(name: str) -> object:
    """Load effectful H6 orchestration surfaces only when requested."""

    if name in _READINESS_EXPORTS:
        from . import h6_readiness

        return getattr(h6_readiness, name)
    if name in _EXPERIMENT_EXPORTS:
        from . import h6_experiment

        return getattr(h6_experiment, name)
    if name in _EXPERIMENT_V3_EXPORTS:
        from . import h6_experiment_v3

        return getattr(h6_experiment_v3, name)
    if name in _CHECKPOINT_CATALOG_V3_EXPORTS:
        from . import h6_checkpoint_catalog_v3

        return getattr(h6_checkpoint_catalog_v3, name)
    if name in _VALIDATION_CAMPAIGN_V3_EXPORTS:
        from . import h6_validation_campaign_v3

        return getattr(h6_validation_campaign_v3, name)
    if name in _VALIDATION_V3_EXPORTS:
        from . import h6_validation_v3

        return getattr(h6_validation_v3, name)
    if name in _TRAINING_ATTEMPT_V3_EXPORTS:
        from . import h6_training_attempt_v3

        return getattr(h6_training_attempt_v3, name)
    if name in _TEST_TRANSACTION_V3_EXPORTS:
        from . import h6_test_transaction_v3

        return getattr(h6_test_transaction_v3, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "A0FactoryInputs",
    "A0FlopLedger",
    "A0FlopTerm",
    "A0FlopWorkload",
    "A0MatchRow",
    "A0ParameterInventory",
    "A5_REFERENCE_PARAMETER_COUNT",
    "AMENDED_MATCHING_SCHEDULE_POLICY",
    "AMENDED_EMISSION_WIDTH_CANDIDATES",
    "AMENDED_LATENT_WIDTH_CANDIDATES",
    "AMENDED_RECOGNITION_WIDTH_CANDIDATES",
    "ARM_MATRIX_ROWS",
    "ARM_MATRIX_SHA256",
    "ArmConfig",
    "ArmExecutionRuntime",
    "ArmMatchingReport",
    "ArmMatrixRow",
    "ArmPathTrace",
    "AmendedMatchingSchedulePolicy",
    "ArmObjectiveInventory",
    "ArmTargetFreeProposalAdapter",
    "ArmTrainingObjectiveAdapter",
    "BuiltArm",
    "BuiltWT103Arm",
    "CapacityAllocation",
    "CurrentPredictionPrerequisiteRefs",
    "DetachedRecognitionLawSnapshot",
    "ExecutionScope",
    "EndpointFormulaProfile",
    "FlashAttentionObservation",
    "FlopTerm",
    "ForbiddenStorageRequest",
    "ForwardTerms",
    "H6AnalyticalFlopLedger",
    "H6A0ArchitectureProfile",
    "H6A0ValidationProfile",
    "H6AttemptCursor",
    "H6AttemptHistoryShardV3",
    "H6AttemptMetricHistoryRecordV3",
    "H6AttemptRecoveryBoundaryV3",
    "H6AttemptSpec",
    "H6CheckpointCatalogEntryV3",
    "H6CheckpointCatalogItemV3",
    "H6CheckpointCatalogV3",
    "H6CheckpointManifest",
    "H6CrossEntropyTerms",
    "H6CausalTransformer",
    "H6ExperimentRunResult",
    "H6EvaluationArmV3",
    "H6FormulaSelection",
    "H6PrimaryJointCandidate",
    "H6PrimaryJointSelection",
    "H6PredictionV3PrerequisiteEvidence",
    "H6ObjectiveManifest",
    "H6ReducedLanguageElboTerms",
    "H6RecoveredTrainingAttemptV3",
    "H6TrainingAuthorization",
    "H6TrainingAttemptHistoryV3",
    "H6TrainingAttemptProgressV3",
    "H6TrainingAttemptResultV3",
    "H6TrainingWorkload",
    "H6TypedTrainingObjective",
    "H6ValidationBoundaryHistoryRecordV3",
    "H6ValidationCampaignResultV3",
    "H6GenerativePriorFeatureProviderV3",
    "H6_VALIDATION_BOUNDARY_CONTRACT_SHA256_V3",
    "H6_TARGET_FREE_DATA_SAFETY_SHA256",
    "H7FixedSourceAssemblyReceipt",
    "H7FixedSourceAssemblySpec",
    "LatentLanguageArmModel",
    "MeanPooledPrefixFloor",
    "MatchingReport",
    "NamedParameterShape",
    "NegativeControlRecord",
    "OptimizerParameterBinding",
    "OptimizerBinding",
    "ParameterRoleRecord",
    "ParameterCountAssessment",
    "PredictorPerturbationObservation",
    "ProposalEvidence",
    "PROPOSED_PREFIX_PRIOR_CONTEXT_WIDTH",
    "RecognitionSnapshot",
    "ScientificStateParticipant",
    "StaticScientificArtifactRef",
    "StaticScientificPreconditionRecord",
    "StepResult",
    "TensorStorageObservation",
    "TrainingEngineError",
    "TrainingSparsityAudit",
    "WT103_STRUCTURED_FACTOR_ELBO_SCHEMA",
    "WT103_STRUCTURED_FACTOR_ELBO_SCHEMA_SHA256",
    "WT103A0Model",
    "WT103ArmBuildRecord",
    "WT103ArmRuntimeComponents",
    "WT103FactorySetIdentity",
    "WT103PredictorSafetyCertificate",
    "AttemptEventSink",
    "AttemptResult",
    "a0_primitive_formula_sha256",
    "analytical_training_flop_ledger",
    "arm_parameter_count",
    "ProducerCompatibilityError",
    "arm_matrix_sha256",
    "audit_arm_matching",
    "audit_wt103_arm_matching",
    "build_a0",
    "build_a1",
    "build_a2",
    "build_a3",
    "build_a4",
    "build_a5",
    "build_arm",
    "build_h7_fixed_a5_arm",
    "build_a0_architecture_profile",
    "build_a0_formula_record",
    "build_h6_evaluation_arm_v3",
    "build_wt103_a0",
    "build_wt103_a5_fixed",
    "build_wt103_a5_nolatent",
    "build_wt103_a5_parent_specific",
    "build_wt103_arm",
    "certify_training_sparsity",
    "certify_wt103_predictor_safety",
    "fixed_source_prior_parameter_count",
    "h6_a0_parameter_count",
    "h6_training_attempt_progress_path_v3",
    "h6_tuning_selection_directory_v3",
    "endpoint_formula_profile",
    "execute_h6_training_attempt_v3",
    "execute_h6_test_transaction_v3",
    "finalize_h6_test_transaction_v3",
    "literal_arm_semantic_payload",
    "load_h6_checkpoint",
    "mean_pooled_no_latent_parameter_count",
    "outcome_blind_feasibility_assessments",
    "parent_specific_pooled_prefix_source_prior_parameter_count",
    "parameter_count_within_tolerance",
    "plan_h6_attempt",
    "prepare_h6_test_transaction_v3",
    "publish_h6_checkpoint_catalog_entry_v3",
    "publish_h6_tuning_selection_v3",
    "recognition_parameter_count",
    "require_h7_fixed_source_assembly",
    "read_h6_prediction_pointer_v3",
    "read_h6_checkpoint_catalog_v3",
    "read_h6_bounded_prefix_certificate_set_for_scoring_v3",
    "read_h6_training_attempt_history_v3",
    "read_h6_training_attempt_progress_v3",
    "reopen_h6_prediction_v3_prerequisite_evidence",
    "read_h6_test_reservation_v3",
    "read_h6_test_terminal_v3",
    "read_h6_tuning_selection_v3",
    "recover_h6_training_attempt_v3",
    "recover_h6_test_transaction_v3",
    "reconstruct_a0_flops",
    "reconstruct_a0_parameters",
    "reopen_h6_terminal_training_attempt_v3",
    "save_h6_checkpoint",
    "select_outcome_blind_allocation",
    "select_parent_specific_primary_allocation",
    "shared_a2_a5_semantic_payload",
    "scorer_dispatch",
    "stable_parameter_key",
    "train_h6_attempt",
    "train_attempt",
    "train_step",
    "run_h6_experiment",
    "run_h6_experiment_v3",
    "run_h6_training_attempt_v3",
    "run_h6_validation_campaign_v3",
    "score_h6_validation_checkpoint_v3",
    "validate_h6_prediction_readiness",
    "validate_existing_h6_prediction_readiness_v3",
    "validate_h6_prediction_readiness_v3",
    "validate_static_scientific_preconditions",
    "guard_flash_attention_request",
    "guard_tensor_request",
    "run_sparsity_negative_controls",
]
