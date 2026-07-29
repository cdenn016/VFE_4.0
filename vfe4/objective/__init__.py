"""Production objectives for the frozen H1 reference calculation."""

import importlib

from vfe4.objective.language_elbo import (
    CompleteLanguageELBOFactorTrace,
    ExactSourceMixtureLaw,
    ExpectationEvaluationMethod,
    FactorPartition,
    H7AuthenticatedEvaluation,
    LanguageElboExpectation,
    LiveEmissionExpectation,
    LiveEmissionExpectationContext,
    MixtureMode,
    MomentProjectedLaw,
    PriorVariant,
    RecognitionConditioningMode,
    RecognitionFamily,
    capture_h7_complete_language_elbo,
    evaluate_emission_only_ablation,
    require_h7_complete_factor_trace,
    require_source_law_for_endpoint,
)
from vfe4.types.h6 import (
    EmissionOnlyAblationTerms,
    H6EndpointLanguageElboTerms,
    H6FactorTerm,
    H6LanguageElboTerms,
)

from vfe4.objective.dependency_graph import (
    FactorDependencyGraph,
    build_h5_reference_dependency_graph,
    expected_affected_factors,
)

from vfe4.objective.h2_information import (
    H2ComponentDiagnostics,
    H2ComponentTerms,
    H2InformationEvaluation,
    RoundingInputs,
    evaluate_information_elbo,
)
from vfe4.objective.h1_local import evaluate_local_elbo
from vfe4.objective.h8_sparse import (
    evaluate_h8_sparse_objective,
    h8_emission_expectation,
)
from vfe4.objective.h1_monolithic import (
    MonolithicElboResult,
    evaluate_monolithic_elbo,
)
from vfe4.objective.h3_gaussian import (
    H3ObjectiveEvaluation,
    evaluate_h3_elbo,
    evaluate_h3_elbo_difference,
)
from vfe4.objective.h7_law_components import (
    build_h7_law_components,
    derive_h7_source_assembly_profile,
)
from vfe4.objective.h5_complete import (
    CacheDisposition,
    CompleteElboEvaluation,
    CompleteElboEvaluator,
    FactorCacheEntry,
    FactorCacheKey,
    FactorEvaluationRecord,
    FactorInputHashRecord,
    StaleFactorCacheError,
    evaluate_h5_complete_elbo,
)
from vfe4.objective.h6_prediction_v3 import (
    ExactSourceMixtureEvaluationV3,
    H6ActiveParameterBlockV3,
    H6EvaluatedRecognitionLawV3,
    H6MixtureModeV3,
    H6ObjectivePartitionV3,
    H6ObjectiveTermV3,
    H6PredictionObjectiveEstimateV3,
    H6SourceRowEvaluationV3,
    H6TerminalComponentEvaluationV3,
    MomentProjectionEvaluationV3,
    evaluate_h6_no_latent_cross_entropy_v3,
    evaluate_h6_prediction_elbo_v3,
    project_terminal_mixture_v3,
)

_H7_LAW_EVIDENCE_EXPORTS = frozenset(
    {
        "H7LawEvaluationEvidence",
        "build_h7_grouped_elbo_record",
        "capture_h7_law_evaluation",
        "require_h7_law_evaluation",
    }
)
_H7_COVARIANCE_EXPORTS = frozenset(
    {
        "H7_COMPLETE_LOCAL_TERM_IDS",
        "H7_INDEPENDENT_H1_NORMALIZATION_IDENTITY_SHA256",
        "H7_INDEPENDENT_H1_PRODUCER_IDENTITY_SHA256",
        "H7_MATRIX_EVIDENCE_NOT_APPLICABLE_REASON",
        "H7_MATRIX_SCORER_RESIDUAL_IDS",
        "H7_POINTWISE_P_SHIFT_INVARIANT_ID",
        "H7_POINTWISE_Q_SHIFT_INVARIANT_ID",
        "H7IndependentH1EvidenceRecord",
        "evaluate_h7_complete_covariance",
        "evaluate_h7_law_pair_covariance",
        "h7_joint_gaussian_kl",
        "require_h7_complete_term_inventory",
        "require_h7_matrix_scorer_residual_inventory",
    }
)


def __getattr__(name: str) -> object:
    """Load training-dependent H7 exports without creating an import cycle."""

    module_name: str
    if name in _H7_LAW_EVIDENCE_EXPORTS:
        module_name = "vfe4.objective.h7_law_evidence"
    elif name in _H7_COVARIANCE_EXPORTS:
        module_name = "vfe4.objective.h7_covariance"
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


__all__ = [
    "EmissionOnlyAblationTerms",
    "CompleteLanguageELBOFactorTrace",
    "ExactSourceMixtureEvaluationV3",
    "ExactSourceMixtureLaw",
    "ExpectationEvaluationMethod",
    "FactorPartition",
    "FactorDependencyGraph",
    "CacheDisposition",
    "CompleteElboEvaluation",
    "CompleteElboEvaluator",
    "FactorCacheEntry",
    "FactorCacheKey",
    "FactorEvaluationRecord",
    "FactorInputHashRecord",
    "H2ComponentDiagnostics",
    "H2ComponentTerms",
    "H2InformationEvaluation",
    "H3ObjectiveEvaluation",
    "H6ActiveParameterBlockV3",
    "H6EvaluatedRecognitionLawV3",
    "H6MixtureModeV3",
    "H6ObjectivePartitionV3",
    "H6ObjectiveTermV3",
    "H6PredictionObjectiveEstimateV3",
    "H6SourceRowEvaluationV3",
    "H6TerminalComponentEvaluationV3",
    "H7_COMPLETE_LOCAL_TERM_IDS",
    "H7_INDEPENDENT_H1_NORMALIZATION_IDENTITY_SHA256",
    "H7_INDEPENDENT_H1_PRODUCER_IDENTITY_SHA256",
    "H7_MATRIX_EVIDENCE_NOT_APPLICABLE_REASON",
    "H7_MATRIX_SCORER_RESIDUAL_IDS",
    "H7_POINTWISE_P_SHIFT_INVARIANT_ID",
    "H7_POINTWISE_Q_SHIFT_INVARIANT_ID",
    "H7IndependentH1EvidenceRecord",
    "H7AuthenticatedEvaluation",
    "H7LawEvaluationEvidence",
    "H6FactorTerm",
    "H6EndpointLanguageElboTerms",
    "H6LanguageElboTerms",
    "LanguageElboExpectation",
    "LiveEmissionExpectation",
    "LiveEmissionExpectationContext",
    "MixtureMode",
    "MomentProjectedLaw",
    "MomentProjectionEvaluationV3",
    "PriorVariant",
    "MonolithicElboResult",
    "RoundingInputs",
    "RecognitionConditioningMode",
    "RecognitionFamily",
    "StaleFactorCacheError",
    "build_h5_reference_dependency_graph",
    "build_h7_grouped_elbo_record",
    "build_h7_law_components",
    "capture_h7_complete_language_elbo",
    "capture_h7_law_evaluation",
    "derive_h7_source_assembly_profile",
    "evaluate_information_elbo",
    "evaluate_emission_only_ablation",
    "evaluate_h3_elbo",
    "evaluate_h3_elbo_difference",
    "evaluate_h5_complete_elbo",
    "evaluate_h6_no_latent_cross_entropy_v3",
    "evaluate_h6_prediction_elbo_v3",
    "evaluate_h7_complete_covariance",
    "evaluate_h7_law_pair_covariance",
    "evaluate_local_elbo",
    "evaluate_h8_sparse_objective",
    "h8_emission_expectation",
    "project_terminal_mixture_v3",
    "evaluate_monolithic_elbo",
    "expected_affected_factors",
    "h7_joint_gaussian_kl",
    "require_h7_complete_term_inventory",
    "require_h7_complete_factor_trace",
    "require_h7_law_evaluation",
    "require_h7_matrix_scorer_residual_inventory",
    "require_source_law_for_endpoint",
]
