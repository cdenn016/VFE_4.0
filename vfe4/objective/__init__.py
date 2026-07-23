"""Production objectives for the frozen H1 reference calculation."""

from vfe4.objective.language_elbo import (
    ExactSourceMixtureLaw,
    ExpectationEvaluationMethod,
    FactorPartition,
    LanguageElboExpectation,
    MomentProjectedLaw,
    RecognitionConditioningMode,
    RecognitionFamily,
    evaluate_emission_only_ablation,
    evaluate_language_elbo,
)
from vfe4.types.h6 import (
    EmissionOnlyAblationTerms,
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
from vfe4.objective.h1_monolithic import MonolithicElboResult, evaluate_monolithic_elbo
from vfe4.objective.h3_gaussian import (
    H3ObjectiveEvaluation,
    evaluate_h3_elbo,
    evaluate_h3_elbo_difference,
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

__all__ = [
    "EmissionOnlyAblationTerms",
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
    "H6FactorTerm",
    "H6LanguageElboTerms",
    "LanguageElboExpectation",
    "MomentProjectedLaw",
    "MonolithicElboResult",
    "RoundingInputs",
    "RecognitionConditioningMode",
    "RecognitionFamily",
    "StaleFactorCacheError",
    "build_h5_reference_dependency_graph",
    "evaluate_information_elbo",
    "evaluate_emission_only_ablation",
    "evaluate_h3_elbo",
    "evaluate_h3_elbo_difference",
    "evaluate_h5_complete_elbo",
    "evaluate_local_elbo",
    "evaluate_language_elbo",
    "evaluate_monolithic_elbo",
    "expected_affected_factors",
]
