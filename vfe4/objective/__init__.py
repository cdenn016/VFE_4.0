"""Production objectives for the frozen H1 reference calculation."""

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
    "MonolithicElboResult",
    "RoundingInputs",
    "StaleFactorCacheError",
    "build_h5_reference_dependency_graph",
    "evaluate_information_elbo",
    "evaluate_h3_elbo",
    "evaluate_h3_elbo_difference",
    "evaluate_h5_complete_elbo",
    "evaluate_local_elbo",
    "evaluate_monolithic_elbo",
    "expected_affected_factors",
]
