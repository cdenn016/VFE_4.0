"""Production objectives for the frozen H1 reference calculation."""

from vfe4.objective.h2_information import (
    H2ComponentDiagnostics,
    H2ComponentTerms,
    H2InformationEvaluation,
    RoundingInputs,
    evaluate_information_elbo,
)
from vfe4.objective.h1_local import evaluate_local_elbo
from vfe4.objective.h1_monolithic import MonolithicElboResult, evaluate_monolithic_elbo
from vfe4.objective.h3_gaussian import H3ObjectiveEvaluation, evaluate_h3_elbo

__all__ = [
    "H2ComponentDiagnostics",
    "H2ComponentTerms",
    "H2InformationEvaluation",
    "H3ObjectiveEvaluation",
    "MonolithicElboResult",
    "RoundingInputs",
    "evaluate_information_elbo",
    "evaluate_h3_elbo",
    "evaluate_local_elbo",
    "evaluate_monolithic_elbo",
]
