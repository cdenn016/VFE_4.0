"""NumPy-only reference oracles."""

from verification.numpy_oracles.h1_elbo import (
    H1EvidenceRecord,
    H1IdentityRecord,
    IndependentNumericalAllowance,
    IndependentTermAllowances,
    IndependentTermRecord,
    h1_all_observation_evidences,
    h1_evidence_and_posterior_kl,
    h1_evidence_enumeration_pair,
    h1_factorized_time_evidence,
    h1_local_diagnostics,
    h1_log_evidence,
    h1_p_weights_q_components_evidence,
    h1_permuted_zm_evidence,
    h1_q_weights_p_components_evidence,
    h1_wrong_recognition_mixture_evidence,
)
from verification.numpy_oracles.h2_moment import (
    H2MomentOracleComponent,
    H2MomentOracleEvaluation,
    H2OracleGaussian,
    H2OracleLocalTerms,
    H2OracleMarginal,
    evaluate_h2_moment_oracle,
)
from verification.numpy_oracles.h3_posterior import (
    H3PosteriorOracleEvaluation,
    evaluate_h3_posterior_oracle,
    reverse_kl_to_oracle,
)

__all__ = [
    "H1EvidenceRecord",
    "H1IdentityRecord",
    "IndependentNumericalAllowance",
    "IndependentTermAllowances",
    "IndependentTermRecord",
    "h1_all_observation_evidences",
    "h1_evidence_and_posterior_kl",
    "h1_evidence_enumeration_pair",
    "h1_factorized_time_evidence",
    "h1_local_diagnostics",
    "h1_log_evidence",
    "h1_p_weights_q_components_evidence",
    "h1_permuted_zm_evidence",
    "h1_q_weights_p_components_evidence",
    "h1_wrong_recognition_mixture_evidence",
    "H2MomentOracleComponent",
    "H2MomentOracleEvaluation",
    "H2OracleGaussian",
    "H2OracleLocalTerms",
    "H2OracleMarginal",
    "evaluate_h2_moment_oracle",
    "H3PosteriorOracleEvaluation",
    "evaluate_h3_posterior_oracle",
    "reverse_kl_to_oracle",
]
