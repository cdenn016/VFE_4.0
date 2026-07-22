"""Public numerical primitives for deterministic H1 evaluation."""

from .categorical import categorical_kl, require_probability_vector, selected_log_softmax
from .gaussian import gaussian_log_prob, require_spd
from .information import InformationGaussian
from .h5_budget import (
    DEFAULT_H5_BUDGET_CONFIG,
    H5BudgetConfig,
    H5CompleteAllowance,
    H5DeltaAllowance,
    H5TermAllowance,
    complete_elbo_allowance,
    epsilon_delta,
    subtraction_rounding_allowance,
    term_allowance,
)
from .linear_gaussian import add_initial_gaussian, add_scalar_conditional
from .precision import DenseCholeskyPrecision
from .quadrature import probabilists_gauss_hermite

__all__ = [
    "categorical_kl",
    "DenseCholeskyPrecision",
    "gaussian_log_prob",
    "DEFAULT_H5_BUDGET_CONFIG",
    "H5BudgetConfig",
    "H5CompleteAllowance",
    "H5DeltaAllowance",
    "H5TermAllowance",
    "InformationGaussian",
    "add_initial_gaussian",
    "add_scalar_conditional",
    "probabilists_gauss_hermite",
    "require_probability_vector",
    "require_spd",
    "selected_log_softmax",
    "complete_elbo_allowance",
    "epsilon_delta",
    "subtraction_rounding_allowance",
    "term_allowance",
]
