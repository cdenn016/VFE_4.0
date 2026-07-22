"""Public numerical primitives for deterministic H1 evaluation."""

from .categorical import categorical_kl, require_probability_vector, selected_log_softmax
from .gaussian import gaussian_log_prob, require_spd
from .information import InformationGaussian
from .linear_gaussian import add_initial_gaussian, add_scalar_conditional
from .precision import DenseCholeskyPrecision
from .quadrature import probabilists_gauss_hermite

__all__ = [
    "categorical_kl",
    "DenseCholeskyPrecision",
    "gaussian_log_prob",
    "InformationGaussian",
    "add_initial_gaussian",
    "add_scalar_conditional",
    "probabilists_gauss_hermite",
    "require_probability_vector",
    "require_spd",
    "selected_log_softmax",
]
