"""Public numerical primitives for deterministic H1 evaluation."""

from .categorical import categorical_kl, require_probability_vector, selected_log_softmax
from .gaussian import gaussian_log_prob, require_spd
from .quadrature import probabilists_gauss_hermite

__all__ = [
    "categorical_kl",
    "gaussian_log_prob",
    "probabilists_gauss_hermite",
    "require_probability_vector",
    "require_spd",
    "selected_log_softmax",
]
