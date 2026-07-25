"""Public numerical primitives for deterministic H1 evaluation."""

from .categorical import (
    AllInvalidSourceRowError,
    MaskedLogProbabilities,
    categorical_kl,
    masked_log_softmax_from_parents,
    require_probability_vector,
    selected_log_softmax,
)
from .block_layout import (
    H8_MAX_STORAGE_SCALARS,
    H8_REFERENCE_CHANNEL_DIMENSION,
    H8_REFERENCE_HORIZON,
    BlockChainLayout,
    BlockId,
)
from .block_tridiagonal import BlockTridiagonalCholesky
from .block_canonical import BlockCanonicalAssembler
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
from .sparse_information import BlockMomentBlocks, FactorBackedInformationGaussian

__all__ = [
    "AllInvalidSourceRowError",
    "BlockChainLayout",
    "BlockCanonicalAssembler",
    "BlockId",
    "BlockMomentBlocks",
    "BlockTridiagonalCholesky",
    "categorical_kl",
    "DenseCholeskyPrecision",
    "gaussian_log_prob",
    "DEFAULT_H5_BUDGET_CONFIG",
    "H5BudgetConfig",
    "H5CompleteAllowance",
    "H5DeltaAllowance",
    "H5TermAllowance",
    "H8_MAX_STORAGE_SCALARS",
    "H8_REFERENCE_CHANNEL_DIMENSION",
    "H8_REFERENCE_HORIZON",
    "InformationGaussian",
    "FactorBackedInformationGaussian",
    "MaskedLogProbabilities",
    "add_initial_gaussian",
    "add_scalar_conditional",
    "probabilists_gauss_hermite",
    "masked_log_softmax_from_parents",
    "require_probability_vector",
    "require_spd",
    "selected_log_softmax",
    "complete_elbo_allowance",
    "epsilon_delta",
    "subtraction_rounding_allowance",
    "term_allowance",
]
