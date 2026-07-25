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

_H8_LAZY_EXPORTS = frozenset(
    {
        "BlockCanonicalAssembler",
        "BlockMomentBlocks",
        "BlockTridiagonalCholesky",
        "FactorBackedInformationGaussian",
    }
)


def __getattr__(name: str) -> object:
    """Load H8 primitives after ``vfe4.types.h8`` finishes initializing."""

    if name == "BlockCanonicalAssembler":
        from .block_canonical import BlockCanonicalAssembler

        return BlockCanonicalAssembler
    if name == "BlockTridiagonalCholesky":
        from .block_tridiagonal import BlockTridiagonalCholesky

        return BlockTridiagonalCholesky
    if name in ("BlockMomentBlocks", "FactorBackedInformationGaussian"):
        from . import sparse_information

        return getattr(sparse_information, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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
