"""Block-local canonical assembly for the H8 Gaussian chains.

This module deliberately has no population-vector or dense-precision helper.
Every update addresses one diagonal block or one neighboring lower block.
"""

from __future__ import annotations

import torch
from torch import Tensor

from vfe4.numerics.block_layout import BlockChainLayout
from vfe4.types.h8 import BlockTridiagonalPrecision


class BlockCanonicalAssembler:
    """Mutable, single-use accumulator for normalized local Gaussian factors."""

    def __init__(self, layout: BlockChainLayout) -> None:
        if type(layout) is not BlockChainLayout:
            raise ValueError("layout must be a BlockChainLayout")
        block = layout.block_size
        self._layout = layout
        self._h = torch.zeros((layout.population_size, block), dtype=torch.float64)
        self._diag = torch.zeros(
            (layout.population_size, block, block), dtype=torch.float64
        )
        self._lower = torch.zeros((layout.horizon, block, block), dtype=torch.float64)
        self._frozen = False

    @property
    def layout(self) -> BlockChainLayout:
        return self._layout

    def add_initial(self, mean: Tensor, covariance: Tensor) -> None:
        """Scatter one normalized Gaussian law for the initial combined slice."""

        self._require_open()
        checked_mean = self._vector(mean, "initial mean")
        precision = self._precision(covariance, "initial covariance")
        self._diag[0].add_(precision)
        self._h[0].add_(precision @ checked_mean)

    def add_transition(
        self,
        receiver_t: int,
        matrix: Tensor,
        offset: Tensor,
        covariance: Tensor,
    ) -> None:
        """Scatter ``y_t | y_{t-1} ~ N(A y_{t-1}+c,R)`` locally."""

        self._require_open()
        if type(receiver_t) is not int or not 1 <= receiver_t <= self._layout.horizon:
            raise ValueError("receiver_t must be in 1..T")
        matrix = self._matrix(matrix, "transition matrix")
        offset = self._vector(offset, "transition offset")
        precision = self._precision(covariance, "transition covariance")
        parent = receiver_t - 1
        self._diag[parent].add_(matrix.T @ precision @ matrix)
        self._diag[receiver_t].add_(precision)
        self._lower[parent].add_(-precision @ matrix)
        self._h[parent].add_(-(matrix.T @ precision @ offset))
        self._h[receiver_t].add_(precision @ offset)

    def add_local_observation(
        self,
        receiver_t: int,
        design: Tensor,
        offset: Tensor,
        observation: Tensor,
        covariance: Tensor,
    ) -> None:
        """Scatter ``x | y_t ~ N(C y_t+c,R)`` into one diagonal block."""

        self._require_open()
        if type(receiver_t) is not int or not 0 <= receiver_t <= self._layout.horizon:
            raise ValueError("receiver_t must be in 0..T")
        if type(design) is not Tensor or design.dtype is not torch.float64 or design.ndim != 2:
            raise ValueError("observation design must be a float64 matrix")
        if design.device.type != "cpu" or design.shape[1] != self._layout.block_size:
            raise ValueError("observation design has incompatible local width")
        width = int(design.shape[0])
        offset = self._vector_of_width(offset, width, "observation offset")
        observation = self._vector_of_width(observation, width, "observation")
        precision = self._precision_of_width(covariance, width, "observation covariance")
        residual_offset = observation - offset
        self._diag[receiver_t].add_(design.T @ precision @ design)
        self._h[receiver_t].add_(design.T @ precision @ residual_offset)

    # The longer name remains useful at call sites that distinguish observations
    # from direct local Gaussian anchors.
    add_local_linear_gaussian = add_local_observation

    def freeze(self) -> tuple[BlockTridiagonalPrecision, Tensor]:
        """Return owned information blocks and permanently close this assembler."""

        self._require_open()
        if not bool(torch.isfinite(self._h).all()) or not bool(torch.isfinite(self._diag).all()) or not bool(torch.isfinite(self._lower).all()):
            raise ValueError("canonical assembly produced nonfinite blocks")
        if not bool(torch.equal(self._diag, self._diag.transpose(-1, -2))):
            raise ValueError("canonical diagonal blocks must be symmetric")
        self._frozen = True
        return BlockTridiagonalPrecision(self._layout, self._diag, self._lower), self._h.detach().clone().contiguous()

    def _require_open(self) -> None:
        if self._frozen:
            raise RuntimeError("canonical assembler is frozen")

    def _vector(self, value: object, name: str) -> Tensor:
        return self._vector_of_width(value, self._layout.block_size, name)

    def _vector_of_width(self, value: object, width: int, name: str) -> Tensor:
        if type(value) is not Tensor or value.dtype is not torch.float64 or value.device.type != "cpu" or tuple(value.shape) != (width,) or not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be a finite CPU float64 vector")
        return value

    def _matrix(self, value: object, name: str) -> Tensor:
        width = self._layout.block_size
        if type(value) is not Tensor or value.dtype is not torch.float64 or value.device.type != "cpu" or tuple(value.shape) != (width, width) or not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be a finite CPU float64 [b,b] block")
        return value

    def _precision(self, covariance: object, name: str) -> Tensor:
        return self._precision_of_width(covariance, self._layout.block_size, name)

    def _precision_of_width(self, covariance: object, width: int, name: str) -> Tensor:
        if type(covariance) is not Tensor or covariance.dtype is not torch.float64 or covariance.device.type != "cpu" or tuple(covariance.shape) != (width, width) or not bool(torch.isfinite(covariance).all()):
            raise ValueError(f"{name} must be a finite CPU float64 square block")
        if not bool(torch.equal(covariance, covariance.T)):
            raise ValueError(f"{name} must be symmetric")
        chol, info = torch.linalg.cholesky_ex(covariance, check_errors=False)
        if int(info.item()) != 0:
            raise ValueError(f"{name} must be strictly positive definite")
        identity = torch.eye(width, dtype=torch.float64)
        return torch.cholesky_solve(identity, chol)


__all__ = ["BlockCanonicalAssembler"]
