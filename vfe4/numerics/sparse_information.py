"""Information Gaussians backed only by an H8 block precision factor."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor

from vfe4.numerics.block_layout import BlockChainLayout
from vfe4.types.h8 import BlockPrecisionFactor, SelectedInverseBlocks


@dataclass(frozen=True, slots=True, init=False)
class BlockMomentBlocks:
    """Selected second moments, stored as diagonal and lower-adjacent blocks."""

    layout: BlockChainLayout
    _diag: Tensor = field(repr=False)
    _lower: Tensor = field(repr=False)

    def __init__(self, layout: BlockChainLayout, diagonal: Tensor, lower: Tensor) -> None:
        if type(layout) is not BlockChainLayout:
            raise ValueError("layout must be a BlockChainLayout")
        expected_diag = (layout.population_size, layout.block_size, layout.block_size)
        expected_lower = (layout.horizon, layout.block_size, layout.block_size)
        for value, shape, name in ((diagonal, expected_diag, "diagonal"), (lower, expected_lower, "lower")):
            if type(value) is not Tensor or value.dtype is not torch.float64 or value.device.type != "cpu" or tuple(value.shape) != shape or not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} moments must be finite CPU float64 blocks")
        object.__setattr__(self, "layout", layout)
        object.__setattr__(self, "_diag", diagonal.detach().clone().contiguous())
        object.__setattr__(self, "_lower", lower.detach().clone().contiguous())

    @property
    def diagonal(self) -> Tensor:
        return self._diag.detach().clone()

    @property
    def lower(self) -> Tensor:
        return self._lower.detach().clone()

    def _block_refs(self) -> tuple[Tensor, Tensor]:
        return self._diag, self._lower


@dataclass(frozen=True, slots=True, init=False)
class FactorBackedInformationGaussian:
    """An immutable block Gaussian retaining ``h`` and a factor, never ``J``."""

    layout: BlockChainLayout
    _h: Tensor = field(repr=False)
    factor: BlockPrecisionFactor = field(repr=False, compare=False)

    @classmethod
    def from_factor(
        cls,
        h: Tensor,
        factor: BlockPrecisionFactor,
    ) -> "FactorBackedInformationGaussian":
        if not isinstance(factor, BlockPrecisionFactor):
            raise ValueError("factor must satisfy the H8 block-factor protocol")
        layout = factor.layout
        if type(layout) is not BlockChainLayout:
            raise ValueError("factor must expose a BlockChainLayout")
        if type(h) is not Tensor or h.dtype is not torch.float64 or h.device.type != "cpu" or tuple(h.shape) != (layout.population_size, layout.block_size) or not bool(torch.isfinite(h).all()):
            raise ValueError("h must be a finite CPU float64 [N,b] block vector")
        result = cls.__new__(cls)
        object.__setattr__(result, "layout", layout)
        object.__setattr__(result, "_h", h.detach().clone().contiguous())
        object.__setattr__(result, "factor", factor)
        # Eagerly establish the finite factor-backed contracts while the source
        # inputs are still in scope; no precision input is retained.
        result.mean()
        result.log_normalizer()
        return result

    @property
    def dimension(self) -> int:
        return self.layout.dimension

    @property
    def h(self) -> Tensor:
        return self._h.detach().clone()

    def mean(self) -> Tensor:
        result = self.factor.solve(self._h)
        self._require_block_vector(result, "mean")
        return result.detach().clone()

    def log_normalizer(self) -> Tensor:
        mean = self.mean()
        inner_product = torch.zeros((), dtype=torch.float64)
        for population in range(self.layout.population_size):
            inner_product = inner_product + torch.dot(self._h[population], mean[population])
        result = 0.5 * (inner_product - self.factor.logdet() + self.dimension * math.log(2.0 * math.pi))
        self._require_scalar(result, "log normalizer")
        return result.detach().clone()

    def entropy(self) -> Tensor:
        result = 0.5 * (self.dimension * (1.0 + math.log(2.0 * math.pi)) - self.factor.logdet())
        self._require_scalar(result, "entropy")
        return result.detach().clone()

    def log_prob(self, value: Tensor) -> Tensor:
        self._require_block_vector(value, "value")
        linear = torch.zeros((), dtype=torch.float64)
        for population in range(self.layout.population_size):
            linear = linear + torch.dot(self._h[population], value[population])
        result = linear - 0.5 * self.factor.quadratic(value) - self.log_normalizer()
        self._require_scalar(result, "log probability")
        return result.detach().clone()

    def selected_moment_blocks(self) -> BlockMomentBlocks:
        selected: SelectedInverseBlocks = self.factor.selected_inverse(self.layout.stored_block_ids)
        inverse_diag, inverse_lower = selected._block_refs()
        mean = self.mean()
        diagonal: list[Tensor] = []
        lower: list[Tensor] = []
        for population in range(self.layout.population_size):
            diagonal.append(inverse_diag[population] + torch.outer(mean[population], mean[population]))
            if population:
                lower.append(inverse_lower[population - 1] + torch.outer(mean[population], mean[population - 1]))
        return BlockMomentBlocks(self.layout, torch.stack(diagonal), torch.stack(lower))

    def sample(self, noise: Tensor) -> Tensor:
        self._require_block_vector(noise, "sample noise")
        result = self.mean() + self.factor.sample(noise)
        self._require_block_vector(result, "sample")
        return result.detach().clone()

    def _require_block_vector(self, value: object, name: str) -> None:
        if type(value) is not Tensor or value.dtype is not torch.float64 or value.device.type != "cpu" or tuple(value.shape) != (self.layout.population_size, self.layout.block_size) or not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be a finite CPU float64 [N,b] block vector")

    @staticmethod
    def _require_scalar(value: object, name: str) -> None:
        if type(value) is not Tensor or value.dtype is not torch.float64 or value.shape != () or not bool(torch.isfinite(value)):
            raise ValueError(f"{name} must be finite")


__all__ = ["BlockMomentBlocks", "FactorBackedInformationGaussian"]
