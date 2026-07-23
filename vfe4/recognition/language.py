"""Typed differentiable recognition families for H6 language training."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor

from vfe4.types.h6 import FrozenTensorSnapshot


RecognitionMode = Literal["filtering", "smoothing"]


@dataclass(frozen=True)
class RecognitionConditioning:
    """Owned observations with regime-specific, receiver-indexed access."""

    mode: RecognitionMode
    horizon: int
    observed_tokens: FrozenTensorSnapshot

    def __post_init__(self) -> None:
        if self.mode not in ("filtering", "smoothing"):
            raise ValueError("mode must be filtering or smoothing")
        if type(self.horizon) is not int or self.horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        if type(self.observed_tokens) is not FrozenTensorSnapshot:
            raise ValueError("observed_tokens must be a FrozenTensorSnapshot")
        self.observed_tokens.assert_intact()
        if self.observed_tokens.shape != (self.horizon,):
            raise ValueError("observed_tokens must have shape (horizon,)")

    @classmethod
    def create(
        cls,
        *,
        mode: RecognitionMode,
        horizon: int,
        observed_tokens: Tensor,
    ) -> "RecognitionConditioning":
        if not isinstance(observed_tokens, Tensor):
            raise ValueError("observed_tokens must be a torch.Tensor")
        if type(horizon) is not int or horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        if observed_tokens.ndim != 1 or tuple(observed_tokens.shape) != (horizon,):
            raise ValueError("observed_tokens must have shape (horizon,)")
        return cls(mode, horizon, FrozenTensorSnapshot.capture(observed_tokens))

    def visible_tokens(self, receiver_t: int) -> Tensor:
        """Return exactly ``I_t``: ``x_<=t`` for filtering or full ``x``."""

        if (
            type(receiver_t) is not int
            or receiver_t < 0
            or receiver_t > self.horizon
        ):
            raise ValueError("receiver_t must lie in [0, horizon]")
        observed = self.observed_tokens.value()
        if self.mode == "smoothing":
            return observed
        return observed[:receiver_t]


def _require_gaussian_snapshots(
    conditioning: object,
    mean: object,
    precision_cholesky: object,
) -> tuple[Tensor, Tensor]:
    if type(conditioning) is not RecognitionConditioning:
        raise ValueError("conditioning must be a RecognitionConditioning")
    conditioning.__post_init__()
    if type(mean) is not FrozenTensorSnapshot:
        raise ValueError("mean must be a FrozenTensorSnapshot")
    if type(precision_cholesky) is not FrozenTensorSnapshot:
        raise ValueError("precision_cholesky must be a FrozenTensorSnapshot")
    mean.assert_intact()
    precision_cholesky.assert_intact()
    live_mean = mean.value()
    live_cholesky = precision_cholesky.value()
    if live_mean.ndim != 1 or live_mean.numel() == 0:
        raise ValueError("mean must be a nonempty vector")
    dimension = live_mean.numel()
    if live_cholesky.shape != (dimension, dimension):
        raise ValueError("precision_cholesky must be square and match mean")
    if not live_mean.is_floating_point() or not live_cholesky.is_floating_point():
        raise ValueError("language Gaussian tensors must be floating point")
    if live_mean.dtype != live_cholesky.dtype:
        raise ValueError("mean and precision_cholesky must share a dtype")
    if live_mean.device != live_cholesky.device:
        raise ValueError("mean and precision_cholesky must share a device")
    if not bool(torch.isfinite(live_mean).all()) or not bool(
        torch.isfinite(live_cholesky).all()
    ):
        raise ValueError("language Gaussian tensors must be finite")
    if not torch.equal(live_cholesky, torch.tril(live_cholesky)):
        raise ValueError("precision_cholesky must be lower triangular")
    if not bool(torch.all(torch.diagonal(live_cholesky) > 0.0)):
        raise ValueError("precision_cholesky must have a positive diagonal")
    return live_mean, live_cholesky


def _capture_gaussian(
    *,
    conditioning: RecognitionConditioning,
    mean: Tensor,
    precision_cholesky: Tensor,
) -> tuple[RecognitionConditioning, FrozenTensorSnapshot, FrozenTensorSnapshot]:
    if type(conditioning) is not RecognitionConditioning:
        raise ValueError("conditioning must be a RecognitionConditioning")
    if not isinstance(mean, Tensor) or not isinstance(precision_cholesky, Tensor):
        raise ValueError("mean and precision_cholesky must be torch.Tensor values")
    return (
        conditioning,
        FrozenTensorSnapshot.capture(mean),
        FrozenTensorSnapshot.capture(precision_cholesky),
    )


def _precision(cholesky_snapshot: FrozenTensorSnapshot) -> Tensor:
    cholesky = cholesky_snapshot.value()
    result = cholesky @ cholesky.transpose(-1, -2)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("derived precision must be finite")
    return result


def _entropy(cholesky_snapshot: FrozenTensorSnapshot) -> Tensor:
    cholesky = cholesky_snapshot.value()
    dimension = cholesky.shape[0]
    result = (
        0.5 * dimension * (1.0 + math.log(2.0 * math.pi))
        - torch.log(torch.diagonal(cholesky)).sum()
    )
    if result.ndim != 0 or not bool(torch.isfinite(result)):
        raise ValueError("derived recognition entropy must be a finite scalar")
    return result


def _rsample(
    mean_snapshot: FrozenTensorSnapshot,
    cholesky_snapshot: FrozenTensorSnapshot,
    noise: Tensor,
) -> Tensor:
    mean = mean_snapshot.value()
    cholesky = cholesky_snapshot.value()
    if not isinstance(noise, Tensor):
        raise ValueError("noise must be a torch.Tensor")
    if noise.shape != mean.shape:
        raise ValueError("noise must have the recognition mean shape")
    if noise.dtype != mean.dtype or noise.device != mean.device:
        raise ValueError("noise must share recognition dtype and device")
    if not bool(torch.isfinite(noise).all()):
        raise ValueError("noise must be finite")
    displacement = torch.linalg.solve_triangular(
        cholesky.transpose(-1, -2), noise.unsqueeze(-1), upper=True
    ).squeeze(-1)
    result = mean + displacement
    if not bool(torch.isfinite(result).all()):
        raise ValueError("reparameterized sample must be finite")
    return result


@dataclass(frozen=True, eq=False)
class StructuredLanguageRecognition:
    """One normalized full-SPD Gaussian recognition component."""

    conditioning: RecognitionConditioning
    mean: FrozenTensorSnapshot
    precision_cholesky: FrozenTensorSnapshot
    family: Literal["structured_full_spd"] = field(
        default="structured_full_spd", init=False
    )

    def __post_init__(self) -> None:
        _require_gaussian_snapshots(
            self.conditioning, self.mean, self.precision_cholesky
        )

    @classmethod
    def create(
        cls,
        *,
        conditioning: RecognitionConditioning,
        mean: Tensor,
        precision_cholesky: Tensor,
    ) -> "StructuredLanguageRecognition":
        owned = _capture_gaussian(
            conditioning=conditioning,
            mean=mean,
            precision_cholesky=precision_cholesky,
        )
        return cls(*owned)

    def mean_value(self) -> Tensor:
        self.__post_init__()
        return self.mean.value()

    def precision_cholesky_value(self) -> Tensor:
        self.__post_init__()
        return self.precision_cholesky.value()

    def precision(self) -> Tensor:
        self.__post_init__()
        return _precision(self.precision_cholesky)

    def entropy(self) -> Tensor:
        self.__post_init__()
        return _entropy(self.precision_cholesky)

    def rsample(self, noise: Tensor) -> Tensor:
        self.__post_init__()
        return _rsample(self.mean, self.precision_cholesky, noise)


@dataclass(frozen=True, eq=False)
class FactorizedLanguageRecognition:
    """A normalized population-block-factorized Gaussian component."""

    conditioning: RecognitionConditioning
    mean: FrozenTensorSnapshot
    precision_cholesky: FrozenTensorSnapshot
    block_sizes: tuple[int, ...]
    family: Literal["population_factorized_block_spd"] = field(
        default="population_factorized_block_spd", init=False
    )

    def __post_init__(self) -> None:
        live_mean, live_cholesky = _require_gaussian_snapshots(
            self.conditioning, self.mean, self.precision_cholesky
        )
        if (
            type(self.block_sizes) is not tuple
            or not self.block_sizes
            or any(type(size) is not int or size <= 0 for size in self.block_sizes)
            or sum(self.block_sizes) != live_mean.numel()
        ):
            raise ValueError("block_sizes must be positive and cover the Gaussian dimension")
        allowed = torch.zeros_like(live_cholesky, dtype=torch.bool)
        start = 0
        for size in self.block_sizes:
            stop = start + size
            allowed[start:stop, start:stop] = True
            start = stop
        if not bool(torch.all(live_cholesky.masked_select(~allowed) == 0.0)):
            raise ValueError("factorized precision_cholesky must be block diagonal")

    @classmethod
    def create(
        cls,
        *,
        conditioning: RecognitionConditioning,
        mean: Tensor,
        precision_cholesky: Tensor,
        block_sizes: tuple[int, ...],
    ) -> "FactorizedLanguageRecognition":
        owned = _capture_gaussian(
            conditioning=conditioning,
            mean=mean,
            precision_cholesky=precision_cholesky,
        )
        return cls(*owned, tuple(block_sizes))

    def mean_value(self) -> Tensor:
        self.__post_init__()
        return self.mean.value()

    def precision_cholesky_value(self) -> Tensor:
        self.__post_init__()
        return self.precision_cholesky.value()

    def precision(self) -> Tensor:
        self.__post_init__()
        return _precision(self.precision_cholesky)

    def entropy(self) -> Tensor:
        self.__post_init__()
        return _entropy(self.precision_cholesky)

    def rsample(self, noise: Tensor) -> Tensor:
        self.__post_init__()
        return _rsample(self.mean, self.precision_cholesky, noise)


__all__ = [
    "FactorizedLanguageRecognition",
    "RecognitionConditioning",
    "RecognitionMode",
    "StructuredLanguageRecognition",
]

