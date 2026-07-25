"""Typed differentiable recognition families for H6 language training."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor

from vfe4.types.h6 import FrozenTensorSnapshot
from vfe4.types.h7 import (
    H7RawTensorIdentity,
    H7SourceBank,
    H7SourceContextView,
)


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


def _require_h7_float64_tensor(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
    ndim: int | None = None,
) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.dtype is not torch.float64
        or (shape is not None and tuple(value.shape) != shape)
        or (ndim is not None and value.ndim != ndim)
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"{name} is not a finite float64 H7 tensor")
    return value


@dataclass(frozen=True, eq=False)
class H7RecognitionAffineTrace:
    """One complete live source-conditioned receiver factor."""

    component_id: str
    bank: H7SourceBank
    receiver_t: int
    source_j: int
    parent_map: Tensor
    same_receiver_model_map: Tensor | None
    offset: Tensor
    covariance: Tensor
    precision: Tensor

    def __post_init__(self) -> None:
        if type(self.component_id) is not str or not self.component_id:
            raise ValueError("H7 recognition component_id is invalid")
        if (
            self.bank not in ("model", "state")
            or type(self.receiver_t) is not int
            or self.receiver_t not in (1, 2)
            or type(self.source_j) is not int
            or self.source_j < 0
            or self.source_j >= self.receiver_t
        ):
            raise ValueError("H7 recognition receiver/source identity is invalid")
        offset = _require_h7_float64_tensor(
            self.offset, name="offset", ndim=1
        )
        dimension = offset.numel()
        if dimension not in (1, 2):
            raise ValueError("H7 recognition width must be one or two")
        shape = (dimension, dimension)
        parent = _require_h7_float64_tensor(
            self.parent_map, name="parent_map", shape=shape
        )
        covariance = _require_h7_float64_tensor(
            self.covariance, name="covariance", shape=shape
        )
        precision = _require_h7_float64_tensor(
            self.precision, name="precision", shape=shape
        )
        if self.bank == "model":
            if self.same_receiver_model_map is not None:
                raise ValueError("model trace cannot carry a state-model map")
        else:
            _require_h7_float64_tensor(
                self.same_receiver_model_map,
                name="same_receiver_model_map",
                shape=shape,
            )
        eps = torch.finfo(torch.float64).eps
        identity = torch.eye(
            dimension, dtype=torch.float64, device=offset.device
        )
        if (
            parent.device != offset.device
            or covariance.device != offset.device
            or precision.device != offset.device
            or not torch.equal(covariance, covariance.T)
            or not torch.equal(precision, precision.T)
            or not torch.allclose(
                precision @ covariance,
                identity,
                rtol=256.0 * eps,
                atol=256.0 * eps,
            )
        ):
            raise ValueError("H7 recognition Gaussian trace is inconsistent")


@dataclass(frozen=True, eq=False)
class H7RecognitionCompleteTrace:
    """Additional complete-factor trace not retained by the H6 Gaussian API."""

    initial_covariance: Tensor
    model_conditionals: tuple[
        H7RecognitionAffineTrace, H7RecognitionAffineTrace
    ]
    state_conditionals: tuple[
        H7RecognitionAffineTrace, H7RecognitionAffineTrace
    ]
    source_context: H7SourceContextView

    def __post_init__(self) -> None:
        for name, bank, values in (
            ("model_conditionals", "model", self.model_conditionals),
            ("state_conditionals", "state", self.state_conditionals),
        ):
            if (
                type(values) is not tuple
                or len(values) != 2
                or tuple(
                    (item.bank, item.receiver_t, item.source_j)
                    for item in values
                    if type(item) is H7RecognitionAffineTrace
                )
                != ((bank, 1, 0), (bank, 2, 1))
            ):
                raise ValueError(f"H7 {name} inventory is invalid")
            for item in values:
                item.__post_init__()
        dimension = self.model_conditionals[0].offset.numel()
        _require_h7_float64_tensor(
            self.initial_covariance,
            name="initial_covariance",
            shape=(2 * dimension, 2 * dimension),
        )
        if (
            type(self.source_context) is not H7SourceContextView
            or dimension != 2
        ):
            raise ValueError(
                "matrix H7 recognition trace requires a source context"
            )
        self.source_context.assert_live()


@dataclass(frozen=True, eq=False)
class H7LanguageRecognitionTrace:
    """Direct live Gaussian references exported by an exact H6 law."""

    initial_mean: Tensor
    initial_precision_cholesky: Tensor
    complete: H7RecognitionCompleteTrace

    def __post_init__(self) -> None:
        self.complete.__post_init__()
        dimension = self.complete.model_conditionals[0].offset.numel()
        mean = _require_h7_float64_tensor(
            self.initial_mean,
            name="initial_mean",
            shape=(2 * dimension,),
        )
        cholesky = _require_h7_float64_tensor(
            self.initial_precision_cholesky,
            name="initial_precision_cholesky",
            shape=(2 * dimension, 2 * dimension),
        )
        if not torch.equal(cholesky, torch.tril(cholesky)) or not bool(
            torch.all(torch.diagonal(cholesky) > 0.0)
        ):
            raise ValueError("initial precision Cholesky is invalid")
        precision = cholesky @ cholesky.T
        eps = torch.finfo(torch.float64).eps
        identity = torch.eye(
            mean.numel(), dtype=torch.float64, device=mean.device
        )
        if (
            self.complete.initial_covariance.device != mean.device
            or not torch.allclose(
                precision @ self.complete.initial_covariance,
                identity,
                rtol=256.0 * eps,
                atol=256.0 * eps,
            )
        ):
            raise ValueError(
                "initial recognition covariance/precision disagree"
            )

    def initial_precision(self) -> Tensor:
        self.__post_init__()
        return self.initial_precision_cholesky @ self.initial_precision_cholesky.T


@dataclass(frozen=True, eq=False)
class StructuredLanguageRecognition:
    """One normalized full-SPD Gaussian recognition component."""

    conditioning: RecognitionConditioning
    mean: FrozenTensorSnapshot
    precision_cholesky: FrozenTensorSnapshot
    h7_trace: H7RecognitionCompleteTrace | None = field(
        default=None, repr=False, compare=False
    )
    _h7_live_mean: Tensor | None = field(
        default=None, repr=False, compare=False
    )
    _h7_live_precision_cholesky: Tensor | None = field(
        default=None, repr=False, compare=False
    )
    _h7_mean_identity: H7RawTensorIdentity | None = field(
        default=None, repr=False, compare=False
    )
    _h7_cholesky_identity: H7RawTensorIdentity | None = field(
        default=None, repr=False, compare=False
    )
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
        h7_trace: H7RecognitionCompleteTrace | None = None,
    ) -> "StructuredLanguageRecognition":
        owned = _capture_gaussian(
            conditioning=conditioning,
            mean=mean,
            precision_cholesky=precision_cholesky,
        )
        live = (
            (mean, precision_cholesky)
            if h7_trace is not None
            else (None, None)
        )
        identities = (
            (
                H7RawTensorIdentity.capture(mean),
                H7RawTensorIdentity.capture(precision_cholesky),
            )
            if h7_trace is not None
            else (None, None)
        )
        return cls(*owned, h7_trace, *live, *identities)

    def export_h7_trace(self) -> H7LanguageRecognitionTrace:
        if (
            type(self.h7_trace) is not H7RecognitionCompleteTrace
            or not isinstance(self._h7_live_mean, Tensor)
            or not isinstance(self._h7_live_precision_cholesky, Tensor)
            or self._h7_mean_identity
            != H7RawTensorIdentity.capture(self._h7_live_mean)
            or self._h7_cholesky_identity
            != H7RawTensorIdentity.capture(
                self._h7_live_precision_cholesky
            )
        ):
            raise ValueError(
                "StructuredLanguageRecognition has no intact complete H7 trace"
            )
        return H7LanguageRecognitionTrace(
            self._h7_live_mean,
            self._h7_live_precision_cholesky,
            self.h7_trace,
        )

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
    h7_trace: H7RecognitionCompleteTrace | None = field(
        default=None, repr=False, compare=False
    )
    _h7_live_mean: Tensor | None = field(
        default=None, repr=False, compare=False
    )
    _h7_live_precision_cholesky: Tensor | None = field(
        default=None, repr=False, compare=False
    )
    _h7_mean_identity: H7RawTensorIdentity | None = field(
        default=None, repr=False, compare=False
    )
    _h7_cholesky_identity: H7RawTensorIdentity | None = field(
        default=None, repr=False, compare=False
    )
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
        h7_trace: H7RecognitionCompleteTrace | None = None,
    ) -> "FactorizedLanguageRecognition":
        owned = _capture_gaussian(
            conditioning=conditioning,
            mean=mean,
            precision_cholesky=precision_cholesky,
        )
        live = (
            (mean, precision_cholesky)
            if h7_trace is not None
            else (None, None)
        )
        identities = (
            (
                H7RawTensorIdentity.capture(mean),
                H7RawTensorIdentity.capture(precision_cholesky),
            )
            if h7_trace is not None
            else (None, None)
        )
        return cls(
            *owned,
            tuple(block_sizes),
            h7_trace,
            *live,
            *identities,
        )

    def export_h7_trace(self) -> H7LanguageRecognitionTrace:
        if (
            type(self.h7_trace) is not H7RecognitionCompleteTrace
            or not isinstance(self._h7_live_mean, Tensor)
            or not isinstance(self._h7_live_precision_cholesky, Tensor)
            or self._h7_mean_identity
            != H7RawTensorIdentity.capture(self._h7_live_mean)
            or self._h7_cholesky_identity
            != H7RawTensorIdentity.capture(
                self._h7_live_precision_cholesky
            )
        ):
            raise ValueError(
                "FactorizedLanguageRecognition has no intact complete H7 trace"
            )
        return H7LanguageRecognitionTrace(
            self._h7_live_mean,
            self._h7_live_precision_cholesky,
            self.h7_trace,
        )

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
    "H7LanguageRecognitionTrace",
    "H7RecognitionAffineTrace",
    "H7RecognitionCompleteTrace",
    "RecognitionConditioning",
    "RecognitionMode",
    "StructuredLanguageRecognition",
]
