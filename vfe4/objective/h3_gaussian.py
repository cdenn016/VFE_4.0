"""Direct differentiable normalized-factor ELBO for H3."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from vfe4.generative.reference_h3 import H3GenerativeModel
from vfe4.recognition.reference_h3 import H3VariationalGaussian


_FACTOR_COUNT = 6
_DIMENSION = 4


def _live_scalar(value: object, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if value.dtype is not torch.float64:
        raise ValueError(f"{name} must use float64")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU")
    if value.shape != ():
        raise ValueError(f"{name} must be scalar")
    if not bool(torch.isfinite(value)):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True, eq=False)
class H3ObjectiveEvaluation:
    """Live factor expectations, entropy, and their direct ELBO sum."""

    expected_log_factors: tuple[Tensor, ...]
    entropy: Tensor
    elbo: Tensor

    def __post_init__(self) -> None:
        if (
            type(self.expected_log_factors) is not tuple
            or len(self.expected_log_factors) != _FACTOR_COUNT
        ):
            raise ValueError("expected_log_factors must contain six tensors")
        checked_factors = tuple(
            _live_scalar(value, f"expected_log_factors[{index}]")
            for index, value in enumerate(self.expected_log_factors)
        )
        entropy = _live_scalar(self.entropy, "entropy")
        elbo = _live_scalar(self.elbo, "elbo")
        if not torch.equal(elbo, sum(checked_factors) + entropy):
            raise ValueError("elbo must equal the six factor terms plus entropy")
        object.__setattr__(self, "expected_log_factors", checked_factors)
        object.__setattr__(self, "entropy", entropy)
        object.__setattr__(self, "elbo", elbo)


def evaluate_h3_elbo(
    model: H3GenerativeModel,
    q: H3VariationalGaussian,
) -> H3ObjectiveEvaluation:
    """Evaluate ``E_q[log p(y,x)] + H(q)`` from six public factors."""

    if not isinstance(model, H3GenerativeModel):
        raise ValueError("model must be an H3GenerativeModel")
    if not isinstance(q, H3VariationalGaussian):
        raise ValueError("variational law must be an H3VariationalGaussian")
    if q.mean.shape != (_DIMENSION,):
        raise ValueError("model and variational dimensions must match")
    if len(model.factors) != _FACTOR_COUNT:
        raise ValueError("model must contain six factors")

    expected_log_factors: list[Tensor] = []
    for factor in model.factors:
        row = factor.row
        if row.shape != q.mean.shape:
            raise ValueError("model and variational dimensions must match")
        target = factor.target
        variance = factor.variance
        mean_residual = row @ q.mean - target
        expected_square = mean_residual.square() + q.linear_variance(row)
        expected_log_factors.append(
            -0.5
            * (
                expected_square / variance
                + torch.log(2.0 * torch.pi * variance)
            )
        )

    frozen_factors = tuple(expected_log_factors)
    entropy = q.entropy()
    elbo = sum(frozen_factors) + entropy
    return H3ObjectiveEvaluation(frozen_factors, entropy, elbo)


__all__ = ["H3ObjectiveEvaluation", "evaluate_h3_elbo"]
