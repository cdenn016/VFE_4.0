"""Direct canonical accumulation for normalized linear-Gaussian factors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor


Scalar = float | Tensor
ParentCoefficients = Sequence[tuple[int, Scalar]]


def add_initial_gaussian(
    h: Tensor,
    J: Tensor,
    indices: tuple[int, int],
    mean: Tensor,
    covariance: Tensor,
) -> None:
    """Add a declared two-dimensional moment Gaussian directly to ``(h, J)``."""

    dimension = _require_accumulators(h, J)
    if type(indices) is not tuple or len(indices) != 2:
        raise ValueError("indices must be a pair")
    if any(type(index) is not int or index < 0 or index >= dimension for index in indices):
        raise ValueError("initial Gaussian index is out of range")
    if len(set(indices)) != 2:
        raise ValueError("initial Gaussian indices must be distinct")
    checked_mean = _require_vector(mean, 2, h.device, "mean")
    checked_covariance = _require_matrix(covariance, 2, h.device, "covariance")
    if not bool(torch.equal(checked_covariance, checked_covariance.transpose(0, 1))):
        raise ValueError("covariance must be symmetric")
    chol, info = torch.linalg.cholesky_ex(checked_covariance, check_errors=False)
    if int(info.item()) != 0 or not bool(torch.isfinite(chol).all()):
        raise ValueError("covariance must be symmetric positive definite")
    identity = torch.eye(2, dtype=torch.float64, device=h.device)
    solved = torch.cholesky_solve(identity, chol)
    precision = torch.empty((2, 2), dtype=torch.float64, device=h.device)
    precision[0, 0] = solved[0, 0]
    precision[1, 1] = solved[1, 1]
    precision[0, 1] = solved[1, 0]
    precision[1, 0] = solved[1, 0]
    natural = precision @ checked_mean
    if not bool(torch.isfinite(precision).all()) or not bool(torch.isfinite(natural).all()):
        raise ValueError("initial Gaussian canonical parameters must be finite")
    for local_row, row in enumerate(indices):
        h[row].add_(natural[local_row])
        for local_column, column in enumerate(indices):
            J[row, column].add_(precision[local_row, local_column])


def add_scalar_conditional(
    h: Tensor,
    J: Tensor,
    target_index: int,
    parent_coefficients: ParentCoefficients,
    offset: Scalar,
    variance: Scalar,
) -> None:
    """Add one normalized affine scalar conditional by an exact outer product.

    Coordinates alternate as ``[z0, m0, z1, m1, ...]``, but their causal
    order within each time step is ``m_t`` before ``z_t``. Every parent must
    therefore precede the target in ``[m0, z0, m1, z1, ...]`` causal order;
    in particular, the numerically later ``m_t`` coordinate may parent
    ``z_t``, while a coordinate from a later time may not.
    """

    dimension = _require_accumulators(h, J)
    if type(target_index) is not int or target_index < 0 or target_index >= dimension:
        raise ValueError("target_index is out of range")
    if not isinstance(parent_coefficients, Sequence) or not parent_coefficients:
        raise ValueError("parent_coefficients must be a nonempty sequence")
    parents: list[int] = []
    checked_coefficients: list[Tensor] = []
    for entry in parent_coefficients:
        if type(entry) is not tuple or len(entry) != 2:
            raise ValueError("each parent coefficient must be an (index, coefficient) pair")
        parent, coefficient = entry
        if type(parent) is not int or parent < 0 or parent >= dimension:
            raise ValueError("parent index is out of range")
        if parent == target_index:
            raise ValueError("parent index cannot equal target_index")
        if parent in parents:
            raise ValueError("parent indices must not be repeated")
        if _causal_rank(parent) >= _causal_rank(target_index):
            raise ValueError("parent indices must causally precede target_index")
        parents.append(parent)
        checked_coefficients.append(_require_scalar(coefficient, h.device, "coefficient"))
    checked_offset = _require_scalar(offset, h.device, "offset")
    checked_variance = _require_scalar(variance, h.device, "variance")
    if not bool(checked_variance > 0):
        raise ValueError("variance must be positive")

    v = torch.zeros(dimension, dtype=torch.float64, device=h.device)
    v[target_index] = 1.0
    for parent, coefficient in zip(parents, checked_coefficients):
        v[parent] = -coefficient
    precision = 1.0 / checked_variance
    J.add_(precision * torch.outer(v, v))
    h.add_(precision * checked_offset * v)
    if not bool(torch.isfinite(h).all()) or not bool(torch.isfinite(J).all()):
        raise ValueError("canonical accumulators must remain finite")


def _require_accumulators(h: object, J: object) -> int:
    if not isinstance(h, Tensor) or not isinstance(J, Tensor):
        raise ValueError("h and J must be torch.Tensor accumulators")
    if h.dtype is not torch.float64 or J.dtype is not torch.float64:
        raise ValueError("h and J must use float64")
    if h.ndim != 1 or h.numel() == 0 or J.ndim != 2 or J.shape != (h.numel(), h.numel()):
        raise ValueError("h and J must have the same dimension")
    if h.device != J.device:
        raise ValueError("h and J must share a device")
    if not bool(torch.isfinite(h).all()) or not bool(torch.isfinite(J).all()):
        raise ValueError("h and J must be finite")
    return h.numel()


def _causal_rank(index: int) -> int:
    time = index // 2
    is_state = index % 2 == 0
    return 2 * time + int(is_state)


def _require_vector(value: object, size: int, device: torch.device, name: str) -> Tensor:
    if not isinstance(value, Tensor) or value.dtype is not torch.float64:
        raise ValueError(f"{name} must be a float64 tensor")
    if value.shape != (size,) or value.device != device or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be a finite vector of shape ({size},) on the accumulator device")
    return value.detach().clone()


def _require_matrix(value: object, size: int, device: torch.device, name: str) -> Tensor:
    if not isinstance(value, Tensor) or value.dtype is not torch.float64:
        raise ValueError(f"{name} must be a float64 tensor")
    if value.shape != (size, size) or value.device != device or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be a finite ({size}, {size}) matrix on the accumulator device")
    return value.detach().clone()


def _require_scalar(value: object, device: torch.device, name: str) -> Tensor:
    if isinstance(value, Tensor):
        if value.dtype is not torch.float64 or value.shape != () or value.device != device:
            raise ValueError(f"{name} must be a float64 scalar on the accumulator device")
        checked = value.detach().clone()
    elif type(value) in (int, float):
        checked = torch.tensor(float(value), dtype=torch.float64, device=device)
    else:
        raise ValueError(f"{name} must be a finite scalar")
    if not bool(torch.isfinite(checked)):
        raise ValueError(f"{name} must be a finite scalar")
    return checked


@dataclass(frozen=True, slots=True)
class RectangularInformationAssembly:
    """Owned model-channel terms induced by one rectangular state factor."""

    schema_version: Literal["rectangular-information-assembly-v1"]
    state_dimension: int
    model_dimension: int
    model_precision_pullback: Tensor
    model_recoil_natural: Tensor

    def __post_init__(self) -> None:
        if self.schema_version != "rectangular-information-assembly-v1":
            raise ValueError("unsupported rectangular information schema")
        if (
            type(self.state_dimension) is not int
            or self.state_dimension < 1
            or type(self.model_dimension) is not int
            or self.model_dimension < 1
        ):
            raise ValueError("rectangular information dimensions must be positive")
        precision = self.model_precision_pullback
        natural = self.model_recoil_natural
        if (
            not isinstance(precision, Tensor)
            or not isinstance(natural, Tensor)
            or precision.dtype is not torch.float64
            or natural.dtype is not torch.float64
            or precision.shape != (self.model_dimension, self.model_dimension)
            or natural.shape != (self.model_dimension,)
            or precision.device != natural.device
            or not bool(torch.isfinite(precision).all())
            or not bool(torch.isfinite(natural).all())
        ):
            raise ValueError("rectangular information tensors have invalid metadata")
        object.__setattr__(
            self,
            "model_precision_pullback",
            precision.detach().clone(),
        )
        object.__setattr__(
            self,
            "model_recoil_natural",
            natural.detach().clone(),
        )


def assemble_rectangular_information(
    *,
    state_precision: Tensor,
    state_model_map: Tensor,
    model_recoil_residual: Tensor,
) -> RectangularInformationAssembly:
    """Assemble ``B.T @ P_z @ B`` and ``B.T @ P_z @ residual``.

    ``state_model_map`` is a morphism from the model channel to the state
    channel and therefore has shape ``(d_z, d_m)``.  The historical scalar
    conditional API remains unchanged.
    """

    if (
        not isinstance(state_precision, Tensor)
        or state_precision.dtype is not torch.float64
        or state_precision.ndim != 2
        or state_precision.shape[0] != state_precision.shape[1]
        or state_precision.shape[0] < 1
        or not bool(torch.isfinite(state_precision).all())
    ):
        raise ValueError("state_precision must be a finite float64 square matrix")
    d_z = state_precision.shape[0]
    if not isinstance(state_model_map, Tensor) or state_model_map.ndim != 2:
        raise ValueError("state_model_map must be a float64 matrix")
    if (
        state_model_map.shape[0] != d_z
        and state_model_map.shape[1] == d_z
    ):
        raise ValueError(
            "state_model_map is transposed; expected (d_z,d_m), not (d_m,d_z)"
        )
    if state_model_map.shape[0] != d_z or state_model_map.shape[1] < 1:
        raise ValueError("state_model_map must have shape (d_z,d_m)")
    d_m = state_model_map.shape[1]
    if (
        state_model_map.dtype is not torch.float64
        or state_model_map.device != state_precision.device
        or not bool(torch.isfinite(state_model_map).all())
    ):
        raise ValueError(
            "state_model_map must be finite float64 on the precision device"
        )
    if (
        not isinstance(model_recoil_residual, Tensor)
        or model_recoil_residual.dtype is not torch.float64
        or model_recoil_residual.shape != (d_z,)
        or model_recoil_residual.device != state_precision.device
        or not bool(torch.isfinite(model_recoil_residual).all())
    ):
        raise ValueError(
            "model_recoil_residual must be a finite float64 state vector"
        )
    if not bool(
        torch.allclose(
            state_precision,
            state_precision.transpose(0, 1),
            rtol=0.0,
            atol=1.0e-14,
        )
    ):
        raise ValueError("state_precision must be symmetric")
    symmetric_precision = 0.5 * (
        state_precision + state_precision.transpose(0, 1)
    )
    _, info = torch.linalg.cholesky_ex(symmetric_precision, check_errors=False)
    if int(info.item()) != 0:
        raise ValueError("state_precision must be positive definite")

    checked_precision = state_precision.detach().clone()
    checked_map = state_model_map.detach().clone()
    checked_residual = model_recoil_residual.detach().clone()
    return RectangularInformationAssembly(
        schema_version="rectangular-information-assembly-v1",
        state_dimension=d_z,
        model_dimension=d_m,
        model_precision_pullback=(
            checked_map.transpose(0, 1) @ checked_precision @ checked_map
        ),
        model_recoil_natural=(
            checked_map.transpose(0, 1)
            @ checked_precision
            @ checked_residual
        ),
    )
