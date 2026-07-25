"""Pure, graph-preserving direct ``GL+(d, R)`` tensor actions for H7.

This module operates on caller-owned tensors until the explicit
``freeze_h7_action`` evidence boundary.  In particular, it never materializes
an inverse and never detaches a tensor participating in the calculation.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Literal, cast

import torch

from vfe4.types.h7 import (
    H7ActionKind,
    H7BorrowedActionView,
    H7BorrowedTensorView,
    H7GLPlus2Action,
    H7OwnedTensorSnapshot,
    H7RawTensorIdentity,
    H7ScalarReplayAction,
    H7TensorActionSnapshot,
)


def _tensor_identity(value: torch.Tensor) -> H7RawTensorIdentity:
    return H7RawTensorIdentity.capture(value)


def _assert_live(view: H7BorrowedTensorView) -> torch.Tensor:
    if type(view) is not H7BorrowedTensorView:
        raise ValueError("borrowed tensor must be an exact H7BorrowedTensorView")
    value = view.tensor
    if not isinstance(value, torch.Tensor):
        raise ValueError("borrowed H7 tensor identity is stale")
    view.assert_intact()
    return value


def _action_elements(action: H7BorrowedActionView) -> tuple[torch.Tensor, ...]:
    if type(action) is not H7BorrowedActionView:
        raise ValueError("action must be an exact H7BorrowedActionView")
    if len(action.elements) != 3:
        raise ValueError("an H7 action must contain exactly three population elements")
    expected_group = "GL+(1,R)" if action.dimension == 1 else "GL+(2,R)"
    if action.dimension not in (1, 2) or action.group != expected_group:
        raise ValueError("H7 action dimension and group disagree")
    values = tuple(_assert_live(element) for element in action.elements)
    for value in values:
        require_direct_gl_plus(value, dimension=action.dimension)
    if action.kind == "diagonal_base" and not all(
        torch.equal(values[0], value) for value in values[1:]
    ):
        raise ValueError(
            "diagonal_base requires one identical element at all populations"
        )
    return values


def require_direct_gl_plus(element: torch.Tensor, *, dimension: int) -> torch.Tensor:
    """Validate one real direct-group element and return the caller object."""

    if not isinstance(element, torch.Tensor):
        raise ValueError("group element must be a torch.Tensor")
    if type(dimension) is not int or dimension not in (1, 2):
        raise ValueError("dimension must be exactly 1 or 2")
    if element.ndim != 2 or tuple(element.shape) != (dimension, dimension):
        raise ValueError("group element has the wrong matrix shape")
    if element.dtype is not torch.float64 or element.is_complex():
        raise ValueError("group element must use real torch.float64")
    if not bool(torch.isfinite(element).all().item()):
        raise ValueError("group element must be finite")
    sign, logabsdet = torch.linalg.slogdet(element)
    if not bool(torch.isfinite(logabsdet).item()) or not bool((sign > 0).item()):
        raise ValueError("group element must have positive determinant")
    return element


def borrow_h7_action(
    elements: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    kind: H7ActionKind,
    dimension: Literal[1, 2],
) -> H7BorrowedActionView:
    """Borrow exactly three live direct-group tensors without copying them."""

    if type(elements) is not tuple or len(elements) != 3:
        raise ValueError("elements must be an exact three-tensor tuple")
    if kind not in ("diagonal_base", "internal_product"):
        raise ValueError("unsupported H7 action kind")
    checked = tuple(
        require_direct_gl_plus(value, dimension=dimension) for value in elements
    )
    if kind == "diagonal_base" and not all(
        torch.equal(checked[0], value) for value in checked[1:]
    ):
        raise ValueError(
            "diagonal_base requires one identical element at all populations"
        )
    views = tuple(
        H7BorrowedTensorView(value, _tensor_identity(value)) for value in checked
    )
    group = "GL+(1,R)" if dimension == 1 else "GL+(2,R)"
    return H7BorrowedActionView(
        elements=cast(
            tuple[H7BorrowedTensorView, H7BorrowedTensorView, H7BorrowedTensorView],
            views,
        ),
        kind=kind,
        dimension=dimension,
        group=group,
    )


def freeze_h7_action(action: H7BorrowedActionView) -> H7TensorActionSnapshot:
    """Cross the sole borrowed-to-owned action evidence boundary."""

    values = _action_elements(action)
    snapshots = tuple(
        # The H7 type owns the canonical bytes and integrity digest.
        H7OwnedTensorSnapshot.capture(value)
        for value in values
    )
    owned = cast(
        tuple[
            H7OwnedTensorSnapshot,
            H7OwnedTensorSnapshot,
            H7OwnedTensorSnapshot,
        ],
        snapshots,
    )
    if action.dimension == 1:
        return H7ScalarReplayAction.create(elements=owned, kind=action.kind)
    return H7GLPlus2Action.create(elements=owned, kind=action.kind)


def block_population_action(action: H7BorrowedActionView) -> torch.Tensor:
    """Return the global ``[z0,m0,z1,m1,z2,m2]`` block action."""

    elements = _action_elements(action)
    return torch.block_diag(
        elements[0],
        elements[0],
        elements[1],
        elements[1],
        elements[2],
        elements[2],
    )


def right_solve(value: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Compute ``value @ right^{-1}`` by solving on the actual operand."""

    if not isinstance(value, torch.Tensor) or not isinstance(right, torch.Tensor):
        raise ValueError("right_solve operands must be tensors")
    if value.ndim != 2 or right.ndim != 2 or right.shape[0] != right.shape[1]:
        raise ValueError("right_solve requires a matrix and a square right operand")
    if value.shape[1] != right.shape[0]:
        raise ValueError("right_solve operand dimensions do not align")
    if value.dtype != right.dtype or value.device != right.device:
        raise ValueError("right_solve operands must share dtype and device")
    return torch.linalg.solve(right.T, value.T).T


def push_vector(value: torch.Tensor, receiver: torch.Tensor) -> torch.Tensor:
    if value.ndim != 1 or receiver.ndim != 2 or receiver.shape[1] != value.shape[0]:
        raise ValueError("vector and receiver dimensions do not align")
    return receiver @ value


def push_covariance(value: torch.Tensor, receiver: torch.Tensor) -> torch.Tensor:
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("covariance must be square")
    if receiver.ndim != 2 or tuple(receiver.shape) != tuple(value.shape):
        raise ValueError("covariance and receiver dimensions do not align")
    return receiver @ value @ receiver.T


def push_precision(value: torch.Tensor, receiver: torch.Tensor) -> torch.Tensor:
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("precision must be square")
    if receiver.ndim != 2 or tuple(receiver.shape) != tuple(value.shape):
        raise ValueError("precision and receiver dimensions do not align")
    right = right_solve(value, receiver)
    return torch.linalg.solve(receiver.T, right)


def push_information_vector(
    value: torch.Tensor, receiver: torch.Tensor
) -> torch.Tensor:
    if value.ndim != 1 or receiver.ndim != 2 or receiver.shape[0] != receiver.shape[1]:
        raise ValueError("information action requires a vector and square receiver")
    if receiver.shape[0] != value.shape[0]:
        raise ValueError("information vector and receiver dimensions do not align")
    return torch.linalg.solve(receiver.T, value)


def push_second_moment(value: torch.Tensor, receiver: torch.Tensor) -> torch.Tensor:
    return push_covariance(value, receiver)


def push_receiver_source_map(
    value: torch.Tensor,
    receiver: torch.Tensor,
    source: torch.Tensor,
) -> torch.Tensor:
    if value.ndim != 2 or receiver.ndim != 2 or source.ndim != 2:
        raise ValueError("map action requires matrix operands")
    if receiver.shape[0] != receiver.shape[1] or source.shape[0] != source.shape[1]:
        raise ValueError("receiver and source actions must be square")
    if value.shape != (receiver.shape[0], source.shape[0]):
        raise ValueError("map, receiver, and source dimensions do not align")
    return receiver @ right_solve(value, source)


def push_same_receiver_morphism(
    value: torch.Tensor,
    state_receiver: torch.Tensor,
    model_receiver: torch.Tensor,
) -> torch.Tensor:
    """Apply the typed state-from-model law ``G_z B G_m^{-1}``."""

    return push_receiver_source_map(value, state_receiver, model_receiver)


def push_decoder(value: torch.Tensor, receiver: torch.Tensor) -> torch.Tensor:
    if value.ndim != 2:
        raise ValueError("decoder must be a matrix")
    return right_solve(value, receiver)


def compose_reframed_frames(
    action: H7BorrowedActionView,
    frames: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    elements = _action_elements(action)
    if type(frames) is not tuple or len(frames) != len(elements):
        raise ValueError("frames must follow the exact three-population order")
    transformed: list[torch.Tensor] = []
    for element, frame in zip(elements, frames, strict=True):
        if not isinstance(frame, torch.Tensor) or tuple(frame.shape) != tuple(
            element.shape
        ):
            raise ValueError("frame and action dimensions do not align")
        transformed.append(element @ frame)
    return tuple(transformed)


def frame_links(
    frames: tuple[torch.Tensor, ...],
) -> MappingProxyType[tuple[int, int], torch.Tensor]:
    if type(frames) is not tuple or not frames:
        raise ValueError("frames must be a nonempty tuple")
    first = frames[0]
    if (
        not isinstance(first, torch.Tensor)
        or first.ndim != 2
        or first.shape[0] != first.shape[1]
    ):
        raise ValueError("frames must be square tensors")
    for frame in frames:
        if not isinstance(frame, torch.Tensor) or tuple(frame.shape) != tuple(
            first.shape
        ):
            raise ValueError("all frames must have the same square shape")
    links = {
        (receiver, source): right_solve(frames[receiver], frames[source])
        for receiver in range(len(frames))
        for source in range(len(frames))
        if receiver != source
    }
    return MappingProxyType(links)


def logabsdet_measure_shift(action: H7BorrowedActionView) -> torch.Tensor:
    terms = tuple(torch.linalg.slogdet(value)[1] for value in _action_elements(action))
    # Each population element acts once on z and once on m.
    return 2.0 * torch.stack(terms).sum()


def centered_logit_projector(
    vocabulary_size: int, *, like: torch.Tensor
) -> torch.Tensor:
    if type(vocabulary_size) is not int or vocabulary_size < 2:
        raise ValueError("vocabulary_size must be an integer at least two")
    if not isinstance(like, torch.Tensor) or not like.dtype.is_floating_point:
        raise ValueError("like must be a real floating tensor")
    identity = torch.eye(vocabulary_size, dtype=like.dtype, device=like.device)
    mean = torch.full_like(identity, 1.0 / vocabulary_size)
    return identity - mean


__all__ = [
    "block_population_action",
    "borrow_h7_action",
    "centered_logit_projector",
    "compose_reframed_frames",
    "frame_links",
    "freeze_h7_action",
    "logabsdet_measure_shift",
    "push_covariance",
    "push_decoder",
    "push_information_vector",
    "push_precision",
    "push_receiver_source_map",
    "push_same_receiver_morphism",
    "push_second_moment",
    "push_vector",
    "require_direct_gl_plus",
    "right_solve",
]
