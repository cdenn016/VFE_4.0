from __future__ import annotations

import pytest
import torch

from vfe4.geometry.group_action import (
    block_population_action,
    borrow_h7_action,
    centered_logit_projector,
    compose_reframed_frames,
    frame_links,
    freeze_h7_action,
    logabsdet_measure_shift,
    push_covariance,
    push_decoder,
    push_information_vector,
    push_precision,
    push_receiver_source_map,
    push_same_receiver_morphism,
    push_second_moment,
    push_vector,
    require_direct_gl_plus,
    right_solve,
)
from vfe4.types.h7 import (
    H7BorrowedActionView,
    H7BorrowedTensorView,
    H7GLPlus2Action,
    H7ScalarReplayAction,
)


def _matrix_action() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([[1.25, 0.20], [0.10, 0.90]], dtype=torch.float64),
        torch.tensor([[0.85, -0.15], [0.25, 1.30]], dtype=torch.float64),
        torch.tensor([[1.10, 0.30], [-0.10, 0.95]], dtype=torch.float64),
    )


def test_h7_direct_actions_borrow_live_tensors_and_freeze_owned_evidence() -> None:
    scalar_values = tuple(
        torch.tensor([[value]], dtype=torch.float64, requires_grad=True)
        for value in (1.25, 1.25, 1.25)
    )
    scalar = borrow_h7_action(
        scalar_values, kind="diagonal_base", dimension=1
    )
    matrix_values = _matrix_action()
    matrix = borrow_h7_action(
        matrix_values, kind="internal_product", dimension=2
    )

    assert all(
        view.tensor is original
        for view, original in zip(scalar.elements, scalar_values, strict=True)
    )
    assert (
        require_direct_gl_plus(scalar_values[0], dimension=1)
        is scalar_values[0]
    )
    scalar_owned = freeze_h7_action(scalar)
    matrix_owned = freeze_h7_action(matrix)
    assert type(scalar_owned) is H7ScalarReplayAction
    assert type(matrix_owned) is H7GLPlus2Action
    assert scalar_owned.group == "GL+(1,R)" and scalar_owned.dimension == 1
    assert matrix_owned.group == "GL+(2,R)" and matrix_owned.dimension == 2
    assert len(scalar_owned.action_sha256) == 64
    assert scalar_owned.action_sha256 != matrix_owned.action_sha256
    capture = scalar_owned.elements[0].capture_identity
    assert capture.object_id == id(scalar_values[0])
    assert capture.storage_data_ptr == scalar_values[0].untyped_storage().data_ptr()
    assert capture.storage_version == scalar_values[0]._version
    assert capture.dtype == "float64"
    assert capture.shape == (1, 1)
    assert capture.device == str(scalar_values[0].device)
    assert capture.contiguous and capture.requires_grad
    assert len(scalar_owned.elements[0].raw_bytes_sha256) == 64
    frozen_gradient = torch.autograd.grad(
        scalar_owned.elements[0].value().sum(), scalar_values[0]
    )[0]
    assert torch.equal(frozen_gradient, torch.ones_like(scalar_values[0]))

    scalar_x = torch.tensor([0.4], dtype=torch.float64)
    scalar_sigma = torch.tensor([[0.7]], dtype=torch.float64)
    scalar_g = scalar_values[0]
    assert torch.allclose(push_vector(scalar_x, scalar_g), scalar_g @ scalar_x)
    assert torch.allclose(
        push_covariance(scalar_sigma, scalar_g), scalar_g.square() * scalar_sigma
    )
    assert torch.allclose(
        push_precision(scalar_sigma, scalar_g), scalar_sigma / scalar_g.square()
    )
    assert torch.allclose(
        push_information_vector(scalar_x, scalar_g), scalar_x / scalar_g.diag()
    )
    assert torch.allclose(
        push_second_moment(scalar_sigma, scalar_g),
        scalar_g.square() * scalar_sigma,
    )
    scalar_source = torch.tensor([[0.8]], dtype=torch.float64)
    assert torch.allclose(
        push_receiver_source_map(scalar_sigma, scalar_g, scalar_source),
        scalar_sigma * scalar_g / scalar_source,
    )
    assert torch.allclose(
        push_same_receiver_morphism(
            scalar_sigma, scalar_g, scalar_source
        ),
        scalar_sigma * scalar_g / scalar_source,
    )
    assert torch.allclose(
        push_decoder(scalar_sigma, scalar_g), scalar_sigma / scalar_g
    )

    frozen_value = scalar_owned.elements[0].value()
    frozen_action_sha256 = scalar_owned.action_sha256
    with torch.no_grad():
        scalar_values[0].add_(1.0)
    assert torch.equal(scalar_owned.elements[0].value(), frozen_value)
    assert scalar_owned.action_sha256 == frozen_action_sha256
    with pytest.raises(ValueError, match="stale"):
        block_population_action(scalar)


def test_h7_tensor_laws_match_direct_change_of_coordinates_and_autograd() -> None:
    g = _matrix_action()[0].requires_grad_()
    x = torch.tensor([0.4, -0.7], dtype=torch.float64, requires_grad=True)
    sigma = torch.tensor(
        [[1.4, 0.2], [0.2, 0.9]], dtype=torch.float64, requires_grad=True
    )
    precision = torch.linalg.solve(sigma, torch.eye(2, dtype=torch.float64))
    h = precision @ x
    second = sigma + torch.outer(x, x)
    decoder = torch.tensor(
        [[0.2, -0.3], [0.5, 0.1], [-0.4, 0.2]], dtype=torch.float64
    )

    pushed_x = push_vector(x, g)
    pushed_sigma = push_covariance(sigma, g)
    pushed_precision = push_precision(precision, g)
    pushed_h = push_information_vector(h, g)
    pushed_second = push_second_moment(second, g)
    pushed_decoder = push_decoder(decoder, g)

    for result in (
        pushed_x,
        pushed_sigma,
        pushed_precision,
        pushed_h,
        pushed_second,
        pushed_decoder,
    ):
        assert result.grad_fn is not None
    assert torch.allclose(pushed_x, g @ x)
    assert torch.allclose(pushed_sigma, g @ sigma @ g.T)
    assert torch.allclose(pushed_precision, torch.linalg.inv(g).T @ precision @ torch.linalg.inv(g))
    assert torch.allclose(pushed_h, torch.linalg.inv(g).T @ h)
    assert torch.allclose(pushed_second, g @ second @ g.T)
    assert torch.allclose(pushed_decoder, decoder @ torch.linalg.inv(g))
    torch.stack(
        (
            pushed_x.sum(),
            pushed_sigma.sum(),
            pushed_precision.sum(),
            pushed_h.sum(),
            pushed_second.sum(),
            pushed_decoder.sum(),
        )
    ).sum().backward()
    assert g.grad is not None and x.grad is not None and sigma.grad is not None


def test_h7_typed_maps_frames_and_population_relations_use_receiver_source_order() -> None:
    state_receiver = torch.tensor(
        [[1.2, 0.1], [0.2, 0.9]], dtype=torch.float64
    )
    model_receiver = torch.tensor(
        [[0.8, -0.2, 0.1], [0.1, 1.1, 0.0], [0.2, 0.1, 1.3]],
        dtype=torch.float64,
    )
    b = torch.tensor(
        [[0.3, -0.1, 0.2], [0.4, 0.5, -0.2]], dtype=torch.float64
    )
    expected_b = state_receiver @ b @ torch.linalg.inv(model_receiver)
    assert torch.allclose(
        push_same_receiver_morphism(b, state_receiver, model_receiver),
        expected_b,
    )
    assert torch.allclose(
        push_receiver_source_map(b, state_receiver, model_receiver),
        expected_b,
    )
    with pytest.raises(ValueError, match="dimensions do not align"):
        push_same_receiver_morphism(b.T, state_receiver, model_receiver)
    with pytest.raises(ValueError, match="dimensions do not align"):
        push_same_receiver_morphism(b, model_receiver, state_receiver)
    equal_b = b[:, :2]
    equal_model_receiver = model_receiver[:2, :2]
    correct_equal = push_same_receiver_morphism(
        equal_b, state_receiver, equal_model_receiver
    )
    reverse_arrow_mutant = (
        torch.linalg.inv(equal_model_receiver) @ equal_b @ state_receiver
    )
    assert not torch.allclose(correct_equal, reverse_arrow_mutant)

    action_values = _matrix_action()
    action = borrow_h7_action(
        action_values, kind="internal_product", dimension=2
    )
    frames = (
        torch.eye(2, dtype=torch.float64),
        torch.tensor([[1.0, 0.2], [0.0, 1.1]], dtype=torch.float64),
        torch.tensor([[0.9, -0.1], [0.3, 1.2]], dtype=torch.float64),
    )
    transformed = compose_reframed_frames(action, frames)
    for transformed_frame, element, frame in zip(
        transformed, action_values, frames, strict=True
    ):
        assert torch.allclose(transformed_frame, element @ frame)
    original_links = frame_links(frames)
    links = frame_links(transformed)
    identity = torch.eye(2, dtype=torch.float64)
    assert len(links) == 6
    assert torch.allclose(links[(2, 1)] @ links[(1, 0)], links[(2, 0)])
    assert torch.allclose(
        links[(0, 2)] @ links[(2, 1)] @ links[(1, 0)], identity
    )
    for key, original_link in original_links.items():
        receiver, source = key
        assert torch.allclose(
            links[key],
            push_receiver_source_map(
                original_link,
                action_values[receiver],
                action_values[source],
            ),
        )
    expected_block = torch.block_diag(
        action_values[0],
        action_values[0],
        action_values[1],
        action_values[1],
        action_values[2],
        action_values[2],
    )
    assert torch.equal(block_population_action(action), expected_block)
    expected_logdet = 2.0 * sum(
        torch.linalg.slogdet(value)[1] for value in action_values
    )
    assert torch.allclose(logabsdet_measure_shift(action), expected_logdet)


def test_h7_rejects_outside_domain_and_centered_projector_is_exactly_centered() -> None:
    negative = torch.tensor([[-1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    with pytest.raises(ValueError, match="positive determinant"):
        require_direct_gl_plus(negative, dimension=2)
    with pytest.raises(ValueError, match="torch.float64"):
        require_direct_gl_plus(torch.eye(2, dtype=torch.float32), dimension=2)
    with pytest.raises(ValueError, match="identical element"):
        borrow_h7_action(
            tuple(
                torch.tensor([[value]], dtype=torch.float64)
                for value in (1.0, 1.1, 1.0)
            ),
            kind="diagonal_base",
            dimension=1,
        )
    forged = H7BorrowedActionView(
        elements=tuple(
            H7BorrowedTensorView.borrow(
                torch.tensor([[value]], dtype=torch.float64)
            )
            for value in (1.0, 1.1, 1.0)
        ),
        kind="diagonal_base",
        dimension=1,
        group="GL+(1,R)",
    )
    with pytest.raises(ValueError, match="identical element"):
        block_population_action(forged)
    projector = centered_logit_projector(
        3, like=torch.zeros((), dtype=torch.float64)
    )
    ones = torch.ones(3, dtype=torch.float64)
    assert torch.allclose(projector @ ones, torch.zeros_like(ones))
    assert torch.allclose(projector @ projector, projector)


def test_h7_each_direct_law_retains_finite_nonzero_operand_gradients() -> None:
    def checked(output: torch.Tensor, *inputs: torch.Tensor) -> None:
        assert output.grad_fn is not None
        gradients = torch.autograd.grad(output.sum(), inputs)
        assert all(
            gradient is not None
            and bool(torch.isfinite(gradient).all().item())
            and bool((gradient != 0).any().item())
            for gradient in gradients
        )

    def matrix(value: list[list[float]]) -> torch.Tensor:
        return torch.tensor(value, dtype=torch.float64, requires_grad=True)

    g = matrix([[1.2, 0.1], [0.2, 0.9]])
    x = torch.tensor([0.3, -0.4], dtype=torch.float64, requires_grad=True)
    checked(push_vector(x, g), x, g)
    g = matrix([[1.2, 0.1], [0.2, 0.9]])
    sigma = matrix([[1.1, 0.2], [0.2, 0.8]])
    checked(push_covariance(sigma, g), sigma, g)
    g = matrix([[1.2, 0.1], [0.2, 0.9]])
    second = matrix([[1.4, -0.1], [-0.1, 1.2]])
    checked(push_second_moment(second, g), second, g)
    g = matrix([[1.2, 0.1], [0.2, 0.9]])
    precision = matrix([[1.3, -0.1], [-0.1, 0.9]])
    checked(push_precision(precision, g), precision, g)
    g = matrix([[1.2, 0.1], [0.2, 0.9]])
    h = torch.tensor([0.2, 0.5], dtype=torch.float64, requires_grad=True)
    checked(push_information_vector(h, g), h, g)
    receiver = matrix([[1.2, 0.1], [0.2, 0.9]])
    source = matrix([[0.9, -0.2], [0.1, 1.1]])
    value = matrix([[0.3, 0.2], [-0.1, 0.4]])
    checked(
        push_receiver_source_map(value, receiver, source),
        value,
        receiver,
        source,
    )
    receiver = matrix([[1.2, 0.1], [0.2, 0.9]])
    source = matrix([[0.9, -0.2], [0.1, 1.1]])
    value = matrix([[0.3, 0.2], [-0.1, 0.4]])
    checked(
        push_same_receiver_morphism(value, receiver, source),
        value,
        receiver,
        source,
    )
    receiver = matrix([[1.2, 0.1], [0.2, 0.9]])
    decoder = matrix([[0.3, 0.2], [-0.1, 0.4], [0.5, -0.2]])
    checked(push_decoder(decoder, receiver), decoder, receiver)
    right = matrix([[0.9, -0.2], [0.1, 1.1]])
    value = matrix([[0.3, 0.2], [-0.1, 0.4]])
    checked(right_solve(value, right), value, right)

    action_values = tuple(
        matrix(values)
        for values in (
            [[1.2, 0.1], [0.2, 0.9]],
            [[0.9, -0.2], [0.1, 1.1]],
            [[1.1, 0.2], [-0.1, 0.95]],
        )
    )
    action = borrow_h7_action(
        action_values, kind="internal_product", dimension=2
    )
    checked(block_population_action(action), *action_values)
    action_values = tuple(
        matrix(values)
        for values in (
            [[1.2, 0.1], [0.2, 0.9]],
            [[0.9, -0.2], [0.1, 1.1]],
            [[1.1, 0.2], [-0.1, 0.95]],
        )
    )
    action = borrow_h7_action(
        action_values, kind="internal_product", dimension=2
    )
    frames = tuple(
        matrix(values)
        for values in (
            [[1.0, 0.1], [0.2, 1.1]],
            [[0.9, -0.1], [0.15, 1.05]],
            [[1.1, 0.2], [-0.05, 0.95]],
        )
    )
    reframed = compose_reframed_frames(action, frames)
    checked(torch.stack(tuple(item.sum() for item in reframed)), *action_values, *frames)
    frames = tuple(
        matrix(values)
        for values in (
            [[1.0, 0.1], [0.2, 1.1]],
            [[0.9, -0.1], [0.15, 1.05]],
            [[1.1, 0.2], [-0.05, 0.95]],
        )
    )
    links = frame_links(frames)
    checked(torch.stack(tuple(item.sum() for item in links.values())), *frames)
    action_values = tuple(
        matrix(values)
        for values in (
            [[1.2, 0.1], [0.2, 0.9]],
            [[0.9, -0.2], [0.1, 1.1]],
            [[1.1, 0.2], [-0.1, 0.95]],
        )
    )
    action = borrow_h7_action(
        action_values, kind="internal_product", dimension=2
    )
    checked(logabsdet_measure_shift(action), *action_values)


def test_h7_production_actions_do_not_materialize_inverse_or_exponential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("forbidden inverse/exponential path")

    original_solve = torch.linalg.solve

    def guarded_solve(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if right.ndim == 2 and right.shape[0] == right.shape[1]:
            identity = torch.eye(
                right.shape[0], dtype=right.dtype, device=right.device
            )
            if torch.equal(right, identity):
                raise AssertionError("solve received a materialized identity RHS")
        return original_solve(left, right)

    monkeypatch.setattr(torch.linalg, "inv", forbidden)
    monkeypatch.setattr(torch.linalg, "pinv", forbidden)
    monkeypatch.setattr(torch, "matrix_exp", forbidden)
    monkeypatch.setattr(torch.linalg, "solve", guarded_solve)

    receiver = torch.tensor(
        [[1.2, 0.1], [0.2, 0.9]], dtype=torch.float64
    )
    source = torch.tensor(
        [[0.9, -0.2], [0.1, 1.1]], dtype=torch.float64
    )
    value = torch.tensor(
        [[0.3, 0.2], [-0.1, 0.4]], dtype=torch.float64
    )
    push_precision(value, receiver)
    right_solve(value, source)
    push_receiver_source_map(value, receiver, source)
    push_decoder(value, receiver)
    frames = (receiver, source, receiver @ source)
    frame_links(frames)
