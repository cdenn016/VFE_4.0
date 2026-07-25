from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch

from vfe4.geometry.group_action import (
    borrow_h7_action,
    logabsdet_measure_shift,
)
from vfe4.recognition.language import (
    FactorizedLanguageRecognition,
    H7RecognitionAffineTrace,
    H7RecognitionCompleteTrace,
    RecognitionConditioning,
    StructuredLanguageRecognition,
)
from vfe4.recognition.pushforward import (
    borrow_h7_recognition,
    freeze_h7_recognition,
    pushforward_h7_recognition,
)
from vfe4.types.h7 import (
    H7BorrowedTensorView,
    H7HistoryValueView,
    H7JacobianMetadataSnapshot,
    H7RecognitionContextSnapshot,
    H7RecognitionSnapshot,
    H7SourceContextView,
    H7SourceScorerRowView,
)


def _borrow(value: torch.Tensor) -> H7BorrowedTensorView:
    return H7BorrowedTensorView.borrow(value)


def _source_context() -> H7SourceContextView:
    prefix_tokens = (0, 2)
    z_values = (
        torch.tensor([0.2, -0.1], dtype=torch.float64),
        torch.tensor([0.3, 0.4], dtype=torch.float64),
    )
    m_values = (
        torch.tensor([-0.2, 0.5], dtype=torch.float64),
        torch.tensor([0.1, -0.3], dtype=torch.float64),
    )
    z_history = tuple(
        H7HistoryValueView("z", index, _borrow(value))
        for index, value in enumerate(z_values)
    )
    m_history = tuple(
        H7HistoryValueView("m", index, _borrow(value))
        for index, value in enumerate(m_values)
    )
    rows: list[H7SourceScorerRowView] = []
    for bank in ("model", "state"):
        for receiver_t in (1, 2):
            source_j = receiver_t - 1
            row_prefix = prefix_tokens[:receiver_t]
            row_prefix_bytes = json.dumps(row_prefix, separators=(",", ":")).encode(
                "ascii"
            )
            bias = 0.05 * receiver_t
            scale = 0.01 if bank == "model" else -0.02
            weighted = sum(
                (index + 1) * (token + 1) for index, token in enumerate(row_prefix)
            )
            prefix_term = bias + scale * weighted
            z_covector = torch.tensor([0.1 * receiver_t, -0.05], dtype=torch.float64)
            m_covector = torch.tensor([-0.04, 0.08 * receiver_t], dtype=torch.float64)
            raw_score = (
                torch.as_tensor(prefix_term, dtype=torch.float64)
                + z_covector @ z_values[source_j]
                + m_covector @ m_values[source_j]
            ).reshape(1)
            rows.append(
                H7SourceScorerRowView.create(
                    bank=bank,
                    receiver_t=receiver_t,
                    source_j=source_j,
                    prefix_tokens=row_prefix,
                    prefix_bytes=row_prefix_bytes,
                    alpha_bias=bias,
                    alpha_token_scale=scale,
                    prefix_term=prefix_term,
                    z_history=z_history,
                    m_history=m_history,
                    z_covector=_borrow(z_covector),
                    m_covector=_borrow(m_covector),
                    mask=(True,),
                    support=(source_j,),
                    raw_scores=_borrow(raw_score),
                    probabilities=_borrow(torch.ones(1, dtype=torch.float64)),
                )
            )
    prefix_bytes = json.dumps(prefix_tokens, separators=(",", ":")).encode("ascii")
    return H7SourceContextView.create(
        prefix_tokens=prefix_tokens,
        prefix_bytes=prefix_bytes,
        z_history=z_history,
        m_history=m_history,
        scorer_rows=tuple(rows),
        source_scorer_profile="h7-linear-history-source-v1",
    )


def _affine(
    *,
    bank: str,
    receiver_t: int,
    source_j: int,
) -> H7RecognitionAffineTrace:
    covariance = torch.diag(
        torch.tensor(
            [0.7 + 0.1 * receiver_t, 1.1],
            dtype=torch.float64,
        )
    )
    precision = torch.diag(1.0 / torch.diagonal(covariance))
    return H7RecognitionAffineTrace(
        component_id=f"q.factorized.{bank}.receiver_{receiver_t}",
        bank=bank,  # type: ignore[arg-type]
        receiver_t=receiver_t,
        source_j=source_j,
        parent_map=torch.eye(2, dtype=torch.float64),
        same_receiver_model_map=(
            None
            if bank == "model"
            else torch.tensor([[0.2, 0.0], [0.1, 0.3]], dtype=torch.float64)
        ),
        offset=torch.tensor(
            [0.1 * receiver_t, -0.2],
            dtype=torch.float64,
            requires_grad=True,
        ),
        covariance=covariance,
        precision=precision,
    )


def test_live_recognition_union_promotes_factorized_and_tracks_entropy() -> None:
    conditioning = RecognitionConditioning.create(
        mode="smoothing",
        horizon=2,
        observed_tokens=torch.tensor([1, 2], dtype=torch.int64),
    )
    mean = torch.tensor(
        [0.2, -0.1, 0.3, 0.4],
        dtype=torch.float64,
        requires_grad=True,
    )
    raw_precision = torch.tensor(
        [0.1, -0.2, 0.05, 0.15],
        dtype=torch.float64,
        requires_grad=True,
    )
    precision_cholesky = torch.diag(torch.exp(raw_precision))
    initial_covariance = torch.diag(torch.exp(-2.0 * raw_precision))
    trace = H7RecognitionCompleteTrace(
        initial_covariance=initial_covariance,
        model_conditionals=(
            _affine(bank="model", receiver_t=1, source_j=0),
            _affine(bank="model", receiver_t=2, source_j=1),
        ),
        state_conditionals=(
            _affine(bank="state", receiver_t=1, source_j=0),
            _affine(bank="state", receiver_t=2, source_j=1),
        ),
        source_context=_source_context(),
    )
    factorized = FactorizedLanguageRecognition.create(
        conditioning=conditioning,
        mean=mean,
        precision_cholesky=precision_cholesky,
        block_sizes=(2, 2),
        h7_trace=trace,
    )
    context = H7RecognitionContextSnapshot.create(
        observation_labels=(1, 2), conditioning="smoothing"
    )
    original = borrow_h7_recognition(factorized, context=context)
    initial = next(
        component for component in original.components if component.receiver_t is None
    )
    assert initial.tensors["mean"].tensor is mean

    elements = (
        torch.tensor(
            [[1.25, 0.1], [0.05, 0.95]],
            dtype=torch.float64,
            requires_grad=True,
        ),
        torch.tensor(
            [[0.85, -0.2], [0.1, 1.15]],
            dtype=torch.float64,
            requires_grad=True,
        ),
        torch.tensor(
            [[1.05, 0.25], [-0.15, 0.9]],
            dtype=torch.float64,
            requires_grad=True,
        ),
    )
    action = borrow_h7_action(elements, kind="internal_product", dimension=2)
    transformed = pushforward_h7_recognition(original, action)
    assert transformed.origin_family == "factorized_diagonal_within_fiber"
    assert transformed.representation == "unrestricted_full_block_pushforward"
    assert transformed.jacobian.entropy_shift is not None
    assert torch.equal(
        transformed.jacobian.entropy_shift.tensor,
        logabsdet_measure_shift(action),
    )
    local_total = (
        transformed.jacobian.initial_logabsdet.tensor
        + torch.stack(
            tuple(
                item.tensor for item in transformed.jacobian.receiver_logabsdet.values()
            )
        ).sum()
    )
    torch.testing.assert_close(
        local_total,
        transformed.jacobian.global_logabsdet.tensor,
        rtol=1.0e-15,
        atol=1.0e-15,
    )
    pushed_initial = next(
        component
        for component in transformed.components
        if component.receiver_t is None
    )
    objective = (
        pushed_initial.tensors["mean"].tensor.sum()
        + pushed_initial.tensors["precision"].tensor.sum()
        + transformed.jacobian.entropy_shift.tensor
    )
    gradients = torch.autograd.grad(objective, (mean, raw_precision, *elements))
    assert all(
        gradient is not None
        and bool(torch.isfinite(gradient).all())
        and bool(torch.any(gradient != 0.0))
        for gradient in gradients
    )
    frozen = freeze_h7_recognition(transformed)
    assert type(frozen) is H7RecognitionSnapshot
    assert type(frozen.jacobian) is H7JacobianMetadataSnapshot
    assert frozen.jacobian.scope == transformed.jacobian.scope
    assert (
        frozen.jacobian.initial_logabsdet.capture_identity
        == transformed.jacobian.initial_logabsdet.identity
    )
    assert tuple(frozen.jacobian.receiver_logabsdet) == tuple(
        transformed.jacobian.receiver_logabsdet
    )
    assert tuple(frozen.jacobian.receiver_logabsdet) == tuple(
        sorted(
            item.component_id
            for item in (*frozen.model_conditionals, *frozen.state_conditionals)
        )
    )
    assert all(
        owned.capture_identity == transformed.jacobian.receiver_logabsdet[name].identity
        for name, owned in frozen.jacobian.receiver_logabsdet.items()
    )
    assert (
        frozen.jacobian.global_logabsdet.capture_identity
        == transformed.jacobian.global_logabsdet.identity
    )
    assert frozen.jacobian.entropy_shift is not None
    assert transformed.jacobian.entropy_shift is not None
    assert (
        frozen.jacobian.entropy_shift.capture_identity
        == transformed.jacobian.entropy_shift.identity
    )
    misbound_jacobian = H7JacobianMetadataSnapshot.create(
        scope="recognition",
        initial_logabsdet=frozen.jacobian.initial_logabsdet,
        receiver_logabsdet={
            f"wrong.receiver.{index}": value
            for index, value in enumerate(frozen.jacobian.receiver_logabsdet.values())
        },
        global_logabsdet=frozen.jacobian.global_logabsdet,
        entropy_shift=frozen.jacobian.entropy_shift,
    )
    with pytest.raises(ValueError, match="conditional component-ID inventory"):
        replace(frozen, jacobian=misbound_jacobian)
    zero_frozen = freeze_h7_recognition(original)
    with pytest.raises(
        ValueError, match="snapshot_sha256 does not match H7RecognitionSnapshot"
    ):
        replace(frozen, jacobian=zero_frozen.jacobian)
    with pytest.raises(
        ValueError,
        match="StructuredLanguageRecognition or FactorizedLanguageRecognition",
    ):
        borrow_h7_recognition(object(), context=context)  # type: ignore[arg-type]
    assert StructuredLanguageRecognition is not FactorizedLanguageRecognition
