from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest
import torch

from vfe4.generative.language import (
    H7LanguageGenerativeGeometry,
    LanguageGenerativeModel,
)
from vfe4.generative.pushforward import (
    borrow_h7_generative,
    freeze_h7_generative,
    pushforward_h7_generative,
)
from vfe4.geometry.group_action import (
    borrow_h7_action,
    logabsdet_measure_shift,
)
from vfe4.types.h6 import (
    CausalDag,
    CausalDagRow,
    H6LanguageStructure,
    VocabularyIdentity,
    ZeroDimensionalBase,
)
from vfe4.types.h7 import (
    H7BorrowedTensorView,
    H7HistoryValueView,
    H7JacobianMetadataSnapshot,
    H7SourceContextView,
    H7SourceScorerRowView,
)


_SHA = "a" * 64


def _borrow(value: torch.Tensor) -> H7BorrowedTensorView:
    return H7BorrowedTensorView.borrow(value)


def _source_context() -> H7SourceContextView:
    prefix_tokens = (0, 2)
    prefix_bytes = json.dumps(prefix_tokens, separators=(",", ":")).encode("ascii")
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
            z_covector = torch.tensor(
                [0.1 * receiver_t, -0.05],
                dtype=torch.float64,
                requires_grad=True,
            )
            m_covector = torch.tensor(
                [-0.04, 0.08 * receiver_t],
                dtype=torch.float64,
                requires_grad=True,
            )
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
    return H7SourceContextView.create(
        prefix_tokens=prefix_tokens,
        prefix_bytes=prefix_bytes,
        z_history=z_history,
        m_history=m_history,
        scorer_rows=tuple(rows),
        source_scorer_profile="h7-linear-history-source-v1",
    )


def _model() -> LanguageGenerativeModel:
    dag = CausalDag.create(
        node_labels=(0, 1, 2),
        rows=(CausalDagRow(1, (0,)), CausalDagRow(2, (1,))),
    )
    structure = H6LanguageStructure.create(
        base=ZeroDimensionalBase.create(),
        dag=dag,
        receiver_labels=(1, 2),
    )
    geometry = H7LanguageGenerativeGeometry(
        frames=(
            torch.eye(2, dtype=torch.float64),
            torch.tensor([[1.1, 0.1], [0.0, 0.9]], dtype=torch.float64),
            torch.tensor([[0.95, -0.1], [0.05, 1.2]], dtype=torch.float64),
        ),
        support_sha256=hashlib.sha256(b"h7-live-support").hexdigest(),
    )
    return LanguageGenerativeModel(
        structure=structure,
        vocabulary=VocabularyIdentity("h6-prefix-small-v1", 3, _SHA),
        model_family_sha256=_SHA,
        latent_dim=2,
        source_prior=None,
        h7_geometry=geometry,
    )


def test_live_generative_pushforward_keeps_graph_and_jacobian_scopes() -> None:
    model = _model()
    context = _source_context()
    original = borrow_h7_generative(model, context=context)
    initial = next(
        component
        for component in original.components
        if component.component_id == "p.initial_joint"
    )
    assert initial.tensors["mean"].tensor is model.initial_mean
    assert (
        next(
            component
            for component in original.components
            if component.component_id == "decoder.1"
        )
        .tensors["state_weight"]
        .tensor
        is model.emission_state_weight
    )

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
    transformed = pushforward_h7_generative(
        original, action, decoder_policy="transform"
    )
    assert torch.equal(
        transformed.jacobian.initial_logabsdet.tensor,
        2.0 * torch.linalg.slogdet(elements[0])[1],
    )
    assert len(transformed.jacobian.receiver_logabsdet) == 4
    assert torch.equal(
        transformed.jacobian.global_logabsdet.tensor,
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
        if component.component_id == "p.initial_joint"
    )
    objective = (
        pushed_initial.tensors["mean"].tensor.sum()
        + transformed.jacobian.global_logabsdet.tensor
    )
    gradients = torch.autograd.grad(objective, (model.initial_mean, *elements))
    assert all(
        gradient is not None
        and bool(torch.isfinite(gradient).all())
        and bool(torch.any(gradient != 0.0))
        for gradient in gradients
    )
    frozen = freeze_h7_generative(transformed, action=action)
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
        sorted(item.component_id for item in frozen.transitions)
    )
    assert all(
        owned.capture_identity == transformed.jacobian.receiver_logabsdet[name].identity
        for name, owned in frozen.jacobian.receiver_logabsdet.items()
    )
    assert (
        frozen.jacobian.global_logabsdet.capture_identity
        == transformed.jacobian.global_logabsdet.identity
    )
    assert frozen.jacobian.entropy_shift is None
    with pytest.raises(
        ValueError,
        match="Jacobian receiver component ID has an ambiguous bank",
    ):
        H7JacobianMetadataSnapshot.create(
            scope="generative",
            initial_logabsdet=frozen.jacobian.initial_logabsdet,
            receiver_logabsdet={
                f"wrong.receiver.{index}": value
                for index, value in enumerate(
                    frozen.jacobian.receiver_logabsdet.values()
                )
            },
            global_logabsdet=frozen.jacobian.global_logabsdet,
            entropy_shift=None,
        )
    misbound_jacobian = H7JacobianMetadataSnapshot.create(
        scope="generative",
        initial_logabsdet=frozen.jacobian.initial_logabsdet,
        receiver_logabsdet={
            f"wrong.{name}": value
            for name, value in frozen.jacobian.receiver_logabsdet.items()
        },
        global_logabsdet=frozen.jacobian.global_logabsdet,
        entropy_shift=None,
    )
    with pytest.raises(ValueError, match="transition component-ID inventory"):
        replace(frozen, jacobian=misbound_jacobian)
    zero_frozen = freeze_h7_generative(original)
    with pytest.raises(
        ValueError, match="snapshot_sha256 does not match H7GenerativeSnapshot"
    ):
        replace(frozen, jacobian=zero_frozen.jacobian)
    assert (
        frozen.initial_joint.mean.raw_bytes_sha256
        == hashlib.sha256(frozen.initial_joint.mean.raw_bytes).hexdigest()
    )
    with pytest.raises(ValueError, match="exact LanguageGenerativeModel"):
        borrow_h7_generative(object(), context=context)  # type: ignore[arg-type]
