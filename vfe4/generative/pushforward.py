"""Graph-preserving H7 pushforwards for exact live generative models."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Literal

import torch

from vfe4.generative.language import (
    H7LanguageGenerativeTrace,
    LanguageGenerativeModel,
)
from vfe4.geometry.group_action import (
    centered_logit_projector,
    frame_links,
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
    H7AffineComponentSnapshot,
    H7BorrowedActionView,
    H7BorrowedTensorView,
    H7DecoderPolicy,
    H7DecoderSnapshot,
    H7GaussianComponentSnapshot,
    H7GenerativeSnapshot,
    H7GenerativeTensorLaw,
    H7HistoryValueSnapshot,
    H7HistoryValueView,
    H7JacobianMetadataSnapshot,
    H7JacobianMetadataView,
    H7OwnedTensorSnapshot,
    H7SourceContextSnapshot,
    H7SourceContextView,
    H7SourceCovectorSnapshot,
    H7SourceScorerRowSnapshot,
    H7SourceScorerRowView,
    H7TensorLawComponent,
    _jacobian_grouped_local_total,
    canonical_h7_bytes,
    h7_owned_sha256,
)


_GAUSSIAN_TENSOR_NAMES = (
    "mean",
    "covariance",
    "precision",
    "information_vector",
    "second_moment",
)
_DECODER_TENSOR_NAMES = {"state_weight", "model_weight", "bias"}


def _borrow(value: torch.Tensor) -> H7BorrowedTensorView:
    if not isinstance(value, torch.Tensor):
        raise ValueError("H7 live values must be tensors")
    return H7BorrowedTensorView.borrow(value)


def _component(
    *,
    component_id: str,
    receiver_t: int | None,
    source_j: int | None,
    tensors: Mapping[str, torch.Tensor],
) -> H7TensorLawComponent:
    return H7TensorLawComponent.create(
        component_id=component_id,
        receiver_t=receiver_t,
        source_j=source_j,
        tensors={name: _borrow(value) for name, value in tensors.items()},
    )


def _action_elements(
    action: H7BorrowedActionView, *, dimension: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if type(action) is not H7BorrowedActionView:
        raise ValueError("action must be an exact borrowed H7 action")
    action.assert_intact()
    if action.dimension != dimension:
        raise ValueError("action dimension does not match the H7 law")
    elements = tuple(
        require_direct_gl_plus(item.tensor, dimension=dimension)
        for item in action.elements
    )
    if action.kind == "diagonal_base" and not all(
        torch.equal(elements[0], item) for item in elements[1:]
    ):
        raise ValueError("diagonal_base action elements must be identical")
    return elements  # type: ignore[return-value]


def _gaussian_from_log_scale(
    mean: torch.Tensor, log_scale: torch.Tensor
) -> dict[str, torch.Tensor]:
    variance = torch.exp(2.0 * log_scale)
    covariance = torch.diag(variance)
    precision = torch.diag(torch.exp(-2.0 * log_scale))
    return {
        "mean": mean,
        "covariance": covariance,
        "precision": precision,
        "information_vector": precision @ mean,
        "second_moment": covariance + torch.outer(mean, mean),
    }


def _gaussian_from_moment_precision(
    mean: torch.Tensor,
    covariance: torch.Tensor,
    precision: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        "mean": mean,
        "covariance": covariance,
        "precision": precision,
        "information_vector": precision @ mean,
        "second_moment": covariance + torch.outer(mean, mean),
    }


def _push_gaussian(
    tensors: Mapping[str, H7BorrowedTensorView],
    receiver: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if set(tensors) != set(_GAUSSIAN_TENSOR_NAMES):
        raise ValueError("Gaussian tensor-law component is incomplete")
    return {
        "mean": push_vector(tensors["mean"].tensor, receiver),
        "covariance": push_covariance(tensors["covariance"].tensor, receiver),
        "precision": push_precision(tensors["precision"].tensor, receiver),
        "information_vector": push_information_vector(
            tensors["information_vector"].tensor, receiver
        ),
        "second_moment": push_second_moment(tensors["second_moment"].tensor, receiver),
    }


def _affine_components(
    components: tuple[H7TensorLawComponent, ...],
) -> tuple[H7TensorLawComponent, ...]:
    required = set(_GAUSSIAN_TENSOR_NAMES) | {"parent_map", "offset"}
    result = tuple(
        component
        for component in components
        if set(component.tensors)
        in (
            required,
            required | {"same_receiver_model_map"},
        )
    )
    if not result:
        raise ValueError("complete H7 law has no affine receiver factors")
    return result


def _zero_jacobian(
    components: tuple[H7TensorLawComponent, ...],
    *,
    scope: Literal["generative", "recognition"],
) -> H7JacobianMetadataView:
    affine = _affine_components(components)
    anchor = components[0].tensors[next(iter(components[0].tensors))].tensor
    zero = anchor.reshape(-1).sum() * 0.0
    zero_view = _borrow(zero)
    return H7JacobianMetadataView.create(
        scope=scope,
        initial_logabsdet=zero_view,
        receiver_logabsdet={component.component_id: zero_view for component in affine},
        global_logabsdet=zero_view,
        entropy_shift=(zero_view if scope == "recognition" else None),
    )


def _action_jacobian(
    components: tuple[H7TensorLawComponent, ...],
    action: H7BorrowedActionView,
    *,
    scope: Literal["generative", "recognition"],
) -> H7JacobianMetadataView:
    elements = _action_elements(action, dimension=action.dimension)
    logdets = tuple(torch.linalg.slogdet(item)[1] for item in elements)
    initial = 2.0 * logdets[0]
    receiver_values = {
        component.component_id: logdets[component.receiver_t]
        for component in _affine_components(components)
        if component.receiver_t is not None
    }
    if len(receiver_values) != len(_affine_components(components)):
        raise ValueError("affine Jacobian scope is missing receiver_t")
    global_shift = logabsdet_measure_shift(action)
    local_total = _jacobian_grouped_local_total(initial, receiver_values)
    eps = torch.finfo(torch.float64).eps
    scale = max(
        1.0,
        float(torch.abs(local_total).item()),
        float(torch.abs(global_shift).item()),
    )
    if not torch.allclose(
        local_total,
        global_shift,
        rtol=64.0 * eps,
        atol=64.0 * eps * scale,
    ):
        raise ValueError(
            "per-factor Jacobian shifts do not sum to logabsdet_measure_shift"
        )
    global_view = _borrow(global_shift)
    return H7JacobianMetadataView.create(
        scope=scope,
        initial_logabsdet=_borrow(initial),
        receiver_logabsdet={
            name: _borrow(value) for name, value in receiver_values.items()
        },
        global_logabsdet=global_view,
        entropy_shift=(global_view if scope == "recognition" else None),
    )


def _validate_source_context(source: H7SourceContextView, *, dimension: int) -> None:
    if type(source) is not H7SourceContextView:
        raise ValueError("context must be an exact H7SourceContextView")
    source.assert_live()
    expected_prefix_bytes = json.dumps(
        source.prefix_tokens, separators=(",", ":")
    ).encode("ascii")
    if (
        dimension != 2
        or len(source.prefix_tokens) != 2
        or source.prefix_bytes != expected_prefix_bytes
        or tuple(
            (item.channel, item.population_label)
            for item in (*source.z_history, *source.m_history)
        )
        != (("z", 0), ("z", 1), ("m", 0), ("m", 1))
        or tuple((row.bank, row.receiver_t, row.source_j) for row in source.scorer_rows)
        != (
            ("model", 1, 0),
            ("model", 2, 1),
            ("state", 1, 0),
            ("state", 2, 1),
        )
    ):
        raise ValueError("live H7 source-context inventory is incomplete")
    for item in (*source.z_history, *source.m_history):
        if item.value.identity.shape != (dimension,):
            raise ValueError("source history width disagrees with the law")
    for row in source.scorer_rows:
        if (
            row.z_history is not source.z_history
            or row.m_history is not source.m_history
            or row.prefix_tokens != source.prefix_tokens[: row.receiver_t]
            or row.prefix_bytes
            != json.dumps(row.prefix_tokens, separators=(",", ":")).encode("ascii")
            or row.mask != (True,)
            or row.support != (row.source_j,)
            or row.z_covector.identity.shape != (dimension,)
            or row.m_covector.identity.shape != (dimension,)
            or row.raw_scores.identity.shape != (1,)
            or row.probabilities.identity.shape != (1,)
            or not torch.equal(
                row.probabilities.tensor,
                torch.ones_like(row.probabilities.tensor),
            )
        ):
            raise ValueError("live H7 source scorer row is incomplete")
        weighted = sum(
            (index + 1) * (token + 1) for index, token in enumerate(row.prefix_tokens)
        )
        expected_prefix = row.alpha_bias + row.alpha_token_scale * weighted
        expected_score = (
            torch.as_tensor(
                expected_prefix,
                dtype=row.raw_scores.tensor.dtype,
                device=row.raw_scores.tensor.device,
            )
            + row.z_covector.tensor @ source.z_history[row.source_j].value.tensor
            + row.m_covector.tensor @ source.m_history[row.source_j].value.tensor
        ).reshape(1)
        eps = torch.finfo(torch.float64).eps
        if row.prefix_term != expected_prefix or not torch.allclose(
            row.raw_scores.tensor,
            expected_score,
            rtol=32.0 * eps,
            atol=32.0 * eps,
        ):
            raise ValueError("live H7 source scorer law is inconsistent")


def _push_source_context(
    source: H7SourceContextView,
    elements: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> H7SourceContextView:
    _validate_source_context(source, dimension=elements[0].shape[0])
    z_history = tuple(
        H7HistoryValueView(
            item.channel,
            item.population_label,
            _borrow(push_vector(item.value.tensor, elements[item.population_label])),
        )
        for item in source.z_history
    )
    m_history = tuple(
        H7HistoryValueView(
            item.channel,
            item.population_label,
            _borrow(push_vector(item.value.tensor, elements[item.population_label])),
        )
        for item in source.m_history
    )
    rows: list[H7SourceScorerRowView] = []
    for row in source.scorer_rows:
        source_action = elements[row.source_j]
        z_covector = push_information_vector(row.z_covector.tensor, source_action)
        m_covector = push_information_vector(row.m_covector.tensor, source_action)
        raw_scores = (
            torch.as_tensor(
                row.prefix_term,
                dtype=z_covector.dtype,
                device=z_covector.device,
            )
            + z_covector @ z_history[row.source_j].value.tensor
            + m_covector @ m_history[row.source_j].value.tensor
        ).reshape(1)
        rows.append(
            H7SourceScorerRowView.create(
                bank=row.bank,
                receiver_t=row.receiver_t,
                source_j=row.source_j,
                prefix_tokens=row.prefix_tokens,
                prefix_bytes=row.prefix_bytes,
                alpha_bias=row.alpha_bias,
                alpha_token_scale=row.alpha_token_scale,
                prefix_term=row.prefix_term,
                z_history=z_history,
                m_history=m_history,
                z_covector=_borrow(z_covector),
                m_covector=_borrow(m_covector),
                mask=row.mask,
                support=row.support,
                raw_scores=_borrow(raw_scores),
                probabilities=row.probabilities,
            )
        )
    result = H7SourceContextView.create(
        prefix_tokens=source.prefix_tokens,
        prefix_bytes=source.prefix_bytes,
        z_history=z_history,
        m_history=m_history,
        scorer_rows=tuple(rows),
        source_scorer_profile=source.source_scorer_profile,
    )
    _validate_source_context(result, dimension=elements[0].shape[0])
    return result


def _live_components(
    trace: H7LanguageGenerativeTrace,
) -> tuple[H7TensorLawComponent, ...]:
    trace.__post_init__()
    components: list[H7TensorLawComponent] = []
    for population, frame in enumerate(trace.frames):
        components.append(
            _component(
                component_id=f"frame.{population}",
                receiver_t=population,
                source_j=None,
                tensors={"frame": frame},
            )
        )
    for (receiver_t, source_j), value in frame_links(trace.frames).items():
        components.append(
            _component(
                component_id=f"link.{receiver_t}<-{source_j}",
                receiver_t=receiver_t,
                source_j=source_j,
                tensors={"map": value},
            )
        )
    components.append(
        _component(
            component_id="p.initial_joint",
            receiver_t=None,
            source_j=None,
            tensors=_gaussian_from_log_scale(
                trace.initial_mean, trace.initial_log_scale
            ),
        )
    )
    model_gaussian = _gaussian_from_log_scale(
        trace.model_transition_bias, trace.model_transition_log_scale
    )
    state_gaussian = _gaussian_from_log_scale(
        trace.state_transition_bias, trace.state_transition_log_scale
    )
    for receiver_t, source_j in ((1, 0), (2, 1)):
        components.append(
            _component(
                component_id=f"p.model.receiver_{receiver_t}",
                receiver_t=receiver_t,
                source_j=source_j,
                tensors={
                    **model_gaussian,
                    "parent_map": trace.model_transition_weight,
                    "offset": trace.model_transition_bias,
                },
            )
        )
        components.append(
            _component(
                component_id=f"p.state.receiver_{receiver_t}",
                receiver_t=receiver_t,
                source_j=source_j,
                tensors={
                    **state_gaussian,
                    "parent_map": trace.state_transition_weight,
                    "same_receiver_model_map": trace.state_model_weight,
                    "offset": trace.state_transition_bias,
                },
            )
        )
    for receiver_t in (1, 2):
        components.append(
            _component(
                component_id=f"decoder.{receiver_t}",
                receiver_t=receiver_t,
                source_j=None,
                tensors={
                    "state_weight": trace.emission_state_weight,
                    "model_weight": trace.emission_model_weight,
                    "bias": trace.emission_bias,
                },
            )
        )
    return tuple(components)


def borrow_h7_generative(
    model: LanguageGenerativeModel,
    *,
    context: H7SourceContextView | None,
) -> H7GenerativeTensorLaw:
    """Borrow the exact live H6 model and its complete additive H7 trace."""

    if type(model) is not LanguageGenerativeModel:
        raise ValueError("model must be an exact LanguageGenerativeModel")
    trace = model.export_h7_trace()
    if context is None:
        raise ValueError("live matrix H7 generative laws require source context")
    _validate_source_context(context, dimension=trace.frames[0].shape[0])
    components = _live_components(trace)
    return H7GenerativeTensorLaw.create(
        components=components,
        source_context=context,
        scalar_source_law=None,
        decoder_policy="transform",
        support_sha256=trace.support_sha256,
        jacobian=_zero_jacobian(components, scope="generative"),
    )


def pushforward_h7_generative(
    law: H7GenerativeTensorLaw,
    action: H7BorrowedActionView,
    *,
    decoder_policy: H7DecoderPolicy = "transform",
) -> H7GenerativeTensorLaw:
    """Push a complete borrowed generative law through one direct action."""

    if type(law) is not H7GenerativeTensorLaw:
        raise ValueError("law must be an exact H7GenerativeTensorLaw")
    law.assert_live()
    if decoder_policy not in ("transform", "fixed"):
        raise ValueError("unsupported decoder policy")
    frame_components = tuple(
        item for item in law.components if set(item.tensors) == {"frame"}
    )
    if tuple(item.component_id for item in frame_components) != (
        "frame.0",
        "frame.1",
        "frame.2",
    ):
        raise ValueError("generative frame inventory is incomplete")
    dimension = frame_components[0].tensors["frame"].identity.shape[0]
    elements = _action_elements(action, dimension=dimension)
    components: list[H7TensorLawComponent] = []
    gaussian_names = set(_GAUSSIAN_TENSOR_NAMES)
    affine_names = gaussian_names | {"parent_map", "offset"}
    for component in law.components:
        names = set(component.tensors)
        if names == {"frame"}:
            receiver_t = component.receiver_t
            if receiver_t is None:
                raise ValueError("frame component has no population")
            tensors = {
                "frame": elements[receiver_t] @ component.tensors["frame"].tensor
            }
        elif names == {"map"}:
            if component.receiver_t is None or component.source_j is None:
                raise ValueError("link component is missing endpoints")
            tensors = {
                "map": push_receiver_source_map(
                    component.tensors["map"].tensor,
                    elements[component.receiver_t],
                    elements[component.source_j],
                )
            }
        elif names == gaussian_names:
            if component.receiver_t is not None or component.source_j is not None:
                raise ValueError("standalone Gaussian must be the initial joint")
            tensors = _push_gaussian(
                component.tensors,
                torch.block_diag(elements[0], elements[0]),
            )
        elif names in (
            affine_names,
            affine_names | {"same_receiver_model_map"},
        ):
            if component.receiver_t is None or component.source_j is None:
                raise ValueError("affine component is missing endpoints")
            receiver = elements[component.receiver_t]
            source = elements[component.source_j]
            tensors = _push_gaussian(
                {name: component.tensors[name] for name in gaussian_names},
                receiver,
            )
            tensors["parent_map"] = push_receiver_source_map(
                component.tensors["parent_map"].tensor, receiver, source
            )
            tensors["offset"] = push_vector(
                component.tensors["offset"].tensor, receiver
            )
            if "same_receiver_model_map" in component.tensors:
                tensors["same_receiver_model_map"] = push_same_receiver_morphism(
                    component.tensors["same_receiver_model_map"].tensor,
                    receiver,
                    receiver,
                )
        elif names == _DECODER_TENSOR_NAMES:
            if component.receiver_t is None:
                raise ValueError("decoder component has no receiver")
            receiver = elements[component.receiver_t]
            tensors = {
                "state_weight": (
                    push_decoder(component.tensors["state_weight"].tensor, receiver)
                    if decoder_policy == "transform"
                    else component.tensors["state_weight"].tensor
                ),
                "model_weight": (
                    push_decoder(component.tensors["model_weight"].tensor, receiver)
                    if decoder_policy == "transform"
                    else component.tensors["model_weight"].tensor
                ),
                "bias": component.tensors["bias"].tensor,
            }
        else:
            raise ValueError(
                f"unsupported generative component {component.component_id}"
            )
        components.append(
            _component(
                component_id=component.component_id,
                receiver_t=component.receiver_t,
                source_j=component.source_j,
                tensors=tensors,
            )
        )
    transformed = tuple(components)
    context = (
        None
        if law.source_context is None
        else _push_source_context(law.source_context, elements)
    )
    return H7GenerativeTensorLaw.create(
        components=transformed,
        source_context=context,
        scalar_source_law=law.scalar_source_law,
        decoder_policy=decoder_policy,
        support_sha256=law.support_sha256,
        jacobian=_action_jacobian(transformed, action, scope="generative"),
    )


def _owned_value(source: H7OwnedTensorSnapshot) -> torch.Tensor:
    source.assert_intact()
    return source.value()


def _snapshot_gaussian(
    source: H7GaussianComponentSnapshot,
) -> dict[str, torch.Tensor]:
    return {
        name: _owned_value(getattr(source, name)) for name in _GAUSSIAN_TENSOR_NAMES
    }


def _source_context_from_snapshot(
    source: H7SourceContextSnapshot,
) -> H7SourceContextView:
    source.__post_init__()
    z_history = tuple(
        H7HistoryValueView(
            item.channel,
            item.population_label,
            _borrow(_owned_value(item.value)),
        )
        for item in source.z_history
    )
    m_history = tuple(
        H7HistoryValueView(
            item.channel,
            item.population_label,
            _borrow(_owned_value(item.value)),
        )
        for item in source.m_history
    )
    rows = tuple(
        H7SourceScorerRowView.create(
            bank=row.bank,
            receiver_t=row.receiver_t,
            source_j=row.source_j,
            prefix_tokens=row.prefix_tokens,
            prefix_bytes=row.prefix_bytes,
            alpha_bias=row.alpha_bias,
            alpha_token_scale=row.alpha_token_scale,
            prefix_term=row.prefix_term,
            z_history=z_history,
            m_history=m_history,
            z_covector=_borrow(_owned_value(row.z_covector.value)),
            m_covector=_borrow(_owned_value(row.m_covector.value)),
            mask=row.mask,
            support=row.support,
            raw_scores=_borrow(_owned_value(row.raw_scores)),
            probabilities=_borrow(_owned_value(row.probabilities)),
        )
        for row in source.scorer_rows
    )
    result = H7SourceContextView.create(
        prefix_tokens=source.prefix_tokens,
        prefix_bytes=source.prefix_bytes,
        z_history=z_history,
        m_history=m_history,
        scorer_rows=rows,
        source_scorer_profile=source.source_scorer_profile,
    )
    _validate_source_context(result, dimension=source.z_history[0].value.shape[0])
    return result


def _borrow_h7_generative_snapshot(
    source: H7GenerativeSnapshot,
    *,
    decoder_policy: H7DecoderPolicy = "transform",
) -> H7GenerativeTensorLaw:
    """Private fixture adapter; not evidence for the live-model seam."""

    if type(source) is not H7GenerativeSnapshot:
        raise ValueError("source must be an exact H7GenerativeSnapshot")
    source.__post_init__()
    components: list[H7TensorLawComponent] = []
    for population, frame in enumerate(source.frames):
        components.append(
            _component(
                component_id=f"frame.{population}",
                receiver_t=population,
                source_j=None,
                tensors={"frame": _owned_value(frame)},
            )
        )
    for (receiver_t, source_j), value in source.ordered_links.items():
        components.append(
            _component(
                component_id=f"link.{receiver_t}<-{source_j}",
                receiver_t=receiver_t,
                source_j=source_j,
                tensors={"map": _owned_value(value)},
            )
        )
    components.append(
        _component(
            component_id=source.initial_joint.component_id,
            receiver_t=None,
            source_j=None,
            tensors=_snapshot_gaussian(source.initial_joint),
        )
    )
    for item in source.transitions:
        tensors = {
            **_snapshot_gaussian(item.receiver_law),
            "parent_map": _owned_value(item.parent_map),
            "offset": _owned_value(item.offset),
        }
        if item.same_receiver_model_map is not None:
            tensors["same_receiver_model_map"] = _owned_value(
                item.same_receiver_model_map
            )
        components.append(
            _component(
                component_id=item.component_id,
                receiver_t=item.receiver_t,
                source_j=item.source_j,
                tensors=tensors,
            )
        )
    for item in source.decoders:
        components.append(
            _component(
                component_id=f"decoder.{item.receiver_t}",
                receiver_t=item.receiver_t,
                source_j=None,
                tensors={
                    "state_weight": _owned_value(item.state_weight),
                    "model_weight": _owned_value(item.model_weight),
                    "bias": _owned_value(item.bias),
                },
            )
        )
    frozen_components = tuple(components)
    return H7GenerativeTensorLaw.create(
        components=frozen_components,
        source_context=(
            None
            if source.source_context is None
            else _source_context_from_snapshot(source.source_context)
        ),
        scalar_source_law=source.scalar_source_law,
        decoder_policy=decoder_policy,
        support_sha256=source.support_sha256,
        jacobian=_zero_jacobian(frozen_components, scope="generative"),
    )


def _pushforward_h7_generative_snapshot(
    source: H7GenerativeSnapshot,
    action: H7BorrowedActionView,
    *,
    decoder_policy: H7DecoderPolicy = "transform",
) -> H7GenerativeTensorLaw:
    return pushforward_h7_generative(
        _borrow_h7_generative_snapshot(source, decoder_policy=decoder_policy),
        action,
        decoder_policy=decoder_policy,
    )


def pushforward_h7_generative_snapshot(
    source: H7GenerativeSnapshot,
    action: H7BorrowedActionView,
    *,
    decoder_policy: H7DecoderPolicy = "transform",
) -> H7GenerativeTensorLaw:
    """Push forward a frozen H7 generative fixture snapshot."""

    return _pushforward_h7_generative_snapshot(
        source,
        action,
        decoder_policy=decoder_policy,
    )


def _freeze_tensor(view: H7BorrowedTensorView) -> H7OwnedTensorSnapshot:
    if type(view) is not H7BorrowedTensorView:
        raise ValueError("freeze requires exact borrowed tensor views")
    view.assert_intact()
    return H7OwnedTensorSnapshot.capture(view.tensor)


def _freeze_jacobian(
    metadata: H7JacobianMetadataView,
) -> H7JacobianMetadataSnapshot:
    if type(metadata) is not H7JacobianMetadataView:
        raise ValueError("freeze requires exact live Jacobian metadata")
    metadata.assert_live()
    return H7JacobianMetadataSnapshot.create(
        scope=metadata.scope,
        initial_logabsdet=_freeze_tensor(metadata.initial_logabsdet),
        receiver_logabsdet={
            name: _freeze_tensor(value)
            for name, value in metadata.receiver_logabsdet.items()
        },
        global_logabsdet=_freeze_tensor(metadata.global_logabsdet),
        entropy_shift=(
            None
            if metadata.entropy_shift is None
            else _freeze_tensor(metadata.entropy_shift)
        ),
    )


def _freeze_gaussian(
    component: H7TensorLawComponent,
    *,
    component_id: str | None = None,
) -> H7GaussianComponentSnapshot:
    if set(component.tensors) != set(_GAUSSIAN_TENSOR_NAMES):
        raise ValueError("Gaussian component is incomplete")
    return H7GaussianComponentSnapshot.create(
        component_id=(component.component_id if component_id is None else component_id),
        receiver_t=component.receiver_t,
        source_j=component.source_j,
        **{
            name: _freeze_tensor(component.tensors[name])
            for name in _GAUSSIAN_TENSOR_NAMES
        },
    )


def _freeze_source_context(
    source: H7SourceContextView,
) -> H7SourceContextSnapshot:
    _validate_source_context(
        source, dimension=source.z_history[0].value.identity.shape[0]
    )
    z_history = tuple(
        H7HistoryValueSnapshot.create(
            channel=item.channel,
            population_label=item.population_label,
            value=_freeze_tensor(item.value),
        )
        for item in source.z_history
    )
    m_history = tuple(
        H7HistoryValueSnapshot.create(
            channel=item.channel,
            population_label=item.population_label,
            value=_freeze_tensor(item.value),
        )
        for item in source.m_history
    )
    rows: list[H7SourceScorerRowSnapshot] = []
    for row in source.scorer_rows:
        z_covector = H7SourceCovectorSnapshot.create(
            bank=row.bank,
            channel="z",
            receiver_t=row.receiver_t,
            source_j=row.source_j,
            value=_freeze_tensor(row.z_covector),
        )
        m_covector = H7SourceCovectorSnapshot.create(
            bank=row.bank,
            channel="m",
            receiver_t=row.receiver_t,
            source_j=row.source_j,
            value=_freeze_tensor(row.m_covector),
        )
        raw_score = float(row.raw_scores.tensor[0])
        row_bytes = canonical_h7_bytes(
            {
                "bank": row.bank,
                "receiver_t": row.receiver_t,
                "source_j": row.source_j,
                "prefix_tokens": row.prefix_tokens,
                "prefix_term": row.prefix_term,
                "z_covector_sha256": z_covector.covector_sha256,
                "m_covector_sha256": m_covector.covector_sha256,
                "raw_score": raw_score,
                "support": row.support,
            }
        )
        rows.append(
            H7SourceScorerRowSnapshot.create(
                bank=row.bank,
                receiver_t=row.receiver_t,
                source_j=row.source_j,
                prefix_tokens=row.prefix_tokens,
                prefix_bytes=row.prefix_bytes,
                prefix_bytes_sha256=hashlib.sha256(row.prefix_bytes).hexdigest(),
                alpha_bias=row.alpha_bias,
                alpha_token_scale=row.alpha_token_scale,
                prefix_term=row.prefix_term,
                z_history=z_history,
                m_history=m_history,
                z_covector=z_covector,
                m_covector=m_covector,
                mask=row.mask,
                support=row.support,
                raw_scores=_freeze_tensor(row.raw_scores),
                probabilities=_freeze_tensor(row.probabilities),
                source_row_raw_bytes=row_bytes,
                row_raw_bytes_sha256=hashlib.sha256(row_bytes).hexdigest(),
            )
        )
    scorer_sha256 = h7_owned_sha256(
        "vfe4.h7.source-scorer.v1",
        tuple(row.row_sha256 for row in rows),
    )
    return H7SourceContextSnapshot.create(
        prefix_tokens=source.prefix_tokens,
        prefix_bytes=source.prefix_bytes,
        prefix_bytes_sha256=hashlib.sha256(source.prefix_bytes).hexdigest(),
        z_history=z_history,
        m_history=m_history,
        scorer_rows=tuple(rows),
        source_scorer_profile=source.source_scorer_profile,
        source_scorer_sha256=scorer_sha256,
    )


def _decoder_class(
    *,
    component: H7TensorLawComponent,
    action: H7BorrowedActionView | None,
    decoder_policy: H7DecoderPolicy,
) -> Literal["transformed", "inside", "outside"]:
    if decoder_policy == "transform":
        return "transformed"
    if action is None:
        raise ValueError("fixed-decoder evidence requires the applied action")
    action.assert_intact()
    receiver_t = component.receiver_t
    if receiver_t is None:
        raise ValueError("decoder component is missing receiver_t")
    receiver = action.elements[receiver_t].tensor
    state = component.tensors["state_weight"].tensor
    model = component.tensors["model_weight"].tensor
    projector = centered_logit_projector(state.shape[0], like=state)
    residuals = tuple(
        right_solve(projector @ value, receiver) - projector @ value
        for value in (state, model)
    )
    scale = max(
        1.0,
        *(float(torch.max(torch.abs(item)).item()) for item in residuals),
    )
    allowance = 256.0 * torch.finfo(state.dtype).eps * scale
    return (
        "inside"
        if all(
            bool(torch.max(torch.abs(item)).item() <= allowance) for item in residuals
        )
        else "outside"
    )


def freeze_h7_generative(
    law: H7GenerativeTensorLaw,
    *,
    action: H7BorrowedActionView | None = None,
) -> H7GenerativeSnapshot:
    """Clone one complete live tensor law into owned H7 evidence."""

    if type(law) is not H7GenerativeTensorLaw:
        raise ValueError("law must be an exact H7GenerativeTensorLaw")
    law.assert_live()
    index = {item.component_id: item for item in law.components}
    if len(index) != len(law.components):
        raise ValueError("generative component IDs must be unique")
    frames = tuple(
        _freeze_tensor(index[f"frame.{population}"].tensors["frame"])
        for population in range(3)
    )
    links = {
        (receiver, source): _freeze_tensor(
            index[f"link.{receiver}<-{source}"].tensors["map"]
        )
        for receiver in range(3)
        for source in range(3)
        if receiver != source
    }
    gaussian_names = set(_GAUSSIAN_TENSOR_NAMES)
    initial_candidates = tuple(
        item
        for item in law.components
        if set(item.tensors) == gaussian_names
        and item.receiver_t is None
        and item.source_j is None
    )
    if len(initial_candidates) != 1:
        raise ValueError("generative initial Gaussian inventory is invalid")
    initial = _freeze_gaussian(initial_candidates[0])
    affine = _affine_components(law.components)
    transitions: list[H7AffineComponentSnapshot] = []
    for item in sorted(
        affine,
        key=lambda value: (
            value.receiver_t,
            1 if "same_receiver_model_map" in value.tensors else 0,
        ),
    ):
        state = "same_receiver_model_map" in item.tensors
        receiver_component = H7TensorLawComponent.create(
            component_id=f"{item.component_id}.receiver",
            receiver_t=item.receiver_t,
            source_j=item.source_j,
            tensors={name: item.tensors[name] for name in _GAUSSIAN_TENSOR_NAMES},
        )
        transitions.append(
            H7AffineComponentSnapshot.create(
                component_id=item.component_id,
                bank=("state" if state else "model"),
                receiver_t=item.receiver_t,
                source_j=item.source_j,
                parent_map=_freeze_tensor(item.tensors["parent_map"]),
                same_receiver_model_map=(
                    _freeze_tensor(item.tensors["same_receiver_model_map"])
                    if state
                    else None
                ),
                offset=_freeze_tensor(item.tensors["offset"]),
                receiver_law=_freeze_gaussian(receiver_component),
            )
        )
    decoders: list[H7DecoderSnapshot] = []
    for receiver_t in (1, 2):
        component = index[f"decoder.{receiver_t}"]
        if set(component.tensors) != _DECODER_TENSOR_NAMES:
            raise ValueError("decoder component is incomplete")
        decoders.append(
            H7DecoderSnapshot.create(
                receiver_t=receiver_t,
                state_weight=_freeze_tensor(component.tensors["state_weight"]),
                model_weight=_freeze_tensor(component.tensors["model_weight"]),
                bias=_freeze_tensor(component.tensors["bias"]),
                centered_stabilizer_class=_decoder_class(
                    component=component,
                    action=action,
                    decoder_policy=law.decoder_policy,
                ),
            )
        )
    expected_count = 3 + 6 + 1 + len(affine) + 2
    if len(law.components) != expected_count:
        raise ValueError("generative tensor law has extra components")
    return H7GenerativeSnapshot.create(
        frames=frames,
        ordered_links=links,
        initial_joint=initial,
        transitions=tuple(transitions),
        source_context=(
            None
            if law.source_context is None
            else _freeze_source_context(law.source_context)
        ),
        scalar_source_law=law.scalar_source_law,
        decoders=tuple(decoders),
        support_sha256=law.support_sha256,
        jacobian=_freeze_jacobian(law.jacobian),
    )


__all__ = [
    "borrow_h7_generative",
    "freeze_h7_generative",
    "pushforward_h7_generative",
    "pushforward_h7_generative_snapshot",
]
