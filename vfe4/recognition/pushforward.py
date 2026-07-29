"""Graph-preserving H7 pushforwards for exact live recognition families."""

from __future__ import annotations

from typing import TypeAlias

import torch

from vfe4.generative.pushforward import (
    _GAUSSIAN_TENSOR_NAMES,
    _action_elements,
    _action_jacobian,
    _component,
    _freeze_gaussian,
    _freeze_jacobian,
    _freeze_source_context,
    _freeze_tensor,
    _gaussian_from_moment_precision,
    _owned_value,
    _push_gaussian,
    _push_source_context,
    _snapshot_gaussian,
    _source_context_from_snapshot,
    _validate_source_context,
    _zero_jacobian,
)
from vfe4.geometry.group_action import (
    push_receiver_source_map,
    push_same_receiver_morphism,
    push_vector,
)
from vfe4.recognition.language import (
    FactorizedLanguageRecognition,
    H7LanguageRecognitionTrace,
    StructuredLanguageRecognition,
)
from vfe4.types.h7 import (
    H7AffineComponentSnapshot,
    H7BorrowedActionView,
    H7RecognitionContextSnapshot,
    H7RecognitionRepresentation,
    H7RecognitionSnapshot,
    H7RecognitionTensorLaw,
    H7SourceContextSnapshot,
    H7SourceContextView,
    H7SourceScorerRowSnapshot,
    H7SourceScorerRowView,
    H7TensorLawComponent,
    h7_owned_sha256,
)


H7RecognitionInput: TypeAlias = (
    StructuredLanguageRecognition | FactorizedLanguageRecognition
)


def _is_exact_diagonal(value: torch.Tensor) -> bool:
    return torch.equal(value, torch.diag(torch.diagonal(value)))


def _trace_components(
    trace: H7LanguageRecognitionTrace,
    *,
    origin: str,
) -> tuple[H7TensorLawComponent, ...]:
    trace.__post_init__()
    precision = trace.initial_precision()
    components = [
        _component(
            component_id=f"q.{origin}.initial_joint",
            receiver_t=None,
            source_j=None,
            tensors=_gaussian_from_moment_precision(
                trace.initial_mean,
                trace.complete.initial_covariance,
                precision,
            ),
        )
    ]
    for item in (
        *trace.complete.model_conditionals,
        *trace.complete.state_conditionals,
    ):
        item.__post_init__()
        tensors = {
            **_gaussian_from_moment_precision(
                item.offset, item.covariance, item.precision
            ),
            "parent_map": item.parent_map,
            "offset": item.offset,
        }
        if item.same_receiver_model_map is not None:
            tensors["same_receiver_model_map"] = item.same_receiver_model_map
        components.append(
            _component(
                component_id=item.component_id,
                receiver_t=item.receiver_t,
                source_j=item.source_j,
                tensors=tensors,
            )
        )
    return tuple(components)


def _require_factorized_diagonal(
    components: tuple[H7TensorLawComponent, ...],
) -> None:
    for component in components:
        if "covariance" in component.tensors and (
            not _is_exact_diagonal(component.tensors["covariance"].tensor)
            or not _is_exact_diagonal(component.tensors["precision"].tensor)
        ):
            raise ValueError(
                "factorized H7 origin requires diagonal-within-fiber tensors"
            )


def borrow_h7_recognition(
    law: H7RecognitionInput,
    *,
    context: H7RecognitionContextSnapshot,
) -> H7RecognitionTensorLaw:
    """Borrow one exact live H6 recognition class without cloning it."""

    if type(law) is StructuredLanguageRecognition:
        origin = "structured_full_block"
        representation: H7RecognitionRepresentation = "structured_full_block"
    elif type(law) is FactorizedLanguageRecognition:
        origin = "factorized_diagonal_within_fiber"
        representation = "factorized_diagonal_within_fiber"
    else:
        raise ValueError(
            "law must be an exact StructuredLanguageRecognition or "
            "FactorizedLanguageRecognition"
        )
    if type(context) is not H7RecognitionContextSnapshot:
        raise ValueError("context must be an exact H7RecognitionContextSnapshot")
    context.__post_init__()
    if law.conditioning.horizon != 2 or law.conditioning.mode != context.conditioning:
        raise ValueError("recognition conditioning disagrees with H7 context")
    trace = law.export_h7_trace()
    components = _trace_components(trace, origin=origin)
    _validate_source_context(trace.complete.source_context, dimension=2)
    if type(law) is FactorizedLanguageRecognition:
        _require_factorized_diagonal(components)
    return H7RecognitionTensorLaw.create(
        origin_family=origin,  # type: ignore[arg-type]
        representation=representation,
        components=components,
        source_rows=trace.complete.source_context.scorer_rows,
        context=context,
        scalar_source_law=None,
        jacobian=_zero_jacobian(components, scope="recognition"),
    )


def _transformed_representation(
    law: H7RecognitionTensorLaw,
    elements: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> H7RecognitionRepresentation:
    if law.origin_family == "structured_full_block":
        return "structured_full_block"
    if law.origin_family != "factorized_diagonal_within_fiber":
        raise ValueError("recognition origin family is outside H7")
    if law.representation == "unrestricted_full_block_pushforward":
        return "unrestricted_full_block_pushforward"
    if law.representation != "factorized_diagonal_within_fiber":
        raise ValueError("factorized recognition representation is invalid")
    _require_factorized_diagonal(law.components)
    if all(_is_exact_diagonal(item) for item in elements):
        return "factorized_diagonal_within_fiber"
    return "unrestricted_full_block_pushforward"


def pushforward_h7_recognition(
    law: H7RecognitionTensorLaw,
    action: H7BorrowedActionView,
) -> H7RecognitionTensorLaw:
    """Push every exact source-conditioned component before marginalization."""

    if type(law) is not H7RecognitionTensorLaw:
        raise ValueError("law must be an exact H7RecognitionTensorLaw")
    law.assert_live()
    initial = tuple(
        item
        for item in law.components
        if item.receiver_t is None and item.source_j is None
    )
    if len(initial) != 1:
        raise ValueError("recognition initial component inventory is invalid")
    dimension = initial[0].tensors["mean"].identity.shape[0] // 2
    elements = _action_elements(action, dimension=dimension)
    gaussian_names = set(_GAUSSIAN_TENSOR_NAMES)
    affine_names = gaussian_names | {"parent_map", "offset"}
    components: list[H7TensorLawComponent] = []
    for component in law.components:
        names = set(component.tensors)
        if names == gaussian_names:
            if component.receiver_t is not None or component.source_j is not None:
                raise ValueError("standalone recognition Gaussian is not initial")
            tensors = _push_gaussian(
                component.tensors,
                torch.block_diag(elements[0], elements[0]),
            )
        elif names in (
            affine_names,
            affine_names | {"same_receiver_model_map"},
        ):
            if component.receiver_t is None or component.source_j is None:
                raise ValueError("recognition affine factor lacks endpoints")
            receiver = elements[component.receiver_t]
            source = elements[component.source_j]
            tensors = _push_gaussian(
                {name: component.tensors[name] for name in _GAUSSIAN_TENSOR_NAMES},
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
        else:
            raise ValueError(
                f"unsupported recognition component {component.component_id}"
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
    if law.source_rows:
        prefix_row = max(law.source_rows, key=lambda row: row.receiver_t)
        source_context = H7SourceContextView.create(
            prefix_tokens=prefix_row.prefix_tokens,
            prefix_bytes=prefix_row.prefix_bytes,
            z_history=law.source_rows[0].z_history,
            m_history=law.source_rows[0].m_history,
            scorer_rows=law.source_rows,
            source_scorer_profile="h7-linear-history-source-v1",
        )
        pushed_rows = _push_source_context(source_context, elements).scorer_rows
    else:
        pushed_rows = ()
    return H7RecognitionTensorLaw.create(
        origin_family=law.origin_family,
        representation=_transformed_representation(law, elements),
        components=transformed,
        source_rows=pushed_rows,
        context=law.context,
        scalar_source_law=law.scalar_source_law,
        jacobian=_action_jacobian(transformed, action, scope="recognition"),
    )


def _borrow_h7_recognition_snapshot(
    source: H7RecognitionSnapshot,
) -> H7RecognitionTensorLaw:
    """Private fixture adapter; not evidence for the live-class seam."""

    if type(source) is not H7RecognitionSnapshot:
        raise ValueError("source must be an exact H7RecognitionSnapshot")
    source.__post_init__()
    components = [
        _component(
            component_id=source.initial_joint.component_id,
            receiver_t=None,
            source_j=None,
            tensors=_snapshot_gaussian(source.initial_joint),
        )
    ]
    for item in (
        *source.model_conditionals,
        *source.state_conditionals,
    ):
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
    frozen_components = tuple(components)
    source_rows: tuple[H7SourceScorerRowView, ...] = ()
    if source.source_rows:
        source_context = _source_context_from_recognition_rows(source.source_rows)
        source_rows = source_context.scorer_rows
    return H7RecognitionTensorLaw.create(
        origin_family=source.origin_family,
        representation=source.representation,
        components=frozen_components,
        source_rows=source_rows,
        context=source.context,
        scalar_source_law=source.scalar_source_law,
        jacobian=_zero_jacobian(frozen_components, scope="recognition"),
    )


def _source_context_from_recognition_rows(
    rows: tuple[H7SourceScorerRowSnapshot, ...],
) -> H7SourceContextView:
    if not rows:
        raise ValueError("fixture recognition rows cannot be empty")
    prefix_row = max(rows, key=lambda row: row.receiver_t)
    scorer_sha256 = h7_owned_sha256(
        "vfe4.h7.source-scorer.v1",
        tuple(row.row_sha256 for row in rows),
    )
    snapshot = H7SourceContextSnapshot.create(
        prefix_tokens=prefix_row.prefix_tokens,
        prefix_bytes=prefix_row.prefix_bytes,
        prefix_bytes_sha256=prefix_row.prefix_bytes_sha256,
        z_history=rows[0].z_history,
        m_history=rows[0].m_history,
        scorer_rows=rows,
        source_scorer_profile="h7-linear-history-source-v1",
        source_scorer_sha256=scorer_sha256,
    )
    return _source_context_from_snapshot(snapshot)


def _pushforward_h7_recognition_snapshot(
    source: H7RecognitionSnapshot,
    action: H7BorrowedActionView,
) -> H7RecognitionTensorLaw:
    return pushforward_h7_recognition(_borrow_h7_recognition_snapshot(source), action)


def pushforward_h7_recognition_snapshot(
    source: H7RecognitionSnapshot,
    action: H7BorrowedActionView,
) -> H7RecognitionTensorLaw:
    """Push forward a frozen H7 recognition fixture snapshot."""

    return _pushforward_h7_recognition_snapshot(source, action)


def _freeze_affine(
    component: H7TensorLawComponent,
) -> H7AffineComponentSnapshot:
    required = set(_GAUSSIAN_TENSOR_NAMES) | {"parent_map", "offset"}
    state = "same_receiver_model_map" in component.tensors
    if set(component.tensors) != (
        required | ({"same_receiver_model_map"} if state else set())
    ):
        raise ValueError("recognition affine component is incomplete")
    receiver_component = H7TensorLawComponent.create(
        component_id=f"{component.component_id}.receiver",
        receiver_t=component.receiver_t,
        source_j=component.source_j,
        tensors={name: component.tensors[name] for name in _GAUSSIAN_TENSOR_NAMES},
    )
    return H7AffineComponentSnapshot.create(
        component_id=component.component_id,
        bank=("state" if state else "model"),
        receiver_t=component.receiver_t,
        source_j=component.source_j,
        parent_map=_freeze_tensor(component.tensors["parent_map"]),
        same_receiver_model_map=(
            _freeze_tensor(component.tensors["same_receiver_model_map"])
            if state
            else None
        ),
        offset=_freeze_tensor(component.tensors["offset"]),
        receiver_law=_freeze_gaussian(receiver_component),
    )


def _freeze_rows(
    rows: tuple[H7SourceScorerRowView, ...],
) -> tuple[H7SourceScorerRowSnapshot, ...]:
    if not rows:
        return ()
    prefix_row = max(rows, key=lambda row: row.receiver_t)
    view = H7SourceContextView.create(
        prefix_tokens=prefix_row.prefix_tokens,
        prefix_bytes=prefix_row.prefix_bytes,
        z_history=rows[0].z_history,
        m_history=rows[0].m_history,
        scorer_rows=rows,
        source_scorer_profile="h7-linear-history-source-v1",
    )
    return _freeze_source_context(view).scorer_rows


def freeze_h7_recognition(
    law: H7RecognitionTensorLaw,
) -> H7RecognitionSnapshot:
    """Clone one exact live recognition tensor law into owned evidence."""

    if type(law) is not H7RecognitionTensorLaw:
        raise ValueError("law must be an exact H7RecognitionTensorLaw")
    law.assert_live()
    initial_candidates = tuple(
        item
        for item in law.components
        if item.receiver_t is None and item.source_j is None
    )
    if len(initial_candidates) != 1:
        raise ValueError("recognition initial component inventory is invalid")
    affine = tuple(
        item
        for item in law.components
        if item.receiver_t is not None and item.source_j is not None
    )
    if len(affine) != len(law.components) - 1:
        raise ValueError("recognition tensor law has extra components")
    model = tuple(
        _freeze_affine(item)
        for item in affine
        if "same_receiver_model_map" not in item.tensors
    )
    state = tuple(
        _freeze_affine(item)
        for item in affine
        if "same_receiver_model_map" in item.tensors
    )
    if law.scalar_source_law is None:
        inventory_valid = (
            tuple(item.receiver_t for item in model) == (1, 2)
            and tuple(item.receiver_t for item in state) == (1, 2)
        )
    else:
        inventory_valid = tuple(item.component_id for item in model) == (
            "h1.q.model.1<-0",
            "h1.q.model.2<-0",
            "h1.q.model.2<-1",
        ) and tuple(item.component_id for item in state) == (
            "h1.q.state.1.a_0.b_0.row_0",
            "h1.q.state.2.a_0.b_0.row_0",
            "h1.q.state.2.a_1.b_0.row_1",
            "h1.q.state.2.a_0.b_1.row_2",
            "h1.q.state.2.a_1.b_1.row_3",
        )
    if not inventory_valid:
        raise ValueError("recognition conditional inventory is incomplete")
    return H7RecognitionSnapshot.create(
        origin_family=law.origin_family,
        representation=law.representation,
        initial_joint=_freeze_gaussian(initial_candidates[0]),
        model_conditionals=model,
        state_conditionals=state,
        source_rows=_freeze_rows(law.source_rows),
        context=law.context,
        scalar_source_law=law.scalar_source_law,
        jacobian=_freeze_jacobian(law.jacobian),
    )


__all__ = [
    "H7RecognitionInput",
    "borrow_h7_recognition",
    "freeze_h7_recognition",
    "pushforward_h7_recognition",
    "pushforward_h7_recognition_snapshot",
]
