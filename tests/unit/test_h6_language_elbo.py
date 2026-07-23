from __future__ import annotations

import hashlib
import importlib
from dataclasses import FrozenInstanceError, dataclass, field

import pytest
import torch

from vfe4.types.h6 import (
    EmissionOnlyAblationTerms,
    FrozenTensorSnapshot,
    H6LanguageElboTerms,
)


PARTITIONS = (
    "model_source",
    "model_transition",
    "state_source",
    "state_transition",
    "emission",
    "entropy",
)


def _objective_api() -> tuple[object, object, object, object, object]:
    module = importlib.import_module("vfe4.objective")
    required = (
        "LanguageElboExpectation",
        "ExactSourceMixtureLaw",
        "MomentProjectedLaw",
        "evaluate_language_elbo",
        "evaluate_emission_only_ablation",
    )
    missing = tuple(name for name in required if not hasattr(module, name))
    assert not missing, f"missing H6 language objective API: {missing}"
    return tuple(  # type: ignore[return-value]
        getattr(module, name) for name in required
    )


def _slots(horizon: int) -> tuple[tuple[str, int], ...]:
    return (("initial", 0),) + tuple(
        (partition, receiver_t)
        for receiver_t in range(1, horizon + 1)
        for partition in PARTITIONS
    )


@dataclass
class SyntheticExpectation:
    horizon: int
    evaluation_method: str
    expectation_identity_sha256: str
    structure_sha256: str
    recognition_family: str
    recognition_conditioning: str
    ordered_slots: tuple[tuple[str, int], ...]
    source_law: object
    values: dict[tuple[str, int], torch.Tensor]
    independent_value: torch.Tensor
    calls: list[tuple[str, int]] = field(default_factory=list)

    def contribution(self, partition: str, receiver_t: int) -> torch.Tensor:
        key = (partition, receiver_t)
        self.calls.append(key)
        return self.values[key]

    def normalized_factor_identity(
        self, partition: str, receiver_t: int
    ) -> str:
        return hashlib.sha256(f"{partition}:{receiver_t}".encode()).hexdigest()

    def independently_accumulated_total(self) -> torch.Tensor:
        return self.independent_value


def _expectation(
    *,
    horizon: int = 2,
    evaluation_method: str = "exact_enumeration",
    recognition_family: str = "structured_full_spd",
    recognition_conditioning: str = "smoothing",
    total_offset: float = 0.0,
    source_law: object | None = None,
    projected: bool = False,
    projection_error: FrozenTensorSnapshot | None = None,
) -> tuple[SyntheticExpectation, tuple[torch.Tensor, ...]]:
    slots = _slots(horizon)
    leaves = tuple(
        torch.tensor(float(index + 1), dtype=torch.float64, requires_grad=True)
        for index in range(len(slots))
    )
    values = dict(zip(slots, leaves, strict=True))
    total = leaves[0]
    for value in leaves[1:]:
        total = total + value
    if total_offset:
        total = total + torch.tensor(total_offset, dtype=torch.float64)
    objective = importlib.import_module("vfe4.objective")
    if source_law is None:
        if projected:
            source_law = objective.MomentProjectedLaw.create(
                law_identity_sha256="d" * 64,
                projection_error=projection_error,
            )
        else:
            source_law = objective.ExactSourceMixtureLaw.create(
                law_identity_sha256="c" * 64
            )
    return (
        SyntheticExpectation(
            horizon=horizon,
            evaluation_method=evaluation_method,
            expectation_identity_sha256="a" * 64,
            structure_sha256="b" * 64,
            recognition_family=recognition_family,
            recognition_conditioning=recognition_conditioning,
            ordered_slots=slots,
            source_law=source_law,
            values=values,
            independent_value=total,
        ),
        leaves,
    )


def test_language_elbo_assembles_the_exact_canonical_horizon_and_stays_live() -> None:
    Protocol, _, _, evaluate, _ = _objective_api()
    expectation, leaves = _expectation()
    assert isinstance(expectation, Protocol)

    result = evaluate(expectation)

    assert type(result) is H6LanguageElboTerms
    expected_slots = _slots(2)
    assert tuple(
        (term.partition, term.receiver_t) for term in result.ordered_factor_terms
    ) == expected_slots
    assert expectation.calls == list(expected_slots)
    assert len(result.ordered_factor_terms) == 13
    assert all(
        type(term.value) is FrozenTensorSnapshot
        for term in result.ordered_factor_terms
    )
    assert all(
        getattr(result, f"{partition}_terms")
        for partition in (
            "emission",
            "initial",
            "state_source",
            "model_source",
            "state_transition",
            "model_transition",
            "entropy",
        )
    )
    assert torch.equal(
        result.complete_decomposition.value(), result.total_language_elbo.value()
    )

    result.total_language_elbo.value().backward()
    assert all(leaf.grad is not None for leaf in leaves)

    original_first = result.ordered_factor_terms[0].value.value().item()
    with torch.no_grad():
        leaves[0].add_(1000.0)
    returned = result.ordered_factor_terms[0].value.value()
    returned.add_(2000.0)
    assert result.ordered_factor_terms[0].value.value().item() == original_first

    graph_check, graph_leaves = _expectation()
    independent_leaf = torch.tensor(
        graph_check.independent_value.item(), dtype=torch.float64, requires_grad=True
    )
    graph_check.independent_value = independent_leaf
    graph_result = evaluate(graph_check)
    graph_result.total_language_elbo.value().backward()
    assert all(leaf.grad is not None for leaf in graph_leaves)
    assert independent_leaf.grad is None


def test_expectation_method_and_projection_treatment_are_explicit_and_distinct(
) -> None:
    _, ExactLaw, ProjectedLaw, evaluate, _ = _objective_api()
    exact, _ = _expectation(evaluation_method="deterministic_quadrature")
    projected, _ = _expectation(
        evaluation_method="reparameterized_mc",
        projected=True,
        projection_error=FrozenTensorSnapshot.capture(
            torch.tensor(0.01, dtype=torch.float64)
        ),
    )
    exact_result = evaluate(exact)
    projected_result = evaluate(projected)
    assert tuple(
        term.factor_identity_sha256
        for term in exact_result.ordered_factor_terms
    ) != tuple(
        term.factor_identity_sha256
        for term in projected_result.ordered_factor_terms
    )
    filtering, _ = _expectation(
        evaluation_method="deterministic_quadrature",
        recognition_conditioning="filtering",
    )
    filtering_result = evaluate(filtering)
    assert tuple(
        term.factor_identity_sha256
        for term in exact_result.ordered_factor_terms
    ) != tuple(
        term.factor_identity_sha256
        for term in filtering_result.ordered_factor_terms
    )

    unspecified, _ = _expectation(evaluation_method="unspecified")
    with pytest.raises(ValueError, match="evaluation_method"):
        evaluate(unspecified)
    with pytest.raises(TypeError, match="create"):
        ExactLaw(law_identity_sha256="c" * 64)
    with pytest.raises(TypeError, match="create"):
        ProjectedLaw(
            law_identity_sha256="d" * 64,
            projection_error=FrozenTensorSnapshot.capture(
                torch.tensor(0.0, dtype=torch.float64)
            ),
        )
    with pytest.raises(TypeError, match="sealed"):
        type("RelabeledExactLaw", (ExactLaw,), {})
    exact_law = ExactLaw.create(law_identity_sha256="c" * 64)
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        setattr(exact_law, "source_treatment", "moment_projection")
    relabeled, _ = _expectation(source_law="moment_projection")
    with pytest.raises(ValueError, match="source_law"):
        evaluate(relabeled)

    changed, _ = _expectation(evaluation_method="deterministic_quadrature")
    changed.values[("emission", 1)] = torch.tensor(
        99.0, dtype=torch.float64, requires_grad=True
    )
    changed.independent_value = changed.values[changed.ordered_slots[0]]
    for slot in changed.ordered_slots[1:]:
        changed.independent_value = changed.independent_value + changed.values[slot]
    changed_result = evaluate(changed)
    exact_emission = exact_result.emission_terms[0]
    changed_emission = changed_result.emission_terms[0]
    assert (
        exact_emission.value.dtype,
        exact_emission.value.shape,
        exact_emission.value.device,
        exact_emission.value.requires_grad,
        exact_emission.value.storage_version,
    ) == (
        changed_emission.value.dtype,
        changed_emission.value.shape,
        changed_emission.value.device,
        changed_emission.value.requires_grad,
        changed_emission.value.storage_version,
    )
    assert (
        exact_emission.value.raw_bytes_sha256
        != changed_emission.value.raw_bytes_sha256
    )
    assert (
        exact_emission.factor_identity_sha256
        != changed_emission.factor_identity_sha256
    )


def test_language_elbo_rejects_wrong_totals_non_scalars_and_non_protocol_inputs(
) -> None:
    _, _, _, evaluate, _ = _objective_api()
    wrong_total, _ = _expectation(total_offset=1.0)
    with pytest.raises(ValueError, match="does not equal"):
        evaluate(wrong_total)

    detached_total, _ = _expectation()
    detached_total.independent_value = detached_total.independent_value.detach()
    with pytest.raises(ValueError, match="detached"):
        evaluate(detached_total)

    vector, _ = _expectation()
    vector.values[("emission", 1)] = torch.ones(2, dtype=torch.float64)
    with pytest.raises(ValueError, match="scalar"):
        evaluate(vector)

    extra, _ = _expectation()
    extra.ordered_slots = extra.ordered_slots + (("emission", 3),)
    duplicate, _ = _expectation()
    duplicate.ordered_slots = duplicate.ordered_slots + (("emission", 1),)
    for invalid_inventory in (extra, duplicate):
        with pytest.raises(ValueError, match="ordered_slots"):
            evaluate(invalid_inventory)

    with pytest.raises(ValueError, match="LanguageElboExpectation"):
        evaluate(object())


def test_emission_only_is_a_distinct_non_elbo_record() -> None:
    _, _, _, _, evaluate_ablation = _objective_api()
    expectation, _ = _expectation()

    result = evaluate_ablation(expectation)

    assert type(result) is EmissionOnlyAblationTerms
    assert result.objective_kind == "emission_only_ablation_non_elbo"
    assert tuple(
        (term.partition, term.receiver_t) for term in result.ordered_emission_terms
    ) == (("emission", 1), ("emission", 2))


def test_private_snapshot_mutation_fails_before_elbo_reuse() -> None:
    _, _, _, evaluate, _ = _objective_api()
    expectation, _ = _expectation(horizon=1)
    result = evaluate(expectation)
    snapshot = result.ordered_factor_terms[0].value
    private = getattr(snapshot, "_FrozenTensorSnapshot__owned")
    with torch.no_grad():
        private.add_(1.0)
    with pytest.raises(ValueError, match="integrity"):
        result.__post_init__()
