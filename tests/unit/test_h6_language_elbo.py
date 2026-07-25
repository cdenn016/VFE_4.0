from __future__ import annotations

import hashlib
import importlib
from dataclasses import FrozenInstanceError, dataclass, field

import pytest
import torch

from vfe4.data.windows import CausalPrefix
from vfe4.generative.source_priors import (
    FixedSourceFactorContext,
    FixedSourcePrior,
    MaskCaseKey,
    NormalizedSourceFactor,
    PrefixConditionedSourceFactorContext,
    PrefixConditionedSourcePrior,
)
from vfe4.training.arms import BuiltArm, LatentLanguageArmModel, build_a5
from vfe4.types.h6 import (
    ArmConfig,
    ArmId,
    CapacityAllocation,
    EmissionOnlyAblationTerms,
    FrozenTensorSnapshot,
    H6EndpointLanguageElboTerms,
    H6LanguageElboTerms,
    H6SourcePriorTrace,
    VocabularyIdentity,
)


PARTITIONS = (
    "model_source",
    "model_transition",
    "state_source",
    "state_transition",
    "emission",
    "entropy",
)


def _objective_api() -> tuple[object, object, object, object]:
    module = importlib.import_module("vfe4.objective")
    required = (
        "LanguageElboExpectation",
        "ExactSourceMixtureLaw",
        "MomentProjectedLaw",
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
    endpoint_config: ArmConfig
    arm: BuiltArm
    source_prior: FixedSourcePrior | PrefixConditionedSourcePrior
    source_factors: dict[tuple[str, int], NormalizedSourceFactor]
    source_contexts: dict[
        tuple[str, int],
        FixedSourceFactorContext | PrefixConditionedSourceFactorContext,
    ]
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
        if partition in ("model_source", "state_source"):
            return self.source_factors[
                (partition, receiver_t)
            ].factor_identity_sha256
        return hashlib.sha256(f"{partition}:{receiver_t}".encode()).hexdigest()

    def normalized_source_factor(
        self, partition: str, receiver_t: int
    ) -> NormalizedSourceFactor:
        if partition not in ("model_source", "state_source"):
            raise KeyError((partition, receiver_t))
        return self.source_factors[(partition, receiver_t)]

    def source_factor_context(
        self, partition: str, receiver_t: int
    ) -> FixedSourceFactorContext | PrefixConditionedSourceFactorContext:
        if partition not in ("model_source", "state_source"):
            raise KeyError((partition, receiver_t))
        return self.source_contexts[(partition, receiver_t)]

    def independently_accumulated_total(self) -> torch.Tensor:
        return self.independent_value


def _endpoint_config(
    *,
    horizon: int,
    prior_variant: str = "fixed",
    mixture_mode: str = "exact",
) -> ArmConfig:
    if prior_variant == "prefix_conditioned":
        config_id = (
            "h6-a5-structured-prefix-exact-complete-latent-smoothing-v1"
        )
    elif mixture_mode == "moment_projection":
        config_id = (
            "h6-a5-structured-fixed-projection-complete-"
            "latent-smoothing-v1"
        )
    else:
        config_id = (
            "h6-a5-structured-fixed-exact-complete-latent-smoothing-v1"
        )
    return ArmConfig.create(
        arm=ArmId.A5,
        config_id=config_id,
        vocabulary=VocabularyIdentity(
            vocabulary_id="h6-objective-test-v1",
            size=258,
            tokenizer_spec_sha256="9" * 64,
        ),
        horizon=horizon,
        latent_enabled=True,
        state_channel_enabled=True,
        model_channel_enabled=True,
        source_mode="categorical",
        map_mode="shared_vertex_coboundary",
        recognition_family="structured",
        recognition_conditioning="smoothing",
        prior_variant=prior_variant,
        mixture_mode=mixture_mode,
        objective_kind="complete_elbo",
        capacity_allocation=CapacityAllocation.create(
            emission_width=4,
            latent_width=2,
            recognition_width=4,
            prior_context_width=(
                2 if prior_variant == "prefix_conditioned" else None
            ),
        ),
    )


def _source_factors_and_contexts(
    config: ArmConfig,
    prior: FixedSourcePrior | PrefixConditionedSourcePrior,
) -> tuple[
    dict[tuple[str, int], NormalizedSourceFactor],
    dict[
        tuple[str, int],
        FixedSourceFactorContext | PrefixConditionedSourceFactorContext,
    ],
]:
    factors: dict[tuple[str, int], NormalizedSourceFactor] = {}
    contexts: dict[
        tuple[str, int],
        FixedSourceFactorContext | PrefixConditionedSourceFactorContext,
    ] = {}
    for receiver_t in range(1, config.horizon + 1):
        if type(prior) is FixedSourcePrior:
            model_factor = prior.model_source_log_probs(
                receiver_t=receiver_t
            )
            state_factor = prior.state_source_log_probs(
                receiver_t=receiver_t
            )
            model_context: (
                FixedSourceFactorContext
                | PrefixConditionedSourceFactorContext
            ) = FixedSourceFactorContext("model", receiver_t)
            state_context: (
                FixedSourceFactorContext
                | PrefixConditionedSourceFactorContext
            ) = FixedSourceFactorContext("state", receiver_t)
        else:
            prefix = CausalPrefix.create(
                receiver_t=receiver_t,
                vocabulary=config.vocabulary,
                token_ids=torch.zeros(
                    receiver_t - 1,
                    dtype=torch.int64,
                ),
            )
            earlier_latents = torch.zeros(
                receiver_t,
                prior.latent_dim,
                dtype=torch.float64,
            )
            model_factor = prior.model_source_log_probs(
                prefix=prefix,
                earlier_latents=earlier_latents,
            )
            state_factor = prior.state_source_log_probs(
                prefix=prefix,
                earlier_latents=earlier_latents,
            )
            model_context = PrefixConditionedSourceFactorContext(
                "model",
                receiver_t,
                prefix,
                earlier_latents,
            )
            state_context = PrefixConditionedSourceFactorContext(
                "state",
                receiver_t,
                prefix,
                earlier_latents,
            )
        factors[("model_source", receiver_t)] = model_factor
        factors[("state_source", receiver_t)] = state_factor
        contexts[("model_source", receiver_t)] = model_context
        contexts[("state_source", receiver_t)] = state_context
    return factors, contexts


def _rekey_source_factor(
    factor: NormalizedSourceFactor,
    *,
    predictor_config_sha256: str | None = None,
    model_family_sha256: str | None = None,
    prior_variant: str | None = None,
    bank: str | None = None,
) -> NormalizedSourceFactor:
    original = factor.mask_case_key
    key = MaskCaseKey(
        fixture_sha256=original.fixture_sha256,
        vocabulary_sha256=original.vocabulary_sha256,
        predictor_config_sha256=(
            predictor_config_sha256
            if predictor_config_sha256 is not None
            else original.predictor_config_sha256
        ),
        model_family_sha256=(
            model_family_sha256
            if model_family_sha256 is not None
            else original.model_family_sha256
        ),
        prior_variant=(
            prior_variant
            if prior_variant is not None
            else original.prior_variant
        ),  # type: ignore[arg-type]
        bank=bank if bank is not None else original.bank,  # type: ignore[arg-type]
        receiver_t=original.receiver_t,
        context_sha256=original.context_sha256,
    )
    return NormalizedSourceFactor.create(
        key=key,
        log_probs=factor.log_probs.value(),
        support_mask=torch.tensor(
            factor.support_mask,
            dtype=torch.bool,
        ),
    )


def _evaluate(expectation: SyntheticExpectation) -> object:
    return expectation.arm.evaluate_complete_language_elbo(expectation)


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
    endpoint_config: ArmConfig | None = None,
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
    resolved_config = endpoint_config or _endpoint_config(
        horizon=horizon,
        mixture_mode="moment_projection" if projected else "exact",
    )
    assert resolved_config.horizon == horizon
    if source_law is None:
        if projected:
            source_law = objective.MomentProjectedLaw.create(
                endpoint_config=resolved_config,
                projection_error=projection_error,
            )
        else:
            source_law = objective.ExactSourceMixtureLaw.create(
                endpoint_config=resolved_config,
            )
    arm = build_a5(resolved_config)
    assert type(arm.model) is LatentLanguageArmModel
    source_prior = arm.model.source_prior
    assert type(source_prior) in (
        FixedSourcePrior,
        PrefixConditionedSourcePrior,
    )
    source_factors, source_contexts = _source_factors_and_contexts(
        resolved_config,
        source_prior,
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
            endpoint_config=resolved_config,
            arm=arm,
            source_prior=source_prior,
            source_factors=source_factors,
            source_contexts=source_contexts,
            values=values,
            independent_value=total,
        ),
        leaves,
    )


def test_source_prior_trace_is_issued_only_from_matching_exact_objects() -> None:
    expectation, _ = _expectation(horizon=2)
    ordered = tuple(
        expectation.normalized_source_factor(partition, receiver_t)
        for receiver_t in range(1, expectation.horizon + 1)
        for partition in ("model_source", "state_source")
    )
    with pytest.raises(TypeError, match="BuiltArm-only"):
        H6SourcePriorTrace()

    base = ordered[0]
    mismatched_records = (
        _rekey_source_factor(
            base,
            predictor_config_sha256=hashlib.sha256(
                b"another-endpoint-config"
            ).hexdigest(),
        ),
        _rekey_source_factor(
            base,
            model_family_sha256=hashlib.sha256(
                b"another-model-family"
            ).hexdigest(),
        ),
        _rekey_source_factor(
            base,
            prior_variant="prefix_conditioned",
        ),
        _rekey_source_factor(
            base,
            bank="state",
        ),
    )
    for mismatched in mismatched_records:
        replaced = (mismatched, *ordered[1:])
        with pytest.raises(ValueError, match="mismatch|does not match"):
            H6SourcePriorTrace._from_live_prior(
                endpoint_config=expectation.endpoint_config,
                source_prior=expectation.source_prior,
                ordered_source_factors=replaced,
            )

    wrong_receiver_order = (
        ordered[2],
        ordered[1],
        ordered[0],
        ordered[3],
    )
    with pytest.raises(ValueError, match="receiver|order"):
        H6SourcePriorTrace._from_live_prior(
            endpoint_config=expectation.endpoint_config,
            source_prior=expectation.source_prior,
            ordered_source_factors=wrong_receiver_order,
        )


def test_built_arm_recomputes_source_factors_from_exact_typed_contexts() -> None:
    fixed, _ = _expectation(horizon=2)
    fixed.source_contexts[("model_source", 1)] = FixedSourceFactorContext(
        "state",
        1,
    )
    with pytest.raises(ValueError, match="bank|receiver"):
        _evaluate(fixed)

    prefix_config = _endpoint_config(
        horizon=2,
        prior_variant="prefix_conditioned",
    )
    prefix, _ = _expectation(
        horizon=2,
        endpoint_config=prefix_config,
    )
    context = prefix.source_factor_context("model_source", 1)
    assert type(context) is PrefixConditionedSourceFactorContext
    prefix.source_contexts[("model_source", 1)] = (
        PrefixConditionedSourceFactorContext(
            "model",
            1,
            context.prefix,
            context.earlier_latents + 1.0,
        )
    )
    with pytest.raises(ValueError, match="recomputation"):
        _evaluate(prefix)


def test_language_elbo_assembles_the_exact_canonical_horizon_and_stays_live() -> None:
    Protocol, _, _, _ = _objective_api()
    expectation, leaves = _expectation()
    assert isinstance(expectation, Protocol)

    result = _evaluate(expectation)

    assert type(result) is H6EndpointLanguageElboTerms
    assert type(result.terms) is H6LanguageElboTerms
    assert (
        result.source_law_marker_identity_sha256
        == expectation.source_law.law_identity_sha256
    )
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
    graph_result = _evaluate(graph_check)
    graph_result.total_language_elbo.value().backward()
    assert all(leaf.grad is not None for leaf in graph_leaves)
    assert independent_leaf.grad is None


def test_expectation_method_and_projection_treatment_are_explicit_and_distinct(
) -> None:
    _, ExactLaw, ProjectedLaw, _ = _objective_api()
    exact, _ = _expectation(evaluation_method="deterministic_quadrature")
    projected, _ = _expectation(
        evaluation_method="reparameterized_mc",
        projected=True,
        projection_error=FrozenTensorSnapshot.capture(
            torch.tensor(0.01, dtype=torch.float64)
        ),
    )
    exact_result = _evaluate(exact)
    projected_result = _evaluate(projected)
    assert not hasattr(
        importlib.import_module("vfe4.objective"),
        "evaluate_language_elbo",
    )
    trace = exact_result.source_prior_trace
    forged_trace = object.__new__(H6SourcePriorTrace)
    for name in (
        "endpoint_config",
        "model_family_sha256",
        "prior_type",
        "prior_model_state_sha256",
        "ordered_source_factor_identities",
        "trace_sha256",
    ):
        object.__setattr__(forged_trace, name, getattr(trace, name))
    object.__setattr__(
        forged_trace,
        "prior_variant",
        "prefix_conditioned",
    )
    with pytest.raises(ValueError, match="relabels"):
        forged_trace.__post_init__()
    forged_law = object.__new__(type(exact.source_law))
    object.__setattr__(
        forged_law, "endpoint_config", exact.source_law.endpoint_config
    )
    object.__setattr__(
        forged_law,
        "law_identity_sha256",
        "f" * 64,
    )
    with pytest.raises(ValueError, match="identity"):
        forged_law.__post_init__()
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
    filtering_result = _evaluate(filtering)
    assert tuple(
        term.factor_identity_sha256
        for term in exact_result.ordered_factor_terms
    ) != tuple(
        term.factor_identity_sha256
        for term in filtering_result.ordered_factor_terms
    )

    unspecified, _ = _expectation(evaluation_method="unspecified")
    with pytest.raises(ValueError, match="evaluation_method"):
        _evaluate(unspecified)
    with pytest.raises(TypeError, match="create"):
        ExactLaw(
            endpoint_config=exact.endpoint_config,
        )
    with pytest.raises(TypeError, match="create"):
        ProjectedLaw(
            endpoint_config=projected.endpoint_config,
            projection_error=FrozenTensorSnapshot.capture(
                torch.tensor(0.0, dtype=torch.float64)
            ),
        )
    with pytest.raises(TypeError, match="sealed"):
        type("RelabeledExactLaw", (ExactLaw,), {})
    exact_law = ExactLaw.create(
        endpoint_config=exact.endpoint_config,
    )
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        setattr(exact_law, "source_treatment", "moment_projection")
    relabeled, _ = _expectation(source_law="moment_projection")
    with pytest.raises(ValueError, match="source_law"):
        _evaluate(relabeled)

    changed, _ = _expectation(evaluation_method="deterministic_quadrature")
    changed.values[("emission", 1)] = torch.tensor(
        99.0, dtype=torch.float64, requires_grad=True
    )
    changed.independent_value = changed.values[changed.ordered_slots[0]]
    for slot in changed.ordered_slots[1:]:
        changed.independent_value = changed.independent_value + changed.values[slot]
    changed_result = _evaluate(changed)
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
    wrong_total, _ = _expectation(total_offset=1.0)
    with pytest.raises(ValueError, match="does not equal"):
        _evaluate(wrong_total)

    detached_total, _ = _expectation()
    detached_total.independent_value = detached_total.independent_value.detach()
    with pytest.raises(ValueError, match="detached"):
        _evaluate(detached_total)

    vector, _ = _expectation()
    vector.values[("emission", 1)] = torch.ones(2, dtype=torch.float64)
    with pytest.raises(ValueError, match="scalar"):
        _evaluate(vector)

    extra, _ = _expectation()
    extra.ordered_slots = extra.ordered_slots + (("emission", 3),)
    duplicate, _ = _expectation()
    duplicate.ordered_slots = duplicate.ordered_slots + (("emission", 1),)
    for invalid_inventory in (extra, duplicate):
        with pytest.raises(ValueError, match="ordered_slots"):
            _evaluate(invalid_inventory)

    with pytest.raises(ValueError, match="LanguageElboExpectation"):
        trace_expectation, _ = _expectation()
        trace_expectation.arm.evaluate_complete_language_elbo(
            object(),  # type: ignore[arg-type]
        )


def test_emission_only_is_a_distinct_non_elbo_record() -> None:
    _, _, _, evaluate_ablation = _objective_api()
    expectation, _ = _expectation()

    result = evaluate_ablation(expectation)

    assert type(result) is EmissionOnlyAblationTerms
    assert result.objective_kind == "emission_only_ablation_non_elbo"
    assert tuple(
        (term.partition, term.receiver_t) for term in result.ordered_emission_terms
    ) == (("emission", 1), ("emission", 2))


def test_private_snapshot_mutation_fails_before_elbo_reuse() -> None:
    expectation, _ = _expectation(horizon=1)
    result = _evaluate(expectation)
    snapshot = result.ordered_factor_terms[0].value
    private = getattr(snapshot, "_FrozenTensorSnapshot__owned")
    with torch.no_grad():
        private.add_(1.0)
    with pytest.raises(ValueError, match="integrity"):
        result.__post_init__()
