"""Canonical differentiable H6 language-ELBO assembly."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Literal, Protocol, final, runtime_checkable

import torch
from torch import Tensor

from vfe4.generative.source_priors import (
    FixedSourceFactorContext,
    NormalizedSourceFactor,
    PrefixConditionedSourceFactorContext,
)
from vfe4.types.h6 import (
    ArmConfig,
    ArmId,
    EmissionOnlyAblationTerms,
    FrozenTensorSnapshot,
    H6EndpointLanguageElboTerms,
    H6FactorTerm,
    H6LanguageElboTerms,
    H6SourcePriorTrace,
    canonical_json_bytes,
    h6_source_law_marker_identity,
)


FactorPartition = Literal[
    "emission",
    "initial",
    "state_source",
    "model_source",
    "state_transition",
    "model_transition",
    "entropy",
]
ExpectationEvaluationMethod = Literal[
    "exact_enumeration",
    "deterministic_quadrature",
    "reparameterized_mc",
]
RecognitionFamily = Literal[
    "structured_full_spd", "population_factorized_block_spd"
]
RecognitionConditioningMode = Literal["filtering", "smoothing"]
MixtureMode = Literal["exact", "moment_projection"]
PriorVariant = Literal["fixed", "prefix_conditioned"]

_EVALUATION_METHODS = {
    "exact_enumeration",
    "deterministic_quadrature",
    "reparameterized_mc",
}
_RECOGNITION_FAMILIES = {
    "structured_full_spd",
    "population_factorized_block_spd",
}
_RECOGNITION_CONDITIONING = {"filtering", "smoothing"}
_PER_RECEIVER_PARTITIONS: tuple[FactorPartition, ...] = (
    "model_source",
    "model_transition",
    "state_source",
    "state_transition",
    "emission",
    "entropy",
)
_TERM_IDENTITY_DOMAIN = b"vfe4.h6.language-elbo-factor.v1\x00"
_H7_COMPLETE_TRACE_DOMAIN = b"vfe4.h7.complete-language-elbo-factor-trace.v1\x00"


@final
@dataclass(frozen=True, slots=True, init=False)
class ExactSourceMixtureLaw:
    """Factory-only marker for exact source-mixture expectations."""

    endpoint_config: ArmConfig
    law_identity_sha256: str

    def __init__(
        self,
        *,
        endpoint_config: ArmConfig,
    ) -> None:
        raise TypeError(
            "ExactSourceMixtureLaw is factory-only; use "
            "ExactSourceMixtureLaw.create"
        )

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("ExactSourceMixtureLaw is sealed")

    def __post_init__(self) -> None:
        _validate_source_endpoint(self.endpoint_config)
        if self.endpoint_config.mixture_mode != "exact":
            raise ValueError(
                "ExactSourceMixtureLaw requires an exact endpoint"
            )
        expected = h6_source_law_marker_identity(
            endpoint_config=self.endpoint_config,
            projection_error=None,
        )
        if self.law_identity_sha256 != expected:
            raise ValueError("exact source-law identity is stale")

    @property
    def prior_variant(self) -> PriorVariant:
        return self.endpoint_config.prior_variant  # type: ignore[return-value]

    @classmethod
    def create(
        cls,
        *,
        endpoint_config: ArmConfig,
    ) -> "ExactSourceMixtureLaw":
        instance = object.__new__(cls)
        object.__setattr__(instance, "endpoint_config", endpoint_config)
        object.__setattr__(
            instance,
            "law_identity_sha256",
            h6_source_law_marker_identity(
                endpoint_config=endpoint_config,
                projection_error=None,
            ),
        )
        instance.__post_init__()
        return instance


@final
@dataclass(frozen=True, slots=True, init=False)
class MomentProjectedLaw:
    """Factory-only marker carrying a measured moment-projection error."""

    endpoint_config: ArmConfig
    law_identity_sha256: str
    projection_error: FrozenTensorSnapshot

    def __init__(
        self,
        *,
        endpoint_config: ArmConfig,
        projection_error: FrozenTensorSnapshot,
    ) -> None:
        raise TypeError(
            "MomentProjectedLaw is factory-only; use MomentProjectedLaw.create"
        )

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("MomentProjectedLaw is sealed")

    def __post_init__(self) -> None:
        _validate_source_endpoint(self.endpoint_config)
        if self.endpoint_config.mixture_mode != "moment_projection":
            raise ValueError(
                "MomentProjectedLaw requires a moment_projection endpoint"
            )
        if type(self.projection_error) is not FrozenTensorSnapshot:
            raise ValueError(
                "MomentProjectedLaw requires a FrozenTensorSnapshot projection_error"
            )
        self.projection_error.assert_intact()
        error = _require_live_scalar(
            self.projection_error.value(), "projection_error"
        )
        if not bool(error >= 0.0):
            raise ValueError("projection_error must be nonnegative")
        expected = h6_source_law_marker_identity(
            endpoint_config=self.endpoint_config,
            projection_error=self.projection_error,
        )
        if self.law_identity_sha256 != expected:
            raise ValueError("moment-projected source-law identity is stale")

    @property
    def prior_variant(self) -> PriorVariant:
        return self.endpoint_config.prior_variant  # type: ignore[return-value]

    @classmethod
    def create(
        cls,
        *,
        endpoint_config: ArmConfig,
        projection_error: FrozenTensorSnapshot,
    ) -> "MomentProjectedLaw":
        instance = object.__new__(cls)
        object.__setattr__(instance, "endpoint_config", endpoint_config)
        object.__setattr__(instance, "projection_error", projection_error)
        object.__setattr__(
            instance,
            "law_identity_sha256",
            h6_source_law_marker_identity(
                endpoint_config=endpoint_config,
                projection_error=projection_error,
            ),
        )
        instance.__post_init__()
        return instance


@runtime_checkable
class LanguageElboExpectation(Protocol):
    """Explicit provider of one evaluated, differentiable ELBO estimator.

    ``evaluation_method`` declares how nonlinear expectations were obtained;
    the assembler verifies bookkeeping and never silently chooses an estimator.
    """

    horizon: int
    evaluation_method: ExpectationEvaluationMethod
    source_law: ExactSourceMixtureLaw | MomentProjectedLaw
    expectation_identity_sha256: str
    structure_sha256: str
    recognition_family: RecognitionFamily
    recognition_conditioning: RecognitionConditioningMode
    ordered_slots: tuple[tuple[FactorPartition, int], ...]

    def contribution(
        self, partition: FactorPartition, receiver_t: int
    ) -> Tensor: ...

    def normalized_factor_identity(
        self, partition: FactorPartition, receiver_t: int
    ) -> str: ...

    def normalized_source_factor(
        self,
        partition: Literal["state_source", "model_source"],
        receiver_t: int,
    ) -> NormalizedSourceFactor: ...

    def source_factor_context(
        self,
        partition: Literal["state_source", "model_source"],
        receiver_t: int,
    ) -> FixedSourceFactorContext | PrefixConditionedSourceFactorContext: ...

    def independently_accumulated_total(self) -> Tensor: ...


def _canonical_slots(horizon: int) -> tuple[tuple[FactorPartition, int], ...]:
    return (("initial", 0),) + tuple(
        (partition, receiver_t)
        for receiver_t in range(1, horizon + 1)
        for partition in _PER_RECEIVER_PARTITIONS
    )


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_live_scalar(value: object, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if value.ndim != 0:
        raise ValueError(f"{name} must be a scalar tensor")
    if not value.is_floating_point():
        raise ValueError(f"{name} must be floating point")
    if not bool(torch.isfinite(value)):
        raise ValueError(f"{name} must be finite")
    return value


def _snapshot_payload(snapshot: FrozenTensorSnapshot) -> dict[str, object]:
    snapshot.assert_intact()
    return {
        "dtype": snapshot.dtype,
        "shape": snapshot.shape,
        "device": snapshot.device,
        "contiguous": snapshot.contiguous,
        "requires_grad": snapshot.requires_grad,
        "storage_version": snapshot.storage_version,
        "raw_bytes_sha256": snapshot.raw_bytes_sha256,
    }


def _validate_source_endpoint(endpoint_config: object) -> ArmConfig:
    if type(endpoint_config) is not ArmConfig:
        raise ValueError("source law requires an exact ArmConfig")
    endpoint_config.__post_init__()
    if (
        endpoint_config.arm not in (ArmId.A2, ArmId.A5)
        or endpoint_config.objective_kind != "complete_elbo"
        or endpoint_config.prior_variant not in ("fixed", "prefix_conditioned")
        or endpoint_config.mixture_mode not in ("exact", "moment_projection")
    ):
        raise ValueError(
            "source law requires a complete categorical A2/A5 endpoint"
        )
    return endpoint_config


def _source_law_payload(
    source_law: ExactSourceMixtureLaw | MomentProjectedLaw,
    source_prior_trace: H6SourcePriorTrace | None = None,
) -> dict[str, object]:
    trace_sha256: str | None = None
    if source_prior_trace is not None:
        if type(source_prior_trace) is not H6SourcePriorTrace:
            raise ValueError("source_prior_trace must be exact")
        source_prior_trace.__post_init__()
        if source_prior_trace.endpoint_config != source_law.endpoint_config:
            raise ValueError("source-prior trace belongs to another source law")
        trace_sha256 = source_prior_trace.trace_sha256
    if type(source_law) is ExactSourceMixtureLaw:
        source_law.__post_init__()
        return {
            "kind": "exact_source_mixture",
            "endpoint_config_sha256": (
                source_law.endpoint_config.config_sha256
            ),
            "prior_variant": source_law.prior_variant,
            "mixture_mode": source_law.endpoint_config.mixture_mode,
            "source_prior_trace_sha256": trace_sha256,
            "source_law_marker_identity_sha256": (
                source_law.law_identity_sha256
            ),
            "projection_error": None,
        }
    if type(source_law) is MomentProjectedLaw:
        source_law.__post_init__()
        return {
            "kind": "moment_projection",
            "endpoint_config_sha256": (
                source_law.endpoint_config.config_sha256
            ),
            "prior_variant": source_law.prior_variant,
            "mixture_mode": source_law.endpoint_config.mixture_mode,
            "source_prior_trace_sha256": trace_sha256,
            "source_law_marker_identity_sha256": (
                source_law.law_identity_sha256
            ),
            "projection_error": _snapshot_payload(source_law.projection_error),
        }
    raise ValueError("unsupported expectation source_law")


def _validate_expectation(
    expectation: object,
) -> LanguageElboExpectation:
    if not isinstance(expectation, LanguageElboExpectation):
        raise ValueError("expectation must implement LanguageElboExpectation")
    if type(expectation.horizon) is not int or expectation.horizon <= 0:
        raise ValueError("expectation horizon must be a positive integer")
    if expectation.evaluation_method not in _EVALUATION_METHODS:
        raise ValueError("unsupported expectation evaluation_method")
    _source_law_payload(expectation.source_law)
    _require_sha256(
        expectation.expectation_identity_sha256,
        "expectation_identity_sha256",
    )
    _require_sha256(expectation.structure_sha256, "structure_sha256")
    if expectation.recognition_family not in _RECOGNITION_FAMILIES:
        raise ValueError("unsupported expectation recognition_family")
    if expectation.recognition_conditioning not in _RECOGNITION_CONDITIONING:
        raise ValueError("unsupported expectation recognition_conditioning")
    if (
        type(expectation.ordered_slots) is not tuple
        or expectation.ordered_slots != _canonical_slots(expectation.horizon)
    ):
        raise ValueError(
            "expectation ordered_slots must equal the canonical 1+6T inventory"
        )
    return expectation


def require_source_law_for_endpoint(
    source_law: ExactSourceMixtureLaw | MomentProjectedLaw,
    *,
    endpoint_config: ArmConfig,
    prior_variant: PriorVariant,
    mixture_mode: MixtureMode,
) -> ExactSourceMixtureLaw | MomentProjectedLaw:
    """Bind both configured source choices to one typed, traced law."""

    checked_config = _validate_source_endpoint(endpoint_config)
    if prior_variant not in ("fixed", "prefix_conditioned"):
        raise ValueError("unsupported source-prior variant")
    if prior_variant != checked_config.prior_variant:
        raise ValueError("prior_variant does not match endpoint_config")
    if mixture_mode != checked_config.mixture_mode:
        raise ValueError("mixture_mode does not match endpoint_config")
    if mixture_mode == "exact":
        if type(source_law) is not ExactSourceMixtureLaw:
            raise ValueError(
                "exact mixture_mode requires an ExactSourceMixtureLaw"
            )
    elif mixture_mode == "moment_projection":
        if type(source_law) is not MomentProjectedLaw:
            raise ValueError(
                "moment_projection mixture_mode requires a "
                "MomentProjectedLaw with a typed projection-error record"
            )
    else:
        raise ValueError("unsupported source-mixture mode")
    source_law.__post_init__()
    if (
        source_law.endpoint_config != checked_config
        or source_law.prior_variant != prior_variant
    ):
        raise ValueError(
            "source-law trace/provenance belongs to another endpoint or prior"
        )
    return source_law


def _factor_identity(
    expectation: LanguageElboExpectation,
    *,
    normalized_factor_identity_sha256: str,
    partition: FactorPartition,
    receiver_t: int,
    contribution: FrozenTensorSnapshot,
    source_prior_trace: H6SourcePriorTrace | None,
) -> str:
    payload = {
        "expectation_identity_sha256": expectation.expectation_identity_sha256,
        "structure_sha256": expectation.structure_sha256,
        "recognition_family": expectation.recognition_family,
        "recognition_conditioning": expectation.recognition_conditioning,
        "evaluation_method": expectation.evaluation_method,
        "source_law": _source_law_payload(
            expectation.source_law,
            source_prior_trace,
        ),
        "normalized_factor_identity_sha256": normalized_factor_identity_sha256,
        "partition": partition,
        "receiver_t": receiver_t,
        "contribution": _snapshot_payload(contribution),
    }
    return hashlib.sha256(
        _TERM_IDENTITY_DOMAIN + canonical_json_bytes(payload)
    ).hexdigest()


def _term(
    expectation: LanguageElboExpectation,
    *,
    partition: FactorPartition,
    receiver_t: int,
    source_prior_trace: H6SourcePriorTrace | None = None,
) -> tuple[H6FactorTerm, Tensor]:
    try:
        value = expectation.contribution(partition, receiver_t)
    except (KeyError, IndexError) as exc:
        raise ValueError(
            f"missing canonical contribution {partition}@{receiver_t}"
        ) from exc
    live_value = _require_live_scalar(value, f"{partition}@{receiver_t}")
    snapshot = FrozenTensorSnapshot.capture(live_value)
    try:
        raw_identity = expectation.normalized_factor_identity(partition, receiver_t)
    except (KeyError, IndexError) as exc:
        raise ValueError(
            f"missing canonical factor identity {partition}@{receiver_t}"
        ) from exc
    _require_sha256(raw_identity, "normalized_factor_identity")
    identity = _factor_identity(
        expectation,
        normalized_factor_identity_sha256=raw_identity,
        partition=partition,
        receiver_t=receiver_t,
        contribution=snapshot,
        source_prior_trace=source_prior_trace,
    )
    return (
        H6FactorTerm(
            receiver_t,
            partition,
            identity,
            snapshot,
        ),
        live_value,
    )


def _evaluate_language_elbo(
    expectation: LanguageElboExpectation,
    *,
    endpoint_config: ArmConfig,
    prior_variant: PriorVariant,
    mixture_mode: MixtureMode,
    source_prior_trace: H6SourcePriorTrace,
) -> H6EndpointLanguageElboTerms:
    """Evaluate the sole complete-ELBO seam with both source choices bound."""

    checked = _validate_expectation(expectation)
    source_law = require_source_law_for_endpoint(
        checked.source_law,
        endpoint_config=endpoint_config,
        prior_variant=prior_variant,
        mixture_mode=mixture_mode,
    )
    if type(source_prior_trace) is not H6SourcePriorTrace:
        raise ValueError("source_prior_trace must be BuiltArm-derived")
    source_prior_trace.__post_init__()
    observed_source_identities: list[
        tuple[Literal["model_source", "state_source"], int, str]
    ] = []
    for receiver_t in range(1, checked.horizon + 1):
        for partition in ("model_source", "state_source"):
            try:
                source_factor = checked.normalized_source_factor(
                    partition,
                    receiver_t,
                )
            except (KeyError, IndexError) as exc:
                raise ValueError(
                    "expectation lacks one exact normalized source factor"
                ) from exc
            if type(source_factor) is not NormalizedSourceFactor:
                raise ValueError(
                    "normalized_source_factor must return an exact "
                    "NormalizedSourceFactor"
                )
            source_factor.__post_init__()
            expected_bank = (
                "model" if partition == "model_source" else "state"
            )
            if (
                source_factor.mask_case_key.bank != expected_bank
                or source_factor.mask_case_key.receiver_t != receiver_t
            ):
                raise ValueError(
                    "normalized source factor has a mismatched bank or receiver"
                )
            reported_identity = _require_sha256(
                checked.normalized_factor_identity(
                    partition,
                    receiver_t,
                ),
                "normalized_source_factor_identity",
            )
            if reported_identity != source_factor.factor_identity_sha256:
                raise ValueError(
                    "normalized source factor identity does not match the "
                    "exact source record"
                )
            observed_source_identities.append(
                (
                    partition,
                    receiver_t,
                    source_factor.factor_identity_sha256,
                )
            )
    if (
        source_prior_trace.endpoint_config != endpoint_config
        or source_prior_trace.prior_variant != prior_variant
        or source_prior_trace.ordered_source_factor_identities
        != tuple(observed_source_identities)
    ):
        raise ValueError(
            "source-prior trace does not match the endpoint and exact ordered "
            "source factors"
        )
    terms: list[H6FactorTerm] = []
    live_values: list[Tensor] = []
    dtype: torch.dtype | None = None
    device: torch.device | None = None
    for partition, receiver_t in _canonical_slots(checked.horizon):
        term, live_value = _term(
            checked,
            partition=partition,
            receiver_t=receiver_t,
            source_prior_trace=source_prior_trace,
        )
        if dtype is None:
            dtype = live_value.dtype
            device = live_value.device
        elif live_value.dtype != dtype or live_value.device != device:
            raise ValueError("every ELBO contribution must share dtype and device")
        terms.append(term)
        live_values.append(live_value)

    training_total = live_values[0]
    for live_value in live_values[1:]:
        training_total = training_total + live_value
    independent_total = _require_live_scalar(
        checked.independently_accumulated_total(),
        "independently_accumulated_total",
    )
    if independent_total.dtype != dtype or independent_total.device != device:
        raise ValueError("independent total must share contribution dtype and device")
    if (
        any(live_value.requires_grad for live_value in live_values)
        and not independent_total.requires_grad
    ):
        raise ValueError(
            "independent total is detached while an ELBO contribution requires grad"
        )
    if not torch.equal(training_total, independent_total):
        raise ValueError(
            "independently accumulated total does not equal the canonical "
            "ordered contribution total"
        )
    ordered_terms = tuple(terms)
    expected_slots = _canonical_slots(checked.horizon)
    observed_slots = tuple(
        (term.partition, term.receiver_t) for term in ordered_terms
    )
    if (
        observed_slots != expected_slots
        or len(ordered_terms) != 1 + 6 * checked.horizon
    ):
        raise ValueError("ordered factor terms do not match the canonical 1+6T slots")
    terms_record = H6LanguageElboTerms.create(
        horizon=checked.horizon,
        ordered_factor_terms=ordered_terms,
        total_language_elbo=training_total,
    )
    return H6EndpointLanguageElboTerms.create(
        endpoint_config=endpoint_config,
        prior_variant=prior_variant,
        mixture_mode=mixture_mode,
        source_prior_trace=source_prior_trace,
        projection_error=(
            source_law.projection_error
            if type(source_law) is MomentProjectedLaw
            else None
        ),
        source_law_marker_identity_sha256=(
            source_law.law_identity_sha256
        ),
        terms=terms_record,
    )


def evaluate_emission_only_ablation(
    expectation: LanguageElboExpectation,
) -> EmissionOnlyAblationTerms:
    """Assemble the separately typed non-ELBO emission-only objective."""

    checked = _validate_expectation(expectation)
    terms = tuple(
        _term(checked, partition="emission", receiver_t=receiver_t)[0]
        for receiver_t in range(1, checked.horizon + 1)
    )
    if tuple(term.receiver_t for term in terms) != tuple(
        range(1, checked.horizon + 1)
    ):
        raise ValueError("emission terms do not cover the canonical horizon")
    return EmissionOnlyAblationTerms.create(ordered_emission_terms=terms)


@final
@dataclass(frozen=True, slots=True)
class CompleteLanguageELBOFactorTrace:
    """Owned H7 view of one authoritative complete post-H6 factor trace."""

    source_trace: H6LanguageElboTerms
    ordered_factor_ids: tuple[str, ...]
    ordered_factor_values: tuple[float, ...]
    total_value: float
    trace_sha256: str

    @classmethod
    def create(
        cls,
        *,
        source_trace: H6LanguageElboTerms,
    ) -> CompleteLanguageELBOFactorTrace:
        if type(source_trace) is not H6LanguageElboTerms:
            raise ValueError("H7 requires an exact immutable H6 factor trace")
        source_trace.__post_init__()
        terms = source_trace.ordered_factor_terms
        if (
            any(term.value.value().numel() != 1 for term in terms)
            or source_trace.total_language_elbo.value().numel() != 1
        ):
            raise ValueError("H7 complete factor values must be scalar")
        factor_ids = tuple(
            term.factor_identity_sha256 for term in terms
        )
        values = tuple(float(term.value.value().item()) for term in terms)
        total_value = float(source_trace.total_language_elbo.value().item())
        semantic = {
            "source_trace_canonical_sha256": source_trace.canonical_sha256,
            "ordered_slots": tuple(
                (term.partition, term.receiver_t) for term in terms
            ),
            "ordered_factor_ids": factor_ids,
            "ordered_factor_values": values,
            "total_value": total_value,
        }
        return cls(
            source_trace=source_trace,
            ordered_factor_ids=factor_ids,
            ordered_factor_values=values,
            total_value=total_value,
            trace_sha256=hashlib.sha256(
                _H7_COMPLETE_TRACE_DOMAIN + canonical_json_bytes(semantic)
            ).hexdigest(),
        )

    def __post_init__(self) -> None:
        if type(self.source_trace) is not H6LanguageElboTerms:
            raise ValueError("complete factor trace must retain its exact H6 source")
        self.source_trace.__post_init__()
        terms = self.source_trace.ordered_factor_terms
        observed_slots = tuple(
            (term.partition, term.receiver_t) for term in terms
        )
        if (
            self.source_trace.horizon != 2
            or observed_slots != _canonical_slots(2)
            or len(terms) != 13
            or any(type(term) is not H6FactorTerm for term in terms)
            or any(term.value.value().numel() != 1 for term in terms)
            or self.source_trace.total_language_elbo.value().numel() != 1
        ):
            raise ValueError(
                "H7 requires the complete ordered T=2 post-H6 factor trace"
            )
        expected_ids = tuple(term.factor_identity_sha256 for term in terms)
        expected_values = tuple(
            float(term.value.value().item()) for term in terms
        )
        expected_total = float(
            self.source_trace.total_language_elbo.value().item()
        )
        if (
            self.ordered_factor_ids != expected_ids
            or self.ordered_factor_values != expected_values
            or self.total_value != expected_total
            or any(not math.isfinite(value) for value in expected_values)
            or not math.isfinite(expected_total)
        ):
            raise ValueError("complete factor-trace values changed after H6")
        semantic = {
            "source_trace_canonical_sha256": self.source_trace.canonical_sha256,
            "ordered_slots": observed_slots,
            "ordered_factor_ids": expected_ids,
            "ordered_factor_values": expected_values,
            "total_value": expected_total,
        }
        expected_hash = hashlib.sha256(
            _H7_COMPLETE_TRACE_DOMAIN + canonical_json_bytes(semantic)
        ).hexdigest()
        if self.trace_sha256 != expected_hash:
            raise ValueError("trace_sha256 does not match the complete H6 trace")


def require_h7_complete_factor_trace(
    trace: H6LanguageElboTerms,
) -> CompleteLanguageELBOFactorTrace:
    """Expose the immutable complete trace without exposing the H6 evaluator.

    H7 may inspect the already-produced factor evidence, but this helper does
    not call, alias, or replace the private post-H6 training-objective seam.
    """

    return CompleteLanguageELBOFactorTrace.create(source_trace=trace)


__all__ = [
    "CompleteLanguageELBOFactorTrace",
    "ExpectationEvaluationMethod",
    "ExactSourceMixtureLaw",
    "FactorPartition",
    "LanguageElboExpectation",
    "MixtureMode",
    "MomentProjectedLaw",
    "PriorVariant",
    "RecognitionConditioningMode",
    "RecognitionFamily",
    "evaluate_emission_only_ablation",
    "require_h7_complete_factor_trace",
    "require_source_law_for_endpoint",
]
