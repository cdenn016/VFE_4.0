"""Canonical differentiable H6 language-ELBO assembly."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal, Protocol, final, runtime_checkable

import torch
from torch import Tensor

from vfe4.types.h6 import (
    EmissionOnlyAblationTerms,
    FrozenTensorSnapshot,
    H6FactorTerm,
    H6LanguageElboTerms,
    canonical_json_bytes,
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


@final
@dataclass(frozen=True, slots=True, init=False)
class ExactSourceMixtureLaw:
    """Factory-only marker for exact source-mixture expectations."""

    law_identity_sha256: str

    def __init__(self, *, law_identity_sha256: str) -> None:
        raise TypeError(
            "ExactSourceMixtureLaw is factory-only; use "
            "ExactSourceMixtureLaw.create"
        )

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("ExactSourceMixtureLaw is sealed")

    def __post_init__(self) -> None:
        _require_sha256(self.law_identity_sha256, "law_identity_sha256")

    @classmethod
    def create(cls, *, law_identity_sha256: str) -> "ExactSourceMixtureLaw":
        instance = object.__new__(cls)
        object.__setattr__(instance, "law_identity_sha256", law_identity_sha256)
        instance.__post_init__()
        return instance


@final
@dataclass(frozen=True, slots=True, init=False)
class MomentProjectedLaw:
    """Factory-only marker carrying a measured moment-projection error."""

    law_identity_sha256: str
    projection_error: FrozenTensorSnapshot

    def __init__(
        self,
        *,
        law_identity_sha256: str,
        projection_error: FrozenTensorSnapshot,
    ) -> None:
        raise TypeError(
            "MomentProjectedLaw is factory-only; use MomentProjectedLaw.create"
        )

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("MomentProjectedLaw is sealed")

    def __post_init__(self) -> None:
        _require_sha256(self.law_identity_sha256, "law_identity_sha256")
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

    @classmethod
    def create(
        cls,
        *,
        law_identity_sha256: str,
        projection_error: FrozenTensorSnapshot,
    ) -> "MomentProjectedLaw":
        instance = object.__new__(cls)
        object.__setattr__(instance, "law_identity_sha256", law_identity_sha256)
        object.__setattr__(instance, "projection_error", projection_error)
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


def _source_law_payload(
    source_law: ExactSourceMixtureLaw | MomentProjectedLaw,
) -> dict[str, object]:
    if type(source_law) is ExactSourceMixtureLaw:
        source_law.__post_init__()
        return {
            "kind": "exact_source_mixture",
            "law_identity_sha256": source_law.law_identity_sha256,
            "projection_error": None,
        }
    if type(source_law) is MomentProjectedLaw:
        source_law.__post_init__()
        return {
            "kind": "moment_projection",
            "law_identity_sha256": source_law.law_identity_sha256,
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


def _factor_identity(
    expectation: LanguageElboExpectation,
    *,
    normalized_factor_identity_sha256: str,
    partition: FactorPartition,
    receiver_t: int,
    contribution: FrozenTensorSnapshot,
) -> str:
    payload = {
        "expectation_identity_sha256": expectation.expectation_identity_sha256,
        "structure_sha256": expectation.structure_sha256,
        "recognition_family": expectation.recognition_family,
        "recognition_conditioning": expectation.recognition_conditioning,
        "evaluation_method": expectation.evaluation_method,
        "source_law": _source_law_payload(expectation.source_law),
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


def evaluate_language_elbo(
    expectation: LanguageElboExpectation,
) -> H6LanguageElboTerms:
    """Assemble exactly ``1 + 6T`` terms from a declared expectation method."""

    checked = _validate_expectation(expectation)
    terms: list[H6FactorTerm] = []
    live_values: list[Tensor] = []
    dtype: torch.dtype | None = None
    device: torch.device | None = None
    for partition, receiver_t in _canonical_slots(checked.horizon):
        term, live_value = _term(
            checked, partition=partition, receiver_t=receiver_t
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
    return H6LanguageElboTerms.create(
        horizon=checked.horizon,
        ordered_factor_terms=ordered_terms,
        total_language_elbo=training_total,
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


__all__ = [
    "ExpectationEvaluationMethod",
    "ExactSourceMixtureLaw",
    "FactorPartition",
    "LanguageElboExpectation",
    "MomentProjectedLaw",
    "RecognitionConditioningMode",
    "RecognitionFamily",
    "evaluate_emission_only_ablation",
    "evaluate_language_elbo",
]
