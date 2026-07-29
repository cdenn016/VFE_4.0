"""Canonical differentiable H6 language-ELBO assembly."""

from __future__ import annotations

import hashlib
import importlib
import math
import weakref
from collections.abc import Callable
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
    H6_EMISSION_ONLY_ABLATION_HASH_DOMAIN,
    H6_OBJECTIVE_EMISSION_ARM_ID,
    H6SourcePriorTrace,
    arm_model_family_sha256,
    canonical_json_bytes,
    h6_source_law_identity,
    h6_source_law_marker_identity,
)
from vfe4.types.h7 import (
    H7_AUTHENTICATED_EVALUATION_HASH_DOMAIN,
    H7_AUTHENTICATED_EVALUATION_ISSUER_ROUTE,
    H7_AUTHENTICATED_EVALUATION_SCOPE,
    H7_RAW_FACTOR_SLOTS,
    H7_RAW_FACTOR_TRACE_ADAPTER_ENTRYPOINT,
    H7_RAW_FACTOR_TRACE_H6_PRODUCER_ROUTE,
    H7_RAW_FACTOR_TRACE_HASH_DOMAIN,
    H7_RAW_FACTOR_TRACE_PRODUCER_CONTRACT_SHA256,
    H7_RAW_FACTOR_TRACE_PRODUCER_KIND,
    H7_RAW_FACTOR_TRACE_REPRESENTATION,
    H7RawFactorTraceEvidence,
    h7_owned_sha256,
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
PriorVariant = Literal["fixed", "parent_specific_pooled_prefix"]

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
_LIVE_EMISSION_CONTEXT_DOMAIN = (
    b"vfe4.h6.live-emission-expectation-context.v1\x00"
)
_LIVE_EMISSION_FACTOR_DOMAIN = b"vfe4.h6.live-emission-factor.v1\x00"
_EMISSION_ONLY_RECORD_DOMAIN = (
    H6_EMISSION_ONLY_ABLATION_HASH_DOMAIN.encode("ascii") + b"\x00"
)
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


@final
@dataclass(frozen=True, slots=True, init=False)
class LiveEmissionExpectationContext:
    """Arm-issued, graph-live support for one observed emission expectation."""

    endpoint_config: ArmConfig
    model_family_sha256: str
    canonical_model_state_sha256: str
    evaluation_method: ExpectationEvaluationMethod
    receiver_t: int
    observed_token_id: int
    state_support: FrozenTensorSnapshot
    model_support: FrozenTensorSnapshot
    normalized_weights: FrozenTensorSnapshot
    context_sha256: str

    def __init__(self) -> None:
        raise TypeError(
            "LiveEmissionExpectationContext is BuiltArm-only; use "
            "BuiltArm.issue_emission_expectation_context"
        )

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("LiveEmissionExpectationContext is sealed")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "endpoint_config_sha256": self.endpoint_config.config_sha256,
            "model_family_sha256": self.model_family_sha256,
            "canonical_model_state_sha256": (
                self.canonical_model_state_sha256
            ),
            "evaluation_method": self.evaluation_method,
            "receiver_t": self.receiver_t,
            "observed_token_id": self.observed_token_id,
            "state_support": _snapshot_payload(self.state_support),
            "model_support": _snapshot_payload(self.model_support),
            "normalized_weights": _snapshot_payload(
                self.normalized_weights
            ),
        }

    def __post_init__(self) -> None:
        endpoint_config = _validate_source_endpoint(self.endpoint_config)
        if (
            endpoint_config.config_id != H6_OBJECTIVE_EMISSION_ARM_ID
            or endpoint_config.objective_kind
            != "emission_only_ablation_non_elbo"
        ):
            raise ValueError(
                "live emission context requires the literal OBJECTIVE endpoint"
            )
        _require_sha256(self.model_family_sha256, "model_family_sha256")
        if (
            self.model_family_sha256
            != arm_model_family_sha256(endpoint_config)
        ):
            raise ValueError(
                "live emission context model family does not match its endpoint"
            )
        _require_sha256(
            self.canonical_model_state_sha256,
            "canonical_model_state_sha256",
        )
        if self.evaluation_method not in _EVALUATION_METHODS:
            raise ValueError(
                "live emission context has an unsupported evaluation_method"
            )
        if (
            type(self.receiver_t) is not int
            or self.receiver_t < 1
            or self.receiver_t > endpoint_config.horizon
        ):
            raise ValueError(
                "live emission context receiver is outside the endpoint horizon"
            )
        if (
            type(self.observed_token_id) is not int
            or self.observed_token_id < 0
            or self.observed_token_id >= endpoint_config.vocabulary.size
        ):
            raise ValueError(
                "live emission context observed_token_id is outside vocabulary"
            )
        snapshots = (
            self.state_support,
            self.model_support,
            self.normalized_weights,
        )
        if any(type(snapshot) is not FrozenTensorSnapshot for snapshot in snapshots):
            raise ValueError(
                "live emission context tensors must be exact frozen snapshots"
            )
        for snapshot in snapshots:
            snapshot.assert_intact()
        expected_width = endpoint_config.capacity_allocation.latent_width
        if len(self.normalized_weights.shape) != 1:
            raise ValueError(
                "live emission context normalized_weights must be a vector"
            )
        sample_count = self.normalized_weights.shape[0]
        if (
            expected_width is None
            or len(self.state_support.shape) != 2
            or self.state_support.shape
            != (sample_count, expected_width)
            or self.model_support.shape
            != (sample_count, expected_width)
            or self.normalized_weights.shape != (sample_count,)
            or sample_count <= 0
        ):
            raise ValueError(
                "live emission context support shapes do not match the "
                "endpoint latent width"
            )
        if (
            self.state_support.dtype != "float64"
            or self.model_support.dtype != "float64"
            or self.normalized_weights.dtype != "float64"
            or len(
                {
                    self.state_support.device,
                    self.model_support.device,
                    self.normalized_weights.device,
                }
            )
            != 1
        ):
            raise ValueError(
                "live emission context tensors must be same-device float64"
            )
        state_support = self.state_support.value()
        model_support = self.model_support.value()
        normalized_weights = self.normalized_weights.value()
        if (
            not bool(torch.all(torch.isfinite(state_support)))
            or not bool(torch.all(torch.isfinite(model_support)))
            or not bool(torch.all(torch.isfinite(normalized_weights)))
            or not bool(torch.all(normalized_weights >= 0.0))
        ):
            raise ValueError(
                "live emission context tensors must be finite with "
                "nonnegative weights"
            )
        one = torch.ones(
            (),
            dtype=normalized_weights.dtype,
            device=normalized_weights.device,
        )
        if not bool(
            torch.isclose(
                normalized_weights.sum(),
                one,
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise ValueError(
                "live emission context weights must sum to one"
            )
        expected = hashlib.sha256(
            _LIVE_EMISSION_CONTEXT_DOMAIN
            + canonical_json_bytes(self.canonical_payload())
        ).hexdigest()
        if self.context_sha256 != expected:
            raise ValueError("live emission context identity is stale")

    @classmethod
    def _from_live_arm(
        cls,
        *,
        endpoint_config: ArmConfig,
        model_family_sha256: str,
        canonical_model_state_sha256: str,
        evaluation_method: ExpectationEvaluationMethod,
        receiver_t: int,
        observed_token_id: int,
        state_support: Tensor,
        model_support: Tensor,
        normalized_weights: Tensor,
    ) -> "LiveEmissionExpectationContext":
        values: dict[str, object] = {
            "endpoint_config": endpoint_config,
            "model_family_sha256": model_family_sha256,
            "canonical_model_state_sha256": canonical_model_state_sha256,
            "evaluation_method": evaluation_method,
            "receiver_t": receiver_t,
            "observed_token_id": observed_token_id,
            "state_support": FrozenTensorSnapshot.capture(state_support),
            "model_support": FrozenTensorSnapshot.capture(model_support),
            "normalized_weights": FrozenTensorSnapshot.capture(
                normalized_weights
            ),
        }
        instance = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(instance, name, value)
        object.__setattr__(
            instance,
            "context_sha256",
            hashlib.sha256(
                _LIVE_EMISSION_CONTEXT_DOMAIN
                + canonical_json_bytes(instance.canonical_payload())
            ).hexdigest(),
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


@runtime_checkable
class LiveEmissionExpectation(LanguageElboExpectation, Protocol):
    """Expectation provider whose emission support was issued by ``BuiltArm``."""

    def emission_expectation_context(
        self,
        receiver_t: int,
    ) -> LiveEmissionExpectationContext: ...


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
    complete_endpoint = (
        endpoint_config.arm in (ArmId.A2, ArmId.A5)
        and endpoint_config.objective_kind == "complete_elbo"
        and endpoint_config.prior_variant
        in ("fixed", "parent_specific_pooled_prefix")
        and endpoint_config.mixture_mode in ("exact", "moment_projection")
    )
    emission_endpoint = (
        endpoint_config.arm is ArmId.A5
        and endpoint_config.config_id == H6_OBJECTIVE_EMISSION_ARM_ID
        and endpoint_config.objective_kind
        == "emission_only_ablation_non_elbo"
        and endpoint_config.prior_variant
        == "parent_specific_pooled_prefix"
        and endpoint_config.mixture_mode == "exact"
    )
    if not (complete_endpoint or emission_endpoint):
        raise ValueError(
            "source law requires a supported complete categorical endpoint or "
            "the literal parent-specific OBJECTIVE endpoint"
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
    if checked_config.objective_kind != "complete_elbo":
        raise ValueError(
            "complete ELBO source-law binding requires a complete endpoint"
        )
    if prior_variant not in ("fixed", "parent_specific_pooled_prefix"):
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


def _ordered_source_factor_identities(
    expectation: LanguageElboExpectation,
) -> tuple[
    tuple[Literal["model_source", "state_source"], int, str],
    ...,
]:
    observed: list[
        tuple[Literal["model_source", "state_source"], int, str]
    ] = []
    for receiver_t in range(1, expectation.horizon + 1):
        for partition in ("model_source", "state_source"):
            try:
                source_factor = expectation.normalized_source_factor(
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
                expectation.normalized_factor_identity(
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
            observed.append(
                (
                    partition,
                    receiver_t,
                    source_factor.factor_identity_sha256,
                )
            )
    return tuple(observed)


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
    observed_source_identities = _ordered_source_factor_identities(checked)
    if (
        source_prior_trace.endpoint_config != endpoint_config
        or source_prior_trace.prior_variant != prior_variant
        or source_prior_trace.ordered_source_factor_identities
        != observed_source_identities
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


def _evaluate_emission_only_ablation(
    expectation: LiveEmissionExpectation,
    *,
    endpoint_config: ArmConfig,
    model_family_sha256: str,
    model: object,
    canonical_model_state_sha256: str,
    source_prior_trace: H6SourcePriorTrace,
) -> EmissionOnlyAblationTerms:
    """Recompute one provenance-bound non-ELBO through the exact live model."""

    checked = _validate_expectation(expectation)
    if not isinstance(expectation, LiveEmissionExpectation):
        raise ValueError(
            "emission-only expectation must provide BuiltArm-issued live "
            "emission contexts"
        )
    validated_config = _validate_source_endpoint(endpoint_config)
    if (
        validated_config.config_id != H6_OBJECTIVE_EMISSION_ARM_ID
        or validated_config.objective_kind
        != "emission_only_ablation_non_elbo"
        or validated_config.prior_variant
        != "parent_specific_pooled_prefix"
        or validated_config.mixture_mode != "exact"
    ):
        raise ValueError(
            "emission-only evaluation requires the literal parent-specific "
            "OBJECTIVE endpoint"
        )
    _require_sha256(model_family_sha256, "model_family_sha256")
    if model_family_sha256 != arm_model_family_sha256(validated_config):
        raise ValueError(
            "emission-only model family does not match its literal endpoint"
        )
    _require_sha256(
        canonical_model_state_sha256,
        "canonical_model_state_sha256",
    )
    from vfe4.predictive import (
        canonical_model_state_sha256 as hash_model_state,
    )
    from vfe4.training.arms import LatentLanguageArmModel

    if type(model) is not LatentLanguageArmModel:
        raise ValueError(
            "emission-only evaluation requires the exact live latent arm model"
        )
    if hash_model_state(model) != canonical_model_state_sha256:
        raise ValueError(
            "emission-only canonical model state does not match the live arm"
        )
    if type(checked.source_law) is not ExactSourceMixtureLaw:
        raise ValueError(
            "emission-only OBJECTIVE endpoint requires its exact source law"
        )
    checked.source_law.__post_init__()
    if checked.source_law.endpoint_config != validated_config:
        raise ValueError(
            "emission-only expectation source law belongs to another endpoint"
        )
    if type(source_prior_trace) is not H6SourcePriorTrace:
        raise ValueError(
            "emission-only source_prior_trace must be BuiltArm-derived"
        )
    source_prior_trace.__post_init__()
    observed_source_identities = _ordered_source_factor_identities(checked)
    if (
        source_prior_trace.endpoint_config != validated_config
        or source_prior_trace.model_family_sha256 != model_family_sha256
        or source_prior_trace.prior_variant
        != "parent_specific_pooled_prefix"
        or source_prior_trace.prior_type
        != "ParentSpecificPooledPrefixSourcePrior"
        or source_prior_trace.ordered_source_factor_identities
        != observed_source_identities
    ):
        raise ValueError(
            "emission-only live-prior trace does not match the endpoint, "
            "model family, or exact ordered source factors"
        )
    terms: list[H6FactorTerm] = []
    for receiver_t in range(1, checked.horizon + 1):
        try:
            context = expectation.emission_expectation_context(receiver_t)
        except (AttributeError, KeyError, IndexError) as exc:
            raise ValueError(
                "expectation lacks one BuiltArm-issued emission context"
            ) from exc
        if type(context) is not LiveEmissionExpectationContext:
            raise ValueError(
                "emission expectation context must be the exact sealed type"
            )
        context.__post_init__()
        if (
            context.endpoint_config != validated_config
            or context.model_family_sha256 != model_family_sha256
            or context.canonical_model_state_sha256
            != canonical_model_state_sha256
            or context.evaluation_method != checked.evaluation_method
            or context.receiver_t != receiver_t
        ):
            raise ValueError(
                "emission context belongs to another endpoint, model state, "
                "evaluation method, or receiver"
            )
        state_support = context.state_support.value()
        model_support = context.model_support.value()
        normalized_weights = context.normalized_weights.value()
        sample_log_probabilities = torch.stack(
            tuple(
                model.emission_log_probs(
                    state=state_support[sample_index],
                    model=model_support[sample_index],
                )[context.observed_token_id]
                for sample_index in range(normalized_weights.numel())
            )
        )
        live_value = _require_live_scalar(
            torch.sum(normalized_weights * sample_log_probabilities),
            f"live emission@{receiver_t}",
        )
        snapshot = FrozenTensorSnapshot.capture(live_value)
        factor_payload = {
            "endpoint_config_sha256": validated_config.config_sha256,
            "model_family_sha256": model_family_sha256,
            "canonical_model_state_sha256": (
                canonical_model_state_sha256
            ),
            "source_law": _source_law_payload(
                checked.source_law,
                source_prior_trace,
            ),
            "emission_context_sha256": context.context_sha256,
            "partition": "emission",
            "receiver_t": receiver_t,
            "contribution": _snapshot_payload(snapshot),
        }
        terms.append(
            H6FactorTerm(
                receiver_t=receiver_t,
                partition="emission",
                factor_identity_sha256=hashlib.sha256(
                    _LIVE_EMISSION_FACTOR_DOMAIN
                    + canonical_json_bytes(factor_payload)
                ).hexdigest(),
                value=snapshot,
            )
        )
    ordered_terms = tuple(terms)
    if tuple(term.receiver_t for term in terms) != tuple(
        range(1, checked.horizon + 1)
    ):
        raise ValueError("emission terms do not cover the canonical horizon")
    total = ordered_terms[0].value.value()
    for term in ordered_terms[1:]:
        total = total + term.value.value()
    total_snapshot = FrozenTensorSnapshot.capture(total)
    source_law_marker_identity_sha256 = (
        checked.source_law.law_identity_sha256
    )
    source_law_identity_sha256 = h6_source_law_identity(
        endpoint_config=validated_config,
        source_prior_trace=source_prior_trace,
        projection_error=None,
    )
    values: dict[str, object] = {
        "objective_kind": "emission_only_ablation_non_elbo",
        "endpoint_config": validated_config,
        "model_family_sha256": model_family_sha256,
        "canonical_model_state_sha256": canonical_model_state_sha256,
        "prior_variant": "parent_specific_pooled_prefix",
        "source_prior_trace": source_prior_trace,
        "source_law_marker_identity_sha256": (
            source_law_marker_identity_sha256
        ),
        "source_law_identity_sha256": source_law_identity_sha256,
        "ordered_emission_terms": ordered_terms,
        "total": total_snapshot,
    }
    result = object.__new__(EmissionOnlyAblationTerms)
    for name, value in values.items():
        object.__setattr__(result, name, value)
    object.__setattr__(
        result,
        "canonical_sha256",
        hashlib.sha256(
            _EMISSION_ONLY_RECORD_DOMAIN
            + canonical_json_bytes(result._payload())
        ).hexdigest(),
    )
    result.__post_init__()
    return result


def evaluate_emission_only_ablation(
    expectation: LiveEmissionExpectation,
    *,
    arm: object,
) -> EmissionOnlyAblationTerms:
    """Route emission-only training through the exact live ``BuiltArm``."""

    from vfe4.training.arms import BuiltArm

    if type(arm) is not BuiltArm:
        raise ValueError(
            "emission-only evaluation requires an exact live BuiltArm"
        )
    return arm.evaluate_emission_only_ablation(expectation)


@final
@dataclass(
    frozen=True,
    slots=True,
    eq=False,
    init=False,
    weakref_slot=True,
)
class H7AuthenticatedEvaluation:
    """Assembly-only receipt for one exact factory ``BuiltArm`` evaluation.

    This authenticates the live assembly route and its bound inputs.  It does
    not claim that the supplied expectation values were derived from the
    retained model state; that requires a separate law-value provenance layer.
    """

    endpoint: H6EndpointLanguageElboTerms
    attestation_scope: Literal["built-arm-complete-elbo-assembly-v1"]
    endpoint_config_sha256: str
    model_family_sha256: str
    canonical_model_state_sha256: str
    elbo_inventory_sha256: str
    evaluator_implementation_sha256: str
    expectation_identity_sha256: str
    expectation_structure_sha256: str
    expectation_source_law_marker_identity_sha256: str
    source_prior_trace_sha256: str
    endpoint_source_law_identity_sha256: str
    endpoint_language_elbo_sha256: str
    producer_route: tuple[str, str]
    issuer_route: Literal[
        "vfe4.objective.language_elbo.capture_h7_complete_language_elbo"
    ]
    attestation_sha256: str

    def __init__(self) -> None:
        raise TypeError(
            "H7AuthenticatedEvaluation is capture-only; use "
            "capture_h7_complete_language_elbo"
        )

    def __copy__(self) -> H7AuthenticatedEvaluation:
        raise TypeError("H7 authenticated evaluation copy is forbidden")

    def __deepcopy__(
        self,
        memo: dict[int, object],
    ) -> H7AuthenticatedEvaluation:
        del memo
        raise TypeError("H7 authenticated evaluation deepcopy is forbidden")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError("H7 authenticated evaluation pickle is forbidden")

    def attestation_payload(self) -> dict[str, object]:
        return {
            "attestation_scope": self.attestation_scope,
            "endpoint_config_sha256": self.endpoint_config_sha256,
            "model_family_sha256": self.model_family_sha256,
            "canonical_model_state_sha256": (
                self.canonical_model_state_sha256
            ),
            "elbo_inventory_sha256": self.elbo_inventory_sha256,
            "evaluator_implementation_sha256": (
                self.evaluator_implementation_sha256
            ),
            "expectation_identity_sha256": (
                self.expectation_identity_sha256
            ),
            "expectation_structure_sha256": (
                self.expectation_structure_sha256
            ),
            "expectation_source_law_marker_identity_sha256": (
                self.expectation_source_law_marker_identity_sha256
            ),
            "source_prior_trace_sha256": self.source_prior_trace_sha256,
            "endpoint_source_law_identity_sha256": (
                self.endpoint_source_law_identity_sha256
            ),
            "endpoint_language_elbo_sha256": (
                self.endpoint_language_elbo_sha256
            ),
            "producer_route": self.producer_route,
            "issuer_route": self.issuer_route,
        }

    def __post_init__(self) -> None:
        if type(self.endpoint) is not H6EndpointLanguageElboTerms:
            raise ValueError(
                "H7 authenticated evaluation requires an exact H6 endpoint"
            )
        self.endpoint.__post_init__()
        if (
            self.attestation_scope != H7_AUTHENTICATED_EVALUATION_SCOPE
            or self.producer_route
            != H7_RAW_FACTOR_TRACE_H6_PRODUCER_ROUTE
            or self.issuer_route
            != H7_AUTHENTICATED_EVALUATION_ISSUER_ROUTE
        ):
            raise ValueError("H7 evaluation assembly attestation route changed")
        for name in (
            "endpoint_config_sha256",
            "model_family_sha256",
            "canonical_model_state_sha256",
            "elbo_inventory_sha256",
            "evaluator_implementation_sha256",
            "expectation_identity_sha256",
            "expectation_structure_sha256",
            "expectation_source_law_marker_identity_sha256",
            "source_prior_trace_sha256",
            "endpoint_source_law_identity_sha256",
            "endpoint_language_elbo_sha256",
            "attestation_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        endpoint = self.endpoint
        if (
            self.endpoint_config_sha256
            != endpoint.endpoint_config.config_sha256
            or self.model_family_sha256
            != endpoint.source_prior_trace.model_family_sha256
            or self.expectation_source_law_marker_identity_sha256
            != endpoint.source_law_marker_identity_sha256
            or self.source_prior_trace_sha256
            != endpoint.source_prior_trace_sha256
            or self.endpoint_source_law_identity_sha256
            != endpoint.source_law_identity_sha256
            or self.endpoint_language_elbo_sha256
            != endpoint.canonical_sha256
        ):
            raise ValueError(
                "H7 authenticated evaluation endpoint binding changed"
            )
        if self.attestation_sha256 != h7_owned_sha256(
            H7_AUTHENTICATED_EVALUATION_HASH_DOMAIN,
            self.attestation_payload(),
        ):
            raise ValueError(
                "H7 authenticated evaluation attestation hash changed"
            )


def _build_h7_authenticated_evaluation_api():
    registry: dict[
        int,
        tuple[
            weakref.ReferenceType[H7AuthenticatedEvaluation],
            tuple[int, bytes, str],
        ],
    ] = {}
    factory_api: tuple[
        Callable[[object], object],
        Callable[
            [object, LanguageElboExpectation],
            H6EndpointLanguageElboTerms,
        ],
        str,
    ] | None = None

    def issuance_snapshot(
        evaluation: H7AuthenticatedEvaluation,
    ) -> tuple[int, bytes, str]:
        return (
            id(evaluation.endpoint),
            canonical_json_bytes(evaluation.attestation_payload()),
            evaluation.attestation_sha256,
        )

    def issue(**values: object) -> H7AuthenticatedEvaluation:
        evaluation = object.__new__(H7AuthenticatedEvaluation)
        for name, value in values.items():
            object.__setattr__(evaluation, name, value)
        object.__setattr__(
            evaluation,
            "attestation_sha256",
            h7_owned_sha256(
                H7_AUTHENTICATED_EVALUATION_HASH_DOMAIN,
                evaluation.attestation_payload(),
            ),
        )
        evaluation.__post_init__()
        identity = id(evaluation)

        def remove(
            reference: weakref.ReferenceType[H7AuthenticatedEvaluation],
        ) -> None:
            current = registry.get(identity)
            if current is not None and current[0] is reference:
                registry.pop(identity, None)

        reference = weakref.ref(evaluation, remove)
        current = registry.get(identity)
        if current is not None and current[0]() is not None:
            raise RuntimeError(
                "H7 evaluation identity was already capture-issued"
            )
        registry[identity] = (reference, issuance_snapshot(evaluation))
        return evaluation

    def require(value: object) -> H7AuthenticatedEvaluation:
        if type(value) is not H7AuthenticatedEvaluation:
            raise ValueError(
                "H7 endpoint adaptation requires an exact authenticated "
                "BuiltArm evaluation receipt"
            )
        current = registry.get(id(value))
        if current is None or current[0]() is not value:
            raise ValueError(
                "H7 evaluation receipt is not a live registered "
                "authentication"
            )
        value.__post_init__()
        if issuance_snapshot(value) != current[1]:
            raise ValueError(
                "H7 evaluation receipt changed after capture issuance"
            )
        return value

    def bind_factory_api_once(
        arm_validator: Callable[[object], object],
        evaluator: Callable[
            [object, LanguageElboExpectation],
            H6EndpointLanguageElboTerms,
        ],
        evaluator_implementation_sha256: str,
    ) -> None:
        nonlocal factory_api
        if factory_api is not None:
            raise RuntimeError("H7 factory API is already bound")
        if not callable(arm_validator) or not callable(evaluator):
            raise TypeError("H7 factory API requires exact callable entries")
        _require_sha256(
            evaluator_implementation_sha256,
            "evaluator_implementation_sha256",
        )
        factory_api = (
            arm_validator,
            evaluator,
            evaluator_implementation_sha256,
        )

    def capture(
        arm: object,
        expectation: LanguageElboExpectation,
    ) -> H7AuthenticatedEvaluation:
        """Capture one BuiltArm assembly without attesting value derivation."""

        from vfe4.predictive.identities import (
            canonical_model_state_sha256,
        )

        if factory_api is None:
            importlib.import_module("vfe4.training.arms")
        if factory_api is None:
            raise RuntimeError(
                "H7 factory API did not bind during arms initialization"
            )
        arm_validator, evaluator, evaluator_sha256 = factory_api
        checked_arm = arm_validator(arm)
        checked_expectation = _validate_expectation(expectation)
        config = checked_arm.config
        if type(config) is not ArmConfig:
            raise ValueError("H7 capture requires an exact endpoint config")
        config.__post_init__()
        source_law = require_source_law_for_endpoint(
            checked_expectation.source_law,
            endpoint_config=config,
            prior_variant=config.prior_variant,  # type: ignore[arg-type]
            mixture_mode=config.mixture_mode,
        )
        model_inventory = getattr(
            checked_arm.model,
            "elbo_factor_inventory",
            None,
        )
        model_inventory_sha256 = getattr(
            checked_arm.model,
            "elbo_inventory_sha256",
            None,
        )
        if (
            config.horizon != 2
            or checked_expectation.horizon != config.horizon
            or source_law.endpoint_config != config
            or checked_arm.model_family_sha256
            != arm_model_family_sha256(config)
            or checked_arm.elbo_factor_inventory != model_inventory
            or checked_arm.elbo_inventory_sha256
            != model_inventory_sha256
        ):
            raise ValueError(
                "factory BuiltArm structure, expectation, or ELBO inventory "
                "changed"
            )
        model_state_before = canonical_model_state_sha256(
            checked_arm.model
        )
        endpoint = evaluator(
            checked_arm,
            checked_expectation,
        )
        model_state_after = canonical_model_state_sha256(checked_arm.model)
        if model_state_after != model_state_before:
            raise ValueError(
                "BuiltArm model state changed during complete ELBO assembly"
            )
        if type(endpoint) is not H6EndpointLanguageElboTerms:
            raise ValueError(
                "BuiltArm returned a non-H6 complete ELBO endpoint"
            )
        endpoint.__post_init__()
        source_prior = getattr(checked_arm.model, "source_prior", None)
        if source_prior is None:
            raise ValueError(
                "complete ELBO BuiltArm lacks its live source prior"
            )
        source_prior_state_sha256 = canonical_model_state_sha256(
            source_prior
        )
        if (
            endpoint.endpoint_config != config
            or endpoint.source_prior_trace.model_family_sha256
            != checked_arm.model_family_sha256
            or endpoint.source_prior_trace.prior_model_state_sha256
            != source_prior_state_sha256
            or endpoint.source_law_marker_identity_sha256
            != source_law.law_identity_sha256
        ):
            raise ValueError(
                "BuiltArm endpoint source-prior, source-law, or config "
                "binding changed"
            )
        return issue(
            endpoint=endpoint,
            attestation_scope=H7_AUTHENTICATED_EVALUATION_SCOPE,
            endpoint_config_sha256=config.config_sha256,
            model_family_sha256=checked_arm.model_family_sha256,
            canonical_model_state_sha256=model_state_before,
            elbo_inventory_sha256=checked_arm.elbo_inventory_sha256,
            evaluator_implementation_sha256=(
                evaluator_sha256
            ),
            expectation_identity_sha256=(
                checked_expectation.expectation_identity_sha256
            ),
            expectation_structure_sha256=(
                checked_expectation.structure_sha256
            ),
            expectation_source_law_marker_identity_sha256=(
                source_law.law_identity_sha256
            ),
            source_prior_trace_sha256=(
                endpoint.source_prior_trace_sha256
            ),
            endpoint_source_law_identity_sha256=(
                endpoint.source_law_identity_sha256
            ),
            endpoint_language_elbo_sha256=endpoint.canonical_sha256,
            producer_route=H7_RAW_FACTOR_TRACE_H6_PRODUCER_ROUTE,
            issuer_route=H7_AUTHENTICATED_EVALUATION_ISSUER_ROUTE,
        )

    return capture, require, bind_factory_api_once


(
    capture_h7_complete_language_elbo,
    _require_h7_authenticated_evaluation,
    _bind_h7_factory_api_once,
) = _build_h7_authenticated_evaluation_api()
del _build_h7_authenticated_evaluation_api


@final
@dataclass(frozen=True, slots=True, init=False)
class CompleteLanguageELBOFactorTrace:
    """Owned H7 view of one registry-authenticated H6 assembly receipt."""

    authenticated_evaluation: H7AuthenticatedEvaluation
    representation: Literal[
        "raw_expected_log_factors_plus_recognition_entropy_v1"
    ]
    producer_kind: Literal["h6_endpoint_complete_elbo_v1"]
    attestation_scope: Literal["built-arm-complete-elbo-assembly-v1"]
    producer_attestation_sha256: str
    endpoint_config_sha256: str
    model_family_sha256: str
    canonical_model_state_sha256: str
    elbo_inventory_sha256: str
    evaluator_implementation_sha256: str
    expectation_identity_sha256: str
    expectation_structure_sha256: str
    expectation_source_law_marker_identity_sha256: str
    h6_producer_route: tuple[str, str]
    issuer_route: Literal[
        "vfe4.objective.language_elbo.capture_h7_complete_language_elbo"
    ]
    h7_adapter_entrypoint: Literal[
        "vfe4.objective.language_elbo.require_h7_complete_factor_trace"
    ]
    endpoint_language_elbo_sha256: str
    source_law_identity_sha256: str
    source_prior_trace_sha256: str
    producer_contract_sha256: str
    ordered_factor_ids: tuple[str, ...]
    ordered_factor_values: tuple[float, ...]
    total_value: float
    trace_sha256: str

    def __init__(self) -> None:
        raise TypeError(
            "CompleteLanguageELBOFactorTrace is authenticated-adapter-only"
        )

    @property
    def source_trace(self) -> H6EndpointLanguageElboTerms:
        return self.authenticated_evaluation.endpoint

    @classmethod
    def create(
        cls,
        **values: object,
    ) -> CompleteLanguageELBOFactorTrace:
        del values
        raise TypeError(
            "CompleteLanguageELBOFactorTrace is authenticated-adapter-only"
        )

    @classmethod
    def _from_authenticated_evaluation(
        cls,
        authenticated_evaluation: H7AuthenticatedEvaluation,
    ) -> CompleteLanguageELBOFactorTrace:
        receipt = _require_h7_authenticated_evaluation(
            authenticated_evaluation
        )
        source_trace = receipt.endpoint
        terms = source_trace.terms.ordered_factor_terms
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
            "representation": H7_RAW_FACTOR_TRACE_REPRESENTATION,
            "producer_kind": H7_RAW_FACTOR_TRACE_PRODUCER_KIND,
            "attestation_scope": receipt.attestation_scope,
            "producer_attestation_sha256": receipt.attestation_sha256,
            "endpoint_config_sha256": receipt.endpoint_config_sha256,
            "model_family_sha256": receipt.model_family_sha256,
            "canonical_model_state_sha256": (
                receipt.canonical_model_state_sha256
            ),
            "elbo_inventory_sha256": receipt.elbo_inventory_sha256,
            "evaluator_implementation_sha256": (
                receipt.evaluator_implementation_sha256
            ),
            "expectation_identity_sha256": (
                receipt.expectation_identity_sha256
            ),
            "expectation_structure_sha256": (
                receipt.expectation_structure_sha256
            ),
            "expectation_source_law_marker_identity_sha256": (
                receipt.expectation_source_law_marker_identity_sha256
            ),
            "h6_producer_route": receipt.producer_route,
            "issuer_route": receipt.issuer_route,
            "h7_adapter_entrypoint": (
                H7_RAW_FACTOR_TRACE_ADAPTER_ENTRYPOINT
            ),
            "endpoint_language_elbo_sha256": source_trace.canonical_sha256,
            "source_law_identity_sha256": (
                source_trace.source_law_identity_sha256
            ),
            "source_prior_trace_sha256": (
                source_trace.source_prior_trace_sha256
            ),
            "producer_contract_sha256": (
                H7_RAW_FACTOR_TRACE_PRODUCER_CONTRACT_SHA256
            ),
            "ordered_slots": tuple(
                (term.partition, term.receiver_t) for term in terms
            ),
            "ordered_factor_ids": factor_ids,
            "ordered_factor_values": values,
            "total_value": total_value,
        }
        instance = object.__new__(cls)
        bound_values: dict[str, object] = {
            "authenticated_evaluation": receipt,
            "representation": H7_RAW_FACTOR_TRACE_REPRESENTATION,
            "producer_kind": H7_RAW_FACTOR_TRACE_PRODUCER_KIND,
            "attestation_scope": receipt.attestation_scope,
            "producer_attestation_sha256": receipt.attestation_sha256,
            "endpoint_config_sha256": receipt.endpoint_config_sha256,
            "model_family_sha256": receipt.model_family_sha256,
            "canonical_model_state_sha256": (
                receipt.canonical_model_state_sha256
            ),
            "elbo_inventory_sha256": receipt.elbo_inventory_sha256,
            "evaluator_implementation_sha256": (
                receipt.evaluator_implementation_sha256
            ),
            "expectation_identity_sha256": (
                receipt.expectation_identity_sha256
            ),
            "expectation_structure_sha256": (
                receipt.expectation_structure_sha256
            ),
            "expectation_source_law_marker_identity_sha256": (
                receipt.expectation_source_law_marker_identity_sha256
            ),
            "h6_producer_route": receipt.producer_route,
            "issuer_route": receipt.issuer_route,
            "h7_adapter_entrypoint": H7_RAW_FACTOR_TRACE_ADAPTER_ENTRYPOINT,
            "endpoint_language_elbo_sha256": source_trace.canonical_sha256,
            "source_law_identity_sha256": (
                source_trace.source_law_identity_sha256
            ),
            "source_prior_trace_sha256": (
                source_trace.source_prior_trace_sha256
            ),
            "producer_contract_sha256": (
                H7_RAW_FACTOR_TRACE_PRODUCER_CONTRACT_SHA256
            ),
            "ordered_factor_ids": factor_ids,
            "ordered_factor_values": values,
            "total_value": total_value,
            "trace_sha256": h7_owned_sha256(
                H7_RAW_FACTOR_TRACE_HASH_DOMAIN,
                semantic,
            ),
        }
        for name, value in bound_values.items():
            object.__setattr__(instance, name, value)
        instance.__post_init__()
        return instance

    def __post_init__(self) -> None:
        receipt = _require_h7_authenticated_evaluation(
            self.authenticated_evaluation
        )
        source_trace = receipt.endpoint
        terms = source_trace.terms.ordered_factor_terms
        observed_slots = tuple(
            (term.partition, term.receiver_t) for term in terms
        )
        if (
            source_trace.endpoint_config.horizon != 2
            or observed_slots != H7_RAW_FACTOR_SLOTS
            or len(terms) != 13
            or any(type(term) is not H6FactorTerm for term in terms)
            or any(term.value.value().numel() != 1 for term in terms)
            or source_trace.total_language_elbo.value().numel() != 1
        ):
            raise ValueError(
                "H7 requires the complete ordered T=2 post-H6 factor trace"
            )
        if (
            self.representation != H7_RAW_FACTOR_TRACE_REPRESENTATION
            or self.producer_kind != H7_RAW_FACTOR_TRACE_PRODUCER_KIND
            or self.attestation_scope != receipt.attestation_scope
            or self.producer_attestation_sha256
            != receipt.attestation_sha256
            or self.endpoint_config_sha256
            != receipt.endpoint_config_sha256
            or self.model_family_sha256 != receipt.model_family_sha256
            or self.canonical_model_state_sha256
            != receipt.canonical_model_state_sha256
            or self.elbo_inventory_sha256
            != receipt.elbo_inventory_sha256
            or self.evaluator_implementation_sha256
            != receipt.evaluator_implementation_sha256
            or self.expectation_identity_sha256
            != receipt.expectation_identity_sha256
            or self.expectation_structure_sha256
            != receipt.expectation_structure_sha256
            or self.expectation_source_law_marker_identity_sha256
            != receipt.expectation_source_law_marker_identity_sha256
            or self.h6_producer_route != receipt.producer_route
            or self.issuer_route != receipt.issuer_route
            or self.h7_adapter_entrypoint
            != H7_RAW_FACTOR_TRACE_ADAPTER_ENTRYPOINT
            or self.endpoint_language_elbo_sha256
            != source_trace.canonical_sha256
            or self.source_law_identity_sha256
            != source_trace.source_law_identity_sha256
            or self.source_prior_trace_sha256
            != source_trace.source_prior_trace_sha256
            or self.producer_contract_sha256
            != H7_RAW_FACTOR_TRACE_PRODUCER_CONTRACT_SHA256
        ):
            raise ValueError(
                "H7 raw representation, attestation, or endpoint binding changed"
            )
        expected_ids = tuple(term.factor_identity_sha256 for term in terms)
        expected_values = tuple(
            float(term.value.value().item()) for term in terms
        )
        expected_total = float(
            source_trace.total_language_elbo.value().item()
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
            "representation": self.representation,
            "producer_kind": self.producer_kind,
            "attestation_scope": self.attestation_scope,
            "producer_attestation_sha256": (
                self.producer_attestation_sha256
            ),
            "endpoint_config_sha256": self.endpoint_config_sha256,
            "model_family_sha256": self.model_family_sha256,
            "canonical_model_state_sha256": (
                self.canonical_model_state_sha256
            ),
            "elbo_inventory_sha256": self.elbo_inventory_sha256,
            "evaluator_implementation_sha256": (
                self.evaluator_implementation_sha256
            ),
            "expectation_identity_sha256": (
                self.expectation_identity_sha256
            ),
            "expectation_structure_sha256": (
                self.expectation_structure_sha256
            ),
            "expectation_source_law_marker_identity_sha256": (
                self.expectation_source_law_marker_identity_sha256
            ),
            "h6_producer_route": self.h6_producer_route,
            "issuer_route": self.issuer_route,
            "h7_adapter_entrypoint": self.h7_adapter_entrypoint,
            "endpoint_language_elbo_sha256": (
                self.endpoint_language_elbo_sha256
            ),
            "source_law_identity_sha256": (
                self.source_law_identity_sha256
            ),
            "source_prior_trace_sha256": (
                self.source_prior_trace_sha256
            ),
            "producer_contract_sha256": self.producer_contract_sha256,
            "ordered_slots": observed_slots,
            "ordered_factor_ids": expected_ids,
            "ordered_factor_values": expected_values,
            "total_value": expected_total,
        }
        expected_hash = h7_owned_sha256(
            H7_RAW_FACTOR_TRACE_HASH_DOMAIN,
            semantic,
        )
        if self.trace_sha256 != expected_hash:
            raise ValueError("trace_sha256 does not match the complete H6 trace")


def require_h7_complete_factor_trace(
    authenticated_evaluation: H7AuthenticatedEvaluation,
) -> CompleteLanguageELBOFactorTrace:
    """Adapt one registered BuiltArm assembly receipt into the H7 raw trace.

    A structurally valid H6 endpoint is insufficient: callers must first use
    :func:`capture_h7_complete_language_elbo` with the exact factory-issued
    ``BuiltArm`` that performed the live assembly.
    """

    receipt = _require_h7_authenticated_evaluation(
        authenticated_evaluation
    )
    return CompleteLanguageELBOFactorTrace._from_authenticated_evaluation(
        receipt
    )


def adapt_h7_raw_factor_trace_evidence(
    trace: CompleteLanguageELBOFactorTrace,
) -> H7RawFactorTraceEvidence:
    """Seal one exact H7 trace into a types-only authenticated evidence record."""

    if type(trace) is not CompleteLanguageELBOFactorTrace:
        raise ValueError(
            "raw trace evidence requires an exact complete H7 trace"
        )
    trace.__post_init__()
    return H7RawFactorTraceEvidence._from_objective_adapter(
        trace_hash_domain=H7_RAW_FACTOR_TRACE_HASH_DOMAIN,
        representation=trace.representation,
        producer_kind=trace.producer_kind,
        attestation_scope=trace.attestation_scope,
        producer_attestation_sha256=(
            trace.producer_attestation_sha256
        ),
        endpoint_config_sha256=trace.endpoint_config_sha256,
        model_family_sha256=trace.model_family_sha256,
        canonical_model_state_sha256=(
            trace.canonical_model_state_sha256
        ),
        elbo_inventory_sha256=trace.elbo_inventory_sha256,
        evaluator_implementation_sha256=(
            trace.evaluator_implementation_sha256
        ),
        expectation_identity_sha256=(
            trace.expectation_identity_sha256
        ),
        expectation_structure_sha256=(
            trace.expectation_structure_sha256
        ),
        expectation_source_law_marker_identity_sha256=(
            trace.expectation_source_law_marker_identity_sha256
        ),
        h6_producer_route=trace.h6_producer_route,
        issuer_route=trace.issuer_route,
        h7_adapter_entrypoint=trace.h7_adapter_entrypoint,
        endpoint_language_elbo_sha256=(
            trace.endpoint_language_elbo_sha256
        ),
        source_law_identity_sha256=trace.source_law_identity_sha256,
        source_prior_trace_sha256=trace.source_prior_trace_sha256,
        producer_contract_sha256=trace.producer_contract_sha256,
        ordered_slots=tuple(
            (term.partition, term.receiver_t)
            for term in trace.source_trace.ordered_factor_terms
        ),
        ordered_factor_ids=trace.ordered_factor_ids,
        ordered_factor_values=trace.ordered_factor_values,
        total_value=trace.total_value,
        trace_sha256=trace.trace_sha256,
    )


__all__ = [
    "CompleteLanguageELBOFactorTrace",
    "ExpectationEvaluationMethod",
    "ExactSourceMixtureLaw",
    "FactorPartition",
    "H7AuthenticatedEvaluation",
    "LanguageElboExpectation",
    "MixtureMode",
    "MomentProjectedLaw",
    "PriorVariant",
    "RecognitionConditioningMode",
    "RecognitionFamily",
    "adapt_h7_raw_factor_trace_evidence",
    "capture_h7_complete_language_elbo",
    "evaluate_emission_only_ablation",
    "require_h7_complete_factor_trace",
    "require_source_law_for_endpoint",
]
