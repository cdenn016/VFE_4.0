"""Typed, fail-closed H6 objective and training-attempt contracts.

This module intentionally contains no corpus access, optimizer construction,
gradient evaluation, or parameter mutation.  Current Task-7 matching reports
are FLOP-incomplete, so the only reachable development behavior is immutable
record construction and refusal before execution.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal, Mapping

import torch

from vfe4.recognition import (
    LanguageRecognitionParameterStore,
    RecognitionConditioning,
)
from vfe4.types.h6 import (
    ArmConfig,
    ArmId,
    EmissionOnlyAblationTerms,
    FlopTerm,
    FrozenTensorSnapshot,
    H6ArmPhaseSchedule,
    H6EndpointLanguageElboTerms,
    H6FactorTerm,
    MatchingReport,
    OptimizerBinding,
    TrainingPhase,
    canonical_json_bytes,
)


ObjectiveKind = Literal[
    "cross_entropy",
    "complete_elbo",
    "emission_only_ablation_non_elbo",
]
_LOWER_HEX = frozenset("0123456789abcdef")
_FULL_PARTITIONS = (
    "initial",
    "state_source",
    "model_source",
    "state_transition",
    "model_transition",
    "emission",
    "entropy",
)
_PARTITIONS_BY_ARM: dict[ArmId, tuple[str, ...]] = {
    ArmId.A1: ("initial", "state_transition", "emission", "entropy"),
    ArmId.A2: _FULL_PARTITIONS,
    ArmId.A3: (
        "initial",
        "state_transition",
        "model_transition",
        "emission",
        "entropy",
    ),
    ArmId.A4: (
        "initial",
        "state_source",
        "state_transition",
        "emission",
        "entropy",
    ),
    ArmId.A5: _FULL_PARTITIONS,
}
_CANONICAL_PER_RECEIVER = (
    "model_source",
    "model_transition",
    "state_source",
    "state_transition",
    "emission",
    "entropy",
)


def _owned_hash(domain: str, payload: object) -> str:
    return hashlib.sha256(
        domain.encode("ascii") + b"\x00" + canonical_json_bytes(payload)
    ).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase 64-hex SHA-256")
    return value


def _require_git_head(value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 40
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise ValueError("source_git_head must be a lowercase 40-hex commit")
    return value


def _tensor_raw_bytes(value: torch.Tensor) -> bytes:
    cpu = value.detach().to(device="cpu").contiguous()
    try:
        return cpu.numpy().tobytes(order="C")
    except (TypeError, RuntimeError):
        return bytes(cpu.view(torch.uint8).reshape(-1).tolist())


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


def _factor_payload(term: H6FactorTerm) -> dict[str, object]:
    term.__post_init__()
    return {
        "partition": term.partition,
        "receiver_t": term.receiver_t,
        "producer_factor_identity_sha256": term.factor_identity_sha256,
        "value": _snapshot_payload(term.value),
    }


def _sum_term_values(terms: tuple[H6FactorTerm, ...]) -> torch.Tensor:
    if not terms:
        raise ValueError("objective terms must be nonempty")
    total = terms[0].value.value()
    for term in terms[1:]:
        total = total + term.value.value()
    return total


def _require_finite_scalar(snapshot: FrozenTensorSnapshot, name: str) -> None:
    snapshot.assert_intact()
    value = snapshot.value()
    if value.ndim != 0 or not bool(torch.isfinite(value).item()):
        raise ValueError(f"{name} must be a finite scalar snapshot")


def _objective_mode(config: ArmConfig) -> ObjectiveKind:
    config.__post_init__()
    if not config.latent_enabled:
        return "cross_entropy"
    if config.objective_kind == "emission_only_ablation_non_elbo":
        return "emission_only_ablation_non_elbo"
    return "complete_elbo"


def _ordered_slots(
    *, partitions: tuple[str, ...], horizon: int
) -> tuple[tuple[str, int], ...]:
    initial = (("initial", 0),) if "initial" in partitions else ()
    per_receiver = tuple(
        (partition, receiver_t)
        for receiver_t in range(1, horizon + 1)
        for partition in _CANONICAL_PER_RECEIVER
        if partition in partitions
    )
    return initial + per_receiver


@dataclass(frozen=True, slots=True)
class ArmObjectiveInventory:
    """Exact live factor inventory for one resolved endpoint config."""

    endpoint_config_sha256: str
    config_id: str
    arm: ArmId
    horizon: int
    latent_enabled: bool
    objective_kind: ObjectiveKind
    partitions: tuple[str, ...]
    ordered_slots: tuple[tuple[str, int], ...]
    inventory_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "endpoint_config_sha256": self.endpoint_config_sha256,
            "config_id": self.config_id,
            "arm": self.arm.value,
            "horizon": self.horizon,
            "latent_enabled": self.latent_enabled,
            "objective_kind": self.objective_kind,
            "partitions": self.partitions,
            "ordered_slots": self.ordered_slots,
        }

    def __post_init__(self) -> None:
        _require_sha256(
            self.endpoint_config_sha256, "endpoint_config_sha256"
        )
        if type(self.config_id) is not str or not self.config_id:
            raise ValueError("config_id must be nonempty")
        if type(self.arm) is not ArmId:
            raise ValueError("arm must be an exact ArmId")
        if type(self.horizon) is not int or self.horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        if type(self.latent_enabled) is not bool:
            raise ValueError("latent_enabled must be a bool")
        if self.objective_kind not in (
            "cross_entropy",
            "complete_elbo",
            "emission_only_ablation_non_elbo",
        ):
            raise ValueError("unsupported objective kind")
        if self.objective_kind in (
            "cross_entropy",
            "emission_only_ablation_non_elbo",
        ):
            expected_partitions = ("emission",)
        else:
            expected_partitions = _PARTITIONS_BY_ARM[self.arm]
        if self.partitions != expected_partitions:
            raise ValueError("partitions do not match the exact endpoint")
        if self.ordered_slots != _ordered_slots(
            partitions=self.partitions, horizon=self.horizon
        ):
            raise ValueError("ordered slots do not match the exact endpoint")
        if self.inventory_sha256 != _owned_hash(
            "vfe4.h6.arm-objective-inventory.v2",
            self.canonical_payload(),
        ):
            raise ValueError("inventory_sha256 is stale")

    @classmethod
    def for_config(cls, *, config: ArmConfig) -> "ArmObjectiveInventory":
        if type(config) is not ArmConfig:
            raise ValueError("config must be an exact ArmConfig")
        config.__post_init__()
        mode = _objective_mode(config)
        partitions = (
            ("emission",)
            if mode != "complete_elbo"
            else _PARTITIONS_BY_ARM[config.arm]
        )
        slots = _ordered_slots(partitions=partitions, horizon=config.horizon)
        payload = {
            "endpoint_config_sha256": config.config_sha256,
            "config_id": config.config_id,
            "arm": config.arm.value,
            "horizon": config.horizon,
            "latent_enabled": config.latent_enabled,
            "objective_kind": mode,
            "partitions": partitions,
            "ordered_slots": slots,
        }
        return cls(
            config.config_sha256,
            config.config_id,
            config.arm,
            config.horizon,
            config.latent_enabled,
            mode,
            partitions,
            slots,
            _owned_hash("vfe4.h6.arm-objective-inventory.v2", payload),
        )


@dataclass(frozen=True, slots=True)
class H6CrossEntropyTerms:
    endpoint_config_sha256: str
    inventory_sha256: str
    ordered_emission_terms: tuple[H6FactorTerm, ...]
    total: FrozenTensorSnapshot
    canonical_sha256: str

    def _payload(self) -> dict[str, object]:
        return {
            "endpoint_config_sha256": self.endpoint_config_sha256,
            "inventory_sha256": self.inventory_sha256,
            "ordered_emission_terms": tuple(
                _factor_payload(term) for term in self.ordered_emission_terms
            ),
            "total": _snapshot_payload(self.total),
        }

    def __post_init__(self) -> None:
        _require_sha256(
            self.endpoint_config_sha256, "endpoint_config_sha256"
        )
        _require_sha256(self.inventory_sha256, "inventory_sha256")
        if (
            type(self.ordered_emission_terms) is not tuple
            or not self.ordered_emission_terms
            or any(
                type(term) is not H6FactorTerm
                or term.partition != "emission"
                for term in self.ordered_emission_terms
            )
        ):
            raise ValueError("cross entropy accepts emission terms only")
        _require_finite_scalar(self.total, "cross-entropy total")
        if not torch.equal(
            _sum_term_values(self.ordered_emission_terms), self.total.value()
        ):
            raise ValueError("cross-entropy terms do not equal their total")
        if self.canonical_sha256 != _owned_hash(
            "vfe4.h6.cross-entropy-terms.v1", self._payload()
        ):
            raise ValueError("cross-entropy objective identity is stale")

    @classmethod
    def create(
        cls,
        *,
        config: ArmConfig,
        inventory: ArmObjectiveInventory,
        ordered_emission_terms: tuple[H6FactorTerm, ...],
    ) -> "H6CrossEntropyTerms":
        if _objective_mode(config) != "cross_entropy":
            raise ValueError("endpoint does not use cross entropy")
        if inventory != ArmObjectiveInventory.for_config(config=config):
            raise ValueError("inventory does not match endpoint config")
        terms = tuple(ordered_emission_terms)
        total = FrozenTensorSnapshot.capture(_sum_term_values(terms))
        provisional = object.__new__(cls)
        object.__setattr__(provisional, "endpoint_config_sha256", config.config_sha256)
        object.__setattr__(provisional, "inventory_sha256", inventory.inventory_sha256)
        object.__setattr__(provisional, "ordered_emission_terms", terms)
        object.__setattr__(provisional, "total", total)
        return cls(
            config.config_sha256,
            inventory.inventory_sha256,
            terms,
            total,
            _owned_hash("vfe4.h6.cross-entropy-terms.v1", provisional._payload()),
        )


@dataclass(frozen=True, slots=True)
class H6ReducedLanguageElboTerms:
    endpoint_config_sha256: str
    inventory_sha256: str
    ordered_factor_terms: tuple[H6FactorTerm, ...]
    total: FrozenTensorSnapshot
    canonical_sha256: str

    def _payload(self) -> dict[str, object]:
        return {
            "endpoint_config_sha256": self.endpoint_config_sha256,
            "inventory_sha256": self.inventory_sha256,
            "ordered_factor_terms": tuple(
                _factor_payload(term) for term in self.ordered_factor_terms
            ),
            "total": _snapshot_payload(self.total),
        }

    def __post_init__(self) -> None:
        _require_sha256(
            self.endpoint_config_sha256, "endpoint_config_sha256"
        )
        _require_sha256(self.inventory_sha256, "inventory_sha256")
        if (
            type(self.ordered_factor_terms) is not tuple
            or not self.ordered_factor_terms
            or any(type(term) is not H6FactorTerm for term in self.ordered_factor_terms)
        ):
            raise ValueError("reduced ELBO requires exact factor terms")
        if len(
            {term.factor_identity_sha256 for term in self.ordered_factor_terms}
        ) != len(self.ordered_factor_terms):
            raise ValueError("reduced ELBO factor identities must be unique")
        _require_finite_scalar(self.total, "reduced ELBO total")
        if not torch.equal(
            _sum_term_values(self.ordered_factor_terms), self.total.value()
        ):
            raise ValueError("reduced ELBO terms do not equal their total")
        if self.canonical_sha256 != _owned_hash(
            "vfe4.h6.reduced-language-elbo.v1", self._payload()
        ):
            raise ValueError("reduced ELBO objective identity is stale")

    @classmethod
    def create(
        cls,
        *,
        config: ArmConfig,
        inventory: ArmObjectiveInventory,
        ordered_factor_terms: tuple[H6FactorTerm, ...],
    ) -> "H6ReducedLanguageElboTerms":
        if config.arm not in (ArmId.A1, ArmId.A3, ArmId.A4):
            raise ValueError("reduced ELBO is only for A1, A3, or A4")
        if inventory != ArmObjectiveInventory.for_config(config=config):
            raise ValueError("inventory does not match endpoint config")
        terms = tuple(ordered_factor_terms)
        slots = tuple((term.partition, term.receiver_t) for term in terms)
        if slots != inventory.ordered_slots:
            raise ValueError("reduced ELBO factor slots are not exact")
        total = FrozenTensorSnapshot.capture(_sum_term_values(terms))
        provisional = object.__new__(cls)
        object.__setattr__(provisional, "endpoint_config_sha256", config.config_sha256)
        object.__setattr__(provisional, "inventory_sha256", inventory.inventory_sha256)
        object.__setattr__(provisional, "ordered_factor_terms", terms)
        object.__setattr__(provisional, "total", total)
        return cls(
            config.config_sha256,
            inventory.inventory_sha256,
            terms,
            total,
            _owned_hash("vfe4.h6.reduced-language-elbo.v1", provisional._payload()),
        )


H6TypedTrainingObjective = (
    H6CrossEntropyTerms
    | H6ReducedLanguageElboTerms
    | H6EndpointLanguageElboTerms
    | EmissionOnlyAblationTerms
)


@dataclass(frozen=True, slots=True)
class ArmTrainingObjectiveAdapter:
    endpoint_config_sha256: str
    config_id: str
    inventory: ArmObjectiveInventory
    phase_schedule: H6ArmPhaseSchedule
    phases: tuple[TrainingPhase, ...]
    adapter_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "endpoint_config_sha256": self.endpoint_config_sha256,
            "config_id": self.config_id,
            "inventory_sha256": self.inventory.inventory_sha256,
            "phase_schedule_sha256": self.phase_schedule.phase_schedule_sha256,
            "phases": tuple(phase.value for phase in self.phases),
        }

    def __post_init__(self) -> None:
        _require_sha256(
            self.endpoint_config_sha256, "endpoint_config_sha256"
        )
        self.inventory.__post_init__()
        self.phase_schedule.__post_init__()
        if (
            self.inventory.endpoint_config_sha256 != self.endpoint_config_sha256
            or self.phase_schedule.endpoint_config_sha256
            != self.endpoint_config_sha256
            or self.config_id != self.inventory.config_id
            or self.phases != self.phase_schedule.phases
            or self.phase_schedule.latent_enabled
            != self.inventory.latent_enabled
        ):
            raise ValueError("objective adapter endpoint bindings are inconsistent")
        if self.adapter_sha256 != _owned_hash(
            "vfe4.h6.arm-objective-adapter.v2", self.canonical_payload()
        ):
            raise ValueError("adapter_sha256 is stale")

    @classmethod
    def create(
        cls,
        *,
        config: ArmConfig,
        inventory: ArmObjectiveInventory,
        phase_schedule: H6ArmPhaseSchedule,
    ) -> "ArmTrainingObjectiveAdapter":
        if type(config) is not ArmConfig:
            raise ValueError("config must be an exact ArmConfig")
        config.__post_init__()
        if inventory != ArmObjectiveInventory.for_config(config=config):
            raise ValueError("inventory does not match endpoint config")
        if phase_schedule.endpoint_config_sha256 != config.config_sha256:
            raise ValueError("phase schedule does not match endpoint config")
        payload = {
            "endpoint_config_sha256": config.config_sha256,
            "config_id": config.config_id,
            "inventory_sha256": inventory.inventory_sha256,
            "phase_schedule_sha256": phase_schedule.phase_schedule_sha256,
            "phases": tuple(phase.value for phase in phase_schedule.phases),
        }
        return cls(
            config.config_sha256,
            config.config_id,
            inventory,
            phase_schedule,
            tuple(phase_schedule.phases),
            _owned_hash("vfe4.h6.arm-objective-adapter.v2", payload),
        )

    def validate_objective(
        self, objective: H6TypedTrainingObjective
    ) -> tuple[tuple[H6FactorTerm, ...], str, FrozenTensorSnapshot]:
        self.__post_init__()
        kind = self.inventory.objective_kind
        if kind == "cross_entropy":
            if type(objective) is not H6CrossEntropyTerms:
                raise ValueError("endpoint requires typed cross-entropy terms")
            objective.__post_init__()
            if (
                objective.endpoint_config_sha256 != self.endpoint_config_sha256
                or objective.inventory_sha256 != self.inventory.inventory_sha256
            ):
                raise ValueError("cross-entropy objective belongs to another endpoint")
            terms = objective.ordered_emission_terms
            producer_sha = objective.canonical_sha256
            total = objective.total
        elif kind == "emission_only_ablation_non_elbo":
            if type(objective) is not EmissionOnlyAblationTerms:
                raise ValueError("endpoint requires emission-only ablation terms")
            objective.__post_init__()
            terms = objective.ordered_emission_terms
            producer_sha = objective.canonical_sha256
            total = objective.total
        elif self.inventory.arm in (ArmId.A1, ArmId.A3, ArmId.A4):
            if type(objective) is not H6ReducedLanguageElboTerms:
                raise ValueError("endpoint requires a typed reduced ELBO")
            objective.__post_init__()
            if (
                objective.endpoint_config_sha256 != self.endpoint_config_sha256
                or objective.inventory_sha256 != self.inventory.inventory_sha256
            ):
                raise ValueError("reduced ELBO belongs to another endpoint")
            terms = objective.ordered_factor_terms
            producer_sha = objective.canonical_sha256
            total = objective.total
        else:
            if type(objective) is not H6EndpointLanguageElboTerms:
                raise ValueError(
                    "endpoint requires an endpoint-bound complete typed ELBO"
                )
            objective.__post_init__()
            if (
                objective.endpoint_config_sha256
                != self.endpoint_config_sha256
                or objective.endpoint_config.config_id != self.config_id
                or objective.source_prior_trace.prior_variant
                != objective.endpoint_config.prior_variant
                or objective.source_prior_trace.prior_type
                != (
                    "FixedSourcePrior"
                    if objective.endpoint_config.prior_variant == "fixed"
                    else "PrefixConditionedSourcePrior"
                )
            ):
                raise ValueError(
                    "complete ELBO source treatment or live-prior trace belongs "
                    "to another endpoint"
                )
            terms = objective.ordered_factor_terms
            producer_sha = objective.canonical_sha256
            total = objective.total_language_elbo
        slots = tuple((term.partition, term.receiver_t) for term in terms)
        if slots != self.inventory.ordered_slots:
            raise ValueError(
                "objective factors are absent, extra, reordered, or wrong-horizon"
            )
        bindings = tuple(
            _owned_hash(
                "vfe4.h6.endpoint-factor-binding.v1",
                {
                    "endpoint_config_sha256": self.endpoint_config_sha256,
                    "inventory_sha256": self.inventory.inventory_sha256,
                    "factor": _factor_payload(term),
                },
            )
            for term in terms
        )
        if len(set(bindings)) != len(bindings):
            raise ValueError("endpoint factor bindings must be unique")
        _require_finite_scalar(total, "objective total")
        return terms, producer_sha, total


@dataclass(frozen=True, slots=True, init=False)
class _DetachedTensor:
    __owned: torch.Tensor
    name: str
    dtype: str
    shape: tuple[int, ...]
    device: str
    raw_bytes_sha256: str
    storage_version: int

    @classmethod
    def capture(cls, *, name: str, value: torch.Tensor) -> "_DetachedTensor":
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise ValueError("detached recognition values must be floating tensors")
        if not bool(torch.isfinite(value.detach()).all().item()):
            raise ValueError("detached recognition values must be finite")
        owned = value.detach().contiguous().clone()
        owned.requires_grad_(False)
        instance = object.__new__(cls)
        object.__setattr__(instance, "_DetachedTensor__owned", owned)
        object.__setattr__(instance, "name", name)
        object.__setattr__(instance, "dtype", str(owned.dtype).removeprefix("torch."))
        object.__setattr__(instance, "shape", tuple(int(size) for size in owned.shape))
        object.__setattr__(instance, "device", str(owned.device))
        object.__setattr__(instance, "raw_bytes_sha256", hashlib.sha256(_tensor_raw_bytes(owned)).hexdigest())
        object.__setattr__(instance, "storage_version", int(owned._version))
        instance.assert_intact()
        return instance

    def assert_intact(self) -> None:
        value = self.__owned
        if (
            not isinstance(value, torch.Tensor)
            or value.requires_grad
            or value.grad_fn is not None
            or not value.is_contiguous()
            or str(value.dtype).removeprefix("torch.") != self.dtype
            or tuple(int(size) for size in value.shape) != self.shape
            or str(value.device) != self.device
            or int(value._version) != self.storage_version
            or hashlib.sha256(_tensor_raw_bytes(value)).hexdigest()
            != self.raw_bytes_sha256
        ):
            raise ValueError("detached recognition tensor integrity failed")

    def tensor(self) -> torch.Tensor:
        self.assert_intact()
        return self.__owned.clone()

    def payload(self) -> dict[str, object]:
        self.assert_intact()
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": self.shape,
            "device": self.device,
            "requires_grad": False,
            "raw_bytes_sha256": self.raw_bytes_sha256,
        }


def _parameter_store_state_sha256(
    config: ArmConfig, store: LanguageRecognitionParameterStore
) -> str:
    records = []
    for name, parameter in store.named_parameters(remove_duplicate=False):
        records.append(
            {
                "name": name,
                "dtype": str(parameter.dtype).removeprefix("torch."),
                "shape": tuple(int(size) for size in parameter.shape),
                "raw_bytes_sha256": hashlib.sha256(
                    _tensor_raw_bytes(parameter)
                ).hexdigest(),
            }
        )
    if not records or len({record["name"] for record in records}) != len(records):
        raise ValueError("recognition parameter-store inventory is invalid")
    return _owned_hash(
        "vfe4.h6.recognition-parameter-store-state.v1",
        {
            "endpoint_config_sha256": config.config_sha256,
            "parameters": tuple(records),
        },
    )


@dataclass(frozen=True, slots=True, init=False)
class DetachedRecognitionLawSnapshot:
    endpoint_config_sha256: str
    family: str
    conditioning: Literal["filtering", "smoothing"]
    parameter_store_state_sha256: str
    tensor_metadata: tuple[tuple[str, str, tuple[int, ...], str, str], ...]
    snapshot_sha256: str
    __tensors: tuple[_DetachedTensor, ...]

    @classmethod
    def capture_from_store(
        cls,
        *,
        config: ArmConfig,
        parameter_store: LanguageRecognitionParameterStore,
        conditioning: RecognitionConditioning,
    ) -> "DetachedRecognitionLawSnapshot":
        if type(config) is not ArmConfig or not config.latent_enabled:
            raise ValueError("detached recognition requires a latent ArmConfig")
        config.__post_init__()
        if type(parameter_store) is not LanguageRecognitionParameterStore:
            raise ValueError("parameter_store must be exact")
        if type(conditioning) is not RecognitionConditioning:
            raise ValueError("conditioning must be exact")
        conditioning.__post_init__()
        expected_channels = 2 if config.model_channel_enabled else 1
        if (
            parameter_store.vocabulary != config.vocabulary
            or parameter_store.horizon != config.horizon
            or parameter_store.latent_width != config.capacity_allocation.latent_width
            or parameter_store.recognition_width
            != config.capacity_allocation.recognition_width
            or parameter_store.channel_count != expected_channels
            or parameter_store.family != config.recognition_family
            or parameter_store.conditioning_mode != config.recognition_conditioning
            or conditioning.mode != config.recognition_conditioning
            or conditioning.horizon != config.horizon
        ):
            raise ValueError("recognition store/conditioning does not match endpoint")
        store_sha = _parameter_store_state_sha256(config, parameter_store)
        law = parameter_store.recognition_law(conditioning)
        law.__post_init__()
        owned = (
            _DetachedTensor.capture(name="mean", value=law.mean.value()),
            _DetachedTensor.capture(
                name="precision_cholesky",
                value=law.precision_cholesky.value(),
            ),
        )
        metadata = tuple(
            (item.name, item.dtype, item.shape, item.device, item.raw_bytes_sha256)
            for item in owned
        )
        payload = {
            "endpoint_config_sha256": config.config_sha256,
            "family": config.recognition_family,
            "conditioning": config.recognition_conditioning,
            "parameter_store_state_sha256": store_sha,
            "tensors": tuple(item.payload() for item in owned),
        }
        instance = object.__new__(cls)
        object.__setattr__(instance, "endpoint_config_sha256", config.config_sha256)
        object.__setattr__(instance, "family", config.recognition_family)
        object.__setattr__(instance, "conditioning", config.recognition_conditioning)
        object.__setattr__(instance, "parameter_store_state_sha256", store_sha)
        object.__setattr__(instance, "tensor_metadata", metadata)
        object.__setattr__(instance, "snapshot_sha256", _owned_hash("vfe4.h6.detached-recognition-law.v2", payload))
        object.__setattr__(instance, "_DetachedRecognitionLawSnapshot__tensors", owned)
        instance.assert_intact()
        return instance

    def assert_intact(self) -> None:
        _require_sha256(
            self.endpoint_config_sha256, "endpoint_config_sha256"
        )
        _require_sha256(
            self.parameter_store_state_sha256,
            "parameter_store_state_sha256",
        )
        if self.family not in ("structured", "factorized"):
            raise ValueError("unsupported detached recognition family")
        if self.conditioning not in ("filtering", "smoothing"):
            raise ValueError("unsupported detached recognition conditioning")
        if tuple(item.name for item in self.__tensors) != (
            "mean",
            "precision_cholesky",
        ):
            raise ValueError("detached recognition tensor inventory changed")
        for item in self.__tensors:
            item.assert_intact()
        metadata = tuple(
            (item.name, item.dtype, item.shape, item.device, item.raw_bytes_sha256)
            for item in self.__tensors
        )
        if metadata != self.tensor_metadata:
            raise ValueError("detached recognition metadata changed")
        payload = {
            "endpoint_config_sha256": self.endpoint_config_sha256,
            "family": self.family,
            "conditioning": self.conditioning,
            "parameter_store_state_sha256": self.parameter_store_state_sha256,
            "tensors": tuple(item.payload() for item in self.__tensors),
        }
        if self.snapshot_sha256 != _owned_hash(
            "vfe4.h6.detached-recognition-law.v2", payload
        ):
            raise ValueError("detached recognition snapshot identity is stale")

    def tensor(self, name: str) -> torch.Tensor:
        self.assert_intact()
        for item in self.__tensors:
            if item.name == name:
                return item.tensor()
        raise KeyError(name)


@dataclass(frozen=True, slots=True)
class H6TrainingAuthorization:
    matching_report_sha256: str
    matching_config_sha256: str
    common_schedule_sha256: str
    optimizer_policy_sha256: str
    flop_term_sha256s: tuple[str, ...]
    optimizer_binding_sha256s: tuple[str, ...]
    endpoint_training_flops: int
    authorization_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "matching_report_sha256": self.matching_report_sha256,
            "matching_config_sha256": self.matching_config_sha256,
            "common_schedule_sha256": self.common_schedule_sha256,
            "optimizer_policy_sha256": self.optimizer_policy_sha256,
            "flop_term_sha256s": self.flop_term_sha256s,
            "optimizer_binding_sha256s": self.optimizer_binding_sha256s,
            "endpoint_training_flops": self.endpoint_training_flops,
        }

    def __post_init__(self) -> None:
        for name in (
            "matching_report_sha256",
            "matching_config_sha256",
            "common_schedule_sha256",
            "optimizer_policy_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if not self.flop_term_sha256s or not self.optimizer_binding_sha256s:
            raise ValueError("training authorization ledgers must be nonempty")
        for digest in self.flop_term_sha256s + self.optimizer_binding_sha256s:
            _require_sha256(digest, "ledger digest")
        if type(self.endpoint_training_flops) is not int or self.endpoint_training_flops <= 0:
            raise ValueError("endpoint_training_flops must be positive")
        if self.authorization_sha256 != _owned_hash(
            "vfe4.h6.training-authorization.v1", self.canonical_payload()
        ):
            raise ValueError("training authorization identity is stale")

    @classmethod
    def create(
        cls,
        *,
        matching_report: MatchingReport,
        matching_config_sha256: str,
        common_schedule_sha256: str,
        optimizer_policy_sha256: str,
        flop_terms: tuple[FlopTerm, ...],
        optimizer_bindings: tuple[OptimizerBinding, ...],
    ) -> "H6TrainingAuthorization":
        _require_eligible_matching_report(matching_report)
        if (
            matching_report.matching_config_sha256 != matching_config_sha256
            or matching_report.common_schedule_sha256 != common_schedule_sha256
        ):
            raise ValueError("matching authorities are not current")
        if (
            type(flop_terms) is not tuple
            or not flop_terms
            or any(type(term) is not FlopTerm for term in flop_terms)
            or type(optimizer_bindings) is not tuple
            or not optimizer_bindings
            or any(type(item) is not OptimizerBinding for item in optimizer_bindings)
        ):
            raise ValueError("authorization requires exact nonempty ledgers")
        for term in flop_terms:
            term.__post_init__()
        for binding in optimizer_bindings:
            binding.__post_init__()
            if binding.optimizer_policy_sha256 != optimizer_policy_sha256:
                raise ValueError("optimizer binding policy is stale")
        counted_flops = sum(term.total_arithmetic_flops for term in flop_terms)
        if counted_flops != matching_report.endpoint_training_flops:
            raise ValueError("operator FLOP ledger does not match report total")
        values = {
            "matching_report_sha256": matching_report.report_sha256,
            "matching_config_sha256": matching_config_sha256,
            "common_schedule_sha256": common_schedule_sha256,
            "optimizer_policy_sha256": optimizer_policy_sha256,
            "flop_term_sha256s": tuple(term.term_sha256 for term in flop_terms),
            "optimizer_binding_sha256s": tuple(
                item.binding_sha256 for item in optimizer_bindings
            ),
            "endpoint_training_flops": counted_flops,
        }
        return cls(
            **values,
            authorization_sha256=_owned_hash(
                "vfe4.h6.training-authorization.v1", values
            ),
        )


@dataclass(frozen=True, slots=True)
class H6AttemptSpec:
    source_git_head: str
    dirty_digest: str
    readiness_sha256: str
    arm: ArmId
    config_id: str
    endpoint_config_sha256: str
    latent_enabled: bool
    objective_kind: ObjectiveKind
    factory_sha256: str
    model_family_sha256: str
    training_authorization_sha256: str
    objective_inventory_sha256: str
    objective_adapter_sha256: str
    h5_binding_sha256: str
    outer_schedule_sha256: str
    phase_schedule_sha256: str
    optimizer_policy_sha256: str
    tuning_cell_sha256: str
    training_seed: int
    data_identity_sha256: str
    window_manifest_sha256: str
    batch_schedule_sha256: str
    estimator_sha256: str
    prefix_certificate_sha256: str
    attempt_spec_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "source_git_head": self.source_git_head,
            "dirty_digest": self.dirty_digest,
            "readiness_sha256": self.readiness_sha256,
            "arm": self.arm.value,
            "config_id": self.config_id,
            "endpoint_config_sha256": self.endpoint_config_sha256,
            "latent_enabled": self.latent_enabled,
            "objective_kind": self.objective_kind,
            "factory_sha256": self.factory_sha256,
            "model_family_sha256": self.model_family_sha256,
            "training_authorization_sha256": self.training_authorization_sha256,
            "objective_inventory_sha256": self.objective_inventory_sha256,
            "objective_adapter_sha256": self.objective_adapter_sha256,
            "h5_binding_sha256": self.h5_binding_sha256,
            "outer_schedule_sha256": self.outer_schedule_sha256,
            "phase_schedule_sha256": self.phase_schedule_sha256,
            "optimizer_policy_sha256": self.optimizer_policy_sha256,
            "tuning_cell_sha256": self.tuning_cell_sha256,
            "training_seed": self.training_seed,
            "data_identity_sha256": self.data_identity_sha256,
            "window_manifest_sha256": self.window_manifest_sha256,
            "batch_schedule_sha256": self.batch_schedule_sha256,
            "estimator_sha256": self.estimator_sha256,
            "prefix_certificate_sha256": self.prefix_certificate_sha256,
        }

    def __post_init__(self) -> None:
        _require_git_head(self.source_git_head)
        if type(self.arm) is not ArmId:
            raise ValueError("arm must be an exact ArmId")
        if type(self.config_id) is not str or not self.config_id:
            raise ValueError("config_id must be nonempty")
        if type(self.latent_enabled) is not bool:
            raise ValueError("latent_enabled must be a bool")
        if self.objective_kind not in (
            "cross_entropy",
            "complete_elbo",
            "emission_only_ablation_non_elbo",
        ):
            raise ValueError("unsupported attempt objective kind")
        if type(self.training_seed) is not int or self.training_seed < 0:
            raise ValueError("training_seed must be nonnegative")
        for name, value in self.canonical_payload().items():
            if name in (
                "source_git_head",
                "arm",
                "config_id",
                "latent_enabled",
                "objective_kind",
                "training_seed",
            ):
                continue
            _require_sha256(value, name)
        if self.attempt_spec_sha256 != _owned_hash(
            "vfe4.h6.attempt-spec.v2", self.canonical_payload()
        ):
            raise ValueError("attempt_spec_sha256 is stale")

    @classmethod
    def create(cls, **values: object) -> "H6AttemptSpec":
        payload = dict(values)
        arm = payload.get("arm")
        if type(arm) is ArmId:
            payload["arm"] = arm.value
        return cls(
            **values,  # type: ignore[arg-type]
            attempt_spec_sha256=_owned_hash(
                "vfe4.h6.attempt-spec.v2", payload
            ),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "H6AttemptSpec":
        expected = set(cls.__dataclass_fields__) - {"attempt_spec_sha256"}
        if type(payload) is not dict or set(payload) != expected:
            raise ValueError("attempt spec payload fields are not exact")
        values = dict(payload)
        try:
            values["arm"] = ArmId(values["arm"])
        except (TypeError, ValueError) as exc:
            raise ValueError("attempt spec arm is invalid") from exc
        return cls.create(**values)


@dataclass(frozen=True, slots=True)
class H6AttemptCursor:
    attempt_spec_sha256: str
    phase_schedule_sha256: str
    latent_enabled: bool
    phases: tuple[TrainingPhase, ...]
    pass_index: int
    batch_index: int
    next_phase: TrainingPhase
    model_update_count: int
    recognition_update_count: int
    validation_boundary_count: int
    checkpoint_boundary_count: int
    data_cursor_sha256: str
    rng_state_sha256: str
    cursor_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "attempt_spec_sha256": self.attempt_spec_sha256,
            "phase_schedule_sha256": self.phase_schedule_sha256,
            "latent_enabled": self.latent_enabled,
            "phases": tuple(phase.value for phase in self.phases),
            "pass_index": self.pass_index,
            "batch_index": self.batch_index,
            "next_phase": self.next_phase.value,
            "model_update_count": self.model_update_count,
            "recognition_update_count": self.recognition_update_count,
            "validation_boundary_count": self.validation_boundary_count,
            "checkpoint_boundary_count": self.checkpoint_boundary_count,
            "data_cursor_sha256": self.data_cursor_sha256,
            "rng_state_sha256": self.rng_state_sha256,
        }

    def __post_init__(self) -> None:
        _require_sha256(self.attempt_spec_sha256, "attempt_spec_sha256")
        _require_sha256(self.phase_schedule_sha256, "phase_schedule_sha256")
        _require_sha256(self.data_cursor_sha256, "data_cursor_sha256")
        _require_sha256(self.rng_state_sha256, "rng_state_sha256")
        if type(self.latent_enabled) is not bool:
            raise ValueError("latent_enabled must be a bool")
        expected_phases = (
            (
                TrainingPhase.RECOGNITION_ADAMW,
                TrainingPhase.IMMUTABLE_DETACHED_SNAPSHOT,
                TrainingPhase.MODEL_ADAMW,
            )
            if self.latent_enabled
            else (TrainingPhase.MODEL_CE_ADAMW,)
        )
        if self.phases != expected_phases or self.next_phase not in self.phases:
            raise ValueError("cursor phase is impossible for its endpoint")
        for name in (
            "pass_index",
            "batch_index",
            "model_update_count",
            "recognition_update_count",
            "validation_boundary_count",
            "checkpoint_boundary_count",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not self.latent_enabled:
            if self.recognition_update_count != 0:
                raise ValueError("no-latent cursor cannot have recognition updates")
        elif self.next_phase is TrainingPhase.RECOGNITION_ADAMW:
            if self.recognition_update_count != self.model_update_count:
                raise ValueError("recognition phase cursor would replay or skip")
        elif self.recognition_update_count != self.model_update_count + 1:
            raise ValueError("latent phase cursor update counts are inconsistent")
        if self.cursor_sha256 != _owned_hash(
            "vfe4.h6.attempt-cursor.v2", self.canonical_payload()
        ):
            raise ValueError("cursor_sha256 is stale")

    @classmethod
    def initial(
        cls,
        *,
        attempt_spec: H6AttemptSpec,
        phase_schedule: H6ArmPhaseSchedule,
        data_cursor_sha256: str,
        rng_state_sha256: str,
    ) -> "H6AttemptCursor":
        attempt_spec.__post_init__()
        phase_schedule.__post_init__()
        if (
            attempt_spec.phase_schedule_sha256
            != phase_schedule.phase_schedule_sha256
            or attempt_spec.endpoint_config_sha256
            != phase_schedule.endpoint_config_sha256
            or attempt_spec.latent_enabled != phase_schedule.latent_enabled
        ):
            raise ValueError("attempt and phase schedule do not match")
        values: dict[str, object] = {
            "attempt_spec_sha256": attempt_spec.attempt_spec_sha256,
            "phase_schedule_sha256": phase_schedule.phase_schedule_sha256,
            "latent_enabled": phase_schedule.latent_enabled,
            "phases": tuple(phase_schedule.phases),
            "pass_index": 0,
            "batch_index": 0,
            "next_phase": phase_schedule.phases[0],
            "model_update_count": 0,
            "recognition_update_count": 0,
            "validation_boundary_count": 0,
            "checkpoint_boundary_count": 0,
            "data_cursor_sha256": data_cursor_sha256,
            "rng_state_sha256": rng_state_sha256,
        }
        payload = {
            **values,
            "phases": tuple(phase.value for phase in phase_schedule.phases),
            "next_phase": phase_schedule.phases[0].value,
        }
        return cls(
            **values,  # type: ignore[arg-type]
            cursor_sha256=_owned_hash("vfe4.h6.attempt-cursor.v2", payload),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "H6AttemptCursor":
        expected = set(cls.__dataclass_fields__) - {"cursor_sha256"}
        if type(payload) is not dict or set(payload) != expected:
            raise ValueError("attempt cursor payload fields are not exact")
        values = dict(payload)
        try:
            values["phases"] = tuple(TrainingPhase(item) for item in values["phases"])
            values["next_phase"] = TrainingPhase(values["next_phase"])
        except (TypeError, ValueError) as exc:
            raise ValueError("cursor phases are invalid") from exc
        canonical = dict(payload)
        return cls(
            **values,  # type: ignore[arg-type]
            cursor_sha256=_owned_hash("vfe4.h6.attempt-cursor.v2", canonical),
        )


@dataclass(frozen=True, slots=True)
class H6ObjectiveManifest:
    attempt_spec_sha256: str
    endpoint_config_sha256: str
    inventory_sha256: str
    adapter_sha256: str
    objective_kind: ObjectiveKind
    producer_objective_sha256: str
    ordered_factor_bindings: tuple[tuple[str, int, str], ...]
    total_raw_bytes_sha256: str
    detached_recognition_snapshot_sha256: str | None
    objective_manifest_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "attempt_spec_sha256": self.attempt_spec_sha256,
            "endpoint_config_sha256": self.endpoint_config_sha256,
            "inventory_sha256": self.inventory_sha256,
            "adapter_sha256": self.adapter_sha256,
            "objective_kind": self.objective_kind,
            "producer_objective_sha256": self.producer_objective_sha256,
            "ordered_factor_bindings": self.ordered_factor_bindings,
            "total_raw_bytes_sha256": self.total_raw_bytes_sha256,
            "detached_recognition_snapshot_sha256": (
                self.detached_recognition_snapshot_sha256
            ),
        }

    def __post_init__(self) -> None:
        for name in (
            "attempt_spec_sha256",
            "endpoint_config_sha256",
            "inventory_sha256",
            "adapter_sha256",
            "producer_objective_sha256",
            "total_raw_bytes_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if self.detached_recognition_snapshot_sha256 is not None:
            _require_sha256(
                self.detached_recognition_snapshot_sha256,
                "detached_recognition_snapshot_sha256",
            )
        if self.objective_kind not in (
            "cross_entropy",
            "complete_elbo",
            "emission_only_ablation_non_elbo",
        ):
            raise ValueError("unsupported objective manifest kind")
        if not self.ordered_factor_bindings:
            raise ValueError("objective manifest factors must be nonempty")
        for partition, receiver_t, digest in self.ordered_factor_bindings:
            if type(partition) is not str or not partition:
                raise ValueError("factor partition must be nonempty")
            if type(receiver_t) is not int or receiver_t < 0:
                raise ValueError("factor receiver must be nonnegative")
            _require_sha256(digest, "factor binding digest")
        if self.objective_manifest_sha256 != _owned_hash(
            "vfe4.h6.objective-manifest.v2", self.canonical_payload()
        ):
            raise ValueError("objective manifest identity is stale")

    @classmethod
    def capture(
        cls,
        *,
        attempt_spec: H6AttemptSpec,
        adapter: ArmTrainingObjectiveAdapter,
        objective: H6TypedTrainingObjective,
        detached_snapshot: DetachedRecognitionLawSnapshot | None,
    ) -> "H6ObjectiveManifest":
        attempt_spec.__post_init__()
        adapter.__post_init__()
        if (
            attempt_spec.endpoint_config_sha256
            != adapter.endpoint_config_sha256
            or attempt_spec.objective_inventory_sha256
            != adapter.inventory.inventory_sha256
            or attempt_spec.objective_adapter_sha256 != adapter.adapter_sha256
        ):
            raise ValueError("attempt and objective adapter identities differ")
        terms, producer_sha, total = adapter.validate_objective(objective)
        if adapter.inventory.latent_enabled:
            if type(detached_snapshot) is not DetachedRecognitionLawSnapshot:
                raise ValueError("latent objective requires detached recognition")
            detached_snapshot.assert_intact()
            if (
                detached_snapshot.endpoint_config_sha256
                != adapter.endpoint_config_sha256
            ):
                raise ValueError("detached recognition belongs to another endpoint")
            detached_sha: str | None = detached_snapshot.snapshot_sha256
        else:
            if detached_snapshot is not None:
                raise ValueError("no-latent objective cannot carry recognition")
            detached_sha = None
        bindings = tuple(
            (
                term.partition,
                term.receiver_t,
                _owned_hash(
                    "vfe4.h6.endpoint-factor-binding.v1",
                    {
                        "endpoint_config_sha256": adapter.endpoint_config_sha256,
                        "inventory_sha256": adapter.inventory.inventory_sha256,
                        "factor": _factor_payload(term),
                    },
                ),
            )
            for term in terms
        )
        values = {
            "attempt_spec_sha256": attempt_spec.attempt_spec_sha256,
            "endpoint_config_sha256": adapter.endpoint_config_sha256,
            "inventory_sha256": adapter.inventory.inventory_sha256,
            "adapter_sha256": adapter.adapter_sha256,
            "objective_kind": adapter.inventory.objective_kind,
            "producer_objective_sha256": producer_sha,
            "ordered_factor_bindings": bindings,
            "total_raw_bytes_sha256": total.raw_bytes_sha256,
            "detached_recognition_snapshot_sha256": detached_sha,
        }
        return cls(
            **values,
            objective_manifest_sha256=_owned_hash(
                "vfe4.h6.objective-manifest.v2", values
            ),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "H6ObjectiveManifest":
        expected = set(cls.__dataclass_fields__) - {"objective_manifest_sha256"}
        if type(payload) is not dict or set(payload) != expected:
            raise ValueError("objective manifest payload fields are not exact")
        values = dict(payload)
        try:
            values["ordered_factor_bindings"] = tuple(
                (str(item[0]), int(item[1]), str(item[2]))
                for item in values["ordered_factor_bindings"]
            )
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError("objective factor bindings are invalid") from exc
        canonical = dict(payload)
        canonical["ordered_factor_bindings"] = values["ordered_factor_bindings"]
        return cls(
            **values,  # type: ignore[arg-type]
            objective_manifest_sha256=_owned_hash(
                "vfe4.h6.objective-manifest.v2", canonical
            ),
        )


def _require_eligible_matching_report(report: MatchingReport) -> None:
    if type(report) is not MatchingReport:
        raise ValueError("matching_report must be exact")
    report.__post_init__()
    if report.status != "ELIGIBLE" or report.eligible is not True:
        raise ValueError(
            f"matching report is {report.status}; training requires ELIGIBLE "
            "status and a complete FLOP proof"
        )
    if (
        report.training_flop_ledger_complete is not True
        or report.training_flop_obligations
        or report.ownership_valid is not True
        or report.common_schedule is not True
        or report.optimizer_policy_match is not True
        or report.obligations
    ):
        raise ValueError("training matching evidence is not closed")


def plan_h6_attempt(
    *,
    config: ArmConfig,
    objective_adapter: ArmTrainingObjectiveAdapter,
    phase_schedule: H6ArmPhaseSchedule,
    matching_report: MatchingReport,
    training_authorization: H6TrainingAuthorization | None = None,
    attempt_spec: H6AttemptSpec | None = None,
) -> H6AttemptSpec:
    """Authorize only an exact, current, FLOP-complete attempt.

    The matching check intentionally runs before any optional attempt object is
    inspected.  It performs no data, optimizer, gradient, or mutation work.
    """

    _require_eligible_matching_report(matching_report)
    if type(config) is not ArmConfig:
        raise ValueError("config must be an exact ArmConfig")
    config.__post_init__()
    objective_adapter.__post_init__()
    phase_schedule.__post_init__()
    if (
        matching_report.endpoint_config_sha256 != config.config_sha256
        or objective_adapter.endpoint_config_sha256 != config.config_sha256
        or phase_schedule.endpoint_config_sha256 != config.config_sha256
    ):
        raise ValueError("attempt endpoint identities are stale")
    if type(training_authorization) is not H6TrainingAuthorization:
        raise ValueError("eligible training requires exact ledger authorization")
    training_authorization.__post_init__()
    if (
        training_authorization.matching_report_sha256
        != matching_report.report_sha256
    ):
        raise ValueError("training authorization belongs to another report")
    if type(attempt_spec) is not H6AttemptSpec:
        raise ValueError("eligible training requires an exact H6AttemptSpec")
    attempt_spec.__post_init__()
    if (
        attempt_spec.arm is not config.arm
        or attempt_spec.config_id != config.config_id
        or attempt_spec.endpoint_config_sha256 != config.config_sha256
        or attempt_spec.latent_enabled != config.latent_enabled
        or attempt_spec.training_authorization_sha256
        != training_authorization.authorization_sha256
        or attempt_spec.objective_inventory_sha256
        != objective_adapter.inventory.inventory_sha256
        or attempt_spec.objective_adapter_sha256
        != objective_adapter.adapter_sha256
        or attempt_spec.objective_kind
        != objective_adapter.inventory.objective_kind
        or attempt_spec.phase_schedule_sha256
        != phase_schedule.phase_schedule_sha256
        or attempt_spec.optimizer_policy_sha256
        != training_authorization.optimizer_policy_sha256
    ):
        raise ValueError("attempt spec does not match current training authorities")
    return attempt_spec


def train_h6_attempt(**values: object) -> None:
    """Keep execution disabled until the separately authorized engine exists."""

    plan_h6_attempt(**values)  # type: ignore[arg-type]
    raise RuntimeError(
        "H6 training execution is disabled during source buildout"
    )


__all__ = [
    "ArmObjectiveInventory",
    "ArmTrainingObjectiveAdapter",
    "DetachedRecognitionLawSnapshot",
    "H6AttemptCursor",
    "H6AttemptSpec",
    "H6CrossEntropyTerms",
    "H6ObjectiveManifest",
    "H6ReducedLanguageElboTerms",
    "H6TrainingAuthorization",
    "H6TypedTrainingObjective",
    "plan_h6_attempt",
    "train_h6_attempt",
]
