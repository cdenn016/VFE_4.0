"""Immutable H6 Prefix and Prediction protocol records.

H6 deliberately owns language-horizon records instead of adapting the H1
two-position structures.  Digest-bearing records distinguish their one owned
integrity digest from referenced content digests, which are verified against
their producer bytes by the corresponding ``verify_*`` method.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, Literal, Protocol, runtime_checkable

import torch

from .results import GateStatus

if TYPE_CHECKING:
    from vfe4.artifacts.h6_matching import H6MatchingSetRecord
    from vfe4.generative.source_priors import (
        FixedSourcePrior,
        NormalizedSourceFactor,
        ParentSpecificPooledPrefixSourcePrior,
    )


_LOWER_HEX = frozenset("0123456789abcdef")
_H5_LABELS = (
    "exact_coordinate",
    "generalized_em",
    "natural_gradient_proposal",
)
H6_PREFIX_REQUIRED_CHECKS = (
    "signature_import",
    "taint_dataflow",
    "dynamic_target_suffix_leakage",
    "source_mask",
    "cache_identity",
    "case_inventory",
    "artifact_identity",
    "data_safety",
)
H1_PREFIX_PRIOR_V2_GENERATIVE_FACTOR_SCHEMA_SHA256 = (
    "0ab33d1cc790711eee82c598bb853d46ab52662eb31e9433e973978e77d9e375"
)


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase 64-hex SHA-256")
    return value


def _require_git_head(value: object, name: str = "git_head") -> str:
    if (
        type(value) is not str
        or len(value) != 40
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase 40-hex Git object name")
    return value


def _require_nonempty(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _canonical(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return value.as_posix()
    if type(value) is bytes:
        return {"hex": value.hex(), "length": len(value)}
    if type(value) is tuple:
        return [_canonical(item) for item in value]
    if type(value) is list:
        return [_canonical(item) for item in value]
    if isinstance(value, Mapping):
        if any(type(key) is not str or not key for key in value):
            raise ValueError("canonical mapping keys must be nonempty strings")
        return {key: _canonical(value[key]) for key in sorted(value)}
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("canonical floats must be finite")
        return value.hex()
    if type(value) in (str, int, bool) or value is None:
        return value
    raise ValueError(f"unsupported canonical value {type(value).__name__}")


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        _canonical(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _owned_hash(domain: str, value: object) -> str:
    return hashlib.sha256(domain.encode("ascii") + b"\x00" + canonical_json_bytes(value)).hexdigest()


def _new_frozen(cls: type[object], **values: object) -> object:
    """Construct a factory-only frozen dataclass and run its invariants."""

    instance = object.__new__(cls)
    for name, value in values.items():
        object.__setattr__(instance, name, value)
    instance.__post_init__()  # type: ignore[attr-defined]
    return instance


def _canonical_json_object(raw_bytes: bytes, name: str) -> dict[str, object]:
    if type(raw_bytes) is not bytes:
        raise ValueError(f"{name} must be immutable bytes")
    try:
        value = json.loads(raw_bytes)
        ordinary_canonical = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
        OverflowError,
    ) as exc:
        raise ValueError(f"{name} must be canonical JSON") from exc
    if type(value) is not dict or ordinary_canonical != raw_bytes:
        raise ValueError(f"{name} must be a canonical JSON object")
    return value


def _manifest_inventory(manifest_bytes: bytes) -> dict[str, str]:
    if type(manifest_bytes) is not bytes:
        raise ValueError("manifest_bytes must be immutable bytes")
    try:
        text = manifest_bytes.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("manifest must be ASCII") from exc
    if not text or not text.endswith("\n") or "\r" in text:
        raise ValueError("manifest must be nonempty canonical LF text")
    inventory: dict[str, str] = {}
    observed_paths: list[str] = []
    for line in text.splitlines():
        if line.count("  ") != 1:
            raise ValueError("manifest lines must contain one SHA/path separator")
        digest, path_text = line.split("  ", 1)
        _require_sha256(digest, "manifest content digest")
        path = PurePosixPath(path_text)
        if (
            not path_text
            or path.is_absolute()
            or path.as_posix() != path_text
            or any(part in (".", "..") for part in path.parts)
            or "\\" in path_text
            or path_text in inventory
        ):
            raise ValueError("manifest contains a noncanonical or duplicate path")
        inventory[path_text] = digest
        observed_paths.append(path_text)
    if observed_paths != sorted(observed_paths):
        raise ValueError("manifest paths must be sorted")
    return inventory


def _require_manifest_members(
    manifest_bytes: bytes, members: Mapping[str, bytes]
) -> None:
    inventory = _manifest_inventory(manifest_bytes)
    for path, content in members.items():
        expected = hashlib.sha256(content).hexdigest()
        if inventory.get(path) != expected:
            raise ValueError(f"manifest does not bind {path}")


def _validated_producer_status(
    *,
    payload_bytes: bytes,
    expected_gate: str,
    git_head: str,
    dirty_digest: str,
    config_bytes: bytes,
    extra_bindings: Mapping[str, str] | None = None,
) -> GateStatus:
    payload = _canonical_json_object(payload_bytes, "validation payload")
    required = {
        "gate": expected_gate,
        "git_head": git_head,
        "dirty_digest": dirty_digest,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        **({} if extra_bindings is None else dict(extra_bindings)),
    }
    for name, expected in required.items():
        if payload.get(name) != expected:
            raise ValueError(f"validation payload {name} does not match producer identity")
    try:
        status = GateStatus(payload.get("status"))
    except ValueError as exc:
        raise ValueError("validation payload has an unsupported status") from exc
    obligations = payload.get("obligations")
    if (
        type(obligations) is not list
        or any(type(item) is not str or not item for item in obligations)
    ):
        raise ValueError("validation payload obligations must be nonempty strings")
    if status is GateStatus.PASS and obligations:
        raise ValueError("PASS validation payload cannot retain an obligation")
    if status is GateStatus.INCONCLUSIVE and not obligations:
        raise ValueError("INCONCLUSIVE validation payload requires an obligation")
    return status


def _tensor_raw_bytes(value: torch.Tensor) -> bytes:
    cpu = value.detach().to(device="cpu").contiguous()
    try:
        return cpu.numpy().tobytes(order="C")
    except (TypeError, RuntimeError):
        return bytes(cpu.view(torch.uint8).reshape(-1).tolist())


class ArmId(str, Enum):
    A0 = "A0"
    A1 = "A1"
    A2 = "A2"
    A3 = "A3"
    A4 = "A4"
    A5 = "A5"


class EvidenceStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"


class TrainingPhase(str, Enum):
    MODEL_CE_ADAMW = "model_ce_adamw"
    RECOGNITION_ADAMW = "recognition_adamw"
    IMMUTABLE_DETACHED_SNAPSHOT = "immutable_detached_snapshot"
    MODEL_ADAMW = "model_adamw"


@dataclass(frozen=True)
class VocabularyIdentity:
    vocabulary_id: str
    size: int
    tokenizer_spec_sha256: str

    def __post_init__(self) -> None:
        _require_nonempty(self.vocabulary_id, "vocabulary_id")
        if type(self.size) is not int or self.size <= 0:
            raise ValueError("size must be a positive integer")
        _require_sha256(self.tokenizer_spec_sha256, "tokenizer_spec_sha256")

    @classmethod
    def from_tokenizer_spec(
        cls, *, vocabulary_id: str, size: int, tokenizer_spec_bytes: bytes
    ) -> "VocabularyIdentity":
        if type(tokenizer_spec_bytes) is not bytes:
            raise ValueError("tokenizer_spec_bytes must be bytes")
        return cls(vocabulary_id, size, hashlib.sha256(tokenizer_spec_bytes).hexdigest())

    def verify_tokenizer_spec(self, tokenizer_spec_bytes: bytes) -> None:
        if type(tokenizer_spec_bytes) is not bytes:
            raise ValueError("tokenizer_spec_bytes must be bytes")
        observed = hashlib.sha256(tokenizer_spec_bytes).hexdigest()
        if observed != self.tokenizer_spec_sha256:
            raise ValueError("tokenizer_spec_sha256 does not match tokenizer bytes")


_H6_CAPACITY_FIELDS = (
    "emission_width",
    "latent_width",
    "recognition_width",
    "prior_context_width",
)
_H6_ARM_SEMANTIC_FIELDS = (
    "latent_enabled",
    "state_channel_enabled",
    "model_channel_enabled",
    "source_mode",
    "map_mode",
    "recognition_family",
    "recognition_conditioning",
    "prior_variant",
    "mixture_mode",
    "objective_kind",
)
_H6_SOURCE_MODES = frozenset(
    {"absent", "immediate_predecessor", "categorical"}
)
_H6_MAP_MODES = frozenset(
    {
        "absent",
        "generic_fixed_frame_non_coboundary",
        "shared_vertex_coboundary",
    }
)
_H6_RECOGNITION_FAMILIES = frozenset(
    {"absent", "structured", "factorized"}
)
_H6_RECOGNITION_CONDITIONINGS = frozenset(
    {"absent", "filtering", "smoothing"}
)
_H6_PRIOR_VARIANTS = frozenset(
    {
        "absent",
        "fixed",
        "parent_specific_pooled_prefix",
        "pooled_history_conditioned",
        "prefix_conditioned",
    }
)
_H6_MIXTURE_MODES = frozenset({"absent", "exact", "moment_projection"})
_H6_OBJECTIVE_KINDS = frozenset(
    {"cross_entropy", "complete_elbo", "emission_only_ablation_non_elbo"}
)
_H6_PARENT_SPECIFIC_EMISSION_CONFIG_ID = (
    "h6-a5-structured-parent-specific-prefix-exact-emission-"
    "latent-smoothing-v2"
)
_H6_ARM_PROFILE_BY_CONFIG_ID = {
    "h6-a0-transformer-v2": (
        ArmId.A0, False, False, False, "absent", "absent", "absent",
        "absent", "absent", "absent", "cross_entropy",
    ),
    "h6-a1-ordinary-latent-v1": (
        ArmId.A1, True, True, False, "absent", "absent", "structured",
        "smoothing", "absent", "absent", "complete_elbo",
    ),
    "h6-a2-generic-map-v1": (
        ArmId.A2, True, True, True, "categorical",
        "generic_fixed_frame_non_coboundary", "structured", "smoothing",
        "fixed", "exact", "complete_elbo",
    ),
    "h6-a3-immediate-predecessor-v1": (
        ArmId.A3, True, True, True, "immediate_predecessor",
        "shared_vertex_coboundary", "structured", "smoothing", "absent",
        "absent", "complete_elbo",
    ),
    "h6-a4-state-only-v1": (
        ArmId.A4, True, True, False, "categorical",
        "shared_vertex_coboundary", "structured", "smoothing", "fixed",
        "exact", "complete_elbo",
    ),
    "h6-a5-structured-fixed-exact-complete-latent-smoothing-v1": (
        ArmId.A5, True, True, True, "categorical",
        "shared_vertex_coboundary", "structured", "smoothing", "fixed",
        "exact", "complete_elbo",
    ),
    "h6-a5-factorized-fixed-exact-complete-latent-smoothing-v1": (
        ArmId.A5, True, True, True, "categorical",
        "shared_vertex_coboundary", "factorized", "smoothing", "fixed",
        "exact", "complete_elbo",
    ),
    "h6-a5-structured-prefix-exact-complete-latent-smoothing-v1": (
        ArmId.A5, True, True, True, "categorical",
        "shared_vertex_coboundary", "structured", "smoothing",
        "prefix_conditioned", "exact", "complete_elbo",
    ),
    (
        "h6-a5-structured-parent-specific-prefix-exact-complete-"
        "latent-smoothing-v2"
    ): (
        ArmId.A5, True, True, True, "categorical",
        "shared_vertex_coboundary", "structured", "smoothing",
        "parent_specific_pooled_prefix", "exact", "complete_elbo",
    ),
    (
        "h6-a5-structured-parent-specific-prefix-exact-emission-"
        "latent-smoothing-v2"
    ): (
        ArmId.A5, True, True, True, "categorical",
        "shared_vertex_coboundary", "structured", "smoothing",
        "parent_specific_pooled_prefix", "exact",
        "emission_only_ablation_non_elbo",
    ),
    "h6-a5-structured-fixed-projection-complete-latent-smoothing-v1": (
        ArmId.A5, True, True, True, "categorical",
        "shared_vertex_coboundary", "structured", "smoothing", "fixed",
        "moment_projection", "complete_elbo",
    ),
    "h6-a5-structured-fixed-exact-emission-latent-smoothing-v1": (
        ArmId.A5, True, True, True, "categorical",
        "shared_vertex_coboundary", "structured", "smoothing", "fixed",
        "exact", "emission_only_ablation_non_elbo",
    ),
    "h6-a5-structured-fixed-exact-complete-nolatent-norecognition-v1": (
        ArmId.A5, False, False, False, "absent", "absent", "absent",
        "absent", "absent", "absent", "complete_elbo",
    ),
    "h6-a5-structured-fixed-exact-complete-latent-filtering-v1": (
        ArmId.A5, True, True, True, "categorical",
        "shared_vertex_coboundary", "structured", "filtering", "fixed",
        "exact", "complete_elbo",
    ),
}


def _require_exact_type_tuple(
    value: object,
    item_type: type[object],
    name: str,
    *,
    nonempty: bool = False,
) -> tuple[object, ...]:
    if (
        type(value) is not tuple
        or (nonempty and not value)
        or any(type(item) is not item_type for item in value)
    ):
        qualifier = "nonempty " if nonempty else ""
        raise ValueError(
            f"{name} must be a {qualifier}tuple of {item_type.__name__}"
        )
    return value


@dataclass(frozen=True, slots=True, init=False)
class AdamWPolicyRecord:
    """The H6 optimizer policy, excluding the preregistered tuning cell."""

    optimizer_class: Literal["AdamW"]
    betas: tuple[float, float]
    eps: float
    amsgrad: Literal[False]
    maximize: Literal[False]
    foreach: Literal[False]
    capturable: Literal[False]
    differentiable: Literal[False]
    fused: Literal[False]
    zero_grad_set_to_none: Literal[True]
    weight_decay_scope: Literal["all_active_parameters"]
    gradient_clip: Literal["always_evaluated_l2_global_scale"]
    gradient_clip_max_norm: float
    optimizer_policy_sha256: str

    def __post_init__(self) -> None:
        expected_fields = (
            "AdamW",
            (0.9, 0.999),
            1.0e-8,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            "all_active_parameters",
            "always_evaluated_l2_global_scale",
            1.0,
        )
        observed_fields = tuple(
            getattr(self, name)
            for name in tuple(self.__dataclass_fields__)[:-1]
        )
        if observed_fields != expected_fields:
            raise ValueError("AdamW policy must equal the frozen H6 contract")
        expected = _owned_hash(
            "vfe4.h6.adamw-policy.v1",
            {
                name: getattr(self, name)
                for name in tuple(self.__dataclass_fields__)[:-1]
            },
        )
        if self.optimizer_policy_sha256 != expected:
            raise ValueError(
                "optimizer_policy_sha256 does not match the AdamW policy"
            )

    @classmethod
    def create(cls) -> "AdamWPolicyRecord":
        values: dict[str, object] = {
            "optimizer_class": "AdamW",
            "betas": (0.9, 0.999),
            "eps": 1.0e-8,
            "amsgrad": False,
            "maximize": False,
            "foreach": False,
            "capturable": False,
            "differentiable": False,
            "fused": False,
            "zero_grad_set_to_none": True,
            "weight_decay_scope": "all_active_parameters",
            "gradient_clip": "always_evaluated_l2_global_scale",
            "gradient_clip_max_norm": 1.0,
        }
        return _new_frozen(
            cls,
            **values,
            optimizer_policy_sha256=_owned_hash(
                "vfe4.h6.adamw-policy.v1", values
            ),
        )  # type: ignore[return-value]


@dataclass(frozen=True, slots=True, init=False)
class CapacityAllocation:
    """Outcome-blind width allocation used by formula-only arm matching."""

    emission_width: int
    latent_width: int | None
    recognition_width: int | None
    prior_context_width: int | None
    allocation_sha256: str

    def __post_init__(self) -> None:
        if type(self.emission_width) is not int or self.emission_width <= 0:
            raise ValueError("emission_width must be a positive integer")
        for name in (
            "latent_width",
            "recognition_width",
            "prior_context_width",
        ):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be None or a positive integer")
        if self.latent_width is None and self.recognition_width is not None:
            raise ValueError(
                "recognition_width requires an applicable latent allocation"
            )
        if self.latent_width is None and self.prior_context_width is not None:
            raise ValueError(
                "prior_context_width requires an applicable latent allocation"
            )
        expected = _owned_hash(
            "vfe4.h6.capacity-allocation.v1",
            {
                name: getattr(self, name)
                for name in _H6_CAPACITY_FIELDS
            },
        )
        if self.allocation_sha256 != expected:
            raise ValueError(
                "allocation_sha256 does not match the capacity allocation"
            )

    @classmethod
    def create(
        cls,
        *,
        emission_width: int,
        latent_width: int | None,
        recognition_width: int | None,
        prior_context_width: int | None = None,
    ) -> "CapacityAllocation":
        values = {
            "emission_width": emission_width,
            "latent_width": latent_width,
            "recognition_width": recognition_width,
            "prior_context_width": prior_context_width,
        }
        return _new_frozen(
            cls,
            **values,
            allocation_sha256=_owned_hash(
                "vfe4.h6.capacity-allocation.v1", values
            ),
        )  # type: ignore[return-value]


@dataclass(frozen=True, slots=True, init=False)
class ArmConfig:
    """Typed, hash-bound semantic and nuisance-capacity arm configuration."""

    arm: ArmId
    config_id: str
    vocabulary: VocabularyIdentity
    horizon: int
    latent_enabled: bool
    state_channel_enabled: bool
    model_channel_enabled: bool
    source_mode: str
    map_mode: str
    recognition_family: str
    recognition_conditioning: str
    prior_variant: str
    mixture_mode: str
    objective_kind: str
    capacity_allocation: CapacityAllocation
    config_sha256: str

    def __post_init__(self) -> None:
        if type(self.arm) is not ArmId:
            raise ValueError("arm must be an ArmId")
        _require_nonempty(self.config_id, "config_id")
        if type(self.vocabulary) is not VocabularyIdentity:
            raise ValueError("vocabulary must be a VocabularyIdentity")
        if type(self.horizon) is not int or self.horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        for name in (
            "latent_enabled",
            "state_channel_enabled",
            "model_channel_enabled",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a bool")
        choices = (
            ("source_mode", _H6_SOURCE_MODES),
            ("map_mode", _H6_MAP_MODES),
            ("recognition_family", _H6_RECOGNITION_FAMILIES),
            (
                "recognition_conditioning",
                _H6_RECOGNITION_CONDITIONINGS,
            ),
            ("prior_variant", _H6_PRIOR_VARIANTS),
            ("mixture_mode", _H6_MIXTURE_MODES),
            ("objective_kind", _H6_OBJECTIVE_KINDS),
        )
        for name, allowed in choices:
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} is not an implemented H6 value")
        expected_profile = _H6_ARM_PROFILE_BY_CONFIG_ID.get(self.config_id)
        observed_profile = (self.arm,) + tuple(
            getattr(self, name) for name in _H6_ARM_SEMANTIC_FIELDS
        )
        if expected_profile is None or observed_profile != expected_profile:
            raise ValueError(
                "config_id and semantic fields must equal one canonical H6 arm profile"
            )
        if type(self.capacity_allocation) is not CapacityAllocation:
            raise ValueError(
                "capacity_allocation must be a CapacityAllocation"
            )
        if self.latent_enabled != (
            self.capacity_allocation.latent_width is not None
        ):
            raise ValueError(
                "latent_enabled must match latent_width applicability"
            )
        recognition_enabled = self.recognition_family != "absent"
        if recognition_enabled != (
            self.capacity_allocation.recognition_width is not None
        ):
            raise ValueError(
                "recognition_family must match recognition_width applicability"
            )
        if (self.recognition_conditioning != "absent") != recognition_enabled:
            raise ValueError(
                "recognition conditioning must match recognition applicability"
            )
        prefix_prior = self.prior_variant in (
            "parent_specific_pooled_prefix",
            "pooled_history_conditioned",
            "prefix_conditioned",
        )
        if prefix_prior != (
            self.capacity_allocation.prior_context_width is not None
        ):
            raise ValueError(
                "prefix-conditioned priors require one live prior_context_width"
            )
        if not self.latent_enabled:
            forbidden = (
                self.state_channel_enabled,
                self.model_channel_enabled,
                self.source_mode != "absent",
                self.map_mode != "absent",
                recognition_enabled,
                self.prior_variant != "absent",
                self.mixture_mode != "absent",
            )
            if any(forbidden):
                raise ValueError(
                    "a no-latent arm cannot retain latent/source/map sectors"
                )
        expected = _owned_hash(
            "vfe4.h6.arm-config.v1", self.canonical_payload()
        )
        if self.config_sha256 != expected:
            raise ValueError("config_sha256 does not match the arm config")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "arm": self.arm.value,
            "config_id": self.config_id,
            "vocabulary": {
                "vocabulary_id": self.vocabulary.vocabulary_id,
                "size": self.vocabulary.size,
                "tokenizer_spec_sha256": (
                    self.vocabulary.tokenizer_spec_sha256
                ),
            },
            "horizon": self.horizon,
            **{
                name: getattr(self, name)
                for name in _H6_ARM_SEMANTIC_FIELDS
            },
            "capacity_allocation": {
                name: getattr(self.capacity_allocation, name)
                for name in _H6_CAPACITY_FIELDS
            },
            "capacity_allocation_sha256": (
                self.capacity_allocation.allocation_sha256
            ),
        }

    @classmethod
    def create(
        cls,
        *,
        arm: ArmId,
        config_id: str,
        vocabulary: VocabularyIdentity,
        horizon: int,
        latent_enabled: bool,
        state_channel_enabled: bool,
        model_channel_enabled: bool,
        source_mode: str,
        map_mode: str,
        recognition_family: str,
        recognition_conditioning: str,
        prior_variant: str,
        mixture_mode: str,
        objective_kind: str,
        capacity_allocation: CapacityAllocation,
    ) -> "ArmConfig":
        values: dict[str, object] = {
            "arm": arm,
            "config_id": config_id,
            "vocabulary": vocabulary,
            "horizon": horizon,
            "latent_enabled": latent_enabled,
            "state_channel_enabled": state_channel_enabled,
            "model_channel_enabled": model_channel_enabled,
            "source_mode": source_mode,
            "map_mode": map_mode,
            "recognition_family": recognition_family,
            "recognition_conditioning": recognition_conditioning,
            "prior_variant": prior_variant,
            "mixture_mode": mixture_mode,
            "objective_kind": objective_kind,
            "capacity_allocation": capacity_allocation,
        }
        provisional = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(provisional, name, value)
        digest = _owned_hash(
            "vfe4.h6.arm-config.v1",
            provisional.canonical_payload(),
        )
        return _new_frozen(
            cls, **values, config_sha256=digest
        )  # type: ignore[return-value]

    def semantic_payload(self) -> dict[str, object]:
        """Return only intervention-bearing fields, excluding identity/capacity."""

        return {
            name: getattr(self, name)
            for name in _H6_ARM_SEMANTIC_FIELDS
        }


@dataclass(frozen=True, slots=True, init=False)
class ParameterRoleRecord:
    qualified_name: str
    role: str
    phase: str
    parameter_key: str
    scalar_count: int
    record_sha256: str

    def __post_init__(self) -> None:
        _require_nonempty(self.qualified_name, "qualified_name")
        _require_nonempty(self.role, "role")
        try:
            phase = TrainingPhase(self.phase)
        except (TypeError, ValueError) as exc:
            raise ValueError("phase is not an H6 training phase") from exc
        if phase is TrainingPhase.IMMUTABLE_DETACHED_SNAPSHOT:
            raise ValueError("snapshot phase cannot own trainable parameters")
        _require_sha256(self.parameter_key, "parameter_key")
        if type(self.scalar_count) is not int or self.scalar_count <= 0:
            raise ValueError("scalar_count must be a positive integer")
        expected = _owned_hash(
            "vfe4.h6.parameter-role.v2",
            {
                "qualified_name": self.qualified_name,
                "role": self.role,
                "phase": self.phase,
                "parameter_key": self.parameter_key,
                "scalar_count": self.scalar_count,
            },
        )
        if self.record_sha256 != expected:
            raise ValueError("record_sha256 does not match parameter role")

    @classmethod
    def create(
        cls,
        *,
        qualified_name: str,
        role: str,
        phase: str,
        parameter_key: str,
        scalar_count: int,
    ) -> "ParameterRoleRecord":
        values = {
            "qualified_name": qualified_name,
            "role": role,
            "phase": phase,
            "parameter_key": parameter_key,
            "scalar_count": scalar_count,
        }
        return _new_frozen(
            cls,
            **values,
            record_sha256=_owned_hash(
                "vfe4.h6.parameter-role.v2", values
            ),
        )  # type: ignore[return-value]


@dataclass(frozen=True, slots=True, init=False)
class OptimizerBinding:
    phase: str
    optimizer_class: Literal["AdamW"]
    optimizer_policy_sha256: str
    parameter_keys: tuple[str, ...]
    binding_sha256: str

    def __post_init__(self) -> None:
        try:
            phase = TrainingPhase(self.phase)
        except (TypeError, ValueError) as exc:
            raise ValueError("phase is not an H6 training phase") from exc
        if phase is TrainingPhase.IMMUTABLE_DETACHED_SNAPSHOT:
            raise ValueError("snapshot phase cannot have an optimizer binding")
        if self.optimizer_class != "AdamW":
            raise ValueError("optimizer_class must be AdamW")
        _require_sha256(
            self.optimizer_policy_sha256, "optimizer_policy_sha256"
        )
        _require_exact_type_tuple(
            self.parameter_keys, str, "parameter_keys", nonempty=True
        )
        if (
            any(
                len(parameter_key) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in parameter_key
                )
                for parameter_key in self.parameter_keys
            )
            or len(set(self.parameter_keys)) != len(self.parameter_keys)
        ):
            raise ValueError(
                "parameter_keys must contain unique lowercase SHA-256 keys"
            )
        expected = _owned_hash(
            "vfe4.h6.optimizer-binding.v2",
            {
                "phase": self.phase,
                "optimizer_class": self.optimizer_class,
                "optimizer_policy_sha256": self.optimizer_policy_sha256,
                "parameter_keys": self.parameter_keys,
            },
        )
        if self.binding_sha256 != expected:
            raise ValueError("binding_sha256 does not match optimizer binding")

    @classmethod
    def create(
        cls,
        *,
        phase: str,
        optimizer_class: str,
        optimizer_policy_sha256: str,
        parameter_keys: tuple[str, ...],
    ) -> "OptimizerBinding":
        values = {
            "phase": phase,
            "optimizer_class": optimizer_class,
            "optimizer_policy_sha256": optimizer_policy_sha256,
            "parameter_keys": tuple(parameter_keys),
        }
        return _new_frozen(
            cls,
            **values,
            binding_sha256=_owned_hash(
                "vfe4.h6.optimizer-binding.v2", values
            ),
        )  # type: ignore[return-value]


@dataclass(frozen=True, slots=True, init=False)
class FlopTerm:
    phase: str
    operation: str
    repetitions: int
    arithmetic_flops_per_repetition: int
    bytes_copied_per_repetition: int
    total_arithmetic_flops: int
    total_bytes_copied: int
    term_sha256: str

    def __post_init__(self) -> None:
        try:
            TrainingPhase(self.phase)
        except (TypeError, ValueError) as exc:
            raise ValueError("phase is not an H6 training phase") from exc
        _require_nonempty(self.operation, "operation")
        if type(self.repetitions) is not int or self.repetitions <= 0:
            raise ValueError("repetitions must be a positive integer")
        for name in (
            "arithmetic_flops_per_repetition",
            "bytes_copied_per_repetition",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.total_arithmetic_flops != (
            self.repetitions * self.arithmetic_flops_per_repetition
        ):
            raise ValueError("total_arithmetic_flops is inconsistent")
        if self.total_bytes_copied != (
            self.repetitions * self.bytes_copied_per_repetition
        ):
            raise ValueError("total_bytes_copied is inconsistent")
        if (
            self.phase == TrainingPhase.IMMUTABLE_DETACHED_SNAPSHOT.value
            and self.arithmetic_flops_per_repetition != 0
        ):
            raise ValueError("immutable snapshots cost zero arithmetic FLOPs")
        expected = _owned_hash(
            "vfe4.h6.flop-term.v1",
            {
                name: getattr(self, name)
                for name in tuple(self.__dataclass_fields__)[:-1]
            },
        )
        if self.term_sha256 != expected:
            raise ValueError("term_sha256 does not match the FLOP term")

    @property
    def arithmetic_flops(self) -> int:
        return self.total_arithmetic_flops

    @property
    def copied_bytes(self) -> int:
        return self.total_bytes_copied

    @property
    def total_flops(self) -> int:
        return self.total_arithmetic_flops

    @classmethod
    def create(
        cls,
        *,
        phase: str,
        operation: str,
        repetitions: int,
        arithmetic_flops_per_repetition: int,
        bytes_copied_per_repetition: int,
    ) -> "FlopTerm":
        values = {
            "phase": phase,
            "operation": operation,
            "repetitions": repetitions,
            "arithmetic_flops_per_repetition": (
                arithmetic_flops_per_repetition
            ),
            "bytes_copied_per_repetition": bytes_copied_per_repetition,
            "total_arithmetic_flops": (
                repetitions * arithmetic_flops_per_repetition
            ),
            "total_bytes_copied": (
                repetitions * bytes_copied_per_repetition
            ),
        }
        return _new_frozen(
            cls,
            **values,
            term_sha256=_owned_hash("vfe4.h6.flop-term.v1", values),
        )  # type: ignore[return-value]


H6_INFERENCE_PARTICLE_COUNTS = (128, 256, 512, 1024)


@dataclass(frozen=True, slots=True, init=False)
class H6PrefixWorkloadPlan:
    """Frozen, outcome-blind bounded workload for the H6 Prefix evidence run.

    This record deliberately has no caller-supplied construction path.  Its
    complete workload, including every derived total and its owned digest, is
    fixed by this source amendment before a Prefix runner may consume it.
    """

    schema_version: Literal["h6-prefix-workload-plan-v1"]
    representative_particle_count: int
    production_particle_counts: tuple[int, ...]
    small_global_case_indices: tuple[int, ...]
    validation_global_case_indices: tuple[int, ...]
    prediction_calls_per_case: int
    representative_full_cases: int
    representative_prediction_calls: int
    representative_particle_call_units: int
    ladder_subset_cases: int
    ladder_subset_prediction_calls: int
    ladder_subset_particle_call_units: int
    amended_total_cases: int
    amended_total_prediction_calls: int
    amended_total_particle_call_units: int
    signature_and_identity_particle_counts: tuple[int, ...]
    exhaustive_representative_checks: tuple[str, ...]
    stratified_subset_particle_counts: tuple[int, ...]
    representative_report_checks: tuple[str, ...]
    finite_smc_accuracy_particle_count: int
    subset_is_estimator_accuracy_evidence: Literal[False]
    workload_plan_sha256: str

    def __new__(cls, *args: object, **kwargs: object) -> "H6PrefixWorkloadPlan":
        if cls is not H6PrefixWorkloadPlan:
            raise TypeError("constructor requires the exact H6PrefixWorkloadPlan type")
        if args or kwargs:
            raise TypeError(
                "H6PrefixWorkloadPlan accepts no caller-supplied values"
            )
        return H6PrefixWorkloadPlan.create()

    def canonical_payload(self) -> dict[str, object]:
        return {
            descriptor.name: getattr(self, descriptor.name)
            for descriptor in fields(self)
            if descriptor.name != "workload_plan_sha256"
        }

    def __post_init__(self) -> None:
        if type(self) is not H6PrefixWorkloadPlan:
            raise TypeError("record requires the exact H6PrefixWorkloadPlan type")
        if self.schema_version != "h6-prefix-workload-plan-v1":
            raise ValueError("H6 Prefix workload schema version is frozen")
        integer_fields = (
            "representative_particle_count",
            "prediction_calls_per_case",
            "representative_full_cases",
            "representative_prediction_calls",
            "representative_particle_call_units",
            "ladder_subset_cases",
            "ladder_subset_prediction_calls",
            "ladder_subset_particle_call_units",
            "amended_total_cases",
            "amended_total_prediction_calls",
            "amended_total_particle_call_units",
            "finite_smc_accuracy_particle_count",
        )
        if any(type(getattr(self, name)) is not int for name in integer_fields):
            raise ValueError("H6 Prefix workload integer fields must be exact integers")
        integer_tuples = (
            "production_particle_counts",
            "small_global_case_indices",
            "validation_global_case_indices",
            "signature_and_identity_particle_counts",
            "stratified_subset_particle_counts",
        )
        for name in integer_tuples:
            value = getattr(self, name)
            if type(value) is not tuple or any(type(item) is not int for item in value):
                raise ValueError(f"{name} must be an exact tuple of integers")
        string_tuples = (
            "exhaustive_representative_checks",
            "representative_report_checks",
        )
        for name in string_tuples:
            value = getattr(self, name)
            if type(value) is not tuple or any(type(item) is not str for item in value):
                raise ValueError(f"{name} must be an exact tuple of strings")
        if type(self.subset_is_estimator_accuracy_evidence) is not bool:
            raise ValueError("subset estimator-accuracy evidence flag must be boolean")

        frozen_particle_counts = (128, 256, 512, 1024)
        frozen_small_global_case_indices = (
            0,
            2186,
            4373,
            6560,
            6561,
            7289,
            8018,
            8747,
            8748,
            8990,
            9233,
            9476,
            9477,
            9557,
            9638,
            9719,
        )
        frozen_validation_global_case_indices = (
            128,
            384,
            640,
            896,
            1152,
            1408,
            1664,
            1920,
            2176,
            2432,
            2688,
            2944,
            3200,
            3456,
            3712,
            3968,
        )
        if self.representative_particle_count != 128:
            raise ValueError("representative particle count must be exactly 128")
        if self.production_particle_counts != frozen_particle_counts:
            raise ValueError("production particle counts must equal the frozen ladder")
        if self.small_global_case_indices != frozen_small_global_case_indices:
            raise ValueError("small global case indices must equal the frozen stratification")
        if self.validation_global_case_indices != frozen_validation_global_case_indices:
            raise ValueError(
                "validation global case indices must equal the frozen stratification"
            )
        if self.prediction_calls_per_case != 5:
            raise ValueError("representative cases require exactly five prediction calls")
        if self.signature_and_identity_particle_counts != frozen_particle_counts:
            raise ValueError("signature and identity checks require every particle level")
        if self.exhaustive_representative_checks != (
            "dynamic_target_suffix_leakage",
            "cache_identity",
        ):
            raise ValueError("stratified Prefix checks do not match the frozen contract")
        if self.stratified_subset_particle_counts != (256, 512, 1024):
            raise ValueError("stratified subsets require exactly 256, 512, and 1024")
        if self.representative_report_checks != (
            "source_mask",
            "case_inventory",
            "validation_data_safety",
        ):
            raise ValueError("representative report checks do not match the frozen contract")
        if self.finite_smc_accuracy_particle_count != 256:
            raise ValueError("finite-SMC accuracy remains a separate 256-particle gate")
        if self.subset_is_estimator_accuracy_evidence is not False:
            raise ValueError("stratified subsets are not estimator-accuracy evidence")

        representative_full_cases = 9_720 + 4_096
        representative_prediction_calls = (
            representative_full_cases * self.prediction_calls_per_case
        )
        representative_particle_call_units = (
            representative_prediction_calls * self.representative_particle_count
        )
        ladder_subset_cases = (
            len(self.small_global_case_indices)
            + len(self.validation_global_case_indices)
        ) * len(self.stratified_subset_particle_counts)
        ladder_subset_prediction_calls = (
            ladder_subset_cases * self.prediction_calls_per_case
        )
        ladder_subset_particle_call_units = (
            (len(self.small_global_case_indices) + len(self.validation_global_case_indices))
            * self.prediction_calls_per_case
            * sum(self.stratified_subset_particle_counts)
        )
        derived_fields = {
            "representative_full_cases": representative_full_cases,
            "representative_prediction_calls": representative_prediction_calls,
            "representative_particle_call_units": representative_particle_call_units,
            "ladder_subset_cases": ladder_subset_cases,
            "ladder_subset_prediction_calls": ladder_subset_prediction_calls,
            "ladder_subset_particle_call_units": ladder_subset_particle_call_units,
            "amended_total_cases": representative_full_cases + ladder_subset_cases,
            "amended_total_prediction_calls": (
                representative_prediction_calls + ladder_subset_prediction_calls
            ),
            "amended_total_particle_call_units": (
                representative_particle_call_units + ladder_subset_particle_call_units
            ),
        }
        for name, expected in derived_fields.items():
            if getattr(self, name) != expected:
                raise ValueError(f"{name} does not match the frozen workload derivation")
        expected_hash = _owned_hash(
            "vfe4.h6.prefix-workload-plan.v1", self.canonical_payload()
        )
        if self.workload_plan_sha256 != expected_hash:
            raise ValueError("workload_plan_sha256 does not match the frozen workload")

    @classmethod
    def create(cls) -> "H6PrefixWorkloadPlan":
        if cls is not H6PrefixWorkloadPlan:
            raise TypeError("factory requires the exact H6PrefixWorkloadPlan type")
        values: dict[str, object] = {
            "schema_version": "h6-prefix-workload-plan-v1",
            "representative_particle_count": 128,
            "production_particle_counts": (128, 256, 512, 1024),
            "small_global_case_indices": (
                0,
                2186,
                4373,
                6560,
                6561,
                7289,
                8018,
                8747,
                8748,
                8990,
                9233,
                9476,
                9477,
                9557,
                9638,
                9719,
            ),
            "validation_global_case_indices": (
                128,
                384,
                640,
                896,
                1152,
                1408,
                1664,
                1920,
                2176,
                2432,
                2688,
                2944,
                3200,
                3456,
                3712,
                3968,
            ),
            "prediction_calls_per_case": 5,
            "representative_full_cases": 13_816,
            "representative_prediction_calls": 69_080,
            "representative_particle_call_units": 8_842_240,
            "ladder_subset_cases": 96,
            "ladder_subset_prediction_calls": 480,
            "ladder_subset_particle_call_units": 286_720,
            "amended_total_cases": 13_912,
            "amended_total_prediction_calls": 69_560,
            "amended_total_particle_call_units": 9_128_960,
            "signature_and_identity_particle_counts": (128, 256, 512, 1024),
            "exhaustive_representative_checks": (
                "dynamic_target_suffix_leakage",
                "cache_identity",
            ),
            "stratified_subset_particle_counts": (256, 512, 1024),
            "representative_report_checks": (
                "source_mask",
                "case_inventory",
                "validation_data_safety",
            ),
            "finite_smc_accuracy_particle_count": 256,
            "subset_is_estimator_accuracy_evidence": False,
        }
        return _new_frozen(
            H6PrefixWorkloadPlan,
            **values,
            workload_plan_sha256=_owned_hash(
                "vfe4.h6.prefix-workload-plan.v1", values
            ),
        )  # type: ignore[return-value]


@dataclass(frozen=True, slots=True, init=False)
class InferenceComputeRecord:
    """Inference FLOPs kept strictly outside training-match eligibility.

    ``scoring_flops`` is the aggregate scoring arithmetic for the declared
    ``replicate_count``.  ``total_flops`` adds the one-time checkpoint-load
    and prefix-cache construction arithmetic for this exact scorer row.
    """

    endpoint_id: str
    scorer_kind: Literal["exact_autoregressive", "weighted_smc"]
    particle_count: int | None
    replicate_count: int
    prefix_cache_mode: str
    checkpoint_load_flops: int
    cache_build_flops: int
    scoring_flops: int
    total_flops: int
    wall_time_seconds: float | None
    record_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in tuple(self.__dataclass_fields__)[:-1]
        }

    def __post_init__(self) -> None:
        _require_nonempty(self.endpoint_id, "endpoint_id")
        if self.scorer_kind not in (
            "exact_autoregressive",
            "weighted_smc",
        ):
            raise ValueError("unsupported inference scorer_kind")
        if self.endpoint_id == "h6-a0-transformer-v2" and (
            self.scorer_kind != "exact_autoregressive"
            or self.particle_count is not None
        ):
            raise ValueError(
                "H6 A0 requires exact autoregressive scoring with "
                "particle_count=None"
            )
        if self.scorer_kind == "exact_autoregressive":
            if self.particle_count is not None:
                raise ValueError(
                    "exact autoregressive scoring requires particle_count=None"
                )
        elif (
            type(self.particle_count) is not int
            or self.particle_count not in H6_INFERENCE_PARTICLE_COUNTS
        ):
            raise ValueError(
                "weighted SMC particle_count must be one of "
                "128, 256, 512, or 1024"
            )
        if type(self.replicate_count) is not int or self.replicate_count <= 0:
            raise ValueError("replicate_count must be a positive integer")
        _require_nonempty(self.prefix_cache_mode, "prefix_cache_mode")
        for name in (
            "checkpoint_load_flops",
            "cache_build_flops",
            "scoring_flops",
            "total_flops",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.scoring_flops <= 0:
            raise ValueError("scoring_flops must be positive")
        if self.total_flops != (
            self.checkpoint_load_flops
            + self.cache_build_flops
            + self.scoring_flops
        ):
            raise ValueError(
                "total_flops must equal checkpoint load, cache build, and "
                "aggregate scoring FLOPs"
            )
        if self.wall_time_seconds is not None and (
            type(self.wall_time_seconds) is not float
            or not math.isfinite(self.wall_time_seconds)
            or self.wall_time_seconds <= 0.0
        ):
            raise ValueError(
                "wall_time_seconds must be None or a positive finite float"
            )
        expected = _owned_hash(
            "vfe4.h6.inference-compute-record.v1",
            self.canonical_payload(),
        )
        if self.record_sha256 != expected:
            raise ValueError("inference compute record digest is stale")

    @classmethod
    def create(
        cls,
        *,
        endpoint_id: str,
        scorer_kind: Literal["exact_autoregressive", "weighted_smc"],
        particle_count: int | None,
        replicate_count: int,
        prefix_cache_mode: str,
        checkpoint_load_flops: int,
        cache_build_flops: int,
        scoring_flops: int,
        total_flops: int,
        wall_time_seconds: float | None,
    ) -> "InferenceComputeRecord":
        values: dict[str, object] = {
            "endpoint_id": endpoint_id,
            "scorer_kind": scorer_kind,
            "particle_count": particle_count,
            "replicate_count": replicate_count,
            "prefix_cache_mode": prefix_cache_mode,
            "checkpoint_load_flops": checkpoint_load_flops,
            "cache_build_flops": cache_build_flops,
            "scoring_flops": scoring_flops,
            "total_flops": total_flops,
            "wall_time_seconds": wall_time_seconds,
        }
        return _new_frozen(
            cls,
            **values,
            record_sha256=_owned_hash(
                "vfe4.h6.inference-compute-record.v1",
                values,
            ),
        )  # type: ignore[return-value]


@dataclass(frozen=True, slots=True, init=False)
class MatchingReport:
    matching_config_sha256: str
    endpoint_config_sha256: str
    reference_config_sha256: str
    endpoint_parameter_count: int
    reference_parameter_count: int
    parameter_relative_difference: float
    endpoint_training_flops: int
    reference_training_flops: int
    flop_relative_difference: float
    parameter_relative_tolerance: float
    flop_relative_tolerance: float
    ownership_valid: bool
    common_schedule: bool
    optimizer_policy_match: bool
    training_flop_ledger_complete: bool
    training_flop_obligations: tuple[str, ...]
    semantic_interventions: tuple[str, ...]
    named_factor: str
    nuisance_capacity_fields: tuple[str, ...]
    capacity_allocation_policy: Literal[
        "outcome_blind_nuisance_reallocation"
    ]
    common_schedule_sha256: str
    status: Literal["ELIGIBLE", "INCONCLUSIVE"]
    eligible: bool
    obligations: tuple[str, ...]
    report_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(
            self.matching_config_sha256, "matching_config_sha256"
        )
        _require_sha256(
            self.endpoint_config_sha256, "endpoint_config_sha256"
        )
        _require_sha256(
            self.reference_config_sha256, "reference_config_sha256"
        )
        for name in (
            "endpoint_parameter_count",
            "reference_parameter_count",
            "endpoint_training_flops",
            "reference_training_flops",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if (
            self.reference_parameter_count <= 0
            or self.reference_training_flops <= 0
        ):
            raise ValueError("reference totals must be positive")
        if (
            type(self.parameter_relative_tolerance) is not float
            or self.parameter_relative_tolerance != 0.01
            or type(self.flop_relative_tolerance) is not float
            or self.flop_relative_tolerance != 0.05
        ):
            raise ValueError("matching tolerances are not the canonical policy")
        expected_parameter_difference = abs(
            self.endpoint_parameter_count - self.reference_parameter_count
        ) / self.reference_parameter_count
        expected_flop_difference = abs(
            self.endpoint_training_flops - self.reference_training_flops
        ) / self.reference_training_flops
        if (
            self.parameter_relative_difference
            != expected_parameter_difference
            or self.flop_relative_difference != expected_flop_difference
        ):
            raise ValueError("matching relative differences are inconsistent")
        for name in (
            "ownership_valid",
            "common_schedule",
            "optimizer_policy_match",
            "training_flop_ledger_complete",
            "eligible",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a bool")
        _require_exact_type_tuple(
            self.training_flop_obligations,
            str,
            "training_flop_obligations",
        )
        _require_exact_type_tuple(
            self.semantic_interventions,
            str,
            "semantic_interventions",
        )
        _require_nonempty(self.named_factor, "named_factor")
        _require_exact_type_tuple(
            self.nuisance_capacity_fields,
            str,
            "nuisance_capacity_fields",
        )
        _require_exact_type_tuple(
            self.obligations, str, "obligations"
        )
        if (
            self.capacity_allocation_policy
            != "outcome_blind_nuisance_reallocation"
        ):
            raise ValueError("capacity allocation policy is not frozen")
        _require_sha256(
            self.common_schedule_sha256, "common_schedule_sha256"
        )
        if self.status not in ("ELIGIBLE", "INCONCLUSIVE"):
            raise ValueError("unsupported matching status")
        if self.eligible != (self.status == "ELIGIBLE"):
            raise ValueError("eligible does not match status")
        if self.eligible != (not self.obligations):
            raise ValueError("eligible reports must have no obligations")
        expected = _owned_hash(
            "vfe4.h6.matching-report.v1",
            {
                name: getattr(self, name)
                for name in tuple(self.__dataclass_fields__)[:-1]
            },
        )
        if self.report_sha256 != expected:
            raise ValueError("report_sha256 does not match matching report")

    @classmethod
    def from_totals(
        cls,
        *,
        matching_config_sha256: str,
        endpoint_config_sha256: str,
        reference_config_sha256: str,
        endpoint_parameter_count: int,
        reference_parameter_count: int,
        endpoint_training_flops: int,
        reference_training_flops: int,
        parameter_relative_tolerance: float,
        flop_relative_tolerance: float,
        ownership_valid: bool,
        common_schedule: bool,
        optimizer_policy_match: bool,
        training_flop_ledger_complete: bool,
        training_flop_obligations: tuple[str, ...],
        semantic_interventions: tuple[str, ...],
        named_factor: str,
        nuisance_capacity_fields: tuple[str, ...],
        common_schedule_sha256: str,
    ) -> "MatchingReport":
        if (
            type(reference_parameter_count) is not int
            or reference_parameter_count <= 0
            or type(reference_training_flops) is not int
            or reference_training_flops <= 0
        ):
            raise ValueError("reference matching totals must be positive")
        parameter_relative_difference = abs(
            endpoint_parameter_count - reference_parameter_count
        ) / reference_parameter_count
        flop_relative_difference = abs(
            endpoint_training_flops - reference_training_flops
        ) / reference_training_flops
        obligations: list[str] = []
        if not ownership_valid:
            obligations.append("resolve exact parameter ownership")
        if not common_schedule:
            obligations.append("restore the common training schedule")
        if not optimizer_policy_match:
            obligations.append("restore the common AdamW policy")
        flop_obligations = tuple(training_flop_obligations)
        if (
            type(training_flop_ledger_complete) is not bool
            or type(flop_obligations) is not tuple
            or any(type(item) is not str or not item for item in flop_obligations)
        ):
            raise ValueError("training FLOP completeness evidence is malformed")
        if not training_flop_ledger_complete:
            obligations.append(
                "provide a complete whole-schedule training FLOP ledger"
            )
            obligations.extend(flop_obligations)
        elif flop_obligations:
            obligations.append(
                "clear all training FLOP ledger obligations before eligibility"
            )
            obligations.extend(flop_obligations)
        if parameter_relative_difference > parameter_relative_tolerance:
            obligations.append(
                "match active parameter count within the hard 1% tolerance"
            )
        if flop_relative_difference > flop_relative_tolerance:
            obligations.append(
                "match whole-schedule FLOPs within the hard 5% tolerance"
            )
        interventions = tuple(semantic_interventions)
        if named_factor not in (
            "whole_declared_architecture",
            "arm_semantics",
        ) and interventions != (named_factor,):
            obligations.append(
                "restrict semantic interventions to the named factor"
            )
        nuisance = tuple(nuisance_capacity_fields)
        if (
            len(set(nuisance)) != len(nuisance)
            or any(field not in _H6_CAPACITY_FIELDS for field in nuisance)
        ):
            obligations.append(
                "restrict nuisance fields to outcome-blind capacity widths"
            )
        values: dict[str, object] = {
            "matching_config_sha256": matching_config_sha256,
            "endpoint_config_sha256": endpoint_config_sha256,
            "reference_config_sha256": reference_config_sha256,
            "endpoint_parameter_count": endpoint_parameter_count,
            "reference_parameter_count": reference_parameter_count,
            "parameter_relative_difference": parameter_relative_difference,
            "endpoint_training_flops": endpoint_training_flops,
            "reference_training_flops": reference_training_flops,
            "flop_relative_difference": flop_relative_difference,
            "parameter_relative_tolerance": parameter_relative_tolerance,
            "flop_relative_tolerance": flop_relative_tolerance,
            "ownership_valid": ownership_valid,
            "common_schedule": common_schedule,
            "optimizer_policy_match": optimizer_policy_match,
            "training_flop_ledger_complete": (
                training_flop_ledger_complete
            ),
            "training_flop_obligations": flop_obligations,
            "semantic_interventions": interventions,
            "named_factor": named_factor,
            "nuisance_capacity_fields": nuisance,
            "capacity_allocation_policy": (
                "outcome_blind_nuisance_reallocation"
            ),
            "common_schedule_sha256": common_schedule_sha256,
            "status": "ELIGIBLE" if not obligations else "INCONCLUSIVE",
            "eligible": not obligations,
            "obligations": tuple(obligations),
        }
        return _new_frozen(
            cls,
            **values,
            report_sha256=_owned_hash(
                "vfe4.h6.matching-report.v1", values
            ),
        )  # type: ignore[return-value]


@dataclass(frozen=True, slots=True, init=False)
class ArmMatrixRow:
    row_id: str
    left_config_id: str
    left_factory_id: str
    right_config_id: str
    right_factory_id: str
    named_factor: str
    semantic_interventions: tuple[str, ...]
    nuisance_capacity_fields: tuple[str, ...]
    capacity_allocation_policy: Literal[
        "outcome_blind_nuisance_reallocation"
    ]
    tuning_estimand: str
    interpretation: str
    confirmatory_seeds: tuple[int, ...]
    checkpoint_template: str
    certificate_key_template: str
    opening_group: str
    nonclaims: tuple[str, ...]
    row_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "row_id",
            "left_config_id",
            "left_factory_id",
            "right_config_id",
            "right_factory_id",
            "named_factor",
            "tuning_estimand",
            "interpretation",
            "checkpoint_template",
            "certificate_key_template",
            "opening_group",
        ):
            _require_nonempty(getattr(self, name), name)
        _require_exact_type_tuple(
            self.semantic_interventions,
            str,
            "semantic_interventions",
            nonempty=True,
        )
        _require_exact_type_tuple(
            self.nuisance_capacity_fields,
            str,
            "nuisance_capacity_fields",
        )
        if any(
            name not in _H6_CAPACITY_FIELDS
            for name in self.nuisance_capacity_fields
        ):
            raise ValueError("matrix row has a non-capacity nuisance field")
        if (
            self.capacity_allocation_policy
            != "outcome_blind_nuisance_reallocation"
        ):
            raise ValueError("matrix row capacity policy is not frozen")
        _require_exact_type_tuple(
            self.confirmatory_seeds,
            int,
            "confirmatory_seeds",
            nonempty=True,
        )
        if self.confirmatory_seeds != tuple(range(2026072101, 2026072109)):
            raise ValueError("matrix row confirmatory seeds are not frozen")
        _require_exact_type_tuple(
            self.nonclaims, str, "nonclaims", nonempty=True
        )
        expected = _owned_hash(
            "vfe4.h6.arm-matrix-row.v1",
            {
                name: getattr(self, name)
                for name in tuple(self.__dataclass_fields__)[:-1]
            },
        )
        if self.row_sha256 != expected:
            raise ValueError("row_sha256 does not match matrix row")

    @classmethod
    def create(
        cls,
        *,
        row_id: str,
        left_config_id: str,
        left_factory_id: str,
        right_config_id: str,
        right_factory_id: str,
        named_factor: str,
        semantic_interventions: tuple[str, ...],
        nuisance_capacity_fields: tuple[str, ...],
        tuning_estimand: str,
        interpretation: str,
        checkpoint_template: str,
        certificate_key_template: str,
        opening_group: str,
        nonclaims: tuple[str, ...],
    ) -> "ArmMatrixRow":
        values: dict[str, object] = {
            "row_id": row_id,
            "left_config_id": left_config_id,
            "left_factory_id": left_factory_id,
            "right_config_id": right_config_id,
            "right_factory_id": right_factory_id,
            "named_factor": named_factor,
            "semantic_interventions": tuple(semantic_interventions),
            "nuisance_capacity_fields": tuple(nuisance_capacity_fields),
            "capacity_allocation_policy": (
                "outcome_blind_nuisance_reallocation"
            ),
            "tuning_estimand": tuning_estimand,
            "interpretation": interpretation,
            "confirmatory_seeds": tuple(range(2026072101, 2026072109)),
            "checkpoint_template": checkpoint_template,
            "certificate_key_template": certificate_key_template,
            "opening_group": opening_group,
            "nonclaims": tuple(nonclaims),
        }
        return _new_frozen(
            cls,
            **values,
            row_sha256=_owned_hash(
                "vfe4.h6.arm-matrix-row.v1", values
            ),
        )  # type: ignore[return-value]


_TOKEN_STORAGE_DOMAIN = b"VFE4-H6-U16LE-TOKENS-V1\x00"


@dataclass(frozen=True, init=False)
class EncodedTokenStorageIdentity:
    storage_schema: Literal["vfe4-h6-u16le-tokens-v1"]
    token_count: int
    byte_length: int
    encoded_token_sha256: str
    _encoded_token_bytes: bytes = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.storage_schema != "vfe4-h6-u16le-tokens-v1":
            raise ValueError("unsupported encoded-token storage schema")
        if type(self.token_count) is not int or self.token_count <= 0:
            raise ValueError("token_count must be positive")
        if type(self._encoded_token_bytes) is not bytes:
            raise ValueError("encoded token bytes must be immutable bytes")
        if len(self._encoded_token_bytes) != 2 * self.token_count:
            raise ValueError("encoded token byte count does not match token_count")
        for offset in range(0, len(self._encoded_token_bytes), 2):
            token_id = int.from_bytes(self._encoded_token_bytes[offset : offset + 2], "little")
            if token_id > 257:
                raise ValueError("encoded token IDs must be in 0..257")
        preimage = (
            _TOKEN_STORAGE_DOMAIN
            + self.token_count.to_bytes(8, "little")
            + self._encoded_token_bytes
        )
        if self.byte_length != len(preimage):
            raise ValueError("byte_length does not match encoded-token preimage")
        if self.encoded_token_sha256 != hashlib.sha256(preimage).hexdigest():
            raise ValueError("encoded_token_sha256 does not match encoded token bytes")

    @classmethod
    def create(cls, *, token_count: int, encoded_token_bytes: bytes) -> "EncodedTokenStorageIdentity":
        if type(encoded_token_bytes) is not bytes:
            raise ValueError("encoded_token_bytes must be bytes")
        preimage = _TOKEN_STORAGE_DOMAIN + token_count.to_bytes(8, "little") + encoded_token_bytes
        return _new_frozen(
            cls,
            storage_schema="vfe4-h6-u16le-tokens-v1",
            token_count=token_count,
            byte_length=len(preimage),
            encoded_token_sha256=hashlib.sha256(preimage).hexdigest(),
            _encoded_token_bytes=bytes(encoded_token_bytes),
        )  # type: ignore[return-value]

    def verify_encoded_token_bytes(self, encoded_token_bytes: bytes) -> None:
        if encoded_token_bytes != self._encoded_token_bytes:
            raise ValueError("encoded_token_sha256 does not match encoded token bytes")
        self.__post_init__()


@dataclass(frozen=True, init=False)
class ValidationSafetyFixture:
    policy: Literal["vfe4-h6-validation-safety-fixture-v1"]
    validation_token_sha256: str
    starts: tuple[int, ...]
    real_target_counts: tuple[int, ...]
    fixture_sha256: str
    _fixture_bytes: bytes = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.policy != "vfe4-h6-validation-safety-fixture-v1":
            raise ValueError("unsupported validation safety policy")
        _require_sha256(self.validation_token_sha256, "validation_token_sha256")
        if (
            type(self.starts) is not tuple
            or len(self.starts) != 4096
            or any(type(value) is not int or value < 0 for value in self.starts)
            or len(set(self.starts)) != 4096
        ):
            raise ValueError("starts must contain exactly 4,096 unique nonnegative integers")
        if (
            type(self.real_target_counts) is not tuple
            or len(self.real_target_counts) != 4096
            or any(type(value) is not int or value <= 0 or value > 32 for value in self.real_target_counts)
        ):
            raise ValueError("real_target_counts must contain 4,096 values in 1..32")
        if type(self._fixture_bytes) is not bytes:
            raise ValueError("fixture bytes must be immutable bytes")
        if self.fixture_sha256 != hashlib.sha256(self._fixture_bytes).hexdigest():
            raise ValueError("fixture_sha256 does not match fixture bytes")

    @classmethod
    def create(
        cls,
        *,
        validation_token_sha256: str,
        starts: tuple[int, ...],
        real_target_counts: tuple[int, ...],
        fixture_bytes: bytes,
    ) -> "ValidationSafetyFixture":
        if type(fixture_bytes) is not bytes:
            raise ValueError("fixture_bytes must be bytes")
        return _new_frozen(
            cls,
            policy="vfe4-h6-validation-safety-fixture-v1",
            validation_token_sha256=validation_token_sha256,
            starts=tuple(starts),
            real_target_counts=tuple(real_target_counts),
            fixture_sha256=hashlib.sha256(fixture_bytes).hexdigest(),
            _fixture_bytes=bytes(fixture_bytes),
        )  # type: ignore[return-value]

    def verify_fixture_bytes(self, fixture_bytes: bytes) -> None:
        if fixture_bytes != self._fixture_bytes:
            raise ValueError("fixture_sha256 does not match fixture bytes")
        self.__post_init__()


@dataclass(frozen=True)
class FrozenBatchSchedule:
    schedule_schema: Literal["vfe4-h6-frozen-batch-schedule-v1"]
    shared_seed: Literal[2026072199]
    zero_based_pass_index: int
    window_count: int
    batch_size: Literal[8]
    drop_last: Literal[False]
    permutation: tuple[int, ...]
    schedule_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schedule_schema != "vfe4-h6-frozen-batch-schedule-v1"
            or self.shared_seed != 2026072199
            or self.batch_size != 8
            or self.drop_last is not False
        ):
            raise ValueError("unsupported frozen batch schedule policy")
        if type(self.zero_based_pass_index) is not int or self.zero_based_pass_index < 0:
            raise ValueError("zero_based_pass_index must be nonnegative")
        if type(self.window_count) is not int or self.window_count <= 0:
            raise ValueError("window_count must be positive")
        if self.permutation != tuple(self.permutation) or sorted(self.permutation) != list(range(self.window_count)):
            raise ValueError("permutation must contain every window exactly once")
        expected = hashlib.sha256(self._preimage()).hexdigest()
        if self.schedule_sha256 != expected:
            raise ValueError("schedule_sha256 does not match the frozen schedule")

    def _preimage(self) -> bytes:
        return (
            b"VFE4-H6-FROZEN-BATCH-SCHEDULE-V1\x00"
            + self.shared_seed.to_bytes(8, "little")
            + self.zero_based_pass_index.to_bytes(8, "little")
            + self.window_count.to_bytes(8, "little")
            + self.batch_size.to_bytes(2, "little")
            + bytes((0,))
            + b"".join(index.to_bytes(8, "little") for index in self.permutation)
        )

    @classmethod
    def create(
        cls,
        *,
        zero_based_pass_index: int,
        window_count: int,
        permutation: tuple[int, ...],
    ) -> "FrozenBatchSchedule":
        frozen_permutation = tuple(permutation)
        preimage = (
            b"VFE4-H6-FROZEN-BATCH-SCHEDULE-V1\x00"
            + (2026072199).to_bytes(8, "little")
            + zero_based_pass_index.to_bytes(8, "little")
            + window_count.to_bytes(8, "little")
            + (8).to_bytes(2, "little")
            + bytes((0,))
            + b"".join(index.to_bytes(8, "little") for index in frozen_permutation)
        )
        return cls(
            "vfe4-h6-frozen-batch-schedule-v1",
            2026072199,
            zero_based_pass_index,
            window_count,
            8,
            False,
            frozen_permutation,
            hashlib.sha256(preimage).hexdigest(),
        )


@dataclass(frozen=True)
class DataIdentity:
    data_schema: Literal["vfe4-h6-data-identity-v1"]
    archive_sha256: str
    train_raw_sha256: str
    validation_raw_sha256: str
    test_raw_sha256: str
    train_tokens: EncodedTokenStorageIdentity
    validation_tokens: EncodedTokenStorageIdentity
    test_tokens: EncodedTokenStorageIdentity
    validation_fixture: ValidationSafetyFixture
    access_policy_sha256: str
    data_identity_sha256: str

    def __post_init__(self) -> None:
        if self.data_schema != "vfe4-h6-data-identity-v1":
            raise ValueError("unsupported data identity schema")
        for name in (
            "archive_sha256", "train_raw_sha256", "validation_raw_sha256",
            "test_raw_sha256", "access_policy_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        for identity in (self.train_tokens, self.validation_tokens, self.test_tokens):
            if type(identity) is not EncodedTokenStorageIdentity:
                raise ValueError("token identities must be EncodedTokenStorageIdentity records")
            identity.__post_init__()
        if type(self.validation_fixture) is not ValidationSafetyFixture:
            raise ValueError("validation_fixture must be a ValidationSafetyFixture")
        self.validation_fixture.__post_init__()
        if self.validation_fixture.validation_token_sha256 != self.validation_tokens.encoded_token_sha256:
            raise ValueError("validation fixture must bind the validation token identity")
        expected = _owned_hash("vfe4.h6.data-identity.v1", self._payload())
        if self.data_identity_sha256 != expected:
            raise ValueError("data_identity_sha256 does not match data identity fields")

    def _payload(self) -> dict[str, object]:
        return {
            "data_schema": self.data_schema,
            "archive_sha256": self.archive_sha256,
            "train_raw_sha256": self.train_raw_sha256,
            "validation_raw_sha256": self.validation_raw_sha256,
            "test_raw_sha256": self.test_raw_sha256,
            "train_token_sha256": self.train_tokens.encoded_token_sha256,
            "validation_token_sha256": self.validation_tokens.encoded_token_sha256,
            "test_token_sha256": self.test_tokens.encoded_token_sha256,
            "validation_fixture_sha256": self.validation_fixture.fixture_sha256,
            "access_policy_sha256": self.access_policy_sha256,
        }

    @classmethod
    def create(cls, **values: object) -> "DataIdentity":
        payload = {"data_schema": "vfe4-h6-data-identity-v1", **values}
        semantic = {
            "data_schema": payload["data_schema"],
            "archive_sha256": payload["archive_sha256"],
            "train_raw_sha256": payload["train_raw_sha256"],
            "validation_raw_sha256": payload["validation_raw_sha256"],
            "test_raw_sha256": payload["test_raw_sha256"],
            "train_token_sha256": payload["train_tokens"].encoded_token_sha256,
            "validation_token_sha256": payload["validation_tokens"].encoded_token_sha256,
            "test_token_sha256": payload["test_tokens"].encoded_token_sha256,
            "validation_fixture_sha256": payload["validation_fixture"].fixture_sha256,
            "access_policy_sha256": payload["access_policy_sha256"],
        }
        return cls(
            "vfe4-h6-data-identity-v1",
            values["archive_sha256"], values["train_raw_sha256"],
            values["validation_raw_sha256"], values["test_raw_sha256"],
            values["train_tokens"], values["validation_tokens"], values["test_tokens"],
            values["validation_fixture"], values["access_policy_sha256"],
            _owned_hash("vfe4.h6.data-identity.v1", semantic),
        )


@dataclass(frozen=True)
class CheckpointIdentity:
    experiment_identity_sha256: str
    config_sha256: str
    model_state_sha256: str
    data_identity_sha256: str
    estimator_sha256: str
    batch_schedule_sha256: str
    checkpoint_identity_sha256: str

    def __post_init__(self) -> None:
        names = (
            "experiment_identity_sha256", "config_sha256", "model_state_sha256",
            "data_identity_sha256", "estimator_sha256", "batch_schedule_sha256",
        )
        for name in names:
            _require_sha256(getattr(self, name), name)
        expected = _owned_hash(
            "vfe4.h6.checkpoint-identity.v1",
            {name: getattr(self, name) for name in names},
        )
        if self.checkpoint_identity_sha256 != expected:
            raise ValueError("checkpoint_identity_sha256 does not match checkpoint fields")

    @classmethod
    def create(cls, **values: str) -> "CheckpointIdentity":
        names = (
            "experiment_identity_sha256", "config_sha256", "model_state_sha256",
            "data_identity_sha256", "estimator_sha256", "batch_schedule_sha256",
        )
        payload = {name: values[name] for name in names}
        return cls(**payload, checkpoint_identity_sha256=_owned_hash(
            "vfe4.h6.checkpoint-identity.v1", payload
        ))


@dataclass(frozen=True)
class ExperimentIdentity:
    checkpoint_set_sha256: str
    current_candidate_sha256: str
    sealed_data_sha256: str
    access_policy_sha256: str
    analysis_sha256: str
    stream_protocol_sha256: str
    experiment_identity_sha256: str

    def __post_init__(self) -> None:
        names = (
            "checkpoint_set_sha256", "current_candidate_sha256", "sealed_data_sha256",
            "access_policy_sha256", "analysis_sha256", "stream_protocol_sha256",
        )
        for name in names:
            _require_sha256(getattr(self, name), name)
        expected = _owned_hash(
            "vfe4.h6.experiment-identity.v1",
            {name: getattr(self, name) for name in names},
        )
        if self.experiment_identity_sha256 != expected:
            raise ValueError("experiment_identity_sha256 does not match experiment fields")

    @classmethod
    def create(cls, **values: str) -> "ExperimentIdentity":
        names = (
            "checkpoint_set_sha256", "current_candidate_sha256", "sealed_data_sha256",
            "access_policy_sha256", "analysis_sha256", "stream_protocol_sha256",
        )
        payload = {name: values[name] for name in names}
        return cls(**payload, experiment_identity_sha256=_owned_hash(
            "vfe4.h6.experiment-identity.v1", payload
        ))


@dataclass(frozen=True)
class SealedSplitHandle:
    split: Literal["train", "validation", "test"]
    data_identity_sha256: str
    sealed_content_sha256: str
    access_policy_sha256: str
    handle_sha256: str

    def __post_init__(self) -> None:
        if self.split not in ("train", "validation", "test"):
            raise ValueError("unsupported sealed split")
        for name in ("data_identity_sha256", "sealed_content_sha256", "access_policy_sha256"):
            _require_sha256(getattr(self, name), name)
        expected = _owned_hash(
            "vfe4.h6.sealed-split-handle.v1",
            {
                "split": self.split,
                "data_identity_sha256": self.data_identity_sha256,
                "sealed_content_sha256": self.sealed_content_sha256,
                "access_policy_sha256": self.access_policy_sha256,
            },
        )
        if self.handle_sha256 != expected:
            raise ValueError("handle_sha256 does not match sealed split fields")

    @classmethod
    def create(cls, **values: object) -> "SealedSplitHandle":
        payload = dict(values)
        return cls(**payload, handle_sha256=_owned_hash(
            "vfe4.h6.sealed-split-handle.v1", payload
        ))


@runtime_checkable
class DurableTestOpeningCapability(Protocol):
    @property
    def proof_identity_sha256(self) -> str: ...


@runtime_checkable
class ValidatedTestOpening(Protocol):
    @property
    def proof_identity_sha256(self) -> str: ...


@dataclass(frozen=True)
class ZeroDimensionalBase:
    base_id: Literal["C0"]
    points: tuple[Literal["*"], ...]
    dimension: Literal[0]
    canonical_sha256: str

    def __post_init__(self) -> None:
        if (self.base_id, self.points, self.dimension) != ("C0", ("*",), 0):
            raise ValueError("ZeroDimensionalBase must be exactly C0={*}")
        expected = _owned_hash(
            "vfe4.h6.zero-dimensional-base.v1",
            {"base_id": self.base_id, "points": self.points, "dimension": self.dimension},
        )
        if self.canonical_sha256 != expected:
            raise ValueError("canonical_sha256 does not match ZeroDimensionalBase")

    @classmethod
    def create(cls) -> "ZeroDimensionalBase":
        payload = {"base_id": "C0", "points": ("*",), "dimension": 0}
        return cls("C0", ("*",), 0, _owned_hash("vfe4.h6.zero-dimensional-base.v1", payload))


@dataclass(frozen=True)
class CausalDagRow:
    receiver_t: int
    parents: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.receiver_t) is not int or self.receiver_t <= 0:
            raise ValueError("receiver_t must be a positive zero-based node label")
        if type(self.parents) is not tuple:
            raise ValueError("parents must be a tuple")
        if (
            not self.parents
            or any(type(parent) is not int or parent < 0 for parent in self.parents)
            or tuple(sorted(set(self.parents))) != self.parents
            or any(parent >= self.receiver_t for parent in self.parents)
        ):
            raise ValueError("parents must be unique increasing declared nodes below receiver")


@dataclass(frozen=True)
class CausalDag:
    labeling: Literal["zero_based"]
    node_labels: tuple[int, ...]
    rows: tuple[CausalDagRow, ...]
    canonical_sha256: str

    def __post_init__(self) -> None:
        if self.labeling != "zero_based":
            raise ValueError("CausalDag labeling must be zero_based")
        if (
            type(self.node_labels) is not tuple
            or len(self.node_labels) < 2
            or self.node_labels != tuple(range(len(self.node_labels)))
        ):
            raise ValueError(
                "canonical_sha256 mismatch: node_labels must be contiguous zero-based labels"
            )
        if type(self.rows) is not tuple or not all(type(row) is CausalDagRow for row in self.rows):
            raise ValueError("rows must be a tuple of exact CausalDagRow records")
        receivers = tuple(row.receiver_t for row in self.rows)
        if receivers != self.node_labels[1:]:
            raise ValueError("receiver rows must name every noninitial receiver exactly once")
        declared = set(self.node_labels)
        if any(parent not in declared for row in self.rows for parent in row.parents):
            raise ValueError("every parent must be a declared node")
        expected = _owned_hash(
            "vfe4.h6.causal-dag.v1",
            {
                "labeling": self.labeling,
                "node_labels": self.node_labels,
                "rows": tuple((row.receiver_t, row.parents) for row in self.rows),
            },
        )
        if self.canonical_sha256 != expected:
            raise ValueError("canonical_sha256 does not match CausalDag")

    @classmethod
    def create(
        cls, *, node_labels: tuple[int, ...], rows: tuple[CausalDagRow, ...]
    ) -> "CausalDag":
        payload = {
            "labeling": "zero_based",
            "node_labels": tuple(node_labels),
            "rows": tuple((row.receiver_t, row.parents) for row in rows),
        }
        return cls(
            "zero_based",
            tuple(node_labels),
            tuple(rows),
            _owned_hash("vfe4.h6.causal-dag.v1", payload),
        )


@dataclass(frozen=True)
class H6LanguageStructure:
    base: ZeroDimensionalBase
    dag: CausalDag
    receiver_labels: tuple[int, ...]
    structure_sha256: str

    def __post_init__(self) -> None:
        if type(self.base) is not ZeroDimensionalBase or type(self.dag) is not CausalDag:
            raise ValueError("structure requires exact base and DAG records")
        expected_receivers = tuple(row.receiver_t for row in self.dag.rows)
        if self.receiver_labels != expected_receivers:
            raise ValueError(
                "structure_sha256 mismatch: receiver_labels must equal the ordered DAG receivers"
            )
        expected = _owned_hash(
            "vfe4.h6.language-structure.v1",
            {
                "base_sha256": self.base.canonical_sha256,
                "dag_sha256": self.dag.canonical_sha256,
                "receiver_labels": self.receiver_labels,
            },
        )
        if self.structure_sha256 != expected:
            raise ValueError("structure_sha256 does not match H6LanguageStructure")

    @classmethod
    def create(
        cls,
        *,
        base: ZeroDimensionalBase,
        dag: CausalDag,
        receiver_labels: tuple[int, ...],
    ) -> "H6LanguageStructure":
        payload = {
            "base_sha256": base.canonical_sha256,
            "dag_sha256": dag.canonical_sha256,
            "receiver_labels": tuple(receiver_labels),
        }
        return cls(
            base,
            dag,
            tuple(receiver_labels),
            _owned_hash("vfe4.h6.language-structure.v1", payload),
        )


@dataclass(frozen=True, init=False)
class FrozenTensorSnapshot:
    __owned: torch.Tensor
    dtype: str
    shape: tuple[int, ...]
    device: str
    contiguous: bool
    requires_grad: bool
    storage_version: int
    raw_bytes_sha256: str

    @classmethod
    def capture(cls, value: torch.Tensor) -> "FrozenTensorSnapshot":
        if not isinstance(value, torch.Tensor):
            raise ValueError("value must be a torch.Tensor")
        owned = value.contiguous().clone()
        instance = object.__new__(cls)
        object.__setattr__(instance, "_FrozenTensorSnapshot__owned", owned)
        object.__setattr__(instance, "dtype", str(owned.dtype).removeprefix("torch."))
        object.__setattr__(instance, "shape", tuple(int(size) for size in owned.shape))
        object.__setattr__(instance, "device", str(owned.device))
        object.__setattr__(instance, "contiguous", True)
        object.__setattr__(instance, "requires_grad", bool(owned.requires_grad))
        object.__setattr__(instance, "storage_version", int(owned._version))
        object.__setattr__(
            instance,
            "raw_bytes_sha256",
            hashlib.sha256(_tensor_raw_bytes(owned)).hexdigest(),
        )
        instance.assert_intact()
        return instance

    def assert_intact(self) -> None:
        owned = self.__owned
        valid = (
            isinstance(owned, torch.Tensor)
            and str(owned.dtype).removeprefix("torch.") == self.dtype
            and tuple(int(size) for size in owned.shape) == self.shape
            and str(owned.device) == self.device
            and bool(owned.is_contiguous()) is self.contiguous
            and bool(owned.requires_grad) is self.requires_grad
            and int(owned._version) == self.storage_version
            and hashlib.sha256(_tensor_raw_bytes(owned)).hexdigest() == self.raw_bytes_sha256
        )
        if not valid:
            raise ValueError("FrozenTensorSnapshot integrity check failed")

    def value(self) -> torch.Tensor:
        self.assert_intact()
        return self.__owned.clone()


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


def _factor_payload(term: "H6FactorTerm") -> dict[str, object]:
    term.value.assert_intact()
    return {
        "receiver_t": term.receiver_t,
        "partition": term.partition,
        "factor_identity_sha256": term.factor_identity_sha256,
        "value": _snapshot_payload(term.value),
    }


@dataclass(frozen=True)
class H6FactorTerm:
    receiver_t: int
    partition: Literal[
        "emission",
        "initial",
        "state_source",
        "model_source",
        "state_transition",
        "model_transition",
        "entropy",
    ]
    factor_identity_sha256: str
    value: FrozenTensorSnapshot

    def __post_init__(self) -> None:
        if type(self.receiver_t) is not int or self.receiver_t < 0:
            raise ValueError("receiver_t must be a nonnegative integer")
        if self.partition not in {
            "emission", "initial", "state_source", "model_source",
            "state_transition", "model_transition", "entropy",
        }:
            raise ValueError("unsupported H6 factor partition")
        _require_sha256(self.factor_identity_sha256, "factor_identity_sha256")
        if type(self.value) is not FrozenTensorSnapshot:
            raise ValueError("value must be a FrozenTensorSnapshot")
        self.value.assert_intact()


@dataclass(frozen=True)
class H6LanguageElboTerms:
    horizon: int
    ordered_factor_terms: tuple[H6FactorTerm, ...]
    emission_terms: tuple[H6FactorTerm, ...]
    initial_terms: tuple[H6FactorTerm, ...]
    state_source_terms: tuple[H6FactorTerm, ...]
    model_source_terms: tuple[H6FactorTerm, ...]
    state_transition_terms: tuple[H6FactorTerm, ...]
    model_transition_terms: tuple[H6FactorTerm, ...]
    entropy_terms: tuple[H6FactorTerm, ...]
    complete_decomposition: FrozenTensorSnapshot
    total_language_elbo: FrozenTensorSnapshot
    equality_checked: Literal[True]
    canonical_sha256: str

    def __post_init__(self) -> None:
        if type(self.horizon) is not int or self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if type(self.ordered_factor_terms) is not tuple or not self.ordered_factor_terms:
            raise ValueError("ordered_factor_terms must be nonempty")
        if len({term.factor_identity_sha256 for term in self.ordered_factor_terms}) != len(self.ordered_factor_terms):
            raise ValueError("factor identities must be unique")
        partitions = {
            "emission": self.emission_terms,
            "initial": self.initial_terms,
            "state_source": self.state_source_terms,
            "model_source": self.model_source_terms,
            "state_transition": self.state_transition_terms,
            "model_transition": self.model_transition_terms,
            "entropy": self.entropy_terms,
        }
        for name, terms in partitions.items():
            expected = tuple(term for term in self.ordered_factor_terms if term.partition == name)
            if terms != expected:
                raise ValueError(f"{name} terms do not match ordered decomposition")
        if any(not terms for terms in partitions.values()):
            raise ValueError("complete language ELBO requires all seven partitions")
        required_receivers = set(range(1, self.horizon + 1))
        if {term.receiver_t for term in self.initial_terms} != {0}:
            raise ValueError("initial terms must cover receiver zero exactly")
        for name, terms in partitions.items():
            if name == "initial":
                continue
            if {term.receiver_t for term in terms} != required_receivers:
                raise ValueError(f"{name} terms do not cover the exact language horizon")
        if self.equality_checked is not True:
            raise ValueError("equality_checked must be true")
        self.complete_decomposition.assert_intact()
        self.total_language_elbo.assert_intact()
        if not torch.equal(self.complete_decomposition.value(), self.total_language_elbo.value()):
            raise ValueError("complete decomposition does not equal total language ELBO")
        if any(term.receiver_t > self.horizon for term in self.ordered_factor_terms):
            raise ValueError("factor receiver exceeds the language horizon")
        expected = _owned_hash("vfe4.h6.language-elbo-terms.v1", self._payload())
        if self.canonical_sha256 != expected:
            raise ValueError("canonical_sha256 does not match language ELBO terms")

    def _payload(self) -> dict[str, object]:
        return {
            "horizon": self.horizon,
            "ordered_factor_terms": tuple(_factor_payload(term) for term in self.ordered_factor_terms),
            "emission_factor_ids": tuple(term.factor_identity_sha256 for term in self.emission_terms),
            "initial_factor_ids": tuple(term.factor_identity_sha256 for term in self.initial_terms),
            "state_source_factor_ids": tuple(term.factor_identity_sha256 for term in self.state_source_terms),
            "model_source_factor_ids": tuple(term.factor_identity_sha256 for term in self.model_source_terms),
            "state_transition_factor_ids": tuple(term.factor_identity_sha256 for term in self.state_transition_terms),
            "model_transition_factor_ids": tuple(term.factor_identity_sha256 for term in self.model_transition_terms),
            "entropy_factor_ids": tuple(term.factor_identity_sha256 for term in self.entropy_terms),
            "complete_decomposition": _snapshot_payload(self.complete_decomposition),
            "total_language_elbo": _snapshot_payload(self.total_language_elbo),
            "equality_checked": self.equality_checked,
        }

    @classmethod
    def create(
        cls,
        *,
        horizon: int,
        ordered_factor_terms: tuple[H6FactorTerm, ...],
        total_language_elbo: torch.Tensor,
    ) -> "H6LanguageElboTerms":
        terms = tuple(ordered_factor_terms)
        if not terms:
            raise ValueError("ordered_factor_terms must be nonempty")
        decomposition = terms[0].value.value()
        for term in terms[1:]:
            decomposition = decomposition + term.value.value()
        total = total_language_elbo
        decomposition_snapshot = FrozenTensorSnapshot.capture(decomposition)
        total_snapshot = FrozenTensorSnapshot.capture(total)
        partitions = {
            name: tuple(term for term in terms if term.partition == name)
            for name in (
                "emission", "initial", "state_source", "model_source",
                "state_transition", "model_transition", "entropy",
            )
        }
        values: dict[str, object] = {
            "horizon": horizon,
            "ordered_factor_terms": terms,
            "emission_terms": partitions["emission"],
            "initial_terms": partitions["initial"],
            "state_source_terms": partitions["state_source"],
            "model_source_terms": partitions["model_source"],
            "state_transition_terms": partitions["state_transition"],
            "model_transition_terms": partitions["model_transition"],
            "entropy_terms": partitions["entropy"],
            "complete_decomposition": decomposition_snapshot,
            "total_language_elbo": total_snapshot,
            "equality_checked": True,
        }
        provisional = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(provisional, name, value)
        digest = _owned_hash("vfe4.h6.language-elbo-terms.v1", provisional._payload())
        return cls(**values, canonical_sha256=digest)


def _is_parent_specific_emission_endpoint(config: ArmConfig) -> bool:
    return (
        config.arm is ArmId.A5
        and config.config_id == _H6_PARENT_SPECIFIC_EMISSION_CONFIG_ID
        and config.latent_enabled
        and config.state_channel_enabled
        and config.model_channel_enabled
        and config.source_mode == "categorical"
        and config.map_mode == "shared_vertex_coboundary"
        and config.recognition_family == "structured"
        and config.recognition_conditioning == "smoothing"
        and config.prior_variant == "parent_specific_pooled_prefix"
        and config.mixture_mode == "exact"
        and config.objective_kind == "emission_only_ablation_non_elbo"
    )


def _supports_live_source_prior_trace(config: ArmConfig) -> bool:
    complete_endpoint = (
        config.arm in (ArmId.A2, ArmId.A5)
        and config.objective_kind == "complete_elbo"
        and config.prior_variant in ("fixed", "parent_specific_pooled_prefix")
        and config.mixture_mode in ("exact", "moment_projection")
    )
    return complete_endpoint or _is_parent_specific_emission_endpoint(config)


H6_EMISSION_ONLY_ABLATION_HASH_DOMAIN = (
    "vfe4.h6.emission-only-ablation.v3"
)


@dataclass(frozen=True, slots=True, init=False)
class H6SourcePriorTrace:
    """Live-prior provenance sealed by ``BuiltArm`` before ELBO evaluation."""

    endpoint_config: ArmConfig
    model_family_sha256: str
    prior_variant: Literal["fixed", "parent_specific_pooled_prefix"]
    prior_type: Literal[
        "FixedSourcePrior",
        "ParentSpecificPooledPrefixSourcePrior",
    ]
    prior_model_state_sha256: str
    ordered_source_factor_identities: tuple[
        tuple[Literal["model_source", "state_source"], int, str], ...
    ]
    trace_sha256: str

    def __init__(self) -> None:
        raise TypeError(
            "H6SourcePriorTrace is BuiltArm-only; use the live arm "
            "evaluation seam"
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "endpoint_config_sha256": self.endpoint_config.config_sha256,
            "model_family_sha256": self.model_family_sha256,
            "prior_variant": self.prior_variant,
            "prior_type": self.prior_type,
            "prior_model_state_sha256": self.prior_model_state_sha256,
            "ordered_source_factor_identities": (
                self.ordered_source_factor_identities
            ),
        }

    def __post_init__(self) -> None:
        if type(self.endpoint_config) is not ArmConfig:
            raise ValueError("trace endpoint_config must be exact")
        self.endpoint_config.__post_init__()
        if not _supports_live_source_prior_trace(self.endpoint_config):
            raise ValueError(
                "source-prior trace requires a supported complete A2/A5 "
                "endpoint or the literal parent-specific OBJECTIVE endpoint"
            )
        _require_sha256(self.model_family_sha256, "model_family_sha256")
        _require_sha256(
            self.prior_model_state_sha256,
            "prior_model_state_sha256",
        )
        expected_type = (
            "FixedSourcePrior"
            if self.endpoint_config.prior_variant == "fixed"
            else "ParentSpecificPooledPrefixSourcePrior"
        )
        if (
            self.prior_variant != self.endpoint_config.prior_variant
            or self.prior_type != expected_type
        ):
            raise ValueError(
                "source-prior trace relabels the configured prior variant"
            )
        expected_slots = tuple(
            (partition, receiver_t)
            for receiver_t in range(1, self.endpoint_config.horizon + 1)
            for partition in ("model_source", "state_source")
        )
        observed_slots = tuple(
            (partition, receiver_t)
            for partition, receiver_t, _ in (
                self.ordered_source_factor_identities
            )
        )
        identities = tuple(
            identity
            for _, _, identity in self.ordered_source_factor_identities
        )
        if (
            observed_slots != expected_slots
            or len(set(identities)) != len(identities)
        ):
            raise ValueError(
                "source-prior trace identities are absent, duplicated, or "
                "reordered"
            )
        for identity in identities:
            _require_sha256(identity, "source_factor_identity_sha256")
        if self.trace_sha256 != _owned_hash(
            "vfe4.h6.source-prior-trace.v1",
            self.canonical_payload(),
        ):
            raise ValueError("source-prior trace identity is stale")

    @classmethod
    def _from_live_prior(
        cls,
        *,
        endpoint_config: ArmConfig,
        source_prior: (
            FixedSourcePrior | ParentSpecificPooledPrefixSourcePrior
        ),
        ordered_source_factors: tuple[NormalizedSourceFactor, ...],
    ) -> "H6SourcePriorTrace":
        from vfe4.generative.source_priors import (
            FixedSourcePrior,
            NormalizedSourceFactor,
            ParentSpecificPooledPrefixSourcePrior,
        )
        from vfe4.predictive.identities import canonical_model_state_sha256

        if type(endpoint_config) is not ArmConfig:
            raise ValueError("trace endpoint_config must be exact")
        endpoint_config.__post_init__()
        if (
            not _supports_live_source_prior_trace(endpoint_config)
        ):
            raise ValueError(
                "source-prior trace requires a supported complete A2/A5 "
                "endpoint or the literal parent-specific OBJECTIVE endpoint"
            )
        expected_prior_type = (
            FixedSourcePrior
            if endpoint_config.prior_variant == "fixed"
            else ParentSpecificPooledPrefixSourcePrior
        )
        if type(source_prior) is not expected_prior_type:
            raise ValueError(
                "live source-prior type does not match endpoint prior_variant"
            )
        expected_model_family_sha256 = arm_model_family_sha256(endpoint_config)
        expected_receivers = tuple(range(1, endpoint_config.horizon + 1))
        source_prior.structure.__post_init__()
        source_prior.vocabulary.__post_init__()
        if (
            source_prior.predictor_config_sha256
            != endpoint_config.config_sha256
            or source_prior.model_family_sha256
            != expected_model_family_sha256
            or source_prior.vocabulary != endpoint_config.vocabulary
            or source_prior.structure.receiver_labels != expected_receivers
        ):
            raise ValueError(
                "live source prior does not match the endpoint config, "
                "model family, vocabulary, or horizon"
            )
        if type(ordered_source_factors) is not tuple:
            raise ValueError("ordered_source_factors must be an exact tuple")
        expected_slots = tuple(
            (bank, receiver_t)
            for receiver_t in expected_receivers
            for bank in ("model", "state")
        )
        observed_slots: list[tuple[str, int]] = []
        ordered_source_factor_identities: list[
            tuple[Literal["model_source", "state_source"], int, str]
        ] = []
        for factor in ordered_source_factors:
            if type(factor) is not NormalizedSourceFactor:
                raise ValueError(
                    "ordered_source_factors must contain exact "
                    "NormalizedSourceFactor records"
                )
            factor.__post_init__()
            key = factor.mask_case_key
            observed_slots.append((key.bank, key.receiver_t))
            expected_support = tuple(
                source_t
                in source_prior.structure.dag.rows[
                    expected_receivers.index(key.receiver_t)
                ].parents
                for source_t in range(key.receiver_t)
            ) if key.receiver_t in expected_receivers else ()
            if (
                key.predictor_config_sha256
                != endpoint_config.config_sha256
                or key.model_family_sha256
                != expected_model_family_sha256
                or key.prior_variant != endpoint_config.prior_variant
                or key.fixture_sha256 != source_prior.fixture_sha256
                or key.vocabulary_sha256 != source_prior.vocabulary_sha256
                or factor.support_mask != expected_support
            ):
                raise ValueError(
                    "normalized source factor does not match the live prior "
                    "config, family, variant, vocabulary, fixture, or support"
                )
            partition: Literal["model_source", "state_source"] = (
                "model_source" if key.bank == "model" else "state_source"
            )
            ordered_source_factor_identities.append(
                (partition, key.receiver_t, factor.factor_identity_sha256)
            )
        if tuple(observed_slots) != expected_slots:
            raise ValueError(
                "normalized source factors have a mismatched bank, receiver, "
                "or order"
            )
        prior_type: Literal[
            "FixedSourcePrior",
            "ParentSpecificPooledPrefixSourcePrior",
        ] = (
            "FixedSourcePrior"
            if type(source_prior) is FixedSourcePrior
            else "ParentSpecificPooledPrefixSourcePrior"
        )
        prior_model_state_sha256 = canonical_model_state_sha256(source_prior)
        values: dict[str, object] = {
            "endpoint_config": endpoint_config,
            "model_family_sha256": expected_model_family_sha256,
            "prior_variant": endpoint_config.prior_variant,
            "prior_type": prior_type,
            "prior_model_state_sha256": prior_model_state_sha256,
            "ordered_source_factor_identities": tuple(
                ordered_source_factor_identities
            ),
        }
        instance = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(instance, name, value)
        object.__setattr__(
            instance,
            "trace_sha256",
            _owned_hash("vfe4.h6.source-prior-trace.v1", {
                "endpoint_config_sha256": endpoint_config.config_sha256,
                "model_family_sha256": expected_model_family_sha256,
                "prior_variant": endpoint_config.prior_variant,
                "prior_type": prior_type,
                "prior_model_state_sha256": prior_model_state_sha256,
                "ordered_source_factor_identities": tuple(
                    ordered_source_factor_identities
                ),
            }),
        )
        instance.__post_init__()
        return instance


def h6_source_law_marker_identity(
    *,
    endpoint_config: ArmConfig,
    projection_error: FrozenTensorSnapshot | None,
) -> str:
    """Derive the typed exact/projected source-law marker identity."""

    if type(endpoint_config) is not ArmConfig:
        raise ValueError("endpoint_config must be an exact ArmConfig")
    endpoint_config.__post_init__()
    if not _supports_live_source_prior_trace(endpoint_config):
        raise ValueError(
            "source law requires a supported complete A2/A5 endpoint or the "
            "literal parent-specific OBJECTIVE endpoint"
        )
    if endpoint_config.mixture_mode == "exact":
        if projection_error is not None:
            raise ValueError("exact source law cannot carry projection error")
        kind = "exact_source_mixture"
        projection_payload = None
    else:
        if type(projection_error) is not FrozenTensorSnapshot:
            raise ValueError(
                "moment-projected source law requires projection error"
            )
        projection_error.assert_intact()
        error = projection_error.value()
        if (
            error.ndim != 0
            or not error.is_floating_point()
            or not bool(torch.isfinite(error))
            or not bool(error >= 0.0)
        ):
            raise ValueError(
                "projection error must be a finite nonnegative scalar"
            )
        kind = "moment_projection"
        projection_payload = _snapshot_payload(projection_error)
    return _owned_hash(
        "vfe4.h6.source-law-marker.v1",
        {
            "kind": kind,
            "endpoint_config_sha256": endpoint_config.config_sha256,
            "prior_variant": endpoint_config.prior_variant,
            "mixture_mode": endpoint_config.mixture_mode,
            "projection_error": projection_payload,
        },
    )


def h6_source_law_identity(
    *,
    endpoint_config: ArmConfig,
    source_prior_trace: H6SourcePriorTrace,
    projection_error: FrozenTensorSnapshot | None,
) -> str:
    """Bind one typed source-law marker to the live-prior trace."""

    if type(source_prior_trace) is not H6SourcePriorTrace:
        raise ValueError("source_prior_trace must be exact")
    source_prior_trace.__post_init__()
    if source_prior_trace.endpoint_config != endpoint_config:
        raise ValueError("source-prior trace belongs to another endpoint")
    marker_sha256 = h6_source_law_marker_identity(
        endpoint_config=endpoint_config,
        projection_error=projection_error,
    )
    return _owned_hash(
        "vfe4.h6.source-law.v3",
        {
            "source_law_marker_identity_sha256": marker_sha256,
            "source_prior_trace_sha256": source_prior_trace.trace_sha256,
        },
    )


@dataclass(frozen=True)
class H6EndpointLanguageElboTerms:
    """Complete ELBO bound to one actual endpoint and source-law trace."""

    endpoint_config: ArmConfig
    prior_variant: Literal["fixed", "parent_specific_pooled_prefix"]
    mixture_mode: Literal["exact", "moment_projection"]
    source_prior_trace: H6SourcePriorTrace
    projection_error: FrozenTensorSnapshot | None
    source_law_marker_identity_sha256: str
    source_law_identity_sha256: str
    terms: H6LanguageElboTerms
    canonical_sha256: str

    @property
    def endpoint_config_sha256(self) -> str:
        return self.endpoint_config.config_sha256

    @property
    def source_prior_trace_sha256(self) -> str:
        return self.source_prior_trace.trace_sha256

    @property
    def horizon(self) -> int:
        return self.terms.horizon

    @property
    def ordered_factor_terms(self) -> tuple[H6FactorTerm, ...]:
        return self.terms.ordered_factor_terms

    @property
    def total_language_elbo(self) -> FrozenTensorSnapshot:
        return self.terms.total_language_elbo

    @property
    def complete_decomposition(self) -> FrozenTensorSnapshot:
        return self.terms.complete_decomposition

    @property
    def emission_terms(self) -> tuple[H6FactorTerm, ...]:
        return self.terms.emission_terms

    @property
    def initial_terms(self) -> tuple[H6FactorTerm, ...]:
        return self.terms.initial_terms

    @property
    def state_source_terms(self) -> tuple[H6FactorTerm, ...]:
        return self.terms.state_source_terms

    @property
    def model_source_terms(self) -> tuple[H6FactorTerm, ...]:
        return self.terms.model_source_terms

    @property
    def state_transition_terms(self) -> tuple[H6FactorTerm, ...]:
        return self.terms.state_transition_terms

    @property
    def model_transition_terms(self) -> tuple[H6FactorTerm, ...]:
        return self.terms.model_transition_terms

    @property
    def entropy_terms(self) -> tuple[H6FactorTerm, ...]:
        return self.terms.entropy_terms

    def _payload(self) -> dict[str, object]:
        return {
            "endpoint_config_sha256": self.endpoint_config.config_sha256,
            "prior_variant": self.prior_variant,
            "mixture_mode": self.mixture_mode,
            "source_prior_trace_sha256": (
                self.source_prior_trace.trace_sha256
            ),
            "projection_error": (
                None
                if self.projection_error is None
                else _snapshot_payload(self.projection_error)
            ),
            "source_law_marker_identity_sha256": (
                self.source_law_marker_identity_sha256
            ),
            "source_law_identity_sha256": self.source_law_identity_sha256,
            "language_elbo_terms_sha256": self.terms.canonical_sha256,
        }

    def __post_init__(self) -> None:
        if type(self.endpoint_config) is not ArmConfig:
            raise ValueError("endpoint_config must be an exact ArmConfig")
        self.endpoint_config.__post_init__()
        if (
            self.endpoint_config.arm not in (ArmId.A2, ArmId.A5)
            or self.endpoint_config.objective_kind != "complete_elbo"
            or self.endpoint_config.prior_variant
            not in ("fixed", "parent_specific_pooled_prefix")
            or self.endpoint_config.mixture_mode
            not in ("exact", "moment_projection")
        ):
            raise ValueError(
                "endpoint-bound complete ELBO requires a categorical A2/A5 "
                "endpoint"
            )
        if (
            self.prior_variant != self.endpoint_config.prior_variant
            or self.mixture_mode != self.endpoint_config.mixture_mode
        ):
            raise ValueError(
                "ELBO source treatment does not match the endpoint config"
            )
        if type(self.source_prior_trace) is not H6SourcePriorTrace:
            raise ValueError("source_prior_trace must be exact")
        self.source_prior_trace.__post_init__()
        if (
            self.source_prior_trace.endpoint_config != self.endpoint_config
            or self.source_prior_trace.prior_variant != self.prior_variant
        ):
            raise ValueError(
                "source-prior trace belongs to another endpoint or prior"
            )
        _require_sha256(
            self.source_law_marker_identity_sha256,
            "source_law_marker_identity_sha256",
        )
        _require_sha256(
            self.source_law_identity_sha256,
            "source_law_identity_sha256",
        )
        if (
            self.source_law_marker_identity_sha256
            != h6_source_law_marker_identity(
                endpoint_config=self.endpoint_config,
                projection_error=self.projection_error,
            )
        ):
            raise ValueError("source-law marker identity is stale")
        if self.source_law_identity_sha256 != h6_source_law_identity(
            endpoint_config=self.endpoint_config,
            source_prior_trace=self.source_prior_trace,
            projection_error=self.projection_error,
        ):
            raise ValueError("source-law identity does not match its trace")
        if type(self.terms) is not H6LanguageElboTerms:
            raise ValueError("terms must be exact H6LanguageElboTerms")
        self.terms.__post_init__()
        if self.terms.horizon != self.endpoint_config.horizon:
            raise ValueError("ELBO horizon does not match the endpoint config")
        if self.canonical_sha256 != _owned_hash(
            "vfe4.h6.endpoint-language-elbo-terms.v1",
            self._payload(),
        ):
            raise ValueError(
                "canonical_sha256 does not match endpoint-bound ELBO terms"
            )

    @classmethod
    def create(
        cls,
        *,
        endpoint_config: ArmConfig,
        prior_variant: Literal["fixed", "parent_specific_pooled_prefix"],
        mixture_mode: Literal["exact", "moment_projection"],
        source_prior_trace: H6SourcePriorTrace,
        projection_error: FrozenTensorSnapshot | None,
        source_law_marker_identity_sha256: str,
        terms: H6LanguageElboTerms,
    ) -> "H6EndpointLanguageElboTerms":
        source_law_identity_sha256 = h6_source_law_identity(
            endpoint_config=endpoint_config,
            source_prior_trace=source_prior_trace,
            projection_error=projection_error,
        )
        values: dict[str, object] = {
            "endpoint_config": endpoint_config,
            "prior_variant": prior_variant,
            "mixture_mode": mixture_mode,
            "source_prior_trace": source_prior_trace,
            "projection_error": projection_error,
            "source_law_marker_identity_sha256": (
                source_law_marker_identity_sha256
            ),
            "source_law_identity_sha256": source_law_identity_sha256,
            "terms": terms,
        }
        provisional = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(provisional, name, value)
        return cls(
            **values,  # type: ignore[arg-type]
            canonical_sha256=_owned_hash(
                "vfe4.h6.endpoint-language-elbo-terms.v1",
                provisional._payload(),
            ),
        )


@dataclass(frozen=True, slots=True, init=False)
class EmissionOnlyAblationTerms:
    objective_kind: Literal["emission_only_ablation_non_elbo"]
    endpoint_config: ArmConfig
    model_family_sha256: str
    canonical_model_state_sha256: str
    prior_variant: Literal["parent_specific_pooled_prefix"]
    source_prior_trace: H6SourcePriorTrace
    source_law_marker_identity_sha256: str
    source_law_identity_sha256: str
    ordered_emission_terms: tuple[H6FactorTerm, ...]
    total: FrozenTensorSnapshot
    canonical_sha256: str

    def __init__(self) -> None:
        raise TypeError(
            "EmissionOnlyAblationTerms is BuiltArm-only; use the live arm "
            "evaluation seam"
        )

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("EmissionOnlyAblationTerms is sealed")

    @property
    def endpoint_config_sha256(self) -> str:
        return self.endpoint_config.config_sha256

    @property
    def source_prior_trace_sha256(self) -> str:
        return self.source_prior_trace.trace_sha256

    @property
    def is_elbo(self) -> Literal[False]:
        return False

    def _payload(self) -> dict[str, object]:
        return {
            "objective_kind": self.objective_kind,
            "endpoint_config_sha256": self.endpoint_config.config_sha256,
            "model_family_sha256": self.model_family_sha256,
            "canonical_model_state_sha256": (
                self.canonical_model_state_sha256
            ),
            "prior_variant": self.prior_variant,
            "source_prior_trace_sha256": self.source_prior_trace.trace_sha256,
            "source_law_marker_identity_sha256": (
                self.source_law_marker_identity_sha256
            ),
            "source_law_identity_sha256": self.source_law_identity_sha256,
            "ordered_emission_terms": tuple(
                _factor_payload(term) for term in self.ordered_emission_terms
            ),
            "total": _snapshot_payload(self.total),
        }

    def __post_init__(self) -> None:
        if self.objective_kind != "emission_only_ablation_non_elbo":
            raise ValueError("emission-only terms must remain a non-ELBO ablation")
        if type(self.endpoint_config) is not ArmConfig:
            raise ValueError("emission-only endpoint_config must be exact")
        self.endpoint_config.__post_init__()
        if not _is_parent_specific_emission_endpoint(self.endpoint_config):
            raise ValueError(
                "emission-only training is authorized only for the literal "
                "parent-specific OBJECTIVE endpoint"
            )
        expected_model_family_sha256 = arm_model_family_sha256(
            self.endpoint_config
        )
        _require_sha256(self.model_family_sha256, "model_family_sha256")
        if self.model_family_sha256 != expected_model_family_sha256:
            raise ValueError(
                "emission-only model family does not match its literal endpoint"
            )
        _require_sha256(
            self.canonical_model_state_sha256,
            "canonical_model_state_sha256",
        )
        if self.prior_variant != "parent_specific_pooled_prefix":
            raise ValueError(
                "emission-only OBJECTIVE endpoint requires the parent-specific "
                "pooled-prefix prior"
            )
        if type(self.source_prior_trace) is not H6SourcePriorTrace:
            raise ValueError("emission-only source-prior trace must be exact")
        self.source_prior_trace.__post_init__()
        if (
            self.source_prior_trace.endpoint_config != self.endpoint_config
            or self.source_prior_trace.model_family_sha256
            != self.model_family_sha256
            or self.source_prior_trace.prior_variant != self.prior_variant
            or self.source_prior_trace.prior_type
            != "ParentSpecificPooledPrefixSourcePrior"
        ):
            raise ValueError(
                "emission-only source-prior trace belongs to another endpoint, "
                "model family, or prior"
            )
        expected_marker = h6_source_law_marker_identity(
            endpoint_config=self.endpoint_config,
            projection_error=None,
        )
        expected_law = h6_source_law_identity(
            endpoint_config=self.endpoint_config,
            source_prior_trace=self.source_prior_trace,
            projection_error=None,
        )
        if (
            self.source_law_marker_identity_sha256 != expected_marker
            or self.source_law_identity_sha256 != expected_law
        ):
            raise ValueError(
                "emission-only source-law identity is stale or belongs to "
                "another live prior"
            )
        expected_receivers = tuple(range(1, self.endpoint_config.horizon + 1))
        if tuple(
            term.receiver_t for term in self.ordered_emission_terms
        ) != expected_receivers:
            raise ValueError(
                "emission-only terms must cover the literal endpoint horizon "
                "exactly once and in order"
            )
        for term in self.ordered_emission_terms:
            term.__post_init__()
            if term.partition != "emission":
                raise ValueError(
                    "emission-only ablation accepts only emission factors"
                )
        self.total.assert_intact()
        total = self.ordered_emission_terms[0].value.value()
        for term in self.ordered_emission_terms[1:]:
            total = total + term.value.value()
        if not torch.equal(total, self.total.value()):
            raise ValueError(
                "emission-only total does not equal its ordered emission terms"
            )
        expected = _owned_hash(
            H6_EMISSION_ONLY_ABLATION_HASH_DOMAIN,
            self._payload(),
        )
        if self.canonical_sha256 != expected:
            raise ValueError("canonical_sha256 does not match emission-only terms")

def arm_model_family_sha256(config: ArmConfig) -> str:
    """Reproduce the exact H6 arm-factory family identity without building a model."""

    if type(config) is not ArmConfig:
        raise ValueError("config must be an exact ArmConfig")
    config.__post_init__()
    v2_a5_configs = {
        (
            "h6-a5-structured-parent-specific-prefix-exact-complete-"
            "latent-smoothing-v2"
        ),
        (
            "h6-a5-structured-parent-specific-prefix-exact-emission-"
            "latent-smoothing-v2"
        ),
    }
    if config.arm is ArmId.A0:
        factory = "build_a0@h6-arm-v2"
    elif config.arm is ArmId.A5 and config.config_id in v2_a5_configs:
        factory = "build_a5@h6-arm-v2"
    else:
        factory = f"build_{config.arm.value.lower()}@h6-arm-v1"
    return _owned_hash(
        "vfe4.h6.arm-model-family.v1",
        {
            "config_sha256": config.config_sha256,
            "factory": factory,
        },
    )


@dataclass(frozen=True)
class H6PrefixProfilePair:
    """Hash-bound small/production pair for one exact Prefix estimator profile."""

    profile_id: str
    small_arm_config: ArmConfig
    production_arm_config: ArmConfig
    estimator: EstimatorSpec
    small_structure: H6LanguageStructure
    production_structure: H6LanguageStructure
    data_safety_sha256: str
    small_model_family_sha256: str
    production_model_family_sha256: str
    profile_pair_sha256: str

    def __post_init__(self) -> None:
        _require_nonempty(self.profile_id, "profile_id")
        if type(self.small_arm_config) is not ArmConfig:
            raise ValueError("small_arm_config must be an exact ArmConfig")
        if type(self.production_arm_config) is not ArmConfig:
            raise ValueError("production_arm_config must be an exact ArmConfig")
        if type(self.estimator) is not EstimatorSpec:
            raise ValueError("estimator must be an exact EstimatorSpec")
        if type(self.small_structure) is not H6LanguageStructure:
            raise ValueError(
                "small_structure must be an exact H6LanguageStructure"
            )
        if type(self.production_structure) is not H6LanguageStructure:
            raise ValueError(
                "production_structure must be an exact H6LanguageStructure"
            )
        self.small_arm_config.__post_init__()
        self.production_arm_config.__post_init__()
        self.estimator.__post_init__()
        self.small_structure.__post_init__()
        self.production_structure.__post_init__()
        for name in (
            "data_safety_sha256",
            "small_model_family_sha256",
            "production_model_family_sha256",
            "profile_pair_sha256",
        ):
            _require_sha256(getattr(self, name), name)

        small = self.small_arm_config
        production = self.production_arm_config
        if (
            small.arm is not production.arm
            or small.config_id != production.config_id
            or small.semantic_payload() != production.semantic_payload()
        ):
            raise ValueError(
                "small and production configs must retain one exact semantic arm profile"
            )
        if (
            small.vocabulary.vocabulary_id != "h6-prefix-small-v1"
            or small.vocabulary.size != 3
            or small.horizon != 4
        ):
            raise ValueError("small Prefix profile must be exactly V=3,T=4")
        if (
            production.vocabulary.vocabulary_id != "wikitext-2-byte-v1"
            or production.vocabulary.size != 258
            or production.horizon != 32
            or (
                production.arm is ArmId.A0
                and production.capacity_allocation.emission_width != 52
            )
        ):
            raise ValueError(
                "production Prefix profile must be exactly V=258,T=32 "
                "and H6 A0 must use hidden width 52"
            )
        for label, structure, horizon in (
            ("small", self.small_structure, 4),
            ("production", self.production_structure, 32),
        ):
            if (
                structure.dag.node_labels != tuple(range(horizon + 1))
                or structure.receiver_labels != tuple(range(1, horizon + 1))
            ):
                raise ValueError(
                    f"{label} Prefix structure does not match its frozen horizon"
                )
        if (
            self.estimator.kind != "weighted_smc"
            or self.estimator.resampling != "systematic_ess_half"
            or self.estimator.particle_count not in (4, 128, 256, 512, 1024)
        ):
            raise ValueError(
                "Prefix profiles require weighted SMC at 4, 128, 256, 512, or 1024 particles"
            )
        if self.small_model_family_sha256 != arm_model_family_sha256(small):
            raise ValueError(
                "small_model_family_sha256 does not match its exact arm factory"
            )
        if self.production_model_family_sha256 != (
            arm_model_family_sha256(production)
        ):
            raise ValueError(
                "production_model_family_sha256 does not match its exact arm factory"
            )
        expected = _owned_hash(
            "vfe4.h6.prefix-profile-pair.v1",
            self.canonical_payload(),
        )
        if self.profile_pair_sha256 != expected:
            raise ValueError(
                "profile_pair_sha256 does not match the Prefix profile pair"
            )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "small_arm_config_sha256": self.small_arm_config.config_sha256,
            "production_arm_config_sha256": (
                self.production_arm_config.config_sha256
            ),
            "estimator_sha256": self.estimator.estimator_sha256,
            "small_structure_sha256": self.small_structure.structure_sha256,
            "production_structure_sha256": (
                self.production_structure.structure_sha256
            ),
            "data_safety_sha256": self.data_safety_sha256,
            "small_model_family_sha256": self.small_model_family_sha256,
            "production_model_family_sha256": (
                self.production_model_family_sha256
            ),
        }

    @classmethod
    def create(
        cls,
        *,
        profile_id: str,
        small_arm_config: ArmConfig,
        production_arm_config: ArmConfig,
        estimator: EstimatorSpec,
        small_structure: H6LanguageStructure,
        production_structure: H6LanguageStructure,
        data_safety_sha256: str,
        small_model_family_sha256: str,
        production_model_family_sha256: str,
    ) -> "H6PrefixProfilePair":
        values = {
            "profile_id": profile_id,
            "small_arm_config": small_arm_config,
            "production_arm_config": production_arm_config,
            "estimator": estimator,
            "small_structure": small_structure,
            "production_structure": production_structure,
            "data_safety_sha256": data_safety_sha256,
            "small_model_family_sha256": small_model_family_sha256,
            "production_model_family_sha256": (
                production_model_family_sha256
            ),
        }
        provisional = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(provisional, name, value)
        return cls(
            **values,
            profile_pair_sha256=_owned_hash(
                "vfe4.h6.prefix-profile-pair.v1",
                provisional.canonical_payload(),
            ),
        )


@dataclass(frozen=True)
class PrefixReportBinding:
    """Owned digest joining the dynamic and static reports used by a certificate."""

    small_report_sha256: str
    small_case_manifest_sha256: str
    validation_report_sha256: str
    validation_case_manifest_sha256: str
    static_report_sha256: str
    static_source_manifest_sha256: str
    static_rules_sha256: str
    static_case_key_manifest_sha256: str
    binding_sha256: str

    def __post_init__(self) -> None:
        for name in tuple(self.__dataclass_fields__):
            _require_sha256(getattr(self, name), name)
        expected = _owned_hash(
            "vfe4.h6.prefix-report-binding.v1",
            self.canonical_payload(include_binding=False),
        )
        if self.binding_sha256 != expected:
            raise ValueError(
                "binding_sha256 does not match the Prefix report binding"
            )

    def canonical_payload(
        self, *, include_binding: bool = True
    ) -> dict[str, object]:
        payload = {
            name: getattr(self, name)
            for name in tuple(self.__dataclass_fields__)
            if name != "binding_sha256"
        }
        if include_binding:
            payload["binding_sha256"] = self.binding_sha256
        return payload

    @classmethod
    def create(
        cls,
        *,
        small_report_sha256: str,
        small_case_manifest_sha256: str,
        validation_report_sha256: str,
        validation_case_manifest_sha256: str,
        static_report_sha256: str,
        static_source_manifest_sha256: str,
        static_rules_sha256: str,
        static_case_key_manifest_sha256: str,
    ) -> "PrefixReportBinding":
        values = {
            "small_report_sha256": small_report_sha256,
            "small_case_manifest_sha256": small_case_manifest_sha256,
            "validation_report_sha256": validation_report_sha256,
            "validation_case_manifest_sha256": (
                validation_case_manifest_sha256
            ),
            "static_report_sha256": static_report_sha256,
            "static_source_manifest_sha256": (
                static_source_manifest_sha256
            ),
            "static_rules_sha256": static_rules_sha256,
            "static_case_key_manifest_sha256": (
                static_case_key_manifest_sha256
            ),
        }
        return cls(
            **values,
            binding_sha256=_owned_hash(
                "vfe4.h6.prefix-report-binding.v1", values
            ),
        )


@dataclass(frozen=True)
class PrefixCaseKey:
    arm: ArmId
    predictor_config_sha256: str
    estimator_sha256: str
    model_family_sha256: str
    vocabulary_sha256: str
    data_safety_sha256: str
    git_head: str
    dirty_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.arm, ArmId):
            raise ValueError("arm must be an ArmId")
        for name in (
            "predictor_config_sha256", "estimator_sha256", "model_family_sha256",
            "vocabulary_sha256", "data_safety_sha256", "dirty_digest",
        ):
            _require_sha256(getattr(self, name), name)
        _require_git_head(self.git_head)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "arm": self.arm.value,
            "predictor_config_sha256": self.predictor_config_sha256,
            "estimator_sha256": self.estimator_sha256,
            "model_family_sha256": self.model_family_sha256,
            "vocabulary_sha256": self.vocabulary_sha256,
            "data_safety_sha256": self.data_safety_sha256,
            "git_head": self.git_head,
            "dirty_digest": self.dirty_digest,
        }


@dataclass(frozen=True)
class PrefixCertificate:
    key: PrefixCaseKey
    validation_payload_canonical_json: bytes
    validation_payload_sha256: str
    status: EvidenceStatus
    obligations: tuple[str, ...]
    certificate_sha256: str

    def __post_init__(self) -> None:
        if type(self.key) is not PrefixCaseKey:
            raise ValueError("key must be a PrefixCaseKey")
        if type(self.validation_payload_canonical_json) is not bytes:
            raise ValueError("validation payload must be immutable bytes")
        observed_payload = hashlib.sha256(self.validation_payload_canonical_json).hexdigest()
        if self.validation_payload_sha256 != observed_payload:
            raise ValueError("validation_payload_sha256 does not match payload bytes")
        if not isinstance(self.status, EvidenceStatus):
            raise ValueError("status must be an EvidenceStatus")
        if type(self.obligations) is not tuple or any(type(item) is not str or not item for item in self.obligations):
            raise ValueError("obligations must be nonempty strings")
        expected = _owned_hash(
            "vfe4.h6.prefix-certificate.v1",
            {
                "key": self.key.canonical_payload(),
                "validation_payload_sha256": self.validation_payload_sha256,
                "status": self.status.value,
                "obligations": self.obligations,
            },
        )
        if self.certificate_sha256 != expected:
            raise ValueError("certificate_sha256 does not match certificate fields")
        try:
            payload = json.loads(self.validation_payload_canonical_json)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("validation payload must be canonical JSON") from exc
        if canonical_json_bytes(payload) != self.validation_payload_canonical_json:
            raise ValueError("validation payload is not canonical JSON")
        if payload.get("key") != _canonical(self.key.canonical_payload()):
            raise ValueError("validation payload key does not match certificate key")
        if payload.get("status") != self.status.value:
            raise ValueError("validation payload status does not match certificate status")
        if payload.get("obligations") != _canonical(self.obligations):
            raise ValueError("validation payload obligations do not match certificate obligations")
        checks = payload.get("checks")
        if not isinstance(checks, dict) or tuple(sorted(checks)) != tuple(sorted(H6_PREFIX_REQUIRED_CHECKS)):
            raise ValueError("validation payload has incomplete required checks")
        if any(type(value) is not bool for value in checks.values()):
            raise ValueError("validation checks must be booleans")
        if self.status is EvidenceStatus.PASS:
            if self.obligations or not all(checks.values()):
                raise ValueError("PASS requires every check and no obligation")
        elif self.status is EvidenceStatus.FAIL:
            if self.obligations or all(checks.values()):
                raise ValueError("FAIL requires a witnessed failed check and no obligation")
        elif not self.obligations or not all(checks.values()):
            raise ValueError("INCONCLUSIVE requires passing completed checks and an obligation")

    @classmethod
    def create(
        cls,
        *,
        key: PrefixCaseKey,
        status: EvidenceStatus,
        checks: Mapping[str, bool],
        obligations: tuple[str, ...],
    ) -> "PrefixCertificate":
        copied_checks = dict(checks)
        payload = canonical_json_bytes(
            {
                "key": key.canonical_payload(),
                "status": status.value,
                "checks": copied_checks,
                "obligations": tuple(obligations),
            }
        )
        payload_sha = hashlib.sha256(payload).hexdigest()
        certificate_sha = _owned_hash(
            "vfe4.h6.prefix-certificate.v1",
            {
                "key": key.canonical_payload(),
                "validation_payload_sha256": payload_sha,
                "status": status.value,
                "obligations": tuple(obligations),
            },
        )
        return cls(key, payload, payload_sha, status, tuple(obligations), certificate_sha)


@dataclass(frozen=True, slots=True, init=False)
class BoundedPrefixReportReference:
    """Exact v2 reference to one scoped report in a bounded Prefix matrix."""

    schema_version: Literal["h6-bounded-prefix-report-reference-v2"]
    profile_pair_sha256: str
    particle_count: int
    case_family: Literal["small", "validation"]
    scope: Literal["representative_exhaustive", "estimator_stratified"]
    report_key: PrefixCaseKey
    report_key_sha256: str
    report_sha256: str
    execution_plan_sha256: str
    workload_plan_sha256: str
    selected_global_indices: tuple[int, ...]
    selection_manifest_sha256: str
    completed_by_position: tuple[int, ...]
    complete_case_manifest_sha256: str | None
    model_state_sha256: str
    proposal_identity_sha256: str
    estimator_semantic_sha256: str
    estimator_artifact_bytes_sha256: str
    reference_sha256: str

    def __post_init__(self) -> None:
        if type(self) is not BoundedPrefixReportReference:
            raise TypeError(
                "record requires the exact BoundedPrefixReportReference type"
            )
        if self.schema_version != "h6-bounded-prefix-report-reference-v2":
            raise ValueError("bounded Prefix report-reference schema is closed")
        if (
            type(self.particle_count) is not int
            or self.particle_count not in H6_INFERENCE_PARTICLE_COUNTS
        ):
            raise ValueError("bounded report particle_count is not on the H6 ladder")
        if self.case_family not in ("small", "validation"):
            raise ValueError("bounded report case_family is unsupported")
        expected_scope = (
            "representative_exhaustive"
            if self.particle_count == 128
            else "estimator_stratified"
        )
        if self.scope != expected_scope:
            raise ValueError("bounded report scope does not match its particle count")
        if type(self.report_key) is not PrefixCaseKey:
            raise ValueError("bounded report key must be an exact PrefixCaseKey")
        self.report_key.__post_init__()
        for name in (
            "profile_pair_sha256",
            "report_key_sha256",
            "report_sha256",
            "execution_plan_sha256",
            "workload_plan_sha256",
            "selection_manifest_sha256",
            "model_state_sha256",
            "proposal_identity_sha256",
            "estimator_semantic_sha256",
            "estimator_artifact_bytes_sha256",
            "reference_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        expected_key_sha256 = _owned_hash(
            "vfe4.h6.prefix-case-key.v2",
            self.report_key.canonical_payload(),
        )
        if self.report_key_sha256 != expected_key_sha256:
            raise ValueError("report_key_sha256 does not match the exact report key")
        if (
            type(self.selected_global_indices) is not tuple
            or not self.selected_global_indices
            or any(type(index) is not int or index < 0 for index in self.selected_global_indices)
            or len(set(self.selected_global_indices))
            != len(self.selected_global_indices)
        ):
            raise ValueError("selected_global_indices must be exact and duplicate-free")
        if (
            type(self.completed_by_position) is not tuple
            or not self.completed_by_position
            or any(type(count) is not int or count < 0 for count in self.completed_by_position)
        ):
            raise ValueError("completed_by_position must contain exact counts")
        if self.scope == "representative_exhaustive":
            _require_sha256(
                self.complete_case_manifest_sha256,
                "complete_case_manifest_sha256",
            )
        elif self.complete_case_manifest_sha256 is not None:
            raise ValueError(
                "stratified report references cannot bind a complete-case manifest"
            )
        expected = _owned_hash(
            "vfe4.h6.bounded-prefix-report-reference.v2",
            self.canonical_payload(include_reference=False),
        )
        if self.reference_sha256 != expected:
            raise ValueError(
                "reference_sha256 does not match the bounded report reference"
            )

    def canonical_payload(
        self,
        *,
        include_reference: bool = True,
    ) -> dict[str, object]:
        payload = {
            "schema_version": self.schema_version,
            "profile_pair_sha256": self.profile_pair_sha256,
            "particle_count": self.particle_count,
            "case_family": self.case_family,
            "scope": self.scope,
            "report_key": self.report_key.canonical_payload(),
            "report_key_sha256": self.report_key_sha256,
            "report_sha256": self.report_sha256,
            "execution_plan_sha256": self.execution_plan_sha256,
            "workload_plan_sha256": self.workload_plan_sha256,
            "selected_global_indices": self.selected_global_indices,
            "selection_manifest_sha256": self.selection_manifest_sha256,
            "completed_by_position": self.completed_by_position,
            "complete_case_manifest_sha256": self.complete_case_manifest_sha256,
            "model_state_sha256": self.model_state_sha256,
            "proposal_identity_sha256": self.proposal_identity_sha256,
            "estimator_semantic_sha256": self.estimator_semantic_sha256,
            "estimator_artifact_bytes_sha256": (
                self.estimator_artifact_bytes_sha256
            ),
        }
        if include_reference:
            payload["reference_sha256"] = self.reference_sha256
        return payload

    @classmethod
    def create(
        cls,
        *,
        profile_pair_sha256: str,
        particle_count: int,
        case_family: Literal["small", "validation"],
        scope: Literal[
            "representative_exhaustive",
            "estimator_stratified",
        ],
        report_key: PrefixCaseKey,
        report_sha256: str,
        execution_plan_sha256: str,
        workload_plan_sha256: str,
        selected_global_indices: tuple[int, ...],
        selection_manifest_sha256: str,
        completed_by_position: tuple[int, ...],
        complete_case_manifest_sha256: str | None,
        model_state_sha256: str,
        proposal_identity_sha256: str,
        estimator_semantic_sha256: str,
        estimator_artifact_bytes_sha256: str,
    ) -> "BoundedPrefixReportReference":
        if cls is not BoundedPrefixReportReference:
            raise TypeError(
                "factory requires the exact BoundedPrefixReportReference type"
            )
        if type(report_key) is not PrefixCaseKey:
            raise ValueError("report_key must be an exact PrefixCaseKey")
        report_key_sha256 = _owned_hash(
            "vfe4.h6.prefix-case-key.v2",
            report_key.canonical_payload(),
        )
        values = {
            "schema_version": "h6-bounded-prefix-report-reference-v2",
            "profile_pair_sha256": profile_pair_sha256,
            "particle_count": particle_count,
            "case_family": case_family,
            "scope": scope,
            "report_key": report_key,
            "report_key_sha256": report_key_sha256,
            "report_sha256": report_sha256,
            "execution_plan_sha256": execution_plan_sha256,
            "workload_plan_sha256": workload_plan_sha256,
            "selected_global_indices": selected_global_indices,
            "selection_manifest_sha256": selection_manifest_sha256,
            "completed_by_position": completed_by_position,
            "complete_case_manifest_sha256": complete_case_manifest_sha256,
            "model_state_sha256": model_state_sha256,
            "proposal_identity_sha256": proposal_identity_sha256,
            "estimator_semantic_sha256": estimator_semantic_sha256,
            "estimator_artifact_bytes_sha256": (
                estimator_artifact_bytes_sha256
            ),
        }
        provisional = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(provisional, name, value)
        return _new_frozen(
            cls,
            **values,
            reference_sha256=_owned_hash(
                "vfe4.h6.bounded-prefix-report-reference.v2",
                provisional.canonical_payload(include_reference=False),
            ),
        )  # type: ignore[return-value]


@dataclass(frozen=True, slots=True, init=False)
class BoundedPrefixReportBinding:
    """Owned identity for one complete eight-report bounded Prefix family."""

    schema_version: Literal["h6-bounded-prefix-report-binding-v2"]
    workload_plan_sha256: str
    semantic_family_sha256: str
    git_head: str
    dirty_digest: str
    source_sha256: str
    profile_pair_sha256s: tuple[str, ...]
    report_references: tuple[BoundedPrefixReportReference, ...]
    higher_n_small_selection_manifest_sha256: str
    higher_n_validation_selection_manifest_sha256: str
    static_report_sha256: str
    static_source_manifest_sha256: str
    static_rules_sha256: str
    static_case_key_manifest_sha256: str
    binding_sha256: str

    def __post_init__(self) -> None:
        if type(self) is not BoundedPrefixReportBinding:
            raise TypeError(
                "record requires the exact BoundedPrefixReportBinding type"
            )
        if self.schema_version != "h6-bounded-prefix-report-binding-v2":
            raise ValueError("bounded Prefix report-binding schema is closed")
        workload = H6PrefixWorkloadPlan()
        if self.workload_plan_sha256 != workload.workload_plan_sha256:
            raise ValueError("bounded report binding has a stale workload plan")
        _require_git_head(self.git_head)
        for name in (
            "workload_plan_sha256",
            "semantic_family_sha256",
            "dirty_digest",
            "source_sha256",
            "higher_n_small_selection_manifest_sha256",
            "higher_n_validation_selection_manifest_sha256",
            "static_report_sha256",
            "static_source_manifest_sha256",
            "static_rules_sha256",
            "static_case_key_manifest_sha256",
            "binding_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if (
            type(self.profile_pair_sha256s) is not tuple
            or len(self.profile_pair_sha256s) != 4
        ):
            raise ValueError("bounded binding requires four profile-pair identities")
        for digest in self.profile_pair_sha256s:
            _require_sha256(digest, "profile_pair_sha256")
        if (
            type(self.report_references) is not tuple
            or len(self.report_references) != 8
            or any(
                type(reference) is not BoundedPrefixReportReference
                for reference in self.report_references
            )
        ):
            raise ValueError("bounded binding requires exactly eight report references")
        expected_order = tuple(
            (
                particle_count,
                case_family,
                (
                    "representative_exhaustive"
                    if particle_count == 128
                    else "estimator_stratified"
                ),
            )
            for particle_count in H6_INFERENCE_PARTICLE_COUNTS
            for case_family in ("small", "validation")
        )
        observed_order = tuple(
            (
                reference.particle_count,
                reference.case_family,
                reference.scope,
            )
            for reference in self.report_references
        )
        if observed_order != expected_order:
            raise ValueError("bounded report references are missing or out of order")
        for index, reference in enumerate(self.report_references):
            reference.__post_init__()
            if (
                reference.profile_pair_sha256
                != self.profile_pair_sha256s[index // 2]
                or reference.workload_plan_sha256
                != self.workload_plan_sha256
                or reference.report_key.git_head != self.git_head
                or reference.report_key.dirty_digest != self.dirty_digest
            ):
                raise ValueError(
                    "bounded report reference does not match its family binding"
                )
        small_refs = self.report_references[2::2]
        validation_refs = self.report_references[3::2]
        if (
            any(
                reference.selection_manifest_sha256
                != self.higher_n_small_selection_manifest_sha256
                for reference in small_refs
            )
            or any(
                reference.selection_manifest_sha256
                != self.higher_n_validation_selection_manifest_sha256
                for reference in validation_refs
            )
        ):
            raise ValueError(
                "higher-N report references do not share their family selection"
            )
        for references, label in (
            (self.report_references[0::2], "small"),
            (self.report_references[1::2], "production"),
        ):
            if (
                len({item.model_state_sha256 for item in references}) != 1
                or len({item.proposal_identity_sha256 for item in references})
                != 1
            ):
                raise ValueError(
                    f"{label} model/proposal identity changes across the N ladder"
                )
        expected = _owned_hash(
            "vfe4.h6.bounded-prefix-report-binding.v2",
            self.canonical_payload(include_binding=False),
        )
        if self.binding_sha256 != expected:
            raise ValueError(
                "binding_sha256 does not match the bounded Prefix binding"
            )

    def canonical_payload(
        self,
        *,
        include_binding: bool = True,
    ) -> dict[str, object]:
        payload = {
            "schema_version": self.schema_version,
            "workload_plan_sha256": self.workload_plan_sha256,
            "semantic_family_sha256": self.semantic_family_sha256,
            "git_head": self.git_head,
            "dirty_digest": self.dirty_digest,
            "source_sha256": self.source_sha256,
            "profile_pair_sha256s": self.profile_pair_sha256s,
            "report_references": tuple(
                reference.canonical_payload()
                for reference in self.report_references
            ),
            "higher_n_small_selection_manifest_sha256": (
                self.higher_n_small_selection_manifest_sha256
            ),
            "higher_n_validation_selection_manifest_sha256": (
                self.higher_n_validation_selection_manifest_sha256
            ),
            "static_report_sha256": self.static_report_sha256,
            "static_source_manifest_sha256": (
                self.static_source_manifest_sha256
            ),
            "static_rules_sha256": self.static_rules_sha256,
            "static_case_key_manifest_sha256": (
                self.static_case_key_manifest_sha256
            ),
        }
        if include_binding:
            payload["binding_sha256"] = self.binding_sha256
        return payload

    @classmethod
    def create(
        cls,
        *,
        workload_plan_sha256: str,
        semantic_family_sha256: str,
        git_head: str,
        dirty_digest: str,
        source_sha256: str,
        profile_pair_sha256s: tuple[str, ...],
        report_references: tuple[BoundedPrefixReportReference, ...],
        higher_n_small_selection_manifest_sha256: str,
        higher_n_validation_selection_manifest_sha256: str,
        static_report_sha256: str,
        static_source_manifest_sha256: str,
        static_rules_sha256: str,
        static_case_key_manifest_sha256: str,
    ) -> "BoundedPrefixReportBinding":
        if cls is not BoundedPrefixReportBinding:
            raise TypeError(
                "factory requires the exact BoundedPrefixReportBinding type"
            )
        values = {
            "schema_version": "h6-bounded-prefix-report-binding-v2",
            "workload_plan_sha256": workload_plan_sha256,
            "semantic_family_sha256": semantic_family_sha256,
            "git_head": git_head,
            "dirty_digest": dirty_digest,
            "source_sha256": source_sha256,
            "profile_pair_sha256s": profile_pair_sha256s,
            "report_references": report_references,
            "higher_n_small_selection_manifest_sha256": (
                higher_n_small_selection_manifest_sha256
            ),
            "higher_n_validation_selection_manifest_sha256": (
                higher_n_validation_selection_manifest_sha256
            ),
            "static_report_sha256": static_report_sha256,
            "static_source_manifest_sha256": static_source_manifest_sha256,
            "static_rules_sha256": static_rules_sha256,
            "static_case_key_manifest_sha256": static_case_key_manifest_sha256,
        }
        provisional = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(provisional, name, value)
        return _new_frozen(
            cls,
            **values,
            binding_sha256=_owned_hash(
                "vfe4.h6.bounded-prefix-report-binding.v2",
                provisional.canonical_payload(include_binding=False),
            ),
        )  # type: ignore[return-value]


@dataclass(frozen=True, slots=True, init=False)
class BoundedPrefixCertificate:
    """Self-validating v2 certificate for one bounded Prefix family."""

    schema_version: Literal["h6-prefix-certificate-v2"]
    semantic_family_sha256: str
    report_binding: BoundedPrefixReportBinding
    validation_payload_canonical_json: bytes
    validation_payload_sha256: str
    status: EvidenceStatus
    obligations: tuple[str, ...]
    checks: Mapping[str, bool]
    certificate_sha256: str

    def __post_init__(self) -> None:
        if type(self) is not BoundedPrefixCertificate:
            raise TypeError(
                "record requires the exact BoundedPrefixCertificate type"
            )
        if self.schema_version != "h6-prefix-certificate-v2":
            raise ValueError("bounded Prefix certificate schema is closed")
        _require_sha256(self.semantic_family_sha256, "semantic_family_sha256")
        if type(self.report_binding) is not BoundedPrefixReportBinding:
            raise ValueError(
                "report_binding must be an exact BoundedPrefixReportBinding"
            )
        self.report_binding.__post_init__()
        if (
            self.semantic_family_sha256
            != self.report_binding.semantic_family_sha256
        ):
            raise ValueError(
                "certificate semantic family does not match its typed binding"
            )
        if type(self.validation_payload_canonical_json) is not bytes:
            raise ValueError("validation payload must be immutable bytes")
        _require_sha256(
            self.validation_payload_sha256,
            "validation_payload_sha256",
        )
        if self.validation_payload_sha256 != hashlib.sha256(
            self.validation_payload_canonical_json
        ).hexdigest():
            raise ValueError(
                "validation_payload_sha256 does not match retained bytes"
            )
        if type(self.status) is not EvidenceStatus:
            raise ValueError("bounded certificate status must be exact")
        if (
            type(self.obligations) is not tuple
            or any(type(item) is not str or not item for item in self.obligations)
            or self.obligations != tuple(sorted(set(self.obligations)))
        ):
            raise ValueError("bounded certificate obligations are not canonical")
        if (
            type(self.checks) is not MappingProxyType
            or tuple(self.checks) != H6_PREFIX_REQUIRED_CHECKS
            or any(type(value) is not bool for value in self.checks.values())
        ):
            raise ValueError("bounded certificate required-check map is incomplete")
        if self.status is EvidenceStatus.PASS:
            if self.obligations or not all(self.checks.values()):
                raise ValueError(
                    "PASS bounded certificates require every check and no obligation"
                )
        elif self.status is EvidenceStatus.FAIL:
            if self.obligations or all(self.checks.values()):
                raise ValueError(
                    "FAIL bounded certificates require a witnessed failed check"
                )
        elif not self.obligations or not all(self.checks.values()):
            raise ValueError(
                "INCONCLUSIVE bounded certificates require obligations without failure"
            )
        payload = _canonical_json_object(
            self.validation_payload_canonical_json,
            "bounded Prefix validation payload",
        )
        expected_payload = {
            "schema_version": self.schema_version,
            "semantic_family_sha256": self.semantic_family_sha256,
            "report_binding": self.report_binding.canonical_payload(),
            "checks": dict(self.checks),
            "status": self.status.value,
            "obligations": self.obligations,
        }
        if payload != json.loads(canonical_json_bytes(expected_payload)):
            raise ValueError(
                "retained validation payload does not match the typed certificate"
            )
        expected_certificate_sha256 = _owned_hash(
            "vfe4.h6.prefix-certificate.v2",
            {
                "schema_version": self.schema_version,
                "semantic_family_sha256": self.semantic_family_sha256,
                "report_binding_sha256": self.report_binding.binding_sha256,
                "validation_payload_sha256": self.validation_payload_sha256,
                "status": self.status.value,
                "obligations": self.obligations,
            },
        )
        if self.certificate_sha256 != expected_certificate_sha256:
            raise ValueError(
                "certificate_sha256 does not match the bounded certificate"
            )

    @classmethod
    def create(
        cls,
        *,
        semantic_family_sha256: str,
        report_binding: BoundedPrefixReportBinding,
        status: EvidenceStatus,
        obligations: tuple[str, ...],
        checks: Mapping[str, bool],
    ) -> "BoundedPrefixCertificate":
        if cls is not BoundedPrefixCertificate:
            raise TypeError(
                "factory requires the exact BoundedPrefixCertificate type"
            )
        if not isinstance(checks, Mapping):
            raise ValueError("checks must be a mapping")
        frozen_checks = MappingProxyType(dict(checks))
        frozen_obligations = tuple(sorted(set(obligations)))
        payload = canonical_json_bytes(
            {
                "schema_version": "h6-prefix-certificate-v2",
                "semantic_family_sha256": semantic_family_sha256,
                "report_binding": report_binding.canonical_payload(),
                "checks": dict(frozen_checks),
                "status": status.value,
                "obligations": frozen_obligations,
            }
        )
        payload_sha256 = hashlib.sha256(payload).hexdigest()
        return _new_frozen(
            cls,
            schema_version="h6-prefix-certificate-v2",
            semantic_family_sha256=semantic_family_sha256,
            report_binding=report_binding,
            validation_payload_canonical_json=payload,
            validation_payload_sha256=payload_sha256,
            status=status,
            obligations=frozen_obligations,
            checks=frozen_checks,
            certificate_sha256=_owned_hash(
                "vfe4.h6.prefix-certificate.v2",
                {
                    "schema_version": "h6-prefix-certificate-v2",
                    "semantic_family_sha256": semantic_family_sha256,
                    "report_binding_sha256": report_binding.binding_sha256,
                    "validation_payload_sha256": payload_sha256,
                    "status": status.value,
                    "obligations": frozen_obligations,
                },
            ),
        )  # type: ignore[return-value]


def require_prefix_pass(
    key: PrefixCaseKey,
    certificates: Mapping[PrefixCaseKey, PrefixCertificate],
) -> PrefixCertificate:
    certificate = certificates.get(key)
    if certificate is None or certificate.key != key or certificate.status is not EvidenceStatus.PASS:
        raise ValueError("exact PASS prefix certificate is required")
    certificate.__post_init__()
    return certificate


@dataclass(frozen=True, init=False)
class PredictionCorrectnessArtifactRef:
    gate: Literal["H1", "H2", "H3", "H5"]
    artifact_path: Path
    manifest_sha256: str
    git_head: str
    dirty_digest: str
    config_sha256: str
    validation_payload_sha256: str
    status: GateStatus
    _manifest_bytes: bytes = field(repr=False, compare=False)
    _config_bytes: bytes = field(repr=False, compare=False)
    _validation_payload_bytes: bytes = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.gate not in ("H1", "H2", "H3", "H5"):
            raise ValueError("gate must be exactly one of H1, H2, H3, H5")
        if not isinstance(self.artifact_path, Path):
            raise ValueError("artifact_path must be a Path")
        expected_path = f"validation/{self.gate.lower()}.json"
        if self.artifact_path.as_posix() != expected_path:
            raise ValueError(f"artifact_path must be exactly {expected_path}")
        for name in ("manifest_sha256", "dirty_digest", "config_sha256", "validation_payload_sha256"):
            _require_sha256(getattr(self, name), name)
        _require_git_head(self.git_head)
        if not isinstance(self.status, GateStatus):
            raise ValueError("status must be a GateStatus")
        expected = {
            "manifest_sha256": hashlib.sha256(self._manifest_bytes).hexdigest(),
            "config_sha256": hashlib.sha256(self._config_bytes).hexdigest(),
            "validation_payload_sha256": hashlib.sha256(self._validation_payload_bytes).hexdigest(),
        }
        for name, digest in expected.items():
            if getattr(self, name) != digest:
                raise ValueError(f"{name} does not match retained producer bytes")
        _require_manifest_members(
            self._manifest_bytes,
            {"config.json": self._config_bytes, expected_path: self._validation_payload_bytes},
        )
        derived_status = _validated_producer_status(
            payload_bytes=self._validation_payload_bytes,
            expected_gate=self.gate,
            git_head=self.git_head,
            dirty_digest=self.dirty_digest,
            config_bytes=self._config_bytes,
        )
        if self.status is not derived_status:
            raise ValueError("status does not match the manifest-linked validation payload")

    @classmethod
    def from_bytes(
        cls,
        *,
        gate: Literal["H1", "H2", "H3", "H5"],
        artifact_path: Path,
        manifest_bytes: bytes,
        git_head: str,
        dirty_digest: str,
        config_bytes: bytes,
        validation_payload_bytes: bytes,
    ) -> "PredictionCorrectnessArtifactRef":
        if gate not in ("H1", "H2", "H3", "H5"):
            raise ValueError("gate must be exactly one of H1, H2, H3, H5")
        for name, value in (
            ("manifest_bytes", manifest_bytes),
            ("config_bytes", config_bytes),
            ("validation_payload_bytes", validation_payload_bytes),
        ):
            if type(value) is not bytes:
                raise ValueError(f"{name} must be bytes")
        status = _validated_producer_status(
            payload_bytes=validation_payload_bytes,
            expected_gate=gate,
            git_head=git_head,
            dirty_digest=dirty_digest,
            config_bytes=config_bytes,
        )
        return _new_frozen(
            cls,
            gate=gate,
            artifact_path=artifact_path,
            manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            git_head=git_head,
            dirty_digest=dirty_digest,
            config_sha256=hashlib.sha256(config_bytes).hexdigest(),
            validation_payload_sha256=hashlib.sha256(validation_payload_bytes).hexdigest(),
            status=status,
            _manifest_bytes=bytes(manifest_bytes),
            _config_bytes=bytes(config_bytes),
            _validation_payload_bytes=bytes(validation_payload_bytes),
        )  # type: ignore[return-value]

    def verify_bytes(
        self, *, manifest_bytes: bytes, validation_payload_bytes: bytes,
        config_bytes: bytes,
    ) -> None:
        if hashlib.sha256(manifest_bytes).hexdigest() != self.manifest_sha256:
            raise ValueError("manifest_sha256 does not match manifest bytes")
        if hashlib.sha256(validation_payload_bytes).hexdigest() != self.validation_payload_sha256:
            raise ValueError("validation_payload_sha256 does not match payload bytes")
        if hashlib.sha256(config_bytes).hexdigest() != self.config_sha256:
            raise ValueError("config_sha256 does not match config bytes")


@dataclass(frozen=True, init=False)
class H1PrefixPriorArtifactRef:
    artifact_path: Path
    manifest_sha256: str
    git_head: str
    dirty_digest: str
    generative_factor_schema_sha256: str
    config_sha256: str
    validation_payload_sha256: str
    status: GateStatus
    _manifest_bytes: bytes = field(repr=False, compare=False)
    _generative_factor_schema_bytes: bytes = field(repr=False, compare=False)
    _config_bytes: bytes = field(repr=False, compare=False)
    _validation_payload_bytes: bytes = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_path, Path):
            raise ValueError("artifact_path must be a Path")
        expected_path = "validation/h1_prefix_prior.json"
        if self.artifact_path.as_posix() != expected_path:
            raise ValueError(f"artifact_path must be exactly {expected_path}")
        for name in (
            "manifest_sha256", "dirty_digest", "generative_factor_schema_sha256",
            "config_sha256", "validation_payload_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        _require_git_head(self.git_head)
        if not isinstance(self.status, GateStatus):
            raise ValueError("status must be a GateStatus")
        checks = {
            "manifest_sha256": self._manifest_bytes,
            "generative_factor_schema_sha256": self._generative_factor_schema_bytes,
            "config_sha256": self._config_bytes,
            "validation_payload_sha256": self._validation_payload_bytes,
        }
        for name, value in checks.items():
            if getattr(self, name) != hashlib.sha256(value).hexdigest():
                raise ValueError(f"{name} does not match retained producer bytes")
        _require_manifest_members(
            self._manifest_bytes,
            {
                "config.json": self._config_bytes,
                "schemas/generative_factor.json": self._generative_factor_schema_bytes,
                expected_path: self._validation_payload_bytes,
            },
        )
        derived_status = _validated_producer_status(
            payload_bytes=self._validation_payload_bytes,
            expected_gate="H1-Prefix-Prior",
            git_head=self.git_head,
            dirty_digest=self.dirty_digest,
            config_bytes=self._config_bytes,
            extra_bindings={
                "generative_factor_schema_sha256": self.generative_factor_schema_sha256
            },
        )
        if self.status is not derived_status:
            raise ValueError("status does not match the manifest-linked validation payload")

    @classmethod
    def from_bytes(
        cls,
        *,
        artifact_path: Path,
        manifest_bytes: bytes,
        git_head: str,
        dirty_digest: str,
        generative_factor_schema_bytes: bytes,
        config_bytes: bytes,
        validation_payload_bytes: bytes,
    ) -> "H1PrefixPriorArtifactRef":
        values = {
            "manifest": manifest_bytes,
            "generative_factor_schema": generative_factor_schema_bytes,
            "config": config_bytes,
            "validation_payload": validation_payload_bytes,
        }
        if any(type(value) is not bytes for value in values.values()):
            raise ValueError("all H1-prefix producer preimages must be bytes")
        status = _validated_producer_status(
            payload_bytes=validation_payload_bytes,
            expected_gate="H1-Prefix-Prior",
            git_head=git_head,
            dirty_digest=dirty_digest,
            config_bytes=config_bytes,
            extra_bindings={
                "generative_factor_schema_sha256": hashlib.sha256(
                    generative_factor_schema_bytes
                ).hexdigest()
            },
        )
        return _new_frozen(
            cls,
            artifact_path=artifact_path,
            manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            git_head=git_head,
            dirty_digest=dirty_digest,
            generative_factor_schema_sha256=hashlib.sha256(generative_factor_schema_bytes).hexdigest(),
            config_sha256=hashlib.sha256(config_bytes).hexdigest(),
            validation_payload_sha256=hashlib.sha256(validation_payload_bytes).hexdigest(),
            status=status,
            _manifest_bytes=bytes(manifest_bytes),
            _generative_factor_schema_bytes=bytes(generative_factor_schema_bytes),
            _config_bytes=bytes(config_bytes),
            _validation_payload_bytes=bytes(validation_payload_bytes),
        )  # type: ignore[return-value]

    def verify_bytes(
        self,
        *,
        manifest_bytes: bytes,
        generative_factor_schema_bytes: bytes,
        config_bytes: bytes,
        validation_payload_bytes: bytes,
    ) -> None:
        checks = {
            "manifest_sha256": manifest_bytes,
            "generative_factor_schema_sha256": generative_factor_schema_bytes,
            "config_sha256": config_bytes,
            "validation_payload_sha256": validation_payload_bytes,
        }
        for name, value in checks.items():
            if hashlib.sha256(value).hexdigest() != getattr(self, name):
                raise ValueError(f"{name} does not match producer bytes")


@dataclass(frozen=True, init=False)
class SmcAccuracyArtifactRef:
    artifact_path: Path
    manifest_sha256: str
    git_head: str
    dirty_digest: str
    estimator_sha256: str
    fixture_set_sha256: str
    validation_payload_sha256: str
    status: GateStatus
    _manifest_bytes: bytes = field(repr=False, compare=False)
    _estimator_preimage_bytes: bytes = field(repr=False, compare=False)
    _fixture_set_bytes: bytes = field(repr=False, compare=False)
    _config_bytes: bytes = field(repr=False, compare=False)
    _validation_payload_bytes: bytes = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_path, Path):
            raise ValueError("artifact_path must be a Path")
        expected_path = "validation/h6_smc_accuracy.json"
        if self.artifact_path.as_posix() != expected_path:
            raise ValueError(f"artifact_path must be exactly {expected_path}")
        for name in (
            "manifest_sha256", "dirty_digest", "estimator_sha256",
            "fixture_set_sha256", "validation_payload_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        _require_git_head(self.git_head)
        if not isinstance(self.status, GateStatus):
            raise ValueError("status must be a GateStatus")
        checks = {
            "manifest_sha256": self._manifest_bytes,
            "estimator_sha256": self._estimator_preimage_bytes,
            "fixture_set_sha256": self._fixture_set_bytes,
            "validation_payload_sha256": self._validation_payload_bytes,
        }
        for name, value in checks.items():
            if getattr(self, name) != hashlib.sha256(value).hexdigest():
                raise ValueError(f"{name} does not match retained producer bytes")
        _require_manifest_members(
            self._manifest_bytes,
            {
                "config.json": self._config_bytes,
                "protocol/estimator.json": self._estimator_preimage_bytes,
                "fixtures/finite_smc.json": self._fixture_set_bytes,
                expected_path: self._validation_payload_bytes,
            },
        )
        derived_status = _validated_producer_status(
            payload_bytes=self._validation_payload_bytes,
            expected_gate="H6-SMC-Accuracy",
            git_head=self.git_head,
            dirty_digest=self.dirty_digest,
            config_bytes=self._config_bytes,
            extra_bindings={
                "estimator_sha256": self.estimator_sha256,
                "fixture_set_sha256": self.fixture_set_sha256,
            },
        )
        if self.status is not derived_status:
            raise ValueError("status does not match the manifest-linked validation payload")

    @classmethod
    def from_bytes(
        cls,
        *,
        artifact_path: Path,
        manifest_bytes: bytes,
        git_head: str,
        dirty_digest: str,
        estimator_preimage_bytes: bytes,
        fixture_set_bytes: bytes,
        config_bytes: bytes,
        validation_payload_bytes: bytes,
    ) -> "SmcAccuracyArtifactRef":
        values = (
            manifest_bytes, estimator_preimage_bytes, fixture_set_bytes, config_bytes,
            validation_payload_bytes,
        )
        if any(type(value) is not bytes for value in values):
            raise ValueError("all SMC producer preimages must be bytes")
        status = _validated_producer_status(
            payload_bytes=validation_payload_bytes,
            expected_gate="H6-SMC-Accuracy",
            git_head=git_head,
            dirty_digest=dirty_digest,
            config_bytes=config_bytes,
            extra_bindings={
                "estimator_sha256": hashlib.sha256(estimator_preimage_bytes).hexdigest(),
                "fixture_set_sha256": hashlib.sha256(fixture_set_bytes).hexdigest(),
            },
        )
        return _new_frozen(
            cls,
            artifact_path=artifact_path,
            manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            git_head=git_head,
            dirty_digest=dirty_digest,
            estimator_sha256=hashlib.sha256(estimator_preimage_bytes).hexdigest(),
            fixture_set_sha256=hashlib.sha256(fixture_set_bytes).hexdigest(),
            validation_payload_sha256=hashlib.sha256(validation_payload_bytes).hexdigest(),
            status=status,
            _manifest_bytes=bytes(manifest_bytes),
            _estimator_preimage_bytes=bytes(estimator_preimage_bytes),
            _fixture_set_bytes=bytes(fixture_set_bytes),
            _config_bytes=bytes(config_bytes),
            _validation_payload_bytes=bytes(validation_payload_bytes),
        )  # type: ignore[return-value]

    def verify_bytes(
        self,
        *,
        manifest_bytes: bytes,
        estimator_preimage_bytes: bytes,
        fixture_set_bytes: bytes,
        config_bytes: bytes,
        validation_payload_bytes: bytes,
    ) -> None:
        checks = {
            "manifest_sha256": manifest_bytes,
            "estimator_sha256": estimator_preimage_bytes,
            "fixture_set_sha256": fixture_set_bytes,
            "validation_payload_sha256": validation_payload_bytes,
        }
        for name, value in checks.items():
            if hashlib.sha256(value).hexdigest() != getattr(self, name):
                raise ValueError(f"{name} does not match producer bytes")
        if config_bytes != self._config_bytes:
            raise ValueError("config bytes do not match retained producer bytes")


@dataclass(frozen=True)
class H5UpdateBinding:
    h5_manifest_sha256: str
    h5_payload_sha256: str
    update_spec_raw_sha256: str
    update_spec_canonical_sha256: str
    objective_schema_sha256: str
    factor_input_schema_sha256: str
    reference_sha256: str
    recognition_state_sha256: str
    model_state_sha256: str
    validation_payload_sha256: str
    enabled_update_labels: tuple[str, ...]
    binding_sha256: str

    _DIGEST_FIELDS: ClassVar[tuple[str, ...]] = (
        "h5_manifest_sha256", "h5_payload_sha256", "update_spec_raw_sha256",
        "update_spec_canonical_sha256", "objective_schema_sha256",
        "factor_input_schema_sha256", "reference_sha256", "recognition_state_sha256",
        "model_state_sha256", "validation_payload_sha256",
    )

    def __post_init__(self) -> None:
        for name in self._DIGEST_FIELDS:
            _require_sha256(getattr(self, name), name)
        if self.enabled_update_labels != _H5_LABELS:
            raise ValueError("enabled_update_labels must equal the three actual H5 labels")
        expected = _owned_hash(
            "vfe4.h6.h5-update-binding.v1",
            {
                **{name: getattr(self, name) for name in self._DIGEST_FIELDS},
                "enabled_update_labels": self.enabled_update_labels,
            },
        )
        if self.binding_sha256 != expected:
            raise ValueError("binding_sha256 does not match H5 producer identities")

    @classmethod
    def from_producer_preimages(
        cls,
        *,
        producer_preimages: Mapping[str, bytes],
        enabled_update_labels: tuple[str, ...],
    ) -> "H5UpdateBinding":
        if set(producer_preimages) != set(cls._DIGEST_FIELDS):
            raise ValueError("producer preimages must match the exact H5 digest inventory")
        digests = {
            name: hashlib.sha256(producer_preimages[name]).hexdigest()
            for name in cls._DIGEST_FIELDS
        }
        binding_sha = _owned_hash(
            "vfe4.h6.h5-update-binding.v1",
            {**digests, "enabled_update_labels": tuple(enabled_update_labels)},
        )
        return cls(
            *(digests[name] for name in cls._DIGEST_FIELDS),
            tuple(enabled_update_labels),
            binding_sha,
        )

    def verify_producer_preimages(self, producer_preimages: Mapping[str, bytes]) -> None:
        if set(producer_preimages) != set(self._DIGEST_FIELDS):
            raise ValueError("producer preimages must match the exact H5 digest inventory")
        for name in self._DIGEST_FIELDS:
            if hashlib.sha256(producer_preimages[name]).hexdigest() != getattr(self, name):
                raise ValueError(f"{name} does not match its producer preimage")


@dataclass(frozen=True)
class H6OuterSchedule:
    schedule_schema: Literal["h6-outer-schedule-v1"]
    optimizer_class: Literal["AdamW"]
    optimizer_policy_sha256: str
    model_updates_per_batch: Literal[1]
    validation_twentieths_per_pass: Literal[20]
    full_passes: Literal[2]
    outer_schedule_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schedule_schema != "h6-outer-schedule-v1"
            or self.optimizer_class != "AdamW"
            or self.model_updates_per_batch != 1
            or self.validation_twentieths_per_pass != 20
            or self.full_passes != 2
        ):
            raise ValueError("outer schedule must equal the frozen H6 contract")
        _require_sha256(self.optimizer_policy_sha256, "optimizer_policy_sha256")
        expected = _owned_hash(
            "vfe4.h6.outer-schedule.v1",
            {
                "schedule_schema": self.schedule_schema,
                "optimizer_class": self.optimizer_class,
                "optimizer_policy_sha256": self.optimizer_policy_sha256,
                "model_updates_per_batch": 1,
                "validation_twentieths_per_pass": 20,
                "full_passes": 2,
            },
        )
        if self.outer_schedule_sha256 != expected:
            raise ValueError("outer_schedule_sha256 does not match schedule")

    @classmethod
    def create(cls, *, optimizer_policy_sha256: str) -> "H6OuterSchedule":
        payload = {
            "schedule_schema": "h6-outer-schedule-v1",
            "optimizer_class": "AdamW",
            "optimizer_policy_sha256": optimizer_policy_sha256,
            "model_updates_per_batch": 1,
            "validation_twentieths_per_pass": 20,
            "full_passes": 2,
        }
        return cls(
            "h6-outer-schedule-v1", "AdamW", optimizer_policy_sha256, 1, 20, 2,
            _owned_hash("vfe4.h6.outer-schedule.v1", payload),
        )


@dataclass(frozen=True)
class H6ArmPhaseSchedule:
    endpoint_config_sha256: str
    latent_enabled: bool
    phases: tuple[TrainingPhase, ...]
    recognition_updates_per_batch: Literal[0, 1]
    model_updates_per_batch: Literal[1]
    no_op_phases: Literal[0]
    phase_schedule_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.endpoint_config_sha256, "endpoint_config_sha256")
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
        expected_recognition = 1 if self.latent_enabled else 0
        if (
            self.phases != expected_phases
            or self.recognition_updates_per_batch != expected_recognition
            or self.model_updates_per_batch != 1
            or self.no_op_phases != 0
        ):
            raise ValueError("phase schedule does not match latent-enabled contract")
        expected = _owned_hash(
            "vfe4.h6.arm-phase-schedule.v1",
            {
                "endpoint_config_sha256": self.endpoint_config_sha256,
                "latent_enabled": self.latent_enabled,
                "phases": tuple(phase.value for phase in self.phases),
                "recognition_updates_per_batch": self.recognition_updates_per_batch,
                "model_updates_per_batch": self.model_updates_per_batch,
                "no_op_phases": self.no_op_phases,
            },
        )
        if self.phase_schedule_sha256 != expected:
            raise ValueError("phase_schedule_sha256 does not match phase schedule")

    @classmethod
    def create(
        cls,
        *,
        endpoint_config_sha256: str,
        latent_enabled: bool,
        phases: tuple[TrainingPhase, ...],
    ) -> "H6ArmPhaseSchedule":
        recognition_updates = 1 if latent_enabled else 0
        payload = {
            "endpoint_config_sha256": endpoint_config_sha256,
            "latent_enabled": latent_enabled,
            "phases": tuple(phase.value for phase in phases),
            "recognition_updates_per_batch": recognition_updates,
            "model_updates_per_batch": 1,
            "no_op_phases": 0,
        }
        return cls(
            endpoint_config_sha256,
            latent_enabled,
            tuple(phases),
            recognition_updates,
            1,
            0,
            _owned_hash("vfe4.h6.arm-phase-schedule.v1", payload),
        )


@dataclass(frozen=True)
class H6TrainingSchedule:
    schedule_schema: Literal["h6-training-schedule-v2"]
    outer: H6OuterSchedule
    endpoint_phases: tuple[H6ArmPhaseSchedule, ...]
    schedule_sha256: str

    def __post_init__(self) -> None:
        if self.schedule_schema != "h6-training-schedule-v2" or type(self.outer) is not H6OuterSchedule:
            raise ValueError("training schedule has the wrong schema or outer record")
        if type(self.endpoint_phases) is not tuple or not self.endpoint_phases:
            raise ValueError("endpoint_phases must be nonempty")
        endpoints = tuple(item.endpoint_config_sha256 for item in self.endpoint_phases)
        if len(set(endpoints)) != len(endpoints):
            raise ValueError("endpoint phase schedules must be unique")
        expected = _owned_hash(
            "vfe4.h6.training-schedule.v2",
            {
                "schedule_schema": self.schedule_schema,
                "outer_schedule_sha256": self.outer.outer_schedule_sha256,
                "phase_schedule_sha256": tuple(item.phase_schedule_sha256 for item in self.endpoint_phases),
            },
        )
        if self.schedule_sha256 != expected:
            raise ValueError("schedule_sha256 does not match training schedule")

    @classmethod
    def create(
        cls, *, outer: H6OuterSchedule, endpoint_phases: tuple[H6ArmPhaseSchedule, ...]
    ) -> "H6TrainingSchedule":
        payload = {
            "schedule_schema": "h6-training-schedule-v2",
            "outer_schedule_sha256": outer.outer_schedule_sha256,
            "phase_schedule_sha256": tuple(item.phase_schedule_sha256 for item in endpoint_phases),
        }
        return cls(
            "h6-training-schedule-v2",
            outer,
            tuple(endpoint_phases),
            _owned_hash("vfe4.h6.training-schedule.v2", payload),
        )


@dataclass(frozen=True, init=False)
class H6PredictionReadinessToken:
    readiness_schema: Literal[
        "h6-prediction-readiness-v1",
        "h6-prediction-readiness-v2",
    ]
    git_head: str
    dirty_digest: str
    experiment_config_sha256: str
    correctness_manifests: tuple[tuple[str, str], ...]
    h1_prefix_prior_manifest_sha256: str
    h1_prefix_prior_generative_factor_schema_sha256: str | None
    smc_bias_semantics_sha256: str | None
    objective_gate_spec_sha256: str | None
    h5_update_binding_sha256: str
    h6_training_schedule_sha256: str
    smc_validation_manifest_sha256: str
    critical_values_sha256: str
    endpoint_smc_protocol_sha256: str
    attribution_matrix_sha256: str
    matching_set_sha256: str
    prefix_certificate_set_sha256: str
    data_identity_sha256: str
    access_policy_sha256: str
    readiness_sha256: str
    status: Literal["PASS"]
    _correctness_artifacts: tuple[PredictionCorrectnessArtifactRef, ...] = field(
        repr=False, compare=False
    )
    _h1_prefix_prior_artifact: H1PrefixPriorArtifactRef = field(repr=False, compare=False)
    _smc_accuracy_artifact: SmcAccuracyArtifactRef = field(repr=False, compare=False)
    _prefix_certificates: tuple[PrefixCertificate, ...] = field(repr=False, compare=False)
    _training_schedule: H6TrainingSchedule = field(repr=False, compare=False)
    _h5_update_binding: H5UpdateBinding = field(repr=False, compare=False)
    _endpoint_smc_protocol: "EndpointSmcProtocol" = field(repr=False, compare=False)
    _data_identity: DataIdentity = field(repr=False, compare=False)
    _matching_set: "H6MatchingSetRecord | None" = field(
        repr=False,
        compare=False,
    )
    _objective_gate_spec: ObjectiveGateSpec | None = field(
        repr=False,
        compare=False,
    )

    _SHA_FIELDS: ClassVar[tuple[str, ...]] = (
        "dirty_digest", "experiment_config_sha256", "h5_update_binding_sha256",
        "h6_training_schedule_sha256", "smc_validation_manifest_sha256",
        "critical_values_sha256", "endpoint_smc_protocol_sha256",
        "attribution_matrix_sha256", "matching_set_sha256",
        "prefix_certificate_set_sha256", "data_identity_sha256", "access_policy_sha256",
    )

    def __post_init__(self) -> None:
        if (
            self.readiness_schema
            not in (
                "h6-prediction-readiness-v1",
                "h6-prediction-readiness-v2",
            )
            or self.status != "PASS"
        ):
            raise ValueError("readiness token must be a supported PASS schema")
        _require_git_head(self.git_head)
        for name in self._SHA_FIELDS:
            _require_sha256(getattr(self, name), name)
        _require_sha256(self.h1_prefix_prior_manifest_sha256, "h1_prefix_prior_manifest_sha256")
        if self.readiness_schema == "h6-prediction-readiness-v1":
            if (
                self.h1_prefix_prior_generative_factor_schema_sha256
                is not None
                or self.smc_bias_semantics_sha256 is not None
                or self.objective_gate_spec_sha256 is not None
                or self._objective_gate_spec is not None
                or self._matching_set is not None
            ):
                raise ValueError("legacy readiness cannot carry amended bindings")
        else:
            from vfe4.evaluation.smc_uncertainty import SMC_BIAS_SEMANTICS

            if (
                self.h1_prefix_prior_generative_factor_schema_sha256
                != H1_PREFIX_PRIOR_V2_GENERATIVE_FACTOR_SCHEMA_SHA256
            ):
                raise ValueError(
                    "amended readiness requires the production scorer-v2 schema"
                )
            if (
                self.smc_bias_semantics_sha256
                != SMC_BIAS_SEMANTICS.semantics_sha256
            ):
                raise ValueError(
                    "amended readiness requires the frozen SMC bias semantics"
                )
            _require_sha256(
                self.objective_gate_spec_sha256,
                "objective_gate_spec_sha256",
            )
            if type(self._objective_gate_spec) is not ObjectiveGateSpec:
                raise ValueError("amended readiness requires the typed OBJECTIVE spec")
            self._objective_gate_spec.__post_init__()
            if (
                self._objective_gate_spec.spec_sha256
                != self.objective_gate_spec_sha256
            ):
                raise ValueError("amended readiness OBJECTIVE binding is stale")
            from vfe4.artifacts.h6_matching import H6MatchingSetRecord

            if type(self._matching_set) is not H6MatchingSetRecord:
                raise ValueError(
                    "amended readiness requires the typed eligible matching set"
                )
            self._matching_set.__post_init__()
            if (
                self._matching_set.status != "ELIGIBLE"
                or self._matching_set.obligations
                or self._matching_set.matching_set_sha256
                != self.matching_set_sha256
                or self._matching_set.git_head != self.git_head
                or self._matching_set.dirty_digest != self.dirty_digest
            ):
                raise ValueError(
                    "amended readiness matching evidence is stale or ineligible"
                )
            if type(self._training_schedule) is not H6TrainingSchedule:
                raise ValueError("typed H6 training schedule is required")
            self._training_schedule.__post_init__()
            scheduled = tuple(
                (
                    item.endpoint_config_sha256,
                    item.latent_enabled,
                )
                for item in self._training_schedule.endpoint_phases
            )
            selected = tuple(
                (
                    item.config.config_sha256,
                    item.config.latent_enabled,
                )
                for item in self._matching_set.ownership_inventories
            )
            if scheduled != selected:
                raise ValueError(
                    "training schedule endpoints differ from selected matching "
                    "ownership inventories"
                )
        gates = tuple(gate for gate, _ in self.correctness_manifests)
        if gates != ("H1", "H2", "H3", "H5"):
            raise ValueError("correctness manifests must be exactly H1, H2, H3, H5")
        for gate, digest in self.correctness_manifests:
            _require_sha256(digest, f"correctness_manifests[{gate}]")
        if tuple(item.gate for item in self._correctness_artifacts) != gates:
            raise ValueError("readiness correctness artifacts do not match gate inventory")
        for artifact in self._correctness_artifacts:
            if type(artifact) is not PredictionCorrectnessArtifactRef:
                raise ValueError("readiness requires typed correctness artifacts")
            artifact.__post_init__()
            if (
                artifact.status is not GateStatus.PASS
                or artifact.git_head != self.git_head
                or artifact.dirty_digest != self.dirty_digest
            ):
                raise ValueError("correctness artifacts must be current exact PASS records")
        derived_manifests = tuple(
            (item.gate, item.manifest_sha256) for item in self._correctness_artifacts
        )
        if self.correctness_manifests != derived_manifests:
            raise ValueError("correctness manifests do not match validated artifacts")
        if type(self._h1_prefix_prior_artifact) is not H1PrefixPriorArtifactRef:
            raise ValueError("typed H1-prefix-prior artifact is required")
        self._h1_prefix_prior_artifact.__post_init__()
        if (
            self._h1_prefix_prior_artifact.status is not GateStatus.PASS
            or self._h1_prefix_prior_artifact.git_head != self.git_head
            or self._h1_prefix_prior_artifact.dirty_digest != self.dirty_digest
            or self._h1_prefix_prior_artifact.manifest_sha256
            != self.h1_prefix_prior_manifest_sha256
        ):
            raise ValueError("H1-prefix-prior artifact must be a current exact PASS record")
        if (
            self.readiness_schema == "h6-prediction-readiness-v2"
            and self._h1_prefix_prior_artifact.generative_factor_schema_sha256
            != self.h1_prefix_prior_generative_factor_schema_sha256
        ):
            raise ValueError("amended readiness H1 scorer-v2 binding is stale")
        if type(self._smc_accuracy_artifact) is not SmcAccuracyArtifactRef:
            raise ValueError("typed SMC accuracy artifact is required")
        self._smc_accuracy_artifact.__post_init__()
        if (
            self._smc_accuracy_artifact.status is not GateStatus.PASS
            or self._smc_accuracy_artifact.git_head != self.git_head
            or self._smc_accuracy_artifact.dirty_digest != self.dirty_digest
            or self._smc_accuracy_artifact.manifest_sha256
            != self.smc_validation_manifest_sha256
        ):
            raise ValueError("SMC artifact must be a current exact PASS record")
        if type(self._training_schedule) is not H6TrainingSchedule:
            raise ValueError("typed H6 training schedule is required")
        self._training_schedule.__post_init__()
        if self._training_schedule.schedule_sha256 != self.h6_training_schedule_sha256:
            raise ValueError("training schedule identity does not match readiness")
        if type(self._h5_update_binding) is not H5UpdateBinding:
            raise ValueError("typed H5 update binding is required")
        self._h5_update_binding.__post_init__()
        if self._h5_update_binding.binding_sha256 != self.h5_update_binding_sha256:
            raise ValueError("H5 update binding identity does not match readiness")
        if type(self._endpoint_smc_protocol) is not EndpointSmcProtocol:
            raise ValueError("typed endpoint SMC protocol is required")
        self._endpoint_smc_protocol.__post_init__()
        if self._endpoint_smc_protocol.protocol_sha256 != self.endpoint_smc_protocol_sha256:
            raise ValueError("endpoint SMC protocol identity does not match readiness")
        if type(self._data_identity) is not DataIdentity:
            raise ValueError("typed data identity is required")
        self._data_identity.__post_init__()
        if (
            self._data_identity.data_identity_sha256 != self.data_identity_sha256
            or self._data_identity.access_policy_sha256 != self.access_policy_sha256
        ):
            raise ValueError("data/access identities do not match readiness")
        if not self._prefix_certificates:
            raise ValueError("readiness requires a nonempty exact Prefix certificate set")
        for certificate in self._prefix_certificates:
            certificate.__post_init__()
            if (
                certificate.status is not EvidenceStatus.PASS
                or certificate.key.git_head != self.git_head
                or certificate.key.dirty_digest != self.dirty_digest
            ):
                raise ValueError("Prefix certificates must be current exact PASS records")
        if self.prefix_certificate_set_sha256 != _prefix_certificate_set_sha256(
            self._prefix_certificates
        ):
            raise ValueError("prefix certificate set identity does not match certificates")
        domain = (
            "vfe4.h6.prediction-readiness.v1"
            if self.readiness_schema == "h6-prediction-readiness-v1"
            else "vfe4.h6.prediction-readiness.v2"
        )
        expected = _owned_hash(domain, self._payload())
        if self.readiness_sha256 != expected:
            raise ValueError("readiness_sha256 does not match readiness fields")

    def _payload(self) -> dict[str, object]:
        return {
            "readiness_schema": self.readiness_schema,
            "git_head": self.git_head,
            **{name: getattr(self, name) for name in self._SHA_FIELDS},
            "correctness_manifests": self.correctness_manifests,
            "h1_prefix_prior_manifest_sha256": self.h1_prefix_prior_manifest_sha256,
            **(
                {}
                if self.readiness_schema == "h6-prediction-readiness-v1"
                else {
                    "h1_prefix_prior_generative_factor_schema_sha256": (
                        self.h1_prefix_prior_generative_factor_schema_sha256
                    ),
                    "smc_bias_semantics_sha256": (
                        self.smc_bias_semantics_sha256
                    ),
                    "objective_gate_spec_sha256": (
                        self.objective_gate_spec_sha256
                    ),
                }
            ),
            "status": self.status,
        }



def _prefix_certificate_set_sha256(
    certificates: tuple[PrefixCertificate, ...],
) -> str:
    ordered = tuple(
        sorted(
            (
                canonical_json_bytes(certificate.key.canonical_payload()).decode("ascii"),
                certificate.certificate_sha256,
            )
            for certificate in certificates
        )
    )
    return _owned_hash("vfe4.h6.prefix-certificate-set.v1", ordered)


def issue_prediction_readiness(
    *,
    git_head: str,
    dirty_digest: str,
    experiment_config_sha256: str,
    correctness_artifacts: tuple[PredictionCorrectnessArtifactRef, ...],
    h1_prefix_prior_artifact: H1PrefixPriorArtifactRef,
    h5_update_binding: H5UpdateBinding,
    h6_training_schedule: H6TrainingSchedule,
    smc_accuracy_artifact: SmcAccuracyArtifactRef,
    critical_values_sha256: str,
    endpoint_smc_protocol: "EndpointSmcProtocol",
    attribution_matrix_sha256: str,
    matching_set_sha256: str,
    prefix_certificates: Mapping[PrefixCaseKey, PrefixCertificate],
    data_identity: DataIdentity,
) -> H6PredictionReadinessToken:
    correctness = tuple(correctness_artifacts)
    if tuple(item.gate for item in correctness) != ("H1", "H2", "H3", "H5"):
        raise ValueError("correctness artifacts must be exactly H1, H2, H3, H5")
    if type(h1_prefix_prior_artifact) is not H1PrefixPriorArtifactRef:
        raise ValueError("typed H1-prefix-prior artifact is required")
    if type(smc_accuracy_artifact) is not SmcAccuracyArtifactRef:
        raise ValueError("typed SMC accuracy artifact is required")
    if type(h5_update_binding) is not H5UpdateBinding:
        raise ValueError("typed H5 update binding is required")
    if type(h6_training_schedule) is not H6TrainingSchedule:
        raise ValueError("typed H6 training schedule is required")
    if type(endpoint_smc_protocol) is not EndpointSmcProtocol:
        raise ValueError("typed endpoint SMC protocol is required")
    if type(data_identity) is not DataIdentity:
        raise ValueError("typed data identity is required")
    for key, certificate in prefix_certificates.items():
        if type(key) is not PrefixCaseKey or type(certificate) is not PrefixCertificate:
            raise ValueError("typed Prefix certificate mapping is required")
        if certificate.key != key:
            raise ValueError("Prefix certificate mapping key does not match certificate")
    certificates = tuple(
        certificate
        for _, certificate in sorted(
            prefix_certificates.items(),
            key=lambda item: canonical_json_bytes(item[0].canonical_payload()),
        )
    )
    values: dict[str, object] = {
        "readiness_schema": "h6-prediction-readiness-v1",
        "git_head": git_head,
        "dirty_digest": dirty_digest,
        "experiment_config_sha256": experiment_config_sha256,
        "correctness_manifests": tuple(
            (artifact.gate, artifact.manifest_sha256) for artifact in correctness
        ),
        "h1_prefix_prior_manifest_sha256": h1_prefix_prior_artifact.manifest_sha256,
        "h1_prefix_prior_generative_factor_schema_sha256": None,
        "smc_bias_semantics_sha256": None,
        "objective_gate_spec_sha256": None,
        "h5_update_binding_sha256": h5_update_binding.binding_sha256,
        "h6_training_schedule_sha256": h6_training_schedule.schedule_sha256,
        "smc_validation_manifest_sha256": smc_accuracy_artifact.manifest_sha256,
        "critical_values_sha256": critical_values_sha256,
        "endpoint_smc_protocol_sha256": endpoint_smc_protocol.protocol_sha256,
        "attribution_matrix_sha256": attribution_matrix_sha256,
        "matching_set_sha256": matching_set_sha256,
        "prefix_certificate_set_sha256": _prefix_certificate_set_sha256(certificates),
        "data_identity_sha256": data_identity.data_identity_sha256,
        "access_policy_sha256": data_identity.access_policy_sha256,
        "status": "PASS",
        "_correctness_artifacts": correctness,
        "_h1_prefix_prior_artifact": h1_prefix_prior_artifact,
        "_smc_accuracy_artifact": smc_accuracy_artifact,
        "_prefix_certificates": certificates,
        "_training_schedule": h6_training_schedule,
        "_h5_update_binding": h5_update_binding,
        "_endpoint_smc_protocol": endpoint_smc_protocol,
        "_data_identity": data_identity,
        "_matching_set": None,
        "_objective_gate_spec": None,
    }
    payload = {
        "readiness_schema": values["readiness_schema"],
        "git_head": values["git_head"],
        **{name: values[name] for name in H6PredictionReadinessToken._SHA_FIELDS},
        "correctness_manifests": values["correctness_manifests"],
        "h1_prefix_prior_manifest_sha256": values["h1_prefix_prior_manifest_sha256"],
        "status": "PASS",
    }
    values["readiness_sha256"] = _owned_hash(
        "vfe4.h6.prediction-readiness.v1", payload
    )
    return _new_frozen(H6PredictionReadinessToken, **values)  # type: ignore[return-value]


def issue_prediction_readiness_v2(
    *,
    git_head: str,
    dirty_digest: str,
    experiment_config_sha256: str,
    correctness_artifacts: tuple[PredictionCorrectnessArtifactRef, ...],
    h1_prefix_prior_artifact: H1PrefixPriorArtifactRef,
    h1_prefix_prior_generative_factor_schema_sha256: str,
    smc_bias_semantics_sha256: str,
    objective_gate_spec: ObjectiveGateSpec,
    h5_update_binding: H5UpdateBinding,
    h6_training_schedule: H6TrainingSchedule,
    smc_accuracy_artifact: SmcAccuracyArtifactRef,
    critical_values_sha256: str,
    endpoint_smc_protocol: "EndpointSmcProtocol",
    attribution_matrix_sha256: str,
    matching_set_sha256: str,
    prefix_certificates: Mapping[PrefixCaseKey, PrefixCertificate],
    data_identity: DataIdentity,
    matching_set: "H6MatchingSetRecord",
) -> H6PredictionReadinessToken:
    """Issue amended readiness with scorer-v2 and OBJECTIVE bindings."""

    if type(objective_gate_spec) is not ObjectiveGateSpec:
        raise ValueError("typed OBJECTIVE gate specification is required")
    objective_gate_spec.__post_init__()
    if type(h1_prefix_prior_artifact) is not H1PrefixPriorArtifactRef:
        raise ValueError("typed H1 prefix-prior artifact is required")
    from vfe4.artifacts.h6_matching import H6MatchingSetRecord

    if type(matching_set) is not H6MatchingSetRecord:
        raise ValueError("typed H6 matching set is required")
    matching_set.__post_init__()
    if (
        matching_set.status != "ELIGIBLE"
        or matching_set.obligations
        or matching_set.matching_set_sha256 != matching_set_sha256
    ):
        raise ValueError("H6 matching set is stale or ineligible")
    if (
        h1_prefix_prior_generative_factor_schema_sha256
        != H1_PREFIX_PRIOR_V2_GENERATIVE_FACTOR_SCHEMA_SHA256
    ):
        raise ValueError(
            "H1 prefix-prior scorer schema must equal the production v2 schema"
        )
    from vfe4.evaluation.smc_uncertainty import SMC_BIAS_SEMANTICS

    if (
        smc_bias_semantics_sha256
        != SMC_BIAS_SEMANTICS.semantics_sha256
    ):
        raise ValueError(
            "SMC bias semantics must equal the frozen production semantics"
        )
    if (
        h1_prefix_prior_artifact.generative_factor_schema_sha256
        != h1_prefix_prior_generative_factor_schema_sha256
    ):
        raise ValueError("H1 prefix-prior artifact is not the configured scorer-v2")
    legacy = issue_prediction_readiness(
        git_head=git_head,
        dirty_digest=dirty_digest,
        experiment_config_sha256=experiment_config_sha256,
        correctness_artifacts=correctness_artifacts,
        h1_prefix_prior_artifact=h1_prefix_prior_artifact,
        h5_update_binding=h5_update_binding,
        h6_training_schedule=h6_training_schedule,
        smc_accuracy_artifact=smc_accuracy_artifact,
        critical_values_sha256=critical_values_sha256,
        endpoint_smc_protocol=endpoint_smc_protocol,
        attribution_matrix_sha256=attribution_matrix_sha256,
        matching_set_sha256=matching_set_sha256,
        prefix_certificates=prefix_certificates,
        data_identity=data_identity,
    )
    values: dict[str, object] = {
        "readiness_schema": "h6-prediction-readiness-v2",
        "git_head": legacy.git_head,
        "dirty_digest": legacy.dirty_digest,
        "experiment_config_sha256": legacy.experiment_config_sha256,
        "correctness_manifests": legacy.correctness_manifests,
        "h1_prefix_prior_manifest_sha256": (
            legacy.h1_prefix_prior_manifest_sha256
        ),
        "h1_prefix_prior_generative_factor_schema_sha256": (
            h1_prefix_prior_generative_factor_schema_sha256
        ),
        "smc_bias_semantics_sha256": smc_bias_semantics_sha256,
        "objective_gate_spec_sha256": objective_gate_spec.spec_sha256,
        **{
            name: getattr(legacy, name)
            for name in H6PredictionReadinessToken._SHA_FIELDS
            if name
            not in {
                "dirty_digest",
                "experiment_config_sha256",
            }
        },
        "status": "PASS",
        "_correctness_artifacts": legacy._correctness_artifacts,
        "_h1_prefix_prior_artifact": legacy._h1_prefix_prior_artifact,
        "_smc_accuracy_artifact": legacy._smc_accuracy_artifact,
        "_prefix_certificates": legacy._prefix_certificates,
        "_training_schedule": legacy._training_schedule,
        "_h5_update_binding": legacy._h5_update_binding,
        "_endpoint_smc_protocol": legacy._endpoint_smc_protocol,
        "_data_identity": legacy._data_identity,
        "_matching_set": matching_set,
        "_objective_gate_spec": objective_gate_spec,
    }
    provisional = object.__new__(H6PredictionReadinessToken)
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    values["readiness_sha256"] = _owned_hash(
        "vfe4.h6.prediction-readiness.v2",
        provisional._payload(),
    )
    return _new_frozen(
        H6PredictionReadinessToken,
        **values,
    )  # type: ignore[return-value]


@dataclass(frozen=True)
class EstimatorSpec:
    schema_version: Literal["h6-estimator-v1"]
    kind: Literal["deterministic_exact", "weighted_smc"]
    particle_count: int | None
    resampling: Literal["none", "systematic_ess_half"]
    dtype: Literal["float64"]
    device: Literal["cpu"]
    estimator_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != "h6-estimator-v1" or self.dtype != "float64" or self.device != "cpu":
            raise ValueError("unsupported H6 estimator schema/device/dtype")
        if self.kind == "deterministic_exact":
            if self.particle_count is not None or self.resampling != "none":
                raise ValueError("deterministic estimator cannot carry particles")
        elif self.kind == "weighted_smc":
            if type(self.particle_count) is not int or self.particle_count <= 0 or self.resampling != "systematic_ess_half":
                raise ValueError("weighted SMC requires particles and systematic resampling")
        else:
            raise ValueError("unsupported estimator kind")
        expected = _owned_hash(
            "vfe4.h6.estimator-spec.v1",
            {
                "schema_version": self.schema_version,
                "kind": self.kind,
                "particle_count": self.particle_count,
                "resampling": self.resampling,
                "dtype": self.dtype,
                "device": self.device,
            },
        )
        if self.estimator_sha256 != expected:
            raise ValueError("estimator_sha256 does not match estimator fields")

    @classmethod
    def create(
        cls,
        *,
        kind: Literal["deterministic_exact", "weighted_smc"],
        particle_count: int | None,
        resampling: Literal["none", "systematic_ess_half"],
        dtype: Literal["float64"] = "float64",
        device: Literal["cpu"] = "cpu",
    ) -> "EstimatorSpec":
        payload = {
            "schema_version": "h6-estimator-v1", "kind": kind,
            "particle_count": particle_count, "resampling": resampling,
            "dtype": dtype, "device": device,
        }
        return cls(
            "h6-estimator-v1", kind, particle_count, resampling, dtype, device,
            _owned_hash("vfe4.h6.estimator-spec.v1", payload),
        )


@dataclass(frozen=True)
class EndpointSmcProtocol:
    protocol_schema: Literal["h6-endpoint-smc-v1"]
    particle_counts: tuple[int, ...]
    replicate_count: Literal[64]
    registry_root_seed: Literal[2026072198]
    common_stream_domain: Literal["h6-wt2-endpoint-mc-v1"]
    simultaneous_interval_count: Literal[352]
    familywise_alpha: float
    critical_value_df63: float
    remainder_contraction: float
    protocol_sha256: str

    def __post_init__(self) -> None:
        expected_values = (
            self.protocol_schema == "h6-endpoint-smc-v1"
            and self.particle_counts == (128, 256, 512, 1024)
            and self.replicate_count == 64
            and self.registry_root_seed == 2026072198
            and self.common_stream_domain == "h6-wt2-endpoint-mc-v1"
            and self.simultaneous_interval_count == 352
            and self.familywise_alpha == 0.01
            and self.critical_value_df63 == 4.5144904535377144
            and self.remainder_contraction == 0.75
        )
        if not expected_values:
            raise ValueError("endpoint SMC protocol does not match the frozen contract")
        expected = _owned_hash(
            "vfe4.h6.endpoint-smc-protocol.v1",
            {
                "protocol_schema": self.protocol_schema,
                "particle_counts": self.particle_counts,
                "replicate_count": self.replicate_count,
                "registry_root_seed": self.registry_root_seed,
                "common_stream_domain": self.common_stream_domain,
                "simultaneous_interval_count": self.simultaneous_interval_count,
                "familywise_alpha": self.familywise_alpha,
                "critical_value_df63": self.critical_value_df63,
                "remainder_contraction": self.remainder_contraction,
            },
        )
        if self.protocol_sha256 != expected:
            raise ValueError("protocol_sha256 does not match endpoint protocol")

    @classmethod
    def create(
        cls,
        *,
        particle_counts: tuple[int, ...],
        replicate_count: int,
        registry_root_seed: int,
        common_stream_domain: str,
        simultaneous_interval_count: int,
        familywise_alpha: float,
        critical_value_df63: float,
        remainder_contraction: float,
    ) -> "EndpointSmcProtocol":
        payload = {
            "protocol_schema": "h6-endpoint-smc-v1",
            "particle_counts": tuple(particle_counts),
            "replicate_count": replicate_count,
            "registry_root_seed": registry_root_seed,
            "common_stream_domain": common_stream_domain,
            "simultaneous_interval_count": simultaneous_interval_count,
            "familywise_alpha": familywise_alpha,
            "critical_value_df63": critical_value_df63,
            "remainder_contraction": remainder_contraction,
        }
        return cls(
            "h6-endpoint-smc-v1",
            tuple(particle_counts),
            replicate_count,
            registry_root_seed,
            common_stream_domain,
            simultaneous_interval_count,
            familywise_alpha,
            critical_value_df63,
            remainder_contraction,
            _owned_hash("vfe4.h6.endpoint-smc-protocol.v1", payload),
        )


@dataclass(frozen=True)
class NllTotals:
    negative_log_likelihood_sum: float
    counted_targets: int

    def __post_init__(self) -> None:
        if type(self.negative_log_likelihood_sum) is not float or not math.isfinite(self.negative_log_likelihood_sum):
            raise ValueError("negative_log_likelihood_sum must be finite")
        if type(self.counted_targets) is not int or self.counted_targets <= 0:
            raise ValueError("counted_targets must be positive")

    @property
    def nats_per_token(self) -> float:
        return self.negative_log_likelihood_sum / self.counted_targets


H6_PREDICTION_DELTA = 0.01005033585350145
H6_OBJECTIVE_DELTA = 0.01005033585350145
H6_OBJECTIVE_COMPLETE_ARM_ID = (
    "h6-a5-structured-parent-specific-prefix-exact-complete-"
    "latent-smoothing-v2"
)
H6_OBJECTIVE_EMISSION_ARM_ID = _H6_PARENT_SPECIFIC_EMISSION_CONFIG_ID


@dataclass(frozen=True, slots=True)
class ObjectiveGateSpec:
    schema_version: Literal["h6-objective-gate-v1"]
    complete_arm_id: str
    emission_arm_id: str
    orientation: Literal["nll_complete_minus_nll_emission"]
    delta_obj: Literal[0.01005033585350145]
    opening_policy: Literal["single_all_or_none"]
    evaluation_order: Literal["OBJECTIVE_then_PRIMARY"]
    spec_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != "h6-objective-gate-v1"
            or self.complete_arm_id != H6_OBJECTIVE_COMPLETE_ARM_ID
            or self.emission_arm_id != H6_OBJECTIVE_EMISSION_ARM_ID
            or self.complete_arm_id == self.emission_arm_id
            or self.orientation != "nll_complete_minus_nll_emission"
            or self.delta_obj != H6_OBJECTIVE_DELTA
            or self.opening_policy != "single_all_or_none"
            or self.evaluation_order != "OBJECTIVE_then_PRIMARY"
        ):
            raise ValueError("objective gate specification differs from the amendment")
        _require_sha256(self.spec_sha256, "spec_sha256")
        if self.spec_sha256 != _owned_hash(
            "vfe4.h6.objective-gate.v1",
            self.canonical_payload(),
        ):
            raise ValueError("spec_sha256 does not match objective gate fields")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "complete_arm_id": self.complete_arm_id,
            "emission_arm_id": self.emission_arm_id,
            "orientation": self.orientation,
            "delta_obj": self.delta_obj.hex(),
            "opening_policy": self.opening_policy,
            "evaluation_order": self.evaluation_order,
        }

    @classmethod
    def create(cls) -> "ObjectiveGateSpec":
        values = {
            "schema_version": "h6-objective-gate-v1",
            "complete_arm_id": H6_OBJECTIVE_COMPLETE_ARM_ID,
            "emission_arm_id": H6_OBJECTIVE_EMISSION_ARM_ID,
            "orientation": "nll_complete_minus_nll_emission",
            "delta_obj": H6_OBJECTIVE_DELTA,
            "opening_policy": "single_all_or_none",
            "evaluation_order": "OBJECTIVE_then_PRIMARY",
        }
        return cls(
            **values,  # type: ignore[arg-type]
            spec_sha256=_owned_hash(
                "vfe4.h6.objective-gate.v1",
                {
                    **values,
                    "delta_obj": H6_OBJECTIVE_DELTA.hex(),
                },
            ),
        )


@dataclass(frozen=True, init=False)
class ObjectiveGateDecision:
    status: EvidenceStatus
    objective_interval: tuple[float, float]
    estimator_complete: bool
    interval_eligible: bool
    obligations: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.status, EvidenceStatus):
            raise ValueError("status must be an EvidenceStatus")
        if (
            type(self.objective_interval) is not tuple
            or len(self.objective_interval) != 2
            or any(
                type(value) is not float or not math.isfinite(value)
                for value in self.objective_interval
            )
            or self.objective_interval[0] > self.objective_interval[1]
        ):
            raise ValueError("objective_interval must be an ordered finite pair")
        if (
            type(self.estimator_complete) is not bool
            or type(self.interval_eligible) is not bool
        ):
            raise ValueError("objective estimator flags must be boolean")
        lower, upper = self.objective_interval
        if not self.estimator_complete:
            expected_status = EvidenceStatus.INCONCLUSIVE
            expected_obligations = (
                "objective actual-endpoint estimator evidence incomplete",
            )
        elif not self.interval_eligible:
            expected_status = EvidenceStatus.INCONCLUSIVE
            expected_obligations = (
                "objective estimator interval is ineligible",
            )
        elif upper <= H6_OBJECTIVE_DELTA:
            expected_status = EvidenceStatus.PASS
            expected_obligations = ()
        elif lower > H6_OBJECTIVE_DELTA:
            expected_status = EvidenceStatus.FAIL
            expected_obligations = ()
        else:
            expected_status = EvidenceStatus.INCONCLUSIVE
            expected_obligations = (
                "objective interval does not cross the frozen upper-bound decision",
            )
        if self.status is not expected_status or self.obligations != expected_obligations:
            raise ValueError("objective decision does not match the frozen interval rule")

    @classmethod
    def classify(
        cls,
        *,
        objective_interval: tuple[float, float],
        estimator_complete: bool,
        interval_eligible: bool,
    ) -> "ObjectiveGateDecision":
        if type(estimator_complete) is not bool or type(interval_eligible) is not bool:
            raise ValueError("objective estimator flags must be boolean")
        lower, upper = objective_interval
        if not estimator_complete:
            status = EvidenceStatus.INCONCLUSIVE
            obligations = (
                "objective actual-endpoint estimator evidence incomplete",
            )
        elif not interval_eligible:
            status = EvidenceStatus.INCONCLUSIVE
            obligations = ("objective estimator interval is ineligible",)
        elif upper <= H6_OBJECTIVE_DELTA:
            status = EvidenceStatus.PASS
            obligations = ()
        elif lower > H6_OBJECTIVE_DELTA:
            status = EvidenceStatus.FAIL
            obligations = ()
        else:
            status = EvidenceStatus.INCONCLUSIVE
            obligations = (
                "objective interval does not cross the frozen upper-bound decision",
            )
        return _new_frozen(
            cls,
            status=status,
            objective_interval=objective_interval,
            estimator_complete=estimator_complete,
            interval_eligible=interval_eligible,
            obligations=obligations,
        )  # type: ignore[return-value]


@dataclass(frozen=True, init=False)
class PredictionDecision:
    status: EvidenceStatus
    primary_interval: tuple[float, float] | None
    obligations: tuple[str, ...]
    estimator_complete: bool

    def __post_init__(self) -> None:
        if not isinstance(self.status, EvidenceStatus):
            raise ValueError("status must be an EvidenceStatus")
        if self.primary_interval is not None:
            if (
                type(self.primary_interval) is not tuple
                or len(self.primary_interval) != 2
                or any(type(value) is not float or not math.isfinite(value) for value in self.primary_interval)
                or self.primary_interval[0] > self.primary_interval[1]
            ):
                raise ValueError("primary_interval must be an ordered finite pair")
        if type(self.estimator_complete) is not bool:
            raise ValueError("estimator_complete must be boolean")
        if not self.estimator_complete:
            if (
                self.status is not EvidenceStatus.INCONCLUSIVE
                or self.obligations != ("actual-endpoint estimator evidence incomplete",)
            ):
                raise ValueError("incomplete estimator evidence must remain INCONCLUSIVE")
            return
        if self.primary_interval is None:
            raise ValueError("complete estimator evidence requires a primary interval")
        lower, upper = self.primary_interval
        if lower > H6_PREDICTION_DELTA:
            expected_status = EvidenceStatus.PASS
            expected_obligations: tuple[str, ...] = ()
        elif upper <= 0.0:
            expected_status = EvidenceStatus.FAIL
            expected_obligations = ()
        else:
            expected_status = EvidenceStatus.INCONCLUSIVE
            expected_obligations = (
                "primary interval does not cross a frozen decision boundary",
            )
        if self.status is not expected_status or self.obligations != expected_obligations:
            raise ValueError("decision does not match the frozen primary interval rule")

    @classmethod
    def classify(
        cls,
        *,
        primary_interval: tuple[float, float] | None,
        estimator_complete: bool,
    ) -> "PredictionDecision":
        if estimator_complete:
            if primary_interval is None:
                raise ValueError("complete estimator evidence requires a primary interval")
            lower, upper = primary_interval
            if lower > H6_PREDICTION_DELTA:
                status = EvidenceStatus.PASS
                obligations: tuple[str, ...] = ()
            elif upper <= 0.0:
                status = EvidenceStatus.FAIL
                obligations = ()
            else:
                status = EvidenceStatus.INCONCLUSIVE
                obligations = (
                    "primary interval does not cross a frozen decision boundary",
                )
        else:
            status = EvidenceStatus.INCONCLUSIVE
            obligations = ("actual-endpoint estimator evidence incomplete",)
        return _new_frozen(
            cls,
            status=status,
            primary_interval=primary_interval,
            obligations=obligations,
            estimator_complete=estimator_complete,
        )  # type: ignore[return-value]


def _ordered_primary_disposition(
    *,
    primary_interval: tuple[float, float] | None,
    estimator_complete: bool,
    interval_eligible: bool,
) -> tuple[EvidenceStatus, str, tuple[str, ...]]:
    if not estimator_complete:
        return (
            EvidenceStatus.INCONCLUSIVE,
            "INCONCLUSIVE",
            ("actual-endpoint estimator evidence incomplete",),
        )
    if not interval_eligible:
        return (
            EvidenceStatus.INCONCLUSIVE,
            "INCONCLUSIVE",
            ("primary estimator interval is ineligible",),
        )
    raw = PredictionDecision.classify(
        primary_interval=primary_interval,
        estimator_complete=True,
    )
    return raw.status, raw.status.value, raw.obligations


@dataclass(frozen=True, init=False)
class OrderedPredictionDecision:
    """One-opening H6 disposition with OBJECTIVE logically before PRIMARY."""

    status: EvidenceStatus
    objective_gate_spec_sha256: str
    objective: ObjectiveGateDecision
    primary_interval: tuple[float, float] | None
    primary_estimator_complete: bool
    primary_interval_eligible: bool
    primary_disposition: Literal[
        "PASS",
        "FAIL",
        "INCONCLUSIVE",
        "NOT_EVALUATED_AFTER_OBJECTIVE_GATE",
    ]
    opening_count: Literal[1]
    test_opening_sha256: str
    raw_endpoint_inventory_sha256: str
    obligations: tuple[str, ...]

    def __post_init__(self) -> None:
        expected_spec = ObjectiveGateSpec.create()
        _require_sha256(
            self.objective_gate_spec_sha256,
            "objective_gate_spec_sha256",
        )
        if self.objective_gate_spec_sha256 != expected_spec.spec_sha256:
            raise ValueError("ordered prediction binds the wrong OBJECTIVE spec")
        if type(self.objective) is not ObjectiveGateDecision:
            raise ValueError("ordered prediction requires an OBJECTIVE decision")
        self.objective.__post_init__()
        if (
            type(self.primary_estimator_complete) is not bool
            or type(self.primary_interval_eligible) is not bool
        ):
            raise ValueError("primary estimator flags must be boolean")
        if self.primary_interval is not None and (
            type(self.primary_interval) is not tuple
            or len(self.primary_interval) != 2
            or any(
                type(value) is not float or not math.isfinite(value)
                for value in self.primary_interval
            )
            or self.primary_interval[0] > self.primary_interval[1]
        ):
            raise ValueError("raw PRIMARY interval must be an ordered finite pair")
        if self.primary_estimator_complete and self.primary_interval is None:
            raise ValueError("complete PRIMARY evidence requires its raw interval")
        if self.opening_count != 1:
            raise ValueError("ordered prediction requires exactly one test opening")
        _require_sha256(self.test_opening_sha256, "test_opening_sha256")
        _require_sha256(
            self.raw_endpoint_inventory_sha256,
            "raw_endpoint_inventory_sha256",
        )
        if self.objective.status is EvidenceStatus.PASS:
            (
                expected_status,
                expected_disposition,
                expected_obligations,
            ) = _ordered_primary_disposition(
                primary_interval=self.primary_interval,
                estimator_complete=self.primary_estimator_complete,
                interval_eligible=self.primary_interval_eligible,
            )
        else:
            expected_status = self.objective.status
            expected_disposition = "NOT_EVALUATED_AFTER_OBJECTIVE_GATE"
            expected_obligations = self.objective.obligations
        if (
            self.status is not expected_status
            or self.primary_disposition != expected_disposition
            or self.obligations != expected_obligations
        ):
            raise ValueError("ordered prediction does not follow OBJECTIVE_then_PRIMARY")

    @classmethod
    def classify(
        cls,
        *,
        objective_gate_spec: ObjectiveGateSpec,
        objective: ObjectiveGateDecision,
        primary_interval: tuple[float, float] | None,
        primary_estimator_complete: bool,
        primary_interval_eligible: bool,
        opening_count: int,
        test_opening_sha256: str,
        raw_endpoint_inventory_sha256: str,
    ) -> "OrderedPredictionDecision":
        if type(objective_gate_spec) is not ObjectiveGateSpec:
            raise ValueError("ordered prediction requires the exact OBJECTIVE spec")
        objective_gate_spec.__post_init__()
        if type(objective) is not ObjectiveGateDecision:
            raise ValueError("ordered prediction requires an OBJECTIVE decision")
        objective.__post_init__()
        if (
            type(primary_estimator_complete) is not bool
            or type(primary_interval_eligible) is not bool
        ):
            raise ValueError("primary estimator flags must be boolean")
        if opening_count != 1:
            raise ValueError("ordered prediction requires exactly one test opening")
        if objective.status is EvidenceStatus.PASS:
            (
                status,
                primary_disposition,
                obligations,
            ) = _ordered_primary_disposition(
                primary_interval=primary_interval,
                estimator_complete=primary_estimator_complete,
                interval_eligible=primary_interval_eligible,
            )
        else:
            status = objective.status
            primary_disposition = "NOT_EVALUATED_AFTER_OBJECTIVE_GATE"
            obligations = objective.obligations
        return _new_frozen(
            cls,
            status=status,
            objective_gate_spec_sha256=objective_gate_spec.spec_sha256,
            objective=objective,
            primary_interval=primary_interval,
            primary_estimator_complete=primary_estimator_complete,
            primary_interval_eligible=primary_interval_eligible,
            primary_disposition=primary_disposition,
            opening_count=1,
            test_opening_sha256=test_opening_sha256,
            raw_endpoint_inventory_sha256=raw_endpoint_inventory_sha256,
            obligations=obligations,
        )  # type: ignore[return-value]


__all__ = [
    "ArmId", "BoundedPrefixCertificate", "BoundedPrefixReportBinding",
    "BoundedPrefixReportReference", "CausalDag", "CausalDagRow",
    "CheckpointIdentity", "DataIdentity",
    "DurableTestOpeningCapability", "EmissionOnlyAblationTerms",
    "EncodedTokenStorageIdentity", "EndpointSmcProtocol", "EstimatorSpec",
    "EvidenceStatus", "ExperimentIdentity", "FrozenBatchSchedule", "FrozenTensorSnapshot",
    "H6_INFERENCE_PARTICLE_COUNTS",
    "H1PrefixPriorArtifactRef", "H5UpdateBinding", "H6ArmPhaseSchedule",
    "H6EndpointLanguageElboTerms", "H6FactorTerm", "H6LanguageElboTerms",
    "H6SourcePriorTrace",
    "H6LanguageStructure", "H6OuterSchedule",
    "H6PrefixProfilePair", "H6PrefixWorkloadPlan",
    "H6PredictionReadinessToken", "H6TrainingSchedule",
    "H1_PREFIX_PRIOR_V2_GENERATIVE_FACTOR_SCHEMA_SHA256",
    "H6_PREFIX_REQUIRED_CHECKS",
    "H6_EMISSION_ONLY_ABLATION_HASH_DOMAIN",
    "H6_OBJECTIVE_COMPLETE_ARM_ID", "H6_OBJECTIVE_DELTA",
    "H6_OBJECTIVE_EMISSION_ARM_ID", "H6_PREDICTION_DELTA",
    "InferenceComputeRecord", "NllTotals",
    "ObjectiveGateDecision", "ObjectiveGateSpec", "OrderedPredictionDecision",
    "PredictionCorrectnessArtifactRef", "PredictionDecision", "PrefixCaseKey",
    "PrefixCertificate",
    "PrefixReportBinding", "SealedSplitHandle",
    "SmcAccuracyArtifactRef", "TrainingPhase", "ValidatedTestOpening",
    "ValidationSafetyFixture", "VocabularyIdentity", "ZeroDimensionalBase",
    "arm_model_family_sha256", "canonical_json_bytes", "h6_source_law_identity",
    "h6_source_law_marker_identity",
    "issue_prediction_readiness", "issue_prediction_readiness_v2",
    "require_prefix_pass",
]
