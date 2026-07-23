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
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import ClassVar, Literal

import torch

from .results import GateStatus


_LOWER_HEX = frozenset("0123456789abcdef")
_H5_LABELS = (
    "exact_coordinate",
    "generalized_em",
    "natural_gradient_proposal",
)
H6_PREFIX_REQUIRED_CHECKS = (
    "signature_import",
    "taint_dataflow",
    "source_mask",
    "cache_identity",
    "data_safety",
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
        if self.equality_checked is not True:
            raise ValueError("equality_checked must be true")
        self.complete_decomposition.assert_intact()
        self.total_language_elbo.assert_intact()
        if not torch.equal(self.complete_decomposition.value(), self.total_language_elbo.value()):
            raise ValueError("complete decomposition does not equal total language ELBO")
        _require_sha256(self.canonical_sha256, "canonical_sha256")


@dataclass(frozen=True)
class EmissionOnlyAblationTerms:
    objective_kind: Literal["emission_only_ablation_non_elbo"]
    ordered_emission_terms: tuple[H6FactorTerm, ...]
    total: FrozenTensorSnapshot
    canonical_sha256: str

    def __post_init__(self) -> None:
        if self.objective_kind != "emission_only_ablation_non_elbo":
            raise ValueError("emission-only terms must remain a non-ELBO ablation")
        if any(term.partition != "emission" for term in self.ordered_emission_terms):
            raise ValueError("emission-only ablation accepts only emission factors")
        self.total.assert_intact()
        _require_sha256(self.canonical_sha256, "canonical_sha256")


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
        checks = payload.get("checks")
        if not isinstance(checks, dict) or tuple(sorted(checks)) != tuple(sorted(H6_PREFIX_REQUIRED_CHECKS)):
            raise ValueError("validation payload has incomplete required checks")
        if any(type(value) is not bool for value in checks.values()):
            raise ValueError("validation checks must be booleans")
        if self.status is EvidenceStatus.PASS:
            if self.obligations or not all(checks.values()):
                raise ValueError("PASS requires every check and no obligation")
        elif self.status is EvidenceStatus.INCONCLUSIVE and not self.obligations:
            raise ValueError("INCONCLUSIVE requires an obligation")

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


def require_prefix_pass(
    key: PrefixCaseKey,
    certificates: Mapping[PrefixCaseKey, PrefixCertificate],
) -> PrefixCertificate:
    certificate = certificates.get(key)
    if certificate is None or certificate.key != key or certificate.status is not EvidenceStatus.PASS:
        raise ValueError("exact PASS prefix certificate is required")
    certificate.__post_init__()
    return certificate


@dataclass(frozen=True)
class PredictionCorrectnessArtifactRef:
    gate: Literal["H1", "H2", "H3", "H5"]
    artifact_path: Path
    manifest_sha256: str
    git_head: str
    dirty_digest: str
    config_sha256: str
    validation_payload_sha256: str
    status: GateStatus

    def __post_init__(self) -> None:
        if self.gate not in ("H1", "H2", "H3", "H5"):
            raise ValueError("gate must be exactly one of H1, H2, H3, H5")
        if not isinstance(self.artifact_path, Path):
            raise ValueError("artifact_path must be a Path")
        for name in ("manifest_sha256", "dirty_digest", "config_sha256", "validation_payload_sha256"):
            _require_sha256(getattr(self, name), name)
        _require_git_head(self.git_head)
        if not isinstance(self.status, GateStatus):
            raise ValueError("status must be a GateStatus")

    @classmethod
    def from_bytes(
        cls,
        *,
        gate: Literal["H1", "H2", "H3", "H5"],
        artifact_path: Path,
        manifest_bytes: bytes,
        git_head: str,
        dirty_digest: str,
        config_sha256: str,
        validation_payload_bytes: bytes,
        status: GateStatus,
    ) -> "PredictionCorrectnessArtifactRef":
        return cls(
            gate,
            artifact_path,
            hashlib.sha256(manifest_bytes).hexdigest(),
            git_head,
            dirty_digest,
            config_sha256,
            hashlib.sha256(validation_payload_bytes).hexdigest(),
            status,
        )

    def verify_bytes(self, *, manifest_bytes: bytes, validation_payload_bytes: bytes) -> None:
        if hashlib.sha256(manifest_bytes).hexdigest() != self.manifest_sha256:
            raise ValueError("manifest_sha256 does not match manifest bytes")
        if hashlib.sha256(validation_payload_bytes).hexdigest() != self.validation_payload_sha256:
            raise ValueError("validation_payload_sha256 does not match payload bytes")


@dataclass(frozen=True)
class H1PrefixPriorArtifactRef:
    artifact_path: Path
    manifest_sha256: str
    git_head: str
    dirty_digest: str
    generative_factor_schema_sha256: str
    config_sha256: str
    validation_payload_sha256: str
    status: GateStatus

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_path, Path):
            raise ValueError("artifact_path must be a Path")
        for name in (
            "manifest_sha256", "dirty_digest", "generative_factor_schema_sha256",
            "config_sha256", "validation_payload_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        _require_git_head(self.git_head)
        if not isinstance(self.status, GateStatus):
            raise ValueError("status must be a GateStatus")


@dataclass(frozen=True)
class SmcAccuracyArtifactRef:
    artifact_path: Path
    manifest_sha256: str
    git_head: str
    dirty_digest: str
    estimator_sha256: str
    fixture_set_sha256: str
    validation_payload_sha256: str
    status: GateStatus

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_path, Path):
            raise ValueError("artifact_path must be a Path")
        for name in (
            "manifest_sha256", "dirty_digest", "estimator_sha256",
            "fixture_set_sha256", "validation_payload_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        _require_git_head(self.git_head)
        if not isinstance(self.status, GateStatus):
            raise ValueError("status must be a GateStatus")


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


@dataclass(frozen=True)
class H6PredictionReadinessToken:
    readiness_schema: Literal["h6-prediction-readiness-v1"]
    git_head: str
    dirty_digest: str
    experiment_config_sha256: str
    correctness_manifests: tuple[tuple[str, str], ...]
    h1_prefix_prior_manifest_sha256: str | None
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

    _SHA_FIELDS: ClassVar[tuple[str, ...]] = (
        "dirty_digest", "experiment_config_sha256", "h5_update_binding_sha256",
        "h6_training_schedule_sha256", "smc_validation_manifest_sha256",
        "critical_values_sha256", "endpoint_smc_protocol_sha256",
        "attribution_matrix_sha256", "matching_set_sha256",
        "prefix_certificate_set_sha256", "data_identity_sha256", "access_policy_sha256",
    )

    def __post_init__(self) -> None:
        if self.readiness_schema != "h6-prediction-readiness-v1" or self.status != "PASS":
            raise ValueError("readiness token must be the PASS v1 schema")
        _require_git_head(self.git_head)
        for name in self._SHA_FIELDS:
            _require_sha256(getattr(self, name), name)
        if self.h1_prefix_prior_manifest_sha256 is not None:
            _require_sha256(self.h1_prefix_prior_manifest_sha256, "h1_prefix_prior_manifest_sha256")
        gates = tuple(gate for gate, _ in self.correctness_manifests)
        if gates != ("H1", "H2", "H3", "H5"):
            raise ValueError("correctness manifests must be exactly H1, H2, H3, H5")
        for gate, digest in self.correctness_manifests:
            _require_sha256(digest, f"correctness_manifests[{gate}]")
        expected = _owned_hash("vfe4.h6.prediction-readiness.v1", self._payload())
        if self.readiness_sha256 != expected:
            raise ValueError("readiness_sha256 does not match readiness fields")

    def _payload(self) -> dict[str, object]:
        return {
            "readiness_schema": self.readiness_schema,
            "git_head": self.git_head,
            **{name: getattr(self, name) for name in self._SHA_FIELDS},
            "correctness_manifests": self.correctness_manifests,
            "h1_prefix_prior_manifest_sha256": self.h1_prefix_prior_manifest_sha256,
            "status": self.status,
        }

    @classmethod
    def create(cls, **values: object) -> "H6PredictionReadinessToken":
        payload = {
            "readiness_schema": "h6-prediction-readiness-v1",
            "git_head": values["git_head"],
            **{name: values[name] for name in cls._SHA_FIELDS},
            "correctness_manifests": tuple(values["correctness_manifests"]),
            "h1_prefix_prior_manifest_sha256": values.get("h1_prefix_prior_manifest_sha256"),
            "status": "PASS",
        }
        digest = _owned_hash("vfe4.h6.prediction-readiness.v1", payload)
        return cls(
            "h6-prediction-readiness-v1",
            values["git_head"],
            values["dirty_digest"],
            values["experiment_config_sha256"],
            tuple(values["correctness_manifests"]),
            values.get("h1_prefix_prior_manifest_sha256"),
            values["h5_update_binding_sha256"],
            values["h6_training_schedule_sha256"],
            values["smc_validation_manifest_sha256"],
            values["critical_values_sha256"],
            values["endpoint_smc_protocol_sha256"],
            values["attribution_matrix_sha256"],
            values["matching_set_sha256"],
            values["prefix_certificate_set_sha256"],
            values["data_identity_sha256"],
            values["access_policy_sha256"],
            digest,
            "PASS",
        )


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


@dataclass(frozen=True)
class PredictionDecision:
    status: EvidenceStatus
    primary_interval: tuple[float, float] | None
    obligations: tuple[str, ...]

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
        if self.status is EvidenceStatus.INCONCLUSIVE and not self.obligations:
            raise ValueError("INCONCLUSIVE requires obligations")


__all__ = [
    "ArmId", "CausalDag", "CausalDagRow", "EmissionOnlyAblationTerms",
    "EndpointSmcProtocol", "EstimatorSpec", "EvidenceStatus", "FrozenTensorSnapshot",
    "H1PrefixPriorArtifactRef", "H5UpdateBinding", "H6ArmPhaseSchedule",
    "H6FactorTerm", "H6LanguageElboTerms", "H6LanguageStructure", "H6OuterSchedule",
    "H6PredictionReadinessToken", "H6TrainingSchedule", "H6_PREFIX_REQUIRED_CHECKS",
    "NllTotals", "PredictionCorrectnessArtifactRef", "PredictionDecision",
    "PrefixCaseKey", "PrefixCertificate", "SmcAccuracyArtifactRef", "TrainingPhase",
    "VocabularyIdentity", "ZeroDimensionalBase", "canonical_json_bytes",
    "require_prefix_pass",
]
