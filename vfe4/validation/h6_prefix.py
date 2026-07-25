"""Bounded dynamic H6 prefix checks with an explicit subset/full boundary."""

from __future__ import annotations

import hashlib
import inspect
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Literal

import torch

from vfe4.data.windows import CausalPrefix
from vfe4.numerics import AllInvalidSourceRowError
from vfe4.predictive import (
    EstimatorIdentity,
    EstimatorStream,
    PrefixCache,
    PriorPrediction,
    PriorPredictor,
    vocabulary_identity_sha256,
)
from vfe4.training.arms import (
    ArmConfig,
    H6CausalTransformer,
    LatentLanguageArmModel,
    MeanPooledPrefixFloor,
    literal_arm_semantic_payload,
)
from vfe4.types import (
    ArmId,
    EvidenceStatus,
    EstimatorSpec,
    PrefixCaseKey,
    ValidationSafetyFixture,
    VocabularyIdentity,
)
from vfe4.types.h6 import arm_model_family_sha256, canonical_json_bytes


SMALL_EXPECTED_BY_POSITION = (6561, 2187, 729, 243)
SMALL_EXPECTED_TOTAL = 9720
VALIDATION_EXPECTED_TOTAL = 4096
MAX_FOCUSED_CASES = 16
PERTURBATION_GENERATOR_SEED = 2026072197
PERTURBATION_FIXTURE_PATH = (
    Path(__file__).with_name("fixtures")
    / "h6_validation_perturbations_v1.json"
)

_LOWER_HEX = frozenset("0123456789abcdef")
_PERTURBATION_CASE_DOMAIN = "vfe4.h6.validation-perturbation-case.v1"
_PERTURBATION_MANIFEST_DOMAIN = (
    "vfe4.h6.validation-perturbations-manifest.v1"
)
_REPORT_CHECK_NAMES = (
    "signature_and_identity",
    "dynamic_target_suffix_leakage",
    "cache_identity",
    "source_mask",
    "case_inventory",
    "validation_data_safety",
)


def _sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase 64-hex SHA-256")
    return value


def _owned_hash(domain: str, payload: object) -> str:
    return hashlib.sha256(
        domain.encode("ascii") + b"\x00" + canonical_json_bytes(payload)
    ).hexdigest()


def _strict_json(raw: bytes, name: str) -> dict[str, object]:
    if type(raw) is not bytes:
        raise ValueError(f"{name} must be immutable bytes")

    def reject_duplicates(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(item: str) -> object:
        raise ValueError(f"nonfinite JSON constant {item!r}")

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not strict UTF-8 JSON") from exc
    if type(value) is not dict:
        raise ValueError(f"{name} must contain one JSON object")
    return value


@dataclass(frozen=True)
class DynamicPrefixCase:
    ordinal: int
    receiver_t: int
    shared_prefix: tuple[int, ...]
    left_tail: tuple[int, ...]
    right_tail: tuple[int, ...]
    case_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "receiver_t": self.receiver_t,
            "shared_prefix": self.shared_prefix,
            "left_tail": self.left_tail,
            "right_tail": self.right_tail,
        }

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("case ordinal must be nonnegative")
        if type(self.receiver_t) is not int or self.receiver_t <= 0:
            raise ValueError("case receiver_t must be positive")
        for name in ("shared_prefix", "left_tail", "right_tail"):
            values = getattr(self, name)
            if type(values) is not tuple or any(
                type(value) is not int or value < 0 for value in values
            ):
                raise ValueError(f"{name} must be an exact nonnegative tuple")
        if len(self.shared_prefix) != self.receiver_t - 1:
            raise ValueError("shared prefix length must equal receiver_t-1")
        if not self.left_tail or len(self.left_tail) != len(self.right_tail):
            raise ValueError("ordered tails must be nonempty and equal length")
        if self.case_sha256 != _owned_hash(
            "vfe4.h6.dynamic-prefix-case.v1", self.canonical_payload()
        ):
            raise ValueError("dynamic prefix case identity is stale")

    @classmethod
    def create(
        cls,
        *,
        ordinal: int,
        receiver_t: int,
        shared_prefix: tuple[int, ...],
        left_tail: tuple[int, ...],
        right_tail: tuple[int, ...],
    ) -> "DynamicPrefixCase":
        payload = {
            "ordinal": ordinal,
            "receiver_t": receiver_t,
            "shared_prefix": tuple(shared_prefix),
            "left_tail": tuple(left_tail),
            "right_tail": tuple(right_tail),
        }
        return cls(
            **payload,
            case_sha256=_owned_hash(
                "vfe4.h6.dynamic-prefix-case.v1", payload
            ),
        )


class PairSideHarness:
    """Hash-bound oracle state that seals one left/right tail before a call."""

    def __init__(self) -> None:
        self._trace: list[str] = []
        self._current_case_sha256: str | None = None
        self._current_side: Literal["left", "right"] | None = None
        self._current_tail: tuple[int, ...] | None = None
        self._current_binding_sha256: str | None = None
        self._current_ordinal: int | None = None

    @property
    def trace_count(self) -> int:
        return len(self._trace)

    @property
    def current_side(self) -> Literal["left", "right"] | None:
        return self._current_side

    @property
    def current_tail(self) -> tuple[int, ...] | None:
        return self._current_tail

    @property
    def current_binding_sha256(self) -> str | None:
        return self._current_binding_sha256

    def bind(
        self,
        *,
        case_sha256: str,
        side: Literal["left", "right"],
        tail: tuple[int, ...],
    ) -> str:
        _sha256(case_sha256, "pair harness case_sha256")
        if side not in ("left", "right"):
            raise ValueError("pair harness side must be left or right")
        if (
            type(tail) is not tuple
            or not tail
            or any(type(token) is not int or token < 0 for token in tail)
        ):
            raise ValueError("pair harness tail must be a nonempty token tuple")
        ordinal = len(self._trace)
        payload = {
            "ordinal": ordinal,
            "case_sha256": case_sha256,
            "side": side,
            "tail": tail,
        }
        binding = _owned_hash("vfe4.h6.pair-side-binding.v1", payload)
        self._current_case_sha256 = case_sha256
        self._current_side = side
        self._current_tail = tail
        self._current_binding_sha256 = binding
        self._current_ordinal = ordinal
        self._trace.append(binding)
        return binding

    def assert_current(
        self,
        *,
        case_sha256: str,
        side: Literal["left", "right"],
        tail: tuple[int, ...],
        binding_sha256: str,
    ) -> None:
        if self._current_ordinal is None:
            raise ValueError("pair harness has no current binding")
        payload = {
            "ordinal": self._current_ordinal,
            "case_sha256": case_sha256,
            "side": side,
            "tail": tail,
        }
        expected = _owned_hash("vfe4.h6.pair-side-binding.v1", payload)
        if (
            binding_sha256 != expected
            or self._current_case_sha256 != case_sha256
            or self._current_side != side
            or self._current_tail != tail
            or self._current_binding_sha256 != binding_sha256
            or not self._trace
            or self._trace[-1] != binding_sha256
        ):
            raise ValueError("pair harness state changed during prediction")

    @property
    def manifest_sha256(self) -> str:
        return _owned_hash(
            "vfe4.h6.pair-side-harness-manifest.v1", tuple(self._trace)
        )


def _json_int_tuple(value: object, name: str) -> tuple[int, ...]:
    if type(value) is not list or any(type(item) is not int for item in value):
        raise ValueError(f"{name} must be a JSON integer array")
    return tuple(value)


@dataclass(frozen=True)
class ValidationPerturbationRecord:
    case_id: str
    case_index: int
    source_window_index: int
    window_start: int
    real_target_count: int
    receiver_t: int
    prefix_token_ids: tuple[int, ...]
    left_tail_token_ids: tuple[int, ...]
    right_tail_token_ids: tuple[int, ...]
    vocabulary_size: int
    vocabulary_sha256: str
    validation_token_sha256: str
    validation_safety_fixture_sha256: str
    case_sha256: str

    def canonical_payload(self, *, include_sha256: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "case_id": self.case_id,
            "case_index": self.case_index,
            "source_window_index": self.source_window_index,
            "window_start": self.window_start,
            "real_target_count": self.real_target_count,
            "receiver_t": self.receiver_t,
            "prefix_token_ids": self.prefix_token_ids,
            "left_tail_token_ids": self.left_tail_token_ids,
            "right_tail_token_ids": self.right_tail_token_ids,
            "vocabulary_size": self.vocabulary_size,
            "vocabulary_sha256": self.vocabulary_sha256,
            "validation_token_sha256": self.validation_token_sha256,
            "validation_safety_fixture_sha256": (
                self.validation_safety_fixture_sha256
            ),
        }
        if include_sha256:
            payload["case_sha256"] = self.case_sha256
        return payload

    def __post_init__(self) -> None:
        if type(self.case_index) is not int:
            raise ValueError("validation perturbation case_index must be an integer")
        if (
            type(self.case_id) is not str
            or self.case_id != f"validation-window-{self.case_index:04d}"
            or not 0 <= self.case_index < VALIDATION_EXPECTED_TOTAL
            or self.source_window_index != self.case_index
            or type(self.window_start) is not int
            or self.window_start < 0
            or type(self.real_target_count) is not int
            or not 1 <= self.real_target_count <= 32
            or type(self.receiver_t) is not int
            or not 1 <= self.receiver_t <= self.real_target_count
            or self.vocabulary_size != 258
        ):
            raise ValueError("validation perturbation record metadata is invalid")
        tail_length = self.real_target_count - self.receiver_t + 1
        if (
            type(self.prefix_token_ids) is not tuple
            or type(self.left_tail_token_ids) is not tuple
            or type(self.right_tail_token_ids) is not tuple
            or len(self.prefix_token_ids) != self.receiver_t - 1
            or len(self.left_tail_token_ids) != tail_length
            or len(self.right_tail_token_ids) != tail_length
        ):
            raise ValueError("validation perturbation sequence lengths are invalid")
        for name in (
            "prefix_token_ids",
            "left_tail_token_ids",
            "right_tail_token_ids",
        ):
            values = getattr(self, name)
            if any(
                type(token) is not int
                or not 0 <= token < self.vocabulary_size
                for token in values
            ):
                raise ValueError(f"{name} contains a token outside the vocabulary")
        if any(
            left == right
            for left, right in zip(
                self.left_tail_token_ids,
                self.right_tail_token_ids,
            )
        ):
            raise ValueError("every varied target/suffix coordinate must differ")
        for name in (
            "vocabulary_sha256",
            "validation_token_sha256",
            "validation_safety_fixture_sha256",
            "case_sha256",
        ):
            _sha256(getattr(self, name), name)
        expected = _owned_hash(
            _PERTURBATION_CASE_DOMAIN,
            self.canonical_payload(include_sha256=False),
        )
        if self.case_sha256 != expected:
            raise ValueError("validation perturbation case identity is stale")

    def dynamic_case(self) -> DynamicPrefixCase:
        return DynamicPrefixCase.create(
            ordinal=self.case_index,
            receiver_t=self.receiver_t,
            shared_prefix=self.prefix_token_ids,
            left_tail=self.left_tail_token_ids,
            right_tail=self.right_tail_token_ids,
        )


@dataclass(frozen=True)
class FrozenValidationPerturbations:
    schema_version: Literal["h6-validation-perturbations-v1"]
    generator_version: Literal["h6-validation-perturbations-v1"]
    seed: Literal[2026072197]
    vocabulary: VocabularyIdentity
    vocabulary_sha256: str
    validation_token_sha256: str
    validation_safety_fixture_sha256: str
    full_count: Literal[4096]
    materialized_count: int
    records: tuple[ValidationPerturbationRecord, ...]
    manifest_sha256: str
    source_fixture_verified: bool
    raw_sha256: str
    _canonical_bytes: bytes = field(repr=False, compare=False)

    def canonical_payload(self, *, include_manifest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "generator_version": self.generator_version,
            "seed": self.seed,
            "vocabulary": {
                "vocabulary_id": self.vocabulary.vocabulary_id,
                "size": self.vocabulary.size,
                "tokenizer_spec_sha256": self.vocabulary.tokenizer_spec_sha256,
                "vocabulary_sha256": self.vocabulary_sha256,
            },
            "validation_token_sha256": self.validation_token_sha256,
            "validation_safety_fixture_sha256": (
                self.validation_safety_fixture_sha256
            ),
            "full_count": self.full_count,
            "materialized_count": self.materialized_count,
            "records": tuple(record.canonical_payload() for record in self.records),
        }
        if include_manifest:
            payload["manifest_sha256"] = self.manifest_sha256
        return payload

    def __post_init__(self) -> None:
        if (
            self.schema_version != "h6-validation-perturbations-v1"
            or self.generator_version != "h6-validation-perturbations-v1"
            or self.seed != PERTURBATION_GENERATOR_SEED
            or self.full_count != VALIDATION_EXPECTED_TOTAL
            or type(self.materialized_count) is not int
            or self.materialized_count <= 0
            or type(self.source_fixture_verified) is not bool
        ):
            raise ValueError("unsupported validation perturbation contract")
        if type(self.vocabulary) is not VocabularyIdentity:
            raise ValueError("perturbations require an exact vocabulary")
        self.vocabulary.__post_init__()
        if self.vocabulary.size != 258:
            raise ValueError("validation perturbations require the V=258 vocabulary")
        if self.vocabulary_sha256 != vocabulary_identity_sha256(self.vocabulary):
            raise ValueError("validation perturbation vocabulary identity is stale")
        for name in (
            "vocabulary_sha256",
            "validation_token_sha256",
            "validation_safety_fixture_sha256",
            "manifest_sha256",
            "raw_sha256",
        ):
            _sha256(getattr(self, name), name)
        if (
            type(self.records) is not tuple
            or len(self.records) != self.materialized_count
            or any(type(item) is not ValidationPerturbationRecord for item in self.records)
        ):
            raise ValueError("perturbation records do not match materialized_count")
        indices = tuple(record.case_index for record in self.records)
        if indices != tuple(sorted(indices)) or len(set(indices)) != len(indices):
            raise ValueError("perturbation record indices must be unique and sorted")
        for record in self.records:
            record.__post_init__()
            if (
                record.vocabulary_size != self.vocabulary.size
                or record.vocabulary_sha256 != self.vocabulary_sha256
                or record.validation_token_sha256 != self.validation_token_sha256
                or record.validation_safety_fixture_sha256
                != self.validation_safety_fixture_sha256
            ):
                raise ValueError("perturbation record identity differs from its root")
        expected_manifest = _owned_hash(
            _PERTURBATION_MANIFEST_DOMAIN,
            self.canonical_payload(include_manifest=False),
        )
        if self.manifest_sha256 != expected_manifest:
            raise ValueError("validation perturbation manifest identity is stale")
        if type(self._canonical_bytes) is not bytes:
            raise ValueError("canonical perturbation bytes must be immutable bytes")
        expected_bytes = canonical_json_bytes(self.canonical_payload())
        if self._canonical_bytes != expected_bytes:
            raise ValueError("validation perturbation bytes are not canonical")
        if hashlib.sha256(self._canonical_bytes).hexdigest() != self.raw_sha256:
            raise ValueError("validation perturbation raw identity is stale")

    @property
    def materialization(self) -> Literal["focused_subset", "authorized_full"]:
        return (
            "authorized_full"
            if self.materialized_count == self.full_count
            and tuple(record.case_index for record in self.records)
            == tuple(range(self.full_count))
            else "focused_subset"
        )

    @property
    def dynamic_cases(self) -> tuple[DynamicPrefixCase, ...]:
        return tuple(record.dynamic_case() for record in self.records)

    @property
    def canonical_bytes(self) -> bytes:
        self.__post_init__()
        return bytes(self._canonical_bytes)


def load_frozen_validation_perturbations(
    source: bytes | Path = PERTURBATION_FIXTURE_PATH,
    *,
    expected_vocabulary: VocabularyIdentity,
    validation_fixture: ValidationSafetyFixture | None = None,
    validation_fixture_bytes: bytes | None = None,
) -> FrozenValidationPerturbations:
    """Load the independent oracle's canonical V=258 case-file schema."""

    if type(source) is bytes:
        encoded = bytes(source)
    elif isinstance(source, Path):
        encoded = source.read_bytes()
    else:
        raise ValueError("perturbation source must be bytes or a Path")
    # Git-managed JSON text may carry one terminal LF.  The hashed oracle
    # payload remains the exact canonical bytes before that text terminator.
    raw = encoded[:-1] if encoded.endswith(b"\n") else encoded
    root = _strict_json(raw, "H6 validation perturbations")
    if canonical_json_bytes(root) != raw:
        raise ValueError("perturbation fixture must use canonical JSON bytes")
    expected_fields = {
        "schema_version",
        "generator_version",
        "seed",
        "vocabulary",
        "validation_token_sha256",
        "validation_safety_fixture_sha256",
        "full_count",
        "materialized_count",
        "records",
        "manifest_sha256",
    }
    if set(root) != expected_fields:
        raise ValueError("perturbation fixture fields are not exact")
    vocabulary = root["vocabulary"]
    if type(vocabulary) is not dict or set(vocabulary) != {
        "vocabulary_id",
        "size",
        "tokenizer_spec_sha256",
        "vocabulary_sha256",
    }:
        raise ValueError("perturbation vocabulary fields are not exact")
    observed_vocabulary = VocabularyIdentity(
        vocabulary["vocabulary_id"],
        vocabulary["size"],
        vocabulary["tokenizer_spec_sha256"],
    )
    if observed_vocabulary != expected_vocabulary:
        raise ValueError("perturbation vocabulary identity is stale")
    observed_vocabulary_sha256 = _sha256(
        vocabulary["vocabulary_sha256"],
        "vocabulary.vocabulary_sha256",
    )
    if observed_vocabulary_sha256 != vocabulary_identity_sha256(observed_vocabulary):
        raise ValueError("perturbation vocabulary hash is stale")
    raw_records = root["records"]
    if type(raw_records) is not list or not raw_records:
        raise ValueError("perturbation records must be a nonempty JSON array")
    record_fields = {
        "case_id",
        "case_index",
        "source_window_index",
        "window_start",
        "real_target_count",
        "receiver_t",
        "prefix_token_ids",
        "left_tail_token_ids",
        "right_tail_token_ids",
        "vocabulary_size",
        "vocabulary_sha256",
        "validation_token_sha256",
        "validation_safety_fixture_sha256",
        "case_sha256",
    }
    records: list[ValidationPerturbationRecord] = []
    for offset, raw_record in enumerate(raw_records):
        if type(raw_record) is not dict or set(raw_record) != record_fields:
            raise ValueError(f"perturbation record {offset} fields are not exact")
        records.append(
            ValidationPerturbationRecord(
                case_id=raw_record["case_id"],
                case_index=raw_record["case_index"],
                source_window_index=raw_record["source_window_index"],
                window_start=raw_record["window_start"],
                real_target_count=raw_record["real_target_count"],
                receiver_t=raw_record["receiver_t"],
                prefix_token_ids=_json_int_tuple(
                    raw_record["prefix_token_ids"],
                    f"records[{offset}].prefix_token_ids",
                ),
                left_tail_token_ids=_json_int_tuple(
                    raw_record["left_tail_token_ids"],
                    f"records[{offset}].left_tail_token_ids",
                ),
                right_tail_token_ids=_json_int_tuple(
                    raw_record["right_tail_token_ids"],
                    f"records[{offset}].right_tail_token_ids",
                ),
                vocabulary_size=raw_record["vocabulary_size"],
                vocabulary_sha256=raw_record["vocabulary_sha256"],
                validation_token_sha256=raw_record["validation_token_sha256"],
                validation_safety_fixture_sha256=(
                    raw_record["validation_safety_fixture_sha256"]
                ),
                case_sha256=raw_record["case_sha256"],
            )
        )
    token_sha = _sha256(root["validation_token_sha256"], "validation_token_sha256")
    fixture_sha = _sha256(
        root["validation_safety_fixture_sha256"],
        "validation_safety_fixture_sha256",
    )
    if (validation_fixture is None) != (validation_fixture_bytes is None):
        raise ValueError("typed validation fixture and bytes must be supplied together")
    source_verified = False
    if validation_fixture is not None:
        if type(validation_fixture) is not ValidationSafetyFixture:
            raise ValueError("validation_fixture must be exact")
        assert validation_fixture_bytes is not None
        validation_fixture.verify_fixture_bytes(validation_fixture_bytes)
        if (
            validation_fixture.fixture_sha256 != fixture_sha
            or validation_fixture.validation_token_sha256 != token_sha
        ):
            raise ValueError("perturbation source fixture identity is stale")
        source_verified = True
    return FrozenValidationPerturbations(
        schema_version=root["schema_version"],
        generator_version=root["generator_version"],
        seed=root["seed"],
        vocabulary=observed_vocabulary,
        vocabulary_sha256=observed_vocabulary_sha256,
        validation_token_sha256=token_sha,
        validation_safety_fixture_sha256=fixture_sha,
        full_count=root["full_count"],
        materialized_count=root["materialized_count"],
        records=tuple(records),
        manifest_sha256=root["manifest_sha256"],
        source_fixture_verified=source_verified,
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        _canonical_bytes=raw,
    )


@dataclass(frozen=True)
class DynamicExecutionPlan:
    mode: Literal["focused_subset", "authorized_full"]
    case_family: Literal["small", "validation"]
    expected_by_position: tuple[int, ...]
    full_expected_count: int
    authorization_sha256: str | None
    plan_sha256: str

    def __post_init__(self) -> None:
        expected = (
            SMALL_EXPECTED_BY_POSITION
            if self.case_family == "small"
            else (VALIDATION_EXPECTED_TOTAL,)
        )
        if (
            self.mode not in ("focused_subset", "authorized_full")
            or self.expected_by_position != expected
            or self.full_expected_count != sum(expected)
        ):
            raise ValueError("dynamic execution plan counts are not frozen")
        if self.mode == "authorized_full":
            _sha256(self.authorization_sha256, "authorization_sha256")
        elif self.authorization_sha256 is not None:
            raise ValueError("focused subset cannot carry full authorization")
        payload = {
            "mode": self.mode,
            "case_family": self.case_family,
            "expected_by_position": self.expected_by_position,
            "full_expected_count": self.full_expected_count,
            "authorization_sha256": self.authorization_sha256,
        }
        if self.plan_sha256 != _owned_hash(
            "vfe4.h6.dynamic-execution-plan.v1", payload
        ):
            raise ValueError("dynamic execution plan identity is stale")

    @classmethod
    def create(
        cls,
        *,
        mode: Literal["focused_subset", "authorized_full"],
        case_family: Literal["small", "validation"],
        authorization_sha256: str | None = None,
    ) -> "DynamicExecutionPlan":
        expected = (
            SMALL_EXPECTED_BY_POSITION
            if case_family == "small"
            else (VALIDATION_EXPECTED_TOTAL,)
        )
        payload = {
            "mode": mode,
            "case_family": case_family,
            "expected_by_position": expected,
            "full_expected_count": sum(expected),
            "authorization_sha256": authorization_sha256,
        }
        return cls(
            **payload,
            plan_sha256=_owned_hash(
                "vfe4.h6.dynamic-execution-plan.v1", payload
            ),
        )


@dataclass(frozen=True)
class SourceMaskObservation:
    case_sha256: str
    config_sha256: str
    bank: Literal["state", "model"]
    receiver_t: int
    declared_parents: tuple[int, ...]
    log_probabilities: tuple[float, ...]
    raw_bytes_sha256: str
    observation_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "case_sha256": self.case_sha256,
            "config_sha256": self.config_sha256,
            "bank": self.bank,
            "receiver_t": self.receiver_t,
            "declared_parents": self.declared_parents,
            "log_probabilities": self.log_probabilities,
            "raw_bytes_sha256": self.raw_bytes_sha256,
        }

    def __post_init__(self) -> None:
        _sha256(self.case_sha256, "case_sha256")
        _sha256(self.config_sha256, "config_sha256")
        if self.bank not in ("state", "model"):
            raise ValueError("source bank is invalid")
        if type(self.receiver_t) is not int or self.receiver_t <= 0:
            raise ValueError("source-mask receiver must be positive")
        if (
            type(self.declared_parents) is not tuple
            or tuple(sorted(set(self.declared_parents)))
            != self.declared_parents
            or any(
                type(item) is not int or not 0 <= item < self.receiver_t
                for item in self.declared_parents
            )
        ):
            raise ValueError("source-mask parents must be exact and causal")
        if (
            type(self.log_probabilities) is not tuple
            or len(self.log_probabilities) != self.receiver_t
            or any(type(item) is not float for item in self.log_probabilities)
        ):
            raise ValueError("source-mask log probabilities have the wrong shape")
        raw = torch.tensor(
            self.log_probabilities, dtype=torch.float64
        ).contiguous().view(torch.uint8).numpy().tobytes(order="C")
        if self.raw_bytes_sha256 != hashlib.sha256(raw).hexdigest():
            raise ValueError("source-mask raw byte identity is stale")
        if self.observation_sha256 != _owned_hash(
            "vfe4.h6.source-mask-observation.v1", self.canonical_payload()
        ):
            raise ValueError("source-mask observation identity is stale")

    @classmethod
    def capture(
        cls,
        *,
        case_sha256: str,
        config_sha256: str,
        bank: Literal["state", "model"],
        receiver_t: int,
        declared_parents: tuple[int, ...],
        log_probabilities: torch.Tensor,
    ) -> "SourceMaskObservation":
        if (
            type(log_probabilities) is not torch.Tensor
            or log_probabilities.dtype is not torch.float64
            or log_probabilities.device.type != "cpu"
            or log_probabilities.shape != (receiver_t,)
            or not log_probabilities.is_contiguous()
        ):
            raise ValueError("source-mask tensor metadata is invalid")
        raw = log_probabilities.detach().view(torch.uint8).numpy().tobytes(order="C")
        values = tuple(float(item) for item in log_probabilities.detach().tolist())
        payload = {
            "case_sha256": case_sha256,
            "config_sha256": config_sha256,
            "bank": bank,
            "receiver_t": receiver_t,
            "declared_parents": tuple(declared_parents),
            "log_probabilities": values,
            "raw_bytes_sha256": hashlib.sha256(raw).hexdigest(),
        }
        return cls(
            **payload,
            observation_sha256=_owned_hash(
                "vfe4.h6.source-mask-observation.v1", payload
            ),
        )


@dataclass(frozen=True)
class AllInvalidSourceObservation:
    config_sha256: str
    receiver_t: int
    outcome: Literal["rejected", "fallback_returned"]
    observed_type: str
    observation_sha256: str

    def __post_init__(self) -> None:
        _sha256(self.config_sha256, "config_sha256")
        if type(self.receiver_t) is not int or self.receiver_t <= 0:
            raise ValueError("all-invalid receiver_t must be positive")
        if self.outcome not in ("rejected", "fallback_returned"):
            raise ValueError("all-invalid probe outcome is invalid")
        if type(self.observed_type) is not str or not self.observed_type:
            raise ValueError("all-invalid observed_type must be nonempty")
        if (
            self.outcome == "rejected"
            and self.observed_type != "AllInvalidSourceRowError"
        ):
            raise ValueError("rejected outcome requires the exact exception")
        payload = {
            "config_sha256": self.config_sha256,
            "receiver_t": self.receiver_t,
            "outcome": self.outcome,
            "observed_type": self.observed_type,
        }
        if self.observation_sha256 != _owned_hash(
            "vfe4.h6.all-invalid-source-observation.v1", payload
        ):
            raise ValueError("all-invalid observation identity is stale")


def observe_all_invalid_source_rejection(
    *,
    config_sha256: str,
    receiver_t: int,
    probe: Callable[[], object],
) -> AllInvalidSourceObservation:
    """Return evidence only when the exact fail-closed exception is observed."""

    if not callable(probe):
        raise ValueError("all-invalid probe must be callable")
    try:
        result = probe()
    except AllInvalidSourceRowError:
        payload = {
            "config_sha256": config_sha256,
            "receiver_t": receiver_t,
            "outcome": "rejected",
            "observed_type": "AllInvalidSourceRowError",
        }
        return AllInvalidSourceObservation(
            **payload,
            observation_sha256=_owned_hash(
                "vfe4.h6.all-invalid-source-observation.v1", payload
            ),
        )
    payload = {
        "config_sha256": config_sha256,
        "receiver_t": receiver_t,
        "outcome": "fallback_returned",
        "observed_type": type(result).__qualname__,
    }
    return AllInvalidSourceObservation(
        **payload,
        observation_sha256=_owned_hash(
            "vfe4.h6.all-invalid-source-observation.v1", payload
        ),
    )


@dataclass(frozen=True)
class TensorByteIdentity:
    vocabulary_sha256: str
    dtype: str
    shape: tuple[int, ...]
    device: str
    contiguous: bool
    byte_count: int
    raw_bytes_sha256: str
    _raw_bytes: bytes = field(repr=False, compare=True)

    def __post_init__(self) -> None:
        _sha256(self.vocabulary_sha256, "vocabulary_sha256")
        if (
            self.dtype != "float64"
            or type(self.shape) is not tuple
            or len(self.shape) != 1
            or any(type(item) is not int or item <= 0 for item in self.shape)
            or self.device != "cpu"
            or self.contiguous is not True
            or type(self.byte_count) is not int
            or self.byte_count != 8 * self.shape[0]
            or type(self._raw_bytes) is not bytes
            or len(self._raw_bytes) != self.byte_count
            or hashlib.sha256(self._raw_bytes).hexdigest()
            != self.raw_bytes_sha256
        ):
            raise ValueError("tensor byte identity is invalid")


@dataclass(frozen=True)
class DynamicCheckResult:
    name: str
    status: EvidenceStatus
    expected_count: int
    completed_count: int
    violation_count: int
    first_counterexample: str | None
    obligations: tuple[str, ...]
    check_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "status": self.status.value,
            "expected_count": self.expected_count,
            "completed_count": self.completed_count,
            "violation_count": self.violation_count,
            "first_counterexample": self.first_counterexample,
            "obligations": self.obligations,
        }

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise ValueError("dynamic check name must be nonempty")
        if not isinstance(self.status, EvidenceStatus):
            raise ValueError("dynamic check status must be EvidenceStatus")
        for name in ("expected_count", "completed_count", "violation_count"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.completed_count > self.expected_count:
            raise ValueError("completed_count cannot exceed expected_count")
        if (
            type(self.obligations) is not tuple
            or any(type(item) is not str or not item for item in self.obligations)
        ):
            raise ValueError("dynamic check obligations are invalid")
        if self.status is EvidenceStatus.FAIL:
            if self.violation_count <= 0 or self.first_counterexample is None:
                raise ValueError("FAIL requires a witnessed counterexample")
        elif self.status is EvidenceStatus.INCONCLUSIVE:
            if self.violation_count or not self.obligations:
                raise ValueError("INCONCLUSIVE requires unresolved obligations")
        elif self.violation_count or self.obligations:
            raise ValueError("PASS cannot carry violations or obligations")
        if self.first_counterexample is not None and (
            type(self.first_counterexample) is not str
            or not self.first_counterexample
        ):
            raise ValueError("first_counterexample must be nonempty when present")
        if self.check_sha256 != _owned_hash(
            "vfe4.h6.dynamic-check-result.v1", self.canonical_payload()
        ):
            raise ValueError("dynamic check result identity is stale")

    @classmethod
    def create(
        cls,
        *,
        name: str,
        status: EvidenceStatus,
        expected_count: int,
        completed_count: int,
        violation_count: int = 0,
        first_counterexample: str | None = None,
        obligations: tuple[str, ...] = (),
    ) -> "DynamicCheckResult":
        payload = {
            "name": name,
            "status": status,
            "expected_count": expected_count,
            "completed_count": completed_count,
            "violation_count": violation_count,
            "first_counterexample": first_counterexample,
            "obligations": tuple(obligations),
        }
        canonical = {
            **payload,
            "status": status.value,
        }
        return cls(
            **payload,
            check_sha256=_owned_hash(
                "vfe4.h6.dynamic-check-result.v1", canonical
            ),
        )


@dataclass(frozen=True)
class DynamicPrefixReport:
    schema_version: Literal["h6-dynamic-prefix-report-v1"]
    key: PrefixCaseKey
    execution_plan_sha256: str
    model_state_sha256: str | None
    proposal_identity_sha256: str | None
    estimator_semantic_sha256: str | None
    estimator_artifact_bytes_sha256: str | None
    stream_seed: int
    completed_by_position: tuple[int, ...]
    checks: tuple[DynamicCheckResult, ...]
    status: EvidenceStatus
    obligations: tuple[str, ...]
    unresolved_diagnostics: tuple[str, ...]
    first_counterexample: str | None
    case_result_manifest_sha256: str
    cache_manifest_sha256: str
    pair_harness_manifest_sha256: str
    mask_manifest_sha256: str
    complete_case_manifest_sha256: str | None
    report_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "key": self.key.canonical_payload(),
            "execution_plan_sha256": self.execution_plan_sha256,
            "model_state_sha256": self.model_state_sha256,
            "proposal_identity_sha256": self.proposal_identity_sha256,
            "estimator_semantic_sha256": self.estimator_semantic_sha256,
            "estimator_artifact_bytes_sha256": (
                self.estimator_artifact_bytes_sha256
            ),
            "stream_seed": self.stream_seed,
            "completed_by_position": self.completed_by_position,
            "checks": tuple(check.canonical_payload() for check in self.checks),
            "status": self.status.value,
            "obligations": self.obligations,
            "unresolved_diagnostics": self.unresolved_diagnostics,
            "first_counterexample": self.first_counterexample,
            "case_result_manifest_sha256": self.case_result_manifest_sha256,
            "cache_manifest_sha256": self.cache_manifest_sha256,
            "pair_harness_manifest_sha256": (
                self.pair_harness_manifest_sha256
            ),
            "mask_manifest_sha256": self.mask_manifest_sha256,
            "complete_case_manifest_sha256": self.complete_case_manifest_sha256,
        }

    def __post_init__(self) -> None:
        if self.schema_version != "h6-dynamic-prefix-report-v1":
            raise ValueError("unsupported dynamic prefix report schema")
        if type(self.key) is not PrefixCaseKey:
            raise ValueError("dynamic prefix report key must be exact")
        self.key.__post_init__()
        for name in (
            "execution_plan_sha256",
            "case_result_manifest_sha256",
            "cache_manifest_sha256",
            "pair_harness_manifest_sha256",
            "mask_manifest_sha256",
            "report_sha256",
        ):
            _sha256(getattr(self, name), name)
        for name in (
            "model_state_sha256",
            "proposal_identity_sha256",
            "estimator_semantic_sha256",
            "estimator_artifact_bytes_sha256",
        ):
            value = getattr(self, name)
            if value is not None:
                _sha256(value, name)
        if self.complete_case_manifest_sha256 is not None:
            _sha256(
                self.complete_case_manifest_sha256,
                "complete_case_manifest_sha256",
            )
        if type(self.stream_seed) is not int or not 0 <= self.stream_seed < 2**64:
            raise ValueError("dynamic prefix stream seed is invalid")
        if (
            type(self.completed_by_position) is not tuple
            or not self.completed_by_position
            or any(type(item) is not int or item < 0 for item in self.completed_by_position)
        ):
            raise ValueError("completed_by_position is invalid")
        if (
            type(self.checks) is not tuple
            or tuple(check.name for check in self.checks) != _REPORT_CHECK_NAMES
        ):
            raise ValueError("dynamic prefix report checks are incomplete")
        for check in self.checks:
            if type(check) is not DynamicCheckResult:
                raise ValueError("dynamic prefix report checks must be exact")
            check.__post_init__()
        signature_check = self.checks[0]
        if signature_check.status is EvidenceStatus.PASS and any(
            getattr(self, name) is None
            for name in (
                "model_state_sha256",
                "proposal_identity_sha256",
                "estimator_semantic_sha256",
                "estimator_artifact_bytes_sha256",
            )
        ):
            raise ValueError("passing identity check requires complete provenance")
        for name in ("obligations", "unresolved_diagnostics"):
            values = getattr(self, name)
            if type(values) is not tuple or any(
                type(item) is not str or not item for item in values
            ):
                raise ValueError(f"{name} must contain nonempty strings")
        witnessed_failure = any(
            check.status is EvidenceStatus.FAIL for check in self.checks
        )
        unresolved = any(
            check.status is EvidenceStatus.INCONCLUSIVE for check in self.checks
        )
        expected_status = (
            EvidenceStatus.FAIL
            if witnessed_failure
            else EvidenceStatus.INCONCLUSIVE
            if unresolved or self.obligations
            else EvidenceStatus.PASS
        )
        if self.status is not expected_status:
            raise ValueError("dynamic prefix report status precedence is invalid")
        if self.status is EvidenceStatus.FAIL:
            if self.obligations or self.first_counterexample is None:
                raise ValueError("failed dynamic report requires one counterexample")
        elif self.status is EvidenceStatus.INCONCLUSIVE:
            if not self.obligations and not unresolved:
                raise ValueError("inconclusive dynamic report needs an obligation")
        elif self.obligations or self.first_counterexample is not None:
            raise ValueError("passing dynamic report cannot carry open evidence")
        inventory = self.checks[_REPORT_CHECK_NAMES.index("case_inventory")]
        if (self.complete_case_manifest_sha256 is None) == (
            inventory.status is EvidenceStatus.PASS
        ):
            raise ValueError("complete case manifest does not match inventory status")
        if self.report_sha256 != _owned_hash(
            "vfe4.h6.dynamic-prefix-report.v1", self.canonical_payload()
        ):
            raise ValueError("dynamic prefix report identity is stale")


class _WitnessedPredictionViolation(ValueError):
    pass


class _WitnessedCacheViolation(ValueError):
    pass


class _UnauditablePrefixCheck(ValueError):
    pass


@dataclass(frozen=True)
class _PredictionObservation:
    tensor: TensorByteIdentity
    cache_sha256: str
    cache_key_sha256: str
    estimator_record_sha256: str
    counter_trace_sha256: str

    def __post_init__(self) -> None:
        if type(self.tensor) is not TensorByteIdentity:
            raise ValueError("prediction observation tensor identity must be exact")
        self.tensor.__post_init__()
        for name in (
            "cache_sha256",
            "cache_key_sha256",
            "estimator_record_sha256",
            "counter_trace_sha256",
        ):
            _sha256(getattr(self, name), name)

    def tensor_payload(self) -> dict[str, object]:
        return {
            "vocabulary_sha256": self.tensor.vocabulary_sha256,
            "dtype": self.tensor.dtype,
            "shape": self.tensor.shape,
            "device": self.tensor.device,
            "contiguous": self.tensor.contiguous,
            "byte_count": self.tensor.byte_count,
            "raw_bytes_sha256": self.tensor.raw_bytes_sha256,
        }

    def cache_payload(self) -> dict[str, object]:
        return {
            "cache_sha256": self.cache_sha256,
            "cache_key_sha256": self.cache_key_sha256,
            "estimator_record_sha256": self.estimator_record_sha256,
            "counter_trace_sha256": self.counter_trace_sha256,
        }


def _tensor_identity(
    prediction: PriorPrediction,
    *,
    vocabulary: VocabularyIdentity,
) -> TensorByteIdentity:
    try:
        prediction.__post_init__()
    except ValueError as exc:
        raise _WitnessedPredictionViolation(
            f"PriorPrediction integrity failed: {exc}"
        ) from exc
    value = prediction.log_probs.value()
    if (
        value.dtype is not torch.float64
        or value.device.type != "cpu"
        or value.shape != (vocabulary.size,)
        or not value.is_contiguous()
    ):
        raise ValueError("prediction tensor metadata violates H6-Prefix")
    raw = value.detach().view(torch.uint8).numpy().tobytes(order="C")
    return TensorByteIdentity(
        vocabulary_sha256=vocabulary_identity_sha256(vocabulary),
        dtype="float64",
        shape=tuple(value.shape),
        device="cpu",
        contiguous=True,
        byte_count=len(raw),
        raw_bytes_sha256=hashlib.sha256(raw).hexdigest(),
        _raw_bytes=raw,
    )


def _validate_prediction_identity(
    prediction: PriorPrediction,
    *,
    prefix: CausalPrefix,
    predictor: PriorPredictor,
    stream: EstimatorStream,
) -> _PredictionObservation:
    if type(prediction) is not PriorPrediction:
        raise _WitnessedPredictionViolation(
            "predictor must return an exact PriorPrediction"
        )
    if type(prediction.cache) is not PrefixCache:
        raise _WitnessedCacheViolation(
            "prediction must carry an exact PrefixCache"
        )
    key = prediction.cache.key
    if (
        prediction.vocabulary != getattr(predictor, "vocabulary", None)
        or
        key.prefix_sha256 != prefix.prefix_sha256
        or key.prefix_tokens
        != tuple(int(item) for item in prefix.token_ids.tolist())
        or key.vocabulary_sha256
        != getattr(predictor, "vocabulary_sha256", None)
        or key.predictor_config_sha256
        != getattr(predictor, "predictor_config_sha256", None)
        or key.model_family_sha256
        != getattr(predictor, "model_family_sha256", None)
        or key.model_state_sha256
        != getattr(predictor, "model_state_sha256", None)
        or key.proposal_identity_sha256
        != getattr(predictor, "proposal_identity_sha256", None)
        or key.estimator_semantic_sha256
        != stream.estimator_semantic_sha256
        or key.estimator_artifact_bytes_sha256
        != stream.estimator_artifact_bytes_sha256
        or key.estimator_stream_sha256 != stream.stream_sha256
        or key.data_safety_sha256
        != getattr(predictor, "data_safety_sha256", None)
    ):
        raise _WitnessedCacheViolation(
            "prediction cache identity is stale or incomplete"
        )
    tensor = _tensor_identity(
        prediction, vocabulary=getattr(predictor, "vocabulary")
    )
    return _PredictionObservation(
        tensor=tensor,
        cache_sha256=prediction.cache.cache_sha256,
        cache_key_sha256=prediction.cache.key.key_sha256,
        estimator_record_sha256=prediction.estimator_record.record_sha256,
        counter_trace_sha256=(
            prediction.estimator_record.counter_trace_sha256
        ),
    )


def _fresh_stream(predictor: PriorPredictor, seed: int) -> EstimatorStream:
    identity = getattr(predictor, "estimator_identity", None)
    if type(identity) is not EstimatorIdentity:
        raise _UnauditablePrefixCheck(
            "predictor lacks an exact EstimatorIdentity"
        )
    return EstimatorStream.create(
        stream_seed=seed,
        estimator_identity=identity,
    )


def _prefix(
    case: DynamicPrefixCase, vocabulary: VocabularyIdentity
) -> CausalPrefix:
    return CausalPrefix.create(
        receiver_t=case.receiver_t,
        vocabulary=vocabulary,
        token_ids=torch.tensor(case.shared_prefix, dtype=torch.int64),
    )


def _cold_identity(
    predictor: PriorPredictor,
    case: DynamicPrefixCase,
    *,
    stream_seed: int,
) -> _PredictionObservation:
    vocabulary = getattr(predictor, "vocabulary")
    prefix = _prefix(case, vocabulary)
    stream = _fresh_stream(predictor, stream_seed)
    prediction = predictor.next_token_log_probs(prefix, stream, None)
    return _validate_prediction_identity(
        prediction,
        prefix=prefix,
        predictor=predictor,
        stream=stream,
    )


def _warm_identity(
    predictor: PriorPredictor,
    case: DynamicPrefixCase,
    *,
    stream_seed: int,
) -> _PredictionObservation:
    vocabulary = getattr(predictor, "vocabulary")
    full = _prefix(case, vocabulary)
    stream = _fresh_stream(predictor, stream_seed)
    if case.receiver_t == 1:
        seed_prediction = predictor.next_token_log_probs(full, stream, None)
    else:
        shorter_case = DynamicPrefixCase.create(
            ordinal=case.ordinal,
            receiver_t=case.receiver_t - 1,
            shared_prefix=case.shared_prefix[:-1],
            left_tail=(case.shared_prefix[-1],),
            right_tail=(case.shared_prefix[-1],),
        )
        shorter = _prefix(shorter_case, vocabulary)
        seed_prediction = predictor.next_token_log_probs(
            shorter, stream, None
        )
    prediction = predictor.next_token_log_probs(
        full, stream, seed_prediction.cache
    )
    return _validate_prediction_identity(
        prediction,
        prefix=full,
        predictor=predictor,
        stream=stream,
    )


def _signature_and_identity_assessment(
    key: PrefixCaseKey,
    predictor: PriorPredictor,
    arm_config: ArmConfig,
) -> tuple[EvidenceStatus, str] | None:
    try:
        parameters = tuple(
            inspect.signature(predictor.next_token_log_probs).parameters
        )
    except (TypeError, ValueError):
        return EvidenceStatus.INCONCLUSIVE, "predictor signature is unresolved"
    if parameters != ("prefix_tokens", "estimator_rng", "cache"):
        return EvidenceStatus.FAIL, "predictor signature exposes a noncausal argument"
    vocabulary = getattr(predictor, "vocabulary", None)
    estimator_spec = getattr(predictor, "estimator_spec", None)
    proposal = getattr(predictor, "proposal", None)
    model = getattr(proposal, "model", None)
    estimator_identity = getattr(predictor, "estimator_identity", None)
    required_values = {
        "vocabulary": vocabulary,
        "estimator_spec": estimator_spec,
        "proposal": proposal,
        "model": model,
        "estimator_identity": estimator_identity,
        "model_state_sha256": getattr(predictor, "model_state_sha256", None),
        "proposal_identity_sha256": getattr(
            predictor, "proposal_identity_sha256", None
        ),
    }
    if any(value is None for value in required_values.values()):
        missing = tuple(
            name for name, value in required_values.items() if value is None
        )
        return (
            EvidenceStatus.INCONCLUSIVE,
            f"predictor identity fields are absent: {','.join(missing)}",
        )
    if (
        type(vocabulary) is not VocabularyIdentity
        or type(estimator_spec) is not EstimatorSpec
        or type(estimator_identity) is not EstimatorIdentity
    ):
        return EvidenceStatus.FAIL, "predictor identity record types are invalid"
    try:
        vocabulary.__post_init__()
        estimator_spec.__post_init__()
        estimator_identity.__post_init__()
    except ValueError as exc:
        return EvidenceStatus.FAIL, f"predictor identity integrity failed: {exc}"
    expected_family_sha256 = arm_model_family_sha256(arm_config)
    if arm_config.latent_enabled:
        model_matches = (
            type(model) is LatentLanguageArmModel
            and model.arm is arm_config.arm
            and model.source_mode == arm_config.source_mode
            and model.map_mode == arm_config.map_mode
            and model.state_channel_enabled is arm_config.state_channel_enabled
            and model.model_channel_enabled is arm_config.model_channel_enabled
        )
    else:
        if arm_config.arm is ArmId.A0:
            model_matches = (
                type(model) is H6CausalTransformer
                and model.family_label == "a0_causal_transformer"
            )
        else:
            model_matches = (
                type(model) is MeanPooledPrefixFloor
                and model.family_label == "a5_mean_pooled_nolatent_floor"
            )
    if (
        not model_matches
        or key.arm is not arm_config.arm
        or arm_config.vocabulary != vocabulary
        or key.predictor_config_sha256 != arm_config.config_sha256
        or key.predictor_config_sha256
        != getattr(predictor, "predictor_config_sha256", None)
        or key.estimator_sha256
        != estimator_spec.estimator_sha256
        or key.model_family_sha256
        != getattr(predictor, "model_family_sha256", None)
        or key.model_family_sha256 != expected_family_sha256
        or getattr(proposal, "model_family_sha256", None)
        != expected_family_sha256
        or key.vocabulary_sha256
        != vocabulary_identity_sha256(vocabulary)
        or key.data_safety_sha256
        != getattr(predictor, "data_safety_sha256", None)
    ):
        return EvidenceStatus.FAIL, "PrefixCaseKey does not match the exact arm"
    return None


def _expected_source_banks(arm_config: ArmConfig) -> tuple[str, ...]:
    if arm_config.source_mode != "categorical":
        return ()
    return (
        ("state", "model")
        if arm_config.model_channel_enabled
        else ("state",)
    )


def _source_mask_assessment(
    observations: tuple[SourceMaskObservation, ...] | None,
    *,
    all_invalid_observation: AllInvalidSourceObservation | None,
    arm_config: ArmConfig,
    cases: tuple[DynamicPrefixCase, ...],
    plan: DynamicExecutionPlan,
) -> tuple[int, int, str | None, tuple[str, ...], str]:
    banks = _expected_source_banks(arm_config)
    supplied = () if observations is None else observations
    if type(supplied) is not tuple or any(
        type(item) is not SourceMaskObservation for item in supplied
    ):
        raise ValueError("source-mask observations must be an exact tuple")
    mask_manifest_sha256 = _owned_hash(
        "vfe4.h6.source-mask-manifest.v1",
        {
            "observations": tuple(
                observation.observation_sha256 for observation in supplied
            ),
            "all_invalid": (
                None
                if all_invalid_observation is None
                else all_invalid_observation.observation_sha256
            ),
        },
    )
    if not banks:
        if supplied or all_invalid_observation is not None:
            return (
                1,
                0,
                "source-mask evidence supplied for a noncategorical arm",
                (),
                mask_manifest_sha256,
            )
        return 0, 0, None, (), mask_manifest_sha256
    case_by_sha = {case.case_sha256: case for case in cases}
    expected_selected = {
        (case.case_sha256, bank) for case in cases for bank in banks
    }
    observed_keys: set[tuple[str, str]] = set()
    completed = 0
    violations = 0
    first: str | None = None
    for observation in supplied:
        observation.__post_init__()
        observation_key = (observation.case_sha256, observation.bank)
        case = case_by_sha.get(observation.case_sha256)
        if observation_key in observed_keys:
            violations += 1
            first = first or f"duplicate-mask:{observation.case_sha256}:{observation.bank}"
            continue
        observed_keys.add(observation_key)
        if (
            observation_key not in expected_selected
            or case is None
            or observation.config_sha256 != arm_config.config_sha256
            or observation.receiver_t != case.receiver_t
            or observation.declared_parents
            != tuple(range(observation.receiver_t))
        ):
            violations += 1
            first = first or (
                f"mask-identity:{observation.case_sha256}:{observation.bank}"
            )
            continue
        values = torch.tensor(
            observation.log_probabilities, dtype=torch.float64
        )
        if (
            not bool(torch.isfinite(values).all())
            or float(torch.logsumexp(values, dim=0).item()) != 0.0
        ):
            violations += 1
            first = first or (
                f"post-softmax-mask:{observation.case_sha256}:{observation.bank}"
            )
            continue
        completed += 1
    obligations: list[str] = []
    missing = expected_selected - observed_keys
    if missing:
        obligations.append("selected source-mask observations are incomplete")
    if all_invalid_observation is None:
        obligations.append("all-invalid source-row rejection is unobserved")
    else:
        all_invalid_observation.__post_init__()
        if all_invalid_observation.config_sha256 != arm_config.config_sha256:
            violations += 1
            first = first or "all-invalid source observation has stale config"
        elif all_invalid_observation.outcome != "rejected":
            violations += 1
            first = first or "all-invalid source row returned a fallback"
    expected_full = plan.full_expected_count * len(banks)
    if completed < expected_full:
        obligations.append("complete source-mask inventory is deferred")
    return (
        violations,
        completed,
        first,
        tuple(dict.fromkeys(obligations)),
        mask_manifest_sha256,
    )


def run_dynamic_prefix_checks(
    *,
    key: PrefixCaseKey,
    predictor: PriorPredictor,
    arm_config: ArmConfig,
    cases: Iterable[DynamicPrefixCase],
    plan: DynamicExecutionPlan,
    stream_seed: int,
    perturbations: FrozenValidationPerturbations | None = None,
    source_mask_observations: tuple[SourceMaskObservation, ...] | None = None,
    all_invalid_observation: AllInvalidSourceObservation | None = None,
    pair_side_harness: PairSideHarness | None = None,
) -> DynamicPrefixReport:
    """Run a bounded family subset; only a later two-family combiner can PASS."""

    if type(key) is not PrefixCaseKey:
        raise ValueError("key must be an exact PrefixCaseKey")
    key.__post_init__()
    if not isinstance(predictor, PriorPredictor):
        raise ValueError("predictor must implement PriorPredictor")
    if type(arm_config) is not ArmConfig:
        raise ValueError("arm_config must be an exact ArmConfig")
    arm_config.__post_init__()
    literal_arm_semantic_payload(arm_config)
    if type(plan) is not DynamicExecutionPlan:
        raise ValueError("plan must be an exact DynamicExecutionPlan")
    plan.__post_init__()
    if type(stream_seed) is not int or not 0 <= stream_seed < 2**64:
        raise ValueError("stream_seed must be an unsigned 64-bit integer")
    if pair_side_harness is not None:
        if type(pair_side_harness) is not PairSideHarness:
            raise ValueError("pair_side_harness must be exact when supplied")
        if pair_side_harness.trace_count:
            raise ValueError("pair_side_harness must be fresh for each report")
    try:
        iterator = iter(cases)
    except TypeError as exc:
        raise ValueError("cases must be iterable") from exc
    if plan.mode == "focused_subset":
        owned_cases = tuple(
            itertools.islice(iterator, MAX_FOCUSED_CASES + 1)
        )
        if len(owned_cases) > MAX_FOCUSED_CASES:
            raise ValueError(
                f"focused runs are hard-capped at {MAX_FOCUSED_CASES} cases"
            )
    else:
        owned_cases = tuple(iterator)
    if (
        not owned_cases
        or any(type(case) is not DynamicPrefixCase for case in owned_cases)
        or len({case.case_sha256 for case in owned_cases}) != len(owned_cases)
    ):
        raise ValueError("dynamic cases must be unique exact records")
    for case in owned_cases:
        case.__post_init__()

    identity_assessment = _signature_and_identity_assessment(
        key, predictor, arm_config
    )
    identity_status = (
        EvidenceStatus.PASS
        if identity_assessment is None
        else identity_assessment[0]
    )
    identity_message = (
        None if identity_assessment is None else identity_assessment[1]
    )
    dynamic_violations = 0
    cache_violations = 0
    data_violations = 0
    dynamic_first: str | None = None
    cache_first: str | None = None
    data_first: str | None = None
    dynamic_obligations: list[str] = []
    cache_obligations: list[str] = []
    if pair_side_harness is None:
        dynamic_obligations.append(
            "pair-side tail-state harness is absent"
        )
    pair_bindings: dict[str, list[str]] = {}

    def bind_pair(
        case: DynamicPrefixCase,
        side: Literal["left", "right"],
        tail: tuple[int, ...],
    ) -> str | None:
        if pair_side_harness is None:
            return None
        binding = pair_side_harness.bind(
            case_sha256=case.case_sha256,
            side=side,
            tail=tail,
        )
        pair_side_harness.assert_current(
            case_sha256=case.case_sha256,
            side=side,
            tail=tail,
            binding_sha256=binding,
        )
        pair_bindings.setdefault(case.case_sha256, []).append(binding)
        return binding

    def assert_pair_unchanged(
        case: DynamicPrefixCase,
        side: Literal["left", "right"],
        tail: tuple[int, ...],
        binding: str | None,
    ) -> None:
        if pair_side_harness is not None and binding is not None:
            pair_side_harness.assert_current(
                case_sha256=case.case_sha256,
                side=side,
                tail=tail,
                binding_sha256=binding,
            )
    mode_by_case: dict[
        str,
        tuple[
            _PredictionObservation,
            _PredictionObservation,
            _PredictionObservation,
        ],
    ] = {}
    complete_modes: dict[
        str,
        tuple[
            _PredictionObservation,
            _PredictionObservation,
            _PredictionObservation,
            _PredictionObservation,
        ],
    ] = {}
    completed_by_position = [0] * len(plan.expected_by_position)
    if identity_status is EvidenceStatus.PASS:
        vocabulary = getattr(predictor, "vocabulary")
        for case in owned_cases:
            position_index = (
                case.receiver_t - 1
                if plan.case_family == "small"
                else 0
            )
            if not 0 <= position_index < len(completed_by_position):
                data_violations += 1
                data_first = data_first or case.case_sha256
                continue
            if any(
                token >= vocabulary.size
                for token in (
                    case.shared_prefix + case.left_tail + case.right_tail
                )
            ):
                data_violations += 1
                data_first = data_first or case.case_sha256
                continue
            try:
                left_binding = bind_pair(case, "left", case.left_tail)
                left = _cold_identity(
                    predictor, case, stream_seed=stream_seed
                )
                assert_pair_unchanged(
                    case, "left", case.left_tail, left_binding
                )
                right_binding = bind_pair(case, "right", case.right_tail)
                right = _cold_identity(
                    predictor, case, stream_seed=stream_seed
                )
                assert_pair_unchanged(
                    case, "right", case.right_tail, right_binding
                )
                warm_binding = bind_pair(case, "left", case.left_tail)
                warm = _warm_identity(
                    predictor, case, stream_seed=stream_seed
                )
                assert_pair_unchanged(
                    case, "left", case.left_tail, warm_binding
                )
                if left.tensor != right.tensor:
                    dynamic_violations += 1
                    dynamic_first = dynamic_first or (
                        f"{case.case_sha256}:pair-side prediction bytes differ"
                    )
                if left.cache_payload() != right.cache_payload():
                    cache_violations += 1
                    cache_first = cache_first or (
                        f"{case.case_sha256}:pair-side cache identity differs"
                    )
                if left != warm:
                    cache_violations += 1
                    cache_first = cache_first or (
                        f"{case.case_sha256}:warm cache replay differs"
                    )
                mode_by_case[case.case_sha256] = (left, right, warm)
            except _WitnessedCacheViolation as exc:
                cache_violations += 1
                cache_first = cache_first or (
                    f"{case.case_sha256}:{type(exc).__name__}:{exc}"
                )
            except _WitnessedPredictionViolation as exc:
                dynamic_violations += 1
                dynamic_first = dynamic_first or (
                    f"{case.case_sha256}:{type(exc).__name__}:{exc}"
                )
            except (AttributeError, NotImplementedError) as exc:
                message = f"{case.case_sha256}:{type(exc).__name__}:{exc}"
                dynamic_obligations.append(message)
                cache_obligations.append(message)
            except (KeyError, RuntimeError, ValueError) as exc:
                dynamic_violations += 1
                dynamic_first = dynamic_first or (
                    f"{case.case_sha256}:{type(exc).__name__}:{exc}"
                )
        for case in reversed(owned_cases):
            modes = mode_by_case.get(case.case_sha256)
            if modes is None:
                continue
            try:
                reverse_binding = bind_pair(case, "left", case.left_tail)
                reverse_identity = _cold_identity(
                    predictor, case, stream_seed=stream_seed
                )
                assert_pair_unchanged(
                    case, "left", case.left_tail, reverse_binding
                )
                if reverse_identity != modes[0]:
                    cache_violations += 1
                    cache_first = cache_first or (
                        f"{case.case_sha256}:reverse-order rebuild differs"
                    )
                complete_modes[case.case_sha256] = (*modes, reverse_identity)
                position_index = (
                    case.receiver_t - 1
                    if plan.case_family == "small"
                    else 0
                )
                completed_by_position[position_index] += 1
            except _WitnessedCacheViolation as exc:
                cache_violations += 1
                cache_first = cache_first or (
                    f"{case.case_sha256}:reverse:{type(exc).__name__}:{exc}"
                )
            except _WitnessedPredictionViolation as exc:
                dynamic_violations += 1
                dynamic_first = dynamic_first or (
                    f"{case.case_sha256}:reverse:{type(exc).__name__}:{exc}"
                )
            except (AttributeError, NotImplementedError) as exc:
                message = (
                    f"{case.case_sha256}:reverse:{type(exc).__name__}:{exc}"
                )
                dynamic_obligations.append(message)
                cache_obligations.append(message)
            except (KeyError, RuntimeError, ValueError) as exc:
                dynamic_violations += 1
                dynamic_first = dynamic_first or (
                    f"{case.case_sha256}:reverse:{type(exc).__name__}:{exc}"
                )
    else:
        reason = identity_message or "predictor identity is unresolved"
        dynamic_obligations.append(
            f"dynamic cases not executed because {reason}"
        )
        cache_obligations.append(
            f"cache cases not executed because {reason}"
        )

    (
        mask_violations,
        mask_completed,
        mask_first,
        mask_obligations,
        mask_manifest_sha,
    ) = _source_mask_assessment(
        source_mask_observations,
        all_invalid_observation=all_invalid_observation,
        arm_config=arm_config,
        cases=owned_cases,
        plan=plan,
    )
    inventory_complete = (
        tuple(completed_by_position) == plan.expected_by_position
        and len(owned_cases) == plan.full_expected_count
    )
    data_obligations: list[str] = []
    if plan.case_family == "validation":
        if perturbations is None:
            data_obligations.append("validation perturbation fixture is absent")
            data_completed = 0
        else:
            perturbations.__post_init__()
            data_completed = min(
                perturbations.materialized_count, plan.full_expected_count
            )
            if (
                perturbations.vocabulary
                != getattr(predictor, "vocabulary", None)
                or tuple(
                    item.case_sha256 for item in perturbations.dynamic_cases
                )
                != tuple(item.case_sha256 for item in owned_cases)
            ):
                data_violations += 1
                data_first = data_first or "validation perturbation identity mismatch"
            if (
                perturbations.materialization != "authorized_full"
                or not perturbations.source_fixture_verified
            ):
                data_obligations.append(
                    "validation perturbation source/full inventory is incomplete"
                )
    else:
        data_completed = 0
        if perturbations is not None:
            data_violations += 1
            data_first = data_first or (
                "validation perturbations were supplied to the small family"
            )

    completed_total = sum(completed_by_position)
    if completed_total < len(owned_cases):
        dynamic_obligations.append("one or more selected dynamic cases are unaudited")
        cache_obligations.append("one or more selected cache cases are unaudited")
    inventory_obligations = (
        ()
        if inventory_complete
        else ("complete dynamic case inventory is deferred",)
    )
    identity_obligations = (
        (identity_message,)
        if identity_status is EvidenceStatus.INCONCLUSIVE
        and identity_message is not None
        else ()
    )
    identity_first = (
        identity_message
        if identity_status is EvidenceStatus.FAIL
        else None
    )
    dynamic_status = (
        EvidenceStatus.FAIL
        if dynamic_violations
        else EvidenceStatus.INCONCLUSIVE
        if dynamic_obligations
        else EvidenceStatus.PASS
    )
    cache_status = (
        EvidenceStatus.FAIL
        if cache_violations
        else EvidenceStatus.INCONCLUSIVE
        if cache_obligations
        else EvidenceStatus.PASS
    )
    mask_status = (
        EvidenceStatus.FAIL
        if mask_violations
        else EvidenceStatus.INCONCLUSIVE
        if mask_obligations
        else EvidenceStatus.PASS
    )
    data_status = (
        EvidenceStatus.FAIL
        if data_violations
        else EvidenceStatus.INCONCLUSIVE
        if data_obligations
        else EvidenceStatus.PASS
    )
    checks = (
        DynamicCheckResult.create(
            name="signature_and_identity",
            status=identity_status,
            expected_count=1,
            completed_count=(
                0 if identity_status is EvidenceStatus.INCONCLUSIVE else 1
            ),
            violation_count=(1 if identity_status is EvidenceStatus.FAIL else 0),
            first_counterexample=identity_first,
            obligations=identity_obligations,
        ),
        DynamicCheckResult.create(
            name="dynamic_target_suffix_leakage",
            status=dynamic_status,
            expected_count=len(owned_cases),
            completed_count=completed_total,
            violation_count=dynamic_violations,
            first_counterexample=dynamic_first,
            obligations=tuple(dict.fromkeys(dynamic_obligations)),
        ),
        DynamicCheckResult.create(
            name="cache_identity",
            status=cache_status,
            expected_count=len(owned_cases),
            completed_count=completed_total,
            violation_count=cache_violations,
            first_counterexample=cache_first,
            obligations=tuple(dict.fromkeys(cache_obligations)),
        ),
        DynamicCheckResult.create(
            name="source_mask",
            status=mask_status,
            expected_count=(
                plan.full_expected_count * len(_expected_source_banks(arm_config))
            ),
            completed_count=mask_completed,
            violation_count=mask_violations,
            first_counterexample=mask_first,
            obligations=mask_obligations,
        ),
        DynamicCheckResult.create(
            name="case_inventory",
            status=(
                EvidenceStatus.PASS
                if inventory_complete
                else EvidenceStatus.INCONCLUSIVE
            ),
            expected_count=plan.full_expected_count,
            completed_count=completed_total,
            obligations=inventory_obligations,
        ),
        DynamicCheckResult.create(
            name="validation_data_safety",
            status=data_status,
            expected_count=(
                plan.full_expected_count
                if plan.case_family == "validation"
                else 0
            ),
            completed_count=data_completed,
            violation_count=data_violations,
            first_counterexample=data_first,
            obligations=tuple(dict.fromkeys(data_obligations)),
        ),
    )
    case_result_rows = tuple(
        {
            "case_sha256": case.case_sha256,
            "pair_binding_sha256s": tuple(
                pair_bindings.get(case.case_sha256, ())
            ),
            "modes": {
                name: observation.tensor_payload()
                for name, observation in zip(
                    ("left", "right", "warm", "reverse"),
                    complete_modes[case.case_sha256],
                    strict=True,
                )
            },
        }
        for case in owned_cases
        if case.case_sha256 in complete_modes
    )
    cache_rows = tuple(
        {
            "case_sha256": case.case_sha256,
            "modes": {
                name: observation.cache_payload()
                for name, observation in zip(
                    ("left", "right", "warm", "reverse"),
                    complete_modes[case.case_sha256],
                    strict=True,
                )
            },
        }
        for case in owned_cases
        if case.case_sha256 in complete_modes
    )
    case_result_manifest_sha = _owned_hash(
        "vfe4.h6.dynamic-case-result-manifest.v1", case_result_rows
    )
    cache_manifest_sha = _owned_hash(
        "vfe4.h6.dynamic-cache-manifest.v1", cache_rows
    )
    pair_harness_manifest_sha = (
        pair_side_harness.manifest_sha256
        if pair_side_harness is not None
        else _owned_hash("vfe4.h6.pair-side-harness-manifest.v1", ())
    )
    manifest_sha = (
        _owned_hash(
            "vfe4.h6.dynamic-case-manifest.v1",
            tuple(case.case_sha256 for case in owned_cases),
        )
        if inventory_complete
        else None
    )
    unresolved_diagnostics = tuple(
        dict.fromkeys(
            item
            for check in checks
            for item in check.obligations
        )
    )
    witnessed_failure = any(
        check.status is EvidenceStatus.FAIL for check in checks
    )
    top_obligations: list[str] = []
    if not witnessed_failure:
        if plan.mode != "authorized_full":
            top_obligations.append("focused subset is not H6-Prefix evidence")
        top_obligations.append(
            "companion 4,096 validation-family report is required"
            if plan.case_family == "small"
            else "companion 9,720 small-family report is required"
        )
    status = (
        EvidenceStatus.FAIL
        if witnessed_failure
        else EvidenceStatus.INCONCLUSIVE
        if top_obligations
        or any(check.status is EvidenceStatus.INCONCLUSIVE for check in checks)
        else EvidenceStatus.PASS
    )
    first_counterexample = next(
        (
            check.first_counterexample
            for check in checks
            if check.status is EvidenceStatus.FAIL
        ),
        None,
    )
    estimator_identity = getattr(predictor, "estimator_identity", None)
    report_values = {
        "schema_version": "h6-dynamic-prefix-report-v1",
        "key": key,
        "execution_plan_sha256": plan.plan_sha256,
        "model_state_sha256": getattr(predictor, "model_state_sha256", None),
        "proposal_identity_sha256": getattr(
            predictor, "proposal_identity_sha256", None
        ),
        "estimator_semantic_sha256": (
            estimator_identity.semantic_sha256
            if type(estimator_identity) is EstimatorIdentity
            else None
        ),
        "estimator_artifact_bytes_sha256": (
            estimator_identity.artifact_bytes_sha256
            if type(estimator_identity) is EstimatorIdentity
            else None
        ),
        "stream_seed": stream_seed,
        "completed_by_position": tuple(completed_by_position),
        "checks": checks,
        "status": status,
        "obligations": tuple(top_obligations),
        "unresolved_diagnostics": unresolved_diagnostics,
        "first_counterexample": first_counterexample,
        "case_result_manifest_sha256": case_result_manifest_sha,
        "cache_manifest_sha256": cache_manifest_sha,
        "pair_harness_manifest_sha256": pair_harness_manifest_sha,
        "mask_manifest_sha256": mask_manifest_sha,
        "complete_case_manifest_sha256": manifest_sha,
    }
    canonical_report_payload = {
        **report_values,
        "key": key.canonical_payload(),
        "checks": tuple(check.canonical_payload() for check in checks),
        "status": status.value,
    }
    report = DynamicPrefixReport(
        **report_values,
        report_sha256=_owned_hash(
            "vfe4.h6.dynamic-prefix-report.v1", canonical_report_payload
        ),
    )
    report.__post_init__()
    return report


__all__ = [
    "AllInvalidSourceObservation",
    "DynamicCheckResult",
    "DynamicExecutionPlan",
    "DynamicPrefixCase",
    "DynamicPrefixReport",
    "FrozenValidationPerturbations",
    "MAX_FOCUSED_CASES",
    "PairSideHarness",
    "PERTURBATION_FIXTURE_PATH",
    "PERTURBATION_GENERATOR_SEED",
    "SMALL_EXPECTED_BY_POSITION",
    "SMALL_EXPECTED_TOTAL",
    "SourceMaskObservation",
    "TensorByteIdentity",
    "VALIDATION_EXPECTED_TOTAL",
    "ValidationPerturbationRecord",
    "load_frozen_validation_perturbations",
    "observe_all_invalid_source_rejection",
    "run_dynamic_prefix_checks",
]
