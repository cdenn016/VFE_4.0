"""Independent deterministic oracles for the H6 prefix-safety inventories.

The helpers in this module intentionally import neither :mod:`vfe4` nor
``torch``.  The exhaustive small-fixture enumeration is reconstructed by
closed-form integer arithmetic, while WikiText-2 perturbations are derived
directly from the identity-bound validation-safety fixture bytes.

Nothing in this module runs an inventory at import time.  The exhaustive
9,720-case iterator and the 4,096-record perturbation materialization are
entered only when a caller explicitly iterates or requests them.  Both paths
also accept deterministic subsets for bounded source tests.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any


ORDERED_TAIL_GENERATOR_VERSION = "h6-prefix-ordered-tail-v1"
VALIDATION_PERTURBATION_SCHEMA_VERSION = "h6-validation-perturbations-v1"
VALIDATION_PERTURBATION_GENERATOR_VERSION = "h6-validation-perturbations-v1"
VALIDATION_PERTURBATION_SEED = 2026072197
VALIDATION_PERTURBATION_COUNT = 4096

SMALL_PREFIX_VOCABULARY_SIZE = 3
SMALL_PREFIX_HORIZON = 4
SMALL_PREFIX_COUNTS_BY_RECEIVER = (6561, 2187, 729, 243)
SMALL_PREFIX_TOTAL_COUNT = 9720

_LOWER_HEX = frozenset("0123456789abcdef")
_VALIDATION_FIXTURE_DOMAIN = b"VFE4-H6-VALIDATION-SAFETY-FIXTURE-V1\x00"
_PERTURBATION_COUNTER_DOMAIN = b"VFE4-H6-VALIDATION-PERTURBATION-COUNTER-V1\x00"
_ORDERED_TAIL_CASE_DOMAIN = "vfe4.h6.ordered-tail-pair.v1"
_PERTURBATION_CASE_DOMAIN = "vfe4.h6.validation-perturbation-case.v1"
_PERTURBATION_MANIFEST_DOMAIN = (
    "vfe4.h6.validation-perturbations-manifest.v1"
)
_VOCABULARY_IDENTITY_DOMAIN = "vfe4.h6.vocabulary-identity.v1"
_VALIDATION_ROW_BYTES = 8 + 2 + 33 * 2

_PERTURBATION_ROOT_FIELDS = frozenset(
    {
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
)
_VOCABULARY_FIELDS = frozenset(
    {
        "vocabulary_id",
        "size",
        "tokenizer_spec_sha256",
        "vocabulary_sha256",
    }
)
_PERTURBATION_RECORD_FIELDS = frozenset(
    {
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
)


def _require_int(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return value


def _require_nonempty(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _owned_hash(domain: str, payload: object) -> str:
    return hashlib.sha256(
        domain.encode("ascii") + b"\x00" + _canonical_json_bytes(payload)
    ).hexdigest()


def _mapping(value: object, fields: frozenset[str], name: str) -> dict[str, Any]:
    if type(value) is not dict or frozenset(value) != fields:
        raise ValueError(f"{name} fields must equal {sorted(fields)!r}")
    return value


def _sequence(value: object, name: str) -> list[Any]:
    if type(value) is not list:
        raise ValueError(f"{name} must be a list")
    return value


def _json_without_duplicate_fields(data: bytes) -> object:
    def checked_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field: {key}")
            result[key] = value
        return result

    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=checked_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"validation perturbations are not canonical UTF-8 JSON: {exc}"
        ) from exc


def _tokens(
    value: object,
    *,
    vocabulary_size: int,
    expected_length: int,
    name: str,
) -> tuple[int, ...]:
    sequence = _sequence(value, name)
    if len(sequence) != expected_length:
        raise ValueError(f"{name} must have length {expected_length}")
    result = tuple(sequence)
    if any(
        type(token_id) is not int
        or token_id < 0
        or token_id >= vocabulary_size
        for token_id in result
    ):
        raise ValueError(f"{name} contains a token outside the vocabulary")
    return result


def _decode_base_v(index: int, *, width: int, vocabulary_size: int) -> tuple[int, ...]:
    _require_int(index, "index")
    _require_int(width, "width")
    _require_int(vocabulary_size, "vocabulary_size", minimum=1)
    limit = vocabulary_size**width
    if index >= limit:
        raise ValueError("base-v index exceeds the requested width")
    values = [0] * width
    remainder = index
    for offset in range(width - 1, -1, -1):
        remainder, values[offset] = divmod(remainder, vocabulary_size)
    return tuple(values)


def _encode_base_v(values: tuple[int, ...], *, vocabulary_size: int) -> int:
    result = 0
    for value in values:
        if type(value) is not int or not 0 <= value < vocabulary_size:
            raise ValueError("base-v digits fall outside the vocabulary")
        result = result * vocabulary_size + value
    return result


@dataclass(frozen=True)
class OrderedTailPositionInventory:
    """Closed-form count and offset for one one-indexed receiver position."""

    receiver_t: int
    prefix_length: int
    tail_length: int
    prefix_count: int
    tail_count: int
    comparison_count: int
    global_offset: int

    def __post_init__(self) -> None:
        _require_int(self.receiver_t, "receiver_t", minimum=1)
        _require_int(self.prefix_length, "prefix_length")
        _require_int(self.tail_length, "tail_length", minimum=1)
        _require_int(self.prefix_count, "prefix_count", minimum=1)
        _require_int(self.tail_count, "tail_count", minimum=1)
        _require_int(self.comparison_count, "comparison_count", minimum=1)
        _require_int(self.global_offset, "global_offset")


@dataclass(frozen=True)
class OrderedTailPairInventory:
    """Lazy enumeration metadata for the exhaustive target/suffix comparison."""

    generator_version: str
    vocabulary_size: int
    horizon: int
    positions: tuple[OrderedTailPositionInventory, ...]
    total_count: int

    def __post_init__(self) -> None:
        if self.generator_version != ORDERED_TAIL_GENERATOR_VERSION:
            raise ValueError("unsupported ordered-tail generator version")
        _require_int(self.vocabulary_size, "vocabulary_size", minimum=1)
        _require_int(self.horizon, "horizon", minimum=1)
        if (
            type(self.positions) is not tuple
            or len(self.positions) != self.horizon
            or not all(
                type(position) is OrderedTailPositionInventory
                for position in self.positions
            )
        ):
            raise ValueError("positions must cover every receiver exactly once")
        offset = 0
        for expected_receiver, position in enumerate(self.positions, start=1):
            if (
                position.receiver_t != expected_receiver
                or position.prefix_length != expected_receiver - 1
                or position.tail_length != self.horizon - expected_receiver + 1
                or position.prefix_count
                != self.vocabulary_size**position.prefix_length
                or position.tail_count
                != self.vocabulary_size**position.tail_length
                or position.comparison_count
                != position.prefix_count * position.tail_count**2
                or position.global_offset != offset
            ):
                raise ValueError("ordered-tail position inventory is inconsistent")
            offset += position.comparison_count
        if self.total_count != offset:
            raise ValueError("ordered-tail total does not match position counts")

    @property
    def counts_by_receiver(self) -> tuple[int, ...]:
        return tuple(position.comparison_count for position in self.positions)

    @property
    def counts_by_position(self) -> tuple[int, ...]:
        return self.counts_by_receiver

    @property
    def total_case_count(self) -> int:
        return self.total_count


def ordered_tail_pair_inventory(
    *,
    vocabulary_size: int = SMALL_PREFIX_VOCABULARY_SIZE,
    horizon: int = SMALL_PREFIX_HORIZON,
) -> OrderedTailPairInventory:
    """Return the closed-form inventory without constructing any sequences."""

    _require_int(vocabulary_size, "vocabulary_size", minimum=1)
    _require_int(horizon, "horizon", minimum=1)
    positions: list[OrderedTailPositionInventory] = []
    offset = 0
    for receiver_t in range(1, horizon + 1):
        prefix_length = receiver_t - 1
        tail_length = horizon - receiver_t + 1
        prefix_count = vocabulary_size**prefix_length
        tail_count = vocabulary_size**tail_length
        comparison_count = prefix_count * tail_count**2
        positions.append(
            OrderedTailPositionInventory(
                receiver_t=receiver_t,
                prefix_length=prefix_length,
                tail_length=tail_length,
                prefix_count=prefix_count,
                tail_count=tail_count,
                comparison_count=comparison_count,
                global_offset=offset,
            )
        )
        offset += comparison_count
    return OrderedTailPairInventory(
        generator_version=ORDERED_TAIL_GENERATOR_VERSION,
        vocabulary_size=vocabulary_size,
        horizon=horizon,
        positions=tuple(positions),
        total_count=offset,
    )


def ordered_tail_pair_counts(
    *,
    vocabulary_size: int = SMALL_PREFIX_VOCABULARY_SIZE,
    horizon: int = SMALL_PREFIX_HORIZON,
) -> tuple[int, ...]:
    """Return the per-position comparison counts in one-indexed order."""

    return ordered_tail_pair_inventory(
        vocabulary_size=vocabulary_size,
        horizon=horizon,
    ).counts_by_receiver


@dataclass(frozen=True)
class OrderedTailPair:
    """One shared-prefix, ordered-left/right-tail comparison case."""

    generator_version: str
    case_id: str
    case_index: int
    position_case_index: int
    receiver_t: int
    vocabulary_size: int
    horizon: int
    prefix: tuple[int, ...]
    left_tail: tuple[int, ...]
    right_tail: tuple[int, ...]
    case_sha256: str

    def __post_init__(self) -> None:
        if self.generator_version != ORDERED_TAIL_GENERATOR_VERSION:
            raise ValueError("unsupported ordered-tail case generator")
        _require_nonempty(self.case_id, "case_id")
        _require_int(self.case_index, "case_index")
        _require_int(self.position_case_index, "position_case_index")
        _require_int(self.receiver_t, "receiver_t", minimum=1)
        _require_int(self.vocabulary_size, "vocabulary_size", minimum=1)
        _require_int(self.horizon, "horizon", minimum=1)
        if not 1 <= self.receiver_t <= self.horizon:
            raise ValueError("receiver_t exceeds the sequence horizon")
        if (
            type(self.prefix) is not tuple
            or type(self.left_tail) is not tuple
            or type(self.right_tail) is not tuple
            or len(self.prefix) != self.receiver_t - 1
            or len(self.left_tail) != self.horizon - self.receiver_t + 1
            or len(self.right_tail) != len(self.left_tail)
        ):
            raise ValueError("prefix/tail lengths do not match receiver and horizon")
        for name, values in (
            ("prefix", self.prefix),
            ("left_tail", self.left_tail),
            ("right_tail", self.right_tail),
        ):
            if any(
                type(value) is not int
                or value < 0
                or value >= self.vocabulary_size
                for value in values
            ):
                raise ValueError(f"{name} contains a token outside the vocabulary")

        inventory = ordered_tail_pair_inventory(
            vocabulary_size=self.vocabulary_size,
            horizon=self.horizon,
        )
        position = inventory.positions[self.receiver_t - 1]
        prefix_index = _encode_base_v(
            self.prefix,
            vocabulary_size=self.vocabulary_size,
        )
        left_index = _encode_base_v(
            self.left_tail,
            vocabulary_size=self.vocabulary_size,
        )
        right_index = _encode_base_v(
            self.right_tail,
            vocabulary_size=self.vocabulary_size,
        )
        expected_position_index = (
            prefix_index * position.tail_count**2
            + left_index * position.tail_count
            + right_index
        )
        expected_case_index = position.global_offset + expected_position_index
        expected_case_id = (
            f"h6-prefix-small-v1:t={self.receiver_t}:"
            f"case={expected_position_index}"
        )
        if (
            self.position_case_index != expected_position_index
            or self.case_index != expected_case_index
            or self.case_id != expected_case_id
        ):
            raise ValueError("ordered-tail indices or case_id are inconsistent")
        expected_sha256 = _owned_hash(
            _ORDERED_TAIL_CASE_DOMAIN,
            _ordered_tail_pair_payload(self, include_sha256=False),
        )
        if self.case_sha256 != expected_sha256:
            raise ValueError("case_sha256 does not match the ordered-tail case")

    @property
    def left_sequence(self) -> tuple[int, ...]:
        return self.prefix + self.left_tail

    @property
    def right_sequence(self) -> tuple[int, ...]:
        return self.prefix + self.right_tail

    @property
    def left_target(self) -> int:
        return self.left_tail[0]

    @property
    def right_target(self) -> int:
        return self.right_tail[0]

    @property
    def left_suffix(self) -> tuple[int, ...]:
        return self.left_tail[1:]

    @property
    def right_suffix(self) -> tuple[int, ...]:
        return self.right_tail[1:]


def _ordered_tail_pair_payload(
    case: OrderedTailPair,
    *,
    include_sha256: bool,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "generator_version": case.generator_version,
        "case_id": case.case_id,
        "case_index": case.case_index,
        "position_case_index": case.position_case_index,
        "receiver_t": case.receiver_t,
        "vocabulary_size": case.vocabulary_size,
        "horizon": case.horizon,
        "prefix": list(case.prefix),
        "left_tail": list(case.left_tail),
        "right_tail": list(case.right_tail),
    }
    if include_sha256:
        payload["case_sha256"] = case.case_sha256
    return payload


def _ordered_tail_pair_from_index(
    inventory: OrderedTailPairInventory,
    case_index: int,
) -> OrderedTailPair:
    if not 0 <= case_index < inventory.total_count:
        raise ValueError("case_index is outside the ordered-tail inventory")
    position = inventory.positions[-1]
    for candidate in inventory.positions:
        if case_index < candidate.global_offset + candidate.comparison_count:
            position = candidate
            break
    position_case_index = case_index - position.global_offset
    pairs_per_prefix = position.tail_count**2
    prefix_index, tail_pair_index = divmod(
        position_case_index,
        pairs_per_prefix,
    )
    left_index, right_index = divmod(
        tail_pair_index,
        position.tail_count,
    )
    prefix = _decode_base_v(
        prefix_index,
        width=position.prefix_length,
        vocabulary_size=inventory.vocabulary_size,
    )
    left_tail = _decode_base_v(
        left_index,
        width=position.tail_length,
        vocabulary_size=inventory.vocabulary_size,
    )
    right_tail = _decode_base_v(
        right_index,
        width=position.tail_length,
        vocabulary_size=inventory.vocabulary_size,
    )
    case_id = (
        f"h6-prefix-small-v1:t={position.receiver_t}:"
        f"case={position_case_index}"
    )
    payload = {
        "generator_version": ORDERED_TAIL_GENERATOR_VERSION,
        "case_id": case_id,
        "case_index": case_index,
        "position_case_index": position_case_index,
        "receiver_t": position.receiver_t,
        "vocabulary_size": inventory.vocabulary_size,
        "horizon": inventory.horizon,
        "prefix": list(prefix),
        "left_tail": list(left_tail),
        "right_tail": list(right_tail),
    }
    return OrderedTailPair(
        generator_version=ORDERED_TAIL_GENERATOR_VERSION,
        case_id=case_id,
        case_index=case_index,
        position_case_index=position_case_index,
        receiver_t=position.receiver_t,
        vocabulary_size=inventory.vocabulary_size,
        horizon=inventory.horizon,
        prefix=prefix,
        left_tail=left_tail,
        right_tail=right_tail,
        case_sha256=_owned_hash(_ORDERED_TAIL_CASE_DOMAIN, payload),
    )


def _normalize_positions(
    positions: Iterable[int] | None,
    *,
    horizon: int,
) -> tuple[int, ...]:
    if positions is None:
        return tuple(range(1, horizon + 1))
    try:
        selected = tuple(positions)
    except TypeError as exc:
        raise ValueError("positions must be an iterable of receiver integers") from exc
    if (
        not selected
        or any(type(position) is not int or not 1 <= position <= horizon for position in selected)
        or len(set(selected)) != len(selected)
    ):
        raise ValueError("positions must be unique one-indexed receivers in range")
    return selected


def _normalize_case_indices(
    case_indices: Iterable[int] | None,
    *,
    total_count: int,
    max_cases: int | None,
) -> tuple[int, ...] | None:
    if max_cases is not None:
        _require_int(max_cases, "max_cases", minimum=1)
    if case_indices is None:
        return None
    try:
        selected = tuple(case_indices)
    except TypeError as exc:
        raise ValueError("case_indices must be an iterable of integers") from exc
    if (
        not selected
        or any(
            type(case_index) is not int
            or case_index < 0
            or case_index >= total_count
            for case_index in selected
        )
        or len(set(selected)) != len(selected)
    ):
        raise ValueError("case_indices must be unique valid global case indices")
    if max_cases is not None:
        selected = selected[:max_cases]
    return selected


def enumerate_ordered_tail_pairs(
    *,
    vocabulary_size: int = SMALL_PREFIX_VOCABULARY_SIZE,
    horizon: int = SMALL_PREFIX_HORIZON,
    positions: Iterable[int] | None = None,
    case_indices: Iterable[int] | None = None,
    max_cases: int | None = None,
) -> Iterator[OrderedTailPair]:
    """Lazily enumerate exact ordered tail pairs, optionally by sparse indices.

    ``case_indices`` are global zero-based indices in the closed-form inventory
    and are emitted in caller order.  When they are absent, receiver positions
    are emitted in ``positions`` order.  ``max_cases`` bounds either traversal
    without first constructing the skipped Cartesian cases.
    """

    inventory = ordered_tail_pair_inventory(
        vocabulary_size=vocabulary_size,
        horizon=horizon,
    )
    selected_positions = _normalize_positions(positions, horizon=horizon)
    explicit_indices = _normalize_case_indices(
        case_indices,
        total_count=inventory.total_count,
        max_cases=max_cases,
    )
    allowed_positions = frozenset(selected_positions)

    if explicit_indices is not None:
        for case_index in explicit_indices:
            case = _ordered_tail_pair_from_index(inventory, case_index)
            if case.receiver_t not in allowed_positions:
                raise ValueError(
                    "case_indices includes a case outside the selected positions"
                )
            yield case
        return

    emitted = 0
    for receiver_t in selected_positions:
        position = inventory.positions[receiver_t - 1]
        for local_index in range(position.comparison_count):
            if max_cases is not None and emitted >= max_cases:
                return
            yield _ordered_tail_pair_from_index(
                inventory,
                position.global_offset + local_index,
            )
            emitted += 1


@dataclass(frozen=True)
class IndependentVocabularyIdentity:
    """Independent reconstruction of the production vocabulary identity."""

    vocabulary_id: str
    size: int
    tokenizer_spec_sha256: str
    vocabulary_sha256: str

    def __post_init__(self) -> None:
        _require_nonempty(self.vocabulary_id, "vocabulary_id")
        _require_int(self.size, "size", minimum=1)
        _require_sha256(self.tokenizer_spec_sha256, "tokenizer_spec_sha256")
        expected = _owned_hash(
            _VOCABULARY_IDENTITY_DOMAIN,
            {
                "vocabulary_id": self.vocabulary_id,
                "size": self.size,
                "tokenizer_spec_sha256": self.tokenizer_spec_sha256,
            },
        )
        if self.vocabulary_sha256 != expected:
            raise ValueError(
                "vocabulary_sha256 does not match the semantic vocabulary"
            )

    @classmethod
    def create(
        cls,
        *,
        vocabulary_id: str,
        size: int,
        tokenizer_spec_sha256: str,
    ) -> "IndependentVocabularyIdentity":
        payload = {
            "vocabulary_id": vocabulary_id,
            "size": size,
            "tokenizer_spec_sha256": tokenizer_spec_sha256,
        }
        return cls(
            vocabulary_id=vocabulary_id,
            size=size,
            tokenizer_spec_sha256=tokenizer_spec_sha256,
            vocabulary_sha256=_owned_hash(_VOCABULARY_IDENTITY_DOMAIN, payload),
        )


@dataclass(frozen=True)
class ValidationPerturbation:
    """One identity-bound validation sequence pair sharing ``x_<t``."""

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

    def __post_init__(self) -> None:
        _require_nonempty(self.case_id, "case_id")
        _require_int(self.case_index, "case_index")
        _require_int(self.source_window_index, "source_window_index")
        _require_int(self.window_start, "window_start")
        _require_int(self.real_target_count, "real_target_count", minimum=1)
        if self.real_target_count > 32:
            raise ValueError("real_target_count must not exceed 32")
        _require_int(self.receiver_t, "receiver_t", minimum=1)
        _require_int(self.vocabulary_size, "vocabulary_size", minimum=1)
        _require_sha256(self.vocabulary_sha256, "vocabulary_sha256")
        _require_sha256(self.validation_token_sha256, "validation_token_sha256")
        _require_sha256(
            self.validation_safety_fixture_sha256,
            "validation_safety_fixture_sha256",
        )
        if self.case_index != self.source_window_index:
            raise ValueError("each perturbation must retain its source-window index")
        if self.case_id != f"validation-window-{self.case_index:04d}":
            raise ValueError("case_id does not match the source-window index")
        if not 1 <= self.receiver_t <= self.real_target_count:
            raise ValueError("receiver_t exceeds the real target sequence")
        tail_length = self.real_target_count - self.receiver_t + 1
        if (
            type(self.prefix_token_ids) is not tuple
            or type(self.left_tail_token_ids) is not tuple
            or type(self.right_tail_token_ids) is not tuple
            or len(self.prefix_token_ids) != self.receiver_t - 1
            or len(self.left_tail_token_ids) != tail_length
            or len(self.right_tail_token_ids) != tail_length
        ):
            raise ValueError("perturbation prefix/tails have inconsistent lengths")
        for name, values in (
            ("prefix_token_ids", self.prefix_token_ids),
            ("left_tail_token_ids", self.left_tail_token_ids),
            ("right_tail_token_ids", self.right_tail_token_ids),
        ):
            if any(
                type(value) is not int
                or value < 0
                or value >= self.vocabulary_size
                for value in values
            ):
                raise ValueError(f"{name} contains a token outside the vocabulary")
        if any(
            left == right
            for left, right in zip(
                self.left_tail_token_ids,
                self.right_tail_token_ids,
            )
        ):
            raise ValueError(
                "left and right current-target/suffix coordinates must vary"
            )
        expected = _owned_hash(
            _PERTURBATION_CASE_DOMAIN,
            _validation_perturbation_payload(self, include_sha256=False),
        )
        if self.case_sha256 != expected:
            raise ValueError(
                "case_sha256 does not match the validation perturbation"
            )

    @property
    def left_sequence(self) -> tuple[int, ...]:
        return self.prefix_token_ids + self.left_tail_token_ids

    @property
    def right_sequence(self) -> tuple[int, ...]:
        return self.prefix_token_ids + self.right_tail_token_ids

    @property
    def left_target(self) -> int:
        return self.left_tail_token_ids[0]

    @property
    def right_target(self) -> int:
        return self.right_tail_token_ids[0]

    @property
    def left_suffix(self) -> tuple[int, ...]:
        return self.left_tail_token_ids[1:]

    @property
    def right_suffix(self) -> tuple[int, ...]:
        return self.right_tail_token_ids[1:]


def _validation_perturbation_payload(
    record: ValidationPerturbation,
    *,
    include_sha256: bool,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "case_id": record.case_id,
        "case_index": record.case_index,
        "source_window_index": record.source_window_index,
        "window_start": record.window_start,
        "real_target_count": record.real_target_count,
        "receiver_t": record.receiver_t,
        "prefix_token_ids": list(record.prefix_token_ids),
        "left_tail_token_ids": list(record.left_tail_token_ids),
        "right_tail_token_ids": list(record.right_tail_token_ids),
        "vocabulary_size": record.vocabulary_size,
        "vocabulary_sha256": record.vocabulary_sha256,
        "validation_token_sha256": record.validation_token_sha256,
        "validation_safety_fixture_sha256": (
            record.validation_safety_fixture_sha256
        ),
    }
    if include_sha256:
        payload["case_sha256"] = record.case_sha256
    return payload


@dataclass(frozen=True)
class FrozenValidationPerturbationSet:
    """Strict frozen case-file record with an optional selected traversal."""

    raw_sha256: str
    schema_version: str
    generator_version: str
    seed: int
    vocabulary: IndependentVocabularyIdentity
    validation_token_sha256: str
    validation_safety_fixture_sha256: str
    full_count: int
    materialized_count: int
    records: tuple[ValidationPerturbation, ...]
    manifest_sha256: str
    selected_records: tuple[ValidationPerturbation, ...]
    _canonical_bytes: bytes = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _require_sha256(self.raw_sha256, "raw_sha256")
        if (
            self.schema_version != VALIDATION_PERTURBATION_SCHEMA_VERSION
            or self.generator_version
            != VALIDATION_PERTURBATION_GENERATOR_VERSION
            or self.seed != VALIDATION_PERTURBATION_SEED
        ):
            raise ValueError("unsupported validation perturbation generator")
        if type(self.vocabulary) is not IndependentVocabularyIdentity:
            raise ValueError("vocabulary must be an independent identity")
        self.vocabulary.__post_init__()
        _require_sha256(self.validation_token_sha256, "validation_token_sha256")
        _require_sha256(
            self.validation_safety_fixture_sha256,
            "validation_safety_fixture_sha256",
        )
        if self.full_count != VALIDATION_PERTURBATION_COUNT:
            raise ValueError("full_count must be the frozen 4,096 inventory")
        _require_int(self.materialized_count, "materialized_count", minimum=1)
        if (
            type(self.records) is not tuple
            or len(self.records) != self.materialized_count
            or not all(type(record) is ValidationPerturbation for record in self.records)
        ):
            raise ValueError("records do not match materialized_count")
        indices = tuple(record.case_index for record in self.records)
        if (
            indices != tuple(sorted(indices))
            or len(set(indices)) != len(indices)
            or any(index >= self.full_count for index in indices)
        ):
            raise ValueError("materialized record indices must be unique and sorted")
        for record in self.records:
            record.__post_init__()
            if (
                record.vocabulary_size != self.vocabulary.size
                or record.vocabulary_sha256 != self.vocabulary.vocabulary_sha256
                or record.validation_token_sha256
                != self.validation_token_sha256
                or record.validation_safety_fixture_sha256
                != self.validation_safety_fixture_sha256
            ):
                raise ValueError("record identity differs from its frozen set")
        if (
            type(self.selected_records) is not tuple
            or not self.selected_records
            or not all(
                type(record) is ValidationPerturbation
                and record in self.records
                for record in self.selected_records
            )
            or len({record.case_index for record in self.selected_records})
            != len(self.selected_records)
        ):
            raise ValueError("selected_records must be a unique nonempty subset")
        _require_sha256(self.manifest_sha256, "manifest_sha256")
        if type(self._canonical_bytes) is not bytes:
            raise ValueError("canonical perturbation bytes must be immutable bytes")
        if hashlib.sha256(self._canonical_bytes).hexdigest() != self.raw_sha256:
            raise ValueError("raw_sha256 does not match the case-file bytes")
        payload = _frozen_perturbation_payload(
            schema_version=self.schema_version,
            generator_version=self.generator_version,
            seed=self.seed,
            vocabulary=self.vocabulary,
            validation_token_sha256=self.validation_token_sha256,
            validation_safety_fixture_sha256=(
                self.validation_safety_fixture_sha256
            ),
            full_count=self.full_count,
            records=self.records,
        )
        expected_manifest = _owned_hash(_PERTURBATION_MANIFEST_DOMAIN, payload)
        if self.manifest_sha256 != expected_manifest:
            raise ValueError("manifest_sha256 does not match the case inventory")
        expected_bytes = _canonical_json_bytes(
            {**payload, "manifest_sha256": self.manifest_sha256}
        )
        if self._canonical_bytes != expected_bytes:
            raise ValueError("case-file bytes are not the exact canonical payload")

    @property
    def is_complete(self) -> bool:
        return (
            self.materialized_count == self.full_count
            and tuple(record.case_index for record in self.records)
            == tuple(range(self.full_count))
        )

    @property
    def selected_count(self) -> int:
        return len(self.selected_records)

    @property
    def canonical_bytes(self) -> bytes:
        self.__post_init__()
        return bytes(self._canonical_bytes)


def _vocabulary_payload(
    vocabulary: IndependentVocabularyIdentity,
) -> dict[str, object]:
    return {
        "vocabulary_id": vocabulary.vocabulary_id,
        "size": vocabulary.size,
        "tokenizer_spec_sha256": vocabulary.tokenizer_spec_sha256,
        "vocabulary_sha256": vocabulary.vocabulary_sha256,
    }


def _frozen_perturbation_payload(
    *,
    schema_version: str,
    generator_version: str,
    seed: int,
    vocabulary: IndependentVocabularyIdentity,
    validation_token_sha256: str,
    validation_safety_fixture_sha256: str,
    full_count: int,
    records: tuple[ValidationPerturbation, ...],
) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "generator_version": generator_version,
        "seed": seed,
        "vocabulary": _vocabulary_payload(vocabulary),
        "validation_token_sha256": validation_token_sha256,
        "validation_safety_fixture_sha256": (
            validation_safety_fixture_sha256
        ),
        "full_count": full_count,
        "materialized_count": len(records),
        "records": [
            _validation_perturbation_payload(record, include_sha256=True)
            for record in records
        ],
    }


def freeze_validation_perturbations(
    *,
    vocabulary: IndependentVocabularyIdentity,
    validation_token_sha256: str,
    validation_safety_fixture_sha256: str,
    records: Iterable[ValidationPerturbation],
    full_count: int = VALIDATION_PERTURBATION_COUNT,
) -> bytes:
    """Serialize a full or explicitly shrunken case set canonically."""

    if type(vocabulary) is not IndependentVocabularyIdentity:
        raise ValueError("vocabulary must be an independent identity")
    vocabulary.__post_init__()
    if vocabulary.size != 258:
        raise ValueError("validation perturbations require vocabulary size 258")
    _require_sha256(validation_token_sha256, "validation_token_sha256")
    _require_sha256(
        validation_safety_fixture_sha256,
        "validation_safety_fixture_sha256",
    )
    if full_count != VALIDATION_PERTURBATION_COUNT:
        raise ValueError("full_count must remain frozen at 4,096")
    try:
        ordered_records = tuple(sorted(records, key=lambda record: record.case_index))
    except (TypeError, AttributeError) as exc:
        raise ValueError(
            "records must contain ValidationPerturbation instances"
        ) from exc
    if (
        not ordered_records
        or not all(type(record) is ValidationPerturbation for record in ordered_records)
    ):
        raise ValueError("at least one exact perturbation record is required")
    indices = tuple(record.case_index for record in ordered_records)
    if (
        len(set(indices)) != len(indices)
        or any(index >= full_count for index in indices)
    ):
        raise ValueError("record indices must be unique and below full_count")
    for record in ordered_records:
        record.__post_init__()
        if (
            record.vocabulary_size != vocabulary.size
            or record.vocabulary_sha256 != vocabulary.vocabulary_sha256
            or record.validation_token_sha256 != validation_token_sha256
            or record.validation_safety_fixture_sha256
            != validation_safety_fixture_sha256
        ):
            raise ValueError("record identity differs from the frozen set")
    payload = _frozen_perturbation_payload(
        schema_version=VALIDATION_PERTURBATION_SCHEMA_VERSION,
        generator_version=VALIDATION_PERTURBATION_GENERATOR_VERSION,
        seed=VALIDATION_PERTURBATION_SEED,
        vocabulary=vocabulary,
        validation_token_sha256=validation_token_sha256,
        validation_safety_fixture_sha256=validation_safety_fixture_sha256,
        full_count=full_count,
        records=ordered_records,
    )
    manifest_sha256 = _owned_hash(_PERTURBATION_MANIFEST_DOMAIN, payload)
    return _canonical_json_bytes(
        {**payload, "manifest_sha256": manifest_sha256}
    )


def _parse_vocabulary(value: object) -> IndependentVocabularyIdentity:
    payload = _mapping(value, _VOCABULARY_FIELDS, "vocabulary")
    return IndependentVocabularyIdentity(
        vocabulary_id=_require_nonempty(
            payload["vocabulary_id"],
            "vocabulary.vocabulary_id",
        ),
        size=_require_int(payload["size"], "vocabulary.size", minimum=1),
        tokenizer_spec_sha256=_require_sha256(
            payload["tokenizer_spec_sha256"],
            "vocabulary.tokenizer_spec_sha256",
        ),
        vocabulary_sha256=_require_sha256(
            payload["vocabulary_sha256"],
            "vocabulary.vocabulary_sha256",
        ),
    )


def _parse_perturbation_record(
    value: object,
    *,
    vocabulary: IndependentVocabularyIdentity,
    validation_token_sha256: str,
    validation_safety_fixture_sha256: str,
    record_offset: int,
) -> ValidationPerturbation:
    payload = _mapping(
        value,
        _PERTURBATION_RECORD_FIELDS,
        f"records[{record_offset}]",
    )
    real_target_count = _require_int(
        payload["real_target_count"],
        f"records[{record_offset}].real_target_count",
        minimum=1,
    )
    receiver_t = _require_int(
        payload["receiver_t"],
        f"records[{record_offset}].receiver_t",
        minimum=1,
    )
    if receiver_t > real_target_count:
        raise ValueError("record receiver_t exceeds real_target_count")
    return ValidationPerturbation(
        case_id=_require_nonempty(
            payload["case_id"],
            f"records[{record_offset}].case_id",
        ),
        case_index=_require_int(
            payload["case_index"],
            f"records[{record_offset}].case_index",
        ),
        source_window_index=_require_int(
            payload["source_window_index"],
            f"records[{record_offset}].source_window_index",
        ),
        window_start=_require_int(
            payload["window_start"],
            f"records[{record_offset}].window_start",
        ),
        real_target_count=real_target_count,
        receiver_t=receiver_t,
        prefix_token_ids=_tokens(
            payload["prefix_token_ids"],
            vocabulary_size=vocabulary.size,
            expected_length=receiver_t - 1,
            name=f"records[{record_offset}].prefix_token_ids",
        ),
        left_tail_token_ids=_tokens(
            payload["left_tail_token_ids"],
            vocabulary_size=vocabulary.size,
            expected_length=real_target_count - receiver_t + 1,
            name=f"records[{record_offset}].left_tail_token_ids",
        ),
        right_tail_token_ids=_tokens(
            payload["right_tail_token_ids"],
            vocabulary_size=vocabulary.size,
            expected_length=real_target_count - receiver_t + 1,
            name=f"records[{record_offset}].right_tail_token_ids",
        ),
        vocabulary_size=_require_int(
            payload["vocabulary_size"],
            f"records[{record_offset}].vocabulary_size",
            minimum=1,
        ),
        vocabulary_sha256=_require_sha256(
            payload["vocabulary_sha256"],
            f"records[{record_offset}].vocabulary_sha256",
        ),
        validation_token_sha256=_require_sha256(
            payload["validation_token_sha256"],
            f"records[{record_offset}].validation_token_sha256",
        ),
        validation_safety_fixture_sha256=_require_sha256(
            payload["validation_safety_fixture_sha256"],
            f"records[{record_offset}].validation_safety_fixture_sha256",
        ),
        case_sha256=_require_sha256(
            payload["case_sha256"],
            f"records[{record_offset}].case_sha256",
        ),
    )


def _selected_records(
    records: tuple[ValidationPerturbation, ...],
    *,
    record_indices: Iterable[int] | None,
    max_cases: int | None,
) -> tuple[ValidationPerturbation, ...]:
    if max_cases is not None:
        _require_int(max_cases, "max_cases", minimum=1)
    if record_indices is None:
        return records if max_cases is None else records[:max_cases]
    try:
        indices = tuple(record_indices)
    except TypeError as exc:
        raise ValueError("record_indices must be an iterable of integers") from exc
    if (
        not indices
        or any(type(index) is not int or index < 0 for index in indices)
        or len(set(indices)) != len(indices)
    ):
        raise ValueError("record_indices must be unique nonnegative integers")
    if max_cases is not None:
        indices = indices[:max_cases]
    by_index = {record.case_index: record for record in records}
    try:
        return tuple(by_index[index] for index in indices)
    except KeyError as exc:
        raise ValueError(
            f"requested record index is not materialized: {exc.args[0]}"
        ) from exc


def load_frozen_validation_perturbations(
    data: bytes,
    *,
    expected_raw_sha256: str | None = None,
    expected_vocabulary_sha256: str | None = None,
    expected_validation_token_sha256: str | None = None,
    expected_validation_safety_fixture_sha256: str | None = None,
    require_complete: bool = False,
    record_indices: Iterable[int] | None = None,
    max_cases: int | None = None,
) -> FrozenValidationPerturbationSet:
    """Parse strict canonical case-file bytes and select a bounded traversal."""

    if type(data) is not bytes:
        raise ValueError("data must be immutable bytes")
    raw_sha256 = hashlib.sha256(data).hexdigest()
    if expected_raw_sha256 is not None:
        _require_sha256(expected_raw_sha256, "expected_raw_sha256")
        if raw_sha256 != expected_raw_sha256:
            raise ValueError("raw perturbation bytes do not match the expected hash")
    root = _mapping(
        _json_without_duplicate_fields(data),
        _PERTURBATION_ROOT_FIELDS,
        "validation perturbation fixture",
    )
    if _canonical_json_bytes(root) != data:
        raise ValueError("validation perturbation bytes are not canonical JSON")
    if (
        root["schema_version"] != VALIDATION_PERTURBATION_SCHEMA_VERSION
        or root["generator_version"]
        != VALIDATION_PERTURBATION_GENERATOR_VERSION
        or root["seed"] != VALIDATION_PERTURBATION_SEED
    ):
        raise ValueError("unsupported validation perturbation identity")
    vocabulary = _parse_vocabulary(root["vocabulary"])
    if vocabulary.size != 258:
        raise ValueError("validation perturbations require vocabulary size 258")
    validation_token_sha256 = _require_sha256(
        root["validation_token_sha256"],
        "validation_token_sha256",
    )
    validation_safety_fixture_sha256 = _require_sha256(
        root["validation_safety_fixture_sha256"],
        "validation_safety_fixture_sha256",
    )
    full_count = _require_int(root["full_count"], "full_count", minimum=1)
    if full_count != VALIDATION_PERTURBATION_COUNT:
        raise ValueError("full_count must remain frozen at 4,096")
    materialized_count = _require_int(
        root["materialized_count"],
        "materialized_count",
        minimum=1,
    )
    raw_records = _sequence(root["records"], "records")
    if len(raw_records) != materialized_count:
        raise ValueError("materialized_count does not match records")
    records = tuple(
        _parse_perturbation_record(
            value,
            vocabulary=vocabulary,
            validation_token_sha256=validation_token_sha256,
            validation_safety_fixture_sha256=(
                validation_safety_fixture_sha256
            ),
            record_offset=offset,
        )
        for offset, value in enumerate(raw_records)
    )
    manifest_sha256 = _require_sha256(
        root["manifest_sha256"],
        "manifest_sha256",
    )
    if expected_vocabulary_sha256 is not None:
        _require_sha256(
            expected_vocabulary_sha256,
            "expected_vocabulary_sha256",
        )
        if vocabulary.vocabulary_sha256 != expected_vocabulary_sha256:
            raise ValueError("vocabulary identity differs from the expected hash")
    if expected_validation_token_sha256 is not None:
        _require_sha256(
            expected_validation_token_sha256,
            "expected_validation_token_sha256",
        )
        if validation_token_sha256 != expected_validation_token_sha256:
            raise ValueError("validation token identity differs from expectation")
    if expected_validation_safety_fixture_sha256 is not None:
        _require_sha256(
            expected_validation_safety_fixture_sha256,
            "expected_validation_safety_fixture_sha256",
        )
        if (
            validation_safety_fixture_sha256
            != expected_validation_safety_fixture_sha256
        ):
            raise ValueError(
                "validation safety fixture identity differs from expectation"
            )
    selected = _selected_records(
        records,
        record_indices=record_indices,
        max_cases=max_cases,
    )
    result = FrozenValidationPerturbationSet(
        raw_sha256=raw_sha256,
        schema_version=VALIDATION_PERTURBATION_SCHEMA_VERSION,
        generator_version=VALIDATION_PERTURBATION_GENERATOR_VERSION,
        seed=VALIDATION_PERTURBATION_SEED,
        vocabulary=vocabulary,
        validation_token_sha256=validation_token_sha256,
        validation_safety_fixture_sha256=(
            validation_safety_fixture_sha256
        ),
        full_count=full_count,
        materialized_count=materialized_count,
        records=records,
        manifest_sha256=manifest_sha256,
        selected_records=selected,
        _canonical_bytes=data,
    )
    if type(require_complete) is not bool:
        raise ValueError("require_complete must be Boolean")
    if require_complete and not result.is_complete:
        raise ValueError("the complete 4,096-record inventory is required")
    return result


@dataclass(frozen=True)
class _ValidationSafetyRow:
    source_window_index: int
    window_start: int
    real_target_count: int
    token_ids: tuple[int, ...]


def _validation_fixture_header(data: bytes) -> tuple[str, int, int]:
    if type(data) is not bytes:
        raise ValueError("validation_safety_fixture_bytes must be immutable bytes")
    header_size = len(_VALIDATION_FIXTURE_DOMAIN) + 32 + 4
    if (
        len(data) < header_size
        or data[: len(_VALIDATION_FIXTURE_DOMAIN)]
        != _VALIDATION_FIXTURE_DOMAIN
    ):
        raise ValueError("validation safety fixture has the wrong domain")
    cursor = len(_VALIDATION_FIXTURE_DOMAIN)
    validation_token_sha256 = data[cursor : cursor + 32].hex()
    cursor += 32
    row_count = int.from_bytes(data[cursor : cursor + 4], "little")
    cursor += 4
    if row_count <= 0:
        raise ValueError("validation safety fixture must contain at least one row")
    expected_length = cursor + row_count * _VALIDATION_ROW_BYTES
    if len(data) != expected_length:
        raise ValueError("validation safety fixture byte length is inconsistent")
    return validation_token_sha256, row_count, cursor


def _validation_safety_row(
    data: bytes,
    *,
    header_size: int,
    source_window_index: int,
    vocabulary_size: int,
) -> _ValidationSafetyRow:
    offset = header_size + source_window_index * _VALIDATION_ROW_BYTES
    row = data[offset : offset + _VALIDATION_ROW_BYTES]
    window_start = int.from_bytes(row[:8], "little")
    real_target_count = int.from_bytes(row[8:10], "little")
    if not 1 <= real_target_count <= 32:
        raise ValueError("validation safety row has an invalid target count")
    token_bytes = row[10:]
    token_ids = tuple(
        int.from_bytes(token_bytes[index : index + 2], "little")
        for index in range(0, len(token_bytes), 2)
    )
    if (
        len(token_ids) != 33
        or any(token_id >= vocabulary_size for token_id in token_ids)
    ):
        raise ValueError("validation safety row contains an invalid token")
    return _ValidationSafetyRow(
        source_window_index=source_window_index,
        window_start=window_start,
        real_target_count=real_target_count,
        token_ids=token_ids,
    )


def _counter_bounded(
    *,
    case_key: bytes,
    purpose: bytes,
    coordinate: int,
    bound: int,
    forbidden: int | None = None,
) -> int:
    _require_int(coordinate, "coordinate")
    _require_int(bound, "bound", minimum=1)
    if type(purpose) is not bytes or not purpose:
        raise ValueError("counter purpose must be nonempty bytes")
    if forbidden is not None and (
        type(forbidden) is not int or not 0 <= forbidden < bound
    ):
        raise ValueError("forbidden counter value lies outside the bound")
    limit = (1 << 64) - ((1 << 64) % bound)
    attempt = 0
    while True:
        digest = hashlib.sha256(
            _PERTURBATION_COUNTER_DOMAIN
            + case_key
            + len(purpose).to_bytes(2, "little")
            + purpose
            + coordinate.to_bytes(8, "little")
            + attempt.to_bytes(8, "little")
        ).digest()
        candidate = int.from_bytes(digest[:8], "little")
        if candidate < limit:
            value = candidate % bound
            if forbidden is None or value != forbidden:
                return value
        attempt += 1


def _validation_perturbation_from_row(
    row: _ValidationSafetyRow,
    *,
    vocabulary: IndependentVocabularyIdentity,
    validation_token_sha256: str,
    validation_safety_fixture_sha256: str,
    seed: int,
) -> ValidationPerturbation:
    case_key = (
        seed.to_bytes(8, "little")
        + bytes.fromhex(validation_token_sha256)
        + bytes.fromhex(validation_safety_fixture_sha256)
        + row.source_window_index.to_bytes(8, "little")
        + row.window_start.to_bytes(8, "little")
    )
    receiver_t = 1 + _counter_bounded(
        case_key=case_key,
        purpose=b"receiver",
        coordinate=0,
        bound=row.real_target_count,
    )
    # The binary row is [unscored preceding input] + [scored targets].
    # H6's public prefix for scored position t therefore uses row[1:t].
    target_sequence = row.token_ids[1 : row.real_target_count + 1]
    prefix = target_sequence[: receiver_t - 1]
    tail_length = row.real_target_count - receiver_t + 1
    left_tail: list[int] = []
    right_tail: list[int] = []
    for coordinate in range(tail_length):
        left = _counter_bounded(
            case_key=case_key,
            purpose=b"left-tail",
            coordinate=coordinate,
            bound=vocabulary.size,
        )
        right = _counter_bounded(
            case_key=case_key,
            purpose=b"right-tail",
            coordinate=coordinate,
            bound=vocabulary.size,
            forbidden=left,
        )
        left_tail.append(left)
        right_tail.append(right)
    case_id = f"validation-window-{row.source_window_index:04d}"
    payload = {
        "case_id": case_id,
        "case_index": row.source_window_index,
        "source_window_index": row.source_window_index,
        "window_start": row.window_start,
        "real_target_count": row.real_target_count,
        "receiver_t": receiver_t,
        "prefix_token_ids": list(prefix),
        "left_tail_token_ids": left_tail,
        "right_tail_token_ids": right_tail,
        "vocabulary_size": vocabulary.size,
        "vocabulary_sha256": vocabulary.vocabulary_sha256,
        "validation_token_sha256": validation_token_sha256,
        "validation_safety_fixture_sha256": (
            validation_safety_fixture_sha256
        ),
    }
    return ValidationPerturbation(
        case_id=case_id,
        case_index=row.source_window_index,
        source_window_index=row.source_window_index,
        window_start=row.window_start,
        real_target_count=row.real_target_count,
        receiver_t=receiver_t,
        prefix_token_ids=prefix,
        left_tail_token_ids=tuple(left_tail),
        right_tail_token_ids=tuple(right_tail),
        vocabulary_size=vocabulary.size,
        vocabulary_sha256=vocabulary.vocabulary_sha256,
        validation_token_sha256=validation_token_sha256,
        validation_safety_fixture_sha256=(
            validation_safety_fixture_sha256
        ),
        case_sha256=_owned_hash(_PERTURBATION_CASE_DOMAIN, payload),
    )


def generate_validation_perturbation_records(
    validation_safety_fixture_bytes: bytes,
    *,
    vocabulary: IndependentVocabularyIdentity,
    seed: int = VALIDATION_PERTURBATION_SEED,
    expected_validation_safety_fixture_sha256: str | None = None,
    expected_validation_token_sha256: str | None = None,
    record_indices: Iterable[int] | None = None,
    max_cases: int | None = None,
) -> tuple[ValidationPerturbation, ...]:
    """Derive independent perturbations directly from safety-fixture bytes."""

    if type(vocabulary) is not IndependentVocabularyIdentity:
        raise ValueError("vocabulary must be an independent identity")
    vocabulary.__post_init__()
    if vocabulary.size != 258:
        raise ValueError("validation perturbations require vocabulary size 258")
    if seed != VALIDATION_PERTURBATION_SEED:
        raise ValueError("the validation perturbation seed is frozen")
    validation_token_sha256, row_count, header_size = _validation_fixture_header(
        validation_safety_fixture_bytes
    )
    fixture_sha256 = hashlib.sha256(validation_safety_fixture_bytes).hexdigest()
    if expected_validation_safety_fixture_sha256 is not None:
        _require_sha256(
            expected_validation_safety_fixture_sha256,
            "expected_validation_safety_fixture_sha256",
        )
        if fixture_sha256 != expected_validation_safety_fixture_sha256:
            raise ValueError("validation safety fixture bytes have the wrong hash")
    if expected_validation_token_sha256 is not None:
        _require_sha256(
            expected_validation_token_sha256,
            "expected_validation_token_sha256",
        )
        if validation_token_sha256 != expected_validation_token_sha256:
            raise ValueError("validation safety fixture names the wrong token hash")
    selected_indices = _normalize_case_indices(
        record_indices,
        total_count=row_count,
        max_cases=max_cases,
    )
    if selected_indices is None:
        selected_indices = tuple(
            range(row_count if max_cases is None else min(row_count, max_cases))
        )
    records = tuple(
        _validation_perturbation_from_row(
            _validation_safety_row(
                validation_safety_fixture_bytes,
                header_size=header_size,
                source_window_index=source_window_index,
                vocabulary_size=vocabulary.size,
            ),
            vocabulary=vocabulary,
            validation_token_sha256=validation_token_sha256,
            validation_safety_fixture_sha256=fixture_sha256,
            seed=seed,
        )
        for source_window_index in selected_indices
    )
    return records


def generate_frozen_validation_perturbations(
    validation_safety_fixture_bytes: bytes,
    *,
    vocabulary: IndependentVocabularyIdentity,
    seed: int = VALIDATION_PERTURBATION_SEED,
    expected_validation_safety_fixture_sha256: str | None = None,
    expected_validation_token_sha256: str | None = None,
    record_indices: Iterable[int] | None = None,
    max_cases: int | None = None,
) -> FrozenValidationPerturbationSet:
    """Generate, freeze, and reparse a full or bounded perturbation set."""

    records = generate_validation_perturbation_records(
        validation_safety_fixture_bytes,
        vocabulary=vocabulary,
        seed=seed,
        expected_validation_safety_fixture_sha256=(
            expected_validation_safety_fixture_sha256
        ),
        expected_validation_token_sha256=expected_validation_token_sha256,
        record_indices=record_indices,
        max_cases=max_cases,
    )
    validation_token_sha256, _, _ = _validation_fixture_header(
        validation_safety_fixture_bytes
    )
    fixture_sha256 = hashlib.sha256(validation_safety_fixture_bytes).hexdigest()
    data = freeze_validation_perturbations(
        vocabulary=vocabulary,
        validation_token_sha256=validation_token_sha256,
        validation_safety_fixture_sha256=fixture_sha256,
        records=records,
    )
    return load_frozen_validation_perturbations(data)


__all__ = [
    "FrozenValidationPerturbationSet",
    "IndependentVocabularyIdentity",
    "ORDERED_TAIL_GENERATOR_VERSION",
    "OrderedTailPair",
    "OrderedTailPairInventory",
    "OrderedTailPositionInventory",
    "SMALL_PREFIX_COUNTS_BY_RECEIVER",
    "SMALL_PREFIX_HORIZON",
    "SMALL_PREFIX_TOTAL_COUNT",
    "SMALL_PREFIX_VOCABULARY_SIZE",
    "VALIDATION_PERTURBATION_COUNT",
    "VALIDATION_PERTURBATION_GENERATOR_VERSION",
    "VALIDATION_PERTURBATION_SCHEMA_VERSION",
    "VALIDATION_PERTURBATION_SEED",
    "ValidationPerturbation",
    "enumerate_ordered_tail_pairs",
    "freeze_validation_perturbations",
    "generate_frozen_validation_perturbations",
    "generate_validation_perturbation_records",
    "load_frozen_validation_perturbations",
    "ordered_tail_pair_counts",
    "ordered_tail_pair_inventory",
]
