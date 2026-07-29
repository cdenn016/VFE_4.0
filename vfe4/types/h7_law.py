"""Immutable records for law-derived H7 ELBO components.

These records deliberately describe values derived from one exact
``H7CompleteLawSnapshot``.  They do not accept H6 factor identities or
caller-supplied signed-KL conventions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final, Literal

from vfe4.types.h7 import (
    H7_GROUPED_EMISSION_TERM_IDS,
    H7_GROUPED_POSITIVE_KL_TERM_IDS,
    H7_RAW_FACTOR_SLOTS,
    h7_owned_sha256,
)


H7_LAW_SEMANTICS: Final = (
    "augmented-joint-with-explicit-source-labels-v1"
)
H7_LAW_QUADRATURE_ORDER: Final = 51
H7_SOURCE_ASSEMBLY_SEMANTICS: Final = (
    "fixed-source-assembly-from-complete-law-v1"
)

H7_ENTROPY_CHILD_IDS: Final = (
    "initial_joint",
    "model_source[1]",
    "model_transition[1]",
    "state_source[1]",
    "state_transition[1]",
    "model_source[2]",
    "model_transition[2]",
    "state_source[2]",
    "state_transition[2]",
)
H7_ENTROPY_OWNER_CHILD_IDS: Final = (
    H7_ENTROPY_CHILD_IDS[:5],
    H7_ENTROPY_CHILD_IDS[5:],
)
H7_SOURCE_ASSEMBLY_ROW_ORDER: Final = (
    ("model_source", 1),
    ("state_source", 1),
    ("model_source", 2),
    ("state_source", 2),
)

H7FixtureId = Literal["h1-v1", "h7-v1"]
H7RawPartition = Literal[
    "initial",
    "model_source",
    "model_transition",
    "state_source",
    "state_transition",
    "emission",
    "entropy",
]
H7RawSlot = tuple[H7RawPartition, int]
H7SourcePartition = Literal["model_source", "state_source"]
H7EntropyChildId = Literal[
    "initial_joint",
    "model_source[1]",
    "model_transition[1]",
    "state_source[1]",
    "state_transition[1]",
    "model_source[2]",
    "model_transition[2]",
    "state_source[2]",
    "state_transition[2]",
]
H7RawTermSemantics = Literal[
    "expected_log_generative_factor",
    "expected_log_emission",
    "recognition_entropy",
]
H7GroupedTermSemantics = Literal[
    "expected_log_emission",
    "positive_kl_q_to_p",
]

_LOWER_HEX = frozenset("0123456789abcdef")
_FLOAT64_EPSILON = math.ulp(1.0)


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase 64-hex SHA-256")
    return value


def _require_finite(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be an exact finite float")
    return value


def _require_operands(
    values: object,
    name: str,
    *,
    complete_law_snapshot_sha256: str | None = None,
) -> tuple[str, ...]:
    if (
        type(values) is not tuple
        or not values
        or len(set(values)) != len(values)
    ):
        raise ValueError(f"{name} must be a nonempty unique tuple")
    for value in values:
        _require_sha256(value, name)
    if (
        complete_law_snapshot_sha256 is not None
        and complete_law_snapshot_sha256 not in values
    ):
        raise ValueError(f"{name} must retain the complete-law snapshot hash")
    return values


def _record_sha256(domain: str, payload: object) -> str:
    return h7_owned_sha256(domain, payload)


def h7_law_equality_tolerance(
    *,
    raw_values: tuple[float, ...],
    grouped_signed_values: tuple[float, ...],
    monolithic_total: float,
) -> float:
    """Return the deterministic float64 algebra allowance for three views."""

    if (
        type(raw_values) is not tuple
        or type(grouped_signed_values) is not tuple
        or not raw_values
        or not grouped_signed_values
    ):
        raise ValueError("H7 equality tolerance requires nonempty exact tuples")
    values = (*raw_values, *grouped_signed_values, monolithic_total)
    for index, value in enumerate(values):
        _require_finite(value, f"equality value[{index}]")
    scale = max(
        1.0,
        math.fsum(abs(value) for value in raw_values),
        math.fsum(abs(value) for value in grouped_signed_values),
        abs(monolithic_total),
    )
    return float(4096.0 * _FLOAT64_EPSILON * scale)


@dataclass(frozen=True)
class H7SourceAssemblyRow:
    """One exact fixed generative source row projected from the complete law."""

    partition: H7SourcePartition
    receiver_t: Literal[1, 2]
    support: tuple[int, ...]
    probabilities: tuple[float, ...]
    complete_law_operand_sha256s: tuple[str, ...]
    row_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (self.partition, self.receiver_t) not in H7_SOURCE_ASSEMBLY_ROW_ORDER:
            raise ValueError("source assembly row is outside the frozen inventory")
        if (
            type(self.support) is not tuple
            or not self.support
            or len(self.support) != len(set(self.support))
            or tuple(sorted(self.support)) != self.support
            or any(
                type(source) is not int
                or source < 0
                or source >= self.receiver_t
                for source in self.support
            )
        ):
            raise ValueError("source assembly support is invalid")
        if (
            type(self.probabilities) is not tuple
            or len(self.probabilities) != len(self.support)
        ):
            raise ValueError("source assembly probabilities do not match support")
        for index, probability in enumerate(self.probabilities):
            checked = _require_finite(
                probability,
                f"source assembly probability[{index}]",
            )
            if checked < 0.0:
                raise ValueError("source assembly probabilities must be nonnegative")
        if not math.isclose(
            math.fsum(self.probabilities),
            1.0,
            rel_tol=0.0,
            abs_tol=64.0 * _FLOAT64_EPSILON * len(self.probabilities),
        ):
            raise ValueError("source assembly probabilities must normalize")
        _require_operands(
            self.complete_law_operand_sha256s,
            "source assembly operands",
        )
        object.__setattr__(
            self,
            "row_sha256",
            _record_sha256(
                "vfe4.h7.law-source-assembly-row.v1",
                {
                    "partition": self.partition,
                    "receiver_t": self.receiver_t,
                    "support": self.support,
                    "probabilities": self.probabilities,
                    "complete_law_operand_sha256s": (
                        self.complete_law_operand_sha256s
                    ),
                },
            ),
        )


@dataclass(frozen=True)
class H7SourceAssemblyProfile:
    """Ordered fixed-source data needed to construct a law-compatible arm."""

    semantics: Literal["fixed-source-assembly-from-complete-law-v1"]
    fixture_id: H7FixtureId
    complete_law_snapshot_sha256: str
    rows: tuple[H7SourceAssemblyRow, ...]
    profile_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.semantics != H7_SOURCE_ASSEMBLY_SEMANTICS:
            raise ValueError("source assembly semantics changed")
        _require_sha256(
            self.complete_law_snapshot_sha256,
            "complete_law_snapshot_sha256",
        )
        if (
            type(self.rows) is not tuple
            or any(type(row) is not H7SourceAssemblyRow for row in self.rows)
            or tuple(
                (row.partition, row.receiver_t) for row in self.rows
            )
            != H7_SOURCE_ASSEMBLY_ROW_ORDER
        ):
            raise ValueError("source assembly row inventory is not exact")
        expected_supports = (
            ((0,), (0,), (0, 1), (0, 1))
            if self.fixture_id == "h1-v1"
            else ((0,), (0,), (1,), (1,))
            if self.fixture_id == "h7-v1"
            else None
        )
        if expected_supports is None or tuple(
            row.support for row in self.rows
        ) != expected_supports:
            raise ValueError("source assembly support disagrees with its fixture")
        if any(
            self.complete_law_snapshot_sha256
            not in row.complete_law_operand_sha256s
            for row in self.rows
        ):
            raise ValueError("source rows must retain their complete-law hash")
        object.__setattr__(
            self,
            "profile_sha256",
            _record_sha256(
                "vfe4.h7.law-source-assembly-profile.v1",
                {
                    "semantics": self.semantics,
                    "fixture_id": self.fixture_id,
                    "complete_law_snapshot_sha256": (
                        self.complete_law_snapshot_sha256
                    ),
                    "rows": self.rows,
                },
            ),
        )


@dataclass(frozen=True)
class H7EntropyChild:
    """One recognition-entropy child with a fixed chronological owner."""

    child_id: H7EntropyChildId
    owner_receiver_t: Literal[1, 2]
    value: float
    complete_law_operand_sha256s: tuple[str, ...]
    child_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        try:
            child_index = H7_ENTROPY_CHILD_IDS.index(self.child_id)
        except ValueError as error:
            raise ValueError("entropy child is outside the exact inventory") from error
        expected_owner = 1 if child_index < 5 else 2
        if self.owner_receiver_t != expected_owner:
            raise ValueError("entropy child has the wrong chronological owner")
        _require_finite(self.value, "entropy child value")
        _require_operands(
            self.complete_law_operand_sha256s,
            "entropy child operands",
        )
        object.__setattr__(
            self,
            "child_sha256",
            _record_sha256(
                "vfe4.h7.law-entropy-child.v1",
                {
                    "child_id": self.child_id,
                    "owner_receiver_t": self.owner_receiver_t,
                    "value": self.value,
                    "complete_law_operand_sha256s": (
                        self.complete_law_operand_sha256s
                    ),
                },
            ),
        )


@dataclass(frozen=True)
class H7EntropySlotOwnership:
    """The immutable child partition owned by one raw entropy slot."""

    raw_slot: Literal[("entropy", 1), ("entropy", 2)]
    children: tuple[H7EntropyChild, ...]
    value: float
    ownership_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.raw_slot not in (("entropy", 1), ("entropy", 2)):
            raise ValueError("entropy ownership raw slot is invalid")
        owner_index = self.raw_slot[1] - 1
        if (
            type(self.children) is not tuple
            or any(type(child) is not H7EntropyChild for child in self.children)
            or tuple(child.child_id for child in self.children)
            != H7_ENTROPY_OWNER_CHILD_IDS[owner_index]
            or any(
                child.owner_receiver_t != self.raw_slot[1]
                for child in self.children
            )
        ):
            raise ValueError("entropy ownership children are missing or reordered")
        _require_finite(self.value, "entropy ownership value")
        if self.value != math.fsum(child.value for child in self.children):
            raise ValueError("entropy ownership value does not reconstruct")
        object.__setattr__(
            self,
            "ownership_sha256",
            _record_sha256(
                "vfe4.h7.law-entropy-slot-ownership.v1",
                {
                    "raw_slot": self.raw_slot,
                    "children": self.children,
                    "value": self.value,
                },
            ),
        )


@dataclass(frozen=True)
class H7ChronologicalEntropyOwnership:
    """The frozen initial-plus-receiver-1 / receiver-2 entropy partition."""

    complete_law_snapshot_sha256: str
    slots: tuple[H7EntropySlotOwnership, H7EntropySlotOwnership]
    ownership_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_sha256(
            self.complete_law_snapshot_sha256,
            "complete_law_snapshot_sha256",
        )
        if (
            type(self.slots) is not tuple
            or len(self.slots) != 2
            or any(
                type(slot) is not H7EntropySlotOwnership for slot in self.slots
            )
            or tuple(slot.raw_slot for slot in self.slots)
            != (("entropy", 1), ("entropy", 2))
        ):
            raise ValueError("chronological entropy ownership is not exact")
        if any(
            self.complete_law_snapshot_sha256
            not in child.complete_law_operand_sha256s
            for slot in self.slots
            for child in slot.children
        ):
            raise ValueError("entropy children must retain the complete-law hash")
        object.__setattr__(
            self,
            "ownership_sha256",
            _record_sha256(
                "vfe4.h7.law-chronological-entropy-ownership.v1",
                {
                    "complete_law_snapshot_sha256": (
                        self.complete_law_snapshot_sha256
                    ),
                    "slots": self.slots,
                },
            ),
        )


@dataclass(frozen=True)
class H7LawRawTerm:
    """One of the exact 13 expected-log-factor/entropy raw slots."""

    slot: H7RawSlot
    semantics: H7RawTermSemantics
    value: float
    complete_law_operand_sha256s: tuple[str, ...]
    raw_term_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.slot not in H7_RAW_FACTOR_SLOTS:
            raise ValueError("raw law slot is outside the exact inventory")
        partition = self.slot[0]
        expected_semantics: H7RawTermSemantics = (
            "expected_log_emission"
            if partition == "emission"
            else "recognition_entropy"
            if partition == "entropy"
            else "expected_log_generative_factor"
        )
        if self.semantics != expected_semantics:
            raise ValueError("raw law term semantics disagree with its slot")
        _require_finite(self.value, "raw law term value")
        _require_operands(
            self.complete_law_operand_sha256s,
            "raw law term operands",
        )
        object.__setattr__(
            self,
            "raw_term_sha256",
            _record_sha256(
                "vfe4.h7.law-raw-term.v1",
                {
                    "slot": self.slot,
                    "semantics": self.semantics,
                    "value": self.value,
                    "complete_law_operand_sha256s": (
                        self.complete_law_operand_sha256s
                    ),
                },
            ),
        )


@dataclass(frozen=True)
class H7LawGroupedTerm:
    """One expected emission or nonnegative positive-KL grouped term."""

    term_id: str
    semantics: H7GroupedTermSemantics
    elbo_sign: Literal[-1, 1]
    value: float
    complete_law_operand_sha256s: tuple[str, ...]
    grouped_term_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.term_id in H7_GROUPED_EMISSION_TERM_IDS:
            expected_semantics = "expected_log_emission"
            expected_sign = 1
        elif self.term_id in H7_GROUPED_POSITIVE_KL_TERM_IDS:
            expected_semantics = "positive_kl_q_to_p"
            expected_sign = -1
        else:
            raise ValueError("grouped law term is outside the exact inventory")
        if self.semantics != expected_semantics or self.elbo_sign != expected_sign:
            raise ValueError("grouped law term semantics or sign changed")
        checked = _require_finite(self.value, "grouped law term value")
        if self.semantics == "positive_kl_q_to_p" and checked < 0.0:
            raise ValueError("positive grouped KL must be nonnegative")
        _require_operands(
            self.complete_law_operand_sha256s,
            "grouped law term operands",
        )
        object.__setattr__(
            self,
            "grouped_term_sha256",
            _record_sha256(
                "vfe4.h7.law-grouped-term.v1",
                {
                    "term_id": self.term_id,
                    "semantics": self.semantics,
                    "elbo_sign": self.elbo_sign,
                    "value": self.value,
                    "complete_law_operand_sha256s": (
                        self.complete_law_operand_sha256s
                    ),
                },
            ),
        )


@dataclass(frozen=True)
class H7LawComponents:
    """Complete raw, grouped, and monolithic views of one exact H7 law."""

    semantics: Literal["augmented-joint-with-explicit-source-labels-v1"]
    fixture_id: H7FixtureId
    complete_law_snapshot_sha256: str
    quadrature_order: Literal[51]
    source_assembly_profile: H7SourceAssemblyProfile
    entropy_ownership: H7ChronologicalEntropyOwnership
    raw_terms: tuple[H7LawRawTerm, ...]
    emission_terms: tuple[H7LawGroupedTerm, ...]
    positive_kl_terms: tuple[H7LawGroupedTerm, ...]
    raw_total: float
    grouped_total: float
    monolithic_total: float
    raw_grouped_equality_residual: float
    grouped_monolithic_equality_residual: float
    equality_tolerance: float
    components_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.semantics != H7_LAW_SEMANTICS:
            raise ValueError("H7 law semantics changed")
        if self.fixture_id not in ("h1-v1", "h7-v1"):
            raise ValueError("unsupported H7 complete-law fixture")
        _require_sha256(
            self.complete_law_snapshot_sha256,
            "complete_law_snapshot_sha256",
        )
        if self.quadrature_order != H7_LAW_QUADRATURE_ORDER:
            raise ValueError("H7 law quadrature order must remain 51")
        if (
            type(self.source_assembly_profile) is not H7SourceAssemblyProfile
            or self.source_assembly_profile.fixture_id != self.fixture_id
            or self.source_assembly_profile.complete_law_snapshot_sha256
            != self.complete_law_snapshot_sha256
        ):
            raise ValueError("H7 source assembly profile is not law-bound")
        if (
            type(self.entropy_ownership)
            is not H7ChronologicalEntropyOwnership
            or self.entropy_ownership.complete_law_snapshot_sha256
            != self.complete_law_snapshot_sha256
        ):
            raise ValueError("H7 entropy ownership is not law-bound")
        if (
            type(self.raw_terms) is not tuple
            or any(type(term) is not H7LawRawTerm for term in self.raw_terms)
            or tuple(term.slot for term in self.raw_terms)
            != H7_RAW_FACTOR_SLOTS
        ):
            raise ValueError("H7 raw law inventory is not the exact 13 slots")
        if (
            type(self.emission_terms) is not tuple
            or any(
                type(term) is not H7LawGroupedTerm
                for term in self.emission_terms
            )
            or tuple(term.term_id for term in self.emission_terms)
            != H7_GROUPED_EMISSION_TERM_IDS
            or type(self.positive_kl_terms) is not tuple
            or any(
                type(term) is not H7LawGroupedTerm
                for term in self.positive_kl_terms
            )
            or tuple(term.term_id for term in self.positive_kl_terms)
            != H7_GROUPED_POSITIVE_KL_TERM_IDS
        ):
            raise ValueError("H7 grouped law inventory is not exact")
        all_terms = (
            *self.raw_terms,
            *self.emission_terms,
            *self.positive_kl_terms,
        )
        if any(
            self.complete_law_snapshot_sha256
            not in term.complete_law_operand_sha256s
            for term in all_terms
        ):
            raise ValueError("every H7 term must retain its complete-law hash")
        raw_term_hashes = {term.raw_term_sha256 for term in self.raw_terms}
        if any(
            raw_term_hashes.intersection(term.complete_law_operand_sha256s)
            for term in (*self.emission_terms, *self.positive_kl_terms)
        ):
            raise ValueError(
                "raw term hashes cannot stand in for grouped law operands"
            )
        entropy_by_slot = {
            slot.raw_slot: slot.value for slot in self.entropy_ownership.slots
        }
        if any(
            term.value != entropy_by_slot[term.slot]
            for term in self.raw_terms
            if term.slot[0] == "entropy"
        ):
            raise ValueError("raw entropy slots disagree with typed ownership")
        for name in (
            "raw_total",
            "grouped_total",
            "monolithic_total",
            "raw_grouped_equality_residual",
            "grouped_monolithic_equality_residual",
            "equality_tolerance",
        ):
            _require_finite(getattr(self, name), name)
        if self.equality_tolerance < 0.0:
            raise ValueError("H7 equality tolerance must be nonnegative")
        expected_raw = math.fsum(term.value for term in self.raw_terms)
        grouped_signed_values = tuple(
            term.elbo_sign * term.value
            for term in (*self.emission_terms, *self.positive_kl_terms)
        )
        expected_grouped = math.fsum(grouped_signed_values)
        if self.raw_total != expected_raw:
            raise ValueError("H7 raw total does not reconstruct")
        if self.grouped_total != expected_grouped:
            raise ValueError("H7 grouped total does not reconstruct")
        if self.raw_grouped_equality_residual != abs(
            expected_raw - expected_grouped
        ):
            raise ValueError("H7 raw/grouped residual was not recomputed")
        if self.grouped_monolithic_equality_residual != abs(
            expected_grouped - self.monolithic_total
        ):
            raise ValueError("H7 grouped/monolithic residual was not recomputed")
        expected_tolerance = h7_law_equality_tolerance(
            raw_values=tuple(term.value for term in self.raw_terms),
            grouped_signed_values=grouped_signed_values,
            monolithic_total=self.monolithic_total,
        )
        if self.equality_tolerance != expected_tolerance:
            raise ValueError("H7 equality tolerance was not recomputed")
        if (
            self.raw_grouped_equality_residual > self.equality_tolerance
            or self.grouped_monolithic_equality_residual
            > self.equality_tolerance
        ):
            raise ValueError("H7 law views fail their deterministic equality gate")
        object.__setattr__(
            self,
            "components_sha256",
            _record_sha256(
                "vfe4.h7.law-components.v1",
                {
                    "semantics": self.semantics,
                    "fixture_id": self.fixture_id,
                    "complete_law_snapshot_sha256": (
                        self.complete_law_snapshot_sha256
                    ),
                    "quadrature_order": self.quadrature_order,
                    "source_assembly_profile": self.source_assembly_profile,
                    "entropy_ownership": self.entropy_ownership,
                    "raw_terms": self.raw_terms,
                    "emission_terms": self.emission_terms,
                    "positive_kl_terms": self.positive_kl_terms,
                    "raw_total": self.raw_total,
                    "grouped_total": self.grouped_total,
                    "monolithic_total": self.monolithic_total,
                    "raw_grouped_equality_residual": (
                        self.raw_grouped_equality_residual
                    ),
                    "grouped_monolithic_equality_residual": (
                        self.grouped_monolithic_equality_residual
                    ),
                    "equality_tolerance": self.equality_tolerance,
                },
            ),
        )


__all__ = [
    "H7ChronologicalEntropyOwnership",
    "H7EntropyChild",
    "H7EntropySlotOwnership",
    "H7LawComponents",
    "H7LawGroupedTerm",
    "H7LawRawTerm",
    "H7SourceAssemblyProfile",
    "H7SourceAssemblyRow",
    "H7_ENTROPY_CHILD_IDS",
    "H7_ENTROPY_OWNER_CHILD_IDS",
    "H7_LAW_QUADRATURE_ORDER",
    "H7_LAW_SEMANTICS",
    "H7_SOURCE_ASSEMBLY_ROW_ORDER",
    "H7_SOURCE_ASSEMBLY_SEMANTICS",
    "h7_law_equality_tolerance",
]
