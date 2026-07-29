from __future__ import annotations

import math
from pathlib import Path

import pytest

from vfe4.objective.h7_law_components import (
    build_h7_law_components,
    derive_h7_source_assembly_profile,
)
from vfe4.types.h7 import (
    H7CompleteLawSnapshot,
    H7_GROUPED_EMISSION_TERM_IDS,
    H7_GROUPED_POSITIVE_KL_TERM_IDS,
    H7_RAW_FACTOR_SLOTS,
    H7_SCALAR_TRIAL_IDS,
)
from vfe4.types.h7_law import H7_LAW_SEMANTICS
from vfe4.validation.h7_fixture import (
    adapt_optional_h1_fixture_bytes,
    parse_h7_fixture_bytes,
)


_ROOT = Path(__file__).resolve().parents[2]
_H1_PATH = _ROOT / "vfe4" / "validation" / "fixtures" / "h1_v1.json"
_H7_PATH = _ROOT / "vfe4" / "validation" / "fixtures" / "h7_v1.json"

_ENTROPY_CHILD_IDS = (
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
_ENTROPY_OWNER_CHILD_IDS = (
    _ENTROPY_CHILD_IDS[:5],
    _ENTROPY_CHILD_IDS[5:],
)


def _h1_law() -> H7CompleteLawSnapshot:
    law = adapt_optional_h1_fixture_bytes(
        _H1_PATH.read_bytes(),
        required_scalar_trials=H7_SCALAR_TRIAL_IDS,
    )
    assert law is not None
    return law


def _h7_laws() -> tuple[H7CompleteLawSnapshot, H7CompleteLawSnapshot]:
    fixture = parse_h7_fixture_bytes(_H7_PATH.read_bytes())
    return tuple(
        H7CompleteLawSnapshot.create(
            fixture_id="h7-v1",
            generative=fixture.generative,
            recognition=recognition,
            raw_fixture_sha256=fixture.raw_fixture_sha256,
            scalar_probe_set=None,
        )
        for recognition in fixture.recognition_families
    )


@pytest.mark.parametrize("law_index", (0, 1))
def test_h7_law_builder_freezes_exact_inventory_and_entropy_ownership(
    law_index: int,
) -> None:
    law = _h7_laws()[law_index]

    components = build_h7_law_components(law)

    assert components.semantics == H7_LAW_SEMANTICS
    assert components.fixture_id == "h7-v1"
    assert components.complete_law_snapshot_sha256 == law.snapshot_sha256
    assert components.quadrature_order == 51
    assert tuple(term.slot for term in components.raw_terms) == H7_RAW_FACTOR_SLOTS
    assert len(components.raw_terms) == 13
    assert tuple(
        term.term_id for term in components.emission_terms
    ) == H7_GROUPED_EMISSION_TERM_IDS
    assert tuple(
        term.term_id for term in components.positive_kl_terms
    ) == H7_GROUPED_POSITIVE_KL_TERM_IDS
    assert tuple(
        child.child_id
        for owner in components.entropy_ownership.slots
        for child in owner.children
    ) == _ENTROPY_CHILD_IDS
    assert tuple(
        tuple(child.child_id for child in owner.children)
        for owner in components.entropy_ownership.slots
    ) == _ENTROPY_OWNER_CHILD_IDS
    assert tuple(
        owner.raw_slot for owner in components.entropy_ownership.slots
    ) == (("entropy", 1), ("entropy", 2))
    raw_by_slot = {term.slot: term.value for term in components.raw_terms}
    assert raw_by_slot[("entropy", 1)] == (
        components.entropy_ownership.slots[0].value
    )
    assert raw_by_slot[("entropy", 2)] == (
        components.entropy_ownership.slots[1].value
    )


def test_h1_law_builder_derives_positive_source_kls_from_exact_law() -> None:
    law = _h1_law()

    components = build_h7_law_components(law)
    positive_kls = {
        term.term_id: term.value for term in components.positive_kl_terms
    }

    expected_model_source_kl_2 = math.fsum(
        q * (math.log(q) - math.log(p))
        for q, p in zip((0.4, 0.6), (0.35, 0.65), strict=True)
    )
    expected_state_source_kl_2 = math.fsum(
        (
            0.4
            * math.fsum(
                q * (math.log(q) - math.log(p))
                for q, p in zip((0.75, 0.25), (0.55, 0.45), strict=True)
            ),
            0.6
            * math.fsum(
                q * (math.log(q) - math.log(p))
                for q, p in zip((0.2, 0.8), (0.55, 0.45), strict=True)
            ),
        )
    )
    assert positive_kls["model_source_kl[1]"] == 0.0
    assert positive_kls["state_source_kl[1]"] == 0.0
    assert positive_kls["model_source_kl[2]"] == pytest.approx(
        expected_model_source_kl_2,
        rel=2.0e-15,
        abs=2.0e-15,
    )
    assert positive_kls["state_source_kl[2]"] == pytest.approx(
        expected_state_source_kl_2,
        rel=2.0e-15,
        abs=2.0e-15,
    )
    assert all(term.value >= 0.0 for term in components.positive_kl_terms)


@pytest.mark.parametrize("fixture_id", ("h1-v1", "h7-v1"))
def test_h7_law_builder_raw_grouped_and_monolithic_views_agree(
    fixture_id: str,
) -> None:
    laws = (_h1_law(),) if fixture_id == "h1-v1" else _h7_laws()

    for law in laws:
        components = build_h7_law_components(law)

        assert components.raw_grouped_equality_residual <= (
            components.equality_tolerance
        )
        assert components.grouped_monolithic_equality_residual <= (
            components.equality_tolerance
        )
        assert components.raw_total == math.fsum(
            term.value for term in components.raw_terms
        )
        assert components.grouped_total == math.fsum(
            (
                *(term.value for term in components.emission_terms),
                *(-term.value for term in components.positive_kl_terms),
            )
        )
        assert all(
            term.semantics == "positive_kl_q_to_p" and term.elbo_sign == -1
            for term in components.positive_kl_terms
        )


@pytest.mark.parametrize("fixture_id", ("h1-v1", "h7-v1"))
def test_h7_law_builder_uses_complete_law_operands_not_raw_term_hashes(
    fixture_id: str,
) -> None:
    laws = (_h1_law(),) if fixture_id == "h1-v1" else _h7_laws()

    for law in laws:
        components = build_h7_law_components(law)
        raw_term_hashes = {term.raw_term_sha256 for term in components.raw_terms}
        grouped_terms = (
            *components.emission_terms,
            *components.positive_kl_terms,
        )

        assert all(
            law.snapshot_sha256 in term.complete_law_operand_sha256s
            for term in (*components.raw_terms, *grouped_terms)
        )
        assert all(
            raw_term_hashes.isdisjoint(term.complete_law_operand_sha256s)
            for term in grouped_terms
        )
        assert all(
            law.snapshot_sha256 in child.complete_law_operand_sha256s
            for owner in components.entropy_ownership.slots
            for child in owner.children
        )


def test_h7_source_assembly_profile_is_law_derived_and_exact() -> None:
    h1 = _h1_law()
    h7 = _h7_laws()[0]

    h1_profile = derive_h7_source_assembly_profile(h1)
    h7_profile = derive_h7_source_assembly_profile(h7)

    assert tuple(
        (row.partition, row.receiver_t, row.support, row.probabilities)
        for row in h1_profile.rows
    ) == (
        ("model_source", 1, (0,), (1.0,)),
        ("state_source", 1, (0,), (1.0,)),
        ("model_source", 2, (0, 1), (0.35, 0.65)),
        ("state_source", 2, (0, 1), (0.55, 0.45)),
    )
    assert tuple(
        (row.partition, row.receiver_t, row.support, row.probabilities)
        for row in h7_profile.rows
    ) == (
        ("model_source", 1, (0,), (1.0,)),
        ("state_source", 1, (0,), (1.0,)),
        ("model_source", 2, (1,), (1.0,)),
        ("state_source", 2, (1,), (1.0,)),
    )
    assert build_h7_law_components(h1).source_assembly_profile == h1_profile
    assert build_h7_law_components(h7).source_assembly_profile == h7_profile
    assert derive_h7_source_assembly_profile(h1) == h1_profile
    assert derive_h7_source_assembly_profile(h7) == h7_profile


def test_h7_law_builder_rejects_nonexact_law_instead_of_signed_values() -> None:
    with pytest.raises(
        ValueError,
        match="exact H7CompleteLawSnapshot",
    ):
        build_h7_law_components(object())  # type: ignore[arg-type]

    with pytest.raises(
        ValueError,
        match="exact H7CompleteLawSnapshot",
    ):
        derive_h7_source_assembly_profile(object())  # type: ignore[arg-type]
