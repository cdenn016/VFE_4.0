from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

import vfe4.generative.pushforward as generative_pushforward
import vfe4.objective.h7_law_evidence as law_evidence
import vfe4.recognition.pushforward as recognition_pushforward
from vfe4.geometry.group_action import borrow_h7_action
from vfe4.objective.h7_law_evidence import (
    H7LawEvaluationEvidence,
    build_h7_grouped_elbo_record,
    capture_h7_law_evaluation,
    require_h7_law_evaluation,
)
from vfe4.training.arms import build_a5
from vfe4.training.h7_assembly import (
    H7FixedSourceAssemblySpec,
    build_h7_fixed_a5_arm,
)
from vfe4.types.h6 import (
    ArmConfig,
    ArmId,
    CapacityAllocation,
    CausalDag,
    CausalDagRow,
    H6LanguageStructure,
    VocabularyIdentity,
    ZeroDimensionalBase,
)
from vfe4.types.h7 import (
    H7CompleteLawSnapshot,
    H7LawPairSnapshot,
    H7TrialSpec,
)
from vfe4.validation.h7_fixture import (
    H7_FIXTURE_PATH,
    adapt_optional_h1_fixture_bytes,
    h7_scalar_trial_specs,
    parse_h7_fixture_bytes,
)


_ROOT = Path(__file__).resolve().parents[2]
_H1_PATH = _ROOT / "vfe4" / "validation" / "fixtures" / "h1_v1.json"


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _config(*, factorized: bool = False) -> ArmConfig:
    family = "factorized" if factorized else "structured"
    return ArmConfig.create(
        arm=ArmId.A5,
        config_id=(
            f"h6-a5-{family}-fixed-exact-complete-"
            "latent-smoothing-v1"
        ),
        vocabulary=VocabularyIdentity(
            vocabulary_id="h7-law-evidence-test-v1",
            size=258,
            tokenizer_spec_sha256=_sha("h7 law evidence tokenizer"),
        ),
        horizon=2,
        latent_enabled=True,
        state_channel_enabled=True,
        model_channel_enabled=True,
        source_mode="categorical",
        map_mode="shared_vertex_coboundary",
        recognition_family=family,
        recognition_conditioning="smoothing",
        prior_variant="fixed",
        mixture_mode="exact",
        objective_kind="complete_elbo",
        capacity_allocation=CapacityAllocation.create(
            emission_width=4,
            latent_width=2,
            recognition_width=4,
            prior_context_width=None,
        ),
    )


def _structure(
    second_receiver_parents: tuple[int, ...],
) -> H6LanguageStructure:
    return H6LanguageStructure.create(
        base=ZeroDimensionalBase.create(),
        dag=CausalDag.create(
            node_labels=(0, 1, 2),
            rows=(
                CausalDagRow(1, (0,)),
                CausalDagRow(2, second_receiver_parents),
            ),
        ),
        receiver_labels=(1, 2),
    )


def _arm(fixture_id: str, *, factorized: bool = False):
    if fixture_id == "h1-v1":
        spec = H7FixedSourceAssemblySpec.from_h1(_structure((0, 1)))
    else:
        spec = H7FixedSourceAssemblySpec.from_h7_v1(_structure((1,)))
    return build_h7_fixed_a5_arm(
        _config(factorized=factorized),
        spec,
    )


def _scalar_pair(
    trial_index: int = 0,
) -> tuple[H7TrialSpec, H7LawPairSnapshot]:
    original = adapt_optional_h1_fixture_bytes(
        _H1_PATH.read_bytes(),
        required_scalar_trials=(
            "scalar-base-transformed",
            "scalar-internal-transformed",
        ),
    )
    assert original is not None
    spec = h7_scalar_trial_specs()[trial_index]
    action = spec.action
    borrowed = borrow_h7_action(
        tuple(item.value() for item in action.elements),
        kind=action.kind,
        dimension=1,
    )
    transformed = H7CompleteLawSnapshot.create(
        fixture_id="h1-v1",
        generative=generative_pushforward.freeze_h7_generative(
            generative_pushforward.pushforward_h7_generative_snapshot(
                original.generative,
                borrowed,
            ),
            action=borrowed,
        ),
        recognition=recognition_pushforward.freeze_h7_recognition(
            recognition_pushforward.pushforward_h7_recognition_snapshot(
                original.recognition,
                borrowed,
            )
        ),
        raw_fixture_sha256=original.raw_fixture_sha256,
        scalar_probe_set=original.scalar_probe_set,
    )
    return spec, H7LawPairSnapshot.create(
        original=original,
        transformed=transformed,
        action_sha256=action.action_sha256,
    )


def _matrix_pair() -> tuple[H7TrialSpec, H7LawPairSnapshot]:
    fixture = parse_h7_fixture_bytes(H7_FIXTURE_PATH.read_bytes())
    spec = next(
        item
        for item in fixture.matrix_trial_specs
        if item.trial_id == "matrix-nonidentity-internal-transformed"
    )
    action = spec.action
    original = H7CompleteLawSnapshot.create(
        fixture_id="h7-v1",
        generative=fixture.generative,
        recognition=fixture.recognition_families[0],
        raw_fixture_sha256=fixture.raw_fixture_sha256,
        scalar_probe_set=None,
    )
    borrowed = borrow_h7_action(
        tuple(item.value() for item in action.elements),
        kind=action.kind,
        dimension=2,
    )
    transformed = H7CompleteLawSnapshot.create(
        fixture_id="h7-v1",
        generative=generative_pushforward.freeze_h7_generative(
            generative_pushforward.pushforward_h7_generative_snapshot(
                original.generative,
                borrowed,
            ),
            action=borrowed,
        ),
        recognition=recognition_pushforward.freeze_h7_recognition(
            recognition_pushforward.pushforward_h7_recognition_snapshot(
                original.recognition,
                borrowed,
            )
        ),
        raw_fixture_sha256=original.raw_fixture_sha256,
        scalar_probe_set=None,
    )
    return spec, H7LawPairSnapshot.create(
        original=original,
        transformed=transformed,
        action_sha256=action.action_sha256,
    )


@pytest.mark.parametrize("fixture_id", ("h1-v1", "h7-v1"))
def test_h7_law_capture_derives_exact_registered_raw_evidence(
    fixture_id: str,
) -> None:
    spec, pair = (
        _scalar_pair() if fixture_id == "h1-v1" else _matrix_pair()
    )
    arm = _arm(fixture_id)

    evidence = capture_h7_law_evaluation(
        trial_spec=spec,
        law_pair=pair,
        action=spec.action,
        role="original",
        arm=arm,
    )

    assert require_h7_law_evaluation(
        evidence,
        trial_spec=spec,
        law_pair=pair,
        action=spec.action,
        role="original",
    ) is evidence
    assert evidence.fixture_id == fixture_id
    assert evidence.law_snapshot_sha256 == pair.original.snapshot_sha256
    assert evidence.assembly_receipt.fixture_id == fixture_id
    assert evidence.law_components.complete_law_snapshot_sha256 == (
        pair.original.snapshot_sha256
    )
    assert evidence.raw_trace_evidence.ordered_factor_values == tuple(
        term.value for term in evidence.law_components.raw_terms
    )
    assert abs(
        evidence.raw_trace_evidence.total_value
        - evidence.law_components.raw_total
    ) <= evidence.law_components.equality_tolerance


def test_h7_law_capture_rejects_ordinary_arm_cross_role_trial_and_clone() -> None:
    spec, pair = _scalar_pair()
    arm = _arm("h1-v1")

    with pytest.raises(ValueError, match="H7-specific|assembly"):
        capture_h7_law_evaluation(
            trial_spec=spec,
            law_pair=pair,
            action=spec.action,
            role="original",
            arm=build_a5(_config()),
        )

    evidence = capture_h7_law_evaluation(
        trial_spec=spec,
        law_pair=pair,
        action=spec.action,
        role="original",
        arm=arm,
    )
    with pytest.raises(ValueError, match="role"):
        require_h7_law_evaluation(
            evidence,
            trial_spec=spec,
            law_pair=pair,
            action=spec.action,
            role="transformed",
        )
    other_spec, _ = _scalar_pair(1)
    with pytest.raises(ValueError, match="trial"):
        require_h7_law_evaluation(
            evidence,
            trial_spec=other_spec,
            law_pair=pair,
            action=spec.action,
            role="original",
        )
    with pytest.raises(TypeError, match="copy"):
        copy.copy(evidence)
    forged = object.__new__(H7LawEvaluationEvidence)
    with pytest.raises(ValueError, match="registered|capture"):
        require_h7_law_evaluation(
            forged,
            trial_spec=spec,
            law_pair=pair,
            action=spec.action,
            role="original",
        )


def test_h7_grouped_record_is_constructed_from_role_bound_law_evidence() -> None:
    spec, pair = _scalar_pair()
    arm = _arm("h1-v1")
    original = capture_h7_law_evaluation(
        trial_spec=spec,
        law_pair=pair,
        action=spec.action,
        role="original",
        arm=arm,
    )
    transformed = capture_h7_law_evaluation(
        trial_spec=spec,
        law_pair=pair,
        action=spec.action,
        role="transformed",
        arm=arm,
    )

    grouped = build_h7_grouped_elbo_record(
        trial_spec=spec,
        law_pair=pair,
        action=spec.action,
        original_evidence=original,
        transformed_evidence=transformed,
    )

    assert grouped.law_pair_sha256 == pair.law_pair_sha256
    assert tuple(term.original_value for term in grouped.emission_terms) == (
        tuple(term.value for term in original.law_components.emission_terms)
    )
    assert tuple(term.original_value for term in grouped.positive_kl_terms) == (
        tuple(
            term.value for term in original.law_components.positive_kl_terms
        )
    )
    assert grouped.original_grouped_total == (
        original.law_components.grouped_total
    )
    assert grouped.transformed_grouped_total == (
        transformed.law_components.grouped_total
    )
    assert grouped.original_raw_grouped_equality_residual <= (
        original.law_components.equality_tolerance
    )
    assert grouped.transformed_raw_grouped_equality_residual <= (
        transformed.law_components.equality_tolerance
    )
    with pytest.raises(ValueError, match="role"):
        build_h7_grouped_elbo_record(
            trial_spec=spec,
            law_pair=pair,
            action=spec.action,
            original_evidence=transformed,
            transformed_evidence=original,
        )


def test_h7_law_capture_uses_import_bound_derivation_and_receipt_apis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, pair = _scalar_pair()
    arm = _arm("h1-v1")

    def forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("mutable module seam was called")

    monkeypatch.setattr(
        law_evidence,
        "build_h7_law_components",
        forbidden,
    )
    monkeypatch.setattr(
        law_evidence,
        "require_h7_fixed_source_assembly",
        forbidden,
    )
    monkeypatch.setattr(
        law_evidence,
        "capture_h7_complete_language_elbo",
        forbidden,
    )
    monkeypatch.setattr(
        law_evidence,
        "require_h7_complete_factor_trace",
        forbidden,
    )
    monkeypatch.setattr(
        law_evidence,
        "adapt_h7_raw_factor_trace_evidence",
        forbidden,
    )

    evidence = capture_h7_law_evaluation(
        trial_spec=spec,
        law_pair=pair,
        action=spec.action,
        role="original",
        arm=arm,
    )

    assert evidence.law_components.complete_law_snapshot_sha256 == (
        pair.original.snapshot_sha256
    )
    assert require_h7_law_evaluation(
        evidence,
        trial_spec=spec,
        law_pair=pair,
        action=spec.action,
        role="original",
    ) is evidence
