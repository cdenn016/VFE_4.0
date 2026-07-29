from __future__ import annotations

import copy
import hashlib
import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

import vfe4.generative.pushforward as generative_pushforward
import vfe4.objective.h7_covariance as covariance
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
    H7AllowanceContribution,
    H7BudgetRecord,
    H7CompleteLawSnapshot,
    H7IndependentH1EvidenceRecord,
    H7LawPairSnapshot,
    H7OperandRecord,
    H7TrialSpec,
)
from vfe4.validation.h7_fixture import (
    H1_FIXTURE_RAW_SHA256,
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


def _matrix_pair(
    recognition_index: int = 0,
) -> tuple[H7TrialSpec, H7LawPairSnapshot]:
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
        recognition=fixture.recognition_families[recognition_index],
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


@pytest.mark.parametrize(
    ("fixture_id", "factorized"),
    (
        ("h1-v1", False),
        ("h7-v1", False),
        ("h7-v1", True),
    ),
)
def test_h7_law_capture_derives_exact_registered_raw_evidence(
    fixture_id: str,
    factorized: bool,
) -> None:
    spec, pair = (
        _scalar_pair()
        if fixture_id == "h1-v1"
        else _matrix_pair(int(factorized))
    )
    arm = _arm(fixture_id, factorized=factorized)

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
    other_transformed = capture_h7_law_evaluation(
        trial_spec=spec,
        law_pair=pair,
        action=spec.action,
        role="transformed",
        arm=_arm("h1-v1"),
    )
    with pytest.raises(ValueError, match="one issued arm assembly"):
        build_h7_grouped_elbo_record(
            trial_spec=spec,
            law_pair=pair,
            action=spec.action,
            original_evidence=original,
            transformed_evidence=other_transformed,
        )
    shifted_first = type(grouped.emission_terms[0]).create(
        term_id=grouped.emission_terms[0].term_id,
        semantics=grouped.emission_terms[0].semantics,
        elbo_sign=grouped.emission_terms[0].elbo_sign,
        original_value=grouped.emission_terms[0].original_value + 1.0,
        transformed_value=(
            grouped.emission_terms[0].transformed_value + 1.0
        ),
        original_complete_law_operand_sha256s=(
            grouped.emission_terms[
                0
            ].original_complete_law_operand_sha256s
        ),
        transformed_complete_law_operand_sha256s=(
            grouped.emission_terms[
                0
            ].transformed_complete_law_operand_sha256s
        ),
        covariance_residual=grouped.emission_terms[0].covariance_residual,
    )
    shifted_emissions = (shifted_first, grouped.emission_terms[1])
    shifted_terms = (*shifted_emissions, *grouped.positive_kl_terms)
    shifted_original_total = math.fsum(
        term.elbo_sign * term.original_value for term in shifted_terms
    )
    shifted_transformed_total = math.fsum(
        term.elbo_sign * term.transformed_value for term in shifted_terms
    )
    with pytest.raises(ValueError, match="float64 equality allowance"):
        type(grouped).create(
            representation=grouped.representation,
            law_pair_sha256=grouped.law_pair_sha256,
            original_raw_trace_evidence=(
                grouped.original_raw_trace_evidence
            ),
            transformed_raw_trace_evidence=(
                grouped.transformed_raw_trace_evidence
            ),
            emission_terms=shifted_emissions,
            positive_kl_terms=grouped.positive_kl_terms,
            entropy_diagnostic=grouped.entropy_diagnostic,
            original_grouped_total=shifted_original_total,
            transformed_grouped_total=shifted_transformed_total,
            original_raw_grouped_equality_residual=abs(
                grouped.original_raw_total - shifted_original_total
            ),
            transformed_raw_grouped_equality_residual=abs(
                grouped.transformed_raw_total - shifted_transformed_total
            ),
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


def _budget(invariant_id: str, category: str, allowance: float) -> H7BudgetRecord:
    operands = tuple(
        H7OperandRecord.create(
            operand_id=f"{invariant_id}:{role}",
            category=category,
            role=role,
            dtype="float64",
            shape=(),
            value_sha256=_sha(f"{invariant_id}:{role}:value"),
            scale=1.0,
            condition_number=1.0,
            normalization=1.0,
            oracle_value=None,
        )
        for role in ("original", "transformed")
    )
    contribution = H7AllowanceContribution.create(
        kind="operation_rounding",
        operation_id=f"{invariant_id}:comparison",
        operation_kind="pair_comparison",
        operation_count=1,
        quadrature_order=None,
        unit_allowance=allowance,
        value=allowance,
    )
    return H7BudgetRecord.create(
        invariant_id=invariant_id,
        category=category,
        operands=operands,
        contributions=(contribution,),
        comparison_normalization=1.0,
        total_allowance=allowance,
    )


def _scalar_objective_budgets(pair: H7LawPairSnapshot) -> dict[str, H7BudgetRecord]:
    assert pair.original.scalar_probe_set is not None
    probes = tuple(
        probe
        for probe in pair.original.scalar_probe_set.probe_pairs
        if probe.action_sha256 == pair.action_sha256
    )
    declarations = {
        "K0_joint_z0_m0": ("local_term", 2.0e-12),
        **{
            term_id: ("local_term", 8.0e-11)
            for term_id in covariance.H7_COMPLETE_LOCAL_TERM_IDS
        },
        covariance.H7_COMPLETE_LOCAL_INVARIANT_ID: (
            "complete_objective",
            8.0e-11,
        ),
        covariance.H7_COMPLETE_MONOLITHIC_INVARIANT_ID: (
            "complete_objective",
            8.0e-11,
        ),
        covariance.H7_POINTWISE_P_SHIFT_INVARIANT_ID: ("density", 2.0e-9),
        covariance.H7_POINTWISE_Q_SHIFT_INVARIANT_ID: ("density", 3.0e-9),
        covariance.H7_POINTWISE_LOG_RATIO_INVARIANT_ID: ("density", 4.0e-9),
        covariance.H7_ENTROPY_SHIFT_INVARIANT_ID: ("density", 8.0e-11),
        covariance.H7_SCALAR_EVIDENCE_INVARIANT_ID: (
            "complete_objective",
            9.0e-11,
        ),
        covariance.H7_SCALAR_POSTERIOR_KL_INVARIANT_ID: (
            "complete_objective",
            1.1e-10,
        ),
    }
    for probe in probes:
        for role, allowance in (
            ("p", 1.2e-9),
            ("q", 1.4e-9),
            ("log_ratio", 1.8e-9),
        ):
            declarations[f"density_probe.{probe.probe_sha256}.{role}"] = (
                "density",
                allowance,
            )
    return {
        invariant_id: _budget(invariant_id, category, allowance)
        for invariant_id, (category, allowance) in declarations.items()
    }


def _matrix_objective_budgets(
    probes: tuple[object, ...],
) -> dict[str, H7BudgetRecord]:
    declarations = {
        "K0_joint_z0_m0": ("local_term", 2.0e-12),
        **{
            term_id: ("local_term", 8.0e-11)
            for term_id in covariance.H7_COMPLETE_LOCAL_TERM_IDS
        },
        covariance.H7_COMPLETE_LOCAL_INVARIANT_ID: (
            "complete_objective",
            8.0e-11,
        ),
        covariance.H7_COMPLETE_MONOLITHIC_INVARIANT_ID: (
            "complete_objective",
            8.0e-11,
        ),
        covariance.H7_POINTWISE_P_SHIFT_INVARIANT_ID: ("density", 2.0e-9),
        covariance.H7_POINTWISE_Q_SHIFT_INVARIANT_ID: ("density", 3.0e-9),
        covariance.H7_POINTWISE_LOG_RATIO_INVARIANT_ID: ("density", 4.0e-9),
        covariance.H7_ENTROPY_SHIFT_INVARIANT_ID: ("density", 8.0e-11),
        **{
            invariant_id: ("vector", 9.0e-11)
            for invariant_id in covariance.H7_MATRIX_SCORER_RESIDUAL_IDS
        },
    }
    for probe in probes:
        roles = (
            ("p", "q", "log_ratio")
            if ".global" in probe.component_id
            else ("p",)
            if probe.component_id.startswith("p.")
            else ("q",)
        )
        for role in roles:
            declarations[f"density_probe.{probe.probe_sha256}.{role}"] = (
                "density",
                1.8e-9,
            )
    return {
        invariant_id: _budget(invariant_id, category, allowance)
        for invariant_id, (category, allowance) in declarations.items()
    }


def test_h7_covariance_consumes_role_evidence_and_positive_grouped_kls() -> None:
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
    posterior_kl = 0.25
    scalar_evidence = H7IndependentH1EvidenceRecord.create(
        fixture_id="h1-v1",
        raw_fixture_sha256=H1_FIXTURE_RAW_SHA256,
        action_sha256=spec.action_sha256,
        normalization_identity_sha256=(
            covariance.H7_INDEPENDENT_H1_NORMALIZATION_IDENTITY_SHA256
        ),
        producer_identity_sha256=(
            covariance.H7_INDEPENDENT_H1_PRODUCER_IDENTITY_SHA256
        ),
        original_log_evidence=(
            original.law_components.grouped_total + posterior_kl
        ),
        transformed_log_evidence=(
            transformed.law_components.grouped_total + posterior_kl
        ),
        original_posterior_kl=posterior_kl,
        transformed_posterior_kl=posterior_kl,
    )
    result = covariance.evaluate_h7_law_pair_covariance(
        pair,
        spec.action,
        original_law_evidence=original,
        transformed_law_evidence=transformed,
        density_probe_pairs=None,
        quadrature_orders=(41, 51),
        budgets_by_invariant=_scalar_objective_budgets(pair),
        scalar_evidence=scalar_evidence,
    )

    expected_initial = original.law_components.positive_kl_terms[0]
    assert result.initial_joint_kl.original_value == expected_initial.value
    assert result.initial_joint_kl.original_value >= 0.0
    assert result.initial_joint_kl.original_value != (
        original.raw_trace_evidence.ordered_factor_values[0]
    )
    assert result.grouped_elbo.original_raw_trace_evidence is (
        original.raw_trace_evidence
    )
    assert result.original_law_evidence_sha256 == original.evidence_sha256
    grouped_original = {
        term.term_id: term.value
        for term in (
            *original.law_components.emission_terms,
            *original.law_components.positive_kl_terms,
        )
    }
    for record in result.local_terms:
        if "_kl[" in record.term_id:
            assert record.original_value == grouped_original[record.term_id]
            assert record.original_value >= 0.0
            assert record.signed_child_ids[0].startswith("+q_")
            assert record.signed_child_ids[1].startswith("-p_")

    assert result.original_complete_local_value == (
        original.law_components.grouped_total
    )
    assert result.grouped_elbo.entropy_diagnostic.additive_in_grouped_elbo is (
        False
    )
    assert "original_factor_trace" not in (
        covariance.evaluate_h7_law_pair_covariance.__annotations__
    )
    with pytest.raises(ValueError):
        replace(
            result.grouped_elbo.positive_kl_terms[0],
            original_value=float("nan"),
        )
    with pytest.raises(ValueError):
        replace(
            result.grouped_elbo.entropy_diagnostic,
            original_entropy=float("inf"),
        )
    with pytest.raises(ValueError):
        replace(result.initial_joint_kl, original_value=float("nan"))
    with pytest.raises(ValueError):
        replace(result.local_terms[0], transformed_value=float("inf"))


@pytest.mark.parametrize(
    "negative_field",
    ("original_posterior_kl", "transformed_posterior_kl"),
)
def test_h7_independent_h1_evidence_rejects_negative_posterior_kl(
    negative_field: str,
) -> None:
    values = {
        "original_posterior_kl": 0.25,
        "transformed_posterior_kl": 0.25,
    }
    values[negative_field] = -0.25

    with pytest.raises(ValueError, match=rf"{negative_field} must be nonnegative"):
        H7IndependentH1EvidenceRecord.create(
            fixture_id="h1-v1",
            raw_fixture_sha256=H1_FIXTURE_RAW_SHA256,
            action_sha256=_sha("negative posterior KL action"),
            normalization_identity_sha256=(
                covariance.H7_INDEPENDENT_H1_NORMALIZATION_IDENTITY_SHA256
            ),
            producer_identity_sha256=(
                covariance.H7_INDEPENDENT_H1_PRODUCER_IDENTITY_SHA256
            ),
            original_log_evidence=0.0,
            transformed_log_evidence=0.0,
            original_posterior_kl=values["original_posterior_kl"],
            transformed_posterior_kl=values["transformed_posterior_kl"],
        )


def test_h7_public_modules_import_in_fresh_orders() -> None:
    for statement in (
        "import vfe4.training; import vfe4.objective",
        "import vfe4.objective; import vfe4.training",
    ):
        completed = subprocess.run(
            [sys.executable, "-c", statement],
            cwd=_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert completed.returncode == 0, completed.stderr


def test_h7_factorized_covariance_uses_registered_law_evidence() -> None:
    spec, pair = _matrix_pair(1)
    arm = _arm("h7-v1", factorized=True)
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
    fixture = parse_h7_fixture_bytes(H7_FIXTURE_PATH.read_bytes())
    probes = tuple(
        probe
        for probe in fixture.density_probe_pairs
        if probe.action_sha256 == spec.action_sha256
        and (
            probe.component_id.startswith("p.")
            or probe.component_id.startswith("q.factorized.")
        )
    )

    result = covariance.evaluate_h7_law_pair_covariance(
        pair,
        spec.action,
        original_law_evidence=original,
        transformed_law_evidence=transformed,
        density_probe_pairs=probes,
        quadrature_orders=(41, 51),
        budgets_by_invariant=_matrix_objective_budgets(probes),
        scalar_evidence=None,
    )

    assert result.factorized_promotion_witness is not None
    assert result.initial_joint_kl.original_value >= 0.0
    assert all(
        record.original_value >= 0.0
        for record in result.local_terms
        if "_kl[" in record.term_id
    )
    assert result.grouped_elbo.original_grouped_total == (
        original.law_components.grouped_total
    )
    assert result.evidence is None
    assert result.posterior_kl is None
    assert (
        result.not_applicable_reason
        == covariance.H7_MATRIX_EVIDENCE_NOT_APPLICABLE_REASON
    )
