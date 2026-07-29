"""Registry-backed H7 evidence derived only from exact complete laws."""

from __future__ import annotations

import math
import weakref
from dataclasses import dataclass
from typing import Literal, final

import torch

from vfe4.generative.source_priors import (
    FixedSourceFactorContext,
    FixedSourcePrior,
    NormalizedSourceFactor,
)
from vfe4.training.arms import BuiltArm
from vfe4.training.h7_assembly import (
    H7FixedSourceAssemblyReceipt,
    require_h7_fixed_source_assembly,
)
from vfe4.types.h7 import (
    H7GLPlus2Action,
    H7GroupedElboRecord,
    H7GroupedElboTermRecord,
    H7LawPairSnapshot,
    H7NonadditiveEntropyDiagnostic,
    H7RawFactorTraceEvidence,
    H7ScalarReplayAction,
    H7TensorActionSnapshot,
    H7TrialSpec,
    canonical_h7_bytes,
    h7_owned_sha256,
)
from vfe4.types.h7_law import (
    H7LawComponents,
    H7SourceAssemblyProfile,
)

from .h7_law_components import build_h7_law_components
from .language_elbo import (
    CompleteLanguageELBOFactorTrace,
    ExactSourceMixtureLaw,
    H7AuthenticatedEvaluation,
    adapt_h7_raw_factor_trace_evidence,
    capture_h7_complete_language_elbo,
    require_h7_complete_factor_trace,
)


H7LawRole = Literal["original", "transformed"]
_DERIVATION_ROUTE = (
    "vfe4.objective.h7_law_components.build_h7_law_components",
    "vfe4.objective.h7_law_evidence.capture_h7_law_evaluation",
)
_EVIDENCE_HASH_DOMAIN = "vfe4.h7.law-evaluation-evidence.v1"
_EXPECTATION_HASH_DOMAIN = "vfe4.h7.law-derived-expectation.v1"
_STRUCTURE_HASH_DOMAIN = "vfe4.h7.law-derived-expectation-structure.v1"
_FLOAT64_EPSILON = math.ulp(1.0)


def _selected_law(
    law_pair: H7LawPairSnapshot,
    role: H7LawRole,
):
    if role == "original":
        return law_pair.original
    if role == "transformed":
        return law_pair.transformed
    raise ValueError("H7 law evidence role must be original or transformed")


def _validate_trial_pair_action(
    trial_spec: H7TrialSpec,
    law_pair: H7LawPairSnapshot,
    action: H7TensorActionSnapshot,
) -> None:
    if type(trial_spec) is not H7TrialSpec:
        raise ValueError("trial_spec must be an exact H7TrialSpec")
    if type(law_pair) is not H7LawPairSnapshot:
        raise ValueError("law_pair must be an exact H7LawPairSnapshot")
    if type(action) not in (H7ScalarReplayAction, H7GLPlus2Action):
        raise ValueError("action must be an exact owned H7 action")
    trial_spec.__post_init__()
    law_pair.__post_init__()
    action.__post_init__()
    if (
        trial_spec.action is not action
        or trial_spec.action_sha256 != action.action_sha256
        or law_pair.action_sha256 != action.action_sha256
        or trial_spec.fixture_id != law_pair.original.fixture_id
        or trial_spec.fixture_id != law_pair.transformed.fixture_id
    ):
        raise ValueError(
            "H7 trial, law pair, action, or fixture binding changed"
        )


def _validate_source_profile(
    profile: H7SourceAssemblyProfile,
    receipt: H7FixedSourceAssemblyReceipt,
) -> None:
    profile.__post_init__()
    receipt.__post_init__()
    if profile.fixture_id != receipt.fixture_id:
        raise ValueError("H7 law and assembly fixture identities disagree")
    observed = receipt.source_rows
    if tuple((row[0], row[1], row[2]) for row in observed) != tuple(
        (row.partition, row.receiver_t, row.support) for row in profile.rows
    ):
        raise ValueError("H7 law and live assembly source supports disagree")
    for profile_row, receipt_row in zip(
        profile.rows,
        observed,
        strict=True,
    ):
        probabilities = receipt_row[3]
        if len(probabilities) != len(profile_row.probabilities):
            raise ValueError(
                "H7 law and live assembly source probabilities disagree"
            )
        allowance = (
            64.0
            * _FLOAT64_EPSILON
            * max(1, len(profile_row.probabilities))
        )
        if any(
            not math.isclose(
                actual,
                expected,
                rel_tol=0.0,
                abs_tol=allowance,
            )
            for actual, expected in zip(
                probabilities,
                profile_row.probabilities,
                strict=True,
            )
        ):
            raise ValueError(
                "H7 law and live assembly source probabilities disagree"
            )


def _recognition_family(law: object) -> str:
    origin = law.recognition.origin_family
    if origin == "structured_full_block":
        return "structured_full_spd"
    if origin == "factorized_diagonal_within_fiber":
        return "population_factorized_block_spd"
    raise ValueError("unsupported H7 recognition origin family")


class _H7LawDerivedExpectation:
    """Private H6 protocol adapter whose values come from H7 law components."""

    def __init__(
        self,
        *,
        trial_spec: H7TrialSpec,
        law_pair: H7LawPairSnapshot,
        action: H7TensorActionSnapshot,
        role: H7LawRole,
        arm: BuiltArm,
        assembly_receipt: H7FixedSourceAssemblyReceipt,
        components: H7LawComponents,
    ) -> None:
        law = _selected_law(law_pair, role)
        self.horizon = 2
        self.evaluation_method = "deterministic_quadrature"
        self.source_law = ExactSourceMixtureLaw.create(
            endpoint_config=arm.config
        )
        self.recognition_family = _recognition_family(law)
        self.recognition_conditioning = "smoothing"
        self.ordered_slots = tuple(term.slot for term in components.raw_terms)
        self.structure_sha256 = h7_owned_sha256(
            _STRUCTURE_HASH_DOMAIN,
            {
                "trial_sha256": trial_spec.trial_sha256,
                "law_pair_sha256": law_pair.law_pair_sha256,
                "action_sha256": action.action_sha256,
                "role": role,
                "law_snapshot_sha256": law.snapshot_sha256,
                "source_profile_sha256": (
                    components.source_assembly_profile.profile_sha256
                ),
                "assembly_sha256": assembly_receipt.assembly_sha256,
            },
        )
        if type(arm.model.source_prior) is not FixedSourcePrior:
            raise ValueError("H7 law expectation requires a fixed source prior")
        source_prior = arm.model.source_prior
        self._source_factors: dict[
            tuple[str, int],
            NormalizedSourceFactor,
        ] = {
            ("model_source", receiver_t): (
                source_prior.model_source_log_probs(receiver_t=receiver_t)
            )
            for receiver_t in (1, 2)
        }
        self._source_factors.update(
            {
                ("state_source", receiver_t): (
                    source_prior.state_source_log_probs(
                        receiver_t=receiver_t
                    )
                )
                for receiver_t in (1, 2)
            }
        )
        raw_terms = {term.slot: term for term in components.raw_terms}
        self._values = {
            slot: torch.tensor(term.value, dtype=torch.float64)
            for slot, term in raw_terms.items()
        }
        self._factor_identities = {
            slot: (
                self._source_factors[slot].factor_identity_sha256
                if slot[0] in ("model_source", "state_source")
                else term.raw_term_sha256
            )
            for slot, term in raw_terms.items()
        }
        self.expectation_identity_sha256 = h7_owned_sha256(
            _EXPECTATION_HASH_DOMAIN,
            {
                "trial_sha256": trial_spec.trial_sha256,
                "law_pair_sha256": law_pair.law_pair_sha256,
                "action_sha256": action.action_sha256,
                "role": role,
                "law_snapshot_sha256": law.snapshot_sha256,
                "components_sha256": components.components_sha256,
                "assembly_sha256": assembly_receipt.assembly_sha256,
                "structure_sha256": self.structure_sha256,
                "factor_identities": tuple(
                    (slot, self._factor_identities[slot])
                    for slot in self.ordered_slots
                ),
            },
        )
        total = self._values[self.ordered_slots[0]]
        for slot in self.ordered_slots[1:]:
            total = total + self._values[slot]
        self._total = total

    def contribution(self, partition: str, receiver_t: int) -> torch.Tensor:
        try:
            return self._values[(partition, receiver_t)]
        except KeyError as exc:
            raise ValueError("unknown H7 law-derived raw slot") from exc

    def normalized_factor_identity(
        self,
        partition: str,
        receiver_t: int,
    ) -> str:
        try:
            return self._factor_identities[(partition, receiver_t)]
        except KeyError as exc:
            raise ValueError("unknown H7 law-derived factor identity") from exc

    def normalized_source_factor(
        self,
        partition: str,
        receiver_t: int,
    ) -> NormalizedSourceFactor:
        try:
            return self._source_factors[(partition, receiver_t)]
        except KeyError as exc:
            raise ValueError("unknown H7 law-derived source factor") from exc

    def source_factor_context(
        self,
        partition: str,
        receiver_t: int,
    ) -> FixedSourceFactorContext:
        if partition not in ("model_source", "state_source"):
            raise ValueError("H7 source context requires a source partition")
        return FixedSourceFactorContext(
            "model" if partition == "model_source" else "state",
            receiver_t,
        )

    def independently_accumulated_total(self) -> torch.Tensor:
        return self._total


@final
@dataclass(
    frozen=True,
    slots=True,
    init=False,
    eq=False,
    weakref_slot=True,
)
class H7LawEvaluationEvidence:
    """Capture-only evidence for one role of one exact H7 law pair."""

    trial_spec: H7TrialSpec
    law_pair: H7LawPairSnapshot
    action: H7TensorActionSnapshot
    role: H7LawRole
    fixture_id: Literal["h1-v1", "h7-v1"]
    law_snapshot_sha256: str
    raw_fixture_sha256: str
    arm: BuiltArm
    assembly_receipt: H7FixedSourceAssemblyReceipt
    law_components: H7LawComponents
    authenticated_evaluation: H7AuthenticatedEvaluation
    factor_trace: CompleteLanguageELBOFactorTrace
    raw_trace_evidence: H7RawFactorTraceEvidence
    derivation_route: tuple[str, str]
    quadrature_order: Literal[51]
    evidence_sha256: str

    def __init__(self) -> None:
        raise TypeError(
            "H7LawEvaluationEvidence is capture-only; use "
            "capture_h7_law_evaluation"
        )

    def __copy__(self) -> H7LawEvaluationEvidence:
        raise TypeError("H7 law evaluation evidence copy is forbidden")

    def __deepcopy__(
        self,
        memo: dict[int, object],
    ) -> H7LawEvaluationEvidence:
        del memo
        raise TypeError("H7 law evaluation evidence deepcopy is forbidden")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError("H7 law evaluation evidence pickle is forbidden")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "trial_sha256": self.trial_spec.trial_sha256,
            "law_pair_sha256": self.law_pair.law_pair_sha256,
            "action_sha256": self.action.action_sha256,
            "role": self.role,
            "fixture_id": self.fixture_id,
            "law_snapshot_sha256": self.law_snapshot_sha256,
            "raw_fixture_sha256": self.raw_fixture_sha256,
            "assembly_sha256": self.assembly_receipt.assembly_sha256,
            "components_sha256": self.law_components.components_sha256,
            "authenticated_evaluation_sha256": (
                self.authenticated_evaluation.attestation_sha256
            ),
            "factor_trace_sha256": self.factor_trace.trace_sha256,
            "raw_trace_evidence_sha256": (
                self.raw_trace_evidence.evidence_sha256
            ),
            "derivation_route": self.derivation_route,
            "quadrature_order": self.quadrature_order,
        }

    def __post_init__(
        self,
        _assembly_validator=require_h7_fixed_source_assembly,
        _source_profile_validator=_validate_source_profile,
    ) -> None:
        _validate_trial_pair_action(
            self.trial_spec,
            self.law_pair,
            self.action,
        )
        law = _selected_law(self.law_pair, self.role)
        receipt = _assembly_validator(self.arm)
        self.law_components.__post_init__()
        self.authenticated_evaluation.__post_init__()
        self.factor_trace.__post_init__()
        self.raw_trace_evidence.__post_init__()
        if (
            self.fixture_id != law.fixture_id
            or self.fixture_id != receipt.fixture_id
            or self.law_snapshot_sha256 != law.snapshot_sha256
            or self.raw_fixture_sha256 != law.raw_fixture_sha256
            or self.assembly_receipt is not receipt
            or self.law_components.complete_law_snapshot_sha256
            != law.snapshot_sha256
            or self.factor_trace.authenticated_evaluation
            is not self.authenticated_evaluation
            or self.raw_trace_evidence.trace_sha256
            != self.factor_trace.trace_sha256
            or self.raw_trace_evidence.ordered_factor_values
            != tuple(term.value for term in self.law_components.raw_terms)
            or self.derivation_route != _DERIVATION_ROUTE
            or self.quadrature_order != 51
        ):
            raise ValueError("H7 law evaluation evidence binding changed")
        _source_profile_validator(
            self.law_components.source_assembly_profile,
            receipt,
        )
        if abs(
            self.raw_trace_evidence.total_value
            - self.law_components.raw_total
        ) > self.law_components.equality_tolerance:
            raise ValueError(
                "H7 authenticated raw total disagrees with its exact law"
            )
        if self.evidence_sha256 != h7_owned_sha256(
            _EVIDENCE_HASH_DOMAIN,
            self.canonical_payload(),
        ):
            raise ValueError("H7 law evaluation evidence hash changed")


def _build_h7_law_evaluation_api():
    assembly_validator = require_h7_fixed_source_assembly
    law_component_builder = build_h7_law_components
    source_profile_validator = _validate_source_profile
    expectation_factory = _H7LawDerivedExpectation
    elbo_capturer = capture_h7_complete_language_elbo
    factor_trace_adapter = require_h7_complete_factor_trace
    raw_trace_adapter = adapt_h7_raw_factor_trace_evidence
    registry: dict[
        int,
        tuple[
            weakref.ReferenceType[H7LawEvaluationEvidence],
            tuple[object, ...],
        ],
    ] = {}

    def issuance_snapshot(
        evidence: H7LawEvaluationEvidence,
    ) -> tuple[object, ...]:
        return (
            id(evidence.trial_spec),
            id(evidence.law_pair),
            id(evidence.action),
            id(evidence.arm),
            id(evidence.assembly_receipt),
            id(evidence.law_components),
            id(evidence.authenticated_evaluation),
            id(evidence.factor_trace),
            id(evidence.raw_trace_evidence),
            canonical_h7_bytes(evidence.canonical_payload()),
            evidence.evidence_sha256,
        )

    def issue(**values: object) -> H7LawEvaluationEvidence:
        evidence = object.__new__(H7LawEvaluationEvidence)
        for name, value in values.items():
            object.__setattr__(evidence, name, value)
        object.__setattr__(
            evidence,
            "evidence_sha256",
            h7_owned_sha256(
                _EVIDENCE_HASH_DOMAIN,
                evidence.canonical_payload(),
            ),
        )
        evidence.__post_init__()
        identity = id(evidence)

        def remove(
            reference: weakref.ReferenceType[H7LawEvaluationEvidence],
        ) -> None:
            current = registry.get(identity)
            if current is not None and current[0] is reference:
                registry.pop(identity, None)

        reference = weakref.ref(evidence, remove)
        current = registry.get(identity)
        if current is not None and current[0]() is not None:
            raise RuntimeError("H7 law evidence identity was already issued")
        registry[identity] = (reference, issuance_snapshot(evidence))
        return evidence

    def capture(
        *,
        trial_spec: H7TrialSpec,
        law_pair: H7LawPairSnapshot,
        action: H7TensorActionSnapshot,
        role: H7LawRole,
        arm: BuiltArm,
    ) -> H7LawEvaluationEvidence:
        _validate_trial_pair_action(trial_spec, law_pair, action)
        law = _selected_law(law_pair, role)
        receipt = assembly_validator(arm)
        components = law_component_builder(law)
        source_profile_validator(
            components.source_assembly_profile,
            receipt,
        )
        expected_arm_family = (
            "structured"
            if law.recognition.origin_family == "structured_full_block"
            else "factorized"
        )
        if arm.config.recognition_family != expected_arm_family:
            raise ValueError(
                "H7 law recognition family disagrees with its assembly arm"
            )
        expectation = expectation_factory(
            trial_spec=trial_spec,
            law_pair=law_pair,
            action=action,
            role=role,
            arm=arm,
            assembly_receipt=receipt,
            components=components,
        )
        authenticated = elbo_capturer(
            arm,
            expectation,
        )
        trace = factor_trace_adapter(authenticated)
        raw_evidence = raw_trace_adapter(trace)
        if (
            authenticated.expectation_identity_sha256
            != expectation.expectation_identity_sha256
            or authenticated.expectation_structure_sha256
            != expectation.structure_sha256
            or raw_evidence.ordered_slots
            != tuple(term.slot for term in components.raw_terms)
            or raw_evidence.ordered_factor_values
            != tuple(term.value for term in components.raw_terms)
        ):
            raise ValueError(
                "H7 law-derived expectation changed during H6 assembly"
            )
        return issue(
            trial_spec=trial_spec,
            law_pair=law_pair,
            action=action,
            role=role,
            fixture_id=law.fixture_id,
            law_snapshot_sha256=law.snapshot_sha256,
            raw_fixture_sha256=law.raw_fixture_sha256,
            arm=arm,
            assembly_receipt=receipt,
            law_components=components,
            authenticated_evaluation=authenticated,
            factor_trace=trace,
            raw_trace_evidence=raw_evidence,
            derivation_route=_DERIVATION_ROUTE,
            quadrature_order=51,
        )

    def require(
        value: object,
        *,
        trial_spec: H7TrialSpec,
        law_pair: H7LawPairSnapshot,
        action: H7TensorActionSnapshot,
        role: H7LawRole,
    ) -> H7LawEvaluationEvidence:
        if type(value) is not H7LawEvaluationEvidence:
            raise ValueError(
                "H7 law evaluation requires exact capture-issued evidence"
            )
        current = registry.get(id(value))
        if current is None or current[0]() is not value:
            raise ValueError(
                "H7 law evaluation evidence is not live and registered"
            )
        if value.role != role:
            raise ValueError("H7 law evidence role binding changed")
        if value.trial_spec is not trial_spec:
            raise ValueError("H7 law evidence trial binding changed")
        if value.law_pair is not law_pair:
            raise ValueError("H7 law evidence law-pair binding changed")
        if value.action is not action:
            raise ValueError("H7 law evidence action binding changed")
        value.__post_init__()
        if issuance_snapshot(value) != current[1]:
            raise ValueError(
                "H7 law evaluation evidence changed after issuance"
            )
        return value

    return capture, require


(
    capture_h7_law_evaluation,
    require_h7_law_evaluation,
) = _build_h7_law_evaluation_api()
del _build_h7_law_evaluation_api


def build_h7_grouped_elbo_record(
    *,
    trial_spec: H7TrialSpec,
    law_pair: H7LawPairSnapshot,
    action: H7TensorActionSnapshot,
    original_evidence: H7LawEvaluationEvidence,
    transformed_evidence: H7LawEvaluationEvidence,
) -> H7GroupedElboRecord:
    """Build the pair-level grouped ELBO only from registered role evidence."""

    original = require_h7_law_evaluation(
        original_evidence,
        trial_spec=trial_spec,
        law_pair=law_pair,
        action=action,
        role="original",
    )
    transformed = require_h7_law_evaluation(
        transformed_evidence,
        trial_spec=trial_spec,
        law_pair=law_pair,
        action=action,
        role="transformed",
    )
    original_components = original.law_components
    transformed_components = transformed.law_components
    emission_terms = tuple(
        H7GroupedElboTermRecord.create(
            term_id=original_term.term_id,
            semantics=original_term.semantics,
            elbo_sign=original_term.elbo_sign,
            original_value=original_term.value,
            transformed_value=transformed_term.value,
            original_complete_law_operand_sha256s=(
                original_term.complete_law_operand_sha256s
            ),
            transformed_complete_law_operand_sha256s=(
                transformed_term.complete_law_operand_sha256s
            ),
            covariance_residual=abs(
                transformed_term.value - original_term.value
            ),
        )
        for original_term, transformed_term in zip(
            original_components.emission_terms,
            transformed_components.emission_terms,
            strict=True,
        )
    )
    positive_kl_terms = tuple(
        H7GroupedElboTermRecord.create(
            term_id=original_term.term_id,
            semantics=original_term.semantics,
            elbo_sign=original_term.elbo_sign,
            original_value=original_term.value,
            transformed_value=transformed_term.value,
            original_complete_law_operand_sha256s=(
                original_term.complete_law_operand_sha256s
            ),
            transformed_complete_law_operand_sha256s=(
                transformed_term.complete_law_operand_sha256s
            ),
            covariance_residual=abs(
                transformed_term.value - original_term.value
            ),
        )
        for original_term, transformed_term in zip(
            original_components.positive_kl_terms,
            transformed_components.positive_kl_terms,
            strict=True,
        )
    )
    original_entropy = float(
        math.fsum(
            slot.value for slot in original_components.entropy_ownership.slots
        )
    )
    transformed_entropy = float(
        math.fsum(
            slot.value
            for slot in transformed_components.entropy_ownership.slots
        )
    )
    original_equality_residual = abs(
        original.raw_trace_evidence.total_value
        - original_components.grouped_total
    )
    transformed_equality_residual = abs(
        transformed.raw_trace_evidence.total_value
        - transformed_components.grouped_total
    )
    if (
        original_equality_residual
        > 2.0 * original_components.equality_tolerance
        or transformed_equality_residual
        > 2.0 * transformed_components.equality_tolerance
    ):
        raise ValueError(
            "H7 authenticated raw and grouped law totals disagree"
        )
    return H7GroupedElboRecord.create(
        representation="expected_emission_minus_positive_kl_v1",
        law_pair_sha256=law_pair.law_pair_sha256,
        original_raw_trace_evidence=original.raw_trace_evidence,
        transformed_raw_trace_evidence=transformed.raw_trace_evidence,
        emission_terms=emission_terms,
        positive_kl_terms=positive_kl_terms,
        entropy_diagnostic=H7NonadditiveEntropyDiagnostic.create(
            original_entropy=original_entropy,
            transformed_entropy=transformed_entropy,
            additive_in_grouped_elbo=False,
            covariance_residual=abs(
                transformed_entropy - original_entropy
            ),
        ),
        original_grouped_total=original_components.grouped_total,
        transformed_grouped_total=transformed_components.grouped_total,
        original_raw_grouped_equality_residual=(
            original_equality_residual
        ),
        transformed_raw_grouped_equality_residual=(
            transformed_equality_residual
        ),
    )


__all__ = [
    "H7LawEvaluationEvidence",
    "H7LawRole",
    "build_h7_grouped_elbo_record",
    "capture_h7_law_evaluation",
    "require_h7_law_evaluation",
]
