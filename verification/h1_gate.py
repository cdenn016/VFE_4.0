"""Fail-closed H1 promotion gate across production and independent paths."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from verification.numpy_oracles import (
    H1EvidenceRecord,
    H1IdentityRecord,
    IndependentTermRecord,
    h1_all_observation_evidences,
    h1_evidence_and_posterior_kl,
    h1_local_diagnostics,
    h1_wrong_recognition_mixture_evidence,
)
from vfe4.artifacts import (
    ArtifactPublicationError,
    build_environment,
    build_provenance,
    publish_run_directory,
)
from vfe4.config import ResolvedConfig, resolve_config
from vfe4.generative import H1GenerativeModel
from vfe4.objective import MonolithicElboResult, evaluate_local_elbo, evaluate_monolithic_elbo
from vfe4.recognition import H1RecognitionLaw
from vfe4.types import ElboTerms, GateResult, GateStatus, InvariantResult
from vfe4.validation import enumerate_source_paths, label_to_index, load_h1_fixture


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPO_ROOT / "vfe4" / "validation" / "fixtures" / "h1_v1.json"
EXPECTED_H1_FIXTURE_SHA256 = (
    "388e38cc8c16d8b5e2c61919c1e712a134d88fb0bbd8ec1f2939b9859c9a583b"
)
_EPSILON = float(np.finfo(np.float64).eps)

H1_MEASUREMENT_NAMES = (
    "monolithic_elbo",
    "local_elbo",
    "evidence_minus_posterior_kl",
)
PAIRWISE_NAMES = (
    "monolithic_vs_local",
    "monolithic_vs_identity",
    "local_vs_identity",
)
TERM_NAMES = (
    "expected_log_emission[0]",
    "expected_log_emission[1]",
    "initial_model_kl",
    "initial_state_kl",
    "model_source_kl[0]",
    "model_source_kl[1]",
    "model_transition_kl[0]",
    "model_transition_kl[1]",
    "state_source_kl[0]",
    "state_source_kl[1]",
    "state_transition_kl[0]",
    "state_transition_kl[1]",
    "joint_recognition_entropy",
    "complete_elbo",
)
_EVIDENCE_LABELS = tuple((first, second) for first in (1, 2, 3) for second in (1, 2, 3))


@dataclass(frozen=True)
class Comparison:
    left: float
    right: float
    left_allowance: float
    right_allowance: float
    rounding: float
    residual: float
    allowance: float
    passed: bool


def pair_comparison(
    left: float, right: float, left_allowance: float, right_allowance: float
) -> Comparison:
    values = (left, right, left_allowance, right_allowance)
    if any(type(value) not in (int, float) or not math.isfinite(float(value)) for value in values):
        raise ValueError("pair comparison values must be finite")
    if left_allowance < 0.0 or right_allowance < 0.0:
        raise ValueError("pair allowances must be nonnegative")
    rounding = 64.0 * _EPSILON * max(1.0, abs(left), abs(right))
    residual = abs(left - right)
    allowance = math.fsum((left_allowance, right_allowance, rounding))
    return Comparison(
        float(left), float(right), float(left_allowance), float(right_allowance),
        rounding, residual, allowance, residual <= allowance,
    )


def _convergence_names() -> tuple[str, ...]:
    names = ["monolithic"]
    names.extend(f"local.{name}" for name in TERM_NAMES)
    names.extend(f"independent.{name}" for name in TERM_NAMES)
    names.extend(
        (
            "identity.posterior_kl",
            "identity.elbo",
            "identity.evidence.probability",
            "identity.evidence.log_probability",
        )
    )
    for first, second in _EVIDENCE_LABELS:
        names.extend((f"evidence.{first}.{second}.probability", f"evidence.{first}.{second}.log_probability"))
    return tuple(names)


CONVERGENCE_NAMES = _convergence_names()
H1_INVARIANT_NAMES = (
    *PAIRWISE_NAMES,
    *TERM_NAMES,
    *(f"convergence.{name}" for name in CONVERGENCE_NAMES),
    "evidence.labels",
    *(f"evidence.{first}.{second}.range" for first, second in _EVIDENCE_LABELS),
    *(f"evidence.{first}.{second}.log_consistency" for first, second in _EVIDENCE_LABELS),
    "evidence.normalization",
    "identity.evidence_probability",
    "identity.evidence_log_probability",
    "identity.decomposition",
    "posterior_kl.nonnegative",
    "elbo.evidence_bound",
    "negative.categorical_source_omission",
    "negative.selected_raw_logit_substitution",
    "negative.recognition_mixture_for_generative_evidence",
)


@dataclass(frozen=True)
class _Evaluation:
    fixture: object
    monolithic: MonolithicElboResult
    local: ElboTerms
    independent: IndependentTermRecord
    identity: H1IdentityRecord
    evidences: tuple[H1EvidenceRecord, ...]
    pairwise: dict[str, Comparison]
    terms: dict[str, Comparison]
    convergence: dict[str, float | None]
    evidence_normalization: dict[str, float | bool]
    negative_controls: dict[str, dict[str, float | bool | str]]
    gate_result: GateResult


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _run_name(timestamp: str, config_hash: str) -> str:
    safe = timestamp.replace("-", "").replace(":", "").replace(".", "")
    return f"verify-h1-{safe}-{config_hash[:12]}"


def _publication_config(config: object) -> ResolvedConfig:
    """Build trustworthy publication metadata from resolved semantic fields."""
    try:
        raw = _raw_from_resolved_fields(config)
        return resolve_config(raw, repo_root=REPO_ROOT)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"resolved config fields cannot be recovered: {exc}") from exc


def _raw_from_resolved_fields(config: object) -> dict[str, object]:
    run = getattr(config, "run")
    artifacts = getattr(config, "artifacts")
    return {
        "schema_version": 1,
        "objective_schema_version": "vfe4-state-elbo-v1",
        "run": {
            "mode": "verify",
            "seed": run.seed,
            "device": "cpu",
            "dtype": "float64",
            "deterministic": True,
        },
        "data": {"kind": "frozen_fixture", "identity": "h1-v1"},
        "model": {
            "horizon": 2,
            "d_z": 1,
            "d_m": 1,
            "vocabulary_size": 3,
            "state_parent_sets": [[0], [0, 1]],
            "model_parent_sets": [[0], [0, 1]],
            "state_source_support": [[0], [0, 1]],
            "model_source_support": [[0], [0, 1]],
            "geometry": "fixed_population_frames",
        },
        "recognition": {
            "conditioning": "smoothing",
            "family": "structured_linear_gaussian_mixture",
            "source_treatment": "exact_enumeration",
        },
        "inference": {"operation": "evaluate_only", "estimator": "deterministic_quadrature"},
        "optimization": {"e_like_update": "none", "m_like_update": "none", "expected_autograd_scope": "none"},
        "validation": {
            "gates": ["H1"],
            "fixture_id": "h1-v1",
            "quadrature_order": 21,
            "convergence_check_order": 17,
            "maximum_convergence_estimate": 1e-9,
        },
        "artifacts": {"run_root": str(artifacts.run_root)},
    }


def _revalidate_config(config: object, canonical: ResolvedConfig) -> None:
    if type(config) is not ResolvedConfig:
        raise ValueError("config must have exact type ResolvedConfig")
    if config != canonical:
        raise ValueError("ResolvedConfig differs from recomputed canonical configuration")
    expected_hash = hashlib.sha256(config.canonical_json.encode("utf-8")).hexdigest()
    if config.config_sha256 != expected_hash:
        raise ValueError("ResolvedConfig hash does not match canonical JSON")


def _validate_fixture(config: ResolvedConfig, fixture: Any) -> None:
    structural = fixture.structural
    expected = config.model
    actual = (
        fixture.fixture_schema_version,
        fixture.fixture_id,
        structural.horizon,
        structural.d_z,
        structural.d_m,
        structural.vocabulary_size,
        structural.state_parent_sets,
        structural.model_parent_sets,
        structural.state_source_support,
        structural.model_source_support,
        fixture.quadrature_order,
        fixture.convergence_check_order,
        fixture.maximum_convergence_estimate,
    )
    required = (
        1,
        config.validation.fixture_id,
        expected.horizon,
        expected.d_z,
        expected.d_m,
        expected.vocabulary_size,
        expected.state_parent_sets,
        expected.model_parent_sets,
        expected.state_source_support,
        expected.model_source_support,
        config.validation.quadrature_order,
        config.validation.convergence_check_order,
        config.validation.maximum_convergence_estimate,
    )
    if actual != required:
        raise ValueError("fixture identity or frozen structural fields do not match config")
    frozen_tags = (
        config.model.geometry,
        config.recognition.conditioning,
        config.recognition.family,
        config.recognition.source_treatment,
        config.inference.operation,
        config.inference.estimator,
        config.optimization.e_like_update,
        config.optimization.m_like_update,
        config.optimization.expected_autograd_scope,
    )
    if frozen_tags != (
        "fixed_population_frames",
        "smoothing",
        "structured_linear_gaussian_mixture",
        "exact_enumeration",
        "evaluate_only",
        "deterministic_quadrature",
        "none",
        "none",
        "none",
    ):
        raise ValueError("geometry, recognition, or evaluation-only tags are incompatible")


def _term_values(record: object) -> dict[str, float]:
    return {
        "expected_log_emission[0]": float(getattr(record, "expected_log_emission")[0]),
        "expected_log_emission[1]": float(getattr(record, "expected_log_emission")[1]),
        "initial_model_kl": float(getattr(record, "initial_model_kl")),
        "initial_state_kl": float(getattr(record, "initial_state_kl")),
        "model_source_kl[0]": float(getattr(record, "model_source_kl")[0]),
        "model_source_kl[1]": float(getattr(record, "model_source_kl")[1]),
        "model_transition_kl[0]": float(getattr(record, "model_transition_kl")[0]),
        "model_transition_kl[1]": float(getattr(record, "model_transition_kl")[1]),
        "state_source_kl[0]": float(getattr(record, "state_source_kl")[0]),
        "state_source_kl[1]": float(getattr(record, "state_source_kl")[1]),
        "state_transition_kl[0]": float(getattr(record, "state_transition_kl")[0]),
        "state_transition_kl[1]": float(getattr(record, "state_transition_kl")[1]),
        "joint_recognition_entropy": float(getattr(record, "joint_recognition_entropy")),
        "complete_elbo": float(getattr(record, "complete_elbo")),
    }


def _term_allowances(record: object) -> tuple[dict[str, float], dict[str, float]]:
    allowance = getattr(record, "allowances")
    totals: dict[str, float] = {}
    convergence = {}
    for name in TERM_NAMES:
        value: Any
        if "[" in name:
            field, index = name[:-1].split("[")
            value = getattr(allowance, field)[int(index)]
        else:
            value = getattr(allowance, name)
        totals[name] = float(value.total)
        convergence[name] = float(value.convergence_estimate)
    return totals, convergence


def _raw_logit_expectation(
    fixture: Any, recognition: H1RecognitionLaw
) -> tuple[float, float]:
    totals = [0.0, 0.0]
    for path in enumerate_source_paths(fixture):
        weight = float(recognition.source_probability(path))
        component = recognition.joint_component(path)
        for time in range(2):
            index = label_to_index(fixture.observation_labels[time], vocabulary_size=3)
            emission = fixture.emissions[time]
            mean = component.mean
            raw = (
                emission.w_z[index] * mean[2 * (time + 1)]
                + emission.w_m[index] * mean[2 * (time + 1) + 1]
                + emission.bias[index]
            )
            totals[time] += weight * float(raw)
    return totals[0], totals[1]


def _negative_controls(
    fixture: Any,
    recognition: H1RecognitionLaw,
    monolithic: MonolithicElboResult,
    identity: H1IdentityRecord,
    config: ResolvedConfig,
    fixture_path: Path,
) -> dict[str, dict[str, float | bool | str]]:
    source_contribution = math.fsum(
        float(recognition.source_probability(path)) * source_log_ratio
        for path, source_log_ratio in zip(
            enumerate_source_paths(fixture), monolithic.component_source_log_ratios
        )
    )
    wrong_source_omission = math.fsum((monolithic.value, -source_contribution))
    source_rounding = 64.0 * _EPSILON * max(
        1.0,
        abs(monolithic.value),
        abs(wrong_source_omission),
        abs(source_contribution),
    )
    source_comparison = pair_comparison(
        monolithic.value,
        wrong_source_omission,
        monolithic.numerical_allowance.total,
        math.fsum((monolithic.numerical_allowance.total, source_rounding)),
    )
    raw = _raw_logit_expectation(fixture, recognition)
    wrong_raw = math.fsum((monolithic.value, -monolithic.expected_log_emission[0], -monolithic.expected_log_emission[1], raw[0], raw[1]))
    raw_residual = abs(wrong_raw - monolithic.value)
    raw_allowance = math.fsum((monolithic.numerical_allowance.total, 64.0 * _EPSILON * max(1.0, abs(wrong_raw), abs(monolithic.value))))
    wrong_evidence = h1_wrong_recognition_mixture_evidence(
        fixture_path,
        fixture.observation_labels,
        quadrature_order=config.validation.quadrature_order,
        convergence_check_order=config.validation.convergence_check_order,
    )
    evidence_rounding = 64.0 * _EPSILON * max(1.0, abs(identity.evidence.probability), abs(wrong_evidence.probability))
    evidence_allowance = math.fsum((identity.evidence.probability_allowance.total, wrong_evidence.probability_allowance.total, evidence_rounding))
    controls = {
        "categorical_source_omission": {
            "domain": "log",
            "correct_value": monolithic.value,
            "wrong_value": wrong_source_omission,
            "residual": source_comparison.residual,
            "allowance": source_comparison.allowance,
            "passed": source_comparison.residual > source_comparison.allowance,
        },
        "selected_raw_logit_substitution": {
            "domain": "log", "residual": raw_residual, "allowance": raw_allowance,
            "passed": raw_residual > raw_allowance,
        },
        "recognition_mixture_for_generative_evidence": {
            "domain": "probability",
            "residual": abs(identity.evidence.probability - wrong_evidence.probability),
            "allowance": evidence_allowance,
            "passed": abs(identity.evidence.probability - wrong_evidence.probability) > evidence_allowance,
        },
    }
    return controls


def _invariant(
    name: str,
    passed: bool,
    value: float | None,
    limit: float | None,
    detail: str,
) -> InvariantResult:
    return InvariantResult(name, passed, value, limit, detail)


def _evaluate(config: ResolvedConfig, fixture_path: Path) -> _Evaluation:
    with torch.no_grad():
        fixture = load_h1_fixture(fixture_path)
        _validate_fixture(config, fixture)
        model = H1GenerativeModel.from_fixture(fixture)
        recognition = H1RecognitionLaw.from_fixture(fixture)
        monolithic = evaluate_monolithic_elbo(
            model,
            recognition,
            quadrature_order=config.validation.quadrature_order,
            convergence_check_order=config.validation.convergence_check_order,
        )
        local = evaluate_local_elbo(
            model,
            recognition,
            quadrature_order=config.validation.quadrature_order,
            convergence_check_order=config.validation.convergence_check_order,
        )
    independent = h1_local_diagnostics(
        fixture_path,
        quadrature_order=config.validation.quadrature_order,
        convergence_check_order=config.validation.convergence_check_order,
    )
    identity = h1_evidence_and_posterior_kl(
        fixture_path,
        quadrature_order=config.validation.quadrature_order,
        convergence_check_order=config.validation.convergence_check_order,
    )
    evidences = h1_all_observation_evidences(
        fixture_path,
        quadrature_order=config.validation.quadrature_order,
        convergence_check_order=config.validation.convergence_check_order,
    )

    measurements = {
        "monolithic_elbo": monolithic.value,
        "local_elbo": local.complete_elbo,
        "evidence_minus_posterior_kl": identity.elbo_from_identity,
    }
    pairwise = {
        "monolithic_vs_local": pair_comparison(monolithic.value, local.complete_elbo, monolithic.numerical_allowance.total, local.allowances.complete_elbo.total),
        "monolithic_vs_identity": pair_comparison(monolithic.value, identity.elbo_from_identity, monolithic.numerical_allowance.total, identity.identity_allowance.total),
        "local_vs_identity": pair_comparison(local.complete_elbo, identity.elbo_from_identity, local.allowances.complete_elbo.total, identity.identity_allowance.total),
    }
    local_values = _term_values(local)
    independent_values = _term_values(independent)
    local_allowance, local_convergence = _term_allowances(local)
    independent_allowance, independent_convergence = _term_allowances(independent)
    terms = {
        name: pair_comparison(local_values[name], independent_values[name], local_allowance[name], independent_allowance[name])
        for name in TERM_NAMES
    }
    convergence: dict[str, float | None] = {
        "monolithic": monolithic.numerical_allowance.convergence_estimate
    }
    convergence.update({f"local.{name}": value for name, value in local_convergence.items()})
    convergence.update({f"independent.{name}": value for name, value in independent_convergence.items()})
    convergence["identity.posterior_kl"] = identity.posterior_kl_allowance.convergence_estimate
    convergence["identity.elbo"] = identity.identity_allowance.convergence_estimate
    convergence["identity.evidence.probability"] = (
        identity.evidence.probability_allowance.convergence_estimate
    )
    convergence["identity.evidence.log_probability"] = (
        identity.evidence.log_probability_allowance.convergence_estimate
    )
    actual_labels = tuple(record.observation_labels for record in evidences)
    labels_valid = (
        actual_labels == _EVIDENCE_LABELS
        and len(actual_labels) == len(_EVIDENCE_LABELS)
        and len(set(actual_labels)) == len(_EVIDENCE_LABELS)
    )
    evidence_slots: list[H1EvidenceRecord | None] = list(
        evidences[: len(_EVIDENCE_LABELS)]
    )
    while len(evidence_slots) < len(_EVIDENCE_LABELS):
        evidence_slots.append(None)
    for labels, record in zip(_EVIDENCE_LABELS, evidence_slots):
        convergence[f"evidence.{labels[0]}.{labels[1]}.probability"] = (
            None if record is None else record.probability_allowance.convergence_estimate
        )
        convergence[f"evidence.{labels[0]}.{labels[1]}.log_probability"] = (
            None
            if record is None
            else record.log_probability_allowance.convergence_estimate
        )
    if tuple(convergence) != CONVERGENCE_NAMES:
        raise ValueError("convergence registry inventory mismatch")

    probability_sum = math.fsum(record.probability for record in evidences)
    sum_rounding = 64.0 * _EPSILON * max(1.0, abs(probability_sum))
    sum_allowance = math.fsum(record.probability_allowance.total for record in evidences) + sum_rounding
    evidence_normalization = {
        "probability_sum": probability_sum,
        "residual": abs(probability_sum - 1.0),
        "rounding": sum_rounding,
        "allowance": sum_allowance,
        "passed": abs(probability_sum - 1.0) <= sum_allowance,
    }
    controls = _negative_controls(
        fixture, recognition, monolithic, identity, config, fixture_path
    )
    invariants: list[InvariantResult] = []
    for name in PAIRWISE_NAMES:
        comparison = pairwise[name]
        invariants.append(_invariant(name, comparison.passed, comparison.residual, comparison.allowance, "absolute pair residual <= pair-local allowance"))
    for name in TERM_NAMES:
        comparison = terms[name]
        invariants.append(_invariant(name, comparison.passed, comparison.residual, comparison.allowance, "homologous term residual <= term-local allowance"))
    for name in CONVERGENCE_NAMES:
        value = convergence[name]
        invariants.append(_invariant(f"convergence.{name}", value is not None and math.isfinite(value) and 0.0 <= value <= config.validation.maximum_convergence_estimate, value, config.validation.maximum_convergence_estimate, "finite nonnegative convergence estimate <= frozen maximum"))
    invariants.append(_invariant("evidence.labels", labels_valid, float(len(set(actual_labels))), 9.0, "labels must be exact unique lexicographic pairs"))
    for labels, record in zip(_EVIDENCE_LABELS, evidence_slots):
        invariants.append(_invariant(f"evidence.{labels[0]}.{labels[1]}.range", record is not None and 0.0 < record.probability <= 1.0, None if record is None else record.probability, 1.0, "probability must lie in (0,1]"))
    for labels, record in zip(_EVIDENCE_LABELS, evidence_slots):
        if record is None:
            invariants.append(_invariant(f"evidence.{labels[0]}.{labels[1]}.log_consistency", False, None, None, "evidence record is unavailable"))
            continue
        if record.probability > 0.0:
            expected_log = math.log(record.probability)
            residual = abs(record.log_probability - expected_log)
        else:
            expected_log = record.log_probability
            residual = math.fsum((1.0, abs(record.log_probability)))
        rounding = 64.0 * _EPSILON * max(1.0, abs(expected_log), abs(record.log_probability))
        allowance = math.fsum((record.log_probability_allowance.total, rounding))
        invariants.append(_invariant(f"evidence.{labels[0]}.{labels[1]}.log_consistency", residual <= allowance, residual, allowance, "log probability must agree with its probability"))
    invariants.append(_invariant("evidence.normalization", bool(evidence_normalization["passed"]), float(evidence_normalization["residual"]), float(evidence_normalization["allowance"]), "probability-domain evidence sum"))
    selected = (
        {record.observation_labels: record for record in evidences}.get(
            identity.evidence.observation_labels, identity.evidence
        )
        if labels_valid
        else identity.evidence
    )
    prob_cmp = pair_comparison(selected.probability, identity.evidence.probability, selected.probability_allowance.total, identity.evidence.probability_allowance.total)
    log_cmp = pair_comparison(selected.log_probability, identity.evidence.log_probability, selected.log_probability_allowance.total, identity.evidence.log_probability_allowance.total)
    invariants.append(_invariant("identity.evidence_probability", prob_cmp.passed, prob_cmp.residual, prob_cmp.allowance, "identity evidence probability matches nine-evidence table"))
    invariants.append(_invariant("identity.evidence_log_probability", log_cmp.passed, log_cmp.residual, log_cmp.allowance, "identity log evidence matches nine-evidence table"))
    identity_expected = identity.evidence.log_probability - identity.posterior_kl
    identity_cmp = pair_comparison(identity.elbo_from_identity, identity_expected, identity.identity_allowance.total, math.fsum((identity.evidence.log_probability_allowance.total, identity.posterior_kl_allowance.total)))
    invariants.append(_invariant("identity.decomposition", identity_cmp.passed, identity_cmp.residual, identity_cmp.allowance, "ELBO = log evidence - posterior KL"))
    kl_limit = -identity.posterior_kl_allowance.total
    invariants.append(_invariant("posterior_kl.nonnegative", identity.posterior_kl >= kl_limit, identity.posterior_kl, kl_limit, "posterior KL >= -allowance"))
    elbo_limit = math.fsum(
        (
            identity.evidence.log_probability,
            identity.identity_allowance.total,
            identity.evidence.log_probability_allowance.total,
        )
    )
    invariants.append(_invariant("elbo.evidence_bound", identity.elbo_from_identity <= elbo_limit, identity.elbo_from_identity, elbo_limit, "ELBO <= log evidence + identity/log allowances"))
    for control_name in (
        "categorical_source_omission",
        "selected_raw_logit_substitution",
        "recognition_mixture_for_generative_evidence",
    ):
        control = controls[control_name]
        invariants.append(_invariant(f"negative.{control_name}", bool(control["passed"]), float(control["residual"]), float(control["allowance"]), "strict negative-control residual > own allowance"))
    if tuple(item.name for item in invariants) != H1_INVARIANT_NAMES:
        raise ValueError("H1 invariant inventory mismatch")
    status = GateStatus.PASS if all(item.passed for item in invariants) else GateStatus.FAIL
    gate_result = GateResult(
        gate="H1",
        status=status,
        fixture_id="h1-v1",
        residual=max(item.residual for item in pairwise.values()),
        calibrated_allowance=max(item.allowance for item in pairwise.values()),
        measurements=measurements,
        invariants=tuple(invariants),
        obligations=(),
    )
    if tuple(gate_result.measurements) != H1_MEASUREMENT_NAMES:
        raise ValueError("H1 measurement inventory mismatch")
    return _Evaluation(fixture, monolithic, local, independent, identity, evidences, pairwise, terms, convergence, evidence_normalization, controls, gate_result)


def _inconclusive(reason: str) -> GateResult:
    return GateResult(
        gate="H1",
        status=GateStatus.INCONCLUSIVE,
        fixture_id="h1-v1",
        residual=None,
        calibrated_allowance=None,
        measurements={name: None for name in H1_MEASUREMENT_NAMES},
        invariants=tuple(InvariantResult(name, False, None, None, "unavailable") for name in H1_INVARIANT_NAMES),
        obligations=(reason,),
    )


def _allowance_payload(value: Any) -> dict[str, float]:
    return {
        "convergence_estimate": float(value.convergence_estimate),
        "rounding_allowance": float(value.rounding_allowance),
        "total": float(value.total),
    }


def _term_payload(record: object) -> dict[str, object]:
    allowances = getattr(record, "allowances")
    return {
        "expected_log_emission": getattr(record, "expected_log_emission"),
        "initial_model_kl": getattr(record, "initial_model_kl"),
        "initial_state_kl": getattr(record, "initial_state_kl"),
        "model_source_kl": getattr(record, "model_source_kl"),
        "model_transition_kl": getattr(record, "model_transition_kl"),
        "state_source_kl": getattr(record, "state_source_kl"),
        "state_transition_kl": getattr(record, "state_transition_kl"),
        "joint_recognition_entropy": getattr(record, "joint_recognition_entropy"),
        "complete_elbo": getattr(record, "complete_elbo"),
        "allowances": {
            "expected_log_emission": tuple(
                _allowance_payload(item) for item in allowances.expected_log_emission
            ),
            "initial_model_kl": _allowance_payload(allowances.initial_model_kl),
            "initial_state_kl": _allowance_payload(allowances.initial_state_kl),
            "model_source_kl": tuple(
                _allowance_payload(item) for item in allowances.model_source_kl
            ),
            "model_transition_kl": tuple(
                _allowance_payload(item) for item in allowances.model_transition_kl
            ),
            "state_source_kl": tuple(
                _allowance_payload(item) for item in allowances.state_source_kl
            ),
            "state_transition_kl": tuple(
                _allowance_payload(item) for item in allowances.state_transition_kl
            ),
            "joint_recognition_entropy": _allowance_payload(
                allowances.joint_recognition_entropy
            ),
            "complete_elbo": _allowance_payload(allowances.complete_elbo),
        },
    }


def _evidence_payload(record: H1EvidenceRecord) -> dict[str, object]:
    return {
        "observation_labels": record.observation_labels,
        "probability": record.probability,
        "log_probability": record.log_probability,
        "probability_allowance": _allowance_payload(record.probability_allowance),
        "log_probability_allowance": _allowance_payload(record.log_probability_allowance),
    }


def _validation_payload(evaluation: _Evaluation | None, result: GateResult) -> dict[str, object]:
    if evaluation is None:
        return {"gate_result": result, "computation": "unavailable"}
    identity = evaluation.identity
    return {
        "gate_result": result,
        "monolithic": {
            "value": evaluation.monolithic.value,
            "component_values": evaluation.monolithic.component_values,
            "component_gaussian_log_ratios": evaluation.monolithic.component_gaussian_log_ratios,
            "component_source_log_ratios": evaluation.monolithic.component_source_log_ratios,
            "component_emission_values": evaluation.monolithic.component_emission_values,
            "expected_log_emission": evaluation.monolithic.expected_log_emission,
            "quadrature_order": evaluation.monolithic.quadrature_order,
            "convergence_check_order": evaluation.monolithic.convergence_check_order,
            "numerical_allowance": _allowance_payload(evaluation.monolithic.numerical_allowance),
        },
        "local_terms": _term_payload(evaluation.local),
        "independent_terms": _term_payload(evaluation.independent),
        "identity": {
            "evidence": _evidence_payload(identity.evidence),
            "posterior_kl": identity.posterior_kl,
            "elbo_from_identity": identity.elbo_from_identity,
            "quadrature_order": identity.quadrature_order,
            "convergence_check_order": identity.convergence_check_order,
            "posterior_kl_allowance": _allowance_payload(identity.posterior_kl_allowance),
            "identity_allowance": _allowance_payload(identity.identity_allowance),
        },
        "evidences": tuple(_evidence_payload(record) for record in evaluation.evidences),
        "convergence_registry": evaluation.convergence,
        "pairwise_residuals": {name: value.residual for name, value in evaluation.pairwise.items()},
        "pairwise_allowances": {name: value.allowance for name, value in evaluation.pairwise.items()},
        "pairwise_comparisons": evaluation.pairwise,
        "term_comparisons": evaluation.terms,
        "evidence_normalization": evaluation.evidence_normalization,
        "negative_controls": evaluation.negative_controls,
    }


def _config_payload(config: ResolvedConfig) -> dict[str, object]:
    payload = json.loads(config.canonical_json)
    payload["config_sha256"] = config.config_sha256
    return payload


def _capture_fixture(path: Path) -> tuple[bytes | None, str | None]:
    try:
        content = path.read_bytes()
    except OSError:
        return None, None
    return content, hashlib.sha256(content).hexdigest()


def run_h1(config: ResolvedConfig) -> tuple[GateResult, Path]:
    """Evaluate H1 and publish a complete run, catching computation only."""
    started = _utc_now()
    fixture_bytes, fixture_observed_sha256 = _capture_fixture(FIXTURE_PATH)
    canonical = _publication_config(config)
    evaluation: _Evaluation | None = None
    try:
        _revalidate_config(config, canonical)
        if fixture_bytes is None or fixture_observed_sha256 is None:
            raise ValueError("H1 fixture is unavailable or unreadable")
        if fixture_observed_sha256 != EXPECTED_H1_FIXTURE_SHA256:
            raise ValueError("H1 fixture content does not match its preregistered SHA-256")
        with tempfile.TemporaryDirectory(prefix="vfe4-h1-fixture-") as temporary:
            fixture_snapshot = Path(temporary) / "h1_v1.json"
            fixture_snapshot.write_bytes(fixture_bytes)
            candidate = _evaluate(canonical, fixture_snapshot)
        evaluation = candidate
        result = candidate.gate_result
    except Exception as exc:
        result = _inconclusive(f"H1 computation requires resolution: {exc}")
    ended = _utc_now()
    provenance = build_provenance(
        repo_root=REPO_ROOT,
        fixture_expected_sha256=EXPECTED_H1_FIXTURE_SHA256,
        fixture_observed_sha256=fixture_observed_sha256,
        config=canonical,
        started_utc=started,
        ended_utc=ended,
        gate_state=result.status.value,
    )
    payloads = {
        "config.json": _config_payload(canonical),
        "provenance.json": provenance,
        "environment.json": build_environment(canonical),
        "validation/h1.json": _validation_payload(evaluation, result),
    }
    run_dir = publish_run_directory(
        canonical.artifacts.run_root,
        _run_name(started, canonical.config_sha256),
        payloads,
    )
    return result, run_dir


__all__ = [
    "ArtifactPublicationError",
    "CONVERGENCE_NAMES",
    "H1_INVARIANT_NAMES",
    "H1_MEASUREMENT_NAMES",
    "PAIRWISE_NAMES",
    "TERM_NAMES",
    "pair_comparison",
    "run_h1",
]
