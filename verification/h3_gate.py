"""Fail-closed H3 structured-posterior adequacy gate."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal, cast

import numpy as np
import torch

from vfe4.config import ResolvedConfig, resolve_config
from vfe4.generative import H3GenerativeModel
from vfe4.inference import optimize_h3_arm
from vfe4.objective import evaluate_h3_elbo
from vfe4.recognition import H3VariationalGaussian
from vfe4.types.h3 import (
    H3ArmResult,
    H3Fixture,
    H3FixtureHashes,
    H3GateResult,
    H3RecognitionFamily,
)
from vfe4.types.results import GateStatus, InvariantResult
from vfe4.validation.h3_fixture import (
    H3_COUPLED_FIXTURE_PATH,
    H3_ZERO_CONTROL_FIXTURE_PATH,
    parse_h3_fixture_bytes,
    validate_independent_control,
)
from verification.h3_budget import (
    C,
    EPS,
    SOLVER_ALLOWANCE_NATS,
    allowance_is_decisive,
    four_operand_identity_allowance,
    pair_allowance,
    scalar_allowance,
    three_operand_identity_allowance,
)
from verification.numpy_oracles import (
    H3PosteriorOracleEvaluation,
    evaluate_h3_posterior_oracle,
    reverse_kl_to_oracle,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
_DIMENSION = 4
_FAMILIES: tuple[H3RecognitionFamily, H3RecognitionFamily] = (
    "structured_full_spd",
    "fine_factorized_diagonal",
)
_FIXTURE_KEYS = ("coupled", "zero_control")

H3_INVARIANT_NAMES = (
    "fixture_hashes_match",
    "independent_control_contract",
    "coupled_frozen_reference_agreement",
    "zero_frozen_reference_agreement",
    "pytorch_numpy_canonical_agreement",
    "posterior_condition_envelope",
    "all_arms_converged",
    "coupled_oracle_gap_minimum",
    "all_invariant_allowances_decisive",
    "coupled_structured_fraction_resolved",
    "coupled_factorized_analytic_gap",
    "coupled_structured_elbo_kl_identity",
    "coupled_factorized_elbo_kl_identity",
    "coupled_delta_adequacy_identity",
    "zero_structured_kl",
    "zero_factorized_kl",
    "zero_delta_adequacy",
    "zero_structured_elbo_kl_identity",
    "zero_factorized_elbo_kl_identity",
)


ThresholdEligibility = Literal["PASS_ELIGIBLE", "FAIL", "INCONCLUSIVE"]


@dataclass(frozen=True)
class H3ThresholdDecision:
    name: str
    operands: tuple[float, float]
    favorable_margin_formula: str
    favorable_direction: Literal["greater_than_positive_allowance"]
    margin: float
    allowance: float
    lower_boundary: float
    upper_boundary: float
    eligibility: ThresholdEligibility
    obligation: str | None

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise ValueError("threshold name must be nonempty")
        if type(self.operands) is not tuple or len(self.operands) != 2:
            raise ValueError("threshold operands must contain two values")
        for name, value in (
            ("operands[0]", self.operands[0]),
            ("operands[1]", self.operands[1]),
            ("margin", self.margin),
            ("allowance", self.allowance),
            ("lower_boundary", self.lower_boundary),
            ("upper_boundary", self.upper_boundary),
        ):
            _finite(value, name)
        if self.allowance < 0.0:
            raise ValueError("threshold allowance must be nonnegative")
        if self.lower_boundary != -self.allowance:
            raise ValueError("threshold lower boundary must equal -allowance")
        if self.upper_boundary != self.allowance:
            raise ValueError("threshold upper boundary must equal allowance")
        if self.favorable_direction != "greater_than_positive_allowance":
            raise ValueError("threshold direction must match the frozen rule")
        if self.eligibility not in ("PASS_ELIGIBLE", "FAIL", "INCONCLUSIVE"):
            raise ValueError("threshold eligibility is invalid")
        if self.eligibility == "INCONCLUSIVE":
            if type(self.obligation) is not str or not self.obligation:
                raise ValueError("indecisive thresholds require an obligation")
        elif self.obligation is not None:
            raise ValueError("decisive thresholds cannot retain an obligation")


@dataclass(frozen=True)
class H3GateEvaluation:
    result: H3GateResult
    fixture_hashes: H3FixtureHashes
    oracle_by_fixture: Mapping[str, H3PosteriorOracleEvaluation]
    arms_by_fixture: Mapping[str, Mapping[str, H3ArmResult]]
    comparisons: Mapping[str, object]
    allowances_by_invariant: Mapping[str, object]
    validation_payload: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.result, H3GateResult):
            raise ValueError("result must be an H3GateResult")
        if not isinstance(self.fixture_hashes, H3FixtureHashes):
            raise ValueError("fixture_hashes must be H3FixtureHashes")
        oracle = _freeze_object_mapping(
            self.oracle_by_fixture,
            H3PosteriorOracleEvaluation,
            "oracle_by_fixture",
        )
        arms = _freeze_arm_mapping(self.arms_by_fixture)
        comparisons = _freeze_json_mapping(self.comparisons, "comparisons")
        allowances = _freeze_json_mapping(
            self.allowances_by_invariant,
            "allowances_by_invariant",
        )
        payload = _freeze_json_mapping(
            self.validation_payload,
            "validation_payload",
        )
        if set(allowances) != set(self.result.allowances_by_invariant):
            raise ValueError("evaluation and result allowance names must match")
        object.__setattr__(self, "oracle_by_fixture", oracle)
        object.__setattr__(self, "arms_by_fixture", arms)
        object.__setattr__(self, "comparisons", comparisons)
        object.__setattr__(self, "allowances_by_invariant", allowances)
        object.__setattr__(self, "validation_payload", payload)


def evaluate_h3(
    config: ResolvedConfig,
    *,
    coupled_fixture_bytes: bytes | None = None,
    zero_control_fixture_bytes: bytes | None = None,
) -> H3GateEvaluation:
    """Evaluate the complete fail-closed H3 promotion gate."""

    checked_config = _validate_config(config)
    coupled_bytes = _capture_bytes(
        coupled_fixture_bytes,
        H3_COUPLED_FIXTURE_PATH,
        "coupled",
    )
    zero_bytes = _capture_bytes(
        zero_control_fixture_bytes,
        H3_ZERO_CONTROL_FIXTURE_PATH,
        "zero-control",
    )
    observed_coupled = hashlib.sha256(coupled_bytes).hexdigest()
    observed_zero = hashlib.sha256(zero_bytes).hexdigest()
    fixture_hashes = H3FixtureHashes(
        coupled_expected_sha256=checked_config.h3.coupled_expected_sha256,
        coupled_observed_sha256=observed_coupled,
        zero_control_expected_sha256=(
            checked_config.h3.zero_control_expected_sha256
        ),
        zero_control_observed_sha256=observed_zero,
    )
    fixture_invariant = InvariantResult(
        name="fixture_hashes_match",
        passed=fixture_hashes.coupled_matches and fixture_hashes.zero_control_matches,
        value=None,
        limit=None,
        detail="both raw fixture SHA-256 digests match the typed H3 profile",
    )
    if not fixture_invariant.passed:
        return _early_inconclusive(
            checked_config,
            fixture_hashes,
            captured_bytes=(coupled_bytes, zero_bytes),
            invariants=(fixture_invariant,),
            obligation="restore exact raw H3 fixture bytes",
        )

    try:
        coupled_fixture = parse_h3_fixture_bytes(
            coupled_bytes,
            expected_fixture_id=checked_config.h3.coupled_fixture_id,
        )
        zero_fixture = parse_h3_fixture_bytes(
            zero_bytes,
            expected_fixture_id=checked_config.h3.zero_control_fixture_id,
        )
        validate_independent_control(coupled_fixture, zero_fixture)
    except (ValueError, RuntimeError) as exc:
        control = InvariantResult(
            name="independent_control_contract",
            passed=False,
            value=None,
            limit=None,
            detail=f"fixture parsing/control validation failed: {type(exc).__name__}",
        )
        return _early_inconclusive(
            checked_config,
            fixture_hashes,
            captured_bytes=(coupled_bytes, zero_bytes),
            invariants=(fixture_invariant, control),
            obligation="restore the strict independent H3 control contract",
        )
    control = InvariantResult(
        name="independent_control_contract",
        passed=True,
        value=None,
        limit=None,
        detail="zero control differs only in preregistered transition/observation data",
    )

    oracles: dict[str, H3PosteriorOracleEvaluation] = {}
    reference_invariants: list[InvariantResult] = []
    reference_allowances: dict[str, object] = {}
    for key, raw_bytes, fixture in (
        ("coupled", coupled_bytes, coupled_fixture),
        ("zero_control", zero_bytes, zero_fixture),
    ):
        invariant_name = f"{key.replace('_control', '')}_frozen_reference_agreement"
        if key == "zero_control":
            invariant_name = "zero_frozen_reference_agreement"
        try:
            oracle = evaluate_h3_posterior_oracle(
                raw_bytes,
                expected_fixture_id=fixture.fixture_id,
            )
            oracles[key] = oracle
            record = _reference_agreement_record(fixture, oracle)
            reference_allowances[invariant_name] = record
            reference_invariants.append(
                _comparison_invariant(invariant_name, record)
            )
        except (ValueError, RuntimeError) as exc:
            reference_invariants.append(
                InvariantResult(
                    name=invariant_name,
                    passed=False,
                    value=None,
                    limit=None,
                    detail=f"frozen-reference evaluation failed: {type(exc).__name__}",
                )
            )
            allowances = {
                name: reference_allowances[name]
                for name in reference_allowances
            }
            return _early_inconclusive(
                checked_config,
                fixture_hashes,
                captured_bytes=(coupled_bytes, zero_bytes),
                invariants=(
                    fixture_invariant,
                    control,
                    *reference_invariants[:-1],
                ),
                obligation=f"resolve {key} frozen-reference agreement",
                oracle_by_fixture=oracles,
                allowances_by_invariant=allowances,
            )

    models = {
        "coupled": H3GenerativeModel.from_fixture(coupled_fixture),
        "zero_control": H3GenerativeModel.from_fixture(zero_fixture),
    }
    canonical_elements: list[dict[str, object]] = []
    for key in _FIXTURE_KEYS:
        canonical_elements.extend(
            _canonical_agreement_elements(key, models[key], oracles[key])
        )
    canonical_record = _element_group_record(canonical_elements)
    canonical_invariant = _comparison_invariant(
        "pytorch_numpy_canonical_agreement",
        canonical_record,
    )
    agreement_invariants = (
        fixture_invariant,
        control,
        *reference_invariants,
        canonical_invariant,
    )
    agreement_allowances = {
        **reference_allowances,
        "pytorch_numpy_canonical_agreement": canonical_record,
    }
    agreement_obligations = tuple(
        f"resolve {item.name}"
        for item in agreement_invariants[2:]
        if not item.passed
    )
    agreement_decisiveness = tuple(
        name
        for name, record in agreement_allowances.items()
        if not bool(cast(Mapping[str, object], record)["decisive"])
    )
    if agreement_obligations or agreement_decisiveness:
        obligations = agreement_obligations or tuple(
            f"reduce nondecisive allowance for {name}"
            for name in agreement_decisiveness
        )
        return _early_inconclusive(
            checked_config,
            fixture_hashes,
            captured_bytes=(coupled_bytes, zero_bytes),
            invariants=agreement_invariants,
            obligation=obligations[0],
            additional_obligations=obligations[1:],
            oracle_by_fixture=oracles,
            comparisons={"canonical_agreement": canonical_record},
            allowances_by_invariant=agreement_allowances,
        )

    arms: dict[str, dict[str, H3ArmResult]] = {}
    for key in _FIXTURE_KEYS:
        arms[key] = {}
        for family in checked_config.h3.recognition_families:
            arms[key][family] = optimize_h3_arm(
                models[key],
                family,
                checked_config.h3.common_initialization,
                checked_config.h3.optimizer,
            )

    envelope_passed, envelope_detail = _posterior_envelope(
        oracles,
        arms,
        checked_config,
    )
    envelope = InvariantResult(
        name="posterior_condition_envelope",
        passed=envelope_passed,
        value=None,
        limit=None,
        detail=envelope_detail,
    )
    if not envelope.passed:
        return _early_inconclusive(
            checked_config,
            fixture_hashes,
            captured_bytes=(coupled_bytes, zero_bytes),
            invariants=(*agreement_invariants, envelope),
            obligation=(
                "restore every H3 posterior and terminal law to the frozen envelope"
            ),
            oracle_by_fixture=oracles,
            arms_by_fixture=arms,
            comparisons={"canonical_agreement": canonical_record},
            allowances_by_invariant=agreement_allowances,
        )
    converged = all(
        result.converged
        for by_family in arms.values()
        for result in by_family.values()
    )
    convergence = InvariantResult(
        name="all_arms_converged",
        passed=converged,
        value=None,
        limit=None,
        detail="all four fresh H3 L-BFGS arms satisfy the frozen terminal rule",
    )
    if not converged:
        return _early_inconclusive(
            checked_config,
            fixture_hashes,
            captured_bytes=(coupled_bytes, zero_bytes),
            invariants=(*agreement_invariants, envelope, convergence),
            obligation="obtain finite converged evidence for every H3 arm",
            oracle_by_fixture=oracles,
            arms_by_fixture=arms,
            comparisons={"canonical_agreement": canonical_record},
            allowances_by_invariant=agreement_allowances,
        )

    return _complete_evaluation(
        checked_config,
        fixture_hashes,
        captured_bytes=(coupled_bytes, zero_bytes),
        fixtures={"coupled": coupled_fixture, "zero_control": zero_fixture},
        models=models,
        oracles=oracles,
        arms=arms,
        prefix_invariants=(*agreement_invariants, envelope, convergence),
        agreement_allowances=agreement_allowances,
    )


def _posterior_envelope(
    oracles: Mapping[str, H3PosteriorOracleEvaluation],
    arms: Mapping[str, Mapping[str, H3ArmResult]],
    config: ResolvedConfig,
) -> tuple[bool, str]:
    if config.h3 is None:
        raise ValueError("H3 envelope requires an H3 profile")
    decision = config.h3.decision
    laws: list[tuple[str, np.ndarray, np.ndarray]] = []
    for key, oracle in oracles.items():
        laws.append((f"{key}.oracle", oracle.mean, oracle.precision))
    for key, by_family in arms.items():
        for family, result in by_family.items():
            if result.terminal_mean is None or result.terminal_precision is None:
                return False, f"{key}.{family} has no finite terminal law"
            laws.append(
                (
                    f"{key}.{family}",
                    np.asarray(result.terminal_mean, dtype=np.float64),
                    np.asarray(result.terminal_precision, dtype=np.float64),
                )
            )
    for name, mean, precision in laws:
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(precision)):
            return False, f"{name} is nonfinite"
        if not np.array_equal(precision, precision.T):
            return False, f"{name} precision is not exactly symmetric"
        eigenvalues = np.linalg.eigvalsh(precision)
        minimum = float(eigenvalues[0])
        maximum = float(eigenvalues[-1])
        condition = maximum / minimum if minimum > 0.0 else float("inf")
        mean_norm = float(np.max(np.abs(mean)))
        if not (
            minimum >= decision.minimum_precision_eigenvalue
            and maximum <= decision.maximum_precision_eigenvalue
            and condition <= decision.maximum_precision_condition_number
            and mean_norm <= decision.maximum_mean_infinity_norm
        ):
            return False, f"{name} lies outside the frozen H3 envelope"
    return True, "all exact and terminal H3 laws lie inside the frozen envelope"


def _complete_evaluation(
    config: ResolvedConfig,
    fixture_hashes: H3FixtureHashes,
    *,
    captured_bytes: tuple[bytes, bytes],
    fixtures: Mapping[str, H3Fixture],
    models: Mapping[str, H3GenerativeModel],
    oracles: Mapping[str, H3PosteriorOracleEvaluation],
    arms: Mapping[str, Mapping[str, H3ArmResult]],
    prefix_invariants: tuple[InvariantResult, ...],
    agreement_allowances: Mapping[str, object],
) -> H3GateEvaluation:
    if config.h3 is None:
        raise ValueError("complete H3 evaluation requires an H3 profile")
    del fixtures  # Raw-byte identity and the independent parsers own fixture evidence.
    terminal = _terminal_records(models, oracles, arms)
    gap = float(oracles["coupled"].analytic_factorized_reverse_kl)
    gap_record = _oracle_gap_scalar(oracles["coupled"])
    constant_half = _exact_scalar_record("minimum_gap_constant", 0.50)
    coupled_structured = terminal["coupled"]["structured_full_spd"]
    coupled_factorized = terminal["coupled"]["fine_factorized_diagonal"]
    zero_structured = terminal["zero_control"]["structured_full_spd"]
    zero_factorized = terminal["zero_control"]["fine_factorized_diagonal"]

    gap_threshold_record = _pair_element(
        "coupled_oracle_gap_minimum",
        gap_record,
        constant_half,
        decisiveness_scale=gap,
    )
    gap_margin = gap - 0.50
    gap_threshold_record["signed_margin"] = gap_margin
    gap_decision = _threshold_decision(
        gap_margin,
        float(gap_threshold_record["final_allowance"]),
        config.h3.coupled_gap_inconclusive_obligation,
        name="coupled_oracle_gap_minimum",
        operands=(gap, 0.50),
        formula="G-0.50",
    )
    gap_threshold_record["passed"] = gap_decision.eligibility != "FAIL"
    gap_threshold_record["threshold_eligibility"] = gap_decision.eligibility

    one_percent_gap = 0.01 * gap
    one_percent_record = _scalar_record(
        "one_percent_coupled_gap",
        one_percent_gap,
        absolute_sum=0.01
        * float(
            oracles["coupled"].diagnostics[
                "analytic_factorized_reverse_kl_absolute_summand_accumulation"
            ]
        ),
        kappas=(float(oracles["coupled"].diagnostics["kappa_2"]),),
        optimized=False,
    )
    structured_threshold_record = _pair_element(
        "coupled_structured_fraction_resolved",
        one_percent_record,
        cast(Mapping[str, object], coupled_structured["kl_scalar"]),
        decisiveness_scale=gap,
    )
    structured_margin = one_percent_gap - float(coupled_structured["kl"])
    structured_threshold_record["signed_margin"] = structured_margin
    structured_decision = _threshold_decision(
        structured_margin,
        float(structured_threshold_record["final_allowance"]),
        config.h3.structured_closure_inconclusive_obligation,
        name="coupled_structured_fraction_resolved",
        operands=(one_percent_gap, float(coupled_structured["kl"])),
        formula="0.01*G-KL_cs",
    )
    structured_threshold_record["passed"] = (
        structured_decision.eligibility != "FAIL"
    )
    structured_threshold_record[
        "threshold_eligibility"
    ] = structured_decision.eligibility

    factorized_gap_record = _pair_element(
        "coupled_factorized_analytic_gap",
        cast(Mapping[str, object], coupled_factorized["kl_scalar"]),
        gap_record,
        decisiveness_scale=gap,
    )
    zero_constant = _exact_scalar_record("exact_zero", 0.0)
    zero_structured_record = _pair_element(
        "zero_structured_kl",
        cast(Mapping[str, object], zero_structured["kl_scalar"]),
        zero_constant,
        decisiveness_scale=gap,
    )
    zero_factorized_record = _pair_element(
        "zero_factorized_kl",
        cast(Mapping[str, object], zero_factorized["kl_scalar"]),
        zero_constant,
        decisiveness_scale=gap,
    )
    zero_delta_record = _pair_element(
        "zero_delta_adequacy",
        cast(Mapping[str, object], zero_structured["elbo_scalar"]),
        cast(Mapping[str, object], zero_factorized["elbo_scalar"]),
        decisiveness_scale=gap,
    )

    identities = {
        "coupled_structured_elbo_kl_identity": _three_identity_record(
            "coupled_structured_elbo_kl_identity",
            _oracle_evidence_scalar(oracles["coupled"]),
            cast(Mapping[str, object], coupled_structured["elbo_scalar"]),
            cast(Mapping[str, object], coupled_structured["kl_scalar"]),
            decisiveness_scale=gap,
        ),
        "coupled_factorized_elbo_kl_identity": _three_identity_record(
            "coupled_factorized_elbo_kl_identity",
            _oracle_evidence_scalar(oracles["coupled"]),
            cast(Mapping[str, object], coupled_factorized["elbo_scalar"]),
            cast(Mapping[str, object], coupled_factorized["kl_scalar"]),
            decisiveness_scale=gap,
        ),
        "zero_structured_elbo_kl_identity": _three_identity_record(
            "zero_structured_elbo_kl_identity",
            _oracle_evidence_scalar(oracles["zero_control"]),
            cast(Mapping[str, object], zero_structured["elbo_scalar"]),
            cast(Mapping[str, object], zero_structured["kl_scalar"]),
            decisiveness_scale=gap,
        ),
        "zero_factorized_elbo_kl_identity": _three_identity_record(
            "zero_factorized_elbo_kl_identity",
            _oracle_evidence_scalar(oracles["zero_control"]),
            cast(Mapping[str, object], zero_factorized["elbo_scalar"]),
            cast(Mapping[str, object], zero_factorized["kl_scalar"]),
            decisiveness_scale=gap,
        ),
    }
    coupled_delta_record = _four_identity_record(
        "coupled_delta_adequacy_identity",
        cast(Mapping[str, object], coupled_structured["elbo_scalar"]),
        cast(Mapping[str, object], coupled_factorized["elbo_scalar"]),
        cast(Mapping[str, object], coupled_factorized["kl_scalar"]),
        cast(Mapping[str, object], coupled_structured["kl_scalar"]),
        decisiveness_scale=gap,
    )

    allowances: dict[str, object] = {
        "coupled_frozen_reference_agreement": agreement_allowances[
            "coupled_frozen_reference_agreement"
        ],
        "zero_frozen_reference_agreement": agreement_allowances[
            "zero_frozen_reference_agreement"
        ],
        "pytorch_numpy_canonical_agreement": agreement_allowances[
            "pytorch_numpy_canonical_agreement"
        ],
        "coupled_oracle_gap_minimum": gap_threshold_record,
        "coupled_structured_fraction_resolved": structured_threshold_record,
        "coupled_factorized_analytic_gap": factorized_gap_record,
        "coupled_structured_elbo_kl_identity": identities[
            "coupled_structured_elbo_kl_identity"
        ],
        "coupled_factorized_elbo_kl_identity": identities[
            "coupled_factorized_elbo_kl_identity"
        ],
        "coupled_delta_adequacy_identity": coupled_delta_record,
        "zero_structured_kl": zero_structured_record,
        "zero_factorized_kl": zero_factorized_record,
        "zero_delta_adequacy": zero_delta_record,
        "zero_structured_elbo_kl_identity": identities[
            "zero_structured_elbo_kl_identity"
        ],
        "zero_factorized_elbo_kl_identity": identities[
            "zero_factorized_elbo_kl_identity"
        ],
    }
    nondecisive = tuple(
        name
        for name, record in allowances.items()
        if not bool(cast(Mapping[str, object], record)["decisive"])
    )
    decisive_invariant = InvariantResult(
        name="all_invariant_allowances_decisive",
        passed=not nondecisive,
        value=None,
        limit=None,
        detail="every allowance is below one percent of its named local scale",
    )

    invariant_by_name: dict[str, InvariantResult] = {
        item.name: item for item in prefix_invariants
    }
    invariant_by_name["coupled_oracle_gap_minimum"] = _threshold_invariant(
        gap_decision
    )
    invariant_by_name["all_invariant_allowances_decisive"] = decisive_invariant
    invariant_by_name[
        "coupled_structured_fraction_resolved"
    ] = _threshold_invariant(structured_decision)
    invariant_by_name["coupled_factorized_analytic_gap"] = _comparison_invariant(
        "coupled_factorized_analytic_gap", factorized_gap_record
    )
    for name, record in identities.items():
        invariant_by_name[name] = _comparison_invariant(name, record)
    invariant_by_name["coupled_delta_adequacy_identity"] = _comparison_invariant(
        "coupled_delta_adequacy_identity", coupled_delta_record
    )
    invariant_by_name["zero_structured_kl"] = _comparison_invariant(
        "zero_structured_kl", zero_structured_record
    )
    invariant_by_name["zero_factorized_kl"] = _comparison_invariant(
        "zero_factorized_kl", zero_factorized_record
    )
    invariant_by_name["zero_delta_adequacy"] = _comparison_invariant(
        "zero_delta_adequacy", zero_delta_record
    )
    invariants = tuple(invariant_by_name[name] for name in H3_INVARIANT_NAMES)

    equality_names = (
        "coupled_factorized_analytic_gap",
        "coupled_structured_elbo_kl_identity",
        "coupled_factorized_elbo_kl_identity",
        "coupled_delta_adequacy_identity",
        "zero_structured_kl",
        "zero_factorized_kl",
        "zero_delta_adequacy",
        "zero_structured_elbo_kl_identity",
        "zero_factorized_elbo_kl_identity",
    )
    equality_failures = tuple(
        name for name in equality_names if not invariant_by_name[name].passed
    )
    upstream = tuple(
        f"reduce nondecisive allowance for {name}" for name in nondecisive
    )
    status, obligations = _status_and_obligations(
        upstream_obligations=upstream,
        threshold_decisions=(gap_decision, structured_decision),
        equality_failures=equality_failures,
    )
    resolved_fraction = (gap - float(coupled_structured["kl"])) / gap
    measurements = {
        "coupled_oracle_gap": gap,
        "coupled_structured_kl": float(coupled_structured["kl"]),
        "coupled_factorized_kl": float(coupled_factorized["kl"]),
        "zero_structured_kl": float(zero_structured["kl"]),
        "zero_factorized_kl": float(zero_factorized["kl"]),
        "coupled_structured_elbo": float(coupled_structured["elbo"]),
        "coupled_factorized_elbo": float(coupled_factorized["elbo"]),
        "zero_structured_elbo": float(zero_structured["elbo"]),
        "zero_factorized_elbo": float(zero_factorized["elbo"]),
        "coupled_elbo_delta": float(coupled_structured["elbo"])
        - float(coupled_factorized["elbo"]),
        "zero_elbo_delta": float(zero_structured["elbo"])
        - float(zero_factorized["elbo"]),
        "resolved_fraction": resolved_fraction,
        "coupled_gap_margin": gap_margin,
        "structured_resolution_margin": structured_margin,
    }
    result = H3GateResult(
        gate="H3",
        coupled_fixture_id="h3-coupled-v1",
        zero_control_fixture_id="h3-zero-control-v1",
        status=status,
        measurements=measurements,
        invariants=invariants,
        allowances_by_invariant=allowances,
        obligations=obligations,
    )
    comparisons = {
        "terminal": terminal,
        "threshold_decisions": {
            gap_decision.name: _threshold_payload(gap_decision),
            structured_decision.name: _threshold_payload(structured_decision),
        },
    }
    payload = _validation_payload(
        config,
        result,
        fixture_hashes,
        captured_bytes=captured_bytes,
        oracle_by_fixture=oracles,
        arms_by_fixture=arms,
        comparisons=comparisons,
        allowances_by_invariant=allowances,
    )
    return H3GateEvaluation(
        result=result,
        fixture_hashes=fixture_hashes,
        oracle_by_fixture=oracles,
        arms_by_fixture=arms,
        comparisons=comparisons,
        allowances_by_invariant=allowances,
        validation_payload=payload,
    )


def _threshold_decision(
    margin: float,
    allowance: float,
    obligation: str,
    *,
    name: str = "threshold",
    operands: tuple[float, float] = (0.0, 0.0),
    formula: str = "left-right",
) -> H3ThresholdDecision:
    """Apply the frozen signed three-way decision at exact boundaries."""

    checked_margin = _finite(margin, "margin")
    checked_allowance = _finite(allowance, "allowance")
    if checked_allowance < 0.0:
        raise ValueError("allowance must be nonnegative")
    if type(obligation) is not str or not obligation:
        raise ValueError("obligation must be nonempty")
    if checked_margin > checked_allowance:
        eligibility: ThresholdEligibility = "PASS_ELIGIBLE"
        open_obligation = None
    elif checked_margin < -checked_allowance:
        eligibility = "FAIL"
        open_obligation = None
    else:
        eligibility = "INCONCLUSIVE"
        open_obligation = obligation
    return H3ThresholdDecision(
        name=name,
        operands=operands,
        favorable_margin_formula=formula,
        favorable_direction="greater_than_positive_allowance",
        margin=checked_margin,
        allowance=checked_allowance,
        lower_boundary=-checked_allowance,
        upper_boundary=checked_allowance,
        eligibility=eligibility,
        obligation=open_obligation,
    )


def _status_and_obligations(
    *,
    upstream_obligations: tuple[str, ...] = (),
    threshold_decisions: tuple[H3ThresholdDecision, ...] = (),
    equality_failures: tuple[str, ...] = (),
) -> tuple[GateStatus, tuple[str, ...]]:
    """Resolve H3 status with integrity first and finite failures second."""

    _require_string_tuple(upstream_obligations, "upstream_obligations")
    _require_string_tuple(equality_failures, "equality_failures")
    if type(threshold_decisions) is not tuple or not all(
        isinstance(item, H3ThresholdDecision) for item in threshold_decisions
    ):
        raise ValueError("threshold_decisions must contain threshold records")
    if upstream_obligations:
        return GateStatus.INCONCLUSIVE, _deduplicate(upstream_obligations)
    if equality_failures or any(
        item.eligibility == "FAIL" for item in threshold_decisions
    ):
        return GateStatus.FAIL, ()
    threshold_obligations = tuple(
        cast(str, item.obligation)
        for item in threshold_decisions
        if item.eligibility == "INCONCLUSIVE"
    )
    if threshold_obligations:
        return GateStatus.INCONCLUSIVE, _deduplicate(threshold_obligations)
    return GateStatus.PASS, ()


def h3_validation_payload(evaluation: H3GateEvaluation) -> dict[str, object]:
    """Return a fresh mutable JSON payload owned by the caller."""

    if not isinstance(evaluation, H3GateEvaluation):
        raise ValueError("evaluation must be an H3GateEvaluation")
    thawed = _thaw_json_like(evaluation.validation_payload)
    if not isinstance(thawed, dict):
        raise RuntimeError("frozen H3 payload must thaw to a dictionary")
    return thawed


def _validate_config(config: object) -> ResolvedConfig:
    if not isinstance(config, ResolvedConfig):
        raise ValueError("config must be a ResolvedConfig")
    if config.validation.gates != ("H1", "H2", "H3") or config.h3 is None:
        raise ValueError("H3 evaluation requires the exact H1/H2/H3 prefix")
    h3 = config.h3
    if h3.recognition_families != _FAMILIES:
        raise ValueError("H3 recognition family order is invalid")
    if h3.optimization_operation != "maximize_direct_h3_elbo_lbfgs":
        raise ValueError("H3 optimization operation is invalid")
    if h3.expected_autograd_scope != "h3_recognition_only":
        raise ValueError("H3 autograd scope is invalid")
    if h3.solver_allowance_nats != SOLVER_ALLOWANCE_NATS:
        raise ValueError("H3 solver allowance is invalid")
    if h3.threshold_decision_rule != "signed_margin_three_way":
        raise ValueError("H3 threshold rule is invalid")
    if h3.minimum_resolved_fraction != 0.99:
        raise ValueError("H3 resolved fraction is invalid")
    canonical_raw = json.loads(config.canonical_json)
    reproduced = resolve_config(canonical_raw, repo_root=REPO_ROOT)
    if (
        reproduced.canonical_json != config.canonical_json
        or reproduced.config_sha256 != config.config_sha256
    ):
        raise ValueError("H3 config canonical identity is invalid")
    return config


def _capture_bytes(value: bytes | None, path: Path, name: str) -> bytes:
    captured = path.read_bytes() if value is None else value
    if type(captured) is not bytes:
        raise ValueError(f"{name} fixture must be raw bytes")
    return captured


def _early_inconclusive(
    config: ResolvedConfig,
    fixture_hashes: H3FixtureHashes,
    *,
    captured_bytes: tuple[bytes, bytes],
    invariants: tuple[InvariantResult, ...],
    obligation: str,
    additional_obligations: tuple[str, ...] = (),
    oracle_by_fixture: Mapping[str, H3PosteriorOracleEvaluation] | None = None,
    arms_by_fixture: Mapping[str, Mapping[str, H3ArmResult]] | None = None,
    comparisons: Mapping[str, object] | None = None,
    allowances_by_invariant: Mapping[str, object] | None = None,
) -> H3GateEvaluation:
    allowances = dict(allowances_by_invariant or {})
    obligations = _deduplicate((obligation, *additional_obligations))
    result = H3GateResult(
        gate="H3",
        coupled_fixture_id="h3-coupled-v1",
        zero_control_fixture_id="h3-zero-control-v1",
        status=GateStatus.INCONCLUSIVE,
        measurements={
            "coupled_fixture_byte_count": float(len(captured_bytes[0])),
            "zero_control_fixture_byte_count": float(len(captured_bytes[1])),
        },
        invariants=invariants,
        allowances_by_invariant=allowances,
        obligations=obligations,
    )
    payload = _validation_payload(
        config,
        result,
        fixture_hashes,
        captured_bytes=captured_bytes,
        oracle_by_fixture=oracle_by_fixture or {},
        arms_by_fixture=arms_by_fixture or {},
        comparisons=comparisons or {},
        allowances_by_invariant=allowances,
    )
    return H3GateEvaluation(
        result=result,
        fixture_hashes=fixture_hashes,
        oracle_by_fixture=oracle_by_fixture or {},
        arms_by_fixture=arms_by_fixture or {},
        comparisons=comparisons or {},
        allowances_by_invariant=allowances,
        validation_payload=payload,
    )


def _validation_payload(
    config: ResolvedConfig,
    result: H3GateResult,
    fixture_hashes: H3FixtureHashes,
    *,
    captured_bytes: tuple[bytes, bytes],
    oracle_by_fixture: Mapping[str, H3PosteriorOracleEvaluation],
    arms_by_fixture: Mapping[str, Mapping[str, H3ArmResult]],
    comparisons: Mapping[str, object],
    allowances_by_invariant: Mapping[str, object],
) -> dict[str, object]:
    canonical = json.loads(config.canonical_json)
    payload: dict[str, object] = {
        "schema_version": 1,
        "gate": "H3",
        "status": result.status.value,
        "obligations": list(result.obligations),
        "fixtures": {
            "coupled": {
                "fixture_id": "h3-coupled-v1",
                "relative_path": H3_COUPLED_FIXTURE_PATH.relative_to(
                    REPO_ROOT
                ).as_posix(),
                "byte_count": len(captured_bytes[0]),
                "hash_domain": "raw_fixture_bytes",
                "expected_sha256": fixture_hashes.coupled_expected_sha256,
                "observed_sha256": fixture_hashes.coupled_observed_sha256,
            },
            "zero_control": {
                "fixture_id": "h3-zero-control-v1",
                "relative_path": H3_ZERO_CONTROL_FIXTURE_PATH.relative_to(
                    REPO_ROOT
                ).as_posix(),
                "byte_count": len(captured_bytes[1]),
                "hash_domain": "raw_fixture_bytes",
                "expected_sha256": fixture_hashes.zero_control_expected_sha256,
                "observed_sha256": fixture_hashes.zero_control_observed_sha256,
            },
        },
        "config_sha256": config.config_sha256,
        "h3_profile": canonical["h3"],
        "oracles": {
            key: _oracle_payload(value)
            for key, value in oracle_by_fixture.items()
        },
        "arms": {
            key: {
                family: _arm_payload(value)
                for family, value in by_family.items()
            }
            for key, by_family in arms_by_fixture.items()
        },
        "measurements": dict(result.measurements),
        "comparisons": _thaw_json_like(comparisons),
        "allowance_constants": {
            "eps": EPS,
            "C": C,
            "dimension": _DIMENSION,
            "solver_allowance_nats": SOLVER_ALLOWANCE_NATS,
            "maximum_allowance_fraction": 0.01,
        },
        "allowances_by_invariant": _thaw_json_like(
            allowances_by_invariant
        ),
        "invariants": [_invariant_payload(item) for item in result.invariants],
        "bounded_claim": (
            "H3 evaluates structured-versus-factorized Gaussian recognition "
            "adequacy on two frozen four-dimensional laws."
        ),
        "nonclaims": [
            "H3 does not establish H4 information-form cost.",
            "H3 does not establish H5 update coherence.",
            "H3 does not establish H6 prefix prediction.",
            "H3 does not establish H7 frame covariance.",
            "H3 does not establish H8 sparse scaling.",
        ],
    }
    _assert_finite_json(payload)
    return payload


def _oracle_payload(value: H3PosteriorOracleEvaluation) -> dict[str, object]:
    return {
        "fixture_id": value.fixture_id,
        "precision": value.precision.tolist(),
        "natural": value.natural.tolist(),
        "mean": value.mean.tolist(),
        "covariance": value.covariance.tolist(),
        "log_evidence": value.log_evidence,
        "analytic_factorized_precision": (
            value.analytic_factorized_precision.tolist()
        ),
        "analytic_factorized_mean": value.analytic_factorized_mean.tolist(),
        "analytic_factorized_reverse_kl": (
            value.analytic_factorized_reverse_kl
        ),
        "diagnostics": _thaw_json_like(value.diagnostics),
    }


def _arm_payload(value: H3ArmResult) -> dict[str, object]:
    return {
        "family": value.family,
        "initialization": {
            "mean": [0.0, 0.0, 0.0, 0.0],
            "precision": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        },
        "converged": value.converged,
        "failure_reason": value.failure_reason,
        "accepted_iterations": value.accepted_iterations,
        "closure_evaluations": value.closure_evaluations,
        "terminal_elbo": value.terminal_elbo,
        "terminal_gradient_infinity_norm": (
            value.terminal_gradient_infinity_norm
        ),
        "terminal_objective_change": value.terminal_objective_change,
        "terminal_mean": value.terminal_mean,
        "terminal_precision_cholesky": value.terminal_precision_cholesky,
        "terminal_precision": value.terminal_precision,
        "accepted_elbos": value.accepted_elbos,
        "trace_hash_domain": "canonical_accepted_trace_json",
        "canonical_trace_sha256": value.canonical_trace_sha256,
    }


def _invariant_payload(value: InvariantResult) -> dict[str, object]:
    return {
        "name": value.name,
        "passed": value.passed,
        "value": value.value,
        "limit": value.limit,
        "detail": value.detail,
    }


def _assert_finite_json(value: object, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError(f"{path} keys must be strings")
            _assert_finite_json(item, f"{path}.{key}")
        return
    if type(value) in (list, tuple):
        for index, item in enumerate(value):
            _assert_finite_json(item, f"{path}[{index}]")
        return
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    raise ValueError(f"{path} must contain finite JSON values")


def _scalar_record(
    name: str,
    value: float,
    *,
    absolute_sum: float,
    kappas: tuple[float, ...],
    optimized: bool,
) -> dict[str, object]:
    checked_value = _finite(value, f"{name}.value")
    checked_absolute = _finite(absolute_sum, f"{name}.absolute_sum")
    if checked_absolute < 0.0:
        raise ValueError("absolute summand accumulation must be nonnegative")
    if type(kappas) is not tuple or not kappas:
        raise ValueError("scalar operand kappas must be a nonempty tuple")
    checked_kappas = tuple(
        _finite(item, f"{name}.kappas[{index}]")
        for index, item in enumerate(kappas)
    )
    allowance = scalar_allowance(
        _DIMENSION,
        value=checked_value,
        absolute_sum=checked_absolute,
        kappas=checked_kappas,
        optimized=optimized,
    )
    return {
        "name": name,
        "value": checked_value,
        "absolute_sum": checked_absolute,
        "condition_numbers": checked_kappas,
        "optimized": optimized,
        "scalar_allowance": allowance,
    }


def _exact_scalar_record(name: str, value: float) -> dict[str, object]:
    return {
        "name": name,
        "value": _finite(value, f"{name}.value"),
        "absolute_sum": abs(float(value)),
        "condition_numbers": (1.0,),
        "optimized": False,
        "scalar_allowance": 0.0,
    }


def _pair_element(
    path: str,
    left: Mapping[str, object],
    right: Mapping[str, object],
    *,
    decisiveness_scale: float,
) -> dict[str, object]:
    left_value = _finite(left["value"], f"{path}.left")
    right_value = _finite(right["value"], f"{path}.right")
    left_allowance = _finite(
        left["scalar_allowance"], f"{path}.left_allowance"
    )
    right_allowance = _finite(
        right["scalar_allowance"], f"{path}.right_allowance"
    )
    allowance = pair_allowance(
        _DIMENSION,
        left=left_value,
        right=right_value,
        left_allowance=left_allowance,
        right_allowance=right_allowance,
    )
    residual = abs(left_value - right_value)
    scale = _finite(decisiveness_scale, f"{path}.decisiveness_scale")
    decisive = allowance_is_decisive(allowance, scale)
    return {
        "path": path,
        "kind": "pair",
        "operands": (dict(left), dict(right)),
        "operand_allowances": (left_allowance, right_allowance),
        "final_allowance": allowance,
        "residual": residual,
        "decisiveness_scale": scale,
        "decisiveness_ratio": allowance / scale,
        "decisive": decisive,
        "passed": residual <= allowance,
    }


def _element_group_record(elements: list[dict[str, object]]) -> dict[str, object]:
    if not elements:
        raise ValueError("element-local allowance group cannot be empty")
    final_allowance = max(float(item["final_allowance"]) for item in elements)
    residual = max(float(item["residual"]) for item in elements)
    maximum_normalized_residual = max(
        float(item["residual"]) / float(item["final_allowance"])
        for item in elements
    )
    ratio = max(float(item["decisiveness_ratio"]) for item in elements)
    return {
        "kind": "pair",
        "elements": tuple(elements),
        "aggregation": "maximum_element_normalized_residual",
        "maximum_normalized_residual": maximum_normalized_residual,
        "final_allowance": final_allowance,
        "residual": residual,
        "decisiveness_scale": min(
            float(item["decisiveness_scale"]) for item in elements
        ),
        "decisiveness_ratio": ratio,
        "decisive": all(bool(item["decisive"]) for item in elements),
        "passed": all(bool(item["passed"]) for item in elements),
    }


def _comparison_invariant(
    name: str,
    record: Mapping[str, object],
) -> InvariantResult:
    if record.get("aggregation") == "maximum_element_normalized_residual":
        value = _finite(
            record["maximum_normalized_residual"],
            f"{name}.maximum_normalized_residual",
        )
        limit = 1.0
        detail = "maximum element-local normalized residual is at most one"
    else:
        value = _finite(record["residual"], f"{name}.residual")
        limit = _finite(record["final_allowance"], f"{name}.allowance")
        detail = "operand-local absolute comparison under its recorded allowance"
    return InvariantResult(
        name=name,
        passed=bool(record["passed"]),
        value=value,
        limit=limit,
        detail=detail,
    )


def _reference_agreement_record(
    fixture: H3Fixture,
    oracle: H3PosteriorOracleEvaluation,
) -> dict[str, object]:
    diagnostics = oracle.diagnostics
    kappa = float(diagnostics["kappa_2"])
    precision_absolute = np.asarray(
        diagnostics["precision_absolute_summand_accumulation"],
        dtype=np.float64,
    )
    natural_absolute = np.asarray(
        diagnostics["natural_absolute_summand_accumulation"],
        dtype=np.float64,
    )
    elements: list[dict[str, object]] = []
    for row in range(_DIMENSION):
        for column in range(_DIMENSION):
            expected = float(fixture.reference_posterior_precision[row][column])
            actual = float(oracle.precision[row, column])
            elements.append(
                _pair_element(
                    f"posterior_precision[{row}][{column}]",
                    _scalar_record(
                        "frozen_reference",
                        expected,
                        absolute_sum=abs(expected),
                        kappas=(kappa,),
                        optimized=False,
                    ),
                    _scalar_record(
                        "numpy_oracle",
                        actual,
                        absolute_sum=float(precision_absolute[row, column]),
                        kappas=tuple(
                            float(item)
                            for item in diagnostics[
                                "canonical_precision_operand_kappas"
                            ][row]
                        ),
                        optimized=False,
                    ),
                    decisiveness_scale=max(1.0, abs(expected), abs(actual)),
                )
            )
    if fixture.reference_posterior_natural is not None:
        for index in range(_DIMENSION):
            expected = float(fixture.reference_posterior_natural[index])
            actual = float(oracle.natural[index])
            elements.append(
                _pair_element(
                    f"posterior_natural[{index}]",
                    _scalar_record(
                        "frozen_reference",
                        expected,
                        absolute_sum=abs(expected),
                        kappas=(1.0,),
                        optimized=False,
                    ),
                    _scalar_record(
                        "numpy_oracle",
                        actual,
                        absolute_sum=float(natural_absolute[index]),
                        kappas=tuple(
                            float(item)
                            for item in diagnostics[
                                "canonical_natural_operand_kappas"
                            ]
                        ),
                        optimized=False,
                    ),
                    decisiveness_scale=max(1.0, abs(expected), abs(actual)),
                )
            )
    if fixture.reference_log_evidence is not None:
        expected = float(fixture.reference_log_evidence)
        actual = float(oracle.log_evidence)
        elements.append(
            _pair_element(
                "log_evidence",
                _scalar_record(
                    "frozen_reference",
                    expected,
                    absolute_sum=abs(expected),
                    kappas=(kappa,),
                    optimized=False,
                ),
                _scalar_record(
                    "numpy_oracle",
                    actual,
                    absolute_sum=float(
                        diagnostics["log_evidence_absolute_summand_accumulation"]
                    ),
                    kappas=(kappa,),
                    optimized=False,
                ),
                decisiveness_scale=max(1.0, abs(expected), abs(actual)),
            )
        )
    expected_gap = (
        0.0
        if fixture.reference_analytic_factorized_reverse_kl is None
        else float(fixture.reference_analytic_factorized_reverse_kl)
    )
    actual_gap = float(oracle.analytic_factorized_reverse_kl)
    elements.append(
        _pair_element(
            "analytic_factorized_reverse_kl",
            _scalar_record(
                "frozen_reference",
                expected_gap,
                absolute_sum=abs(expected_gap),
                kappas=(kappa,),
                optimized=False,
            ),
            _scalar_record(
                "numpy_oracle",
                actual_gap,
                absolute_sum=float(
                    diagnostics[
                        "analytic_factorized_reverse_kl_absolute_summand_accumulation"
                    ]
                ),
                kappas=(kappa,),
                optimized=False,
            ),
            decisiveness_scale=max(1.0, abs(expected_gap), abs(actual_gap)),
        )
    )
    return _element_group_record(elements)


def _canonical_agreement_elements(
    fixture_key: str,
    model: H3GenerativeModel,
    oracle: H3PosteriorOracleEvaluation,
) -> list[dict[str, object]]:
    canonical = model.canonical_joint()
    precision = canonical.precision.detach().numpy()
    natural = canonical.natural.detach().numpy()
    precision_parts = np.stack(
        tuple(
            np.outer(factor.row.detach().numpy(), factor.row.detach().numpy())
            / float(factor.variance.item())
            for factor in model.factors
        )
    )
    natural_parts = np.stack(
        tuple(
            float(factor.target.item())
            * factor.row.detach().numpy()
            / float(factor.variance.item())
            for factor in model.factors
        )
    )
    precision_absolute = np.sum(np.abs(precision_parts), axis=0)
    natural_absolute = np.sum(np.abs(natural_parts), axis=0)
    kappa = _condition_number(oracle.precision)
    elements: list[dict[str, object]] = []
    for row in range(_DIMENSION):
        for column in range(_DIMENSION):
            left_value = float(precision[row, column])
            right_value = float(oracle.precision[row, column])
            elements.append(
                _pair_element(
                    f"{fixture_key}.precision[{row}][{column}]",
                    _scalar_record(
                        "pytorch_canonical",
                        left_value,
                        absolute_sum=float(precision_absolute[row, column]),
                        kappas=(kappa,),
                        optimized=False,
                    ),
                    _scalar_record(
                        "numpy_canonical",
                        right_value,
                        absolute_sum=float(
                            oracle.diagnostics[
                                "precision_absolute_summand_accumulation"
                            ][row][column]
                        ),
                        kappas=(kappa,),
                        optimized=False,
                    ),
                    decisiveness_scale=max(
                        1.0, abs(left_value), abs(right_value)
                    ),
                )
            )
    for index in range(_DIMENSION):
        left_value = float(natural[index])
        right_value = float(oracle.natural[index])
        elements.append(
            _pair_element(
                f"{fixture_key}.natural[{index}]",
                _scalar_record(
                    "pytorch_canonical",
                    left_value,
                    absolute_sum=float(natural_absolute[index]),
                    kappas=(1.0,),
                    optimized=False,
                ),
                _scalar_record(
                    "numpy_canonical",
                    right_value,
                    absolute_sum=float(
                        oracle.diagnostics[
                            "natural_absolute_summand_accumulation"
                        ][index]
                    ),
                    kappas=(1.0,),
                    optimized=False,
                ),
                decisiveness_scale=max(1.0, abs(left_value), abs(right_value)),
            )
        )
    return elements


def _condition_number(precision: np.ndarray) -> float:
    eigenvalues = np.linalg.eigvalsh(np.asarray(precision, dtype=np.float64))
    if not np.all(np.isfinite(eigenvalues)) or float(eigenvalues[0]) <= 0.0:
        return float("inf")
    return float(eigenvalues[-1] / eigenvalues[0])


def _terminal_records(
    models: Mapping[str, H3GenerativeModel],
    oracles: Mapping[str, H3PosteriorOracleEvaluation],
    arms: Mapping[str, Mapping[str, H3ArmResult]],
) -> dict[str, dict[str, dict[str, object]]]:
    records: dict[str, dict[str, dict[str, object]]] = {}
    for fixture_key in _FIXTURE_KEYS:
        records[fixture_key] = {}
        oracle = oracles[fixture_key]
        posterior_kappa = _condition_number(oracle.precision)
        for family in _FAMILIES:
            arm = arms[fixture_key][family]
            if (
                arm.terminal_mean is None
                or arm.terminal_precision is None
                or arm.terminal_precision_cholesky is None
                or arm.terminal_elbo is None
            ):
                raise ValueError("converged H3 arm lacks terminal evidence")
            mean = np.asarray(arm.terminal_mean, dtype=np.float64)
            precision = np.asarray(arm.terminal_precision, dtype=np.float64)
            cholesky = np.asarray(
                arm.terminal_precision_cholesky, dtype=np.float64
            )
            q = H3VariationalGaussian(
                family=family,
                mean=torch.tensor(mean, dtype=torch.float64, device="cpu"),
                precision_cholesky=torch.tensor(
                    cholesky, dtype=torch.float64, device="cpu"
                ),
            )
            direct = evaluate_h3_elbo(models[fixture_key], q)
            elbo = float(direct.elbo.detach().item())
            if abs(elbo - arm.terminal_elbo) > 64.0 * EPS * max(
                1.0, abs(elbo), abs(arm.terminal_elbo)
            ):
                raise ValueError("accepted and reconstructed terminal ELBO disagree")
            factor_terms = tuple(
                float(value.detach().item())
                for value in direct.expected_log_factors
            )
            entropy = float(direct.entropy.detach().item())
            elbo_absolute = math.fsum(
                (abs(entropy), *(abs(value) for value in factor_terms))
            )
            q_kappa = _condition_number(precision)
            elbo_scalar = _scalar_record(
                f"{fixture_key}.{family}.elbo",
                elbo,
                absolute_sum=elbo_absolute,
                kappas=(q_kappa, posterior_kappa),
                optimized=True,
            )
            kl, kl_parts, kl_absolute = _terminal_kl_parts(
                oracle,
                mean=mean,
                precision=precision,
            )
            public_kl = reverse_kl_to_oracle(
                oracle,
                mean=mean,
                precision=precision,
            )
            if abs(kl - public_kl) > 64.0 * EPS * max(
                1.0, abs(kl), abs(public_kl), kl_absolute
            ):
                raise ValueError("gate-local and oracle reverse KL disagree")
            kl_scalar = _scalar_record(
                f"{fixture_key}.{family}.reverse_kl",
                kl,
                absolute_sum=kl_absolute,
                kappas=(q_kappa, posterior_kappa),
                optimized=True,
            )
            records[fixture_key][family] = {
                "elbo": elbo,
                "elbo_terms": factor_terms,
                "entropy": entropy,
                "elbo_absolute_summand_accumulation": elbo_absolute,
                "elbo_scalar": elbo_scalar,
                "kl": kl,
                "kl_parts": kl_parts,
                "kl_absolute_summand_accumulation": kl_absolute,
                "kl_scalar": kl_scalar,
                "q_condition_number": q_kappa,
                "posterior_condition_number": posterior_kappa,
            }
    return records


def _terminal_kl_parts(
    oracle: H3PosteriorOracleEvaluation,
    *,
    mean: np.ndarray,
    precision: np.ndarray,
) -> tuple[float, tuple[dict[str, object], ...], float]:
    q_cholesky = np.linalg.cholesky(precision)
    identity = np.eye(_DIMENSION, dtype=np.float64)
    covariance = np.linalg.solve(
        q_cholesky.T,
        np.linalg.solve(q_cholesky, identity),
    )
    delta = oracle.mean - mean
    q_sign, q_logdet = np.linalg.slogdet(precision)
    p_sign, p_logdet = np.linalg.slogdet(oracle.precision)
    if q_sign != 1.0 or p_sign != 1.0:
        raise ValueError("terminal precision determinants must be positive")
    named_values = (
        ("trace_Jp_Sigmaq", float(np.trace(oracle.precision @ covariance))),
        ("quadratic_mean", float(delta @ oracle.precision @ delta)),
        ("minus_dimension", -float(_DIMENSION)),
        ("logdet_Jq", float(q_logdet)),
        ("minus_logdet_Jp", -float(p_logdet)),
    )
    values = tuple(value for _, value in named_values)
    kl = 0.5 * math.fsum(values)
    absolute = 0.5 * math.fsum(abs(value) for value in values)
    parts = tuple(
        {"name": name, "value": value} for name, value in named_values
    )
    return _finite(kl, "terminal reverse KL"), parts, absolute


def _oracle_gap_scalar(
    oracle: H3PosteriorOracleEvaluation,
) -> dict[str, object]:
    return _scalar_record(
        "coupled_analytic_factorized_reverse_kl",
        oracle.analytic_factorized_reverse_kl,
        absolute_sum=float(
            oracle.diagnostics[
                "analytic_factorized_reverse_kl_absolute_summand_accumulation"
            ]
        ),
        kappas=(float(oracle.diagnostics["kappa_2"]),),
        optimized=False,
    )


def _oracle_evidence_scalar(
    oracle: H3PosteriorOracleEvaluation,
) -> dict[str, object]:
    return _scalar_record(
        f"{oracle.fixture_id}.log_evidence",
        oracle.log_evidence,
        absolute_sum=float(
            oracle.diagnostics["log_evidence_absolute_summand_accumulation"]
        ),
        kappas=(float(oracle.diagnostics["kappa_2"]),),
        optimized=False,
    )


def _three_identity_record(
    name: str,
    evidence: Mapping[str, object],
    elbo: Mapping[str, object],
    kl: Mapping[str, object],
    *,
    decisiveness_scale: float,
) -> dict[str, object]:
    operands = (
        _finite(evidence["value"], f"{name}.evidence"),
        _finite(elbo["value"], f"{name}.elbo"),
        _finite(kl["value"], f"{name}.kl"),
    )
    operand_allowances = (
        _finite(evidence["scalar_allowance"], f"{name}.evidence_allowance"),
        _finite(elbo["scalar_allowance"], f"{name}.elbo_allowance"),
        _finite(kl["scalar_allowance"], f"{name}.kl_allowance"),
    )
    allowance = three_operand_identity_allowance(
        _DIMENSION,
        operands=operands,
        operand_allowances=operand_allowances,
    )
    residual = abs(operands[0] - operands[1] - operands[2])
    scale = _finite(decisiveness_scale, f"{name}.decisiveness_scale")
    return {
        "kind": "three_operand_identity",
        "operands": (dict(evidence), dict(elbo), dict(kl)),
        "operand_allowances": operand_allowances,
        "final_allowance": allowance,
        "residual": residual,
        "decisiveness_scale": scale,
        "decisiveness_ratio": allowance / scale,
        "decisive": allowance_is_decisive(allowance, scale),
        "passed": residual <= allowance,
    }


def _four_identity_record(
    name: str,
    elbo_structured: Mapping[str, object],
    elbo_factorized: Mapping[str, object],
    kl_factorized: Mapping[str, object],
    kl_structured: Mapping[str, object],
    *,
    decisiveness_scale: float,
) -> dict[str, object]:
    records = (
        elbo_structured,
        elbo_factorized,
        kl_factorized,
        kl_structured,
    )
    operands = tuple(
        _finite(record["value"], f"{name}.operands[{index}]")
        for index, record in enumerate(records)
    )
    operand_allowances = tuple(
        _finite(
            record["scalar_allowance"],
            f"{name}.operand_allowances[{index}]",
        )
        for index, record in enumerate(records)
    )
    allowance = four_operand_identity_allowance(
        _DIMENSION,
        operands=cast(tuple[float, float, float, float], operands),
        operand_allowances=cast(
            tuple[float, float, float, float], operand_allowances
        ),
    )
    residual = abs(operands[0] - operands[1] - operands[2] + operands[3])
    scale = _finite(decisiveness_scale, f"{name}.decisiveness_scale")
    return {
        "kind": "four_operand_identity",
        "operands": tuple(dict(record) for record in records),
        "operand_allowances": operand_allowances,
        "final_allowance": allowance,
        "residual": residual,
        "decisiveness_scale": scale,
        "decisiveness_ratio": allowance / scale,
        "decisive": allowance_is_decisive(allowance, scale),
        "passed": residual <= allowance,
    }


def _threshold_invariant(decision: H3ThresholdDecision) -> InvariantResult:
    return InvariantResult(
        name=decision.name,
        passed=decision.eligibility != "FAIL",
        value=decision.margin,
        limit=decision.allowance,
        detail=f"signed threshold eligibility={decision.eligibility}",
    )


def _threshold_payload(value: H3ThresholdDecision) -> dict[str, object]:
    return {
        "name": value.name,
        "operands": value.operands,
        "favorable_margin_formula": value.favorable_margin_formula,
        "favorable_direction": value.favorable_direction,
        "signed_margin": value.margin,
        "pair_allowance": value.allowance,
        "lower_boundary": value.lower_boundary,
        "upper_boundary": value.upper_boundary,
        "eligibility": value.eligibility,
        "obligation": value.obligation,
    }


def _finite(value: object, name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be finite numeric data")
    checked = float(value)
    if not math.isfinite(checked):
        raise ValueError(f"{name} must be finite numeric data")
    return checked


def _require_string_tuple(value: object, name: str) -> None:
    if type(value) is not tuple or not all(
        type(item) is str and item for item in value
    ):
        raise ValueError(f"{name} must contain nonempty strings")


def _deduplicate(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _freeze_json_like(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        copied: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str or not key:
                raise ValueError(f"{name} keys must be nonempty strings")
            copied[key] = _freeze_json_like(item, f"{name}.{key}")
        return MappingProxyType(copied)
    if type(value) in (list, tuple):
        return tuple(
            _freeze_json_like(item, f"{name}[{index}]")
            for index, item in enumerate(value)
        )
    if value is None or type(value) in (str, bool):
        return value
    if type(value) in (int, float):
        _finite(value, name)
        return value
    raise ValueError(f"{name} must contain finite JSON-compatible values")


def _thaw_json_like(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json_like(item) for key, item in value.items()}
    if type(value) is tuple:
        return [_thaw_json_like(item) for item in value]
    return value


def _freeze_json_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    frozen = _freeze_json_like(value, name)
    if not isinstance(frozen, Mapping):
        raise RuntimeError(f"{name} must freeze to a mapping")
    return frozen


def _freeze_object_mapping(
    value: object,
    expected_type: type[object],
    name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    copied: dict[str, object] = {}
    for key, item in value.items():
        if type(key) is not str or not key:
            raise ValueError(f"{name} keys must be nonempty strings")
        if not isinstance(item, expected_type):
            raise ValueError(f"{name}[{key!r}] has the wrong record type")
        copied[key] = item
    return MappingProxyType(copied)


def _freeze_arm_mapping(
    value: object,
) -> Mapping[str, Mapping[str, H3ArmResult]]:
    if not isinstance(value, Mapping):
        raise ValueError("arms_by_fixture must be a mapping")
    copied: dict[str, Mapping[str, H3ArmResult]] = {}
    for fixture_key, family_mapping in value.items():
        if type(fixture_key) is not str or not fixture_key:
            raise ValueError("arm fixture keys must be nonempty strings")
        if not isinstance(family_mapping, Mapping):
            raise ValueError("each arm fixture value must be a mapping")
        families: dict[str, H3ArmResult] = {}
        for family, result in family_mapping.items():
            if type(family) is not str or not isinstance(result, H3ArmResult):
                raise ValueError("arm mappings must contain H3ArmResult values")
            families[family] = result
        copied[fixture_key] = MappingProxyType(families)
    return MappingProxyType(copied)


__all__ = [
    "H3GateEvaluation",
    "H3ThresholdDecision",
    "H3_INVARIANT_NAMES",
    "evaluate_h3",
    "h3_validation_payload",
]
