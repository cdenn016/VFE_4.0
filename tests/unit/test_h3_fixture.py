from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import MappingProxyType

import pytest

import vfe4.types as public_types
import vfe4.validation as public_validation
from vfe4.types.h3 import (
    H3ArmResult,
    H3DecisionConfig,
    H3FixtureHashes,
    H3GateResult,
    H3InitializationConfig,
    H3OptimizationConfig,
)
from vfe4.types.results import GateStatus, InvariantResult
from vfe4.validation.h3_fixture import (
    H3_COUPLED_FIXTURE_PATH,
    H3_COUPLED_SHA256,
    H3_ZERO_CONTROL_FIXTURE_PATH,
    H3_ZERO_CONTROL_SHA256,
    parse_h3_fixture_bytes,
    validate_independent_control,
)


ROOT_FIELDS = {
    "fixture_schema_version",
    "fixture_id",
    "kind",
    "horizon",
    "dimensions",
    "continuous_order",
    "initial",
    "transitions",
    "observation",
    "reference",
}
FACTOR_FIELDS = {"factor_id", "row", "target", "variance"}


def _bytes(path: Path) -> bytes:
    return path.read_bytes()


def _raw(path: Path) -> dict[str, object]:
    result = json.loads(_bytes(path))
    assert type(result) is dict
    return result


def _encoded(raw: dict[str, object]) -> bytes:
    return json.dumps(raw, allow_nan=True).encode("utf-8")


def _parse_pair():
    coupled = parse_h3_fixture_bytes(
        _bytes(H3_COUPLED_FIXTURE_PATH), expected_fixture_id="h3-coupled-v1"
    )
    zero = parse_h3_fixture_bytes(
        _bytes(H3_ZERO_CONTROL_FIXTURE_PATH),
        expected_fixture_id="h3-zero-control-v1",
    )
    return coupled, zero


def test_frozen_raw_fixture_schemas_hashes_and_factor_order() -> None:
    coupled_raw = _raw(H3_COUPLED_FIXTURE_PATH)
    zero_raw = _raw(H3_ZERO_CONTROL_FIXTURE_PATH)
    assert set(coupled_raw) == ROOT_FIELDS
    assert set(zero_raw) == ROOT_FIELDS
    for raw in (coupled_raw, zero_raw):
        assert raw["dimensions"] == {"d_z": 1, "d_m": 1, "joint_dimension": 4}
        assert raw["continuous_order"] == ["z0", "m0", "z1", "m1"]
        factors = [
            *raw["initial"]["factors"],  # type: ignore[index]
            *raw["transitions"]["factors"],  # type: ignore[index]
            *raw["observation"]["factors"],  # type: ignore[index]
        ]
        assert len(factors) == 6
        assert all(set(factor) == FACTOR_FIELDS for factor in factors)
        assert [factor["factor_id"] for factor in factors] == [
            "z0_prior",
            "m0_prior",
            "m1_transition",
            "z1_transition",
            "z1_observation",
            "m1_observation",
        ]
        assert all(
            type(factor["variance"]) in (int, float)
            and factor["variance"] > 0.0
            for factor in factors
        )
    coupled_bytes = _bytes(H3_COUPLED_FIXTURE_PATH)
    zero_bytes = _bytes(H3_ZERO_CONTROL_FIXTURE_PATH)
    assert H3_COUPLED_FIXTURE_PATH != H3_ZERO_CONTROL_FIXTURE_PATH
    assert coupled_bytes != zero_bytes
    assert H3_COUPLED_SHA256 != H3_ZERO_CONTROL_SHA256
    assert hashlib.sha256(coupled_bytes).hexdigest() == H3_COUPLED_SHA256
    assert hashlib.sha256(zero_bytes).hexdigest() == H3_ZERO_CONTROL_SHA256


def test_parser_produces_immutable_h3_only_records_and_frozen_references() -> None:
    coupled, zero = _parse_pair()
    assert coupled.fixture_id == "h3-coupled-v1"
    assert coupled.kind == "coupled"
    assert zero.fixture_id == "h3-zero-control-v1"
    assert zero.kind == "zero_control"
    assert coupled.horizon == zero.horizon == 1
    assert (coupled.d_z, coupled.d_m, coupled.dimension) == (1, 1, 4)
    assert coupled.continuous_order == zero.continuous_order == (
        "z0",
        "m0",
        "z1",
        "m1",
    )
    assert len(coupled.factors) == len(zero.factors) == 6
    assert coupled.reference_posterior_precision == (
        (2.96, 0.0, -2.8, 1.68),
        (0.0, 2.77777777777778, 0.0, -2.22222222222222),
        (-2.8, 0.0, 5.5625, -2.4),
        (1.68, -2.22222222222222, -2.4, 5.78027777777778),
    )
    assert coupled.reference_posterior_natural == (0.0, 0.0, 1.71875, 0.3125)
    assert coupled.reference_log_evidence == -2.6536596233553
    assert coupled.reference_analytic_factorized_reverse_kl == 0.6815463199745935
    assert zero.reference_posterior_precision == (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 5.5625, 0.0),
        (0.0, 0.0, 0.0, 4.34027777777778),
    )
    assert zero.reference_posterior_natural is None
    assert zero.reference_log_evidence is None
    assert zero.reference_analytic_factorized_reverse_kl is None
    with pytest.raises(FrozenInstanceError):
        coupled.horizon = 2  # type: ignore[misc]
    assert type(coupled.factors) is tuple
    assert type(coupled.factors[0].row) is tuple


def test_separately_parsed_zero_control_satisfies_independence_contract() -> None:
    coupled, zero = _parse_pair()
    validate_independent_control(coupled, zero)
    assert coupled.initial_factors == zero.initial_factors
    assert coupled.observation_map == zero.observation_map
    assert coupled.observation_covariance == zero.observation_covariance
    assert coupled.transition_factors != zero.transition_factors
    assert coupled.observation_values != zero.observation_values
    assert zero.transition_factors[0].row == (0.0, 0.0, 0.0, 1.0)
    assert zero.transition_factors[1].row == (0.0, 0.0, 1.0, 0.0)


@pytest.mark.parametrize("field", sorted(ROOT_FIELDS))
def test_parser_rejects_missing_root_fields(field: str) -> None:
    raw = _raw(H3_COUPLED_FIXTURE_PATH)
    del raw[field]
    with pytest.raises(ValueError):
        parse_h3_fixture_bytes(_encoded(raw), expected_fixture_id="h3-coupled-v1")


def test_parser_rejects_unknown_fields_and_wrong_identity() -> None:
    raw = _raw(H3_COUPLED_FIXTURE_PATH)
    raw["unknown"] = 1
    with pytest.raises(ValueError):
        parse_h3_fixture_bytes(_encoded(raw), expected_fixture_id="h3-coupled-v1")
    with pytest.raises(ValueError):
        parse_h3_fixture_bytes(
            _bytes(H3_COUPLED_FIXTURE_PATH),
            expected_fixture_id="h3-zero-control-v1",
        )


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), float("-inf")])
def test_parser_rejects_boolean_or_nonfinite_numeric_data(value: object) -> None:
    raw = _raw(H3_COUPLED_FIXTURE_PATH)
    raw["initial"]["factors"][0]["variance"] = value  # type: ignore[index]
    with pytest.raises(ValueError):
        parse_h3_fixture_bytes(_encoded(raw), expected_fixture_id="h3-coupled-v1")


def test_parser_rejects_bad_rows_variances_duplicates_and_order() -> None:
    cases: list[dict[str, object]] = []
    wrong_row = _raw(H3_COUPLED_FIXTURE_PATH)
    wrong_row["transitions"]["factors"][0]["row"] = [0.0, 1.0]  # type: ignore[index]
    cases.append(wrong_row)
    nonpositive = _raw(H3_COUPLED_FIXTURE_PATH)
    nonpositive["observation"]["factors"][0]["variance"] = 0.0  # type: ignore[index]
    cases.append(nonpositive)
    duplicate = _raw(H3_COUPLED_FIXTURE_PATH)
    duplicate["transitions"]["factors"][0]["factor_id"] = "z0_prior"  # type: ignore[index]
    cases.append(duplicate)
    wrong_order = _raw(H3_COUPLED_FIXTURE_PATH)
    wrong_order["continuous_order"] = ["m0", "z0", "z1", "m1"]
    cases.append(wrong_order)
    for raw in cases:
        with pytest.raises(ValueError):
            parse_h3_fixture_bytes(_encoded(raw), expected_fixture_id="h3-coupled-v1")


def test_parser_rejects_observation_map_covariance_and_reference_shape_changes() -> None:
    wrong_map = _raw(H3_COUPLED_FIXTURE_PATH)
    wrong_map["observation"]["map"][0] = [0.0, 0.0, 0.0, 1.0]  # type: ignore[index]
    wrong_covariance = _raw(H3_COUPLED_FIXTURE_PATH)
    wrong_covariance["observation"]["covariance"][0][1] = 0.1  # type: ignore[index]
    wrong_reference = _raw(H3_COUPLED_FIXTURE_PATH)
    wrong_reference["reference"]["posterior_precision"] = [[1.0]]  # type: ignore[index]
    for raw in (wrong_map, wrong_covariance, wrong_reference):
        with pytest.raises(ValueError):
            parse_h3_fixture_bytes(_encoded(raw), expected_fixture_id="h3-coupled-v1")


def test_control_validator_rejects_couplings_initial_drift_and_offdiagonal_reference() -> None:
    coupled, zero = _parse_pair()
    changed_parent = replace(
        zero,
        transition_factors=(
            replace(zero.transition_factors[0], row=(0.0, -0.1, 0.0, 1.0)),
            zero.transition_factors[1],
        ),
    )
    changed_initial_law = replace(
        zero,
        initial_factors=(
            replace(zero.initial_factors[0], variance=1.1),
            zero.initial_factors[1],
        ),
    )
    changed_observation = replace(zero, observation_values=(0.5, -0.7))
    offdiagonal_precision = replace(
        zero,
        reference_posterior_precision=(
            (1.0, 0.01, 0.0, 0.0),
            (0.01, 1.0, 0.0, 0.0),
            (0.0, 0.0, 5.5625, 0.0),
            (0.0, 0.0, 0.0, 4.34027777777778),
        ),
    )
    for invalid_control in (
        changed_parent,
        changed_initial_law,
        changed_observation,
        offdiagonal_precision,
    ):
        with pytest.raises(ValueError):
            validate_independent_control(coupled, invalid_control)


def test_h3_public_package_surfaces_export_declared_types_and_hash_mapping() -> None:
    for name in (
        "H3FixtureId",
        "H3FixtureKind",
        "H3Matrix4",
        "H3RecognitionFamily",
        "H3Vector4",
    ):
        assert name in public_types.__all__
        assert hasattr(public_types, name)
    assert "H3_EXPECTED_SHA256_BY_FIXTURE_ID" in public_validation.__all__
    assert public_validation.H3_EXPECTED_SHA256_BY_FIXTURE_ID == {
        "h3-coupled-v1": H3_COUPLED_SHA256,
        "h3-zero-control-v1": H3_ZERO_CONTROL_SHA256,
    }


def test_h3_configuration_and_arm_records_are_immutable_and_frozen() -> None:
    initialization = H3InitializationConfig()
    optimizer = H3OptimizationConfig()
    decisions = H3DecisionConfig()
    assert initialization.mean == (0.0, 0.0, 0.0, 0.0)
    assert initialization.precision[0] == (1.0, 0.0, 0.0, 0.0)
    assert optimizer.maximum_closure_evaluations == 5_000
    assert decisions.minimum_coupled_gap_nats == 0.50
    with pytest.raises(ValueError):
        H3InitializationConfig(mean=(1.0, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        H3OptimizationConfig(maximum_closure_evaluations=4_999)
    with pytest.raises(ValueError):
        H3DecisionConfig(maximum_allowance_fraction=0.02)
    arm = H3ArmResult(
        family="fine_factorized_diagonal",
        converged=True,
        failure_reason=None,
        accepted_iterations=4,
        closure_evaluations=8,
        terminal_elbo=-2.0,
        terminal_gradient_infinity_norm=1.0e-10,
        terminal_objective_change=1.0e-13,
        terminal_mean=(0.0, 0.0, 0.0, 0.0),
        terminal_precision_cholesky=initialization.precision,
        terminal_precision=initialization.precision,
        accepted_elbos=(-3.0, -2.5, -2.25, -2.0),
        canonical_trace_sha256="a" * 64,
    )
    assert arm.converged
    with pytest.raises(FrozenInstanceError):
        arm.accepted_iterations = 5  # type: ignore[misc]


def _gate_result(
    *,
    status: GateStatus,
    invariants: tuple[InvariantResult, ...],
    allowances: dict[str, object],
    obligations: tuple[str, ...] = (),
) -> H3GateResult:
    return H3GateResult(
        gate="H3",
        coupled_fixture_id="h3-coupled-v1",
        zero_control_fixture_id="h3-zero-control-v1",
        status=status,
        measurements={"gap": 0.6815463199745935},
        invariants=invariants,
        allowances_by_invariant=allowances,
        obligations=obligations,
    )


def test_h3_gate_result_freezes_exact_allowance_mapping_without_singular_allowance() -> None:
    invariants = (
        InvariantResult("fixture_hashes_match", True, None, None, "raw bytes match"),
        InvariantResult("coupled_factorized_analytic_gap", True, 0.0, 1.0e-7, "closed"),
    )
    result = _gate_result(
        status=GateStatus.PASS,
        invariants=invariants,
        allowances={
            "coupled_factorized_analytic_gap": {
                "kind": "pair",
                "final_allowance": 1.0e-7,
            }
        },
    )
    assert isinstance(result.allowances_by_invariant, MappingProxyType)
    assert isinstance(
        result.allowances_by_invariant["coupled_factorized_analytic_gap"],
        MappingProxyType,
    )
    assert not hasattr(result, "calibrated_allowance")
    assert not hasattr(result, "residual")
    with pytest.raises(TypeError):
        result.allowances_by_invariant["extra"] = {}  # type: ignore[index]


def test_h3_gate_result_rejects_missing_extra_or_invented_eligibility_allowances() -> None:
    comparison = InvariantResult(
        "coupled_factorized_analytic_gap", True, 0.0, 1.0e-7, "closed"
    )
    hashes = InvariantResult("fixture_hashes_match", True, None, None, "matched")
    with pytest.raises(ValueError):
        _gate_result(
            status=GateStatus.PASS,
            invariants=(hashes, comparison),
            allowances={},
        )
    with pytest.raises(ValueError):
        _gate_result(
            status=GateStatus.PASS,
            invariants=(hashes, comparison),
            allowances={
                "coupled_factorized_analytic_gap": {"kind": "pair"},
                "extra": {"kind": "pair"},
            },
        )
    with pytest.raises(ValueError):
        _gate_result(
            status=GateStatus.PASS,
            invariants=(hashes, comparison),
            allowances={
                "fixture_hashes_match": {"kind": "pair"},
                "coupled_factorized_analytic_gap": {"kind": "pair"},
            },
        )


def test_h3_gate_result_fail_closed_status_consistency() -> None:
    passing = InvariantResult(
        "coupled_factorized_analytic_gap", True, 0.0, 1.0e-7, "closed"
    )
    failing = InvariantResult(
        "coupled_factorized_analytic_gap", False, 0.1, 1.0e-7, "outside"
    )
    allowance = {
        "coupled_factorized_analytic_gap": {
            "kind": "pair",
            "final_allowance": 1.0e-7,
        }
    }
    with pytest.raises(ValueError):
        _gate_result(status=GateStatus.PASS, invariants=(failing,), allowances=allowance)
    with pytest.raises(ValueError):
        _gate_result(status=GateStatus.FAIL, invariants=(passing,), allowances=allowance)
    with pytest.raises(ValueError):
        _gate_result(
            status=GateStatus.INCONCLUSIVE,
            invariants=(passing,),
            allowances=allowance,
        )
    inconclusive = _gate_result(
        status=GateStatus.INCONCLUSIVE,
        invariants=(passing,),
        allowances=allowance,
        obligations=("obtain missing evidence",),
    )
    assert inconclusive.status is GateStatus.INCONCLUSIVE
    assert _gate_result(
        status=GateStatus.FAIL, invariants=(failing,), allowances=allowance
    ).status is GateStatus.FAIL


def test_h3_gate_result_rejects_conclusive_missing_decision_evidence() -> None:
    missing_value = InvariantResult(
        "coupled_factorized_analytic_gap", True, None, 1.0e-7, "missing value"
    )
    missing_limit = InvariantResult(
        "coupled_factorized_analytic_gap", False, 0.1, None, "missing limit"
    )
    allowance = {
        "coupled_factorized_analytic_gap": {
            "kind": "pair",
            "final_allowance": 1.0e-7,
        }
    }
    with pytest.raises(ValueError):
        _gate_result(
            status=GateStatus.PASS,
            invariants=(missing_value,),
            allowances=allowance,
        )
    with pytest.raises(ValueError):
        _gate_result(
            status=GateStatus.FAIL,
            invariants=(missing_limit,),
            allowances=allowance,
        )


def test_h3_gate_result_rejects_failed_eligibility_as_fail_decision() -> None:
    failed_eligibility = InvariantResult(
        "fixture_hashes_match", False, 0.0, 0.0, "fixture bytes differ"
    )
    passing_decision = InvariantResult(
        "coupled_factorized_analytic_gap", True, 0.0, 1.0e-7, "closed"
    )
    allowance = {
        "coupled_factorized_analytic_gap": {
            "kind": "pair",
            "final_allowance": 1.0e-7,
        }
    }
    with pytest.raises(ValueError):
        _gate_result(
            status=GateStatus.FAIL,
            invariants=(failed_eligibility, passing_decision),
            allowances=allowance,
        )


def test_fixture_hash_record_requires_distinct_expected_domains() -> None:
    hashes = H3FixtureHashes(
        coupled_expected_sha256="a" * 64,
        coupled_observed_sha256="a" * 64,
        zero_control_expected_sha256="b" * 64,
        zero_control_observed_sha256="b" * 64,
    )
    assert hashes.coupled_matches and hashes.zero_control_matches

    corrupted = H3FixtureHashes(
        coupled_expected_sha256="a" * 64,
        coupled_observed_sha256="c" * 64,
        zero_control_expected_sha256="b" * 64,
        zero_control_observed_sha256="c" * 64,
    )
    assert not corrupted.coupled_matches
    assert not corrupted.zero_control_matches

    with pytest.raises(ValueError):
        H3FixtureHashes(
            coupled_expected_sha256="a" * 64,
            coupled_observed_sha256="a" * 64,
            zero_control_expected_sha256="a" * 64,
            zero_control_observed_sha256="b" * 64,
        )
