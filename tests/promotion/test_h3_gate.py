from __future__ import annotations

import dataclasses
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

import verification.h3_gate as gate_module
from vfe4.config import ResolvedConfig, resolve_config
from vfe4.types.results import GateStatus
from vfe4.validation.h3_fixture import (
    H3_COUPLED_FIXTURE_PATH,
    H3_ZERO_CONTROL_FIXTURE_PATH,
)
from verification.h3_gate import (
    H3_INVARIANT_NAMES,
    H3GateEvaluation,
    _status_and_obligations,
    _threshold_decision,
    evaluate_h3,
    h3_validation_payload,
)


EXPECTED_H3_INVARIANT_NAMES = (
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


def _raw_config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "objective_schema_version": "vfe4-state-elbo-v1",
        "run": {
            "mode": "verify",
            "seed": 20260721,
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
        "inference": {
            "operation": "evaluate_only",
            "estimator": "deterministic_quadrature",
        },
        "optimization": {
            "e_like_update": "none",
            "m_like_update": "none",
            "expected_autograd_scope": "none",
        },
        "validation": {
            "gates": ["H1", "H2", "H3"],
            "fixture_id": "h1-v1",
            "quadrature_order": 21,
            "convergence_check_order": 17,
            "maximum_convergence_estimate": 1e-9,
        },
        "artifacts": {"run_root": "runs"},
        "h3": {
            "coupled_fixture_id": "h3-coupled-v1",
            "coupled_expected_sha256": (
                "6779f5b0a2e27aa5e203764bcc4d84c1b1daedb9423fcefdf28dce3cf7e40e03"
            ),
            "zero_control_fixture_id": "h3-zero-control-v1",
            "zero_control_expected_sha256": (
                "ba600e09e0ae7e2b7576fbf4446a8e5b38a605c7621eb0cd5586689dccb89acf"
            ),
            "recognition_families": [
                "structured_full_spd",
                "fine_factorized_diagonal",
            ],
            "common_initialization": {
                "mean": [0.0, 0.0, 0.0, 0.0],
                "precision": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            },
            "optimization_operation": "maximize_direct_h3_elbo_lbfgs",
            "expected_autograd_scope": "h3_recognition_only",
            "optimizer": {
                "learning_rate": 1.0,
                "maximum_iterations_per_step": 1,
                "maximum_evaluations_per_step": 25,
                "tolerance_gradient": 1.0e-12,
                "tolerance_change": 1.0e-18,
                "history_size": 20,
                "line_search": "strong_wolfe",
                "maximum_accepted_iterations": 200,
                "maximum_closure_evaluations": 5_000,
                "terminal_gradient_infinity_norm": 1.0e-8,
                "terminal_objective_change": 1.0e-12,
                "required_consecutive_accepted_iterations": 3,
            },
            "decision": {
                "dimension": 4,
                "minimum_precision_eigenvalue": 1.0e-4,
                "maximum_precision_eigenvalue": 1.0e4,
                "maximum_precision_condition_number": 1.0e6,
                "maximum_mean_infinity_norm": 4.0,
                "minimum_coupled_gap_nats": 0.50,
                "maximum_structured_gap_fraction": 0.01,
                "maximum_allowance_fraction": 0.01,
            },
            "solver_allowance_nats": 1.0e-7,
            "threshold_decision_rule": "signed_margin_three_way",
            "minimum_resolved_fraction": 0.99,
            "coupled_gap_inconclusive_obligation": (
                "resolve coupled gap threshold outside allowance"
            ),
            "structured_closure_inconclusive_obligation": (
                "resolve structured closure threshold outside allowance"
            ),
        },
    }


@pytest.fixture(scope="module")
def h3_config(tmp_path_factory: pytest.TempPathFactory) -> ResolvedConfig:
    return resolve_config(
        _raw_config(),
        repo_root=tmp_path_factory.mktemp("h3-config"),
    )


@pytest.fixture(scope="module")
def h3_evaluation(h3_config: ResolvedConfig) -> H3GateEvaluation:
    return evaluate_h3(h3_config)


def test_h3_gate_exposes_the_task_five_api() -> None:
    assert H3GateEvaluation.__name__ == "H3GateEvaluation"
    assert callable(evaluate_h3)
    assert callable(h3_validation_payload)
    assert callable(_threshold_decision)
    assert callable(_status_and_obligations)


def test_h3_invariant_inventory_is_literal_complete_and_ordered() -> None:
    assert len(EXPECTED_H3_INVARIANT_NAMES) == 19
    assert len(set(EXPECTED_H3_INVARIANT_NAMES)) == 19
    assert H3_INVARIANT_NAMES == EXPECTED_H3_INVARIANT_NAMES


def test_frozen_h3_pair_passes_with_exact_inventory_and_allowance_kinds(
    h3_evaluation: H3GateEvaluation,
) -> None:
    result = h3_evaluation.result

    assert result.status is GateStatus.PASS
    assert result.obligations == ()
    assert tuple(item.name for item in result.invariants) == (
        EXPECTED_H3_INVARIANT_NAMES
    )
    assert all(item.passed for item in result.invariants)
    assert len(h3_evaluation.allowances_by_invariant) == 14
    kinds = Counter(
        record["kind"]
        for record in h3_evaluation.allowances_by_invariant.values()
    )
    assert kinds == {
        "pair": 9,
        "three_operand_identity": 4,
        "four_operand_identity": 1,
    }
    assert tuple(h3_evaluation.arms_by_fixture) == ("coupled", "zero_control")
    assert all(
        tuple(by_family) == (
            "structured_full_spd",
            "fine_factorized_diagonal",
        )
        for by_family in h3_evaluation.arms_by_fixture.values()
    )


@pytest.mark.parametrize(
    ("margin", "expected"),
    (
        (math.nextafter(1.0, math.inf), "PASS_ELIGIBLE"),
        (1.0, "INCONCLUSIVE"),
        (0.0, "INCONCLUSIVE"),
        (-1.0, "INCONCLUSIVE"),
        (math.nextafter(-1.0, -math.inf), "FAIL"),
    ),
)
def test_signed_threshold_boundaries_are_exact(
    margin: float, expected: str
) -> None:
    decision = _threshold_decision(
        margin,
        1.0,
        "resolve threshold",
    )

    assert decision.eligibility == expected
    assert decision.lower_boundary == -1.0
    assert decision.upper_boundary == 1.0
    assert (decision.obligation is not None) == (expected == "INCONCLUSIVE")


@pytest.mark.parametrize(
    ("name", "formula", "obligation"),
    (
        (
            "coupled_oracle_gap_minimum",
            "G-0.50",
            "resolve coupled gap threshold outside allowance",
        ),
        (
            "coupled_structured_fraction_resolved",
            "0.01*G-KL_cs",
            "resolve structured closure threshold outside allowance",
        ),
    ),
)
@pytest.mark.parametrize(
    ("margin", "expected"),
    (
        (math.nextafter(1.0, math.inf), "PASS_ELIGIBLE"),
        (1.0, "INCONCLUSIVE"),
        (0.0, "INCONCLUSIVE"),
        (-1.0, "INCONCLUSIVE"),
        (math.nextafter(-1.0, -math.inf), "FAIL"),
    ),
)
def test_each_named_threshold_has_exact_boundary_table_and_obligation(
    name: str,
    formula: str,
    obligation: str,
    margin: float,
    expected: str,
) -> None:
    decision = _threshold_decision(
        margin,
        1.0,
        obligation,
        name=name,
        formula=formula,
    )

    assert decision.eligibility == expected
    assert decision.name == name
    assert decision.favorable_margin_formula == formula
    assert decision.obligation == (
        obligation if expected == "INCONCLUSIVE" else None
    )


def test_status_precedence_separates_integrity_finite_failure_and_ambiguity() -> None:
    passing = _threshold_decision(2.0, 1.0, "unused")
    ambiguous = _threshold_decision(0.0, 1.0, "resolve threshold")
    failing = _threshold_decision(-2.0, 1.0, "unused")

    assert _status_and_obligations(
        threshold_decisions=(passing, passing)
    ) == (GateStatus.PASS, ())
    assert _status_and_obligations(
        threshold_decisions=(ambiguous, passing)
    ) == (GateStatus.INCONCLUSIVE, ("resolve threshold",))
    assert _status_and_obligations(
        threshold_decisions=(ambiguous, failing)
    ) == (GateStatus.FAIL, ())
    assert _status_and_obligations(
        threshold_decisions=(ambiguous, passing),
        equality_failures=("finite equality miss",),
    ) == (GateStatus.FAIL, ())
    assert _status_and_obligations(
        upstream_obligations=("restore evidence",),
        threshold_decisions=(failing, passing),
        equality_failures=("finite equality miss",),
    ) == (GateStatus.INCONCLUSIVE, ("restore evidence",))


def test_hash_mismatch_stops_before_all_downstream_consumers(
    h3_config: ResolvedConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"parse": 0, "oracle": 0, "optimize": 0}

    def forbidden(name: str) -> Any:
        def inner(*_args: object, **_kwargs: object) -> object:
            calls[name] += 1
            raise AssertionError(f"{name} must not run after a hash mismatch")

        return inner

    monkeypatch.setattr(gate_module, "parse_h3_fixture_bytes", forbidden("parse"))
    monkeypatch.setattr(
        gate_module, "evaluate_h3_posterior_oracle", forbidden("oracle")
    )
    monkeypatch.setattr(gate_module, "optimize_h3_arm", forbidden("optimize"))
    coupled = H3_COUPLED_FIXTURE_PATH.read_bytes() + b"\n"
    evaluation = evaluate_h3(
        h3_config,
        coupled_fixture_bytes=coupled,
        zero_control_fixture_bytes=H3_ZERO_CONTROL_FIXTURE_PATH.read_bytes(),
    )

    assert evaluation.result.status is GateStatus.INCONCLUSIVE
    assert tuple(item.name for item in evaluation.result.invariants) == (
        "fixture_hashes_match",
    )
    assert calls == {"parse": 0, "oracle": 0, "optimize": 0}


def test_identical_observed_fixture_bytes_are_inconclusive_before_consumers(
    h3_config: ResolvedConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"parse": 0, "oracle": 0, "optimize": 0}

    def forbidden(name: str) -> Any:
        def inner(*_args: object, **_kwargs: object) -> object:
            calls[name] += 1
            raise AssertionError(f"{name} must not run after equal fixture hashes")

        return inner

    monkeypatch.setattr(gate_module, "parse_h3_fixture_bytes", forbidden("parse"))
    monkeypatch.setattr(
        gate_module, "evaluate_h3_posterior_oracle", forbidden("oracle")
    )
    monkeypatch.setattr(gate_module, "optimize_h3_arm", forbidden("optimize"))
    coupled = H3_COUPLED_FIXTURE_PATH.read_bytes()

    evaluation = evaluate_h3(
        h3_config,
        coupled_fixture_bytes=coupled,
        zero_control_fixture_bytes=coupled,
    )

    assert evaluation.result.status is GateStatus.INCONCLUSIVE
    assert tuple(item.name for item in evaluation.result.invariants) == (
        "fixture_hashes_match",
    )
    assert evaluation.fixture_hashes.coupled_observed_sha256 == (
        evaluation.fixture_hashes.zero_control_observed_sha256
    )
    assert calls == {"parse": 0, "oracle": 0, "optimize": 0}


def test_control_contract_failure_stops_before_oracle_and_optimization(
    h3_config: ResolvedConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"oracle": 0, "optimize": 0}

    monkeypatch.setattr(
        gate_module,
        "validate_independent_control",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("forced control off-diagonal")
        ),
    )

    def forbidden(name: str) -> Any:
        def inner(*_args: object, **_kwargs: object) -> object:
            calls[name] += 1
            raise AssertionError(f"{name} must not run after control failure")

        return inner

    monkeypatch.setattr(
        gate_module, "evaluate_h3_posterior_oracle", forbidden("oracle")
    )
    monkeypatch.setattr(gate_module, "optimize_h3_arm", forbidden("optimize"))

    evaluation = evaluate_h3(h3_config)

    assert evaluation.result.status is GateStatus.INCONCLUSIVE
    assert tuple(item.name for item in evaluation.result.invariants) == (
        "fixture_hashes_match",
        "independent_control_contract",
    )
    assert calls == {"oracle": 0, "optimize": 0}


def _install_baseline_arms(
    monkeypatch: pytest.MonkeyPatch,
    h3_evaluation: H3GateEvaluation,
    *,
    first_replacement: object | None = None,
) -> None:
    ordered = [
        h3_evaluation.arms_by_fixture[fixture][family]
        for fixture in ("coupled", "zero_control")
        for family in ("structured_full_spd", "fine_factorized_diagonal")
    ]
    if first_replacement is not None:
        ordered[0] = first_replacement  # type: ignore[assignment]
    iterator = iter(ordered)
    monkeypatch.setattr(
        gate_module,
        "optimize_h3_arm",
        lambda *_args, **_kwargs: next(iterator),
    )


def test_nonconvergence_is_inconclusive_not_finite_inadequacy(
    h3_config: ResolvedConfig,
    h3_evaluation: H3GateEvaluation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = h3_evaluation.arms_by_fixture["coupled"]["structured_full_spd"]
    replacement = dataclasses.replace(
        original,
        converged=False,
        failure_reason="maximum_accepted_iterations_reached",
    )
    _install_baseline_arms(
        monkeypatch,
        h3_evaluation,
        first_replacement=replacement,
    )

    evaluation = evaluate_h3(h3_config)

    assert evaluation.result.status is GateStatus.INCONCLUSIVE
    assert tuple(item.name for item in evaluation.result.invariants)[-1] == (
        "all_arms_converged"
    )
    assert any("converged" in item for item in evaluation.result.obligations)


def test_reference_disagreement_and_envelope_failure_are_inconclusive(
    h3_config: ResolvedConfig,
    h3_evaluation: H3GateEvaluation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gate_module,
        "evaluate_h3_posterior_oracle",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("forced reference disagreement")
        ),
    )
    reference = evaluate_h3(h3_config)
    assert reference.result.status is GateStatus.INCONCLUSIVE
    assert reference.result.obligations == (
        "resolve coupled frozen-reference agreement",
    )

    monkeypatch.undo()
    _install_baseline_arms(monkeypatch, h3_evaluation)
    monkeypatch.setattr(
        gate_module,
        "_posterior_envelope",
        lambda *_args, **_kwargs: (False, "forced outside envelope"),
    )
    envelope = evaluate_h3(h3_config)
    assert envelope.result.status is GateStatus.INCONCLUSIVE
    assert tuple(item.name for item in envelope.result.invariants)[-1] == (
        "posterior_condition_envelope"
    )


def test_nonfinite_eligibility_failure_is_inconclusive(
    h3_config: ResolvedConfig,
    h3_evaluation: H3GateEvaluation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_baseline_arms(monkeypatch, h3_evaluation)
    monkeypatch.setattr(
        gate_module,
        "_posterior_envelope",
        lambda *_args, **_kwargs: (False, "forced nonfinite terminal law"),
    )

    evaluation = evaluate_h3(h3_config)

    assert evaluation.result.status is GateStatus.INCONCLUSIVE
    assert evaluation.result.invariants[-1].name == "posterior_condition_envelope"
    assert "nonfinite" in evaluation.result.invariants[-1].detail


@pytest.mark.parametrize(
    ("forced_path", "expected_status"),
    (
        ("finite_miss", GateStatus.FAIL),
        ("nondecisive", GateStatus.INCONCLUSIVE),
    ),
)
def test_actual_gate_path_distinguishes_finite_miss_from_nondecisive_allowance(
    forced_path: str,
    expected_status: GateStatus,
    h3_config: ResolvedConfig,
    h3_evaluation: H3GateEvaluation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_baseline_arms(monkeypatch, h3_evaluation)
    original_pair = gate_module._pair_element

    def forced_pair(*args: object, **kwargs: object) -> dict[str, object]:
        record = original_pair(*args, **kwargs)
        if args[0] == "zero_delta_adequacy":
            if forced_path == "finite_miss":
                record["residual"] = 2.0 * float(record["final_allowance"])
                record["passed"] = False
            else:
                record["final_allowance"] = 0.01 * float(
                    record["decisiveness_scale"]
                )
                record["decisiveness_ratio"] = 0.01
                record["decisive"] = False
        return record

    monkeypatch.setattr(gate_module, "_pair_element", forced_pair)

    evaluation = evaluate_h3(h3_config)

    assert evaluation.result.status is expected_status
    if expected_status is GateStatus.FAIL:
        assert evaluation.result.obligations == ()
        assert not next(
            item
            for item in evaluation.result.invariants
            if item.name == "zero_delta_adequacy"
        ).passed
    else:
        assert any(
            "zero_delta_adequacy" in item
            for item in evaluation.result.obligations
        )


def test_canonical_records_retain_every_element_local_allowance(
    h3_evaluation: H3GateEvaluation,
) -> None:
    expected_counts = {
        "coupled_frozen_reference_agreement": 22,
        "zero_frozen_reference_agreement": 17,
        "pytorch_numpy_canonical_agreement": 40,
    }
    for name, expected_count in expected_counts.items():
        record = h3_evaluation.allowances_by_invariant[name]
        elements = record["elements"]
        assert len(elements) == expected_count
        assert record["aggregation"] == "maximum_element_normalized_residual"
        assert record["maximum_normalized_residual"] == max(
            element["residual"] / element["final_allowance"]
            for element in elements
        )
        assert record["decisive"]
        for element in elements:
            assert tuple(element) == (
                "path",
                "kind",
                "operands",
                "operand_allowances",
                "final_allowance",
                "residual",
                "decisiveness_scale",
                "decisiveness_ratio",
                "decisive",
                "passed",
            )
            assert element["decisive"]
            assert element["decisiveness_ratio"] < 0.01
            assert len(element["operands"]) == 2


def test_terminal_kl_records_have_exact_five_signed_summands(
    h3_evaluation: H3GateEvaluation,
) -> None:
    expected_names = (
        "trace_Jp_Sigmaq",
        "quadratic_mean",
        "minus_dimension",
        "logdet_Jq",
        "minus_logdet_Jp",
    )
    terminal = h3_evaluation.comparisons["terminal"]
    for by_family in terminal.values():
        for record in by_family.values():
            parts = record["kl_parts"]
            assert tuple(part["name"] for part in parts) == expected_names
            values = tuple(part["value"] for part in parts)
            assert record["kl"] == pytest.approx(
                0.5 * math.fsum(values), rel=0.0, abs=0.0
            )
            assert record["kl_absolute_summand_accumulation"] == pytest.approx(
                0.5 * math.fsum(abs(value) for value in values),
                rel=0.0,
                abs=0.0,
            )


def test_evaluation_is_deeply_immutable_and_payload_copies_are_fresh(
    h3_evaluation: H3GateEvaluation,
) -> None:
    with pytest.raises(TypeError):
        h3_evaluation.validation_payload["status"] = "fail"  # type: ignore[index]
    with pytest.raises(TypeError):
        h3_evaluation.allowances_by_invariant[
            "coupled_oracle_gap_minimum"
        ]["final_allowance"] = 9.0  # type: ignore[index]
    oracle_mean = h3_evaluation.oracle_by_fixture["coupled"].mean
    with pytest.raises(ValueError):
        oracle_mean.setflags(write=True)
    with pytest.raises(ValueError):
        oracle_mean[0] = 9.0

    first = h3_validation_payload(h3_evaluation)
    second = h3_validation_payload(h3_evaluation)
    first["status"] = "mutated"
    first["fixtures"]["coupled"]["byte_count"] = -1  # type: ignore[index]

    assert second["status"] == "pass"
    assert second["fixtures"]["coupled"]["byte_count"] > 0  # type: ignore[index]
    json.dumps(second, allow_nan=False)
