from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from verification.h2_budget import path_allowance
from verification.h2_gate import (
    H2_INVARIANT_NAMES,
    H2_NEGATIVE_CONTROL_NAMES,
    evaluate_h2,
    h2_validation_payload,
)
from vfe4.config import resolve_config
from vfe4.types import GateResult, GateStatus, InvariantResult


REPO_ROOT = Path(__file__).resolve().parents[2]

# Independent, literal test oracle. None of these inventories is imported from
# the gate, so removing a family from both production registry builders remains
# visible here.
_EXPECTED_BLOCKS_BY_COMPONENT = (
    ("0-1", "1-3", "0-3-2", "1-5", "0-5-4", "2-3", "4-5"),
    ("0-1", "1-3", "0-3-2", "1-5", "2-5-4", "2-3", "4-5"),
    ("0-1", "1-3", "0-3-2", "3-5", "0-5-4", "2-3", "4-5"),
    ("0-1", "1-3", "0-3-2", "3-5", "2-5-4", "2-3", "4-5"),
)
_EXPECTED_COMPONENT_SCALARS = (
    "q_log_normalizer.info_vs_numpy",
    "p_log_normalizer.info_vs_numpy",
    "q_entropy.info_vs_numpy",
    "gaussian_kl.info_vs_h1",
    "gaussian_kl.info_vs_numpy",
    "gaussian_log_ratio.info_vs_h1",
    "gaussian_log_ratio.info_vs_numpy",
    "source_log_ratio.info_vs_h1",
    "source_log_ratio.info_vs_numpy",
    "emission[0].info_vs_h1",
    "emission[0].info_vs_numpy",
    "emission[1].info_vs_h1",
    "emission[1].info_vs_numpy",
    "complete_value.info_vs_h1",
    "complete_value.info_vs_numpy",
)
_EXPECTED_LOCAL_FIELDS = (
    "expected_log_emission[0]",
    "expected_log_emission[1]",
    "initial_model_kl",
    "initial_state_kl",
    "model_source_kl[0]",
    "model_transition_kl[0]",
    "model_source_kl[1]",
    "model_transition_kl[1]",
    "state_source_kl[0]",
    "state_transition_kl[0]",
    "state_source_kl[1]",
    "state_transition_kl[1]",
    "joint_recognition_entropy",
)
_EXPECTED_CONDITIONS = (
    "condition.component.0.q",
    "condition.component.0.p",
    "condition.component.1.q",
    "condition.component.1.p",
    "condition.component.2.q",
    "condition.component.2.p",
    "condition.component.3.q",
    "condition.component.3.p",
)
_EXPECTED_NEGATIVE_INVARIANTS = (
    "negative.misread_h_as_mu",
    "negative.reversed_log_determinant_ratio",
    "negative.diagonal_inverse_emission_marginal",
    "negative.forbidden_inverse_path",
)


def _literal_expected_h2_invariant_names() -> tuple[str, ...]:
    names = ["fixture.sha256"]
    for component, blocks in enumerate(_EXPECTED_BLOCKS_BY_COMPONENT):
        for law in ("q", "p"):
            prefix = f"component.{component}.{law}"
            names.extend(
                (
                    f"{prefix}.mean.info_vs_h1",
                    f"{prefix}.mean.info_vs_numpy",
                    f"{prefix}.backward.mean",
                )
            )
            for block in blocks:
                names.extend(
                    (
                        f"{prefix}.covariance[{block}].info_vs_h1",
                        f"{prefix}.covariance[{block}].info_vs_numpy",
                        f"{prefix}.backward.covariance[{block}]",
                    )
                )
        names.extend(
            f"component.{component}.{suffix}"
            for suffix in _EXPECTED_COMPONENT_SCALARS
        )
    for field in _EXPECTED_LOCAL_FIELDS:
        names.extend(
            (
                f"aggregate.local.{field}.info_vs_h1",
                f"aggregate.local.{field}.info_vs_numpy",
            )
        )
    names.extend(
        (
            "aggregate.complete_elbo.info_vs_h1_monolithic",
            "aggregate.complete_elbo.info_vs_h1_local",
            "aggregate.complete_elbo.info_vs_numpy_component",
            "aggregate.complete_elbo.info_local_vs_numpy_local",
        )
    )
    names.extend(_EXPECTED_CONDITIONS)
    names.extend(_EXPECTED_NEGATIVE_INVARIANTS)
    return tuple(names)


EXPECTED_H2_INVARIANT_NAMES = _literal_expected_h2_invariant_names()


def _resolved_config(tmp_path: Path):
    return resolve_config(
        {
            "schema_version": 1,
            "objective_schema_version": "vfe4-state-elbo-v1",
            "run": {
                "mode": "verify",
                "seed": 0,
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
                "gates": ["H1"],
                "fixture_id": "h1-v1",
                "quadrature_order": 21,
                "convergence_check_order": 17,
                "maximum_convergence_estimate": 1e-9,
            },
            "artifacts": {"run_root": str(tmp_path / "runs")},
        },
        repo_root=REPO_ROOT,
    )


def test_h2_gate_has_exact_inventory_and_passes_all_three_paths(tmp_path: Path) -> None:
    evaluation = evaluate_h2(_resolved_config(tmp_path))

    assert evaluation.result.status is GateStatus.PASS
    assert evaluation.result.gate == "H2"
    assert len(EXPECTED_H2_INVARIANT_NAMES) == 295
    assert len(set(EXPECTED_H2_INVARIANT_NAMES)) == 295
    assert H2_INVARIANT_NAMES == EXPECTED_H2_INVARIANT_NAMES
    assert (
        tuple(item.name for item in evaluation.result.invariants)
        == EXPECTED_H2_INVARIANT_NAMES
    )
    assert all(item.passed for item in evaluation.result.invariants)
    assert tuple(evaluation.negative_controls) == H2_NEGATIVE_CONTROL_NAMES == (
        "misread_h_as_mu",
        "reversed_log_determinant_ratio",
        "diagonal_inverse_emission_marginal",
        "forbidden_inverse_path",
    )
    assert all(control.passed for control in evaluation.negative_controls.values())
    assert all(
        control.detected and control.residual >= control.decisiveness_limit
        for name, control in evaluation.negative_controls.items()
        if name != "forbidden_inverse_path"
    )
    inverse = evaluation.negative_controls["forbidden_inverse_path"]
    assert inverse.detected
    assert inverse.forbidden_attempts == 0
    assert inverse.injected_attempts == 1
    assert inverse.solve_rhs_widths
    assert max(inverse.solve_rhs_widths) < 6
    assert inverse.selected_column_sets
    assert all(len(columns) < 6 for columns in inverse.selected_column_sets)

    assert len(evaluation.information.components) == 4
    assert len(evaluation.moment.q_components) == 4
    assert len(evaluation.moment.p_components) == 4
    assert len(evaluation.oracle.components) == 4
    assert len(evaluation.comparisons) == 282
    assert max(comparison.residual for comparison in evaluation.comparisons.values()) <= max(
        comparison.allowance for comparison in evaluation.comparisons.values()
    )

    payload = h2_validation_payload(evaluation)
    assert payload["gate_result"] is evaluation.result
    assert payload["fixture_observed_sha256"] == (
        "388e38cc8c16d8b5e2c61919c1e712a134d88fb0bbd8ec1f2939b9859c9a583b"
    )
    assert tuple(payload["comparisons"]) == tuple(evaluation.comparisons)
    assert tuple(payload["negative_controls"]) == H2_NEGATIVE_CONTROL_NAMES


def test_every_mathematical_control_residual_must_be_decisive() -> None:
    import verification.h2_gate as gate_module

    control = gate_module._decisive_control(
        (
            ("decisive", np.array([0.0, 0.0]), np.array([2.0, -2.0])),
            ("indecisive", np.array([1.0, -1.0]), np.array([1.0001, -1.0])),
        )
    )

    assert not control.passed
    assert not control.detected
    assert len(control.residual_records) == 2
    assert control.residual_records[0].passed
    assert not control.residual_records[1].passed
    assert control.weakest_margin == control.residual_records[1].margin < 0.0
    assert control.correct_value == (1.0, -1.0)
    assert control.wrong_value == (1.0001, -1.0)


def test_finite_invariant_failure_precedes_indecisive_control() -> None:
    import verification.h2_gate as gate_module

    control = gate_module._decisive_control(
        (("indecisive", 1.0, 1.0001),)
    )
    invariants = (
        InvariantResult(
            "comparison.finite_failure", False, 2.0, 1.0, "finite miss"
        ),
        InvariantResult(
            "negative.mixed", False, control.residual, control.decisiveness_limit,
            "indecisive control",
        ),
    )

    status, obligations = gate_module._status_and_obligations(
        invariants, {"mixed": control}
    )
    assert status is GateStatus.FAIL
    assert obligations == ()


def test_numpy_local_allowances_use_each_terms_recorded_spd_operands(
    tmp_path: Path,
) -> None:
    import verification.h2_gate as gate_module

    evaluation = evaluate_h2(_resolved_config(tmp_path))
    assert evaluation.oracle is not None
    oracle = evaluation.oracle
    values = gate_module._local_values(oracle.local_terms)
    allowances = gate_module._local_allowances_oracle(oracle)
    for name in _EXPECTED_LOCAL_FIELDS:
        kappas = oracle.local_terms.spd_operand_kappas[name]
        assert kappas
        expected = path_allowance(
            6,
            kappas,
            values[name],
            oracle.local_terms.absolute_summand_accumulation[name],
        )
        assert allowances[name] == expected

    assert len(oracle.local_terms.spd_operand_kappas["initial_model_kl"]) == 2
    assert len(oracle.local_terms.spd_operand_kappas["initial_state_kl"]) == 2
    assert all(
        value > 1.0
        for value in oracle.local_terms.spd_operand_kappas["initial_state_kl"]
    )


@pytest.mark.parametrize("gate", ["H1", "H2"])
def test_generic_gate_result_status_invariants_are_unchanged(gate: str) -> None:
    passing = InvariantResult("finite", True, 0.0, 1.0, "finite residual")
    failing = InvariantResult("finite", False, 2.0, 1.0, "finite residual")
    valid = GateResult(
        gate=gate,
        status=GateStatus.PASS,
        fixture_id="h1-v1",
        residual=0.0,
        calibrated_allowance=1.0,
        measurements={"value": 0.0},
        invariants=(passing,),
        obligations=(),
    )
    assert valid.gate == gate

    with pytest.raises(ValueError, match="pass requires"):
        GateResult(
            gate=gate,
            status=GateStatus.PASS,
            fixture_id="h1-v1",
            residual=2.0,
            calibrated_allowance=1.0,
            measurements={"value": 0.0},
            invariants=(failing,),
            obligations=(),
        )
    with pytest.raises(ValueError, match="fail requires"):
        GateResult(
            gate=gate,
            status=GateStatus.FAIL,
            fixture_id="h1-v1",
            residual=0.0,
            calibrated_allowance=1.0,
            measurements={"value": 0.0},
            invariants=(passing,),
            obligations=(),
        )
    with pytest.raises(ValueError, match="inconclusive results require"):
        GateResult(
            gate=gate,
            status=GateStatus.INCONCLUSIVE,
            fixture_id="h1-v1",
            residual=None,
            calibrated_allowance=None,
            measurements={},
            invariants=(),
            obligations=(),
        )


def test_h2_gate_fails_closed_for_hash_mismatch_and_indecisive_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import verification.h2_gate as gate_module

    monkeypatch.setattr(gate_module, "EXPECTED_H1_FIXTURE_SHA256", "0" * 64)
    mismatch = evaluate_h2(_resolved_config(tmp_path))
    assert mismatch.result.status is GateStatus.INCONCLUSIVE
    assert mismatch.result.obligations

    monkeypatch.undo()
    monkeypatch.setattr(gate_module, "CONTROL_DECISIVENESS", float("inf"))
    indecisive = evaluate_h2(_resolved_config(tmp_path))
    assert indecisive.result.status is GateStatus.INCONCLUSIVE
    assert any("indecisive" in obligation for obligation in indecisive.result.obligations)
