from __future__ import annotations

from pathlib import Path

import pytest

from verification.h2_gate import (
    H2_INVARIANT_NAMES,
    H2_NEGATIVE_CONTROL_NAMES,
    evaluate_h2,
    h2_validation_payload,
)
from vfe4.config import resolve_config
from vfe4.types import GateResult, GateStatus, InvariantResult


REPO_ROOT = Path(__file__).resolve().parents[2]


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
    assert tuple(item.name for item in evaluation.result.invariants) == H2_INVARIANT_NAMES
    assert len(H2_INVARIANT_NAMES) == len(set(H2_INVARIANT_NAMES))
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
    assert len(evaluation.comparisons) > 100
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
