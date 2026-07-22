"""Click-to-run H1/H2/H3 verifier with one editable configuration mapping."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path

from verification.run_gates import VerificationRunResult, run_verification
from vfe4.artifacts import ArtifactPublicationError
from vfe4.config import resolve_config
from vfe4.types import GateStatus


CONFIG: dict[str, object] = {
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
    "artifacts": {"run_root": "runs"},
}


_REPO_ROOT = Path(__file__).resolve().parent


def main(config: Mapping[str, object] = CONFIG) -> VerificationRunResult:
    resolved = resolve_config(config, repo_root=_REPO_ROOT)
    result = run_verification(resolved)
    for gate_result in result.gate_results:
        print(f"{gate_result.gate}: {gate_result.status.value}")
    print(f"artifact: {result.run_directory}")
    return result


def _script_main() -> int:
    try:
        result = main()
    except ArtifactPublicationError as exc:
        print(f"artifact unavailable: {exc}", file=sys.stderr)
        return 2
    except (TypeError, ValueError) as exc:
        print(f"H1/H2/H3 configuration invalid: {exc}", file=sys.stderr)
        return 2
    return 0 if all(
        item.status is GateStatus.PASS for item in result.gate_results
    ) else 1


if __name__ == "__main__":
    raise SystemExit(_script_main())
