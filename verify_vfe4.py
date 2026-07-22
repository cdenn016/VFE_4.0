"""Click-to-run H1/H2 verifier with one editable configuration mapping."""

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
        "gates": ["H1", "H2"],
        "fixture_id": "h1-v1",
        "quadrature_order": 21,
        "convergence_check_order": 17,
        "maximum_convergence_estimate": 1e-9,
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
        print(f"H1/H2 configuration invalid: {exc}", file=sys.stderr)
        return 2
    return (
        0
        if tuple(item.gate for item in result.gate_results) == ("H1", "H2")
        and all(item.status is GateStatus.PASS for item in result.gate_results)
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(_script_main())
