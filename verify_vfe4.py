"""Click-to-run VFE4 verifier with one editable configuration mapping."""

from __future__ import annotations

import copy
import hmac
import json
import sys
from collections.abc import Mapping
from pathlib import Path

from vfe4.types.h4 import H4_PRIMARY_TIMED_BALANCE, H4_PROBLEM_SEEDS
from vfe4.types.h5_schema import (
    H5_FACTOR_INPUT_SCHEMA_SHA256,
    H5_FACTOR_UNIVERSE,
    H5_MODEL_BLOCK_UNIVERSE,
    H5_OBJECTIVE_SCHEMA_SHA256,
    H5_RECOGNITION_COORDINATE_UNIVERSE,
)
from vfe4.types.updates import H5UpdateRule, UpdateLabel
from vfe4.config import H8ValidationConfig
from vfe4.validation.h7_fixture import h7_validation_config_mapping


CONFIG: dict[str, object] = {
    "launcher_schema": "vfe4-verify-click-run-v1",
    "operations": {
        "h1_h5": {
            "enabled": False,
            "authorization": None,
            "publish_prediction_correctness": False,
            "config": {
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
        "gates": ["H1", "H2", "H3", "H4", "H5"],
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
    "h4": {
        "schema_version": "h4-validation-config-v1",
        "solve_protocol": {
            "protocol_id": "h4-single-pass-v1",
            "dtype": "float64",
            "device": "cpu",
            "factor_passes": 1,
            "solver_relative_budget": 1.0e-9,
            "stopping_rule": "complete_schedule_finite_spd",
        },
        "traversal": {
            "horizons": [7, 15, 31],
            "seeds": list(H4_PROBLEM_SEEDS),
            "kinds": ["coupled", "zero_control"],
            "d_z": 4,
            "d_m": 4,
            "dimensions": [64, 128, 256],
            "primary_horizon": 31,
            "primary_kind": "coupled",
            "primary_dimension": 256,
        },
        "timing": {
            "parity_expression": (
                "(horizon_index + seed_index + kind_index + pair_index) % 2 == 0"
            ),
            "warmup_pair_indices": [0, 1, 2],
            "timed_pair_indices": list(range(3, 14)),
            "timed_repetitions_per_problem": 11,
            "warmups_count_toward_balance": False,
            "primary_timed_balance": [list(row) for row in H4_PRIMARY_TIMED_BALANCE],
            "primary_5_ab_6_ba_rows": 10,
            "primary_6_ab_5_ba_rows": 10,
            "primary_timed_ab_total": 110,
            "primary_timed_ba_total": 110,
            "clock": "time.perf_counter_ns",
            "timer_boundary": "fresh_native_solver_call_v1",
            "between_repetitions": "timer_reads_and_preallocated_assignments_only",
        },
        "bootstrap": {
            "seed": 20260721,
            "replicates": 100000,
            "inferential_units": 20,
            "index_low": 0,
            "index_high": 20,
            "endpoint": False,
            "index_dtype": "<i8",
            "index_shape": [100000, 20],
            "statistic": "mean_log_seed_ratio",
            "percentiles": [2.5, 97.5],
            "percentile_method": "linear",
            "percentile_space": "log_then_exp",
            "digest_domain": "vfe4.h4.bootstrap-indices.v1",
            "expected_index_sha256": (
                "a254e18bccc519a719e9f4b409f45cc9ae4a2a321903531cd8fd73433687cd14"
            ),
        },
        "condition_envelope": {
            "posterior_minimum_eigenvalue": 1.0e-6,
            "posterior_maximum_eigenvalue": 1.0e6,
            "posterior_maximum_condition_number": 1.0e8,
            "posterior_minimum_cholesky_pivot": 1.0e-3,
            "posterior_maximum_mean_infinity_norm": 16.0,
            "innovation_minimum_eigenvalue": 1.0e-6,
            "innovation_maximum_eigenvalue": 1.0e6,
            "innovation_maximum_condition_number": 1.0e8,
            "inclusive": True,
        },
        "allowance": {
            "float64_epsilon": 2.220446049250313e-16,
            "rounding_constant": 4096,
            "solver_relative_budget": 1.0e-9,
            "maximum_allowance_scale_fraction": 1.0e-4,
            "decisiveness_comparison": "strict_less_than",
            "element_stream_domain": "vfe4.h4.allowance-element-stream.v1",
            "maximum_chunk_rows": 4096,
        },
        "environment": {
            "device": "cpu",
            "dtype": "float64",
            "intra_op_threads": 1,
            "alter_inter_op_threads": False,
            "cuda_expected": False,
            "gc_policy": "restore_exact_prior_enabled_state",
            "power_policy_field_order": [
                "active_power_scheme",
                "cpu_frequency_governor",
                "energy_performance_preference",
                "low_power_mode",
            ],
            "power_policy_capture": "typed_best_effort_outside_timing",
        },
        "primary_effect_threshold": 0.80,
        "maximum_validation_payload_bytes": 67_108_864,
    },
    "h5": {
        "schema_version": "h5-validation-config-v1",
        "fixture_id": "h5-conditional-update-v1",
        "fixture_schema_version": 1,
        "recognition_family": "continuous_mean_field_conditional_categorical",
        "h1_fixture_id": "h1-v1",
        "h1_fixture_raw_sha256": (
            "388e38cc8c16d8b5e2c61919c1e712a134d88fb0bbd8ec1f2939b9859c9a583b"
        ),
        "update_spec_raw_sha256": (
            "9dd42603419952a2ffa4b6602971240ec00572283557d672ae6ee106c31dd91c"
        ),
        "update_spec_canonical_sha256": (
            "0e4e870dd725aeaec77ffd128ba85dbf619df5b0261b2178e6a115a8970715d6"
        ),
        "objective_schema_sha256": H5_OBJECTIVE_SCHEMA_SHA256,
        "factor_input_schema_version": "h5-factor-input-v1",
        "factor_input_schema_sha256": H5_FACTOR_INPUT_SCHEMA_SHA256,
        "factor_universe": list(H5_FACTOR_UNIVERSE),
        "recognition_coordinate_universe": list(H5_RECOGNITION_COORDINATE_UNIVERSE),
        "model_block_universe": list(H5_MODEL_BLOCK_UNIVERSE),
        "enabled_update_rules": [item.value for item in H5UpdateRule],
        "enabled_update_labels": [
            UpdateLabel.EXACT_COORDINATE.value,
            UpdateLabel.GENERALIZED_EM.value,
            UpdateLabel.NATURAL_GRADIENT_PROPOSAL.value,
        ],
        "positive_case_ids": [
            "exact_gaussian_e_coordinate",
            "exact_categorical_source_coordinate",
            "exact_gaussian_m_coordinate_fixed_recognition",
            "accepted_resolved_generalized_em",
            "rejected_proposal_rollback",
        ],
        "control_ids": [
            "child_factor_omission_detected",
            "emission_factor_omission_detected",
            "unresolved_gem_acceptance_detected",
            "natural_gradient_mislabel_detected",
            "rejection_mutation_detected",
            "changed_input_equal_value_detected",
            "changed_value_unchanged_input_not_affected",
        ],
        "quadrature_orders": [21, 17],
        "allowance_policy": "deterministic_convergence_plus_rounding_v1",
        "rounding_constant": 4096,
        "stochastic_contribution": 0.0,
        "epsilon_delta_formula": "before_total+after_total+subtraction_rounding",
        "mm_proof_artifact": None,
    },
    "artifacts": {"run_root": "runs"},
            },
        },
        "h1_prefix_prior": {
            "enabled": False,
            "authorization": None,
            "config": {},
        },
        "h1_prefix_prior_v2": {
            "enabled": False,
            "authorization": None,
            "config": {
                "schema_version": "h1-prefix-prior-config-v2",
                "operation": "H1-Prefix-Prior",
                "source": "CURRENT_CANDIDATE",
                "fixture": {
                    "fixture_id": "h1-prefix-prior-scorer-v2",
                    "fixture_sha256": (
                        "6b0e855482b8f335bec73e4b0976a1317d7ce4cf3ff050670b3950e271c57fde"
                    ),
                    "base_fixture_sha256": (
                        "388e38cc8c16d8b5e2c61919c1e712a134d88fb0bbd8ec1f2939b9859c9a583b"
                    ),
                },
                "generative_factor_schema_sha256": (
                    "0ab33d1cc790711eee82c598bb853d46ab52662eb31e9433e973978e77d9e375"
                ),
                "scorer_schema": (
                    "parent-specific-pooled-prefix-bilinear-v1"
                ),
                "model": {
                    "horizon": 2,
                    "d_z": 1,
                    "d_m": 1,
                    "vocabulary_size": 3,
                    "state_parent_sets": [[0], [0, 1]],
                    "model_parent_sets": [[0], [0, 1]],
                    "latent_projection_policy": "nonzero_bank_projections",
                    "parent_history_policy": (
                        "active_swapped_distinct_nonzero"
                    ),
                    "prefix_policy": "strictly_prior_tokens",
                },
                "quadrature": {
                    "order": 21,
                    "convergence_check_order": 17,
                    "maximum_convergence_estimate": 1.0e-9,
                },
                "artifact_root": "runs",
            },
        },
        "h6_prefix": {
            "enabled": False,
            "authorization": None,
            "config": {},
        },
        "h6_smc_accuracy": {
            "enabled": False,
            "authorization": None,
            "config": {
                "artifact_root": "runs",
                "run_name": "h6-smc-accuracy-current-candidate",
                "fixture_paths": [
                    "verification/fixtures/h6_smc_finite_01.json",
                    "verification/fixtures/h6_smc_finite_02.json",
                    "verification/fixtures/h6_smc_finite_03.json",
                    "verification/fixtures/h6_smc_finite_04.json",
                ],
                "replicate_seeds": list(range(2026072300, 2026072812)),
                "particle_count": 256,
                "horizon_limit": None,
            },
        },
    },
}

_h7_scientific_config = copy.deepcopy(
    CONFIG["operations"]["h1_h5"]["config"]  # type: ignore[index]
)
_h7_scientific_config["validation"]["gates"] = [  # type: ignore[index]
    "H1",
    "H2",
    "H3",
    "H4",
    "H5",
    "H6-Prefix",
    "H7",
]
_h7_scientific_config["h7"] = h7_validation_config_mapping()
CONFIG["operations"]["h7"] = {  # type: ignore[index]
    "enabled": False,
    "authorization": None,
    "config": _h7_scientific_config,
}
CONFIG["operations"]["h8_preflight"] = {  # type: ignore[index]
    "enabled": False,
    "authorization": None,
    "config": {
        "schema_version": "h8-preflight-config-v1",
        "operation": "H8-Preflight",
        "target_operation": "h8",
        "inspection_policy": "metadata_only",
        "write_artifact": False,
    },
}
_h8_scientific_config = copy.deepcopy(_h7_scientific_config)
_h8_scientific_config["validation"]["gates"].append("H8")  # type: ignore[index]
_h8_scientific_config["h8"] = json.loads(
    H8ValidationConfig.create().canonical_json
)
CONFIG["operations"]["h8"] = {  # type: ignore[index]
    "enabled": False,
    "authorization": None,
    "config": _h8_scientific_config,
}
del _h8_scientific_config
del _h7_scientific_config


_REPO_ROOT = Path(__file__).resolve().parent
_VERIFY_AUTHORIZATIONS = {
    "h1_h5": "AUTHORIZE_VFE4_H1_H5_VERIFICATION_V1",
    "h1_prefix_prior": "AUTHORIZE_VFE4_H1_PREFIX_PRIOR_V1",
    "h1_prefix_prior_v2": "AUTHORIZE_VFE4_H1_PREFIX_PRIOR_SCORER_V2",
    "h6_prefix": "AUTHORIZE_VFE4_H6_PREFIX_BOUNDED_WORKLOAD_V3",
    "h6_smc_accuracy": "AUTHORIZE_VFE4_H6_SMC_ACCURACY_FULL_GRID_V1",
    "h7": "AUTHORIZE_VFE4_H7_FRAME_COVARIANCE_V1",
    "h8_preflight": "AUTHORIZE_VFE4_H8_ZERO_COMPUTE_PREFLIGHT_V1",
    "h8": "AUTHORIZE_VFE4_H8_SPARSE_SCALE_RUNTIME_V1",
}
_VERIFY_OPERATION_NAMES = tuple(_VERIFY_AUTHORIZATIONS)


def _mapping(value: object, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(
        type(key) is not str for key in value
    ):
        raise ValueError(f"{location} must be a string-keyed mapping")
    return value


def project_h1_h5_compatibility_config(
    config: Mapping[str, object] = CONFIG,
) -> Mapping[str, object]:
    """Purely project the H7 scientific config onto the legacy H1--H5 prefix."""

    operations = _mapping(_mapping(config, "CONFIG")["operations"], "operations")
    h7_entry = _mapping(operations["h7"], "operations.h7")
    raw_h7 = copy.deepcopy(dict(_mapping(h7_entry["config"], "operations.h7.config")))
    from vfe4.config import resolve_config

    resolved_h7 = resolve_config(raw_h7, repo_root=_REPO_ROOT)
    resolved_h7_validation = getattr(resolved_h7, "validation", None)
    if resolved_h7_validation is None or resolved_h7_validation.gates != (
        "H1",
        "H2",
        "H3",
        "H4",
        "H5",
        "H6-Prefix",
        "H7",
    ):
        raise ValueError("H1--H5 projection requires the exact H7 config")
    raw_h7["validation"]["gates"] = [  # type: ignore[index]
        "H1",
        "H2",
        "H3",
        "H4",
        "H5",
    ]
    raw_h7.pop("h7")
    projected = resolve_config(raw_h7, repo_root=_REPO_ROOT)
    projected_validation = getattr(projected, "validation", None)
    if projected_validation is None or projected_validation.gates != (
        "H1",
        "H2",
        "H3",
        "H4",
        "H5",
    ):
        raise ValueError("H1--H5 compatibility projection changed its prefix")
    return raw_h7


def _selected_operation(
    config: Mapping[str, object],
) -> tuple[str, Mapping[str, object], bool] | None:
    if set(config) != {"launcher_schema", "operations"}:
        raise ValueError("verify CONFIG has unknown or missing root keys")
    if config["launcher_schema"] != "vfe4-verify-click-run-v1":
        raise ValueError("verify CONFIG launcher_schema is unsupported")
    operations = _mapping(config["operations"], "operations")
    if tuple(operations) != _VERIFY_OPERATION_NAMES:
        raise ValueError("verify CONFIG operations are incomplete or reordered")
    enabled: list[tuple[str, Mapping[str, object]]] = []
    for name in _VERIFY_OPERATION_NAMES:
        entry = _mapping(operations[name], f"operations.{name}")
        expected_fields = {"enabled", "authorization", "config"}
        if name == "h1_h5":
            expected_fields.add("publish_prediction_correctness")
        if set(entry) != expected_fields:
            raise ValueError(f"operations.{name} has unknown or missing keys")
        if type(entry["enabled"]) is not bool:
            raise ValueError(f"operations.{name}.enabled must be boolean")
        if name == "h1_h5" and type(
            entry["publish_prediction_correctness"]
        ) is not bool:
            raise ValueError(
                "operations.h1_h5.publish_prediction_correctness must be boolean"
            )
        _mapping(entry["config"], f"operations.{name}.config")
        if entry["enabled"]:
            enabled.append((name, entry))
    if not enabled:
        return None
    if len(enabled) != 1:
        raise ValueError("enable exactly one verify operation")
    name, entry = enabled[0]
    authorization = entry["authorization"]
    if type(authorization) is not str or not hmac.compare_digest(
        authorization,
        _VERIFY_AUTHORIZATIONS[name],
    ):
        raise PermissionError(
            f"operations.{name}.authorization does not equal its explicit phrase"
        )
    publish_prediction_correctness = False
    if name == "h1_h5":
        selected_publish_value = entry["publish_prediction_correctness"]
        if type(selected_publish_value) is not bool:
            raise ValueError(
                "operations.h1_h5.publish_prediction_correctness must be boolean"
            )
        publish_prediction_correctness = selected_publish_value
    return (
        name,
        _mapping(entry["config"], f"operations.{name}.config"),
        publish_prediction_correctness,
    )


def _run_h1_h5(
    scientific: Mapping[str, object],
    *,
    publish_prediction_correctness: bool,
) -> object:
    from verification.run_gates import run_verification
    from vfe4.config import resolve_config

    resolved = resolve_config(scientific, repo_root=_REPO_ROOT)
    result = run_verification(  # type: ignore[arg-type]
        resolved,
        publish_prediction_correctness=publish_prediction_correctness,
    )
    for gate_result in result.gate_results:
        print(f"{gate_result.gate}: {gate_result.status.value}")
    print(f"artifact: {result.run_directory}")
    for gate, root, manifest_sha256 in result.prediction_correctness_artifacts:
        print(
            f"prediction correctness {gate}: {root} "
            f"(manifest {manifest_sha256})"
        )
    return result


def _run_projected(
    operation: str,
    scientific: Mapping[str, object],
) -> object:
    if operation == "h8":
        from verification.run_gates import (
            parse_h8_reference_registry_bytes,
            run_h8_verification,
        )
        from vfe4.artifacts import current_source_identity
        from vfe4.config import bind_h8_current_refs, resolve_config

        unbound = resolve_config(scientific, repo_root=_REPO_ROOT)
        git_head, dirty_digest, _ = current_source_identity(
            _REPO_ROOT,
            unbound.artifacts.run_root,
        )
        registry_path = (
            _REPO_ROOT
            / ".verification"
            / f"h8-current-candidate-{git_head}-refs.json"
        )
        if (
            not registry_path.is_file()
            or registry_path.is_symlink()
            or registry_path.parent.resolve(strict=False)
            != (_REPO_ROOT / ".verification").resolve(strict=False)
        ):
            raise ValueError(
                "the exact current-HEAD H8 reference registry is unavailable"
            )
        registry_bytes = registry_path.read_bytes()
        refs = parse_h8_reference_registry_bytes(registry_bytes)
        if (
            refs.candidate_head != git_head
            or refs.candidate_dirty_digest != dirty_digest
        ):
            raise ValueError("the exact current-HEAD H8 registry is stale")
        resolved = bind_h8_current_refs(scientific, refs)
        result = run_h8_verification(
            resolved,
            registry_path=registry_path,
            registry_bytes=registry_bytes,
        )
        for name in (
            "h1_h5",
            "h1_prefix_prior",
            "h6_prefix",
            "h7",
            "h6_prediction",
        ):
            print(f"{name}: {getattr(refs, name).status}")
        print(f"H8: {result.gate_results[0].status.value}")
        print(f"artifact: {result.run_directory}")
        return result

    if operation == "h7":
        from verification.run_gates import run_h7_verification
        from vfe4.config import resolve_config

        resolved = resolve_config(scientific, repo_root=_REPO_ROOT)
        result = run_h7_verification(resolved)  # type: ignore[arg-type]
        for gate_result in result.gate_results:
            print(f"{gate_result.gate}: {gate_result.status.value}")
        print(f"artifact: {result.run_directory}")
        return result

    from vfe4.artifacts.h6 import (
        project_h1_prefix_prior_config,
        project_h1_prefix_prior_v2_config,
        project_h6_prefix_config,
    )
    from verification.projected_runner import run_projected_current_candidate

    if operation == "h1_prefix_prior_v2":
        from vfe4.artifacts.provenance import current_source_identity

        bound = copy.deepcopy(dict(scientific))
        artifact_root = Path(bound["artifact_root"])
        if not artifact_root.is_absolute():
            artifact_root = _REPO_ROOT / artifact_root
        git_head, dirty_digest, source_sha256 = current_source_identity(
            _REPO_ROOT,
            artifact_root,
        )
        bound["source"] = {
            "git_head": git_head,
            "dirty_digest": dirty_digest,
            "source_sha256": source_sha256,
        }
        projected = project_h1_prefix_prior_v2_config(bound)
    elif operation == "h1_prefix_prior":
        projected = project_h1_prefix_prior_config(scientific)
    else:
        projected = project_h6_prefix_config(scientific)
    result = run_projected_current_candidate(
        config=projected,
        junit_sha256=None,
        predecessor_refs={},
    )
    print(f"artifact: {result.artifact_path}")
    return result


def _run_h6_smc_accuracy(scientific: Mapping[str, object]) -> object:
    from verification.h6_smc_gate import (
        publish_h6_smc_accuracy_artifact,
    )

    expected = {
        "artifact_root",
        "run_name",
        "fixture_paths",
        "replicate_seeds",
        "particle_count",
        "horizon_limit",
    }
    if set(scientific) != expected:
        raise ValueError(
            "h6_smc_accuracy config has unknown or missing keys"
        )
    raw_paths = scientific["fixture_paths"]
    raw_seeds = scientific["replicate_seeds"]
    if (
        type(raw_paths) is not list
        or len(raw_paths) != 4
        or any(type(value) is not str or not value for value in raw_paths)
    ):
        raise ValueError("h6_smc_accuracy fixture_paths must list four paths")
    if (
        type(raw_seeds) is not list
        or not raw_seeds
        or any(type(value) is not int for value in raw_seeds)
    ):
        raise ValueError("h6_smc_accuracy replicate_seeds must be an integer list")
    artifact_root = scientific["artifact_root"]
    run_name = scientific["run_name"]
    particle_count = scientific["particle_count"]
    horizon_limit = scientific["horizon_limit"]
    if type(artifact_root) is not str or not artifact_root:
        raise ValueError("h6_smc_accuracy artifact_root must be a path string")
    if type(run_name) is not str or not run_name:
        raise ValueError("h6_smc_accuracy run_name must be nonempty")
    if type(particle_count) is not int or particle_count <= 0:
        raise ValueError("h6_smc_accuracy particle_count must be positive")
    if horizon_limit is not None and (
        type(horizon_limit) is not int or horizon_limit <= 0
    ):
        raise ValueError("h6_smc_accuracy horizon_limit must be null or positive")
    fixture_paths = tuple(
        (Path(value) if Path(value).is_absolute() else _REPO_ROOT / value)
        for value in raw_paths
    )
    runs = Path(artifact_root)
    if not runs.is_absolute():
        runs = _REPO_ROOT / runs
    report, root = publish_h6_smc_accuracy_artifact(
        repository_root=_REPO_ROOT,
        artifact_root=runs,
        run_name=run_name,
        fixture_paths=fixture_paths,
        replicate_seeds=tuple(raw_seeds),
        particle_count=particle_count,
        horizon_limit=horizon_limit,
    )
    print(f"H6-SMC-Accuracy: {report.status}")
    print(f"artifact: {root}")
    return report, root


def _run_h8_preflight(
    request: Mapping[str, object],
    *,
    target_scientific: Mapping[str, object],
) -> object:
    from verification.h8_preflight import (
        capture_current_candidate,
        inspect_h8_preflight,
    )

    candidate = capture_current_candidate(
        repository_root=_REPO_ROOT,
        target_scientific_config=target_scientific,
    )
    result = inspect_h8_preflight(
        repository_root=_REPO_ROOT,
        target_scientific_config=target_scientific,
        request=request,
        candidate=candidate,
    )
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    return result


def main(config: Mapping[str, object] = CONFIG) -> object | None:
    selected = _selected_operation(_mapping(config, "CONFIG"))
    if selected is None:
        return None
    operation, scientific, publish_prediction_correctness = selected
    if operation == "h1_h5":
        return _run_h1_h5(
            scientific,
            publish_prediction_correctness=publish_prediction_correctness,
        )
    if operation == "h6_smc_accuracy":
        return _run_h6_smc_accuracy(scientific)
    if operation == "h8_preflight":
        operations = _mapping(
            _mapping(config, "CONFIG")["operations"],
            "operations",
        )
        h8_entry = _mapping(operations["h8"], "operations.h8")
        return _run_h8_preflight(
            scientific,
            target_scientific=_mapping(
                h8_entry["config"],
                "operations.h8.config",
            ),
        )
    return _run_projected(operation, scientific)


def _script_main() -> int:
    try:
        result = main()
    except (OSError, PermissionError, RuntimeError, TypeError, ValueError) as exc:
        print(f"VFE4 verification configuration invalid: {exc}", file=sys.stderr)
        return 2
    if result is None:
        print("VFE4 verify launcher is idle; enable exactly one CONFIG operation.")
        return 0
    gate_results = getattr(result, "gate_results", None)
    if gate_results is None:
        return 0
    return 0 if all(
        getattr(item.status, "value", None) == "pass" for item in gate_results
    ) else 1


if __name__ == "__main__":
    raise SystemExit(_script_main())
