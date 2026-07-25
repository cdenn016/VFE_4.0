from __future__ import annotations

import copy
import dataclasses
import hashlib
import inspect
import json
from pathlib import Path
from typing import Callable

import pytest

from vfe4.config import (
    H3ValidationConfig,
    H4ValidationConfig,
    H5ValidationConfig,
    H7ValidationConfig,
    H8ValidationConfig,
    ResolvedConfig,
    bind_h8_current_refs,
    project_h7_compatibility_config,
    resolve_config,
    resolve_h4_validation_config,
    resolve_h8_validation_config,
)
from vfe4.config.control_paths import is_repository_control_path
from vfe4.types.h3 import (
    H3DecisionConfig,
    H3InitializationConfig,
    H3OptimizationConfig,
)
from vfe4.types.h4 import H4_PRIMARY_TIMED_BALANCE, H4_PROBLEM_SEEDS, H4SolveProtocol
from vfe4.types.h5_schema import (
    H5_FACTOR_INPUT_SCHEMA_SHA256,
    H5_FACTOR_INPUT_SCHEMA_VERSION,
    H5_FACTOR_UNIVERSE,
    H5_MODEL_BLOCK_UNIVERSE,
    H5_OBJECTIVE_SCHEMA_SHA256,
    H5_RECOGNITION_COORDINATE_UNIVERSE,
)
from vfe4.types.h7 import H7PredecessorReference
from vfe4.types.h8 import (
    H8_CORRECTNESS_CASES,
    H8_H7_PLAN_SHA256,
    H8_INTERPRETATION_SHA256,
    H8_PRODUCTION_SEEDS,
    H8_PROFILER_API_CONTRACT_SHA256,
)
from vfe4.types.updates import H5_RULE_CONTRACTS, H5UpdateRule, UpdateLabel
from vfe4.validation.h5_update_spec import EXPECTED_H5_UPDATE_SPEC_RAW_SHA256
from vfe4.validation.h7_fixture import (
    H7_DENSITY_PROBE_TABLE_RAW_SHA256,
    h7_validation_config_mapping,
)


def _raw_h4_section() -> dict[str, object]:
    return {
        "schema_version": "h4-validation-config-v1",
        "solve_protocol": {
            "protocol_id": "h4-single-pass-v1", "dtype": "float64",
            "device": "cpu", "factor_passes": 1,
            "solver_relative_budget": 1.0e-9,
            "stopping_rule": "complete_schedule_finite_spd",
        },
        "traversal": {
            "horizons": [7, 15, 31], "seeds": list(H4_PROBLEM_SEEDS),
            "kinds": ["coupled", "zero_control"], "d_z": 4, "d_m": 4,
            "dimensions": [64, 128, 256], "primary_horizon": 31,
            "primary_kind": "coupled", "primary_dimension": 256,
        },
        "timing": {
            "parity_expression": "(horizon_index + seed_index + kind_index + pair_index) % 2 == 0",
            "warmup_pair_indices": [0, 1, 2],
            "timed_pair_indices": list(range(3, 14)),
            "timed_repetitions_per_problem": 11,
            "warmups_count_toward_balance": False,
            "primary_timed_balance": [list(row) for row in H4_PRIMARY_TIMED_BALANCE],
            "primary_5_ab_6_ba_rows": 10, "primary_6_ab_5_ba_rows": 10,
            "primary_timed_ab_total": 110, "primary_timed_ba_total": 110,
            "clock": "time.perf_counter_ns",
            "timer_boundary": "fresh_native_solver_call_v1",
            "between_repetitions": "timer_reads_and_preallocated_assignments_only",
        },
        "bootstrap": {
            "seed": 20260721, "replicates": 100000, "inferential_units": 20,
            "index_low": 0, "index_high": 20, "endpoint": False,
            "index_dtype": "<i8", "index_shape": [100000, 20],
            "statistic": "mean_log_seed_ratio", "percentiles": [2.5, 97.5],
            "percentile_method": "linear", "percentile_space": "log_then_exp",
            "digest_domain": "vfe4.h4.bootstrap-indices.v1",
            "expected_index_sha256": "a254e18bccc519a719e9f4b409f45cc9ae4a2a321903531cd8fd73433687cd14",
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
            "float64_epsilon": 2.220446049250313e-16, "rounding_constant": 4096,
            "solver_relative_budget": 1.0e-9,
            "maximum_allowance_scale_fraction": 1.0e-4,
            "decisiveness_comparison": "strict_less_than",
            "element_stream_domain": "vfe4.h4.allowance-element-stream.v1",
            "maximum_chunk_rows": 4096,
        },
        "environment": {
            "device": "cpu", "dtype": "float64", "intra_op_threads": 1,
            "alter_inter_op_threads": False, "cuda_expected": False,
            "gc_policy": "restore_exact_prior_enabled_state",
            "power_policy_field_order": ["active_power_scheme", "cpu_frequency_governor", "energy_performance_preference", "low_power_mode"],
            "power_policy_capture": "typed_best_effort_outside_timing",
        },
        "primary_effect_threshold": 0.80,
        "maximum_validation_payload_bytes": 67_108_864,
    }


def test_standalone_h4_resolver_freezes_complete_protocol_and_hash() -> None:
    raw = _raw_h4_section()
    resolved = resolve_h4_validation_config(raw)
    assert type(resolved) is H4ValidationConfig
    assert type(resolved.solve_protocol) is H4SolveProtocol
    assert resolved.solve_protocol.solver_relative_budget == resolved.allowance.solver_relative_budget
    assert resolved.timing.primary_timed_balance == H4_PRIMARY_TIMED_BALANCE
    assert resolved.maximum_validation_payload_bytes == 67_108_864
    assert hashlib.sha256(resolved.canonical_json.encode("utf-8")).hexdigest() == resolved.config_sha256
    raw["timing"]["warmup_pair_indices"] = [0, 2, 1]  # type: ignore[index]
    with pytest.raises(ValueError):
        resolve_h4_validation_config(raw)


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
            "gates": ["H1", "H2"],
            "fixture_id": "h1-v1",
            "quadrature_order": 21,
            "convergence_check_order": 17,
            "maximum_convergence_estimate": 1e-9,
        },
        "artifacts": {"run_root": "runs"},
    }


def _raw_h3() -> dict[str, object]:
    return {
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
    }


def _raw_h3_config() -> dict[str, object]:
    raw = _raw_config()
    raw["validation"]["gates"] = ["H1", "H2", "H3"]  # type: ignore[index]
    raw["h3"] = _raw_h3()
    return raw


def _raw_h5_section() -> dict[str, object]:
    return {
        "schema_version": "h5-validation-config-v1",
        "fixture_id": "h5-conditional-update-v1",
        "fixture_schema_version": 1,
        "recognition_family": "continuous_mean_field_conditional_categorical",
        "h1_fixture_id": "h1-v1",
        "h1_fixture_raw_sha256": (
            "388e38cc8c16d8b5e2c61919c1e712a134d88fb0bbd8ec1f2939b9859c9a583b"
        ),
        "update_spec_raw_sha256": EXPECTED_H5_UPDATE_SPEC_RAW_SHA256,
        "update_spec_canonical_sha256": (
            "0e4e870dd725aeaec77ffd128ba85dbf619df5b0261b2178e6a115a8970715d6"
        ),
        "objective_schema_sha256": H5_OBJECTIVE_SCHEMA_SHA256,
        "factor_input_schema_version": H5_FACTOR_INPUT_SCHEMA_VERSION,
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
    }


def _raw_h5_config() -> dict[str, object]:
    raw = _raw_h3_config()
    raw["validation"]["gates"] = ["H1", "H2", "H3", "H4", "H5"]  # type: ignore[index]
    raw["h4"] = _raw_h4_section()
    raw["h5"] = _raw_h5_section()
    return raw


def _raw_h7_config() -> dict[str, object]:
    raw = _raw_h5_config()
    raw["validation"]["gates"] = [  # type: ignore[index]
        "H1", "H2", "H3", "H4", "H5", "H6-Prefix", "H7"
    ]
    raw["h7"] = h7_validation_config_mapping()
    return raw


def _raw_h8_section() -> dict[str, object]:
    return json.loads(H8ValidationConfig.create().canonical_json)


def _raw_h8_config() -> dict[str, object]:
    raw = _raw_h7_config()
    raw["validation"]["gates"] = [  # type: ignore[index]
        "H1",
        "H2",
        "H3",
        "H4",
        "H5",
        "H6-Prefix",
        "H7",
        "H8",
    ]
    raw["h8"] = _raw_h8_section()
    return raw


def _h7_reference(name: str) -> H7PredecessorReference:
    digest = hashlib.sha256(name.encode("ascii")).hexdigest()
    return H7PredecessorReference.create(
        artifact_path=f"runs/{name}",
        git_head="a" * 40,
        dirty_digest="b" * 64,
        junit_sha256="c" * 64,
        manifest_sha256=digest,
        payload_hashes={f"validation/{name}.json": digest},
        ledger_path=f".verification/{name}-ledger.json",
        ledger_sha256=digest,
    )


def test_h8_static_protocol_and_h7_projection_are_exact_and_pure() -> None:
    raw = _raw_h8_config()
    before = copy.deepcopy(raw)
    resolved = resolve_h8_validation_config(raw["h8"])  # type: ignore[arg-type]
    refs = {
        "h1_h5": _h7_reference("h1_h5"),
        "h1_prefix_prior": _h7_reference("h1_prefix_prior"),
        "h6_prefix": _h7_reference("h6_prefix"),
    }
    projected = project_h7_compatibility_config(raw, refs)

    assert resolved.T == 128
    assert (resolved.N, resolved.K, resolved.d_z, resolved.d_m) == (
        129,
        20,
        20,
        20,
    )
    assert (resolved.b, resolved.D, resolved.V) == (40, 5160, 3)
    assert resolved.seeds == H8_PRODUCTION_SEEDS
    assert resolved.cold_repetitions == 5
    assert resolved.max_seconds == 60.0
    assert resolved.max_process_incremental_mib == 128
    assert resolved.max_torch_population_mib == 64
    assert (resolved.max_rhs_width, resolved.sample_width) == (40, 1)
    assert resolved.correctness_seed_table == H8_CORRECTNESS_CASES
    assert resolved.profiler_api_contract_sha256 == (
        H8_PROFILER_API_CONTRACT_SHA256
    )
    assert resolved.interpretation_sha256 == H8_INTERPRETATION_SHA256
    assert resolved.h7_plan_sha256 == H8_H7_PLAN_SHA256
    assert projected == _raw_h7_config()
    assert raw == before
    assert tuple(inspect.signature(bind_h8_current_refs).parameters) == (
        "raw_h8_config",
        "refs",
    )


def test_h8_static_protocol_rejects_drift_and_earlier_prefix_section() -> None:
    for field, changed_value in (
        ("K", 40),
        ("d_m", 19),
        ("cold_repetitions", 6),
        ("torch_version", "2.9.0"),
        ("correctness_seed_table", list(reversed(H8_CORRECTNESS_CASES))),
        ("interpretation_sha256", "0" * 64),
    ):
        raw_h8 = _raw_h8_section()
        raw_h8[field] = changed_value
        with pytest.raises(ValueError, match="h8|H8"):
            resolve_h8_validation_config(raw_h8)

    earlier = _raw_config()
    earlier["h8"] = _raw_h8_section()
    with pytest.raises(ValueError, match="config|h8"):
        resolve_config(earlier, repo_root=Path("C:/repo"))


def test_h7_config_is_exact_and_shorter_prefixes_do_not_read_h7_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = _raw_h7_config()
    resolved = resolve_config(raw, repo_root=tmp_path)
    assert resolved.validation.gates == (
        "H1", "H2", "H3", "H4", "H5", "H6-Prefix", "H7"
    )
    assert isinstance(resolved.h7, H7ValidationConfig)
    assert tuple(spec.trial_id for spec in resolved.h7.required_trial_specs) == (
        "scalar-base-transformed",
        "scalar-internal-transformed",
        "matrix-identity-base-transformed",
        "matrix-identity-internal-transformed",
        "matrix-nonidentity-base-transformed",
        "matrix-nonidentity-internal-transformed",
        "matrix-fixed-decoder-centered-stabilizer",
        "matrix-fixed-decoder-outside-stabilizer",
    )
    assert resolved.h7.predecessor_keys == (
        "h1_h5", "h1_prefix_prior", "h6_prefix"
    )
    assert (
        resolved.h7.density_probe_table_raw_sha256
        == H7_DENSITY_PROBE_TABLE_RAW_SHA256
    )

    changed = copy.deepcopy(raw)
    changed["h7"]["actions"]["internal"][0][0][0] = 1.2500000001  # type: ignore[index]
    with pytest.raises(ValueError):
        resolve_config(changed, repo_root=tmp_path)
    changed = copy.deepcopy(raw)
    changed["h7"]["oracle_decimal_precision"] = 100.0  # type: ignore[index]
    with pytest.raises(ValueError):
        resolve_config(changed, repo_root=tmp_path)
    changed = copy.deepcopy(raw)
    changed["h7"]["gauss_hermite_orders"][0] = 41.0  # type: ignore[index]
    with pytest.raises(ValueError):
        resolve_config(changed, repo_root=tmp_path)
    changed = copy.deepcopy(raw)
    changed["validation"]["gates"].append("H8")  # type: ignore[index]
    with pytest.raises(ValueError):
        resolve_config(changed, repo_root=tmp_path)

    import pathlib

    shorter = _raw_h5_config()
    monkeypatch.setattr(
        pathlib.Path,
        "read_bytes",
        lambda _path: pytest.fail("shorter prefix read H7 fixture bytes"),
    )
    shorter_resolved = resolve_config(shorter, repo_root=tmp_path)
    assert shorter_resolved.h7 is None
    assert '"h7"' not in shorter_resolved.canonical_json


def test_h7_launcher_projection_is_pure_and_h6_interfaces_remain_frozen() -> None:
    import verify_vfe4
    from vfe4.artifacts.h6 import (
        CandidateArtifactReference,
        project_h6_prefix_config,
        run_projected_current_candidate,
    )

    before = copy.deepcopy(verify_vfe4.CONFIG)
    projected = verify_vfe4.project_h1_h5_compatibility_config(
        verify_vfe4.CONFIG
    )

    assert projected["validation"]["gates"] == [
        "H1", "H2", "H3", "H4", "H5"
    ]
    assert "h7" not in projected and "h8" not in projected
    assert verify_vfe4.CONFIG == before
    assert "h8" not in verify_vfe4.CONFIG["operations"]
    assert tuple(inspect.signature(project_h6_prefix_config).parameters) == (
        "raw_config",
    )
    runner = inspect.signature(run_projected_current_candidate)
    assert tuple(runner.parameters) == (
        "config",
        "junit_sha256",
        "predecessor_refs",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in runner.parameters.values()
    )
    assert runner.return_annotation in (
        CandidateArtifactReference,
        "CandidateArtifactReference",
    )


def test_coupled_h1_h5_config_reuses_h4_and_freezes_h5_without_fixture_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pathlib

    raw = _raw_h5_config()
    standalone_h4 = resolve_h4_validation_config(raw["h4"])  # type: ignore[arg-type]
    monkeypatch.setattr(
        pathlib.Path,
        "read_bytes",
        lambda _path: pytest.fail("configuration resolution performed fixture I/O"),
    )

    resolved = resolve_config(raw, repo_root=tmp_path)

    assert resolved.validation.gates == ("H1", "H2", "H3", "H4", "H5")
    assert resolved.h4 == standalone_h4
    assert resolved.h4.canonical_json == standalone_h4.canonical_json
    assert resolved.h4.config_sha256 == standalone_h4.config_sha256
    assert resolved.h4.solve_protocol == standalone_h4.solve_protocol
    assert isinstance(resolved.h5, H5ValidationConfig)
    assert resolved.h5.update_spec_raw_sha256 == EXPECTED_H5_UPDATE_SPEC_RAW_SHA256
    assert resolved.h5.factor_universe == H5_FACTOR_UNIVERSE
    assert resolved.h5.recognition_coordinate_universe == H5_RECOGNITION_COORDINATE_UNIVERSE
    assert resolved.h5.model_block_universe == H5_MODEL_BLOCK_UNIVERSE
    assert resolved.h5.enabled_update_rules == tuple(H5UpdateRule)
    assert tuple(H5_RULE_CONTRACTS[rule] for rule in resolved.h5.enabled_update_rules) == (
        (UpdateLabel.EXACT_COORDINATE, ("q[z0]",), (), (1.0,)),
        (UpdateLabel.EXACT_COORDINATE, ("q[source_row_a2]",), (), (1.0,)),
        (UpdateLabel.EXACT_COORDINATE, (), ("theta[state_transition_2]",), (1.0,)),
        (
            UpdateLabel.GENERALIZED_EM,
            (),
            ("theta[emission_1]",),
            (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625,
             0.0078125, 0.00390625, 0.001953125, 0.0009765625),
        ),
        (UpdateLabel.NATURAL_GRADIENT_PROPOSAL, ("q[z1]",), (), (64.0,)),
    )
    assert resolved.h5.enabled_update_labels == (
        UpdateLabel.EXACT_COORDINATE,
        UpdateLabel.GENERALIZED_EM,
        UpdateLabel.NATURAL_GRADIENT_PROPOSAL,
    )
    assert resolved.h5.mm_proof_artifact is None
    assert "missing_mm" not in resolved.h5.canonical_json
    projected = json.loads(resolved.canonical_json)["h4"]
    assert json.dumps(projected, sort_keys=True, separators=(",", ":")) == resolved.h4.canonical_json


@pytest.mark.parametrize(
    "gates",
    [
        ["H1", "H2", "H3", "H4"],
        ["H1", "H2", "H3", "H5"],
        ["H1", "H2", "H3", "H5", "H4"],
        ["H1", "H2", "H3", "H4", "H5", "H5"],
    ],
)
def test_coupled_prefix_is_all_or_nothing(tmp_path: Path, gates: list[str]) -> None:
    raw = _raw_h5_config()
    raw["validation"]["gates"] = gates  # type: ignore[index]
    with pytest.raises(ValueError, match="validation.gates"):
        resolve_config(raw, repo_root=tmp_path)


@pytest.mark.parametrize("missing", ["h4", "h5", "both"])
def test_coupled_prefix_requires_both_typed_sections(
    tmp_path: Path, missing: str
) -> None:
    raw = _raw_h5_config()
    if missing in ("h4", "both"):
        raw.pop("h4")
    if missing in ("h5", "both"):
        raw.pop("h5")
    with pytest.raises(ValueError, match="h4|h5"):
        resolve_config(raw, repo_root=tmp_path)


def test_h5_rejects_unsupported_mm_during_resolution(tmp_path: Path) -> None:
    raw = _raw_h5_config()
    raw["h5"]["enabled_update_labels"].append(UpdateLabel.VALID_MM.value)  # type: ignore[index,union-attr]
    with pytest.raises(ValueError, match="VALID_MM|valid_mm|MM"):
        resolve_config(raw, repo_root=tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("update_spec_raw_sha256", EXPECTED_H5_UPDATE_SPEC_RAW_SHA256[:12]),
        ("update_spec_canonical_sha256", "0" * 64),
        ("objective_schema_sha256", "0" * 64),
        ("factor_input_schema_sha256", "0" * 64),
        ("quadrature_orders", [17, 21]),
        ("allowance_policy", "stochastic"),
        ("rounding_constant", 4095),
        ("stochastic_contribution", 1.0e-12),
        ("epsilon_delta_formula", "after-before"),
        ("enabled_update_rules", [item.value for item in reversed(tuple(H5UpdateRule))]),
    ],
)
def test_h5_resolution_rejects_frozen_identity_and_budget_drift(
    tmp_path: Path, field: str, value: object
) -> None:
    raw = _raw_h5_config()
    raw["h5"][field] = value  # type: ignore[index]
    with pytest.raises(ValueError, match="h5|H5"):
        resolve_config(raw, repo_root=tmp_path)


def test_h5_documentation_uses_exact_manuscripts_path_case() -> None:
    root = Path(__file__).resolve().parents[2]
    text = "\n".join(
        (root / relative).read_text(encoding="utf-8")
        for relative in (
            "README.md",
            "docs/preregistrations/2026-07-21-h5-update-coherence.md",
        )
    )
    assert "Manuscripts/VFE4_gauge_causal_elbo_whitepaper.tex" in text
    assert "Manuscripts/MAgent_exact_elbo_whitepaper.tex" in text
    assert "manuscripts/VFE4_gauge_causal_elbo_whitepaper.tex" not in text
    assert "manuscripts/MAgent_exact_elbo_whitepaper.tex" not in text


def _reordered(value: object) -> object:
    if isinstance(value, dict):
        return {key: _reordered(item) for key, item in reversed(tuple(value.items()))}
    if isinstance(value, list):
        return [_reordered(item) for item in value]
    return value


def test_resolve_config_builds_the_frozen_h1_h2_record(tmp_path: Path) -> None:
    resolved = resolve_config(_raw_config(), repo_root=tmp_path)

    assert isinstance(resolved, ResolvedConfig)
    assert resolved.run.mode == "verify"
    assert resolved.run.seed == 20260721
    assert resolved.model.state_parent_sets == ((0,), (0, 1))
    assert resolved.validation.gates == ("H1", "H2")
    assert resolved.h3 is None
    assert resolved.h4 is None and resolved.h5 is None
    assert resolved.artifacts.run_root == (tmp_path / "runs").resolve()
    assert json.loads(resolved.canonical_json)["artifacts"]["run_root"] == (
        tmp_path / "runs"
    ).resolve().as_posix()
    assert len(resolved.config_sha256) == 64


def test_resolve_config_accepts_the_h1_compatibility_prefix(tmp_path: Path) -> None:
    raw = _raw_config()
    raw["validation"]["gates"] = ["H1"]  # type: ignore[index]

    resolved = resolve_config(raw, repo_root=tmp_path)

    assert resolved.validation.gates == ("H1",)
    assert resolved.h3 is None
    assert resolved.h4 is None and resolved.h5 is None


def test_resolve_config_builds_the_exact_frozen_h3_profile(tmp_path: Path) -> None:
    raw = _raw_h3_config()
    before = copy.deepcopy(raw)

    resolved = resolve_config(raw, repo_root=tmp_path)

    assert resolved.validation.gates == ("H1", "H2", "H3")
    assert isinstance(resolved.h3, H3ValidationConfig)
    assert resolved.h4 is None and resolved.h5 is None
    assert tuple(field.name for field in dataclasses.fields(resolved.h3)) == (
        "coupled_fixture_id",
        "coupled_expected_sha256",
        "zero_control_fixture_id",
        "zero_control_expected_sha256",
        "recognition_families",
        "common_initialization",
        "optimization_operation",
        "expected_autograd_scope",
        "optimizer",
        "decision",
        "solver_allowance_nats",
        "threshold_decision_rule",
        "minimum_resolved_fraction",
        "coupled_gap_inconclusive_obligation",
        "structured_closure_inconclusive_obligation",
    )
    assert resolved.h3.coupled_fixture_id == "h3-coupled-v1"
    assert resolved.h3.coupled_expected_sha256 == (
        "6779f5b0a2e27aa5e203764bcc4d84c1b1daedb9423fcefdf28dce3cf7e40e03"
    )
    assert resolved.h3.zero_control_fixture_id == "h3-zero-control-v1"
    assert resolved.h3.zero_control_expected_sha256 == (
        "ba600e09e0ae7e2b7576fbf4446a8e5b38a605c7621eb0cd5586689dccb89acf"
    )
    assert resolved.h3.recognition_families == (
        "structured_full_spd",
        "fine_factorized_diagonal",
    )
    assert resolved.h3.common_initialization == H3InitializationConfig()
    assert resolved.h3.optimization_operation == "maximize_direct_h3_elbo_lbfgs"
    assert resolved.h3.expected_autograd_scope == "h3_recognition_only"
    assert resolved.h3.optimizer == H3OptimizationConfig()
    assert resolved.h3.optimizer.tolerance_change == 1.0e-18
    assert resolved.h3.decision == H3DecisionConfig()
    assert resolved.h3.solver_allowance_nats == 1.0e-7
    assert resolved.h3.threshold_decision_rule == "signed_margin_three_way"
    assert resolved.h3.minimum_resolved_fraction == 0.99
    assert resolved.h3.coupled_gap_inconclusive_obligation == (
        "resolve coupled gap threshold outside allowance"
    )
    assert resolved.h3.structured_closure_inconclusive_obligation == (
        "resolve structured closure threshold outside allowance"
    )
    assert json.loads(resolved.canonical_json)["h3"] == _raw_h3()
    assert raw == before
    with pytest.raises(dataclasses.FrozenInstanceError):
        resolved.h3.minimum_resolved_fraction = 1.0  # type: ignore[misc]


@pytest.mark.parametrize("gates", [["H1"], ["H1", "H2"]])
def test_shorter_prefixes_reject_an_h3_section(
    tmp_path: Path, gates: list[str]
) -> None:
    raw = _raw_config()
    raw["validation"]["gates"] = gates  # type: ignore[index]
    raw["h3"] = _raw_h3()

    with pytest.raises(ValueError, match="h3"):
        resolve_config(raw, repo_root=tmp_path)


@pytest.mark.parametrize(
    "h3_value",
    [pytest.param(None, id="null"), pytest.param({}, id="missing")],
)
def test_h3_prefix_requires_the_exact_non_null_h3_section(
    tmp_path: Path, h3_value: object
) -> None:
    raw = _raw_config()
    raw["validation"]["gates"] = ["H1", "H2", "H3"]  # type: ignore[index]
    if h3_value is None:
        raw["h3"] = None

    with pytest.raises(ValueError, match="h3"):
        resolve_config(raw, repo_root=tmp_path)


@pytest.mark.parametrize(
    "gates",
    [[], ["H2"], ["H2", "H1"], ["H1", "H1"], ["H1", "H2", "H2"], ["H1", "H3"]],
)
def test_resolve_config_rejects_non_prefix_gate_lists(
    tmp_path: Path, gates: list[str]
) -> None:
    raw = _raw_config()
    raw["validation"]["gates"] = gates  # type: ignore[index]

    with pytest.raises(ValueError, match="validation.gates"):
        resolve_config(raw, repo_root=tmp_path)


def test_resolved_nested_records_are_frozen(tmp_path: Path) -> None:
    resolved = resolve_config(_raw_config(), repo_root=tmp_path)

    with pytest.raises(dataclasses.FrozenInstanceError):
        resolved.model.horizon = 3  # type: ignore[misc]


def test_hash_is_stable_when_mapping_keys_are_reordered(tmp_path: Path) -> None:
    raw = _raw_h3_config()
    reordered = _reordered(raw)

    original = resolve_config(raw, repo_root=tmp_path)
    reordered_result = resolve_config(reordered, repo_root=tmp_path)  # type: ignore[arg-type]

    assert reordered_result.canonical_json == original.canonical_json
    assert reordered_result.config_sha256 == original.config_sha256


def test_resolve_config_does_not_mutate_input(tmp_path: Path) -> None:
    raw = _raw_h3_config()
    before = copy.deepcopy(raw)

    resolve_config(raw, repo_root=tmp_path)

    assert raw == before


def test_h1_h5_configs_remain_free_of_h6_sections_and_byte_stable(tmp_path: Path) -> None:
    raw = _raw_h5_config()
    resolved = resolve_config(raw, repo_root=tmp_path)

    assert resolved.h6_prefix is None
    assert resolved.h6_prediction is None
    payload = json.loads(resolved.canonical_json)
    assert "h6_prefix" not in payload
    assert "h6_prediction" not in payload


@pytest.mark.parametrize(
    ("gates", "expected_length", "expected_sha256"),
    [
        (
            ["H1"],
            927,
            "07af2a1848128a85180190d18d3d76715ae031649ef6b1b10cee74b1eee0e818",
        ),
        (
            ["H1", "H2"],
            932,
            "ab4b89257fe36e82eab77e9e91aadd67446e5e69a379e59a73e968cc5f5007a7",
        ),
    ],
)
def test_compatibility_prefix_canonical_json_is_byte_identical(
    gates: list[str], expected_length: int, expected_sha256: str
) -> None:
    raw = _raw_config()
    raw["validation"]["gates"] = gates  # type: ignore[index]

    resolved = resolve_config(raw, repo_root=Path("C:/repo"))
    canonical_bytes = resolved.canonical_json.encode("utf-8")

    assert len(canonical_bytes) == expected_length
    assert hashlib.sha256(canonical_bytes).hexdigest() == expected_sha256
    assert resolved.config_sha256 == expected_sha256
    assert "h3" not in json.loads(resolved.canonical_json)
    assert '"h3":null' not in resolved.canonical_json


def test_relative_run_root_is_resolved_against_repo_root(tmp_path: Path) -> None:
    raw = _raw_config()
    raw["artifacts"] = {"run_root": "outputs/h1"}

    resolved = resolve_config(raw, repo_root=tmp_path)

    assert resolved.artifacts.run_root == (tmp_path / "outputs" / "h1").resolve()


@pytest.mark.parametrize("run_root", [".", ".."])
def test_run_root_cannot_equal_or_contain_repository(
    tmp_path: Path, run_root: str
) -> None:
    raw = _raw_config()
    raw["artifacts"] = {"run_root": run_root}
    with pytest.raises(ValueError, match="contain the repository"):
        resolve_config(raw, repo_root=tmp_path)


@pytest.mark.parametrize(
    "run_root",
    [
        ".verification",
        ".verification/runs",
        ".VeRiFiCaTiOn/RUNS",
        ".verification/../.verification/alias",
        ".verification.../x",
        ".verification..../x",
        ".VeRiFiCaTiOn... /x",
        ".git",
        ".git/objects",
        ".GIT/objects",
        "alias/../.git/worktrees",
        ".git.../x",
        ".git..../x",
        ".GiT... /objects",
    ],
)
def test_run_root_cannot_enter_repository_control_trees(
    tmp_path: Path, run_root: str
) -> None:
    raw = _raw_config()
    raw["artifacts"] = {"run_root": run_root}

    with pytest.raises(ValueError, match="control tree"):
        resolve_config(raw, repo_root=tmp_path)


@pytest.mark.parametrize("control", [".verification/x", ".git/objects"])
def test_extended_prefix_and_ordinary_control_paths_have_same_decision(
    tmp_path: Path, control: str
) -> None:
    ordinary = tmp_path / control
    extended = Path("\\\\?\\" + str(ordinary))

    assert is_repository_control_path(ordinary, tmp_path)
    assert is_repository_control_path(extended, tmp_path)

    raw = _raw_config()
    raw["artifacts"] = {"run_root": str(extended)}
    with pytest.raises(ValueError, match="control tree"):
        resolve_config(raw, repo_root=tmp_path)


def test_extended_prefix_run_root_cannot_equal_or_contain_repository(
    tmp_path: Path,
) -> None:
    for unsafe in (tmp_path, tmp_path.parent):
        raw = _raw_config()
        raw["artifacts"] = {"run_root": "\\\\?\\" + str(unsafe)}
        with pytest.raises(ValueError, match="contain the repository"):
            resolve_config(raw, repo_root=tmp_path)


def test_external_run_root_remains_valid(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    external = tmp_path / "external runs"
    raw = _raw_config()
    raw["artifacts"] = {"run_root": str(external)}

    resolved = resolve_config(raw, repo_root=repo)

    assert resolved.artifacts.run_root == external.resolve()


Mutation = Callable[[dict[str, object]], None]


def _add_unknown_top_level(raw: dict[str, object]) -> None:
    raw["extra"] = "not allowed"


def _add_unknown_run_key(raw: dict[str, object]) -> None:
    raw["run"]["workers"] = 1  # type: ignore[index]


def _remove_data_identity(raw: dict[str, object]) -> None:
    del raw["data"]["identity"]  # type: ignore[index]


def _bool_seed(raw: dict[str, object]) -> None:
    raw["run"]["seed"] = True  # type: ignore[index]


def _invalid_gates(raw: dict[str, object]) -> None:
    raw["validation"]["gates"] = ["H1", "H1"]  # type: ignore[index]


def _non_cpu_device(raw: dict[str, object]) -> None:
    raw["run"]["device"] = "cuda"  # type: ignore[index]


def _non_float64_dtype(raw: dict[str, object]) -> None:
    raw["run"]["dtype"] = "float32"  # type: ignore[index]


def _changed_model_structure(raw: dict[str, object]) -> None:
    raw["model"]["horizon"] = 3  # type: ignore[index]


def _changed_recognition_literal(raw: dict[str, object]) -> None:
    raw["recognition"]["conditioning"] = "filtering"  # type: ignore[index]


def _changed_inference_literal(raw: dict[str, object]) -> None:
    raw["inference"]["operation"] = "train"  # type: ignore[index]


def _changed_optimization_literal(raw: dict[str, object]) -> None:
    raw["optimization"]["e_like_update"] = "gradient"  # type: ignore[index]


def _changed_convergence_limit(raw: dict[str, object]) -> None:
    raw["validation"]["maximum_convergence_estimate"] = 1e-8  # type: ignore[index]


@pytest.mark.parametrize(
    "mutate",
    [
        _add_unknown_top_level,
        _add_unknown_run_key,
        _remove_data_identity,
        _bool_seed,
        _invalid_gates,
        _non_cpu_device,
        _non_float64_dtype,
        _changed_model_structure,
        _changed_recognition_literal,
        _changed_inference_literal,
        _changed_optimization_literal,
        _changed_convergence_limit,
    ],
)
def test_resolve_config_rejects_unknown_or_invalid_values(
    tmp_path: Path, mutate: Mutation
) -> None:
    raw = _raw_config()
    mutate(raw)

    with pytest.raises(ValueError):
        resolve_config(raw, repo_root=tmp_path)


def _reorder_h3_families(raw: dict[str, object]) -> None:
    raw["h3"]["recognition_families"] = [  # type: ignore[index]
        "fine_factorized_diagonal",
        "structured_full_spd",
    ]


def _use_old_h3_tolerance_change(raw: dict[str, object]) -> None:
    raw["h3"]["optimizer"]["tolerance_change"] = 1.0e-15  # type: ignore[index]


def _add_unknown_h3_key(raw: dict[str, object]) -> None:
    raw["h3"]["extra"] = "not allowed"  # type: ignore[index]


def _add_unknown_h3_optimizer_key(raw: dict[str, object]) -> None:
    raw["h3"]["optimizer"]["momentum"] = 0.0  # type: ignore[index]


def _remove_h3_decision_key(raw: dict[str, object]) -> None:
    del raw["h3"]["decision"]["dimension"]  # type: ignore[index]


def _integer_h3_optimizer_float(raw: dict[str, object]) -> None:
    raw["h3"]["optimizer"]["learning_rate"] = 1  # type: ignore[index]


def _boolean_h3_solver_allowance(raw: dict[str, object]) -> None:
    raw["h3"]["solver_allowance_nats"] = True  # type: ignore[index]


def _integer_h3_decision_float(raw: dict[str, object]) -> None:
    raw["h3"]["decision"]["maximum_mean_infinity_norm"] = 4  # type: ignore[index]


def _integer_h3_initialization_float(raw: dict[str, object]) -> None:
    raw["h3"]["common_initialization"]["mean"][0] = 0  # type: ignore[index]


@pytest.mark.parametrize(
    "mutate",
    [
        _reorder_h3_families,
        _use_old_h3_tolerance_change,
        _add_unknown_h3_key,
        _add_unknown_h3_optimizer_key,
        _remove_h3_decision_key,
        _integer_h3_optimizer_float,
        _boolean_h3_solver_allowance,
        _integer_h3_decision_float,
        _integer_h3_initialization_float,
    ],
)
def test_resolve_config_rejects_changed_or_malformed_h3_values(
    tmp_path: Path, mutate: Mutation
) -> None:
    raw = _raw_h3_config()
    mutate(raw)

    with pytest.raises(ValueError):
        resolve_config(raw, repo_root=tmp_path)
