from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import pytest

from vfe4.config import H3ValidationConfig, ResolvedConfig, resolve_config
from vfe4.config.control_paths import is_repository_control_path
from vfe4.types.h3 import (
    H3DecisionConfig,
    H3InitializationConfig,
    H3OptimizationConfig,
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


def test_resolve_config_builds_the_exact_frozen_h3_profile(tmp_path: Path) -> None:
    raw = _raw_h3_config()
    before = copy.deepcopy(raw)

    resolved = resolve_config(raw, repo_root=tmp_path)

    assert resolved.validation.gates == ("H1", "H2", "H3")
    assert isinstance(resolved.h3, H3ValidationConfig)
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
