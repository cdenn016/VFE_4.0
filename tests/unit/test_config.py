from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path
from typing import Any, Callable

import pytest

from vfe4.config import ResolvedConfig, resolve_config
from vfe4.config.control_paths import is_repository_control_path


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
    raw = _raw_config()
    reordered = _reordered(raw)

    original = resolve_config(raw, repo_root=tmp_path)
    reordered_result = resolve_config(reordered, repo_root=tmp_path)  # type: ignore[arg-type]

    assert reordered_result.canonical_json == original.canonical_json
    assert reordered_result.config_sha256 == original.config_sha256


def test_resolve_config_does_not_mutate_input(tmp_path: Path) -> None:
    raw = _raw_config()
    before = copy.deepcopy(raw)

    resolve_config(raw, repo_root=tmp_path)

    assert raw == before


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
