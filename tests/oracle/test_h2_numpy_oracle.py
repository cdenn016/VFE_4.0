from __future__ import annotations

import ast
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from verification.numpy_oracles.h2_moment import evaluate_h2_moment_oracle
from vfe4.generative import H1GenerativeModel
from vfe4.objective import evaluate_local_elbo, evaluate_monolithic_elbo
from vfe4.recognition import H1RecognitionLaw
from vfe4.types import SourcePath
from vfe4.validation import load_h1_fixture


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "vfe4" / "validation" / "fixtures" / "h1_v1.json"
PATHS = tuple(
    SourcePath((0, state_source), (0, model_source))
    for model_source in range(2)
    for state_source in range(2)
)


def test_h2_oracle_is_independent_numpy_and_covers_every_dense_component() -> None:
    source_path = REPO_ROOT / "verification" / "numpy_oracles" / "h2_moment.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    assert imported.isdisjoint({"torch", "vfe4", "verification"})
    assert "h1_elbo" not in source_path.read_text(encoding="utf-8")

    oracle = evaluate_h2_moment_oracle(FIXTURE)
    fixture = load_h1_fixture(FIXTURE)
    model = H1GenerativeModel.from_fixture(fixture)
    recognition = H1RecognitionLaw.from_fixture(fixture)

    assert tuple(component.path for component in oracle.components) == (
        ((0, 0), (0, 0)),
        ((0, 1), (0, 0)),
        ((0, 0), (0, 1)),
        ((0, 1), (0, 1)),
    )
    assert tuple(component.weight for component in oracle.components) == (
        0.3,
        0.1,
        0.12,
        0.48,
    )

    for path, component in zip(PATHS, oracle.components):
        q = recognition.joint_component(path)
        p = model.joint_component(path)
        for actual, expected in (
            (component.q.mean, q.mean.numpy()),
            (component.q.covariance, q.covariance.numpy()),
            (component.p.mean, p.mean.numpy()),
            (component.p.covariance, p.covariance.numpy()),
        ):
            assert actual.dtype == np.float64
            assert actual.shape in ((6,), (6, 6))
            assert float(np.max(np.abs(actual - expected))) <= 2.0e-15
        for law in (component.q, component.p):
            assert float(np.max(np.abs(law.precision @ law.mean - law.h))) <= 3.0e-15
            assert float(np.max(np.abs(law.precision @ law.covariance - np.eye(6)))) <= 3.0e-15
            assert law.minimum_cholesky_pivot > 0.0
            assert law.lambda_min > 0.0
            assert law.lambda_max >= law.lambda_min
            assert law.kappa_2 >= 1.0
            assert set(law.absolute_summand_accumulation) == {
                "mean",
                "covariance",
                "precision",
                "h",
                "log_normalizer",
                "entropy",
            }
        assert tuple(marginal.indices for marginal in component.emission_marginals) == (
            (2, 3),
            (4, 5),
        )
        for marginal in component.emission_marginals:
            index = list(marginal.indices)
            assert float(np.max(np.abs(marginal.mean - component.q.mean[index]))) == 0.0
            assert float(
                np.max(
                    np.abs(
                        marginal.covariance
                        - component.q.covariance[np.ix_(index, index)]
                    )
                )
            ) == 0.0
        assert component.gaussian_kl >= 0.0
        assert component.gaussian_log_ratio == -component.gaussian_kl
        assert component.complete_value == math.fsum(
            (
                component.gaussian_log_ratio,
                component.source_log_ratio,
                *component.expected_log_emission,
            )
        )
        assert math.isfinite(component.q.log_normalizer)
        assert math.isfinite(component.p.log_normalizer)
        assert math.isfinite(component.q.entropy)
        assert set(component.absolute_summand_accumulation) == {
            "gaussian_kl",
            "gaussian_log_ratio",
            "source_log_ratio",
            "expected_log_emission[0]",
            "expected_log_emission[1]",
            "complete_value",
        }


def test_h2_oracle_reconstructs_h1_component_local_and_complete_elbo() -> None:
    oracle = evaluate_h2_moment_oracle(FIXTURE, quadrature_order=21)
    fixture = load_h1_fixture(FIXTURE)
    model = H1GenerativeModel.from_fixture(fixture)
    recognition = H1RecognitionLaw.from_fixture(fixture)
    monolithic = evaluate_monolithic_elbo(
        model,
        recognition,
        quadrature_order=21,
        convergence_check_order=17,
    )
    local = evaluate_local_elbo(
        model,
        recognition,
        quadrature_order=21,
        convergence_check_order=17,
    )

    assert max(
        abs(left.complete_value - right)
        for left, right in zip(oracle.components, monolithic.component_values)
    ) <= 2.0e-13
    assert max(
        abs(left.gaussian_log_ratio - right)
        for left, right in zip(
            oracle.components, monolithic.component_gaussian_log_ratios
        )
    ) <= 5.0e-14
    assert max(
        abs(left.source_log_ratio - right)
        for left, right in zip(oracle.components, monolithic.component_source_log_ratios)
    ) <= 2.0e-15
    assert max(
        abs(left.expected_log_emission[time] - right[time])
        for left, right in zip(oracle.components, monolithic.component_emission_values)
        for time in range(2)
    ) <= 2.0e-13

    oracle_local = oracle.local_terms
    for left, right in (
        (oracle_local.expected_log_emission, local.expected_log_emission),
        (oracle_local.model_source_kl, local.model_source_kl),
        (oracle_local.model_transition_kl, local.model_transition_kl),
        (oracle_local.state_source_kl, local.state_source_kl),
        (oracle_local.state_transition_kl, local.state_transition_kl),
    ):
        assert max(abs(a - b) for a, b in zip(left, right)) <= 2.0e-13
    assert abs(oracle_local.initial_model_kl - local.initial_model_kl) <= 2.0e-14
    assert abs(oracle_local.initial_state_kl - local.initial_state_kl) <= 2.0e-14
    assert (
        abs(oracle_local.joint_recognition_entropy - local.joint_recognition_entropy)
        <= 2.0e-13
    )
    assert len(oracle.signed_local_terms) == 12
    assert oracle.local_terms.complete_elbo == math.fsum(oracle.signed_local_terms)
    assert oracle.complete_elbo == math.fsum(
        component.weight * component.complete_value for component in oracle.components
    )
    assert abs(oracle.complete_elbo - monolithic.value) <= 2.0e-13
    assert abs(oracle.local_terms.complete_elbo - local.complete_elbo) <= 2.0e-13
    assert abs(oracle.complete_elbo - oracle.local_terms.complete_elbo) <= 2.0e-13
    assert oracle.joint_recognition_entropy == math.fsum(
        (oracle.source_entropy, oracle.weighted_component_entropy)
    )
    expected_metadata_names = {
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
        "complete_elbo",
    }
    assert set(oracle.local_terms.spd_operand_kappas) == expected_metadata_names
    assert len(oracle.local_terms.spd_operand_kappas["expected_log_emission[0]"]) == 4
    assert len(oracle.local_terms.spd_operand_kappas["expected_log_emission[1]"]) == 4
    assert len(oracle.local_terms.spd_operand_kappas["joint_recognition_entropy"]) == 4
    assert all(
        math.isfinite(kappa) and kappa >= 1.0
        for kappas in oracle.local_terms.spd_operand_kappas.values()
        for kappa in kappas
    )


def test_h2_oracle_rejects_nonfrozen_order_and_non_json_bytes(tmp_path: Path) -> None:
    with np.testing.assert_raises_regex(ValueError, "quadrature_order"):
        evaluate_h2_moment_oracle(FIXTURE, quadrature_order=17)
    bad = tmp_path / "bad.json"
    bad.write_bytes(b"not-json")
    with np.testing.assert_raises(ValueError):
        evaluate_h2_moment_oracle(bad)


def test_rectangular_information_assembly_matches_independent_dense_oracle() -> None:
    import importlib

    import torch

    fixture_module = importlib.import_module(
        "vfe4.validation.h2_h5_rectangular_fixture"
    )
    linear_gaussian = importlib.import_module("vfe4.numerics.linear_gaussian")
    rectangular_oracle = importlib.import_module(
        "verification.numpy_oracles.h2_h5_rectangular"
    )

    source_path = (
        REPO_ROOT / "verification" / "numpy_oracles" / "h2_h5_rectangular.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    production_modules = {
        "vfe4.numerics.linear_gaussian",
        "vfe4.inference.h5_updates",
    }
    assert all(
        not (
            isinstance(node, ast.ImportFrom)
            and node.module in production_modules
        )
        for node in ast.walk(tree)
    )

    fixture = fixture_module.load_h2_h5_rectangular_fixture()
    assert (
        fixture.seed,
        fixture.horizon,
        fixture.d_z,
        fixture.d_m,
        fixture.observation_dimension,
    ) == (31337, 3, 2, 3, 2)
    assert fixture.dense_parents == ((0,), (0, 1), (0, 1, 2))
    assert fixture.raw_sha256 == fixture_module.H2_H5_RECTANGULAR_RAW_SHA256
    assert (
        fixture.canonical_sha256
        == fixture_module.H2_H5_RECTANGULAR_CANONICAL_SHA256
    )
    assert (
        fixture_module.h2_h5_rectangular_raw_bytes(fixture)
        == fixture_module.h2_h5_rectangular_raw_bytes(
            fixture_module.load_h2_h5_rectangular_fixture()
        )
    )
    assert (
        fixture_module.h2_h5_rectangular_canonical_bytes(fixture)
        == fixture_module.h2_h5_rectangular_canonical_bytes(
            fixture_module.load_h2_h5_rectangular_fixture()
        )
    )

    oracle = rectangular_oracle.evaluate_rectangular_update_oracle(
        fixture, time_index=2
    )
    assert oracle.fixture_raw_sha256 == (
        "6925ffe08e4d8acbc7790b6318f3e26a0509a8208ebf062f62f721332d194aa5"
    )
    assert oracle.fixture_canonical_sha256 == (
        "02add1038f70cedd2cb5b0adad0c3b23696960f9fe2a0c4942df7eea77e3f58c"
    )
    for channel in ("state", "model"):
        probe_objective = getattr(oracle, f"{channel}_probe_objective")
        solved_objective = getattr(oracle, f"{channel}_solved_objective")
        completion_gap = getattr(
            oracle, f"{channel}_completion_square_gap"
        )
        assert probe_objective > solved_objective
        assert completion_gap > 0.0
        assert math.isclose(
            probe_objective - solved_objective,
            completion_gap,
            rel_tol=2.0e-13,
            abs_tol=2.0e-13,
        )
        assert (
            getattr(oracle, f"{channel}_solution_gradient_max_abs")
            <= 5.0e-13
        )
    forged_offsets = tuple(
        (
            (row[0] + 0.125,) + row[1:]
            if index == 1
            else row
        )
        for index, row in enumerate(fixture.state_offsets)
    )
    forged = replace(fixture, state_offsets=forged_offsets)
    with np.testing.assert_raises_regex(ValueError, "frozen rectangular C5"):
        rectangular_oracle.evaluate_rectangular_update_oracle(
            forged, time_index=2
        )
    state_precision = torch.tensor(
        fixture.state_precisions[1], dtype=torch.float64
    )
    state_model_map = torch.tensor(
        fixture.state_model_maps[1], dtype=torch.float64
    )
    model_recoil_residual = torch.tensor(
        oracle.model_recoil_residual, dtype=torch.float64
    )
    production = linear_gaussian.assemble_rectangular_information(
        state_precision=state_precision,
        state_model_map=state_model_map,
        model_recoil_residual=model_recoil_residual,
    )

    assert tuple(production.model_precision_pullback.shape) == (3, 3)
    assert tuple(production.model_recoil_natural.shape) == (3,)
    np.testing.assert_allclose(
        production.model_precision_pullback.numpy(),
        np.asarray(oracle.model_precision_pullback),
        rtol=0.0,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        production.model_recoil_natural.numpy(),
        np.asarray(oracle.model_recoil_natural),
        rtol=0.0,
        atol=2.0e-14,
    )
    with np.testing.assert_raises_regex(ValueError, "transposed"):
        linear_gaussian.assemble_rectangular_information(
            state_precision=state_precision,
            state_model_map=state_model_map.T,
            model_recoil_residual=model_recoil_residual,
        )
