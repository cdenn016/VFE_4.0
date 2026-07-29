from __future__ import annotations

import ast
import math
from dataclasses import fields
from pathlib import Path

import pytest

from verification.numpy_oracles.h6_depth import evaluate_depth2_oracle
from vfe4.generative.h6_depth import (
    depth2_complete_log_joint,
    enumerate_depth2_source_paths,
    evaluate_depth2_normalization,
)
from vfe4.objective.h6_depth import (
    evaluate_depth2_local_objective,
    evaluate_depth2_monolithic_objective,
)
from vfe4.types.h6_depth import (
    Depth2CascadeSpec,
    build_tiny_depth2_probe,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_depth2_cascade_is_normalized_and_complete_objective_matches() -> None:
    oracle_source = (
        REPO_ROOT / "verification" / "numpy_oracles" / "h6_depth.py"
    )
    oracle_tree = ast.parse(oracle_source.read_text(encoding="utf-8"))
    forbidden_roots = ("vfe4", "torch", "numpy")
    for node in ast.walk(oracle_tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module is None or not node.module.startswith(
                forbidden_roots
            )
        if isinstance(node, ast.Import):
            assert all(
                not alias.name.startswith(forbidden_roots)
                for alias in node.names
            )

    probe = build_tiny_depth2_probe()
    cascade = probe.cascade
    assert "model_depth" not in {field.name for field in fields(Depth2CascadeSpec)}
    assert cascade.scientific_disposition == (
        "nonblocking_composition_risk_probe"
    )
    assert cascade.layer1_parameter_owner != cascade.layer2_parameter_owner
    for banks in (
        cascade.source_banks,
        probe.recognition.source_posteriors,
    ):
        for layer in (1, 2):
            for channel in ("state", "model"):
                bank = banks.bank(layer, channel)
                assert bank.conditioning == "causal_latent_prefix_only"
                for receiver_t, row in enumerate(bank.rows, start=1):
                    assert row.parents == tuple(range(receiver_t))
                    assert math.fsum(row.probabilities) == pytest.approx(
                        1.0, abs=2.0e-15
                    )

    production_normalization = evaluate_depth2_normalization(cascade)
    independent = evaluate_depth2_oracle(probe)
    assert production_normalization.source_path_count == 16
    assert production_normalization.token_sequence_count == 9
    assert production_normalization.emission_region_count == 4
    assert production_normalization.gaussian_factor_count == 12
    assert independent.source_path_count == 16
    assert independent.token_sequence_count == 9
    assert independent.emission_region_count == 4
    assert independent.gaussian_factor_count == 12
    assert production_normalization.source_mass == pytest.approx(
        1.0, abs=2.0e-15
    )
    assert independent.source_mass == pytest.approx(1.0, abs=2.0e-15)
    assert production_normalization.total_mass == pytest.approx(
        1.0, abs=2.0e-15
    )
    assert production_normalization.maximum_emission_mass_error <= 2.0e-15
    assert independent.maximum_emission_mass_error <= 2.0e-15

    paths = enumerate_depth2_source_paths(cascade)
    assert len(paths) == 16
    q = probe.recognition
    point_log_joint = depth2_complete_log_joint(
        cascade,
        paths[0],
        state_values=tuple(
            tuple(
                q.marginal(layer, "state", receiver_t).mean
                for receiver_t in range(3)
            )
            for layer in (1, 2)
        ),
        model_values=tuple(
            tuple(
                q.marginal(layer, "model", receiver_t).mean
                for receiver_t in range(3)
            )
            for layer in (1, 2)
        ),
        tokens=probe.observed_tokens,
    )
    assert math.isfinite(point_log_joint)

    local = evaluate_depth2_local_objective(probe)
    monolithic = evaluate_depth2_monolithic_objective(probe)
    assert len(local.terms) == 42
    assert tuple(term.name for term in local.terms) == (
        independent.objective_term_names
    )
    for production, oracle in zip(
        (term.value for term in local.terms),
        independent.objective_term_values,
        strict=True,
    ):
        assert production == pytest.approx(oracle, abs=2.0e-14)
    assert local.total == pytest.approx(monolithic, abs=2.0e-14)
    assert local.total == pytest.approx(
        independent.objective_total, abs=2.0e-14
    )
    partition_counts = {
        partition: sum(term.partition == partition for term in local.terms)
        for partition in (
            "initial",
            "source",
            "transition",
            "emission",
            "recognition_entropy",
        )
    }
    assert partition_counts == {
        "initial": 4,
        "source": 8,
        "transition": 8,
        "emission": 2,
        "recognition_entropy": 20,
    }

    protected_surfaces = (
        REPO_ROOT / "vfe4" / "config",
        REPO_ROOT / "vfe4" / "training",
    )
    for surface in protected_surfaces:
        for source in surface.glob("*.py"):
            assert "h6_depth" not in source.read_text(encoding="utf-8")
    assert "h6_depth" not in (
        REPO_ROOT / "run_h6_prediction_v3.py"
    ).read_text(encoding="utf-8")
