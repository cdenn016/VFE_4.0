from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path

import pytest
import torch

from verification.h1_gate import (
    H1_INVARIANT_NAMES,
    H1_MEASUREMENT_NAMES,
    PAIRWISE_NAMES,
    TERM_NAMES,
    run_h1,
)
from vfe4.config import ResolvedConfig, resolve_config
from vfe4.objective import evaluate_monolithic_elbo
from vfe4.recognition import H1RecognitionLaw
from vfe4.generative import H1GenerativeModel
from vfe4.types import GateResult, GateStatus, InvariantResult
from vfe4.validation import enumerate_source_paths, load_h1_fixture


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = REPO_ROOT / "vfe4" / "validation" / "fixtures" / "h1_v1.json"


def _raw_config(run_root: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "objective_schema_version": "vfe4-state-elbo-v1",
        "run": {"mode": "verify", "seed": 20260721, "device": "cpu", "dtype": "float64", "deterministic": True},
        "data": {"kind": "frozen_fixture", "identity": "h1-v1"},
        "model": {
            "horizon": 2, "d_z": 1, "d_m": 1, "vocabulary_size": 3,
            "state_parent_sets": [[0], [0, 1]], "model_parent_sets": [[0], [0, 1]],
            "state_source_support": [[0], [0, 1]], "model_source_support": [[0], [0, 1]],
            "geometry": "fixed_population_frames",
        },
        "recognition": {"conditioning": "smoothing", "family": "structured_linear_gaussian_mixture", "source_treatment": "exact_enumeration"},
        "inference": {"operation": "evaluate_only", "estimator": "deterministic_quadrature"},
        "optimization": {"e_like_update": "none", "m_like_update": "none", "expected_autograd_scope": "none"},
        "validation": {"gates": ["H1"], "fixture_id": "h1-v1", "quadrature_order": 21, "convergence_check_order": 17, "maximum_convergence_estimate": 1e-9},
        "artifacts": {"run_root": str(run_root)},
    }


def _config(tmp_path: Path) -> ResolvedConfig:
    return resolve_config(_raw_config(tmp_path / "runs"), repo_root=REPO_ROOT)


def _payload(run_dir: Path) -> dict[str, object]:
    return json.loads((run_dir / "validation" / "h1.json").read_text(encoding="utf-8"))


def test_monolithic_result_exposes_validated_per_path_emissions_and_reconstructs() -> None:
    fixture = load_h1_fixture(FIXTURE_PATH)
    model = H1GenerativeModel.from_fixture(fixture)
    recognition = H1RecognitionLaw.from_fixture(fixture)

    result = evaluate_monolithic_elbo(model, recognition, quadrature_order=21, convergence_check_order=17)

    assert len(result.component_emission_values) == 4
    assert all(len(values) == 2 for values in result.component_emission_values)
    for value, gaussian, source, emissions in zip(
        result.component_values,
        result.component_gaussian_log_ratios,
        result.component_source_log_ratios,
        result.component_emission_values,
    ):
        assert value == pytest.approx(math.fsum((gaussian, source, *emissions)), abs=2e-14)
    weights = [float(recognition.source_probability(path)) for path in enumerate_source_paths(fixture)]
    for time in range(2):
        reconstructed = math.fsum(weight * values[time] for weight, values in zip(weights, result.component_emission_values))
        assert result.expected_log_emission[time] == pytest.approx(reconstructed, abs=2e-14)


def test_gate_result_enforces_generic_status_consistency() -> None:
    good = InvariantResult("i", True, 0.0, 0.0, "ok")
    bad = InvariantResult("i", False, 1.0, 0.0, "failed")
    base = dict(gate="H1", fixture_id="h1-v1", residual=0.0, calibrated_allowance=1.0, measurements={"m": 0.0})

    GateResult(status=GateStatus.PASS, invariants=(good,), obligations=(), **base)
    GateResult(status=GateStatus.FAIL, invariants=(bad,), obligations=(), **base)
    GateResult(status=GateStatus.INCONCLUSIVE, invariants=(InvariantResult("i", False, None, None, "unavailable"),), obligations=("compute",), **{**base, "residual": None, "calibrated_allowance": None, "measurements": {"m": None}})
    GateResult(status=GateStatus.INCONCLUSIVE, invariants=(), obligations=("compute",), **{**base, "residual": None, "calibrated_allowance": None, "measurements": {}})
    for kwargs in (
        dict(status=GateStatus.PASS, invariants=(), obligations=()),
        dict(status=GateStatus.PASS, invariants=(bad,), obligations=()),
        dict(status=GateStatus.PASS, invariants=(good,), obligations=("x",)),
        dict(status=GateStatus.FAIL, invariants=(good,), obligations=()),
        dict(status=GateStatus.FAIL, invariants=(InvariantResult("i", False, None, 0.0, "x"),), obligations=()),
        dict(status=GateStatus.INCONCLUSIVE, invariants=(bad,), obligations=()),
    ):
        with pytest.raises(ValueError):
            GateResult(**kwargs, **base)
    with pytest.raises(ValueError, match="nonempty"):
        GateResult(status=GateStatus.PASS, invariants=(good,), obligations=(), **{**base, "measurements": {}})
    with pytest.raises(ValueError, match="unique"):
        GateResult(status=GateStatus.PASS, invariants=(good, good), obligations=(), **base)


def test_h1_gate_passes_with_exact_inventories_and_complete_artifact(tmp_path: Path) -> None:
    result, run_dir = run_h1(_config(tmp_path))

    assert result.status is GateStatus.PASS
    assert tuple(result.measurements) == H1_MEASUREMENT_NAMES
    assert tuple(item.name for item in result.invariants) == H1_INVARIANT_NAMES
    assert set(PAIRWISE_NAMES) <= set(H1_INVARIANT_NAMES)
    assert set(TERM_NAMES) <= set(H1_INVARIANT_NAMES)
    assert result.obligations == ()
    payload = _payload(run_dir)
    assert set(payload) >= {
        "gate_result", "monolithic", "local_terms", "independent_terms", "identity",
        "evidences", "convergence_registry", "pairwise_residuals", "pairwise_allowances",
        "term_comparisons", "negative_controls",
    }
    assert len(payload["monolithic"]["component_emission_values"]) == 4
    assert len(payload["evidences"]) == 9


def test_h1_uses_no_grad_cpu_float64_for_production_evaluators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import verification.h1_gate as gate

    seen: list[tuple[str, bool, str, torch.dtype]] = []
    original_monolithic = gate.evaluate_monolithic_elbo
    original_local = gate.evaluate_local_elbo

    def monolithic(model: object, recognition: object, **kwargs: object) -> object:
        seen.append(("monolithic", torch.is_grad_enabled(), model.factors.initial_joint.mean.device.type, model.factors.initial_joint.mean.dtype))
        return original_monolithic(model, recognition, **kwargs)

    def local(model: object, recognition: object, **kwargs: object) -> object:
        seen.append(("local", torch.is_grad_enabled(), model.factors.initial_joint.mean.device.type, model.factors.initial_joint.mean.dtype))
        return original_local(model, recognition, **kwargs)

    monkeypatch.setattr(gate, "evaluate_monolithic_elbo", monolithic)
    monkeypatch.setattr(gate, "evaluate_local_elbo", local)

    result, _ = run_h1(_config(tmp_path))

    assert result.status is GateStatus.PASS
    assert seen == [
        ("monolithic", False, "cpu", torch.float64),
        ("local", False, "cpu", torch.float64),
    ]


def test_mutated_resolved_config_is_inconclusive_and_skips_evaluators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import verification.h1_gate as gate

    config = _config(tmp_path)
    object.__setattr__(config, "config_sha256", "0" * 64)
    monkeypatch.setattr(gate, "evaluate_monolithic_elbo", lambda *a, **k: pytest.fail("called"))
    monkeypatch.setattr(gate, "evaluate_local_elbo", lambda *a, **k: pytest.fail("called"))

    result, run_dir = run_h1(config)

    assert result.status is GateStatus.INCONCLUSIVE
    assert result.obligations
    assert all(value is None for value in result.measurements.values())
    assert run_dir.exists()


def test_mutated_canonical_json_still_publishes_inconclusive_without_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import verification.h1_gate as gate

    config = _config(tmp_path)
    object.__setattr__(config, "canonical_json", "{not-json")
    monkeypatch.setattr(gate, "evaluate_monolithic_elbo", lambda *a, **k: pytest.fail("called"))

    result, run_dir = run_h1(config)

    assert result.status is GateStatus.INCONCLUSIVE
    assert run_dir.exists()


def test_computation_exception_publishes_manifest_valid_inconclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import verification.h1_gate as gate

    monkeypatch.setattr(gate, "load_h1_fixture", lambda path: (_ for _ in ()).throw(ValueError("malformed readable fixture")))

    result, run_dir = run_h1(_config(tmp_path))

    assert result.status is GateStatus.INCONCLUSIVE
    assert result.obligations and "malformed readable fixture" in result.obligations[0]
    assert all(value is None for value in result.measurements.values())
    manifest = (run_dir / "manifest.sha256").read_text(encoding="utf-8")
    for line in manifest.splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256((run_dir / name).read_bytes()).hexdigest() == digest


def test_unexpected_evaluator_exception_still_publishes_inconclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import verification.h1_gate as gate

    monkeypatch.setattr(gate, "evaluate_monolithic_elbo", lambda *a, **k: (_ for _ in ()).throw(KeyError("malformed evaluator")))
    result, run_dir = run_h1(_config(tmp_path))
    assert result.status is GateStatus.INCONCLUSIVE
    assert result.obligations and "malformed evaluator" in result.obligations[0]
    assert run_dir.exists()


def test_pair_roundoff_is_local_not_global_and_boundary_is_inclusive() -> None:
    from verification.h1_gate import pair_comparison

    tiny = pair_comparison(1.0, 1.0 + 1e-12, 0.0, 0.0)
    huge = pair_comparison(1e12, 1e12, 0.0, 0.0)
    assert tiny.rounding < huge.rounding
    assert not pair_comparison(1.0, 1.0 + 1e-12, 0.0, 0.0).passed
    boundary = pair_comparison(1.0, 1.0, 0.0, 0.0)
    assert boundary.residual <= boundary.allowance and boundary.passed


def test_exact_term_names_are_fourteen_and_local_allowances_are_individual(tmp_path: Path) -> None:
    result, run_dir = run_h1(_config(tmp_path))
    assert result.status is GateStatus.PASS
    payload = _payload(run_dir)
    comparisons = payload["term_comparisons"]
    assert list(comparisons) == sorted(TERM_NAMES)
    assert len(comparisons) == 14
    for name, comparison in comparisons.items():
        expected_rounding = 64.0 * math.ulp(1.0) * max(1.0, abs(comparison["left"]), abs(comparison["right"]))
        assert comparison["rounding"].hex() == expected_rounding.hex()
        assert comparison["allowance"].hex() == math.fsum((comparison["left_allowance"], comparison["right_allowance"], comparison["rounding"])).hex()


def test_evidence_labels_are_exact_lexicographic_unique_and_probability_sum_is_gated(tmp_path: Path) -> None:
    result, run_dir = run_h1(_config(tmp_path))
    assert result.status is GateStatus.PASS
    payload = _payload(run_dir)
    labels = [tuple(item["observation_labels"]) for item in payload["evidences"]]
    assert labels == [(i, j) for i in (1, 2, 3) for j in (1, 2, 3)]
    assert len(set(labels)) == 9
    probabilities = [item["probability"] for item in payload["evidences"]]
    assert all(0.0 < value <= 1.0 for value in probabilities)
    normalization = payload["evidence_normalization"]
    assert normalization["residual"] == abs(math.fsum(probabilities) - 1.0)
    assert normalization["allowance"] == math.fsum(item["probability_allowance"]["total"] for item in payload["evidences"]) + normalization["rounding"]


def test_convergence_registry_is_flat_and_every_estimate_is_gated(tmp_path: Path) -> None:
    result, run_dir = run_h1(_config(tmp_path))
    assert result.status is GateStatus.PASS
    registry = _payload(run_dir)["convergence_registry"]
    assert len(registry) == 49
    assert all("." in name or name == "monolithic" for name in registry)
    assert all(0.0 <= value <= 1e-9 for value in registry.values())


def test_negative_controls_are_exactly_three_and_each_is_detected(tmp_path: Path) -> None:
    result, run_dir = run_h1(_config(tmp_path))
    assert result.status is GateStatus.PASS
    controls = _payload(run_dir)["negative_controls"]
    assert list(controls) == sorted([
        "recognition_source_entropy_omission",
        "selected_raw_logit_substitution",
        "recognition_mixture_for_generative_evidence",
    ])
    assert controls["recognition_source_entropy_omission"]["domain"] == "log"
    assert controls["selected_raw_logit_substitution"]["domain"] == "log"
    assert controls["recognition_mixture_for_generative_evidence"]["domain"] == "probability"
    assert all(item["residual"] > item["allowance"] and item["passed"] for item in controls.values())


def test_no_production_module_imports_verification() -> None:
    offenders = []
    for path in (REPO_ROOT / "vfe4").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "import verification" in text or "from verification" in text:
            offenders.append(path)
    assert offenders == []


@pytest.mark.parametrize(
    "target",
    ["monolithic", "independent_term", "posterior_kl", "evidence_sum", "convergence"],
)
def test_finite_injected_disagreements_are_fail_not_inconclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    import verification.h1_gate as gate

    if target == "monolithic":
        original = gate.evaluate_monolithic_elbo
        monkeypatch.setattr(gate, "evaluate_monolithic_elbo", lambda *a, **k: dataclasses.replace(original(*a, **k), value=original(*a, **k).value + 1e-3))
    elif target == "independent_term":
        original = gate.h1_local_diagnostics
        monkeypatch.setattr(gate, "h1_local_diagnostics", lambda *a, **k: dataclasses.replace(original(*a, **k), initial_model_kl=original(*a, **k).initial_model_kl + 1e-3))
    elif target == "posterior_kl":
        original = gate.h1_evidence_and_posterior_kl
        monkeypatch.setattr(gate, "h1_evidence_and_posterior_kl", lambda *a, **k: dataclasses.replace(original(*a, **k), posterior_kl=-1e-3))
    elif target == "evidence_sum":
        original = gate.h1_all_observation_evidences
        def changed(*args: object, **kwargs: object) -> tuple[object, ...]:
            records = original(*args, **kwargs)
            probability = records[0].probability + 1e-3
            return (dataclasses.replace(records[0], probability=probability, log_probability=math.log(probability)), *records[1:])
        monkeypatch.setattr(gate, "h1_all_observation_evidences", changed)
    else:
        original = gate.evaluate_monolithic_elbo
        def changed(*args: object, **kwargs: object) -> object:
            record = original(*args, **kwargs)
            allowance = dataclasses.replace(record.numerical_allowance, convergence_estimate=2e-9)
            return dataclasses.replace(record, numerical_allowance=allowance)
        monkeypatch.setattr(gate, "evaluate_monolithic_elbo", changed)

    result, _ = run_h1(_config(tmp_path))
    assert result.status is GateStatus.FAIL
    assert any(not invariant.passed and invariant.value is not None for invariant in result.invariants)


def test_nonfinite_injected_computation_is_inconclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import verification.h1_gate as gate

    original = gate.evaluate_monolithic_elbo
    monkeypatch.setattr(gate, "evaluate_monolithic_elbo", lambda *a, **k: dataclasses.replace(original(*a, **k), value=float("nan")))
    result, _ = run_h1(_config(tmp_path))
    assert result.status is GateStatus.INCONCLUSIVE
    assert result.obligations


@pytest.mark.parametrize("probability", [-0.1, 1.1])
def test_finite_evidence_range_disagreement_is_fail_not_inconclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, probability: float
) -> None:
    import verification.h1_gate as gate

    original = gate.h1_all_observation_evidences
    def changed(*args: object, **kwargs: object) -> tuple[object, ...]:
        records = original(*args, **kwargs)
        log_probability = math.log(probability) if probability > 0.0 else records[0].log_probability
        return (dataclasses.replace(records[0], probability=probability, log_probability=log_probability), *records[1:])
    monkeypatch.setattr(gate, "h1_all_observation_evidences", changed)

    result, _ = run_h1(_config(tmp_path))
    assert result.status is GateStatus.FAIL


@pytest.mark.parametrize(
    "control",
    [
        "recognition_source_entropy_omission",
        "selected_raw_logit_substitution",
        "recognition_mixture_for_generative_evidence",
    ],
)
def test_each_negative_control_noop_independently_fails_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, control: str
) -> None:
    import verification.h1_gate as gate

    original = gate._negative_controls
    def no_op(*args: object, **kwargs: object) -> dict[str, dict[str, object]]:
        controls = original(*args, **kwargs)
        controls[control] = {**controls[control], "residual": controls[control]["allowance"], "passed": False}
        return controls
    monkeypatch.setattr(gate, "_negative_controls", no_op)
    result, _ = run_h1(_config(tmp_path))
    assert result.status is GateStatus.FAIL
    assert not next(item for item in result.invariants if item.name == f"negative.{control}").passed
