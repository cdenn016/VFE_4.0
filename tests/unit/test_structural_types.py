from __future__ import annotations

import dataclasses
import sys
from collections.abc import Mapping
from types import MappingProxyType
from typing import get_type_hints

import pytest
import torch

from vfe4.types import (
    ElboTermAllowances,
    ElboTerms,
    CurrentH8PrerequisiteRefs,
    GateResult,
    GateStatus,
    H7PredecessorReference,
    H8GateResult,
    H8H1H5Reference,
    H8H1PrefixPriorReference,
    H8H6PredictionReference,
    H8H6PrefixReference,
    H8H7Reference,
    H8_H7_PLAN_SHA256,
    H8_INTERPRETATION_SHA256,
    InvariantResult,
    NumericalAllowance,
    PopulationFrames,
    SourcePath,
    StructuralData,
)


def _structure(**overrides: object) -> StructuralData:
    values: dict[str, object] = {
        "horizon": 2,
        "d_z": 1,
        "d_m": 1,
        "vocabulary_size": 3,
        "state_parent_sets": ((0,), (0, 1)),
        "model_parent_sets": ((0,), (0, 1)),
        "state_source_support": ((0,), (0, 1)),
        "model_source_support": ((0,), (0, 1)),
    }
    values.update(overrides)
    return StructuralData(**values)  # type: ignore[arg-type]


def test_structural_data_accepts_the_h1_shape() -> None:
    assert _structure().state_parent_sets == ((0,), (0, 1))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("state_parent_sets", ((0,),)),
        ("model_parent_sets", ((0,), (0, 2))),
        ("model_source_support", ((0,), (0, 2))),
    ],
)
def test_structural_data_rejects_malformed_or_out_of_range_sets(
    field: str, value: object
) -> None:
    with pytest.raises(ValueError, match=field):
        _structure(**{field: value})


def test_structural_data_is_frozen() -> None:
    structure = _structure()

    with pytest.raises(dataclasses.FrozenInstanceError):
        structure.horizon = 3  # type: ignore[misc]


def test_source_path_requires_nonnegative_coordinate_pairs() -> None:
    assert SourcePath((0, 1), (2, 3)).b == (2, 3)
    with pytest.raises(ValueError, match="a"):
        SourcePath((0,), (2, 3))  # type: ignore[arg-type]


def test_population_frames_returns_scalar_ratio_and_owns_tensor() -> None:
    raw = torch.tensor([2.0, 4.0, 8.0], dtype=torch.float64)
    frames = PopulationFrames(raw)
    raw[0] = 20.0

    ratio = frames.omega(2, 1)

    assert ratio.dtype is torch.float64
    assert ratio.item() == pytest.approx(2.0)
    returned = frames.values
    returned[0] = 99.0
    assert frames.values[0].item() == pytest.approx(2.0)


@pytest.mark.parametrize(
    "value",
    [
        torch.ones((1, 3), dtype=torch.float64),
        torch.tensor([1.0, 0.0, 2.0], dtype=torch.float64),
        torch.tensor([1.0, float("nan"), 2.0], dtype=torch.float64),
        torch.ones(3, dtype=torch.float32),
    ],
)
def test_population_frames_rejects_invalid_values(value: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="values"):
        PopulationFrames(value)


@pytest.mark.parametrize("receiver, source", [(-1, 0), (0, 3)])
def test_population_frames_checks_omega_indices(receiver: int, source: int) -> None:
    frames = PopulationFrames(torch.ones(3, dtype=torch.float64))

    with pytest.raises(ValueError, match="index"):
        frames.omega(receiver, source)


def test_population_frames_rejects_an_overflowing_derived_ratio() -> None:
    frames = PopulationFrames(
        torch.tensor([sys.float_info.max, sys.float_info.min, 1.0], dtype=torch.float64)
    )

    with pytest.raises(ValueError, match="omega"):
        frames.omega(0, 1)


def _allowance() -> NumericalAllowance:
    return NumericalAllowance(convergence_estimate=0.0, rounding_allowance=1e-15)


def test_numerical_allowance_is_nonnegative_and_sums_its_components() -> None:
    allowance = NumericalAllowance(0.125, 0.25)

    assert allowance.total == pytest.approx(0.375)
    with pytest.raises(ValueError, match="convergence_estimate"):
        NumericalAllowance(-1.0, 0.0)


def test_numerical_allowance_rejects_an_overflowing_total() -> None:
    with pytest.raises(ValueError, match="total"):
        NumericalAllowance(sys.float_info.max, sys.float_info.max)


def _term_allowances() -> ElboTermAllowances:
    allowance = _allowance()
    return ElboTermAllowances(
        expected_log_emission=(allowance, allowance),
        initial_model_kl=allowance,
        initial_state_kl=allowance,
        model_source_kl=(allowance, allowance),
        model_transition_kl=(allowance, allowance),
        state_source_kl=(allowance, allowance),
        state_transition_kl=(allowance, allowance),
        joint_recognition_entropy=allowance,
        complete_elbo=allowance,
    )


def test_elbo_terms_accepts_consistent_partition_without_double_counting_entropy() -> None:
    terms = ElboTerms(
        expected_log_emission=(-2.0, -3.0),
        initial_model_kl=1.0,
        initial_state_kl=2.0,
        model_source_kl=(0.5, 0.25),
        model_transition_kl=(0.75, 0.5),
        state_source_kl=(0.25, 0.125),
        state_transition_kl=(0.5, 0.25),
        joint_recognition_entropy=1.25,
        allowances=_term_allowances(),
        complete_elbo=-11.125,
    )

    assert terms.complete_elbo == pytest.approx(-11.125)


def test_elbo_terms_rejects_an_inconsistent_complete_total() -> None:
    with pytest.raises(ValueError, match="complete_elbo"):
        ElboTerms(
            expected_log_emission=(-2.0, -3.0),
            initial_model_kl=1.0,
            initial_state_kl=2.0,
            model_source_kl=(0.5, 0.25),
            model_transition_kl=(0.75, 0.5),
            state_source_kl=(0.25, 0.125),
            state_transition_kl=(0.5, 0.25),
            joint_recognition_entropy=1.25,
            allowances=_term_allowances(),
            complete_elbo=-11.0,
        )


def test_elbo_terms_rejects_an_overflowing_derived_objective() -> None:
    with pytest.raises(ValueError, match="expected objective"):
        ElboTerms(
            expected_log_emission=(sys.float_info.max, sys.float_info.max),
            initial_model_kl=0.0,
            initial_state_kl=0.0,
            model_source_kl=(0.0, 0.0),
            model_transition_kl=(0.0, 0.0),
            state_source_kl=(0.0, 0.0),
            state_transition_kl=(0.0, 0.0),
            joint_recognition_entropy=0.0,
            allowances=_term_allowances(),
            complete_elbo=0.0,
        )


def test_gate_result_uses_an_immutable_copy_of_measurements() -> None:
    measurements = {"elbo": 2.0}
    result = GateResult(
        gate="H1",
        status=GateStatus.PASS,
        fixture_id="h1-v1",
        residual=0.0,
        calibrated_allowance=1e-12,
        measurements=measurements,
        invariants=(InvariantResult("normalization", True, 1.0, 1.0, "ok"),),
        obligations=(),
    )
    measurements["elbo"] = 3.0

    assert isinstance(result.measurements, MappingProxyType)
    assert result.measurements["elbo"] == pytest.approx(2.0)
    with pytest.raises(TypeError):
        result.measurements["new"] = 1.0  # type: ignore[index]


def test_gate_result_requires_obligation_when_inconclusive() -> None:
    with pytest.raises(ValueError, match="obligation"):
        GateResult(
            gate="H1",
            status=GateStatus.INCONCLUSIVE,
            fixture_id="h1-v1",
            residual=None,
            calibrated_allowance=None,
            measurements={"elbo": None},
            invariants=(),
            obligations=(),
        )


@pytest.mark.parametrize(
    ("residual", "calibrated_allowance"),
    [
        (float("nan"), None),
        (float("inf"), None),
        (None, float("nan")),
        (None, float("inf")),
    ],
)
def test_inconclusive_gate_rejects_nonfinite_optional_scalars(
    residual: float | None, calibrated_allowance: float | None
) -> None:
    with pytest.raises(ValueError):
        GateResult(
            gate="H1",
            status=GateStatus.INCONCLUSIVE,
            fixture_id="h1-v1",
            residual=residual,
            calibrated_allowance=calibrated_allowance,
            measurements={"elbo": None},
            invariants=(),
            obligations=("obtain the unavailable measurement",),
        )


def _current_h8_refs() -> tuple[
    CurrentH8PrerequisiteRefs,
    dict[str, H7PredecessorReference],
]:
    digest = "a" * 64
    head = "1" * 40
    h7_compatibility_refs = {
        key: H7PredecessorReference.create(
            artifact_path=f"{key}-artifact",
            git_head=head,
            dirty_digest=digest,
            junit_sha256=digest,
            junit_path=f"{key}-junit.xml",
            manifest_sha256=digest,
            payload_hashes={f"{key}.json": digest},
            ledger_path=f"{key}-ledger",
            ledger_sha256=digest,
        )
        for key in ("h1_h5", "h1_prefix_prior", "h6_prefix")
    }

    def common(key: str) -> dict[str, object]:
        transitive = h7_compatibility_refs[key]
        return {
            "artifact_path": transitive.artifact_path,
            "manifest_sha256": transitive.manifest_sha256,
            "result_path": f"{key}-result",
            "result_sha256": digest,
            "content_hashes": {f"{key}-content": digest},
            "payload_hashes": dict(transitive.payload_hashes),
            "ledger_path": transitive.ledger_path,
            "ledger_sha256": transitive.ledger_sha256,
            "producer_head": transitive.git_head,
            "producer_dirty_digest": transitive.dirty_digest,
            "candidate_junit_sha256": transitive.junit_sha256,
            "status": "pass",
        }

    h7_common: dict[str, object] = {
        "artifact_path": "h7-artifact",
        "manifest_sha256": digest,
        "result_path": "h7-result",
        "result_sha256": digest,
        "content_hashes": {"h7-content": digest},
        "payload_hashes": {"h7.json": digest},
        "ledger_path": "h7-ledger",
        "ledger_sha256": digest,
        "producer_head": head,
        "producer_dirty_digest": digest,
        "candidate_junit_sha256": digest,
        "status": "pass",
    }
    prediction_common = {
        **h7_common,
        "artifact_path": "prediction-artifact",
        "result_path": "prediction-result",
        "content_hashes": {"prediction-content": digest},
        "payload_hashes": {"prediction.json": digest},
        "ledger_path": "prediction-ledger",
        "candidate_junit_sha256": digest,
    }
    refs = CurrentH8PrerequisiteRefs(
        candidate_head=head,
        candidate_dirty_digest=digest,
        candidate_junit_sha256=digest,
        h7_compatibility_refs=h7_compatibility_refs,
        h1_h5=H8H1H5Reference(
            kind="h1_h5", **common("h1_h5")  # type: ignore[arg-type]
        ),
        h1_prefix_prior=H8H1PrefixPriorReference(
            kind="h1_prefix_prior",
            **common("h1_prefix_prior"),  # type: ignore[arg-type]
        ),
        h6_prefix=H8H6PrefixReference(
            kind="h6_prefix",
            certificate_set_sha256=digest,
            certificate_hashes={"certificate": digest},
            **common("h6_prefix"),  # type: ignore[arg-type]
        ),
        h7=H8H7Reference(
            kind="h7",
            result_pointer_path="h7-result-pointer",
            result_pointer_sha256=digest,
            fixture_set_sha256=digest,
            **h7_common,  # type: ignore[arg-type]
        ),
        h6_prediction=H8H6PredictionReference(
            kind="h6_prediction",
            prediction_schema="h6-prediction-amended-v2",
            config_schema="h6-prediction-config-v2",
            readiness_schema="h6-prediction-readiness-v2",
            metrics_schema="h6-prediction-metrics-v2",
            result_schema="h6-prediction-result-v2",
            experiment_sha256=digest,
            config_sha256=digest,
            readiness_artifact_path="prediction-readiness",
            readiness_manifest_sha256=digest,
            readiness_sha256=digest,
            correctness_artifact_paths={
                gate: f"prediction-{gate.lower()}-correctness"
                for gate in ("H1", "H2", "H3", "H5")
            },
            h1_prefix_prior_artifact_path="prediction-h1-prefix-prior",
            smc_accuracy_artifact_path="prediction-smc-accuracy",
            smc_accuracy_manifest_sha256=digest,
            h6_prefix_artifact_path="prediction-h6-prefix",
            h6_prefix_manifest_sha256=digest,
            blinded_data_artifact_path="prediction-blinded-data",
            blinded_data_manifest_sha256=digest,
            matching_artifact_path="prediction-matching",
            matching_manifest_sha256=digest,
            matching_set_sha256=digest,
            h1_prefix_prior_generative_factor_schema_sha256=digest,
            smc_bias_semantics_sha256=digest,
            objective_gate_spec_sha256=digest,
            metrics_sha256=digest,
            **prediction_common,  # type: ignore[arg-type]
        ),
        registry_sha256=digest,
    )
    return refs, h7_compatibility_refs


def test_current_h8_h7_references_are_exact_lossless_and_immutable() -> None:
    refs, source = _current_h8_refs()
    original = tuple(source.items())

    assert (
        get_type_hints(CurrentH8PrerequisiteRefs)["h7_compatibility_refs"]
        == Mapping[str, H7PredecessorReference]
    )
    assert tuple(refs.h7_compatibility_refs.items()) == original
    assert all(
        retained is supplied
        and retained.reference_sha256 == supplied.reference_sha256
        and retained.payload_hashes == supplied.payload_hashes
        for (_, retained), (_, supplied) in zip(
            refs.h7_compatibility_refs.items(),
            original,
            strict=True,
        )
    )

    source.clear()
    assert tuple(refs.h7_compatibility_refs.items()) == original
    with pytest.raises(TypeError):
        refs.h7_compatibility_refs["h1_h5"] = original[0][1]  # type: ignore[index]
    with pytest.raises(TypeError):
        original[0][1].payload_hashes["replacement"] = "b" * 64  # type: ignore[index]


def test_current_h8_rejects_a_lossy_untyped_h7_reference() -> None:
    refs, source = _current_h8_refs()
    source["h6_prefix"] = {"reference_sha256": "a" * 64}  # type: ignore[assignment]

    with pytest.raises(ValueError, match="exact types"):
        dataclasses.replace(refs, h7_compatibility_refs=source)


def test_current_h8_rejects_direct_reference_drift_from_h7_transitive_bytes() -> None:
    refs, _source = _current_h8_refs()
    changed = dataclasses.replace(
        refs.h1_prefix_prior,
        result_path="an-independent-result-path-is-allowed",
    )
    assert dataclasses.replace(refs, h1_prefix_prior=changed)

    changed_payload = dataclasses.replace(
        refs.h1_prefix_prior,
        payload_hashes={"different.json": "b" * 64},
    )
    with pytest.raises(ValueError, match="H7 transitive"):
        dataclasses.replace(refs, h1_prefix_prior=changed_payload)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("producer_head", "2" * 40),
        ("producer_dirty_digest", "b" * 64),
        ("candidate_junit_sha256", "b" * 64),
    ),
)
def test_current_h8_rejects_amended_prediction_from_another_candidate(
    field: str,
    value: str,
) -> None:
    refs, _source = _current_h8_refs()
    changed_prediction = dataclasses.replace(
        refs.h6_prediction,
        **{field: value},
    )

    with pytest.raises(ValueError, match="same candidate and JUnit"):
        dataclasses.replace(refs, h6_prediction=changed_prediction)


def test_h8_task7_result_and_pins_are_fail_closed() -> None:
    digest = "a" * 64
    result = H8GateResult(
        gate="H8",
        status=GateStatus.INCONCLUSIVE,
        config_sha256=digest,
        candidate_junit_sha256=None,
        current_refs_registry_sha256=None,
        h7_manifest_sha256=None,
        h6_prediction_manifest_sha256=None,
        correctness=(),
        production_runs=(),
        profiler_runs=(),
        controls=(),
        obligations=("produce current prerequisite evidence",),
    )
    assert dataclasses.is_dataclass(result)
    assert H8GateResult.__dataclass_params__.frozen
    assert result.status is GateStatus.INCONCLUSIVE
    assert (
        H8_H7_PLAN_SHA256,
        H8_INTERPRETATION_SHA256,
    ) == (
        "3549153ac123b26f1d2372c59e80db93a78ed451fd4724781280dd7f413f1242",
        "e3fd048126c8133384e026826cf00bbea08280f4e248bc4cd5689e8f9f26e865",
    )

    with pytest.raises(ValueError, match="witnessed-failure evidence"):
        dataclasses.replace(
            result,
            status=GateStatus.FAIL,
            candidate_junit_sha256=digest,
            current_refs_registry_sha256=digest,
            h7_manifest_sha256=digest,
            h6_prediction_manifest_sha256=digest,
            obligations=(),
        )
