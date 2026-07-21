from __future__ import annotations

import dataclasses
import sys
from types import MappingProxyType

import pytest
import torch

from vfe4.types import (
    ElboTermAllowances,
    ElboTerms,
    GateResult,
    GateStatus,
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
