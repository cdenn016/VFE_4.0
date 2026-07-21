from __future__ import annotations

import dataclasses
import math

import pytest

from vfe4.generative import H1GenerativeModel
from vfe4.objective import evaluate_local_elbo
from vfe4.recognition import H1RecognitionLaw
from vfe4.validation import load_h1_fixture


def _evaluate() -> object:
    fixture = load_h1_fixture()
    return evaluate_local_elbo(
        H1GenerativeModel.from_fixture(fixture),
        H1RecognitionLaw.from_fixture(fixture),
        quadrature_order=fixture.quadrature_order,
        convergence_check_order=fixture.convergence_check_order,
    )


def test_local_elbo_has_finite_signed_terms_and_frozen_array_lengths() -> None:
    terms = _evaluate()

    assert len(terms.expected_log_emission) == 2  # type: ignore[attr-defined]
    assert all(value < 0.0 and math.isfinite(value) for value in terms.expected_log_emission)  # type: ignore[attr-defined]
    for name in (
        "model_source_kl",
        "model_transition_kl",
        "state_source_kl",
        "state_transition_kl",
    ):
        values = getattr(terms, name)
        assert len(values) == 2
        assert all(value >= 0.0 and math.isfinite(value) for value in values)
    assert terms.initial_model_kl >= 0.0  # type: ignore[attr-defined]
    assert terms.initial_state_kl >= 0.0  # type: ignore[attr-defined]
    assert terms.joint_recognition_entropy > 0.0  # type: ignore[attr-defined]
    assert terms.complete_elbo < 0.0  # type: ignore[attr-defined]


def test_local_elbo_records_term_shaped_convergence_and_rounding_allowances() -> None:
    terms = _evaluate()
    allowances = terms.allowances  # type: ignore[attr-defined]

    assert len(allowances.expected_log_emission) == 2
    assert len(allowances.model_source_kl) == 2
    assert len(allowances.model_transition_kl) == 2
    assert len(allowances.state_source_kl) == 2
    assert len(allowances.state_transition_kl) == 2
    for field in dataclasses.fields(allowances):
        value = getattr(allowances, field.name)
        values = value if isinstance(value, tuple) else (value,)
        assert all(item.convergence_estimate >= 0.0 for item in values)
        assert all(item.rounding_allowance >= 0.0 for item in values)
        assert all(math.isfinite(item.total) for item in values)
    assert allowances.complete_elbo.convergence_estimate < 1e-9


@pytest.mark.parametrize(
    ("quadrature_order", "convergence_check_order"),
    [(17, 17), (21, 21), (21, 15), (True, 17)],
)
def test_local_elbo_rejects_nonfrozen_quadrature_orders(
    quadrature_order: int, convergence_check_order: int
) -> None:
    fixture = load_h1_fixture()

    with pytest.raises(ValueError, match="quadrature"):
        evaluate_local_elbo(
            H1GenerativeModel.from_fixture(fixture),
            H1RecognitionLaw.from_fixture(fixture),
            quadrature_order=quadrature_order,
            convergence_check_order=convergence_check_order,
        )
