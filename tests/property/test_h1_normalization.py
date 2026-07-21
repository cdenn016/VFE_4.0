from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from verification.numpy_oracles.h1_elbo import (
    H1EvidenceRecord,
    IndependentNumericalAllowance,
    _label_to_index,
    h1_all_observation_evidences,
    h1_factorized_time_evidence,
    h1_log_evidence,
    h1_p_weights_q_components_evidence,
    h1_permuted_zm_evidence,
    h1_q_weights_p_components_evidence,
    h1_wrong_recognition_mixture_evidence,
)


FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "vfe4"
    / "validation"
    / "fixtures"
    / "h1_v1.json"
)


def _raw_fixture() -> dict[str, object]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _write_fixture(tmp_path: Path, name: str, raw: dict[str, object]) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def test_all_nine_observation_evidences_normalize_in_one_vectorized_evaluation() -> None:
    records = h1_all_observation_evidences(
        FIXTURE_PATH,
        quadrature_order=21,
        convergence_check_order=17,
    )

    assert tuple(record.observation_labels for record in records) == tuple(
        (first, second) for first in (1, 2, 3) for second in (1, 2, 3)
    )
    assert all(0.0 < record.probability <= 1.0 for record in records)
    residual = abs(math.fsum(record.probability for record in records) - 1.0)
    allowance = math.fsum(record.probability_allowance.total for record in records)
    assert residual <= allowance


def test_evidence_is_bitwise_independent_of_recognition_fields(tmp_path: Path) -> None:
    baseline = h1_log_evidence(
        FIXTURE_PATH, (1, 2), quadrature_order=21, convergence_check_order=17
    )
    raw = _raw_fixture()
    recognition = raw["recognition"]
    assert isinstance(recognition, dict)
    recognition["initial_mean"] = [2.0, -3.0]
    recognition["model_source_probabilities"] = [[1.0], [0.7, 0.3]]
    recognition["state_source_probabilities_given_model_source"] = [
        [[1.0]],
        [[0.1, 0.9], [0.65, 0.35]],
    ]
    recognition["model_kernels"][1][0]["offset"] = 1.75
    recognition["state_kernels"][1][3]["z_slope"] = -1.2
    mutated_path = _write_fixture(tmp_path, "recognition-mutated.json", raw)
    mutated = h1_log_evidence(
        mutated_path, (1, 2), quadrature_order=21, convergence_check_order=17
    )

    assert mutated.probability == baseline.probability
    assert mutated.log_probability == baseline.log_probability
    assert mutated.probability_allowance == baseline.probability_allowance
    assert mutated.log_probability_allowance == baseline.log_probability_allowance


@pytest.mark.parametrize(
    ("name", "mutator"),
    [
        (
            "prior",
            lambda raw: raw["model_source_priors"].__setitem__(1, [0.5, 0.5]),
        ),
        (
            "kernel",
            lambda raw: raw["model_offsets"].__setitem__(1, 0.2),
        ),
    ],
)
def test_evidence_changes_under_generative_prior_and_kernel_mutations(
    tmp_path: Path, name: str, mutator: object
) -> None:
    baseline = h1_log_evidence(
        FIXTURE_PATH, (1, 2), quadrature_order=21, convergence_check_order=17
    )
    raw = _raw_fixture()
    mutator(raw)
    mutated = h1_log_evidence(
        _write_fixture(tmp_path, f"generative-{name}.json", raw),
        (1, 2),
        quadrature_order=21,
        convergence_check_order=17,
    )
    allowance = baseline.probability_allowance.total + mutated.probability_allowance.total

    assert abs(mutated.probability - baseline.probability) > allowance


def test_recognition_mixture_substitution_is_detected() -> None:
    correct = h1_log_evidence(
        FIXTURE_PATH, (1, 2), quadrature_order=21, convergence_check_order=17
    )
    wrong = h1_wrong_recognition_mixture_evidence(
        FIXTURE_PATH, (1, 2), quadrature_order=21, convergence_check_order=17
    )
    allowance = correct.probability_allowance.total + wrong.probability_allowance.total

    assert abs(correct.probability - wrong.probability) > allowance


@pytest.mark.parametrize(
    "injection",
    [
        h1_q_weights_p_components_evidence,
        h1_p_weights_q_components_evidence,
        h1_permuted_zm_evidence,
    ],
)
def test_partial_mixture_and_coordinate_permutation_injections_are_detected(
    injection: object,
) -> None:
    correct = h1_log_evidence(
        FIXTURE_PATH, (1, 2), quadrature_order=21, convergence_check_order=17
    )
    wrong = injection(
        FIXTURE_PATH, (1, 2), quadrature_order=21, convergence_check_order=17
    )
    allowance = correct.probability_allowance.total + wrong.probability_allowance.total

    assert abs(correct.probability - wrong.probability) > allowance


def test_cross_time_likelihood_factorization_is_detected() -> None:
    correct = h1_log_evidence(
        FIXTURE_PATH, (1, 2), quadrature_order=21, convergence_check_order=17
    )
    wrong = h1_factorized_time_evidence(
        FIXTURE_PATH, (1, 2), quadrature_order=21, convergence_check_order=17
    )
    allowance = correct.probability_allowance.total + wrong.probability_allowance.total

    assert math.isclose(correct.probability, 0.14371954991133756, rel_tol=0.0, abs_tol=2.0e-15)
    assert math.isclose(wrong.probability, 0.14646363216498376, rel_tol=0.0, abs_tol=2.0e-15)
    assert abs(correct.probability - wrong.probability) > allowance


def test_one_based_label_mapping_is_exact_and_invalid_labels_fail() -> None:
    assert tuple(_label_to_index(label) for label in (1, 2, 3)) == (0, 1, 2)
    for label in (0, 4, True, 1.0):
        with pytest.raises(ValueError, match="label"):
            _label_to_index(label)
        with pytest.raises(ValueError, match="label"):
            h1_log_evidence(
                FIXTURE_PATH,
                (label, 2),
                quadrature_order=21,
                convergence_check_order=17,
            )


def test_positive_recognition_mass_outside_generative_support_is_rejected(
    tmp_path: Path,
) -> None:
    raw = copy.deepcopy(_raw_fixture())
    raw["model_source_priors"][1] = [1.0, 0.0]
    path = _write_fixture(tmp_path, "outside-support.json", raw)

    with pytest.raises(ValueError, match="outside positive generative support"):
        h1_wrong_recognition_mixture_evidence(
            path, (1, 2), quadrature_order=21, convergence_check_order=17
        )


def test_evidence_record_rejects_inconsistent_probability_log_pair() -> None:
    allowance = IndependentNumericalAllowance(0.0, 0.0)
    with pytest.raises(ValueError, match="inconsistent"):
        H1EvidenceRecord((1, 2), 0.25, math.log(0.5), allowance, allowance)


@pytest.mark.parametrize(
    ("quadrature_order", "check_order"),
    [(17, 17), (21, 21), (19, 17), (21, False)],
)
def test_independent_evidence_rejects_nonfrozen_orders(
    quadrature_order: int, check_order: int
) -> None:
    with pytest.raises(ValueError, match="order"):
        h1_log_evidence(
            FIXTURE_PATH,
            (1, 2),
            quadrature_order=quadrature_order,
            convergence_check_order=check_order,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda raw: raw["initial_joint"].__setitem__(
            "covariance", [[1.0, 2.0], [2.0, 1.0]]
        ),
        lambda raw: raw["initial_joint"].__setitem__("mean", [math.nan, 0.0]),
    ],
)
def test_independent_evidence_rejects_non_spd_and_nonfinite_raw_data(
    tmp_path: Path, mutation: object
) -> None:
    raw = _raw_fixture()
    mutation(raw)
    path = _write_fixture(tmp_path, "invalid-numerics.json", raw)

    with pytest.raises(ValueError):
        h1_log_evidence(
            path, (1, 2), quadrature_order=21, convergence_check_order=17
        )
