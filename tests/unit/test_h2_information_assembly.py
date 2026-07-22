from __future__ import annotations

import pytest
import torch

from vfe4.generative import H1GenerativeModel, assemble_generative_information
from vfe4.numerics import add_initial_gaussian, add_scalar_conditional
from vfe4.recognition import H1RecognitionLaw, assemble_recognition_information
from vfe4.types import SourcePath
from vfe4.validation import enumerate_source_paths, load_h1_fixture


_Y = torch.tensor([0.11, -0.19, 0.27, 0.08, -0.16, 0.23], dtype=torch.float64)


@pytest.mark.parametrize("path_index", range(4))
def test_recognition_information_matches_normalized_h1_component(path_index: int) -> None:
    fixture = load_h1_fixture()
    law = H1RecognitionLaw.from_fixture(fixture)
    path = enumerate_source_paths(fixture)[path_index]

    information = assemble_recognition_information(law.factors, path)

    mean = information.mean()
    torch.testing.assert_close(information.J @ mean - information.h, torch.zeros(6, dtype=torch.float64))
    expected = law.log_prob(_Y, path) - torch.log(law.source_probability(path))
    assert information.log_prob(_Y).item() == pytest.approx(expected.item(), abs=1e-12)


@pytest.mark.parametrize("path_index", range(4))
def test_generative_information_matches_normalized_h1_component(path_index: int) -> None:
    fixture = load_h1_fixture()
    model = H1GenerativeModel.from_fixture(fixture)
    path = enumerate_source_paths(fixture)[path_index]

    information = assemble_generative_information(model.factors, path)

    mean = information.mean()
    torch.testing.assert_close(information.J @ mean - information.h, torch.zeros(6, dtype=torch.float64))
    expected = (
        model.log_joint(_Y, path)
        - model.source_log_prob(path)
        - model.emission_log_prob(_Y, fixture.observation_labels)
    )
    assert information.log_prob(_Y).item() == pytest.approx(expected.item(), abs=1e-12)


def test_assemblers_do_not_call_h1_joint_component(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = load_h1_fixture()
    law = H1RecognitionLaw.from_fixture(fixture)
    model = H1GenerativeModel.from_fixture(fixture)
    path = enumerate_source_paths(fixture)[3]

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("H2 assembly must not call an H1 joint_component")

    monkeypatch.setattr(H1RecognitionLaw, "joint_component", forbidden)
    monkeypatch.setattr(H1GenerativeModel, "joint_component", forbidden)

    assert assemble_recognition_information(law.factors, path).dimension == 6
    assert assemble_generative_information(model.factors, path).dimension == 6


@pytest.mark.parametrize(
    "parents, message",
    [
        (((0, 0.2), (0, -0.1)), "repeated"),
        (((0, 0.2), (3, -0.1)), "out of range"),
    ],
)
def test_scalar_conditional_rejects_repeated_or_forward_parent(
    parents: tuple[tuple[int, float], ...], message: str
) -> None:
    h = torch.zeros(3, dtype=torch.float64)
    J = torch.zeros((3, 3), dtype=torch.float64)

    with pytest.raises(ValueError, match=message):
        add_scalar_conditional(h, J, 2, parents, 0.0, 1.0)


def test_scalar_conditional_rejects_nonpositive_variance() -> None:
    h = torch.zeros(3, dtype=torch.float64)
    J = torch.zeros((3, 3), dtype=torch.float64)

    with pytest.raises(ValueError, match="variance.*positive"):
        add_scalar_conditional(h, J, 2, ((0, 0.2),), 0.0, 0.0)


@pytest.mark.parametrize(
    "assembler, factors",
    [
        (
            assemble_recognition_information,
            lambda: H1RecognitionLaw.from_fixture(load_h1_fixture()).factors,
        ),
        (
            assemble_generative_information,
            lambda: H1GenerativeModel.from_fixture(load_h1_fixture()).factors,
        ),
    ],
)
def test_assemblers_reject_out_of_support_source(
    assembler: object, factors: object
) -> None:
    path = SourcePath(a=(0, 2), b=(0, 0))

    with pytest.raises(ValueError, match="source support"):
        assembler(factors(), path)  # type: ignore[operator]


@pytest.mark.parametrize(
    "operation",
    [
        lambda h, J: add_initial_gaussian(
            h,
            J,
            (0, 1),
            torch.zeros(2, dtype=torch.float64),
            torch.eye(2, dtype=torch.float64),
        ),
        lambda h, J: add_scalar_conditional(h, J, 2, ((0, 0.2),), 0.0, 1.0),
    ],
)
def test_canonical_accumulators_reject_wrong_h_J_shapes(operation: object) -> None:
    h = torch.zeros(3, dtype=torch.float64)
    J = torch.zeros((2, 2), dtype=torch.float64)

    with pytest.raises(ValueError, match="same dimension"):
        operation(h, J)  # type: ignore[operator]
