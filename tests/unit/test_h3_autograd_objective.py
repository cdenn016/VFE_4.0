from __future__ import annotations

import ast
import math
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import torch

import vfe4.objective.h3_gaussian as objective_module
import vfe4.recognition.reference_h3 as recognition_module
from vfe4.generative import H3GenerativeModel
from vfe4.numerics.information import InformationGaussian
from vfe4.objective import H3ObjectiveEvaluation, evaluate_h3_elbo
from vfe4.recognition import (
    FactorizedH3Parameters,
    H3RecognitionFamily,
    H3VariationalGaussian,
    StructuredH3Parameters,
    make_h3_parameters,
)
from vfe4.types.h1 import GaussianLaw
from vfe4.types.h3 import H3InitializationConfig
from vfe4.validation.h3_fixture import (
    H3_COUPLED_FIXTURE_PATH,
    parse_h3_fixture_bytes,
)


STRICT_LOWER_ROWS = (1, 2, 2, 3, 3, 3)
STRICT_LOWER_COLUMNS = (0, 0, 1, 0, 1, 2)


def _model() -> H3GenerativeModel:
    fixture = parse_h3_fixture_bytes(
        H3_COUPLED_FIXTURE_PATH.read_bytes(),
        expected_fixture_id="h3-coupled-v1",
    )
    return H3GenerativeModel.from_fixture(fixture)


def _structured_q(
    mean: torch.Tensor,
    raw_diagonal: torch.Tensor,
    raw_lower: torch.Tensor,
) -> H3VariationalGaussian:
    cholesky = torch.zeros((4, 4), dtype=torch.float64)
    cholesky[STRICT_LOWER_ROWS, STRICT_LOWER_COLUMNS] = raw_lower
    cholesky = cholesky + torch.diag(torch.exp(raw_diagonal))
    return H3VariationalGaussian(
        family="structured_full_spd",
        mean=mean,
        precision_cholesky=cholesky,
    )


def _factorized_q(
    mean: torch.Tensor, raw_diagonal: torch.Tensor
) -> H3VariationalGaussian:
    return H3VariationalGaussian(
        family="fine_factorized_diagonal",
        mean=mean,
        precision_cholesky=torch.diag(torch.exp(raw_diagonal)),
    )


def test_factory_materializes_fresh_exact_common_initializations() -> None:
    initialization = H3InitializationConfig()
    structured = make_h3_parameters(
        "structured_full_spd", initialization
    )
    structured_again = make_h3_parameters(
        "structured_full_spd", initialization
    )
    factorized = make_h3_parameters(
        "fine_factorized_diagonal", initialization
    )

    assert isinstance(structured, StructuredH3Parameters)
    assert isinstance(factorized, FactorizedH3Parameters)
    assert tuple(dict(structured.named_parameters())) == (
        "mean",
        "raw_diagonal",
        "raw_lower",
    )
    assert tuple(dict(factorized.named_parameters())) == (
        "mean",
        "raw_diagonal",
    )
    assert not hasattr(factorized, "raw_lower")
    assert sum(parameter.numel() for parameter in structured.parameters()) == 14
    assert sum(parameter.numel() for parameter in factorized.parameters()) == 8
    assert {
        id(parameter) for parameter in structured.parameters()
    }.isdisjoint(id(parameter) for parameter in structured_again.parameters())

    identity = torch.eye(4, dtype=torch.float64)
    zero = torch.zeros(4, dtype=torch.float64)
    for module in (structured, structured_again, factorized):
        q = module()
        assert torch.equal(q.mean, zero)
        assert torch.equal(q.precision_cholesky, identity)
        assert torch.equal(q.precision(), identity)
        assert tuple(q.precision().shape) == (4, 4)
        assert all(
            parameter.is_leaf
            and parameter.requires_grad
            and parameter.dtype is torch.float64
            and parameter.device.type == "cpu"
            for parameter in module.parameters()
        )


@pytest.mark.parametrize(
    "family",
    ("structured_full_spd", "fine_factorized_diagonal"),
)
def test_direct_elbo_reconstructs_six_normalized_factor_expectations(
    family: H3RecognitionFamily,
) -> None:
    model = _model()
    q = make_h3_parameters(family, H3InitializationConfig())()
    evaluation = evaluate_h3_elbo(model, q)

    expected = tuple(
        -0.5
        * (
            ((factor.row @ q.mean - factor.target).square()
             + q.linear_variance(factor.row))
            / factor.variance
            + torch.log(2.0 * torch.pi * factor.variance)
        )
        for factor in model.factors
    )

    assert isinstance(evaluation, H3ObjectiveEvaluation)
    assert len(evaluation.expected_log_factors) == 6
    assert all(value.shape == () for value in evaluation.expected_log_factors)
    assert all(
        torch.equal(actual, wanted)
        for actual, wanted in zip(
            evaluation.expected_log_factors, expected, strict=True
        )
    )
    assert torch.equal(evaluation.entropy, q.entropy())
    assert torch.equal(
        evaluation.elbo,
        sum(evaluation.expected_log_factors) + evaluation.entropy,
    )
    assert evaluation.elbo.requires_grad
    with pytest.raises(FrozenInstanceError):
        evaluation.elbo = torch.zeros((), dtype=torch.float64)  # type: ignore[misc]


@pytest.mark.parametrize(
    "family",
    ("structured_full_spd", "fine_factorized_diagonal"),
)
def test_direct_elbo_reaches_every_registered_parameter(
    family: H3RecognitionFamily,
) -> None:
    module = make_h3_parameters(family, H3InitializationConfig())
    evaluation = evaluate_h3_elbo(_model(), module())
    parameters = tuple(module.parameters())
    gradients = torch.autograd.grad(evaluation.elbo, parameters)

    assert len(gradients) == len(parameters)
    assert all(
        gradient is not None and bool(torch.isfinite(gradient).all())
        for gradient in gradients
    )
    if family == "structured_full_spd":
        assert gradients[-1].shape == (6,)
        assert bool(torch.any(gradients[-1] != 0.0))


def test_variance_precision_and_entropy_share_one_cholesky_orientation() -> None:
    mean = torch.tensor([0.1, -0.2, 0.3, -0.4], dtype=torch.float64)
    cholesky = torch.tensor(
        (
            (1.2, 0.0, 0.0, 0.0),
            (0.2, 0.9, 0.0, 0.0),
            (-0.1, 0.3, 1.1, 0.0),
            (0.4, -0.2, 0.25, 0.8),
        ),
        dtype=torch.float64,
    )
    row = torch.tensor([0.7, -0.5, 0.2, 0.9], dtype=torch.float64)
    q = H3VariationalGaussian(
        "structured_full_spd", mean, cholesky
    )
    precision = cholesky @ cholesky.T
    expected_variance = row @ torch.linalg.solve(precision, row)
    expected_entropy = (
        0.5 * 4 * (1.0 + math.log(2.0 * math.pi))
        - torch.log(torch.diagonal(cholesky)).sum()
    )

    assert torch.equal(q.precision(), precision)
    assert torch.allclose(
        q.linear_variance(row), expected_variance, atol=1.0e-14, rtol=0.0
    )
    assert torch.equal(q.entropy(), expected_entropy)


def test_gradcheck_covers_explicit_structured_and_factorized_parameters() -> None:
    model = _model()
    structured_inputs = (
        torch.zeros(4, dtype=torch.float64, requires_grad=True),
        torch.zeros(4, dtype=torch.float64, requires_grad=True),
        torch.zeros(6, dtype=torch.float64, requires_grad=True),
    )
    factorized_inputs = (
        torch.zeros(4, dtype=torch.float64, requires_grad=True),
        torch.zeros(4, dtype=torch.float64, requires_grad=True),
    )

    assert torch.autograd.gradcheck(
        lambda mean, diagonal, lower: evaluate_h3_elbo(
            model, _structured_q(mean, diagonal, lower)
        ).elbo,
        structured_inputs,
    )
    assert torch.autograd.gradcheck(
        lambda mean, diagonal: evaluate_h3_elbo(
            model, _factorized_q(mean, diagonal)
        ).elbo,
        factorized_inputs,
    )


def test_h3_objective_avoids_all_graph_breaking_h1_h2_surfaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    q = make_h3_parameters(
        "structured_full_spd", H3InitializationConfig()
    )()

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("graph-breaking or diagnostic path was called")

    monkeypatch.setattr(GaussianLaw, "__init__", forbidden)
    monkeypatch.setattr(InformationGaussian, "from_information", forbidden)
    monkeypatch.setattr(model, "canonical_joint", forbidden)
    monkeypatch.setattr(model, "log_joint", forbidden)

    evaluation = evaluate_h3_elbo(model, q)
    gradients = torch.autograd.grad(evaluation.elbo, (q.mean,))
    assert evaluation.elbo.requires_grad
    assert gradients[0] is not None


def test_live_h3_modules_have_no_forbidden_runtime_dependencies() -> None:
    allowed_roots = {"__future__", "dataclasses", "math", "typing", "torch", "vfe4"}
    for module in (recognition_module, objective_module):
        source = Path(module.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_roots = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported_roots.update(
            node.module.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        )
        assert imported_roots <= allowed_roots
        for forbidden in (
            ".detach(",
            ".item(",
            ".numpy(",
            "GaussianLaw",
            "InformationGaussian",
            "canonical_joint(",
            "log_joint(",
            "torch.linalg.inv",
            "torch.linalg.pinv",
        ):
            assert forbidden not in source


def test_variational_gaussian_rejects_malformed_tensor_contracts() -> None:
    mean = torch.zeros(4, dtype=torch.float64)
    identity = torch.eye(4, dtype=torch.float64)

    invalid_cases = (
        ("unknown", mean, identity),
        ("structured_full_spd", mean.to(torch.float32), identity),
        ("structured_full_spd", torch.zeros(3, dtype=torch.float64), identity),
        (
            "structured_full_spd",
            torch.tensor([0.0, 0.0, float("nan"), 0.0], dtype=torch.float64),
            identity,
        ),
        ("structured_full_spd", mean, torch.eye(3, dtype=torch.float64)),
        (
            "structured_full_spd",
            mean,
            identity + torch.triu(torch.ones((4, 4), dtype=torch.float64), diagonal=1),
        ),
        (
            "fine_factorized_diagonal",
            mean,
            identity + torch.tensor(
                ((0.0, 0.0, 0.0, 0.0), (0.1, 0.0, 0.0, 0.0),
                 (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)),
                dtype=torch.float64,
            ),
        ),
        (
            "structured_full_spd",
            mean,
            torch.diag(torch.tensor((1.0, 1.0, 0.0, 1.0), dtype=torch.float64)),
        ),
    )
    for family, invalid_mean, invalid_cholesky in invalid_cases:
        with pytest.raises(ValueError):
            H3VariationalGaussian(  # type: ignore[arg-type]
                family, invalid_mean, invalid_cholesky
            )

    with pytest.raises(ValueError, match="CPU"):
        H3VariationalGaussian(
            "structured_full_spd",
            torch.empty(4, dtype=torch.float64, device="meta"),
            identity,
        )


def test_variational_gaussian_and_objective_reject_bad_runtime_values() -> None:
    q = make_h3_parameters(
        "structured_full_spd", H3InitializationConfig()
    )()
    for row in (
        torch.zeros(3, dtype=torch.float64),
        torch.zeros(4, dtype=torch.float32),
        torch.tensor([0.0, 0.0, float("inf"), 0.0], dtype=torch.float64),
    ):
        with pytest.raises(ValueError):
            q.linear_variance(row)

    with pytest.raises(ValueError, match="family"):
        make_h3_parameters("unknown", H3InitializationConfig())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="initialization"):
        make_h3_parameters("structured_full_spd", object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="model"):
        evaluate_h3_elbo(object(), q)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="variational"):
        evaluate_h3_elbo(_model(), object())  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        H3ObjectiveEvaluation(
            expected_log_factors=(torch.zeros((), dtype=torch.float64),),
            entropy=torch.zeros((), dtype=torch.float64),
            elbo=torch.zeros((), dtype=torch.float64),
        )
