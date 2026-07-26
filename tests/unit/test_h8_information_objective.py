"""Cohesive production-contract tests for the sparse H8 objective."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import FrozenInstanceError, fields, replace

import numpy as np
import pytest
import torch

from vfe4.generative import reference_h8 as h8_reference
from vfe4.generative.reference_h8 import (
    H8Problem,
    build_h8_generative,
    h8_sample_noise,
    make_h8_problem,
    validate_h8_problem,
)
from vfe4.numerics.block_canonical import BlockCanonicalAssembler
from vfe4.numerics.block_layout import BlockChainLayout
from vfe4.numerics.block_tridiagonal import BlockTridiagonalCholesky
from vfe4.numerics.sparse_information import FactorBackedInformationGaussian
from vfe4.objective.h8_sparse import (
    evaluate_h8_sparse_objective,
    h8_emission_expectation,
)
from vfe4.recognition.reference_h8 import build_h8_recognition
from vfe4.types import h8 as h8_types


def _guard_global_allocations(
    monkeypatch: pytest.MonkeyPatch,
    layout: BlockChainLayout,
) -> None:
    """Reject a global axis/result while allowing registered block shapes."""

    def checked(name: str, original: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        def wrapper(*args: object, **kwargs: object) -> torch.Tensor:
            result = original(*args, **kwargs)
            shape = tuple(int(item) for item in result.shape)
            assert layout.dimension not in shape, (name, shape)
            assert shape != (layout.dimension, layout.dimension), (name, shape)
            return result

        return wrapper

    for name in ("zeros", "empty", "eye", "outer", "cat", "stack"):
        monkeypatch.setattr(torch, name, checked(name, getattr(torch, name)))


def _small_problem(seed: int = 2026072121) -> H8Problem:
    return make_h8_problem(horizon=2, channel_dimension=1, problem_seed=seed)


class _ZeroDrawGenerator:
    """One fixed draw stream for the hand-derived objective fixture."""

    def standard_normal(
        self,
        *,
        size: tuple[int, ...],
        dtype: type[np.float64],
    ) -> np.ndarray:
        assert dtype is np.float64
        return np.zeros(size, dtype=np.float64)


def _hand_derived_problem(monkeypatch: pytest.MonkeyPatch) -> H8Problem:
    monkeypatch.setattr(
        np.random,
        "Generator",
        lambda bit_generator: _ZeroDrawGenerator(),
    )
    return make_h8_problem(
        horizon=1,
        channel_dimension=1,
        problem_seed=2,
    )


def test_canonical_scatter_has_exact_signs_and_only_local_support() -> None:
    layout = BlockChainLayout(horizon=2, d_z=1, d_m=1)
    assembler = BlockCanonicalAssembler(layout)
    assembler.add_initial(
        torch.tensor([1.0, -2.0], dtype=torch.float64),
        torch.diag(torch.tensor([2.0, 4.0], dtype=torch.float64)),
    )
    assembler.add_transition(
        2,
        torch.diag(torch.tensor([2.0, 4.0], dtype=torch.float64)),
        torch.tensor([0.5, -1.0], dtype=torch.float64),
        torch.diag(torch.tensor([4.0, 8.0], dtype=torch.float64)),
    )
    precision, h = assembler.freeze()
    diagonal, lower = precision._block_refs()

    assert torch.equal(h[0], torch.tensor([0.5, -0.5], dtype=torch.float64))
    assert torch.equal(h[1], torch.tensor([-0.25, 0.5], dtype=torch.float64))
    assert torch.equal(h[2], torch.tensor([0.125, -0.125], dtype=torch.float64))
    assert torch.equal(
        diagonal[0],
        torch.diag(torch.tensor([0.5, 0.25], dtype=torch.float64)),
    )
    assert torch.equal(
        diagonal[1],
        torch.diag(torch.tensor([1.0, 2.0], dtype=torch.float64)),
    )
    assert torch.equal(
        diagonal[2],
        torch.diag(torch.tensor([0.25, 0.125], dtype=torch.float64)),
    )
    assert torch.count_nonzero(lower[0]).item() == 0
    assert torch.equal(
        lower[1],
        torch.diag(torch.tensor([-0.5, -0.5], dtype=torch.float64)),
    )
    with pytest.raises(RuntimeError, match="frozen"):
        assembler.add_initial(h[0], torch.eye(2, dtype=torch.float64))

    observation = BlockCanonicalAssembler(layout)
    observation.add_local_observation(
        1,
        torch.tensor([[1.0, 2.0]], dtype=torch.float64),
        torch.tensor([0.5], dtype=torch.float64),
        torch.tensor([1.5], dtype=torch.float64),
        torch.tensor([[2.0]], dtype=torch.float64),
    )
    observed_precision, observed_h = observation.freeze()
    observed_diagonal, observed_lower = observed_precision._block_refs()
    assert torch.equal(
        observed_diagonal[1],
        torch.tensor([[0.5, 1.0], [1.0, 2.0]], dtype=torch.float64),
    )
    assert torch.equal(
        observed_h[1],
        torch.tensor([0.5, 1.0], dtype=torch.float64),
    )
    assert torch.count_nonzero(observed_diagonal[[0, 2]]).item() == 0
    assert torch.count_nonzero(observed_lower).item() == 0


def test_factor_backing_owns_inputs_and_exposes_selected_local_moments() -> None:
    layout = BlockChainLayout(horizon=1, d_z=1, d_m=1)
    assembler = BlockCanonicalAssembler(layout)
    covariance = torch.eye(2, dtype=torch.float64)
    assembler.add_initial(torch.zeros(2, dtype=torch.float64), covariance)
    assembler.add_transition(
        1,
        0.25 * torch.eye(2, dtype=torch.float64),
        torch.zeros(2, dtype=torch.float64),
        covariance,
    )
    precision, h = assembler.freeze()
    factor = BlockTridiagonalCholesky.factorize(precision)
    gaussian = FactorBackedInformationGaussian.from_factor(h, factor)
    original_mean = gaussian.mean()
    assert torch.abs(factor.trace_inverse_product(precision) - layout.dimension) < 1e-13
    h.add_(100.0)
    precision._block_refs()[0].add_(100.0)

    moments = gaussian.selected_moment_blocks()
    assert torch.equal(gaussian.mean(), original_mean)
    assert moments.diagonal.shape == (2, 2, 2)
    assert moments.lower.shape == (1, 2, 2)
    for forbidden in ("J", "Sigma", "covariance", "flatten", "dense"):
        assert not hasattr(gaussian, forbidden)


def test_problem_identity_is_factory_only_complete_and_consumer_validated() -> None:
    first = _small_problem()
    repeated = _small_problem()
    different = _small_problem(2026072122)

    with pytest.raises(TypeError, match="factory-only"):
        H8Problem()
    assert validate_h8_problem(first) is first
    assert first.serialized_bytes == repeated.serialized_bytes
    assert first.input_sha256 == repeated.input_sha256
    assert first.serialized_bytes != different.serialized_bytes
    assert all(
        transition.parent_t == transition.receiver_t - 1
        and transition.source_support == (transition.receiver_t - 1,)
        for transition in (
            *first.model_transitions,
            *first.state_transitions,
            *first.recognition.transitions,
        )
    )

    corrupted = _small_problem()
    object.__setattr__(
        corrupted.model_transitions[0],
        "source_support",
        (corrupted.model_transitions[0].receiver_t,),
    )
    for consumer in (validate_h8_problem, build_h8_generative, build_h8_recognition):
        with pytest.raises(ValueError):
            consumer(corrupted)

    bad_hash = _small_problem()
    good_recognition = build_h8_recognition(bad_hash)
    object.__setattr__(bad_hash, "input_sha256", "0" * 64)
    with pytest.raises(ValueError):
        h8_sample_noise(bad_hash, sample_noise_seed=2026172121)
    with pytest.raises(ValueError):
        evaluate_h8_sparse_objective(bad_hash, good_recognition.gaussian)


def test_complete_objective_is_normalized_local_and_allocation_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = _small_problem()
    _guard_global_allocations(monkeypatch, problem.layout)

    def forbidden_dense(*args: object, **kwargs: object) -> torch.Tensor:
        del args, kwargs
        raise AssertionError("dense inverse/covariance API is forbidden")

    monkeypatch.setattr(torch.linalg, "inv", forbidden_dense)
    monkeypatch.setattr(torch, "inverse", forbidden_dense)
    monkeypatch.setattr(torch, "cov", forbidden_dense)
    generative = build_h8_generative(problem)
    recognition = build_h8_recognition(problem)
    terms = evaluate_h8_sparse_objective(problem, recognition.gaussian)

    expected_ids = (
        "initial_joint",
        "model_transition:0001",
        "model_transition:0002",
        "state_transition:0001",
        "state_transition:0002",
        "emission_order21:0001",
        "emission_order21:0002",
        "emission_order17:0001",
        "emission_order17:0002",
    )
    observed_ids = tuple(
        term.factor_id
        for term in (
            terms.initial_joint,
            *terms.model_transitions,
            *terms.state_transitions,
            *terms.emissions_order21,
            *terms.emissions_order17,
        )
    )
    assert observed_ids == expected_ids
    assert terms.model_source_kl == terms.state_source_kl == terms.source_entropy == 0.0
    assert math.isfinite(terms.complete_order21)
    assert math.isfinite(terms.log_normalizer)
    assert math.isfinite(float(generative.gaussian.log_prob(generative.gaussian.mean())))

def test_fixed_quadrature_orders_use_stable_normalized_log_softmax() -> None:
    mean = torch.tensor(0.3, dtype=torch.float64)
    variance = torch.tensor(9.0, dtype=torch.float64)
    alpha = torch.tensor((-0.5, 0.25, 0.75), dtype=torch.float64)
    bias = torch.tensor((1_000.0, 1_000.0, 1_000.0), dtype=torch.float64)
    order21 = h8_emission_expectation(
        mean, variance, alpha, bias, 2, order=21
    )
    order17 = h8_emission_expectation(
        mean, variance, alpha, bias, 2, order=17
    )

    assert math.isfinite(order21)
    assert math.isfinite(order17)
    assert order21 <= 0.0 and order17 <= 0.0
    assert order21 != order17


def test_fixed_hand_derived_fixture_pins_every_objective_component(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = _hand_derived_problem(monkeypatch)
    recognition = build_h8_recognition(problem)
    log_softmax_grad_states: list[bool] = []
    original_log_softmax = torch.log_softmax

    def observed_log_softmax(
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        log_softmax_grad_states.append(torch.is_grad_enabled())
        return original_log_softmax(*args, **kwargs)

    monkeypatch.setattr(torch, "log_softmax", observed_log_softmax)
    with torch.enable_grad():
        terms = evaluate_h8_sparse_objective(problem, recognition.gaussian)
    mean = recognition.gaussian.mean()
    selected = recognition.gaussian.selected_moment_blocks()
    entropy = recognition.gaussian.entropy()
    log_normalizer = recognition.gaussian.log_normalizer()

    log_variance_normalizer = math.log(math.pi / 2.0)
    expected_scalar_gaussian = -0.5 * (
        1.0 + log_variance_normalizer
    )
    expected_initial_joint = 2.0 * expected_scalar_gaussian
    expected_entropy = 2.0 * (1.0 + log_variance_normalizer)
    expected_log_normalizer = 2.0 * log_variance_normalizer
    expected_emission = -math.log(3.0)

    assert terms.initial_joint.value == pytest.approx(
        expected_initial_joint,
        abs=1e-13,
    )
    assert terms.model_transitions[0].value == pytest.approx(
        expected_scalar_gaussian,
        abs=1e-13,
    )
    assert terms.state_transitions[0].value == pytest.approx(
        expected_scalar_gaussian,
        abs=1e-13,
    )
    assert terms.emissions_order21[0].value == pytest.approx(
        expected_emission,
        abs=1e-13,
    )
    assert terms.emissions_order17[0].value == pytest.approx(
        expected_emission,
        abs=1e-13,
    )
    assert terms.quadrature_absolute_difference == pytest.approx(0.0, abs=1e-13)
    assert terms.recognition_entropy == pytest.approx(
        expected_entropy,
        abs=1e-13,
    )
    assert terms.log_normalizer == pytest.approx(
        expected_log_normalizer,
        abs=1e-13,
    )
    assert terms.complete_order21 == pytest.approx(
        expected_emission,
        abs=1e-13,
    )

    diagonal = selected.diagonal
    lower = selected.lower
    assert diagonal[0, 0, 0].item() == pytest.approx(0.25, abs=1e-14)
    assert diagonal[0, 1, 1].item() == pytest.approx(0.25, abs=1e-14)
    assert diagonal[1, 0, 0].item() == pytest.approx(0.25, abs=1e-14)
    assert diagonal[1, 1, 1].item() == pytest.approx(0.25, abs=1e-14)
    assert torch.count_nonzero(
        diagonal
        - 0.25
        * torch.eye(2, dtype=torch.float64).repeat(2, 1, 1)
    ).item() == 0
    assert torch.count_nonzero(lower).item() == 0

    logits = (
        torch.tensor(problem.alpha, dtype=torch.float64)
        * torch.tensor(0.0, dtype=torch.float64)
        + torch.tensor(problem.emissions[0].bias, dtype=torch.float64)
    )
    probabilities = torch.softmax(logits, dim=0)
    assert probabilities.sum().item() == pytest.approx(1.0, abs=1e-15)
    assert torch.equal(
        probabilities,
        torch.full((3,), 1.0 / 3.0, dtype=torch.float64),
    )

    assert not mean.requires_grad
    assert not diagonal.requires_grad
    assert not lower.requires_grad
    assert not entropy.requires_grad
    assert not log_normalizer.requires_grad
    assert type(terms.complete_order21) is float
    assert log_softmax_grad_states
    assert not any(log_softmax_grad_states)


def test_h8_problem_evidence_is_exact_owned_and_single_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch lossy/stale evidence and repeated raw-norm calculations."""

    assert hasattr(h8_types, "H8LocalSPDDiagnostics")
    assert hasattr(h8_types, "H8TransitionNorms")
    assert hasattr(h8_types, "H8ProductionProblemEvidence")

    observed_norm_shapes: list[tuple[int, ...]] = []
    observed_raw_norms: list[float] = []
    observed_cholesky_shapes: list[tuple[int, ...]] = []
    original_norm = np.linalg.norm
    original_cholesky = np.linalg.cholesky

    def observed_norm(
        value: np.ndarray,
        *args: object,
        **kwargs: object,
    ) -> np.floating[object]:
        observed_ord = kwargs.get("ord", args[0] if args else None)
        assert observed_ord == 2
        observed_norm_shapes.append(tuple(int(item) for item in value.shape))
        result = original_norm(value, *args, **kwargs)
        observed_raw_norms.append(float(result))
        return result

    monkeypatch.setattr(np.linalg, "norm", observed_norm)

    def observed_cholesky(value: np.ndarray) -> np.ndarray:
        observed_cholesky_shapes.append(
            tuple(int(item) for item in value.shape)
        )
        return original_cholesky(value)

    monkeypatch.setattr(np.linalg, "cholesky", observed_cholesky)
    problem = make_h8_problem(
        horizon=2,
        channel_dimension=2,
        problem_seed=2026072121,
    )
    assert observed_norm_shapes == [(2, 2)] * 6 + [(4, 4)] * 2
    assert len(observed_raw_norms) == 4 * problem.layout.horizon
    assert observed_cholesky_shapes == (
        [(4, 4)] + [(2, 2)] * 4 + [(4, 4)] * 3
    )

    evidence = problem.problem_evidence
    local = evidence.local_spd_diagnostics
    norms = evidence.transition_norms
    assert type(evidence) is h8_types.H8ProductionProblemEvidence
    assert type(local) is h8_types.H8LocalSPDDiagnostics
    assert type(norms) is h8_types.H8TransitionNorms
    assert tuple(field.name for field in fields(type(evidence))) == (
        "generative_sha256",
        "recognition_sha256",
        "local_spd_diagnostics",
        "transition_norms",
        "observation_sha256",
    )
    assert tuple(field.name for field in fields(type(local))) == (
        "schema_version",
        "horizon",
        "generative_initial_min_pivot",
        "model_transition_min_pivots",
        "state_transition_min_pivots",
        "recognition_initial_min_pivot",
        "recognition_transition_min_pivots",
        "global_min_pivot",
    )
    assert tuple(field.name for field in fields(type(norms))) == (
        "schema_version",
        "horizon",
        "norm",
        "model_transition_norms",
        "state_transition_norms",
        "state_model_coupling_norms",
        "recognition_transition_norms",
        "max_model_transition_norm",
        "max_state_transition_norm",
        "max_state_model_coupling_norm",
        "max_recognition_transition_norm",
    )
    assert local.schema_version == "h8-local-spd-diagnostics-v1"
    assert norms.schema_version == "h8-transition-norms-v1"
    assert norms.norm == "operator_2"
    assert type(local.schema_version) is str
    assert type(norms.schema_version) is str
    assert type(norms.norm) is str
    assert local.horizon == norms.horizon == problem.layout.horizon == 2

    generative = h8_reference._generative_evidence_payload(problem)
    recognition = h8_reference._recognition_evidence_payload(problem)
    observations = h8_reference._observation_evidence_payload(problem)
    assert generative["domain"] == "vfe4.h8.generative-evidence.v1"
    assert generative["schema_version"] == "h8-generative-evidence-v1"
    assert recognition["domain"] == "vfe4.h8.recognition-evidence.v1"
    assert recognition["schema_version"] == "h8-recognition-evidence-v1"
    assert tuple(generative) == (
        "domain",
        "schema_version",
        "layout",
        "problem_seed",
        "vocabulary_size",
        "alpha",
        "initial",
        "model_transitions",
        "state_transitions",
        "emissions",
    )
    assert tuple(recognition) == (
        "domain",
        "schema_version",
        "layout",
        "problem_seed",
        "initial",
        "transitions",
    )
    assert tuple(generative["model_transitions"][0]) == (
        "receiver_t",
        "parent_t",
        "source_support",
        "matrix",
        "offset",
        "covariance",
    )
    assert tuple(generative["state_transitions"][0]) == (
        "receiver_t",
        "parent_t",
        "source_support",
        "state_matrix",
        "model_matrix",
        "offset",
        "covariance",
    )
    assert tuple(generative["emissions"][0]) == (
        "receiver_t",
        "weight",
        "bias",
        "observation",
    )
    assert tuple(recognition["transitions"][0]) == (
        "receiver_t",
        "parent_t",
        "source_support",
        "matrix",
        "offset",
        "covariance",
    )

    expected_alpha_leaf = {
        "shape": [3],
        "dtype": "<f8",
        "raw_sha256": hashlib.sha256(
            problem.alpha.tobytes(order="C")
        ).hexdigest(),
    }
    expected_recognition_mean_leaf = {
        "shape": [4],
        "dtype": "<f8",
        "raw_sha256": hashlib.sha256(
            problem.recognition.initial_mean.tobytes(order="C")
        ).hexdigest(),
    }
    assert generative["alpha"] == expected_alpha_leaf
    assert recognition["initial"]["mean"] == expected_recognition_mean_leaf
    assert observations == {
        "domain": "vfe4.h8.observations.v1",
        "records": [[1, 0], [2, 1]],
    }
    expected_observation_bytes = (
        b'{"domain":"vfe4.h8.observations.v1","records":[[1,0],[2,1]]}'
    )

    def canonical(payload: object) -> bytes:
        return json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")

    assert canonical(observations) == expected_observation_bytes
    assert evidence.generative_sha256 == hashlib.sha256(
        canonical(generative)
    ).hexdigest()
    assert evidence.recognition_sha256 == hashlib.sha256(
        canonical(recognition)
    ).hexdigest()
    assert evidence.observation_sha256 == hashlib.sha256(
        expected_observation_bytes
    ).hexdigest()
    assert len(observed_cholesky_shapes) == 3 * problem.layout.horizon + 2

    expected_model_pivots = tuple(
        float(np.min(np.diag(np.linalg.cholesky(item.covariance))))
        for item in problem.model_transitions
    )
    expected_state_pivots = tuple(
        float(np.min(np.diag(np.linalg.cholesky(item.covariance))))
        for item in problem.state_transitions
    )
    expected_recognition_pivots = tuple(
        float(np.min(np.diag(np.linalg.cholesky(item.covariance))))
        for item in problem.recognition.transitions
    )
    expected_initial_pivot = float(
        np.min(np.diag(np.linalg.cholesky(problem.initial_covariance)))
    )
    expected_recognition_initial_pivot = float(
        np.min(
            np.diag(
                np.linalg.cholesky(problem.recognition.initial_covariance)
            )
        )
    )
    assert local.generative_initial_min_pivot == expected_initial_pivot
    assert local.model_transition_min_pivots == expected_model_pivots
    assert local.state_transition_min_pivots == expected_state_pivots
    assert (
        local.recognition_initial_min_pivot
        == expected_recognition_initial_pivot
    )
    assert (
        local.recognition_transition_min_pivots
        == expected_recognition_pivots
    )
    assert local.global_min_pivot == min(
        expected_initial_pivot,
        *expected_model_pivots,
        *expected_state_pivots,
        expected_recognition_initial_pivot,
        *expected_recognition_pivots,
    )

    assert len(norms.model_transition_norms) == 2
    assert len(norms.state_transition_norms) == 2
    assert len(norms.state_model_coupling_norms) == 2
    assert len(norms.recognition_transition_norms) == 2
    assert norms.model_transition_norms == (
        min(0.35, observed_raw_norms[0]),
        min(0.35, observed_raw_norms[3]),
    )
    assert norms.state_transition_norms == (
        min(0.35, observed_raw_norms[1]),
        min(0.35, observed_raw_norms[4]),
    )
    assert norms.state_model_coupling_norms == (
        min(0.20, observed_raw_norms[2]),
        min(0.20, observed_raw_norms[5]),
    )
    assert norms.recognition_transition_norms == (
        min(0.35, observed_raw_norms[6]),
        min(0.35, observed_raw_norms[7]),
    )
    assert norms.max_model_transition_norm == max(
        norms.model_transition_norms
    )
    assert norms.max_state_transition_norm == max(
        norms.state_transition_norms
    )
    assert norms.max_state_model_coupling_norm == max(
        norms.state_model_coupling_norms
    )
    assert norms.max_recognition_transition_norm == max(
        norms.recognition_transition_norms
    )
    assert norms.max_model_transition_norm <= 0.35
    assert norms.max_state_transition_norm <= 0.35
    assert norms.max_state_model_coupling_norm <= 0.20
    assert norms.max_recognition_transition_norm <= 0.35

    assert validate_h8_problem(problem) is problem
    h8_reference._canonical_evidence_bytes(generative)
    h8_reference._canonical_evidence_bytes(recognition)
    h8_reference._canonical_evidence_bytes(observations)
    assert len(observed_norm_shapes) == 4 * problem.layout.horizon
    with pytest.raises(FrozenInstanceError):
        setattr(evidence, "generative_sha256", "0" * 64)

    class _StringSubclass(str):
        pass

    for record, changes, message in (
        (
            local,
            {
                "schema_version": _StringSubclass(
                    "h8-local-spd-diagnostics-v1"
                )
            },
            "schema",
        ),
        (
            norms,
            {
                "schema_version": _StringSubclass(
                    "h8-transition-norms-v1"
                )
            },
            "schema",
        ),
        (
            norms,
            {"norm": _StringSubclass("operator_2")},
            "operator_2",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            replace(record, **changes)

    original_evidence = evidence
    object.__setattr__(
        problem,
        "problem_evidence",
        replace(evidence, generative_sha256="0" * 64),
    )
    with pytest.raises(ValueError, match="problem evidence"):
        validate_h8_problem(problem)
    object.__setattr__(problem, "problem_evidence", original_evidence)

    shifted_local = replace(
        local,
        generative_initial_min_pivot=local.generative_initial_min_pivot + 1.0,
        model_transition_min_pivots=tuple(
            value + 1.0 for value in local.model_transition_min_pivots
        ),
        state_transition_min_pivots=tuple(
            value + 1.0 for value in local.state_transition_min_pivots
        ),
        recognition_initial_min_pivot=(
            local.recognition_initial_min_pivot + 1.0
        ),
        recognition_transition_min_pivots=tuple(
            value + 1.0
            for value in local.recognition_transition_min_pivots
        ),
        global_min_pivot=local.global_min_pivot + 1.0,
    )
    object.__setattr__(
        problem,
        "problem_evidence",
        replace(evidence, local_spd_diagnostics=shifted_local),
    )
    with pytest.raises(ValueError, match="problem evidence"):
        validate_h8_problem(problem)
    object.__setattr__(problem, "problem_evidence", original_evidence)

    with pytest.raises(ValueError, match="model_transition_norms"):
        replace(
            norms,
            model_transition_norms=norms.model_transition_norms[:-1],
        )
    with pytest.raises(ValueError, match="finite nonnegative"):
        replace(
            norms,
            state_transition_norms=(math.inf, *norms.state_transition_norms[1:]),
            max_state_transition_norm=math.inf,
        )
    with pytest.raises(ValueError, match="maximum"):
        replace(
            norms,
            max_recognition_transition_norm=(
                norms.max_recognition_transition_norm + 0.01
            ),
        )
    with pytest.raises(ValueError, match="contraction bound"):
        replace(
            norms,
            state_model_coupling_norms=(0.21, 0.21),
            max_state_model_coupling_norm=0.21,
        )
    assert len(observed_norm_shapes) == 4 * problem.layout.horizon
