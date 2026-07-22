from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import vfe4.inference.h3_optimize as optimize_module
from vfe4.generative import H3GenerativeModel
from vfe4.inference import optimize_h3_arm
from vfe4.types.h3 import (
    H3ArmResult,
    H3InitializationConfig,
    H3OptimizationConfig,
    H3RecognitionFamily,
)
from vfe4.validation.h3_fixture import (
    H3_COUPLED_FIXTURE_PATH,
    H3_ZERO_CONTROL_FIXTURE_PATH,
    parse_h3_fixture_bytes,
)


FAMILIES: tuple[H3RecognitionFamily, ...] = (
    "structured_full_spd",
    "fine_factorized_diagonal",
)


def _model(fixture_id: str) -> H3GenerativeModel:
    path = (
        H3_COUPLED_FIXTURE_PATH
        if fixture_id == "h3-coupled-v1"
        else H3_ZERO_CONTROL_FIXTURE_PATH
    )
    fixture = parse_h3_fixture_bytes(
        path.read_bytes(), expected_fixture_id=fixture_id  # type: ignore[arg-type]
    )
    return H3GenerativeModel.from_fixture(fixture)


def _canonical_trace_digest(
    family: H3RecognitionFamily,
    accepted_records: list[dict[str, object]],
) -> str:
    payload = {
        "schema_version": 1,
        "family": family,
        "accepted": accepted_records,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _result_json(result: H3ArmResult) -> str:
    return json.dumps(
        asdict(result),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _zero_grad(parameters: tuple[torch.nn.Parameter, ...]) -> None:
    for parameter in parameters:
        parameter.grad = None


def _connected_elbo(q: Any, *, target: float = 1.0) -> torch.Tensor:
    graph_connection = 0.0 * (q.mean.sum() + q.precision_cholesky.sum())
    return -(q.mean[0] - target).square() + graph_connection


class _NaNGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, value: torch.Tensor) -> torch.Tensor:
        del ctx
        return value.new_zeros(())

    @staticmethod
    def backward(ctx: object, gradient: torch.Tensor) -> torch.Tensor:
        del ctx
        return torch.full_like(gradient, float("nan"))


def test_h3_arm_result_requires_trace_digest_and_count_consistency() -> None:
    kwargs: dict[str, object] = {
        "family": "structured_full_spd",
        "converged": False,
        "failure_reason": "line_search_exception",
        "accepted_iterations": 0,
        "closure_evaluations": 0,
        "terminal_elbo": None,
        "terminal_gradient_infinity_norm": None,
        "terminal_objective_change": None,
        "terminal_mean": None,
        "terminal_precision_cholesky": None,
        "terminal_precision": None,
        "accepted_elbos": (),
        "canonical_trace_sha256": "0" * 64,
    }
    assert H3ArmResult(**kwargs).canonical_trace_sha256 == "0" * 64  # type: ignore[arg-type]

    mismatch = dict(kwargs, accepted_iterations=1)
    with pytest.raises(ValueError, match="accepted_iterations"):
        H3ArmResult(**mismatch)  # type: ignore[arg-type]

    missing_digest = dict(kwargs, canonical_trace_sha256=None)
    with pytest.raises(ValueError, match="canonical_trace_sha256"):
        H3ArmResult(**missing_digest)  # type: ignore[arg-type]


def test_four_real_arms_are_fresh_exact_and_repeat_deterministically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_lbfgs = torch.optim.LBFGS
    constructions: list[
        tuple[
            tuple[torch.nn.Parameter, ...],
            tuple[torch.Tensor, ...],
            dict[str, object],
        ]
    ] = []

    class CountingLBFGS:
        def __init__(self, parameters: Any, **kwargs: object) -> None:
            owned = tuple(parameters)
            constructions.append(
                (
                    owned,
                    tuple(parameter.detach().clone() for parameter in owned),
                    dict(kwargs),
                )
            )
            self._inner = real_lbfgs(owned, **kwargs)

        def zero_grad(self, *, set_to_none: bool = True) -> None:
            self._inner.zero_grad(set_to_none=set_to_none)

        def step(self, closure: Any) -> torch.Tensor:
            return self._inner.step(closure)

    monkeypatch.setattr(torch.optim, "LBFGS", CountingLBFGS)
    initialization = H3InitializationConfig()
    config = H3OptimizationConfig()
    first: dict[tuple[str, H3RecognitionFamily], H3ArmResult] = {}
    for fixture_id in ("h3-coupled-v1", "h3-zero-control-v1"):
        model = _model(fixture_id)
        for family in FAMILIES:
            first[(fixture_id, family)] = optimize_h3_arm(
                model, family, initialization, config
            )

    assert len(constructions) == 4
    expected_kwargs = {
        "lr": 1.0,
        "max_iter": 1,
        "max_eval": 25,
        "tolerance_grad": 1.0e-12,
        "tolerance_change": 1.0e-18,
        "history_size": 20,
        "line_search_fn": "strong_wolfe",
    }
    all_id_sets: list[set[int]] = []
    assert config.tolerance_change == (
        config.terminal_gradient_infinity_norm**2 / 100.0
    )
    for owned_parameters, initial_values, kwargs in constructions:
        assert kwargs == expected_kwargs
        assert torch.equal(initial_values[0], torch.zeros(4, dtype=torch.float64))
        assert torch.equal(initial_values[1], torch.zeros(4, dtype=torch.float64))
        if len(initial_values) == 3:
            assert torch.equal(initial_values[2], torch.zeros(6, dtype=torch.float64))
        all_id_sets.append({id(parameter) for parameter in owned_parameters})
    assert all(
        left.isdisjoint(right)
        for index, left in enumerate(all_id_sets)
        for right in all_id_sets[index + 1 :]
    )

    monkeypatch.setattr(torch.optim, "LBFGS", real_lbfgs)
    second: dict[tuple[str, H3RecognitionFamily], H3ArmResult] = {}
    for fixture_id in ("h3-coupled-v1", "h3-zero-control-v1"):
        model = _model(fixture_id)
        for family in FAMILIES:
            second[(fixture_id, family)] = optimize_h3_arm(
                model, family, initialization, config
            )

    for construction_index, (key, result) in enumerate(first.items()):
        repeated = second[key]
        parameter_block_norms = tuple(
            float(parameter.grad.detach().abs().max().item())
            for parameter in constructions[construction_index][0]
            if parameter.grad is not None
        )
        assert result.converged, (
            key,
            result.terminal_gradient_infinity_norm,
            result.terminal_objective_change,
            result.closure_evaluations,
            result.accepted_elbos[-8:],
            parameter_block_norms,
        )
        assert result.failure_reason is None
        assert result.accepted_iterations == len(result.accepted_elbos)
        assert result.accepted_iterations >= 4
        assert result.closure_evaluations <= 5_000
        assert result.terminal_gradient_infinity_norm is not None
        assert result.terminal_gradient_infinity_norm <= 1.0e-8
        assert result.terminal_objective_change is not None
        assert result.terminal_objective_change <= 1.0e-12
        assert _result_json(result) == _result_json(repeated)


def test_first_accepted_point_cannot_count_and_trace_schema_is_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_difference = optimize_module.evaluate_h3_elbo_difference
    reference_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []

    def observe_difference(
        model: H3GenerativeModel,
        q: Any,
        reference: Any,
    ) -> torch.Tensor:
        reference_pairs.append(
            (q.mean.detach().clone(), reference.mean.detach().clone())
        )
        return real_difference(model, q, reference)

    class SetTargetLBFGS:
        def __init__(self, parameters: Any, **_kwargs: object) -> None:
            self.parameters = tuple(parameters)

        def zero_grad(self, *, set_to_none: bool = True) -> None:
            assert set_to_none
            _zero_grad(self.parameters)

        def step(self, closure: Any) -> torch.Tensor:
            loss = closure()
            with torch.no_grad():
                self.parameters[0][0] = 1.0
            return loss

    monkeypatch.setattr(torch.optim, "LBFGS", SetTargetLBFGS)
    monkeypatch.setattr(
        optimize_module, "evaluate_h3_elbo_difference", observe_difference
    )
    monkeypatch.setattr(
        optimize_module,
        "evaluate_h3_elbo",
        lambda _model, q: SimpleNamespace(elbo=_connected_elbo(q)),
    )
    result = optimize_h3_arm(
        _model("h3-coupled-v1"),
        "structured_full_spd",
        H3InitializationConfig(),
        H3OptimizationConfig(),
    )

    records = [
        {
            "accepted_iteration": index,
            "elbo": 0.0,
            "gradient_infinity_norm": 0.0,
            "objective_change": None if index == 1 else 0.0,
        }
        for index in range(1, 5)
    ]
    assert result.converged
    assert result.accepted_iterations == 4
    assert result.closure_evaluations == 4
    assert len(reference_pairs) == 4
    assert all(torch.equal(q_mean, reference_mean) for q_mean, reference_mean in reference_pairs)
    assert result.accepted_elbos == (0.0, 0.0, 0.0, 0.0)
    assert result.canonical_trace_sha256 == _canonical_trace_digest(
        "structured_full_spd", records
    )


def test_closure_budget_is_local_transactional_and_checked_before_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objective_calls = 0
    created_modules: list[torch.nn.Module] = []
    real_factory = optimize_module.make_h3_parameters

    def capture_factory(
        family: H3RecognitionFamily,
        initialization: H3InitializationConfig,
    ) -> torch.nn.Module:
        module = real_factory(family, initialization)
        created_modules.append(module)
        return module

    def evaluate(
        _model: object, q: Any, _reference: Any
    ) -> torch.Tensor:
        nonlocal objective_calls
        objective_calls += 1
        return _connected_elbo(q)

    class ExhaustingLBFGS:
        zero_grad_calls = 0

        def __init__(self, parameters: Any, **_kwargs: object) -> None:
            self.parameters = tuple(parameters)

        def zero_grad(self, *, set_to_none: bool = True) -> None:
            assert set_to_none
            type(self).zero_grad_calls += 1
            _zero_grad(self.parameters)

        def step(self, closure: Any) -> torch.Tensor:
            with torch.no_grad():
                self.parameters[0][0] = 9.0
            loss = torch.zeros((), dtype=torch.float64)
            for _ in range(5_001):
                loss = closure()
            return loss

    monkeypatch.setattr(torch.optim, "LBFGS", ExhaustingLBFGS)
    monkeypatch.setattr(optimize_module, "make_h3_parameters", capture_factory)
    monkeypatch.setattr(optimize_module, "evaluate_h3_elbo_difference", evaluate)
    result = optimize_h3_arm(
        _model("h3-coupled-v1"),
        "structured_full_spd",
        H3InitializationConfig(),
        H3OptimizationConfig(),
    )

    assert not result.converged
    assert result.failure_reason == "closure_evaluation_budget_exhausted"
    assert result.closure_evaluations == 5_000
    assert objective_calls == 5_000
    assert ExhaustingLBFGS.zero_grad_calls == 5_000
    assert result.accepted_iterations == 0
    assert result.accepted_elbos == ()
    assert result.terminal_mean is None
    assert len(created_modules) == 1
    restored = created_modules[0]()
    assert torch.equal(restored.mean, torch.zeros(4, dtype=torch.float64))
    assert result.canonical_trace_sha256 == _canonical_trace_digest(
        "structured_full_spd", []
    )


@pytest.mark.parametrize(
    ("mode", "failure_reason"),
    (
        ("objective", "nonfinite_closure_objective"),
        ("gradient", "nonfinite_closure_gradient"),
    ),
)
def test_nonfinite_closure_failures_are_typed_and_transactional(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    failure_reason: str,
) -> None:
    class OneClosureLBFGS:
        def __init__(self, parameters: Any, **_kwargs: object) -> None:
            self.parameters = tuple(parameters)

        def zero_grad(self, *, set_to_none: bool = True) -> None:
            assert set_to_none
            _zero_grad(self.parameters)

        def step(self, closure: Any) -> torch.Tensor:
            with torch.no_grad():
                self.parameters[0][0] = 9.0
            return closure()

    def evaluate(
        _model: object, q: Any, _reference: Any
    ) -> torch.Tensor:
        connected = q.mean.sum() + q.precision_cholesky.sum()
        if mode == "objective":
            elbo = connected * 0.0 + torch.tensor(
                float("nan"), dtype=torch.float64
            )
        else:
            elbo = _NaNGradient.apply(connected)
        return elbo

    monkeypatch.setattr(torch.optim, "LBFGS", OneClosureLBFGS)
    monkeypatch.setattr(optimize_module, "evaluate_h3_elbo_difference", evaluate)
    result = optimize_h3_arm(
        _model("h3-coupled-v1"),
        "structured_full_spd",
        H3InitializationConfig(),
        H3OptimizationConfig(),
    )
    assert result.failure_reason == failure_reason
    assert not result.converged
    assert result.closure_evaluations == 1
    assert result.accepted_iterations == 0
    assert result.terminal_mean is None


def test_line_search_exception_restores_the_last_accepted_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class AcceptThenRaiseLBFGS:
        def __init__(self, parameters: Any, **_kwargs: object) -> None:
            self.parameters = tuple(parameters)
            self.steps = 0

        def zero_grad(self, *, set_to_none: bool = True) -> None:
            assert set_to_none
            _zero_grad(self.parameters)

        def step(self, closure: Any) -> torch.Tensor:
            self.steps += 1
            loss = closure()
            with torch.no_grad():
                self.parameters[0][0] = 1.0 if self.steps == 1 else 9.0
            if self.steps == 2:
                raise RuntimeError("forced line-search failure")
            return loss

    monkeypatch.setattr(torch.optim, "LBFGS", AcceptThenRaiseLBFGS)
    monkeypatch.setattr(
        optimize_module,
        "evaluate_h3_elbo",
        lambda _model, q: SimpleNamespace(elbo=_connected_elbo(q)),
    )
    result = optimize_h3_arm(
        _model("h3-coupled-v1"),
        "structured_full_spd",
        H3InitializationConfig(),
        H3OptimizationConfig(),
    )
    assert result.failure_reason == "line_search_exception"
    assert result.accepted_iterations == 1
    assert result.accepted_elbos == (-0.0,)
    assert result.terminal_mean == (1.0, 0.0, 0.0, 0.0)
    assert result.terminal_objective_change is None


@pytest.mark.parametrize(
    ("mode", "failure_reason"),
    (
        ("objective", "nonfinite_accepted_objective"),
        ("gradient", "nonfinite_accepted_gradient"),
    ),
)
def test_nonfinite_accepted_diagnostics_abort_the_step(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    failure_reason: str,
) -> None:
    class SetTargetLBFGS:
        def __init__(self, parameters: Any, **_kwargs: object) -> None:
            self.parameters = tuple(parameters)

        def zero_grad(self, *, set_to_none: bool = True) -> None:
            assert set_to_none
            _zero_grad(self.parameters)

        def step(self, closure: Any) -> torch.Tensor:
            loss = closure()
            with torch.no_grad():
                self.parameters[0][0] = 1.0
            return loss

    def evaluate(_model: object, q: Any) -> SimpleNamespace:
        connected = q.mean.sum() + q.precision_cholesky.sum()
        if mode == "objective":
            elbo = connected * 0.0 + torch.tensor(
                float("nan"), dtype=torch.float64
            )
        else:
            elbo = _NaNGradient.apply(connected)
        return SimpleNamespace(elbo=elbo)

    monkeypatch.setattr(torch.optim, "LBFGS", SetTargetLBFGS)
    monkeypatch.setattr(optimize_module, "evaluate_h3_elbo", evaluate)
    result = optimize_h3_arm(
        _model("h3-coupled-v1"),
        "structured_full_spd",
        H3InitializationConfig(),
        H3OptimizationConfig(),
    )
    assert result.failure_reason == failure_reason
    assert result.accepted_iterations == 0
    assert result.accepted_elbos == ()
    assert result.terminal_mean is None


def test_nonfinite_terminal_law_aborts_before_acceptance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InfiniteLawLBFGS:
        def __init__(self, parameters: Any, **_kwargs: object) -> None:
            self.parameters = tuple(parameters)

        def zero_grad(self, *, set_to_none: bool = True) -> None:
            assert set_to_none
            _zero_grad(self.parameters)

        def step(self, closure: Any) -> torch.Tensor:
            loss = closure()
            with torch.no_grad():
                self.parameters[1][0] = float("inf")
            return loss

    monkeypatch.setattr(torch.optim, "LBFGS", InfiniteLawLBFGS)
    result = optimize_h3_arm(
        _model("h3-coupled-v1"),
        "structured_full_spd",
        H3InitializationConfig(),
        H3OptimizationConfig(),
    )
    assert result.failure_reason == "nonfinite_terminal_law"
    assert result.accepted_iterations == 0
    assert result.terminal_mean is None


def test_terminal_envelope_is_checked_without_changing_the_law(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SetOutsideEnvelopeLBFGS:
        def __init__(self, parameters: Any, **_kwargs: object) -> None:
            self.parameters = tuple(parameters)

        def zero_grad(self, *, set_to_none: bool = True) -> None:
            assert set_to_none
            _zero_grad(self.parameters)

        def step(self, closure: Any) -> torch.Tensor:
            loss = closure()
            with torch.no_grad():
                self.parameters[0][0] = 5.0
            return loss

    monkeypatch.setattr(torch.optim, "LBFGS", SetOutsideEnvelopeLBFGS)
    monkeypatch.setattr(
        optimize_module,
        "evaluate_h3_elbo",
        lambda _model, q: SimpleNamespace(elbo=_connected_elbo(q, target=5.0)),
    )
    result = optimize_h3_arm(
        _model("h3-coupled-v1"),
        "structured_full_spd",
        H3InitializationConfig(),
        H3OptimizationConfig(),
    )
    assert not result.converged
    assert result.failure_reason == "terminal_law_outside_envelope"
    assert result.accepted_iterations == 1
    assert result.terminal_mean == (5.0, 0.0, 0.0, 0.0)


def test_maximum_accepted_iterations_is_explicit_and_finite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NoOpLBFGS:
        def __init__(self, parameters: Any, **_kwargs: object) -> None:
            self.parameters = tuple(parameters)

        def zero_grad(self, *, set_to_none: bool = True) -> None:
            assert set_to_none
            _zero_grad(self.parameters)

        def step(self, closure: Any) -> torch.Tensor:
            return closure()

    monkeypatch.setattr(torch.optim, "LBFGS", NoOpLBFGS)
    result = optimize_h3_arm(
        _model("h3-coupled-v1"),
        "structured_full_spd",
        H3InitializationConfig(),
        H3OptimizationConfig(),
    )
    assert not result.converged
    assert result.failure_reason == "maximum_accepted_iterations_reached"
    assert result.accepted_iterations == 200
    assert len(result.accepted_elbos) == 200
    assert result.terminal_elbo is not None
    assert result.terminal_mean == (0.0, 0.0, 0.0, 0.0)


def test_optimize_h3_arm_rejects_invalid_public_inputs() -> None:
    model = _model("h3-coupled-v1")
    initialization = H3InitializationConfig()
    config = H3OptimizationConfig()
    with pytest.raises(ValueError, match="model"):
        optimize_h3_arm(object(), "structured_full_spd", initialization, config)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="family"):
        optimize_h3_arm(model, "unknown", initialization, config)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="initialization"):
        optimize_h3_arm(model, "structured_full_spd", object(), config)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="config"):
        optimize_h3_arm(model, "structured_full_spd", initialization, object())  # type: ignore[arg-type]


def test_altered_factory_initialization_is_rejected_before_lbfgs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_factory = optimize_module.make_h3_parameters
    optimizer_constructed = False

    def altered_factory(
        family: H3RecognitionFamily,
        initialization: H3InitializationConfig,
    ) -> torch.nn.Module:
        module = real_factory(family, initialization)
        with torch.no_grad():
            module.mean[0] = 1.0  # type: ignore[attr-defined]
        return module

    class ForbiddenLBFGS:
        def __init__(self, _parameters: Any, **_kwargs: object) -> None:
            nonlocal optimizer_constructed
            optimizer_constructed = True

    monkeypatch.setattr(optimize_module, "make_h3_parameters", altered_factory)
    monkeypatch.setattr(torch.optim, "LBFGS", ForbiddenLBFGS)
    with pytest.raises(ValueError, match="exact zero mean and identity precision"):
        optimize_h3_arm(
            _model("h3-coupled-v1"),
            "structured_full_spd",
            H3InitializationConfig(),
            H3OptimizationConfig(),
        )
    assert not optimizer_constructed
