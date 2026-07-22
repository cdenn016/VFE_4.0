"""Deterministic fresh-arm L-BFGS optimization for the frozen H3 gate."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn

from vfe4.generative.reference_h3 import H3GenerativeModel
from vfe4.objective.h3_gaussian import (
    evaluate_h3_elbo,
    evaluate_h3_elbo_difference,
)
from vfe4.recognition.reference_h3 import (
    H3VariationalGaussian,
    make_h3_parameters,
)
from vfe4.types.h3 import (
    H3ArmResult,
    H3DecisionConfig,
    H3InitializationConfig,
    H3Matrix4,
    H3OptimizationConfig,
    H3RecognitionFamily,
    H3Vector4,
)


class H3ClosureBudgetExhausted(RuntimeError):
    """Internal signal raised before a closure exceeds its frozen budget."""


class _NonfiniteClosureObjective(RuntimeError):
    pass


class _NonfiniteClosureGradient(RuntimeError):
    pass


class _NonfiniteAcceptedObjective(RuntimeError):
    pass


class _NonfiniteAcceptedGradient(RuntimeError):
    pass


class _NonfiniteTerminalLaw(RuntimeError):
    pass


@dataclass(frozen=True)
class _TerminalEvidence:
    elbo: float
    gradient_infinity_norm: float
    objective_change: float | None
    mean: H3Vector4
    precision_cholesky: H3Matrix4
    precision: H3Matrix4


def optimize_h3_arm(
    model: H3GenerativeModel,
    family: H3RecognitionFamily,
    initialization: H3InitializationConfig,
    config: H3OptimizationConfig,
) -> H3ArmResult:
    """Optimize one fresh H3 recognition arm under the frozen contract."""

    _validate_inputs(model, family, initialization, config)
    module = make_h3_parameters(family, initialization)
    parameters = tuple(module.parameters())
    _require_exact_common_initialization(module, parameters)
    optimizer = torch.optim.LBFGS(
        parameters,
        lr=config.learning_rate,
        max_iter=config.maximum_iterations_per_step,
        max_eval=config.maximum_evaluations_per_step,
        tolerance_grad=config.tolerance_gradient,
        tolerance_change=config.tolerance_change,
        history_size=config.history_size,
        line_search_fn=config.line_search,
    )

    closure_evaluations = 0
    accepted_records: list[dict[str, object]] = []
    terminal: _TerminalEvidence | None = None
    convergence_streak = 0
    closure_reference: H3VariationalGaussian | None = None

    def closure() -> Tensor:
        nonlocal closure_evaluations, closure_reference
        if closure_evaluations >= config.maximum_closure_evaluations:
            raise H3ClosureBudgetExhausted
        closure_evaluations += 1
        optimizer.zero_grad(set_to_none=True)
        try:
            q = module()
            if closure_reference is None:
                closure_reference = _detached_reference(q)
            loss = -evaluate_h3_elbo_difference(model, q, closure_reference)
        except (ValueError, RuntimeError) as exc:
            raise _NonfiniteClosureObjective from exc
        if not bool(torch.isfinite(loss)):
            raise _NonfiniteClosureObjective
        try:
            loss.backward()
        except RuntimeError as exc:
            raise _NonfiniteClosureGradient from exc
        if not _all_gradients_finite(parameters):
            raise _NonfiniteClosureGradient
        return loss

    for _ in range(config.maximum_accepted_iterations):
        closure_reference = None
        pre_step = _snapshot_parameters(parameters)
        try:
            optimizer.step(closure)
        except H3ClosureBudgetExhausted:
            _restore_parameters(parameters, pre_step)
            return _result(
                family,
                converged=False,
                failure_reason="closure_evaluation_budget_exhausted",
                closure_evaluations=closure_evaluations,
                accepted_records=accepted_records,
                terminal=terminal,
            )
        except _NonfiniteClosureObjective:
            _restore_parameters(parameters, pre_step)
            return _result(
                family,
                converged=False,
                failure_reason="nonfinite_closure_objective",
                closure_evaluations=closure_evaluations,
                accepted_records=accepted_records,
                terminal=terminal,
            )
        except _NonfiniteClosureGradient:
            _restore_parameters(parameters, pre_step)
            return _result(
                family,
                converged=False,
                failure_reason="nonfinite_closure_gradient",
                closure_evaluations=closure_evaluations,
                accepted_records=accepted_records,
                terminal=terminal,
            )
        except Exception:
            _restore_parameters(parameters, pre_step)
            return _result(
                family,
                converged=False,
                failure_reason="line_search_exception",
                closure_evaluations=closure_evaluations,
                accepted_records=accepted_records,
                terminal=terminal,
            )

        try:
            diagnostic = _accepted_diagnostic(
                model,
                module,
                optimizer,
                parameters,
                previous_elbo=(
                    cast(float, accepted_records[-1]["elbo"])
                    if accepted_records
                    else None
                ),
            )
        except _NonfiniteAcceptedObjective:
            _restore_parameters(parameters, pre_step)
            return _result(
                family,
                converged=False,
                failure_reason="nonfinite_accepted_objective",
                closure_evaluations=closure_evaluations,
                accepted_records=accepted_records,
                terminal=terminal,
            )
        except _NonfiniteAcceptedGradient:
            _restore_parameters(parameters, pre_step)
            return _result(
                family,
                converged=False,
                failure_reason="nonfinite_accepted_gradient",
                closure_evaluations=closure_evaluations,
                accepted_records=accepted_records,
                terminal=terminal,
            )
        except _NonfiniteTerminalLaw:
            _restore_parameters(parameters, pre_step)
            return _result(
                family,
                converged=False,
                failure_reason="nonfinite_terminal_law",
                closure_evaluations=closure_evaluations,
                accepted_records=accepted_records,
                terminal=terminal,
            )

        terminal = diagnostic
        accepted_iteration = len(accepted_records) + 1
        accepted_records.append(
            {
                "accepted_iteration": accepted_iteration,
                "elbo": diagnostic.elbo,
                "gradient_infinity_norm": diagnostic.gradient_infinity_norm,
                "objective_change": diagnostic.objective_change,
            }
        )

        if not _inside_terminal_envelope(terminal):
            return _result(
                family,
                converged=False,
                failure_reason="terminal_law_outside_envelope",
                closure_evaluations=closure_evaluations,
                accepted_records=accepted_records,
                terminal=terminal,
            )

        if (
            diagnostic.objective_change is not None
            and diagnostic.gradient_infinity_norm
            <= config.terminal_gradient_infinity_norm
            and diagnostic.objective_change <= config.terminal_objective_change
        ):
            convergence_streak += 1
        else:
            convergence_streak = 0
        if convergence_streak >= config.required_consecutive_accepted_iterations:
            return _result(
                family,
                converged=True,
                failure_reason=None,
                closure_evaluations=closure_evaluations,
                accepted_records=accepted_records,
                terminal=terminal,
            )

    if terminal is None:
        raise RuntimeError("H3 optimizer completed no accepted iterations")
    if not _inside_terminal_envelope(terminal):
        return _result(
            family,
            converged=False,
            failure_reason="terminal_law_outside_envelope",
            closure_evaluations=closure_evaluations,
            accepted_records=accepted_records,
            terminal=terminal,
        )
    return _result(
        family,
        converged=False,
        failure_reason="maximum_accepted_iterations_reached",
        closure_evaluations=closure_evaluations,
        accepted_records=accepted_records,
        terminal=terminal,
    )


def _validate_inputs(
    model: object,
    family: object,
    initialization: object,
    config: object,
) -> None:
    if not isinstance(model, H3GenerativeModel):
        raise ValueError("model must be an H3GenerativeModel")
    if family not in ("structured_full_spd", "fine_factorized_diagonal"):
        raise ValueError("family must be an H3 recognition family")
    if not isinstance(initialization, H3InitializationConfig):
        raise ValueError("initialization must be an H3InitializationConfig")
    if not isinstance(config, H3OptimizationConfig):
        raise ValueError("config must be an H3OptimizationConfig")


def _require_exact_common_initialization(
    module: nn.Module,
    parameters: tuple[nn.Parameter, ...],
) -> None:
    if not parameters:
        raise ValueError("H3 parameter module must register nonempty parameters")
    try:
        q = cast(H3VariationalGaussian, module())
        precision = q.precision()
    except (ValueError, RuntimeError) as exc:
        raise ValueError(
            "H3 parameter module must realize exact zero mean and identity precision"
        ) from exc
    expected_mean = torch.zeros(4, dtype=torch.float64, device="cpu")
    expected_precision = torch.eye(4, dtype=torch.float64, device="cpu")
    if (
        q.mean.dtype is not torch.float64
        or q.mean.device.type != "cpu"
        or precision.dtype is not torch.float64
        or precision.device.type != "cpu"
        or not torch.equal(q.mean, expected_mean)
        or not torch.equal(precision, expected_precision)
    ):
        raise ValueError(
            "H3 parameter module must realize exact zero mean and identity precision"
        )


def _accepted_diagnostic(
    model: H3GenerativeModel,
    module: nn.Module,
    optimizer: object,
    parameters: tuple[nn.Parameter, ...],
    *,
    previous_elbo: float | None,
) -> _TerminalEvidence:
    try:
        q = cast(H3VariationalGaussian, module())
    except (ValueError, RuntimeError) as exc:
        raise _NonfiniteTerminalLaw from exc
    try:
        evaluation = evaluate_h3_elbo(model, q)
        elbo_tensor = evaluation.elbo
    except (ValueError, RuntimeError) as exc:
        raise _NonfiniteAcceptedObjective from exc
    if not bool(torch.isfinite(elbo_tensor)):
        raise _NonfiniteAcceptedObjective
    cast(object, optimizer).zero_grad(set_to_none=True)  # type: ignore[attr-defined]
    try:
        (-elbo_tensor).backward()
    except RuntimeError as exc:
        raise _NonfiniteAcceptedGradient from exc
    if not _all_gradients_finite(parameters):
        raise _NonfiniteAcceptedGradient
    elbo = float(elbo_tensor.detach().item())
    gradient_infinity_norm = _gradient_infinity_norm(parameters)
    objective_change = (
        None if previous_elbo is None else abs(elbo - previous_elbo)
    )
    try:
        mean, cholesky, precision = _snapshot_law(q)
    except (ValueError, RuntimeError) as exc:
        raise _NonfiniteTerminalLaw from exc
    return _TerminalEvidence(
        elbo=elbo,
        gradient_infinity_norm=gradient_infinity_norm,
        objective_change=objective_change,
        mean=mean,
        precision_cholesky=cholesky,
        precision=precision,
    )


def _all_gradients_finite(parameters: tuple[nn.Parameter, ...]) -> bool:
    return all(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        for parameter in parameters
    )


def _gradient_infinity_norm(parameters: tuple[nn.Parameter, ...]) -> float:
    flattened = torch.cat(
        tuple(
            cast(Tensor, parameter.grad).detach().reshape(-1)
            for parameter in parameters
        )
    )
    return float(flattened.abs().max().item())


def _snapshot_parameters(
    parameters: tuple[nn.Parameter, ...],
) -> tuple[Tensor, ...]:
    return tuple(parameter.detach().clone() for parameter in parameters)


def _restore_parameters(
    parameters: tuple[nn.Parameter, ...], snapshot: tuple[Tensor, ...]
) -> None:
    with torch.no_grad():
        for parameter, value in zip(parameters, snapshot, strict=True):
            parameter.copy_(value)


def _snapshot_law(
    q: H3VariationalGaussian,
) -> tuple[H3Vector4, H3Matrix4, H3Matrix4]:
    with torch.no_grad():
        precision = q.precision()
        if not (
            bool(torch.isfinite(q.mean).all())
            and bool(torch.isfinite(q.precision_cholesky).all())
            and bool(torch.isfinite(precision).all())
        ):
            raise _NonfiniteTerminalLaw
        mean = cast(
            H3Vector4,
            tuple(float(value) for value in q.mean.detach().tolist()),
        )
        cholesky = _matrix_tuple(q.precision_cholesky)
        precision_tuple = _matrix_tuple(precision)
    return mean, cholesky, precision_tuple


def _detached_reference(q: H3VariationalGaussian) -> H3VariationalGaussian:
    return H3VariationalGaussian(
        family=q.family,
        mean=q.mean.detach().clone(),
        precision_cholesky=q.precision_cholesky.detach().clone(),
    )


def _matrix_tuple(value: Tensor) -> H3Matrix4:
    return cast(
        H3Matrix4,
        tuple(
            tuple(float(item) for item in row)
            for row in value.detach().tolist()
        ),
    )


def _inside_terminal_envelope(terminal: _TerminalEvidence) -> bool:
    decision = H3DecisionConfig()
    mean = torch.tensor(terminal.mean, dtype=torch.float64, device="cpu")
    precision = torch.tensor(
        terminal.precision, dtype=torch.float64, device="cpu"
    )
    eigenvalues = torch.linalg.eigvalsh(precision)
    if not bool(torch.isfinite(eigenvalues).all()):
        return False
    minimum = float(eigenvalues[0].item())
    maximum = float(eigenvalues[-1].item())
    condition = maximum / minimum if minimum > 0.0 else float("inf")
    mean_infinity_norm = float(mean.abs().max().item())
    return (
        minimum >= decision.minimum_precision_eigenvalue
        and maximum <= decision.maximum_precision_eigenvalue
        and condition <= decision.maximum_precision_condition_number
        and mean_infinity_norm <= decision.maximum_mean_infinity_norm
    )


def _trace_sha256(
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


def _result(
    family: H3RecognitionFamily,
    *,
    converged: bool,
    failure_reason: str | None,
    closure_evaluations: int,
    accepted_records: list[dict[str, object]],
    terminal: _TerminalEvidence | None,
) -> H3ArmResult:
    accepted_elbos = tuple(
        cast(float, record["elbo"]) for record in accepted_records
    )
    return H3ArmResult(
        family=family,
        converged=converged,
        failure_reason=failure_reason,
        accepted_iterations=len(accepted_records),
        closure_evaluations=closure_evaluations,
        terminal_elbo=None if terminal is None else terminal.elbo,
        terminal_gradient_infinity_norm=(
            None if terminal is None else terminal.gradient_infinity_norm
        ),
        terminal_objective_change=(
            None if terminal is None else terminal.objective_change
        ),
        terminal_mean=None if terminal is None else terminal.mean,
        terminal_precision_cholesky=(
            None if terminal is None else terminal.precision_cholesky
        ),
        terminal_precision=None if terminal is None else terminal.precision,
        accepted_elbos=accepted_elbos,
        canonical_trace_sha256=_trace_sha256(family, accepted_records),
    )


__all__ = ["optimize_h3_arm"]
