"""Reusable literal 12-cell producer for bounded H8 correctness evidence."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
import torch
from torch import Tensor

from verification.h8_budget import compare_operands, literal_residual, make_operand_record
from verification.numpy_oracles.h8_dense import (
    NumpyDenseH8Result,
    evaluate_h8_numpy_dense,
)
from verification.torch_references.h8_dense import (
    TorchDenseH8Result,
    evaluate_h8_torch_dense,
)
from vfe4.generative.reference_h8 import (
    H8Problem,
    build_h8_generative,
    make_h8_problem,
)
from vfe4.objective.h8_sparse import evaluate_h8_sparse_objective
from vfe4.recognition.reference_h8 import build_h8_recognition
from vfe4.types.h8 import (
    H8AllowanceRecord,
    H8CorrectnessCell,
    H8CorrectnessControlResult,
    H8CorrectnessEndpointRecord,
    H8CorrectnessSourceResult,
    H8InvariantRecord,
    H8ObjectiveTerms,
    H8_CORRECTNESS_CASES,
    H8_CORRECTNESS_CONTROL_IDS,
    H8_CORRECTNESS_ORDERED_SOURCE_PAIRS,
    H8_CORRECTNESS_SOURCES,
    BlockTridiagonalPrecision,
    h8_correctness_endpoint_ids,
)
from vfe4.types.results import GateStatus


H8_LITERAL_CORRECTNESS_GRID = (
    (1, 1, 2026072111, 2026172111),
    (1, 2, 2026072112, 2026172112),
    (1, 4, 2026072114, 2026172114),
    (2, 1, 2026072121, 2026172121),
    (2, 2, 2026072122, 2026172122),
    (2, 4, 2026072124, 2026172124),
    (4, 1, 2026072141, 2026172141),
    (4, 2, 2026072142, 2026172142),
    (4, 4, 2026072144, 2026172144),
    (8, 1, 2026072181, 2026172181),
    (8, 2, 2026072182, 2026172182),
    (8, 4, 2026072184, 2026172184),
)


@dataclass(frozen=True, slots=True)
class _EndpointMetadata:
    shape: tuple[int, ...]
    scalar_count: int
    infinity_norm: float
    absolute_sum_bound: float
    local_operation_count: int
    solver_produced: bool
    quadrature_convergence: float


@dataclass(frozen=True, slots=True)
class _SourceResult:
    values: Mapping[str, object]
    operands: Mapping[str, object]
    off_diagonal_trace: float
    independent_sample: Tensor | None = None


def produce_h8_correctness_grid() -> tuple[H8CorrectnessCell, ...]:
    """Produce the exact literal grid without deriving a seed from dimensions."""

    if H8_LITERAL_CORRECTNESS_GRID != H8_CORRECTNESS_CASES:
        raise RuntimeError("literal H8 producer rows differ from the Task 1 registry")
    return tuple(
        _produce_cell(
            cell_id=cell_id,
            horizon=horizon,
            channel=channel,
            problem_seed=problem_seed,
            sample_noise_seed=sample_noise_seed,
        )
        for cell_id, (
            horizon,
            channel,
            problem_seed,
            sample_noise_seed,
        ) in enumerate(H8_LITERAL_CORRECTNESS_GRID, start=1)
    )


def _produce_cell(
    *,
    cell_id: int,
    horizon: int,
    channel: int,
    problem_seed: int,
    sample_noise_seed: int,
) -> H8CorrectnessCell:
    problem = make_h8_problem(
        horizon=horizon,
        channel_dimension=channel,
        problem_seed=problem_seed,
    )
    # The complete problem byte stream exists before this independent generator
    # is constructed.  It makes exactly one N*b draw and one C-order reshape.
    noise_generator = np.random.Generator(np.random.PCG64(sample_noise_seed))
    noise = np.ascontiguousarray(
        noise_generator.standard_normal(
            size=(problem.layout.population_size * problem.layout.block_size),
            dtype=np.float64,
        ).reshape(
            (problem.layout.population_size, problem.layout.block_size),
            order="C",
        ),
        dtype="<f8",
    )
    noise.setflags(write=False)
    raw_sources: Mapping[str, _SourceResult] = MappingProxyType(
        {
            "block": _evaluate_block_source(problem, noise, sample_noise_seed),
            "dense_torch": _adapt_torch(
                evaluate_h8_torch_dense(
                    problem,
                    torch.tensor(noise, dtype=torch.float64),
                )
            ),
            "numpy": _adapt_numpy(evaluate_h8_numpy_dense(problem, noise)),
        }
    )
    source_results = _retain_source_results(raw_sources, horizon)
    comparisons = _all_ordered_comparisons(source_results)
    controls = _wrong_path_controls(raw_sources, comparisons)
    invariants = _cell_invariants(
        source_results=source_results,
        comparisons=comparisons,
        controls=controls,
        horizon=horizon,
    )
    child_statuses = (
        *(comparison.status for comparison in comparisons),
        *(control.status for control in controls),
        *(invariant.status for invariant in invariants),
    )
    status = (
        GateStatus.FAIL
        if GateStatus.FAIL in child_statuses
        else GateStatus.INCONCLUSIVE
        if GateStatus.INCONCLUSIVE in child_statuses
        else GateStatus.PASS
    )
    return H8CorrectnessCell(
        cell_id=cell_id,
        layout=problem.layout,
        problem_seed=problem_seed,
        sample_noise_seed=sample_noise_seed,
        problem_sha256=problem.input_sha256,
        sample_noise_sha256=hashlib.sha256(noise.tobytes(order="C")).hexdigest(),
        source_results=source_results,
        pair_comparisons=comparisons,
        wrong_path_controls=controls,
        invariants=invariants,
        status=status,
        obligations=(
            ("one_or_more_correctness_decisions_remain_inconclusive",)
            if status is GateStatus.INCONCLUSIVE
            else ()
        ),
    )


def _retain_source_results(
    sources: Mapping[str, _SourceResult],
    horizon: int,
) -> tuple[H8CorrectnessSourceResult, ...]:
    expected_endpoints = h8_correctness_endpoint_ids(horizon)
    retained: list[H8CorrectnessSourceResult] = []
    for source in H8_CORRECTNESS_SOURCES:
        source_result = sources[source]
        if tuple(source_result.values) != expected_endpoints:
            raise ValueError(f"{source} raw endpoint inventory is incomplete or reordered")
        endpoints: list[H8CorrectnessEndpointRecord] = []
        for endpoint_id in expected_endpoints:
            raw_values = _scalars(source_result.values[endpoint_id])
            metadata = source_result.operands[endpoint_id]
            operand = make_operand_record(
                operand_id=f"{source}:{endpoint_id}",
                shape=metadata.shape,
                infinity_norm=max(abs(value) for value in raw_values),
                absolute_sum_bound=math.fsum(abs(value) for value in raw_values),
                local_operation_count=int(metadata.local_operation_count),
                source=source,
                solver_produced=bool(metadata.solver_produced),
                quadrature_convergence=float(metadata.quadrature_convergence),
                condition_provenance=(
                    "local_cholesky_pivot"
                    if metadata.solver_produced
                    else None
                ),
            )
            endpoints.append(
                H8CorrectnessEndpointRecord(
                    endpoint_id=endpoint_id,
                    raw_values=raw_values,
                    operand=operand,
                )
            )
        retained.append(
            H8CorrectnessSourceResult(
                source=source,
                endpoints=tuple(endpoints),
            )
        )
    return tuple(retained)


def _all_ordered_comparisons(
    source_results: tuple[H8CorrectnessSourceResult, ...],
) -> tuple[H8AllowanceRecord, ...]:
    lookup = {
        (source.source, endpoint.endpoint_id): endpoint
        for source in source_results
        for endpoint in source.endpoints
    }
    endpoint_ids = tuple(
        endpoint.endpoint_id for endpoint in source_results[0].endpoints
    )
    comparisons: list[H8AllowanceRecord] = []
    for endpoint_id in endpoint_ids:
        for left_source, right_source in H8_CORRECTNESS_ORDERED_SOURCE_PAIRS:
            left = lookup[(left_source, endpoint_id)]
            right = lookup[(right_source, endpoint_id)]
            comparisons.append(
                compare_operands(
                    comparison_id=(
                        f"{endpoint_id}:{left_source}->{right_source}"
                    ),
                    left=left.operand,
                    right=right.operand,
                    residual=literal_residual(
                        left.raw_values,
                        right.raw_values,
                    ),
                )
            )
    return tuple(comparisons)


def _wrong_path_controls(
    sources: Mapping[str, _SourceResult],
    comparisons: tuple[H8AllowanceRecord, ...],
) -> tuple[H8CorrectnessControlResult, ...]:
    block = sources["block"]
    correct = {
        record.comparison_id: record
        for record in comparisons
        if record.comparison_id.endswith(":block->dense_torch")
    }
    solve = torch.as_tensor(block.values["solve"]).clone()
    solve.reshape(-1)[0] += 1.0
    wrong_values = {
        "perturbed_solve_element": ("solve", solve),
        "reversed_logdet_sign": (
            "logdet",
            -float(block.values["logdet"]),
        ),
        "transposed_adjacent_covariance": (
            "selected_lower",
            torch.as_tensor(block.values["selected_lower"]).transpose(-1, -2),
        ),
        "duplicated_offdiagonal_trace": (
            "sparse_trace",
            float(block.values["sparse_trace"]) + block.off_diagonal_trace,
        ),
        "omitted_entropy": (
            "objective:complete_order21",
            float(block.values["objective:complete_order21"])
            - float(block.values["entropy"]),
        ),
        "independent_sample_noise": ("sample", block.independent_sample),
    }
    if tuple(wrong_values) != H8_CORRECTNESS_CONTROL_IDS:
        raise RuntimeError("wrong-path control order differs from its frozen registry")
    results: list[H8CorrectnessControlResult] = []
    for control_id, (endpoint_id, wrong_value) in wrong_values.items():
        comparison = correct[f"{endpoint_id}:block->dense_torch"]
        residual = literal_residual(
            _scalars(block.values[endpoint_id]),
            _scalars(wrong_value),
        )
        decisive = (
            comparison.decisive
            and math.isfinite(residual)
            and residual > comparison.allowance
        )
        results.append(
            H8CorrectnessControlResult(
                control_id=control_id,
                residual=residual,
                allowance=comparison.allowance,
                decisive=decisive,
                status=(
                    GateStatus.PASS if decisive else GateStatus.INCONCLUSIVE
                ),
                obligations=(
                    ()
                    if decisive
                    else (f"{control_id}_did_not_exceed_its_own_allowance",)
                ),
            )
        )
    return tuple(results)


def _cell_invariants(
    *,
    source_results: tuple[H8CorrectnessSourceResult, ...],
    comparisons: tuple[H8AllowanceRecord, ...],
    controls: tuple[H8CorrectnessControlResult, ...],
    horizon: int,
) -> tuple[H8InvariantRecord, ...]:
    endpoint_count = len(h8_correctness_endpoint_ids(horizon))
    expected_source_endpoints = len(H8_CORRECTNESS_SOURCES) * endpoint_count
    observed_source_endpoints = sum(
        len(source.endpoints) for source in source_results
    )
    expected_comparisons = (
        len(H8_CORRECTNESS_ORDERED_SOURCE_PAIRS) * endpoint_count
    )
    passed_comparisons = sum(
        comparison.status is GateStatus.PASS for comparison in comparisons
    )
    passed_controls = sum(control.status is GateStatus.PASS for control in controls)
    comparison_status = _aggregate_decision_status(
        tuple(comparison.status for comparison in comparisons)
    )
    control_status = _aggregate_decision_status(
        tuple(control.status for control in controls)
    )
    return (
        H8InvariantRecord(
            invariant_id="source_endpoint_inventory_complete",
            status=(
                GateStatus.PASS
                if observed_source_endpoints == expected_source_endpoints
                else GateStatus.FAIL
            ),
            value=observed_source_endpoints,
            limit=expected_source_endpoints,
            detail="retained raw endpoints across the exact three source records",
            obligations=(),
        ),
        H8InvariantRecord(
            invariant_id="six_direction_pair_inventory_complete",
            status=(
                GateStatus.PASS
                if len(comparisons) == expected_comparisons
                else GateStatus.FAIL
            ),
            value=len(comparisons),
            limit=expected_comparisons,
            detail="observed comparisons versus six directions per endpoint",
            obligations=(),
        ),
        H8InvariantRecord(
            invariant_id="all_pair_decisions_pass",
            status=comparison_status,
            value=passed_comparisons,
            limit=expected_comparisons,
            detail="decisive passing pair decisions versus the frozen inventory",
            obligations=_decision_obligations(
                comparison_status,
                "pair_decisions",
            ),
        ),
        H8InvariantRecord(
            invariant_id="six_wrong_path_controls_complete",
            status=(
                GateStatus.PASS
                if len(controls) == len(H8_CORRECTNESS_CONTROL_IDS)
                else GateStatus.FAIL
            ),
            value=len(controls),
            limit=len(H8_CORRECTNESS_CONTROL_IDS),
            detail="observed wrong-path controls versus the exact six controls",
            obligations=(),
        ),
        H8InvariantRecord(
            invariant_id="all_wrong_path_controls_decisive",
            status=control_status,
            value=passed_controls,
            limit=len(H8_CORRECTNESS_CONTROL_IDS),
            detail="controls whose perturbation exceeds its correct-path allowance",
            obligations=_decision_obligations(
                control_status,
                "wrong_path_controls",
            ),
        ),
    )


def _aggregate_decision_status(
    statuses: tuple[GateStatus, ...],
) -> GateStatus:
    if GateStatus.FAIL in statuses:
        return GateStatus.FAIL
    if GateStatus.INCONCLUSIVE in statuses:
        return GateStatus.INCONCLUSIVE
    return GateStatus.PASS


def _decision_obligations(
    status: GateStatus,
    label: str,
) -> tuple[str, ...]:
    return (
        (f"{label}_contain_an_inconclusive_decision",)
        if status is GateStatus.INCONCLUSIVE
        else ()
    )


def _evaluate_block_source(
    problem: H8Problem,
    noise: np.ndarray,
    sample_noise_seed: int,
) -> _SourceResult:
    recognition = build_h8_recognition(problem).gaussian
    factor = recognition.factor
    information = recognition.h
    forward = factor.solve_factor(information, transpose=False)
    backward = factor.solve_factor(forward, transpose=True)
    solved = factor.solve(information)
    dense_factor = _dense_factor(
        factor.diagonal_factor,
        factor.lower_factor,
    )
    reconstruction = dense_factor @ dense_factor.T
    noise_tensor = torch.tensor(noise, dtype=torch.float64)
    selected = factor.selected_inverse(problem.layout.stored_block_ids)
    selected_diagonal, selected_lower = selected._block_refs()
    generative_factor = build_h8_generative(problem).gaussian.factor
    generative_precision = _precision_from_factor(
        problem,
        generative_factor.diagonal_factor,
        generative_factor.lower_factor,
    )
    sparse_trace = float(
        factor.trace_inverse_product(generative_precision).item()
    )
    generative_diagonal, generative_lower = generative_precision._block_refs()
    off_diagonal_trace = float(
        (2.0 * torch.sum(generative_lower * selected_lower)).item()
    )
    objective = evaluate_h8_sparse_objective(problem, recognition)
    values: dict[str, object] = {
        "factor_reconstruction": reconstruction,
        "forward_substitution": forward,
        "backward_substitution": backward,
        "solve": solved,
        "logdet": float(factor.logdet().item()),
        "quadratic": float(factor.quadratic(noise_tensor).item()),
        "sample": recognition.sample(noise_tensor),
        "selected_diagonal": selected_diagonal,
        "selected_lower": selected_lower,
        "sparse_trace": sparse_trace,
        "entropy": float(recognition.entropy().item()),
        "log_normalizer": float(recognition.log_normalizer().item()),
    }
    values.update(_production_objective_values(objective))
    dimension = problem.layout.dimension
    operands = MappingProxyType(
        {
            endpoint_id: _metadata_from_value(
                value,
                operations=_operation_count(endpoint_id, dimension),
                solver=_solver_endpoint(endpoint_id),
                quadrature=_production_quadrature(endpoint_id, objective),
            )
            for endpoint_id, value in values.items()
        }
    )
    independent_generator = np.random.Generator(
        np.random.PCG64(88000000 + sample_noise_seed)
    )
    independent_noise = torch.tensor(
        independent_generator.standard_normal(
            size=(
                problem.layout.population_size,
                problem.layout.block_size,
            ),
            dtype=np.float64,
        ),
        dtype=torch.float64,
    )
    independent_sample = recognition.mean() + factor.sample(independent_noise)
    if generative_diagonal.shape != selected_diagonal.shape:
        raise RuntimeError("generative and recognition block inventories differ")
    return _SourceResult(
        values=MappingProxyType(values),
        operands=operands,
        off_diagonal_trace=off_diagonal_trace,
        independent_sample=independent_sample,
    )


def _dense_factor(diagonal: Tensor, lower: Tensor) -> Tensor:
    population, block, _ = diagonal.shape
    dense = torch.zeros(
        (population * block, population * block),
        dtype=torch.float64,
    )
    for index in range(population):
        target = slice(index * block, (index + 1) * block)
        dense[target, target] = diagonal[index]
        if index:
            parent = slice((index - 1) * block, index * block)
            dense[target, parent] = lower[index - 1]
    return dense


def _precision_from_factor(
    problem: H8Problem,
    diagonal_factor: Tensor,
    lower_factor: Tensor,
) -> BlockTridiagonalPrecision:
    diagonal: list[Tensor] = []
    lower: list[Tensor] = []
    for population in range(problem.layout.population_size):
        local = diagonal_factor[population] @ diagonal_factor[population].T
        if population:
            local = (
                local
                + lower_factor[population - 1] @ lower_factor[population - 1].T
            )
        diagonal.append(local)
        if population < problem.layout.horizon:
            lower.append(
                lower_factor[population] @ diagonal_factor[population].T
            )
    return BlockTridiagonalPrecision(
        problem.layout,
        torch.stack(diagonal),
        torch.stack(lower),
    )


def _production_objective_values(
    objective: H8ObjectiveTerms,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for term in (
        objective.initial_joint,
        *objective.model_transitions,
        *objective.state_transitions,
        *objective.emissions_order21,
        *objective.emissions_order17,
    ):
        result[f"objective:{term.factor_id}"] = term.value
    result.update(
        {
            "objective:recognition_entropy": objective.recognition_entropy,
            "objective:log_normalizer": objective.log_normalizer,
            "objective:model_source_kl": objective.model_source_kl,
            "objective:state_source_kl": objective.state_source_kl,
            "objective:source_entropy": objective.source_entropy,
            "objective:complete_order21": objective.complete_order21,
        }
    )
    return result


def _adapt_torch(result: TorchDenseH8Result) -> _SourceResult:
    values: dict[str, object] = {
        endpoint_id: getattr(result, endpoint_id)
        for endpoint_id in (
            "factor_reconstruction",
            "forward_substitution",
            "backward_substitution",
            "solve",
            "logdet",
            "quadratic",
            "sample",
            "selected_diagonal",
            "selected_lower",
            "sparse_trace",
            "entropy",
            "log_normalizer",
        )
    }
    values.update(_reference_objective_values(result.objective))
    return _SourceResult(
        values=MappingProxyType(values),
        operands=result.operands,
        off_diagonal_trace=0.0,
    )


def _adapt_numpy(result: NumpyDenseH8Result) -> _SourceResult:
    values: dict[str, object] = {
        endpoint_id: getattr(result, endpoint_id)
        for endpoint_id in (
            "factor_reconstruction",
            "forward_substitution",
            "backward_substitution",
            "solve",
            "logdet",
            "quadratic",
            "sample",
            "selected_diagonal",
            "selected_lower",
            "sparse_trace",
            "entropy",
            "log_normalizer",
        )
    }
    values.update(_reference_objective_values(result.objective))
    return _SourceResult(
        values=MappingProxyType(values),
        operands=result.operands,
        off_diagonal_trace=0.0,
    )


def _reference_objective_values(objective: object) -> dict[str, float]:
    values: dict[str, float] = {}
    for term in (
        objective.initial_joint,
        *objective.model_transitions,
        *objective.state_transitions,
        *objective.emissions_order21,
        *objective.emissions_order17,
    ):
        values[f"objective:{term.factor_id}"] = term.value
    values.update(
        {
            "objective:recognition_entropy": objective.recognition_entropy,
            "objective:log_normalizer": objective.log_normalizer,
            "objective:model_source_kl": objective.model_source_kl,
            "objective:state_source_kl": objective.state_source_kl,
            "objective:source_entropy": objective.source_entropy,
            "objective:complete_order21": objective.complete_order21,
        }
    )
    return values


def _scalars(value: object) -> tuple[float, ...]:
    if type(value) is Tensor:
        return tuple(float(item) for item in value.detach().reshape(-1).tolist())
    if isinstance(value, np.ndarray):
        return tuple(float(item) for item in value.reshape(-1).tolist())
    if isinstance(value, (int, float)) and type(value) is not bool:
        return (float(value),)
    raise ValueError("comparison endpoint must be an array, tensor, or scalar")


def _metadata_from_value(
    value: object,
    *,
    operations: int,
    solver: bool,
    quadrature: float,
) -> _EndpointMetadata:
    scalars = _scalars(value)
    if type(value) is Tensor:
        shape = tuple(int(item) for item in value.shape) or (1,)
    elif isinstance(value, np.ndarray):
        shape = tuple(int(item) for item in value.shape) or (1,)
    else:
        shape = (1,)
    return _EndpointMetadata(
        shape=shape,
        scalar_count=len(scalars),
        infinity_norm=max(abs(item) for item in scalars),
        absolute_sum_bound=math.fsum(abs(item) for item in scalars),
        local_operation_count=operations,
        solver_produced=solver,
        quadrature_convergence=quadrature,
    )


def _operation_count(endpoint: str, dimension: int) -> int:
    if endpoint == "logdet":
        return dimension + 1
    if endpoint in ("entropy", "log_normalizer"):
        return 2 * dimension + 5
    return max(1, 2 * dimension * dimension)


def _solver_endpoint(endpoint: str) -> bool:
    return endpoint in {
        "forward_substitution",
        "backward_substitution",
        "solve",
        "sample",
        "selected_diagonal",
        "selected_lower",
        "log_normalizer",
    }


def _production_quadrature(
    endpoint: str,
    objective: H8ObjectiveTerms,
) -> float:
    if endpoint == "objective:complete_order21":
        return float(objective.quadrature_absolute_difference)
    if endpoint.startswith(("objective:emission_order21:", "objective:emission_order17:")):
        receiver = int(endpoint.rsplit(":", maxsplit=1)[1])
        return abs(
            objective.emissions_order21[receiver - 1].value
            - objective.emissions_order17[receiver - 1].value
        )
    return 0.0


__all__ = [
    "H8_LITERAL_CORRECTNESS_GRID",
    "produce_h8_correctness_grid",
]
