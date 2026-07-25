"""Source-cohesive H8 budget, parser, and literal-grid verification."""

from __future__ import annotations

import ast
import hashlib
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from verification.h8_budget import (
    compare_operands,
    gamma,
    make_operand_record,
    operand_components,
    reduction_component,
)
from verification.h8_correctness import (
    H8_LITERAL_CORRECTNESS_GRID,
    produce_h8_correctness_grid,
)
from verification.numpy_oracles.h8_dense import evaluate_h8_numpy_dense
from verification.torch_references.h8_dense import evaluate_h8_torch_dense
from vfe4.generative.reference_h8 import (
    H8Problem,
    h8_sample_noise,
    make_h8_problem,
)
from vfe4.types.h8 import (
    H8AllowanceRecord,
    H8_CORRECTNESS_CONTROL_IDS,
    H8_CORRECTNESS_ORDERED_SOURCE_PAIRS,
    H8_CORRECTNESS_SOURCES,
    h8_correctness_endpoint_ids,
)
from vfe4.types.results import GateStatus


def test_h8_budget_literal_formulas_and_boundaries() -> None:
    assert gamma(1) == math.ulp(1.0) / (1.0 - math.ulp(1.0))
    with pytest.raises(ValueError):
        gamma(True)
    with pytest.raises(ValueError):
        gamma(0)
    with pytest.raises(ValueError):
        gamma(math.ceil(1.0 / math.ulp(1.0)))

    left = make_operand_record(
        operand_id="left:unequal-shape",
        shape=(2, 3),
        infinity_norm=7.0,
        absolute_sum_bound=19.0,
        local_operation_count=11,
        source="block",
        solver_produced=True,
        quadrature_convergence=3e-12,
        condition_provenance="local_cholesky_pivot",
    )
    right = make_operand_record(
        operand_id="right:unequal-shape",
        shape=(6,),
        infinity_norm=5.0,
        absolute_sum_bound=17.0,
        local_operation_count=13,
        source="numpy",
        solver_produced=False,
        quadrature_convergence=7e-12,
    )
    left_components = operand_components(left)
    right_components = operand_components(right)
    assert left_components[1] == 7e-9
    assert right_components[1] == 0.0
    expected_reduction = (
        4096.0 * gamma(7) * max(1.0, left.infinity_norm, right.infinity_norm)
    )
    assert (
        reduction_component(left, right, compared_scalar_count=6)
        == expected_reduction
    )
    record = compare_operands(
        comparison_id="left->right",
        left=left,
        right=right,
        residual=0.0,
    )
    assert record.allowance == math.fsum(
        (*left_components, *right_components, expected_reduction)
    )
    assert record.left_solver_component == 7e-9
    assert record.right_solver_component == 0.0

    with pytest.raises(ValueError):
        make_operand_record(
            operand_id="global-condition",
            shape=(1,),
            infinity_norm=1.0,
            absolute_sum_bound=1.0,
            local_operation_count=1,
            source="numpy",
            solver_produced=False,
            condition_provenance="global_kappa_1_estimate",
        )
    with pytest.raises(ValueError):
        compare_operands(
            comparison_id="mismatched-count",
            left=left,
            right=make_operand_record(
                operand_id="wrong-count",
                shape=(5,),
                infinity_norm=1.0,
                absolute_sum_bound=1.0,
                local_operation_count=1,
                source="numpy",
                solver_produced=False,
            ),
            residual=0.0,
        )


def test_h8_budget_strict_decisiveness_and_inclusive_residual() -> None:
    below = _fraction_boundary_record(math.nextafter(1e-4, 0.0), 0.0)
    equal = _fraction_boundary_record(1e-4, 0.0)
    above = _fraction_boundary_record(math.nextafter(1e-4, math.inf), 0.0)
    assert below.decisive and below.status is GateStatus.PASS
    assert not equal.decisive and equal.status is GateStatus.INCONCLUSIVE
    assert not above.decisive and above.status is GateStatus.INCONCLUSIVE

    allowance = below.allowance
    under = _fraction_boundary_record(
        math.nextafter(1e-4, 0.0),
        math.nextafter(allowance, 0.0),
    )
    on = _fraction_boundary_record(
        math.nextafter(1e-4, 0.0),
        allowance,
    )
    over = _fraction_boundary_record(
        math.nextafter(1e-4, 0.0),
        math.nextafter(allowance, math.inf),
    )
    assert under.status is GateStatus.PASS
    assert on.status is GateStatus.PASS
    assert over.status is GateStatus.FAIL


def test_h8_references_reject_bounds_before_dense_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid = make_h8_problem(
        horizon=1,
        channel_dimension=1,
        problem_seed=2026072111,
    )
    carriers = (
        _replace_chunk(valid, 5, b"integer:9"),
        _replace_chunk(valid, 6, b"integer:5"),
        _replace_chunk(valid, 7, b"integer:2"),
    )

    def forbidden_numpy_allocation(*args: object, **kwargs: object) -> object:
        raise AssertionError("NumPy allocation occurred before the hard bound")

    monkeypatch.setattr(
        "verification.numpy_oracles.h8_dense.np.zeros",
        forbidden_numpy_allocation,
    )
    for carrier in carriers:
        with pytest.raises(ValueError):
            evaluate_h8_numpy_dense(carrier, np.empty((0,), dtype="<f8"))
    monkeypatch.undo()

    def forbidden_torch_allocation(*args: object, **kwargs: object) -> object:
        raise AssertionError("PyTorch allocation occurred before the hard bound")

    monkeypatch.setattr(
        "verification.torch_references.h8_dense.torch.zeros",
        forbidden_torch_allocation,
    )
    for carrier in carriers:
        with pytest.raises(ValueError):
            evaluate_h8_torch_dense(
                carrier,
                torch.empty((0,), dtype=torch.float64),
            )


def test_h8_dense_parsers_reject_noncanonical_preimages_and_rehashed_tampering() -> None:
    problem = make_h8_problem(
        horizon=1,
        channel_dimension=1,
        problem_seed=2026072111,
    )
    noise = h8_sample_noise(problem, sample_noise_seed=2026172111)
    noncanonical = (
        _replace_chunk(problem, 5, b"integer:01"),
        _replace_chunk(problem, 4, b"tuple:03"),
        _replace_chunk(problem, 10, b"array:(3, )"),
    )
    for carrier in noncanonical:
        with pytest.raises(ValueError, match="canonical|marker"):
            evaluate_h8_numpy_dense(carrier, noise)
        with pytest.raises(ValueError, match="canonical|marker"):
            evaluate_h8_torch_dense(
                carrier,
                torch.tensor(noise, dtype=torch.float64),
            )

    chunks = _chunks(problem.serialized_bytes)
    changed_initial_mean = bytearray(chunks[13])
    changed_initial_mean[0] ^= 1
    chunks[13] = bytes(changed_initial_mean)
    tampered_bytes = _frame(chunks)
    tampered = _SemanticCarrier(
        original=problem,
        serialized_bytes=tampered_bytes,
        input_sha256=hashlib.sha256(tampered_bytes).hexdigest(),
    )
    with pytest.raises(ValueError, match="carrier bytes|preimage"):
        evaluate_h8_numpy_dense(tampered, noise)
    with pytest.raises(ValueError, match="carrier bytes|preimage"):
        evaluate_h8_torch_dense(
            tampered,
            torch.tensor(noise, dtype=torch.float64),
        )


def test_h8_dense_references_do_not_call_block_or_objective_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = make_h8_problem(
        horizon=1,
        channel_dimension=1,
        problem_seed=2026072111,
    )
    noise = h8_sample_noise(problem, sample_noise_seed=2026172111)

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("dense reference called a production H8 helper")

    monkeypatch.setattr(
        "vfe4.generative.reference_h8.validate_h8_problem",
        forbidden,
    )
    monkeypatch.setattr(
        "vfe4.numerics.block_canonical.BlockCanonicalAssembler",
        forbidden,
    )
    monkeypatch.setattr(
        "vfe4.numerics.block_tridiagonal.BlockTridiagonalCholesky",
        forbidden,
    )
    monkeypatch.setattr(
        "vfe4.objective.h8_sparse.evaluate_h8_sparse_objective",
        forbidden,
    )
    numpy_result = evaluate_h8_numpy_dense(problem, noise)
    torch_result = evaluate_h8_torch_dense(
        problem,
        torch.tensor(noise, dtype=torch.float64),
    )
    assert numpy_result.input_sha256 == problem.input_sha256
    assert torch_result.input_sha256 == problem.input_sha256


def test_production_modules_cannot_import_verification_dense_references() -> None:
    root = Path(__file__).resolve().parents[2]
    for source_path in sorted((root / "vfe4").rglob("*.py")):
        tree = ast.parse(
            source_path.read_text(encoding="utf-8"),
            source_path.as_posix(),
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported = (node.module or "",)
            else:
                continue
            assert all(
                name != "verification" and not name.startswith("verification.")
                for name in imported
            ), f"{source_path} imports verification code"


def test_h8_literal_three_way_correctness_grid() -> None:
    cells = produce_h8_correctness_grid()
    assert len(cells) == len(H8_LITERAL_CORRECTNESS_GRID) == 12
    assert tuple(
        (
            cell.layout.horizon,
            cell.layout.d_z,
            cell.problem_seed,
            cell.sample_noise_seed,
        )
        for cell in cells
    ) == H8_LITERAL_CORRECTNESS_GRID
    for cell in cells:
        endpoint_ids = h8_correctness_endpoint_ids(cell.layout.horizon)
        assert tuple(result.source for result in cell.source_results) == (
            H8_CORRECTNESS_SOURCES
        )
        assert all(
            tuple(endpoint.endpoint_id for endpoint in result.endpoints)
            == endpoint_ids
            for result in cell.source_results
        )
        assert len(cell.pair_comparisons) == (
            len(endpoint_ids) * len(H8_CORRECTNESS_ORDERED_SOURCE_PAIRS)
        )
        assert tuple(
            control.control_id for control in cell.wrong_path_controls
        ) == H8_CORRECTNESS_CONTROL_IDS
        invariant_by_id = {
            invariant.invariant_id: invariant for invariant in cell.invariants
        }
        assert (
            invariant_by_id["all_pair_decisions_pass"].value
            == invariant_by_id["all_pair_decisions_pass"].limit
        )
        assert (
            invariant_by_id["all_wrong_path_controls_decisive"].value
            == invariant_by_id["all_wrong_path_controls_decisive"].limit
        )
        assert cell.status is GateStatus.PASS
    first = cells[0]
    with pytest.raises(ValueError, match="block, dense_torch, and numpy"):
        replace(first, source_results=tuple(reversed(first.source_results)))
    truncated_source = replace(
        first.source_results[0],
        endpoints=first.source_results[0].endpoints[:-1],
    )
    with pytest.raises(ValueError, match="endpoint inventory"):
        replace(
            first,
            source_results=(truncated_source, *first.source_results[1:]),
        )
    with pytest.raises(ValueError, match="six directions"):
        replace(first, pair_comparisons=first.pair_comparisons[:-1])
    with pytest.raises(ValueError, match="six-control inventory"):
        replace(first, wrong_path_controls=first.wrong_path_controls[:-1])


def _fraction_boundary_record(
    desired_fraction: float,
    residual: float,
) -> H8AllowanceRecord:
    left = make_operand_record(
        operand_id="boundary:left",
        shape=(1,),
        infinity_norm=1.0,
        absolute_sum_bound=1.0,
        local_operation_count=1,
        source="block",
        solver_produced=False,
    )
    right_without_quadrature = make_operand_record(
        operand_id="boundary:right",
        shape=(1,),
        infinity_norm=1.0,
        absolute_sum_bound=1.0,
        local_operation_count=1,
        source="numpy",
        solver_produced=False,
    )
    base = math.fsum(
        (
            *operand_components(left),
            *operand_components(right_without_quadrature),
            reduction_component(
                left,
                right_without_quadrature,
                compared_scalar_count=1,
            ),
        )
    )
    quadrature = desired_fraction - base
    record: H8AllowanceRecord | None = None
    for _ in range(4):
        right = make_operand_record(
            operand_id="boundary:right",
            shape=(1,),
            infinity_norm=1.0,
            absolute_sum_bound=1.0,
            local_operation_count=1,
            source="numpy",
            solver_produced=False,
            quadrature_convergence=quadrature,
        )
        record = compare_operands(
            comparison_id="boundary:left->right",
            left=left,
            right=right,
            residual=float(residual),
        )
        if record.allowance_scale_fraction == desired_fraction:
            return record
        quadrature += desired_fraction - record.allowance_scale_fraction
    if record is None or record.allowance_scale_fraction != desired_fraction:
        raise AssertionError("could not construct the exact allowance boundary")
    return record


@dataclass(frozen=True, slots=True)
class _Carrier:
    serialized_bytes: bytes
    input_sha256: str


@dataclass(frozen=True, slots=True)
class _SemanticCarrier:
    original: H8Problem
    serialized_bytes: bytes
    input_sha256: str

    def __getattr__(self, name: str) -> object:
        return getattr(self.original, name)


def _replace_chunk(
    problem: H8Problem,
    chunk_index: int,
    replacement: bytes,
) -> _Carrier:
    chunks = _chunks(problem.serialized_bytes)
    chunks[chunk_index] = replacement
    serialized = _frame(chunks)
    return _Carrier(serialized, hashlib.sha256(serialized).hexdigest())


def _chunks(serialized: bytes) -> list[bytes]:
    chunks: list[bytes] = []
    offset = 0
    while offset < len(serialized):
        length = int.from_bytes(serialized[offset : offset + 8], "big")
        start = offset + 8
        end = start + length
        chunks.append(serialized[start:end])
        offset = end
    return chunks


def _frame(chunks: list[bytes]) -> bytes:
    return b"".join(
        len(chunk).to_bytes(8, "big") + chunk for chunk in chunks
    )
