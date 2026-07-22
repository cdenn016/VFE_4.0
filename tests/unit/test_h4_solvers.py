from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
from dataclasses import FrozenInstanceError, fields, replace

import numpy as np
import pytest
import torch

import vfe4.inference.h4_solvers as solver_module
from vfe4.generative.reference_h4 import (
    canonical_h4_gaussian,
    h4_anchor_from_h3,
    make_h4_problem,
)
from vfe4.inference.h4_instrumentation import (
    CountingOperationRecorder,
    InstrumentedLinearAlgebra,
    NullOperationRecorder,
)
from vfe4.inference.h4_solvers import (
    H4GaussianSolver,
    H4InnovationDiagnostic,
    H4MaterializedProblem,
    H4NativeDiagnostics,
    InformationFormH4Solver,
    MomentFormH4Solver,
    evaluate_h4_native_diagnostics,
    materialize_h4_problem,
    solve_information_form,
    solve_moment_form,
    to_common_terminal_law,
)
from vfe4.types.h4 import (
    H4NeutralProblem,
    H4SolveProtocol,
    h4_problem_digest,
)
from vfe4.validation.h3_fixture import (
    H3_COUPLED_FIXTURE_PATH,
    H3_ZERO_CONTROL_FIXTURE_PATH,
    parse_h3_fixture_bytes,
)


MATERIALIZED_FIELDS = (
    "materialization_version",
    "problem_id",
    "problem_sha256",
    "protocol_id",
    "dtype",
    "device",
    "source_kind",
    "seed",
    "kind",
    "horizon",
    "d_z",
    "d_m",
    "dimension",
    "coordinate_order",
    "factor_ids",
    "factor_roles",
    "factor_time_indices",
    "factor_normalized_coordinate_indices",
    "factor_parent_coordinate_indices",
    "_factor_matrices",
    "_factor_targets",
    "_factor_covariances",
    "tensor_sha256",
)
INNOVATION_FIELDS = (
    "factor_id",
    "time_index",
    "parent_coordinate_indices",
    "innovation_dimension",
    "minimum_eigenvalue",
    "maximum_eigenvalue",
    "condition_number",
    "minimum_cholesky_pivot",
)
DIAGNOSTIC_FIELDS = (
    "problem_id",
    "problem_sha256",
    "protocol_id",
    "arm",
    "factor_count",
    "replayed_result",
    "innovation_diagnostics",
    "finite",
    "spd",
    "replay_matches_result",
)


def _fixture(kind: str):
    if kind == "coupled":
        path, fixture_id = H3_COUPLED_FIXTURE_PATH, "h3-coupled-v1"
    else:
        path, fixture_id = H3_ZERO_CONTROL_FIXTURE_PATH, "h3-zero-control-v1"
    return parse_h3_fixture_bytes(path.read_bytes(), expected_fixture_id=fixture_id)


def _h3_problem(kind: str = "coupled") -> H4NeutralProblem:
    return h4_anchor_from_h3(_fixture(kind))


def _rehash(problem: H4NeutralProblem, factors: tuple) -> H4NeutralProblem:
    temporary = replace(
        problem,
        factor_schedule=factors,
        canonical_sha256="0" * 64,
    )
    return replace(temporary, canonical_sha256=h4_problem_digest(temporary))


def _shift_observation(
    problem: H4NeutralProblem,
    factor_id: str,
    delta: tuple[float, ...],
) -> H4NeutralProblem:
    factors = list(problem.factor_schedule)
    index = next(i for i, factor in enumerate(factors) if factor.factor_id == factor_id)
    factor = factors[index]
    assert factor.role == "observation" and len(delta) == len(factor.target)
    shifted_draws = []
    for draw in factor.raw_draws:
        if draw.name.startswith("observed_target["):
            shifted_draws.append(
                replace(
                    draw,
                    values=tuple(value + change for value, change in zip(draw.values, delta, strict=True)),
                )
            )
        else:
            shifted_draws.append(draw)
    factors[index] = replace(
        factor,
        target=tuple(value + change for value, change in zip(factor.target, delta, strict=True)),
        raw_draws=tuple(shifted_draws),
    )
    return _rehash(problem, tuple(factors))


def _independent_materialized_digest(value: H4MaterializedProblem) -> str:
    metadata = {
        name: getattr(value, name)
        for name in MATERIALIZED_FIELDS
        if name not in {
            "_factor_matrices",
            "_factor_targets",
            "_factor_covariances",
            "tensor_sha256",
        }
    }
    encoded = json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(b"vfe4.h4.materialized-problem.v1\x00")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)
    for role, tensors in (
        ("matrix", value._factor_matrices),
        ("target", value._factor_targets),
        ("covariance", value._factor_covariances),
    ):
        for index, tensor in enumerate(tensors):
            header = json.dumps(
                {
                    "role": role,
                    "index": index,
                    "shape": tuple(tensor.shape),
                    "dtype": str(tensor.dtype).removeprefix("torch."),
                    "device": str(tensor.device),
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            raw = tensor.detach().numpy().tobytes(order="C")
            digest.update(len(header).to_bytes(8, "big"))
            digest.update(header)
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
    return digest.hexdigest()


def _null(materialized: H4MaterializedProblem, arm: str) -> InstrumentedLinearAlgebra:
    return InstrumentedLinearAlgebra(
        problem_id=materialized.problem_id,
        arm=arm,  # type: ignore[arg-type]
        recorder=NullOperationRecorder(),
    )


def _tuple_array(value: tuple) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def test_runtime_records_and_solver_signatures_are_exact() -> None:
    assert tuple(field.name for field in fields(H4MaterializedProblem)) == MATERIALIZED_FIELDS
    assert tuple(field.name for field in fields(H4InnovationDiagnostic)) == INNOVATION_FIELDS
    assert tuple(field.name for field in fields(H4NativeDiagnostics)) == DIAGNOSTIC_FIELDS
    assert "__slots__" in H4MaterializedProblem.__dict__
    assert "__slots__" in H4InnovationDiagnostic.__dict__
    assert "__slots__" in H4NativeDiagnostics.__dict__
    assert tuple(inspect.signature(materialize_h4_problem).parameters) == (
        "problem",
        "protocol",
    )
    assert tuple(inspect.signature(H4GaussianSolver.solve).parameters) == (
        "self",
        "materialized",
        "protocol",
        "linalg",
    )
    for function in (solve_information_form, solve_moment_form):
        assert tuple(inspect.signature(function).parameters) == (
            "materialized",
            "protocol",
            "linalg",
        )
    for function in (to_common_terminal_law, evaluate_h4_native_diagnostics):
        assert tuple(inspect.signature(function).parameters) == (
            "materialized",
            "result",
            "linalg",
        )


def test_materialization_is_raw_only_owned_and_digest_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    problem = _h3_problem()
    protocol = H4SolveProtocol()

    def forbidden(*args, **kwargs):
        raise AssertionError("derived numerical operation ran during materialization")

    with monkeypatch.context() as context:
        for owner, name in (
            (torch.linalg, "cholesky"),
            (torch.linalg, "solve_triangular"),
            (torch.linalg, "inv"),
            (torch.linalg, "slogdet"),
            (torch.linalg, "eigvalsh"),
            (torch.linalg, "cond"),
            (torch, "matmul"),
            (torch, "mm"),
            (torch, "bmm"),
        ):
            context.setattr(owner, name, forbidden)
        materialized = materialize_h4_problem(problem, protocol)

    assert materialized.materialization_version == "h4-materialized-problem-v1"
    assert materialized.problem_id == problem.problem_id
    assert materialized.problem_sha256 == problem.canonical_sha256
    assert materialized.protocol_id == protocol.protocol_id
    assert materialized.dtype == "float64" and materialized.device == "cpu"
    assert materialized.tensor_sha256 == _independent_materialized_digest(materialized)
    assert not hasattr(materialized, "factor_matrices")
    with pytest.raises((AttributeError, FrozenInstanceError)):
        materialized.problem_id = "changed"  # type: ignore[misc]

    tensors = (
        *materialized._factor_matrices,
        *materialized._factor_targets,
        *materialized._factor_covariances,
    )
    assert len({tensor.untyped_storage().data_ptr() for tensor in tensors}) == len(tensors)
    assert all(
        tensor.dtype is torch.float64
        and tensor.device.type == "cpu"
        and tensor.is_contiguous()
        and tensor.is_leaf
        and tensor._base is None
        and not tensor.requires_grad
        for tensor in tensors
    )
    for factor, matrix, target, covariance in zip(
        problem.factor_schedule,
        materialized._factor_matrices,
        materialized._factor_targets,
        materialized._factor_covariances,
        strict=True,
    ):
        assert torch.equal(matrix, torch.tensor(factor.matrix, dtype=torch.float64))
        assert torch.equal(target, torch.tensor(factor.target, dtype=torch.float64))
        assert torch.equal(covariance, torch.tensor(factor.covariance, dtype=torch.float64))

    changed_targets = list(materialized._factor_targets)
    changed_targets[0] = changed_targets[0] + 1.0
    with pytest.raises(ValueError, match="tensor_sha256"):
        replace(materialized, _factor_targets=tuple(changed_targets))


def test_both_native_arms_are_clone_free_and_do_not_mutate_shared_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = H4SolveProtocol()
    materialized = materialize_h4_problem(_h3_problem(), protocol)
    before = tuple(
        tensor.detach().numpy().tobytes(order="C")
        for tensor in (
            *materialized._factor_matrices,
            *materialized._factor_targets,
            *materialized._factor_covariances,
        )
    )
    digest = materialized.tensor_sha256

    def no_clone(*args, **kwargs):
        raise AssertionError("solver cloned materialized tensors")

    monkeypatch.setattr(torch.Tensor, "clone", no_clone)
    information = solve_information_form(materialized, protocol, _null(materialized, "information"))
    moment = solve_moment_form(materialized, protocol, _null(materialized, "moment"))
    assert information.problem_id == moment.problem_id == materialized.problem_id
    assert information.factor_count == moment.factor_count == len(materialized.factor_ids)
    assert materialized.tensor_sha256 == digest == _independent_materialized_digest(materialized)
    after = tuple(
        tensor.detach().numpy().tobytes(order="C")
        for tensor in (
            *materialized._factor_matrices,
            *materialized._factor_targets,
            *materialized._factor_covariances,
        )
    )
    assert after == before


@pytest.mark.parametrize("kind", ("coupled", "zero_control"))
def test_h3_information_and_moment_arms_match_exact_gaussian_and_selected_blocks(kind: str) -> None:
    problem = _h3_problem(kind)
    protocol = H4SolveProtocol()
    materialized = materialize_h4_problem(problem, protocol)
    expected_J, expected_h, _, expected_log_z = canonical_h4_gaussian(problem)
    expected_mean = np.linalg.solve(expected_J, expected_h)
    expected_covariance = np.linalg.inv(expected_J)

    information = InformationFormH4Solver().solve(
        materialized,
        protocol,
        _null(materialized, "information"),
    )
    moment = MomentFormH4Solver().solve(
        materialized,
        protocol,
        _null(materialized, "moment"),
    )
    assert information.native_information is not None
    assert moment.native_moment is not None
    np.testing.assert_allclose(_tuple_array(information.native_information.J), expected_J, rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(_tuple_array(information.native_information.h), expected_h, rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(_tuple_array(information.native_information.mean), expected_mean, rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(_tuple_array(moment.native_moment.mean), expected_mean, rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(_tuple_array(moment.native_moment.covariance), expected_covariance, rtol=0.0, atol=3.0e-14)
    assert math.isclose(information.native_information.complete_objective, expected_log_z, rel_tol=0.0, abs_tol=2.0e-14)
    assert math.isclose(moment.native_moment.complete_objective, expected_log_z, rel_tol=0.0, abs_tol=2.0e-14)

    information_law = to_common_terminal_law(
        materialized,
        information,
        _null(materialized, "information"),
    )
    moment_law = to_common_terminal_law(
        materialized,
        moment,
        _null(materialized, "moment"),
    )
    assert tuple(item.name for item in information_law.selected_moments) == (
        "initial",
        "terminal",
        "observation[1]",
    )
    assert all(len(item.mean) == 2 for item in information_law.selected_moments)
    np.testing.assert_allclose(_tuple_array(information_law.J), expected_J, rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(_tuple_array(moment_law.J), expected_J, rtol=0.0, atol=5.0e-14)
    np.testing.assert_allclose(_tuple_array(information_law.h), expected_h, rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(_tuple_array(moment_law.h), expected_h, rtol=0.0, atol=5.0e-14)
    assert information_law.complete_objective == information.native_information.complete_objective
    assert moment_law.complete_objective == moment.native_moment.complete_objective
    for left, right in zip(information_law.selected_moments, moment_law.selected_moments, strict=True):
        np.testing.assert_allclose(_tuple_array(left.mean), _tuple_array(right.mean), rtol=0.0, atol=4.0e-14)
        np.testing.assert_allclose(_tuple_array(left.covariance), _tuple_array(right.covariance), rtol=0.0, atol=5.0e-14)


class _ResidualTrace(InstrumentedLinearAlgebra):
    def __init__(self, *, problem_id: str) -> None:
        object.__setattr__(self, "vector_rhs", [])
        super().__init__(problem_id=problem_id, arm="moment", recorder=NullOperationRecorder())

    def triangular_solve(
        self,
        triangular: torch.Tensor,
        rhs: torch.Tensor,
        *,
        upper: bool = False,
    ) -> torch.Tensor:
        if rhs.ndim == 1:
            self.vector_rhs.append(tuple(float(value) for value in rhs.tolist()))
        return super().triangular_solve(triangular, rhs, upper=upper)


class _SelectedVectorTrace(InstrumentedLinearAlgebra):
    def __init__(self, *, problem_id: str, watched: tuple[int, ...]) -> None:
        object.__setattr__(self, "watched", watched)
        object.__setattr__(self, "vectors", [])
        super().__init__(problem_id=problem_id, arm="moment", recorder=NullOperationRecorder())

    def selected_block_extract(
        self,
        value: torch.Tensor,
        row_indices: tuple[int, ...],
        column_indices: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        result = super().selected_block_extract(value, row_indices, column_indices)
        if value.ndim == 1 and row_indices == self.watched:
            self.vectors.append(tuple(float(item) for item in result.tolist()))
        return result


def test_h3_second_observation_consumes_the_first_observation_posterior() -> None:
    base = _h3_problem()
    changed = _shift_observation(base, "z1_observation", (0.5,))
    protocol = H4SolveProtocol()
    base_materialized = materialize_h4_problem(base, protocol)
    changed_materialized = materialize_h4_problem(changed, protocol)
    base_trace = _ResidualTrace(problem_id=base.problem_id)
    changed_trace = _ResidualTrace(problem_id=changed.problem_id)

    base_result = solve_moment_form(base_materialized, protocol, base_trace)
    changed_result = solve_moment_form(changed_materialized, protocol, changed_trace)
    assert len(base_trace.vector_rhs) == len(changed_trace.vector_rhs) == 2
    assert base_trace.vector_rhs[1] != changed_trace.vector_rhs[1]
    assert base_result.native_moment is not None and changed_result.native_moment is not None
    assert base_result.native_moment.complete_objective != changed_result.native_moment.complete_objective


def test_scaled_next_transition_consumes_the_preceding_observation_posterior() -> None:
    base = make_h4_problem(seed=104729, kind="coupled", horizon=7)
    changed = _shift_observation(base, "observation[1]", (0.5,) * 8)
    protocol = H4SolveProtocol()
    base_materialized = materialize_h4_problem(base, protocol)
    changed_materialized = materialize_h4_problem(changed, protocol)
    watched_m1 = tuple(range(12, 16))
    base_trace = _SelectedVectorTrace(problem_id=base.problem_id, watched=watched_m1)
    changed_trace = _SelectedVectorTrace(problem_id=changed.problem_id, watched=watched_m1)

    solve_moment_form(base_materialized, protocol, base_trace)
    solve_moment_form(changed_materialized, protocol, changed_trace)
    assert len(base_trace.vectors) == len(changed_trace.vectors) == 1
    assert base_trace.vectors[0] != changed_trace.vectors[0]


def test_scaled_selected_blocks_are_eight_dimensional() -> None:
    problem = make_h4_problem(seed=104729, kind="zero_control", horizon=7)
    protocol = H4SolveProtocol()
    materialized = materialize_h4_problem(problem, protocol)
    result = solve_moment_form(materialized, protocol, _null(materialized, "moment"))
    law = to_common_terminal_law(materialized, result, _null(materialized, "moment"))
    assert tuple(item.name for item in law.selected_moments) == (
        "initial",
        "terminal",
        *(f"observation[{time}]" for time in range(1, 8)),
    )
    assert all(len(item.mean) == 8 and len(item.covariance) == 8 for item in law.selected_moments)


class _SolveShapeTrace(InstrumentedLinearAlgebra):
    def __init__(self, *, problem_id: str, arm: str) -> None:
        object.__setattr__(self, "rhs_shapes", [])
        super().__init__(
            problem_id=problem_id,
            arm=arm,  # type: ignore[arg-type]
            recorder=NullOperationRecorder(),
        )

    def triangular_solve(
        self,
        triangular: torch.Tensor,
        rhs: torch.Tensor,
        *,
        upper: bool = False,
    ) -> torch.Tensor:
        self.rhs_shapes.append(tuple(rhs.shape))
        return super().triangular_solve(triangular, rhs, upper=upper)


def test_information_conversion_solves_only_selected_inverse_columns_and_residual_is_exact() -> None:
    protocol = H4SolveProtocol()
    materialized = materialize_h4_problem(_h3_problem(), protocol)
    result = solve_information_form(materialized, protocol, _null(materialized, "information"))
    trace = _SolveShapeTrace(problem_id=materialized.problem_id, arm="information")
    law = to_common_terminal_law(materialized, result, trace)
    assert trace.rhs_shapes
    assert all(shape == (4, 2) for shape in trace.rhs_shapes)

    J = _tuple_array(law.J)
    h = _tuple_array(law.h)
    mean = _tuple_array(law.mean)
    numerator = np.max(np.abs(J @ mean - h))
    scale = max(1.0, np.max(np.sum(np.abs(J), axis=1)) * np.max(np.abs(mean)) + np.max(np.abs(h)))
    assert law.stopping_residual == numerator / scale


def test_native_diagnostics_replay_exactly_and_require_null_identity_binding() -> None:
    protocol = H4SolveProtocol()
    materialized = materialize_h4_problem(_h3_problem(), protocol)
    moment = solve_moment_form(materialized, protocol, _null(materialized, "moment"))
    diagnostics = evaluate_h4_native_diagnostics(
        materialized,
        moment,
        _null(materialized, "moment"),
    )
    assert type(diagnostics) is H4NativeDiagnostics
    assert diagnostics.replayed_result == moment
    assert diagnostics.finite is diagnostics.spd is diagnostics.replay_matches_result is True
    assert tuple(item.factor_id for item in diagnostics.innovation_diagnostics) == (
        "z1_observation",
        "m1_observation",
    )
    assert all(item.innovation_dimension == 1 for item in diagnostics.innovation_diagnostics)

    with pytest.raises(ValueError, match="null-recorder"):
        evaluate_h4_native_diagnostics(
            materialized,
            moment,
            InstrumentedLinearAlgebra(
                problem_id=materialized.problem_id,
                arm="moment",
                recorder=CountingOperationRecorder(),
            ),
        )
    with pytest.raises(ValueError, match="facade"):
        evaluate_h4_native_diagnostics(
            materialized,
            moment,
            InstrumentedLinearAlgebra(
                problem_id="wrong",
                arm="moment",
                recorder=NullOperationRecorder(),
            ),
        )

    assert moment.native_moment is not None
    altered_state = replace(
        moment.native_moment,
        complete_objective=math.nextafter(
            moment.native_moment.complete_objective,
            math.inf,
        ),
    )
    altered = replace(moment, native_moment=altered_state)
    with pytest.raises(ValueError, match="replay"):
        evaluate_h4_native_diagnostics(
            materialized,
            altered,
            _null(materialized, "moment"),
        )

    information = solve_information_form(materialized, protocol, _null(materialized, "information"))
    information_diagnostics = evaluate_h4_native_diagnostics(
        materialized,
        information,
        _null(materialized, "information"),
    )
    assert information_diagnostics.innovation_diagnostics == ()


def test_common_conversion_rejects_result_and_facade_identity_mismatches() -> None:
    protocol = H4SolveProtocol()
    materialized = materialize_h4_problem(_h3_problem(), protocol)
    result = solve_information_form(materialized, protocol, _null(materialized, "information"))
    with pytest.raises(ValueError, match="facade"):
        to_common_terminal_law(
            materialized,
            result,
            InstrumentedLinearAlgebra(
                problem_id="wrong",
                arm="information",
                recorder=NullOperationRecorder(),
            ),
        )
    with pytest.raises(ValueError, match="result"):
        to_common_terminal_law(
            materialized,
            replace(result, problem_id="wrong"),
            _null(materialized, "information"),
        )


def test_solver_independence_and_facade_completeness(monkeypatch: pytest.MonkeyPatch) -> None:
    protocol = H4SolveProtocol()
    materialized = materialize_h4_problem(_h3_problem(), protocol)

    def forbidden(*args, **kwargs):
        raise AssertionError("forbidden independent-arm dependency")

    with monkeypatch.context() as context:
        context.setattr(solver_module, "solve_information_form", forbidden)
        context.setattr(torch.linalg, "inv", forbidden)
        context.setattr(torch.linalg, "pinv", forbidden)
        context.setattr(torch, "cholesky_inverse", forbidden)
        result = MomentFormH4Solver().solve(
            materialized,
            protocol,
            _null(materialized, "moment"),
        )
        assert result.arm == "moment"

    with monkeypatch.context() as context:
        context.setattr(solver_module, "solve_moment_form", forbidden)
        result = InformationFormH4Solver().solve(
            materialized,
            protocol,
            _null(materialized, "information"),
        )
        assert result.arm == "information"

    tree = ast.parse(inspect.getsource(solver_module))
    assert not any(isinstance(node, ast.MatMult) for node in ast.walk(tree))
    forbidden_torch_calls: list[str] = []
    linalg_calls: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        value = node.func.value
        if isinstance(value, ast.Name) and value.id == "torch" and node.func.attr in {
            "matmul",
            "mm",
            "bmm",
        }:
            forbidden_torch_calls.append(node.func.attr)
        if (
            isinstance(value, ast.Attribute)
            and isinstance(value.value, ast.Name)
            and value.value.id == "torch"
            and value.attr == "linalg"
        ):
            linalg_calls.append(node.func.attr)
    assert forbidden_torch_calls == []
    assert set(linalg_calls) <= {"eigvalsh"}


def test_counting_pass_records_every_defined_operation_kind_used_by_moment_arm() -> None:
    protocol = H4SolveProtocol()
    materialized = materialize_h4_problem(_h3_problem(), protocol)
    recorder = CountingOperationRecorder()
    result = solve_moment_form(
        materialized,
        protocol,
        InstrumentedLinearAlgebra(
            problem_id=materialized.problem_id,
            arm="moment",
            recorder=recorder,
        ),
    )
    assert result.arm == "moment"
    assert {record.operation for record in recorder.snapshot()} == {
        "cholesky",
        "triangular_solve",
        "matrix_multiply",
        "symmetric_rank_update",
        "selected_block_extract",
    }
    assert all(record.problem_id == materialized.problem_id and record.arm == "moment" for record in recorder.snapshot())
