from __future__ import annotations

import hashlib
import json
import math
from dataclasses import FrozenInstanceError, fields

import numpy as np
import pytest

from vfe4.generative.reference_h4 import (
    canonical_h4_gaussian,
    h4_anchor_from_h3,
    make_h4_problem,
    parse_h4_problem_bytes,
)
from vfe4.types.h4 import (
    H4_ALLOWANCE_INVARIANT_NAMES,
    H4_INVARIANT_NAMES,
    H4_MEASUREMENT_NAMES,
    H4_PRIMARY_MEASUREMENTS_UNAVAILABLE_AFTER_ANCHOR_FAIL,
    H4AffineGaussianFactor,
    H4GateResult,
    H4MemoryRecord,
    H4NativeInformationState,
    H4NativeMomentState,
    H4OperationRecord,
    H4RawDraw,
    H4SelectedMoment,
    H4SolveProtocol,
    H4SolverResult,
    H4TerminalLaw,
    H4TimingRecord,
    canonical_h4_problem_bytes,
)
from vfe4.types.results import GateStatus, InvariantResult
from vfe4.validation.h3_fixture import (
    H3_COUPLED_FIXTURE_PATH,
    H3_ZERO_CONTROL_FIXTURE_PATH,
    parse_h3_fixture_bytes,
)

EXPECTED_INVARIANTS = (
    "h3_anchor_identity", "fixed_seed_problem_identity", "coupled_zero_control_contract",
    "cpu_float64_one_thread", "shared_protocol_identity", "scaled_condition_envelope",
    "complete_repetition_table", "primary_timed_order_balance", "exact_posterior_gap_equivalence",
    "terminal_h_equivalence", "terminal_J_equivalence", "selected_moment_equivalence",
    "complete_objective_equivalence", "all_equivalence_allowances_decisive",
    "real_operation_instrumentation", "primary_seed_level_inference", "primary_effect_threshold",
)
EXPECTED_MEASUREMENTS = (
    "primary_seed_ratio_geometric_mean", "primary_bootstrap_lower", "primary_bootstrap_upper",
    "primary_effect_threshold", "primary_timed_ab_total", "primary_timed_ba_total",
    "maximum_solver_stopping_residual", "maximum_allowance_scale_fraction",
)
EXPECTED_UNAVAILABLE = (
    "primary_seed_ratio_geometric_mean", "primary_bootstrap_lower", "primary_bootstrap_upper",
    "primary_timed_ab_total", "primary_timed_ba_total",
)
EXPECTED_ALLOWANCES = (
    "h3_anchor_identity", "exact_posterior_gap_equivalence", "terminal_h_equivalence",
    "terminal_J_equivalence", "selected_moment_equivalence", "complete_objective_equivalence",
)


def _fixture(path, fixture_id):
    return parse_h3_fixture_bytes(path.read_bytes(), expected_fixture_id=fixture_id)


def _independent_draws(seed: int):
    rng = np.random.Generator(np.random.PCG64(seed))
    return (
        rng.standard_normal((4, 4)),
        rng.standard_normal((4, 4)),
        rng.standard_normal((4, 4)),
        rng.uniform(-0.25, 0.25, size=4),
        rng.uniform(-0.25, 0.25, size=4),
        rng.uniform(0.5, 1.5, size=4),
        rng.uniform(0.5, 1.5, size=4),
        rng.standard_normal((8, 8)),
        rng.uniform(-0.25, 0.25, size=8),
        rng.uniform(0.75, 1.25, size=8),
        rng.uniform(-1.0, 1.0, size=8),
    )


def test_scaled_problem_has_exact_schema_coordinate_schedule_and_pcg64_provenance() -> None:
    problem = make_h4_problem(seed=104729, kind="coupled", horizon=7)
    assert tuple(field.name for field in fields(problem)) == (
        "problem_id", "source_kind", "seed", "kind", "horizon", "d_z",
        "d_m", "dimension", "coordinate_order", "factor_schedule",
        "canonical_sha256",
    )
    assert problem.source_kind == "scaled_pcg64"
    assert problem.problem_id == "h4-coupled-T7-dz4-dm4-seed104729-v1"
    assert problem.dimension == 64
    assert problem.coordinate_order[:8] == (
        "z[0,0]", "z[0,1]", "z[0,2]", "z[0,3]",
        "m[0,0]", "m[0,1]", "m[0,2]", "m[0,3]",
    )
    expected_ids = ("initial_joint", *(item for t in range(1, 8) for item in (f"m_transition[{t}]", f"z_transition[{t}]", f"observation[{t}]")))
    assert tuple(f.factor_id for f in problem.factor_schedule) == expected_ids
    assert tuple(f.factor_id for f in problem.factor_schedule[:4]) == (
        "initial_joint", "m_transition[1]", "z_transition[1]", "observation[1]",
    )
    assert len(problem.factor_schedule) == 22
    initial, m_factor, z_factor, observation = problem.factor_schedule[:4]
    assert initial.role == "initial"
    assert initial.normalized_coordinate_indices == tuple(range(8))
    assert initial.parent_coordinate_indices == ()
    assert m_factor.normalized_coordinate_indices == (12, 13, 14, 15)
    assert m_factor.parent_coordinate_indices == (4, 5, 6, 7)
    assert z_factor.normalized_coordinate_indices == (8, 9, 10, 11)
    assert z_factor.parent_coordinate_indices == (0, 1, 2, 3, 12, 13, 14, 15)
    assert observation.normalized_coordinate_indices == ()
    assert observation.parent_coordinate_indices == tuple(range(8, 16))
    draws = tuple(sorted((*m_factor.raw_draws, *z_factor.raw_draws, *observation.raw_draws), key=lambda d: d.draw_index))
    assert tuple(draw.draw_index for draw in draws) == tuple(range(11))
    assert tuple(draw.name for draw in draws) == (
        "A_m[1]", "A_z[1]", "B[1]", "c_m[1]", "c_z[1]", "R_m[1]",
        "R_z[1]", "G[1]", "observation_offset[1]", "observation_noise[1]",
        "observed_target[1]",
    )
    expected = _independent_draws(104729)
    for draw, values in zip(draws, expected, strict=True):
        assert np.array_equal(np.asarray(draw.values).reshape(draw.shape), values)
    raw_a_m, raw_a_z, raw_b, _, _, _, _, raw_g, offset, _, target = expected
    clip = lambda value: value * min(1.0, 0.65 / np.linalg.norm(value, 2))
    assert np.allclose(np.asarray(m_factor.matrix)[:, 4:8], -clip(raw_a_m))
    joined = clip(np.concatenate((raw_a_z, raw_b), axis=1))
    assert np.allclose(np.asarray(z_factor.matrix)[:, :4], -joined[:, :4])
    assert np.allclose(np.asarray(z_factor.matrix)[:, 12:16], -joined[:, 4:])
    assert np.allclose(np.asarray(observation.matrix)[:, 8:16], np.eye(8) + .05 * raw_g / max(1.0, np.linalg.norm(raw_g, 2)))
    assert np.allclose(np.asarray(observation.target), target - offset)
    with pytest.raises(FrozenInstanceError):
        problem.seed = 1  # type: ignore[misc]


@pytest.mark.parametrize(("horizon", "dimension"), ((7, 64), (15, 128), (31, 256)))
def test_every_scaled_horizon_has_complete_local_schedule_and_global_draw_indices(horizon: int, dimension: int) -> None:
    problem = make_h4_problem(seed=104729, kind="coupled", horizon=horizon)  # type: ignore[arg-type]
    expected_coordinates = tuple(f"{prefix}[{time},{component}]" for time in range(horizon + 1) for prefix in ("z", "m") for component in range(4))
    expected_ids = ("initial_joint", *(item for time in range(1, horizon + 1) for item in (f"m_transition[{time}]", f"z_transition[{time}]", f"observation[{time}]")))
    assert problem.dimension == dimension and problem.coordinate_order == expected_coordinates
    assert tuple(factor.factor_id for factor in problem.factor_schedule) == expected_ids
    observed_indices = tuple(sorted(draw.draw_index for factor in problem.factor_schedule for draw in factor.raw_draws))
    assert observed_indices == tuple(range(11 * horizon))


def test_zero_control_and_canonical_bytes_have_only_frozen_differences() -> None:
    coupled = make_h4_problem(seed=130363, kind="coupled", horizon=7)
    zero = make_h4_problem(seed=130363, kind="zero_control", horizon=7)
    assert coupled.canonical_sha256 != zero.canonical_sha256
    assert coupled.coordinate_order == zero.coordinate_order
    for left, right in zip(coupled.factor_schedule, zero.factor_schedule, strict=True):
        assert left.factor_id == right.factor_id
        assert left.role == right.role
        assert left.time_index == right.time_index
        assert left.normalized_coordinate_indices == right.normalized_coordinate_indices
        assert left.parent_coordinate_indices == right.parent_coordinate_indices
        assert left.target == right.target and left.covariance == right.covariance
        assert left.raw_draws == right.raw_draws
        if left.role == "transition":
            left_matrix, right_matrix = np.asarray(left.matrix), np.asarray(right.matrix)
            assert np.array_equal(right_matrix[:, right.parent_coordinate_indices], np.zeros((len(right.target), len(right.parent_coordinate_indices))))
            allowed = set(left.parent_coordinate_indices)
            for column in range(coupled.dimension):
                if column not in allowed:
                    assert np.array_equal(left_matrix[:, column], right_matrix[:, column])
        else:
            assert left.matrix == right.matrix
    bytes_one = canonical_h4_problem_bytes(coupled)
    bytes_two = canonical_h4_problem_bytes(make_h4_problem(seed=130363, kind="coupled", horizon=7))
    assert bytes_one == bytes_two
    envelope = json.loads(bytes_one)
    core = envelope["problem"]
    digest = hashlib.sha256(b"vfe4.h4.neutral-problem.v1\0" + json.dumps(core, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    assert envelope == {"schema_version": "h4-neutral-problem-v1", "canonical_sha256": digest, "problem": core}
    assert digest == coupled.canonical_sha256
    assert parse_h4_problem_bytes(bytes_one) == coupled
    assert coupled.canonical_sha256 != make_h4_problem(seed=130364, kind="coupled", horizon=7).canonical_sha256
    assert coupled.canonical_sha256 != make_h4_problem(seed=130363, kind="coupled", horizon=15).canonical_sha256
    malformed_schema = json.loads(bytes_one); malformed_schema["schema_version"] = "wrong"
    with pytest.raises(ValueError): parse_h4_problem_bytes(json.dumps(malformed_schema).encode())
    malformed_digest = json.loads(bytes_one); malformed_digest["canonical_sha256"] = "0" * 64
    with pytest.raises(ValueError): parse_h4_problem_bytes(json.dumps(malformed_digest).encode())
    malformed_extra = json.loads(bytes_one); malformed_extra["extra"] = 1
    with pytest.raises(ValueError): parse_h4_problem_bytes(json.dumps(malformed_extra).encode())


def test_h3_structural_adapter_and_independent_canonical_assembly() -> None:
    coupled_fixture = _fixture(H3_COUPLED_FIXTURE_PATH, "h3-coupled-v1")
    zero_fixture = _fixture(H3_ZERO_CONTROL_FIXTURE_PATH, "h3-zero-control-v1")
    for fixture in (coupled_fixture, zero_fixture):
        problem = h4_anchor_from_h3(fixture)
        assert problem.source_kind == "h3_anchor" and problem.seed == 0
        assert problem.problem_id == f"h4-anchor-{fixture.fixture_id}"
        assert problem.coordinate_order == ("z0", "m0", "z1", "m1")
        assert tuple(f.factor_id for f in problem.factor_schedule) == tuple(f.factor_id for f in fixture.factors)
        assert all(f.raw_draws == () for f in problem.factor_schedule)
        assert tuple((factor.role, factor.time_index, factor.normalized_coordinate_indices, factor.parent_coordinate_indices) for factor in problem.factor_schedule) == (
            ("initial", 0, (0,), ()), ("initial", 0, (1,), ()),
            ("transition", 1, (3,), (1,)), ("transition", 1, (2,), (0, 3)),
            ("observation", 1, (), (2,)), ("observation", 1, (), (3,)),
        )
        J, h, c, log_z = canonical_h4_gaussian(problem)
        expected_J = sum(np.outer(np.asarray(f.row), np.asarray(f.row)) / f.variance for f in fixture.factors)
        expected_h = sum(f.target * np.asarray(f.row) / f.variance for f in fixture.factors)
        expected_c = sum(-.5 * (f.target * f.target / f.variance + math.log(2.0 * math.pi * f.variance)) for f in fixture.factors)
        assert np.allclose(J, expected_J) and np.allclose(h, expected_h) and math.isclose(c, expected_c)
        expected_log_z = expected_c + .5 * expected_h @ np.linalg.solve(expected_J, expected_h) - .5 * np.linalg.slogdet(expected_J)[1] + 2.0 * math.log(2.0 * math.pi)
        assert math.isclose(log_z, expected_log_z)
        if fixture.reference_log_evidence is not None:
            assert math.isclose(log_z, fixture.reference_log_evidence, rel_tol=0.0, abs_tol=2e-14)


def test_records_reject_malformed_inputs_and_gate_freezes_exact_early_failure_shape() -> None:
    with pytest.raises(ValueError):
        H4RawDraw(0, "bad", (2,), (1.0,))
    with pytest.raises(ValueError):
        H4AffineGaussianFactor("x", "initial", 0, (0,), (), ((1.0,),), (0.0,), ((0.0,),), ())
    moment = H4SelectedMoment("initial", (0.0,), ((1.0,),))
    assert moment.mean == (0.0,)
    invariants = tuple(
        InvariantResult(
            name,
            False,
            1.0 if name == "h3_anchor_identity" else None,
            0.0 if name == "h3_anchor_identity" else None,
            "anchor_miss" if name == "h3_anchor_identity" else "not_evaluated_after_decisive_h3_anchor_failure",
        )
        for name in EXPECTED_INVARIANTS
    )
    measurements = {name: None for name in EXPECTED_MEASUREMENTS}
    measurements.update({"primary_effect_threshold": .80, "maximum_solver_stopping_residual": .0, "maximum_allowance_scale_fraction": .0})
    numerical = {"applicable": True, "dimension": 4, "operands": (1.0,), "absolute_summands": (1.0,), "condition_numbers": (1.0,), "operation_counts": (1,), "solver_contribution": 0.0, "invariant_scale": 1.0, "final_allowance": 0.0, "ratio": 0.0}
    allowances = {name: numerical if name == "h3_anchor_identity" else {"applicable": False, "reason": "not_evaluated_after_decisive_h3_anchor_failure"} for name in EXPECTED_ALLOWANCES}
    result = H4GateResult("H4", GateStatus.FAIL, measurements, invariants, allowances, ())
    assert H4_INVARIANT_NAMES == EXPECTED_INVARIANTS and H4_MEASUREMENT_NAMES == EXPECTED_MEASUREMENTS and H4_PRIMARY_MEASUREMENTS_UNAVAILABLE_AFTER_ANCHOR_FAIL == EXPECTED_UNAVAILABLE and H4_ALLOWANCE_INVARIANT_NAMES == EXPECTED_ALLOWANCES
    assert tuple(result.measurements) == EXPECTED_MEASUREMENTS
    assert tuple(name for name, value in result.measurements.items() if value is None) == EXPECTED_UNAVAILABLE
    with pytest.raises(ValueError):
        H4GateResult("H4", GateStatus.FAIL, {name: 0.0 for name in EXPECTED_MEASUREMENTS}, invariants, allowances, ())
    bad_allowances = dict(allowances); bad_allowances["h3_anchor_identity"] = {"applicable": True}
    with pytest.raises(ValueError): H4GateResult("H4", GateStatus.FAIL, measurements, invariants, bad_allowances, ())


def test_every_task_one_record_has_frozen_fields_and_positive_negative_validation() -> None:
    assert tuple(field.name for field in fields(H4RawDraw)) == ("draw_index", "name", "shape", "values")
    assert tuple(field.name for field in fields(H4AffineGaussianFactor)) == ("factor_id", "role", "time_index", "normalized_coordinate_indices", "parent_coordinate_indices", "matrix", "target", "covariance", "raw_draws")
    assert tuple(field.name for field in fields(H4SolveProtocol)) == ("protocol_id", "dtype", "device", "factor_passes", "solver_relative_budget", "stopping_rule")
    assert H4SolveProtocol().solver_relative_budget == 1e-9
    with pytest.raises(ValueError): H4SolveProtocol(solver_relative_budget=1e-8)
    selected = (
        H4SelectedMoment("initial", (0.0,), ((1.0,),)),
        H4SelectedMoment("terminal", (0.0,), ((1.0,),)),
        H4SelectedMoment("observation[1]", (0.0,), ((1.0,),)),
    )
    law = H4TerminalLaw("information", (0.0,), ((1.0,),), (0.0,), selected, 0.0, 0.0)
    information = H4NativeInformationState((0.0,), ((1.0,),), (0.0,), 0.0)
    moment = H4NativeMomentState((0.0,), ((1.0,),), 0.0)
    assert law.selected_moments == selected and moment.covariance == ((1.0,),)
    with pytest.raises(ValueError): H4TerminalLaw("information", (0.0,), ((1.0,),), (0.0,), (selected[0], selected[0]), 0.0, 0.0)
    result = H4SolverResult("p", "a" * 64, "information", "h4-single-pass-v1", 1, information, None)
    assert result.native_information is information
    with pytest.raises(ValueError): H4SolverResult("p", "a" * 64, "information", "h4-single-pass-v1", 1, information, moment)
    timing = H4TimingRecord("p", 0, 0, 0, 0, 104729, "coupled", 7, 0, 3, "moment_then_information", 1, 1)
    assert timing.order == "moment_then_information"
    with pytest.raises(ValueError): H4TimingRecord("p", 0, 0, 0, 0, 104729, "coupled", 7, 0, 3, "information_then_moment", 1, 1)
    operation = H4OperationRecord("p", "information", "cholesky", ((2, 2),), (2, 2), 1)
    assert operation.count == 1
    with pytest.raises(ValueError): H4OperationRecord("p", "information", "cholesky", ((0, 2),), (2, 2), 1)
    memory = H4MemoryRecord("p", "moment", None, -5, ("python_peak_bytes",))
    assert memory.process_working_set_delta_bytes == -5
    with pytest.raises(ValueError): H4MemoryRecord("p", "moment", None, -5, ())
