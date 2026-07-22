from __future__ import annotations

import hashlib
import json
import math
from dataclasses import FrozenInstanceError, fields, replace

import numpy as np
import pytest

import vfe4.types.h4 as h4_types_module
from vfe4.generative.reference_h4 import (
    canonical_h4_gaussian,
    h4_anchor_from_h3,
    make_h4_problem,
    parse_h4_problem_bytes,
)
from vfe4.types.h4 import (
    H4_ALLOWANCE_ELEMENT_COUNTS,
    H4_ALLOWANCE_INVARIANT_NAMES,
    H4_INVARIANT_NAMES,
    H4_MEASUREMENT_NAMES,
    H4_PRIMARY_TIMED_AB_TOTAL,
    H4_PRIMARY_TIMED_BALANCE,
    H4_PRIMARY_TIMED_BA_TOTAL,
    H4_PRIMARY_MEASUREMENTS_UNAVAILABLE_AFTER_ANCHOR_FAIL,
    H4AllowanceElement,
    H4AllowanceOperationCount,
    H4AllowanceOperand,
    H4ApplicableAllowance,
    H4InapplicableAllowance,
    H4IntervalDecision,
    H4AffineGaussianFactor,
    H4GateResult,
    H4MemoryRecord,
    H4NeutralProblem,
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
    classify_h4_interval,
)
from vfe4.types.results import GateStatus, InvariantResult
from vfe4.validation.h3_fixture import (
    H3_COUPLED_FIXTURE_PATH,
    H3_ZERO_CONTROL_FIXTURE_PATH,
    parse_h3_fixture_bytes,
)
from verification.h3_budget import pair_allowance, scalar_allowance
from verification.numpy_oracles.h3_posterior import evaluate_h3_posterior_oracle

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


def test_corrected_allowance_and_interval_public_contract_is_frozen() -> None:
    assert H4_ALLOWANCE_ELEMENT_COUNTS == (
        ("h3_anchor_identity", 184),
        ("exact_posterior_gap_equivalence", 2_640),
        ("terminal_h_equivalence", 394_240),
        ("terminal_J_equivalence", 75_694_080),
        ("selected_moment_equivalence", 3_738_240),
        ("complete_objective_equivalence", 2_640),
    )
    assert sum(count for _, count in H4_ALLOWANCE_ELEMENT_COUNTS) == 79_832_024
    assert (H4_PRIMARY_TIMED_AB_TOTAL, H4_PRIMARY_TIMED_BA_TOTAL) == (110, 110)
    assert len(H4_PRIMARY_TIMED_BALANCE) == 20
    assert tuple(field.name for field in fields(H4AllowanceOperationCount)) == ("label", "count")
    assert tuple(field.name for field in fields(H4AllowanceOperand)) == (
        "label", "value", "value_norm", "absolute_summand_accumulation",
        "condition_numbers", "operation_counts", "solver_produced",
        "rounding_allowance", "solver_allowance", "total_allowance",
    )
    assert tuple(field.name for field in fields(H4AllowanceElement))[-7:] == (
        "comparison_reduction_allowance", "residual", "normalized_residual",
        "final_allowance", "allowance_scale_ratio", "decisive", "passed",
    )
    assert tuple(field.name for field in fields(H4ApplicableAllowance))[0:4] == (
        "applicable", "invariant", "element_stream_domain", "expected_element_count",
    )
    assert tuple(field.name for field in fields(H4InapplicableAllowance)) == ("applicable", "reason")
    assert tuple(field.name for field in fields(H4IntervalDecision)) == (
        "lower", "upper", "threshold", "classification", "invariant_passed",
        "invariant_value", "invariant_limit", "invariant_detail",
        "status_if_other_invariants_eligible", "obligation",
    )
    assert classify_h4_interval(0.70, 0.80).classification == "support"
    assert classify_h4_interval(0.80, 0.80).classification == "boundary"
    assert classify_h4_interval(0.80, 0.90).classification == "no_support"
    assert classify_h4_interval(0.70, 0.90).classification == "crossing"
EXPECTED_APPLICABLE_ALLOWANCE_FIELDS = (
    "applicable", "dimension", "operands", "absolute_summands", "condition_numbers",
    "operation_counts", "solver_contribution", "invariant_scale", "final_allowance",
    "allowance_scale_ratio",
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

def _numerical_allowance(
    invariant: str = "h3_anchor_identity",
) -> H4ApplicableAllowance:
    operand = lambda label: H4AllowanceOperand(
        label, 0.0, 0.0, 0.0, (1.0,), (), False, 0.0, 0.0, 0.0,
    )
    comparison = 4096.0 * (
        3.0 * 2.220446049250313e-16
        / (1.0 - 3.0 * 2.220446049250313e-16)
    )
    anchor = invariant == "h3_anchor_identity"
    element = H4AllowanceElement(
        0, invariant,  # type: ignore[arg-type]
        "h4-anchor-h3-zero-control-v1" if anchor else "h4-coupled-T7-dz4-dm4-seed104729-v1",
        "adapter_to_oracle" if anchor else "solver_to_oracle",
        None if anchor else 0, None if anchor else "information", "value", (1,), 0,
        1.0, operand("left"), operand("right"), comparison, 0.0, 0.0,
        comparison, comparison, True, True,
    )
    count = dict(H4_ALLOWANCE_ELEMENT_COUNTS)[invariant]
    return H4ApplicableAllowance(
        True, invariant, "vfe4.h4.allowance-element-stream.v1", count, count,
        "a" * 64, 0.0, element, comparison, element, None, None, True, True,
    )


def _complete_measurements(lower: float, upper: float) -> dict[str, float]:
    return {
        "primary_seed_ratio_geometric_mean": (lower + upper) / 2.0,
        "primary_bootstrap_lower": lower,
        "primary_bootstrap_upper": upper,
        "primary_effect_threshold": 0.80,
        "primary_timed_ab_total": 110.0,
        "primary_timed_ba_total": 110.0,
        "maximum_solver_stopping_residual": 0.0,
        "maximum_allowance_scale_fraction": 0.0,
    }


def _complete_invariants(lower: float, upper: float) -> tuple[InvariantResult, ...]:
    values = [InvariantResult(name, True, 0.0, 1.0, "closed") for name in EXPECTED_INVARIANTS]
    if (lower, upper) == (0.80, 0.80):
        values[16] = InvariantResult("primary_effect_threshold", False, 0.80, 0.80, "bootstrap_interval_equals_threshold")
    elif upper <= 0.80:
        values[16] = InvariantResult("primary_effect_threshold", True, upper, 0.80, "bootstrap_interval_supports_effect")
    elif lower >= 0.80:
        values[16] = InvariantResult("primary_effect_threshold", False, lower, 0.80, "bootstrap_interval_excludes_support")
    else:
        values[16] = InvariantResult("primary_effect_threshold", False, lower, 0.80, "bootstrap_interval_crosses_threshold")
    return tuple(values)


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
    assert np.array_equal(np.asarray(m_factor.matrix)[:, 4:8], -clip(raw_a_m))
    joined = clip(np.concatenate((raw_a_z, raw_b), axis=1))
    assert np.array_equal(np.asarray(z_factor.matrix)[:, :4], -joined[:, :4])
    assert np.array_equal(np.asarray(z_factor.matrix)[:, 12:16], -joined[:, 4:])
    assert np.array_equal(np.asarray(observation.matrix)[:, 8:16], np.eye(8) + .05 * raw_g / max(1.0, np.linalg.norm(raw_g, 2)));
    assert np.array_equal(np.asarray(observation.target), target - offset)
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


@pytest.mark.parametrize("kind", ("coupled", "zero_control"))
def test_full_t31_pcg64_stream_never_restarts_and_pins_every_raw_draw(kind: str) -> None:
    seed, horizon = 155921, 31
    problem = make_h4_problem(seed=seed, kind=kind, horizon=horizon)  # type: ignore[arg-type]
    initial = problem.factor_schedule[0]
    assert np.array_equal(np.asarray(initial.matrix), np.column_stack((np.eye(8), np.zeros((8, problem.dimension - 8)))))
    assert initial.target == (0.0,) * 8 and np.array_equal(np.asarray(initial.covariance), np.eye(8))
    rng = np.random.Generator(np.random.PCG64(seed))
    for time in range(1, horizon + 1):
        expected = _independent_draws_from_rng(rng)
        factors = problem.factor_schedule[1 + 3 * (time - 1):1 + 3 * time]
        draws = tuple(sorted((draw for factor in factors for draw in factor.raw_draws), key=lambda draw: draw.draw_index))
        assert tuple(draw.draw_index for draw in draws) == tuple(range(11 * (time - 1), 11 * time))
        for draw, value in zip(draws, expected, strict=True):
            assert np.array_equal(np.asarray(draw.values).reshape(draw.shape), value)
        raw_a_m, raw_a_z, raw_b, c_m, c_z, r_m, r_z, raw_g, offset, observation_noise, target = expected
        m_factor, z_factor, observation = factors
        clip = lambda value: value * min(1.0, 0.65 / np.linalg.norm(value, 2))
        joined = clip(np.concatenate((raw_a_z, raw_b), axis=1))
        assert np.array_equal(np.asarray(m_factor.matrix)[:, m_factor.normalized_coordinate_indices], np.eye(4))
        expected_m_parent = -clip(raw_a_m) if kind == "coupled" else np.zeros((4, 4))
        expected_z_parent = -joined if kind == "coupled" else np.zeros((4, 8))
        assert np.array_equal(np.asarray(m_factor.matrix)[:, m_factor.parent_coordinate_indices], expected_m_parent)
        assert m_factor.target == tuple(c_m) and np.array_equal(np.asarray(m_factor.covariance), np.diag(r_m))
        assert np.array_equal(np.asarray(z_factor.matrix)[:, z_factor.normalized_coordinate_indices], np.eye(4))
        assert np.array_equal(np.asarray(z_factor.matrix)[:, z_factor.parent_coordinate_indices], expected_z_parent)
        assert z_factor.target == tuple(c_z) and np.array_equal(np.asarray(z_factor.covariance), np.diag(r_z))
        assert np.array_equal(np.asarray(observation.matrix)[:, observation.parent_coordinate_indices], np.eye(8) + .05 * raw_g / max(1.0, np.linalg.norm(raw_g, 2)))
        assert observation.target == tuple(target - offset) and np.array_equal(np.asarray(observation.covariance), np.diag(observation_noise))
        for factor in factors:
            supported = set(factor.normalized_coordinate_indices) | set(factor.parent_coordinate_indices)
            assert np.all(np.asarray(factor.matrix)[:, tuple(index for index in range(problem.dimension) if index not in supported)] == 0.0)


def _independent_draws_from_rng(rng: np.random.Generator):
    return (
        rng.standard_normal((4,4)), rng.standard_normal((4,4)), rng.standard_normal((4,4)),
        rng.uniform(-.25,.25,4), rng.uniform(-.25,.25,4), rng.uniform(.5,1.5,4),
        rng.uniform(.5,1.5,4), rng.standard_normal((8,8)), rng.uniform(-.25,.25,8),
        rng.uniform(.75,1.25,8), rng.uniform(-1.,1.,8),
    )


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
    recomputed_wrong_id = json.loads(bytes_one); recomputed_wrong_id["problem"]["problem_id"] = "h4-coupled-T7-dz4-dm4-seed130363-v0"
    recomputed_wrong_id["canonical_sha256"] = hashlib.sha256(b"vfe4.h4.neutral-problem.v1\0" + json.dumps(recomputed_wrong_id["problem"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    with pytest.raises(ValueError): parse_h4_problem_bytes(json.dumps(recomputed_wrong_id, sort_keys=True, separators=(",", ":")).encode())
    recomputed_wrong_draw = json.loads(bytes_one); recomputed_wrong_draw["problem"]["factor_schedule"][1]["raw_draws"][0]["values"][0] += 1.0
    recomputed_wrong_draw["canonical_sha256"] = hashlib.sha256(b"vfe4.h4.neutral-problem.v1\0" + json.dumps(recomputed_wrong_draw["problem"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    with pytest.raises(ValueError): parse_h4_problem_bytes(json.dumps(recomputed_wrong_draw, sort_keys=True, separators=(",", ":")).encode())
    recomputed_wrong_factor = json.loads(bytes_one); recomputed_wrong_factor["problem"]["factor_schedule"][1]["matrix"][0][4] += .01
    recomputed_wrong_factor["canonical_sha256"] = hashlib.sha256(b"vfe4.h4.neutral-problem.v1\0" + json.dumps(recomputed_wrong_factor["problem"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    with pytest.raises(ValueError): parse_h4_problem_bytes(json.dumps(recomputed_wrong_factor, sort_keys=True, separators=(",", ":")).encode())
    for broken in (
        bytes_one.replace(b'"schema_version"', b'"schema_version","schema_version"', 1),
        bytes_one.replace(b'"factor_id"', b'"unknown"', 1),
        bytes_one.replace(b'"draw_index"', b'"unknown_draw"', 1),
        b'{"schema_version":"h4-neutral-problem-v1","canonical_sha256":"' + coupled.canonical_sha256.encode() + b'","problem":NaN}',
        b" " + bytes_one,
    ):
        with pytest.raises(ValueError): parse_h4_problem_bytes(broken)


@pytest.mark.parametrize("kind", ("coupled", "zero_control"))
@pytest.mark.parametrize(
    "path",
    (
        ("factor_schedule", 0, "target", 0),
        ("factor_schedule", 0, "matrix", 0, 1),
    ),
)
def test_scaled_parser_rejects_signed_zero_provenance_forgery(kind: str, path: tuple[object, ...]) -> None:
    envelope = json.loads(canonical_h4_problem_bytes(make_h4_problem(seed=130363, kind=kind, horizon=7)))
    target = envelope["problem"]
    for part in path:
        target = target[part]
    target = -0.0
    container = envelope["problem"]
    for part in path[:-1]:
        container = container[part]
    container[path[-1]] = target
    core = envelope["problem"]
    envelope["canonical_sha256"] = hashlib.sha256(
        b"vfe4.h4.neutral-problem.v1\0" + json.dumps(
            core, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    forged = json.dumps(envelope, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    assert b"-0.0" in forged
    with pytest.raises(ValueError, match="frozen PCG64 provenance"):
        parse_h4_problem_bytes(forged)


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
        if fixture.kind == "coupled":
            assert np.allclose(J, np.asarray(((2.96, 0.0, -2.8, 1.68), (0.0, 2.77777777777778, 0.0, -2.22222222222222), (-2.8, 0.0, 5.5625, -2.4), (1.68, -2.22222222222222, -2.4, 5.78027777777778))), rtol=0.0, atol=1.0e-14)
            assert np.allclose(h, np.asarray((0.0, 0.0, 1.71875, 0.3125)), rtol=0.0, atol=1.0e-14)
        else:
            assert np.allclose(J, np.asarray(((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 5.5625, 0.0), (0.0, 0.0, 0.0, 4.34027777777778))), rtol=0.0, atol=1.0e-14)
            assert np.allclose(h, np.asarray((0.0, 0.0, 0.625, -1.09375)), rtol=0.0, atol=1.0e-14)
        expected_J = sum(np.outer(np.asarray(f.row), np.asarray(f.row)) / f.variance for f in fixture.factors)
        expected_h = sum(f.target * np.asarray(f.row) / f.variance for f in fixture.factors)
        expected_c = sum(-.5 * (f.target * f.target / f.variance + math.log(2.0 * math.pi * f.variance)) for f in fixture.factors)
        assert np.allclose(J, expected_J) and np.allclose(h, expected_h) and math.isclose(c, expected_c)
        expected_log_z = expected_c + .5 * expected_h @ np.linalg.solve(expected_J, expected_h) - .5 * np.linalg.slogdet(expected_J)[1] + 2.0 * math.log(2.0 * math.pi)
        assert math.isclose(log_z, expected_log_z)
        oracle = evaluate_h3_posterior_oracle(
            (H3_COUPLED_FIXTURE_PATH if fixture.kind == "coupled" else H3_ZERO_CONTROL_FIXTURE_PATH).read_bytes(),
            expected_fixture_id=fixture.fixture_id,
        )
        assert fixture.reference_log_evidence == (-2.6536596233553 if fixture.kind == "coupled" else None)
        if fixture.kind == "coupled":
            assert math.isclose(log_z, -2.6536596233553, rel_tol=0.0, abs_tol=1.0e-12)
        left = scalar_allowance(4, value=log_z, absolute_sum=float(oracle.diagnostics["log_evidence_absolute_summand_accumulation"]), kappas=tuple(oracle.diagnostics["log_evidence_operand_kappas"]), optimized=False)
        right = scalar_allowance(4, value=oracle.log_evidence, absolute_sum=float(oracle.diagnostics["log_evidence_absolute_summand_accumulation"]), kappas=tuple(oracle.diagnostics["log_evidence_operand_kappas"]), optimized=False)
        assert abs(log_z - oracle.log_evidence) <= pair_allowance(4, left=log_z, right=oracle.log_evidence, left_allowance=left, right_allowance=right)


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
    numerical = _numerical_allowance()
    allowances = {name: numerical if name == "h3_anchor_identity" else H4InapplicableAllowance(False, "not_evaluated_after_decisive_h3_anchor_failure") for name in EXPECTED_ALLOWANCES}
    result = H4GateResult("H4", GateStatus.FAIL, measurements, invariants, allowances, ())
    assert H4_INVARIANT_NAMES == EXPECTED_INVARIANTS and H4_MEASUREMENT_NAMES == EXPECTED_MEASUREMENTS and H4_PRIMARY_MEASUREMENTS_UNAVAILABLE_AFTER_ANCHOR_FAIL == EXPECTED_UNAVAILABLE and H4_ALLOWANCE_INVARIANT_NAMES == EXPECTED_ALLOWANCES
    assert tuple(result.measurements) == EXPECTED_MEASUREMENTS
    assert tuple(name for name, value in result.measurements.items() if value is None) == EXPECTED_UNAVAILABLE
    with pytest.raises(ValueError):
        H4GateResult("H4", GateStatus.FAIL, {name: 0.0 for name in EXPECTED_MEASUREMENTS}, invariants, allowances, ())
    bad_allowances = dict(allowances); bad_allowances["h3_anchor_identity"] = {"applicable": True}
    with pytest.raises(ValueError): H4GateResult("H4", GateStatus.FAIL, measurements, invariants, bad_allowances, ())


def test_gate_rejects_contradictory_pass_fail_inconclusive_and_allowance_shapes() -> None:
    measurements = _complete_measurements(.70, .79)
    passing = _complete_invariants(.70, .79)
    allowances = {name: _numerical_allowance(name) for name in EXPECTED_ALLOWANCES}
    assert type(H4GateResult("H4", GateStatus.PASS, measurements, passing, allowances, ()).allowances_by_invariant["h3_anchor_identity"]) is H4ApplicableAllowance
    failed = list(passing); failed[8] = InvariantResult("exact_posterior_gap_equivalence", False, 2.0, 1.0, "miss")
    assert H4GateResult("H4", GateStatus.FAIL, measurements, tuple(failed), allowances, ()).status is GateStatus.FAIL
    for index in (13, 14, 15):
        forbidden = list(passing); forbidden[index] = InvariantResult(EXPECTED_INVARIANTS[index], False, 2.0, 1.0, "eligibility_miss")
        with pytest.raises(ValueError): H4GateResult("H4", GateStatus.FAIL, measurements, tuple(forbidden), allowances, ())
    crossing_measurements = _complete_measurements(.70, .90)
    crossing = _complete_invariants(.70, .90)
    assert H4GateResult("H4", GateStatus.INCONCLUSIVE, crossing_measurements, crossing, allowances, ("primary_effect_threshold: bootstrap_interval_crosses_threshold",)).status is GateStatus.INCONCLUSIVE
    boundary_measurements = _complete_measurements(.80, .80)
    boundary = _complete_invariants(.80, .80)
    assert H4GateResult("H4", GateStatus.INCONCLUSIVE, boundary_measurements, boundary, allowances, ("primary_effect_threshold: bootstrap_interval_equals_threshold",)).status is GateStatus.INCONCLUSIVE
    no_support = _complete_invariants(.90, 1.10)
    no_support_measurements = _complete_measurements(.90, 1.10)
    assert H4GateResult("H4", GateStatus.FAIL, no_support_measurements, tuple(no_support), allowances, ()).status is GateStatus.FAIL
    ambiguous = list(passing); ambiguous[5] = InvariantResult("scaled_condition_envelope", False, None, None, "not_evaluated_after_inconclusive_eligibility")
    ambiguous[8] = InvariantResult("exact_posterior_gap_equivalence", False, 2.0, 1.0, "miss")
    assert H4GateResult("H4", GateStatus.INCONCLUSIVE, measurements, tuple(ambiguous), allowances, ("scaled_condition_envelope: not_evaluated_after_inconclusive_eligibility",)).status is GateStatus.INCONCLUSIVE
    inapplicable = {name: _numerical_allowance(name) for name in EXPECTED_ALLOWANCES}; inapplicable["exact_posterior_gap_equivalence"] = H4InapplicableAllowance(False, "not_evaluated_after_decisive_h3_anchor_failure")
    inconclusive_equivalence = list(passing); inconclusive_equivalence[8] = InvariantResult("exact_posterior_gap_equivalence", False, None, None, "not_evaluated_after_inconclusive_eligibility")
    with pytest.raises(ValueError): H4GateResult("H4", GateStatus.INCONCLUSIVE, measurements, tuple(inconclusive_equivalence), inapplicable, ("exact_posterior_gap_equivalence: not_evaluated_after_inconclusive_eligibility",))
    with pytest.raises(ValueError): H4GateResult("H4", GateStatus.PASS, measurements, tuple(failed), allowances, ())
    with pytest.raises(ValueError): H4GateResult("H4", GateStatus.FAIL, measurements, passing, allowances, ())
    with pytest.raises(ValueError): H4GateResult("H4", GateStatus.INCONCLUSIVE, measurements, tuple(failed), allowances, ())
    bad = {name: _numerical_allowance(name) for name in EXPECTED_ALLOWANCES}; bad["h3_anchor_identity"] = {"applicable": True, "dimension": 4}
    with pytest.raises(ValueError): H4GateResult("H4", GateStatus.PASS, measurements, passing, bad, ())
    with pytest.raises(ValueError):
        replace(_numerical_allowance(), maximum_allowance_scale_ratio=.5)


@pytest.mark.parametrize(
    ("lower", "upper", "status", "obligations"),
    (
        (.70, .79, GateStatus.PASS, ()),
        (.80, .80, GateStatus.INCONCLUSIVE, ("primary_effect_threshold: bootstrap_interval_equals_threshold",)),
        (.81, .90, GateStatus.FAIL, ()),
    ),
)
def test_gate_interval_classifier_preserves_point_interval_semantics(
    lower: float,
    upper: float,
    status: GateStatus,
    obligations: tuple[str, ...],
) -> None:
    allowances = {name: _numerical_allowance(name) for name in EXPECTED_ALLOWANCES}
    assert H4GateResult(
        "H4", status, _complete_measurements(lower, upper),
        _complete_invariants(lower, upper), allowances, obligations,
    ).status is status


def test_gate_rejects_interval_classifier_and_comparison_contradictions() -> None:
    allowances = {name: _numerical_allowance(name) for name in EXPECTED_ALLOWANCES}
    support_measurements = _complete_measurements(.70, .79)
    support_invariants = list(_complete_invariants(.70, .79))
    for threshold in (None, .79, .81):
        bad = dict(support_measurements); bad["primary_effect_threshold"] = threshold
        with pytest.raises(ValueError):
            H4GateResult("H4", GateStatus.PASS, bad, tuple(support_invariants), allowances, ())
    for lower, upper in ((.90, .80), (0.0, .80)):
        bad = _complete_measurements(.70, .79); bad.update({"primary_bootstrap_lower": lower, "primary_bootstrap_upper": upper})
        with pytest.raises(ValueError):
            H4GateResult("H4", GateStatus.PASS, bad, tuple(support_invariants), allowances, ())
    wrong_interval = list(support_invariants)
    wrong_interval[16] = InvariantResult("primary_effect_threshold", False, .70, .80, "bootstrap_interval_crosses_threshold")
    with pytest.raises(ValueError):
        H4GateResult("H4", GateStatus.PASS, support_measurements, tuple(wrong_interval), allowances, ())
    unresolved = list(_complete_invariants(.90, 1.10))
    unresolved[8] = InvariantResult("exact_posterior_gap_equivalence", False, None, None, "not_evaluated_after_inconclusive_eligibility")
    with pytest.raises(ValueError):
        H4GateResult("H4", GateStatus.FAIL, _complete_measurements(.90, 1.10), tuple(unresolved), allowances, ())
    contradictory = list(_complete_invariants(.70, .79))
    contradictory[8] = InvariantResult("exact_posterior_gap_equivalence", False, 1.0, 1.0, "not_a_miss")
    with pytest.raises(ValueError):
        H4GateResult("H4", GateStatus.INCONCLUSIVE, support_measurements, tuple(contradictory), allowances, ("exact_posterior_gap_equivalence: not_evaluated_after_inconclusive_eligibility",))


def test_gate_requires_exact_nonanchor_missing_interval_sentinels() -> None:
    allowances = {name: _numerical_allowance(name) for name in EXPECTED_ALLOWANCES}
    measurements = _complete_measurements(.70, .79)
    measurements.update({
        "primary_seed_ratio_geometric_mean": None,
        "primary_bootstrap_lower": None,
        "primary_bootstrap_upper": None,
    })
    invariants = list(_complete_invariants(.70, .79))
    unavailable = InvariantResult(
        "primary_seed_level_inference", False, None, None,
        "not_evaluated_after_inconclusive_eligibility",
    )
    invariants[15] = unavailable
    invariants[16] = InvariantResult(
        "primary_effect_threshold", False, None, None,
        "not_evaluated_after_inconclusive_eligibility",
    )
    obligations = (
        "primary_seed_level_inference: not_evaluated_after_inconclusive_eligibility",
        "primary_effect_threshold: not_evaluated_after_inconclusive_eligibility",
    )
    assert H4GateResult(
        "H4", GateStatus.INCONCLUSIVE, measurements, tuple(invariants), allowances, obligations,
    ).status is GateStatus.INCONCLUSIVE
    finite_threshold = list(invariants)
    finite_threshold[16] = InvariantResult(
        "primary_effect_threshold", True, .79, .80, "bootstrap_interval_supports_effect",
    )
    with pytest.raises(ValueError):
        H4GateResult(
            "H4", GateStatus.INCONCLUSIVE, measurements, tuple(finite_threshold), allowances, obligations,
        )


def test_gate_rejects_anchor_unavailable_sentinel_outside_anchor_fail() -> None:
    allowances = {name: _numerical_allowance(name) for name in EXPECTED_ALLOWANCES}
    invariants = list(_complete_invariants(.70, .79))
    invariants[8] = InvariantResult(
        "exact_posterior_gap_equivalence", False, None, None,
        "not_evaluated_after_decisive_h3_anchor_failure",
    )
    with pytest.raises(ValueError, match="reserved for anchor FAIL"):
        H4GateResult(
            "H4", GateStatus.INCONCLUSIVE, _complete_measurements(.70, .79),
            tuple(invariants), allowances,
            ("exact_posterior_gap_equivalence: not_evaluated_after_inconclusive_eligibility",),
        )


def test_gate_rejects_decisive_anchor_miss_supplied_as_inconclusive() -> None:
    allowances = {name: _numerical_allowance(name) for name in EXPECTED_ALLOWANCES}
    invariants = list(_complete_invariants(.70, .79))
    invariants[0] = InvariantResult("h3_anchor_identity", False, 2.0, 1.0, "anchor_miss")
    with pytest.raises(ValueError, match="decisive anchor miss requires exact early FAIL"):
        H4GateResult(
            "H4", GateStatus.INCONCLUSIVE, _complete_measurements(.70, .79),
            tuple(invariants), allowances,
            ("h3_anchor_identity: not_evaluated_after_inconclusive_eligibility",),
        )


@pytest.mark.parametrize(("value", "limit"), ((-1.0, 1.0), (-2.0, -1.0)))
def test_gate_rejects_negative_anchor_comparison_values(value: float, limit: float) -> None:
    allowances = {name: _numerical_allowance(name) for name in EXPECTED_ALLOWANCES}
    invariants = list(_complete_invariants(.70, .79))
    invariants[0] = InvariantResult("h3_anchor_identity", True, value, limit, "anchor_pass")
    with pytest.raises(ValueError, match="residual and limit must be nonnegative"):
        H4GateResult(
            "H4", GateStatus.PASS, _complete_measurements(.70, .79),
            tuple(invariants), allowances, (),
        )


@pytest.mark.parametrize(
    ("passed", "value", "limit", "status", "obligations"),
    (
        (True, 2.0, 1.0, GateStatus.PASS, ()),
        (
            False, 1.0, 1.0, GateStatus.INCONCLUSIVE,
            ("h3_anchor_identity: not_evaluated_after_inconclusive_eligibility",),
        ),
    ),
)
def test_gate_rejects_anchor_pass_miss_flag_contradictions(
    passed: bool,
    value: float,
    limit: float,
    status: GateStatus,
    obligations: tuple[str, ...],
) -> None:
    allowances = {name: _numerical_allowance(name) for name in EXPECTED_ALLOWANCES}
    invariants = list(_complete_invariants(.70, .79))
    invariants[0] = InvariantResult("h3_anchor_identity", passed, value, limit, "anchor")
    with pytest.raises(ValueError):
        H4GateResult(
            "H4", status, _complete_measurements(.70, .79),
            tuple(invariants), allowances, obligations,
        )


def test_gate_rejects_reserved_anchor_sentinel_on_invariant_zero() -> None:
    invariants = tuple(
        InvariantResult(
            name,
            False,
            2.0 if index == 0 else None,
            1.0 if index == 0 else None,
            "not_evaluated_after_decisive_h3_anchor_failure",
        )
        for index, name in enumerate(EXPECTED_INVARIANTS)
    )
    measurements = {name: None for name in EXPECTED_MEASUREMENTS}
    measurements.update({
        "primary_effect_threshold": .80,
        "maximum_solver_stopping_residual": .0,
        "maximum_allowance_scale_fraction": .0,
    })
    allowances = {
        name: (
            _numerical_allowance(name)
            if name == "h3_anchor_identity"
            else H4InapplicableAllowance(
                False, "not_evaluated_after_decisive_h3_anchor_failure",
            )
        )
        for name in EXPECTED_ALLOWANCES
    }
    with pytest.raises(ValueError, match="reserved sentinel is forbidden on invariant zero"):
        H4GateResult("H4", GateStatus.FAIL, measurements, invariants, allowances, ())


def test_gate_preserves_ordinary_unresolved_anchor_as_inconclusive() -> None:
    invariants = list(_complete_invariants(.70, .79))
    invariants[0] = InvariantResult(
        "h3_anchor_identity", False, None, None,
        "not_evaluated_after_inconclusive_eligibility",
    )
    allowances = {name: _numerical_allowance(name) for name in EXPECTED_ALLOWANCES}
    allowances["h3_anchor_identity"] = H4InapplicableAllowance(
        False, "not_evaluated_after_inconclusive_eligibility",
    )
    obligations = (
        "h3_anchor_identity: not_evaluated_after_inconclusive_eligibility",
    )
    assert H4GateResult(
        "H4", GateStatus.INCONCLUSIVE, _complete_measurements(.70, .79),
        tuple(invariants), allowances, obligations,
    ).status is GateStatus.INCONCLUSIVE


def test_every_task_one_record_has_frozen_fields_and_positive_negative_validation() -> None:
    assert tuple(field.name for field in fields(H4RawDraw)) == ("draw_index", "name", "shape", "values")
    assert tuple(field.name for field in fields(H4AffineGaussianFactor)) == ("factor_id", "role", "time_index", "normalized_coordinate_indices", "parent_coordinate_indices", "matrix", "target", "covariance", "raw_draws")
    assert tuple(field.name for field in fields(H4NeutralProblem)) == ("problem_id", "source_kind", "seed", "kind", "horizon", "d_z", "d_m", "dimension", "coordinate_order", "factor_schedule", "canonical_sha256")
    assert tuple(field.name for field in fields(H4SolveProtocol)) == ("protocol_id", "dtype", "device", "factor_passes", "solver_relative_budget", "stopping_rule")
    assert tuple(field.name for field in fields(H4SelectedMoment)) == ("name", "mean", "covariance")
    assert tuple(field.name for field in fields(H4TerminalLaw)) == ("arm", "h", "J", "mean", "selected_moments", "complete_objective", "stopping_residual")
    assert tuple(field.name for field in fields(H4NativeInformationState)) == ("h", "J", "mean", "complete_objective")
    assert tuple(field.name for field in fields(H4NativeMomentState)) == ("mean", "covariance", "complete_objective")
    assert tuple(field.name for field in fields(H4SolverResult)) == ("problem_id", "problem_sha256", "arm", "protocol_id", "factor_count", "native_information", "native_moment")
    assert tuple(field.name for field in fields(H4TimingRecord)) == ("problem_id", "problem_index", "horizon_index", "seed_index", "kind_index", "seed", "kind", "horizon", "repetition_index", "pair_index", "order", "information_nanoseconds", "moment_nanoseconds")
    assert tuple(field.name for field in fields(H4OperationRecord)) == ("problem_id", "arm", "operation", "operand_shapes", "result_shape", "count")
    assert tuple(field.name for field in fields(H4MemoryRecord)) == ("problem_id", "arm", "python_peak_bytes", "process_working_set_delta_bytes", "unavailable_fields")
    assert tuple(field.name for field in fields(H4GateResult)) == ("gate", "status", "measurements", "invariants", "allowances_by_invariant", "obligations")
    assert H4SolveProtocol().solver_relative_budget == 1e-9
    with pytest.raises(ValueError): H4SolveProtocol(solver_relative_budget=1e-8)
    with pytest.raises(ValueError): H4SolveProtocol(factor_passes=1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError): H4SolveProtocol(factor_passes=True)  # type: ignore[arg-type]
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
    timing_values = ["p", 0, 0, 0, 0, 104729, "coupled", 7, 0, 3, "moment_then_information", 1, 1]
    for index in (1, 2, 3, 4, 5, 7, 8, 9, 11, 12):
        invalid = list(timing_values); invalid[index] = float(invalid[index])
        with pytest.raises(ValueError): H4TimingRecord(*invalid)  # type: ignore[arg-type]
    for index in (1, 2, 3, 4, 8):
        invalid = list(timing_values); invalid[index] = bool(invalid[index])
        with pytest.raises(ValueError): H4TimingRecord(*invalid)  # type: ignore[arg-type]
    operation = H4OperationRecord("p", "information", "cholesky", ((2, 2),), (2, 2), 1)
    assert operation.count == 1
    with pytest.raises(ValueError): H4OperationRecord("p", "information", "cholesky", ((0, 2),), (2, 2), 1)
    memory = H4MemoryRecord("p", "moment", None, -5, ("python_peak_bytes",))
    assert memory.process_working_set_delta_bytes == -5
    with pytest.raises(ValueError): H4MemoryRecord("p", "moment", None, -5, ())


def test_public_state_constructors_retain_full_spd_validation() -> None:
    indefinite = ((1.0, 2.0), (2.0, 1.0))
    mean = (0.0, 0.0)
    with pytest.raises(ValueError, match="positive definite"):
        H4SelectedMoment("initial", mean, indefinite)
    with pytest.raises(ValueError, match="positive definite"):
        H4NativeMomentState(mean, indefinite, 0.0)
    with pytest.raises(ValueError, match="positive definite"):
        H4NativeInformationState(mean, indefinite, mean, 0.0)

    selected = (
        H4SelectedMoment("initial", mean, ((1.0, 0.0), (0.0, 1.0))),
        H4SelectedMoment("terminal", mean, ((1.0, 0.0), (0.0, 1.0))),
        H4SelectedMoment("observation[1]", mean, ((1.0, 0.0), (0.0, 1.0))),
    )
    with pytest.raises(ValueError, match="positive definite"):
        H4TerminalLaw("information", mean, indefinite, mean, selected, 0.0, 0.0)

    for name in (
        "_H4SpdProof",
        "_h4_native_information_from_proven_spd",
        "_h4_native_moment_from_proven_spd",
        "_h4_selected_moment_from_proven_spd",
        "_h4_terminal_law_from_proven_spd",
    ):
        assert name not in h4_types_module.__all__


def test_gate_requires_exact_base_invariant_result_records() -> None:
    allowances = {name: _numerical_allowance(name) for name in EXPECTED_ALLOWANCES}
    invariants = list(_complete_invariants(.70, .79))

    class InvariantSubclass(InvariantResult):
        pass

    invariants[0] = InvariantSubclass("h3_anchor_identity", True, 0.0, 1.0, "closed")
    with pytest.raises(ValueError, match="exact InvariantResult"):
        H4GateResult("H4", GateStatus.PASS, _complete_measurements(.70, .79), tuple(invariants), allowances, ())


def test_scaled_problem_rejects_wrong_id_duplicate_draw_and_unordered_metadata() -> None:
    problem = make_h4_problem(seed=104729, kind="coupled", horizon=7)
    with pytest.raises(ValueError): replace(problem, problem_id="h4-coupled-T7-dz4-dm4-seed104729-v0")
    m_factor = problem.factor_schedule[1]
    with pytest.raises(ValueError):
        replace(m_factor, raw_draws=(m_factor.raw_draws[0], replace(m_factor.raw_draws[1], draw_index=m_factor.raw_draws[0].draw_index), m_factor.raw_draws[2]))
    with pytest.raises(ValueError): replace(m_factor, parent_coordinate_indices=(5, 4, 6, 7))
