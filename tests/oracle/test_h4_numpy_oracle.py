from __future__ import annotations

import hashlib
import inspect
import json
import math
from dataclasses import replace

import numpy as np
import pytest

from vfe4.generative.reference_h4 import h4_anchor_from_h3, make_h4_problem
from vfe4.types.h4 import canonical_h4_problem_bytes
from vfe4.validation.h3_fixture import (
    H3_COUPLED_FIXTURE_PATH,
    H3_ZERO_CONTROL_FIXTURE_PATH,
    parse_h3_fixture_bytes,
)
from verification.numpy_oracles import h4_gaussian
from verification.numpy_oracles.h4_gaussian import (
    H4OracleEvaluation,
    evaluate_h4_oracle,
    reverse_kl_to_h4_oracle,
)


_H4_PROBLEM_DOMAIN = b"vfe4.h4.neutral-problem.v1\x00"


def _rehashed_payload(payload: bytes, mutation: str) -> bytes:
    envelope = json.loads(payload)
    core = envelope["problem"]
    schedule = core["factor_schedule"]
    if mutation == "raw_index":
        schedule[1]["raw_draws"][0]["draw_index"] += 1000
    elif mutation == "raw_name":
        schedule[1]["raw_draws"][0]["name"] = "wrong-draw"
    elif mutation == "raw_shape":
        schedule[1]["raw_draws"][0]["shape"] = [16]
    elif mutation == "raw_value":
        schedule[1]["raw_draws"][0]["values"][0] += 1.0e-12
    elif mutation == "raw_order":
        schedule[1]["raw_draws"][0], schedule[1]["raw_draws"][1] = (
            schedule[1]["raw_draws"][1], schedule[1]["raw_draws"][0],
        )
    elif mutation == "coordinate_order":
        core["coordinate_order"][0], core["coordinate_order"][1] = (
            core["coordinate_order"][1], core["coordinate_order"][0],
        )
    elif mutation == "factor_role":
        schedule[1]["role"] = "initial"
    elif mutation == "factor_time":
        schedule[1]["time_index"] = 2
    elif mutation == "factor_metadata":
        schedule[1]["parent_coordinate_indices"] = [0, 4, 5, 6, 7]
    elif mutation == "h3_factor_id":
        schedule[0]["factor_id"] = "not-z0-prior"
    elif mutation == "h3_coordinate_order":
        core["coordinate_order"][0], core["coordinate_order"][1] = (
            core["coordinate_order"][1], core["coordinate_order"][0],
        )
    elif mutation == "h3_factor_metadata":
        schedule[4]["parent_coordinate_indices"] = [0, 2]
    else:  # pragma: no cover - closed test helper universe
        raise AssertionError(f"unknown mutation: {mutation}")
    canonical_core = json.dumps(
        core, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    envelope["canonical_sha256"] = hashlib.sha256(
        _H4_PROBLEM_DOMAIN + canonical_core,
    ).hexdigest()
    return json.dumps(
        envelope, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def test_h4_oracle_module_is_independent_and_closes_both_h3_routes() -> None:
    source = inspect.getsource(h4_gaussian)
    assert "import torch" not in source
    assert "from vfe4" not in source and "import vfe4" not in source
    for path, fixture_id in (
        (H3_COUPLED_FIXTURE_PATH, "h3-coupled-v1"),
        (H3_ZERO_CONTROL_FIXTURE_PATH, "h3-zero-control-v1"),
    ):
        fixture = parse_h3_fixture_bytes(path.read_bytes(), expected_fixture_id=fixture_id)
        payload = canonical_h4_problem_bytes(h4_anchor_from_h3(fixture))
        oracle = evaluate_h4_oracle(payload)
        assert type(oracle) is H4OracleEvaluation
        assert oracle.dimension == 4 and oracle.source_kind == "h3_anchor"
        assert oracle.route_agreement.eligible
        assert oracle.operand_evidence == (
            oracle.route_agreement.canonical_operand,
            oracle.route_agreement.predictive_operand,
        )
        assert tuple(item.name for item in oracle.selected_moments) == (
            "initial", "terminal", "observation[1]",
        )
        kl = reverse_kl_to_h4_oracle(
            oracle, mean=oracle.mean, precision=oracle.precision,
        )
        assert abs(kl.value) <= 1.0e-12


def test_h4_scaled_oracle_uses_global_scatter_and_oriented_kl() -> None:
    problem = make_h4_problem(seed=104729, kind="coupled", horizon=7)
    oracle = evaluate_h4_oracle(canonical_h4_problem_bytes(problem))
    assert oracle.dimension == 64
    assert len(oracle.innovation_diagnostics) == 7
    assert tuple(item.name for item in oracle.selected_moments) == (
        "initial", "terminal", *(f"observation[{time}]" for time in range(1, 8)),
    )
    assert oracle.selected_moments[0].coordinate_indices == tuple(range(8))
    assert oracle.selected_moments[1].coordinate_indices == tuple(range(56, 64))
    perturbed = list(oracle.mean)
    perturbed[0] += 0.25
    kl = reverse_kl_to_h4_oracle(
        oracle, mean=tuple(perturbed), precision=oracle.precision,
    )
    assert kl.value > 0.0 and kl.quadratic_mean_term > 0.0
    assert not hasattr(np.linalg, "_h4_inv_called")


def test_h4_canonical_route_counts_executable_accumulations_exactly() -> None:
    oracle = evaluate_h4_oracle(canonical_h4_problem_bytes(
        make_h4_problem(seed=104729, kind="coupled", horizon=7),
    ))
    counts = dict(oracle.route_agreement.canonical_operand.operation_counts)
    factor_count = len(oracle.factor_ids)
    dimension = oracle.dimension
    assert tuple(counts) == (
        "factor_covariance_cholesky", "factor_triangular_solves",
        "factor_assembly_matmuls", "factor_quadratics",
        "factor_logdet_reductions", "factor_J_sum_reduction",
        "factor_h_sum_reduction", "factor_c_scalar_combinations",
        "factor_c_sum_reduction", "posterior_precision_symmetrization",
        "posterior_precision_cholesky", "posterior_natural_solve",
        "posterior_quadratic", "posterior_logdet_reduction",
        "route_sum_reduction",
    )
    assert counts["factor_J_sum_reduction"] == factor_count * dimension * dimension
    assert counts["factor_h_sum_reduction"] == factor_count * dimension
    assert counts["factor_c_scalar_combinations"] == 4 * factor_count
    assert counts["factor_c_sum_reduction"] == factor_count
    assert counts["posterior_precision_symmetrization"] == 2 * dimension * dimension
    assert tuple(label for label, _ in oracle.route_agreement.predictive_operand.operation_counts) == (
        "affine_propagation_matmuls", "innovation_assembly",
        "innovation_cholesky", "innovation_triangular_solves",
        "innovation_quadratics", "innovation_logdet_reductions",
        "kalman_gain_solves", "mean_updates", "covariance_updates",
        "route_sum_reduction",
    )


@pytest.mark.parametrize(
    ("horizon", "expected_canonical_depth", "expected_predictive_depth"),
    (
        (7, 29_290, 7_097),
        (15, 115_458, 20_169),
        (31, 459_826, 64_745),
    ),
)
def test_h4_scaled_route_allowances_use_longest_scalar_dependency_depth(
    horizon: int,
    expected_canonical_depth: int,
    expected_predictive_depth: int,
) -> None:
    oracle = evaluate_h4_oracle(canonical_h4_problem_bytes(
        make_h4_problem(seed=104729, kind="coupled", horizon=horizon),
    ))
    canonical = oracle.route_agreement.canonical_operand
    predictive = oracle.route_agreement.predictive_operand
    assert (canonical.rounding_depth, predictive.rounding_depth) == (
        expected_canonical_depth, expected_predictive_depth,
    )
    assert sum(count for _, count in canonical.operation_counts) > canonical.rounding_depth
    assert sum(count for _, count in predictive.operation_counts) > predictive.rounding_depth
    assert predictive.condition_numbers[:2] == (1.0, 1.0)
    assert oracle.route_agreement.eligible


def test_h4_anchor_route_depths_follow_the_same_source_level_recurrences() -> None:
    for path, fixture_id in (
        (H3_COUPLED_FIXTURE_PATH, "h3-coupled-v1"),
        (H3_ZERO_CONTROL_FIXTURE_PATH, "h3-zero-control-v1"),
    ):
        fixture = parse_h3_fixture_bytes(path.read_bytes(), expected_fixture_id=fixture_id)
        oracle = evaluate_h4_oracle(canonical_h4_problem_bytes(h4_anchor_from_h3(fixture)))
        assert (
            oracle.route_agreement.canonical_operand.rounding_depth,
            oracle.route_agreement.predictive_operand.rounding_depth,
        ) == (139, 111)
        assert oracle.route_agreement.predictive_operand.condition_numbers[:3] == (
            1.0, 1.0, 1.0,
        )


@pytest.mark.parametrize("malformed_depth", (0, -1, True, 1.0))
def test_h4_operand_rejects_malformed_rounding_depth(malformed_depth: object) -> None:
    oracle = evaluate_h4_oracle(canonical_h4_problem_bytes(
        make_h4_problem(seed=104729, kind="coupled", horizon=7),
    ))
    with pytest.raises(ValueError, match="rounding depth"):
        replace(
            oracle.route_agreement.canonical_operand,
            rounding_depth=malformed_depth,
        )


@pytest.mark.parametrize(
    ("source_kind", "operand_path"),
    (
        ("scaled_pcg64", "canonical"),
        ("scaled_pcg64", "predictive"),
        ("h3_anchor", "canonical"),
        ("h3_anchor", "predictive"),
    ),
)
def test_h4_evaluation_rejects_positive_off_by_one_route_depth(
    source_kind: str,
    operand_path: str,
) -> None:
    if source_kind == "scaled_pcg64":
        payload = canonical_h4_problem_bytes(
            make_h4_problem(seed=104729, kind="coupled", horizon=15),
        )
    else:
        fixture = parse_h3_fixture_bytes(
            H3_COUPLED_FIXTURE_PATH.read_bytes(),
            expected_fixture_id="h3-coupled-v1",
        )
        payload = canonical_h4_problem_bytes(h4_anchor_from_h3(fixture))
    oracle = evaluate_h4_oracle(payload)
    canonical = oracle.route_agreement.canonical_operand
    predictive = oracle.route_agreement.predictive_operand
    if operand_path == "canonical":
        canonical = replace(canonical, rounding_depth=canonical.rounding_depth + 1)
    else:
        predictive = replace(predictive, rounding_depth=predictive.rounding_depth + 1)
    forged_agreement = h4_gaussian._route_agreement(
        oracle.problem_id, oracle.problem_sha256, canonical, predictive,
    )
    with pytest.raises(ValueError, match="rounding depth"):
        replace(
            oracle,
            route_agreement=forged_agreement,
            operand_evidence=(canonical, predictive),
        )


def test_h4_canonical_absolute_accumulation_is_outward_rounded_fsum() -> None:
    oracle = evaluate_h4_oracle(canonical_h4_problem_bytes(
        make_h4_problem(seed=104729, kind="coupled", horizon=7),
    ))
    precision = np.asarray(oracle.precision, dtype=np.float64)
    natural = np.asarray(oracle.natural, dtype=np.float64)
    mean = np.asarray(oracle.mean, dtype=np.float64)
    lower = np.linalg.cholesky(precision)
    components = (
        abs(oracle.constant),
        abs(0.5 * float(natural @ mean)),
        abs(-float(np.sum(np.log(np.diag(lower)), dtype=np.float64))),
        abs(0.5 * oracle.dimension * math.log(2.0 * math.pi)),
    )
    assert oracle.route_agreement.canonical_operand.absolute_summand_accumulation == (
        math.nextafter(math.fsum(components), math.inf)
    )


def test_h4_oracle_rejects_duplicate_keys_and_wrong_core_digest() -> None:
    payload = canonical_h4_problem_bytes(make_h4_problem(seed=104729, kind="zero_control", horizon=7))
    text = payload.decode("utf-8")
    duplicate = text.replace('{"canonical_sha256":', '{"schema_version":"h4-neutral-problem-v1","canonical_sha256":', 1)
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_h4_oracle(duplicate.encode("utf-8"))
    envelope = json.loads(payload)
    envelope["canonical_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="digest"):
        evaluate_h4_oracle(json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode())


@pytest.mark.parametrize(
    "mutation",
    (
        "raw_index", "raw_name", "raw_shape", "raw_value", "raw_order",
        "coordinate_order", "factor_role", "factor_time", "factor_metadata",
    ),
)
def test_h4_oracle_parser_rejects_rehashed_scaled_source_identity_mutations(
    mutation: str,
) -> None:
    payload = canonical_h4_problem_bytes(
        make_h4_problem(seed=104729, kind="coupled", horizon=7),
    )
    with pytest.raises(ValueError, match="H4|scaled|raw|coordinate|factor"):
        h4_gaussian._parse(_rehashed_payload(payload, mutation))


@pytest.mark.parametrize(
    "mutation", ("h3_factor_id", "h3_coordinate_order", "h3_factor_metadata"),
)
def test_h4_oracle_parser_rejects_rehashed_h3_source_identity_mutations(
    mutation: str,
) -> None:
    fixture = parse_h3_fixture_bytes(
        H3_COUPLED_FIXTURE_PATH.read_bytes(), expected_fixture_id="h3-coupled-v1",
    )
    payload = canonical_h4_problem_bytes(h4_anchor_from_h3(fixture))
    with pytest.raises(ValueError, match="H3|H4|coordinate|factor"):
        h4_gaussian._parse(_rehashed_payload(payload, mutation))


def test_h4_predictive_covariance_correction_is_computed_once_and_reused() -> None:
    source = inspect.getsource(h4_gaussian._predictive_route)
    assert source.count("gain @ innovation @ gain.T") == 1
