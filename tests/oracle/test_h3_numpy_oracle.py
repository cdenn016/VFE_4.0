from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import numpy as np
import pytest

import verification.numpy_oracles.h3_posterior as oracle_module
from verification.h3_budget import (
    C,
    EPS,
    SOLVER_ALLOWANCE_NATS,
    allowance_is_decisive,
    four_operand_identity_allowance,
    gamma_n,
    operation_count,
    pair_allowance,
    scalar_allowance,
    three_operand_identity_allowance,
)
from verification.numpy_oracles.h3_posterior import (
    H3PosteriorOracleEvaluation,
    evaluate_h3_posterior_oracle,
    reverse_kl_to_oracle,
)
from vfe4.validation.h3_fixture import (
    H3_COUPLED_FIXTURE_PATH,
    H3_ZERO_CONTROL_FIXTURE_PATH,
)


DIMENSION = 4
COUPLED_PRECISION = np.asarray(
    (
        (2.96, 0.0, -2.8, 1.68),
        (0.0, 2.77777777777778, 0.0, -2.22222222222222),
        (-2.8, 0.0, 5.5625, -2.4),
        (1.68, -2.22222222222222, -2.4, 5.78027777777778),
    ),
    dtype=np.float64,
)
COUPLED_NATURAL = np.asarray((0.0, 0.0, 1.71875, 0.3125), dtype=np.float64)
COUPLED_EVIDENCE = -2.6536596233553
COUPLED_GAP = 0.6815463199745935
ZERO_PRECISION = np.diag(
    np.asarray((1.0, 1.0, 5.5625, 4.34027777777778), dtype=np.float64)
)


def _oracle(path: Path, fixture_id: str) -> H3PosteriorOracleEvaluation:
    return evaluate_h3_posterior_oracle(
        path.read_bytes(), expected_fixture_id=fixture_id
    )


def _scalar_budget(
    value: float, absolute_sum: float, kappa: float, *, optimized: bool = False
) -> float:
    return scalar_allowance(
        DIMENSION,
        value=value,
        absolute_sum=absolute_sum,
        kappas=(kappa,),
        optimized=optimized,
    )


def _assert_pair(
    left: float,
    right: float,
    *,
    left_absolute_sum: float,
    right_absolute_sum: float,
    kappa: float,
) -> None:
    left_allowance = _scalar_budget(left, left_absolute_sum, kappa)
    right_allowance = _scalar_budget(right, right_absolute_sum, kappa)
    allowance = pair_allowance(
        DIMENSION,
        left=left,
        right=right,
        left_allowance=left_allowance,
        right_allowance=right_allowance,
    )
    assert abs(left - right) <= allowance


def test_h3_oracle_source_is_independent_standard_library_and_numpy_only() -> None:
    source_path = Path(oracle_module.__file__).resolve()
    source = source_path.read_text(encoding="utf-8")
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

    assert imported_roots.isdisjoint({"torch", "vfe4", "verification"})
    assert "np.linalg.cholesky" in source
    assert "np.linalg.solve" in source
    assert "np.linalg.slogdet" in source
    assert "np.linalg.inv" not in source
    assert "np.linalg.pinv" not in source
    assert "jitter" not in source.lower()


def test_coupled_oracle_reconstructs_frozen_canonical_evidence_and_gap() -> None:
    oracle = _oracle(H3_COUPLED_FIXTURE_PATH, "h3-coupled-v1")
    diagnostics = oracle.diagnostics
    kappa = float(diagnostics["kappa_2"])
    precision_absolute = diagnostics["precision_absolute_summand_accumulation"]
    natural_absolute = diagnostics["natural_absolute_summand_accumulation"]

    assert oracle.fixture_id == "h3-coupled-v1"
    assert oracle.precision.dtype == np.float64
    assert oracle.precision.shape == (4, 4)
    assert oracle.natural.shape == (4,)
    assert oracle.mean.shape == (4,)
    assert oracle.covariance.shape == (4, 4)
    for row in range(4):
        for column in range(4):
            _assert_pair(
                float(oracle.precision[row, column]),
                float(COUPLED_PRECISION[row, column]),
                left_absolute_sum=float(precision_absolute[row][column]),
                right_absolute_sum=abs(float(COUPLED_PRECISION[row, column])),
                kappa=kappa,
            )
    for index in range(4):
        _assert_pair(
            float(oracle.natural[index]),
            float(COUPLED_NATURAL[index]),
            left_absolute_sum=float(natural_absolute[index]),
            right_absolute_sum=abs(float(COUPLED_NATURAL[index])),
            kappa=kappa,
        )
    _assert_pair(
        oracle.log_evidence,
        COUPLED_EVIDENCE,
        left_absolute_sum=float(
            diagnostics["log_evidence_absolute_summand_accumulation"]
        ),
        right_absolute_sum=abs(COUPLED_EVIDENCE),
        kappa=kappa,
    )
    _assert_pair(
        oracle.analytic_factorized_reverse_kl,
        COUPLED_GAP,
        left_absolute_sum=float(
            diagnostics[
                "analytic_factorized_reverse_kl_absolute_summand_accumulation"
            ]
        ),
        right_absolute_sum=abs(COUPLED_GAP),
        kappa=kappa,
    )
    assert np.array_equal(
        oracle.analytic_factorized_precision,
        np.diag(np.diag(oracle.precision)),
    )
    assert np.array_equal(oracle.analytic_factorized_mean, oracle.mean)
    assert float(diagnostics["minimum_cholesky_pivot"]) > 0.0
    assert float(diagnostics["lambda_min"]) > 0.0
    assert float(diagnostics["lambda_max"]) >= float(diagnostics["lambda_min"])
    assert kappa >= 1.0


def test_zero_control_oracle_derives_diagonal_posterior_and_zero_gap() -> None:
    oracle = _oracle(H3_ZERO_CONTROL_FIXTURE_PATH, "h3-zero-control-v1")
    diagnostics = oracle.diagnostics
    kappa = float(diagnostics["kappa_2"])
    precision_absolute = diagnostics["precision_absolute_summand_accumulation"]

    for row in range(4):
        for column in range(4):
            _assert_pair(
                float(oracle.precision[row, column]),
                float(ZERO_PRECISION[row, column]),
                left_absolute_sum=float(precision_absolute[row][column]),
                right_absolute_sum=abs(float(ZERO_PRECISION[row, column])),
                kappa=kappa,
            )
    zero_allowance = pair_allowance(
        DIMENSION,
        left=oracle.analytic_factorized_reverse_kl,
        right=0.0,
        left_allowance=_scalar_budget(
            oracle.analytic_factorized_reverse_kl,
            float(
                diagnostics[
                    "analytic_factorized_reverse_kl_absolute_summand_accumulation"
                ]
            ),
            kappa,
        ),
        right_allowance=0.0,
    )
    assert abs(oracle.analytic_factorized_reverse_kl) <= zero_allowance
    assert np.array_equal(
        oracle.analytic_factorized_precision, oracle.precision
    )


def test_reverse_kl_is_oriented_q_to_p_and_uses_terminal_precision() -> None:
    oracle = _oracle(H3_COUPLED_FIXTURE_PATH, "h3-coupled-v1")
    exact_kl = reverse_kl_to_oracle(
        oracle, mean=oracle.mean.copy(), precision=oracle.precision.copy()
    )
    factorized_kl = reverse_kl_to_oracle(
        oracle,
        mean=oracle.analytic_factorized_mean.copy(),
        precision=oracle.analytic_factorized_precision.copy(),
    )
    shifted_mean = oracle.mean.copy()
    shifted_mean[0] += 0.25
    shifted_kl = reverse_kl_to_oracle(
        oracle, mean=shifted_mean, precision=oracle.precision.copy()
    )

    exact_allowance = pair_allowance(
        DIMENSION,
        left=exact_kl,
        right=0.0,
        left_allowance=_scalar_budget(
            exact_kl,
            max(1.0, abs(exact_kl)),
            float(oracle.diagnostics["kappa_2"]),
        ),
        right_allowance=0.0,
    )
    factorized_allowance = pair_allowance(
        DIMENSION,
        left=factorized_kl,
        right=oracle.analytic_factorized_reverse_kl,
        left_allowance=_scalar_budget(
            factorized_kl,
            max(1.0, abs(factorized_kl)),
            float(oracle.diagnostics["kappa_2"]),
        ),
        right_allowance=_scalar_budget(
            oracle.analytic_factorized_reverse_kl,
            float(
                oracle.diagnostics[
                    "analytic_factorized_reverse_kl_absolute_summand_accumulation"
                ]
            ),
            float(oracle.diagnostics["kappa_2"]),
        ),
    )
    expected_shift = 0.5 * 0.25**2 * oracle.precision[0, 0]

    assert abs(exact_kl) <= exact_allowance
    assert (
        abs(factorized_kl - oracle.analytic_factorized_reverse_kl)
        <= factorized_allowance
    )
    assert abs(shifted_kl - expected_shift) <= _scalar_budget(
        shifted_kl,
        abs(shifted_kl) + abs(expected_shift),
        float(oracle.diagnostics["kappa_2"]),
    )


def test_oracle_outputs_are_owned_read_only_and_solve_closes() -> None:
    oracle = _oracle(H3_COUPLED_FIXTURE_PATH, "h3-coupled-v1")
    for array in (
        oracle.precision,
        oracle.natural,
        oracle.mean,
        oracle.covariance,
        oracle.analytic_factorized_precision,
        oracle.analytic_factorized_mean,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flat[0] = 99.0
    residual = oracle.precision @ oracle.covariance - np.eye(4, dtype=np.float64)
    allowance = _scalar_budget(
        float(np.max(np.abs(residual))),
        float(np.sum(np.abs(oracle.precision @ oracle.covariance))) + 4.0,
        float(oracle.diagnostics["kappa_2"]),
    )
    assert float(np.max(np.abs(residual))) <= allowance


def test_oracle_rejects_bad_bytes_identity_reference_and_non_spd_laws() -> None:
    coupled_bytes = H3_COUPLED_FIXTURE_PATH.read_bytes()
    with pytest.raises(ValueError):
        evaluate_h3_posterior_oracle(
            b"not-json", expected_fixture_id="h3-coupled-v1"
        )
    with pytest.raises(ValueError):
        evaluate_h3_posterior_oracle(
            coupled_bytes, expected_fixture_id="h3-zero-control-v1"
        )
    with pytest.raises(ValueError):
        evaluate_h3_posterior_oracle(
            coupled_bytes, expected_fixture_id="unknown"
        )

    raw = json.loads(coupled_bytes)
    raw["reference"]["log_evidence"] = -2.0
    with pytest.raises(ValueError, match="reference"):
        evaluate_h3_posterior_oracle(
            json.dumps(raw).encode("utf-8"),
            expected_fixture_id="h3-coupled-v1",
        )

    with pytest.raises(ValueError, match="symmetric"):
        reverse_kl_to_oracle(
            _oracle(H3_COUPLED_FIXTURE_PATH, "h3-coupled-v1"),
            mean=np.zeros(4, dtype=np.float64),
            precision=np.asarray(
                (
                    (1.0, 0.1, 0.0, 0.0),
                    (0.0, 1.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0, 0.0),
                    (0.0, 0.0, 0.0, 1.0),
                ),
                dtype=np.float64,
            ),
        )
    with pytest.raises(ValueError, match="positive definite"):
        reverse_kl_to_oracle(
            _oracle(H3_COUPLED_FIXTURE_PATH, "h3-coupled-v1"),
            mean=np.zeros(4, dtype=np.float64),
            precision=np.diag(np.asarray((1.0, 1.0, 1.0, -1.0))),
        )


def test_h3_budget_implements_exact_operand_local_formulas() -> None:
    assert EPS == float(np.finfo(np.float64).eps)
    assert C == 4096.0
    assert SOLVER_ALLOWANCE_NATS == 1.0e-7
    assert operation_count(4) == 128
    assert gamma_n(128) == 128 * EPS / (1.0 - 128 * EPS)

    scalar = scalar_allowance(
        4,
        value=-3.0,
        absolute_sum=7.0,
        kappas=(2.0, 5.0),
        optimized=False,
    )
    optimized = scalar_allowance(
        4,
        value=-3.0,
        absolute_sum=7.0,
        kappas=(2.0, 5.0),
        optimized=True,
    )
    expected_scalar = C * gamma_n(128) * 5.0 * 7.0
    assert scalar == expected_scalar
    assert optimized == SOLVER_ALLOWANCE_NATS + expected_scalar

    pair = pair_allowance(
        4,
        left=-3.0,
        right=2.0,
        left_allowance=scalar,
        right_allowance=optimized,
    )
    assert pair == scalar + optimized + C * gamma_n(6) * 5.0

    three = three_operand_identity_allowance(
        4,
        operands=(-3.0, 2.0, -4.0),
        operand_allowances=(scalar, optimized, scalar),
    )
    assert three == math.fsum((scalar, optimized, scalar)) + C * gamma_n(7) * 9.0

    four = four_operand_identity_allowance(
        4,
        operands=(-3.0, 2.0, -4.0, 5.0),
        operand_allowances=(scalar, optimized, scalar, optimized),
    )
    assert four == math.fsum((scalar, optimized, scalar, optimized)) + C * gamma_n(8) * 14.0


def test_h3_budget_decisiveness_is_strict_at_one_percent_boundary() -> None:
    scale = 0.6815463199745935
    boundary = 0.01 * scale
    assert allowance_is_decisive(math.nextafter(boundary, 0.0), scale)
    assert not allowance_is_decisive(boundary, scale)
    assert not allowance_is_decisive(math.nextafter(boundary, math.inf), scale)


@pytest.mark.parametrize(
    "call",
    (
        lambda: gamma_n(True),
        lambda: gamma_n(0),
        lambda: operation_count(3),
        lambda: scalar_allowance(
            4, value=0.0, absolute_sum=1.0, kappas=(), optimized=False
        ),
        lambda: scalar_allowance(
            4, value=0.0, absolute_sum=-1.0, kappas=(1.0,), optimized=False
        ),
        lambda: scalar_allowance(
            4,
            value=0.0,
            absolute_sum=1.0,
            kappas=(float("inf"),),
            optimized=False,
        ),
        lambda: scalar_allowance(
            4, value=0.0, absolute_sum=1.0, kappas=(1.0,), optimized=1
        ),
        lambda: pair_allowance(
            4,
            left=0.0,
            right=0.0,
            left_allowance=-1.0,
            right_allowance=0.0,
        ),
        lambda: three_operand_identity_allowance(
            4, operands=(1.0, 2.0), operand_allowances=(0.0, 0.0)
        ),
        lambda: four_operand_identity_allowance(
            4,
            operands=(1.0, 2.0, 3.0, 4.0),
            operand_allowances=(0.0, 0.0, 0.0),
        ),
        lambda: allowance_is_decisive(-1.0, 1.0),
        lambda: allowance_is_decisive(0.0, 0.0),
    ),
)
def test_h3_budget_rejects_malformed_inputs(call) -> None:
    with pytest.raises(ValueError):
        call()
