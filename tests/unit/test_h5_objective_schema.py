from __future__ import annotations

import hashlib
import json
import math

import pytest

from vfe4.types.h5_schema import (
    H5_ANALYTIC_FACTOR_OPERATION_COUNTS,
    H5_ANALYTIC_OPERATION_COUNTS,
    H5_C,
    H5_CANDIDATE_COMPARISON_OPERATION_COUNTS,
    H5_DERIVED_TERM_IDS,
    H5_DIAGNOSTIC_TERM_IDS,
    H5_EPS,
    H5_FACTOR_INPUT_FIELDS,
    H5_FACTOR_INPUT_SCHEMA_DOMAIN,
    H5_FACTOR_INPUT_SCHEMA_SHA256,
    H5_FACTOR_UNIVERSE,
    H5_MODEL_BLOCK_UNIVERSE,
    H5_NONCLAIM_IDS,
    H5_OBJECTIVE_SCHEMA_DOMAIN,
    H5_OBJECTIVE_SCHEMA_SHA256,
    H5_QUADRATURE_ORDERS,
    H5_RECOGNITION_COORDINATE_UNIVERSE,
    H5_SIGNED_TERM_IDS,
    H5_SIGNED_TERM_SIGNS,
    canonical_h5_factor_input_schema_core_bytes,
    canonical_h5_objective_schema_core_bytes,
    emission_operation_count,
    gamma_n,
)


def test_authoritative_identifier_and_term_universes_are_exact() -> None:
    assert H5_FACTOR_UNIVERSE == (
        "initial_joint",
        "model_source[1]",
        "model_transition[1]",
        "state_source[1]",
        "state_transition[1]",
        "emission[1]",
        "model_source[2]",
        "model_transition[2]",
        "state_source[2]",
        "state_transition[2]",
        "emission[2]",
        "recognition_entropy",
    )
    assert H5_RECOGNITION_COORDINATE_UNIVERSE == (
        "q[z0]",
        "q[m0]",
        "q[z1]",
        "q[m1]",
        "q[z2]",
        "q[m2]",
        "q[model_source_b1]",
        "q[state_source_a1_b0]",
        "q[model_source_b2]",
        "q[source_row_a2]",
        "q[state_source_a2_b1]",
    )
    assert H5_MODEL_BLOCK_UNIVERSE == (
        "theta[state_transition_2]",
        "theta[emission_1]",
        "theta[shared_decoder_transition]",
    )
    assert H5_SIGNED_TERM_IDS == (
        "expected_log_emission[1]",
        "expected_log_emission[2]",
        "initial_model_kl",
        "initial_state_kl",
        "model_source_kl[1]",
        "model_source_kl[2]",
        "model_transition_kl[1]",
        "model_transition_kl[2]",
        "state_source_kl[1]",
        "state_source_kl[2]",
        "state_transition_kl[1]",
        "state_transition_kl[2]",
    )
    assert H5_SIGNED_TERM_SIGNS == (1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)
    assert H5_DIAGNOSTIC_TERM_IDS == ("joint_recognition_entropy",)
    assert H5_DERIVED_TERM_IDS == ("complete_elbo",)
    assert H5_NONCLAIM_IDS == (
        "no_h4_cost_claim",
        "no_h6_prediction_claim",
        "no_h7_scaling_claim",
        "no_h8_training_or_readiness_claim",
    )
    assert H5_QUADRATURE_ORDERS == (21, 17)


def test_schema_hashes_are_domain_separated_recomputations_of_exact_cores() -> None:
    factor_core = canonical_h5_factor_input_schema_core_bytes()
    objective_core = canonical_h5_objective_schema_core_bytes()
    expected_factor_core = [
        [
            "initial_joint",
            "model_source[1]",
            "model_transition[1]",
            "state_source[1]",
            "state_transition[1]",
            "emission[1]",
            "model_source[2]",
            "model_transition[2]",
            "state_source[2]",
            "state_transition[2]",
            "emission[2]",
            "recognition_entropy",
        ],
        [
            "schema_version",
            "factor_id",
            "normalized_factor",
            "observation",
            "recognition_inputs",
        ],
        [
            ["initial_joint", ["h1.initial_joint", "q[z0]", "q[m0]"]],
            [
                "model_source[1]",
                ["h1.model_source_priors[1]", "q[model_source_b1]"],
            ],
            [
                "model_transition[1]",
                [
                    "h1.model_transition[1]",
                    "q[m0]",
                    "q[m1]",
                    "q[model_source_b1]",
                ],
            ],
            [
                "state_source[1]",
                [
                    "h1.state_source_priors[1]",
                    "q[model_source_b1]",
                    "q[state_source_a1_b0]",
                ],
            ],
            [
                "state_transition[1]",
                [
                    "h1.state_transition[1]",
                    "q[z0]",
                    "q[z1]",
                    "q[m1]",
                    "q[model_source_b1]",
                    "q[state_source_a1_b0]",
                ],
            ],
            [
                "emission[1]",
                [
                    "theta[emission_1]",
                    "theta[shared_decoder_transition]",
                    "q[z1]",
                    "q[m1]",
                    "h1.observation_label[t=1]",
                ],
            ],
            [
                "model_source[2]",
                ["h1.model_source_priors[2]", "q[model_source_b2]"],
            ],
            [
                "model_transition[2]",
                [
                    "h1.model_transition[2]",
                    "q[m0]",
                    "q[m1]",
                    "q[m2]",
                    "q[model_source_b2]",
                ],
            ],
            [
                "state_source[2]",
                [
                    "h1.state_source_priors[2]",
                    "q[model_source_b2]",
                    "q[source_row_a2]",
                    "q[state_source_a2_b1]",
                ],
            ],
            [
                "state_transition[2]",
                [
                    "theta[state_transition_2]",
                    "theta[shared_decoder_transition]",
                    "q[z0]",
                    "q[z1]",
                    "q[z2]",
                    "q[m2]",
                    "q[model_source_b2]",
                    "q[source_row_a2]",
                    "q[state_source_a2_b1]",
                ],
            ],
            [
                "emission[2]",
                [
                    "h1.emission[2]",
                    "theta[shared_decoder_transition]",
                    "q[z2]",
                    "q[m2]",
                    "h1.observation_label[t=2]",
                ],
            ],
            ["recognition_entropy", ["recognition_snapshot"]],
        ],
    ]
    expected_factor_bytes = json.dumps(
        expected_factor_core,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    expected_factor_hash = hashlib.sha256(
        b"vfe4.h5.factor-input-schema.v1\x00" + expected_factor_bytes
    ).hexdigest()
    expected_objective_core = [
        expected_factor_hash,
        expected_factor_core[0],
        [
            "q[z0]",
            "q[m0]",
            "q[z1]",
            "q[m1]",
            "q[z2]",
            "q[m2]",
            "q[model_source_b1]",
            "q[state_source_a1_b0]",
            "q[model_source_b2]",
            "q[source_row_a2]",
            "q[state_source_a2_b1]",
        ],
        [
            "theta[state_transition_2]",
            "theta[emission_1]",
            "theta[shared_decoder_transition]",
        ],
        [
            "expected_log_emission[1]",
            "expected_log_emission[2]",
            "initial_model_kl",
            "initial_state_kl",
            "model_source_kl[1]",
            "model_source_kl[2]",
            "model_transition_kl[1]",
            "model_transition_kl[2]",
            "state_source_kl[1]",
            "state_source_kl[2]",
            "state_transition_kl[1]",
            "state_transition_kl[2]",
        ],
        [1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        ["joint_recognition_entropy"],
        ["complete_elbo"],
        [
            [
                "q[z0]",
                [
                    "initial_joint",
                    "state_transition[1]",
                    "state_transition[2]",
                    "recognition_entropy",
                ],
            ],
            [
                "q[m0]",
                [
                    "initial_joint",
                    "model_transition[1]",
                    "model_transition[2]",
                    "recognition_entropy",
                ],
            ],
            [
                "q[z1]",
                [
                    "state_transition[1]",
                    "emission[1]",
                    "state_transition[2]",
                    "recognition_entropy",
                ],
            ],
            [
                "q[m1]",
                [
                    "model_transition[1]",
                    "state_transition[1]",
                    "emission[1]",
                    "model_transition[2]",
                    "recognition_entropy",
                ],
            ],
            [
                "q[z2]",
                ["state_transition[2]", "emission[2]", "recognition_entropy"],
            ],
            [
                "q[m2]",
                [
                    "model_transition[2]",
                    "state_transition[2]",
                    "emission[2]",
                    "recognition_entropy",
                ],
            ],
            [
                "q[model_source_b1]",
                [
                    "model_source[1]",
                    "model_transition[1]",
                    "state_source[1]",
                    "state_transition[1]",
                    "recognition_entropy",
                ],
            ],
            [
                "q[state_source_a1_b0]",
                ["state_source[1]", "state_transition[1]", "recognition_entropy"],
            ],
            [
                "q[model_source_b2]",
                [
                    "model_source[2]",
                    "model_transition[2]",
                    "state_source[2]",
                    "state_transition[2]",
                    "recognition_entropy",
                ],
            ],
            [
                "q[source_row_a2]",
                ["state_source[2]", "state_transition[2]", "recognition_entropy"],
            ],
            [
                "q[state_source_a2_b1]",
                ["state_source[2]", "state_transition[2]", "recognition_entropy"],
            ],
        ],
        [
            ["theta[state_transition_2]", ["state_transition[2]"]],
            ["theta[emission_1]", ["emission[1]"]],
            [
                "theta[shared_decoder_transition]",
                ["state_transition[2]", "emission[1]", "emission[2]"],
            ],
        ],
        expected_factor_core[2],
        [
            [
                "shared_decoder_transition",
                "theta[shared_decoder_transition].s",
                [
                    "state_transition[2].B:add",
                    "emission[1].w_z[0]:add",
                    "emission[2].w_z[0]:add",
                ],
            ]
        ],
        [21, 17],
        [
            ["initial_model_kl", 192],
            ["initial_state_kl", 192],
            ["model_source_kl[1]", 32],
            ["model_source_kl[2]", 64],
            ["model_transition_kl[1]", 192],
            ["model_transition_kl[2]", 320],
            ["state_source_kl[1]", 32],
            ["state_source_kl[2]", 96],
            ["state_transition_kl[1]", 256],
            ["state_transition_kl[2]", 448],
            ["joint_recognition_entropy", 320],
        ],
        [
            ["initial_joint", 256],
            ["model_source[1]", 32],
            ["model_transition[1]", 192],
            ["state_source[1]", 32],
            ["state_transition[1]", 256],
            ["model_source[2]", 64],
            ["model_transition[2]", 320],
            ["state_source[2]", 96],
            ["state_transition[2]", 448],
            ["recognition_entropy", 320],
        ],
        [
            ["exact_z0.mean", 512],
            ["exact_z0.variance", 512],
            ["exact_source_row_a2.probability[0]", 512],
            ["exact_source_row_a2.probability[1]", 512],
            ["exact_state_transition_2_m.alpha_0", 4096],
            ["exact_state_transition_2_m.alpha_1", 4096],
            ["exact_state_transition_2_m.B_base", 4096],
            ["exact_state_transition_2_m.c", 4096],
            ["exact_state_transition_2_m.R", 4096],
        ],
        [
            "h5-term-budget-v1",
            "h5-complete-budget-v1",
            "h5-delta-budget-v1",
            "h5-candidate-comparison-v1",
        ],
    ]
    expected_objective_bytes = json.dumps(
        expected_objective_core,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")

    assert H5_FACTOR_INPUT_FIELDS == tuple(expected_factor_core[1])
    assert H5_FACTOR_INPUT_SCHEMA_DOMAIN == b"vfe4.h5.factor-input-schema.v1\x00"
    assert H5_OBJECTIVE_SCHEMA_DOMAIN == b"vfe4.h5.objective-schema.v1\x00"
    assert json.loads(factor_core) == expected_factor_core
    assert json.loads(objective_core) == expected_objective_core
    assert factor_core == expected_factor_bytes
    assert objective_core == expected_objective_bytes
    assert H5_FACTOR_INPUT_SCHEMA_SHA256 == expected_factor_hash
    assert H5_OBJECTIVE_SCHEMA_SHA256 == hashlib.sha256(
        b"vfe4.h5.objective-schema.v1\x00" + expected_objective_bytes
    ).hexdigest()
    assert len(H5_FACTOR_INPUT_SCHEMA_SHA256) == 64
    assert len(H5_OBJECTIVE_SCHEMA_SHA256) == 64
    assert H5_FACTOR_INPUT_SCHEMA_SHA256 != H5_OBJECTIVE_SCHEMA_SHA256


def test_frozen_operation_count_tables_and_emission_formula_are_exact() -> None:
    assert dict(H5_ANALYTIC_OPERATION_COUNTS) == {
        "initial_model_kl": 192,
        "initial_state_kl": 192,
        "model_source_kl[1]": 32,
        "model_source_kl[2]": 64,
        "model_transition_kl[1]": 192,
        "model_transition_kl[2]": 320,
        "state_source_kl[1]": 32,
        "state_source_kl[2]": 96,
        "state_transition_kl[1]": 256,
        "state_transition_kl[2]": 448,
        "joint_recognition_entropy": 320,
    }
    assert dict(H5_ANALYTIC_FACTOR_OPERATION_COUNTS) == {
        "initial_joint": 256,
        "model_source[1]": 32,
        "model_transition[1]": 192,
        "state_source[1]": 32,
        "state_transition[1]": 256,
        "model_source[2]": 64,
        "model_transition[2]": 320,
        "state_source[2]": 96,
        "state_transition[2]": 448,
        "recognition_entropy": 320,
    }
    assert dict(H5_CANDIDATE_COMPARISON_OPERATION_COUNTS) == {
        "exact_z0.mean": 512,
        "exact_z0.variance": 512,
        "exact_source_row_a2.probability[0]": 512,
        "exact_source_row_a2.probability[1]": 512,
        "exact_state_transition_2_m.alpha_0": 4096,
        "exact_state_transition_2_m.alpha_1": 4096,
        "exact_state_transition_2_m.B_base": 4096,
        "exact_state_transition_2_m.c": 4096,
        "exact_state_transition_2_m.R": 4096,
    }
    assert emission_operation_count(21) == 32 * 21 * 21 + 8 * 21 + 32
    assert emission_operation_count(17) == 32 * 17 * 17 + 8 * 17 + 32
    with pytest.raises(ValueError):
        emission_operation_count(19)


def test_binary64_rounding_constants_are_literal_and_fail_closed() -> None:
    assert H5_EPS == math.ulp(1.0)
    assert H5_C == 4096.0
    assert gamma_n(3) == (3 * H5_EPS) / (1.0 - 3 * H5_EPS)
    with pytest.raises(ValueError):
        gamma_n(0)
    with pytest.raises(ValueError):
        gamma_n(True)
