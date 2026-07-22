"""Dependency-neutral immutable schema constants for the H5 update gate."""

from __future__ import annotations

import hashlib
import json
import math
from types import MappingProxyType
from typing import Final, Mapping


H5_UPDATE_SPEC_DOMAIN: Final = b"vfe4.h5.update-spec.v1\x00"
H5_UPDATE_REQUEST_DOMAIN: Final = b"vfe4.h5.update-request.v1\x00"
H5_REFERENCE_STATE_DOMAIN: Final = b"vfe4.h5.reference-state.v1\x00"
H5_RECOGNITION_SNAPSHOT_DOMAIN: Final = b"vfe4.h5.recognition-snapshot.v1\x00"
H5_MODEL_SNAPSHOT_DOMAIN: Final = b"vfe4.h5.model-snapshot.v1\x00"
H5_LIVE_STATE_DOMAIN: Final = b"vfe4.h5.live-state.v1\x00"
H5_CANDIDATE_DOMAIN: Final = b"vfe4.h5.candidate.v1\x00"
H5_SEMANTIC_STATE_DOMAIN: Final = b"vfe4.h5.semantic-state.v1\x00"
H5_ATTEMPT_DOMAIN: Final = b"vfe4.h5.attempt.v1\x00"
H5_TRANSACTION_DOMAIN: Final = b"vfe4.h5.transaction.v1\x00"
H5_FACTOR_INPUT_SCHEMA_DOMAIN: Final = b"vfe4.h5.factor-input-schema.v1\x00"
H5_FACTOR_INPUT_DOMAIN: Final = b"vfe4.h5.factor-input.v1\x00"
H5_FROZEN_COMPLEMENT_DOMAIN: Final = b"vfe4.h5.frozen-complement.v1\x00"
H5_OPTIMIZER_STATE_DOMAIN: Final = b"vfe4.h5.optimizer-state.v1\x00"
H5_RNG_STATE_DOMAIN: Final = b"vfe4.h5.rng-state.v1\x00"
H5_OBJECTIVE_SCHEMA_DOMAIN: Final = b"vfe4.h5.objective-schema.v1\x00"
H5_VALIDATION_PAYLOAD_DOMAIN: Final = b"vfe4.h5.validation-payload.v1\x00"

H5_HASH_DOMAINS: Final = (
    H5_UPDATE_SPEC_DOMAIN,
    H5_UPDATE_REQUEST_DOMAIN,
    H5_REFERENCE_STATE_DOMAIN,
    H5_RECOGNITION_SNAPSHOT_DOMAIN,
    H5_MODEL_SNAPSHOT_DOMAIN,
    H5_LIVE_STATE_DOMAIN,
    H5_CANDIDATE_DOMAIN,
    H5_SEMANTIC_STATE_DOMAIN,
    H5_ATTEMPT_DOMAIN,
    H5_TRANSACTION_DOMAIN,
    H5_FACTOR_INPUT_SCHEMA_DOMAIN,
    H5_FACTOR_INPUT_DOMAIN,
    H5_FROZEN_COMPLEMENT_DOMAIN,
    H5_OPTIMIZER_STATE_DOMAIN,
    H5_RNG_STATE_DOMAIN,
    H5_OBJECTIVE_SCHEMA_DOMAIN,
    H5_VALIDATION_PAYLOAD_DOMAIN,
)

H5_FACTOR_UNIVERSE: Final = (
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

H5_RECOGNITION_COORDINATE_UNIVERSE: Final = (
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

H5_MODEL_BLOCK_UNIVERSE: Final = (
    "theta[state_transition_2]",
    "theta[emission_1]",
    "theta[shared_decoder_transition]",
)

H5_SIGNED_TERM_IDS: Final = (
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
H5_SIGNED_TERM_SIGNS: Final = (1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)
H5_DIAGNOSTIC_TERM_IDS: Final = ("joint_recognition_entropy",)
H5_DERIVED_TERM_IDS: Final = ("complete_elbo",)
H5_NONCLAIM_IDS: Final = (
    "no_h4_cost_claim",
    "no_h6_prediction_claim",
    "no_h7_scaling_claim",
    "no_h8_training_or_readiness_claim",
)

H5_QUADRATURE_ORDERS: Final = (21, 17)
H5_FACTOR_INPUT_SCHEMA_VERSION: Final = "h5-factor-input-v1"
H5_H1_FIXTURE_RAW_SHA256: Final = (
    "388e38cc8c16d8b5e2c61919c1e712a134d88fb0bbd8ec1f2939b9859c9a583b"
)
H5_FACTOR_INPUT_FIELDS: Final = (
    "schema_version",
    "factor_id",
    "normalized_factor",
    "observation",
    "recognition_inputs",
)

H5_RECONSTRUCTION_ROWS: Final = (
    ("initial_joint", ("h1.initial_joint", "q[z0]", "q[m0]")),
    ("model_source[1]", ("h1.model_source_priors[1]", "q[model_source_b1]")),
    (
        "model_transition[1]",
        ("h1.model_transition[1]", "q[m0]", "q[m1]", "q[model_source_b1]"),
    ),
    (
        "state_source[1]",
        ("h1.state_source_priors[1]", "q[model_source_b1]", "q[state_source_a1_b0]"),
    ),
    (
        "state_transition[1]",
        (
            "h1.state_transition[1]",
            "q[z0]",
            "q[z1]",
            "q[m1]",
            "q[model_source_b1]",
            "q[state_source_a1_b0]",
        ),
    ),
    (
        "emission[1]",
        (
            "theta[emission_1]",
            "theta[shared_decoder_transition]",
            "q[z1]",
            "q[m1]",
            "h1.observation_label[t=1]",
        ),
    ),
    ("model_source[2]", ("h1.model_source_priors[2]", "q[model_source_b2]")),
    (
        "model_transition[2]",
        (
            "h1.model_transition[2]",
            "q[m0]",
            "q[m1]",
            "q[m2]",
            "q[model_source_b2]",
        ),
    ),
    (
        "state_source[2]",
        (
            "h1.state_source_priors[2]",
            "q[model_source_b2]",
            "q[source_row_a2]",
            "q[state_source_a2_b1]",
        ),
    ),
    (
        "state_transition[2]",
        (
            "theta[state_transition_2]",
            "theta[shared_decoder_transition]",
            "q[z0]",
            "q[z1]",
            "q[z2]",
            "q[m2]",
            "q[model_source_b2]",
            "q[source_row_a2]",
            "q[state_source_a2_b1]",
        ),
    ),
    (
        "emission[2]",
        (
            "h1.emission[2]",
            "theta[shared_decoder_transition]",
            "q[z2]",
            "q[m2]",
            "h1.observation_label[t=2]",
        ),
    ),
    ("recognition_entropy", ("recognition_snapshot",)),
)

H5_SHARED_PARAMETER_GROUP_ROWS: Final = (
    (
        "shared_decoder_transition",
        "theta[shared_decoder_transition].s",
        (
            "state_transition[2].B:add",
            "emission[1].w_z[0]:add",
            "emission[2].w_z[0]:add",
        ),
    ),
)

H5_VARIABLE_DEPENDENCY_ROWS: Final = (
    (
        "q[z0]",
        ("initial_joint", "state_transition[1]", "state_transition[2]", "recognition_entropy"),
    ),
    (
        "q[m0]",
        ("initial_joint", "model_transition[1]", "model_transition[2]", "recognition_entropy"),
    ),
    (
        "q[z1]",
        ("state_transition[1]", "emission[1]", "state_transition[2]", "recognition_entropy"),
    ),
    (
        "q[m1]",
        (
            "model_transition[1]",
            "state_transition[1]",
            "emission[1]",
            "model_transition[2]",
            "recognition_entropy",
        ),
    ),
    ("q[z2]", ("state_transition[2]", "emission[2]", "recognition_entropy")),
    (
        "q[m2]",
        ("model_transition[2]", "state_transition[2]", "emission[2]", "recognition_entropy"),
    ),
    (
        "q[model_source_b1]",
        (
            "model_source[1]",
            "model_transition[1]",
            "state_source[1]",
            "state_transition[1]",
            "recognition_entropy",
        ),
    ),
    (
        "q[state_source_a1_b0]",
        ("state_source[1]", "state_transition[1]", "recognition_entropy"),
    ),
    (
        "q[model_source_b2]",
        (
            "model_source[2]",
            "model_transition[2]",
            "state_source[2]",
            "state_transition[2]",
            "recognition_entropy",
        ),
    ),
    (
        "q[source_row_a2]",
        ("state_source[2]", "state_transition[2]", "recognition_entropy"),
    ),
    (
        "q[state_source_a2_b1]",
        ("state_source[2]", "state_transition[2]", "recognition_entropy"),
    ),
)

H5_PARAMETER_DEPENDENCY_ROWS: Final = (
    ("theta[state_transition_2]", ("state_transition[2]",)),
    ("theta[emission_1]", ("emission[1]",)),
    (
        "theta[shared_decoder_transition]",
        ("state_transition[2]", "emission[1]", "emission[2]"),
    ),
)

H5_EPS: Final = math.ulp(1.0)
H5_C: Final = 4096.0


def _immutable_mapping(rows: tuple[tuple[str, int], ...]) -> Mapping[str, int]:
    return MappingProxyType(dict(rows))


H5_ANALYTIC_OPERATION_COUNTS: Final = _immutable_mapping(
    (
        ("initial_model_kl", 192),
        ("initial_state_kl", 192),
        ("model_source_kl[1]", 32),
        ("model_source_kl[2]", 64),
        ("model_transition_kl[1]", 192),
        ("model_transition_kl[2]", 320),
        ("state_source_kl[1]", 32),
        ("state_source_kl[2]", 96),
        ("state_transition_kl[1]", 256),
        ("state_transition_kl[2]", 448),
        ("joint_recognition_entropy", 320),
    )
)

H5_ANALYTIC_FACTOR_OPERATION_COUNTS: Final = _immutable_mapping(
    (
        ("initial_joint", 256),
        ("model_source[1]", 32),
        ("model_transition[1]", 192),
        ("state_source[1]", 32),
        ("state_transition[1]", 256),
        ("model_source[2]", 64),
        ("model_transition[2]", 320),
        ("state_source[2]", 96),
        ("state_transition[2]", 448),
        ("recognition_entropy", 320),
    )
)

H5_CANDIDATE_COMPARISON_OPERATION_COUNTS: Final = _immutable_mapping(
    (
        ("exact_z0.mean", 512),
        ("exact_z0.variance", 512),
        ("exact_source_row_a2.probability[0]", 512),
        ("exact_source_row_a2.probability[1]", 512),
        ("exact_state_transition_2_m.alpha_0", 4096),
        ("exact_state_transition_2_m.alpha_1", 4096),
        ("exact_state_transition_2_m.B_base", 4096),
        ("exact_state_transition_2_m.c", 4096),
        ("exact_state_transition_2_m.R", 4096),
    )
)

H5_FORMULA_TAGS: Final = (
    "h5-term-budget-v1",
    "h5-complete-budget-v1",
    "h5-delta-budget-v1",
    "h5-candidate-comparison-v1",
)


def gamma_n(n: int) -> float:
    """Return the frozen binary64 forward-error factor for ``n`` operations."""
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    numerator = n * H5_EPS
    if numerator >= 1.0:
        raise ValueError("n is too large for a finite gamma_n")
    return numerator / (1.0 - numerator)


def emission_operation_count(order: int) -> int:
    """Return the frozen conservative emission-evaluator operation count."""
    if type(order) is not int or order not in H5_QUADRATURE_ORDERS:
        raise ValueError("H5 quadrature order must be 21 or 17")
    return 32 * order * order + 8 * order + 32


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_h5_factor_input_schema_core_bytes() -> bytes:
    """Encode the exact domain-independent factor-input schema core."""
    return _canonical_json_bytes(
        (H5_FACTOR_UNIVERSE, H5_FACTOR_INPUT_FIELDS, H5_RECONSTRUCTION_ROWS)
    )


H5_FACTOR_INPUT_SCHEMA_SHA256: Final = hashlib.sha256(
    H5_FACTOR_INPUT_SCHEMA_DOMAIN + canonical_h5_factor_input_schema_core_bytes()
).hexdigest()


def canonical_h5_objective_schema_core_bytes() -> bytes:
    """Encode the exact domain-independent complete-objective schema core."""
    return _canonical_json_bytes(
        (
            H5_FACTOR_INPUT_SCHEMA_SHA256,
            H5_FACTOR_UNIVERSE,
            H5_RECOGNITION_COORDINATE_UNIVERSE,
            H5_MODEL_BLOCK_UNIVERSE,
            H5_SIGNED_TERM_IDS,
            H5_SIGNED_TERM_SIGNS,
            H5_DIAGNOSTIC_TERM_IDS,
            H5_DERIVED_TERM_IDS,
            H5_VARIABLE_DEPENDENCY_ROWS,
            H5_PARAMETER_DEPENDENCY_ROWS,
            H5_RECONSTRUCTION_ROWS,
            H5_SHARED_PARAMETER_GROUP_ROWS,
            H5_QUADRATURE_ORDERS,
            tuple(H5_ANALYTIC_OPERATION_COUNTS.items()),
            tuple(H5_ANALYTIC_FACTOR_OPERATION_COUNTS.items()),
            tuple(H5_CANDIDATE_COMPARISON_OPERATION_COUNTS.items()),
            H5_FORMULA_TAGS,
        )
    )


H5_OBJECTIVE_SCHEMA_SHA256: Final = hashlib.sha256(
    H5_OBJECTIVE_SCHEMA_DOMAIN + canonical_h5_objective_schema_core_bytes()
).hexdigest()


__all__ = [
    name
    for name in tuple(globals())
    if name.startswith("H5_")
] + [
    "canonical_h5_factor_input_schema_core_bytes",
    "canonical_h5_objective_schema_core_bytes",
    "emission_operation_count",
    "gamma_n",
]
