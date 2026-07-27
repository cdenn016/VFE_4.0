"""Pure, import-cycle-free protocol identity for the frozen H8 child protocol."""

from __future__ import annotations

import hashlib

from verification.h8_budget import (
    EPS,
    H8_OPERATION_SCOPES,
    H8_REQUIRED_NUMPY_PRODUCERS,
    H8_REQUIRED_PASS_DECISIONS,
    H8_REQUIRED_RESIDUALS,
    H8_SCALE_RESIDUAL_SPECS,
    H8_SETUP_SCOPES,
    MAX_ALLOWANCE_FRACTION,
    ROUNDING_MULTIPLIER,
    SOLVER_RELATIVE_BUDGET,
    canonical_json_bytes,
)
from verification.h8_wire import (
    H8_CHILD_ENVELOPE_KEYS,
    H8_CHILD_IDENTITY_KEYS,
    H8_CHILD_MODULE,
    H8_CHILD_REQUEST_KEYS,
    H8_CHILD_RESULT_KEYS,
    H8_CHILD_SCHEMA_VERSION,
    H8_MAX_PROCESS_INCREMENTAL_BYTES,
    H8_MAX_SECONDS,
    H8_MAX_STORAGE_SCALARS,
    H8_MAX_TORCH_POPULATION_BYTES,
    H8_MIN_CHOLESKY_PIVOT,
    H8_NEGATIVE_CONTROL_IDS,
    H8_LAYOUT_D_M,
    H8_LAYOUT_D_Z,
    H8_LAYOUT_HORIZON,
    H8_PRODUCTION_SAMPLE_SEED_PAIRS,
    H8_PRODUCTION_SEEDS,
    H8_PROFILER_API_CONTRACT_SHA256,
    H8_PROFILER_INVOCATION_ITEMS,
    H8_PROFILER_MEMORY_SOURCE_SHA256,
    H8_PROFILER_SOURCE_SHA256,
    H8_PROFILER_TORCH_VERSION,
    H8_REQUIRED_OPERATIONS,
    H8_THREAD_ENVIRONMENT_ITEMS,
    H8_TORCH_NUM_INTEROP_THREADS,
    H8_TORCH_NUM_THREADS,
)
from vfe4.config.schema import H8ValidationConfig
from vfe4.numerics.block_tridiagonal import (
    H8_HAGER_HIGHAM_1NORM_POLICY,
    H8_HAGER_HIGHAM_MAXIMUM_ITERATIONS,
)
from vfe4.types.h8 import (
    H8_CORRECTNESS_CASES,
    H8_CORRECTNESS_CONTROL_IDS,
    H8_CORRECTNESS_ORDERED_SOURCE_PAIRS,
    H8_CORRECTNESS_SOURCES,
    H8_PROBLEM_DRAW_SCHEMA_SHA256,
)


_H8_PARENT_POLICY = "continue_after_inconclusive_stop_after_first_fail"
_H8_NEGATIVE_CONTROL_CONTRACT = (
    ("torch_matrix_d_d", "torch.empty", ("dispatch",)),
    ("torch_flat_d2", "torch.empty", ("dispatch",)),
    ("torch_near_d2", "torch.empty", ("dispatch",)),
    ("torch_length_d", "torch.empty", ("dispatch",)),
    ("torch_block_pair_slab", "torch.empty", ("dispatch",)),
    (
        "torch_triangular_pair_storage",
        "torch.empty",
        ("dispatch",),
    ),
    ("torch_pair_stack", "torch.stack", ("dispatch",)),
    (
        "torch_eye_full_rhs",
        "torch.eye",
        ("backend", "dispatch"),
    ),
    (
        "torch_dense_eigvalsh",
        "torch.linalg.eigvalsh",
        ("dispatch",),
    ),
    ("numpy_matrix_d_d", "numpy.empty", ("numpy_guard",)),
    ("numpy_outer_d_d", "numpy.outer", ("numpy_guard",)),
    ("numpy_matmul_d_d", "numpy.matmul", ("numpy_guard",)),
)


def _h8_protocol_preimage(config: H8ValidationConfig) -> dict[str, object]:
    """Return the complete pure v2 parent/child protocol preimage."""

    if type(config) is not H8ValidationConfig:
        raise ValueError("config must be an exact H8ValidationConfig")
    config.__post_init__()
    if config != H8ValidationConfig.create():
        raise ValueError("H8 validation configuration is stale")
    if config.child_schema != H8_CHILD_SCHEMA_VERSION:
        raise ValueError(
            "H8 config child schema does not match the executable protocol"
        )
    if config.torch_version != H8_PROFILER_TORCH_VERSION:
        raise ValueError(
            "H8 config Torch version does not match the executable protocol"
        )
    if tuple(item[0] for item in _H8_NEGATIVE_CONTROL_CONTRACT) != (
        H8_NEGATIVE_CONTROL_IDS
    ):
        raise ValueError("H8 negative-control contract order drifted")
    return {
        "domain": "vfe4.h8.parent-child-protocol.v2",
        "validation_config": {
            "schema_version": config.schema_version,
            "config_sha256": config.config_sha256,
            "factor_schema": config.factor_schema,
            "selected_inverse_schema": config.selected_inverse_schema,
            "condition_estimator_schema": config.condition_estimator_schema,
            "allocation_schema": config.allocation_schema,
            "profiler_raw_event_schema": config.profiler_raw_event_schema,
            "child_schema": config.child_schema,
        },
        "child_contract": {
            "module": H8_CHILD_MODULE,
            "schema_version": H8_CHILD_SCHEMA_VERSION,
            "request_fields": H8_CHILD_REQUEST_KEYS,
            "envelope_fields": H8_CHILD_ENVELOPE_KEYS,
            "result_fields": H8_CHILD_RESULT_KEYS,
            "identity_fields": H8_CHILD_IDENTITY_KEYS,
        },
        "source_identities": {
            "generator_schema": config.generator_schema,
            "problem_draw_descriptor": config.problem_draw_descriptor,
            "problem_draw_schema_sha256": H8_PROBLEM_DRAW_SCHEMA_SHA256,
            "serialization_point": config.serialization_point,
            "sample_schema": config.sample_schema,
            "torch_version": config.torch_version,
            "profiler_memory_source_sha256": (
                H8_PROFILER_MEMORY_SOURCE_SHA256
            ),
            "profiler_source_sha256": H8_PROFILER_SOURCE_SHA256,
            "profiler_api_contract_sha256": (
                H8_PROFILER_API_CONTRACT_SHA256
            ),
            "profiler_api": {
                "torch_version": H8_PROFILER_TORCH_VERSION,
                "memory_profile_source_sha256": (
                    H8_PROFILER_MEMORY_SOURCE_SHA256
                ),
                "profiler_source_sha256": H8_PROFILER_SOURCE_SHA256,
                "api_contract_sha256": H8_PROFILER_API_CONTRACT_SHA256,
                "invocation": dict(H8_PROFILER_INVOCATION_ITEMS),
            },
        },
        "execution_inventories": {
            "production_order": tuple(
                (seed, repetition)
                for seed in H8_PRODUCTION_SEEDS
                for repetition in range(config.cold_repetitions)
            ),
            "production_sample_seed_pairs": (
                H8_PRODUCTION_SAMPLE_SEED_PAIRS
            ),
            "profiler_seed_order": H8_PRODUCTION_SEEDS,
            "cold_repetitions": config.cold_repetitions,
            "correctness_seed_table": H8_CORRECTNESS_CASES,
            "correctness_sources": H8_CORRECTNESS_SOURCES,
            "correctness_ordered_source_pairs": (
                H8_CORRECTNESS_ORDERED_SOURCE_PAIRS
            ),
            "correctness_control_ids": H8_CORRECTNESS_CONTROL_IDS,
            "required_operations": H8_REQUIRED_OPERATIONS,
            "operation_scopes": tuple(
                (name, H8_OPERATION_SCOPES[name])
                for name in H8_REQUIRED_OPERATIONS
            ),
            "required_residuals": H8_REQUIRED_RESIDUALS,
            "setup_scopes": tuple(sorted(H8_SETUP_SCOPES)),
            "scale_residual_specs": tuple(
                (
                    residual_id,
                    H8_SCALE_RESIDUAL_SPECS[residual_id],
                )
                for residual_id in H8_REQUIRED_RESIDUALS
            ),
            "required_pass_decisions": H8_REQUIRED_PASS_DECISIONS,
            "required_numpy_producers": tuple(
                sorted(H8_REQUIRED_NUMPY_PRODUCERS)
            ),
            "negative_controls": tuple(
                {
                    "control_id": control_id,
                    "requested_operation": requested_operation,
                    "assigned_channels": assigned_channels,
                }
                for (
                    control_id,
                    requested_operation,
                    assigned_channels,
                ) in _H8_NEGATIVE_CONTROL_CONTRACT
            ),
        },
        "numerical_contract": {
            "eps": EPS,
            "rounding_multiplier": ROUNDING_MULTIPLIER,
            "solver_relative_budget": SOLVER_RELATIVE_BUDGET,
            "max_allowance_fraction": MAX_ALLOWANCE_FRACTION,
            "minimum_cholesky_pivot": H8_MIN_CHOLESKY_PIVOT,
            "condition_estimator": {
                "schema": config.condition_estimator_schema,
                "norm": "matrix_1_norm",
                "maximum_iterations": (
                    H8_HAGER_HIGHAM_MAXIMUM_ITERATIONS
                ),
                "policy": H8_HAGER_HIGHAM_1NORM_POLICY,
                "estimate_is_diagnostic_not_exact_spectrum": True,
            },
            "residual_allowance_policy": {
                "allowance_sum": "math.fsum",
                "component_order": (
                    "left_rounding",
                    "left_solver",
                    "left_quadrature",
                    "right_rounding",
                    "right_solver",
                    "right_quadrature",
                    "pair_reduction",
                ),
                "gamma": "n_times_eps_over_1_minus_n_times_eps",
                "operand_rounding": (
                    "rounding_multiplier_times_gamma_local_operation_count"
                    "_times_max_1_absolute_sum_bound"
                ),
                "operand_solver": (
                    "solver_relative_budget_times_max_1_infinity_norm"
                    "_iff_solver_produced_else_zero"
                ),
                "pair_reduction": (
                    "rounding_multiplier_times_gamma_compared_scalar_count"
                    "_plus_1_times_max_1_left_inf_right_inf"
                ),
                "scale": (
                    "max(1,left_infinity_norm,right_infinity_norm)"
                ),
                "decisive_operator": "<",
                "decisive_fraction": MAX_ALLOWANCE_FRACTION,
                "decisive_equality_status": "inconclusive",
                "residual_pass_operator": "<=",
                "residual_equality_status": "pass",
                "condition_estimate_in_allowance": False,
            },
        },
        "boundary_contract": {
            "limits_are_inclusive": True,
            "max_seconds": H8_MAX_SECONDS,
            "max_process_incremental_bytes": (
                H8_MAX_PROCESS_INCREMENTAL_BYTES
            ),
            "max_torch_population_bytes": (
                H8_MAX_TORCH_POPULATION_BYTES
            ),
            "max_storage_scalars_per_category": H8_MAX_STORAGE_SCALARS,
            "max_rhs_width": config.max_rhs_width,
            "sample_width": config.sample_width,
            "offband_fill_limit": 0,
            "forbidden_attempt_limit": 0,
        },
        "runtime_contract": {
            "device": "cpu",
            "dtype": "float64",
            "grad_enabled": False,
            "scale_layout": {
                "horizon": H8_LAYOUT_HORIZON,
                "d_z": H8_LAYOUT_D_Z,
                "d_m": H8_LAYOUT_D_M,
            },
            "thread_environment": H8_THREAD_ENVIRONMENT_ITEMS,
            "torch_num_threads": H8_TORCH_NUM_THREADS,
            "torch_num_interop_threads": H8_TORCH_NUM_INTEROP_THREADS,
            "fresh_process_per_request": True,
            "launch": {
                "argv_tail": ("-m", H8_CHILD_MODULE),
                "canonical_stdin_one_line": True,
                "capture_stdout": True,
                "capture_stderr": True,
                "timeout_seconds": H8_MAX_SECONDS,
            },
            "conservative_hwm_formulas": {
                "primary": (
                    "max(0,post_lifetime_peak-pre_current_rss)"
                ),
                "supplementary": (
                    "max(0,post_lifetime_peak-pre_lifetime_peak)"
                ),
            },
        },
        "parent_policy": {
            "attempt_policy": _H8_PARENT_POLICY,
            "timeout_seconds": H8_MAX_SECONDS,
            "capture_stdout": True,
            "capture_stderr": True,
        },
    }


def build_h8_protocol_sha256(config: H8ValidationConfig) -> str:
    """Bind every frozen parent/child protocol decision to one digest."""

    preimage = _h8_protocol_preimage(config)
    return hashlib.sha256(canonical_json_bytes(preimage)).hexdigest()


__all__ = ["_h8_protocol_preimage", "build_h8_protocol_sha256"]
