"""Parent-owned planning and issued-prefix execution for the frozen H8 protocol."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

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
    H8ChildInvocation,
    H8ChildProcessRecord,
    build_h8_child_invocation,
    canonical_json_bytes,
    classify_h8_child_outcome,
    make_h8_child_attempt_record,
    run_h8_child,
    _require_h8_issued_attempt_binding,
)
from verification.h8_gate import H8PrerequisiteArtifactValidation
from verification.h8_wire import (
    H8_CHILD_ENVELOPE_KEYS,
    H8_CHILD_IDENTITY_ENV,
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
    H8ChildAttemptRecord,
    H8ChildRequest,
)
from vfe4.types.results import GateStatus


_HEX = frozenset("0123456789abcdef")
_H8_CHILD_MODULE = H8_CHILD_MODULE
_H8_PARENT_POLICY = "continue_after_inconclusive_stop_after_first_fail"
_H8_REVALIDATED_REFERENCE_NAMES = (
    "h1_h5",
    "h1_prefix_prior",
    "h6_prefix",
    "h7",
    "h6_prediction",
)
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


def _sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _canonical_request_bytes(request: H8ChildRequest) -> bytes:
    return canonical_json_bytes(
        {
            "mode": request.mode,
            "seed": request.seed,
            "repetition": request.repetition,
            "config_sha256": request.config_sha256,
            "protocol_sha256": request.protocol_sha256,
            "control_id": request.control_id,
        }
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
            "module": _H8_CHILD_MODULE,
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
                "argv_tail": ("-m", _H8_CHILD_MODULE),
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


@dataclass(frozen=True, slots=True, init=False)
class H8ChildStartAuthorization:
    """Factory-derived, immutable authority for the exact H8 child prefix."""

    config_sha256: str
    protocol_sha256: str
    current_registry_sha256: str
    prerequisite_validation: H8PrerequisiteArtifactValidation
    prerequisite_validation_sha256: str
    correctness_statuses: tuple[tuple[int, GateStatus], ...]
    obligations: tuple[str, ...]
    valid_start: bool
    authorization_sha256: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError(
            "H8ChildStartAuthorization is factory-only; "
            "use derive_h8_child_start_authorization"
        )


_ISSUED_H8_AUTHORIZATIONS: dict[
    int,
    tuple[H8ChildStartAuthorization, bytes],
] = {}


def _validate_correctness_statuses(
    correctness_statuses: object,
) -> tuple[tuple[int, GateStatus], ...]:
    if type(correctness_statuses) is not tuple or any(
        type(item) is not tuple
        or len(item) != 2
        or type(item[0]) is not int
        or type(item[1]) is not GateStatus
        for item in correctness_statuses
    ):
        raise ValueError(
            "correctness_statuses must contain exact cell IDs and GateStatus values"
        )
    return correctness_statuses


def _authorization_obligations(
    *,
    current_registry_sha256: str,
    prerequisite_validation: H8PrerequisiteArtifactValidation,
    correctness_statuses: tuple[tuple[int, GateStatus], ...],
) -> tuple[str, ...]:
    obligations = list(prerequisite_validation.obligations)
    if prerequisite_validation.registry_sha256 != current_registry_sha256:
        obligations.append("h8_prerequisite_validation_registry_mismatch")
    if (
        prerequisite_validation.revalidated_reference_names
        != _H8_REVALIDATED_REFERENCE_NAMES
    ):
        obligations.append("h8_prerequisite_revalidation_inventory_incomplete")
    if tuple(cell_id for cell_id, _status in correctness_statuses) != tuple(
        range(1, 13)
    ):
        obligations.append("h8_correctness_cell_inventory_incomplete")
    if not all(
        status is GateStatus.PASS for _cell_id, status in correctness_statuses
    ):
        obligations.append("h8_correctness_cells_not_all_pass")
    return tuple(dict.fromkeys(obligations))


def _authorization_preimage(
    *,
    config_sha256: str,
    protocol_sha256: str,
    current_registry_sha256: str,
    prerequisite_validation_sha256: str,
    correctness_statuses: tuple[tuple[int, GateStatus], ...],
    obligations: tuple[str, ...],
    valid_start: bool,
) -> dict[str, object]:
    return {
        "domain": "vfe4.h8.child-start-authorization.v1",
        "config_sha256": config_sha256,
        "protocol_sha256": protocol_sha256,
        "current_registry_sha256": current_registry_sha256,
        "prerequisite_validation_sha256": prerequisite_validation_sha256,
        "correctness_statuses": tuple(
            (cell_id, status.value)
            for cell_id, status in correctness_statuses
        ),
        "obligations": obligations,
        "valid_start": valid_start,
    }


def _validate_h8_child_start_authorization(
    authorization: object,
) -> H8ChildStartAuthorization:
    if type(authorization) is not H8ChildStartAuthorization:
        raise ValueError(
            "authorization must be an exact H8ChildStartAuthorization"
        )
    issued = _ISSUED_H8_AUTHORIZATIONS.get(id(authorization))
    if issued is None or issued[0] is not authorization:
        raise ValueError("authorization must be factory-issued")
    config_sha256 = _sha256(
        authorization.config_sha256,
        "authorization.config_sha256",
    )
    protocol_sha256 = _sha256(
        authorization.protocol_sha256,
        "authorization.protocol_sha256",
    )
    registry_sha256 = _sha256(
        authorization.current_registry_sha256,
        "authorization.current_registry_sha256",
    )
    if (
        type(authorization.prerequisite_validation)
        is not H8PrerequisiteArtifactValidation
    ):
        raise ValueError(
            "authorization prerequisite validation is not exact"
        )
    authorization.prerequisite_validation.__post_init__()
    prerequisite_validation_sha256 = _sha256(
        authorization.prerequisite_validation_sha256,
        "authorization.prerequisite_validation_sha256",
    )
    if (
        prerequisite_validation_sha256
        != authorization.prerequisite_validation.validation_sha256
    ):
        raise ValueError(
            "authorization prerequisite-validation SHA-256 is stale"
        )
    correctness_statuses = _validate_correctness_statuses(
        authorization.correctness_statuses
    )
    expected_obligations = _authorization_obligations(
        current_registry_sha256=registry_sha256,
        prerequisite_validation=authorization.prerequisite_validation,
        correctness_statuses=correctness_statuses,
    )
    if (
        type(authorization.obligations) is not tuple
        or any(
            type(item) is not str or not item
            for item in authorization.obligations
        )
        or len(set(authorization.obligations))
        != len(authorization.obligations)
        or authorization.obligations != expected_obligations
    ):
        raise ValueError("authorization obligations are stale")
    if (
        type(authorization.valid_start) is not bool
        or authorization.valid_start is not (not expected_obligations)
    ):
        raise ValueError("authorization start decision is stale")
    preimage_bytes = canonical_json_bytes(
        _authorization_preimage(
            config_sha256=config_sha256,
            protocol_sha256=protocol_sha256,
            current_registry_sha256=registry_sha256,
            prerequisite_validation_sha256=(
                prerequisite_validation_sha256
            ),
            correctness_statuses=correctness_statuses,
            obligations=expected_obligations,
            valid_start=authorization.valid_start,
        )
    )
    if preimage_bytes != issued[1]:
        raise ValueError("authorization owned preimage is stale")
    expected_sha256 = hashlib.sha256(preimage_bytes).hexdigest()
    if authorization.authorization_sha256 != expected_sha256:
        raise ValueError("authorization SHA-256 is stale")
    return authorization


def derive_h8_child_start_authorization(
    *,
    config: H8ValidationConfig,
    current_registry_sha256: str,
    prerequisite_validation: H8PrerequisiteArtifactValidation,
    correctness_statuses: tuple[tuple[int, GateStatus], ...],
) -> H8ChildStartAuthorization:
    """Derive child-start authority without accepting caller-owned validity."""

    if type(config) is not H8ValidationConfig:
        raise ValueError("config must be an exact H8ValidationConfig")
    config_sha256 = _sha256(config.config_sha256, "config.config_sha256")
    protocol_sha256 = build_h8_protocol_sha256(config)
    registry_sha256 = _sha256(
        current_registry_sha256,
        "current_registry_sha256",
    )
    if type(prerequisite_validation) is not H8PrerequisiteArtifactValidation:
        raise ValueError(
            "prerequisite_validation must be an exact "
            "H8PrerequisiteArtifactValidation"
        )
    prerequisite_validation.__post_init__()
    checked_statuses = _validate_correctness_statuses(correctness_statuses)
    obligations = _authorization_obligations(
        current_registry_sha256=registry_sha256,
        prerequisite_validation=prerequisite_validation,
        correctness_statuses=checked_statuses,
    )
    valid_start = not obligations
    preimage_bytes = canonical_json_bytes(
        _authorization_preimage(
            config_sha256=config_sha256,
            protocol_sha256=protocol_sha256,
            current_registry_sha256=registry_sha256,
            prerequisite_validation_sha256=(
                prerequisite_validation.validation_sha256
            ),
            correctness_statuses=checked_statuses,
            obligations=obligations,
            valid_start=valid_start,
        )
    )
    authorization = object.__new__(H8ChildStartAuthorization)
    for name, value in (
        ("config_sha256", config_sha256),
        ("protocol_sha256", protocol_sha256),
        ("current_registry_sha256", registry_sha256),
        ("prerequisite_validation", prerequisite_validation),
        (
            "prerequisite_validation_sha256",
            prerequisite_validation.validation_sha256,
        ),
        ("correctness_statuses", checked_statuses),
        ("obligations", obligations),
        ("valid_start", valid_start),
        (
            "authorization_sha256",
            hashlib.sha256(preimage_bytes).hexdigest(),
        ),
    ):
        object.__setattr__(authorization, name, value)
    _ISSUED_H8_AUTHORIZATIONS[id(authorization)] = (
        authorization,
        preimage_bytes,
    )
    return _validate_h8_child_start_authorization(authorization)


def build_h8_child_request_plan(
    *,
    config_sha256: str,
    protocol_sha256: str,
) -> tuple[H8ChildRequest, ...]:
    """Return the immutable, ordered 30-request H8 child plan."""

    return (
        *(
            H8ChildRequest(
                mode="production",
                seed=seed,
                repetition=repetition,
                config_sha256=config_sha256,
                protocol_sha256=protocol_sha256,
                control_id=None,
            )
            for seed in H8_PRODUCTION_SEEDS
            for repetition in range(5)
        ),
        *(
            H8ChildRequest(
                mode="profiler",
                seed=seed,
                repetition=None,
                config_sha256=config_sha256,
                protocol_sha256=protocol_sha256,
                control_id=None,
            )
            for seed in H8_PRODUCTION_SEEDS
        ),
        *(
            H8ChildRequest(
                mode="negative_control",
                seed=H8_PRODUCTION_SEEDS[0],
                repetition=None,
                config_sha256=config_sha256,
                protocol_sha256=protocol_sha256,
                control_id=control_id,
            )
            for control_id in H8_NEGATIVE_CONTROL_IDS
        ),
    )


def _resolved_repository_root(repository_root: str | Path) -> Path:
    root = Path(repository_root).resolve()
    if not root.is_absolute() or not root.is_dir():
        raise ValueError(
            "repository_root must resolve to an existing absolute directory"
        )
    return root


def _parent_identity_json_bytes(invocation: H8ChildInvocation) -> bytes:
    identity_json = invocation.environment.get(H8_CHILD_IDENTITY_ENV)
    if type(identity_json) is not str:
        raise ValueError("parent identity JSON is absent from the launch")
    try:
        identity_bytes = identity_json.encode("ascii")
        if canonical_json_bytes(json.loads(identity_bytes)) != identity_bytes:
            raise ValueError
    except (UnicodeEncodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("parent identity JSON bytes are not canonical") from error
    return identity_bytes


def build_h8_launch_contract_sha256(
    invocation: H8ChildInvocation,
    *,
    repository_root: str | Path,
) -> str:
    """Bind an exact launch, including raw stdin and parent identity bytes."""

    if type(invocation) is not H8ChildInvocation:
        raise ValueError("invocation must be an exact H8ChildInvocation")
    invocation.__post_init__()
    root = _resolved_repository_root(repository_root)
    if invocation.cwd.resolve() != root:
        raise ValueError("repository_root must equal the invocation cwd")
    identity_bytes = _parent_identity_json_bytes(invocation)
    preimage = {
        "domain": "vfe4.h8.parent-launch-contract.v1",
        "repository_root": str(root),
        "argv": invocation.argv,
        "stdin_hex": invocation.stdin.hex(),
        "environment": dict(invocation.environment),
        "parent_identity_json_hex": identity_bytes.hex(),
        "timeout_seconds": invocation.timeout_seconds,
        "capture_stdout": invocation.capture_stdout,
        "capture_stderr": invocation.capture_stderr,
    }
    return hashlib.sha256(canonical_json_bytes(preimage)).hexdigest()


@dataclass(frozen=True, slots=True)
class H8IssuedLaunchRecord:
    """One issued H8 launch with every parent-owned boundary retained."""

    request: H8ChildRequest
    invocation: H8ChildInvocation
    process_record: H8ChildProcessRecord
    repository_root: Path
    launch_contract_sha256: str
    attempt: H8ChildAttemptRecord

    def __post_init__(self) -> None:
        if type(self.request) is not H8ChildRequest:
            raise ValueError("request must be an exact H8ChildRequest")
        self.request.__post_init__()
        if type(self.invocation) is not H8ChildInvocation:
            raise ValueError("invocation must be an exact H8ChildInvocation")
        if type(self.process_record) is not H8ChildProcessRecord:
            raise ValueError(
                "process_record must be an exact H8ChildProcessRecord"
            )
        self.process_record.__post_init__()
        root = _resolved_repository_root(self.repository_root)
        if self.repository_root != root or self.invocation.cwd != root:
            raise ValueError("issued launch repository root is not exact")
        expected_contract = build_h8_launch_contract_sha256(
            self.invocation,
            repository_root=root,
        )
        if self.launch_contract_sha256 != expected_contract:
            raise ValueError("issued launch contract SHA-256 is stale")
        if self.invocation.stdin != _canonical_request_bytes(self.request) + b"\n":
            raise ValueError("issued request does not match invocation stdin")
        if type(self.attempt) is not H8ChildAttemptRecord:
            raise ValueError(
                "issued request, process, and attempt identities are not cross-bound"
            )
        _require_h8_issued_attempt_binding(
            self.attempt,
            request=self.request,
            invocation=self.invocation,
            process_record=self.process_record,
        )


@dataclass(frozen=True, slots=True)
class H8ParentAttemptRun:
    """The exact preallocated plan and the immutable issued launch prefix."""

    authorization: H8ChildStartAuthorization
    request_plan: tuple[H8ChildRequest, ...]
    issued: tuple[H8IssuedLaunchRecord, ...]
    attempts: tuple[H8ChildAttemptRecord, ...]

    def __post_init__(self) -> None:
        _validate_h8_child_start_authorization(self.authorization)
        if (
            type(self.request_plan) is not tuple
            or any(type(item) is not H8ChildRequest for item in self.request_plan)
            or type(self.issued) is not tuple
            or any(type(item) is not H8IssuedLaunchRecord for item in self.issued)
            or type(self.attempts) is not tuple
            or any(
                type(item) is not H8ChildAttemptRecord for item in self.attempts
            )
        ):
            raise ValueError("parent run inventories must be exact typed tuples")
        for item in self.issued:
            item.__post_init__()
        if not self.authorization.valid_start:
            if self.request_plan or self.issued or self.attempts:
                raise ValueError(
                    "unauthorized parent runs cannot retain issued records"
                )
            return
        expected_plan = build_h8_child_request_plan(
            config_sha256=self.authorization.config_sha256,
            protocol_sha256=self.authorization.protocol_sha256,
        )
        if self.request_plan != expected_plan:
            raise ValueError("parent request plan is not the frozen plan")
        if tuple(item.request for item in self.issued) != self.request_plan[
            : len(self.issued)
        ]:
            raise ValueError("issued launches are not an exact request prefix")
        if tuple(item.attempt for item in self.issued) != self.attempts:
            raise ValueError("issued launches and attempts are not cross-bound")
        failed_indices = tuple(
            index
            for index, attempt in enumerate(self.attempts)
            if attempt.status is GateStatus.FAIL
        )
        if failed_indices and failed_indices != (len(self.attempts) - 1,):
            raise ValueError("parent run continued after the first witnessed FAIL")


def _abnormal_spawn_record(
    error: OSError,
    *,
    parent_elapsed_ns: int,
) -> H8ChildProcessRecord:
    error_code = (
        -abs(error.errno)
        if type(error.errno) is int and error.errno != 0
        else -1
    )
    return H8ChildProcessRecord(
        timed_out=False,
        exit_code=error_code,
        stdout=b"",
        stderr=str(error).encode("utf-8", errors="replace"),
        parent_elapsed_ns=parent_elapsed_ns,
    )


def _run_h8_parent_attempt_with_runner(
    *,
    authorization: H8ChildStartAuthorization,
    repository_root: str | Path,
    identities: Mapping[str, object],
    base_environment: Mapping[str, str] | None = None,
    child_runner: Callable[
        [H8ChildInvocation],
        H8ChildProcessRecord,
    ] = run_h8_child,
) -> H8ParentAttemptRun:
    """Issue the authorized H8 prefix once, stopping at the first FAIL."""

    authorization = _validate_h8_child_start_authorization(authorization)
    if not authorization.valid_start:
        return H8ParentAttemptRun(
            authorization=authorization,
            request_plan=(),
            issued=(),
            attempts=(),
        )
    if not callable(child_runner):
        raise ValueError("child_runner must be callable")
    root = _resolved_repository_root(repository_root)
    request_plan = build_h8_child_request_plan(
        config_sha256=authorization.config_sha256,
        protocol_sha256=authorization.protocol_sha256,
    )
    issued: list[H8IssuedLaunchRecord] = []
    attempts: list[H8ChildAttemptRecord] = []
    for request in request_plan:
        invocation = build_h8_child_invocation(
            {
                name: getattr(request, name)
                for name in H8_CHILD_REQUEST_KEYS
            },
            repository_root=root,
            identities=identities,
            base_environment=base_environment,
        )
        launch_contract_sha256 = build_h8_launch_contract_sha256(
            invocation,
            repository_root=root,
        )
        started = time.perf_counter_ns()
        try:
            process_record = child_runner(invocation)
        except OSError as error:
            process_record = _abnormal_spawn_record(
                error,
                parent_elapsed_ns=time.perf_counter_ns() - started,
            )
        if type(process_record) is not H8ChildProcessRecord:
            raise ValueError(
                "child_runner must return an exact H8ChildProcessRecord"
            )
        process_record.__post_init__()
        if (
            build_h8_launch_contract_sha256(
                invocation,
                repository_root=root,
            )
            != launch_contract_sha256
        ):
            raise ValueError("launch contract drifted while the child ran")
        decision = classify_h8_child_outcome(
            process_record,
            valid_start=authorization.valid_start,
            invocation=invocation,
        )
        attempt = make_h8_child_attempt_record(
            request,
            invocation,
            process_record,
            decision,
        )
        issued.append(
            H8IssuedLaunchRecord(
                request=request,
                invocation=invocation,
                process_record=process_record,
                repository_root=root,
                launch_contract_sha256=launch_contract_sha256,
                attempt=attempt,
            )
        )
        attempts.append(attempt)
        if attempt.status is GateStatus.FAIL:
            break
    return H8ParentAttemptRun(
        authorization=authorization,
        request_plan=request_plan,
        issued=tuple(issued),
        attempts=tuple(attempts),
    )


def _run_h8_parent_attempt_for_test(
    *,
    authorization: H8ChildStartAuthorization,
    repository_root: str | Path,
    identities: Mapping[str, object],
    child_runner: Callable[
        [H8ChildInvocation],
        H8ChildProcessRecord,
    ],
    base_environment: Mapping[str, str] | None = None,
) -> H8ParentAttemptRun:
    """Private fake-runner seam; production authority never accepts it."""

    return _run_h8_parent_attempt_with_runner(
        authorization=authorization,
        repository_root=repository_root,
        identities=identities,
        base_environment=base_environment,
        child_runner=child_runner,
    )


def run_h8_parent_attempt(
    *,
    authorization: H8ChildStartAuthorization,
    repository_root: str | Path,
    identities: Mapping[str, object],
    base_environment: Mapping[str, str] | None = None,
) -> H8ParentAttemptRun:
    """Run the fixed production child entry point for one authorized prefix."""

    return _run_h8_parent_attempt_with_runner(
        authorization=authorization,
        repository_root=repository_root,
        identities=identities,
        base_environment=base_environment,
        child_runner=run_h8_child,
    )


__all__ = [
    "H8ChildStartAuthorization",
    "H8IssuedLaunchRecord",
    "H8ParentAttemptRun",
    "build_h8_child_request_plan",
    "build_h8_launch_contract_sha256",
    "build_h8_protocol_sha256",
    "derive_h8_child_start_authorization",
    "run_h8_parent_attempt",
]
