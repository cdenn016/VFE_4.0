"""Pure H6 arm capacity, ownership, and arithmetic matching.

Candidate enumeration is formula-only and lazy.  This module never opens a
corpus, evaluates a model, inspects gradients, or reads a predictive metric.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

from vfe4.config.schema import H6ArmMatchingResolvedConfig
from vfe4.types.h6 import (
    AdamWPolicyRecord,
    ArmConfig,
    ArmMatrixRow,
    CapacityAllocation,
    FlopTerm,
    MatchingReport,
    OptimizerBinding,
    ParameterRoleRecord,
    TrainingPhase,
    canonical_json_bytes,
)


H6_ADAMW_POLICY = AdamWPolicyRecord.create()
A5_REFERENCE_ALLOCATION = CapacityAllocation.create(
    emission_width=64,
    latent_width=16,
    recognition_width=64,
)


def _hash(domain: bytes, payload: object) -> str:
    return hashlib.sha256(domain + b"\x00" + canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class MatchingSchedulePolicy:
    reference_allocation_sha256: str
    emission_width_candidates: tuple[int, ...]
    latent_width_candidates: tuple[int, ...]
    recognition_width_candidates: tuple[int, ...]
    parameter_relative_tolerance: float
    flop_relative_tolerance: float
    optimizer_policy_sha256: str
    full_passes: int
    model_updates_per_batch: int
    validation_boundary_policy: str
    checkpoint_boundary_policy: str
    excluded_operations: tuple[str, ...]
    policy_sha256: str

    def __post_init__(self) -> None:
        expected_fields = (
            A5_REFERENCE_ALLOCATION.allocation_sha256,
            (48, 64, 80, 96),
            (8, 16, 24, 32),
            (32, 64, 96),
            0.01,
            0.05,
            H6_ADAMW_POLICY.optimizer_policy_sha256,
            2,
            1,
            "twentieths_of_each_pass_v1",
            "terminal_only_v1",
            (
                "data_io",
                "validation",
                "checkpoint_serialization",
                "test_scoring",
            ),
        )
        if tuple(
            getattr(self, name)
            for name in tuple(self.__dataclass_fields__)[:-1]
        ) != expected_fields:
            raise ValueError("matching schedule policy is frozen")
        expected = _hash(
            b"vfe4.h6.matching-schedule-policy.v1",
            {
                name: getattr(self, name)
                for name in tuple(self.__dataclass_fields__)[:-1]
            },
        )
        if self.policy_sha256 != expected:
            raise ValueError("matching schedule policy digest does not match")


_MATCHING_SCHEDULE_PAYLOAD = {
    "reference_allocation_sha256": (
        A5_REFERENCE_ALLOCATION.allocation_sha256
    ),
    "emission_width_candidates": (48, 64, 80, 96),
    "latent_width_candidates": (8, 16, 24, 32),
    "recognition_width_candidates": (32, 64, 96),
    "parameter_relative_tolerance": 0.01,
    "flop_relative_tolerance": 0.05,
    "optimizer_policy_sha256": H6_ADAMW_POLICY.optimizer_policy_sha256,
    "full_passes": 2,
    "model_updates_per_batch": 1,
    "validation_boundary_policy": "twentieths_of_each_pass_v1",
    "checkpoint_boundary_policy": "terminal_only_v1",
    "excluded_operations": (
        "data_io",
        "validation",
        "checkpoint_serialization",
        "test_scoring",
    ),
}
MATCHING_SCHEDULE_POLICY = MatchingSchedulePolicy(
    **_MATCHING_SCHEDULE_PAYLOAD,
    policy_sha256=_hash(
        b"vfe4.h6.matching-schedule-policy.v1",
        _MATCHING_SCHEDULE_PAYLOAD,
    ),
)
EMISSION_WIDTH_CANDIDATES = (
    MATCHING_SCHEDULE_POLICY.emission_width_candidates
)
LATENT_WIDTH_CANDIDATES = (
    MATCHING_SCHEDULE_POLICY.latent_width_candidates
)
RECOGNITION_WIDTH_CANDIDATES = (
    MATCHING_SCHEDULE_POLICY.recognition_width_candidates
)


def _require_matching_config(
    matching_config: H6ArmMatchingResolvedConfig,
) -> H6ArmMatchingResolvedConfig:
    if type(matching_config) is not H6ArmMatchingResolvedConfig:
        raise ValueError(
            "matching_config must be an exact H6ArmMatchingResolvedConfig"
        )
    matching_config.__post_init__()
    expected = MATCHING_SCHEDULE_POLICY
    if (
        matching_config.adamw_policy != H6_ADAMW_POLICY
        or matching_config.reference_allocation
        != A5_REFERENCE_ALLOCATION
        or matching_config.reference_allocation.allocation_sha256
        != expected.reference_allocation_sha256
        or matching_config.emission_width_candidates
        != expected.emission_width_candidates
        or matching_config.latent_width_candidates
        != expected.latent_width_candidates
        or matching_config.recognition_width_candidates
        != expected.recognition_width_candidates
        or matching_config.parameter_relative_tolerance
        != expected.parameter_relative_tolerance
        or matching_config.flop_relative_tolerance
        != expected.flop_relative_tolerance
        or matching_config.matching_schedule_sha256
        != expected.policy_sha256
    ):
        raise ValueError(
            "resolved matching config does not equal the executable canonical policy"
        )
    if (
        matching_config.arm_configs[5].capacity_allocation
        != matching_config.reference_allocation
    ):
        raise ValueError(
            "resolved A5 allocation does not equal the reference allocation"
        )
    return matching_config

_A5_CONFIG_ID = (
    "h6-a5-structured-fixed-exact-complete-latent-smoothing-v1"
)
_A5_FACTORY_ID = "build_a5@h6-arm-v1"
_CHECKPOINT_TEMPLATE = (
    "checkpoints/{config_sha256}/{seed}/terminal.pt"
)
_CERTIFICATE_KEY_TEMPLATE = (
    "certificates/{prefix_case_key_sha256}.json"
)
_OPENING_GROUP = "h6-prediction-global-test-opening-v1"


def _matrix_row(
    *,
    row_id: str,
    left_config_id: str,
    left_factory_id: str,
    right_config_id: str,
    right_factory_id: str,
    named_factor: str,
    tuning_estimand: str,
    interpretation: str,
    nonclaim: str,
    additional_nonclaims: tuple[str, ...] = (),
) -> ArmMatrixRow:
    return ArmMatrixRow.create(
        row_id=row_id,
        left_config_id=left_config_id,
        left_factory_id=left_factory_id,
        right_config_id=right_config_id,
        right_factory_id=right_factory_id,
        named_factor=named_factor,
        semantic_interventions=(named_factor,),
        nuisance_capacity_fields=(
            "emission_width",
            "latent_width",
            "recognition_width",
        ),
        tuning_estimand=tuning_estimand,
        interpretation=interpretation,
        checkpoint_template=_CHECKPOINT_TEMPLATE,
        certificate_key_template=_CERTIFICATE_KEY_TEMPLATE,
        opening_group=_OPENING_GROUP,
        nonclaims=(nonclaim,) + additional_nonclaims,
    )


ARM_MATRIX_ROWS = (
    _matrix_row(
        row_id="PRIMARY",
        left_config_id="h6-a0-ar-v1",
        left_factory_id="build_a0@h6-arm-v1",
        right_config_id=_A5_CONFIG_ID,
        right_factory_id=_A5_FACTORY_ID,
        named_factor="whole_declared_architecture",
        tuning_estimand="equal_grid",
        interpretation="primary",
        nonclaim="not_component_attribution",
    ),
    _matrix_row(
        row_id="MAP",
        left_config_id="h6-a2-generic-map-v1",
        left_factory_id="build_a2@h6-arm-v1",
        right_config_id=_A5_CONFIG_ID,
        right_factory_id=_A5_FACTORY_ID,
        named_factor="map_mode",
        tuning_estimand="equal_grid",
        interpretation="conditional",
        nonclaim="not_h7_covariance",
        additional_nonclaims=(
            "generic_fixed_frame_non_coboundary_not_h7_covariance",
            "not_connection_curvature_or_holonomy",
        ),
    ),
    _matrix_row(
        row_id="STRUCTURE",
        left_config_id=(
            "h6-a5-factorized-fixed-exact-complete-latent-smoothing-v1"
        ),
        left_factory_id=_A5_FACTORY_ID,
        right_config_id=_A5_CONFIG_ID,
        right_factory_id=_A5_FACTORY_ID,
        named_factor="recognition_family",
        tuning_estimand="shared_a5",
        interpretation="conditional",
        nonclaim="conditional_on_a5_tuning",
    ),
    _matrix_row(
        row_id="PRIOR",
        left_config_id=_A5_CONFIG_ID,
        left_factory_id=_A5_FACTORY_ID,
        right_config_id=(
            "h6-a5-structured-prefix-exact-complete-latent-smoothing-v1"
        ),
        right_factory_id=_A5_FACTORY_ID,
        named_factor="prior_variant",
        tuning_estimand="shared_a5",
        interpretation="descriptive",
        nonclaim="changed_joint_descriptive",
    ),
    _matrix_row(
        row_id="MIXTURE",
        left_config_id=_A5_CONFIG_ID,
        left_factory_id=_A5_FACTORY_ID,
        right_config_id=(
            "h6-a5-structured-fixed-projection-complete-latent-smoothing-v1"
        ),
        right_factory_id=_A5_FACTORY_ID,
        named_factor="mixture_mode",
        tuning_estimand="shared_a5",
        interpretation="descriptive",
        nonclaim="projection_not_exact",
    ),
    _matrix_row(
        row_id="OBJECTIVE",
        left_config_id=_A5_CONFIG_ID,
        left_factory_id=_A5_FACTORY_ID,
        right_config_id=(
            "h6-a5-structured-fixed-exact-emission-latent-smoothing-v1"
        ),
        right_factory_id=_A5_FACTORY_ID,
        named_factor="objective_kind",
        tuning_estimand="shared_a5",
        interpretation="conditional",
        nonclaim="emission_not_elbo",
    ),
    _matrix_row(
        row_id="LATENT",
        left_config_id=_A5_CONFIG_ID,
        left_factory_id=_A5_FACTORY_ID,
        right_config_id=(
            "h6-a5-structured-fixed-exact-complete-"
            "nolatent-norecognition-v1"
        ),
        right_factory_id=_A5_FACTORY_ID,
        named_factor="latent_channel",
        tuning_estimand="shared_a5",
        interpretation="descriptive",
        nonclaim="latent_capacity_descriptive",
    ),
    _matrix_row(
        row_id="RECOGNITION",
        left_config_id=_A5_CONFIG_ID,
        left_factory_id=_A5_FACTORY_ID,
        right_config_id=(
            "h6-a5-structured-fixed-exact-complete-latent-filtering-v1"
        ),
        right_factory_id=_A5_FACTORY_ID,
        named_factor="recognition_conditioning",
        tuning_estimand="shared_a5",
        interpretation="conditional",
        nonclaim="recognition_not_used_for_scoring",
    ),
)


def arm_matrix_sha256(rows: tuple[ArmMatrixRow, ...]) -> str:
    """Hash the ordered, exact eight-row attribution matrix."""

    if (
        type(rows) is not tuple
        or len(rows) != 8
        or any(type(row) is not ArmMatrixRow for row in rows)
        or tuple(row.row_id for row in rows)
        != (
            "PRIMARY",
            "MAP",
            "STRUCTURE",
            "PRIOR",
            "MIXTURE",
            "OBJECTIVE",
            "LATENT",
            "RECOGNITION",
        )
    ):
        raise ValueError("arm matrix must contain the exact ordered eight rows")
    return _hash(
        b"vfe4.h6.arm-matrix.v1",
        tuple(row.row_sha256 for row in rows),
    )


ARM_MATRIX_SHA256 = arm_matrix_sha256(ARM_MATRIX_ROWS)


def _positive_dimension(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_count(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def dense_matmul_flops(m: int, n: int, k: int) -> int:
    """Return the frozen ``2*m*n*k`` arithmetic convention."""

    return (
        2
        * _positive_dimension(m, "m")
        * _positive_dimension(n, "n")
        * _positive_dimension(k, "k")
    )


def dense_matvec_flops(m: int, n: int) -> int:
    """Return the frozen ``2*m*n`` arithmetic convention."""

    return 2 * _positive_dimension(m, "m") * _positive_dimension(n, "n")


def scalar_flops(count: int) -> int:
    """Count scalar arithmetic operations one-for-one."""

    return _nonnegative_count(count, "count")


def log_softmax_flops(length: int) -> int:
    """Return the frozen length-``n`` ``5*n-1`` convention."""

    return 5 * _positive_dimension(length, "length") - 1


def backward_flops(differentiable_forward_flops: int) -> int:
    """Return twice the differentiable forward arithmetic."""

    return 2 * _nonnegative_count(
        differentiable_forward_flops,
        "differentiable_forward_flops",
    )


def l2_clip_scale_flops(active_gradient_scalars: int) -> int:
    """Return the always-evaluated global L2 clip/scale cost ``3*P+3``."""

    return 3 * _nonnegative_count(
        active_gradient_scalars, "active_gradient_scalars"
    ) + 3


def adamw_flops(active_parameter_scalars: int) -> int:
    """Return the frozen AdamW cost ``18*P``."""

    return 18 * _nonnegative_count(
        active_parameter_scalars, "active_parameter_scalars"
    )


def immutable_snapshot_flop_term(
    *, repetitions: int, bytes_copied_per_repetition: int
) -> FlopTerm:
    """Record detached snapshot traffic without inventing arithmetic FLOPs."""

    return FlopTerm.create(
        phase=TrainingPhase.IMMUTABLE_DETACHED_SNAPSHOT.value,
        operation="immutable_detached_snapshot",
        repetitions=repetitions,
        arithmetic_flops_per_repetition=0,
        bytes_copied_per_repetition=bytes_copied_per_repetition,
    )


def candidate_allocations(
    config: ArmConfig,
    *,
    matching_config: H6ArmMatchingResolvedConfig,
) -> Iterator[CapacityAllocation]:
    """Yield the applicable literal Cartesian product in field order."""

    if type(config) is not ArmConfig:
        raise ValueError("config must be an ArmConfig")
    policy = _require_matching_config(matching_config)
    latent_axis: tuple[int | None, ...] = (
        policy.latent_width_candidates if config.latent_enabled else (None,)
    )
    recognition_axis: tuple[int | None, ...] = (
        policy.recognition_width_candidates
        if config.recognition_family != "absent"
        else (None,)
    )
    for emission_width in policy.emission_width_candidates:
        for latent_width in latent_axis:
            for recognition_width in recognition_axis:
                yield CapacityAllocation.create(
                    emission_width=emission_width,
                    latent_width=latent_width,
                    recognition_width=recognition_width,
                )


def capacity_candidate_count(
    config: ArmConfig,
    *,
    matching_config: H6ArmMatchingResolvedConfig,
) -> int:
    """Return the exact formula-only candidate count without enumeration."""

    if type(config) is not ArmConfig:
        raise ValueError("config must be an ArmConfig")
    policy = _require_matching_config(matching_config)
    return (
        len(policy.emission_width_candidates)
        * (
            len(policy.latent_width_candidates)
            if config.latent_enabled
            else 1
        )
        * (
            len(policy.recognition_width_candidates)
            if config.recognition_family != "absent"
            else 1
        )
    )


class _ParameterLike(Protocol):
    requires_grad: bool

    def numel(self) -> int: ...


class _ModuleLike(Protocol):
    def named_parameters(
        self, *, remove_duplicate: bool
    ) -> Iterator[tuple[str, _ParameterLike]]: ...


class _BuiltArmLike(Protocol):
    config: ArmConfig
    model: _ModuleLike
    recognition_store: _ModuleLike | None
    parameter_roles: tuple[ParameterRoleRecord, ...]
    optimizer_bindings: tuple[OptimizerBinding, ...]
    flop_terms: tuple[FlopTerm, ...]
    training_flop_ledger_complete: bool
    training_flop_obligations: tuple[str, ...]


def _owned_parameters(
    arm: _BuiltArmLike,
) -> tuple[tuple[str, _ParameterLike], ...]:
    records = tuple(
        (f"model.{name}", parameter)
        for name, parameter in arm.model.named_parameters(
            remove_duplicate=False
        )
    )
    if arm.recognition_store is not None:
        records += tuple(
            (f"recognition_store.{name}", parameter)
            for name, parameter in arm.recognition_store.named_parameters(
                remove_duplicate=False
            )
        )
    return records


def audit_parameter_ownership(arm: _BuiltArmLike) -> None:
    """Reject every undeclared, duplicate, frozen, dormant, or no-op parameter."""

    if type(arm.config) is not ArmConfig:
        raise ValueError("arm config must be an ArmConfig")
    owned = _owned_parameters(arm)
    owned_ids = [id(parameter) for _, parameter in owned]
    if len(owned_ids) != len(set(owned_ids)):
        raise ValueError("a parameter object is owned by more than one store")
    frozen = tuple(
        name for name, parameter in owned if parameter.requires_grad is not True
    )
    if frozen:
        raise ValueError(
            f"frozen filler or dormant parameters are forbidden: {frozen!r}"
        )

    roles = tuple(arm.parameter_roles)
    if any(type(record) is not ParameterRoleRecord for record in roles):
        raise ValueError("parameter roles must be exact ParameterRoleRecord values")
    role_ids = [record.parameter_id for record in roles]
    if len(role_ids) != len(set(role_ids)):
        raise ValueError("a parameter has more than one declared role")
    active_by_id = {
        id(parameter): (name, parameter) for name, parameter in owned
    }
    missing_roles = set(active_by_id) - set(role_ids)
    unknown_roles = set(role_ids) - set(active_by_id)
    if missing_roles:
        raise ValueError(f"dormant or unbound active parameters: {sorted(missing_roles)!r}")
    if unknown_roles:
        raise ValueError(f"declared roles reference unknown parameters: {sorted(unknown_roles)!r}")
    for record in roles:
        observed_name, parameter = active_by_id[record.parameter_id]
        if record.qualified_name != observed_name:
            raise ValueError(
                "parameter role qualified name does not match its owner"
            )
        if record.scalar_count != parameter.numel():
            raise ValueError("parameter role scalar count does not match")

    bindings = tuple(arm.optimizer_bindings)
    if any(type(binding) is not OptimizerBinding for binding in bindings):
        raise ValueError("optimizer bindings must be exact OptimizerBinding values")
    binding_ids = [
        parameter_id
        for binding in bindings
        for parameter_id in binding.parameter_ids
    ]
    if len(binding_ids) != len(set(binding_ids)):
        raise ValueError("a parameter is bound to more than one optimizer")
    missing_bindings = set(role_ids) - set(binding_ids)
    unknown_bindings = set(binding_ids) - set(role_ids)
    if missing_bindings:
        raise ValueError(f"unbound active parameters: {sorted(missing_bindings)!r}")
    if unknown_bindings:
        raise ValueError(
            f"optimizer bindings reference unknown parameters: {sorted(unknown_bindings)!r}"
        )
    binding_phase_by_id = {
        parameter_id: binding.phase
        for binding in bindings
        for parameter_id in binding.parameter_ids
    }
    for record in roles:
        if binding_phase_by_id[record.parameter_id] != record.phase:
            raise ValueError("parameter role phase does not match optimizer phase")
    if any(
        binding.optimizer_policy_sha256
        != H6_ADAMW_POLICY.optimizer_policy_sha256
        for binding in bindings
    ):
        raise ValueError("optimizer binding does not use the frozen AdamW policy")

    expected_phases = (
        {
            TrainingPhase.RECOGNITION_ADAMW.value,
            TrainingPhase.MODEL_ADAMW.value,
        }
        if arm.config.latent_enabled
        else {TrainingPhase.MODEL_CE_ADAMW.value}
    )
    observed_phases = {binding.phase for binding in bindings}
    if observed_phases != expected_phases:
        raise ValueError("missing, extra, or no-op optimizer phase")
    if arm.config.latent_enabled != (arm.recognition_store is not None):
        raise ValueError("recognition parameter store applicability is inconsistent")


def _capacity_differences(
    endpoint: CapacityAllocation,
    reference: CapacityAllocation,
) -> tuple[str, ...]:
    return tuple(
        name
        for name in (
            "emission_width",
            "latent_width",
            "recognition_width",
        )
        if getattr(endpoint, name) != getattr(reference, name)
    )


def _semantic_differences(
    endpoint: ArmConfig, reference: ArmConfig
) -> tuple[str, ...]:
    endpoint_payload = endpoint.semantic_payload()
    reference_payload = reference.semantic_payload()
    return tuple(
        name
        for name in endpoint_payload
        if endpoint_payload[name] != reference_payload[name]
    )


def _training_flop_review(
    arm: _BuiltArmLike, *, endpoint_name: str
) -> tuple[bool, tuple[str, ...]]:
    declared_complete = (
        getattr(arm, "training_flop_ledger_complete", None) is True
    )
    raw_obligations = getattr(arm, "training_flop_obligations", ())
    obligations: list[str] = []
    if (
        type(raw_obligations) is not tuple
        or any(type(item) is not str or not item for item in raw_obligations)
    ):
        obligations.append(
            f"{endpoint_name}: training FLOP obligations are malformed"
        )
    else:
        obligations.extend(
            f"{endpoint_name}: {item}" for item in raw_obligations
        )
    if not declared_complete:
        obligations.extend(
            (
                f"{endpoint_name}: missing operator-complete forward, backward, "
                "L2 clip/scale, and AdamW arithmetic terms",
                f"{endpoint_name}: missing full batches-times-passes repetition proof",
            )
        )
    if any(
        term.operation.startswith("INCOMPLETE_")
        for term in arm.flop_terms
    ):
        obligations.append(
            f"{endpoint_name}: lower-bound one-step terms cannot certify "
            "whole-schedule training FLOPs"
        )
    if arm.config.map_mode == "shared_vertex_coboundary":
        operations = {term.operation for term in arm.flop_terms}
        for operation in ("matrix_exp", "matrix_inverse_or_solve"):
            if operation not in operations:
                obligations.append(
                    f"{endpoint_name}: {operation} arithmetic is uncounted"
                )
    return declared_complete and not obligations, tuple(dict.fromkeys(obligations))


def _common_schedule_is_proven(
    endpoint: _BuiltArmLike,
    reference: _BuiltArmLike,
    *,
    expected_schedule_sha256: str,
) -> bool:
    endpoint_hash = getattr(endpoint, "training_schedule_policy_sha256", None)
    reference_hash = getattr(reference, "training_schedule_policy_sha256", None)
    endpoint_batches = getattr(endpoint, "training_batches_per_pass", None)
    reference_batches = getattr(reference, "training_batches_per_pass", None)
    return (
        endpoint_hash == reference_hash == expected_schedule_sha256
        and type(endpoint_batches) is int
        and endpoint_batches > 0
        and endpoint_batches == reference_batches
    )


def audit_arm_matching(
    endpoint: _BuiltArmLike,
    reference: _BuiltArmLike,
    *,
    matching_config: H6ArmMatchingResolvedConfig,
    named_factor: str,
    nuisance_capacity_fields: tuple[str, ...],
) -> MatchingReport:
    """Construct a fail-closed hard-tolerance report from declared ledgers."""

    resolved_matching = _require_matching_config(matching_config)
    configured_hashes = {
        item.config_sha256 for item in resolved_matching.arm_configs
    }
    if endpoint.config.config_sha256 not in configured_hashes:
        raise ValueError(
            "endpoint config is not bound by the resolved matching config"
        )
    configured_reference = resolved_matching.arm_configs[5]
    if (
        reference.config.config_sha256
        != configured_reference.config_sha256
        or reference.config.capacity_allocation
        != resolved_matching.reference_allocation
    ):
        raise ValueError(
            "reference arm does not equal the resolved canonical A5 reference"
        )
    ownership_valid = True
    try:
        audit_parameter_ownership(endpoint)
        audit_parameter_ownership(reference)
    except ValueError:
        ownership_valid = False

    declared_nuisance = tuple(nuisance_capacity_fields)
    actual_nuisance = _capacity_differences(
        endpoint.config.capacity_allocation,
        reference.config.capacity_allocation,
    )
    report_nuisance = declared_nuisance
    if declared_nuisance != actual_nuisance:
        report_nuisance += ("undeclared_capacity_difference",)

    endpoint_policy_hashes = {
        binding.optimizer_policy_sha256
        for binding in endpoint.optimizer_bindings
    }
    reference_policy_hashes = {
        binding.optimizer_policy_sha256
        for binding in reference.optimizer_bindings
    }
    optimizer_policy_match = (
        endpoint_policy_hashes
        == reference_policy_hashes
        == {
            resolved_matching.adamw_policy.optimizer_policy_sha256
        }
    )
    endpoint_flops_complete, endpoint_flop_obligations = (
        _training_flop_review(endpoint, endpoint_name="endpoint")
    )
    reference_flops_complete, reference_flop_obligations = (
        _training_flop_review(reference, endpoint_name="reference")
    )
    training_flop_ledger_complete = (
        endpoint_flops_complete and reference_flops_complete
    )

    return MatchingReport.from_totals(
        matching_config_sha256=resolved_matching.config_sha256,
        endpoint_config_sha256=endpoint.config.config_sha256,
        reference_config_sha256=reference.config.config_sha256,
        endpoint_parameter_count=sum(
            record.scalar_count for record in endpoint.parameter_roles
        ),
        reference_parameter_count=sum(
            record.scalar_count for record in reference.parameter_roles
        ),
        endpoint_training_flops=sum(
            term.total_arithmetic_flops for term in endpoint.flop_terms
        ),
        reference_training_flops=sum(
            term.total_arithmetic_flops for term in reference.flop_terms
        ),
        parameter_relative_tolerance=(
            resolved_matching.parameter_relative_tolerance
        ),
        flop_relative_tolerance=resolved_matching.flop_relative_tolerance,
        ownership_valid=ownership_valid,
        common_schedule=_common_schedule_is_proven(
            endpoint,
            reference,
            expected_schedule_sha256=(
                resolved_matching.matching_schedule_sha256
            ),
        ),
        optimizer_policy_match=optimizer_policy_match,
        training_flop_ledger_complete=training_flop_ledger_complete,
        training_flop_obligations=(
            endpoint_flop_obligations + reference_flop_obligations
        ),
        semantic_interventions=_semantic_differences(
            endpoint.config, reference.config
        ),
        named_factor=named_factor,
        nuisance_capacity_fields=report_nuisance,
        common_schedule_sha256=(
            resolved_matching.matching_schedule_sha256
        ),
    )


__all__ = [
    "A5_REFERENCE_ALLOCATION",
    "ARM_MATRIX_ROWS",
    "ARM_MATRIX_SHA256",
    "EMISSION_WIDTH_CANDIDATES",
    "H6_ADAMW_POLICY",
    "LATENT_WIDTH_CANDIDATES",
    "MATCHING_SCHEDULE_POLICY",
    "RECOGNITION_WIDTH_CANDIDATES",
    "AdamWPolicyRecord",
    "ArmConfig",
    "ArmMatrixRow",
    "CapacityAllocation",
    "FlopTerm",
    "MatchingReport",
    "OptimizerBinding",
    "ParameterRoleRecord",
    "adamw_flops",
    "arm_matrix_sha256",
    "audit_arm_matching",
    "audit_parameter_ownership",
    "backward_flops",
    "candidate_allocations",
    "capacity_candidate_count",
    "dense_matmul_flops",
    "dense_matvec_flops",
    "immutable_snapshot_flop_term",
    "l2_clip_scale_flops",
    "log_softmax_flops",
    "scalar_flops",
]
