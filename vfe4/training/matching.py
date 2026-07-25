"""Pure H6 arm capacity, ownership, and arithmetic matching.

Candidate enumeration is formula-only and lazy.  This module never opens a
corpus, evaluates a model, inspects gradients, or reads a predictive metric.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, Protocol

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

from .parameter_counts import (
    AMENDED_EMISSION_WIDTH_CANDIDATES,
    AMENDED_LATENT_WIDTH_CANDIDATES,
    AMENDED_RECOGNITION_WIDTH_CANDIDATES,
    PROPOSED_PREFIX_PRIOR_CONTEXT_WIDTH,
    arm_parameter_count,
    fixed_source_prior_parameter_count,
    prefix_conditioned_source_prior_parameter_count,
    recognition_parameter_count,
)


H6_ADAMW_POLICY = AdamWPolicyRecord.create()
A5_REFERENCE_ALLOCATION = CapacityAllocation.create(
    emission_width=64,
    latent_width=16,
    recognition_width=64,
)


def _hash(domain: bytes, payload: object) -> str:
    return hashlib.sha256(domain + b"\x00" + canonical_json_bytes(payload)).hexdigest()


def stable_parameter_key(*, qualified_name: str, phase: str) -> str:
    """Return the process-independent key used in ownership artifacts."""

    if type(qualified_name) is not str or not qualified_name:
        raise ValueError("qualified_name must be a nonempty string")
    try:
        parsed_phase = TrainingPhase(phase)
    except (TypeError, ValueError) as exc:
        raise ValueError("phase is not an H6 training phase") from exc
    if parsed_phase is TrainingPhase.IMMUTABLE_DETACHED_SNAPSHOT:
        raise ValueError("snapshot phase cannot own a parameter")
    return _hash(
        b"vfe4.h6.parameter-key.v1",
        {
            "qualified_name": qualified_name,
            "phase": phase,
        },
    )


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


@dataclass(frozen=True, slots=True)
class AmendedMatchingSchedulePolicy:
    """Outcome-blind v2 policy frozen before any predictive outcome."""

    schema_version: Literal["h6-amended-matching-policy-v2"]
    reference_allocation_sha256: str
    emission_width_candidates: tuple[int, ...]
    latent_width_candidates: tuple[int, ...]
    recognition_width_candidates: tuple[int, ...]
    prior_context_width: int
    parameter_relative_tolerance: Literal[0.01]
    flop_relative_tolerance: Literal[0.05]
    optimizer_policy_sha256: str
    sequence_length: Literal[32]
    window_stride: Literal[32]
    batch_size: Literal[8]
    drop_last: Literal[False]
    full_passes: Literal[2]
    model_updates_per_batch: Literal[1]
    validation_boundary_policy: Literal["twentieths_of_each_pass_v1"]
    checkpoint_boundary_policy: Literal["terminal_only_v1"]
    selection_rule: Literal["first_lexicographic_hard_eligible"]
    forbidden_inputs: tuple[str, ...]
    excluded_operations: tuple[str, ...]
    policy_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in tuple(self.__dataclass_fields__)[:-1]
        }

    def __post_init__(self) -> None:
        expected = {
            "schema_version": "h6-amended-matching-policy-v2",
            "reference_allocation_sha256": (
                A5_REFERENCE_ALLOCATION.allocation_sha256
            ),
            "emission_width_candidates": (
                AMENDED_EMISSION_WIDTH_CANDIDATES
            ),
            "latent_width_candidates": AMENDED_LATENT_WIDTH_CANDIDATES,
            "recognition_width_candidates": (
                AMENDED_RECOGNITION_WIDTH_CANDIDATES
            ),
            "prior_context_width": PROPOSED_PREFIX_PRIOR_CONTEXT_WIDTH,
            "parameter_relative_tolerance": 0.01,
            "flop_relative_tolerance": 0.05,
            "optimizer_policy_sha256": (
                H6_ADAMW_POLICY.optimizer_policy_sha256
            ),
            "sequence_length": 32,
            "window_stride": 32,
            "batch_size": 8,
            "drop_last": False,
            "full_passes": 2,
            "model_updates_per_batch": 1,
            "validation_boundary_policy": (
                "twentieths_of_each_pass_v1"
            ),
            "checkpoint_boundary_policy": "terminal_only_v1",
            "selection_rule": "first_lexicographic_hard_eligible",
            "forbidden_inputs": (
                "corpus_bytes",
                "loss",
                "gradients",
                "validation_metrics",
                "test_metrics",
                "prediction_flops",
            ),
            "excluded_operations": (
                "data_io",
                "validation",
                "checkpoint_serialization",
                "test_scoring",
                "prediction_particle_propagation",
                "prediction_cache",
            ),
        }
        if self.canonical_payload() != expected:
            raise ValueError("amended matching policy is not the frozen v2 policy")
        if self.policy_sha256 != _hash(
            b"vfe4.h6.amended-matching-policy.v2", expected
        ):
            raise ValueError("amended matching policy digest does not match")


_AMENDED_POLICY_PAYLOAD = {
    "schema_version": "h6-amended-matching-policy-v2",
    "reference_allocation_sha256": (
        A5_REFERENCE_ALLOCATION.allocation_sha256
    ),
    "emission_width_candidates": AMENDED_EMISSION_WIDTH_CANDIDATES,
    "latent_width_candidates": AMENDED_LATENT_WIDTH_CANDIDATES,
    "recognition_width_candidates": (
        AMENDED_RECOGNITION_WIDTH_CANDIDATES
    ),
    "prior_context_width": PROPOSED_PREFIX_PRIOR_CONTEXT_WIDTH,
    "parameter_relative_tolerance": 0.01,
    "flop_relative_tolerance": 0.05,
    "optimizer_policy_sha256": H6_ADAMW_POLICY.optimizer_policy_sha256,
    "sequence_length": 32,
    "window_stride": 32,
    "batch_size": 8,
    "drop_last": False,
    "full_passes": 2,
    "model_updates_per_batch": 1,
    "validation_boundary_policy": "twentieths_of_each_pass_v1",
    "checkpoint_boundary_policy": "terminal_only_v1",
    "selection_rule": "first_lexicographic_hard_eligible",
    "forbidden_inputs": (
        "corpus_bytes",
        "loss",
        "gradients",
        "validation_metrics",
        "test_metrics",
        "prediction_flops",
    ),
    "excluded_operations": (
        "data_io",
        "validation",
        "checkpoint_serialization",
        "test_scoring",
        "prediction_particle_propagation",
        "prediction_cache",
    ),
}
AMENDED_MATCHING_SCHEDULE_POLICY = AmendedMatchingSchedulePolicy(
    **_AMENDED_POLICY_PAYLOAD,  # type: ignore[arg-type]
    policy_sha256=_hash(
        b"vfe4.h6.amended-matching-policy.v2",
        _AMENDED_POLICY_PAYLOAD,
    ),
)


@dataclass(frozen=True, slots=True)
class H6TrainingWorkload:
    """Data-identity-bound multiplier for every active training phase."""

    schema_version: Literal["h6-training-workload-v1"]
    train_token_count: int
    train_token_sha256: str
    sequence_length: Literal[32]
    window_stride: Literal[32]
    batch_size: Literal[8]
    drop_last: Literal[False]
    full_passes: Literal[2]
    window_count: int
    full_batches_per_pass: int
    tail_batch_size: int
    batches_per_pass: int
    model_update_opportunities: int
    validation_boundaries_per_pass: tuple[int, ...]
    matching_schedule_policy_sha256: str
    workload_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in tuple(self.__dataclass_fields__)[:-1]
        }

    def __post_init__(self) -> None:
        if (
            self.schema_version != "h6-training-workload-v1"
            or type(self.train_token_count) is not int
            or self.train_token_count < 2
            or self.sequence_length != 32
            or self.window_stride != 32
            or self.batch_size != 8
            or self.drop_last is not False
            or self.full_passes != 2
        ):
            raise ValueError("training workload does not match the frozen policy")
        if (
            type(self.train_token_sha256) is not str
            or len(self.train_token_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.train_token_sha256
            )
        ):
            raise ValueError("train_token_sha256 must be lowercase SHA-256")
        expected_windows = (
            self.train_token_count - 2
        ) // self.window_stride + 1
        expected_full_batches, expected_tail = divmod(
            expected_windows, self.batch_size
        )
        expected_batches = expected_full_batches + bool(expected_tail)
        expected_boundaries = tuple(
            dict.fromkeys(
                (
                    k * expected_batches + 19
                )
                // 20
                for k in range(1, 21)
            )
        )
        if (
            self.window_count != expected_windows
            or self.full_batches_per_pass != expected_full_batches
            or self.tail_batch_size != expected_tail
            or self.batches_per_pass != expected_batches
            or self.model_update_opportunities
            != expected_batches * self.full_passes
            or self.validation_boundaries_per_pass
            != expected_boundaries
            or self.matching_schedule_policy_sha256
            != AMENDED_MATCHING_SCHEDULE_POLICY.policy_sha256
        ):
            raise ValueError("training workload counts are inconsistent")
        if self.workload_sha256 != _hash(
            b"vfe4.h6.training-workload.v1",
            self.canonical_payload(),
        ):
            raise ValueError("training workload digest does not match")

    @classmethod
    def from_train_tokens(
        cls,
        *,
        train_token_count: int,
        train_token_sha256: str,
    ) -> "H6TrainingWorkload":
        if type(train_token_count) is not int or train_token_count < 2:
            raise ValueError("train_token_count must be at least two")
        window_count = (train_token_count - 2) // 32 + 1
        full_batches, tail_batch_size = divmod(window_count, 8)
        batches_per_pass = full_batches + bool(tail_batch_size)
        values: dict[str, object] = {
            "schema_version": "h6-training-workload-v1",
            "train_token_count": train_token_count,
            "train_token_sha256": train_token_sha256,
            "sequence_length": 32,
            "window_stride": 32,
            "batch_size": 8,
            "drop_last": False,
            "full_passes": 2,
            "window_count": window_count,
            "full_batches_per_pass": full_batches,
            "tail_batch_size": tail_batch_size,
            "batches_per_pass": batches_per_pass,
            "model_update_opportunities": 2 * batches_per_pass,
            "validation_boundaries_per_pass": tuple(
                dict.fromkeys(
                    (k * batches_per_pass + 19) // 20
                    for k in range(1, 21)
                )
            ),
            "matching_schedule_policy_sha256": (
                AMENDED_MATCHING_SCHEDULE_POLICY.policy_sha256
            ),
        }
        return cls(
            **values,  # type: ignore[arg-type]
            workload_sha256=_hash(
                b"vfe4.h6.training-workload.v1", values
            ),
        )


@dataclass(frozen=True, slots=True)
class EndpointFormulaProfile:
    config_id: str
    arm: Literal["A0", "A1", "A2", "A3", "A4", "A5"]
    latent_enabled: bool
    channel_count: Literal[0, 1, 2]
    source_mode: Literal[
        "absent", "immediate_predecessor", "categorical"
    ]
    map_mode: Literal[
        "absent",
        "generic_fixed_frame_non_coboundary",
        "shared_vertex_coboundary",
    ]
    recognition_family: Literal["absent", "structured", "factorized"]
    recognition_conditioning: Literal[
        "absent", "filtering", "smoothing"
    ]
    prior_variant: Literal["absent", "fixed", "prefix_conditioned"]
    mixture_mode: Literal["absent", "exact", "moment_projection"]
    objective_kind: Literal[
        "cross_entropy",
        "complete_elbo",
        "emission_only_ablation_non_elbo",
    ]
    profile_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in tuple(self.__dataclass_fields__)[:-1]
        }

    def __post_init__(self) -> None:
        if type(self.config_id) is not str or not self.config_id:
            raise ValueError("config_id must be nonempty")
        if self.latent_enabled != (self.channel_count in (1, 2)):
            raise ValueError("channel count does not match latent applicability")
        if self.latent_enabled != (
            self.recognition_family != "absent"
            and self.recognition_conditioning != "absent"
        ):
            raise ValueError("recognition profile does not match latent applicability")
        if (self.source_mode == "categorical") != (
            self.prior_variant in ("fixed", "prefix_conditioned")
            and self.mixture_mode in ("exact", "moment_projection")
        ):
            raise ValueError("categorical source profile is incomplete")
        if self.profile_sha256 != _hash(
            b"vfe4.h6.endpoint-formula-profile.v1",
            self.canonical_payload(),
        ):
            raise ValueError("endpoint formula profile digest does not match")


_ENDPOINT_PROFILE_FIELDS: dict[str, tuple[object, ...]] = {
    "h6-a0-ar-v1": (
        "A0", False, 0, "absent", "absent", "absent", "absent",
        "absent", "absent", "cross_entropy",
    ),
    "h6-a1-ordinary-latent-v1": (
        "A1", True, 1, "absent", "absent", "structured", "smoothing",
        "absent", "absent", "complete_elbo",
    ),
    "h6-a2-generic-map-v1": (
        "A2", True, 2, "categorical",
        "generic_fixed_frame_non_coboundary", "structured", "smoothing",
        "fixed", "exact", "complete_elbo",
    ),
    "h6-a3-immediate-predecessor-v1": (
        "A3", True, 2, "immediate_predecessor",
        "shared_vertex_coboundary", "structured", "smoothing", "absent",
        "absent", "complete_elbo",
    ),
    "h6-a4-state-only-v1": (
        "A4", True, 1, "categorical", "shared_vertex_coboundary",
        "structured", "smoothing", "fixed", "exact", "complete_elbo",
    ),
    "h6-a5-structured-fixed-exact-complete-latent-smoothing-v1": (
        "A5", True, 2, "categorical", "shared_vertex_coboundary",
        "structured", "smoothing", "fixed", "exact", "complete_elbo",
    ),
    "h6-a5-factorized-fixed-exact-complete-latent-smoothing-v1": (
        "A5", True, 2, "categorical", "shared_vertex_coboundary",
        "factorized", "smoothing", "fixed", "exact", "complete_elbo",
    ),
    "h6-a5-structured-prefix-exact-complete-latent-smoothing-v1": (
        "A5", True, 2, "categorical", "shared_vertex_coboundary",
        "structured", "smoothing", "prefix_conditioned", "exact",
        "complete_elbo",
    ),
    "h6-a5-structured-fixed-projection-complete-latent-smoothing-v1": (
        "A5", True, 2, "categorical", "shared_vertex_coboundary",
        "structured", "smoothing", "fixed", "moment_projection",
        "complete_elbo",
    ),
    "h6-a5-structured-fixed-exact-emission-latent-smoothing-v1": (
        "A5", True, 2, "categorical", "shared_vertex_coboundary",
        "structured", "smoothing", "fixed", "exact",
        "emission_only_ablation_non_elbo",
    ),
    (
        "h6-a5-structured-fixed-exact-complete-"
        "nolatent-norecognition-v1"
    ): (
        "A5", False, 0, "absent", "absent", "absent", "absent",
        "absent", "absent", "complete_elbo",
    ),
    "h6-a5-structured-fixed-exact-complete-latent-filtering-v1": (
        "A5", True, 2, "categorical", "shared_vertex_coboundary",
        "structured", "filtering", "fixed", "exact", "complete_elbo",
    ),
}


def endpoint_formula_profile(config_id: str) -> EndpointFormulaProfile:
    """Return one of the twelve closed formula profiles."""

    values = _ENDPOINT_PROFILE_FIELDS.get(config_id)
    if values is None:
        raise ValueError("config_id is not one of the twelve H6 endpoints")
    field_names = tuple(EndpointFormulaProfile.__dataclass_fields__)[1:-1]
    payload = {
        "config_id": config_id,
        **dict(zip(field_names, values, strict=True)),
    }
    return EndpointFormulaProfile(
        **payload,  # type: ignore[arg-type]
        profile_sha256=_hash(
            b"vfe4.h6.endpoint-formula-profile.v1", payload
        ),
    )


def _profile_for_config(config: ArmConfig) -> EndpointFormulaProfile:
    if type(config) is not ArmConfig:
        raise ValueError("endpoint_config must be an exact ArmConfig")
    config.__post_init__()
    profile = endpoint_formula_profile(config.config_id)
    if (
        profile.arm != config.arm.value
        or profile.latent_enabled != config.latent_enabled
        or profile.channel_count
        != int(config.state_channel_enabled)
        + int(config.model_channel_enabled)
        or profile.source_mode != config.source_mode
        or profile.map_mode != config.map_mode
        or profile.recognition_family != config.recognition_family
        or profile.recognition_conditioning
        != config.recognition_conditioning
        or profile.prior_variant != config.prior_variant
        or profile.mixture_mode != config.mixture_mode
        or profile.objective_kind != config.objective_kind
    ):
        raise ValueError(
            "endpoint config and analytical formula profile disagree"
        )
    return profile


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


def matrix_solve_lu_flops(dimension: int) -> int:
    """Count one dense LU solve with a dense ``d x d`` right-hand side."""

    d = _positive_dimension(dimension, "dimension")
    factorization = 2 * d * (d - 1) * (d + 1) // 3
    forward_substitution = d * d * (d - 1)
    backward_substitution = d**3
    return factorization + forward_substitution + backward_substitution


def matrix_exp_pade13_flops(dimension: int) -> int:
    """Count one live matrix-exponential call by the frozen Padé-13 convention.

    This is the preregistered analytical convention used for every
    ``torch.matrix_exp`` invocation in the live endpoint.  Call
    multiplicities are derived from the uncached per-edge implementation;
    this is not a claim about backend instruction counts.
    """

    d = _positive_dimension(dimension, "dimension")
    return (
        12 * d**3
        + 30 * d**2
        + 1
        + matrix_solve_lu_flops(d)
    )


@dataclass(frozen=True, slots=True)
class H6AnalyticalFlopLedger:
    schema_version: Literal["h6-analytical-flop-ledger-v1"]
    endpoint_config_sha256: str
    endpoint_profile_sha256: str
    allocation_sha256: str
    workload_sha256: str
    terms: tuple[FlopTerm, ...]
    total_arithmetic_flops: int
    total_bytes_copied: int
    status: Literal["COMPLETE", "INCONCLUSIVE"]
    obligations: tuple[str, ...]
    ledger_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "endpoint_config_sha256": self.endpoint_config_sha256,
            "endpoint_profile_sha256": self.endpoint_profile_sha256,
            "allocation_sha256": self.allocation_sha256,
            "workload_sha256": self.workload_sha256,
            "term_sha256s": tuple(term.term_sha256 for term in self.terms),
            "total_arithmetic_flops": self.total_arithmetic_flops,
            "total_bytes_copied": self.total_bytes_copied,
            "status": self.status,
            "obligations": self.obligations,
        }

    def __post_init__(self) -> None:
        if self.schema_version != "h6-analytical-flop-ledger-v1":
            raise ValueError("unsupported analytical FLOP ledger")
        for name in (
            "endpoint_config_sha256",
            "endpoint_profile_sha256",
            "allocation_sha256",
            "workload_sha256",
        ):
            value = getattr(self, name)
            if (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be lowercase SHA-256")
        if (
            type(self.terms) is not tuple
            or not self.terms
            or any(type(term) is not FlopTerm for term in self.terms)
        ):
            raise ValueError("analytical FLOP ledger requires exact terms")
        for term in self.terms:
            term.__post_init__()
            if term.operation.startswith("INCOMPLETE_"):
                raise ValueError("analytical ledger cannot retain lower-bound terms")
        if self.total_arithmetic_flops != sum(
            term.total_arithmetic_flops for term in self.terms
        ):
            raise ValueError("analytical FLOP total is inconsistent")
        if self.total_bytes_copied != sum(
            term.total_bytes_copied for term in self.terms
        ):
            raise ValueError("analytical copy total is inconsistent")
        if self.status == "COMPLETE":
            if self.obligations:
                raise ValueError("complete ledger cannot retain obligations")
        elif self.status == "INCONCLUSIVE":
            if not self.obligations:
                raise ValueError("inconclusive ledger requires obligations")
        else:
            raise ValueError("unsupported analytical ledger status")
        if self.ledger_sha256 != _hash(
            b"vfe4.h6.analytical-flop-ledger.v1",
            self.canonical_payload(),
        ):
            raise ValueError("analytical FLOP ledger digest does not match")

    @classmethod
    def create(
        cls,
        *,
        endpoint_config_sha256: str,
        profile: EndpointFormulaProfile,
        allocation: CapacityAllocation,
        workload: H6TrainingWorkload,
        terms: tuple[FlopTerm, ...],
        obligations: tuple[str, ...] = (),
    ) -> "H6AnalyticalFlopLedger":
        status: Literal["COMPLETE", "INCONCLUSIVE"] = (
            "COMPLETE" if not obligations else "INCONCLUSIVE"
        )
        values: dict[str, object] = {
            "schema_version": "h6-analytical-flop-ledger-v1",
            "endpoint_config_sha256": endpoint_config_sha256,
            "endpoint_profile_sha256": profile.profile_sha256,
            "allocation_sha256": allocation.allocation_sha256,
            "workload_sha256": workload.workload_sha256,
            "terms": tuple(terms),
            "total_arithmetic_flops": sum(
                term.total_arithmetic_flops for term in terms
            ),
            "total_bytes_copied": sum(
                term.total_bytes_copied for term in terms
            ),
            "status": status,
            "obligations": tuple(obligations),
        }
        payload = {
            "schema_version": values["schema_version"],
            "endpoint_config_sha256": values["endpoint_config_sha256"],
            "endpoint_profile_sha256": values["endpoint_profile_sha256"],
            "allocation_sha256": values["allocation_sha256"],
            "workload_sha256": values["workload_sha256"],
            "term_sha256s": tuple(term.term_sha256 for term in terms),
            "total_arithmetic_flops": values["total_arithmetic_flops"],
            "total_bytes_copied": values["total_bytes_copied"],
            "status": values["status"],
            "obligations": values["obligations"],
        }
        return cls(
            **values,  # type: ignore[arg-type]
            ledger_sha256=_hash(
                b"vfe4.h6.analytical-flop-ledger.v1", payload
            ),
        )


def _parameter_components(
    profile: EndpointFormulaProfile,
    allocation: CapacityAllocation,
    *,
    vocabulary_size: int,
    horizon: int,
) -> tuple[int, int]:
    if not profile.latent_enabled:
        total = arm_parameter_count(
            profile.arm,
            vocabulary_size=vocabulary_size,
            horizon=horizon,
            emission_width=allocation.emission_width,
        )
        return total, 0
    if (
        allocation.latent_width is None
        or allocation.recognition_width is None
    ):
        raise ValueError("latent formula profile requires latent/recognition widths")
    recognition = recognition_parameter_count(
        vocabulary_size=vocabulary_size,
        latent_width=allocation.latent_width,
        recognition_width=allocation.recognition_width,
        channel_count=profile.channel_count,  # type: ignore[arg-type]
        family=profile.recognition_family,  # type: ignore[arg-type]
    )
    total = arm_parameter_count(
        profile.arm,
        vocabulary_size=vocabulary_size,
        horizon=horizon,
        emission_width=allocation.emission_width,
        latent_width=allocation.latent_width,
        recognition_width=allocation.recognition_width,
        recognition_family=profile.recognition_family,  # type: ignore[arg-type]
    )
    if profile.prior_variant == "prefix_conditioned":
        if allocation.prior_context_width is None:
            raise ValueError("prefix prior requires prior_context_width")
        total = (
            total
            - fixed_source_prior_parameter_count(
                horizon=horizon, bank_count=2
            )
            + prefix_conditioned_source_prior_parameter_count(
                vocabulary_size=vocabulary_size,
                horizon=horizon,
                latent_width=allocation.latent_width,
                context_width=allocation.prior_context_width,
                gauge_anchored=True,
            )
        )
    return total - recognition, recognition


def _recognition_costs(
    profile: EndpointFormulaProfile,
    *,
    horizon: int,
    latent_width: int,
    recognition_width: int,
) -> dict[str, int]:
    gaussian_dimension = profile.channel_count * latent_width
    context = (
        horizon * recognition_width
        if profile.recognition_conditioning == "smoothing"
        else (3 * horizon - 1) * recognition_width
    )
    return {
        "recognition_context_reduction": context,
        "recognition_mean_dense_matvec_bias": (
            2 * gaussian_dimension * recognition_width
            + gaussian_dimension
        ),
        "recognition_precision_cholesky": 4 * gaussian_dimension,
    }


def _ar_objective_costs(
    *, vocabulary_size: int, horizon: int, emission_width: int
) -> dict[str, int]:
    return {
        "autoregressive_prefix_context_reduction": (
            emission_width * horizon * (horizon - 1) // 2
        ),
        "autoregressive_emission_dense_matvec": (
            horizon * 2 * vocabulary_size * emission_width
        ),
        "autoregressive_emission_bias": horizon * vocabulary_size,
        "autoregressive_log_softmax": (
            horizon * (5 * vocabulary_size - 1)
        ),
        "autoregressive_target_nll": horizon,
    }


def _latent_objective_costs(
    profile: EndpointFormulaProfile,
    *,
    vocabulary_size: int,
    horizon: int,
    emission_width: int,
    latent_width: int,
    prior_context_width: int | None,
) -> dict[str, int]:
    channels = profile.channel_count
    emission = horizon * (
        2 * emission_width * latent_width * channels
        + max(0, channels - 1) * emission_width
        + emission_width
        + 2 * vocabulary_size * emission_width
        + vocabulary_size
        + (5 * vocabulary_size - 1)
        + 1
    )
    costs: dict[str, int] = {
        "normalized_emission": emission,
    }
    if profile.objective_kind == "emission_only_ablation_non_elbo":
        return costs
    costs["initial_diagonal_gaussian"] = 9 * channels * latent_width
    if profile.source_mode == "immediate_predecessor":
        edge_count = horizon
        nonanchor_source_edge_count = max(0, horizon - 1)
    elif profile.source_mode == "categorical":
        edge_count = horizon * (horizon + 1) // 2
        nonanchor_source_edge_count = horizon * (horizon - 1) // 2
    else:
        edge_count = horizon
        nonanchor_source_edge_count = 0
    if profile.map_mode == "shared_vertex_coboundary":
        costs["shared_coboundary_receiver_matrix_exp_pade13"] = (
            channels
            * edge_count
            * matrix_exp_pade13_flops(latent_width)
        )
        costs["shared_coboundary_source_matrix_exp_pade13"] = (
            channels
            * nonanchor_source_edge_count
            * matrix_exp_pade13_flops(latent_width)
        )
        costs["shared_coboundary_source_inverse_lu"] = (
            channels * edge_count * matrix_solve_lu_flops(latent_width)
        )
        costs["shared_coboundary_frame_product_dense_matmul"] = (
            channels
            * edge_count
            * dense_matmul_flops(
                latent_width,
                latent_width,
                latent_width,
            )
        )
    costs["transition_dense_matvec_bias"] = (
        channels * edge_count * (2 * latent_width**2 + latent_width)
    )
    if profile.arm in ("A2", "A5"):
        costs["same_receiver_b_dense_matvec_add"] = edge_count * (
            2 * latent_width**2 + latent_width
        )
    costs["transition_diagonal_gaussian"] = (
        channels * edge_count * 9 * latent_width
    )
    if profile.source_mode == "categorical":
        banks = channels
        if profile.prior_variant == "fixed":
            costs["fixed_source_log_softmax"] = banks * (
                5 * horizon * (horizon + 1) // 2 - horizon
            )
        else:
            if prior_context_width is None:
                raise ValueError("prefix prior formula requires context width")
            prefix_cost = 0
            for receiver_t in range(1, horizon + 1):
                prefix_cost += (
                    (receiver_t - 1) * prior_context_width
                    + 2
                    * receiver_t
                    * prior_context_width
                    * latent_width
                    + receiver_t * prior_context_width
                    + prior_context_width
                    + 2
                    * (receiver_t - 1)
                    * prior_context_width
                    + (receiver_t - 1)
                    + (5 * receiver_t - 1)
                )
            costs["prefix_conditioned_source_prior"] = banks * prefix_cost
        if profile.mixture_mode == "exact":
            costs["exact_source_mixture_reduction"] = banks * sum(
                5 * receiver_t
                for receiver_t in range(1, horizon + 1)
            )
        else:
            costs["moment_projection_source_mixture"] = banks * sum(
                (6 * receiver_t - 2) * latent_width
                for receiver_t in range(1, horizon + 1)
            )
    gaussian_dimension = channels * latent_width
    costs["recognition_entropy"] = 2 * gaussian_dimension + 1
    return costs


def _batch_shapes(
    workload: H6TrainingWorkload,
) -> tuple[tuple[str, int, int], ...]:
    shapes: list[tuple[str, int, int]] = []
    full_repetitions = (
        workload.full_batches_per_pass * workload.full_passes
    )
    if full_repetitions:
        shapes.append(("full_batch", workload.batch_size, full_repetitions))
    if workload.tail_batch_size:
        shapes.append(
            (
                "tail_batch",
                workload.tail_batch_size,
                workload.full_passes,
            )
        )
    return tuple(shapes)


def _phase_terms(
    *,
    phase: TrainingPhase,
    costs_per_window: dict[str, int],
    parameter_count: int,
    workload: H6TrainingWorkload,
) -> list[FlopTerm]:
    terms: list[FlopTerm] = []
    for batch_label, batch_size, repetitions in _batch_shapes(workload):
        for operation, cost in costs_per_window.items():
            forward = batch_size * cost
            terms.append(
                FlopTerm.create(
                    phase=phase.value,
                    operation=f"forward::{operation}::{batch_label}",
                    repetitions=repetitions,
                    arithmetic_flops_per_repetition=forward,
                    bytes_copied_per_repetition=0,
                )
            )
            terms.append(
                FlopTerm.create(
                    phase=phase.value,
                    operation=f"backward::{operation}::{batch_label}",
                    repetitions=repetitions,
                    arithmetic_flops_per_repetition=backward_flops(forward),
                    bytes_copied_per_repetition=0,
                )
            )
        terms.append(
            FlopTerm.create(
                phase=phase.value,
                operation=f"l2_global_clip_scale::{batch_label}",
                repetitions=repetitions,
                arithmetic_flops_per_repetition=l2_clip_scale_flops(
                    parameter_count
                ),
                bytes_copied_per_repetition=0,
            )
        )
        terms.append(
            FlopTerm.create(
                phase=phase.value,
                operation=f"adamw_update::{batch_label}",
                repetitions=repetitions,
                arithmetic_flops_per_repetition=adamw_flops(
                    parameter_count
                ),
                bytes_copied_per_repetition=0,
            )
        )
    return terms


def analytical_training_flop_ledger(
    *,
    endpoint_config: ArmConfig,
    workload: H6TrainingWorkload,
) -> H6AnalyticalFlopLedger:
    """Build the complete operator and whole-workload training ledger."""

    profile = _profile_for_config(endpoint_config)
    allocation = endpoint_config.capacity_allocation
    vocabulary_size = endpoint_config.vocabulary.size
    horizon = endpoint_config.horizon
    allocation.__post_init__()
    workload.__post_init__()
    model_parameters, recognition_parameters = _parameter_components(
        profile,
        allocation,
        vocabulary_size=vocabulary_size,
        horizon=horizon,
    )
    terms: list[FlopTerm] = []
    if not profile.latent_enabled:
        objective_costs = _ar_objective_costs(
            vocabulary_size=vocabulary_size,
            horizon=horizon,
            emission_width=allocation.emission_width,
        )
        terms.extend(
            _phase_terms(
                phase=TrainingPhase.MODEL_CE_ADAMW,
                costs_per_window=objective_costs,
                parameter_count=model_parameters,
                workload=workload,
            )
        )
    else:
        latent_width = allocation.latent_width
        recognition_width = allocation.recognition_width
        if latent_width is None or recognition_width is None:
            raise ValueError("latent ledger requires both live widths")
        recognition_costs = _recognition_costs(
            profile,
            horizon=horizon,
            latent_width=latent_width,
            recognition_width=recognition_width,
        )
        objective_costs = _latent_objective_costs(
            profile,
            vocabulary_size=vocabulary_size,
            horizon=horizon,
            emission_width=allocation.emission_width,
            latent_width=latent_width,
            prior_context_width=allocation.prior_context_width,
        )
        terms.extend(
            _phase_terms(
                phase=TrainingPhase.RECOGNITION_ADAMW,
                costs_per_window={**recognition_costs, **objective_costs},
                parameter_count=recognition_parameters,
                workload=workload,
            )
        )
        gaussian_dimension = profile.channel_count * latent_width
        snapshot_bytes_per_window = (
            gaussian_dimension + gaussian_dimension**2
        ) * 8
        for batch_label, batch_size, repetitions in _batch_shapes(workload):
            terms.append(
                FlopTerm.create(
                    phase=(
                        TrainingPhase.IMMUTABLE_DETACHED_SNAPSHOT.value
                    ),
                    operation=(
                        "immutable_detached_recognition_snapshot::"
                        f"{batch_label}"
                    ),
                    repetitions=repetitions,
                    arithmetic_flops_per_repetition=0,
                    bytes_copied_per_repetition=(
                        batch_size * snapshot_bytes_per_window
                    ),
                )
            )
        terms.extend(
            _phase_terms(
                phase=TrainingPhase.MODEL_ADAMW,
                costs_per_window=objective_costs,
                parameter_count=model_parameters,
                workload=workload,
            )
        )
    return H6AnalyticalFlopLedger.create(
        endpoint_config_sha256=endpoint_config.config_sha256,
        profile=profile,
        allocation=allocation,
        workload=workload,
        terms=tuple(terms),
    )


@dataclass(frozen=True, slots=True)
class H6FormulaSelection:
    config_id: str
    endpoint_template: ArmConfig
    reference_config: ArmConfig
    endpoint_profile_sha256: str
    reference_profile_sha256: str
    workload: H6TrainingWorkload
    policy_sha256: str
    candidate_count_evaluated: int
    selected_endpoint_config: ArmConfig | None
    selected_allocation: CapacityAllocation | None
    parameter_count: int | None
    training_flops: int | None
    parameter_relative_difference: float | None
    flop_relative_difference: float | None
    ledger: H6AnalyticalFlopLedger | None
    reference_ledger: H6AnalyticalFlopLedger
    status: Literal["ELIGIBLE", "INCONCLUSIVE"]
    obligations: tuple[str, ...]
    selection_sha256: str

    @property
    def endpoint_template_config_sha256(self) -> str:
        return self.endpoint_template.config_sha256

    @property
    def reference_config_sha256(self) -> str:
        return self.reference_config.config_sha256

    @property
    def workload_sha256(self) -> str:
        return self.workload.workload_sha256

    def canonical_payload(self) -> dict[str, object]:
        return {
            "config_id": self.config_id,
            "endpoint_template_config_sha256": (
                self.endpoint_template.config_sha256
            ),
            "reference_config_sha256": self.reference_config.config_sha256,
            "endpoint_profile_sha256": self.endpoint_profile_sha256,
            "reference_profile_sha256": self.reference_profile_sha256,
            "workload_sha256": self.workload.workload_sha256,
            "policy_sha256": self.policy_sha256,
            "candidate_count_evaluated": self.candidate_count_evaluated,
            "selected_endpoint_config_sha256": (
                None
                if self.selected_endpoint_config is None
                else self.selected_endpoint_config.config_sha256
            ),
            "selected_allocation_sha256": (
                None
                if self.selected_allocation is None
                else self.selected_allocation.allocation_sha256
            ),
            "parameter_count": self.parameter_count,
            "training_flops": self.training_flops,
            "parameter_relative_difference": (
                self.parameter_relative_difference
            ),
            "flop_relative_difference": self.flop_relative_difference,
            "ledger_sha256": (
                None if self.ledger is None else self.ledger.ledger_sha256
            ),
            "reference_ledger_sha256": (
                self.reference_ledger.ledger_sha256
            ),
            "status": self.status,
            "obligations": self.obligations,
        }

    def __post_init__(self) -> None:
        endpoint_profile = _profile_for_config(self.endpoint_template)
        reference_profile = _profile_for_config(self.reference_config)
        if self.config_id != self.endpoint_template.config_id:
            raise ValueError("selection config_id does not match its template")
        if (
            self.reference_config.config_id
            != "h6-a5-structured-fixed-exact-complete-latent-smoothing-v1"
            or self.reference_config.capacity_allocation
            != A5_REFERENCE_ALLOCATION
        ):
            raise ValueError("selection reference is not the frozen A5 endpoint")
        if (
            self.endpoint_template.vocabulary != self.reference_config.vocabulary
            or self.endpoint_template.horizon != self.reference_config.horizon
        ):
            raise ValueError(
                "selection endpoint and reference do not share dimensions"
            )
        if (
            self.endpoint_profile_sha256 != endpoint_profile.profile_sha256
            or self.reference_profile_sha256
            != reference_profile.profile_sha256
        ):
            raise ValueError("selection profile identity is stale")
        if type(self.workload) is not H6TrainingWorkload:
            raise ValueError("selection workload must be exact")
        self.workload.__post_init__()
        if self.policy_sha256 != AMENDED_MATCHING_SCHEDULE_POLICY.policy_sha256:
            raise ValueError("selection does not bind the amended policy")
        if type(self.candidate_count_evaluated) is not int or (
            self.candidate_count_evaluated <= 0
        ):
            raise ValueError("selection must evaluate a positive bounded inventory")
        candidate_limit = (
            len(AMENDED_EMISSION_WIDTH_CANDIDATES)
            if not endpoint_profile.latent_enabled
            else (
                len(AMENDED_EMISSION_WIDTH_CANDIDATES)
                * len(AMENDED_LATENT_WIDTH_CANDIDATES)
                * len(AMENDED_RECOGNITION_WIDTH_CANDIDATES)
            )
        )
        if self.candidate_count_evaluated > candidate_limit:
            raise ValueError("selection evaluated outside the frozen inventory")
        if type(self.reference_ledger) is not H6AnalyticalFlopLedger:
            raise ValueError("selection requires an exact reference ledger")
        self.reference_ledger.__post_init__()
        if (
            self.reference_ledger.endpoint_config_sha256
            != self.reference_config.config_sha256
            or self.reference_ledger.endpoint_profile_sha256
            != reference_profile.profile_sha256
            or self.reference_ledger.allocation_sha256
            != self.reference_config.capacity_allocation.allocation_sha256
            or self.reference_ledger.workload_sha256
            != self.workload.workload_sha256
            or self.reference_ledger.status != "COMPLETE"
        ):
            raise ValueError(
                "selection reference config/profile/allocation/workload/ledger "
                "bindings disagree"
            )
        eligible = self.status == "ELIGIBLE"
        if eligible != (
            self.selected_endpoint_config is not None
            and self.selected_allocation is not None
            and self.parameter_count is not None
            and self.training_flops is not None
            and self.parameter_relative_difference is not None
            and self.flop_relative_difference is not None
            and self.ledger is not None
            and not self.obligations
        ):
            raise ValueError("selection status and retained evidence disagree")
        if eligible:
            selected_config = self.selected_endpoint_config
            selected_allocation = self.selected_allocation
            ledger = self.ledger
            if (
                type(selected_config) is not ArmConfig
                or type(selected_allocation) is not CapacityAllocation
                or type(ledger) is not H6AnalyticalFlopLedger
            ):
                raise ValueError("eligible selection evidence is not exact")
            selected_profile = _profile_for_config(selected_config)
            selected_allocation.__post_init__()
            ledger.__post_init__()
            if (
                selected_config.config_id != self.config_id
                or selected_config.semantic_payload()
                != self.endpoint_template.semantic_payload()
                or selected_config.vocabulary != self.endpoint_template.vocabulary
                or selected_config.horizon != self.endpoint_template.horizon
                or selected_config.capacity_allocation != selected_allocation
                or selected_profile.profile_sha256
                != self.endpoint_profile_sha256
                or ledger.endpoint_config_sha256
                != selected_config.config_sha256
                or ledger.endpoint_profile_sha256
                != self.endpoint_profile_sha256
                or ledger.allocation_sha256
                != selected_allocation.allocation_sha256
                or ledger.workload_sha256 != self.workload.workload_sha256
                or ledger.status != "COMPLETE"
            ):
                raise ValueError(
                    "selection endpoint config/profile/allocation/workload/"
                    "ledger bindings disagree"
                )
            model_count, recognition_count = _parameter_components(
                selected_profile,
                selected_allocation,
                vocabulary_size=selected_config.vocabulary.size,
                horizon=selected_config.horizon,
            )
            reference_model, reference_recognition = _parameter_components(
                reference_profile,
                self.reference_config.capacity_allocation,
                vocabulary_size=self.reference_config.vocabulary.size,
                horizon=self.reference_config.horizon,
            )
            expected_parameter_count = model_count + recognition_count
            reference_parameter_count = (
                reference_model + reference_recognition
            )
            expected_parameter_difference = abs(
                expected_parameter_count - reference_parameter_count
            ) / reference_parameter_count
            expected_flop_difference = abs(
                ledger.total_arithmetic_flops
                - self.reference_ledger.total_arithmetic_flops
            ) / self.reference_ledger.total_arithmetic_flops
            if (
                self.parameter_count != expected_parameter_count
                or self.training_flops != ledger.total_arithmetic_flops
                or self.parameter_relative_difference
                != expected_parameter_difference
                or self.flop_relative_difference != expected_flop_difference
            ):
                raise ValueError("selection derived totals are stale")
            if (
                expected_parameter_difference > 0.01
                or expected_flop_difference > 0.05
            ):
                raise ValueError("eligible selection violates a hard gate")
        if not eligible and not self.obligations:
            raise ValueError("inconclusive selection requires obligations")
        if not eligible and any(
            value is not None
            for value in (
                self.selected_endpoint_config,
                self.selected_allocation,
                self.parameter_count,
                self.training_flops,
                self.parameter_relative_difference,
                self.flop_relative_difference,
                self.ledger,
            )
        ):
            raise ValueError(
                "inconclusive selection cannot retain partial endpoint evidence"
            )
        if self.selection_sha256 != _hash(
            b"vfe4.h6.formula-selection.v1", self.canonical_payload()
        ):
            raise ValueError("formula selection digest does not match")


def _amended_candidate_allocations(
    profile: EndpointFormulaProfile,
) -> Iterator[CapacityAllocation]:
    if not profile.latent_enabled:
        for emission_width in AMENDED_EMISSION_WIDTH_CANDIDATES:
            yield CapacityAllocation.create(
                emission_width=emission_width,
                latent_width=None,
                recognition_width=None,
            )
        return
    for emission_width in AMENDED_EMISSION_WIDTH_CANDIDATES:
        for latent_width in AMENDED_LATENT_WIDTH_CANDIDATES:
            for recognition_width in AMENDED_RECOGNITION_WIDTH_CANDIDATES:
                yield CapacityAllocation.create(
                    emission_width=emission_width,
                    latent_width=latent_width,
                    recognition_width=recognition_width,
                    prior_context_width=(
                        PROPOSED_PREFIX_PRIOR_CONTEXT_WIDTH
                        if profile.prior_variant == "prefix_conditioned"
                        else None
                    ),
                )


def _config_with_allocation(
    template: ArmConfig,
    allocation: CapacityAllocation,
) -> ArmConfig:
    _profile_for_config(template)
    allocation.__post_init__()
    return ArmConfig.create(
        arm=template.arm,
        config_id=template.config_id,
        vocabulary=template.vocabulary,
        horizon=template.horizon,
        latent_enabled=template.latent_enabled,
        state_channel_enabled=template.state_channel_enabled,
        model_channel_enabled=template.model_channel_enabled,
        source_mode=template.source_mode,
        map_mode=template.map_mode,
        recognition_family=template.recognition_family,
        recognition_conditioning=template.recognition_conditioning,
        prior_variant=template.prior_variant,
        mixture_mode=template.mixture_mode,
        objective_kind=template.objective_kind,
        capacity_allocation=allocation,
    )


def select_outcome_blind_allocation(
    *,
    endpoint_template: ArmConfig,
    reference_config: ArmConfig,
    workload: H6TrainingWorkload,
) -> H6FormulaSelection:
    """Select the first hard-eligible amended candidate without outcome inputs."""

    profile = _profile_for_config(endpoint_template)
    reference_profile = _profile_for_config(reference_config)
    if (
        reference_config.config_id
        != "h6-a5-structured-fixed-exact-complete-latent-smoothing-v1"
        or reference_config.capacity_allocation != A5_REFERENCE_ALLOCATION
    ):
        raise ValueError("reference_config must be the frozen A5 reference")
    if (
        endpoint_template.vocabulary != reference_config.vocabulary
        or endpoint_template.horizon != reference_config.horizon
    ):
        raise ValueError("endpoint and reference dimensions must match")
    reference_ledger = analytical_training_flop_ledger(
        endpoint_config=reference_config,
        workload=workload,
    )
    reference_model, reference_recognition = _parameter_components(
        reference_profile,
        reference_config.capacity_allocation,
        vocabulary_size=reference_config.vocabulary.size,
        horizon=reference_config.horizon,
    )
    reference_parameters = reference_model + reference_recognition
    evaluated = 0
    selected: tuple[
        ArmConfig,
        CapacityAllocation,
        int,
        H6AnalyticalFlopLedger,
        float,
        float,
    ] | None = None
    for allocation in _amended_candidate_allocations(profile):
        evaluated += 1
        candidate_config = _config_with_allocation(
            endpoint_template,
            allocation,
        )
        model_count, recognition_count = _parameter_components(
            profile,
            allocation,
            vocabulary_size=candidate_config.vocabulary.size,
            horizon=candidate_config.horizon,
        )
        parameter_count = model_count + recognition_count
        parameter_difference = abs(
            parameter_count - reference_parameters
        ) / reference_parameters
        if parameter_difference > 0.01:
            continue
        ledger = analytical_training_flop_ledger(
            endpoint_config=candidate_config,
            workload=workload,
        )
        flop_difference = abs(
            ledger.total_arithmetic_flops
            - reference_ledger.total_arithmetic_flops
        ) / reference_ledger.total_arithmetic_flops
        if flop_difference <= 0.05 and ledger.status == "COMPLETE":
            selected = (
                candidate_config,
                allocation,
                parameter_count,
                ledger,
                parameter_difference,
                flop_difference,
            )
            break
    if selected is None:
        values: dict[str, object] = {
            "config_id": endpoint_template.config_id,
            "endpoint_template": endpoint_template,
            "reference_config": reference_config,
            "endpoint_profile_sha256": profile.profile_sha256,
            "reference_profile_sha256": reference_profile.profile_sha256,
            "workload": workload,
            "policy_sha256": (
                AMENDED_MATCHING_SCHEDULE_POLICY.policy_sha256
            ),
            "candidate_count_evaluated": evaluated,
            "selected_endpoint_config": None,
            "selected_allocation": None,
            "parameter_count": None,
            "training_flops": None,
            "parameter_relative_difference": None,
            "flop_relative_difference": None,
            "ledger": None,
            "reference_ledger": reference_ledger,
            "status": "INCONCLUSIVE",
            "obligations": (
                "no amended predeclared allocation satisfies both hard gates",
            ),
        }
    else:
        (
            selected_config,
            allocation,
            parameter_count,
            ledger,
            parameter_diff,
            flop_diff,
        ) = selected
        values = {
            "config_id": endpoint_template.config_id,
            "endpoint_template": endpoint_template,
            "reference_config": reference_config,
            "endpoint_profile_sha256": profile.profile_sha256,
            "reference_profile_sha256": reference_profile.profile_sha256,
            "workload": workload,
            "policy_sha256": (
                AMENDED_MATCHING_SCHEDULE_POLICY.policy_sha256
            ),
            "candidate_count_evaluated": evaluated,
            "selected_endpoint_config": selected_config,
            "selected_allocation": allocation,
            "parameter_count": parameter_count,
            "training_flops": ledger.total_arithmetic_flops,
            "parameter_relative_difference": parameter_diff,
            "flop_relative_difference": flop_diff,
            "ledger": ledger,
            "reference_ledger": reference_ledger,
            "status": "ELIGIBLE",
            "obligations": (),
        }
    provisional = object.__new__(H6FormulaSelection)
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    return H6FormulaSelection(
        **values,  # type: ignore[arg-type]
        selection_sha256=_hash(
            b"vfe4.h6.formula-selection.v1",
            provisional.canonical_payload(),
        ),
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
                    prior_context_width=(
                        config.capacity_allocation.prior_context_width
                        if config.prior_variant == "prefix_conditioned"
                        else None
                    ),
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
    if any(
        left_parameter is right_parameter
        for left_index, (_, left_parameter) in enumerate(owned)
        for right_parameter in (
            parameter for _, parameter in owned[left_index + 1 :]
        )
    ):
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
    role_keys = [record.parameter_key for record in roles]
    if len(role_keys) != len(set(role_keys)):
        raise ValueError("a parameter has more than one declared role")
    model_phase = (
        TrainingPhase.MODEL_ADAMW.value
        if arm.config.latent_enabled
        else TrainingPhase.MODEL_CE_ADAMW.value
    )
    active_by_key = {
        stable_parameter_key(
            qualified_name=name,
            phase=(
                TrainingPhase.RECOGNITION_ADAMW.value
                if name.startswith("recognition_store.")
                else model_phase
            ),
        ): (name, parameter)
        for name, parameter in owned
    }
    missing_roles = set(active_by_key) - set(role_keys)
    unknown_roles = set(role_keys) - set(active_by_key)
    if missing_roles:
        raise ValueError(f"dormant or unbound active parameters: {sorted(missing_roles)!r}")
    if unknown_roles:
        raise ValueError(f"declared roles reference unknown parameters: {sorted(unknown_roles)!r}")
    for record in roles:
        observed_name, parameter = active_by_key[record.parameter_key]
        if record.qualified_name != observed_name:
            raise ValueError(
                "parameter role qualified name does not match its owner"
            )
        if record.scalar_count != parameter.numel():
            raise ValueError("parameter role scalar count does not match")

    bindings = tuple(arm.optimizer_bindings)
    if any(type(binding) is not OptimizerBinding for binding in bindings):
        raise ValueError("optimizer bindings must be exact OptimizerBinding values")
    binding_keys = [
        parameter_key
        for binding in bindings
        for parameter_key in binding.parameter_keys
    ]
    if len(binding_keys) != len(set(binding_keys)):
        raise ValueError("a parameter is bound to more than one optimizer")
    missing_bindings = set(role_keys) - set(binding_keys)
    unknown_bindings = set(binding_keys) - set(role_keys)
    if missing_bindings:
        raise ValueError(f"unbound active parameters: {sorted(missing_bindings)!r}")
    if unknown_bindings:
        raise ValueError(
            f"optimizer bindings reference unknown parameters: {sorted(unknown_bindings)!r}"
        )
    binding_phase_by_key = {
        parameter_key: binding.phase
        for binding in bindings
        for parameter_key in binding.parameter_keys
    }
    for record in roles:
        if binding_phase_by_key[record.parameter_key] != record.phase:
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
    workload: H6TrainingWorkload | None = None,
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
    if workload is None:
        endpoint_terms = endpoint.flop_terms
        reference_terms = reference.flop_terms
        endpoint_flops_complete, endpoint_flop_obligations = (
            _training_flop_review(endpoint, endpoint_name="endpoint")
        )
        reference_flops_complete, reference_flop_obligations = (
            _training_flop_review(reference, endpoint_name="reference")
        )
        common_schedule = _common_schedule_is_proven(
            endpoint,
            reference,
            expected_schedule_sha256=(
                resolved_matching.matching_schedule_sha256
            ),
        )
        common_schedule_sha256 = (
            resolved_matching.matching_schedule_sha256
        )
    else:
        workload.__post_init__()
        endpoint_ledger = analytical_training_flop_ledger(
            endpoint_config=endpoint.config,
            workload=workload,
        )
        reference_ledger = analytical_training_flop_ledger(
            endpoint_config=reference.config,
            workload=workload,
        )
        endpoint_terms = endpoint_ledger.terms
        reference_terms = reference_ledger.terms
        endpoint_flops_complete = endpoint_ledger.status == "COMPLETE"
        reference_flops_complete = reference_ledger.status == "COMPLETE"
        endpoint_flop_obligations = endpoint_ledger.obligations
        reference_flop_obligations = reference_ledger.obligations
        common_schedule = True
        common_schedule_sha256 = (
            AMENDED_MATCHING_SCHEDULE_POLICY.policy_sha256
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
            term.total_arithmetic_flops for term in endpoint_terms
        ),
        reference_training_flops=sum(
            term.total_arithmetic_flops for term in reference_terms
        ),
        parameter_relative_tolerance=(
            resolved_matching.parameter_relative_tolerance
        ),
        flop_relative_tolerance=resolved_matching.flop_relative_tolerance,
        ownership_valid=ownership_valid,
        common_schedule=common_schedule,
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
        common_schedule_sha256=common_schedule_sha256,
    )


__all__ = [
    "A5_REFERENCE_ALLOCATION",
    "AMENDED_MATCHING_SCHEDULE_POLICY",
    "ARM_MATRIX_ROWS",
    "ARM_MATRIX_SHA256",
    "EMISSION_WIDTH_CANDIDATES",
    "H6_ADAMW_POLICY",
    "LATENT_WIDTH_CANDIDATES",
    "MATCHING_SCHEDULE_POLICY",
    "RECOGNITION_WIDTH_CANDIDATES",
    "AdamWPolicyRecord",
    "AmendedMatchingSchedulePolicy",
    "ArmConfig",
    "ArmMatrixRow",
    "CapacityAllocation",
    "EndpointFormulaProfile",
    "FlopTerm",
    "H6AnalyticalFlopLedger",
    "MatchingReport",
    "H6FormulaSelection",
    "H6TrainingWorkload",
    "OptimizerBinding",
    "ParameterRoleRecord",
    "adamw_flops",
    "analytical_training_flop_ledger",
    "arm_matrix_sha256",
    "audit_arm_matching",
    "audit_parameter_ownership",
    "backward_flops",
    "candidate_allocations",
    "capacity_candidate_count",
    "dense_matmul_flops",
    "dense_matvec_flops",
    "endpoint_formula_profile",
    "immutable_snapshot_flop_term",
    "l2_clip_scale_flops",
    "log_softmax_flops",
    "matrix_exp_pade13_flops",
    "matrix_solve_lu_flops",
    "scalar_flops",
    "select_outcome_blind_allocation",
    "stable_parameter_key",
]
