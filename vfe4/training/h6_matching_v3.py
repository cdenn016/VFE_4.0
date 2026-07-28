"""Outcome-blind analytical matching for executable H6 Prediction v3.

This module is additive.  It consumes the historical v2 analytical ledger as
an immutable base calculation, then adds every operation introduced by the
v3 categorical-recognition and terminal-mixture implementation under new
policy, workload, and ledger hash domains.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

from vfe4.types.h6 import (
    ArmConfig,
    ArmMatrixRow,
    CapacityAllocation,
    FlopTerm,
    MatchingReport,
    TrainingPhase,
    canonical_json_bytes,
)

from .matching import (
    A5_REFERENCE_ALLOCATION,
    ARM_MATRIX_ROWS,
    H6_ADAMW_POLICY,
    H6TrainingWorkload,
    PRIMARY_LATENT_WIDTH_CANDIDATES,
    PRIMARY_PRIOR_CONTEXT_WIDTH_CANDIDATES,
    PRIMARY_RECOGNITION_WIDTH_CANDIDATES,
    adamw_flops,
    analytical_training_flop_ledger,
    endpoint_formula_profile,
)
from .parameter_counts import (
    AMENDED_EMISSION_WIDTH_CANDIDATES,
    AMENDED_LATENT_WIDTH_CANDIDATES,
    AMENDED_RECOGNITION_WIDTH_CANDIDATES,
    PROPOSED_PREFIX_PRIOR_CONTEXT_WIDTH,
    arm_parameter_count_v3,
    arm_source_bank_count_v3,
    fixed_source_prior_parameter_count,
    parent_specific_pooled_prefix_source_prior_parameter_count,
    recognition_source_parameter_count_v3,
)


_LOWER_HEX = frozenset("0123456789abcdef")

PRIMARY_EMISSION_WIDTH_CANDIDATES_V3 = (72, 84, 85, 86, 87, 88, 89)
PRIMARY_JOINT_CANDIDATE_COUNT_V3 = 378

H6_MATCHING_V3_ESTIMATOR_TERM_NAMES = (
    "categorical_context_residual_dot",
    "categorical_lag_scalar",
    "positive_support_log_softmax",
    "terminal_rank_one_component_shift",
    "terminal_component_realization",
    "exact_nested_source_reduction",
    "moment_projection_construction",
    "recognition_phase_backward",
    "global_norm_clipping",
    "adamw_update",
)
H6_MATCHING_V3_EXCLUDED_OPERATIONS = (
    "data_io",
    "validation",
    "checkpoint_serialization",
    "test_scoring",
    "cpu_to_cuda_noise_transfer",
    "prediction_particle_propagation",
    "prediction_cache",
)
H6_MATCHING_V3_ENDPOINT_CONFIG_IDS = (
    "h6-a0-transformer-v2",
    "h6-a1-ordinary-latent-v1",
    "h6-a2-generic-map-v1",
    "h6-a3-immediate-predecessor-v1",
    "h6-a4-state-only-v1",
    "h6-a5-structured-fixed-exact-complete-latent-smoothing-v1",
    "h6-a5-factorized-fixed-exact-complete-latent-smoothing-v1",
    ("h6-a5-structured-parent-specific-prefix-exact-complete-latent-smoothing-v2"),
    "h6-a5-structured-fixed-projection-complete-latent-smoothing-v1",
    ("h6-a5-structured-parent-specific-prefix-exact-emission-latent-smoothing-v2"),
    ("h6-a5-structured-fixed-exact-complete-nolatent-norecognition-v1"),
    "h6-a5-structured-fixed-exact-complete-latent-filtering-v1",
)
_PRIMARY_A0_CONFIG_ID = H6_MATCHING_V3_ENDPOINT_CONFIG_IDS[0]
_REFERENCE_CONFIG_ID = H6_MATCHING_V3_ENDPOINT_CONFIG_IDS[5]
_PRIMARY_A5_CONFIG_ID = H6_MATCHING_V3_ENDPOINT_CONFIG_IDS[7]
_OBJECTIVE_A5_CONFIG_ID = H6_MATCHING_V3_ENDPOINT_CONFIG_IDS[9]
_COMPONENT_SELECTION_CONFIG_IDS = tuple(
    config_id
    for config_id in H6_MATCHING_V3_ENDPOINT_CONFIG_IDS
    if config_id
    not in (
        _PRIMARY_A0_CONFIG_ID,
        _REFERENCE_CONFIG_ID,
        _PRIMARY_A5_CONFIG_ID,
        _OBJECTIVE_A5_CONFIG_ID,
    )
)


def _hash(domain: str, payload: object) -> str:
    return hashlib.sha256(
        domain.encode("ascii") + b"\x00" + canonical_json_bytes(payload)
    ).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class H6MatchingPolicyV3:
    schema_version: Literal["h6-amended-matching-policy-v3"]
    reference_allocation_sha256: str
    emission_width_candidates: tuple[int, ...]
    latent_width_candidates: tuple[int, ...]
    recognition_width_candidates: tuple[int, ...]
    primary_latent_width_candidates: tuple[int, ...]
    primary_prior_context_width_candidates: tuple[int, ...]
    primary_emission_width_candidates: tuple[int, ...]
    primary_recognition_width_candidates: tuple[int, ...]
    primary_joint_candidate_count: Literal[378]
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
    estimator_term_names: tuple[str, ...]
    policy_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name) for name in tuple(self.__dataclass_fields__)[:-1]
        }

    def __post_init__(self) -> None:
        expected = _matching_policy_payload_v3()
        if self.canonical_payload() != expected:
            raise ValueError("matching v3 policy differs from the frozen design")
        if self.policy_sha256 != _hash(
            "vfe4.h6.amended-matching-policy.v3",
            expected,
        ):
            raise ValueError("matching v3 policy digest is stale")


def _matching_policy_payload_v3() -> dict[str, object]:
    return {
        "schema_version": "h6-amended-matching-policy-v3",
        "reference_allocation_sha256": (A5_REFERENCE_ALLOCATION.allocation_sha256),
        "emission_width_candidates": AMENDED_EMISSION_WIDTH_CANDIDATES,
        "latent_width_candidates": AMENDED_LATENT_WIDTH_CANDIDATES,
        "recognition_width_candidates": (AMENDED_RECOGNITION_WIDTH_CANDIDATES),
        "primary_latent_width_candidates": PRIMARY_LATENT_WIDTH_CANDIDATES,
        "primary_prior_context_width_candidates": (
            PRIMARY_PRIOR_CONTEXT_WIDTH_CANDIDATES
        ),
        "primary_emission_width_candidates": (PRIMARY_EMISSION_WIDTH_CANDIDATES_V3),
        "primary_recognition_width_candidates": (PRIMARY_RECOGNITION_WIDTH_CANDIDATES),
        "primary_joint_candidate_count": PRIMARY_JOINT_CANDIDATE_COUNT_V3,
        "prior_context_width": PROPOSED_PREFIX_PRIOR_CONTEXT_WIDTH,
        "parameter_relative_tolerance": 0.01,
        "flop_relative_tolerance": 0.05,
        "optimizer_policy_sha256": (H6_ADAMW_POLICY.optimizer_policy_sha256),
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
        "excluded_operations": H6_MATCHING_V3_EXCLUDED_OPERATIONS,
        "estimator_term_names": H6_MATCHING_V3_ESTIMATOR_TERM_NAMES,
    }


_POLICY_VALUES = _matching_policy_payload_v3()
H6_MATCHING_POLICY_V3 = H6MatchingPolicyV3(
    **_POLICY_VALUES,  # type: ignore[arg-type]
    policy_sha256=_hash(
        "vfe4.h6.amended-matching-policy.v3",
        _POLICY_VALUES,
    ),
)


@dataclass(frozen=True, slots=True)
class H6TrainingWorkloadV3:
    schema_version: Literal["h6-training-workload-v3"]
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
    matching_policy_sha256: str
    workload_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name) for name in tuple(self.__dataclass_fields__)[:-1]
        }

    def __post_init__(self) -> None:
        if (
            self.schema_version != "h6-training-workload-v3"
            or type(self.train_token_count) is not int
            or self.train_token_count < 2
            or self.sequence_length != 32
            or self.window_stride != 32
            or self.batch_size != 8
            or self.drop_last is not False
            or self.full_passes != 2
        ):
            raise ValueError("training workload v3 is not frozen")
        _require_sha256(self.train_token_sha256, "train_token_sha256")
        windows = (self.train_token_count - 2) // 32 + 1
        full_batches, tail = divmod(windows, 8)
        batches = full_batches + bool(tail)
        boundaries = tuple(
            dict.fromkeys((index * batches + 19) // 20 for index in range(1, 21))
        )
        if (
            self.window_count != windows
            or self.full_batches_per_pass != full_batches
            or self.tail_batch_size != tail
            or self.batches_per_pass != batches
            or self.model_update_opportunities != 2 * batches
            or self.validation_boundaries_per_pass != boundaries
            or self.matching_policy_sha256 != H6_MATCHING_POLICY_V3.policy_sha256
        ):
            raise ValueError("training workload v3 counts are inconsistent")
        if self.workload_sha256 != _hash(
            "vfe4.h6.training-workload.v3",
            self.canonical_payload(),
        ):
            raise ValueError("training workload v3 digest is stale")

    @classmethod
    def from_train_tokens(
        cls,
        *,
        train_token_count: int,
        train_token_sha256: str,
    ) -> "H6TrainingWorkloadV3":
        if type(train_token_count) is not int or train_token_count < 2:
            raise ValueError("train_token_count must be at least two")
        windows = (train_token_count - 2) // 32 + 1
        full_batches, tail = divmod(windows, 8)
        batches = full_batches + bool(tail)
        values = {
            "schema_version": "h6-training-workload-v3",
            "train_token_count": train_token_count,
            "train_token_sha256": train_token_sha256,
            "sequence_length": 32,
            "window_stride": 32,
            "batch_size": 8,
            "drop_last": False,
            "full_passes": 2,
            "window_count": windows,
            "full_batches_per_pass": full_batches,
            "tail_batch_size": tail,
            "batches_per_pass": batches,
            "model_update_opportunities": 2 * batches,
            "validation_boundaries_per_pass": tuple(
                dict.fromkeys((index * batches + 19) // 20 for index in range(1, 21))
            ),
            "matching_policy_sha256": H6_MATCHING_POLICY_V3.policy_sha256,
        }
        return cls(
            **values,  # type: ignore[arg-type]
            workload_sha256=_hash(
                "vfe4.h6.training-workload.v3",
                values,
            ),
        )

    def legacy_projection(self) -> H6TrainingWorkload:
        """Create the exact historical workload used only as base arithmetic."""

        return H6TrainingWorkload.from_train_tokens(
            train_token_count=self.train_token_count,
            train_token_sha256=self.train_token_sha256,
        )


@dataclass(frozen=True, slots=True)
class H6AnalyticalFlopLedgerV3:
    schema_version: Literal["h6-analytical-flop-ledger-v3"]
    endpoint_config_sha256: str
    matching_policy_sha256: str
    workload_sha256: str
    legacy_base_ledger_sha256: str
    source_bank_count: Literal[0, 1, 2]
    source_parameter_count: int
    terms: tuple[FlopTerm, ...]
    excluded_operations: tuple[str, ...]
    total_arithmetic_flops: int
    total_bytes_copied: int
    ledger_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "endpoint_config_sha256": self.endpoint_config_sha256,
            "matching_policy_sha256": self.matching_policy_sha256,
            "workload_sha256": self.workload_sha256,
            "legacy_base_ledger_sha256": self.legacy_base_ledger_sha256,
            "source_bank_count": self.source_bank_count,
            "source_parameter_count": self.source_parameter_count,
            "term_sha256": tuple(term.term_sha256 for term in self.terms),
            "excluded_operations": self.excluded_operations,
            "total_arithmetic_flops": self.total_arithmetic_flops,
            "total_bytes_copied": self.total_bytes_copied,
        }

    def __post_init__(self) -> None:
        for name in (
            "endpoint_config_sha256",
            "matching_policy_sha256",
            "workload_sha256",
            "legacy_base_ledger_sha256",
            "ledger_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if (
            self.schema_version != "h6-analytical-flop-ledger-v3"
            or self.matching_policy_sha256 != H6_MATCHING_POLICY_V3.policy_sha256
            or self.source_bank_count not in (0, 1, 2)
            or type(self.source_parameter_count) is not int
            or self.source_parameter_count < 0
            or type(self.terms) is not tuple
            or not self.terms
            or self.excluded_operations != H6_MATCHING_V3_EXCLUDED_OPERATIONS
        ):
            raise ValueError("analytical v3 ledger fields are invalid")
        if self.total_arithmetic_flops != sum(
            term.total_arithmetic_flops for term in self.terms
        ):
            raise ValueError("v3 ledger arithmetic total is inconsistent")
        if self.total_bytes_copied != sum(
            term.total_bytes_copied for term in self.terms
        ):
            raise ValueError("v3 ledger byte total is inconsistent")
        if self.ledger_sha256 != _hash(
            "vfe4.h6.analytical-flop-ledger.v3",
            self.canonical_payload(),
        ):
            raise ValueError("analytical v3 ledger digest is stale")


@dataclass(frozen=True, slots=True)
class H6PrimaryMatchingCandidateV3:
    """One exact v3 PRIMARY candidate in ascending ``(d,c,e,r)`` order."""

    ordinal: int
    endpoint_config: ArmConfig
    ledger: H6AnalyticalFlopLedgerV3
    a0_config_sha256: str
    a0_ledger_sha256: str
    a0_parameter_count: int
    a0_training_flops: int
    endpoint_parameter_count: int
    endpoint_training_flops: int
    parameter_relative_difference: float
    flop_relative_difference: float
    hard_eligible: bool
    candidate_sha256: str

    @property
    def allocation_key(self) -> tuple[int, int, int, int]:
        allocation = self.endpoint_config.capacity_allocation
        if (
            allocation.latent_width is None
            or allocation.prior_context_width is None
            or allocation.recognition_width is None
        ):
            raise ValueError("PRIMARY candidate lacks four live widths")
        return (
            allocation.latent_width,
            allocation.prior_context_width,
            allocation.emission_width,
            allocation.recognition_width,
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "allocation_key": self.allocation_key,
            "endpoint_config_sha256": self.endpoint_config.config_sha256,
            "ledger_sha256": self.ledger.ledger_sha256,
            "a0_config_sha256": self.a0_config_sha256,
            "a0_ledger_sha256": self.a0_ledger_sha256,
            "a0_parameter_count": self.a0_parameter_count,
            "a0_training_flops": self.a0_training_flops,
            "endpoint_parameter_count": self.endpoint_parameter_count,
            "endpoint_training_flops": self.endpoint_training_flops,
            "parameter_relative_difference": (self.parameter_relative_difference),
            "flop_relative_difference": self.flop_relative_difference,
            "hard_eligible": self.hard_eligible,
        }

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("PRIMARY candidate ordinal must be nonnegative")
        if type(self.endpoint_config) is not ArmConfig:
            raise ValueError("PRIMARY candidate config must be exact")
        if type(self.ledger) is not H6AnalyticalFlopLedgerV3:
            raise ValueError("PRIMARY candidate ledger must be exact v3")
        self.endpoint_config.__post_init__()
        self.ledger.__post_init__()
        for name in (
            "a0_config_sha256",
            "a0_ledger_sha256",
            "candidate_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if (
            self.endpoint_config.config_id != _PRIMARY_A5_CONFIG_ID
            or self.endpoint_config.prior_variant != "parent_specific_pooled_prefix"
            or self.ledger.endpoint_config_sha256 != self.endpoint_config.config_sha256
        ):
            raise ValueError("PRIMARY candidate config/ledger binding is stale")
        for name in (
            "a0_parameter_count",
            "a0_training_flops",
            "endpoint_parameter_count",
            "endpoint_training_flops",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        expected_endpoint_count = endpoint_parameter_count_v3(self.endpoint_config)
        expected_parameter_difference = (
            abs(expected_endpoint_count - self.a0_parameter_count)
            / self.a0_parameter_count
        )
        expected_flop_difference = (
            abs(self.ledger.total_arithmetic_flops - self.a0_training_flops)
            / self.a0_training_flops
        )
        expected_eligible = (
            expected_parameter_difference
            <= H6_MATCHING_POLICY_V3.parameter_relative_tolerance
            and expected_flop_difference
            <= H6_MATCHING_POLICY_V3.flop_relative_tolerance
        )
        if (
            self.endpoint_parameter_count != expected_endpoint_count
            or self.endpoint_training_flops != self.ledger.total_arithmetic_flops
            or self.parameter_relative_difference != expected_parameter_difference
            or self.flop_relative_difference != expected_flop_difference
            or self.hard_eligible is not expected_eligible
        ):
            raise ValueError("PRIMARY candidate derived gates are stale")
        if self.candidate_sha256 != _hash(
            "vfe4.h6.primary-matching-candidate.v3",
            self.canonical_payload(),
        ):
            raise ValueError("PRIMARY candidate digest is stale")

    @classmethod
    def create(
        cls,
        *,
        ordinal: int,
        endpoint_config: ArmConfig,
        ledger: H6AnalyticalFlopLedgerV3,
        a0_config: ArmConfig,
        a0_ledger: H6AnalyticalFlopLedgerV3,
    ) -> "H6PrimaryMatchingCandidateV3":
        a0_parameter_count = endpoint_parameter_count_v3(a0_config)
        endpoint_parameter_count = endpoint_parameter_count_v3(endpoint_config)
        parameter_difference = (
            abs(endpoint_parameter_count - a0_parameter_count) / a0_parameter_count
        )
        flop_difference = (
            abs(ledger.total_arithmetic_flops - a0_ledger.total_arithmetic_flops)
            / a0_ledger.total_arithmetic_flops
        )
        values = {
            "ordinal": ordinal,
            "endpoint_config": endpoint_config,
            "ledger": ledger,
            "a0_config_sha256": a0_config.config_sha256,
            "a0_ledger_sha256": a0_ledger.ledger_sha256,
            "a0_parameter_count": a0_parameter_count,
            "a0_training_flops": a0_ledger.total_arithmetic_flops,
            "endpoint_parameter_count": endpoint_parameter_count,
            "endpoint_training_flops": ledger.total_arithmetic_flops,
            "parameter_relative_difference": parameter_difference,
            "flop_relative_difference": flop_difference,
            "hard_eligible": (
                parameter_difference
                <= H6_MATCHING_POLICY_V3.parameter_relative_tolerance
                and flop_difference <= H6_MATCHING_POLICY_V3.flop_relative_tolerance
            ),
        }
        provisional = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(provisional, name, value)
        return cls(
            **values,  # type: ignore[arg-type]
            candidate_sha256=_hash(
                "vfe4.h6.primary-matching-candidate.v3",
                provisional.canonical_payload(),
            ),
        )


@dataclass(frozen=True, slots=True)
class H6PrimaryMatchingSelectionV3:
    """Complete v3 378-row search with first-eligible fail-closed selection."""

    schema_version: Literal["h6-primary-matching-selection-v3"]
    matching_policy_sha256: str
    workload_sha256: str
    a0_config: ArmConfig
    a5_template: ArmConfig
    candidates: tuple[H6PrimaryMatchingCandidateV3, ...]
    candidate_inventory_sha256: str
    status: Literal["ELIGIBLE", "INCONCLUSIVE"]
    selected_candidate_sha256: str | None
    obligations: tuple[str, ...]
    selection_sha256: str

    @property
    def selected_candidate(self) -> H6PrimaryMatchingCandidateV3 | None:
        if self.selected_candidate_sha256 is None:
            return None
        return next(
            candidate
            for candidate in self.candidates
            if candidate.candidate_sha256 == self.selected_candidate_sha256
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "matching_policy_sha256": self.matching_policy_sha256,
            "workload_sha256": self.workload_sha256,
            "a0_config_sha256": self.a0_config.config_sha256,
            "a5_template_config_sha256": self.a5_template.config_sha256,
            "candidate_inventory_sha256": self.candidate_inventory_sha256,
            "candidate_count": len(self.candidates),
            "status": self.status,
            "selected_candidate_sha256": self.selected_candidate_sha256,
            "obligations": self.obligations,
        }

    def __post_init__(self) -> None:
        if self.schema_version != "h6-primary-matching-selection-v3":
            raise ValueError("PRIMARY v3 selection schema is frozen")
        for name in (
            "matching_policy_sha256",
            "workload_sha256",
            "candidate_inventory_sha256",
            "selection_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if self.matching_policy_sha256 != H6_MATCHING_POLICY_V3.policy_sha256:
            raise ValueError("PRIMARY selection does not bind policy v3")
        if (
            type(self.a0_config) is not ArmConfig
            or type(self.a5_template) is not ArmConfig
        ):
            raise ValueError("PRIMARY selection endpoints must be exact")
        self.a0_config.__post_init__()
        self.a5_template.__post_init__()
        a0_allocation = self.a0_config.capacity_allocation
        if (
            self.a0_config.config_id != _PRIMARY_A0_CONFIG_ID
            or (
                a0_allocation.emission_width,
                a0_allocation.latent_width,
                a0_allocation.recognition_width,
                a0_allocation.prior_context_width,
            )
            != (52, None, None, None)
            or self.a5_template.config_id != _PRIMARY_A5_CONFIG_ID
            or self.a5_template.prior_variant != "parent_specific_pooled_prefix"
            or self.a0_config.vocabulary != self.a5_template.vocabulary
            or self.a0_config.horizon != self.a5_template.horizon
        ):
            raise ValueError("PRIMARY v3 endpoint templates are not frozen")
        if (
            type(self.candidates) is not tuple
            or len(self.candidates) != PRIMARY_JOINT_CANDIDATE_COUNT_V3
            or any(
                type(candidate) is not H6PrimaryMatchingCandidateV3
                for candidate in self.candidates
            )
        ):
            raise ValueError("PRIMARY v3 candidate inventory is incomplete")
        expected_axes = tuple(
            (d, c, e, r)
            for d in PRIMARY_LATENT_WIDTH_CANDIDATES
            for c in PRIMARY_PRIOR_CONTEXT_WIDTH_CANDIDATES
            for e in PRIMARY_EMISSION_WIDTH_CANDIDATES_V3
            for r in PRIMARY_RECOGNITION_WIDTH_CANDIDATES
        )
        if (
            tuple(candidate.ordinal for candidate in self.candidates)
            != tuple(range(PRIMARY_JOINT_CANDIDATE_COUNT_V3))
            or tuple(candidate.allocation_key for candidate in self.candidates)
            != expected_axes
        ):
            raise ValueError("PRIMARY v3 candidates are missing, extra, or reordered")
        for candidate in self.candidates:
            candidate.__post_init__()
            if (
                candidate.a0_config_sha256 != self.a0_config.config_sha256
                or candidate.ledger.workload_sha256 != self.workload_sha256
            ):
                raise ValueError("PRIMARY v3 candidate preimages drifted")
        expected_inventory_sha256 = _hash(
            "vfe4.h6.primary-matching-candidate-inventory.v3",
            tuple(candidate.candidate_sha256 for candidate in self.candidates),
        )
        if self.candidate_inventory_sha256 != expected_inventory_sha256:
            raise ValueError("PRIMARY v3 candidate inventory digest is stale")
        expected_selected = next(
            (candidate for candidate in self.candidates if candidate.hard_eligible),
            None,
        )
        if expected_selected is None:
            if (
                self.status != "INCONCLUSIVE"
                or self.selected_candidate_sha256 is not None
                or self.obligations
                != ("no v3 PRIMARY candidate closes both frozen hard gates",)
            ):
                raise ValueError("PRIMARY v3 empty eligible set must fail closed")
        elif (
            self.status != "ELIGIBLE"
            or self.selected_candidate_sha256 != expected_selected.candidate_sha256
            or self.obligations
        ):
            raise ValueError(
                "PRIMARY v3 selection is not the first hard-eligible candidate"
            )
        if self.selection_sha256 != _hash(
            "vfe4.h6.primary-matching-selection.v3",
            self.canonical_payload(),
        ):
            raise ValueError("PRIMARY v3 selection digest is stale")

    @classmethod
    def create(
        cls,
        *,
        a0_config: ArmConfig,
        a5_template: ArmConfig,
        workload: H6TrainingWorkloadV3,
    ) -> "H6PrimaryMatchingSelectionV3":
        if type(workload) is not H6TrainingWorkloadV3:
            raise ValueError("PRIMARY v3 search requires an exact workload")
        workload.__post_init__()
        a0_ledger = analytical_training_flop_ledger_v3(
            endpoint_config=a0_config,
            workload=workload,
        )
        candidates: list[H6PrimaryMatchingCandidateV3] = []
        for d in PRIMARY_LATENT_WIDTH_CANDIDATES:
            for c in PRIMARY_PRIOR_CONTEXT_WIDTH_CANDIDATES:
                for e in PRIMARY_EMISSION_WIDTH_CANDIDATES_V3:
                    for r in PRIMARY_RECOGNITION_WIDTH_CANDIDATES:
                        endpoint_config = _config_with_allocation_v3(
                            a5_template,
                            CapacityAllocation.create(
                                emission_width=e,
                                latent_width=d,
                                recognition_width=r,
                                prior_context_width=c,
                            ),
                        )
                        ledger = analytical_training_flop_ledger_v3(
                            endpoint_config=endpoint_config,
                            workload=workload,
                        )
                        candidates.append(
                            H6PrimaryMatchingCandidateV3.create(
                                ordinal=len(candidates),
                                endpoint_config=endpoint_config,
                                ledger=ledger,
                                a0_config=a0_config,
                                a0_ledger=a0_ledger,
                            )
                        )
        candidate_records = tuple(candidates)
        selected = next(
            (candidate for candidate in candidate_records if candidate.hard_eligible),
            None,
        )
        inventory_sha256 = _hash(
            "vfe4.h6.primary-matching-candidate-inventory.v3",
            tuple(candidate.candidate_sha256 for candidate in candidate_records),
        )
        values = {
            "schema_version": "h6-primary-matching-selection-v3",
            "matching_policy_sha256": H6_MATCHING_POLICY_V3.policy_sha256,
            "workload_sha256": workload.workload_sha256,
            "a0_config": a0_config,
            "a5_template": a5_template,
            "candidates": candidate_records,
            "candidate_inventory_sha256": inventory_sha256,
            "status": ("ELIGIBLE" if selected is not None else "INCONCLUSIVE"),
            "selected_candidate_sha256": (
                None if selected is None else selected.candidate_sha256
            ),
            "obligations": (
                ()
                if selected is not None
                else ("no v3 PRIMARY candidate closes both frozen hard gates",)
            ),
        }
        provisional = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(provisional, name, value)
        return cls(
            **values,  # type: ignore[arg-type]
            selection_sha256=_hash(
                "vfe4.h6.primary-matching-selection.v3",
                provisional.canonical_payload(),
            ),
        )


def primary_matching_diagnostics_v3(
    selection: H6PrimaryMatchingSelectionV3,
) -> tuple[dict[str, object], ...]:
    """Return deterministic closest-candidate diagnostics without selection."""

    if type(selection) is not H6PrimaryMatchingSelectionV3:
        raise ValueError("diagnostics require an exact PRIMARY v3 selection")
    selection.__post_init__()
    minimum_flop = min(
        selection.candidates,
        key=lambda candidate: (
            candidate.flop_relative_difference,
            candidate.ordinal,
        ),
    )
    minimum_parameter = min(
        selection.candidates,
        key=lambda candidate: (
            candidate.parameter_relative_difference,
            candidate.ordinal,
        ),
    )
    within_parameter_gate = tuple(
        candidate
        for candidate in selection.candidates
        if candidate.parameter_relative_difference
        <= H6_MATCHING_POLICY_V3.parameter_relative_tolerance
    )
    if not within_parameter_gate:
        raise ValueError("PRIMARY diagnostics found no parameter-gate candidate")
    minimum_flop_with_parameter_gate = min(
        within_parameter_gate,
        key=lambda candidate: (
            candidate.flop_relative_difference,
            candidate.ordinal,
        ),
    )
    return tuple(
        {
            "criterion": criterion,
            "allocation": candidate.allocation_key,
            "parameter_count": candidate.endpoint_parameter_count,
            "training_flops": candidate.endpoint_training_flops,
            "parameter_relative_difference": (candidate.parameter_relative_difference),
            "flop_relative_difference": candidate.flop_relative_difference,
            "hard_eligible": candidate.hard_eligible,
        }
        for criterion, candidate in (
            ("minimum_flop_relative_difference", minimum_flop),
            (
                "minimum_parameter_relative_difference",
                minimum_parameter,
            ),
            (
                "minimum_flop_gap_within_parameter_gate",
                minimum_flop_with_parameter_gate,
            ),
        )
    )


def _component_candidate_allocations_v3(
    template: ArmConfig,
) -> tuple[CapacityAllocation, ...]:
    profile = endpoint_formula_profile(template.config_id)
    if not profile.latent_enabled:
        return tuple(
            CapacityAllocation.create(
                emission_width=emission_width,
                latent_width=None,
                recognition_width=None,
            )
            for emission_width in AMENDED_EMISSION_WIDTH_CANDIDATES
        )
    return tuple(
        CapacityAllocation.create(
            emission_width=emission_width,
            latent_width=latent_width,
            recognition_width=recognition_width,
            prior_context_width=(
                H6_MATCHING_POLICY_V3.prior_context_width
                if profile.prior_variant == "parent_specific_pooled_prefix"
                else None
            ),
        )
        for emission_width in AMENDED_EMISSION_WIDTH_CANDIDATES
        for latent_width in AMENDED_LATENT_WIDTH_CANDIDATES
        for recognition_width in AMENDED_RECOGNITION_WIDTH_CANDIDATES
    )


def _component_selection_values_v3(
    *,
    endpoint_template: ArmConfig,
    reference_config: ArmConfig,
    workload: H6TrainingWorkloadV3,
) -> dict[str, object]:
    if endpoint_template.config_id not in _COMPONENT_SELECTION_CONFIG_IDS:
        raise ValueError("component selection endpoint identity is not frozen")
    if (
        reference_config.config_id != _REFERENCE_CONFIG_ID
        or reference_config.capacity_allocation != A5_REFERENCE_ALLOCATION
        or endpoint_template.vocabulary != reference_config.vocabulary
        or endpoint_template.horizon != reference_config.horizon
    ):
        raise ValueError("component selection reference is not frozen")
    reference_count = endpoint_parameter_count_v3(reference_config)
    reference_ledger = analytical_training_flop_ledger_v3(
        endpoint_config=reference_config,
        workload=workload,
    )
    selected_config: ArmConfig | None = None
    selected_ledger: H6AnalyticalFlopLedgerV3 | None = None
    parameter_count: int | None = None
    training_flops: int | None = None
    parameter_difference: float | None = None
    flop_difference: float | None = None
    evaluated = 0
    for allocation in _component_candidate_allocations_v3(endpoint_template):
        evaluated += 1
        candidate_config = _config_with_allocation_v3(
            endpoint_template,
            allocation,
        )
        candidate_count = endpoint_parameter_count_v3(candidate_config)
        candidate_parameter_difference = (
            abs(candidate_count - reference_count) / reference_count
        )
        if (
            candidate_parameter_difference
            > H6_MATCHING_POLICY_V3.parameter_relative_tolerance
        ):
            continue
        candidate_ledger = analytical_training_flop_ledger_v3(
            endpoint_config=candidate_config,
            workload=workload,
        )
        candidate_flop_difference = (
            abs(
                candidate_ledger.total_arithmetic_flops
                - reference_ledger.total_arithmetic_flops
            )
            / reference_ledger.total_arithmetic_flops
        )
        if candidate_flop_difference <= H6_MATCHING_POLICY_V3.flop_relative_tolerance:
            selected_config = candidate_config
            selected_ledger = candidate_ledger
            parameter_count = candidate_count
            training_flops = candidate_ledger.total_arithmetic_flops
            parameter_difference = candidate_parameter_difference
            flop_difference = candidate_flop_difference
            break
    selected = selected_config is not None
    return {
        "schema_version": "h6-component-matching-selection-v3",
        "endpoint_template": endpoint_template,
        "reference_config": reference_config,
        "workload": workload,
        "matching_policy_sha256": H6_MATCHING_POLICY_V3.policy_sha256,
        "candidate_count_evaluated": evaluated,
        "selected_endpoint_config": selected_config,
        "selected_ledger": selected_ledger,
        "parameter_count": parameter_count,
        "training_flops": training_flops,
        "parameter_relative_difference": parameter_difference,
        "flop_relative_difference": flop_difference,
        "status": "ELIGIBLE" if selected else "INCONCLUSIVE",
        "obligations": (
            ()
            if selected
            else ("no v3 component candidate closes both frozen hard gates",)
        ),
    }


@dataclass(frozen=True, slots=True)
class H6ComponentMatchingSelectionV3:
    """One exact component search with recomputed first-eligible selection."""

    schema_version: Literal["h6-component-matching-selection-v3"]
    endpoint_template: ArmConfig
    reference_config: ArmConfig
    workload: H6TrainingWorkloadV3
    matching_policy_sha256: str
    candidate_count_evaluated: int
    selected_endpoint_config: ArmConfig | None
    selected_ledger: H6AnalyticalFlopLedgerV3 | None
    parameter_count: int | None
    training_flops: int | None
    parameter_relative_difference: float | None
    flop_relative_difference: float | None
    status: Literal["ELIGIBLE", "INCONCLUSIVE"]
    obligations: tuple[str, ...]
    selection_sha256: str

    @property
    def config_id(self) -> str:
        return self.endpoint_template.config_id

    @property
    def active_config(self) -> ArmConfig:
        return (
            self.endpoint_template
            if self.selected_endpoint_config is None
            else self.selected_endpoint_config
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "endpoint_template_config_sha256": (self.endpoint_template.config_sha256),
            "reference_config_sha256": (self.reference_config.config_sha256),
            "workload_sha256": self.workload.workload_sha256,
            "matching_policy_sha256": self.matching_policy_sha256,
            "candidate_count_evaluated": self.candidate_count_evaluated,
            "selected_endpoint_config_sha256": (
                None
                if self.selected_endpoint_config is None
                else self.selected_endpoint_config.config_sha256
            ),
            "selected_ledger_sha256": (
                None
                if self.selected_ledger is None
                else self.selected_ledger.ledger_sha256
            ),
            "parameter_count": self.parameter_count,
            "training_flops": self.training_flops,
            "parameter_relative_difference": (self.parameter_relative_difference),
            "flop_relative_difference": self.flop_relative_difference,
            "status": self.status,
            "obligations": self.obligations,
        }

    def __post_init__(self) -> None:
        if (
            type(self.endpoint_template) is not ArmConfig
            or type(self.reference_config) is not ArmConfig
        ):
            raise ValueError("component selection configs must be exact")
        if type(self.workload) is not H6TrainingWorkloadV3:
            raise ValueError("component selection workload must be exact")
        self.endpoint_template.__post_init__()
        self.reference_config.__post_init__()
        self.workload.__post_init__()
        _require_sha256(
            self.matching_policy_sha256,
            "matching_policy_sha256",
        )
        _require_sha256(self.selection_sha256, "selection_sha256")
        expected = _component_selection_values_v3(
            endpoint_template=self.endpoint_template,
            reference_config=self.reference_config,
            workload=self.workload,
        )
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(
                    "component v3 selection is not the exact "
                    "first-lexicographic hard-eligible recomputation"
                )
        if self.selection_sha256 != _hash(
            "vfe4.h6.component-matching-selection.v3",
            self.canonical_payload(),
        ):
            raise ValueError("component v3 selection digest is stale")

    @classmethod
    def create(
        cls,
        *,
        endpoint_template: ArmConfig,
        reference_config: ArmConfig,
        workload: H6TrainingWorkloadV3,
    ) -> "H6ComponentMatchingSelectionV3":
        values = _component_selection_values_v3(
            endpoint_template=endpoint_template,
            reference_config=reference_config,
            workload=workload,
        )
        provisional = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(provisional, name, value)
        return cls(
            **values,  # type: ignore[arg-type]
            selection_sha256=_hash(
                "vfe4.h6.component-matching-selection.v3",
                provisional.canonical_payload(),
            ),
        )


@dataclass(frozen=True, slots=True)
class H6MatrixMatchingReportV3:
    row: ArmMatrixRow
    report: MatchingReport
    record_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "row_sha256": self.row.row_sha256,
            "report_sha256": self.report.report_sha256,
        }

    def __post_init__(self) -> None:
        if type(self.row) is not ArmMatrixRow or self.row not in ARM_MATRIX_ROWS:
            raise ValueError("v3 matrix report row is not frozen")
        if type(self.report) is not MatchingReport:
            raise ValueError("v3 matrix report must be exact")
        self.row.__post_init__()
        self.report.__post_init__()
        if self.record_sha256 != _hash(
            "vfe4.h6.matrix-matching-report.v3",
            self.canonical_payload(),
        ):
            raise ValueError("v3 matrix report record digest is stale")

    @classmethod
    def create(
        cls,
        *,
        row: ArmMatrixRow,
        report: MatchingReport,
    ) -> "H6MatrixMatchingReportV3":
        values = {"row": row, "report": report}
        provisional = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(provisional, name, value)
        return cls(
            **values,
            record_sha256=_hash(
                "vfe4.h6.matrix-matching-report.v3",
                provisional.canonical_payload(),
            ),
        )


@dataclass(frozen=True, slots=True)
class H6MatchingSetV3:
    """Complete regenerated v3 endpoint, ledger, and matching inventory."""

    schema_version: Literal["h6-amended-matching-set-v3"]
    git_head: str
    dirty_digest: str
    matching_policy_sha256: str
    workload: H6TrainingWorkloadV3
    parameter_relative_tolerance: Literal[0.01]
    flop_relative_tolerance: Literal[0.05]
    selection_rule: Literal["first_lexicographic_hard_eligible"]
    primary_selection: H6PrimaryMatchingSelectionV3
    component_selections: tuple[H6ComponentMatchingSelectionV3, ...]
    endpoint_configs: tuple[ArmConfig, ...]
    endpoint_ledgers: tuple[H6AnalyticalFlopLedgerV3, ...]
    matrix_reports: tuple[H6MatrixMatchingReportV3, ...]
    status: Literal["ELIGIBLE", "INCONCLUSIVE"]
    obligations: tuple[str, ...]
    matching_set_sha256: str

    @property
    def workload_sha256(self) -> str:
        return self.workload.workload_sha256

    @property
    def endpoint_config_sha256s(self) -> tuple[str, ...]:
        return tuple(config.config_sha256 for config in self.endpoint_configs)

    @property
    def ledger_sha256s(self) -> tuple[str, ...]:
        return tuple(ledger.ledger_sha256 for ledger in self.endpoint_ledgers)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "git_head": self.git_head,
            "dirty_digest": self.dirty_digest,
            "matching_policy_sha256": self.matching_policy_sha256,
            "workload_sha256": self.workload_sha256,
            "parameter_relative_tolerance": (self.parameter_relative_tolerance),
            "flop_relative_tolerance": self.flop_relative_tolerance,
            "selection_rule": self.selection_rule,
            "primary_selection_sha256": (self.primary_selection.selection_sha256),
            "component_selection_sha256s": tuple(
                selection.selection_sha256 for selection in self.component_selections
            ),
            "endpoint_config_sha256s": self.endpoint_config_sha256s,
            "ledger_sha256s": self.ledger_sha256s,
            "matrix_report_sha256s": tuple(
                record.record_sha256 for record in self.matrix_reports
            ),
            "status": self.status,
            "obligations": self.obligations,
        }

    def __post_init__(self) -> None:
        if (
            self.schema_version != "h6-amended-matching-set-v3"
            or type(self.git_head) is not str
            or len(self.git_head) != 40
            or any(character not in _LOWER_HEX for character in self.git_head)
        ):
            raise ValueError("matching set v3 source identity is invalid")
        for name in (
            "dirty_digest",
            "matching_policy_sha256",
            "matching_set_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if (
            self.matching_policy_sha256 != H6_MATCHING_POLICY_V3.policy_sha256
            or self.parameter_relative_tolerance
            != H6_MATCHING_POLICY_V3.parameter_relative_tolerance
            or self.flop_relative_tolerance
            != H6_MATCHING_POLICY_V3.flop_relative_tolerance
            or self.selection_rule != H6_MATCHING_POLICY_V3.selection_rule
        ):
            raise ValueError("matching set v3 policy/tolerances are not frozen")
        if type(self.workload) is not H6TrainingWorkloadV3:
            raise ValueError("matching set requires an exact v3 workload")
        self.workload.__post_init__()
        if type(self.primary_selection) is not H6PrimaryMatchingSelectionV3:
            raise ValueError("matching set lacks exact PRIMARY selection")
        self.primary_selection.__post_init__()
        if (
            self.primary_selection.workload_sha256 != self.workload_sha256
            or self.primary_selection.matching_policy_sha256
            != self.matching_policy_sha256
        ):
            raise ValueError("matching set PRIMARY selection preimages drifted")
        a0_ledger = analytical_training_flop_ledger_v3(
            endpoint_config=self.primary_selection.a0_config,
            workload=self.workload,
        )
        a0_parameter_count = endpoint_parameter_count_v3(
            self.primary_selection.a0_config
        )
        for candidate in self.primary_selection.candidates:
            if (
                candidate.a0_ledger_sha256 != a0_ledger.ledger_sha256
                or candidate.a0_parameter_count != a0_parameter_count
                or candidate.a0_training_flops != a0_ledger.total_arithmetic_flops
                or candidate.ledger
                != analytical_training_flop_ledger_v3(
                    endpoint_config=candidate.endpoint_config,
                    workload=self.workload,
                )
            ):
                raise ValueError("PRIMARY candidate is not the exact v3 regeneration")
        if (
            type(self.component_selections) is not tuple
            or any(
                type(selection) is not H6ComponentMatchingSelectionV3
                for selection in self.component_selections
            )
            or tuple(selection.config_id for selection in self.component_selections)
            != _COMPONENT_SELECTION_CONFIG_IDS
        ):
            raise ValueError(
                "matching set component selections are incomplete or reordered"
            )
        for selection in self.component_selections:
            selection.__post_init__()
        if (
            type(self.endpoint_configs) is not tuple
            or len(self.endpoint_configs) != len(H6_MATCHING_V3_ENDPOINT_CONFIG_IDS)
            or any(type(config) is not ArmConfig for config in self.endpoint_configs)
            or tuple(config.config_id for config in self.endpoint_configs)
            != H6_MATCHING_V3_ENDPOINT_CONFIG_IDS
        ):
            raise ValueError(
                "matching set requires the complete ordered twelve-endpoint inventory"
            )
        for config in self.endpoint_configs:
            config.__post_init__()
            endpoint_formula_profile(config.config_id).__post_init__()
        first = self.endpoint_configs[0]
        if any(
            config.vocabulary != first.vocabulary or config.horizon != first.horizon
            for config in self.endpoint_configs
        ):
            raise ValueError("matching endpoint dimensions differ")
        if any(
            selection.reference_config != self.endpoint_configs[5]
            or selection.workload != self.workload
            for selection in self.component_selections
        ):
            raise ValueError("matching component selection preimages drifted")
        selected_primary = self.primary_selection.selected_candidate
        expected_primary = (
            self.primary_selection.a5_template
            if selected_primary is None
            else selected_primary.endpoint_config
        )
        if (
            self.primary_selection.a0_config != self.endpoint_configs[0]
            or self.endpoint_configs[7] != expected_primary
            or self.endpoint_configs[9].capacity_allocation
            != expected_primary.capacity_allocation
        ):
            raise ValueError(
                "matching endpoint inventory differs from PRIMARY selection"
            )
        component_by_id = {
            selection.config_id: selection for selection in self.component_selections
        }
        for config in self.endpoint_configs:
            selection = component_by_id.get(config.config_id)
            if selection is not None and config != selection.active_config:
                raise ValueError(
                    "matching endpoint differs from first-eligible component selection"
                )
        if (
            type(self.endpoint_ledgers) is not tuple
            or len(self.endpoint_ledgers) != len(self.endpoint_configs)
            or any(
                type(ledger) is not H6AnalyticalFlopLedgerV3
                for ledger in self.endpoint_ledgers
            )
        ):
            raise ValueError("matching set requires twelve regenerated v3 ledgers")
        for config, ledger in zip(
            self.endpoint_configs,
            self.endpoint_ledgers,
            strict=True,
        ):
            ledger.__post_init__()
            expected_ledger = analytical_training_flop_ledger_v3(
                endpoint_config=config,
                workload=self.workload,
            )
            if ledger != expected_ledger:
                raise ValueError("matching ledger is not the exact v3 regeneration")
        expected_reports = _derive_matrix_reports_v3(
            endpoint_configs=self.endpoint_configs,
            endpoint_ledgers=self.endpoint_ledgers,
            workload=self.workload,
        )
        if (
            type(self.matrix_reports) is not tuple
            or tuple(record.row for record in self.matrix_reports) != ARM_MATRIX_ROWS
            or self.matrix_reports != expected_reports
        ):
            raise ValueError(
                "matching set requires the complete ordered matching report inventory"
            )
        primary_report = self.matrix_reports[0].report
        expected_status = (
            "ELIGIBLE"
            if self.primary_selection.status == "ELIGIBLE" and primary_report.eligible
            else "INCONCLUSIVE"
        )
        expected_obligations = (
            ()
            if expected_status == "ELIGIBLE"
            else tuple(
                dict.fromkeys(
                    self.primary_selection.obligations + primary_report.obligations
                )
            )
        )
        if self.status != expected_status or self.obligations != expected_obligations:
            raise ValueError("matching set v3 authorization status is stale")
        if self.matching_set_sha256 != _hash(
            "vfe4.h6.amended-matching-set.v3",
            self.canonical_payload(),
        ):
            raise ValueError("matching set v3 digest is stale")

    @classmethod
    def create(
        cls,
        *,
        git_head: str,
        dirty_digest: str,
        workload: H6TrainingWorkloadV3,
        endpoint_templates: tuple[ArmConfig, ...],
    ) -> "H6MatchingSetV3":
        if type(workload) is not H6TrainingWorkloadV3:
            raise ValueError("matching set requires an exact v3 workload")
        workload.__post_init__()
        if (
            type(endpoint_templates) is not tuple
            or len(endpoint_templates) != len(H6_MATCHING_V3_ENDPOINT_CONFIG_IDS)
            or any(type(config) is not ArmConfig for config in endpoint_templates)
            or tuple(config.config_id for config in endpoint_templates)
            != H6_MATCHING_V3_ENDPOINT_CONFIG_IDS
        ):
            raise ValueError(
                "matching template inventory must contain the complete ordered "
                "twelve endpoints"
            )
        for config in endpoint_templates:
            allocation = config.capacity_allocation
            expected = (
                (52, None, None, None)
                if config.config_id == _PRIMARY_A0_CONFIG_ID
                else (
                    (64, None, None, None)
                    if not config.latent_enabled
                    else (
                        (89, 2, 113, 6)
                        if config.prior_variant == "parent_specific_pooled_prefix"
                        else (64, 16, 64, None)
                    )
                )
            )
            if (
                allocation.emission_width,
                allocation.latent_width,
                allocation.recognition_width,
                allocation.prior_context_width,
            ) != expected:
                raise ValueError("matching template allocation inventory is not frozen")
        reference = endpoint_templates[5]
        if reference.capacity_allocation != A5_REFERENCE_ALLOCATION:
            raise ValueError("matching reference allocation is not frozen")
        primary_selection = H6PrimaryMatchingSelectionV3.create(
            a0_config=endpoint_templates[0],
            a5_template=endpoint_templates[7],
            workload=workload,
        )
        component_selections = tuple(
            H6ComponentMatchingSelectionV3.create(
                endpoint_template=template,
                reference_config=reference,
                workload=workload,
            )
            for template in endpoint_templates
            if template.config_id in _COMPONENT_SELECTION_CONFIG_IDS
        )
        selected = primary_selection.selected_candidate
        active_primary = (
            endpoint_templates[7] if selected is None else selected.endpoint_config
        )
        endpoint_configs = list(endpoint_templates)
        component_by_id = {
            selection.config_id: selection for selection in component_selections
        }
        for index, template in enumerate(endpoint_configs):
            component_selection = component_by_id.get(template.config_id)
            if component_selection is not None:
                endpoint_configs[index] = component_selection.active_config
        endpoint_configs[7] = active_primary
        endpoint_configs[9] = _config_with_allocation_v3(
            endpoint_templates[9],
            active_primary.capacity_allocation,
        )
        exact_configs = tuple(endpoint_configs)
        endpoint_ledgers = tuple(
            analytical_training_flop_ledger_v3(
                endpoint_config=config,
                workload=workload,
            )
            for config in exact_configs
        )
        matrix_reports = _derive_matrix_reports_v3(
            endpoint_configs=exact_configs,
            endpoint_ledgers=endpoint_ledgers,
            workload=workload,
        )
        primary_report = matrix_reports[0].report
        status = (
            "ELIGIBLE"
            if primary_selection.status == "ELIGIBLE" and primary_report.eligible
            else "INCONCLUSIVE"
        )
        obligations = (
            ()
            if status == "ELIGIBLE"
            else tuple(
                dict.fromkeys(
                    primary_selection.obligations + primary_report.obligations
                )
            )
        )
        values = {
            "schema_version": "h6-amended-matching-set-v3",
            "git_head": git_head,
            "dirty_digest": dirty_digest,
            "matching_policy_sha256": H6_MATCHING_POLICY_V3.policy_sha256,
            "workload": workload,
            "parameter_relative_tolerance": (
                H6_MATCHING_POLICY_V3.parameter_relative_tolerance
            ),
            "flop_relative_tolerance": (H6_MATCHING_POLICY_V3.flop_relative_tolerance),
            "selection_rule": H6_MATCHING_POLICY_V3.selection_rule,
            "primary_selection": primary_selection,
            "component_selections": component_selections,
            "endpoint_configs": exact_configs,
            "endpoint_ledgers": endpoint_ledgers,
            "matrix_reports": matrix_reports,
            "status": status,
            "obligations": obligations,
        }
        provisional = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(provisional, name, value)
        return cls(
            **values,  # type: ignore[arg-type]
            matching_set_sha256=_hash(
                "vfe4.h6.amended-matching-set.v3",
                provisional.canonical_payload(),
            ),
        )


def _config_with_allocation_v3(
    template: ArmConfig,
    allocation: CapacityAllocation,
) -> ArmConfig:
    if type(template) is not ArmConfig:
        raise ValueError("endpoint template must be an exact ArmConfig")
    if type(allocation) is not CapacityAllocation:
        raise ValueError("endpoint allocation must be exact")
    template.__post_init__()
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


def _derive_matrix_reports_v3(
    *,
    endpoint_configs: tuple[ArmConfig, ...],
    endpoint_ledgers: tuple[H6AnalyticalFlopLedgerV3, ...],
    workload: H6TrainingWorkloadV3,
) -> tuple[H6MatrixMatchingReportV3, ...]:
    if tuple(
        config.config_id for config in endpoint_configs
    ) != H6_MATCHING_V3_ENDPOINT_CONFIG_IDS or len(endpoint_ledgers) != len(
        endpoint_configs
    ):
        raise ValueError("v3 matrix derivation requires all twelve endpoints")
    config_by_id = {config.config_id: config for config in endpoint_configs}
    ledger_by_id = {
        config.config_id: ledger
        for config, ledger in zip(
            endpoint_configs,
            endpoint_ledgers,
            strict=True,
        )
    }
    records: list[H6MatrixMatchingReportV3] = []
    for row in ARM_MATRIX_ROWS:
        reference = config_by_id[row.left_config_id]
        endpoint = config_by_id[row.right_config_id]
        reference_ledger = ledger_by_id[row.left_config_id]
        endpoint_ledger = ledger_by_id[row.right_config_id]
        report = MatchingReport.from_totals(
            matching_config_sha256=H6_MATCHING_POLICY_V3.policy_sha256,
            endpoint_config_sha256=endpoint.config_sha256,
            reference_config_sha256=reference.config_sha256,
            endpoint_parameter_count=endpoint_parameter_count_v3(endpoint),
            reference_parameter_count=endpoint_parameter_count_v3(reference),
            endpoint_training_flops=(endpoint_ledger.total_arithmetic_flops),
            reference_training_flops=(reference_ledger.total_arithmetic_flops),
            parameter_relative_tolerance=(
                H6_MATCHING_POLICY_V3.parameter_relative_tolerance
            ),
            flop_relative_tolerance=(H6_MATCHING_POLICY_V3.flop_relative_tolerance),
            ownership_valid=True,
            common_schedule=True,
            optimizer_policy_match=True,
            training_flop_ledger_complete=True,
            training_flop_obligations=(),
            semantic_interventions=row.semantic_interventions,
            named_factor=row.named_factor,
            nuisance_capacity_fields=row.nuisance_capacity_fields,
            common_schedule_sha256=workload.workload_sha256,
        )
        records.append(
            H6MatrixMatchingReportV3.create(
                row=row,
                report=report,
            )
        )
    return tuple(records)


def _v3_extra_costs_per_window(
    endpoint_config: ArmConfig,
) -> dict[str, int]:
    banks = (
        arm_source_bank_count_v3(endpoint_config.arm.value)
        if endpoint_config.latent_enabled
        else 0
    )
    if banks == 0:
        return {}
    allocation = endpoint_config.capacity_allocation
    latent_width = allocation.latent_width
    recognition_width = allocation.recognition_width
    if latent_width is None or recognition_width is None:
        raise ValueError("source-bank endpoint lacks live widths")
    horizon = endpoint_config.horizon
    support_entries = horizon * (horizon + 1) // 2
    channels = int(endpoint_config.state_channel_enabled) + int(
        endpoint_config.model_channel_enabled
    )
    dimension = channels * latent_width
    terminal_components = horizon**banks
    projection_cost = 0
    if endpoint_config.mixture_mode == "moment_projection":
        projection_cost = (
            terminal_components * (2 * dimension * dimension + 5 * dimension)
            + dimension**3 // 3
        )
    return {
        "categorical_context_residual_dot": (
            banks * support_entries * (3 * recognition_width - 1)
        ),
        "categorical_lag_scalar": 4 * banks * support_entries,
        "positive_support_log_softmax": banks
        * sum(5 * length - 1 for length in range(1, horizon + 1)),
        "terminal_rank_one_component_shift": (
            2 * banks * terminal_components * latent_width
        ),
        "terminal_component_realization": (terminal_components * dimension),
        "exact_nested_source_reduction": (terminal_components * (8 + 6 * banks)),
        "moment_projection_construction": projection_cost,
    }


def analytical_training_flop_ledger_v3(
    *,
    endpoint_config: ArmConfig,
    workload: H6TrainingWorkloadV3,
) -> H6AnalyticalFlopLedgerV3:
    """Extend the exact v2 base ledger with every v3 estimator operation."""

    if type(endpoint_config) is not ArmConfig:
        raise ValueError("endpoint_config must be an exact ArmConfig")
    if type(workload) is not H6TrainingWorkloadV3:
        raise ValueError("workload must be an exact H6TrainingWorkloadV3")
    endpoint_config.__post_init__()
    workload.__post_init__()
    base = analytical_training_flop_ledger(
        endpoint_config=endpoint_config,
        workload=workload.legacy_projection(),
    )
    terms = list(base.terms)
    banks = (
        arm_source_bank_count_v3(endpoint_config.arm.value)
        if endpoint_config.latent_enabled
        else 0
    )
    allocation = endpoint_config.capacity_allocation
    source_parameters = 0
    if banks:
        if allocation.latent_width is None or allocation.recognition_width is None:
            raise ValueError("v3 source-bank endpoint lacks live widths")
        source_parameters = recognition_source_parameter_count_v3(
            bank_count=banks,
            recognition_width=allocation.recognition_width,
            latent_width=allocation.latent_width,
        )
        costs = _v3_extra_costs_per_window(endpoint_config)
        windows_per_campaign = workload.window_count * workload.full_passes
        for phase in (
            TrainingPhase.RECOGNITION_ADAMW,
            TrainingPhase.MODEL_ADAMW,
        ):
            for operation, cost in costs.items():
                terms.append(
                    FlopTerm.create(
                        phase=phase.value,
                        operation=f"v3::{operation}",
                        repetitions=windows_per_campaign,
                        arithmetic_flops_per_repetition=cost,
                        bytes_copied_per_repetition=0,
                    )
                )
        differentiable_forward = sum(costs.values())
        terms.extend(
            (
                FlopTerm.create(
                    phase=TrainingPhase.RECOGNITION_ADAMW.value,
                    operation="v3::recognition_phase_backward",
                    repetitions=windows_per_campaign,
                    arithmetic_flops_per_repetition=(2 * differentiable_forward),
                    bytes_copied_per_repetition=0,
                ),
                FlopTerm.create(
                    phase=TrainingPhase.RECOGNITION_ADAMW.value,
                    operation="v3::global_norm_clipping",
                    repetitions=workload.model_update_opportunities,
                    # The v2 base term already owns the fixed ``+3`` global
                    # norm overhead.  Only the three scalar operations for
                    # each added source parameter are incremental here.
                    arithmetic_flops_per_repetition=3 * source_parameters,
                    bytes_copied_per_repetition=0,
                ),
                FlopTerm.create(
                    phase=TrainingPhase.RECOGNITION_ADAMW.value,
                    operation="v3::adamw_update",
                    repetitions=workload.model_update_opportunities,
                    arithmetic_flops_per_repetition=(adamw_flops(source_parameters)),
                    bytes_copied_per_repetition=0,
                ),
            )
        )
    values = {
        "schema_version": "h6-analytical-flop-ledger-v3",
        "endpoint_config_sha256": endpoint_config.config_sha256,
        "matching_policy_sha256": H6_MATCHING_POLICY_V3.policy_sha256,
        "workload_sha256": workload.workload_sha256,
        "legacy_base_ledger_sha256": base.ledger_sha256,
        "source_bank_count": banks,
        "source_parameter_count": source_parameters,
        "terms": tuple(terms),
        "excluded_operations": H6_MATCHING_V3_EXCLUDED_OPERATIONS,
        "total_arithmetic_flops": sum(term.total_arithmetic_flops for term in terms),
        "total_bytes_copied": sum(term.total_bytes_copied for term in terms),
    }
    provisional = object.__new__(H6AnalyticalFlopLedgerV3)
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    return H6AnalyticalFlopLedgerV3(
        **values,  # type: ignore[arg-type]
        ledger_sha256=_hash(
            "vfe4.h6.analytical-flop-ledger.v3",
            provisional.canonical_payload(),
        ),
    )


def endpoint_parameter_count_v3(endpoint_config: ArmConfig) -> int:
    """Count one exact endpoint with the additive source-bank inventory."""

    if type(endpoint_config) is not ArmConfig:
        raise ValueError("endpoint_config must be an exact ArmConfig")
    endpoint_config.__post_init__()
    allocation = endpoint_config.capacity_allocation
    parameter_count = arm_parameter_count_v3(
        endpoint_config.arm.value,  # type: ignore[arg-type]
        vocabulary_size=endpoint_config.vocabulary.size,
        horizon=endpoint_config.horizon,
        emission_width=allocation.emission_width,
        latent_width=allocation.latent_width,
        recognition_width=allocation.recognition_width,
        recognition_family=(
            "factorized"
            if endpoint_config.recognition_family == "factorized"
            else "structured"
        ),
    )
    if endpoint_config.prior_variant == "parent_specific_pooled_prefix":
        latent_width = allocation.latent_width
        context_width = allocation.prior_context_width
        if latent_width is None or context_width is None:
            raise ValueError("parent-specific prior requires latent and context widths")
        parameter_count -= fixed_source_prior_parameter_count(
            horizon=endpoint_config.horizon,
            bank_count=2,
        )
        parameter_count += parent_specific_pooled_prefix_source_prior_parameter_count(
            vocabulary_size=endpoint_config.vocabulary.size,
            horizon=endpoint_config.horizon,
            latent_width=latent_width,
            context_width=context_width,
            gauge_anchored=True,
        )
    return parameter_count


__all__ = [
    "H6AnalyticalFlopLedgerV3",
    "H6ComponentMatchingSelectionV3",
    "H6MatrixMatchingReportV3",
    "H6MatchingSetV3",
    "H6MatchingPolicyV3",
    "H6PrimaryMatchingCandidateV3",
    "H6PrimaryMatchingSelectionV3",
    "H6TrainingWorkloadV3",
    "H6_MATCHING_V3_ENDPOINT_CONFIG_IDS",
    "H6_MATCHING_POLICY_V3",
    "H6_MATCHING_V3_ESTIMATOR_TERM_NAMES",
    "H6_MATCHING_V3_EXCLUDED_OPERATIONS",
    "PRIMARY_EMISSION_WIDTH_CANDIDATES_V3",
    "PRIMARY_JOINT_CANDIDATE_COUNT_V3",
    "analytical_training_flop_ledger_v3",
    "endpoint_parameter_count_v3",
    "primary_matching_diagnostics_v3",
]
