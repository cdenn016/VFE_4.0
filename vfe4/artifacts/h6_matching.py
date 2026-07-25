"""Immutable, manifest-linked H6 formula-matching evidence.

Schema v2 records the resolved primary policy, workload, complete ordered
324-candidate joint inventory, exact joint selection, and separate component
match disclosures.  Reading an artifact reconstructs every typed record,
reruns both selectors, checks every candidate and digest, and rebuilds exact
active endpoint ownership before deriving each row from its literal left and
right configs.  Only the eligible PRIMARY report authorizes the matching set;
each component report retains its own hard-gate status.  An empty primary
eligible set remains a serializable INCONCLUSIVE record with no fabricated
endpoint evidence.  This module never opens a corpus, trains an arm, or reads
a predictive outcome.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from vfe4.artifacts.atomic import (
    canonical_json_bytes as artifact_json_bytes,
    publish_run_directory,
)
from vfe4.artifacts.provenance import source_candidate_sha256
from vfe4.config.schema import H6PrimaryMatchingResolvedConfig
from vfe4.evaluation.compute_ledger import (
    _InferenceInclusiveComputeReport,
    _build_inference_inclusive_compute_report,
)
from vfe4.training.matching import (
    ARM_MATRIX_ROWS,
    ARM_MATRIX_SHA256,
    H6_ADAMW_POLICY,
    H6AnalyticalFlopLedger,
    H6FormulaSelection,
    H6PrimaryJointCandidate,
    H6PrimaryJointSelection,
    H6TrainingWorkload,
    analytical_training_flop_ledger,
    audit_parameter_ownership,
    select_outcome_blind_allocation,
    select_parent_specific_primary_allocation,
    stable_parameter_key,
)
from vfe4.types.h6 import (
    AdamWPolicyRecord,
    ArmConfig,
    ArmId,
    ArmMatrixRow,
    CapacityAllocation,
    FlopTerm,
    InferenceComputeRecord,
    MatchingReport,
    OptimizerBinding,
    ParameterRoleRecord,
    TrainingPhase,
    VocabularyIdentity,
    canonical_json_bytes as h6_canonical_json_bytes,
)


H6_MATCHING_ENDPOINT_LAYOUT: tuple[tuple[str, str], ...] = (
    ("h6-a0-transformer-v2", "build_a0@h6-arm-v2"),
    ("h6-a1-ordinary-latent-v1", "build_a1@h6-arm-v1"),
    ("h6-a2-generic-map-v1", "build_a2@h6-arm-v1"),
    ("h6-a3-immediate-predecessor-v1", "build_a3@h6-arm-v1"),
    ("h6-a4-state-only-v1", "build_a4@h6-arm-v1"),
    (
        "h6-a5-structured-fixed-exact-complete-latent-smoothing-v1",
        "build_a5@h6-arm-v1",
    ),
    (
        "h6-a5-factorized-fixed-exact-complete-latent-smoothing-v1",
        "build_a5@h6-arm-v1",
    ),
    (
        "h6-a5-structured-parent-specific-prefix-exact-complete-"
        "latent-smoothing-v2",
        "build_a5@h6-arm-v2",
    ),
    (
        "h6-a5-structured-fixed-projection-complete-latent-smoothing-v1",
        "build_a5@h6-arm-v1",
    ),
    (
        "h6-a5-structured-parent-specific-prefix-exact-emission-"
        "latent-smoothing-v2",
        "build_a5@h6-arm-v2",
    ),
    (
        "h6-a5-structured-fixed-exact-complete-"
        "nolatent-norecognition-v1",
        "build_a5@h6-arm-v1",
    ),
    (
        "h6-a5-structured-fixed-exact-complete-latent-filtering-v1",
        "build_a5@h6-arm-v1",
    ),
)

_REFERENCE_CONFIG_ID = H6_MATCHING_ENDPOINT_LAYOUT[5][0]
_PRIMARY_A0_CONFIG_ID = H6_MATCHING_ENDPOINT_LAYOUT[0][0]
_PRIMARY_A5_CONFIG_ID = H6_MATCHING_ENDPOINT_LAYOUT[7][0]
_OBJECTIVE_ABLATION_CONFIG_ID = H6_MATCHING_ENDPOINT_LAYOUT[9][0]
_PRIMARY_BOUND_CONFIG_IDS = frozenset(
    (
        _PRIMARY_A0_CONFIG_ID,
        _PRIMARY_A5_CONFIG_ID,
        _OBJECTIVE_ABLATION_CONFIG_ID,
    )
)
_COMPONENT_SELECTION_CONFIG_IDS = tuple(
    config_id
    for config_id, _ in H6_MATCHING_ENDPOINT_LAYOUT
    if config_id not in _PRIMARY_BOUND_CONFIG_IDS
)
_PAYLOAD_PATHS = (
    "matching/endpoints.json",
    "matching/matrix_reports.json",
    "validation/h6_matching_set.json",
)
_ALL_FILES = frozenset((*_PAYLOAD_PATHS, "manifest.sha256"))
_LOWER_HEX = frozenset("0123456789abcdef")
_MATCHING_REPORT_FIELDS = tuple(MatchingReport.__dataclass_fields__)

# Kept as a public compatibility surface.  v2 closes the former source
# blocker; an empty tuple is the only accurate current value.
H6_MATCHING_PUBLICATION_SOURCE_BLOCKERS: tuple[()] = ()


class H6MatchingPublicationBlocked(RuntimeError):
    """Retained for callers that explicitly reject legacy v1 artifacts."""


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase 64-hex SHA-256")
    return value


def _require_git_head(value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 40
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise ValueError("git_head must be a lowercase 40-hex object name")
    return value


def _owned_hash(domain: str, payload: object) -> str:
    return hashlib.sha256(
        domain.encode("ascii") + b"\x00" + h6_canonical_json_bytes(payload)
    ).hexdigest()


def _new_frozen(cls: type[object], **values: object) -> object:
    instance = object.__new__(cls)
    for name, value in values.items():
        object.__setattr__(instance, name, value)
    instance.__post_init__()  # type: ignore[attr-defined]
    return instance


def _same_json_value(left: object, right: object) -> bool:
    return artifact_json_bytes(left) == artifact_json_bytes(right)


def _exact_keys(
    payload: Mapping[str, object],
    expected: set[str],
    name: str,
) -> None:
    if set(payload) != expected:
        raise ValueError(f"{name} has missing or unknown fields")


def _string_tuple(
    value: object, name: str, *, nonempty: bool = False
) -> tuple[str, ...]:
    if (
        type(value) is not list
        or any(type(item) is not str or not item for item in value)
        or (nonempty and not value)
    ):
        qualifier = "nonempty " if nonempty else ""
        raise ValueError(
            f"{name} must be a {qualifier}JSON array of nonempty strings"
        )
    return tuple(value)


def _int_tuple(
    value: object, name: str, *, nonempty: bool = False
) -> tuple[int, ...]:
    if (
        type(value) is not list
        or any(type(item) is not int for item in value)
        or (nonempty and not value)
    ):
        qualifier = "nonempty " if nonempty else ""
        raise ValueError(f"{name} must be a {qualifier}JSON integer array")
    return tuple(value)


def _capacity_payload(value: CapacityAllocation) -> dict[str, object]:
    return {
        "emission_width": value.emission_width,
        "latent_width": value.latent_width,
        "recognition_width": value.recognition_width,
        "prior_context_width": value.prior_context_width,
        "allocation_sha256": value.allocation_sha256,
    }


def _vocabulary_payload(value: VocabularyIdentity) -> dict[str, object]:
    return {
        "vocabulary_id": value.vocabulary_id,
        "size": value.size,
        "tokenizer_spec_sha256": value.tokenizer_spec_sha256,
    }


def _arm_config_payload(value: ArmConfig) -> dict[str, object]:
    return {
        "arm": value.arm.value,
        "config_id": value.config_id,
        "vocabulary": _vocabulary_payload(value.vocabulary),
        "horizon": value.horizon,
        "latent_enabled": value.latent_enabled,
        "state_channel_enabled": value.state_channel_enabled,
        "model_channel_enabled": value.model_channel_enabled,
        "source_mode": value.source_mode,
        "map_mode": value.map_mode,
        "recognition_family": value.recognition_family,
        "recognition_conditioning": value.recognition_conditioning,
        "prior_variant": value.prior_variant,
        "mixture_mode": value.mixture_mode,
        "objective_kind": value.objective_kind,
        "capacity_allocation": _capacity_payload(
            value.capacity_allocation
        ),
        "config_sha256": value.config_sha256,
    }


def _workload_payload(value: H6TrainingWorkload) -> dict[str, object]:
    return {
        name: getattr(value, name)
        for name in value.__dataclass_fields__
    }


def _flop_term_payload(value: FlopTerm) -> dict[str, object]:
    return {
        name: getattr(value, name)
        for name in value.__dataclass_fields__
    }


def _ledger_payload(value: H6AnalyticalFlopLedger) -> dict[str, object]:
    return {
        "schema_version": value.schema_version,
        "endpoint_config_sha256": value.endpoint_config_sha256,
        "endpoint_profile_sha256": value.endpoint_profile_sha256,
        "allocation_sha256": value.allocation_sha256,
        "workload_sha256": value.workload_sha256,
        "terms": tuple(_flop_term_payload(term) for term in value.terms),
        "total_arithmetic_flops": value.total_arithmetic_flops,
        "total_bytes_copied": value.total_bytes_copied,
        "status": value.status,
        "obligations": value.obligations,
        "ledger_sha256": value.ledger_sha256,
    }


def _selection_payload(value: H6FormulaSelection) -> dict[str, object]:
    return {
        "config_id": value.config_id,
        "endpoint_template": _arm_config_payload(value.endpoint_template),
        "reference_config": _arm_config_payload(value.reference_config),
        "endpoint_profile_sha256": value.endpoint_profile_sha256,
        "reference_profile_sha256": value.reference_profile_sha256,
        "workload": _workload_payload(value.workload),
        "policy_sha256": value.policy_sha256,
        "candidate_count_evaluated": value.candidate_count_evaluated,
        "selected_endpoint_config": (
            None
            if value.selected_endpoint_config is None
            else _arm_config_payload(value.selected_endpoint_config)
        ),
        "selected_allocation": (
            None
            if value.selected_allocation is None
            else _capacity_payload(value.selected_allocation)
        ),
        "parameter_count": value.parameter_count,
        "training_flops": value.training_flops,
        "parameter_relative_difference": (
            value.parameter_relative_difference
        ),
        "flop_relative_difference": value.flop_relative_difference,
        "ledger": (
            None if value.ledger is None else _ledger_payload(value.ledger)
        ),
        "reference_ledger": _ledger_payload(value.reference_ledger),
        "status": value.status,
        "obligations": value.obligations,
        "selection_sha256": value.selection_sha256,
    }


def _primary_matching_config_payload(
    value: H6PrimaryMatchingResolvedConfig,
) -> dict[str, object]:
    return {
        "schema_version": value.schema_version,
        "operation": value.operation,
        "a0_config": _arm_config_payload(value.a0_config),
        "a5_template": _arm_config_payload(value.a5_template),
        "latent_width_candidates": value.latent_width_candidates,
        "prior_context_width_candidates": (
            value.prior_context_width_candidates
        ),
        "emission_width_candidates": value.emission_width_candidates,
        "recognition_width_candidates": (
            value.recognition_width_candidates
        ),
        "parameter_relative_tolerance": (
            value.parameter_relative_tolerance
        ),
        "flop_relative_tolerance": value.flop_relative_tolerance,
        "matching_policy_sha256": value.matching_policy_sha256,
        "canonical_json": value.canonical_json,
        "config_sha256": value.config_sha256,
    }


def _primary_candidate_payload(
    value: H6PrimaryJointCandidate,
) -> dict[str, object]:
    return {
        name: getattr(value, name)
        for name in value.__dataclass_fields__
    }


def _primary_selection_payload(
    value: H6PrimaryJointSelection,
) -> dict[str, object]:
    return {
        "schema_version": value.schema_version,
        "matching_config_sha256": value.matching_config_sha256,
        "matching_policy_sha256": value.matching_policy_sha256,
        "a0_config_sha256": value.a0_config_sha256,
        "a5_template_config_sha256": value.a5_template_config_sha256,
        "workload_sha256": value.workload_sha256,
        "candidates": tuple(
            _primary_candidate_payload(candidate)
            for candidate in value.candidates
        ),
        "candidate_inventory_sha256": (
            value.candidate_inventory_sha256
        ),
        "status": value.status,
        "selected_candidate_sha256": value.selected_candidate_sha256,
        "obligations": value.obligations,
        "selection_sha256": value.selection_sha256,
    }


def _role_payload(value: ParameterRoleRecord) -> dict[str, object]:
    return {
        name: getattr(value, name)
        for name in value.__dataclass_fields__
    }


def _binding_payload(value: OptimizerBinding) -> dict[str, object]:
    return {
        name: getattr(value, name)
        for name in value.__dataclass_fields__
    }


def _optimizer_policy_payload(
    value: AdamWPolicyRecord,
) -> dict[str, object]:
    return {
        name: getattr(value, name)
        for name in value.__dataclass_fields__
    }


def _row_payload(value: ArmMatrixRow) -> dict[str, object]:
    return {
        name: getattr(value, name)
        for name in value.__dataclass_fields__
    }


def _matching_report_payload(value: MatchingReport) -> dict[str, object]:
    return {
        name: getattr(value, name)
        for name in _MATCHING_REPORT_FIELDS
    }


@dataclass(frozen=True, slots=True, init=False)
class H6MatchingOwnershipRecord:
    """One selected arm's stable parameter and optimizer preimages."""

    config: ArmConfig
    parameter_roles: tuple[ParameterRoleRecord, ...]
    optimizer_bindings: tuple[OptimizerBinding, ...]
    inventory_sha256: str

    def __post_init__(self) -> None:
        if type(self.config) is not ArmConfig:
            raise ValueError("ownership config must be an exact ArmConfig")
        self.config.__post_init__()
        if (
            type(self.parameter_roles) is not tuple
            or not self.parameter_roles
            or any(
                type(record) is not ParameterRoleRecord
                for record in self.parameter_roles
            )
        ):
            raise ValueError(
                "ownership requires a nonempty exact parameter-role inventory"
            )
        if (
            type(self.optimizer_bindings) is not tuple
            or not self.optimizer_bindings
            or any(
                type(binding) is not OptimizerBinding
                for binding in self.optimizer_bindings
            )
        ):
            raise ValueError(
                "ownership requires a nonempty exact optimizer inventory"
            )
        for record in self.parameter_roles:
            record.__post_init__()
            if record.parameter_key != stable_parameter_key(
                qualified_name=record.qualified_name,
                phase=record.phase,
            ):
                raise ValueError(
                    "parameter role does not use its stable name/phase key"
                )
        for binding in self.optimizer_bindings:
            binding.__post_init__()
        role_names = tuple(
            record.qualified_name for record in self.parameter_roles
        )
        role_keys = tuple(
            record.parameter_key for record in self.parameter_roles
        )
        if (
            len(set(role_names)) != len(role_names)
            or len(set(role_keys)) != len(role_keys)
        ):
            raise ValueError("parameter-role inventory is not one-to-one")
        binding_keys = tuple(
            key
            for binding in self.optimizer_bindings
            for key in binding.parameter_keys
        )
        if (
            len(set(binding_keys)) != len(binding_keys)
            or set(binding_keys) != set(role_keys)
        ):
            raise ValueError(
                "optimizer bindings must cover every role exactly once"
            )
        phase_by_key = {
            key: binding.phase
            for binding in self.optimizer_bindings
            for key in binding.parameter_keys
        }
        if any(
            phase_by_key[record.parameter_key] != record.phase
            for record in self.parameter_roles
        ):
            raise ValueError("optimizer and parameter-role phases disagree")
        if any(
            binding.optimizer_policy_sha256
            != H6_ADAMW_POLICY.optimizer_policy_sha256
            for binding in self.optimizer_bindings
        ):
            raise ValueError(
                "optimizer binding differs from the exact AdamW policy"
            )
        model_phase = (
            TrainingPhase.MODEL_ADAMW.value
            if self.config.latent_enabled
            else TrainingPhase.MODEL_CE_ADAMW.value
        )
        expected_phases = (
            {
                model_phase,
                TrainingPhase.RECOGNITION_ADAMW.value,
            }
            if self.config.latent_enabled
            else {model_phase}
        )
        if {
            binding.phase for binding in self.optimizer_bindings
        } != expected_phases:
            raise ValueError("ownership has a missing or extra optimizer phase")
        for record in self.parameter_roles:
            recognition = record.qualified_name.startswith(
                "recognition_store."
            )
            if recognition != (
                record.phase == TrainingPhase.RECOGNITION_ADAMW.value
            ):
                raise ValueError(
                    "parameter owner prefix and optimizer phase disagree"
                )
            if not recognition and record.phase != model_phase:
                raise ValueError("model parameter has the wrong optimizer phase")
        expected = _owned_hash(
            "vfe4.h6.matching-ownership.v2",
            {
                "config_sha256": self.config.config_sha256,
                "parameter_role_sha256s": tuple(
                    record.record_sha256
                    for record in self.parameter_roles
                ),
                "optimizer_binding_sha256s": tuple(
                    binding.binding_sha256
                    for binding in self.optimizer_bindings
                ),
                "optimizer_policy_sha256": (
                    H6_ADAMW_POLICY.optimizer_policy_sha256
                ),
            },
        )
        if self.inventory_sha256 != expected:
            raise ValueError(
                "inventory_sha256 does not bind the ownership preimages"
            )

    @property
    def parameter_count(self) -> int:
        return sum(record.scalar_count for record in self.parameter_roles)

    @classmethod
    def create(
        cls,
        *,
        config: ArmConfig,
        parameter_roles: tuple[ParameterRoleRecord, ...],
        optimizer_bindings: tuple[OptimizerBinding, ...],
    ) -> "H6MatchingOwnershipRecord":
        values = {
            "config": config,
            "parameter_roles": tuple(parameter_roles),
            "optimizer_bindings": tuple(optimizer_bindings),
        }
        payload = {
            "config_sha256": config.config_sha256,
            "parameter_role_sha256s": tuple(
                record.record_sha256 for record in parameter_roles
            ),
            "optimizer_binding_sha256s": tuple(
                binding.binding_sha256 for binding in optimizer_bindings
            ),
            "optimizer_policy_sha256": (
                H6_ADAMW_POLICY.optimizer_policy_sha256
            ),
        }
        return _new_frozen(
            cls,
            **values,
            inventory_sha256=_owned_hash(
                "vfe4.h6.matching-ownership.v2", payload
            ),
        )  # type: ignore[return-value]

    def canonical_payload(self) -> dict[str, object]:
        return {
            "config": _arm_config_payload(self.config),
            "parameter_roles": tuple(
                _role_payload(record) for record in self.parameter_roles
            ),
            "optimizer_bindings": tuple(
                _binding_payload(binding)
                for binding in self.optimizer_bindings
            ),
            "inventory_sha256": self.inventory_sha256,
        }


@dataclass(frozen=True, slots=True, init=False)
class H6MatrixMatchingReportRecord:
    """One arithmetic report plus fail-closed matched-claim authorization."""

    row: ArmMatrixRow
    report: MatchingReport
    matched_claim_authorized: bool
    selection_obligations: tuple[str, ...]
    record_sha256: str

    def __post_init__(self) -> None:
        if type(self.row) is not ArmMatrixRow:
            raise ValueError("matrix row must be an exact ArmMatrixRow")
        if type(self.report) is not MatchingReport:
            raise ValueError("matrix report must be an exact MatchingReport")
        self.row.__post_init__()
        self.report.__post_init__()
        if type(self.matched_claim_authorized) is not bool:
            raise ValueError("matched_claim_authorized must be a bool")
        if (
            type(self.selection_obligations) is not tuple
            or any(
                type(obligation) is not str or not obligation
                for obligation in self.selection_obligations
            )
            or len(set(self.selection_obligations))
            != len(self.selection_obligations)
        ):
            raise ValueError(
                "selection obligations must be unique nonempty strings"
            )
        expected_authorization = (
            self.row.row_id != "OBJECTIVE"
            and self.report.eligible
            and not self.selection_obligations
        )
        if self.matched_claim_authorized is not expected_authorization:
            raise ValueError(
                "matched-claim authorization does not follow report and "
                "selection evidence"
            )
        expected = _owned_hash(
            "vfe4.h6.matrix-matching-report.v2",
            {
                "row_sha256": self.row.row_sha256,
                "report_sha256": self.report.report_sha256,
                "matched_claim_authorized": (
                    self.matched_claim_authorized
                ),
                "selection_obligations": self.selection_obligations,
            },
        )
        if self.record_sha256 != expected:
            raise ValueError("matrix report record digest is stale")

    @classmethod
    def create(
        cls,
        *,
        row: ArmMatrixRow,
        report: MatchingReport,
        selection_obligations: tuple[str, ...],
    ) -> "H6MatrixMatchingReportRecord":
        obligations = tuple(dict.fromkeys(selection_obligations))
        authorized = (
            row.row_id != "OBJECTIVE"
            and report.eligible
            and not obligations
        )
        return _new_frozen(
            cls,
            row=row,
            report=report,
            matched_claim_authorized=authorized,
            selection_obligations=obligations,
            record_sha256=_owned_hash(
                "vfe4.h6.matrix-matching-report.v2",
                {
                    "row_sha256": row.row_sha256,
                    "report_sha256": report.report_sha256,
                    "matched_claim_authorized": authorized,
                    "selection_obligations": obligations,
                },
            ),
        )  # type: ignore[return-value]

    def canonical_payload(self) -> dict[str, object]:
        return {
            "row": _row_payload(self.row),
            "report": _matching_report_payload(self.report),
            "matched_claim_authorized": self.matched_claim_authorized,
            "selection_obligations": self.selection_obligations,
            "record_sha256": self.record_sha256,
        }


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
            "prior_context_width",
        )
        if getattr(endpoint, name) != getattr(reference, name)
    )


def _config_with_allocation(
    template: ArmConfig,
    allocation: CapacityAllocation,
    *,
    config_id: str | None = None,
    objective_kind: str | None = None,
) -> ArmConfig:
    return ArmConfig.create(
        arm=template.arm,
        config_id=template.config_id if config_id is None else config_id,
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
        objective_kind=(
            template.objective_kind
            if objective_kind is None
            else objective_kind
        ),  # type: ignore[arg-type]
        capacity_allocation=allocation,
    )


def _selected_primary_configs(
    *,
    matching_config: H6PrimaryMatchingResolvedConfig,
    primary_selection: H6PrimaryJointSelection,
) -> dict[str, ArmConfig] | None:
    if primary_selection.status != "ELIGIBLE" or primary_selection.obligations:
        return None
    selected = primary_selection.selected_candidate
    if selected is None:
        return None
    allocation = CapacityAllocation.create(
        emission_width=selected.emission_width,
        latent_width=selected.latent_width,
        recognition_width=selected.recognition_width,
        prior_context_width=selected.prior_context_width,
    )
    primary_a5 = _config_with_allocation(
        matching_config.a5_template,
        allocation,
    )
    objective_ablation = _config_with_allocation(
        matching_config.a5_template,
        allocation,
        config_id=_OBJECTIVE_ABLATION_CONFIG_ID,
        objective_kind="emission_only_ablation_non_elbo",
    )
    return {
        _PRIMARY_A0_CONFIG_ID: matching_config.a0_config,
        _PRIMARY_A5_CONFIG_ID: primary_a5,
        _OBJECTIVE_ABLATION_CONFIG_ID: objective_ablation,
    }


def _validate_primary_prediction_endpoint_configs(
    *,
    matching_config: H6PrimaryMatchingResolvedConfig,
    primary_selection: H6PrimaryJointSelection,
    configs: Mapping[str, ArmConfig],
) -> None:
    """Require raw PRIMARY/OBJECTIVE configs to come from one selected A5."""

    selected = primary_selection.selected_candidate
    if (
        primary_selection.status != "ELIGIBLE"
        or primary_selection.obligations
        or selected is None
    ):
        raise ValueError(
            "prediction endpoint ownership requires an eligible PRIMARY selection"
        )
    expected_allocation = CapacityAllocation.create(
        emission_width=selected.emission_width,
        latent_width=selected.latent_width,
        recognition_width=selected.recognition_width,
        prior_context_width=selected.prior_context_width,
    )
    a0 = configs.get(_PRIMARY_A0_CONFIG_ID)
    a5 = configs.get(_PRIMARY_A5_CONFIG_ID)
    objective = configs.get(_OBJECTIVE_ABLATION_CONFIG_ID)
    if (
        a0 != matching_config.a0_config
        or a5 is None
        or objective is None
        or a5.capacity_allocation != expected_allocation
        or objective.capacity_allocation != expected_allocation
        or a5.prior_variant != "parent_specific_pooled_prefix"
        or objective.prior_variant != "parent_specific_pooled_prefix"
        or a5.objective_kind != "complete_elbo"
        or objective.objective_kind != "emission_only_ablation_non_elbo"
    ):
        raise ValueError(
            "PRIMARY/OBJECTIVE configs are not the exact jointly selected "
            "prediction endpoints"
        )
    semantic_differences = tuple(
        name
        for name, value in a5.semantic_payload().items()
        if objective.semantic_payload()[name] != value
    )
    if semantic_differences != ("objective_kind",):
        raise ValueError(
            "OBJECTIVE endpoint must differ from selected A5 only by "
            "objective_kind"
        )


def _validate_primary_preimages(
    *,
    matching_config: H6PrimaryMatchingResolvedConfig,
    workload: H6TrainingWorkload,
    primary_selection: H6PrimaryJointSelection,
) -> None:
    if type(matching_config) is not H6PrimaryMatchingResolvedConfig:
        raise ValueError(
            "matching artifact requires an exact resolved primary config"
        )
    if type(workload) is not H6TrainingWorkload:
        raise ValueError("matching artifact requires an exact workload")
    if type(primary_selection) is not H6PrimaryJointSelection:
        raise ValueError(
            "matching artifact requires an exact primary joint selection"
        )
    matching_config.__post_init__()
    workload.__post_init__()
    primary_selection.__post_init__()
    for candidate in primary_selection.candidates:
        if type(candidate) is not H6PrimaryJointCandidate:
            raise ValueError(
                "primary selection contains a non-candidate inventory row"
            )
        candidate.__post_init__()
    if (
        primary_selection.matching_config_sha256
        != matching_config.config_sha256
        or primary_selection.matching_policy_sha256
        != matching_config.matching_policy_sha256
        or primary_selection.a0_config_sha256
        != matching_config.a0_config.config_sha256
        or primary_selection.a5_template_config_sha256
        != matching_config.a5_template.config_sha256
        or primary_selection.workload_sha256 != workload.workload_sha256
    ):
        raise ValueError(
            "primary selection does not bind the resolved config, policy, "
            "endpoints, and workload"
        )
    recomputed = select_parent_specific_primary_allocation(
        matching_config=matching_config,
        a0_config=matching_config.a0_config,
        a5_template=matching_config.a5_template,
        workload=workload,
    )
    if recomputed != primary_selection:
        raise ValueError(
            "serialized primary joint selection is not the exact "
            "324-candidate recomputation"
        )


def _validate_component_selections(
    *,
    component_selections: tuple[H6FormulaSelection, ...],
    workload: H6TrainingWorkload,
) -> None:
    if (
        type(component_selections) is not tuple
        or any(
            type(selection) is not H6FormulaSelection
            for selection in component_selections
        )
        or tuple(
            selection.config_id for selection in component_selections
        )
        != _COMPONENT_SELECTION_CONFIG_IDS
    ):
        raise ValueError(
            "component formula selections are missing, extra, or reordered; "
            "legacy PRIMARY evidence is forbidden"
        )
    for selection in component_selections:
        selection.__post_init__()
        if (
            selection.config_id in _PRIMARY_BOUND_CONFIG_IDS
            or selection.workload != workload
        ):
            raise ValueError(
                "component selections cannot supply PRIMARY or OBJECTIVE "
                "evidence and must bind the primary workload"
            )
        recomputed = select_outcome_blind_allocation(
            endpoint_template=selection.endpoint_template,
            reference_config=selection.reference_config,
            workload=selection.workload,
        )
        if recomputed != selection:
            raise ValueError(
                "serialized component formula selection is not the exact "
                "outcome-blind recomputation"
            )


def _row_selection_obligations(
    *,
    row: ArmMatrixRow,
    component_selections: tuple[H6FormulaSelection, ...],
) -> tuple[str, ...]:
    relevant_ids = frozenset((row.left_config_id, row.right_config_id))
    return tuple(
        dict.fromkeys(
            obligation
            for selection in component_selections
            if selection.config_id in relevant_ids
            and selection.status != "ELIGIBLE"
            for obligation in selection.obligations
        )
    )


def _component_disclosure_config(
    selection: H6FormulaSelection,
) -> ArmConfig:
    """Return the exact active endpoint disclosed by one component search.

    A hard-eligible search binds its selected nuisance allocation.  An
    unmatched search retains the predeclared active template solely for
    parameter/FLOP disclosure; it cannot become matched evidence.
    """

    selection.__post_init__()
    if selection.status == "ELIGIBLE":
        selected = selection.selected_endpoint_config
        if selected is None or selection.ledger is None:
            raise ValueError("eligible component selection lacks exact evidence")
        return selected
    if selection.status != "INCONCLUSIVE":
        raise ValueError("component selection has an unsupported status")
    return selection.endpoint_template


def _resolved_endpoint_configs(
    *,
    matching_config: H6PrimaryMatchingResolvedConfig,
    primary_selection: H6PrimaryJointSelection,
    component_selections: tuple[H6FormulaSelection, ...],
) -> dict[str, ArmConfig] | None:
    """Resolve PRIMARY endpoints and exact component disclosure endpoints."""

    primary_configs = _selected_primary_configs(
        matching_config=matching_config,
        primary_selection=primary_selection,
    )
    if primary_configs is None:
        return None
    configs = dict(primary_configs)
    for selection in component_selections:
        selected = _component_disclosure_config(selection)
        if selected.config_id != selection.config_id:
            raise ValueError(
                "component disclosure config identity differs from its endpoint"
            )
        configs[selected.config_id] = selected
    expected_ids = tuple(
        config_id for config_id, _ in H6_MATCHING_ENDPOINT_LAYOUT
    )
    if len(configs) != len(expected_ids) or set(configs) != set(expected_ids):
        raise ValueError("resolved endpoint inventory is incomplete")
    ordered = {
        config_id: configs[config_id]
        for config_id in expected_ids
    }
    _validate_primary_prediction_endpoint_configs(
        matching_config=matching_config,
        primary_selection=primary_selection,
        configs=ordered,
    )
    return ordered


def _resolved_endpoint_ledgers(
    *,
    configs: Mapping[str, ArmConfig],
    workload: H6TrainingWorkload,
    primary_selection: H6PrimaryJointSelection,
    component_selections: tuple[H6FormulaSelection, ...],
) -> dict[str, H6AnalyticalFlopLedger]:
    component_by_id = {
        selection.config_id: selection
        for selection in component_selections
    }
    ledgers: dict[str, H6AnalyticalFlopLedger] = {}
    for config_id, config in configs.items():
        if config_id in _PRIMARY_BOUND_CONFIG_IDS:
            ledger = analytical_training_flop_ledger(
                endpoint_config=config,
                workload=workload,
            )
        else:
            selection = component_by_id[config_id]
            selected_ledger = selection.ledger
            ledger = (
                selected_ledger
                if selection.status == "ELIGIBLE"
                and selected_ledger is not None
                else analytical_training_flop_ledger(
                    endpoint_config=config,
                    workload=workload,
                )
            )
        if (
            ledger.endpoint_config_sha256 != config.config_sha256
            or ledger.workload_sha256 != workload.workload_sha256
        ):
            raise ValueError(
                "endpoint ledger does not bind its literal config and workload"
            )
        ledgers[config_id] = ledger
    selected = primary_selection.selected_candidate
    if selected is None:
        raise ValueError("primary endpoint ledgers require a selected candidate")
    if (
        ledgers[_PRIMARY_A0_CONFIG_ID].ledger_sha256
        != selected.a0_ledger_sha256
        or ledgers[_PRIMARY_A5_CONFIG_ID].ledger_sha256
        != selected.a5_ledger_sha256
    ):
        raise ValueError(
            "selected primary candidate ledger identities are stale"
        )
    return ledgers


def _validate_matrix_layout_identities() -> None:
    factory_by_config_id = dict(H6_MATCHING_ENDPOINT_LAYOUT)
    for row in ARM_MATRIX_ROWS:
        if (
            factory_by_config_id.get(row.left_config_id)
            != row.left_factory_id
            or factory_by_config_id.get(row.right_config_id)
            != row.right_factory_id
        ):
            raise ValueError(
                f"{row.row_id} matrix row config/factory identity is stale"
            )


def _validate_matrix_endpoint_identities(
    *,
    configs: Mapping[str, ArmConfig],
) -> None:
    _validate_matrix_layout_identities()
    factory_by_config_id = dict(H6_MATCHING_ENDPOINT_LAYOUT)
    if tuple(configs) != tuple(factory_by_config_id):
        raise ValueError(
            "matrix endpoint configs are missing, extra, or reordered"
        )
    for config_id, config in configs.items():
        if config.config_id != config_id:
            raise ValueError(
                "matrix endpoint config does not match its literal identity"
            )


def _derive_matrix_reports(
    *,
    matching_config: H6PrimaryMatchingResolvedConfig,
    workload: H6TrainingWorkload,
    primary_selection: H6PrimaryJointSelection,
    component_selections: tuple[H6FormulaSelection, ...],
    ownership_inventories: tuple[H6MatchingOwnershipRecord, ...],
) -> tuple[tuple[ArmMatrixRow, MatchingReport], ...]:
    """Derive each report from the literal row's left and right endpoints."""

    _validate_primary_preimages(
        matching_config=matching_config,
        workload=workload,
        primary_selection=primary_selection,
    )
    _validate_component_selections(
        component_selections=component_selections,
        workload=workload,
    )
    _validate_matrix_layout_identities()
    configs = _resolved_endpoint_configs(
        matching_config=matching_config,
        primary_selection=primary_selection,
        component_selections=component_selections,
    )
    if configs is None:
        if ownership_inventories:
            raise ValueError(
                "inconclusive endpoint search cannot retain ownership evidence"
            )
        return ()
    _validate_matrix_endpoint_identities(configs=configs)
    ledgers = _resolved_endpoint_ledgers(
        configs=configs,
        workload=workload,
        primary_selection=primary_selection,
        component_selections=component_selections,
    )
    if (
        type(ownership_inventories) is not tuple
        or tuple(
            inventory.config for inventory in ownership_inventories
        )
        != tuple(configs.values())
    ):
        raise ValueError(
            "ownership inventories do not match the literal endpoint order"
        )
    ownership_by_sha256 = {
        inventory.config.config_sha256: inventory
        for inventory in ownership_inventories
    }
    results: list[tuple[ArmMatrixRow, MatchingReport]] = []
    for row in ARM_MATRIX_ROWS:
        reference_config = configs[row.left_config_id]
        endpoint_config = configs[row.right_config_id]
        reference_ledger = ledgers[row.left_config_id]
        endpoint_ledger = ledgers[row.right_config_id]
        reference_inventory = ownership_by_sha256[
            reference_config.config_sha256
        ]
        endpoint_inventory = ownership_by_sha256[
            endpoint_config.config_sha256
        ]
        report = MatchingReport.from_totals(
            matching_config_sha256=matching_config.config_sha256,
            endpoint_config_sha256=endpoint_config.config_sha256,
            reference_config_sha256=reference_config.config_sha256,
            endpoint_parameter_count=endpoint_inventory.parameter_count,
            reference_parameter_count=reference_inventory.parameter_count,
            endpoint_training_flops=(
                endpoint_ledger.total_arithmetic_flops
            ),
            reference_training_flops=(
                reference_ledger.total_arithmetic_flops
            ),
            parameter_relative_tolerance=(
                matching_config.parameter_relative_tolerance
            ),
            flop_relative_tolerance=matching_config.flop_relative_tolerance,
            ownership_valid=True,
            common_schedule=True,
            optimizer_policy_match=True,
            training_flop_ledger_complete=(
                endpoint_ledger.status == "COMPLETE"
                and reference_ledger.status == "COMPLETE"
            ),
            training_flop_obligations=(
                endpoint_ledger.obligations
                + reference_ledger.obligations
            ),
            semantic_interventions=row.semantic_interventions,
            named_factor=row.named_factor,
            nuisance_capacity_fields=_capacity_differences(
                endpoint_config.capacity_allocation,
                reference_config.capacity_allocation,
            ),
            common_schedule_sha256=(
                workload.matching_schedule_policy_sha256
            ),
        )
        if (
            report.reference_config_sha256
            != configs[row.left_config_id].config_sha256
            or report.endpoint_config_sha256
            != configs[row.right_config_id].config_sha256
        ):
            raise ValueError(
                f"{row.row_id} report is not bound left-to-right"
            )
        results.append((row, report))
    return tuple(results)


def derive_h6_matrix_reports(
    *,
    matching_config: H6PrimaryMatchingResolvedConfig,
    workload: H6TrainingWorkload,
    primary_selection: H6PrimaryJointSelection,
    component_selections: tuple[H6FormulaSelection, ...],
    ownership_inventories: tuple[H6MatchingOwnershipRecord, ...],
) -> tuple[tuple[ArmMatrixRow, MatchingReport], ...]:
    """Public pure helper used by click-to-run evidence assembly."""

    return _derive_matrix_reports(
        matching_config=matching_config,
        workload=workload,
        primary_selection=primary_selection,
        component_selections=component_selections,
        ownership_inventories=ownership_inventories,
    )


@dataclass(frozen=True, slots=True, init=False)
class H6MatchingSetRecord:
    """Exact H6 matching evidence with PRIMARY-only set authorization.

    Component selections and matrix rows remain exact disclosures.  Each
    wrapper combines its arithmetic ``MatchingReport`` with the relevant
    selection obligations before authorizing a matched component conclusion;
    component rows never promote or demote PRIMARY.
    """

    schema_version: Literal["h6-matching-set-v2"]
    git_head: str
    dirty_digest: str
    source_sha256: str
    primary_matching_config: H6PrimaryMatchingResolvedConfig
    workload: H6TrainingWorkload
    primary_selection: H6PrimaryJointSelection
    optimizer_policy: AdamWPolicyRecord
    component_selections: tuple[H6FormulaSelection, ...]
    ownership_inventories: tuple[H6MatchingOwnershipRecord, ...]
    matrix_reports: tuple[H6MatrixMatchingReportRecord, ...]
    arm_matrix_sha256: str
    status: Literal["ELIGIBLE", "INCONCLUSIVE"]
    obligations: tuple[str, ...]
    matching_set_sha256: str

    @property
    def matching_config_sha256(self) -> str:
        return self.primary_matching_config.config_sha256

    @property
    def matching_policy_sha256(self) -> str:
        return self.primary_matching_config.matching_policy_sha256

    @property
    def authorizing_matching_report_ids(self) -> tuple[str, ...]:
        """Return the report IDs allowed to authorize H6 matching readiness."""

        return tuple(
            record.row.row_id
            for record in self.matrix_reports
            if record.row.row_id == "PRIMARY"
            and record.matched_claim_authorized
        )

    @property
    def eligible_component_report_ids(self) -> tuple[str, ...]:
        """Return independently eligible non-PRIMARY matching disclosures."""

        return tuple(
            record.row.row_id
            for record in self.matrix_reports
            if record.row.row_id not in ("PRIMARY", "OBJECTIVE")
            and record.matched_claim_authorized
        )

    @property
    def unmatched_report_ids(self) -> tuple[str, ...]:
        """Return rows that cannot authorize a matched conclusion."""

        return tuple(
            record.row.row_id
            for record in self.matrix_reports
            if not record.matched_claim_authorized
        )

    def __post_init__(self) -> None:
        if self.schema_version != "h6-matching-set-v2":
            raise ValueError("unsupported H6 matching-set schema")
        _require_git_head(self.git_head)
        _require_sha256(self.dirty_digest, "dirty_digest")
        _require_sha256(self.source_sha256, "source_sha256")
        _require_sha256(
            self.matching_config_sha256, "matching_config_sha256"
        )
        if self.source_sha256 != source_candidate_sha256(
            git_head_value=self.git_head,
            dirty_digest_value=self.dirty_digest,
        ):
            raise ValueError("source SHA-256 does not bind source preimages")
        _validate_primary_preimages(
            matching_config=self.primary_matching_config,
            workload=self.workload,
            primary_selection=self.primary_selection,
        )
        _validate_component_selections(
            component_selections=self.component_selections,
            workload=self.workload,
        )
        if (
            type(self.optimizer_policy) is not AdamWPolicyRecord
            or self.optimizer_policy != H6_ADAMW_POLICY
        ):
            raise ValueError("matching set requires the exact AdamW policy")
        self.optimizer_policy.__post_init__()
        if (
            type(self.ownership_inventories) is not tuple
            or any(
                type(inventory) is not H6MatchingOwnershipRecord
                for inventory in self.ownership_inventories
            )
        ):
            raise ValueError("ownership inventories must be exact records")
        for inventory in self.ownership_inventories:
            inventory.__post_init__()
        selected_hashes = tuple(
            inventory.config.config_sha256
            for inventory in self.ownership_inventories
        )
        if len(set(selected_hashes)) != len(selected_hashes):
            raise ValueError("selected endpoint identities must be unique")
        inventory_by_config_id = {
            inventory.config.config_id: inventory
            for inventory in self.ownership_inventories
        }
        selected_primary = self.primary_selection.selected_candidate
        if selected_primary is not None:
            for selection in self.component_selections:
                disclosure_config = _component_disclosure_config(selection)
                inventory = inventory_by_config_id.get(selection.config_id)
                if (
                    inventory is None
                    or inventory.config != disclosure_config
                    or (
                        selection.status == "ELIGIBLE"
                        and inventory.parameter_count
                        != selection.parameter_count
                    )
                ):
                    raise ValueError(
                        "component parameter-role inventory differs from its "
                        "exact disclosure endpoint"
                    )
            a0_inventory = inventory_by_config_id.get(
                _PRIMARY_A0_CONFIG_ID
            )
            a5_inventory = inventory_by_config_id.get(
                _PRIMARY_A5_CONFIG_ID
            )
            if (
                a0_inventory is None
                or a5_inventory is None
                or a0_inventory.parameter_count
                != selected_primary.a0_parameter_count
                or a5_inventory.parameter_count
                != selected_primary.a5_parameter_count
            ):
                raise ValueError(
                    "primary endpoint ownership counts differ from the "
                    "selected joint candidate"
                )
        for inventory in self.ownership_inventories:
            from vfe4.training.arms import build_arm

            rebuilt_arm = build_arm(inventory.config.arm, inventory.config)
            audit_parameter_ownership(rebuilt_arm)
            if (
                rebuilt_arm.config != inventory.config
                or rebuilt_arm.parameter_roles != inventory.parameter_roles
                or rebuilt_arm.optimizer_bindings
                != inventory.optimizer_bindings
            ):
                raise ValueError(
                    "serialized ownership inventory differs from the "
                    "independently rebuilt arm"
                )
        if self.arm_matrix_sha256 != ARM_MATRIX_SHA256:
            raise ValueError("matching set does not bind the exact arm matrix")
        if (
            type(self.matrix_reports) is not tuple
            or any(
                type(record) is not H6MatrixMatchingReportRecord
                for record in self.matrix_reports
            )
        ):
            raise ValueError("matrix reports must be exact records")
        for record in self.matrix_reports:
            record.__post_init__()
        derived = _derive_matrix_reports(
            matching_config=self.primary_matching_config,
            workload=self.workload,
            primary_selection=self.primary_selection,
            component_selections=self.component_selections,
            ownership_inventories=self.ownership_inventories,
        )
        if derived and tuple(
            record.row for record in self.matrix_reports
        ) != ARM_MATRIX_ROWS:
            raise ValueError("matrix rows are missing, altered, or reordered")
        if not derived and self.matrix_reports:
            raise ValueError(
                "unresolved joint selection cannot retain matrix reports"
            )
        expected_bound_reports = tuple(
            H6MatrixMatchingReportRecord.create(
                row=row,
                report=report,
                selection_obligations=_row_selection_obligations(
                    row=row,
                    component_selections=self.component_selections,
                ),
            )
            for row, report in derived
        )
        if self.matrix_reports != expected_bound_reports:
            raise ValueError(
                "matrix reports are not derivable from the exact preimages"
            )
        if self.primary_selection.status != "ELIGIBLE":
            expected_status = "INCONCLUSIVE"
            expected_obligations = self.primary_selection.obligations
            if self.ownership_inventories or self.matrix_reports:
                raise ValueError(
                    "inconclusive PRIMARY cannot retain endpoint evidence"
                )
        else:
            if (
                not derived
                or self.authorizing_matching_report_ids != ("PRIMARY",)
            ):
                raise ValueError(
                    "eligible matching set requires the sole exact PRIMARY "
                    "hard-match authorization report"
                )
            expected_status = "ELIGIBLE"
            expected_obligations = ()
        if (
            self.status != expected_status
            or self.obligations != expected_obligations
        ):
            raise ValueError(
                "matching-set status and obligations are not exactly derived"
            )
        if self.matching_set_sha256 != _owned_hash(
            "vfe4.h6.matching-set.v2", self._identity_payload()
        ):
            raise ValueError("matching-set digest does not bind all preimages")

    def _identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "git_head": self.git_head,
            "dirty_digest": self.dirty_digest,
            "source_sha256": self.source_sha256,
            "matching_config_sha256": self.matching_config_sha256,
            "matching_policy_sha256": self.matching_policy_sha256,
            "workload_sha256": self.workload.workload_sha256,
            "primary_selection_sha256": (
                self.primary_selection.selection_sha256
            ),
            "primary_candidate_inventory_sha256": (
                self.primary_selection.candidate_inventory_sha256
            ),
            "primary_candidate_count": len(
                self.primary_selection.candidates
            ),
            "optimizer_policy_sha256": (
                self.optimizer_policy.optimizer_policy_sha256
            ),
            "component_selection_sha256s": tuple(
                selection.selection_sha256
                for selection in self.component_selections
            ),
            "ownership_inventory_sha256s": tuple(
                inventory.inventory_sha256
                for inventory in self.ownership_inventories
            ),
            "matrix_report_record_sha256s": tuple(
                record.record_sha256 for record in self.matrix_reports
            ),
            "arm_matrix_sha256": self.arm_matrix_sha256,
            "status": self.status,
            "obligations": self.obligations,
        }

    @classmethod
    def create(
        cls,
        *,
        git_head: str,
        dirty_digest: str,
        source_sha256: str,
        primary_matching_config: H6PrimaryMatchingResolvedConfig,
        workload: H6TrainingWorkload,
        primary_selection: H6PrimaryJointSelection,
        component_selections: tuple[H6FormulaSelection, ...],
    ) -> "H6MatchingSetRecord":
        _validate_primary_preimages(
            matching_config=primary_matching_config,
            workload=workload,
            primary_selection=primary_selection,
        )
        _validate_component_selections(
            component_selections=component_selections,
            workload=workload,
        )
        configs = _resolved_endpoint_configs(
            matching_config=primary_matching_config,
            primary_selection=primary_selection,
            component_selections=component_selections,
        )
        ownership_inventories: tuple[H6MatchingOwnershipRecord, ...] = ()
        if configs is not None:
            from vfe4.training.arms import build_arm

            inventories: list[H6MatchingOwnershipRecord] = []
            for config in configs.values():
                arm = build_arm(config.arm, config)
                inventories.append(
                    H6MatchingOwnershipRecord.create(
                        config=config,
                        parameter_roles=arm.parameter_roles,
                        optimizer_bindings=arm.optimizer_bindings,
                    )
                )
            ownership_inventories = tuple(inventories)
        matrix_reports = _derive_matrix_reports(
            matching_config=primary_matching_config,
            workload=workload,
            primary_selection=primary_selection,
            component_selections=component_selections,
            ownership_inventories=ownership_inventories,
        )
        bound_reports = tuple(
            H6MatrixMatchingReportRecord.create(
                row=row,
                report=report,
                selection_obligations=_row_selection_obligations(
                    row=row,
                    component_selections=component_selections,
                ),
            )
            for row, report in matrix_reports
        )
        if primary_selection.status != "ELIGIBLE":
            status = "INCONCLUSIVE"
            obligations = primary_selection.obligations
        else:
            primary_reports = tuple(
                record
                for record in bound_reports
                if record.row.row_id == "PRIMARY"
            )
            if (
                len(primary_reports) != 1
                or not primary_reports[0].matched_claim_authorized
                or not primary_reports[0].report.eligible
                or primary_reports[0].report.obligations
            ):
                raise ValueError(
                    "eligible primary selection did not derive one exact "
                    "eligible PRIMARY report"
                )
            status = "ELIGIBLE"
            obligations = ()
        values: dict[str, object] = {
            "schema_version": "h6-matching-set-v2",
            "git_head": git_head,
            "dirty_digest": dirty_digest,
            "source_sha256": source_sha256,
            "primary_matching_config": primary_matching_config,
            "workload": workload,
            "primary_selection": primary_selection,
            "optimizer_policy": H6_ADAMW_POLICY,
            "component_selections": tuple(component_selections),
            "ownership_inventories": ownership_inventories,
            "matrix_reports": bound_reports,
            "arm_matrix_sha256": ARM_MATRIX_SHA256,
            "status": status,
            "obligations": obligations,
        }
        provisional = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(provisional, name, value)
        return _new_frozen(
            cls,
            **values,
            matching_set_sha256=_owned_hash(
                "vfe4.h6.matching-set.v2",
                provisional._identity_payload(),
            ),
        )  # type: ignore[return-value]


def _inference_scorer_authorization(
    configs: tuple[ArmConfig, ...],
) -> tuple[
    tuple[
        str,
        Literal["exact_autoregressive", "weighted_smc"],
    ],
    ...,
]:
    """Bind inference scoring to the selected typed endpoint inventory."""

    if type(configs) is not tuple or not configs:
        raise ValueError(
            "inference scorer authorization requires exact typed configs"
        )
    rows: list[
        tuple[
            str,
            Literal["exact_autoregressive", "weighted_smc"],
        ]
    ] = []
    observed: set[str] = set()
    for config in configs:
        if type(config) is not ArmConfig:
            raise ValueError(
                "inference scorer authorization requires exact typed configs"
            )
        config.__post_init__()
        if config.config_id in observed:
            raise ValueError(
                "inference scorer authorization rejects duplicate endpoints"
            )
        observed.add(config.config_id)
        scorer_kind: Literal[
            "exact_autoregressive",
            "weighted_smc",
        ] = (
            "weighted_smc"
            if config.latent_enabled
            else "exact_autoregressive"
        )
        rows.append((config.config_id, scorer_kind))
    return tuple(rows)


def derive_h6_inference_inclusive_compute_report(
    *,
    matching_set: H6MatchingSetRecord,
    inference_records: tuple[InferenceComputeRecord, ...],
) -> _InferenceInclusiveComputeReport:
    """Add disclosure rows without changing one training eligibility byte."""

    if type(matching_set) is not H6MatchingSetRecord:
        raise ValueError("matching_set must be an exact H6MatchingSetRecord")
    matching_set.__post_init__()
    if matching_set.status != "ELIGIBLE" or matching_set.obligations:
        raise ValueError(
            "inference-inclusive reporting requires an eligible "
            "PRIMARY training-compute match"
        )
    if not matching_set.ownership_inventories:
        raise ValueError(
            "inference-inclusive reporting requires selected endpoint "
            "ownership"
        )
    eligibility_before = h6_canonical_json_bytes(
        matching_set._identity_payload()
    )
    matching_set_sha256_before = matching_set.matching_set_sha256
    scorer_authorization = _inference_scorer_authorization(
        tuple(
            inventory.config
            for inventory in matching_set.ownership_inventories
        )
    )
    training_flops: list[tuple[str, int]] = []
    for inventory in matching_set.ownership_inventories:
        ledger = analytical_training_flop_ledger(
            endpoint_config=inventory.config,
            workload=matching_set.workload,
        )
        ledger.__post_init__()
        if ledger.status != "COMPLETE" or ledger.obligations:
            raise ValueError(
                "inference disclosure requires the complete training ledger "
                "already used by eligibility"
            )
        training_flops.append(
            (
                inventory.config.config_id,
                ledger.total_arithmetic_flops,
            )
        )
    report = _build_inference_inclusive_compute_report(
        training_matching_set_sha256=matching_set.matching_set_sha256,
        training_flops_by_endpoint=tuple(training_flops),
        scorer_authorization=scorer_authorization,
        inference_records=inference_records,
    )
    if (
        matching_set.matching_set_sha256 != matching_set_sha256_before
        or h6_canonical_json_bytes(matching_set._identity_payload())
        != eligibility_before
    ):
        raise RuntimeError(
            "inference disclosure changed training matching eligibility"
        )
    return report


def _endpoints_payload(record: H6MatchingSetRecord) -> dict[str, object]:
    return {
        "schema_version": "h6-matching-evidence-v2",
        "matching_set_sha256": record.matching_set_sha256,
        "primary_matching_config": _primary_matching_config_payload(
            record.primary_matching_config
        ),
        "workload": _workload_payload(record.workload),
        "primary_joint_selection": _primary_selection_payload(
            record.primary_selection
        ),
        "optimizer_policy": _optimizer_policy_payload(
            record.optimizer_policy
        ),
        "component_formula_selections": tuple(
            _selection_payload(selection)
            for selection in record.component_selections
        ),
        "ownership_inventories": tuple(
            inventory.canonical_payload()
            for inventory in record.ownership_inventories
        ),
    }


def _reports_payload(record: H6MatchingSetRecord) -> dict[str, object]:
    return {
        "schema_version": "h6-matching-matrix-reports-v2",
        "matching_set_sha256": record.matching_set_sha256,
        "reports": tuple(
            item.canonical_payload() for item in record.matrix_reports
        ),
    }


def _validation_payload(record: H6MatchingSetRecord) -> dict[str, object]:
    return {
        "schema_version": record.schema_version,
        "git_head": record.git_head,
        "dirty_digest": record.dirty_digest,
        "source_sha256": record.source_sha256,
        "matching_config_sha256": record.matching_config_sha256,
        "matching_policy_sha256": record.matching_policy_sha256,
        "optimizer_policy_sha256": (
            record.optimizer_policy.optimizer_policy_sha256
        ),
        "workload_sha256": record.workload.workload_sha256,
        "primary_selection_sha256": (
            record.primary_selection.selection_sha256
        ),
        "primary_candidate_inventory_sha256": (
            record.primary_selection.candidate_inventory_sha256
        ),
        "primary_candidate_count": len(record.primary_selection.candidates),
        "arm_matrix_sha256": record.arm_matrix_sha256,
        "matching_set_sha256": record.matching_set_sha256,
        "component_formula_selection_count": len(
            record.component_selections
        ),
        "ownership_inventory_count": len(
            record.ownership_inventories
        ),
        "matrix_report_count": len(record.matrix_reports),
        "authorizing_matching_report_ids": (
            record.authorizing_matching_report_ids
        ),
        "eligible_component_report_ids": (
            record.eligible_component_report_ids
        ),
        "unmatched_report_ids": record.unmatched_report_ids,
        "status": record.status,
        "obligations": record.obligations,
    }


def publish_h6_matching_set(
    *,
    artifact_root: Path,
    run_name: str,
    git_head: str,
    dirty_digest: str,
    source_sha256: str,
    primary_matching_config: H6PrimaryMatchingResolvedConfig,
    workload: H6TrainingWorkload,
    primary_selection: H6PrimaryJointSelection,
    component_selections: tuple[H6FormulaSelection, ...],
) -> tuple[H6MatchingSetRecord, Path]:
    """Publish an absent-directory v2 artifact after full typed derivation."""

    record = H6MatchingSetRecord.create(
        git_head=git_head,
        dirty_digest=dirty_digest,
        source_sha256=source_sha256,
        primary_matching_config=primary_matching_config,
        workload=workload,
        primary_selection=primary_selection,
        component_selections=component_selections,
    )
    run_directory = publish_run_directory(
        artifact_root,
        run_name,
        {
            "matching/endpoints.json": _endpoints_payload(record),
            "matching/matrix_reports.json": _reports_payload(record),
            "validation/h6_matching_set.json": _validation_payload(record),
        },
    )
    return record, run_directory


def _is_redirect(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction is not None and is_junction())


def _manifest_inventory(manifest_bytes: bytes) -> dict[str, str]:
    try:
        text = manifest_bytes.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("matching manifest must be ASCII") from exc
    if not text or not text.endswith("\n") or "\r" in text:
        raise ValueError("matching manifest must be canonical LF text")
    inventory: dict[str, str] = {}
    observed_paths: list[str] = []
    for line in text.splitlines():
        if line.count("  ") != 1:
            raise ValueError("matching manifest has a malformed line")
        digest, path_text = line.split("  ", 1)
        _require_sha256(digest, "matching manifest digest")
        path = PurePosixPath(path_text)
        if (
            not path_text
            or path.is_absolute()
            or path.as_posix() != path_text
            or any(part in (".", "..") for part in path.parts)
            or "\\" in path_text
            or path_text in inventory
        ):
            raise ValueError("matching manifest has a noncanonical path")
        inventory[path_text] = digest
        observed_paths.append(path_text)
    if observed_paths != sorted(observed_paths):
        raise ValueError("matching manifest paths must be sorted")
    return inventory


def _read_canonical_json_object(
    path: Path,
) -> tuple[bytes, dict[str, object]]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"matching artifact JSON is unreadable: {path.name}"
        ) from exc
    if type(value) is not dict or artifact_json_bytes(value) != raw:
        raise ValueError(
            f"matching artifact JSON is not canonical: {path.name}"
        )
    return raw, value


def _load_payloads(
    artifact_root: Path,
    *,
    expected_manifest_sha256: str | None,
) -> dict[str, dict[str, object]]:
    if not isinstance(artifact_root, Path):
        raise ValueError("artifact_root must be a pathlib.Path")
    if expected_manifest_sha256 is not None:
        _require_sha256(
            expected_manifest_sha256, "expected_manifest_sha256"
        )
    if not artifact_root.exists() or not artifact_root.is_dir():
        raise ValueError(
            "matching artifact root must be an existing directory"
        )
    if _is_redirect(artifact_root):
        raise ValueError("matching artifact root cannot be a redirect")
    try:
        root = artifact_root.resolve(strict=True)
    except OSError as exc:
        raise ValueError("matching artifact root cannot be resolved") from exc
    observed_files: set[str] = set()
    for path in root.rglob("*"):
        if _is_redirect(path):
            raise ValueError("matching artifact cannot contain a redirect")
        if path.is_file():
            observed_files.add(path.relative_to(root).as_posix())
        elif not path.is_dir():
            raise ValueError("matching artifact contains a non-file entry")
    if observed_files != _ALL_FILES:
        raise ValueError(
            "matching artifact has a missing, extra, or unlisted file"
        )
    try:
        manifest_bytes = (root / "manifest.sha256").read_bytes()
    except OSError as exc:
        raise ValueError("matching manifest is unreadable") from exc
    observed_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if (
        expected_manifest_sha256 is not None
        and observed_manifest_sha256 != expected_manifest_sha256
    ):
        raise ValueError("matching manifest SHA-256 differs from expectation")
    inventory = _manifest_inventory(manifest_bytes)
    if set(inventory) != set(_PAYLOAD_PATHS):
        raise ValueError(
            "matching manifest must bind the exact payload inventory"
        )
    payloads: dict[str, dict[str, object]] = {}
    for relative in _PAYLOAD_PATHS:
        raw, payload = _read_canonical_json_object(root / relative)
        if hashlib.sha256(raw).hexdigest() != inventory[relative]:
            raise ValueError(f"matching manifest does not bind {relative}")
        payloads[relative] = payload
    return payloads


def _read_capacity(raw: object, name: str) -> CapacityAllocation:
    if type(raw) is not dict:
        raise ValueError(f"{name} must be a JSON object")
    _exact_keys(
        raw,
        {
            "emission_width",
            "latent_width",
            "recognition_width",
            "prior_context_width",
            "allocation_sha256",
        },
        name,
    )
    result = CapacityAllocation.create(
        emission_width=raw["emission_width"],  # type: ignore[arg-type]
        latent_width=raw["latent_width"],  # type: ignore[arg-type]
        recognition_width=raw["recognition_width"],  # type: ignore[arg-type]
        prior_context_width=raw["prior_context_width"],  # type: ignore[arg-type]
    )
    if not _same_json_value(_capacity_payload(result), raw):
        raise ValueError(f"{name} digest or fields are stale")
    return result


def _read_vocabulary(raw: object, name: str) -> VocabularyIdentity:
    if type(raw) is not dict:
        raise ValueError(f"{name} must be a JSON object")
    _exact_keys(
        raw,
        {"vocabulary_id", "size", "tokenizer_spec_sha256"},
        name,
    )
    result = VocabularyIdentity(
        vocabulary_id=raw["vocabulary_id"],  # type: ignore[arg-type]
        size=raw["size"],  # type: ignore[arg-type]
        tokenizer_spec_sha256=raw[
            "tokenizer_spec_sha256"
        ],  # type: ignore[arg-type]
    )
    if not _same_json_value(_vocabulary_payload(result), raw):
        raise ValueError(f"{name} fields are stale")
    return result


def _read_arm_config(raw: object, name: str) -> ArmConfig:
    if type(raw) is not dict:
        raise ValueError(f"{name} must be a JSON object")
    _exact_keys(
        raw,
        {
            "arm",
            "config_id",
            "vocabulary",
            "horizon",
            "latent_enabled",
            "state_channel_enabled",
            "model_channel_enabled",
            "source_mode",
            "map_mode",
            "recognition_family",
            "recognition_conditioning",
            "prior_variant",
            "mixture_mode",
            "objective_kind",
            "capacity_allocation",
            "config_sha256",
        },
        name,
    )
    try:
        arm = ArmId(raw["arm"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}.arm is invalid") from exc
    result = ArmConfig.create(
        arm=arm,
        config_id=raw["config_id"],  # type: ignore[arg-type]
        vocabulary=_read_vocabulary(
            raw["vocabulary"], f"{name}.vocabulary"
        ),
        horizon=raw["horizon"],  # type: ignore[arg-type]
        latent_enabled=raw["latent_enabled"],  # type: ignore[arg-type]
        state_channel_enabled=raw[
            "state_channel_enabled"
        ],  # type: ignore[arg-type]
        model_channel_enabled=raw[
            "model_channel_enabled"
        ],  # type: ignore[arg-type]
        source_mode=raw["source_mode"],  # type: ignore[arg-type]
        map_mode=raw["map_mode"],  # type: ignore[arg-type]
        recognition_family=raw[
            "recognition_family"
        ],  # type: ignore[arg-type]
        recognition_conditioning=raw[
            "recognition_conditioning"
        ],  # type: ignore[arg-type]
        prior_variant=raw["prior_variant"],  # type: ignore[arg-type]
        mixture_mode=raw["mixture_mode"],  # type: ignore[arg-type]
        objective_kind=raw["objective_kind"],  # type: ignore[arg-type]
        capacity_allocation=_read_capacity(
            raw["capacity_allocation"],
            f"{name}.capacity_allocation",
        ),
    )
    if not _same_json_value(_arm_config_payload(result), raw):
        raise ValueError(f"{name} digest or fields are stale")
    return result


def _read_workload(raw: object, name: str) -> H6TrainingWorkload:
    if type(raw) is not dict:
        raise ValueError(f"{name} must be a JSON object")
    expected = set(H6TrainingWorkload.__dataclass_fields__)
    _exact_keys(raw, expected, name)
    values = dict(raw)
    values["validation_boundaries_per_pass"] = _int_tuple(
        raw["validation_boundaries_per_pass"],
        f"{name}.validation_boundaries_per_pass",
    )
    result = H6TrainingWorkload(**values)  # type: ignore[arg-type]
    if not _same_json_value(_workload_payload(result), raw):
        raise ValueError(f"{name} digest or fields are stale")
    return result


def _read_flop_term(raw: object, name: str) -> FlopTerm:
    if type(raw) is not dict:
        raise ValueError(f"{name} must be a JSON object")
    _exact_keys(raw, set(FlopTerm.__dataclass_fields__), name)
    result = FlopTerm.create(
        phase=raw["phase"],  # type: ignore[arg-type]
        operation=raw["operation"],  # type: ignore[arg-type]
        repetitions=raw["repetitions"],  # type: ignore[arg-type]
        arithmetic_flops_per_repetition=raw[
            "arithmetic_flops_per_repetition"
        ],  # type: ignore[arg-type]
        bytes_copied_per_repetition=raw[
            "bytes_copied_per_repetition"
        ],  # type: ignore[arg-type]
    )
    if not _same_json_value(_flop_term_payload(result), raw):
        raise ValueError(f"{name} digest or totals are stale")
    return result


def _read_ledger(raw: object, name: str) -> H6AnalyticalFlopLedger:
    if type(raw) is not dict:
        raise ValueError(f"{name} must be a JSON object")
    _exact_keys(
        raw,
        {
            "schema_version",
            "endpoint_config_sha256",
            "endpoint_profile_sha256",
            "allocation_sha256",
            "workload_sha256",
            "terms",
            "total_arithmetic_flops",
            "total_bytes_copied",
            "status",
            "obligations",
            "ledger_sha256",
        },
        name,
    )
    raw_terms = raw["terms"]
    if type(raw_terms) is not list or not raw_terms:
        raise ValueError(f"{name}.terms must be a nonempty JSON array")
    terms = tuple(
        _read_flop_term(term, f"{name}.terms[{index}]")
        for index, term in enumerate(raw_terms)
    )
    result = H6AnalyticalFlopLedger(
        schema_version=raw["schema_version"],  # type: ignore[arg-type]
        endpoint_config_sha256=raw[
            "endpoint_config_sha256"
        ],  # type: ignore[arg-type]
        endpoint_profile_sha256=raw[
            "endpoint_profile_sha256"
        ],  # type: ignore[arg-type]
        allocation_sha256=raw[
            "allocation_sha256"
        ],  # type: ignore[arg-type]
        workload_sha256=raw["workload_sha256"],  # type: ignore[arg-type]
        terms=terms,
        total_arithmetic_flops=raw[
            "total_arithmetic_flops"
        ],  # type: ignore[arg-type]
        total_bytes_copied=raw[
            "total_bytes_copied"
        ],  # type: ignore[arg-type]
        status=raw["status"],  # type: ignore[arg-type]
        obligations=_string_tuple(
            raw["obligations"], f"{name}.obligations"
        ),
        ledger_sha256=raw["ledger_sha256"],  # type: ignore[arg-type]
    )
    if not _same_json_value(_ledger_payload(result), raw):
        raise ValueError(f"{name} digest or fields are stale")
    return result


def _read_selection(raw: object, name: str) -> H6FormulaSelection:
    if type(raw) is not dict:
        raise ValueError(f"{name} must be a JSON object")
    _exact_keys(
        raw,
        {
            "config_id",
            "endpoint_template",
            "reference_config",
            "endpoint_profile_sha256",
            "reference_profile_sha256",
            "workload",
            "policy_sha256",
            "candidate_count_evaluated",
            "selected_endpoint_config",
            "selected_allocation",
            "parameter_count",
            "training_flops",
            "parameter_relative_difference",
            "flop_relative_difference",
            "ledger",
            "reference_ledger",
            "status",
            "obligations",
            "selection_sha256",
        },
        name,
    )
    selected_config = (
        None
        if raw["selected_endpoint_config"] is None
        else _read_arm_config(
            raw["selected_endpoint_config"],
            f"{name}.selected_endpoint_config",
        )
    )
    selected_allocation = (
        None
        if raw["selected_allocation"] is None
        else _read_capacity(
            raw["selected_allocation"],
            f"{name}.selected_allocation",
        )
    )
    ledger = (
        None
        if raw["ledger"] is None
        else _read_ledger(raw["ledger"], f"{name}.ledger")
    )
    result = H6FormulaSelection(
        config_id=raw["config_id"],  # type: ignore[arg-type]
        endpoint_template=_read_arm_config(
            raw["endpoint_template"], f"{name}.endpoint_template"
        ),
        reference_config=_read_arm_config(
            raw["reference_config"], f"{name}.reference_config"
        ),
        endpoint_profile_sha256=raw[
            "endpoint_profile_sha256"
        ],  # type: ignore[arg-type]
        reference_profile_sha256=raw[
            "reference_profile_sha256"
        ],  # type: ignore[arg-type]
        workload=_read_workload(raw["workload"], f"{name}.workload"),
        policy_sha256=raw["policy_sha256"],  # type: ignore[arg-type]
        candidate_count_evaluated=raw[
            "candidate_count_evaluated"
        ],  # type: ignore[arg-type]
        selected_endpoint_config=selected_config,
        selected_allocation=selected_allocation,
        parameter_count=raw["parameter_count"],  # type: ignore[arg-type]
        training_flops=raw["training_flops"],  # type: ignore[arg-type]
        parameter_relative_difference=raw[
            "parameter_relative_difference"
        ],  # type: ignore[arg-type]
        flop_relative_difference=raw[
            "flop_relative_difference"
        ],  # type: ignore[arg-type]
        ledger=ledger,
        reference_ledger=_read_ledger(
            raw["reference_ledger"], f"{name}.reference_ledger"
        ),
        status=raw["status"],  # type: ignore[arg-type]
        obligations=_string_tuple(
            raw["obligations"], f"{name}.obligations"
        ),
        selection_sha256=raw["selection_sha256"],  # type: ignore[arg-type]
    )
    if not _same_json_value(_selection_payload(result), raw):
        raise ValueError(f"{name} digest or fields are stale")
    return result


def _read_primary_matching_config(
    raw: object,
) -> H6PrimaryMatchingResolvedConfig:
    name = "primary_matching_config"
    if type(raw) is not dict:
        raise ValueError(f"{name} must be a JSON object")
    _exact_keys(
        raw,
        set(H6PrimaryMatchingResolvedConfig.__dataclass_fields__),
        name,
    )
    result = H6PrimaryMatchingResolvedConfig(
        schema_version=raw["schema_version"],  # type: ignore[arg-type]
        operation=raw["operation"],  # type: ignore[arg-type]
        a0_config=_read_arm_config(
            raw["a0_config"], f"{name}.a0_config"
        ),
        a5_template=_read_arm_config(
            raw["a5_template"], f"{name}.a5_template"
        ),
        latent_width_candidates=_int_tuple(
            raw["latent_width_candidates"],
            f"{name}.latent_width_candidates",
            nonempty=True,
        ),  # type: ignore[arg-type]
        prior_context_width_candidates=_int_tuple(
            raw["prior_context_width_candidates"],
            f"{name}.prior_context_width_candidates",
            nonempty=True,
        ),  # type: ignore[arg-type]
        emission_width_candidates=_int_tuple(
            raw["emission_width_candidates"],
            f"{name}.emission_width_candidates",
            nonempty=True,
        ),  # type: ignore[arg-type]
        recognition_width_candidates=_int_tuple(
            raw["recognition_width_candidates"],
            f"{name}.recognition_width_candidates",
            nonempty=True,
        ),  # type: ignore[arg-type]
        parameter_relative_tolerance=raw[
            "parameter_relative_tolerance"
        ],  # type: ignore[arg-type]
        flop_relative_tolerance=raw[
            "flop_relative_tolerance"
        ],  # type: ignore[arg-type]
        matching_policy_sha256=raw[
            "matching_policy_sha256"
        ],  # type: ignore[arg-type]
        canonical_json=raw["canonical_json"],  # type: ignore[arg-type]
        config_sha256=raw["config_sha256"],  # type: ignore[arg-type]
    )
    if not _same_json_value(_primary_matching_config_payload(result), raw):
        raise ValueError(
            "primary matching config digest or fields are stale"
        )
    return result


def _read_primary_candidate(
    raw: object,
    name: str,
) -> H6PrimaryJointCandidate:
    if type(raw) is not dict:
        raise ValueError(f"{name} must be a JSON object")
    _exact_keys(
        raw,
        set(H6PrimaryJointCandidate.__dataclass_fields__),
        name,
    )
    values = dict(raw)
    values["obligations"] = _string_tuple(
        raw["obligations"], f"{name}.obligations"
    )
    result = H6PrimaryJointCandidate(**values)  # type: ignore[arg-type]
    if not _same_json_value(_primary_candidate_payload(result), raw):
        raise ValueError(f"{name} digest or fields are stale")
    return result


def _read_primary_selection(
    raw: object,
) -> H6PrimaryJointSelection:
    name = "primary_joint_selection"
    if type(raw) is not dict:
        raise ValueError(f"{name} must be a JSON object")
    _exact_keys(
        raw,
        {
            "schema_version",
            "matching_config_sha256",
            "matching_policy_sha256",
            "a0_config_sha256",
            "a5_template_config_sha256",
            "workload_sha256",
            "candidates",
            "candidate_inventory_sha256",
            "status",
            "selected_candidate_sha256",
            "obligations",
            "selection_sha256",
        },
        name,
    )
    raw_candidates = raw["candidates"]
    if type(raw_candidates) is not list:
        raise ValueError(
            "primary_joint_selection.candidates must be a JSON array"
        )
    result = H6PrimaryJointSelection(
        schema_version=raw["schema_version"],  # type: ignore[arg-type]
        matching_config_sha256=raw[
            "matching_config_sha256"
        ],  # type: ignore[arg-type]
        matching_policy_sha256=raw[
            "matching_policy_sha256"
        ],  # type: ignore[arg-type]
        a0_config_sha256=raw[
            "a0_config_sha256"
        ],  # type: ignore[arg-type]
        a5_template_config_sha256=raw[
            "a5_template_config_sha256"
        ],  # type: ignore[arg-type]
        workload_sha256=raw["workload_sha256"],  # type: ignore[arg-type]
        candidates=tuple(
            _read_primary_candidate(
                candidate,
                f"{name}.candidates[{index}]",
            )
            for index, candidate in enumerate(raw_candidates)
        ),
        candidate_inventory_sha256=raw[
            "candidate_inventory_sha256"
        ],  # type: ignore[arg-type]
        status=raw["status"],  # type: ignore[arg-type]
        selected_candidate_sha256=raw[
            "selected_candidate_sha256"
        ],  # type: ignore[arg-type]
        obligations=_string_tuple(
            raw["obligations"],
            f"{name}.obligations",
        ),
        selection_sha256=raw["selection_sha256"],  # type: ignore[arg-type]
    )
    if not _same_json_value(_primary_selection_payload(result), raw):
        raise ValueError(
            "primary joint selection digest or fields are stale"
        )
    return result


def _read_parameter_role(
    raw: object, name: str
) -> ParameterRoleRecord:
    if type(raw) is not dict:
        raise ValueError(f"{name} must be a JSON object")
    _exact_keys(raw, set(ParameterRoleRecord.__dataclass_fields__), name)
    result = ParameterRoleRecord.create(
        qualified_name=raw["qualified_name"],  # type: ignore[arg-type]
        role=raw["role"],  # type: ignore[arg-type]
        phase=raw["phase"],  # type: ignore[arg-type]
        parameter_key=raw["parameter_key"],  # type: ignore[arg-type]
        scalar_count=raw["scalar_count"],  # type: ignore[arg-type]
    )
    if not _same_json_value(_role_payload(result), raw):
        raise ValueError(f"{name} digest or fields are stale")
    return result


def _read_optimizer_binding(
    raw: object, name: str
) -> OptimizerBinding:
    if type(raw) is not dict:
        raise ValueError(f"{name} must be a JSON object")
    _exact_keys(raw, set(OptimizerBinding.__dataclass_fields__), name)
    result = OptimizerBinding.create(
        phase=raw["phase"],  # type: ignore[arg-type]
        optimizer_class=raw["optimizer_class"],  # type: ignore[arg-type]
        optimizer_policy_sha256=raw[
            "optimizer_policy_sha256"
        ],  # type: ignore[arg-type]
        parameter_keys=_string_tuple(
            raw["parameter_keys"], f"{name}.parameter_keys", nonempty=True
        ),
    )
    if not _same_json_value(_binding_payload(result), raw):
        raise ValueError(f"{name} digest or fields are stale")
    return result


def _read_ownership(
    raw: object, name: str
) -> H6MatchingOwnershipRecord:
    if type(raw) is not dict:
        raise ValueError(f"{name} must be a JSON object")
    _exact_keys(
        raw,
        {
            "config",
            "parameter_roles",
            "optimizer_bindings",
            "inventory_sha256",
        },
        name,
    )
    raw_roles = raw["parameter_roles"]
    raw_bindings = raw["optimizer_bindings"]
    if type(raw_roles) is not list or type(raw_bindings) is not list:
        raise ValueError(
            f"{name} role and binding inventories must be JSON arrays"
        )
    result = H6MatchingOwnershipRecord.create(
        config=_read_arm_config(raw["config"], f"{name}.config"),
        parameter_roles=tuple(
            _read_parameter_role(role, f"{name}.parameter_roles[{index}]")
            for index, role in enumerate(raw_roles)
        ),
        optimizer_bindings=tuple(
            _read_optimizer_binding(
                binding, f"{name}.optimizer_bindings[{index}]"
            )
            for index, binding in enumerate(raw_bindings)
        ),
    )
    if not _same_json_value(result.canonical_payload(), raw):
        raise ValueError(f"{name} digest or fields are stale")
    return result


def _read_optimizer_policy(raw: object) -> AdamWPolicyRecord:
    if type(raw) is not dict:
        raise ValueError("optimizer policy must be a JSON object")
    _exact_keys(
        raw,
        set(AdamWPolicyRecord.__dataclass_fields__),
        "optimizer policy",
    )
    if not _same_json_value(_optimizer_policy_payload(H6_ADAMW_POLICY), raw):
        raise ValueError("optimizer policy differs from the exact policy")
    H6_ADAMW_POLICY.__post_init__()
    return H6_ADAMW_POLICY


def _read_matching_report(payload: object, name: str) -> MatchingReport:
    if type(payload) is not dict:
        raise ValueError(f"{name} must be a JSON object")
    _exact_keys(payload, set(_MATCHING_REPORT_FIELDS), name)
    report = MatchingReport.from_totals(
        matching_config_sha256=payload[
            "matching_config_sha256"
        ],  # type: ignore[arg-type]
        endpoint_config_sha256=payload[
            "endpoint_config_sha256"
        ],  # type: ignore[arg-type]
        reference_config_sha256=payload[
            "reference_config_sha256"
        ],  # type: ignore[arg-type]
        endpoint_parameter_count=payload[
            "endpoint_parameter_count"
        ],  # type: ignore[arg-type]
        reference_parameter_count=payload[
            "reference_parameter_count"
        ],  # type: ignore[arg-type]
        endpoint_training_flops=payload[
            "endpoint_training_flops"
        ],  # type: ignore[arg-type]
        reference_training_flops=payload[
            "reference_training_flops"
        ],  # type: ignore[arg-type]
        parameter_relative_tolerance=payload[
            "parameter_relative_tolerance"
        ],  # type: ignore[arg-type]
        flop_relative_tolerance=payload[
            "flop_relative_tolerance"
        ],  # type: ignore[arg-type]
        ownership_valid=payload["ownership_valid"],  # type: ignore[arg-type]
        common_schedule=payload["common_schedule"],  # type: ignore[arg-type]
        optimizer_policy_match=payload[
            "optimizer_policy_match"
        ],  # type: ignore[arg-type]
        training_flop_ledger_complete=payload[
            "training_flop_ledger_complete"
        ],  # type: ignore[arg-type]
        training_flop_obligations=_string_tuple(
            payload["training_flop_obligations"],
            f"{name}.training_flop_obligations",
        ),
        semantic_interventions=_string_tuple(
            payload["semantic_interventions"],
            f"{name}.semantic_interventions",
        ),
        named_factor=payload["named_factor"],  # type: ignore[arg-type]
        nuisance_capacity_fields=_string_tuple(
            payload["nuisance_capacity_fields"],
            f"{name}.nuisance_capacity_fields",
        ),
        common_schedule_sha256=payload[
            "common_schedule_sha256"
        ],  # type: ignore[arg-type]
    )
    if not _same_json_value(_matching_report_payload(report), payload):
        raise ValueError(f"{name} digest or derived fields are stale")
    return report


def _read_row(raw: object, name: str) -> ArmMatrixRow:
    if type(raw) is not dict:
        raise ValueError(f"{name} must be a JSON object")
    _exact_keys(raw, set(ArmMatrixRow.__dataclass_fields__), name)
    result = ArmMatrixRow.create(
        row_id=raw["row_id"],  # type: ignore[arg-type]
        left_config_id=raw["left_config_id"],  # type: ignore[arg-type]
        left_factory_id=raw["left_factory_id"],  # type: ignore[arg-type]
        right_config_id=raw["right_config_id"],  # type: ignore[arg-type]
        right_factory_id=raw["right_factory_id"],  # type: ignore[arg-type]
        named_factor=raw["named_factor"],  # type: ignore[arg-type]
        semantic_interventions=_string_tuple(
            raw["semantic_interventions"],
            f"{name}.semantic_interventions",
            nonempty=True,
        ),
        nuisance_capacity_fields=_string_tuple(
            raw["nuisance_capacity_fields"],
            f"{name}.nuisance_capacity_fields",
        ),
        tuning_estimand=raw["tuning_estimand"],  # type: ignore[arg-type]
        interpretation=raw["interpretation"],  # type: ignore[arg-type]
        checkpoint_template=raw[
            "checkpoint_template"
        ],  # type: ignore[arg-type]
        certificate_key_template=raw[
            "certificate_key_template"
        ],  # type: ignore[arg-type]
        opening_group=raw["opening_group"],  # type: ignore[arg-type]
        nonclaims=_string_tuple(
            raw["nonclaims"], f"{name}.nonclaims", nonempty=True
        ),
    )
    if not _same_json_value(_row_payload(result), raw):
        raise ValueError(f"{name} digest or frozen fields are stale")
    return result


def _read_matrix_record(
    raw: object, name: str
) -> H6MatrixMatchingReportRecord:
    if type(raw) is not dict:
        raise ValueError(f"{name} must be a JSON object")
    _exact_keys(
        raw,
        {
            "row",
            "report",
            "matched_claim_authorized",
            "selection_obligations",
            "record_sha256",
        },
        name,
    )
    result = H6MatrixMatchingReportRecord.create(
        row=_read_row(raw["row"], f"{name}.row"),
        report=_read_matching_report(
            raw["report"], f"{name}.report"
        ),
        selection_obligations=_string_tuple(
            raw["selection_obligations"],
            f"{name}.selection_obligations",
        ),
    )
    if not _same_json_value(result.canonical_payload(), raw):
        raise ValueError(f"{name} digest or fields are stale")
    return result


def read_h6_matching_set(
    artifact_root: Path,
    *,
    expected_manifest_sha256: str | None = None,
    expected_set_sha256: str | None = None,
    expected_git_head: str | None = None,
    expected_dirty_digest: str | None = None,
) -> H6MatchingSetRecord:
    """Reconstruct and independently derive a manifest-bound v2 artifact."""

    if expected_set_sha256 is not None:
        _require_sha256(expected_set_sha256, "expected_set_sha256")
    if (expected_git_head is None) != (expected_dirty_digest is None):
        raise ValueError(
            "expected_git_head and expected_dirty_digest must be supplied together"
        )
    if expected_git_head is not None:
        _require_git_head(expected_git_head)
        _require_sha256(expected_dirty_digest, "expected_dirty_digest")
    payloads = _load_payloads(
        artifact_root,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    endpoints = payloads["matching/endpoints.json"]
    reports = payloads["matching/matrix_reports.json"]
    validation = payloads["validation/h6_matching_set.json"]
    if (
        endpoints.get("schema_version") != "h6-matching-evidence-v2"
        or "matching_policy" in endpoints
        or "formula_selections" in endpoints
    ):
        raise H6MatchingPublicationBlocked(
            "legacy H6 PRIMARY evidence without the resolved joint search "
            "is not authorized"
        )
    _exact_keys(
        endpoints,
        {
            "schema_version",
            "matching_set_sha256",
            "primary_matching_config",
            "workload",
            "primary_joint_selection",
            "optimizer_policy",
            "component_formula_selections",
            "ownership_inventories",
        },
        "matching evidence payload",
    )
    primary_matching_config = _read_primary_matching_config(
        endpoints["primary_matching_config"]
    )
    workload = _read_workload(endpoints["workload"], "workload")
    primary_selection = _read_primary_selection(
        endpoints["primary_joint_selection"]
    )
    _read_optimizer_policy(endpoints["optimizer_policy"])
    raw_selections = endpoints["component_formula_selections"]
    raw_ownership = endpoints["ownership_inventories"]
    if type(raw_selections) is not list or type(raw_ownership) is not list:
        raise ValueError(
            "matching selections and ownership must be JSON arrays"
    )
    selections = tuple(
        _read_selection(
            item,
            f"component_formula_selections[{index}]",
        )
        for index, item in enumerate(raw_selections)
    )
    ownership = tuple(
        _read_ownership(item, f"ownership_inventories[{index}]")
        for index, item in enumerate(raw_ownership)
    )
    _exact_keys(
        reports,
        {"schema_version", "matching_set_sha256", "reports"},
        "matching matrix reports payload",
    )
    if reports["schema_version"] != "h6-matching-matrix-reports-v2":
        raise ValueError("unsupported matching matrix-report schema")
    raw_reports = reports["reports"]
    if type(raw_reports) is not list:
        raise ValueError("matrix reports must be a JSON array")
    report_records = tuple(
        _read_matrix_record(item, f"matrix_reports[{index}]")
        for index, item in enumerate(raw_reports)
    )
    _exact_keys(
        validation,
        {
            "schema_version",
            "git_head",
            "dirty_digest",
            "source_sha256",
            "matching_config_sha256",
            "matching_policy_sha256",
            "optimizer_policy_sha256",
            "workload_sha256",
            "primary_selection_sha256",
            "primary_candidate_inventory_sha256",
            "primary_candidate_count",
            "arm_matrix_sha256",
            "matching_set_sha256",
            "component_formula_selection_count",
            "ownership_inventory_count",
            "matrix_report_count",
            "authorizing_matching_report_ids",
            "eligible_component_report_ids",
            "unmatched_report_ids",
            "status",
            "obligations",
        },
        "matching-set validation payload",
    )
    record = H6MatchingSetRecord.create(
        git_head=validation["git_head"],  # type: ignore[arg-type]
        dirty_digest=validation["dirty_digest"],  # type: ignore[arg-type]
        source_sha256=validation["source_sha256"],  # type: ignore[arg-type]
        primary_matching_config=primary_matching_config,
        workload=workload,
        primary_selection=primary_selection,
        component_selections=selections,
    )
    if (
        record.ownership_inventories != ownership
        or record.matrix_reports != report_records
        or not _same_json_value(_endpoints_payload(record), endpoints)
        or not _same_json_value(_reports_payload(record), reports)
    ):
        raise ValueError(
            "matching endpoint or report evidence is not the exact "
            "independent reconstruction"
        )
    if not _same_json_value(_validation_payload(record), validation):
        raise ValueError(
            "matching-set validation is stale or omits a preimage"
        )
    for payload in (endpoints, reports):
        if payload["matching_set_sha256"] != record.matching_set_sha256:
            raise ValueError("matching payloads bind a different matching set")
    if (
        expected_set_sha256 is not None
        and record.matching_set_sha256 != expected_set_sha256
    ):
        raise ValueError("matching-set SHA-256 differs from expectation")
    if (
        expected_git_head is not None
        and (
            record.git_head != expected_git_head
            or record.dirty_digest != expected_dirty_digest
        )
    ):
        raise ValueError(
            "matching artifact source revision differs from expectation"
        )
    record.__post_init__()
    return record


__all__ = [
    "H6_MATCHING_ENDPOINT_LAYOUT",
    "H6_MATCHING_PUBLICATION_SOURCE_BLOCKERS",
    "H6MatchingOwnershipRecord",
    "H6MatrixMatchingReportRecord",
    "H6MatchingPublicationBlocked",
    "H6MatchingSetRecord",
    "derive_h6_inference_inclusive_compute_report",
    "derive_h6_matrix_reports",
    "publish_h6_matching_set",
    "read_h6_matching_set",
]
