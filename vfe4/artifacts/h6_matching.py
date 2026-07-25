"""Immutable, manifest-linked H6 formula-matching evidence.

Schema v2 is deliberately self-contained.  It records the selected and
reference arm configurations, the complete formula selections and FLOP
ledgers, every stable parameter-role/optimizer binding, and the exact eight
attribution rows and reports.  Reading an artifact reconstructs every typed
record, reruns the formula-only selector, rebuilds each selected arm to audit
its parameter ownership, and derives the eight reports again before returning
an eligible set.  It never opens a corpus, trains an arm, or reads a predictive
outcome.
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
from vfe4.training.matching import (
    AMENDED_MATCHING_SCHEDULE_POLICY,
    ARM_MATRIX_ROWS,
    ARM_MATRIX_SHA256,
    H6_ADAMW_POLICY,
    AmendedMatchingSchedulePolicy,
    H6AnalyticalFlopLedger,
    H6FormulaSelection,
    H6TrainingWorkload,
    audit_parameter_ownership,
    select_outcome_blind_allocation,
    stable_parameter_key,
)
from vfe4.types.h6 import (
    AdamWPolicyRecord,
    ArmConfig,
    ArmId,
    ArmMatrixRow,
    CapacityAllocation,
    FlopTerm,
    MatchingReport,
    OptimizerBinding,
    ParameterRoleRecord,
    TrainingPhase,
    VocabularyIdentity,
    canonical_json_bytes as h6_canonical_json_bytes,
)


H6_MATCHING_ENDPOINT_LAYOUT: tuple[tuple[str, str], ...] = (
    ("h6-a0-ar-v1", "build_a0@h6-arm-v1"),
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
        "h6-a5-structured-prefix-exact-complete-latent-smoothing-v1",
        "build_a5@h6-arm-v1",
    ),
    (
        "h6-a5-structured-fixed-projection-complete-latent-smoothing-v1",
        "build_a5@h6-arm-v1",
    ),
    (
        "h6-a5-structured-fixed-exact-emission-latent-smoothing-v1",
        "build_a5@h6-arm-v1",
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
_MATRIX_COMPARED_CONFIG_IDS = (
    H6_MATCHING_ENDPOINT_LAYOUT[0][0],
    H6_MATCHING_ENDPOINT_LAYOUT[2][0],
    H6_MATCHING_ENDPOINT_LAYOUT[6][0],
    H6_MATCHING_ENDPOINT_LAYOUT[7][0],
    H6_MATCHING_ENDPOINT_LAYOUT[8][0],
    H6_MATCHING_ENDPOINT_LAYOUT[9][0],
    H6_MATCHING_ENDPOINT_LAYOUT[10][0],
    H6_MATCHING_ENDPOINT_LAYOUT[11][0],
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


def _matching_policy_payload(
    value: AmendedMatchingSchedulePolicy,
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
    """One report bound to the complete exact frozen matrix row."""

    row: ArmMatrixRow
    report: MatchingReport
    record_sha256: str

    def __post_init__(self) -> None:
        if type(self.row) is not ArmMatrixRow:
            raise ValueError("matrix row must be an exact ArmMatrixRow")
        if type(self.report) is not MatchingReport:
            raise ValueError("matrix report must be an exact MatchingReport")
        self.row.__post_init__()
        self.report.__post_init__()
        expected = _owned_hash(
            "vfe4.h6.matrix-matching-report.v2",
            {
                "row_sha256": self.row.row_sha256,
                "report_sha256": self.report.report_sha256,
            },
        )
        if self.record_sha256 != expected:
            raise ValueError("matrix report record digest is stale")

    @classmethod
    def create(
        cls, *, row: ArmMatrixRow, report: MatchingReport
    ) -> "H6MatrixMatchingReportRecord":
        return _new_frozen(
            cls,
            row=row,
            report=report,
            record_sha256=_owned_hash(
                "vfe4.h6.matrix-matching-report.v2",
                {
                    "row_sha256": row.row_sha256,
                    "report_sha256": report.report_sha256,
                },
            ),
        )  # type: ignore[return-value]

    def canonical_payload(self) -> dict[str, object]:
        return {
            "row": _row_payload(self.row),
            "report": _matching_report_payload(self.report),
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
        )
        if getattr(endpoint, name) != getattr(reference, name)
    )


def _derive_matrix_reports(
    *,
    matching_config_sha256: str,
    selections: tuple[H6FormulaSelection, ...],
    ownership_inventories: tuple[H6MatchingOwnershipRecord, ...],
) -> tuple[tuple[ArmMatrixRow, MatchingReport], ...]:
    """Derive all eight reports solely from exact serialized preimages."""

    _require_sha256(matching_config_sha256, "matching_config_sha256")
    selection_by_id = {
        selection.config_id: selection for selection in selections
    }
    ownership_by_sha256 = {
        inventory.config.config_sha256: inventory
        for inventory in ownership_inventories
    }
    reference_selection = selection_by_id[_REFERENCE_CONFIG_ID]
    reference_config = reference_selection.reference_config
    reference_inventory = ownership_by_sha256[
        reference_config.config_sha256
    ]
    results: list[tuple[ArmMatrixRow, MatchingReport]] = []
    for row, config_id in zip(
        ARM_MATRIX_ROWS, _MATRIX_COMPARED_CONFIG_IDS, strict=True
    ):
        selection = selection_by_id[config_id]
        endpoint_config = selection.selected_endpoint_config
        endpoint_ledger = selection.ledger
        if (
            type(endpoint_config) is not ArmConfig
            or type(endpoint_ledger) is not H6AnalyticalFlopLedger
        ):
            raise ValueError("matrix endpoint selection is not eligible")
        endpoint_inventory = ownership_by_sha256[
            endpoint_config.config_sha256
        ]
        report = MatchingReport.from_totals(
            matching_config_sha256=matching_config_sha256,
            endpoint_config_sha256=endpoint_config.config_sha256,
            reference_config_sha256=reference_config.config_sha256,
            endpoint_parameter_count=endpoint_inventory.parameter_count,
            reference_parameter_count=reference_inventory.parameter_count,
            endpoint_training_flops=(
                endpoint_ledger.total_arithmetic_flops
            ),
            reference_training_flops=(
                selection.reference_ledger.total_arithmetic_flops
            ),
            parameter_relative_tolerance=0.01,
            flop_relative_tolerance=0.05,
            ownership_valid=True,
            common_schedule=True,
            optimizer_policy_match=True,
            training_flop_ledger_complete=(
                endpoint_ledger.status == "COMPLETE"
                and selection.reference_ledger.status == "COMPLETE"
            ),
            training_flop_obligations=(
                endpoint_ledger.obligations
                + selection.reference_ledger.obligations
            ),
            semantic_interventions=row.semantic_interventions,
            named_factor=row.named_factor,
            nuisance_capacity_fields=_capacity_differences(
                endpoint_config.capacity_allocation,
                reference_config.capacity_allocation,
            ),
            common_schedule_sha256=(
                AMENDED_MATCHING_SCHEDULE_POLICY.policy_sha256
            ),
        )
        results.append((row, report))
    return tuple(results)


def derive_h6_matrix_reports(
    *,
    matching_config_sha256: str,
    selections: tuple[H6FormulaSelection, ...],
    ownership_inventories: tuple[H6MatchingOwnershipRecord, ...],
) -> tuple[tuple[ArmMatrixRow, MatchingReport], ...]:
    """Public pure helper used by click-to-run evidence assembly."""

    return _derive_matrix_reports(
        matching_config_sha256=matching_config_sha256,
        selections=selections,
        ownership_inventories=ownership_inventories,
    )


@dataclass(frozen=True, slots=True, init=False)
class H6MatchingSetRecord:
    """Complete eligible H6 matching evidence under artifact schema v2."""

    schema_version: Literal["h6-matching-set-v2"]
    git_head: str
    dirty_digest: str
    source_sha256: str
    matching_config_sha256: str
    matching_policy: AmendedMatchingSchedulePolicy
    optimizer_policy: AdamWPolicyRecord
    selections: tuple[H6FormulaSelection, ...]
    ownership_inventories: tuple[H6MatchingOwnershipRecord, ...]
    matrix_reports: tuple[H6MatrixMatchingReportRecord, ...]
    arm_matrix_sha256: str
    status: Literal["ELIGIBLE"]
    obligations: tuple[()]
    matching_set_sha256: str

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
        if (
            type(self.matching_policy)
            is not AmendedMatchingSchedulePolicy
            or self.matching_policy != AMENDED_MATCHING_SCHEDULE_POLICY
        ):
            raise ValueError("matching set requires the exact amended policy")
        self.matching_policy.__post_init__()
        if (
            type(self.optimizer_policy) is not AdamWPolicyRecord
            or self.optimizer_policy != H6_ADAMW_POLICY
        ):
            raise ValueError("matching set requires the exact AdamW policy")
        self.optimizer_policy.__post_init__()
        if (
            type(self.selections) is not tuple
            or len(self.selections) != len(H6_MATCHING_ENDPOINT_LAYOUT)
            or any(
                type(selection) is not H6FormulaSelection
                for selection in self.selections
            )
        ):
            raise ValueError("matching set requires twelve exact selections")
        for selection in self.selections:
            selection.__post_init__()
        if tuple(
            selection.config_id for selection in self.selections
        ) != tuple(config_id for config_id, _ in H6_MATCHING_ENDPOINT_LAYOUT):
            raise ValueError("formula selections are missing or reordered")
        if any(
            selection.status != "ELIGIBLE"
            or selection.obligations
            or selection.selected_endpoint_config is None
            or selection.selected_allocation is None
            or selection.ledger is None
            for selection in self.selections
        ):
            raise ValueError("every formula selection must be eligible")
        workloads = tuple(
            selection.workload for selection in self.selections
        )
        if any(workload != workloads[0] for workload in workloads[1:]):
            raise ValueError("all selections must bind one exact workload")
        reference_configs = tuple(
            selection.reference_config for selection in self.selections
        )
        if any(
            reference != reference_configs[0]
            for reference in reference_configs[1:]
        ):
            raise ValueError("all selections must bind one exact reference")
        reference_ledgers = tuple(
            selection.reference_ledger for selection in self.selections
        )
        if any(
            ledger != reference_ledgers[0]
            for ledger in reference_ledgers[1:]
        ):
            raise ValueError(
                "all selections must bind one exact reference ledger"
            )
        base_selection = self.selections[5]
        if (
            base_selection.selected_endpoint_config
            != base_selection.reference_config
            or base_selection.selected_allocation
            != base_selection.reference_config.capacity_allocation
            or base_selection.ledger != base_selection.reference_ledger
        ):
            raise ValueError(
                "canonical A5 selection must equal the exact reference"
            )
        if (
            type(self.ownership_inventories) is not tuple
            or len(self.ownership_inventories) != len(self.selections)
            or any(
                type(inventory) is not H6MatchingOwnershipRecord
                for inventory in self.ownership_inventories
            )
        ):
            raise ValueError(
                "matching set requires twelve exact ownership inventories"
            )
        for inventory in self.ownership_inventories:
            inventory.__post_init__()
        selected_configs = tuple(
            selection.selected_endpoint_config
            for selection in self.selections
        )
        if tuple(
            inventory.config for inventory in self.ownership_inventories
        ) != selected_configs:
            raise ValueError(
                "ownership inventories are missing, extra, or reordered"
            )
        selected_hashes = tuple(
            inventory.config.config_sha256
            for inventory in self.ownership_inventories
        )
        if len(set(selected_hashes)) != len(selected_hashes):
            raise ValueError("selected endpoint identities must be unique")
        for selection, inventory in zip(
            self.selections,
            self.ownership_inventories,
            strict=True,
        ):
            if inventory.parameter_count != selection.parameter_count:
                raise ValueError(
                    "parameter-role inventory differs from formula selection"
                )
            recomputed_selection = select_outcome_blind_allocation(
                endpoint_template=selection.endpoint_template,
                reference_config=selection.reference_config,
                workload=selection.workload,
            )
            if recomputed_selection != selection:
                raise ValueError(
                    "serialized formula selection is not the exact first "
                    "outcome-blind eligible allocation"
                )
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
            or len(self.matrix_reports) != len(ARM_MATRIX_ROWS)
            or any(
                type(record) is not H6MatrixMatchingReportRecord
                for record in self.matrix_reports
            )
        ):
            raise ValueError("matching set requires eight exact matrix reports")
        for record in self.matrix_reports:
            record.__post_init__()
        if tuple(record.row for record in self.matrix_reports) != ARM_MATRIX_ROWS:
            raise ValueError("matrix rows are missing, altered, or reordered")
        derived = _derive_matrix_reports(
            matching_config_sha256=self.matching_config_sha256,
            selections=self.selections,
            ownership_inventories=self.ownership_inventories,
        )
        if tuple(
            (record.row, record.report)
            for record in self.matrix_reports
        ) != derived:
            raise ValueError(
                "matrix reports are not derivable from the exact preimages"
            )
        if any(
            record.report.status != "ELIGIBLE"
            or record.report.eligible is not True
            or record.report.obligations
            for record in self.matrix_reports
        ):
            raise ValueError("all derived matrix reports must be eligible")
        if self.status != "ELIGIBLE" or self.obligations != ():
            raise ValueError("matching-set status is not derived eligibility")
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
            "matching_policy_sha256": self.matching_policy.policy_sha256,
            "optimizer_policy_sha256": (
                self.optimizer_policy.optimizer_policy_sha256
            ),
            "selection_sha256s": tuple(
                selection.selection_sha256
                for selection in self.selections
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
        matching_config_sha256: str,
        selections: tuple[H6FormulaSelection, ...],
        ownership_inventories: tuple[H6MatchingOwnershipRecord, ...],
        matrix_reports: tuple[tuple[ArmMatrixRow, MatchingReport], ...],
    ) -> "H6MatchingSetRecord":
        bound_reports = tuple(
            H6MatrixMatchingReportRecord.create(row=row, report=report)
            for row, report in matrix_reports
        )
        values: dict[str, object] = {
            "schema_version": "h6-matching-set-v2",
            "git_head": git_head,
            "dirty_digest": dirty_digest,
            "source_sha256": source_sha256,
            "matching_config_sha256": matching_config_sha256,
            "matching_policy": AMENDED_MATCHING_SCHEDULE_POLICY,
            "optimizer_policy": H6_ADAMW_POLICY,
            "selections": tuple(selections),
            "ownership_inventories": tuple(ownership_inventories),
            "matrix_reports": bound_reports,
            "arm_matrix_sha256": ARM_MATRIX_SHA256,
            "status": "ELIGIBLE",
            "obligations": (),
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


def _endpoints_payload(record: H6MatchingSetRecord) -> dict[str, object]:
    return {
        "schema_version": "h6-matching-evidence-v2",
        "matching_set_sha256": record.matching_set_sha256,
        "matching_policy": _matching_policy_payload(
            record.matching_policy
        ),
        "optimizer_policy": _optimizer_policy_payload(
            record.optimizer_policy
        ),
        "formula_selections": tuple(
            _selection_payload(selection)
            for selection in record.selections
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
        "matching_policy_sha256": record.matching_policy.policy_sha256,
        "optimizer_policy_sha256": (
            record.optimizer_policy.optimizer_policy_sha256
        ),
        "workload_sha256": record.selections[0].workload.workload_sha256,
        "arm_matrix_sha256": record.arm_matrix_sha256,
        "matching_set_sha256": record.matching_set_sha256,
        "formula_selection_count": len(record.selections),
        "ownership_inventory_count": len(
            record.ownership_inventories
        ),
        "matrix_report_count": len(record.matrix_reports),
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
    matching_config_sha256: str,
    selections: tuple[H6FormulaSelection, ...],
    ownership_inventories: tuple[H6MatchingOwnershipRecord, ...],
    matrix_reports: tuple[tuple[ArmMatrixRow, MatchingReport], ...],
) -> tuple[H6MatchingSetRecord, Path]:
    """Publish an absent-directory v2 artifact after full typed derivation."""

    record = H6MatchingSetRecord.create(
        git_head=git_head,
        dirty_digest=dirty_digest,
        source_sha256=source_sha256,
        matching_config_sha256=matching_config_sha256,
        selections=selections,
        ownership_inventories=ownership_inventories,
        matrix_reports=matrix_reports,
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


def _read_matching_policy(raw: object) -> AmendedMatchingSchedulePolicy:
    if type(raw) is not dict:
        raise ValueError("matching policy must be a JSON object")
    _exact_keys(
        raw,
        set(AmendedMatchingSchedulePolicy.__dataclass_fields__),
        "matching policy",
    )
    if not _same_json_value(
        _matching_policy_payload(AMENDED_MATCHING_SCHEDULE_POLICY), raw
    ):
        raise ValueError("matching policy differs from the exact policy")
    AMENDED_MATCHING_SCHEDULE_POLICY.__post_init__()
    return AMENDED_MATCHING_SCHEDULE_POLICY


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
    _exact_keys(raw, {"row", "report", "record_sha256"}, name)
    result = H6MatrixMatchingReportRecord.create(
        row=_read_row(raw["row"], f"{name}.row"),
        report=_read_matching_report(
            raw["report"], f"{name}.report"
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
    _exact_keys(
        endpoints,
        {
            "schema_version",
            "matching_set_sha256",
            "matching_policy",
            "optimizer_policy",
            "formula_selections",
            "ownership_inventories",
        },
        "matching evidence payload",
    )
    if endpoints["schema_version"] != "h6-matching-evidence-v2":
        raise H6MatchingPublicationBlocked(
            "legacy H6 matching artifacts are not authorized for readiness"
        )
    _read_matching_policy(endpoints["matching_policy"])
    _read_optimizer_policy(endpoints["optimizer_policy"])
    raw_selections = endpoints["formula_selections"]
    raw_ownership = endpoints["ownership_inventories"]
    if type(raw_selections) is not list or type(raw_ownership) is not list:
        raise ValueError(
            "matching selections and ownership must be JSON arrays"
        )
    selections = tuple(
        _read_selection(item, f"formula_selections[{index}]")
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
            "arm_matrix_sha256",
            "matching_set_sha256",
            "formula_selection_count",
            "ownership_inventory_count",
            "matrix_report_count",
            "status",
            "obligations",
        },
        "matching-set validation payload",
    )
    record = H6MatchingSetRecord.create(
        git_head=validation["git_head"],  # type: ignore[arg-type]
        dirty_digest=validation["dirty_digest"],  # type: ignore[arg-type]
        source_sha256=validation["source_sha256"],  # type: ignore[arg-type]
        matching_config_sha256=validation[
            "matching_config_sha256"
        ],  # type: ignore[arg-type]
        selections=selections,
        ownership_inventories=ownership,
        matrix_reports=tuple(
            (item.row, item.report) for item in report_records
        ),
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
    "derive_h6_matrix_reports",
    "publish_h6_matching_set",
    "read_h6_matching_set",
]
