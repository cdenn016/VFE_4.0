"""Canonical H6-Prediction v3 validation and checkpoint-selection artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from vfe4.artifacts.atomic import (
    ArtifactPublicationError,
    canonical_json_bytes as artifact_json_bytes,
    publish_run_directory,
)
from vfe4.training.checkpoint_v3 import (
    H6CheckpointV3,
    decode_h6_checkpoint_v3,
)
from vfe4.training.h6_experiment_v3 import (
    H6_CONFIRMATORY_SEEDS_V3,
    H6_TUNED_ENDPOINT_CONFIG_IDS_V3,
    H6_TUNING_CELLS_V3,
    H6_TUNING_SEEDS_V3,
    H6ExperimentPlanV3,
    H6PlannedAttemptV3,
    H6TuningCellV3,
)
from vfe4.training.h6_matching_v3 import H6_MATCHING_V3_ENDPOINT_CONFIG_IDS
from vfe4.predictive.proposal import CounterPurpose
from vfe4.types.h6 import TrainingPhase, canonical_json_bytes
from vfe4.types.h6_prediction_v3 import (
    H6_PREDICTION_METRICS_SCHEMA,
    H6_PREDICTION_RESULT_SCHEMA,
    H6_RAW_ENDPOINT_INVENTORY_SCHEMA,
    H6_SCORING_INVENTORY_SHA256,
)


_LOWER_HEX = frozenset("0123456789abcdef")
_A5_PRIMARY_ENDPOINT_CONFIG_ID = H6_TUNED_ENDPOINT_CONFIG_IDS_V3[-1]
_BUNDLE_FILENAME = "validation_bundle.json"
_MAXIMUM_BUNDLE_BYTES = 16 * 1024 * 1024


def _build_validation_record_origin_api():
    authority = object()

    class ValidationRecordOrigin:
        __slots__ = ()

        def __new__(cls, supplied_authority: object):
            if supplied_authority is not authority:
                raise TypeError("validation-record origins are scorer-issued only")
            return super().__new__(cls)

    origin = ValidationRecordOrigin(authority)

    def issue() -> object:
        return origin

    def validate(value: object) -> bool:
        return value is origin

    return issue, validate


(
    _issue_validation_record_origin,
    _is_validation_record_origin,
) = _build_validation_record_origin_api()
del _build_validation_record_origin_api


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


def _require_endpoint(value: object, *, tuned_only: bool = False) -> str:
    inventory = (
        H6_TUNED_ENDPOINT_CONFIG_IDS_V3
        if tuned_only
        else H6_MATCHING_V3_ENDPOINT_CONFIG_IDS
    )
    if type(value) is not str or value not in inventory:
        qualifier = "tuned " if tuned_only else ""
        raise ValueError(f"endpoint ID is outside the frozen {qualifier}inventory")
    return value


def _require_plan(plan: object) -> H6ExperimentPlanV3:
    if type(plan) is not H6ExperimentPlanV3:
        raise ValueError("an exact H6 experiment plan v3 is required")
    plan.__post_init__()
    return plan


def _config_sha256_by_endpoint(plan: H6ExperimentPlanV3) -> dict[str, str]:
    return {config.config_id: config.config_sha256 for config in plan.endpoint_configs}


def _attempt_key(
    attempt: H6PlannedAttemptV3,
) -> tuple[str, str | None, int]:
    return (
        attempt.endpoint_config_id,
        None if attempt.tuning_cell is None else attempt.tuning_cell.cell_sha256,
        attempt.training_seed,
    )


def _terminal_phase(attempt: H6PlannedAttemptV3) -> TrainingPhase:
    return (
        TrainingPhase.MODEL_CE_ADAMW
        if attempt.attempt_spec.recognition_factory_sha256 is None
        else TrainingPhase.MODEL_ADAMW
    )


def _expected_module_roots(attempt: H6PlannedAttemptV3) -> tuple[str, ...]:
    return (
        ("model", "recognition")
        if attempt.attempt_spec.recognition_factory_sha256 is not None
        else ("model",)
    )


def _validate_planned_checkpoint_v3(
    *,
    checkpoint: H6CheckpointV3,
    planned_attempt: H6PlannedAttemptV3,
    plan: H6ExperimentPlanV3,
    stage: Literal["tuning", "confirmatory"],
) -> bytes:
    """Validate plan membership plus terminal cursor, state, and encoded bytes."""

    checked_plan = _require_plan(plan)
    if type(planned_attempt) is not H6PlannedAttemptV3:
        raise ValueError("checkpoint binding requires an exact planned attempt")
    planned_attempt.__post_init__()
    if planned_attempt.stage != stage:
        raise ValueError("planned checkpoint stage is incorrect")
    inventory = (
        checked_plan.tuning_attempts
        if stage == "tuning"
        else checked_plan.confirmatory_attempts
    )
    by_sha256 = {attempt.planned_attempt_sha256: attempt for attempt in inventory}
    canonical_attempt = by_sha256.get(planned_attempt.planned_attempt_sha256)
    if canonical_attempt != planned_attempt:
        raise ValueError("planned attempt is not a member of the experiment plan")
    if type(checkpoint) is not H6CheckpointV3:
        raise ValueError("checkpoint binding requires an exact H6 checkpoint v3")
    checkpoint.__post_init__()
    if checkpoint.attempt_spec != planned_attempt.attempt_spec:
        raise ValueError("checkpoint does not bind the planned attempt spec")
    if (
        checkpoint.attempt_spec.experiment_config_sha256
        != checked_plan.experiment_config_sha256
        or checkpoint.attempt_spec.readiness_sha256 != checked_plan.readiness_sha256
        or checkpoint.attempt_spec.endpoint_config_sha256
        != planned_attempt.endpoint_config_sha256
        or checkpoint.attempt_spec.runtime_identity_sha256
        != checked_plan.training_schedule.runtime_identity_sha256
    ):
        raise ValueError("checkpoint plan/config/runtime identity drift")
    if stage == "tuning":
        tuning_cell = planned_attempt.tuning_cell
        if tuning_cell is None or _optimizer_cell(checkpoint) != (
            tuning_cell.learning_rate,
            tuning_cell.weight_decay,
        ):
            raise ValueError(
                "tuning checkpoint optimizer cell does not match the planned cell"
            )
    terminal_phase = _terminal_phase(planned_attempt)
    if (
        checkpoint.objective_manifest.phase is not terminal_phase
        or checkpoint.cursor.pass_index != planned_attempt.terminal_pass_index
        or checkpoint.cursor.batch_index != planned_attempt.terminal_batch_index
        or checkpoint.cursor.example_ordinal != planned_attempt.terminal_example_ordinal
        or checkpoint.cursor.draw_block != planned_attempt.terminal_draw_block
        or checkpoint.cursor.counter_consumption_sha256
        != planned_attempt.terminal_counter_consumption_sha256
        or checkpoint.cursor.permutation_sha256
        != planned_attempt.terminal_permutation_sha256
        or checkpoint.cursor.recognition_update_count
        != planned_attempt.terminal_recognition_update_count
        or checkpoint.cursor.model_update_count
        != planned_attempt.terminal_model_update_count
        or checkpoint.cursor.validation_boundary_count
        != planned_attempt.terminal_validation_boundary_count
        or checkpoint.cursor.checkpoint_boundary_count
        != planned_attempt.terminal_checkpoint_boundary_count
    ):
        raise ValueError("checkpoint does not match the declared terminal boundary")
    expected_roots = _expected_module_roots(planned_attempt)
    module_roots = tuple(
        dict.fromkeys(
            record.name.partition(".")[0] for record in checkpoint.module_tensors
        )
    )
    optimizer_roots = tuple(optimizer.name for optimizer in checkpoint.optimizers)
    if module_roots != expected_roots or optimizer_roots != expected_roots:
        raise ValueError("terminal checkpoint module/optimizer state is incomplete")
    raw = checkpoint.to_bytes()
    reopened = decode_h6_checkpoint_v3(raw)
    if (
        reopened.checkpoint_sha256 != checkpoint.checkpoint_sha256
        or reopened.to_bytes() != raw
    ):
        raise ValueError("terminal checkpoint bytes do not round-trip canonically")
    return raw


def _optimizer_cell(checkpoint: H6CheckpointV3) -> tuple[float, float]:
    observed: set[tuple[float, float]] = set()
    for optimizer in checkpoint.optimizers:
        for group in optimizer.groups:
            hyperparameters = dict(group.hyperparameters)
            learning_rate = hyperparameters.get("lr")
            weight_decay = hyperparameters.get("weight_decay")
            if type(learning_rate) is not float or type(weight_decay) is not float:
                raise ValueError("terminal AdamW group lacks exact tuning values")
            observed.add((learning_rate, weight_decay))
    if len(observed) != 1:
        raise ValueError("terminal checkpoint optimizer tuning cell drift")
    return next(iter(observed))


@dataclass(frozen=True, slots=True, init=False)
class H6ValidationRecordV3:
    """One immutable tuning-checkpoint validation prior-NLL record."""

    record_schema: Literal["h6-validation-record-v3"]
    scoring_method: Literal["target-blind-prefix-prior-nll"]
    scoring_device: Literal["cpu"]
    scoring_dtype: Literal["float64"]
    experiment_config_sha256: str
    plan_sha256: str
    endpoint_config_id: str
    endpoint_config_sha256: str
    tuning_cell: H6TuningCellV3
    training_seed: int
    attempt_spec_sha256: str
    checkpoint_sha256: str
    checkpoint_bytes_sha256: str
    readiness_sha256: str
    matching_set_sha256: str
    data_identity_sha256: str
    runtime_identity_sha256: str
    counted_target_total: int
    total_prior_nll: float
    mean_prior_nll: float
    validation_record_sha256: str
    _scoring_origin: object = field(repr=False, compare=False)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "record_schema": self.record_schema,
            "scoring_method": self.scoring_method,
            "scoring_device": self.scoring_device,
            "scoring_dtype": self.scoring_dtype,
            "experiment_config_sha256": self.experiment_config_sha256,
            "plan_sha256": self.plan_sha256,
            "endpoint_config_id": self.endpoint_config_id,
            "endpoint_config_sha256": self.endpoint_config_sha256,
            "tuning_cell": {
                "learning_rate": self.tuning_cell.learning_rate,
                "weight_decay": self.tuning_cell.weight_decay,
                "cell_sha256": self.tuning_cell.cell_sha256,
            },
            "training_seed": self.training_seed,
            "attempt_spec_sha256": self.attempt_spec_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_bytes_sha256": self.checkpoint_bytes_sha256,
            "readiness_sha256": self.readiness_sha256,
            "matching_set_sha256": self.matching_set_sha256,
            "data_identity_sha256": self.data_identity_sha256,
            "runtime_identity_sha256": self.runtime_identity_sha256,
            "counted_target_total": self.counted_target_total,
            "total_prior_nll": self.total_prior_nll,
            "mean_prior_nll": self.mean_prior_nll,
        }

    def __post_init__(self) -> None:
        if not _is_validation_record_origin(self._scoring_origin):
            raise ValueError("validation record is not scorer-authenticated")
        if (
            self.record_schema != "h6-validation-record-v3"
            or self.scoring_method != "target-blind-prefix-prior-nll"
            or self.scoring_device != "cpu"
            or self.scoring_dtype != "float64"
        ):
            raise ValueError("validation scoring contract is stale")
        _require_endpoint(self.endpoint_config_id, tuned_only=True)
        if type(self.tuning_cell) is not H6TuningCellV3:
            raise ValueError("validation record requires an exact tuning cell")
        self.tuning_cell.__post_init__()
        if self.training_seed not in H6_TUNING_SEEDS_V3:
            raise ValueError("validation record seed is outside the tuning inventory")
        for name in (
            "experiment_config_sha256",
            "plan_sha256",
            "endpoint_config_sha256",
            "attempt_spec_sha256",
            "checkpoint_sha256",
            "checkpoint_bytes_sha256",
            "readiness_sha256",
            "matching_set_sha256",
            "data_identity_sha256",
            "runtime_identity_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if type(self.counted_target_total) is not int or self.counted_target_total <= 0:
            raise ValueError("validation target total must be positive")
        if (
            type(self.total_prior_nll) is not float
            or not math.isfinite(self.total_prior_nll)
            or self.total_prior_nll < 0.0
            or self.mean_prior_nll != self.total_prior_nll / self.counted_target_total
        ):
            raise ValueError("validation prior NLL totals are invalid")
        if self.validation_record_sha256 != _hash(
            "vfe4.h6.validation-record.v3", self.canonical_payload()
        ):
            raise ValueError("validation-record identity is stale")


def _create_h6_validation_record_v3(
    *,
    experiment_config_sha256: str,
    plan_sha256: str,
    endpoint_config_id: str,
    endpoint_config_sha256: str,
    tuning_cell: H6TuningCellV3,
    training_seed: int,
    attempt_spec_sha256: str,
    checkpoint_sha256: str,
    checkpoint_bytes_sha256: str,
    readiness_sha256: str,
    matching_set_sha256: str,
    data_identity_sha256: str,
    runtime_identity_sha256: str,
    counted_target_total: int,
    total_prior_nll: float,
) -> H6ValidationRecordV3:
    """Issue one scorer-authenticated validation record."""

    if type(counted_target_total) is not int or counted_target_total <= 0:
        raise ValueError("validation target total must be positive")
    total = float(total_prior_nll)
    values = {
        "record_schema": "h6-validation-record-v3",
        "scoring_method": "target-blind-prefix-prior-nll",
        "scoring_device": "cpu",
        "scoring_dtype": "float64",
        "experiment_config_sha256": experiment_config_sha256,
        "plan_sha256": plan_sha256,
        "endpoint_config_id": endpoint_config_id,
        "endpoint_config_sha256": endpoint_config_sha256,
        "tuning_cell": tuning_cell,
        "training_seed": training_seed,
        "attempt_spec_sha256": attempt_spec_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_bytes_sha256": checkpoint_bytes_sha256,
        "readiness_sha256": readiness_sha256,
        "matching_set_sha256": matching_set_sha256,
        "data_identity_sha256": data_identity_sha256,
        "runtime_identity_sha256": runtime_identity_sha256,
        "counted_target_total": counted_target_total,
        "total_prior_nll": total,
        "mean_prior_nll": total / counted_target_total,
    }
    payload = {
        **values,
        "tuning_cell": {
            "learning_rate": tuning_cell.learning_rate,
            "weight_decay": tuning_cell.weight_decay,
            "cell_sha256": tuning_cell.cell_sha256,
        },
    }
    record = object.__new__(H6ValidationRecordV3)
    for name, value in (
        *values.items(),
        (
            "validation_record_sha256",
            _hash("vfe4.h6.validation-record.v3", payload),
        ),
        ("_scoring_origin", _issue_validation_record_origin()),
    ):
        object.__setattr__(record, name, value)
    record.__post_init__()
    return record


@dataclass(frozen=True, slots=True)
class H6EndpointTuningSelectionV3:
    endpoint_config_id: str
    endpoint_config_sha256: str
    source_endpoint_config_id: str
    tuning_cell: H6TuningCellV3
    source_validation_record_sha256s: tuple[str, ...]
    endpoint_selection_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "endpoint_config_id": self.endpoint_config_id,
            "endpoint_config_sha256": self.endpoint_config_sha256,
            "source_endpoint_config_id": self.source_endpoint_config_id,
            "tuning_cell": {
                "learning_rate": self.tuning_cell.learning_rate,
                "weight_decay": self.tuning_cell.weight_decay,
                "cell_sha256": self.tuning_cell.cell_sha256,
            },
            "source_validation_record_sha256s": (self.source_validation_record_sha256s),
        }

    def __post_init__(self) -> None:
        _require_endpoint(self.endpoint_config_id)
        _require_endpoint(self.source_endpoint_config_id, tuned_only=True)
        _require_sha256(self.endpoint_config_sha256, "endpoint_config_sha256")
        if type(self.tuning_cell) is not H6TuningCellV3:
            raise ValueError("endpoint selection requires an exact tuning cell")
        self.tuning_cell.__post_init__()
        if (
            type(self.source_validation_record_sha256s) is not tuple
            or len(self.source_validation_record_sha256s) != 12
            or len(set(self.source_validation_record_sha256s)) != 12
        ):
            raise ValueError("endpoint selection must bind all 12 source records")
        for digest in self.source_validation_record_sha256s:
            _require_sha256(digest, "source validation-record SHA-256")
        if self.endpoint_config_id in H6_TUNED_ENDPOINT_CONFIG_IDS_V3:
            if self.source_endpoint_config_id != self.endpoint_config_id:
                raise ValueError("tuned endpoints must select independently")
        elif self.source_endpoint_config_id != _A5_PRIMARY_ENDPOINT_CONFIG_ID:
            raise ValueError("untuned endpoints must inherit the A5-primary cell")
        if self.endpoint_selection_sha256 != _hash(
            "vfe4.h6.endpoint-tuning-selection.v3", self.canonical_payload()
        ):
            raise ValueError("endpoint tuning-selection identity is stale")

    @classmethod
    def create(
        cls,
        *,
        endpoint_config_id: str,
        endpoint_config_sha256: str,
        source_endpoint_config_id: str,
        tuning_cell: H6TuningCellV3,
        source_validation_record_sha256s: tuple[str, ...],
    ) -> "H6EndpointTuningSelectionV3":
        values = {
            "endpoint_config_id": endpoint_config_id,
            "endpoint_config_sha256": endpoint_config_sha256,
            "source_endpoint_config_id": source_endpoint_config_id,
            "tuning_cell": tuning_cell,
            "source_validation_record_sha256s": tuple(source_validation_record_sha256s),
        }
        payload = {
            **values,
            "tuning_cell": {
                "learning_rate": tuning_cell.learning_rate,
                "weight_decay": tuning_cell.weight_decay,
                "cell_sha256": tuning_cell.cell_sha256,
            },
        }
        return cls(
            **values,
            endpoint_selection_sha256=_hash(
                "vfe4.h6.endpoint-tuning-selection.v3", payload
            ),
        )


@dataclass(frozen=True, slots=True)
class H6TuningSelectionV3:
    selection_schema: Literal["h6-tuning-selection-v3"]
    experiment_config_sha256: str
    plan_sha256: str
    readiness_sha256: str
    matching_set_sha256: str
    data_identity_sha256: str
    runtime_identity_sha256: str
    tuning_validation_records: tuple[H6ValidationRecordV3, ...]
    endpoint_selections: tuple[H6EndpointTuningSelectionV3, ...]
    tuning_selection_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "selection_schema": self.selection_schema,
            "experiment_config_sha256": self.experiment_config_sha256,
            "plan_sha256": self.plan_sha256,
            "readiness_sha256": self.readiness_sha256,
            "matching_set_sha256": self.matching_set_sha256,
            "data_identity_sha256": self.data_identity_sha256,
            "runtime_identity_sha256": self.runtime_identity_sha256,
            "tuning_validation_records": tuple(
                record.canonical_payload()
                | {"validation_record_sha256": record.validation_record_sha256}
                for record in self.tuning_validation_records
            ),
            "endpoint_selections": tuple(
                selection.canonical_payload()
                | {"endpoint_selection_sha256": (selection.endpoint_selection_sha256)}
                for selection in self.endpoint_selections
            ),
        }

    def __post_init__(self) -> None:
        if self.selection_schema != "h6-tuning-selection-v3":
            raise ValueError("unsupported tuning-selection schema")
        for name in (
            "experiment_config_sha256",
            "plan_sha256",
            "readiness_sha256",
            "matching_set_sha256",
            "data_identity_sha256",
            "runtime_identity_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if (
            type(self.tuning_validation_records) is not tuple
            or len(self.tuning_validation_records) != 72
            or any(
                type(record) is not H6ValidationRecordV3
                for record in self.tuning_validation_records
            )
        ):
            raise ValueError("tuning selection must publish all 72 validation records")
        if (
            type(self.endpoint_selections) is not tuple
            or tuple(item.endpoint_config_id for item in self.endpoint_selections)
            != H6_MATCHING_V3_ENDPOINT_CONFIG_IDS
        ):
            raise ValueError("tuning selection endpoint inventory is incomplete")
        for record in self.tuning_validation_records:
            record.__post_init__()
        for selection in self.endpoint_selections:
            selection.__post_init__()
        expected_identity = (
            self.experiment_config_sha256,
            self.plan_sha256,
            self.readiness_sha256,
            self.matching_set_sha256,
            self.data_identity_sha256,
            self.runtime_identity_sha256,
        )
        if {
            _record_identities(record) for record in self.tuning_validation_records
        } != {expected_identity}:
            raise ValueError("tuning records do not match selection authority")
        by_key: dict[
            tuple[str, float, float, int],
            H6ValidationRecordV3,
        ] = {}
        for record in self.tuning_validation_records:
            key = (
                record.endpoint_config_id,
                record.tuning_cell.learning_rate,
                record.tuning_cell.weight_decay,
                record.training_seed,
            )
            if key in by_key:
                raise ValueError("tuning record inventory contains duplicates")
            by_key[key] = record
        expected_keys = {
            (endpoint_id, learning_rate, weight_decay, seed)
            for endpoint_id in H6_TUNED_ENDPOINT_CONFIG_IDS_V3
            for learning_rate, weight_decay in H6_TUNING_CELLS_V3
            for seed in H6_TUNING_SEEDS_V3
        }
        if set(by_key) != expected_keys:
            raise ValueError("tuning record inventory is incomplete")
        for endpoint in self.endpoint_selections:
            source_id = endpoint.source_endpoint_config_id
            source_records = tuple(
                by_key[(source_id, learning_rate, weight_decay, seed)]
                for learning_rate, weight_decay in H6_TUNING_CELLS_V3
                for seed in H6_TUNING_SEEDS_V3
            )
            if endpoint.source_validation_record_sha256s != tuple(
                record.validation_record_sha256 for record in source_records
            ):
                raise ValueError("endpoint selection source-record binding changed")
            candidates = tuple(
                (
                    math.fsum(
                        by_key[
                            (source_id, learning_rate, weight_decay, seed)
                        ].mean_prior_nll
                        for seed in H6_TUNING_SEEDS_V3
                    )
                    / len(H6_TUNING_SEEDS_V3),
                    learning_rate,
                    weight_decay,
                )
                for learning_rate, weight_decay in H6_TUNING_CELLS_V3
            )
            winner = min(candidates)
            if (
                endpoint.tuning_cell.learning_rate,
                endpoint.tuning_cell.weight_decay,
            ) != winner[1:]:
                raise ValueError("endpoint selected cell is not the frozen winner")
            if endpoint.endpoint_config_id in H6_TUNED_ENDPOINT_CONFIG_IDS_V3:
                config_sha256s = {
                    record.endpoint_config_sha256 for record in source_records
                }
                if config_sha256s != {endpoint.endpoint_config_sha256}:
                    raise ValueError("tuned endpoint config binding changed")
        attempts_by_endpoint_seed: dict[tuple[str, int], set[str]] = {}
        for record in self.tuning_validation_records:
            attempts_by_endpoint_seed.setdefault(
                (record.endpoint_config_id, record.training_seed),
                set(),
            ).add(record.attempt_spec_sha256)
        if len(attempts_by_endpoint_seed) != 12 or any(
            len(digests) != 1 for digests in attempts_by_endpoint_seed.values()
        ):
            raise ValueError("tuning attempt-spec binding is inconsistent")
        for name in (
            "checkpoint_sha256",
            "checkpoint_bytes_sha256",
            "validation_record_sha256",
        ):
            values = tuple(
                getattr(record, name) for record in self.tuning_validation_records
            )
            if len(set(values)) != 72:
                raise ValueError(f"tuning {name} inventory contains duplicates")
        if self.tuning_selection_sha256 != _hash(
            "vfe4.h6.tuning-selection.v3", self.canonical_payload()
        ):
            raise ValueError("tuning-selection identity is stale")


@dataclass(frozen=True, slots=True)
class H6CheckpointCandidateV3:
    endpoint_config_id: str
    endpoint_config_sha256: str
    tuning_cell: H6TuningCellV3
    training_seed: int
    planned_attempt_sha256: str
    attempt_spec_sha256: str
    checkpoint_sha256: str
    checkpoint_bytes_sha256: str
    experiment_config_sha256: str
    plan_sha256: str
    readiness_sha256: str
    matching_set_sha256: str
    data_identity_sha256: str
    runtime_identity_sha256: str
    candidate_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "endpoint_config_id": self.endpoint_config_id,
            "endpoint_config_sha256": self.endpoint_config_sha256,
            "tuning_cell": {
                "learning_rate": self.tuning_cell.learning_rate,
                "weight_decay": self.tuning_cell.weight_decay,
                "cell_sha256": self.tuning_cell.cell_sha256,
            },
            "training_seed": self.training_seed,
            "planned_attempt_sha256": self.planned_attempt_sha256,
            "attempt_spec_sha256": self.attempt_spec_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_bytes_sha256": self.checkpoint_bytes_sha256,
            "experiment_config_sha256": self.experiment_config_sha256,
            "plan_sha256": self.plan_sha256,
            "readiness_sha256": self.readiness_sha256,
            "matching_set_sha256": self.matching_set_sha256,
            "data_identity_sha256": self.data_identity_sha256,
            "runtime_identity_sha256": self.runtime_identity_sha256,
        }

    def __post_init__(self) -> None:
        _require_endpoint(self.endpoint_config_id)
        if type(self.tuning_cell) is not H6TuningCellV3:
            raise ValueError("checkpoint candidate requires an exact tuning cell")
        self.tuning_cell.__post_init__()
        if self.training_seed not in H6_CONFIRMATORY_SEEDS_V3:
            raise ValueError("checkpoint candidate seed is not confirmatory")
        for name in (
            "endpoint_config_sha256",
            "planned_attempt_sha256",
            "attempt_spec_sha256",
            "checkpoint_sha256",
            "checkpoint_bytes_sha256",
            "experiment_config_sha256",
            "plan_sha256",
            "readiness_sha256",
            "matching_set_sha256",
            "data_identity_sha256",
            "runtime_identity_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if self.candidate_sha256 != _hash(
            "vfe4.h6.checkpoint-candidate.v3", self.canonical_payload()
        ):
            raise ValueError("checkpoint-candidate identity is stale")

    @classmethod
    def create(
        cls,
        *,
        checkpoint: H6CheckpointV3,
        planned_attempt: H6PlannedAttemptV3,
        plan: H6ExperimentPlanV3,
        tuning_selection: H6TuningSelectionV3,
    ) -> "H6CheckpointCandidateV3":
        raw = _validate_planned_checkpoint_v3(
            checkpoint=checkpoint,
            planned_attempt=planned_attempt,
            plan=plan,
            stage="confirmatory",
        )
        if type(tuning_selection) is not H6TuningSelectionV3:
            raise ValueError("candidate requires an exact tuning selection")
        tuning_selection.__post_init__()
        if (
            tuning_selection.plan_sha256 != plan.plan_sha256
            or tuning_selection.experiment_config_sha256
            != plan.experiment_config_sha256
        ):
            raise ValueError("candidate tuning/plan identity drift")
        selected = {
            item.endpoint_config_id: item
            for item in tuning_selection.endpoint_selections
        }[planned_attempt.endpoint_config_id]
        if planned_attempt.tuning_cell_source != (
            f"selected:{selected.source_endpoint_config_id}"
        ):
            raise ValueError("confirmatory planned-attempt selection source drift")
        if _optimizer_cell(checkpoint) != (
            selected.tuning_cell.learning_rate,
            selected.tuning_cell.weight_decay,
        ):
            raise ValueError("terminal checkpoint does not use the selected cell")
        values = {
            "endpoint_config_id": planned_attempt.endpoint_config_id,
            "endpoint_config_sha256": planned_attempt.endpoint_config_sha256,
            "tuning_cell": selected.tuning_cell,
            "training_seed": planned_attempt.training_seed,
            "planned_attempt_sha256": planned_attempt.planned_attempt_sha256,
            "attempt_spec_sha256": planned_attempt.attempt_spec.attempt_spec_sha256,
            "checkpoint_sha256": checkpoint.checkpoint_sha256,
            "checkpoint_bytes_sha256": hashlib.sha256(raw).hexdigest(),
            "experiment_config_sha256": plan.experiment_config_sha256,
            "plan_sha256": plan.plan_sha256,
            "readiness_sha256": plan.readiness_sha256,
            "matching_set_sha256": plan.matching_set_sha256,
            "data_identity_sha256": planned_attempt.attempt_spec.data_identity_sha256,
            "runtime_identity_sha256": (plan.training_schedule.runtime_identity_sha256),
        }
        payload = {
            **values,
            "tuning_cell": {
                "learning_rate": selected.tuning_cell.learning_rate,
                "weight_decay": selected.tuning_cell.weight_decay,
                "cell_sha256": selected.tuning_cell.cell_sha256,
            },
        }
        return cls(
            **values,  # type: ignore[arg-type]
            candidate_sha256=_hash("vfe4.h6.checkpoint-candidate.v3", payload),
        )


@dataclass(frozen=True, slots=True)
class H6CheckpointSelectionV3:
    selection_schema: Literal["h6-checkpoint-selection-v3"]
    experiment_config_sha256: str
    plan_sha256: str
    tuning_selection_sha256: str
    readiness_sha256: str
    matching_set_sha256: str
    data_identity_sha256: str
    runtime_identity_sha256: str
    checkpoints: tuple[H6CheckpointCandidateV3, ...]
    checkpoint_selection_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "selection_schema": self.selection_schema,
            "experiment_config_sha256": self.experiment_config_sha256,
            "plan_sha256": self.plan_sha256,
            "tuning_selection_sha256": self.tuning_selection_sha256,
            "readiness_sha256": self.readiness_sha256,
            "matching_set_sha256": self.matching_set_sha256,
            "data_identity_sha256": self.data_identity_sha256,
            "runtime_identity_sha256": self.runtime_identity_sha256,
            "checkpoints": tuple(
                checkpoint.canonical_payload()
                | {"candidate_sha256": checkpoint.candidate_sha256}
                for checkpoint in self.checkpoints
            ),
        }

    def __post_init__(self) -> None:
        if self.selection_schema != "h6-checkpoint-selection-v3":
            raise ValueError("unsupported checkpoint-selection schema")
        for name in (
            "experiment_config_sha256",
            "plan_sha256",
            "tuning_selection_sha256",
            "readiness_sha256",
            "matching_set_sha256",
            "data_identity_sha256",
            "runtime_identity_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if (
            type(self.checkpoints) is not tuple
            or len(self.checkpoints) != 96
            or any(
                type(checkpoint) is not H6CheckpointCandidateV3
                for checkpoint in self.checkpoints
            )
        ):
            raise ValueError("checkpoint selection requires the exact 96-row inventory")
        expected = tuple(
            (endpoint_id, seed)
            for endpoint_id in H6_MATCHING_V3_ENDPOINT_CONFIG_IDS
            for seed in H6_CONFIRMATORY_SEEDS_V3
        )
        if (
            tuple(
                (item.endpoint_config_id, item.training_seed)
                for item in self.checkpoints
            )
            != expected
        ):
            raise ValueError("checkpoint selection inventory is incomplete")
        for checkpoint in self.checkpoints:
            checkpoint.__post_init__()
        expected_identity = (
            self.experiment_config_sha256,
            self.plan_sha256,
            self.readiness_sha256,
            self.matching_set_sha256,
            self.data_identity_sha256,
            self.runtime_identity_sha256,
        )
        if {_record_identities(checkpoint) for checkpoint in self.checkpoints} != {
            expected_identity
        }:
            raise ValueError("checkpoint candidates do not match selection authority")
        for name in (
            "planned_attempt_sha256",
            "attempt_spec_sha256",
            "checkpoint_sha256",
            "checkpoint_bytes_sha256",
            "candidate_sha256",
        ):
            values = tuple(getattr(checkpoint, name) for checkpoint in self.checkpoints)
            if len(set(values)) != 96:
                raise ValueError(
                    f"checkpoint {name} inventory must contain 96 unique values"
                )
        if self.checkpoint_selection_sha256 != _hash(
            "vfe4.h6.checkpoint-selection.v3", self.canonical_payload()
        ):
            raise ValueError("checkpoint-selection identity is stale")


@dataclass(frozen=True, slots=True)
class H6ValidationBundleV3:
    bundle_schema: Literal["h6-validation-bundle-v3"]
    experiment_config_sha256: str
    plan_sha256: str
    tuning_selection: H6TuningSelectionV3
    checkpoint_selection: H6CheckpointSelectionV3
    validation_bundle_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "bundle_schema": self.bundle_schema,
            "experiment_config_sha256": self.experiment_config_sha256,
            "plan_sha256": self.plan_sha256,
            "tuning_selection": self.tuning_selection.canonical_payload()
            | {
                "tuning_selection_sha256": (
                    self.tuning_selection.tuning_selection_sha256
                )
            },
            "checkpoint_selection": self.checkpoint_selection.canonical_payload()
            | {
                "checkpoint_selection_sha256": (
                    self.checkpoint_selection.checkpoint_selection_sha256
                )
            },
        }

    def __post_init__(self) -> None:
        if self.bundle_schema != "h6-validation-bundle-v3":
            raise ValueError("unsupported validation-bundle schema")
        _require_sha256(self.experiment_config_sha256, "experiment_config_sha256")
        _require_sha256(self.plan_sha256, "plan_sha256")
        if (
            type(self.tuning_selection) is not H6TuningSelectionV3
            or type(self.checkpoint_selection) is not H6CheckpointSelectionV3
        ):
            raise ValueError("validation bundle requires exact selection records")
        self.tuning_selection.__post_init__()
        self.checkpoint_selection.__post_init__()
        if (
            self.tuning_selection.experiment_config_sha256
            != self.experiment_config_sha256
            or self.checkpoint_selection.experiment_config_sha256
            != self.experiment_config_sha256
            or self.tuning_selection.plan_sha256 != self.plan_sha256
            or self.checkpoint_selection.plan_sha256 != self.plan_sha256
            or self.checkpoint_selection.tuning_selection_sha256
            != self.tuning_selection.tuning_selection_sha256
            or (
                self.checkpoint_selection.readiness_sha256,
                self.checkpoint_selection.matching_set_sha256,
                self.checkpoint_selection.data_identity_sha256,
                self.checkpoint_selection.runtime_identity_sha256,
            )
            != (
                self.tuning_selection.readiness_sha256,
                self.tuning_selection.matching_set_sha256,
                self.tuning_selection.data_identity_sha256,
                self.tuning_selection.runtime_identity_sha256,
            )
        ):
            raise ValueError("validation bundle selection identity drift")
        selected = {
            item.endpoint_config_id: item
            for item in self.tuning_selection.endpoint_selections
        }
        for candidate in self.checkpoint_selection.checkpoints:
            endpoint = selected[candidate.endpoint_config_id]
            if (
                candidate.endpoint_config_sha256 != endpoint.endpoint_config_sha256
                or candidate.tuning_cell != endpoint.tuning_cell
            ):
                raise ValueError(
                    "checkpoint candidate does not match its tuning selection"
                )
        if self.validation_bundle_sha256 != _hash(
            "vfe4.h6.validation-bundle.v3", self.canonical_payload()
        ):
            raise ValueError("validation-bundle identity is stale")

    @classmethod
    def create(
        cls,
        *,
        plan: H6ExperimentPlanV3,
        tuning_selection: H6TuningSelectionV3,
        checkpoint_selection: H6CheckpointSelectionV3,
    ) -> "H6ValidationBundleV3":
        checked_plan = _require_plan(plan)
        tuning_selection.__post_init__()
        checkpoint_selection.__post_init__()
        config_sha256s = _config_sha256_by_endpoint(checked_plan)
        if {
            item.endpoint_config_id: item.endpoint_config_sha256
            for item in tuning_selection.endpoint_selections
        } != config_sha256s:
            raise ValueError("validation bundle endpoint config binding changed")
        values = {
            "bundle_schema": "h6-validation-bundle-v3",
            "experiment_config_sha256": checked_plan.experiment_config_sha256,
            "plan_sha256": checked_plan.plan_sha256,
            "tuning_selection": tuning_selection,
            "checkpoint_selection": checkpoint_selection,
        }
        payload = {
            "bundle_schema": values["bundle_schema"],
            "experiment_config_sha256": values["experiment_config_sha256"],
            "plan_sha256": values["plan_sha256"],
            "tuning_selection": tuning_selection.canonical_payload()
            | {"tuning_selection_sha256": (tuning_selection.tuning_selection_sha256)},
            "checkpoint_selection": checkpoint_selection.canonical_payload()
            | {
                "checkpoint_selection_sha256": (
                    checkpoint_selection.checkpoint_selection_sha256
                )
            },
        }
        return cls(
            **values,  # type: ignore[arg-type]
            validation_bundle_sha256=_hash("vfe4.h6.validation-bundle.v3", payload),
        )

    def artifact_payload(self) -> dict[str, object]:
        self.__post_init__()
        return self.canonical_payload() | {
            "validation_bundle_sha256": self.validation_bundle_sha256
        }


def _record_identities(
    record: H6ValidationRecordV3 | H6CheckpointCandidateV3,
) -> tuple[str, str, str, str, str, str]:
    return (
        record.experiment_config_sha256,
        record.plan_sha256,
        record.readiness_sha256,
        record.matching_set_sha256,
        record.data_identity_sha256,
        record.runtime_identity_sha256,
    )


def select_h6_tuning_v3(
    records: tuple[H6ValidationRecordV3, ...],
    plan: H6ExperimentPlanV3,
) -> H6TuningSelectionV3:
    """Apply the frozen two-seed mean and `(NLL, lr, wd)` tie-break."""

    checked_plan = _require_plan(plan)
    if (
        type(records) is not tuple
        or len(records) != 72
        or any(type(record) is not H6ValidationRecordV3 for record in records)
    ):
        raise ValueError("tuning selection requires all 72 validation records")
    for record in records:
        record.__post_init__()
    expected_identity = (
        checked_plan.experiment_config_sha256,
        checked_plan.plan_sha256,
        checked_plan.readiness_sha256,
        checked_plan.matching_set_sha256,
        checked_plan.tuning_attempts[0].attempt_spec.data_identity_sha256,
        checked_plan.training_schedule.runtime_identity_sha256,
    )
    if {_record_identities(record) for record in records} != {expected_identity}:
        raise ValueError("tuning validation plan/config/data/runtime identity drift")
    config_sha256s = _config_sha256_by_endpoint(checked_plan)
    attempts = {
        _attempt_key(attempt): attempt for attempt in checked_plan.tuning_attempts
    }
    by_key: dict[tuple[str, str, int], H6ValidationRecordV3] = {}
    for record in records:
        key = (
            record.endpoint_config_id,
            record.tuning_cell.cell_sha256,
            record.training_seed,
        )
        attempt = attempts.get(key)
        if (
            attempt is None
            or record.endpoint_config_sha256
            != config_sha256s[record.endpoint_config_id]
            or record.attempt_spec_sha256 != attempt.attempt_spec.attempt_spec_sha256
            or key in by_key
        ):
            raise ValueError("tuning validation plan/attempt inventory drift")
        by_key[key] = record
    cells = checked_plan.tuning_cells
    expected_keys = {
        (endpoint_id, cell.cell_sha256, seed)
        for endpoint_id in H6_TUNED_ENDPOINT_CONFIG_IDS_V3
        for cell in cells
        for seed in H6_TUNING_SEEDS_V3
    }
    if set(by_key) != expected_keys:
        raise ValueError("tuning endpoint/cell/seed inventory is incomplete")

    source_selections: dict[str, tuple[H6TuningCellV3, tuple[str, ...]]] = {}
    for endpoint_id in H6_TUNED_ENDPOINT_CONFIG_IDS_V3:
        candidates: list[tuple[float, float, float, H6TuningCellV3]] = []
        for cell in cells:
            mean = math.fsum(
                by_key[(endpoint_id, cell.cell_sha256, seed)].mean_prior_nll
                for seed in H6_TUNING_SEEDS_V3
            ) / len(H6_TUNING_SEEDS_V3)
            candidates.append((mean, cell.learning_rate, cell.weight_decay, cell))
        selected_cell = min(candidates, key=lambda item: item[:3])[3]
        source_records = tuple(
            by_key[(endpoint_id, cell.cell_sha256, seed)]
            for cell in cells
            for seed in H6_TUNING_SEEDS_V3
        )
        source_selections[endpoint_id] = (
            selected_cell,
            tuple(item.validation_record_sha256 for item in source_records),
        )

    endpoint_selections: list[H6EndpointTuningSelectionV3] = []
    for endpoint_id in H6_MATCHING_V3_ENDPOINT_CONFIG_IDS:
        source_id = (
            endpoint_id
            if endpoint_id in H6_TUNED_ENDPOINT_CONFIG_IDS_V3
            else _A5_PRIMARY_ENDPOINT_CONFIG_ID
        )
        cell, source_records = source_selections[source_id]
        endpoint_selections.append(
            H6EndpointTuningSelectionV3.create(
                endpoint_config_id=endpoint_id,
                endpoint_config_sha256=config_sha256s[endpoint_id],
                source_endpoint_config_id=source_id,
                tuning_cell=cell,
                source_validation_record_sha256s=source_records,
            )
        )
    ordered_records = tuple(
        by_key[(endpoint_id, cell.cell_sha256, seed)]
        for endpoint_id in H6_TUNED_ENDPOINT_CONFIG_IDS_V3
        for cell in cells
        for seed in H6_TUNING_SEEDS_V3
    )
    values = {
        "selection_schema": "h6-tuning-selection-v3",
        "experiment_config_sha256": expected_identity[0],
        "plan_sha256": expected_identity[1],
        "readiness_sha256": expected_identity[2],
        "matching_set_sha256": expected_identity[3],
        "data_identity_sha256": expected_identity[4],
        "runtime_identity_sha256": expected_identity[5],
        "tuning_validation_records": ordered_records,
        "endpoint_selections": tuple(endpoint_selections),
    }
    payload = {
        **{
            name: value
            for name, value in values.items()
            if name
            not in {
                "tuning_validation_records",
                "endpoint_selections",
            }
        },
        "tuning_validation_records": tuple(
            record.canonical_payload()
            | {"validation_record_sha256": record.validation_record_sha256}
            for record in ordered_records
        ),
        "endpoint_selections": tuple(
            selection.canonical_payload()
            | {"endpoint_selection_sha256": selection.endpoint_selection_sha256}
            for selection in endpoint_selections
        ),
    }
    return H6TuningSelectionV3(
        **values,  # type: ignore[arg-type]
        tuning_selection_sha256=_hash("vfe4.h6.tuning-selection.v3", payload),
    )


def bind_h6_checkpoint_selection_v3(
    checkpoint_bindings: tuple[tuple[H6PlannedAttemptV3, H6CheckpointV3], ...],
    plan: H6ExperimentPlanV3,
    tuning_selection: H6TuningSelectionV3,
) -> H6CheckpointSelectionV3:
    """Validate and bind the exact 12-by-8 terminal checkpoint inventory."""

    checked_plan = _require_plan(plan)
    if (
        type(checkpoint_bindings) is not tuple
        or len(checkpoint_bindings) != 96
        or any(
            type(item) is not tuple or len(item) != 2 for item in checkpoint_bindings
        )
    ):
        raise ValueError("checkpoint selection requires 96 planned checkpoint pairs")
    if type(tuning_selection) is not H6TuningSelectionV3:
        raise ValueError("checkpoint selection requires exact tuning selection")
    tuning_selection.__post_init__()
    if (
        tuning_selection.plan_sha256 != checked_plan.plan_sha256
        or tuning_selection.experiment_config_sha256
        != checked_plan.experiment_config_sha256
    ):
        raise ValueError("checkpoint selection tuning/plan identity drift")
    by_key: dict[tuple[str, int], H6CheckpointCandidateV3] = {}
    for planned_attempt, checkpoint in checkpoint_bindings:
        candidate = H6CheckpointCandidateV3.create(
            checkpoint=checkpoint,
            planned_attempt=planned_attempt,
            plan=checked_plan,
            tuning_selection=tuning_selection,
        )
        key = (candidate.endpoint_config_id, candidate.training_seed)
        if key in by_key:
            raise ValueError("checkpoint selection inventory contains duplicates")
        by_key[key] = candidate
    expected_keys = {
        (endpoint_id, seed)
        for endpoint_id in H6_MATCHING_V3_ENDPOINT_CONFIG_IDS
        for seed in H6_CONFIRMATORY_SEEDS_V3
    }
    if set(by_key) != expected_keys:
        raise ValueError("checkpoint endpoint/seed inventory is incomplete")
    ordered = tuple(
        by_key[(endpoint_id, seed)]
        for endpoint_id in H6_MATCHING_V3_ENDPOINT_CONFIG_IDS
        for seed in H6_CONFIRMATORY_SEEDS_V3
    )
    if len({item.checkpoint_sha256 for item in ordered}) != 96:
        raise ValueError("confirmatory checkpoints must have unique identities")
    values = {
        "selection_schema": "h6-checkpoint-selection-v3",
        "experiment_config_sha256": checked_plan.experiment_config_sha256,
        "plan_sha256": checked_plan.plan_sha256,
        "tuning_selection_sha256": tuning_selection.tuning_selection_sha256,
        "readiness_sha256": tuning_selection.readiness_sha256,
        "matching_set_sha256": tuning_selection.matching_set_sha256,
        "data_identity_sha256": tuning_selection.data_identity_sha256,
        "runtime_identity_sha256": tuning_selection.runtime_identity_sha256,
        "checkpoints": ordered,
    }
    payload = {
        **{name: value for name, value in values.items() if name != "checkpoints"},
        "checkpoints": tuple(
            candidate.canonical_payload()
            | {"candidate_sha256": candidate.candidate_sha256}
            for candidate in ordered
        ),
    }
    return H6CheckpointSelectionV3(
        **values,  # type: ignore[arg-type]
        checkpoint_selection_sha256=_hash("vfe4.h6.checkpoint-selection.v3", payload),
    )


def publish_h6_validation_bundle_v3(
    *,
    run_root: Path,
    run_name: str,
    bundle: H6ValidationBundleV3,
) -> Path:
    """Publish the complete validation trace by one no-replace directory commit."""

    if type(bundle) is not H6ValidationBundleV3:
        raise ArtifactPublicationError("an exact validation bundle is required")
    bundle.__post_init__()
    return publish_run_directory(
        run_root,
        run_name,
        {_BUNDLE_FILENAME: bundle.artifact_payload()},
    )


def _mapping(
    value: object,
    name: str,
    expected_keys: frozenset[str],
) -> Mapping[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise ArtifactPublicationError(f"{name} must be one JSON object")
    if frozenset(value) != expected_keys:
        raise ArtifactPublicationError(f"{name} has an unexpected field inventory")
    return value


def _is_redirect(path: Path, status: os.stat_result) -> bool:
    if stat.S_ISLNK(status.st_mode):
        return True
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if getattr(status, "st_file_attributes", 0) & reparse_flag:
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(callable(is_junction) and is_junction())


def _read_bounded_regular_file_once(
    path: Path,
    *,
    maximum_bytes: int,
    label: str,
) -> bytes:
    """Read one bounded non-redirected file through one verified handle."""

    try:
        parent_before = os.lstat(path.parent)
        path_before = os.lstat(path)
    except OSError as exc:
        raise ArtifactPublicationError(f"{label} is unavailable") from exc
    if (
        not stat.S_ISDIR(parent_before.st_mode)
        or _is_redirect(path.parent, parent_before)
        or not stat.S_ISREG(path_before.st_mode)
        or _is_redirect(path, path_before)
        or path_before.st_size > maximum_bytes
    ):
        raise ArtifactPublicationError(
            f"{label} must be a bounded non-redirected regular file"
        )

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ArtifactPublicationError(f"{label} cannot be opened safely") from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino, opened.st_size)
            != (path_before.st_dev, path_before.st_ino, path_before.st_size)
            or opened.st_size > maximum_bytes
        ):
            raise ArtifactPublicationError(f"{label} identity changed before reading")
        chunks: list[bytes] = []
        total = 0
        while total <= maximum_bytes:
            chunk = os.read(
                descriptor,
                min(65_536, maximum_bytes + 1 - total),
            )
            if not chunk:
                break
            total += len(chunk)
            chunks.append(chunk)
        handle_after = os.fstat(descriptor)
        path_after = os.lstat(path)
        parent_after = os.lstat(path.parent)
        if (
            total > maximum_bytes
            or _is_redirect(path, path_after)
            or (handle_after.st_dev, handle_after.st_ino, handle_after.st_size)
            != (opened.st_dev, opened.st_ino, opened.st_size)
            or (path_after.st_dev, path_after.st_ino, path_after.st_size)
            != (opened.st_dev, opened.st_ino, opened.st_size)
            or (parent_after.st_dev, parent_after.st_ino)
            != (parent_before.st_dev, parent_before.st_ino)
        ):
            raise ArtifactPublicationError(f"{label} changed while reading")
    except ArtifactPublicationError:
        raise
    except OSError as exc:
        raise ArtifactPublicationError(f"{label} cannot be read safely") from exc
    finally:
        os.close(descriptor)
    content = b"".join(chunks)
    if len(content) != opened.st_size:
        raise ArtifactPublicationError(f"{label} length changed while reading")
    return content


def _cell_from_payload(value: object) -> H6TuningCellV3:
    payload = _mapping(
        value,
        "tuning cell",
        frozenset({"learning_rate", "weight_decay", "cell_sha256"}),
    )
    try:
        cell = H6TuningCellV3.create(
            learning_rate=payload["learning_rate"],  # type: ignore[arg-type]
            weight_decay=payload["weight_decay"],  # type: ignore[arg-type]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactPublicationError(
            "validation bundle tuning cell is invalid"
        ) from exc
    if payload.get("cell_sha256") != cell.cell_sha256:
        raise ArtifactPublicationError("validation bundle tuning-cell identity drift")
    return cell


def _validation_record_from_payload(value: object) -> H6ValidationRecordV3:
    payload = _mapping(
        value,
        "validation record",
        frozenset(
            {
                "record_schema",
                "scoring_method",
                "scoring_device",
                "scoring_dtype",
                "experiment_config_sha256",
                "plan_sha256",
                "endpoint_config_id",
                "endpoint_config_sha256",
                "tuning_cell",
                "training_seed",
                "attempt_spec_sha256",
                "checkpoint_sha256",
                "checkpoint_bytes_sha256",
                "readiness_sha256",
                "matching_set_sha256",
                "data_identity_sha256",
                "runtime_identity_sha256",
                "counted_target_total",
                "total_prior_nll",
                "mean_prior_nll",
                "validation_record_sha256",
            }
        ),
    )
    try:
        record = _create_h6_validation_record_v3(
            experiment_config_sha256=payload["experiment_config_sha256"],  # type: ignore[arg-type]
            plan_sha256=payload["plan_sha256"],  # type: ignore[arg-type]
            endpoint_config_id=payload["endpoint_config_id"],  # type: ignore[arg-type]
            endpoint_config_sha256=payload["endpoint_config_sha256"],  # type: ignore[arg-type]
            tuning_cell=_cell_from_payload(payload["tuning_cell"]),
            training_seed=payload["training_seed"],  # type: ignore[arg-type]
            attempt_spec_sha256=payload["attempt_spec_sha256"],  # type: ignore[arg-type]
            checkpoint_sha256=payload["checkpoint_sha256"],  # type: ignore[arg-type]
            checkpoint_bytes_sha256=payload["checkpoint_bytes_sha256"],  # type: ignore[arg-type]
            readiness_sha256=payload["readiness_sha256"],  # type: ignore[arg-type]
            matching_set_sha256=payload["matching_set_sha256"],  # type: ignore[arg-type]
            data_identity_sha256=payload["data_identity_sha256"],  # type: ignore[arg-type]
            runtime_identity_sha256=payload["runtime_identity_sha256"],  # type: ignore[arg-type]
            counted_target_total=payload["counted_target_total"],  # type: ignore[arg-type]
            total_prior_nll=payload["total_prior_nll"],  # type: ignore[arg-type]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactPublicationError("validation record cannot be reopened") from exc
    if artifact_json_bytes(dict(payload)) != artifact_json_bytes(
        record.canonical_payload()
        | {"validation_record_sha256": record.validation_record_sha256}
    ):
        raise ArtifactPublicationError("validation-record identity changed")
    return record


def _endpoint_selection_from_payload(
    value: object,
) -> H6EndpointTuningSelectionV3:
    payload = _mapping(
        value,
        "endpoint selection",
        frozenset(
            {
                "endpoint_config_id",
                "endpoint_config_sha256",
                "source_endpoint_config_id",
                "tuning_cell",
                "source_validation_record_sha256s",
                "endpoint_selection_sha256",
            }
        ),
    )
    try:
        selection = H6EndpointTuningSelectionV3.create(
            endpoint_config_id=payload["endpoint_config_id"],  # type: ignore[arg-type]
            endpoint_config_sha256=payload["endpoint_config_sha256"],  # type: ignore[arg-type]
            source_endpoint_config_id=payload["source_endpoint_config_id"],  # type: ignore[arg-type]
            tuning_cell=_cell_from_payload(payload["tuning_cell"]),
            source_validation_record_sha256s=tuple(
                payload["source_validation_record_sha256s"]  # type: ignore[arg-type]
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactPublicationError("endpoint selection cannot be reopened") from exc
    if artifact_json_bytes(dict(payload)) != artifact_json_bytes(
        selection.canonical_payload()
        | {"endpoint_selection_sha256": selection.endpoint_selection_sha256}
    ):
        raise ArtifactPublicationError("endpoint-selection identity changed")
    return selection


def _tuning_selection_from_payload(value: object) -> H6TuningSelectionV3:
    payload = _mapping(
        value,
        "tuning selection",
        frozenset(
            {
                "selection_schema",
                "experiment_config_sha256",
                "plan_sha256",
                "readiness_sha256",
                "matching_set_sha256",
                "data_identity_sha256",
                "runtime_identity_sha256",
                "tuning_validation_records",
                "endpoint_selections",
                "tuning_selection_sha256",
            }
        ),
    )
    try:
        records = tuple(
            _validation_record_from_payload(item)
            for item in payload["tuning_validation_records"]  # type: ignore[union-attr]
        )
        endpoints = tuple(
            _endpoint_selection_from_payload(item)
            for item in payload["endpoint_selections"]  # type: ignore[union-attr]
        )
        selection = H6TuningSelectionV3(
            selection_schema=payload["selection_schema"],  # type: ignore[arg-type]
            experiment_config_sha256=payload["experiment_config_sha256"],  # type: ignore[arg-type]
            plan_sha256=payload["plan_sha256"],  # type: ignore[arg-type]
            readiness_sha256=payload["readiness_sha256"],  # type: ignore[arg-type]
            matching_set_sha256=payload["matching_set_sha256"],  # type: ignore[arg-type]
            data_identity_sha256=payload["data_identity_sha256"],  # type: ignore[arg-type]
            runtime_identity_sha256=payload["runtime_identity_sha256"],  # type: ignore[arg-type]
            tuning_validation_records=records,
            endpoint_selections=endpoints,
            tuning_selection_sha256=payload["tuning_selection_sha256"],  # type: ignore[arg-type]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactPublicationError("tuning selection cannot be reopened") from exc
    if artifact_json_bytes(dict(payload)) != artifact_json_bytes(
        selection.canonical_payload()
        | {"tuning_selection_sha256": selection.tuning_selection_sha256}
    ):
        raise ArtifactPublicationError("tuning-selection payload changed")
    return selection


def _candidate_from_payload(value: object) -> H6CheckpointCandidateV3:
    payload = _mapping(
        value,
        "checkpoint candidate",
        frozenset(
            {
                "endpoint_config_id",
                "endpoint_config_sha256",
                "tuning_cell",
                "training_seed",
                "planned_attempt_sha256",
                "attempt_spec_sha256",
                "checkpoint_sha256",
                "checkpoint_bytes_sha256",
                "experiment_config_sha256",
                "plan_sha256",
                "readiness_sha256",
                "matching_set_sha256",
                "data_identity_sha256",
                "runtime_identity_sha256",
                "candidate_sha256",
            }
        ),
    )
    try:
        candidate = H6CheckpointCandidateV3(
            endpoint_config_id=payload["endpoint_config_id"],  # type: ignore[arg-type]
            endpoint_config_sha256=payload["endpoint_config_sha256"],  # type: ignore[arg-type]
            tuning_cell=_cell_from_payload(payload["tuning_cell"]),
            training_seed=payload["training_seed"],  # type: ignore[arg-type]
            planned_attempt_sha256=payload["planned_attempt_sha256"],  # type: ignore[arg-type]
            attempt_spec_sha256=payload["attempt_spec_sha256"],  # type: ignore[arg-type]
            checkpoint_sha256=payload["checkpoint_sha256"],  # type: ignore[arg-type]
            checkpoint_bytes_sha256=payload["checkpoint_bytes_sha256"],  # type: ignore[arg-type]
            experiment_config_sha256=payload["experiment_config_sha256"],  # type: ignore[arg-type]
            plan_sha256=payload["plan_sha256"],  # type: ignore[arg-type]
            readiness_sha256=payload["readiness_sha256"],  # type: ignore[arg-type]
            matching_set_sha256=payload["matching_set_sha256"],  # type: ignore[arg-type]
            data_identity_sha256=payload["data_identity_sha256"],  # type: ignore[arg-type]
            runtime_identity_sha256=payload["runtime_identity_sha256"],  # type: ignore[arg-type]
            candidate_sha256=payload["candidate_sha256"],  # type: ignore[arg-type]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactPublicationError(
            "checkpoint candidate cannot be reopened"
        ) from exc
    if artifact_json_bytes(dict(payload)) != artifact_json_bytes(
        candidate.canonical_payload() | {"candidate_sha256": candidate.candidate_sha256}
    ):
        raise ArtifactPublicationError("checkpoint-candidate payload changed")
    return candidate


def _checkpoint_selection_from_payload(value: object) -> H6CheckpointSelectionV3:
    payload = _mapping(
        value,
        "checkpoint selection",
        frozenset(
            {
                "selection_schema",
                "experiment_config_sha256",
                "plan_sha256",
                "tuning_selection_sha256",
                "readiness_sha256",
                "matching_set_sha256",
                "data_identity_sha256",
                "runtime_identity_sha256",
                "checkpoints",
                "checkpoint_selection_sha256",
            }
        ),
    )
    try:
        selection = H6CheckpointSelectionV3(
            selection_schema=payload["selection_schema"],  # type: ignore[arg-type]
            experiment_config_sha256=payload["experiment_config_sha256"],  # type: ignore[arg-type]
            plan_sha256=payload["plan_sha256"],  # type: ignore[arg-type]
            tuning_selection_sha256=payload["tuning_selection_sha256"],  # type: ignore[arg-type]
            readiness_sha256=payload["readiness_sha256"],  # type: ignore[arg-type]
            matching_set_sha256=payload["matching_set_sha256"],  # type: ignore[arg-type]
            data_identity_sha256=payload["data_identity_sha256"],  # type: ignore[arg-type]
            runtime_identity_sha256=payload["runtime_identity_sha256"],  # type: ignore[arg-type]
            checkpoints=tuple(
                _candidate_from_payload(item)
                for item in payload["checkpoints"]  # type: ignore[union-attr]
            ),
            checkpoint_selection_sha256=payload["checkpoint_selection_sha256"],  # type: ignore[arg-type]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactPublicationError(
            "checkpoint selection cannot be reopened"
        ) from exc
    if artifact_json_bytes(dict(payload)) != artifact_json_bytes(
        selection.canonical_payload()
        | {"checkpoint_selection_sha256": (selection.checkpoint_selection_sha256)}
    ):
        raise ArtifactPublicationError("checkpoint-selection payload changed")
    return selection


def read_h6_validation_bundle_v3(
    run_directory: Path,
    *,
    expected_plan_sha256: str,
    expected_experiment_config_sha256: str,
    expected_validation_bundle_sha256: str,
) -> H6ValidationBundleV3:
    """Authenticate and reconstruct one complete immutable validation bundle."""

    _require_sha256(expected_plan_sha256, "expected_plan_sha256")
    _require_sha256(
        expected_experiment_config_sha256,
        "expected_experiment_config_sha256",
    )
    _require_sha256(
        expected_validation_bundle_sha256,
        "expected_validation_bundle_sha256",
    )
    if not isinstance(run_directory, Path) or not run_directory.is_absolute():
        raise ArtifactPublicationError("run_directory must be an absolute Path")

    try:
        root_status = os.lstat(run_directory)
    except OSError as exc:
        raise ArtifactPublicationError(
            "validation run directory is unavailable"
        ) from exc
    if not stat.S_ISDIR(root_status.st_mode) or _is_redirect(
        run_directory, root_status
    ):
        raise ArtifactPublicationError("validation run directory is unavailable")
    try:
        children = tuple(run_directory.iterdir())
    except OSError as exc:
        raise ArtifactPublicationError(
            "validation run inventory is unavailable"
        ) from exc
    names = {path.name for path in children}
    if names != {_BUNDLE_FILENAME, "manifest.sha256"}:
        raise ArtifactPublicationError("validation run inventory is not exact")
    bundle_path = run_directory / _BUNDLE_FILENAME
    manifest_path = run_directory / "manifest.sha256"
    raw = _read_bounded_regular_file_once(
        bundle_path,
        maximum_bytes=_MAXIMUM_BUNDLE_BYTES,
        label="validation bundle",
    )
    manifest_raw = _read_bounded_regular_file_once(
        manifest_path,
        maximum_bytes=256,
        label="validation manifest",
    )
    expected_manifest = (
        f"{hashlib.sha256(raw).hexdigest()}  {_BUNDLE_FILENAME}\n".encode("ascii")
    )
    if manifest_raw != expected_manifest:
        raise ArtifactPublicationError("validation bundle manifest changed")

    def reject_duplicates(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("validation bundle contains duplicate JSON keys")
            result[key] = value
        return result

    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ArtifactPublicationError(
            "validation bundle is not canonical JSON"
        ) from exc
    if artifact_json_bytes(payload) != raw:
        raise ArtifactPublicationError("validation bundle JSON is not canonical")
    root = _mapping(
        payload,
        "validation bundle",
        frozenset(
            {
                "bundle_schema",
                "experiment_config_sha256",
                "plan_sha256",
                "tuning_selection",
                "checkpoint_selection",
                "validation_bundle_sha256",
            }
        ),
    )
    if root["validation_bundle_sha256"] != expected_validation_bundle_sha256:
        raise ArtifactPublicationError("validation bundle identity is not expected")
    try:
        bundle = H6ValidationBundleV3(
            bundle_schema=root["bundle_schema"],  # type: ignore[arg-type]
            experiment_config_sha256=root["experiment_config_sha256"],  # type: ignore[arg-type]
            plan_sha256=root["plan_sha256"],  # type: ignore[arg-type]
            tuning_selection=_tuning_selection_from_payload(root["tuning_selection"]),
            checkpoint_selection=_checkpoint_selection_from_payload(
                root["checkpoint_selection"]
            ),
            validation_bundle_sha256=root["validation_bundle_sha256"],  # type: ignore[arg-type]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactPublicationError("validation bundle cannot be reopened") from exc
    if (
        bundle.plan_sha256 != expected_plan_sha256
        or bundle.experiment_config_sha256 != expected_experiment_config_sha256
        or bundle.validation_bundle_sha256 != expected_validation_bundle_sha256
    ):
        raise ArtifactPublicationError("validation bundle authority drift")
    if artifact_json_bytes(dict(root)) != artifact_json_bytes(
        bundle.artifact_payload()
    ):
        raise ArtifactPublicationError("validation-bundle payload changed")
    return bundle


_A0_TEST_ENDPOINT_CONFIG_ID = H6_MATCHING_V3_ENDPOINT_CONFIG_IDS[0]
_COMPLETE_A5_TEST_ENDPOINT_CONFIG_ID = H6_MATCHING_V3_ENDPOINT_CONFIG_IDS[5]
_EMISSION_A5_TEST_ENDPOINT_CONFIG_ID = H6_MATCHING_V3_ENDPOINT_CONFIG_IDS[9]
_H6_TEST_PARTICLE_COUNTS = (128, 256, 512, 1024)
_H6_TEST_REPLICATES = tuple(range(64))
H6_WEIGHTED_COMMON_STREAM_ROOT_SEED = 2026072198
H6_WEIGHTED_COMMON_STREAM_DOMAIN = "h6-wt2-endpoint-mc-v1"


def h6_weighted_common_stream_sha256_v3(
    *,
    replicate_id: int,
) -> str:
    """Return the sole frozen common-stream identity for one replicate."""

    if type(replicate_id) is not int or replicate_id not in _H6_TEST_REPLICATES:
        raise ValueError("replicate_id is outside the frozen 64 replicates")
    return _hash(
        H6_WEIGHTED_COMMON_STREAM_DOMAIN,
        {
            "root_seed": H6_WEIGHTED_COMMON_STREAM_ROOT_SEED,
            "replicate_id": replicate_id,
            "purpose_stream_sha256s": tuple(
                (
                    purpose.value,
                    hashlib.sha256(
                        (
                            f"{H6_WEIGHTED_COMMON_STREAM_DOMAIN}|"
                            f"{H6_WEIGHTED_COMMON_STREAM_ROOT_SEED}|"
                            f"{replicate_id}|{purpose.value}"
                        ).encode("ascii")
                    ).hexdigest(),
                )
                for purpose in CounterPurpose
            ),
        },
    )


def _require_finite_float(value: object, name: str, *, nonnegative: bool) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite float")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _require_confirmatory_seed(value: object) -> int:
    if type(value) is not int or value not in H6_CONFIRMATORY_SEEDS_V3:
        raise ValueError("training seed is outside the frozen confirmatory inventory")
    return value


H6_WEIGHTED_COMMON_STREAM_REGISTRY_SHA256 = _hash(
    f"{H6_WEIGHTED_COMMON_STREAM_DOMAIN}.registry",
    {
        "root_seed": H6_WEIGHTED_COMMON_STREAM_ROOT_SEED,
        "entries": tuple(
            {
                "replicate_id": replicate_id,
                "common_stream_sha256": h6_weighted_common_stream_sha256_v3(
                    replicate_id=replicate_id,
                ),
            }
            for replicate_id in _H6_TEST_REPLICATES
        ),
    },
)


@dataclass(frozen=True, slots=True)
class H6ExactA0CorpusTotalV3:
    """One exact A0 corpus total; weighted-estimator fields do not exist."""

    row_schema: Literal["h6-exact-a0-corpus-total-v3"]
    row_kind: Literal["exact_a0_corpus_total"]
    endpoint_config_id: str
    training_seed: int
    checkpoint_sha256: str
    counted_test_targets: int
    exact_total_nll: float
    opening_proof_sha256: str
    row_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "row_schema": self.row_schema,
            "row_kind": self.row_kind,
            "endpoint_config_id": self.endpoint_config_id,
            "training_seed": self.training_seed,
            "checkpoint_sha256": self.checkpoint_sha256,
            "counted_test_targets": self.counted_test_targets,
            "exact_total_nll": self.exact_total_nll,
            "opening_proof_sha256": self.opening_proof_sha256,
        }

    def artifact_payload(self) -> dict[str, object]:
        return {**self.semantic_payload(), "row_sha256": self.row_sha256}

    def __post_init__(self) -> None:
        if (
            self.row_schema != "h6-exact-a0-corpus-total-v3"
            or self.row_kind != "exact_a0_corpus_total"
        ):
            raise ValueError("exact A0 row discriminator is not frozen")
        if self.endpoint_config_id != _A0_TEST_ENDPOINT_CONFIG_ID:
            raise ValueError("exact A0 rows require the frozen A0 endpoint")
        _require_confirmatory_seed(self.training_seed)
        _require_sha256(self.checkpoint_sha256, "checkpoint_sha256")
        if type(self.counted_test_targets) is not int or self.counted_test_targets <= 0:
            raise ValueError("counted_test_targets must be a positive integer")
        _require_finite_float(
            self.exact_total_nll,
            "exact_total_nll",
            nonnegative=True,
        )
        _require_sha256(self.opening_proof_sha256, "opening_proof_sha256")
        _require_sha256(self.row_sha256, "row_sha256")
        if self.row_sha256 != _hash(
            "vfe4.h6.exact-a0-corpus-total.v3",
            self.semantic_payload(),
        ):
            raise ValueError("exact A0 row SHA-256 does not match its fields")

    @classmethod
    def create(
        cls,
        *,
        endpoint_config_id: str,
        training_seed: int,
        checkpoint_sha256: str,
        counted_test_targets: int,
        exact_total_nll: float,
        opening_proof_sha256: str,
    ) -> H6ExactA0CorpusTotalV3:
        payload = {
            "row_schema": "h6-exact-a0-corpus-total-v3",
            "row_kind": "exact_a0_corpus_total",
            "endpoint_config_id": endpoint_config_id,
            "training_seed": training_seed,
            "checkpoint_sha256": checkpoint_sha256,
            "counted_test_targets": counted_test_targets,
            "exact_total_nll": exact_total_nll,
            "opening_proof_sha256": opening_proof_sha256,
        }
        return cls(
            **payload,  # type: ignore[arg-type]
            row_sha256=_hash(
                "vfe4.h6.exact-a0-corpus-total.v3",
                payload,
            ),
        )


@dataclass(frozen=True, slots=True)
class H6WeightedA5CorpusTotalV3:
    """One weighted A5 SMC corpus-total estimate."""

    row_schema: Literal["h6-weighted-a5-corpus-total-v3"]
    row_kind: Literal["weighted_a5_smc_corpus_total"]
    endpoint_role: Literal["complete_a5", "emission_a5"]
    endpoint_config_id: str
    training_seed: int
    checkpoint_sha256: str
    particle_count: Literal[128, 256, 512, 1024]
    replicate_id: int
    common_stream_sha256: str
    counted_test_targets: int
    weighted_total_nll: float
    monte_carlo_half_width: float
    smc_bias_bound: float
    opening_proof_sha256: str
    row_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "row_schema": self.row_schema,
            "row_kind": self.row_kind,
            "endpoint_role": self.endpoint_role,
            "endpoint_config_id": self.endpoint_config_id,
            "training_seed": self.training_seed,
            "checkpoint_sha256": self.checkpoint_sha256,
            "particle_count": self.particle_count,
            "replicate_id": self.replicate_id,
            "common_stream_sha256": self.common_stream_sha256,
            "counted_test_targets": self.counted_test_targets,
            "weighted_total_nll": self.weighted_total_nll,
            "monte_carlo_half_width": self.monte_carlo_half_width,
            "smc_bias_bound": self.smc_bias_bound,
            "opening_proof_sha256": self.opening_proof_sha256,
        }

    def artifact_payload(self) -> dict[str, object]:
        return {**self.semantic_payload(), "row_sha256": self.row_sha256}

    def __post_init__(self) -> None:
        if (
            self.row_schema != "h6-weighted-a5-corpus-total-v3"
            or self.row_kind != "weighted_a5_smc_corpus_total"
        ):
            raise ValueError("weighted A5 row discriminator is not frozen")
        expected_endpoint = {
            "complete_a5": _COMPLETE_A5_TEST_ENDPOINT_CONFIG_ID,
            "emission_a5": _EMISSION_A5_TEST_ENDPOINT_CONFIG_ID,
        }.get(self.endpoint_role)
        if expected_endpoint is None or self.endpoint_config_id != expected_endpoint:
            raise ValueError(
                "weighted A5 row role requires its frozen weighted A5 endpoint"
            )
        _require_confirmatory_seed(self.training_seed)
        _require_sha256(self.checkpoint_sha256, "checkpoint_sha256")
        if (
            type(self.particle_count) is not int
            or self.particle_count not in _H6_TEST_PARTICLE_COUNTS
        ):
            raise ValueError("particle_count is outside the frozen particle levels")
        if (
            type(self.replicate_id) is not int
            or self.replicate_id not in _H6_TEST_REPLICATES
        ):
            raise ValueError("replicate_id is outside the frozen 64 replicates")
        _require_sha256(self.common_stream_sha256, "common_stream_sha256")
        if self.common_stream_sha256 != h6_weighted_common_stream_sha256_v3(
            replicate_id=self.replicate_id,
        ):
            raise ValueError(
                "common stream is not derived from the frozen replicate registry"
            )
        if type(self.counted_test_targets) is not int or self.counted_test_targets <= 0:
            raise ValueError("counted_test_targets must be a positive integer")
        _require_finite_float(
            self.weighted_total_nll,
            "weighted_total_nll",
            nonnegative=True,
        )
        _require_finite_float(
            self.monte_carlo_half_width,
            "monte_carlo_half_width",
            nonnegative=True,
        )
        _require_finite_float(
            self.smc_bias_bound,
            "smc_bias_bound",
            nonnegative=True,
        )
        _require_sha256(self.opening_proof_sha256, "opening_proof_sha256")
        _require_sha256(self.row_sha256, "row_sha256")
        if self.row_sha256 != _hash(
            "vfe4.h6.weighted-a5-corpus-total.v3",
            self.semantic_payload(),
        ):
            raise ValueError("weighted A5 row SHA-256 does not match its fields")

    @classmethod
    def create(
        cls,
        *,
        endpoint_role: Literal["complete_a5", "emission_a5"],
        endpoint_config_id: str,
        training_seed: int,
        checkpoint_sha256: str,
        particle_count: Literal[128, 256, 512, 1024],
        replicate_id: int,
        counted_test_targets: int,
        weighted_total_nll: float,
        monte_carlo_half_width: float,
        smc_bias_bound: float,
        opening_proof_sha256: str,
    ) -> H6WeightedA5CorpusTotalV3:
        payload = {
            "row_schema": "h6-weighted-a5-corpus-total-v3",
            "row_kind": "weighted_a5_smc_corpus_total",
            "endpoint_role": endpoint_role,
            "endpoint_config_id": endpoint_config_id,
            "training_seed": training_seed,
            "checkpoint_sha256": checkpoint_sha256,
            "particle_count": particle_count,
            "replicate_id": replicate_id,
            "common_stream_sha256": h6_weighted_common_stream_sha256_v3(
                replicate_id=replicate_id,
            ),
            "counted_test_targets": counted_test_targets,
            "weighted_total_nll": weighted_total_nll,
            "monte_carlo_half_width": monte_carlo_half_width,
            "smc_bias_bound": smc_bias_bound,
            "opening_proof_sha256": opening_proof_sha256,
        }
        return cls(
            **payload,  # type: ignore[arg-type]
            row_sha256=_hash(
                "vfe4.h6.weighted-a5-corpus-total.v3",
                payload,
            ),
        )


@dataclass(frozen=True, slots=True)
class H6RawEndpointInventoryV4:
    """The complete, strictly discriminated held-out H6 row inventory."""

    inventory_schema: Literal["h6-raw-endpoint-inventory-v4"]
    scoring_inventory_sha256: str
    common_stream_registry_sha256: str
    opening_proof_sha256: str
    logical_row_count: Literal[4104]
    exact_a0_rows: tuple[H6ExactA0CorpusTotalV3, ...]
    complete_a5_rows: tuple[H6WeightedA5CorpusTotalV3, ...]
    emission_a5_rows: tuple[H6WeightedA5CorpusTotalV3, ...]
    inventory_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "inventory_schema": self.inventory_schema,
            "scoring_inventory_sha256": self.scoring_inventory_sha256,
            "common_stream_registry_sha256": (self.common_stream_registry_sha256),
            "opening_proof_sha256": self.opening_proof_sha256,
            "logical_row_count": self.logical_row_count,
            "exact_a0_rows": tuple(
                row.artifact_payload() for row in self.exact_a0_rows
            ),
            "complete_a5_rows": tuple(
                row.artifact_payload() for row in self.complete_a5_rows
            ),
            "emission_a5_rows": tuple(
                row.artifact_payload() for row in self.emission_a5_rows
            ),
        }

    def artifact_payload(self) -> dict[str, object]:
        return {
            **self.semantic_payload(),
            "inventory_sha256": self.inventory_sha256,
        }

    def __post_init__(self) -> None:
        if (
            self.inventory_schema != H6_RAW_ENDPOINT_INVENTORY_SCHEMA
            or self.scoring_inventory_sha256 != H6_SCORING_INVENTORY_SHA256
            or self.common_stream_registry_sha256
            != H6_WEIGHTED_COMMON_STREAM_REGISTRY_SHA256
            or self.logical_row_count != 4104
        ):
            raise ValueError("raw endpoint inventory is not the frozen 4104-row v4")
        if (
            type(self.exact_a0_rows) is not tuple
            or type(self.complete_a5_rows) is not tuple
            or type(self.emission_a5_rows) is not tuple
            or len(self.exact_a0_rows) != 8
            or len(self.complete_a5_rows) != 2048
            or len(self.emission_a5_rows) != 2048
        ):
            raise ValueError("raw endpoint inventory must contain exactly 4104 rows")
        if any(type(row) is not H6ExactA0CorpusTotalV3 for row in self.exact_a0_rows):
            raise ValueError("exact inventory partition contains a non-exact-A0 row")
        if any(
            type(row) is not H6WeightedA5CorpusTotalV3
            for row in (*self.complete_a5_rows, *self.emission_a5_rows)
        ):
            raise ValueError(
                "weighted inventory partition contains a non-weighted-A5 row"
            )
        for row in (
            *self.exact_a0_rows,
            *self.complete_a5_rows,
            *self.emission_a5_rows,
        ):
            row.__post_init__()
        expected_exact_keys = tuple(H6_CONFIRMATORY_SEEDS_V3)
        if (
            tuple(row.training_seed for row in self.exact_a0_rows)
            != expected_exact_keys
        ):
            raise ValueError("exact A0 rows are duplicated or not in frozen seed order")
        expected_weighted_keys = tuple(
            (seed, replicate_id, particle_count)
            for seed in H6_CONFIRMATORY_SEEDS_V3
            for replicate_id in _H6_TEST_REPLICATES
            for particle_count in _H6_TEST_PARTICLE_COUNTS
        )
        for role, rows in (
            ("complete_a5", self.complete_a5_rows),
            ("emission_a5", self.emission_a5_rows),
        ):
            if any(row.endpoint_role != role for row in rows):
                raise ValueError("weighted row is in the wrong discriminated partition")
            keys = tuple(
                (row.training_seed, row.replicate_id, row.particle_count)
                for row in rows
            )
            if keys != expected_weighted_keys:
                raise ValueError(
                    "weighted rows contain a duplicate or violate frozen ordering"
                )
        row_ids = tuple(
            row.row_sha256
            for row in (
                *self.exact_a0_rows,
                *self.complete_a5_rows,
                *self.emission_a5_rows,
            )
        )
        if len(set(row_ids)) != 4104:
            raise ValueError("raw endpoint inventory contains a duplicate row")
        openings = {
            row.opening_proof_sha256
            for row in (
                *self.exact_a0_rows,
                *self.complete_a5_rows,
                *self.emission_a5_rows,
            )
        }
        if openings != {self.opening_proof_sha256}:
            raise ValueError("raw endpoint inventory contains another opening")
        expected_streams = tuple(
            h6_weighted_common_stream_sha256_v3(
                replicate_id=replicate_id,
            )
            for _seed in H6_CONFIRMATORY_SEEDS_V3
            for replicate_id in _H6_TEST_REPLICATES
            for _particle_count in _H6_TEST_PARTICLE_COUNTS
        )
        complete_streams = tuple(
            row.common_stream_sha256 for row in self.complete_a5_rows
        )
        emission_streams = tuple(
            row.common_stream_sha256 for row in self.emission_a5_rows
        )
        if (
            complete_streams != expected_streams
            or emission_streams != expected_streams
            or len(set(complete_streams)) != 64
        ):
            raise ValueError(
                "weighted A5 rows do not use the frozen 64-stream registry"
            )
        target_counts = {
            row.counted_test_targets
            for row in (
                *self.exact_a0_rows,
                *self.complete_a5_rows,
                *self.emission_a5_rows,
            )
        }
        if len(target_counts) != 1:
            raise ValueError("raw rows disagree on counted test targets")
        _require_sha256(self.opening_proof_sha256, "opening_proof_sha256")
        _require_sha256(self.inventory_sha256, "inventory_sha256")
        if self.inventory_sha256 != _hash(
            "vfe4.h6.raw-endpoint-inventory.v4",
            self.semantic_payload(),
        ):
            raise ValueError("raw endpoint inventory SHA-256 does not match its rows")

    @classmethod
    def create(
        cls,
        *,
        exact_a0_rows: tuple[H6ExactA0CorpusTotalV3, ...],
        complete_a5_rows: tuple[H6WeightedA5CorpusTotalV3, ...],
        emission_a5_rows: tuple[H6WeightedA5CorpusTotalV3, ...],
    ) -> H6RawEndpointInventoryV4:
        if (
            type(exact_a0_rows) is not tuple
            or type(complete_a5_rows) is not tuple
            or type(emission_a5_rows) is not tuple
            or not exact_a0_rows
        ):
            raise ValueError(
                "raw endpoint inventory rows must be nonempty exact tuples"
            )
        opening = exact_a0_rows[0].opening_proof_sha256
        payload = {
            "inventory_schema": H6_RAW_ENDPOINT_INVENTORY_SCHEMA,
            "scoring_inventory_sha256": H6_SCORING_INVENTORY_SHA256,
            "common_stream_registry_sha256": (
                H6_WEIGHTED_COMMON_STREAM_REGISTRY_SHA256
            ),
            "opening_proof_sha256": opening,
            "logical_row_count": 4104,
            "exact_a0_rows": tuple(row.artifact_payload() for row in exact_a0_rows),
            "complete_a5_rows": tuple(
                row.artifact_payload() for row in complete_a5_rows
            ),
            "emission_a5_rows": tuple(
                row.artifact_payload() for row in emission_a5_rows
            ),
        }
        return cls(
            inventory_schema=H6_RAW_ENDPOINT_INVENTORY_SCHEMA,
            scoring_inventory_sha256=H6_SCORING_INVENTORY_SHA256,
            common_stream_registry_sha256=(H6_WEIGHTED_COMMON_STREAM_REGISTRY_SHA256),
            opening_proof_sha256=opening,
            logical_row_count=4104,
            exact_a0_rows=exact_a0_rows,
            complete_a5_rows=complete_a5_rows,
            emission_a5_rows=emission_a5_rows,
            inventory_sha256=_hash(
                "vfe4.h6.raw-endpoint-inventory.v4",
                payload,
            ),
        )

    @classmethod
    def from_rows(
        cls,
        rows: tuple[H6ExactA0CorpusTotalV3 | H6WeightedA5CorpusTotalV3, ...],
    ) -> H6RawEndpointInventoryV4:
        if type(rows) is not tuple or len(rows) != 4104:
            raise ValueError("raw endpoint inventory must contain exactly 4104 rows")
        exact = tuple(row for row in rows if type(row) is H6ExactA0CorpusTotalV3)
        complete = tuple(
            row
            for row in rows
            if type(row) is H6WeightedA5CorpusTotalV3
            and row.endpoint_role == "complete_a5"
        )
        emission = tuple(
            row
            for row in rows
            if type(row) is H6WeightedA5CorpusTotalV3
            and row.endpoint_role == "emission_a5"
        )
        if len(exact) + len(complete) + len(emission) != 4104:
            raise ValueError("raw endpoint inventory contains another row type")
        return cls.create(
            exact_a0_rows=exact,
            complete_a5_rows=complete,
            emission_a5_rows=emission,
        )


@dataclass(frozen=True, slots=True)
class H6PredictionMetricsV3:
    """Raw-row bindings for the frozen H6 contrasts."""

    metrics_schema: Literal["h6-prediction-metrics-v3"]
    raw_inventory_sha256: str
    primary_a0_row_sha256s: tuple[str, ...]
    primary_complete_a5_row_sha256s: tuple[str, ...]
    objective_complete_a5_row_sha256s: tuple[str, ...]
    objective_emission_a5_row_sha256s: tuple[str, ...]
    metrics_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "metrics_schema": self.metrics_schema,
            "raw_inventory_sha256": self.raw_inventory_sha256,
            "primary_a0_row_sha256s": self.primary_a0_row_sha256s,
            "primary_complete_a5_row_sha256s": (self.primary_complete_a5_row_sha256s),
            "objective_complete_a5_row_sha256s": (
                self.objective_complete_a5_row_sha256s
            ),
            "objective_emission_a5_row_sha256s": (
                self.objective_emission_a5_row_sha256s
            ),
        }

    def artifact_payload(self) -> dict[str, object]:
        return {**self.semantic_payload(), "metrics_sha256": self.metrics_sha256}

    def __post_init__(self) -> None:
        if self.metrics_schema != H6_PREDICTION_METRICS_SCHEMA:
            raise ValueError("prediction metrics schema is not v3")
        _require_sha256(self.raw_inventory_sha256, "raw_inventory_sha256")
        if (
            len(self.primary_a0_row_sha256s) != 8
            or len(self.primary_complete_a5_row_sha256s) != 2048
            or len(self.objective_complete_a5_row_sha256s) != 2048
            or len(self.objective_emission_a5_row_sha256s) != 2048
            or self.primary_complete_a5_row_sha256s
            != self.objective_complete_a5_row_sha256s
        ):
            raise ValueError("prediction metrics do not reuse the frozen raw row sets")
        for digest in (
            *self.primary_a0_row_sha256s,
            *self.primary_complete_a5_row_sha256s,
            *self.objective_complete_a5_row_sha256s,
            *self.objective_emission_a5_row_sha256s,
        ):
            _require_sha256(digest, "metric row SHA-256")
        _require_sha256(self.metrics_sha256, "metrics_sha256")
        if self.metrics_sha256 != _hash(
            "vfe4.h6.prediction-metrics.v3",
            self.semantic_payload(),
        ):
            raise ValueError("prediction metrics SHA-256 does not match its fields")

    @classmethod
    def from_raw_inventory(
        cls,
        inventory: H6RawEndpointInventoryV4,
    ) -> H6PredictionMetricsV3:
        if type(inventory) is not H6RawEndpointInventoryV4:
            raise ValueError("prediction metrics require an exact raw inventory v4")
        inventory.__post_init__()
        a0_ids = tuple(row.row_sha256 for row in inventory.exact_a0_rows)
        complete_ids = tuple(row.row_sha256 for row in inventory.complete_a5_rows)
        emission_ids = tuple(row.row_sha256 for row in inventory.emission_a5_rows)
        payload = {
            "metrics_schema": H6_PREDICTION_METRICS_SCHEMA,
            "raw_inventory_sha256": inventory.inventory_sha256,
            "primary_a0_row_sha256s": a0_ids,
            "primary_complete_a5_row_sha256s": complete_ids,
            "objective_complete_a5_row_sha256s": complete_ids,
            "objective_emission_a5_row_sha256s": emission_ids,
        }
        return cls(
            **payload,  # type: ignore[arg-type]
            metrics_sha256=_hash(
                "vfe4.h6.prediction-metrics.v3",
                payload,
            ),
        )


@dataclass(frozen=True, slots=True)
class H6PredictionResultV3:
    """Exact terminal H6-Prediction v3 result binding."""

    result_schema: Literal["h6-prediction-result-v3"]
    reservation_sha256: str
    opening_proof_sha256: str
    raw_inventory_sha256: str
    metrics_sha256: str
    logical_row_count: Literal[4104]
    result_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "result_schema": self.result_schema,
            "reservation_sha256": self.reservation_sha256,
            "opening_proof_sha256": self.opening_proof_sha256,
            "raw_inventory_sha256": self.raw_inventory_sha256,
            "metrics_sha256": self.metrics_sha256,
            "logical_row_count": self.logical_row_count,
        }

    def artifact_payload(self) -> dict[str, object]:
        return {**self.semantic_payload(), "result_sha256": self.result_sha256}

    def __post_init__(self) -> None:
        if (
            self.result_schema != H6_PREDICTION_RESULT_SCHEMA
            or self.logical_row_count != 4104
        ):
            raise ValueError("prediction result is not the exact v3 schema")
        for name in (
            "reservation_sha256",
            "opening_proof_sha256",
            "raw_inventory_sha256",
            "metrics_sha256",
            "result_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if self.result_sha256 != _hash(
            "vfe4.h6.prediction-result.v3",
            self.semantic_payload(),
        ):
            raise ValueError("prediction result SHA-256 does not match its fields")

    @classmethod
    def create(
        cls,
        *,
        reservation_sha256: str,
        opening_proof_sha256: str,
        inventory: H6RawEndpointInventoryV4,
        metrics: H6PredictionMetricsV3,
    ) -> H6PredictionResultV3:
        if type(inventory) is not H6RawEndpointInventoryV4:
            raise ValueError("prediction result requires an exact raw inventory v4")
        if type(metrics) is not H6PredictionMetricsV3:
            raise ValueError("prediction result requires exact prediction metrics v3")
        inventory.__post_init__()
        metrics.__post_init__()
        if (
            inventory.opening_proof_sha256 != opening_proof_sha256
            or metrics.raw_inventory_sha256 != inventory.inventory_sha256
        ):
            raise ValueError("prediction result authorities are not cross-bound")
        payload = {
            "result_schema": H6_PREDICTION_RESULT_SCHEMA,
            "reservation_sha256": reservation_sha256,
            "opening_proof_sha256": opening_proof_sha256,
            "raw_inventory_sha256": inventory.inventory_sha256,
            "metrics_sha256": metrics.metrics_sha256,
            "logical_row_count": 4104,
        }
        return cls(
            **payload,  # type: ignore[arg-type]
            result_sha256=_hash(
                "vfe4.h6.prediction-result.v3",
                payload,
            ),
        )


def _exact_a0_from_payload(value: object) -> H6ExactA0CorpusTotalV3:
    payload = _mapping(
        value,
        "exact A0 row",
        frozenset(
            {
                "row_schema",
                "row_kind",
                "endpoint_config_id",
                "training_seed",
                "checkpoint_sha256",
                "counted_test_targets",
                "exact_total_nll",
                "opening_proof_sha256",
                "row_sha256",
            }
        ),
    )
    try:
        return H6ExactA0CorpusTotalV3(**payload)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ArtifactPublicationError("exact A0 row cannot be reopened") from exc


def _weighted_a5_from_payload(value: object) -> H6WeightedA5CorpusTotalV3:
    payload = _mapping(
        value,
        "weighted A5 row",
        frozenset(
            {
                "row_schema",
                "row_kind",
                "endpoint_role",
                "endpoint_config_id",
                "training_seed",
                "checkpoint_sha256",
                "particle_count",
                "replicate_id",
                "common_stream_sha256",
                "counted_test_targets",
                "weighted_total_nll",
                "monte_carlo_half_width",
                "smc_bias_bound",
                "opening_proof_sha256",
                "row_sha256",
            }
        ),
    )
    try:
        return H6WeightedA5CorpusTotalV3(**payload)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ArtifactPublicationError("weighted A5 row cannot be reopened") from exc


def _raw_inventory_from_payload(value: object) -> H6RawEndpointInventoryV4:
    payload = _mapping(
        value,
        "raw endpoint inventory",
        frozenset(
            {
                "inventory_schema",
                "scoring_inventory_sha256",
                "common_stream_registry_sha256",
                "opening_proof_sha256",
                "logical_row_count",
                "exact_a0_rows",
                "complete_a5_rows",
                "emission_a5_rows",
                "inventory_sha256",
            }
        ),
    )
    try:
        return H6RawEndpointInventoryV4(
            inventory_schema=payload["inventory_schema"],  # type: ignore[arg-type]
            scoring_inventory_sha256=payload["scoring_inventory_sha256"],  # type: ignore[arg-type]
            common_stream_registry_sha256=payload[  # type: ignore[arg-type]
                "common_stream_registry_sha256"
            ],
            opening_proof_sha256=payload["opening_proof_sha256"],  # type: ignore[arg-type]
            logical_row_count=payload["logical_row_count"],  # type: ignore[arg-type]
            exact_a0_rows=tuple(
                _exact_a0_from_payload(row)
                for row in payload["exact_a0_rows"]  # type: ignore[union-attr]
            ),
            complete_a5_rows=tuple(
                _weighted_a5_from_payload(row)
                for row in payload["complete_a5_rows"]  # type: ignore[union-attr]
            ),
            emission_a5_rows=tuple(
                _weighted_a5_from_payload(row)
                for row in payload["emission_a5_rows"]  # type: ignore[union-attr]
            ),
            inventory_sha256=payload["inventory_sha256"],  # type: ignore[arg-type]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactPublicationError(
            "raw endpoint inventory cannot be reopened"
        ) from exc


def _prediction_metrics_from_payload(value: object) -> H6PredictionMetricsV3:
    payload = _mapping(
        value,
        "prediction metrics",
        frozenset(
            {
                "metrics_schema",
                "raw_inventory_sha256",
                "primary_a0_row_sha256s",
                "primary_complete_a5_row_sha256s",
                "objective_complete_a5_row_sha256s",
                "objective_emission_a5_row_sha256s",
                "metrics_sha256",
            }
        ),
    )
    try:
        return H6PredictionMetricsV3(
            metrics_schema=payload["metrics_schema"],  # type: ignore[arg-type]
            raw_inventory_sha256=payload["raw_inventory_sha256"],  # type: ignore[arg-type]
            primary_a0_row_sha256s=tuple(  # type: ignore[arg-type]
                payload["primary_a0_row_sha256s"]  # type: ignore[union-attr]
            ),
            primary_complete_a5_row_sha256s=tuple(  # type: ignore[arg-type]
                payload["primary_complete_a5_row_sha256s"]  # type: ignore[union-attr]
            ),
            objective_complete_a5_row_sha256s=tuple(  # type: ignore[arg-type]
                payload["objective_complete_a5_row_sha256s"]  # type: ignore[union-attr]
            ),
            objective_emission_a5_row_sha256s=tuple(  # type: ignore[arg-type]
                payload["objective_emission_a5_row_sha256s"]  # type: ignore[union-attr]
            ),
            metrics_sha256=payload["metrics_sha256"],  # type: ignore[arg-type]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactPublicationError("prediction metrics cannot be reopened") from exc


def publish_h6_prediction_result_v3(
    run_root: Path,
    run_name: str,
    *,
    result: H6PredictionResultV3,
    inventory: H6RawEndpointInventoryV4,
    metrics: H6PredictionMetricsV3,
) -> Path:
    """No-replace publish one exact result and its complete evidence."""

    if (
        type(result) is not H6PredictionResultV3
        or type(inventory) is not H6RawEndpointInventoryV4
        or type(metrics) is not H6PredictionMetricsV3
    ):
        raise ArtifactPublicationError("exact prediction result records are required")
    result.__post_init__()
    inventory.__post_init__()
    metrics.__post_init__()
    if (
        result.opening_proof_sha256 != inventory.opening_proof_sha256
        or result.raw_inventory_sha256 != inventory.inventory_sha256
        or result.metrics_sha256 != metrics.metrics_sha256
        or metrics.raw_inventory_sha256 != inventory.inventory_sha256
    ):
        raise ArtifactPublicationError("prediction result records are not cross-bound")
    published = publish_run_directory(
        run_root,
        run_name,
        {
            "metrics.json": metrics.artifact_payload(),
            "raw_inventory.json": inventory.artifact_payload(),
            "result.json": result.artifact_payload(),
        },
    )
    reopened = read_h6_prediction_result_v3(
        published,
        expected_result_sha256=result.result_sha256,
    )
    if reopened != (result, inventory, metrics):
        raise ArtifactPublicationError("published prediction result changed")
    return published


def read_h6_prediction_result_v3(
    run_directory: Path,
    *,
    expected_result_sha256: str | None = None,
) -> tuple[
    H6PredictionResultV3,
    H6RawEndpointInventoryV4,
    H6PredictionMetricsV3,
]:
    """Authenticate and reopen an exact H6-Prediction v3 result."""

    if expected_result_sha256 is not None:
        _require_sha256(expected_result_sha256, "expected_result_sha256")
    if not isinstance(run_directory, Path) or not run_directory.is_absolute():
        raise ArtifactPublicationError("result run_directory must be absolute")
    try:
        root_status = os.lstat(run_directory)
        children = tuple(run_directory.iterdir())
    except OSError as exc:
        raise ArtifactPublicationError("prediction result is unavailable") from exc
    if not stat.S_ISDIR(root_status.st_mode) or _is_redirect(
        run_directory, root_status
    ):
        raise ArtifactPublicationError("prediction result directory is redirected")
    filenames = ("metrics.json", "raw_inventory.json", "result.json")
    if {child.name for child in children} != {*filenames, "manifest.sha256"}:
        raise ArtifactPublicationError("prediction result inventory is not exact")
    raw_by_name = {
        name: _read_bounded_regular_file_once(
            run_directory / name,
            maximum_bytes=32 * 1024 * 1024,
            label=f"prediction {name}",
        )
        for name in filenames
    }
    manifest = _read_bounded_regular_file_once(
        run_directory / "manifest.sha256",
        maximum_bytes=512,
        label="prediction result manifest",
    )
    expected_manifest = "".join(
        f"{hashlib.sha256(raw_by_name[name]).hexdigest()}  {name}\n"
        for name in filenames
    ).encode("ascii")
    if manifest != expected_manifest:
        raise ArtifactPublicationError("prediction result manifest changed")

    def reject_duplicates(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("prediction result contains duplicate JSON keys")
            result[key] = item
        return result

    decoded: dict[str, object] = {}
    try:
        for name, raw in raw_by_name.items():
            value = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=reject_duplicates,
            )
            if artifact_json_bytes(value) != raw:
                raise ValueError("noncanonical JSON")
            decoded[name] = value
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ArtifactPublicationError(
            "prediction result JSON is not canonical"
        ) from exc
    inventory = _raw_inventory_from_payload(decoded["raw_inventory.json"])
    metrics = _prediction_metrics_from_payload(decoded["metrics.json"])
    result_payload = _mapping(
        decoded["result.json"],
        "prediction result",
        frozenset(
            {
                "result_schema",
                "reservation_sha256",
                "opening_proof_sha256",
                "raw_inventory_sha256",
                "metrics_sha256",
                "logical_row_count",
                "result_sha256",
            }
        ),
    )
    try:
        result = H6PredictionResultV3(**result_payload)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ArtifactPublicationError("prediction result cannot be reopened") from exc
    if (
        (
            expected_result_sha256 is not None
            and result.result_sha256 != expected_result_sha256
        )
        or result.opening_proof_sha256 != inventory.opening_proof_sha256
        or result.raw_inventory_sha256 != inventory.inventory_sha256
        or result.metrics_sha256 != metrics.metrics_sha256
        or metrics.raw_inventory_sha256 != inventory.inventory_sha256
    ):
        raise ArtifactPublicationError("prediction result authority drift")
    return result, inventory, metrics


__all__ = [
    "H6CheckpointCandidateV3",
    "H6CheckpointSelectionV3",
    "H6EndpointTuningSelectionV3",
    "H6ExactA0CorpusTotalV3",
    "H6PredictionMetricsV3",
    "H6PredictionResultV3",
    "H6RawEndpointInventoryV4",
    "H6TuningSelectionV3",
    "H6ValidationBundleV3",
    "H6ValidationRecordV3",
    "H6WeightedA5CorpusTotalV3",
    "H6_WEIGHTED_COMMON_STREAM_DOMAIN",
    "H6_WEIGHTED_COMMON_STREAM_REGISTRY_SHA256",
    "H6_WEIGHTED_COMMON_STREAM_ROOT_SEED",
    "bind_h6_checkpoint_selection_v3",
    "h6_weighted_common_stream_sha256_v3",
    "publish_h6_prediction_result_v3",
    "publish_h6_validation_bundle_v3",
    "read_h6_prediction_result_v3",
    "read_h6_validation_bundle_v3",
    "select_h6_tuning_v3",
]
