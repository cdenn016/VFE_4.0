"""Exact H6-Prediction v3 attempt-to-engine execution bindings.

This module closes the intentional gap between the outcome-blind experiment
plan and confirmatory execution.  Tuning attempts already own their literal
cell.  Confirmatory attempts instead consume one authenticated tuning
selection and bind the endpoint-specific winner without rewriting the frozen
plan.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from vfe4.artifacts.h6_prediction_v3 import (
    H6EndpointTuningSelectionV3,
    H6PredictionV3Authorities,
    H6TuningSelectionV3,
)
from vfe4.types.h6 import ArmConfig, canonical_json_bytes

from .h6_engine_v3 import H6EngineAuthorityV3
from .h6_experiment_v3 import (
    H6ExperimentPlanV3,
    H6PlannedAttemptV3,
    H6TuningCellV3,
)
from .matching import H6_ADAMW_POLICY


def _build_execution_origin_api():
    authority = object()

    class ExecutionOrigin:
        __slots__ = ()

        def __new__(cls, supplied_authority: object):
            if supplied_authority is not authority:
                raise TypeError("executable-attempt origins are binder-issued only")
            return super().__new__(cls)

    origin = ExecutionOrigin(authority)

    def issue() -> object:
        return origin

    def validate(value: object) -> bool:
        return value is origin

    return issue, validate


_issue_execution_origin, _is_execution_origin = _build_execution_origin_api()
del _build_execution_origin_api


def _hash(domain: str, payload: object) -> str:
    return hashlib.sha256(
        domain.encode("ascii") + b"\x00" + canonical_json_bytes(payload)
    ).hexdigest()


def _raw_authority_cross_check(
    authorities: H6PredictionV3Authorities,
) -> None:
    config = authorities.config
    matching_set = authorities.matching_set
    readiness = authorities.readiness
    plan = authorities.plan
    if (
        config.config_sha256 != readiness.experiment_config_sha256
        or config.config_sha256 != plan.experiment_config_sha256
        or config.source.git_head != matching_set.git_head
        or config.source.git_head != readiness.git_head
        or config.source.git_head != plan.git_head
        or config.source.dirty_digest != matching_set.dirty_digest
        or config.source.dirty_digest != readiness.dirty_digest
        or config.source.dirty_digest != plan.dirty_digest
        or config.matching_policy_sha256 != matching_set.matching_policy_sha256
        or config.matching_policy_sha256 != readiness.matching_policy_sha256
        or config.matching_policy_sha256 != plan.matching_policy_sha256
        or config.matching_set_sha256 != matching_set.matching_set_sha256
        or config.matching_set_sha256 != readiness.matching_set_sha256
        or config.matching_set_sha256 != plan.matching_set_sha256
        or config.training_schedule != plan.training_schedule
        or config.training_schedule.schedule_sha256
        != readiness.training_schedule_sha256
        or config.runtime.runtime_identity_sha256
        != readiness.runtime_identity_sha256
        or config.runtime.runtime_identity_sha256
        != plan.training_schedule.runtime_identity_sha256
        or config.data_identity_sha256 != readiness.data_identity_sha256
        or tuple(matching_set.endpoint_configs) != tuple(plan.endpoint_configs)
        or tuple(report.record_sha256 for report in matching_set.matrix_reports)
        != plan.matching_report_sha256s
    ):
        raise ValueError(
            "config/readiness/plan/matching authority identities drifted"
        )


def _canonical_planned_attempt(
    *,
    plan: H6ExperimentPlanV3,
    planned_attempt: object,
) -> H6PlannedAttemptV3:
    if type(planned_attempt) is not H6PlannedAttemptV3:
        raise ValueError("execution requires an exact v3 planned attempt")
    inventory = (
        plan.tuning_attempts
        if planned_attempt.stage == "tuning"
        else plan.confirmatory_attempts
        if planned_attempt.stage == "confirmatory"
        else ()
    )
    canonical = {
        item.planned_attempt_sha256: item for item in inventory
    }.get(planned_attempt.planned_attempt_sha256)
    if canonical != planned_attempt:
        raise ValueError(
            "planned attempt or endpoint is not an exact member of the plan"
        )
    planned_attempt.__post_init__()
    return planned_attempt


def _endpoint_config(
    *,
    authorities: H6PredictionV3Authorities,
    planned_attempt: H6PlannedAttemptV3,
) -> ArmConfig:
    plan_configs = {
        config.config_id: config for config in authorities.plan.endpoint_configs
    }
    matching_configs = {
        config.config_id: config
        for config in authorities.matching_set.endpoint_configs
    }
    config = plan_configs.get(planned_attempt.endpoint_config_id)
    if (
        type(config) is not ArmConfig
        or matching_configs.get(planned_attempt.endpoint_config_id) != config
        or config.config_sha256 != planned_attempt.endpoint_config_sha256
        or config.config_sha256
        != planned_attempt.attempt_spec.endpoint_config_sha256
        or config.config_id != planned_attempt.attempt_spec.endpoint_id
        or config.objective_kind != planned_attempt.attempt_spec.objective_kind
        or config.latent_enabled
        != (
            planned_attempt.attempt_spec.recognition_factory_sha256
            is not None
        )
        or config.horizon + 1 != planned_attempt.receiver_count
    ):
        raise ValueError("planned attempt endpoint/config identity drift")
    config.__post_init__()
    ledger_by_endpoint = {
        endpoint.config_id: ledger.ledger_sha256
        for endpoint, ledger in zip(
            authorities.matching_set.endpoint_configs,
            authorities.matching_set.endpoint_ledgers,
            strict=True,
        )
    }
    if (
        planned_attempt.matching_ledger_sha256
        != ledger_by_endpoint[config.config_id]
        or planned_attempt.matching_report_sha256s
        != authorities.plan.matching_report_sha256s
        or planned_attempt.matching_set_sha256
        != authorities.matching_set.matching_set_sha256
        or planned_attempt.matching_policy_sha256
        != authorities.matching_set.matching_policy_sha256
    ):
        raise ValueError("planned attempt matching identity drift")
    spec = planned_attempt.attempt_spec
    if (
        spec.git_head != authorities.plan.git_head
        or spec.dirty_digest != authorities.plan.dirty_digest
        or spec.experiment_config_sha256
        != authorities.config.config_sha256
        or spec.readiness_sha256 != authorities.readiness.readiness_sha256
        or spec.training_schedule_sha256
        != authorities.plan.training_schedule.schedule_sha256
        or spec.runtime_identity_sha256
        != authorities.config.runtime.runtime_identity_sha256
        or spec.data_identity_sha256
        != authorities.config.data_identity_sha256
        or spec.optimizer_policy_sha256
        != authorities.plan.training_schedule.outer.optimizer_policy_sha256
    ):
        raise ValueError(
            "planned attempt config/readiness/plan/runtime identity drift"
        )
    return config


def _selected_cell(
    *,
    authorities: H6PredictionV3Authorities,
    planned_attempt: H6PlannedAttemptV3,
    tuning_selection: object,
) -> tuple[
    H6TuningCellV3,
    str,
    H6TuningSelectionV3 | None,
    H6EndpointTuningSelectionV3 | None,
]:
    if planned_attempt.stage == "tuning":
        if tuning_selection is not None:
            raise ValueError(
                "tuning execution cannot consume an outcome tuning selection"
            )
        if (
            type(planned_attempt.tuning_cell) is not H6TuningCellV3
            or planned_attempt.tuning_cell_source != "literal-six-cell-v1"
        ):
            raise ValueError("tuning attempt lost its embedded literal cell")
        planned_attempt.tuning_cell.__post_init__()
        return (
            planned_attempt.tuning_cell,
            planned_attempt.tuning_cell_source,
            None,
            None,
        )

    if type(tuning_selection) is not H6TuningSelectionV3:
        raise ValueError(
            "confirmatory execution requires an exact v3 tuning selection"
        )
    if (
        tuning_selection.experiment_config_sha256
        != authorities.config.config_sha256
        or tuning_selection.plan_sha256 != authorities.plan.plan_sha256
        or tuning_selection.readiness_sha256
        != authorities.readiness.readiness_sha256
        or tuning_selection.matching_set_sha256
        != authorities.matching_set.matching_set_sha256
        or tuning_selection.data_identity_sha256
        != authorities.config.data_identity_sha256
        or tuning_selection.runtime_identity_sha256
        != authorities.config.runtime.runtime_identity_sha256
    ):
        raise ValueError("tuning selection plan/authority identities drifted")
    tuning_selection.__post_init__()
    by_endpoint = {
        item.endpoint_config_id: item
        for item in tuning_selection.endpoint_selections
    }
    selected = by_endpoint.get(planned_attempt.endpoint_config_id)
    if (
        type(selected) is not H6EndpointTuningSelectionV3
        or selected.endpoint_config_sha256
        != planned_attempt.endpoint_config_sha256
        or planned_attempt.tuning_cell is not None
        or planned_attempt.tuning_cell_source
        != f"selected:{selected.source_endpoint_config_id}"
    ):
        raise ValueError(
            "confirmatory attempt endpoint selection source/cell drift"
        )
    selected.__post_init__()
    return (
        selected.tuning_cell,
        planned_attempt.tuning_cell_source,
        tuning_selection,
        selected,
    )


def _engine_authority(
    *,
    authorities: H6PredictionV3Authorities,
    planned_attempt: H6PlannedAttemptV3,
    tuning_cell: H6TuningCellV3,
) -> H6EngineAuthorityV3:
    spec = planned_attempt.attempt_spec
    return H6EngineAuthorityV3.create(
        attempt_spec_sha256=spec.attempt_spec_sha256,
        endpoint_config_sha256=planned_attempt.endpoint_config_sha256,
        readiness_sha256=authorities.readiness.readiness_sha256,
        readiness_matching_set_sha256=(
            authorities.readiness.matching_set_sha256
        ),
        matching_set_sha256=planned_attempt.matching_set_sha256,
        matching_policy_sha256=planned_attempt.matching_policy_sha256,
        readiness_training_schedule_sha256=(
            authorities.readiness.training_schedule_sha256
        ),
        training_schedule_sha256=spec.training_schedule_sha256,
        readiness_runtime_identity_sha256=(
            authorities.readiness.runtime_identity_sha256
        ),
        runtime_identity_sha256=spec.runtime_identity_sha256,
        planned_attempt_sha256=planned_attempt.planned_attempt_sha256,
        endpoint_config_id=planned_attempt.endpoint_config_id,
        matching_ledger_sha256=planned_attempt.matching_ledger_sha256,
        matching_report_sha256s=planned_attempt.matching_report_sha256s,
        receiver_count=planned_attempt.receiver_count,
        state_categorical_enabled=(
            planned_attempt.state_categorical_enabled
        ),
        model_categorical_enabled=(
            planned_attempt.model_categorical_enabled
        ),
        tuning_cell_sha256=tuning_cell.cell_sha256,
        optimizer_policy_sha256=spec.optimizer_policy_sha256,
        optimizer_learning_rate=tuning_cell.learning_rate,
        optimizer_weight_decay=tuning_cell.weight_decay,
        objective_kind=spec.objective_kind,
        latent_enabled=spec.recognition_factory_sha256 is not None,
    )


@dataclass(frozen=True, slots=True, init=False)
class H6ExecutableAttemptV3:
    """One exact plan member with the cell and engine authority it may execute."""

    authorities: H6PredictionV3Authorities
    planned_attempt: H6PlannedAttemptV3
    endpoint_config: ArmConfig
    tuning_cell: H6TuningCellV3
    tuning_cell_source: str
    tuning_selection: H6TuningSelectionV3 | None
    endpoint_selection: H6EndpointTuningSelectionV3 | None
    engine_authority: H6EngineAuthorityV3
    executable_attempt_sha256: str
    _origin: object = field(repr=False, compare=False)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "authority_sha256": self.authorities.authority_sha256,
            "plan_sha256": self.authorities.plan.plan_sha256,
            "planned_attempt_sha256": (
                self.planned_attempt.planned_attempt_sha256
            ),
            "endpoint_config_id": self.endpoint_config.config_id,
            "endpoint_config_sha256": self.endpoint_config.config_sha256,
            "tuning_cell_sha256": self.tuning_cell.cell_sha256,
            "tuning_cell_source": self.tuning_cell_source,
            "tuning_selection_sha256": (
                None
                if self.tuning_selection is None
                else self.tuning_selection.tuning_selection_sha256
            ),
            "endpoint_selection_sha256": (
                None
                if self.endpoint_selection is None
                else self.endpoint_selection.endpoint_selection_sha256
            ),
            "engine_authority_sha256": (
                self.engine_authority.authority_sha256
            ),
        }

    def __post_init__(self) -> None:
        if not _is_execution_origin(self._origin):
            raise ValueError("executable attempt is not binder-authenticated")
        if type(self.authorities) is not H6PredictionV3Authorities:
            raise ValueError("executable attempt requires exact v3 authorities")
        _raw_authority_cross_check(self.authorities)
        if (
            type(self.planned_attempt) is not H6PlannedAttemptV3
            or type(self.endpoint_config) is not ArmConfig
            or type(self.tuning_cell) is not H6TuningCellV3
            or type(self.engine_authority) is not H6EngineAuthorityV3
        ):
            raise ValueError("executable attempt contains a non-v3 binding")
        self.planned_attempt.__post_init__()
        self.endpoint_config.__post_init__()
        self.tuning_cell.__post_init__()
        self.engine_authority.__post_init__()
        plan_attempts = {
            item.planned_attempt_sha256: item
            for item in self.authorities.plan.attempts
        }
        if (
            plan_attempts.get(self.planned_attempt.planned_attempt_sha256)
            != self.planned_attempt
            or self.endpoint_config.config_id
            != self.planned_attempt.endpoint_config_id
            or self.endpoint_config.config_sha256
            != self.planned_attempt.endpoint_config_sha256
        ):
            raise ValueError("executable attempt is outside the exact plan")
        if self.planned_attempt.stage == "tuning":
            if (
                self.tuning_cell != self.planned_attempt.tuning_cell
                or self.tuning_cell_source != "literal-six-cell-v1"
                or self.tuning_selection is not None
                or self.endpoint_selection is not None
            ):
                raise ValueError("executable tuning-cell binding drift")
        else:
            if (
                type(self.tuning_selection) is not H6TuningSelectionV3
                or type(self.endpoint_selection)
                is not H6EndpointTuningSelectionV3
                or self.endpoint_selection not in (
                    self.tuning_selection.endpoint_selections
                )
                or self.endpoint_selection.endpoint_config_id
                != self.planned_attempt.endpoint_config_id
                or self.endpoint_selection.tuning_cell != self.tuning_cell
                or self.tuning_cell_source
                != (
                    "selected:"
                    f"{self.endpoint_selection.source_endpoint_config_id}"
                )
            ):
                raise ValueError("executable confirmatory selection drift")
        engine = self.engine_authority
        if (
            engine.attempt_spec_sha256
            != self.planned_attempt.attempt_spec.attempt_spec_sha256
            or engine.planned_attempt_sha256
            != self.planned_attempt.planned_attempt_sha256
            or engine.endpoint_config_sha256
            != self.endpoint_config.config_sha256
            or engine.endpoint_config_id != self.endpoint_config.config_id
            or engine.readiness_sha256
            != self.authorities.readiness.readiness_sha256
            or engine.matching_set_sha256
            != self.authorities.matching_set.matching_set_sha256
            or engine.tuning_cell_sha256 != self.tuning_cell.cell_sha256
            or engine.optimizer_policy_sha256
            != H6_ADAMW_POLICY.optimizer_policy_sha256
            or engine.optimizer_learning_rate
            != self.tuning_cell.learning_rate
            or engine.optimizer_weight_decay
            != self.tuning_cell.weight_decay
        ):
            raise ValueError("executable engine authority drift")
        if self.executable_attempt_sha256 != _hash(
            "vfe4.h6.executable-attempt.v3", self.canonical_payload()
        ):
            raise ValueError("executable-attempt identity is stale")


def bind_h6_executable_attempt_v3(
    *,
    authorities: object,
    planned_attempt: object,
    tuning_selection: object = None,
) -> H6ExecutableAttemptV3:
    """Bind one exact v3 plan member to its immutable optimizer cell."""

    if type(authorities) is not H6PredictionV3Authorities:
        raise ValueError(
            "execution requires an exact H6-Prediction v3 authorities bundle"
        )
    _raw_authority_cross_check(authorities)
    canonical_attempt = _canonical_planned_attempt(
        plan=authorities.plan,
        planned_attempt=planned_attempt,
    )
    endpoint_config = _endpoint_config(
        authorities=authorities,
        planned_attempt=canonical_attempt,
    )
    (
        tuning_cell,
        tuning_cell_source,
        checked_selection,
        endpoint_selection,
    ) = _selected_cell(
        authorities=authorities,
        planned_attempt=canonical_attempt,
        tuning_selection=tuning_selection,
    )
    # All work above is pure boundary checking.  Re-run the complete
    # config/matching/readiness/plan regeneration before issuing the engine
    # authority, which is the first executable capability produced here.
    authorities.__post_init__()
    engine_authority = _engine_authority(
        authorities=authorities,
        planned_attempt=canonical_attempt,
        tuning_cell=tuning_cell,
    )
    values = {
        "authorities": authorities,
        "planned_attempt": canonical_attempt,
        "endpoint_config": endpoint_config,
        "tuning_cell": tuning_cell,
        "tuning_cell_source": tuning_cell_source,
        "tuning_selection": checked_selection,
        "endpoint_selection": endpoint_selection,
        "engine_authority": engine_authority,
    }
    provisional = object.__new__(H6ExecutableAttemptV3)
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    object.__setattr__(
        provisional,
        "executable_attempt_sha256",
        _hash(
            "vfe4.h6.executable-attempt.v3",
            provisional.canonical_payload(),
        ),
    )
    object.__setattr__(provisional, "_origin", _issue_execution_origin())
    provisional.__post_init__()
    return provisional


__all__ = [
    "H6ExecutableAttemptV3",
    "bind_h6_executable_attempt_v3",
]
