from __future__ import annotations

import dataclasses
import hashlib
import importlib
from collections.abc import Iterator
from typing import Any

import pytest

import test_h6_readiness_v3 as readiness_fixtures
import vfe4.artifacts.h6_prediction_v3 as prediction_artifacts
from vfe4.artifacts.h6_prediction_v3 import (
    H6PredictionV3Authorities,
    H6TuningSelectionV3,
    select_h6_tuning_v3,
)
from vfe4.training.h6_experiment_v3 import (
    H6_TUNED_ENDPOINT_CONFIG_IDS_V3,
    H6ExperimentPlanV3,
    H6PlannedAttemptV3,
    H6TuningCellV3,
    plan_h6_experiment_v3,
)
from vfe4.training.h6_matching_v3 import (
    H6_MATCHING_V3_ENDPOINT_CONFIG_IDS,
)
from vfe4.training.h6_readiness import (
    _derive_h6_prediction_readiness_v3 as validate_h6_prediction_readiness_v3,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _unsafe_replace(record: Any, **changes: object) -> Any:
    forged = object.__new__(type(record))
    for field in dataclasses.fields(record):
        object.__setattr__(
            forged,
            field.name,
            changes.get(field.name, getattr(record, field.name)),
        )
    return forged


@pytest.fixture(scope="module")
def authorities(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[H6PredictionV3Authorities]:
    matching_set = readiness_fixtures._matching_set()
    config = readiness_fixtures._config(
        matching_set=matching_set,
        artifact_root=tmp_path_factory.mktemp("h6-execution-authorities"),
    )
    readiness = validate_h6_prediction_readiness_v3(
        config=config,
        matching_set=matching_set,
        git_head=readiness_fixtures._GIT_HEAD,
        dirty_digest=readiness_fixtures._DIRTY_DIGEST,
    )
    plan = plan_h6_experiment_v3(
        readiness=readiness,
        matching_set=matching_set,
        training_schedule=config.training_schedule,
        runtime_identity=config.runtime,
    )
    yield H6PredictionV3Authorities.create(
        config=config,
        matching_set=matching_set,
        readiness=readiness,
        plan=plan,
    )


def _validation_record(
    *,
    plan: H6ExperimentPlanV3,
    attempt: H6PlannedAttemptV3,
    mean_prior_nll: float,
) -> object:
    cell = attempt.tuning_cell
    assert type(cell) is H6TuningCellV3
    return prediction_artifacts._create_h6_validation_record_v3(
        experiment_config_sha256=plan.experiment_config_sha256,
        plan_sha256=plan.plan_sha256,
        endpoint_config_id=attempt.endpoint_config_id,
        endpoint_config_sha256=attempt.endpoint_config_sha256,
        tuning_cell=cell,
        training_seed=attempt.training_seed,
        attempt_spec_sha256=attempt.attempt_spec.attempt_spec_sha256,
        checkpoint_sha256=_digest(
            f"checkpoint:{attempt.planned_attempt_sha256}"
        ),
        checkpoint_bytes_sha256=_digest(
            f"checkpoint-bytes:{attempt.planned_attempt_sha256}"
        ),
        readiness_sha256=plan.readiness_sha256,
        matching_set_sha256=plan.matching_set_sha256,
        data_identity_sha256=attempt.attempt_spec.data_identity_sha256,
        runtime_identity_sha256=(
            plan.training_schedule.runtime_identity_sha256
        ),
        counted_target_total=10,
        total_prior_nll=10.0 * mean_prior_nll,
    )


@pytest.fixture(scope="module")
def tuning_selection(
    authorities: H6PredictionV3Authorities,
) -> H6TuningSelectionV3:
    plan = authorities.plan
    endpoint_indices = {
        endpoint_id: index
        for index, endpoint_id in enumerate(H6_TUNED_ENDPOINT_CONFIG_IDS_V3)
    }
    cell_indices = {
        cell.cell_sha256: index
        for index, cell in enumerate(plan.tuning_cells)
    }
    records = tuple(
        _validation_record(
            plan=plan,
            attempt=attempt,
            mean_prior_nll=(
                1.0
                if cell_indices[attempt.tuning_cell.cell_sha256]
                == endpoint_indices[attempt.endpoint_config_id]
                else 2.0 + cell_indices[attempt.tuning_cell.cell_sha256]
            ),
        )
        for attempt in plan.tuning_attempts
        if attempt.tuning_cell is not None
    )
    return select_h6_tuning_v3(records, plan)


def _execution_api() -> Any:
    return importlib.import_module("vfe4.training.h6_execution_v3")


def test_tuning_attempt_binds_only_its_embedded_cell(
    authorities: H6PredictionV3Authorities,
) -> None:
    execution = _execution_api()
    attempt = authorities.plan.tuning_attempts[0]
    original_plan_sha256 = authorities.plan.plan_sha256

    bound = execution.bind_h6_executable_attempt_v3(
        authorities=authorities,
        planned_attempt=attempt,
    )

    assert type(bound) is execution.H6ExecutableAttemptV3
    assert bound.planned_attempt is attempt
    assert bound.endpoint_config == authorities.plan.endpoint_configs[0]
    assert bound.tuning_cell is attempt.tuning_cell
    assert bound.tuning_cell_source == "literal-six-cell-v1"
    assert bound.tuning_selection is None
    assert bound.endpoint_selection is None
    assert bound.engine_authority.optimizer_learning_rate == 0.0001
    assert bound.engine_authority.optimizer_weight_decay == 0.0
    assert (
        bound.engine_authority.planned_attempt_sha256
        == attempt.planned_attempt_sha256
    )
    bound.__post_init__()
    assert authorities.plan.plan_sha256 == original_plan_sha256
    assert authorities.plan.confirmatory_attempts[0].tuning_cell is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        bound.tuning_cell_source = "selected:outcome"  # type: ignore[misc]


def test_confirmatory_attempt_binds_exact_endpoint_selected_cell_and_source(
    authorities: H6PredictionV3Authorities,
    tuning_selection: H6TuningSelectionV3,
) -> None:
    execution = _execution_api()
    endpoint_id = H6_MATCHING_V3_ENDPOINT_CONFIG_IDS[6]
    attempt = next(
        item
        for item in authorities.plan.confirmatory_attempts
        if item.endpoint_config_id == endpoint_id
    )
    selected = next(
        item
        for item in tuning_selection.endpoint_selections
        if item.endpoint_config_id == endpoint_id
    )

    bound = execution.bind_h6_executable_attempt_v3(
        authorities=authorities,
        planned_attempt=attempt,
        tuning_selection=tuning_selection,
    )

    assert attempt.tuning_cell is None
    assert bound.tuning_cell == selected.tuning_cell
    assert bound.endpoint_selection == selected
    assert bound.tuning_cell_source == (
        f"selected:{selected.source_endpoint_config_id}"
    )
    assert (
        bound.engine_authority.tuning_cell_sha256
        == selected.tuning_cell.cell_sha256
    )
    assert (
        bound.engine_authority.optimizer_learning_rate
        == selected.tuning_cell.learning_rate
    )
    assert (
        bound.engine_authority.optimizer_weight_decay
        == selected.tuning_cell.weight_decay
    )
    assert bound.tuning_selection is tuning_selection
    bound.engine_authority.__post_init__()
    assert authorities.plan.confirmatory_attempts[
        authorities.plan.confirmatory_attempts.index(attempt)
    ].tuning_cell is None


def test_execution_binding_rejects_outcome_override_and_missing_selection(
    authorities: H6PredictionV3Authorities,
    tuning_selection: H6TuningSelectionV3,
) -> None:
    execution = _execution_api()

    with pytest.raises(ValueError, match="tuning.*selection|outcome"):
        execution.bind_h6_executable_attempt_v3(
            authorities=authorities,
            planned_attempt=authorities.plan.tuning_attempts[0],
            tuning_selection=tuning_selection,
        )

    with pytest.raises(ValueError, match="confirmatory.*selection"):
        execution.bind_h6_executable_attempt_v3(
            authorities=authorities,
            planned_attempt=authorities.plan.confirmatory_attempts[0],
        )


def test_execution_binding_rejects_authority_attempt_and_selection_drift(
    authorities: H6PredictionV3Authorities,
    tuning_selection: H6TuningSelectionV3,
) -> None:
    execution = _execution_api()
    attempt = authorities.plan.confirmatory_attempts[0]

    with pytest.raises(ValueError, match="exact.*v3 authorit"):
        execution.bind_h6_executable_attempt_v3(
            authorities=object(),
            planned_attempt=attempt,
            tuning_selection=tuning_selection,
        )

    drifted_plan = _unsafe_replace(
        authorities.plan,
        readiness_sha256="0" * 64,
    )
    drifted_authorities = _unsafe_replace(authorities, plan=drifted_plan)
    with pytest.raises(ValueError, match="readiness|plan|authority"):
        execution.bind_h6_executable_attempt_v3(
            authorities=drifted_authorities,
            planned_attempt=attempt,
            tuning_selection=tuning_selection,
        )

    drifted_attempt = _unsafe_replace(
        attempt,
        endpoint_config_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="endpoint|attempt|plan"):
        execution.bind_h6_executable_attempt_v3(
            authorities=authorities,
            planned_attempt=drifted_attempt,
            tuning_selection=tuning_selection,
        )

    drifted_selection = _unsafe_replace(
        tuning_selection,
        plan_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="selection.*plan|authority"):
        execution.bind_h6_executable_attempt_v3(
            authorities=authorities,
            planned_attempt=attempt,
            tuning_selection=drifted_selection,
        )


def test_execution_binding_is_exact_v3_only(
    authorities: H6PredictionV3Authorities,
) -> None:
    execution = _execution_api()
    attempt = authorities.plan.tuning_attempts[0]

    class RelabeledLegacyAuthorities:
        config = authorities.config
        matching_set = authorities.matching_set
        readiness = authorities.readiness
        plan = authorities.plan
        authority_sha256 = authorities.authority_sha256

    with pytest.raises(ValueError, match="exact.*v3 authorit"):
        execution.bind_h6_executable_attempt_v3(
            authorities=RelabeledLegacyAuthorities(),
            planned_attempt=attempt,
        )
    with pytest.raises(ValueError, match="exact.*planned attempt"):
        execution.bind_h6_executable_attempt_v3(
            authorities=authorities,
            planned_attempt=object(),
        )
