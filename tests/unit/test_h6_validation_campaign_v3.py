from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import test_h6_readiness_v3 as readiness_fixtures
import test_h6_validation_v3 as validation_fixtures
from vfe4.artifacts.atomic import ArtifactPublicationError
from vfe4.artifacts.h6_prediction_v3 import H6PredictionV3Authorities
from vfe4.training.h6_experiment_v3 import plan_h6_experiment_v3
from vfe4.training.h6_readiness import (
    _derive_h6_prediction_readiness_v3 as validate_h6_prediction_readiness_v3,
)


@pytest.fixture(scope="module")
def authorities(
    tmp_path_factory: pytest.TempPathFactory,
) -> H6PredictionV3Authorities:
    matching_set = readiness_fixtures._matching_set()
    config = readiness_fixtures._config(
        matching_set=matching_set,
        artifact_root=tmp_path_factory.mktemp("h6-campaign-authority"),
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
    return H6PredictionV3Authorities.create(
        config=config,
        matching_set=matching_set,
        readiness=readiness,
        plan=plan,
    )


def test_tuning_selection_is_no_replace_published_and_reopened(
    tmp_path: Path,
) -> None:
    from vfe4.training.h6_validation_campaign_v3 import (
        publish_h6_tuning_selection_v3,
        read_h6_tuning_selection_v3,
    )

    selection = validation_fixtures._tuning_selection()
    run_root = (tmp_path / "validation").resolve()
    published = publish_h6_tuning_selection_v3(
        run_root=run_root,
        run_name="TUNING_SELECTION",
        selection=selection,
    )
    assert read_h6_tuning_selection_v3(
        published,
        expected_plan_sha256=selection.plan_sha256,
        expected_experiment_config_sha256=(
            selection.experiment_config_sha256
        ),
        expected_tuning_selection_sha256=(
            selection.tuning_selection_sha256
        ),
    ) == selection
    with pytest.raises(ArtifactPublicationError, match="already exists"):
        publish_h6_tuning_selection_v3(
            run_root=run_root,
            run_name="TUNING_SELECTION",
            selection=selection,
        )


def test_validation_campaign_scores_exact_tuning_catalog_then_publishes_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    authorities: H6PredictionV3Authorities,
) -> None:
    import vfe4.training.h6_validation_campaign_v3 as campaign

    records = tuple(
        validation_fixtures._validation_record(
            plan=authorities.plan,
            attempt=attempt,
            mean_prior_nll=float(
                authorities.plan.tuning_cells.index(attempt.tuning_cell)
            ),
        )
        for attempt in authorities.plan.tuning_attempts
    )
    by_attempt = {
        attempt.planned_attempt_sha256: record
        for attempt, record in zip(
            authorities.plan.tuning_attempts,
            records,
            strict=True,
        )
    }
    items = tuple(
        SimpleNamespace(
            executable_attempt=SimpleNamespace(planned_attempt=attempt),
            checkpoint=SimpleNamespace(
                checkpoint_sha256=by_attempt[
                    attempt.planned_attempt_sha256
                ].checkpoint_sha256
            ),
        )
        for attempt in authorities.plan.tuning_attempts
    )
    read_calls: list[object] = []
    capabilities: list[object] = []
    scored: list[str] = []

    def read_catalog(*args: object, **kwargs: object) -> object:
        read_calls.append((args, kwargs))
        assert kwargs["required_inventory"] == "tuning"
        return SimpleNamespace(tuning_items=items)

    def issue_capability(*args: object) -> object:
        capability = object()
        capabilities.append((args, capability))
        return capability

    def score(
        *,
        capability: object,
        checkpoint: object,
        planned_attempt: object,
        plan: object,
    ) -> object:
        assert capability is capabilities[-1][1]
        assert plan is authorities.plan
        record = by_attempt[planned_attempt.planned_attempt_sha256]
        assert checkpoint.checkpoint_sha256 == record.checkpoint_sha256
        scored.append(planned_attempt.planned_attempt_sha256)
        return record

    monkeypatch.setattr(
        campaign,
        "read_h6_checkpoint_catalog_v3",
        read_catalog,
    )
    monkeypatch.setattr(
        campaign,
        "issue_h6_validation_capability_v3",
        issue_capability,
    )
    monkeypatch.setattr(
        campaign,
        "score_h6_validation_checkpoint_v3",
        score,
    )
    validation_directory = (tmp_path / "validation" / "COMPLETE").resolve()

    result = campaign.run_h6_validation_campaign_v3(
        authorities=authorities,
        store=object(),
        checkpoint_catalog_root=(tmp_path / "catalog").resolve(),
        maximum_checkpoint_bytes=16 * 1024 * 1024,
        validation_bundle_directory=validation_directory,
    )

    assert result.state == "TUNING_SELECTED"
    assert result.validation_bundle is None
    assert result.tuning_selection.tuning_validation_records == records
    assert result.published_directory == (
        validation_directory.parent / "COMPLETE-TUNING-SELECTION"
    )
    assert len(read_calls) == 1
    assert len(capabilities) == 1
    assert len(scored) == 72
    assert tuple(scored) == tuple(
        attempt.planned_attempt_sha256
        for attempt in authorities.plan.tuning_attempts
    )


def test_validation_campaign_accepts_plan_ordered_confirmatory_subset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    authorities: H6PredictionV3Authorities,
) -> None:
    import vfe4.training.h6_validation_campaign_v3 as campaign

    selection = validation_fixtures._tuning_selection()
    validation_directory = (tmp_path / "validation" / "COMPLETE").resolve()
    selection_directory = campaign.h6_tuning_selection_directory_v3(
        validation_directory
    )
    selection_directory.mkdir(parents=True)
    selected_attempts = (
        authorities.plan.confirmatory_attempts[5],
        authorities.plan.confirmatory_attempts[42],
    )
    tuning_items = tuple(
        SimpleNamespace(
            entry=SimpleNamespace(
                planned_attempt_sha256=attempt.planned_attempt_sha256
            )
        )
        for attempt in authorities.plan.tuning_attempts
    )
    confirmatory_items = tuple(
        SimpleNamespace(
            entry=SimpleNamespace(
                planned_attempt_sha256=attempt.planned_attempt_sha256
            )
        )
        for attempt in selected_attempts
    )
    monkeypatch.setattr(
        campaign,
        "read_h6_tuning_selection_v3",
        lambda *args, **kwargs: selection,
    )
    monkeypatch.setattr(
        campaign,
        "read_h6_checkpoint_catalog_v3",
        lambda *args, **kwargs: SimpleNamespace(
            tuning_items=tuning_items,
            confirmatory_items=confirmatory_items,
        ),
    )

    result = campaign.run_h6_validation_campaign_v3(
        authorities=authorities,
        store=object(),
        checkpoint_catalog_root=(tmp_path / "catalog").resolve(),
        maximum_checkpoint_bytes=16 * 1024 * 1024,
        validation_bundle_directory=validation_directory,
    )

    assert result.state == "TUNING_SELECTED"
    assert result.validation_bundle is None
    assert result.tuning_selection is selection
