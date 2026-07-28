from __future__ import annotations

from pathlib import Path

import pytest
import torch

import test_h6_readiness_v3 as readiness_fixtures
import test_h6_validation_v3 as validation_fixtures
from vfe4.artifacts.atomic import ArtifactPublicationError
from vfe4.artifacts.h6_prediction_v3 import H6PredictionV3Authorities
from vfe4.training.arms import build_arm_model
from vfe4.training.h6_execution_v3 import bind_h6_executable_attempt_v3
from vfe4.training.h6_experiment_v3 import plan_h6_experiment_v3
from vfe4.training.h6_readiness import (
    _derive_h6_prediction_readiness_v3 as validate_h6_prediction_readiness_v3,
)


@pytest.fixture(scope="module")
def tuning_authority(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[H6PredictionV3Authorities, object, object]:
    matching_set = readiness_fixtures._matching_set()
    config = readiness_fixtures._config(
        matching_set=matching_set,
        artifact_root=tmp_path_factory.mktemp("h6-catalog-authority"),
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
    authorities = H6PredictionV3Authorities.create(
        config=config,
        matching_set=matching_set,
        readiness=readiness,
        plan=plan,
    )
    attempt = plan.tuning_attempts[0]
    executable = bind_h6_executable_attempt_v3(
        authorities=authorities,
        planned_attempt=attempt,
    )
    checkpoint = validation_fixtures._terminal_checkpoint(
        attempt,
        runtime=config.runtime,
        cell=executable.tuning_cell,
    )
    return authorities, executable, checkpoint


def test_checkpoint_catalog_publishes_and_reopens_one_bound_entry(
    tmp_path: Path,
    tuning_authority: tuple[H6PredictionV3Authorities, object, object],
) -> None:
    from vfe4.training.h6_checkpoint_catalog_v3 import (
        H6CheckpointCatalogV3,
        publish_h6_checkpoint_catalog_entry_v3,
        read_h6_checkpoint_catalog_v3,
    )

    authorities, executable, checkpoint = tuning_authority
    checkpoint_path = (tmp_path / "checkpoints" / "tuning.h6v3").resolve()
    checkpoint_path.parent.mkdir()
    checkpoint_path.write_bytes(checkpoint.to_bytes())
    catalog_root = (tmp_path / "catalog").resolve()

    published = publish_h6_checkpoint_catalog_entry_v3(
        catalog_root=catalog_root,
        checkpoint_path=checkpoint_path,
        maximum_checkpoint_bytes=16 * 1024 * 1024,
        executable_attempt=executable,
        checkpoint=checkpoint,
    )
    catalog = read_h6_checkpoint_catalog_v3(
        catalog_root,
        authorities=authorities,
        maximum_checkpoint_bytes=16 * 1024 * 1024,
    )

    assert published.name == executable.planned_attempt.planned_attempt_sha256
    assert type(catalog) is H6CheckpointCatalogV3
    assert catalog.authority_sha256 == authorities.authority_sha256
    assert len(catalog.items) == 1
    item = catalog.items[0]
    assert item.executable_attempt == executable
    assert item.checkpoint == checkpoint
    assert item.entry.checkpoint_path == checkpoint_path.as_posix()
    assert item.entry.stage == "tuning"
    assert item.entry.tuning_cell_sha256 == executable.tuning_cell.cell_sha256
    with pytest.raises(ArtifactPublicationError, match="already exists"):
        publish_h6_checkpoint_catalog_entry_v3(
            catalog_root=catalog_root,
            checkpoint_path=checkpoint_path,
            maximum_checkpoint_bytes=16 * 1024 * 1024,
            executable_attempt=executable,
            checkpoint=checkpoint,
        )


def test_checkpoint_catalog_rejects_foreign_inventory_and_changed_checkpoint(
    tmp_path: Path,
    tuning_authority: tuple[H6PredictionV3Authorities, object, object],
) -> None:
    from vfe4.training.h6_checkpoint_catalog_v3 import (
        publish_h6_checkpoint_catalog_entry_v3,
        read_h6_checkpoint_catalog_v3,
    )

    authorities, executable, checkpoint = tuning_authority
    checkpoint_path = (tmp_path / "checkpoints" / "tuning.h6v3").resolve()
    checkpoint_path.parent.mkdir()
    checkpoint_path.write_bytes(checkpoint.to_bytes())
    catalog_root = (tmp_path / "catalog").resolve()
    publish_h6_checkpoint_catalog_entry_v3(
        catalog_root=catalog_root,
        checkpoint_path=checkpoint_path,
        maximum_checkpoint_bytes=16 * 1024 * 1024,
        executable_attempt=executable,
        checkpoint=checkpoint,
    )

    (catalog_root / "foreign").mkdir()
    with pytest.raises(ArtifactPublicationError, match="inventory|foreign|plan"):
        read_h6_checkpoint_catalog_v3(
            catalog_root,
            authorities=authorities,
            maximum_checkpoint_bytes=16 * 1024 * 1024,
        )
    (catalog_root / "foreign").rmdir()
    checkpoint_path.write_bytes(checkpoint.to_bytes() + b"changed")
    with pytest.raises(
        (ArtifactPublicationError, ValueError),
        match="checkpoint|canonical|digest|trailing",
    ):
        read_h6_checkpoint_catalog_v3(
            catalog_root,
            authorities=authorities,
            maximum_checkpoint_bytes=16 * 1024 * 1024,
        )


def test_evaluation_arm_is_exact_frozen_cpu_model(
    tuning_authority: tuple[H6PredictionV3Authorities, object, object],
) -> None:
    from vfe4.training.h6_validation_v3 import (
        H6EvaluationArmV3,
        build_h6_evaluation_arm_v3,
    )

    authorities, executable, _ = tuning_authority
    model = build_arm_model(executable.endpoint_config)
    checkpoint = validation_fixtures._terminal_checkpoint(
        executable.planned_attempt,
        runtime=authorities.config.runtime,
        cell=executable.tuning_cell,
        model=model,
    )

    evaluation = build_h6_evaluation_arm_v3(
        checkpoint,
        plan=authorities.plan,
        planned_attempt=executable.planned_attempt,
        evaluation_role="validation",
    )

    assert type(evaluation) is H6EvaluationArmV3
    assert evaluation.checkpoint_sha256 == checkpoint.checkpoint_sha256
    assert evaluation.planned_attempt_sha256 == (
        executable.planned_attempt.planned_attempt_sha256
    )
    assert evaluation.training_seed == executable.planned_attempt.training_seed
    assert evaluation.config == executable.endpoint_config
    assert evaluation.evaluation_role == "validation"
    assert evaluation.model.training is False
    assert all(parameter.device.type == "cpu" for parameter in evaluation.model.parameters())
    assert all(parameter.dtype is torch.float64 for parameter in evaluation.model.parameters())
    assert all(not parameter.requires_grad for parameter in evaluation.model.parameters())
    evaluation.__post_init__()
    with pytest.raises(TypeError):
        H6EvaluationArmV3(
            checkpoint_sha256=evaluation.checkpoint_sha256,
            checkpoint_bytes_sha256=evaluation.checkpoint_bytes_sha256,
            planned_attempt_sha256=evaluation.planned_attempt_sha256,
            attempt_spec_sha256=evaluation.attempt_spec_sha256,
            endpoint_config_id=evaluation.endpoint_config_id,
            endpoint_config_sha256=evaluation.endpoint_config_sha256,
            training_seed=evaluation.training_seed,
            config=evaluation.config,
            model=evaluation.model,
            evaluation_role=evaluation.evaluation_role,
            _checkpoint_model_records=evaluation._checkpoint_model_records,
        )
    with torch.no_grad():
        next(evaluation.model.parameters()).add_(1.0)
    with pytest.raises(ValueError, match="model|checkpoint|bytes|state"):
        evaluation.__post_init__()
