from __future__ import annotations

import hashlib
import inspect
from dataclasses import FrozenInstanceError
from typing import get_args

import pytest

import vfe4.training.h6_experiment as experiment


def test_h6_experiment_surface_is_exact_immutable_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operations = {
        "plan": "AUTHORIZE_VFE4_H6_EXPERIMENT_PLAN_V1",
        "train": "AUTHORIZE_VFE4_H6_TRAINING_V1",
        "score_validation": "AUTHORIZE_VFE4_H6_VALIDATION_SCORING_V1",
        "reserve_test_opening": "AUTHORIZE_VFE4_H6_TEST_RESERVATION_V1",
        "score_test": "AUTHORIZE_VFE4_H6_ONE_TIME_TEST_SCORING_V1",
    }
    parameters = inspect.signature(experiment.run_h6_experiment).parameters
    assert all(
        item.kind is inspect.Parameter.KEYWORD_ONLY for item in parameters.values()
    )
    assert get_args(experiment.H6ExperimentOperation) == tuple(operations)

    sha = "a" * 64
    with pytest.raises(TypeError):
        experiment.H6ExperimentRunResult()  # type: ignore[call-arg]
    result = experiment._completed_result(
        operation="plan",
        config_sha256=sha,
        readiness_sha256=sha,
        authorization_sha256=experiment._AUTHORIZATION_SHA256["plan"],
        payload={"artifact_sha256": sha},
    )
    assert result.payload_sha256 != sha
    with pytest.raises(FrozenInstanceError):
        result.status = "COMPLETED"  # type: ignore[misc]

    with pytest.raises(ValueError, match="exact H6PredictionResolvedConfig"):
        experiment._revalidate_operation_inputs(
            config=object(),
            readiness=object(),
            prerequisite_refs=object(),
            operation="plan",
        )

    revalidated: list[str] = []
    monkeypatch.setattr(
        experiment,
        "_revalidate_operation_inputs",
        lambda **values: revalidated.append(values["operation"]),
    )
    for operation, phrase in operations.items():
        with pytest.raises(RuntimeError, match="unavailable"):
            experiment.run_h6_experiment(
                config=object(),  # type: ignore[arg-type]
                readiness=object(),  # type: ignore[arg-type]
                prerequisite_refs=object(),  # type: ignore[arg-type]
                operation=operation,  # type: ignore[arg-type]
                authorization_sha256=hashlib.sha256(phrase.encode("ascii")).hexdigest(),
            )
    assert revalidated == list(operations)

    with pytest.raises(PermissionError, match="authorization"):
        experiment.run_h6_experiment(
            config=object(),  # type: ignore[arg-type]
            readiness=object(),  # type: ignore[arg-type]
            prerequisite_refs=object(),  # type: ignore[arg-type]
            operation="plan",
            authorization_sha256="0" * 64,
        )
    assert revalidated == list(operations)
