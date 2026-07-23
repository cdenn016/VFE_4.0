"""Click-to-run H6 readiness and experiment launcher.

Edit ``CONFIG`` in this file, enable exactly one operation, provide that
operation's explicit authorization phrase, and click Run.  Importing this
module and running it with every operation disabled performs no repository,
artifact, data, model, or training work.
"""

from __future__ import annotations

import hashlib
import hmac
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


CONFIG: dict[str, object] = {
    "launcher_schema": "vfe4-train-click-run-v1",
    "operations": {
        "prediction_readiness": {
            "enabled": False,
            # Set only when intentionally validating the frozen prerequisite set.
            "authorization": None,
            "config": {
                "scientific_config": {},
                "prerequisite_refs": {},
            },
        },
        "plan": {
            "enabled": False,
            "authorization": None,
            "config": {
                "scientific_config": {},
                "prerequisite_refs": {},
                "readiness_artifact_root": None,
            },
        },
        "train": {
            "enabled": False,
            "authorization": None,
            "config": {
                "scientific_config": {},
                "prerequisite_refs": {},
                "readiness_artifact_root": None,
            },
        },
        "score_validation": {
            "enabled": False,
            "authorization": None,
            "config": {
                "scientific_config": {},
                "prerequisite_refs": {},
                "readiness_artifact_root": None,
            },
        },
        "reserve_test_opening": {
            "enabled": False,
            "authorization": None,
            "config": {
                "scientific_config": {},
                "prerequisite_refs": {},
                "readiness_artifact_root": None,
            },
        },
        "score_test": {
            "enabled": False,
            "authorization": None,
            "config": {
                "scientific_config": {},
                "prerequisite_refs": {},
                "readiness_artifact_root": None,
            },
        },
    },
}


_AUTHORIZATION_PHRASES = {
    "prediction_readiness": "AUTHORIZE_VFE4_H6_PREDICTION_READINESS_V1",
    "plan": "AUTHORIZE_VFE4_H6_EXPERIMENT_PLAN_V1",
    "train": "AUTHORIZE_VFE4_H6_TRAINING_V1",
    "score_validation": "AUTHORIZE_VFE4_H6_VALIDATION_SCORING_V1",
    "reserve_test_opening": "AUTHORIZE_VFE4_H6_TEST_RESERVATION_V1",
    "score_test": "AUTHORIZE_VFE4_H6_ONE_TIME_TEST_SCORING_V1",
}
_OPERATION_NAMES = tuple(_AUTHORIZATION_PHRASES)
_REPO_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class TrainLauncherResult:
    launcher_schema: Literal["vfe4-train-click-run-v1"]
    operation: str | None
    status: Literal["IDLE", "COMPLETED"]
    _payload: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.launcher_schema != "vfe4-train-click-run-v1":
            raise ValueError("unsupported train launcher schema")
        if self.status not in ("IDLE", "COMPLETED"):
            raise ValueError("train launcher status must be IDLE or COMPLETED")
        if self.status == "IDLE":
            if self.operation is not None or self._payload is not None:
                raise ValueError("an idle launcher cannot retain an operation")
        elif self.operation not in _OPERATION_NAMES or self._payload is None:
            raise ValueError("a completed launcher result requires a payload")


def _mapping(value: object, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(
        type(key) is not str for key in value
    ):
        raise ValueError(f"{location} must be a string-keyed mapping")
    return value


def _selected_operation(
    config: Mapping[str, object],
) -> tuple[str, Mapping[str, object], str] | None:
    if set(config) != {"launcher_schema", "operations"}:
        raise ValueError("train CONFIG has unknown or missing root keys")
    if config["launcher_schema"] != "vfe4-train-click-run-v1":
        raise ValueError("train CONFIG launcher_schema is unsupported")
    operations = _mapping(config["operations"], "operations")
    if tuple(operations) != _OPERATION_NAMES:
        raise ValueError("train CONFIG operations are incomplete or reordered")
    enabled: list[tuple[str, Mapping[str, object]]] = []
    for name in _OPERATION_NAMES:
        entry = _mapping(operations[name], f"operations.{name}")
        if set(entry) != {"enabled", "authorization", "config"}:
            raise ValueError(f"operations.{name} has unknown or missing keys")
        if type(entry["enabled"]) is not bool:
            raise ValueError(f"operations.{name}.enabled must be boolean")
        _mapping(entry["config"], f"operations.{name}.config")
        if entry["enabled"]:
            enabled.append((name, entry))
    if not enabled:
        return None
    if len(enabled) != 1:
        raise ValueError("enable exactly one train operation")
    name, entry = enabled[0]
    authorization = entry["authorization"]
    if type(authorization) is not str or not hmac.compare_digest(
        authorization,
        _AUTHORIZATION_PHRASES[name],
    ):
        raise PermissionError(
            f"operations.{name}.authorization does not equal its explicit phrase"
        )
    return (
        name,
        _mapping(entry["config"], f"operations.{name}.config"),
        authorization,
    )


def _inputs(raw: Mapping[str, object]) -> tuple[object, object]:
    if set(raw) != {"scientific_config", "prerequisite_refs"}:
        raise ValueError(
            "readiness config must contain scientific_config and prerequisite_refs"
        )
    scientific = _mapping(raw["scientific_config"], "scientific_config")
    references = _mapping(raw["prerequisite_refs"], "prerequisite_refs")
    return scientific, references


def _experiment_inputs(
    raw: Mapping[str, object],
) -> tuple[object, object, Path]:
    if set(raw) != {
        "scientific_config",
        "prerequisite_refs",
        "readiness_artifact_root",
    }:
        raise ValueError(
            "experiment config must contain scientific_config, "
            "prerequisite_refs, and readiness_artifact_root"
        )
    scientific = _mapping(raw["scientific_config"], "scientific_config")
    references = _mapping(raw["prerequisite_refs"], "prerequisite_refs")
    root = raw["readiness_artifact_root"]
    if type(root) is str:
        readiness_root = Path(root)
    elif isinstance(root, Path):
        readiness_root = root
    else:
        raise ValueError("readiness_artifact_root must be a path string or Path")
    if not readiness_root.is_absolute():
        readiness_root = _REPO_ROOT / readiness_root
    return scientific, references, readiness_root.resolve(strict=False)


def _run_readiness(raw: Mapping[str, object]) -> object:
    from vfe4.config import resolve_h6_prediction_config
    from vfe4.training.h6_readiness import (
        CurrentPredictionPrerequisiteRefs,
        validate_h6_prediction_readiness,
    )

    scientific, references = _inputs(raw)
    resolved = resolve_h6_prediction_config(scientific, repo_root=_REPO_ROOT)
    typed_refs = CurrentPredictionPrerequisiteRefs.from_mapping(
        references, repo_root=_REPO_ROOT
    )
    return validate_h6_prediction_readiness(
        config=resolved,
        prerequisite_refs=typed_refs,
    )


def _run_experiment(
    operation: str,
    raw: Mapping[str, object],
    authorization: str,
) -> object:
    from vfe4.config import resolve_h6_prediction_config
    from vfe4.training.h6_experiment import run_h6_experiment
    from vfe4.training.h6_readiness import (
        CurrentPredictionPrerequisiteRefs,
        _load_published_h6_prediction_readiness,
    )

    scientific, references, readiness_root = _experiment_inputs(raw)
    resolved = resolve_h6_prediction_config(scientific, repo_root=_REPO_ROOT)
    typed_refs = CurrentPredictionPrerequisiteRefs.from_mapping(
        references, repo_root=_REPO_ROOT
    )
    readiness = _load_published_h6_prediction_readiness(
        config=resolved,
        prerequisite_refs=typed_refs,
        artifact_root=readiness_root,
    )
    return run_h6_experiment(
        config=resolved,
        readiness=readiness,
        prerequisite_refs=typed_refs,
        operation=operation,
        authorization_sha256=hashlib.sha256(
            authorization.encode("ascii")
        ).hexdigest(),
    )


def main(config: Mapping[str, object] = CONFIG) -> TrainLauncherResult:
    selected = _selected_operation(_mapping(config, "CONFIG"))
    if selected is None:
        return TrainLauncherResult(
            "vfe4-train-click-run-v1", None, "IDLE"
        )
    operation, raw, authorization = selected
    payload = (
        _run_readiness(raw)
        if operation == "prediction_readiness"
        else _run_experiment(operation, raw, authorization)
    )
    return TrainLauncherResult(
        "vfe4-train-click-run-v1", operation, "COMPLETED", payload
    )


def _script_main() -> int:
    try:
        result = main()
    except (OSError, PermissionError, RuntimeError, TypeError, ValueError) as exc:
        print(f"VFE4 train operation unavailable: {exc}", file=sys.stderr)
        return 2
    if result.status == "IDLE":
        print("VFE4 train launcher is idle; enable exactly one CONFIG operation.")
    else:
        print(f"VFE4 train operation completed: {result.operation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_script_main())
