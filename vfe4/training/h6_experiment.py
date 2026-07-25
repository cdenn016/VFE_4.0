"""Fail-closed orchestration surface for authorized H6 experiment operations."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, cast

from vfe4.config.schema import (
    H6DataConfig,
    H6PredictionV2ResolvedConfig,
    H6SourceIdentity,
)
from vfe4.types.h6 import (
    EndpointSmcProtocol,
    H6PredictionReadinessToken,
    H6TrainingSchedule,
    ObjectiveGateSpec,
    canonical_json_bytes,
)

if TYPE_CHECKING:
    from vfe4.training.h6_readiness import CurrentPredictionPrerequisiteRefs


_REPO_ROOT = Path(__file__).resolve().parents[2]
H6ExperimentOperation = Literal[
    "plan",
    "train",
    "score_validation",
    "reserve_test_opening",
    "score_test",
]
_OPERATIONS: tuple[H6ExperimentOperation, ...] = (
    "plan",
    "train",
    "score_validation",
    "reserve_test_opening",
    "score_test",
)
_AUTHORIZATION_PHRASES: dict[H6ExperimentOperation, str] = {
    "plan": "AUTHORIZE_VFE4_H6_EXPERIMENT_PLAN_V1",
    "train": "AUTHORIZE_VFE4_H6_TRAINING_V1",
    "score_validation": "AUTHORIZE_VFE4_H6_VALIDATION_SCORING_V1",
    "reserve_test_opening": "AUTHORIZE_VFE4_H6_TEST_RESERVATION_V1",
    "score_test": "AUTHORIZE_VFE4_H6_ONE_TIME_TEST_SCORING_V1",
}
_AUTHORIZATION_SHA256 = {
    operation: hashlib.sha256(phrase.encode("ascii")).hexdigest()
    for operation, phrase in _AUTHORIZATION_PHRASES.items()
}
_RESULT_ISSUER = object()
_OPERATION_BLOCKERS: dict[H6ExperimentOperation, str] = {
    "plan": (
        "H6 experiment plan orchestration is unavailable: no implementation "
        "reconstructs the frozen endpoint matrix, eligible matching reports, "
        "and exact H6AttemptSpec records"
    ),
    "train": (
        "H6 experiment training is unavailable: train_h6_attempt intentionally "
        "refuses execution until the separately authorized training engine exists"
    ),
    "score_validation": (
        "H6 validation scoring is unavailable: no checkpoint-set validation "
        "scorer or atomic validation-result publisher exists"
    ),
    "reserve_test_opening": (
        "H6 test-opening reservation is unavailable: no experiment-level "
        "checkpoint/current-candidate identity assembler exists"
    ),
    "score_test": (
        "H6 test scoring is unavailable: no experiment-level validated-opening "
        "consumer or endpoint result publisher exists"
    ),
}


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True, init=False)
class H6ExperimentRunResult:
    """Immutable envelope returned only by a genuinely completed operation."""

    result_schema: Literal["h6-experiment-run-result-v1"]
    operation: H6ExperimentOperation
    status: Literal["COMPLETED"]
    config_sha256: str
    readiness_sha256: str
    authorization_sha256: str
    payload_sha256: str
    result_sha256: str
    _payload: Mapping[str, object] = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)

    def __new__(cls) -> "H6ExperimentRunResult":
        raise TypeError(
            "H6ExperimentRunResult is issued only by a completed bound operation"
        )

    def __post_init__(self) -> None:
        if self.result_schema != "h6-experiment-run-result-v1":
            raise ValueError("unsupported H6 experiment result schema")
        if self.operation not in _OPERATIONS or self.status != "COMPLETED":
            raise ValueError("H6 experiment result is not an exact completed operation")
        for name in (
            "config_sha256",
            "readiness_sha256",
            "authorization_sha256",
            "payload_sha256",
            "result_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if not hmac.compare_digest(
            self.authorization_sha256, _AUTHORIZATION_SHA256[self.operation]
        ):
            raise ValueError("result authorization does not match its operation")
        if self._issuer is not _RESULT_ISSUER or not isinstance(
            self._payload, Mapping
        ):
            raise ValueError("completed results require the private bound issuer")
        payload_sha256 = hashlib.sha256(
            canonical_json_bytes(self._payload)
        ).hexdigest()
        if self.payload_sha256 != payload_sha256:
            raise ValueError("completed result payload identity is stale")
        result_payload = {
            "result_schema": self.result_schema,
            "operation": self.operation,
            "status": self.status,
            "config_sha256": self.config_sha256,
            "readiness_sha256": self.readiness_sha256,
            "authorization_sha256": self.authorization_sha256,
            "payload_sha256": self.payload_sha256,
        }
        expected = hashlib.sha256(
            b"VFE4-H6-EXPERIMENT-RUN-RESULT-V1\x00"
            + canonical_json_bytes(result_payload)
        ).hexdigest()
        if self.result_sha256 != expected:
            raise ValueError("completed result identity is stale")


def _completed_result(
    *,
    operation: H6ExperimentOperation,
    config_sha256: str,
    readiness_sha256: str,
    authorization_sha256: str,
    payload: Mapping[str, object],
) -> H6ExperimentRunResult:
    """Private constructor reserved for a future genuinely completed branch."""

    owned_payload = json.loads(canonical_json_bytes(payload))
    if type(owned_payload) is not dict:
        raise ValueError("completed operation payload must be a JSON object")
    payload_sha256 = hashlib.sha256(
        canonical_json_bytes(owned_payload)
    ).hexdigest()
    result_payload = {
        "result_schema": "h6-experiment-run-result-v1",
        "operation": operation,
        "status": "COMPLETED",
        "config_sha256": config_sha256,
        "readiness_sha256": readiness_sha256,
        "authorization_sha256": authorization_sha256,
        "payload_sha256": payload_sha256,
    }
    result = object.__new__(H6ExperimentRunResult)
    values: dict[str, object] = {
        **result_payload,
        "result_sha256": hashlib.sha256(
            b"VFE4-H6-EXPERIMENT-RUN-RESULT-V1\x00"
            + canonical_json_bytes(result_payload)
        ).hexdigest(),
        "_payload": MappingProxyType(owned_payload),
        "_issuer": _RESULT_ISSUER,
    }
    for name, value in values.items():
        object.__setattr__(result, name, value)
    result.__post_init__()
    return result


def _revalidate_config(config: object) -> H6PredictionV2ResolvedConfig:
    if type(config) is not H6PredictionV2ResolvedConfig:
        raise ValueError(
            "amended H6 dispatch requires an exact "
            "H6PredictionV2ResolvedConfig"
        )
    if (
        config.schema_version != "h6-prediction-config-v2"
        or config.operation != "H6-Prediction"
        or type(config.source) is not H6SourceIdentity
        or type(config.data) is not H6DataConfig
        or type(config.training_schedule) is not H6TrainingSchedule
        or type(config.endpoint_smc_protocol) is not EndpointSmcProtocol
        or type(config.objective_gate) is not ObjectiveGateSpec
        or not isinstance(config.artifact_root, Path)
    ):
        raise ValueError("H6 Prediction configuration surface is stale")
    if (
        type(config.source.git_head) is not str
        or len(config.source.git_head) != 40
        or any(
            character not in "0123456789abcdef" for character in config.source.git_head
        )
        or config.data.schema_version != "h6-data-config-v1"
    ):
        raise ValueError("H6 Prediction source/data identity is stale")
    _require_sha256(config.source.dirty_digest, "source.dirty_digest")
    _require_sha256(config.source.source_sha256, "source.source_sha256")
    if tuple(gate for gate, _ in config.correctness_manifests) != (
        "H1",
        "H2",
        "H3",
        "H5",
    ):
        raise ValueError("H6 Prediction correctness manifests are not exact")
    for gate, digest in config.correctness_manifests:
        _require_sha256(digest, f"correctness_manifests[{gate}]")
    for name in (
        "h1_prefix_prior_manifest_sha256",
        "h1_prefix_prior_generative_factor_schema_sha256",
        "smc_bias_semantics_sha256",
        "smc_validation_manifest_sha256",
        "prefix_certificate_set_sha256",
        "h5_update_binding_sha256",
        "critical_values_sha256",
        "attribution_matrix_sha256",
        "matching_set_sha256",
        "data_identity_sha256",
        "access_policy_sha256",
    ):
        _require_sha256(getattr(config, name), name)
    config.training_schedule.outer.__post_init__()
    for phase_schedule in config.training_schedule.endpoint_phases:
        phase_schedule.__post_init__()
    config.training_schedule.__post_init__()
    config.endpoint_smc_protocol.__post_init__()
    config.objective_gate.__post_init__()
    from vfe4.evaluation.smc_uncertainty import SMC_BIAS_SEMANTICS

    if (
        config.smc_bias_semantics_sha256
        != SMC_BIAS_SEMANTICS.semantics_sha256
    ):
        raise ValueError(
            "H6 Prediction SMC bias semantics identity is stale"
        )
    if (
        type(config.canonical_json) is not str
        or _require_sha256(config.config_sha256, "config_sha256")
        != hashlib.sha256(config.canonical_json.encode("utf-8")).hexdigest()
    ):
        raise ValueError("H6 Prediction configuration identity is stale")
    from vfe4.config import resolve_h6_prediction_v2_config

    try:
        raw = json.loads(config.canonical_json)
        canonical = resolve_h6_prediction_v2_config(
            raw,
            repo_root=_REPO_ROOT,
        )
    except (TypeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(
            "H6 Prediction configuration is not canonical v2"
        ) from exc
    if canonical != config:
        raise ValueError(
            "H6 Prediction typed fields differ from the resolved canonical config"
        )
    return config


def _revalidate_operation_inputs(
    *,
    config: object,
    readiness: object,
    prerequisite_refs: object,
    operation: H6ExperimentOperation,
) -> None:
    from vfe4.training.h6_readiness import (
        CurrentPredictionPrerequisiteRefs,
        _revalidate_h6_prediction_readiness_inputs,
    )

    typed_config = _revalidate_config(config)
    if type(readiness) is not H6PredictionReadinessToken:
        raise ValueError("readiness must be an exact H6PredictionReadinessToken")
    if type(prerequisite_refs) is not CurrentPredictionPrerequisiteRefs:
        raise ValueError(
            "prerequisite_refs must be exact CurrentPredictionPrerequisiteRefs"
        )
    readiness.__post_init__()
    if readiness.readiness_schema != "h6-prediction-readiness-v2":
        raise ValueError("legacy v1 readiness cannot dispatch amended H6")
    expected = {
        "git_head": typed_config.source.git_head,
        "dirty_digest": typed_config.source.dirty_digest,
        "experiment_config_sha256": typed_config.config_sha256,
        "correctness_manifests": typed_config.correctness_manifests,
        "h1_prefix_prior_manifest_sha256": (
            typed_config.h1_prefix_prior_manifest_sha256
        ),
        "h1_prefix_prior_generative_factor_schema_sha256": (
            typed_config.h1_prefix_prior_generative_factor_schema_sha256
        ),
        "smc_bias_semantics_sha256": (
            typed_config.smc_bias_semantics_sha256
        ),
        "objective_gate_spec_sha256": (
            typed_config.objective_gate.spec_sha256
        ),
        "h5_update_binding_sha256": typed_config.h5_update_binding_sha256,
        "h6_training_schedule_sha256": (typed_config.training_schedule.schedule_sha256),
        "smc_validation_manifest_sha256": (typed_config.smc_validation_manifest_sha256),
        "critical_values_sha256": typed_config.critical_values_sha256,
        "endpoint_smc_protocol_sha256": (
            typed_config.endpoint_smc_protocol.protocol_sha256
        ),
        "attribution_matrix_sha256": typed_config.attribution_matrix_sha256,
        "matching_set_sha256": typed_config.matching_set_sha256,
        "prefix_certificate_set_sha256": (typed_config.prefix_certificate_set_sha256),
        "data_identity_sha256": typed_config.data_identity_sha256,
        "access_policy_sha256": typed_config.access_policy_sha256,
    }
    for name, value in expected.items():
        if getattr(readiness, name) != value:
            raise ValueError(f"readiness {name} does not match the exact config")

    fresh = _revalidate_h6_prediction_readiness_inputs(
        config=typed_config,
        prerequisite_refs=prerequisite_refs,
    )
    fresh.__post_init__()
    if fresh != readiness or not hmac.compare_digest(
        fresh.readiness_sha256, readiness.readiness_sha256
    ):
        raise ValueError(
            "readiness does not match the current exact prerequisite identities"
        )
    if operation not in _OPERATIONS:  # defensive after the public operation check
        raise ValueError("unsupported H6 experiment operation")


def run_h6_experiment(
    *,
    config: H6PredictionV2ResolvedConfig,
    readiness: H6PredictionReadinessToken,
    prerequisite_refs: CurrentPredictionPrerequisiteRefs,
    operation: Literal[
        "plan",
        "train",
        "score_validation",
        "reserve_test_opening",
        "score_test",
    ],
    authorization_sha256: str,
) -> H6ExperimentRunResult:
    """Authorize and revalidate one exact H6 operation before lazy dispatch.

    Source buildout does not yet contain a complete experiment-level planner,
    training engine, validation scorer, durable-opening assembler, or test
    scorer.  Each branch therefore raises its precise blocker after all
    authorization and current-identity checks and before importing any of
    those effectful layers.
    """

    if type(operation) is not str or operation not in _OPERATIONS:
        raise ValueError("unsupported H6 experiment operation")
    typed_operation = cast(H6ExperimentOperation, operation)
    supplied_authorization = _require_sha256(
        authorization_sha256, "authorization_sha256"
    )
    if not hmac.compare_digest(
        supplied_authorization, _AUTHORIZATION_SHA256[typed_operation]
    ):
        raise PermissionError(
            f"{typed_operation} authorization does not match its explicit phrase"
        )

    _revalidate_operation_inputs(
        config=config,
        readiness=readiness,
        prerequisite_refs=prerequisite_refs,
        operation=typed_operation,
    )
    raise RuntimeError(_OPERATION_BLOCKERS[typed_operation])


__all__ = ["H6ExperimentRunResult", "run_h6_experiment"]
