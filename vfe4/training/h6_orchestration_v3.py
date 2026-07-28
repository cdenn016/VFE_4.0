"""Path-only click orchestration for executable H6-Prediction v3."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from vfe4.artifacts.atomic import ArtifactPublicationError
from vfe4.artifacts.h6_prediction_v3 import (
    H6PredictionV3Authorities,
    publish_h6_prediction_v3_authorities,
    read_h6_prediction_v3_authorities,
)
from vfe4.config import H6PredictionV3ResolvedConfig
from vfe4.data.byte_tokenizer import ByteTokenizerV1
from vfe4.data.h6_sealed_store_v3 import (
    reopen_authenticated_blinded_store_v3,
)
from vfe4.data.wikitext2 import BlindedCorpusStore
from vfe4.training.h6_experiment_v3 import plan_h6_experiment_v3
from vfe4.training.h6_matching_v3 import (
    H6MatchingSetV3,
    build_h6_matching_set_v3,
)
from vfe4.training.h6_readiness import (
    read_h6_prefix_authorities_for_scoring_v3,
    reopen_h6_prediction_v3_prerequisite_evidence,
    validate_h6_prediction_readiness_v3,
)


H6V3Operation = Literal[
    "prediction_readiness",
    "plan",
    "train",
    "score_validation",
]

_LOWER_HEX = frozenset("0123456789abcdef")
_OPERATION_AUTHORIZATION_SHA256 = {
    operation: hashlib.sha256(phrase.encode("ascii")).hexdigest()
    for operation, phrase in (
        (
            "prediction_readiness",
            "AUTHORIZE_VFE4_H6_PREDICTION_READINESS_V1",
        ),
        ("plan", "AUTHORIZE_VFE4_H6_EXPERIMENT_PLAN_V1"),
        ("train", "AUTHORIZE_VFE4_H6_TRAINING_V1"),
        (
            "score_validation",
            "AUTHORIZE_VFE4_H6_VALIDATION_SCORING_V1",
        ),
    )
}
_OPERATION_PATH_FIELDS = frozenset(
    {
        "scientific_config",
        "correctness_artifact_roots",
        "h1_prefix_prior_artifact_root",
        "smc_accuracy_artifact_root",
        "h6_prefix_artifact_root",
        "h6_prefix_manifest_sha256",
        "h6_prefix_junit_sha256",
        "blinded_store_manifest_path",
        "blinded_store_artifact_root",
        "authorities_run_root",
        "authorities_run_name",
        "authorities_directory",
        "planned_attempt_sha256",
        "checkpoint_path",
        "maximum_checkpoint_bytes",
        "validation_bundle_directory",
        "transaction_pointer_root",
        "transaction_pointer_name",
    }
)


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _absolute_path(value: object, name: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise ValueError(f"{name} must be a path string or pathlib Path")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute")
    resolved = path.resolve(strict=False)
    if resolved.as_posix() != path.as_posix():
        raise ValueError(f"{name} must be canonical")
    return resolved


def _portable_name(value: object, name: str) -> str:
    if (
        type(value) is not str
        or not value
        or value in (".", "..")
        or "/" in value
        or "\\" in value
    ):
        raise ValueError(f"{name} must be one portable component")
    return value


def _correctness_artifact_roots(
    value: object,
) -> tuple[tuple[Literal["H1", "H2", "H3", "H5"], Path], ...]:
    if not isinstance(value, Mapping) or set(value) != {"H1", "H2", "H3", "H5"}:
        raise ValueError(
            "correctness_artifact_roots must contain exactly H1, H2, H3, H5 "
            "in frozen order"
        )
    return tuple(
        (
            gate,  # type: ignore[misc]
            _absolute_path(value[gate], f"correctness_artifact_roots[{gate}]"),
        )
        for gate in ("H1", "H2", "H3", "H5")
    )


def _contains_callable(value: object) -> bool:
    if callable(value):
        return True
    if isinstance(value, Mapping):
        return any(
            _contains_callable(key) or _contains_callable(item)
            for key, item in value.items()
        )
    if isinstance(value, (tuple, list)):
        return any(_contains_callable(item) for item in value)
    return False


@dataclass(frozen=True, slots=True)
class H6OperationPathsV3:
    correctness_artifact_roots: tuple[
        tuple[Literal["H1", "H2", "H3", "H5"], Path], ...
    ]
    h1_prefix_prior_artifact_root: Path
    smc_accuracy_artifact_root: Path
    h6_prefix_artifact_root: Path
    h6_prefix_manifest_sha256: str
    h6_prefix_junit_sha256: str
    blinded_store_manifest_path: Path
    blinded_store_artifact_root: Path
    authorities_run_root: Path
    authorities_run_name: str
    authorities_directory: Path
    planned_attempt_sha256: str
    checkpoint_path: Path
    maximum_checkpoint_bytes: int
    validation_bundle_directory: Path
    transaction_pointer_root: Path
    transaction_pointer_name: str

    def __post_init__(self) -> None:
        for name in (
            "h1_prefix_prior_artifact_root",
            "smc_accuracy_artifact_root",
            "h6_prefix_artifact_root",
            "blinded_store_manifest_path",
            "blinded_store_artifact_root",
            "authorities_run_root",
            "authorities_directory",
            "checkpoint_path",
            "validation_bundle_directory",
            "transaction_pointer_root",
        ):
            _absolute_path(getattr(self, name), name)
        if (
            type(self.correctness_artifact_roots) is not tuple
            or tuple(gate for gate, _ in self.correctness_artifact_roots)
            != ("H1", "H2", "H3", "H5")
        ):
            raise ValueError(
                "correctness_artifact_roots must contain exactly H1, H2, H3, H5 "
                "in frozen order"
            )
        for gate, root in self.correctness_artifact_roots:
            _absolute_path(root, f"correctness_artifact_roots[{gate}]")
        _require_sha256(
            self.h6_prefix_manifest_sha256,
            "h6_prefix_manifest_sha256",
        )
        _require_sha256(
            self.h6_prefix_junit_sha256,
            "h6_prefix_junit_sha256",
        )
        _portable_name(self.authorities_run_name, "authorities_run_name")
        _portable_name(
            self.transaction_pointer_name,
            "transaction_pointer_name",
        )
        _require_sha256(
            self.planned_attempt_sha256,
            "planned_attempt_sha256",
        )
        if (
            type(self.maximum_checkpoint_bytes) is not int
            or self.maximum_checkpoint_bytes <= 0
        ):
            raise ValueError(
                "maximum_checkpoint_bytes must be a positive exact integer"
            )
        if (
            self.authorities_directory
            != self.authorities_run_root / self.authorities_run_name
        ):
            raise ValueError(
                "authorities_directory must equal run root plus run name"
            )

    @property
    def checkpoint_catalog_root(self) -> Path:
        """Return the sole catalog root associated with this checkpoint lane."""

        return self.checkpoint_path.parent / "CATALOG"

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, object],
    ) -> "H6OperationPathsV3":
        if not isinstance(raw, Mapping) or frozenset(raw) != _OPERATION_PATH_FIELDS:
            raise ValueError(
                "operation config field inventory is incomplete or contains unknown keys"
            )
        if _contains_callable(raw):
            raise ValueError("operation config cannot contain callbacks")
        scientific = raw["scientific_config"]
        if not isinstance(scientific, Mapping) or not scientific:
            raise ValueError("scientific_config must be one nonempty mapping")
        return cls(
            correctness_artifact_roots=_correctness_artifact_roots(
                raw["correctness_artifact_roots"]
            ),
            h1_prefix_prior_artifact_root=_absolute_path(
                raw["h1_prefix_prior_artifact_root"],
                "h1_prefix_prior_artifact_root",
            ),
            smc_accuracy_artifact_root=_absolute_path(
                raw["smc_accuracy_artifact_root"],
                "smc_accuracy_artifact_root",
            ),
            h6_prefix_artifact_root=_absolute_path(
                raw["h6_prefix_artifact_root"],
                "h6_prefix_artifact_root",
            ),
            h6_prefix_manifest_sha256=_require_sha256(
                raw["h6_prefix_manifest_sha256"],
                "h6_prefix_manifest_sha256",
            ),
            h6_prefix_junit_sha256=_require_sha256(
                raw["h6_prefix_junit_sha256"],
                "h6_prefix_junit_sha256",
            ),
            blinded_store_manifest_path=_absolute_path(
                raw["blinded_store_manifest_path"],
                "blinded_store_manifest_path",
            ),
            blinded_store_artifact_root=_absolute_path(
                raw["blinded_store_artifact_root"],
                "blinded_store_artifact_root",
            ),
            authorities_run_root=_absolute_path(
                raw["authorities_run_root"],
                "authorities_run_root",
            ),
            authorities_run_name=_portable_name(
                raw["authorities_run_name"],
                "authorities_run_name",
            ),
            authorities_directory=_absolute_path(
                raw["authorities_directory"],
                "authorities_directory",
            ),
            planned_attempt_sha256=_require_sha256(
                raw["planned_attempt_sha256"],
                "planned_attempt_sha256",
            ),
            checkpoint_path=_absolute_path(
                raw["checkpoint_path"],
                "checkpoint_path",
            ),
            maximum_checkpoint_bytes=raw["maximum_checkpoint_bytes"],  # type: ignore[arg-type]
            validation_bundle_directory=_absolute_path(
                raw["validation_bundle_directory"],
                "validation_bundle_directory",
            ),
            transaction_pointer_root=_absolute_path(
                raw["transaction_pointer_root"],
                "transaction_pointer_root",
            ),
            transaction_pointer_name=_portable_name(
                raw["transaction_pointer_name"],
                "transaction_pointer_name",
            ),
        )


def _reopen_store_for_config(
    *,
    config: H6PredictionV3ResolvedConfig,
    paths: H6OperationPathsV3,
) -> BlindedCorpusStore:
    store = reopen_authenticated_blinded_store_v3(
        paths.blinded_store_manifest_path,
        paths.blinded_store_artifact_root,
    )
    if type(store) is not BlindedCorpusStore:
        raise ValueError("sealed-store reopener did not return an exact store")
    identity = store.data_identity
    if (
        store.data_identity_sha256 != config.data_identity_sha256
        or identity.access_policy_sha256 != config.access_policy_sha256
    ):
        raise ValueError("reopened store differs from resolved config authority")
    return store


def _matching_from_store(
    *,
    config: H6PredictionV3ResolvedConfig,
    store: BlindedCorpusStore,
) -> H6MatchingSetV3:
    identity = store.data_identity
    train_tokens = identity.train_tokens
    matching_set = build_h6_matching_set_v3(
        git_head=config.source.git_head,
        dirty_digest=config.source.dirty_digest,
        train_token_count=train_tokens.token_count,
        train_token_sha256=train_tokens.encoded_token_sha256,
        vocabulary=ByteTokenizerV1().vocabulary_identity,
        horizon=32,
    )
    if matching_set.matching_set_sha256 != config.matching_set_sha256:
        raise ValueError("regenerated matching set differs from resolved config")
    return matching_set


def _reopen_authorities(
    *,
    config: H6PredictionV3ResolvedConfig,
    paths: H6OperationPathsV3,
) -> H6PredictionV3Authorities:
    return read_h6_prediction_v3_authorities(
        paths.authorities_directory,
        expected_config_sha256=config.config_sha256,
        expected_matching_set_sha256=config.matching_set_sha256,
    )


def _publish_and_reopen_authorities(
    *,
    config: H6PredictionV3ResolvedConfig,
    paths: H6OperationPathsV3,
    matching_set: H6MatchingSetV3,
    readiness: object,
    plan: object,
) -> H6PredictionV3Authorities:
    try:
        published = publish_h6_prediction_v3_authorities(
            run_root=paths.authorities_run_root,
            run_name=paths.authorities_run_name,
            config=config,
            matching_set=matching_set,
            readiness=readiness,  # type: ignore[arg-type]
            plan=plan,  # type: ignore[arg-type]
        )
    except ArtifactPublicationError:
        if not os.path.lexists(paths.authorities_directory):
            raise
        published = paths.authorities_directory
    reopened = read_h6_prediction_v3_authorities(
        published,
        expected_config_sha256=config.config_sha256,
        expected_matching_set_sha256=matching_set.matching_set_sha256,
        expected_readiness_sha256=getattr(readiness, "readiness_sha256", None),
        expected_plan_sha256=getattr(plan, "plan_sha256", None),
    )
    if (
        reopened.config != config
        or reopened.matching_set != matching_set
        or reopened.readiness != readiness
        or reopened.plan != plan
    ):
        raise ArtifactPublicationError("published authority chain changed")
    return reopened


def run_h6_experiment_v3(
    *,
    operation: H6V3Operation,
    config: H6PredictionV3ResolvedConfig,
    runtime: object | None,
    operation_config: Mapping[str, object],
    authorization_sha256: str,
) -> object:
    """Run one authorized path-only v3 operation."""

    if type(config) is not H6PredictionV3ResolvedConfig:
        raise ValueError("orchestration requires an exact resolved v3 config")
    if operation not in _OPERATION_AUTHORIZATION_SHA256:
        raise ValueError("unknown H6-Prediction v3 operation")
    if authorization_sha256 != _OPERATION_AUTHORIZATION_SHA256[operation]:
        raise PermissionError("operation authorization digest is not exact")
    paths = H6OperationPathsV3.from_mapping(operation_config)

    if operation == "prediction_readiness":
        if runtime is not None:
            raise ValueError("readiness cannot receive a configured CUDA runtime")
        prerequisite_evidence = reopen_h6_prediction_v3_prerequisite_evidence(
            config=config,
            correctness_artifact_roots=paths.correctness_artifact_roots,
            h1_prefix_prior_artifact_root=(
                paths.h1_prefix_prior_artifact_root
            ),
            smc_accuracy_artifact_root=paths.smc_accuracy_artifact_root,
            h6_prefix_artifact_root=paths.h6_prefix_artifact_root,
            h6_prefix_manifest_sha256=paths.h6_prefix_manifest_sha256,
            h6_prefix_junit_sha256=paths.h6_prefix_junit_sha256,
        )
        store = _reopen_store_for_config(config=config, paths=paths)
        matching_set = _matching_from_store(config=config, store=store)
        readiness = validate_h6_prediction_readiness_v3(
            config=config,
            matching_set=matching_set,
            git_head=config.source.git_head,
            dirty_digest=config.source.dirty_digest,
            prerequisite_evidence=prerequisite_evidence,
        )
        plan = plan_h6_experiment_v3(
            readiness=readiness,
            matching_set=matching_set,
            training_schedule=config.training_schedule,
            runtime_identity=config.runtime,
        )
        return _publish_and_reopen_authorities(
            config=config,
            paths=paths,
            matching_set=matching_set,
            readiness=readiness,
            plan=plan,
        )

    authorities = _reopen_authorities(config=config, paths=paths)
    (
        prefix_certificate_set,
        direct_certificate,
    ) = read_h6_prefix_authorities_for_scoring_v3(
        paths.h6_prefix_artifact_root,
        expected_manifest_sha256=paths.h6_prefix_manifest_sha256,
        expected_junit_sha256=paths.h6_prefix_junit_sha256,
        readiness=authorities.readiness,
    )
    if (
        prefix_certificate_set.source_sha256
        != config.source.source_sha256
        or direct_certificate.source_sha256
        != config.source.source_sha256
    ):
        raise ValueError(
            "reopened Prefix authorities differ from resolved config"
        )
    store = _reopen_store_for_config(config=config, paths=paths)
    if store.data_identity_sha256 != authorities.readiness.data_identity_sha256:
        raise ValueError("reopened store and authority data identities differ")
    if operation == "plan":
        if runtime is not None:
            raise ValueError("planning cannot receive a configured CUDA runtime")
        return authorities.plan

    if operation == "train":
        from vfe4.training.h6_checkpoint_catalog_v3 import (
            publish_h6_checkpoint_catalog_entry_v3,
            read_h6_checkpoint_catalog_v3,
        )
        from vfe4.training.h6_training_attempt_v3 import (
            run_h6_training_attempt_v3,
        )
        from vfe4.training.h6_validation_campaign_v3 import (
            h6_tuning_selection_directory_v3,
            read_h6_tuning_selection_v3,
        )

        result = run_h6_training_attempt_v3(
            authorities=authorities,
            store=store,
            runtime=runtime,
            planned_attempt_sha256=paths.planned_attempt_sha256,
            checkpoint_path=paths.checkpoint_path,
            maximum_checkpoint_bytes=paths.maximum_checkpoint_bytes,
            validation_bundle_directory=paths.validation_bundle_directory,
        )
        expected_entry_directory = (
            paths.checkpoint_catalog_root / result.planned_attempt_sha256
        )
        try:
            publish_h6_checkpoint_catalog_entry_v3(
                catalog_root=paths.checkpoint_catalog_root,
                checkpoint_path=result.checkpoint_path,
                maximum_checkpoint_bytes=paths.maximum_checkpoint_bytes,
                executable_attempt=result.executable_attempt,
                checkpoint=result.terminal_checkpoint,
            )
        except ArtifactPublicationError:
            if not os.path.lexists(expected_entry_directory):
                raise
        tuning_selection = None
        if result.stage == "confirmatory":
            tuning_selection = read_h6_tuning_selection_v3(
                h6_tuning_selection_directory_v3(
                    paths.validation_bundle_directory
                ),
                expected_plan_sha256=authorities.plan.plan_sha256,
                expected_experiment_config_sha256=(
                    authorities.config.config_sha256
                ),
            )
        catalog = read_h6_checkpoint_catalog_v3(
            paths.checkpoint_catalog_root,
            authorities=authorities,
            maximum_checkpoint_bytes=paths.maximum_checkpoint_bytes,
            tuning_selection=tuning_selection,
            required_inventory="partial",
        )
        matching_items = tuple(
            item
            for item in catalog.items
            if item.entry.planned_attempt_sha256
            == result.planned_attempt_sha256
        )
        if len(matching_items) != 1:
            raise ArtifactPublicationError(
                "terminal checkpoint is absent from its exact catalog"
            )
        reopened_item = matching_items[0]
        if (
            reopened_item.executable_attempt.executable_attempt_sha256
            != result.executable_attempt.executable_attempt_sha256
            or reopened_item.checkpoint.checkpoint_sha256
            != result.terminal_checkpoint.checkpoint_sha256
            or reopened_item.checkpoint.to_bytes()
            != result.terminal_checkpoint.to_bytes()
        ):
            raise ArtifactPublicationError(
                "terminal checkpoint catalog entry changed after publication"
            )
        return result

    from vfe4.training.h6_validation_campaign_v3 import (
        run_h6_validation_campaign_v3,
    )

    return run_h6_validation_campaign_v3(
        authorities=authorities,
        store=store,
        checkpoint_catalog_root=paths.checkpoint_catalog_root,
        maximum_checkpoint_bytes=paths.maximum_checkpoint_bytes,
        validation_bundle_directory=paths.validation_bundle_directory,
    )


__all__ = [
    "H6OperationPathsV3",
    "run_h6_experiment_v3",
]
