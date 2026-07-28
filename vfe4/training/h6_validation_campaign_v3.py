"""Two-stage CPU validation campaign and durable H6 v3 selections."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from vfe4.artifacts.atomic import (
    ArtifactPublicationError,
    canonical_json_bytes,
    publish_run_directory,
)
from vfe4.artifacts.h6_prediction_v3 import (
    H6PredictionV3Authorities,
    H6TuningSelectionV3,
    H6ValidationBundleV3,
    H6ValidationRecordV3,
    _tuning_selection_from_payload,
    bind_h6_checkpoint_selection_v3,
    publish_h6_validation_bundle_v3,
    read_h6_validation_bundle_v3,
    select_h6_tuning_v3,
)
from vfe4.data.access import issue_h6_validation_capability_v3
from vfe4.training.h6_checkpoint_catalog_v3 import (
    _is_redirect,
    _read_bounded_regular_file_once,
    read_h6_checkpoint_catalog_v3,
)
from vfe4.training.h6_validation_v3 import (
    score_h6_validation_checkpoint_v3,
)


_SELECTION_FILENAME = "tuning_selection.json"
_MAXIMUM_SELECTION_BYTES = 4 * 1024 * 1024
_LOWER_HEX = frozenset("0123456789abcdef")


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _canonical_directory(path: object, name: str) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError(f"{name} must be an absolute pathlib Path")
    resolved = path.resolve(strict=False)
    if resolved.as_posix() != path.as_posix():
        raise ValueError(f"{name} must be canonical")
    return resolved


def h6_tuning_selection_directory_v3(
    validation_bundle_directory: Path,
) -> Path:
    """Derive the sole tuning-selection directory from the final bundle path."""

    final = _canonical_directory(
        validation_bundle_directory,
        "validation_bundle_directory",
    )
    return final.with_name(f"{final.name}-TUNING-SELECTION")


def publish_h6_tuning_selection_v3(
    *,
    run_root: Path,
    run_name: str,
    selection: H6TuningSelectionV3,
) -> Path:
    """Publish all 72 tuning scores and their frozen cell selections."""

    if type(selection) is not H6TuningSelectionV3:
        raise ArtifactPublicationError("an exact tuning selection v3 is required")
    selection.__post_init__()
    return publish_run_directory(
        run_root,
        run_name,
        {
            _SELECTION_FILENAME: selection.canonical_payload()
            | {"tuning_selection_sha256": selection.tuning_selection_sha256}
        },
    )


def read_h6_tuning_selection_v3(
    run_directory: Path,
    *,
    expected_plan_sha256: str,
    expected_experiment_config_sha256: str,
    expected_tuning_selection_sha256: str | None = None,
) -> H6TuningSelectionV3:
    """Authenticate and reconstruct one standalone tuning selection."""

    directory = _canonical_directory(run_directory, "run_directory")
    _require_sha256(expected_plan_sha256, "expected_plan_sha256")
    _require_sha256(
        expected_experiment_config_sha256,
        "expected_experiment_config_sha256",
    )
    if expected_tuning_selection_sha256 is not None:
        _require_sha256(
            expected_tuning_selection_sha256,
            "expected_tuning_selection_sha256",
        )
    try:
        root_status = directory.lstat()
        children = tuple(directory.iterdir())
    except OSError as exc:
        raise ArtifactPublicationError(
            "tuning selection directory is unavailable"
        ) from exc
    if (
        not stat.S_ISDIR(root_status.st_mode)
        or _is_redirect(directory, root_status)
        or {path.name for path in children}
        != {_SELECTION_FILENAME, "manifest.sha256"}
    ):
        raise ArtifactPublicationError(
            "tuning selection directory inventory is not exact"
        )
    payload_raw = _read_bounded_regular_file_once(
        directory / _SELECTION_FILENAME,
        maximum_bytes=_MAXIMUM_SELECTION_BYTES,
        label="tuning selection",
    )
    manifest_raw = _read_bounded_regular_file_once(
        directory / "manifest.sha256",
        maximum_bytes=256,
        label="tuning selection manifest",
    )
    expected_manifest = (
        f"{hashlib.sha256(payload_raw).hexdigest()}  {_SELECTION_FILENAME}\n"
    ).encode("ascii")
    if manifest_raw != expected_manifest:
        raise ArtifactPublicationError("tuning selection manifest changed")

    def reject_duplicates(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("tuning selection has duplicate JSON keys")
            result[key] = value
        return result

    try:
        payload = json.loads(
            payload_raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ArtifactPublicationError(
            "tuning selection is not canonical JSON"
        ) from exc
    if canonical_json_bytes(payload) != payload_raw:
        raise ArtifactPublicationError("tuning selection JSON is not canonical")
    selection = _tuning_selection_from_payload(payload)
    if (
        selection.plan_sha256 != expected_plan_sha256
        or selection.experiment_config_sha256
        != expected_experiment_config_sha256
        or (
            expected_tuning_selection_sha256 is not None
            and selection.tuning_selection_sha256
            != expected_tuning_selection_sha256
        )
    ):
        raise ArtifactPublicationError("tuning selection authority drift")
    return selection


@dataclass(frozen=True, slots=True)
class H6ValidationCampaignResultV3:
    state: Literal["TUNING_SELECTED", "COMPLETE"]
    tuning_selection: H6TuningSelectionV3
    validation_bundle: H6ValidationBundleV3 | None
    published_directory: Path

    def __post_init__(self) -> None:
        if type(self.tuning_selection) is not H6TuningSelectionV3:
            raise ValueError("validation campaign requires exact tuning selection")
        self.tuning_selection.__post_init__()
        directory = _canonical_directory(
            self.published_directory,
            "published_directory",
        )
        if directory != self.published_directory:
            raise ValueError("published validation directory is not canonical")
        if self.state == "TUNING_SELECTED":
            if self.validation_bundle is not None:
                raise ValueError("tuning-only campaign cannot carry a final bundle")
        elif self.state == "COMPLETE":
            if type(self.validation_bundle) is not H6ValidationBundleV3:
                raise ValueError("complete campaign requires exact validation bundle")
            self.validation_bundle.__post_init__()
            if (
                self.validation_bundle.tuning_selection
                != self.tuning_selection
            ):
                raise ValueError("complete bundle tuning selection changed")
        else:
            raise ValueError("validation campaign state is invalid")


def _publish_and_reopen_selection(
    *,
    selection_directory: Path,
    selection: H6TuningSelectionV3,
) -> H6TuningSelectionV3:
    try:
        published = publish_h6_tuning_selection_v3(
            run_root=selection_directory.parent,
            run_name=selection_directory.name,
            selection=selection,
        )
    except ArtifactPublicationError:
        if not os.path.lexists(selection_directory):
            raise
        published = selection_directory
    reopened = read_h6_tuning_selection_v3(
        published,
        expected_plan_sha256=selection.plan_sha256,
        expected_experiment_config_sha256=(
            selection.experiment_config_sha256
        ),
        expected_tuning_selection_sha256=(
            selection.tuning_selection_sha256
        ),
    )
    if reopened != selection:
        raise ArtifactPublicationError("published tuning selection changed")
    return reopened


def _score_tuning_catalog(
    *,
    authorities: H6PredictionV3Authorities,
    store: object,
    catalog: object,
) -> H6TuningSelectionV3:
    items = tuple(getattr(catalog, "tuning_items", ()))
    expected = tuple(
        attempt.planned_attempt_sha256
        for attempt in authorities.plan.tuning_attempts
    )
    observed = tuple(
        item.executable_attempt.planned_attempt.planned_attempt_sha256
        for item in items
    )
    if observed != expected:
        raise ArtifactPublicationError(
            "validation campaign tuning checkpoint inventory is incomplete"
        )
    records: list[H6ValidationRecordV3] = []
    capability = issue_h6_validation_capability_v3(
        store,  # type: ignore[arg-type]
        authorities.readiness,
        authorities.plan,
    )
    for item in items:
        attempt = item.executable_attempt.planned_attempt
        record = score_h6_validation_checkpoint_v3(
            capability=capability,
            checkpoint=item.checkpoint,
            planned_attempt=attempt,
            plan=authorities.plan,
        )
        if (
            type(record) is not H6ValidationRecordV3
            or record.checkpoint_sha256
            != item.checkpoint.checkpoint_sha256
            or record.attempt_spec_sha256
            != attempt.attempt_spec.attempt_spec_sha256
        ):
            raise ArtifactPublicationError(
                "validation scorer changed its checkpoint authority"
            )
        records.append(record)
    return select_h6_tuning_v3(tuple(records), authorities.plan)


def run_h6_validation_campaign_v3(
    *,
    authorities: H6PredictionV3Authorities,
    store: object,
    checkpoint_catalog_root: Path,
    maximum_checkpoint_bytes: int,
    validation_bundle_directory: Path,
) -> H6ValidationCampaignResultV3:
    """Advance validation from 72 tuning scores to the exact 96-checkpoint bundle."""

    if type(authorities) is not H6PredictionV3Authorities:
        raise ValueError("validation campaign requires exact v3 authorities")
    authorities.__post_init__()
    catalog_root = _canonical_directory(
        checkpoint_catalog_root,
        "checkpoint_catalog_root",
    )
    final_directory = _canonical_directory(
        validation_bundle_directory,
        "validation_bundle_directory",
    )
    selection_directory = h6_tuning_selection_directory_v3(final_directory)

    if not os.path.lexists(selection_directory):
        tuning_catalog = read_h6_checkpoint_catalog_v3(
            catalog_root,
            authorities=authorities,
            maximum_checkpoint_bytes=maximum_checkpoint_bytes,
            required_inventory="tuning",
        )
        selection = _score_tuning_catalog(
            authorities=authorities,
            store=store,
            catalog=tuning_catalog,
        )
        reopened = _publish_and_reopen_selection(
            selection_directory=selection_directory,
            selection=selection,
        )
        return H6ValidationCampaignResultV3(
            state="TUNING_SELECTED",
            tuning_selection=reopened,
            validation_bundle=None,
            published_directory=selection_directory,
        )

    selection = read_h6_tuning_selection_v3(
        selection_directory,
        expected_plan_sha256=authorities.plan.plan_sha256,
        expected_experiment_config_sha256=authorities.config.config_sha256,
    )
    catalog = read_h6_checkpoint_catalog_v3(
        catalog_root,
        authorities=authorities,
        maximum_checkpoint_bytes=maximum_checkpoint_bytes,
        tuning_selection=selection,
        required_inventory="partial",
    )
    expected_tuning = tuple(
        attempt.planned_attempt_sha256
        for attempt in authorities.plan.tuning_attempts
    )
    observed_tuning = tuple(
        item.entry.planned_attempt_sha256 for item in catalog.tuning_items
    )
    if observed_tuning != expected_tuning:
        raise ArtifactPublicationError(
            "persisted tuning selection lost its checkpoint inventory"
        )
    expected_confirmatory = tuple(
        attempt.planned_attempt_sha256
        for attempt in authorities.plan.confirmatory_attempts
    )
    observed_confirmatory = tuple(
        item.entry.planned_attempt_sha256
        for item in catalog.confirmatory_items
    )
    if observed_confirmatory != expected_confirmatory:
        observed_set = set(observed_confirmatory)
        if (
            len(observed_confirmatory) < len(expected_confirmatory)
            and len(observed_set) == len(observed_confirmatory)
            and observed_confirmatory
            == tuple(
                digest
                for digest in expected_confirmatory
                if digest in observed_set
            )
        ):
            return H6ValidationCampaignResultV3(
                state="TUNING_SELECTED",
                tuning_selection=selection,
                validation_bundle=None,
                published_directory=selection_directory,
            )
        raise ArtifactPublicationError(
            "confirmatory checkpoint inventory is foreign or unordered"
        )

    checkpoint_selection = bind_h6_checkpoint_selection_v3(
        tuple(
            (
                item.executable_attempt.planned_attempt,
                item.checkpoint,
            )
            for item in catalog.confirmatory_items
        ),
        authorities.plan,
        selection,
    )
    bundle = H6ValidationBundleV3.create(
        plan=authorities.plan,
        tuning_selection=selection,
        checkpoint_selection=checkpoint_selection,
    )
    if os.path.lexists(final_directory):
        published = final_directory
    else:
        published = publish_h6_validation_bundle_v3(
            run_root=final_directory.parent,
            run_name=final_directory.name,
            bundle=bundle,
        )
    reopened_bundle = read_h6_validation_bundle_v3(
        published,
        expected_plan_sha256=authorities.plan.plan_sha256,
        expected_experiment_config_sha256=authorities.config.config_sha256,
        expected_validation_bundle_sha256=bundle.validation_bundle_sha256,
    )
    if reopened_bundle != bundle:
        raise ArtifactPublicationError("published validation bundle changed")
    return H6ValidationCampaignResultV3(
        state="COMPLETE",
        tuning_selection=selection,
        validation_bundle=reopened_bundle,
        published_directory=final_directory,
    )


__all__ = [
    "H6ValidationCampaignResultV3",
    "h6_tuning_selection_directory_v3",
    "publish_h6_tuning_selection_v3",
    "read_h6_tuning_selection_v3",
    "run_h6_validation_campaign_v3",
]
