"""Fail-closed loading of finalized metric and result-table figure inputs."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal

from vfe4.artifacts.manifest import ArtifactIntegrityRecord
from vfe4.recording.metrics import (
    _read_regular_bytes,
    WT103_METRIC_SEMANTIC_BY_NAME,
    WT103_METRIC_UNIT_BY_NAME,
    metrics_csv_bytes,
    validate_required_metric_families,
)
from vfe4.recording.tables import load_finalized_metric_rows
from vfe4.types.figures import (
    FigureExperimentIndexIdentity,
    FigureInputIdentity,
    FigureResultRow,
    FigureSpec,
    figure_panel_applicability,
    figure_series_metric_names,
)
from vfe4.types.training import (
    EndpointInventory,
    MetricRecord,
    WT103ArmSpec,
    canonical_json_bytes,
    owned_sha256,
)

from .finalized_index import (
    ReadOnlyFigureIndexError,
    ValidatedRunManifest,
    validate_finalized_experiment_index,
)
from .spec import validate_figure_registry


class FigureInputError(ValueError):
    """Finalized figure inputs are incomplete, inconsistent, or mutable."""


def _sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FigureInputError(f"{name} must be a lowercase SHA-256")
    return value


def _text(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise FigureInputError(f"{name} must be nonempty text")
    return value


def _exact_keys(
    value: object,
    keys: tuple[str, ...],
    name: str,
) -> dict[str, object]:
    expected = set(keys)
    observed = set(value) if type(value) is dict else set()
    if (
        type(value) is not dict
        or observed != expected
    ):
        raise FigureInputError(
            f"{name} has an invalid key set; "
            f"missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )
    return value


def _reject_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise FigureInputError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _load_canonical_json(
    path: Path,
    *,
    expected_sha256: str,
) -> dict[str, object]:
    payload = _read_regular_bytes(path)
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise FigureInputError(f"artifact SHA-256 changed: {path}")
    try:
        document = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                FigureInputError(f"JSON contains nonfinite value {value}")
            ),
        )
    except FigureInputError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise FigureInputError(f"artifact JSON is invalid: {path}") from exc
    if type(document) is not dict or canonical_json_bytes(document) != payload:
        raise FigureInputError(f"artifact JSON is not canonical: {path}")
    return document


def _relative_path(value: object, name: str) -> str:
    text = _text(value, name)
    posix = PurePosixPath(text)
    windows = PureWindowsPath(text)
    if (
        "\\" in text
        or posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or bool(windows.root)
        or posix.as_posix() != text
        or any(part in ("", ".", "..") for part in posix.parts)
    ):
        raise FigureInputError(f"{name} must be a contained POSIX path")
    return text


def _contained_path(root: Path, relative: str) -> Path:
    candidate = root.joinpath(*PurePosixPath(relative).parts)
    try:
        if candidate.resolve(strict=True).relative_to(
            root.resolve(strict=True)
        ) is None:
            raise AssertionError
    except (OSError, ValueError) as exc:
        raise FigureInputError(
            f"figure input escapes its closed root: {relative}"
        ) from exc
    return candidate


@dataclasses.dataclass(frozen=True, slots=True)
class MetricSource:
    terminal_checkpoint_key: str
    run_id: str
    arm_id: str
    seed_id: int
    checkpoint_role: Literal["terminal_scoring"]
    run_manifest_path: str
    metrics_jsonl_path: str
    metrics_csv_path: str
    metrics_jsonl_sha256: str
    metrics_csv_sha256: str

    def __post_init__(self) -> None:
        for name in ("terminal_checkpoint_key", "run_id", "arm_id"):
            _text(getattr(self, name), name)
        if type(self.seed_id) is not int or self.seed_id < 0:
            raise FigureInputError("seed_id must be nonnegative")
        if self.checkpoint_role != "terminal_scoring":
            raise FigureInputError(
                "figures accept only terminal_scoring checkpoints"
            )
        for name in (
            "run_manifest_path",
            "metrics_jsonl_path",
            "metrics_csv_path",
        ):
            _relative_path(getattr(self, name), name)
        _sha256(self.metrics_jsonl_sha256, "metrics_jsonl_sha256")
        _sha256(self.metrics_csv_sha256, "metrics_csv_sha256")

    @classmethod
    def from_document(cls, value: object) -> "MetricSource":
        document = _exact_keys(
            value,
            tuple(field.name for field in dataclasses.fields(cls)),
            "metric_source",
        )
        try:
            return cls(**document)  # type: ignore[arg-type]
        except TypeError as exc:
            raise FigureInputError("metric source types are invalid") from exc


@dataclasses.dataclass(frozen=True, slots=True)
class FigureApplicabilityRow:
    panel_key: str
    arm_id: str
    applicability: Literal["applicable", "not_applicable"]
    reason: str

    def __post_init__(self) -> None:
        for name in ("panel_key", "arm_id", "reason"):
            _text(getattr(self, name), name)
        if self.applicability not in (
            "applicable",
            "not_applicable",
        ):
            raise FigureInputError("unknown figure applicability")

    @classmethod
    def from_document(cls, value: object) -> "FigureApplicabilityRow":
        document = _exact_keys(
            value,
            tuple(field.name for field in dataclasses.fields(cls)),
            "figure_applicability",
        )
        try:
            return cls(**document)  # type: ignore[arg-type]
        except TypeError as exc:
            raise FigureInputError(
                "figure applicability types are invalid"
            ) from exc


@dataclasses.dataclass(frozen=True, slots=True)
class FigurePoint:
    series_key: str
    arm_id: str
    seed_id: int
    source_record_sha256: str
    source_phase: str
    source_split: Literal[
        "train",
        "validation",
        "test",
        "not_applicable",
    ]
    source_step: int
    source_pass_index: int
    counted_targets: int
    metric_name: str
    applicability: Literal["applicable"]
    applicability_reason: str
    numerator: float | None
    denominator: int | None
    value: float
    lower: float | None
    upper: float | None
    units: str
    result_role: str

    def __post_init__(self) -> None:
        for name in (
            "series_key",
            "arm_id",
            "source_phase",
            "source_split",
            "metric_name",
            "applicability_reason",
            "units",
            "result_role",
        ):
            _text(getattr(self, name), name)
        _sha256(self.source_record_sha256, "source_record_sha256")
        if (
            type(self.seed_id) is not int
            or self.seed_id < 0
            or type(self.source_step) is not int
            or self.source_step < 0
            or type(self.source_pass_index) is not int
            or self.source_pass_index < 0
            or type(self.counted_targets) is not int
            or self.counted_targets < 0
            or self.applicability != "applicable"
            or type(self.value) is not float
            or not math.isfinite(self.value)
        ):
            raise FigureInputError("figure point scalar fields are invalid")
        semantic = WT103_METRIC_SEMANTIC_BY_NAME.get(self.metric_name)
        if semantic is None:
            raise FigureInputError("figure point metric semantic is unknown")
        if semantic == "scalar":
            if self.numerator is not None or self.denominator is not None:
                raise FigureInputError(
                    "scalar figure points forbid raw numerator/denominator"
                )
        elif (
            type(self.numerator) is not float
            or not math.isfinite(self.numerator)
            or type(self.denominator) is not int
            or self.denominator <= 0
        ):
            raise FigureInputError(
                "derived figure points require valid raw "
                "numerator/denominator"
            )
        if (self.lower is None) != (self.upper is None):
            raise FigureInputError(
                "figure point interval bounds apply jointly"
            )
        if self.lower is not None and (
            type(self.lower) is not float
            or not math.isfinite(self.lower)
            or type(self.upper) is not float
            or not math.isfinite(self.upper)
            or self.lower > self.value
            or self.upper < self.value
        ):
            raise FigureInputError("figure point interval is invalid")

    @classmethod
    def from_document(cls, value: object) -> "FigurePoint":
        document = _exact_keys(
            value,
            tuple(field.name for field in dataclasses.fields(cls)),
            "figure_point",
        )
        try:
            return cls(**document)  # type: ignore[arg-type]
        except TypeError as exc:
            raise FigureInputError("figure point types are invalid") from exc


@dataclasses.dataclass(frozen=True, slots=True)
class FinalResultTable:
    schema_version: Literal["wt103-final-result-table-v2"]
    endpoint_inventory_sha256: str
    metrics_jsonl_sha256s: tuple[str, ...]
    aggregation_sha256: str
    result_rows: tuple[FigureResultRow, ...]
    figure_panel_keys: tuple[str, ...]
    figure_series_keys: tuple[str, ...]
    applicability_rows: tuple[FigureApplicabilityRow, ...]
    points: tuple[FigurePoint, ...]
    result_table_sha256: str

    @property
    def result_row_keys(self) -> tuple[str, ...]:
        """Return the ordered row keys without duplicating serialized state."""

        return tuple(row.result_row_key for row in self.result_rows)

    def __post_init__(self) -> None:
        if self.schema_version != "wt103-final-result-table-v2":
            raise FigureInputError("unsupported final result-table schema")
        _sha256(
            self.endpoint_inventory_sha256,
            "endpoint_inventory_sha256",
        )
        _sha256(self.aggregation_sha256, "aggregation_sha256")
        for index, digest in enumerate(self.metrics_jsonl_sha256s):
            _sha256(digest, f"metrics_jsonl_sha256s[{index}]")
        for name in (
            "figure_panel_keys",
            "figure_series_keys",
        ):
            values = getattr(self, name)
            if (
                type(values) is not tuple
                or not values
                or any(type(item) is not str or not item for item in values)
                or len(set(values)) != len(values)
            ):
                raise FigureInputError(f"{name} is invalid")
        if (
            type(self.result_rows) is not tuple
            or not self.result_rows
            or any(type(row) is not FigureResultRow for row in self.result_rows)
            or len(set(self.result_row_keys)) != len(self.result_rows)
            or
            type(self.applicability_rows) is not tuple
            or not self.applicability_rows
            or any(
                type(row) is not FigureApplicabilityRow
                for row in self.applicability_rows
            )
            or type(self.points) is not tuple
            or not self.points
            or any(type(point) is not FigurePoint for point in self.points)
        ):
            raise FigureInputError(
                "result table rows must be exact immutable records"
            )
        expected = owned_sha256(
            "vfe4.wt103.final-result-table.v2",
            {
                field.name: getattr(self, field.name)
                for field in dataclasses.fields(self)
                if field.name != "result_table_sha256"
            },
        )
        _sha256(self.result_table_sha256, "result_table_sha256")
        if self.result_table_sha256 != expected:
            raise FigureInputError(
                "result_table_sha256 does not match the table"
            )

    @classmethod
    def from_document(cls, value: object) -> "FinalResultTable":
        keys = tuple(field.name for field in dataclasses.fields(cls))
        document = _exact_keys(value, keys, "result_table")
        converted = dict(document)
        for name in (
            "metrics_jsonl_sha256s",
            "figure_panel_keys",
            "figure_series_keys",
        ):
            if type(converted[name]) is not list:
                raise FigureInputError(f"result_table.{name} must be a list")
            converted[name] = tuple(converted[name])  # type: ignore[arg-type]
        result_rows = converted["result_rows"]
        rows = converted["applicability_rows"]
        points = converted["points"]
        if (
            type(result_rows) is not list
            or type(rows) is not list
            or type(points) is not list
        ):
            raise FigureInputError(
                "result-table row collections must be lists"
            )
        converted["result_rows"] = tuple(
            FigureResultRow.from_document(row) for row in result_rows
        )
        converted["applicability_rows"] = tuple(
            FigureApplicabilityRow.from_document(row) for row in rows
        )
        converted["points"] = tuple(
            FigurePoint.from_document(point) for point in points
        )
        try:
            return cls(**converted)  # type: ignore[arg-type]
        except TypeError as exc:
            raise FigureInputError(
                "result-table types are invalid"
            ) from exc

    @classmethod
    def create(cls, **values: object) -> "FinalResultTable":
        payload = {
            "schema_version": "wt103-final-result-table-v2",
            **values,
        }
        return cls(
            **payload,
            result_table_sha256=owned_sha256(
                "vfe4.wt103.final-result-table.v2",
                payload,
            ),
        )  # type: ignore[arg-type]


@dataclasses.dataclass(frozen=True, slots=True)
class LoadedFigureInputs:
    experiment_index_identity: FigureExperimentIndexIdentity
    metric_sources: tuple[MetricSource, ...]
    metric_records: tuple[
        tuple[MetricSource, tuple[MetricRecord, ...]],
        ...,
    ]
    result_table: FinalResultTable
    specs: tuple[FigureSpec, ...]
    identity: FigureInputIdentity


def _artifact_record_from_document(
    value: object,
) -> ArtifactIntegrityRecord:
    document = _exact_keys(
        value,
        (
            "kind",
            "record_sha256",
            "relative_path",
            "schema_version",
            "sha256",
            "size_bytes",
        ),
        "artifact_record",
    )
    try:
        record = ArtifactIntegrityRecord(**document)  # type: ignore[arg-type]
        record.__post_init__()
    except (TypeError, ValueError) as exc:
        raise FigureInputError(
            "artifact record is invalid"
        ) from exc
    return record


def _manifest_source(
    *,
    root: Path,
    entry: object,
    manifests_by_path: dict[str, ValidatedRunManifest],
) -> MetricSource | None:
    document = _exact_keys(
        entry,
        (
            "disposition",
            "manifest_identity_sha256",
            "manifest_sha256",
            "relative_manifest_path",
            "run_id",
            "run_role",
        ),
        "experiment_index.run",
    )
    if (
        document["run_role"] != "confirmation"
        or document["disposition"] != "success"
    ):
        return None
    relative_manifest = _relative_path(
        document["relative_manifest_path"],
        "relative_manifest_path",
    )
    identity = manifests_by_path.get(relative_manifest)
    if identity is None:
        raise FigureInputError(
            "confirmation manifest lacks its read-only validated identity"
        )
    manifest = identity.document
    _exact_keys(
        manifest,
        (
            "artifact_records",
            "checkpoint_artifact_records",
            "checkpoints",
            "disposition",
            "ended_utc",
            "environment_sha256",
            "experiment_plan_sha256",
            "failure_record_sha256",
            "manifest_sha256",
            "monotonic_duration_seconds",
            "provenance_sha256",
            "reservation_sha256",
            "resume_count",
            "resume_lineage_sha256",
            "run_id",
            "run_role",
            "schema_version",
            "started_utc",
        ),
        "run_manifest",
    )
    if (
        manifest["run_id"] != document["run_id"]
        or manifest["run_role"] != "confirmation"
        or manifest["disposition"] != "success"
    ):
        raise FigureInputError(
            "confirmation manifest differs from its experiment-index entry"
        )
    checkpoints = manifest["checkpoints"]
    if type(checkpoints) is not list:
        raise FigureInputError("run_manifest.checkpoints must be a list")
    terminal = tuple(
        checkpoint
        for checkpoint in checkpoints
        if type(checkpoint) is dict
        and checkpoint.get("checkpoint_role") == "terminal_scoring"
    )
    if len(terminal) != 1:
        raise FigureInputError(
            "successful confirmation must expose one terminal checkpoint"
        )
    terminal_record = _exact_keys(
        terminal[0],
        (
            "artifact_sha256",
            "checkpoint_identity_sha256",
            "checkpoint_manifest_body_sha256",
            "checkpoint_payload_sha256",
            "checkpoint_role",
            "logical_key",
            "schema_version",
            "scientific_state_sha256",
            "size_bytes",
        ),
        "terminal_checkpoint",
    )
    logical_key = _text(
        terminal_record["logical_key"],
        "terminal_checkpoint.logical_key",
    )
    try:
        prefix, seed_text = logical_key.rsplit("/seed=", 1)
        arm_id = prefix.removeprefix("terminal/")
        if prefix == arm_id:
            raise ValueError
        seed_id = int(seed_text)
    except (ValueError, TypeError) as exc:
        raise FigureInputError(
            "terminal checkpoint logical key is malformed"
        ) from exc
    artifact_values = manifest["artifact_records"]
    if type(artifact_values) is not list:
        raise FigureInputError(
            "run_manifest.artifact_records must be a list"
        )
    records = tuple(
        _artifact_record_from_document(value) for value in artifact_values
    )
    by_path = {record.relative_path: record for record in records}
    if (
        len(by_path) != len(records)
        or set(by_path) != {"metrics.csv", "metrics.jsonl"}
        or any(record.kind != "file" for record in records)
    ):
        raise FigureInputError(
            "confirmation run must contain exactly finalized metrics JSONL/CSV"
        )
    run_id = _text(document["run_id"], "experiment_index.run_id")
    run_prefix = f"runs/{run_id}"
    return MetricSource(
        terminal_checkpoint_key=logical_key,
        run_id=run_id,
        arm_id=arm_id,
        seed_id=seed_id,
        checkpoint_role="terminal_scoring",
        run_manifest_path=relative_manifest,
        metrics_jsonl_path=f"{run_prefix}/metrics.jsonl",
        metrics_csv_path=f"{run_prefix}/metrics.csv",
        metrics_jsonl_sha256=by_path["metrics.jsonl"].sha256,
        metrics_csv_sha256=by_path["metrics.csv"].sha256,
    )


def _ordered_metric_sources(
    index_document: dict[str, object],
    *,
    root: Path,
    inventory: EndpointInventory,
    manifests_by_path: dict[str, ValidatedRunManifest],
) -> tuple[MetricSource, ...]:
    entries = index_document["runs"]
    if type(entries) is not list:
        raise FigureInputError("experiment_index.runs must be a list")
    by_key: dict[str, MetricSource] = {}
    for entry in entries:
        source = _manifest_source(
            root=root,
            entry=entry,
            manifests_by_path=manifests_by_path,
        )
        if source is None:
            continue
        if source.terminal_checkpoint_key in by_key:
            raise FigureInputError(
                "two successful confirmations claim one terminal key"
            )
        by_key[source.terminal_checkpoint_key] = source
    expected = set(inventory.terminal_checkpoint_keys)
    if set(by_key) != expected or len(by_key) != len(
        inventory.terminal_checkpoint_keys
    ):
        missing = sorted(expected - set(by_key))
        extra = sorted(set(by_key) - expected)
        raise FigureInputError(
            "final experiment index has an incomplete confirmation "
            f"inventory; missing={missing}, extra={extra}"
        )
    return tuple(by_key[key] for key in inventory.terminal_checkpoint_keys)


def _validate_metric_sources(
    sources: tuple[MetricSource, ...],
    inventory: EndpointInventory,
) -> dict[str, WT103ArmSpec]:
    if len(sources) != len(inventory.terminal_checkpoint_keys):
        raise FigureInputError(
            "final experiment index lacks the exact confirmation inventory"
        )
    arms = {arm.arm_id: arm for arm in inventory.arms}
    observed_paths: set[str] = set()
    for expected_key, source in zip(
        inventory.terminal_checkpoint_keys,
        sources,
    ):
        prefix, seed_text = expected_key.rsplit("/seed=", 1)
        expected_arm = prefix.removeprefix("terminal/")
        if (
            source.terminal_checkpoint_key != expected_key
            or source.arm_id != expected_arm
            or source.seed_id != int(seed_text)
            or source.arm_id not in arms
        ):
            raise FigureInputError(
                "metric source is not the ordered terminal checkpoint"
            )
        for path in (
            source.run_manifest_path,
            source.metrics_jsonl_path,
            source.metrics_csv_path,
        ):
            if path in observed_paths:
                raise FigureInputError("run-group artifact paths are reused")
            observed_paths.add(path)
    return arms


def _validate_result_table(
    table: FinalResultTable,
    *,
    inventory: EndpointInventory,
    sources: tuple[MetricSource, ...],
    arms: dict[str, WT103ArmSpec],
    metric_records: tuple[
        tuple[MetricSource, tuple[MetricRecord, ...]],
        ...,
    ],
) -> None:
    if (
        table.endpoint_inventory_sha256
        != inventory.endpoint_inventory_sha256
        or table.metrics_jsonl_sha256s
        != tuple(source.metrics_jsonl_sha256 for source in sources)
        or table.result_row_keys != inventory.result_row_keys
        or table.figure_panel_keys != inventory.figure_panel_keys
        or table.figure_series_keys != inventory.figure_series_keys
    ):
        raise FigureInputError(
            "result table differs from its frozen endpoint inputs"
        )
    for row, arm, result_row_key in zip(
        table.result_rows,
        inventory.arms,
        inventory.result_row_keys,
        strict=True,
    ):
        if arm.result_role in ("PRIMARY_REFERENCE", "PRIMARY_ENDPOINT"):
            applicability = "decision_bearing"
            reason = "a0_minus_parent_specific_complete_primary_pair"
        elif arm.result_role == "OBJECTIVE_GATE":
            applicability = "decision_bearing"
            reason = "complete_minus_emission_objective_gate"
        else:
            applicability = "descriptive_only"
            reason = "control_is_reported_but_cannot_rescue_or_reverse_primary"
        if (
            row.result_row_key != result_row_key
            or row.arm_id != arm.arm_id
            or row.scorer_kind != arm.scorer_kind
            or row.applicability != applicability
            or row.applicability_reason != reason
            or row.result_role != arm.result_role
        ):
            raise FigureInputError(
                "typed result row differs from the frozen arm inventory"
            )
    expected_applicability = tuple(
        (panel, arm.arm_id)
        for panel in inventory.figure_panel_keys
        for arm in inventory.arms
    )
    observed_applicability = tuple(
        (row.panel_key, row.arm_id) for row in table.applicability_rows
    )
    if observed_applicability != expected_applicability:
        raise FigureInputError(
            "result-table applicability grid is incomplete or reordered"
        )
    for row in table.applicability_rows:
        expected_status, expected_reason = figure_panel_applicability(
            row.panel_key,
            arms[row.arm_id],
        )
        if (
            row.applicability != expected_status
            or row.reason != expected_reason
        ):
            raise FigureInputError(
                "result-table applicability contradicts the frozen arm "
                "semantics"
            )
    series_order = {
        key: index for index, key in enumerate(inventory.figure_series_keys)
    }
    applicable_series = {
        key
        for key in inventory.figure_series_keys
        if figure_panel_applicability(
            key.split("/")[0],
            arms[key.split("/")[1]],
        )[0]
        == "applicable"
    }
    expected_roles = {
        arm.arm_id: arm.result_role for arm in inventory.arms
    }
    if (
        type(metric_records) is not tuple
        or len(metric_records) != len(sources)
    ):
        raise FigureInputError(
            "authoritative metric-record inventory is incomplete"
        )
    authoritative_records: dict[
        str,
        tuple[MetricSource, MetricRecord],
    ] = {}
    for expected_source, item in zip(
        sources,
        metric_records,
        strict=True,
    ):
        if (
            type(item) is not tuple
            or len(item) != 2
            or item[0] != expected_source
            or type(item[1]) is not tuple
            or not item[1]
        ):
            raise FigureInputError(
                "authoritative metric records differ from their source"
            )
        source, records = item
        for record in records:
            if type(record) is not MetricRecord:
                raise FigureInputError(
                    "authoritative metric record is not exact"
                )
            record.__post_init__()
            if any(
                value.name in WT103_METRIC_UNIT_BY_NAME
                and value.units
                != WT103_METRIC_UNIT_BY_NAME[value.name]
                for value in record.values
            ):
                raise FigureInputError(
                    "authoritative record violates frozen metric units"
                )
            if record.record_sha256 in authoritative_records:
                raise FigureInputError(
                    "authoritative metric record identity is reused"
                )
            authoritative_records[record.record_sha256] = (
                source,
                record,
            )
    observed_sort_keys: list[tuple[int, str, int, int]] = []
    seeds_by_series: dict[str, set[int]] = {
        key: set() for key in inventory.figure_series_keys
        if key in applicable_series
    }
    points_by_series_seed_metric: dict[
        tuple[str, int, str],
        list[int],
    ] = {}
    for point in table.points:
        expected_units = WT103_METRIC_UNIT_BY_NAME.get(
            point.metric_name
        )
        if expected_units is None or point.units != expected_units:
            raise FigureInputError(
                "result-table point violates frozen metric units"
            )
        authoritative = authoritative_records.get(
            point.source_record_sha256
        )
        if authoritative is None:
            raise FigureInputError(
                "result-table point lacks its authoritative metric record"
            )
        source, record = authoritative
        values = {value.name: value for value in record.values}
        metric = values.get(point.metric_name)
        counted_targets = values.get("counted_targets")
        if point.lower is not None or point.upper is not None:
            raise FigureInputError(
                "result-table point has an unauthoritative interval"
            )
        if (
            source.arm_id != point.arm_id
            or source.seed_id != point.seed_id
            or record.arm_id != point.arm_id
            or record.seed_id != point.seed_id
            or record.phase != point.source_phase
            or record.split != point.source_split
            or record.step != point.source_step
            or record.pass_index != point.source_pass_index
            or counted_targets is None
            or counted_targets.applicability != "applicable"
            or counted_targets.value != float(point.counted_targets)
            or metric is None
            or metric.applicability != point.applicability
            or metric.reason != point.applicability_reason
            or metric.numerator != point.numerator
            or metric.denominator != point.denominator
            or metric.value != point.value
            or metric.units != point.units
        ):
            raise FigureInputError(
                "result-table point differs from its authoritative metric"
            )
        if (
            point.series_key not in series_order
            or point.series_key not in applicable_series
            or not point.series_key.startswith(f"{point.series_key.split('/')[0]}/{point.arm_id}/")
            or point.arm_id not in arms
            or point.seed_id not in inventory.confirmatory_seed_ids
            or point.result_role != expected_roles[point.arm_id]
        ):
            raise FigureInputError(
                "result-table point is outside the frozen arm/series inventory"
            )
        seeds_by_series[point.series_key].add(point.seed_id)
        key = (point.series_key, point.seed_id, point.metric_name)
        counts = points_by_series_seed_metric.setdefault(key, [])
        counts.append(point.counted_targets)
        observed_sort_keys.append(
            (
                series_order[point.series_key],
                point.metric_name,
                point.seed_id,
                point.counted_targets,
            )
        )
    if observed_sort_keys != sorted(observed_sort_keys):
        raise FigureInputError("result-table points are not canonically ordered")
    if any(
        seeds != set(inventory.confirmatory_seed_ids)
        for seeds in seeds_by_series.values()
    ):
        raise FigureInputError(
            "every figure series requires all confirmatory seed traces"
        )
    if any(
        counts != sorted(set(counts))
        for counts in points_by_series_seed_metric.values()
    ):
        raise FigureInputError(
            "per-seed figure x coordinates must be unique and ascending"
        )
    for series_key in inventory.figure_series_keys:
        if series_key not in applicable_series:
            continue
        arm_id = series_key.split("/")[1]
        expected_metrics = {
            name
            for name in figure_series_metric_names(
                series_key,
                arms[arm_id],
            )
            if WT103_METRIC_SEMANTIC_BY_NAME[name]
            != "not_applicable_only"
        }
        reference_coordinates: tuple[int, ...] | None = None
        for seed_id in inventory.confirmatory_seed_ids:
            observed = {
                metric_name
                for observed_series, observed_seed, metric_name
                in points_by_series_seed_metric
                if observed_series == series_key
                and observed_seed == seed_id
            }
            if observed != expected_metrics:
                raise FigureInputError(
                    "result-table series lacks its exact semantic columns: "
                    f"{series_key}/seed={seed_id}"
                )
            for metric_name in sorted(expected_metrics):
                coordinates = tuple(
                    points_by_series_seed_metric[
                        (series_key, seed_id, metric_name)
                    ]
                )
                if reference_coordinates is None:
                    reference_coordinates = coordinates
                elif coordinates != reference_coordinates:
                    raise FigureInputError(
                        "series metrics/seeds use different x coordinates"
                    )


def load_figure_inputs(
    *,
    run_group_manifest_path: Path,
    inventory: EndpointInventory,
    specs: tuple[FigureSpec, ...],
) -> LoadedFigureInputs:
    """Load one final experiment index, its confirmations, and result table."""

    if not isinstance(run_group_manifest_path, Path):
        raise FigureInputError(
            "run_group_manifest_path must be one explicit pathlib.Path"
        )
    if type(inventory) is not EndpointInventory:
        raise FigureInputError("inventory must be exact")
    inventory.__post_init__()
    specs = validate_figure_registry(inventory=inventory, specs=specs)
    try:
        validated_index = validate_finalized_experiment_index(
            run_group_manifest_path,
            endpoint_inventory=inventory,
        )
    except ReadOnlyFigureIndexError as exc:
        raise FigureInputError(
            f"final experiment index is invalid: {exc}"
        ) from exc
    index_identity = validated_index.identity
    if index_identity.stage != "final":
        raise FigureInputError(
            "figures require the immutable final experiment index"
        )
    root = run_group_manifest_path.parent
    index_document = validated_index.document
    _exact_keys(
        index_document,
        (
            "artifact_records",
            "experiment_plan_sha256",
            "index_sha256",
            "runs",
            "schema_version",
            "stage",
        ),
        "experiment_index",
    )
    plan_document = validated_index.plan.document
    if (
        plan_document.get("endpoint_inventory_sha256")
        != inventory.endpoint_inventory_sha256
        or plan_document.get("experiment_plan_sha256")
        != index_identity.experiment_plan_sha256
    ):
        raise FigureInputError(
            "experiment plan differs from the figure endpoint inventory"
        )
    sources = _ordered_metric_sources(
        index_document,
        root=root,
        inventory=inventory,
        manifests_by_path=validated_index.manifest_by_relative_path(),
    )
    arms = _validate_metric_sources(sources, inventory)
    loaded_metrics: list[
        tuple[MetricSource, tuple[MetricRecord, ...]]
    ] = []
    csv_hashes: list[str] = []
    for source in sources:
        metric_path = _contained_path(root, source.metrics_jsonl_path)
        metric_payload = _read_regular_bytes(metric_path)
        if (
            hashlib.sha256(metric_payload).hexdigest()
            != source.metrics_jsonl_sha256
        ):
            raise FigureInputError(
                "metrics JSONL differs from its validated manifest record"
            )
        records = load_finalized_metric_rows(metric_path)
        if _read_regular_bytes(metric_path) != metric_payload:
            raise FigureInputError(
                "metrics JSONL changed during read-only figure loading"
            )
        if any(
            record.run_id != source.run_id
            or record.arm_id != source.arm_id
            or record.seed_id != source.seed_id
            for record in records
        ):
            raise FigureInputError(
                "metric log contains another run, arm, or seed"
            )
        validate_required_metric_families(
            records,
            arm_spec=arms[source.arm_id],
        )
        regenerated = metrics_csv_bytes(records)
        published = _read_regular_bytes(
            _contained_path(root, source.metrics_csv_path)
        )
        if (
            regenerated != published
            or hashlib.sha256(published).hexdigest()
            != source.metrics_csv_sha256
        ):
            raise FigureInputError(
                "published metrics.csv differs from regenerated JSONL"
            )
        loaded_metrics.append((source, records))
        csv_hashes.append(source.metrics_csv_sha256)
    artifact_values = index_document["artifact_records"]
    if type(artifact_values) is not list:
        raise FigureInputError(
            "experiment_index.artifact_records must be a list"
        )
    group_records = tuple(
        _artifact_record_from_document(value) for value in artifact_values
    )
    if (
        len(group_records) != 1
        or group_records[0].kind != "file"
        or group_records[0].relative_path != "result-table.json"
    ):
        raise FigureInputError(
            "final experiment index must bind exactly one result-table.json"
        )
    result_record = group_records[0]
    table = FinalResultTable.from_document(
        _load_canonical_json(
            _contained_path(root, result_record.relative_path),
            expected_sha256=result_record.sha256,
        )
    )
    _validate_result_table(
        table,
        inventory=inventory,
        sources=sources,
        arms=arms,
        metric_records=tuple(loaded_metrics),
    )
    identity = FigureInputIdentity.create(
        endpoint_inventory_sha256=inventory.endpoint_inventory_sha256,
        run_group_manifest_sha256=index_identity.payload_sha256,
        metrics_jsonl_sha256s=tuple(
            source.metrics_jsonl_sha256 for source in sources
        ),
        result_table_sha256=result_record.sha256,
        regenerated_csv_sha256s=tuple(csv_hashes),
    )
    return LoadedFigureInputs(
        experiment_index_identity=index_identity,
        metric_sources=sources,
        metric_records=tuple(loaded_metrics),
        result_table=table,
        specs=specs,
        identity=identity,
    )


__all__ = [
    "FigureApplicabilityRow",
    "FigureInputError",
    "FigurePoint",
    "FinalResultTable",
    "LoadedFigureInputs",
    "MetricSource",
    "load_figure_inputs",
]
