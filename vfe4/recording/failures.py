"""Independent canonical failure ledger for WT103 run-terminal events."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Literal

from vfe4.types.training import (
    WT103_ARM_IDS,
    canonical_json_bytes,
    owned_sha256,
)

from .metrics import (
    MetricDurabilityBackend,
    MetricLogError,
    _append_canonical_line,
    _read_regular_bytes,
    _reject_duplicate_keys,
    _split_jsonl_payload,
    _valid_sha256,
)


_ZERO_SHA256 = "0" * 64
_FAILURE_KEYS = frozenset(
    {
        "schema_version",
        "ordinal",
        "utc_timestamp",
        "monotonic_ns",
        "run_id",
        "arm_id",
        "seed_id",
        "phase",
        "step",
        "pass_index",
        "cursor_sha256",
        "checkpoint_identity_sha256",
        "retry_classification",
        "scientific_state_advanced",
        "terminal_disposition",
        "exception_type",
        "message",
        "message_sha256",
        "terminal",
        "previous_record_sha256",
        "record_sha256",
    }
)


class FailureLogError(ValueError):
    """The independent failure ledger is malformed or discontinuous."""


@dataclasses.dataclass(frozen=True, slots=True)
class FailureRecord:
    schema_version: Literal["wt103-failure-record-v1"]
    ordinal: int
    utc_timestamp: str
    monotonic_ns: int
    run_id: str
    arm_id: str
    seed_id: int
    phase: str
    step: int
    pass_index: int
    cursor_sha256: str | None
    checkpoint_identity_sha256: str | None
    retry_classification: Literal[
        "not_retryable",
        "infrastructure_retry_permitted_exact_restoration",
    ]
    scientific_state_advanced: bool
    terminal_disposition: Literal["failed", "inconclusive"]
    exception_type: str
    message: str
    message_sha256: str
    terminal: Literal[True]
    previous_record_sha256: str
    record_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != "wt103-failure-record-v1":
            raise FailureLogError("unsupported failure schema")
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise FailureLogError("failure ordinal must be nonnegative")
        try:
            parsed = datetime.fromisoformat(
                self.utc_timestamp.removesuffix("Z") + "+00:00"
            )
        except (AttributeError, ValueError) as exc:
            raise FailureLogError(
                "failure utc_timestamp is not canonical"
            ) from exc
        if (
            not self.utc_timestamp.endswith("Z")
            or parsed.tzinfo != timezone.utc
            or parsed.isoformat(
                timespec="microseconds"
            ).replace("+00:00", "Z")
            != self.utc_timestamp
        ):
            raise FailureLogError(
                "failure utc_timestamp is not canonical"
            )
        if type(self.monotonic_ns) is not int or self.monotonic_ns < 0:
            raise FailureLogError("failure monotonic_ns must be nonnegative")
        for name in ("run_id", "phase", "exception_type", "message"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise FailureLogError(f"{name} must be nonempty text")
        if self.arm_id not in WT103_ARM_IDS:
            raise FailureLogError("failure arm_id is not frozen")
        for name in ("seed_id", "step", "pass_index"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise FailureLogError(f"{name} must be nonnegative")
        for name in ("cursor_sha256", "checkpoint_identity_sha256"):
            value = getattr(self, name)
            if value is not None and not _valid_sha256(value):
                raise FailureLogError(f"{name} is invalid")
        if self.retry_classification not in (
            "not_retryable",
            "infrastructure_retry_permitted_exact_restoration",
        ):
            raise FailureLogError("failure retry classification is invalid")
        if type(self.scientific_state_advanced) is not bool:
            raise FailureLogError(
                "scientific_state_advanced must be exact bool"
            )
        if self.terminal_disposition not in ("failed", "inconclusive"):
            raise FailureLogError(
                "terminal_disposition must be failed or inconclusive"
            )
        if (
            self.retry_classification
            == "infrastructure_retry_permitted_exact_restoration"
            and self.scientific_state_advanced
        ):
            raise FailureLogError(
                "retry permission requires proof of no state advancement"
            )
        if self.terminal is not True:
            raise FailureLogError("failure records are terminal")
        if not _valid_sha256(self.message_sha256) or self.message_sha256 != (
            hashlib.sha256(self.message.encode("utf-8")).hexdigest()
        ):
            raise FailureLogError("message_sha256 does not match message")
        if not _valid_sha256(self.previous_record_sha256):
            raise FailureLogError("previous_record_sha256 is invalid")
        payload = {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name != "record_sha256"
        }
        expected = owned_sha256(
            "vfe4.wt103.failure-record.v1",
            payload,
        )
        if self.record_sha256 != expected:
            raise FailureLogError("failure record hash is inconsistent")


def create_failure_record(
    *,
    ordinal: int,
    utc_timestamp: str,
    monotonic_ns: int,
    run_id: str,
    arm_id: str,
    seed_id: int,
    phase: str,
    step: int,
    pass_index: int,
    cursor_sha256: str | None,
    checkpoint_identity_sha256: str | None,
    retry_classification: Literal[
        "not_retryable",
        "infrastructure_retry_permitted_exact_restoration",
    ],
    scientific_state_advanced: bool,
    terminal_disposition: Literal["failed", "inconclusive"],
    exception: BaseException,
    previous_record_sha256: str,
) -> FailureRecord:
    if not isinstance(exception, BaseException):
        raise FailureLogError("exception must be a BaseException")
    message = str(exception)
    if not message:
        message = type(exception).__name__
    payload = {
        "schema_version": "wt103-failure-record-v1",
        "ordinal": ordinal,
        "utc_timestamp": utc_timestamp,
        "monotonic_ns": monotonic_ns,
        "run_id": run_id,
        "arm_id": arm_id,
        "seed_id": seed_id,
        "phase": phase,
        "step": step,
        "pass_index": pass_index,
        "cursor_sha256": cursor_sha256,
        "checkpoint_identity_sha256": checkpoint_identity_sha256,
        "retry_classification": retry_classification,
        "scientific_state_advanced": scientific_state_advanced,
        "terminal_disposition": terminal_disposition,
        "exception_type": type(exception).__name__,
        "message": message,
        "message_sha256": hashlib.sha256(message.encode("utf-8")).hexdigest(),
        "terminal": True,
        "previous_record_sha256": previous_record_sha256,
    }
    return FailureRecord(
        **payload,
        record_sha256=owned_sha256(
            "vfe4.wt103.failure-record.v1",
            payload,
        ),
    )  # type: ignore[arg-type]


def _from_line(line: bytes) -> FailureRecord:
    try:
        document = json.loads(
            line.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeError, json.JSONDecodeError, MetricLogError) as exc:
        raise FailureLogError(f"failure JSON is invalid: {exc}") from exc
    if type(document) is not dict or frozenset(document) != _FAILURE_KEYS:
        raise FailureLogError("failure record has an invalid key set")
    if canonical_json_bytes(document) != line:
        raise FailureLogError("failure record is not canonical JSON")
    try:
        return FailureRecord(**document)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise FailureLogError(f"failure record is invalid: {exc}") from exc


def _decode(path: Path) -> tuple[tuple[FailureRecord, ...], bool]:
    try:
        payload = _read_regular_bytes(path)
    except MetricLogError as exc:
        raise FailureLogError(str(exc)) from exc
    try:
        lines, incomplete = _split_jsonl_payload(
            payload,
            label="failure log",
        )
    except MetricLogError as exc:
        raise FailureLogError(str(exc)) from exc
    records: list[FailureRecord] = []
    previous = _ZERO_SHA256
    for ordinal, line in enumerate(lines):
        if not line:
            raise FailureLogError("failure log contains an empty complete row")
        record = _from_line(line)
        if (
            record.ordinal != ordinal
            or record.previous_record_sha256 != previous
        ):
            raise FailureLogError("failure ordinal/hash chain is inconsistent")
        records.append(record)
        previous = record.record_sha256
    return tuple(records), incomplete


def validate_failure_log(path: Path) -> tuple[FailureRecord, ...]:
    records, _incomplete = _decode(path)
    return records


def append_failure(
    path: Path,
    record: FailureRecord,
    *,
    durability_backend: MetricDurabilityBackend,
) -> object:
    if type(record) is not FailureRecord:
        raise FailureLogError("record must be an exact FailureRecord")
    record.__post_init__()
    records, incomplete = _decode(path)
    if incomplete:
        raise FailureLogError(
            "failure log has an incomplete fragment; recover it first"
        )
    previous = _ZERO_SHA256 if not records else records[-1].record_sha256
    if (
        record.ordinal != len(records)
        or record.previous_record_sha256 != previous
    ):
        raise FailureLogError(
            "failure append does not extend the current chain"
        )
    try:
        payload = b"".join(
            canonical_json_bytes(existing) + b"\n"
            for existing in records
        )
        payload += canonical_json_bytes(record) + b"\n"
        return _append_canonical_line(
            path,
            payload,
            durability_backend=durability_backend,
        )
    except MetricLogError as exc:
        raise FailureLogError(str(exc)) from exc


def recover_incomplete_failure_fragment(
    path: Path,
    *,
    durability_backend: MetricDurabilityBackend,
) -> object:
    records, incomplete = _decode(path)
    if not incomplete:
        raise FailureLogError(
            "failure log has no proven incomplete final fragment"
        )
    payload = b"".join(
        canonical_json_bytes(record) + b"\n"
        for record in records
    )
    try:
        return _append_canonical_line(
            path,
            payload,
            durability_backend=durability_backend,
        )
    except MetricLogError as exc:
        raise FailureLogError(str(exc)) from exc


__all__ = [
    "FailureLogError",
    "FailureRecord",
    "append_failure",
    "create_failure_record",
    "recover_incomplete_failure_fragment",
    "validate_failure_log",
]
