"""Bounded stdout progress telemetry for click-run WikiText-103 operations.

Progress events are an operator convenience only.  Durable experiment
artifacts remain the scientific authority.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Protocol, TextIO, runtime_checkable


_PREFIX = "VFE4_PROGRESS "
_SCHEMA_VERSION = "wt103-progress-event-v1"


def _canonical_event_json(
    event: str,
    payload: Mapping[str, object],
) -> str:
    if (
        type(event) is not str
        or not event
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789_"
            for character in event
        )
    ):
        raise ValueError("progress event name is not canonical")
    if not isinstance(payload, Mapping):
        raise TypeError("progress payload must be a mapping")
    if "event" in payload or "schema_version" in payload:
        raise ValueError("progress payload cannot replace envelope fields")
    document = {
        "schema_version": _SCHEMA_VERSION,
        "event": event,
        **dict(payload),
    }
    try:
        return json.dumps(
            document,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("progress payload is not canonical JSON") from exc


@runtime_checkable
class ProgressReporter(Protocol):
    """In-process sink for one already validated progress event."""

    def report(
        self,
        event: str,
        payload: Mapping[str, object],
        /,
    ) -> None: ...


class NullProgressReporter:
    """Silent default for library callers."""

    def report(
        self,
        event: str,
        payload: Mapping[str, object],
        /,
    ) -> None:
        del event, payload


class ConsoleProgressReporter:
    """Write one canonical JSON line per lifecycle event."""

    def __init__(self, *, stream: TextIO | None = None) -> None:
        self._stream = sys.stdout if stream is None else stream

    def report(
        self,
        event: str,
        payload: Mapping[str, object],
        /,
    ) -> None:
        line = _PREFIX + _canonical_event_json(event, payload)
        print(line, file=self._stream, flush=True)


_NULL_REPORTER = NullProgressReporter()
_CURRENT_REPORTER: ContextVar[ProgressReporter] = ContextVar(
    "vfe4_wt103_progress_reporter",
    default=_NULL_REPORTER,
)


@contextmanager
def use_progress_reporter(
    reporter: ProgressReporter | None,
) -> Iterator[ProgressReporter]:
    """Install one reporter without changing orchestration driver protocols."""

    selected = _NULL_REPORTER if reporter is None else reporter
    if not isinstance(selected, ProgressReporter):
        raise TypeError("progress reporter does not implement report()")
    token = _CURRENT_REPORTER.set(selected)
    try:
        yield selected
    finally:
        _CURRENT_REPORTER.reset(token)


def emit_progress(event: str, **payload: object) -> None:
    """Validate and synchronously publish one progress event.

    Reporter exceptions deliberately propagate.  The caller therefore cannot
    continue past a lifecycle boundary after claiming it to a failed sink.
    """

    _canonical_event_json(event, payload)
    _CURRENT_REPORTER.get().report(event, payload)


__all__ = [
    "ConsoleProgressReporter",
    "NullProgressReporter",
    "ProgressReporter",
    "emit_progress",
    "use_progress_reporter",
]
