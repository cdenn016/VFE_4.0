from __future__ import annotations

import io
from contextlib import redirect_stdout

import pytest


def test_progress_reporters_are_canonical_silent_and_ordered() -> None:
    from vfe4.training.progress import (
        ConsoleProgressReporter,
        NullProgressReporter,
        emit_progress,
        use_progress_reporter,
    )

    stream = io.StringIO()
    with use_progress_reporter(ConsoleProgressReporter(stream=stream)):
        emit_progress("plan_ready", z_value=2, a_value="first")
        emit_progress("resources_forecast", forecast_sha256="f" * 64)

    assert stream.getvalue().splitlines() == [
        (
            'VFE4_PROGRESS {"a_value":"first","event":"plan_ready",'
            '"schema_version":"wt103-progress-event-v1","z_value":2}'
        ),
        (
            'VFE4_PROGRESS {"event":"resources_forecast",'
            '"forecast_sha256":"'
            + "f" * 64
            + '","schema_version":"wt103-progress-event-v1"}'
        ),
    ]

    silent = io.StringIO()
    with redirect_stdout(silent), use_progress_reporter(
        NullProgressReporter()
    ):
        emit_progress("run_resolved", config_sha256="a" * 64)
    assert silent.getvalue() == ""


def test_progress_reporter_failure_propagates_without_a_success_line() -> None:
    from vfe4.training.progress import emit_progress, use_progress_reporter

    class _FailingReporter:
        def report(self, event: str, payload: object, /) -> None:
            del event, payload
            raise RuntimeError("progress sink failed")

    with use_progress_reporter(_FailingReporter()):
        with pytest.raises(RuntimeError, match="progress sink failed"):
            emit_progress("attempt_finished", disposition="success")
