"""Read-only finalized metric-table projections."""

from __future__ import annotations

from pathlib import Path

from vfe4.types.training import MetricRecord

from .metrics import _read_regular_bytes, validate_metric_log


def load_finalized_metric_rows(path: Path) -> tuple[MetricRecord, ...]:
    """Load a complete log and reject any incomplete terminal fragment."""

    payload = _read_regular_bytes(path)
    if payload and not payload.endswith(b"\n"):
        raise ValueError("finalized metric log has an incomplete fragment")
    records = validate_metric_log(path)
    if not records:
        raise ValueError("finalized metric log is empty")
    return records


__all__ = ["load_finalized_metric_rows"]
