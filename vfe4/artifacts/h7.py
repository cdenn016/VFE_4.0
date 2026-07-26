"""Atomic H7 evidence serialization surfaces."""

from __future__ import annotations

import json
import math
import struct
from typing import Any

import torch

from vfe4.types.h7 import (
    H7_MATRIX_TRIAL_IDS,
    H7_REQUIRED_TRIAL_IDS,
    H7_SCALAR_TRIAL_IDS,
    H7OwnedTensorSnapshot,
    H7Task5PrecisionCaptureBatch,
    h7_owned_sha256,
)
from vfe4.validation.h7_fixture import (
    H1_FIXTURE_RAW_SHA256,
    H7_FIXTURE_RAW_SHA256,
)


_PRECISION_TABLE_SCHEMA = "h7-mp-precision-operands-v2"
_PRECISION_SOURCE_CONTRACT = (
    "task5-production-covariance-and-precision-v2"
)
_BINARY64_TEXT_POLICY = "python-repr-binary64-roundtrip-v1"
_COVARIANCE_VALUES_DOMAIN = (
    "vfe4.h7.mp-serialized-covariance-values.v2"
)
_PRECISION_VALUES_DOMAIN = "vfe4.h7.mp-serialized-precision-values.v2"
_PRECISION_ROW_DOMAIN = "vfe4.h7.mp-serialized-precision-operand.v2"
_PRECISION_SET_DOMAIN = "vfe4.h7.mp-serialized-precision-set.v2"
_EXPECTED_BATCH_KEYS = (
    *(
        (trial_id, "h1-v1", "structured_full_block")
        for trial_id in H7_SCALAR_TRIAL_IDS
    ),
    *(
        key
        for trial_id in H7_MATRIX_TRIAL_IDS
        for key in (
            (trial_id, "h7-v1", "structured_full_block"),
            (
                trial_id,
                "h7-v1",
                "factorized_diagonal_within_fiber",
            ),
        )
    ),
)


def _canonical_table_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _canonical_binary64_token(value: float) -> str:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError("H7 precision table values must be finite binary64")
    token = repr(value)
    reparsed = float(token)
    if (
        repr(reparsed) != token
        or struct.pack("<d", reparsed) != struct.pack("<d", value)
    ):
        raise ValueError(
            "H7 precision table value violates the binary64 text policy"
        )
    return token


def _tensor_values(snapshot: H7OwnedTensorSnapshot) -> list[Any]:
    if type(snapshot) is not H7OwnedTensorSnapshot:
        raise ValueError("H7 precision table requires exact owned snapshots")
    snapshot.assert_intact()
    if (
        snapshot.dtype != "float64"
        or snapshot.device != "cpu"
        or len(snapshot.shape) != 2
        or snapshot.shape[0] <= 0
        or snapshot.shape[0] != snapshot.shape[1]
    ):
        raise ValueError(
            "H7 precision table tensors must be square CPU float64 matrices"
        )
    tensor = snapshot.value()
    if tensor.dtype != torch.float64 or tensor.device.type != "cpu":
        raise ValueError("H7 precision table tensor ownership changed")
    rows: list[Any] = []
    for row in tensor.tolist():
        rows.append([_canonical_binary64_token(item) for item in row])
    return rows


def _serialized_snapshot_sha256(
    snapshot: H7OwnedTensorSnapshot,
) -> str:
    """Hash the canonical contiguous CPU float64 serialization snapshot."""

    return H7OwnedTensorSnapshot.capture(
        snapshot.value().contiguous()
    ).snapshot_sha256


def _value_sha256(
    *,
    domain: str,
    trial_id: str,
    gaussian_id: str,
    source_kind: str,
    shape: list[str],
    value_name: str,
    values: list[Any],
) -> str:
    return h7_owned_sha256(
        domain,
        {
            "trial_id": trial_id,
            "gaussian_id": gaussian_id,
            "source_kind": source_kind,
            "shape": shape,
            value_name: values,
        },
    )


def _validate_batch_inventory(
    batches: tuple[H7Task5PrecisionCaptureBatch, ...],
) -> None:
    if (
        type(batches) is not tuple
        or len(batches) != len(_EXPECTED_BATCH_KEYS)
        or any(type(item) is not H7Task5PrecisionCaptureBatch for item in batches)
    ):
        raise ValueError(
            "H7 precision table requires the exact fourteen capture batches"
        )
    observed_keys = tuple(
        (batch.trial_id, batch.fixture_id, batch.recognition_family)
        for batch in batches
    )
    if observed_keys != _EXPECTED_BATCH_KEYS:
        raise ValueError("H7 precision capture batches are reordered or missing")
    for batch in batches:
        batch.__post_init__()
        expected_raw = (
            H1_FIXTURE_RAW_SHA256
            if batch.fixture_id == "h1-v1"
            else H7_FIXTURE_RAW_SHA256
        )
        if batch.raw_fixture_sha256 != expected_raw:
            raise ValueError(
                "H7 precision capture batch changed raw fixture identity"
            )


def build_h7_task5_precision_operand_table_bytes(
    batches: tuple[H7Task5PrecisionCaptureBatch, ...],
) -> bytes:
    """Serialize the closed production Task-5 covariance/precision inventory."""

    _validate_batch_inventory(batches)
    records: list[dict[str, object]] = []
    source_counts = {"owned_component": 0, "assembled_global": 0}
    for batch in batches:
        for operand in batch.operands:
            operand.__post_init__()
            shape = [str(item) for item in operand.covariance.shape]
            covariance_values = _tensor_values(operand.covariance)
            precision_values = _tensor_values(operand.precision)
            identity = {
                "trial_id": operand.trial_id,
                "gaussian_id": operand.gaussian_id,
                "source_kind": operand.source_kind,
                "shape": shape,
            }
            row: dict[str, object] = {
                "row_index": str(len(records)),
                **identity,
                "covariance_values": covariance_values,
                "covariance_values_sha256": _value_sha256(
                    domain=_COVARIANCE_VALUES_DOMAIN,
                    **identity,
                    value_name="covariance_values",
                    values=covariance_values,
                ),
                "covariance_snapshot_sha256": (
                    _serialized_snapshot_sha256(operand.covariance)
                ),
                "precision_values": precision_values,
                "precision_values_sha256": _value_sha256(
                    domain=_PRECISION_VALUES_DOMAIN,
                    **identity,
                    value_name="precision_values",
                    values=precision_values,
                ),
                "precision_snapshot_sha256": (
                    _serialized_snapshot_sha256(operand.precision)
                ),
            }
            row["record_sha256"] = h7_owned_sha256(
                _PRECISION_ROW_DOMAIN,
                row,
            )
            records.append(row)
            source_counts[operand.source_kind] += 1
    if (
        len(records) != 192
        or source_counts
        != {"owned_component": 152, "assembled_global": 40}
    ):
        raise ValueError("H7 precision table inventory/cardinality changed")
    root: dict[str, object] = {
        "precision_table_schema": _PRECISION_TABLE_SCHEMA,
        "h1_raw_fixture_sha256": H1_FIXTURE_RAW_SHA256,
        "h7_raw_fixture_sha256": H7_FIXTURE_RAW_SHA256,
        "ordered_trial_ids": list(H7_REQUIRED_TRIAL_IDS),
        "source_contract": _PRECISION_SOURCE_CONTRACT,
        "binary64_text_policy": _BINARY64_TEXT_POLICY,
        "records": records,
    }
    root["precision_set_sha256"] = h7_owned_sha256(
        _PRECISION_SET_DOMAIN,
        root,
    )
    return _canonical_table_bytes(root)


__all__ = ["build_h7_task5_precision_operand_table_bytes"]
