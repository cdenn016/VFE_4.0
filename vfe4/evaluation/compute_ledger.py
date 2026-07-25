"""Inference-inclusive H6 compute disclosure, separate from match eligibility."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

from vfe4.types.h6 import (
    H6_INFERENCE_PARTICLE_COUNTS,
    InferenceComputeRecord,
    canonical_json_bytes,
)


_LOWER_HEX = frozenset("0123456789abcdef")


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase 64-hex SHA-256")
    return value


def _owned_hash(domain: str, payload: object) -> str:
    return hashlib.sha256(
        domain.encode("ascii") + b"\x00" + canonical_json_bytes(payload)
    ).hexdigest()


def _new_frozen(cls: type[object], **values: object) -> object:
    instance = object.__new__(cls)
    for name, value in values.items():
        object.__setattr__(instance, name, value)
    instance.__post_init__()  # type: ignore[attr-defined]
    return instance


def _record_payload(record: InferenceComputeRecord) -> dict[str, object]:
    record.__post_init__()
    return {
        **record.canonical_payload(),
        "record_sha256": record.record_sha256,
    }


def _particle_order(record: InferenceComputeRecord) -> int:
    return -1 if record.particle_count is None else record.particle_count


@dataclass(frozen=True, slots=True, init=False)
class _InferenceInclusiveComputeRow:
    """One training-plus-inference disclosure row without an eligibility flag."""

    training_flops: int
    inference: InferenceComputeRecord
    declared_workload_total_flops: int
    row_sha256: str

    @property
    def endpoint_id(self) -> str:
        return self.inference.endpoint_id

    def canonical_payload(self) -> dict[str, object]:
        return {
            "endpoint_id": self.inference.endpoint_id,
            "training_flops": self.training_flops,
            "inference": _record_payload(self.inference),
            "declared_workload_total_flops": (
                self.declared_workload_total_flops
            ),
        }

    def __post_init__(self) -> None:
        if type(self.training_flops) is not int or self.training_flops <= 0:
            raise ValueError("training_flops must be a positive integer")
        if type(self.inference) is not InferenceComputeRecord:
            raise ValueError("inference row requires an exact compute record")
        self.inference.__post_init__()
        if self.declared_workload_total_flops != (
            self.training_flops + self.inference.total_flops
        ):
            raise ValueError(
                "declared-workload total must equal training and inference "
                "FLOPs"
            )
        expected = _owned_hash(
            "vfe4.h6.inference-inclusive-compute-row.v1",
            self.canonical_payload(),
        )
        if self.row_sha256 != expected:
            raise ValueError("inference-inclusive row digest is stale")

    @classmethod
    def create(
        cls,
        *,
        training_flops: int,
        inference: InferenceComputeRecord,
    ) -> "_InferenceInclusiveComputeRow":
        if type(inference) is not InferenceComputeRecord:
            raise ValueError("inference row requires an exact compute record")
        inference.__post_init__()
        values: dict[str, object] = {
            "training_flops": training_flops,
            "inference": inference,
            "declared_workload_total_flops": (
                training_flops + inference.total_flops
            ),
        }
        return _new_frozen(
            cls,
            **values,
            row_sha256=_owned_hash(
                "vfe4.h6.inference-inclusive-compute-row.v1",
                {
                    "endpoint_id": inference.endpoint_id,
                    "training_flops": training_flops,
                    "inference": _record_payload(inference),
                    "declared_workload_total_flops": values[
                        "declared_workload_total_flops"
                    ],
                },
            ),
        )  # type: ignore[return-value]


@dataclass(frozen=True, slots=True, init=False)
class _InferenceInclusiveComputeReport:
    """A disclosure table that cannot authorize or alter training matching."""

    schema_version: Literal["h6-inference-inclusive-compute-v1"]
    training_matching_set_sha256: str
    scientific_match_claim: Literal["training-compute-matched"]
    rows: tuple[_InferenceInclusiveComputeRow, ...]
    report_sha256: str

    @property
    def training_flops_by_endpoint(self) -> tuple[tuple[str, int], ...]:
        observed: dict[str, int] = {}
        for row in self.rows:
            observed.setdefault(row.endpoint_id, row.training_flops)
        return tuple(observed.items())

    @property
    def scoring_flops_by_endpoint_and_particle(
        self,
    ) -> tuple[tuple[str, int | None, int], ...]:
        return tuple(
            (
                row.endpoint_id,
                row.inference.particle_count,
                row.inference.scoring_flops,
            )
            for row in self.rows
        )

    @property
    def declared_workload_totals(
        self,
    ) -> tuple[tuple[str, int | None, int], ...]:
        return tuple(
            (
                row.endpoint_id,
                row.inference.particle_count,
                row.declared_workload_total_flops,
            )
            for row in self.rows
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "training_matching_set_sha256": (
                self.training_matching_set_sha256
            ),
            "scientific_match_claim": self.scientific_match_claim,
            "rows": tuple(
                {
                    **row.canonical_payload(),
                    "row_sha256": row.row_sha256,
                }
                for row in self.rows
            ),
        }

    def __post_init__(self) -> None:
        if self.schema_version != "h6-inference-inclusive-compute-v1":
            raise ValueError("unsupported inference-inclusive report schema")
        _require_sha256(
            self.training_matching_set_sha256,
            "training_matching_set_sha256",
        )
        if self.scientific_match_claim != "training-compute-matched":
            raise ValueError(
                "the scientific match claim must be training-compute-matched"
            )
        if (
            type(self.rows) is not tuple
            or not self.rows
            or any(
                type(row) is not _InferenceInclusiveComputeRow
                for row in self.rows
            )
        ):
            raise ValueError(
                "inference-inclusive report requires exact nonempty rows"
            )
        for row in self.rows:
            row.__post_init__()
        expected_order = tuple(
            sorted(
                self.rows,
                key=lambda row: (
                    row.endpoint_id,
                    _particle_order(row.inference),
                ),
            )
        )
        if self.rows != expected_order:
            raise ValueError(
                "inference-inclusive rows must use canonical endpoint/particle "
                "order"
            )
        grouped: dict[str, list[_InferenceInclusiveComputeRow]] = {}
        for row in self.rows:
            grouped.setdefault(row.endpoint_id, []).append(row)
        for endpoint_id, endpoint_rows in grouped.items():
            training_flops = {
                row.training_flops for row in endpoint_rows
            }
            scorer_kinds = {
                row.inference.scorer_kind for row in endpoint_rows
            }
            if len(training_flops) != 1 or len(scorer_kinds) != 1:
                raise ValueError(
                    "each endpoint requires one training total and scorer kind"
                )
            scorer_kind = endpoint_rows[0].inference.scorer_kind
            if scorer_kind == "exact_autoregressive":
                if (
                    len(endpoint_rows) != 1
                    or endpoint_rows[0].inference.particle_count is not None
                ):
                    raise ValueError(
                        "exact autoregressive endpoint requires one "
                        "particle-free row"
                    )
                continue
            particles = tuple(
                row.inference.particle_count for row in endpoint_rows
            )
            if particles != H6_INFERENCE_PARTICLE_COUNTS:
                raise ValueError(
                    "weighted SMC requires the complete ordered "
                    "128/256/512/1024 particle table"
                )
            if (
                len(
                    {
                        row.inference.replicate_count
                        for row in endpoint_rows
                    }
                )
                != 1
                or len(
                    {
                        row.inference.prefix_cache_mode
                        for row in endpoint_rows
                    }
                )
                != 1
                or len(
                    {
                        row.inference.checkpoint_load_flops
                        for row in endpoint_rows
                    }
                )
                != 1
                or len(
                    {
                        row.inference.cache_build_flops
                        for row in endpoint_rows
                    }
                )
                != 1
            ):
                raise ValueError(
                    "weighted SMC particle rows must share replicate, cache, "
                    "and one-time setup contracts"
                )
            scoring_flops = tuple(
                row.inference.scoring_flops for row in endpoint_rows
            )
            if any(
                right <= left
                for left, right in zip(
                    scoring_flops[:-1],
                    scoring_flops[1:],
                    strict=True,
                )
            ):
                raise ValueError(
                    "weighted SMC scoring FLOPs must increase strictly with "
                    "particle count"
                )
        expected = _owned_hash(
            "vfe4.h6.inference-inclusive-compute-report.v1",
            self.canonical_payload(),
        )
        if self.report_sha256 != expected:
            raise ValueError("inference-inclusive report digest is stale")

    @classmethod
    def create(
        cls,
        *,
        training_matching_set_sha256: str,
        rows: tuple[_InferenceInclusiveComputeRow, ...],
    ) -> "_InferenceInclusiveComputeReport":
        if (
            type(rows) is not tuple
            or not rows
            or any(
                type(row) is not _InferenceInclusiveComputeRow
                for row in rows
            )
        ):
            raise ValueError(
                "inference-inclusive report requires exact nonempty rows"
            )
        for row in rows:
            row.__post_init__()
        ordered_rows = tuple(
            sorted(
                tuple(rows),
                key=lambda row: (
                    row.endpoint_id,
                    _particle_order(row.inference),
                ),
            )
        )
        values: dict[str, object] = {
            "schema_version": "h6-inference-inclusive-compute-v1",
            "training_matching_set_sha256": training_matching_set_sha256,
            "scientific_match_claim": "training-compute-matched",
            "rows": ordered_rows,
        }
        provisional = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(provisional, name, value)
        return _new_frozen(
            cls,
            **values,
            report_sha256=_owned_hash(
                "vfe4.h6.inference-inclusive-compute-report.v1",
                provisional.canonical_payload(),
            ),
        )  # type: ignore[return-value]


def _build_inference_inclusive_compute_report(
    *,
    training_matching_set_sha256: str,
    training_flops_by_endpoint: tuple[tuple[str, int], ...],
    scorer_authorization: tuple[
        tuple[
            str,
            Literal["exact_autoregressive", "weighted_smc"],
        ],
        ...,
    ],
    inference_records: tuple[InferenceComputeRecord, ...],
) -> _InferenceInclusiveComputeReport:
    """Build a complete disclosure without evaluating a matching predicate."""

    if (
        type(training_flops_by_endpoint) is not tuple
        or not training_flops_by_endpoint
    ):
        raise ValueError("training FLOP inventory must be a nonempty tuple")
    training: dict[str, int] = {}
    for item in training_flops_by_endpoint:
        if (
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or not item[0]
            or type(item[1]) is not int
            or item[1] <= 0
            or item[0] in training
        ):
            raise ValueError(
                "training FLOP inventory has a malformed or duplicate row"
            )
        training[item[0]] = item[1]
    if type(scorer_authorization) is not tuple or not scorer_authorization:
        raise ValueError(
            "scorer authorization must be a nonempty exact tuple"
        )
    authorized: dict[
        str,
        Literal["exact_autoregressive", "weighted_smc"],
    ] = {}
    for item in scorer_authorization:
        if (
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or not item[0]
            or item[1] not in (
                "exact_autoregressive",
                "weighted_smc",
            )
            or item[0] in authorized
        ):
            raise ValueError(
                "scorer authorization has a malformed or duplicate row"
            )
        authorized[item[0]] = item[1]
    if set(authorized) != set(training):
        raise ValueError(
            "training and scorer authorization inventories must cover "
            "identical endpoints"
        )
    if (
        type(inference_records) is not tuple
        or not inference_records
        or any(
            type(record) is not InferenceComputeRecord
            for record in inference_records
        )
    ):
        raise ValueError("inference inventory must contain exact records")
    for record in inference_records:
        record.__post_init__()
    record_endpoints = {record.endpoint_id for record in inference_records}
    if record_endpoints != set(training):
        raise ValueError(
            "training and inference inventories must cover identical endpoints"
        )
    for record in inference_records:
        if record.scorer_kind != authorized[record.endpoint_id]:
            raise ValueError(
                "inference scorer differs from typed endpoint authorization"
            )
    rows = tuple(
        _InferenceInclusiveComputeRow.create(
            training_flops=training[record.endpoint_id],
            inference=record,
        )
        for record in inference_records
    )
    return _InferenceInclusiveComputeReport.create(
        training_matching_set_sha256=training_matching_set_sha256,
        rows=rows,
    )


__all__: list[str] = []
