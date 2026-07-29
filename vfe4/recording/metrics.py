"""Canonical authoritative WT103 metric JSONL and deterministic CSV export."""

from __future__ import annotations

import csv
import dataclasses
import io
import json
import math
import os
import stat
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Protocol

from vfe4.types.training import (
    MetricRecord,
    MetricValue,
    WT103ArmSpec,
    canonical_json_bytes,
    owned_sha256,
)

WT103_REQUIRED_METRIC_FAMILIES = (
    "train_cross_entropy",
    "complete_elbo",
    "expected_log_emission",
    "initial_model_cross_entropy",
    "initial_state_cross_entropy",
    "model_source_cross_entropy",
    "model_source_kl",
    "model_transition_cross_entropy",
    "state_source_cross_entropy",
    "state_source_kl",
    "state_transition_cross_entropy",
    "continuous_recognition_entropy",
    "conditional_source_entropy_estimate",
    "joint_recognition_entropy_estimate",
    "estimator_error_bound",
    "emission_only_non_elbo",
    "prior_nll_sum",
    "prior_nll_per_token",
    "perplexity",
    "estimator_stream",
    "particle_count",
    "cache_audit_passed",
    "source_entropy",
    "source_support_size",
    "effective_source_count",
    "accepted_proposals",
    "rejected_proposals",
    "acceptance_rate",
    "objective_before",
    "objective_after",
    "snapshot_identity_present",
    "learning_rate",
    "scheduler_ordinal",
    "gradient_pre_clip_l2",
    "gradient_post_clip_l2",
    "minimum_cholesky_pivot",
    "failed_pivots",
    "spd_projections",
    "condition_estimate",
    "solve_residual",
    "damping_events",
    "gradient_l2",
    "gradient_inf",
    "counted_targets",
    "tokens_per_second",
    "data_wait_seconds",
    "forward_seconds",
    "inference_seconds",
    "backward_seconds",
    "update_seconds",
    "evaluation_seconds",
    "checkpoint_seconds",
    "wall_seconds",
    "process_rss_bytes",
    "process_hwm_bytes",
    "cuda_allocated_bytes",
    "cuda_reserved_bytes",
    "cuda_peak_allocated_bytes",
    "cuda_peak_reserved_bytes",
    "allocation_retries",
    "oom_count",
)

WT103_METRIC_UNIT_BY_NAME = MappingProxyType(
    {
        "train_cross_entropy": "nats_per_token",
        "complete_elbo": "nats_per_token",
        "expected_log_emission": "nats_per_token",
        "initial_model_cross_entropy": "nats_per_token",
        "initial_state_cross_entropy": "nats_per_token",
        "model_source_cross_entropy": "nats_per_token",
        "model_source_kl": "nats_per_token",
        "model_transition_cross_entropy": "nats_per_token",
        "state_source_cross_entropy": "nats_per_token",
        "state_source_kl": "nats_per_token",
        "state_transition_cross_entropy": "nats_per_token",
        "continuous_recognition_entropy": "nats_per_token",
        "conditional_source_entropy_estimate": "nats_per_token",
        "joint_recognition_entropy_estimate": "nats_per_token",
        "estimator_error_bound": "nats_per_token",
        "emission_only_non_elbo": "nats_per_token",
        "prior_nll_sum": "nats",
        "prior_nll_per_token": "nats_per_token",
        "perplexity": "perplexity",
        "estimator_stream": "stream_index",
        "particle_count": "particles",
        "cache_audit_passed": "boolean",
        "source_entropy": "nats_per_source_row",
        "source_support_size": "sources",
        "effective_source_count": "effective_sources",
        "accepted_proposals": "proposals",
        "rejected_proposals": "proposals",
        "acceptance_rate": "fraction",
        "objective_before": "nats_per_token",
        "objective_after": "nats_per_token",
        "snapshot_identity_present": "boolean",
        "learning_rate": "scalar",
        "scheduler_ordinal": "update_index",
        "gradient_pre_clip_l2": "l2_norm",
        "gradient_post_clip_l2": "l2_norm",
        "minimum_cholesky_pivot": "scalar",
        "failed_pivots": "count",
        "spd_projections": "count",
        "condition_estimate": "ratio",
        "solve_residual": "scalar",
        "damping_events": "count",
        "gradient_l2": "l2_norm",
        "gradient_inf": "linf_norm",
        "counted_targets": "targets",
        "tokens_per_second": "tokens_per_second",
        "data_wait_seconds": "seconds",
        "forward_seconds": "seconds",
        "inference_seconds": "seconds",
        "backward_seconds": "seconds",
        "update_seconds": "seconds",
        "evaluation_seconds": "seconds",
        "checkpoint_seconds": "seconds",
        "wall_seconds": "seconds",
        "process_rss_bytes": "bytes",
        "process_hwm_bytes": "bytes",
        "cuda_allocated_bytes": "bytes",
        "cuda_reserved_bytes": "bytes",
        "cuda_peak_allocated_bytes": "bytes",
        "cuda_peak_reserved_bytes": "bytes",
        "allocation_retries": "count",
        "oom_count": "count",
    }
)
if tuple(WT103_METRIC_UNIT_BY_NAME) != WT103_REQUIRED_METRIC_FAMILIES:
    raise RuntimeError("required metric and frozen-unit inventories differ")

WT103_METRIC_SEMANTIC_BY_NAME = MappingProxyType(
    {
        "train_cross_entropy": "ratio",
        "complete_elbo": "ratio",
        "expected_log_emission": "ratio",
        "initial_model_cross_entropy": "ratio",
        "initial_state_cross_entropy": "ratio",
        "model_source_cross_entropy": "ratio",
        "model_source_kl": "ratio",
        "model_transition_cross_entropy": "ratio",
        "state_source_cross_entropy": "ratio",
        "state_source_kl": "ratio",
        "state_transition_cross_entropy": "ratio",
        "continuous_recognition_entropy": "ratio",
        "conditional_source_entropy_estimate": "ratio",
        "joint_recognition_entropy_estimate": "ratio",
        "estimator_error_bound": "not_applicable_only",
        "emission_only_non_elbo": "ratio",
        "prior_nll_sum": "scalar",
        "prior_nll_per_token": "ratio",
        "perplexity": "exp_ratio",
        "estimator_stream": "scalar",
        "particle_count": "scalar",
        "cache_audit_passed": "scalar",
        "source_entropy": "ratio",
        "source_support_size": "ratio",
        "effective_source_count": "exp_ratio",
        "accepted_proposals": "scalar",
        "rejected_proposals": "scalar",
        "acceptance_rate": "ratio",
        "objective_before": "ratio",
        "objective_after": "ratio",
        "snapshot_identity_present": "scalar",
        "learning_rate": "scalar",
        "scheduler_ordinal": "scalar",
        "gradient_pre_clip_l2": "scalar",
        "gradient_post_clip_l2": "scalar",
        "minimum_cholesky_pivot": "scalar",
        "failed_pivots": "scalar",
        "spd_projections": "scalar",
        "condition_estimate": "scalar",
        "solve_residual": "scalar",
        "damping_events": "scalar",
        "gradient_l2": "scalar",
        "gradient_inf": "scalar",
        "counted_targets": "scalar",
        "tokens_per_second": "tokens_per_second",
        "data_wait_seconds": "scalar",
        "forward_seconds": "scalar",
        "inference_seconds": "scalar",
        "backward_seconds": "scalar",
        "update_seconds": "scalar",
        "evaluation_seconds": "scalar",
        "checkpoint_seconds": "scalar",
        "wall_seconds": "scalar",
        "process_rss_bytes": "scalar",
        "process_hwm_bytes": "scalar",
        "cuda_allocated_bytes": "scalar",
        "cuda_reserved_bytes": "scalar",
        "cuda_peak_allocated_bytes": "scalar",
        "cuda_peak_reserved_bytes": "scalar",
        "allocation_retries": "scalar",
        "oom_count": "scalar",
    }
)
if tuple(WT103_METRIC_SEMANTIC_BY_NAME) != WT103_REQUIRED_METRIC_FAMILIES:
    raise RuntimeError("required metric and semantic inventories differ")


_ZERO_SHA256 = "0" * 64
WT103_UNAVAILABLE_ESTIMATOR_BOUND_REASON = (
    "no_preregistered_finite_bound_for_single_sample_mc"
)
WT103_SOURCE_KL_DIAGNOSTIC_REASON = (
    "derived_source_kl_diagnostic:objective_term=false"
)
_MAXIMUM_METRIC_LOG_BYTES = 512 * 1024 * 1024
_METRIC_RECORD_KEYS = frozenset(
    {
        "schema_version",
        "ordinal",
        "utc_timestamp",
        "monotonic_ns",
        "run_id",
        "arm_id",
        "seed_id",
        "phase",
        "split",
        "step",
        "pass_index",
        "previous_record_sha256",
        "values",
        "record_sha256",
    }
)
_METRIC_VALUE_KEYS = frozenset(
    {
        "name",
        "applicability",
        "reason",
        "numerator",
        "denominator",
        "value",
        "units",
    }
)
_CSV_COLUMNS = (
    "ordinal",
    "utc_timestamp",
    "monotonic_ns",
    "run_id",
    "arm_id",
    "seed_id",
    "phase",
    "split",
    "step",
    "pass_index",
    "metric_name",
    "applicability",
    "reason",
    "numerator",
    "denominator",
    "value",
    "units",
    "record_sha256",
)


class MetricLogError(ValueError):
    """A metric record, chain, or durable append violated the contract."""


class MetricDurabilityBackend(Protocol):
    def publish_bytes(self, path: Path, payload: bytes) -> object: ...


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def metric_family_units(name: str) -> str:
    """Return the frozen exact unit for one required metric family."""

    try:
        return WT103_METRIC_UNIT_BY_NAME[name]
    except (KeyError, TypeError) as exc:
        raise MetricLogError(f"unknown metric family {name!r}") from exc


def _validate_metric_units(name: str, units: str) -> None:
    expected = WT103_METRIC_UNIT_BY_NAME.get(name)
    if expected is not None and units != expected:
        raise MetricLogError(
            f"metric family {name!r} requires frozen units {expected!r}"
        )


def _validate_metric_value(metric: MetricValue) -> None:
    _validate_metric_units(metric.name, metric.units)
    semantic = WT103_METRIC_SEMANTIC_BY_NAME.get(metric.name)
    if (
        semantic == "not_applicable_only"
        and metric.applicability == "applicable"
    ):
        raise MetricLogError(
            f"metric family {metric.name!r} is not-applicable-only"
        )
    if metric.applicability != "applicable" or semantic is None:
        return
    if (
        metric.name in {"model_source_kl", "state_source_kl"}
        and metric.reason != WT103_SOURCE_KL_DIAGNOSTIC_REASON
    ):
        raise MetricLogError(
            f"metric family {metric.name!r} requires the canonical "
            "diagnostic-only marker"
        )
    if semantic == "scalar":
        if metric.numerator is not None or metric.denominator is not None:
            raise MetricLogError(
                f"scalar metric family {metric.name!r} forbids raw "
                "numerator and denominator"
            )
        return
    if metric.numerator is None or metric.denominator is None:
        raise MetricLogError(
            f"derived metric family {metric.name!r} requires a raw "
            "numerator and denominator"
        )
    if semantic == "ratio":
        expected = metric.numerator / metric.denominator
    elif semantic == "exp_ratio":
        try:
            expected = math.exp(metric.numerator / metric.denominator)
        except OverflowError as exc:
            raise MetricLogError(
                f"metric family {metric.name!r} exact derivation overflowed"
            ) from exc
    elif semantic == "tokens_per_second":
        expected = metric.numerator / (
            metric.denominator / 1_000_000_000.0
        )
    else:  # pragma: no cover - guarded by the closed module inventory
        raise RuntimeError(f"unknown metric semantic {semantic!r}")
    if not math.isfinite(expected) or metric.value != expected:
        raise MetricLogError(
            f"metric family {metric.name!r} disagrees with its exact derivation"
        )


def applicable_metric(
    *,
    name: str,
    numerator: float | None,
    denominator: int | None,
    value: float,
    units: str,
    reason: str = "measured",
) -> MetricValue:
    metric = MetricValue(
        name=name,
        applicability="applicable",
        reason=reason,
        numerator=numerator,
        denominator=denominator,
        value=value,
        units=units,
    )
    _validate_metric_value(metric)
    return metric


def not_applicable_metric(
    *,
    name: str,
    reason: str,
    units: str,
) -> MetricValue:
    metric = MetricValue(
        name=name,
        applicability="not_applicable",
        reason=reason,
        numerator=None,
        denominator=None,
        value=None,
        units=units,
    )
    _validate_metric_value(metric)
    return metric


def metric_family_applicability(
    arm_spec: WT103ArmSpec,
    name: str,
) -> tuple[bool, str]:
    if type(arm_spec) is not WT103ArmSpec:
        raise MetricLogError("arm_spec must be an exact WT103ArmSpec")
    arm_spec.__post_init__()
    if name not in WT103_REQUIRED_METRIC_FAMILIES:
        raise MetricLogError(f"unknown metric family {name!r}")
    if name == "train_cross_entropy":
        applicable = arm_spec.training_objective == "cross_entropy"
        return (
            applicable,
            "cross_entropy_objective"
            if applicable
            else "arm_objective_is_not_cross_entropy",
        )
    complete_names = {
        "complete_elbo",
        "expected_log_emission",
        "initial_model_cross_entropy",
        "initial_state_cross_entropy",
        "model_source_cross_entropy",
        "model_source_kl",
        "model_transition_cross_entropy",
        "state_source_cross_entropy",
        "state_source_kl",
        "state_transition_cross_entropy",
        "continuous_recognition_entropy",
        "conditional_source_entropy_estimate",
        "joint_recognition_entropy_estimate",
        "estimator_error_bound",
    }
    if name in complete_names:
        applicable = arm_spec.training_objective == "complete_elbo"
        return (
            applicable,
            "complete_elbo_objective"
            if applicable
            else "arm_objective_is_not_complete_elbo",
        )
    if name == "emission_only_non_elbo":
        applicable = (
            arm_spec.training_objective
            == "emission_only_ablation_non_elbo"
        )
        return (
            applicable,
            "emission_only_objective"
            if applicable
            else "arm_objective_is_not_emission_only",
        )
    if name in {"estimator_stream", "particle_count"}:
        applicable = arm_spec.scorer_kind == "weighted_smc"
        return (
            applicable,
            "weighted_smc_scorer"
            if applicable
            else "exact_scorer_has_no_estimator_stream_or_particles",
        )
    latent_names = {
        "source_entropy",
        "source_support_size",
        "effective_source_count",
        "snapshot_identity_present",
        "minimum_cholesky_pivot",
        "failed_pivots",
        "spd_projections",
        "condition_estimate",
        "solve_residual",
        "damping_events",
        "inference_seconds",
    }
    if name in latent_names:
        return (
            arm_spec.latent_enabled,
            "latent_path_active"
            if arm_spec.latent_enabled
            else "arm_has_no_latent_or_recognition_path",
        )
    return True, "required_for_every_applicable_arm"


def validate_required_metric_families(
    records: tuple[MetricRecord, ...],
    *,
    arm_spec: WT103ArmSpec,
) -> None:
    """Validate required families over a complete multi-phase metric log.

    A record contains only metrics actually observed in that phase. Required
    family coverage is the union across the finalized log; an applicable
    family is omitted, rather than fabricated or mislabeled not-applicable, in
    phases where it was not observed.
    """

    if (
        type(records) is not tuple
        or not records
        or any(type(record) is not MetricRecord for record in records)
    ):
        raise MetricLogError(
            "required-family validation needs exact metric records"
        )
    values_by_name: dict[str, list[MetricValue]] = {
        name: [] for name in WT103_REQUIRED_METRIC_FAMILIES
    }
    latent_source_names = (
        "source_entropy",
        "source_support_size",
        "effective_source_count",
    )
    for record in records:
        record.__post_init__()
        if record.arm_id != arm_spec.arm_id:
            raise MetricLogError(
                "metric family validation cannot mix arm IDs"
            )
        source_values = tuple(
            value
            for value in record.values
            if value.name in latent_source_names
        )
        if arm_spec.latent_enabled and source_values:
            source_applicabilities = {
                value.applicability for value in source_values
            }
            if len(source_applicabilities) != 1:
                raise MetricLogError(
                    "one metric record mixes zero and positive source rows"
                )
            if any(
                value.applicability == "not_applicable"
                and value.reason != "source_row_count_is_zero"
                for value in source_values
            ):
                raise MetricLogError(
                    "latent source metrics require the canonical zero-row "
                    "N/A reason"
                )
        for value in record.values:
            if value.name in values_by_name:
                _validate_metric_value(value)
                values_by_name[value.name].append(value)
    missing = tuple(
        name for name, values in values_by_name.items() if not values
    )
    if missing:
        raise MetricLogError(
            "required metric families are missing: " + ",".join(missing)
        )
    for name, values in values_by_name.items():
        applicable, reason = metric_family_applicability(arm_spec, name)
        is_latent_source = (
            arm_spec.latent_enabled and name in latent_source_names
        )
        is_complete_estimator_bound = (
            arm_spec.training_objective == "complete_elbo"
            and name == "estimator_error_bound"
        )
        if is_complete_estimator_bound:
            if any(
                value.applicability != "not_applicable"
                or value.reason != WT103_UNAVAILABLE_ESTIMATOR_BOUND_REASON
                for value in values
            ):
                raise MetricLogError(
                    "estimator_error_bound must use the canonical "
                    "unbounded-estimator N/A record"
                )
        elif is_latent_source:
            if any(
                value.applicability == "not_applicable"
                and value.reason != "source_row_count_is_zero"
                for value in values
            ):
                raise MetricLogError(
                    "latent source metrics require the canonical zero-row "
                    "N/A reason"
                )
        else:
            expected = "applicable" if applicable else "not_applicable"
            if any(value.applicability != expected for value in values):
                raise MetricLogError(
                    f"metric family {name!r} has wrong arm applicability"
                )
        if any(
            value.units != WT103_METRIC_UNIT_BY_NAME[name]
            for value in values
        ):
            raise MetricLogError(
                f"metric family {name!r} has noncanonical units"
            )
        if not applicable and any(value.reason != reason for value in values):
            raise MetricLogError(
                f"metric family {name!r} has a noncanonical N/A reason"
            )
    if arm_spec.latent_enabled:
        positive_source_rows_exist = any(
            value.applicability == "applicable"
            for name in latent_source_names
            for value in values_by_name[name]
        )
        if positive_source_rows_exist and any(
            not any(
                value.applicability == "applicable"
                for value in values_by_name[name]
            )
            for name in latent_source_names
        ):
            raise MetricLogError(
                "positive source rows require applicable observations for "
                "every latent source metric"
            )


def source_entropy_metrics(
    *,
    entropy_sum: float,
    source_row_count: int,
) -> tuple[MetricValue, MetricValue]:
    if (
        type(entropy_sum) is not float
        or not math.isfinite(entropy_sum)
        or entropy_sum < 0.0
        or type(source_row_count) is not int
        or source_row_count < 0
    ):
        raise MetricLogError(
            "source entropy accumulator must be finite with nonnegative rows"
        )
    if source_row_count == 0:
        reason = "source_row_count_is_zero"
        return (
            not_applicable_metric(
                name="source_entropy",
                reason=reason,
                units="nats_per_source_row",
            ),
            not_applicable_metric(
                name="effective_source_count",
                reason=reason,
                units="effective_sources",
            ),
        )
    mean_entropy = entropy_sum / source_row_count
    effective_count = math.exp(mean_entropy)
    if not math.isfinite(effective_count):
        raise MetricLogError("effective source count overflowed")
    return (
        applicable_metric(
            name="source_entropy",
            numerator=entropy_sum,
            denominator=source_row_count,
            value=mean_entropy,
            units="nats_per_source_row",
        ),
        applicable_metric(
            name="effective_source_count",
            numerator=entropy_sum,
            denominator=source_row_count,
            value=effective_count,
            units="effective_sources",
            reason="exp(source_entropy_sum/source_row_count)",
        ),
    )


def create_metric_record(
    *,
    ordinal: int,
    utc_timestamp: str,
    monotonic_ns: int,
    run_id: str,
    arm_id: str,
    seed_id: int,
    phase: str,
    split: Literal["train", "validation", "test", "not_applicable"],
    step: int,
    pass_index: int,
    previous_record_sha256: str,
    values: tuple[MetricValue, ...],
) -> MetricRecord:
    for value in values:
        _validate_metric_value(value)
    payload = {
        "schema_version": "wt103-metric-record-v1",
        "ordinal": ordinal,
        "utc_timestamp": utc_timestamp,
        "monotonic_ns": monotonic_ns,
        "run_id": run_id,
        "arm_id": arm_id,
        "seed_id": seed_id,
        "phase": phase,
        "split": split,
        "step": step,
        "pass_index": pass_index,
        "previous_record_sha256": previous_record_sha256,
        "values": values,
    }
    return MetricRecord(
        **payload,
        record_sha256=owned_sha256(
            "vfe4.wt103.metric-record.v1",
            payload,
        ),
    )  # type: ignore[arg-type]


def _reject_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise MetricLogError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _metric_value_from_document(value: object) -> MetricValue:
    if type(value) is not dict or frozenset(value) != _METRIC_VALUE_KEYS:
        raise MetricLogError("metric value has an open or invalid key set")
    try:
        metric = MetricValue(**value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise MetricLogError(f"metric value is invalid: {exc}") from exc
    _validate_metric_value(metric)
    return metric


def _metric_record_from_line(line: bytes) -> MetricRecord:
    try:
        document = json.loads(
            line.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_raise_nonfinite(value)),
        )
    except MetricLogError:
        raise
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise MetricLogError(f"metric JSON is invalid: {exc}") from exc
    if type(document) is not dict or frozenset(document) != _METRIC_RECORD_KEYS:
        raise MetricLogError("metric record has an open or invalid key set")
    if canonical_json_bytes(document) != line:
        raise MetricLogError("metric record is not canonical JSON")
    raw_values = document["values"]
    if type(raw_values) is not list or not raw_values:
        raise MetricLogError("metric values must be a nonempty JSON list")
    values = tuple(_metric_value_from_document(value) for value in raw_values)
    payload = dict(document)
    payload["values"] = values
    try:
        return MetricRecord(**payload)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise MetricLogError(f"metric record is invalid: {exc}") from exc


def _raise_nonfinite(value: str) -> object:
    raise MetricLogError(f"nonfinite JSON constant {value!r} is forbidden")


def _is_redirect_or_reparse(path: Path, status: os.stat_result) -> bool:
    if stat.S_ISLNK(status.st_mode):
        return True
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if bool(getattr(status, "st_file_attributes", 0) & reparse):
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction is not None and is_junction())


def _validate_parent(path: Path) -> None:
    try:
        status = path.lstat()
    except OSError as exc:
        raise MetricLogError(f"metric parent is unavailable: {exc}") from exc
    if not stat.S_ISDIR(status.st_mode) or _is_redirect_or_reparse(path, status):
        raise MetricLogError(
            "metric parent must be a regular nonlink directory"
        )


def _regular_or_absent(path: Path) -> None:
    _validate_parent(path.parent)
    try:
        status = path.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise MetricLogError(f"metric log metadata failed: {exc}") from exc
    if not stat.S_ISREG(status.st_mode) or _is_redirect_or_reparse(path, status):
        raise MetricLogError("metric log must be a regular nonlink file")
    if status.st_size > _MAXIMUM_METRIC_LOG_BYTES:
        raise MetricLogError("metric log exceeds its maximum size")


def _read_regular_bytes(path: Path) -> bytes:
    _regular_or_absent(path)
    if not path.exists():
        return b""
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size > _MAXIMUM_METRIC_LOG_BYTES
        ):
            raise MetricLogError("metric descriptor is not a bounded file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise MetricLogError("metric log changed while it was read")
    return b"".join(chunks)


def _split_jsonl_payload(
    payload: bytes,
    *,
    label: str,
) -> tuple[list[bytes], bool]:
    if not payload:
        return [], False
    rows = payload.split(b"\n")
    if rows[-1] == b"":
        rows.pop()
        return rows, False
    fragment = rows.pop()
    try:
        text = fragment.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        if exc.end != len(fragment):
            raise MetricLogError(
                f"{label} final fragment has malformed UTF-8"
            ) from exc
        return rows, True
    try:
        json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: _raise_nonfinite(value),
        )
    except MetricLogError:
        raise
    except json.JSONDecodeError as exc:
        proven_terminal_truncation = (
            exc.msg.startswith("Unterminated")
            or exc.pos >= len(text) - 1
        )
        if not proven_terminal_truncation:
            raise MetricLogError(
                f"{label} final fragment is malformed, not proven incomplete"
            ) from exc
        return rows, True
    except (TypeError, ValueError) as exc:
        raise MetricLogError(
            f"{label} final fragment is malformed"
        ) from exc
    raise MetricLogError(
        f"{label} contains a complete record without its required newline"
    )


def _decode_metric_log(path: Path) -> tuple[tuple[MetricRecord, ...], bool]:
    payload = _read_regular_bytes(path)
    rows, incomplete = _split_jsonl_payload(payload, label="metric log")
    records: list[MetricRecord] = []
    expected_previous = _ZERO_SHA256
    for ordinal, line in enumerate(rows):
        if not line:
            raise MetricLogError("metric log contains an empty complete row")
        record = _metric_record_from_line(line)
        if (
            record.ordinal != ordinal
            or record.previous_record_sha256 != expected_previous
        ):
            raise MetricLogError(
                "metric ordinal or previous-record hash is inconsistent"
            )
        records.append(record)
        expected_previous = record.record_sha256
    return tuple(records), incomplete


def validate_metric_log(path: Path) -> tuple[MetricRecord, ...]:
    """Validate all complete rows; ignore only an incomplete final fragment."""

    if not isinstance(path, Path):
        raise MetricLogError("metric path must be a Path")
    records, _incomplete = _decode_metric_log(path)
    return records


def recover_incomplete_metric_fragment(
    path: Path,
    *,
    durability_backend: MetricDurabilityBackend,
) -> object:
    """Durably truncate only a final fragment proven to be incomplete."""

    records, incomplete = _decode_metric_log(path)
    if not incomplete:
        raise MetricLogError(
            "metric log has no proven incomplete final fragment"
        )
    payload = b"".join(
        canonical_json_bytes(record) + b"\n"
        for record in records
    )
    return _append_canonical_line(
        path,
        payload,
        durability_backend=durability_backend,
    )


def _append_canonical_line(
    path: Path,
    payload: bytes,
    *,
    durability_backend: MetricDurabilityBackend,
) -> object:
    if not callable(getattr(durability_backend, "publish_bytes", None)):
        raise MetricLogError("durability backend must expose publish_bytes")
    try:
        identity = durability_backend.publish_bytes(path, payload)
    except Exception as exc:
        raise MetricLogError(
            f"durable log publication failed: {exc}"
        ) from exc
    if _read_regular_bytes(path) != payload:
        raise MetricLogError(
            "durable log reopen bytes do not match publication"
        )
    return identity


def append_metric(
    path: Path,
    record: MetricRecord,
    *,
    durability_backend: MetricDurabilityBackend,
) -> object:
    if type(record) is not MetricRecord:
        raise MetricLogError("record must be an exact MetricRecord")
    record.__post_init__()
    records, incomplete = _decode_metric_log(path)
    if incomplete:
        raise MetricLogError(
            "metric log has an incomplete final fragment; recover it first"
        )
    expected_previous = (
        _ZERO_SHA256 if not records else records[-1].record_sha256
    )
    if (
        record.ordinal != len(records)
        or record.previous_record_sha256 != expected_previous
    ):
        raise MetricLogError(
            "metric append does not extend the exact current chain"
        )
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


def _decimal(value: float | None) -> str:
    if value is None:
        return ""
    if not math.isfinite(value):
        raise MetricLogError("CSV cannot encode nonfinite metrics")
    return format(value, ".17g")


def metrics_csv_bytes(records: tuple[MetricRecord, ...]) -> bytes:
    """Render the deterministic audit projection without filesystem writes."""

    if (
        type(records) is not tuple
        or not records
        or any(type(record) is not MetricRecord for record in records)
    ):
        raise MetricLogError(
            "CSV projection requires a nonempty exact metric tuple"
        )
    expected_previous = _ZERO_SHA256
    for ordinal, record in enumerate(records):
        record.__post_init__()
        if (
            record.ordinal != ordinal
            or record.previous_record_sha256 != expected_previous
        ):
            raise MetricLogError(
                "CSV projection metric chain is inconsistent"
            )
        expected_previous = record.record_sha256
    text = io.StringIO(newline="")
    writer = csv.writer(text, lineterminator="\n")
    writer.writerow(_CSV_COLUMNS)
    for record in records:
        for value in record.values:
            writer.writerow(
                (
                    str(record.ordinal),
                    record.utc_timestamp,
                    str(record.monotonic_ns),
                    record.run_id,
                    record.arm_id,
                    str(record.seed_id),
                    record.phase,
                    record.split,
                    str(record.step),
                    str(record.pass_index),
                    value.name,
                    value.applicability,
                    value.reason,
                    _decimal(value.numerator),
                    "" if value.denominator is None else str(value.denominator),
                    _decimal(value.value),
                    value.units,
                    record.record_sha256,
                )
            )
    return text.getvalue().encode("utf-8")


def export_metrics_csv(
    *,
    log_path: Path,
    output_path: Path,
    durability_backend: MetricDurabilityBackend,
) -> bytes:
    records, incomplete = _decode_metric_log(log_path)
    if incomplete:
        raise MetricLogError(
            "cannot export a metric log with an incomplete final fragment"
        )
    if not callable(getattr(durability_backend, "publish_bytes", None)):
        raise MetricLogError("durability backend must expose publish_bytes")
    payload = metrics_csv_bytes(records)
    try:
        durability_backend.publish_bytes(output_path, payload)
    except Exception as exc:
        raise MetricLogError(f"CSV publication failed: {exc}") from exc
    try:
        observed = _read_regular_bytes(output_path)
    except (OSError, MetricLogError) as exc:
        raise MetricLogError(f"CSV cannot be reopened: {exc}") from exc
    if observed != payload:
        raise MetricLogError("CSV reopen bytes do not match publication")
    return payload


@dataclasses.dataclass(frozen=True, slots=True)
class UpdateControlRecord:
    schema_version: Literal["wt103-update-controls-v2"]
    learning_rate: float
    scheduler_ordinal: int
    scheduler_state_sha256: str
    amp_applicability: Literal["applicable", "not_applicable"]
    amp_scale: float | None
    amp_overflow: bool | None
    clipping_threshold: float
    gradient_norm_applicability: Literal["applicable", "not_applicable"]
    pre_clip_norm: float | None
    post_clip_norm: float | None
    pre_clip_inf_norm: float | None
    post_clip_inf_norm: float | None
    clipped: bool | None
    adamw_beta1: float
    adamw_beta2: float
    adamw_epsilon: float
    adamw_weight_decay: float
    adamw_amsgrad: bool
    adamw_maximize: bool
    adamw_capturable: bool
    adamw_differentiable: bool
    adamw_foreach: bool
    adamw_fused: bool
    control_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != "wt103-update-controls-v2":
            raise MetricLogError("unsupported update-control schema")
        for name in (
            "learning_rate",
            "clipping_threshold",
            "adamw_beta1",
            "adamw_beta2",
            "adamw_epsilon",
            "adamw_weight_decay",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise MetricLogError(f"{name} must be a finite plain float")
        if (
            self.learning_rate < 0.0
            or self.clipping_threshold <= 0.0
            or not 0.0 <= self.adamw_beta1 < 1.0
            or not 0.0 <= self.adamw_beta2 < 1.0
            or self.adamw_epsilon <= 0.0
            or self.adamw_weight_decay < 0.0
        ):
            raise MetricLogError("update-control numeric bounds are invalid")
        if type(self.scheduler_ordinal) is not int or self.scheduler_ordinal < 0:
            raise MetricLogError("scheduler_ordinal must be nonnegative")
        if not _valid_sha256(self.scheduler_state_sha256):
            raise MetricLogError("scheduler_state_sha256 is invalid")
        if self.amp_applicability == "applicable":
            if (
                type(self.amp_scale) is not float
                or not math.isfinite(self.amp_scale)
                or self.amp_scale <= 0.0
                or type(self.amp_overflow) is not bool
            ):
                raise MetricLogError("applicable AMP controls are incomplete")
        elif self.amp_applicability == "not_applicable":
            if self.amp_scale is not None or self.amp_overflow is not None:
                raise MetricLogError(
                    "not-applicable AMP controls cannot fabricate values"
                )
        else:
            raise MetricLogError("unknown AMP applicability")
        if self.gradient_norm_applicability == "applicable":
            if (
                type(self.pre_clip_norm) is not float
                or not math.isfinite(self.pre_clip_norm)
                or self.pre_clip_norm < 0.0
                or type(self.post_clip_norm) is not float
                or not math.isfinite(self.post_clip_norm)
                or self.post_clip_norm < 0.0
                or type(self.pre_clip_inf_norm) is not float
                or not math.isfinite(self.pre_clip_inf_norm)
                or self.pre_clip_inf_norm < 0.0
                or type(self.post_clip_inf_norm) is not float
                or not math.isfinite(self.post_clip_inf_norm)
                or self.post_clip_inf_norm < 0.0
                or type(self.clipped) is not bool
            ):
                raise MetricLogError(
                    "applicable gradient-norm controls are incomplete"
                )
            if self.clipped != (
                self.pre_clip_norm > self.clipping_threshold
            ):
                raise MetricLogError(
                    "clipped must be derived from the pre-clip norm"
                )
        elif self.gradient_norm_applicability == "not_applicable":
            if any(
                value is not None
                for value in (
                    self.pre_clip_norm,
                    self.post_clip_norm,
                    self.pre_clip_inf_norm,
                    self.post_clip_inf_norm,
                    self.clipped,
                )
            ):
                raise MetricLogError(
                    "not-applicable gradient norms cannot fabricate values"
                )
        else:
            raise MetricLogError("unknown gradient-norm applicability")
        for name in (
            "adamw_amsgrad",
            "adamw_maximize",
            "adamw_capturable",
            "adamw_differentiable",
            "adamw_foreach",
            "adamw_fused",
        ):
            if type(getattr(self, name)) is not bool:
                raise MetricLogError(f"{name} must be exact bool")
        expected = owned_sha256(
            "vfe4.wt103.update-controls.v1",
            {
                field.name: getattr(self, field.name)
                for field in dataclasses.fields(self)
                if field.name != "control_sha256"
            },
        )
        if self.control_sha256 != expected:
            raise MetricLogError("control_sha256 does not match controls")

    @classmethod
    def create(cls, **values: object) -> "UpdateControlRecord":
        payload = {
            "schema_version": "wt103-update-controls-v2",
            **values,
        }
        return cls(
            **payload,
            control_sha256=owned_sha256(
                "vfe4.wt103.update-controls.v1",
                payload,
            ),
        )  # type: ignore[arg-type]


__all__ = [
    "MetricDurabilityBackend",
    "MetricLogError",
    "UpdateControlRecord",
    "WT103_METRIC_SEMANTIC_BY_NAME",
    "WT103_METRIC_UNIT_BY_NAME",
    "WT103_REQUIRED_METRIC_FAMILIES",
    "WT103_SOURCE_KL_DIAGNOSTIC_REASON",
    "WT103_UNAVAILABLE_ESTIMATOR_BOUND_REASON",
    "append_metric",
    "applicable_metric",
    "create_metric_record",
    "export_metrics_csv",
    "metrics_csv_bytes",
    "metric_family_units",
    "not_applicable_metric",
    "metric_family_applicability",
    "recover_incomplete_metric_fragment",
    "source_entropy_metrics",
    "validate_required_metric_families",
    "validate_metric_log",
]
