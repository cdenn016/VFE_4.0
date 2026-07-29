from __future__ import annotations

import dataclasses
import math
import os
from pathlib import Path

import pytest


class _Backend:
    def __init__(self) -> None:
        self.writes: list[tuple[Path, bytes]] = []

    def publish_bytes(self, path: Path, payload: bytes) -> object:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        self.writes.append((path, payload))
        return object()


def _record(
    *,
    ordinal: int,
    previous: str,
    values: tuple[object, ...],
    arm_id: str = "WT103-A0-AR-v1",
):
    from vfe4.recording.metrics import create_metric_record

    return create_metric_record(
        ordinal=ordinal,
        utc_timestamp=(
            f"2026-07-28T12:00:{ordinal:02d}.000000Z"
        ),
        monotonic_ns=1_000_000 + ordinal,
        run_id="run-a0-1",
        arm_id=arm_id,
        seed_id=2026072101,
        phase="model_ce_adam_proposal",
        split="train",
        step=ordinal,
        pass_index=0,
        previous_record_sha256=previous,
        values=values,
    )


_RATIO_METRIC_FAMILIES = frozenset(
    {
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
        "emission_only_non_elbo",
        "prior_nll_per_token",
        "source_entropy",
        "source_support_size",
        "acceptance_rate",
        "objective_before",
        "objective_after",
    }
)
_EXP_RATIO_METRIC_FAMILIES = frozenset(
    {"perplexity", "effective_source_count"}
)
_TOKENS_PER_SECOND_METRIC_FAMILIES = frozenset({"tokens_per_second"})
_NOT_APPLICABLE_ONLY_METRIC_FAMILIES = frozenset(
    {"estimator_error_bound"}
)
_SCALAR_METRIC_FAMILIES = frozenset(
    {
        "prior_nll_sum",
        "estimator_stream",
        "particle_count",
        "cache_audit_passed",
        "accepted_proposals",
        "rejected_proposals",
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
    }
)
_LATENT_SOURCE_METRIC_FAMILIES = frozenset(
    {"source_entropy", "source_support_size", "effective_source_count"}
)


def _valid_required_metric(name: str):
    from vfe4.recording.metrics import (
        WT103_REQUIRED_METRIC_FAMILIES,
        WT103_SOURCE_KL_DIAGNOSTIC_REASON,
        WT103_UNAVAILABLE_ESTIMATOR_BOUND_REASON,
        applicable_metric,
        metric_family_units,
        not_applicable_metric,
    )

    semantic_inventory = (
        _RATIO_METRIC_FAMILIES
        | _EXP_RATIO_METRIC_FAMILIES
        | _TOKENS_PER_SECOND_METRIC_FAMILIES
        | _NOT_APPLICABLE_ONLY_METRIC_FAMILIES
        | _SCALAR_METRIC_FAMILIES
    )
    assert semantic_inventory == frozenset(WT103_REQUIRED_METRIC_FAMILIES)
    if name in _NOT_APPLICABLE_ONLY_METRIC_FAMILIES:
        return not_applicable_metric(
            name=name,
            reason=WT103_UNAVAILABLE_ESTIMATOR_BOUND_REASON,
            units=metric_family_units(name),
        )
    if name in _RATIO_METRIC_FAMILIES:
        numerator = 6.0
        denominator = 3
        value = numerator / denominator
    elif name in _EXP_RATIO_METRIC_FAMILIES:
        numerator = math.log(4.0) * 3.0
        denominator = 3
        value = math.exp(numerator / denominator)
    elif name in _TOKENS_PER_SECOND_METRIC_FAMILIES:
        numerator = 30.0
        denominator = 2_000_000_000
        value = numerator / (denominator / 1_000_000_000.0)
    else:
        numerator = None
        denominator = None
        value = 2.0
    return applicable_metric(
        name=name,
        numerator=numerator,
        denominator=denominator,
        value=value,
        units=metric_family_units(name),
        reason=(
            WT103_SOURCE_KL_DIAGNOSTIC_REASON
            if name in {"model_source_kl", "state_source_kl"}
            else "measured"
        ),
    )


def _required_values_for_arm(
    arm: object,
    *,
    zero_source_rows: bool = False,
) -> tuple[object, ...]:
    from vfe4.recording.metrics import (
        WT103_REQUIRED_METRIC_FAMILIES,
        metric_family_applicability,
        metric_family_units,
        not_applicable_metric,
    )

    values = []
    for name in WT103_REQUIRED_METRIC_FAMILIES:
        applicable, reason = metric_family_applicability(arm, name)
        if (
            applicable
            and zero_source_rows
            and name in _LATENT_SOURCE_METRIC_FAMILIES
        ):
            values.append(
                not_applicable_metric(
                    name=name,
                    reason="source_row_count_is_zero",
                    units=metric_family_units(name),
                )
            )
        elif applicable:
            values.append(_valid_required_metric(name))
        else:
            values.append(
                not_applicable_metric(
                    name=name,
                    reason=reason,
                    units=metric_family_units(name),
                )
            )
    return tuple(values)


def test_every_required_metric_family_has_closed_exact_semantics() -> None:
    from vfe4.recording.metrics import (
        WT103_REQUIRED_METRIC_FAMILIES,
        WT103_SOURCE_KL_DIAGNOSTIC_REASON,
        applicable_metric,
        metric_family_units,
    )
    from vfe4.types.training import MetricValue

    for name in WT103_REQUIRED_METRIC_FAMILIES:
        assert _valid_required_metric(name).name == name

    retired_external_names = (
        "initial_model_kl",
        "initial_state_kl",
        "model_transition_kl",
        "state_transition_kl",
        "joint_recognition_entropy",
    )
    for name in retired_external_names:
        assert name not in WT103_REQUIRED_METRIC_FAMILIES
        with pytest.raises(ValueError, match="unknown metric family"):
            metric_family_units(name)

    derived_names = (
        _RATIO_METRIC_FAMILIES
        | _EXP_RATIO_METRIC_FAMILIES
        | _TOKENS_PER_SECOND_METRIC_FAMILIES
    )
    for name in derived_names:
        reason = (
            WT103_SOURCE_KL_DIAGNOSTIC_REASON
            if name in {"model_source_kl", "state_source_kl"}
            else "measured"
        )
        with pytest.raises(ValueError, match="raw numerator.*denominator"):
            applicable_metric(
                name=name,
                numerator=None,
                denominator=None,
                value=2.0,
                units=metric_family_units(name),
                reason=reason,
            )
        valid = _valid_required_metric(name)
        assert valid.value is not None
        with pytest.raises(ValueError, match="exact derivation"):
            applicable_metric(
                name=name,
                numerator=valid.numerator,
                denominator=valid.denominator,
                value=valid.value + 0.25,
                units=valid.units,
                reason=reason,
            )

    for name in _SCALAR_METRIC_FAMILIES:
        with pytest.raises(ValueError, match="forbids raw"):
            applicable_metric(
                name=name,
                numerator=2.0,
                denominator=1,
                value=2.0,
                units=metric_family_units(name),
            )

    with pytest.raises(ValueError, match="not-applicable-only"):
        applicable_metric(
            name="estimator_error_bound",
            numerator=0.0,
            denominator=1,
            value=0.0,
            units=metric_family_units("estimator_error_bound"),
        )
    with pytest.raises(ValueError, match="diagnostic-only marker"):
        applicable_metric(
            name="model_source_kl",
            numerator=0.0,
            denominator=1,
            value=0.0,
            units=metric_family_units("model_source_kl"),
            reason="train_objective_projection",
        )

    bypass = MetricValue(
        name="prior_nll_per_token",
        applicability="applicable",
        reason="direct construction",
        numerator=6.0,
        denominator=3,
        value=7.0,
        units="nats_per_token",
    )
    with pytest.raises(ValueError, match="exact derivation"):
        _record(
            ordinal=0,
            previous="0" * 64,
            values=(bypass,),
        )


def test_metric_jsonl_is_canonical_hash_chained_and_strictly_validated(
    tmp_path: Path,
) -> None:
    from vfe4.recording.metrics import (
        append_metric,
        applicable_metric,
        validate_metric_log,
    )

    path = tmp_path / "metrics.jsonl"
    backend = _Backend()
    first = _record(
        ordinal=0,
        previous="0" * 64,
        values=(
            applicable_metric(
                name="train_cross_entropy",
                numerator=3.0,
                denominator=2,
                value=1.5,
                units="nats_per_token",
            ),
        ),
    )
    second = _record(
        ordinal=1,
        previous=first.record_sha256,
        values=(
            applicable_metric(
                name="learning_rate",
                numerator=None,
                denominator=None,
                value=1.0e-4,
                units="scalar",
            ),
        ),
    )
    append_metric(path, first, durability_backend=backend)
    append_metric(path, second, durability_backend=backend)

    assert validate_metric_log(path) == (first, second)
    payload = path.read_bytes()
    assert payload.endswith(b"\n")
    assert b" " not in payload

    corrupted = payload.replace(b'"ordinal":1', b'"ordinal":2')
    path.write_bytes(corrupted)
    with pytest.raises(ValueError, match="canonical|ordinal|hash|record"):
        validate_metric_log(path)


def test_incomplete_final_fragment_is_recoverable_but_never_appended_over(
    tmp_path: Path,
) -> None:
    from vfe4.recording.metrics import (
        append_metric,
        applicable_metric,
        recover_incomplete_metric_fragment,
        validate_metric_log,
    )

    path = tmp_path / "metrics.jsonl"
    backend = _Backend()
    first = _record(
        ordinal=0,
        previous="0" * 64,
        values=(
            applicable_metric(
                name="loss",
                numerator=1.0,
                denominator=1,
                value=1.0,
                units="nats",
            ),
        ),
    )
    append_metric(path, first, durability_backend=backend)
    with path.open("ab") as handle:
        handle.write(b'{"incomplete":')
    assert validate_metric_log(path) == (first,)

    second = _record(
        ordinal=1,
        previous=first.record_sha256,
        values=first.values,
    )
    with pytest.raises(ValueError, match="incomplete"):
        append_metric(path, second, durability_backend=backend)
    recover_incomplete_metric_fragment(
        path,
        durability_backend=backend,
    )
    append_metric(path, second, durability_backend=backend)
    assert validate_metric_log(path) == (first, second)


def test_complete_record_without_newline_is_not_misclassified_as_truncation(
    tmp_path: Path,
) -> None:
    from vfe4.recording.metrics import (
        applicable_metric,
        validate_metric_log,
    )
    from vfe4.types.training import canonical_json_bytes

    record = _record(
        ordinal=0,
        previous="0" * 64,
        values=(
            applicable_metric(
                name="loss",
                numerator=1.0,
                denominator=1,
                value=1.0,
                units="nats",
            ),
        ),
    )
    path = tmp_path / "metrics.jsonl"
    path.write_bytes(canonical_json_bytes(record))
    with pytest.raises(ValueError, match="complete record"):
        validate_metric_log(path)


def test_csv_export_is_deterministic_and_round_trip_decimal_exact(
    tmp_path: Path,
) -> None:
    from vfe4.recording.metrics import (
        append_metric,
        applicable_metric,
        export_metrics_csv,
    )

    log_path = tmp_path / "metrics.jsonl"
    record = _record(
        ordinal=0,
        previous="0" * 64,
        values=(
            applicable_metric(
                name="learning_rate",
                numerator=None,
                denominator=None,
                value=0.00030000000000000003,
                units="scalar",
            ),
        ),
    )
    backend = _Backend()
    append_metric(log_path, record, durability_backend=backend)
    output = tmp_path / "metrics.csv"
    first = export_metrics_csv(
        log_path=log_path,
        output_path=output,
        durability_backend=backend,
    )
    second = export_metrics_csv(
        log_path=log_path,
        output_path=output,
        durability_backend=backend,
    )

    assert first == second == output.read_bytes()
    assert b"0.00030000000000000003" in first
    assert first.count(b"\n") == 2


def test_effective_source_count_and_not_applicable_values_are_exact() -> None:
    from vfe4.recording.metrics import (
        metric_family_units,
        not_applicable_metric,
        source_entropy_metrics,
    )

    values = source_entropy_metrics(
        entropy_sum=math.log(4.0) * 3.0,
        source_row_count=3,
    )
    assert values[0].value == pytest.approx(math.log(4.0))
    assert values[1].value == pytest.approx(4.0)
    empty = source_entropy_metrics(
        entropy_sum=0.0,
        source_row_count=0,
    )
    assert all(
        value.applicability == "not_applicable"
        and value.value is None
        for value in empty
    )

    absent = not_applicable_metric(
        name="source_entropy",
        reason="arm_has_no_source_mixture",
        units=metric_family_units("source_entropy"),
    )
    assert absent.numerator is None
    assert absent.denominator is None
    assert absent.value is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        absent.value = 0.0  # type: ignore[misc]


def test_update_control_metrics_capture_every_effective_adamw_amp_and_clip_field() -> None:
    from vfe4.recording.metrics import UpdateControlRecord

    record = UpdateControlRecord.create(
        learning_rate=3.0e-4,
        scheduler_ordinal=7,
        scheduler_state_sha256="1" * 64,
        amp_applicability="applicable",
        amp_scale=65536.0,
        amp_overflow=False,
        clipping_threshold=1.0,
        gradient_norm_applicability="applicable",
        pre_clip_norm=2.0,
        post_clip_norm=1.0,
        pre_clip_inf_norm=1.5,
        post_clip_inf_norm=0.75,
        clipped=True,
        adamw_beta1=0.9,
        adamw_beta2=0.95,
        adamw_epsilon=1.0e-8,
        adamw_weight_decay=0.01,
        adamw_amsgrad=False,
        adamw_maximize=False,
        adamw_capturable=False,
        adamw_differentiable=False,
        adamw_foreach=False,
        adamw_fused=True,
    )

    assert record.scheduler_ordinal == 7
    assert record.amp_overflow is False
    assert record.clipped is True
    assert record.pre_clip_inf_norm == 1.5
    assert record.post_clip_inf_norm == 0.75
    assert record.adamw_foreach is False
    assert len(record.control_sha256) == 64


def test_failure_ledger_survives_an_independent_metric_fsync_failure(
    tmp_path: Path,
) -> None:
    from vfe4.recording.failures import (
        append_failure,
        create_failure_record,
        validate_failure_log,
    )
    from vfe4.recording.metrics import (
        append_metric,
        applicable_metric,
    )

    metric = _record(
        ordinal=0,
        previous="0" * 64,
        values=(
            applicable_metric(
                name="loss",
                numerator=1.0,
                denominator=1,
                value=1.0,
                units="nats",
            ),
        ),
    )

    class _FailingBackend:
        def publish_bytes(self, _path: Path, _payload: bytes) -> object:
            raise OSError("injected metric publication failure")

    with pytest.raises(ValueError, match="injected"):
        append_metric(
            tmp_path / "metrics.jsonl",
            metric,
            durability_backend=_FailingBackend(),
        )

    failure = create_failure_record(
        ordinal=0,
        utc_timestamp="2026-07-28T12:00:00.000000Z",
        monotonic_ns=2_000_000,
        run_id="run-a0-1",
        arm_id="WT103-A0-AR-v1",
        seed_id=2026072101,
        phase="metric_append",
        step=0,
        pass_index=0,
        cursor_sha256="2" * 64,
        checkpoint_identity_sha256=None,
        retry_classification="not_retryable",
        scientific_state_advanced=False,
        terminal_disposition="failed",
        exception=OSError("injected metric publication failure"),
        previous_record_sha256="0" * 64,
    )
    failure_path = tmp_path / "failures.jsonl"
    append_failure(
        failure_path,
        failure,
        durability_backend=_Backend(),
    )
    assert validate_failure_log(failure_path) == (failure,)


def test_metric_append_uses_real_probed_platform_durability_backend(
    tmp_path: Path,
) -> None:
    from vfe4.artifacts.durability import (
        PosixDurabilityBackend,
        WindowsDurabilityBackend,
    )
    from vfe4.recording.metrics import append_metric, applicable_metric

    backend = (
        WindowsDurabilityBackend()
        if os.name == "nt"
        else PosixDurabilityBackend()
    )
    probe = backend.probe(tmp_path)
    assert probe.status == "pass"
    record = _record(
        ordinal=0,
        previous="0" * 64,
        values=(
            applicable_metric(
                name="loss",
                numerator=1.0,
                denominator=1,
                value=1.0,
                units="nats",
            ),
        ),
    )
    identity = append_metric(
        tmp_path / "metrics.jsonl",
        record,
        durability_backend=backend,
    )
    assert identity.reopen_verified is True


def test_finalized_metric_families_require_explicit_arm_applicability() -> None:
    from vfe4.recording.metrics import (
        metric_family_units,
        not_applicable_metric,
        validate_required_metric_families,
    )
    from vfe4.types.training import default_wt103_arm_specs

    arm = default_wt103_arm_specs()[0]
    complete = _record(
        ordinal=0,
        previous="0" * 64,
        values=_required_values_for_arm(arm),
    )
    validate_required_metric_families((complete,), arm_spec=arm)

    midpoint = len(complete.values) // 2
    first = _record(
        ordinal=0,
        previous="0" * 64,
        values=complete.values[:midpoint],
    )
    second = _record(
        ordinal=1,
        previous=first.record_sha256,
        values=complete.values[midpoint:],
    )
    validate_required_metric_families((first, second), arm_spec=arm)

    incomplete = _record(
        ordinal=0,
        previous="0" * 64,
        values=complete.values[:-1],
    )
    with pytest.raises(ValueError, match="missing"):
        validate_required_metric_families((incomplete,), arm_spec=arm)

    wrong_nonlatent_reason = tuple(
        (
            not_applicable_metric(
                name=value.name,
                reason="source_row_count_is_zero",
                units=metric_family_units(value.name),
            )
            if value.name in _LATENT_SOURCE_METRIC_FAMILIES
            else value
        )
        for value in complete.values
    )
    wrong_nonlatent = _record(
        ordinal=0,
        previous="0" * 64,
        values=wrong_nonlatent_reason,
    )
    with pytest.raises(ValueError, match="noncanonical N/A reason"):
        validate_required_metric_families(
            (wrong_nonlatent,),
            arm_spec=arm,
        )


def test_finalized_latent_source_metrics_distinguish_zero_and_positive_rows() -> None:
    from vfe4.recording.metrics import (
        metric_family_units,
        not_applicable_metric,
        validate_required_metric_families,
    )
    from vfe4.types.training import default_wt103_arm_specs

    arm = next(
        candidate
        for candidate in default_wt103_arm_specs()
        if candidate.latent_enabled
    )
    zero = _record(
        ordinal=0,
        previous="0" * 64,
        arm_id=arm.arm_id,
        values=_required_values_for_arm(arm, zero_source_rows=True),
    )
    validate_required_metric_families((zero,), arm_spec=arm)

    positive_source = _record(
        ordinal=1,
        previous=zero.record_sha256,
        arm_id=arm.arm_id,
        values=tuple(
            _valid_required_metric(name)
            for name in _LATENT_SOURCE_METRIC_FAMILIES
        ),
    )
    validate_required_metric_families(
        (zero, positive_source),
        arm_spec=arm,
    )

    fabricated_values = tuple(
        (
            not_applicable_metric(
                name=value.name,
                reason="source_rows_not_seen",
                units=metric_family_units(value.name),
            )
            if value.name in _LATENT_SOURCE_METRIC_FAMILIES
            else value
        )
        for value in zero.values
    )
    fabricated = _record(
        ordinal=0,
        previous="0" * 64,
        arm_id=arm.arm_id,
        values=fabricated_values,
    )
    with pytest.raises(ValueError, match="canonical zero-row N/A"):
        validate_required_metric_families((fabricated,), arm_spec=arm)

    mixed_values = tuple(
        (
            not_applicable_metric(
                name=value.name,
                reason="source_row_count_is_zero",
                units=metric_family_units(value.name),
            )
            if value.name == "effective_source_count"
            else value
        )
        for value in _required_values_for_arm(arm)
    )
    mixed = _record(
        ordinal=0,
        previous="0" * 64,
        arm_id=arm.arm_id,
        values=mixed_values,
    )
    with pytest.raises(ValueError, match="mixes zero and positive source rows"):
        validate_required_metric_families((mixed,), arm_spec=arm)

    entropy_only = _record(
        ordinal=1,
        previous=zero.record_sha256,
        arm_id=arm.arm_id,
        values=(_valid_required_metric("source_entropy"),),
    )
    with pytest.raises(ValueError, match="positive source rows require"):
        validate_required_metric_families(
            (zero, entropy_only),
            arm_spec=arm,
        )


def test_complete_elbo_bound_is_canonically_unavailable_only() -> None:
    from vfe4.recording.metrics import (
        metric_family_units,
        not_applicable_metric,
        validate_required_metric_families,
    )
    from vfe4.types.training import default_wt103_arm_specs

    arm = next(
        candidate
        for candidate in default_wt103_arm_specs()
        if candidate.training_objective == "complete_elbo"
    )
    unbounded = _record(
        ordinal=0,
        previous="0" * 64,
        arm_id=arm.arm_id,
        values=_required_values_for_arm(arm),
    )
    validate_required_metric_families((unbounded,), arm_spec=arm)

    fabricated_values = tuple(
        (
            not_applicable_metric(
                name=value.name,
                reason="single_sample_mc",
                units=metric_family_units(value.name),
            )
            if value.name == "estimator_error_bound"
            else value
        )
        for value in unbounded.values
    )
    fabricated = _record(
        ordinal=0,
        previous="0" * 64,
        arm_id=arm.arm_id,
        values=fabricated_values,
    )
    with pytest.raises(ValueError, match="canonical unbounded-estimator"):
        validate_required_metric_families((fabricated,), arm_spec=arm)

    from vfe4.recording.metrics import applicable_metric

    with pytest.raises(ValueError, match="not-applicable-only"):
        applicable_metric(
            name="estimator_error_bound",
            numerator=0.0,
            denominator=1,
            value=0.0,
            units=metric_family_units("estimator_error_bound"),
        )
