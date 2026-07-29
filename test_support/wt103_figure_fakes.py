"""Small finalized in-memory figure records for focused renderer tests."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

from vfe4.artifacts.durability import DurableFileIdentity
from vfe4.artifacts.manifest import ArtifactIntegrityRecord
from vfe4.artifacts.run_directory import (
    ExperimentPlan,
    finalize_run,
    publish_experiment_index,
    publish_experiment_plan,
    reserve_run,
)
from vfe4.checkpoint.schema import make_checkpoint_identity
from vfe4.config import (
    default_training_config_mapping,
    resolve_training_config,
)
from vfe4.figures.load import (
    FigureApplicabilityRow,
    FigurePoint,
    FinalResultTable,
    LoadedFigureInputs,
    MetricSource,
)
from vfe4.recording.metrics import (
    WT103_METRIC_SEMANTIC_BY_NAME,
    WT103_REQUIRED_METRIC_FAMILIES,
    WT103_SOURCE_KL_DIAGNOSTIC_REASON,
    WT103_UNAVAILABLE_ESTIMATOR_BOUND_REASON,
    applicable_metric,
    create_metric_record,
    metric_family_applicability,
    metric_family_units,
    metrics_csv_bytes,
    not_applicable_metric,
)
from vfe4.types.figures import (
    FigureExperimentIndexIdentity,
    FigureInputIdentity,
    FigureResultRow,
    WT103_FIGURE_PROVENANCE,
    default_figure_specs,
    figure_panel_applicability,
    figure_series_metric_names,
)
from vfe4.types.training import MetricRecord, canonical_json_bytes, owned_sha256


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _metric_units(name: str) -> str:
    return metric_family_units(name)


def _metric_scalar(name: str, arm_index: int, seed_index: int) -> float:
    if name == "counted_targets":
        return 10_000.0
    if name.endswith("_bytes"):
        return float(1_000_000 + 10_000 * arm_index + 100 * seed_index)
    if name == "tokens_per_second":
        return float(2_000 + 10 * arm_index + seed_index)
    if name == "acceptance_rate":
        return 0.75
    return float(1.0 + 0.02 * arm_index + 0.0001 * seed_index)


def _final_metric_record(
    *,
    run_id: str,
    arm,
    seed_id: int,
    arm_index: int,
    seed_index: int,
):
    values = []
    for name in WT103_REQUIRED_METRIC_FAMILIES:
        applicable, reason = metric_family_applicability(arm, name)
        if name == "estimator_error_bound" and applicable:
            values.append(
                not_applicable_metric(
                    name=name,
                    reason=WT103_UNAVAILABLE_ESTIMATOR_BOUND_REASON,
                    units=_metric_units(name),
                )
            )
            continue
        if applicable:
            if name in {"model_source_kl", "state_source_kl"}:
                reason = WT103_SOURCE_KL_DIAGNOSTIC_REASON
            value = _metric_scalar(name, arm_index, seed_index)
            semantic = WT103_METRIC_SEMANTIC_BY_NAME[name]
            if semantic == "scalar":
                numerator = None
                denominator = None
            elif semantic == "ratio":
                denominator = 4
                numerator = value * denominator
            elif semantic == "exp_ratio":
                denominator = 4
                numerator = math.log(value) * denominator
            elif semantic == "tokens_per_second":
                denominator = 1_000_000_000
                numerator = value
            else:  # pragma: no cover - guarded by the closed metric schema
                raise AssertionError(f"unknown metric semantic {semantic!r}")
            values.append(
                applicable_metric(
                    name=name,
                    numerator=numerator,
                    denominator=denominator,
                    value=value,
                    units=_metric_units(name),
                    reason=reason,
                )
            )
        else:
            values.append(
                not_applicable_metric(
                    name=name,
                    reason=reason,
                    units=_metric_units(name),
                )
            )
    return create_metric_record(
        ordinal=0,
        utc_timestamp="2026-07-28T00:00:00.000000Z",
        monotonic_ns=0,
        run_id=run_id,
        arm_id=arm.arm_id,
        seed_id=seed_id,
        phase="terminal_validation",
        split="validation",
        step=1,
        pass_index=1,
        previous_record_sha256="0" * 64,
        values=tuple(values),
    )


def _figure_points(inventory, records_by_arm_seed) -> tuple[FigurePoint, ...]:
    arms = {arm.arm_id: arm for arm in inventory.arms}
    points: list[FigurePoint] = []
    for series_key in inventory.figure_series_keys:
        arm_id = series_key.split("/")[1]
        arm = arms[arm_id]
        if (
            figure_panel_applicability(
                series_key.split("/")[0],
                arm,
            )[0]
            == "not_applicable"
        ):
            continue
        for metric_name in sorted(
            figure_series_metric_names(series_key, arm)
        ):
            for seed_id in inventory.confirmatory_seed_ids:
                record = records_by_arm_seed[(arm_id, seed_id)]
                metric = next(
                    value
                    for value in record.values
                    if value.name == metric_name
                )
                if metric.applicability == "not_applicable":
                    if metric_name != "estimator_error_bound":
                        raise AssertionError(
                            "fixture series metric must be applicable"
                        )
                    continue
                assert metric.value is not None
                points.append(
                    FigurePoint(
                        series_key=series_key,
                        arm_id=arm_id,
                        seed_id=seed_id,
                        source_record_sha256=record.record_sha256,
                        source_phase=record.phase,
                        source_split=record.split,
                        source_step=record.step,
                        source_pass_index=record.pass_index,
                        counted_targets=10_000,
                        metric_name=metric_name,
                        applicability="applicable",
                        applicability_reason=metric.reason,
                        numerator=metric.numerator,
                        denominator=metric.denominator,
                        value=metric.value,
                        lower=None,
                        upper=None,
                        units=metric.units,
                        result_role=arm.result_role,
                    )
                )
    return tuple(points)


def _result_rows(inventory, records_by_arm_seed) -> tuple[FigureResultRow, ...]:
    rows = []
    for arm, result_row_key in zip(
        inventory.arms,
        inventory.result_row_keys,
        strict=True,
    ):
        seed_nll = tuple(
            next(
                value.value
                for value in records_by_arm_seed[(arm.arm_id, seed_id)].values
                if value.name == "prior_nll_per_token"
            )
            for seed_id in inventory.confirmatory_seed_ids
        )
        if any(type(value) is not float for value in seed_nll):
            raise AssertionError("fixture prior NLL must be applicable")
        typed_nll = tuple(float(value) for value in seed_nll)
        mean_nll = math.fsum(typed_nll) / len(typed_nll)
        if arm.result_role in ("PRIMARY_REFERENCE", "PRIMARY_ENDPOINT"):
            applicability = "decision_bearing"
            reason = "a0_minus_parent_specific_complete_primary_pair"
        elif arm.result_role == "OBJECTIVE_GATE":
            applicability = "decision_bearing"
            reason = "complete_minus_emission_objective_gate"
        else:
            applicability = "descriptive_only"
            reason = "control_is_reported_but_cannot_rescue_or_reverse_primary"
        rows.append(
            FigureResultRow.create(
                result_row_key=result_row_key,
                arm_id=arm.arm_id,
                scorer_kind=arm.scorer_kind,
                applicability=applicability,
                applicability_reason=reason,
                result_role=arm.result_role,
                seed_nll_per_token=typed_nll,
                seed_perplexity=tuple(math.exp(value) for value in typed_nll),
                mean_nll_per_token=mean_nll,
                mean_perplexity=math.exp(mean_nll),
                status="pass",
            )
        )
    return tuple(rows)


def finalized_figure_inputs() -> LoadedFigureInputs:
    inventory = WT103_FIGURE_PROVENANCE.endpoint_inventory
    arms = {arm.arm_id: arm for arm in inventory.arms}
    metrics_hashes = tuple(
        _sha(f"metrics-jsonl-{key}")
        for key in inventory.terminal_checkpoint_keys
    )
    csv_hashes = tuple(
        _sha(f"metrics-csv-{key}")
        for key in inventory.terminal_checkpoint_keys
    )
    sources = tuple(
        MetricSource(
            terminal_checkpoint_key=key,
            run_id=f"run-{index:02d}",
            arm_id=key.split("/")[1],
            seed_id=int(key.rsplit("=", 1)[-1]),
            checkpoint_role="terminal_scoring",
            run_manifest_path=(
                f"runs/run-{index:02d}/run-manifest.json"
            ),
            metrics_jsonl_path=f"runs/run-{index:02d}/metrics.jsonl",
            metrics_csv_path=f"runs/run-{index:02d}/metrics.csv",
            metrics_jsonl_sha256=metrics_hashes[index],
            metrics_csv_sha256=csv_hashes[index],
        )
        for index, key in enumerate(inventory.terminal_checkpoint_keys)
    )
    applicability = tuple(
        FigureApplicabilityRow(
            panel_key=panel,
            arm_id=arm.arm_id,
            applicability=figure_panel_applicability(panel, arm)[0],
            reason=figure_panel_applicability(panel, arm)[1],
        )
        for panel in inventory.figure_panel_keys
        for arm in inventory.arms
    )
    records_by_arm_seed = {}
    metric_records = []
    arm_order = {
        arm.arm_id: index for index, arm in enumerate(inventory.arms)
    }
    seed_order = {
        seed_id: index
        for index, seed_id in enumerate(inventory.confirmatory_seed_ids)
    }
    for source in sources:
        arm = arms[source.arm_id]
        record = _final_metric_record(
            run_id=source.run_id,
            arm=arm,
            seed_id=source.seed_id,
            arm_index=arm_order[source.arm_id],
            seed_index=seed_order[source.seed_id],
        )
        records_by_arm_seed[(source.arm_id, source.seed_id)] = record
        metric_records.append((source, (record,)))
    points = _figure_points(inventory, records_by_arm_seed)
    result_rows = _result_rows(inventory, records_by_arm_seed)
    table = FinalResultTable.create(
        endpoint_inventory_sha256=inventory.endpoint_inventory_sha256,
        metrics_jsonl_sha256s=metrics_hashes,
        aggregation_sha256=_sha("estimator-aggregation"),
        result_rows=result_rows,
        figure_panel_keys=inventory.figure_panel_keys,
        figure_series_keys=inventory.figure_series_keys,
        applicability_rows=applicability,
        points=points,
    )
    index_semantic = {
        "stage": "final",
        "experiment_plan_sha256": _sha("experiment-plan"),
        "run_manifest_sha256s": tuple(
            _sha(f"run-manifest-{index}") for index in range(len(sources))
        ),
        "artifact_record_sha256s": (_sha("result-table-record"),),
        "payload_sha256": _sha("experiment-index"),
        "size_bytes": 1,
    }
    index_identity = FigureExperimentIndexIdentity(
        index_path=Path("experiment-index.json"),
        **index_semantic,
        identity_sha256=owned_sha256(
            "vfe4.wt103.experiment-index-identity.v1",
            index_semantic,
        ),
    )
    result_table_file_sha256 = hashlib.sha256(
        canonical_json_bytes(table)
    ).hexdigest()
    identity = FigureInputIdentity.create(
        endpoint_inventory_sha256=inventory.endpoint_inventory_sha256,
        run_group_manifest_sha256=index_identity.payload_sha256,
        metrics_jsonl_sha256s=metrics_hashes,
        result_table_sha256=result_table_file_sha256,
        regenerated_csv_sha256s=csv_hashes,
    )
    return LoadedFigureInputs(
        experiment_index_identity=index_identity,
        metric_sources=sources,
        metric_records=tuple(metric_records),
        result_table=table,
        specs=default_figure_specs(inventory),
        identity=identity,
    )


class FilesystemFigureBackend:
    volume_identity = "fixture-volume"

    def create_exclusive(
        self,
        path: Path,
        payload: bytes,
    ) -> DurableFileIdentity:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(payload)
        return DurableFileIdentity.create(
            operation="exclusive_create",
            payload=payload,
            volume_identity=self.volume_identity,
        )

    def replace_durable(
        self,
        path: Path,
        payload: bytes,
    ) -> DurableFileIdentity:
        if not path.is_file():
            raise FileNotFoundError(path)
        path.write_bytes(payload)
        return DurableFileIdentity.create(
            operation="replace",
            payload=payload,
            volume_identity=self.volume_identity,
        )

    def publish_bytes(
        self,
        path: Path,
        payload: bytes,
    ) -> DurableFileIdentity:
        if path.exists():
            return self.replace_durable(path, payload)
        return self.create_exclusive(path, payload)


def _integrity(
    relative_path: str,
    payload: bytes,
) -> ArtifactIntegrityRecord:
    return ArtifactIntegrityRecord.create(
        kind="file",
        relative_path=relative_path,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _experiment_plan() -> ExperimentPlan:
    inventory = resolve_training_config(
        default_training_config_mapping()
    ).endpoint_inventory
    return ExperimentPlan.create(
        experiment_id="wt103-figure-fixture",
        endpoint_inventory=inventory,
        git_head="1" * 40,
        dirty_digest=_sha("dirty"),
        config_sha256=_sha("config"),
        source_record_sha256=_sha("source"),
        tokenizer_spec_sha256=_sha("tokenizer"),
        token_cache_set_sha256=_sha("cache"),
        window_manifest_sha256s=(
            _sha("train-window"),
            _sha("validation-window"),
            _sha("test-window"),
        ),
        schedule_set_sha256=_sha("schedule"),
        factory_set_sha256=_sha("factory"),
        objective_sha256=_sha("objective"),
        checkpoint_schema_sha256=_sha("checkpoint"),
        resource_forecast_sha256=_sha("forecast"),
        expected_run_artifact_paths=("metrics.csv", "metrics.jsonl"),
        expected_group_artifact_paths=("result-table.json",),
    )


def _metric_payloads(
    *,
    run_id: str,
    arm_id: str,
    seed_id: int,
) -> tuple[bytes, bytes, MetricRecord]:
    inventory = resolve_training_config(
        default_training_config_mapping()
    ).endpoint_inventory
    arm = next(item for item in inventory.arms if item.arm_id == arm_id)
    record = _final_metric_record(
        run_id=run_id,
        arm=arm,
        seed_id=seed_id,
        arm_index=next(
            index
            for index, candidate in enumerate(inventory.arms)
            if candidate.arm_id == arm_id
        ),
        seed_index=inventory.confirmatory_seed_ids.index(seed_id),
    )
    records = (record,)
    return (
        canonical_json_bytes(record) + b"\n",
        metrics_csv_bytes(records),
        record,
    )


def build_finalized_figure_experiment(root: Path) -> Path:
    """Publish a complete 40-confirmation fixture through the real lifecycle."""

    backend = FilesystemFigureBackend()
    plan = _experiment_plan()
    plan_identity = publish_experiment_plan(root, plan, backend=backend)
    inventory = resolve_training_config(
        default_training_config_mapping()
    ).endpoint_inventory
    manifests = []
    metrics_hashes = []
    records_by_arm_seed = {}
    for index, tuning_attempt_key in enumerate(plan.tuning_attempt_keys):
        run_id = f"tuning-{index:02d}"
        reserved = reserve_run(
            root,
            run_id,
            run_role="tuning",
            started_utc="2026-07-28T00:00:00Z",
            plan=plan_identity,
            backend=backend,
            tuning_attempt_key=tuning_attempt_key,
        )
        tuning_csv = b"fixture-tuning-metrics\n"
        tuning_jsonl = b'{"fixture":"tuning-metrics"}\n'
        for name, payload in (
            ("metrics.csv", tuning_csv),
            ("metrics.jsonl", tuning_jsonl),
        ):
            backend.create_exclusive(
                reserved.inprogress_path / name,
                payload,
            )
        manifests.append(
            finalize_run(
                reserved,
                disposition="success",
                checkpoints=(),
                checkpoint_artifact_records=(),
                artifact_records=(
                    _integrity("metrics.csv", tuning_csv),
                    _integrity("metrics.jsonl", tuning_jsonl),
                ),
                environment_sha256=_sha("environment"),
                provenance_sha256=_sha("provenance"),
                ended_utc="2026-07-28T00:00:01Z",
                monotonic_duration_seconds=1.0,
                failure_record_sha256=None,
                backend=backend,
            )
        )
    for index, logical_key in enumerate(
        inventory.terminal_checkpoint_keys
    ):
        prefix, seed_text = logical_key.rsplit("/seed=", 1)
        arm_id = prefix.removeprefix("terminal/")
        seed_id = int(seed_text)
        run_id = f"confirmation-{index:02d}"
        reserved = reserve_run(
            root,
            run_id,
            run_role="confirmation",
            started_utc="2026-07-28T00:00:00Z",
            plan=plan_identity,
            backend=backend,
        )
        metrics_jsonl, metrics_csv, record = _metric_payloads(
            run_id=run_id,
            arm_id=arm_id,
            seed_id=seed_id,
        )
        records_by_arm_seed[(arm_id, seed_id)] = record
        for name, payload in (
            ("metrics.csv", metrics_csv),
            ("metrics.jsonl", metrics_jsonl),
        ):
            backend.create_exclusive(
                reserved.inprogress_path / name,
                payload,
            )
        checkpoint_payload = (
            f"fixture-terminal-checkpoint-{index:02d}\n".encode("ascii")
        )
        backend.create_exclusive(
            reserved.inprogress_path / "terminal-scoring.pt",
            checkpoint_payload,
        )
        checkpoint = make_checkpoint_identity(
            logical_key=logical_key,
            checkpoint_role="terminal_scoring",
            scientific_state_sha256=_sha(f"state-{index}"),
            checkpoint_payload_sha256=hashlib.sha256(
                checkpoint_payload
            ).hexdigest(),
            checkpoint_manifest_body_sha256=_sha(
                f"manifest-{index}"
            ),
            size_bytes=len(checkpoint_payload),
        )
        manifests.append(
            finalize_run(
                reserved,
                disposition="success",
                checkpoints=(checkpoint,),
                checkpoint_artifact_records=(
                    _integrity(
                        "terminal-scoring.pt",
                        checkpoint_payload,
                    ),
                ),
                artifact_records=(
                    _integrity("metrics.csv", metrics_csv),
                    _integrity("metrics.jsonl", metrics_jsonl),
                ),
                environment_sha256=_sha("environment"),
                provenance_sha256=_sha("provenance"),
                ended_utc="2026-07-28T00:00:01Z",
                monotonic_duration_seconds=1.0,
                failure_record_sha256=None,
                backend=backend,
            )
        )
        metrics_hashes.append(hashlib.sha256(metrics_jsonl).hexdigest())
    template = finalized_figure_inputs().result_table
    table = FinalResultTable.create(
        endpoint_inventory_sha256=template.endpoint_inventory_sha256,
        metrics_jsonl_sha256s=tuple(metrics_hashes),
        aggregation_sha256=template.aggregation_sha256,
        result_rows=_result_rows(inventory, records_by_arm_seed),
        figure_panel_keys=template.figure_panel_keys,
        figure_series_keys=template.figure_series_keys,
        applicability_rows=template.applicability_rows,
        points=_figure_points(inventory, records_by_arm_seed),
    )
    table_payload = canonical_json_bytes(table)
    backend.create_exclusive(root / "result-table.json", table_payload)
    publish_experiment_index(
        root,
        plan=plan_identity,
        run_manifests=tuple(manifests),
        stage="final",
        artifact_records=(
            _integrity("result-table.json", table_payload),
        ),
        backend=backend,
    )
    return root / "experiment-index.json"


__all__ = [
    "FilesystemFigureBackend",
    "build_finalized_figure_experiment",
    "finalized_figure_inputs",
]
