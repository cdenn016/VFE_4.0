from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib
import json
from pathlib import Path

import pytest

from test_support.wt103_figure_fakes import (
    build_finalized_figure_experiment,
    finalized_figure_inputs,
)
from vfe4.config import (
    default_figure_config_mapping,
    default_training_config_mapping,
    resolve_figure_config,
    resolve_training_config,
)
from vfe4.figures import (
    FigureInputError,
    REQUIRED_FIGURE_FUNCTIONS,
    load_figure_inputs,
    validate_figure_registry,
)
from vfe4.figures.render import _publish_or_validate_set
from vfe4.artifacts.environment import (
    parse_lock_input_manifest,
    render_dependency_lock,
)


def test_figure_modules_are_import_safe_and_forbid_runtime_dependencies() -> None:
    root = Path(__file__).resolve().parents[2]
    package = root / "vfe4" / "figures"
    forbidden = {
        "torch",
        "tiktoken",
        "train_vfe4",
        "vfe4.artifacts.run_directory",
        "vfe4.checkpoint",
        "vfe4.training",
        "vfe4.data",
    }
    for path in sorted(package.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imports.add(node.module)
        assert not any(
            imported == name or imported.startswith(f"{name}.")
            for imported in imports
            for name in forbidden
        ), (path.name, sorted(imports))


def test_frozen_figure_registry_matches_every_inventory_panel_and_series() -> None:
    training = resolve_training_config(default_training_config_mapping())
    raw = default_figure_config_mapping(training.endpoint_inventory)
    resolved = resolve_figure_config(raw)

    assert tuple(spec.figure_id for spec in resolved.specs) == (
        training.endpoint_inventory.figure_panel_keys
    )
    assert tuple(name for name, _function in REQUIRED_FIGURE_FUNCTIONS) == (
        training.endpoint_inventory.figure_panel_keys
    )
    assert tuple(
        series.series_key
        for spec in resolved.specs
        for series in spec.series
    ) == training.endpoint_inventory.figure_series_keys
    assert validate_figure_registry(
        inventory=training.endpoint_inventory,
        specs=resolved.specs,
    ) == resolved.specs
    assert all(
        series.color.startswith("#") and len(series.color) == 7
        for spec in resolved.specs
        for series in spec.series
    )
    arm_styles: dict[str, set[tuple[str, str]]] = {}
    for spec in resolved.specs:
        for series in spec.series:
            arm_styles.setdefault(series.arm_id, set()).add(
                (series.color, series.marker)
            )
    assert all(len(styles) == 1 for styles in arm_styles.values())
    assert all(spec.alt_text and spec.caption for spec in resolved.specs)


def test_figure_config_is_bound_to_frozen_read_only_provenance() -> None:
    from vfe4.types.figures import (
        FrozenFigureProvenance,
        WT103_FIGURE_PROVENANCE,
    )

    provenance = WT103_FIGURE_PROVENANCE
    assert type(provenance) is FrozenFigureProvenance
    provenance.__post_init__()
    raw = default_figure_config_mapping(provenance.endpoint_inventory)
    resolved = resolve_figure_config(raw)
    assert raw["figure_provenance_sha256"] == provenance.provenance_sha256
    assert (
        resolved.figure_provenance_sha256
        == provenance.provenance_sha256
    )
    changed = dict(raw)
    changed["figure_provenance_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="figure_provenance_sha256"):
        resolve_figure_config(changed)

    launcher_source = (
        Path(__file__).resolve().parents[2] / "generate_vfe4_figures.py"
    ).read_text(encoding="utf-8")
    assert "default_training_config_mapping" not in launcher_source
    assert "resolve_training_config" not in launcher_source


def test_result_table_embeds_typed_rows_and_aggregation_identity() -> None:
    from vfe4.figures.load import FinalResultTable
    from vfe4.types.figures import FigureResultRow
    from vfe4.types.training import canonical_json_bytes

    table = finalized_figure_inputs().result_table
    assert table.aggregation_sha256 != "0" * 64
    assert len(table.result_rows) == len(table.result_row_keys) == 5
    assert all(type(row) is FigureResultRow for row in table.result_rows)
    assert tuple(row.result_row_key for row in table.result_rows) == (
        table.result_row_keys
    )
    document = json.loads(canonical_json_bytes(table))
    assert "result_row_keys" not in document
    assert len(document["result_rows"]) == 5
    assert (
        FinalResultTable.from_document(document)
        == table
    )
    with pytest.raises(ValueError, match="row_sha256"):
        dataclasses.replace(
            table.result_rows[0],
            mean_nll_per_token=(
                table.result_rows[0].mean_nll_per_token + 1.0
            ),
        )


def test_output_format_preflight_checks_the_installed_agg_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    from vfe4.figures.render import preflight_figure_output_formats

    specs = finalized_figure_inputs().specs
    assert preflight_figure_output_formats(specs) == (
        "svg",
        "png",
        "pdf",
    )
    monkeypatch.setattr(
        FigureCanvasAgg,
        "get_supported_filetypes",
        lambda _self: {"svg": "Scalable Vector Graphics", "png": "PNG"},
    )
    with pytest.raises(FigureInputError, match="unsupported.*pdf"):
        preflight_figure_output_formats(specs)


def test_result_table_requires_full_series_and_explicit_applicability() -> None:
    inputs = finalized_figure_inputs()
    table = inputs.result_table

    with pytest.raises(FigureInputError, match="every figure series"):
        from vfe4.figures.load import (
            FinalResultTable,
            _validate_result_table,
        )

        arms = {
            arm.arm_id: arm
            for arm in resolve_training_config(
                default_training_config_mapping()
            ).endpoint_inventory.arms
        }
        _validate_result_table(
            FinalResultTable.create(
                endpoint_inventory_sha256=table.endpoint_inventory_sha256,
                metrics_jsonl_sha256s=table.metrics_jsonl_sha256s,
                aggregation_sha256=table.aggregation_sha256,
                result_rows=table.result_rows,
                figure_panel_keys=table.figure_panel_keys,
                figure_series_keys=table.figure_series_keys,
                applicability_rows=table.applicability_rows,
                points=tuple(
                    point
                    for point in table.points
                    if point.series_key != table.figure_series_keys[-1]
                ),
            ),
            inventory=resolve_training_config(
                default_training_config_mapping()
            ).endpoint_inventory,
            sources=inputs.metric_sources,
            arms=arms,
            metric_records=inputs.metric_records,
        )

    first_not_applicable = next(
        index
        for index, row in enumerate(table.applicability_rows)
        if row.applicability == "not_applicable"
    )
    altered_rows = list(table.applicability_rows)
    altered_rows[first_not_applicable] = dataclasses.replace(
        altered_rows[first_not_applicable],
        applicability="applicable",
        reason="fabricated",
    )
    with pytest.raises(FigureInputError, match="applicability"):
        from vfe4.figures.load import (
            FinalResultTable,
            _validate_result_table,
        )

        inventory = resolve_training_config(
            default_training_config_mapping()
        ).endpoint_inventory
        _validate_result_table(
            FinalResultTable.create(
                endpoint_inventory_sha256=table.endpoint_inventory_sha256,
                metrics_jsonl_sha256s=table.metrics_jsonl_sha256s,
                aggregation_sha256=table.aggregation_sha256,
                result_rows=table.result_rows,
                figure_panel_keys=table.figure_panel_keys,
                figure_series_keys=table.figure_series_keys,
                applicability_rows=tuple(altered_rows),
                points=table.points,
            ),
            inventory=inventory,
            sources=inputs.metric_sources,
            arms={arm.arm_id: arm for arm in inventory.arms},
            metric_records=inputs.metric_records,
        )

    altered_rows[first_not_applicable] = dataclasses.replace(
        table.applicability_rows[first_not_applicable],
        reason="caller_selected_reason",
    )
    with pytest.raises(FigureInputError, match="frozen arm semantics"):
        from vfe4.figures.load import _validate_result_table

        inventory = resolve_training_config(
            default_training_config_mapping()
        ).endpoint_inventory
        _validate_result_table(
            FinalResultTable.create(
                endpoint_inventory_sha256=table.endpoint_inventory_sha256,
                metrics_jsonl_sha256s=table.metrics_jsonl_sha256s,
                aggregation_sha256=table.aggregation_sha256,
                result_rows=table.result_rows,
                figure_panel_keys=table.figure_panel_keys,
                figure_series_keys=table.figure_series_keys,
                applicability_rows=tuple(altered_rows),
                points=table.points,
            ),
            inventory=inventory,
            sources=inputs.metric_sources,
            arms={arm.arm_id: arm for arm in inventory.arms},
            metric_records=inputs.metric_records,
        )

    with pytest.raises(FigureInputError, match="raw numerator"):
        dataclasses.replace(
            table.points[0],
            numerator=None,
            denominator=None,
        )


def test_result_table_values_are_exact_views_of_authoritative_metrics() -> None:
    from vfe4.figures.load import FinalResultTable, _validate_result_table

    inputs = finalized_figure_inputs()
    table = inputs.result_table
    inventory = resolve_training_config(
        default_training_config_mapping()
    ).endpoint_inventory
    changed = list(table.points)
    first = changed[0]
    changed[0] = dataclasses.replace(
        first,
        numerator=first.numerator + 1.0,
        value=first.value + 1.0,
    )
    rehashed = FinalResultTable.create(
        endpoint_inventory_sha256=table.endpoint_inventory_sha256,
        metrics_jsonl_sha256s=table.metrics_jsonl_sha256s,
        aggregation_sha256=table.aggregation_sha256,
        result_rows=table.result_rows,
        figure_panel_keys=table.figure_panel_keys,
        figure_series_keys=table.figure_series_keys,
        applicability_rows=table.applicability_rows,
        points=tuple(changed),
    )
    with pytest.raises(FigureInputError, match="authoritative metric"):
        _validate_result_table(
            rehashed,
            inventory=inventory,
            sources=inputs.metric_sources,
            arms={arm.arm_id: arm for arm in inventory.arms},
            metric_records=inputs.metric_records,
        )
    changed[0] = dataclasses.replace(
        first,
        lower=first.value - 1.0,
        upper=first.value + 1.0,
    )
    interval_table = FinalResultTable.create(
        endpoint_inventory_sha256=table.endpoint_inventory_sha256,
        metrics_jsonl_sha256s=table.metrics_jsonl_sha256s,
        aggregation_sha256=table.aggregation_sha256,
        result_rows=table.result_rows,
        figure_panel_keys=table.figure_panel_keys,
        figure_series_keys=table.figure_series_keys,
        applicability_rows=table.applicability_rows,
        points=tuple(changed),
    )
    with pytest.raises(FigureInputError, match="unauthoritative interval"):
        _validate_result_table(
            interval_table,
            inventory=inventory,
            sources=inputs.metric_sources,
            arms={arm.arm_id: arm for arm in inventory.arms},
            metric_records=inputs.metric_records,
        )


def test_metric_units_are_frozen_at_creation_and_figure_load() -> None:
    from vfe4.figures.load import FinalResultTable, _validate_result_table
    from vfe4.recording.metrics import (
        MetricLogError,
        WT103_METRIC_UNIT_BY_NAME,
        applicable_metric,
    )
    from vfe4.types.training import MetricRecord, owned_sha256

    with pytest.raises(MetricLogError, match="frozen units"):
        applicable_metric(
            name="prior_nll_per_token",
            numerator=1.0,
            denominator=1,
            value=1.0,
            units="seconds",
        )

    inputs = finalized_figure_inputs()
    for spec in inputs.specs:
        for panel in spec.panels:
            assert {
                WT103_METRIC_UNIT_BY_NAME[name]
                for name in panel.y_columns
            } == {panel.units}
    inventory = resolve_training_config(
        default_training_config_mapping()
    ).endpoint_inventory
    first = inputs.result_table.points[0]
    source_index = next(
        index
        for index, (_source, records) in enumerate(inputs.metric_records)
        if records[0].record_sha256 == first.source_record_sha256
    )
    source, records = inputs.metric_records[source_index]
    record = records[0]
    altered_values = tuple(
        dataclasses.replace(value, units="seconds")
        if value.name == first.metric_name
        else value
        for value in record.values
    )
    altered_payload = {
        "schema_version": record.schema_version,
        "ordinal": record.ordinal,
        "utc_timestamp": record.utc_timestamp,
        "monotonic_ns": record.monotonic_ns,
        "run_id": record.run_id,
        "arm_id": record.arm_id,
        "seed_id": record.seed_id,
        "phase": record.phase,
        "split": record.split,
        "step": record.step,
        "pass_index": record.pass_index,
        "previous_record_sha256": record.previous_record_sha256,
        "values": altered_values,
    }
    altered_record = MetricRecord(
        **altered_payload,
        record_sha256=owned_sha256(
            "vfe4.wt103.metric-record.v1",
            altered_payload,
        ),
    )
    altered_metric_records = list(inputs.metric_records)
    altered_metric_records[source_index] = (source, (altered_record,))
    altered_points = tuple(
        dataclasses.replace(
            point,
            source_record_sha256=altered_record.record_sha256,
            units=(
                "seconds"
                if point.metric_name == first.metric_name
                else point.units
            ),
        )
        if point.source_record_sha256 == record.record_sha256
        else point
        for point in inputs.result_table.points
    )
    altered_table = FinalResultTable.create(
        endpoint_inventory_sha256=(
            inputs.result_table.endpoint_inventory_sha256
        ),
        metrics_jsonl_sha256s=inputs.result_table.metrics_jsonl_sha256s,
        aggregation_sha256=inputs.result_table.aggregation_sha256,
        result_rows=inputs.result_table.result_rows,
        figure_panel_keys=inputs.result_table.figure_panel_keys,
        figure_series_keys=inputs.result_table.figure_series_keys,
        applicability_rows=inputs.result_table.applicability_rows,
        points=altered_points,
    )
    with pytest.raises(FigureInputError, match="frozen metric units"):
        _validate_result_table(
            altered_table,
            inventory=inventory,
            sources=inputs.metric_sources,
            arms={arm.arm_id: arm for arm in inventory.arms},
            metric_records=tuple(altered_metric_records),
        )
    from vfe4.figures.plots import (
        plot_training_objective_and_validation,
    )

    training_spec = next(
        spec
        for spec in inputs.specs
        if spec.figure_id == "training-objective-and-validation"
    )
    with pytest.raises(FigureInputError, match="axis spec"):
        plot_training_objective_and_validation(
            training_spec,
            altered_table,
        )


def test_seed_variability_uses_distinct_quantity_axes_and_auditable_rows() -> None:
    from vfe4.figures.plots import (
        plot_seed_variability,
        plot_spd_health,
    )
    from vfe4.figures.render import _sidecar_rows

    inputs = finalized_figure_inputs()
    spec = next(
        item for item in inputs.specs if item.figure_id == "seed-variability"
    )
    figure = plot_seed_variability(spec, inputs.result_table)
    labels = tuple(axis.get_ylabel() for axis in figure.axes)
    assert labels == (
        "Prior NLL (nats/token)",
        "Perplexity",
        "Objective (nats/token)",
        "Acceptance rate",
        "Peak allocated bytes",
    )
    spd_spec = next(
        item for item in inputs.specs if item.figure_id == "spd-health"
    )
    spd = plot_spd_health(spd_spec, inputs.result_table)
    assert tuple(axis.get_ylabel() for axis in spd.axes) == (
        "Minimum Cholesky pivot",
        "Failed pivots",
        "Condition estimate",
        "Solve residual",
        "Damping events",
        "SPD projections",
    )
    rows = _sidecar_rows(spec, inputs.result_table)
    raw = next(row for row in rows if row["row_kind"] == "raw_seed")
    assert raw["numerator"] is not None
    assert raw["denominator"] is not None


def test_dependency_declarations_pin_figures_without_replacing_torch() -> None:
    root = Path(__file__).resolve().parents[2]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    lock = (root / "requirements-wt103.lock").read_text(encoding="utf-8")
    lock_input = json.loads(
        (root / "requirements-wt103.lock-input.json").read_text(
            encoding="utf-8"
        )
    )
    attributes = (root / ".gitattributes").read_text(encoding="utf-8")

    assert '"matplotlib==3.10.6"' in pyproject
    assert '"tiktoken==0.12.0"' in pyproject
    assert "matplotlib==3.10.6" in lock
    assert "590f5925c2d650b5c9d813c5b3b5fc53f2929c3f8ef463e4ecfa7e052044fb2b" in lock
    assert "tiktoken==0.12.0" in lock
    assert "ffc5288f34a8bc02e1ea7047b8d041104791d2ddbf42d1e5fa07822cbffe16bd" in lock
    assert "torch==" not in lock
    assert "pip freeze" not in lock.casefold()
    assert lock_input["schema_version"] == "wt103-lock-input-manifest-v1"
    assert lock_input["target_python_version"] == "3.12"
    assert lock_input["writer_code_sha256"] == hashlib.sha256(
        (root / "vfe4" / "artifacts" / "environment.py").read_bytes()
    ).hexdigest()
    assert lock_input["task13_obligations"] == [
        "task13_capture_exact_installed_record_sha256:matplotlib",
        "task13_capture_exact_installed_record_sha256:tiktoken",
    ]
    assert all(
        requirement["expected_installed_record_sha256"] is None
        for requirement in lock_input["requirements"]
    )
    reopened = parse_lock_input_manifest(
        (root / "requirements-wt103.lock-input.json").read_bytes()
    )
    assert render_dependency_lock(reopened) == (
        root / "requirements-wt103.lock"
    ).read_bytes()
    assert "requirements-wt103.lock text eol=lf" in attributes.splitlines()
    assert (
        "requirements-wt103.lock-input.json text eol=lf"
        in attributes.splitlines()
    )


def test_content_addressed_set_resumes_only_a_canonical_prefix(
    tmp_path: Path,
) -> None:
    from test_support.wt103_figure_fakes import FilesystemFigureBackend

    output = tmp_path / "set"
    output.mkdir()
    payloads = {
        "first.data.csv": b"first\n",
        "first.svg": b"<svg/>",
        "figure-set.json": b"{}\n",
    }
    (output / "first.data.csv").write_bytes(payloads["first.data.csv"])
    _publish_or_validate_set(
        output,
        payloads,
        backend=FilesystemFigureBackend(),
    )
    assert {
        path.name: path.read_bytes() for path in output.iterdir()
    } == payloads

    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "first.svg").write_bytes(payloads["first.svg"])
    with pytest.raises(FigureInputError, match="publication order"):
        _publish_or_validate_set(
            broken,
            payloads,
            backend=FilesystemFigureBackend(),
        )


def test_click_launcher_is_import_safe_idle_and_rejects_implicit_paths() -> None:
    launcher = importlib.import_module("generate_vfe4_figures")

    assert launcher.main() is None
    assert launcher.CONFIG["operation"] == "idle"
    altered = dict(launcher.CONFIG)
    altered["operation"] = "render"
    altered["run_group_manifest_path"] = "../newest/experiment-index.json"
    altered["figure_root"] = "figures"
    with pytest.raises(FigureInputError, match="absolute explicit"):
        launcher.main(altered)

    root = Path(__file__).resolve().parents[2]
    tree = ast.parse(
        (root / "generate_vfe4_figures.py").read_text(encoding="utf-8")
    )
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    assert not {
        "argparse",
        "torch",
        "tiktoken",
        "train_vfe4",
        "vfe4.training",
        "vfe4.checkpoint",
        "vfe4.data",
    }.intersection(imports)
    source = (root / "generate_vfe4_figures.py").read_text(
        encoding="utf-8"
    )
    assert 'probe.status != "pass"' in source
    assert 'probe.status != "PASS"' not in source


def test_loader_requires_exact_40_confirmation_index_and_csv_bytes(
    tmp_path: Path,
) -> None:
    training = resolve_training_config(default_training_config_mapping())
    index_path = build_finalized_figure_experiment(
        tmp_path / "experiment"
    )
    inputs = load_figure_inputs(
        run_group_manifest_path=index_path,
        inventory=training.endpoint_inventory,
        specs=resolve_figure_config(
            default_figure_config_mapping(training.endpoint_inventory)
        ).specs,
    )

    assert len(inputs.metric_sources) == 40
    assert tuple(
        source.terminal_checkpoint_key
        for source in inputs.metric_sources
    ) == training.endpoint_inventory.terminal_checkpoint_keys
    first_csv = index_path.parent / inputs.metric_sources[0].metrics_csv_path
    first_csv.write_bytes(first_csv.read_bytes() + b"tampered\n")
    with pytest.raises(FigureInputError, match="final experiment index"):
        load_figure_inputs(
            run_group_manifest_path=index_path,
            inventory=training.endpoint_inventory,
            specs=inputs.specs,
        )
