"""Deterministic, durability-backed publication of a complete figure set."""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import io
import math
import os
import stat
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from vfe4.artifacts.durability import DurabilityCollisionError
from vfe4.recording.metrics import _read_regular_bytes
from vfe4.types.figures import (
    FigureOutputIdentity,
    FigureSetManifest,
    FigureSpec,
)
from vfe4.types.training import canonical_json_bytes, owned_sha256

from .load import FigureInputError, FinalResultTable, LoadedFigureInputs
from .plots import (
    plot_complete_elbo_decomposition,
    plot_seed_variability,
    plot_source_entropy_effective_count,
    plot_spd_health,
    plot_terminal_prior_nll_ppl,
    plot_throughput_memory,
    plot_training_objective_and_validation,
    plot_update_acceptance,
)


_POINTWISE_T_CRITICAL_DF7 = 2.364624251592784
_FIXED_PDF_TIME = datetime(2000, 1, 1, tzinfo=timezone.utc)
_RC_PARAMS = {
    "font.family": "DejaVu Sans",
    "font.size": 8.0,
    "axes.labelsize": 8.0,
    "axes.titlesize": 9.0,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.0,
    "lines.linewidth": 1.0,
    "axes.linewidth": 0.7,
    "savefig.dpi": 300,
    "figure.dpi": 100,
    "svg.hashsalt": "vfe4-wt103-figure-v1",
    "path.simplify": False,
    "agg.path.chunksize": 0,
    "pdf.compression": 0,
}


class FigureDurabilityBackend(Protocol):
    def create_exclusive(self, path: Path, payload: bytes) -> object: ...

    def replace_durable(self, path: Path, payload: bytes) -> object: ...


@dataclasses.dataclass(frozen=True, slots=True)
class RenderedFigureSet:
    output_path: Path
    manifest_path: Path
    index_path: Path
    manifest: FigureSetManifest


def _is_redirect_or_reparse(path: Path, status: os.stat_result) -> bool:
    if stat.S_ISLNK(status.st_mode):
        return True
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if bool(getattr(status, "st_file_attributes", 0) & reparse):
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(callable(is_junction) and is_junction())


def validate_figure_output_root(figure_root: Path) -> Path:
    """Reject redirected existing output-root components before publication."""

    if not isinstance(figure_root, Path):
        raise FigureInputError("figure_root must be pathlib.Path")
    current = figure_root
    while True:
        try:
            status = current.lstat()
        except FileNotFoundError:
            parent = current.parent
            if parent == current:
                raise FigureInputError(
                    "figure output root has no inspectable anchor"
                )
            current = parent
            continue
        except OSError as exc:
            raise FigureInputError(
                f"figure output root metadata is unavailable: {current}"
            ) from exc
        if (
            not stat.S_ISDIR(status.st_mode)
            or _is_redirect_or_reparse(current, status)
        ):
            raise FigureInputError(
                "figure output root contains a symlink, junction, or "
                f"reparse component: {current}"
            )
        parent = current.parent
        if parent == current:
            return figure_root
        current = parent


def preflight_figure_output_formats(
    specs: tuple[FigureSpec, ...],
) -> tuple[str, ...]:
    """Fail before input access when Agg cannot write a requested format."""

    if (
        type(specs) is not tuple
        or not specs
        or any(type(spec) is not FigureSpec for spec in specs)
    ):
        raise FigureInputError(
            "format preflight requires exact immutable figure specs"
        )
    import matplotlib

    if matplotlib.__version__ != "3.10.6":
        raise FigureInputError(
            "figure rendering requires matplotlib==3.10.6"
        )
    matplotlib.use("Agg", force=True)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    supported = FigureCanvasAgg(Figure()).get_supported_filetypes()
    requested = tuple(
        dict.fromkeys(
            format_name
            for spec in specs
            for format_name in spec.formats
        )
    )
    unsupported = tuple(
        format_name
        for format_name in requested
        if format_name not in supported
    )
    if unsupported:
        raise FigureInputError(
            "requested figure format is unsupported: "
            + ", ".join(unsupported)
        )
    return requested


def _plot_function(spec: FigureSpec, table: FinalResultTable):
    if spec.figure_id == "training-objective-and-validation":
        return plot_training_objective_and_validation(spec, table)
    if spec.figure_id == "terminal-prior-nll-ppl":
        return plot_terminal_prior_nll_ppl(spec, table)
    if spec.figure_id == "complete-elbo-decomposition":
        return plot_complete_elbo_decomposition(spec, table)
    if spec.figure_id == "source-entropy-effective-count":
        return plot_source_entropy_effective_count(spec, table)
    if spec.figure_id == "update-acceptance":
        return plot_update_acceptance(spec, table)
    if spec.figure_id == "spd-health":
        return plot_spd_health(spec, table)
    if spec.figure_id == "throughput-memory":
        return plot_throughput_memory(spec, table)
    if spec.figure_id == "seed-variability":
        return plot_seed_variability(spec, table)
    raise FigureInputError(f"unknown frozen figure ID {spec.figure_id!r}")


def _figure_environment_sha256() -> str:
    import matplotlib

    return owned_sha256(
        "vfe4.wt103.figure-environment.v1",
        {
            "matplotlib_version": matplotlib.__version__,
            "backend": "Agg",
            "font_family": "DejaVu Sans",
            "svg_hashsalt": "vfe4-wt103-figure-v1",
            "metadata_policy": "fixed_no_current_timestamp",
            "rc_params": tuple(sorted(_RC_PARAMS.items())),
        },
    )


def _sidecar_rows(
    spec: FigureSpec,
    table: FinalResultTable,
) -> tuple[dict[str, object], ...]:
    series_keys = {series.series_key for series in spec.series}
    points = tuple(
        point for point in table.points if point.series_key in series_keys
    )
    rows: list[dict[str, object]] = []
    for point in points:
        rows.append(
            {
                "row_kind": "raw_seed",
                "series_key": point.series_key,
                "arm_id": point.arm_id,
                "metric_name": point.metric_name,
                "seed_id": point.seed_id,
                "source_record_sha256": point.source_record_sha256,
                "source_phase": point.source_phase,
                "source_split": point.source_split,
                "source_step": point.source_step,
                "source_pass_index": point.source_pass_index,
                "counted_targets": point.counted_targets,
                "numerator": point.numerator,
                "denominator": point.denominator,
                "value": point.value,
                "lower": point.lower,
                "upper": point.upper,
                "units": point.units,
                "applicability": point.applicability,
                "applicability_reason": point.applicability_reason,
                "result_role": point.result_role,
            }
        )
    grouping = tuple(
        dict.fromkeys(
            (
                point.series_key,
                point.arm_id,
                point.metric_name,
                point.units,
                point.result_role,
            )
            for point in points
        )
    )
    for series_key, arm_id, metric_name, units, result_role in grouping:
        selected = tuple(
            point
            for point in points
            if point.series_key == series_key
            and point.metric_name == metric_name
        )
        x_values = tuple(
            sorted({point.counted_targets for point in selected})
        )
        for counted_targets in x_values:
            values = tuple(
                point.value
                for point in selected
                if point.counted_targets == counted_targets
            )
            mean = math.fsum(values) / len(values)
            half_width = (
                0.0
                if len(values) == 1
                else _POINTWISE_T_CRITICAL_DF7
                * statistics.stdev(values)
                / math.sqrt(len(values))
            )
            rows.append(
                {
                    "row_kind": "descriptive_mean_df7_interval",
                    "series_key": series_key,
                    "arm_id": arm_id,
                    "metric_name": metric_name,
                    "seed_id": None,
                    "source_record_sha256": None,
                    "source_phase": None,
                    "source_split": None,
                    "source_step": None,
                    "source_pass_index": None,
                    "counted_targets": counted_targets,
                    "numerator": None,
                    "denominator": None,
                    "value": mean,
                    "lower": mean - half_width,
                    "upper": mean + half_width,
                    "units": units,
                    "applicability": "applicable",
                    "applicability_reason": (
                        "descriptive_pointwise_mean_and_df7_interval"
                    ),
                    "result_role": result_role,
                }
            )
    for applicability in table.applicability_rows:
        if (
            applicability.panel_key == spec.figure_id
            and applicability.applicability == "not_applicable"
        ):
            rows.append(
                {
                    "row_kind": "not_applicable",
                    "series_key": None,
                    "arm_id": applicability.arm_id,
                    "metric_name": None,
                    "seed_id": None,
                    "source_record_sha256": None,
                    "source_phase": None,
                    "source_split": None,
                    "source_step": None,
                    "source_pass_index": None,
                    "counted_targets": None,
                    "numerator": None,
                    "denominator": None,
                    "value": None,
                    "lower": None,
                    "upper": None,
                    "units": None,
                    "applicability": "not_applicable",
                    "applicability_reason": applicability.reason,
                    "result_role": None,
                }
            )
    return tuple(rows)


def _json_sidecar(
    *,
    spec: FigureSpec,
    inputs: LoadedFigureInputs,
    rows: tuple[dict[str, object], ...],
) -> bytes:
    return canonical_json_bytes(
        {
            "schema_version": "wt103-figure-data-v1",
            "figure_id": spec.figure_id,
            "spec_sha256": spec.spec_sha256,
            "figure_input_sha256": inputs.identity.input_sha256,
            "rows": rows,
            "caption": spec.caption,
            "alt_text": spec.alt_text,
        }
    )


def _csv_sidecar(rows: tuple[dict[str, object], ...]) -> bytes:
    columns = (
        "row_kind",
        "series_key",
        "arm_id",
        "metric_name",
        "seed_id",
        "source_record_sha256",
        "source_phase",
        "source_split",
        "source_step",
        "source_pass_index",
        "counted_targets",
        "numerator",
        "denominator",
        "value",
        "lower",
        "upper",
        "units",
        "applicability",
        "applicability_reason",
        "result_role",
    )
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow(
            tuple(
                ""
                if row[column] is None
                else (
                    format(row[column], ".17g")
                    if type(row[column]) is float
                    else str(row[column])
                )
                for column in columns
            )
        )
    return stream.getvalue().encode("utf-8")


def _image_bytes(figure: object, spec: FigureSpec) -> dict[str, bytes]:
    outputs: dict[str, bytes] = {}
    metadata = {
        "svg": {
            "Creator": "VFE4",
            "Date": "2000-01-01T00:00:00Z",
            "Title": spec.figure_id,
        },
        "png": {
            "Software": "VFE4 matplotlib 3.10.6",
            "Title": spec.figure_id,
        },
        "pdf": {
            "Creator": "VFE4",
            "Producer": "VFE4 matplotlib 3.10.6",
            "CreationDate": _FIXED_PDF_TIME,
            "ModDate": _FIXED_PDF_TIME,
            "Title": spec.figure_id,
        },
    }
    for format_name in ("svg", "png", "pdf"):
        stream = io.BytesIO()
        figure.savefig(
            stream,
            format=format_name,
            dpi=300,
            metadata=metadata[format_name],
            bbox_inches="tight",
            facecolor="white",
            edgecolor="white",
        )
        outputs[format_name] = stream.getvalue()
    if (
        b"<svg" not in outputs["svg"][:1024]
        or not outputs["png"].startswith(b"\x89PNG\r\n\x1a\n")
        or not outputs["pdf"].startswith(b"%PDF-")
    ):
        raise FigureInputError("renderer produced an invalid image format")
    return outputs


def _render_one(
    spec: FigureSpec,
    inputs: LoadedFigureInputs,
    *,
    environment_sha256: str,
) -> tuple[FigureOutputIdentity, dict[str, bytes]]:
    import matplotlib

    if matplotlib.__version__ != "3.10.6":
        raise FigureInputError(
            "figure rendering requires matplotlib==3.10.6"
        )
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    rows = _sidecar_rows(spec, inputs.result_table)
    json_bytes = _json_sidecar(spec=spec, inputs=inputs, rows=rows)
    csv_bytes = _csv_sidecar(rows)
    caption_bytes = (spec.caption + "\n").encode("utf-8")
    alt_bytes = (spec.alt_text + "\n").encode("utf-8")
    with plt.rc_context(_RC_PARAMS):
        figure = _plot_function(spec, inputs.result_table)
        try:
            images = _image_bytes(figure, spec)
        finally:
            plt.close(figure)
    payloads = {
        f"{spec.figure_id}.data.csv": csv_bytes,
        f"{spec.figure_id}.data.json": json_bytes,
        f"{spec.figure_id}.caption.txt": caption_bytes,
        f"{spec.figure_id}.alt.txt": alt_bytes,
        f"{spec.figure_id}.svg": images["svg"],
        f"{spec.figure_id}.png": images["png"],
        f"{spec.figure_id}.pdf": images["pdf"],
    }
    identity = FigureOutputIdentity.create(
        figure_id=spec.figure_id,
        spec_sha256=spec.spec_sha256,
        input_sha256=inputs.identity.input_sha256,
        environment_sha256=environment_sha256,
        svg_sha256=hashlib.sha256(images["svg"]).hexdigest(),
        png_sha256=hashlib.sha256(images["png"]).hexdigest(),
        pdf_sha256=hashlib.sha256(images["pdf"]).hexdigest(),
        csv_sha256=hashlib.sha256(csv_bytes).hexdigest(),
        json_sha256=hashlib.sha256(json_bytes).hexdigest(),
        caption_sha256=hashlib.sha256(caption_bytes).hexdigest(),
        alt_text_sha256=hashlib.sha256(alt_bytes).hexdigest(),
    )
    return identity, payloads


def _publish_and_reopen(
    path: Path,
    payload: bytes,
    *,
    backend: FigureDurabilityBackend,
) -> None:
    try:
        backend.create_exclusive(path, payload)
    except (DurabilityCollisionError, FileExistsError):
        try:
            observed = _read_regular_bytes(path)
        except (FigureInputError, OSError) as exc:
            raise FigureInputError(
                f"figure artifact collision cannot be reopened: {path}: {exc}"
            ) from exc
        if observed != payload:
            raise FigureInputError(
                f"figure artifact collision has different bytes: {path}"
            )
    except Exception as exc:
        raise FigureInputError(
            f"figure artifact publication failed: {path}: {exc}"
        ) from exc
    if _read_regular_bytes(path) != payload:
        raise FigureInputError(
            f"figure artifact reopen mismatch: {path}"
        )


def _publish_or_validate_set(
    output_path: Path,
    payloads: dict[str, bytes],
    *,
    backend: FigureDurabilityBackend,
) -> None:
    try:
        entries = tuple(output_path.iterdir())
    except OSError as exc:
        raise FigureInputError(
            "existing figure set cannot be inspected"
        ) from exc
    if any(not path.is_file() for path in entries):
        raise FigureInputError(
            "existing content-addressed figure set contains non-files"
        )
    names = {path.name for path in entries}
    unexpected = names - set(payloads)
    if unexpected:
        raise FigureInputError(
            "existing content-addressed figure set has unexpected files: "
            + ",".join(sorted(unexpected))
        )
    ordered_names = tuple(payloads)
    first_missing = next(
        (
            index
            for index, name in enumerate(ordered_names)
            if name not in names
        ),
        len(ordered_names),
    )
    if any(name in names for name in ordered_names[first_missing + 1 :]):
        raise FigureInputError(
            "existing content-addressed figure set violates publication order"
        )
    for index, name in enumerate(ordered_names):
        expected = payloads[name]
        path = output_path / name
        if index < first_missing:
            if _read_regular_bytes(path) != expected:
                raise FigureInputError(
                    "existing content-addressed figure bytes do not match: "
                    f"{name}"
                )
            continue
        _publish_and_reopen(path, expected, backend=backend)
    if {path.name for path in output_path.iterdir()} != set(payloads):
        raise FigureInputError(
            "content-addressed figure set did not close its exact inventory"
        )
    for name, expected in payloads.items():
        if _read_regular_bytes(output_path / name) != expected:
            raise FigureInputError(
                "existing content-addressed figure bytes do not match: "
                f"{name}"
            )


def _publish_figure_index(
    *,
    figure_root: Path,
    manifest: FigureSetManifest,
    manifest_payload: bytes,
    backend: FigureDurabilityBackend,
) -> Path:
    body = {
        "schema_version": "wt103-figure-index-v1",
        "figure_set_sha256": manifest.figure_set_sha256,
        "figure_input_sha256": manifest.figure_input_sha256,
        "relative_manifest_path": (
            f"{manifest.figure_set_sha256}/figure-set.json"
        ),
        "manifest_payload_sha256": hashlib.sha256(
            manifest_payload
        ).hexdigest(),
    }
    payload = canonical_json_bytes(
        {
            **body,
            "index_sha256": owned_sha256(
                "vfe4.wt103.figure-index.v1",
                body,
            ),
        }
    )
    path = figure_root / "figure-index.json"
    if path.exists():
        if _read_regular_bytes(path) != payload:
            replace = getattr(backend, "replace_durable", None)
            if not callable(replace):
                raise FigureInputError(
                    "durability backend cannot atomically update figure index"
                )
            try:
                replace(path, payload)
            except Exception as exc:
                raise FigureInputError(
                    f"figure-index replacement failed: {exc}"
                ) from exc
    else:
        _publish_and_reopen(path, payload, backend=backend)
    if _read_regular_bytes(path) != payload:
        raise FigureInputError("figure index reopen mismatch")
    return path


def render_figure_set(
    *,
    inputs: LoadedFigureInputs,
    figure_root: Path,
    durability_backend: FigureDurabilityBackend,
) -> RenderedFigureSet:
    """Render all eight figures in memory, then publish sidecars first."""

    if type(inputs) is not LoadedFigureInputs:
        raise FigureInputError("inputs must be exact LoadedFigureInputs")
    validate_figure_output_root(figure_root)
    if "v3_transformer" in str(figure_root).casefold():
        raise FigureInputError("figure output cannot target a V3 path")
    if not callable(getattr(durability_backend, "create_exclusive", None)):
        raise FigureInputError(
            "durability backend must expose create_exclusive"
        )
    if not callable(getattr(durability_backend, "replace_durable", None)):
        raise FigureInputError(
            "durability backend must expose replace_durable"
        )
    preflight_figure_output_formats(inputs.specs)
    environment_sha256 = _figure_environment_sha256()
    rendered = tuple(
        _render_one(
            spec,
            inputs,
            environment_sha256=environment_sha256,
        )
        for spec in inputs.specs
    )
    manifest = FigureSetManifest.create(
        endpoint_inventory_sha256=(
            inputs.identity.endpoint_inventory_sha256
        ),
        figure_input_sha256=inputs.identity.input_sha256,
        outputs=tuple(identity for identity, _payloads in rendered),
    )
    output_path = figure_root / manifest.figure_set_sha256
    figure_root.mkdir(parents=True, exist_ok=True)
    manifest_payload = canonical_json_bytes(manifest)
    all_payloads: dict[str, bytes] = {}
    for suffix in (
        ".data.csv",
        ".data.json",
        ".caption.txt",
        ".alt.txt",
    ):
        for _identity, payloads in rendered:
            name = next(
                name for name in payloads if name.endswith(suffix)
            )
            all_payloads[name] = payloads[name]
    for suffix in (".svg", ".png", ".pdf"):
        for _identity, payloads in rendered:
            name = next(
                name for name in payloads if name.endswith(suffix)
            )
            all_payloads[name] = payloads[name]
    all_payloads["figure-set.json"] = manifest_payload
    if output_path.exists():
        _publish_or_validate_set(
            output_path,
            all_payloads,
            backend=durability_backend,
        )
    else:
        try:
            output_path.mkdir()
        except FileExistsError as exc:
            raise FigureInputError(
                "content-addressed figure-set publication raced"
            ) from exc
        _publish_or_validate_set(
            output_path,
            all_payloads,
            backend=durability_backend,
        )
    manifest_path = output_path / "figure-set.json"
    index_path = _publish_figure_index(
        figure_root=figure_root,
        manifest=manifest,
        manifest_payload=manifest_payload,
        backend=durability_backend,
    )
    return RenderedFigureSet(
        output_path=output_path,
        manifest_path=manifest_path,
        index_path=index_path,
        manifest=manifest,
    )


__all__ = [
    "FigureDurabilityBackend",
    "RenderedFigureSet",
    "preflight_figure_output_formats",
    "render_figure_set",
    "validate_figure_output_root",
]
