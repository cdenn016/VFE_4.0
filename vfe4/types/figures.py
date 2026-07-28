"""Immutable, renderer-independent WikiText-103 figure records."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Literal

from .training import (
    EndpointInventory,
    WT103_FIGURE_PANEL_KEYS,
    owned_sha256,
)


def _sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _text(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be nonempty text")
    return value


def _text_tuple(
    value: object,
    name: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if (
        type(value) is not tuple
        or (not allow_empty and not value)
        or any(type(item) is not str or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"{name} must be a unique immutable text tuple")
    return value


def _payload(value: object, omitted: tuple[str, ...]) -> dict[str, object]:
    return {
        item.name: getattr(value, item.name)
        for item in fields(value)  # type: ignore[arg-type]
        if item.name not in omitted
    }


@dataclass(frozen=True, slots=True)
class FigureSeriesSpec:
    schema_version: Literal["wt103-figure-series-spec-v1"]
    series_key: str
    arm_id: str
    source_columns: tuple[str, ...]
    aggregation: str
    uncertainty_interval: str
    applicability: Literal["applicable", "not_applicable"]
    applicability_reason: str
    color: str
    marker: str
    series_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != "wt103-figure-series-spec-v1":
            raise ValueError("unsupported figure series schema")
        for name in (
            "series_key",
            "arm_id",
            "aggregation",
            "uncertainty_interval",
            "applicability_reason",
            "color",
            "marker",
        ):
            _text(getattr(self, name), name)
        _text_tuple(self.source_columns, "source_columns")
        if self.applicability not in ("applicable", "not_applicable"):
            raise ValueError("figure series applicability is invalid")
        expected = owned_sha256(
            "vfe4.wt103.figure-series-spec.v1",
            _payload(self, ("series_sha256",)),
        )
        _sha256(self.series_sha256, "series_sha256")
        if self.series_sha256 != expected:
            raise ValueError("series_sha256 does not match series spec")


@dataclass(frozen=True, slots=True)
class FigurePanelSpec:
    schema_version: Literal["wt103-figure-panel-spec-v1"]
    panel_key: str
    title: str
    x_column: str
    y_columns: tuple[str, ...]
    x_label: str
    y_label: str
    units: str
    x_scale: Literal["linear", "log"]
    y_scale: Literal["linear", "log"]
    series_keys: tuple[str, ...]
    panel_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != "wt103-figure-panel-spec-v1":
            raise ValueError("unsupported figure panel schema")
        for name in (
            "panel_key",
            "title",
            "x_column",
            "x_label",
            "y_label",
            "units",
        ):
            _text(getattr(self, name), name)
        _text_tuple(self.y_columns, "y_columns")
        _text_tuple(self.series_keys, "series_keys")
        if self.x_scale not in ("linear", "log") or self.y_scale not in (
            "linear",
            "log",
        ):
            raise ValueError("figure scales are invalid")
        expected = owned_sha256(
            "vfe4.wt103.figure-panel-spec.v1",
            _payload(self, ("panel_sha256",)),
        )
        _sha256(self.panel_sha256, "panel_sha256")
        if self.panel_sha256 != expected:
            raise ValueError("panel_sha256 does not match panel spec")


@dataclass(frozen=True, slots=True)
class FigureSpec:
    schema_version: Literal["wt103-figure-spec-v1"]
    figure_id: str
    panels: tuple[FigurePanelSpec, ...]
    series: tuple[FigureSeriesSpec, ...]
    source_columns: tuple[str, ...]
    aggregation: str
    uncertainty_interval: str
    font_family: Literal["DejaVu Sans"]
    svg_hashsalt: str
    formats: tuple[Literal["svg", "png", "pdf"], ...]
    data_sidecars: tuple[Literal["csv", "json"], ...]
    caption: str
    alt_text: str
    spec_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != "wt103-figure-spec-v1":
            raise ValueError("unsupported figure spec schema")
        _text(self.figure_id, "figure_id")
        if (
            type(self.panels) is not tuple
            or not self.panels
            or any(type(item) is not FigurePanelSpec for item in self.panels)
            or len({item.panel_key for item in self.panels}) != len(self.panels)
        ):
            raise ValueError("panels must contain unique exact panel records")
        if (
            type(self.series) is not tuple
            or not self.series
            or any(type(item) is not FigureSeriesSpec for item in self.series)
            or len({item.series_key for item in self.series}) != len(self.series)
        ):
            raise ValueError("series must contain unique exact series records")
        panel_series = {key for panel in self.panels for key in panel.series_keys}
        if panel_series != {item.series_key for item in self.series}:
            raise ValueError("figure panel and series inventories disagree")
        _text_tuple(self.source_columns, "source_columns")
        for name in (
            "aggregation",
            "uncertainty_interval",
            "svg_hashsalt",
            "caption",
            "alt_text",
        ):
            _text(getattr(self, name), name)
        if self.font_family != "DejaVu Sans":
            raise ValueError("figure font family is frozen")
        if self.formats != ("svg", "png", "pdf"):
            raise ValueError("required figure formats are frozen")
        if self.data_sidecars != ("csv", "json"):
            raise ValueError("required data sidecars are frozen")
        expected = owned_sha256(
            "vfe4.wt103.figure-spec.v1",
            _payload(self, ("spec_sha256",)),
        )
        _sha256(self.spec_sha256, "spec_sha256")
        if self.spec_sha256 != expected:
            raise ValueError("spec_sha256 does not match figure spec")


@dataclass(frozen=True, slots=True)
class FigureInputIdentity:
    schema_version: Literal["wt103-figure-input-identity-v1"]
    endpoint_inventory_sha256: str
    run_group_manifest_sha256: str
    metrics_jsonl_sha256s: tuple[str, ...]
    result_table_sha256: str
    regenerated_csv_sha256s: tuple[str, ...]
    input_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != "wt103-figure-input-identity-v1":
            raise ValueError("unsupported figure input schema")
        for name in (
            "endpoint_inventory_sha256",
            "run_group_manifest_sha256",
            "result_table_sha256",
        ):
            _sha256(getattr(self, name), name)
        for name in ("metrics_jsonl_sha256s", "regenerated_csv_sha256s"):
            values = getattr(self, name)
            if type(values) is not tuple or not values:
                raise ValueError(f"{name} must be a nonempty tuple")
            for index, digest in enumerate(values):
                _sha256(digest, f"{name}[{index}]")
        expected = owned_sha256(
            "vfe4.wt103.figure-input-identity.v1",
            _payload(self, ("input_sha256",)),
        )
        _sha256(self.input_sha256, "input_sha256")
        if self.input_sha256 != expected:
            raise ValueError("input_sha256 does not match figure inputs")


@dataclass(frozen=True, slots=True)
class FigureOutputIdentity:
    schema_version: Literal["wt103-figure-output-identity-v1"]
    figure_id: str
    spec_sha256: str
    input_sha256: str
    environment_sha256: str
    svg_sha256: str
    png_sha256: str
    pdf_sha256: str
    csv_sha256: str
    json_sha256: str
    caption_sha256: str
    alt_text_sha256: str
    output_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != "wt103-figure-output-identity-v1":
            raise ValueError("unsupported figure output schema")
        _text(self.figure_id, "figure_id")
        for name in tuple(self.__dataclass_fields__)[2:-1]:
            _sha256(getattr(self, name), name)
        expected = owned_sha256(
            "vfe4.wt103.figure-output-identity.v1",
            _payload(self, ("output_sha256",)),
        )
        _sha256(self.output_sha256, "output_sha256")
        if self.output_sha256 != expected:
            raise ValueError("output_sha256 does not match figure outputs")


@dataclass(frozen=True, slots=True)
class FigureSetManifest:
    schema_version: Literal["wt103-figure-set-manifest-v1"]
    endpoint_inventory_sha256: str
    figure_input_sha256: str
    outputs: tuple[FigureOutputIdentity, ...]
    figure_set_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != "wt103-figure-set-manifest-v1":
            raise ValueError("unsupported figure-set schema")
        _sha256(
            self.endpoint_inventory_sha256,
            "endpoint_inventory_sha256",
        )
        _sha256(self.figure_input_sha256, "figure_input_sha256")
        if (
            type(self.outputs) is not tuple
            or len(self.outputs) != len(WT103_FIGURE_PANEL_KEYS)
            or any(type(item) is not FigureOutputIdentity for item in self.outputs)
            or tuple(item.figure_id for item in self.outputs) != WT103_FIGURE_PANEL_KEYS
        ):
            raise ValueError("figure-set output inventory is incomplete")
        expected = owned_sha256(
            "vfe4.wt103.figure-set-manifest.v1",
            _payload(self, ("figure_set_sha256",)),
        )
        _sha256(self.figure_set_sha256, "figure_set_sha256")
        if self.figure_set_sha256 != expected:
            raise ValueError("figure_set_sha256 does not match manifest")


def _series_spec(series_key: str, index: int) -> FigureSeriesSpec:
    parts = series_key.split("/")
    payload = {
        "schema_version": "wt103-figure-series-spec-v1",
        "series_key": series_key,
        "arm_id": parts[1],
        "source_columns": (
            "arm_id",
            "seed_id",
            "counted_targets",
            parts[-1],
            "applicability",
            "applicability_reason",
        ),
        "aggregation": "frozen_metric_numerator_denominator_only",
        "uncertainty_interval": (
            "preregistered_estimator_or_seed_interval_as_applicable"
        ),
        "applicability": "applicable",
        "applicability_reason": "derived_from_exact_arm_spec",
        "color": f"C{index % 10}",
        "marker": ("o", "s", "^", "D", "v")[index % 5],
    }
    return FigureSeriesSpec(
        **payload,
        series_sha256=owned_sha256(
            "vfe4.wt103.figure-series-spec.v1",
            payload,
        ),
    )


def default_figure_specs(
    inventory: EndpointInventory,
) -> tuple[FigureSpec, ...]:
    """Build the fixed eight-spec registry from one endpoint inventory."""

    if type(inventory) is not EndpointInventory:
        raise ValueError("inventory must be an exact EndpointInventory")
    inventory.__post_init__()
    specs: list[FigureSpec] = []
    all_series = inventory.figure_series_keys
    for figure_index, figure_id in enumerate(WT103_FIGURE_PANEL_KEYS):
        prefix = f"{figure_id}/"
        series_keys = tuple(key for key in all_series if key.startswith(prefix))
        if not series_keys:
            raise ValueError(f"figure {figure_id} has no derived series")
        series = tuple(
            _series_spec(key, index) for index, key in enumerate(series_keys)
        )
        panel_payload = {
            "schema_version": "wt103-figure-panel-spec-v1",
            "panel_key": figure_id,
            "title": figure_id.replace("-", " ").title(),
            "x_column": "counted_targets",
            "y_columns": tuple(
                sorted(
                    {
                        column
                        for item in series
                        for column in item.source_columns
                        if column
                        not in (
                            "arm_id",
                            "seed_id",
                            "counted_targets",
                            "applicability",
                            "applicability_reason",
                        )
                    }
                )
            ),
            "x_label": "Counted training targets",
            "y_label": figure_id.replace("-", " "),
            "units": "record-defined",
            "x_scale": "linear",
            "y_scale": "linear",
            "series_keys": series_keys,
        }
        panel = FigurePanelSpec(
            **panel_payload,
            panel_sha256=owned_sha256(
                "vfe4.wt103.figure-panel-spec.v1",
                panel_payload,
            ),
        )
        spec_payload = {
            "schema_version": "wt103-figure-spec-v1",
            "figure_id": figure_id,
            "panels": (panel,),
            "series": series,
            "source_columns": tuple(
                sorted({column for item in series for column in item.source_columns})
            ),
            "aggregation": "finalized_manifest_validated_records_only",
            "uncertainty_interval": (
                "preregistered_estimator_or_seed_interval_as_applicable"
            ),
            "font_family": "DejaVu Sans",
            "svg_hashsalt": "vfe4-wt103-figure-v1",
            "formats": ("svg", "png", "pdf"),
            "data_sidecars": ("csv", "json"),
            "caption": (
                f"{figure_id}: finalized inventory-derived series only; "
                "inapplicable fields are explicitly labeled."
            ),
            "alt_text": (
                f"Deterministic {figure_id} figure for the exact ordered "
                "WikiText-103 arm inventory."
            ),
        }
        specs.append(
            FigureSpec(
                **spec_payload,
                spec_sha256=owned_sha256(
                    "vfe4.wt103.figure-spec.v1",
                    spec_payload,
                ),
            )
        )
    return tuple(specs)


__all__ = [
    "FigureInputIdentity",
    "FigureOutputIdentity",
    "FigurePanelSpec",
    "FigureSeriesSpec",
    "FigureSetManifest",
    "FigureSpec",
    "default_figure_specs",
]
