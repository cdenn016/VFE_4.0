"""Immutable, renderer-independent WikiText-103 figure records."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal

from .training import (
    EndpointInventory,
    EstimatorProtocol,
    WT103ArmSpec,
    WT103_FIGURE_PANEL_KEYS,
    WT103_TUNING_CELLS,
    default_wt103_arm_specs,
    default_wt103_gate_specs,
    owned_sha256,
)

_OKABE_ITO = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#000000",
    "#F0E442",
)

_FIGURE_TEXT = {
    "training-objective-and-validation": (
        "Training objective, target-blind validation prior NLL, and "
        "validation perplexity by counted target. Thin traces are the eight "
        "confirmatory seeds; heavy traces and bands are the seed mean and "
        "pointwise df=7 interval.",
        "Three panels compare all five arms over counted training targets: "
        "the arm-specific training objective, target-blind validation NLL in "
        "nats per token, and validation perplexity. Each arm retains its own "
        "color and marker; thin seed traces sit beneath a mean trace and "
        "pointwise df=7 band.",
    ),
    "terminal-prior-nll-ppl": (
        "Terminal target-blind prior NLL and its derived perplexity, kept on "
        "separate axes for every ordered arm and confirmatory seed.",
        "Two panels show terminal prior NLL in nats per token and perplexity "
        "for all five arms. Raw seed traces, mean traces, and df=7 intervals "
        "remain visually distinct.",
    ),
    "complete-elbo-decomposition": (
        "Complete-ELBO terms are reported only for complete-objective arms; "
        "source KL is diagnostic-only, the unavailable estimator bound is "
        "canonically N/A, and the emission-only ablation remains explicitly "
        "labeled as a non-ELBO quantity.",
        "The complete-objective decomposition separates expected log "
        "emission; initial, source, and transition cross-entropies; "
        "continuous recognition entropy; conditional source entropy; their "
        "joint recognition-entropy estimate; and complete ELBO. Source-KL "
        "diagnostics are excluded from the objective, the estimator bound is "
        "N/A, and the emission-only arm is labeled non-ELBO.",
    ),
    "source-entropy-effective-count": (
        "Latent-source entropy and effective source count for applicable "
        "arms, with nonlatent applicability stated explicitly.",
        "Two panels show source entropy in nats and effective source count "
        "for latent-enabled arms. The nonlatent arm is explicitly marked not "
        "applicable.",
    ),
    "update-acceptance": (
        "Accepted and rejected validity-only update proposals by arm and "
        "counted target; no monotone-ELBO interpretation is implied.",
        "Accepted and rejected proposal counts are plotted for each arm over "
        "counted training targets, with raw seed traces and seed summaries.",
    ),
    "spd-health": (
        "SPD pivots, failures, conditioning, residuals, damping, and "
        "projection diagnostics for latent-enabled arms. Distinct units "
        "remain on distinct axes, and no unrecorded endpoint bound is drawn.",
        "SPD health traces show minimum Cholesky pivot, failed pivots, "
        "condition estimate, solve residual, damping events, and SPD "
        "projections for latent-enabled arms on six unit-specific axes; "
        "nonlatent status is stated explicitly.",
    ),
    "throughput-memory": (
        "Throughput, phase time, and host/device memory endpoints for every "
        "arm. Seed summaries are shown without inventing unrecorded "
        "capacity intervals.",
        "Three panels compare tokens per second, phase durations, and host "
        "plus CUDA memory metrics across all arms. Raw seed traces and df=7 "
        "summaries are shown without an unrecorded endpoint interval.",
    ),
    "seed-variability": (
        "Terminal per-seed NLL, perplexity, applicable objective, update "
        "acceptance, and device-memory values remain separate by arm.",
        "A categorical terminal summary shows every confirmatory seed for "
        "each arm and metric, plus the mean and df=7 interval. Objective "
        "metrics appear only where applicable.",
    ),
}


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
class FigureResultRow:
    """One typed terminal result row embedded in the figure table."""

    result_row_key: str
    arm_id: str
    scorer_kind: Literal["exact_autoregressive", "weighted_smc"]
    applicability: Literal["decision_bearing", "descriptive_only"]
    applicability_reason: str
    result_role: str
    seed_nll_per_token: tuple[float, ...]
    seed_perplexity: tuple[float, ...]
    mean_nll_per_token: float
    mean_perplexity: float
    status: Literal["pass", "fail", "inconclusive"]
    row_sha256: str

    def __post_init__(self) -> None:
        expected = owned_sha256(
            "vfe4.wt103.result-row.v1",
            _payload(self, ("row_sha256",)),
        )
        _sha256(self.row_sha256, "row_sha256")
        if self.row_sha256 != expected:
            raise ValueError("row_sha256 does not match result row")
        for name in (
            "result_row_key",
            "arm_id",
            "applicability_reason",
            "result_role",
        ):
            _text(getattr(self, name), name)
        if self.scorer_kind not in (
            "exact_autoregressive",
            "weighted_smc",
        ):
            raise ValueError("result-row scorer kind is invalid")
        if self.applicability not in (
            "decision_bearing",
            "descriptive_only",
        ):
            raise ValueError("result-row applicability is invalid")
        if self.status not in ("pass", "fail", "inconclusive"):
            raise ValueError("result-row status is invalid")
        if (
            type(self.seed_nll_per_token) is not tuple
            or len(self.seed_nll_per_token) != 8
            or type(self.seed_perplexity) is not tuple
            or len(self.seed_perplexity) != 8
        ):
            raise ValueError(
                "result row requires eight immutable seed observations"
            )
        for nll, perplexity in zip(
            self.seed_nll_per_token,
            self.seed_perplexity,
            strict=True,
        ):
            if (
                type(nll) is not float
                or not math.isfinite(nll)
                or nll < 0.0
                or type(perplexity) is not float
                or not math.isfinite(perplexity)
                or perplexity != math.exp(nll)
            ):
                raise ValueError(
                    "result-row seed NLL/perplexity is not exact"
                )
        expected_mean = math.fsum(self.seed_nll_per_token) / len(
            self.seed_nll_per_token
        )
        if (
            type(self.mean_nll_per_token) is not float
            or self.mean_nll_per_token != expected_mean
            or type(self.mean_perplexity) is not float
            or self.mean_perplexity != math.exp(expected_mean)
        ):
            raise ValueError("result-row means are not exact")
    @classmethod
    def create(cls, **values: object) -> "FigureResultRow":
        return cls(
            **values,
            row_sha256=owned_sha256(
                "vfe4.wt103.result-row.v1",
                values,
            ),
        )  # type: ignore[arg-type]

    @classmethod
    def from_document(cls, value: object) -> "FigureResultRow":
        if type(value) is not dict or set(value) != {
            field.name for field in fields(cls)
        }:
            raise ValueError("result-row document has an invalid key set")
        converted = dict(value)
        for name in ("seed_nll_per_token", "seed_perplexity"):
            if type(converted[name]) is not list:
                raise ValueError(f"result-row {name} must be a list")
            converted[name] = tuple(converted[name])  # type: ignore[arg-type]
        return cls(**converted)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class FrozenFigureProvenance:
    """Read-only endpoint and figure-spec identity for offline rendering."""

    schema_version: Literal["wt103-frozen-figure-provenance-v1"]
    endpoint_inventory: EndpointInventory
    figure_spec_sha256s: tuple[str, ...]
    provenance_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schema_version
            != "wt103-frozen-figure-provenance-v1"
            or type(self.endpoint_inventory) is not EndpointInventory
        ):
            raise ValueError("frozen figure provenance schema is invalid")
        self.endpoint_inventory.__post_init__()
        expected_specs = default_figure_specs(self.endpoint_inventory)
        if self.figure_spec_sha256s != tuple(
            spec.spec_sha256 for spec in expected_specs
        ):
            raise ValueError(
                "frozen figure provenance differs from its exact registry"
            )
        expected = owned_sha256(
            "vfe4.wt103.frozen-figure-provenance.v1",
            _payload(self, ("provenance_sha256",)),
        )
        _sha256(self.provenance_sha256, "provenance_sha256")
        if self.provenance_sha256 != expected:
            raise ValueError(
                "provenance_sha256 does not match frozen figure provenance"
            )


@dataclass(frozen=True, slots=True)
class FigureExperimentIndexIdentity:
    """Read-only identity derived from one immutable experiment index."""

    index_path: Path
    stage: Literal["pretest", "final"]
    experiment_plan_sha256: str
    run_manifest_sha256s: tuple[str, ...]
    artifact_record_sha256s: tuple[str, ...]
    payload_sha256: str
    size_bytes: int
    identity_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name not in ("index_path", "identity_sha256")
        }

    def __post_init__(self) -> None:
        if (
            not isinstance(self.index_path, Path)
            or self.index_path.name != "experiment-index.json"
            or self.stage not in ("pretest", "final")
            or type(self.run_manifest_sha256s) is not tuple
            or type(self.artifact_record_sha256s) is not tuple
            or type(self.size_bytes) is not int
            or self.size_bytes <= 0
        ):
            raise ValueError("figure experiment-index identity is invalid")
        for name in ("experiment_plan_sha256", "payload_sha256"):
            _sha256(getattr(self, name), name)
        for name in (
            "run_manifest_sha256s",
            "artifact_record_sha256s",
        ):
            values = getattr(self, name)
            for index, digest in enumerate(values):
                _sha256(digest, f"{name}[{index}]")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must be unique")
        expected = owned_sha256(
            "vfe4.wt103.experiment-index-identity.v1",
            self.semantic_payload(),
        )
        _sha256(self.identity_sha256, "identity_sha256")
        if self.identity_sha256 != expected:
            raise ValueError(
                "identity_sha256 does not match figure experiment index"
            )


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

    @classmethod
    def create(cls, **values: object) -> "FigureInputIdentity":
        payload = {
            "schema_version": "wt103-figure-input-identity-v1",
            **values,
        }
        return cls(
            **payload,
            input_sha256=owned_sha256(
                "vfe4.wt103.figure-input-identity.v1",
                payload,
            ),
        )  # type: ignore[arg-type]


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

    @classmethod
    def create(cls, **values: object) -> "FigureOutputIdentity":
        payload = {
            "schema_version": "wt103-figure-output-identity-v1",
            **values,
        }
        return cls(
            **payload,
            output_sha256=owned_sha256(
                "vfe4.wt103.figure-output-identity.v1",
                payload,
            ),
        )  # type: ignore[arg-type]


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

    @classmethod
    def create(cls, **values: object) -> "FigureSetManifest":
        payload = {
            "schema_version": "wt103-figure-set-manifest-v1",
            **values,
        }
        return cls(
            **payload,
            figure_set_sha256=owned_sha256(
                "vfe4.wt103.figure-set-manifest.v1",
                payload,
            ),
        )  # type: ignore[arg-type]


def figure_series_metric_names(
    series_key: str,
    arm: WT103ArmSpec,
) -> tuple[str, ...]:
    """Return the exact semantic columns plotted by one frozen series."""

    if type(arm) is not WT103ArmSpec:
        raise ValueError("arm must be an exact WT103ArmSpec")
    arm.__post_init__()
    suffix = series_key.rsplit("/", 1)[-1]
    if suffix == "train-objective":
        return {
            "cross_entropy": ("train_cross_entropy",),
            "complete_elbo": ("complete_elbo",),
            "emission_only_ablation_non_elbo": (
                "emission_only_non_elbo",
            ),
        }[arm.training_objective]
    if suffix == "validation-nll":
        return ("prior_nll_per_token", "perplexity")
    if suffix == "nll":
        return ("prior_nll_per_token",)
    if suffix == "ppl":
        return ("perplexity",)
    if suffix == "complete":
        return (
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
        )
    if suffix == "emission-non-elbo":
        return ("emission_only_non_elbo",)
    if suffix == "entropy":
        return ("source_entropy",)
    if suffix == "effective-count":
        return ("effective_source_count",)
    if suffix == "updates":
        return ("accepted_proposals", "rejected_proposals")
    if suffix == "health":
        return (
            "minimum_cholesky_pivot",
            "failed_pivots",
            "condition_estimate",
            "solve_residual",
            "damping_events",
            "spd_projections",
        )
    if suffix == "resources":
        phase_times: tuple[str, ...] = (
            "data_wait_seconds",
            "forward_seconds",
        )
        if arm.latent_enabled:
            phase_times += ("inference_seconds",)
        phase_times += (
            "backward_seconds",
            "update_seconds",
            "evaluation_seconds",
            "checkpoint_seconds",
            "wall_seconds",
        )
        return (
            "tokens_per_second",
            *phase_times,
            "process_rss_bytes",
            "process_hwm_bytes",
            "cuda_allocated_bytes",
            "cuda_reserved_bytes",
            "cuda_peak_allocated_bytes",
            "cuda_peak_reserved_bytes",
        )
    if suffix == "variability":
        objective = (
            ("complete_elbo",)
            if arm.training_objective == "complete_elbo"
            else (
                ("emission_only_non_elbo",)
                if arm.training_objective
                == "emission_only_ablation_non_elbo"
                else ()
            )
        )
        return (
            "prior_nll_per_token",
            "perplexity",
            *objective,
            "acceptance_rate",
            "cuda_peak_allocated_bytes",
        )
    raise ValueError(f"unknown frozen figure-series suffix {suffix!r}")


def figure_panel_applicability(
    panel_key: str,
    arm: WT103ArmSpec,
) -> tuple[Literal["applicable", "not_applicable"], str]:
    """Derive one panel/arm applicability cell and its canonical reason."""

    if panel_key not in WT103_FIGURE_PANEL_KEYS:
        raise ValueError("unknown frozen figure panel")
    if type(arm) is not WT103ArmSpec:
        raise ValueError("arm must be an exact WT103ArmSpec")
    arm.__post_init__()
    if panel_key == "complete-elbo-decomposition":
        if arm.training_objective in (
            "complete_elbo",
            "emission_only_ablation_non_elbo",
        ):
            return ("applicable", "arm_objective_has_decomposition_series")
        return (
            "not_applicable",
            "cross_entropy_objective_has_no_elbo_decomposition",
        )
    if panel_key in (
        "source-entropy-effective-count",
        "spd-health",
    ):
        if arm.latent_enabled:
            return ("applicable", "latent_path_active")
        return (
            "not_applicable",
            "arm_has_no_latent_or_recognition_path",
        )
    return ("applicable", "required_for_every_arm")


def _series_spec(
    series_key: str,
    arm: WT103ArmSpec,
    arm_index: int,
) -> FigureSeriesSpec:
    parts = series_key.split("/")
    metrics = figure_series_metric_names(series_key, arm)
    applicability, applicability_reason = figure_panel_applicability(
        parts[0],
        arm,
    )
    payload = {
        "schema_version": "wt103-figure-series-spec-v1",
        "series_key": series_key,
        "arm_id": parts[1],
        "source_columns": (
            "arm_id",
            "seed_id",
            "counted_targets",
            *metrics,
            "applicability",
            "applicability_reason",
        ),
        "aggregation": "frozen_metric_numerator_denominator_only",
        "uncertainty_interval": (
            "preregistered_estimator_or_seed_interval_as_applicable"
        ),
        "applicability": applicability,
        "applicability_reason": applicability_reason,
        "color": _OKABE_ITO[arm_index % len(_OKABE_ITO)],
        "marker": ("o", "s", "^", "D", "v")[arm_index % 5],
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
    arm_order = {
        arm.arm_id: index for index, arm in enumerate(inventory.arms)
    }
    arms = {arm.arm_id: arm for arm in inventory.arms}
    for figure_id in WT103_FIGURE_PANEL_KEYS:
        prefix = f"{figure_id}/"
        series_keys = tuple(key for key in all_series if key.startswith(prefix))
        if not series_keys:
            raise ValueError(f"figure {figure_id} has no derived series")
        series = tuple(
            _series_spec(
                key,
                arms[key.split("/")[1]],
                arm_order[key.split("/")[1]],
            )
            for key in series_keys
        )
        if figure_id == "training-objective-and-validation":
            panel_definitions = (
                (
                    "training-objective",
                    (
                        "train_cross_entropy",
                        "complete_elbo",
                        "emission_only_non_elbo",
                    ),
                    "Training objective (nats/token)",
                    "nats_per_token",
                ),
                (
                    "validation-nll",
                    ("prior_nll_per_token",),
                    "Prior NLL (nats/token)",
                    "nats_per_token",
                ),
                (
                    "validation-perplexity",
                    ("perplexity",),
                    "Perplexity",
                    "perplexity",
                ),
            )
        elif figure_id == "terminal-prior-nll-ppl":
            panel_definitions = (
                (
                    "prior-nll",
                    ("prior_nll_per_token",),
                    "Prior NLL (nats/token)",
                    "nats_per_token",
                ),
                (
                    "perplexity",
                    ("perplexity",),
                    "Perplexity",
                    "perplexity",
                ),
            )
        elif figure_id == "complete-elbo-decomposition":
            panel_definitions = (
                (
                    "decomposition",
                    (
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
                    ),
                    "ELBO / emission objective (nats/token)",
                    "nats_per_token",
                ),
            )
        elif figure_id == "source-entropy-effective-count":
            panel_definitions = (
                (
                    "source-entropy",
                    ("source_entropy",),
                    "Source entropy (nats/source row)",
                    "nats_per_source_row",
                ),
                (
                    "effective-source-count",
                    ("effective_source_count",),
                    "Effective source count",
                    "effective_sources",
                ),
            )
        elif figure_id == "update-acceptance":
            panel_definitions = (
                (
                    "proposal-counts",
                    ("accepted_proposals", "rejected_proposals"),
                    "Accepted/rejected proposals",
                    "proposals",
                ),
            )
        elif figure_id == "throughput-memory":
            panel_definitions = (
                (
                    "throughput",
                    ("tokens_per_second",),
                    "Throughput",
                    "tokens_per_second",
                ),
                (
                    "phase-time",
                    (
                        "data_wait_seconds",
                        "forward_seconds",
                        "inference_seconds",
                        "backward_seconds",
                        "update_seconds",
                        "evaluation_seconds",
                        "checkpoint_seconds",
                        "wall_seconds",
                    ),
                    "Phase time",
                    "seconds",
                ),
                (
                    "memory",
                    (
                        "process_rss_bytes",
                        "process_hwm_bytes",
                        "cuda_allocated_bytes",
                        "cuda_reserved_bytes",
                        "cuda_peak_allocated_bytes",
                        "cuda_peak_reserved_bytes",
                    ),
                    "Host/device memory",
                    "bytes",
                ),
            )
        elif figure_id == "seed-variability":
            panel_definitions = (
                (
                    "prior-nll",
                    ("prior_nll_per_token",),
                    "Prior NLL (nats/token)",
                    "nats_per_token",
                ),
                (
                    "perplexity",
                    ("perplexity",),
                    "Perplexity",
                    "perplexity",
                ),
                (
                    "objective",
                    ("complete_elbo", "emission_only_non_elbo"),
                    "Objective (nats/token)",
                    "nats_per_token",
                ),
                (
                    "acceptance-rate",
                    ("acceptance_rate",),
                    "Acceptance rate",
                    "fraction",
                ),
                (
                    "peak-allocated-bytes",
                    ("cuda_peak_allocated_bytes",),
                    "Peak allocated bytes",
                    "bytes",
                ),
            )
        elif figure_id == "spd-health":
            panel_definitions = (
                (
                    "minimum-cholesky-pivot",
                    ("minimum_cholesky_pivot",),
                    "Minimum Cholesky pivot",
                    "scalar",
                ),
                (
                    "failed-pivots",
                    ("failed_pivots",),
                    "Failed pivots",
                    "count",
                ),
                (
                    "condition-estimate",
                    ("condition_estimate",),
                    "Condition estimate",
                    "ratio",
                ),
                (
                    "solve-residual",
                    ("solve_residual",),
                    "Solve residual",
                    "scalar",
                ),
                (
                    "damping-events",
                    ("damping_events",),
                    "Damping events",
                    "count",
                ),
                (
                    "spd-projections",
                    ("spd_projections",),
                    "SPD projections",
                    "count",
                ),
            )
        else:
            raise ValueError(f"unknown frozen figure {figure_id!r}")
        panels = []
        for suffix, y_columns, y_label, units in panel_definitions:
            panel_payload = {
                "schema_version": "wt103-figure-panel-spec-v1",
                "panel_key": f"{figure_id}/{suffix}",
                "title": y_label,
                "x_column": (
                    "arm_id"
                    if figure_id == "seed-variability"
                    else "counted_targets"
                ),
                "y_columns": y_columns,
                "x_label": (
                    "Arm"
                    if figure_id == "seed-variability"
                    else "Counted training targets"
                ),
                "y_label": (
                    {
                        "throughput": "Tokens per second",
                        "phase-time": "Seconds",
                        "memory": "Bytes",
                    }[suffix]
                    if figure_id == "throughput-memory"
                    else y_label
                ),
                "units": units,
                "x_scale": "linear",
                "y_scale": "linear",
                "series_keys": series_keys,
            }
            panels.append(
                FigurePanelSpec(
                    **panel_payload,
                    panel_sha256=owned_sha256(
                        "vfe4.wt103.figure-panel-spec.v1",
                        panel_payload,
                    ),
                )
            )
        spec_payload = {
            "schema_version": "wt103-figure-spec-v1",
            "figure_id": figure_id,
            "panels": tuple(panels),
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
            "caption": _FIGURE_TEXT[figure_id][0],
            "alt_text": _FIGURE_TEXT[figure_id][1],
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


def _frozen_figure_provenance() -> FrozenFigureProvenance:
    inventory = EndpointInventory.create(
        default_wt103_arm_specs(),
        default_wt103_gate_specs(),
        WT103_TUNING_CELLS,
        (2026072199, 2026072200),
        tuple(range(2026072101, 2026072109)),
        EstimatorProtocol.create(),
    )
    spec_sha256s = tuple(
        spec.spec_sha256 for spec in default_figure_specs(inventory)
    )
    payload = {
        "schema_version": "wt103-frozen-figure-provenance-v1",
        "endpoint_inventory": inventory,
        "figure_spec_sha256s": spec_sha256s,
    }
    return FrozenFigureProvenance(
        **payload,
        provenance_sha256=owned_sha256(
            "vfe4.wt103.frozen-figure-provenance.v1",
            payload,
        ),
    )


WT103_FIGURE_PROVENANCE = _frozen_figure_provenance()


__all__ = [
    "FigureExperimentIndexIdentity",
    "FigureInputIdentity",
    "FigureOutputIdentity",
    "FigurePanelSpec",
    "FigureResultRow",
    "FigureSeriesSpec",
    "FigureSetManifest",
    "FigureSpec",
    "FrozenFigureProvenance",
    "WT103_FIGURE_PROVENANCE",
    "default_figure_specs",
    "figure_panel_applicability",
    "figure_series_metric_names",
]
