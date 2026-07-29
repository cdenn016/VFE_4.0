"""Frozen WikiText-103 figure registry and exact inventory validation."""

from __future__ import annotations

from vfe4.types.figures import (
    FigureExperimentIndexIdentity,
    FigureInputIdentity,
    FigureOutputIdentity,
    FigurePanelSpec,
    FigureResultRow,
    FigureSeriesSpec,
    FigureSetManifest,
    FigureSpec,
    FrozenFigureProvenance,
    WT103_FIGURE_PROVENANCE,
    default_figure_specs,
    figure_panel_applicability,
    figure_series_metric_names,
)
from vfe4.types.training import (
    EndpointInventory,
    WT103_FIGURE_PANEL_KEYS,
)


REQUIRED_FIGURE_FUNCTIONS = (
    (
        "training-objective-and-validation",
        "plot_training_objective_and_validation",
    ),
    ("terminal-prior-nll-ppl", "plot_terminal_prior_nll_ppl"),
    (
        "complete-elbo-decomposition",
        "plot_complete_elbo_decomposition",
    ),
    (
        "source-entropy-effective-count",
        "plot_source_entropy_effective_count",
    ),
    ("update-acceptance", "plot_update_acceptance"),
    ("spd-health", "plot_spd_health"),
    ("throughput-memory", "plot_throughput_memory"),
    ("seed-variability", "plot_seed_variability"),
)


def validate_figure_registry(
    *,
    inventory: EndpointInventory,
    specs: tuple[FigureSpec, ...],
) -> tuple[FigureSpec, ...]:
    """Reject any panel, series, order, or semantic-spec drift."""

    if type(inventory) is not EndpointInventory:
        raise ValueError("inventory must be an exact EndpointInventory")
    inventory.__post_init__()
    if (
        type(specs) is not tuple
        or any(type(spec) is not FigureSpec for spec in specs)
    ):
        raise ValueError("specs must be an exact immutable FigureSpec tuple")
    expected = default_figure_specs(inventory)
    if specs != expected:
        raise ValueError(
            "figure specs differ from the inventory-derived frozen registry"
        )
    if (
        tuple(spec.figure_id for spec in specs)
        != WT103_FIGURE_PANEL_KEYS
        or tuple(name for name, _function in REQUIRED_FIGURE_FUNCTIONS)
        != inventory.figure_panel_keys
        or tuple(
            series.series_key
            for spec in specs
            for series in spec.series
        )
        != inventory.figure_series_keys
    ):
        raise ValueError("figure panel or series inventory is incomplete")
    return specs


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
    "REQUIRED_FIGURE_FUNCTIONS",
    "WT103_FIGURE_PROVENANCE",
    "default_figure_specs",
    "figure_panel_applicability",
    "figure_series_metric_names",
    "validate_figure_registry",
]
