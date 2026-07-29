"""Eight explicit, deterministic WikiText-103 plot functions."""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable
from typing import TYPE_CHECKING

from vfe4.types.figures import FigureSeriesSpec, FigureSpec

from .load import FigureInputError, FigurePoint, FinalResultTable

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


_POINTWISE_T_CRITICAL_DF7 = 2.364624251592784
_LINE_STYLES = ("-", "--", "-.", ":")


def _pyplot():
    import matplotlib

    if matplotlib.__version__ != "3.10.6":
        raise FigureInputError(
            "figure rendering requires matplotlib==3.10.6"
        )
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _series_points(
    table: FinalResultTable,
    series_key: str,
    *,
    metric_filter: Callable[[str], bool] | None = None,
) -> tuple[FigurePoint, ...]:
    points = tuple(
        point
        for point in table.points
        if point.series_key == series_key
        and (
            metric_filter is None
            or metric_filter(point.metric_name)
        )
    )
    if not points:
        raise FigureInputError(
            f"final result table lacks plotted points for {series_key}"
        )
    return points


def _draw_series(
    ax: "Axes",
    *,
    series: FigureSeriesSpec,
    points: tuple[FigurePoint, ...],
    expected_units: str,
) -> None:
    metric_groups = tuple(
        dict.fromkeys((point.metric_name, point.units) for point in points)
    )
    if (
        not metric_groups
        or any(units != expected_units for _name, units in metric_groups)
        or len({units for _name, units in metric_groups}) != 1
    ):
        raise FigureInputError(
            f"plotted series units differ from axis spec: "
            f"{series.series_key}"
        )
    for metric_index, (metric_name, _units) in enumerate(metric_groups):
        metric_points = tuple(
            point for point in points if point.metric_name == metric_name
        )
        seed_ids = tuple(dict.fromkeys(point.seed_id for point in metric_points))
        by_seed = {
            seed: tuple(
                point for point in metric_points if point.seed_id == seed
            )
            for seed in seed_ids
        }
        x_inventory = tuple(
            point.counted_targets for point in by_seed[seed_ids[0]]
        )
        if any(
            tuple(point.counted_targets for point in trace) != x_inventory
            for trace in by_seed.values()
        ):
            raise FigureInputError(
                f"seed traces use different x coordinates for {series.series_key}"
            )
        style = _LINE_STYLES[metric_index % len(_LINE_STYLES)]
        for trace in by_seed.values():
            ax.plot(
                x_inventory,
                tuple(point.value for point in trace),
                color=series.color,
                linestyle=style,
                linewidth=0.7,
                alpha=0.28,
                marker=series.marker,
                markersize=2.2,
                label="_nolegend_",
            )
            endpoint = trace[-1]
            if endpoint.lower is not None and endpoint.upper is not None:
                ax.errorbar(
                    (endpoint.counted_targets,),
                    (endpoint.value,),
                    yerr=(
                        (endpoint.value - endpoint.lower,),
                        (endpoint.upper - endpoint.value,),
                    ),
                    color=series.color,
                    fmt="none",
                    capsize=1.8,
                    elinewidth=0.6,
                    alpha=0.5,
                    label="_nolegend_",
                )
        means: list[float] = []
        half_widths: list[float] = []
        for x_value in x_inventory:
            values = tuple(
                next(
                    point.value
                    for point in trace
                    if point.counted_targets == x_value
                )
                for trace in by_seed.values()
            )
            mean = math.fsum(values) / len(values)
            half_width = (
                0.0
                if len(values) == 1
                else _POINTWISE_T_CRITICAL_DF7
                * statistics.stdev(values)
                / math.sqrt(len(values))
            )
            means.append(mean)
            half_widths.append(half_width)
        label = f"{series.arm_id} - {metric_name}"
        ax.plot(
            x_inventory,
            means,
            color=series.color,
            linestyle=style,
            linewidth=1.45,
            marker=series.marker,
            markersize=3.5,
            label=label,
        )
        ax.fill_between(
            x_inventory,
            tuple(
                mean - half
                for mean, half in zip(means, half_widths)
            ),
            tuple(
                mean + half
                for mean, half in zip(means, half_widths)
            ),
            color=series.color,
            alpha=0.12,
            linewidth=0.0,
        )


def _annotate_not_applicable(
    ax: "Axes",
    *,
    table: FinalResultTable,
    panel_key: str,
) -> None:
    rows = tuple(
        row
        for row in table.applicability_rows
        if row.panel_key == panel_key
        and row.applicability == "not_applicable"
    )
    if not rows:
        return
    text = "Not applicable:\n" + "\n".join(
        f"{row.arm_id}: {row.reason}" for row in rows
    )
    ax.text(
        1.01,
        0.02,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6,
        color="#333333",
    )


def _finish_axis(
    ax: "Axes",
    *,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            frameon=False,
            fontsize=5.5,
            loc="best",
            ncols=1,
        )


def _split_plot(
    spec: FigureSpec,
    table: FinalResultTable,
    *,
    left_title: str,
    right_title: str,
    left_filter: Callable[[str, str], bool],
    right_filter: Callable[[str, str], bool],
    left_ylabel: str,
    right_ylabel: str,
    left_metric_filter: Callable[[str], bool] | None = None,
    right_metric_filter: Callable[[str], bool] | None = None,
) -> "Figure":
    if (
        len(spec.panels) != 2
        or tuple(panel.title for panel in spec.panels)
        != (left_title, right_title)
        or tuple(panel.y_label for panel in spec.panels)
        != (left_ylabel, right_ylabel)
    ):
        raise FigureInputError(
            f"rendered axes differ from figure spec: {spec.figure_id}"
        )
    plt = _pyplot()
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for series in spec.series:
        if series.applicability == "not_applicable":
            continue
        if left_filter(series.series_key, series.arm_id):
            _draw_series(
                axes[0],
                series=series,
                points=_series_points(
                    table,
                    series.series_key,
                    metric_filter=left_metric_filter,
                ),
                expected_units=spec.panels[0].units,
            )
        if right_filter(series.series_key, series.arm_id):
            _draw_series(
                axes[1],
                series=series,
                points=_series_points(
                    table,
                    series.series_key,
                    metric_filter=right_metric_filter,
                ),
                expected_units=spec.panels[1].units,
            )
    _finish_axis(
        axes[0],
        title=left_title,
        x_label="Counted training targets",
        y_label=left_ylabel,
    )
    _finish_axis(
        axes[1],
        title=right_title,
        x_label="Counted training targets",
        y_label=right_ylabel,
    )
    _annotate_not_applicable(
        axes[1],
        table=table,
        panel_key=spec.figure_id,
    )
    axes[0].text(
        -0.13,
        1.06,
        "A",
        transform=axes[0].transAxes,
        fontweight="bold",
    )
    axes[1].text(
        -0.13,
        1.06,
        "B",
        transform=axes[1].transAxes,
        fontweight="bold",
    )
    figure.tight_layout()
    return figure


def _single_plot(
    spec: FigureSpec,
    table: FinalResultTable,
    *,
    y_label: str,
) -> "Figure":
    if len(spec.panels) != 1 or spec.panels[0].y_label != y_label:
        raise FigureInputError(
            f"rendered axis differs from figure spec: {spec.figure_id}"
        )
    plt = _pyplot()
    figure, ax = plt.subplots(figsize=(7.2, 3.6))
    for series in spec.series:
        if series.applicability == "not_applicable":
            continue
        _draw_series(
            ax,
            series=series,
            points=_series_points(table, series.series_key),
            expected_units=spec.panels[0].units,
        )
    _finish_axis(
        ax,
        title=spec.panels[0].title,
        x_label="Counted training targets",
        y_label=y_label,
    )
    _annotate_not_applicable(
        ax,
        table=table,
        panel_key=spec.figure_id,
    )
    ax.text(
        -0.08,
        1.04,
        "A",
        transform=ax.transAxes,
        fontweight="bold",
    )
    figure.tight_layout()
    return figure


def _triple_plot(
    spec: FigureSpec,
    table: FinalResultTable,
    *,
    titles: tuple[str, str, str],
    series_filters: tuple[
        Callable[[str, str], bool],
        Callable[[str, str], bool],
        Callable[[str, str], bool],
    ],
    metric_filters: tuple[
        Callable[[str], bool],
        Callable[[str], bool],
        Callable[[str], bool],
    ],
    y_labels: tuple[str, str, str],
) -> "Figure":
    if (
        len(spec.panels) != 3
        or tuple(panel.title for panel in spec.panels) != titles
        or tuple(panel.y_label for panel in spec.panels) != y_labels
    ):
        raise FigureInputError(
            f"rendered axes differ from figure spec: {spec.figure_id}"
        )
    plt = _pyplot()
    figure, axes = plt.subplots(1, 3, figsize=(10.8, 3.2))
    for series in spec.series:
        if series.applicability == "not_applicable":
            continue
        for axis_index, (
            axis,
            series_filter,
            metric_filter,
        ) in enumerate(zip(axes, series_filters, metric_filters)):
            if series_filter(series.series_key, series.arm_id):
                _draw_series(
                    axis,
                    series=series,
                    points=_series_points(
                        table,
                        series.series_key,
                        metric_filter=metric_filter,
                    ),
                    expected_units=spec.panels[axis_index].units,
                )
    for index, axis in enumerate(axes):
        _finish_axis(
            axis,
            title=titles[index],
            x_label="Counted training targets",
            y_label=y_labels[index],
        )
        axis.text(
            -0.13,
            1.06,
            "ABC"[index],
            transform=axis.transAxes,
            fontweight="bold",
        )
    _annotate_not_applicable(
        axes[-1],
        table=table,
        panel_key=spec.figure_id,
    )
    figure.tight_layout()
    return figure


def plot_training_objective_and_validation(
    spec: FigureSpec,
    table: FinalResultTable,
) -> "Figure":
    return _triple_plot(
        spec,
        table,
        titles=(
            "Training objective (nats/token)",
            "Prior NLL (nats/token)",
            "Perplexity",
        ),
        series_filters=(
            lambda key, _arm: key.endswith("/train-objective"),
            lambda key, _arm: key.endswith("/validation-nll"),
            lambda key, _arm: key.endswith("/validation-nll"),
        ),
        metric_filters=(
            lambda _name: True,
            lambda name: name == "prior_nll_per_token",
            lambda name: name == "perplexity",
        ),
        y_labels=(
            "Training objective (nats/token)",
            "Prior NLL (nats/token)",
            "Perplexity",
        ),
    )


def plot_terminal_prior_nll_ppl(
    spec: FigureSpec,
    table: FinalResultTable,
) -> "Figure":
    return _split_plot(
        spec,
        table,
        left_title="Prior NLL (nats/token)",
        right_title="Perplexity",
        left_filter=lambda key, _arm: key.endswith("/nll"),
        right_filter=lambda key, _arm: key.endswith("/ppl"),
        left_ylabel="Prior NLL (nats/token)",
        right_ylabel="Perplexity",
    )


def plot_complete_elbo_decomposition(
    spec: FigureSpec,
    table: FinalResultTable,
) -> "Figure":
    return _single_plot(
        spec,
        table,
        y_label="ELBO / emission objective (nats/token)",
    )


def plot_source_entropy_effective_count(
    spec: FigureSpec,
    table: FinalResultTable,
) -> "Figure":
    return _split_plot(
        spec,
        table,
        left_title="Source entropy (nats/source row)",
        right_title="Effective source count",
        left_filter=lambda key, _arm: key.endswith("/entropy"),
        right_filter=lambda key, _arm: key.endswith("/effective-count"),
        left_ylabel="Source entropy (nats/source row)",
        right_ylabel="Effective source count",
    )


def plot_update_acceptance(
    spec: FigureSpec,
    table: FinalResultTable,
) -> "Figure":
    return _single_plot(
        spec,
        table,
        y_label="Accepted/rejected proposals",
    )


def plot_spd_health(
    spec: FigureSpec,
    table: FinalResultTable,
) -> "Figure":
    plt = _pyplot()
    figure, axes = plt.subplots(
        len(spec.panels),
        1,
        figsize=(8.0, 16.0),
        squeeze=False,
    )
    for row, panel in enumerate(spec.panels):
        axis = axes[row][0]
        included = frozenset(panel.y_columns)
        for series in spec.series:
            if series.applicability == "not_applicable":
                continue
            _draw_series(
                axis,
                series=series,
                points=_series_points(
                    table,
                    series.series_key,
                    metric_filter=lambda name: name in included,
                ),
                expected_units=panel.units,
            )
        _finish_axis(
            axis,
            title=panel.title,
            x_label=panel.x_label,
            y_label=panel.y_label,
        )
    _annotate_not_applicable(
        axes[-1][0],
        table=table,
        panel_key=spec.figure_id,
    )
    figure.tight_layout()
    return figure


def plot_throughput_memory(
    spec: FigureSpec,
    table: FinalResultTable,
) -> "Figure":
    return _triple_plot(
        spec,
        table,
        titles=("Throughput", "Phase time", "Host/device memory"),
        series_filters=(
            lambda _key, _arm: True,
            lambda _key, _arm: True,
            lambda _key, _arm: True,
        ),
        metric_filters=(
            lambda name: name == "tokens_per_second",
            lambda name: name.endswith("_seconds"),
            lambda name: name.endswith("_bytes"),
        ),
        y_labels=(
            "Tokens per second",
            "Seconds",
            "Bytes",
        ),
    )


def plot_seed_variability(
    spec: FigureSpec,
    table: FinalResultTable,
) -> "Figure":
    plt = _pyplot()
    families = tuple(
        (panel, frozenset(panel.y_columns)) for panel in spec.panels
    )
    figure, axes = plt.subplots(
        len(families),
        1,
        figsize=(12.0, 15.0),
        squeeze=False,
    )
    for row, (panel, included_names) in enumerate(families):
        ax = axes[row][0]
        labels: list[str] = []
        category_index = 0
        for series in spec.series:
            if series.applicability == "not_applicable":
                continue
            points = _series_points(table, series.series_key)
            metric_names = tuple(
                name
                for name in dict.fromkeys(
                    point.metric_name for point in points
                )
                if name in included_names
            )
            for metric_name in metric_names:
                metric_points = tuple(
                    point
                    for point in points
                    if point.metric_name == metric_name
                )
                if {
                    point.units for point in metric_points
                } != {panel.units}:
                    raise FigureInputError(
                        "seed-variability units differ from axis spec"
                    )
                latest_by_seed: list[FigurePoint] = []
                for seed_id in tuple(
                    dict.fromkeys(
                        point.seed_id for point in metric_points
                    )
                ):
                    trace = tuple(
                        point
                        for point in metric_points
                        if point.seed_id == seed_id
                    )
                    latest_by_seed.append(
                        max(
                            trace,
                            key=lambda point: point.counted_targets,
                        )
                    )
                values = tuple(
                    point.value for point in latest_by_seed
                )
                offsets = tuple(
                    (index - (len(values) - 1) / 2.0) * 0.035
                    for index in range(len(values))
                )
                ax.scatter(
                    tuple(
                        category_index + offset for offset in offsets
                    ),
                    values,
                    color=series.color,
                    marker=series.marker,
                    s=18,
                    alpha=0.65,
                )
                mean = math.fsum(values) / len(values)
                half_width = (
                    0.0
                    if len(values) == 1
                    else _POINTWISE_T_CRITICAL_DF7
                    * statistics.stdev(values)
                    / math.sqrt(len(values))
                )
                ax.errorbar(
                    (category_index,),
                    (mean,),
                    yerr=((half_width,), (half_width,)),
                    color="#000000",
                    marker="_",
                    capsize=3,
                    linewidth=1.0,
                )
                labels.append(f"{series.arm_id}\n{metric_name}")
                category_index += 1
        ax.set_xticks(
            tuple(range(len(labels))),
            labels,
            rotation=25,
            ha="right",
        )
        _finish_axis(
            ax,
            title=panel.title,
            x_label=panel.x_label,
            y_label=panel.y_label,
        )
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    _annotate_not_applicable(
        axes[-1][0],
        table=table,
        panel_key=spec.figure_id,
    )
    figure.tight_layout()
    return figure


__all__ = [
    "plot_complete_elbo_decomposition",
    "plot_seed_variability",
    "plot_source_entropy_effective_count",
    "plot_spd_health",
    "plot_terminal_prior_nll_ppl",
    "plot_throughput_memory",
    "plot_training_objective_and_validation",
    "plot_update_acceptance",
]
