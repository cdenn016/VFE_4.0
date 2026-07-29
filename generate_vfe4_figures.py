"""Click-to-run VFE4 figure regeneration from one immutable experiment index."""

from __future__ import annotations

import os
from pathlib import Path

from vfe4.artifacts.durability import (
    DurabilityBackend,
    PosixDurabilityBackend,
    WindowsDurabilityBackend,
)
from vfe4.config import (
    default_figure_config_mapping,
    resolve_figure_config,
)
from vfe4.figures import (
    FigureInputError,
    RenderedFigureSet,
    load_figure_inputs,
    preflight_figure_output_formats,
    render_figure_set,
    validate_figure_output_root,
)
from vfe4.types.figures import WT103_FIGURE_PROVENANCE


# Edit this dictionary, then click Run. Importing this module performs no I/O.
CONFIG: dict[str, object] = default_figure_config_mapping(
    WT103_FIGURE_PROVENANCE.endpoint_inventory
)


def _explicit_existing_index(value: str) -> Path:
    if any(character in value for character in "*?[]"):
        raise FigureInputError("run-group manifest path cannot contain a glob")
    path = Path(value)
    if (
        not path.is_absolute()
        or ".." in path.parts
        or path.name != "experiment-index.json"
    ):
        raise FigureInputError(
            "render requires one absolute explicit experiment-index.json path"
        )
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise FigureInputError(
            "the explicit experiment index is unavailable"
        ) from exc
    if resolved != path or not resolved.is_file():
        raise FigureInputError(
            "the experiment index must be a regular explicit path"
        )
    return resolved


def _explicit_figure_root(value: str, *, experiment_root: Path) -> Path:
    if any(character in value for character in "*?[]"):
        raise FigureInputError("figure root cannot contain a glob")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise FigureInputError("figure root must be one absolute explicit path")
    expected = experiment_root / "figures"
    if path != expected:
        raise FigureInputError(
            "figure root must be the experiment's exact figures directory"
        )
    return path


def _platform_backend() -> DurabilityBackend:
    if os.name == "nt":
        return WindowsDurabilityBackend()
    if os.name == "posix":
        return PosixDurabilityBackend()
    raise FigureInputError(
        f"no figure durability backend exists for platform {os.name!r}"
    )


def main(
    config: object = CONFIG,
    *,
    durability_backend: DurabilityBackend | None = None,
) -> RenderedFigureSet | None:
    """Resolve the editable dictionary and render only on explicit request."""

    resolved = resolve_figure_config(config)
    if resolved.operation == "idle":
        return None
    preflight_figure_output_formats(resolved.specs)
    index_path = _explicit_existing_index(
        resolved.run_group_manifest_path
    )
    figure_root = _explicit_figure_root(
        resolved.figure_root,
        experiment_root=index_path.parent,
    )
    validate_figure_output_root(figure_root)
    inputs = load_figure_inputs(
        run_group_manifest_path=index_path,
        inventory=WT103_FIGURE_PROVENANCE.endpoint_inventory,
        specs=resolved.specs,
    )
    backend = durability_backend
    if backend is None:
        figure_root.mkdir(parents=True, exist_ok=True)
        backend = _platform_backend()
        probe = backend.probe(figure_root)
        if probe.status != "pass":
            errors = tuple(
                f"{item.phase}:{item.exception_type}:{item.message}"
                for item in probe.errors
            )
            raise FigureInputError(
                "figure-root durability probe did not pass: "
                + ",".join((*errors, *probe.obligations))
            )
    rendered = render_figure_set(
        inputs=inputs,
        figure_root=figure_root,
        durability_backend=backend,
    )
    print(f"figure_set_sha256={rendered.manifest.figure_set_sha256}")
    print(f"figure_manifest={rendered.manifest_path}")
    return rendered


if __name__ == "__main__":
    main()
