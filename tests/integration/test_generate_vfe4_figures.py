from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from test_support.wt103_figure_fakes import (
    FilesystemFigureBackend,
    finalized_figure_inputs,
)
from vfe4.figures import FigureInputError, render_figure_set


def _hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(root.iterdir())
        if path.is_file()
    }


def test_all_eight_figures_and_sidecars_are_byte_stable_across_roots(
    tmp_path: Path,
) -> None:
    inputs = finalized_figure_inputs()
    backend = FilesystemFigureBackend()

    first = render_figure_set(
        inputs=inputs,
        figure_root=tmp_path / "first",
        durability_backend=backend,
    )
    second = render_figure_set(
        inputs=inputs,
        figure_root=tmp_path / "second",
        durability_backend=backend,
    )

    assert first.manifest == second.manifest
    assert first.output_path.name == first.manifest.figure_set_sha256
    assert first.index_path.read_bytes() == second.index_path.read_bytes()
    assert len(first.manifest.outputs) == 8
    assert _hashes(first.output_path) == _hashes(second.output_path)
    assert len(_hashes(first.output_path)) == 8 * 7 + 1
    for output in first.manifest.outputs:
        prefix = output.figure_id
        assert (first.output_path / f"{prefix}.svg").is_file()
        assert (first.output_path / f"{prefix}.png").is_file()
        assert (first.output_path / f"{prefix}.pdf").is_file()
        assert (first.output_path / f"{prefix}.data.csv").is_file()
        assert (first.output_path / f"{prefix}.data.json").is_file()
        assert (first.output_path / f"{prefix}.caption.txt").is_file()
        assert (first.output_path / f"{prefix}.alt.txt").is_file()

    (first.output_path / "unexpected").mkdir()
    with pytest.raises(FigureInputError, match="contains non-files"):
        render_figure_set(
            inputs=inputs,
            figure_root=tmp_path / "first",
            durability_backend=backend,
        )


def test_renderer_requires_complete_durability_backend(tmp_path: Path) -> None:
    class CreateOnlyBackend:
        def create_exclusive(self, path: Path, payload: bytes) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)

    with pytest.raises(FigureInputError, match="replace_durable"):
        render_figure_set(
            inputs=finalized_figure_inputs(),
            figure_root=tmp_path / "figures",
            durability_backend=CreateOnlyBackend(),  # type: ignore[arg-type]
        )


def test_format_preflight_runs_before_experiment_index_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import generate_vfe4_figures as launcher

    config = copy.deepcopy(launcher.CONFIG)
    config["operation"] = "render"
    config["run_group_manifest_path"] = "not-yet-inspected"
    events: list[str] = []

    def reject_formats(_specs: object) -> None:
        events.append("format_preflight")
        raise FigureInputError("requested figure format is unsupported: pdf")

    monkeypatch.setattr(
        launcher,
        "preflight_figure_output_formats",
        reject_formats,
    )
    monkeypatch.setattr(
        launcher,
        "_explicit_existing_index",
        lambda _value: (_ for _ in ()).throw(
            AssertionError("experiment index was inspected before preflight")
        ),
    )
    with pytest.raises(FigureInputError, match="unsupported: pdf"):
        launcher.main(config)
    assert events == ["format_preflight"]
