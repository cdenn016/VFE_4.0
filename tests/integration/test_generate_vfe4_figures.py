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


def test_figure_root_rejects_symlink_junction_or_reparse_before_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import generate_vfe4_figures as launcher

    experiment_root = tmp_path / "redirected-experiment"
    experiment_root.mkdir()
    figure_root = experiment_root / "figures"
    original_is_junction = getattr(Path, "is_junction", None)

    def is_junction(path: Path) -> bool:
        return path == experiment_root or bool(
            original_is_junction is not None
            and original_is_junction(path)
        )

    monkeypatch.setattr(Path, "is_junction", is_junction, raising=False)
    mkdir_calls: list[Path] = []
    original_mkdir = Path.mkdir

    def record_mkdir(path: Path, *args: object, **kwargs: object) -> None:
        if path == figure_root:
            mkdir_calls.append(path)
        original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", record_mkdir)
    with pytest.raises(FigureInputError, match="symlink|junction|reparse"):
        render_figure_set(
            inputs=finalized_figure_inputs(),
            figure_root=figure_root,
            durability_backend=FilesystemFigureBackend(),
        )

    events: list[str] = []

    class ProbeBackend:
        def probe(self, path: Path) -> None:
            events.append(f"probe:{path}")

    config = copy.deepcopy(launcher.CONFIG)
    config["operation"] = "render"
    config["run_group_manifest_path"] = str(
        experiment_root / "experiment-index.json"
    )
    config["figure_root"] = str(figure_root)
    monkeypatch.setattr(
        launcher,
        "_explicit_existing_index",
        lambda _value: experiment_root / "experiment-index.json",
    )
    monkeypatch.setattr(
        launcher,
        "load_figure_inputs",
        lambda **_kwargs: events.append("load"),
    )
    monkeypatch.setattr(
        launcher,
        "_platform_backend",
        lambda: ProbeBackend(),
    )
    monkeypatch.setattr(
        launcher,
        "render_figure_set",
        lambda **_kwargs: events.append("render"),
    )
    with pytest.raises(FigureInputError, match="symlink|junction|reparse"):
        launcher.main(config)

    assert mkdir_calls == []
    assert events == []


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
