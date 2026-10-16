"""Saved-run inspection must not enter continuation or restore RNG streams."""

from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytest
import torch

from vfe4.config import SimulationConfig, load_config
from vfe4.inspection import load_inspection_checkpoint, open_saved_run
from vfe4.model import build_model, initialize_sections
from vfe4.presets import resolve_preset
from vfe4.recorder import RecorderState


def _checkpoint(path: Path, cuda_metadata: bool = False, empty_history: bool = False) -> Path:
    config = load_config(resolve_preset('tiny_spatial_2d', {
        'manifold': {'grid_shape': (2, 2)},
        'population': {'n_agents': 2, 'center_grid': (2, 1), 'patch_shape': (1, 1),
                       'coverage': 'allow_uncovered'}, 'bundle': {'K': 1}}))
    model = build_model(config)
    state = initialize_sections(model, config)
    state.sweep = 3
    rows = [] if empty_history else [{'sweep': 0, 'vfe_total': 12.5}, {'sweep': 2, 'vfe_total': 8.25}]
    data = {'model': model, 'state': state, 'recorder': RecorderState(rows),
            'cpu_rng': torch.zeros_like(torch.get_rng_state())}
    if cuda_metadata:
        raw = deepcopy(config.raw)
        raw['runtime']['device'] = 'cuda'
        model.config = SimulationConfig(raw, torch.device('cuda'), config.dtype)
        model.domain.manifold.device = torch.device('cuda')
        data['cuda_rng'] = torch.zeros_like(torch.get_rng_state())
    torch.save(data, path)
    return path


def _forbid(*args: object, **kwargs: object) -> None:
    raise AssertionError('Inspection entered an inference, RNG restoration or CUDA path')


def test_cpu_inspection_maps_cuda_metadata_without_rng_restoration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _checkpoint(tmp_path / 'final_state.pt', cuda_metadata=True)
    saved = path.read_bytes()
    rng = torch.get_rng_state().clone()
    monkeypatch.setattr(torch, 'set_rng_state', _forbid)
    monkeypatch.setattr(torch, 'use_deterministic_algorithms', _forbid)
    monkeypatch.setattr(torch.cuda, 'set_rng_state', _forbid)
    monkeypatch.setattr(torch.cuda, 'is_available', _forbid)
    model, state, recorder = load_inspection_checkpoint(path, 'cpu')
    assert model.config.device == torch.device('cpu')
    assert model.config['runtime']['device'] == 'cpu'
    assert model.domain.manifold.device == torch.device('cpu')
    assert model.frame_field.seed_frames.device.type == state.beta.device.type == 'cpu'
    assert model.truth.dtype == state.belief.buckets[0].mu.dtype == torch.float64
    assert state.sweep == 3 and len(recorder.rows) == 2
    assert torch.equal(rng, torch.get_rng_state()) and path.read_bytes() == saved
    assert torch.load(path, weights_only=False, map_location='cpu')['model'].config['runtime']['device'] == 'cuda'


def test_run_directory_selects_final_state(tmp_path: Path) -> None:
    _checkpoint(tmp_path / 'final_state.pt')
    _, state, _ = load_inspection_checkpoint(tmp_path)
    assert state.sweep == 3


@pytest.mark.parametrize('name', ['missing.pt', 'missing-directory/final_state.pt'])
def test_missing_checkpoint_is_reported(tmp_path: Path, name: str) -> None:
    with pytest.raises(FileNotFoundError):
        load_inspection_checkpoint(tmp_path / name)


def test_unsupported_display_device_is_rejected_before_loading(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='cpu or cuda'):
        load_inspection_checkpoint(tmp_path / 'unused.pt', 'meta')


def test_inspector_restores_exact_history_and_has_no_inference_controls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vfe4 import inference, objective, simulation
    path = _checkpoint(tmp_path / 'final_state.pt')
    rng = torch.get_rng_state().clone()
    for module, name in ((inference, 'step'), (objective, 'evaluate_vfe'), (simulation, 'resume_simulation')):
        monkeypatch.setattr(module, name, _forbid)
    viewer = open_saved_run(path, block=False)
    try:
        assert viewer.inspection_only and viewer.paused
        assert viewer.history == [(0, 12.5), (2, 8.25)]
        assert list(viewer.axes[0].lines[0].get_xdata()) == [0, 2]
        assert viewer.snapshot.sweep == 3 and not viewer.once
        assert not viewer.buttons[0].active and not viewer.buttons[1].active
        viewer.toggle(None)
        viewer.single(None)
        assert viewer.paused and not viewer.once
        viewer.component_input.set_val('0')
        viewer.context_input.set_val('2')
        assert viewer.history == [(0, 12.5), (2, 8.25)]
        assert 'saved' in viewer.figure._suptitle.get_text().lower()
        assert torch.equal(rng, torch.get_rng_state())
        viewer.stop(None)
        assert viewer.stopped and not plt.fignum_exists(viewer.figure.number)
    finally:
        plt.close(viewer.figure)


def test_empty_saved_history_does_not_evaluate_or_invent_vfe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vfe4 import objective
    path = _checkpoint(tmp_path / 'final_state.pt', empty_history=True)
    monkeypatch.setattr(objective, 'evaluate_vfe', _forbid)
    viewer = open_saved_run(path, block=False)
    try:
        assert viewer.history == [] and not viewer.axes[0].lines
        assert any('unavailable' in text.get_text().lower() for text in viewer.axes[0].texts)
    finally:
        plt.close(viewer.figure)


def test_launcher_forwards_literal_config_relative_to_its_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import run_viewer
    from vfe4 import inspection
    calls = []

    def capture(path: Path, device: str = 'cpu') -> None:
        calls.append((path, device))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_viewer, 'CONFIG', {'checkpoint': 'runs/example', 'device': 'cpu', 'cpu_threads': 2})
    monkeypatch.setattr(inspection, 'open_saved_run', capture)
    monkeypatch.setattr(matplotlib, 'use', lambda backend: None)
    monkeypatch.setattr(torch, 'set_num_threads', lambda count: calls.append(count))
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        monkeypatch.setenv(name, '1')
    run_viewer.main()
    assert calls == [2, (Path(run_viewer.__file__).resolve().parent / 'runs/example', 'cpu')]
    assert all(__import__('os').environ[name] == '2' for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'))


@pytest.mark.parametrize('threads', [0, 5, True])
def test_launcher_enforces_bounded_cpu_threads(threads: int, monkeypatch: pytest.MonkeyPatch) -> None:
    import run_viewer
    monkeypatch.setitem(run_viewer.CONFIG, 'cpu_threads', threads)
    with pytest.raises(ValueError, match='1 through 4'):
        run_viewer.main()
