from pathlib import Path

import torch

from vfe4.config import load_config
from vfe4.presets import resolve_preset
from vfe4.simulation import run_simulation, resume_simulation


def test_resume_matches_uninterrupted(tmp_path: Path) -> None:
    full = load_config(resolve_preset('tiny_point', {'output': {'directory': str(tmp_path / 'full'), 'figures': False}}))
    split = load_config(resolve_preset('tiny_point', {
        'inference': {'sweeps': 2}, 'output': {'directory': str(tmp_path / 'split'), 'figures': False}}))
    a, b = run_simulation(full), run_simulation(split)
    resumed = resume_simulation(b.run_directory / 'final_state.pt', 2)
    assert resumed.final_state.sweep == 4
    for lhs, rhs in zip(a.final_state.belief.buckets, resumed.final_state.belief.buckets):
        assert torch.equal(lhs.mu, rhs.mu)
        assert torch.equal(lhs.chol, rhs.chol)
    assert torch.equal(a.final_state.beta, resumed.final_state.beta)

