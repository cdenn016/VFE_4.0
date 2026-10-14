"""Failed sweeps are rolled back, named, and never pooled as successful runs."""

from pathlib import Path
import json

import pytest
import torch

from vfe4.config import load_config
from vfe4.inference import make_schedule, step
from vfe4.model import build_model, initialize_sections
from vfe4.presets import resolve_preset
from vfe4.simulation import resume_simulation, run_simulation


def test_failed_sweep_restores_state_and_names_block() -> None:
    config = load_config(resolve_preset('tiny_point', {}))
    model = build_model(config)
    state = initialize_sections(model, config)
    schedule = make_schedule(model, config)
    state, _ = step(state, model, schedule, config)
    before = {name: getattr(state.belief.buckets[0], name).clone() for name in ('mu', 'chol')}
    beta, sweep = state.beta.clone(), state.sweep
    bank = next(b for b in model.records if b.kind == 'private')
    bank.observed = bank.observed.clone()
    bank.observed[0, 0] = float('inf')
    with pytest.raises((ValueError, RuntimeError)) as error:
        step(state, model, schedule, config)
    assert 'agents' in str(error.value) or 'descent' in str(error.value)
    assert state.sweep == sweep
    assert torch.equal(state.beta, beta)
    for name, value in before.items():
        assert torch.equal(getattr(state.belief.buckets[0], name), value)


def test_cholesky_failure_names_agent_and_channel() -> None:
    from vfe4.gaussian import checked_cholesky
    batch = torch.eye(2, dtype=torch.float64).repeat(3, 1, 1)
    batch[1, 0, 0] = float('inf')
    batch[2, 1, 1] = -1.0
    with pytest.raises(ValueError, match=r'model covariance update at agents \[7, 9\]'):
        checked_cholesky(batch, torch.tensor([5, 7, 9]), 'model covariance update')


def test_resume_reapplies_determinism_and_records_provenance(tmp_path: Path) -> None:
    raw = {'runtime': {'deterministic': True}, 'inference': {'sweeps': 2},
           'output': {'directory': str(tmp_path), 'figures': False}}
    first = run_simulation(load_config(resolve_preset('tiny_point', raw)))
    torch.use_deterministic_algorithms(False)
    resumed = resume_simulation(first.run_directory / 'final_state.pt', 1)
    assert torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)
    summary = json.loads((resumed.run_directory / 'summary.json').read_text())
    assert summary['resumed_from'].endswith('final_state.pt')
    structure = json.loads((resumed.run_directory / 'structure.json').read_text())
    assert structure['fixed_sector_rank_available'] == 1.0
