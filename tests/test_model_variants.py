from pathlib import Path

import torch

from vfe4.config import load_config
from vfe4.presets import resolve_preset
from vfe4.model import build_model, initialize_sections
from vfe4.inference import make_schedule, step, _channel_sweep
from vfe4.simulation import run_simulation, resume_simulation


def test_node_blocks_preserve_family() -> None:
    config = load_config(resolve_preset('tiny_spatial_2d', {
        'manifold': {'grid_shape': (3, 3)},
        'population': {'n_agents': 1, 'center_grid': (1, 1), 'patch_shape': (3, 3)},
        'recognition': {'covariance_structure': 'node_blocks'}, 'bundle': {'K': 2}}))
    model = build_model(config)
    state = initialize_sections(model, config)
    state, report = step(state, model, make_schedule(model, config), config)
    assert report.objective_after < report.objective_before
    assert state.belief.buckets[0].chol.ndim == 4


def test_full_patch_natural_coordinates() -> None:
    config = load_config(resolve_preset('tiny_point', {'inference': {'method': 'fisher_natural', 'belief_weight': 0.4}}))
    model = build_model(config)
    state = initialize_sections(model, config)
    state, report = step(state, model, make_schedule(model, config), config)
    assert report.objective_after < report.objective_before


def test_learned_rows_resume(tmp_path: Path) -> None:
    common = {'communication': {'source_rate': 0.1}, 'output': {'figures': False, 'directory': str(tmp_path)}}
    a = run_simulation(load_config(resolve_preset('tiny_point', common)))
    common['inference'] = {'sweeps': 2}
    b = run_simulation(load_config(resolve_preset('tiny_point', common)))
    resumed = resume_simulation(b.run_directory / 'final_state.pt', 2)
    assert torch.equal(a.final_state.log_pi, resumed.final_state.log_pi)
    assert torch.equal(a.final_state.beta, resumed.final_state.beta)


def test_sender_is_updated_without_own_receiving_record() -> None:
    from dataclasses import replace
    config = load_config(resolve_preset('tiny_point', {'population': {'n_agents': 2, 'lattice_shape': (2,)}}))
    model = build_model(config)
    state = initialize_sections(model, config)
    bank = next(b for b in model.records if b.kind == 'communication' and b.channel == 0)
    wanted = ((bank.receiver == 0) & (bank.sender == 1)).nonzero().flatten()
    selected = replace(bank, receiver=bank.receiver[wanted], sender=bank.sender[wanted],
        query=bank.query[wanted], key=bank.key[wanted], precision=bank.precision[wanted],
        candidate_id=bank.candidate_id[wanted], observed=torch.ones_like(bank.observed[wanted]))
    model.records = (selected,)
    _channel_sweep(state, model, make_schedule(model, config), config, 0, 1.0)
    assert torch.linalg.vector_norm(state.belief.buckets[0].mu[1]) > 0
