import pytest

from vfe4.config import load_config
from vfe4.presets import resolve_preset


def test_config_is_resolved_and_override_isolated() -> None:
    raw = resolve_preset('tiny_point', {'bundle': {'K': 3}})
    config = load_config(raw)
    assert config['bundle']['K'] == 3
    assert resolve_preset('tiny_point', {})['bundle']['K'] == 2
    assert str(config.device) == 'cpu'


def test_unknown_keys_fail() -> None:
    with pytest.raises(KeyError, match='populaton'):
        resolve_preset('tiny_point', {'populaton': {'n_agents': 4}})
    raw = resolve_preset('tiny_point', {})
    raw['inference']['typo'] = 2
    with pytest.raises(KeyError, match='inference.typo'):
        load_config(raw)


def test_explicit_cuda_has_no_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('torch.cuda.is_available', lambda: False)
    with pytest.raises(RuntimeError, match='CUDA'):
        load_config(resolve_preset('tiny_point', {'runtime': {'device': 'cuda'}}))


def test_dimension_and_population_validation() -> None:
    with pytest.raises(ValueError, match='dimension'):
        load_config(resolve_preset('tiny_spatial_2d', {'manifold': {'periods': (1.0,)}}))
    with pytest.raises(ValueError, match='lattice_shape'):
        load_config(resolve_preset('tiny_point', {'population': {'n_agents': 5}}))

