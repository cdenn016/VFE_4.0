import torch

from vfe4.config import load_config
from vfe4.presets import resolve_preset
from vfe4.geometry.base import FlatTorus, build_domain


def test_periodic_displacement_and_grid() -> None:
    torus = FlatTorus((1.0, 1.0), (4, 4))
    x = torch.tensor([[0.95, 0.5]], dtype=torch.float64)
    y = torch.tensor([[0.05, 0.5]], dtype=torch.float64)
    torch.testing.assert_close(torus.displacement(x, y), torch.tensor([[0.1, 0.0]], dtype=x.dtype))
    assert torus.grid().shape == (16, 2)
    assert (torus.grid() < 1).all()
    torch.testing.assert_close(torus.volume_weights().sum(), torch.tensor(1.0, dtype=x.dtype))


def test_point_and_three_dimensional_domains() -> None:
    point = build_domain(load_config(resolve_preset('tiny_point', {})))
    assert point.contexts.shape == (1, 0)
    assert point.cell_volume.tolist() == [1.0]
    domain = build_domain(load_config(resolve_preset('tiny_spatial_3d', {})))
    assert domain.contexts.shape == (64, 3)
    torch.testing.assert_close(domain.manifold.wrap(domain.contexts + 1), domain.contexts)
