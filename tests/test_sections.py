import torch

from vfe4.config import load_config
from vfe4.presets import resolve_preset
from vfe4.geometry.base import build_domain
from vfe4.geometry.sections import build_incidence, basis_weights, evaluate_gaussian_coefficients


def test_cross_node_variance() -> None:
    Sigma = torch.tensor([[2.0, 0.4], [0.4, 1.0]], dtype=torch.float64)
    _, result = evaluate_gaussian_coefficients(torch.zeros(2, dtype=Sigma.dtype),
        torch.linalg.cholesky(Sigma), torch.tensor([0.5, 0.5], dtype=Sigma.dtype), torch.eye(1, dtype=Sigma.dtype))
    torch.testing.assert_close(result[0, 0], torch.tensor(0.95, dtype=Sigma.dtype))


def test_sparse_incidence_and_cardinal_support() -> None:
    cfg = load_config(resolve_preset('tiny_spatial_2d', {}))
    domain = build_domain(cfg)
    incidence = build_incidence(domain, cfg)
    assert incidence.agent_id.numel() == 4
    weights, present = basis_weights(domain, incidence, torch.tensor([0, 0]),
        torch.tensor([[0.0, 0.0], [0.5, 0.5]], dtype=cfg.dtype))
    assert present.tolist() == [True, False]
    assert weights[0].tolist() == [1.0]
    assert weights[1].tolist() == [0.0]
    seam_weights, seam = basis_weights(domain, incidence, torch.tensor([0]),
        torch.tensor([[0.99, 0.0]], dtype=cfg.dtype))
    assert seam.item() and seam_weights.item() == 1.0

