import torch

from vfe4.geometry.groups import GLReal


def test_ratios_and_covariance_reframing() -> None:
    group = GLReal(2)
    A = torch.tensor([[-1.0, 0.2], [0.0, 1.2]], dtype=torch.float64)
    B = torch.tensor([[1.1, 0.0], [0.3, 0.9]], dtype=A.dtype)
    C = torch.tensor([[0.7, 0.4], [0.0, 1.0]], dtype=A.dtype)
    torch.testing.assert_close(group.ratio(A, B) @ group.ratio(B, C), group.ratio(A, C))
    assert group.validate(A)
    Sigma = torch.tensor([[2.0, 0.3], [0.3, 1.0]], dtype=A.dtype)
    torch.testing.assert_close(group.transport_covariance(Sigma, A), A @ Sigma @ A.T)

