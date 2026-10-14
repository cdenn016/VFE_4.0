import torch

from vfe4.analytics.coherence import coherent_partition
from vfe4.analytics.holonomy import fixed_sector_rank
from vfe4.analytics.coarse import discarded_conditional_kl


def test_partition_and_fixed_sector() -> None:
    src, dst = torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3])
    labels = coherent_partition(torch.tensor([0.1, 0.9, 0.2]), src, dst, 0.5, 4)
    assert labels[0] == labels[1] and labels[2] == labels[3] and labels[1] != labels[2]
    I = torch.eye(2, dtype=torch.float64)
    R = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=I.dtype)
    src, dst = torch.tensor([1, 2, 0]), torch.tensor([0, 1, 2])
    assert fixed_sector_rank(torch.stack([I, I, I]), src, dst, 3) == 2
    assert fixed_sector_rank(torch.stack([I, I, R]), src, dst, 3) == 0


def test_conditional_kl_simple_independent_case() -> None:
    mu_q = torch.tensor([1.0, 2.0], dtype=torch.float64)
    mu_p = torch.zeros(2, dtype=mu_q.dtype)
    covariance = torch.eye(2, dtype=mu_q.dtype)
    S = torch.tensor([[1.0, 0.0]], dtype=mu_q.dtype)
    value = discarded_conditional_kl(mu_q, covariance, mu_p, covariance, S)
    torch.testing.assert_close(value, torch.tensor(2.0, dtype=mu_q.dtype))

