import torch

from vfe4.source_learning import source_prior_step


def test_source_prior_natural_step() -> None:
    log_pi = torch.tensor([0.5, 0.5], dtype=torch.float64).log()
    beta = torch.tensor([0.9, 0.1], dtype=log_pi.dtype)
    new = source_prior_step(log_pi, beta, torch.tensor([0, 2]), 0.1)
    torch.testing.assert_close(new.exp(), torch.tensor([0.54, 0.46], dtype=log_pi.dtype))
    assert (beta * (beta.log() - new)).sum() < (beta * (beta.log() - log_pi)).sum()
    assert torch.equal(source_prior_step(log_pi, beta, torch.tensor([0, 2]), 0.0), log_pi)
