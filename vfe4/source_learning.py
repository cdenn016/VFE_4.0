import math

import torch


def source_prior_step(log_pi: torch.Tensor, beta: torch.Tensor, row_offset: torch.Tensor,
                      rate: float) -> torch.Tensor:
    r"""pi'=(1-rate)pi+rate beta: c14 eq:phil-source-learning, a simplex Fisher step."""
    if not 0 <= rate <= 1:
        raise ValueError('Source learning rate must belong to [0,1]')
    if rate == 0:
        return log_pi
    sums = torch.segment_reduce(beta, 'sum', lengths=row_offset.diff())
    tolerance = 1e-8 if beta.dtype == torch.float64 else 1e-5
    if torch.any(beta < 0) or not torch.allclose(sums, torch.ones_like(sums), atol=tolerance, rtol=tolerance):
        raise ValueError('Source recognition rows must be normalized and nonnegative')
    result = beta.log() if rate == 1 else torch.logaddexp(log_pi + math.log1p(-rate), beta.log() + math.log(rate))
    if not torch.isfinite(result).all():
        raise ValueError('Source update loses finite positive support numerically; use rate below one')
    return result
