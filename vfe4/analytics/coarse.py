import torch

from ..gaussian import gaussian_kl


def discarded_conditional_kl(mu_q: torch.Tensor, Sigma_q: torch.Tensor, mu_p: torch.Tensor,
                            Sigma_p: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    r"""Delta=KL(q||Pi)-KL(qS||PiS), c09 thm:rg-exact-coarse-vfe for full-row-rank S.

    Requires the actual covered Gaussian posterior; a mixture or factorized
    substitute is not admitted. No clamping hides a negative numerical result.
    """
    return gaussian_kl(mu_q, Sigma_q, mu_p, Sigma_p) - gaussian_kl(
        S @ mu_q, S @ Sigma_q @ S.T, S @ mu_p, S @ Sigma_p @ S.T)
