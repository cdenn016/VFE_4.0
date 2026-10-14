"""GL(K,R) defining representation and batched congruence operations."""

import torch


class GLReal:
    def __init__(self, K: int, device: torch.device | str = 'cpu', dtype: torch.dtype = torch.float64) -> None:
        self.K, self.device, self.dtype = K, device, dtype

    def identity(self, batch_shape: tuple[int, ...] = ()) -> torch.Tensor:
        return torch.eye(self.K, device=self.device, dtype=self.dtype).expand(*batch_shape, self.K, self.K)

    def compose(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return A @ B

    def ratio(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        r"""A B^{-1}, using a transposed solve instead of constructing B^{-1}."""
        X, info = torch.linalg.solve_ex(B.mT, A.mT, check_errors=False)
        if torch.any(info != 0):
            raise ValueError('Singular frame in group ratio')
        return X.mT

    def exp(self, omega: torch.Tensor) -> torch.Tensor:
        return torch.linalg.matrix_exp(omega)

    def representation(self, G: torch.Tensor, channel: str = 'belief') -> torch.Tensor:
        return G

    def algebra_basis(self) -> torch.Tensor:
        return torch.eye(self.K**2, device=self.device, dtype=self.dtype).reshape(self.K**2, self.K, self.K)

    def validate(self, G: torch.Tensor) -> bool:
        sign, logabsdet = torch.linalg.slogdet(G)
        return bool(torch.all(sign != 0) and torch.isfinite(logabsdet).all())

    def transport_covariance(self, Sigma: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        r"""Sigma' = G Sigma G^T (not diagonal projection)."""
        return G @ Sigma @ G.mT
