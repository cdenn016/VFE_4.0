"""Authoritative tensor contracts for the finite design and recognition law."""

from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass
class Domain:
    contexts: torch.Tensor
    periods: torch.Tensor
    grid_shape: tuple[int, ...]
    cell_volume: torch.Tensor
    manifold: Any


@dataclass
class Incidence:
    agent_id: torch.Tensor
    context_id: torch.Tensor
    patch_offset: torch.Tensor
    local_node_id: torch.Tensor
    centers: torch.Tensor


@dataclass
class SourceSupport:
    receiver_agent: torch.Tensor
    source_agent: torch.Tensor
    context_id: torch.Tensor
    log_pi: torch.Tensor
    row_offset: torch.Tensor
    receiver_incidence: torch.Tensor
    source_incidence: torch.Tensor
    log_weights: torch.Tensor
    n_incidence: int
    incidence_agent: Optional[torch.Tensor] = None
    incidence_context: Optional[torch.Tensor] = None


@dataclass
class CandidateRows:
    row_offset: torch.Tensor
    receiver_incidence: torch.Tensor
    source_incidence: torch.Tensor
    copy_id: torch.Tensor
    log_pi: torch.Tensor
    row_id: torch.Tensor
    channel: torch.Tensor


@dataclass
class CopyTable:
    receiver_agent: torch.Tensor
    source_agent: torch.Tensor
    context_id: torch.Tensor
    primitive_id: torch.Tensor
    orientation: torch.Tensor
    is_self: torch.Tensor


@dataclass
class LoopTable:
    offset: torch.Tensor
    primitive_id: torch.Tensor
    orientation: torch.Tensor
    base_incidence: torch.Tensor
    weight: torch.Tensor


@dataclass
class GaussianBucket:
    agent_ids: torch.Tensor
    incidence_ids: torch.Tensor
    mu: torch.Tensor
    chol: torch.Tensor

    def covariance(self) -> torch.Tensor:
        if self.chol.ndim == 4:
            blocks = self.chol @ self.chol.mT
            B, P, K, _ = blocks.shape
            result = blocks.new_zeros(B, P, K, P, K)
            idx = torch.arange(P, device=blocks.device)
            # Separated advanced indices put the node axis before the batch axis.
            result[:, idx, :, idx, :] = blocks.transpose(0, 1)
            return result.reshape(B, P * K, P * K)
        return self.chol @ self.chol.mT


@dataclass
class PatchGaussians:
    buckets: tuple[GaussianBucket, ...]
    covariance_structure: str

    def marginals(self, n_incidence: int, K: int) -> tuple[torch.Tensor, torch.Tensor]:
        mu = self.buckets[0].mu.new_empty(n_incidence, K)
        covariance = mu.new_empty(n_incidence, K, K)
        for bucket in self.buckets:
            B, P = bucket.incidence_ids.shape
            mu[bucket.incidence_ids.flatten()] = bucket.mu.reshape(-1, K)
            if bucket.chol.ndim == 4:
                node_cov = bucket.chol @ bucket.chol.mT
            else:
                full = bucket.covariance().reshape(B, P, K, P, K)
                node_cov = full.diagonal(dim1=1, dim2=3).permute(0, 3, 1, 2)
            covariance[bucket.incidence_ids.flatten()] = node_cov.reshape(-1, K, K)
        return mu, covariance


@dataclass
class LinkState:
    mode: str
    omega_mean: torch.Tensor
    precision: torch.Tensor
    rng_state: torch.Tensor
    standard_normals: Optional[torch.Tensor] = None
    diagnostic_rng_state: Optional[torch.Tensor] = None


@dataclass
class SectionState:
    belief: PatchGaussians
    model: PatchGaussians
    beta: torch.Tensor
    log_pi: torch.Tensor
    links: LinkState
    sweep: int = 0

