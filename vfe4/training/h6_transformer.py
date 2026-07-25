"""Frozen one-block causal Transformer for the amended H6 A0 endpoint."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from vfe4.data.windows import CausalPrefix
from vfe4.types.h6 import ArmId, VocabularyIdentity


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _owned_hash(domain: str, payload: object) -> str:
    return hashlib.sha256(
        domain.encode("ascii") + b"\x00" + _canonical_bytes(payload)
    ).hexdigest()


def _deterministic_matrix(rows: int, columns: int, *, scale: float) -> Tensor:
    row = torch.arange(1, rows + 1, dtype=torch.float64).unsqueeze(1)
    column = torch.arange(1, columns + 1, dtype=torch.float64).unsqueeze(0)
    values = (
        torch.sin(0.017 * row * column)
        + torch.cos(0.031 * row + 0.043 * column)
    )
    values = values - values.mean()
    return scale * values / torch.max(torch.abs(values)).clamp_min(1.0e-12)


@dataclass(frozen=True, slots=True)
class H6A0ArchitectureProfile:
    """Exact architecture identity for the H6-scale A0 comparator."""

    schema_version: Literal["h6-a0-architecture-v2"] = (
        "h6-a0-architecture-v2"
    )
    vocabulary_size: Literal[258] = 258
    position_capacity: Literal[32] = 32
    hidden_width: Literal[52] = 52
    attention_heads: Literal[2] = 2
    head_width: Literal[26] = 26
    block_count: Literal[1] = 1
    architecture_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        expected = (
            "h6-a0-architecture-v2",
            258,
            32,
            52,
            2,
            26,
            1,
        )
        observed = (
            self.schema_version,
            self.vocabulary_size,
            self.position_capacity,
            self.hidden_width,
            self.attention_heads,
            self.head_width,
            self.block_count,
        )
        if observed != expected:
            raise ValueError("H6 A0 architecture must equal the frozen profile")
        if self.hidden_width != self.attention_heads * self.head_width:
            raise ValueError("H6 A0 requires two equal attention heads")
        payload = {
            name: getattr(self, name)
            for name in (
                "schema_version",
                "vocabulary_size",
                "position_capacity",
                "hidden_width",
                "attention_heads",
                "head_width",
                "block_count",
            )
        }
        object.__setattr__(
            self,
            "architecture_sha256",
            _owned_hash("vfe4.h6.a0-architecture.v2", payload),
        )

    @classmethod
    def create(cls) -> "H6A0ArchitectureProfile":
        return cls()


@dataclass(frozen=True, slots=True)
class H6A0ValidationProfile:
    """Explicitly bounded causal-Transformer projection for prefix checks."""

    vocabulary_size: int
    position_capacity: int
    hidden_width: int
    attention_heads: Literal[2] = 2
    head_width: int = field(init=False)
    block_count: Literal[1] = 1
    schema_version: Literal["h6-a0-validation-projection-v2"] = (
        "h6-a0-validation-projection-v2"
    )
    architecture_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("vocabulary_size", "position_capacity", "hidden_width"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.vocabulary_size not in (3, 258):
            raise ValueError("H6 A0 validation projection requires V=3 or V=258")
        if self.position_capacity != 4:
            raise ValueError("H6 A0 validation projection requires horizon 4")
        if self.hidden_width not in (4, 48):
            raise ValueError(
                "H6 A0 validation projection requires hidden width 4 or 48"
            )
        if self.hidden_width % self.attention_heads:
            raise ValueError("H6 A0 requires two equal attention heads")
        object.__setattr__(
            self,
            "head_width",
            self.hidden_width // self.attention_heads,
        )
        payload = {
            name: getattr(self, name)
            for name in (
                "schema_version",
                "vocabulary_size",
                "position_capacity",
                "hidden_width",
                "attention_heads",
                "head_width",
                "block_count",
            )
        }
        object.__setattr__(
            self,
            "architecture_sha256",
            _owned_hash("vfe4.h6.a0-validation-projection.v2", payload),
        )

    @classmethod
    def create(
        cls,
        *,
        vocabulary_size: int,
        position_capacity: int,
        hidden_width: int,
    ) -> "H6A0ValidationProfile":
        return cls(
            vocabulary_size=vocabulary_size,
            position_capacity=position_capacity,
            hidden_width=hidden_width,
        )


H6TransformerProfile = H6A0ArchitectureProfile | H6A0ValidationProfile


class H6ScaledDotProductAttention(nn.Module):
    """Single CPU-float64 math-SDPA boundary owned by the H6 comparator."""

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        for tensor in (query, key, value):
            if tensor.device.type != "cpu" or tensor.dtype is not torch.float64:
                raise ValueError("H6 A0 SDPA requires CPU float64 tensors")
        with sdpa_kernel(SDPBackend.MATH):
            return F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=0.0,
                is_causal=True,
            )


class H6CausalTransformer(nn.Module):
    """One-block, two-head, pre-norm causal Transformer for H6 A0."""

    def __init__(
        self,
        *,
        vocabulary: VocabularyIdentity,
        profile: H6TransformerProfile,
    ) -> None:
        super().__init__()
        if type(vocabulary) is not VocabularyIdentity:
            raise ValueError("vocabulary must be an exact VocabularyIdentity")
        vocabulary.__post_init__()
        if type(profile) not in (
            H6A0ArchitectureProfile,
            H6A0ValidationProfile,
        ):
            raise ValueError(
                "profile must be an exact H6 Transformer profile"
            )
        profile.__post_init__()
        if vocabulary.size != profile.vocabulary_size:
            raise ValueError("vocabulary size does not match H6 A0 profile")
        family_label = "a0_causal_transformer"
        self.arm = ArmId.A0
        self.vocabulary = vocabulary
        self.profile = profile
        self.family_label = family_label
        self.horizon = profile.position_capacity
        self.emission_width = profile.hidden_width

        h = profile.hidden_width
        self.token_embedding = nn.Embedding(
            profile.vocabulary_size, h, dtype=torch.float64
        )
        self.position_embedding = nn.Embedding(
            profile.position_capacity, h, dtype=torch.float64
        )
        self.attention_norm = nn.LayerNorm(h, dtype=torch.float64)
        self.qkv_projection = nn.Linear(h, 3 * h, dtype=torch.float64)
        self.scaled_dot_product_attention = H6ScaledDotProductAttention()
        self.attention_output = nn.Linear(h, h, dtype=torch.float64)
        self.mlp_norm = nn.LayerNorm(h, dtype=torch.float64)
        self.mlp_input = nn.Linear(h, 4 * h, dtype=torch.float64)
        self.mlp_output = nn.Linear(4 * h, h, dtype=torch.float64)
        self.final_norm = nn.LayerNorm(h, dtype=torch.float64)
        self.decoder = nn.Linear(
            h, profile.vocabulary_size, dtype=torch.float64
        )
        self._initialize_deterministically()

        self.elbo_factor_inventory = ("emission",)
        self.elbo_inventory_sha256 = _owned_hash(
            "vfe4.h6.arm-elbo-inventory.v2",
            {
                "architecture_sha256": profile.architecture_sha256,
                "family": family_label,
                "partitions": self.elbo_factor_inventory,
            },
        )

    def _initialize_deterministically(self) -> None:
        matrices = (
            (self.token_embedding.weight, 0.125),
            (self.position_embedding.weight, 0.050),
            (self.qkv_projection.weight, 0.080),
            (self.attention_output.weight, 0.080),
            (self.mlp_input.weight, 0.080),
            (self.mlp_output.weight, 0.080),
            (self.decoder.weight, 0.100),
        )
        with torch.no_grad():
            for parameter, scale in matrices:
                parameter.copy_(
                    _deterministic_matrix(
                        parameter.shape[0],
                        parameter.shape[1],
                        scale=scale,
                    )
                )
            for module in (
                self.qkv_projection,
                self.attention_output,
                self.mlp_input,
                self.mlp_output,
                self.decoder,
            ):
                module.bias.zero_()

    def _exact_token_ids(self, token_ids: object) -> Tensor:
        if (
            type(token_ids) is not torch.Tensor
            or token_ids.device.type != "cpu"
            or token_ids.dtype is not torch.int64
            or token_ids.ndim != 1
            or not token_ids.is_contiguous()
        ):
            raise ValueError(
                "H6 A0 tokens must be contiguous CPU int64 rank one"
            )
        maximum_tokens = self.profile.position_capacity - 1
        if token_ids.numel() > maximum_tokens:
            raise ValueError(
                "H6 A0 prefix exceeds receiver horizon "
                f"{self.profile.position_capacity}"
            )
        if token_ids.numel() and (
            bool(torch.any(token_ids < 0).item())
            or bool(torch.any(token_ids >= self.vocabulary.size).item())
        ):
            raise ValueError("H6 A0 token IDs fall outside the vocabulary")
        return token_ids

    def sequence_log_probs(self, token_ids: Tensor) -> Tensor:
        """Return causal next-token rows for BOS and every supplied token."""

        checked = self._exact_token_ids(token_ids)
        if (
            self.token_embedding.weight.device.type != "cpu"
            or self.token_embedding.weight.dtype is not torch.float64
        ):
            raise ValueError("H6 A0 parameters must remain CPU float64")

        h = self.profile.hidden_width
        bos = torch.zeros(
            (1, h),
            dtype=torch.float64,
            device=self.token_embedding.weight.device,
        )
        token_rows = self.token_embedding(checked)
        hidden = torch.cat((bos, token_rows), dim=0)
        positions = torch.arange(
            hidden.shape[0],
            dtype=torch.int64,
            device=hidden.device,
        )
        hidden = hidden + self.position_embedding(positions)

        normalized = self.attention_norm(hidden)
        qkv = self.qkv_projection(normalized)
        qkv = qkv.reshape(
            hidden.shape[0],
            3,
            self.profile.attention_heads,
            self.profile.head_width,
        )
        query, key, value = (
            qkv[:, index].permute(1, 0, 2).unsqueeze(0)
            for index in range(3)
        )
        attended = self.scaled_dot_product_attention(query, key, value)
        attended = attended.squeeze(0).permute(1, 0, 2).reshape(
            hidden.shape[0], h
        )
        hidden = hidden + self.attention_output(attended)

        mlp_hidden = F.gelu(
            self.mlp_input(self.mlp_norm(hidden)),
            approximate="tanh",
        )
        hidden = hidden + self.mlp_output(mlp_hidden)
        logits = self.decoder(self.final_norm(hidden))
        return F.log_softmax(logits, dim=-1)

    def prefix_log_probs(self, prefix: CausalPrefix) -> Tensor:
        if type(prefix) is not CausalPrefix:
            raise ValueError("prefix must be an exact target-free CausalPrefix")
        prefix.__post_init__()
        if prefix.vocabulary != self.vocabulary:
            raise ValueError("causal prefix vocabulary does not match H6 A0")
        if prefix.receiver_t > self.profile.position_capacity:
            raise ValueError(
                "causal prefix exceeds receiver horizon "
                f"{self.profile.position_capacity}"
            )
        return self.sequence_log_probs(prefix.token_ids)[-1]


__all__ = [
    "H6A0ArchitectureProfile",
    "H6A0ValidationProfile",
    "H6CausalTransformer",
    "H6ScaledDotProductAttention",
]
