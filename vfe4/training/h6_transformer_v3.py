"""CUDA-aware H6 v3 training Transformer with CPU-reference state names."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from vfe4.data.windows import CausalPrefix
from vfe4.types.h6 import VocabularyIdentity

from .h6_transformer import (
    H6CausalTransformer,
    H6TransformerProfile,
)


class H6TrainingScaledDotProductAttentionV3(nn.Module):
    """Math-only float64 SDPA for CUDA training or a bounded CPU fixture."""

    def __init__(self, *, allow_synthetic_cpu: bool = False) -> None:
        super().__init__()
        if type(allow_synthetic_cpu) is not bool:
            raise ValueError("allow_synthetic_cpu must be an exact bool")
        self._allow_synthetic_cpu = allow_synthetic_cpu

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        if any(type(tensor) is not Tensor for tensor in (query, key, value)):
            raise ValueError("H6 v3 SDPA inputs must be exact tensors")
        devices = {tensor.device for tensor in (query, key, value)}
        dtypes = {tensor.dtype for tensor in (query, key, value)}
        if len(devices) != 1 or dtypes != {torch.float64}:
            raise ValueError("H6 v3 SDPA inputs must share one float64 device")
        device = devices.pop()
        if device != torch.device("cuda:0") and not (
            self._allow_synthetic_cpu and device.type == "cpu"
        ):
            raise ValueError(
                "H6 v3 SDPA requires cuda:0, except bounded synthetic CPU tests"
            )
        with sdpa_kernel(SDPBackend.MATH):
            return F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=0.0,
                is_causal=True,
            )


class H6TrainingCausalTransformerV3(H6CausalTransformer):
    """Device-aware training copy of the strict CPU scoring Transformer."""

    def __init__(
        self,
        *,
        vocabulary: VocabularyIdentity,
        profile: H6TransformerProfile,
        allow_synthetic_cpu: bool = False,
    ) -> None:
        if type(allow_synthetic_cpu) is not bool:
            raise ValueError("allow_synthetic_cpu must be an exact bool")
        super().__init__(
            vocabulary=vocabulary,
            profile=profile,
        )
        self._allow_synthetic_cpu = allow_synthetic_cpu
        self.scaled_dot_product_attention = H6TrainingScaledDotProductAttentionV3(
            allow_synthetic_cpu=allow_synthetic_cpu,
        )

    def _training_device(self) -> torch.device:
        state = tuple(self.state_dict().values())
        if not state:
            raise ValueError("H6 v3 Transformer has no state")
        devices = {tensor.device for tensor in state}
        if len(devices) != 1:
            raise ValueError("H6 v3 Transformer state spans multiple devices")
        for tensor in state:
            if tensor.is_complex() or (
                tensor.is_floating_point() and tensor.dtype is not torch.float64
            ):
                raise ValueError("H6 v3 Transformer state must remain real float64")
        device = devices.pop()
        if device == torch.device("cuda:0"):
            return device
        if self._allow_synthetic_cpu and device.type == "cpu":
            return device
        raise ValueError("H6 v3 Transformer requires cuda:0 float64 in production")

    def _exact_token_ids(self, token_ids: object) -> Tensor:
        device = self._training_device()
        if (
            type(token_ids) is not Tensor
            or token_ids.device != device
            or token_ids.dtype is not torch.int64
            or token_ids.ndim != 1
            or not token_ids.is_contiguous()
        ):
            raise ValueError(
                "H6 v3 tokens must be contiguous int64 rank one on the training device"
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
        """Return causal next-token rows on the bound training device."""

        checked = self._exact_token_ids(token_ids)
        device = self._training_device()
        h = self.profile.hidden_width
        bos = torch.zeros(
            (1, h),
            dtype=torch.float64,
            device=device,
        )
        token_rows = self.token_embedding(checked)
        hidden = torch.cat((bos, token_rows), dim=0)
        positions = torch.arange(
            hidden.shape[0],
            dtype=torch.int64,
            device=device,
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
            qkv[:, index].permute(1, 0, 2).unsqueeze(0) for index in range(3)
        )
        attended = self.scaled_dot_product_attention(query, key, value)
        attended = (
            attended.squeeze(0)
            .permute(1, 0, 2)
            .reshape(
                hidden.shape[0],
                h,
            )
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
        token_ids = prefix.token_ids.to(
            device=self._training_device(),
            dtype=torch.int64,
        )
        return self.sequence_log_probs(token_ids)[-1]


__all__ = [
    "H6TrainingCausalTransformerV3",
    "H6TrainingScaledDotProductAttentionV3",
]
