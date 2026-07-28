"""Exact WikiText-103 A0 module and immutable arm build records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from vfe4.types.training import WT103ArmSpec, owned_sha256


ExecutionScope = Literal[
    "nonproduction_synthetic_smoke",
    "production_source_lock_verified",
]


def _text_tuple(
    value: object,
    name: str,
    *,
    allow_empty: bool,
) -> tuple[str, ...]:
    if (
        type(value) is not tuple
        or (not allow_empty and not value)
        or any(type(item) is not str or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"{name} must be a unique immutable text tuple")
    return value


def _sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class OptimizerParameterBinding:
    """Exact optimizer-to-parameter ownership."""

    optimizer_id: Literal["model_adamw", "recognition_adamw"]
    parameter_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.optimizer_id not in ("model_adamw", "recognition_adamw"):
            raise ValueError("optimizer_id is not a frozen WT103 optimizer")
        _text_tuple(
            self.parameter_names,
            "parameter_names",
            allow_empty=False,
        )


@dataclass(frozen=True, slots=True)
class WT103ArmRuntimeComponents:
    """Explicit runtime inventory supplied to one direct arm constructor."""

    model: nn.Module
    model_parameter_names: tuple[str, ...]
    latent_parameter_names: tuple[str, ...]
    source_parameter_names: tuple[str, ...]
    frame_parameter_names: tuple[str, ...]
    recognition_parameter_names: tuple[str, ...]
    optimizer_bindings: tuple[OptimizerParameterBinding, ...]
    filler_parameter_names: tuple[str, ...]
    dormant_parameter_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.model, nn.Module):
            raise ValueError("model must be a torch.nn.Module")
        inventories = (
            ("model_parameter_names", self.model_parameter_names),
            ("latent_parameter_names", self.latent_parameter_names),
            ("source_parameter_names", self.source_parameter_names),
            ("frame_parameter_names", self.frame_parameter_names),
            ("recognition_parameter_names", self.recognition_parameter_names),
        )
        for name, values in inventories:
            _text_tuple(values, name, allow_empty=True)
        _text_tuple(
            self.filler_parameter_names,
            "filler_parameter_names",
            allow_empty=True,
        )
        _text_tuple(
            self.dormant_parameter_names,
            "dormant_parameter_names",
            allow_empty=True,
        )
        if self.filler_parameter_names:
            raise ValueError("filler parameters are forbidden")
        if self.dormant_parameter_names:
            raise ValueError("dormant parameters are forbidden")

        actual = tuple(name for name, _ in self.model.named_parameters())
        if not actual:
            raise ValueError("arm runtime must expose active parameters")
        if len(set(actual)) != len(actual):
            raise ValueError("model parameter names must be unique")
        classified = tuple(
            item
            for _, inventory in inventories
            for item in inventory
        )
        if (
            len(set(classified)) != len(classified)
            or set(classified) != set(actual)
        ):
            raise ValueError(
                "every runtime parameter must have exactly one scientific role"
            )

        if (
            type(self.optimizer_bindings) is not tuple
            or not self.optimizer_bindings
            or any(
                type(item) is not OptimizerParameterBinding
                for item in self.optimizer_bindings
            )
            or len({item.optimizer_id for item in self.optimizer_bindings})
            != len(self.optimizer_bindings)
        ):
            raise ValueError("optimizer bindings must be unique exact records")
        optimizer_names = tuple(
            name
            for binding in self.optimizer_bindings
            for name in binding.parameter_names
        )
        if (
            len(set(optimizer_names)) != len(optimizer_names)
            or set(optimizer_names) != set(actual)
        ):
            raise ValueError(
                "every active parameter must enter an optimizer exactly once"
            )

    @classmethod
    def create(
        cls,
        *,
        model: nn.Module,
        model_parameter_names: tuple[str, ...],
        latent_parameter_names: tuple[str, ...],
        source_parameter_names: tuple[str, ...],
        frame_parameter_names: tuple[str, ...],
        recognition_parameter_names: tuple[str, ...],
        optimizer_bindings: tuple[OptimizerParameterBinding, ...],
        filler_parameter_names: tuple[str, ...],
        dormant_parameter_names: tuple[str, ...],
    ) -> "WT103ArmRuntimeComponents":
        return cls(
            model=model,
            model_parameter_names=model_parameter_names,
            latent_parameter_names=latent_parameter_names,
            source_parameter_names=source_parameter_names,
            frame_parameter_names=frame_parameter_names,
            recognition_parameter_names=recognition_parameter_names,
            optimizer_bindings=optimizer_bindings,
            filler_parameter_names=filler_parameter_names,
            dormant_parameter_names=dormant_parameter_names,
        )


class _A0Block(nn.Module):
    def __init__(
        self,
        *,
        hidden_width: int,
        layer_norm_epsilon: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(
            hidden_width,
            eps=layer_norm_epsilon,
            elementwise_affine=True,
            bias=True,
            device=device,
            dtype=dtype,
        )
        self.qkv = nn.Linear(
            hidden_width,
            3 * hidden_width,
            bias=True,
            device=device,
            dtype=dtype,
        )
        self.attention_output = nn.Linear(
            hidden_width,
            hidden_width,
            bias=True,
            device=device,
            dtype=dtype,
        )
        self.ln2 = nn.LayerNorm(
            hidden_width,
            eps=layer_norm_epsilon,
            elementwise_affine=True,
            bias=True,
            device=device,
            dtype=dtype,
        )
        self.mlp_input = nn.Linear(
            hidden_width,
            4 * hidden_width,
            bias=True,
            device=device,
            dtype=dtype,
        )
        self.mlp_output = nn.Linear(
            4 * hidden_width,
            hidden_width,
            bias=True,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        value: Tensor,
        *,
        attention_heads: int,
    ) -> Tensor:
        batch_size, sequence_length, hidden_width = value.shape
        head_width = hidden_width // attention_heads
        normalized = self.ln1(value)
        packed = self.qkv(normalized)
        query, key, carried = packed.chunk(3, dim=-1)

        def heads(tensor: Tensor) -> Tensor:
            return tensor.reshape(
                batch_size,
                sequence_length,
                attention_heads,
                head_width,
            ).transpose(1, 2)

        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            attended = F.scaled_dot_product_attention(
                heads(query),
                heads(key),
                heads(carried),
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
                scale=None,
                enable_gqa=False,
            )
        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .reshape(batch_size, sequence_length, hidden_width)
        )
        value = value + self.attention_output(attended)
        mlp_input = self.mlp_input(self.ln2(value))
        activated = F.gelu(mlp_input, approximate="tanh")
        return value + self.mlp_output(activated)


class WT103A0Model(nn.Module):
    """One-block, two-head, full-causal A0 with chunked decoding."""

    def __init__(
        self,
        *,
        vocabulary_size: int,
        positional_capacity: int,
        hidden_width: int,
        attention_heads: int,
        layer_norm_epsilon: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        for value, name in (
            (vocabulary_size, "vocabulary_size"),
            (positional_capacity, "positional_capacity"),
            (hidden_width, "hidden_width"),
            (attention_heads, "attention_heads"),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive exact int")
        if attention_heads != 2 or hidden_width % attention_heads:
            raise ValueError("A0 requires exactly two equal-width heads")
        if (
            type(layer_norm_epsilon) is not float
            or layer_norm_epsilon != 1.0e-5
        ):
            raise ValueError("A0 LayerNorm epsilon must be exactly 1e-5")
        if type(device) is not torch.device:
            raise ValueError("device must be an explicit torch.device")
        if type(dtype) is not torch.dtype:
            raise ValueError("dtype must be an explicit torch.dtype")

        self.vocabulary_size = vocabulary_size
        self.positional_capacity = positional_capacity
        self.hidden_width = hidden_width
        self.attention_heads = attention_heads
        self.token_embedding = nn.Embedding(
            vocabulary_size,
            hidden_width,
            device=device,
            dtype=dtype,
        )
        self.position_embedding = nn.Embedding(
            positional_capacity,
            hidden_width,
            device=device,
            dtype=dtype,
        )
        self.block = _A0Block(
            hidden_width=hidden_width,
            layer_norm_epsilon=layer_norm_epsilon,
            device=device,
            dtype=dtype,
        )
        self.final_norm = nn.LayerNorm(
            hidden_width,
            eps=layer_norm_epsilon,
            elementwise_affine=True,
            bias=True,
            device=device,
            dtype=dtype,
        )
        self.decoder = nn.Linear(
            hidden_width,
            vocabulary_size,
            bias=True,
            device=device,
            dtype=dtype,
        )

    def encode(self, input_ids: Tensor) -> Tensor:
        if (
            type(input_ids) is not Tensor
            or input_ids.ndim != 2
            or input_ids.shape[1] > self.positional_capacity
        ):
            raise ValueError(
                "input_ids must be [B,L] within positional capacity"
            )
        positions = torch.arange(
            input_ids.shape[1],
            device=input_ids.device,
            dtype=torch.int64,
        )
        value = self.token_embedding(input_ids) + self.position_embedding(
            positions
        ).unsqueeze(0)
        return self.final_norm(
            self.block(value, attention_heads=self.attention_heads)
        )

    def iter_decoder_logits(
        self,
        hidden: Tensor,
        *,
        decoder_chunk_size: int,
    ):
        if (
            type(hidden) is not Tensor
            or hidden.ndim != 3
            or hidden.shape[-1] != self.hidden_width
            or type(decoder_chunk_size) is not int
            or decoder_chunk_size <= 0
        ):
            raise ValueError("decoder inputs or explicit chunk size are invalid")
        flat = hidden.reshape(-1, self.hidden_width)
        for start in range(0, flat.shape[0], decoder_chunk_size):
            end = min(start + decoder_chunk_size, flat.shape[0])
            yield start, end, self.decoder(flat[start:end])


@dataclass(frozen=True, slots=True)
class WT103ArmBuildRecord:
    """Canonical constructor result without serializing the live module."""

    schema_version: Literal["wt103-arm-build-record-v1"]
    spec: WT103ArmSpec
    constructor_id: str
    execution_scope: ExecutionScope
    model_family_id: str
    training_objective: str
    scorer_kind: str
    update_phases: tuple[str, ...]
    model_parameter_names: tuple[str, ...]
    latent_parameter_names: tuple[str, ...]
    source_parameter_names: tuple[str, ...]
    frame_parameter_names: tuple[str, ...]
    recognition_parameter_names: tuple[str, ...]
    optimizer_bindings: tuple[OptimizerParameterBinding, ...]
    filler_parameter_names: tuple[str, ...]
    dormant_parameter_names: tuple[str, ...]
    architecture_sha256: str | None
    formula_sha256: str | None
    flop_ledger_sha256: str | None
    build_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in tuple(self.__dataclass_fields__)[:-1]
        }

    def __post_init__(self) -> None:
        if self.schema_version != "wt103-arm-build-record-v1":
            raise ValueError("unsupported WT103 arm build schema")
        if type(self.spec) is not WT103ArmSpec:
            raise ValueError("build record requires an exact WT103ArmSpec")
        self.spec.__post_init__()
        if self.execution_scope not in (
            "nonproduction_synthetic_smoke",
            "production_source_lock_verified",
        ):
            raise ValueError("unknown WT103 execution scope")
        if (
            self.training_objective != self.spec.training_objective
            or self.scorer_kind != self.spec.scorer_kind
            or self.update_phases != self.spec.update_phases
        ):
            raise ValueError("build record changed frozen arm semantics")
        for name in (
            "model_parameter_names",
            "latent_parameter_names",
            "source_parameter_names",
            "frame_parameter_names",
            "recognition_parameter_names",
            "filler_parameter_names",
            "dormant_parameter_names",
        ):
            _text_tuple(
                getattr(self, name),
                name,
                allow_empty=True,
            )
        if self.filler_parameter_names or self.dormant_parameter_names:
            raise ValueError("build records cannot contain filler/dormant state")
        if (
            type(self.optimizer_bindings) is not tuple
            or any(
                type(item) is not OptimizerParameterBinding
                for item in self.optimizer_bindings
            )
        ):
            raise ValueError("build optimizer bindings are invalid")
        for name in (
            "architecture_sha256",
            "formula_sha256",
            "flop_ledger_sha256",
        ):
            digest = getattr(self, name)
            if digest is not None:
                _sha256(digest, name)
        expected = owned_sha256(
            "vfe4.wt103.arm-build-record.v1",
            self.semantic_payload(),
        )
        _sha256(self.build_sha256, "build_sha256")
        if self.build_sha256 != expected:
            raise ValueError("build_sha256 does not match the build record")

    @classmethod
    def create(cls, **values: object) -> "WT103ArmBuildRecord":
        payload = {
            "schema_version": "wt103-arm-build-record-v1",
            **values,
        }
        return cls(
            **payload,
            build_sha256=owned_sha256(
                "vfe4.wt103.arm-build-record.v1",
                payload,
            ),
        )  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class BuiltWT103Arm:
    """Live runtime paired with its canonical construction record."""

    record: WT103ArmBuildRecord
    runtime: WT103ArmRuntimeComponents

    def __post_init__(self) -> None:
        if type(self.record) is not WT103ArmBuildRecord:
            raise ValueError("record must be an exact WT103ArmBuildRecord")
        if type(self.runtime) is not WT103ArmRuntimeComponents:
            raise ValueError(
                "runtime must be exact WT103ArmRuntimeComponents"
            )
        self.record.__post_init__()
        self.runtime.__post_init__()
        for name in (
            "model_parameter_names",
            "latent_parameter_names",
            "source_parameter_names",
            "frame_parameter_names",
            "recognition_parameter_names",
            "optimizer_bindings",
            "filler_parameter_names",
            "dormant_parameter_names",
        ):
            if getattr(self.record, name) != getattr(self.runtime, name):
                raise ValueError(
                    f"runtime and build record disagree on {name}"
                )


__all__ = [
    "BuiltWT103Arm",
    "ExecutionScope",
    "OptimizerParameterBinding",
    "WT103A0Model",
    "WT103ArmBuildRecord",
    "WT103ArmRuntimeComponents",
]
