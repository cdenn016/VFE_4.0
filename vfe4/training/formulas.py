"""Analytical parameter and semantic-FLOP ledgers for WT103 A0."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from vfe4.types.training import (
    A0ArchitectureProfile,
    A0FormulaRecord,
    WT103_A0_HIDDEN_WIDTH_CANDIDATES,
    owned_sha256,
)

from .wt103_models import WT103A0Model


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive exact int")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative exact int")
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
class NamedParameterShape:
    """One unique named parameter and its exact logical shape."""

    name: str
    shape: tuple[int, ...]
    numel: int

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise ValueError("parameter name must be nonempty text")
        if (
            type(self.shape) is not tuple
            or not self.shape
            or any(type(item) is not int or item <= 0 for item in self.shape)
        ):
            raise ValueError("parameter shape must be a positive int tuple")
        expected = math.prod(self.shape)
        if type(self.numel) is not int or self.numel != expected:
            raise ValueError("parameter numel does not match its shape")


@dataclass(frozen=True, slots=True)
class A0ParameterInventory:
    """Named-shape reconstruction of the complete A0 parameter set."""

    schema_version: Literal["wt103-a0-parameter-inventory-v1"]
    vocabulary_size: int
    positional_capacity: int
    hidden_width: int
    parameters: tuple[NamedParameterShape, ...]
    parameter_count: int
    inventory_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in tuple(self.__dataclass_fields__)[:-1]
        }

    def __post_init__(self) -> None:
        if self.schema_version != "wt103-a0-parameter-inventory-v1":
            raise ValueError("unsupported A0 parameter inventory schema")
        _positive_int(self.vocabulary_size, "vocabulary_size")
        _positive_int(self.positional_capacity, "positional_capacity")
        _positive_int(self.hidden_width, "hidden_width")
        if (
            type(self.parameters) is not tuple
            or not self.parameters
            or any(type(item) is not NamedParameterShape for item in self.parameters)
            or len({item.name for item in self.parameters}) != len(self.parameters)
        ):
            raise ValueError("A0 parameter rows must be unique exact records")
        expected_count = sum(item.numel for item in self.parameters)
        formula_count = (
            2 * self.vocabulary_size * self.hidden_width
            + self.positional_capacity * self.hidden_width
            + 12 * self.hidden_width**2
            + 15 * self.hidden_width
            + self.vocabulary_size
        )
        if (
            type(self.parameter_count) is not int
            or self.parameter_count != expected_count
            or self.parameter_count != formula_count
        ):
            raise ValueError("A0 parameter count does not match named shapes")
        expected = owned_sha256(
            "vfe4.wt103.a0-parameter-inventory.v1",
            self.semantic_payload(),
        )
        _sha256(self.inventory_sha256, "inventory_sha256")
        if self.inventory_sha256 != expected:
            raise ValueError("A0 parameter inventory hash does not match")

    @classmethod
    def create(
        cls,
        *,
        vocabulary_size: int,
        positional_capacity: int,
        hidden_width: int,
        parameters: tuple[NamedParameterShape, ...],
    ) -> "A0ParameterInventory":
        payload = {
            "schema_version": "wt103-a0-parameter-inventory-v1",
            "vocabulary_size": vocabulary_size,
            "positional_capacity": positional_capacity,
            "hidden_width": hidden_width,
            "parameters": parameters,
            "parameter_count": sum(item.numel for item in parameters),
        }
        return cls(
            **payload,
            inventory_sha256=owned_sha256(
                "vfe4.wt103.a0-parameter-inventory.v1",
                payload,
            ),
        )  # type: ignore[arg-type]


_A0_PARAMETER_SHAPES = (
    ("token_embedding.weight", lambda v, p, h: (v, h)),
    ("position_embedding.weight", lambda v, p, h: (p, h)),
    ("block.ln1.weight", lambda v, p, h: (h,)),
    ("block.ln1.bias", lambda v, p, h: (h,)),
    ("block.qkv.weight", lambda v, p, h: (3 * h, h)),
    ("block.qkv.bias", lambda v, p, h: (3 * h,)),
    ("block.attention_output.weight", lambda v, p, h: (h, h)),
    ("block.attention_output.bias", lambda v, p, h: (h,)),
    ("block.ln2.weight", lambda v, p, h: (h,)),
    ("block.ln2.bias", lambda v, p, h: (h,)),
    ("block.mlp_input.weight", lambda v, p, h: (4 * h, h)),
    ("block.mlp_input.bias", lambda v, p, h: (4 * h,)),
    ("block.mlp_output.weight", lambda v, p, h: (h, 4 * h)),
    ("block.mlp_output.bias", lambda v, p, h: (h,)),
    ("final_norm.weight", lambda v, p, h: (h,)),
    ("final_norm.bias", lambda v, p, h: (h,)),
    ("decoder.weight", lambda v, p, h: (v, h)),
    ("decoder.bias", lambda v, p, h: (v,)),
)


def reconstruct_a0_parameters(
    model: WT103A0Model,
    *,
    vocabulary_size: int,
    positional_capacity: int,
    hidden_width: int,
) -> A0ParameterInventory:
    """Reconstruct ``P_A0`` from the live module's exact named tensors."""

    if type(model) is not WT103A0Model:
        raise ValueError("model must be the exact WT103A0Model")
    for observed, expected, name in (
        (model.vocabulary_size, vocabulary_size, "vocabulary_size"),
        (model.positional_capacity, positional_capacity, "positional_capacity"),
        (model.hidden_width, hidden_width, "hidden_width"),
    ):
        if type(expected) is not int or observed != expected:
            raise ValueError(f"model {name} does not match reconstruction input")
    named = tuple(model.named_parameters())
    if len({id(parameter) for _, parameter in named}) != len(named):
        raise ValueError("A0 parameters must use unique live parameter objects")
    expected_shapes = tuple(
        (name, shape(vocabulary_size, positional_capacity, hidden_width))
        for name, shape in _A0_PARAMETER_SHAPES
    )
    observed_shapes = tuple(
        (name, tuple(int(item) for item in parameter.shape))
        for name, parameter in named
    )
    if observed_shapes != expected_shapes:
        raise ValueError("A0 named parameter shape inventory changed")
    return A0ParameterInventory.create(
        vocabulary_size=vocabulary_size,
        positional_capacity=positional_capacity,
        hidden_width=hidden_width,
        parameters=tuple(
            NamedParameterShape(
                name=name,
                shape=shape,
                numel=math.prod(shape),
            )
            for name, shape in observed_shapes
        ),
    )


@dataclass(frozen=True, slots=True)
class A0FlopWorkload:
    """Explicit semantic workload; decoder chunk size is storage-only."""

    batch_size: int
    sequence_length: int
    vocabulary_size: int
    hidden_width: int
    parameter_count: int
    decoder_chunk_size: int
    optimizer_steps: int
    validation_batches: int

    def __post_init__(self) -> None:
        for name in (
            "batch_size",
            "sequence_length",
            "vocabulary_size",
            "hidden_width",
            "parameter_count",
            "decoder_chunk_size",
            "optimizer_steps",
        ):
            _positive_int(getattr(self, name), name)
        _nonnegative_int(self.validation_batches, "validation_batches")
        if self.hidden_width % 2:
            raise ValueError("A0 hidden width must split across two heads")


@dataclass(frozen=True, slots=True)
class A0FlopTerm:
    name: str
    phase: Literal["forward", "backward", "adamw"]
    flops: int

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise ValueError("FLOP term name must be nonempty")
        if self.phase not in ("forward", "backward", "adamw"):
            raise ValueError("unknown A0 FLOP phase")
        _nonnegative_int(self.flops, "flops")


def a0_primitive_formula_sha256() -> str:
    """Canonical identity of the hand-enumerated primitive convention."""

    return owned_sha256(
        "vfe4.wt103.a0-primitive-flop-policy.v1",
        {
            "multiply": 1,
            "add_subtract_divide_exp_log_tanh_rsqrt": 1,
            "comparison_and_indexing": 0,
            "linear_forward_biased": "2*m*k*n+m*n",
            "linear_backward_biased": "4*m*k*n+m*n",
            "causal_qk_plus_av_forward": "4*B*pairs*h",
            "causal_qk_plus_av_backward": "8*B*pairs*h",
            "causal_softmax_forward_and_backward": "B*2*(4*pairs-L)",
            "layer_norm_forward": "B*L*(7*h+2)",
            "layer_norm_backward": "B*L*(12*h+5)",
            "gelu_tanh_forward": "9*B*L*4*h",
            "gelu_tanh_backward": "12*B*L*4*h",
            "adamw_per_parameter": "3+4+3+2+3",
        },
    )


@dataclass(frozen=True, slots=True)
class A0FlopLedger:
    schema_version: Literal["wt103-a0-semantic-train-flops-v1"]
    batch_size: int
    sequence_length: int
    vocabulary_size: int
    hidden_width: int
    parameter_count: int
    optimizer_steps: int
    validation_batches: int
    terms: tuple[A0FlopTerm, ...]
    forward_and_ce_flops_per_batch: int
    backward_flops_per_batch: int
    adamw_flops_per_step: int
    train_flops_per_step: int
    semantic_train_flops: int
    primitive_formula_sha256: str
    ledger_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in tuple(self.__dataclass_fields__)[:-1]
        }

    def __post_init__(self) -> None:
        if self.schema_version != "wt103-a0-semantic-train-flops-v1":
            raise ValueError("unsupported A0 FLOP ledger schema")
        for name in (
            "batch_size",
            "sequence_length",
            "vocabulary_size",
            "hidden_width",
            "parameter_count",
            "optimizer_steps",
            "forward_and_ce_flops_per_batch",
            "backward_flops_per_batch",
            "adamw_flops_per_step",
            "train_flops_per_step",
            "semantic_train_flops",
        ):
            _positive_int(getattr(self, name), name)
        _nonnegative_int(self.validation_batches, "validation_batches")
        if (
            type(self.terms) is not tuple
            or not self.terms
            or any(type(item) is not A0FlopTerm for item in self.terms)
            or len({item.name for item in self.terms}) != len(self.terms)
        ):
            raise ValueError("A0 FLOP terms must be unique exact records")
        forward = sum(
            item.flops for item in self.terms if item.phase == "forward"
        )
        backward = sum(
            item.flops for item in self.terms if item.phase == "backward"
        )
        adamw = sum(
            item.flops for item in self.terms if item.phase == "adamw"
        )
        train = forward + backward + adamw
        whole = (
            self.optimizer_steps * train
            + self.validation_batches * forward
        )
        if (
            self.forward_and_ce_flops_per_batch != forward
            or self.backward_flops_per_batch != backward
            or self.adamw_flops_per_step != adamw
            or self.train_flops_per_step != train
            or self.semantic_train_flops != whole
        ):
            raise ValueError("A0 FLOP aggregates do not match primitive terms")
        if self.primitive_formula_sha256 != a0_primitive_formula_sha256():
            raise ValueError("A0 primitive formula identity changed")
        expected = owned_sha256(
            "vfe4.wt103.a0-semantic-train-flops.v1",
            self.semantic_payload(),
        )
        _sha256(self.ledger_sha256, "ledger_sha256")
        if self.ledger_sha256 != expected:
            raise ValueError("A0 FLOP ledger hash does not match")

    @classmethod
    def create(
        cls,
        *,
        workload: A0FlopWorkload,
        terms: tuple[A0FlopTerm, ...],
    ) -> "A0FlopLedger":
        forward = sum(item.flops for item in terms if item.phase == "forward")
        backward = sum(item.flops for item in terms if item.phase == "backward")
        adamw = sum(item.flops for item in terms if item.phase == "adamw")
        train = forward + backward + adamw
        payload = {
            "schema_version": "wt103-a0-semantic-train-flops-v1",
            "batch_size": workload.batch_size,
            "sequence_length": workload.sequence_length,
            "vocabulary_size": workload.vocabulary_size,
            "hidden_width": workload.hidden_width,
            "parameter_count": workload.parameter_count,
            "optimizer_steps": workload.optimizer_steps,
            "validation_batches": workload.validation_batches,
            "terms": terms,
            "forward_and_ce_flops_per_batch": forward,
            "backward_flops_per_batch": backward,
            "adamw_flops_per_step": adamw,
            "train_flops_per_step": train,
            "semantic_train_flops": (
                workload.optimizer_steps * train
                + workload.validation_batches * forward
            ),
            "primitive_formula_sha256": a0_primitive_formula_sha256(),
        }
        return cls(
            **payload,
            ledger_sha256=owned_sha256(
                "vfe4.wt103.a0-semantic-train-flops.v1",
                payload,
            ),
        )  # type: ignore[arg-type]


def _linear_forward(
    *,
    rows: int,
    inputs: int,
    outputs: int,
) -> int:
    return 2 * rows * inputs * outputs + rows * outputs


def _linear_backward(
    *,
    rows: int,
    inputs: int,
    outputs: int,
) -> int:
    return 4 * rows * inputs * outputs + rows * outputs


def reconstruct_a0_flops(workload: A0FlopWorkload) -> A0FlopLedger:
    """Hand-enumerate one whole A0 schedule without profiler estimates."""

    if type(workload) is not A0FlopWorkload:
        raise ValueError("workload must be an exact A0FlopWorkload")
    workload.__post_init__()
    batch = workload.batch_size
    length = workload.sequence_length
    vocabulary = workload.vocabulary_size
    width = workload.hidden_width
    rows = batch * length
    pairs = length * (length + 1) // 2
    heads = 2
    terms = (
        A0FlopTerm("input_composition", "forward", rows * width),
        A0FlopTerm(
            "layer_norm_1_forward",
            "forward",
            rows * (7 * width + 2),
        ),
        A0FlopTerm(
            "qkv_linear_forward",
            "forward",
            _linear_forward(rows=rows, inputs=width, outputs=3 * width),
        ),
        A0FlopTerm(
            "causal_qk_plus_av_forward",
            "forward",
            4 * batch * pairs * width,
        ),
        A0FlopTerm(
            "causal_softmax_forward",
            "forward",
            batch * heads * (4 * pairs - length),
        ),
        A0FlopTerm(
            "attention_output_linear_forward",
            "forward",
            _linear_forward(rows=rows, inputs=width, outputs=width),
        ),
        A0FlopTerm("attention_residual_forward", "forward", rows * width),
        A0FlopTerm(
            "layer_norm_2_forward",
            "forward",
            rows * (7 * width + 2),
        ),
        A0FlopTerm(
            "mlp_input_linear_forward",
            "forward",
            _linear_forward(rows=rows, inputs=width, outputs=4 * width),
        ),
        A0FlopTerm(
            "gelu_tanh_forward",
            "forward",
            9 * rows * 4 * width,
        ),
        A0FlopTerm(
            "mlp_output_linear_forward",
            "forward",
            _linear_forward(rows=rows, inputs=4 * width, outputs=width),
        ),
        A0FlopTerm("mlp_residual_forward", "forward", rows * width),
        A0FlopTerm(
            "final_layer_norm_forward",
            "forward",
            rows * (7 * width + 2),
        ),
        A0FlopTerm(
            "decoder_linear_forward",
            "forward",
            _linear_forward(
                rows=rows,
                inputs=width,
                outputs=vocabulary,
            ),
        ),
        A0FlopTerm(
            "cross_entropy_forward",
            "forward",
            rows * (3 * vocabulary + 1),
        ),
        A0FlopTerm(
            "cross_entropy_backward",
            "backward",
            rows * 3 * vocabulary,
        ),
        A0FlopTerm(
            "decoder_linear_backward",
            "backward",
            _linear_backward(
                rows=rows,
                inputs=width,
                outputs=vocabulary,
            ),
        ),
        A0FlopTerm(
            "final_layer_norm_backward",
            "backward",
            rows * (12 * width + 5),
        ),
        A0FlopTerm("mlp_residual_backward", "backward", rows * width),
        A0FlopTerm(
            "mlp_output_linear_backward",
            "backward",
            _linear_backward(rows=rows, inputs=4 * width, outputs=width),
        ),
        A0FlopTerm(
            "gelu_tanh_backward",
            "backward",
            12 * rows * 4 * width,
        ),
        A0FlopTerm(
            "mlp_input_linear_backward",
            "backward",
            _linear_backward(rows=rows, inputs=width, outputs=4 * width),
        ),
        A0FlopTerm(
            "layer_norm_2_backward",
            "backward",
            rows * (12 * width + 5),
        ),
        A0FlopTerm("attention_residual_backward", "backward", rows * width),
        A0FlopTerm(
            "attention_output_linear_backward",
            "backward",
            _linear_backward(rows=rows, inputs=width, outputs=width),
        ),
        A0FlopTerm(
            "causal_qk_plus_av_backward",
            "backward",
            8 * batch * pairs * width,
        ),
        A0FlopTerm(
            "causal_softmax_backward",
            "backward",
            batch * heads * (4 * pairs - length),
        ),
        A0FlopTerm(
            "qkv_linear_backward",
            "backward",
            _linear_backward(rows=rows, inputs=width, outputs=3 * width),
        ),
        A0FlopTerm(
            "layer_norm_1_backward",
            "backward",
            rows * (12 * width + 5),
        ),
        A0FlopTerm(
            "embedding_scatter_backward",
            "backward",
            2 * rows * width,
        ),
        A0FlopTerm(
            "adamw_first_moment",
            "adamw",
            3 * workload.parameter_count,
        ),
        A0FlopTerm(
            "adamw_second_moment",
            "adamw",
            4 * workload.parameter_count,
        ),
        A0FlopTerm(
            "adamw_normalization",
            "adamw",
            3 * workload.parameter_count,
        ),
        A0FlopTerm(
            "adamw_parameter_step",
            "adamw",
            2 * workload.parameter_count,
        ),
        A0FlopTerm(
            "adamw_weight_decay",
            "adamw",
            3 * workload.parameter_count,
        ),
    )
    return A0FlopLedger.create(workload=workload, terms=terms)


def build_a0_formula_record(
    *,
    inventory: A0ParameterInventory,
    ledger: A0FlopLedger,
) -> A0FormulaRecord:
    """Bind production WT103 named shapes to the semantic whole-schedule FLOPs."""

    if type(inventory) is not A0ParameterInventory:
        raise ValueError("inventory must be an exact A0ParameterInventory")
    if type(ledger) is not A0FlopLedger:
        raise ValueError("ledger must be an exact A0FlopLedger")
    inventory.__post_init__()
    ledger.__post_init__()
    if (
        inventory.vocabulary_size != 50_257
        or inventory.positional_capacity != 128
        or inventory.hidden_width not in WT103_A0_HIDDEN_WIDTH_CANDIDATES
        or ledger.vocabulary_size != inventory.vocabulary_size
        or ledger.hidden_width != inventory.hidden_width
        or ledger.parameter_count != inventory.parameter_count
    ):
        raise ValueError("A0 formula record requires exact WT103 dimensions")
    payload = {
        "schema_version": "wt103-a0-formula-record-v1",
        "vocabulary_size": 50_257,
        "positional_capacity": 128,
        "hidden_width": inventory.hidden_width,
        "parameter_count": inventory.parameter_count,
        "semantic_train_flops": ledger.semantic_train_flops,
        "multiply_flops": 1,
        "add_subtract_divide_exp_log_tanh_rsqrt_flops": 1,
        "comparison_and_indexing_flops": 0,
    }
    return A0FormulaRecord(
        **payload,
        formula_sha256=owned_sha256(
            "vfe4.wt103.a0-formula-record.v1",
            payload,
        ),
    )


def build_a0_architecture_profile(
    *,
    hidden_width: int,
    formula: A0FormulaRecord,
    source_lock_scope: Literal[
        "candidate_unverified",
        "production_source_lock_verified",
    ],
    pytorch_version: str,
    sdpa_api_sha256: str,
    flash_backend_sha256: str,
) -> A0ArchitectureProfile:
    """Construct the exact A0 profile from explicit formula/runtime identities."""

    if type(formula) is not A0FormulaRecord:
        raise ValueError("formula must be an exact A0FormulaRecord")
    formula.__post_init__()
    if hidden_width != formula.hidden_width:
        raise ValueError("A0 architecture width and formula width disagree")
    return A0ArchitectureProfile.create(
        block_count=1,
        hidden_width=hidden_width,
        attention_heads=2,
        head_width=hidden_width // 2,
        attention_context="full_causal_inclusive_self",
        attention_allowed_keys="range(0,q+1)",
        attention_semantic_pair_count="L*(L+1)//2",
        attention_implementation=(
            "torch.nn.functional.scaled_dot_product_attention"
        ),
        attention_backend_policy="flash_attention_only_no_fallback",
        pytorch_sdpa_api_binding=(
            "torch.nn.attention.sdpa_kernel("
            "backends=[SDPBackend.FLASH_ATTENTION])"
        ),
        enabled_backends=("FLASH_ATTENTION",),
        alternative_backends_disabled=True,
        source_lock_scope=source_lock_scope,
        pytorch_version=pytorch_version,
        sdpa_api_sha256=sdpa_api_sha256,
        flash_backend_sha256=flash_backend_sha256,
        attention_is_causal=True,
        attention_mask_argument=None,
        attention_scale="1/sqrt(head_width)",
        attention_dropout_probability=0.0,
        attention_returns_weights=False,
        grouped_query_attention=False,
        backend_fallback_allowed=False,
        fused_full_attention_allowed=True,
        fused_attention_materialization="forbidden",
        token_embedding="learned[V,h]_no_bias",
        positional_encoding="learned_absolute",
        positional_capacity=128,
        position_interpolation=False,
        input_composition="token_embedding_plus_position_embedding",
        normalization=(
            "LayerNorm(eps=1e-5,elementwise_affine=true,bias=true)"
        ),
        normalization_placement="pre_norm_with_final_norm",
        residual_topology=(
            "x=x+attn(ln1(x));x=x+mlp(ln2(x));y=ln_f(x)"
        ),
        qkv_projection=(
            "Linear(in=h,out=3h,weight[3h,h],bias[3h])"
        ),
        attention_output_projection=(
            "Linear(in=h,out=h,weight[h,h],bias[h])"
        ),
        mlp_input_projection=(
            "Linear(in=h,out=4h,weight[4h,h],bias[4h])"
        ),
        activation="gelu_tanh_approximation",
        mlp_output_projection=(
            "Linear(in=4h,out=h,weight[h,4h],bias[h])"
        ),
        decoder_projection=(
            "untied_Linear(in=h,out=V,weight[V,h],bias[V])"
        ),
        all_dropout_probabilities=0.0,
        candidate_hidden_widths=WT103_A0_HIDDEN_WIDTH_CANDIDATES,
        parameter_relative_tolerance=0.01,
        flop_relative_tolerance=0.05,
        candidate_selection_key=(
            "abs_log_parameter_ratio_abs_log_flop_ratio_hidden_width"
        ),
        parameter_formula_schema="wt103-a0-parameter-formula-v1",
        flop_formula_schema="wt103-a0-semantic-train-flops-v1",
        formula_sha256=formula.formula_sha256,
    )


__all__ = [
    "A0FlopLedger",
    "A0FlopTerm",
    "A0FlopWorkload",
    "A0ParameterInventory",
    "NamedParameterShape",
    "a0_primitive_formula_sha256",
    "build_a0_architecture_profile",
    "build_a0_formula_record",
    "reconstruct_a0_flops",
    "reconstruct_a0_parameters",
]
