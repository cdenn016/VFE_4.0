"""Verification-only PyTorch references that are never production exports."""

from verification.torch_references.h8_dense import (
    TorchDenseH8Result,
    TorchH8Objective,
    TorchH8ObjectiveTerm,
    TorchOperandMetadata,
    evaluate_h8_torch_dense,
)

__all__ = [
    "TorchDenseH8Result",
    "TorchH8Objective",
    "TorchH8ObjectiveTerm",
    "TorchOperandMetadata",
    "evaluate_h8_torch_dense",
]
