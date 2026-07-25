"""Block-local construction of the normalized H8 recognition chain."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vfe4.generative.reference_h8 import H8Problem, validate_h8_problem
from vfe4.numerics.block_canonical import BlockCanonicalAssembler
from vfe4.numerics.block_tridiagonal import BlockTridiagonalCholesky
from vfe4.numerics.sparse_information import FactorBackedInformationGaussian


@dataclass(frozen=True, slots=True)
class H8RecognitionModel:
    """Factor-backed recognition law with no retained precision input."""

    gaussian: FactorBackedInformationGaussian
    input_sha256: str
    information_scalar_count: int
    factor_scalar_count: int


def build_h8_recognition(problem: H8Problem) -> H8RecognitionModel:
    problem = validate_h8_problem(problem)
    layout = problem.layout
    assembler = BlockCanonicalAssembler(layout)
    assembler.add_initial(_tensor(problem.recognition.initial_mean), _tensor(problem.recognition.initial_covariance))
    for transition in problem.recognition.transitions:
        if transition.source_support != (transition.parent_t,):
            raise ValueError("recognition source support must be singleton")
        assembler.add_transition(
            transition.receiver_t,
            _tensor(transition.matrix),
            _tensor(transition.offset),
            _tensor(transition.covariance),
        )
    precision, h = assembler.freeze()
    factor = BlockTridiagonalCholesky.factorize(precision)
    # The temporary precision is intentionally not stored in the returned model.
    gaussian = FactorBackedInformationGaussian.from_factor(h, factor)
    return H8RecognitionModel(
        gaussian=gaussian,
        input_sha256=problem.input_sha256,
        information_scalar_count=layout.information_scalar_count,
        factor_scalar_count=factor.storage.factor_scalar_count,
    )


assemble_h8_recognition = build_h8_recognition


def _tensor(value: object) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.float64, device="cpu")


__all__ = ["H8RecognitionModel", "assemble_h8_recognition", "build_h8_recognition"]
