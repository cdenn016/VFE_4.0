"""Direct information-form assembly of frozen H1 generative factors."""

from __future__ import annotations

import torch

from vfe4.numerics.information import InformationGaussian
from vfe4.numerics.linear_gaussian import add_initial_gaussian, add_scalar_conditional
from vfe4.types.h1 import H1GenerativeFactorRecord
from vfe4.types.structural import SourcePath


def assemble_generative_information(
    factors: H1GenerativeFactorRecord, path: SourcePath
) -> InformationGaussian:
    """Assemble one normalized generative component directly in canonical form."""

    if not isinstance(factors, H1GenerativeFactorRecord):
        raise ValueError("factors must be an H1GenerativeFactorRecord")
    checked_path = _require_path(path)
    h = torch.zeros(6, dtype=torch.float64)
    J = torch.zeros((6, 6), dtype=torch.float64)
    add_initial_gaussian(
        h,
        J,
        (0, 1),
        factors.initial_joint.mean,
        factors.initial_joint.covariance,
    )
    for time in (1, 2):
        a = checked_path.a[time - 1]
        b = checked_path.b[time - 1]
        model_transition = factors.model_transitions[time - 1]
        state_transition = factors.state_transitions[time - 1]
        model_index = 2 * time + 1
        state_index = 2 * time
        add_scalar_conditional(
            h,
            J,
            model_index,
            ((2 * b + 1, model_transition.source_slopes[b]),),
            model_transition.offset,
            model_transition.variance,
        )
        add_scalar_conditional(
            h,
            J,
            state_index,
            (
                (2 * a, state_transition.source_slopes[a]),
                (model_index, state_transition.model_slope),
            ),
            state_transition.offset,
            state_transition.variance,
        )
    return InformationGaussian.from_information(h, J)


def _require_path(path: object) -> SourcePath:
    if (
        not isinstance(path, SourcePath)
        or path.a[0] != 0
        or path.b[0] != 0
        or path.a[1] not in (0, 1)
        or path.b[1] not in (0, 1)
    ):
        raise ValueError("path is outside the frozen H1 source support")
    return path
