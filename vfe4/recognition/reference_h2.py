"""Direct information-form assembly of frozen H1 recognition factors."""

from __future__ import annotations

import torch

from vfe4.numerics.information import InformationGaussian
from vfe4.numerics.linear_gaussian import add_initial_gaussian, add_scalar_conditional
from vfe4.types.h1 import H1RecognitionFactorRecord
from vfe4.types.structural import SourcePath


def assemble_recognition_information(
    factors: H1RecognitionFactorRecord, path: SourcePath
) -> InformationGaussian:
    """Assemble one normalized recognition component directly in canonical form."""

    if not isinstance(factors, H1RecognitionFactorRecord):
        raise ValueError("factors must be an H1RecognitionFactorRecord")
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
        model_kernel = factors.model_kernels[time - 1]
        state_kernel = factors.state_kernels[time - 1]
        state_slot = 0 if time == 1 else a + 2 * b
        model_index = 2 * time + 1
        state_index = 2 * time
        add_scalar_conditional(
            h,
            J,
            model_index,
            ((2 * b + 1, model_kernel.slopes[b]),),
            model_kernel.offsets[b],
            model_kernel.variances[b],
        )
        add_scalar_conditional(
            h,
            J,
            state_index,
            (
                (2 * a, state_kernel.z_slopes[state_slot]),
                (model_index, state_kernel.m_slopes[state_slot]),
            ),
            state_kernel.offsets[state_slot],
            state_kernel.variances[state_slot],
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
