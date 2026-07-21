from __future__ import annotations

import math

import pytest
import torch

from vfe4.numerics import probabilists_gauss_hermite


def test_probabilists_gauss_hermite_integrates_standard_normal_moments_through_six() -> None:
    nodes, weights = probabilists_gauss_hermite(7, dtype=torch.float64)

    expected = {0: 1.0, 1: 0.0, 2: 1.0, 3: 0.0, 4: 3.0, 5: 0.0, 6: 15.0}
    for degree, target in expected.items():
        actual = torch.sum(weights * nodes.pow(degree)).item()
        assert actual == pytest.approx(target, abs=1e-12)
    assert torch.sum(weights).item() == pytest.approx(1.0, abs=64 * math.ulp(1.0))


@pytest.mark.parametrize("order", [0, -1, True])
def test_probabilists_gauss_hermite_rejects_invalid_order(order: int) -> None:
    with pytest.raises(ValueError, match="order"):
        probabilists_gauss_hermite(order, dtype=torch.float64)


def test_probabilists_gauss_hermite_requires_float64() -> None:
    with pytest.raises(ValueError, match="float64"):
        probabilists_gauss_hermite(7, dtype=torch.float32)
