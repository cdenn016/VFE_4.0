from __future__ import annotations

import pytest
import torch

from vfe4.numerics import categorical_kl, require_probability_vector, selected_log_softmax


def test_probability_vector_requires_normalized_nonnegative_float64() -> None:
    value = torch.tensor([0.25, 0.75], dtype=torch.float64)

    result = require_probability_vector(value, name="q")

    value[0] = 0.5
    assert result.dtype is torch.float64
    assert result.tolist() == pytest.approx([0.25, 0.75])


@pytest.mark.parametrize(
    "value",
    [
        torch.tensor([0.2, 0.7], dtype=torch.float64),
        torch.tensor([1.1, -0.1], dtype=torch.float64),
        torch.tensor([[0.5, 0.5]], dtype=torch.float64),
        torch.tensor([0.5, 0.5], dtype=torch.float32),
    ],
)
def test_probability_vector_rejects_invalid_distribution(value: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="q"):
        require_probability_vector(value, name="q")


def test_categorical_kl_permits_zero_q_mass() -> None:
    q = torch.tensor([1.0, 0.0], dtype=torch.float64)
    p = torch.tensor([0.5, 0.5], dtype=torch.float64)

    result = categorical_kl(q, p, name="posterior")

    assert result.item() == pytest.approx(torch.log(torch.tensor(2.0)).item())


def test_categorical_kl_rejects_q_mass_outside_p_support() -> None:
    q = torch.tensor([0.5, 0.5], dtype=torch.float64)
    p = torch.tensor([1.0, 0.0], dtype=torch.float64)

    with pytest.raises(ValueError, match="posterior"):
        categorical_kl(q, p, name="posterior")


def test_selected_log_softmax_selects_the_requested_entry() -> None:
    logits = torch.tensor([1.0, 0.0, -1.0], dtype=torch.float64)

    result = selected_log_softmax(logits, 1)

    assert result.item() == pytest.approx(torch.log_softmax(logits, dim=0)[1].item())


@pytest.mark.parametrize("index", [-1, 3])
def test_selected_log_softmax_rejects_out_of_range_index(index: int) -> None:
    with pytest.raises(ValueError, match="index"):
        selected_log_softmax(torch.ones(3, dtype=torch.float64), index)
