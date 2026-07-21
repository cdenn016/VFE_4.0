from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from vfe4.types import SourcePath
from vfe4.validation import enumerate_source_paths, label_to_index, load_h1_fixture


def _raw_fixture() -> dict[str, object]:
    fixture_path = (
        Path(__file__).resolve().parents[2]
        / "vfe4"
        / "validation"
        / "fixtures"
        / "h1_v1.json"
    )
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def _write_fixture(tmp_path: Path, raw: dict[str, object]) -> Path:
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def test_loads_the_frozen_h1_fixture_and_exact_source_order() -> None:
    fixture = load_h1_fixture()

    assert fixture.fixture_schema_version == 1
    assert fixture.fixture_id == "h1-v1"
    assert fixture.continuous_order == ("z0", "m0", "z1", "m1", "z2", "m2")
    assert fixture.observation_labels == (1, 2)
    assert fixture.structural.state_parent_sets == ((0,), (0, 1))
    assert fixture.structural.model_parent_sets == ((0,), (0, 1))
    assert fixture.quadrature_order == 21
    assert fixture.convergence_check_order == 17
    assert fixture.maximum_convergence_estimate == 1e-9
    assert enumerate_source_paths(fixture) == (
        SourcePath(a=(0, 0), b=(0, 0)),
        SourcePath(a=(0, 1), b=(0, 0)),
        SourcePath(a=(0, 0), b=(0, 1)),
        SourcePath(a=(0, 1), b=(0, 1)),
    )


def test_frames_and_source_tables_are_exact_and_normalized() -> None:
    fixture = load_h1_fixture()

    assert fixture.frames.omega(1, 0).item() == pytest.approx(1.25)
    assert fixture.frames.omega(2, 0).item() == pytest.approx(0.8)
    assert fixture.frames.omega(2, 1).item() == pytest.approx(0.64)
    for row in (*fixture.model_source_priors, *fixture.state_source_priors):
        assert row.dtype is torch.float64
        assert row.sum().item() == pytest.approx(1.0)
    assert torch.equal(
        fixture.model_source_priors[1],
        torch.tensor([0.35, 0.65], dtype=torch.float64),
    )
    assert torch.equal(
        fixture.state_source_priors[1],
        torch.tensor([0.55, 0.45], dtype=torch.float64),
    )


@pytest.mark.parametrize(("label", "expected"), [(1, 0), (2, 1), (3, 2)])
def test_label_to_index_is_checked_one_based(label: int, expected: int) -> None:
    assert label_to_index(label) == expected


@pytest.mark.parametrize("label", [0, 4, True, 1.0])
def test_label_to_index_rejects_nonlabels(label: object) -> None:
    with pytest.raises(ValueError, match="label"):
        label_to_index(label)  # type: ignore[arg-type]


def test_fixture_tensor_accessors_do_not_alias_input_or_internal_storage() -> None:
    fixture = load_h1_fixture()

    prior = fixture.model_source_priors[1]
    prior[0] = 99.0
    mean = fixture.initial_joint.mean
    mean[0] = 99.0
    decoder = fixture.emissions[0].w_z
    decoder[0] = 99.0

    assert fixture.model_source_priors[1][0].item() == pytest.approx(0.35)
    assert fixture.initial_joint.mean[0].item() == pytest.approx(0.2)
    assert fixture.emissions[0].w_z[0].item() == pytest.approx(0.2)


def test_loader_rejects_malformed_json_and_unknown_fields(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON"):
        load_h1_fixture(malformed)

    raw = _raw_fixture()
    raw["unexpected"] = 1
    with pytest.raises(ValueError, match="fields"):
        load_h1_fixture(_write_fixture(tmp_path, raw))


def test_loader_rejects_non_spd_initial_covariance(tmp_path: Path) -> None:
    raw = _raw_fixture()
    raw["initial_joint"]["covariance"] = [[1.0, 2.0], [2.0, 1.0]]  # type: ignore[index]

    with pytest.raises(ValueError, match="positive definite"):
        load_h1_fixture(_write_fixture(tmp_path, raw))


@pytest.mark.parametrize(
    ("field", "value"),
    [("model_variances", [0.42, 0.0]), ("state_variances", [-0.1, 0.48])],
)
def test_loader_rejects_nonpositive_generative_variances(
    tmp_path: Path, field: str, value: list[float]
) -> None:
    raw = _raw_fixture()
    raw[field] = value

    with pytest.raises(ValueError, match="variance"):
        load_h1_fixture(_write_fixture(tmp_path, raw))


def test_loader_rejects_nonpositive_recognition_variance(tmp_path: Path) -> None:
    raw = _raw_fixture()
    raw["recognition"]["model_kernels"][1][0]["variance"] = 0.0  # type: ignore[index]

    with pytest.raises(ValueError, match="variance"):
        load_h1_fixture(_write_fixture(tmp_path, raw))


def test_loader_rejects_recognition_mass_outside_generative_support(
    tmp_path: Path,
) -> None:
    raw = _raw_fixture()
    raw["model_source_priors"] = [[1.0], [0.0, 1.0]]

    with pytest.raises(ValueError, match="support"):
        load_h1_fixture(_write_fixture(tmp_path, raw))
