from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

import vfe4.validation.h5_update_spec as h5_spec
from vfe4.types.h5_schema import (
    H5_FACTOR_INPUT_SCHEMA_SHA256,
    H5_FACTOR_UNIVERSE,
    H5_MODEL_BLOCK_UNIVERSE,
    H5_OBJECTIVE_SCHEMA_SHA256,
    H5_RECOGNITION_COORDINATE_UNIVERSE,
)
from vfe4.validation.h5_update_spec import (
    EXPECTED_H1_FIXTURE_RAW_SHA256,
    EXPECTED_H5_UPDATE_SPEC_RAW_SHA256,
    build_h5_reference_state,
    canonical_h5_update_specification_bytes,
    parse_h5_update_spec_bytes,
)


ROOT = Path(__file__).parents[2]
H1_BYTES = (ROOT / "vfe4/validation/fixtures/h1_v1.json").read_bytes()
H5_PATH = ROOT / "vfe4/validation/fixtures/h5_conditional_update_v1.json"
H5_BYTES = H5_PATH.read_bytes()
PINNED_RAW_SHA256 = "9dd42603419952a2ffa4b6602971240ec00572283557d672ae6ee106c31dd91c"


def _semantic_parse(monkeypatch: pytest.MonkeyPatch, data: bytes):
    monkeypatch.setattr(
        h5_spec,
        "EXPECTED_H5_UPDATE_SPEC_RAW_SHA256",
        hashlib.sha256(data).hexdigest(),
    )
    return parse_h5_update_spec_bytes(data)


def test_exact_raw_fixture_bytes_and_hash_are_pinned() -> None:
    assert not H5_BYTES.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in H5_BYTES
    assert H5_BYTES.endswith(b"\n")
    assert not H5_BYTES.endswith(b"\n\n")
    assert hashlib.sha256(H5_BYTES).hexdigest() == PINNED_RAW_SHA256
    assert EXPECTED_H5_UPDATE_SPEC_RAW_SHA256 == PINNED_RAW_SHA256
    assert hashlib.sha256(H1_BYTES).hexdigest() == EXPECTED_H1_FIXTURE_RAW_SHA256


def test_parser_builds_exact_specification_and_reference_hashes() -> None:
    specification = parse_h5_update_spec_bytes(H5_BYTES)
    assert specification.fixture_id == "h5-conditional-update-v1"
    assert specification.fixture_schema_version == 1
    assert specification.recognition_family == (
        "continuous_mean_field_conditional_categorical"
    )
    assert specification.h1_fixture_id == "h1-v1"
    assert specification.h1_fixture_sha256 == EXPECTED_H1_FIXTURE_RAW_SHA256
    assert specification.factor_universe == H5_FACTOR_UNIVERSE
    assert specification.recognition_coordinate_universe == (
        H5_RECOGNITION_COORDINATE_UNIVERSE
    )
    assert specification.model_block_universe == H5_MODEL_BLOCK_UNIVERSE
    assert specification.quadrature_orders == (21, 17)
    assert specification.factor_input_schema_sha256 == H5_FACTOR_INPUT_SCHEMA_SHA256
    assert specification.initial_model.objective_schema_sha256 == (
        H5_OBJECTIVE_SCHEMA_SHA256
    )
    assert specification.canonical_bytes == canonical_h5_update_specification_bytes(
        specification
    )
    assert specification.canonical_sha256 == hashlib.sha256(
        h5_spec.H5_UPDATE_SPEC_DOMAIN + specification.canonical_bytes
    ).hexdigest()
    assert specification.raw_sha256 == PINNED_RAW_SHA256

    reference = build_h5_reference_state(H1_BYTES, H5_BYTES)
    assert reference.h1_fixture_sha256 == EXPECTED_H1_FIXTURE_RAW_SHA256
    assert reference.update_spec_raw_sha256 == PINNED_RAW_SHA256
    assert reference.objective_schema_sha256 == H5_OBJECTIVE_SCHEMA_SHA256
    assert reference.factor_input_schema_sha256 == H5_FACTOR_INPUT_SCHEMA_SHA256


def test_conditional_family_reconstructs_source_independent_h1_record() -> None:
    state = build_h5_reference_state(H1_BYTES, H5_BYTES)
    equivalent = state.specification.as_h1_recognition_record()
    assert equivalent.initial_joint.mean.tolist() == [-0.10, 0.25]
    assert torch.equal(
        equivalent.initial_joint.covariance,
        torch.diag(torch.tensor([0.65, 0.78], dtype=torch.float64)),
    )
    for record in equivalent.model_kernels:
        assert torch.equal(record.slopes, torch.zeros_like(record.slopes))
        assert len(
            set(zip(record.offsets.tolist(), record.variances.tolist(), strict=True))
        ) == 1
    assert equivalent.model_kernels[0].offsets.tolist() == [0.175]
    assert equivalent.model_kernels[0].variances.tolist() == [1.21]
    assert equivalent.model_kernels[1].offsets.tolist() == [0.14, 0.14]
    assert equivalent.model_kernels[1].variances.tolist() == [1.40, 1.40]
    for record in equivalent.state_kernels:
        assert torch.equal(record.z_slopes, torch.zeros_like(record.z_slopes))
        assert torch.equal(record.m_slopes, torch.zeros_like(record.m_slopes))
        assert len(
            set(zip(record.offsets.tolist(), record.variances.tolist(), strict=True))
        ) == 1
    assert equivalent.state_kernels[0].offsets.tolist() == [0.05]
    assert equivalent.state_kernels[0].variances.tolist() == [0.96]
    assert equivalent.state_kernels[1].offsets.tolist() == [-0.04] * 4
    assert equivalent.state_kernels[1].variances.tolist() == [0.90] * 4
    assert tuple(equivalent.model_source_probabilities[0].tolist()) == (1.0,)
    assert tuple(equivalent.model_source_probabilities[1].tolist()) == (0.4, 0.6)
    assert torch.equal(
        equivalent.state_source_probabilities_given_model_source[0],
        torch.tensor([[1.0]], dtype=torch.float64),
    )
    assert tuple(
        equivalent.state_source_probabilities_given_model_source[1][0].tolist()
    ) == (0.75, 0.25)
    assert tuple(
        equivalent.state_source_probabilities_given_model_source[1][1].tolist()
    ) == (0.2, 0.8)


def test_raw_hash_is_checked_before_utf8_or_json_decode() -> None:
    with pytest.raises(ValueError, match="raw SHA-256"):
        parse_h5_update_spec_bytes(b"\xffnot-json")
    with pytest.raises(ValueError, match="raw SHA-256"):
        parse_h5_update_spec_bytes(H5_BYTES[:-1])


def test_parser_rejects_duplicate_keys_and_object_field_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = H5_BYTES.decode("utf-8")
    duplicate = text.replace(
        '  "fixture_id": "h5-conditional-update-v1",',
        '  "fixture_id": "h5-conditional-update-v1",\n  "fixture_id": "h5-conditional-update-v1",',
        1,
    ).encode()
    with pytest.raises(ValueError, match="duplicate"):
        _semantic_parse(monkeypatch, duplicate)

    decoded = json.loads(text)
    reordered = {"fixture_schema_version": decoded["fixture_schema_version"]}
    reordered.update({key: value for key, value in decoded.items() if key != "fixture_schema_version"})
    data = (json.dumps(reordered, separators=(",", ":")) + "\n").encode()
    with pytest.raises(ValueError, match="field order"):
        _semantic_parse(monkeypatch, data)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda root: root.pop("source_row_a2"), "fields"),
        (lambda root: root.__setitem__("extra", 1), "fields"),
        (
            lambda root: root["continuous_recognition"][0].__setitem__(
                0, "q[z_zero]"
            ),
            "continuous_recognition",
        ),
        (
            lambda root: root["categorical_recognition"][3][3].__setitem__(
                slice(None), [0.5, 0.6]
            ),
            "probabilities",
        ),
        (
            lambda root: root["categorical_recognition"][3][3].__setitem__(
                slice(None), [1.0, 0.0]
            ),
            "positive",
        ),
        (
            lambda root: root["shared_parameter_groups"][0][2].__setitem__(
                0, "wrong-consumer"
            ),
            "shared",
        ),
        (
            lambda root: root["factor_reconstruction"][0][1].__setitem__(
                1, "q[z1]"
            ),
            "reconstruction",
        ),
    ),
)
def test_parser_rejects_missing_extra_alias_probability_and_schema_drift(
    monkeypatch: pytest.MonkeyPatch,
    mutation,
    message: str,
) -> None:
    root = json.loads(H5_BYTES)
    mutation(root)
    data = (json.dumps(root, separators=(",", ":"), ensure_ascii=True) + "\n").encode()
    with pytest.raises(ValueError, match=message):
        _semantic_parse(monkeypatch, data)


def test_reference_builder_rejects_wrong_h1_hash_before_state_construction() -> None:
    with pytest.raises(ValueError, match="H1 raw SHA-256"):
        build_h5_reference_state(H1_BYTES + b"\n", H5_BYTES)
