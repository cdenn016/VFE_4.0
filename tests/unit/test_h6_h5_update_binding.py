from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from vfe4.artifacts.atomic import canonical_json_bytes
from vfe4.training.h6_readiness import _load_h5_update_binding
from vfe4.types.h6 import H5UpdateBinding


_LABELS = (
    "exact_coordinate",
    "generalized_em",
    "natural_gradient_proposal",
)
_INTRINSIC_PREIMAGES = {
    "update_spec_raw_sha256": b"raw H5 update specification",
    "update_spec_canonical_sha256": b"domain\x00canonical H5 update specification",
    "objective_schema_sha256": b"domain\x00H5 objective schema",
    "factor_input_schema_sha256": b"domain\x00H5 factor-input schema",
    "reference_sha256": b"domain\x00H5 reference state",
    "recognition_state_sha256": b"domain\x00H5 recognition state",
    "model_state_sha256": b"domain\x00H5 model state",
    "validation_payload_sha256": b"domain\x00H5 validation payload",
}


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_h5_update_binding_loader_binds_the_exact_ten_producer_preimages(
    tmp_path: Path,
) -> None:
    intrinsic_digests = {
        name: _sha256(preimage)
        for name, preimage in _INTRINSIC_PREIMAGES.items()
    }
    validation_bytes = canonical_json_bytes(
        {
            "schema_version": "vfe4-prediction-correctness-v1",
            "gate": "H5",
            "git_head": "1" * 40,
            "dirty_digest": "2" * 64,
            "config_sha256": "3" * 64,
            "status": "pass",
            "obligations": [],
            "producer_validation": {
                "payload_sha256": intrinsic_digests[
                    "validation_payload_sha256"
                ],
            },
        }
    )
    provenance_bytes = canonical_json_bytes(
        {
            "h5_config": {
                "enabled_update_labels": list(_LABELS),
                **{
                    name: intrinsic_digests[name]
                    for name in (
                        "update_spec_raw_sha256",
                        "update_spec_canonical_sha256",
                        "objective_schema_sha256",
                        "factor_input_schema_sha256",
                    )
                },
            },
            "h5_state_hashes": {
                "reference_sha256": intrinsic_digests["reference_sha256"],
                "recognition_sha256": intrinsic_digests[
                    "recognition_state_sha256"
                ],
                "model_sha256": intrinsic_digests["model_state_sha256"],
                "validation_payload_sha256": intrinsic_digests[
                    "validation_payload_sha256"
                ],
            },
            "h5_update_binding_preimages": {
                "schema_version": "h5-update-binding-preimages-v1",
                "encoding": "hex",
                "preimages": {
                    name: preimage.hex()
                    for name, preimage in _INTRINSIC_PREIMAGES.items()
                },
            },
        }
    )
    payloads = {
        "provenance.json": provenance_bytes,
        "validation/h5.json": validation_bytes,
    }
    manifest_bytes = "".join(
        f"{_sha256(payload)}  {name}\n"
        for name, payload in sorted(payloads.items())
    ).encode("ascii")
    for name, payload in payloads.items():
        path = tmp_path / Path(*name.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    (tmp_path / "manifest.sha256").write_bytes(manifest_bytes)

    all_preimages = {
        "h5_manifest_sha256": manifest_bytes,
        "h5_payload_sha256": validation_bytes,
        **_INTRINSIC_PREIMAGES,
    }
    expected = H5UpdateBinding.from_producer_preimages(
        producer_preimages=all_preimages,
        enabled_update_labels=_LABELS,
    )

    loaded = _load_h5_update_binding(
        tmp_path,
        expected_binding_sha256=expected.binding_sha256,
    )

    assert loaded == expected
    loaded.verify_producer_preimages(all_preimages)
    for name, preimage in all_preimages.items():
        mutated = dict(all_preimages)
        mutated[name] = preimage[:-1] + bytes((preimage[-1] ^ 1,))
        with pytest.raises(ValueError, match=name):
            loaded.verify_producer_preimages(mutated)
