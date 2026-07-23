from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import verify_vfe4
from verification.run_gates import (
    _h3_projection,
    _legacy_projection,
    _publish_prediction_correctness_artifacts,
)
from vfe4.artifacts.atomic import canonical_json_bytes as producer_json_bytes
from vfe4.config import resolve_config
from vfe4.training.h6_readiness import _load_prediction_correctness_artifact
from vfe4.types import GateStatus
from vfe4.types.h6 import (
    PredictionCorrectnessArtifactRef,
    canonical_json_bytes as h6_owned_json_bytes,
)


_GATES = ("H1", "H2", "H3", "H5")
_HEAD = "1" * 40
_DIRTY = "2" * 64


def _manifest(payloads: dict[str, bytes]) -> bytes:
    return "".join(
        f"{hashlib.sha256(payloads[name]).hexdigest()}  {name}\n"
        for name in sorted(payloads)
    ).encode("ascii")


def test_prediction_reference_accepts_ordinary_producer_json_without_changing_h6_hashes(
) -> None:
    config_bytes = producer_json_bytes({"threshold": 1.25})
    validation_bytes = producer_json_bytes(
        {
            "schema_version": "vfe4-prediction-correctness-v1",
            "gate": "H1",
            "git_head": _HEAD,
            "dirty_digest": _DIRTY,
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "status": "pass",
            "obligations": [],
            "producer_validation": {"ordinary_float": 1.25},
        }
    )
    manifest_bytes = _manifest(
        {
            "config.json": config_bytes,
            "validation/h1.json": validation_bytes,
        }
    )

    reference = PredictionCorrectnessArtifactRef.from_bytes(
        gate="H1",
        artifact_path=Path("validation/h1.json"),
        manifest_bytes=manifest_bytes,
        git_head=_HEAD,
        dirty_digest=_DIRTY,
        config_bytes=config_bytes,
        validation_payload_bytes=validation_bytes,
    )

    assert reference.status is GateStatus.PASS
    assert b'"ordinary_float":1.25' in validation_bytes
    assert h6_owned_json_bytes({"ordinary_float": 1.25}) == (
        b'{"ordinary_float":"0x1.4000000000000p+0"}'
    )


def test_correctness_publisher_writes_four_manifest_linked_roots_without_h4(
    tmp_path: Path,
) -> None:
    raw = copy.deepcopy(
        verify_vfe4.CONFIG["operations"]["h1_h5"]["config"]
    )
    raw["artifacts"]["run_root"] = str(tmp_path)
    canonical = resolve_config(raw, repo_root=verify_vfe4._REPO_ROOT)
    legacy = _legacy_projection(canonical)
    h3 = _h3_projection(canonical)
    results = tuple(
        (
            gate,
            SimpleNamespace(
                gate=gate,
                status=GateStatus.PASS,
                obligations=(),
            ),
        )
        for gate in _GATES
    )
    producer_validations = (
        (
            "H1",
            {
                "gate_result": {
                    "gate": "H1",
                    "status": "pass",
                    "obligations": [],
                },
                "measurement": 0.25,
            },
        ),
        (
            "H2",
            {
                "gate_result": {
                    "gate": "H2",
                    "status": "pass",
                    "obligations": [],
                },
                "measurement": 1.25,
            },
        ),
        (
            "H3",
            {
                "gate": "H3",
                "status": "pass",
                "obligations": [],
                "measurement": 2.25,
            },
        ),
        (
            "H5",
            {
                "result": {
                    "gate": "H5",
                    "status": "pass",
                    "obligations": [],
                },
                "measurement": 3.25,
            },
        ),
    )
    gate_configs = (
        ("H1", legacy),
        ("H2", legacy),
        ("H3", h3),
        ("H5", canonical),
    )
    source_provenance = {
        "git_head": _HEAD,
        "dirty_digest": _DIRTY,
        "h5_config": {"marker": "config"},
        "h5_state_hashes": {"marker": "state"},
        "h5_update_hash_records": {"marker": "updates"},
        "h5_update_binding_preimages": {"marker": "preimages"},
    }
    mismatched = list(producer_validations)
    mismatched[-1] = (
        "H5",
        {
            "result": {
                "gate": "H5",
                "status": "fail",
                "obligations": [],
            }
        },
    )
    with pytest.raises(ValueError, match="differs from its typed gate result"):
        _publish_prediction_correctness_artifacts(
            run_root=tmp_path,
            started_utc="2026-07-23T12:34:56.000000Z",
            source_provenance=source_provenance,
            gate_configs=gate_configs,
            gate_results=results,
            producer_validations=tuple(mismatched),
        )
    assert not tuple(tmp_path.iterdir())

    references = _publish_prediction_correctness_artifacts(
        run_root=tmp_path,
        started_utc="2026-07-23T12:34:56.000000Z",
        source_provenance=source_provenance,
        gate_configs=gate_configs,
        gate_results=results,
        producer_validations=producer_validations,
    )

    assert tuple(gate for gate, _, _ in references) == _GATES
    assert len({root for _, root, _ in references}) == 4
    for gate, root, manifest_sha256 in references:
        paths = tuple(
            path.relative_to(root).as_posix()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        )
        assert paths == (
            "config.json",
            "manifest.sha256",
            "provenance.json",
            f"validation/{gate.lower()}.json",
        )
        assert "validation/h4.json" not in paths
        manifest_bytes = (root / "manifest.sha256").read_bytes()
        assert hashlib.sha256(manifest_bytes).hexdigest() == manifest_sha256
        validation_bytes = (
            root / "validation" / f"{gate.lower()}.json"
        ).read_bytes()
        validation = json.loads(validation_bytes)
        config_bytes = (root / "config.json").read_bytes()
        gate_config = dict(gate_configs)[gate]
        assert config_bytes == gate_config.canonical_json.encode("utf-8")
        assert hashlib.sha256(config_bytes).hexdigest() == gate_config.config_sha256
        assert validation == {
            "schema_version": "vfe4-prediction-correctness-v1",
            "gate": gate,
            "git_head": _HEAD,
            "dirty_digest": _DIRTY,
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "status": "pass",
            "obligations": [],
            "producer_validation": dict(producer_validations)[gate],
        }
        provenance = json.loads((root / "provenance.json").read_bytes())
        assert provenance["gate"] == gate
        assert provenance["git_head"] == _HEAD
        assert provenance["dirty_digest"] == _DIRTY
        assert provenance["config_sha256"] == validation["config_sha256"]
        assert provenance["source_sha256"] == hashlib.sha256(
            b"VFE4-H6-SOURCE-CANDIDATE-V1\x00"
            + bytes.fromhex(_HEAD)
            + bytes.fromhex(_DIRTY)
        ).hexdigest()
        if gate == "H5":
            assert provenance["h5_config"] == {"marker": "config"}
            assert provenance["h5_state_hashes"] == {"marker": "state"}
            assert provenance["h5_update_hash_records"] == {
                "marker": "updates"
            }
            assert provenance["h5_update_binding_preimages"] == {
                "marker": "preimages"
            }
        else:
            assert not any(name.startswith("h5_") for name in provenance)
        loaded = _load_prediction_correctness_artifact(
            gate=gate,
            root=root,
            expected_manifest_sha256=manifest_sha256,
            expected_git_head=_HEAD,
            expected_dirty_digest=_DIRTY,
        )
        assert loaded.status is GateStatus.PASS
        assert loaded.config_sha256 == validation["config_sha256"]


def test_h1_h5_launcher_opt_in_is_boolean_and_authorized_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, bool]] = []
    marker = object()
    monkeypatch.setattr(
        verify_vfe4,
        "_run_h1_h5",
        lambda raw, *, publish_prediction_correctness: (
            calls.append((raw, publish_prediction_correctness)) or marker
        ),
    )
    config = copy.deepcopy(verify_vfe4.CONFIG)
    h1_h5 = config["operations"]["h1_h5"]
    h1_h5["enabled"] = True
    h1_h5["publish_prediction_correctness"] = True

    with pytest.raises(PermissionError, match="explicit phrase"):
        verify_vfe4.main(config)
    assert calls == []

    h1_h5["authorization"] = verify_vfe4._VERIFY_AUTHORIZATIONS["h1_h5"]
    assert verify_vfe4.main(config) is marker
    assert calls == [(h1_h5["config"], True)]

    invalid = copy.deepcopy(verify_vfe4.CONFIG)
    invalid["operations"]["h1_h5"]["publish_prediction_correctness"] = 1
    with pytest.raises(ValueError, match="publish_prediction_correctness.*boolean"):
        verify_vfe4.main(invalid)
