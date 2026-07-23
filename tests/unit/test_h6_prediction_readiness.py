from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vfe4.training.h6_readiness import (
    CurrentPredictionPrerequisiteRefs,
    PREDICTION_READINESS_SOURCE_BLOCKERS,
    ProducerCompatibilityError,
    _load_prediction_correctness_artifact,
    _revalidate_h6_prediction_readiness_inputs,
)


def test_prediction_readiness_roots_exclude_h4_and_current_gate_payload_blocks(
    tmp_path: Path,
) -> None:
    assert PREDICTION_READINESS_SOURCE_BLOCKERS == (
        "separate manifest-linked H1/H2/H3/H5 correctness producers are absent",
        "finite-SMC lacks a manifest-linked config/estimator/fixture publisher",
        "H5 does not publish the ten exact update-binding preimages",
        "blinded data does not publish retained typed DataIdentity preimages",
        "arm matching lacks an immutable manifest-linked matching-set publisher",
    )
    with pytest.raises(ProducerCompatibilityError) as unavailable:
        _revalidate_h6_prediction_readiness_inputs(
            config=object(),  # type: ignore[arg-type]
            prerequisite_refs=object(),  # type: ignore[arg-type]
        )
    assert all(
        blocker in str(unavailable.value)
        for blocker in PREDICTION_READINESS_SOURCE_BLOCKERS
    )

    roots = {gate: tmp_path / gate.lower() for gate in ("H1", "H2", "H3", "H5")}
    refs = CurrentPredictionPrerequisiteRefs.from_mapping(
        roots,
        h1_prefix_prior_artifact_root=tmp_path / "h1-prefix-prior",
        smc_accuracy_artifact_root=tmp_path / "smc",
        h6_prefix_artifact_root=tmp_path / "prefix",
        blinded_data_artifact_root=tmp_path / "blinded",
        matching_artifact_root=tmp_path / "matching",
    )
    assert tuple(gate for gate, _ in refs.correctness_artifact_roots) == (
        "H1",
        "H2",
        "H3",
        "H5",
    )
    launcher_refs = CurrentPredictionPrerequisiteRefs.from_mapping(
        {
            "correctness_artifact_roots": {
                gate: root.relative_to(tmp_path).as_posix()
                for gate, root in roots.items()
            },
            "h1_prefix_prior_artifact_root": "h1-prefix-prior",
            "smc_accuracy_artifact_root": "smc",
            "h6_prefix_artifact_root": "prefix",
            "blinded_data_artifact_root": "blinded",
            "matching_artifact_root": "matching",
        },
        repo_root=tmp_path,
    )
    assert launcher_refs == refs
    with pytest.raises(ValueError, match="exactly H1, H2, H3, H5"):
        CurrentPredictionPrerequisiteRefs.from_mapping(
            {**roots, "H4": tmp_path / "h4"},
            h1_prefix_prior_artifact_root=tmp_path / "h1-prefix-prior",
            smc_accuracy_artifact_root=tmp_path / "smc",
            h6_prefix_artifact_root=tmp_path / "prefix",
            blinded_data_artifact_root=tmp_path / "blinded",
            matching_artifact_root=tmp_path / "matching",
        )

    root = roots["H1"]
    (root / "validation").mkdir(parents=True)
    config = b'{"config_sha256":"' + b"a" * 64 + b'"}'
    provenance = json.dumps(
        {"git_head": "b" * 40, "dirty_digest": "c" * 64},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    current_payload = b'{"gate_result":{"gate":"H1","status":"pass"}}'
    payloads = {
        "config.json": config,
        "provenance.json": provenance,
        "validation/h1.json": current_payload,
    }
    manifest = "".join(
        f"{hashlib.sha256(payload).hexdigest()}  {name}\n"
        for name, payload in sorted(payloads.items())
    ).encode("ascii")
    for name, payload in payloads.items():
        path = root / Path(*name.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    (root / "manifest.sha256").write_bytes(manifest)

    with pytest.raises(
        ProducerCompatibilityError,
        match=r"H1.*direct gate/git_head/dirty_digest/config_sha256/status/obligations",
    ):
        _load_prediction_correctness_artifact(
            gate="H1",
            root=root,
            expected_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
            expected_git_head="b" * 40,
            expected_dirty_digest="c" * 64,
        )
