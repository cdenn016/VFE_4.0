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
    adjudicate_h6_prediction_opening,
)
from vfe4.evaluation.smc_uncertainty import (
    SMC_BIAS_SEMANTICS,
    inflate_paired_interval,
)
from vfe4.types.h6 import EvidenceStatus


def test_prediction_readiness_roots_exclude_h4_and_current_gate_payload_blocks(
    tmp_path: Path,
) -> None:
    assert PREDICTION_READINESS_SOURCE_BLOCKERS == ()

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


def test_objective_gate_blocks_primary_without_a_second_opening() -> None:
    def interval(value: float, *, error_radius: float = 0.0):
        return inflate_paired_interval(
            (value,) * 8,
            (error_radius,) * 8,
            (0.0,) * 8,
            (0.0,) * 8,
        )

    objective = interval(0.02)
    raw_primary = interval(0.03)
    decision, metrics_bytes = adjudicate_h6_prediction_opening(
        objective_interval=objective,
        primary_interval=raw_primary,
        objective_estimator_complete=True,
        primary_estimator_complete=True,
        test_opening_sha256="d" * 64,
        raw_endpoint_inventory_sha256="e" * 64,
        opening_count=1,
    )
    assert decision.status is EvidenceStatus.FAIL
    assert (
        decision.primary_disposition
        == "NOT_EVALUATED_AFTER_OBJECTIVE_GATE"
    )
    assert decision.primary_interval == (raw_primary.lower, raw_primary.upper)
    assert decision.opening_count == 1

    payload = json.loads(metrics_bytes)
    assert payload["schema"] == "h6-prediction-metrics-v2"
    assert (
        payload["smc_bias_semantics_sha256"]
        == SMC_BIAS_SEMANTICS.semantics_sha256
    )
    assert payload["opening_count"] == 1
    assert payload["test_opening_sha256"] == "d" * 64
    assert payload["raw_endpoint_inventory_sha256"] == "e" * 64
    assert payload["primary_interval"] == {
        "lower": raw_primary.lower,
        "upper": raw_primary.upper,
    }
    assert (
        payload["primary_disposition"]
        == "NOT_EVALUATED_AFTER_OBJECTIVE_GATE"
    )

    ineligible_primary, _ = adjudicate_h6_prediction_opening(
        objective_interval=interval(0.0),
        primary_interval=interval(0.03, error_radius=0.02),
        objective_estimator_complete=True,
        primary_estimator_complete=True,
        test_opening_sha256="d" * 64,
        raw_endpoint_inventory_sha256="e" * 64,
        opening_count=1,
    )
    assert ineligible_primary.status is EvidenceStatus.INCONCLUSIVE
    assert ineligible_primary.primary_disposition == "INCONCLUSIVE"
    assert ineligible_primary.obligations == (
        "primary estimator interval is ineligible",
    )

    with pytest.raises(ValueError, match="exactly one test opening"):
        adjudicate_h6_prediction_opening(
            objective_interval=objective,
            primary_interval=raw_primary,
            objective_estimator_complete=True,
            primary_estimator_complete=True,
            test_opening_sha256="d" * 64,
            raw_endpoint_inventory_sha256="e" * 64,
            opening_count=2,
        )
