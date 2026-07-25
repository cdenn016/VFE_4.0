"""Compact source contract for the fail-closed H7 gate boundary."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import vfe4.types.h7 as h7_types
import verification.h7_gate as h7_gate_module
from verification.h7_gate import (
    H7_ACTIVE_SCORER_PROFILE,
    H7_CAPTURED_FIXTURE_PATHS,
    H7_FROZEN_SOURCE_FIXTURE_HASHES,
    H7_PREDECESSOR_KEYS,
    H7_SOURCE_ONLY_OBLIGATIONS,
    H7DependencyClosure,
    _classify_h7_status_from_state,
    _inventory_obligations,
    _validate_predecessor_files,
    assemble_h7_gate_evaluation,
    validate_h7_predecessor_registry,
)
from verification.run_gates import _combined_provenance, run_verification
from vfe4.artifacts import publish_run_directory
from vfe4.types import H7GateResult as PublicH7GateResult
from vfe4.types.h7 import (
    H7_CONTROL_IDS,
    H7_REQUIRED_TRIAL_IDS,
    H7ControlResult,
    H7InconclusiveOutcome,
    H7PredecessorReference,
)
from vfe4.types.results import GateStatus, H7GateResult


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _controlled_predecessor(
    repo_root: Path,
    *,
    name: str,
    git_head: str,
    declared_dirty_digest: str,
    artifact_dirty_digest: str,
    junit_sha256: str,
    live_artifact_revision: str,
    ledger_artifact_revision: str,
    ledger_errors: tuple[str, ...] = (),
) -> tuple[H7PredecessorReference, SimpleNamespace, str]:
    artifact = repo_root / name / "artifact"
    validation = artifact / "validation"
    validation.mkdir(parents=True)
    payload_name = "validation/h5.json"
    payload = _json_bytes(
        {
            "gate": "H5",
            "status": "pass",
            "schema_version": 1,
            "git_head": git_head,
            "dirty_digest": artifact_dirty_digest,
            "junit_sha256": junit_sha256,
        }
    )
    payload_path = validation / "h5.json"
    payload_path.write_bytes(payload)
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    manifest = f"{payload_sha256}  {payload_name}\n".encode("utf-8")
    (artifact / "manifest.sha256").write_bytes(manifest)

    ledger = {
        "schema_version": "1.0",
        "mode": "closure",
        "artifact_revision": ledger_artifact_revision,
        "claims": [
            {
                "artifact_revision": ledger_artifact_revision,
                "state": "EVIDENCE_VERIFIED",
                "open_obligations": [],
                "evidence_invalidated": False,
            }
        ],
    }
    ledger_path = repo_root / ".verification" / f"{name}.json"
    ledger_path.parent.mkdir(exist_ok=True)
    ledger_bytes = _json_bytes(ledger)
    ledger_path.write_bytes(ledger_bytes)
    reference = H7PredecessorReference.create(
        artifact_path=str(artifact),
        git_head=git_head,
        dirty_digest=declared_dirty_digest,
        junit_sha256=junit_sha256,
        manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        payload_hashes={payload_name: payload_sha256},
        ledger_path=str(ledger_path),
        ledger_sha256=hashlib.sha256(ledger_bytes).hexdigest(),
    )
    validated_ledgers: list[dict[str, object]] = []

    def validate_ledger(ledger_value: dict[str, object]) -> list[str]:
        validated_ledgers.append(ledger_value)
        return list(ledger_errors)

    validator_api = SimpleNamespace(
        validate_ledger=validate_ledger,
        validated_ledgers=validated_ledgers,
    )
    return reference, validator_api, live_artifact_revision


def test_h7_result_has_one_owner_and_closed_inventories() -> None:
    assert PublicH7GateResult is H7GateResult
    assert "H7GateResult" not in vars(h7_types)
    assert H7_REQUIRED_TRIAL_IDS == (
        "scalar-base-transformed",
        "scalar-internal-transformed",
        "matrix-identity-base-transformed",
        "matrix-identity-internal-transformed",
        "matrix-nonidentity-base-transformed",
        "matrix-nonidentity-internal-transformed",
        "matrix-fixed-decoder-centered-stabilizer",
        "matrix-fixed-decoder-outside-stabilizer",
    )
    assert H7_CONTROL_IDS == (
        "wrong_covariance_congruence",
        "wrong_precision_congruence",
        "history_scorer_wrong_source_inverse",
        "reversed_link_order",
        "reverse_arrow_B",
        "wrong_decoder_dual_action",
        "fixed_decoder_outside_stabilizer",
        "omitted_density_jacobian",
        "reversed_logdet_sign",
        "entropy_false_invariance",
        "changed_h1_source_probability",
        "diagonal_for_internal_action",
    )
    assert H7_PREDECESSOR_KEYS == ("h1_h5", "h1_prefix_prior", "h6_prefix")


def test_gate_public_boundary_does_not_accept_prevalidated_provenance() -> None:
    parameters = inspect.signature(assemble_h7_gate_evaluation).parameters
    assert {
        "repo_root",
        "predecessor_entries",
        "git_head",
        "dirty_digest",
        "junit_sha256",
        "scorer_profile",
        "captured_fixture_bytes",
    }.issubset(parameters)
    assert {
        "dependency_closure",
        "predecessor_validation",
        "expected_negative_false_acceptance",
    }.isdisjoint(parameters)
    assert "ledger_validator_sha256" in {
        item.name for item in fields(H7DependencyClosure)
    }


def test_candidate_junit_flows_through_real_minimal_h1_h5_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vfe4.artifacts.provenance as provenance_module

    git_head = "a" * 40
    dirty_digest = "b" * 64
    junit_sha256 = "c" * 64
    revision = f"git:{git_head}:sha256:{dirty_digest}"
    monkeypatch.setattr(provenance_module, "git_head", lambda _root: git_head)
    monkeypatch.setattr(
        provenance_module,
        "dirty_content_digest",
        lambda _root, _runs: dirty_digest,
    )
    config = SimpleNamespace(
        objective_schema_version="vfe4-state-elbo-v1",
        config_sha256="d" * 64,
        artifacts=SimpleNamespace(run_root=tmp_path / "runs"),
        run=SimpleNamespace(
            device="cpu",
            dtype="float64",
            seed=20260721,
            deterministic=True,
        ),
    )
    import verification.run_gates as gates_module

    h1 = SimpleNamespace(
        fixture_observed_sha256=gates_module.EXPECTED_H1_FIXTURE_SHA256,
        result=SimpleNamespace(gate="H1", status=GateStatus.PASS),
    )
    legacy = _combined_provenance(
        config,
        h1,
        None,
        None,
        None,
        None,
        "2026-07-23T00:00:00Z",
        "2026-07-23T00:00:01Z",
        None,
    )
    bound = _combined_provenance(
        config,
        h1,
        None,
        None,
        None,
        None,
        "2026-07-23T00:00:00Z",
        "2026-07-23T00:00:01Z",
        junit_sha256,
    )
    assert "junit_sha256" not in legacy
    assert bound["junit_sha256"] == junit_sha256
    assert set(bound) == {*legacy, "junit_sha256"}
    assert all(bound[key] == value for key, value in legacy.items())
    assert (
        inspect.signature(run_verification)
        .parameters["candidate_junit_sha256"]
        .default
        is None
    )

    artifact = publish_run_directory(
        tmp_path / "published",
        "h1-h5-current-candidate",
        {
            "config.json": {"schema_version": 1},
            "provenance.json": bound,
            "validation/h5.json": {
                "gate": "H5",
                "status": "pass",
                "schema_version": 1,
                "git_head": git_head,
                "dirty_digest": dirty_digest,
                "junit_sha256": junit_sha256,
            },
        },
    )
    manifest_bytes = (artifact / "manifest.sha256").read_bytes()
    payload_hashes = {
        line.split("  ", 1)[1]: line.split("  ", 1)[0]
        for line in manifest_bytes.decode("ascii").splitlines()
    }
    ledger_path = tmp_path / ".verification" / "h1-h5-ledger.json"
    ledger_path.parent.mkdir()
    ledger_bytes = _json_bytes(
        {
            "schema_version": "1.0",
            "mode": "closure",
            "artifact_revision": revision,
            "claims": [
                {
                    "artifact_revision": revision,
                    "state": "EVIDENCE_VERIFIED",
                    "open_obligations": [],
                    "evidence_invalidated": False,
                }
            ],
        }
    )
    ledger_path.write_bytes(ledger_bytes)
    reference = H7PredecessorReference.create(
        artifact_path=str(artifact),
        git_head=git_head,
        dirty_digest=dirty_digest,
        junit_sha256=junit_sha256,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        payload_hashes=payload_hashes,
        ledger_path=str(ledger_path),
        ledger_sha256=hashlib.sha256(ledger_bytes).hexdigest(),
    )
    validator_api = SimpleNamespace(validate_ledger=lambda _ledger: [])
    _validate_predecessor_files(
        "h1_h5",
        reference,
        repo_root=tmp_path,
        validator_api=validator_api,
        live_artifact_revision=revision,
    )


def test_h7_assembly_reuses_each_captured_fixture_without_reopening(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    git_head = "a" * 40
    dirty_digest = "b" * 64
    junit_sha256 = "c" * 64
    revision = f"git:{git_head}:sha256:{dirty_digest}"
    fixture_paths = {}
    for relative, content in zip(
        H7_CAPTURED_FIXTURE_PATHS,
        (b"h1 captured bytes", b"h7 captured bytes"),
        strict=True,
    ):
        path = tmp_path / Path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        fixture_paths[relative] = path
    validator_path = tmp_path / "verification_gate.py"
    validator_path.write_bytes(b"validator source")
    monkeypatch.setattr(
        h7_gate_module,
        "H7_REQUIRED_DEPENDENCY_PATHS",
        H7_CAPTURED_FIXTURE_PATHS,
    )
    monkeypatch.setattr(
        h7_gate_module,
        "_verification_ledger_validator_path",
        lambda: validator_path,
    )
    monkeypatch.setattr(
        h7_gate_module,
        "_load_verification_gate_api",
        lambda **_kwargs: SimpleNamespace(
            capture_artifact_revision=lambda _root: revision,
            validate_ledger=lambda _ledger: [],
        ),
    )
    real_read_bytes = Path.read_bytes
    reads = {relative: 0 for relative in H7_CAPTURED_FIXTURE_PATHS}

    def counted_read_bytes(path: Path) -> bytes:
        for relative, fixture_path in fixture_paths.items():
            if path.resolve(strict=False) == fixture_path.resolve(strict=False):
                reads[relative] += 1
        return real_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", counted_read_bytes)
    captured = {
        relative: fixture_paths[relative].read_bytes()
        for relative in H7_CAPTURED_FIXTURE_PATHS
    }
    evaluation = assemble_h7_gate_evaluation(
        repo_root=tmp_path,
        captured_fixture_bytes=captured,
        predecessor_entries=(),
        git_head=git_head,
        dirty_digest=dirty_digest,
        junit_sha256=junit_sha256,
        scorer_profile=H7_ACTIVE_SCORER_PROFILE,
        fixture_hashes={
            "density_probe_table_raw_sha256": (
                H7_FROZEN_SOURCE_FIXTURE_HASHES[
                    "density_probe_table_raw_sha256"
                ]
            ),
            "density_probe_set_sha256": H7_FROZEN_SOURCE_FIXTURE_HASHES[
                "density_probe_set_sha256"
            ],
        },
        trials=(),
        controls=(),
        oracle_obligations=H7_SOURCE_ONLY_OBLIGATIONS,
    )

    assert reads == {relative: 1 for relative in H7_CAPTURED_FIXTURE_PATHS}
    assert evaluation.result.fixture_hashes["h1_fixture_raw_sha256"] == (
        hashlib.sha256(captured[H7_CAPTURED_FIXTURE_PATHS[0]]).hexdigest()
    )
    assert evaluation.result.fixture_hashes["h7_fixture_raw_sha256"] == (
        hashlib.sha256(captured[H7_CAPTURED_FIXTURE_PATHS[1]]).hexdigest()
    )


def test_live_revision_digest_must_equal_declared_dirty_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    git_head = "a" * 40
    declared_dirty_digest = "b" * 64
    fake_api = SimpleNamespace(
        capture_artifact_revision=lambda _root: (
            f"git:{git_head}:sha256:{'c' * 64}"
        ),
        validate_ledger=lambda _ledger: [],
    )
    monkeypatch.setattr(
        h7_gate_module,
        "_load_verification_gate_api",
        lambda **_kwargs: fake_api,
    )
    validation = validate_h7_predecessor_registry(
        (),
        repo_root=tmp_path,
        git_head=git_head,
        dirty_digest=declared_dirty_digest,
        junit_sha256="d" * 64,
        scorer_profile=H7_ACTIVE_SCORER_PROFILE,
        ledger_validator_sha256="e" * 64,
    )
    assert (
        "live verification artifact revision differs from the H7 candidate"
        in validation.obligations
    )


def test_malformed_ledger_is_rejected_by_deterministic_validator(
    tmp_path: Path,
) -> None:
    git_head = "a" * 40
    dirty_digest = "b" * 64
    junit_sha256 = "c" * 64
    revision = f"git:{git_head}:sha256:{dirty_digest}"
    reference, validator_api, live_revision = _controlled_predecessor(
        tmp_path,
        name="malformed-ledger",
        git_head=git_head,
        declared_dirty_digest=dirty_digest,
        artifact_dirty_digest=dirty_digest,
        junit_sha256=junit_sha256,
        live_artifact_revision=revision,
        ledger_artifact_revision=revision,
        ledger_errors=("ledger: malformed closure record",),
    )
    with pytest.raises(
        ValueError,
        match="deterministic ledger validation failed",
    ):
        _validate_predecessor_files(
            "h1_h5",
            reference,
            repo_root=tmp_path,
            validator_api=validator_api,
            live_artifact_revision=live_revision,
        )
    assert len(validator_api.validated_ledgers) == 1


def test_artifact_and_ledger_are_cross_bound_to_one_candidate(
    tmp_path: Path,
) -> None:
    git_head = "a" * 40
    dirty_digest = "b" * 64
    junit_sha256 = "c" * 64
    revision = f"git:{git_head}:sha256:{dirty_digest}"
    wrong_revision = f"git:{git_head}:sha256:{'d' * 64}"

    stale_artifact, artifact_api, artifact_live = _controlled_predecessor(
        tmp_path,
        name="stale-artifact",
        git_head=git_head,
        declared_dirty_digest=dirty_digest,
        artifact_dirty_digest="d" * 64,
        junit_sha256=junit_sha256,
        live_artifact_revision=revision,
        ledger_artifact_revision=revision,
    )
    with pytest.raises(ValueError, match="artifact dirty_digest differs"):
        _validate_predecessor_files(
            "h1_h5",
            stale_artifact,
            repo_root=tmp_path,
            validator_api=artifact_api,
            live_artifact_revision=artifact_live,
        )

    stale_ledger, ledger_api, ledger_live = _controlled_predecessor(
        tmp_path,
        name="stale-ledger",
        git_head=git_head,
        declared_dirty_digest=dirty_digest,
        artifact_dirty_digest=dirty_digest,
        junit_sha256=junit_sha256,
        live_artifact_revision=revision,
        ledger_artifact_revision=wrong_revision,
    )
    with pytest.raises(ValueError, match="not validated at this candidate"):
        _validate_predecessor_files(
            "h1_h5",
            stale_ledger,
            repo_root=tmp_path,
            validator_api=ledger_api,
            live_artifact_revision=ledger_live,
        )
    assert len(ledger_api.validated_ledgers) == 1


def test_expected_negative_paths_and_complete_pass_fail_precedence() -> None:
    assert (
        _classify_h7_status_from_state(
            obligations=(),
            failed_invariant_ids=(),
            expected_negative_state="success",
        )
        is GateStatus.PASS
    )
    assert (
        _classify_h7_status_from_state(
            obligations=(),
            failed_invariant_ids=(),
            expected_negative_state="false_acceptance",
        )
        is GateStatus.FAIL
    )
    assert (
        _classify_h7_status_from_state(
            obligations=(),
            failed_invariant_ids=(),
            expected_negative_state="inconclusive",
        )
        is GateStatus.INCONCLUSIVE
    )
    assert (
        _classify_h7_status_from_state(
            obligations=(),
            failed_invariant_ids=("finite covariance violation",),
            expected_negative_state="success",
        )
        is GateStatus.FAIL
    )
    assert (
        _classify_h7_status_from_state(
            obligations=("missing current predecessor",),
            failed_invariant_ids=("finite covariance violation",),
            expected_negative_state="false_acceptance",
        )
        is GateStatus.INCONCLUSIVE
    )


def test_exact_trial_and_control_order_and_decisiveness_boundary() -> None:
    assert (
        _inventory_obligations(
            trial_ids=H7_REQUIRED_TRIAL_IDS,
            control_ids=H7_CONTROL_IDS,
        )
        == ()
    )
    assert _inventory_obligations(
        trial_ids=H7_REQUIRED_TRIAL_IDS,
        control_ids=tuple(reversed(H7_CONTROL_IDS)),
    ) == ("required H7 control inventory is missing, duplicated, or reordered",)
    observed_ids: list[str] = []
    matching_allowance = 1e-12
    invariant_scale = 1.0
    boundary = max(100.0 * matching_allowance, 1e-8 * invariant_scale)
    for control_id in H7_CONTROL_IDS:
        at_boundary = H7ControlResult.create(
            control_id=control_id,
            target_invariant_id=f"{control_id}.target",
            wrong_residual=boundary,
            invariant_scale=invariant_scale,
            matching_correct_allowance=matching_allowance,
            decisiveness_limit=boundary,
            detected=False,
        )
        immediately_above = H7ControlResult.create(
            control_id=control_id,
            target_invariant_id=f"{control_id}.target",
            wrong_residual=math.nextafter(boundary, math.inf),
            invariant_scale=invariant_scale,
            matching_correct_allowance=matching_allowance,
            decisiveness_limit=boundary,
            detected=True,
        )
        assert not at_boundary.detected
        assert immediately_above.detected
        observed_ids.append(control_id)
    assert tuple(observed_ids) == H7_CONTROL_IDS


def test_source_only_result_owns_mappings_and_rejects_hash_drift() -> None:
    mutable_hashes = dict(H7_FROZEN_SOURCE_FIXTURE_HASHES)
    outcome = H7InconclusiveOutcome.create(
        kind="INCONCLUSIVE",
        obligations=H7_SOURCE_ONLY_OBLIGATIONS,
    )
    result = H7GateResult.create(
        gate="H7",
        status=GateStatus.INCONCLUSIVE,
        fixture_hashes=mutable_hashes,
        predecessor_references={},
        trials=(),
        controls=(),
        outcome=outcome,
        obligations=H7_SOURCE_ONLY_OBLIGATIONS,
    )
    mutable_hashes["h1_fixture_raw_sha256"] = "0" * 64
    assert (
        result.fixture_hashes["h1_fixture_raw_sha256"]
        == H7_FROZEN_SOURCE_FIXTURE_HASHES["h1_fixture_raw_sha256"]
    )
    with pytest.raises(TypeError):
        result.fixture_hashes["h1_fixture_raw_sha256"] = "0" * 64  # type: ignore[index]
    with pytest.raises(ValueError, match="result_sha256"):
        replace(result, result_sha256="0" * 64)
