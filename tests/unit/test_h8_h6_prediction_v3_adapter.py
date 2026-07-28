from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from verification import h8_h6_prediction_v3 as adapter
from vfe4.training.h6_test_transaction_v3 import H6TestReservationV3
from vfe4.types.h8 import (
    H8H6PredictionReference,
    H8H6PredictionV3Reference,
)
from vfe4.types.h6_prediction_v3 import H6_SCORING_INVENTORY_SHA256


def _sha(character: str = "a") -> str:
    return character * 64


_CLOSURE_CLAIM_ID = "h6-prediction-v3-exact-artifact-closure"
_CLOSURE_CLAIM_STATEMENT = (
    "The exact H6-Prediction v3 candidate JUnit, authorities, validation, "
    "one-shot transaction, and result artifacts are evidence-verified at "
    "the live producer revision."
)
_A0_DIRECT_EXACT_PREFIX_CERTIFICATE_SHA256 = _sha("9")


def _hash_evidence_location(path: Path, digest: str) -> str:
    return json.dumps(
        {
            "path": path.resolve(strict=True).as_posix(),
            "sha256": digest,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _write_test_manifest(directory: Path, payload_name: str) -> tuple[Path, str]:
    directory.mkdir(parents=True, exist_ok=True)
    payload = b"{}\n"
    (directory / payload_name).write_bytes(payload)
    manifest_path = directory / "manifest.sha256"
    manifest_path.write_bytes(
        f"{hashlib.sha256(payload).hexdigest()}  {payload_name}\n".encode(
            "ascii"
        )
    )
    return manifest_path, hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def _reference(tmp_path: Path) -> H8H6PredictionV3Reference:
    evidence = tmp_path / "evidence"
    reservation = H6TestReservationV3.create(
        experiment_config_sha256=_sha("e"),
        readiness_sha256=_sha("f"),
        plan_sha256=_sha("1"),
        experiment_identity_sha256=_sha("8"),
        data_identity_sha256=_sha("a"),
        sealed_test_sha256=_sha("b"),
        test_inventory_sha256=_sha("c"),
        access_policy_sha256=_sha("d"),
        tuning_selection_sha256=_sha("e"),
        checkpoint_selection_sha256=_sha("4"),
        validation_bundle_sha256=_sha("3"),
        scoring_inventory_sha256=H6_SCORING_INVENTORY_SHA256,
        expected_row_count=4104,
        result_root=evidence,
        state_root=evidence / "state",
        pointer_root=evidence / "pointer",
        pointer_name="current",
    )
    return H8H6PredictionV3Reference(
        kind="h6_prediction",
        config_schema="h6-prediction-config-v3",
        readiness_schema="h6-prediction-readiness-v3",
        raw_inventory_schema="h6-raw-endpoint-inventory-v4",
        metrics_schema="h6-prediction-metrics-v3",
        result_schema="h6-prediction-result-v3",
        artifact_path=(evidence / "RESULT").as_posix(),
        manifest_sha256=_sha(),
        result_path=(evidence / "RESULT" / "result.json").as_posix(),
        result_sha256=_sha("b"),
        content_hashes={"result.json": _sha("b")},
        payload_hashes={
            "metrics.json": _sha("c"),
            "raw_inventory.json": _sha("d"),
            "result.json": _sha("b"),
        },
        authorities_path=(evidence / "authorities").as_posix(),
        authorities_manifest_sha256=_sha("e"),
        authorities_sha256=_sha("d"),
        config_sha256=_sha("e"),
        readiness_sha256=_sha("f"),
        plan_sha256=_sha("1"),
        matching_set_sha256=_sha("2"),
        validation_bundle_path=(evidence / "validation").as_posix(),
        validation_bundle_manifest_sha256=_sha("2"),
        validation_bundle_sha256=_sha("3"),
        checkpoint_selection_sha256=_sha("4"),
        reservation_path=(evidence / "reservation.json").as_posix(),
        reservation_sha256=reservation.reservation_sha256,
        reservation_file_sha256=_sha("5"),
        terminal_path=(evidence / "state" / "TERMINAL").as_posix(),
        terminal_sha256=_sha("6"),
        terminal_manifest_sha256=_sha("6"),
        finalized_path=(evidence / "state" / "FINALIZED").as_posix(),
        finalized_manifest_sha256=_sha("6"),
        pointer_path=(evidence / "pointer" / "current").as_posix(),
        pointer_sha256=_sha("7"),
        pointer_manifest_sha256=_sha("7"),
        experiment_identity_sha256=_sha("8"),
        opening_proof_sha256=reservation.opening_proof_sha256,
        raw_inventory_sha256=_sha("0"),
        metrics_sha256=_sha("c"),
        result_record_sha256=_sha("4"),
        ledger_path=(evidence / "ledger.json").as_posix(),
        ledger_sha256=_sha("d"),
        ledger_validator_sha256=_sha("e"),
        artifact_revision=f"git:{'1' * 40}:sha256:{_sha('2')}",
        producer_head="1" * 40,
        producer_dirty_digest=_sha("2"),
        candidate_junit_path=(evidence / "candidate-junit.xml").as_posix(),
        candidate_junit_sha256=_sha("3"),
        status="pass",
    )


def _reservation_for_reference(
    reference: H8H6PredictionV3Reference,
) -> H6TestReservationV3:
    evidence = Path(reference.reservation_path).parent
    reservation = H6TestReservationV3.create(
        experiment_config_sha256=reference.config_sha256,
        readiness_sha256=reference.readiness_sha256,
        plan_sha256=reference.plan_sha256,
        experiment_identity_sha256=reference.experiment_identity_sha256,
        data_identity_sha256=_sha("a"),
        sealed_test_sha256=_sha("b"),
        test_inventory_sha256=_sha("c"),
        access_policy_sha256=_sha("d"),
        tuning_selection_sha256=_sha("e"),
        checkpoint_selection_sha256=reference.checkpoint_selection_sha256,
        validation_bundle_sha256=reference.validation_bundle_sha256,
        scoring_inventory_sha256=H6_SCORING_INVENTORY_SHA256,
        expected_row_count=4104,
        result_root=evidence,
        state_root=evidence / "state",
        pointer_root=evidence / "pointer",
        pointer_name="current",
    )
    return reservation


def _write_bound_ledger(
    tmp_path: Path,
    reference: H8H6PredictionV3Reference,
    *,
    ledger_mutator: object | None = None,
) -> tuple[H8H6PredictionV3Reference, dict[str, object]]:
    (tmp_path / ".git").mkdir(exist_ok=True)
    verification_root = tmp_path / ".verification"
    verification_root.mkdir(exist_ok=True)
    junit_path = verification_root / "candidate-junit.xml"
    junit_bytes = b'<testsuite tests="1" failures="0" errors="0"/>\n'
    junit_path.write_bytes(junit_bytes)
    artifact_root = Path(reference.artifact_path)
    artifact_root.mkdir(parents=True, exist_ok=True)
    result_path = artifact_root / "result.json"
    result_bytes = b"{}\n"
    result_path.write_bytes(result_bytes)
    manifest_path = artifact_root / "manifest.sha256"
    manifest_path.write_bytes(
        f"{hashlib.sha256(result_bytes).hexdigest()}  result.json\n".encode(
            "ascii"
        )
    )
    authorities_manifest, authorities_manifest_sha256 = _write_test_manifest(
        Path(reference.authorities_path),
        "authorities.json",
    )
    validation_manifest, validation_manifest_sha256 = _write_test_manifest(
        Path(reference.validation_bundle_path),
        "validation.json",
    )
    reservation = _reservation_for_reference(reference)
    reservation_path = Path(reference.reservation_path)
    reservation_bytes = reservation.canonical_bytes()
    reservation_path.write_bytes(reservation_bytes)
    terminal_manifest, terminal_manifest_sha256 = _write_test_manifest(
        Path(reference.terminal_path),
        "terminal.json",
    )
    finalized_manifest, finalized_manifest_sha256 = _write_test_manifest(
        Path(reference.finalized_path),
        "terminal.json",
    )
    pointer_manifest, pointer_manifest_sha256 = _write_test_manifest(
        Path(reference.pointer_path),
        "pointer.json",
    )
    reference = dataclasses.replace(
        reference,
        result_path=result_path.as_posix(),
        result_sha256=hashlib.sha256(result_bytes).hexdigest(),
        content_hashes={
            "result.json": hashlib.sha256(result_bytes).hexdigest()
        },
        payload_hashes={
            "metrics.json": reference.payload_hashes["metrics.json"],
            "raw_inventory.json": reference.payload_hashes["raw_inventory.json"],
            "result.json": hashlib.sha256(result_bytes).hexdigest(),
        },
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        authorities_manifest_sha256=authorities_manifest_sha256,
        validation_bundle_manifest_sha256=validation_manifest_sha256,
        reservation_file_sha256=hashlib.sha256(reservation_bytes).hexdigest(),
        terminal_manifest_sha256=terminal_manifest_sha256,
        finalized_manifest_sha256=finalized_manifest_sha256,
        pointer_manifest_sha256=pointer_manifest_sha256,
        ledger_validator_sha256=(
            adapter.H8_PREDICTION_V3_LEDGER_VALIDATOR_SHA256
        ),
        candidate_junit_path=junit_path.as_posix(),
        candidate_junit_sha256=hashlib.sha256(junit_bytes).hexdigest(),
    )
    evidence = [
        {
            "id": "e-manifest",
            "kind": "mechanical",
            "location": _hash_evidence_location(
                manifest_path,
                reference.manifest_sha256,
            ),
            "artifact_revision": reference.artifact_revision,
        },
        {
            "id": "e-junit",
            "kind": "mechanical",
            "location": _hash_evidence_location(
                junit_path,
                reference.candidate_junit_sha256,
            ),
            "artifact_revision": reference.artifact_revision,
        },
        {
            "id": "e-result",
            "kind": "mechanical",
            "location": _hash_evidence_location(
                result_path,
                reference.result_sha256,
            ),
            "artifact_revision": reference.artifact_revision,
        },
        {
            "id": "e-authorities",
            "kind": "mechanical",
            "location": _hash_evidence_location(
                authorities_manifest,
                reference.authorities_manifest_sha256,
            ),
            "artifact_revision": reference.artifact_revision,
        },
        {
            "id": "e-validation",
            "kind": "mechanical",
            "location": _hash_evidence_location(
                validation_manifest,
                reference.validation_bundle_manifest_sha256,
            ),
            "artifact_revision": reference.artifact_revision,
        },
        {
            "id": "e-reservation",
            "kind": "mechanical",
            "location": _hash_evidence_location(
                reservation_path,
                reference.reservation_file_sha256,
            ),
            "artifact_revision": reference.artifact_revision,
        },
        {
            "id": "e-terminal",
            "kind": "mechanical",
            "location": _hash_evidence_location(
                terminal_manifest,
                reference.terminal_manifest_sha256,
            ),
            "artifact_revision": reference.artifact_revision,
        },
        {
            "id": "e-finalized",
            "kind": "mechanical",
            "location": _hash_evidence_location(
                finalized_manifest,
                reference.finalized_manifest_sha256,
            ),
            "artifact_revision": reference.artifact_revision,
        },
        {
            "id": "e-pointer",
            "kind": "mechanical",
            "location": _hash_evidence_location(
                pointer_manifest,
                reference.pointer_manifest_sha256,
            ),
            "artifact_revision": reference.artifact_revision,
        },
    ]
    ledger: dict[str, object] = {
        "schema_version": "1.0",
        "mode": "closure",
        "artifact_revision": reference.artifact_revision,
        "claims": [
            {
                "id": _CLOSURE_CLAIM_ID,
                "domain": "experiment",
                "statement": _CLOSURE_CLAIM_STATEMENT,
                "state": "EVIDENCE_VERIFIED",
                "artifact_revision": reference.artifact_revision,
                "open_obligations": [],
                "evidence_invalidated": False,
                "evidence": evidence,
            }
        ],
    }
    if callable(ledger_mutator):
        ledger_mutator(ledger)
    ledger_path = verification_root / "h6-prediction-ledger.json"
    ledger_bytes = json.dumps(
        ledger,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    ledger_path.write_bytes(ledger_bytes)
    return (
        dataclasses.replace(
            reference,
            ledger_path=ledger_path.as_posix(),
            ledger_sha256=hashlib.sha256(ledger_bytes).hexdigest(),
        ),
        ledger,
    )


def _install_valid_native_reopeners(
    monkeypatch: pytest.MonkeyPatch,
    reference: H8H6PredictionV3Reference,
) -> list[str]:
    calls: list[str] = []
    monkeypatch.setattr(adapter, "_validate_manifest_digest", lambda *args, **kwargs: None)
    monkeypatch.setattr(adapter, "_validate_result_file", lambda *args, **kwargs: None)
    config = SimpleNamespace(
        schema_version=reference.config_schema,
        config_sha256=reference.config_sha256,
        a0_direct_exact_prefix_certificate_sha256=(
            _A0_DIRECT_EXACT_PREFIX_CERTIFICATE_SHA256
        ),
        source=SimpleNamespace(
            git_head=reference.producer_head,
            dirty_digest=reference.producer_dirty_digest,
        ),
    )
    readiness = SimpleNamespace(
        readiness_schema=reference.readiness_schema,
        readiness_sha256=reference.readiness_sha256,
        experiment_config_sha256=reference.config_sha256,
        matching_set_sha256=reference.matching_set_sha256,
        a0_direct_exact_prefix_certificate_sha256=(
            _A0_DIRECT_EXACT_PREFIX_CERTIFICATE_SHA256
        ),
    )
    plan = SimpleNamespace(
        plan_schema="h6-experiment-plan-v3",
        plan_sha256=reference.plan_sha256,
        experiment_config_sha256=reference.config_sha256,
        readiness_sha256=reference.readiness_sha256,
        matching_set_sha256=reference.matching_set_sha256,
    )
    matching = SimpleNamespace(
        schema_version="h6-amended-matching-set-v3",
        matching_set_sha256=reference.matching_set_sha256,
    )
    monkeypatch.setattr(
        adapter,
        "read_h6_prediction_v3_authorities",
        lambda *args, **kwargs: (
            calls.append("config/readiness/plan/matching")
            or SimpleNamespace(
                authority_sha256=reference.authorities_sha256,
                config=config,
                readiness=readiness,
                plan=plan,
                matching_set=matching,
            )
        ),
    )
    selection = SimpleNamespace(
        checkpoint_selection_sha256=reference.checkpoint_selection_sha256,
        experiment_config_sha256=reference.config_sha256,
        readiness_sha256=reference.readiness_sha256,
        plan_sha256=reference.plan_sha256,
        matching_set_sha256=reference.matching_set_sha256,
    )
    bundle = SimpleNamespace(
        experiment_config_sha256=reference.config_sha256,
        plan_sha256=reference.plan_sha256,
        checkpoint_selection=selection,
        validation_bundle_sha256=reference.validation_bundle_sha256,
    )
    monkeypatch.setattr(
        adapter,
        "read_h6_validation_bundle_v3",
        lambda *args, **kwargs: calls.append("validation/checkpoint") or bundle,
    )
    reservation = _reservation_for_reference(reference)
    monkeypatch.setattr(
        adapter,
        "read_h6_test_reservation_v3",
        lambda *args, **kwargs: calls.append("reservation") or reservation,
    )
    result = SimpleNamespace(
        result_schema=reference.result_schema,
        reservation_sha256=reference.reservation_sha256,
        opening_proof_sha256=reference.opening_proof_sha256,
        raw_inventory_sha256=reference.raw_inventory_sha256,
        metrics_sha256=reference.metrics_sha256,
        logical_row_count=4104,
        result_sha256=reference.result_record_sha256,
    )
    inventory = SimpleNamespace(
        inventory_schema=reference.raw_inventory_schema,
        inventory_sha256=reference.raw_inventory_sha256,
        opening_proof_sha256=reference.opening_proof_sha256,
        logical_row_count=4104,
    )
    metrics = SimpleNamespace(
        metrics_schema=reference.metrics_schema,
        raw_inventory_sha256=reference.raw_inventory_sha256,
        metrics_sha256=reference.metrics_sha256,
    )
    monkeypatch.setattr(
        adapter,
        "read_h6_prediction_result_v3",
        lambda *args, **kwargs: calls.append("raw/metrics/result")
        or (result, inventory, metrics),
    )
    terminal = SimpleNamespace(
        state="FINALIZED",
        reservation_sha256=reference.reservation_sha256,
        result_sha256=reference.result_record_sha256,
        terminal_sha256=reference.terminal_sha256,
    )
    monkeypatch.setattr(
        adapter,
        "read_h6_test_terminal_v3",
        lambda path, **kwargs: calls.append(Path(path).name) or terminal,
    )
    pointer = SimpleNamespace(
        reservation_sha256=reference.reservation_sha256,
        terminal_sha256=reference.terminal_sha256,
        result_sha256=reference.result_record_sha256,
        pointer_sha256=reference.pointer_sha256,
    )
    monkeypatch.setattr(
        adapter,
        "read_h6_prediction_pointer_v3",
        lambda *args, **kwargs: calls.append("pointer") or pointer,
    )
    monkeypatch.setattr(
        adapter,
        "_validate_ledger",
        lambda *args, **kwargs: calls.append("ledger"),
    )
    return calls


def test_h8_reopens_and_validates_h6_v3_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = _reference(tmp_path)
    calls = _install_valid_native_reopeners(monkeypatch, reference)

    adapter.validate_h8_h6_prediction_v3_reference(
        reference,
        expected_a0_direct_exact_prefix_certificate_sha256=(
            _A0_DIRECT_EXACT_PREFIX_CERTIFICATE_SHA256
        ),
    )

    assert calls == [
        "config/readiness/plan/matching",
        "validation/checkpoint",
        "reservation",
        "raw/metrics/result",
        "TERMINAL",
        "FINALIZED",
        "pointer",
        "ledger",
    ]


def test_h8_rejects_direct_a0_certificate_sha_drift_before_downstream_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = _reference(tmp_path)
    calls = _install_valid_native_reopeners(monkeypatch, reference)

    with pytest.raises(ValueError, match="authority bundle drifted"):
        adapter.validate_h8_h6_prediction_v3_reference(
            reference,
            expected_a0_direct_exact_prefix_certificate_sha256=_sha("8"),
        )

    assert calls == ["config/readiness/plan/matching"]


def test_h8_rejects_v2_or_drifted_v3_predecessor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = _reference(tmp_path)
    legacy = H8H6PredictionReference(
        kind="h6_prediction",
        prediction_schema="h6-prediction-amended-v2",
        config_schema="h6-prediction-config-v2",
        readiness_schema="h6-prediction-readiness-v2",
        metrics_schema="h6-prediction-metrics-v2",
        result_schema="h6-prediction-result-v2",
        artifact_path=reference.artifact_path,
        manifest_sha256=reference.manifest_sha256,
        result_path=reference.result_path,
        result_sha256=reference.result_sha256,
        content_hashes=reference.content_hashes,
        payload_hashes=reference.payload_hashes,
        experiment_sha256=reference.config_sha256,
        config_sha256=reference.config_sha256,
        readiness_artifact_path=reference.authorities_path,
        readiness_manifest_sha256=_sha(),
        readiness_sha256=reference.readiness_sha256,
        correctness_artifact_paths={
            gate: reference.authorities_path for gate in ("H1", "H2", "H3", "H5")
        },
        h1_prefix_prior_artifact_path=reference.authorities_path,
        smc_accuracy_artifact_path=reference.authorities_path,
        smc_accuracy_manifest_sha256=_sha(),
        h6_prefix_artifact_path=reference.authorities_path,
        h6_prefix_manifest_sha256=_sha(),
        blinded_data_artifact_path=reference.authorities_path,
        blinded_data_manifest_sha256=_sha(),
        matching_artifact_path=reference.authorities_path,
        matching_manifest_sha256=_sha(),
        matching_set_sha256=reference.matching_set_sha256,
        h1_prefix_prior_generative_factor_schema_sha256=_sha(),
        smc_bias_semantics_sha256=_sha(),
        objective_gate_spec_sha256=_sha(),
        metrics_sha256=reference.metrics_sha256,
        ledger_path=reference.ledger_path,
        ledger_sha256=reference.ledger_sha256,
        producer_head=reference.producer_head,
        producer_dirty_digest=reference.producer_dirty_digest,
        candidate_junit_sha256=reference.candidate_junit_sha256,
        status="pass",
    )
    with pytest.raises(ValueError, match="exact H6-Prediction v3"):
        adapter.validate_h8_h6_prediction_v3_reference(
            legacy,  # type: ignore[arg-type]
            expected_a0_direct_exact_prefix_certificate_sha256=(
                _A0_DIRECT_EXACT_PREFIX_CERTIFICATE_SHA256
            ),
        )

    _install_valid_native_reopeners(monkeypatch, reference)
    monkeypatch.setattr(
        adapter,
        "read_h6_test_terminal_v3",
        lambda path, **kwargs: SimpleNamespace(
            state="INCONCLUSIVE",
            reservation_sha256=reference.reservation_sha256,
            result_sha256=None,
            terminal_sha256=reference.terminal_sha256,
        ),
    )
    with pytest.raises(ValueError, match="FINALIZED"):
        adapter.validate_h8_h6_prediction_v3_reference(
            reference,
            expected_a0_direct_exact_prefix_certificate_sha256=(
                _A0_DIRECT_EXACT_PREFIX_CERTIFICATE_SHA256
            ),
        )


@pytest.mark.parametrize(
    "field_name",
    (
        "artifact_path",
        "result_path",
        "terminal_path",
        "finalized_path",
        "pointer_path",
    ),
)
def test_h8_prediction_v3_paths_are_exact_reservation_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
) -> None:
    reference = _reference(tmp_path)
    reservation = _reservation_for_reference(reference)
    reference = dataclasses.replace(
        reference,
        reservation_sha256=reservation.reservation_sha256,
        opening_proof_sha256=reservation.opening_proof_sha256,
    )
    _install_valid_native_reopeners(monkeypatch, reference)
    forged = dataclasses.replace(
        reference,
        **{field_name: (tmp_path / "forged-output").as_posix()},
    )

    with pytest.raises(ValueError, match="authenticated reservation output"):
        adapter.validate_h8_h6_prediction_v3_reference(
            forged,
            expected_a0_direct_exact_prefix_certificate_sha256=(
                _A0_DIRECT_EXACT_PREFIX_CERTIFICATE_SHA256
            ),
        )


def test_h8_prediction_v3_ledger_uses_hash_bound_validator_and_live_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference, expected_ledger = _write_bound_ledger(
        tmp_path,
        _reference(tmp_path),
    )
    calls: list[object] = []
    validator_api = SimpleNamespace(
        validate_ledger=lambda ledger: calls.append(ledger) or [],
        capture_artifact_revision=lambda root: (
            calls.append(root) or reference.artifact_revision
        ),
    )

    def load_validator(*, expected_sha256: str) -> object:
        calls.append(expected_sha256)
        return validator_api

    monkeypatch.setattr(adapter, "_load_verification_gate_api", load_validator)
    adapter._validate_ledger(reference)

    assert calls == [
        reference.ledger_validator_sha256,
        tmp_path.resolve(strict=True),
        expected_ledger,
    ]


@pytest.mark.parametrize(
    "case",
    (
        "triage",
        "empty_claims",
        "nonverified",
        "open_obligation",
        "invalidated",
        "stale_claim",
        "missing_result_identity",
        "wrong_claim_id",
        "path_only_evidence",
        "wrong_hash_evidence",
        "validator_error",
        "stale_live_revision",
    ),
)
def test_h8_prediction_v3_ledger_rejects_nonclosure_or_unbound_claims(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    def mutate(ledger: dict[str, object]) -> None:
        claims = ledger["claims"]
        assert isinstance(claims, list)
        claim = claims[0]
        assert isinstance(claim, dict)
        if case == "triage":
            ledger["mode"] = "triage"
        elif case == "empty_claims":
            ledger["claims"] = []
        elif case == "nonverified":
            claim["state"] = "INCONCLUSIVE"
        elif case == "open_obligation":
            claim["open_obligations"] = ["unresolved"]
        elif case == "invalidated":
            claim["evidence_invalidated"] = True
        elif case == "stale_claim":
            claim["artifact_revision"] = (
                f"git:{'f' * 40}:sha256:{_sha('f')}"
            )
        elif case == "missing_result_identity":
            evidence = claim["evidence"]
            assert isinstance(evidence, list)
            claim["evidence"] = [
                record
                for record in evidence
                if isinstance(record, dict) and record["id"] != "e-result"
            ]
        elif case == "wrong_claim_id":
            claim["id"] = "generic-all-green-claim"
        elif case in ("path_only_evidence", "wrong_hash_evidence"):
            evidence = claim["evidence"]
            assert isinstance(evidence, list)
            record = evidence[0]
            assert isinstance(record, dict)
            location = json.loads(record["location"])
            assert isinstance(location, dict)
            if case == "path_only_evidence":
                record["location"] = location["path"]
            else:
                location["sha256"] = _sha("f")
                record["location"] = json.dumps(
                    location,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )

    reference, _ledger = _write_bound_ledger(
        tmp_path,
        _reference(tmp_path),
        ledger_mutator=mutate,
    )
    validator_errors = ["invalid ledger"] if case == "validator_error" else []
    live_revision = (
        f"git:{'f' * 40}:sha256:{_sha('f')}"
        if case == "stale_live_revision"
        else reference.artifact_revision
    )
    monkeypatch.setattr(
        adapter,
        "_load_verification_gate_api",
        lambda **_kwargs: SimpleNamespace(
            validate_ledger=lambda _ledger: validator_errors,
            capture_artifact_revision=lambda _root: live_revision,
        ),
    )

    with pytest.raises(
        ValueError,
        match="ledger|revision|identity|closure|canonical",
    ):
        adapter._validate_ledger(reference)


@pytest.mark.parametrize(
    "artifact_name",
    (
        "candidate_junit",
        "result",
        "result_manifest",
        "authorities_manifest",
        "validation_manifest",
        "reservation",
        "terminal_manifest",
        "finalized_manifest",
        "pointer_manifest",
    ),
)
def test_h8_prediction_v3_closure_rehashes_every_referenced_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_name: str,
) -> None:
    reference, _ledger = _write_bound_ledger(
        tmp_path,
        _reference(tmp_path),
    )
    paths = {
        "candidate_junit": Path(reference.candidate_junit_path),
        "result": Path(reference.result_path),
        "result_manifest": Path(reference.artifact_path) / "manifest.sha256",
        "authorities_manifest": (
            Path(reference.authorities_path) / "manifest.sha256"
        ),
        "validation_manifest": (
            Path(reference.validation_bundle_path) / "manifest.sha256"
        ),
        "reservation": Path(reference.reservation_path),
        "terminal_manifest": (
            Path(reference.terminal_path) / "manifest.sha256"
        ),
        "finalized_manifest": (
            Path(reference.finalized_path) / "manifest.sha256"
        ),
        "pointer_manifest": (
            Path(reference.pointer_path) / "manifest.sha256"
        ),
    }
    target = paths[artifact_name]
    target.write_bytes(target.read_bytes() + b"x")
    monkeypatch.setattr(
        adapter,
        "_load_verification_gate_api",
        lambda **_kwargs: SimpleNamespace(
            validate_ledger=lambda _ledger: [],
            capture_artifact_revision=lambda _root: (
                reference.artifact_revision
            ),
        ),
    )

    with pytest.raises(ValueError, match="differs|changed"):
        adapter._validate_ledger(reference)


def test_h8_prediction_v3_loads_exact_installed_ledger_validator() -> None:
    validator_path = adapter._verification_ledger_validator_path()
    validator_sha256 = hashlib.sha256(validator_path.read_bytes()).hexdigest()
    api = adapter._load_verification_gate_api(
        expected_sha256=validator_sha256,
    )

    assert callable(api.validate_ledger)
    assert callable(api.capture_artifact_revision)
    with pytest.raises(ValueError, match="pinned"):
        adapter._load_verification_gate_api(expected_sha256=_sha("f"))


def test_h8_prediction_v3_rejects_self_consistent_unpinned_validator_before_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alternate = tmp_path / "verification_gate.py"
    alternate.write_text(
        "def validate_ledger(_ledger): return []\n"
        "def capture_artifact_revision(_root): return 'forged'\n",
        encoding="utf-8",
    )
    alternate_sha256 = hashlib.sha256(alternate.read_bytes()).hexdigest()
    imports: list[Path] = []
    monkeypatch.setattr(
        adapter,
        "_verification_ledger_validator_path",
        lambda: alternate,
    )

    def unexpected_import(_name: str, path: Path) -> object:
        imports.append(path)
        pytest.fail("an unpinned validator must be rejected before import")

    monkeypatch.setattr(
        adapter.importlib.util,
        "spec_from_file_location",
        unexpected_import,
    )
    with pytest.raises(ValueError, match="pinned"):
        adapter._load_verification_gate_api(
            expected_sha256=alternate_sha256,
        )
    assert imports == []


@pytest.mark.parametrize(
    "unsafe_name",
    (
        "../ledger.json",
        "/absolute.json",
        "C:/absolute.json",
        "nested/../result.json",
        r"nested\result.json",
        "./result.json",
    ),
)
def test_h8_prediction_v3_content_hash_names_are_safe_manifest_relative(
    tmp_path: Path,
    unsafe_name: str,
) -> None:
    reference = _reference(tmp_path)
    with pytest.raises(ValueError, match="safe manifest-relative"):
        dataclasses.replace(
            reference,
            content_hashes={unsafe_name: reference.result_sha256},
        )
