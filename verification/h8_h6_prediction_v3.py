"""Exact H8 adapter for the executable H6-Prediction v3 predecessor."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import NoReturn, cast

from vfe4.artifacts.h6_prediction_v3 import (
    read_h6_prediction_result_v3,
    read_h6_validation_bundle_v3,
)
from vfe4.training.h6_test_transaction_v3 import H6TestReservationV3
from vfe4.types.h8 import H8H6PredictionV3Reference


_MAXIMUM_MANIFEST_BYTES = 4 * 1024
_MAXIMUM_RESULT_PAYLOAD_BYTES = 32 * 1024 * 1024
_MAXIMUM_LEDGER_BYTES = 16 * 1024 * 1024
_MAXIMUM_JUNIT_BYTES = 64 * 1024 * 1024
H8_PREDICTION_V3_LEDGER_VALIDATOR_SHA256 = (
    "a8a799496762910c463ecc179a4d63dc40107fcbe81553add189de7ed1ce4c95"
)
H6_PREDICTION_V3_CLOSURE_CLAIM_ID = (
    "h6-prediction-v3-exact-artifact-closure"
)
H6_PREDICTION_V3_CLOSURE_CLAIM_STATEMENT = (
    "The exact H6-Prediction v3 candidate JUnit, authorities, validation, "
    "one-shot transaction, and result artifacts are evidence-verified at "
    "the live producer revision."
)


def _missing_native_reader(*_args: object, **_kwargs: object) -> NoReturn:
    raise RuntimeError("required H6-Prediction v3 native reader is unavailable")


try:
    from vfe4.artifacts.h6_prediction_v3 import (
        read_h6_prediction_v3_authorities,
    )
except ImportError:
    read_h6_prediction_v3_authorities = _missing_native_reader

try:
    from vfe4.training.h6_test_transaction_v3 import (
        read_h6_prediction_pointer_v3,
        read_h6_test_reservation_v3,
        read_h6_test_terminal_v3,
    )
except ImportError:
    read_h6_prediction_pointer_v3 = _missing_native_reader
    read_h6_test_reservation_v3 = _missing_native_reader
    read_h6_test_terminal_v3 = _missing_native_reader


def _read_regular_file(
    path: Path,
    *,
    name: str,
    maximum_bytes: int,
) -> bytes:
    if type(maximum_bytes) is not int or maximum_bytes <= 0:
        raise ValueError(f"{name} maximum byte count must be positive")
    if not path.is_absolute():
        raise ValueError(f"{name} path must be absolute")
    try:
        parent_before = path.parent.lstat()
        before = path.lstat()
    except OSError as exc:
        raise ValueError(f"{name} is unavailable") from exc
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if (
        not stat.S_ISDIR(parent_before.st_mode)
        or stat.S_ISLNK(parent_before.st_mode)
        or bool(
            getattr(parent_before, "st_file_attributes", 0)
            & reparse_flag
        )
        or not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or bool(getattr(before, "st_file_attributes", 0) & reparse_flag)
        or before.st_size > maximum_bytes
    ):
        raise ValueError(f"{name} must be a bounded regular nonredirected file")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        if (
            opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
            or not stat.S_ISREG(opened.st_mode)
            or opened.st_size != before.st_size
            or opened.st_size > maximum_bytes
        ):
            raise ValueError(f"{name} identity changed before open")
        chunks: list[bytes] = []
        total = 0
        while total <= maximum_bytes:
            chunk = os.read(
                descriptor,
                min(65_536, maximum_bytes + 1 - total),
            )
            if not chunk:
                break
            total += len(chunk)
            chunks.append(chunk)
        after = os.fstat(descriptor)
        path_after = path.lstat()
        parent_after = path.parent.lstat()
        if (
            total > maximum_bytes
            or after.st_dev != opened.st_dev
            or after.st_ino != opened.st_ino
            or after.st_size != opened.st_size
            or after.st_mtime_ns != opened.st_mtime_ns
            or path_after.st_dev != opened.st_dev
            or path_after.st_ino != opened.st_ino
            or path_after.st_size != opened.st_size
            or parent_after.st_dev != parent_before.st_dev
            or parent_after.st_ino != parent_before.st_ino
        ):
            raise ValueError(f"{name} changed while reading")
        content = b"".join(chunks)
        if len(content) != opened.st_size:
            raise ValueError(f"{name} length changed while reading")
        return content
    finally:
        os.close(descriptor)


def _validate_manifest_digest(
    root: Path,
    expected_sha256: str,
    *,
    name: str,
    expected_payload_hashes: Mapping[str, str] | None = None,
) -> None:
    manifest = _read_regular_file(
        root / "manifest.sha256",
        name=f"{name} manifest",
        maximum_bytes=_MAXIMUM_MANIFEST_BYTES,
    )
    if hashlib.sha256(manifest).hexdigest() != expected_sha256:
        raise ValueError(f"{name} manifest differs from the registry reference")
    if expected_payload_hashes is not None:
        expected = "".join(
            f"{digest}  {payload_name}\n"
            for payload_name, digest in expected_payload_hashes.items()
        ).encode("ascii")
        if manifest != expected:
            raise ValueError(f"{name} manifest inventory differs from the reference")


def _validate_result_file(reference: H8H6PredictionV3Reference) -> None:
    result_path = Path(reference.result_path)
    expected_path = Path(reference.artifact_path) / "result.json"
    if result_path.resolve(strict=False) != expected_path.resolve(strict=False):
        raise ValueError("H6-Prediction v3 result path is not its native result file")
    raw = _read_regular_file(
        result_path,
        name="H6-Prediction v3 result",
        maximum_bytes=_MAXIMUM_RESULT_PAYLOAD_BYTES,
    )
    if hashlib.sha256(raw).hexdigest() != reference.result_sha256:
        raise ValueError("H6-Prediction v3 result bytes differ from the reference")
    artifact_root = Path(reference.artifact_path)
    for payload_name, expected_sha256 in reference.content_hashes.items():
        relative = PurePosixPath(payload_name)
        if (
            payload_name != relative.as_posix()
            or relative.is_absolute()
            or "." in relative.parts
            or ".." in relative.parts
            or "\\" in payload_name
            or ":" in payload_name
            or payload_name not in reference.payload_hashes
        ):
            raise ValueError(
                "H6-Prediction v3 content hash name is not an exact safe "
                "manifest-relative name"
            )
        payload = _read_regular_file(
            artifact_root.joinpath(*relative.parts),
            name=f"H6-Prediction v3 content {payload_name}",
            maximum_bytes=_MAXIMUM_RESULT_PAYLOAD_BYTES,
        )
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise ValueError(
                "H6-Prediction v3 content bytes differ from the reference"
            )


@dataclass(frozen=True)
class _VerificationGateApi:
    validate_ledger: Callable[[dict[str, object]], list[str]]
    capture_artifact_revision: Callable[..., str]


def _verification_ledger_validator_path() -> Path:
    path = (
        Path.home()
        / ".codex"
        / "skills"
        / "verification"
        / "scripts"
        / "verification_gate.py"
    ).resolve(strict=True)
    if not path.is_file() or path.is_symlink():
        raise ValueError("installed verification-ledger validator is unavailable")
    return path


def _load_verification_gate_api(
    *,
    expected_sha256: str,
) -> _VerificationGateApi:
    """Load the exact installed deterministic validator bound by source hash."""

    if expected_sha256 != H8_PREDICTION_V3_LEDGER_VALIDATOR_SHA256:
        raise ValueError(
            "H6-Prediction v3 ledger validator differs from the pinned "
            "tracked dependency"
        )
    path = _verification_ledger_validator_path()
    before = path.read_bytes()
    if hashlib.sha256(before).hexdigest() != expected_sha256:
        raise ValueError("verification-ledger validator changed after source capture")
    spec = importlib.util.spec_from_file_location(
        "_vfe4_h8_prediction_v3_ledger_validator",
        path,
    )
    if spec is None or spec.loader is None:
        raise ValueError("verification-ledger validator cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("verification-ledger validator changed while loading")
    validate_ledger = getattr(module, "validate_ledger", None)
    capture_revision = getattr(module, "capture_artifact_revision", None)
    if not callable(validate_ledger) or not callable(capture_revision):
        raise ValueError("verification-ledger validator API is incomplete")
    return _VerificationGateApi(
        validate_ledger=cast(
            Callable[[dict[str, object]], list[str]],
            validate_ledger,
        ),
        capture_artifact_revision=cast(Callable[..., str], capture_revision),
    )


def _prediction_repository_root(
    reference: H8H6PredictionV3Reference,
) -> Path:
    ledger_path = Path(reference.ledger_path)
    junit_path = Path(reference.candidate_junit_path)
    if not ledger_path.is_absolute() or not junit_path.is_absolute():
        raise ValueError("prediction ledger and JUnit paths must be absolute")
    root: Path | None = None
    for candidate in (ledger_path.parent, *ledger_path.parents):
        if os.path.lexists(candidate / ".git"):
            root = candidate.resolve(strict=True)
            break
    if root is None or not root.is_dir() or root.is_symlink():
        raise ValueError("prediction ledger is not inside one Git worktree")
    verification_root = root / ".verification"
    for path, name in (
        (ledger_path, "prediction ledger"),
        (junit_path, "candidate JUnit"),
    ):
        try:
            path.resolve(strict=False).relative_to(verification_root)
        except ValueError as exc:
            raise ValueError(f"{name} is outside the candidate worktree") from exc
    return root


def _decode_ledger(raw: bytes) -> dict[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("prediction ledger contains duplicate JSON keys")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("prediction ledger is not strict JSON") from exc
    if type(value) is not dict:
        raise ValueError("prediction ledger must encode one JSON object")
    return cast(dict[str, object], value)


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


def _recompute_evidence_location(
    path: Path,
    *,
    expected_sha256: str,
    name: str,
    maximum_bytes: int,
) -> str:
    raw = _read_regular_file(
        path,
        name=name,
        maximum_bytes=maximum_bytes,
    )
    observed_sha256 = hashlib.sha256(raw).hexdigest()
    if observed_sha256 != expected_sha256:
        raise ValueError(f"{name} differs from the H8 v5 reference")
    return _hash_evidence_location(path, observed_sha256)


def _validate_ledger(reference: H8H6PredictionV3Reference) -> None:
    raw = _read_regular_file(
        Path(reference.ledger_path),
        name="H6-Prediction v3 ledger",
        maximum_bytes=_MAXIMUM_LEDGER_BYTES,
    )
    if hashlib.sha256(raw).hexdigest() != reference.ledger_sha256:
        raise ValueError("H6-Prediction v3 ledger differs from the reference")
    ledger = _decode_ledger(raw)
    validator_api = _load_verification_gate_api(
        expected_sha256=reference.ledger_validator_sha256,
    )
    repository_root = _prediction_repository_root(reference)
    try:
        live_artifact_revision = validator_api.capture_artifact_revision(
            repository_root
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(
            "H6-Prediction v3 live artifact revision is unavailable"
        ) from exc
    if live_artifact_revision != reference.artifact_revision:
        raise ValueError(
            "H6-Prediction v3 ledger artifact revision is not live"
        )
    errors = validator_api.validate_ledger(ledger)
    if type(errors) is not list or any(type(error) is not str for error in errors):
        raise ValueError("deterministic ledger validator returned an invalid result")
    if errors:
        raise ValueError(
            "deterministic ledger validation failed: " + "; ".join(errors)
        )
    claims = ledger.get("claims")
    if (
        ledger.get("schema_version") != "1.0"
        or ledger.get("mode") != "closure"
        or ledger.get("artifact_revision") != live_artifact_revision
        or not isinstance(claims, list)
        or not claims
        or any(
            not isinstance(claim, Mapping)
            or claim.get("artifact_revision") != live_artifact_revision
            or claim.get("state") != "EVIDENCE_VERIFIED"
            or claim.get("open_obligations") != []
            or claim.get("evidence_invalidated") is not False
            for claim in claims
        )
    ):
        raise ValueError(
            "H6-Prediction v3 ledger is not an all-verified closure ledger"
        )

    exact_closure_claims = tuple(
        claim
        for claim in claims
        if isinstance(claim, Mapping)
        and claim.get("id") == H6_PREDICTION_V3_CLOSURE_CLAIM_ID
    )
    if len(exact_closure_claims) != 1:
        raise ValueError(
            "H6-Prediction v3 ledger lacks one exact closure claim"
        )
    closure_claim = exact_closure_claims[0]
    if (
        closure_claim.get("domain") != "experiment"
        or closure_claim.get("statement")
        != H6_PREDICTION_V3_CLOSURE_CLAIM_STATEMENT
    ):
        raise ValueError(
            "H6-Prediction v3 closure claim contract is not exact"
        )

    manifest_path = Path(reference.artifact_path) / "manifest.sha256"
    required_locations = {
        _recompute_evidence_location(
            Path(reference.candidate_junit_path),
            expected_sha256=reference.candidate_junit_sha256,
            name="H6-Prediction v3 candidate JUnit",
            maximum_bytes=_MAXIMUM_JUNIT_BYTES,
        ),
        _recompute_evidence_location(
            Path(reference.result_path),
            expected_sha256=reference.result_sha256,
            name="H6-Prediction v3 result",
            maximum_bytes=_MAXIMUM_RESULT_PAYLOAD_BYTES,
        ),
        _recompute_evidence_location(
            manifest_path,
            expected_sha256=reference.manifest_sha256,
            name="H6-Prediction v3 result manifest",
            maximum_bytes=_MAXIMUM_MANIFEST_BYTES,
        ),
        _recompute_evidence_location(
            Path(reference.authorities_path) / "manifest.sha256",
            expected_sha256=reference.authorities_manifest_sha256,
            name="H6-Prediction v3 authorities manifest",
            maximum_bytes=_MAXIMUM_MANIFEST_BYTES,
        ),
        _recompute_evidence_location(
            Path(reference.validation_bundle_path) / "manifest.sha256",
            expected_sha256=reference.validation_bundle_manifest_sha256,
            name="H6-Prediction v3 validation manifest",
            maximum_bytes=_MAXIMUM_MANIFEST_BYTES,
        ),
        _recompute_evidence_location(
            Path(reference.reservation_path),
            expected_sha256=reference.reservation_file_sha256,
            name="H6-Prediction v3 reservation marker",
            maximum_bytes=_MAXIMUM_RESULT_PAYLOAD_BYTES,
        ),
        _recompute_evidence_location(
            Path(reference.terminal_path) / "manifest.sha256",
            expected_sha256=reference.terminal_manifest_sha256,
            name="H6-Prediction v3 terminal manifest",
            maximum_bytes=_MAXIMUM_MANIFEST_BYTES,
        ),
        _recompute_evidence_location(
            Path(reference.finalized_path) / "manifest.sha256",
            expected_sha256=reference.finalized_manifest_sha256,
            name="H6-Prediction v3 finalized manifest",
            maximum_bytes=_MAXIMUM_MANIFEST_BYTES,
        ),
        _recompute_evidence_location(
            Path(reference.pointer_path) / "manifest.sha256",
            expected_sha256=reference.pointer_manifest_sha256,
            name="H6-Prediction v3 pointer manifest",
            maximum_bytes=_MAXIMUM_MANIFEST_BYTES,
        ),
    }
    evidence = closure_claim.get("evidence")
    if not isinstance(evidence, list):
        raise ValueError("H6-Prediction v3 closure claim lacks evidence")
    current_mechanical_locations = {
        record.get("location")
        for record in evidence
        if isinstance(record, Mapping)
        and record.get("kind") in ("mechanical", "reproduced_output")
        and record.get("artifact_revision") == live_artifact_revision
        and type(record.get("location")) is str
    }
    if not required_locations.issubset(current_mechanical_locations):
        raise ValueError(
            "H6-Prediction v3 closure claim lacks canonical hash-bearing "
            "evidence for every exact artifact identity"
        )


def validate_h8_h6_prediction_v3_reference(
    reference: H8H6PredictionV3Reference,
    *,
    expected_a0_direct_exact_prefix_certificate_sha256: str,
) -> None:
    """Reopen every native v3 authority and require one FINALIZED transaction."""

    if type(reference) is not H8H6PredictionV3Reference:
        raise ValueError("H8 requires an exact H6-Prediction v3 reference")
    reference.__post_init__()

    authorities_root = Path(reference.authorities_path)
    _validate_manifest_digest(
        authorities_root,
        reference.authorities_manifest_sha256,
        name="H6-Prediction v3 authorities",
    )
    authorities = read_h6_prediction_v3_authorities(
        authorities_root,
        expected_authority_sha256=reference.authorities_sha256,
    )
    config = authorities.config
    readiness = authorities.readiness
    plan = authorities.plan
    matching_set = authorities.matching_set
    source = config.source
    if (
        authorities.authority_sha256 != reference.authorities_sha256
        or config.schema_version != reference.config_schema
        or config.config_sha256 != reference.config_sha256
        or source.git_head != reference.producer_head
        or source.dirty_digest != reference.producer_dirty_digest
        or readiness.readiness_schema != reference.readiness_schema
        or readiness.readiness_sha256 != reference.readiness_sha256
        or readiness.experiment_config_sha256 != reference.config_sha256
        or config.a0_direct_exact_prefix_certificate_sha256
        != expected_a0_direct_exact_prefix_certificate_sha256
        or readiness.a0_direct_exact_prefix_certificate_sha256
        != expected_a0_direct_exact_prefix_certificate_sha256
        or readiness.matching_set_sha256 != reference.matching_set_sha256
        or plan.plan_schema != "h6-experiment-plan-v3"
        or plan.plan_sha256 != reference.plan_sha256
        or plan.experiment_config_sha256 != reference.config_sha256
        or plan.readiness_sha256 != reference.readiness_sha256
        or plan.matching_set_sha256 != reference.matching_set_sha256
        or matching_set.schema_version != "h6-amended-matching-set-v3"
        or matching_set.matching_set_sha256 != reference.matching_set_sha256
    ):
        raise ValueError("H6-Prediction v3 authority bundle drifted")

    bundle = read_h6_validation_bundle_v3(
        Path(reference.validation_bundle_path),
        expected_plan_sha256=reference.plan_sha256,
        expected_experiment_config_sha256=reference.config_sha256,
        expected_validation_bundle_sha256=reference.validation_bundle_sha256,
    )
    selection = bundle.checkpoint_selection
    if (
        bundle.experiment_config_sha256 != reference.config_sha256
        or bundle.plan_sha256 != reference.plan_sha256
        or bundle.validation_bundle_sha256 != reference.validation_bundle_sha256
        or selection.checkpoint_selection_sha256
        != reference.checkpoint_selection_sha256
        or selection.experiment_config_sha256 != reference.config_sha256
        or selection.readiness_sha256 != reference.readiness_sha256
        or selection.plan_sha256 != reference.plan_sha256
        or selection.matching_set_sha256 != reference.matching_set_sha256
    ):
        raise ValueError("H6-Prediction v3 validation/checkpoint selection drifted")

    reservation = read_h6_test_reservation_v3(
        Path(reference.reservation_path),
    )
    if (
        type(reservation) is not H6TestReservationV3
        or reservation.state != "RESERVED"
        or reservation.experiment_config_sha256 != reference.config_sha256
        or reservation.readiness_sha256 != reference.readiness_sha256
        or reservation.plan_sha256 != reference.plan_sha256
        or reservation.experiment_identity_sha256
        != reference.experiment_identity_sha256
        or reservation.checkpoint_selection_sha256
        != reference.checkpoint_selection_sha256
        or reservation.validation_bundle_sha256
        != reference.validation_bundle_sha256
        or reservation.opening_proof_sha256 != reference.opening_proof_sha256
        or reservation.reservation_sha256 != reference.reservation_sha256
        or reservation.expected_row_count != 4104
    ):
        raise ValueError("H6-Prediction v3 reservation journal drifted")
    expected_output_paths = {
        "artifact_path": Path(reservation.result_root_path) / "RESULT",
        "result_path": (
            Path(reservation.result_root_path) / "RESULT" / "result.json"
        ),
        "terminal_path": Path(reservation.state_root_path) / "TERMINAL",
        "finalized_path": Path(reservation.state_root_path) / "FINALIZED",
        "pointer_path": (
            Path(reservation.pointer_root_path) / reservation.pointer_name
        ),
    }
    for field_name, expected_path in expected_output_paths.items():
        declared_path = Path(getattr(reference, field_name))
        if (
            not declared_path.is_absolute()
            or declared_path.resolve(strict=False)
            != expected_path.resolve(strict=False)
        ):
            raise ValueError(
                "H6-Prediction v3 path differs from its authenticated "
                f"reservation output: {field_name}"
            )

    _validate_manifest_digest(
        Path(reference.artifact_path),
        reference.manifest_sha256,
        name="H6-Prediction v3 result",
        expected_payload_hashes=reference.payload_hashes,
    )
    _validate_result_file(reference)
    result, inventory, metrics = read_h6_prediction_result_v3(
        Path(reference.artifact_path),
        expected_result_sha256=reference.result_record_sha256,
    )
    if (
        result.result_schema != reference.result_schema
        or result.reservation_sha256 != reference.reservation_sha256
        or result.opening_proof_sha256 != reference.opening_proof_sha256
        or result.raw_inventory_sha256 != reference.raw_inventory_sha256
        or result.metrics_sha256 != reference.metrics_sha256
        or result.logical_row_count != 4104
        or result.result_sha256 != reference.result_record_sha256
        or inventory.inventory_schema != reference.raw_inventory_schema
        or inventory.inventory_sha256 != reference.raw_inventory_sha256
        or inventory.opening_proof_sha256 != reference.opening_proof_sha256
        or inventory.logical_row_count != 4104
        or metrics.metrics_schema != reference.metrics_schema
        or metrics.raw_inventory_sha256 != reference.raw_inventory_sha256
        or metrics.metrics_sha256 != reference.metrics_sha256
    ):
        raise ValueError("H6-Prediction v3 raw/metrics/result authority drifted")

    terminal = read_h6_test_terminal_v3(
        Path(reference.terminal_path),
    )
    finalized = read_h6_test_terminal_v3(
        Path(reference.finalized_path),
    )
    if (
        terminal.state != "FINALIZED"
        or finalized.state != "FINALIZED"
        or terminal.reservation_sha256 != reference.reservation_sha256
        or finalized.reservation_sha256 != reference.reservation_sha256
        or terminal.result_sha256 != reference.result_record_sha256
        or finalized.result_sha256 != reference.result_record_sha256
        or terminal.terminal_sha256 != reference.terminal_sha256
        or finalized.terminal_sha256 != reference.terminal_sha256
    ):
        raise ValueError("H6-Prediction v3 transaction is not exact FINALIZED")

    pointer = read_h6_prediction_pointer_v3(
        Path(reference.pointer_path),
    )
    if (
        pointer.reservation_sha256 != reference.reservation_sha256
        or pointer.terminal_sha256 != reference.terminal_sha256
        or pointer.result_sha256 != reference.result_record_sha256
        or pointer.pointer_sha256 != reference.pointer_sha256
    ):
        raise ValueError("H6-Prediction v3 result pointer drifted")
    _validate_ledger(reference)


__all__ = ["validate_h8_h6_prediction_v3_reference"]
