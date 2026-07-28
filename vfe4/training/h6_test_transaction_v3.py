"""One-shot H6-Prediction v3 held-out scoring transaction."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from vfe4.artifacts.atomic import (
    ArtifactPublicationError,
    canonical_json_bytes,
    publish_run_directory,
)
from vfe4.artifacts.h6_prediction_v3 import (
    H6PredictionMetricsV3,
    H6PredictionResultV3,
    H6RawEndpointInventoryV4,
    H6ValidationBundleV3,
    publish_h6_prediction_result_v3,
    read_h6_prediction_result_v3,
)
from vfe4.config.schema import H6PredictionV3ResolvedConfig
from vfe4.data.wikitext2 import BlindedCorpusStore
from vfe4.training.h6_experiment_v3 import H6ExperimentPlanV3
from vfe4.types.h6 import ExperimentIdentity
from vfe4.types.h6_prediction_v3 import (
    H6PredictionV3ReadinessToken,
    H6_SCORING_INVENTORY_SHA256,
)


_LOWER_HEX = frozenset("0123456789abcdef")
_RESERVATION_SCHEMA = "h6-test-transaction-reservation-v3"
_TERMINAL_SCHEMA = "h6-test-transaction-terminal-v3"
_POINTER_SCHEMA = "h6-prediction-pointer-v3"
_ATOMICITY_DISCLOSURE = "individual-no-replace-directories-no-cross-directory-atomicity"


def _hash(domain: str, payload: object) -> str:
    return hashlib.sha256(
        domain.encode("ascii") + b"\x00" + canonical_json_bytes(payload)
    ).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _canonical_output_root(path: Path, name: str) -> tuple[str, tuple[int, int]]:
    if not isinstance(path, Path):
        raise ValueError(f"{name} must be a pathlib Path")
    resolved = path.resolve(strict=False)
    try:
        resolved.mkdir(parents=True, exist_ok=True)
        status = resolved.lstat()
    except OSError as exc:
        raise ValueError(f"{name} cannot be prepared") from exc
    if not stat.S_ISDIR(status.st_mode) or _is_redirect(resolved, status):
        raise ValueError(f"{name} must be a nonredirected directory")
    return resolved.as_posix(), (status.st_dev, status.st_ino)


def _require_output_root(
    path_value: object,
    identity: object,
    name: str,
) -> Path:
    if type(path_value) is not str or not path_value:
        raise ValueError(f"{name}_path must be canonical")
    path = Path(path_value)
    if not path.is_absolute() or path.resolve(strict=False).as_posix() != path_value:
        raise ValueError(f"{name}_path must be canonical and absolute")
    if (
        type(identity) is not tuple
        or len(identity) != 2
        or any(type(value) is not int for value in identity)
    ):
        raise ValueError(f"{name}_identity must be an exact directory identity")
    try:
        status = path.lstat()
    except OSError as exc:
        raise ValueError(f"{name} is unavailable") from exc
    if (
        not stat.S_ISDIR(status.st_mode)
        or _is_redirect(path, status)
        or (status.st_dev, status.st_ino) != identity
    ):
        raise ValueError(f"{name} identity changed")
    return path


def _require_pointer_name(value: object) -> str:
    if (
        type(value) is not str
        or not value
        or value in (".", "..")
        or "/" in value
        or "\\" in value
    ):
        raise ValueError("pointer_name must be one portable component")
    return value


@dataclass(frozen=True, slots=True)
class H6TestReservationV3:
    reservation_schema: Literal["h6-test-transaction-reservation-v3"]
    state: Literal["RESERVED"]
    experiment_config_sha256: str
    readiness_sha256: str
    plan_sha256: str
    experiment_identity_sha256: str
    data_identity_sha256: str
    sealed_test_sha256: str
    test_inventory_sha256: str
    access_policy_sha256: str
    tuning_selection_sha256: str
    checkpoint_selection_sha256: str
    validation_bundle_sha256: str
    scoring_inventory_sha256: str
    expected_row_count: Literal[4104]
    result_root_path: str
    result_root_identity: tuple[int, int]
    state_root_path: str
    state_root_identity: tuple[int, int]
    pointer_root_path: str
    pointer_root_identity: tuple[int, int]
    pointer_name: str
    output_namespace_sha256: str
    opening_proof_sha256: str
    reservation_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "reservation_schema": self.reservation_schema,
            "state": self.state,
            "experiment_config_sha256": self.experiment_config_sha256,
            "readiness_sha256": self.readiness_sha256,
            "plan_sha256": self.plan_sha256,
            "experiment_identity_sha256": self.experiment_identity_sha256,
            "data_identity_sha256": self.data_identity_sha256,
            "sealed_test_sha256": self.sealed_test_sha256,
            "test_inventory_sha256": self.test_inventory_sha256,
            "access_policy_sha256": self.access_policy_sha256,
            "tuning_selection_sha256": self.tuning_selection_sha256,
            "checkpoint_selection_sha256": self.checkpoint_selection_sha256,
            "validation_bundle_sha256": self.validation_bundle_sha256,
            "scoring_inventory_sha256": self.scoring_inventory_sha256,
            "expected_row_count": self.expected_row_count,
            "result_root_path": self.result_root_path,
            "result_root_identity": self.result_root_identity,
            "state_root_path": self.state_root_path,
            "state_root_identity": self.state_root_identity,
            "pointer_root_path": self.pointer_root_path,
            "pointer_root_identity": self.pointer_root_identity,
            "pointer_name": self.pointer_name,
            "output_namespace_sha256": self.output_namespace_sha256,
            "opening_proof_sha256": self.opening_proof_sha256,
        }

    def artifact_payload(self) -> dict[str, object]:
        return {
            **self.semantic_payload(),
            "reservation_sha256": self.reservation_sha256,
        }

    def opening_payload(self) -> dict[str, object]:
        payload = self.semantic_payload()
        payload.pop("opening_proof_sha256")
        return payload

    def canonical_bytes(self) -> bytes:
        self.__post_init__()
        return canonical_json_bytes(self.artifact_payload())

    def __post_init__(self) -> None:
        if (
            self.reservation_schema != _RESERVATION_SCHEMA
            or self.state != "RESERVED"
            or self.expected_row_count != 4104
            or self.scoring_inventory_sha256 != H6_SCORING_INVENTORY_SHA256
        ):
            raise ValueError("H6 test reservation is not the frozen RESERVED record")
        for name in (
            "experiment_config_sha256",
            "readiness_sha256",
            "plan_sha256",
            "experiment_identity_sha256",
            "data_identity_sha256",
            "sealed_test_sha256",
            "test_inventory_sha256",
            "access_policy_sha256",
            "tuning_selection_sha256",
            "checkpoint_selection_sha256",
            "validation_bundle_sha256",
            "scoring_inventory_sha256",
            "output_namespace_sha256",
            "opening_proof_sha256",
            "reservation_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        _require_output_root(
            self.result_root_path,
            self.result_root_identity,
            "result_root",
        )
        _require_output_root(
            self.state_root_path,
            self.state_root_identity,
            "state_root",
        )
        _require_output_root(
            self.pointer_root_path,
            self.pointer_root_identity,
            "pointer_root",
        )
        _require_pointer_name(self.pointer_name)
        output_payload = {
            "result_root_path": self.result_root_path,
            "result_root_identity": self.result_root_identity,
            "state_root_path": self.state_root_path,
            "state_root_identity": self.state_root_identity,
            "pointer_root_path": self.pointer_root_path,
            "pointer_root_identity": self.pointer_root_identity,
            "pointer_name": self.pointer_name,
        }
        if self.output_namespace_sha256 != _hash(
            "vfe4.h6.test-output-namespace.v3",
            output_payload,
        ):
            raise ValueError("output namespace SHA-256 does not match destinations")
        if self.opening_proof_sha256 != _hash(
            "vfe4.h6.test-opening-proof.v3",
            self.opening_payload(),
        ):
            raise ValueError("opening proof SHA-256 does not match reservation")
        if self.reservation_sha256 != _hash(
            "vfe4.h6.test-transaction-reservation.v3",
            self.semantic_payload(),
        ):
            raise ValueError("reservation SHA-256 does not match its authorities")

    @classmethod
    def create(
        cls,
        *,
        experiment_config_sha256: str,
        readiness_sha256: str,
        plan_sha256: str,
        experiment_identity_sha256: str,
        data_identity_sha256: str,
        sealed_test_sha256: str,
        test_inventory_sha256: str,
        access_policy_sha256: str,
        tuning_selection_sha256: str,
        checkpoint_selection_sha256: str,
        validation_bundle_sha256: str,
        scoring_inventory_sha256: str,
        expected_row_count: Literal[4104],
        result_root: Path,
        state_root: Path,
        pointer_root: Path,
        pointer_name: str,
    ) -> H6TestReservationV3:
        result_path, result_identity = _canonical_output_root(
            result_root,
            "result_root",
        )
        state_path, state_identity = _canonical_output_root(
            state_root,
            "state_root",
        )
        pointer_path, pointer_identity = _canonical_output_root(
            pointer_root,
            "pointer_root",
        )
        pointer_component = _require_pointer_name(pointer_name)
        output_payload = {
            "result_root_path": result_path,
            "result_root_identity": result_identity,
            "state_root_path": state_path,
            "state_root_identity": state_identity,
            "pointer_root_path": pointer_path,
            "pointer_root_identity": pointer_identity,
            "pointer_name": pointer_component,
        }
        opening_payload = {
            "reservation_schema": _RESERVATION_SCHEMA,
            "state": "RESERVED",
            "experiment_config_sha256": experiment_config_sha256,
            "readiness_sha256": readiness_sha256,
            "plan_sha256": plan_sha256,
            "experiment_identity_sha256": experiment_identity_sha256,
            "data_identity_sha256": data_identity_sha256,
            "sealed_test_sha256": sealed_test_sha256,
            "test_inventory_sha256": test_inventory_sha256,
            "access_policy_sha256": access_policy_sha256,
            "tuning_selection_sha256": tuning_selection_sha256,
            "checkpoint_selection_sha256": checkpoint_selection_sha256,
            "validation_bundle_sha256": validation_bundle_sha256,
            "scoring_inventory_sha256": scoring_inventory_sha256,
            "expected_row_count": expected_row_count,
            **output_payload,
            "output_namespace_sha256": _hash(
                "vfe4.h6.test-output-namespace.v3",
                output_payload,
            ),
        }
        payload = {
            **opening_payload,
            "opening_proof_sha256": _hash(
                "vfe4.h6.test-opening-proof.v3",
                opening_payload,
            ),
        }
        return cls(
            **payload,  # type: ignore[arg-type]
            reservation_sha256=_hash(
                "vfe4.h6.test-transaction-reservation.v3",
                payload,
            ),
        )

    @classmethod
    def from_canonical_bytes(cls, raw: bytes) -> H6TestReservationV3:
        if type(raw) is not bytes or not raw:
            raise ValueError("reservation bytes must be nonempty bytes")

        def reject_duplicates(
            pairs: list[tuple[str, object]],
        ) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("reservation contains duplicate JSON keys")
                result[key] = value
            return result

        try:
            payload = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=reject_duplicates,
            )
        except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("reservation bytes are not canonical JSON") from exc
        if type(payload) is not dict or canonical_json_bytes(payload) != raw:
            raise ValueError("reservation bytes are not canonical JSON")
        values = dict(payload)
        for name in (
            "result_root_identity",
            "state_root_identity",
            "pointer_root_identity",
        ):
            value = values.get(name)
            if type(value) is list:
                values[name] = tuple(value)
        try:
            reservation = cls(**values)
        except (TypeError, ValueError) as exc:
            raise ValueError("reservation bytes are not an exact v3 record") from exc
        if reservation.canonical_bytes() != raw:
            raise ValueError("reservation bytes contain unknown fields")
        return reservation


def preflight_h6_test_output_namespace_v3(
    reservation: H6TestReservationV3,
) -> None:
    """Fail before reservation unless every bound no-replace name is absent."""

    if type(reservation) is not H6TestReservationV3:
        raise ValueError("exact H6 test reservation v3 is required")
    reservation.__post_init__()
    candidates = (
        Path(reservation.result_root_path) / "RESULT",
        Path(reservation.state_root_path) / "FINALIZED",
        Path(reservation.state_root_path) / "INCONCLUSIVE",
        Path(reservation.state_root_path) / "TERMINAL",
        Path(reservation.pointer_root_path) / reservation.pointer_name,
    )
    normalized_identities = tuple(
        os.path.normcase(os.path.abspath(os.path.normpath(os.fspath(candidate))))
        for candidate in candidates
    )
    if len(set(normalized_identities)) != len(normalized_identities):
        raise ValueError("H6 test output namespace targets must be pairwise distinct")
    collisions = tuple(path for path in candidates if os.path.lexists(path))
    if collisions:
        raise ValueError(
            "H6 test output namespace is not available: "
            + ", ".join(path.name for path in collisions)
        )


@dataclass(frozen=True, slots=True)
class H6TestTerminalV3:
    terminal_schema: Literal["h6-test-transaction-terminal-v3"]
    state: Literal["FINALIZED", "INCONCLUSIVE"]
    reservation_sha256: str
    result_sha256: str | None
    reason: str | None
    terminal_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "terminal_schema": self.terminal_schema,
            "state": self.state,
            "reservation_sha256": self.reservation_sha256,
            "result_sha256": self.result_sha256,
            "reason": self.reason,
        }

    def artifact_payload(self) -> dict[str, object]:
        return {**self.semantic_payload(), "terminal_sha256": self.terminal_sha256}

    def __post_init__(self) -> None:
        if self.terminal_schema != _TERMINAL_SCHEMA:
            raise ValueError("H6 terminal schema is not frozen")
        _require_sha256(self.reservation_sha256, "reservation_sha256")
        if self.state == "FINALIZED":
            _require_sha256(self.result_sha256, "result_sha256")
            if self.reason is not None:
                raise ValueError("FINALIZED terminal cannot carry a failure reason")
        elif self.state == "INCONCLUSIVE":
            if (
                self.result_sha256 is not None
                or type(self.reason) is not str
                or not self.reason
            ):
                raise ValueError(
                    "INCONCLUSIVE terminal requires only a nonempty reason"
                )
        else:
            raise ValueError("terminal state must be FINALIZED or INCONCLUSIVE")
        _require_sha256(self.terminal_sha256, "terminal_sha256")
        if self.terminal_sha256 != _hash(
            "vfe4.h6.test-transaction-terminal.v3",
            self.semantic_payload(),
        ):
            raise ValueError("terminal SHA-256 does not match its fields")

    @classmethod
    def create(
        cls,
        *,
        state: Literal["FINALIZED", "INCONCLUSIVE"],
        reservation_sha256: str,
        result_sha256: str | None,
        reason: str | None,
    ) -> H6TestTerminalV3:
        payload = {
            "terminal_schema": _TERMINAL_SCHEMA,
            "state": state,
            "reservation_sha256": reservation_sha256,
            "result_sha256": result_sha256,
            "reason": reason,
        }
        return cls(
            **payload,  # type: ignore[arg-type]
            terminal_sha256=_hash(
                "vfe4.h6.test-transaction-terminal.v3",
                payload,
            ),
        )


@dataclass(frozen=True, slots=True)
class H6PredictionPointerV3:
    pointer_schema: Literal["h6-prediction-pointer-v3"]
    reservation_sha256: str
    terminal_sha256: str
    result_sha256: str
    publication_atomicity: Literal[
        "individual-no-replace-directories-no-cross-directory-atomicity"
    ]
    pointer_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "pointer_schema": self.pointer_schema,
            "reservation_sha256": self.reservation_sha256,
            "terminal_sha256": self.terminal_sha256,
            "result_sha256": self.result_sha256,
            "publication_atomicity": self.publication_atomicity,
        }

    def artifact_payload(self) -> dict[str, object]:
        return {**self.semantic_payload(), "pointer_sha256": self.pointer_sha256}

    def __post_init__(self) -> None:
        if (
            self.pointer_schema != _POINTER_SCHEMA
            or self.publication_atomicity != _ATOMICITY_DISCLOSURE
        ):
            raise ValueError("H6 prediction pointer publication semantics changed")
        for name in (
            "reservation_sha256",
            "terminal_sha256",
            "result_sha256",
            "pointer_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if self.pointer_sha256 != _hash(
            "vfe4.h6.prediction-pointer.v3",
            self.semantic_payload(),
        ):
            raise ValueError("prediction pointer SHA-256 does not match its fields")

    @classmethod
    def create(
        cls,
        *,
        reservation_sha256: str,
        terminal_sha256: str,
        result_sha256: str,
        publication_atomicity: Literal[
            "individual-no-replace-directories-no-cross-directory-atomicity"
        ],
    ) -> H6PredictionPointerV3:
        payload = {
            "pointer_schema": _POINTER_SCHEMA,
            "reservation_sha256": reservation_sha256,
            "terminal_sha256": terminal_sha256,
            "result_sha256": result_sha256,
            "publication_atomicity": publication_atomicity,
        }
        return cls(
            **payload,  # type: ignore[arg-type]
            pointer_sha256=_hash(
                "vfe4.h6.prediction-pointer.v3",
                payload,
            ),
        )


@dataclass(frozen=True, slots=True)
class H6TestFinalizationPathsV3:
    terminal: H6TestTerminalV3
    result_directory: Path
    pointer_directory: Path
    state_alias_directory: Path
    terminal_directory: Path


def _is_redirect(path: Path, status: os.stat_result) -> bool:
    if stat.S_ISLNK(status.st_mode):
        return True
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(getattr(status, "st_file_attributes", 0) & reparse_flag)


def _read_bound_file(
    path: Path,
    *,
    maximum_bytes: int = 1024 * 1024,
) -> bytes:
    if type(maximum_bytes) is not int or maximum_bytes <= 0:
        raise ArtifactPublicationError("journal byte bound is invalid")
    try:
        before = path.lstat()
    except OSError as exc:
        raise ArtifactPublicationError(f"journal file is unavailable: {path}") from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or _is_redirect(path, before)
        or before.st_size > maximum_bytes
    ):
        raise ArtifactPublicationError("journal entry is not a bound regular file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags | nofollow)
    try:
        opened = os.fstat(descriptor)
        if (
            opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
            or not stat.S_ISREG(opened.st_mode)
            or opened.st_size > maximum_bytes
        ):
            raise ArtifactPublicationError("journal file identity changed before open")
        chunks: list[bytes] = []
        remaining = maximum_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        if sum(map(len, chunks)) > maximum_bytes:
            raise ArtifactPublicationError("journal file exceeds its byte bound")
        after = os.fstat(descriptor)
        if (
            after.st_dev != opened.st_dev
            or after.st_ino != opened.st_ino
            or after.st_size != opened.st_size
            or after.st_mtime_ns != opened.st_mtime_ns
        ):
            raise ArtifactPublicationError("journal file changed while reading")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _read_published_payload(directory: Path, filename: str) -> dict[str, object]:
    try:
        directory_status = directory.lstat()
    except OSError as exc:
        raise ArtifactPublicationError(
            "published journal directory is unavailable"
        ) from exc
    if not stat.S_ISDIR(directory_status.st_mode) or _is_redirect(
        directory, directory_status
    ):
        raise ArtifactPublicationError(
            "published journal path is not a bound directory"
        )
    try:
        names = {entry.name for entry in directory.iterdir()}
    except OSError as exc:
        raise ArtifactPublicationError(
            "published journal inventory is unavailable"
        ) from exc
    if names != {filename, "manifest.sha256"}:
        raise ArtifactPublicationError("published journal inventory is not exact")
    manifest = _read_bound_file(directory / "manifest.sha256")
    payload_bytes = _read_bound_file(directory / filename)
    expected_manifest = (
        f"{hashlib.sha256(payload_bytes).hexdigest()}  {filename}\n".encode("utf-8")
    )
    if manifest != expected_manifest:
        raise ArtifactPublicationError("published journal manifest is not exact")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload_bytes.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ArtifactPublicationError("published journal JSON is invalid") from exc
    if type(value) is not dict or canonical_json_bytes(value) != payload_bytes:
        raise ArtifactPublicationError("published journal JSON is not canonical")
    return value


def _reservation_from_directory(directory: Path) -> H6TestReservationV3:
    payload = _read_published_payload(directory, "reservation.json")
    try:
        reservation = H6TestReservationV3(**payload)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ArtifactPublicationError(
            "RESERVED journal cannot be authenticated"
        ) from exc
    if reservation.artifact_payload() != payload:
        raise ArtifactPublicationError("RESERVED journal contains unknown fields")
    return reservation


def _reservation_from_marker(path: Path) -> H6TestReservationV3:
    raw = _read_bound_file(path)
    try:
        return H6TestReservationV3.from_canonical_bytes(raw)
    except ValueError as exc:
        raise ArtifactPublicationError(
            "authoritative RESERVED marker cannot be authenticated"
        ) from exc


def _bound_output_roots(
    reservation: H6TestReservationV3,
) -> tuple[Path, Path, Path]:
    reservation.__post_init__()
    return (
        _require_output_root(
            reservation.result_root_path,
            reservation.result_root_identity,
            "result_root",
        ),
        _require_output_root(
            reservation.state_root_path,
            reservation.state_root_identity,
            "state_root",
        ),
        _require_output_root(
            reservation.pointer_root_path,
            reservation.pointer_root_identity,
            "pointer_root",
        ),
    )


def _terminal_from_directory(directory: Path) -> H6TestTerminalV3:
    payload = _read_published_payload(directory, "terminal.json")
    try:
        terminal = H6TestTerminalV3(**payload)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ArtifactPublicationError(
            "terminal journal cannot be authenticated"
        ) from exc
    if terminal.artifact_payload() != payload:
        raise ArtifactPublicationError("terminal journal contains unknown fields")
    return terminal


def _pointer_from_directory(directory: Path) -> H6PredictionPointerV3:
    payload = _read_published_payload(directory, "pointer.json")
    try:
        pointer = H6PredictionPointerV3(**payload)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ArtifactPublicationError(
            "prediction pointer cannot be authenticated"
        ) from exc
    if pointer.artifact_payload() != payload:
        raise ArtifactPublicationError("prediction pointer contains unknown fields")
    return pointer


def read_h6_test_reservation_v3(path: Path) -> H6TestReservationV3:
    """Authenticate one bounded authoritative RESERVED marker."""

    if not isinstance(path, Path) or not path.is_absolute():
        raise ArtifactPublicationError(
            "reservation path must be an absolute pathlib Path"
        )
    return _reservation_from_marker(path)


def read_h6_test_terminal_v3(directory: Path) -> H6TestTerminalV3:
    """Authenticate one exact immutable terminal journal directory."""

    if not isinstance(directory, Path) or not directory.is_absolute():
        raise ArtifactPublicationError(
            "terminal directory must be an absolute pathlib Path"
        )
    return _terminal_from_directory(directory)


def read_h6_prediction_pointer_v3(directory: Path) -> H6PredictionPointerV3:
    """Authenticate one exact immutable prediction pointer directory."""

    if not isinstance(directory, Path) or not directory.is_absolute():
        raise ArtifactPublicationError(
            "pointer directory must be an absolute pathlib Path"
        )
    return _pointer_from_directory(directory)


def _publish_or_validate_pointer(
    *,
    pointer_root: Path,
    pointer_name: str,
    pointer: H6PredictionPointerV3,
) -> Path:
    path = pointer_root / pointer_name
    if os.path.lexists(path):
        if _pointer_from_directory(path) != pointer:
            raise ArtifactPublicationError("existing prediction pointer differs")
        return path
    published = publish_run_directory(
        pointer_root,
        pointer_name,
        {"pointer.json": pointer.artifact_payload()},
    )
    if _pointer_from_directory(published) != pointer:
        raise ArtifactPublicationError("published prediction pointer changed")
    return published


def _publish_or_validate_terminal_alias(
    *,
    state_root: Path,
    name: Literal["FINALIZED", "INCONCLUSIVE", "TERMINAL"],
    terminal: H6TestTerminalV3,
) -> Path:
    path = state_root / name
    if os.path.lexists(path):
        if _terminal_from_directory(path) != terminal:
            raise ArtifactPublicationError(
                f"existing {name} transaction record differs"
            )
        return path
    published = publish_run_directory(
        state_root,
        name,
        {"terminal.json": terminal.artifact_payload()},
    )
    if _terminal_from_directory(published) != terminal:
        raise ArtifactPublicationError(f"published {name} record changed")
    return published


def finalize_h6_test_transaction_v3(
    *,
    reservation_path: Path,
    result: H6PredictionResultV3,
    inventory: H6RawEndpointInventoryV4,
    metrics: H6PredictionMetricsV3,
) -> H6TestFinalizationPathsV3:
    """Publish/reopen result, pointer, alias, then commit TERMINAL last."""

    reservation = _reservation_from_marker(reservation_path)
    result_root, state_root, pointer_root = _bound_output_roots(reservation)
    if (
        type(result) is not H6PredictionResultV3
        or result.reservation_sha256 != reservation.reservation_sha256
        or result.opening_proof_sha256 != reservation.opening_proof_sha256
    ):
        raise ValueError("prediction result does not bind authoritative RESERVED")
    result_directory = result_root / "RESULT"
    if os.path.lexists(result_directory):
        reopened = read_h6_prediction_result_v3(
            result_directory,
            expected_result_sha256=result.result_sha256,
        )
        if reopened != (result, inventory, metrics):
            raise ArtifactPublicationError("existing prediction result differs")
    else:
        result_directory = publish_h6_prediction_result_v3(
            result_root,
            "RESULT",
            result=result,
            inventory=inventory,
            metrics=metrics,
        )
    terminal = H6TestTerminalV3.create(
        state="FINALIZED",
        reservation_sha256=reservation.reservation_sha256,
        result_sha256=result.result_sha256,
        reason=None,
    )
    pointer = H6PredictionPointerV3.create(
        reservation_sha256=reservation.reservation_sha256,
        terminal_sha256=terminal.terminal_sha256,
        result_sha256=result.result_sha256,
        publication_atomicity=_ATOMICITY_DISCLOSURE,
    )
    pointer_directory = _publish_or_validate_pointer(
        pointer_root=pointer_root,
        pointer_name=reservation.pointer_name,
        pointer=pointer,
    )
    state_alias_directory = _publish_or_validate_terminal_alias(
        state_root=state_root,
        name="FINALIZED",
        terminal=terminal,
    )
    # This fixed no-replace record is the sole authoritative terminal commit.
    terminal_directory = _publish_or_validate_terminal_alias(
        state_root=state_root,
        name="TERMINAL",
        terminal=terminal,
    )
    return H6TestFinalizationPathsV3(
        terminal=terminal,
        result_directory=result_directory,
        pointer_directory=pointer_directory,
        state_alias_directory=state_alias_directory,
        terminal_directory=terminal_directory,
    )


def recover_h6_test_transaction_v3(
    reservation_path: Path,
    *,
    capability_issuer: Callable[[], object],
) -> H6TestTerminalV3:
    if not callable(capability_issuer):
        raise ValueError("capability_issuer must be callable")
    reservation = _reservation_from_marker(reservation_path)
    result_root, state_root, _pointer_root = _bound_output_roots(reservation)
    terminal_path = state_root / "TERMINAL"
    if os.path.lexists(terminal_path):
        terminal = _terminal_from_directory(terminal_path)
        if terminal.state == "FINALIZED":
            result, inventory, metrics = read_h6_prediction_result_v3(
                result_root / "RESULT",
                expected_result_sha256=terminal.result_sha256,  # type: ignore[arg-type]
            )
            repaired = finalize_h6_test_transaction_v3(
                reservation_path=reservation_path,
                result=result,
                inventory=inventory,
                metrics=metrics,
            )
            if repaired.terminal != terminal:
                raise ArtifactPublicationError("recovered terminal identity changed")
        else:
            try:
                _publish_or_validate_terminal_alias(
                    state_root=state_root,
                    name="INCONCLUSIVE",
                    terminal=terminal,
                )
            except ArtifactPublicationError:
                pass
        return terminal

    if os.path.lexists(result_root / "RESULT"):
        try:
            result, inventory, metrics = read_h6_prediction_result_v3(
                result_root / "RESULT",
            )
            return finalize_h6_test_transaction_v3(
                reservation_path=reservation_path,
                result=result,
                inventory=inventory,
                metrics=metrics,
            ).terminal
        except (ArtifactPublicationError, TypeError, ValueError):
            pass
    terminal = H6TestTerminalV3.create(
        state="INCONCLUSIVE",
        reservation_sha256=reservation.reservation_sha256,
        result_sha256=None,
        reason=("authoritative RESERVED has no complete conflict-free bound result"),
    )
    try:
        _publish_or_validate_terminal_alias(
            state_root=state_root,
            name="INCONCLUSIVE",
            terminal=terminal,
        )
    except ArtifactPublicationError:
        # The alias is explicitly nonauthoritative. A conflicting partial alias
        # cannot prevent the sole fixed terminal commit.
        pass
    _publish_or_validate_terminal_alias(
        state_root=state_root,
        name="TERMINAL",
        terminal=terminal,
    )
    return terminal


def _validate_eligibility(
    *,
    config: object,
    readiness: object,
    plan: object,
    validation_bundle: object,
    store: object,
    experiment_identity: object,
) -> tuple[
    H6PredictionV3ResolvedConfig,
    H6PredictionV3ReadinessToken,
    H6ExperimentPlanV3,
    H6ValidationBundleV3,
    BlindedCorpusStore,
    ExperimentIdentity,
]:
    if type(config) is not H6PredictionV3ResolvedConfig:
        raise ValueError("exact H6-Prediction v3 resolved config is required")
    if (
        config.schema_version != "h6-prediction-config-v3"
        or config.operation != "H6-Prediction"
        or config.expected_test_row_count != 4104
        or config.scoring_inventory_sha256 != H6_SCORING_INVENTORY_SHA256
        or type(config.canonical_json) is not str
        or hashlib.sha256(config.canonical_json.encode("utf-8")).hexdigest()
        != config.config_sha256
    ):
        raise ValueError("exact H6-Prediction v3 config is stale")
    if type(readiness) is not H6PredictionV3ReadinessToken:
        raise ValueError("exact H6-Prediction v3 readiness is required")
    if type(plan) is not H6ExperimentPlanV3:
        raise ValueError("exact H6-Prediction v3 plan is required")
    if type(validation_bundle) is not H6ValidationBundleV3:
        raise ValueError("exact H6-Prediction v3 validation bundle is required")
    if type(store) is not BlindedCorpusStore:
        raise ValueError("exact H6-Prediction v3 blinded store is required")
    if type(experiment_identity) is not ExperimentIdentity:
        raise ValueError("exact H6-Prediction v3 experiment identity is required")
    try:
        readiness.__post_init__()
        plan.__post_init__()
        validation_bundle.__post_init__()
        experiment_identity.__post_init__()
    except ValueError as exc:
        raise ValueError("H6-Prediction v3 eligibility authority is stale") from exc
    checkpoint_selection = validation_bundle.checkpoint_selection
    if (
        readiness.status != "PASS"
        or config.config_sha256 != readiness.experiment_config_sha256
        or plan.experiment_config_sha256 != config.config_sha256
        or validation_bundle.experiment_config_sha256 != config.config_sha256
        or plan.readiness_sha256 != readiness.readiness_sha256
        or checkpoint_selection.readiness_sha256 != readiness.readiness_sha256
        or validation_bundle.plan_sha256 != plan.plan_sha256
        or config.data_identity_sha256 != readiness.data_identity_sha256
        or checkpoint_selection.data_identity_sha256 != readiness.data_identity_sha256
        or config.access_policy_sha256 != readiness.access_policy_sha256
        or config.matching_set_sha256 != readiness.matching_set_sha256
        or config.runtime.runtime_identity_sha256 != readiness.runtime_identity_sha256
    ):
        raise ValueError("H6-Prediction v3 complete eligibility is not cross-bound")
    return config, readiness, plan, validation_bundle, store, experiment_identity


def _validate_inventory_checkpoints(
    inventory: H6RawEndpointInventoryV4,
    bundle: H6ValidationBundleV3,
) -> None:
    inventory.__post_init__()
    selected = {
        (candidate.endpoint_config_id, candidate.training_seed): (
            candidate.checkpoint_sha256
        )
        for candidate in bundle.checkpoint_selection.checkpoints
    }
    for row in (
        *inventory.exact_a0_rows,
        *inventory.complete_a5_rows,
        *inventory.emission_a5_rows,
    ):
        if (
            selected.get((row.endpoint_config_id, row.training_seed))
            != row.checkpoint_sha256
        ):
            raise ValueError("held-out row does not use the selected checkpoint")


@dataclass(frozen=True, slots=True)
class H6TestTransactionOutcomeV3:
    reservation: H6TestReservationV3
    terminal: H6TestTerminalV3
    inventory: H6RawEndpointInventoryV4
    metrics: H6PredictionMetricsV3
    reserved_directory: Path
    result_directory: Path
    terminal_directory: Path
    pointer_directory: Path


def execute_h6_test_transaction_v3(
    *,
    config: object,
    readiness: object,
    plan: object,
    validation_bundle: object,
    store: object,
    journal_root: Path,
    score_inventory: Callable[[object, str], H6RawEndpointInventoryV4],
    experiment_identity: object = None,
    journal_name: str | None = None,
    pointer_root: Path | None = None,
    pointer_name: str = "current",
) -> H6TestTransactionOutcomeV3:
    (
        exact_config,
        exact_readiness,
        exact_plan,
        exact_bundle,
        exact_store,
        exact_experiment,
    ) = _validate_eligibility(
        config=config,
        readiness=readiness,
        plan=plan,
        validation_bundle=validation_bundle,
        store=store,
        experiment_identity=experiment_identity,
    )
    if not callable(score_inventory):
        raise ValueError("score_inventory must be callable")
    from vfe4.data.access import (
        h6_test_reservation_destination_v3,
        open_test_for_scoring_with_receipt,
        registered_h6_test_reservation_path_v3,
        reserve_and_issue_durable_test_opening_capability_v3,
        validated_test_opening_identity,
    )

    transaction_run_name = journal_name or f"h6-test-{exact_config.config_sha256[:16]}"
    if (
        type(transaction_run_name) is not str
        or not transaction_run_name
        or transaction_run_name in (".", "..")
        or "/" in transaction_run_name
        or "\\" in transaction_run_name
    ):
        raise ValueError("journal_name must be one portable directory component")
    result_root = journal_root / transaction_run_name
    exact_pointer_root = (
        pointer_root or exact_config.artifact_root / "h6-prediction-pointers"
    )
    reservation_path = h6_test_reservation_destination_v3(exact_store)
    reservation = H6TestReservationV3.create(
        experiment_config_sha256=exact_config.config_sha256,
        readiness_sha256=exact_readiness.readiness_sha256,
        plan_sha256=exact_plan.plan_sha256,
        experiment_identity_sha256=(exact_experiment.experiment_identity_sha256),
        data_identity_sha256=exact_readiness.data_identity_sha256,
        sealed_test_sha256=exact_store.sealed_test_handle.sealed_content_sha256,
        test_inventory_sha256=exact_store.sealed_test_handle.handle_sha256,
        access_policy_sha256=exact_readiness.access_policy_sha256,
        tuning_selection_sha256=(exact_bundle.tuning_selection.tuning_selection_sha256),
        checkpoint_selection_sha256=(
            exact_bundle.checkpoint_selection.checkpoint_selection_sha256
        ),
        validation_bundle_sha256=exact_bundle.validation_bundle_sha256,
        scoring_inventory_sha256=H6_SCORING_INVENTORY_SHA256,
        expected_row_count=4104,
        result_root=result_root,
        state_root=result_root / "STATE",
        pointer_root=exact_pointer_root,
        pointer_name=pointer_name,
    )
    preflight_h6_test_output_namespace_v3(reservation)
    try:
        opening = reserve_and_issue_durable_test_opening_capability_v3(
            store=exact_store,
            readiness=exact_readiness,
            plan=exact_plan,
            experiment_identity=exact_experiment,
            reservation=reservation,
        )
        registered_path = registered_h6_test_reservation_path_v3(
            exact_store,
            opening,
        )
        if (
            registered_path != reservation_path
            or _reservation_from_marker(registered_path) != reservation
            or opening.proof_identity_sha256 != reservation.opening_proof_sha256
        ):
            raise ValueError("access issuer changed authoritative RESERVED")
        test_windows, receipt = open_test_for_scoring_with_receipt(
            exact_store,
            opening,
        )
        if validated_test_opening_identity(receipt) != reservation.opening_proof_sha256:
            raise ValueError("consumed opening receipt does not bind RESERVED")
        inventory = score_inventory(
            test_windows,
            reservation.opening_proof_sha256,
        )
        if type(inventory) is not H6RawEndpointInventoryV4:
            raise ValueError("scorer did not return an exact raw inventory v4")
        if inventory.opening_proof_sha256 != reservation.opening_proof_sha256:
            raise ValueError("raw inventory does not bind the reserved opening")
        _validate_inventory_checkpoints(inventory, exact_bundle)
        metrics = H6PredictionMetricsV3.from_raw_inventory(inventory)
        result = H6PredictionResultV3.create(
            reservation_sha256=reservation.reservation_sha256,
            opening_proof_sha256=reservation.opening_proof_sha256,
            inventory=inventory,
            metrics=metrics,
        )
        finalized = finalize_h6_test_transaction_v3(
            reservation_path=reservation_path,
            result=result,
            inventory=inventory,
            metrics=metrics,
        )
        return H6TestTransactionOutcomeV3(
            reservation=reservation,
            terminal=finalized.terminal,
            inventory=inventory,
            metrics=metrics,
            reserved_directory=reservation_path,
            result_directory=finalized.result_directory,
            terminal_directory=finalized.terminal_directory,
            pointer_directory=finalized.pointer_directory,
        )
    except BaseException:
        if os.path.lexists(reservation_path):
            recover_h6_test_transaction_v3(
                reservation_path,
                capability_issuer=lambda: None,
            )
        raise


__all__ = [
    "H6PredictionPointerV3",
    "H6TestReservationV3",
    "H6TestTerminalV3",
    "H6TestTransactionOutcomeV3",
    "execute_h6_test_transaction_v3",
    "finalize_h6_test_transaction_v3",
    "preflight_h6_test_output_namespace_v3",
    "read_h6_prediction_pointer_v3",
    "read_h6_test_reservation_v3",
    "read_h6_test_terminal_v3",
    "recover_h6_test_transaction_v3",
]
