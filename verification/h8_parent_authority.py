"""Dependency-neutral, process-local authority for one H8 parent attempt run."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from verification.h8_wire import H8_CHILD_REQUEST_KEYS
from vfe4.types.h8 import H8ChildAttemptRecord


_HEX = frozenset("0123456789abcdef")


def _sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _authority_preimage(
    *,
    child_config_sha256: str,
    protocol_sha256: str,
    authorization_sha256: str,
    current_registry_sha256: str,
    prerequisite_validation_sha256: str,
    correctness_statuses_sha256: str,
    request_plan_sha256: str,
    issued_prefix_sha256: str,
) -> bytes:
    return _canonical_json_bytes(
        {
            "domain": "vfe4.h8.parent-attempt-authority.v1",
            "child_config_sha256": child_config_sha256,
            "protocol_sha256": protocol_sha256,
            "authorization_sha256": authorization_sha256,
            "current_registry_sha256": current_registry_sha256,
            "prerequisite_validation_sha256": (
                prerequisite_validation_sha256
            ),
            "correctness_statuses_sha256": correctness_statuses_sha256,
            "request_plan_sha256": request_plan_sha256,
            "issued_prefix_sha256": issued_prefix_sha256,
        }
    )


def h8_correctness_statuses_sha256(
    statuses: tuple[tuple[int, str], ...],
) -> str:
    """Bind exact cell IDs and status strings without importing gate types."""

    if type(statuses) is not tuple or any(
        type(item) is not tuple
        or len(item) != 2
        or type(item[0]) is not int
        or type(item[1]) is not str
        or not item[1]
        for item in statuses
    ):
        raise ValueError(
            "correctness statuses must contain exact integer IDs and strings"
        )
    return hashlib.sha256(
        _canonical_json_bytes(
            {
                "domain": "vfe4.h8.correctness-statuses.v1",
                "statuses": statuses,
            }
        )
    ).hexdigest()


def _source_authorization_commitments(
    source_authorization: object,
) -> dict[str, str]:
    statuses = getattr(source_authorization, "correctness_statuses", None)
    if type(statuses) is not tuple:
        raise ValueError("source authorization correctness statuses are absent")
    status_items = tuple(
        (
            item[0],
            getattr(item[1], "value", None),
        )
        if type(item) is tuple and len(item) == 2
        else (-1, None)
        for item in statuses
    )
    correctness_statuses_sha256 = h8_correctness_statuses_sha256(
        status_items  # type: ignore[arg-type]
    )
    obligations = getattr(source_authorization, "obligations", None)
    valid_start = getattr(source_authorization, "valid_start", None)
    if (
        type(obligations) is not tuple
        or any(type(item) is not str or not item for item in obligations)
        or type(valid_start) is not bool
    ):
        raise ValueError("source authorization decision context is malformed")
    prerequisite_validation = getattr(
        source_authorization,
        "prerequisite_validation",
        None,
    )
    prerequisite_validation_sha256 = _sha256(
        getattr(
            source_authorization,
            "prerequisite_validation_sha256",
            None,
        ),
        "source authorization prerequisite_validation_sha256",
    )
    if (
        prerequisite_validation is None
        or getattr(prerequisite_validation, "validation_sha256", None)
        != prerequisite_validation_sha256
    ):
        raise ValueError(
            "source authorization prerequisite validation is stale"
        )
    commitments = {
        "child_config_sha256": _sha256(
            getattr(source_authorization, "config_sha256", None),
            "source authorization config_sha256",
        ),
        "protocol_sha256": _sha256(
            getattr(source_authorization, "protocol_sha256", None),
            "source authorization protocol_sha256",
        ),
        "current_registry_sha256": _sha256(
            getattr(source_authorization, "current_registry_sha256", None),
            "source authorization current_registry_sha256",
        ),
        "prerequisite_validation_sha256": (
            prerequisite_validation_sha256
        ),
        "correctness_statuses_sha256": correctness_statuses_sha256,
    }
    authorization_preimage = _canonical_json_bytes(
        {
            "domain": "vfe4.h8.child-start-authorization.v1",
            "config_sha256": commitments["child_config_sha256"],
            "protocol_sha256": commitments["protocol_sha256"],
            "current_registry_sha256": (
                commitments["current_registry_sha256"]
            ),
            "prerequisite_validation_sha256": (
                commitments["prerequisite_validation_sha256"]
            ),
            "correctness_statuses": status_items,
            "obligations": obligations,
            "valid_start": valid_start,
        }
    )
    authorization_sha256 = hashlib.sha256(
        authorization_preimage
    ).hexdigest()
    if (
        getattr(source_authorization, "authorization_sha256", None)
        != authorization_sha256
    ):
        raise ValueError("source authorization SHA-256 is stale")
    return {
        **commitments,
        "authorization_sha256": authorization_sha256,
    }


def _request_plan_sha256(request_plan: object) -> str:
    if type(request_plan) is not tuple:
        raise ValueError("source request plan must be an exact tuple")
    requests = []
    for request in request_plan:
        validator = getattr(request, "__post_init__", None)
        if not callable(validator):
            raise ValueError("source request plan contains an invalid request")
        validator()
        requests.append(
            {
                name: getattr(request, name)
                for name in H8_CHILD_REQUEST_KEYS
            }
        )
    return hashlib.sha256(
        _canonical_json_bytes(
            {
                "domain": "vfe4.h8.parent-request-plan.v1",
                "requests": tuple(requests),
            }
        )
    ).hexdigest()


def _issued_prefix_sha256(issued_prefix: object) -> str:
    if type(issued_prefix) is not tuple:
        raise ValueError("source issued prefix must be an exact tuple")
    issued = []
    for item in issued_prefix:
        validator = getattr(item, "__post_init__", None)
        if not callable(validator):
            raise ValueError("source issued prefix contains an invalid record")
        validator()
        issued.append(
            {
                "launch_contract_sha256": item.launch_contract_sha256,
                "timed_out": item.process_record.timed_out,
                "exit_code": item.process_record.exit_code,
                "parent_elapsed_ns": item.process_record.parent_elapsed_ns,
                "status": item.attempt.status.value,
                "reasons": item.attempt.reasons,
                "request_sha256": item.attempt.request_sha256,
                "identities_sha256": item.attempt.identities_sha256,
                "stdout_sha256": item.attempt.stdout_sha256,
                "stderr_sha256": item.attempt.stderr_sha256,
            }
        )
    return hashlib.sha256(
        _canonical_json_bytes(
            {
                "domain": "vfe4.h8.parent-issued-prefix.v1",
                "issued": tuple(issued),
            }
        )
    ).hexdigest()


@dataclass(frozen=True, slots=True, init=False)
class H8ParentAttemptAuthority:
    """Factory-issued authority for one exact validated parent-run object."""

    child_config_sha256: str
    protocol_sha256: str
    authorization_sha256: str
    current_registry_sha256: str
    prerequisite_validation_sha256: str
    correctness_statuses_sha256: str
    request_plan_sha256: str
    issued_prefix_sha256: str
    attempts: tuple[H8ChildAttemptRecord, ...]
    authority_sha256: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError(
            "H8ParentAttemptAuthority is factory-only; "
            "run the H8 parent orchestrator"
        )


@dataclass(frozen=True, slots=True)
class _IssuedAuthority:
    authority: H8ParentAttemptAuthority
    source_run: object
    source_authorization: object
    request_plan: object
    issued_prefix: object
    attempts: tuple[H8ChildAttemptRecord, ...]
    preimage: bytes


_ISSUED_H8_PARENT_AUTHORITIES: dict[int, _IssuedAuthority] = {}


def _issue_h8_parent_attempt_authority(
    *,
    source_run: object,
    source_authorization: object,
    request_plan: object,
    issued_prefix: object,
    attempts: tuple[H8ChildAttemptRecord, ...],
) -> H8ParentAttemptAuthority:
    """Issue authority only for identities retained by one validated source run."""

    if source_run is None:
        raise ValueError("source run must be present")
    if (
        getattr(source_run, "authorization", None) is not source_authorization
        or getattr(source_run, "request_plan", None) is not request_plan
        or getattr(source_run, "issued", None) is not issued_prefix
        or getattr(source_run, "attempts", None) is not attempts
    ):
        raise ValueError("authority inputs must retain exact source-run identities")
    if type(attempts) is not tuple or any(
        type(item) is not H8ChildAttemptRecord for item in attempts
    ):
        raise ValueError("authority attempts must be exact typed records")
    source_commitments = _source_authorization_commitments(
        source_authorization
    )
    checked = {
        **source_commitments,
        "request_plan_sha256": _request_plan_sha256(request_plan),
        "issued_prefix_sha256": _issued_prefix_sha256(issued_prefix),
    }
    preimage = _authority_preimage(**checked)
    authority = object.__new__(H8ParentAttemptAuthority)
    for name, value in (
        *checked.items(),
        ("attempts", attempts),
        ("authority_sha256", hashlib.sha256(preimage).hexdigest()),
    ):
        object.__setattr__(authority, name, value)
    _ISSUED_H8_PARENT_AUTHORITIES[id(authority)] = _IssuedAuthority(
        authority=authority,
        source_run=source_run,
        source_authorization=source_authorization,
        request_plan=request_plan,
        issued_prefix=issued_prefix,
        attempts=attempts,
        preimage=preimage,
    )
    return require_h8_parent_attempt_authority(
        authority,
        source_run=source_run,
    )


def require_h8_parent_attempt_authority(
    value: object,
    *,
    source_run: object | None = None,
) -> H8ParentAttemptAuthority:
    """Require the exact registered authority, optionally for one exact source run."""

    if type(value) is not H8ParentAttemptAuthority:
        raise ValueError(
            "parent authority must be an exact H8ParentAttemptAuthority"
        )
    issued = _ISSUED_H8_PARENT_AUTHORITIES.get(id(value))
    if issued is None or issued.authority is not value:
        raise ValueError("parent authority must be factory-issued")
    if source_run is not None and issued.source_run is not source_run:
        raise ValueError("parent authority belongs to a different source run")
    if (
        getattr(issued.source_run, "authorization", None)
        is not issued.source_authorization
    ):
        raise ValueError("parent authority source authorization identity is stale")
    if (
        getattr(issued.source_run, "request_plan", None)
        is not issued.request_plan
        or getattr(issued.source_run, "issued", None)
        is not issued.issued_prefix
        or getattr(issued.source_run, "attempts", None) is not issued.attempts
        or value.attempts is not issued.attempts
    ):
        raise ValueError("parent authority source-run identity is stale")
    source_validator = getattr(issued.source_run, "__post_init__", None)
    if not callable(source_validator):
        raise ValueError("parent authority source run cannot be revalidated")
    source_validator()
    source_commitments = _source_authorization_commitments(
        issued.source_authorization
    )
    for attempt in value.attempts:
        attempt.__post_init__()
    recomputed = {
        **source_commitments,
        "request_plan_sha256": _request_plan_sha256(
            issued.request_plan
        ),
        "issued_prefix_sha256": _issued_prefix_sha256(
            issued.issued_prefix
        ),
    }
    if any(
        getattr(value, name) != expected
        for name, expected in recomputed.items()
    ):
        raise ValueError("parent authority source-derived digest is stale")
    checked = {
        name: _sha256(getattr(value, name), name)
        for name in (
            "child_config_sha256",
            "protocol_sha256",
            "authorization_sha256",
            "current_registry_sha256",
            "prerequisite_validation_sha256",
            "correctness_statuses_sha256",
            "request_plan_sha256",
            "issued_prefix_sha256",
        )
    }
    preimage = _authority_preimage(**checked)
    if preimage != issued.preimage:
        raise ValueError("parent authority canonical preimage is stale")
    expected_sha256 = hashlib.sha256(preimage).hexdigest()
    if value.authority_sha256 != expected_sha256:
        raise ValueError("parent authority SHA-256 is stale")
    return value


__all__ = [
    "H8ParentAttemptAuthority",
    "h8_correctness_statuses_sha256",
    "require_h8_parent_attempt_authority",
]
