"""Dependency-neutral, process-local authority for one H8 parent attempt run."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

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
    request_plan_sha256: str,
    issued_prefix_sha256: str,
) -> bytes:
    return _canonical_json_bytes(
        {
            "domain": "vfe4.h8.parent-attempt-authority.v1",
            "child_config_sha256": child_config_sha256,
            "protocol_sha256": protocol_sha256,
            "authorization_sha256": authorization_sha256,
            "request_plan_sha256": request_plan_sha256,
            "issued_prefix_sha256": issued_prefix_sha256,
        }
    )


@dataclass(frozen=True, slots=True, init=False)
class H8ParentAttemptAuthority:
    """Factory-issued authority for one exact validated parent-run object."""

    child_config_sha256: str
    protocol_sha256: str
    authorization_sha256: str
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
    request_plan: object
    issued_prefix: object
    attempts: tuple[H8ChildAttemptRecord, ...]
    preimage: bytes


_ISSUED_H8_PARENT_AUTHORITIES: dict[int, _IssuedAuthority] = {}


def _issue_h8_parent_attempt_authority(
    *,
    source_run: object,
    request_plan: object,
    issued_prefix: object,
    attempts: tuple[H8ChildAttemptRecord, ...],
    child_config_sha256: str,
    protocol_sha256: str,
    authorization_sha256: str,
    request_plan_sha256: str,
    issued_prefix_sha256: str,
) -> H8ParentAttemptAuthority:
    """Issue authority only for identities retained by one validated source run."""

    if source_run is None:
        raise ValueError("source run must be present")
    if (
        getattr(source_run, "request_plan", None) is not request_plan
        or getattr(source_run, "issued", None) is not issued_prefix
        or getattr(source_run, "attempts", None) is not attempts
    ):
        raise ValueError("authority inputs must retain exact source-run identities")
    if type(attempts) is not tuple or any(
        type(item) is not H8ChildAttemptRecord for item in attempts
    ):
        raise ValueError("authority attempts must be exact typed records")
    checked = {
        "child_config_sha256": _sha256(
            child_config_sha256,
            "child_config_sha256",
        ),
        "protocol_sha256": _sha256(protocol_sha256, "protocol_sha256"),
        "authorization_sha256": _sha256(
            authorization_sha256,
            "authorization_sha256",
        ),
        "request_plan_sha256": _sha256(
            request_plan_sha256,
            "request_plan_sha256",
        ),
        "issued_prefix_sha256": _sha256(
            issued_prefix_sha256,
            "issued_prefix_sha256",
        ),
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
    for attempt in value.attempts:
        attempt.__post_init__()
    checked = {
        name: _sha256(getattr(value, name), name)
        for name in (
            "child_config_sha256",
            "protocol_sha256",
            "authorization_sha256",
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
    "require_h8_parent_attempt_authority",
]
