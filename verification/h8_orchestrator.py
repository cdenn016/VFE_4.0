"""Parent-owned planning and issued-prefix execution for the frozen H8 protocol."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from verification.h8_budget import (
    H8ChildInvocation,
    H8ChildProcessRecord,
    build_h8_child_invocation,
    canonical_json_bytes,
    classify_h8_child_outcome,
    make_h8_child_attempt_record,
    run_h8_child,
    _require_h8_issued_attempt_binding,
)
from verification.h8_gate import H8PrerequisiteArtifactValidation
from verification.h8_parent_authority import (
    H8ParentAttemptAuthority,
    _issue_h8_parent_attempt_authority,
)
from verification.h8_parent_identities import (
    collect_h8_runtime_identities as _collect_h8_runtime_identities,
)
from verification.h8_protocol import (
    _h8_protocol_preimage,  # noqa: F401
    build_h8_protocol_sha256,
)
from verification.h8_wire import (
    H8_CHILD_IDENTITY_ENV,
    H8_CHILD_REQUEST_KEYS,
    H8_NEGATIVE_CONTROL_IDS,
    H8_PRODUCTION_SEEDS,
)
from vfe4.config.schema import H8ValidationConfig
from vfe4.types.h8 import (
    H8ChildAttemptRecord,
    H8ChildRequest,
)
from vfe4.types.results import GateStatus


_HEX = frozenset("0123456789abcdef")
_H8_REVALIDATED_REFERENCE_NAMES = (
    "h1_h5",
    "h1_prefix_prior",
    "h6_prefix",
    "h7",
    "h6_prediction",
)


def _sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _canonical_request_bytes(request: H8ChildRequest) -> bytes:
    return canonical_json_bytes(
        {
            "mode": request.mode,
            "seed": request.seed,
            "repetition": request.repetition,
            "config_sha256": request.config_sha256,
            "protocol_sha256": request.protocol_sha256,
            "control_id": request.control_id,
        }
    )




@dataclass(frozen=True, slots=True, init=False)
class H8ChildStartAuthorization:
    """Factory-derived, immutable authority for the exact H8 child prefix."""

    config_sha256: str
    protocol_sha256: str
    current_registry_sha256: str
    prerequisite_validation: H8PrerequisiteArtifactValidation
    prerequisite_validation_sha256: str
    correctness_statuses: tuple[tuple[int, GateStatus], ...]
    obligations: tuple[str, ...]
    valid_start: bool
    authorization_sha256: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError(
            "H8ChildStartAuthorization is factory-only; "
            "use derive_h8_child_start_authorization"
        )


_ISSUED_H8_AUTHORIZATIONS: dict[
    int,
    tuple[H8ChildStartAuthorization, bytes],
] = {}


def _validate_correctness_statuses(
    correctness_statuses: object,
) -> tuple[tuple[int, GateStatus], ...]:
    if type(correctness_statuses) is not tuple or any(
        type(item) is not tuple
        or len(item) != 2
        or type(item[0]) is not int
        or type(item[1]) is not GateStatus
        for item in correctness_statuses
    ):
        raise ValueError(
            "correctness_statuses must contain exact cell IDs and GateStatus values"
        )
    return correctness_statuses


def _authorization_obligations(
    *,
    current_registry_sha256: str,
    prerequisite_validation: H8PrerequisiteArtifactValidation,
    correctness_statuses: tuple[tuple[int, GateStatus], ...],
) -> tuple[str, ...]:
    obligations = list(prerequisite_validation.obligations)
    if prerequisite_validation.registry_sha256 != current_registry_sha256:
        obligations.append("h8_prerequisite_validation_registry_mismatch")
    if (
        prerequisite_validation.revalidated_reference_names
        != _H8_REVALIDATED_REFERENCE_NAMES
    ):
        obligations.append("h8_prerequisite_revalidation_inventory_incomplete")
    if tuple(cell_id for cell_id, _status in correctness_statuses) != tuple(
        range(1, 13)
    ):
        obligations.append("h8_correctness_cell_inventory_incomplete")
    if not all(
        status is GateStatus.PASS for _cell_id, status in correctness_statuses
    ):
        obligations.append("h8_correctness_cells_not_all_pass")
    return tuple(dict.fromkeys(obligations))


def _authorization_preimage(
    *,
    config_sha256: str,
    protocol_sha256: str,
    current_registry_sha256: str,
    prerequisite_validation_sha256: str,
    correctness_statuses: tuple[tuple[int, GateStatus], ...],
    obligations: tuple[str, ...],
    valid_start: bool,
) -> dict[str, object]:
    return {
        "domain": "vfe4.h8.child-start-authorization.v1",
        "config_sha256": config_sha256,
        "protocol_sha256": protocol_sha256,
        "current_registry_sha256": current_registry_sha256,
        "prerequisite_validation_sha256": prerequisite_validation_sha256,
        "correctness_statuses": tuple(
            (cell_id, status.value)
            for cell_id, status in correctness_statuses
        ),
        "obligations": obligations,
        "valid_start": valid_start,
    }


def _validate_h8_child_start_authorization(
    authorization: object,
) -> H8ChildStartAuthorization:
    if type(authorization) is not H8ChildStartAuthorization:
        raise ValueError(
            "authorization must be an exact H8ChildStartAuthorization"
        )
    issued = _ISSUED_H8_AUTHORIZATIONS.get(id(authorization))
    if issued is None or issued[0] is not authorization:
        raise ValueError("authorization must be factory-issued")
    config_sha256 = _sha256(
        authorization.config_sha256,
        "authorization.config_sha256",
    )
    protocol_sha256 = _sha256(
        authorization.protocol_sha256,
        "authorization.protocol_sha256",
    )
    registry_sha256 = _sha256(
        authorization.current_registry_sha256,
        "authorization.current_registry_sha256",
    )
    if (
        type(authorization.prerequisite_validation)
        is not H8PrerequisiteArtifactValidation
    ):
        raise ValueError(
            "authorization prerequisite validation is not exact"
        )
    authorization.prerequisite_validation.__post_init__()
    prerequisite_validation_sha256 = _sha256(
        authorization.prerequisite_validation_sha256,
        "authorization.prerequisite_validation_sha256",
    )
    if (
        prerequisite_validation_sha256
        != authorization.prerequisite_validation.validation_sha256
    ):
        raise ValueError(
            "authorization prerequisite-validation SHA-256 is stale"
        )
    correctness_statuses = _validate_correctness_statuses(
        authorization.correctness_statuses
    )
    expected_obligations = _authorization_obligations(
        current_registry_sha256=registry_sha256,
        prerequisite_validation=authorization.prerequisite_validation,
        correctness_statuses=correctness_statuses,
    )
    if (
        type(authorization.obligations) is not tuple
        or any(
            type(item) is not str or not item
            for item in authorization.obligations
        )
        or len(set(authorization.obligations))
        != len(authorization.obligations)
        or authorization.obligations != expected_obligations
    ):
        raise ValueError("authorization obligations are stale")
    if (
        type(authorization.valid_start) is not bool
        or authorization.valid_start is not (not expected_obligations)
    ):
        raise ValueError("authorization start decision is stale")
    preimage_bytes = canonical_json_bytes(
        _authorization_preimage(
            config_sha256=config_sha256,
            protocol_sha256=protocol_sha256,
            current_registry_sha256=registry_sha256,
            prerequisite_validation_sha256=(
                prerequisite_validation_sha256
            ),
            correctness_statuses=correctness_statuses,
            obligations=expected_obligations,
            valid_start=authorization.valid_start,
        )
    )
    if preimage_bytes != issued[1]:
        raise ValueError("authorization owned preimage is stale")
    expected_sha256 = hashlib.sha256(preimage_bytes).hexdigest()
    if authorization.authorization_sha256 != expected_sha256:
        raise ValueError("authorization SHA-256 is stale")
    return authorization


def derive_h8_child_start_authorization(
    *,
    config: H8ValidationConfig,
    current_registry_sha256: str,
    prerequisite_validation: H8PrerequisiteArtifactValidation,
    correctness_statuses: tuple[tuple[int, GateStatus], ...],
) -> H8ChildStartAuthorization:
    """Derive child-start authority without accepting caller-owned validity."""

    if type(config) is not H8ValidationConfig:
        raise ValueError("config must be an exact H8ValidationConfig")
    config_sha256 = _sha256(config.config_sha256, "config.config_sha256")
    protocol_sha256 = build_h8_protocol_sha256(config)
    registry_sha256 = _sha256(
        current_registry_sha256,
        "current_registry_sha256",
    )
    if type(prerequisite_validation) is not H8PrerequisiteArtifactValidation:
        raise ValueError(
            "prerequisite_validation must be an exact "
            "H8PrerequisiteArtifactValidation"
        )
    prerequisite_validation.__post_init__()
    checked_statuses = _validate_correctness_statuses(correctness_statuses)
    obligations = _authorization_obligations(
        current_registry_sha256=registry_sha256,
        prerequisite_validation=prerequisite_validation,
        correctness_statuses=checked_statuses,
    )
    valid_start = not obligations
    preimage_bytes = canonical_json_bytes(
        _authorization_preimage(
            config_sha256=config_sha256,
            protocol_sha256=protocol_sha256,
            current_registry_sha256=registry_sha256,
            prerequisite_validation_sha256=(
                prerequisite_validation.validation_sha256
            ),
            correctness_statuses=checked_statuses,
            obligations=obligations,
            valid_start=valid_start,
        )
    )
    authorization = object.__new__(H8ChildStartAuthorization)
    for name, value in (
        ("config_sha256", config_sha256),
        ("protocol_sha256", protocol_sha256),
        ("current_registry_sha256", registry_sha256),
        ("prerequisite_validation", prerequisite_validation),
        (
            "prerequisite_validation_sha256",
            prerequisite_validation.validation_sha256,
        ),
        ("correctness_statuses", checked_statuses),
        ("obligations", obligations),
        ("valid_start", valid_start),
        (
            "authorization_sha256",
            hashlib.sha256(preimage_bytes).hexdigest(),
        ),
    ):
        object.__setattr__(authorization, name, value)
    _ISSUED_H8_AUTHORIZATIONS[id(authorization)] = (
        authorization,
        preimage_bytes,
    )
    return _validate_h8_child_start_authorization(authorization)


def build_h8_child_request_plan(
    *,
    config_sha256: str,
    protocol_sha256: str,
) -> tuple[H8ChildRequest, ...]:
    """Return the immutable, ordered 30-request H8 child plan."""

    return (
        *(
            H8ChildRequest(
                mode="production",
                seed=seed,
                repetition=repetition,
                config_sha256=config_sha256,
                protocol_sha256=protocol_sha256,
                control_id=None,
            )
            for seed in H8_PRODUCTION_SEEDS
            for repetition in range(5)
        ),
        *(
            H8ChildRequest(
                mode="profiler",
                seed=seed,
                repetition=None,
                config_sha256=config_sha256,
                protocol_sha256=protocol_sha256,
                control_id=None,
            )
            for seed in H8_PRODUCTION_SEEDS
        ),
        *(
            H8ChildRequest(
                mode="negative_control",
                seed=H8_PRODUCTION_SEEDS[0],
                repetition=None,
                config_sha256=config_sha256,
                protocol_sha256=protocol_sha256,
                control_id=control_id,
            )
            for control_id in H8_NEGATIVE_CONTROL_IDS
        ),
    )


def _resolved_repository_root(repository_root: str | Path) -> Path:
    root = Path(repository_root).resolve()
    if not root.is_absolute() or not root.is_dir():
        raise ValueError(
            "repository_root must resolve to an existing absolute directory"
        )
    return root


def _parent_identity_json_bytes(invocation: H8ChildInvocation) -> bytes:
    identity_json = invocation.environment.get(H8_CHILD_IDENTITY_ENV)
    if type(identity_json) is not str:
        raise ValueError("parent identity JSON is absent from the launch")
    try:
        identity_bytes = identity_json.encode("ascii")
        if canonical_json_bytes(json.loads(identity_bytes)) != identity_bytes:
            raise ValueError
    except (UnicodeEncodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("parent identity JSON bytes are not canonical") from error
    return identity_bytes


def build_h8_launch_contract_sha256(
    invocation: H8ChildInvocation,
    *,
    repository_root: str | Path,
) -> str:
    """Bind an exact launch, including raw stdin and parent identity bytes."""

    if type(invocation) is not H8ChildInvocation:
        raise ValueError("invocation must be an exact H8ChildInvocation")
    invocation.__post_init__()
    root = _resolved_repository_root(repository_root)
    if invocation.cwd.resolve() != root:
        raise ValueError("repository_root must equal the invocation cwd")
    identity_bytes = _parent_identity_json_bytes(invocation)
    preimage = {
        "domain": "vfe4.h8.parent-launch-contract.v1",
        "repository_root": str(root),
        "argv": invocation.argv,
        "stdin_hex": invocation.stdin.hex(),
        "environment": dict(invocation.environment),
        "parent_identity_json_hex": identity_bytes.hex(),
        "timeout_seconds": invocation.timeout_seconds,
        "capture_stdout": invocation.capture_stdout,
        "capture_stderr": invocation.capture_stderr,
    }
    return hashlib.sha256(canonical_json_bytes(preimage)).hexdigest()


@dataclass(frozen=True, slots=True)
class H8IssuedLaunchRecord:
    """One issued H8 launch with every parent-owned boundary retained."""

    request: H8ChildRequest
    invocation: H8ChildInvocation
    process_record: H8ChildProcessRecord
    repository_root: Path
    launch_contract_sha256: str
    attempt: H8ChildAttemptRecord

    def __post_init__(self) -> None:
        if type(self.request) is not H8ChildRequest:
            raise ValueError("request must be an exact H8ChildRequest")
        self.request.__post_init__()
        if type(self.invocation) is not H8ChildInvocation:
            raise ValueError("invocation must be an exact H8ChildInvocation")
        if type(self.process_record) is not H8ChildProcessRecord:
            raise ValueError(
                "process_record must be an exact H8ChildProcessRecord"
            )
        self.process_record.__post_init__()
        root = _resolved_repository_root(self.repository_root)
        if self.repository_root != root or self.invocation.cwd != root:
            raise ValueError("issued launch repository root is not exact")
        expected_contract = build_h8_launch_contract_sha256(
            self.invocation,
            repository_root=root,
        )
        if self.launch_contract_sha256 != expected_contract:
            raise ValueError("issued launch contract SHA-256 is stale")
        if self.invocation.stdin != _canonical_request_bytes(self.request) + b"\n":
            raise ValueError("issued request does not match invocation stdin")
        if type(self.attempt) is not H8ChildAttemptRecord:
            raise ValueError(
                "issued request, process, and attempt identities are not cross-bound"
            )
        _require_h8_issued_attempt_binding(
            self.attempt,
            request=self.request,
            invocation=self.invocation,
            process_record=self.process_record,
        )


@dataclass(frozen=True, slots=True)
class H8ParentAttemptRun:
    """The exact preallocated plan and the immutable issued launch prefix."""

    authorization: H8ChildStartAuthorization
    request_plan: tuple[H8ChildRequest, ...]
    issued: tuple[H8IssuedLaunchRecord, ...]
    attempts: tuple[H8ChildAttemptRecord, ...]

    def __post_init__(self) -> None:
        _validate_h8_child_start_authorization(self.authorization)
        if (
            type(self.request_plan) is not tuple
            or any(type(item) is not H8ChildRequest for item in self.request_plan)
            or type(self.issued) is not tuple
            or any(type(item) is not H8IssuedLaunchRecord for item in self.issued)
            or type(self.attempts) is not tuple
            or any(
                type(item) is not H8ChildAttemptRecord for item in self.attempts
            )
        ):
            raise ValueError("parent run inventories must be exact typed tuples")
        for item in self.issued:
            item.__post_init__()
        if not self.authorization.valid_start:
            if self.request_plan or self.issued or self.attempts:
                raise ValueError(
                    "unauthorized parent runs cannot retain issued records"
                )
            return
        expected_plan = build_h8_child_request_plan(
            config_sha256=self.authorization.config_sha256,
            protocol_sha256=self.authorization.protocol_sha256,
        )
        if self.request_plan != expected_plan:
            raise ValueError("parent request plan is not the frozen plan")
        if tuple(item.request for item in self.issued) != self.request_plan[
            : len(self.issued)
        ]:
            raise ValueError("issued launches are not an exact request prefix")
        if tuple(item.attempt for item in self.issued) != self.attempts:
            raise ValueError("issued launches and attempts are not cross-bound")
        failed_indices = tuple(
            index
            for index, attempt in enumerate(self.attempts)
            if attempt.status is GateStatus.FAIL
        )
        if failed_indices and failed_indices != (len(self.attempts) - 1,):
            raise ValueError("parent run continued after the first witnessed FAIL")


def _mint_h8_parent_attempt_authority(
    parent_run: H8ParentAttemptRun,
) -> H8ParentAttemptAuthority:
    """Mint authority only after exact parent-run validation succeeds."""

    if type(parent_run) is not H8ParentAttemptRun:
        raise ValueError("parent_run must be an exact H8ParentAttemptRun")
    parent_run.__post_init__()
    if not parent_run.authorization.valid_start:
        raise ValueError("an unauthorized parent run cannot mint authority")
    return _issue_h8_parent_attempt_authority(
        source_run=parent_run,
        source_authorization=parent_run.authorization,
        request_plan=parent_run.request_plan,
        issued_prefix=parent_run.issued,
        attempts=parent_run.attempts,
    )


def collect_h8_parent_identities() -> Mapping[str, object]:
    """Collect the patchable parent-owned H8 runtime identity inventory."""

    return _collect_h8_runtime_identities()


def _abnormal_spawn_record(
    error: OSError,
    *,
    parent_elapsed_ns: int,
) -> H8ChildProcessRecord:
    error_code = (
        -abs(error.errno)
        if type(error.errno) is int and error.errno != 0
        else -1
    )
    return H8ChildProcessRecord(
        timed_out=False,
        exit_code=error_code,
        stdout=b"",
        stderr=str(error).encode("utf-8", errors="replace"),
        parent_elapsed_ns=parent_elapsed_ns,
    )


def _run_h8_parent_attempt_with_runner(
    *,
    authorization: H8ChildStartAuthorization,
    repository_root: str | Path,
    identities: Mapping[str, object],
    base_environment: Mapping[str, str] | None = None,
    child_runner: Callable[
        [H8ChildInvocation],
        H8ChildProcessRecord,
    ] = run_h8_child,
) -> H8ParentAttemptRun:
    """Issue the authorized H8 prefix once, stopping at the first FAIL."""

    authorization = _validate_h8_child_start_authorization(authorization)
    if not authorization.valid_start:
        return H8ParentAttemptRun(
            authorization=authorization,
            request_plan=(),
            issued=(),
            attempts=(),
        )
    if not callable(child_runner):
        raise ValueError("child_runner must be callable")
    root = _resolved_repository_root(repository_root)
    request_plan = build_h8_child_request_plan(
        config_sha256=authorization.config_sha256,
        protocol_sha256=authorization.protocol_sha256,
    )
    issued: list[H8IssuedLaunchRecord] = []
    attempts: list[H8ChildAttemptRecord] = []
    for request in request_plan:
        invocation = build_h8_child_invocation(
            {
                name: getattr(request, name)
                for name in H8_CHILD_REQUEST_KEYS
            },
            repository_root=root,
            identities=identities,
            base_environment=base_environment,
        )
        launch_contract_sha256 = build_h8_launch_contract_sha256(
            invocation,
            repository_root=root,
        )
        started = time.perf_counter_ns()
        try:
            process_record = child_runner(invocation)
        except OSError as error:
            process_record = _abnormal_spawn_record(
                error,
                parent_elapsed_ns=time.perf_counter_ns() - started,
            )
        if type(process_record) is not H8ChildProcessRecord:
            raise ValueError(
                "child_runner must return an exact H8ChildProcessRecord"
            )
        process_record.__post_init__()
        if (
            build_h8_launch_contract_sha256(
                invocation,
                repository_root=root,
            )
            != launch_contract_sha256
        ):
            raise ValueError("launch contract drifted while the child ran")
        decision = classify_h8_child_outcome(
            process_record,
            valid_start=authorization.valid_start,
            invocation=invocation,
        )
        attempt = make_h8_child_attempt_record(
            request,
            invocation,
            process_record,
            decision,
        )
        issued.append(
            H8IssuedLaunchRecord(
                request=request,
                invocation=invocation,
                process_record=process_record,
                repository_root=root,
                launch_contract_sha256=launch_contract_sha256,
                attempt=attempt,
            )
        )
        attempts.append(attempt)
        if attempt.status is GateStatus.FAIL:
            break
    return H8ParentAttemptRun(
        authorization=authorization,
        request_plan=request_plan,
        issued=tuple(issued),
        attempts=tuple(attempts),
    )


def _run_h8_parent_attempt_for_test(
    *,
    authorization: H8ChildStartAuthorization,
    repository_root: str | Path,
    identities: Mapping[str, object],
    child_runner: Callable[
        [H8ChildInvocation],
        H8ChildProcessRecord,
    ],
    base_environment: Mapping[str, str] | None = None,
) -> H8ParentAttemptRun:
    """Private fake-runner seam; production authority never accepts it."""

    return _run_h8_parent_attempt_with_runner(
        authorization=authorization,
        repository_root=repository_root,
        identities=identities,
        base_environment=base_environment,
        child_runner=child_runner,
    )


def run_h8_parent_attempt(
    *,
    authorization: H8ChildStartAuthorization,
    repository_root: str | Path,
    base_environment: Mapping[str, str] | None = None,
) -> H8ParentAttemptAuthority:
    """Run the fixed production child entry point for one authorized prefix."""

    identities = collect_h8_parent_identities()
    return _mint_h8_parent_attempt_authority(
        _run_h8_parent_attempt_with_runner(
            authorization=authorization,
            repository_root=repository_root,
            identities=identities,
            base_environment=base_environment,
            child_runner=run_h8_child,
        )
    )


__all__ = [
    "H8ChildStartAuthorization",
    "H8IssuedLaunchRecord",
    "H8ParentAttemptAuthority",
    "H8ParentAttemptRun",
    "build_h8_child_request_plan",
    "build_h8_launch_contract_sha256",
    "build_h8_protocol_sha256",
    "collect_h8_parent_identities",
    "derive_h8_child_start_authorization",
    "run_h8_parent_attempt",
]
