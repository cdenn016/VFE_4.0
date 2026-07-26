"""Literal operand-shaped numerical budgets for the H8 correctness grid.

This module deliberately knows nothing about an H8 model, factor, or
quadrature implementation.  It consumes only the immutable Task 1 operand
records and applies the preregistered formulas without a convenience
``tolerance`` parameter.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import time
import zlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from vfe4.numerics.block_layout import (
    H8_MAX_STORAGE_SCALARS,
    BlockChainLayout,
    BlockId,
)
from vfe4.types.h8 import (
    BackendCounterSnapshot,
    BlockFillRecord,
    BlockStorageRecord,
    BlockWorkspaceRecord,
    H8_MAX_PROCESS_INCREMENTAL_BYTES,
    H8_MAX_SECONDS,
    H8_MAX_TORCH_POPULATION_BYTES,
    H8_MIN_CHOLESKY_PIVOT,
    H8AllocationRecord,
    H8AllowanceRecord,
    H8ChildAttemptRecord,
    H8ChildRequest,
    H8ChildResult,
    H8ControlResult,
    H8InvariantRecord,
    H8ObjectiveTerm,
    H8ObjectiveTerms,
    H8OperandRecord,
    H8ProfilerEventRecord,
    H8ResourceRecord,
    H8TensorKey,
    SparseConditionDiagnostics,
)
from vfe4.types.results import GateStatus


EPS = float.fromhex("0x1.0000000000000p-52")
ROUNDING_MULTIPLIER = 4096.0
SOLVER_RELATIVE_BUDGET = 1e-9
MAX_ALLOWANCE_FRACTION = 1e-4
H8_CHILD_SCHEMA_VERSION = "h8-child-v1"
H8_CHILD_IDENTITY_ENV = "VFE4_H8_CHILD_IDENTITIES_JSON"
H8_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
H8_CHILD_REQUEST_KEYS = (
    "mode",
    "seed",
    "repetition",
    "config_sha256",
    "protocol_sha256",
    "control_id",
)
H8_CHILD_ENVELOPE_KEYS = (
    "schema_version",
    "mode",
    "seed",
    "repetition",
    "control_id",
    "request_sha256",
    "config_sha256",
    "protocol_sha256",
    "status",
    "obligations",
    "identities",
    "result",
    "control",
    "error",
)
H8_CHILD_RESULT_KEYS = (
    "input_sha256",
    "sample_noise_sha256",
    "objective",
    "storage",
    "fill",
    "workspace",
    "counters",
    "allocation",
    "resources",
    "diagnostics",
    "operation_reachability",
    "residuals",
    "resource_decisions",
    "invariants",
)
H8_CHILD_IDENTITY_KEYS = ("hardware", "affinity", "thread", "blas")
H8_REQUIRED_OPERATIONS = (
    "factorization",
    "forward_substitution",
    "backward_substitution",
    "mean_solve",
    "logdet",
    "selected_inverse",
    "sample_width_one",
    "quadratic",
    "sparse_trace",
    "condition_estimate",
    "entropy",
    "log_normalizer",
    "complete_objective",
)
H8_OPERATION_SCOPES = {
    operation: f"production.{scope}"
    for operation, scope in (
        ("factorization", "factorization"),
        ("forward_substitution", "forward_substitution"),
        ("backward_substitution", "backward_substitution"),
        ("mean_solve", "mean_solve"),
        ("logdet", "logdet"),
        ("selected_inverse", "selected_inverse"),
        ("sample_width_one", "sample_width_one"),
        ("quadratic", "quadratic"),
        ("sparse_trace", "sparse_trace"),
        ("condition_estimate", "condition_estimate"),
        ("entropy", "entropy"),
        ("log_normalizer", "log_normalizer"),
        ("complete_objective", "complete_objective"),
    )
}
H8_SETUP_SCOPES = frozenset(
    {"production.problem_build", "production.assembly"}
)
H8_REQUIRED_RESIDUALS = (
    "factor_reconstruction",
    "solve",
    "backward_substitution",
    "selected_diagonal_symmetry",
)
H8_REQUIRED_PASS_DECISIONS = (
    "time_pass",
    "process_memory_pass",
    "torch_memory_pass",
    "rhs_width_pass",
    "sample_width_pass",
    "storage_pass",
    "offband_fill_pass",
    "forbidden_attempts_zero",
    "pivot_margin_pass",
    "finite_pass",
    "residual_allowances_pass",
    "dispatch_observed",
    "dispatch_backend_cross_check_pass",
)
H8_MAX_RESIDUAL_ALLOWANCE_FRACTION = 1e-4
H8_SCALE_LAYOUT = BlockChainLayout(horizon=128, d_z=20, d_m=20)
H8_SCALE_RESIDUAL_SPECS = {
    "factor_reconstruction": (
        (
            "factor_reconstruction:factor_diagonal->assembled_diagonal",
            "block:factor_diagonal",
            "block:assembled_diagonal",
            (129, 40, 40),
            False,
            False,
        ),
        (
            "factor_reconstruction:factor_lower->assembled_lower",
            "block:factor_lower",
            "block:assembled_lower",
            (128, 40, 40),
            False,
            False,
        ),
    ),
    "solve": (
        (
            "solve:precision_times_mean->information",
            "block:precision_times_mean",
            "block:information",
            (129, 40),
            True,
            False,
        ),
    ),
    "backward_substitution": (
        (
            "backward_substitution:explicit_backward->mean",
            "block:explicit_backward",
            "block:mean",
            (129, 40),
            True,
            True,
        ),
    ),
    "selected_diagonal_symmetry": (
        (
            "selected_diagonal_symmetry:diagonal->transpose",
            "block:selected_diagonal",
            "block:selected_diagonal_transpose",
            (129, 40, 40),
            True,
            True,
        ),
    ),
}
H8_REQUIRED_NUMPY_PRODUCERS = frozenset(
    {
        "numpy.random.Generator.standard_normal",
        "numpy.asarray",
        "numpy.ascontiguousarray",
        "numpy.multiply",
        "numpy.add",
        "numpy.all",
        "numpy.divide",
        "numpy.eye",
        "numpy.isfinite",
        "numpy.matmul",
        "numpy.sqrt",
        "numpy.transpose",
        "numpy.linalg.cholesky",
        "numpy.linalg.norm",
    }
)
_SHA256_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True, slots=True)
class H8ChildInvocation:
    """Exact, immutable parent-side child launch contract."""

    argv: tuple[str, ...]
    cwd: Path
    stdin: bytes
    environment: Mapping[str, str]
    timeout_seconds: float
    capture_stdout: bool = True
    capture_stderr: bool = True

    def __post_init__(self) -> None:
        expected_argv = (sys.executable, "-m", "verification.h8_child")
        if self.argv != expected_argv:
            raise ValueError("child argv must use the exact H8 child module")
        if not isinstance(self.cwd, Path) or not self.cwd.is_absolute():
            raise ValueError("child cwd must be an absolute Path")
        if type(self.stdin) is not bytes:
            raise ValueError("child stdin must be immutable bytes")
        if not isinstance(self.environment, Mapping):
            raise ValueError("child environment must map strings to strings")
        owned_environment = dict(self.environment)
        if any(
            type(key) is not str
            or not key
            or type(value) is not str
            for key, value in owned_environment.items()
        ):
            raise ValueError("child environment must map strings to strings")
        if any(
            owned_environment.get(name) != "1"
            for name in H8_THREAD_ENVIRONMENT
        ):
            raise ValueError(
                "child thread environment must retain every frozen one-thread value"
            )
        object.__setattr__(
            self,
            "environment",
            MappingProxyType(owned_environment),
        )
        if (
            type(self.timeout_seconds) is not float
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds != H8_MAX_SECONDS
        ):
            raise ValueError("child timeout must equal the exact H8 timeout")
        if (
            type(self.capture_stdout) is not bool
            or type(self.capture_stderr) is not bool
            or not self.capture_stdout
            or not self.capture_stderr
        ):
            raise ValueError("child launch must capture stdout and stderr")


@dataclass(frozen=True, slots=True)
class H8ChildProcessRecord:
    """Raw process evidence retained even when child JSON cannot be parsed."""

    timed_out: bool
    exit_code: int | None
    stdout: bytes
    stderr: bytes
    parent_elapsed_ns: int

    def __post_init__(self) -> None:
        if type(self.timed_out) is not bool:
            raise ValueError("timed_out must be a bool")
        if self.exit_code is not None and type(self.exit_code) is not int:
            raise ValueError("exit_code must be an integer or None")
        if type(self.stdout) is not bytes or type(self.stderr) is not bytes:
            raise ValueError("stdout and stderr must be exact byte strings")
        if type(self.parent_elapsed_ns) is not int or self.parent_elapsed_ns < 0:
            raise ValueError("parent_elapsed_ns must be nonnegative")
        if self.timed_out and self.exit_code is not None:
            raise ValueError("a timed-out child has no observed exit code")

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
        *,
        exit_code: int = 0,
        stderr: bytes = b"",
        parent_elapsed_ns: int = 0,
    ) -> "H8ChildProcessRecord":
        return cls(
            timed_out=False,
            exit_code=exit_code,
            stdout=canonical_json_bytes(payload) + b"\n",
            stderr=stderr,
            parent_elapsed_ns=parent_elapsed_ns,
        )


@dataclass(frozen=True, slots=True)
class H8ChildDecision:
    """Parent-side status with raw process identities preserved."""

    status: GateStatus
    reasons: tuple[str, ...]
    payload: Mapping[str, object] | None
    timed_out: bool
    exit_code: int | None
    parent_elapsed_ns: int
    stdout_sha256: str
    stderr_sha256: str


def canonical_json_bytes(value: object) -> bytes:
    """Return strict UTF-8 canonical JSON without a trailing newline."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def make_h8_identity_record(
    kind: str,
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Bind one exact identity payload to its own canonical SHA-256."""

    if kind not in H8_CHILD_IDENTITY_KEYS:
        raise ValueError("identity kind is outside the frozen inventory")
    if not isinstance(payload, Mapping):
        raise ValueError("identity payload must be a mapping")
    body = {"kind": kind, **dict(payload)}
    if "sha256" in body:
        raise ValueError("identity payload cannot prestate its own hash")
    return {
        **body,
        "sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
    }


def _validate_identity_records(
    identities: object,
    *,
    allow_observability_error: bool = False,
) -> dict[str, dict[str, object]]:
    if not isinstance(identities, Mapping) or tuple(sorted(identities)) != tuple(
        sorted(H8_CHILD_IDENTITY_KEYS)
    ):
        raise ValueError("child identities must use the exact frozen key set")
    checked: dict[str, dict[str, object]] = {}
    for kind in H8_CHILD_IDENTITY_KEYS:
        value = identities[kind]
        if not isinstance(value, Mapping):
            raise ValueError(f"{kind} identity must be a mapping")
        record = dict(value)
        digest = record.pop("sha256", None)
        if record.get("kind") != kind or not _is_sha256(digest):
            raise ValueError(f"{kind} identity metadata is invalid")
        expected = hashlib.sha256(canonical_json_bytes(record)).hexdigest()
        if digest != expected:
            raise ValueError(f"{kind} identity SHA-256 mismatch")
        if allow_observability_error and set(record) == {
            "kind",
            "observability_error",
        }:
            if (
                type(record["observability_error"]) is not str
                or not record["observability_error"]
            ):
                raise ValueError(f"{kind} observability error is invalid")
            checked[kind] = {**record, "sha256": digest}
            continue
        _validate_identity_payload(kind, record)
        checked[kind] = {**record, "sha256": digest}
    return checked


def _validate_identity_payload(
    kind: str,
    record: Mapping[str, object],
) -> None:
    keys = set(record)
    if kind == "hardware":
        if keys != {
            "kind",
            "platform",
            "system",
            "machine",
            "processor",
            "cpu_count",
            "python",
            "implementation",
        }:
            raise ValueError("hardware identity does not use its exact schema")
        if any(
            type(record[name]) is not str
            for name in ("platform", "system", "machine", "processor", "python", "implementation")
        ) or any(
            not record[name]
            for name in ("platform", "system", "python", "implementation")
        ):
            raise ValueError("hardware identity text is unavailable")
        if type(record["cpu_count"]) is not int or record["cpu_count"] <= 0:
            raise ValueError("hardware CPU count is unavailable")
        return
    if kind == "affinity":
        common = {"kind", "adapter"}
        cpu_schema = common | {"cpus"}
        mask_schema = common | {"process_mask", "system_mask"}
        if keys not in (cpu_schema, mask_schema):
            raise ValueError("affinity identity does not use an exact schema")
        if type(record["adapter"]) is not str or not record["adapter"]:
            raise ValueError("affinity adapter is unavailable")
        if keys == cpu_schema:
            cpus = record["cpus"]
            if (
                type(cpus) is not list
                or not cpus
                or any(type(cpu) is not int or cpu < 0 for cpu in cpus)
                or len(cpus) != len(set(cpus))
            ):
                raise ValueError("affinity CPU inventory is invalid")
        elif any(
            type(record[name]) is not int or record[name] <= 0
            for name in ("process_mask", "system_mask")
        ):
            raise ValueError("affinity masks are invalid")
        return
    if kind == "thread":
        if keys != {
            "kind",
            "environment",
            "torch_num_threads",
            "torch_num_interop_threads",
        }:
            raise ValueError("thread identity does not use its exact schema")
        environment = record["environment"]
        if (
            not isinstance(environment, Mapping)
            or set(environment) != set(H8_THREAD_ENVIRONMENT)
            or any(environment[name] != "1" for name in H8_THREAD_ENVIRONMENT)
            or record["torch_num_threads"] != 1
            or record["torch_num_interop_threads"] != 1
        ):
            raise ValueError("thread identity is not the frozen one-thread record")
        return
    if kind == "blas":
        if keys != {
            "kind",
            "torch_version",
            "numpy_version",
            "torch_config",
            "numpy_config",
        }:
            raise ValueError("BLAS identity does not use its exact schema")
        if any(
            type(record[name]) is not str
            for name in (
                "torch_version",
                "numpy_version",
                "torch_config",
                "numpy_config",
            )
        ) or not record["torch_version"] or not record["numpy_version"]:
            raise ValueError("BLAS identity is unavailable")
        return
    raise ValueError("identity kind is outside the frozen inventory")


def _validate_child_request(request: object) -> dict[str, object]:
    if not isinstance(request, Mapping) or set(request) != set(H8_CHILD_REQUEST_KEYS):
        raise ValueError("H8 child request must use the exact six-field schema")
    copied = {name: request[name] for name in H8_CHILD_REQUEST_KEYS}
    H8ChildRequest(
        mode=copied["mode"],  # type: ignore[arg-type]
        seed=copied["seed"],  # type: ignore[arg-type]
        repetition=copied["repetition"],  # type: ignore[arg-type]
        config_sha256=copied["config_sha256"],  # type: ignore[arg-type]
        protocol_sha256=copied["protocol_sha256"],  # type: ignore[arg-type]
        control_id=copied["control_id"],  # type: ignore[arg-type]
    )
    return copied


def build_h8_child_invocation(
    request: Mapping[str, object],
    *,
    repository_root: str | Path,
    identities: Mapping[str, object],
    base_environment: Mapping[str, str] | None = None,
) -> H8ChildInvocation:
    """Build the literal subprocess contract without launching a child."""

    checked_request = _validate_child_request(request)
    checked_identities = _validate_identity_records(identities)
    root = Path(repository_root).resolve()
    if not root.is_dir():
        raise ValueError("repository_root must be an existing directory")
    environment = dict(os.environ if base_environment is None else base_environment)
    for name in H8_THREAD_ENVIRONMENT:
        environment[name] = "1"
    environment[H8_CHILD_IDENTITY_ENV] = canonical_json_bytes(
        checked_identities
    ).decode("ascii")
    timeout = H8_MAX_SECONDS if checked_request["mode"] == "production" else 60.0
    return H8ChildInvocation(
        argv=(sys.executable, "-m", "verification.h8_child"),
        cwd=root,
        stdin=canonical_json_bytes(checked_request) + b"\n",
        environment=environment,
        timeout_seconds=timeout,
    )


def run_h8_child(invocation: H8ChildInvocation) -> H8ChildProcessRecord:
    """Launch exactly one cold child and retain every raw process endpoint."""

    if type(invocation) is not H8ChildInvocation:
        raise ValueError("invocation must be an H8ChildInvocation")
    started = time.perf_counter_ns()
    try:
        completed = subprocess.run(  # noqa: S603
            invocation.argv,
            input=invocation.stdin,
            cwd=invocation.cwd,
            env=dict(invocation.environment),
            capture_output=True,
            timeout=invocation.timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        stdout = _timeout_bytes(error.stdout)
        try:
            parse_h8_child_stdout(stdout)
        except ValueError:
            pass
        elapsed = time.perf_counter_ns() - started
        return H8ChildProcessRecord(
            timed_out=True,
            exit_code=None,
            stdout=stdout,
            stderr=_timeout_bytes(error.stderr),
            parent_elapsed_ns=elapsed,
        )
    try:
        parse_h8_child_stdout(completed.stdout)
    except ValueError:
        pass
    return H8ChildProcessRecord(
        timed_out=False,
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        parent_elapsed_ns=time.perf_counter_ns() - started,
    )


def _timeout_bytes(value: bytes | str | None) -> bytes:
    if value is None:
        return b""
    return value if type(value) is bytes else value.encode("utf-8", errors="replace")


def parse_h8_child_stdout(stdout: bytes) -> dict[str, object]:
    """Parse one strict, canonical, newline-terminated H8 child envelope."""

    if type(stdout) is not bytes or not stdout.endswith(b"\n"):
        raise ValueError("child stdout must contain one canonical JSON line")
    if stdout.count(b"\n") != 1 or not stdout[:-1]:
        raise ValueError("child stdout must contain one canonical JSON line")
    try:
        decoded = stdout[:-1].decode("utf-8", errors="strict")
        value = json.loads(
            decoded,
            parse_constant=lambda constant: (_raise_nonfinite(constant)),
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("child stdout is not one JSON object") from error
    if not isinstance(value, dict):
        raise ValueError("child stdout must contain one JSON object")
    if canonical_json_bytes(value) + b"\n" != stdout:
        raise ValueError("child stdout JSON is not canonical")
    _validate_child_envelope(value)
    return value


def decode_h8_child_result(
    envelope: Mapping[str, object],
) -> H8ChildResult:
    """Decode one validated successful production/profiler envelope."""

    decoded = _validate_child_envelope(envelope)
    if type(decoded) is not H8ChildResult:
        raise ValueError(
            "envelope does not contain a successful H8 child result"
        )
    return decoded


def decode_h8_control_result(
    envelope: Mapping[str, object],
) -> H8ControlResult:
    """Decode one validated successful negative-control envelope."""

    decoded = _validate_child_envelope(envelope)
    if type(decoded) is not H8ControlResult:
        raise ValueError(
            "envelope does not contain a successful H8 control result"
        )
    return decoded


def _raise_nonfinite(constant: str) -> object:
    raise ValueError(f"child output contains a nonfinite JSON value: {constant}")


def _reject_duplicate_pairs(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _validate_child_envelope(
    value: Mapping[str, object],
) -> H8ChildResult | H8ControlResult | None:
    if set(value) != set(H8_CHILD_ENVELOPE_KEYS):
        raise ValueError("child envelope does not use its exact key set")
    if value["schema_version"] != H8_CHILD_SCHEMA_VERSION:
        raise ValueError("child schema version mismatch")
    mode = value["mode"]
    if mode not in ("production", "profiler", "negative_control"):
        raise ValueError("child envelope mode is outside the frozen union")
    if type(value["seed"]) is not int or value["seed"] <= 0:
        raise ValueError("child envelope seed must be a positive integer")
    repetition = value["repetition"]
    control_id = value["control_id"]
    if mode == "production":
        if type(repetition) is not int or not 0 <= repetition <= 4 or control_id is not None:
            raise ValueError("production result identity is invalid")
    elif mode == "profiler":
        if repetition is not None or control_id is not None:
            raise ValueError("profiler result identity is invalid")
    elif repetition is not None or type(control_id) is not str:
        raise ValueError("negative-control result identity is invalid")
    for name in ("request_sha256", "config_sha256", "protocol_sha256"):
        if not _is_sha256(value[name]):
            raise ValueError(f"{name} is not a lowercase SHA-256")
    status = value["status"]
    if status not in ("pass", "fail", "inconclusive"):
        raise ValueError("child status is outside the closed union")
    obligations = value["obligations"]
    if (
        type(obligations) is not list
        or len(obligations) != len(set(obligations))
        or any(type(item) is not str or not item for item in obligations)
    ):
        raise ValueError("child obligations must be unique nonempty strings")
    if status == "inconclusive" and not obligations:
        raise ValueError("inconclusive child result requires an obligation")
    if status != "inconclusive" and obligations:
        raise ValueError("conclusive child result cannot retain obligations")
    _validate_identity_records(
        value["identities"],
        allow_observability_error=status != "pass",
    )
    result, control, error = value["result"], value["control"], value["error"]
    if mode in ("production", "profiler") and status == "pass":
        if not isinstance(result, Mapping) or control is not None or error is not None:
            raise ValueError("successful production/profiler result is incomplete")
        if set(result) != set(H8_CHILD_RESULT_KEYS):
            raise ValueError("child result does not use its exact key set")
    decoded: H8ChildResult | H8ControlResult | None = None
    if mode == "negative_control" and status == "pass":
        if result is not None or not isinstance(control, Mapping) or error is not None:
            raise ValueError("successful negative-control result is incomplete")
        decoded = _validate_negative_control_pass(
            control,
            control_id=control_id,
        )
    if error is not None:
        if not isinstance(error, Mapping) or set(error) != {
            "kind",
            "message",
            "witnessed_violation",
        }:
            raise ValueError("child error does not use its exact key set")
        if (
            type(error["kind"]) is not str
            or not error["kind"]
            or type(error["message"]) is not str
            or not error["message"]
            or type(error["witnessed_violation"]) is not bool
        ):
            raise ValueError("child error fields are invalid")
    if result is not None and (
        not isinstance(result, Mapping) or set(result) != set(H8_CHILD_RESULT_KEYS)
    ):
        raise ValueError("child result does not use its exact key set")
    if status == "pass" and mode in ("production", "profiler"):
        decoded = _validate_complete_pass_result(
            result,
            mode=mode,
            seed=value["seed"],
            repetition=repetition,
        )
    return decoded


def _decode_json_shapes(value: object, *, name: str) -> tuple[tuple[int, ...], ...]:
    shapes: list[tuple[int, ...]] = []
    for index, item in enumerate(_exact_list(value, name=name)):
        dimensions = tuple(
            _exact_list(item, name=f"{name}[{index}]")
        )
        if any(type(dimension) is not int or dimension < 0 for dimension in dimensions):
            raise ValueError(f"{name}[{index}] has an invalid dimension")
        shapes.append(dimensions)
    return tuple(shapes)


def _decode_control_summary(value: object) -> H8ControlResult:
    record = _exact_mapping(
        value,
        (
            "control_id",
            "requested_operation",
            "logical_shapes",
            "assigned_channels",
            "observed_channels",
            "execution_witnessed",
            "event_sha256",
            "assignment_complete",
            "detected",
            "status",
            "obligations",
        ),
        name="control.summary",
    )
    try:
        status = GateStatus(record["status"])
    except (TypeError, ValueError) as error:
        raise ValueError("control.summary status is invalid") from error
    return H8ControlResult(
        control_id=record["control_id"],  # type: ignore[arg-type]
        requested_operation=record["requested_operation"],  # type: ignore[arg-type]
        logical_shapes=_decode_json_shapes(
            record["logical_shapes"],
            name="control.summary.logical_shapes",
        ),
        assigned_channels=tuple(
            _exact_list(
                record["assigned_channels"],
                name="control.summary.assigned_channels",
            )
        ),
        observed_channels=tuple(
            _exact_list(
                record["observed_channels"],
                name="control.summary.observed_channels",
            )
        ),
        execution_witnessed=record["execution_witnessed"],  # type: ignore[arg-type]
        event_sha256=record["event_sha256"],  # type: ignore[arg-type]
        assignment_complete=record["assignment_complete"],  # type: ignore[arg-type]
        detected=record["detected"],  # type: ignore[arg-type]
        status=status,
        obligations=tuple(
            _exact_list(
                record["obligations"],
                name="control.summary.obligations",
            )
        ),
    )


def _decode_storage_span(value: object, *, name: str) -> object:
    from vfe4.inference.h8_allocation import H8StorageSpan

    record = _exact_mapping(
        value,
        (
            "storage_key",
            "device",
            "pointer",
            "span_start",
            "span_end",
            "nbytes",
        ),
        name=name,
    )
    return H8StorageSpan(**record)  # type: ignore[arg-type]


def _decode_dispatch_control_event(value: object, *, name: str) -> object:
    from vfe4.inference.h8_allocation import H8DispatchEvent

    keys = (
        "sequence",
        "operator",
        "semantic_site",
        "control_id",
        "input_shapes",
        "output_shapes",
        "physical_output_shapes",
        "stack_member_shapes",
        "stack_member_count",
        "dtype",
        "device",
        "float64_equivalent_scalars",
        "classifications",
        "storage_spans",
        "alias_storage_keys",
        "new_storage_keys",
        "allocated_float64_equivalent_scalars",
        "live_float64_equivalent_scalars_by_site",
        "stack",
        "executed",
        "forbidden_reason",
        "live_storage_bytes_after",
        "population_live_storage_bytes_after",
    )
    record = _exact_mapping(value, keys, name=name)
    for field in (
        "sequence",
        "stack_member_count",
        "float64_equivalent_scalars",
        "allocated_float64_equivalent_scalars",
        "live_storage_bytes_after",
        "population_live_storage_bytes_after",
    ):
        if type(record[field]) is not int or record[field] < 0:  # type: ignore[operator]
            raise ValueError(f"{name}.{field} must be a nonnegative integer")
    if (
        type(record["operator"]) is not str
        or not record["operator"]
        or (
            record["semantic_site"] is not None
            and type(record["semantic_site"]) is not str
        )
        or (
            record["control_id"] is not None
            and type(record["control_id"]) is not str
        )
        or (
            record["dtype"] is not None
            and type(record["dtype"]) is not str
        )
        or (
            record["device"] is not None
            and type(record["device"]) is not str
        )
        or type(record["executed"]) is not bool
        or (
            record["forbidden_reason"] is not None
            and (
                type(record["forbidden_reason"]) is not str
                or not record["forbidden_reason"]
            )
        )
    ):
        raise ValueError(f"{name} scalar fields are invalid")
    tuple_fields = {}
    for field in (
        "classifications",
        "alias_storage_keys",
        "new_storage_keys",
        "stack",
    ):
        values = tuple(_exact_list(record[field], name=f"{name}.{field}"))
        if any(type(item) is not str or not item for item in values):
            raise ValueError(f"{name}.{field} must contain nonempty strings")
        tuple_fields[field] = values
    live_by_site: list[tuple[str, int]] = []
    for index, item in enumerate(
        _exact_list(
            record["live_float64_equivalent_scalars_by_site"],
            name=f"{name}.live_float64_equivalent_scalars_by_site",
        )
    ):
        pair = _exact_list(
            item,
            name=(
                f"{name}.live_float64_equivalent_scalars_by_site[{index}]"
            ),
        )
        if (
            len(pair) != 2
            or type(pair[0]) is not str
            or not pair[0]
            or type(pair[1]) is not int
            or pair[1] < 0
        ):
            raise ValueError(f"{name} live-site record is invalid")
        live_by_site.append((pair[0], pair[1]))
    spans = tuple(
        _decode_storage_span(item, name=f"{name}.storage_spans[{index}]")
        for index, item in enumerate(
            _exact_list(record["storage_spans"], name=f"{name}.storage_spans")
        )
    )
    if (
        record["stack_member_count"]
        != len(
            _decode_json_shapes(
                record["stack_member_shapes"],
                name=f"{name}.stack_member_shapes",
            )
        )
        or len(set(tuple_fields["alias_storage_keys"]))
        != len(tuple_fields["alias_storage_keys"])
        or len(set(tuple_fields["new_storage_keys"]))
        != len(tuple_fields["new_storage_keys"])
        or set(tuple_fields["alias_storage_keys"])
        & set(tuple_fields["new_storage_keys"])
        or tuple(live_by_site) != tuple(sorted(live_by_site))
        or len({site for site, _value in live_by_site}) != len(live_by_site)
    ):
        raise ValueError(f"{name} nested integer/key endpoints are inconsistent")
    return H8DispatchEvent(
        sequence=record["sequence"],  # type: ignore[arg-type]
        operator=record["operator"],  # type: ignore[arg-type]
        semantic_site=record["semantic_site"],  # type: ignore[arg-type]
        control_id=record["control_id"],  # type: ignore[arg-type]
        input_shapes=_decode_json_shapes(
            record["input_shapes"],
            name=f"{name}.input_shapes",
        ),
        output_shapes=_decode_json_shapes(
            record["output_shapes"],
            name=f"{name}.output_shapes",
        ),
        physical_output_shapes=_decode_json_shapes(
            record["physical_output_shapes"],
            name=f"{name}.physical_output_shapes",
        ),
        stack_member_shapes=_decode_json_shapes(
            record["stack_member_shapes"],
            name=f"{name}.stack_member_shapes",
        ),
        stack_member_count=record["stack_member_count"],  # type: ignore[arg-type]
        dtype=record["dtype"],  # type: ignore[arg-type]
        device=record["device"],  # type: ignore[arg-type]
        float64_equivalent_scalars=record[  # type: ignore[arg-type]
            "float64_equivalent_scalars"
        ],
        classifications=tuple_fields["classifications"],
        storage_spans=spans,
        alias_storage_keys=tuple_fields["alias_storage_keys"],
        new_storage_keys=tuple_fields["new_storage_keys"],
        allocated_float64_equivalent_scalars=record[  # type: ignore[arg-type]
            "allocated_float64_equivalent_scalars"
        ],
        live_float64_equivalent_scalars_by_site=tuple(live_by_site),
        stack=tuple_fields["stack"],
        executed=record["executed"],  # type: ignore[arg-type]
        forbidden_reason=record["forbidden_reason"],  # type: ignore[arg-type]
        live_storage_bytes_after=record["live_storage_bytes_after"],  # type: ignore[arg-type]
        population_live_storage_bytes_after=record[  # type: ignore[arg-type]
            "population_live_storage_bytes_after"
        ],
    )


def _dispatch_dtype_itemsize(dtype: str | None) -> int | None:
    return {
        "torch.float64": 8,
        "torch.float32": 4,
        "torch.float16": 2,
        "torch.bfloat16": 2,
        "torch.int64": 8,
        "torch.int32": 4,
        "torch.int16": 2,
        "torch.int8": 1,
        "torch.uint8": 1,
        "torch.bool": 1,
    }.get(dtype)


def _replay_dispatch_events(
    events: tuple[object, ...],
    *,
    allocation: Mapping[str, object],
) -> None:
    if not events:
        raise ValueError("PASS dispatch replay requires events")
    live_peak = 0
    population_peak = 0
    forbidden_count = 0
    for index, event in enumerate(events):
        output_shapes = event.output_shapes
        physical_shapes = event.physical_output_shapes
        spans = event.storage_spans
        classifications = event.classifications
        span_keys = tuple(span.storage_key for span in spans)
        new_keys = set(event.new_storage_keys)
        alias_keys = set(event.alias_storage_keys)
        itemsize = _dispatch_dtype_itemsize(event.dtype)
        if (
            event.sequence != index
            or event.control_id is not None
            or len(output_shapes) != len(physical_shapes)
            or len(output_shapes) != len(spans)
            or len(output_shapes) != len(classifications)
            or not new_keys.issubset(span_keys)
            or not alias_keys.issubset(span_keys)
            or any(
                event.device is not None and span.device != event.device
                for span in spans
            )
            or event.population_live_storage_bytes_after
            > event.live_storage_bytes_after
        ):
            raise ValueError("PASS dispatch replay structure is inconsistent")
        if spans:
            if itemsize is None or event.device is None:
                raise ValueError("PASS dispatch output dtype/device is unavailable")
            equivalents = tuple(
                math.ceil(math.prod(shape) * itemsize / 8)
                for shape in output_shapes
            )
            if event.float64_equivalent_scalars != sum(equivalents):
                raise ValueError("PASS dispatch logical scalar endpoint drifted")
            allocated = 0
            counted: set[str] = set()
            for span, equivalent in zip(spans, equivalents, strict=True):
                if span.storage_key in new_keys and span.storage_key not in counted:
                    allocated += equivalent
                    counted.add(span.storage_key)
            if event.allocated_float64_equivalent_scalars != allocated:
                raise ValueError("PASS dispatch allocation endpoint drifted")
        elif (
            event.dtype is not None
            or event.device is not None
            or event.float64_equivalent_scalars != 0
            or event.allocated_float64_equivalent_scalars != 0
        ):
            raise ValueError("PASS dispatch empty-output endpoints are inconsistent")
        if event.forbidden_reason is not None:
            forbidden_count += 1
        elif not event.executed and event.operator != "PREEXISTING":
            raise ValueError("PASS dispatch contains an unexecuted production event")
        live_peak = max(live_peak, event.live_storage_bytes_after)
        population_peak = max(
            population_peak,
            event.population_live_storage_bytes_after,
        )
    if (
        allocation["dispatch_event_count"] != len(events)
        or allocation["dispatch_forbidden_attempt_count"] != forbidden_count
        or forbidden_count != 0
        or allocation["dispatch_live_peak_bytes"] != live_peak
        or allocation["torch_population_peak_bytes"] != population_peak
        or allocation["dispatch_trace_sha256"]
        != _task5_canonical_sha256(list(events))
    ):
        raise ValueError("PASS dispatch replay endpoints or trace digest drifted")


def _decode_numpy_control_event(value: object, *, name: str) -> object:
    from vfe4.inference.h8_allocation import H8NumpyGuardEvent

    record = _exact_mapping(
        value,
        (
            "sequence",
            "operator",
            "semantic_site",
            "control_id",
            "input_shapes",
            "output_shapes",
            "dtype",
            "float64_equivalent_scalars",
            "executed",
            "forbidden_reason",
        ),
        name=name,
    )
    if (
        type(record["sequence"]) is not int
        or record["sequence"] < 0  # type: ignore[operator]
        or type(record["operator"]) is not str
        or not record["operator"]
        or (
            record["semantic_site"] is not None
            and type(record["semantic_site"]) is not str
        )
        or (
            record["control_id"] is not None
            and type(record["control_id"]) is not str
        )
        or (
            record["dtype"] is not None
            and type(record["dtype"]) is not str
        )
        or type(record["float64_equivalent_scalars"]) is not int
        or record["float64_equivalent_scalars"] < 0  # type: ignore[operator]
        or type(record["executed"]) is not bool
        or (
            record["forbidden_reason"] is not None
            and (
                type(record["forbidden_reason"]) is not str
                or not record["forbidden_reason"]
            )
        )
    ):
        raise ValueError(f"{name} fields are invalid")
    return H8NumpyGuardEvent(
        sequence=record["sequence"],  # type: ignore[arg-type]
        operator=record["operator"],  # type: ignore[arg-type]
        semantic_site=record["semantic_site"],  # type: ignore[arg-type]
        control_id=record["control_id"],  # type: ignore[arg-type]
        input_shapes=_decode_json_shapes(
            record["input_shapes"],
            name=f"{name}.input_shapes",
        ),
        output_shapes=_decode_json_shapes(
            record["output_shapes"],
            name=f"{name}.output_shapes",
        ),
        dtype=record["dtype"],  # type: ignore[arg-type]
        float64_equivalent_scalars=record[  # type: ignore[arg-type]
            "float64_equivalent_scalars"
        ],
        executed=record["executed"],  # type: ignore[arg-type]
        forbidden_reason=record["forbidden_reason"],  # type: ignore[arg-type]
    )


def _validate_negative_control_pass(
    value: object,
    *,
    control_id: object,
) -> H8ControlResult:
    from vfe4.inference.h8_allocation import (
        h8_control_detected_pre_execution,
        h8_negative_control_specs,
        make_h8_control_result,
    )

    wrapper = _exact_mapping(
        value,
        ("summary", "evidence"),
        name="control",
    )
    summary = _decode_control_summary(wrapper["summary"])
    if type(control_id) is not str or summary.control_id != control_id:
        raise ValueError("control identity does not match the envelope")
    specs = tuple(
        item
        for item in h8_negative_control_specs(H8_SCALE_LAYOUT)
        if item.control_id == control_id
    )
    if len(specs) != 1:
        raise ValueError("control identity is outside the frozen inventory")
    spec = specs[0]
    evidence = wrapper["evidence"]
    if type(evidence) is not dict:
        raise ValueError("control PASS requires raw event evidence")
    if control_id.startswith("numpy_"):
        expected_keys = (
            "events",
            "operation_returned",
            "caught_forbidden",
            "pre_execution_detected",
            "executed_past_detector",
        )
        record = _exact_mapping(evidence, expected_keys, name="control.evidence")
        events = tuple(
            _decode_numpy_control_event(
                item,
                name=f"control.evidence.events[{index}]",
            )
            for index, item in enumerate(
                _exact_list(record["events"], name="control.evidence.events")
            )
        )
        backend = None
        payload: object = {
            "events": events,
            "operation_returned": record["operation_returned"],
            "caught_forbidden": record["caught_forbidden"],
            "pre_execution_detected": record["pre_execution_detected"],
            "executed_past_detector": record["executed_past_detector"],
        }
    else:
        expected_keys = (
            "dispatch",
            "backend",
            "operation_returned",
            "caught_forbidden",
            "pre_execution_detected",
            "executed_past_detector",
        )
        record = _exact_mapping(evidence, expected_keys, name="control.evidence")
        events = tuple(
            _decode_dispatch_control_event(
                item,
                name=f"control.evidence.dispatch[{index}]",
            )
            for index, item in enumerate(
                _exact_list(record["dispatch"], name="control.evidence.dispatch")
            )
        )
        backend_value = record["backend"]
        if control_id == "torch_eye_full_rhs":
            backend_record = _exact_mapping(
                backend_value,
                (
                    "before",
                    "after",
                    "detected",
                    "executed_past_detector",
                    "unexpected_exception",
                ),
                name="control.evidence.backend",
            )
            before = tuple(
                _exact_list(
                    backend_record["before"],
                    name="control.evidence.backend.before",
                )
            )
            after = tuple(
                _exact_list(
                    backend_record["after"],
                    name="control.evidence.backend.after",
                )
            )
            if (
                any(type(item) is not int or item <= 0 for item in before + after)
                or after != before + (H8_SCALE_LAYOUT.dimension,)
                or backend_record["detected"] is not True
                or backend_record["executed_past_detector"] is not False
                or backend_record["unexpected_exception"] is not None
            ):
                raise ValueError("control backend evidence is inconsistent")
            backend: object | None = {
                "before": before,
                "after": after,
                "detected": True,
                "executed_past_detector": False,
                "unexpected_exception": None,
            }
        elif backend_value is not None:
            raise ValueError("control has an unexpected backend witness")
        else:
            backend = None
        payload = {
            "dispatch": events,
            "backend": backend,
            "operation_returned": record["operation_returned"],
            "caught_forbidden": record["caught_forbidden"],
            "pre_execution_detected": record["pre_execution_detected"],
            "executed_past_detector": record["executed_past_detector"],
        }
    if (
        type(record["operation_returned"]) is not bool
        or type(record["caught_forbidden"]) is not bool
        or type(record["pre_execution_detected"]) is not bool
        or type(record["executed_past_detector"]) is not bool
        or not events
        or any(
            event.control_id != control_id
            or event.executed
            or event.forbidden_reason != spec.expected_reason
            for event in events
        )
    ):
        raise ValueError("control raw event evidence is inconsistent")
    detected = h8_control_detected_pre_execution(
        spec,
        events,
        operation_returned=record["operation_returned"],  # type: ignore[arg-type]
        caught_forbidden=record["caught_forbidden"],  # type: ignore[arg-type]
    )
    executed_past = record["operation_returned"] or any(
        event.executed for event in events
    )
    if (
        record["pre_execution_detected"] is not detected
        or record["executed_past_detector"] is not executed_past
        or detected is not True
        or executed_past is not False
    ):
        raise ValueError("control detector decisions do not derive from raw events")
    observed = ["numpy_guard" if control_id.startswith("numpy_") else "dispatch"]
    if control_id == "torch_eye_full_rhs":
        observed.append("backend")
    recomputed = make_h8_control_result(
        spec,
        observed_channels=tuple(observed),  # type: ignore[arg-type]
        detected=detected,
        event_payload=payload,
    )
    if summary != recomputed or summary.status is not GateStatus.PASS:
        raise ValueError("control summary does not match raw event reconstruction")
    return summary


def _validate_complete_pass_result(
    result: object,
    *,
    mode: object,
    seed: object,
    repetition: object,
) -> H8ChildResult:
    if not isinstance(result, Mapping):
        raise ValueError("PASS requires complete nested evidence")
    if not _is_sha256(result.get("input_sha256")) or not _is_sha256(
        result.get("sample_noise_sha256")
    ):
        raise ValueError("PASS requires complete nested evidence hashes")
    for name in (
        "objective",
        "storage",
        "fill",
        "workspace",
        "counters",
        "allocation",
        "resources",
        "diagnostics",
        "operation_reachability",
        "residuals",
        "resource_decisions",
    ):
        if not isinstance(result.get(name), Mapping) or not result[name]:
            raise ValueError(f"PASS requires complete nested evidence: {name}")
    invariants = result.get("invariants")
    if type(invariants) is not list or not invariants:
        raise ValueError("PASS requires complete nested evidence: invariants")
    typed = _validate_typed_pass_sections(result)
    reachability = result["operation_reachability"]
    if (
        set(reachability) != set(H8_REQUIRED_OPERATIONS)
        or any(reachability[name] is not True for name in H8_REQUIRED_OPERATIONS)
    ):
        raise ValueError("PASS requires exact true operation reachability")
    residuals = result["residuals"]
    if set(residuals) != set(H8_REQUIRED_RESIDUALS) or any(
        type(residuals[name]) not in (int, float)
        or not math.isfinite(float(residuals[name]))
        or float(residuals[name]) < 0.0
        for name in H8_REQUIRED_RESIDUALS
    ):
        raise ValueError("PASS requires exact finite residual endpoints")
    decisions = result["resource_decisions"]
    missing_decisions = set(H8_REQUIRED_PASS_DECISIONS) - set(decisions)
    if missing_decisions or any(
        decisions[name] is not True for name in H8_REQUIRED_PASS_DECISIONS
    ):
        raise ValueError("PASS requires every resource decision to be true")
    _validate_residual_evidence(
        decisions.get("residual_allowances"),
        residuals=residuals,
    )
    allocation = result["allocation"]
    typed_allocation = _validate_allocation_evidence(allocation, mode=mode)
    dispatch_events = allocation.get("dispatch_events")
    dispatch_cross_check = allocation.get("dispatch_cross_check")
    numpy_guard_events = allocation.get("numpy_guard_events")
    if (
        type(dispatch_events) is not list
        or not dispatch_events
        or not isinstance(dispatch_cross_check, Mapping)
        or dispatch_cross_check.get("complete") is not True
        or type(dispatch_cross_check.get("reconciled_operation_counts")) is not list
        or not dispatch_cross_check["reconciled_operation_counts"]
        or (
            type(numpy_guard_events) is not list
            or not numpy_guard_events
        )
    ):
        raise ValueError("PASS requires complete allocation event evidence")
    _validate_numpy_preflight_events(numpy_guard_events)
    _validate_pass_relationships(
        result,
        typed=typed,
        mode=mode,
    )
    if mode == "profiler":
        profiler_events = allocation.get("profiler_events")
        if (
            type(profiler_events) is not list
            or not profiler_events
            or allocation.get("profiler_all_joined_and_liveness_reconciled") is not True
        ):
            raise ValueError("profiler PASS requires joined raw profiler evidence")
    return H8ChildResult(
        mode=mode,  # type: ignore[arg-type]
        seed=seed,  # type: ignore[arg-type]
        repetition=repetition,  # type: ignore[arg-type]
        input_sha256=result["input_sha256"],  # type: ignore[arg-type]
        objective=typed["objective"],  # type: ignore[arg-type]
        storage=typed["storage"],  # type: ignore[arg-type]
        fill=typed["fill"],  # type: ignore[arg-type]
        workspace=typed["workspace"],  # type: ignore[arg-type]
        counters=typed["counters"],  # type: ignore[arg-type]
        allocation=typed_allocation,
        resources=typed["resources"],  # type: ignore[arg-type]
        invariants=typed["invariants"],  # type: ignore[arg-type]
    )


def _validate_allocation_evidence(
    value: object,
    *,
    mode: object,
) -> H8AllocationRecord:
    base_keys = (
        "dispatch_trace_sha256",
        "dispatch_event_count",
        "dispatch_events",
        "dispatch_scope_witnesses",
        "dispatch_cross_check",
        "dispatch_forbidden_attempt_count",
        "dispatch_live_peak_bytes",
        "torch_population_peak_bytes",
        "profiler_trace_sha256",
        "profiler_events",
        "profiler_lossy_rows",
        "preexisting_storage_count",
        "preexisting_bytes",
        "baseline_live_bytes",
        "profiler_reconstructed_live_peak_bytes",
        "profiler_all_joined_and_liveness_reconciled",
        "numpy_guard_event_count",
        "numpy_guard_events",
        "numpy_inventory",
        "numpy_inventory_sha256",
        "backend_forbidden_attempt_count",
        "observed_channels",
    )
    allocation = _exact_mapping(
        value,
        base_keys + (("profiler_api",) if mode == "profiler" else ()),
        name="allocation",
    )
    dispatch_values = _exact_list(
        allocation["dispatch_events"],
        name="allocation.dispatch_events",
    )
    dispatch_events = tuple(
        _decode_dispatch_control_event(
            item,
            name=f"allocation.dispatch_events[{index}]",
        )
        for index, item in enumerate(dispatch_values)
    )
    _replay_dispatch_events(dispatch_events, allocation=allocation)
    if (
        not _is_sha256(allocation["dispatch_trace_sha256"])
        or type(allocation["dispatch_event_count"]) is not int
        or allocation["dispatch_event_count"] != len(dispatch_events)
        or not dispatch_events
        or any(
            event.sequence != index
            or event.control_id is not None
            or event.forbidden_reason is not None
            for index, event in enumerate(dispatch_events)
        )
    ):
        raise ValueError("PASS dispatch event inventory is inconsistent")
    cross_check = _exact_mapping(
        allocation["dispatch_cross_check"],
        (
            "complete",
            "obligations",
            "backend_forbidden_attempt_count",
            "dispatch_forbidden_attempt_count",
            "reconciled_operation_counts",
        ),
        name="allocation.dispatch_cross_check",
    )
    reconciled = _exact_list(
        cross_check["reconciled_operation_counts"],
        name="allocation.dispatch_cross_check.reconciled_operation_counts",
    )
    if (
        cross_check["complete"] is not True
        or cross_check["obligations"] != []
        or not reconciled
        or any(
            type(item) is not list
            or len(item) != 3
            or type(item[0]) is not str
            or not item[0]
            or type(item[1]) is not int
            or item[1] <= 0
            or item[1] != item[2]
            for item in reconciled
        )
    ):
        raise ValueError("PASS dispatch/backend cross-check is inconsistent")
    scopes = _exact_list(
        allocation["dispatch_scope_witnesses"],
        name="allocation.dispatch_scope_witnesses",
    )
    observed_scope_names: set[str] = set()
    for index, item in enumerate(scopes):
        scope = _exact_mapping(
            item,
            (
                "semantic_scope",
                "event_start",
                "event_end",
                "event_count",
                "callback_completed",
            ),
            name=f"allocation.dispatch_scope_witnesses[{index}]",
        )
        if (
            type(scope["semantic_scope"]) is not str
            or not scope["semantic_scope"]
            or type(scope["event_start"]) is not int
            or type(scope["event_end"]) is not int
            or type(scope["event_count"]) is not int
            or scope["event_start"] < 0  # type: ignore[operator]
            or scope["event_end"] < scope["event_start"]  # type: ignore[operator]
            or scope["event_count"]  # type: ignore[operator]
            != scope["event_end"] - scope["event_start"]  # type: ignore[operator]
            or scope["event_count"] <= 0  # type: ignore[operator]
            or scope["event_end"] > len(dispatch_events)  # type: ignore[operator]
            or scope["callback_completed"] is not True
        ):
            raise ValueError("PASS dispatch scope witness is inconsistent")
        observed_scope_names.add(scope["semantic_scope"])  # type: ignore[arg-type]
    expected_scope_names = (
        set(H8_OPERATION_SCOPES.values()) | set(H8_SETUP_SCOPES)
    )
    if (
        observed_scope_names != expected_scope_names
        or len(scopes) != len(expected_scope_names)
    ):
        raise ValueError("PASS operation-to-scope inventory is not exact")
    inventory = _exact_list(
        allocation["numpy_inventory"],
        name="allocation.numpy_inventory",
    )
    for index, item in enumerate(inventory):
        entry = _exact_mapping(
            item,
            ("site", "shape", "dtype", "nbytes", "sha256"),
            name=f"allocation.numpy_inventory[{index}]",
        )
        if (
            type(entry["site"]) is not str
            or not entry["site"]
            or any(
                type(dimension) is not int or dimension < 0
                for dimension in _exact_list(
                    entry["shape"],
                    name=f"allocation.numpy_inventory[{index}].shape",
                )
            )
            or type(entry["dtype"]) is not str
            or not entry["dtype"]
            or type(entry["nbytes"]) is not int
            or entry["nbytes"] <= 0  # type: ignore[operator]
            or not _is_sha256(entry["sha256"])
        ):
            raise ValueError("PASS NumPy inventory entry is invalid")
    if (
        not inventory
        or allocation["numpy_inventory_sha256"]
        != hashlib.sha256(canonical_json_bytes(inventory)).hexdigest()
        or allocation["numpy_guard_event_count"]
        != len(_exact_list(
            allocation["numpy_guard_events"],
            name="allocation.numpy_guard_events",
        ))
        or allocation["dispatch_forbidden_attempt_count"] != 0
        or allocation["backend_forbidden_attempt_count"] != 0
        or allocation["torch_population_peak_bytes"]
        > allocation["dispatch_live_peak_bytes"]  # type: ignore[operator]
    ):
        raise ValueError("PASS allocation endpoint relationships are inconsistent")
    expected_channels = (
        ["dispatch", "profiler", "numpy_guard", "backend", "os_hwm"]
        if mode == "profiler"
        else ["dispatch", "numpy_guard", "backend", "os_hwm"]
    )
    if allocation["observed_channels"] != expected_channels:
        raise ValueError("PASS allocation channel inventory is inconsistent")
    profiler_fields = (
        "preexisting_storage_count",
        "preexisting_bytes",
        "baseline_live_bytes",
        "profiler_reconstructed_live_peak_bytes",
    )
    profiler_records: tuple[H8ProfilerEventRecord, ...] = ()
    if mode == "profiler":
        api = _exact_mapping(
            allocation["profiler_api"],
            (
                "torch_version",
                "memory_profile_source_sha256",
                "profiler_source_sha256",
                "api_contract_sha256",
            ),
            name="allocation.profiler_api",
        )
        profiler_events = _exact_list(
            allocation["profiler_events"],
            name="allocation.profiler_events",
        )
        if (
            api["torch_version"] != "2.9.1"
            or any(not _is_sha256(api[name]) for name in tuple(api)[1:])
            or not _is_sha256(allocation["profiler_trace_sha256"])
            or not profiler_events
            or allocation["profiler_lossy_rows"] != []
            or allocation["profiler_all_joined_and_liveness_reconciled"]
            is not True
            or any(
                type(allocation[name]) is not int
                or allocation[name] < 0  # type: ignore[operator]
                for name in profiler_fields
            )
        ):
            raise ValueError("PASS profiler endpoint inventory is inconsistent")
        profiler_records = _decode_profiler_events(profiler_events)
        _replay_profiler_events(
            profiler_records,
            allocation=allocation,
        )
    elif (
        allocation["profiler_trace_sha256"] is not None
        or allocation["profiler_events"] != []
        or allocation["profiler_lossy_rows"] != []
        or any(allocation[name] is not None for name in profiler_fields)
        or allocation["profiler_all_joined_and_liveness_reconciled"] is not None
    ):
        raise ValueError("production PASS cannot contain profiler endpoints")
    return H8AllocationRecord(
        dispatch_trace_sha256=allocation["dispatch_trace_sha256"],  # type: ignore[arg-type]
        dispatch_event_count=allocation["dispatch_event_count"],  # type: ignore[arg-type]
        dispatch_forbidden_attempt_count=allocation[  # type: ignore[arg-type]
            "dispatch_forbidden_attempt_count"
        ],
        dispatch_live_peak_bytes=allocation["dispatch_live_peak_bytes"],  # type: ignore[arg-type]
        torch_population_peak_bytes=allocation[  # type: ignore[arg-type]
            "torch_population_peak_bytes"
        ],
        profiler_trace_sha256=allocation["profiler_trace_sha256"],  # type: ignore[arg-type]
        profiler_events=profiler_records,
        profiler_lossy_rows=(),
        preexisting_storage_count=allocation["preexisting_storage_count"],  # type: ignore[arg-type]
        preexisting_bytes=allocation["preexisting_bytes"],  # type: ignore[arg-type]
        baseline_live_bytes=allocation["baseline_live_bytes"],  # type: ignore[arg-type]
        profiler_reconstructed_live_peak_bytes=allocation[  # type: ignore[arg-type]
            "profiler_reconstructed_live_peak_bytes"
        ],
        profiler_all_joined_and_liveness_reconciled=allocation[  # type: ignore[arg-type]
            "profiler_all_joined_and_liveness_reconciled"
        ],
        numpy_guard_event_count=allocation["numpy_guard_event_count"],  # type: ignore[arg-type]
        backend_forbidden_attempt_count=allocation[  # type: ignore[arg-type]
            "backend_forbidden_attempt_count"
        ],
        observed_channels=tuple(
            _exact_list(
                allocation["observed_channels"],
                name="allocation.observed_channels",
            )
        ),
    )


def _task5_canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _decode_profiler_events(
    events: list[object],
) -> tuple[H8ProfilerEventRecord, ...]:
    keys = (
        "source_row_index",
        "timestamp_ns",
        "action",
        "tensor_key",
        "version",
        "nbytes",
        "dtype",
        "device",
        "operator",
        "stack",
        "logical_shape",
        "classification",
        "matched_event_node_indices",
        "join_witness_sha256",
        "live_bytes_after",
    )
    records: list[H8ProfilerEventRecord] = []
    for index, item in enumerate(events):
        event = _exact_mapping(
            item,
            keys,
            name=f"allocation.profiler_events[{index}]",
        )
        action = event["action"]
        key = _exact_mapping(
            event["tensor_key"],
            ("tensor_id", "storage_ptr", "allocation_id", "device"),
            name=f"allocation.profiler_events[{index}].tensor_key",
        )
        stack = tuple(
            _exact_list(
                event["stack"],
                name=f"allocation.profiler_events[{index}].stack",
            )
        )
        logical_shape = tuple(
            _exact_list(
                event["logical_shape"],
                name=f"allocation.profiler_events[{index}].logical_shape",
            )
        )
        matched = tuple(
            _exact_list(
                event["matched_event_node_indices"],
                name=(
                    f"allocation.profiler_events[{index}]"
                    ".matched_event_node_indices"
                ),
            )
        )
        if (
            type(event["source_row_index"]) is not int
            or event["source_row_index"] < 0  # type: ignore[operator]
            or type(event["timestamp_ns"]) is not int
            or event["timestamp_ns"] < -1  # type: ignore[operator]
            or (
                event["timestamp_ns"] == -1
                and action != "PREEXISTING"
            )
            or action
            not in ("PREEXISTING", "CREATE", "INCREMENT_VERSION", "DESTROY")
            or any(
                type(key[name]) is not int or key[name] < 0  # type: ignore[operator]
                for name in ("tensor_id", "storage_ptr", "allocation_id")
            )
            or type(key["device"]) is not str
            or not key["device"]
            or type(event["version"]) is not int
            or event["version"] < 0  # type: ignore[operator]
            or type(event["nbytes"]) is not int
            or any(
                type(event[name]) is not str or not event[name]
                for name in (
                    "dtype",
                    "device",
                    "operator",
                    "classification",
                )
            )
            or not _is_sha256(event["join_witness_sha256"])
            or type(event["live_bytes_after"]) is not int
            or event["live_bytes_after"] < 0  # type: ignore[operator]
            or any(type(frame) is not str or not frame for frame in stack)
            or not stack
            or any(
                type(dimension) is not int or dimension < 0
                for dimension in logical_shape
            )
            or not matched
            or len(set(matched)) != len(matched)
            or any(type(node) is not int or node < 0 for node in matched)
        ):
            raise ValueError("PASS profiler event schema is invalid")
        try:
            tensor_key = H8TensorKey(
                tensor_id=key["tensor_id"],  # type: ignore[arg-type]
                storage_ptr=key["storage_ptr"],  # type: ignore[arg-type]
                allocation_id=key["allocation_id"],  # type: ignore[arg-type]
                device=key["device"],  # type: ignore[arg-type]
            )
            records.append(
                H8ProfilerEventRecord(
                    source_row_index=event["source_row_index"],  # type: ignore[arg-type]
                    timestamp_ns=event["timestamp_ns"],  # type: ignore[arg-type]
                    action=action,  # type: ignore[arg-type]
                    tensor_key=tensor_key,
                    version=event["version"],  # type: ignore[arg-type]
                    nbytes=event["nbytes"],  # type: ignore[arg-type]
                    dtype=event["dtype"],  # type: ignore[arg-type]
                    device=event["device"],  # type: ignore[arg-type]
                    operator=event["operator"],  # type: ignore[arg-type]
                    stack=stack,
                    logical_shape=logical_shape,
                    classification=event["classification"],  # type: ignore[arg-type]
                    matched_event_node_indices=matched,
                    join_witness_sha256=event["join_witness_sha256"],  # type: ignore[arg-type]
                    live_bytes_after=event["live_bytes_after"],  # type: ignore[arg-type]
                )
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"PASS profiler event reconstruction failed: {error}"
            ) from error
    return tuple(records)


def _replay_profiler_events(
    records: tuple[H8ProfilerEventRecord, ...],
    *,
    allocation: Mapping[str, object],
) -> None:
    if (
        not records
        or len({record.source_row_index for record in records}) != len(records)
        or tuple(records)
        != tuple(
            sorted(
                records,
                key=lambda record: (
                    record.timestamp_ns,
                    record.source_row_index,
                ),
            )
        )
    ):
        raise ValueError("PASS profiler source-row ordering is invalid")
    live: dict[H8TensorKey, tuple[int, int, bool]] = {}
    storage_members: dict[tuple[int, str], set[H8TensorKey]] = {}
    storage_sizes: dict[tuple[int, str], int] = {}
    storage_pointers: dict[tuple[int, str], int] = {}
    established: set[H8TensorKey] = set()
    established_storage: set[tuple[int, str]] = set()
    baseline_storage: set[tuple[int, str]] = set()
    baseline_bytes = 0
    live_bytes = 0
    peak_bytes = 0
    saw_nonpreexisting = False
    for record in records:
        if record.device != record.tensor_key.device or record.device != "cpu":
            raise ValueError("PASS profiler device identity is inconsistent")
        storage_id = (
            record.tensor_key.allocation_id,
            record.tensor_key.device,
        )
        members = storage_members.get(storage_id)
        if record.action == "PREEXISTING":
            if saw_nonpreexisting or record.version != 0 or record.nbytes <= 0:
                raise ValueError("PASS profiler baseline ordering is invalid")
            baseline = True
        elif record.action == "CREATE":
            saw_nonpreexisting = True
            if record.version != 0 or record.nbytes <= 0:
                raise ValueError("PASS profiler CREATE transition is invalid")
            baseline = False
        elif record.action == "INCREMENT_VERSION":
            saw_nonpreexisting = True
            state = live.get(record.tensor_key)
            if (
                state is None
                or record.nbytes != 0
                or record.version != state[0] + 1
            ):
                raise ValueError("PASS profiler version transition is invalid")
            live[record.tensor_key] = (record.version, state[1], state[2])
            if record.live_bytes_after != live_bytes:
                raise ValueError("PASS profiler live-byte endpoint drifted")
            peak_bytes = max(peak_bytes, live_bytes)
            continue
        else:
            saw_nonpreexisting = True
            state = live.get(record.tensor_key)
            if (
                state is None
                or record.version != state[0]
                or record.nbytes != -state[1]
                or members is None
                or record.tensor_key not in members
            ):
                raise ValueError("PASS profiler DESTROY transition is invalid")
            del live[record.tensor_key]
            members.remove(record.tensor_key)
            if not members:
                storage_members.pop(storage_id)
                live_bytes -= storage_sizes[storage_id]
            if record.live_bytes_after != live_bytes:
                raise ValueError("PASS profiler live-byte endpoint drifted")
            peak_bytes = max(peak_bytes, live_bytes)
            continue
        if record.tensor_key in established:
            raise ValueError("PASS profiler identity is established twice")
        if members is None:
            if storage_id in established_storage:
                raise ValueError("PASS profiler storage identity is established twice")
            members = set()
            storage_members[storage_id] = members
            storage_sizes[storage_id] = record.nbytes
            storage_pointers[storage_id] = record.tensor_key.storage_ptr
            established_storage.add(storage_id)
            live_bytes += record.nbytes
            if baseline:
                baseline_storage.add(storage_id)
                baseline_bytes += record.nbytes
        elif (
            storage_sizes[storage_id] != record.nbytes
            or storage_pointers[storage_id] != record.tensor_key.storage_ptr
        ):
            raise ValueError("PASS profiler alias storage metadata drifted")
        members.add(record.tensor_key)
        established.add(record.tensor_key)
        live[record.tensor_key] = (
            record.version,
            record.nbytes,
            baseline,
        )
        if record.live_bytes_after != live_bytes:
            raise ValueError("PASS profiler live-byte endpoint drifted")
        peak_bytes = max(peak_bytes, live_bytes)
    if any(not state[2] for state in live.values()):
        raise ValueError("PASS profiler replay retains created storage")
    if (
        allocation["preexisting_storage_count"] != len(baseline_storage)
        or allocation["preexisting_bytes"] != baseline_bytes
        or allocation["baseline_live_bytes"] != baseline_bytes
        or allocation["profiler_reconstructed_live_peak_bytes"] != peak_bytes
        or allocation["profiler_trace_sha256"]
        != _task5_canonical_sha256(list(records))
    ):
        raise ValueError("PASS profiler replay endpoints or trace digest drifted")


def _exact_mapping(
    value: object,
    keys: tuple[str, ...],
    *,
    name: str,
) -> dict[str, object]:
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(f"{name} does not use its exact key set")
    return value


def _exact_list(value: object, *, name: str) -> list[object]:
    if type(value) is not list:
        raise ValueError(f"{name} must be a JSON list")
    return value


def _decode_layout(value: object, *, name: str) -> BlockChainLayout:
    record = _exact_mapping(
        value,
        ("horizon", "d_z", "d_m"),
        name=name,
    )
    return BlockChainLayout(
        horizon=record["horizon"],  # type: ignore[arg-type]
        d_z=record["d_z"],  # type: ignore[arg-type]
        d_m=record["d_m"],  # type: ignore[arg-type]
    )


def _decode_block_id(value: object, *, name: str) -> BlockId:
    record = _exact_mapping(
        value,
        ("kind", "row", "column"),
        name=name,
    )
    return BlockId(
        kind=record["kind"],  # type: ignore[arg-type]
        row=record["row"],  # type: ignore[arg-type]
        column=record["column"],  # type: ignore[arg-type]
    )


def _decode_objective_term(value: object, *, name: str) -> H8ObjectiveTerm:
    record = _exact_mapping(
        value,
        (
            "factor_id",
            "role",
            "receiver_t",
            "value",
            "absolute_sum_bound",
        ),
        name=name,
    )
    return H8ObjectiveTerm(
        factor_id=record["factor_id"],  # type: ignore[arg-type]
        role=record["role"],  # type: ignore[arg-type]
        receiver_t=record["receiver_t"],  # type: ignore[arg-type]
        value=record["value"],  # type: ignore[arg-type]
        absolute_sum_bound=record["absolute_sum_bound"],  # type: ignore[arg-type]
    )


def _decode_objective(value: object) -> H8ObjectiveTerms:
    record = _exact_mapping(
        value,
        (
            "horizon",
            "initial_joint",
            "model_transitions",
            "state_transitions",
            "emissions_order21",
            "emissions_order17",
            "recognition_entropy",
            "log_normalizer",
            "model_source_kl",
            "state_source_kl",
            "source_entropy",
            "quadrature_absolute_difference",
            "complete_order21",
            "absolute_term_sum",
        ),
        name="objective",
    )

    def series(field: str) -> tuple[H8ObjectiveTerm, ...]:
        return tuple(
            _decode_objective_term(
                item,
                name=f"objective.{field}[{index}]",
            )
            for index, item in enumerate(
                _exact_list(record[field], name=f"objective.{field}")
            )
        )

    return H8ObjectiveTerms(
        horizon=record["horizon"],  # type: ignore[arg-type]
        initial_joint=_decode_objective_term(
            record["initial_joint"],
            name="objective.initial_joint",
        ),
        model_transitions=series("model_transitions"),
        state_transitions=series("state_transitions"),
        emissions_order21=series("emissions_order21"),
        emissions_order17=series("emissions_order17"),
        recognition_entropy=record["recognition_entropy"],  # type: ignore[arg-type]
        log_normalizer=record["log_normalizer"],  # type: ignore[arg-type]
        model_source_kl=record["model_source_kl"],  # type: ignore[arg-type]
        state_source_kl=record["state_source_kl"],  # type: ignore[arg-type]
        source_entropy=record["source_entropy"],  # type: ignore[arg-type]
        quadrature_absolute_difference=record[  # type: ignore[arg-type]
            "quadrature_absolute_difference"
        ],
        complete_order21=record["complete_order21"],  # type: ignore[arg-type]
        absolute_term_sum=record["absolute_term_sum"],  # type: ignore[arg-type]
    )


def _decode_storage(value: object) -> BlockStorageRecord:
    record = _exact_mapping(
        value,
        (
            "layout",
            "precision_scalar_count",
            "factor_scalar_count",
            "selected_inverse_scalar_count",
            "information_scalar_count",
            "upper_block_scalar_count",
        ),
        name="storage",
    )
    return BlockStorageRecord(
        layout=_decode_layout(record["layout"], name="storage.layout"),
        precision_scalar_count=record["precision_scalar_count"],  # type: ignore[arg-type]
        factor_scalar_count=record["factor_scalar_count"],  # type: ignore[arg-type]
        selected_inverse_scalar_count=record[  # type: ignore[arg-type]
            "selected_inverse_scalar_count"
        ],
        information_scalar_count=record["information_scalar_count"],  # type: ignore[arg-type]
        upper_block_scalar_count=record["upper_block_scalar_count"],  # type: ignore[arg-type]
    )


def _decode_fill(value: object) -> BlockFillRecord:
    record = _exact_mapping(
        value,
        (
            "layout",
            "stored_block_ids",
            "observed_offband_blocks",
            "duplicated_upper_blocks",
        ),
        name="fill",
    )
    blocks = tuple(
        _decode_block_id(item, name=f"fill.stored_block_ids[{index}]")
        for index, item in enumerate(
            _exact_list(
                record["stored_block_ids"],
                name="fill.stored_block_ids",
            )
        )
    )
    return BlockFillRecord(
        layout=_decode_layout(record["layout"], name="fill.layout"),
        stored_block_ids=blocks,
        observed_offband_blocks=record["observed_offband_blocks"],  # type: ignore[arg-type]
        duplicated_upper_blocks=record["duplicated_upper_blocks"],  # type: ignore[arg-type]
    )


def _decode_workspace(value: object) -> BlockWorkspaceRecord:
    record = _exact_mapping(
        value,
        (
            "maximum_shape",
            "maximum_scalar_count",
            "maximum_rhs_width",
            "attempted_forbidden_rhs_widths",
        ),
        name="workspace",
    )
    return BlockWorkspaceRecord(
        maximum_shape=tuple(
            _exact_list(
                record["maximum_shape"],
                name="workspace.maximum_shape",
            )
        ),
        maximum_scalar_count=record["maximum_scalar_count"],  # type: ignore[arg-type]
        maximum_rhs_width=record["maximum_rhs_width"],  # type: ignore[arg-type]
        attempted_forbidden_rhs_widths=tuple(
            _exact_list(
                record["attempted_forbidden_rhs_widths"],
                name="workspace.attempted_forbidden_rhs_widths",
            )
        ),
    )


def _decode_counters(value: object) -> BackendCounterSnapshot:
    record = _exact_mapping(
        value,
        (
            "layout",
            "factorization_calls",
            "forward_substitution_calls",
            "backward_substitution_calls",
            "solve_calls",
            "logdet_calls",
            "selected_inverse_calls",
            "sample_calls",
            "quadratic_calls",
            "trace_calls",
            "sparse_matvec_calls",
            "maximum_rhs_width",
            "maximum_sample_rhs_width",
            "selected_block_ids",
            "selected_block_count",
            "attempted_forbidden_selected_blocks",
            "attempted_forbidden_rhs_widths",
        ),
        name="counters",
    )
    blocks = tuple(
        _decode_block_id(item, name=f"counters.selected_block_ids[{index}]")
        for index, item in enumerate(
            _exact_list(
                record["selected_block_ids"],
                name="counters.selected_block_ids",
            )
        )
    )
    return BackendCounterSnapshot(
        layout=_decode_layout(record["layout"], name="counters.layout"),
        factorization_calls=record["factorization_calls"],  # type: ignore[arg-type]
        forward_substitution_calls=record[  # type: ignore[arg-type]
            "forward_substitution_calls"
        ],
        backward_substitution_calls=record[  # type: ignore[arg-type]
            "backward_substitution_calls"
        ],
        solve_calls=record["solve_calls"],  # type: ignore[arg-type]
        logdet_calls=record["logdet_calls"],  # type: ignore[arg-type]
        selected_inverse_calls=record["selected_inverse_calls"],  # type: ignore[arg-type]
        sample_calls=record["sample_calls"],  # type: ignore[arg-type]
        quadratic_calls=record["quadratic_calls"],  # type: ignore[arg-type]
        trace_calls=record["trace_calls"],  # type: ignore[arg-type]
        sparse_matvec_calls=record["sparse_matvec_calls"],  # type: ignore[arg-type]
        maximum_rhs_width=record["maximum_rhs_width"],  # type: ignore[arg-type]
        maximum_sample_rhs_width=record[  # type: ignore[arg-type]
            "maximum_sample_rhs_width"
        ],
        selected_block_ids=blocks,
        selected_block_count=record["selected_block_count"],  # type: ignore[arg-type]
        attempted_forbidden_selected_blocks=record[  # type: ignore[arg-type]
            "attempted_forbidden_selected_blocks"
        ],
        attempted_forbidden_rhs_widths=tuple(
            _exact_list(
                record["attempted_forbidden_rhs_widths"],
                name="counters.attempted_forbidden_rhs_widths",
            )
        ),
    )


def _decode_resources(value: object) -> H8ResourceRecord:
    keys = (
        "adapter",
        "adapter_sha256",
        "pre_current_rss_bytes",
        "pre_lifetime_peak_bytes",
        "pre_private_bytes",
        "post_current_rss_bytes",
        "post_lifetime_peak_bytes",
        "post_private_bytes",
        "conservative_incremental_hwm_bytes",
        "peak_to_peak_diagnostic_bytes",
        "parent_elapsed_ns",
        "child_elapsed_ns",
    )
    record = _exact_mapping(value, keys, name="resources")
    return H8ResourceRecord(**record)  # type: ignore[arg-type]


def _decode_diagnostics(value: object) -> SparseConditionDiagnostics:
    record = _exact_mapping(
        value,
        (
            "estimator",
            "kappa_1_estimate",
            "iterations",
            "convergence_reason",
            "index_sha256",
            "sign_sha256",
            "per_block_min_pivots",
            "global_min_pivot",
            "per_block_pivot_margins",
            "global_pivot_margin",
        ),
        name="diagnostics",
    )
    return SparseConditionDiagnostics(
        estimator=record["estimator"],  # type: ignore[arg-type]
        kappa_1_estimate=record["kappa_1_estimate"],  # type: ignore[arg-type]
        iterations=record["iterations"],  # type: ignore[arg-type]
        convergence_reason=record["convergence_reason"],  # type: ignore[arg-type]
        index_sha256=record["index_sha256"],  # type: ignore[arg-type]
        sign_sha256=record["sign_sha256"],  # type: ignore[arg-type]
        per_block_min_pivots=tuple(
            _exact_list(
                record["per_block_min_pivots"],
                name="diagnostics.per_block_min_pivots",
            )
        ),
        global_min_pivot=record["global_min_pivot"],  # type: ignore[arg-type]
        per_block_pivot_margins=tuple(
            _exact_list(
                record["per_block_pivot_margins"],
                name="diagnostics.per_block_pivot_margins",
            )
        ),
        global_pivot_margin=record["global_pivot_margin"],  # type: ignore[arg-type]
    )


def _decode_invariants(value: object) -> tuple[H8InvariantRecord, ...]:
    records: list[H8InvariantRecord] = []
    for index, item in enumerate(_exact_list(value, name="invariants")):
        record = _exact_mapping(
            item,
            (
                "invariant_id",
                "status",
                "value",
                "limit",
                "detail",
                "obligations",
            ),
            name=f"invariants[{index}]",
        )
        try:
            status = GateStatus(record["status"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"invariants[{index}] status is invalid") from error
        records.append(
            H8InvariantRecord(
                invariant_id=record["invariant_id"],  # type: ignore[arg-type]
                status=status,
                value=record["value"],  # type: ignore[arg-type]
                limit=record["limit"],  # type: ignore[arg-type]
                detail=record["detail"],  # type: ignore[arg-type]
                obligations=tuple(
                    _exact_list(
                        record["obligations"],
                        name=f"invariants[{index}].obligations",
                    )
                ),
            )
        )
    return tuple(records)


def _validate_typed_pass_sections(
    result: Mapping[str, object],
) -> dict[str, object]:
    decoders = {
        "objective": _decode_objective,
        "storage": _decode_storage,
        "fill": _decode_fill,
        "workspace": _decode_workspace,
        "counters": _decode_counters,
        "resources": _decode_resources,
        "diagnostics": _decode_diagnostics,
        "invariants": _decode_invariants,
    }
    decoded: dict[str, object] = {}
    for name, decoder in decoders.items():
        try:
            decoded[name] = decoder(result[name])
        except (TypeError, ValueError) as error:
            raise ValueError(f"PASS {name} evidence is invalid: {error}") from error
    return decoded


def _decode_operand(value: object, *, name: str) -> H8OperandRecord:
    record = _exact_mapping(
        value,
        (
            "operand_id",
            "shape",
            "scalar_count",
            "infinity_norm",
            "absolute_sum_bound",
            "local_operation_count",
            "source",
            "condition_provenance",
            "solver_produced",
            "quadrature_convergence",
        ),
        name=name,
    )
    return H8OperandRecord(
        operand_id=record["operand_id"],  # type: ignore[arg-type]
        shape=tuple(_exact_list(record["shape"], name=f"{name}.shape")),
        scalar_count=record["scalar_count"],  # type: ignore[arg-type]
        infinity_norm=record["infinity_norm"],  # type: ignore[arg-type]
        absolute_sum_bound=record["absolute_sum_bound"],  # type: ignore[arg-type]
        local_operation_count=record["local_operation_count"],  # type: ignore[arg-type]
        source=record["source"],  # type: ignore[arg-type]
        condition_provenance=record["condition_provenance"],  # type: ignore[arg-type]
        solver_produced=record["solver_produced"],  # type: ignore[arg-type]
        quadrature_convergence=record["quadrature_convergence"],  # type: ignore[arg-type]
    )


def _decode_float64_endpoint(
    value: object,
    *,
    name: str,
) -> tuple[tuple[int, ...], bytes, float, float]:
    record = _exact_mapping(
        value,
        (
            "encoding",
            "shape",
            "scalar_count",
            "raw_nbytes",
            "raw_sha256",
            "compressed_nbytes",
            "payload_b64",
        ),
        name=name,
    )
    shape = tuple(_exact_list(record["shape"], name=f"{name}.shape"))
    if (
        record["encoding"] != "float64-le-zlib-base64-v1"
        or not shape
        or any(type(dimension) is not int or dimension <= 0 for dimension in shape)
        or type(record["scalar_count"]) is not int
        or record["scalar_count"] != math.prod(shape)
        or record["scalar_count"]
        > H8_SCALE_LAYOUT.population_size * H8_SCALE_LAYOUT.block_size**2
        or type(record["raw_nbytes"]) is not int
        or record["raw_nbytes"] != record["scalar_count"] * 8
        or type(record["compressed_nbytes"]) is not int
        or record["compressed_nbytes"] <= 0
        or type(record["payload_b64"]) is not str
        or not record["payload_b64"]
        or not _is_sha256(record["raw_sha256"])
    ):
        raise ValueError(f"{name} metadata is invalid")
    try:
        compressed = base64.b64decode(record["payload_b64"], validate=True)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{name} base64 payload is invalid") from error
    if (
        len(compressed) != record["compressed_nbytes"]
        or base64.b64encode(compressed).decode("ascii") != record["payload_b64"]
    ):
        raise ValueError(f"{name} compressed extent is inconsistent")
    expected_nbytes = record["raw_nbytes"]
    decoder = zlib.decompressobj()
    try:
        raw = decoder.decompress(compressed, expected_nbytes + 1)
        if decoder.unconsumed_tail or len(raw) > expected_nbytes:
            raise ValueError(f"{name} expands beyond its frozen extent")
        remaining = expected_nbytes + 1 - len(raw)
        if remaining > 0:
            raw += decoder.flush(remaining)
    except zlib.error as error:
        raise ValueError(f"{name} zlib payload is invalid") from error
    if (
        not decoder.eof
        or decoder.unused_data
        or decoder.unconsumed_tail
        or len(raw) != expected_nbytes
        or hashlib.sha256(raw).hexdigest() != record["raw_sha256"]
    ):
        raise ValueError(f"{name} raw endpoint identity is inconsistent")
    absolute_values: list[float] = []
    infinity_norm = 0.0
    for (endpoint,) in struct.iter_unpack("<d", raw):
        if not math.isfinite(endpoint):
            raise ValueError(f"{name} contains a nonfinite endpoint")
        absolute = abs(endpoint)
        infinity_norm = max(infinity_norm, absolute)
        absolute_values.append(absolute)
    return shape, raw, infinity_norm, math.fsum(absolute_values)


def _raw_endpoint_residual(left: bytes, right: bytes, *, name: str) -> float:
    if len(left) != len(right) or not left:
        raise ValueError(f"{name} endpoint extents disagree")
    residual = 0.0
    for (left_value,), (right_value,) in zip(
        struct.iter_unpack("<d", left),
        struct.iter_unpack("<d", right),
        strict=True,
    ):
        difference = abs(left_value - right_value)
        if not math.isfinite(difference):
            raise ValueError(f"{name} residual is nonfinite")
        residual = max(residual, difference)
    return residual


def _decode_allowance(value: object, *, name: str) -> H8AllowanceRecord:
    wrapper = _exact_mapping(
        value,
        ("allowance", "left_endpoint", "right_endpoint"),
        name=name,
    )
    record = _exact_mapping(
        wrapper["allowance"],
        (
            "comparison_id",
            "left",
            "right",
            "compared_scalar_count",
            "left_rounding_component",
            "left_solver_component",
            "left_quadrature_component",
            "right_rounding_component",
            "right_solver_component",
            "right_quadrature_component",
            "reduction_component",
            "allowance",
            "scale",
            "residual",
            "allowance_scale_fraction",
            "decisive",
            "status",
            "obligations",
        ),
        name=name,
    )
    left = _decode_operand(record["left"], name=f"{name}.left")
    right = _decode_operand(record["right"], name=f"{name}.right")
    left_shape, left_raw, left_norm, left_sum = _decode_float64_endpoint(
        wrapper["left_endpoint"],
        name=f"{name}.left_endpoint",
    )
    right_shape, right_raw, right_norm, right_sum = _decode_float64_endpoint(
        wrapper["right_endpoint"],
        name=f"{name}.right_endpoint",
    )
    try:
        status = GateStatus(record["status"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name}.status is invalid") from error
    observed = H8AllowanceRecord(
        comparison_id=record["comparison_id"],  # type: ignore[arg-type]
        left=left,
        right=right,
        compared_scalar_count=record["compared_scalar_count"],  # type: ignore[arg-type]
        left_rounding_component=record["left_rounding_component"],  # type: ignore[arg-type]
        left_solver_component=record["left_solver_component"],  # type: ignore[arg-type]
        left_quadrature_component=record["left_quadrature_component"],  # type: ignore[arg-type]
        right_rounding_component=record["right_rounding_component"],  # type: ignore[arg-type]
        right_solver_component=record["right_solver_component"],  # type: ignore[arg-type]
        right_quadrature_component=record["right_quadrature_component"],  # type: ignore[arg-type]
        reduction_component=record["reduction_component"],  # type: ignore[arg-type]
        allowance=record["allowance"],  # type: ignore[arg-type]
        scale=record["scale"],  # type: ignore[arg-type]
        residual=record["residual"],  # type: ignore[arg-type]
        allowance_scale_fraction=record["allowance_scale_fraction"],  # type: ignore[arg-type]
        decisive=record["decisive"],  # type: ignore[arg-type]
        status=status,
        obligations=tuple(
            _exact_list(record["obligations"], name=f"{name}.obligations")
        ),
    )
    recomputed_left = make_operand_record(
        operand_id=left.operand_id,
        shape=left_shape,
        infinity_norm=left_norm,
        absolute_sum_bound=left_sum,
        local_operation_count=left.local_operation_count,
        source=left.source,
        solver_produced=left.solver_produced,
        quadrature_convergence=left.quadrature_convergence,
        condition_provenance=left.condition_provenance,
    )
    recomputed_right = make_operand_record(
        operand_id=right.operand_id,
        shape=right_shape,
        infinity_norm=right_norm,
        absolute_sum_bound=right_sum,
        local_operation_count=right.local_operation_count,
        source=right.source,
        solver_produced=right.solver_produced,
        quadrature_convergence=right.quadrature_convergence,
        condition_provenance=right.condition_provenance,
    )
    if left != recomputed_left or right != recomputed_right:
        raise ValueError(f"{name} operand statistics do not match raw endpoints")
    expected = compare_operands(
        comparison_id=observed.comparison_id,
        left=recomputed_left,
        right=recomputed_right,
        residual=_raw_endpoint_residual(left_raw, right_raw, name=name),
        compared_scalar_count=len(left_raw) // 8,
    )
    if observed != expected:
        raise ValueError(f"{name} allowance does not match frozen reconstruction")
    return observed


def _validate_residual_evidence(
    value: object,
    *,
    residuals: Mapping[str, object],
) -> None:
    groups = _exact_mapping(
        value,
        tuple(H8_REQUIRED_RESIDUALS),
        name="resource_decisions.residual_allowances",
    )
    for residual_id, specs in H8_SCALE_RESIDUAL_SPECS.items():
        group = _exact_mapping(
            groups[residual_id],
            (
                "residual_id",
                "aggregation",
                "residual",
                "comparisons",
                "decisive",
                "passed",
            ),
            name=f"residual_allowances.{residual_id}",
        )
        comparisons = tuple(
            _decode_allowance(
                item,
                name=f"residual_allowances.{residual_id}.comparisons[{index}]",
            )
            for index, item in enumerate(
                _exact_list(
                    group["comparisons"],
                    name=f"residual_allowances.{residual_id}.comparisons",
                )
            )
        )
        if len(comparisons) != len(specs):
            raise ValueError(f"{residual_id} allowance comparison count is incomplete")
        for comparison, spec in zip(comparisons, specs, strict=True):
            (
                comparison_id,
                left_id,
                right_id,
                shape,
                left_solver,
                right_solver,
            ) = spec
            if (
                comparison.comparison_id != comparison_id
                or comparison.left.operand_id != left_id
                or comparison.right.operand_id != right_id
                or comparison.left.shape != shape
                or comparison.right.shape != shape
                or comparison.left.local_operation_count != shape[-1]
                or comparison.right.local_operation_count != shape[-1]
                or comparison.left.source != "block"
                or comparison.right.source != "block"
                or comparison.left.condition_provenance is not None
                or comparison.right.condition_provenance is not None
                or comparison.left.solver_produced is not left_solver
                or comparison.right.solver_produced is not right_solver
                or comparison.left.quadrature_convergence != 0.0
                or comparison.right.quadrature_convergence != 0.0
            ):
                raise ValueError(
                    f"{residual_id} allowance operand identity is not frozen"
                )
        aggregate = max(item.residual for item in comparisons)
        if (
            group["residual_id"] != residual_id
            or group["aggregation"]
            != "max_residual_all_comparisons_must_pass"
            or type(group["residual"]) is not float
            or group["residual"] != aggregate
            or group["residual"] != residuals[residual_id]
            or group["decisive"] is not all(item.decisive for item in comparisons)
            or group["passed"]
            is not all(item.status is GateStatus.PASS for item in comparisons)
            or group["decisive"] is not True
            or group["passed"] is not True
        ):
            raise ValueError(f"{residual_id} allowance aggregate is inconsistent")


def _validate_numpy_preflight_events(value: object) -> None:
    events = _exact_list(value, name="allocation.numpy_guard_events")
    operators: set[str] = set()
    keys = (
        "sequence",
        "operator",
        "semantic_site",
        "control_id",
        "input_shapes",
        "output_shapes",
        "dtype",
        "float64_equivalent_scalars",
        "executed",
        "forbidden_reason",
    )
    for index, item in enumerate(events):
        event = _exact_mapping(
            item,
            keys,
            name=f"allocation.numpy_guard_events[{index}]",
        )
        if (
            type(event["sequence"]) is not int
            or event["sequence"] != index
            or type(event["operator"]) is not str
            or not event["operator"]
            or type(event["semantic_site"]) is not str
            or not event["semantic_site"]
            or event["control_id"] is not None
            or type(event["input_shapes"]) is not list
            or type(event["output_shapes"]) is not list
            or type(event["float64_equivalent_scalars"]) is not int
            or event["float64_equivalent_scalars"] < 0
            or event["executed"] is not True
            or event["forbidden_reason"] is not None
        ):
            raise ValueError("PASS NumPy event lacks exact preflight evidence")
        operators.add(event["operator"])
    if not H8_REQUIRED_NUMPY_PRODUCERS.issubset(operators):
        raise ValueError("PASS NumPy producer preflight inventory is incomplete")


def _derive_operation_reachability(
    result: Mapping[str, object],
    *,
    counters: BackendCounterSnapshot,
    diagnostics: SparseConditionDiagnostics,
) -> dict[str, bool]:
    allocation = result["allocation"]
    if not isinstance(allocation, Mapping):
        raise ValueError("operation reachability lacks allocation evidence")
    completed_scopes = {
        item["semantic_scope"]
        for item in _exact_list(
            allocation["dispatch_scope_witnesses"],
            name="allocation.dispatch_scope_witnesses",
        )
        if isinstance(item, Mapping)
        and item.get("callback_completed") is True
        and type(item.get("event_count")) is int
        and item["event_count"] > 0  # type: ignore[operator]
    }
    backend = {
        "factorization": counters.factorization_calls > 0,
        "forward_substitution": counters.forward_substitution_calls > 0,
        "backward_substitution": counters.backward_substitution_calls > 0,
        "mean_solve": counters.solve_calls > 0,
        "logdet": counters.logdet_calls > 0,
        "selected_inverse": counters.selected_inverse_calls > 0,
        "sample_width_one": (
            counters.sample_calls > 0
            and counters.maximum_sample_rhs_width == 1
        ),
        "quadratic": counters.quadratic_calls > 0,
        "sparse_trace": counters.trace_calls > 0,
        "condition_estimate": (
            counters.sparse_matvec_calls > 0 and diagnostics.iterations > 0
        ),
        "entropy": True,
        "log_normalizer": True,
        "complete_objective": True,
    }
    return {
        operation: H8_OPERATION_SCOPES[operation] in completed_scopes
        and backend[operation]
        for operation in H8_REQUIRED_OPERATIONS
    }


def _validate_pass_relationships(
    result: Mapping[str, object],
    *,
    typed: Mapping[str, object],
    mode: object,
) -> None:
    objective = typed["objective"]
    storage = typed["storage"]
    fill = typed["fill"]
    workspace = typed["workspace"]
    counters = typed["counters"]
    resources = typed["resources"]
    diagnostics = typed["diagnostics"]
    invariants = typed["invariants"]
    if (
        type(objective) is not H8ObjectiveTerms
        or type(storage) is not BlockStorageRecord
        or type(fill) is not BlockFillRecord
        or type(workspace) is not BlockWorkspaceRecord
        or type(counters) is not BackendCounterSnapshot
        or type(resources) is not H8ResourceRecord
        or type(diagnostics) is not SparseConditionDiagnostics
        or type(invariants) is not tuple
    ):
        raise ValueError("PASS typed result reconstruction is incomplete")
    if (
        objective.horizon != H8_SCALE_LAYOUT.horizon
        or storage.layout != H8_SCALE_LAYOUT
        or fill.layout != H8_SCALE_LAYOUT
        or counters.layout != H8_SCALE_LAYOUT
        or not storage.matches_expectation
        or not fill.matches_expected_fill
        or workspace.maximum_shape
        != (H8_SCALE_LAYOUT.block_size, H8_SCALE_LAYOUT.block_size)
        or workspace.maximum_scalar_count
        != H8_SCALE_LAYOUT.block_size**2
        or workspace.maximum_rhs_width != H8_SCALE_LAYOUT.block_size
        or workspace.attempted_forbidden_rhs_widths
        or counters.factorization_calls != 1
        or counters.forward_substitution_calls < counters.solve_calls + 1
        or counters.backward_substitution_calls < counters.solve_calls + 2
        or any(
            getattr(counters, name) <= 0
            for name in (
                "solve_calls",
                "logdet_calls",
                "selected_inverse_calls",
                "sample_calls",
                "quadratic_calls",
                "trace_calls",
                "sparse_matvec_calls",
            )
        )
        or counters.selected_inverse_calls < 2
        or counters.maximum_rhs_width != H8_SCALE_LAYOUT.block_size
        or counters.maximum_sample_rhs_width != 1
        or not counters.selected_coverage_complete
        or counters.attempted_forbidden_selected_blocks != 0
        or counters.attempted_forbidden_rhs_widths
        or len(diagnostics.per_block_min_pivots)
        != H8_SCALE_LAYOUT.population_size
        or diagnostics.global_pivot_margin < 0.0
        or resources.child_elapsed_ns > int(H8_MAX_SECONDS * 1e9)
        or resources.conservative_incremental_hwm_bytes
        > H8_MAX_PROCESS_INCREMENTAL_BYTES
    ):
        raise ValueError("PASS typed endpoint relationships are inconsistent")
    observed_reachability = result["operation_reachability"]
    derived_reachability = _derive_operation_reachability(
        result,
        counters=counters,
        diagnostics=diagnostics,
    )
    if (
        type(observed_reachability) is not dict
        or set(observed_reachability) != set(H8_REQUIRED_OPERATIONS)
        or observed_reachability != derived_reachability
        or not all(derived_reachability.values())
    ):
        raise ValueError(
            "PASS operation reachability does not derive from scopes and counters"
        )
    decisions = _exact_mapping(
        result["resource_decisions"],
        (
            *H8_REQUIRED_PASS_DECISIONS,
            "residual_allowances",
            "dispatch_backend_cross_check_obligations",
            "conservative_incremental_hwm_bytes",
            "torch_population_peak_bytes",
            *(
                (
                    "profiler_join_pass",
                    "profiler_reconstructed_live_peak_bytes",
                )
                if mode == "profiler"
                else ()
            ),
        ),
        name="resource_decisions",
    )
    allocation = result["allocation"]
    if not isinstance(allocation, Mapping):
        raise ValueError("PASS allocation evidence is unavailable")
    if (
        decisions["conservative_incremental_hwm_bytes"]
        != resources.conservative_incremental_hwm_bytes
        or decisions["torch_population_peak_bytes"]
        != allocation.get("torch_population_peak_bytes")
        or decisions["dispatch_backend_cross_check_obligations"] != []
        or (
            mode == "profiler"
            and (
                decisions["profiler_join_pass"] is not True
                or decisions["profiler_reconstructed_live_peak_bytes"]
                != allocation.get("profiler_reconstructed_live_peak_bytes")
            )
        )
    ):
        raise ValueError("PASS resource decisions do not derive from raw endpoints")
    expected_invariants = (
        "time_pass",
        "process_memory_pass",
        "torch_memory_pass",
        "rhs_width_pass",
        "sample_width_pass",
        "storage_pass",
        "offband_fill_pass",
        "forbidden_attempts_zero",
        "pivot_margin_pass",
        "finite_pass",
        "residual_allowances_pass",
        "dispatch_observed",
        *(("profiler_join_pass",) if mode == "profiler" else ()),
        "dispatch_backend_cross_check_pass",
    )
    if tuple(item.invariant_id for item in invariants) != expected_invariants:
        raise ValueError("PASS invariant inventory is incomplete or reordered")
    for item in invariants:
        if (
            item.status is not GateStatus.PASS
            or item.value != 1
            or item.limit != 1
            or item.detail != f"{item.invariant_id}=True"
            or item.obligations
            or decisions[item.invariant_id] is not True
        ):
            raise ValueError("PASS invariant does not derive from its decision")


def classify_h8_child_outcome(
    record: H8ChildProcessRecord,
    *,
    valid_start: bool,
    invocation: H8ChildInvocation | None = None,
) -> H8ChildDecision:
    """Apply witnessed-failure dominance without discarding malformed evidence."""

    if type(record) is not H8ChildProcessRecord:
        raise ValueError("record must be an H8ChildProcessRecord")
    if type(valid_start) is not bool:
        raise ValueError("valid_start must be a bool")
    stdout_sha256 = hashlib.sha256(record.stdout).hexdigest()
    stderr_sha256 = hashlib.sha256(record.stderr).hexdigest()
    if not valid_start:
        return H8ChildDecision(
            status=GateStatus.INCONCLUSIVE,
            reasons=("child_start_not_established",),
            payload=None,
            timed_out=record.timed_out,
            exit_code=record.exit_code,
            parent_elapsed_ns=record.parent_elapsed_ns,
            stdout_sha256=stdout_sha256,
            stderr_sha256=stderr_sha256,
        )
    reasons: list[str] = []
    witnessed_failure = False
    identity_verified = False
    if record.timed_out:
        witnessed_failure = True
        reasons.append("child_timeout")
    else:
        if record.exit_code != 0:
            witnessed_failure = True
            reasons.append(_nonzero_exit_reason(record.exit_code))
        if record.parent_elapsed_ns > int(H8_MAX_SECONDS * 1e9):
            witnessed_failure = True
            reasons.append("parent_elapsed_budget_breach")
    payload: dict[str, object] | None = None
    try:
        payload = parse_h8_child_stdout(record.stdout)
    except ValueError as error:
        if not record.timed_out:
            if "nonfinite" in str(error):
                witnessed_failure = True
                reasons.append("nonfinite_child_result")
            if "duplicate JSON key" in str(error):
                witnessed_failure = True
                reasons.append("duplicate_child_identity")
            raw_witness = _invalid_stdout_witness(record.stdout)
            if raw_witness is not None:
                witnessed_failure = True
                reasons.append(raw_witness)
            reasons.append("invalid_child_stdout")
    if payload is not None:
        if invocation is None:
            reasons.append("expected_child_identity_unavailable")
        else:
            try:
                identity_verified = _verify_result_identity(
                    payload,
                    invocation,
                )
                if not identity_verified:
                    reasons.append("child_environment_identity_unavailable")
            except ValueError:
                witnessed_failure = True
                reasons.append("child_request_or_environment_identity_mismatch")
        error = payload.get("error")
        if isinstance(error, Mapping) and error.get("witnessed_violation") is True:
            witnessed_failure = True
            reasons.append(f"witnessed_{error['kind']}")
        if payload["status"] == "fail":
            witnessed_failure = True
            reasons.append("child_reported_witnessed_failure")
        elif payload["status"] == "inconclusive":
            reasons.extend(str(item) for item in payload["obligations"])  # type: ignore[union-attr]
        if _payload_has_resource_failure(payload):
            witnessed_failure = True
            reasons.append("finite_resource_budget_breach")
        if _payload_has_operation_failure(payload):
            witnessed_failure = True
            reasons.append("required_operation_omission")
    status = (
        GateStatus.FAIL
        if witnessed_failure
        else GateStatus.INCONCLUSIVE
        if (
            payload is None
            or payload["status"] == "inconclusive"
            or not identity_verified
        )
        else GateStatus.PASS
    )
    return H8ChildDecision(
        status=status,
        reasons=tuple(dict.fromkeys(reasons)),
        payload=payload,
        timed_out=record.timed_out,
        exit_code=record.exit_code,
        parent_elapsed_ns=record.parent_elapsed_ns,
        stdout_sha256=stdout_sha256,
        stderr_sha256=stderr_sha256,
    )


def make_h8_child_attempt_record(
    request: H8ChildRequest,
    invocation: H8ChildInvocation,
    process_record: H8ChildProcessRecord,
    decision: H8ChildDecision,
) -> H8ChildAttemptRecord:
    """Bind one exact request, launch, process, and decision into evidence."""

    if type(request) is not H8ChildRequest:
        raise ValueError("request must be an H8ChildRequest")
    if type(invocation) is not H8ChildInvocation:
        raise ValueError("invocation must be an H8ChildInvocation")
    invocation.__post_init__()
    if type(process_record) is not H8ChildProcessRecord:
        raise ValueError("process_record must be an H8ChildProcessRecord")
    if type(decision) is not H8ChildDecision:
        raise ValueError("decision must be an H8ChildDecision")

    invocation_request, request_bytes = _decode_h8_invocation_request(
        invocation
    )
    if invocation_request != request:
        raise ValueError("request does not match canonical invocation stdin")
    identity_bytes = _canonical_h8_invocation_identity_bytes(invocation)

    stdout_sha256 = hashlib.sha256(process_record.stdout).hexdigest()
    stderr_sha256 = hashlib.sha256(process_record.stderr).hexdigest()
    if (
        type(decision.timed_out) is not bool
        or decision.timed_out is not process_record.timed_out
        or type(decision.exit_code) is not type(process_record.exit_code)
        or decision.exit_code != process_record.exit_code
        or type(decision.parent_elapsed_ns) is not int
        or decision.parent_elapsed_ns != process_record.parent_elapsed_ns
        or type(decision.stdout_sha256) is not str
        or decision.stdout_sha256 != stdout_sha256
        or type(decision.stderr_sha256) is not str
        or decision.stderr_sha256 != stderr_sha256
    ):
        raise ValueError("decision does not match process endpoints")
    recomputed_decision = classify_h8_child_outcome(
        process_record,
        valid_start=True,
        invocation=invocation,
    )
    if decision != recomputed_decision:
        raise ValueError("decision does not match fresh child classification")
    decision = recomputed_decision

    typed_result: H8ChildResult | H8ControlResult | None = None
    operation_reachability: Mapping[str, object] | None = None
    residuals: Mapping[str, object] | None = None
    resource_decisions: Mapping[str, object] | None = None
    nonpass_envelope: Mapping[str, object] | None = None
    if decision.payload is not None:
        if not isinstance(decision.payload, Mapping):
            raise ValueError("decision payload must be a mapping or None")
        process_payload = parse_h8_child_stdout(process_record.stdout)
        if process_payload != decision.payload:
            raise ValueError("decision payload does not match process stdout")
        identity_verified = True
        try:
            identity_verified = _verify_result_identity(
                decision.payload,
                invocation,
            )
        except ValueError:
            identity_verified = False
            if (
                decision.status is not GateStatus.FAIL
                or "child_request_or_environment_identity_mismatch"
                not in decision.reasons
            ):
                raise
        trusted_payload = identity_verified and not process_record.timed_out
        if trusted_payload and decision.payload["status"] == "pass":
            typed_result = (
                decode_h8_control_result(decision.payload)
                if decision.payload["mode"] == "negative_control"
                else decode_h8_child_result(decision.payload)
            )
        if (
            process_record.timed_out
            or not identity_verified
            or decision.payload["status"] != "pass"
        ):
            nonpass_envelope = decision.payload
        result_payload = decision.payload["result"]
        if (
            trusted_payload
            and decision.payload["mode"] in ("production", "profiler")
            and result_payload is not None
        ):
            if not isinstance(result_payload, Mapping):
                raise ValueError("child result payload must be a mapping")
            endpoints: dict[str, Mapping[str, object]] = {}
            for name in (
                "operation_reachability",
                "residuals",
                "resource_decisions",
            ):
                endpoint = result_payload[name]
                if (
                    decision.payload["status"] == "pass"
                    and not isinstance(endpoint, Mapping)
                ):
                    raise ValueError(f"child result {name} must be a mapping")
                if isinstance(endpoint, Mapping):
                    endpoints[name] = endpoint
            observed_reachability = endpoints.get("operation_reachability")
            if observed_reachability is not None:
                if all(
                    type(operation) is str
                    and operation
                    and type(reached) is bool
                    for operation, reached in observed_reachability.items()
                ):
                    operation_reachability = (
                        {
                            operation: observed_reachability[operation]
                            for operation in H8_REQUIRED_OPERATIONS
                        }
                        if set(observed_reachability)
                        == set(H8_REQUIRED_OPERATIONS)
                        else observed_reachability
                    )
                elif decision.payload["status"] == "pass":
                    raise ValueError(
                        "child result operation reachability must be boolean"
                    )
            observed_residuals = endpoints.get("residuals")
            if observed_residuals is not None:
                if all(
                    type(name) is str
                    and name
                    and type(residual) in (int, float)
                    and math.isfinite(float(residual))
                    and float(residual) >= 0.0
                    for name, residual in observed_residuals.items()
                ):
                    residuals = {
                        name: float(residual)
                        for name, residual in observed_residuals.items()
                    }
                elif decision.payload["status"] == "pass":
                    raise ValueError(
                        "child result residuals must be finite and nonnegative"
                    )
            resource_decisions = endpoints.get("resource_decisions")

    return H8ChildAttemptRecord(
        request=request,
        status=decision.status,
        reasons=decision.reasons,
        result=typed_result,
        timed_out=process_record.timed_out,
        exit_code=process_record.exit_code,
        parent_elapsed_ns=process_record.parent_elapsed_ns,
        request_sha256=hashlib.sha256(request_bytes).hexdigest(),
        identities_sha256=hashlib.sha256(identity_bytes).hexdigest(),
        stdout_sha256=stdout_sha256,
        stderr_sha256=stderr_sha256,
        operation_reachability=operation_reachability,  # type: ignore[arg-type]
        residuals=residuals,  # type: ignore[arg-type]
        resource_decisions=resource_decisions,
        nonpass_envelope=nonpass_envelope,
    )


def _decode_h8_invocation_request(
    invocation: H8ChildInvocation,
) -> tuple[H8ChildRequest, bytes]:
    stdin = invocation.stdin
    if (
        type(stdin) is not bytes
        or not stdin.endswith(b"\n")
        or stdin.count(b"\n") != 1
        or not stdin[:-1]
    ):
        raise ValueError("invocation stdin must be one canonical request line")
    request_bytes = stdin[:-1]
    try:
        decoded = request_bytes.decode("utf-8", errors="strict")
        value = json.loads(
            decoded,
            parse_constant=lambda constant: (_raise_nonfinite(constant)),
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invocation stdin is not one JSON request") from error
    if canonical_json_bytes(value) != request_bytes:
        raise ValueError("invocation stdin request is not canonical")
    checked = _validate_child_request(value)
    return (
        H8ChildRequest(
            mode=checked["mode"],  # type: ignore[arg-type]
            seed=checked["seed"],  # type: ignore[arg-type]
            repetition=checked["repetition"],  # type: ignore[arg-type]
            config_sha256=checked["config_sha256"],  # type: ignore[arg-type]
            protocol_sha256=checked["protocol_sha256"],  # type: ignore[arg-type]
            control_id=checked["control_id"],  # type: ignore[arg-type]
        ),
        request_bytes,
    )


def _canonical_h8_invocation_identity_bytes(
    invocation: H8ChildInvocation,
) -> bytes:
    if not isinstance(invocation.environment, Mapping):
        raise ValueError("invocation environment must be a mapping")
    identity_json = invocation.environment.get(H8_CHILD_IDENTITY_ENV)
    if type(identity_json) is not str:
        raise ValueError("invocation child identities are unavailable")
    try:
        identity_bytes = identity_json.encode("ascii", errors="strict")
        value = json.loads(
            identity_json,
            parse_constant=lambda constant: (_raise_nonfinite(constant)),
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (UnicodeEncodeError, json.JSONDecodeError) as error:
        raise ValueError("invocation child identities are not valid JSON") from error
    checked = _validate_identity_records(value)
    if canonical_json_bytes(checked) != identity_bytes:
        raise ValueError("invocation child identities are not canonical")
    return identity_bytes


def _verify_result_identity(
    payload: Mapping[str, object],
    invocation: H8ChildInvocation,
) -> bool:
    if type(invocation) is not H8ChildInvocation:
        raise ValueError("invocation must be an H8ChildInvocation")
    request = json.loads(
        invocation.stdin.decode("utf-8"),
        object_pairs_hook=_reject_duplicate_pairs,
    )
    checked_request = _validate_child_request(request)
    expected_request_sha256 = hashlib.sha256(invocation.stdin[:-1]).hexdigest()
    fields = (
        ("mode", "mode"),
        ("seed", "seed"),
        ("repetition", "repetition"),
        ("control_id", "control_id"),
        ("config_sha256", "config_sha256"),
        ("protocol_sha256", "protocol_sha256"),
    )
    if any(
        payload[result_name] != checked_request[request_name]
        for result_name, request_name in fields
    ):
        raise ValueError("child result request identity mismatch")
    if payload["request_sha256"] != expected_request_sha256:
        raise ValueError("child result request SHA-256 mismatch")
    expected_identity_json = invocation.environment.get(H8_CHILD_IDENTITY_ENV)
    if expected_identity_json is None:
        raise ValueError("parent-frozen child identities are unavailable")
    expected_identities = _validate_identity_records(
        json.loads(
            expected_identity_json,
            object_pairs_hook=_reject_duplicate_pairs,
        )
    )
    observed_identities = _validate_identity_records(
        payload["identities"],
        allow_observability_error=payload.get("status") != "pass",
    )
    identity_unavailable = False
    for name in H8_CHILD_IDENTITY_KEYS:
        observed = observed_identities[name]
        if "observability_error" in observed:
            identity_unavailable = True
        elif observed["sha256"] != expected_identities[name]["sha256"]:
            raise ValueError("child environment identity hash mismatch")
    return not identity_unavailable


def _payload_has_resource_failure(payload: Mapping[str, object]) -> bool:
    result = payload.get("result")
    if not isinstance(result, Mapping):
        return False
    decisions = result.get("resource_decisions")
    if isinstance(decisions, Mapping):
        for name in (
            "time_pass",
            "process_memory_pass",
            "torch_memory_pass",
            "rhs_width_pass",
            "sample_width_pass",
            "storage_pass",
            "offband_fill_pass",
            "forbidden_attempts_zero",
            "pivot_margin_pass",
            "finite_pass",
            "residual_allowances_pass",
            "profiler_join_pass",
            "dispatch_backend_cross_check_pass",
        ):
            if decisions.get(name) is False:
                return True
        if _raw_residual_allowance_failure(
            decisions.get("residual_allowances")
        ):
            return True
        if _raw_endpoint_exceeds(
            decisions,
            "conservative_incremental_hwm_bytes",
            H8_MAX_PROCESS_INCREMENTAL_BYTES,
        ) or _raw_endpoint_exceeds(
            decisions,
            "torch_population_peak_bytes",
            H8_MAX_TORCH_POPULATION_BYTES,
        ):
            return True

    resources = result.get("resources")
    if isinstance(resources, Mapping) and (
        _raw_endpoint_exceeds(
            resources,
            "child_elapsed_ns",
            int(H8_MAX_SECONDS * 1e9),
        )
        or _raw_endpoint_exceeds(
            resources,
            "conservative_incremental_hwm_bytes",
            H8_MAX_PROCESS_INCREMENTAL_BYTES,
        )
        or _raw_endpoint_is_positive(resources, "parent_elapsed_ns")
    ):
        return True

    allocation = result.get("allocation")
    if isinstance(allocation, Mapping):
        if (
            _raw_endpoint_exceeds(
                allocation,
                "torch_population_peak_bytes",
                H8_MAX_TORCH_POPULATION_BYTES,
            )
            or _raw_endpoint_exceeds(
                allocation,
                "profiler_reconstructed_live_peak_bytes",
                H8_MAX_TORCH_POPULATION_BYTES,
            )
            or _raw_endpoint_is_positive(
                allocation,
                "dispatch_forbidden_attempt_count",
            )
            or _raw_endpoint_is_positive(
                allocation,
                "backend_forbidden_attempt_count",
            )
        ):
            return True
        cross_check = allocation.get("dispatch_cross_check")
        if isinstance(cross_check, Mapping) and (
            _raw_endpoint_is_positive(
                cross_check,
                "backend_forbidden_attempt_count",
            )
            or _raw_endpoint_is_positive(
                cross_check,
                "dispatch_forbidden_attempt_count",
            )
        ):
            return True
        for name in ("dispatch_events", "numpy_guard_events"):
            events = allocation.get(name)
            if type(events) is list and any(
                isinstance(event, Mapping)
                and type(event.get("forbidden_reason")) is str
                and bool(event["forbidden_reason"])
                for event in events
            ):
                return True

    storage = result.get("storage")
    if isinstance(storage, Mapping) and (
        any(
            _raw_endpoint_exceeds(
                storage,
                name,
                H8_MAX_STORAGE_SCALARS,
            )
            for name in (
                "precision_scalar_count",
                "factor_scalar_count",
                "selected_inverse_scalar_count",
            )
        )
        or _raw_endpoint_is_positive(storage, "upper_block_scalar_count")
    ):
        return True

    fill = result.get("fill")
    if isinstance(fill, Mapping) and (
        _raw_endpoint_is_positive(fill, "observed_offband_blocks")
        or _raw_endpoint_is_positive(fill, "duplicated_upper_blocks")
    ):
        return True

    workspace = result.get("workspace")
    if isinstance(workspace, Mapping) and (
        _raw_endpoint_exceeds(
            workspace,
            "maximum_scalar_count",
            H8_SCALE_LAYOUT.block_size**2,
        )
        or _raw_endpoint_exceeds(
            workspace,
            "maximum_rhs_width",
            H8_SCALE_LAYOUT.block_size,
        )
        or _raw_positive_sequence(
            workspace.get("attempted_forbidden_rhs_widths")
        )
        or _raw_dimension_exceeds(
            workspace.get("maximum_shape"),
            H8_SCALE_LAYOUT.block_size,
        )
    ):
        return True

    counters = result.get("counters")
    if isinstance(counters, Mapping) and (
        _raw_endpoint_exceeds(
            counters,
            "maximum_rhs_width",
            H8_SCALE_LAYOUT.block_size,
        )
        or _raw_endpoint_exceeds(
            counters,
            "maximum_sample_rhs_width",
            1,
        )
        or _raw_endpoint_is_positive(
            counters,
            "attempted_forbidden_selected_blocks",
        )
        or _raw_positive_sequence(
            counters.get("attempted_forbidden_rhs_widths")
        )
    ):
        return True

    diagnostics = result.get("diagnostics")
    if isinstance(diagnostics, Mapping) and (
        _raw_endpoint_below(
            diagnostics,
            "global_min_pivot",
            H8_MIN_CHOLESKY_PIVOT,
        )
        or _raw_endpoint_below(diagnostics, "global_pivot_margin", 0.0)
        or _raw_sequence_below(
            diagnostics.get("per_block_min_pivots"),
            H8_MIN_CHOLESKY_PIVOT,
        )
        or _raw_sequence_below(
            diagnostics.get("per_block_pivot_margins"),
            0.0,
        )
    ):
        return True

    return False


def _payload_has_operation_failure(payload: Mapping[str, object]) -> bool:
    result = payload.get("result")
    if not isinstance(result, Mapping):
        return False
    reachability = result.get("operation_reachability")
    if isinstance(reachability, Mapping) and any(
        operation in reachability and reachability[operation] is False
        for operation in H8_REQUIRED_OPERATIONS
    ):
        return True
    if payload.get("status") != "pass":
        return False
    counters = result.get("counters")
    if not isinstance(counters, Mapping):
        return False
    required_positive = (
        "forward_substitution_calls",
        "backward_substitution_calls",
        "solve_calls",
        "logdet_calls",
        "selected_inverse_calls",
        "sample_calls",
        "quadratic_calls",
        "trace_calls",
        "sparse_matvec_calls",
    )
    if any(
        _raw_endpoint_is_nonpositive(counters, name)
        for name in required_positive
    ):
        return True
    factorization_calls = _raw_finite_number(
        counters.get("factorization_calls")
    )
    selected_inverse_calls = _raw_finite_number(
        counters.get("selected_inverse_calls")
    )
    return (
        factorization_calls is not None
        and factorization_calls != 1.0
    ) or (
        selected_inverse_calls is not None
        and selected_inverse_calls < 2.0
    )


def _raw_finite_number(value: object) -> float | None:
    if type(value) not in (int, float):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _raw_endpoint_exceeds(
    record: Mapping[str, object],
    name: str,
    limit: int | float,
) -> bool:
    value = _raw_finite_number(record.get(name))
    return value is not None and value > float(limit)


def _raw_endpoint_below(
    record: Mapping[str, object],
    name: str,
    limit: int | float,
) -> bool:
    value = _raw_finite_number(record.get(name))
    return value is not None and value < float(limit)


def _raw_endpoint_is_positive(
    record: Mapping[str, object],
    name: str,
) -> bool:
    value = _raw_finite_number(record.get(name))
    return value is not None and value > 0.0


def _raw_endpoint_is_nonpositive(
    record: Mapping[str, object],
    name: str,
) -> bool:
    value = _raw_finite_number(record.get(name))
    return value is not None and value <= 0.0


def _raw_positive_sequence(value: object) -> bool:
    return type(value) is list and any(
        (number := _raw_finite_number(item)) is not None and number > 0.0
        for item in value
    )


def _raw_dimension_exceeds(value: object, limit: int) -> bool:
    return type(value) is list and any(
        (number := _raw_finite_number(item)) is not None
        and number > float(limit)
        for item in value
    )


def _raw_sequence_below(value: object, limit: int | float) -> bool:
    return type(value) is list and any(
        (number := _raw_finite_number(item)) is not None
        and number < float(limit)
        for item in value
    )


def _raw_residual_allowance_failure(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    for group in value.values():
        if not isinstance(group, Mapping):
            continue
        if group.get("passed") is False:
            return True
        comparisons = group.get("comparisons")
        if type(comparisons) is not list:
            continue
        for comparison in comparisons:
            if not isinstance(comparison, Mapping):
                continue
            allowance = comparison.get("allowance")
            if not isinstance(allowance, Mapping):
                continue
            residual = _raw_finite_number(allowance.get("residual"))
            threshold = _raw_finite_number(allowance.get("allowance"))
            if (
                residual is not None
                and threshold is not None
                and residual > threshold
            ):
                return True
    return False


def _invalid_stdout_witness(stdout: bytes) -> str | None:
    if (
        type(stdout) is not bytes
        or not stdout.endswith(b"\n")
        or stdout.count(b"\n") != 1
    ):
        return None
    try:
        value = json.loads(
            stdout[:-1].decode("utf-8", errors="strict"),
            parse_constant=lambda constant: (_raise_nonfinite(constant)),
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(value, Mapping):
        return None
    error = value.get("error")
    if isinstance(error, Mapping) and error.get("witnessed_violation") is True:
        return "invalid_stdout_retains_witnessed_error"
    if _payload_has_resource_failure(value):
        return "invalid_stdout_retains_witnessed_resource_failure"
    if _payload_has_operation_failure(value):
        return "invalid_stdout_retains_witnessed_operation_omission"
    return None


def _nonzero_exit_reason(exit_code: int | None) -> str:
    if exit_code is None:
        return "nonzero_child_exit"
    unsigned = exit_code & 0xFFFFFFFF
    if unsigned in (0xC0000017, 0xC000009A) or exit_code in (-9, -11):
        return "witnessed_oom_or_abnormal_exit"
    return "nonzero_child_exit"


def conservative_hwm_endpoints(
    *,
    pre_current_rss_bytes: int,
    pre_lifetime_peak_bytes: int,
    post_lifetime_peak_bytes: int,
) -> tuple[int, int]:
    """Return the primary conservative HWM and supplementary peak delta."""

    values = (
        pre_current_rss_bytes,
        pre_lifetime_peak_bytes,
        post_lifetime_peak_bytes,
    )
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("HWM endpoints must be nonnegative integers")
    return (
        max(0, post_lifetime_peak_bytes - pre_current_rss_bytes),
        max(0, post_lifetime_peak_bytes - pre_lifetime_peak_bytes),
    )


def windows_process_memory_layout(
    *,
    pointer_size: int,
) -> tuple[tuple[str, ...], int]:
    """Return the frozen PROCESS_MEMORY_COUNTERS_EX layout and native size."""

    if pointer_size not in (4, 8):
        raise ValueError("pointer_size must be four or eight bytes")
    fields = (
        "cb",
        "PageFaultCount",
        "PeakWorkingSetSize",
        "WorkingSetSize",
        "QuotaPeakPagedPoolUsage",
        "QuotaPagedPoolUsage",
        "QuotaPeakNonPagedPoolUsage",
        "QuotaNonPagedPoolUsage",
        "PagefileUsage",
        "PeakPagefileUsage",
        "PrivateUsage",
    )
    return fields, 80 if pointer_size == 8 else 44


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in _SHA256_HEX for character in value)
    )


def gamma(local_operation_count: int) -> float:
    """Return ``n*eps/(1-n*eps)`` on the frozen admissible domain."""

    if type(local_operation_count) is not int or local_operation_count <= 0:
        raise ValueError("local_operation_count must be a positive integer")
    product = local_operation_count * EPS
    if product >= 1.0:
        raise ValueError("local_operation_count * EPS must be below one")
    return product / (1.0 - product)


def make_operand_record(
    *,
    operand_id: str,
    shape: tuple[int, ...],
    infinity_norm: float,
    absolute_sum_bound: float,
    local_operation_count: int,
    source: str,
    solver_produced: bool,
    quadrature_convergence: float = 0.0,
    condition_provenance: str | None = None,
) -> H8OperandRecord:
    """Construct one named operand without admitting a global condition scale."""

    if condition_provenance is not None:
        normalized = condition_provenance.casefold().replace("-", "_")
        if "global" in normalized or "kappa" in normalized:
            raise ValueError("global condition estimates cannot enter an H8 budget")
    if type(shape) is not tuple or not shape:
        raise ValueError("shape must be a nonempty tuple")
    scalar_count = math.prod(shape)
    return H8OperandRecord(
        operand_id=operand_id,
        shape=shape,
        scalar_count=scalar_count,
        infinity_norm=infinity_norm,
        absolute_sum_bound=absolute_sum_bound,
        local_operation_count=local_operation_count,
        source=source,  # type: ignore[arg-type]
        condition_provenance=condition_provenance,
        solver_produced=solver_produced,
        quadrature_convergence=quadrature_convergence,
    )


def operand_components(operand: H8OperandRecord) -> tuple[float, float, float]:
    """Return rounding, one optional solver term, and quadrature contribution."""

    if type(operand) is not H8OperandRecord:
        raise ValueError("operand must be an H8OperandRecord")
    if operand.condition_provenance is not None:
        normalized = operand.condition_provenance.casefold().replace("-", "_")
        if "global" in normalized or "kappa" in normalized:
            raise ValueError("global condition estimates cannot enter an H8 budget")
    rounding = (
        ROUNDING_MULTIPLIER
        * gamma(operand.local_operation_count)
        * max(1.0, operand.absolute_sum_bound)
    )
    solver = (
        SOLVER_RELATIVE_BUDGET * max(1.0, operand.infinity_norm)
        if operand.solver_produced
        else 0.0
    )
    return rounding, solver, operand.quadrature_convergence


def operand_allowance(operand: H8OperandRecord) -> float:
    """Return the literal sum of the three operand-local components."""

    return math.fsum(operand_components(operand))


def reduction_component(
    left: H8OperandRecord,
    right: H8OperandRecord,
    *,
    compared_scalar_count: int,
) -> float:
    """Return the one pair-reduction component for named equal-sized operands."""

    _require_pair(left, right, compared_scalar_count)
    return (
        ROUNDING_MULTIPLIER
        * gamma(compared_scalar_count + 1)
        * max(1.0, left.infinity_norm, right.infinity_norm)
    )


def compare_operands(
    *,
    comparison_id: str,
    left: H8OperandRecord,
    right: H8OperandRecord,
    residual: float,
    compared_scalar_count: int | None = None,
) -> H8AllowanceRecord:
    """Apply the strict decisiveness and inclusive residual boundaries."""

    if type(comparison_id) is not str or not comparison_id:
        raise ValueError("comparison_id must name the ordered operand pair")
    count = left.scalar_count if compared_scalar_count is None else compared_scalar_count
    _require_pair(left, right, count)
    if type(residual) is not float or not math.isfinite(residual) or residual < 0.0:
        raise ValueError("residual must be a finite nonnegative float")
    left_rounding, left_solver, left_quadrature = operand_components(left)
    right_rounding, right_solver, right_quadrature = operand_components(right)
    reduction = reduction_component(
        left,
        right,
        compared_scalar_count=count,
    )
    components = (
        left_rounding,
        left_solver,
        left_quadrature,
        right_rounding,
        right_solver,
        right_quadrature,
        reduction,
    )
    allowance = math.fsum(components)
    scale = max(1.0, left.infinity_norm, right.infinity_norm)
    fraction = allowance / scale
    decisive = fraction < MAX_ALLOWANCE_FRACTION
    status = (
        GateStatus.INCONCLUSIVE
        if not decisive
        else GateStatus.PASS
        if residual <= allowance
        else GateStatus.FAIL
    )
    obligations = (
        ("allowance_fraction_not_strictly_below_1e-4",)
        if status is GateStatus.INCONCLUSIVE
        else ()
    )
    return H8AllowanceRecord(
        comparison_id=comparison_id,
        left=left,
        right=right,
        compared_scalar_count=count,
        left_rounding_component=left_rounding,
        left_solver_component=left_solver,
        left_quadrature_component=left_quadrature,
        right_rounding_component=right_rounding,
        right_solver_component=right_solver,
        right_quadrature_component=right_quadrature,
        reduction_component=reduction,
        allowance=allowance,
        scale=scale,
        residual=residual,
        allowance_scale_fraction=fraction,
        decisive=decisive,
        status=status,
        obligations=obligations,
    )


def literal_residual(left: Iterable[float], right: Iterable[float]) -> float:
    """Return the maximum absolute scalar residual without ``allclose``."""

    left_values = tuple(left)
    right_values = tuple(right)
    if len(left_values) != len(right_values) or not left_values:
        raise ValueError("residual operands must have the same positive scalar count")
    residual = 0.0
    for left_value, right_value in zip(left_values, right_values, strict=True):
        if (
            type(left_value) is bool
            or type(right_value) is bool
            or not isinstance(left_value, (int, float))
            or not isinstance(right_value, (int, float))
        ):
            raise ValueError("residual operands must contain real scalars, not bools")
        difference = abs(float(left_value) - float(right_value))
        if not math.isfinite(difference):
            raise ValueError("residual operands and differences must be finite")
        residual = max(residual, difference)
    return residual


# Operation-specific names make the comparison source auditable.  The formula
# is deliberately identical: each caller supplies its own named operands and
# operation count instead of selecting a hidden global tolerance.
solve_residual = literal_residual
reconstruction_residual = literal_residual
trace_residual = literal_residual
entropy_residual = literal_residual
log_normalizer_residual = literal_residual
quadrature_residual = literal_residual


def _require_pair(
    left: object,
    right: object,
    compared_scalar_count: object,
) -> tuple[H8OperandRecord, H8OperandRecord, int]:
    if type(left) is not H8OperandRecord or type(right) is not H8OperandRecord:
        raise ValueError("an allowance requires two named H8 operands")
    if not left.operand_id or not right.operand_id:
        raise ValueError("an allowance requires named operands")
    if type(compared_scalar_count) is not int or compared_scalar_count <= 0:
        raise ValueError("compared_scalar_count must be a positive integer")
    if (
        left.scalar_count != right.scalar_count
        or left.scalar_count != compared_scalar_count
    ):
        raise ValueError("operand scalar counts must match the comparison count")
    return left, right, compared_scalar_count


__all__ = [
    "EPS",
    "H8_CHILD_ENVELOPE_KEYS",
    "H8_CHILD_IDENTITY_ENV",
    "H8_CHILD_IDENTITY_KEYS",
    "H8_CHILD_REQUEST_KEYS",
    "H8_CHILD_RESULT_KEYS",
    "H8_CHILD_SCHEMA_VERSION",
    "H8_THREAD_ENVIRONMENT",
    "H8ChildDecision",
    "H8ChildInvocation",
    "H8ChildProcessRecord",
    "MAX_ALLOWANCE_FRACTION",
    "ROUNDING_MULTIPLIER",
    "SOLVER_RELATIVE_BUDGET",
    "build_h8_child_invocation",
    "canonical_json_bytes",
    "compare_operands",
    "conservative_hwm_endpoints",
    "decode_h8_child_result",
    "decode_h8_control_result",
    "entropy_residual",
    "gamma",
    "literal_residual",
    "log_normalizer_residual",
    "make_h8_child_attempt_record",
    "make_h8_identity_record",
    "make_operand_record",
    "operand_allowance",
    "operand_components",
    "parse_h8_child_stdout",
    "quadrature_residual",
    "reconstruction_residual",
    "reduction_component",
    "run_h8_child",
    "solve_residual",
    "trace_residual",
    "windows_process_memory_layout",
    "classify_h8_child_outcome",
]
