"""Isolated stdin/stdout runner for the frozen H8 sparse-scale exercise.

Only standard-library modules and the standard-library-only stable wire
authority are imported at module load time.  NumPy, PyTorch, and runtime
project modules are imported from :func:`main` only after the five one-thread
environment variables and the exact request have been validated.  The process
writes exactly one canonical JSON line to stdout.
"""

from __future__ import annotations

import base64
import contextlib
import ctypes
import dataclasses
import hashlib
import io
import json
import math
import os
import platform
import struct
import sys
import time
import zlib
from collections.abc import Callable, Mapping, Sequence
from ctypes import wintypes
from enum import Enum
from pathlib import Path
from typing import Any

from verification import h8_wire as _h8_wire


_SCHEMA_VERSION = "h8-child-v2"
_IDENTITY_ENV = "VFE4_H8_CHILD_IDENTITIES_JSON"
_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
_THREAD_ENVIRONMENT_VALUE = "1"
_TORCH_NUM_THREADS = 1
_TORCH_NUM_INTEROP_THREADS = 1
_REQUEST_KEYS = (
    "mode",
    "seed",
    "repetition",
    "config_sha256",
    "protocol_sha256",
    "control_id",
)
_ENVELOPE_KEYS = (
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
_RESULT_KEYS = (
    "input_sha256",
    "sample_noise_sha256",
    "problem_evidence",
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
_IDENTITY_KEYS = ("hardware", "affinity", "thread", "blas")
_MODES = ("production", "profiler", "negative_control")
_CONTROL_IDS = (
    "torch_matrix_d_d",
    "torch_flat_d2",
    "torch_near_d2",
    "torch_length_d",
    "torch_block_pair_slab",
    "torch_triangular_pair_storage",
    "torch_pair_stack",
    "torch_eye_full_rhs",
    "torch_dense_eigvalsh",
    "numpy_matrix_d_d",
    "numpy_outer_d_d",
    "numpy_matmul_d_d",
)
_PRODUCTION_SAMPLE_SEEDS = {
    20260721: 20261721,
    20260722: 20261722,
    20260723: 20261723,
}
_COLD_REPETITIONS = 5
_LAYOUT_HORIZON = 128
_LAYOUT_D_Z = 20
_LAYOUT_D_M = 20
_MAX_RHS_WIDTH = 40
_SAMPLE_WIDTH = 1
_MAX_STORAGE_SCALARS = 411_200
_OFFBAND_FILL_LIMIT = 0
_FORBIDDEN_ATTEMPT_LIMIT = 0
_REQUIRED_OPERATIONS = (
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
_MAX_PROCESS_BYTES = 128 * 1024 * 1024
_MAX_TORCH_BYTES = 64 * 1024 * 1024
_MAX_SECONDS = 60.0
_MIN_PIVOT = 1e-8
_PROFILER_TORCH_VERSION = "2.9.1"
_PROFILER_MEMORY_SOURCE_SHA256 = (
    "b80b4d5b58e91d581b18082c462ec7f088ec6b46ea50a1a62e2714d517a6a1b1"
)
_PROFILER_SOURCE_SHA256 = (
    "2c35f649219fb912728819b7dc0be5a5f1bd54c1efcd9502b62d976aeb278d22"
)
_PROFILER_API_CONTRACT_SHA256 = (
    "161a78f04c26fba19bb01ba6417f2cf8c00730ebeb8d007a4af0f4da433ba043"
)
_PROFILER_INVOCATION_ITEMS = (
    ("activities", ("CPU",)),
    ("profile_memory", True),
    ("record_shapes", True),
    ("with_stack", True),
)
_WINDOWS_MEMORY_FIELDS = (
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
_WINDOWS_MEMORY_DESCRIPTOR = (
    "PROCESS_MEMORY_COUNTERS_EX|"
    + ",".join(_WINDOWS_MEMORY_FIELDS)
    + "|native-size=80@ptr8,44@ptr4"
)
_LINUX_MEMORY_DESCRIPTOR = (
    "resource.getrusage(RUSAGE_SELF).ru_maxrss*1024"
    "|current=/proc/self/status:VmRSS*1024"
    "|private=/proc/self/smaps_rollup:(Private_Clean+Private_Dirty)*1024"
)
_DARWIN_MEMORY_DESCRIPTOR = (
    "resource.getrusage(RUSAGE_SELF).ru_maxrss"
    "|current=proc_pid_rusage:RUSAGE_INFO_V2.ri_resident_size"
    "|private=proc_pid_rusage:RUSAGE_INFO_V2.ri_phys_footprint"
)


class _ChildObservabilityError(RuntimeError):
    pass


class _ChildWitnessedFailure(RuntimeError):
    pass


class _ChildLocalContractDrift(RuntimeError):
    pass


class _ProfilerUnavailable(_ChildObservabilityError):
    pass


def _validate_child_local_contract() -> None:
    """Reject local executable mirrors before relying on their parser/emitter."""

    try:
        if type(_PRODUCTION_SAMPLE_SEEDS) is not dict:
            raise TypeError("production sample-seed mirror is not a dict")
        local_sample_seed_pairs = tuple(_PRODUCTION_SAMPLE_SEEDS.items())
        local_thread_environment_items = tuple(
            (name, _THREAD_ENVIRONMENT_VALUE)
            for name in _THREAD_ENVIRONMENT
        )
    except Exception as error:
        raise _ChildLocalContractDrift(
            "child-local execution inventory is unavailable"
        ) from error
    mirrors = (
        ("schema version", _SCHEMA_VERSION, _h8_wire.H8_CHILD_SCHEMA_VERSION),
        ("identity environment", _IDENTITY_ENV, _h8_wire.H8_CHILD_IDENTITY_ENV),
        ("thread environment", _THREAD_ENVIRONMENT, _h8_wire.H8_THREAD_ENVIRONMENT),
        (
            "thread environment values",
            local_thread_environment_items,
            _h8_wire.H8_THREAD_ENVIRONMENT_ITEMS,
        ),
        (
            "torch intra-op threads",
            _TORCH_NUM_THREADS,
            _h8_wire.H8_TORCH_NUM_THREADS,
        ),
        (
            "torch inter-op threads",
            _TORCH_NUM_INTEROP_THREADS,
            _h8_wire.H8_TORCH_NUM_INTEROP_THREADS,
        ),
        ("request keys", _REQUEST_KEYS, _h8_wire.H8_CHILD_REQUEST_KEYS),
        ("envelope keys", _ENVELOPE_KEYS, _h8_wire.H8_CHILD_ENVELOPE_KEYS),
        ("result keys", _RESULT_KEYS, _h8_wire.H8_CHILD_RESULT_KEYS),
        ("identity keys", _IDENTITY_KEYS, _h8_wire.H8_CHILD_IDENTITY_KEYS),
        ("modes", _MODES, _h8_wire.H8_CHILD_MODES),
        ("control IDs", _CONTROL_IDS, _h8_wire.H8_NEGATIVE_CONTROL_IDS),
        (
            "production sample seeds",
            local_sample_seed_pairs,
            _h8_wire.H8_PRODUCTION_SAMPLE_SEED_PAIRS,
        ),
        (
            "cold repetitions",
            _COLD_REPETITIONS,
            _h8_wire.H8_COLD_REPETITIONS,
        ),
        (
            "scale horizon",
            _LAYOUT_HORIZON,
            _h8_wire.H8_LAYOUT_HORIZON,
        ),
        ("scale d_z", _LAYOUT_D_Z, _h8_wire.H8_LAYOUT_D_Z),
        ("scale d_m", _LAYOUT_D_M, _h8_wire.H8_LAYOUT_D_M),
        (
            "maximum RHS width",
            _MAX_RHS_WIDTH,
            _h8_wire.H8_MAX_RHS_WIDTH,
        ),
        ("sample width", _SAMPLE_WIDTH, _h8_wire.H8_SAMPLE_WIDTH),
        (
            "maximum storage scalars",
            _MAX_STORAGE_SCALARS,
            _h8_wire.H8_MAX_STORAGE_SCALARS,
        ),
        (
            "offband fill limit",
            _OFFBAND_FILL_LIMIT,
            _h8_wire.H8_OFFBAND_FILL_LIMIT,
        ),
        (
            "forbidden-attempt limit",
            _FORBIDDEN_ATTEMPT_LIMIT,
            _h8_wire.H8_FORBIDDEN_ATTEMPT_LIMIT,
        ),
        (
            "required operations",
            _REQUIRED_OPERATIONS,
            _h8_wire.H8_REQUIRED_OPERATIONS,
        ),
        (
            "maximum process bytes",
            _MAX_PROCESS_BYTES,
            _h8_wire.H8_MAX_PROCESS_INCREMENTAL_BYTES,
        ),
        (
            "maximum torch bytes",
            _MAX_TORCH_BYTES,
            _h8_wire.H8_MAX_TORCH_POPULATION_BYTES,
        ),
        ("maximum seconds", _MAX_SECONDS, _h8_wire.H8_MAX_SECONDS),
        (
            "minimum pivot",
            _MIN_PIVOT,
            _h8_wire.H8_MIN_CHOLESKY_PIVOT,
        ),
        (
            "profiler torch version",
            _PROFILER_TORCH_VERSION,
            _h8_wire.H8_PROFILER_TORCH_VERSION,
        ),
        (
            "profiler memory source",
            _PROFILER_MEMORY_SOURCE_SHA256,
            _h8_wire.H8_PROFILER_MEMORY_SOURCE_SHA256,
        ),
        (
            "profiler source",
            _PROFILER_SOURCE_SHA256,
            _h8_wire.H8_PROFILER_SOURCE_SHA256,
        ),
        (
            "profiler API contract",
            _PROFILER_API_CONTRACT_SHA256,
            _h8_wire.H8_PROFILER_API_CONTRACT_SHA256,
        ),
        (
            "profiler invocation",
            _PROFILER_INVOCATION_ITEMS,
            _h8_wire.H8_PROFILER_INVOCATION_ITEMS,
        ),
    )
    for name, local, stable in mirrors:
        try:
            matches = type(local) is type(stable) and local == stable
        except Exception as error:
            raise _ChildLocalContractDrift(
                f"child-local {name} cannot be compared to the stable parent wire"
            ) from error
        if not matches:
            raise _ChildLocalContractDrift(
                f"child-local {name} drifted from the stable parent wire"
            )

    probe_sha256 = "2" * 64
    probe_error = _ChildLocalContractDrift("emitter inventory probe")
    detail = f"{type(probe_error).__name__}: {probe_error}"
    seed = _h8_wire.H8_PRODUCTION_SAMPLE_SEED_PAIRS[0][0]
    probe_requests = (
        {
            "mode": "production",
            "seed": seed,
            "repetition": 0,
            "config_sha256": "0" * 64,
            "protocol_sha256": "1" * 64,
            "control_id": None,
        },
        {
            "mode": "profiler",
            "seed": seed,
            "repetition": None,
            "config_sha256": "0" * 64,
            "protocol_sha256": "1" * 64,
            "control_id": None,
        },
        {
            "mode": "negative_control",
            "seed": seed,
            "repetition": None,
            "config_sha256": "0" * 64,
            "protocol_sha256": "1" * 64,
            "control_id": _h8_wire.H8_NEGATIVE_CONTROL_IDS[0],
        },
    )
    for probe_request in probe_requests:
        expected = _h8_wire.build_h8_local_contract_drift_envelope(
            probe_request,
            probe_sha256,
            detail,
        )
        try:
            observed = _error_envelope(
                probe_request,
                probe_sha256,
                expected["identities"],  # type: ignore[arg-type]
                kind=_h8_wire.H8_LOCAL_CONTRACT_DRIFT_KIND,
                error=probe_error,
                witnessed=False,
            )
            observed_line = _canonical_json_bytes(observed) + b"\n"
            observed_error = (
                observed.get("error")
                if isinstance(observed, Mapping)
                else None
            )
            emitter_matches = (
                type(observed) is dict
                and tuple(observed) == _h8_wire.H8_CHILD_ENVELOPE_KEYS
                and isinstance(observed_error, Mapping)
                and tuple(observed_error) == _h8_wire.H8_CHILD_ERROR_KEYS
                and observed == expected
                and observed_line == _h8_wire.canonical_json_line(expected)
            )
        except _ChildLocalContractDrift:
            raise
        except Exception as error:
            raise _ChildLocalContractDrift(
                "child-local error emitter cannot reproduce the stable inventory"
            ) from error
        if not emitter_matches:
            raise _ChildLocalContractDrift(
                "child-local error emitter drifted from the stable parent wire"
            )


def _runtime_protocol_sha256(config: object) -> str:
    """Recompute the v2 protocol after runtime imports and reject local drift."""

    from verification.h8_protocol import build_h8_protocol_sha256
    from vfe4.config.schema import H8ValidationConfig

    if type(config) is not H8ValidationConfig:
        raise _ChildObservabilityError(
            "child runtime config is not exact H8ValidationConfig"
        )
    local_contract = (
        _SCHEMA_VERSION,
        _REQUEST_KEYS,
        _ENVELOPE_KEYS,
        _RESULT_KEYS,
        _IDENTITY_KEYS,
    )
    stable_contract = (
        _h8_wire.H8_CHILD_SCHEMA_VERSION,
        _h8_wire.H8_CHILD_REQUEST_KEYS,
        _h8_wire.H8_CHILD_ENVELOPE_KEYS,
        _h8_wire.H8_CHILD_RESULT_KEYS,
        _h8_wire.H8_CHILD_IDENTITY_KEYS,
    )
    if local_contract != stable_contract:
        raise _ChildLocalContractDrift(
            "child-local schema or key inventory drifted from the stable parent wire"
        )
    if config.child_schema != _h8_wire.H8_CHILD_SCHEMA_VERSION:
        raise _ChildObservabilityError(
            "runtime config child schema drifted from the stable parent wire"
        )
    try:
        return build_h8_protocol_sha256(config)
    except (TypeError, ValueError) as error:
        raise _ChildObservabilityError(
            f"child protocol preimage is unavailable: {error}"
        ) from error


@dataclasses.dataclass(frozen=True, slots=True)
class _ProfilerAllocationFact:
    node_index: int
    timestamp_ns: int
    tensor_key: Any
    nbytes: int
    operator: str
    stack: tuple[str, ...]
    action: str | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class _ProfilerTensorFact:
    node_index: int
    tensor_key: Any
    dtype: str
    logical_shape: tuple[int, ...]
    operator: str
    stack: tuple[str, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class _ProfilerVersionFact:
    node_index: int
    timestamp_ns: int
    action: str
    tensor_key: Any
    version: int
    tensor_node_indices: tuple[int, ...]
    operator: str = ""
    stack: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True, slots=True)
class _ProfilerSourceIndexes:
    allocations: tuple[_ProfilerAllocationFact, ...]
    tensors: tuple[_ProfilerTensorFact, ...]
    versions: tuple[_ProfilerVersionFact, ...]
    storage_sizes: tuple[tuple[Any, int], ...]


@dataclasses.dataclass(frozen=True, slots=True)
class _ProfilerJoinFact:
    tensor_key: Any
    dtype: str
    operator: str
    stack: tuple[str, ...]
    logical_shape: tuple[int, ...]
    storage_nbytes: int
    matched_event_node_indices: tuple[int, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class _DecodedPrivateProfilerRow:
    source_row_index: int
    timestamp_ns: int
    action: str
    tensor_key: Any
    source_version: int
    raw_version: int
    raw_nbytes: int


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _identity_record(kind: str, payload: Mapping[str, object]) -> dict[str, object]:
    body = {"kind": kind, **dict(payload)}
    return {**body, "sha256": _sha256(_canonical_json_bytes(body))}


def _validate_identities(value: object) -> dict[str, dict[str, object]]:
    if not isinstance(value, Mapping) or set(value) != set(_IDENTITY_KEYS):
        raise _ChildObservabilityError("parent identity inventory is unavailable")
    checked: dict[str, dict[str, object]] = {}
    for kind in _IDENTITY_KEYS:
        item = value[kind]
        if not isinstance(item, Mapping):
            raise _ChildObservabilityError(f"parent {kind} identity is unavailable")
        record = dict(item)
        digest = record.pop("sha256", None)
        if record.get("kind") != kind or not _is_sha256(digest):
            raise _ChildObservabilityError(f"parent {kind} identity is malformed")
        if digest != _sha256(_canonical_json_bytes(record)):
            raise _ChildWitnessedFailure(f"parent {kind} identity hash mismatch")
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
            "release",
            "system",
            "machine",
            "processor",
            "cpu_count",
            "python",
            "implementation",
        }:
            raise _ChildObservabilityError("parent hardware identity schema drifted")
        if (
            any(
                type(record[name]) is not str
                for name in (
                    "platform",
                    "release",
                    "system",
                    "machine",
                    "processor",
                    "python",
                    "implementation",
                )
            )
            or any(
                not record[name]
                for name in ("platform", "release", "system", "python", "implementation")
            )
            or type(record["cpu_count"]) is not int
            or record["cpu_count"] <= 0
        ):
            raise _ChildObservabilityError("parent hardware identity is unavailable")
        return
    if kind == "affinity":
        common = {"kind", "adapter"}
        if keys == common | {"cpus"}:
            cpus = record["cpus"]
            if (
                type(cpus) is not list
                or not cpus
                or any(type(cpu) is not int or cpu < 0 for cpu in cpus)
                or len(cpus) != len(set(cpus))
            ):
                raise _ChildObservabilityError("parent affinity CPU inventory is invalid")
        elif keys == common | {"process_mask", "system_mask"}:
            if any(
                type(record[name]) is not int or record[name] <= 0
                for name in ("process_mask", "system_mask")
            ):
                raise _ChildObservabilityError("parent affinity masks are invalid")
        else:
            raise _ChildObservabilityError("parent affinity identity schema drifted")
        if type(record["adapter"]) is not str or not record["adapter"]:
            raise _ChildObservabilityError("parent affinity adapter is unavailable")
        return
    if kind == "thread":
        if keys != {
            "kind",
            "environment",
            "torch_num_threads",
            "torch_num_interop_threads",
        }:
            raise _ChildObservabilityError("parent thread identity schema drifted")
        environment = record["environment"]
        if (
            not isinstance(environment, Mapping)
            or set(environment) != set(_THREAD_ENVIRONMENT)
            or any(
                environment[name] != _THREAD_ENVIRONMENT_VALUE
                for name in _THREAD_ENVIRONMENT
            )
            or record["torch_num_threads"] != _TORCH_NUM_THREADS
            or record["torch_num_interop_threads"]
            != _TORCH_NUM_INTEROP_THREADS
        ):
            raise _ChildObservabilityError("parent thread identity is not one-thread")
        return
    if kind == "blas":
        if keys != {
            "kind",
            "torch_version",
            "numpy_version",
            "torch_config",
            "numpy_config",
        }:
            raise _ChildObservabilityError("parent BLAS identity schema drifted")
        if (
            any(
                type(record[name]) is not str
                for name in (
                    "torch_version",
                    "numpy_version",
                    "torch_config",
                    "numpy_config",
                )
            )
            or not record["torch_version"]
            or not record["numpy_version"]
        ):
            raise _ChildObservabilityError("parent BLAS identity is unavailable")
        return
    raise _ChildObservabilityError("parent identity kind is outside the frozen inventory")


def _require_thread_environment() -> dict[str, str]:
    observed = {name: os.environ.get(name) for name in _THREAD_ENVIRONMENT}
    missing = tuple(
        name
        for name, value in observed.items()
        if value != _THREAD_ENVIRONMENT_VALUE
    )
    if missing:
        raise _ChildObservabilityError(
            f"one-thread environment is missing or mismatched: {missing!r}"
        )
    return {
        name: _THREAD_ENVIRONMENT_VALUE
        for name in _THREAD_ENVIRONMENT
    }


def _parse_request(raw: bytes) -> tuple[dict[str, object], str]:
    if not raw.endswith(b"\n") or raw.count(b"\n") != 1 or not raw[:-1]:
        raise ValueError("stdin must contain exactly one canonical JSON line")
    try:
        value = json.loads(
            raw[:-1].decode("utf-8", errors="strict"),
            parse_constant=lambda constant: (_raise_nonfinite(constant)),
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("stdin is not one JSON object") from error
    if not isinstance(value, dict) or set(value) != set(_REQUEST_KEYS):
        raise ValueError("request does not use the exact H8ChildRequest schema")
    if _canonical_json_bytes(value) + b"\n" != raw:
        raise ValueError("request JSON is not canonical")
    mode = value["mode"]
    seed = value["seed"]
    repetition = value["repetition"]
    control_id = value["control_id"]
    if mode not in _MODES:
        raise ValueError("request mode is outside the frozen union")
    if type(seed) is not int or seed <= 0:
        raise ValueError("request seed must be a positive integer")
    if not _is_sha256(value["config_sha256"]) or not _is_sha256(
        value["protocol_sha256"]
    ):
        raise ValueError("request hashes must be lowercase SHA-256 values")
    if mode == "production":
        if (
            seed not in _PRODUCTION_SAMPLE_SEEDS
            or type(repetition) is not int
            or not 0 <= repetition < _COLD_REPETITIONS
            or control_id is not None
        ):
            raise ValueError("production request identity is invalid")
    elif mode == "profiler":
        if (
            seed not in _PRODUCTION_SAMPLE_SEEDS
            or repetition is not None
            or control_id is not None
        ):
            raise ValueError("profiler request identity is invalid")
    elif (
        repetition is not None
        or control_id not in _CONTROL_IDS
        or seed not in _PRODUCTION_SAMPLE_SEEDS
    ):
        raise ValueError("negative-control request identity is invalid")
    return value, _sha256(raw[:-1])


def _raise_nonfinite(constant: str) -> object:
    raise ValueError(f"nonfinite JSON constant is forbidden: {constant}")


def _reject_duplicate_pairs(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _fallback_request(raw: bytes) -> tuple[dict[str, object], str]:
    return (
        {
            "mode": "production",
            "seed": 1,
            "repetition": 0,
            "config_sha256": "0" * 64,
            "protocol_sha256": "0" * 64,
            "control_id": None,
        },
        _sha256(raw),
    )


def _fallback_identities(message: str) -> dict[str, dict[str, object]]:
    return {
        kind: _identity_record(kind, {"observability_error": message})
        for kind in _IDENTITY_KEYS
    }


def _hardware_payload() -> dict[str, object]:
    payload = {
        "platform": platform.platform(),
        "release": platform.release(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
    }
    if payload["cpu_count"] is None:
        raise _ChildObservabilityError("hardware CPU count is unavailable")
    return payload


def _affinity_payload() -> dict[str, object]:
    if hasattr(os, "sched_getaffinity"):
        affinity = sorted(os.sched_getaffinity(0))
        return {"adapter": "os.sched_getaffinity", "cpus": affinity}
    if sys.platform == "win32":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_process = kernel32.GetCurrentProcess
        get_process.restype = ctypes.c_void_p
        get_affinity = kernel32.GetProcessAffinityMask
        get_affinity.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        )
        process = get_process()
        if not process:
            raise _ChildObservabilityError(
                "GetCurrentProcess returned a null affinity handle"
            )
        process_mask = ctypes.c_size_t()
        system_mask = ctypes.c_size_t()
        if not get_affinity(
            process,
            ctypes.byref(process_mask),
            ctypes.byref(system_mask),
        ):
            code = ctypes.get_last_error()
            if not code:
                raise _ChildObservabilityError(
                    "GetProcessAffinityMask failed without an error code"
                )
            raise _ChildObservabilityError(
                f"GetProcessAffinityMask failed with error {code}"
            )
        return {
            "adapter": "GetProcessAffinityMask",
            "process_mask": process_mask.value,
            "system_mask": system_mask.value,
        }
    raise _ChildObservabilityError("process affinity API is unavailable")


def _blas_payload(torch: Any, np: Any) -> dict[str, object]:
    numpy_buffer = io.StringIO()
    with contextlib.redirect_stdout(numpy_buffer):
        np.show_config()
    return {
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "torch_config": str(torch.__config__.show()),
        "numpy_config": numpy_buffer.getvalue(),
    }


def _collect_identities(
    *,
    torch: Any,
    np: Any,
    thread_environment: Mapping[str, str],
) -> dict[str, dict[str, object]]:
    try:
        return {
            "hardware": _identity_record("hardware", _hardware_payload()),
            "affinity": _identity_record("affinity", _affinity_payload()),
            "thread": _identity_record(
                "thread",
                {
                    "environment": dict(thread_environment),
                    "torch_num_threads": int(torch.get_num_threads()),
                    "torch_num_interop_threads": int(
                        torch.get_num_interop_threads()
                    ),
                },
            ),
            "blas": _identity_record("blas", _blas_payload(torch, np)),
        }
    except (_ChildObservabilityError, _ChildWitnessedFailure):
        raise
    except Exception as error:
        raise _ChildObservabilityError(
            f"hardware/affinity/thread/BLAS identity is unavailable: {error}"
        ) from error


def _verify_identity_match(
    expected: Mapping[str, Mapping[str, object]],
    observed: Mapping[str, Mapping[str, object]],
) -> None:
    for kind in _IDENTITY_KEYS:
        if expected[kind]["sha256"] != observed[kind]["sha256"]:
            raise _ChildWitnessedFailure(f"{kind} identity hash mismatch")


def _set_and_verify_torch_threads(torch: Any) -> None:
    try:
        torch.set_num_threads(_TORCH_NUM_THREADS)
        torch.set_num_interop_threads(_TORCH_NUM_INTEROP_THREADS)
        intra = int(torch.get_num_threads())
        inter = int(torch.get_num_interop_threads())
    except Exception as error:
        raise _ChildObservabilityError(
            f"PyTorch thread setter/getter failed: {type(error).__name__}: {error}"
        ) from error
    if (
        intra != _TORCH_NUM_THREADS
        or inter != _TORCH_NUM_INTEROP_THREADS
    ):
        raise _ChildWitnessedFailure(
            f"PyTorch thread identity mismatch: intra={intra}, inter={inter}"
        )


def _load_runtime() -> tuple[Any, Any]:
    import numpy as np
    import torch

    return torch, np


def _windows_memory_snapshot() -> tuple[str, str, int, int, int]:
    pointer_size = ctypes.sizeof(ctypes.c_void_p)
    expected_size = 80 if pointer_size == 8 else 44 if pointer_size == 4 else None
    if expected_size is None:
        raise _ChildObservabilityError("unsupported Windows pointer size")

    class ProcessMemoryCountersEx(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    observed_fields = tuple(name for name, _type in ProcessMemoryCountersEx._fields_)
    observed_size = ctypes.sizeof(ProcessMemoryCountersEx)
    if observed_fields != _WINDOWS_MEMORY_FIELDS or observed_size != expected_size:
        raise _ChildObservabilityError(
            "PROCESS_MEMORY_COUNTERS_EX native layout mismatch"
        )
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    try:
        get_memory = psapi.GetProcessMemoryInfo
        get_process = kernel32.GetCurrentProcess
    except AttributeError as error:
        raise _ChildObservabilityError(
            "Windows process-memory symbol is unavailable"
        ) from error
    get_process.argtypes = []
    get_process.restype = wintypes.HANDLE
    get_memory.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCountersEx),
        wintypes.DWORD,
    ]
    get_memory.restype = wintypes.BOOL
    ctypes.set_last_error(0)
    process = get_process()
    if not process:
        code = ctypes.get_last_error()
        raise _ChildObservabilityError(
            f"GetCurrentProcess returned a null handle with error {code}"
        )
    counters = ProcessMemoryCountersEx()
    counters.cb = observed_size
    if counters.cb != expected_size:
        raise _ChildObservabilityError("PROCESS_MEMORY_COUNTERS_EX cb mismatch")
    ctypes.set_last_error(0)
    if not get_memory(process, ctypes.byref(counters), counters.cb):
        code = ctypes.get_last_error()
        if not code:
            raise _ChildObservabilityError(
                "GetProcessMemoryInfo failed without an error code"
            )
        raise _ChildObservabilityError(
            f"GetProcessMemoryInfo failed with error {code}"
        )
    values = (
        int(counters.WorkingSetSize),
        int(counters.PeakWorkingSetSize),
        int(counters.PrivateUsage),
    )
    if any(value < 0 for value in values):
        raise _ChildObservabilityError("Windows process-memory field is negative")
    return (
        "windows.PROCESS_MEMORY_COUNTERS_EX",
        _sha256(_WINDOWS_MEMORY_DESCRIPTOR.encode("ascii")),
        *values,
    )


def _read_linux_kib_field(path: Path, field: str) -> int:
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except OSError as error:
        raise _ChildObservabilityError(f"{path} is unavailable: {error}") from error
    prefix = f"{field}:"
    matches = [line for line in lines if line.startswith(prefix)]
    if len(matches) != 1:
        raise _ChildObservabilityError(f"{path} lacks one {field} field")
    parts = matches[0].split()
    if len(parts) != 3 or parts[2] != "kB" or not parts[1].isdigit():
        raise _ChildObservabilityError(f"{path} {field} units/layout are invalid")
    return int(parts[1]) * 1024


def _linux_memory_snapshot() -> tuple[str, str, int, int, int]:
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF)
    lifetime = int(usage.ru_maxrss) * 1024
    current = _read_linux_kib_field(Path("/proc/self/status"), "VmRSS")
    private = _read_linux_kib_field(
        Path("/proc/self/smaps_rollup"),
        "Private_Clean",
    ) + _read_linux_kib_field(Path("/proc/self/smaps_rollup"), "Private_Dirty")
    if min(current, lifetime, private) < 0:
        raise _ChildObservabilityError("Linux process-memory field is negative")
    return (
        "linux.resource+procfs",
        _sha256(_LINUX_MEMORY_DESCRIPTOR.encode("ascii")),
        current,
        lifetime,
        private,
    )


def _darwin_memory_snapshot() -> tuple[str, str, int, int, int]:
    import resource

    class RusageInfoV2(ctypes.Structure):
        _fields_ = [
            ("ri_uuid", ctypes.c_uint8 * 16),
            ("ri_user_time", ctypes.c_uint64),
            ("ri_system_time", ctypes.c_uint64),
            ("ri_pkg_idle_wkups", ctypes.c_uint64),
            ("ri_interrupt_wkups", ctypes.c_uint64),
            ("ri_pageins", ctypes.c_uint64),
            ("ri_wired_size", ctypes.c_uint64),
            ("ri_resident_size", ctypes.c_uint64),
            ("ri_phys_footprint", ctypes.c_uint64),
            ("ri_proc_start_abstime", ctypes.c_uint64),
            ("ri_proc_exit_abstime", ctypes.c_uint64),
            ("ri_child_user_time", ctypes.c_uint64),
            ("ri_child_system_time", ctypes.c_uint64),
            ("ri_child_pkg_idle_wkups", ctypes.c_uint64),
            ("ri_child_interrupt_wkups", ctypes.c_uint64),
            ("ri_child_pageins", ctypes.c_uint64),
            ("ri_child_elapsed_abstime", ctypes.c_uint64),
            ("ri_diskio_bytesread", ctypes.c_uint64),
            ("ri_diskio_byteswritten", ctypes.c_uint64),
        ]

    libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    proc_pid_rusage = libproc.proc_pid_rusage
    proc_pid_rusage.argtypes = (
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(RusageInfoV2),
    )
    proc_pid_rusage.restype = ctypes.c_int
    info = RusageInfoV2()
    if proc_pid_rusage(os.getpid(), 2, ctypes.byref(info)) != 0:
        code = ctypes.get_errno()
        raise _ChildObservabilityError(
            f"proc_pid_rusage failed with errno {code}"
        )
    lifetime = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    current = int(info.ri_resident_size)
    private = int(info.ri_phys_footprint)
    if min(current, lifetime, private) < 0:
        raise _ChildObservabilityError("Darwin process-memory field is negative")
    return (
        "darwin.resource+proc_pid_rusage",
        _sha256(_DARWIN_MEMORY_DESCRIPTOR.encode("ascii")),
        current,
        lifetime,
        private,
    )


def _memory_snapshot() -> tuple[str, str, int, int, int]:
    if sys.platform == "win32":
        return _windows_memory_snapshot()
    if sys.platform.startswith("linux"):
        return _linux_memory_snapshot()
    if sys.platform == "darwin":
        return _darwin_memory_snapshot()
    raise _ChildObservabilityError(
        f"no explicit HWM adapter for platform {sys.platform!r}"
    )


def _resource_record(
    pre: tuple[str, str, int, int, int],
    post: tuple[str, str, int, int, int],
    *,
    child_elapsed_ns: int,
) -> dict[str, object]:
    if pre[:2] != post[:2]:
        raise _ChildObservabilityError("process-memory adapter identity drifted")
    pre_current, pre_peak, pre_private = pre[2:]
    post_current, post_peak, post_private = post[2:]
    return {
        "adapter": pre[0],
        "adapter_sha256": pre[1],
        "pre_current_rss_bytes": pre_current,
        "pre_lifetime_peak_bytes": pre_peak,
        "pre_private_bytes": pre_private,
        "post_current_rss_bytes": post_current,
        "post_lifetime_peak_bytes": post_peak,
        "post_private_bytes": post_private,
        "conservative_incremental_hwm_bytes": max(0, post_peak - pre_current),
        "peak_to_peak_diagnostic_bytes": max(0, post_peak - pre_peak),
        "parent_elapsed_ns": 0,
        "child_elapsed_ns": child_elapsed_ns,
    }


def _checked_call(torch: Any, callback: Callable[[], Any]) -> Any:
    if torch.is_grad_enabled():
        raise _ChildWitnessedFailure("autograd became enabled before an H8 callback")
    value = callback()
    if torch.is_grad_enabled():
        raise _ChildWitnessedFailure("autograd became enabled after an H8 callback")
    return value


def _scoped_call(
    torch: Any,
    dispatch: Any,
    semantic_site: str,
    callback: Callable[[], Any],
) -> Any:
    event_start = len(dispatch.events)
    with dispatch:
        with dispatch.semantic_site(semantic_site):
            value = _checked_call(torch, callback)
    event_end = len(dispatch.events)
    witnesses = getattr(dispatch, "_h8_child_scope_witnesses", None)
    if witnesses is None:
        witnesses = []
        setattr(dispatch, "_h8_child_scope_witnesses", witnesses)
    witnesses.append(
        {
            "semantic_scope": semantic_site,
            "event_start": event_start,
            "event_end": event_end,
            "event_count": event_end - event_start,
            "callback_completed": True,
        }
    )
    return value


def _numpy_scoped_call(
    torch: Any,
    np: Any,
    guard: Any,
    semantic_site: str,
    callback: Callable[[], Any],
    *,
    logical_output_shapes: tuple[tuple[int, ...], ...] = (),
) -> Any:
    with guard:
        _install_numpy_input_registration(np=np, guard=guard)
        with guard.semantic_site(
            semantic_site,
            logical_output_shapes=logical_output_shapes,
        ):
            return _checked_call(torch, callback)


def _install_numpy_input_registration(*, np: Any, guard: Any) -> None:
    if np is None:
        return
    originals = tuple(getattr(guard, "_originals", ()))
    for owner, name, _original in originals:
        guarded = getattr(owner, name)

        def registering(
            *args: object,
            _guarded: Callable[..., object] = guarded,
            **kwargs: object,
        ) -> object:
            for array in _walk_numpy_arrays(np, (args, kwargs)):
                if id(array) not in guard._logical_arrays:
                    _register_one_numpy_array(guard, array)
            return _guarded(*args, **kwargs)

        setattr(owner, name, registering)


def _walk_numpy_arrays(np: Any, value: object) -> Sequence[Any]:
    arrays: list[Any] = []
    seen_objects: set[int] = set()
    seen_arrays: set[int] = set()

    def visit(item: object) -> None:
        item_id = id(item)
        if isinstance(item, np.ndarray):
            if item_id not in seen_arrays:
                arrays.append(item)
                seen_arrays.add(item_id)
            return
        if item_id in seen_objects:
            return
        seen_objects.add(item_id)
        if dataclasses.is_dataclass(item) and not isinstance(item, type):
            for field in dataclasses.fields(item):
                visit(getattr(item, field.name))
        elif isinstance(item, Mapping):
            for key, nested in item.items():
                visit(key)
                visit(nested)
        elif isinstance(item, (tuple, list)):
            for nested in item:
                visit(nested)

    visit(value)
    return tuple(arrays)


def _register_numpy_inventory(
    np: Any,
    guard: Any,
    value: object,
) -> list[dict[str, object]]:
    inventory: list[dict[str, object]] = []
    for array in _walk_numpy_arrays(np, value):
        selected_site, shape = _register_one_numpy_array(guard, array)
        if not bool(array.flags.c_contiguous):
            raise _ChildObservabilityError(
                "production NumPy inventory is not C-contiguous"
            )
        inventory.append(
            {
                "site": selected_site,
                "shape": list(shape),
                "dtype": str(array.dtype),
                "nbytes": int(array.nbytes),
                "sha256": _sha256(memoryview(array).cast("B")),
            }
        )
    if not inventory:
        raise _ChildObservabilityError("production NumPy inventory is empty")
    return inventory


def _register_one_numpy_array(
    guard: Any,
    array: Any,
) -> tuple[str, tuple[int, ...]]:
    if getattr(guard, "_entered", False) and id(array) not in guard._logical_arrays:
        base = getattr(array, "base", None)
        registered_base = False
        seen: set[int] = set()
        while base is not None and id(base) not in seen:
            seen.add(id(base))
            if id(base) in guard._logical_arrays:
                registered_base = True
                break
            base = getattr(base, "base", None)
        if not registered_base:
            raise _ChildWitnessedFailure(
                "unguarded NumPy producer reached a guarded operation"
            )
    shape = tuple(int(dimension) for dimension in array.shape)
    layout = guard.policy.layout
    population_axis = any(
        dimension in (layout.horizon, layout.population_size)
        for dimension in shape
    )
    candidates = (
        ("objective.population", "sample", "local")
        if population_axis
        else ("local",)
    )
    selected_site: str | None = None
    last_error: BaseException | None = None
    for site in candidates:
        try:
            guard.policy.classify_allocation(
                site=site,
                logical_shape=shape,
                operator="PREEXISTING",
                itemsize=int(array.dtype.itemsize),
            )
            selected_site = site
            break
        except RuntimeError as error:
            last_error = error
    if selected_site is None:
        if last_error is not None:
            raise last_error
        raise _ChildWitnessedFailure("NumPy inventory has no semantic site")
    guard.register_preexisting(
        array,
        site=selected_site,
        logical_shape=shape,
    )
    return selected_site, shape


def _encoded_float64_endpoint(
    torch: Any,
    value: Any,
) -> tuple[dict[str, object], bytes]:
    if (
        value.dtype != torch.float64
        or str(value.device) != "cpu"
        or sys.byteorder != "little"
    ):
        raise _ChildObservabilityError(
            "residual endpoint requires little-endian CPU float64 storage"
        )
    contiguous = value.detach().contiguous()
    array = contiguous.numpy()
    shape = tuple(int(dimension) for dimension in array.shape)
    raw = memoryview(array).cast("B").tobytes()
    scalar_count = math.prod(shape)
    if scalar_count <= 0 or len(raw) != scalar_count * 8:
        raise _ChildWitnessedFailure("residual endpoint byte extent is invalid")
    compressed = zlib.compress(raw, level=9)
    return (
        {
            "encoding": "float64-le-zlib-base64-v1",
            "shape": shape,
            "scalar_count": scalar_count,
            "raw_nbytes": len(raw),
            "raw_sha256": _sha256(raw),
            "compressed_nbytes": len(compressed),
            "payload_b64": base64.b64encode(compressed).decode("ascii"),
        },
        raw,
    )


def _raw_float64_stats(raw: bytes) -> tuple[int, float, float]:
    if not raw or len(raw) % 8:
        raise _ChildWitnessedFailure("residual endpoint bytes are malformed")
    infinity_norm = 0.0
    absolute_values: list[float] = []
    for (value,) in struct.iter_unpack("<d", raw):
        if not math.isfinite(value):
            raise _ChildWitnessedFailure("residual endpoint is nonfinite")
        absolute = abs(value)
        infinity_norm = max(infinity_norm, absolute)
        absolute_values.append(absolute)
    return len(raw) // 8, infinity_norm, math.fsum(absolute_values)


def _raw_float64_residual(left: bytes, right: bytes) -> float:
    if len(left) != len(right) or not left:
        raise _ChildWitnessedFailure("residual endpoint byte extents disagree")
    residual = 0.0
    for (left_value,), (right_value,) in zip(
        struct.iter_unpack("<d", left),
        struct.iter_unpack("<d", right),
        strict=True,
    ):
        difference = abs(left_value - right_value)
        if not math.isfinite(difference):
            raise _ChildWitnessedFailure("residual difference is nonfinite")
        residual = max(residual, difference)
    return residual


def _residual_allowance_group(
    torch: Any,
    residual_id: str,
    pairs: Sequence[tuple[Any, Any]],
    *,
    expected_specs: Mapping[str, Sequence[tuple[object, ...]]] | None = None,
) -> dict[str, object]:
    from verification.h8_budget import (
        H8_SCALE_RESIDUAL_SPECS,
        compare_operands,
        make_operand_record,
    )

    specs = (
        H8_SCALE_RESIDUAL_SPECS
        if expected_specs is None
        else expected_specs
    )
    if residual_id not in specs or len(pairs) != len(specs[residual_id]):
        raise _ChildWitnessedFailure("residual comparison inventory drifted")
    records: list[dict[str, object]] = []
    for pair, spec in zip(pairs, specs[residual_id], strict=True):
        left, right = pair
        (
            comparison_id,
            left_id,
            right_id,
            expected_shape,
            left_solver,
            right_solver,
        ) = spec
        left_endpoint, left_raw = _encoded_float64_endpoint(torch, left)
        right_endpoint, right_raw = _encoded_float64_endpoint(torch, right)
        left_count, left_inf, left_sum = _raw_float64_stats(left_raw)
        right_count, right_inf, right_sum = _raw_float64_stats(right_raw)
        left_shape = tuple(int(dimension) for dimension in left.shape)
        right_shape = tuple(int(dimension) for dimension in right.shape)
        if (
            left_shape != expected_shape
            or right_shape != expected_shape
            or left_count != right_count
        ):
            raise _ChildWitnessedFailure(
                f"{residual_id} operand shape/count witness drifted"
            )
        left_record = make_operand_record(
            operand_id=str(left_id),
            shape=left_shape,
            infinity_norm=left_inf,
            absolute_sum_bound=left_sum,
            local_operation_count=left_shape[-1],
            source="block",
            solver_produced=bool(left_solver),
        )
        right_record = make_operand_record(
            operand_id=str(right_id),
            shape=right_shape,
            infinity_norm=right_inf,
            absolute_sum_bound=right_sum,
            local_operation_count=right_shape[-1],
            source="block",
            solver_produced=bool(right_solver),
        )
        records.append(
            {
                "allowance": compare_operands(
                    comparison_id=str(comparison_id),
                    left=left_record,
                    right=right_record,
                    residual=_raw_float64_residual(left_raw, right_raw),
                ),
                "left_endpoint": left_endpoint,
                "right_endpoint": right_endpoint,
            }
        )
    residual = max(
        float(getattr(record["allowance"], "residual"))
        for record in records
    )
    decisive = all(
        getattr(record["allowance"], "decisive") is True
        for record in records
    )
    passed = all(
        getattr(getattr(record["allowance"], "status"), "value", None)
        == "pass"
        for record in records
    )
    return {
        "residual_id": residual_id,
        "aggregation": "max_residual_all_comparisons_must_pass",
        "residual": residual,
        "comparisons": tuple(records),
        "decisive": decisive,
        "passed": passed,
    }


def _assemble_recognition(problem: Any, torch: Any) -> tuple[Any, Any]:
    from vfe4.numerics.block_canonical import BlockCanonicalAssembler

    assembler = BlockCanonicalAssembler(problem.layout)

    def tensor(value: Any) -> Any:
        return torch.tensor(value, dtype=torch.float64, device="cpu")

    assembler.add_initial(
        tensor(problem.recognition.initial_mean),
        tensor(problem.recognition.initial_covariance),
    )
    for transition in problem.recognition.transitions:
        if transition.source_support != (transition.parent_t,):
            raise _ChildWitnessedFailure(
                "recognition source support is not the frozen singleton"
            )
        assembler.add_transition(
            transition.receiver_t,
            tensor(transition.matrix),
            tensor(transition.offset),
            tensor(transition.covariance),
        )
    return assembler.freeze()


def _operation_graph(
    torch: Any,
    np: Any,
    seed: int,
    dispatch: Any,
    numpy_guard: Any,
) -> dict[str, object]:
    from vfe4.generative.reference_h8 import make_h8_problem, h8_sample_noise
    from vfe4.numerics.block_tridiagonal import BlockTridiagonalCholesky
    from vfe4.numerics.sparse_information import FactorBackedInformationGaussian
    from vfe4.objective.h8_sparse import evaluate_h8_sparse_objective

    if torch.is_grad_enabled():
        raise _ChildWitnessedFailure("production graph entered with autograd enabled")
    problem = _scoped_call(
        torch,
        dispatch,
        "production.problem_build",
        lambda: _numpy_scoped_call(
            torch,
            np,
            numpy_guard,
            "local",
            lambda: make_h8_problem(
                problem_seed=seed,
                allocation_guard=numpy_guard,
            ),
        ),
    )
    numpy_inventory = _register_numpy_inventory(np, numpy_guard, problem)
    sample_noise_array = _scoped_call(
        torch,
        dispatch,
        "production.problem_build",
        lambda: _numpy_scoped_call(
            torch,
            np,
            numpy_guard,
            "sample",
            lambda: h8_sample_noise(
                problem,
                sample_noise_seed=_PRODUCTION_SAMPLE_SEEDS[seed],
                allocation_guard=numpy_guard,
            ),
            logical_output_shapes=(
                (problem.layout.population_size, problem.layout.block_size),
            ),
        ),
    )
    numpy_inventory.extend(
        _register_numpy_inventory(np, numpy_guard, sample_noise_array)
    )
    sample_noise_sha256 = _sha256(memoryview(sample_noise_array).cast("B"))
    noise = _scoped_call(
        torch,
        dispatch,
        "production.problem_build",
        lambda: torch.tensor(
            np.asarray(sample_noise_array),
            dtype=torch.float64,
            device="cpu",
        ),
    )
    precision, information = _scoped_call(
        torch,
        dispatch,
        "production.assembly",
        lambda: _assemble_recognition(problem, torch),
    )
    original_diagonal, original_lower = precision._block_refs()
    factor = _scoped_call(
        torch,
        dispatch,
        "production.factorization",
        lambda: BlockTridiagonalCholesky.factorize(precision),
    )
    per_block_pivots = tuple(float(value) for value in factor._pivots)
    per_block_pivot_margins = tuple(value - _MIN_PIVOT for value in per_block_pivots)
    if (
        len(per_block_pivots) != problem.layout.population_size
        or not all(math.isfinite(value) for value in per_block_pivots)
        or not all(math.isfinite(value) for value in per_block_pivot_margins)
    ):
        raise _ChildWitnessedFailure("factor pivot inventory is invalid")
    reconstructed_diagonal, reconstructed_lower = _scoped_call(
        torch,
        dispatch,
        "production.factorization",
        factor._precision_blocks,
    )
    reconstruction_allowance = _scoped_call(
        torch,
        dispatch,
        "production.factorization",
        lambda: _residual_allowance_group(
            torch,
            "factor_reconstruction",
            (
                (reconstructed_diagonal, original_diagonal),
                (reconstructed_lower, original_lower),
            ),
        ),
    )
    reconstruction_residual = float(reconstruction_allowance["residual"])
    precision = None
    original_diagonal = None
    original_lower = None
    reconstructed_diagonal = None
    reconstructed_lower = None
    mean = _scoped_call(
        torch,
        dispatch,
        "production.mean_solve",
        lambda: factor.solve(information),
    )
    forward = _scoped_call(
        torch,
        dispatch,
        "production.forward_substitution",
        lambda: factor.solve_factor(information, transpose=False),
    )
    backward = _scoped_call(
        torch,
        dispatch,
        "production.backward_substitution",
        lambda: factor.solve_factor(forward, transpose=True),
    )
    backward_allowance = _scoped_call(
        torch,
        dispatch,
        "production.backward_substitution",
        lambda: _residual_allowance_group(
            torch,
            "backward_substitution",
            ((backward, mean),),
        ),
    )
    backward_residual = float(backward_allowance["residual"])
    logdet = _scoped_call(
        torch,
        dispatch,
        "production.logdet",
        factor.logdet,
    )
    selected = _scoped_call(
        torch,
        dispatch,
        "production.selected_inverse",
        lambda: factor.selected_inverse(problem.layout.stored_block_ids),
    )
    selected_diagonal, selected_lower = selected._block_refs()
    selected_symmetry_allowance = _scoped_call(
        torch,
        dispatch,
        "production.selected_inverse",
        lambda: _residual_allowance_group(
            torch,
            "selected_diagonal_symmetry",
            (
                (
                    selected_diagonal,
                    selected_diagonal.transpose(-1, -2),
                ),
            ),
        ),
    )
    selected_diagonal_symmetry = float(selected_symmetry_allowance["residual"])
    selected_lower_is_finite = _scoped_call(
        torch,
        dispatch,
        "production.selected_inverse",
        lambda: bool(torch.isfinite(selected_lower).all()),
    )
    if not selected_lower_is_finite:
        raise _ChildWitnessedFailure(
            "selected lower inverse contains a nonfinite endpoint"
        )
    sample_offset = _scoped_call(
        torch,
        dispatch,
        "production.sample_width_one",
        lambda: factor.sample(noise),
    )
    sample = _scoped_call(
        torch,
        dispatch,
        "production.sample_width_one",
        lambda: mean + sample_offset,
    )
    quadratic = _scoped_call(
        torch,
        dispatch,
        "production.quadratic",
        lambda: factor.quadratic(sample),
    )
    trace_diagonal, trace_lower = _scoped_call(
        torch,
        dispatch,
        "production.sparse_trace",
        factor._precision_blocks,
    )
    trace_precision = _scoped_call(
        torch,
        dispatch,
        "production.sparse_trace",
        lambda: _reconstructed_precision(problem.layout, trace_diagonal, trace_lower),
    )
    sparse_trace = _scoped_call(
        torch,
        dispatch,
        "production.sparse_trace",
        lambda: factor.trace_inverse_product(trace_precision),
    )
    trace_precision = None
    trace_diagonal = None
    trace_lower = None
    diagnostics = _scoped_call(
        torch,
        dispatch,
        "production.condition_estimate",
        lambda: factor.diagnostics,
    )
    if (
        diagnostics.per_block_min_pivots != per_block_pivots
        or diagnostics.per_block_pivot_margins != per_block_pivot_margins
    ):
        raise _ChildWitnessedFailure(
            "condition diagnostic pivot inventory drifted"
        )
    solve_product = _scoped_call(
        torch,
        dispatch,
        "production.condition_estimate",
        lambda: factor.sparse_matvec(mean),
    )
    solve_allowance = _scoped_call(
        torch,
        dispatch,
        "production.condition_estimate",
        lambda: _residual_allowance_group(
            torch,
            "solve",
            ((solve_product, information),),
        ),
    )
    solve_residual = float(solve_allowance["residual"])
    solve_product = None
    gaussian = _scoped_call(
        torch,
        dispatch,
        "production.log_normalizer",
        lambda: FactorBackedInformationGaussian.from_factor(information, factor),
    )
    entropy = _scoped_call(
        torch,
        dispatch,
        "production.entropy",
        gaussian.entropy,
    )
    log_normalizer = _scoped_call(
        torch,
        dispatch,
        "production.log_normalizer",
        gaussian.log_normalizer,
    )
    objective = _scoped_call(
        torch,
        dispatch,
        "production.complete_objective",
        lambda: evaluate_h8_sparse_objective(problem, gaussian),
    )
    counters = factor.counters
    storage = factor.storage
    fill = factor.fill
    workspace = factor.workspace
    backend_reachability = {
        "factorization": counters.factorization_calls > 0,
        "forward_substitution": counters.forward_substitution_calls > 0,
        "backward_substitution": counters.backward_substitution_calls > 0,
        "mean_solve": counters.solve_calls > 0,
        "logdet": counters.logdet_calls > 0,
        "selected_inverse": (
            counters.selected_inverse_calls > 0
            and counters.selected_coverage_complete
        ),
        "sample_width_one": (
            counters.sample_calls > 0
            and counters.maximum_sample_rhs_width == _SAMPLE_WIDTH
        ),
        "quadratic": counters.quadratic_calls > 0,
        "sparse_trace": counters.trace_calls > 0,
        "condition_estimate": diagnostics.iterations > 0,
        "entropy": _scoped_call(
            torch,
            dispatch,
            "production.entropy",
            lambda: math.isfinite(float(entropy.item())),
        ),
        "log_normalizer": _scoped_call(
            torch,
            dispatch,
            "production.log_normalizer",
            lambda: math.isfinite(float(log_normalizer.item())),
        ),
        "complete_objective": math.isfinite(objective.complete_order21),
    }
    scoped_reachability = _scoped_dispatch_reachability(dispatch)
    operation_reachability = {
        name: backend_reachability[name] and scoped_reachability[name]
        for name in _REQUIRED_OPERATIONS
    }
    if tuple(operation_reachability) != _REQUIRED_OPERATIONS:
        raise _ChildWitnessedFailure("operation inventory order drifted")
    if not all(operation_reachability.values()):
        raise _ChildWitnessedFailure("completed graph omitted a required operation")
    scalar_endpoints = _scoped_call(
        torch,
        dispatch,
        "production.complete_objective",
        lambda: (
            float(logdet.item()),
            float(quadratic.item()),
            float(sparse_trace.item()),
            float(entropy.item()),
            float(log_normalizer.item()),
            float(objective.complete_order21),
            diagnostics.global_min_pivot,
            diagnostics.global_pivot_margin,
        ),
    )
    if not all(math.isfinite(value) for value in scalar_endpoints):
        raise _ChildWitnessedFailure("nonfinite production endpoint was witnessed")
    residuals = {
        "factor_reconstruction": reconstruction_residual,
        "solve": solve_residual,
        "backward_substitution": backward_residual,
        "selected_diagonal_symmetry": selected_diagonal_symmetry,
    }
    residual_allowances = {
        "factor_reconstruction": reconstruction_allowance,
        "solve": solve_allowance,
        "backward_substitution": backward_allowance,
        "selected_diagonal_symmetry": selected_symmetry_allowance,
    }
    return {
        "problem": problem,
        "input_sha256": problem.input_sha256,
        "sample_noise_sha256": sample_noise_sha256,
        "objective": objective,
        "storage": storage,
        "fill": fill,
        "workspace": workspace,
        "counters": counters,
        "diagnostics": diagnostics,
        "operation_reachability": operation_reachability,
        "residuals": residuals,
        "residual_allowances": residual_allowances,
        "numpy_inventory": numpy_inventory,
        "factor": factor,
    }


def _reconstructed_precision(layout: Any, diagonal: Any, lower: Any) -> Any:
    from vfe4.types.h8 import BlockTridiagonalPrecision

    return BlockTridiagonalPrecision(layout, diagonal, lower)


def _dispatch_allocation_record(
    graph: Mapping[str, object],
    dispatch: Any,
    numpy_guard: Any,
) -> tuple[dict[str, object], Any]:
    from vfe4.inference.h8_allocation import cross_check_h8_backend_dispatch

    counters = graph["counters"]
    cross_check = cross_check_h8_backend_dispatch(
        layout=graph["problem"].layout,
        counters=counters,
        storage=graph["storage"],
        workspace=graph["workspace"],
        dispatch=dispatch,
    )
    numpy_inventory = _jsonable(graph["numpy_inventory"])
    return {
        "dispatch_trace_sha256": dispatch.trace_sha256,
        "dispatch_event_count": len(dispatch.events),
        "dispatch_events": _jsonable(dispatch.events),
        "dispatch_scope_witnesses": _jsonable(
            tuple(getattr(dispatch, "_h8_child_scope_witnesses", ()))
        ),
        "dispatch_cross_check": _jsonable(cross_check),
        "dispatch_forbidden_attempt_count": dispatch.forbidden_attempt_count,
        "dispatch_live_peak_bytes": dispatch.live_peak_bytes,
        "torch_population_peak_bytes": dispatch.population_live_peak_bytes,
        "profiler_trace_sha256": None,
        "profiler_events": [],
        "profiler_lossy_rows": [],
        "preexisting_storage_count": None,
        "preexisting_bytes": None,
        "baseline_live_bytes": None,
        "profiler_reconstructed_live_peak_bytes": None,
        "profiler_all_joined_and_liveness_reconciled": None,
        "numpy_guard_event_count": len(numpy_guard.events),
        "numpy_guard_events": _jsonable(numpy_guard.events),
        "numpy_inventory": numpy_inventory,
        "numpy_inventory_sha256": _sha256(_canonical_json_bytes(numpy_inventory)),
        "backend_forbidden_attempt_count": (
            counters.attempted_forbidden_selected_blocks
            + len(counters.attempted_forbidden_rhs_widths)
        ),
        "observed_channels": ["dispatch", "numpy_guard", "backend", "os_hwm"],
    }, cross_check


def _scoped_dispatch_reachability(dispatch: Any) -> dict[str, bool]:
    sites = {
        "factorization": "production.factorization",
        "forward_substitution": "production.forward_substitution",
        "backward_substitution": "production.backward_substitution",
        "mean_solve": "production.mean_solve",
        "logdet": "production.logdet",
        "selected_inverse": "production.selected_inverse",
        "sample_width_one": "production.sample_width_one",
        "quadratic": "production.quadratic",
        "sparse_trace": "production.sparse_trace",
        "condition_estimate": "production.condition_estimate",
        "entropy": "production.entropy",
        "log_normalizer": "production.log_normalizer",
        "complete_objective": "production.complete_objective",
    }
    witnesses = tuple(getattr(dispatch, "_h8_child_scope_witnesses", ()))
    observed_scopes = {
        witness.get("semantic_scope")
        for witness in witnesses
        if isinstance(witness, Mapping)
    }
    expected_scopes = set(sites.values()) | {
        "production.problem_build",
        "production.assembly",
    }
    if (
        observed_scopes != expected_scopes
        or len(witnesses) != len(expected_scopes)
    ):
        raise _ChildWitnessedFailure(
            "operation-to-scope witness inventory drifted"
        )
    return {
        operation: any(
            witness.get("semantic_scope") == site
            and witness.get("callback_completed") is True
            and type(witness.get("event_count")) is int
            and witness["event_count"] > 0
            for witness in witnesses
        )
        for operation, site in sites.items()
    }


def _resource_decisions(
    *,
    graph: Mapping[str, object],
    resources: Mapping[str, object],
    elapsed_ns: int,
    torch_peak_bytes: int | None,
    dispatch_observed: bool,
) -> dict[str, object]:
    counters = graph["counters"]
    storage = graph["storage"]
    fill = graph["fill"]
    process_bytes = int(resources["conservative_incremental_hwm_bytes"])
    residual_allowances = graph["residual_allowances"]
    residual_decisive = all(
        record["decisive"] is True for record in residual_allowances.values()
    )
    residual_pass = all(
        record["passed"] is True for record in residual_allowances.values()
    )
    return {
        "time_pass": elapsed_ns <= int(_MAX_SECONDS * 1e9),
        "process_memory_pass": process_bytes <= _MAX_PROCESS_BYTES,
        "torch_memory_pass": (
            None
            if torch_peak_bytes is None
            else torch_peak_bytes <= _MAX_TORCH_BYTES
        ),
        "rhs_width_pass": counters.maximum_rhs_width <= _MAX_RHS_WIDTH,
        "sample_width_pass": (
            counters.maximum_sample_rhs_width == _SAMPLE_WIDTH
        ),
        "storage_pass": (
            storage.matches_expectation
            and storage.precision_scalar_count <= _MAX_STORAGE_SCALARS
            and storage.factor_scalar_count <= _MAX_STORAGE_SCALARS
            and storage.selected_inverse_scalar_count
            <= _MAX_STORAGE_SCALARS
        ),
        "offband_fill_pass": (
            fill.matches_expected_fill
            and fill.observed_offband_blocks <= _OFFBAND_FILL_LIMIT
            and fill.duplicated_upper_blocks <= _OFFBAND_FILL_LIMIT
        ),
        "forbidden_attempts_zero": (
            counters.attempted_forbidden_selected_blocks
            <= _FORBIDDEN_ATTEMPT_LIMIT
            and len(counters.attempted_forbidden_rhs_widths)
            <= _FORBIDDEN_ATTEMPT_LIMIT
        ),
        "pivot_margin_pass": graph["diagnostics"].global_min_pivot >= _MIN_PIVOT,
        "finite_pass": all(
            math.isfinite(float(value)) for value in graph["residuals"].values()
        ),
        "residual_allowances_pass": (
            residual_pass if residual_decisive else None
        ),
        "residual_allowances": _jsonable(residual_allowances),
        "dispatch_observed": dispatch_observed,
        "conservative_incremental_hwm_bytes": process_bytes,
        "torch_population_peak_bytes": torch_peak_bytes,
    }


def _invariant_records(
    decisions: Mapping[str, object],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for name, value in decisions.items():
        if not name.endswith("_pass") and name not in (
            "forbidden_attempts_zero",
            "dispatch_observed",
        ):
            continue
        status = (
            "pass"
            if value is True
            else "fail"
            if value is False
            else "inconclusive"
        )
        records.append(
            {
                "invariant_id": name,
                "status": status,
                "value": int(value) if type(value) is bool else value,
                "limit": 1,
                "detail": f"{name}={value!r}",
                "obligations": (
                    [] if status != "inconclusive" else [f"{name}_unavailable"]
                ),
            }
        )
    return records


def _production_result_payload(
    *,
    graph: Mapping[str, object],
    allocation: Mapping[str, object],
    resources: Mapping[str, object],
    decisions: Mapping[str, object],
) -> dict[str, object]:
    """Serialize the shared production/profiler body without schema drift."""

    problem = graph["problem"]
    problem_evidence = getattr(problem, "problem_evidence", None)
    if problem_evidence is None:
        raise _ChildObservabilityError(
            "production problem evidence is unavailable"
        )
    result = {
        "input_sha256": graph["input_sha256"],
        "sample_noise_sha256": graph["sample_noise_sha256"],
        "problem_evidence": _jsonable(problem_evidence),
        "objective": _jsonable(graph["objective"]),
        "storage": _jsonable(graph["storage"]),
        "fill": _jsonable(graph["fill"]),
        "workspace": _jsonable(graph["workspace"]),
        "counters": _jsonable(graph["counters"]),
        "allocation": dict(allocation),
        "resources": dict(resources),
        "diagnostics": _jsonable(graph["diagnostics"]),
        "operation_reachability": graph["operation_reachability"],
        "residuals": graph["residuals"],
        "resource_decisions": dict(decisions),
        "invariants": _invariant_records(decisions),
    }
    if tuple(result) != _RESULT_KEYS:
        raise _ChildLocalContractDrift(
            "production/profiler result key order drifted"
        )
    return result


def _run_production(torch: Any, np: Any, seed: int) -> dict[str, object]:
    from vfe4.inference.h8_allocation import (
        H8AllocationPolicy,
        H8DispatchTrace,
        H8NumpyAllocationGuard,
    )
    from vfe4.numerics.block_layout import BlockChainLayout

    policy = H8AllocationPolicy(
        BlockChainLayout(
            horizon=_LAYOUT_HORIZON,
            d_z=_LAYOUT_D_Z,
            d_m=_LAYOUT_D_M,
        )
    )
    dispatch = H8DispatchTrace(policy)
    numpy_guard = H8NumpyAllocationGuard(policy)
    pre = _memory_snapshot()
    started = time.perf_counter_ns()
    with torch.no_grad():
        graph = _operation_graph(torch, np, seed, dispatch, numpy_guard)
    elapsed = time.perf_counter_ns() - started
    post = _memory_snapshot()
    resources = _resource_record(pre, post, child_elapsed_ns=elapsed)
    allocation, cross_check = _dispatch_allocation_record(
        graph,
        dispatch,
        numpy_guard,
    )
    decisions = _resource_decisions(
        graph=graph,
        resources=resources,
        elapsed_ns=elapsed,
        torch_peak_bytes=dispatch.population_live_peak_bytes,
        dispatch_observed=cross_check.complete,
    )
    decisions["dispatch_backend_cross_check_pass"] = cross_check.complete
    decisions["dispatch_backend_cross_check_obligations"] = list(
        cross_check.obligations
    )
    return _production_result_payload(
        graph=graph,
        allocation=allocation,
        resources=resources,
        decisions=decisions,
    )


def _raw_profiler_rows(
    profile: Any,
) -> tuple[object, Sequence[object], Sequence[object]]:
    try:
        memory_profile = profile._memory_profile()
        timeline = tuple(memory_profile.timeline)
        roots = tuple(profile.profiler.kineto_results.experimental_event_tree())
    except Exception as error:
        raise _ProfilerUnavailable(
            f"pinned profiler private API is unavailable: {type(error).__name__}: {error}"
        ) from error
    if not timeline or not roots:
        raise _ProfilerUnavailable("profiler timeline or event tree is empty")
    return memory_profile, timeline, roots


def _decode_private_profiler_row(
    row: object,
    *,
    source_row_index: int,
    tensor_key_type: Any,
) -> _DecodedPrivateProfilerRow:
    try:
        if type(row) is not tuple or len(row) != 4:
            raise ValueError("timeline row is not an exact four-tuple")
        timestamp_ns, action_value, keyed_version, nbytes = row
        action = getattr(action_value, "name")
        if action not in (
            "PREEXISTING",
            "CREATE",
            "INCREMENT_VERSION",
            "DESTROY",
        ):
            raise ValueError("timeline action is outside the pinned union")
        if (
            type(timestamp_ns) is not int
            or timestamp_ns < -1
            or (timestamp_ns == -1 and action != "PREEXISTING")
            or type(nbytes) is not int
            or nbytes <= 0
            or type(keyed_version) is not tuple
            or len(keyed_version) != 2
        ):
            raise ValueError("timeline scalar/schema fields are invalid")
        private_key, source_version = keyed_version
        if type(source_version) is not int or source_version < 0:
            raise ValueError("timeline source version is invalid")
        key = _structured_tensor_key(private_key, tensor_key_type)
        if key is None:
            raise ValueError("timeline TensorKey schema is unavailable")
        raw_version = (
            source_version + 1
            if action == "INCREMENT_VERSION"
            else source_version
        )
        raw_nbytes = (
            0
            if action == "INCREMENT_VERSION"
            else -nbytes
            if action == "DESTROY"
            else nbytes
        )
        return _DecodedPrivateProfilerRow(
            source_row_index=source_row_index,
            timestamp_ns=timestamp_ns,
            action=action,
            tensor_key=key,
            source_version=source_version,
            raw_version=raw_version,
            raw_nbytes=raw_nbytes,
        )
    except _ProfilerUnavailable:
        raise
    except (
        AttributeError,
        IndexError,
        OverflowError,
        TypeError,
        ValueError,
    ) as error:
        raise _ProfilerUnavailable(
            "pinned profiler timeline schema/API drifted: "
            f"{type(error).__name__}: {error}"
        ) from error


def _profiler_records(
    profile: Any,
    *,
    policy: Any,
) -> tuple[dict[str, object], int]:
    from vfe4.inference.h8_allocation import (
        H8ProfilerEnrichment,
        H8ProfilerObservabilityGap,
        H8RawProfilerEvent,
        parse_h8_profiler_events,
    )
    from vfe4.types.h8 import H8TensorKey

    memory_profile, timeline, roots = _raw_profiler_rows(profile)
    nodes = tuple(_walk_event_nodes(roots))
    indexes = _profiler_source_indexes(memory_profile, nodes, H8TensorKey)
    raw_rows: list[Any] = []
    enrichments: list[Any] = []
    live_storage_nbytes: dict[Any, int] = {}
    for source_row_index, row in enumerate(timeline):
        decoded = _decode_private_profiler_row(
            row,
            source_row_index=source_row_index,
            tensor_key_type=H8TensorKey,
        )
        action = decoded.action
        key = decoded.tensor_key
        raw_version = decoded.raw_version
        raw_nbytes = decoded.raw_nbytes
        fact = _join_profiler_source_facts(
            indexes,
            action=action,
            timestamp_ns=decoded.timestamp_ns,
            tensor_key=key,
            version=raw_version,
            nbytes=raw_nbytes,
        )
        signed_nbytes = raw_nbytes
        if action in ("PREEXISTING", "CREATE"):
            storage_nbytes = signed_nbytes
            if storage_nbytes <= 0 or key in live_storage_nbytes:
                raise _ChildWitnessedFailure(
                    "invalid profiler live-establishing storage identity"
                )
            live_storage_nbytes[key] = storage_nbytes
        else:
            storage_nbytes = live_storage_nbytes.get(key, 0)
            if storage_nbytes <= 0:
                raise _ChildWitnessedFailure(
                    "profiler version/destroy references a dead identity"
                )
            if action == "DESTROY":
                if signed_nbytes != -storage_nbytes:
                    raise _ChildWitnessedFailure(
                        "profiler destroy byte witness drifted"
                    )
                live_storage_nbytes.pop(key)
        if fact.storage_nbytes != storage_nbytes:
            raise _ProfilerUnavailable(
                "profiler event-tree storage-byte witness is unavailable or inconsistent"
            )
        try:
            raw_rows.append(
                H8RawProfilerEvent(
                    source_row_index=source_row_index,
                    timestamp_ns=decoded.timestamp_ns,
                    action=action,
                    tensor_key=key,
                    version=raw_version,
                    nbytes=signed_nbytes,
                )
            )
        except (TypeError, ValueError) as error:
            raise _ProfilerUnavailable(
                f"profiler raw-row schema drifted: {error}"
            ) from error
        alias_of = None
        if action in ("PREEXISTING", "CREATE"):
            alias_candidates = tuple(
                live_key
                for live_key in live_storage_nbytes
                if live_key != key
                and live_key.storage_ptr == key.storage_ptr
                and live_key.allocation_id == key.allocation_id
                and live_key.device == key.device
            )
            if len(alias_candidates) > 1:
                raise _ProfilerUnavailable(
                    "profiler alias target is missing or nonunique"
                )
            alias_of = alias_candidates[0] if alias_candidates else None
        try:
            enrichments.append(
                H8ProfilerEnrichment(
                    source_row_index=source_row_index,
                    tensor_key=key,
                    version=raw_version,
                    dtype=fact.dtype,
                    device=key.device,
                    operator=fact.operator,
                    stack=fact.stack,
                    logical_shape=fact.logical_shape,
                    classification="joined_profiler_event",
                    matched_event_node_indices=fact.matched_event_node_indices,
                    storage_span_start=key.storage_ptr,
                    storage_span_end=key.storage_ptr + fact.storage_nbytes,
                    storage_nbytes=fact.storage_nbytes,
                    alias_of=alias_of,
                )
            )
        except (TypeError, ValueError) as error:
            raise _ProfilerUnavailable(
                f"profiler enrichment schema drifted: {error}"
            ) from error
    try:
        parsed = parse_h8_profiler_events(raw_rows, enrichments, policy=policy)
    except H8ProfilerObservabilityGap as error:
        raise _ProfilerUnavailable(str(error)) from error
    allocation = {
        "dispatch_trace_sha256": None,
        "dispatch_event_count": 0,
        "dispatch_forbidden_attempt_count": 0,
        "dispatch_live_peak_bytes": 0,
        "torch_population_peak_bytes": 0,
        "profiler_trace_sha256": parsed.trace_sha256,
        "profiler_events": _jsonable(parsed.events),
        "profiler_lossy_rows": [],
        "preexisting_storage_count": parsed.preexisting_storage_count,
        "preexisting_bytes": parsed.preexisting_bytes,
        "baseline_live_bytes": parsed.baseline_live_bytes,
        "profiler_reconstructed_live_peak_bytes": parsed.live_peak_bytes,
        "profiler_all_joined_and_liveness_reconciled": (
            parsed.all_joined_and_liveness_reconciled
        ),
        "numpy_guard_event_count": 0,
        "backend_forbidden_attempt_count": 0,
        "observed_channels": ["profiler", "backend", "os_hwm"],
    }
    return allocation, parsed.live_peak_bytes


def _walk_event_nodes(roots: Sequence[object]) -> Sequence[object]:
    nodes: list[object] = []
    stack = list(reversed(roots))
    while stack:
        node = stack.pop()
        nodes.append(node)
        children = getattr(node, "children", ())
        stack.extend(reversed(tuple(children)))
    return nodes


def _join_profiler_source_facts(
    indexes: _ProfilerSourceIndexes,
    *,
    action: str,
    timestamp_ns: int,
    tensor_key: object,
    version: int,
    nbytes: int,
) -> _ProfilerJoinFact:
    if (
        action
        not in ("PREEXISTING", "CREATE", "INCREMENT_VERSION", "DESTROY")
        or type(timestamp_ns) is not int
        or type(version) is not int
        or type(nbytes) is not int
    ):
        raise _ProfilerUnavailable("profiler join key is malformed")
    versions = tuple(
        fact
        for fact in indexes.versions
        if fact.action == action
        and fact.timestamp_ns == timestamp_ns
        and fact.tensor_key == tensor_key
        and fact.version == version
    )
    if len(versions) != 1:
        raise _ProfilerUnavailable(
            "profiler data-flow version join is missing or nonunique"
        )
    version_fact = versions[0]
    tensor_facts = tuple(
        fact
        for fact in indexes.tensors
        if fact.tensor_key == tensor_key
        and fact.node_index in version_fact.tensor_node_indices
    )
    metadata = {
        (fact.dtype, fact.logical_shape)
        for fact in tensor_facts
    }
    if not tensor_facts or len(metadata) != 1:
        raise _ProfilerUnavailable(
            "profiler tensor-metadata join is missing or inconsistent"
        )
    allocation_facts = tuple(
        fact
        for fact in indexes.allocations
        if fact.tensor_key == tensor_key
        and fact.timestamp_ns == timestamp_ns
        and fact.nbytes == abs(nbytes)
        and fact.action == action
    )
    if action in ("CREATE", "DESTROY") and len(allocation_facts) != 1:
        raise _ProfilerUnavailable(
            "profiler allocation-node join is missing or nonunique"
        )
    if action not in ("CREATE", "DESTROY") and allocation_facts:
        raise _ProfilerUnavailable(
            "profiler action has an unexpected allocation-node join"
        )
    storage_sizes = tuple(
        size for key, size in indexes.storage_sizes if key == tensor_key
    )
    if len(storage_sizes) != 1 or storage_sizes[0] <= 0:
        raise _ProfilerUnavailable(
            "profiler size-map join is missing or nonunique"
        )
    storage_nbytes = storage_sizes[0]
    expected_nbytes = (
        0
        if action == "INCREMENT_VERSION"
        else -storage_nbytes
        if action == "DESTROY"
        else storage_nbytes
    )
    if nbytes != expected_nbytes:
        raise _ProfilerUnavailable(
            "profiler timeline and size-map byte witnesses disagree"
        )
    tensor_fact = min(tensor_facts, key=lambda item: item.node_index)
    allocation_fact = allocation_facts[0] if allocation_facts else None
    operator = (
        version_fact.operator
        or (allocation_fact.operator if allocation_fact is not None else "")
        or tensor_fact.operator
    )
    stack = (
        version_fact.stack
        or (allocation_fact.stack if allocation_fact is not None else ())
        or tensor_fact.stack
    )
    if not operator or not stack:
        raise _ProfilerUnavailable(
            "profiler operator/source-stack witness is unavailable"
        )
    matched_indices = tuple(
        sorted(
            {
                version_fact.node_index,
                *(fact.node_index for fact in tensor_facts),
                *(
                    ()
                    if allocation_fact is None
                    else (allocation_fact.node_index,)
                ),
            }
        )
    )
    dtype, logical_shape = next(iter(metadata))
    return _ProfilerJoinFact(
        tensor_key=tensor_key,
        dtype=dtype,
        operator=operator,
        stack=stack,
        logical_shape=logical_shape,
        storage_nbytes=storage_nbytes,
        matched_event_node_indices=matched_indices,
    )


def _profiler_source_indexes(
    memory_profile: object,
    nodes: Sequence[object],
    tensor_key_type: Any,
) -> _ProfilerSourceIndexes:
    try:
        return _build_profiler_source_indexes(
            memory_profile,
            nodes,
            tensor_key_type,
        )
    except _ProfilerUnavailable:
        raise
    except (
        AttributeError,
        IndexError,
        KeyError,
        OverflowError,
        TypeError,
        ValueError,
    ) as error:
        raise _ProfilerUnavailable(
            "pinned profiler private schema does not match its source contract: "
            f"{type(error).__name__}: {error}"
        ) from error


def _build_profiler_source_indexes(
    memory_profile: object,
    nodes: Sequence[object],
    tensor_key_type: Any,
) -> _ProfilerSourceIndexes:
    node_indices = {id(node): index for index, node in enumerate(nodes)}
    allocations: list[_ProfilerAllocationFact] = []
    tensors: list[_ProfilerTensorFact] = []
    private_keys: dict[object, object] = {}
    for node_index, node in enumerate(nodes):
        typed = getattr(node, "typed")
        if type(typed) is not tuple or len(typed) != 2:
            raise _ProfilerUnavailable("profiler event typed union drifted")
        event_type = getattr(typed[0], "name", None)
        fields = typed[1]
        timestamp = getattr(node, "start_time_ns")
        if type(timestamp) is not int:
            raise _ProfilerUnavailable("profiler event timestamp schema drifted")
        if event_type == "Allocation":
            private_key = fields
            key = _structured_tensor_key(private_key, tensor_key_type)
            allocation_size = getattr(fields, "alloc_size")
            if key is None or type(allocation_size) is not int or allocation_size == 0:
                continue
            allocations.append(
                _ProfilerAllocationFact(
                    node_index=node_index,
                    timestamp_ns=timestamp,
                    tensor_key=key,
                    nbytes=abs(allocation_size),
                    operator=_profiler_owner_operator(node),
                    stack=_structured_stack(node),
                    action="CREATE" if allocation_size > 0 else "DESTROY",
                )
            )
        elif event_type == "TorchOp":
            operator = str(getattr(fields, "name"))
            stack = _structured_stack(node)
            for candidate in _walk_profiler_tensor_metadata(
                getattr(fields, "inputs")
            ):
                key = _structured_tensor_key(candidate, tensor_key_type)
                shape = _structured_shape(candidate)
                dtype = getattr(candidate, "dtype", None)
                if key is None or shape is None or dtype is None:
                    continue
                fact = _ProfilerTensorFact(
                    node_index=node_index,
                    tensor_key=key,
                    dtype=str(dtype),
                    logical_shape=shape,
                    operator=operator,
                    stack=stack,
                )
                if fact not in tensors:
                    tensors.append(fact)
    if not allocations or not tensors:
        raise _ProfilerUnavailable(
            "profiler allocation or tensor-metadata source index is empty"
        )
    versions: list[_ProfilerVersionFact] = []
    flow_tensor_indices: dict[object, set[int]] = {}
    flow_nodes = tuple(memory_profile._data_flow_graph.flow_nodes)  # type: ignore[attr-defined]
    for flow_node in flow_nodes:
        event = flow_node._event
        event_index = node_indices.get(id(event))
        if event_index is None:
            raise _ProfilerUnavailable(
                "data-flow source event is absent from the event tree"
            )
        subtree_indices = {
            node_indices[id(item)]
            for item in _walk_event_nodes((event,))
            if id(item) in node_indices
        }
        operator = str(getattr(event, "name"))
        stack = _structured_stack(event)
        for private_key, edge in flow_node._edges.items():
            key = _structured_tensor_key(private_key, tensor_key_type)
            if key is None:
                continue
            private_keys.setdefault(key, private_key)
            tensor_indices = tuple(
                fact.node_index
                for fact in tensors
                if fact.tensor_key == key
                and fact.node_index in subtree_indices
            )
            flow_tensor_indices.setdefault(key, set()).update(tensor_indices)
            if edge.is_allocation:
                matches = tuple(
                    fact
                    for fact in allocations
                    if fact.tensor_key == key
                    and fact.action == "CREATE"
                    and fact.node_index in subtree_indices
                )
                if len(matches) != 1:
                    raise _ProfilerUnavailable(
                        "data-flow allocation source is missing or nonunique"
                    )
                versions.append(
                    _ProfilerVersionFact(
                        node_index=event_index,
                        timestamp_ns=matches[0].timestamp_ns,
                        action="CREATE",
                        tensor_key=key,
                        version=0,
                        tensor_node_indices=tensor_indices,
                        operator=operator,
                        stack=stack,
                    )
                )
            elif edge.mutated:
                input_version = edge.input_version
                if type(input_version) is not int:
                    raise _ProfilerUnavailable(
                        "data-flow mutation lacks its input version"
                    )
                versions.append(
                    _ProfilerVersionFact(
                        node_index=event_index,
                        timestamp_ns=int(getattr(event, "start_time_ns")),
                        action="INCREMENT_VERSION",
                        tensor_key=key,
                        version=input_version + 1,
                        tensor_node_indices=tensor_indices,
                        operator=operator,
                        stack=stack,
                    )
                )
            if edge.is_deletion:
                deletion_version = (
                    0 if edge.is_allocation else edge.input_version
                )
                matches = tuple(
                    fact
                    for fact in allocations
                    if fact.tensor_key == key
                    and fact.action == "DESTROY"
                    and fact.node_index in subtree_indices
                )
                if len(matches) != 1 or type(deletion_version) is not int:
                    raise _ProfilerUnavailable(
                        "data-flow deletion source is missing or nonunique"
                    )
                versions.append(
                    _ProfilerVersionFact(
                        node_index=event_index,
                        timestamp_ns=matches[0].timestamp_ns,
                        action="DESTROY",
                        tensor_key=key,
                        version=deletion_version,
                        tensor_node_indices=tensor_indices,
                        operator=operator,
                        stack=stack,
                    )
                )
    positive_allocation_keys = {
        fact.tensor_key
        for fact in allocations
        if fact.action == "CREATE"
    }
    snapshot = memory_profile._category_snapshot()  # type: ignore[attr-defined]
    for private_key, version in snapshot:
        key = _structured_tensor_key(private_key, tensor_key_type)
        if key is None or version != 0 or key in positive_allocation_keys:
            continue
        private_keys.setdefault(key, private_key)
        tensor_indices = tuple(sorted(flow_tensor_indices.get(key, ())))
        if not tensor_indices:
            continue
        representative = min(tensor_indices)
        tensor_fact = next(
            fact
            for fact in tensors
            if fact.tensor_key == key and fact.node_index == representative
        )
        versions.append(
            _ProfilerVersionFact(
                node_index=representative,
                timestamp_ns=-1,
                action="PREEXISTING",
                tensor_key=key,
                version=0,
                tensor_node_indices=tensor_indices,
                operator=tensor_fact.operator,
                stack=tensor_fact.stack,
            )
        )
    sizes: list[tuple[object, int]] = []
    for key, private_key in private_keys.items():
        size = int(memory_profile._size_map[private_key])  # type: ignore[attr-defined]
        if size <= 0:
            raise _ProfilerUnavailable("profiler size map contains a nonpositive size")
        sizes.append((key, size))
    return _ProfilerSourceIndexes(
        allocations=tuple(allocations),
        tensors=tuple(tensors),
        versions=tuple(versions),
        storage_sizes=tuple(sizes),
    )


def _walk_profiler_tensor_metadata(value: object) -> Sequence[object]:
    if isinstance(value, (tuple, list)):
        return tuple(
            item
            for nested in value
            for item in _walk_profiler_tensor_metadata(nested)
        )
    return (
        (value,)
        if value is not None
        and hasattr(value, "sizes")
        and hasattr(value, "storage_data_ptr")
        else ()
    )


def _profiler_owner_operator(node: object) -> str:
    current = node
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        typed = getattr(current, "typed", None)
        if (
            type(typed) is tuple
            and len(typed) == 2
            and getattr(typed[0], "name", None) == "TorchOp"
        ):
            name = getattr(typed[1], "name", None)
            if name is not None and str(name):
                return str(name)
        current = getattr(current, "parent", None)
    return "[memory]"


def _structured_tensor_key(value: object, tensor_key_type: Any) -> Any | None:
    if isinstance(value, tensor_key_type):
        return value
    if value is None:
        return None
    tensor_id = getattr(value, "id", getattr(value, "tensor_id", None))
    storage = getattr(value, "storage", None)
    storage_ptr = getattr(
        storage,
        "ptr",
        getattr(
            value,
            "ptr",
            getattr(
                value,
                "storage_data_ptr",
                getattr(value, "storage_ptr", None),
            ),
        ),
    )
    allocation_id = getattr(
        storage,
        "allocation_id",
        getattr(value, "allocation_id", None),
    )
    device = getattr(value, "device", None)
    if (
        type(tensor_id) is not int
        or type(storage_ptr) is not int
        or type(allocation_id) is not int
        or device is None
    ):
        return None
    try:
        return tensor_key_type(
            tensor_id=tensor_id,
            storage_ptr=storage_ptr,
            allocation_id=allocation_id,
            device=str(device),
        )
    except (TypeError, ValueError):
        return None


def _structured_shape(candidate: object) -> tuple[int, ...] | None:
    for value in (
        getattr(candidate, "sizes", None),
        getattr(candidate, "shape", None),
    ):
        if isinstance(value, (tuple, list)):
            try:
                return tuple(int(dimension) for dimension in value)
            except (TypeError, ValueError):
                return None
    return None


def _structured_stack(node: object) -> tuple[str, ...]:
    stack_value = getattr(node, "stack", ())
    stack = (
        tuple(str(item) for item in stack_value if str(item))
        if isinstance(stack_value, (tuple, list))
        else ()
    )
    if stack:
        return stack
    frames: list[str] = []
    parent = node
    seen: set[int] = set()
    while parent is not None and id(parent) not in seen:
        seen.add(id(parent))
        typed = getattr(parent, "typed", None)
        fields = (
            typed[1]
            if type(typed) is tuple and len(typed) == 2
            else None
        )
        for frame_name in ("caller", "callsite"):
            frame = getattr(fields, frame_name, None)
            if frame is None:
                continue
            file_name = getattr(frame, "file_name", None)
            line_number = getattr(frame, "line_number", None)
            function_name = getattr(frame, "function_name", None)
            if (
                file_name is not None
                and type(line_number) is int
                and function_name is not None
            ):
                frames.append(f"{file_name}:{line_number}:{function_name}")
        name = getattr(parent, "name", None)
        if name is not None and str(name):
            frames.append(str(name))
        parent = getattr(parent, "parent", None)
    return tuple(reversed(frames)) or ("[profiler-root]",)


def _run_profiler(torch: Any, np: Any, seed: int) -> dict[str, object]:
    from vfe4.inference.h8_allocation import (
        H8AllocationPolicy,
        H8DispatchTrace,
        H8NumpyAllocationGuard,
    )
    from vfe4.numerics.block_layout import BlockChainLayout

    if (
        str(torch.__version__).split("+", maxsplit=1)[0]
        != _PROFILER_TORCH_VERSION
    ):
        raise _ProfilerUnavailable(
            "profiler mode requires exactly "
            f"torch=={_PROFILER_TORCH_VERSION}"
        )
    profiler_api = _verify_profiler_pins(torch)
    policy = H8AllocationPolicy(
        BlockChainLayout(
            horizon=_LAYOUT_HORIZON,
            d_z=_LAYOUT_D_Z,
            d_m=_LAYOUT_D_M,
        )
    )
    dispatch = H8DispatchTrace(policy)
    numpy_guard = H8NumpyAllocationGuard(policy)
    pre = _memory_snapshot()
    started = time.perf_counter_ns()
    with torch.no_grad():
        with torch.profiler.profile(
            activities=[
                getattr(torch.profiler.ProfilerActivity, activity)
                for activity in _PROFILER_INVOCATION_ITEMS[0][1]
            ],
            profile_memory=_PROFILER_INVOCATION_ITEMS[1][1],
            record_shapes=_PROFILER_INVOCATION_ITEMS[2][1],
            with_stack=_PROFILER_INVOCATION_ITEMS[3][1],
        ) as profile:
            graph = _operation_graph(torch, np, seed, dispatch, numpy_guard)
    elapsed = time.perf_counter_ns() - started
    post = _memory_snapshot()
    profiler_allocation, profiler_peak = _profiler_records(
        profile,
        policy=H8AllocationPolicy(
            BlockChainLayout(
                horizon=_LAYOUT_HORIZON,
                d_z=_LAYOUT_D_Z,
                d_m=_LAYOUT_D_M,
            )
        ),
    )
    allocation, cross_check = _dispatch_allocation_record(
        graph,
        dispatch,
        numpy_guard,
    )
    for name in (
        "profiler_trace_sha256",
        "profiler_events",
        "profiler_lossy_rows",
        "preexisting_storage_count",
        "preexisting_bytes",
        "baseline_live_bytes",
        "profiler_reconstructed_live_peak_bytes",
        "profiler_all_joined_and_liveness_reconciled",
    ):
        allocation[name] = profiler_allocation[name]
    allocation["profiler_api"] = profiler_api
    allocation["observed_channels"] = [
        "dispatch",
        "profiler",
        "numpy_guard",
        "backend",
        "os_hwm",
    ]
    resources = _resource_record(pre, post, child_elapsed_ns=elapsed)
    decisions = _resource_decisions(
        graph=graph,
        resources=resources,
        elapsed_ns=elapsed,
        torch_peak_bytes=dispatch.population_live_peak_bytes,
        dispatch_observed=cross_check.complete,
    )
    decisions["profiler_join_pass"] = bool(
        allocation["profiler_all_joined_and_liveness_reconciled"]
    )
    decisions["profiler_reconstructed_live_peak_bytes"] = profiler_peak
    decisions["dispatch_backend_cross_check_pass"] = cross_check.complete
    decisions["dispatch_backend_cross_check_obligations"] = list(
        cross_check.obligations
    )
    return _production_result_payload(
        graph=graph,
        allocation=allocation,
        resources=resources,
        decisions=decisions,
    )


def _verify_profiler_pins(torch: Any) -> dict[str, object]:
    torch_root = Path(torch.__file__).resolve().parent
    paths = {
        "memory_profile": torch_root / "profiler" / "_memory_profiler.py",
        "profiler": torch_root / "profiler" / "profiler.py",
    }
    observed: dict[str, str] = {}
    for name, path in paths.items():
        try:
            observed[name] = _sha256(path.read_bytes())
        except OSError as error:
            raise _ProfilerUnavailable(
                f"pinned profiler source is unavailable: {path}: {error}"
            ) from error
    if observed["memory_profile"] != _PROFILER_MEMORY_SOURCE_SHA256:
        raise _ProfilerUnavailable("pinned memory-profiler source hash mismatch")
    if observed["profiler"] != _PROFILER_SOURCE_SHA256:
        raise _ProfilerUnavailable("pinned profiler source hash mismatch")
    return {
        "torch_version": _PROFILER_TORCH_VERSION,
        "memory_profile_source_sha256": observed["memory_profile"],
        "profiler_source_sha256": observed["profiler"],
        "api_contract_sha256": _PROFILER_API_CONTRACT_SHA256,
    }


class _LogicalBackendControlAdapter:
    def __init__(self, backend: Any) -> None:
        self._backend = backend

    @property
    def counters(self) -> Any:
        return self._backend.counters

    def solve(self, rhs: Any) -> Any:
        shape = tuple(int(dimension) for dimension in rhs.shape)
        self._backend._record_rejected_shape(shape)
        raise ValueError("logical full-width RHS rejected before materialization")


def _make_control_backend(torch: Any, layout: Any) -> Any:
    from vfe4.numerics.block_tridiagonal import BlockTridiagonalCholesky

    identity = torch.eye(layout.block_size, dtype=torch.float64, device="cpu")
    diagonal = (
        identity.unsqueeze(0)
        .expand(layout.population_size, -1, -1)
        .clone()
    )
    lower = torch.zeros(
        (layout.horizon, layout.block_size, layout.block_size),
        dtype=torch.float64,
        device="cpu",
    )
    backend = BlockTridiagonalCholesky._from_validated_factors(
        layout=layout,
        diagonal_factor=diagonal,
        lower_factor=lower,
        pivots=tuple(1.0 for _ in range(layout.population_size)),
        input_precision_scalar_count=layout.band_storage_scalar_count,
    )
    return _LogicalBackendControlAdapter(backend)


def _run_negative_control(
    torch: Any,
    np: Any,
    *,
    control_id: str,
) -> dict[str, object]:
    from vfe4.inference.h8_allocation import (
        H8AllocationPolicy,
        H8DispatchTrace,
        H8NumpyAllocationGuard,
        execute_h8_numpy_negative_control_evidence,
        execute_h8_torch_negative_control_evidence,
    )
    from vfe4.numerics.block_layout import BlockChainLayout

    layout = BlockChainLayout(
        horizon=_LAYOUT_HORIZON,
        d_z=_LAYOUT_D_Z,
        d_m=_LAYOUT_D_M,
    )
    policy = H8AllocationPolicy(layout)
    if control_id.startswith("numpy_"):
        guard = H8NumpyAllocationGuard(policy)
        kwargs: dict[str, object] = {}
        if control_id == "numpy_outer_d_d":
            kwargs = {
                "outer_left": np.empty((layout.dimension,), dtype=np.float64),
                "outer_right": np.empty((layout.dimension,), dtype=np.float64),
            }
        elif control_id == "numpy_matmul_d_d":
            kwargs = {
                "matmul_left": np.empty((layout.dimension, 1), dtype=np.float64),
                "matmul_right": np.empty((1, layout.dimension), dtype=np.float64),
            }
        result, evidence = execute_h8_numpy_negative_control_evidence(
            control_id,
            guard,
            **kwargs,
        )
    else:
        trace = H8DispatchTrace(policy)
        backend = (
            _make_control_backend(torch, layout)
            if control_id == "torch_eye_full_rhs"
            else None
        )
        meta_options = {"dtype": torch.float64, "device": "meta"}
        result, evidence = execute_h8_torch_negative_control_evidence(
            control_id,
            trace,
            backend=backend,
            pair_member_meta=torch.empty((layout.block_size, layout.block_size), **meta_options),
            full_rhs_meta=torch.empty(
                (layout.population_size, layout.block_size, layout.dimension),
                **meta_options,
            ),
            dense_matrix_meta=torch.empty(
                (layout.dimension, layout.dimension),
                **meta_options,
            ),
        )
    return {
        "summary": _jsonable(result),
        "evidence": _jsonable(evidence),
    }


def _jsonable(value: object) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, bytes):
        return {"nbytes": len(value), "sha256": _sha256(value)}
    if type(value) in (str, int, float, bool) or value is None:
        if type(value) is float and not math.isfinite(value):
            raise _ChildWitnessedFailure("nonfinite serialization endpoint")
        return value
    raise TypeError(f"unsupported child serialization type: {type(value).__name__}")


def _envelope(
    request: Mapping[str, object],
    request_sha256: str,
    *,
    status: str,
    obligations: Sequence[str],
    identities: Mapping[str, object],
    result: Mapping[str, object] | None = None,
    control: Mapping[str, object] | None = None,
    error: Mapping[str, object] | None = None,
) -> dict[str, object]:
    value = {
        "schema_version": _SCHEMA_VERSION,
        "mode": request["mode"],
        "seed": request["seed"],
        "repetition": request["repetition"],
        "control_id": request["control_id"],
        "request_sha256": request_sha256,
        "config_sha256": request["config_sha256"],
        "protocol_sha256": request["protocol_sha256"],
        "status": status,
        "obligations": list(obligations),
        "identities": identities,
        "result": result,
        "control": control,
        "error": error,
    }
    if tuple(value) != _ENVELOPE_KEYS:
        raise _ChildLocalContractDrift("child envelope key order drifted")
    return value


def _error_envelope(
    request: Mapping[str, object],
    request_sha256: str,
    identities: Mapping[str, object],
    *,
    kind: str,
    error: BaseException,
    witnessed: bool,
) -> dict[str, object]:
    status = "fail" if witnessed else "inconclusive"
    obligations = () if witnessed else (kind,)
    return _envelope(
        request,
        request_sha256,
        status=status,
        obligations=obligations,
        identities=identities,
        error={
            "kind": kind,
            "message": f"{type(error).__name__}: {error}",
            "witnessed_violation": witnessed,
        },
    )


def main() -> int:
    raw = sys.stdin.buffer.read()
    stable_request: dict[str, object] | None
    stable_request_sha256: str | None
    stable_recovery_error: Exception | None = None
    try:
        stable_request, stable_request_sha256 = (
            _h8_wire.recover_h8_child_request(raw)
        )
    except ValueError:
        stable_request = None
        stable_request_sha256 = None
    except Exception as error:
        stable_request = None
        stable_request_sha256 = None
        stable_recovery_error = error

    if stable_request is None:
        request, request_sha256 = _fallback_request(raw)
    else:
        request = stable_request
        request_sha256 = stable_request_sha256
        assert request_sha256 is not None
        try:
            _validate_child_local_contract()
        except _ChildLocalContractDrift as error:
            sys.stdout.buffer.write(
                _h8_wire.canonical_h8_local_contract_drift_line(
                    request,
                    request_sha256,
                    f"{type(error).__name__}: {error}",
                )
            )
            sys.stdout.buffer.flush()
            return 0

    identities: dict[str, dict[str, object]] = _fallback_identities(
        "runtime not initialized"
    )
    try:
        thread_environment = _require_thread_environment()
        if stable_recovery_error is not None:
            raise stable_recovery_error
        if stable_request is None:
            request, request_sha256 = _parse_request(raw)
        else:
            try:
                local_request, local_request_sha256 = _parse_request(raw)
            except Exception as error:
                raise _ChildLocalContractDrift(
                    "child-local request parser rejected the stable parent request"
                ) from error
            if (
                local_request != stable_request
                or local_request_sha256 != stable_request_sha256
            ):
                raise _ChildLocalContractDrift(
                    "child-local request recovery drifted from the stable parent wire"
                )
            request = stable_request
            request_sha256 = stable_request_sha256
            assert request_sha256 is not None
        expected_raw = os.environ.get(_IDENTITY_ENV)
        if expected_raw is None:
            raise _ChildObservabilityError("parent identity environment is absent")
        expected_identities = _validate_identities(
            json.loads(
                expected_raw,
                object_pairs_hook=_reject_duplicate_pairs,
            )
        )
        with contextlib.redirect_stdout(sys.stderr):
            torch, np = _load_runtime()
            _set_and_verify_torch_threads(torch)
            identities = _collect_identities(
                torch=torch,
                np=np,
                thread_environment=thread_environment,
            )
            _verify_identity_match(expected_identities, identities)
            from vfe4.config.schema import H8ValidationConfig

            try:
                runtime_config = H8ValidationConfig.create()
            except (TypeError, ValueError) as error:
                raise _ChildObservabilityError(
                    f"child runtime config is unavailable: {error}"
                ) from error
            runtime_protocol_sha256 = _runtime_protocol_sha256(
                runtime_config
            )
            if (
                request["config_sha256"] != runtime_config.config_sha256
                or request["protocol_sha256"] != runtime_protocol_sha256
            ):
                raise _ChildObservabilityError(
                    "child request config/protocol identity drifted from "
                    "post-import recomputation"
                )
            if request["mode"] == "negative_control":
                with torch.no_grad():
                    control = _run_negative_control(
                        torch,
                        np,
                        control_id=str(request["control_id"]),
                    )
                summary = control["summary"]
                if not isinstance(summary, Mapping):
                    raise _ChildObservabilityError(
                        "negative-control summary is unavailable"
                    )
                status = str(summary["status"])
                if status == "inconclusive":
                    obligations = tuple(summary["obligations"])
                else:
                    obligations = ()
                envelope = _envelope(
                    request,
                    request_sha256,
                    status=status,
                    obligations=obligations,
                    identities=identities,
                    control=control,
                )
            else:
                result = (
                    _run_production(torch, np, int(request["seed"]))
                    if request["mode"] == "production"
                    else _run_profiler(torch, np, int(request["seed"]))
                )
                decisions = result["resource_decisions"]
                witnessed_failure = any(
                    decisions.get(name) is False
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
                    )
                    if name in decisions and name != "dispatch_observed"
                )
                if witnessed_failure:
                    status, obligations = "fail", ()
                elif decisions.get("residual_allowances_pass") is None:
                    status = "inconclusive"
                    obligations = ("residual_allowance_not_decisive",)
                elif decisions.get("dispatch_observed") is not True:
                    status = "inconclusive"
                    obligations = ("dispatch_request_result_trace_unavailable",)
                else:
                    status, obligations = "pass", ()
                envelope = _envelope(
                    request,
                    request_sha256,
                    status=status,
                    obligations=obligations,
                    identities=identities,
                    result=result,
                )
        exit_code = 0
    except _ChildLocalContractDrift as error:
        if stable_request is None or stable_request_sha256 is None:
            raise
        envelope_line = _h8_wire.canonical_h8_local_contract_drift_line(
            stable_request,
            stable_request_sha256,
            f"{type(error).__name__}: {error}",
        )
        sys.stdout.buffer.write(envelope_line)
        sys.stdout.buffer.flush()
        return 0
    except MemoryError as error:
        envelope = _error_envelope(
            request,
            request_sha256,
            identities,
            kind="memory_error",
            error=error,
            witnessed=True,
        )
        exit_code = 70
    except _ChildWitnessedFailure as error:
        envelope = _error_envelope(
            request,
            request_sha256,
            identities,
            kind="witnessed_contract_failure",
            error=error,
            witnessed=True,
        )
        exit_code = 71
    except _ProfilerUnavailable as error:
        envelope = _error_envelope(
            request,
            request_sha256,
            identities,
            kind="profiler_observability_gap",
            error=error,
            witnessed=False,
        )
        exit_code = 0
    except _ChildObservabilityError as error:
        envelope = _error_envelope(
            request,
            request_sha256,
            identities,
            kind="environment_observability_gap",
            error=error,
            witnessed=False,
        )
        exit_code = 0
    except ValueError as error:
        envelope = _error_envelope(
            request,
            request_sha256,
            identities,
            kind="invalid_request_or_contract",
            error=error,
            witnessed=False,
        )
        exit_code = 64
    except Exception as error:
        envelope = _error_envelope(
            request,
            request_sha256,
            identities,
            kind="abnormal_child_exception",
            error=error,
            witnessed=True,
        )
        exit_code = 72
    sys.stdout.buffer.write(_canonical_json_bytes(envelope) + b"\n")
    sys.stdout.buffer.flush()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
