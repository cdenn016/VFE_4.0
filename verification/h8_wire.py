"""Stable, standard-library-only authority for the H8 child-v2 wire.

The executable child deliberately retains local mirrors of this contract so
that drift is observable.  Recovery and drift reporting must not depend on
those mirrors: otherwise a drifted child can no longer produce an envelope
that the parent can parse.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from types import MappingProxyType


H8_CHILD_MODULE = "verification.h8_child"
H8_CHILD_SCHEMA_VERSION = "h8-child-v2"
H8_CHILD_IDENTITY_ENV = "VFE4_H8_CHILD_IDENTITIES_JSON"
H8_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
H8_THREAD_ENVIRONMENT_VALUE = "1"
H8_THREAD_ENVIRONMENT_ITEMS = tuple(
    (name, H8_THREAD_ENVIRONMENT_VALUE) for name in H8_THREAD_ENVIRONMENT
)
H8_TORCH_NUM_THREADS = 1
H8_TORCH_NUM_INTEROP_THREADS = 1
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
H8_CHILD_IDENTITY_KEYS = ("hardware", "affinity", "thread", "blas")
H8_CHILD_ERROR_KEYS = ("kind", "message", "witnessed_violation")
H8_CHILD_MODES = ("production", "profiler", "negative_control")
H8_NEGATIVE_CONTROL_IDS = (
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
H8_PRODUCTION_SAMPLE_SEED_PAIRS = (
    (20260721, 20261721),
    (20260722, 20261722),
    (20260723, 20261723),
)
H8_PRODUCTION_SEEDS = tuple(
    problem_seed
    for problem_seed, _sample_seed in H8_PRODUCTION_SAMPLE_SEED_PAIRS
)
H8_PRODUCTION_SAMPLE_SEEDS: Mapping[int, int] = MappingProxyType(
    dict(H8_PRODUCTION_SAMPLE_SEED_PAIRS)
)
H8_COLD_REPETITIONS = 5
H8_LAYOUT_HORIZON = 128
H8_LAYOUT_D_Z = 20
H8_LAYOUT_D_M = 20
H8_MAX_RHS_WIDTH = H8_LAYOUT_D_Z + H8_LAYOUT_D_M
H8_SAMPLE_WIDTH = 1
H8_MAX_STORAGE_SCALARS = 411_200
H8_OFFBAND_FILL_LIMIT = 0
H8_FORBIDDEN_ATTEMPT_LIMIT = 0
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
H8_MAX_PROCESS_INCREMENTAL_BYTES = 128 * 1024 * 1024
H8_MAX_TORCH_POPULATION_BYTES = 64 * 1024 * 1024
H8_MAX_SECONDS = 60.0
H8_MIN_CHOLESKY_PIVOT = 1e-8
H8_PROFILER_TORCH_VERSION = "2.9.1"
H8_PROFILER_MEMORY_SOURCE_SHA256 = (
    "b80b4d5b58e91d581b18082c462ec7f088ec6b46ea50a1a62e2714d517a6a1b1"
)
H8_PROFILER_SOURCE_SHA256 = (
    "2c35f649219fb912728819b7dc0be5a5f1bd54c1efcd9502b62d976aeb278d22"
)
H8_PROFILER_API_CONTRACT_SHA256 = (
    "161a78f04c26fba19bb01ba6417f2cf8c00730ebeb8d007a4af0f4da433ba043"
)
H8_PROFILER_INVOCATION_ITEMS = (
    ("activities", ("CPU",)),
    ("profile_memory", True),
    ("record_shapes", True),
    ("with_stack", True),
)
H8_LOCAL_CONTRACT_DRIFT_KIND = "child_local_contract_drift"

_HEX = frozenset("0123456789abcdef")


def canonical_json_bytes(value: object) -> bytes:
    """Return strict canonical UTF-8 JSON without a trailing newline."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_line(value: object) -> bytes:
    """Return one strict canonical JSON line."""

    return canonical_json_bytes(value) + b"\n"


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


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in _HEX for character in value)
    )


def _validate_request(value: object) -> dict[str, object]:
    if (
        type(value) is not dict
        or set(value) != set(H8_CHILD_REQUEST_KEYS)
    ):
        raise ValueError("request does not use the exact H8ChildRequest schema")
    checked = {name: value[name] for name in H8_CHILD_REQUEST_KEYS}
    mode = checked["mode"]
    seed = checked["seed"]
    repetition = checked["repetition"]
    control_id = checked["control_id"]
    if mode not in H8_CHILD_MODES:
        raise ValueError("request mode is outside the frozen union")
    if type(seed) is not int or seed <= 0:
        raise ValueError("request seed must be a positive integer")
    if not _is_sha256(checked["config_sha256"]) or not _is_sha256(
        checked["protocol_sha256"]
    ):
        raise ValueError("request hashes must be lowercase SHA-256 values")
    if mode == "production":
        if (
            seed not in H8_PRODUCTION_SAMPLE_SEEDS
            or type(repetition) is not int
            or not 0 <= repetition < H8_COLD_REPETITIONS
            or control_id is not None
        ):
            raise ValueError("production request identity is invalid")
    elif mode == "profiler":
        if (
            seed not in H8_PRODUCTION_SAMPLE_SEEDS
            or repetition is not None
            or control_id is not None
        ):
            raise ValueError("profiler request identity is invalid")
    elif (
        seed not in H8_PRODUCTION_SAMPLE_SEEDS
        or repetition is not None
        or control_id not in H8_NEGATIVE_CONTROL_IDS
    ):
        raise ValueError("negative-control request identity is invalid")
    return checked


def recover_h8_child_request(
    raw: bytes,
) -> tuple[dict[str, object], str]:
    """Recover one exact canonical request independently of child mirrors."""

    if (
        type(raw) is not bytes
        or not raw.endswith(b"\n")
        or raw.count(b"\n") != 1
        or not raw[:-1]
    ):
        raise ValueError("stdin must contain exactly one canonical JSON line")
    try:
        value = json.loads(
            raw[:-1].decode("utf-8", errors="strict"),
            parse_constant=lambda constant: _raise_nonfinite(constant),
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("stdin is not one JSON object") from error
    checked = _validate_request(value)
    if canonical_json_bytes(value) + b"\n" != raw:
        raise ValueError("request JSON is not canonical")
    return checked, hashlib.sha256(raw[:-1]).hexdigest()


def _identity_record(
    kind: str,
    payload: Mapping[str, object],
) -> dict[str, object]:
    body = {"kind": kind, **dict(payload)}
    return {
        **body,
        "sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
    }


def _unavailable_identities(message: str) -> dict[str, dict[str, object]]:
    return {
        kind: _identity_record(kind, {"observability_error": message})
        for kind in H8_CHILD_IDENTITY_KEYS
    }


def build_h8_local_contract_drift_envelope(
    request: Mapping[str, object],
    request_sha256: str,
    detail: str,
) -> dict[str, object]:
    """Build the stable parent-parseable INCONCLUSIVE drift envelope."""

    checked = _validate_request(dict(request))
    if not _is_sha256(request_sha256):
        raise ValueError("request_sha256 must be a lowercase SHA-256")
    if type(detail) is not str or not detail:
        raise ValueError("local-contract drift detail must be nonempty")
    identities = _unavailable_identities(
        "child-local contract drift prevented runtime identity collection"
    )
    error = {
        "kind": H8_LOCAL_CONTRACT_DRIFT_KIND,
        "message": detail,
        "witnessed_violation": False,
    }
    if tuple(error) != H8_CHILD_ERROR_KEYS:
        raise RuntimeError("stable H8 child error inventory drifted")
    envelope = {
        "schema_version": H8_CHILD_SCHEMA_VERSION,
        "mode": checked["mode"],
        "seed": checked["seed"],
        "repetition": checked["repetition"],
        "control_id": checked["control_id"],
        "request_sha256": request_sha256,
        "config_sha256": checked["config_sha256"],
        "protocol_sha256": checked["protocol_sha256"],
        "status": "inconclusive",
        "obligations": [H8_LOCAL_CONTRACT_DRIFT_KIND],
        "identities": identities,
        "result": None,
        "control": None,
        "error": error,
    }
    if tuple(envelope) != H8_CHILD_ENVELOPE_KEYS:
        raise RuntimeError("stable H8 child envelope inventory drifted")
    return envelope


def canonical_h8_local_contract_drift_line(
    request: Mapping[str, object],
    request_sha256: str,
    detail: str,
) -> bytes:
    """Serialize the stable local-drift envelope as one canonical line."""

    return canonical_json_line(
        build_h8_local_contract_drift_envelope(
            request,
            request_sha256,
            detail,
        )
    )


__all__ = [
    "H8_CHILD_ENVELOPE_KEYS",
    "H8_CHILD_ERROR_KEYS",
    "H8_CHILD_IDENTITY_ENV",
    "H8_CHILD_IDENTITY_KEYS",
    "H8_CHILD_MODES",
    "H8_CHILD_MODULE",
    "H8_CHILD_REQUEST_KEYS",
    "H8_CHILD_RESULT_KEYS",
    "H8_CHILD_SCHEMA_VERSION",
    "H8_COLD_REPETITIONS",
    "H8_LOCAL_CONTRACT_DRIFT_KIND",
    "H8_MAX_PROCESS_INCREMENTAL_BYTES",
    "H8_MAX_RHS_WIDTH",
    "H8_MAX_SECONDS",
    "H8_MAX_STORAGE_SCALARS",
    "H8_MAX_TORCH_POPULATION_BYTES",
    "H8_MIN_CHOLESKY_PIVOT",
    "H8_NEGATIVE_CONTROL_IDS",
    "H8_OFFBAND_FILL_LIMIT",
    "H8_FORBIDDEN_ATTEMPT_LIMIT",
    "H8_LAYOUT_D_M",
    "H8_LAYOUT_D_Z",
    "H8_LAYOUT_HORIZON",
    "H8_PRODUCTION_SAMPLE_SEED_PAIRS",
    "H8_PRODUCTION_SAMPLE_SEEDS",
    "H8_PRODUCTION_SEEDS",
    "H8_PROFILER_API_CONTRACT_SHA256",
    "H8_PROFILER_INVOCATION_ITEMS",
    "H8_PROFILER_MEMORY_SOURCE_SHA256",
    "H8_PROFILER_SOURCE_SHA256",
    "H8_PROFILER_TORCH_VERSION",
    "H8_REQUIRED_OPERATIONS",
    "H8_SAMPLE_WIDTH",
    "H8_THREAD_ENVIRONMENT",
    "H8_THREAD_ENVIRONMENT_ITEMS",
    "H8_THREAD_ENVIRONMENT_VALUE",
    "H8_TORCH_NUM_INTEROP_THREADS",
    "H8_TORCH_NUM_THREADS",
    "build_h8_local_contract_drift_envelope",
    "canonical_h8_local_contract_drift_line",
    "canonical_json_bytes",
    "canonical_json_line",
    "recover_h8_child_request",
]
