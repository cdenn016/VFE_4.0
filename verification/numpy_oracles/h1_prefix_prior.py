"""Independent NumPy oracle for the bounded H1 prefix-prior fixture.

This module intentionally has no dependency on :mod:`vfe4`.  It parses the
data-only wrapper, reconstructs normalized source priors from the frozen
hex-float parameters, and delegates only the derived ordinary-H1 calculation
to the existing independent NumPy H1 oracle.
"""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from .h1_elbo import (
    H1IdentityRecord,
    IndependentTermRecord,
    h1_evidence_and_posterior_kl,
    h1_local_diagnostics,
)


_ROOT_FIELDS = frozenset(
    {
        "fixture_schema_version",
        "fixture_id",
        "base_h1_fixture",
        "structure",
        "active_case_id",
        "prefix_cases",
        "current_target_token_id",
        "earlier_latents",
        "prefix_prior_parameters",
        "generative_factor_schema",
        "quadrature",
    }
)
_SCHEMA_FIELDS = frozenset(
    {
        "schema_version",
        "prior_variant",
        "conditioning_inputs",
        "forbidden_inputs",
        "target_blind",
        "receiver_t",
        "source_banks",
        "context_dim",
        "token_pooling",
        "latent_projection_policy",
        "support_mask_application",
        "normalization",
    }
)
_EXPECTED_SCHEMA = {
    "schema_version": "h1-prefix-prior-generative-factor-v1",
    "prior_variant": "prefix_conditioned",
    "conditioning_inputs": ["prior_token_ids", "earlier_latents"],
    "forbidden_inputs": ["current_target", "future_tokens", "recognition_state"],
    "target_blind": True,
    "receiver_t": 2,
    "source_banks": ["state", "model"],
    "context_dim": 1,
    "token_pooling": "mean",
    "latent_projection_policy": "exact_zero",
    "support_mask_application": "before_normalization",
    "normalization": "masked_log_softmax_from_parents",
}
_EXPECTED_BASE_SHA256 = (
    "388e38cc8c16d8b5e2c61919c1e712a134d88fb0bbd8ec1f2939b9859c9a583b"
)


@dataclass(frozen=True)
class IndependentPrefixCase:
    case_id: str
    prefix_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class IndependentH1PrefixPriorFixture:
    raw_sha256: str
    fixture_id: str
    base_fixture_relative_path: str
    base_fixture_sha256: str
    structure: tuple[int, int, int, int]
    state_parent_sets: tuple[tuple[int, ...], ...]
    model_parent_sets: tuple[tuple[int, ...], ...]
    active_case_id: str
    prefix_cases: tuple[IndependentPrefixCase, ...]
    current_target_token_id: int
    earlier_latents: tuple[tuple[float, ...], ...]
    context_dim: int
    token_embedding: tuple[tuple[float, ...], ...]
    state_latent_projection: tuple[tuple[float, ...], ...]
    model_latent_projection: tuple[tuple[float, ...], ...]
    state_parent_keys_t2: tuple[tuple[float, ...], ...]
    model_parent_keys_t2: tuple[tuple[float, ...], ...]
    state_biases_t2: tuple[float, ...]
    model_biases_t2: tuple[float, ...]
    generative_factor_schema: MappingProxyType[str, object]
    generative_factor_schema_bytes: bytes
    generative_factor_schema_sha256: str
    quadrature_order: int
    convergence_check_order: int
    maximum_convergence_estimate: float


@dataclass(frozen=True)
class PrefixPriorProbabilities:
    case_id: str
    prefix_token_ids: tuple[int, ...]
    state_log_probabilities: tuple[float, float]
    model_log_probabilities: tuple[float, float]
    state_probabilities: tuple[float, float]
    model_probabilities: tuple[float, float]


@dataclass(frozen=True)
class H1PrefixPriorOracleRecord:
    probabilities: PrefixPriorProbabilities
    derived_h1_fixture_sha256: str
    local_terms: IndependentTermRecord
    identity: H1IdentityRecord


def _mapping(value: object, fields: frozenset[str], name: str) -> dict[str, Any]:
    if type(value) is not dict or frozenset(value) != fields:
        raise ValueError(f"{name} fields must equal {sorted(fields)!r}")
    return value


def _sequence(value: object, length: int, name: str) -> list[Any]:
    if type(value) is not list or len(value) != length:
        raise ValueError(f"{name} must be a list of length {length}")
    return value


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _hex_float(value: object, name: str) -> float:
    if type(value) is not str:
        raise ValueError(f"{name} must be a canonical hexadecimal float string")
    try:
        parsed = float.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{name} is not a hexadecimal float") from exc
    if not math.isfinite(parsed) or parsed.hex() != value:
        raise ValueError(f"{name} must be a canonical finite hexadecimal float")
    return parsed


def _hex_vector(value: object, length: int, name: str) -> tuple[float, ...]:
    return tuple(
        _hex_float(item, f"{name}[{index}]")
        for index, item in enumerate(_sequence(value, length, name))
    )


def _hex_matrix(
    value: object, rows: int, columns: int, name: str
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        _hex_vector(row, columns, f"{name}[{index}]")
        for index, row in enumerate(_sequence(value, rows, name))
    )


def parse_h1_prefix_prior_fixture(
    fixture_bytes: bytes,
) -> IndependentH1PrefixPriorFixture:
    """Parse one immutable raw fixture snapshot with strict field inventories."""

    if type(fixture_bytes) is not bytes:
        raise ValueError("fixture_bytes must be immutable bytes")
    try:
        root = _mapping(
            json.loads(fixture_bytes.decode("utf-8")), _ROOT_FIELDS, "fixture"
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"prefix-prior fixture is not valid UTF-8 JSON: {exc}") from exc
    if (
        root["fixture_schema_version"] != "h1-prefix-prior-fixture-v1"
        or root["fixture_id"] != "h1-prefix-prior-v1"
    ):
        raise ValueError("prefix-prior fixture identity is unsupported")

    base = _mapping(
        root["base_h1_fixture"],
        frozenset({"relative_path", "raw_sha256"}),
        "base_h1_fixture",
    )
    if (
        base["relative_path"] != "vfe4/validation/fixtures/h1_v1.json"
        or base["raw_sha256"] != _EXPECTED_BASE_SHA256
    ):
        raise ValueError("base H1 fixture identity is not frozen")

    structure = _mapping(
        root["structure"],
        frozenset(
            {
                "horizon",
                "d_z",
                "d_m",
                "vocabulary_size",
                "state_parent_sets",
                "model_parent_sets",
            }
        ),
        "structure",
    )
    dimensions = (
        structure["horizon"],
        structure["d_z"],
        structure["d_m"],
        structure["vocabulary_size"],
    )
    if dimensions != (2, 1, 1, 3):
        raise ValueError("prefix-prior fixture must use T=2, d_z=d_m=1, V=3")
    state_parent_sets = tuple(
        tuple(row)
        for row in _sequence(structure["state_parent_sets"], 2, "state_parent_sets")
    )
    model_parent_sets = tuple(
        tuple(row)
        for row in _sequence(structure["model_parent_sets"], 2, "model_parent_sets")
    )
    if state_parent_sets != ((0,), (0, 1)) or model_parent_sets != (
        (0,),
        (0, 1),
    ):
        raise ValueError("both source banks must expose both causal parents at t=2")

    cases_raw = _sequence(root["prefix_cases"], 2, "prefix_cases")
    cases: list[IndependentPrefixCase] = []
    for index, raw_case in enumerate(cases_raw):
        case = _mapping(
            raw_case,
            frozenset({"case_id", "prefix_token_ids"}),
            f"prefix_cases[{index}]",
        )
        case_id = case["case_id"]
        token_ids = tuple(_sequence(case["prefix_token_ids"], 1, "prefix_token_ids"))
        if (
            type(case_id) is not str
            or not case_id
            or any(type(token) is not int or not 0 <= token < 3 for token in token_ids)
        ):
            raise ValueError("prefix cases require named one-token prior histories")
        cases.append(IndependentPrefixCase(case_id, token_ids))
    if (
        tuple(case.case_id for case in cases)
        != ("prefix-token-0", "prefix-token-2")
        or len({case.prefix_token_ids for case in cases}) != 2
        or root["active_case_id"] != "prefix-token-0"
    ):
        raise ValueError("the exact two-prefix inventory is required")
    current_target = root["current_target_token_id"]
    if type(current_target) is not int or current_target != 1:
        raise ValueError("the frozen current target must be zero-based token 1")

    earlier_latents = _hex_matrix(root["earlier_latents"], 2, 1, "earlier_latents")
    parameters = _mapping(
        root["prefix_prior_parameters"],
        frozenset(
            {
                "context_dim",
                "token_embedding",
                "state_latent_projection",
                "model_latent_projection",
                "state_parent_keys_t2",
                "model_parent_keys_t2",
                "state_biases_t2",
                "model_biases_t2",
            }
        ),
        "prefix_prior_parameters",
    )
    if parameters["context_dim"] != 1:
        raise ValueError("the bounded fixture requires context_dim=1")
    token_embedding = _hex_matrix(
        parameters["token_embedding"], 3, 1, "token_embedding"
    )
    state_projection = _hex_matrix(
        parameters["state_latent_projection"], 1, 1, "state_latent_projection"
    )
    model_projection = _hex_matrix(
        parameters["model_latent_projection"], 1, 1, "model_latent_projection"
    )
    if (
        earlier_latents != ((0.0,), (0.0,))
        or state_projection != ((0.0,),)
        or model_projection != ((0.0,),)
    ):
        raise ValueError("the bounded prefix-prior fixture requires zero latent inputs")
    state_keys = _hex_matrix(
        parameters["state_parent_keys_t2"], 2, 1, "state_parent_keys_t2"
    )
    model_keys = _hex_matrix(
        parameters["model_parent_keys_t2"], 2, 1, "model_parent_keys_t2"
    )
    state_biases = _hex_vector(parameters["state_biases_t2"], 2, "state_biases_t2")
    model_biases = _hex_vector(parameters["model_biases_t2"], 2, "model_biases_t2")

    schema = _mapping(root["generative_factor_schema"], _SCHEMA_FIELDS, "schema")
    if schema != _EXPECTED_SCHEMA:
        raise ValueError("generative-factor schema differs from the frozen target-blind form")
    schema_bytes = _canonical_json_bytes(schema)

    quadrature = _mapping(
        root["quadrature"],
        frozenset(
            {"order", "convergence_check_order", "maximum_convergence_estimate"}
        ),
        "quadrature",
    )
    if quadrature != {
        "order": 21,
        "convergence_check_order": 17,
        "maximum_convergence_estimate": 1e-9,
    }:
        raise ValueError("quadrature policy differs from calibrated H1")

    fixture = IndependentH1PrefixPriorFixture(
        raw_sha256=hashlib.sha256(fixture_bytes).hexdigest(),
        fixture_id="h1-prefix-prior-v1",
        base_fixture_relative_path=base["relative_path"],
        base_fixture_sha256=base["raw_sha256"],
        structure=(2, 1, 1, 3),
        state_parent_sets=state_parent_sets,
        model_parent_sets=model_parent_sets,
        active_case_id=root["active_case_id"],
        prefix_cases=tuple(cases),
        current_target_token_id=current_target,
        earlier_latents=earlier_latents,
        context_dim=1,
        token_embedding=token_embedding,
        state_latent_projection=state_projection,
        model_latent_projection=model_projection,
        state_parent_keys_t2=state_keys,
        model_parent_keys_t2=model_keys,
        state_biases_t2=state_biases,
        model_biases_t2=model_biases,
        generative_factor_schema=MappingProxyType(dict(schema)),
        generative_factor_schema_bytes=schema_bytes,
        generative_factor_schema_sha256=hashlib.sha256(schema_bytes).hexdigest(),
        quadrature_order=21,
        convergence_check_order=17,
        maximum_convergence_estimate=1e-9,
    )
    first = prefix_prior_probabilities(fixture, case_id="prefix-token-0")
    second = prefix_prior_probabilities(fixture, case_id="prefix-token-2")
    if (
        first.state_probabilities == second.state_probabilities
        or first.model_probabilities == second.model_probabilities
    ):
        raise ValueError("the two frozen prefixes must produce distinct priors")
    return fixture


def load_h1_prefix_prior_fixture(path: Path) -> IndependentH1PrefixPriorFixture:
    if not isinstance(path, Path):
        raise ValueError("path must be a pathlib.Path")
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"prefix-prior fixture is unreadable: {exc}") from exc
    return parse_h1_prefix_prior_fixture(content)


def _case(
    fixture: IndependentH1PrefixPriorFixture,
    *,
    case_id: str | None,
    use_current_target: bool,
) -> tuple[str, tuple[int, ...]]:
    if type(fixture) is not IndependentH1PrefixPriorFixture:
        raise ValueError("fixture must be an independent prefix-prior fixture")
    if type(use_current_target) is not bool:
        raise ValueError("use_current_target must be Boolean")
    if use_current_target:
        if case_id is not None:
            raise ValueError("case_id and use_current_target are mutually exclusive")
        return "negative-current-target", (fixture.current_target_token_id,)
    selected = fixture.active_case_id if case_id is None else case_id
    if type(selected) is not str:
        raise ValueError("case_id must be a string or None")
    for candidate in fixture.prefix_cases:
        if candidate.case_id == selected:
            return candidate.case_id, candidate.prefix_token_ids
    raise ValueError(f"unknown prefix case: {selected}")


def _log_softmax(values: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    maximum = float(np.max(values))
    shifted = values - maximum
    log_normalizer = maximum + math.log(math.fsum(float(np.exp(item)) for item in shifted))
    logs = tuple(float(item - log_normalizer) for item in values)
    probabilities = tuple(float(math.exp(item)) for item in logs)
    return (logs[0], logs[1]), (probabilities[0], probabilities[1])


def prefix_prior_probabilities(
    fixture: IndependentH1PrefixPriorFixture,
    *,
    case_id: str | None = None,
    use_current_target: bool = False,
) -> PrefixPriorProbabilities:
    """Compute the two normalized source rows using independent NumPy algebra."""

    selected_id, token_ids = _case(
        fixture, case_id=case_id, use_current_target=use_current_target
    )
    token_context = np.mean(
        np.asarray(
            [fixture.token_embedding[token_id] for token_id in token_ids],
            dtype=np.float64,
        ),
        axis=0,
    )
    latent_history = np.asarray(fixture.earlier_latents, dtype=np.float64)
    state_projection = np.asarray(fixture.state_latent_projection, dtype=np.float64)
    model_projection = np.asarray(fixture.model_latent_projection, dtype=np.float64)
    state_context = token_context + np.mean(
        latent_history @ state_projection.T, axis=0
    )
    model_context = token_context + np.mean(
        latent_history @ model_projection.T, axis=0
    )
    state_logits = (
        np.asarray(fixture.state_parent_keys_t2, dtype=np.float64) @ state_context
        + np.asarray(fixture.state_biases_t2, dtype=np.float64)
    )
    model_logits = (
        np.asarray(fixture.model_parent_keys_t2, dtype=np.float64) @ model_context
        + np.asarray(fixture.model_biases_t2, dtype=np.float64)
    )
    state_logs, state_probabilities = _log_softmax(state_logits)
    model_logs, model_probabilities = _log_softmax(model_logits)
    return PrefixPriorProbabilities(
        selected_id,
        token_ids,
        state_logs,
        model_logs,
        state_probabilities,
        model_probabilities,
    )


def derived_h1_fixture_bytes(
    fixture: IndependentH1PrefixPriorFixture,
    *,
    base_fixture_bytes: bytes,
    case_id: str | None = None,
    use_current_target: bool = False,
) -> bytes:
    """Create canonical ordinary-H1 fixture bytes with the selected priors."""

    if type(base_fixture_bytes) is not bytes:
        raise ValueError("base_fixture_bytes must be immutable bytes")
    if hashlib.sha256(base_fixture_bytes).hexdigest() != fixture.base_fixture_sha256:
        raise ValueError("base H1 fixture bytes do not match the frozen raw hash")
    try:
        payload = json.loads(base_fixture_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"base H1 fixture is not valid UTF-8 JSON: {exc}") from exc
    if type(payload) is not dict or payload.get("fixture_id") != "h1-v1":
        raise ValueError("base H1 fixture payload has the wrong identity")
    probabilities = prefix_prior_probabilities(
        fixture, case_id=case_id, use_current_target=use_current_target
    )
    payload["state_source_priors"] = [
        [1.0],
        list(probabilities.state_probabilities),
    ]
    payload["model_source_priors"] = [
        [1.0],
        list(probabilities.model_probabilities),
    ]
    return _canonical_json_bytes(payload)


def evaluate_h1_prefix_prior_oracle(
    fixture_bytes: bytes,
    *,
    base_fixture_bytes: bytes,
    case_id: str | None = None,
    use_current_target: bool = False,
) -> H1PrefixPriorOracleRecord:
    """Evaluate local terms and evidence-minus-posterior-KL independently."""

    fixture = parse_h1_prefix_prior_fixture(fixture_bytes)
    probabilities = prefix_prior_probabilities(
        fixture, case_id=case_id, use_current_target=use_current_target
    )
    derived = derived_h1_fixture_bytes(
        fixture,
        base_fixture_bytes=base_fixture_bytes,
        case_id=case_id,
        use_current_target=use_current_target,
    )
    with tempfile.TemporaryDirectory(prefix="vfe4-h1-prefix-oracle-") as temporary:
        path = Path(temporary) / "h1_v1.json"
        path.write_bytes(derived)
        local = h1_local_diagnostics(
            path,
            quadrature_order=fixture.quadrature_order,
            convergence_check_order=fixture.convergence_check_order,
        )
        identity = h1_evidence_and_posterior_kl(
            path,
            quadrature_order=fixture.quadrature_order,
            convergence_check_order=fixture.convergence_check_order,
        )
    return H1PrefixPriorOracleRecord(
        probabilities,
        hashlib.sha256(derived).hexdigest(),
        local,
        identity,
    )


__all__ = [
    "H1PrefixPriorOracleRecord",
    "IndependentH1PrefixPriorFixture",
    "IndependentPrefixCase",
    "PrefixPriorProbabilities",
    "derived_h1_fixture_bytes",
    "evaluate_h1_prefix_prior_oracle",
    "load_h1_prefix_prior_fixture",
    "parse_h1_prefix_prior_fixture",
    "prefix_prior_probabilities",
]
