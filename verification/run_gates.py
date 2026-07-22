"""Ordered, atomic publication for the implemented H1/H2/H3 prefixes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from verification.h1_gate import (
    EXPECTED_H1_FIXTURE_SHA256,
    FIXTURE_PATH,
    H1GateEvaluation,
    evaluate_h1,
)
from verification.h2_gate import H2GateEvaluation, evaluate_h2, h2_validation_payload
from verification.h3_gate import H3GateEvaluation, evaluate_h3, h3_validation_payload
from vfe4.artifacts import build_environment, build_provenance, publish_run_directory
from vfe4.config import ResolvedConfig, resolve_config
from vfe4.types import GateResult, GateStatus, H3GateResult
from vfe4.validation import (
    H3_COUPLED_FIXTURE_PATH,
    H3_ZERO_CONTROL_FIXTURE_PATH,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
_ALLOWED_PREFIXES = (
    ("H1",),
    ("H1", "H2"),
    ("H1", "H2", "H3"),
)


@dataclass(frozen=True)
class VerificationRunResult:
    gate_results: tuple[GateResult | H3GateResult, ...]
    run_directory: Path

    def __post_init__(self) -> None:
        if type(self.gate_results) is not tuple or not all(
            isinstance(result, (GateResult, H3GateResult))
            for result in self.gate_results
        ):
            raise ValueError("gate_results must contain immutable gate results")
        gate_names = tuple(result.gate for result in self.gate_results)
        if gate_names not in _ALLOWED_PREFIXES:
            raise ValueError("gate_results must contain an implemented ordered prefix")
        if not isinstance(self.run_directory, Path):
            raise ValueError("run_directory must be a Path")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _run_name(timestamp: str, config_hash: str, gates: tuple[str, ...]) -> str:
    safe = timestamp.replace("-", "").replace(":", "").replace(".", "")
    prefix = "-".join(gate.lower() for gate in gates)
    return f"verify-{prefix}-{safe}-{config_hash[:12]}"


def _config_payload(config: ResolvedConfig) -> dict[str, object]:
    payload = json.loads(config.canonical_json)
    payload["config_sha256"] = config.config_sha256
    return payload


def _canonical_config(config: object) -> ResolvedConfig:
    if type(config) is not ResolvedConfig:
        raise ValueError("config must have exact type ResolvedConfig")
    try:
        raw = json.loads(config.canonical_json)
        reproduced = resolve_config(raw, repo_root=REPO_ROOT)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"resolved config cannot be reproduced: {exc}") from exc
    if reproduced != config:
        raise ValueError("resolved config differs from its canonical reproduction")
    return reproduced


def _legacy_projection(config: ResolvedConfig) -> ResolvedConfig:
    """Project a validated prefix onto the unchanged H1/H2 config contract."""

    raw = json.loads(config.canonical_json)
    legacy_gates = ["H1"] if config.validation.gates == ("H1",) else ["H1", "H2"]
    raw["validation"]["gates"] = legacy_gates
    raw.pop("h3", None)
    return resolve_config(raw, repo_root=REPO_ROOT)


def _aggregate_state(
    results: tuple[GateResult | H3GateResult, ...],
) -> GateStatus:
    if any(result.status is GateStatus.FAIL for result in results):
        return GateStatus.FAIL
    if any(result.status is GateStatus.INCONCLUSIVE for result in results):
        return GateStatus.INCONCLUSIVE
    return GateStatus.PASS


def _combined_provenance(
    config: ResolvedConfig,
    h1: H1GateEvaluation,
    h2: H2GateEvaluation | None,
    h3: H3GateEvaluation | None,
    started_utc: str,
    ended_utc: str,
) -> dict[str, object]:
    if h2 is not None and h1.fixture_observed_sha256 != h2.fixture_observed_sha256:
        raise ValueError("ordered legacy gates reported different fixture snapshots")
    evaluations = (h1, *((h2,) if h2 is not None else ()), *((h3,) if h3 is not None else ()))
    results = tuple(evaluation.result for evaluation in evaluations)
    provenance = build_provenance(
        repo_root=REPO_ROOT,
        fixture_expected_sha256=EXPECTED_H1_FIXTURE_SHA256,
        fixture_observed_sha256=h1.fixture_observed_sha256,
        config=config,
        started_utc=started_utc,
        ended_utc=ended_utc,
        gate_state=_aggregate_state(results).value,
    )
    provenance["gate_states"] = {
        result.gate: result.status.value for result in results
    }
    provenance["fixture_consumers"] = tuple(
        result.gate for result in results if result.gate in ("H1", "H2")
    )
    if h3 is not None:
        hashes = h3.fixture_hashes
        provenance["fixture_hashes"] = {
            "h1-v1": {
                "expected_sha256": EXPECTED_H1_FIXTURE_SHA256,
                "observed_sha256": h1.fixture_observed_sha256,
                "hash_domain": "raw_fixture_bytes",
            },
            "h3-coupled-v1": {
                "expected_sha256": hashes.coupled_expected_sha256,
                "observed_sha256": hashes.coupled_observed_sha256,
                "hash_domain": "raw_fixture_bytes",
            },
            "h3-zero-control-v1": {
                "expected_sha256": hashes.zero_control_expected_sha256,
                "observed_sha256": hashes.zero_control_observed_sha256,
                "hash_domain": "raw_fixture_bytes",
            },
        }
        provenance["gate_fixture_consumers"] = {
            "H1": ("h1-v1",),
            "H2": ("h1-v1",),
            "H3": ("h3-coupled-v1", "h3-zero-control-v1"),
        }
        canonical_payload = json.loads(config.canonical_json)
        provenance["h3_profile"] = canonical_payload["h3"]
    return provenance


def run_verification(config: ResolvedConfig) -> VerificationRunResult:
    """Evaluate one implemented prefix from one capture set and publish once."""

    canonical = _canonical_config(config)
    gates = canonical.validation.gates
    if gates not in _ALLOWED_PREFIXES:
        raise ValueError("run_verification requires an implemented ordered gate prefix")
    legacy = _legacy_projection(canonical)
    started = _utc_now()

    h1_bytes = FIXTURE_PATH.read_bytes()
    coupled_bytes: bytes | None = None
    zero_control_bytes: bytes | None = None
    if gates == ("H1", "H2", "H3"):
        coupled_bytes = H3_COUPLED_FIXTURE_PATH.read_bytes()
        zero_control_bytes = H3_ZERO_CONTROL_FIXTURE_PATH.read_bytes()

    h1 = evaluate_h1(legacy, fixture_bytes=h1_bytes)
    h2: H2GateEvaluation | None = None
    h3: H3GateEvaluation | None = None
    results: list[GateResult | H3GateResult] = [h1.result]
    validation_payloads: dict[str, object] = {
        "validation/h1.json": h1.validation_payload,
    }
    if "H2" in gates:
        h2 = evaluate_h2(legacy, fixture_bytes=h1_bytes)
        results.append(h2.result)
        validation_payloads["validation/h2.json"] = h2_validation_payload(h2)
    if "H3" in gates:
        if coupled_bytes is None or zero_control_bytes is None:
            raise RuntimeError("H3 fixture capture is unavailable")
        h3 = evaluate_h3(
            canonical,
            coupled_fixture_bytes=coupled_bytes,
            zero_control_fixture_bytes=zero_control_bytes,
        )
        results.append(h3.result)
        validation_payloads["validation/h3.json"] = h3_validation_payload(h3)

    frozen_results = tuple(results)
    ended = _utc_now()
    payloads = {
        "config.json": _config_payload(canonical),
        "provenance.json": _combined_provenance(
            canonical,
            h1,
            h2,
            h3,
            started,
            ended,
        ),
        "environment.json": build_environment(canonical),
        **validation_payloads,
    }
    run_directory = publish_run_directory(
        canonical.artifacts.run_root,
        _run_name(started, canonical.config_sha256, gates),
        payloads,
    )
    return VerificationRunResult(frozen_results, run_directory)


__all__ = ["VerificationRunResult", "run_verification"]
