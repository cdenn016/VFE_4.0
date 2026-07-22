"""Ordered, atomic publication for the implemented H1/H2 gate prefix."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from verification.h1_gate import (
    EXPECTED_H1_FIXTURE_SHA256,
    FIXTURE_PATH,
    H1GateEvaluation,
    _publication_config,
    evaluate_h1,
)
from verification.h2_gate import H2GateEvaluation, evaluate_h2, h2_validation_payload
from vfe4.artifacts import build_environment, build_provenance, publish_run_directory
from vfe4.config import ResolvedConfig
from vfe4.types import GateResult, GateStatus


@dataclass(frozen=True)
class VerificationRunResult:
    gate_results: tuple[GateResult, GateResult]
    run_directory: Path

    def __post_init__(self) -> None:
        if type(self.gate_results) is not tuple or tuple(
            result.gate for result in self.gate_results
        ) != ("H1", "H2"):
            raise ValueError("gate_results must contain ordered H1 and H2 results")
        if not isinstance(self.run_directory, Path):
            raise ValueError("run_directory must be a Path")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _run_name(timestamp: str, config_hash: str) -> str:
    safe = timestamp.replace("-", "").replace(":", "").replace(".", "")
    return f"verify-h1-h2-{safe}-{config_hash[:12]}"


def _config_payload(config: ResolvedConfig) -> dict[str, object]:
    payload = json.loads(config.canonical_json)
    payload["config_sha256"] = config.config_sha256
    return payload


def _aggregate_state(results: tuple[GateResult, GateResult]) -> GateStatus:
    if any(result.status is GateStatus.FAIL for result in results):
        return GateStatus.FAIL
    if any(result.status is GateStatus.INCONCLUSIVE for result in results):
        return GateStatus.INCONCLUSIVE
    return GateStatus.PASS


def _combined_provenance(
    config: ResolvedConfig,
    h1: H1GateEvaluation,
    h2: H2GateEvaluation,
    started_utc: str,
    ended_utc: str,
) -> dict[str, object]:
    if h1.fixture_observed_sha256 != h2.fixture_observed_sha256:
        raise ValueError("ordered gates reported different fixture snapshots")
    results = (h1.result, h2.result)
    provenance = build_provenance(
        repo_root=Path(__file__).resolve().parents[1],
        fixture_expected_sha256=EXPECTED_H1_FIXTURE_SHA256,
        fixture_observed_sha256=h1.fixture_observed_sha256,
        config=config,
        started_utc=started_utc,
        ended_utc=ended_utc,
        gate_state=_aggregate_state(results).value,
    )
    provenance["gate_states"] = {
        "H1": h1.result.status.value,
        "H2": h2.result.status.value,
    }
    provenance["fixture_consumers"] = ("H1", "H2")
    return provenance


def run_verification(config: ResolvedConfig) -> VerificationRunResult:
    """Evaluate H1 then H2 from one capture and publish one atomic run."""

    canonical = _publication_config(config)
    if canonical.validation.gates != ("H1", "H2"):
        raise ValueError("run_verification requires validation.gates == ('H1', 'H2')")
    started = _utc_now()
    fixture_bytes = FIXTURE_PATH.read_bytes()
    h1 = evaluate_h1(config, fixture_bytes=fixture_bytes)
    h2 = evaluate_h2(canonical, fixture_bytes=fixture_bytes)
    results = (h1.result, h2.result)
    ended = _utc_now()
    payloads = {
        "config.json": _config_payload(canonical),
        "provenance.json": _combined_provenance(canonical, h1, h2, started, ended),
        "environment.json": build_environment(canonical),
        "validation/h1.json": h1.validation_payload,
        "validation/h2.json": h2_validation_payload(h2),
    }
    run_directory = publish_run_directory(
        canonical.artifacts.run_root,
        _run_name(started, canonical.config_sha256),
        payloads,
    )
    return VerificationRunResult(results, run_directory)


__all__ = ["VerificationRunResult", "run_verification"]
