"""Ordered, atomic publication for the implemented H1/H2/H3 prefixes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from verification.h1_gate import (
    EXPECTED_H1_FIXTURE_SHA256,
    FIXTURE_PATH,
    H1GateEvaluation,
    evaluate_h1,
)
from verification.h2_gate import H2GateEvaluation, evaluate_h2, h2_validation_payload
from verification.h3_gate import H3GateEvaluation, evaluate_h3, h3_validation_payload
from verification.h4_gate import (
    H4GateEvaluation,
    evaluate_h4,
    h4_validation_artifact,
    h4_validation_payload,
)
from verification.h5_gate import (
    H5GateEvaluation,
    H5GateResult,
    H5PreflightPhase,
    evaluate_h5,
    h5_update_binding_preimages,
    h5_validation_payload,
)
from vfe4.artifacts import (
    build_environment,
    build_provenance,
    canonical_json_bytes,
    publish_run_directory,
    source_candidate_sha256,
)
from vfe4.config import ResolvedConfig, resolve_config
from vfe4.types import GateResult, GateStatus, H3GateResult, H4GateResult
from vfe4.validation import (
    H3_COUPLED_FIXTURE_PATH,
    H3_ZERO_CONTROL_FIXTURE_PATH,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
H5_UPDATE_SPEC_FIXTURE_PATH = (
    REPO_ROOT / "vfe4" / "validation" / "fixtures" / "h5_conditional_update_v1.json"
)
_ALLOWED_PREFIXES = (
    ("H1",),
    ("H1", "H2"),
    ("H1", "H2", "H3"),
    ("H1", "H2", "H3", "H4", "H5"),
)
_PREDICTION_CORRECTNESS_GATES: tuple[
    Literal["H1", "H2", "H3", "H5"], ...
] = ("H1", "H2", "H3", "H5")
_LOWER_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class VerificationRunResult:
    gate_results: tuple[GateResult | H3GateResult | H4GateResult | H5GateResult, ...]
    run_directory: Path
    prediction_correctness_artifacts: tuple[
        tuple[Literal["H1", "H2", "H3", "H5"], Path, str], ...
    ] = ()

    def __post_init__(self) -> None:
        if type(self.gate_results) is not tuple or not all(
            isinstance(result, (GateResult, H3GateResult, H4GateResult, H5GateResult))
            for result in self.gate_results
        ):
            raise ValueError("gate_results must contain immutable gate results")
        gate_names = tuple(result.gate for result in self.gate_results)
        if gate_names not in _ALLOWED_PREFIXES:
            raise ValueError("gate_results must contain an implemented ordered prefix")
        if not isinstance(self.run_directory, Path):
            raise ValueError("run_directory must be a Path")
        if type(self.prediction_correctness_artifacts) is not tuple:
            raise ValueError("prediction_correctness_artifacts must be a tuple")
        correctness_gates: list[str] = []
        correctness_roots: list[Path] = []
        for reference in self.prediction_correctness_artifacts:
            if type(reference) is not tuple or len(reference) != 3:
                raise ValueError(
                    "prediction correctness references must be "
                    "(gate, root, manifest_sha256) tuples"
                )
            gate, root, manifest_sha256 = reference
            if gate not in _PREDICTION_CORRECTNESS_GATES:
                raise ValueError("prediction correctness gate is unsupported")
            if not isinstance(root, Path):
                raise ValueError("prediction correctness root must be a Path")
            if (
                type(manifest_sha256) is not str
                or len(manifest_sha256) != 64
                or any(character not in _LOWER_HEX for character in manifest_sha256)
            ):
                raise ValueError(
                    "prediction correctness manifest SHA-256 must be lowercase 64-hex"
                )
            correctness_gates.append(gate)
            correctness_roots.append(root)
        if correctness_gates and tuple(correctness_gates) != _PREDICTION_CORRECTNESS_GATES:
            raise ValueError(
                "prediction correctness references must contain H1, H2, H3, H5 "
                "in frozen order"
            )
        if correctness_gates and gate_names != ("H1", "H2", "H3", "H4", "H5"):
            raise ValueError(
                "prediction correctness references require the full H1--H5 result"
            )
        if len(set(correctness_roots)) != len(correctness_roots):
            raise ValueError("prediction correctness roots must be distinct")


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


def _raw_config_payload(config: ResolvedConfig) -> tuple[dict[str, object], bytes]:
    """Recover the exact canonical config bytes consumed by one gate."""

    if type(config) is not ResolvedConfig:
        raise ValueError("prediction correctness config must be a ResolvedConfig")
    try:
        payload = json.loads(config.canonical_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("prediction correctness config JSON is invalid") from exc
    if type(payload) is not dict:
        raise ValueError("prediction correctness config must be one JSON object")
    raw_bytes = canonical_json_bytes(payload)
    if raw_bytes != config.canonical_json.encode("utf-8"):
        raise ValueError(
            "prediction correctness config bytes differ from canonical_json"
        )
    if hashlib.sha256(raw_bytes).hexdigest() != config.config_sha256:
        raise ValueError(
            "prediction correctness raw config SHA-256 differs from config"
        )
    return payload, raw_bytes


def _ordered_prediction_inputs(
    value: object,
    *,
    name: str,
) -> dict[str, object]:
    if (
        type(value) is not tuple
        or any(type(item) is not tuple or len(item) != 2 for item in value)
    ):
        raise ValueError(f"{name} must be an ordered tuple of gate/value pairs")
    gates = tuple(item[0] for item in value)
    if gates != _PREDICTION_CORRECTNESS_GATES:
        raise ValueError(f"{name} must contain H1, H2, H3, H5 in frozen order")
    return {item[0]: item[1] for item in value}


def _prediction_correctness_run_name(
    timestamp: str,
    gate: Literal["H1", "H2", "H3", "H5"],
    config_sha256: str,
) -> str:
    safe = timestamp.replace("-", "").replace(":", "").replace(".", "")
    return (
        f"verify-prediction-correctness-{gate.lower()}-"
        f"{safe}-{config_sha256[:12]}"
    )


def _producer_validation_result_fields(
    *,
    gate: Literal["H1", "H2", "H3", "H5"],
    producer_validation: Mapping[str, object],
) -> tuple[str, str, tuple[str, ...]]:
    if gate in ("H1", "H2"):
        record = producer_validation.get("gate_result")
    elif gate == "H5":
        record = producer_validation.get("result")
    else:
        record = producer_validation
    if isinstance(record, Mapping):
        nested_gate = record.get("gate")
        nested_status = record.get("status")
        nested_obligations = record.get("obligations")
    else:
        nested_gate = getattr(record, "gate", None)
        nested_status = getattr(record, "status", None)
        nested_obligations = getattr(record, "obligations", None)
    if isinstance(nested_status, GateStatus):
        nested_status = nested_status.value
    if (
        type(nested_gate) is not str
        or type(nested_status) is not str
        or type(nested_obligations) not in (tuple, list)
        or any(
            type(item) is not str or not item
            for item in nested_obligations
        )
    ):
        raise ValueError(
            f"{gate} producer validation does not expose a typed result identity"
        )
    return nested_gate, nested_status, tuple(nested_obligations)


def _publish_prediction_correctness_artifacts(
    *,
    run_root: Path,
    started_utc: str,
    source_provenance: Mapping[str, object],
    gate_configs: tuple[
        tuple[Literal["H1", "H2", "H3", "H5"], ResolvedConfig], ...
    ],
    gate_results: tuple[
        tuple[
            Literal["H1", "H2", "H3", "H5"],
            GateResult | H3GateResult | H5GateResult,
        ],
        ...,
    ],
    producer_validations: tuple[
        tuple[Literal["H1", "H2", "H3", "H5"], object], ...
    ],
) -> tuple[
    tuple[Literal["H1", "H2", "H3", "H5"], Path, str], ...
]:
    """Publish Prediction-only gate roots from already-computed evaluations."""

    if not isinstance(run_root, Path):
        raise ValueError("prediction correctness run_root must be a Path")
    if type(started_utc) is not str or not started_utc:
        raise ValueError("prediction correctness started_utc must be nonempty")
    if not isinstance(source_provenance, Mapping):
        raise ValueError("prediction correctness source provenance must be a mapping")
    configs = _ordered_prediction_inputs(gate_configs, name="gate_configs")
    results = _ordered_prediction_inputs(gate_results, name="gate_results")
    validations = _ordered_prediction_inputs(
        producer_validations,
        name="producer_validations",
    )
    git_head = source_provenance.get("git_head")
    dirty_digest = source_provenance.get("dirty_digest")
    if (
        type(git_head) is not str
        or len(git_head) != 40
        or any(character not in _LOWER_HEX for character in git_head)
    ):
        raise ValueError("prediction correctness git_head must be lowercase 40-hex")
    if (
        type(dirty_digest) is not str
        or len(dirty_digest) != 64
        or any(character not in _LOWER_HEX for character in dirty_digest)
    ):
        raise ValueError(
            "prediction correctness dirty_digest must be lowercase 64-hex"
        )
    recorded_dirty_content = source_provenance.get("dirty_content_digest")
    if (
        recorded_dirty_content is not None
        and recorded_dirty_content != dirty_digest
    ):
        raise ValueError(
            "prediction correctness source dirty digests disagree"
        )
    source_sha256 = source_candidate_sha256(
        git_head_value=git_head,
        dirty_digest_value=dirty_digest,
    )

    prepared: list[
        tuple[
            Literal["H1", "H2", "H3", "H5"],
            str,
            dict[str, object],
            dict[str, object],
            dict[str, object],
        ]
    ] = []
    for gate in _PREDICTION_CORRECTNESS_GATES:
        config = configs[gate]
        if type(config) is not ResolvedConfig:
            raise ValueError(f"{gate} correctness config has the wrong type")
        config_payload, config_bytes = _raw_config_payload(config)
        config_sha256 = hashlib.sha256(config_bytes).hexdigest()
        result = results[gate]
        if getattr(result, "gate", None) != gate:
            raise ValueError(f"{gate} correctness result gate differs")
        status = getattr(result, "status", None)
        obligations = getattr(result, "obligations", None)
        if not isinstance(status, GateStatus):
            raise ValueError(f"{gate} correctness status is not typed")
        if (
            type(obligations) is not tuple
            or any(type(item) is not str or not item for item in obligations)
        ):
            raise ValueError(
                f"{gate} correctness obligations must be nonempty strings"
            )
        producer_validation = validations[gate]
        if not isinstance(producer_validation, Mapping):
            raise ValueError(
                f"{gate} producer_validation must be a mapping"
            )
        nested_gate, nested_status, nested_obligations = (
            _producer_validation_result_fields(
                gate=gate,
                producer_validation=producer_validation,
            )
        )
        if (
            nested_gate != gate
            or nested_status != status.value
            or nested_obligations != obligations
        ):
            raise ValueError(
                f"{gate} producer validation differs from its typed gate result"
            )
        validation_payload = {
            "schema_version": "vfe4-prediction-correctness-v1",
            "gate": gate,
            "git_head": git_head,
            "dirty_digest": dirty_digest,
            "config_sha256": config_sha256,
            "status": status.value,
            "obligations": obligations,
            "producer_validation": producer_validation,
        }
        provenance_payload = {
            "schema_version": "vfe4-prediction-correctness-provenance-v1",
            "gate": gate,
            "git_head": git_head,
            "dirty_digest": dirty_digest,
            "dirty_content_digest": dirty_digest,
            "source_sha256": source_sha256,
            "config_sha256": config_sha256,
            "status": status.value,
        }
        if gate == "H5":
            h5_fields = (
                "h5_config",
                "h5_state_hashes",
                "h5_update_hash_records",
                "h5_update_binding_preimages",
            )
            missing_h5_fields = tuple(
                name for name in h5_fields if source_provenance.get(name) is None
            )
            if status is GateStatus.PASS and missing_h5_fields:
                raise RuntimeError(
                    "PASS H5 correctness publication lacks producer provenance: "
                    + ", ".join(missing_h5_fields)
                )
            for name in h5_fields:
                value = source_provenance.get(name)
                if value is not None:
                    provenance_payload[name] = value
        prepared.append(
            (
                gate,
                config_sha256,
                config_payload,
                validation_payload,
                provenance_payload,
            )
        )

    references: list[
        tuple[Literal["H1", "H2", "H3", "H5"], Path, str]
    ] = []
    for (
        gate,
        config_sha256,
        config_payload,
        validation_payload,
        provenance_payload,
    ) in prepared:
        root = publish_run_directory(
            run_root,
            _prediction_correctness_run_name(
                started_utc,
                gate,
                config_sha256,
            ),
            {
                "config.json": config_payload,
                "provenance.json": provenance_payload,
                f"validation/{gate.lower()}.json": validation_payload,
            },
        )
        manifest_sha256 = hashlib.sha256(
            (root / "manifest.sha256").read_bytes()
        ).hexdigest()
        references.append((gate, root, manifest_sha256))
    return tuple(references)


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
    raw.pop("h4", None)
    raw.pop("h5", None)
    return resolve_config(raw, repo_root=REPO_ROOT)


def _h3_projection(config: ResolvedConfig) -> ResolvedConfig:
    """Project the coupled config onto H3's unchanged exact contract."""

    raw = json.loads(config.canonical_json)
    raw["validation"]["gates"] = ["H1", "H2", "H3"]
    raw.pop("h4", None)
    raw.pop("h5", None)
    return resolve_config(raw, repo_root=REPO_ROOT)


def _aggregate_state(
    results: tuple[GateResult | H3GateResult | H4GateResult | H5GateResult, ...],
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
    h4: H4GateEvaluation | None,
    h5: H5GateEvaluation | None,
    started_utc: str,
    ended_utc: str,
) -> dict[str, object]:
    if h2 is not None and h1.fixture_observed_sha256 != h2.fixture_observed_sha256:
        raise ValueError("ordered legacy gates reported different fixture snapshots")
    evaluations = (
        h1,
        *((h2,) if h2 is not None else ()),
        *((h3,) if h3 is not None else ()),
        *((h4,) if h4 is not None else ()),
        *((h5,) if h5 is not None else ()),
    )
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
        result.gate for result in results if result.gate in ("H1", "H2", "H5")
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
    if h4 is not None and h5 is not None:
        if config.h4 is None or config.h5 is None:
            raise ValueError("coupled provenance requires typed H4 and H5 config")
        if h4.h4_config_sha256 != config.h4.config_sha256:
            raise ValueError("H4 evaluation/config identity differs")
        h4_projection = json.dumps(
            json.loads(config.canonical_json)["h4"],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        if (
            h4_projection != config.h4.canonical_json
            or hashlib.sha256(h4_projection.encode("utf-8")).hexdigest()
            != config.h4.config_sha256
        ):
            raise ValueError("full resolved config H4 projection drifted")
        h5_raw = h5.result.update_spec_raw_sha256
        if (
            h5.result.preflight.phase is H5PreflightPhase.READY
            and h5_raw != config.h5.update_spec_raw_sha256
        ):
            raise ValueError("H5 evaluation/config raw fixture identity differs")
        provenance["fixture_hashes"]["h5-conditional-update-v1"] = {
            "expected_sha256": config.h5.update_spec_raw_sha256,
            "observed_sha256": h5_raw,
            "hash_domain": "raw_fixture_bytes",
        }
        provenance["gate_fixture_consumers"]["H4"] = (
            "h3-coupled-v1", "h3-zero-control-v1",
        )
        provenance["gate_fixture_consumers"]["H5"] = (
            "h1-v1", "h5-conditional-update-v1",
        )
        provenance["h4_config_sha256"] = config.h4.config_sha256
        provenance["h4_projection_sha256"] = hashlib.sha256(
            h4_projection.encode("utf-8")
        ).hexdigest()
        provenance["full_config_sha256"] = config.config_sha256
        provenance["h4_bounded_claim"] = h4.bounded_claim
        provenance["h4_nonclaims"] = h4.nonclaims
        reference = h5.reference
        provenance["h5_config"] = {
            "config_sha256": config.h5.config_sha256,
            "update_spec_raw_sha256": config.h5.update_spec_raw_sha256,
            "update_spec_canonical_sha256": h5.result.update_spec_canonical_sha256,
            "objective_schema_sha256": h5.result.objective_schema_sha256,
            "factor_input_schema_version": h5.result.factor_input_schema_version,
            "factor_input_schema_sha256": h5.result.factor_input_schema_sha256,
            "recognition_family": config.h5.recognition_family,
            "factor_universe": config.h5.factor_universe,
            "recognition_coordinate_universe": config.h5.recognition_coordinate_universe,
            "model_block_universe": config.h5.model_block_universe,
            "enabled_update_rules": tuple(item.value for item in config.h5.enabled_update_rules),
            "enabled_update_labels": tuple(item.value for item in config.h5.enabled_update_labels),
            "positive_case_ids": config.h5.positive_case_ids,
            "control_ids": config.h5.control_ids,
            "quadrature_orders": config.h5.quadrature_orders,
            "allowance_policy": config.h5.allowance_policy,
            "rounding_constant": config.h5.rounding_constant,
            "stochastic_contribution": config.h5.stochastic_contribution,
            "epsilon_delta_formula": config.h5.epsilon_delta_formula,
        }
        provenance["h5_state_hashes"] = {
            "reference_sha256": h5.result.reference_sha256,
            "recognition_sha256": (
                reference.initial_recognition.state_sha256 if reference is not None else None
            ),
            "model_sha256": (
                reference.initial_model.state_sha256 if reference is not None else None
            ),
            "validation_payload_sha256": h5.validation_payload.payload_sha256,
        }
        provenance["h5_update_hash_records"] = {
            "positive": tuple(
                {
                    "case_id": item.case_id.value,
                    "hashes": asdict(item.outcome.hashes),
                }
                for item in (h5.result.positive_cases or ())
            ),
            "controls": tuple(
                {
                    "control_id": item.control_id.value,
                    "hashes": asdict(item.outcome.hashes),
                }
                for item in (h5.result.controls or ())
            ),
        }
        provenance["h5_bounded_claim"] = (
            "deterministic complete-objective update coherence for the frozen H5 cases"
        )
        provenance["h5_nonclaims"] = h5.validation_payload.nonclaims
        if (
            h5.result.preflight.phase is H5PreflightPhase.READY
            and h5.reference is not None
        ):
            provenance["h5_update_binding_preimages"] = (
                h5_update_binding_preimages(h5)
            )
    return provenance


def run_verification(
    config: ResolvedConfig,
    *,
    publish_prediction_correctness: bool = False,
) -> VerificationRunResult:
    """Evaluate one implemented prefix from one capture set and publish once."""

    if type(publish_prediction_correctness) is not bool:
        raise ValueError("publish_prediction_correctness must be a boolean")
    canonical = _canonical_config(config)
    gates = canonical.validation.gates
    if gates not in _ALLOWED_PREFIXES:
        raise ValueError("run_verification requires an implemented ordered gate prefix")
    if (
        publish_prediction_correctness
        and gates != ("H1", "H2", "H3", "H4", "H5")
    ):
        raise ValueError(
            "Prediction correctness publication requires the full H1--H5 "
            "evaluation operation"
        )
    legacy = _legacy_projection(canonical)
    h3_config = _h3_projection(canonical) if "H3" in gates else None
    started = _utc_now()

    h1_bytes = FIXTURE_PATH.read_bytes()
    coupled_bytes: bytes | None = None
    zero_control_bytes: bytes | None = None
    h5_update_spec_bytes: bytes | None = None
    observed_h5_sha256: str | None = None
    expected_h5_sha256: str | None = None
    h5_update_spec_digest_matches: bool | None = None
    if "H3" in gates:
        coupled_bytes = H3_COUPLED_FIXTURE_PATH.read_bytes()
        zero_control_bytes = H3_ZERO_CONTROL_FIXTURE_PATH.read_bytes()
    if gates == ("H1", "H2", "H3", "H4", "H5"):
        if canonical.h5 is None:
            raise ValueError("coupled prefix lacks its typed H5 config")
        h5_update_spec_bytes = H5_UPDATE_SPEC_FIXTURE_PATH.read_bytes()
        observed_h5_sha256 = hashlib.sha256(h5_update_spec_bytes).hexdigest()
        expected_h5_sha256 = canonical.h5.update_spec_raw_sha256
        h5_update_spec_digest_matches = observed_h5_sha256 == expected_h5_sha256

    h1 = evaluate_h1(legacy, fixture_bytes=h1_bytes)
    h2: H2GateEvaluation | None = None
    h3: H3GateEvaluation | None = None
    h4: H4GateEvaluation | None = None
    h5: H5GateEvaluation | None = None
    results: list[GateResult | H3GateResult | H4GateResult | H5GateResult] = [h1.result]
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
        if h3_config is None:
            raise RuntimeError("H3 config projection is unavailable")
        h3 = evaluate_h3(
            h3_config,
            coupled_fixture_bytes=coupled_bytes,
            zero_control_fixture_bytes=zero_control_bytes,
        )
        results.append(h3.result)
        validation_payloads["validation/h3.json"] = h3_validation_payload(h3)
    if "H4" in gates:
        if canonical.h4 is None or coupled_bytes is None or zero_control_bytes is None:
            raise RuntimeError("H4 typed config or captured anchor bytes are unavailable")
        h4 = evaluate_h4(
            canonical.h4,
            h3_coupled_bytes=coupled_bytes,
            h3_zero_bytes=zero_control_bytes,
        )
        h4_artifact = h4_validation_artifact(h4)
        results.append(h4.result)
        validation_payloads["validation/h4.json"] = h4_validation_payload(h4_artifact)
    if "H5" in gates:
        if (
            h5_update_spec_bytes is None
            or observed_h5_sha256 is None
            or expected_h5_sha256 is None
            or h5_update_spec_digest_matches is None
        ):
            raise RuntimeError("H5 update-spec preflight capture is unavailable")
        h5 = evaluate_h5(
            canonical,
            h1_fixture_bytes=h1_bytes,
            h5_update_spec_bytes=h5_update_spec_bytes,
        )
        if (
            h5.result.update_spec_raw_sha256 != observed_h5_sha256
            or (
                h5.result.update_spec_raw_sha256 == expected_h5_sha256
            ) is not h5_update_spec_digest_matches
        ):
            raise RuntimeError("H5 typed result differs from runner raw digest preflight")
        results.append(h5.result)
        validation_payloads["validation/h5.json"] = h5_validation_payload(h5)

    frozen_results = tuple(results)
    ended = _utc_now()
    provenance = _combined_provenance(
        canonical,
        h1,
        h2,
        h3,
        h4,
        h5,
        started,
        ended,
    )
    payloads = {
        "config.json": _config_payload(canonical),
        "provenance.json": provenance,
        "environment.json": build_environment(canonical),
        **validation_payloads,
    }
    run_directory = publish_run_directory(
        canonical.artifacts.run_root,
        _run_name(started, canonical.config_sha256, gates),
        payloads,
    )
    correctness_artifacts: tuple[
        tuple[Literal["H1", "H2", "H3", "H5"], Path, str], ...
    ] = ()
    if publish_prediction_correctness:
        if h2 is None or h3 is None or h3_config is None or h5 is None:
            raise RuntimeError(
                "Prediction correctness publication lacks computed H1/H2/H3/H5 "
                "evaluations"
            )
        correctness_artifacts = _publish_prediction_correctness_artifacts(
            run_root=canonical.artifacts.run_root,
            started_utc=started,
            source_provenance=provenance,
            gate_configs=(
                ("H1", legacy),
                ("H2", legacy),
                ("H3", h3_config),
                ("H5", canonical),
            ),
            gate_results=(
                ("H1", h1.result),
                ("H2", h2.result),
                ("H3", h3.result),
                ("H5", h5.result),
            ),
            producer_validations=(
                ("H1", validation_payloads["validation/h1.json"]),
                ("H2", validation_payloads["validation/h2.json"]),
                ("H3", validation_payloads["validation/h3.json"]),
                ("H5", validation_payloads["validation/h5.json"]),
            ),
        )
    return VerificationRunResult(
        frozen_results,
        run_directory,
        correctness_artifacts,
    )


__all__ = ["VerificationRunResult", "run_verification"]
