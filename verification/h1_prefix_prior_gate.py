"""Fail-closed H1 gate for the prefix-conditioned source-prior variant."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import torch

from verification.h1_gate import TERM_NAMES, Comparison, pair_comparison
from verification.numpy_oracles.h1_prefix_prior import (
    H1PrefixPriorOracleRecord,
    PrefixPriorProbabilities,
    evaluate_h1_prefix_prior_oracle,
    evaluate_parent_specific_h1_prefix_prior_oracle,
    parent_specific_prefix_prior_probabilities,
    parse_h1_prefix_prior_fixture,
    parse_parent_specific_h1_prefix_prior_fixture,
    prefix_prior_probabilities,
)
from vfe4.artifacts.atomic import canonical_json_bytes, publish_run_directory
from vfe4.artifacts.provenance import (
    current_source_identity,
    dirty_content_digest,
    git_head,
    source_candidate_sha256,
)
from vfe4.config import (
    H1PrefixPriorResolvedConfig,
    resolve_h1_prefix_prior_config,
)
from vfe4.data.windows import CausalPrefix
from vfe4.generative import (
    H1GenerativeModel,
    ParentSpecificPooledPrefixSourcePrior,
    PooledHistoryConditionedSourcePrior,
)
from vfe4.objective import (
    MonolithicElboResult,
    evaluate_local_elbo,
    evaluate_monolithic_elbo,
)
from vfe4.recognition import H1RecognitionLaw
from vfe4.types.h6 import (
    CausalDag,
    CausalDagRow,
    H1_PREFIX_PRIOR_V2_GENERATIVE_FACTOR_SCHEMA_SHA256,
    H6LanguageStructure,
    VocabularyIdentity,
    ZeroDimensionalBase,
    canonical_json_bytes as canonical_h6_json_bytes,
)
from vfe4.types.results import (
    ElboTerms,
    GateStatus,
    H1PrefixPriorGateResult,
    H1PrefixPriorV2GateResult,
    InvariantResult,
)
from vfe4.validation import load_h1_fixture


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = (
    REPO_ROOT / "vfe4" / "validation" / "fixtures" / "h1_prefix_prior_v1.json"
)
V2_FIXTURE_PATH = (
    REPO_ROOT / "vfe4" / "validation" / "fixtures" / "h1_prefix_prior_v2.json"
)
BASE_FIXTURE_PATH = (
    REPO_ROOT / "vfe4" / "validation" / "fixtures" / "h1_v1.json"
)
EXPECTED_H1_PREFIX_PRIOR_FIXTURE_SHA256 = (
    "b6638ea3b64c7fd68882cbaced914e4d17d2cd03c8b6b8a939fd575a1b9f43f1"
)
EXPECTED_H1_PREFIX_PRIOR_V2_FIXTURE_SHA256 = (
    "6b0e855482b8f335bec73e4b0976a1317d7ce4cf3ff050670b3950e271c57fde"
)
EXPECTED_H1_BASE_FIXTURE_SHA256 = (
    "388e38cc8c16d8b5e2c61919c1e712a134d88fb0bbd8ec1f2939b9859c9a583b"
)
GENERATIVE_FACTOR_SCHEMA = {
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
GENERATIVE_FACTOR_SCHEMA_BYTES = canonical_json_bytes(GENERATIVE_FACTOR_SCHEMA)
EXPECTED_GENERATIVE_FACTOR_SCHEMA_SHA256 = (
    "f38a83b80e046e1d4115a9eca2ccc3afe080fd6b0352fcef399afaf30bea6816"
)
if (
    hashlib.sha256(GENERATIVE_FACTOR_SCHEMA_BYTES).hexdigest()
    != EXPECTED_GENERATIVE_FACTOR_SCHEMA_SHA256
):
    raise RuntimeError("frozen H1 prefix-prior schema hash is inconsistent")

PARENT_SPECIFIC_GENERATIVE_FACTOR_SCHEMA = {
    "schema_version": "h1-prefix-prior-generative-factor-v2",
    "prior_variant": "parent_specific_pooled_prefix",
    "scorer_schema": "parent-specific-pooled-prefix-bilinear-v1",
    "conditioning_inputs": [
        "prior_token_ids",
        "supported_parent_latents",
    ],
    "forbidden_inputs": [
        "current_target",
        "future_tokens",
        "recognition_state",
    ],
    "target_blind": True,
    "receiver_t": 2,
    "source_banks": ["state", "model"],
    "context_dim": 1,
    "token_summary": "mean-prior-token-embeddings-v1",
    "parent_content": "bank-projection-of-candidate-row-v1",
    "anchor": "last-declared-parent-complete-score-subtraction-v1",
    "support_mask_application": "before_normalization",
    "normalization": "masked-log-softmax-from-declared-parents-v1",
}
PARENT_SPECIFIC_GENERATIVE_FACTOR_SCHEMA_BYTES = canonical_json_bytes(
    PARENT_SPECIFIC_GENERATIVE_FACTOR_SCHEMA
)
EXPECTED_PARENT_SPECIFIC_GENERATIVE_FACTOR_SCHEMA_SHA256 = (
    H1_PREFIX_PRIOR_V2_GENERATIVE_FACTOR_SCHEMA_SHA256
)
if (
    hashlib.sha256(
        PARENT_SPECIFIC_GENERATIVE_FACTOR_SCHEMA_BYTES
    ).hexdigest()
    != EXPECTED_PARENT_SPECIFIC_GENERATIVE_FACTOR_SCHEMA_SHA256
):
    raise RuntimeError(
        "frozen parent-specific H1 prefix-prior schema hash is inconsistent"
    )

_EPSILON = float(np.finfo(np.float64).eps)
H1_PREFIX_PRIOR_MEASUREMENT_NAMES = (
    "monolithic_elbo",
    "local_elbo",
    "evidence_minus_posterior_kl",
)
H1_PREFIX_PRIOR_CONVERGENCE_NAMES = (
    "active.monolithic",
    *(f"active.local.{name}" for name in TERM_NAMES),
    *(f"active.independent.{name}" for name in TERM_NAMES),
    "active.identity.posterior_kl",
    "active.identity.elbo",
    "active.identity.evidence.probability",
    "active.identity.evidence.log_probability",
    "negative.current_target.monolithic",
)
H1_PREFIX_PRIOR_INVARIANT_NAMES = (
    "monolithic_vs_local",
    "monolithic_vs_identity",
    "local_vs_identity",
    *(f"oracle.{name}" for name in TERM_NAMES),
    *(f"convergence.{name}" for name in H1_PREFIX_PRIOR_CONVERGENCE_NAMES),
    "source_prior.state.production_vs_oracle.active",
    "source_prior.model.production_vs_oracle.active",
    "source_prior.state.production_vs_oracle.alternate",
    "source_prior.model.production_vs_oracle.alternate",
    "source_prior.state.normalized.active",
    "source_prior.model.normalized.active",
    "source_prior.state.normalized.alternate",
    "source_prior.model.normalized.alternate",
    "source_prior.state.normalized.current_target_control",
    "source_prior.model.normalized.current_target_control",
    "source_prior.state.prefix_distinct",
    "source_prior.model.prefix_distinct",
    "schema.target_blind",
    "latent_projection.exact_zero",
    "negative.current_target_as_prefix",
)
_FIXTURE_FIELDS = frozenset(
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
_TOKENIZER_SPEC_SHA256 = hashlib.sha256(
    b"vfe4-h1-prefix-prior-zero-based-vocabulary-v1"
).hexdigest()


@dataclass(frozen=True)
class _ProductionPriorRecord:
    case_id: str
    prefix_token_ids: tuple[int, ...]
    state_log_probabilities: tuple[float, float]
    model_log_probabilities: tuple[float, float]
    state_probabilities: tuple[float, float]
    model_probabilities: tuple[float, float]
    state_factor_identity_sha256: str
    model_factor_identity_sha256: str


@dataclass(frozen=True)
class _ProductionH1Record:
    monolithic: MonolithicElboResult
    local: ElboTerms


@dataclass(frozen=True)
class ParentSpecificProductionPriorRecord:
    case_id: str
    prefix_token_ids: tuple[int, ...]
    state_log_probabilities: tuple[float, float]
    model_log_probabilities: tuple[float, float]
    state_probabilities: tuple[float, float]
    model_probabilities: tuple[float, float]
    state_support_mask: tuple[bool, bool]
    model_support_mask: tuple[bool, bool]
    state_factor_identity_sha256: str
    model_factor_identity_sha256: str


@dataclass(frozen=True)
class ParentSpecificH1PrefixPriorEvaluation:
    status: GateStatus
    fixture_sha256: str
    generative_factor_schema_sha256: str
    production_priors: Mapping[str, ParentSpecificProductionPriorRecord]
    independent_priors: Mapping[str, PrefixPriorProbabilities]
    production_objectives: Mapping[str, _ProductionH1Record]
    independent_objectives: Mapping[str, H1PrefixPriorOracleRecord]
    invariants: tuple[InvariantResult, ...]

    def __post_init__(self) -> None:
        if self.status is GateStatus.PASS and not all(
            item.passed for item in self.invariants
        ):
            raise ValueError(
                "parent-specific H1 PASS requires every invariant"
            )


@dataclass(frozen=True)
class H1PrefixPriorGateEvaluation:
    result: H1PrefixPriorGateResult
    validation_payload: dict[str, object]
    fixture_sha256: str | None
    base_fixture_sha256: str | None
    generative_factor_schema_bytes: bytes


@dataclass(frozen=True)
class ParentSpecificH1PrefixPriorArtifactEvaluation:
    result: H1PrefixPriorV2GateResult
    validation_payload: dict[str, object]
    fixture_sha256: str
    base_fixture_sha256: str
    generative_factor_schema_bytes: bytes


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _run_name(timestamp: str, config_sha256: str) -> str:
    safe = timestamp.replace("-", "").replace(":", "").replace(".", "")
    return f"verify-h1-prefix-prior-{safe}-{config_sha256[:12]}"


def _optional_sha256(value: object, name: str) -> str | None:
    if value is None:
        return None
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be None or exact lowercase SHA-256")
    return value


def _mapping(value: object, fields: frozenset[str], name: str) -> dict[str, Any]:
    if type(value) is not dict or frozenset(value) != fields:
        raise ValueError(f"{name} fields must equal {sorted(fields)!r}")
    return value


def _sequence(value: object, length: int, name: str) -> list[Any]:
    if type(value) is not list or len(value) != length:
        raise ValueError(f"{name} must be a list of length {length}")
    return value


def _hex_float(value: object, name: str) -> float:
    if type(value) is not str:
        raise ValueError(f"{name} must be a canonical hexadecimal float")
    try:
        parsed = float.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{name} is not a hexadecimal float") from exc
    if not math.isfinite(parsed) or parsed.hex() != value:
        raise ValueError(f"{name} must be a canonical finite hexadecimal float")
    return parsed


def _hex_tensor(
    value: object, rows: int, columns: int, name: str
) -> torch.Tensor:
    outer = _sequence(value, rows, name)
    return torch.tensor(
        [
            [
                _hex_float(item, f"{name}[{row_index}][{column_index}]")
                for column_index, item in enumerate(
                    _sequence(row, columns, f"{name}[{row_index}]")
                )
            ]
            for row_index, row in enumerate(outer)
        ],
        dtype=torch.float64,
    )


def _hex_vector(value: object, length: int, name: str) -> torch.Tensor:
    return torch.tensor(
        [
            _hex_float(item, f"{name}[{index}]")
            for index, item in enumerate(_sequence(value, length, name))
        ],
        dtype=torch.float64,
    )


def _fixture_root(
    fixture_bytes: bytes, config: H1PrefixPriorResolvedConfig
) -> dict[str, Any]:
    if type(fixture_bytes) is not bytes:
        raise ValueError("fixture_bytes must be immutable bytes")
    if hashlib.sha256(fixture_bytes).hexdigest() != config.fixture_sha256:
        raise ValueError("prefix-prior fixture bytes do not match config")
    try:
        root = _mapping(
            json.loads(fixture_bytes.decode("utf-8")),
            _FIXTURE_FIELDS,
            "prefix-prior fixture",
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"prefix-prior fixture is not valid UTF-8 JSON: {exc}") from exc
    if (
        root["fixture_schema_version"] != "h1-prefix-prior-fixture-v1"
        or root["fixture_id"] != config.fixture_id
    ):
        raise ValueError("prefix-prior fixture identity differs from config")
    base = _mapping(
        root["base_h1_fixture"],
        frozenset({"relative_path", "raw_sha256"}),
        "base_h1_fixture",
    )
    if base != {
        "relative_path": "vfe4/validation/fixtures/h1_v1.json",
        "raw_sha256": config.base_fixture_sha256,
    }:
        raise ValueError("base H1 fixture identity differs from config")
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
    if structure != {
        "horizon": config.horizon,
        "d_z": config.d_z,
        "d_m": config.d_m,
        "vocabulary_size": config.vocabulary_size,
        "state_parent_sets": [list(row) for row in config.state_parent_sets],
        "model_parent_sets": [list(row) for row in config.model_parent_sets],
    }:
        raise ValueError("prefix-prior structure differs from config")
    if root["generative_factor_schema"] != GENERATIVE_FACTOR_SCHEMA:
        raise ValueError("fixture generative-factor schema is not frozen")
    if (
        hashlib.sha256(canonical_json_bytes(root["generative_factor_schema"])).hexdigest()
        != config.generative_factor_schema_sha256
    ):
        raise ValueError("generative-factor schema digest differs from config")
    quadrature = _mapping(
        root["quadrature"],
        frozenset(
            {"order", "convergence_check_order", "maximum_convergence_estimate"}
        ),
        "quadrature",
    )
    if quadrature != {
        "order": config.quadrature_order,
        "convergence_check_order": config.convergence_check_order,
        "maximum_convergence_estimate": config.maximum_convergence_estimate,
    }:
        raise ValueError("prefix-prior quadrature differs from config")
    if (
        root["active_case_id"] != "prefix-token-0"
        or root["current_target_token_id"] != 1
    ):
        raise ValueError("prefix-prior active case or target control is not frozen")
    cases = _sequence(root["prefix_cases"], 2, "prefix_cases")
    expected_cases = (
        {"case_id": "prefix-token-0", "prefix_token_ids": [0]},
        {"case_id": "prefix-token-2", "prefix_token_ids": [2]},
    )
    if tuple(cases) != expected_cases:
        raise ValueError("the exact two-prefix inventory is required")
    return root


def _case_tokens(
    root: dict[str, Any],
    *,
    case_id: str | None = None,
    use_current_target: bool = False,
) -> tuple[str, tuple[int, ...]]:
    if use_current_target:
        if case_id is not None:
            raise ValueError("case_id and use_current_target are mutually exclusive")
        return "negative-current-target", (root["current_target_token_id"],)
    selected = root["active_case_id"] if case_id is None else case_id
    for case in root["prefix_cases"]:
        if case["case_id"] == selected:
            return selected, tuple(case["prefix_token_ids"])
    raise ValueError(f"unknown prefix case: {selected}")


def _production_prior(
    root: dict[str, Any],
    config: H1PrefixPriorResolvedConfig,
    *,
    case_id: str | None = None,
    use_current_target: bool = False,
) -> _ProductionPriorRecord:
    selected_id, token_ids = _case_tokens(
        root, case_id=case_id, use_current_target=use_current_target
    )
    base = ZeroDimensionalBase.create()
    dag = CausalDag.create(
        node_labels=(0, 1, 2),
        rows=(CausalDagRow(1, (0,)), CausalDagRow(2, (0, 1))),
    )
    structure = H6LanguageStructure.create(
        base=base, dag=dag, receiver_labels=(1, 2)
    )
    vocabulary = VocabularyIdentity(
        "h1-prefix-prior-v1",
        3,
        _TOKENIZER_SPEC_SHA256,
    )
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
        raise ValueError("prefix-prior context_dim must be one")
    prior = PooledHistoryConditionedSourcePrior(
        structure=structure,
        vocabulary=vocabulary,
        fixture_sha256=config.fixture_sha256,
        predictor_config_sha256=config.config_sha256,
        model_family_sha256=config.generative_factor_schema_sha256,
        latent_dim=1,
        context_dim=1,
    )
    token_embedding = _hex_tensor(
        parameters["token_embedding"], 3, 1, "token_embedding"
    )
    state_projection = _hex_tensor(
        parameters["state_latent_projection"],
        1,
        1,
        "state_latent_projection",
    )
    model_projection = _hex_tensor(
        parameters["model_latent_projection"],
        1,
        1,
        "model_latent_projection",
    )
    state_keys = _hex_tensor(
        parameters["state_parent_keys_t2"], 2, 1, "state_parent_keys_t2"
    )
    model_keys = _hex_tensor(
        parameters["model_parent_keys_t2"], 2, 1, "model_parent_keys_t2"
    )
    state_biases = _hex_vector(
        parameters["state_biases_t2"], 2, "state_biases_t2"
    )
    model_biases = _hex_vector(
        parameters["model_biases_t2"], 2, "model_biases_t2"
    )
    earlier_latents = _hex_tensor(root["earlier_latents"], 2, 1, "earlier_latents")
    if (
        not torch.equal(state_projection, torch.zeros_like(state_projection))
        or not torch.equal(model_projection, torch.zeros_like(model_projection))
        or not torch.equal(earlier_latents, torch.zeros_like(earlier_latents))
    ):
        raise ValueError("the bounded fixture requires exact-zero latent projections")
    with torch.no_grad():
        prior.token_embedding.weight.copy_(token_embedding)
        prior.state_latent_projection.weight.copy_(state_projection)
        prior.model_latent_projection.weight.copy_(model_projection)
        prior.state_source_free_parent_keys[0].copy_(
            state_keys[:1] - state_keys[1]
        )
        prior.model_source_free_parent_keys[0].copy_(
            model_keys[:1] - model_keys[1]
        )
        prior.state_source_free_biases[0].copy_(
            state_biases[:1] - state_biases[1]
        )
        prior.model_source_free_biases[0].copy_(
            model_biases[:1] - model_biases[1]
        )
        prefix = CausalPrefix.create(
            receiver_t=2,
            vocabulary=vocabulary,
            token_ids=torch.tensor(token_ids, dtype=torch.int64),
        )
        state = prior.state_source_log_probs(
            prefix=prefix, earlier_latents=earlier_latents
        )
        model = prior.model_source_log_probs(
            prefix=prefix, earlier_latents=earlier_latents
        )
        state_logs = state.log_probs.value()
        model_logs = model.log_probs.value()
        state_probabilities = torch.exp(state_logs)
        model_probabilities = torch.exp(model_logs)
    return _ProductionPriorRecord(
        selected_id,
        token_ids,
        tuple(float(value) for value in state_logs.tolist()),  # type: ignore[arg-type]
        tuple(float(value) for value in model_logs.tolist()),  # type: ignore[arg-type]
        tuple(float(value) for value in state_probabilities.tolist()),  # type: ignore[arg-type]
        tuple(float(value) for value in model_probabilities.tolist()),  # type: ignore[arg-type]
        state.factor_identity_sha256,
        model.factor_identity_sha256,
    )


def _derived_h1_fixture_bytes(
    base_fixture_bytes: bytes, prior: _ProductionPriorRecord
) -> bytes:
    if type(base_fixture_bytes) is not bytes:
        raise ValueError("base_fixture_bytes must be immutable bytes")
    if hashlib.sha256(base_fixture_bytes).hexdigest() != EXPECTED_H1_BASE_FIXTURE_SHA256:
        raise ValueError("base H1 fixture bytes do not match the frozen raw hash")
    try:
        payload = json.loads(base_fixture_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"base H1 fixture is not valid UTF-8 JSON: {exc}") from exc
    if type(payload) is not dict or payload.get("fixture_id") != "h1-v1":
        raise ValueError("base H1 fixture has the wrong identity")
    payload["state_source_priors"] = [[1.0], list(prior.state_probabilities)]
    payload["model_source_priors"] = [[1.0], list(prior.model_probabilities)]
    return canonical_json_bytes(payload)


def _production_h1(
    base_fixture_bytes: bytes,
    prior: _ProductionPriorRecord,
    config: H1PrefixPriorResolvedConfig,
) -> _ProductionH1Record:
    derived = _derived_h1_fixture_bytes(base_fixture_bytes, prior)
    with tempfile.TemporaryDirectory(prefix="vfe4-h1-prefix-production-") as temporary:
        path = Path(temporary) / "h1_v1.json"
        path.write_bytes(derived)
        fixture = load_h1_fixture(path)
        model = H1GenerativeModel.from_fixture(fixture)
        recognition = H1RecognitionLaw.from_fixture(fixture)
        with torch.no_grad():
            monolithic = evaluate_monolithic_elbo(
                model,
                recognition,
                quadrature_order=config.quadrature_order,
                convergence_check_order=config.convergence_check_order,
            )
            local = evaluate_local_elbo(
                model,
                recognition,
                quadrature_order=config.quadrature_order,
                convergence_check_order=config.convergence_check_order,
            )
    return _ProductionH1Record(monolithic, local)


_V2_FIXTURE_FIELDS = frozenset(
    {
        "fixture_schema_version",
        "fixture_id",
        "base_h1_fixture",
        "structure",
        "fixed_target_free_prefix",
        "target_suffix_controls",
        "parent_latent_histories",
        "prefix_prior_parameters",
        "generative_factor_schema",
        "quadrature",
    }
)


def _parent_specific_fixture_root(
    fixture_bytes: bytes,
) -> dict[str, Any]:
    if type(fixture_bytes) is not bytes:
        raise ValueError("fixture_bytes must be immutable bytes")
    if hashlib.sha256(fixture_bytes).hexdigest() != (
        EXPECTED_H1_PREFIX_PRIOR_V2_FIXTURE_SHA256
    ):
        raise ValueError("parent-specific fixture raw hash is not frozen")
    try:
        root = _mapping(
            json.loads(fixture_bytes.decode("utf-8")),
            _V2_FIXTURE_FIELDS,
            "parent-specific fixture",
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"parent-specific fixture is not valid UTF-8 JSON: {exc}"
        ) from exc
    if (
        root["fixture_schema_version"] != "h1-prefix-prior-fixture-v2"
        or root["fixture_id"] != "h1-prefix-prior-scorer-v2"
        or root["generative_factor_schema"]
        != PARENT_SPECIFIC_GENERATIVE_FACTOR_SCHEMA
    ):
        raise ValueError("parent-specific fixture identity is not frozen")
    if root["base_h1_fixture"] != {
        "relative_path": "vfe4/validation/fixtures/h1_v1.json",
        "raw_sha256": EXPECTED_H1_BASE_FIXTURE_SHA256,
    }:
        raise ValueError("parent-specific base fixture is not frozen")
    if root["structure"] != {
        "horizon": 2,
        "d_z": 1,
        "d_m": 1,
        "vocabulary_size": 3,
        "state_parent_sets": [[0], [0, 1]],
        "model_parent_sets": [[0], [0, 1]],
    }:
        raise ValueError("parent-specific H1 structure is not frozen")
    if root["fixed_target_free_prefix"] != {
        "case_id": "fixed-prefix-token-2",
        "prefix_token_ids": [2],
    }:
        raise ValueError("parent-specific target-free prefix is not frozen")
    if root["quadrature"] != {
        "order": 21,
        "convergence_check_order": 17,
        "maximum_convergence_estimate": 1e-9,
    }:
        raise ValueError("parent-specific quadrature policy is not frozen")
    return root


def _parent_specific_production_prior(
    root: dict[str, Any],
    *,
    fixture_sha256: str,
    case_id: str,
    history_id: str,
    prefix_token_ids: tuple[int, ...],
) -> ParentSpecificProductionPriorRecord:
    if history_id not in ("active", "swapped"):
        raise ValueError("history_id must be active or swapped")
    base = ZeroDimensionalBase.create()
    dag = CausalDag.create(
        node_labels=(0, 1, 2),
        rows=(CausalDagRow(1, (0,)), CausalDagRow(2, (0, 1))),
    )
    structure = H6LanguageStructure.create(
        base=base,
        dag=dag,
        receiver_labels=(1, 2),
    )
    vocabulary = VocabularyIdentity(
        "h1-prefix-prior-scorer-v2",
        3,
        _TOKENIZER_SPEC_SHA256,
    )
    prior = ParentSpecificPooledPrefixSourcePrior(
        structure=structure,
        vocabulary=vocabulary,
        fixture_sha256=fixture_sha256,
        predictor_config_sha256=hashlib.sha256(
            b"h1-prefix-prior-scorer-v2-production"
        ).hexdigest(),
        model_family_sha256=(
            EXPECTED_PARENT_SPECIFIC_GENERATIVE_FACTOR_SCHEMA_SHA256
        ),
        latent_dim=1,
        context_dim=1,
    )
    parameters = _mapping(
        root["prefix_prior_parameters"],
        frozenset(
            {
                "context_dim",
                "token_embedding",
                "state_latent_projection",
                "model_latent_projection",
                "state_free_parent_keys_t2",
                "model_free_parent_keys_t2",
                "state_free_biases_t2",
                "model_free_biases_t2",
            }
        ),
        "prefix_prior_parameters",
    )
    if parameters["context_dim"] != 1:
        raise ValueError("parent-specific context_dim must equal one")
    histories = _mapping(
        root["parent_latent_histories"],
        frozenset({"active", "swapped"}),
        "parent_latent_histories",
    )
    selected_history = _mapping(
        histories[history_id],
        frozenset({"state", "model"}),
        f"parent_latent_histories.{history_id}",
    )
    with torch.no_grad():
        prior.token_embedding.weight.copy_(
            _hex_tensor(
                parameters["token_embedding"],
                3,
                1,
                "token_embedding",
            )
        )
        prior.state_latent_projection.weight.copy_(
            _hex_tensor(
                parameters["state_latent_projection"],
                1,
                1,
                "state_latent_projection",
            )
        )
        prior.model_latent_projection.weight.copy_(
            _hex_tensor(
                parameters["model_latent_projection"],
                1,
                1,
                "model_latent_projection",
            )
        )
        prior.state_source_free_parent_keys[0].copy_(
            _hex_tensor(
                parameters["state_free_parent_keys_t2"],
                1,
                1,
                "state_free_parent_keys_t2",
            )
        )
        prior.model_source_free_parent_keys[0].copy_(
            _hex_tensor(
                parameters["model_free_parent_keys_t2"],
                1,
                1,
                "model_free_parent_keys_t2",
            )
        )
        prior.state_source_free_biases[0].copy_(
            _hex_vector(
                parameters["state_free_biases_t2"],
                1,
                "state_free_biases_t2",
            )
        )
        prior.model_source_free_biases[0].copy_(
            _hex_vector(
                parameters["model_free_biases_t2"],
                1,
                "model_free_biases_t2",
            )
        )
        prefix = CausalPrefix.create(
            receiver_t=2,
            vocabulary=vocabulary,
            token_ids=torch.tensor(prefix_token_ids, dtype=torch.int64),
        )
        state = prior.state_source_log_probs(
            prefix=prefix,
            earlier_latents=_hex_tensor(
                selected_history["state"],
                2,
                1,
                f"parent_latent_histories.{history_id}.state",
            ),
        )
        model = prior.model_source_log_probs(
            prefix=prefix,
            earlier_latents=_hex_tensor(
                selected_history["model"],
                2,
                1,
                f"parent_latent_histories.{history_id}.model",
            ),
        )
        state_logs = state.log_probs.value()
        model_logs = model.log_probs.value()
        state_probabilities = torch.exp(state_logs)
        model_probabilities = torch.exp(model_logs)
    return ParentSpecificProductionPriorRecord(
        case_id=case_id,
        prefix_token_ids=prefix_token_ids,
        state_log_probabilities=tuple(
            float(value) for value in state_logs.tolist()
        ),
        model_log_probabilities=tuple(
            float(value) for value in model_logs.tolist()
        ),
        state_probabilities=tuple(
            float(value) for value in state_probabilities.tolist()
        ),
        model_probabilities=tuple(
            float(value) for value in model_probabilities.tolist()
        ),
        state_support_mask=state.support_mask,
        model_support_mask=model.support_mask,
        state_factor_identity_sha256=state.factor_identity_sha256,
        model_factor_identity_sha256=model.factor_identity_sha256,
    )


def _production_h1_parent_specific(
    base_fixture_bytes: bytes,
    prior: ParentSpecificProductionPriorRecord,
) -> _ProductionH1Record:
    if type(base_fixture_bytes) is not bytes:
        raise ValueError("base_fixture_bytes must be immutable bytes")
    if hashlib.sha256(base_fixture_bytes).hexdigest() != (
        EXPECTED_H1_BASE_FIXTURE_SHA256
    ):
        raise ValueError("base H1 fixture bytes do not match the frozen hash")
    try:
        payload = json.loads(base_fixture_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"base H1 fixture is not valid UTF-8 JSON: {exc}"
        ) from exc
    if type(payload) is not dict or payload.get("fixture_id") != "h1-v1":
        raise ValueError("base H1 fixture has the wrong identity")
    payload["state_source_priors"] = [
        [1.0],
        list(prior.state_probabilities),
    ]
    payload["model_source_priors"] = [
        [1.0],
        list(prior.model_probabilities),
    ]
    derived = canonical_json_bytes(payload)
    with tempfile.TemporaryDirectory(
        prefix="vfe4-h1-parent-specific-production-"
    ) as temporary:
        path = Path(temporary) / "h1_v1.json"
        path.write_bytes(derived)
        fixture = load_h1_fixture(path)
        model = H1GenerativeModel.from_fixture(fixture)
        recognition = H1RecognitionLaw.from_fixture(fixture)
        with torch.no_grad():
            monolithic = evaluate_monolithic_elbo(
                model,
                recognition,
                quadrature_order=21,
                convergence_check_order=17,
            )
            local = evaluate_local_elbo(
                model,
                recognition,
                quadrature_order=21,
                convergence_check_order=17,
            )
    return _ProductionH1Record(monolithic, local)


def _term_value(record: object, name: str) -> float:
    if "[" in name:
        field, index_text = name[:-1].split("[")
        return float(getattr(record, field)[int(index_text)])
    return float(getattr(record, name))


def _term_allowance(record: object, name: str) -> float:
    allowances = getattr(record, "allowances")
    if "[" in name:
        field, index_text = name[:-1].split("[")
        return float(getattr(allowances, field)[int(index_text)].total)
    return float(getattr(allowances, name).total)


def _term_convergence(record: object, name: str) -> float:
    allowances = getattr(record, "allowances")
    if "[" in name:
        field, index_text = name[:-1].split("[")
        return float(
            getattr(allowances, field)[int(index_text)].convergence_estimate
        )
    return float(getattr(allowances, name).convergence_estimate)


def _invariant(
    name: str,
    passed: bool,
    value: float | None,
    limit: float | None,
    detail: str,
) -> InvariantResult:
    return InvariantResult(name, passed, value, limit, detail)


def _normalization_residual(probabilities: tuple[float, float]) -> float:
    return abs(math.fsum(probabilities) - 1.0)


def _prior_residual(
    left: tuple[float, float], right: tuple[float, float]
) -> float:
    return max(abs(a - b) for a, b in zip(left, right, strict=True))


def evaluate_parent_specific_h1_prefix_prior(
    fixture_bytes: bytes,
    *,
    base_fixture_bytes: bytes,
) -> ParentSpecificH1PrefixPriorEvaluation:
    """Evaluate the scorer-v2 sibling without publishing an evidence run."""

    root = _parent_specific_fixture_root(fixture_bytes)
    independent_fixture = (
        parse_parent_specific_h1_prefix_prior_fixture(fixture_bytes)
    )
    fixture_sha256 = hashlib.sha256(fixture_bytes).hexdigest()
    fixed_prefix = tuple(
        root["fixed_target_free_prefix"]["prefix_token_ids"]
    )
    production_priors: dict[
        str, ParentSpecificProductionPriorRecord
    ] = {
        history_id: _parent_specific_production_prior(
            root,
            fixture_sha256=fixture_sha256,
            case_id=history_id,
            history_id=history_id,
            prefix_token_ids=fixed_prefix,
        )
        for history_id in ("active", "swapped")
    }
    for control in root["target_suffix_controls"]:
        production_priors[control["case_id"]] = (
            _parent_specific_production_prior(
                root,
                fixture_sha256=fixture_sha256,
                case_id=control["case_id"],
                history_id="active",
                prefix_token_ids=tuple(control["full_token_ids"][:1]),
            )
        )
    independent_priors = {
        history_id: parent_specific_prefix_prior_probabilities(
            independent_fixture,
            history_id=history_id,
        )
        for history_id in ("active", "swapped")
    }
    production_objectives = {
        history_id: _production_h1_parent_specific(
            base_fixture_bytes,
            production_priors[history_id],
        )
        for history_id in ("active", "swapped")
    }
    independent_objectives = {
        history_id: evaluate_parent_specific_h1_prefix_prior_oracle(
            fixture_bytes,
            base_fixture_bytes=base_fixture_bytes,
            history_id=history_id,
        )
        for history_id in ("active", "swapped")
    }

    prior_allowance = 256.0 * _EPSILON
    invariants: list[InvariantResult] = []
    for history_id in ("active", "swapped"):
        production_prior = production_priors[history_id]
        independent_prior = independent_priors[history_id]
        for bank in ("state", "model"):
            production_probabilities = getattr(
                production_prior, f"{bank}_probabilities"
            )
            independent_probabilities = getattr(
                independent_prior, f"{bank}_probabilities"
            )
            prior_residual = _prior_residual(
                production_probabilities,
                independent_probabilities,
            )
            invariants.append(
                _invariant(
                    (
                        f"source_prior.{bank}."
                        f"production_vs_oracle.{history_id}"
                    ),
                    prior_residual <= prior_allowance,
                    prior_residual,
                    prior_allowance,
                    "Torch scorer-v2 agrees with independent NumPy",
                )
            )
            normalization_residual = _normalization_residual(
                production_probabilities
            )
            invariants.append(
                _invariant(
                    f"source_prior.{bank}.normalized.{history_id}",
                    normalization_residual <= prior_allowance,
                    normalization_residual,
                    prior_allowance,
                    "supported scorer-v2 probabilities sum to one",
                )
            )
            support = getattr(
                production_prior, f"{bank}_support_mask"
            )
            support_is_exact = support == (True, True)
            invariants.append(
                _invariant(
                    f"source_prior.{bank}.support.{history_id}",
                    support_is_exact,
                    0.0 if support_is_exact else 1.0,
                    0.0,
                    "the declared t=2 parent support is exact",
                )
            )

        production_h1 = production_objectives[history_id]
        independent_h1 = independent_objectives[history_id]
        objective_comparisons = {
            "monolithic_vs_local": pair_comparison(
                production_h1.monolithic.value,
                production_h1.local.complete_elbo,
                production_h1.monolithic.numerical_allowance.total,
                production_h1.local.allowances.complete_elbo.total,
            ),
            "monolithic_vs_identity": pair_comparison(
                production_h1.monolithic.value,
                independent_h1.identity.elbo_from_identity,
                production_h1.monolithic.numerical_allowance.total,
                independent_h1.identity.identity_allowance.total,
            ),
            "local_vs_independent_local": pair_comparison(
                production_h1.local.complete_elbo,
                independent_h1.local_terms.complete_elbo,
                production_h1.local.allowances.complete_elbo.total,
                independent_h1.local_terms.allowances.complete_elbo.total,
            ),
            "independent_local_vs_identity": pair_comparison(
                independent_h1.local_terms.complete_elbo,
                independent_h1.identity.elbo_from_identity,
                independent_h1.local_terms.allowances.complete_elbo.total,
                independent_h1.identity.identity_allowance.total,
            ),
        }
        invariants.extend(
            _invariant(
                f"objective.{history_id}.{name}",
                comparison.passed,
                comparison.residual,
                comparison.allowance,
                "complete-objective decompositions agree",
            )
            for name, comparison in objective_comparisons.items()
        )
        invariants.extend(
            _invariant(
                f"objective.{history_id}.term.{term_name}",
                comparison.passed,
                comparison.residual,
                comparison.allowance,
                "production and independent complete-ELBO terms agree",
            )
            for term_name in TERM_NAMES
            for comparison in (
                pair_comparison(
                    _term_value(production_h1.local, term_name),
                    _term_value(independent_h1.local_terms, term_name),
                    _term_allowance(production_h1.local, term_name),
                    _term_allowance(
                        independent_h1.local_terms,
                        term_name,
                    ),
                ),
            )
        )

    active = production_priors["active"]
    swapped = production_priors["swapped"]
    for bank in ("state", "model"):
        swap_residual = _prior_residual(
            getattr(swapped, f"{bank}_probabilities"),
            tuple(reversed(getattr(active, f"{bank}_probabilities"))),
        )
        invariants.append(
            _invariant(
                f"source_prior.{bank}.parent_assignment_swaps",
                swap_residual <= prior_allowance,
                swap_residual,
                prior_allowance,
                "swapping candidate rows swaps supported assignments",
            )
        )
    control_a = production_priors["target_suffix_a"]
    control_b = production_priors["target_suffix_b"]
    target_suffix_blind = (
        root["target_suffix_controls"][0]["full_token_ids"][1:]
        != root["target_suffix_controls"][1]["full_token_ids"][1:]
        and control_a.state_factor_identity_sha256
        == control_b.state_factor_identity_sha256
        and control_a.model_factor_identity_sha256
        == control_b.model_factor_identity_sha256
    )
    invariants.append(
        _invariant(
            "joint.current_target_and_suffix_blind",
            target_suffix_blind,
            0.0 if target_suffix_blind else 1.0,
            0.0,
            "changed target/suffix bytes never enter the frozen prefix joint",
        )
    )
    active_value = production_objectives["active"].monolithic.value
    swapped_value = production_objectives["swapped"].monolithic.value
    objective_distinction = abs(active_value - swapped_value)
    objective_allowance = math.fsum(
        (
            production_objectives[
                "active"
            ].monolithic.numerical_allowance.total,
            production_objectives[
                "swapped"
            ].monolithic.numerical_allowance.total,
        )
    )
    invariants.append(
        _invariant(
            "joint.parent_swap_changes_complete_objective",
            objective_distinction > objective_allowance,
            objective_distinction,
            objective_allowance,
            "parent assignment changes the complete joint beyond allowance",
        )
    )
    schema_is_v2 = (
        root["generative_factor_schema"]
        == PARENT_SPECIFIC_GENERATIVE_FACTOR_SCHEMA
        and root["generative_factor_schema"]["prior_variant"]
        == "parent_specific_pooled_prefix"
    )
    invariants.append(
        _invariant(
            "schema.parent_specific_scorer_v2",
            schema_is_v2,
            0.0 if schema_is_v2 else 1.0,
            0.0,
            "v1 prefix_conditioned is not deserialized as scorer-v2",
        )
    )
    status = (
        GateStatus.PASS
        if all(item.passed for item in invariants)
        else GateStatus.FAIL
    )
    return ParentSpecificH1PrefixPriorEvaluation(
        status=status,
        fixture_sha256=fixture_sha256,
        generative_factor_schema_sha256=(
            EXPECTED_PARENT_SPECIFIC_GENERATIVE_FACTOR_SCHEMA_SHA256
        ),
        production_priors=MappingProxyType(production_priors),
        independent_priors=MappingProxyType(independent_priors),
        production_objectives=MappingProxyType(production_objectives),
        independent_objectives=MappingProxyType(
            independent_objectives
        ),
        invariants=tuple(invariants),
    )


def _canonical_parent_specific_config(config: object) -> object:
    from vfe4.config import (
        H1PrefixPriorV2ResolvedConfig,
        resolve_h1_prefix_prior_v2_config,
    )

    if type(config) is not H1PrefixPriorV2ResolvedConfig:
        raise ValueError(
            "config must be an exact H1PrefixPriorV2ResolvedConfig"
        )
    try:
        raw = json.loads(config.canonical_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"scorer-v2 config canonical JSON is invalid: {exc}") from exc
    canonical = resolve_h1_prefix_prior_v2_config(raw, repo_root=REPO_ROOT)
    if canonical != config:
        raise ValueError("resolved scorer-v2 config is not canonical")
    if hashlib.sha256(config.canonical_json.encode("utf-8")).hexdigest() != (
        config.config_sha256
    ):
        raise ValueError("scorer-v2 config hash is inconsistent")
    expected_source_sha256 = source_candidate_sha256(
        git_head_value=canonical.source.git_head,
        dirty_digest_value=canonical.source.dirty_digest,
    )
    if canonical.source.source_sha256 != expected_source_sha256:
        raise ValueError(
            "scorer-v2 source_sha256 does not bind its Git head and dirty digest"
        )
    return canonical


def _parent_specific_computation(
    evaluation: ParentSpecificH1PrefixPriorEvaluation,
) -> dict[str, object]:
    return {
        "production_priors": {
            name: asdict(record)
            for name, record in evaluation.production_priors.items()
        },
        "independent_priors": {
            name: asdict(record)
            for name, record in evaluation.independent_priors.items()
        },
        "production_complete_objectives": {
            name: {
                "monolithic": record.monolithic.value,
                "local": record.local.complete_elbo,
            }
            for name, record in evaluation.production_objectives.items()
        },
        "independent_complete_objectives": {
            name: {
                "local": record.local_terms.complete_elbo,
                "identity": record.identity.elbo_from_identity,
            }
            for name, record in evaluation.independent_objectives.items()
        },
    }


def evaluate_parent_specific_h1_prefix_prior_artifact(
    config: object,
    *,
    fixture_bytes: bytes | None = None,
    base_fixture_bytes: bytes | None = None,
    junit_sha256: str | None = None,
) -> ParentSpecificH1PrefixPriorArtifactEvaluation:
    """Evaluate scorer-v2 and bind its typed validation payload."""

    canonical = _canonical_parent_specific_config(config)
    validated_junit = _optional_sha256(junit_sha256, "junit_sha256")
    if fixture_bytes is None:
        captured_fixture = V2_FIXTURE_PATH.read_bytes()
    elif type(fixture_bytes) is bytes:
        captured_fixture = fixture_bytes
    else:
        raise ValueError("fixture_bytes must be immutable bytes")
    if base_fixture_bytes is None:
        captured_base = BASE_FIXTURE_PATH.read_bytes()
    elif type(base_fixture_bytes) is bytes:
        captured_base = base_fixture_bytes
    else:
        raise ValueError("base_fixture_bytes must be immutable bytes")
    fixture_sha256 = hashlib.sha256(captured_fixture).hexdigest()
    base_sha256 = hashlib.sha256(captured_base).hexdigest()
    computation: dict[str, object] | str
    try:
        if fixture_sha256 != canonical.fixture_sha256:
            raise ValueError("scorer-v2 fixture differs from resolved config")
        if base_sha256 != canonical.base_fixture_sha256:
            raise ValueError("base H1 fixture differs from resolved config")
        evaluation = evaluate_parent_specific_h1_prefix_prior(
            captured_fixture,
            base_fixture_bytes=captured_base,
        )
        obligations: tuple[str, ...] = ()
        invariants = evaluation.invariants
        status = evaluation.status
        computation = _parent_specific_computation(evaluation)
    except Exception as exc:
        status = GateStatus.INCONCLUSIVE
        obligations = (
            f"H1 scorer-v2 computation requires resolution: {exc}",
        )
        invariants = (
            InvariantResult(
                "scorer_v2.computation_available",
                False,
                None,
                None,
                "unavailable",
            ),
        )
        computation = "unavailable"
    result = H1PrefixPriorV2GateResult(
        gate="H1-Prefix-Prior",
        status=status,
        fixture_id="h1-prefix-prior-scorer-v2",
        scorer_schema="parent-specific-pooled-prefix-bilinear-v1",
        fixture_sha256=fixture_sha256,
        generative_factor_schema_sha256=(
            EXPECTED_PARENT_SPECIFIC_GENERATIVE_FACTOR_SCHEMA_SHA256
        ),
        invariants=invariants,
        obligations=obligations,
    )
    validation = {
        "schema_version": "h1-prefix-prior-validation-v3",
        "gate": result.gate,
        "status": result.status.value,
        "obligations": result.obligations,
        "git_head": canonical.source.git_head,
        "dirty_digest": canonical.source.dirty_digest,
        "source_sha256": canonical.source.source_sha256,
        "config_sha256": canonical.config_sha256,
        "junit_sha256": validated_junit,
        "fixture_id": result.fixture_id,
        "fixture_sha256": fixture_sha256,
        "base_fixture_sha256": base_sha256,
        "generative_factor_schema_sha256": (
            result.generative_factor_schema_sha256
        ),
        "scorer_schema": result.scorer_schema,
        "latent_projection_policy": canonical.latent_projection_policy,
        "parent_history_policy": canonical.parent_history_policy,
        "invariants": tuple(asdict(item) for item in result.invariants),
        "computation": computation,
    }
    return ParentSpecificH1PrefixPriorArtifactEvaluation(
        result=result,
        validation_payload=validation,
        fixture_sha256=fixture_sha256,
        base_fixture_sha256=base_sha256,
        generative_factor_schema_bytes=(
            PARENT_SPECIFIC_GENERATIVE_FACTOR_SCHEMA_BYTES
        ),
    )


def parent_specific_h1_prefix_prior_artifact_payloads(
    config: object,
    evaluation: ParentSpecificH1PrefixPriorArtifactEvaluation,
) -> dict[str, object]:
    """Return the exact scorer-v2 payload inventory for its typed reference."""

    canonical = _canonical_parent_specific_config(config)
    if type(evaluation) is not ParentSpecificH1PrefixPriorArtifactEvaluation:
        raise ValueError(
            "evaluation must be a ParentSpecificH1PrefixPriorArtifactEvaluation"
        )
    validation = evaluation.validation_payload
    if (
        validation.get("config_sha256") != canonical.config_sha256
        or validation.get("git_head") != canonical.source.git_head
        or validation.get("dirty_digest") != canonical.source.dirty_digest
        or validation.get("source_sha256")
        != canonical.source.source_sha256
        or validation.get("generative_factor_schema_sha256")
        != canonical.generative_factor_schema_sha256
        or evaluation.generative_factor_schema_bytes
        != PARENT_SPECIFIC_GENERATIVE_FACTOR_SCHEMA_BYTES
    ):
        raise ValueError("scorer-v2 evaluation does not match its resolved config")
    return {
        "config.json": json.loads(canonical.canonical_json),
        "schemas/generative_factor.json": json.loads(
            PARENT_SPECIFIC_GENERATIVE_FACTOR_SCHEMA_BYTES.decode("utf-8")
        ),
        "validation/h1_prefix_prior.json": json.loads(
            canonical_h6_json_bytes(validation).decode("utf-8")
        ),
    }


def run_parent_specific_h1_prefix_prior(
    config: object,
    *,
    junit_sha256: str | None = None,
) -> tuple[H1PrefixPriorV2GateResult, Path]:
    """Evaluate and atomically publish the scorer-v2 prerequisite."""

    canonical = _canonical_parent_specific_config(config)
    observed_head, observed_dirty, observed_source = current_source_identity(
        REPO_ROOT,
        canonical.artifact_root,
    )
    if (
        canonical.source.git_head != observed_head
        or canonical.source.dirty_digest != observed_dirty
        or canonical.source.source_sha256 != observed_source
    ):
        raise ValueError(
            "scorer-v2 source identity differs from the live candidate"
        )
    evaluation = evaluate_parent_specific_h1_prefix_prior_artifact(
        canonical,
        junit_sha256=junit_sha256,
    )
    run_dir = publish_run_directory(
        canonical.artifact_root,
        _run_name(_utc_now(), canonical.config_sha256),
        parent_specific_h1_prefix_prior_artifact_payloads(
            canonical,
            evaluation,
        ),
    )
    return evaluation.result, run_dir


def _closed_evaluation(
    *,
    root: dict[str, Any],
    config: H1PrefixPriorResolvedConfig,
    fixture_bytes: bytes,
    base_fixture_bytes: bytes,
) -> tuple[
    H1PrefixPriorGateResult,
    dict[str, object],
]:
    active = _production_prior(root, config)
    alternate = _production_prior(root, config, case_id="prefix-token-2")
    current_target = _production_prior(root, config, use_current_target=True)
    production = _production_h1(base_fixture_bytes, active, config)
    current_target_h1 = _production_h1(base_fixture_bytes, current_target, config)
    oracle: H1PrefixPriorOracleRecord = evaluate_h1_prefix_prior_oracle(
        fixture_bytes,
        base_fixture_bytes=base_fixture_bytes,
    )
    independent_fixture = parse_h1_prefix_prior_fixture(fixture_bytes)
    alternate_oracle = prefix_prior_probabilities(
        independent_fixture,
        case_id="prefix-token-2",
    )

    measurements = {
        "monolithic_elbo": production.monolithic.value,
        "local_elbo": production.local.complete_elbo,
        "evidence_minus_posterior_kl": oracle.identity.elbo_from_identity,
    }
    pairwise = {
        "monolithic_vs_local": pair_comparison(
            production.monolithic.value,
            production.local.complete_elbo,
            production.monolithic.numerical_allowance.total,
            production.local.allowances.complete_elbo.total,
        ),
        "monolithic_vs_identity": pair_comparison(
            production.monolithic.value,
            oracle.identity.elbo_from_identity,
            production.monolithic.numerical_allowance.total,
            oracle.identity.identity_allowance.total,
        ),
        "local_vs_identity": pair_comparison(
            production.local.complete_elbo,
            oracle.identity.elbo_from_identity,
            production.local.allowances.complete_elbo.total,
            oracle.identity.identity_allowance.total,
        ),
    }
    terms = {
        name: pair_comparison(
            _term_value(production.local, name),
            _term_value(oracle.local_terms, name),
            _term_allowance(production.local, name),
            _term_allowance(oracle.local_terms, name),
        )
        for name in TERM_NAMES
    }
    convergence = {
        "active.monolithic": float(
            production.monolithic.numerical_allowance.convergence_estimate
        ),
        **{
            f"active.local.{name}": _term_convergence(production.local, name)
            for name in TERM_NAMES
        },
        **{
            f"active.independent.{name}": _term_convergence(
                oracle.local_terms, name
            )
            for name in TERM_NAMES
        },
        "active.identity.posterior_kl": float(
            oracle.identity.posterior_kl_allowance.convergence_estimate
        ),
        "active.identity.elbo": float(
            oracle.identity.identity_allowance.convergence_estimate
        ),
        "active.identity.evidence.probability": float(
            oracle.identity.evidence.probability_allowance.convergence_estimate
        ),
        "active.identity.evidence.log_probability": float(
            oracle.identity.evidence.log_probability_allowance.convergence_estimate
        ),
        "negative.current_target.monolithic": float(
            current_target_h1.monolithic.numerical_allowance.convergence_estimate
        ),
    }
    if tuple(convergence) != H1_PREFIX_PRIOR_CONVERGENCE_NAMES:
        raise ValueError("H1 prefix-prior convergence inventory mismatch")
    prior_allowance = 256.0 * _EPSILON
    active_state_oracle_residual = _prior_residual(
        active.state_probabilities, oracle.probabilities.state_probabilities
    )
    active_model_oracle_residual = _prior_residual(
        active.model_probabilities, oracle.probabilities.model_probabilities
    )
    alternate_state_oracle_residual = _prior_residual(
        alternate.state_probabilities,
        alternate_oracle.state_probabilities,
    )
    alternate_model_oracle_residual = _prior_residual(
        alternate.model_probabilities,
        alternate_oracle.model_probabilities,
    )
    normalization_allowance = 256.0 * _EPSILON
    state_distinction = _prior_residual(
        active.state_probabilities, alternate.state_probabilities
    )
    model_distinction = _prior_residual(
        active.model_probabilities, alternate.model_probabilities
    )
    negative = pair_comparison(
        production.monolithic.value,
        current_target_h1.monolithic.value,
        production.monolithic.numerical_allowance.total,
        current_target_h1.monolithic.numerical_allowance.total,
    )
    parameters = root["prefix_prior_parameters"]
    latent_zero = (
        parameters["state_latent_projection"] == [["0x0.0p+0"]]
        and parameters["model_latent_projection"] == [["0x0.0p+0"]]
        and root["earlier_latents"] == [["0x0.0p+0"], ["0x0.0p+0"]]
    )

    invariants: list[InvariantResult] = [
        _invariant(
            name,
            comparison.passed,
            comparison.residual,
            comparison.allowance,
            "calibrated H1 pair residual is within composed allowance",
        )
        for name, comparison in pairwise.items()
    ]
    invariants.extend(
        _invariant(
            f"oracle.{name}",
            comparison.passed,
            comparison.residual,
            comparison.allowance,
            "production local term matches the independent NumPy term",
        )
        for name, comparison in terms.items()
    )
    invariants.extend(
        _invariant(
            f"convergence.{name}",
            math.isfinite(value)
            and 0.0 <= value <= config.maximum_convergence_estimate,
            value,
            config.maximum_convergence_estimate,
            "finite nonnegative convergence estimate is within the configured maximum",
        )
        for name, value in convergence.items()
    )
    invariants.extend(
        (
            _invariant(
                "source_prior.state.production_vs_oracle.active",
                active_state_oracle_residual <= prior_allowance,
                active_state_oracle_residual,
                prior_allowance,
                "active Torch and independent NumPy state priors agree",
            ),
            _invariant(
                "source_prior.model.production_vs_oracle.active",
                active_model_oracle_residual <= prior_allowance,
                active_model_oracle_residual,
                prior_allowance,
                "active Torch and independent NumPy model priors agree",
            ),
            _invariant(
                "source_prior.state.production_vs_oracle.alternate",
                alternate_state_oracle_residual <= prior_allowance,
                alternate_state_oracle_residual,
                prior_allowance,
                "alternate Torch and independent NumPy state priors agree",
            ),
            _invariant(
                "source_prior.model.production_vs_oracle.alternate",
                alternate_model_oracle_residual <= prior_allowance,
                alternate_model_oracle_residual,
                prior_allowance,
                "alternate Torch and independent NumPy model priors agree",
            ),
        )
    )
    for label, record in (
        ("active", active),
        ("alternate", alternate),
        ("current_target_control", current_target),
    ):
        state_residual = _normalization_residual(record.state_probabilities)
        model_residual = _normalization_residual(record.model_probabilities)
        invariants.extend(
            (
                _invariant(
                    f"source_prior.state.normalized.{label}",
                    state_residual <= normalization_allowance,
                    state_residual,
                    normalization_allowance,
                    "state source probabilities sum to one",
                ),
                _invariant(
                    f"source_prior.model.normalized.{label}",
                    model_residual <= normalization_allowance,
                    model_residual,
                    normalization_allowance,
                    "model source probabilities sum to one",
                ),
            )
        )
    invariants.extend(
        (
            _invariant(
                "source_prior.state.prefix_distinct",
                state_distinction > prior_allowance,
                state_distinction,
                prior_allowance,
                "the two prior-token prefixes induce distinct state priors",
            ),
            _invariant(
                "source_prior.model.prefix_distinct",
                model_distinction > prior_allowance,
                model_distinction,
                prior_allowance,
                "the two prior-token prefixes induce distinct model priors",
            ),
            _invariant(
                "schema.target_blind",
                root["generative_factor_schema"] == GENERATIVE_FACTOR_SCHEMA,
                1.0
                if root["generative_factor_schema"] == GENERATIVE_FACTOR_SCHEMA
                else 0.0,
                1.0,
                "schema admits prior tokens and forbids current-target flow",
            ),
            _invariant(
                "latent_projection.exact_zero",
                latent_zero,
                0.0 if latent_zero else 1.0,
                0.0,
                "both latent projections and the bounded latent history are zero",
            ),
            _invariant(
                "negative.current_target_as_prefix",
                negative.residual > negative.allowance,
                negative.residual,
                negative.allowance,
                "supplying the current target changes the H1 objective beyond allowance",
            ),
        )
    )
    if tuple(item.name for item in invariants) != H1_PREFIX_PRIOR_INVARIANT_NAMES:
        raise ValueError("H1 prefix-prior invariant inventory mismatch")
    all_comparisons: tuple[Comparison, ...] = (
        *pairwise.values(),
        *terms.values(),
    )
    status = GateStatus.PASS if all(item.passed for item in invariants) else GateStatus.FAIL
    result = H1PrefixPriorGateResult(
        gate="H1-Prefix-Prior",
        status=status,
        fixture_id="h1-prefix-prior-v1",
        residual=max(item.residual for item in all_comparisons),
        calibrated_allowance=max(item.allowance for item in all_comparisons),
        measurements=measurements,
        invariants=tuple(invariants),
        obligations=(),
    )
    negative_payload = {
        "current_target_as_prefix": {
            "active_prefix_token_ids": active.prefix_token_ids,
            "supplied_current_target_token_ids": current_target.prefix_token_ids,
            "correct_value": production.monolithic.value,
            "wrong_value": current_target_h1.monolithic.value,
            "residual": negative.residual,
            "allowance": negative.allowance,
            "passed": negative.residual > negative.allowance,
        }
    }
    computation = {
        "source_priors": {
            "active": asdict(active),
            "alternate": asdict(alternate),
            "current_target_control": asdict(current_target),
            "independent_active": asdict(oracle.probabilities),
            "independent_alternate": asdict(alternate_oracle),
        },
        "convergence_estimates": convergence,
        "monolithic": {
            "value": production.monolithic.value,
            "allowance": production.monolithic.numerical_allowance.total,
        },
        "local": {
            "complete_elbo": production.local.complete_elbo,
            "allowance": production.local.allowances.complete_elbo.total,
        },
        "independent_identity": {
            "log_evidence": oracle.identity.evidence.log_probability,
            "posterior_kl": oracle.identity.posterior_kl,
            "elbo_from_identity": oracle.identity.elbo_from_identity,
            "allowance": oracle.identity.identity_allowance.total,
            "derived_h1_fixture_sha256": oracle.derived_h1_fixture_sha256,
        },
        "pairwise_comparisons": {
            name: asdict(comparison) for name, comparison in pairwise.items()
        },
        "term_comparisons": {
            name: asdict(comparison) for name, comparison in terms.items()
        },
        "negative_controls": negative_payload,
    }
    return result, computation


def _inconclusive(reason: str) -> H1PrefixPriorGateResult:
    return H1PrefixPriorGateResult(
        gate="H1-Prefix-Prior",
        status=GateStatus.INCONCLUSIVE,
        fixture_id="h1-prefix-prior-v1",
        residual=None,
        calibrated_allowance=None,
        measurements={name: None for name in H1_PREFIX_PRIOR_MEASUREMENT_NAMES},
        invariants=tuple(
            InvariantResult(name, False, None, None, "unavailable")
            for name in H1_PREFIX_PRIOR_INVARIANT_NAMES
        ),
        obligations=(reason,),
    )


def _validation_payload(
    config: H1PrefixPriorResolvedConfig,
    result: H1PrefixPriorGateResult,
    *,
    fixture_sha256: str | None,
    base_fixture_sha256: str | None,
    computation: dict[str, object] | None,
) -> dict[str, object]:
    measurements = dict(result.measurements)
    return {
        "schema_version": "h1-prefix-prior-validation-v1",
        "gate": "H1-Prefix-Prior",
        "status": result.status.value,
        "obligations": result.obligations,
        "git_head": config.source.git_head,
        "dirty_digest": config.source.dirty_digest,
        "source_sha256": config.source.source_sha256,
        "config_sha256": config.config_sha256,
        "fixture_id": config.fixture_id,
        "fixture_sha256": fixture_sha256,
        "base_fixture_sha256": base_fixture_sha256,
        "generative_factor_schema_sha256": (
            config.generative_factor_schema_sha256
        ),
        "measurements": measurements,
        "measurement_hex": {
            name: None if value is None else float(value).hex()
            for name, value in measurements.items()
        },
        "residual": result.residual,
        "calibrated_allowance": result.calibrated_allowance,
        "invariants": tuple(asdict(item) for item in result.invariants),
        "negative_controls": (
            {}
            if computation is None
            else computation["negative_controls"]
        ),
        "computation": "unavailable" if computation is None else computation,
    }


def _canonical_config(
    config: H1PrefixPriorResolvedConfig,
) -> H1PrefixPriorResolvedConfig:
    if type(config) is not H1PrefixPriorResolvedConfig:
        raise ValueError("config must be an exact H1PrefixPriorResolvedConfig")
    try:
        raw = json.loads(config.canonical_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"config canonical JSON is invalid: {exc}") from exc
    canonical = resolve_h1_prefix_prior_config(raw, repo_root=REPO_ROOT)
    if canonical != config:
        raise ValueError("resolved H1 prefix-prior config is not canonical")
    if hashlib.sha256(config.canonical_json.encode("utf-8")).hexdigest() != (
        config.config_sha256
    ):
        raise ValueError("H1 prefix-prior config hash is inconsistent")
    return canonical


def evaluate_h1_prefix_prior(
    config: H1PrefixPriorResolvedConfig,
    *,
    fixture_bytes: bytes | None = None,
    base_fixture_bytes: bytes | None = None,
) -> H1PrefixPriorGateEvaluation:
    """Evaluate the bounded prerequisite from immutable snapshots without publishing."""

    canonical = _canonical_config(config)
    if fixture_bytes is None:
        try:
            captured_fixture = FIXTURE_PATH.read_bytes()
        except OSError:
            captured_fixture = None
    else:
        captured_fixture = fixture_bytes if type(fixture_bytes) is bytes else None
    if base_fixture_bytes is None:
        try:
            captured_base = BASE_FIXTURE_PATH.read_bytes()
        except OSError:
            captured_base = None
    else:
        captured_base = (
            base_fixture_bytes if type(base_fixture_bytes) is bytes else None
        )
    fixture_sha256 = (
        None
        if captured_fixture is None
        else hashlib.sha256(captured_fixture).hexdigest()
    )
    base_sha256 = (
        None if captured_base is None else hashlib.sha256(captured_base).hexdigest()
    )
    computation: dict[str, object] | None = None
    try:
        if captured_fixture is None or captured_base is None:
            raise ValueError("both prefix-prior and base H1 fixture bytes are required")
        if fixture_sha256 != EXPECTED_H1_PREFIX_PRIOR_FIXTURE_SHA256:
            raise ValueError("prefix-prior fixture raw hash is not frozen")
        if base_sha256 != EXPECTED_H1_BASE_FIXTURE_SHA256:
            raise ValueError("base H1 fixture raw hash is not frozen")
        root = _fixture_root(captured_fixture, canonical)
        result, computation = _closed_evaluation(
            root=root,
            config=canonical,
            fixture_bytes=captured_fixture,
            base_fixture_bytes=captured_base,
        )
    except Exception as exc:
        result = _inconclusive(
            f"H1 prefix-prior computation requires resolution: {exc}"
        )
    payload = _validation_payload(
        canonical,
        result,
        fixture_sha256=fixture_sha256,
        base_fixture_sha256=base_sha256,
        computation=computation,
    )
    return H1PrefixPriorGateEvaluation(
        result,
        payload,
        fixture_sha256,
        base_sha256,
        GENERATIVE_FACTOR_SCHEMA_BYTES,
    )


def h1_prefix_prior_artifact_payloads(
    config: H1PrefixPriorResolvedConfig,
    evaluation: H1PrefixPriorGateEvaluation,
) -> dict[str, object]:
    """Return the exact canonical payload inventory required by the typed reference."""

    canonical = _canonical_config(config)
    if type(evaluation) is not H1PrefixPriorGateEvaluation:
        raise ValueError("evaluation must be an H1PrefixPriorGateEvaluation")
    validation = evaluation.validation_payload
    if (
        validation.get("config_sha256") != canonical.config_sha256
        or validation.get("git_head") != canonical.source.git_head
        or validation.get("dirty_digest") != canonical.source.dirty_digest
        or validation.get("generative_factor_schema_sha256")
        != canonical.generative_factor_schema_sha256
    ):
        raise ValueError("evaluation payload does not match the resolved config")
    if evaluation.generative_factor_schema_bytes != GENERATIVE_FACTOR_SCHEMA_BYTES:
        raise ValueError("evaluation schema bytes differ from the frozen schema")
    canonical_validation = json.loads(
        canonical_h6_json_bytes(validation).decode("utf-8")
    )
    return {
        "config.json": json.loads(canonical.canonical_json),
        "schemas/generative_factor.json": json.loads(
            GENERATIVE_FACTOR_SCHEMA_BYTES.decode("utf-8")
        ),
        "validation/h1_prefix_prior.json": canonical_validation,
    }


def run_h1_prefix_prior(
    config: H1PrefixPriorResolvedConfig,
) -> tuple[H1PrefixPriorGateResult, Path]:
    """Evaluate and atomically publish the separate H1 prefix-prior artifact."""

    canonical = _canonical_config(config)
    observed_head = git_head(REPO_ROOT)
    observed_dirty = dirty_content_digest(REPO_ROOT, canonical.artifact_root)
    if (
        canonical.source.git_head != observed_head
        or canonical.source.dirty_digest != observed_dirty
    ):
        raise ValueError(
            "H1 prefix-prior source revision/digest differs from the live candidate"
        )
    evaluation = evaluate_h1_prefix_prior(canonical)
    run_dir = publish_run_directory(
        canonical.artifact_root,
        _run_name(_utc_now(), canonical.config_sha256),
        h1_prefix_prior_artifact_payloads(canonical, evaluation),
    )
    return evaluation.result, run_dir


__all__ = [
    "EXPECTED_GENERATIVE_FACTOR_SCHEMA_SHA256",
    "EXPECTED_H1_BASE_FIXTURE_SHA256",
    "EXPECTED_H1_PREFIX_PRIOR_FIXTURE_SHA256",
    "EXPECTED_H1_PREFIX_PRIOR_V2_FIXTURE_SHA256",
    "EXPECTED_PARENT_SPECIFIC_GENERATIVE_FACTOR_SCHEMA_SHA256",
    "GENERATIVE_FACTOR_SCHEMA",
    "GENERATIVE_FACTOR_SCHEMA_BYTES",
    "H1PrefixPriorGateEvaluation",
    "H1_PREFIX_PRIOR_INVARIANT_NAMES",
    "H1_PREFIX_PRIOR_MEASUREMENT_NAMES",
    "PARENT_SPECIFIC_GENERATIVE_FACTOR_SCHEMA",
    "PARENT_SPECIFIC_GENERATIVE_FACTOR_SCHEMA_BYTES",
    "ParentSpecificH1PrefixPriorArtifactEvaluation",
    "ParentSpecificH1PrefixPriorEvaluation",
    "ParentSpecificProductionPriorRecord",
    "evaluate_h1_prefix_prior",
    "evaluate_parent_specific_h1_prefix_prior",
    "evaluate_parent_specific_h1_prefix_prior_artifact",
    "h1_prefix_prior_artifact_payloads",
    "parent_specific_h1_prefix_prior_artifact_payloads",
    "run_h1_prefix_prior",
    "run_parent_specific_h1_prefix_prior",
]
