"""Fail-closed H1 gate for the prefix-conditioned source-prior variant."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from verification.h1_gate import TERM_NAMES, Comparison, pair_comparison
from verification.numpy_oracles.h1_prefix_prior import (
    H1PrefixPriorOracleRecord,
    evaluate_h1_prefix_prior_oracle,
    parse_h1_prefix_prior_fixture,
    prefix_prior_probabilities,
)
from vfe4.artifacts import (
    canonical_json_bytes,
    dirty_content_digest,
    git_head,
    publish_run_directory,
)
from vfe4.config import (
    H1PrefixPriorResolvedConfig,
    resolve_h1_prefix_prior_config,
)
from vfe4.data.windows import CausalPrefix
from vfe4.generative import H1GenerativeModel, PrefixConditionedSourcePrior
from vfe4.objective import (
    MonolithicElboResult,
    evaluate_local_elbo,
    evaluate_monolithic_elbo,
)
from vfe4.recognition import H1RecognitionLaw
from vfe4.types.h6 import (
    CausalDag,
    CausalDagRow,
    H6LanguageStructure,
    VocabularyIdentity,
    ZeroDimensionalBase,
    canonical_json_bytes as canonical_h6_json_bytes,
)
from vfe4.types.results import (
    ElboTerms,
    GateStatus,
    H1PrefixPriorGateResult,
    InvariantResult,
)
from vfe4.validation import load_h1_fixture


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = (
    REPO_ROOT / "vfe4" / "validation" / "fixtures" / "h1_prefix_prior_v1.json"
)
BASE_FIXTURE_PATH = (
    REPO_ROOT / "vfe4" / "validation" / "fixtures" / "h1_v1.json"
)
EXPECTED_H1_PREFIX_PRIOR_FIXTURE_SHA256 = (
    "b6638ea3b64c7fd68882cbaced914e4d17d2cd03c8b6b8a939fd575a1b9f43f1"
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
class H1PrefixPriorGateEvaluation:
    result: H1PrefixPriorGateResult
    validation_payload: dict[str, object]
    fixture_sha256: str | None
    base_fixture_sha256: str | None
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
    prior = PrefixConditionedSourcePrior(
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
        prior.state_parent_keys[1].copy_(state_keys)
        prior.model_parent_keys[1].copy_(model_keys)
        prior.state_biases[1].copy_(state_biases)
        prior.model_biases[1].copy_(model_biases)
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
    "GENERATIVE_FACTOR_SCHEMA",
    "GENERATIVE_FACTOR_SCHEMA_BYTES",
    "H1PrefixPriorGateEvaluation",
    "H1_PREFIX_PRIOR_INVARIANT_NAMES",
    "H1_PREFIX_PRIOR_MEASUREMENT_NAMES",
    "evaluate_h1_prefix_prior",
    "h1_prefix_prior_artifact_payloads",
    "run_h1_prefix_prior",
]
