"""Complete deterministic H5 ELBO with an auditable factor-by-factor trace."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from itertools import product
from typing import Literal, Protocol, Self

import torch

from vfe4.numerics.h5_budget import (
    H5CompleteAllowance,
    H5TermAllowance,
    complete_elbo_allowance,
    term_allowance,
)
from vfe4.numerics.quadrature import probabilists_gauss_hermite
from vfe4.types.h5_schema import (
    H5_ANALYTIC_FACTOR_OPERATION_COUNTS,
    H5_ANALYTIC_OPERATION_COUNTS,
    H5_C,
    H5_FACTOR_INPUT_DOMAIN,
    H5_FACTOR_INPUT_FIELDS,
    H5_FACTOR_INPUT_SCHEMA_DOMAIN,
    H5_FACTOR_INPUT_SCHEMA_SHA256,
    H5_FACTOR_INPUT_SCHEMA_VERSION,
    H5_FACTOR_UNIVERSE,
    H5_OBJECTIVE_SCHEMA_SHA256,
    H5_QUADRATURE_ORDERS,
    H5_RECOGNITION_COORDINATE_UNIVERSE,
    H5_SIGNED_TERM_IDS,
    H5_SIGNED_TERM_SIGNS,
    canonical_h5_factor_input_schema_core_bytes,
    emission_operation_count,
    gamma_n,
)
from vfe4.types.results import ElboTermAllowances, ElboTerms, NumericalAllowance
from vfe4.types.updates import (
    CategoricalRecognitionCoordinate,
    GaussianRecognitionCoordinate,
    H5CandidateSnapshot,
    H5LiveState,
    H5ModelSnapshot,
    H5ReferenceState,
    RecognitionSnapshot,
    h5_semantic_state_sha256,
)


_LOWER_HEX = frozenset("0123456789abcdef")
_GAUSSIAN_IDS = H5_RECOGNITION_COORDINATE_UNIVERSE[:6]
_CATEGORICAL_IDS = H5_RECOGNITION_COORDINATE_UNIVERSE[6:]
_LOG_2_PI = math.log(2.0 * math.pi)
_MISSING_CACHE_ENTRY = object()


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase 64-hex SHA-256")
    return value


def _finite(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite binary64 float")
    return value


def _finite_tuple(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
) -> tuple[float, ...]:
    if type(value) is not tuple or not value:
        raise ValueError(f"{name} must be a nonempty tuple")
    checked: list[float] = []
    for index, item in enumerate(value):
        number = _finite(item, f"{name}[{index}]")
        if minimum is not None and number < minimum:
            raise ValueError(f"{name}[{index}] must be at least {minimum}")
        checked.append(number)
    return tuple(checked)


def _canonicalize(value: object) -> object:
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("canonical H5 factor inputs must be finite")
        return value.hex()
    if type(value) in (str, int, bool) or value is None:
        return value
    if type(value) in (tuple, list):
        return [_canonicalize(item) for item in value]
    if isinstance(value, Mapping):
        if not all(type(key) is str and key for key in value):
            raise ValueError("canonical H5 factor-input keys must be nonempty strings")
        return {key: _canonicalize(value[key]) for key in sorted(value)}
    raise ValueError(f"unsupported H5 factor-input value: {type(value).__name__}")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        _canonicalize(value),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


class CacheDisposition(str, Enum):
    REEVALUATED = "reevaluated"
    REUSED = "reused"


@dataclass(frozen=True)
class FactorInputHashRecord:
    factor_id: str
    input_schema_version: Literal["h5-factor-input-v1"]
    input_schema_sha256: str = field(init=False)
    canonical_input_bytes: bytes
    input_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.factor_id not in H5_FACTOR_UNIVERSE:
            raise ValueError("factor_id is outside the H5 factor universe")
        if self.input_schema_version != H5_FACTOR_INPUT_SCHEMA_VERSION:
            raise ValueError("unsupported H5 factor-input schema version")
        if type(self.canonical_input_bytes) is not bytes:
            raise ValueError("canonical_input_bytes must be bytes")
        canonical = memoryview(self.canonical_input_bytes).tobytes()
        if not canonical.startswith(H5_FACTOR_INPUT_DOMAIN):
            raise ValueError("canonical_input_bytes must include the H5 input domain")
        try:
            decoded = json.loads(canonical[len(H5_FACTOR_INPUT_DOMAIN) :])
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("canonical_input_bytes are not canonical JSON") from exc
        if type(decoded) is not dict or set(decoded) != set(H5_FACTOR_INPUT_FIELDS):
            raise ValueError("canonical factor input must contain exactly five fields")
        if (
            decoded["schema_version"] != H5_FACTOR_INPUT_SCHEMA_VERSION
            or decoded["factor_id"] != self.factor_id
        ):
            raise ValueError("canonical factor input identity does not match its record")
        reencoded = H5_FACTOR_INPUT_DOMAIN + json.dumps(
            decoded,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        if reencoded != canonical:
            raise ValueError("canonical factor input JSON is not normalized")
        schema_hash = hashlib.sha256(
            H5_FACTOR_INPUT_SCHEMA_DOMAIN
            + canonical_h5_factor_input_schema_core_bytes()
        ).hexdigest()
        if schema_hash != H5_FACTOR_INPUT_SCHEMA_SHA256:
            raise ValueError("factor-input schema constant failed recomputation")
        object.__setattr__(self, "canonical_input_bytes", canonical)
        object.__setattr__(self, "input_schema_sha256", schema_hash)
        object.__setattr__(self, "input_sha256", hashlib.sha256(canonical).hexdigest())


def _expected_factor_counts(factor_id: str) -> tuple[int, int]:
    if factor_id in ("emission[1]", "emission[2]"):
        return emission_operation_count(21), emission_operation_count(17)
    if factor_id in H5_ANALYTIC_FACTOR_OPERATION_COUNTS:
        count = H5_ANALYTIC_FACTOR_OPERATION_COUNTS[factor_id]
        return count, count
    raise ValueError("factor_id is outside the frozen operation-count table")


@dataclass(frozen=True)
class FactorEvaluationRecord:
    factor_id: str
    input_hash: FactorInputHashRecord
    frozen_complement_sha256: str
    value_order_21: float
    value_order_17: float
    absolute_summands_order_21: tuple[float, ...]
    absolute_summands_order_17: tuple[float, ...]
    condition_numbers_order_21: tuple[float, ...]
    condition_numbers_order_17: tuple[float, ...]
    operation_count_order_21: int
    operation_count_order_17: int
    cache_disposition: CacheDisposition

    def __post_init__(self) -> None:
        if self.factor_id not in H5_FACTOR_UNIVERSE:
            raise ValueError("factor_id is outside the H5 factor universe")
        if not isinstance(self.input_hash, FactorInputHashRecord):
            raise ValueError("input_hash must be a FactorInputHashRecord")
        if self.input_hash.factor_id != self.factor_id:
            raise ValueError("factor_id must equal input_hash.factor_id")
        _require_sha256(self.frozen_complement_sha256, "frozen_complement_sha256")
        value_21 = _finite(self.value_order_21, "value_order_21")
        value_17 = _finite(self.value_order_17, "value_order_17")
        if self.factor_id not in ("emission[1]", "emission[2]") and (
            value_21.hex() != value_17.hex()
        ):
            raise ValueError("analytic H5 factor values must be order-identical")
        summands_21 = _finite_tuple(
            self.absolute_summands_order_21,
            "absolute_summands_order_21",
            minimum=0.0,
        )
        summands_17 = _finite_tuple(
            self.absolute_summands_order_17,
            "absolute_summands_order_17",
            minimum=0.0,
        )
        conditions_21 = _finite_tuple(
            self.condition_numbers_order_21,
            "condition_numbers_order_21",
            minimum=1.0,
        )
        conditions_17 = _finite_tuple(
            self.condition_numbers_order_17,
            "condition_numbers_order_17",
            minimum=1.0,
        )
        if (
            type(self.operation_count_order_21) is not int
            or type(self.operation_count_order_17) is not int
            or (
                self.operation_count_order_21,
                self.operation_count_order_17,
            )
            != _expected_factor_counts(self.factor_id)
        ):
            raise ValueError("factor operation counts do not match the frozen H5 table")
        if not isinstance(self.cache_disposition, CacheDisposition):
            raise ValueError("cache_disposition must be a CacheDisposition")
        object.__setattr__(self, "absolute_summands_order_21", summands_21)
        object.__setattr__(self, "absolute_summands_order_17", summands_17)
        object.__setattr__(self, "condition_numbers_order_21", conditions_21)
        object.__setattr__(self, "condition_numbers_order_17", conditions_17)


@dataclass(frozen=True)
class FactorCacheKey:
    factor_id: str
    input_hash: FactorInputHashRecord
    quadrature_orders: tuple[Literal[21], Literal[17]]
    frozen_complement_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.input_hash, FactorInputHashRecord):
            raise ValueError("input_hash must be a FactorInputHashRecord")
        if self.factor_id != self.input_hash.factor_id:
            raise ValueError("factor_id must equal input_hash.factor_id")
        if self.quadrature_orders != H5_QUADRATURE_ORDERS:
            raise ValueError("quadrature_orders must equal (21, 17)")
        _require_sha256(self.frozen_complement_sha256, "frozen_complement_sha256")
        object.__setattr__(self, "quadrature_orders", tuple(self.quadrature_orders))


@dataclass(frozen=True)
class FactorCacheEntry:
    key: FactorCacheKey
    record: FactorEvaluationRecord

    def __post_init__(self) -> None:
        if not isinstance(self.key, FactorCacheKey):
            raise ValueError("key must be a FactorCacheKey")
        if not isinstance(self.record, FactorEvaluationRecord):
            raise ValueError("record must be a FactorEvaluationRecord")


class StaleFactorCacheError(ValueError):
    """Raised when an exact H5 cache key carries an inconsistent payload."""

    def __init__(self, factor_id: str) -> None:
        self.factor_id = factor_id
        super().__init__(f"stale H5 factor cache entry for {factor_id}")


@dataclass(frozen=True, init=False)
class CompleteElboEvaluation:
    terms: ElboTerms
    factor_records: tuple[FactorEvaluationRecord, ...]
    term_allowances: tuple[H5TermAllowance, ...]
    diagnostic_allowances: tuple[H5TermAllowance, ...]
    complete_allowance: H5CompleteAllowance
    objective_schema_sha256: str = field(init=False)
    evaluated_state_sha256: str = field(init=False)
    frozen_complement_sha256: str

    @classmethod
    def build(
        cls,
        *,
        state: H5LiveState | H5CandidateSnapshot,
        terms: ElboTerms,
        factor_records: tuple[FactorEvaluationRecord, ...],
        term_allowances: tuple[H5TermAllowance, ...],
        diagnostic_allowances: tuple[H5TermAllowance, ...],
        complete_allowance: H5CompleteAllowance,
        frozen_complement_sha256: str,
    ) -> Self:
        if not isinstance(state, (H5LiveState, H5CandidateSnapshot)):
            raise ValueError("state must be an H5 live state or candidate")
        if not isinstance(terms, ElboTerms):
            raise ValueError("terms must be ElboTerms")
        complement = _require_sha256(
            frozen_complement_sha256, "frozen_complement_sha256"
        )
        if (
            type(factor_records) is not tuple
            or tuple(record.factor_id for record in factor_records)
            != H5_FACTOR_UNIVERSE
            or not all(isinstance(record, FactorEvaluationRecord) for record in factor_records)
        ):
            raise ValueError("factor_records must equal the H5 factor-universe order")
        if any(record.frozen_complement_sha256 != complement for record in factor_records):
            raise ValueError("all factor records must bind the same frozen complement")
        if (
            type(term_allowances) is not tuple
            or tuple(item.term_id for item in term_allowances) != H5_SIGNED_TERM_IDS
            or not all(isinstance(item, H5TermAllowance) for item in term_allowances)
        ):
            raise ValueError("term_allowances must equal the signed H5 term order")
        if (
            type(diagnostic_allowances) is not tuple
            or tuple(item.term_id for item in diagnostic_allowances)
            != ("joint_recognition_entropy",)
            or not all(isinstance(item, H5TermAllowance) for item in diagnostic_allowances)
        ):
            raise ValueError("diagnostic_allowances must contain joint entropy only")
        if not isinstance(complete_allowance, H5CompleteAllowance):
            raise ValueError("complete_allowance must be an H5CompleteAllowance")
        if complete_allowance.term_allowances != term_allowances:
            raise ValueError("complete allowance must embed the signed term allowances")
        expected_values = _elbo_term_values(terms)
        if tuple(value.hex() for value in expected_values) != tuple(
            item.value_order_21.hex() for item in term_allowances
        ):
            raise ValueError("ElboTerms do not match the order-21 term trace")
        signed_21 = math.fsum(item.signed_reported_value for item in term_allowances)
        signed_17 = math.fsum(
            item.objective_sign * item.value_order_17 for item in term_allowances
        )
        raw_21 = math.fsum(record.value_order_21 for record in factor_records)
        raw_17 = math.fsum(record.value_order_17 for record in factor_records)
        reduction_17 = _complete_reduction_rounding(
            tuple(
                item.objective_sign * item.value_order_17
                for item in term_allowances
            )
        )
        for name, left, right, allowance in (
            (
                "partitioned order-21 objective",
                signed_21,
                terms.complete_elbo,
                complete_allowance.reduction_rounding,
            ),
            (
                "raw order-21 factor trace",
                raw_21,
                terms.complete_elbo,
                complete_allowance.reduction_rounding,
            ),
            ("raw order-17 factor trace", raw_17, signed_17, reduction_17),
        ):
            residual = abs(left - right)
            if not math.isfinite(residual) or residual > allowance:
                raise ValueError(f"{name} is inconsistent with the complete objective")
        result = object.__new__(cls)
        object.__setattr__(result, "terms", terms)
        object.__setattr__(result, "factor_records", tuple(factor_records))
        object.__setattr__(result, "term_allowances", tuple(term_allowances))
        object.__setattr__(
            result, "diagnostic_allowances", tuple(diagnostic_allowances)
        )
        object.__setattr__(result, "complete_allowance", complete_allowance)
        object.__setattr__(result, "objective_schema_sha256", H5_OBJECTIVE_SCHEMA_SHA256)
        object.__setattr__(
            result,
            "evaluated_state_sha256",
            h5_semantic_state_sha256(state.recognition, state.model),
        )
        object.__setattr__(result, "frozen_complement_sha256", complement)
        return result


class CompleteElboEvaluator(Protocol):
    def evaluate(
        self,
        state: H5LiveState | H5CandidateSnapshot,
        *,
        frozen_complement_sha256: str,
        cache: Mapping[FactorCacheKey, FactorCacheEntry] | None = None,
    ) -> CompleteElboEvaluation: ...


@dataclass(frozen=True)
class _Metric:
    value: float
    absolute_summands: tuple[float, ...]
    condition_numbers: tuple[float, ...]
    operation_count: int

    def __post_init__(self) -> None:
        _finite(self.value, "metric value")
        _finite_tuple(self.absolute_summands, "absolute_summands", minimum=0.0)
        _finite_tuple(self.condition_numbers, "condition_numbers", minimum=1.0)
        if type(self.operation_count) is not int or self.operation_count <= 0:
            raise ValueError("operation_count must be a positive integer")


@dataclass(frozen=True)
class _RecognitionValues:
    gaussians: Mapping[str, tuple[float, float]]
    categoricals: Mapping[str, tuple[float, ...]]
    categorical_cores: Mapping[str, object]


def evaluate_h5_complete_elbo(
    reference: H5ReferenceState,
    state: H5LiveState | H5CandidateSnapshot,
    *,
    frozen_complement_sha256: str,
    cache: Mapping[FactorCacheKey, FactorCacheEntry] | None = None,
) -> CompleteElboEvaluation:
    if not isinstance(reference, H5ReferenceState):
        raise ValueError("reference must be an H5ReferenceState")
    if not isinstance(state, (H5LiveState, H5CandidateSnapshot)):
        raise ValueError("state must be an H5 live state or candidate")
    complement = _require_sha256(
        frozen_complement_sha256, "frozen_complement_sha256"
    )
    if cache is not None and not isinstance(cache, Mapping):
        raise ValueError("cache must be a mapping or None")
    if (
        reference.objective_schema_sha256 != H5_OBJECTIVE_SCHEMA_SHA256
        or state.model.objective_schema_sha256 != H5_OBJECTIVE_SCHEMA_SHA256
    ):
        raise ValueError("H5 objective schema identity mismatch")
    raw_h1 = _decode_h1(reference)
    recognition = _recognition_values(state.recognition)
    normalized_factors, observations = _effective_factors(raw_h1, state.model)
    input_hashes = tuple(
        _factor_input_hash(
            factor_id,
            normalized_factors[factor_id],
            observations[factor_id],
            state.recognition,
        )
        for factor_id in H5_FACTOR_UNIVERSE
    )

    factor_records: list[FactorEvaluationRecord] = []
    for factor_id, input_hash in zip(H5_FACTOR_UNIVERSE, input_hashes, strict=True):
        key = FactorCacheKey(
            factor_id, input_hash, H5_QUADRATURE_ORDERS, complement
        )
        cached = (
            cache.get(key, _MISSING_CACHE_ENTRY)
            if cache is not None
            else _MISSING_CACHE_ENTRY
        )
        if cached is not _MISSING_CACHE_ENTRY:
            factor_records.append(_validated_reuse(key, cached))
            continue
        metric_21, metric_17 = _evaluate_factor_pair(
            factor_id,
            normalized_factors[factor_id],
            observations[factor_id],
            recognition,
        )
        factor_records.append(
            FactorEvaluationRecord(
                factor_id,
                input_hash,
                complement,
                metric_21.value,
                metric_17.value,
                metric_21.absolute_summands,
                metric_17.absolute_summands,
                metric_21.condition_numbers,
                metric_17.condition_numbers,
                metric_21.operation_count,
                metric_17.operation_count,
                CacheDisposition.REEVALUATED,
            )
        )

    terms_21 = _evaluate_term_metrics(
        raw_h1, normalized_factors, observations, recognition, order=21
    )
    terms_17 = _evaluate_term_metrics(
        raw_h1, normalized_factors, observations, recognition, order=17
    )
    signed_allowances = tuple(
        term_allowance(
            term_id,
            objective_sign=H5_SIGNED_TERM_SIGNS[index],  # type: ignore[arg-type]
            value_order_21=terms_21[term_id].value,
            value_order_17=terms_17[term_id].value,
            absolute_summands_order_21=terms_21[term_id].absolute_summands,
            absolute_summands_order_17=terms_17[term_id].absolute_summands,
            condition_numbers_order_21=terms_21[term_id].condition_numbers,
            condition_numbers_order_17=terms_17[term_id].condition_numbers,
            operation_count_order_21=terms_21[term_id].operation_count,
            operation_count_order_17=terms_17[term_id].operation_count,
        )
        for index, term_id in enumerate(H5_SIGNED_TERM_IDS)
    )
    entropy_21 = terms_21["joint_recognition_entropy"]
    entropy_17 = terms_17["joint_recognition_entropy"]
    diagnostic_allowances = (
        term_allowance(
            "joint_recognition_entropy",
            objective_sign=0,
            value_order_21=entropy_21.value,
            value_order_17=entropy_17.value,
            absolute_summands_order_21=entropy_21.absolute_summands,
            absolute_summands_order_17=entropy_17.absolute_summands,
            condition_numbers_order_21=entropy_21.condition_numbers,
            condition_numbers_order_17=entropy_17.condition_numbers,
            operation_count_order_21=entropy_21.operation_count,
            operation_count_order_17=entropy_17.operation_count,
        ),
    )
    signed_values_21 = tuple(item.signed_reported_value for item in signed_allowances)
    signed_values_17 = tuple(
        item.objective_sign * item.value_order_17 for item in signed_allowances
    )
    complete_21 = math.fsum(signed_values_21)
    complete_17 = math.fsum(signed_values_17)
    complete_allowance = complete_elbo_allowance(
        signed_allowances, signed_values_21
    )
    terms = _build_elbo_terms(
        terms_21,
        signed_allowances,
        diagnostic_allowances[0],
        complete_allowance,
        complete_21,
        complete_17,
    )
    return CompleteElboEvaluation.build(
        state=state,
        terms=terms,
        factor_records=tuple(factor_records),
        term_allowances=signed_allowances,
        diagnostic_allowances=diagnostic_allowances,
        complete_allowance=complete_allowance,
        frozen_complement_sha256=complement,
    )


def _validated_reuse(
    key: FactorCacheKey, entry: object
) -> FactorEvaluationRecord:
    if not isinstance(entry, FactorCacheEntry):
        raise StaleFactorCacheError(key.factor_id)
    record = entry.record
    try:
        rebuilt_input = FactorInputHashRecord(
            record.input_hash.factor_id,
            record.input_hash.input_schema_version,
            record.input_hash.canonical_input_bytes,
        )
    except (TypeError, ValueError) as exc:
        raise StaleFactorCacheError(key.factor_id) from exc
    if (
        entry.key != key
        or record.factor_id != key.factor_id
        or rebuilt_input != key.input_hash
        or record.input_hash != key.input_hash
        or record.frozen_complement_sha256 != key.frozen_complement_sha256
        or record.cache_disposition is not CacheDisposition.REEVALUATED
    ):
        raise StaleFactorCacheError(key.factor_id)
    try:
        return replace(record, cache_disposition=CacheDisposition.REUSED)
    except (TypeError, ValueError) as exc:
        raise StaleFactorCacheError(key.factor_id) from exc


def _complete_reduction_rounding(signed_terms: tuple[float, ...]) -> float:
    if type(signed_terms) is not tuple or len(signed_terms) != len(
        H5_SIGNED_TERM_IDS
    ):
        raise ValueError("signed_terms must contain the twelve H5 objective terms")
    checked = tuple(
        _finite(value, f"signed_terms[{index}]")
        for index, value in enumerate(signed_terms)
    )
    return float(
        H5_C
        * gamma_n(13)
        * max(1.0, math.fsum(abs(value) for value in checked))
    )


def _decode_h1(reference: H5ReferenceState) -> dict[str, object]:
    try:
        value = json.loads(reference.raw_h1_fixture_bytes)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("captured H1 fixture bytes are not valid JSON") from exc
    if type(value) is not dict or value.get("fixture_id") != "h1-v1":
        raise ValueError("captured H1 fixture identity is invalid")
    return value


def _recognition_values(snapshot: RecognitionSnapshot) -> _RecognitionValues:
    if not isinstance(snapshot, RecognitionSnapshot):
        raise ValueError("recognition must be a RecognitionSnapshot")
    gaussians = {
        coordinate.coordinate_id: (
            coordinate.mean.values[0],
            coordinate.variance.values[0],
        )
        for coordinate in snapshot.gaussians
    }
    categoricals = {
        coordinate.coordinate_id: tuple(coordinate.probabilities.values)
        for coordinate in snapshot.categoricals
    }
    categorical_cores = {
        coordinate.coordinate_id: _categorical_core(coordinate)
        for coordinate in snapshot.categoricals
    }
    if tuple(gaussians) != _GAUSSIAN_IDS or tuple(categoricals) != _CATEGORICAL_IDS:
        raise ValueError("recognition coordinates do not match the H5 universe")
    return _RecognitionValues(gaussians, categoricals, categorical_cores)


def _categorical_core(coordinate: CategoricalRecognitionCoordinate) -> object:
    return {
        "kind": "categorical",
        "support": coordinate.support,
        "conditioned_on": coordinate.conditioned_on,
        "probabilities": coordinate.probabilities.values,
    }


def _gaussian_core(coordinate: GaussianRecognitionCoordinate) -> object:
    return {
        "kind": "gaussian",
        "mean": coordinate.mean.values[0],
        "variance": coordinate.variance.values[0],
    }


def _model_blocks(model: H5ModelSnapshot) -> dict[str, dict[str, tuple[float, ...]]]:
    return {
        block.block_id: {name: tuple(value.values) for name, value in block.values}
        for block in model.parameter_blocks
    }


def _float_sequence(value: object, name: str) -> tuple[float, ...]:
    if type(value) is not list:
        raise ValueError(f"{name} must be a JSON array")
    checked: list[float] = []
    for index, item in enumerate(value):
        if type(item) not in (int, float) or not math.isfinite(float(item)):
            raise ValueError(f"{name}[{index}] must be finite numeric data")
        checked.append(float(item))
    return tuple(checked)


def _effective_factors(
    h1: Mapping[str, object], model: H5ModelSnapshot
) -> tuple[dict[str, object | None], dict[str, object | None]]:
    frames = _float_sequence(h1["frames"], "frames")
    model_priors = tuple(
        _float_sequence(row, f"model_source_priors[{index}]")
        for index, row in enumerate(h1["model_source_priors"])  # type: ignore[arg-type]
    )
    state_priors = tuple(
        _float_sequence(row, f"state_source_priors[{index}]")
        for index, row in enumerate(h1["state_source_priors"])  # type: ignore[arg-type]
    )
    model_offsets = _float_sequence(h1["model_offsets"], "model_offsets")
    model_variances = _float_sequence(h1["model_variances"], "model_variances")
    state_offsets = _float_sequence(h1["state_offsets"], "state_offsets")
    state_variances = _float_sequence(h1["state_variances"], "state_variances")
    state_model_slopes = _float_sequence(
        h1["state_model_slopes"], "state_model_slopes"
    )
    initial = h1["initial_joint"]
    if type(initial) is not dict:
        raise ValueError("initial_joint must be a JSON object")
    initial_mean = _float_sequence(initial["mean"], "initial_joint.mean")
    initial_covariance = tuple(
        _float_sequence(row, f"initial_joint.covariance[{index}]")
        for index, row in enumerate(initial["covariance"])  # type: ignore[arg-type]
    )
    decoder = h1["decoder"]
    if type(decoder) is not list or len(decoder) != 2:
        raise ValueError("decoder must contain two records")
    decoder_rows: list[dict[str, tuple[float, ...]]] = []
    for time, row in enumerate(decoder, start=1):
        if type(row) is not dict:
            raise ValueError(f"decoder[{time}] must be a JSON object")
        decoder_rows.append(
            {
                name: _float_sequence(row[name], f"decoder[{time}].{name}")
                for name in ("w_z", "w_m", "bias")
            }
        )
    blocks = _model_blocks(model)
    transition = blocks["theta[state_transition_2]"]
    emission_1 = blocks["theta[emission_1]"]
    shared = blocks["theta[shared_decoder_transition]"]["s"][0]
    shared_core = {
        "source": "theta[shared_decoder_transition].s",
        "value": shared,
    }
    emission_1_w_z = list(emission_1["w_z"])
    emission_1_w_z[0] += shared
    emission_2_w_z = list(decoder_rows[1]["w_z"])
    emission_2_w_z[0] += shared
    factors: dict[str, object | None] = {
        "initial_joint": {
            "kind": "joint_gaussian",
            "mean": initial_mean,
            "covariance": initial_covariance,
        },
        "model_source[1]": {
            "kind": "categorical_prior",
            "probabilities": model_priors[0],
        },
        "model_transition[1]": {
            "kind": "scalar_linear_gaussian",
            "sources": (0,),
            "alpha": (frames[1] / frames[0],),
            "c": model_offsets[0],
            "R": model_variances[0],
        },
        "state_source[1]": {
            "kind": "categorical_prior",
            "probabilities": state_priors[0],
        },
        "state_transition[1]": {
            "kind": "scalar_state_gaussian",
            "sources": (0,),
            "alpha": (frames[1] / frames[0],),
            "B": state_model_slopes[0],
            "c": state_offsets[0],
            "R": state_variances[0],
        },
        "emission[1]": {
            "kind": "categorical_emission",
            "w_z": tuple(emission_1_w_z),
            "w_m": emission_1["w_m"],
            "bias": emission_1["bias"],
            "shared_parameter": shared_core,
        },
        "model_source[2]": {
            "kind": "categorical_prior",
            "probabilities": model_priors[1],
        },
        "model_transition[2]": {
            "kind": "scalar_linear_gaussian",
            "sources": (0, 1),
            "alpha": (frames[2] / frames[0], frames[2] / frames[1]),
            "c": model_offsets[1],
            "R": model_variances[1],
        },
        "state_source[2]": {
            "kind": "categorical_prior",
            "probabilities": state_priors[1],
        },
        "state_transition[2]": {
            "kind": "scalar_state_gaussian",
            "sources": (0, 1),
            "alpha": (transition["alpha_0"][0], transition["alpha_1"][0]),
            "B": transition["B_base"][0] + shared,
            "c": transition["c"][0],
            "R": transition["R"][0],
            "shared_parameter": shared_core,
        },
        "emission[2]": {
            "kind": "categorical_emission",
            "w_z": tuple(emission_2_w_z),
            "w_m": decoder_rows[1]["w_m"],
            "bias": decoder_rows[1]["bias"],
            "shared_parameter": shared_core,
        },
        "recognition_entropy": None,
    }
    labels = h1["observation_labels"]
    if type(labels) is not list or len(labels) != 2 or any(type(item) is not int for item in labels):
        raise ValueError("observation_labels must contain two integers")
    observations: dict[str, object | None] = {factor_id: None for factor_id in H5_FACTOR_UNIVERSE}
    observations["emission[1]"] = {
        "time": 1,
        "label": labels[0],
        "selected_index": labels[0] - 1,
    }
    observations["emission[2]"] = {
        "time": 2,
        "label": labels[1],
        "selected_index": labels[1] - 1,
    }
    return factors, observations


def _recognition_input_core(
    snapshot: RecognitionSnapshot, factor_id: str
) -> tuple[tuple[str, object], ...]:
    gaussian = {
        coordinate.coordinate_id: _gaussian_core(coordinate)
        for coordinate in snapshot.gaussians
    }
    categorical = {
        coordinate.coordinate_id: _categorical_core(coordinate)
        for coordinate in snapshot.categoricals
    }
    all_coordinates = {**gaussian, **categorical}
    if factor_id == "recognition_entropy":
        identifiers = H5_RECOGNITION_COORDINATE_UNIVERSE
    else:
        bindings = dict(
            (record.factor_id, record.bindings)
            for record in snapshot_model_reconstruction_placeholder()
        )[factor_id]
        identifiers = tuple(binding for binding in bindings if binding.startswith("q["))
    return tuple((identifier, all_coordinates[identifier]) for identifier in identifiers)


def snapshot_model_reconstruction_placeholder() -> tuple[object, ...]:
    # The schema constants are authoritative; this tiny immutable adapter avoids
    # importing validation or storing a second mutable reconstruction table.
    from vfe4.types.h5_schema import H5_RECONSTRUCTION_ROWS

    @dataclass(frozen=True)
    class _Row:
        factor_id: str
        bindings: tuple[str, ...]

    return tuple(_Row(factor_id, bindings) for factor_id, bindings in H5_RECONSTRUCTION_ROWS)


def _factor_input_hash(
    factor_id: str,
    normalized_factor: object | None,
    observation: object | None,
    recognition: RecognitionSnapshot,
) -> FactorInputHashRecord:
    core = {
        "schema_version": H5_FACTOR_INPUT_SCHEMA_VERSION,
        "factor_id": factor_id,
        "normalized_factor": normalized_factor,
        "observation": observation,
        "recognition_inputs": _recognition_input_core(recognition, factor_id),
    }
    canonical = H5_FACTOR_INPUT_DOMAIN + _canonical_json_bytes(core)
    return FactorInputHashRecord(
        factor_id, H5_FACTOR_INPUT_SCHEMA_VERSION, canonical
    )


def _evaluate_factor_pair(
    factor_id: str,
    normalized_factor: object | None,
    observation: object | None,
    recognition: _RecognitionValues,
) -> tuple[_Metric, _Metric]:
    if factor_id in ("emission[1]", "emission[2]"):
        return (
            _emission_metric(factor_id, normalized_factor, observation, recognition, 21),
            _emission_metric(factor_id, normalized_factor, observation, recognition, 17),
        )
    metric = _analytic_factor_metric(
        factor_id, normalized_factor, recognition
    )
    return metric, metric


def _analytic_factor_metric(
    factor_id: str,
    normalized_factor: object | None,
    recognition: _RecognitionValues,
) -> _Metric:
    count = H5_ANALYTIC_FACTOR_OPERATION_COUNTS[factor_id]
    if factor_id == "initial_joint":
        assert isinstance(normalized_factor, dict)
        q_mean = (
            recognition.gaussians["q[z0]"][0],
            recognition.gaussians["q[m0]"][0],
        )
        q_variances = (
            recognition.gaussians["q[z0]"][1],
            recognition.gaussians["q[m0]"][1],
        )
        p_mean = normalized_factor["mean"]
        p_covariance = normalized_factor["covariance"]
        p_tensor = torch.tensor(p_covariance, dtype=torch.float64)
        q_covariance = torch.diag(torch.tensor(q_variances, dtype=torch.float64))
        displacement = torch.tensor(q_mean, dtype=torch.float64) - torch.tensor(
            p_mean, dtype=torch.float64
        )
        chol = torch.linalg.cholesky(p_tensor)
        trace = float(torch.trace(torch.cholesky_solve(q_covariance, chol)).item())
        quadratic = float(
            torch.dot(
                displacement,
                torch.cholesky_solve(displacement.unsqueeze(1), chol).squeeze(1),
            ).item()
        )
        log_determinant = float(
            (2.0 * torch.log(torch.diagonal(chol)).sum()).item()
        )
        summands = (
            -_LOG_2_PI,
            -0.5 * log_determinant,
            -0.5 * trace,
            -0.5 * quadratic,
        )
        condition = float(torch.linalg.cond(p_tensor).item())
        return _metric(math.fsum(summands), summands, (condition,), count)
    if factor_id.startswith("model_source"):
        assert isinstance(normalized_factor, dict)
        time = 1 if factor_id.endswith("[1]") else 2
        q = recognition.categoricals[f"q[model_source_b{time}]"]
        p = normalized_factor["probabilities"]
        summands = tuple(q_value * math.log(p_value) for q_value, p_value in zip(q, p, strict=True))
        return _metric(math.fsum(summands), summands, (1.0,), count)
    if factor_id.startswith("state_source"):
        assert isinstance(normalized_factor, dict)
        time = 1 if factor_id.endswith("[1]") else 2
        gamma = recognition.categoricals[f"q[model_source_b{time}]"]
        rows = (
            (recognition.categoricals["q[state_source_a1_b0]"],)
            if time == 1
            else (
                recognition.categoricals["q[source_row_a2]"],
                recognition.categoricals["q[state_source_a2_b1]"],
            )
        )
        if time == 1:
            rows = (recognition.categoricals["q[state_source_a1_b0]"],)
        p = normalized_factor["probabilities"]
        summands = tuple(
            gamma[b] * row[a] * math.log(p[a])
            for b, row in enumerate(rows)
            for a in range(len(row))
        )
        return _metric(math.fsum(summands), summands, (1.0,), count)
    if factor_id.startswith("model_transition"):
        assert isinstance(normalized_factor, dict)
        time = 1 if factor_id.endswith("[1]") else 2
        gamma = recognition.categoricals[f"q[model_source_b{time}]"]
        target = recognition.gaussians[f"q[m{time}]"]
        summands: list[float] = []
        for b, weight in enumerate(gamma):
            parent = recognition.gaussians[f"q[m{b}]"]
            value, parts = _expected_scalar_log_normal(
                target,
                ((normalized_factor["alpha"][b], parent),),
                normalized_factor["c"],
                normalized_factor["R"],
            )
            del value
            summands.extend(weight * part for part in parts)
        return _metric(math.fsum(summands), tuple(summands), (1.0,), count)
    if factor_id.startswith("state_transition"):
        assert isinstance(normalized_factor, dict)
        time = 1 if factor_id.endswith("[1]") else 2
        gamma = recognition.categoricals[f"q[model_source_b{time}]"]
        rows = (
            (recognition.categoricals["q[state_source_a1_b0]"],)
            if time == 1
            else (
                recognition.categoricals["q[source_row_a2]"],
                recognition.categoricals["q[state_source_a2_b1]"],
            )
        )
        if time == 1:
            rows = (recognition.categoricals["q[state_source_a1_b0]"],)
        target = recognition.gaussians[f"q[z{time}]"]
        model_value = recognition.gaussians[f"q[m{time}]"]
        summands = []
        for b, row in enumerate(rows):
            for a, conditional_weight in enumerate(row):
                parent = recognition.gaussians[f"q[z{a}]"]
                _, parts = _expected_scalar_log_normal(
                    target,
                    (
                        (normalized_factor["alpha"][a], parent),
                        (normalized_factor["B"], model_value),
                    ),
                    normalized_factor["c"],
                    normalized_factor["R"],
                )
                weight = gamma[b] * conditional_weight
                summands.extend(weight * part for part in parts)
        return _metric(math.fsum(summands), tuple(summands), (1.0,), count)
    if factor_id == "recognition_entropy":
        value, summands = _recognition_entropy(recognition)
        return _metric(value, summands, (1.0,), count)
    raise ValueError("unknown H5 factor")


def _metric(
    value: float,
    signed_summands: tuple[float, ...] | list[float],
    conditions: tuple[float, ...],
    count: int,
) -> _Metric:
    checked_value = float(value)
    absolute = tuple(abs(float(item)) for item in signed_summands)
    if not absolute:
        absolute = (0.0,)
    return _Metric(checked_value, absolute, tuple(float(v) for v in conditions), count)


def _expected_scalar_log_normal(
    target: tuple[float, float],
    parents: tuple[tuple[float, tuple[float, float]], ...],
    offset: float,
    variance: float,
) -> tuple[float, tuple[float, float, float]]:
    target_mean, target_variance = target
    residual_mean = target_mean - offset - math.fsum(
        coefficient * parent[0] for coefficient, parent in parents
    )
    residual_variance = target_variance + math.fsum(
        coefficient * coefficient * parent[1] for coefficient, parent in parents
    )
    expected_square = residual_variance + residual_mean * residual_mean
    parts = (
        -0.5 * _LOG_2_PI,
        -0.5 * math.log(variance),
        -0.5 * expected_square / variance,
    )
    return math.fsum(parts), parts


def _emission_metric(
    factor_id: str,
    normalized_factor: object | None,
    observation: object | None,
    recognition: _RecognitionValues,
    order: Literal[21, 17],
) -> _Metric:
    assert isinstance(normalized_factor, dict)
    assert isinstance(observation, dict)
    time = 1 if factor_id == "emission[1]" else 2
    z_mean, z_variance = recognition.gaussians[f"q[z{time}]"]
    m_mean, m_variance = recognition.gaussians[f"q[m{time}]"]
    mean = torch.tensor((z_mean, m_mean), dtype=torch.float64)
    covariance = torch.diag(torch.tensor((z_variance, m_variance), dtype=torch.float64))
    chol = torch.linalg.cholesky(covariance)
    nodes, weights = probabilists_gauss_hermite(order, dtype=torch.float64)
    w_z = torch.tensor(normalized_factor["w_z"], dtype=torch.float64)
    w_m = torch.tensor(normalized_factor["w_m"], dtype=torch.float64)
    bias = torch.tensor(normalized_factor["bias"], dtype=torch.float64)
    selected_index = observation["selected_index"]
    contributions: list[float] = []
    for first, second in product(range(order), repeat=2):
        standard = torch.stack((nodes[first], nodes[second]))
        value = mean + chol @ standard
        logits = w_z * value[0] + w_m * value[1] + bias
        selected = torch.log_softmax(logits, dim=0)[selected_index]
        contributions.append(
            float(weights[first].item())
            * float(weights[second].item())
            * float(selected.item())
        )
    condition = float(torch.linalg.cond(covariance).item())
    return _metric(
        math.fsum(contributions),
        tuple(contributions),
        (condition,),
        emission_operation_count(order),
    )


def _recognition_entropy(
    recognition: _RecognitionValues,
) -> tuple[float, tuple[float, ...]]:
    contributions = [
        0.5 * math.log(2.0 * math.pi * math.e * variance)
        for _, variance in recognition.gaussians.values()
    ]
    for time in (1, 2):
        gamma = recognition.categoricals[f"q[model_source_b{time}]"]
        contributions.extend(-value * math.log(value) for value in gamma)
        rows = (
            (recognition.categoricals["q[state_source_a1_b0]"],)
            if time == 1
            else (
                recognition.categoricals["q[source_row_a2]"],
                recognition.categoricals["q[state_source_a2_b1]"],
            )
        )
        for b, row in enumerate(rows):
            contributions.extend(-gamma[b] * value * math.log(value) for value in row)
    return math.fsum(contributions), tuple(contributions)


def _evaluate_term_metrics(
    h1: Mapping[str, object],
    normalized_factors: Mapping[str, object | None],
    observations: Mapping[str, object | None],
    recognition: _RecognitionValues,
    *,
    order: Literal[21, 17],
) -> dict[str, _Metric]:
    metrics: dict[str, _Metric] = {}
    metrics["expected_log_emission[1]"] = _emission_metric(
        "emission[1]",
        normalized_factors["emission[1]"],
        observations["emission[1]"],
        recognition,
        order,
    )
    metrics["expected_log_emission[2]"] = _emission_metric(
        "emission[2]",
        normalized_factors["emission[2]"],
        observations["emission[2]"],
        recognition,
        order,
    )
    initial = normalized_factors["initial_joint"]
    assert isinstance(initial, dict)
    p_mean = initial["mean"]
    p_covariance = initial["covariance"]
    q_m = recognition.gaussians["q[m0]"]
    metrics["initial_model_kl"] = _conditional_kl_metric(
        q_m[1],
        p_covariance[1][1],
        (q_m[0] - p_mean[1]) ** 2,
        H5_ANALYTIC_OPERATION_COUNTS["initial_model_kl"],
        (1.0,),
    )
    p_slope = p_covariance[0][1] / p_covariance[1][1]
    p_offset = p_mean[0] - p_slope * p_mean[1]
    p_variance = p_covariance[0][0] - (
        p_covariance[0][1] ** 2 / p_covariance[1][1]
    )
    q_z = recognition.gaussians["q[z0]"]
    slope_difference = -p_slope
    offset_difference = q_z[0] - p_offset
    mean_square = (
        slope_difference * slope_difference * q_m[1]
        + (slope_difference * q_m[0] + offset_difference) ** 2
    )
    metrics["initial_state_kl"] = _conditional_kl_metric(
        q_z[1],
        p_variance,
        mean_square,
        H5_ANALYTIC_OPERATION_COUNTS["initial_state_kl"],
        (1.0,),
    )
    for time in (1, 2):
        model_source_id = f"model_source_kl[{time}]"
        model_factor = normalized_factors[f"model_source[{time}]"]
        assert isinstance(model_factor, dict)
        gamma = recognition.categoricals[f"q[model_source_b{time}]"]
        model_contributions = tuple(
            q * (math.log(q) - math.log(p))
            for q, p in zip(gamma, model_factor["probabilities"], strict=True)
        )
        metrics[model_source_id] = _metric(
            math.fsum(model_contributions),
            model_contributions,
            (1.0,),
            H5_ANALYTIC_OPERATION_COUNTS[model_source_id],
        )
        state_source_id = f"state_source_kl[{time}]"
        state_factor = normalized_factors[f"state_source[{time}]"]
        assert isinstance(state_factor, dict)
        rows = (
            (recognition.categoricals["q[state_source_a1_b0]"],)
            if time == 1
            else (
                recognition.categoricals["q[source_row_a2]"],
                recognition.categoricals["q[state_source_a2_b1]"],
            )
        )
        state_contributions = tuple(
            gamma[b] * q * (math.log(q) - math.log(state_factor["probabilities"][a]))
            for b, row in enumerate(rows)
            for a, q in enumerate(row)
        )
        metrics[state_source_id] = _metric(
            math.fsum(state_contributions),
            state_contributions,
            (1.0,),
            H5_ANALYTIC_OPERATION_COUNTS[state_source_id],
        )
        model_transition_id = f"model_transition_kl[{time}]"
        model_transition = normalized_factors[f"model_transition[{time}]"]
        assert isinstance(model_transition, dict)
        target_m = recognition.gaussians[f"q[m{time}]"]
        model_kl_parts: list[float] = []
        model_kl_values: list[float] = []
        for b, weight in enumerate(gamma):
            parent_m = recognition.gaussians[f"q[m{b}]"]
            alpha = model_transition["alpha"][b]
            mean_square = alpha * alpha * parent_m[1] + (
                target_m[0]
                - alpha * parent_m[0]
                - model_transition["c"]
            ) ** 2
            value, parts = _conditional_kl_parts(
                target_m[1], model_transition["R"], mean_square
            )
            model_kl_values.append(weight * value)
            model_kl_parts.extend(weight * part for part in parts)
        metrics[model_transition_id] = _metric(
            math.fsum(model_kl_values),
            tuple(model_kl_parts),
            (1.0,),
            H5_ANALYTIC_OPERATION_COUNTS[model_transition_id],
        )
        state_transition_id = f"state_transition_kl[{time}]"
        state_transition = normalized_factors[f"state_transition[{time}]"]
        assert isinstance(state_transition, dict)
        target_z = recognition.gaussians[f"q[z{time}]"]
        model_value = recognition.gaussians[f"q[m{time}]"]
        state_kl_parts: list[float] = []
        state_kl_values: list[float] = []
        for b, row in enumerate(rows):
            for a, conditional_weight in enumerate(row):
                parent_z = recognition.gaussians[f"q[z{a}]"]
                alpha = state_transition["alpha"][a]
                B = state_transition["B"]
                mean_square = (
                    alpha * alpha * parent_z[1]
                    + B * B * model_value[1]
                    + (
                        target_z[0]
                        - alpha * parent_z[0]
                        - B * model_value[0]
                        - state_transition["c"]
                    )
                    ** 2
                )
                value, parts = _conditional_kl_parts(
                    target_z[1], state_transition["R"], mean_square
                )
                weight = gamma[b] * conditional_weight
                state_kl_values.append(weight * value)
                state_kl_parts.extend(weight * part for part in parts)
        metrics[state_transition_id] = _metric(
            math.fsum(state_kl_values),
            tuple(state_kl_parts),
            (1.0,),
            H5_ANALYTIC_OPERATION_COUNTS[state_transition_id],
        )
    entropy, entropy_parts = _recognition_entropy(recognition)
    metrics["joint_recognition_entropy"] = _metric(
        entropy,
        entropy_parts,
        (1.0,),
        H5_ANALYTIC_OPERATION_COUNTS["joint_recognition_entropy"],
    )
    if tuple(metrics) != (
        "expected_log_emission[1]",
        "expected_log_emission[2]",
        "initial_model_kl",
        "initial_state_kl",
        "model_source_kl[1]",
        "state_source_kl[1]",
        "model_transition_kl[1]",
        "state_transition_kl[1]",
        "model_source_kl[2]",
        "state_source_kl[2]",
        "model_transition_kl[2]",
        "state_transition_kl[2]",
        "joint_recognition_entropy",
    ):
        raise AssertionError("internal H5 term construction order changed")
    return metrics


def _conditional_kl_parts(
    q_variance: float, p_variance: float, mean_square: float
) -> tuple[float, tuple[float, float, float]]:
    parts = (
        0.5 * math.log(p_variance / q_variance),
        0.5 * (q_variance + mean_square) / p_variance,
        -0.5,
    )
    value = math.fsum(parts)
    if value < -64.0 * math.ulp(1.0):
        raise ValueError("conditional Gaussian KL must be nonnegative")
    return max(0.0, value), parts


def _conditional_kl_metric(
    q_variance: float,
    p_variance: float,
    mean_square: float,
    count: int,
    conditions: tuple[float, ...],
) -> _Metric:
    value, parts = _conditional_kl_parts(q_variance, p_variance, mean_square)
    return _metric(value, parts, conditions, count)


def _basic_allowance(value: H5TermAllowance) -> NumericalAllowance:
    rounding = math.fsum(
        (
            value.rounding_order_21,
            value.rounding_order_17,
            value.comparison_rounding,
        )
    )
    return NumericalAllowance(value.convergence_estimate, rounding)


def _build_elbo_terms(
    metrics: Mapping[str, _Metric],
    signed: tuple[H5TermAllowance, ...],
    diagnostic: H5TermAllowance,
    complete: H5CompleteAllowance,
    complete_21: float,
    complete_17: float,
) -> ElboTerms:
    by_id = {item.term_id: item for item in signed}
    complete_convergence = abs(complete_21 - complete_17)
    complete_rounding = max(0.0, complete.total - complete_convergence)
    allowances = ElboTermAllowances(
        expected_log_emission=(
            _basic_allowance(by_id["expected_log_emission[1]"]),
            _basic_allowance(by_id["expected_log_emission[2]"]),
        ),
        initial_model_kl=_basic_allowance(by_id["initial_model_kl"]),
        initial_state_kl=_basic_allowance(by_id["initial_state_kl"]),
        model_source_kl=(
            _basic_allowance(by_id["model_source_kl[1]"]),
            _basic_allowance(by_id["model_source_kl[2]"]),
        ),
        model_transition_kl=(
            _basic_allowance(by_id["model_transition_kl[1]"]),
            _basic_allowance(by_id["model_transition_kl[2]"]),
        ),
        state_source_kl=(
            _basic_allowance(by_id["state_source_kl[1]"]),
            _basic_allowance(by_id["state_source_kl[2]"]),
        ),
        state_transition_kl=(
            _basic_allowance(by_id["state_transition_kl[1]"]),
            _basic_allowance(by_id["state_transition_kl[2]"]),
        ),
        joint_recognition_entropy=_basic_allowance(diagnostic),
        complete_elbo=NumericalAllowance(
            float(complete_convergence), float(complete_rounding)
        ),
    )
    return ElboTerms(
        expected_log_emission=(
            metrics["expected_log_emission[1]"].value,
            metrics["expected_log_emission[2]"].value,
        ),
        initial_model_kl=metrics["initial_model_kl"].value,
        initial_state_kl=metrics["initial_state_kl"].value,
        model_source_kl=(
            metrics["model_source_kl[1]"].value,
            metrics["model_source_kl[2]"].value,
        ),
        model_transition_kl=(
            metrics["model_transition_kl[1]"].value,
            metrics["model_transition_kl[2]"].value,
        ),
        state_source_kl=(
            metrics["state_source_kl[1]"].value,
            metrics["state_source_kl[2]"].value,
        ),
        state_transition_kl=(
            metrics["state_transition_kl[1]"].value,
            metrics["state_transition_kl[2]"].value,
        ),
        joint_recognition_entropy=metrics["joint_recognition_entropy"].value,
        allowances=allowances,
        complete_elbo=float(complete_21),
    )


def _elbo_term_values(terms: ElboTerms) -> tuple[float, ...]:
    return (
        terms.expected_log_emission[0],
        terms.expected_log_emission[1],
        terms.initial_model_kl,
        terms.initial_state_kl,
        terms.model_source_kl[0],
        terms.model_source_kl[1],
        terms.model_transition_kl[0],
        terms.model_transition_kl[1],
        terms.state_source_kl[0],
        terms.state_source_kl[1],
        terms.state_transition_kl[0],
        terms.state_transition_kl[1],
    )


__all__ = [
    "CacheDisposition",
    "CompleteElboEvaluation",
    "CompleteElboEvaluator",
    "FactorCacheEntry",
    "FactorCacheKey",
    "FactorEvaluationRecord",
    "FactorInputHashRecord",
    "StaleFactorCacheError",
    "evaluate_h5_complete_elbo",
]
