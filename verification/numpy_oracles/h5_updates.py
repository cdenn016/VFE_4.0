"""NumPy-only byte oracle for the closed H5 exact updates."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import struct
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Self

import numpy as np


_SEMANTIC_DOMAIN = b"vfe4.h5.semantic-state.v1\x00"
_H1_RAW_SHA256 = "388e38cc8c16d8b5e2c61919c1e712a134d88fb0bbd8ec1f2939b9859c9a583b"
_H5_RAW_SHA256 = "9dd42603419952a2ffa4b6602971240ec00572283557d672ae6ee106c31dd91c"
_GAUSSIAN_IDS = ("q[z0]", "q[m0]", "q[z1]", "q[m1]", "q[z2]", "q[m2]")
_CATEGORICAL_IDS = (
    "q[model_source_b1]",
    "q[state_source_a1_b0]",
    "q[model_source_b2]",
    "q[source_row_a2]",
    "q[state_source_a2_b1]",
)
_BLOCK_IDS = (
    "theta[state_transition_2]",
    "theta[emission_1]",
    "theta[shared_decoder_transition]",
)
_BLOCK_FIELDS = {
    "theta[state_transition_2]": ("alpha_0", "alpha_1", "B_base", "c", "R"),
    "theta[emission_1]": ("w_z", "w_m", "bias"),
    "theta[shared_decoder_transition]": ("s",),
}
_OBJECTIVE_SCHEMA_SHA256 = (
    "b6af943b135b5acc01f9950502fb4a68554eab10005e44e43a7e7f213e4357ac"
)
_CATEGORICAL_METADATA = {
    "q[model_source_b1]": ((0,), ()),
    "q[state_source_a1_b0]": ((0,), (("b1", 0),)),
    "q[model_source_b2]": ((0, 1), ()),
    "q[source_row_a2]": ((0, 1), (("b2", 0),)),
    "q[state_source_a2_b1]": ((0, 1), (("b2", 1),)),
}
_RECONSTRUCTION_RECORDS = (
    ("initial_joint", ("h1.initial_joint", "q[z0]", "q[m0]")),
    ("model_source[1]", ("h1.model_source_priors[1]", "q[model_source_b1]")),
    (
        "model_transition[1]",
        ("h1.model_transition[1]", "q[m0]", "q[m1]", "q[model_source_b1]"),
    ),
    (
        "state_source[1]",
        ("h1.state_source_priors[1]", "q[model_source_b1]", "q[state_source_a1_b0]"),
    ),
    (
        "state_transition[1]",
        (
            "h1.state_transition[1]",
            "q[z0]",
            "q[z1]",
            "q[m1]",
            "q[model_source_b1]",
            "q[state_source_a1_b0]",
        ),
    ),
    (
        "emission[1]",
        (
            "theta[emission_1]",
            "theta[shared_decoder_transition]",
            "q[z1]",
            "q[m1]",
            "h1.observation_label[t=1]",
        ),
    ),
    ("model_source[2]", ("h1.model_source_priors[2]", "q[model_source_b2]")),
    (
        "model_transition[2]",
        (
            "h1.model_transition[2]",
            "q[m0]",
            "q[m1]",
            "q[m2]",
            "q[model_source_b2]",
        ),
    ),
    (
        "state_source[2]",
        (
            "h1.state_source_priors[2]",
            "q[model_source_b2]",
            "q[source_row_a2]",
            "q[state_source_a2_b1]",
        ),
    ),
    (
        "state_transition[2]",
        (
            "theta[state_transition_2]",
            "theta[shared_decoder_transition]",
            "q[z0]",
            "q[z1]",
            "q[z2]",
            "q[m2]",
            "q[model_source_b2]",
            "q[source_row_a2]",
            "q[state_source_a2_b1]",
        ),
    ),
    (
        "emission[2]",
        (
            "h1.emission[2]",
            "theta[shared_decoder_transition]",
            "q[z2]",
            "q[m2]",
            "h1.observation_label[t=2]",
        ),
    ),
    ("recognition_entropy", ("recognition_snapshot",)),
)
_SHARED_GROUPS = (
    (
        "shared_decoder_transition",
        "theta[shared_decoder_transition].s",
        (
            "state_transition[2].B:add",
            "emission[1].w_z[0]:add",
            "emission[2].w_z[0]:add",
        ),
    ),
)
_SIGNED_TERM_IDS = (
    "expected_log_emission[1]",
    "expected_log_emission[2]",
    "initial_model_kl",
    "initial_state_kl",
    "model_source_kl[1]",
    "model_source_kl[2]",
    "model_transition_kl[1]",
    "model_transition_kl[2]",
    "state_source_kl[1]",
    "state_source_kl[2]",
    "state_transition_kl[1]",
    "state_transition_kl[2]",
)
_SIGNED_TERM_SIGNS = (1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)
_ANALYTIC_OPERATION_COUNTS = MappingProxyType(
    {
        "initial_model_kl": 192,
        "initial_state_kl": 192,
        "model_source_kl[1]": 32,
        "model_source_kl[2]": 64,
        "model_transition_kl[1]": 192,
        "model_transition_kl[2]": 320,
        "state_source_kl[1]": 32,
        "state_source_kl[2]": 96,
        "state_transition_kl[1]": 256,
        "state_transition_kl[2]": 448,
        "joint_recognition_entropy": 320,
    }
)
_ANALYTIC_FACTOR_OPERATION_COUNTS = MappingProxyType(
    {
        "initial_joint": 256,
        "model_source[1]": 32,
        "model_transition[1]": 192,
        "state_source[1]": 32,
        "state_transition[1]": 256,
        "model_source[2]": 64,
        "model_transition[2]": 320,
        "state_source[2]": 96,
        "state_transition[2]": 448,
        "recognition_entropy": 320,
    }
)
_CANDIDATE_COMPARISON_OPERATION_COUNTS = MappingProxyType(
    {
        "exact_z0.mean": 512,
        "exact_z0.variance": 512,
        "exact_source_row_a2.probability[0]": 512,
        "exact_source_row_a2.probability[1]": 512,
        "exact_state_transition_2_m.alpha_0": 4096,
        "exact_state_transition_2_m.alpha_1": 4096,
        "exact_state_transition_2_m.B_base": 4096,
        "exact_state_transition_2_m.c": 4096,
        "exact_state_transition_2_m.R": 4096,
    }
)
_EPSILON = float(np.finfo(np.float64).eps)
_C = 4096.0
_LOG_2_PI = math.log(2.0 * math.pi)


def _tuple_tree(value: object) -> object:
    if type(value) is list:
        return tuple(_tuple_tree(item) for item in value)
    return value


def _finite(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite binary64 float")
    return value


def _nonnegative(value: object, name: str) -> float:
    checked = _finite(value, name)
    if checked < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return checked


def _finite_tuple(
    value: object,
    name: str,
    *,
    nonnegative: bool,
    minimum: float | None = None,
) -> tuple[float, ...]:
    if type(value) is not tuple or not value:
        raise ValueError(f"{name} must be a nonempty tuple")
    checked: list[float] = []
    for index, item in enumerate(value):
        number = _finite(item, f"{name}[{index}]")
        if nonnegative and number < 0.0:
            raise ValueError(f"{name}[{index}] must be nonnegative")
        if minimum is not None and number < minimum:
            raise ValueError(f"{name}[{index}] must be at least {minimum}")
        checked.append(number)
    return tuple(checked)


def _operation_count(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _gamma_n(count: int) -> float:
    checked = _operation_count(count, "operation count")
    return (checked * _EPSILON) / (1.0 - checked * _EPSILON)


def _emission_operation_count(order: int) -> int:
    if order not in (21, 17):
        raise ValueError("oracle quadrature order must be 21 or 17")
    return 32 * order * order + 8 * order + 32


def _expected_operation_counts(term_id: str) -> tuple[int, int]:
    if term_id in ("expected_log_emission[1]", "expected_log_emission[2]"):
        return _emission_operation_count(21), _emission_operation_count(17)
    try:
        count = _ANALYTIC_OPERATION_COUNTS[term_id]
    except KeyError as exc:
        raise ValueError("term_id is outside the closed oracle term universe") from exc
    return count, count


def _rounding_allowance(
    value: float,
    absolute_summands: tuple[float, ...],
    condition_numbers: tuple[float, ...],
    operation_count: int,
) -> float:
    return float(
        _C
        * _gamma_n(operation_count)
        * max(1.0, *condition_numbers)
        * max(1.0, abs(value), math.fsum(absolute_summands))
    )


def _comparison_rounding(value_order_21: float, value_order_17: float) -> float:
    return float(
        _C
        * _gamma_n(3)
        * max(
            1.0,
            abs(value_order_21),
            abs(value_order_17),
            abs(value_order_21) + abs(value_order_17),
        )
    )


def _canonicalize(value: object) -> object:
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("canonical oracle floats must be finite")
        return value.hex()
    if type(value) in (str, int, bool) or value is None:
        return value
    if type(value) is tuple or type(value) is list:
        return [_canonicalize(item) for item in value]
    if type(value) is dict:
        if not all(type(key) is str and key for key in value):
            raise ValueError("canonical oracle keys must be nonempty strings")
        return {key: _canonicalize(value[key]) for key in sorted(value)}
    raise ValueError(f"unsupported oracle value {type(value).__name__}")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        _canonicalize(value),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def _decode_float_hex(value: object) -> object:
    if type(value) is str and value.startswith(("0x", "-0x")):
        try:
            return float.fromhex(value)
        except ValueError as exc:
            raise ValueError("invalid canonical binary64 hex") from exc
    if type(value) is list:
        return [_decode_float_hex(item) for item in value]
    if type(value) is dict:
        return {key: _decode_float_hex(item) for key, item in value.items()}
    return value


def _parse_json_bytes(data: bytes, name: str) -> dict[str, object]:
    if type(data) is not bytes:
        raise ValueError(f"{name} must be bytes")
    try:
        raw = json.loads(data)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} must be valid JSON") from exc
    if type(raw) is not dict:
        raise ValueError(f"{name} must decode to an object")
    decoded = _decode_float_hex(raw)
    assert isinstance(decoded, dict)
    if _canonical_json(decoded) != data:
        raise ValueError(f"{name} must use the exact canonical JSON encoding")
    return decoded


def _validate_tensor(value: object, shape: tuple[int, ...], name: str) -> None:
    if type(value) is not dict or set(value) != {"dtype", "shape", "values"}:
        raise ValueError(f"{name} is not a frozen tensor")
    if value["dtype"] != "float64" or tuple(value["shape"]) != shape:
        raise ValueError(f"{name} tensor metadata changed")
    values = value["values"]
    if type(values) is not list or len(values) != (math.prod(shape) if shape else 1):
        raise ValueError(f"{name} tensor values have the wrong length")
    if any(type(item) is not float or not math.isfinite(item) for item in values):
        raise ValueError(f"{name} tensor values must be finite binary64")


def _validate_recognition(value: object) -> dict[str, object]:
    if type(value) is not dict or set(value) != {"schema_version", "gaussians", "categoricals"}:
        raise ValueError("recognition JSON has the wrong schema")
    if value["schema_version"] != "h5-recognition-snapshot-v1":
        raise ValueError("recognition schema version changed")
    gaussians = value["gaussians"]
    categoricals = value["categoricals"]
    if (
        type(gaussians) is not list
        or not all(type(item) is list and len(item) == 3 for item in gaussians)
        or tuple(item[0] for item in gaussians) != _GAUSSIAN_IDS
    ):
        raise ValueError("Gaussian universe/order changed")
    if (
        type(categoricals) is not list
        or not all(type(item) is list and len(item) == 4 for item in categoricals)
        or tuple(item[0] for item in categoricals) != _CATEGORICAL_IDS
    ):
        raise ValueError("categorical universe/order changed")
    for item in gaussians:
        _validate_tensor(item[1], (), f"{item[0]}.mean")
        _validate_tensor(item[2], (), f"{item[0]}.variance")
        if item[2]["values"][0] <= 0.0:
            raise ValueError("Gaussian variance must be positive")
    for item in categoricals:
        support = item[1]
        expected_support, expected_condition = _CATEGORICAL_METADATA[item[0]]
        if _tuple_tree(support) != expected_support:
            raise ValueError(f"{item[0]} support changed")
        if _tuple_tree(item[2]) != expected_condition:
            raise ValueError(f"{item[0]} condition changed")
        _validate_tensor(item[3], (len(support),), f"{item[0]}.probabilities")
        probabilities = item[3]["values"]
        if any(p <= 0.0 for p in probabilities) or abs(math.fsum(probabilities) - 1.0) > 1e-13:
            raise ValueError("categorical probabilities are invalid")
    return value


def _validate_model(value: object) -> dict[str, object]:
    if type(value) is not dict or set(value) != {
        "schema_version",
        "objective_schema_sha256",
        "parameter_blocks",
        "reconstruction_records",
        "shared_groups",
    }:
        raise ValueError("model JSON has the wrong schema")
    if value["schema_version"] != "h5-model-snapshot-v1":
        raise ValueError("model schema version changed")
    if value["objective_schema_sha256"] != _OBJECTIVE_SCHEMA_SHA256:
        raise ValueError("model objective schema identity changed")
    if _tuple_tree(value["reconstruction_records"]) != _RECONSTRUCTION_RECORDS:
        raise ValueError("model reconstruction records changed")
    if _tuple_tree(value["shared_groups"]) != _SHARED_GROUPS:
        raise ValueError("model shared-parameter groups changed")
    blocks = value["parameter_blocks"]
    if (
        type(blocks) is not list
        or not all(type(item) is list and len(item) == 2 for item in blocks)
        or tuple(item[0] for item in blocks) != _BLOCK_IDS
    ):
        raise ValueError("model block universe/order changed")
    for block_id, rows in blocks:
        if (
            type(rows) is not list
            or not all(type(item) is list and len(item) == 2 for item in rows)
            or tuple(item[0] for item in rows) != _BLOCK_FIELDS[block_id]
        ):
            raise ValueError("model block field order changed")
        for name, tensor in rows:
            shape = (3,) if name in ("w_z", "w_m", "bias") else ()
            _validate_tensor(tensor, shape, f"{block_id}.{name}")
        if block_id == "theta[state_transition_2]" and rows[-1][1]["values"][0] <= 0.0:
            raise ValueError("state-transition R must be positive")
    return value


def _semantic_bytes(recognition: dict[str, object], model: dict[str, object]) -> bytes:
    recognition_json = _canonical_json(recognition)
    model_json = _canonical_json(model)
    return (
        _SEMANTIC_DOMAIN
        + struct.pack(">Q", len(recognition_json))
        + recognition_json
        + struct.pack(">Q", len(model_json))
        + model_json
    )


def _parse_semantic_state(data: bytes) -> tuple[dict[str, object], dict[str, object]]:
    if type(data) is not bytes or not data.startswith(_SEMANTIC_DOMAIN):
        raise ValueError("live state bytes must use the H5 semantic-state domain")
    cursor = len(_SEMANTIC_DOMAIN)
    if len(data) < cursor + 8:
        raise ValueError("semantic state is truncated")
    recognition_length = struct.unpack(">Q", data[cursor : cursor + 8])[0]
    cursor += 8
    recognition_json = data[cursor : cursor + recognition_length]
    cursor += recognition_length
    if len(data) < cursor + 8:
        raise ValueError("semantic state is truncated before model length")
    model_length = struct.unpack(">Q", data[cursor : cursor + 8])[0]
    cursor += 8
    model_json = data[cursor : cursor + model_length]
    cursor += model_length
    if cursor != len(data):
        raise ValueError("semantic state has trailing or truncated bytes")
    recognition = _validate_recognition(
        _parse_json_bytes(recognition_json, "recognition JSON")
    )
    model = _validate_model(_parse_json_bytes(model_json, "model JSON"))
    if _semantic_bytes(recognition, model) != data:
        raise ValueError("semantic state failed exact re-encoding")
    return recognition, model


@dataclass(frozen=True)
class H5OracleTermEvidence:
    schema_version: Literal["h5-oracle-term-evidence-v1"]
    term_id: str
    objective_sign: Literal[-1, 1]
    value_order_21: float
    value_order_17: float
    signed_reported_value: float
    absolute_summands_order_21: tuple[float, ...]
    absolute_summands_order_17: tuple[float, ...]
    condition_numbers_order_21: tuple[float, ...]
    condition_numbers_order_17: tuple[float, ...]
    operation_count_order_21: int
    operation_count_order_17: int
    convergence_estimate: float
    rounding_order_21: float
    rounding_order_17: float
    comparison_rounding: float
    total: float

    def __post_init__(self) -> None:
        if self.schema_version != "h5-oracle-term-evidence-v1":
            raise ValueError("unsupported H5 oracle term-evidence schema")
        if self.term_id not in _SIGNED_TERM_IDS:
            raise ValueError("term_id is outside the closed oracle term universe")
        expected_sign = _SIGNED_TERM_SIGNS[_SIGNED_TERM_IDS.index(self.term_id)]
        if type(self.objective_sign) is not int or self.objective_sign != expected_sign:
            raise ValueError("objective_sign does not match the signed term order")
        value_21 = _finite(self.value_order_21, "value_order_21")
        value_17 = _finite(self.value_order_17, "value_order_17")
        if self.term_id not in (
            "expected_log_emission[1]",
            "expected_log_emission[2]",
        ) and value_21.hex() != value_17.hex():
            raise ValueError("analytic oracle terms must be order-identical")
        expected_signed = float(self.objective_sign * value_21)
        if _finite(self.signed_reported_value, "signed_reported_value").hex() != expected_signed.hex():
            raise ValueError("signed_reported_value must be recomputed exactly")
        summands_21 = _finite_tuple(
            self.absolute_summands_order_21,
            "absolute_summands_order_21",
            nonnegative=True,
        )
        summands_17 = _finite_tuple(
            self.absolute_summands_order_17,
            "absolute_summands_order_17",
            nonnegative=True,
        )
        conditions_21 = _finite_tuple(
            self.condition_numbers_order_21,
            "condition_numbers_order_21",
            nonnegative=True,
            minimum=1.0,
        )
        conditions_17 = _finite_tuple(
            self.condition_numbers_order_17,
            "condition_numbers_order_17",
            nonnegative=True,
            minimum=1.0,
        )
        count_21 = _operation_count(
            self.operation_count_order_21, "operation_count_order_21"
        )
        count_17 = _operation_count(
            self.operation_count_order_17, "operation_count_order_17"
        )
        if (count_21, count_17) != _expected_operation_counts(self.term_id):
            raise ValueError("operation counts do not match the frozen oracle table")
        convergence = abs(value_21 - value_17)
        rounding_21 = _rounding_allowance(
            value_21, summands_21, conditions_21, count_21
        )
        rounding_17 = _rounding_allowance(
            value_17, summands_17, conditions_17, count_17
        )
        comparison = _comparison_rounding(value_21, value_17)
        total = math.fsum((convergence, rounding_21, rounding_17, comparison))
        for name, actual, expected in (
            ("convergence_estimate", self.convergence_estimate, convergence),
            ("rounding_order_21", self.rounding_order_21, rounding_21),
            ("rounding_order_17", self.rounding_order_17, rounding_17),
            ("comparison_rounding", self.comparison_rounding, comparison),
            ("total", self.total, total),
        ):
            if _nonnegative(actual, name).hex() != float(expected).hex():
                raise ValueError(f"{name} must be recomputed exactly")
        object.__setattr__(self, "absolute_summands_order_21", summands_21)
        object.__setattr__(self, "absolute_summands_order_17", summands_17)
        object.__setattr__(self, "condition_numbers_order_21", conditions_21)
        object.__setattr__(self, "condition_numbers_order_17", conditions_17)


def _copy_oracle_term(value: object) -> H5OracleTermEvidence:
    if not isinstance(value, H5OracleTermEvidence):
        raise ValueError("complete term traces require oracle term evidence")
    return H5OracleTermEvidence(
        value.schema_version,
        value.term_id,
        value.objective_sign,
        value.value_order_21,
        value.value_order_17,
        value.signed_reported_value,
        value.absolute_summands_order_21,
        value.absolute_summands_order_17,
        value.condition_numbers_order_21,
        value.condition_numbers_order_17,
        value.operation_count_order_21,
        value.operation_count_order_17,
        value.convergence_estimate,
        value.rounding_order_21,
        value.rounding_order_17,
        value.comparison_rounding,
        value.total,
    )


def _complete_operand_aggregates(
    trace: tuple[H5OracleTermEvidence, ...],
) -> tuple[float, int, tuple[float, ...], tuple[float, ...], float, float, float]:
    if (
        type(trace) is not tuple
        or tuple(item.term_id for item in trace) != _SIGNED_TERM_IDS
        or tuple(item.objective_sign for item in trace) != _SIGNED_TERM_SIGNS
    ):
        raise ValueError("complete term trace must use the exact signed term order")
    signed_values = tuple(item.signed_reported_value for item in trace)
    value = float(math.fsum(signed_values))
    operation_count = 13 + sum(
        item.operation_count_order_21 + item.operation_count_order_17
        for item in trace
    )
    conditions = tuple(
        number
        for item in trace
        for values in (
            item.condition_numbers_order_21,
            item.condition_numbers_order_17,
        )
        for number in values
    )
    summands = tuple(
        number
        for item in trace
        for values in (
            item.absolute_summands_order_21,
            item.absolute_summands_order_17,
        )
        for number in values
    ) + tuple(abs(number) for number in signed_values)
    convergence = float(math.fsum(item.convergence_estimate for item in trace))
    reduction = float(
        _C * _gamma_n(13) * max(1.0, math.fsum(abs(value) for value in signed_values))
    )
    rounding = float(
        math.fsum(
            tuple(
                number
                for item in trace
                for number in (
                    item.rounding_order_21,
                    item.rounding_order_17,
                    item.comparison_rounding,
                )
            )
            + (reduction,)
        )
    )
    allowance = float(math.fsum((convergence, rounding)))
    return (
        value,
        operation_count,
        conditions,
        summands,
        convergence,
        rounding,
        allowance,
    )


@dataclass(frozen=True, init=False)
class H5OracleOperandEvidence:
    schema_version: Literal["h5-oracle-operand-evidence-v1"]
    operand: Literal["before", "after", "delta"]
    complete_term_trace: tuple[H5OracleTermEvidence, ...]
    value: float
    operation_count: int
    condition_numbers: tuple[float, ...]
    absolute_summands: tuple[float, ...]
    convergence: float
    rounding: float
    allowance: float

    @classmethod
    def _new(
        cls,
        *,
        operand: Literal["before", "after", "delta"],
        complete_term_trace: tuple[H5OracleTermEvidence, ...],
        value: float,
        operation_count: int,
        condition_numbers: tuple[float, ...],
        absolute_summands: tuple[float, ...],
        convergence: float,
        rounding: float,
        allowance: float,
    ) -> Self:
        result = object.__new__(cls)
        object.__setattr__(result, "schema_version", "h5-oracle-operand-evidence-v1")
        object.__setattr__(result, "operand", operand)
        object.__setattr__(result, "complete_term_trace", complete_term_trace)
        object.__setattr__(result, "value", value)
        object.__setattr__(result, "operation_count", operation_count)
        object.__setattr__(result, "condition_numbers", condition_numbers)
        object.__setattr__(result, "absolute_summands", absolute_summands)
        object.__setattr__(result, "convergence", convergence)
        object.__setattr__(result, "rounding", rounding)
        object.__setattr__(result, "allowance", allowance)
        result.__post_init__()
        return result

    def __post_init__(self) -> None:
        if self.schema_version != "h5-oracle-operand-evidence-v1":
            raise ValueError("unsupported H5 oracle operand-evidence schema")
        if self.operand in ("before", "after"):
            expected = _complete_operand_aggregates(self.complete_term_trace)
            actual = (
                self.value,
                self.operation_count,
                self.condition_numbers,
                self.absolute_summands,
                self.convergence,
                self.rounding,
                self.allowance,
            )
            if actual != expected:
                raise ValueError("complete operand aggregates must be recomputed exactly")
            return
        if self.operand != "delta" or self.complete_term_trace != ():
            raise ValueError("delta operands require an empty complete-term trace")
        _finite(self.value, "delta value")
        if self.operation_count != 3:
            raise ValueError("delta operation_count must equal three")
        if self.condition_numbers != (1.0,):
            raise ValueError("delta condition_numbers must equal (1.0,)")
        summands = _finite_tuple(
            self.absolute_summands,
            "delta absolute_summands",
            nonnegative=True,
        )
        if len(summands) != 2:
            raise ValueError("delta absolute_summands must contain before and after")
        _nonnegative(self.convergence, "delta convergence")
        _nonnegative(self.rounding, "delta rounding")
        _nonnegative(self.allowance, "delta allowance")

    @classmethod
    def from_complete_terms(
        cls,
        *,
        operand: Literal["before", "after"],
        complete_term_trace: tuple[H5OracleTermEvidence, ...],
    ) -> Self:
        if operand not in ("before", "after"):
            raise ValueError("complete operand role must be before or after")
        if type(complete_term_trace) is not tuple:
            raise ValueError("complete_term_trace must be a tuple")
        copied = tuple(_copy_oracle_term(item) for item in complete_term_trace)
        aggregates = _complete_operand_aggregates(copied)
        return cls._new(
            operand=operand,
            complete_term_trace=copied,
            value=aggregates[0],
            operation_count=aggregates[1],
            condition_numbers=aggregates[2],
            absolute_summands=aggregates[3],
            convergence=aggregates[4],
            rounding=aggregates[5],
            allowance=aggregates[6],
        )

    @classmethod
    def from_delta(cls, *, before: Self, after: Self) -> Self:
        if not isinstance(before, cls) or not isinstance(after, cls):
            raise ValueError("delta operands require typed before and after evidence")
        rebuilt_before = cls.from_complete_terms(
            operand="before", complete_term_trace=before.complete_term_trace
        )
        rebuilt_after = cls.from_complete_terms(
            operand="after", complete_term_trace=after.complete_term_trace
        )
        if before != rebuilt_before or after != rebuilt_after:
            raise ValueError("delta source operands failed aggregate revalidation")
        value = float(after.value - before.value)
        subtraction = float(
            _C
            * _gamma_n(3)
            * max(
                1.0,
                abs(before.value),
                abs(after.value),
                abs(value),
                abs(before.value) + abs(after.value),
            )
        )
        convergence = float(math.fsum((before.convergence, after.convergence)))
        rounding = subtraction
        allowance = float(
            math.fsum((before.allowance, after.allowance, subtraction))
        )
        return cls._new(
            operand="delta",
            complete_term_trace=(),
            value=value,
            operation_count=3,
            condition_numbers=(1.0,),
            absolute_summands=(abs(before.value), abs(after.value)),
            convergence=convergence,
            rounding=rounding,
            allowance=allowance,
        )


@dataclass(frozen=True)
class H5OracleUpdate:
    schema_version: Literal["h5-oracle-update-v1"]
    rule: str
    candidate_recognition_json: bytes
    candidate_model_json: bytes
    candidate_condition_numbers: tuple[tuple[str, float], ...]
    semantic_state_sha256: str = field(init=False)
    before: H5OracleOperandEvidence
    after: H5OracleOperandEvidence
    delta: H5OracleOperandEvidence

    def __post_init__(self) -> None:
        if self.schema_version != "h5-oracle-update-v1":
            raise ValueError("unsupported H5 oracle schema")
        if self.rule not in {
            "exact_z0",
            "exact_source_row_a2",
            "exact_state_transition_2_m",
            "generalized_em_emission_1",
            "natural_gradient_z1",
        }:
            raise ValueError("oracle rule is outside H5")
        recognition = _validate_recognition(
            _parse_json_bytes(self.candidate_recognition_json, "candidate recognition")
        )
        model = _validate_model(
            _parse_json_bytes(self.candidate_model_json, "candidate model")
        )
        if type(self.candidate_condition_numbers) is not tuple:
            raise ValueError("candidate_condition_numbers must be a tuple")
        conditions: list[tuple[str, float]] = []
        for item in self.candidate_condition_numbers:
            if type(item) is not tuple or len(item) != 2 or type(item[0]) is not str:
                raise ValueError("candidate condition record is malformed")
            value = _finite(item[1], "candidate condition")
            if value < 1.0:
                raise ValueError("candidate condition must be at least one")
            conditions.append((item[0], value))
        if self.rule == "exact_state_transition_2_m":
            if tuple(name for name, _ in conditions) != ("G_condition_number",):
                raise ValueError("exact M oracle requires the G condition number")
        elif conditions:
            raise ValueError("only exact M may carry an oracle condition number")
        if not isinstance(self.before, H5OracleOperandEvidence) or not isinstance(
            self.after, H5OracleOperandEvidence
        ):
            raise ValueError("oracle update requires before and after operand evidence")
        rebuilt_before = H5OracleOperandEvidence.from_complete_terms(
            operand="before", complete_term_trace=self.before.complete_term_trace
        )
        rebuilt_after = H5OracleOperandEvidence.from_complete_terms(
            operand="after", complete_term_trace=self.after.complete_term_trace
        )
        if self.before != rebuilt_before or self.after != rebuilt_after:
            raise ValueError("oracle complete operands failed aggregate revalidation")
        rebuilt_delta = H5OracleOperandEvidence.from_delta(
            before=rebuilt_before, after=rebuilt_after
        )
        if self.delta != rebuilt_delta:
            raise ValueError("oracle delta must be recomputed from before and after")
        object.__setattr__(self, "candidate_recognition_json", _canonical_json(recognition))
        object.__setattr__(self, "candidate_model_json", _canonical_json(model))
        object.__setattr__(self, "candidate_condition_numbers", tuple(conditions))
        object.__setattr__(self, "before", rebuilt_before)
        object.__setattr__(self, "after", rebuilt_after)
        object.__setattr__(self, "delta", rebuilt_delta)
        object.__setattr__(
            self,
            "semantic_state_sha256",
            hashlib.sha256(_semantic_bytes(recognition, model)).hexdigest(),
        )


def _parse_fixtures(h1_bytes: bytes, update_bytes: bytes) -> tuple[dict[str, object], dict[str, object]]:
    if type(h1_bytes) is not bytes or type(update_bytes) is not bytes:
        raise ValueError("oracle fixtures must be bytes")
    h1_sha256 = hashlib.sha256(h1_bytes).hexdigest()
    update_sha256 = hashlib.sha256(update_bytes).hexdigest()
    if h1_sha256 != _H1_RAW_SHA256:
        raise ValueError(
            f"H1 raw SHA-256 mismatch: expected {_H1_RAW_SHA256}, got {h1_sha256}"
        )
    if update_sha256 != _H5_RAW_SHA256:
        raise ValueError(
            f"H5 raw SHA-256 mismatch: expected {_H5_RAW_SHA256}, got {update_sha256}"
        )
    try:
        h1 = json.loads(h1_bytes)
        update = json.loads(update_bytes)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("oracle fixture bytes must be valid JSON") from exc
    if type(h1) is not dict or h1.get("fixture_id") != "h1-v1":
        raise ValueError("H1 fixture identity changed")
    if type(update) is not dict or update.get("fixture_id") != "h5-conditional-update-v1":
        raise ValueError("H5 update fixture identity changed")
    if update.get("h1_fixture_sha256") != h1_sha256:
        raise ValueError("H5 update fixture does not bind the H1 bytes")
    return h1, update


def _gaussians(recognition: dict[str, object]) -> dict[str, tuple[float, float]]:
    return {
        item[0]: (item[1]["values"][0], item[2]["values"][0])
        for item in recognition["gaussians"]
    }


def _categoricals(recognition: dict[str, object]) -> dict[str, tuple[float, ...]]:
    return {
        item[0]: tuple(item[3]["values"])
        for item in recognition["categoricals"]
    }


def _blocks(model: dict[str, object]) -> dict[str, dict[str, np.ndarray]]:
    return {
        item[0]: {
            name: np.asarray(tensor["values"], dtype=np.float64).reshape(
                tuple(tensor["shape"])
            )
            for name, tensor in item[1]
        }
        for item in model["parameter_blocks"]
    }


def _set_gaussian(recognition: dict[str, object], coordinate_id: str, mean: float, variance: float) -> None:
    item = next(row for row in recognition["gaussians"] if row[0] == coordinate_id)
    item[1]["values"] = [float(mean)]
    item[2]["values"] = [float(variance)]


def _set_categorical(recognition: dict[str, object], coordinate_id: str, values: tuple[float, ...]) -> None:
    item = next(row for row in recognition["categoricals"] if row[0] == coordinate_id)
    item[3]["values"] = [float(value) for value in values]


def _set_block(model: dict[str, object], block_id: str, values: dict[str, float]) -> None:
    block = next(row for row in model["parameter_blocks"] if row[0] == block_id)
    for name, tensor in block[1]:
        if name in values:
            tensor["values"] = [float(values[name])]


def _expected_log_normal(
    target: tuple[float, float],
    parents: tuple[tuple[float, tuple[float, float]], ...],
    offset: float,
    variance: float,
) -> float:
    residual = target[0] - offset - math.fsum(
        coefficient * parent[0] for coefficient, parent in parents
    )
    residual_variance = target[1] + math.fsum(
        coefficient * coefficient * parent[1] for coefficient, parent in parents
    )
    return -0.5 * (
        _LOG_2_PI
        + math.log(variance)
        + (residual_variance + residual * residual) / variance
    )


@dataclass(frozen=True)
class _OracleMetric:
    value: float
    absolute_summands: tuple[float, ...]
    condition_numbers: tuple[float, ...]
    operation_count: int

    def __post_init__(self) -> None:
        _finite(self.value, "metric value")
        _finite_tuple(
            self.absolute_summands, "metric absolute_summands", nonnegative=True
        )
        _finite_tuple(
            self.condition_numbers,
            "metric condition_numbers",
            nonnegative=True,
            minimum=1.0,
        )
        _operation_count(self.operation_count, "metric operation_count")


def _metric(
    value: float,
    signed_summands: tuple[float, ...] | list[float],
    condition_numbers: tuple[float, ...],
    operation_count: int,
) -> _OracleMetric:
    absolute = tuple(abs(float(item)) for item in signed_summands) or (0.0,)
    return _OracleMetric(
        float(value),
        absolute,
        tuple(float(item) for item in condition_numbers),
        operation_count,
    )


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
        raise ValueError("oracle conditional Gaussian KL must be nonnegative")
    return max(0.0, value), parts


def _conditional_kl_metric(
    q_variance: float,
    p_variance: float,
    mean_square: float,
    operation_count: int,
) -> _OracleMetric:
    value, parts = _conditional_kl_parts(q_variance, p_variance, mean_square)
    return _metric(value, parts, (1.0,), operation_count)


def _emission_metric(
    h1: dict[str, object],
    qg: dict[str, tuple[float, float]],
    blocks: dict[str, dict[str, np.ndarray]],
    *,
    time: Literal[1, 2],
    order: Literal[21, 17],
) -> _OracleMetric:
    shared = float(blocks["theta[shared_decoder_transition]"]["s"])
    if time == 1:
        emission = blocks["theta[emission_1]"]
        w_z = np.array(emission["w_z"], dtype=np.float64, copy=True)
        w_m = np.asarray(emission["w_m"], dtype=np.float64)
        bias = np.asarray(emission["bias"], dtype=np.float64)
    else:
        decoder = h1["decoder"][1]
        w_z = np.asarray(decoder["w_z"], dtype=np.float64).copy()
        w_m = np.asarray(decoder["w_m"], dtype=np.float64)
        bias = np.asarray(decoder["bias"], dtype=np.float64)
    w_z[0] += shared
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    nodes = math.sqrt(2.0) * nodes
    weights = weights / math.sqrt(math.pi)
    z_mean, z_variance = qg[f"q[z{time}]"]
    m_mean, m_variance = qg[f"q[m{time}]"]
    z_points = z_mean + math.sqrt(z_variance) * nodes
    m_points = m_mean + math.sqrt(m_variance) * nodes
    selected_index = h1["observation_labels"][time - 1] - 1
    contributions: list[float] = []
    for first in range(order):
        for second in range(order):
            logits = (
                z_points[first] * w_z + m_points[second] * w_m + bias
            )
            maximum = float(np.max(logits))
            log_normalizer = maximum + math.log(
                math.fsum(math.exp(float(item) - maximum) for item in logits)
            )
            selected = float(logits[selected_index]) - log_normalizer
            contributions.append(
                float(weights[first]) * float(weights[second]) * selected
            )
    covariance = np.diag((z_variance, m_variance))
    condition = float(np.linalg.cond(covariance))
    return _metric(
        math.fsum(contributions),
        contributions,
        (condition,),
        _emission_operation_count(order),
    )


def _term_metrics(
    h1: dict[str, object],
    recognition: dict[str, object],
    model: dict[str, object],
    *,
    order: Literal[21, 17],
) -> dict[str, _OracleMetric]:
    qg = _gaussians(recognition)
    qc = _categoricals(recognition)
    blocks = _blocks(model)
    metrics: dict[str, _OracleMetric] = {
        "expected_log_emission[1]": _emission_metric(
            h1, qg, blocks, time=1, order=order
        ),
        "expected_log_emission[2]": _emission_metric(
            h1, qg, blocks, time=2, order=order
        ),
    }
    initial = h1["initial_joint"]
    p_mean = initial["mean"]
    p_covariance = initial["covariance"]
    q_m = qg["q[m0]"]
    metrics["initial_model_kl"] = _conditional_kl_metric(
        q_m[1],
        float(p_covariance[1][1]),
        (q_m[0] - float(p_mean[1])) ** 2,
        _ANALYTIC_OPERATION_COUNTS["initial_model_kl"],
    )
    p_slope = float(p_covariance[0][1]) / float(p_covariance[1][1])
    p_offset = float(p_mean[0]) - p_slope * float(p_mean[1])
    p_variance = float(p_covariance[0][0]) - (
        float(p_covariance[0][1]) ** 2 / float(p_covariance[1][1])
    )
    q_z = qg["q[z0]"]
    slope_difference = -p_slope
    offset_difference = q_z[0] - p_offset
    mean_square = slope_difference * slope_difference * q_m[1] + (
        slope_difference * q_m[0] + offset_difference
    ) ** 2
    metrics["initial_state_kl"] = _conditional_kl_metric(
        q_z[1],
        p_variance,
        mean_square,
        _ANALYTIC_OPERATION_COUNTS["initial_state_kl"],
    )
    shared = float(blocks["theta[shared_decoder_transition]"]["s"])
    for time in (1, 2):
        gamma = qc[f"q[model_source_b{time}]"]
        model_source_id = f"model_source_kl[{time}]"
        model_contributions = tuple(
            q * (math.log(q) - math.log(float(p)))
            for q, p in zip(
                gamma, h1["model_source_priors"][time - 1], strict=True
            )
        )
        metrics[model_source_id] = _metric(
            math.fsum(model_contributions),
            model_contributions,
            (1.0,),
            _ANALYTIC_OPERATION_COUNTS[model_source_id],
        )
        rows = (
            (qc["q[state_source_a1_b0]"],)
            if time == 1
            else (qc["q[source_row_a2]"], qc["q[state_source_a2_b1]"])
        )
        state_source_id = f"state_source_kl[{time}]"
        state_prior = h1["state_source_priors"][time - 1]
        state_contributions = tuple(
            gamma[b]
            * q
            * (math.log(q) - math.log(float(state_prior[a])))
            for b, row in enumerate(rows)
            for a, q in enumerate(row)
        )
        metrics[state_source_id] = _metric(
            math.fsum(state_contributions),
            state_contributions,
            (1.0,),
            _ANALYTIC_OPERATION_COUNTS[state_source_id],
        )
        if time == 1:
            alpha_m = (float(h1["frames"][1]) / float(h1["frames"][0]),)
            c_m = float(h1["model_offsets"][0])
            R_m = float(h1["model_variances"][0])
            alpha_z = alpha_m
            B = float(h1["state_model_slopes"][0])
            c_z = float(h1["state_offsets"][0])
            R_z = float(h1["state_variances"][0])
        else:
            alpha_m = (
                float(h1["frames"][2]) / float(h1["frames"][0]),
                float(h1["frames"][2]) / float(h1["frames"][1]),
            )
            c_m = float(h1["model_offsets"][1])
            R_m = float(h1["model_variances"][1])
            transition = blocks["theta[state_transition_2]"]
            alpha_z = (
                float(transition["alpha_0"]),
                float(transition["alpha_1"]),
            )
            B = float(transition["B_base"]) + shared
            c_z = float(transition["c"])
            R_z = float(transition["R"])
        model_transition_id = f"model_transition_kl[{time}]"
        target_m = qg[f"q[m{time}]"]
        model_values: list[float] = []
        model_parts: list[float] = []
        for b, weight in enumerate(gamma):
            parent_m = qg[f"q[m{b}]"]
            mean_square = alpha_m[b] ** 2 * parent_m[1] + (
                target_m[0] - alpha_m[b] * parent_m[0] - c_m
            ) ** 2
            value, parts = _conditional_kl_parts(target_m[1], R_m, mean_square)
            model_values.append(weight * value)
            model_parts.extend(weight * part for part in parts)
        metrics[model_transition_id] = _metric(
            math.fsum(model_values),
            model_parts,
            (1.0,),
            _ANALYTIC_OPERATION_COUNTS[model_transition_id],
        )
        state_transition_id = f"state_transition_kl[{time}]"
        target_z = qg[f"q[z{time}]"]
        model_value = qg[f"q[m{time}]"]
        state_values: list[float] = []
        state_parts: list[float] = []
        for b, row in enumerate(rows):
            for a, conditional_weight in enumerate(row):
                parent_z = qg[f"q[z{a}]"]
                mean_square = (
                    alpha_z[a] ** 2 * parent_z[1]
                    + B * B * model_value[1]
                    + (
                        target_z[0]
                        - alpha_z[a] * parent_z[0]
                        - B * model_value[0]
                        - c_z
                    )
                    ** 2
                )
                value, parts = _conditional_kl_parts(target_z[1], R_z, mean_square)
                weight = gamma[b] * conditional_weight
                state_values.append(weight * value)
                state_parts.extend(weight * part for part in parts)
        metrics[state_transition_id] = _metric(
            math.fsum(state_values),
            state_parts,
            (1.0,),
            _ANALYTIC_OPERATION_COUNTS[state_transition_id],
        )
    if set(metrics) != set(_SIGNED_TERM_IDS):
        raise AssertionError("oracle complete-term construction changed")
    return metrics


def _build_term_evidence(
    term_id: str,
    objective_sign: int,
    metric_21: _OracleMetric,
    metric_17: _OracleMetric,
) -> H5OracleTermEvidence:
    value_21 = metric_21.value
    value_17 = metric_17.value
    convergence = abs(value_21 - value_17)
    rounding_21 = _rounding_allowance(
        value_21,
        metric_21.absolute_summands,
        metric_21.condition_numbers,
        metric_21.operation_count,
    )
    rounding_17 = _rounding_allowance(
        value_17,
        metric_17.absolute_summands,
        metric_17.condition_numbers,
        metric_17.operation_count,
    )
    comparison = _comparison_rounding(value_21, value_17)
    total = math.fsum((convergence, rounding_21, rounding_17, comparison))
    return H5OracleTermEvidence(
        "h5-oracle-term-evidence-v1",
        term_id,
        objective_sign,  # type: ignore[arg-type]
        value_21,
        value_17,
        float(objective_sign * value_21),
        metric_21.absolute_summands,
        metric_17.absolute_summands,
        metric_21.condition_numbers,
        metric_17.condition_numbers,
        metric_21.operation_count,
        metric_17.operation_count,
        float(convergence),
        float(rounding_21),
        float(rounding_17),
        float(comparison),
        float(total),
    )


def _complete_term_trace(
    h1: dict[str, object],
    recognition: dict[str, object],
    model: dict[str, object],
) -> tuple[H5OracleTermEvidence, ...]:
    metrics_21 = _term_metrics(h1, recognition, model, order=21)
    metrics_17 = _term_metrics(h1, recognition, model, order=17)
    return tuple(
        _build_term_evidence(
            term_id,
            _SIGNED_TERM_SIGNS[index],
            metrics_21[term_id],
            metrics_17[term_id],
        )
        for index, term_id in enumerate(_SIGNED_TERM_IDS)
    )


def _result(
    rule: str,
    h1: dict[str, object],
    before_recognition: dict[str, object],
    before_model: dict[str, object],
    after_recognition: dict[str, object],
    after_model: dict[str, object],
    conditions: tuple[tuple[str, float], ...],
) -> H5OracleUpdate:
    before = H5OracleOperandEvidence.from_complete_terms(
        operand="before",
        complete_term_trace=_complete_term_trace(
            h1, before_recognition, before_model
        ),
    )
    after = H5OracleOperandEvidence.from_complete_terms(
        operand="after",
        complete_term_trace=_complete_term_trace(
            h1, after_recognition, after_model
        ),
    )
    delta = H5OracleOperandEvidence.from_delta(before=before, after=after)
    return H5OracleUpdate(
        "h5-oracle-update-v1",
        rule,
        _canonical_json(after_recognition),
        _canonical_json(after_model),
        conditions,
        before,
        after,
        delta,
    )


def oracle_exact_e_block(
    h1_fixture_bytes: bytes, update_spec_bytes: bytes, live_state_bytes: bytes
) -> H5OracleUpdate:
    h1, _ = _parse_fixtures(h1_fixture_bytes, update_spec_bytes)
    recognition, model = _parse_semantic_state(live_state_bytes)
    after_recognition = copy.deepcopy(recognition)
    after_model = copy.deepcopy(model)
    qg = _gaussians(recognition)
    qc = _categoricals(recognition)
    blocks = _blocks(model)
    covariance = np.asarray(h1["initial_joint"]["covariance"], dtype=np.float64)
    precision = np.linalg.inv(covariance)
    information = precision @ np.asarray(h1["initial_joint"]["mean"], dtype=np.float64)
    transition = blocks["theta[state_transition_2]"]
    shared = float(blocks["theta[shared_decoder_transition]"]["s"])
    rows = (qc["q[source_row_a2]"], qc["q[state_source_a2_b1]"])
    gamma = qc["q[model_source_b2]"]
    w20 = math.fsum(gamma[b] * rows[b][0] for b in (0, 1))
    alpha10 = h1["frames"][1] / h1["frames"][0]
    alpha20 = float(transition["alpha_0"])
    B2 = float(transition["B_base"]) + shared
    J = (
        float(precision[0, 0])
        + alpha10 * alpha10 / h1["state_variances"][0]
        + w20 * alpha20 * alpha20 / float(transition["R"])
    )
    h = (
        float(information[0])
        - float(precision[0, 1]) * qg["q[m0]"][0]
        + alpha10
        * (
            qg["q[z1]"][0]
            - h1["state_model_slopes"][0] * qg["q[m1]"][0]
            - h1["state_offsets"][0]
        )
        / h1["state_variances"][0]
        + w20
        * alpha20
        * (qg["q[z2]"][0] - B2 * qg["q[m2]"][0] - float(transition["c"]))
        / float(transition["R"])
    )
    _set_gaussian(after_recognition, "q[z0]", h / J, 1.0 / J)
    return _result(
        "exact_z0", h1, recognition, model, after_recognition, after_model, ()
    )


def oracle_exact_source_row(
    h1_fixture_bytes: bytes, update_spec_bytes: bytes, live_state_bytes: bytes
) -> H5OracleUpdate:
    h1, _ = _parse_fixtures(h1_fixture_bytes, update_spec_bytes)
    recognition, model = _parse_semantic_state(live_state_bytes)
    after_recognition = copy.deepcopy(recognition)
    after_model = copy.deepcopy(model)
    qg = _gaussians(recognition)
    blocks = _blocks(model)
    transition = blocks["theta[state_transition_2]"]
    shared = float(blocks["theta[shared_decoder_transition]"]["s"])
    B = float(transition["B_base"]) + shared
    c = float(transition["c"])
    R = float(transition["R"])
    logits = []
    for a, alpha in enumerate((float(transition["alpha_0"]), float(transition["alpha_1"]))):
        residual = qg["q[z2]"][0] - alpha * qg[f"q[z{a}]"][0] - B * qg["q[m2]"][0] - c
        ell = -0.5 * (
            _LOG_2_PI
            + math.log(R)
            + (
                qg["q[z2]"][1]
                + alpha * alpha * qg[f"q[z{a}]"][1]
                + B * B * qg["q[m2]"][1]
                + residual * residual
            )
            / R
        )
        logits.append(math.log(h1["state_source_priors"][1][a]) + ell)
    maximum = max(logits)
    denominator = math.fsum(math.exp(item - maximum) for item in logits)
    probabilities = tuple(math.exp(item - maximum) / denominator for item in logits)
    _set_categorical(after_recognition, "q[source_row_a2]", probabilities)
    return _result(
        "exact_source_row_a2", h1, recognition, model, after_recognition, after_model, ()
    )


def oracle_exact_m_block(
    h1_fixture_bytes: bytes, update_spec_bytes: bytes, live_state_bytes: bytes
) -> H5OracleUpdate:
    h1, _ = _parse_fixtures(h1_fixture_bytes, update_spec_bytes)
    recognition, model = _parse_semantic_state(live_state_bytes)
    after_recognition = copy.deepcopy(recognition)
    after_model = copy.deepcopy(model)
    qg = _gaussians(recognition)
    qc = _categoricals(recognition)
    gamma = qc["q[model_source_b2]"]
    rows = (qc["q[source_row_a2]"], qc["q[state_source_a2_b1]"])
    G = np.zeros((4, 4), dtype=np.float64)
    g = np.zeros(4, dtype=np.float64)
    for b in (0, 1):
        for a in (0, 1):
            weight = gamma[b] * rows[b][a]
            z_mean, z_variance = qg[f"q[z{a}]"]
            m_mean, m_variance = qg["q[m2]"]
            means = np.asarray(
                (z_mean if a == 0 else 0.0, z_mean if a == 1 else 0.0, m_mean, 1.0),
                dtype=np.float64,
            )
            moment = np.outer(means, means)
            moment[a, a] += z_variance
            moment[2, 2] += m_variance
            G += weight * moment
            g += weight * means * qg["q[z2]"][0]
    G = 0.5 * (G + G.T)
    chol = np.linalg.cholesky(G)
    theta = np.linalg.solve(chol.T, np.linalg.solve(chol, g))
    R = (
        qg["q[z2]"][1]
        + qg["q[z2]"][0] ** 2
        - 2.0 * float(theta @ g)
        + float(theta @ G @ theta)
    )
    shared = float(_blocks(model)["theta[shared_decoder_transition]"]["s"])
    _set_block(
        after_model,
        "theta[state_transition_2]",
        {
            "alpha_0": float(theta[0]),
            "alpha_1": float(theta[1]),
            "B_base": float(theta[2]) - shared,
            "c": float(theta[3]),
            "R": float(R),
        },
    )
    return _result(
        "exact_state_transition_2_m",
        h1,
        recognition,
        model,
        after_recognition,
        after_model,
        (("G_condition_number", float(np.linalg.cond(G))),),
    )


def oracle_complete_delta(
    h1_fixture_bytes: bytes,
    update_spec_bytes: bytes,
    before_state_bytes: bytes,
    after_state_bytes: bytes,
    *,
    rule: str,
) -> H5OracleUpdate:
    h1, _ = _parse_fixtures(h1_fixture_bytes, update_spec_bytes)
    before_recognition, before_model = _parse_semantic_state(before_state_bytes)
    after_recognition, after_model = _parse_semantic_state(after_state_bytes)
    conditions: tuple[tuple[str, float], ...] = ()
    if rule == "exact_state_transition_2_m":
        raise ValueError("exact M complete-delta calls must retain the G condition number")
    return _result(
        rule,
        h1,
        before_recognition,
        before_model,
        after_recognition,
        after_model,
        conditions,
    )


__all__ = [
    "H5OracleTermEvidence",
    "H5OracleOperandEvidence",
    "H5OracleUpdate",
    "oracle_exact_e_block",
    "oracle_exact_source_row",
    "oracle_exact_m_block",
    "oracle_complete_delta",
]
