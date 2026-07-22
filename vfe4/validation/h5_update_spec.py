"""Strict raw-byte parser and reference builder for the H5 update specification."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

from vfe4.types.h5_schema import (
    H5_FACTOR_INPUT_SCHEMA_VERSION,
    H5_FACTOR_UNIVERSE,
    H5_H1_FIXTURE_RAW_SHA256,
    H5_MODEL_BLOCK_UNIVERSE,
    H5_QUADRATURE_ORDERS,
    H5_RECOGNITION_COORDINATE_UNIVERSE,
    H5_RECONSTRUCTION_ROWS,
    H5_SHARED_PARAMETER_GROUP_ROWS,
    H5_UPDATE_SPEC_DOMAIN,
)
from vfe4.types.updates import (
    CategoricalRecognitionCoordinate,
    FactorReconstructionRecord,
    FrozenByteState,
    FrozenTensorValue,
    GaussianRecognitionCoordinate,
    H5ModelSnapshot,
    H5ReferenceState,
    ModelParameterBlock,
    RecognitionSnapshot,
    SharedParameterGroup,
    UpdateSpecification,
)


EXPECTED_H1_FIXTURE_RAW_SHA256 = H5_H1_FIXTURE_RAW_SHA256
EXPECTED_H5_UPDATE_SPEC_RAW_SHA256 = (
    "9dd42603419952a2ffa4b6602971240ec00572283557d672ae6ee106c31dd91c"
)

_ROOT_FIELD_ORDER = (
    "fixture_id",
    "fixture_schema_version",
    "recognition_family",
    "h1_fixture_id",
    "h1_fixture_sha256",
    "factor_input_schema_version",
    "factor_universe",
    "recognition_coordinate_universe",
    "model_block_universe",
    "quadrature_orders",
    "continuous_recognition",
    "categorical_recognition",
    "model_parameter_blocks",
    "factor_reconstruction",
    "shared_parameter_groups",
    "source_row_a2",
)
_SOURCE_ROW_FIELD_ORDER = (
    "coordinate_id",
    "time",
    "condition",
    "support",
    "initial_probabilities",
)
_CONTINUOUS_VALUES = (
    ("q[z0]", -0.10, 0.65),
    ("q[m0]", 0.25, 0.78),
    ("q[z1]", 0.05, 0.96),
    ("q[m1]", 0.175, 1.21),
    ("q[z2]", -0.04, 0.90),
    ("q[m2]", 0.14, 1.40),
)
_CATEGORICAL_VALUES = (
    ("q[model_source_b1]", (0,), (), (1.0,)),
    ("q[state_source_a1_b0]", (0,), (("b1", 0),), (1.0,)),
    ("q[model_source_b2]", (0, 1), (), (0.4, 0.6)),
    ("q[source_row_a2]", (0, 1), (("b2", 0),), (0.75, 0.25)),
    ("q[state_source_a2_b1]", (0, 1), (("b2", 1),), (0.2, 0.8)),
)
_MODEL_VALUES: tuple[tuple[str, tuple[tuple[str, float | tuple[float, ...]], ...]], ...] = (
    (
        "theta[state_transition_2]",
        (
            ("alpha_0", 0.8),
            ("alpha_1", 0.64),
            ("B_base", -0.35),
            ("c", 0.08),
            ("R", 0.48),
        ),
    ),
    (
        "theta[emission_1]",
        (
            ("w_z", (0.2, -0.4, 0.1)),
            ("w_m", (0.3, 0.2, -0.5)),
            ("bias", (0.05, -0.1, 0.15)),
        ),
    ),
    ("theta[shared_decoder_transition]", (("s", 0.0),)),
)


@dataclass(frozen=True)
class _JSONObject:
    pairs: tuple[tuple[str, object], ...]


def _object_pairs(pairs: list[tuple[str, object]]) -> _JSONObject:
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        duplicate = next(key for index, key in enumerate(keys) if key in keys[:index])
        raise ValueError(f"duplicate JSON key: {duplicate}")
    return _JSONObject(tuple(pairs))


def _reject_constant(value: str) -> object:
    raise ValueError(f"nonfinite JSON constant is forbidden: {value}")


def _ordered_object(
    value: object, expected_order: tuple[str, ...], name: str
) -> dict[str, object]:
    if not isinstance(value, _JSONObject):
        raise ValueError(f"{name} must be a JSON object")
    keys = tuple(key for key, _ in value.pairs)
    if set(keys) != set(expected_order):
        raise ValueError(f"{name} fields must equal {list(expected_order)}")
    if keys != expected_order:
        raise ValueError(f"{name} field order must equal {list(expected_order)}")
    return dict(value.pairs)


def _list(value: object, size: int, name: str) -> list[object]:
    if type(value) is not list or len(value) != size:
        raise ValueError(f"{name} must be a list of length {size}")
    return value


def _string(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _integer(value: object, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def _number(value: object, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite numeric data")
    return float(value)


def _string_tuple(value: object, expected: tuple[str, ...], name: str) -> tuple[str, ...]:
    items = _list(value, len(expected), name)
    result = tuple(_string(item, f"{name}[{index}]") for index, item in enumerate(items))
    if result != expected:
        raise ValueError(f"{name} must equal the frozen H5 order")
    return result


def _int_tuple(value: object, expected: tuple[int, ...], name: str) -> tuple[int, ...]:
    items = _list(value, len(expected), name)
    result = tuple(_integer(item, f"{name}[{index}]") for index, item in enumerate(items))
    if result != expected:
        raise ValueError(f"{name} must equal {expected}")
    return result


def _float_tuple(
    value: object, expected: tuple[float, ...], name: str
) -> tuple[float, ...]:
    items = _list(value, len(expected), name)
    result = tuple(_number(item, f"{name}[{index}]") for index, item in enumerate(items))
    if result != expected:
        raise ValueError(f"{name} must equal the frozen H5 values")
    return result


def _probability_tuple(
    value: object, expected: tuple[float, ...], name: str
) -> tuple[float, ...]:
    items = _list(value, len(expected), name)
    result = tuple(_number(item, f"{name}[{index}]") for index, item in enumerate(items))
    if any(probability <= 0.0 for probability in result):
        raise ValueError(f"{name} must be positive on its declared support")
    allowance = 64.0 * math.ulp(1.0) * max(1, len(result))
    if abs(math.fsum(result) - 1.0) > allowance:
        raise ValueError(f"{name} probabilities must be normalized")
    if result != expected:
        raise ValueError(f"{name} must equal the frozen H5 probabilities")
    return result


def _condition_tuple(
    value: object, expected: tuple[tuple[str, int], ...], name: str
) -> tuple[tuple[str, int], ...]:
    rows = _list(value, len(expected), name)
    result: list[tuple[str, int]] = []
    for index, row in enumerate(rows):
        pair = _list(row, 2, f"{name}[{index}]")
        result.append(
            (
                _string(pair[0], f"{name}[{index}][0]"),
                _integer(pair[1], f"{name}[{index}][1]"),
            )
        )
    if tuple(result) != expected:
        raise ValueError(f"{name} must equal the frozen H5 conditioning")
    return tuple(result)


def _decode_checked_json(data: bytes) -> dict[str, object]:
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"H5 update specification is not UTF-8: {exc}") from exc
    try:
        decoded = json.loads(
            text,
            object_pairs_hook=_object_pairs,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"H5 update specification JSON is invalid: {exc}") from exc
    return _ordered_object(decoded, _ROOT_FIELD_ORDER, "fixture")


def parse_h5_update_spec_bytes(data: bytes) -> UpdateSpecification:
    """Parse only the one raw-byte-pinned H5 v1 conditional update fixture."""
    if type(data) is not bytes:
        raise ValueError("H5 update specification must be bytes")
    raw_bytes = memoryview(data).tobytes()
    raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if raw_sha256 != EXPECTED_H5_UPDATE_SPEC_RAW_SHA256:
        raise ValueError(
            "H5 update-spec raw SHA-256 mismatch: "
            f"expected {EXPECTED_H5_UPDATE_SPEC_RAW_SHA256}, got {raw_sha256}"
        )

    root = _decode_checked_json(raw_bytes)
    if root["fixture_id"] != "h5-conditional-update-v1":
        raise ValueError("fixture_id must equal h5-conditional-update-v1")
    if _integer(root["fixture_schema_version"], "fixture_schema_version") != 1:
        raise ValueError("fixture_schema_version must equal 1")
    if root["recognition_family"] != "continuous_mean_field_conditional_categorical":
        raise ValueError("recognition_family must equal the H5 conditional family")
    if root["h1_fixture_id"] != "h1-v1":
        raise ValueError("h1_fixture_id must equal h1-v1")
    if root["h1_fixture_sha256"] != EXPECTED_H1_FIXTURE_RAW_SHA256:
        raise ValueError("h1_fixture_sha256 must equal the full frozen H1 raw SHA-256")
    if root["factor_input_schema_version"] != H5_FACTOR_INPUT_SCHEMA_VERSION:
        raise ValueError("factor_input_schema_version must equal h5-factor-input-v1")
    factor_universe = _string_tuple(
        root["factor_universe"], H5_FACTOR_UNIVERSE, "factor_universe"
    )
    coordinate_universe = _string_tuple(
        root["recognition_coordinate_universe"],
        H5_RECOGNITION_COORDINATE_UNIVERSE,
        "recognition_coordinate_universe",
    )
    model_universe = _string_tuple(
        root["model_block_universe"], H5_MODEL_BLOCK_UNIVERSE, "model_block_universe"
    )
    quadrature_orders = _int_tuple(
        root["quadrature_orders"], H5_QUADRATURE_ORDERS, "quadrature_orders"
    )

    continuous_rows = _list(
        root["continuous_recognition"],
        len(_CONTINUOUS_VALUES),
        "continuous_recognition",
    )
    gaussians: list[GaussianRecognitionCoordinate] = []
    for index, (row_value, expected) in enumerate(
        zip(continuous_rows, _CONTINUOUS_VALUES, strict=True)
    ):
        row = _list(row_value, 3, f"continuous_recognition[{index}]")
        identifier = _string(row[0], f"continuous_recognition[{index}][0]")
        mean = _number(row[1], f"continuous_recognition[{index}][1]")
        variance = _number(row[2], f"continuous_recognition[{index}][2]")
        if (identifier, mean, variance) != expected:
            raise ValueError(
                f"continuous_recognition[{index}] must equal the frozen H5 coordinate"
            )
        gaussians.append(
            GaussianRecognitionCoordinate(
                identifier,
                FrozenTensorValue("float64", (), (mean,)),
                FrozenTensorValue("float64", (), (variance,)),
            )
        )

    categorical_rows = _list(
        root["categorical_recognition"],
        len(_CATEGORICAL_VALUES),
        "categorical_recognition",
    )
    categoricals: list[CategoricalRecognitionCoordinate] = []
    for index, (row_value, expected) in enumerate(
        zip(categorical_rows, _CATEGORICAL_VALUES, strict=True)
    ):
        row = _list(row_value, 4, f"categorical_recognition[{index}]")
        identifier = _string(row[0], f"categorical_recognition[{index}][0]")
        support = _int_tuple(
            row[1], expected[1], f"categorical_recognition[{index}].support"
        )
        conditions = _condition_tuple(
            row[2], expected[2], f"categorical_recognition[{index}].conditioned_on"
        )
        probabilities = _probability_tuple(
            row[3], expected[3], f"categorical_recognition[{index}].probabilities"
        )
        if identifier != expected[0]:
            raise ValueError(
                f"categorical_recognition[{index}] coordinate_id must equal H5 order"
            )
        categoricals.append(
            CategoricalRecognitionCoordinate(
                identifier,
                support,
                conditions,
                FrozenTensorValue("float64", (len(probabilities),), probabilities),
            )
        )
    recognition = RecognitionSnapshot(
        "h5-recognition-snapshot-v1", tuple(gaussians), tuple(categoricals)
    )

    model_rows = _list(
        root["model_parameter_blocks"], len(_MODEL_VALUES), "model_parameter_blocks"
    )
    parameter_blocks: list[ModelParameterBlock] = []
    for block_index, (row_value, expected_block) in enumerate(
        zip(model_rows, _MODEL_VALUES, strict=True)
    ):
        row = _list(row_value, 2, f"model_parameter_blocks[{block_index}]")
        block_id = _string(row[0], f"model_parameter_blocks[{block_index}][0]")
        if block_id != expected_block[0]:
            raise ValueError("model_parameter_blocks must use exact H5 block order")
        value_rows = _list(
            row[1], len(expected_block[1]), f"model_parameter_blocks[{block_index}][1]"
        )
        values: list[tuple[str, FrozenTensorValue]] = []
        for value_index, (value_row, expected_value) in enumerate(
            zip(value_rows, expected_block[1], strict=True)
        ):
            pair = _list(
                value_row,
                2,
                f"model_parameter_blocks[{block_index}][1][{value_index}]",
            )
            name = _string(pair[0], "model parameter name")
            if name != expected_value[0]:
                raise ValueError("model parameter names must equal the frozen H5 order")
            expected_data = expected_value[1]
            if type(expected_data) is tuple:
                numbers = _float_tuple(pair[1], expected_data, f"{block_id}.{name}")
                tensor = FrozenTensorValue("float64", (len(numbers),), numbers)
            else:
                number = _number(pair[1], f"{block_id}.{name}")
                if number != expected_data:
                    raise ValueError(f"{block_id}.{name} must equal the frozen H5 value")
                tensor = FrozenTensorValue("float64", (), (number,))
            values.append((name, tensor))
        parameter_blocks.append(ModelParameterBlock(block_id, tuple(values)))

    reconstruction_rows = _list(
        root["factor_reconstruction"],
        len(H5_RECONSTRUCTION_ROWS),
        "factor_reconstruction",
    )
    reconstruction_records: list[FactorReconstructionRecord] = []
    for index, (row_value, expected) in enumerate(
        zip(reconstruction_rows, H5_RECONSTRUCTION_ROWS, strict=True)
    ):
        row = _list(row_value, 2, f"factor_reconstruction[{index}]")
        factor_id = _string(row[0], f"factor_reconstruction[{index}][0]")
        bindings = _string_tuple(
            row[1], expected[1], f"factor_reconstruction[{index}][1]"
        )
        if factor_id != expected[0]:
            raise ValueError("factor_reconstruction must use exact H5 factor order")
        reconstruction_records.append(FactorReconstructionRecord(factor_id, bindings))

    shared_rows = _list(
        root["shared_parameter_groups"],
        len(H5_SHARED_PARAMETER_GROUP_ROWS),
        "shared_parameter_groups",
    )
    shared_groups: list[SharedParameterGroup] = []
    for index, (row_value, expected) in enumerate(
        zip(shared_rows, H5_SHARED_PARAMETER_GROUP_ROWS, strict=True)
    ):
        row = _list(row_value, 3, f"shared_parameter_groups[{index}]")
        group_id = _string(row[0], f"shared_parameter_groups[{index}][0]")
        source = _string(row[1], f"shared_parameter_groups[{index}][1]")
        consumers = _string_tuple(
            row[2], expected[2], f"shared_parameter_groups[{index}][2]"
        )
        if (group_id, source, consumers) != expected:
            raise ValueError("shared_parameter_groups must equal the frozen H5 shared group")
        shared_groups.append(SharedParameterGroup(group_id, source, consumers))

    source_row = _ordered_object(root["source_row_a2"], _SOURCE_ROW_FIELD_ORDER, "source_row_a2")
    source_row_core = (
        source_row["coordinate_id"],
        _integer(source_row["time"], "source_row_a2.time"),
        _condition_tuple(
            [source_row["condition"]], (("b2", 0),), "source_row_a2.condition"
        )[0],
        _int_tuple(source_row["support"], (0, 1), "source_row_a2.support"),
        _float_tuple(
            source_row["initial_probabilities"],
            (0.75, 0.25),
            "source_row_a2.initial_probabilities",
        ),
    )
    if source_row_core != ("q[source_row_a2]", 2, ("b2", 0), (0, 1), (0.75, 0.25)):
        raise ValueError("source_row_a2 must equal the frozen H5 row")

    model = H5ModelSnapshot(
        "h5-model-snapshot-v1",
        tuple(parameter_blocks),
        tuple(reconstruction_records),
        tuple(shared_groups),
    )
    specification = UpdateSpecification(
        raw_bytes,
        "h5-conditional-update-v1",
        1,
        "continuous_mean_field_conditional_categorical",
        "h1-v1",
        EXPECTED_H1_FIXTURE_RAW_SHA256,
        H5_FACTOR_INPUT_SCHEMA_VERSION,
        factor_universe,
        coordinate_universe,
        model_universe,
        quadrature_orders,
        tuple(reconstruction_records),
        tuple(shared_groups),
        recognition,
        model,
    )
    if specification.raw_sha256 != EXPECTED_H5_UPDATE_SPEC_RAW_SHA256:
        raise ValueError("constructed H5 specification raw digest changed unexpectedly")
    return specification


def canonical_h5_update_specification_bytes(
    specification: UpdateSpecification,
) -> bytes:
    if not isinstance(specification, UpdateSpecification):
        raise ValueError("specification must be an UpdateSpecification")
    return memoryview(specification.canonical_bytes).tobytes()


def build_h5_reference_state(
    h1_fixture_bytes: bytes,
    h5_update_spec_bytes: bytes,
) -> H5ReferenceState:
    if type(h1_fixture_bytes) is not bytes:
        raise ValueError("H1 fixture must be supplied as bytes")
    h1_raw = memoryview(h1_fixture_bytes).tobytes()
    h1_sha256 = hashlib.sha256(h1_raw).hexdigest()
    if h1_sha256 != EXPECTED_H1_FIXTURE_RAW_SHA256:
        raise ValueError(
            "H1 raw SHA-256 mismatch: "
            f"expected {EXPECTED_H1_FIXTURE_RAW_SHA256}, got {h1_sha256}"
        )
    specification = parse_h5_update_spec_bytes(h5_update_spec_bytes)
    return H5ReferenceState(
        "h5-reference-state-v1",
        h1_raw,
        specification.raw_bytes,
        specification,
        specification.initial_recognition,
        specification.initial_model,
        FrozenByteState("h5-no-optimizer-v1", b'{"kind":"none"}'),
        FrozenByteState(
            "h5-deterministic-rng-v1",
            b'{"algorithm":"none","counter":0}',
        ),
    )


__all__ = [
    "EXPECTED_H1_FIXTURE_RAW_SHA256",
    "EXPECTED_H5_UPDATE_SPEC_RAW_SHA256",
    "H5_UPDATE_SPEC_DOMAIN",
    "build_h5_reference_state",
    "canonical_h5_update_specification_bytes",
    "parse_h5_update_spec_bytes",
]
