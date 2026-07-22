"""Immutable H5 update, snapshot, request, and provenance records."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Literal, Mapping

import torch
from torch import Tensor

from vfe4.types.h1 import (
    GaussianLaw,
    H1RecognitionFactorRecord,
    RecognitionModelKernelRecord,
    RecognitionStateKernelRecord,
)
from vfe4.types.h5_schema import (
    H5_CANDIDATE_DOMAIN,
    H5_FACTOR_INPUT_SCHEMA_SHA256,
    H5_FACTOR_INPUT_SCHEMA_VERSION,
    H5_FACTOR_UNIVERSE,
    H5_H1_FIXTURE_RAW_SHA256,
    H5_LIVE_STATE_DOMAIN,
    H5_MODEL_BLOCK_UNIVERSE,
    H5_MODEL_SNAPSHOT_DOMAIN,
    H5_OBJECTIVE_SCHEMA_SHA256,
    H5_OPTIMIZER_STATE_DOMAIN,
    H5_QUADRATURE_ORDERS,
    H5_RECOGNITION_COORDINATE_UNIVERSE,
    H5_RECOGNITION_SNAPSHOT_DOMAIN,
    H5_RECONSTRUCTION_ROWS,
    H5_REFERENCE_STATE_DOMAIN,
    H5_RNG_STATE_DOMAIN,
    H5_SEMANTIC_STATE_DOMAIN,
    H5_SHARED_PARAMETER_GROUP_ROWS,
    H5_UPDATE_REQUEST_DOMAIN,
    H5_UPDATE_SPEC_DOMAIN,
)


_LOWER_HEX = frozenset("0123456789abcdef")
_GAUSSIAN_IDS = H5_RECOGNITION_COORDINATE_UNIVERSE[:6]
_CATEGORICAL_IDS = H5_RECOGNITION_COORDINATE_UNIVERSE[6:]
_CATEGORICAL_SHAPES = {
    "q[model_source_b1]": ((0,), (), (1.0,)),
    "q[state_source_a1_b0]": ((0,), (("b1", 0),), (1.0,)),
    "q[model_source_b2]": ((0, 1), (), None),
    "q[source_row_a2]": ((0, 1), (("b2", 0),), None),
    "q[state_source_a2_b1]": ((0, 1), (("b2", 1),), None),
}
_MODEL_VALUE_SCHEMAS = {
    "theta[state_transition_2]": (
        ("alpha_0", ()),
        ("alpha_1", ()),
        ("B_base", ()),
        ("c", ()),
        ("R", ()),
    ),
    "theta[emission_1]": (("w_z", (3,)), ("w_m", (3,)), ("bias", (3,))),
    "theta[shared_decoder_transition]": (("s", ()),),
}
_RECONSTRUCTION_BY_FACTOR = dict(H5_RECONSTRUCTION_ROWS)


def _copy_bytes(value: object, name: str) -> bytes:
    if type(value) is not bytes:
        raise ValueError(f"{name} must be bytes")
    return memoryview(value).tobytes()


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase 64-hex SHA-256")
    return value


def _require_nonempty_string(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _require_exact_tuple(value: object, name: str) -> tuple[object, ...]:
    if type(value) is not tuple:
        raise ValueError(f"{name} must be a tuple")
    return value


def _finite_float(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite binary64 float")
    return value


def _canonicalize(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("canonical H5 floats must be finite")
        return value.hex()
    if type(value) in (str, int, bool) or value is None:
        return value
    if type(value) is bytes:
        return {"hex": value.hex(), "length": len(value)}
    if type(value) is tuple:
        return [_canonicalize(item) for item in value]
    if isinstance(value, Mapping):
        if not all(type(key) is str and key for key in value):
            raise ValueError("canonical H5 mapping keys must be nonempty strings")
        return {key: _canonicalize(value[key]) for key in sorted(value)}
    raise ValueError(f"unsupported canonical H5 value: {type(value).__name__}")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        _canonicalize(value),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def _domain_hash(domain: bytes, core: bytes) -> str:
    return hashlib.sha256(domain + core).hexdigest()


@dataclass(frozen=True)
class FrozenTensorValue:
    dtype: Literal["float64"]
    shape: tuple[int, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.dtype != "float64":
            raise ValueError("dtype must equal float64")
        _require_exact_tuple(self.shape, "shape")
        if any(type(dimension) is not int or dimension < 0 for dimension in self.shape):
            raise ValueError("shape must contain nonnegative integer dimensions")
        expected = math.prod(self.shape) if self.shape else 1
        _require_exact_tuple(self.values, "values")
        if len(self.values) != expected:
            raise ValueError("values length must equal the row-major shape product")
        checked = tuple(
            _finite_float(value, f"values[{index}]")
            for index, value in enumerate(self.values)
        )
        object.__setattr__(self, "shape", tuple(self.shape))
        object.__setattr__(self, "values", checked)

    @classmethod
    def from_tensor(cls, value: Tensor) -> FrozenTensorValue:
        if not isinstance(value, Tensor):
            raise ValueError("value must be a torch.Tensor")
        owned = value.detach().to(device="cpu", dtype=torch.float64).contiguous().clone()
        if not bool(torch.isfinite(owned).all()):
            raise ValueError("value must be finite")
        flattened = tuple(float(item) for item in owned.reshape(-1).tolist())
        return cls("float64", tuple(int(size) for size in owned.shape), flattened)

    def to_tensor(self) -> Tensor:
        return torch.tensor(self.values, dtype=torch.float64).reshape(self.shape).clone()


def _clone_tensor(value: FrozenTensorValue, name: str) -> FrozenTensorValue:
    if not isinstance(value, FrozenTensorValue):
        raise ValueError(f"{name} must be a FrozenTensorValue")
    return FrozenTensorValue(value.dtype, tuple(value.shape), tuple(value.values))


@dataclass(frozen=True)
class FrozenByteState:
    schema_version: str
    payload: bytes
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version not in (
            "h5-no-optimizer-v1",
            "h5-deterministic-rng-v1",
        ):
            raise ValueError("unsupported frozen byte-state schema")
        payload = _copy_bytes(self.payload, "payload")
        domain = (
            H5_OPTIMIZER_STATE_DOMAIN
            if self.schema_version == "h5-no-optimizer-v1"
            else H5_RNG_STATE_DOMAIN
        )
        core = _canonical_json_bytes((self.schema_version, payload))
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "state_sha256", _domain_hash(domain, core))


def _clone_byte_state(value: FrozenByteState, name: str) -> FrozenByteState:
    if not isinstance(value, FrozenByteState):
        raise ValueError(f"{name} must be a FrozenByteState")
    return FrozenByteState(value.schema_version, _copy_bytes(value.payload, name))


def _clone_typed_byte_state(
    value: FrozenByteState,
    *,
    expected_schema: str,
    name: str,
) -> FrozenByteState:
    cloned = _clone_byte_state(value, name)
    if cloned.schema_version != expected_schema:
        raise ValueError(f"{name} schema must equal {expected_schema}")
    return cloned


@dataclass(frozen=True)
class GaussianRecognitionCoordinate:
    coordinate_id: str
    mean: FrozenTensorValue
    variance: FrozenTensorValue

    def __post_init__(self) -> None:
        if self.coordinate_id not in _GAUSSIAN_IDS:
            raise ValueError("Gaussian coordinate_id is outside the H5 universe")
        mean = _clone_tensor(self.mean, "mean")
        variance = _clone_tensor(self.variance, "variance")
        if mean.shape != () or variance.shape != ():
            raise ValueError("Gaussian mean and variance must be scalar values")
        if variance.values[0] <= 0.0:
            raise ValueError("Gaussian variance must be strictly positive")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "variance", variance)


def _clone_gaussian(
    value: GaussianRecognitionCoordinate, name: str
) -> GaussianRecognitionCoordinate:
    if not isinstance(value, GaussianRecognitionCoordinate):
        raise ValueError(f"{name} must be a GaussianRecognitionCoordinate")
    return GaussianRecognitionCoordinate(value.coordinate_id, value.mean, value.variance)


@dataclass(frozen=True)
class CategoricalRecognitionCoordinate:
    coordinate_id: str
    support: tuple[int, ...]
    conditioned_on: tuple[tuple[str, int], ...]
    probabilities: FrozenTensorValue

    def __post_init__(self) -> None:
        if self.coordinate_id not in _CATEGORICAL_IDS:
            raise ValueError("categorical coordinate_id is outside the H5 universe")
        _require_exact_tuple(self.support, "support")
        if (
            not self.support
            or any(type(item) is not int or item < 0 for item in self.support)
            or len(set(self.support)) != len(self.support)
        ):
            raise ValueError("support must be unique nonnegative integers")
        _require_exact_tuple(self.conditioned_on, "conditioned_on")
        checked_conditions: list[tuple[str, int]] = []
        for index, condition in enumerate(self.conditioned_on):
            if (
                type(condition) is not tuple
                or len(condition) != 2
                or type(condition[0]) is not str
                or not condition[0]
                or type(condition[1]) is not int
                or condition[1] < 0
            ):
                raise ValueError(f"conditioned_on[{index}] is invalid")
            checked_conditions.append((condition[0], condition[1]))
        probabilities = _clone_tensor(self.probabilities, "probabilities")
        if probabilities.shape != (len(self.support),):
            raise ValueError("probabilities shape must match support")
        if any(value <= 0.0 for value in probabilities.values):
            raise ValueError("probabilities must be positive on declared support")
        allowance = 64.0 * math.ulp(1.0) * max(1, len(probabilities.values))
        if abs(math.fsum(probabilities.values) - 1.0) > allowance:
            raise ValueError("probabilities must be normalized")
        expected_support, expected_condition, singleton = _CATEGORICAL_SHAPES[
            self.coordinate_id
        ]
        if tuple(self.support) != expected_support or tuple(checked_conditions) != expected_condition:
            raise ValueError("categorical support/conditioning does not match H5")
        if singleton is not None and probabilities.values != singleton:
            raise ValueError("singleton categorical probability must equal one")
        object.__setattr__(self, "support", tuple(self.support))
        object.__setattr__(self, "conditioned_on", tuple(checked_conditions))
        object.__setattr__(self, "probabilities", probabilities)


def _clone_categorical(
    value: CategoricalRecognitionCoordinate, name: str
) -> CategoricalRecognitionCoordinate:
    if not isinstance(value, CategoricalRecognitionCoordinate):
        raise ValueError(f"{name} must be a CategoricalRecognitionCoordinate")
    return CategoricalRecognitionCoordinate(
        value.coordinate_id,
        tuple(value.support),
        tuple((key, index) for key, index in value.conditioned_on),
        value.probabilities,
    )


def _tensor_core(value: FrozenTensorValue) -> object:
    return {"dtype": value.dtype, "shape": value.shape, "values": value.values}


def _recognition_core(snapshot: RecognitionSnapshot) -> object:
    return {
        "schema_version": snapshot.schema_version,
        "gaussians": tuple(
            (
                coordinate.coordinate_id,
                _tensor_core(coordinate.mean),
                _tensor_core(coordinate.variance),
            )
            for coordinate in snapshot.gaussians
        ),
        "categoricals": tuple(
            (
                coordinate.coordinate_id,
                coordinate.support,
                coordinate.conditioned_on,
                _tensor_core(coordinate.probabilities),
            )
            for coordinate in snapshot.categoricals
        ),
    }


def canonical_h5_recognition_snapshot_bytes(snapshot: RecognitionSnapshot) -> bytes:
    if not isinstance(snapshot, RecognitionSnapshot):
        raise ValueError("snapshot must be a RecognitionSnapshot")
    return _canonical_json_bytes(_recognition_core(snapshot))


@dataclass(frozen=True)
class RecognitionSnapshot:
    schema_version: Literal["h5-recognition-snapshot-v1"]
    gaussians: tuple[GaussianRecognitionCoordinate, ...]
    categoricals: tuple[CategoricalRecognitionCoordinate, ...]
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != "h5-recognition-snapshot-v1":
            raise ValueError("unsupported recognition snapshot schema")
        _require_exact_tuple(self.gaussians, "gaussians")
        _require_exact_tuple(self.categoricals, "categoricals")
        gaussians = tuple(
            _clone_gaussian(value, f"gaussians[{index}]")
            for index, value in enumerate(self.gaussians)
        )
        categoricals = tuple(
            _clone_categorical(value, f"categoricals[{index}]")
            for index, value in enumerate(self.categoricals)
        )
        if tuple(value.coordinate_id for value in gaussians) != _GAUSSIAN_IDS:
            raise ValueError("Gaussian coordinates must equal the H5 universe order")
        if tuple(value.coordinate_id for value in categoricals) != _CATEGORICAL_IDS:
            raise ValueError("categorical coordinates must equal the H5 universe order")
        object.__setattr__(self, "gaussians", gaussians)
        object.__setattr__(self, "categoricals", categoricals)
        object.__setattr__(
            self,
            "state_sha256",
            _domain_hash(
                H5_RECOGNITION_SNAPSHOT_DOMAIN,
                canonical_h5_recognition_snapshot_bytes(self),
            ),
        )


def _clone_recognition(value: RecognitionSnapshot, name: str) -> RecognitionSnapshot:
    if not isinstance(value, RecognitionSnapshot):
        raise ValueError(f"{name} must be a RecognitionSnapshot")
    return RecognitionSnapshot(value.schema_version, value.gaussians, value.categoricals)


@dataclass(frozen=True)
class ModelParameterBlock:
    block_id: str
    values: tuple[tuple[str, FrozenTensorValue], ...]

    def __post_init__(self) -> None:
        if self.block_id not in H5_MODEL_BLOCK_UNIVERSE:
            raise ValueError("model block_id is outside the H5 universe")
        _require_exact_tuple(self.values, "values")
        checked: list[tuple[str, FrozenTensorValue]] = []
        for index, item in enumerate(self.values):
            if type(item) is not tuple or len(item) != 2 or type(item[0]) is not str:
                raise ValueError(f"values[{index}] must be a named tensor pair")
            checked.append((item[0], _clone_tensor(item[1], f"values[{index}][1]")))
        schema = _MODEL_VALUE_SCHEMAS[self.block_id]
        if tuple((name, value.shape) for name, value in checked) != schema:
            raise ValueError("model block field names/shapes do not match H5")
        if self.block_id == "theta[state_transition_2]" and checked[-1][1].values[0] <= 0.0:
            raise ValueError("theta[state_transition_2].R must be positive")
        object.__setattr__(self, "values", tuple(checked))


def _clone_model_block(value: ModelParameterBlock, name: str) -> ModelParameterBlock:
    if not isinstance(value, ModelParameterBlock):
        raise ValueError(f"{name} must be a ModelParameterBlock")
    return ModelParameterBlock(value.block_id, tuple(value.values))


@dataclass(frozen=True)
class FactorReconstructionRecord:
    factor_id: str
    bindings: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.factor_id not in H5_FACTOR_UNIVERSE:
            raise ValueError("factor_id is outside the H5 factor universe")
        _require_exact_tuple(self.bindings, "bindings")
        if (
            not self.bindings
            or any(type(value) is not str or not value for value in self.bindings)
            or len(set(self.bindings)) != len(self.bindings)
        ):
            raise ValueError("bindings must be unique nonempty strings")
        if tuple(self.bindings) != _RECONSTRUCTION_BY_FACTOR[self.factor_id]:
            raise ValueError("factor reconstruction bindings do not match H5")
        object.__setattr__(self, "bindings", tuple(self.bindings))


def _clone_reconstruction(
    value: FactorReconstructionRecord, name: str
) -> FactorReconstructionRecord:
    if not isinstance(value, FactorReconstructionRecord):
        raise ValueError(f"{name} must be a FactorReconstructionRecord")
    return FactorReconstructionRecord(value.factor_id, tuple(value.bindings))


@dataclass(frozen=True)
class SharedParameterGroup:
    group_id: Literal["shared_decoder_transition"]
    source: Literal["theta[shared_decoder_transition].s"]
    consumers: tuple[str, ...]

    def __post_init__(self) -> None:
        expected = H5_SHARED_PARAMETER_GROUP_ROWS[0]
        if (self.group_id, self.source, self.consumers) != expected:
            raise ValueError("shared parameter group must match H5 exactly")
        object.__setattr__(self, "consumers", tuple(self.consumers))


def _clone_shared_group(value: SharedParameterGroup, name: str) -> SharedParameterGroup:
    if not isinstance(value, SharedParameterGroup):
        raise ValueError(f"{name} must be a SharedParameterGroup")
    return SharedParameterGroup(value.group_id, value.source, tuple(value.consumers))


def _model_core(snapshot: H5ModelSnapshot) -> object:
    return {
        "schema_version": snapshot.schema_version,
        "objective_schema_sha256": snapshot.objective_schema_sha256,
        "parameter_blocks": tuple(
            (
                block.block_id,
                tuple((name, _tensor_core(value)) for name, value in block.values),
            )
            for block in snapshot.parameter_blocks
        ),
        "reconstruction_records": tuple(
            (record.factor_id, record.bindings)
            for record in snapshot.reconstruction_records
        ),
        "shared_groups": tuple(
            (group.group_id, group.source, group.consumers)
            for group in snapshot.shared_groups
        ),
    }


def canonical_h5_model_snapshot_bytes(snapshot: H5ModelSnapshot) -> bytes:
    if not isinstance(snapshot, H5ModelSnapshot):
        raise ValueError("snapshot must be an H5ModelSnapshot")
    return _canonical_json_bytes(_model_core(snapshot))


@dataclass(frozen=True)
class H5ModelSnapshot:
    schema_version: Literal["h5-model-snapshot-v1"]
    parameter_blocks: tuple[ModelParameterBlock, ...]
    reconstruction_records: tuple[FactorReconstructionRecord, ...]
    shared_groups: tuple[SharedParameterGroup, ...]
    objective_schema_sha256: str = field(init=False)
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != "h5-model-snapshot-v1":
            raise ValueError("unsupported model snapshot schema")
        _require_exact_tuple(self.parameter_blocks, "parameter_blocks")
        _require_exact_tuple(self.reconstruction_records, "reconstruction_records")
        _require_exact_tuple(self.shared_groups, "shared_groups")
        blocks = tuple(
            _clone_model_block(value, f"parameter_blocks[{index}]")
            for index, value in enumerate(self.parameter_blocks)
        )
        records = tuple(
            _clone_reconstruction(value, f"reconstruction_records[{index}]")
            for index, value in enumerate(self.reconstruction_records)
        )
        groups = tuple(
            _clone_shared_group(value, f"shared_groups[{index}]")
            for index, value in enumerate(self.shared_groups)
        )
        if tuple(value.block_id for value in blocks) != H5_MODEL_BLOCK_UNIVERSE:
            raise ValueError("model blocks must equal the H5 universe order")
        if tuple((value.factor_id, value.bindings) for value in records) != H5_RECONSTRUCTION_ROWS:
            raise ValueError("reconstruction records must equal the H5 schema")
        if tuple((value.group_id, value.source, value.consumers) for value in groups) != H5_SHARED_PARAMETER_GROUP_ROWS:
            raise ValueError("shared groups must equal the H5 schema")
        object.__setattr__(self, "parameter_blocks", blocks)
        object.__setattr__(self, "reconstruction_records", records)
        object.__setattr__(self, "shared_groups", groups)
        object.__setattr__(self, "objective_schema_sha256", H5_OBJECTIVE_SCHEMA_SHA256)
        object.__setattr__(
            self,
            "state_sha256",
            _domain_hash(H5_MODEL_SNAPSHOT_DOMAIN, canonical_h5_model_snapshot_bytes(self)),
        )


def _clone_model(value: H5ModelSnapshot, name: str) -> H5ModelSnapshot:
    if not isinstance(value, H5ModelSnapshot):
        raise ValueError(f"{name} must be an H5ModelSnapshot")
    return H5ModelSnapshot(
        value.schema_version,
        value.parameter_blocks,
        value.reconstruction_records,
        value.shared_groups,
    )


def _update_spec_core(specification: UpdateSpecification) -> object:
    source_row = next(
        coordinate
        for coordinate in specification.initial_recognition.categoricals
        if coordinate.coordinate_id == "q[source_row_a2]"
    )
    return {
        "fixture_id": specification.fixture_id,
        "fixture_schema_version": specification.fixture_schema_version,
        "recognition_family": specification.recognition_family,
        "h1_fixture_id": specification.h1_fixture_id,
        "h1_fixture_sha256": specification.h1_fixture_sha256,
        "factor_input_schema_version": specification.factor_input_schema_version,
        "factor_universe": specification.factor_universe,
        "recognition_coordinate_universe": specification.recognition_coordinate_universe,
        "model_block_universe": specification.model_block_universe,
        "quadrature_orders": specification.quadrature_orders,
        "continuous_recognition": tuple(
            (
                coordinate.coordinate_id,
                coordinate.mean.values[0],
                coordinate.variance.values[0],
            )
            for coordinate in specification.initial_recognition.gaussians
        ),
        "categorical_recognition": tuple(
            (
                coordinate.coordinate_id,
                coordinate.support,
                coordinate.conditioned_on,
                coordinate.probabilities.values,
            )
            for coordinate in specification.initial_recognition.categoricals
        ),
        "model_parameter_blocks": tuple(
            (
                block.block_id,
                tuple((name, _tensor_core(value)) for name, value in block.values),
            )
            for block in specification.initial_model.parameter_blocks
        ),
        "factor_reconstruction": tuple(
            (record.factor_id, record.bindings)
            for record in specification.reconstruction_records
        ),
        "shared_parameter_groups": tuple(
            (group.group_id, group.source, group.consumers)
            for group in specification.shared_groups
        ),
        "source_row_a2": {
            "coordinate_id": source_row.coordinate_id,
            "time": 2,
            "condition": source_row.conditioned_on[0],
            "support": source_row.support,
            "initial_probabilities": source_row.probabilities.values,
        },
    }


@dataclass(frozen=True)
class UpdateSpecification:
    raw_bytes: bytes = field(repr=False)
    fixture_id: Literal["h5-conditional-update-v1"]
    fixture_schema_version: Literal[1]
    recognition_family: Literal["continuous_mean_field_conditional_categorical"]
    h1_fixture_id: Literal["h1-v1"]
    h1_fixture_sha256: str
    factor_input_schema_version: Literal["h5-factor-input-v1"]
    factor_input_schema_sha256: str = field(init=False)
    factor_universe: tuple[str, ...]
    recognition_coordinate_universe: tuple[str, ...]
    model_block_universe: tuple[str, ...]
    quadrature_orders: tuple[Literal[21], Literal[17]]
    reconstruction_records: tuple[FactorReconstructionRecord, ...]
    shared_groups: tuple[SharedParameterGroup, ...]
    initial_recognition: RecognitionSnapshot
    initial_model: H5ModelSnapshot
    canonical_bytes: bytes = field(init=False)
    canonical_sha256: str = field(init=False)
    raw_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        raw_bytes = _copy_bytes(self.raw_bytes, "raw_bytes")
        if (
            self.fixture_id != "h5-conditional-update-v1"
            or type(self.fixture_schema_version) is not int
            or self.fixture_schema_version != 1
            or self.recognition_family != "continuous_mean_field_conditional_categorical"
            or self.h1_fixture_id != "h1-v1"
            or self.factor_input_schema_version != H5_FACTOR_INPUT_SCHEMA_VERSION
        ):
            raise ValueError("unsupported H5 update specification identity")
        _require_sha256(self.h1_fixture_sha256, "h1_fixture_sha256")
        if self.h1_fixture_sha256 != H5_H1_FIXTURE_RAW_SHA256:
            raise ValueError("h1_fixture_sha256 must equal the frozen H1 raw digest")
        if self.factor_universe != H5_FACTOR_UNIVERSE:
            raise ValueError("factor universe does not match H5")
        if self.recognition_coordinate_universe != H5_RECOGNITION_COORDINATE_UNIVERSE:
            raise ValueError("recognition coordinate universe does not match H5")
        if self.model_block_universe != H5_MODEL_BLOCK_UNIVERSE:
            raise ValueError("model block universe does not match H5")
        if self.quadrature_orders != H5_QUADRATURE_ORDERS:
            raise ValueError("quadrature orders do not match H5")
        records = tuple(
            _clone_reconstruction(value, f"reconstruction_records[{index}]")
            for index, value in enumerate(
                _require_exact_tuple(self.reconstruction_records, "reconstruction_records")
            )
        )
        groups = tuple(
            _clone_shared_group(value, f"shared_groups[{index}]")
            for index, value in enumerate(_require_exact_tuple(self.shared_groups, "shared_groups"))
        )
        recognition = _clone_recognition(self.initial_recognition, "initial_recognition")
        model = _clone_model(self.initial_model, "initial_model")
        if model.reconstruction_records != records or model.shared_groups != groups:
            raise ValueError("initial model reconstruction/shared records must match specification")
        object.__setattr__(self, "raw_bytes", raw_bytes)
        object.__setattr__(self, "factor_universe", tuple(self.factor_universe))
        object.__setattr__(
            self, "recognition_coordinate_universe", tuple(self.recognition_coordinate_universe)
        )
        object.__setattr__(self, "model_block_universe", tuple(self.model_block_universe))
        object.__setattr__(self, "quadrature_orders", tuple(self.quadrature_orders))
        object.__setattr__(self, "reconstruction_records", records)
        object.__setattr__(self, "shared_groups", groups)
        object.__setattr__(self, "initial_recognition", recognition)
        object.__setattr__(self, "initial_model", model)
        object.__setattr__(self, "factor_input_schema_sha256", H5_FACTOR_INPUT_SCHEMA_SHA256)
        canonical = _canonical_json_bytes(_update_spec_core(self))
        object.__setattr__(self, "canonical_bytes", canonical)
        object.__setattr__(
            self, "canonical_sha256", _domain_hash(H5_UPDATE_SPEC_DOMAIN, canonical)
        )
        object.__setattr__(self, "raw_sha256", hashlib.sha256(raw_bytes).hexdigest())

    def as_h1_recognition_record(self) -> H1RecognitionFactorRecord:
        gaussian = {
            coordinate.coordinate_id: coordinate for coordinate in self.initial_recognition.gaussians
        }
        categorical = {
            coordinate.coordinate_id: coordinate
            for coordinate in self.initial_recognition.categoricals
        }

        def scalar(identifier: str, field_name: str) -> float:
            value = getattr(gaussian[identifier], field_name)
            return value.values[0]

        initial = GaussianLaw(
            torch.tensor(
                [scalar("q[z0]", "mean"), scalar("q[m0]", "mean")],
                dtype=torch.float64,
            ),
            torch.diag(
                torch.tensor(
                    [scalar("q[z0]", "variance"), scalar("q[m0]", "variance")],
                    dtype=torch.float64,
                )
            ),
        )
        model_kernels = tuple(
            RecognitionModelKernelRecord(
                torch.zeros(size, dtype=torch.float64),
                torch.full((size,), scalar(f"q[m{time}]", "mean"), dtype=torch.float64),
                torch.full(
                    (size,), scalar(f"q[m{time}]", "variance"), dtype=torch.float64
                ),
            )
            for time, size in ((1, 1), (2, 2))
        )
        state_kernels = tuple(
            RecognitionStateKernelRecord(
                torch.zeros(size, dtype=torch.float64),
                torch.zeros(size, dtype=torch.float64),
                torch.full((size,), scalar(f"q[z{time}]", "mean"), dtype=torch.float64),
                torch.full(
                    (size,), scalar(f"q[z{time}]", "variance"), dtype=torch.float64
                ),
            )
            for time, size in ((1, 1), (2, 4))
        )
        model_probabilities = (
            categorical["q[model_source_b1]"].probabilities.to_tensor(),
            categorical["q[model_source_b2]"].probabilities.to_tensor(),
        )
        state_probabilities = (
            categorical["q[state_source_a1_b0]"].probabilities.to_tensor().reshape(1, 1),
            torch.stack(
                (
                    categorical["q[source_row_a2]"].probabilities.to_tensor(),
                    categorical["q[state_source_a2_b1]"].probabilities.to_tensor(),
                )
            ),
        )
        return H1RecognitionFactorRecord(
            initial,
            model_probabilities,
            state_probabilities,
            model_kernels,  # type: ignore[arg-type]
            state_kernels,  # type: ignore[arg-type]
        )


def _reference_hash_core(state: H5ReferenceState) -> tuple[str, ...]:
    return (
        state.h1_fixture_sha256,
        state.update_spec_raw_sha256,
        state.specification.canonical_sha256,
        state.objective_schema_sha256,
        state.factor_input_schema_sha256,
        state.initial_recognition.state_sha256,
        state.initial_model.state_sha256,
        state.initial_optimizer_state.state_sha256,
        state.initial_rng_state.state_sha256,
    )


def canonical_h5_reference_state_bytes(state: H5ReferenceState) -> bytes:
    if not isinstance(state, H5ReferenceState):
        raise ValueError("state must be an H5ReferenceState")
    return _canonical_json_bytes(_reference_hash_core(state))


@dataclass(frozen=True)
class H5ReferenceState:
    schema_version: Literal["h5-reference-state-v1"]
    raw_h1_fixture_bytes: bytes
    raw_update_spec_bytes: bytes
    h1_fixture_sha256: str = field(init=False)
    update_spec_raw_sha256: str = field(init=False)
    objective_schema_sha256: str = field(init=False)
    factor_input_schema_sha256: str = field(init=False)
    specification: UpdateSpecification
    initial_recognition: RecognitionSnapshot
    initial_model: H5ModelSnapshot
    initial_optimizer_state: FrozenByteState
    initial_rng_state: FrozenByteState
    reference_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != "h5-reference-state-v1":
            raise ValueError("unsupported H5 reference-state schema")
        raw_h1 = _copy_bytes(self.raw_h1_fixture_bytes, "raw_h1_fixture_bytes")
        raw_update = _copy_bytes(self.raw_update_spec_bytes, "raw_update_spec_bytes")
        if not isinstance(self.specification, UpdateSpecification):
            raise ValueError("specification must be an UpdateSpecification")
        specification = self.specification
        if raw_update != specification.raw_bytes:
            raise ValueError("raw update-spec bytes must equal specification.raw_bytes")
        h1_sha256 = hashlib.sha256(raw_h1).hexdigest()
        if h1_sha256 != specification.h1_fixture_sha256:
            raise ValueError("H1 raw SHA-256 does not match the update specification")
        recognition = _clone_recognition(self.initial_recognition, "initial_recognition")
        model = _clone_model(self.initial_model, "initial_model")
        if recognition.state_sha256 != specification.initial_recognition.state_sha256:
            raise ValueError("initial recognition must match the update specification")
        if model.state_sha256 != specification.initial_model.state_sha256:
            raise ValueError("initial model must match the update specification")
        optimizer = _clone_typed_byte_state(
            self.initial_optimizer_state,
            expected_schema="h5-no-optimizer-v1",
            name="initial optimizer state",
        )
        rng = _clone_typed_byte_state(
            self.initial_rng_state,
            expected_schema="h5-deterministic-rng-v1",
            name="initial RNG state",
        )
        if optimizer.payload != b'{"kind":"none"}':
            raise ValueError("initial optimizer state must be h5-no-optimizer-v1")
        if rng.payload != b'{"algorithm":"none","counter":0}':
            raise ValueError("initial RNG state must be deterministic counter zero")
        object.__setattr__(self, "raw_h1_fixture_bytes", raw_h1)
        object.__setattr__(self, "raw_update_spec_bytes", raw_update)
        object.__setattr__(self, "h1_fixture_sha256", h1_sha256)
        object.__setattr__(self, "update_spec_raw_sha256", hashlib.sha256(raw_update).hexdigest())
        object.__setattr__(self, "objective_schema_sha256", H5_OBJECTIVE_SCHEMA_SHA256)
        object.__setattr__(self, "factor_input_schema_sha256", H5_FACTOR_INPUT_SCHEMA_SHA256)
        object.__setattr__(self, "initial_recognition", recognition)
        object.__setattr__(self, "initial_model", model)
        object.__setattr__(self, "initial_optimizer_state", optimizer)
        object.__setattr__(self, "initial_rng_state", rng)
        object.__setattr__(
            self,
            "reference_sha256",
            _domain_hash(H5_REFERENCE_STATE_DOMAIN, canonical_h5_reference_state_bytes(self)),
        )


def canonical_h5_live_state_bytes(state: H5LiveState) -> bytes:
    if not isinstance(state, H5LiveState):
        raise ValueError("state must be an H5LiveState")
    return _canonical_json_bytes(
        (
            state.schema_version,
            state.recognition.state_sha256,
            state.model.state_sha256,
            state.optimizer_state.state_sha256,
            state.rng_state.state_sha256,
        )
    )


@dataclass(frozen=True)
class H5LiveState:
    schema_version: Literal["h5-live-state-v1"]
    recognition: RecognitionSnapshot
    model: H5ModelSnapshot
    optimizer_state: FrozenByteState
    rng_state: FrozenByteState
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != "h5-live-state-v1":
            raise ValueError("unsupported H5 live-state schema")
        object.__setattr__(self, "recognition", _clone_recognition(self.recognition, "recognition"))
        object.__setattr__(self, "model", _clone_model(self.model, "model"))
        object.__setattr__(
            self,
            "optimizer_state",
            _clone_typed_byte_state(
                self.optimizer_state,
                expected_schema="h5-no-optimizer-v1",
                name="optimizer state",
            ),
        )
        object.__setattr__(
            self,
            "rng_state",
            _clone_typed_byte_state(
                self.rng_state,
                expected_schema="h5-deterministic-rng-v1",
                name="RNG state",
            ),
        )
        object.__setattr__(
            self,
            "state_sha256",
            _domain_hash(H5_LIVE_STATE_DOMAIN, canonical_h5_live_state_bytes(self)),
        )


class UpdateLabel(str, Enum):
    EXACT_COORDINATE = "exact_coordinate"
    VALID_MM = "valid_mm"
    GENERALIZED_EM = "generalized_em"
    NATURAL_GRADIENT_PROPOSAL = "natural_gradient_proposal"
    SGD_PROPOSAL = "sgd_proposal"
    ADAM_PROPOSAL = "adam_proposal"
    TRUNCATED_ITERATION = "truncated_iteration"


class H5UpdateRule(str, Enum):
    EXACT_Z0 = "exact_z0"
    EXACT_SOURCE_ROW_A2 = "exact_source_row_a2"
    EXACT_STATE_TRANSITION_2_M = "exact_state_transition_2_m"
    GENERALIZED_EM_EMISSION_1 = "generalized_em_emission_1"
    NATURAL_GRADIENT_Z1 = "natural_gradient_z1"


H5_RULE_CONTRACTS = MappingProxyType(
    {
        H5UpdateRule.EXACT_Z0: (
            UpdateLabel.EXACT_COORDINATE,
            ("q[z0]",),
            (),
            (1.0,),
        ),
        H5UpdateRule.EXACT_SOURCE_ROW_A2: (
            UpdateLabel.EXACT_COORDINATE,
            ("q[source_row_a2]",),
            (),
            (1.0,),
        ),
        H5UpdateRule.EXACT_STATE_TRANSITION_2_M: (
            UpdateLabel.EXACT_COORDINATE,
            (),
            ("theta[state_transition_2]",),
            (1.0,),
        ),
        H5UpdateRule.GENERALIZED_EM_EMISSION_1: (
            UpdateLabel.GENERALIZED_EM,
            (),
            ("theta[emission_1]",),
            (
                1.0,
                0.5,
                0.25,
                0.125,
                0.0625,
                0.03125,
                0.015625,
                0.0078125,
                0.00390625,
                0.001953125,
                0.0009765625,
            ),
        ),
        H5UpdateRule.NATURAL_GRADIENT_Z1: (
            UpdateLabel.NATURAL_GRADIENT_PROPOSAL,
            ("q[z1]",),
            (),
            (64.0,),
        ),
    }
)


def _request_core(request: UpdateRequest) -> object:
    return {
        "schema_version": request.schema_version,
        "request_id": request.request_id,
        "rule": request.rule,
        "requested_label": request.requested_label,
        "variables": request.variables,
        "parameters": request.parameters,
        "damping_schedule": request.damping_schedule,
    }


def canonical_h5_update_request_bytes(request: UpdateRequest) -> bytes:
    if not isinstance(request, UpdateRequest):
        raise ValueError("request must be an UpdateRequest")
    return _canonical_json_bytes(_request_core(request))


@dataclass(frozen=True)
class UpdateRequest:
    schema_version: Literal["h5-update-request-v1"]
    request_id: str
    rule: H5UpdateRule
    requested_label: UpdateLabel
    variables: tuple[str, ...]
    parameters: tuple[str, ...]
    damping_schedule: tuple[float, ...]
    request_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != "h5-update-request-v1":
            raise ValueError("unsupported H5 update-request schema")
        _require_nonempty_string(self.request_id, "request_id")
        if not isinstance(self.rule, H5UpdateRule):
            raise ValueError("rule must be an H5UpdateRule")
        if not isinstance(self.requested_label, UpdateLabel):
            raise ValueError("requested_label must be an UpdateLabel")
        _require_exact_tuple(self.variables, "variables")
        _require_exact_tuple(self.parameters, "parameters")
        _require_exact_tuple(self.damping_schedule, "damping_schedule")
        if any(type(value) is not str or not value for value in self.variables + self.parameters):
            raise ValueError("active variables and parameters must be nonempty strings")
        schedule = tuple(
            _finite_float(value, f"damping_schedule[{index}]")
            for index, value in enumerate(self.damping_schedule)
        )
        contract = H5_RULE_CONTRACTS[self.rule]
        if (self.requested_label, self.variables, self.parameters, schedule) != contract:
            raise ValueError("update request does not match its exact H5 rule contract")
        object.__setattr__(self, "variables", tuple(self.variables))
        object.__setattr__(self, "parameters", tuple(self.parameters))
        object.__setattr__(self, "damping_schedule", schedule)
        object.__setattr__(
            self,
            "request_sha256",
            _domain_hash(H5_UPDATE_REQUEST_DOMAIN, canonical_h5_update_request_bytes(self)),
        )


def _candidate_core(candidate: H5CandidateSnapshot) -> object:
    return {
        "schema_version": candidate.schema_version,
        "rule": candidate.rule,
        "request_sha256": candidate.request_sha256,
        "producer_label": candidate.producer_label,
        "variables": candidate.variables,
        "parameters": candidate.parameters,
        "damping": candidate.damping,
        "numerical_diagnostics": candidate.numerical_diagnostics,
        "recognition": _recognition_core(candidate.recognition),
        "model": _model_core(candidate.model),
    }


def canonical_h5_candidate_snapshot_bytes(candidate: H5CandidateSnapshot) -> bytes:
    if not isinstance(candidate, H5CandidateSnapshot):
        raise ValueError("candidate must be an H5CandidateSnapshot")
    return _canonical_json_bytes(_candidate_core(candidate))


@dataclass(frozen=True)
class H5CandidateSnapshot:
    schema_version: Literal["h5-candidate-v1"]
    rule: H5UpdateRule
    request_sha256: str
    producer_label: UpdateLabel
    variables: tuple[str, ...]
    parameters: tuple[str, ...]
    damping: float
    numerical_diagnostics: tuple[tuple[str, float], ...]
    recognition: RecognitionSnapshot
    model: H5ModelSnapshot
    candidate_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != "h5-candidate-v1":
            raise ValueError("unsupported H5 candidate schema")
        if not isinstance(self.rule, H5UpdateRule):
            raise ValueError("rule must be an H5UpdateRule")
        _require_sha256(self.request_sha256, "request_sha256")
        if not isinstance(self.producer_label, UpdateLabel):
            raise ValueError("producer_label must be an UpdateLabel")
        _require_exact_tuple(self.variables, "variables")
        _require_exact_tuple(self.parameters, "parameters")
        damping = _finite_float(self.damping, "damping")
        contract_label, contract_variables, contract_parameters, contract_schedule = (
            H5_RULE_CONTRACTS[self.rule]
        )
        if (
            self.producer_label != contract_label
            or self.variables != contract_variables
            or self.parameters != contract_parameters
            or damping not in contract_schedule
        ):
            raise ValueError("candidate provenance does not match its H5 rule")
        _require_exact_tuple(self.numerical_diagnostics, "numerical_diagnostics")
        diagnostics: list[tuple[str, float]] = []
        for index, item in enumerate(self.numerical_diagnostics):
            if type(item) is not tuple or len(item) != 2 or type(item[0]) is not str:
                raise ValueError(f"numerical_diagnostics[{index}] must be a named float pair")
            diagnostics.append((item[0], _finite_float(item[1], f"diagnostic[{index}]")))
        if self.rule is H5UpdateRule.EXACT_STATE_TRANSITION_2_M:
            if len(diagnostics) != 1 or diagnostics[0][0] != "G_condition_number" or diagnostics[0][1] < 1.0:
                raise ValueError("exact M candidate requires one G_condition_number diagnostic")
        elif diagnostics:
            raise ValueError("only the exact M candidate may carry diagnostics")
        object.__setattr__(self, "variables", tuple(self.variables))
        object.__setattr__(self, "parameters", tuple(self.parameters))
        object.__setattr__(self, "damping", damping)
        object.__setattr__(self, "numerical_diagnostics", tuple(diagnostics))
        object.__setattr__(self, "recognition", _clone_recognition(self.recognition, "recognition"))
        object.__setattr__(self, "model", _clone_model(self.model, "model"))
        object.__setattr__(
            self,
            "candidate_sha256",
            _domain_hash(H5_CANDIDATE_DOMAIN, canonical_h5_candidate_snapshot_bytes(self)),
        )


def canonical_h5_semantic_state_bytes(
    recognition: RecognitionSnapshot, model: H5ModelSnapshot
) -> bytes:
    recognition_bytes = canonical_h5_recognition_snapshot_bytes(recognition)
    model_bytes = canonical_h5_model_snapshot_bytes(model)
    return (
        H5_SEMANTIC_STATE_DOMAIN
        + struct.pack(">Q", len(recognition_bytes))
        + recognition_bytes
        + struct.pack(">Q", len(model_bytes))
        + model_bytes
    )


def h5_semantic_state_sha256(
    recognition: RecognitionSnapshot, model: H5ModelSnapshot
) -> str:
    return hashlib.sha256(canonical_h5_semantic_state_bytes(recognition, model)).hexdigest()


def initial_live(reference: H5ReferenceState) -> H5LiveState:
    if not isinstance(reference, H5ReferenceState):
        raise ValueError("reference must be an H5ReferenceState")
    return H5LiveState(
        "h5-live-state-v1",
        reference.initial_recognition,
        reference.initial_model,
        reference.initial_optimizer_state,
        reference.initial_rng_state,
    )


__all__ = [
    "CategoricalRecognitionCoordinate",
    "FactorReconstructionRecord",
    "FrozenByteState",
    "FrozenTensorValue",
    "GaussianRecognitionCoordinate",
    "H5CandidateSnapshot",
    "H5LiveState",
    "H5ModelSnapshot",
    "H5ReferenceState",
    "H5UpdateRule",
    "H5_RULE_CONTRACTS",
    "ModelParameterBlock",
    "RecognitionSnapshot",
    "SharedParameterGroup",
    "UpdateLabel",
    "UpdateRequest",
    "UpdateSpecification",
    "canonical_h5_candidate_snapshot_bytes",
    "canonical_h5_live_state_bytes",
    "canonical_h5_model_snapshot_bytes",
    "canonical_h5_recognition_snapshot_bytes",
    "canonical_h5_reference_state_bytes",
    "canonical_h5_semantic_state_bytes",
    "canonical_h5_update_request_bytes",
    "h5_semantic_state_sha256",
    "initial_live",
]
