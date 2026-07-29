"""Independent 100-decimal oracle for H7 frame covariance.

Raw JSON bytes and immutable Task-6 budget records cross the numerical
boundary.  The arithmetic implementation does not import or call the
production objective, PyTorch, NumPy, or a production-facing verification
module.  The separately named wiring function accepts its Task-5 evaluator as
an injected callable, and every JSON number remains a decimal string until the
arithmetic site that consumes it.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import struct
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence, cast

import mpmath as mp

from verification.mp_oracles.h7_budget_protocol import (
    H7BackwardBudgetAggregate,
    H7BackwardOperandInput,
    H7BackwardResidualRecord,
    H7BoundBudget,
    H7BudgetFormula,
    H7OperandRecord,
    build_h7_backward_records,
    build_h7_budget,
)


H7_COMPLETE_LOCAL_TERM_IDS: tuple[str, ...] = (
    "expected_log_emission[1]",
    "expected_log_emission[2]",
    "model_source_kl[1]",
    "state_source_kl[1]",
    "model_transition_kl[1]",
    "state_transition_kl[1]",
    "model_source_kl[2]",
    "state_source_kl[2]",
    "model_transition_kl[2]",
    "state_transition_kl[2]",
    "joint_recognition_entropy",
)
_TASK5_COMPLETE_LOCAL_INVARIANT_ID = "complete_local_elbo"
_TASK5_COMPLETE_MONOLITHIC_INVARIANT_ID = "complete_monolithic_elbo"
_TASK5_POINTWISE_P_SHIFT_INVARIANT_ID = "complete_pointwise_p_density_shift"
_TASK5_POINTWISE_Q_SHIFT_INVARIANT_ID = "complete_pointwise_q_density_shift"
_TASK5_POINTWISE_LOG_RATIO_INVARIANT_ID = "complete_pointwise_log_ratio"
_TASK5_ENTROPY_SHIFT_INVARIANT_ID = "joint_recognition_entropy_shift"
_TASK5_SCALAR_EVIDENCE_INVARIANT_ID = "scalar_log_evidence_and_elbo_kl_identity"
_TASK5_SCALAR_POSTERIOR_KL_INVARIANT_ID = "scalar_posterior_kl_invariance"
H7_REQUIRED_TRIAL_IDS: tuple[str, ...] = (
    "scalar-base-transformed",
    "scalar-internal-transformed",
    "matrix-identity-base-transformed",
    "matrix-identity-internal-transformed",
    "matrix-nonidentity-base-transformed",
    "matrix-nonidentity-internal-transformed",
    "matrix-fixed-decoder-centered-stabilizer",
    "matrix-fixed-decoder-outside-stabilizer",
)
_TRIAL_GEOMETRY_CONTRACT = MappingProxyType(
    {
        "scalar-base-transformed": ("h1_v1", "transform"),
        "scalar-internal-transformed": ("h1_v1", "transform"),
        "matrix-identity-base-transformed": ("identity", "transform"),
        "matrix-identity-internal-transformed": ("identity", "transform"),
        "matrix-nonidentity-base-transformed": ("nonidentity", "transform"),
        "matrix-nonidentity-internal-transformed": ("nonidentity", "transform"),
        "matrix-fixed-decoder-centered-stabilizer": ("nonidentity", "fixed"),
        "matrix-fixed-decoder-outside-stabilizer": ("nonidentity", "fixed"),
    }
)
_MATRIX_RECOGNITION_FAMILIES = (
    "structured_full_block",
    "factorized_diagonal_within_fiber",
)
_MATRIX_EVIDENCE_NOT_APPLICABLE = (
    "analytic evidence/posterior KL is not applicable to the nonconjugate "
    "h7-v1 categorical-emission matrix fixture"
)
_H1_RAW_SHA256 = "388e38cc8c16d8b5e2c61919c1e712a134d88fb0bbd8ec1f2939b9859c9a583b"
_H7_RAW_SHA256 = "d2ed126c3deab3eafc7b94f81f13152be63eb854e3e62e03f1494dea163666d4"
_MATRIX_PROBE_RAW_SHA256 = (
    "4857af296e84a33f47964c3bca65e0d42967009aa5c79a52bcc98d6db04382c6"
)
_MATRIX_PROBE_SET_SHA256 = (
    "f002618a32270846c83fedf9888bc06a01d755019edc6421526aee33f89fb42f"
)
# These identities are deliberately unmeasured until the frozen scalar and
# precision tables exist and the authorized Task-6 calibration run is performed.
_SCALAR_PROBE_RAW_SHA256: str | None = None
_SCALAR_PROBE_SET_SHA256: str | None = None
_PRECISION_OPERAND_RAW_SHA256: str | None = None
_PRECISION_OPERAND_SET_SHA256: str | None = None
_ORACLE_INVENTORY_SHA256: str | None = None
_PRECISION_OPERAND_COUNT = 192
_PRECISION_TABLE_SCHEMA = "h7-mp-precision-operands-v2"
_PRECISION_SOURCE_CONTRACT = (
    "task5-production-covariance-and-precision-v2"
)
_BINARY64_TEXT_POLICY = "python-repr-binary64-roundtrip-v1"
_COVARIANCE_VALUES_DOMAIN = (
    "vfe4.h7.mp-serialized-covariance-values.v2"
)
_PRECISION_VALUES_DOMAIN = "vfe4.h7.mp-serialized-precision-values.v2"
_PRECISION_ROW_DOMAIN = "vfe4.h7.mp-serialized-precision-operand.v2"
_PRECISION_SET_DOMAIN = "vfe4.h7.mp-serialized-precision-set.v2"
_SCALAR_PRECISION_IDS = (
    "scalar.p.initial_joint",
    "scalar.q.initial_joint",
    "scalar.p.p.model.receiver_1.source_0.receiver_offset",
    "scalar.p.p.state.receiver_1.source_0.receiver_offset",
    "scalar.p.p.model.receiver_2.source_0.receiver_offset",
    "scalar.p.p.state.receiver_2.source_0.receiver_offset",
    "scalar.p.p.model.receiver_2.source_1.receiver_offset",
    "scalar.p.p.state.receiver_2.source_1.receiver_offset",
    "scalar.q_model.q.model.receiver_1.source_0.receiver_offset",
    "scalar.q_model.q.model.receiver_2.source_0.receiver_offset",
    "scalar.q_model.q.model.receiver_2.source_1.receiver_offset",
    "scalar.q_state.q.state.receiver_1.a_0.b_0.receiver_offset",
    "scalar.q_state.q.state.receiver_2.a_0.b_0.receiver_offset",
    "scalar.q_state.q.state.receiver_2.a_1.b_0.receiver_offset",
    "scalar.q_state.q.state.receiver_2.a_0.b_1.receiver_offset",
    "scalar.q_state.q.state.receiver_2.a_1.b_1.receiver_offset",
    "scalar.q.global[h1-path-0:a0-b0]",
    "scalar.q.global[h1-path-1:a1-b0]",
    "scalar.q.global[h1-path-2:a0-b1]",
    "scalar.q.global[h1-path-3:a1-b1]",
    "scalar.p.global[h1-path-0:a0-b0]",
    "scalar.p.global[h1-path-1:a1-b0]",
    "scalar.p.global[h1-path-2:a0-b1]",
    "scalar.p.global[h1-path-3:a1-b1]",
)
_MATRIX_PRECISION_SUFFIXES = (
    "p.initial_joint",
    "q.initial_joint",
    "p.p.model.receiver_1.receiver_offset",
    "p.p.state.receiver_1.receiver_offset",
    "p.p.model.receiver_2.receiver_offset",
    "p.p.state.receiver_2.receiver_offset",
    "q_model.q.{family}.model.receiver_1.receiver_offset",
    "q_model.q.{family}.model.receiver_2.receiver_offset",
    "q_state.q.{family}.state.receiver_1.receiver_offset",
    "q_state.q.{family}.state.receiver_2.receiver_offset",
    "q.global[matrix-singleton-path]",
    "p.global[matrix-singleton-path]",
)
_H1_PATH_IDS = (
    "h1-path-0:a0-b0",
    "h1-path-1:a1-b0",
    "h1-path-2:a0-b1",
    "h1-path-3:a1-b1",
)
_MATRIX_ACTION_SHA256 = {
    "diagonal": "581dff7f0496fd0b51235979269d57c61725fdf7ba73249c81e5563fa6ef810f",
    "fixed_decoder_stabilizer": (
        "a07752d82866285d3258832aa8d000f5b639b1fcb5b25b8774b6d6ca756cc78c"
    ),
    "internal": "bd1cfcba5fbe54c2a8ffe3ded38e10a6aead260b69f1efdb31e72e22a17263a2",
}
_PROBE_COMPONENTS: tuple[tuple[str, str, int, str], ...] = (
    ("p.initial_joint", "initial", 4, "initial_joint"),
    ("p.model.receiver_1", "model:1<-0", 2, "receiver_model"),
    ("p.model.receiver_2", "model:2<-1", 2, "receiver_model"),
    ("p.state.receiver_1", "state:1<-0", 2, "receiver_state"),
    ("p.state.receiver_2", "state:2<-1", 2, "receiver_state"),
    ("q.structured.initial_joint", "initial", 4, "initial_joint"),
    ("q.structured.model.receiver_1", "model:1<-0", 2, "receiver_model"),
    ("q.structured.model.receiver_2", "model:2<-1", 2, "receiver_model"),
    ("q.structured.state.receiver_1", "state:1<-0", 2, "receiver_state"),
    ("q.structured.state.receiver_2", "state:2<-1", 2, "receiver_state"),
    ("q.factorized.initial_joint", "initial", 4, "initial_joint"),
    ("q.factorized.model.receiver_1", "model:1<-0", 2, "receiver_model"),
    ("q.factorized.model.receiver_2", "model:2<-1", 2, "receiver_model"),
    ("q.factorized.state.receiver_1", "state:1<-0", 2, "receiver_state"),
    ("q.factorized.state.receiver_2", "state:2<-1", 2, "receiver_state"),
    ("p.global", "matrix-singleton-path", 12, "global"),
    ("q.structured.global", "matrix-singleton-path", 12, "global"),
    ("q.factorized.global", "matrix-singleton-path", 12, "global"),
)
_PROBE_ANCHOR_SHA256 = {
    "p.initial_joint": "369a9e44db283110ab29d80d74c9229ccc5af8cb1188749bdc99fe2b59c404be",
    "p.model.receiver_1": "18e0a1cad43ceb6b40af6c3b82dec813b53ab2dba69e2704f9a20c4dd3c916b6",
    "p.model.receiver_2": "5d8a7d7f1614e9db9051b3ead02b2e010533122443b09786170141efc038f605",
    "p.state.receiver_1": "8df4b33d0a5d7f6c8b5283a0faa7831f8b76b91d261bdcf0ff489077404d3242",
    "p.state.receiver_2": "304142e102bf246806136dadec426d9dc101fadc4e17cac5302c2ee7b92fc62f",
    "q.structured.initial_joint": (
        "0c038a63d66efd2f4b43c6e1fc1e07a17a4aa797c7307575c54b0d85063e57a0"
    ),
    "q.structured.model.receiver_1": (
        "e52ccf722ac14fa8b893eea3f4d609d0d8e48e490a38ac11ea19835d97d1c347"
    ),
    "q.structured.model.receiver_2": (
        "5d3ae9deaf8ec117d2ecf7c8a5be6f4a5f70234a3509d12e339bff7800e80156"
    ),
    "q.structured.state.receiver_1": (
        "ec31675b253cdbc57dce45a544a3c68bb55d4c95178c1d097f82e08edf5dd472"
    ),
    "q.structured.state.receiver_2": (
        "98d892147fa8ff6c6742e79604db17807196c3df024d170abe95a99880ec2327"
    ),
    "q.factorized.initial_joint": (
        "e28edcd9d0d782333cd3384afdded5df1c67103cee0bd369d9d72ad87df2de21"
    ),
    "q.factorized.model.receiver_1": (
        "2f665dfae1f2f8c6247e3c080f41092ab1dedc2cf47efd1cea2582cf820425ca"
    ),
    "q.factorized.model.receiver_2": (
        "75b918625f0747b773ff757f3196a1eb91b56aac5c84eb6cc16ad0ad014dc3f6"
    ),
    "q.factorized.state.receiver_1": (
        "34d7dbfc585741e212b79bff8cd69bbb996335e4ba7909b8ec8a82c2bddf3933"
    ),
    "q.factorized.state.receiver_2": (
        "c3939f5248d8f55c6a17213d9cfe71bc7eb4dce17bba2ae40c5e8603b002347f"
    ),
    "p.global": "d88fd278ab372df48542df159096fec3bebd6f59768af849123526e2d537752c",
    "q.structured.global": (
        "c7df941af0f4db1bf4f56cc3bb339b9adfc5b10d40f0e7588e8cefdc5395602d"
    ),
    "q.factorized.global": (
        "4a0a233678e2071d31e3ead8ba2046f9ae289303a4ef6c4e43b36c6f8b18c16c"
    ),
}


class H7OracleInconclusive(ValueError):
    """Fail-closed raw-input/oracle error with an explicit terminal state."""

    status: Literal["INCONCLUSIVE"] = "INCONCLUSIVE"


class _H7ExternalDataError(ValueError):
    """A malformed external byte/schema value, never a programming defect."""


@dataclass(frozen=True)
class MPGaussHermiteRule:
    order: int
    jacobi: mp.matrix
    physicists_nodes: tuple[mp.mpf, ...]
    standard_normal_nodes: tuple[mp.mpf, ...]
    weights: tuple[mp.mpf, ...]


@dataclass(frozen=True)
class MPValueRecord:
    operand_id: str
    shape: tuple[int, ...]
    decimal_values: tuple[str, ...]
    value_sha256: str
    scale: str
    condition_number: str
    normalization: str


@dataclass(frozen=True)
class MPLawValues:
    values: tuple[MPValueRecord, ...]


@dataclass(frozen=True)
class MPSourcePathRecord:
    path_id: str
    a: tuple[int, int]
    b: tuple[int, int]
    q_probability: str
    p_probability: str


@dataclass(frozen=True)
class MPScorerRowRecord:
    bank: Literal["model", "state"]
    receiver_t: int
    source_j: int
    prefix_tokens: tuple[int, ...]
    prefix_term: str
    support: tuple[int, ...]
    mask: tuple[bool, ...]
    original_probability: str
    transformed_probability: str
    original_z_covector: tuple[str, ...]
    transformed_z_covector: tuple[str, ...]
    original_m_covector: tuple[str, ...]
    transformed_m_covector: tuple[str, ...]
    original_raw_score: str
    transformed_raw_score: str


@dataclass(frozen=True)
class MPProbeEvaluationRecord:
    probe_id: str
    component_id: str
    source_id: str
    observations: tuple[tuple[str, str, str, str, str], ...]


@dataclass(frozen=True)
class MPTrialResult:
    trial_id: str
    fixture_id: Literal["h1-v1", "h7-v1"]
    frame_profile: Literal["identity", "nonidentity", "h1_v1"]
    decoder_policy: Literal["transform", "fixed"]
    action_sha256: str
    recognition_families: tuple[str, ...]
    source_paths: tuple[MPSourcePathRecord, ...]
    original: MPLawValues
    transformed: MPLawValues
    recovered: MPLawValues
    backward_records: tuple[H7BackwardResidualRecord, ...]
    backward_bound_budgets: tuple[H7BoundBudget, ...]
    backward_inventory_size: Literal[218] | int
    r_back_max: str
    scalar_items: tuple[tuple[str, str], ...]
    status_items: tuple[tuple[str, str], ...]
    scorer_rows: tuple[MPScorerRowRecord, ...]
    probe_evaluations: tuple[MPProbeEvaluationRecord, ...]

    def __post_init__(self) -> None:
        original_ids = tuple(item.operand_id for item in self.original.values)
        transformed_ids = tuple(item.operand_id for item in self.transformed.values)
        recovered_ids = tuple(item.operand_id for item in self.recovered.values)
        record_ids = tuple(item.operand_id for item in self.backward_records)
        if (
            self.backward_inventory_size != len(self.original.values)
            or self.backward_inventory_size != len(self.transformed.values)
            or self.backward_inventory_size != len(self.recovered.values)
            or self.backward_inventory_size != len(self.backward_records)
            or self.backward_inventory_size != len(self.backward_bound_budgets)
            or (self.fixture_id == "h7-v1" and self.backward_inventory_size != 218)
            or any(
                type(item) is not H7BackwardResidualRecord
                for item in self.backward_records
            )
            or any(
                type(item) is not H7BoundBudget for item in self.backward_bound_budgets
            )
            or original_ids != transformed_ids
            or original_ids != recovered_ids
            or original_ids != record_ids
            or type(self.action_sha256) is not str
            or len(self.action_sha256) != 64
            or any(
                character not in "0123456789abcdef" for character in self.action_sha256
            )
            or _TRIAL_GEOMETRY_CONTRACT.get(self.trial_id)
            != (self.frame_profile, self.decoder_policy)
            or any(
                record.original_sha256 != original.value_sha256
                or record.transformed_sha256 != transformed.value_sha256
                or record.recovered_sha256 != recovered.value_sha256
                or record.budget.budget_sha256 != bound.budget.budget_sha256
                or bound.formula.category != "backward"
                for original, transformed, recovered, record, bound in zip(
                    self.original.values,
                    self.transformed.values,
                    self.recovered.values,
                    self.backward_records,
                    self.backward_bound_budgets,
                    strict=True,
                )
            )
        ):
            raise ValueError("trial backward inventory is incomplete or unbound")

    @property
    def scalar_map(self) -> Mapping[str, str]:
        return MappingProxyType(dict(self.scalar_items))

    @property
    def status_map(self) -> Mapping[str, str]:
        return MappingProxyType(dict(self.status_items))


@dataclass(frozen=True)
class H7MPOracleResult:
    status: Literal["EVIDENCE_VERIFIED", "INCONCLUSIVE"]
    open_obligations: tuple[str, ...]
    decimal_precision: int
    gauss_hermite_orders: tuple[int, int]
    raw_fixture_sha256: tuple[str, ...]
    h1_source_paths: tuple[MPSourcePathRecord, ...]
    h7_source_path: MPSourcePathRecord
    trials: tuple[MPTrialResult, ...]
    inventory_sha256: str | None


@dataclass(frozen=True)
class MPTask5OracleComparison:
    value_id: str
    production_value: float
    oracle_value: str
    absolute_delta: str


@dataclass(frozen=True)
class MPTask5WiringResult:
    trial_id: str
    fixture_id: Literal["h1-v1", "h7-v1"]
    production_evaluation: object
    comparisons: tuple[MPTask5OracleComparison, ...]
    status_comparisons: tuple[tuple[str, str, str], ...]


_MPTransformKind = Literal[
    "left",
    "covariance",
    "precision",
    "information",
    "receiver_source",
    "decoder",
]
_MPLeafCategory = Literal[
    "vector",
    "information",
    "offset",
    "decoder",
    "covariance",
    "precision",
    "second_moment",
    "map",
]


@dataclass(frozen=True)
class _MPInventoryOperand:
    record: MPValueRecord
    category: _MPLeafCategory
    transform_kind: _MPTransformKind
    action_indices: tuple[int, ...]


def _expected_precision_row_identities() -> tuple[
    tuple[str, str, Literal["owned_component", "assembled_global"]],
    ...,
]:
    rows: list[
        tuple[str, str, Literal["owned_component", "assembled_global"]]
    ] = []
    for trial_id in H7_REQUIRED_TRIAL_IDS[:2]:
        rows.extend(
            (
                trial_id,
                gaussian_id,
                "owned_component" if index < 16 else "assembled_global",
            )
            for index, gaussian_id in enumerate(_SCALAR_PRECISION_IDS)
        )
    for trial_id in H7_REQUIRED_TRIAL_IDS[2:]:
        for family in ("structured", "factorized"):
            gaussian_ids = tuple(
                f"{family}.{suffix.format(family=family)}"
                for suffix in _MATRIX_PRECISION_SUFFIXES
            )
            rows.extend(
                (
                    trial_id,
                    gaussian_id,
                    "owned_component" if index < 10 else "assembled_global",
                )
                for index, gaussian_id in enumerate(gaussian_ids)
            )
    if len(rows) != _PRECISION_OPERAND_COUNT:
        raise AssertionError("internal H7 precision inventory changed")
    return tuple(rows)


@dataclass
class _MPPrecisionOperandSource:
    records: tuple[Mapping[str, Any], ...]
    precision_set_sha256: str
    index: int = 0

    def consume(
        self,
        *,
        trial_id: str,
        gaussian_id: str,
        covariance: mp.matrix,
    ) -> mp.matrix:
        if self.index >= len(self.records):
            raise _H7ExternalDataError("serialized precision inventory is incomplete")
        location = f"precision_operand_table.records[{self.index}]"
        record = self.records[self.index]
        expected_shape = (covariance.rows, covariance.cols)
        raw_shape = _exact_sequence(record["shape"], 2, f"{location}.shape")
        shape = tuple(_as_int(item) for item in raw_shape)
        if (
            record["row_index"] != str(self.index)
            or record["trial_id"] != trial_id
            or record["gaussian_id"] != gaussian_id
            or shape != expected_shape
        ):
            raise _H7ExternalDataError(
                f"{location} changed precision identity/order/shape"
            )
        serialized_covariance = _matrix(record["covariance_values"])
        covariance_error = _max_abs(serialized_covariance - covariance)
        covariance_scale = max(
            _matrix_scale(serialized_covariance),
            _matrix_scale(covariance),
        )
        try:
            covariance_condition = max(
                _condition_spd(serialized_covariance),
                _condition_spd(covariance),
            )
        except ValueError as error:
            raise _H7ExternalDataError(
                f"{location} covariance is not positive definite"
            ) from error
        covariance_allowance = (
            256
            * (mp.mpf(2) ** -52)
            * max(mp.mpf("1"), covariance_condition)
            * covariance_scale
        )
        if covariance_error > covariance_allowance:
            raise _H7ExternalDataError(
                f"{location} serialized covariance disagrees numerically "
                "with the independent covariance"
            )
        precision = _matrix(record["precision_values"])
        _validate_serialized_precision(
            serialized_covariance,
            precision,
            location,
        )
        self.index += 1
        return precision

    def require_complete(self) -> None:
        if self.index != len(self.records):
            raise _H7ExternalDataError(
                "serialized precision inventory has unconsumed records"
            )


@dataclass(frozen=True)
class _Moments:
    mean: mp.matrix
    covariance: mp.matrix


@dataclass(frozen=True)
class _ObjectiveValues:
    scalars_41: tuple[tuple[str, mp.mpf], ...]
    scalars_51: tuple[tuple[str, mp.mpf], ...]
    q_moments: tuple[tuple[str, _Moments], ...]
    p_moments: tuple[tuple[str, _Moments], ...]

    @property
    def values_41(self) -> dict[str, mp.mpf]:
        return dict(self.scalars_41)

    @property
    def values_51(self) -> dict[str, mp.mpf]:
        return dict(self.scalars_51)


def standard_normal_gauss_hermite(order: int) -> MPGaussHermiteRule:
    """Construct a normalized standard-normal rule from its Jacobi matrix."""

    if type(order) is not int or order <= 0:
        raise ValueError("Gauss-Hermite order must be a positive integer")
    zero = mp.mpf("0")
    jacobi = mp.matrix(order, order)
    for row in range(order):
        for column in range(order):
            jacobi[row, column] = zero
    for k in range(1, order):
        off_diagonal = mp.sqrt(mp.mpf(k) / 2)
        jacobi[k - 1, k] = off_diagonal
        jacobi[k, k - 1] = off_diagonal
    eigenvalues, eigenvectors = mp.eigsy(jacobi)
    physicists = tuple(mp.mpf(eigenvalues[index]) for index in range(order))
    nodes = tuple(mp.sqrt(2) * item for item in physicists)
    weights = tuple(mp.mpf(eigenvectors[0, index]) ** 2 for index in range(order))
    return MPGaussHermiteRule(
        order=order,
        jacobi=jacobi,
        physicists_nodes=physicists,
        standard_normal_nodes=nodes,
        weights=weights,
    )


def evaluate_h7_from_raw_bytes(
    h1_fixture_bytes: bytes,
    h7_fixture_bytes: bytes,
    h7_density_probe_bytes: bytes,
    h1_scalar_probe_bytes: bytes | None = None,
    precision_operand_bytes: bytes | None = None,
) -> H7MPOracleResult:
    """Validate raw inputs and evaluate only a fully preregistered inventory.

    The scalar-probe table, serialized precision table, and final Task-6
    inventory digest are intentionally unmeasured in source. Until all are
    frozen, this entry point validates every supplied byte record and returns
    ``INCONCLUSIVE`` without regenerating an operand or treating an observed
    digest as expected.
    """

    for name, value in (
        ("h1_fixture_bytes", h1_fixture_bytes),
        ("h7_fixture_bytes", h7_fixture_bytes),
        ("h7_density_probe_bytes", h7_density_probe_bytes),
    ):
        if type(value) is not bytes or not value:
            raise H7OracleInconclusive(f"{name} must be nonempty exact bytes")
    if h1_scalar_probe_bytes is not None and (
        type(h1_scalar_probe_bytes) is not bytes or not h1_scalar_probe_bytes
    ):
        raise H7OracleInconclusive(
            "h1_scalar_probe_bytes must be absent or nonempty exact bytes"
        )
    if precision_operand_bytes is not None and (
        type(precision_operand_bytes) is not bytes or not precision_operand_bytes
    ):
        raise H7OracleInconclusive(
            "precision_operand_bytes must be absent or nonempty exact bytes"
        )
    previous_precision = mp.mp.dps
    mp.mp.dps = 100
    try:
        h1 = _parse_raw_json(h1_fixture_bytes)
        h7 = _parse_raw_json(h7_fixture_bytes)
        probes = _parse_raw_json(h7_density_probe_bytes)
        _validate_h1_fixture(h1)
        _validate_h7_fixture(h7)
        _validate_matrix_probe_table(probes, h7)
        raw_hashes = (
            hashlib.sha256(h1_fixture_bytes).hexdigest(),
            hashlib.sha256(h7_fixture_bytes).hexdigest(),
            hashlib.sha256(h7_density_probe_bytes).hexdigest(),
        )
        expected_hashes = (
            _H1_RAW_SHA256,
            _H7_RAW_SHA256,
            _MATRIX_PROBE_RAW_SHA256,
        )
        if raw_hashes != expected_hashes:
            raise _H7ExternalDataError(
                "raw H1/H7/matrix-probe bytes changed from the frozen inputs"
            )
        h1_paths = _h1_source_paths(h1)
        h7_path = MPSourcePathRecord(
            path_id="matrix-singleton-path",
            a=(0, 1),
            b=(0, 1),
            q_probability="1.0",
            p_probability="1.0",
        )
        missing_optional_inputs: list[str] = []
        if h1_scalar_probe_bytes is None:
            missing_optional_inputs.extend(
                (
                    "frozen raw scalar density-probe table is missing",
                    "scalar probe raw/set hashes are UNMEASURED",
                )
            )
        if precision_operand_bytes is None:
            missing_optional_inputs.extend(
                (
                    "frozen serialized precision-operand table is missing",
                    "precision operand raw/set hashes are UNMEASURED",
                )
            )
        if missing_optional_inputs:
            return _inconclusive_result(
                raw_hashes,
                h1_paths,
                h7_path,
                *missing_optional_inputs,
                "Task-6 oracle inventory hash is UNMEASURED",
            )
        assert h1_scalar_probe_bytes is not None
        assert precision_operand_bytes is not None
        scalar_probes = _parse_raw_json(h1_scalar_probe_bytes)
        _validate_scalar_probe_table(scalar_probes, h1, h1_paths)
        precision_operands = _parse_raw_json(precision_operand_bytes)
        if _canonical_bytes(precision_operands) + b"\n" != precision_operand_bytes:
            raise _H7ExternalDataError(
                "precision operand bytes are not canonical newline JSON"
            )
        precision_source = _validate_precision_operand_table(precision_operands)
        scalar_raw_sha256 = hashlib.sha256(h1_scalar_probe_bytes).hexdigest()
        precision_raw_sha256 = hashlib.sha256(precision_operand_bytes).hexdigest()
        all_raw_hashes = (
            *raw_hashes,
            scalar_raw_sha256,
            precision_raw_sha256,
        )
        if (
            _SCALAR_PROBE_RAW_SHA256 is None
            or _SCALAR_PROBE_SET_SHA256 is None
            or _PRECISION_OPERAND_RAW_SHA256 is None
            or _PRECISION_OPERAND_SET_SHA256 is None
            or _ORACLE_INVENTORY_SHA256 is None
        ):
            return _inconclusive_result(
                all_raw_hashes,
                h1_paths,
                h7_path,
                "scalar probe raw/set hashes are UNMEASURED",
                "precision operand raw/set hashes are UNMEASURED",
                "Task-6 oracle inventory hash is UNMEASURED",
            )
        if (
            scalar_raw_sha256 != _SCALAR_PROBE_RAW_SHA256
            or scalar_probes["probe_set_sha256"] != _SCALAR_PROBE_SET_SHA256
            or precision_raw_sha256 != _PRECISION_OPERAND_RAW_SHA256
            or precision_source.precision_set_sha256 != _PRECISION_OPERAND_SET_SHA256
        ):
            raise _H7ExternalDataError(
                "raw scalar/precision identities changed from preregistration"
            )
        probe_records = cast(list[dict[str, Any]], probes["records"])
        scalar_probe_records = cast(list[dict[str, Any]], scalar_probes["records"])
        trials = (
            _evaluate_scalar_trial(
                "scalar-base-transformed",
                h1,
                h1_paths,
                ("1.25", "1.25", "1.25"),
                scalar_probe_records,
                precision_source,
            ),
            _evaluate_scalar_trial(
                "scalar-internal-transformed",
                h1,
                h1_paths,
                ("0.8", "1.1", "1.4"),
                scalar_probe_records,
                precision_source,
            ),
            _evaluate_matrix_trial(
                "matrix-identity-base-transformed",
                h7,
                (h7_path,),
                frame_profile="identity",
                action_profile="diagonal",
                decoder_policy="transform",
                probe_records=probe_records,
                precision_source=precision_source,
            ),
            _evaluate_matrix_trial(
                "matrix-identity-internal-transformed",
                h7,
                (h7_path,),
                frame_profile="identity",
                action_profile="internal",
                decoder_policy="transform",
                probe_records=probe_records,
                precision_source=precision_source,
            ),
            _evaluate_matrix_trial(
                "matrix-nonidentity-base-transformed",
                h7,
                (h7_path,),
                frame_profile="nonidentity",
                action_profile="diagonal",
                decoder_policy="transform",
                probe_records=probe_records,
                precision_source=precision_source,
            ),
            _evaluate_matrix_trial(
                "matrix-nonidentity-internal-transformed",
                h7,
                (h7_path,),
                frame_profile="nonidentity",
                action_profile="internal",
                decoder_policy="transform",
                probe_records=probe_records,
                precision_source=precision_source,
            ),
            _evaluate_matrix_trial(
                "matrix-fixed-decoder-centered-stabilizer",
                h7,
                (h7_path,),
                frame_profile="nonidentity",
                action_profile="fixed_decoder_stabilizer",
                decoder_policy="fixed",
                probe_records=probe_records,
                precision_source=precision_source,
            ),
            _evaluate_matrix_trial(
                "matrix-fixed-decoder-outside-stabilizer",
                h7,
                (h7_path,),
                frame_profile="nonidentity",
                action_profile="diagonal",
                decoder_policy="fixed",
                probe_records=probe_records,
                precision_source=precision_source,
            ),
        )
        precision_source.require_complete()
        semantic = {
            "decimal_precision": 100,
            "gauss_hermite_orders": (41, 51),
            "raw_fixture_sha256": all_raw_hashes,
            "trials": [
                {
                    "trial_id": trial.trial_id,
                    "frame_profile": trial.frame_profile,
                    "decoder_policy": trial.decoder_policy,
                    "action_sha256": trial.action_sha256,
                    "source_paths": [item.path_id for item in trial.source_paths],
                    "original": [item.value_sha256 for item in trial.original.values],
                    "transformed": [
                        item.value_sha256 for item in trial.transformed.values
                    ],
                    "backward": [
                        {
                            "record_sha256": record.backward_sha256,
                            "bound_budget_sha256": bound.bound_sha256,
                        }
                        for record, bound in zip(
                            trial.backward_records,
                            trial.backward_bound_budgets,
                            strict=True,
                        )
                    ],
                    "backward_inventory_size": trial.backward_inventory_size,
                    "r_back_max": trial.r_back_max,
                    "scalars": trial.scalar_items,
                    "probes": [item.probe_id for item in trial.probe_evaluations],
                }
                for trial in trials
            ],
        }
        inventory_sha256 = hashlib.sha256(_canonical_bytes(semantic)).hexdigest()
        if inventory_sha256 != _ORACLE_INVENTORY_SHA256:
            raise _H7ExternalDataError(
                "Task-6 oracle inventory hash changed from preregistration"
            )
        return H7MPOracleResult(
            status="EVIDENCE_VERIFIED",
            open_obligations=(),
            decimal_precision=100,
            gauss_hermite_orders=(41, 51),
            raw_fixture_sha256=all_raw_hashes,
            h1_source_paths=h1_paths,
            h7_source_path=h7_path,
            trials=trials,
            inventory_sha256=inventory_sha256,
        )
    except _H7ExternalDataError as error:
        raise H7OracleInconclusive(str(error)) from error
    finally:
        mp.mp.dps = previous_precision


def _inconclusive_result(
    raw_hashes: tuple[str, ...],
    h1_paths: tuple[MPSourcePathRecord, ...],
    h7_path: MPSourcePathRecord,
    *obligations: str,
) -> H7MPOracleResult:
    return H7MPOracleResult(
        status="INCONCLUSIVE",
        open_obligations=tuple(obligations),
        decimal_precision=100,
        gauss_hermite_orders=(41, 51),
        raw_fixture_sha256=raw_hashes,
        h1_source_paths=h1_paths,
        h7_source_path=h7_path,
        trials=(),
        inventory_sha256=None,
    )


def _parse_raw_json(raw: bytes) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise _H7ExternalDataError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise _H7ExternalDataError(f"nonfinite JSON number {value!r}")

    try:
        parsed = json.loads(
            raw.decode("utf-8"),
            parse_float=str,
            parse_int=str,
            parse_constant=reject_constant,
            object_pairs_hook=pairs_hook,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _H7ExternalDataError("fixture must be strict UTF-8 JSON") from error
    if type(parsed) is not dict:
        raise _H7ExternalDataError("fixture root must be an object")
    return parsed


def _exact_fields(
    value: object,
    expected: set[str] | frozenset[str],
    location: str,
) -> Mapping[str, Any]:
    if type(value) is not dict or set(value) != set(expected):
        raise _H7ExternalDataError(f"{location} must have the exact closed field set")
    return cast(dict[str, Any], value)


def _exact_sequence(value: object, length: int, location: str) -> list[Any]:
    if type(value) is not list or len(value) != length:
        raise _H7ExternalDataError(f"{location} must contain exactly {length} entries")
    return cast(list[Any], value)


def _numeric_tensor(value: object, shape: tuple[int, ...], location: str) -> None:
    if not shape:
        _mp(value)
        return
    rows = _exact_sequence(value, shape[0], location)
    for index, item in enumerate(rows):
        _numeric_tensor(item, shape[1:], f"{location}[{index}]")


def _require_positive_normalized_simplex(
    value: object,
    length: int,
    location: str,
) -> None:
    row = _exact_sequence(value, length, location)
    probabilities = tuple(_mp(item) for item in row)
    if any(item <= 0 for item in probabilities) or mp.fsum(probabilities) != 1:
        raise _H7ExternalDataError(f"{location} must be a positive normalized simplex")


def _require_spd_tensor(
    value: object,
    dimension: int,
    location: str,
) -> None:
    _numeric_tensor(value, (dimension, dimension), location)
    matrix = _matrix(cast(Sequence[Sequence[object]], value))
    if _max_abs(matrix - _transpose(matrix)) > mp.mpf("1e-70") * _matrix_scale(matrix):
        raise _H7ExternalDataError(f"{location} must be symmetric")
    try:
        _condition_spd(matrix)
    except ValueError as error:
        raise _H7ExternalDataError(f"{location} must be positive definite") from error


def _require_sha256(value: object, location: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise _H7ExternalDataError(f"{location} must be a lowercase SHA-256 digest")
    return value


def _h7_canonical(value: object) -> object:
    if type(value) is bytes:
        return {"hex": value.hex(), "length": len(value)}
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("canonical H7 float must be finite")
        return value.hex()
    if isinstance(value, Mapping):
        if not all(type(key) is str and key for key in value):
            raise ValueError("oracle H7 canonical mappings require string keys")
        return {cast(str, key): _h7_canonical(value[key]) for key in sorted(value)}
    if type(value) in (tuple, list):
        return [_h7_canonical(item) for item in value]
    if type(value) in (str, int, bool) or value is None:
        return value
    raise ValueError(f"unsupported oracle H7 canonical type {type(value).__name__}")


def _h7_hash(domain: str, value: object) -> str:
    return hashlib.sha256(
        domain.encode("ascii")
        + b"\x00"
        + json.dumps(
            _h7_canonical(value),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _finite_binary64(value: object, location: str) -> float:
    _mp(value)
    try:
        converted = float(cast(str, value))
    except (OverflowError, ValueError) as error:
        raise _H7ExternalDataError(
            f"{location} is outside the finite binary64 domain"
        ) from error
    if not math.isfinite(converted):
        raise _H7ExternalDataError(f"{location} is outside the finite binary64 domain")
    return converted


def _canonical_binary64(value: object, location: str) -> float:
    if type(value) is not str or not value:
        raise _H7ExternalDataError(
            f"{location} must be a canonical binary64 text token"
        )
    try:
        converted = float(value)
    except (OverflowError, ValueError) as error:
        raise _H7ExternalDataError(
            f"{location} is outside the finite binary64 domain"
        ) from error
    if (
        not math.isfinite(converted)
        or repr(converted) != value
        or struct.pack("<d", float(repr(converted)))
        != struct.pack("<d", converted)
    ):
        raise _H7ExternalDataError(
            f"{location} violates the canonical binary64 text policy"
        )
    return converted


def _canonical_binary64_tensor(
    value: object,
    shape: tuple[int, ...],
    location: str,
) -> None:
    if not shape:
        _canonical_binary64(value, location)
        return
    rows = _exact_sequence(value, shape[0], location)
    for index, item in enumerate(rows):
        _canonical_binary64_tensor(
            item,
            shape[1:],
            f"{location}[{index}]",
        )


def _owned_snapshot_semantic(
    raw_values: object,
    shape: tuple[int, ...],
    location: str,
) -> Mapping[str, object]:
    _numeric_tensor(raw_values, shape, location)
    flattened: list[float] = []

    def visit(value: object, remaining: tuple[int, ...]) -> None:
        if not remaining:
            flattened.append(_finite_binary64(value, location))
            return
        for item in cast(list[Any], value):
            visit(item, remaining[1:])

    visit(raw_values, shape)
    raw_bytes = b"".join(struct.pack("<d", item) for item in flattened)
    raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    snapshot_semantic = {
        "capture_contract": {
            "dtype": "float64",
            "shape": shape,
            "device": "cpu",
            "contiguous": True,
            "requires_grad": False,
        },
        "owned_storage_version": 0,
        "dtype": "float64",
        "shape": shape,
        "device": "cpu",
        "raw_bytes": raw_bytes,
        "raw_bytes_sha256": raw_sha256,
    }
    return {
        "dtype": "float64",
        "shape": shape,
        "device": "cpu",
        "raw_bytes_sha256": raw_sha256,
        "snapshot_sha256": _h7_hash(
            "vfe4.h7.owned-tensor-snapshot.v1",
            snapshot_semantic,
        ),
    }


def _density_probe_semantic(
    record: Mapping[str, Any],
    dimension: int,
    location: str,
) -> Mapping[str, object]:
    return {
        "probe_id": record["probe_id"],
        "fixture_id": record["fixture_id"],
        "component_id": record["component_id"],
        "source_id": record["source_id"],
        "action_sha256": record["action_sha256"],
        "anchor_sha256": record["anchor_sha256"],
        "anchor_provenance": record["anchor_provenance"],
        "x": _owned_snapshot_semantic(
            record["x"],
            (dimension,),
            f"{location}.x",
        ),
        "x_prime": _owned_snapshot_semantic(
            record["x_prime"],
            (dimension,),
            f"{location}.x_prime",
        ),
        "initial_log_jacobian_shift": _finite_binary64(
            record["initial_log_jacobian_shift"],
            f"{location}.initial_log_jacobian_shift",
        ),
        "receiver_log_jacobian_shift": _finite_binary64(
            record["receiver_log_jacobian_shift"],
            f"{location}.receiver_log_jacobian_shift",
        ),
        "global_log_jacobian_shift": _finite_binary64(
            record["global_log_jacobian_shift"],
            f"{location}.global_log_jacobian_shift",
        ),
    }


def _density_probe_hash(semantic: Mapping[str, object]) -> str:
    return _h7_hash("vfe4.h7.density-probe-pair.v1", semantic)


def _action_hash(
    raw_elements: object,
    *,
    dimension: int,
    kind: str,
    scalar: bool,
    location: str,
) -> str:
    elements = _exact_sequence(raw_elements, 3, location)
    owned = tuple(
        _owned_snapshot_semantic(
            item,
            (dimension, dimension),
            f"{location}[{index}]",
        )
        for index, item in enumerate(elements)
    )
    semantic = {
        "elements": owned,
        "kind": kind,
        "dimension": dimension,
        "group": f"GL+({dimension},R)",
        "representation": ("standard_scalar" if scalar else "direct_gl_plus_2"),
    }
    domain = (
        "vfe4.h7.scalar-replay-action.v1" if scalar else "vfe4.h7.gl-plus-2-action.v1"
    )
    return _h7_hash(domain, semantic)


def _validate_h1_fixture(h1: Mapping[str, Any]) -> None:
    root_fields = {
        "fixture_schema_version",
        "fixture_id",
        "continuous_order",
        "vocabulary_labels",
        "observation_label_base",
        "observation_labels",
        "frames",
        "initial_joint",
        "model_source_priors",
        "state_source_priors",
        "model_offsets",
        "model_variances",
        "state_offsets",
        "state_variances",
        "state_model_slopes",
        "decoder",
        "recognition",
        "quadrature",
    }
    root = _exact_fields(h1, root_fields, "h1")
    continuous_order = _exact_sequence(
        root["continuous_order"],
        6,
        "h1.continuous_order",
    )
    vocabulary_labels = _exact_sequence(
        root["vocabulary_labels"],
        3,
        "h1.vocabulary_labels",
    )
    observation_labels = _exact_sequence(
        root["observation_labels"],
        2,
        "h1.observation_labels",
    )
    if (
        _as_int(root["fixture_schema_version"]) != 1
        or root["fixture_id"] != "h1-v1"
        or tuple(continuous_order) != ("z0", "m0", "z1", "m1", "z2", "m2")
        or tuple(_as_int(item) for item in vocabulary_labels) != (1, 2, 3)
        or _as_int(root["observation_label_base"]) != 1
        or tuple(_as_int(item) for item in observation_labels) != (1, 2)
    ):
        raise _H7ExternalDataError("H1 fixture identity/order/labels changed")
    frames = _exact_sequence(root["frames"], 3, "h1.frames")
    _numeric_tensor(frames, (3,), "h1.frames")
    if any(_mp(item) <= 0 for item in frames):
        raise _H7ExternalDataError("H1 frames must remain in GL+(1,R)")
    initial = _exact_fields(
        root["initial_joint"], {"mean", "covariance"}, "h1.initial_joint"
    )
    _numeric_tensor(initial["mean"], (2,), "h1.initial_joint.mean")
    _require_spd_tensor(
        initial["covariance"],
        2,
        "h1.initial_joint.covariance",
    )
    for name, shape in (
        ("model_source_priors", (2,)),
        ("state_source_priors", (2,)),
    ):
        rows = _exact_sequence(root[name], 2, f"h1.{name}")
        _numeric_tensor(rows[0], (1,), f"h1.{name}[0]")
        _numeric_tensor(rows[1], shape, f"h1.{name}[1]")
        _require_positive_normalized_simplex(
            rows[0],
            1,
            f"h1.{name}[0]",
        )
        _require_positive_normalized_simplex(
            rows[1],
            shape[0],
            f"h1.{name}[1]",
        )
    for name in (
        "model_offsets",
        "model_variances",
        "state_offsets",
        "state_variances",
        "state_model_slopes",
    ):
        _numeric_tensor(root[name], (2,), f"h1.{name}")
    for name in ("model_variances", "state_variances"):
        variances = _exact_sequence(root[name], 2, f"h1.{name}")
        if any(_mp(item) <= 0 for item in variances):
            raise _H7ExternalDataError(f"h1.{name} must be strictly positive")
    decoder_rows = _exact_sequence(root["decoder"], 2, "h1.decoder")
    for index, raw_decoder in enumerate(decoder_rows):
        decoder = _exact_fields(
            raw_decoder, {"w_z", "w_m", "bias"}, f"h1.decoder[{index}]"
        )
        for name in ("w_z", "w_m", "bias"):
            _numeric_tensor(decoder[name], (3,), f"h1.decoder[{index}].{name}")
    recognition = _exact_fields(
        root["recognition"],
        {
            "initial_mean",
            "initial_covariance",
            "model_source_probabilities",
            "state_source_probabilities_given_model_source",
            "model_kernels",
            "state_kernels",
        },
        "h1.recognition",
    )
    _numeric_tensor(recognition["initial_mean"], (2,), "h1.recognition.initial_mean")
    _require_spd_tensor(
        recognition["initial_covariance"],
        2,
        "h1.recognition.initial_covariance",
    )
    model_probabilities = _exact_sequence(
        recognition["model_source_probabilities"],
        2,
        "h1.recognition.model_source_probabilities",
    )
    _numeric_tensor(
        model_probabilities[0],
        (1,),
        "h1.recognition.model_source_probabilities[0]",
    )
    _numeric_tensor(
        model_probabilities[1],
        (2,),
        "h1.recognition.model_source_probabilities[1]",
    )
    _require_positive_normalized_simplex(
        model_probabilities[0],
        1,
        "h1.recognition.model_source_probabilities[0]",
    )
    _require_positive_normalized_simplex(
        model_probabilities[1],
        2,
        "h1.recognition.model_source_probabilities[1]",
    )
    state_probabilities = _exact_sequence(
        recognition["state_source_probabilities_given_model_source"],
        2,
        "h1.recognition.state_source_probabilities_given_model_source",
    )
    _numeric_tensor(
        state_probabilities[0],
        (1, 1),
        "h1.recognition.state_source_probabilities_given_model_source[0]",
    )
    _numeric_tensor(
        state_probabilities[1],
        (2, 2),
        "h1.recognition.state_source_probabilities_given_model_source[1]",
    )
    first_state_probability_rows = _exact_sequence(
        state_probabilities[0],
        1,
        "h1.recognition.state_source_probabilities_given_model_source[0]",
    )
    _require_positive_normalized_simplex(
        first_state_probability_rows[0],
        1,
        "h1.recognition.state_source_probabilities_given_model_source[0][0]",
    )
    second_state_probability_rows = _exact_sequence(
        state_probabilities[1],
        2,
        "h1.recognition.state_source_probabilities_given_model_source[1]",
    )
    for index, row in enumerate(second_state_probability_rows):
        _require_positive_normalized_simplex(
            row,
            2,
            f"h1.recognition.state_source_probabilities_given_model_source[1][{index}]",
        )
    model_kernels = _exact_sequence(
        recognition["model_kernels"], 2, "h1.recognition.model_kernels"
    )
    for receiver_index, expected_rows in enumerate((1, 2)):
        rows = _exact_sequence(
            model_kernels[receiver_index],
            expected_rows,
            f"h1.recognition.model_kernels[{receiver_index}]",
        )
        for row_index, raw_row in enumerate(rows):
            row = _exact_fields(
                raw_row,
                {"slope", "offset", "variance"},
                f"h1.recognition.model_kernels[{receiver_index}][{row_index}]",
            )
            for value in row.values():
                _mp(value)
            if _mp(row["variance"]) <= 0:
                raise _H7ExternalDataError(
                    "H1 model-kernel variance must be strictly positive"
                )
    state_kernels = _exact_sequence(
        recognition["state_kernels"], 2, "h1.recognition.state_kernels"
    )
    first_state_row = _exact_fields(
        _exact_sequence(state_kernels[0], 1, "h1.recognition.state_kernels[0]")[0],
        {"z_slope", "m_slope", "offset", "variance"},
        "h1.recognition.state_kernels[0][0]",
    )
    for value in first_state_row.values():
        _mp(value)
    if _mp(first_state_row["variance"]) <= 0:
        raise _H7ExternalDataError("H1 state-kernel variance must be strictly positive")
    second_state_rows = _exact_sequence(
        state_kernels[1], 4, "h1.recognition.state_kernels[1]"
    )
    expected_sources = ((0, 0), (1, 0), (0, 1), (1, 1))
    for index, (raw_row, expected_source) in enumerate(
        zip(second_state_rows, expected_sources, strict=True)
    ):
        row = _exact_fields(
            raw_row,
            {"a", "b", "z_slope", "m_slope", "offset", "variance"},
            f"h1.recognition.state_kernels[1][{index}]",
        )
        if (_as_int(row["a"]), _as_int(row["b"])) != expected_source:
            raise _H7ExternalDataError("H1 state-kernel source order changed")
        for name in ("z_slope", "m_slope", "offset", "variance"):
            _mp(row[name])
        if _mp(row["variance"]) <= 0:
            raise _H7ExternalDataError(
                "H1 state-kernel variance must be strictly positive"
            )
    quadrature = _exact_fields(
        root["quadrature"],
        {"order", "convergence_check_order", "maximum_convergence_estimate"},
        "h1.quadrature",
    )
    if (
        _as_int(quadrature["order"]) != 21
        or _as_int(quadrature["convergence_check_order"]) != 17
        or _mp(quadrature["maximum_convergence_estimate"]) != mp.mpf("1e-9")
    ):
        raise _H7ExternalDataError("H1 quadrature declaration changed")


def _validate_h7_fixture(h7: Mapping[str, Any]) -> None:
    root_fields = {
        "fixture_schema_version",
        "fixture_id",
        "group",
        "representations",
        "horizon",
        "dimensions",
        "continuous_order",
        "state_parent_sets",
        "model_parent_sets",
        "state_source_support",
        "model_source_support",
        "observation_label_base",
        "observation_labels",
        "frame_profiles",
        "actions",
        "generative",
        "recognition",
        "density_probes",
        "oracle",
    }
    root = _exact_fields(h7, root_fields, "h7")
    representations = _exact_fields(
        root["representations"], {"state", "model"}, "h7.representations"
    )
    dimensions = _exact_fields(
        root["dimensions"], {"d_z", "d_m", "D", "V"}, "h7.dimensions"
    )
    continuous_order = _exact_sequence(
        root["continuous_order"],
        12,
        "h7.continuous_order",
    )
    observation_labels = _exact_sequence(
        root["observation_labels"],
        2,
        "h7.observation_labels",
    )
    expected_order = tuple(
        f"{channel}{time}[{index}]"
        for time in range(3)
        for channel in ("z", "m")
        for index in range(2)
    )
    if (
        _as_int(root["fixture_schema_version"]) != 1
        or root["fixture_id"] != "h7-v1"
        or root["group"] != "GL+(2,R)"
        or representations != {"state": "standard", "model": "standard"}
        or tuple(_as_int(dimensions[key]) for key in ("d_z", "d_m", "D", "V"))
        != (2, 2, 12, 3)
        or _as_int(root["horizon"]) != 2
        or tuple(continuous_order) != expected_order
        or _as_int(root["observation_label_base"]) != 0
        or tuple(_as_int(item) for item in observation_labels) != (0, 2)
    ):
        raise _H7ExternalDataError("H7 fixture identity/dimensions/order changed")
    for name in (
        "state_parent_sets",
        "model_parent_sets",
        "state_source_support",
        "model_source_support",
    ):
        table = _exact_sequence(root[name], 2, f"h7.{name}")
        observed = tuple(
            tuple(
                _as_int(item)
                for item in _exact_sequence(
                    row,
                    1,
                    f"h7.{name}[{index}]",
                )
            )
            for index, row in enumerate(table)
        )
        if observed != ((0,), (1,)):
            raise _H7ExternalDataError(f"h7.{name} changed from the frozen chain")
    frames = _exact_fields(
        root["frame_profiles"], {"identity", "nonidentity"}, "h7.frame_profiles"
    )
    actions = _exact_fields(
        root["actions"],
        {"diagonal", "internal", "fixed_decoder_stabilizer"},
        "h7.actions",
    )
    for name, values in frames.items():
        _numeric_tensor(values, (3, 2, 2), f"h7.frame_profiles.{name}")
        for index, item in enumerate(values):
            if mp.det(_matrix(item)) <= 0:
                raise _H7ExternalDataError(
                    f"h7.frame_profiles.{name}[{index}] is not in GL+"
                )
    for name, values in actions.items():
        _numeric_tensor(values, (3, 2, 2), f"h7.actions.{name}")
        for index, item in enumerate(values):
            if mp.det(_matrix(item)) <= 0:
                raise _H7ExternalDataError(f"h7.actions.{name}[{index}] is not in GL+")
    _validate_h7_generative(root["generative"])
    _validate_h7_recognition(root["recognition"])
    _validate_h7_probe_inventory(root["density_probes"])
    oracle = _exact_fields(
        root["oracle"], {"decimal_precision", "gauss_hermite_orders"}, "h7.oracle"
    )
    oracle_orders = _exact_sequence(
        oracle["gauss_hermite_orders"],
        2,
        "h7.oracle.gauss_hermite_orders",
    )
    if _as_int(oracle["decimal_precision"]) != 100 or tuple(
        _as_int(item) for item in oracle_orders
    ) != (41, 51):
        raise _H7ExternalDataError("H7 oracle precision/order declaration changed")


def _validate_h7_generative(value: object) -> None:
    root = _exact_fields(
        value,
        {
            "initial_mean",
            "initial_covariance",
            "model_source_probabilities",
            "state_source_probabilities",
            "source_scorer_profile",
            "model_offsets",
            "model_receiver_covariances",
            "state_offsets",
            "state_receiver_covariances",
            "B",
            "decoder",
        },
        "h7.generative",
    )
    _numeric_tensor(root["initial_mean"], (4,), "h7.generative.initial_mean")
    _require_spd_tensor(
        root["initial_covariance"],
        4,
        "h7.generative.initial_covariance",
    )
    for name in ("model_source_probabilities", "state_source_probabilities"):
        rows = _exact_sequence(root[name], 2, f"h7.generative.{name}")
        _numeric_tensor(rows, (2, 1), f"h7.generative.{name}")
        for index, row in enumerate(rows):
            _require_positive_normalized_simplex(
                row,
                1,
                f"h7.generative.{name}[{index}]",
            )
    for name in ("model_offsets", "state_offsets"):
        _numeric_tensor(root[name], (2, 2), f"h7.generative.{name}")
    for name in (
        "model_receiver_covariances",
        "state_receiver_covariances",
    ):
        rows = _exact_sequence(root[name], 2, f"h7.generative.{name}")
        for index, row in enumerate(rows):
            _require_spd_tensor(
                row,
                2,
                f"h7.generative.{name}[{index}]",
            )
    _numeric_tensor(root["B"], (2, 2, 2), "h7.generative.B")
    scorer = _exact_fields(
        root["source_scorer_profile"],
        {
            "profile_id",
            "law",
            "prefix_tokens",
            "alpha_bias",
            "alpha_token_scale",
            "z_history",
            "m_history",
            "r_z",
            "r_m",
        },
        "h7.generative.source_scorer_profile",
    )
    prefix_tokens = _exact_sequence(
        scorer["prefix_tokens"],
        2,
        "h7.scorer.prefix_tokens",
    )
    if (
        scorer["profile_id"] != "h7-linear-history-source-v1"
        or scorer["law"] != "alpha(prefix)+r_z^T z_j+r_m^T m_j"
        or tuple(_as_int(item) for item in prefix_tokens) != (0, 2)
    ):
        raise _H7ExternalDataError("H7 scorer identity/order changed")
    for name in ("alpha_bias", "alpha_token_scale", "r_z", "r_m"):
        banks = _exact_fields(scorer[name], {"model", "state"}, f"h7.scorer.{name}")
        shape = (2,) if name.startswith("alpha_") else (2, 2)
        for bank, entries in banks.items():
            _numeric_tensor(entries, shape, f"h7.scorer.{name}.{bank}")
    _numeric_tensor(scorer["z_history"], (2, 2), "h7.scorer.z_history")
    _numeric_tensor(scorer["m_history"], (2, 2), "h7.scorer.m_history")
    decoders = _exact_sequence(root["decoder"], 2, "h7.generative.decoder")
    for index, value in enumerate(decoders):
        decoder = _exact_fields(
            value, {"W_z", "W_m", "bias"}, f"h7.generative.decoder[{index}]"
        )
        _numeric_tensor(decoder["W_z"], (3, 2), f"h7.decoder[{index}].W_z")
        _numeric_tensor(decoder["W_m"], (3, 2), f"h7.decoder[{index}].W_m")
        _numeric_tensor(decoder["bias"], (3,), f"h7.decoder[{index}].bias")


def _validate_h7_recognition(value: object) -> None:
    root = _exact_fields(
        value,
        {
            "family_id",
            "initial_mean",
            "initial_covariance",
            "model_source_probabilities",
            "state_source_probabilities_given_model_source",
            "model_parent_maps",
            "model_offsets",
            "model_receiver_covariances",
            "state_parent_maps",
            "state_model_maps",
            "state_offsets",
            "state_receiver_covariances",
            "factorized_fixture",
        },
        "h7.recognition",
    )
    if root["family_id"] != "structured-full-block-v1":
        raise _H7ExternalDataError("H7 structured recognition identity changed")
    _numeric_tensor(root["initial_mean"], (4,), "h7.recognition.initial_mean")
    _require_spd_tensor(
        root["initial_covariance"],
        4,
        "h7.recognition.initial_covariance",
    )
    model_probability_rows = _exact_sequence(
        root["model_source_probabilities"],
        2,
        "h7.recognition.model_source_probabilities",
    )
    for index, row in enumerate(model_probability_rows):
        _require_positive_normalized_simplex(
            row,
            1,
            f"h7.recognition.model_source_probabilities[{index}]",
        )
    state_probability_tables = _exact_sequence(
        root["state_source_probabilities_given_model_source"],
        2,
        "h7.recognition.state_source_probabilities_given_model_source",
    )
    for time_index, table in enumerate(state_probability_tables):
        rows = _exact_sequence(
            table,
            1,
            "h7.recognition."
            f"state_source_probabilities_given_model_source[{time_index}]",
        )
        _require_positive_normalized_simplex(
            rows[0],
            1,
            "h7.recognition."
            f"state_source_probabilities_given_model_source[{time_index}][0]",
        )
    for name in (
        "model_parent_maps",
        "state_parent_maps",
        "state_model_maps",
    ):
        _numeric_tensor(root[name], (2, 2, 2), f"h7.recognition.{name}")
    for name in (
        "model_receiver_covariances",
        "state_receiver_covariances",
    ):
        rows = _exact_sequence(root[name], 2, f"h7.recognition.{name}")
        for index, row in enumerate(rows):
            _require_spd_tensor(
                row,
                2,
                f"h7.recognition.{name}[{index}]",
            )
    for name in ("model_offsets", "state_offsets"):
        _numeric_tensor(root[name], (2, 2), f"h7.recognition.{name}")
    factorized = _exact_fields(
        root["factorized_fixture"],
        {
            "family_id",
            "representation",
            "shared_fields",
            "initial_mean",
            "initial_diagonal_covariance",
            "model_receiver_diagonal_covariances",
            "state_receiver_diagonal_covariances",
            "generic_gl_plus_2_output_representation",
        },
        "h7.recognition.factorized_fixture",
    )
    expected_shared = (
        "model_source_probabilities",
        "state_source_probabilities_given_model_source",
        "model_parent_maps",
        "model_offsets",
        "state_parent_maps",
        "state_model_maps",
        "state_offsets",
    )
    shared_fields = _exact_sequence(
        factorized["shared_fields"],
        len(expected_shared),
        "h7.recognition.factorized_fixture.shared_fields",
    )
    if (
        factorized["family_id"] != "factorized-diagonal-within-fiber-v1"
        or factorized["representation"] != "factorized_diagonal_within_fiber"
        or tuple(shared_fields) != expected_shared
        or factorized["generic_gl_plus_2_output_representation"]
        != "unrestricted_full_block_pushforward"
    ):
        raise _H7ExternalDataError("H7 factorized recognition identity changed")
    _numeric_tensor(factorized["initial_mean"], (4,), "h7.factorized.initial_mean")
    _numeric_tensor(
        factorized["initial_diagonal_covariance"],
        (4,),
        "h7.factorized.initial_diagonal_covariance",
    )
    _numeric_tensor(
        factorized["model_receiver_diagonal_covariances"],
        (2, 2),
        "h7.factorized.model_receiver_diagonal_covariances",
    )
    _numeric_tensor(
        factorized["state_receiver_diagonal_covariances"],
        (2, 2),
        "h7.factorized.state_receiver_diagonal_covariances",
    )
    factorized_covariance_values = (
        *cast(list[object], factorized["initial_diagonal_covariance"]),
        *(
            item
            for row in cast(
                list[list[object]],
                factorized["model_receiver_diagonal_covariances"],
            )
            for item in row
        ),
        *(
            item
            for row in cast(
                list[list[object]],
                factorized["state_receiver_diagonal_covariances"],
            )
            for item in row
        ),
    )
    if any(_mp(item) <= 0 for item in factorized_covariance_values):
        raise _H7ExternalDataError(
            "H7 factorized covariance diagonals must be strictly positive"
        )


def _direction_ids(dimension: int) -> tuple[str, ...]:
    return (
        "zero",
        *tuple(
            direction
            for index in range(dimension)
            for direction in (f"+e{index}", f"-e{index}")
        ),
    )


def _validate_h7_probe_inventory(value: object) -> None:
    root = _exact_fields(
        value,
        {
            "probe_set_schema",
            "whitened_scale",
            "anchor_policy",
            "anchor_provenance",
            "pair_law",
            "direction_ids_by_dimension",
            "components",
        },
        "h7.density_probes",
    )
    if (
        root["probe_set_schema"] != "h7-density-probe-pairs-v1"
        or _mp(root["whitened_scale"]) != mp.mpf("0.25")
        or root["anchor_policy"] != "original_component_mean"
        or root["anchor_provenance"]
        != "raw_fixture_component_mean_and_lower_cholesky_v1"
        or root["pair_law"] != "x=anchor+L@(scale*direction);x_prime=G_component@x"
    ):
        raise _H7ExternalDataError("H7 density-probe policy changed")
    directions = _exact_fields(
        root["direction_ids_by_dimension"], {"2", "4", "12"}, "h7.probe.directions"
    )
    for dimension in (2, 4, 12):
        expected_directions = _direction_ids(dimension)
        observed_directions = _exact_sequence(
            directions[str(dimension)],
            len(expected_directions),
            f"h7.probe.directions[{dimension}]",
        )
        if tuple(observed_directions) != expected_directions:
            raise _H7ExternalDataError(
                f"H7 density directions changed for dimension {dimension}"
            )
    raw_components = _exact_sequence(root["components"], 18, "h7.probe.components")
    observed: list[tuple[str, str, int, str]] = []
    for index, raw_component in enumerate(raw_components):
        component = _exact_fields(
            raw_component,
            {"component_id", "source_id", "dimension", "shift_scope"},
            f"h7.probe.components[{index}]",
        )
        observed.append(
            (
                cast(str, component["component_id"]),
                cast(str, component["source_id"]),
                _as_int(component["dimension"]),
                cast(str, component["shift_scope"]),
            )
        )
    if tuple(observed) != _PROBE_COMPONENTS:
        raise _H7ExternalDataError("H7 density-probe component inventory changed")


def _probe_action(
    component_id: str,
    actions: tuple[mp.matrix, mp.matrix, mp.matrix],
) -> mp.matrix:
    if component_id.endswith(".initial_joint"):
        return _block_diag(actions[0], actions[0])
    if ".receiver_1" in component_id:
        return actions[1]
    if ".receiver_2" in component_id:
        return actions[2]
    if component_id.endswith(".global"):
        return _block_diag(
            actions[0],
            actions[0],
            actions[1],
            actions[1],
            actions[2],
            actions[2],
        )
    raise ValueError("probe component has no declared action")


def _require_action_relation(
    record: Mapping[str, Any],
    action: mp.matrix,
    dimension: int,
    location: str,
) -> None:
    _numeric_tensor(record["x"], (dimension,), f"{location}.x")
    _numeric_tensor(record["x_prime"], (dimension,), f"{location}.x_prime")
    expected = action * _vector(record["x"])
    observed = _vector(record["x_prime"])
    error = _max_abs(expected - observed)
    scale = max(_matrix_scale(expected), _matrix_scale(observed))
    if error > mp.mpf("2e-14") * scale:
        raise _H7ExternalDataError(
            f"{location}.x_prime violates the declared action relation"
        )


def _validate_matrix_probe_table(
    probes: Mapping[str, Any],
    h7: Mapping[str, Any],
) -> None:
    root = _exact_fields(
        probes,
        {"probe_table_schema", "probe_set_sha256", "records"},
        "matrix_probe_table",
    )
    if (
        root["probe_table_schema"] != "h7-density-probe-table-v1"
        or root["probe_set_sha256"] != _MATRIX_PROBE_SET_SHA256
    ):
        raise _H7ExternalDataError("matrix probe table identity changed")
    records = _exact_sequence(root["records"], 486, "matrix_probe_table.records")
    record_fields = {
        "row_index",
        "probe_id",
        "fixture_id",
        "component_id",
        "source_id",
        "action_sha256",
        "anchor_sha256",
        "anchor_provenance",
        "x",
        "x_prime",
        "initial_log_jacobian_shift",
        "receiver_log_jacobian_shift",
        "global_log_jacobian_shift",
        "probe_sha256",
    }
    actions_by_name = {
        name: cast(
            tuple[mp.matrix, mp.matrix, mp.matrix],
            tuple(_matrix(item) for item in h7["actions"][name]),
        )
        for name in ("diagonal", "fixed_decoder_stabilizer", "internal")
    }
    for action_name in ("diagonal", "fixed_decoder_stabilizer", "internal"):
        observed_action_sha256 = _action_hash(
            h7["actions"][action_name],
            dimension=2,
            kind=("internal_product" if action_name == "internal" else "diagonal_base"),
            scalar=False,
            location=f"h7.actions.{action_name}",
        )
        if observed_action_sha256 != _MATRIX_ACTION_SHA256[action_name]:
            raise _H7ExternalDataError(f"matrix action hash changed for {action_name}")
    row_index = 0
    row_hashes: list[str] = []
    pair_records: list[Mapping[str, object]] = []
    for action_name in ("diagonal", "fixed_decoder_stabilizer", "internal"):
        actions = actions_by_name[action_name]
        for component_id, source_id, dimension, shift_scope in _PROBE_COMPONENTS:
            for direction_id in _direction_ids(dimension):
                location = f"matrix_probe_table.records[{row_index}]"
                record = _exact_fields(records[row_index], record_fields, location)
                if (
                    _as_int(record["row_index"]) != row_index
                    or record["probe_id"]
                    != f"{action_name}:{component_id}:{direction_id}"
                    or record["fixture_id"] != "h7-v1"
                    or record["component_id"] != component_id
                    or record["source_id"] != source_id
                    or record["action_sha256"] != _MATRIX_ACTION_SHA256[action_name]
                    or record["anchor_sha256"] != _PROBE_ANCHOR_SHA256[component_id]
                    or record["anchor_provenance"]
                    != "raw_fixture_component_mean_and_lower_cholesky_v1"
                ):
                    raise _H7ExternalDataError(
                        f"{location} changed identity/order/hash"
                    )
                row_hashes.append(
                    _require_sha256(record["probe_sha256"], f"{location}.probe_sha256")
                )
                _require_sha256(record["anchor_sha256"], f"{location}.anchor_sha256")
                for name in (
                    "initial_log_jacobian_shift",
                    "receiver_log_jacobian_shift",
                    "global_log_jacobian_shift",
                ):
                    _mp(record[name])
                nonzero_scopes = tuple(
                    name
                    for name in (
                        "initial_log_jacobian_shift",
                        "receiver_log_jacobian_shift",
                        "global_log_jacobian_shift",
                    )
                    if _mp(record[name]) != 0
                )
                expected_nonzero = {
                    "initial_joint": ("initial_log_jacobian_shift",),
                    "receiver_model": ("receiver_log_jacobian_shift",),
                    "receiver_state": ("receiver_log_jacobian_shift",),
                    "global": ("global_log_jacobian_shift",),
                }[shift_scope]
                if nonzero_scopes != expected_nonzero:
                    raise _H7ExternalDataError(
                        f"{location} changed Jacobian-shift scope"
                    )
                _require_action_relation(
                    record,
                    _probe_action(component_id, actions),
                    dimension,
                    location,
                )
                semantic = _density_probe_semantic(record, dimension, location)
                if _density_probe_hash(semantic) != record["probe_sha256"]:
                    raise _H7ExternalDataError(
                        f"{location}.probe_sha256 does not bind its row"
                    )
                pair_records.append(
                    {**semantic, "probe_sha256": record["probe_sha256"]}
                )
                row_index += 1
    if row_index != len(records) or len(set(row_hashes)) != 486:
        raise _H7ExternalDataError(
            "matrix probe rows/hashes are missing, duplicated, or reordered"
        )
    if (
        _h7_hash("vfe4.h7.density-probe-set.v1", tuple(pair_records))
        != root["probe_set_sha256"]
    ):
        raise _H7ExternalDataError(
            "matrix probe tuple hash does not bind the exact row tuple"
        )


def _validate_scalar_probe_table(
    probes: Mapping[str, Any],
    h1: Mapping[str, Any],
    paths: tuple[MPSourcePathRecord, ...],
) -> None:
    root = _exact_fields(
        probes,
        {
            "probe_table_schema",
            "fixture_id",
            "raw_fixture_sha256",
            "ordered_source_path_ids",
            "scalar_trial_action_sha256",
            "anchor_provenance",
            "probe_set_sha256",
            "records",
        },
        "scalar_probe_table",
    )
    declared_path_ids = _exact_sequence(
        root["ordered_source_path_ids"],
        len(_H1_PATH_IDS),
        "scalar_probe_table.ordered_source_path_ids",
    )
    declared_action_hashes = _exact_sequence(
        root["scalar_trial_action_sha256"],
        2,
        "scalar_probe_table.scalar_trial_action_sha256",
    )
    path_ids = tuple(path.path_id for path in paths)
    if (
        root["probe_table_schema"] != "h7-scalar-density-probe-table-v1"
        or root["fixture_id"] != "h1-v1"
        or root["raw_fixture_sha256"] != _H1_RAW_SHA256
        or tuple(declared_path_ids) != _H1_PATH_IDS
        or path_ids != _H1_PATH_IDS
        or root["anchor_provenance"] != "original-generative-conditional-global-mean-v1"
    ):
        raise _H7ExternalDataError("scalar probe table identity/path order changed")
    action_hashes = tuple(declared_action_hashes)
    if len(set(action_hashes)) != 2:
        raise _H7ExternalDataError("scalar probe action-hash inventory changed")
    for index, digest in enumerate(action_hashes):
        _require_sha256(digest, f"scalar_probe_table.action_sha256[{index}]")
    _require_sha256(root["probe_set_sha256"], "scalar_probe_table.probe_set_sha256")
    records = _exact_sequence(root["records"], 8, "scalar_probe_table.records")
    record_fields = {
        "row_index",
        "probe_id",
        "fixture_id",
        "component_id",
        "source_id",
        "action_sha256",
        "anchor_sha256",
        "anchor_provenance",
        "x",
        "x_prime",
        "initial_log_jacobian_shift",
        "receiver_log_jacobian_shift",
        "global_log_jacobian_shift",
        "probe_sha256",
    }
    trial_actions = (
        ("scalar-base-transformed", ("1.25", "1.25", "1.25")),
        ("scalar-internal-transformed", ("0.8", "1.1", "1.4")),
    )
    expected_action_hashes = tuple(
        _action_hash(
            [[[value]] for value in action_values],
            dimension=1,
            kind=(
                "diagonal_base"
                if trial_id == "scalar-base-transformed"
                else "internal_product"
            ),
            scalar=True,
            location=f"scalar_probe_table.actions.{trial_id}",
        )
        for trial_id, action_values in trial_actions
    )
    if action_hashes != expected_action_hashes:
        raise _H7ExternalDataError(
            "scalar probe action hashes do not bind the frozen actions"
        )
    h1_law = _make_h1_law(h1, paths)
    expected_anchor_by_path = {
        path.path_id: _joint_moments(h1_law, path, role="p").mean for path in paths
    }
    anchor_by_path: dict[str, str] = {}
    row_hashes: list[str] = []
    pair_records: list[Mapping[str, object]] = []
    row_index = 0
    for trial_index, (trial_id, action_values) in enumerate(trial_actions):
        action = _diag(tuple(value for value in action_values for _channel in range(2)))
        expected_shift = 2 * mp.fsum(mp.log(_mp(item)) for item in action_values)
        for path_id in path_ids:
            location = f"scalar_probe_table.records[{row_index}]"
            record = _exact_fields(records[row_index], record_fields, location)
            expected_probe_id = f"{trial_id}:h1.p.global.source_path:{path_id}"
            if (
                _as_int(record["row_index"]) != row_index
                or record["probe_id"] != expected_probe_id
                or record["fixture_id"] != "h1-v1"
                or record["component_id"] != "h1.p.global.source_path"
                or record["source_id"] != path_id
                or record["action_sha256"] != action_hashes[trial_index]
                or _mp(record["initial_log_jacobian_shift"]) != 0
                or _mp(record["receiver_log_jacobian_shift"]) != 0
                or abs(_mp(record["global_log_jacobian_shift"]) - expected_shift)
                > mp.mpf("2e-14")
                or type(record["anchor_provenance"]) is not str
                or _H1_RAW_SHA256 not in record["anchor_provenance"]
                or path_id not in record["anchor_provenance"]
                or root["anchor_provenance"] not in record["anchor_provenance"]
            ):
                raise _H7ExternalDataError(
                    f"{location} changed identity/order/action/scope"
                )
            anchor_sha256 = _require_sha256(
                record["anchor_sha256"], f"{location}.anchor_sha256"
            )
            if path_id in anchor_by_path and anchor_by_path[path_id] != anchor_sha256:
                raise _H7ExternalDataError(
                    "scalar anchor hash changed between trial actions"
                )
            anchor_by_path[path_id] = anchor_sha256
            row_hashes.append(
                _require_sha256(record["probe_sha256"], f"{location}.probe_sha256")
            )
            _require_action_relation(record, action, 6, location)
            serialized_anchor = _vector(record["x"])
            expected_anchor = expected_anchor_by_path[path_id]
            anchor_error = _max_abs(serialized_anchor - expected_anchor)
            anchor_scale = max(
                _matrix_scale(serialized_anchor),
                _matrix_scale(expected_anchor),
            )
            if anchor_error > mp.mpf("2e-14") * anchor_scale:
                raise _H7ExternalDataError(
                    f"{location}.x is not the direct H1 generative conditional "
                    "global mean"
                )
            semantic = _density_probe_semantic(record, 6, location)
            expected_anchor_sha256 = _h7_hash(
                "vfe4.h7.scalar-density-anchor.v1",
                {
                    "raw_fixture_sha256": _H1_RAW_SHA256,
                    "source_id": path_id,
                    "anchor": semantic["x"],
                },
            )
            if anchor_sha256 != expected_anchor_sha256:
                raise _H7ExternalDataError(f"{location}.anchor_sha256 does not bind x")
            if _density_probe_hash(semantic) != record["probe_sha256"]:
                raise _H7ExternalDataError(
                    f"{location}.probe_sha256 does not bind its row"
                )
            pair_records.append({**semantic, "probe_sha256": record["probe_sha256"]})
            row_index += 1
    if (
        row_index != len(records)
        or len(anchor_by_path) != 4
        or len(set(anchor_by_path.values())) != 4
        or len(set(row_hashes)) != 8
    ):
        raise _H7ExternalDataError(
            "scalar probe hashes are missing, duplicated, or reordered"
        )
    scalar_set_semantic = {
        "fixture_id": root["fixture_id"],
        "raw_fixture_sha256": _H1_RAW_SHA256,
        "ordered_source_path_ids": path_ids,
        "scalar_trial_action_sha256": action_hashes,
        "anchor_provenance": root["anchor_provenance"],
        "probe_pairs": tuple(pair_records),
    }
    if (
        _h7_hash("vfe4.h7.scalar-probe-set.v1", scalar_set_semantic)
        != root["probe_set_sha256"]
    ):
        raise _H7ExternalDataError(
            "scalar probe tuple hash does not bind the exact row tuple"
        )


def build_h7_scalar_probe_table_bytes(h1_fixture_bytes: bytes) -> bytes:
    """Build the exact frozen scalar H1 density-probe table from raw bytes."""

    if type(h1_fixture_bytes) is not bytes or not h1_fixture_bytes:
        raise ValueError("h1_fixture_bytes must be nonempty exact bytes")
    previous_precision = mp.mp.dps
    mp.mp.dps = 100
    try:
        h1 = _parse_raw_json(h1_fixture_bytes)
        _validate_h1_fixture(h1)
        if hashlib.sha256(h1_fixture_bytes).hexdigest() != _H1_RAW_SHA256:
            raise _H7ExternalDataError("raw H1 fixture identity changed")
        paths = _h1_source_paths(h1)
        h1_law = _make_h1_law(h1, paths)
        anchor_provenance = "original-generative-conditional-global-mean-v1"
        trial_actions = (
            ("scalar-base-transformed", ("1.25", "1.25", "1.25")),
            ("scalar-internal-transformed", ("0.8", "1.1", "1.4")),
        )
        action_hashes = tuple(
            _action_hash(
                [[[value]] for value in action_values],
                dimension=1,
                kind=(
                    "diagonal_base"
                    if trial_id == "scalar-base-transformed"
                    else "internal_product"
                ),
                scalar=True,
                location=f"scalar_probe_table.actions.{trial_id}",
            )
            for trial_id, action_values in trial_actions
        )
        records: list[dict[str, object]] = []
        pair_records: list[Mapping[str, object]] = []
        row_index = 0
        for trial_index, (trial_id, action_values) in enumerate(trial_actions):
            action = _diag(
                tuple(value for value in action_values for _channel in range(2))
            )
            global_shift = _decimal(
                2 * mp.fsum(mp.log(_mp(value)) for value in action_values)
            )
            for path in paths:
                anchor = _joint_moments(h1_law, path, role="p").mean
                x = [_decimal(anchor[index]) for index in range(anchor.rows)]
                x_prime_matrix = action * anchor
                x_prime = [
                    _decimal(x_prime_matrix[index])
                    for index in range(x_prime_matrix.rows)
                ]
                record: dict[str, object] = {
                    "row_index": str(row_index),
                    "probe_id": (
                        f"{trial_id}:h1.p.global.source_path:{path.path_id}"
                    ),
                    "fixture_id": "h1-v1",
                    "component_id": "h1.p.global.source_path",
                    "source_id": path.path_id,
                    "action_sha256": action_hashes[trial_index],
                    "anchor_sha256": "",
                    "anchor_provenance": (
                        f"{anchor_provenance}|raw_fixture_sha256={_H1_RAW_SHA256}"
                        f"|source_id={path.path_id}"
                    ),
                    "x": x,
                    "x_prime": x_prime,
                    "initial_log_jacobian_shift": "0.0",
                    "receiver_log_jacobian_shift": "0.0",
                    "global_log_jacobian_shift": global_shift,
                    "probe_sha256": "",
                }
                semantic = _density_probe_semantic(
                    record,
                    6,
                    f"scalar_probe_table.records[{row_index}]",
                )
                record["anchor_sha256"] = _h7_hash(
                    "vfe4.h7.scalar-density-anchor.v1",
                    {
                        "raw_fixture_sha256": _H1_RAW_SHA256,
                        "source_id": path.path_id,
                        "anchor": semantic["x"],
                    },
                )
                semantic = _density_probe_semantic(
                    record,
                    6,
                    f"scalar_probe_table.records[{row_index}]",
                )
                record["probe_sha256"] = _density_probe_hash(semantic)
                final_semantic = _density_probe_semantic(
                    record,
                    6,
                    f"scalar_probe_table.records[{row_index}]",
                )
                pair_records.append(
                    {**final_semantic, "probe_sha256": record["probe_sha256"]}
                )
                records.append(record)
                row_index += 1
        table = {
            "probe_table_schema": "h7-scalar-density-probe-table-v1",
            "fixture_id": "h1-v1",
            "raw_fixture_sha256": _H1_RAW_SHA256,
            "ordered_source_path_ids": list(_H1_PATH_IDS),
            "scalar_trial_action_sha256": list(action_hashes),
            "anchor_provenance": anchor_provenance,
            "probe_set_sha256": _h7_hash(
                "vfe4.h7.scalar-probe-set.v1",
                {
                    "fixture_id": "h1-v1",
                    "raw_fixture_sha256": _H1_RAW_SHA256,
                    "ordered_source_path_ids": _H1_PATH_IDS,
                    "scalar_trial_action_sha256": action_hashes,
                    "anchor_provenance": anchor_provenance,
                    "probe_pairs": tuple(pair_records),
                },
            ),
            "records": records,
        }
        table_bytes = _canonical_bytes(table) + b"\n"
        _validate_scalar_probe_table(
            _parse_raw_json(table_bytes),
            h1,
            paths,
        )
        return table_bytes
    finally:
        mp.mp.dps = previous_precision


def _validate_precision_operand_table(
    value: Mapping[str, Any],
) -> _MPPrecisionOperandSource:
    root = _exact_fields(
        value,
        {
            "precision_table_schema",
            "h1_raw_fixture_sha256",
            "h7_raw_fixture_sha256",
            "ordered_trial_ids",
            "source_contract",
            "binary64_text_policy",
            "precision_set_sha256",
            "records",
        },
        "precision_operand_table",
    )
    ordered_trial_ids = _exact_sequence(
        root["ordered_trial_ids"],
        len(H7_REQUIRED_TRIAL_IDS),
        "precision_operand_table.ordered_trial_ids",
    )
    if (
        root["precision_table_schema"] != _PRECISION_TABLE_SCHEMA
        or root["h1_raw_fixture_sha256"] != _H1_RAW_SHA256
        or root["h7_raw_fixture_sha256"] != _H7_RAW_SHA256
        or tuple(ordered_trial_ids) != H7_REQUIRED_TRIAL_IDS
        or root["source_contract"] != _PRECISION_SOURCE_CONTRACT
        or root["binary64_text_policy"] != _BINARY64_TEXT_POLICY
    ):
        raise _H7ExternalDataError(
            "precision operand table identity/source contract changed"
        )
    set_sha256 = _require_sha256(
        root["precision_set_sha256"],
        "precision_operand_table.precision_set_sha256",
    )
    records = _exact_sequence(
        root["records"],
        _PRECISION_OPERAND_COUNT,
        "precision_operand_table.records",
    )
    record_fields = {
        "row_index",
        "trial_id",
        "gaussian_id",
        "source_kind",
        "shape",
        "covariance_values",
        "covariance_values_sha256",
        "covariance_snapshot_sha256",
        "precision_values",
        "precision_values_sha256",
        "precision_snapshot_sha256",
        "record_sha256",
    }
    validated: list[Mapping[str, Any]] = []
    record_hashes: list[str] = []
    expected_rows = _expected_precision_row_identities()
    source_counts = {"owned_component": 0, "assembled_global": 0}
    for index, raw_record in enumerate(records):
        location = f"precision_operand_table.records[{index}]"
        record = _exact_fields(raw_record, record_fields, location)
        shape_values = _exact_sequence(record["shape"], 2, f"{location}.shape")
        shape = tuple(_as_int(item) for item in shape_values)
        expected_trial_id, expected_gaussian_id, expected_source_kind = (
            expected_rows[index]
        )
        if (
            record["row_index"] != str(index)
            or record["trial_id"] != expected_trial_id
            or record["gaussian_id"] != expected_gaussian_id
            or record["source_kind"] != expected_source_kind
            or len(shape) != 2
            or shape[0] <= 0
            or shape[0] != shape[1]
        ):
            raise _H7ExternalDataError(
                f"{location} changed identity/order/square shape"
            )
        is_global = (
            ".global[" in cast(str, record["gaussian_id"])
            and cast(str, record["gaussian_id"]).endswith("]")
        )
        if is_global != (record["source_kind"] == "assembled_global"):
            raise _H7ExternalDataError(
                f"{location} changed global precision source kind"
            )
        source_counts[cast(str, record["source_kind"])] += 1
        square_shape = cast(tuple[int, int], shape)
        _canonical_binary64_tensor(
            record["covariance_values"],
            square_shape,
            f"{location}.covariance_values",
        )
        _canonical_binary64_tensor(
            record["precision_values"],
            square_shape,
            f"{location}.precision_values",
        )
        covariance_snapshot = _owned_snapshot_semantic(
            record["covariance_values"],
            square_shape,
            f"{location}.covariance_values",
        )
        precision_snapshot = _owned_snapshot_semantic(
            record["precision_values"],
            square_shape,
            f"{location}.precision_values",
        )
        for name in (
            "covariance_values_sha256",
            "covariance_snapshot_sha256",
            "precision_values_sha256",
            "precision_snapshot_sha256",
            "record_sha256",
        ):
            _require_sha256(record[name], f"{location}.{name}")
        if (
            record["covariance_snapshot_sha256"]
            != covariance_snapshot["snapshot_sha256"]
            or record["precision_snapshot_sha256"]
            != precision_snapshot["snapshot_sha256"]
        ):
            raise _H7ExternalDataError(
                f"{location} tensor snapshot hash does not bind its values"
            )
        identity = {
            "trial_id": record["trial_id"],
            "gaussian_id": record["gaussian_id"],
            "source_kind": record["source_kind"],
            "shape": record["shape"],
        }
        if record["covariance_values_sha256"] != _h7_hash(
            _COVARIANCE_VALUES_DOMAIN,
            {
                **identity,
                "covariance_values": record["covariance_values"],
            },
        ):
            raise _H7ExternalDataError(
                f"{location}.covariance_values_sha256 does not bind values"
            )
        if record["precision_values_sha256"] != _h7_hash(
            _PRECISION_VALUES_DOMAIN,
            {
                **identity,
                "precision_values": record["precision_values"],
            },
        ):
            raise _H7ExternalDataError(
                f"{location}.precision_values_sha256 does not bind values"
            )
        semantic = {
            key: record[key]
            for key in record_fields
            if key != "record_sha256"
        }
        if record["record_sha256"] != _h7_hash(
            _PRECISION_ROW_DOMAIN,
            semantic,
        ):
            raise _H7ExternalDataError(
                f"{location}.record_sha256 does not bind the exact row"
            )
        record_hashes.append(cast(str, record["record_sha256"]))
        validated.append(record)
    if (
        len(set(record_hashes)) != _PRECISION_OPERAND_COUNT
        or source_counts
        != {"owned_component": 152, "assembled_global": 40}
    ):
        raise _H7ExternalDataError(
            "precision operand rows/hashes are missing or duplicated"
        )
    set_semantic = {
        "precision_table_schema": root["precision_table_schema"],
        "h1_raw_fixture_sha256": root["h1_raw_fixture_sha256"],
        "h7_raw_fixture_sha256": root["h7_raw_fixture_sha256"],
        "ordered_trial_ids": root["ordered_trial_ids"],
        "source_contract": root["source_contract"],
        "binary64_text_policy": root["binary64_text_policy"],
        "records": list(validated),
    }
    if set_sha256 != _h7_hash(
        _PRECISION_SET_DOMAIN,
        set_semantic,
    ):
        raise _H7ExternalDataError(
            "precision operand tuple hash does not bind the exact row tuple"
        )
    return _MPPrecisionOperandSource(
        records=tuple(validated),
        precision_set_sha256=set_sha256,
    )


def _validate_serialized_precision(
    covariance: mp.matrix,
    precision: mp.matrix,
    location: str,
) -> None:
    if (
        covariance.rows != covariance.cols
        or precision.rows != covariance.rows
        or precision.cols != covariance.cols
    ):
        raise _H7ExternalDataError(
            f"{location} covariance/precision dimensions disagree"
        )
    symmetry_error = _max_abs(precision - _transpose(precision))
    symmetry_scale = _matrix_scale(precision)
    binary64_unit = mp.mpf(2) ** -52
    if symmetry_error > 64 * binary64_unit * symmetry_scale:
        raise _H7ExternalDataError(f"{location} precision is not symmetric")
    try:
        covariance_condition = _condition_spd(covariance)
        precision_condition = _condition_spd(precision)
    except ValueError as error:
        raise _H7ExternalDataError(
            f"{location} precision is not positive definite"
        ) from error
    left = precision * covariance
    right = covariance * precision
    identity_error = max(
        abs(left[row, column] - (1 if row == column else 0))
        for row in range(left.rows)
        for column in range(left.cols)
    )
    identity_error = max(
        identity_error,
        *(
            abs(right[row, column] - (1 if row == column else 0))
            for row in range(right.rows)
            for column in range(right.cols)
        ),
    )
    inverse_allowance = (
        64 * binary64_unit * max(covariance_condition, precision_condition)
    )
    if identity_error > inverse_allowance:
        raise _H7ExternalDataError(
            f"{location} precision is not the two-sided inverse of its "
            "serialized covariance"
        )


def _mp(value: object) -> mp.mpf:
    if type(value) is not str or not value:
        raise _H7ExternalDataError("fixture numbers must remain decimal strings")
    try:
        result = mp.mpf(value)
    except (TypeError, ValueError) as error:
        raise _H7ExternalDataError("fixture number is not decimal") from error
    if not mp.isfinite(result):
        raise _H7ExternalDataError("fixture number must be finite")
    return result


def _as_int(value: object) -> int:
    if type(value) is not str:
        raise _H7ExternalDataError("fixture integer must remain a decimal string")
    try:
        result = int(value)
    except ValueError as error:
        raise _H7ExternalDataError("fixture integer is not decimal") from error
    if str(result) != value:
        raise _H7ExternalDataError("fixture integer must be canonical")
    return result


def _vector(values: Sequence[object]) -> mp.matrix:
    return mp.matrix([_mp(item) for item in values])


def _matrix(values: Sequence[Sequence[object]]) -> mp.matrix:
    rows = tuple(tuple(_mp(item) for item in row) for row in values)
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("matrix rows must be nonempty and rectangular")
    return mp.matrix(rows)


def _diag(values: Sequence[object]) -> mp.matrix:
    result = mp.matrix(len(values), len(values))
    for row in range(len(values)):
        for column in range(len(values)):
            result[row, column] = _mp(values[row]) if row == column else mp.mpf("0")
    return result


def _zeros(rows: int, columns: int) -> mp.matrix:
    result = mp.matrix(rows, columns)
    for row in range(rows):
        for column in range(columns):
            result[row, column] = mp.mpf("0")
    return result


def _transpose(value: mp.matrix) -> mp.matrix:
    return value.T


def _solve_left(coefficient: mp.matrix, rhs: mp.matrix) -> mp.matrix:
    """Solve against each actual RHS column; never synthesize an identity."""

    if coefficient.rows != coefficient.cols or coefficient.rows != rhs.rows:
        raise ValueError("direct solve shapes disagree")
    result = _zeros(coefficient.cols, rhs.cols)
    for column in range(rhs.cols):
        actual_rhs = mp.matrix([rhs[row, column] for row in range(rhs.rows)])
        solution = mp.lu_solve(coefficient, actual_rhs)
        for row in range(coefficient.cols):
            result[row, column] = solution[row]
    return result


def _solve_right(lhs: mp.matrix, coefficient: mp.matrix) -> mp.matrix:
    return _transpose(_solve_left(_transpose(coefficient), _transpose(lhs)))


def _congruence(action: mp.matrix, covariance: mp.matrix) -> mp.matrix:
    return action * covariance * _transpose(action)


def _block_diag(*blocks: mp.matrix) -> mp.matrix:
    rows = sum(block.rows for block in blocks)
    columns = sum(block.cols for block in blocks)
    result = _zeros(rows, columns)
    row_offset = 0
    column_offset = 0
    for block in blocks:
        for row in range(block.rows):
            for column in range(block.cols):
                result[row_offset + row, column_offset + column] = block[row, column]
        row_offset += block.rows
        column_offset += block.cols
    return result


def _outer(left: mp.matrix, right: mp.matrix) -> mp.matrix:
    return left * _transpose(right)


def _logdet_spd(value: mp.matrix) -> mp.mpf:
    factor = mp.cholesky(value)
    return 2 * mp.fsum(mp.log(factor[index, index]) for index in range(value.rows))


def _condition_spd(value: mp.matrix) -> mp.mpf:
    eigenvalues, _ = mp.eigsy(value)
    smallest = mp.mpf(eigenvalues[0])
    largest = mp.mpf(eigenvalues[value.rows - 1])
    if smallest <= 0:
        raise ValueError("SPD operand has a nonpositive eigenvalue")
    return largest / smallest


def _matrix_scale(value: mp.matrix) -> mp.mpf:
    return max(
        mp.mpf("1"),
        *(
            abs(value[row, column])
            for row in range(value.rows)
            for column in range(value.cols)
        ),
    )


def _max_abs(value: mp.matrix) -> mp.mpf:
    return max(
        abs(value[row, column])
        for row in range(value.rows)
        for column in range(value.cols)
    )


def _decimal(value: mp.mpf) -> str:
    checked = mp.mpf(value)
    if not mp.isfinite(checked):
        raise ValueError("oracle value must be finite")
    if checked == mp.floor(checked):
        return f"{int(checked)}.0"
    return mp.nstr(
        checked,
        n=100,
        strip_zeros=False,
        min_fixed=-1000,
        max_fixed=1000,
    )


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _value_record(
    operand_id: str,
    value: mp.matrix,
    *,
    vector: bool = False,
    condition_number: mp.mpf | None = None,
) -> MPValueRecord:
    shape = (value.rows,) if vector else (value.rows, value.cols)
    decimals = tuple(
        _decimal(value[row, column])
        for row in range(value.rows)
        for column in range(value.cols)
    )
    semantic = {
        "operand_id": operand_id,
        "shape": shape,
        "decimal_values": decimals,
    }
    scale = _matrix_scale(value)
    condition = mp.mpf("1") if condition_number is None else condition_number
    normalization = max(mp.mpf("1"), scale)
    return MPValueRecord(
        operand_id=operand_id,
        shape=shape,
        decimal_values=decimals,
        value_sha256=hashlib.sha256(_canonical_bytes(semantic)).hexdigest(),
        scale=_decimal(scale),
        condition_number=_decimal(condition),
        normalization=_decimal(normalization),
    )


def _h1_source_paths(h1: Mapping[str, Any]) -> tuple[MPSourcePathRecord, ...]:
    generative_model = cast(list[list[object]], h1["model_source_priors"])
    generative_state = cast(list[list[object]], h1["state_source_priors"])
    recognition = cast(dict[str, Any], h1["recognition"])
    recognition_model = cast(
        list[list[object]],
        recognition["model_source_probabilities"],
    )
    recognition_state = cast(
        list[list[list[object]]],
        recognition["state_source_probabilities_given_model_source"],
    )
    declarations = (
        ("h1-path-0:a0-b0", 0, 0),
        ("h1-path-1:a1-b0", 1, 0),
        ("h1-path-2:a0-b1", 0, 1),
        ("h1-path-3:a1-b1", 1, 1),
    )
    result: list[MPSourcePathRecord] = []
    for path_id, second_a, second_b in declarations:
        a = (0, second_a)
        b = (0, second_b)
        q_probability = mp.mpf("1")
        p_probability = mp.mpf("1")
        for index in range(2):
            q_probability *= _mp(recognition_model[index][b[index]])
            q_probability *= _mp(recognition_state[index][b[index]][a[index]])
            p_probability *= _mp(generative_model[index][b[index]])
            p_probability *= _mp(generative_state[index][a[index]])
        result.append(
            MPSourcePathRecord(
                path_id=path_id,
                a=a,
                b=b,
                q_probability=_decimal(q_probability),
                p_probability=_decimal(p_probability),
            )
        )
    if mp.fsum(_mp(item.q_probability) for item in result) != 1:
        raise _H7ExternalDataError("H1 recognition source paths do not normalize")
    return tuple(result)


def _component(
    component_id: str,
    bank: str,
    receiver_t: int,
    source_j: int,
    parent_map: mp.matrix,
    model_map: mp.matrix | None,
    offset: mp.matrix,
    covariance: mp.matrix,
) -> dict[str, Any]:
    return {
        "component_id": component_id,
        "bank": bank,
        "receiver_t": receiver_t,
        "source_j": source_j,
        "parent_map": parent_map,
        "model_map": model_map,
        "offset": offset,
        "covariance": covariance,
    }


def _make_h1_law(
    data: Mapping[str, Any],
    paths: tuple[MPSourcePathRecord, ...],
) -> dict[str, Any]:
    frames = tuple(mp.matrix([[_mp(item)]]) for item in data["frames"])
    p_components: dict[tuple[str, int, int], dict[str, Any]] = {}
    for receiver_t in (1, 2):
        for source_j in range(receiver_t):
            link = _solve_right(frames[receiver_t], frames[source_j])
            p_components[("model", receiver_t, source_j)] = _component(
                f"p.model.receiver_{receiver_t}.source_{source_j}",
                "model",
                receiver_t,
                source_j,
                link,
                None,
                mp.matrix([_mp(data["model_offsets"][receiver_t - 1])]),
                mp.matrix([[_mp(data["model_variances"][receiver_t - 1])]]),
            )
            p_components[("state", receiver_t, source_j)] = _component(
                f"p.state.receiver_{receiver_t}.source_{source_j}",
                "state",
                receiver_t,
                source_j,
                link,
                mp.matrix([[_mp(data["state_model_slopes"][receiver_t - 1])]]),
                mp.matrix([_mp(data["state_offsets"][receiver_t - 1])]),
                mp.matrix([[_mp(data["state_variances"][receiver_t - 1])]]),
            )
    recognition = cast(dict[str, Any], data["recognition"])
    source_factors: dict[str, tuple[tuple[mp.mpf, mp.mpf, mp.mpf, mp.mpf], ...]] = {}
    for path in paths:
        source_factors[path.path_id] = tuple(
            (
                _mp(recognition["model_source_probabilities"][index][path.b[index]]),
                _mp(
                    recognition["state_source_probabilities_given_model_source"][index][
                        path.b[index]
                    ][path.a[index]]
                ),
                _mp(data["model_source_priors"][index][path.b[index]]),
                _mp(data["state_source_priors"][index][path.a[index]]),
            )
            for index in range(2)
        )
    q_model: dict[tuple[int, int], dict[str, Any]] = {}
    for receiver_t, rows in enumerate(recognition["model_kernels"], start=1):
        for source_j, row in enumerate(rows):
            q_model[(receiver_t, source_j)] = _component(
                f"q.model.receiver_{receiver_t}.source_{source_j}",
                "model",
                receiver_t,
                source_j,
                mp.matrix([[_mp(row["slope"])]]),
                None,
                mp.matrix([_mp(row["offset"])]),
                mp.matrix([[_mp(row["variance"])]]),
            )
    q_state: dict[tuple[int, int, int], dict[str, Any]] = {}
    first = recognition["state_kernels"][0][0]
    q_state[(1, 0, 0)] = _component(
        "q.state.receiver_1.a_0.b_0",
        "state",
        1,
        0,
        mp.matrix([[_mp(first["z_slope"])]]),
        mp.matrix([[_mp(first["m_slope"])]]),
        mp.matrix([_mp(first["offset"])]),
        mp.matrix([[_mp(first["variance"])]]),
    )
    for row in recognition["state_kernels"][1]:
        a = _as_int(row["a"])
        b = _as_int(row["b"])
        q_state[(2, a, b)] = _component(
            f"q.state.receiver_2.a_{a}.b_{b}",
            "state",
            2,
            a,
            mp.matrix([[_mp(row["z_slope"])]]),
            mp.matrix([[_mp(row["m_slope"])]]),
            mp.matrix([_mp(row["offset"])]),
            mp.matrix([[_mp(row["variance"])]]),
        )
    decoders = tuple(
        {
            "receiver_t": receiver_t,
            "state_weight": mp.matrix([[_mp(item)] for item in decoder["w_z"]]),
            "model_weight": mp.matrix([[_mp(item)] for item in decoder["w_m"]]),
            "bias": _vector(decoder["bias"]),
        }
        for receiver_t, decoder in enumerate(data["decoder"], start=1)
    )
    return {
        "fixture_id": "h1-v1",
        "dimension": 1,
        "frames": frames,
        "p_initial": {
            "mean": _vector(data["initial_joint"]["mean"]),
            "covariance": _matrix(data["initial_joint"]["covariance"]),
        },
        "q_initial": {
            "mean": _vector(recognition["initial_mean"]),
            "covariance": _matrix(recognition["initial_covariance"]),
        },
        "p_components": p_components,
        "q_model": q_model,
        "q_state": q_state,
        "decoders": decoders,
        "observation_labels": tuple(
            _as_int(item) - 1 for item in data["observation_labels"]
        ),
        "source_paths": paths,
        "source_factors": source_factors,
        "scorer_rows": (),
        "recognition_family": "scalar_h1_v1",
    }


def _make_h7_law(
    data: Mapping[str, Any],
    paths: tuple[MPSourcePathRecord, ...],
    *,
    frame_profile: Literal["identity", "nonidentity"],
    recognition_family: str,
) -> dict[str, Any]:
    frames = tuple(_matrix(item) for item in data["frame_profiles"][frame_profile])
    generative = cast(dict[str, Any], data["generative"])
    p_components: dict[tuple[str, int, int], dict[str, Any]] = {}
    for receiver_t, source_j in ((1, 0), (2, 1)):
        link = _solve_right(frames[receiver_t], frames[source_j])
        p_components[("model", receiver_t, source_j)] = _component(
            f"p.model.receiver_{receiver_t}",
            "model",
            receiver_t,
            source_j,
            link,
            None,
            _vector(generative["model_offsets"][receiver_t - 1]),
            _matrix(generative["model_receiver_covariances"][receiver_t - 1]),
        )
        p_components[("state", receiver_t, source_j)] = _component(
            f"p.state.receiver_{receiver_t}",
            "state",
            receiver_t,
            source_j,
            link,
            _matrix(generative["B"][receiver_t - 1]),
            _vector(generative["state_offsets"][receiver_t - 1]),
            _matrix(generative["state_receiver_covariances"][receiver_t - 1]),
        )
    recognition = cast(dict[str, Any], data["recognition"])
    factorized = cast(dict[str, Any], recognition["factorized_fixture"])
    if recognition_family == "structured_full_block":
        initial_covariance = _matrix(recognition["initial_covariance"])
        model_covariances = recognition["model_receiver_covariances"]
        state_covariances = recognition["state_receiver_covariances"]
    elif recognition_family == "factorized_diagonal_within_fiber":
        initial_covariance = _diag(factorized["initial_diagonal_covariance"])
        model_covariances = tuple(
            _diag(item) for item in factorized["model_receiver_diagonal_covariances"]
        )
        state_covariances = tuple(
            _diag(item) for item in factorized["state_receiver_diagonal_covariances"]
        )
    else:
        raise ValueError("unsupported H7 recognition family")
    q_model: dict[tuple[int, int], dict[str, Any]] = {}
    q_state: dict[tuple[int, int, int], dict[str, Any]] = {}
    for receiver_t, source_j in ((1, 0), (2, 1)):
        model_covariance = (
            model_covariances[receiver_t - 1]
            if recognition_family == "factorized_diagonal_within_fiber"
            else _matrix(model_covariances[receiver_t - 1])
        )
        state_covariance = (
            state_covariances[receiver_t - 1]
            if recognition_family == "factorized_diagonal_within_fiber"
            else _matrix(state_covariances[receiver_t - 1])
        )
        q_model[(receiver_t, source_j)] = _component(
            f"q.{_family_prefix(recognition_family)}.model.receiver_{receiver_t}",
            "model",
            receiver_t,
            source_j,
            _matrix(recognition["model_parent_maps"][receiver_t - 1]),
            None,
            _vector(recognition["model_offsets"][receiver_t - 1]),
            model_covariance,
        )
        q_state[(receiver_t, source_j, source_j)] = _component(
            f"q.{_family_prefix(recognition_family)}.state.receiver_{receiver_t}",
            "state",
            receiver_t,
            source_j,
            _matrix(recognition["state_parent_maps"][receiver_t - 1]),
            _matrix(recognition["state_model_maps"][receiver_t - 1]),
            _vector(recognition["state_offsets"][receiver_t - 1]),
            state_covariance,
        )
    decoders = tuple(
        {
            "receiver_t": receiver_t,
            "state_weight": _matrix(decoder["W_z"]),
            "model_weight": _matrix(decoder["W_m"]),
            "bias": _vector(decoder["bias"]),
        }
        for receiver_t, decoder in enumerate(generative["decoder"], start=1)
    )
    return {
        "fixture_id": "h7-v1",
        "dimension": 2,
        "frames": frames,
        "p_initial": {
            "mean": _vector(generative["initial_mean"]),
            "covariance": _matrix(generative["initial_covariance"]),
        },
        "q_initial": {
            "mean": _vector(
                (
                    recognition["initial_mean"]
                    if recognition_family == "structured_full_block"
                    else factorized["initial_mean"]
                )
            ),
            "covariance": initial_covariance,
        },
        "p_components": p_components,
        "q_model": q_model,
        "q_state": q_state,
        "decoders": decoders,
        "observation_labels": tuple(
            _as_int(item) for item in data["observation_labels"]
        ),
        "source_paths": paths,
        "source_factors": {
            paths[0].path_id: (
                (mp.mpf("1"), mp.mpf("1"), mp.mpf("1"), mp.mpf("1")),
                (mp.mpf("1"), mp.mpf("1"), mp.mpf("1"), mp.mpf("1")),
            )
        },
        "scorer_rows": _h7_scorer_rows(generative["source_scorer_profile"]),
        "recognition_family": recognition_family,
    }


def _family_prefix(family: str) -> str:
    return "structured" if family == "structured_full_block" else "factorized"


def _h7_scorer_rows(profile: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    prefix_tokens = tuple(_as_int(item) for item in profile["prefix_tokens"])
    z_history = tuple(_vector(item) for item in profile["z_history"])
    m_history = tuple(_vector(item) for item in profile["m_history"])
    rows: list[dict[str, Any]] = []
    for bank in ("model", "state"):
        for receiver_t, source_j in ((1, 0), (2, 1)):
            prefix = prefix_tokens[:receiver_t]
            weighted = sum(
                (index + 1) * (token + 1) for index, token in enumerate(prefix)
            )
            alpha_bias = _mp(profile["alpha_bias"][bank][receiver_t - 1])
            alpha_scale = _mp(profile["alpha_token_scale"][bank][receiver_t - 1])
            prefix_term = alpha_bias + alpha_scale * weighted
            z_covector = _vector(profile["r_z"][bank][receiver_t - 1])
            m_covector = _vector(profile["r_m"][bank][receiver_t - 1])
            raw_score = (
                prefix_term
                + (_transpose(z_covector) * z_history[source_j])[0]
                + (_transpose(m_covector) * m_history[source_j])[0]
            )
            rows.append(
                {
                    "bank": bank,
                    "receiver_t": receiver_t,
                    "source_j": source_j,
                    "prefix_tokens": prefix,
                    "prefix_term": prefix_term,
                    "z_history": z_history,
                    "m_history": m_history,
                    "z_covector": z_covector,
                    "m_covector": m_covector,
                    "support": (source_j,),
                    "mask": (True,),
                    "probability": mp.mpf("1"),
                    "raw_score": raw_score,
                }
            )
    return tuple(rows)


def _transform_law(
    law: Mapping[str, Any],
    actions: tuple[mp.matrix, mp.matrix, mp.matrix],
    *,
    decoder_policy: Literal["transform", "fixed"],
) -> dict[str, Any]:
    initial_action = _block_diag(actions[0], actions[0])
    p_initial = _transform_gaussian(law["p_initial"], initial_action)
    q_initial = _transform_gaussian(law["q_initial"], initial_action)
    p_components = {
        key: _transform_component(component, actions)
        for key, component in law["p_components"].items()
    }
    q_model = {
        key: _transform_component(component, actions)
        for key, component in law["q_model"].items()
    }
    q_state = {
        key: _transform_component(component, actions)
        for key, component in law["q_state"].items()
    }
    decoders = []
    for decoder in law["decoders"]:
        receiver_t = decoder["receiver_t"]
        if decoder_policy == "transform":
            state_weight = _solve_right(
                decoder["state_weight"],
                actions[receiver_t],
            )
            model_weight = _solve_right(
                decoder["model_weight"],
                actions[receiver_t],
            )
        else:
            state_weight = mp.matrix(decoder["state_weight"])
            model_weight = mp.matrix(decoder["model_weight"])
        decoders.append(
            {
                "receiver_t": receiver_t,
                "state_weight": state_weight,
                "model_weight": model_weight,
                "bias": mp.matrix(decoder["bias"]),
            }
        )
    return {
        **law,
        "frames": tuple(
            action * frame for action, frame in zip(actions, law["frames"], strict=True)
        ),
        "p_initial": p_initial,
        "q_initial": q_initial,
        "p_components": p_components,
        "q_model": q_model,
        "q_state": q_state,
        "decoders": tuple(decoders),
        "scorer_rows": _transform_scorer_rows(law["scorer_rows"], actions),
    }


def _transform_gaussian(
    gaussian: Mapping[str, mp.matrix],
    action: mp.matrix,
) -> dict[str, mp.matrix]:
    return {
        "mean": action * gaussian["mean"],
        "covariance": _congruence(action, gaussian["covariance"]),
    }


def _transform_component(
    component: Mapping[str, Any],
    actions: tuple[mp.matrix, mp.matrix, mp.matrix],
) -> dict[str, Any]:
    receiver_t = component["receiver_t"]
    source_j = component["source_j"]
    receiver_action = actions[receiver_t]
    source_action = actions[source_j]
    model_map = component["model_map"]
    transformed_model_map = (
        None
        if model_map is None
        else _solve_right(receiver_action * model_map, receiver_action)
    )
    return {
        **component,
        "parent_map": _solve_right(
            receiver_action * component["parent_map"],
            source_action,
        ),
        "model_map": transformed_model_map,
        "offset": receiver_action * component["offset"],
        "covariance": _congruence(
            receiver_action,
            component["covariance"],
        ),
    }


def _transform_scorer_rows(
    rows: tuple[dict[str, Any], ...],
    actions: tuple[mp.matrix, mp.matrix, mp.matrix],
) -> tuple[dict[str, Any], ...]:
    transformed: list[dict[str, Any]] = []
    for row in rows:
        source_action = actions[row["source_j"]]
        z_history = tuple(
            actions[index] * item for index, item in enumerate(row["z_history"])
        )
        m_history = tuple(
            actions[index] * item for index, item in enumerate(row["m_history"])
        )
        z_covector = _solve_left(
            _transpose(source_action),
            row["z_covector"],
        )
        m_covector = _solve_left(
            _transpose(source_action),
            row["m_covector"],
        )
        raw_score = (
            row["prefix_term"]
            + (_transpose(z_covector) * z_history[row["source_j"]])[0]
            + (_transpose(m_covector) * m_history[row["source_j"]])[0]
        )
        transformed.append(
            {
                **row,
                "z_history": z_history,
                "m_history": m_history,
                "z_covector": z_covector,
                "m_covector": m_covector,
                "raw_score": raw_score,
            }
        )
    return tuple(transformed)


def _recover_covariance(transformed: mp.matrix, action: mp.matrix) -> mp.matrix:
    """Recover ``C`` from ``A C A^T`` using only the actual operands as RHS."""

    return _solve_right(_solve_left(action, transformed), _transpose(action))


def _recover_gaussian(
    gaussian: Mapping[str, mp.matrix],
    action: mp.matrix,
) -> dict[str, mp.matrix]:
    return {
        "mean": _solve_left(action, gaussian["mean"]),
        "covariance": _recover_covariance(gaussian["covariance"], action),
    }


def _recover_component(
    component: Mapping[str, Any],
    actions: tuple[mp.matrix, mp.matrix, mp.matrix],
) -> dict[str, Any]:
    receiver_t = cast(int, component["receiver_t"])
    source_j = cast(int, component["source_j"])
    receiver_action = actions[receiver_t]
    source_action = actions[source_j]
    transformed_model_map = component["model_map"]
    return {
        **component,
        "parent_map": (
            _solve_left(receiver_action, component["parent_map"]) * source_action
        ),
        "model_map": (
            None
            if transformed_model_map is None
            else _solve_left(receiver_action, transformed_model_map) * receiver_action
        ),
        "offset": _solve_left(receiver_action, component["offset"]),
        "covariance": _recover_covariance(
            component["covariance"],
            receiver_action,
        ),
    }


def _recover_scorer_rows(
    rows: tuple[dict[str, Any], ...],
    actions: tuple[mp.matrix, mp.matrix, mp.matrix],
) -> tuple[dict[str, Any], ...]:
    recovered: list[dict[str, Any]] = []
    for row in rows:
        source_action = actions[row["source_j"]]
        z_history = tuple(
            _solve_left(actions[index], item)
            for index, item in enumerate(row["z_history"])
        )
        m_history = tuple(
            _solve_left(actions[index], item)
            for index, item in enumerate(row["m_history"])
        )
        z_covector = _transpose(source_action) * row["z_covector"]
        m_covector = _transpose(source_action) * row["m_covector"]
        raw_score = (
            row["prefix_term"]
            + (_transpose(z_covector) * z_history[row["source_j"]])[0]
            + (_transpose(m_covector) * m_history[row["source_j"]])[0]
        )
        recovered.append(
            {
                **row,
                "z_history": z_history,
                "m_history": m_history,
                "z_covector": z_covector,
                "m_covector": m_covector,
                "raw_score": raw_score,
            }
        )
    return tuple(recovered)


def _recover_law(
    transformed: Mapping[str, Any],
    actions: tuple[mp.matrix, mp.matrix, mp.matrix],
    *,
    decoder_policy: Literal["transform", "fixed"],
) -> dict[str, Any]:
    initial_action = _block_diag(actions[0], actions[0])
    decoders: list[dict[str, Any]] = []
    for decoder in transformed["decoders"]:
        receiver_t = decoder["receiver_t"]
        if decoder_policy == "transform":
            state_weight = decoder["state_weight"] * actions[receiver_t]
            model_weight = decoder["model_weight"] * actions[receiver_t]
        else:
            state_weight = mp.matrix(decoder["state_weight"])
            model_weight = mp.matrix(decoder["model_weight"])
        decoders.append(
            {
                "receiver_t": receiver_t,
                "state_weight": state_weight,
                "model_weight": model_weight,
                "bias": mp.matrix(decoder["bias"]),
            }
        )
    return {
        **transformed,
        "frames": tuple(
            _solve_left(action, frame)
            for action, frame in zip(
                actions,
                transformed["frames"],
                strict=True,
            )
        ),
        "p_initial": _recover_gaussian(transformed["p_initial"], initial_action),
        "q_initial": _recover_gaussian(transformed["q_initial"], initial_action),
        "p_components": {
            key: _recover_component(component, actions)
            for key, component in transformed["p_components"].items()
        },
        "q_model": {
            key: _recover_component(component, actions)
            for key, component in transformed["q_model"].items()
        },
        "q_state": {
            key: _recover_component(component, actions)
            for key, component in transformed["q_state"].items()
        },
        "decoders": tuple(decoders),
        "scorer_rows": _recover_scorer_rows(
            transformed["scorer_rows"],
            actions,
        ),
    }


def _require_exact_discrete_identity(
    original: Mapping[str, Any],
    transformed: Mapping[str, Any],
    recovered: Mapping[str, Any],
) -> None:
    for name in ("observation_labels", "source_paths", "source_factors"):
        if not (
            original[name] == transformed[name] and transformed[name] == recovered[name]
        ):
            raise ValueError(f"exact nonnumeric law field {name} changed")
    for role, law in (
        ("transformed", transformed),
        ("recovered", recovered),
    ):
        for index, (original_decoder, compared_decoder) in enumerate(
            zip(original["decoders"], law["decoders"], strict=True)
        ):
            if (
                original_decoder["receiver_t"] != compared_decoder["receiver_t"]
                or original_decoder["bias"].rows != compared_decoder["bias"].rows
                or original_decoder["bias"].cols != compared_decoder["bias"].cols
                or _max_abs(original_decoder["bias"] - compared_decoder["bias"]) != 0
            ):
                raise ValueError(
                    f"{role} decoder[{index}] receiver/bias identity changed"
                )
        for index, (original_row, compared_row) in enumerate(
            zip(original["scorer_rows"], law["scorer_rows"], strict=True)
        ):
            exact_fields = (
                "bank",
                "receiver_t",
                "source_j",
                "prefix_tokens",
                "support",
                "mask",
                "probability",
            )
            if any(original_row[name] != compared_row[name] for name in exact_fields):
                raise ValueError(f"{role} scorer row {index} exact fields changed")


def _block_indices(channel: str, population: int, dimension: int) -> tuple[int, ...]:
    start = 2 * population * dimension
    if channel == "m":
        start += dimension
    return tuple(range(start, start + dimension))


def _subvector(value: mp.matrix, indices: tuple[int, ...]) -> mp.matrix:
    return mp.matrix([value[index] for index in indices])


def _submatrix(
    value: mp.matrix,
    rows: tuple[int, ...],
    columns: tuple[int, ...],
) -> mp.matrix:
    return mp.matrix([[value[row, column] for column in columns] for row in rows])


def _joint_moments(
    law: Mapping[str, Any],
    path: MPSourcePathRecord,
    *,
    role: Literal["p", "q"],
) -> _Moments:
    dimension = cast(int, law["dimension"])
    total = 6 * dimension
    mean = _zeros(total, 1)
    covariance = _zeros(total, total)
    initial = law["p_initial"] if role == "p" else law["q_initial"]
    for row in range(2 * dimension):
        mean[row] = initial["mean"][row]
        for column in range(2 * dimension):
            covariance[row, column] = initial["covariance"][row, column]
    active = list(range(2 * dimension))
    for receiver_t in (1, 2):
        a = path.a[receiver_t - 1]
        b = path.b[receiver_t - 1]
        model = (
            law["p_components"][("model", receiver_t, b)]
            if role == "p"
            else law["q_model"][(receiver_t, b)]
        )
        model_target = _block_indices("m", receiver_t, dimension)
        model_parent = _block_indices("m", b, dimension)
        _insert_affine_moments(
            mean,
            covariance,
            active=tuple(active),
            target=model_target,
            parent_blocks=((model_parent, model["parent_map"]),),
            offset=model["offset"],
            noise_covariance=model["covariance"],
        )
        active.extend(model_target)
        state = (
            law["p_components"][("state", receiver_t, a)]
            if role == "p"
            else law["q_state"][(receiver_t, a, b)]
        )
        state_target = _block_indices("z", receiver_t, dimension)
        state_parent = _block_indices("z", a, dimension)
        _insert_affine_moments(
            mean,
            covariance,
            active=tuple(active),
            target=state_target,
            parent_blocks=(
                (state_parent, state["parent_map"]),
                (model_target, state["model_map"]),
            ),
            offset=state["offset"],
            noise_covariance=state["covariance"],
        )
        active.extend(state_target)
    return _Moments(mean=mean, covariance=covariance)


def _insert_affine_moments(
    mean: mp.matrix,
    covariance: mp.matrix,
    *,
    active: tuple[int, ...],
    target: tuple[int, ...],
    parent_blocks: tuple[tuple[tuple[int, ...], mp.matrix], ...],
    offset: mp.matrix,
    noise_covariance: mp.matrix,
) -> None:
    linear = _zeros(len(target), mean.rows)
    for indices, parent_map in parent_blocks:
        for row in range(parent_map.rows):
            for column, index in enumerate(indices):
                linear[row, index] += parent_map[row, column]
    target_mean = linear * mean + offset
    active_covariance = _submatrix(
        covariance,
        tuple(range(covariance.rows)),
        active,
    )
    cross = linear * active_covariance
    target_covariance = linear * covariance * _transpose(linear) + noise_covariance
    for row, target_row in enumerate(target):
        mean[target_row] = target_mean[row]
        for active_column, source_column in enumerate(active):
            covariance[target_row, source_column] = cross[row, active_column]
            covariance[source_column, target_row] = cross[row, active_column]
        for column, target_column in enumerate(target):
            covariance[target_row, target_column] = target_covariance[row, column]


def _trace(value: mp.matrix) -> mp.mpf:
    if value.rows != value.cols:
        raise ValueError("trace requires a square matrix")
    return mp.fsum(value[index, index] for index in range(value.rows))


def _gaussian_kl(
    q_mean: mp.matrix,
    q_covariance: mp.matrix,
    p_mean: mp.matrix,
    p_covariance: mp.matrix,
) -> mp.mpf:
    displacement = q_mean - p_mean
    solved_covariance = _solve_left(p_covariance, q_covariance)
    solved_displacement = _solve_left(p_covariance, displacement)
    quadratic = (_transpose(displacement) * solved_displacement)[0]
    return (
        _trace(solved_covariance)
        + quadratic
        - q_mean.rows
        + _logdet_spd(p_covariance)
        - _logdet_spd(q_covariance)
    ) / 2


def _gaussian_entropy(covariance: mp.matrix) -> mp.mpf:
    return (
        covariance.rows * (mp.mpf("1") + mp.log(2 * mp.pi)) + _logdet_spd(covariance)
    ) / 2


def _expected_affine_kl(
    law: Mapping[str, Any],
    q_moments: _Moments,
    path: MPSourcePathRecord,
    *,
    bank: Literal["model", "state"],
    receiver_t: int,
) -> mp.mpf:
    dimension = cast(int, law["dimension"])
    source_j = path.b[receiver_t - 1] if bank == "model" else path.a[receiver_t - 1]
    q_component = (
        law["q_model"][(receiver_t, source_j)]
        if bank == "model"
        else law["q_state"][
            (
                receiver_t,
                source_j,
                path.b[receiver_t - 1],
            )
        ]
    )
    p_component = law["p_components"][(bank, receiver_t, source_j)]
    difference = _zeros(dimension, 6 * dimension)
    source_indices = _block_indices(
        "m" if bank == "model" else "z",
        source_j,
        dimension,
    )
    parent_difference = q_component["parent_map"] - p_component["parent_map"]
    for row in range(dimension):
        for column, index in enumerate(source_indices):
            difference[row, index] = parent_difference[row, column]
    if bank == "state":
        model_indices = _block_indices("m", receiver_t, dimension)
        model_difference = q_component["model_map"] - p_component["model_map"]
        for row in range(dimension):
            for column, index in enumerate(model_indices):
                difference[row, index] = model_difference[row, column]
    displacement_mean = (
        difference * q_moments.mean + q_component["offset"] - p_component["offset"]
    )
    displacement_covariance = difference * q_moments.covariance * _transpose(difference)
    p_covariance = p_component["covariance"]
    q_covariance = q_component["covariance"]
    solved_q = _solve_left(p_covariance, q_covariance)
    solved_displacement_covariance = _solve_left(
        p_covariance,
        displacement_covariance,
    )
    solved_mean = _solve_left(p_covariance, displacement_mean)
    expected_quadratic = (
        _trace(solved_displacement_covariance)
        + (_transpose(displacement_mean) * solved_mean)[0]
    )
    return (
        _trace(solved_q)
        + expected_quadratic
        - dimension
        + _logdet_spd(p_covariance)
        - _logdet_spd(q_covariance)
    ) / 2


def _hstack(left: mp.matrix, right: mp.matrix) -> mp.matrix:
    if left.rows != right.rows:
        raise ValueError("horizontal stack row counts disagree")
    result = _zeros(left.rows, left.cols + right.cols)
    for row in range(left.rows):
        for column in range(left.cols):
            result[row, column] = left[row, column]
        for column in range(right.cols):
            result[row, left.cols + column] = right[row, column]
    return result


def _logsumexp(values: tuple[mp.mpf, ...]) -> mp.mpf:
    maximum = max(values)
    return maximum + mp.log(mp.fsum(mp.exp(item - maximum) for item in values))


def _emission_contrast_parameters(
    law: Mapping[str, Any],
    moments: _Moments,
    receiver_t: int,
) -> tuple[mp.matrix, mp.matrix, int]:
    decoder = law["decoders"][receiver_t - 1]
    dimension = cast(int, law["dimension"])
    indices = (
        *_block_indices("z", receiver_t, dimension),
        *_block_indices("m", receiver_t, dimension),
    )
    latent_mean = _subvector(moments.mean, indices)
    latent_covariance = _submatrix(moments.covariance, indices, indices)
    weight = _hstack(decoder["state_weight"], decoder["model_weight"])
    logits_mean = weight * latent_mean + decoder["bias"]
    logits_covariance = weight * latent_covariance * _transpose(weight)
    if logits_mean.rows != 3:
        raise ValueError("H7 contrast oracle requires exactly three logits")
    contrast = mp.matrix(
        (
            (mp.mpf("1"), mp.mpf("0"), mp.mpf("-1")),
            (mp.mpf("0"), mp.mpf("1"), mp.mpf("-1")),
        )
    )
    return (
        contrast * logits_mean,
        contrast * logits_covariance * _transpose(contrast),
        law["observation_labels"][receiver_t - 1],
    )


def _expected_log_emission(
    law: Mapping[str, Any],
    moments: _Moments,
    *,
    receiver_t: int,
    order: int,
) -> mp.mpf:
    contrast_mean, contrast_covariance, selected = _emission_contrast_parameters(
        law,
        moments,
        receiver_t,
    )
    factor = mp.cholesky(contrast_covariance)
    rule = standard_normal_gauss_hermite(order)
    result = mp.mpf("0")
    for first in range(order):
        for second in range(order):
            standard = mp.matrix(
                [
                    rule.standard_normal_nodes[first],
                    rule.standard_normal_nodes[second],
                ]
            )
            contrast = contrast_mean + factor * standard
            augmented = (contrast[0], contrast[1], mp.mpf("0"))
            selected_value = augmented[selected] - _logsumexp(augmented)
            result += rule.weights[first] * rule.weights[second] * selected_value
    return result


def _objective_values(
    law: Mapping[str, Any],
) -> _ObjectiveValues:
    paths = cast(tuple[MPSourcePathRecord, ...], law["source_paths"])
    q_moments = tuple(
        (path.path_id, _joint_moments(law, path, role="q")) for path in paths
    )
    p_moments = tuple(
        (path.path_id, _joint_moments(law, path, role="p")) for path in paths
    )
    q_by_path = dict(q_moments)
    p_by_path = dict(p_moments)
    values = {
        order: _objective_at_order(
            law,
            paths,
            q_by_path,
            p_by_path,
            order,
        )
        for order in (41, 51)
    }
    return _ObjectiveValues(
        scalars_41=tuple(values[41].items()),
        scalars_51=tuple(values[51].items()),
        q_moments=q_moments,
        p_moments=p_moments,
    )


def _objective_at_order(
    law: Mapping[str, Any],
    paths: tuple[MPSourcePathRecord, ...],
    q_by_path: Mapping[str, _Moments],
    p_by_path: Mapping[str, _Moments],
    order: int,
) -> dict[str, mp.mpf]:
    k0 = _gaussian_kl(
        law["q_initial"]["mean"],
        law["q_initial"]["covariance"],
        law["p_initial"]["mean"],
        law["p_initial"]["covariance"],
    )
    local = {term_id: mp.mpf("0") for term_id in H7_COMPLETE_LOCAL_TERM_IDS}
    monolithic = mp.mpf("0")
    source_entropy = mp.mpf("0")
    gaussian_entropy = mp.mpf("0")
    for path in paths:
        q_probability = _mp(path.q_probability)
        p_probability = _mp(path.p_probability)
        if q_probability <= 0:
            continue
        q_moments = q_by_path[path.path_id]
        p_moments = p_by_path[path.path_id]
        emissions: list[mp.mpf] = []
        factors = law["source_factors"][path.path_id]
        for receiver_t in (1, 2):
            emission = _expected_log_emission(
                law,
                q_moments,
                receiver_t=receiver_t,
                order=order,
            )
            emissions.append(emission)
            local[f"expected_log_emission[{receiver_t}]"] += q_probability * emission
            q_model, q_state, p_model, p_state = factors[receiver_t - 1]
            local[f"model_source_kl[{receiver_t}]"] += q_probability * (
                mp.log(p_model) - mp.log(q_model)
            )
            local[f"state_source_kl[{receiver_t}]"] += q_probability * (
                mp.log(p_state) - mp.log(q_state)
            )
            for bank in ("model", "state"):
                local[f"{bank}_transition_kl[{receiver_t}]"] -= (
                    q_probability
                    * _expected_affine_kl(
                        law,
                        q_moments,
                        path,
                        bank=cast(Literal["model", "state"], bank),
                        receiver_t=receiver_t,
                    )
                )
        source_entropy -= q_probability * mp.log(q_probability)
        gaussian_entropy += q_probability * _gaussian_entropy(q_moments.covariance)
        monolithic += q_probability * (
            -_gaussian_kl(
                q_moments.mean,
                q_moments.covariance,
                p_moments.mean,
                p_moments.covariance,
            )
            + mp.log(p_probability)
            - mp.log(q_probability)
            + mp.fsum(emissions)
        )
    local["joint_recognition_entropy"] = source_entropy + gaussian_entropy
    decomposed_terms = tuple(
        value
        for term_id, value in local.items()
        if term_id != "joint_recognition_entropy"
    )
    complete_local = -k0 + mp.fsum(decomposed_terms)
    return {
        "K0_joint_z0_m0": k0,
        **local,
        "complete_local_elbo": complete_local,
        "complete_monolithic_elbo": monolithic,
        "complete_local_monolithic_delta": abs(complete_local - monolithic),
    }


def _gaussian_inventory_records(
    prefix: str,
    mean: mp.matrix,
    covariance: mp.matrix,
    *,
    trial_id: str,
    precision_source: _MPPrecisionOperandSource,
    action_indices: tuple[int, ...],
    mean_category: Literal["vector", "offset"] = "vector",
) -> tuple[_MPInventoryOperand, ...]:
    """Materialize one Gaussian inventory from an approved precision row."""

    condition = _condition_spd(covariance)
    precision = precision_source.consume(
        trial_id=trial_id,
        gaussian_id=prefix,
        covariance=covariance,
    )
    precision_condition = _condition_spd(precision)
    information = precision * mean
    second_moment = covariance + _outer(mean, mean)
    return (
        _MPInventoryOperand(
            _value_record(f"{prefix}.mean", mean, vector=True),
            mean_category,
            "left",
            action_indices,
        ),
        _MPInventoryOperand(
            _value_record(
                f"{prefix}.covariance",
                covariance,
                condition_number=condition,
            ),
            "covariance",
            "covariance",
            action_indices,
        ),
        _MPInventoryOperand(
            _value_record(
                f"{prefix}.precision",
                precision,
                condition_number=precision_condition,
            ),
            "precision",
            "precision",
            action_indices,
        ),
        _MPInventoryOperand(
            _value_record(f"{prefix}.h_information", information, vector=True),
            "information",
            "information",
            action_indices,
        ),
        _MPInventoryOperand(
            _value_record(
                f"{prefix}.J_precision",
                precision,
                condition_number=precision_condition,
            ),
            "precision",
            "precision",
            action_indices,
        ),
        _MPInventoryOperand(
            _value_record(
                f"{prefix}.M_second_moment",
                second_moment,
                condition_number=_condition_spd(second_moment),
            ),
            "second_moment",
            "covariance",
            action_indices,
        ),
    )


def _collect_original_inventory(
    law: Mapping[str, Any],
    objective: _ObjectiveValues,
    *,
    prefix: str,
    trial_id: str,
    precision_source: _MPPrecisionOperandSource,
) -> tuple[_MPInventoryOperand, ...]:
    values: list[_MPInventoryOperand] = []
    initial_indices = (0, 0)
    values.extend(
        _gaussian_inventory_records(
            f"{prefix}.p.initial_joint",
            law["p_initial"]["mean"],
            law["p_initial"]["covariance"],
            trial_id=trial_id,
            precision_source=precision_source,
            action_indices=initial_indices,
        )
    )
    values.extend(
        _gaussian_inventory_records(
            f"{prefix}.q.initial_joint",
            law["q_initial"]["mean"],
            law["q_initial"]["covariance"],
            trial_id=trial_id,
            precision_source=precision_source,
            action_indices=initial_indices,
        )
    )
    for index, frame in enumerate(law["frames"]):
        values.append(
            _MPInventoryOperand(
                _value_record(f"{prefix}.U[{index}]", frame),
                "map",
                "left",
                (index,),
            )
        )
    links = {
        (receiver_t, source_j): _solve_right(
            law["frames"][receiver_t],
            law["frames"][source_j],
        )
        for receiver_t in range(3)
        for source_j in range(3)
        if receiver_t != source_j
    }
    for (receiver_t, source_j), link in links.items():
        values.append(
            _MPInventoryOperand(
                _value_record(
                    f"{prefix}.Omega[{receiver_t}<-{source_j}]",
                    link,
                ),
                "map",
                "receiver_source",
                (receiver_t, source_j),
            )
        )
    component_groups = (
        ("p", law["p_components"].values()),
        ("q_model", law["q_model"].values()),
        ("q_state", law["q_state"].values()),
    )
    for group, components in component_groups:
        for component in components:
            receiver_t = cast(int, component["receiver_t"])
            source_j = cast(int, component["source_j"])
            base = f"{prefix}.{group}.{component['component_id']}"
            values.append(
                _MPInventoryOperand(
                    _value_record(f"{base}.parent_map", component["parent_map"]),
                    "map",
                    "receiver_source",
                    (receiver_t, source_j),
                )
            )
            if component["model_map"] is not None:
                values.append(
                    _MPInventoryOperand(
                        _value_record(
                            f"{base}.B_model_map",
                            component["model_map"],
                        ),
                        "map",
                        "receiver_source",
                        (receiver_t, receiver_t),
                    )
                )
            values.extend(
                _gaussian_inventory_records(
                    f"{base}.receiver_offset",
                    component["offset"],
                    component["covariance"],
                    trial_id=trial_id,
                    precision_source=precision_source,
                    action_indices=(receiver_t,),
                    mean_category="offset",
                )
            )
    for decoder in law["decoders"]:
        receiver_t = cast(int, decoder["receiver_t"])
        values.extend(
            (
                _MPInventoryOperand(
                    _value_record(
                        f"{prefix}.decoder[{receiver_t}].state_weight",
                        decoder["state_weight"],
                    ),
                    "decoder",
                    "decoder",
                    (receiver_t,),
                ),
                _MPInventoryOperand(
                    _value_record(
                        f"{prefix}.decoder[{receiver_t}].model_weight",
                        decoder["model_weight"],
                    ),
                    "decoder",
                    "decoder",
                    (receiver_t,),
                ),
            )
        )
    if law["scorer_rows"]:
        first_row = law["scorer_rows"][0]
        for channel in ("z", "m"):
            for index, history in enumerate(first_row[f"{channel}_history"]):
                values.append(
                    _MPInventoryOperand(
                        _value_record(
                            f"{prefix}.source_scorer.{channel}_history[{index}]",
                            history,
                            vector=True,
                        ),
                        "vector",
                        "left",
                        (index,),
                    )
                )
        for row in law["scorer_rows"]:
            source_j = cast(int, row["source_j"])
            row_id = f"{row['bank']}[{row['receiver_t']}<-{source_j}]"
            for channel in ("z", "m"):
                values.append(
                    _MPInventoryOperand(
                        _value_record(
                            f"{prefix}.source_scorer.{row_id}.{channel}_covector",
                            row[f"{channel}_covector"],
                            vector=True,
                        ),
                        "information",
                        "information",
                        (source_j,),
                    )
                )
    global_indices = tuple(index for index in range(3) for _channel in range(2))
    for role, moments_by_path in (
        ("q", objective.q_moments),
        ("p", objective.p_moments),
    ):
        for path_id, moments in moments_by_path:
            values.extend(
                _gaussian_inventory_records(
                    f"{prefix}.{role}.global[{path_id}]",
                    moments.mean,
                    moments.covariance,
                    trial_id=trial_id,
                    precision_source=precision_source,
                    action_indices=global_indices,
                )
            )
    frozen = tuple(values)
    _require_backward_inventory(
        tuple(item.record for item in frozen),
        law,
        prefix,
    )
    return frozen


def _record_matrix(record: MPValueRecord) -> mp.matrix:
    values = tuple(_mp(item) for item in record.decimal_values)
    if len(record.shape) == 1:
        if len(values) != record.shape[0]:
            raise ValueError("vector record shape/value count changed")
        return mp.matrix(values)
    if len(record.shape) == 2:
        rows, columns = record.shape
        if len(values) != rows * columns:
            raise ValueError("matrix record shape/value count changed")
        return mp.matrix(
            [values[row * columns : (row + 1) * columns] for row in range(rows)]
        )
    raise ValueError("inventory record must be a vector or matrix")


def _identity_matrix(dimension: int) -> mp.matrix:
    result = _zeros(dimension, dimension)
    for index in range(dimension):
        result[index, index] = mp.mpf("1")
    return result


def _block_action(
    actions: tuple[mp.matrix, mp.matrix, mp.matrix],
    indices: tuple[int, ...],
) -> mp.matrix:
    if not indices:
        raise ValueError("inventory operand has no action indices")
    return _block_diag(*(actions[index] for index in indices))


def _effective_operand_actions(
    operand: _MPInventoryOperand,
    actions: tuple[mp.matrix, mp.matrix, mp.matrix],
    *,
    decoder_policy: Literal["transform", "fixed"],
) -> tuple[mp.matrix, ...]:
    if operand.transform_kind == "receiver_source":
        if len(operand.action_indices) != 2:
            raise ValueError("receiver/source inventory action must name two fibers")
        return tuple(actions[index] for index in operand.action_indices)
    block = _block_action(actions, operand.action_indices)
    if operand.transform_kind == "decoder" and decoder_policy == "fixed":
        block = _identity_matrix(block.rows)
    if operand.transform_kind in ("covariance", "precision"):
        return (block, block)
    return (block,)


def _inventory_value_record(
    operand: _MPInventoryOperand,
    value: mp.matrix,
) -> MPValueRecord:
    condition = (
        _condition_spd(value)
        if operand.category in ("covariance", "precision", "second_moment")
        else None
    )
    return _value_record(
        operand.record.operand_id,
        value,
        vector=len(operand.record.shape) == 1,
        condition_number=condition,
    )


def _transform_inventory(
    inventory: tuple[_MPInventoryOperand, ...],
    actions: tuple[mp.matrix, mp.matrix, mp.matrix],
    *,
    decoder_policy: Literal["transform", "fixed"],
) -> tuple[_MPInventoryOperand, ...]:
    transformed: list[_MPInventoryOperand] = []
    for operand in inventory:
        source = _record_matrix(operand.record)
        effective_actions = _effective_operand_actions(
            operand,
            actions,
            decoder_policy=decoder_policy,
        )
        if operand.transform_kind == "left":
            value = effective_actions[0] * source
        elif operand.transform_kind == "covariance":
            value = _congruence(effective_actions[0], source)
        elif operand.transform_kind == "precision":
            value = _solve_right(
                _solve_left(_transpose(effective_actions[0]), source),
                effective_actions[1],
            )
        elif operand.transform_kind == "information":
            value = _solve_left(_transpose(effective_actions[0]), source)
        elif operand.transform_kind == "receiver_source":
            value = _solve_right(
                effective_actions[0] * source,
                effective_actions[1],
            )
        elif operand.transform_kind == "decoder":
            if decoder_policy == "fixed":
                value = source * effective_actions[0]
            else:
                value = _solve_right(source, effective_actions[0])
        else:  # pragma: no cover - closed Literal defense
            raise ValueError("unsupported inventory forward transform")
        transformed.append(
            _MPInventoryOperand(
                _inventory_value_record(operand, value),
                operand.category,
                operand.transform_kind,
                operand.action_indices,
            )
        )
    return tuple(transformed)


def _recover_inventory(
    transformed: tuple[_MPInventoryOperand, ...],
    actions: tuple[mp.matrix, mp.matrix, mp.matrix],
    *,
    decoder_policy: Literal["transform", "fixed"],
) -> tuple[_MPInventoryOperand, ...]:
    recovered: list[_MPInventoryOperand] = []
    for operand in transformed:
        source = _record_matrix(operand.record)
        effective_actions = _effective_operand_actions(
            operand,
            actions,
            decoder_policy=decoder_policy,
        )
        if operand.transform_kind == "left":
            value = _solve_left(effective_actions[0], source)
        elif operand.transform_kind == "covariance":
            value = _recover_covariance(source, effective_actions[0])
        elif operand.transform_kind == "precision":
            value = _transpose(effective_actions[0]) * source * effective_actions[1]
        elif operand.transform_kind == "information":
            value = _transpose(effective_actions[0]) * source
        elif operand.transform_kind == "receiver_source":
            value = _solve_left(effective_actions[0], source) * effective_actions[1]
        elif operand.transform_kind == "decoder":
            value = source * effective_actions[0]
        else:  # pragma: no cover - closed Literal defense
            raise ValueError("unsupported inventory inverse transform")
        recovered.append(
            _MPInventoryOperand(
                _inventory_value_record(operand, value),
                operand.category,
                operand.transform_kind,
                operand.action_indices,
            )
        )
    return tuple(recovered)


def _inventory_values(
    inventory: tuple[_MPInventoryOperand, ...],
) -> MPLawValues:
    return MPLawValues(tuple(item.record for item in inventory))


def _require_backward_inventory(
    values: tuple[MPValueRecord, ...],
    law: Mapping[str, Any],
    prefix: str,
) -> None:
    operand_ids = tuple(item.operand_id for item in values)
    if len(set(operand_ids)) != len(operand_ids):
        raise ValueError("backward operand IDs must be unique")
    expected: list[str] = [
        *_gaussian_operand_ids(f"{prefix}.p.initial_joint"),
        *_gaussian_operand_ids(f"{prefix}.q.initial_joint"),
        *(f"{prefix}.U[{index}]" for index in range(3)),
        *(
            f"{prefix}.Omega[{receiver}<-{source}]"
            for receiver in range(3)
            for source in range(3)
            if receiver != source
        ),
    ]
    for group, components in (
        ("p", law["p_components"].values()),
        ("q_model", law["q_model"].values()),
        ("q_state", law["q_state"].values()),
    ):
        for component in components:
            base = f"{prefix}.{group}.{component['component_id']}"
            expected.append(f"{base}.parent_map")
            if component["model_map"] is not None:
                expected.append(f"{base}.B_model_map")
            expected.extend(_gaussian_operand_ids(f"{base}.receiver_offset"))
    for decoder in law["decoders"]:
        receiver_t = decoder["receiver_t"]
        expected.extend(
            (
                f"{prefix}.decoder[{receiver_t}].state_weight",
                f"{prefix}.decoder[{receiver_t}].model_weight",
            )
        )
    if law["scorer_rows"]:
        expected.extend(
            f"{prefix}.source_scorer.{channel}_history[{index}]"
            for channel in ("z", "m")
            for index in range(2)
        )
        expected.extend(
            f"{prefix}.source_scorer.{row['bank']}"
            f"[{row['receiver_t']}<-{row['source_j']}].{channel}_covector"
            for row in law["scorer_rows"]
            for channel in ("z", "m")
        )
    expected.extend(
        operand_id
        for role in ("q", "p")
        for path in law["source_paths"]
        for operand_id in _gaussian_operand_ids(
            f"{prefix}.{role}.global[{path.path_id}]"
        )
    )
    if operand_ids != tuple(expected):
        raise ValueError(
            "backward operand IDs are missing, extra, duplicated, or reordered"
        )
    banned_fragments = (
        ".bias",
        ".support",
        ".probability",
        ".observation_label",
    )
    if any(
        fragment in operand_id
        for operand_id in operand_ids
        for fragment in banned_fragments
    ):
        raise ValueError("exact-identity fields cannot enter normalized r_back")


def _gaussian_operand_ids(prefix: str) -> tuple[str, ...]:
    return (
        f"{prefix}.mean",
        f"{prefix}.covariance",
        f"{prefix}.precision",
        f"{prefix}.h_information",
        f"{prefix}.J_precision",
        f"{prefix}.M_second_moment",
    )


def _condition_general(value: mp.matrix) -> mp.mpf:
    gram = _transpose(value) * value
    eigenvalues, _ = mp.eigsy(gram)
    smallest = mp.mpf(eigenvalues[0])
    largest = mp.mpf(eigenvalues[gram.rows - 1])
    if smallest <= 0:
        raise ValueError("action operand is singular")
    return max(mp.mpf("1"), mp.sqrt(largest / smallest))


def _h7_operand(
    record: MPValueRecord,
    *,
    operand_id: str,
    category: _MPLeafCategory | Literal["backward"],
    role: Literal["original", "transformed", "reference", "recovered"],
) -> H7OperandRecord:
    return H7OperandRecord.create(
        operand_id=operand_id,
        category=category,
        role=role,
        dtype="mpmath-decimal-100",
        shape=record.shape,
        value_sha256=record.value_sha256,
        scale=float(_mp(record.scale)),
        condition_number=float(_mp(record.condition_number)),
        normalization=float(_mp(record.normalization)),
        oracle_value=None,
    )


def _action_value_record(operand_id: str, action: mp.matrix) -> MPValueRecord:
    return _value_record(
        operand_id,
        action,
        condition_number=_condition_general(action),
    )


def _operation_kinds(
    operand: _MPInventoryOperand,
    *,
    decoder_policy: Literal["transform", "fixed"],
) -> tuple[
    Literal["matrix_product", "direct_solve"],
    Literal["matrix_product", "direct_solve"],
]:
    if operand.transform_kind in ("left", "covariance"):
        return "matrix_product", "direct_solve"
    if operand.transform_kind in ("precision", "information"):
        return "direct_solve", "matrix_product"
    if operand.transform_kind == "receiver_source":
        return "direct_solve", "direct_solve"
    if operand.transform_kind == "decoder":
        if decoder_policy == "fixed":
            return "matrix_product", "matrix_product"
        return "direct_solve", "matrix_product"
    raise ValueError("unsupported inventory budget transform")


def _build_leaf_budget(
    *,
    phase: Literal["forward", "inverse"],
    inventory_operand: _MPInventoryOperand,
    source: MPValueRecord,
    result: MPValueRecord,
    effective_actions: tuple[mp.matrix, ...],
    operation_kind: Literal["matrix_product", "direct_solve"],
) -> H7BoundBudget:
    prefix = f"backward:{inventory_operand.record.operand_id}:{phase}"
    original_id = f"{prefix}.input"
    transformed_id = f"{prefix}.output"
    source_id = f"{prefix}.source"
    action_ids = tuple(
        f"{prefix}.action[{index}]" for index in range(len(effective_actions))
    )
    operands = (
        _h7_operand(
            source,
            operand_id=original_id,
            category=inventory_operand.category,
            role="original",
        ),
        _h7_operand(
            result,
            operand_id=transformed_id,
            category=inventory_operand.category,
            role="transformed",
        ),
        _h7_operand(
            source,
            operand_id=source_id,
            category=inventory_operand.category,
            role="reference",
        ),
        *tuple(
            _h7_operand(
                _action_value_record(action_id, action),
                operand_id=action_id,
                category=inventory_operand.category,
                role="reference",
            )
            for action_id, action in zip(
                action_ids,
                effective_actions,
                strict=True,
            )
        ),
    )
    spd_operand_ids = (
        (source_id,)
        if inventory_operand.category in ("covariance", "precision", "second_moment")
        else ()
    )
    formula = H7BudgetFormula(
        category=inventory_operand.category,
        operation_kind=operation_kind,
        dimension_operand_id=source_id,
        compared_operand_ids=(original_id, transformed_id),
        source_operand_ids=(source_id,),
        direct_action_operand_ids=action_ids,
        spd_operand_ids=spd_operand_ids,
        frame_operand_ids=(),
        link_operand_ids=(),
        signed_summand_operand_ids=(),
        child_budgets=(),
        forward_budget=None,
        inverse_action_budget=None,
    )
    return build_h7_budget(
        invariant_id=prefix,
        category=inventory_operand.category,
        operands=operands,
        formula=formula,
    )


def _build_bound_backward_records(
    original: tuple[_MPInventoryOperand, ...],
    transformed: tuple[_MPInventoryOperand, ...],
    recovered: tuple[_MPInventoryOperand, ...],
    actions: tuple[mp.matrix, mp.matrix, mp.matrix],
    *,
    decoder_policy: Literal["transform", "fixed"],
) -> H7BackwardBudgetAggregate:
    original_ids = tuple(item.record.operand_id for item in original)
    if original_ids != tuple(
        item.record.operand_id for item in transformed
    ) or original_ids != tuple(item.record.operand_id for item in recovered):
        raise ValueError("original/transformed/recovered backward order changed")
    inputs: list[H7BackwardOperandInput] = []
    for original_item, transformed_item, recovered_item in zip(
        original,
        transformed,
        recovered,
        strict=True,
    ):
        if (
            original_item.category != transformed_item.category
            or original_item.category != recovered_item.category
            or original_item.record.shape != transformed_item.record.shape
            or original_item.record.shape != recovered_item.record.shape
        ):
            raise ValueError("backward operand category/shape changed")
        effective_actions = _effective_operand_actions(
            original_item,
            actions,
            decoder_policy=decoder_policy,
        )
        forward_kind, inverse_kind = _operation_kinds(
            original_item,
            decoder_policy=decoder_policy,
        )
        forward_budget = _build_leaf_budget(
            phase="forward",
            inventory_operand=original_item,
            source=original_item.record,
            result=transformed_item.record,
            effective_actions=effective_actions,
            operation_kind=forward_kind,
        )
        inverse_budget = _build_leaf_budget(
            phase="inverse",
            inventory_operand=original_item,
            source=transformed_item.record,
            result=recovered_item.record,
            effective_actions=effective_actions,
            operation_kind=inverse_kind,
        )
        original_values = tuple(
            _mp(item) for item in original_item.record.decimal_values
        )
        recovered_values = tuple(
            _mp(item) for item in recovered_item.record.decimal_values
        )
        numerator = mp.sqrt(
            mp.fsum(
                (recovered_value - original_value) ** 2
                for original_value, recovered_value in zip(
                    original_values,
                    recovered_values,
                    strict=True,
                )
            )
        )
        normalization = max(
            mp.mpf("1"),
            mp.sqrt(mp.fsum(value**2 for value in original_values)),
        )
        prefix = f"backward:{original_item.record.operand_id}"
        original_id = f"{prefix}.original"
        transformed_id = f"{prefix}.transformed"
        recovered_id = f"{prefix}.recovered"
        action_ids = tuple(
            f"{prefix}.action[{index}]" for index in range(len(effective_actions))
        )
        backward_operands = (
            _h7_operand(
                original_item.record,
                operand_id=original_id,
                category="backward",
                role="original",
            ),
            _h7_operand(
                transformed_item.record,
                operand_id=transformed_id,
                category="backward",
                role="transformed",
            ),
            _h7_operand(
                recovered_item.record,
                operand_id=recovered_id,
                category="backward",
                role="recovered",
            ),
            *tuple(
                _h7_operand(
                    _action_value_record(action_id, action),
                    operand_id=action_id,
                    category="backward",
                    role="reference",
                )
                for action_id, action in zip(
                    action_ids,
                    effective_actions,
                    strict=True,
                )
            ),
        )
        formula = H7BudgetFormula(
            category="backward",
            operation_kind="direct_solve",
            dimension_operand_id=None,
            compared_operand_ids=(
                original_id,
                transformed_id,
                recovered_id,
            ),
            source_operand_ids=(),
            direct_action_operand_ids=action_ids,
            spd_operand_ids=(),
            frame_operand_ids=(),
            link_operand_ids=(),
            signed_summand_operand_ids=(),
            child_budgets=(),
            forward_budget=forward_budget,
            inverse_action_budget=inverse_budget,
        )
        inputs.append(
            H7BackwardOperandInput(
                operand_id=original_item.record.operand_id,
                original_sha256=original_item.record.value_sha256,
                transformed_sha256=transformed_item.record.value_sha256,
                recovered_sha256=recovered_item.record.value_sha256,
                numerator=float(numerator),
                normalization=float(normalization),
                operands=backward_operands,
                formula=formula,
            )
        )
    frozen_inputs = tuple(inputs)
    return build_h7_backward_records(
        frozen_inputs,
        required_operand_ids=tuple(item.operand_id for item in frozen_inputs),
    )


def _gaussian_logpdf(
    value: mp.matrix,
    mean: mp.matrix,
    covariance: mp.matrix,
) -> mp.mpf:
    if value.rows != mean.rows or value.cols != 1:
        raise ValueError("density value and mean shapes disagree")
    displacement = value - mean
    solved = _solve_left(covariance, displacement)
    quadratic = (_transpose(displacement) * solved)[0]
    return -(value.rows * mp.log(2 * mp.pi) + _logdet_spd(covariance) + quadratic) / 2


def _point_log_likelihood(
    law: Mapping[str, Any],
    value: mp.matrix,
) -> mp.mpf:
    dimension = cast(int, law["dimension"])
    terms: list[mp.mpf] = []
    for receiver_t in (1, 2):
        decoder = law["decoders"][receiver_t - 1]
        z = _subvector(value, _block_indices("z", receiver_t, dimension))
        m = _subvector(value, _block_indices("m", receiver_t, dimension))
        logits = (
            decoder["state_weight"] * z + decoder["model_weight"] * m + decoder["bias"]
        )
        selected = law["observation_labels"][receiver_t - 1]
        values = tuple(logits[index] for index in range(logits.rows))
        terms.append(values[selected] - _logsumexp(values))
    return mp.fsum(terms)


def _global_log_density(
    law: Mapping[str, Any],
    moments: _Moments,
    path: MPSourcePathRecord,
    value: mp.matrix,
    *,
    role: Literal["p", "q"],
) -> mp.mpf:
    probability = _mp(path.p_probability if role == "p" else path.q_probability)
    result = mp.log(probability) + _gaussian_logpdf(
        value,
        moments.mean,
        moments.covariance,
    )
    if role == "p":
        result += _point_log_likelihood(law, value)
    return result


def _component_gaussian(
    law: Mapping[str, Any],
    component_id: str,
) -> tuple[mp.matrix, mp.matrix]:
    family_prefix = _family_prefix(law["recognition_family"])
    if component_id == "p.initial_joint":
        initial = law["p_initial"]
        return initial["mean"], initial["covariance"]
    if component_id == f"q.{family_prefix}.initial_joint":
        initial = law["q_initial"]
        return initial["mean"], initial["covariance"]
    components = (
        *law["p_components"].values(),
        *law["q_model"].values(),
        *law["q_state"].values(),
    )
    matches = tuple(
        component
        for component in components
        if component["component_id"] == component_id
    )
    if len(matches) != 1:
        raise ValueError(f"density component {component_id!r} is ambiguous")
    return matches[0]["offset"], matches[0]["covariance"]


def _probe_scope_shift(record: Mapping[str, Any]) -> mp.mpf:
    values = (
        _mp(record["initial_log_jacobian_shift"]),
        _mp(record["receiver_log_jacobian_shift"]),
        _mp(record["global_log_jacobian_shift"]),
    )
    nonzero = tuple(item for item in values if item != 0)
    if len(nonzero) > 1:
        raise ValueError("density probe mixes Jacobian scopes")
    return mp.mpf("0") if not nonzero else nonzero[0]


def _probe_evaluations(
    original: Mapping[str, Any],
    transformed: Mapping[str, Any],
    original_objective: _ObjectiveValues,
    transformed_objective: _ObjectiveValues,
    records: list[dict[str, Any]],
    *,
    action_profile: str,
) -> tuple[tuple[MPProbeEvaluationRecord, ...], dict[str, mp.mpf]]:
    family_prefix = _family_prefix(original["recognition_family"])
    original_q = dict(original_objective.q_moments)
    original_p = dict(original_objective.p_moments)
    transformed_q = dict(transformed_objective.q_moments)
    transformed_p = dict(transformed_objective.p_moments)
    evaluations: list[MPProbeEvaluationRecord] = []
    maxima = {"p": mp.mpf("0"), "q": mp.mpf("0"), "log_ratio": mp.mpf("0")}
    for record in records:
        probe_id = cast(str, record["probe_id"])
        component_id = cast(str, record["component_id"])
        if not probe_id.startswith(f"{action_profile}:"):
            continue
        if component_id.startswith("q.") and not component_id.startswith(
            f"q.{family_prefix}."
        ):
            continue
        x = _vector(record["x"])
        x_prime = _vector(record["x_prime"])
        shift = _probe_scope_shift(record)
        observations: list[tuple[str, str, str, str, str]] = []
        if component_id.endswith(".global"):
            path_id = cast(str, record["source_id"])
            path = next(
                item for item in original["source_paths"] if item.path_id == path_id
            )
            original_values: dict[str, mp.mpf] = {}
            transformed_values: dict[str, mp.mpf] = {}
            for role in ("p", "q"):
                original_values[role] = _global_log_density(
                    original,
                    original_p[path_id] if role == "p" else original_q[path_id],
                    path,
                    x,
                    role=cast(Literal["p", "q"], role),
                )
                transformed_values[role] = _global_log_density(
                    transformed,
                    transformed_p[path_id] if role == "p" else transformed_q[path_id],
                    path,
                    x_prime,
                    role=cast(Literal["p", "q"], role),
                )
                residual = abs(transformed_values[role] - original_values[role] + shift)
                maxima[role] = max(maxima[role], residual)
                observations.append(
                    (
                        role,
                        _decimal(original_values[role]),
                        _decimal(transformed_values[role]),
                        _decimal(-shift),
                        _decimal(residual),
                    )
                )
            original_ratio = original_values["p"] - original_values["q"]
            transformed_ratio = transformed_values["p"] - transformed_values["q"]
            ratio_residual = abs(transformed_ratio - original_ratio)
            maxima["log_ratio"] = max(maxima["log_ratio"], ratio_residual)
            observations.append(
                (
                    "log_ratio",
                    _decimal(original_ratio),
                    _decimal(transformed_ratio),
                    "0.0",
                    _decimal(ratio_residual),
                )
            )
        else:
            role = "p" if component_id.startswith("p.") else "q"
            original_mean, original_covariance = _component_gaussian(
                original,
                component_id,
            )
            transformed_mean, transformed_covariance = _component_gaussian(
                transformed,
                component_id,
            )
            original_value = _gaussian_logpdf(
                x,
                original_mean,
                original_covariance,
            )
            transformed_value = _gaussian_logpdf(
                x_prime,
                transformed_mean,
                transformed_covariance,
            )
            residual = abs(transformed_value - original_value + shift)
            maxima[role] = max(maxima[role], residual)
            observations.append(
                (
                    role,
                    _decimal(original_value),
                    _decimal(transformed_value),
                    _decimal(-shift),
                    _decimal(residual),
                )
            )
        evaluations.append(
            MPProbeEvaluationRecord(
                probe_id=probe_id,
                component_id=component_id,
                source_id=cast(str, record["source_id"]),
                observations=tuple(observations),
            )
        )
    if not evaluations:
        raise ValueError("matrix trial has no corresponding density probes")
    return tuple(evaluations), maxima


def _scalar_probe_evaluations(
    original: Mapping[str, Any],
    transformed: Mapping[str, Any],
    original_objective: _ObjectiveValues,
    transformed_objective: _ObjectiveValues,
    records: list[dict[str, Any]],
    *,
    trial_id: str,
) -> tuple[tuple[MPProbeEvaluationRecord, ...], dict[str, mp.mpf]]:
    original_q = dict(original_objective.q_moments)
    original_p = dict(original_objective.p_moments)
    transformed_q = dict(transformed_objective.q_moments)
    transformed_p = dict(transformed_objective.p_moments)
    evaluations: list[MPProbeEvaluationRecord] = []
    maxima = {"p": mp.mpf("0"), "q": mp.mpf("0"), "log_ratio": mp.mpf("0")}
    matching_records = tuple(
        record
        for record in records
        if cast(str, record["probe_id"]).startswith(f"{trial_id}:")
    )
    if len(matching_records) != len(original["source_paths"]):
        raise ValueError("scalar trial probe inventory is missing or duplicated")
    for record, path in zip(
        matching_records,
        original["source_paths"],
        strict=True,
    ):
        if record["source_id"] != path.path_id:
            raise ValueError("scalar probe/source-path order changed")
        x = _vector(record["x"])
        x_prime = _vector(record["x_prime"])
        shift = _mp(record["global_log_jacobian_shift"])
        observations: list[tuple[str, str, str, str, str]] = []
        role_values: dict[str, tuple[mp.mpf, mp.mpf]] = {}
        for role in ("p", "q"):
            original_value = _global_log_density(
                original,
                original_p[path.path_id] if role == "p" else original_q[path.path_id],
                path,
                x,
                role=cast(Literal["p", "q"], role),
            )
            transformed_value = _global_log_density(
                transformed,
                transformed_p[path.path_id]
                if role == "p"
                else transformed_q[path.path_id],
                path,
                x_prime,
                role=cast(Literal["p", "q"], role),
            )
            residual = abs(transformed_value - original_value + shift)
            maxima[role] = max(maxima[role], residual)
            role_values[role] = (original_value, transformed_value)
            observations.append(
                (
                    role,
                    _decimal(original_value),
                    _decimal(transformed_value),
                    _decimal(-shift),
                    _decimal(residual),
                )
            )
        original_ratio = role_values["p"][0] - role_values["q"][0]
        transformed_ratio = role_values["p"][1] - role_values["q"][1]
        ratio_residual = abs(transformed_ratio - original_ratio)
        maxima["log_ratio"] = max(maxima["log_ratio"], ratio_residual)
        observations.append(
            (
                "log_ratio",
                _decimal(original_ratio),
                _decimal(transformed_ratio),
                "0.0",
                _decimal(ratio_residual),
            )
        )
        evaluations.append(
            MPProbeEvaluationRecord(
                probe_id=cast(str, record["probe_id"]),
                component_id=cast(str, record["component_id"]),
                source_id=path.path_id,
                observations=tuple(observations),
            )
        )
    return tuple(evaluations), maxima


def _joint_contrast_parameters(
    law: Mapping[str, Any],
    moments: _Moments,
) -> tuple[mp.matrix, mp.matrix, tuple[int, int]]:
    dimension = cast(int, law["dimension"])
    total = 6 * dimension
    linear = _zeros(4, total)
    bias = _zeros(4, 1)
    selected: list[int] = []
    contrast = mp.matrix(
        (
            (mp.mpf("1"), mp.mpf("0"), mp.mpf("-1")),
            (mp.mpf("0"), mp.mpf("1"), mp.mpf("-1")),
        )
    )
    for receiver_t in (1, 2):
        decoder = law["decoders"][receiver_t - 1]
        combined = _hstack(
            decoder["state_weight"],
            decoder["model_weight"],
        )
        contrast_weight = contrast * combined
        contrast_bias = contrast * decoder["bias"]
        indices = (
            *_block_indices("z", receiver_t, dimension),
            *_block_indices("m", receiver_t, dimension),
        )
        row_offset = 2 * (receiver_t - 1)
        for row in range(2):
            bias[row_offset + row] = contrast_bias[row]
            for column, index in enumerate(indices):
                linear[row_offset + row, index] = contrast_weight[row, column]
        selected.append(law["observation_labels"][receiver_t - 1])
    mean = linear * moments.mean + bias
    covariance = linear * moments.covariance * _transpose(linear)
    return mean, covariance, cast(tuple[int, int], tuple(selected))


def _emission_probability_product(
    contrasts: mp.matrix,
    selected: tuple[int, int],
) -> mp.mpf:
    result = mp.mpf("1")
    for receiver_t in (0, 1):
        values = (
            contrasts[2 * receiver_t],
            contrasts[2 * receiver_t + 1],
            mp.mpf("0"),
        )
        result *= mp.exp(values[selected[receiver_t]] - _logsumexp(values))
    return result


def _scalar_evidence(
    law: Mapping[str, Any],
    *,
    order: int,
) -> mp.mpf:
    rule = standard_normal_gauss_hermite(order)
    total = mp.mpf("0")
    for path in law["source_paths"]:
        moments = _joint_moments(law, path, role="p")
        contrast_mean, contrast_covariance, selected = _joint_contrast_parameters(
            law,
            moments,
        )
        factor = mp.cholesky(contrast_covariance)
        expectation = mp.mpf("0")
        for indices in itertools.product(range(order), repeat=4):
            standard = mp.matrix(
                [rule.standard_normal_nodes[index] for index in indices]
            )
            weight = mp.fprod(rule.weights[index] for index in indices)
            expectation += weight * _emission_probability_product(
                contrast_mean + factor * standard,
                selected,
            )
        total += _mp(path.p_probability) * expectation
    if total <= 0:
        raise ValueError("scalar evidence must be positive")
    return mp.log(total)


def _scalar_posterior_kl(
    law: Mapping[str, Any],
    *,
    log_evidence: mp.mpf,
    order: int,
) -> mp.mpf:
    """Integrate ``E_q[log q - log p(z,path|x)]`` directly.

    This is intentionally independent of the evidence-minus-ELBO identity;
    that identity is recorded later as a separate residual.
    """

    contributions: list[mp.mpf] = []
    for path in law["source_paths"]:
        q_probability = _mp(path.q_probability)
        if q_probability <= 0:
            continue
        p_probability = _mp(path.p_probability)
        q_moments = _joint_moments(law, path, role="q")
        p_moments = _joint_moments(law, path, role="p")
        expected_log_likelihood = mp.fsum(
            _expected_log_emission(
                law,
                q_moments,
                receiver_t=receiver_t,
                order=order,
            )
            for receiver_t in (1, 2)
        )
        contributions.append(
            q_probability
            * (
                mp.log(q_probability)
                - mp.log(p_probability)
                + _gaussian_kl(
                    q_moments.mean,
                    q_moments.covariance,
                    p_moments.mean,
                    p_moments.covariance,
                )
                - expected_log_likelihood
                + log_evidence
            )
        )
    return mp.fsum(contributions)


def _moments_only(law: Mapping[str, Any]) -> _ObjectiveValues:
    paths = law["source_paths"]
    return _ObjectiveValues(
        scalars_41=(),
        scalars_51=(),
        q_moments=tuple(
            (path.path_id, _joint_moments(law, path, role="q")) for path in paths
        ),
        p_moments=tuple(
            (path.path_id, _joint_moments(law, path, role="p")) for path in paths
        ),
    )


def _global_log_jacobian(
    actions: tuple[mp.matrix, mp.matrix, mp.matrix],
) -> mp.mpf:
    return mp.fsum(2 * mp.log(mp.det(item)) for item in actions)


def _pair_scalar_items(
    original: _ObjectiveValues,
    transformed: _ObjectiveValues,
    actions: tuple[mp.matrix, mp.matrix, mp.matrix],
    density_maxima: Mapping[str, mp.mpf],
    *,
    prefix: str,
) -> list[tuple[str, str]]:
    original_41 = original.values_41
    original_51 = original.values_51
    transformed_41 = transformed.values_41
    transformed_51 = transformed.values_51
    items: list[tuple[str, str]] = []
    name_prefix = f"{prefix}." if prefix else ""
    log_jacobian = _global_log_jacobian(actions)
    items.extend(
        (
            (f"{name_prefix}logJ_G", _decimal(log_jacobian)),
            (
                f"{name_prefix}initial_log_jacobian_shift",
                _decimal(2 * mp.log(mp.det(actions[0]))),
            ),
            (
                f"{name_prefix}receiver_log_jacobian_shift[1]",
                _decimal(mp.log(mp.det(actions[1]))),
            ),
            (
                f"{name_prefix}receiver_log_jacobian_shift[2]",
                _decimal(mp.log(mp.det(actions[2]))),
            ),
        )
    )
    for name, original_value in original_51.items():
        transformed_value = transformed_51[name]
        expected_shift = (
            log_jacobian if name == "joint_recognition_entropy" else mp.mpf("0")
        )
        items.extend(
            (
                (f"{name_prefix}{name}", _decimal(original_value)),
                (
                    f"{name_prefix}transformed.{name}",
                    _decimal(transformed_value),
                ),
                (
                    f"{name_prefix}residual.{name}",
                    _decimal(abs(transformed_value - original_value - expected_shift)),
                ),
            )
        )
    for receiver_t in (1, 2):
        name = f"expected_log_emission[{receiver_t}]"
        items.extend(
            (
                (
                    f"{name_prefix}gh41.{name}",
                    _decimal(original_41[name]),
                ),
                (
                    f"{name_prefix}gh51.{name}",
                    _decimal(original_51[name]),
                ),
                (
                    f"{name_prefix}gh41_gh51_delta.{name}",
                    _decimal(abs(original_51[name] - original_41[name])),
                ),
                (
                    f"{name_prefix}transformed.gh41.{name}",
                    _decimal(transformed_41[name]),
                ),
                (
                    f"{name_prefix}transformed.gh51.{name}",
                    _decimal(transformed_51[name]),
                ),
                (
                    f"{name_prefix}transformed.gh41_gh51_delta.{name}",
                    _decimal(abs(transformed_51[name] - transformed_41[name])),
                ),
            )
        )
    items.extend(
        (
            (
                f"{name_prefix}complete_pointwise_p_density_shift",
                _decimal(density_maxima["p"]),
            ),
            (
                f"{name_prefix}complete_pointwise_q_density_shift",
                _decimal(density_maxima["q"]),
            ),
            (
                f"{name_prefix}complete_pointwise_log_ratio",
                _decimal(density_maxima["log_ratio"]),
            ),
        )
    )
    return items


def _scorer_record_pairs(
    original_rows: tuple[dict[str, Any], ...],
    transformed_rows: tuple[dict[str, Any], ...],
) -> tuple[MPScorerRowRecord, ...]:
    if len(original_rows) != len(transformed_rows):
        raise ValueError("scorer row inventories disagree")
    records: list[MPScorerRowRecord] = []
    for original, transformed in zip(
        original_rows,
        transformed_rows,
        strict=True,
    ):
        if (
            original["bank"],
            original["receiver_t"],
            original["source_j"],
        ) != (
            transformed["bank"],
            transformed["receiver_t"],
            transformed["source_j"],
        ):
            raise ValueError("scorer row identities changed")
        records.append(
            MPScorerRowRecord(
                bank=original["bank"],
                receiver_t=original["receiver_t"],
                source_j=original["source_j"],
                prefix_tokens=original["prefix_tokens"],
                prefix_term=_decimal(original["prefix_term"]),
                support=original["support"],
                mask=original["mask"],
                original_probability=_decimal(original["probability"]),
                transformed_probability=_decimal(transformed["probability"]),
                original_z_covector=tuple(
                    _decimal(original["z_covector"][index])
                    for index in range(original["z_covector"].rows)
                ),
                transformed_z_covector=tuple(
                    _decimal(transformed["z_covector"][index])
                    for index in range(transformed["z_covector"].rows)
                ),
                original_m_covector=tuple(
                    _decimal(original["m_covector"][index])
                    for index in range(original["m_covector"].rows)
                ),
                transformed_m_covector=tuple(
                    _decimal(transformed["m_covector"][index])
                    for index in range(transformed["m_covector"].rows)
                ),
                original_raw_score=_decimal(original["raw_score"]),
                transformed_raw_score=_decimal(transformed["raw_score"]),
            )
        )
    return tuple(records)


def _evaluate_scalar_trial(
    trial_id: str,
    data: Mapping[str, Any],
    paths: tuple[MPSourcePathRecord, ...],
    action_values: tuple[str, str, str],
    scalar_probe_records: list[dict[str, Any]],
    precision_source: _MPPrecisionOperandSource,
) -> MPTrialResult:
    original = _make_h1_law(data, paths)
    action_kind = (
        "diagonal_base" if trial_id == "scalar-base-transformed" else "internal_product"
    )
    action_sha256 = _action_hash(
        [[[value]] for value in action_values],
        dimension=1,
        kind=action_kind,
        scalar=True,
        location=f"scalar_trial.actions.{trial_id}",
    )
    actions = cast(
        tuple[mp.matrix, mp.matrix, mp.matrix],
        tuple(mp.matrix([[_mp(item)]]) for item in action_values),
    )
    transformed = _transform_law(
        original,
        actions,
        decoder_policy="transform",
    )
    recovered = _recover_law(
        transformed,
        actions,
        decoder_policy="transform",
    )
    _require_exact_discrete_identity(original, transformed, recovered)
    original_objective = _objective_values(original)
    transformed_objective = _objective_values(transformed)
    probe_evaluations, density_maxima = _scalar_probe_evaluations(
        original,
        transformed,
        original_objective,
        transformed_objective,
        scalar_probe_records,
        trial_id=trial_id,
    )
    scalar_items = _pair_scalar_items(
        original_objective,
        transformed_objective,
        actions,
        density_maxima,
        prefix="",
    )
    original_evidence_17 = _scalar_evidence(original, order=17)
    original_evidence_21 = _scalar_evidence(original, order=21)
    transformed_evidence_17 = _scalar_evidence(transformed, order=17)
    transformed_evidence_21 = _scalar_evidence(transformed, order=21)
    original_elbo_21 = _objective_at_order(
        original,
        paths,
        dict(original_objective.q_moments),
        dict(original_objective.p_moments),
        21,
    )["complete_local_elbo"]
    transformed_elbo_21 = _objective_at_order(
        transformed,
        paths,
        dict(transformed_objective.q_moments),
        dict(transformed_objective.p_moments),
        21,
    )["complete_local_elbo"]
    original_posterior_kl_17 = _scalar_posterior_kl(
        original,
        log_evidence=original_evidence_17,
        order=17,
    )
    original_posterior_kl_21 = _scalar_posterior_kl(
        original,
        log_evidence=original_evidence_21,
        order=21,
    )
    transformed_posterior_kl_17 = _scalar_posterior_kl(
        transformed,
        log_evidence=transformed_evidence_17,
        order=17,
    )
    transformed_posterior_kl_21 = _scalar_posterior_kl(
        transformed,
        log_evidence=transformed_evidence_21,
        order=21,
    )
    scalar_items.extend(
        (
            ("scalar_log_evidence", _decimal(original_evidence_21)),
            (
                "transformed.scalar_log_evidence",
                _decimal(transformed_evidence_21),
            ),
            (
                "scalar_log_evidence_residual",
                _decimal(abs(transformed_evidence_21 - original_evidence_21)),
            ),
            (
                "scalar_evidence_gh17_gh21_delta",
                _decimal(abs(original_evidence_21 - original_evidence_17)),
            ),
            (
                "transformed.scalar_evidence_gh17_gh21_delta",
                _decimal(abs(transformed_evidence_21 - transformed_evidence_17)),
            ),
            ("scalar_posterior_kl", _decimal(original_posterior_kl_21)),
            (
                "transformed.scalar_posterior_kl",
                _decimal(transformed_posterior_kl_21),
            ),
            (
                "scalar_posterior_kl_residual",
                _decimal(abs(transformed_posterior_kl_21 - original_posterior_kl_21)),
            ),
            (
                "scalar_posterior_kl_gh17_gh21_delta",
                _decimal(abs(original_posterior_kl_21 - original_posterior_kl_17)),
            ),
            (
                "transformed.scalar_posterior_kl_gh17_gh21_delta",
                _decimal(
                    abs(transformed_posterior_kl_21 - transformed_posterior_kl_17)
                ),
            ),
            (
                "scalar_evidence_elbo_posterior_kl_residual",
                _decimal(
                    abs(
                        original_evidence_21
                        - original_elbo_21
                        - original_posterior_kl_21
                    )
                ),
            ),
            (
                "transformed.scalar_evidence_elbo_posterior_kl_residual",
                _decimal(
                    abs(
                        transformed_evidence_21
                        - transformed_elbo_21
                        - transformed_posterior_kl_21
                    )
                ),
            ),
        )
    )
    original_inventory = _collect_original_inventory(
        original,
        original_objective,
        prefix="scalar",
        trial_id=trial_id,
        precision_source=precision_source,
    )
    transformed_inventory = _transform_inventory(
        original_inventory,
        actions,
        decoder_policy="transform",
    )
    recovered_inventory = _recover_inventory(
        transformed_inventory,
        actions,
        decoder_policy="transform",
    )
    backward = _build_bound_backward_records(
        original_inventory,
        transformed_inventory,
        recovered_inventory,
        actions,
        decoder_policy="transform",
    )
    original_values = _inventory_values(original_inventory)
    transformed_values = _inventory_values(transformed_inventory)
    recovered_values = _inventory_values(recovered_inventory)
    return MPTrialResult(
        trial_id=trial_id,
        fixture_id="h1-v1",
        frame_profile="h1_v1",
        decoder_policy="transform",
        action_sha256=action_sha256,
        recognition_families=("scalar_h1_v1",),
        source_paths=paths,
        original=original_values,
        transformed=transformed_values,
        recovered=recovered_values,
        backward_records=backward.records,
        backward_bound_budgets=backward.bound_budgets,
        backward_inventory_size=len(backward.records),
        r_back_max=_decimal(mp.mpf(str(backward.maximum))),
        scalar_items=tuple(scalar_items),
        status_items=(),
        scorer_rows=(),
        probe_evaluations=probe_evaluations,
    )


def _evaluate_matrix_trial(
    trial_id: str,
    data: Mapping[str, Any],
    paths: tuple[MPSourcePathRecord, ...],
    *,
    frame_profile: str,
    action_profile: str,
    decoder_policy: Literal["transform", "fixed"],
    probe_records: list[dict[str, Any]],
    precision_source: _MPPrecisionOperandSource,
) -> MPTrialResult:
    action_kind = (
        "internal_product" if action_profile == "internal" else "diagonal_base"
    )
    action_sha256 = _action_hash(
        data["actions"][action_profile],
        dimension=2,
        kind=action_kind,
        scalar=False,
        location=f"matrix_trial.actions.{trial_id}",
    )
    actions = cast(
        tuple[mp.matrix, mp.matrix, mp.matrix],
        tuple(_matrix(item) for item in data["actions"][action_profile]),
    )
    original_inventory: list[_MPInventoryOperand] = []
    transformed_inventory: list[_MPInventoryOperand] = []
    recovered_inventory: list[_MPInventoryOperand] = []
    scalar_items: list[tuple[str, str]] = []
    all_probes: list[MPProbeEvaluationRecord] = []
    scorer_rows: tuple[MPScorerRowRecord, ...] = ()
    for family in _MATRIX_RECOGNITION_FAMILIES:
        original = _make_h7_law(
            data,
            paths,
            frame_profile=frame_profile,
            recognition_family=family,
        )
        transformed = _transform_law(
            original,
            actions,
            decoder_policy=decoder_policy,
        )
        recovered = _recover_law(
            transformed,
            actions,
            decoder_policy=decoder_policy,
        )
        _require_exact_discrete_identity(original, transformed, recovered)
        original_objective = _objective_values(original)
        transformed_objective = _objective_values(transformed)
        probes, density_maxima = _probe_evaluations(
            original,
            transformed,
            original_objective,
            transformed_objective,
            probe_records,
            action_profile=action_profile,
        )
        prefix = "" if family == "structured_full_block" else "factorized"
        scalar_items.extend(
            _pair_scalar_items(
                original_objective,
                transformed_objective,
                actions,
                density_maxima,
                prefix=prefix,
            )
        )
        family_tag = _family_prefix(family)
        family_original_inventory = _collect_original_inventory(
            original,
            original_objective,
            prefix=family_tag,
            trial_id=trial_id,
            precision_source=precision_source,
        )
        family_transformed_inventory = _transform_inventory(
            family_original_inventory,
            actions,
            decoder_policy=decoder_policy,
        )
        family_recovered_inventory = _recover_inventory(
            family_transformed_inventory,
            actions,
            decoder_policy=decoder_policy,
        )
        original_inventory.extend(family_original_inventory)
        transformed_inventory.extend(family_transformed_inventory)
        recovered_inventory.extend(family_recovered_inventory)
        all_probes.extend(probes)
        if not scorer_rows:
            scorer_rows = _scorer_record_pairs(
                original["scorer_rows"],
                transformed["scorer_rows"],
            )
    frozen_original_inventory = tuple(original_inventory)
    frozen_transformed_inventory = tuple(transformed_inventory)
    frozen_recovered_inventory = tuple(recovered_inventory)
    if (
        len(frozen_original_inventory) != 218
        or len(frozen_transformed_inventory) != 218
        or len(frozen_recovered_inventory) != 218
    ):
        raise ValueError(
            "H7 matrix backward inventory must contain exactly 218 operands"
        )
    backward = _build_bound_backward_records(
        frozen_original_inventory,
        frozen_transformed_inventory,
        frozen_recovered_inventory,
        actions,
        decoder_policy=decoder_policy,
    )
    original_law_values = _inventory_values(frozen_original_inventory)
    transformed_law_values = _inventory_values(frozen_transformed_inventory)
    recovered_law_values = _inventory_values(frozen_recovered_inventory)
    return MPTrialResult(
        trial_id=trial_id,
        fixture_id="h7-v1",
        frame_profile=frame_profile,
        decoder_policy=decoder_policy,
        action_sha256=action_sha256,
        recognition_families=_MATRIX_RECOGNITION_FAMILIES,
        source_paths=paths,
        original=original_law_values,
        transformed=transformed_law_values,
        recovered=recovered_law_values,
        backward_records=backward.records,
        backward_bound_budgets=backward.bound_budgets,
        backward_inventory_size=218,
        r_back_max=_decimal(mp.mpf(str(backward.maximum))),
        scalar_items=tuple(scalar_items),
        status_items=(
            (
                "matrix_evidence_not_applicable",
                _MATRIX_EVIDENCE_NOT_APPLICABLE,
            ),
        ),
        scorer_rows=scorer_rows,
        probe_evaluations=tuple(all_probes),
    )


def evaluate_h7_task5_wiring(
    original: object,
    transformed: object,
    action: object,
    *,
    trial_spec: object,
    original_law_evidence: object,
    transformed_law_evidence: object,
    density_probe_pairs: object,
    quadrature_orders: tuple[int, int],
    budgets_by_invariant: Mapping[str, object],
    oracle_trial: MPTrialResult,
    task5_evaluator: object,
    scalar_evidence: object | None = None,
    oracle_prefix: str = "",
) -> MPTask5WiringResult:
    """Call an injected Task-5 evaluator and compare one bound MP trial."""

    if type(oracle_trial) is not MPTrialResult:
        raise ValueError("oracle_trial must be an exact MPTrialResult")
    if not callable(task5_evaluator):
        raise ValueError("task5_evaluator must be callable")
    trial_validator = getattr(trial_spec, "__post_init__", None)
    trial_action = getattr(trial_spec, "action", None)
    action_validator = getattr(action, "__post_init__", None)
    action_elements = getattr(action, "elements", None)
    trial_action_elements = getattr(trial_action, "elements", None)
    if (
        not callable(trial_validator)
        or not callable(action_validator)
        or type(action) is not type(trial_action)
        or type(action_elements) is not tuple
        or type(trial_action_elements) is not tuple
        or len(action_elements) != 3
        or len(trial_action_elements) != 3
        or any(
            type(item) is not type(expected)
            or not callable(getattr(item, "assert_intact", None))
            or type(getattr(item, "snapshot_sha256", None)) is not str
            for item, expected in zip(
                action_elements,
                trial_action_elements,
                strict=True,
            )
        )
    ):
        raise H7OracleInconclusive(
            "Task-5 action must be an exact intact owned H7 action"
        )
    if (
        type(getattr(trial_spec, "trial_id", None)) is not str
        or type(getattr(trial_spec, "fixture_id", None)) is not str
        or type(getattr(trial_spec, "frame_profile", None)) is not str
        or type(getattr(trial_spec, "decoder_policy", None)) is not str
        or type(getattr(trial_spec, "action_sha256", None)) is not str
    ):
        raise H7OracleInconclusive(
            "Task-5 trial binding lacks exact owned identity fields"
        )
    try:
        trial_validator()
        action_validator()
        for item in action_elements:
            item.assert_intact()
    except ValueError as error:
        raise H7OracleInconclusive(
            "Task-5 action must be an exact intact owned H7 action"
        ) from error
    if (
        trial_spec.trial_id != oracle_trial.trial_id
        or trial_spec.fixture_id != oracle_trial.fixture_id
        or trial_spec.frame_profile != oracle_trial.frame_profile
        or trial_spec.decoder_policy != oracle_trial.decoder_policy
        or trial_spec.action_sha256 != oracle_trial.action_sha256
        or type(action) is not type(trial_action)
        or action.action_sha256 != trial_spec.action_sha256
        or action.action_sha256 != oracle_trial.action_sha256
    ):
        raise H7OracleInconclusive(
            "Task-5 trial/action/frame/decoder contract does not bind the oracle trial"
        )
    if oracle_prefix not in ("", "factorized."):
        raise ValueError("oracle_prefix must select one frozen recognition family")
    expected_raw_sha256 = (
        _H1_RAW_SHA256 if oracle_trial.fixture_id == "h1-v1" else _H7_RAW_SHA256
    )
    if (
        getattr(original, "fixture_id", None) != oracle_trial.fixture_id
        or getattr(transformed, "fixture_id", None) != oracle_trial.fixture_id
        or getattr(original, "raw_fixture_sha256", None) != expected_raw_sha256
        or getattr(transformed, "raw_fixture_sha256", None) != expected_raw_sha256
    ):
        raise ValueError("Task-5 inputs do not bind the oracle trial fixture")
    if oracle_trial.fixture_id == "h7-v1":
        origin_family = getattr(
            getattr(original, "recognition", None),
            "origin_family",
            None,
        )
        expected_prefix = (
            ""
            if origin_family == "structured_full_block"
            else (
                "factorized."
                if origin_family == "factorized_diagonal_within_fiber"
                else None
            )
        )
        if oracle_prefix != expected_prefix:
            raise ValueError(
                "oracle_prefix does not bind the production recognition family"
            )
    elif oracle_prefix:
        raise ValueError("scalar Task-5 wiring cannot select a matrix-family prefix")
    production = task5_evaluator(
        original,
        transformed,
        action,
        original_law_evidence=original_law_evidence,
        transformed_law_evidence=transformed_law_evidence,
        density_probe_pairs=density_probe_pairs,
        quadrature_orders=quadrature_orders,
        budgets_by_invariant=budgets_by_invariant,
        scalar_evidence=scalar_evidence,
    )
    production_validator = getattr(production, "__post_init__", None)
    if not callable(production_validator):
        raise H7OracleInconclusive(
            "Task-5 returned a mislabeled objective-evaluation record"
        )
    try:
        production_validator()
    except ValueError as error:
        raise H7OracleInconclusive(
            "Task-5 returned a non-intact objective-evaluation record"
        ) from error
    required_production_fields = (
        "original_complete_local_value",
        "transformed_complete_local_value",
        "initial_joint_kl",
        "local_terms",
        "complete_local",
        "complete_monolithic",
        "p_density_shift",
        "q_density_shift",
        "log_ratio",
        "entropy_shift",
        "scalar_evidence",
        "evidence",
        "posterior_kl",
        "not_applicable_reason",
    )
    if any(
        not hasattr(production, field_name) for field_name in required_production_fields
    ):
        raise H7OracleInconclusive(
            "Task-5 returned missing objective-evaluation records"
        )
    residual_contract = (
        (production.complete_local, _TASK5_COMPLETE_LOCAL_INVARIANT_ID),
        (
            production.complete_monolithic,
            _TASK5_COMPLETE_MONOLITHIC_INVARIANT_ID,
        ),
        (production.p_density_shift, _TASK5_POINTWISE_P_SHIFT_INVARIANT_ID),
        (production.q_density_shift, _TASK5_POINTWISE_Q_SHIFT_INVARIANT_ID),
        (production.log_ratio, _TASK5_POINTWISE_LOG_RATIO_INVARIANT_ID),
        (production.entropy_shift, _TASK5_ENTROPY_SHIFT_INVARIANT_ID),
    )
    if (
        production.initial_joint_kl.term_id != "K0_joint_z0_m0"
        or production.initial_joint_kl.residual.invariant_id != "K0_joint_z0_m0"
        or tuple(item.term_id for item in production.local_terms)
        != H7_COMPLETE_LOCAL_TERM_IDS
        or any(
            item.residual.invariant_id != item.term_id
            for item in production.local_terms
        )
        or any(
            type(getattr(record, "invariant_id", None)) is not str
            or type(getattr(record, "value", None)) is not float
            or record.invariant_id != invariant_id
            for record, invariant_id in residual_contract
        )
    ):
        raise H7OracleInconclusive(
            "Task-5 returned missing, extra, or mislabeled objective records"
        )
    oracle_values = oracle_trial.scalar_map
    comparisons: list[MPTask5OracleComparison] = []
    status_comparisons: list[tuple[str, str, str]] = []
    consumed_oracle_keys: list[str] = []

    def oracle_value(value_id: str) -> str:
        oracle_key = f"{oracle_prefix}{value_id}"
        if oracle_key in consumed_oracle_keys:
            raise H7OracleInconclusive(
                f"Task-5 wiring consumed duplicate oracle value {oracle_key!r}"
            )
        try:
            value = oracle_values[oracle_key]
        except KeyError as error:
            raise H7OracleInconclusive(
                f"oracle trial lacks Task-5 value {oracle_key!r}"
            ) from error
        consumed_oracle_keys.append(oracle_key)
        return value

    def oracle_mpf(value_id: str) -> mp.mpf:
        value = oracle_value(value_id)
        try:
            return _mp(value)
        except _H7ExternalDataError as error:
            raise H7OracleInconclusive(
                f"oracle trial has a malformed Task-5 value {value_id!r}"
            ) from error

    def append_values(
        value_id: str,
        production_value: float,
        expected_oracle_value: str,
    ) -> None:
        if (
            type(production_value) is not float
            or not math.isfinite(production_value)
            or type(expected_oracle_value) is not str
            or not expected_oracle_value
        ):
            raise H7OracleInconclusive(
                f"Task-5 comparison {value_id!r} lacks exact finite values"
            )
        try:
            parsed_oracle_value = _mp(expected_oracle_value)
        except _H7ExternalDataError as error:
            raise H7OracleInconclusive(
                f"oracle trial has a malformed Task-5 value {value_id!r}"
            ) from error
        absolute_delta = abs(mp.mpf(str(production_value)) - parsed_oracle_value)
        comparisons.append(
            MPTask5OracleComparison(
                value_id=value_id,
                production_value=production_value,
                oracle_value=expected_oracle_value,
                absolute_delta=_decimal(absolute_delta),
            )
        )

    def append_comparison(value_id: str, production_value: float) -> None:
        oracle_key = f"{oracle_prefix}{value_id}"
        expected_oracle_value = oracle_value(value_id)
        append_values(oracle_key, production_value, expected_oracle_value)

    def append_positive_kl_comparison(
        value_id: str,
        production_value: float,
    ) -> None:
        oracle_key = f"{oracle_prefix}{value_id}"
        signed_negative_kl = oracle_mpf(value_id)
        positive_kl = -signed_negative_kl
        if positive_kl < 0:
            raise H7OracleInconclusive(
                f"oracle local KL {oracle_key!r} is not a signed "
                "negative-KL contribution"
            )
        append_values(
            oracle_key,
            production_value,
            _decimal(positive_kl),
        )

    append_comparison(
        "complete_local_elbo",
        production.original_complete_local_value,
    )
    append_comparison(
        "transformed.complete_local_elbo",
        production.transformed_complete_local_value,
    )
    append_comparison(
        "residual.complete_local_elbo",
        production.complete_local.value,
    )
    append_comparison(
        "K0_joint_z0_m0",
        production.initial_joint_kl.original_value,
    )
    append_comparison(
        "transformed.K0_joint_z0_m0",
        production.initial_joint_kl.transformed_value,
    )
    append_comparison(
        "residual.K0_joint_z0_m0",
        production.initial_joint_kl.residual.value,
    )
    entropy_oracle_value: str | None = None
    for local_term in production.local_terms:
        if local_term.term_id.endswith(("_kl[1]", "_kl[2]")):
            append_positive_kl_comparison(
                local_term.term_id,
                local_term.original_value,
            )
            append_positive_kl_comparison(
                f"transformed.{local_term.term_id}",
                local_term.transformed_value,
            )
        else:
            append_comparison(
                local_term.term_id,
                local_term.original_value,
            )
            append_comparison(
                f"transformed.{local_term.term_id}",
                local_term.transformed_value,
            )
        if local_term.term_id == "joint_recognition_entropy":
            entropy_oracle_value = oracle_value(
                "residual.joint_recognition_entropy"
            )
            append_values(
                f"{oracle_prefix}residual.joint_recognition_entropy",
                local_term.residual.value,
                entropy_oracle_value,
            )
        else:
            append_comparison(
                f"residual.{local_term.term_id}",
                local_term.residual.value,
            )
    monolithic_oracle_value = _decimal(
        max(
            oracle_mpf("residual.complete_monolithic_elbo"),
            oracle_mpf("complete_local_monolithic_delta"),
            oracle_mpf("transformed.complete_local_monolithic_delta"),
        )
    )
    append_values(
        f"{oracle_prefix}complete_monolithic_elbo",
        production.complete_monolithic.value,
        monolithic_oracle_value,
    )
    append_comparison(
        "complete_pointwise_p_density_shift",
        production.p_density_shift.value,
    )
    append_comparison(
        "complete_pointwise_q_density_shift",
        production.q_density_shift.value,
    )
    append_comparison(
        "complete_pointwise_log_ratio",
        production.log_ratio.value,
    )
    if entropy_oracle_value is None:
        raise H7OracleInconclusive(
            "Task-5 production omitted joint-recognition entropy"
        )
    append_values(
        f"{oracle_prefix}{_TASK5_ENTROPY_SHIFT_INVARIANT_ID}",
        production.entropy_shift.value,
        entropy_oracle_value,
    )
    if oracle_trial.fixture_id == "h1-v1":
        evidence_identity = production.scalar_evidence
        evidence_validator = getattr(evidence_identity, "__post_init__", None)
        if (
            evidence_identity is not scalar_evidence
            or not callable(evidence_validator)
            or getattr(evidence_identity, "fixture_id", None) != oracle_trial.fixture_id
            or getattr(evidence_identity, "raw_fixture_sha256", None)
            != expected_raw_sha256
            or getattr(evidence_identity, "action_sha256", None)
            != oracle_trial.action_sha256
            or type(getattr(production.evidence, "value", None)) is not float
            or production.evidence.invariant_id != _TASK5_SCALAR_EVIDENCE_INVARIANT_ID
            or type(getattr(production.posterior_kl, "value", None)) is not float
            or production.posterior_kl.invariant_id
            != _TASK5_SCALAR_POSTERIOR_KL_INVARIANT_ID
            or production.not_applicable_reason is not None
            or oracle_trial.status_items
        ):
            raise H7OracleInconclusive(
                "scalar Task-5 evaluation has missing, extra, or mislabeled "
                "evidence/posterior-KL identity records"
            )
        try:
            evidence_validator()
        except ValueError as error:
            raise H7OracleInconclusive(
                "scalar Task-5 evidence identity is not intact"
            ) from error
        append_comparison(
            "scalar_log_evidence",
            production.scalar_evidence.original_log_evidence,
        )
        append_comparison(
            "transformed.scalar_log_evidence",
            production.scalar_evidence.transformed_log_evidence,
        )
        append_comparison(
            "scalar_posterior_kl",
            production.scalar_evidence.original_posterior_kl,
        )
        append_comparison(
            "transformed.scalar_posterior_kl",
            production.scalar_evidence.transformed_posterior_kl,
        )
        evidence_oracle_value = _decimal(
            max(
                oracle_mpf("scalar_log_evidence_residual"),
                oracle_mpf("scalar_evidence_elbo_posterior_kl_residual"),
                oracle_mpf("transformed.scalar_evidence_elbo_posterior_kl_residual"),
            )
        )
        append_values(
            "scalar_log_evidence_and_elbo_kl_identity",
            production.evidence.value,
            evidence_oracle_value,
        )
        append_comparison(
            "scalar_posterior_kl_residual",
            production.posterior_kl.value,
        )
    else:
        if (
            production.scalar_evidence is not None
            or production.evidence is not None
            or production.posterior_kl is not None
            or type(production.not_applicable_reason) is not str
        ):
            raise H7OracleInconclusive(
                "matrix Task-5 evaluation fabricated evidence/posterior KL"
            )
        try:
            expected_status = oracle_trial.status_map["matrix_evidence_not_applicable"]
        except KeyError as error:
            raise H7OracleInconclusive(
                "oracle trial lacks matrix evidence applicability status"
            ) from error
        if (
            oracle_trial.status_items
            != (("matrix_evidence_not_applicable", expected_status),)
            or production.not_applicable_reason != expected_status
        ):
            raise H7OracleInconclusive(
                "matrix Task-5 evidence applicability status disagrees with "
                "the oracle trial"
            )
        status_comparisons.append(
            (
                "matrix_evidence_not_applicable",
                production.not_applicable_reason,
                expected_status,
            )
        )
    local_value_ids = tuple(
        value_id
        for term_id in H7_COMPLETE_LOCAL_TERM_IDS
        for value_id in (
            term_id,
            f"transformed.{term_id}",
            f"residual.{term_id}",
        )
    )
    common_consumed_ids = (
        "complete_local_elbo",
        "transformed.complete_local_elbo",
        "residual.complete_local_elbo",
        "K0_joint_z0_m0",
        "transformed.K0_joint_z0_m0",
        "residual.K0_joint_z0_m0",
        *local_value_ids,
        "residual.complete_monolithic_elbo",
        "complete_local_monolithic_delta",
        "transformed.complete_local_monolithic_delta",
        "complete_pointwise_p_density_shift",
        "complete_pointwise_q_density_shift",
        "complete_pointwise_log_ratio",
    )
    common_comparison_ids = (
        *common_consumed_ids[: 6 + len(local_value_ids)],
        "complete_monolithic_elbo",
        *common_consumed_ids[-3:],
        _TASK5_ENTROPY_SHIFT_INVARIANT_ID,
    )
    scalar_consumed_ids = (
        "scalar_log_evidence",
        "transformed.scalar_log_evidence",
        "scalar_posterior_kl",
        "transformed.scalar_posterior_kl",
        "scalar_log_evidence_residual",
        "scalar_evidence_elbo_posterior_kl_residual",
        "transformed.scalar_evidence_elbo_posterior_kl_residual",
        "scalar_posterior_kl_residual",
    )
    scalar_comparison_ids = (
        "scalar_log_evidence",
        "transformed.scalar_log_evidence",
        "scalar_posterior_kl",
        "transformed.scalar_posterior_kl",
        "scalar_log_evidence_and_elbo_kl_identity",
        "scalar_posterior_kl_residual",
    )
    expected_consumed_keys = tuple(
        f"{oracle_prefix}{value_id}"
        for value_id in (
            *common_consumed_ids,
            *(scalar_consumed_ids if oracle_trial.fixture_id == "h1-v1" else ()),
        )
    )
    expected_comparison_ids = tuple(
        f"{oracle_prefix}{value_id}"
        for value_id in (
            *common_comparison_ids,
            *(scalar_comparison_ids if oracle_trial.fixture_id == "h1-v1" else ()),
        )
    )
    observed_comparison_ids = tuple(item.value_id for item in comparisons)
    if (
        tuple(consumed_oracle_keys) != expected_consumed_keys
        or observed_comparison_ids != expected_comparison_ids
        or len(set(observed_comparison_ids)) != len(observed_comparison_ids)
    ):
        raise H7OracleInconclusive(
            "Task-5 wiring produced missing, extra, duplicate, or mislabeled "
            "comparison records"
        )
    return MPTask5WiringResult(
        trial_id=oracle_trial.trial_id,
        fixture_id=oracle_trial.fixture_id,
        production_evaluation=production,
        comparisons=tuple(comparisons),
        status_comparisons=tuple(status_comparisons),
    )


__all__ = [
    "H7_COMPLETE_LOCAL_TERM_IDS",
    "H7OracleInconclusive",
    "H7_REQUIRED_TRIAL_IDS",
    "H7MPOracleResult",
    "MPGaussHermiteRule",
    "MPLawValues",
    "MPProbeEvaluationRecord",
    "MPScorerRowRecord",
    "MPSourcePathRecord",
    "MPTask5OracleComparison",
    "MPTask5WiringResult",
    "MPTrialResult",
    "MPValueRecord",
    "build_h7_scalar_probe_table_bytes",
    "evaluate_h7_from_raw_bytes",
    "evaluate_h7_task5_wiring",
    "standard_normal_gauss_hermite",
]
