"""Frozen deterministic inputs for the synthetic H8 chain.

Only this module owns the PCG64 draw stream.  Consumers receive immutable
little-endian float64 arrays and never receive an RNG object.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np
import torch

from vfe4.numerics.block_canonical import BlockCanonicalAssembler
from vfe4.numerics.block_layout import BlockChainLayout
from vfe4.numerics.block_tridiagonal import BlockTridiagonalCholesky
from vfe4.numerics.sparse_information import FactorBackedInformationGaussian
from vfe4.types.h8 import (
    H8_PROBLEM_DRAW_SCHEMA_SHA256,
    H8LocalSPDDiagnostics,
    H8ProductionProblemEvidence,
    H8TransitionNorms,
)


H8_GENERATOR_SCHEMA = "h8-synthetic-chain-v1"
H8_SAMPLE_SCHEMA = "h8-pcg64-sample-v1"
H8_DRAW_DESCRIPTOR = (
    "numpy.Generator(numpy.PCG64(problem_seed))|method=standard_normal|"
    "dtype=float64|order=C|initial:mu0[b],Q0[b,b]|"
    "transition:t=1..T:{A_m[K,K],c_m[K],Q_m[K,K],A_z[K,K],B[K,K],c_z[K],Q_z[K,K]}|"
    "recognition_initial:mu_q0[b],Q_q0[b,b]|"
    "recognition_transition:t=1..T:{A_q[b,b],c_q[b],Q_q[b,b]}|"
    "emission:t=1..T:{w[b],beta[V]}|"
    "normal_map_variance=1/dim=>multiply_standard_normal_by_1/sqrt(dim)|"
    "serialize=after_all_problem_draws_before_sample_rng|bytes=little-endian-f8-C-contiguous"
)


def _frozen(value: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype="<f8")
    array.setflags(write=False)
    return array


def _spd(q: np.ndarray, width: int) -> np.ndarray:
    product = np.matmul(q, np.transpose(q))
    scaled = np.divide(np.multiply(0.05, product), width)
    diagonal = np.multiply(0.25, np.eye(width, dtype=np.float64))
    return _frozen(np.add(diagonal, scaled))


def _contract(matrix: np.ndarray, radius: float) -> tuple[np.ndarray, float]:
    raw_norm = float(np.linalg.norm(matrix, ord=2))
    numerator = np.multiply(radius, matrix)
    denominator = max(radius, raw_norm)
    contracted = _frozen(np.divide(numerator, denominator))
    contracted_norm = radius * (raw_norm / denominator)
    return contracted, contracted_norm


@dataclass(frozen=True, slots=True)
class H8ModelTransition:
    receiver_t: int
    parent_t: int
    matrix: np.ndarray
    offset: np.ndarray
    covariance: np.ndarray
    source_support: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class H8StateTransition:
    receiver_t: int
    parent_t: int
    state_matrix: np.ndarray
    model_matrix: np.ndarray
    offset: np.ndarray
    covariance: np.ndarray
    source_support: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class H8RecognitionSpecification:
    initial_mean: np.ndarray
    initial_covariance: np.ndarray
    transitions: tuple[H8ModelTransition, ...]


@dataclass(frozen=True, slots=True)
class H8Emission:
    receiver_t: int
    weight: np.ndarray
    bias: np.ndarray
    observation: int


@dataclass(frozen=True, slots=True, init=False)
class H8Problem:
    layout: BlockChainLayout
    problem_seed: int
    vocabulary_size: int
    alpha: np.ndarray
    initial_mean: np.ndarray
    initial_covariance: np.ndarray
    model_transitions: tuple[H8ModelTransition, ...]
    state_transitions: tuple[H8StateTransition, ...]
    emissions: tuple[H8Emission, ...]
    recognition: H8RecognitionSpecification
    serialized_bytes: bytes
    input_sha256: str
    problem_evidence: H8ProductionProblemEvidence
    draw_schema_sha256: str = H8_PROBLEM_DRAW_SCHEMA_SHA256

    def __init__(self) -> None:
        raise TypeError("H8Problem is factory-only; use make_h8_problem()")


@dataclass(frozen=True, slots=True)
class H8GenerativeModel:
    """The normalized combined-chain law, retained only through its factor."""

    gaussian: FactorBackedInformationGaussian
    input_sha256: str
    information_scalar_count: int
    factor_scalar_count: int


def make_h8_problem(
    *,
    horizon: int = 128,
    channel_dimension: int = 20,
    problem_seed: int,
    allocation_guard: object | None = None,
) -> H8Problem:
    """Build the literal H8 draw stream without vectorizing across time."""

    layout = BlockChainLayout(horizon=horizon, d_z=channel_dimension, d_m=channel_dimension)
    if type(problem_seed) is not int or problem_seed <= 0:
        raise ValueError("problem_seed must be a positive integer")
    width = layout.block_size
    k = channel_dimension
    rng = np.random.Generator(np.random.PCG64(problem_seed))

    def sn(shape: tuple[int, ...]) -> np.ndarray:
        drawn = (
            rng.standard_normal(size=shape, dtype=np.float64)
            if allocation_guard is None
            else allocation_guard.standard_normal(
                rng,
                size=shape,
                dtype=np.float64,
            )
        )
        return np.ascontiguousarray(drawn)

    initial_mean = _frozen(np.multiply(0.1, sn((width,))))
    initial_covariance = _spd(sn((width, width)), width)
    model: list[H8ModelTransition] = []
    state: list[H8StateTransition] = []
    model_norms: list[float] = []
    state_norms: list[float] = []
    coupling_norms: list[float] = []
    for receiver_t in range(1, layout.population_size):
        a_m, a_m_norm = _contract(
            np.divide(sn((k, k)), np.sqrt(k)),
            0.35,
        )
        c_m = _frozen(np.multiply(0.05, sn((k,))))
        r_m = _spd(sn((k, k)), k)
        a_z, a_z_norm = _contract(
            np.divide(sn((k, k)), np.sqrt(k)),
            0.35,
        )
        b, b_norm = _contract(
            np.divide(sn((k, k)), np.sqrt(k)),
            0.20,
        )
        c_z = _frozen(np.multiply(0.05, sn((k,))))
        r_z = _spd(sn((k, k)), k)
        model.append(
            H8ModelTransition(
                receiver_t,
                receiver_t - 1,
                a_m,
                c_m,
                r_m,
                (receiver_t - 1,),
            )
        )
        state.append(
            H8StateTransition(
                receiver_t,
                receiver_t - 1,
                a_z,
                b,
                c_z,
                r_z,
                (receiver_t - 1,),
            )
        )
        model_norms.append(a_m_norm)
        state_norms.append(a_z_norm)
        coupling_norms.append(b_norm)
    recognition_initial_mean = _frozen(np.multiply(0.1, sn((width,))))
    recognition_initial_covariance = _spd(sn((width, width)), width)
    recognition_transitions: list[H8ModelTransition] = []
    recognition_norms: list[float] = []
    for receiver_t in range(1, layout.population_size):
        a_q, a_q_norm = _contract(
            np.divide(sn((width, width)), np.sqrt(width)),
            0.35,
        )
        c_q = _frozen(np.multiply(0.05, sn((width,))))
        r_q = _spd(sn((width, width)), width)
        recognition_transitions.append(
            H8ModelTransition(
                receiver_t,
                receiver_t - 1,
                a_q,
                c_q,
                r_q,
                (receiver_t - 1,),
            )
        )
        recognition_norms.append(a_q_norm)
    alpha = _frozen(np.asarray((-0.5, 0.25, 0.75), dtype=np.float64))
    emissions: list[H8Emission] = []
    for receiver_t in range(1, layout.population_size):
        weight = _frozen(np.divide(sn((width,)), np.sqrt(width)))
        bias = _frozen(np.multiply(0.1, sn((3,))))
        emissions.append(H8Emission(receiver_t, weight, bias, (problem_seed + receiver_t) % 3))
    recognition = H8RecognitionSpecification(
        recognition_initial_mean,
        recognition_initial_covariance,
        tuple(recognition_transitions),
    )
    model_tuple = tuple(model)
    state_tuple = tuple(state)
    emission_tuple = tuple(emissions)
    transition_norms = H8TransitionNorms(
        schema_version="h8-transition-norms-v1",
        horizon=layout.horizon,
        norm="operator_2",
        model_transition_norms=tuple(model_norms),
        state_transition_norms=tuple(state_norms),
        state_model_coupling_norms=tuple(coupling_norms),
        recognition_transition_norms=tuple(recognition_norms),
        max_model_transition_norm=max(model_norms),
        max_state_transition_norm=max(state_norms),
        max_state_model_coupling_norm=max(coupling_norms),
        max_recognition_transition_norm=max(recognition_norms),
    )
    serialized = _serialize_problem(
        layout=layout,
        problem_seed=problem_seed,
        vocabulary_size=3,
        alpha=alpha,
        initial_mean=initial_mean,
        initial_covariance=initial_covariance,
        model_transitions=model_tuple,
        state_transitions=state_tuple,
        emissions=emission_tuple,
        recognition=recognition,
        draw_schema_sha256=H8_PROBLEM_DRAW_SCHEMA_SHA256,
    )
    problem = _new_problem(
        layout=layout,
        problem_seed=problem_seed,
        vocabulary_size=3,
        alpha=alpha,
        initial_mean=initial_mean,
        initial_covariance=initial_covariance,
        model_transitions=model_tuple,
        state_transitions=state_tuple,
        emissions=emission_tuple,
        recognition=recognition,
        serialized_bytes=serialized,
        input_sha256=hashlib.sha256(serialized).hexdigest(),
        draw_schema_sha256=H8_PROBLEM_DRAW_SCHEMA_SHA256,
    )
    problem_evidence = _derive_problem_evidence(problem, transition_norms)
    object.__setattr__(problem, "problem_evidence", problem_evidence)
    _validate_problem_serialized_identity(problem)
    return problem


build_h8_problem = make_h8_problem


def build_h8_generative(problem: H8Problem) -> H8GenerativeModel:
    """Assemble the factorwise-equivalent combined local transition chain."""

    problem = validate_h8_problem(problem)
    layout = problem.layout
    assembler = BlockCanonicalAssembler(layout)
    assembler.add_initial(_torch(problem.initial_mean), _torch(problem.initial_covariance))
    k = layout.d_z
    for model, state in zip(problem.model_transitions, problem.state_transitions, strict=True):
        if model.receiver_t != state.receiver_t or model.source_support != (model.parent_t,) or state.source_support != (state.parent_t,):
            raise ValueError("generative factors must have matching singleton parents")
        a_m, c_m, r_m = _torch(model.matrix), _torch(model.offset), _torch(model.covariance)
        a_z, b, c_z, r_z = _torch(state.state_matrix), _torch(state.model_matrix), _torch(state.offset), _torch(state.covariance)
        matrix = torch.empty((layout.block_size, layout.block_size), dtype=torch.float64)
        matrix[:k, :k] = a_z
        matrix[:k, k:] = b @ a_m
        matrix[k:, :k] = torch.zeros((k, k), dtype=torch.float64)
        matrix[k:, k:] = a_m
        offset = torch.cat((c_z + b @ c_m, c_m))
        covariance = torch.empty((layout.block_size, layout.block_size), dtype=torch.float64)
        covariance[:k, :k] = r_z + b @ r_m @ b.T
        covariance[:k, k:] = b @ r_m
        covariance[k:, :k] = r_m @ b.T
        covariance[k:, k:] = r_m
        assembler.add_transition(model.receiver_t, matrix, offset, covariance)
    precision, h = assembler.freeze()
    factor = BlockTridiagonalCholesky.factorize(precision)
    return H8GenerativeModel(
        gaussian=FactorBackedInformationGaussian.from_factor(h, factor),
        input_sha256=problem.input_sha256,
        information_scalar_count=layout.information_scalar_count,
        factor_scalar_count=factor.storage.factor_scalar_count,
    )


def h8_sample_noise(
    problem: H8Problem,
    *,
    sample_noise_seed: int,
    allocation_guard: object | None = None,
) -> np.ndarray:
    """Draw the independent fixed sample stream exactly once in C order."""

    problem = validate_h8_problem(problem)
    if type(sample_noise_seed) is not int or sample_noise_seed <= 0:
        raise ValueError("sample_noise_seed must be a positive integer")
    rng = np.random.Generator(np.random.PCG64(sample_noise_seed))
    drawn = (
        rng.standard_normal(
            size=(problem.layout.population_size, problem.layout.block_size),
            dtype=np.float64,
        )
        if allocation_guard is None
        else allocation_guard.standard_normal(
            rng,
            size=(problem.layout.population_size, problem.layout.block_size),
            dtype=np.float64,
        )
    )
    return _frozen(drawn)


def validate_h8_problem(problem: object) -> H8Problem:
    """Revalidate one immutable problem and its complete canonical identity."""

    if type(problem) is not H8Problem:
        raise ValueError("problem must be a factory-created H8Problem")
    evidence = getattr(problem, "problem_evidence", None)
    if type(evidence) is not H8ProductionProblemEvidence:
        raise ValueError("problem evidence must be exact typed evidence")
    evidence.__post_init__()
    derived_evidence = _derive_problem_evidence(
        problem,
        evidence.transition_norms,
    )
    if evidence != derived_evidence:
        raise ValueError("problem evidence does not match the semantic fields")
    _validate_problem_serialized_identity(problem)
    return problem


def _derive_problem_evidence(
    problem: H8Problem,
    transition_norms: H8TransitionNorms,
) -> H8ProductionProblemEvidence:
    """Validate semantic fields once and derive hashes plus local pivots."""

    layout = problem.layout
    if type(layout) is not BlockChainLayout or layout.d_z != layout.d_m:
        raise ValueError("H8 requires one exact equal-channel BlockChainLayout")
    if type(problem.problem_seed) is not int or problem.problem_seed <= 0:
        raise ValueError("problem_seed must be a positive integer")
    if problem.vocabulary_size != 3:
        raise ValueError("H8 vocabulary size must be exactly three")
    descriptor_sha256 = hashlib.sha256(H8_DRAW_DESCRIPTOR.encode("ascii")).hexdigest()
    if (
        descriptor_sha256 != H8_PROBLEM_DRAW_SCHEMA_SHA256
        or problem.draw_schema_sha256 != H8_PROBLEM_DRAW_SCHEMA_SHA256
    ):
        raise ValueError("H8 draw-schema identity does not match the frozen descriptor")
    block = layout.block_size
    channel = layout.d_z
    _require_array(problem.alpha, (3,), "alpha")
    if not np.array_equal(
        problem.alpha,
        np.asarray((-0.5, 0.25, 0.75), dtype="<f8"),
    ):
        raise ValueError("H8 alpha must match the frozen categorical slopes")
    _require_array(problem.initial_mean, (block,), "initial_mean")
    _, generative_initial_pivot = _require_spd_array(
        problem.initial_covariance,
        (block, block),
        "initial_covariance",
    )
    if (
        type(problem.model_transitions) is not tuple
        or len(problem.model_transitions) != layout.horizon
        or type(problem.state_transitions) is not tuple
        or len(problem.state_transitions) != layout.horizon
        or type(problem.emissions) is not tuple
        or len(problem.emissions) != layout.horizon
    ):
        raise ValueError("H8 generative series must contain exactly T records")
    model_pivots: list[float] = []
    state_pivots: list[float] = []
    for receiver_t, (model, state, emission) in enumerate(
        zip(
            problem.model_transitions,
            problem.state_transitions,
            problem.emissions,
            strict=True,
        ),
        start=1,
    ):
        _require_transition_identity(model, receiver_t, "model transition")
        _require_array(model.matrix, (channel, channel), "model matrix")
        _require_array(model.offset, (channel,), "model offset")
        _, model_pivot = _require_spd_array(
            model.covariance,
            (channel, channel),
            "model covariance",
        )
        model_pivots.append(model_pivot)
        if type(state) is not H8StateTransition:
            raise ValueError("state transitions must use exact H8 records")
        _require_parent_identity(
            state.receiver_t,
            state.parent_t,
            state.source_support,
            receiver_t,
            "state transition",
        )
        _require_array(
            state.state_matrix,
            (channel, channel),
            "state matrix",
        )
        _require_array(
            state.model_matrix,
            (channel, channel),
            "state model matrix",
        )
        _require_array(state.offset, (channel,), "state offset")
        _, state_pivot = _require_spd_array(
            state.covariance,
            (channel, channel),
            "state covariance",
        )
        state_pivots.append(state_pivot)
        if (
            type(emission) is not H8Emission
            or type(emission.receiver_t) is not int
            or emission.receiver_t != receiver_t
            or type(emission.observation) is not int
            or not 0 <= emission.observation < 3
            or emission.observation
            != (problem.problem_seed + receiver_t) % 3
        ):
            raise ValueError("emission identity is outside the frozen H8 schema")
        _require_array(emission.weight, (block,), "emission weight")
        _require_array(emission.bias, (3,), "emission bias")
    recognition = problem.recognition
    if type(recognition) is not H8RecognitionSpecification:
        raise ValueError("recognition must use the exact H8 specification")
    _require_array(
        recognition.initial_mean,
        (block,),
        "recognition initial mean",
    )
    _, recognition_initial_pivot = _require_spd_array(
        recognition.initial_covariance,
        (block, block),
        "recognition initial covariance",
    )
    if (
        type(recognition.transitions) is not tuple
        or len(recognition.transitions) != layout.horizon
    ):
        raise ValueError("recognition must contain exactly T transitions")
    recognition_pivots: list[float] = []
    for receiver_t, transition in enumerate(recognition.transitions, start=1):
        _require_transition_identity(
            transition,
            receiver_t,
            "recognition transition",
        )
        _require_array(
            transition.matrix,
            (block, block),
            "recognition matrix",
        )
        _require_array(
            transition.offset,
            (block,),
            "recognition offset",
        )
        _, recognition_pivot = _require_spd_array(
            transition.covariance,
            (block, block),
            "recognition covariance",
        )
        recognition_pivots.append(recognition_pivot)
    if type(transition_norms) is not H8TransitionNorms:
        raise ValueError("problem transition norms must be exact typed evidence")
    transition_norms.__post_init__()
    if transition_norms.horizon != layout.horizon:
        raise ValueError("problem transition norms must match the layout horizon")
    local_spd_diagnostics = H8LocalSPDDiagnostics(
        schema_version="h8-local-spd-diagnostics-v1",
        horizon=layout.horizon,
        generative_initial_min_pivot=generative_initial_pivot,
        model_transition_min_pivots=tuple(model_pivots),
        state_transition_min_pivots=tuple(state_pivots),
        recognition_initial_min_pivot=recognition_initial_pivot,
        recognition_transition_min_pivots=tuple(recognition_pivots),
        global_min_pivot=min(
            generative_initial_pivot,
            *model_pivots,
            *state_pivots,
            recognition_initial_pivot,
            *recognition_pivots,
        ),
    )
    generative_payload = _generative_evidence_payload(problem)
    recognition_payload = _recognition_evidence_payload(problem)
    observation_payload = _observation_evidence_payload(problem)
    return H8ProductionProblemEvidence(
        generative_sha256=hashlib.sha256(
            _canonical_evidence_bytes(generative_payload)
        ).hexdigest(),
        recognition_sha256=hashlib.sha256(
            _canonical_evidence_bytes(recognition_payload)
        ).hexdigest(),
        local_spd_diagnostics=local_spd_diagnostics,
        transition_norms=transition_norms,
        observation_sha256=hashlib.sha256(
            _canonical_evidence_bytes(observation_payload)
        ).hexdigest(),
    )


def _validate_problem_serialized_identity(problem: H8Problem) -> None:
    """Validate the pre-existing aggregate byte identity without new SVDs."""

    layout = problem.layout
    serialized = _serialize_problem(
        layout=layout,
        problem_seed=problem.problem_seed,
        vocabulary_size=problem.vocabulary_size,
        alpha=problem.alpha,
        initial_mean=problem.initial_mean,
        initial_covariance=problem.initial_covariance,
        model_transitions=problem.model_transitions,
        state_transitions=problem.state_transitions,
        emissions=problem.emissions,
        recognition=problem.recognition,
        draw_schema_sha256=problem.draw_schema_sha256,
    )
    if type(problem.serialized_bytes) is not bytes or problem.serialized_bytes != serialized:
        raise ValueError("serialized H8 bytes do not match every semantic field")
    if (
        type(problem.input_sha256) is not str
        or problem.input_sha256 != hashlib.sha256(serialized).hexdigest()
    ):
        raise ValueError("H8 input hash does not match the serialized bytes")


def _array_evidence_payload(value: np.ndarray) -> dict[str, object]:
    """Return the exact v4 array leaf without copying or reserializing NumPy."""

    return {
        "shape": [int(item) for item in value.shape],
        "dtype": "<f8",
        "raw_sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    }


def _layout_evidence_payload(layout: BlockChainLayout) -> dict[str, int]:
    return {
        "horizon": layout.horizon,
        "d_z": layout.d_z,
        "d_m": layout.d_m,
    }


def _generative_evidence_payload(problem: H8Problem) -> dict[str, object]:
    """Build the frozen h8-generative-evidence-v1 JSON preimage."""

    return {
        "domain": "vfe4.h8.generative-evidence.v1",
        "schema_version": "h8-generative-evidence-v1",
        "layout": _layout_evidence_payload(problem.layout),
        "problem_seed": problem.problem_seed,
        "vocabulary_size": problem.vocabulary_size,
        "alpha": _array_evidence_payload(problem.alpha),
        "initial": {
            "mean": _array_evidence_payload(problem.initial_mean),
            "covariance": _array_evidence_payload(problem.initial_covariance),
        },
        "model_transitions": [
            {
                "receiver_t": transition.receiver_t,
                "parent_t": transition.parent_t,
                "source_support": list(transition.source_support),
                "matrix": _array_evidence_payload(transition.matrix),
                "offset": _array_evidence_payload(transition.offset),
                "covariance": _array_evidence_payload(transition.covariance),
            }
            for transition in problem.model_transitions
        ],
        "state_transitions": [
            {
                "receiver_t": transition.receiver_t,
                "parent_t": transition.parent_t,
                "source_support": list(transition.source_support),
                "state_matrix": _array_evidence_payload(
                    transition.state_matrix
                ),
                "model_matrix": _array_evidence_payload(
                    transition.model_matrix
                ),
                "offset": _array_evidence_payload(transition.offset),
                "covariance": _array_evidence_payload(transition.covariance),
            }
            for transition in problem.state_transitions
        ],
        "emissions": [
            {
                "receiver_t": emission.receiver_t,
                "weight": _array_evidence_payload(emission.weight),
                "bias": _array_evidence_payload(emission.bias),
                "observation": emission.observation,
            }
            for emission in problem.emissions
        ],
    }


def _recognition_evidence_payload(problem: H8Problem) -> dict[str, object]:
    """Build the frozen h8-recognition-evidence-v1 JSON preimage."""

    recognition = problem.recognition
    return {
        "domain": "vfe4.h8.recognition-evidence.v1",
        "schema_version": "h8-recognition-evidence-v1",
        "layout": _layout_evidence_payload(problem.layout),
        "problem_seed": problem.problem_seed,
        "initial": {
            "mean": _array_evidence_payload(recognition.initial_mean),
            "covariance": _array_evidence_payload(
                recognition.initial_covariance
            ),
        },
        "transitions": [
            {
                "receiver_t": transition.receiver_t,
                "parent_t": transition.parent_t,
                "source_support": list(transition.source_support),
                "matrix": _array_evidence_payload(transition.matrix),
                "offset": _array_evidence_payload(transition.offset),
                "covariance": _array_evidence_payload(transition.covariance),
            }
            for transition in recognition.transitions
        ],
    }


def _observation_evidence_payload(problem: H8Problem) -> dict[str, object]:
    """Build the exact ordered observation JSON preimage."""

    return {
        "domain": "vfe4.h8.observations.v1",
        "records": [
            [emission.receiver_t, emission.observation]
            for emission in problem.emissions
        ],
    }


def _canonical_evidence_bytes(payload: object) -> bytes:
    """Encode strict ASCII canonical JSON for one domain-separated preimage."""

    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _serialize_problem(
    *,
    layout: BlockChainLayout,
    problem_seed: int,
    vocabulary_size: int,
    alpha: np.ndarray,
    initial_mean: np.ndarray,
    initial_covariance: np.ndarray,
    model_transitions: tuple[H8ModelTransition, ...],
    state_transitions: tuple[H8StateTransition, ...],
    emissions: tuple[H8Emission, ...],
    recognition: H8RecognitionSpecification,
    draw_schema_sha256: str,
) -> bytes:
    chunks: list[bytes] = [
        b"schema:" + H8_GENERATOR_SCHEMA.encode("ascii"),
        b"descriptor:" + H8_DRAW_DESCRIPTOR.encode("ascii"),
        b"draw-schema-sha256:" + draw_schema_sha256.encode("ascii"),
    ]

    def append(value: object) -> None:
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value, dtype="<f8")
            chunks.append(b"array:" + str(array.shape).encode("ascii"))
            chunks.append(array.tobytes(order="C"))
        elif isinstance(value, (tuple, list)):
            chunks.append(b"tuple:" + str(len(value)).encode("ascii"))
            for item in value:
                append(item)
        elif isinstance(value, H8ModelTransition):
            chunks.append(b"model-transition")
            append(
                (
                    value.receiver_t,
                    value.parent_t,
                    value.source_support,
                    value.matrix,
                    value.offset,
                    value.covariance,
                )
            )
        elif isinstance(value, H8StateTransition):
            chunks.append(b"state-transition")
            append(
                (
                    value.receiver_t,
                    value.parent_t,
                    value.source_support,
                    value.state_matrix,
                    value.model_matrix,
                    value.offset,
                    value.covariance,
                )
            )
        elif isinstance(value, H8RecognitionSpecification):
            chunks.append(b"recognition")
            append(
                (
                    value.initial_mean,
                    value.initial_covariance,
                    value.transitions,
                )
            )
        elif isinstance(value, H8Emission):
            chunks.append(b"emission")
            append((value.receiver_t, value.weight, value.bias, value.observation))
        elif isinstance(value, BlockChainLayout):
            chunks.append(b"layout")
            append((value.horizon, value.d_z, value.d_m))
        elif type(value) is int:
            chunks.append(b"integer:" + str(value).encode("ascii"))
        else:
            raise TypeError("unsupported H8 serialization field")

    for value in (
        layout,
        problem_seed,
        vocabulary_size,
        alpha,
        initial_mean,
        initial_covariance,
        model_transitions,
        state_transitions,
        recognition,
        emissions,
    ):
        append(value)
    return b"".join(
        len(chunk).to_bytes(8, byteorder="big", signed=False) + chunk
        for chunk in chunks
    )


def _new_problem(**values: object) -> H8Problem:
    problem = object.__new__(H8Problem)
    for name, value in values.items():
        object.__setattr__(problem, name, value)
    return problem


def _require_parent_identity(
    receiver_t: object,
    parent_t: object,
    source_support: object,
    expected_receiver: int,
    name: str,
) -> None:
    expected_parent = expected_receiver - 1
    if (
        type(receiver_t) is not int
        or type(parent_t) is not int
        or type(source_support) is not tuple
        or receiver_t != expected_receiver
        or parent_t != expected_parent
        or source_support != (expected_parent,)
    ):
        raise ValueError(
            f"{name} must bind parent_t==receiver_t-1 and its singleton support"
        )


def _require_transition_identity(
    transition: object,
    receiver_t: int,
    name: str,
) -> None:
    if type(transition) is not H8ModelTransition:
        raise ValueError(f"{name} must use an exact H8 transition record")
    _require_parent_identity(
        transition.receiver_t,
        transition.parent_t,
        transition.source_support,
        receiver_t,
        name,
    )


def _require_array(
    value: object,
    shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    if (
        type(value) is not np.ndarray
        or value.dtype.str != "<f8"
        or tuple(value.shape) != shape
        or not value.flags.c_contiguous
        or value.flags.writeable
        or not bool(np.all(np.isfinite(value)))
    ):
        raise ValueError(
            f"{name} must be immutable C-contiguous little-endian float64"
        )
    return value


def _require_spd_array(
    value: object,
    shape: tuple[int, ...],
    name: str,
) -> tuple[np.ndarray, float]:
    checked = _require_array(value, shape, name)
    if not np.array_equal(checked, checked.T):
        raise ValueError(f"{name} must be symmetric")
    try:
        factor = np.linalg.cholesky(checked)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{name} must be strictly positive definite") from error
    if not bool(np.all(np.isfinite(factor))):
        raise ValueError(f"{name} Cholesky factor must be finite")
    minimum_pivot = float(np.min(np.diag(factor)))
    if not math.isfinite(minimum_pivot) or minimum_pivot <= 0.0:
        raise ValueError(f"{name} Cholesky pivot must be finite and positive")
    return checked, minimum_pivot


def _torch(value: object) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.float64, device="cpu")


__all__ = [
    "H8_DRAW_DESCRIPTOR",
    "H8_GENERATOR_SCHEMA",
    "H8_SAMPLE_SCHEMA",
    "H8Emission",
    "H8GenerativeModel",
    "H8ModelTransition",
    "H8Problem",
    "H8RecognitionSpecification",
    "H8StateTransition",
    "build_h8_problem",
    "build_h8_generative",
    "h8_sample_noise",
    "make_h8_problem",
    "validate_h8_problem",
]
