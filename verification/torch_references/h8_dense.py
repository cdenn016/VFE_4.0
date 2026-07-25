"""Bounded dense-PyTorch H8 adapter used only by correctness preflight.

This module owns its parser, dense assembler, and complete objective.  It does
not call block assembly, the sparse objective, or production quadrature.  The
only reused numerical primitives are the already-bounded dense precision and
information Gaussian classes.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import torch
from torch import Tensor

from vfe4.numerics.information import InformationGaussian
from vfe4.numerics.precision import DenseCholeskyPrecision


_MAX_HORIZON = 8
_MAX_CHANNEL = 4
_VOCABULARY = 3
_GENERATOR_SCHEMA = b"schema:h8-synthetic-chain-v1"
_DRAW_SCHEMA_SHA256 = (
    "7b657e72219f044147a7b414354d34c82bbd5a66d24f669285906d54534723c0"
)
_DRAW_DESCRIPTOR = (
    "numpy.Generator(numpy.PCG64(problem_seed))|method=standard_normal|"
    "dtype=float64|order=C|initial:mu0[b],Q0[b,b]|"
    "transition:t=1..T:{A_m[K,K],c_m[K],Q_m[K,K],A_z[K,K],B[K,K],c_z[K],Q_z[K,K]}|"
    "recognition_initial:mu_q0[b],Q_q0[b,b]|"
    "recognition_transition:t=1..T:{A_q[b,b],c_q[b],Q_q[b,b]}|"
    "emission:t=1..T:{w[b],beta[V]}|"
    "normal_map_variance=1/dim=>multiply_standard_normal_by_1/sqrt(dim)|"
    "serialize=after_all_problem_draws_before_sample_rng|bytes=little-endian-f8-C-contiguous"
)


@dataclass(frozen=True, slots=True)
class TorchOperandMetadata:
    shape: tuple[int, ...]
    scalar_count: int
    infinity_norm: float
    absolute_sum_bound: float
    local_operation_count: int
    solver_produced: bool
    quadrature_convergence: float = 0.0


@dataclass(frozen=True, slots=True)
class TorchH8ObjectiveTerm:
    factor_id: str
    value: float
    absolute_sum_bound: float


@dataclass(frozen=True, slots=True)
class TorchH8Objective:
    initial_joint: TorchH8ObjectiveTerm
    model_transitions: tuple[TorchH8ObjectiveTerm, ...]
    state_transitions: tuple[TorchH8ObjectiveTerm, ...]
    emissions_order21: tuple[TorchH8ObjectiveTerm, ...]
    emissions_order17: tuple[TorchH8ObjectiveTerm, ...]
    recognition_entropy: float
    log_normalizer: float
    model_source_kl: float
    state_source_kl: float
    source_entropy: float
    quadrature_absolute_difference: float
    complete_order21: float
    absolute_term_sum: float


@dataclass(frozen=True, slots=True)
class TorchDenseH8Result:
    input_sha256: str
    horizon: int
    d_z: int
    d_m: int
    precision: Tensor
    information: Tensor
    cholesky: Tensor
    factor_reconstruction: Tensor
    forward_substitution: Tensor
    backward_substitution: Tensor
    solve: Tensor
    logdet: float
    quadratic: float
    sample_noise: Tensor
    sample: Tensor
    covariance: Tensor
    selected_diagonal: Tensor
    selected_lower: Tensor
    sparse_trace: float
    entropy: float
    log_normalizer: float
    objective: TorchH8Objective
    operands: Mapping[str, TorchOperandMetadata]


@dataclass(frozen=True, slots=True)
class _Transition:
    receiver_t: int
    parent_t: int
    source_support: tuple[int, ...]
    matrix: Tensor
    offset: Tensor
    covariance: Tensor


@dataclass(frozen=True, slots=True)
class _StateTransition:
    receiver_t: int
    parent_t: int
    source_support: tuple[int, ...]
    state_matrix: Tensor
    model_matrix: Tensor
    offset: Tensor
    covariance: Tensor


@dataclass(frozen=True, slots=True)
class _Emission:
    receiver_t: int
    weight: Tensor
    bias: Tensor
    observation: int


@dataclass(frozen=True, slots=True)
class _ParsedProblem:
    horizon: int
    d_z: int
    d_m: int
    problem_seed: int
    vocabulary_size: int
    alpha: Tensor
    initial_mean: Tensor
    initial_covariance: Tensor
    model_transitions: tuple[_Transition, ...]
    state_transitions: tuple[_StateTransition, ...]
    recognition_initial_mean: Tensor
    recognition_initial_covariance: Tensor
    recognition_transitions: tuple[_Transition, ...]
    emissions: tuple[_Emission, ...]
    input_sha256: str


class _ChunkReader:
    __slots__ = ("_buffer", "_offset")

    def __init__(self, serialized: bytes) -> None:
        if type(serialized) is not bytes or not serialized:
            raise ValueError("serialized H8 input must be nonempty immutable bytes")
        self._buffer = memoryview(serialized)
        self._offset = 0

    def chunk(self) -> memoryview:
        if self._offset + 8 > len(self._buffer):
            raise ValueError("truncated H8 chunk length")
        length = int.from_bytes(
            self._buffer[self._offset : self._offset + 8],
            byteorder="big",
            signed=False,
        )
        self._offset += 8
        end = self._offset + length
        if length <= 0 or end > len(self._buffer):
            raise ValueError("invalid or truncated H8 chunk")
        value = self._buffer[self._offset : end]
        self._offset = end
        return value

    def exact(self, expected: bytes) -> None:
        if self.chunk() != expected:
            raise ValueError(f"serialized H8 marker must be {expected!r}")

    def integer(self) -> int:
        raw = self.chunk().tobytes()
        if not raw.startswith(b"integer:"):
            raise ValueError("serialized H8 integer marker is missing")
        digits = raw.removeprefix(b"integer:")
        if (
            not digits
            or not digits.isdigit()
            or (len(digits) > 1 and digits.startswith(b"0"))
        ):
            raise ValueError("serialized H8 integer is not canonically encoded")
        try:
            value = int(digits.decode("ascii"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise ValueError("serialized H8 integer is invalid") from exc
        if raw != f"integer:{value}".encode("ascii"):
            raise ValueError("serialized H8 integer is not canonically encoded")
        return value

    def tuple_start(self, expected_length: int) -> None:
        self.exact(f"tuple:{expected_length}".encode("ascii"))

    def array(self, expected_shape: tuple[int, ...], name: str) -> Tensor:
        header = self.chunk().tobytes()
        expected_header = b"array:" + str(expected_shape).encode("ascii")
        if header != expected_header:
            raise ValueError(f"{name} serialized shape header is not canonical")
        count = math.prod(expected_shape)
        payload = self.chunk()
        if len(payload) != count * 8:
            raise ValueError(f"{name} serialized byte count is inconsistent")
        values = struct.unpack(f"<{count}d", payload)
        result = torch.tensor(values, dtype=torch.float64, device="cpu").reshape(
            expected_shape
        )
        if not bool(torch.isfinite(result).all()):
            raise ValueError(f"{name} must be finite")
        return result

    def finish(self) -> None:
        if self._offset != len(self._buffer):
            raise ValueError("serialized H8 input contains trailing chunks")


@torch.no_grad()
def evaluate_h8_torch_dense(
    problem: object,
    sample_noise: Tensor,
) -> TorchDenseH8Result:
    parsed = _parse_problem(problem)
    horizon = parsed.horizon
    channel = parsed.d_z
    population = horizon + 1
    block = parsed.d_z + parsed.d_m
    dimension = population * block
    noise_blocks = _require_noise(sample_noise, population, block)
    noise = noise_blocks.reshape(dimension)

    precision, information = _assemble_recognition(parsed)
    dense_factor = DenseCholeskyPrecision(precision)
    gaussian = InformationGaussian.from_information(
        information,
        precision,
        factor_factory=DenseCholeskyPrecision,
    )
    cholesky = torch.linalg.cholesky(precision)
    reconstruction = cholesky @ cholesky.T
    forward = torch.linalg.solve_triangular(
        cholesky,
        information.unsqueeze(1),
        upper=False,
    ).squeeze(1)
    backward = torch.linalg.solve_triangular(
        cholesky.T,
        forward.unsqueeze(1),
        upper=True,
    ).squeeze(1)
    solved = dense_factor.solve(information)
    logdet = float(dense_factor.logdet().item())
    quadratic = float(dense_factor.quadratic(noise).item())
    sample = solved + dense_factor.sample(noise)
    covariance = torch.cholesky_solve(
        torch.eye(dimension, dtype=torch.float64, device="cpu"),
        cholesky,
    )
    selected_diagonal = torch.stack(
        tuple(
            covariance[
                population_index * block : (population_index + 1) * block,
                population_index * block : (population_index + 1) * block,
            ]
            for population_index in range(population)
        ),
        dim=0,
    )
    selected_lower = torch.stack(
        tuple(
            covariance[
                population_index * block : (population_index + 1) * block,
                (population_index - 1) * block : population_index * block,
            ]
            for population_index in range(1, population)
        ),
        dim=0,
    )
    generative_precision = _assemble_generative_precision(parsed)
    sparse_trace = float(torch.trace(generative_precision @ covariance).item())
    entropy = float(gaussian.entropy().item())
    log_normalizer = float(gaussian.log_normalizer().item())
    objective = _evaluate_objective(
        parsed,
        solved,
        covariance,
        entropy,
        log_normalizer,
    )
    endpoints: dict[str, tuple[object, int, bool, float]] = {
        "factor_reconstruction": (reconstruction, 2 * dimension, False, 0.0),
        "forward_substitution": (forward, dimension * dimension, True, 0.0),
        "backward_substitution": (backward, dimension * dimension, True, 0.0),
        "solve": (solved, 2 * dimension * dimension, True, 0.0),
        "logdet": (logdet, dimension + 1, False, 0.0),
        "quadratic": (quadratic, 2 * dimension * dimension, False, 0.0),
        "sample": (sample, dimension * dimension, True, 0.0),
        "selected_diagonal": (
            selected_diagonal,
            2 * dimension * dimension,
            True,
            0.0,
        ),
        "selected_lower": (
            selected_lower,
            2 * dimension * dimension,
            True,
            0.0,
        ),
        "sparse_trace": (
            sparse_trace,
            2 * dimension * dimension,
            False,
            0.0,
        ),
        "entropy": (entropy, dimension + 4, False, 0.0),
        "log_normalizer": (
            log_normalizer,
            2 * dimension + 5,
            True,
            0.0,
        ),
    }
    for term in _objective_terms(objective):
        quadrature = _objective_quadrature(objective, term.factor_id)
        endpoints[f"objective:{term.factor_id}"] = (
            term.value,
            max(1, dimension * dimension),
            False,
            quadrature,
        )
    operands = MappingProxyType(
        {
            name: _metadata(value, operations, solver, quadrature)
            for name, (value, operations, solver, quadrature) in endpoints.items()
        }
    )
    return TorchDenseH8Result(
        input_sha256=parsed.input_sha256,
        horizon=horizon,
        d_z=channel,
        d_m=channel,
        precision=_owned(precision),
        information=_owned(information),
        cholesky=_owned(cholesky),
        factor_reconstruction=_owned(reconstruction),
        forward_substitution=_owned(forward),
        backward_substitution=_owned(backward),
        solve=_owned(solved),
        logdet=logdet,
        quadratic=quadratic,
        sample_noise=_owned(noise_blocks),
        sample=_owned(sample.reshape(population, block)),
        covariance=_owned(covariance),
        selected_diagonal=_owned(selected_diagonal),
        selected_lower=_owned(selected_lower),
        sparse_trace=sparse_trace,
        entropy=entropy,
        log_normalizer=log_normalizer,
        objective=objective,
        operands=operands,
    )


def _parse_problem(problem: object) -> _ParsedProblem:
    serialized = getattr(problem, "serialized_bytes", None)
    claimed_sha256 = getattr(problem, "input_sha256", None)
    if type(serialized) is not bytes or type(claimed_sha256) is not str:
        raise ValueError("problem must expose immutable serialized_bytes and input_sha256")
    actual_sha256 = hashlib.sha256(serialized).hexdigest()
    reader = _ChunkReader(serialized)
    reader.exact(_GENERATOR_SCHEMA)
    reader.exact(b"descriptor:" + _DRAW_DESCRIPTOR.encode("ascii"))
    reader.exact(b"draw-schema-sha256:" + _DRAW_SCHEMA_SHA256.encode("ascii"))
    reader.exact(b"layout")
    reader.tuple_start(3)
    horizon = reader.integer()
    d_z = reader.integer()
    d_m = reader.integer()
    _require_bounded_layout(horizon, d_z, d_m)

    population = horizon + 1
    block = d_z + d_m
    problem_seed = reader.integer()
    vocabulary_size = reader.integer()
    if problem_seed <= 0 or vocabulary_size != _VOCABULARY:
        raise ValueError("serialized H8 seed or vocabulary is invalid")
    alpha = reader.array((_VOCABULARY,), "alpha")
    expected_alpha = torch.tensor((-0.5, 0.25, 0.75), dtype=torch.float64)
    if not bool(torch.equal(alpha, expected_alpha)):
        raise ValueError("alpha does not match the frozen H8 slopes")
    initial_mean = reader.array((block,), "initial_mean")
    initial_covariance = _spd(
        reader.array((block, block), "initial_covariance"),
        "initial_covariance",
    )
    reader.tuple_start(horizon)
    model = tuple(
        _read_transition(reader, receiver_t, d_z, "model")
        for receiver_t in range(1, population)
    )
    reader.tuple_start(horizon)
    states = tuple(
        _read_state_transition(reader, receiver_t, d_z)
        for receiver_t in range(1, population)
    )
    reader.exact(b"recognition")
    reader.tuple_start(3)
    recognition_initial_mean = reader.array(
        (block,),
        "recognition_initial_mean",
    )
    recognition_initial_covariance = _spd(
        reader.array(
            (block, block),
            "recognition_initial_covariance",
        ),
        "recognition_initial_covariance",
    )
    reader.tuple_start(horizon)
    recognition = tuple(
        _read_transition(reader, receiver_t, block, "recognition")
        for receiver_t in range(1, population)
    )
    reader.tuple_start(horizon)
    emissions = tuple(
        _read_emission(reader, receiver_t, block, problem_seed)
        for receiver_t in range(1, population)
    )
    reader.finish()
    parsed = _ParsedProblem(
        horizon=horizon,
        d_z=d_z,
        d_m=d_m,
        problem_seed=problem_seed,
        vocabulary_size=vocabulary_size,
        alpha=alpha,
        initial_mean=initial_mean,
        initial_covariance=initial_covariance,
        model_transitions=model,
        state_transitions=states,
        recognition_initial_mean=recognition_initial_mean,
        recognition_initial_covariance=recognition_initial_covariance,
        recognition_transitions=recognition,
        emissions=emissions,
        input_sha256=actual_sha256,
    )
    canonical = _serialize_canonical_problem(parsed)
    if canonical != serialized:
        raise ValueError(
            "serialized H8 input differs from its independently reconstructed "
            "canonical preimage"
        )
    _validate_carrier_semantics(problem, parsed)
    if actual_sha256 != claimed_sha256:
        raise ValueError("canonical H8 SHA-256 does not match its carrier")
    return parsed


def _serialize_canonical_problem(problem: _ParsedProblem) -> bytes:
    """Independently reconstruct every byte in the frozen H8 preimage."""

    chunks: list[bytes] = [
        _GENERATOR_SCHEMA,
        b"descriptor:" + _DRAW_DESCRIPTOR.encode("ascii"),
        b"draw-schema-sha256:" + _DRAW_SCHEMA_SHA256.encode("ascii"),
    ]

    def integer(value: int) -> None:
        chunks.append(f"integer:{value}".encode("ascii"))

    def tuple_start(length: int) -> None:
        chunks.append(f"tuple:{length}".encode("ascii"))

    def array(value: Tensor) -> None:
        chunks.append(b"array:" + str(tuple(value.shape)).encode("ascii"))
        values = tuple(float(item) for item in value.reshape(-1).tolist())
        chunks.append(struct.pack(f"<{len(values)}d", *values))

    chunks.append(b"layout")
    tuple_start(3)
    integer(problem.horizon)
    integer(problem.d_z)
    integer(problem.d_m)
    integer(problem.problem_seed)
    integer(problem.vocabulary_size)
    array(problem.alpha)
    array(problem.initial_mean)
    array(problem.initial_covariance)
    tuple_start(problem.horizon)
    for transition in problem.model_transitions:
        chunks.append(b"model-transition")
        tuple_start(6)
        integer(transition.receiver_t)
        integer(transition.parent_t)
        tuple_start(1)
        integer(transition.source_support[0])
        array(transition.matrix)
        array(transition.offset)
        array(transition.covariance)
    tuple_start(problem.horizon)
    for transition in problem.state_transitions:
        chunks.append(b"state-transition")
        tuple_start(7)
        integer(transition.receiver_t)
        integer(transition.parent_t)
        tuple_start(1)
        integer(transition.source_support[0])
        array(transition.state_matrix)
        array(transition.model_matrix)
        array(transition.offset)
        array(transition.covariance)
    chunks.append(b"recognition")
    tuple_start(3)
    array(problem.recognition_initial_mean)
    array(problem.recognition_initial_covariance)
    tuple_start(problem.horizon)
    for transition in problem.recognition_transitions:
        chunks.append(b"model-transition")
        tuple_start(6)
        integer(transition.receiver_t)
        integer(transition.parent_t)
        tuple_start(1)
        integer(transition.source_support[0])
        array(transition.matrix)
        array(transition.offset)
        array(transition.covariance)
    tuple_start(problem.horizon)
    for emission in problem.emissions:
        chunks.append(b"emission")
        tuple_start(4)
        integer(emission.receiver_t)
        array(emission.weight)
        array(emission.bias)
        integer(emission.observation)
    return b"".join(
        len(chunk).to_bytes(8, byteorder="big", signed=False) + chunk
        for chunk in chunks
    )


def _validate_carrier_semantics(
    carrier: object,
    parsed: _ParsedProblem,
) -> None:
    """Bind every parsed field to the immutable factory carrier independently."""

    layout = getattr(carrier, "layout", None)
    if (
        getattr(layout, "horizon", None) != parsed.horizon
        or getattr(layout, "d_z", None) != parsed.d_z
        or getattr(layout, "d_m", None) != parsed.d_m
        or getattr(carrier, "problem_seed", None) != parsed.problem_seed
        or getattr(carrier, "vocabulary_size", None) != parsed.vocabulary_size
        or getattr(carrier, "draw_schema_sha256", None)
        != _DRAW_SCHEMA_SHA256
    ):
        raise ValueError("serialized layout/identity differs from its H8 carrier")
    _require_same_carrier_array(getattr(carrier, "alpha", None), parsed.alpha, "alpha")
    _require_same_carrier_array(
        getattr(carrier, "initial_mean", None),
        parsed.initial_mean,
        "initial_mean",
    )
    _require_same_carrier_array(
        getattr(carrier, "initial_covariance", None),
        parsed.initial_covariance,
        "initial_covariance",
    )
    _require_same_transition_series(
        getattr(carrier, "model_transitions", None),
        parsed.model_transitions,
        "model_transitions",
    )
    carrier_states = getattr(carrier, "state_transitions", None)
    if type(carrier_states) is not tuple or len(carrier_states) != len(
        parsed.state_transitions
    ):
        raise ValueError("state transition carrier inventory differs from bytes")
    for carrier_state, parsed_state in zip(
        carrier_states,
        parsed.state_transitions,
        strict=True,
    ):
        if (
            getattr(carrier_state, "receiver_t", None)
            != parsed_state.receiver_t
            or getattr(carrier_state, "parent_t", None) != parsed_state.parent_t
            or getattr(carrier_state, "source_support", None)
            != parsed_state.source_support
        ):
            raise ValueError("state transition identity differs from bytes")
        for field_name in (
            "state_matrix",
            "model_matrix",
            "offset",
            "covariance",
        ):
            _require_same_carrier_array(
                getattr(carrier_state, field_name, None),
                getattr(parsed_state, field_name),
                f"state_transitions.{field_name}",
            )
    carrier_recognition = getattr(carrier, "recognition", None)
    _require_same_carrier_array(
        getattr(carrier_recognition, "initial_mean", None),
        parsed.recognition_initial_mean,
        "recognition.initial_mean",
    )
    _require_same_carrier_array(
        getattr(carrier_recognition, "initial_covariance", None),
        parsed.recognition_initial_covariance,
        "recognition.initial_covariance",
    )
    _require_same_transition_series(
        getattr(carrier_recognition, "transitions", None),
        parsed.recognition_transitions,
        "recognition.transitions",
    )
    carrier_emissions = getattr(carrier, "emissions", None)
    if type(carrier_emissions) is not tuple or len(carrier_emissions) != len(
        parsed.emissions
    ):
        raise ValueError("emission carrier inventory differs from bytes")
    for carrier_emission, parsed_emission in zip(
        carrier_emissions,
        parsed.emissions,
        strict=True,
    ):
        if (
            getattr(carrier_emission, "receiver_t", None)
            != parsed_emission.receiver_t
            or getattr(carrier_emission, "observation", None)
            != parsed_emission.observation
        ):
            raise ValueError("emission identity differs from bytes")
        _require_same_carrier_array(
            getattr(carrier_emission, "weight", None),
            parsed_emission.weight,
            "emission.weight",
        )
        _require_same_carrier_array(
            getattr(carrier_emission, "bias", None),
            parsed_emission.bias,
            "emission.bias",
        )


def _require_same_transition_series(
    carrier_series: object,
    parsed_series: tuple[_Transition, ...],
    name: str,
) -> None:
    if type(carrier_series) is not tuple or len(carrier_series) != len(parsed_series):
        raise ValueError(f"{name} carrier inventory differs from bytes")
    for carrier_transition, parsed_transition in zip(
        carrier_series,
        parsed_series,
        strict=True,
    ):
        if (
            getattr(carrier_transition, "receiver_t", None)
            != parsed_transition.receiver_t
            or getattr(carrier_transition, "parent_t", None)
            != parsed_transition.parent_t
            or getattr(carrier_transition, "source_support", None)
            != parsed_transition.source_support
        ):
            raise ValueError(f"{name} identity differs from bytes")
        for field_name in ("matrix", "offset", "covariance"):
            _require_same_carrier_array(
                getattr(carrier_transition, field_name, None),
                getattr(parsed_transition, field_name),
                f"{name}.{field_name}",
            )


def _require_same_carrier_array(
    carrier_value: object,
    parsed_value: Tensor,
    name: str,
) -> None:
    dtype = getattr(carrier_value, "dtype", None)
    flags = getattr(carrier_value, "flags", None)
    tobytes = getattr(carrier_value, "tobytes", None)
    try:
        carrier_bytes = tobytes(order="C") if callable(tobytes) else None
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} carrier bytes are unavailable") from exc
    if (
        getattr(dtype, "str", None) != "<f8"
        or tuple(getattr(carrier_value, "shape", ())) != tuple(parsed_value.shape)
        or not bool(getattr(flags, "c_contiguous", False))
        or bool(getattr(flags, "writeable", True))
        or carrier_bytes != _tensor_little_endian_bytes(parsed_value)
    ):
        raise ValueError(f"{name} carrier bytes differ from serialized preimage")


def _tensor_little_endian_bytes(value: Tensor) -> bytes:
    values = tuple(float(item) for item in value.reshape(-1).tolist())
    return struct.pack(f"<{len(values)}d", *values)


def _read_transition(
    reader: _ChunkReader,
    receiver_t: int,
    width: int,
    name: str,
) -> _Transition:
    reader.exact(b"model-transition")
    reader.tuple_start(6)
    receiver = reader.integer()
    parent = reader.integer()
    reader.tuple_start(1)
    support = (reader.integer(),)
    _require_parent(receiver, parent, support, receiver_t, name)
    return _Transition(
        receiver,
        parent,
        support,
        reader.array((width, width), f"{name}_matrix"),
        reader.array((width,), f"{name}_offset"),
        _spd(
            reader.array((width, width), f"{name}_covariance"),
            f"{name}_covariance",
        ),
    )


def _read_state_transition(
    reader: _ChunkReader,
    receiver_t: int,
    width: int,
) -> _StateTransition:
    reader.exact(b"state-transition")
    reader.tuple_start(7)
    receiver = reader.integer()
    parent = reader.integer()
    reader.tuple_start(1)
    support = (reader.integer(),)
    _require_parent(receiver, parent, support, receiver_t, "state")
    return _StateTransition(
        receiver,
        parent,
        support,
        reader.array((width, width), "state_matrix"),
        reader.array((width, width), "state_model_matrix"),
        reader.array((width,), "state_offset"),
        _spd(
            reader.array((width, width), "state_covariance"),
            "state_covariance",
        ),
    )


def _read_emission(
    reader: _ChunkReader,
    receiver_t: int,
    block: int,
    problem_seed: int,
) -> _Emission:
    reader.exact(b"emission")
    reader.tuple_start(4)
    receiver = reader.integer()
    weight = reader.array((block,), "emission_weight")
    bias = reader.array((_VOCABULARY,), "emission_bias")
    observation = reader.integer()
    if receiver != receiver_t or observation != (problem_seed + receiver_t) % 3:
        raise ValueError("emission identity is outside the frozen H8 schema")
    return _Emission(receiver, weight, bias, observation)


def _require_bounded_layout(horizon: object, d_z: object, d_m: object) -> None:
    if type(horizon) is not int or not 1 <= horizon <= _MAX_HORIZON:
        raise ValueError("dense H8 references require 1 <= T <= 8")
    if type(d_z) is not int or type(d_m) is not int:
        raise ValueError("dense H8 channel dimensions must be integers")
    if not 1 <= d_z <= _MAX_CHANNEL or not 1 <= d_m <= _MAX_CHANNEL:
        raise ValueError("dense H8 references require 1 <= K <= 4")
    if d_z != d_m:
        raise ValueError("dense H8 references require equal d_z and d_m")


def _require_parent(
    receiver: int,
    parent: int,
    support: tuple[int, ...],
    expected_receiver: int,
    name: str,
) -> None:
    if (
        receiver != expected_receiver
        or parent != expected_receiver - 1
        or support != (expected_receiver - 1,)
    ):
        raise ValueError(f"{name} transition must use its singleton previous slice")


def _spd(value: Tensor, name: str) -> Tensor:
    if not bool(torch.equal(value, value.T)):
        raise ValueError(f"{name} must be exactly symmetric")
    factor, info = torch.linalg.cholesky_ex(value, check_errors=False)
    if int(info.item()) != 0 or not bool(torch.isfinite(factor).all()):
        raise ValueError(f"{name} must be positive definite")
    return value


def _require_noise(value: object, population: int, block: int) -> Tensor:
    if (
        type(value) is not Tensor
        or value.dtype is not torch.float64
        or value.device.type != "cpu"
        or tuple(value.shape) != (population, block)
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError("sample noise must be finite CPU float64 [N,b]")
    return value


def _assemble_recognition(problem: _ParsedProblem) -> tuple[Tensor, Tensor]:
    population = problem.horizon + 1
    block = problem.d_z + problem.d_m
    dimension = population * block
    precision = torch.zeros((dimension, dimension), dtype=torch.float64)
    information = torch.zeros((dimension,), dtype=torch.float64)
    _scatter_initial(
        precision,
        information,
        tuple(range(block)),
        problem.recognition_initial_mean,
        problem.recognition_initial_covariance,
    )
    for transition in problem.recognition_transitions:
        target = tuple(
            range(transition.receiver_t * block, (transition.receiver_t + 1) * block)
        )
        parent = tuple(
            range(transition.parent_t * block, (transition.parent_t + 1) * block)
        )
        _scatter_conditional(
            precision,
            information,
            target,
            parent,
            transition.matrix,
            transition.offset,
            transition.covariance,
        )
    return 0.5 * (precision + precision.T), information


def _assemble_generative_precision(problem: _ParsedProblem) -> Tensor:
    population = problem.horizon + 1
    channel = problem.d_z
    block = problem.d_z + problem.d_m
    dimension = population * block
    precision = torch.zeros((dimension, dimension), dtype=torch.float64)
    information = torch.zeros((dimension,), dtype=torch.float64)
    _scatter_initial(
        precision,
        information,
        tuple(range(block)),
        problem.initial_mean,
        problem.initial_covariance,
    )
    for model, state in zip(
        problem.model_transitions,
        problem.state_transitions,
        strict=True,
    ):
        parent_base = model.parent_t * block
        target_base = model.receiver_t * block
        _scatter_conditional(
            precision,
            information,
            tuple(range(target_base + channel, target_base + block)),
            tuple(range(parent_base + channel, parent_base + block)),
            model.matrix,
            model.offset,
            model.covariance,
        )
        _scatter_conditional(
            precision,
            information,
            tuple(range(target_base, target_base + channel)),
            tuple(range(parent_base, parent_base + channel))
            + tuple(range(target_base + channel, target_base + block)),
            torch.cat((state.state_matrix, state.model_matrix), dim=1),
            state.offset,
            state.covariance,
        )
    return 0.5 * (precision + precision.T)


def _scatter_initial(
    precision: Tensor,
    information: Tensor,
    target: tuple[int, ...],
    mean: Tensor,
    covariance: Tensor,
) -> None:
    local_precision = _precision(covariance)
    index = torch.tensor(target, dtype=torch.int64)
    precision[index[:, None], index[None, :]] += local_precision
    information[index] += local_precision @ mean


def _scatter_conditional(
    precision: Tensor,
    information: Tensor,
    target: tuple[int, ...],
    parent: tuple[int, ...],
    matrix: Tensor,
    offset: Tensor,
    covariance: Tensor,
) -> None:
    local_precision = _precision(covariance)
    target_index = torch.tensor(target, dtype=torch.int64)
    parent_index = torch.tensor(parent, dtype=torch.int64)
    precision[target_index[:, None], target_index[None, :]] += local_precision
    precision[parent_index[:, None], parent_index[None, :]] += (
        matrix.T @ local_precision @ matrix
    )
    cross = -local_precision @ matrix
    precision[target_index[:, None], parent_index[None, :]] += cross
    precision[parent_index[:, None], target_index[None, :]] += cross.T
    information[target_index] += local_precision @ offset
    information[parent_index] -= matrix.T @ local_precision @ offset


def _precision(covariance: Tensor) -> Tensor:
    factor = torch.linalg.cholesky(covariance)
    result = torch.cholesky_solve(
        torch.eye(covariance.shape[0], dtype=torch.float64),
        factor,
    )
    return 0.5 * (result + result.T)


def _evaluate_objective(
    problem: _ParsedProblem,
    mean: Tensor,
    covariance: Tensor,
    entropy: float,
    log_normalizer: float,
) -> TorchH8Objective:
    channel = problem.d_z
    block = problem.d_z + problem.d_m
    initial_indices = tuple(range(block))
    initial = _term(
        "initial_joint",
        _normal_factor_expectation(
            mean[list(initial_indices)],
            covariance[list(initial_indices)][:, list(initial_indices)],
            problem.initial_mean,
            problem.initial_covariance,
        ),
    )
    model_terms: list[TorchH8ObjectiveTerm] = []
    state_terms: list[TorchH8ObjectiveTerm] = []
    emission21: list[TorchH8ObjectiveTerm] = []
    emission17: list[TorchH8ObjectiveTerm] = []
    for model, state, emission in zip(
        problem.model_transitions,
        problem.state_transitions,
        problem.emissions,
        strict=True,
    ):
        parent_base = model.parent_t * block
        target_base = model.receiver_t * block
        model_parent = tuple(range(parent_base + channel, parent_base + block))
        model_target = tuple(range(target_base + channel, target_base + block))
        model_terms.append(
            _term(
                f"model_transition:{model.receiver_t:04d}",
                _conditional_factor_expectation(
                    mean,
                    covariance,
                    model_target,
                    model_parent,
                    model.matrix,
                    model.offset,
                    model.covariance,
                ),
            )
        )
        state_target = tuple(range(target_base, target_base + channel))
        state_parent = tuple(range(parent_base, parent_base + channel)) + tuple(
            range(target_base + channel, target_base + block)
        )
        state_terms.append(
            _term(
                f"state_transition:{state.receiver_t:04d}",
                _conditional_factor_expectation(
                    mean,
                    covariance,
                    state_target,
                    state_parent,
                    torch.cat((state.state_matrix, state.model_matrix), dim=1),
                    state.offset,
                    state.covariance,
                ),
            )
        )
        local = tuple(range(target_base, target_base + block))
        local_mean = mean[list(local)]
        local_covariance = covariance[list(local)][:, list(local)]
        scalar_mean = float(torch.dot(emission.weight, local_mean).item())
        scalar_variance = float(
            (emission.weight @ local_covariance @ emission.weight).item()
        )
        value21 = _emission_expectation(
            scalar_mean,
            scalar_variance,
            problem.alpha,
            emission.bias,
            emission.observation,
            order=21,
        )
        value17 = _emission_expectation(
            scalar_mean,
            scalar_variance,
            problem.alpha,
            emission.bias,
            emission.observation,
            order=17,
        )
        emission21.append(
            _term(f"emission_order21:{emission.receiver_t:04d}", value21)
        )
        emission17.append(
            _term(f"emission_order17:{emission.receiver_t:04d}", value17)
        )
    complete_values = (
        initial.value,
        *(term.value for term in model_terms),
        *(term.value for term in state_terms),
        *(term.value for term in emission21),
        entropy,
    )
    return TorchH8Objective(
        initial_joint=initial,
        model_transitions=tuple(model_terms),
        state_transitions=tuple(state_terms),
        emissions_order21=tuple(emission21),
        emissions_order17=tuple(emission17),
        recognition_entropy=entropy,
        log_normalizer=log_normalizer,
        model_source_kl=0.0,
        state_source_kl=0.0,
        source_entropy=0.0,
        quadrature_absolute_difference=math.fsum(
            abs(left.value - right.value)
            for left, right in zip(emission21, emission17, strict=True)
        ),
        complete_order21=math.fsum(complete_values),
        absolute_term_sum=math.fsum(abs(value) for value in complete_values),
    )


def _conditional_factor_expectation(
    mean: Tensor,
    covariance: Tensor,
    target: tuple[int, ...],
    parent: tuple[int, ...],
    matrix: Tensor,
    offset: Tensor,
    factor_covariance: Tensor,
) -> float:
    target_mean = mean[list(target)]
    parent_mean = mean[list(parent)]
    target_covariance = covariance[list(target)][:, list(target)]
    parent_covariance = covariance[list(parent)][:, list(parent)]
    target_parent = covariance[list(target)][:, list(parent)]
    residual_mean = target_mean - matrix @ parent_mean - offset
    residual_covariance = (
        target_covariance
        - target_parent @ matrix.T
        - matrix @ target_parent.T
        + matrix @ parent_covariance @ matrix.T
    )
    return _normal_factor_expectation(
        residual_mean,
        residual_covariance,
        torch.zeros_like(residual_mean),
        factor_covariance,
    )


def _normal_factor_expectation(
    mean: Tensor,
    covariance: Tensor,
    location: Tensor,
    factor_covariance: Tensor,
) -> float:
    residual = mean - location
    precision = _precision(factor_covariance)
    factor = torch.linalg.cholesky(factor_covariance)
    logdet = 2.0 * torch.log(torch.diagonal(factor)).sum()
    quadratic = (
        torch.trace(precision @ covariance)
        + residual @ precision @ residual
    )
    return float(
        (
            -0.5
            * (
                mean.numel() * math.log(2.0 * math.pi)
                + logdet
                + quadratic
            )
        ).item()
    )


def _emission_expectation(
    mean: float,
    variance: float,
    alpha: Tensor,
    bias: Tensor,
    observation: int,
    *,
    order: int,
) -> float:
    if order not in (17, 21) or variance < 0.0 or not math.isfinite(variance):
        raise ValueError("emission quadrature input is outside the frozen domain")
    nodes, weights = _probabilists_gauss_hermite(order)
    values: list[float] = []
    for node, weight in zip(nodes, weights, strict=True):
        scalar = mean + math.sqrt(variance) * float(node.item())
        logits = alpha * scalar + bias
        log_probability = torch.log_softmax(logits, dim=0)[observation]
        values.append(float(weight.item()) * float(log_probability.item()))
    return math.fsum(values)


def _probabilists_gauss_hermite(order: int) -> tuple[Tensor, Tensor]:
    jacobi = torch.zeros((order, order), dtype=torch.float64)
    off_diagonal = torch.sqrt(torch.arange(1, order, dtype=torch.float64))
    positions = torch.arange(order - 1)
    jacobi[positions, positions + 1] = off_diagonal
    jacobi[positions + 1, positions] = off_diagonal
    nodes, eigenvectors = torch.linalg.eigh(jacobi)
    return nodes, eigenvectors[0, :] ** 2


def _term(factor_id: str, value: float) -> TorchH8ObjectiveTerm:
    return TorchH8ObjectiveTerm(factor_id, value, abs(value))


def _objective_terms(
    objective: TorchH8Objective,
) -> tuple[TorchH8ObjectiveTerm, ...]:
    return (
        objective.initial_joint,
        *objective.model_transitions,
        *objective.state_transitions,
        *objective.emissions_order21,
        *objective.emissions_order17,
        _term("recognition_entropy", objective.recognition_entropy),
        _term("log_normalizer", objective.log_normalizer),
        _term("model_source_kl", objective.model_source_kl),
        _term("state_source_kl", objective.state_source_kl),
        _term("source_entropy", objective.source_entropy),
        _term("complete_order21", objective.complete_order21),
    )


def _objective_quadrature(
    objective: TorchH8Objective,
    factor_id: str,
) -> float:
    if factor_id == "complete_order21":
        return objective.quadrature_absolute_difference
    if factor_id.startswith(("emission_order21:", "emission_order17:")):
        receiver = int(factor_id.rsplit(":", maxsplit=1)[1])
        return abs(
            objective.emissions_order21[receiver - 1].value
            - objective.emissions_order17[receiver - 1].value
        )
    return 0.0


def _metadata(
    value: object,
    operations: int,
    solver: bool,
    quadrature: float,
) -> TorchOperandMetadata:
    if type(value) is Tensor:
        shape = tuple(int(item) for item in value.shape)
        flattened = value.reshape(-1)
        infinity_norm = float(torch.max(torch.abs(flattened)).item())
        absolute_sum = float(torch.sum(torch.abs(flattened)).item())
    else:
        scalar = float(value)
        shape = (1,)
        infinity_norm = abs(scalar)
        absolute_sum = abs(scalar)
    return TorchOperandMetadata(
        shape=shape,
        scalar_count=math.prod(shape),
        infinity_norm=infinity_norm,
        absolute_sum_bound=absolute_sum,
        local_operation_count=operations,
        solver_produced=solver,
        quadrature_convergence=quadrature,
    )


def _owned(value: Tensor) -> Tensor:
    return value.detach().clone().contiguous()


__all__ = [
    "TorchDenseH8Result",
    "TorchH8Objective",
    "TorchH8ObjectiveTerm",
    "TorchOperandMetadata",
    "evaluate_h8_torch_dense",
]
